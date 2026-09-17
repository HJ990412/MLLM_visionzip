"""Full-shard MT-GQA Turn-1-piggyback system evaluation.

The unit of durability is an *image*, not a dialogue.  Every dialogue assigned
to one image is evaluated while one temporary ``visionzip_image_only`` store
is live, a content-hashed immutable image artifact is published, and only then
that experiment-owned temporary payload is removed.  This bounds disk usage
without weakening resume safety.

Turn 1 of the first dialogue is special only as a persistence opportunity: all
four counterfactual arms execute the ordinary pixel path, and the last Prefix
arm in the balanced order captures saliency and the already-produced past KV
from that same forward.  It is persisted immediately and exactly once.  Every
other Turn 1 remains an ordinary pixel request.  On Turns 2--3 ReComp executes
vision, while all cache arms read the same physical SSD store.

The reported paper TTFT is ``end_to_end_ttft_ms``.  Cold page-cache
conditioning happens before the request timestamp.  The timestamp itself is
before prompt construction/tokenization/H2D and ends only after the greedy
first output token has been selected and CUDA synchronized by ``Server``.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import psutil
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mmimpress.config import CHUNK_SIZE, MODEL_ID  # noqa: E402
from mmimpress.cvpr25 import (budget_chunk_count, prefix_chunk_ids,
                              permutation_sha256)  # noqa: E402
from mmimpress.dataset import exact_score  # noqa: E402
from mmimpress.model import LlavaRunner  # noqa: E402
from mmimpress.piggyback import (VisionForwardCapture,  # noqa: E402
                                 deterministic_method_rotation,
                                 persist_captured_visual_prefix,
                                 sha256_file, stable_json_sha256)


SCHEMA_VERSION = "mt-gqa-full-shard-turn1-piggyback-v1"
DATASET = "gqa_testdev_balanced_mt3"
BENCHMARK_TYPE = "MT-GQA-reconstructed"
DEFAULT_INDEX = Path("data/mt_gqa/dialogues.json")
METHOD_KEYS = ("recompute", "fullload", "prefix25", "prefix45")
METHODS = {
    "recompute": {"label": "ReComp", "budget": None},
    "fullload": {"label": "FullLoad", "budget": 1.0},
    "prefix25": {"label": "ImageOnly Prefix25", "budget": 0.25},
    "prefix45": {"label": "ImageOnly Prefix45", "budget": 0.45},
}
OWNER_FILE = ".mt_gqa_temp_store_owner.json"
MAX_NEW_TOKENS = 16
SEED = 1234
MIN_SHARD_SIZE = 40
MAX_SHARD_SIZE = 60
DEFAULT_SHARD_SIZE = 50
MIN_FREE_AFTER_GIB = 30.0
BUILD_HEADROOM_GIB = 3.0

WARMUP_PROMPT = (
    "USER: <image>\nThis is an unmeasured synthetic serving warm-up. "
    "Describe the image briefly. ASSISTANT:"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_hash(value: Any) -> str:
    return stable_json_sha256(value)


def _load_visdial_helpers():
    """Load the final VisDial runner once and reuse its timing machinery."""
    name = "_mt_gqa_reused_visdial_turn1_runner"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parent / "28_eval_visdial_turn1_piggyback.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import timing helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _runtime_classes():
    """Import transformer-dependent serving classes only for a GPU run."""
    from mmimpress.serve import ImageContext, Server
    return ImageContext, Server


def _load_mt_helpers():
    """Return optional canonical helpers without making CPU import depend on it."""
    try:
        return importlib.import_module("mmimpress.mt_gqa")
    except ModuleNotFoundError as exc:
        if exc.name != "mmimpress.mt_gqa":
            raise
        return None


def _dialog_id(dialog: Mapping[str, Any]) -> str:
    value = dialog.get("dialog_id", dialog.get("dialogue_id"))
    if value is None:
        raise ValueError("dialog has no dialog_id")
    return str(value)


def _image_id(dialog: Mapping[str, Any]) -> str:
    if dialog.get("image_id") is not None:
        value = dialog["image_id"]
    elif dialog.get("image_ids"):
        value = dialog["image_ids"][0]
    elif dialog.get("images"):
        value = dialog["images"][0].get("image_id")
    else:
        value = None
    if value is None:
        raise ValueError(f"dialog {_dialog_id(dialog)} has no image ID")
    value = str(value)
    if value in {"", ".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"unsafe image ID {value!r}")
    return value


def _global_ordinal(dialog: Mapping[str, Any], fallback: int) -> int:
    for key in ("global_dialog_ordinal", "dialog_ordinal",
                "global_ordinal"):
        if key in dialog:
            value = int(dialog[key])
            break
    else:
        value = int(fallback)
    if value < 0:
        raise ValueError("global dialogue ordinal must be nonnegative")
    return value


def _turn_id(turn: Mapping[str, Any], fallback: int) -> int:
    return int(turn.get("turn_id", turn.get("round_id", fallback)))


def _question(turn: Mapping[str, Any]) -> str:
    value = turn.get("question", turn.get("text"))
    if value is None:
        raise ValueError("turn has no question")
    return str(value)


def _question_id(turn: Mapping[str, Any], dialog_id: str, turn_id: int) -> str:
    return str(turn.get("question_id", turn.get("id",
               f"{dialog_id}:turn{turn_id}")))


def _gold_answer(turn: Mapping[str, Any]) -> str:
    value = turn.get("gold_answer", turn.get("answer", turn.get(
        "gold", turn.get("answers"))))
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("turn has empty gold answers")
        value = value[0]
    if value is None:
        raise ValueError("turn has no gold answer")
    return str(value)


def _normalise_dialogues(payload: Any) -> tuple[list[dict[str, Any]], dict]:
    if isinstance(payload, list):
        raw_dialogs = payload
        envelope = {}
    elif isinstance(payload, Mapping) and isinstance(
            payload.get("dialogs", payload.get("dialogues")), list):
        list_key = "dialogs" if isinstance(payload.get("dialogs"), list) \
            else "dialogues"
        raw_dialogs = payload[list_key]
        envelope = {str(k): v for k, v in payload.items() if k != list_key}
    else:
        raise ValueError("dialogues JSON must be a list or an object with dialogs[]")
    if not raw_dialogs:
        raise ValueError("dialogue workload is empty")

    dialogs: list[dict[str, Any]] = []
    seen_dialog_ids: set[str] = set()
    seen_ordinals: set[int] = set()
    seen_question_ids: set[str] = set()
    for fallback, raw in enumerate(raw_dialogs):
        if not isinstance(raw, Mapping):
            raise ValueError(f"dialog {fallback} is not an object")
        dialog = dict(raw)
        did = _dialog_id(dialog)
        image_id = _image_id(dialog)
        ordinal = _global_ordinal(dialog, fallback)
        turns = dialog.get("turns")
        if not isinstance(turns, list) or len(turns) != 3:
            raise ValueError(f"{did} must contain exactly three turns")
        turn_ids = [_turn_id(turn, i + 1) for i, turn in enumerate(turns)]
        if turn_ids != [1, 2, 3]:
            raise ValueError(f"{did} turn IDs must be exactly [1, 2, 3]")
        for i, turn in enumerate(turns, 1):
            _question(turn)
            _gold_answer(turn)
            qid = _question_id(turn, did, i)
            if qid in seen_question_ids:
                raise ValueError(f"duplicate question ID: {qid}")
            seen_question_ids.add(qid)
        if did in seen_dialog_ids:
            raise ValueError(f"duplicate dialog ID: {did}")
        if ordinal in seen_ordinals:
            raise ValueError(f"duplicate global dialogue ordinal: {ordinal}")
        seen_dialog_ids.add(did)
        seen_ordinals.add(ordinal)
        dialog["dialog_id"] = did
        dialog["image_id"] = image_id
        dialog["global_dialog_ordinal"] = ordinal
        dialogs.append(dialog)
    # Canonical file order must agree with the immutable global ordinal.  This
    # prevents partial slicing or sharding from silently changing arm order.
    if [d["global_dialog_ordinal"] for d in dialogs] != sorted(seen_ordinals):
        raise ValueError("dialogue file is not ordered by global ordinal")
    return dialogs, envelope


def _dialogue_workload_hash(dialogs: Sequence[Mapping[str, Any]]) -> str:
    rows: list[str] = []
    for dialog in dialogs:
        did = _dialog_id(dialog)
        ordinal = int(dialog["global_dialog_ordinal"])
        image_id = _image_id(dialog)
        for fallback, turn in enumerate(dialog["turns"], 1):
            tid = _turn_id(turn, fallback)
            # Match the builder's portable request-key hash: source text and
            # image paths may be represented differently without changing the
            # frozen dialogue identity/order.
            rows.append(f"{did}\t{tid}\t{_question_id(turn, did, tid)}")
    framed = "".join(f"{row}\n" for row in rows)
    return _sha256_bytes(framed.encode("utf-8"))


def resolve_dialogues(path: Path | str, *, expected_dialogs: int | None = None,
                      expected_dialogues_sha256: str | None = None,
                      expected_workload_sha256: str | None = None) -> dict:
    """Load and fail-close the full canonical workload before any slicing."""
    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"dialogues must be a regular file: {path}")
    payload = json.loads(path.read_text())
    dialogs, envelope = _normalise_dialogues(payload)
    helper = _load_mt_helpers()
    canonical_validation = None
    if (helper is not None
            and envelope.get("schema_version") == getattr(
                helper, "SCHEMA_VERSION", object())):
        canonical_validation = helper.validate_dialogues(
            dialogs, check_images=True)
    file_sha = sha256_file(path)
    workload_sha = _dialogue_workload_hash(dialogs)
    recorded_hash = envelope.get("artifact_content_sha256")
    if recorded_hash is not None:
        if not isinstance(payload, Mapping):
            raise AssertionError("impossible envelope state")
        body = {k: v for k, v in payload.items()
                if k != "artifact_content_sha256"}
        if str(recorded_hash) != _json_hash(body):
            raise ValueError("dialogues top-level content hash mismatch")
    checks = (
        (expected_dialogs, len(dialogs), "dialog count"),
        (expected_dialogues_sha256, file_sha, "dialogues file SHA256"),
        (expected_workload_sha256, workload_sha, "workload SHA256"),
    )
    for expected, observed, label in checks:
        if expected is not None and expected != observed:
            raise ValueError(f"{label} mismatch: expected {expected}, got {observed}")
    image_ids = [_image_id(d) for d in dialogs]
    benchmark_type = str(envelope.get("benchmark_type", BENCHMARK_TYPE))
    if benchmark_type != BENCHMARK_TYPE:
        raise ValueError(
            f"this frozen runner requires {BENCHMARK_TYPE}, got "
            f"{benchmark_type}")
    return {
        "path": path,
        "dialogs": dialogs,
        "envelope": envelope,
        "benchmark_type": benchmark_type,
        "official_benchmark_identity_claimed": False,
        "benchmark_disclaimer": str(getattr(
            helper, "DISCLAIMER",
            "Deterministic MT-GQA reconstruction; official artifact identity "
            "is not claimed.")),
        "dialogues_file_sha256": file_sha,
        "workload_sha256": workload_sha,
        "artifact_content_sha256": recorded_hash,
        "canonical_validation": canonical_validation,
        "n_dialogs": len(dialogs),
        "n_turns": len(dialogs) * 3,
        "n_images": len(set(image_ids)),
    }


def select_dialogues(workload: Mapping[str, Any],
                     max_dialogs: int | None) -> dict:
    """Create a smoke/pilot view without renumbering global ordinals."""
    full = list(workload["dialogs"])
    if max_dialogs is None:
        selected = full
    else:
        if int(max_dialogs) not in (10, 100):
            raise ValueError("max-dialogs is restricted to the 10/100 smoke/pilot")
        if int(max_dialogs) > len(full):
            raise ValueError("max-dialogs exceeds the frozen workload")
        selected = full[:int(max_dialogs)]
    out = dict(workload)
    out.update({
        "dialogs": selected,
        "n_dialogs": len(selected),
        "n_turns": len(selected) * 3,
        "n_images": len({_image_id(d) for d in selected}),
        "partial_workload": len(selected) != len(full),
        "source_full_n_dialogs": len(full),
        "source_full_workload_sha256": workload["workload_sha256"],
        "selected_workload_sha256": _dialogue_workload_hash(selected),
    })
    return out


def group_dialogues_by_image(dialogs: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Stable image groups; each group retains every selected dialogue."""
    positions: dict[str, int] = {}
    groups: list[dict] = []
    for dialog in dialogs:
        image_id = _image_id(dialog)
        if image_id not in positions:
            positions[image_id] = len(groups)
            groups.append({
                "image_id": image_id,
                "image_ordinal": len(groups),
                "dialogs": [],
            })
        groups[positions[image_id]]["dialogs"].append(dialog)
    return groups


def shard_image_groups(groups: Sequence[Mapping[str, Any]], shard_index: int,
                       shard_size: int = DEFAULT_SHARD_SIZE) -> dict:
    if not MIN_SHARD_SIZE <= int(shard_size) <= MAX_SHARD_SIZE:
        raise ValueError("shard-size must be between 40 and 60 images")
    if int(shard_index) < 0:
        raise ValueError("shard-index must be nonnegative")
    n_shards = math.ceil(len(groups) / int(shard_size))
    if n_shards == 0 or int(shard_index) >= n_shards:
        raise ValueError(f"shard index {shard_index} outside {n_shards} shards")
    start = int(shard_index) * int(shard_size)
    stop = min(start + int(shard_size), len(groups))
    return {
        "groups": list(groups[start:stop]),
        "start": start,
        "stop": stop,
        "n_shards": n_shards,
    }


def method_order(global_dialog_ordinal: int, seed: int = SEED) -> tuple[str, ...]:
    """One cyclic order per dialogue, reused unchanged on all three turns."""
    if int(seed) != SEED:
        raise ValueError("the frozen method-order contract uses seed 1234")
    # The experiment seed fixes workload construction; it does not phase-shift
    # the counterbalancing cycle.  Thus zero-based D1 begins with ReComp, D2
    # with FullLoad, and so on, exactly as preregistered.
    return deterministic_method_rotation(
        METHOD_KEYS, int(global_dialog_ordinal), 0)


def designated_source_method(order: Sequence[str]) -> str:
    prefixes = [method for method in order if str(method).startswith("prefix")]
    if set(prefixes) != {"prefix25", "prefix45"}:
        raise ValueError("order must contain both and only the two Prefix arms")
    return prefixes[-1]


def _fallback_prior_history_text(dialog: Mapping[str, Any], turn_id: int) -> str:
    chunks = []
    for fallback, turn in enumerate(dialog["turns"], 1):
        tid = _turn_id(turn, fallback)
        if tid >= int(turn_id):
            break
        chunks.extend((f"Q{tid}: {_question(turn)}",
                       f"A{tid}: {_gold_answer(turn)}"))
    return "\n".join(chunks)


def prior_history_text(dialog: Mapping[str, Any], turn_id: int) -> str:
    helper = _load_mt_helpers()
    if helper is not None:
        for name in ("mt_gqa_history_text", "prior_history_text",
                     "mt_gqa_prior_history_text", "gold_history_text"):
            fn = getattr(helper, name, None)
            if fn is not None:
                return str(fn(dialog, int(turn_id)))
    return _fallback_prior_history_text(dialog, turn_id)


def _fallback_gold_history_prompt(dialog: Mapping[str, Any], turn_id: int) -> str:
    lines = ["USER: <image>"]
    for fallback, turn in enumerate(dialog["turns"], 1):
        tid = _turn_id(turn, fallback)
        if tid > int(turn_id):
            break
        if tid == 1:
            lines.append(_question(turn))
        else:
            lines.append(f"USER: {_question(turn)}")
        if tid < int(turn_id):
            lines.append(f"ASSISTANT: {_gold_answer(turn)}")
        else:
            lines.append("Answer the question using a single word or phrase. ASSISTANT:")
    return "\n".join(lines)


def gold_history_prompt(dialog: Mapping[str, Any], turn_id: int) -> str:
    helper = _load_mt_helpers()
    if helper is not None:
        for name in ("gold_history_prompt", "mt_gqa_prompt", "dialog_prompt"):
            fn = getattr(helper, name, None)
            if fn is not None:
                body = str(fn(dialog, int(turn_id))).strip()
                # ``mmimpress.mt_gqa`` deliberately renders the causal body;
                # this serving runner owns the raw Vicuna/LLaVA turn wrapper.
                if body.startswith("USER:") and body.endswith("ASSISTANT:"):
                    return body
                return f"USER: {body} ASSISTANT:"
    return _fallback_gold_history_prompt(dialog, turn_id)


def resolve_image_path(dialog: Mapping[str, Any]) -> Path:
    value = dialog.get("image_path")
    if value is None and dialog.get("images"):
        value = dialog["images"][0].get("image_path")
    if value is None:
        image_id = _image_id(dialog)
        candidates = [
            ROOT / "data/gqa_large/images" / f"{image_id}.jpg",
            ROOT / "data/gqa_large/images" / f"{image_id}.png",
        ]
        found = next((path for path in candidates if path.is_file()), None)
        if found is None:
            raise FileNotFoundError(f"no image for {image_id}")
        return found.resolve()
    helper = _load_mt_helpers()
    if helper is not None:
        fn = getattr(helper, "resolve_image_path", None)
        if fn is not None:
            path = Path(fn(value)).resolve()
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"image is not a regular file: {path}")
            return path
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"image is not a regular file: {path}")
    return path


def _read_owner(root: Path) -> dict:
    marker = root / OWNER_FILE
    if not marker.is_file() or marker.is_symlink():
        raise ValueError(f"missing regular ownership marker: {marker}")
    return json.loads(marker.read_text())


def _write_exclusive_json(path: Path, value: Any) -> None:
    """Fsync and atomically publish without replacing an existing artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=1, ensure_ascii=False,
                      allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _claim_temp_root(root: Path | str, experiment_id: str,
                     dataset: str = DATASET) -> dict:
    root = Path(root)
    if not root.is_absolute():
        raise ValueError("temporary root must be absolute")
    if root.exists() and root.is_symlink():
        raise ValueError(f"temporary root is a symlink: {root}")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    wanted = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": str(experiment_id),
        "dataset": str(dataset),
        "purpose": "temporary_visual_kv_only",
    }
    marker = root / OWNER_FILE
    if marker.exists():
        if _read_owner(root) != wanted:
            raise ValueError(f"existing temporary root has another owner: {root}")
    else:
        _write_exclusive_json(marker, wanted)
    payload = root / "payload"
    if payload.exists() and payload.is_symlink():
        raise ValueError(f"temporary payload root is a symlink: {payload}")
    payload.mkdir(exist_ok=True)
    return wanted


def assert_owned_temp_path(path: Path | str, root: Path | str,
                           experiment_id: str,
                           expected_image_id: str | None = None) -> Path:
    """Permit deletion only for ``root/payload/<one exact image leaf>``."""
    candidate, root = Path(path), Path(root)
    if not candidate.is_absolute() or not root.is_absolute():
        raise ValueError("owned temporary paths must be absolute")
    if root.is_symlink() or candidate.is_symlink():
        raise ValueError("owned temporary paths may not be symlinks")
    root_resolved = root.resolve()
    payload = root_resolved / "payload"
    candidate_resolved = candidate.resolve(strict=False)
    if payload.is_symlink() or candidate_resolved.parent != payload:
        raise ValueError(f"target is not one direct temporary image leaf: {path}")
    if candidate_resolved.name in {"", ".", ".."}:
        raise ValueError(f"invalid temporary image leaf: {path}")
    if (expected_image_id is not None
            and candidate_resolved.name != str(expected_image_id)):
        raise ValueError("temporary leaf does not match the expected image")
    owner = _read_owner(root_resolved)
    if (owner.get("schema_version") != SCHEMA_VERSION
            or owner.get("experiment_id") != str(experiment_id)
            or owner.get("dataset") != DATASET
            or owner.get("purpose") != "temporary_visual_kv_only"):
        raise ValueError(f"temporary-root ownership mismatch: {root_resolved}")
    return candidate_resolved


def remove_owned_temp_store(path: Path | str, root: Path | str,
                            experiment_id: str, image_id: str) -> bool:
    """Remove one verified experiment-owned image store, never an ancestor."""
    target = assert_owned_temp_path(path, root, experiment_id, image_id)
    if not os.path.lexists(target):
        return False
    if target.is_symlink() or not target.is_dir():
        raise ValueError(f"temporary store is not a real directory: {target}")
    shutil.rmtree(target)
    return True


def _capacity_guard(path: Path, *, reserve_bytes: int,
                    extra_headroom_bytes: int = 0) -> dict:
    usage = shutil.disk_usage(path)
    required = int(reserve_bytes) + int(extra_headroom_bytes)
    used_pct = 100.0 * float(usage.used) / float(usage.used + usage.free)
    if int(usage.free) < required:
        raise RuntimeError(
            f"free-space guard failed: free={usage.free}, required={required}")
    if used_pct >= 96.0:
        raise RuntimeError(
            f"disk-used guard failed: {used_pct:.3f}% is not below 96%")
    return {
        "disk_total_bytes": int(usage.total),
        "disk_used_bytes": int(usage.used),
        "disk_free_bytes": int(usage.free),
        "disk_used_percent": float(used_pct),
        "reserve_bytes": int(reserve_bytes),
        "extra_headroom_bytes": int(extra_headroom_bytes),
    }


def validate_prefix_selection(selected_per_layer: Sequence[Sequence[int]],
                              n_chunks_total: int, budget: float) -> list[int]:
    """Prove an observed Prefix request is exact physical first-k."""
    wanted_count = budget_chunk_count(int(n_chunks_total), float(budget))
    wanted = prefix_chunk_ids(int(n_chunks_total), float(budget))
    if len(wanted) != wanted_count:
        raise AssertionError("budget_chunk_count and prefix_chunk_ids disagree")
    if not selected_per_layer:
        raise AssertionError("prefix request recorded no selected chunks")
    for layer in selected_per_layer:
        if [int(value) for value in layer] != wanted:
            raise AssertionError("prefix request is not exact first-k")
    return wanted


def validate_nested_prefixes(prefix25_row: Mapping[str, Any],
                             prefix45_row: Mapping[str, Any]) -> None:
    n25 = int(prefix25_row["n_chunks_total"])
    n45 = int(prefix45_row["n_chunks_total"])
    if n25 != n45:
        raise AssertionError("Prefix25/45 saw different chunk universes")
    ids25 = validate_prefix_selection(
        prefix25_row["selected_chunk_ids_per_layer"], n25, 0.25)
    ids45 = validate_prefix_selection(
        prefix45_row["selected_chunk_ids_per_layer"], n45, 0.45)
    if not set(ids25).issubset(ids45):
        raise AssertionError("Prefix25 is not a subset of Prefix45")


def _image_artifact_path(run_dir: Path, image_id: str) -> Path:
    _image_id({"dialog_id": "path-check", "image_id": image_id})
    return run_dir / "images" / f"{image_id}.json"


def _artifact_body_hash(artifact: Mapping[str, Any]) -> str:
    return _json_hash({k: v for k, v in artifact.items()
                       if k != "artifact_content_sha256"})


def _expected_row_keys(dialogs: Sequence[Mapping[str, Any]]) -> list[tuple]:
    keys = []
    for dialog in dialogs:
        did = _dialog_id(dialog)
        for fallback, turn in enumerate(dialog["turns"], 1):
            tid = _turn_id(turn, fallback)
            for method in method_order(int(dialog["global_dialog_ordinal"])):
                keys.append((did, tid, method))
    return keys


def validate_resume_artifact(path: Path, *, experiment_id: str,
                             image_group: Mapping[str, Any], shard_index: int,
                             workload: Mapping[str, Any], seed: int) -> dict:
    """Fail closed before skipping a previously published image artifact."""
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"resume artifact is not a regular file: {path}")
    artifact = json.loads(path.read_text())
    image_id = str(image_group["image_id"])
    expected = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": str(experiment_id),
        "dataset": DATASET,
        "benchmark_type": workload["benchmark_type"],
        "image_id": image_id,
        "image_ordinal": int(image_group["image_ordinal"]),
        "shard_index": int(shard_index),
        "dialogues_file_sha256": str(workload["dialogues_file_sha256"]),
        "source_full_workload_sha256": str(
            workload["source_full_workload_sha256"]),
        "selected_workload_sha256": str(workload["selected_workload_sha256"]),
        "model_revision": workload.get("model_revision"),
    }
    mismatch = {key: (artifact.get(key), value)
                for key, value in expected.items()
                if artifact.get(key) != value}
    if mismatch:
        raise ValueError(f"resume artifact identity mismatch: {mismatch}")
    if artifact.get("artifact_content_sha256") != _artifact_body_hash(artifact):
        raise ValueError("resume artifact content hash mismatch")
    rows = artifact.get("rows")
    manifest = artifact.get("store_manifest")
    if not isinstance(rows, list) or not isinstance(manifest, Mapping):
        raise ValueError("resume artifact omits rows/store manifest")
    expected_keys = _expected_row_keys(image_group["dialogs"])
    observed_keys = [(str(r.get("dialog_id")), int(r.get("turn_id", -1)),
                      str(r.get("method_key"))) for r in rows]
    if observed_keys != expected_keys:
        raise ValueError("resume artifact has wrong row coverage or execution order")
    store_id = str(manifest.get("meta_sha256", ""))
    if not store_id:
        raise ValueError("resume artifact has no physical store identity")
    by_request: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    session_ids_by_dialog: dict[str, set[str]] = defaultdict(set)
    context_ids_by_dialog: dict[str, set[str]] = defaultdict(set)
    first_dialog = _dialog_id(image_group["dialogs"][0])
    for row in rows:
        method = str(row["method_key"])
        did, tid = str(row["dialog_id"]), int(row["turn_id"])
        if row.get("method") != METHODS[method]["label"]:
            raise ValueError("resume row has wrong method label")
        if row.get("budget") != METHODS[method]["budget"]:
            raise ValueError("resume row has wrong budget")
        if row.get("model_revision") != workload.get("model_revision"):
            raise ValueError("resume row has wrong model revision")
        if (row.get("gpu_request_cache_fresh") is not True
                or row.get("text_kv_reused_from_prior_turn") is not False):
            raise ValueError("resume row violates dialogue cache isolation")
        session_id = row.get("dialogue_session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("resume row has no dialogue session ID")
        session_ids_by_dialog[did].add(session_id)
        context_id = row.get("context_instance_id")
        if context_id is not None:
            if not isinstance(context_id, str) or not context_id:
                raise ValueError("resume row has invalid context instance ID")
            context_ids_by_dialog[did].add(context_id)
        order = list(method_order(int(row["global_dialog_ordinal"]), seed))
        if row.get("method_order") != order:
            raise ValueError("resume row has wrong balanced order")
        if not float(row["end_to_end_ttft_ms"]) < float(row["request_e2e_ms"]):
            raise ValueError("resume artifact violates TTFT < E2E")
        should_use = tid >= 2 and method != "recompute"
        if bool(row.get("used_by_request")) != should_use:
            raise ValueError("resume row has wrong store-use semantics")
        if any(int(row.get(key, -1)) != 0 for key in (
                "layout_questions_used", "layout_answers_used",
                "calibration_questions", "future_turns_used_for_layout")):
            raise ValueError("resume row contains calibration/layout leakage")
        if tid == 1:
            if (int(row.get("ssd_read_bytes", -1)) != 0
                    or int(row.get("vision_forward_count", -1)) != 1):
                raise ValueError("Turn 1 must be a normal pixel request")
        elif method == "recompute":
            if (int(row.get("ssd_read_bytes", -1)) != 0
                    or int(row.get("vision_forward_count", -1)) != 1):
                raise ValueError("ReComp baseline contract failed")
        else:
            if (row.get("store_id") != store_id
                    or int(row.get("vision_forward_count", -1)) != 0
                    or row.get("page_cache_conditioning_excluded_from_ttft") is not True):
                raise ValueError("cache arm shared-store contract failed")
            if (method == "fullload"
                    and int(row.get("ssd_read_bytes", -1)) != int(
                        manifest["visual_kv_bytes"])):
                raise ValueError("resume FullLoad byte count is incomplete")
            if method.startswith("prefix"):
                validate_prefix_selection(
                    row["selected_chunk_ids_per_layer"],
                    int(row["n_chunks_total"]), float(row["budget"]))
                if any(int(row.get(key, -1)) != 0 for key in (
                        "static_score_calls", "query_score_calls",
                        "diversity_calls")):
                    raise ValueError("Prefix row contains online selector calls")
        # The source may persist midway through first-dialog T1.  Physical
        # existence is chronology, while used_by_request remains false.
        if did != first_dialog and tid == 1 and not row.get(
                "physical_store_exists_at_request_start"):
            raise ValueError("later dialogue Turn 1 should see the physical store")
        by_request[(did, tid)][method] = row
    for (did, tid), request_rows in by_request.items():
        if len(request_rows) != 4:
            raise ValueError(f"incomplete method coverage: {did}/T{tid}")
        prompt_hashes = {r.get("prompt_sha256") for r in request_rows.values()}
        history_hashes = {r.get("text_history_sha256") for r in request_rows.values()}
        suffix_hashes = {r.get("suffix_ids_sha256") for r in request_rows.values()}
        if len(prompt_hashes) != 1 or len(history_hashes) != 1 or len(suffix_hashes) != 1:
            raise ValueError(f"method input mismatch: {did}/T{tid}")
        if tid == 1:
            if len({r.get("prediction") for r in request_rows.values()}) != 1:
                raise ValueError(f"Turn 1 predictions differ: {did}")
            if len({r.get("first_token_id") for r in request_rows.values()}) != 1:
                raise ValueError(f"Turn 1 first tokens differ: {did}")
        if tid >= 2:
            validate_nested_prefixes(
                request_rows["prefix25"], request_rows["prefix45"])
    if any(len(values) != 1 for values in session_ids_by_dialog.values()):
        raise ValueError("resume artifact changes session ID within dialogue")
    if len({next(iter(values)) for values in session_ids_by_dialog.values()}) \
            != len(session_ids_by_dialog):
        raise ValueError("resume artifact reuses session ID across dialogues")
    if (set(context_ids_by_dialog) != set(session_ids_by_dialog)
            or any(len(values) != 1
                   for values in context_ids_by_dialog.values())):
        raise ValueError("resume artifact context-instance coverage mismatch")
    if len({next(iter(values)) for values in context_ids_by_dialog.values()}) \
            != len(context_ids_by_dialog):
        raise ValueError("resume artifact reuses ImageContext across dialogues")
    first_dialog_obj = image_group["dialogs"][0]
    first_did = _dialog_id(first_dialog_obj)
    first_order = list(method_order(
        int(first_dialog_obj["global_dialog_ordinal"]), seed))
    source_method = designated_source_method(first_order)
    source_position = first_order.index(source_method)
    source_rows = [row for row in rows
                   if bool(row.get("persistence_source_request"))]
    if len(source_rows) != 1:
        raise ValueError("resume artifact does not prove one store build")
    source_row = source_rows[0]
    if (str(source_row.get("dialog_id")) != first_did
            or int(source_row.get("turn_id", -1)) != 1
            or source_row.get("method_key") != source_method
            or int(source_row.get("method_order_position", -1))
            != source_position):
        raise ValueError("resume persistence source is not designated T1 Prefix")
    first_t1 = by_request[(first_did, 1)]
    for position, method in enumerate(first_order):
        expected_exists = position > source_position
        if bool(first_t1[method].get(
                "physical_store_exists_at_request_start")) != expected_exists:
            raise ValueError("resume first-T1 physical-store chronology mismatch")
    cached_rows = [row for row in rows if row.get("used_by_request") is True]
    if {row.get("permutation_sha256") for row in cached_rows} != {
            manifest.get("permutation_sha256")}:
        raise ValueError("resume cache permutation fingerprints differ")
    for method in ("prefix25", "prefix45"):
        fingerprints = set()
        for row in cached_rows:
            if row["method_key"] != method:
                continue
            expected_fingerprint = _json_hash(
                row["selected_chunk_ids_per_layer"])
            if row.get("selection_fingerprint_sha256") \
                    != expected_fingerprint:
                raise ValueError("resume Prefix selection fingerprint mismatch")
            fingerprints.add(expected_fingerprint)
        if len(fingerprints) != 1:
            raise ValueError("resume Prefix selection changed across requests")
    if not artifact.get("validation", {}).get("passed"):
        raise ValueError("resume artifact validation did not pass")
    persistence = artifact.get("persistence_overhead")
    if (not isinstance(persistence, Mapping)
            or int(persistence.get("store_build_count", -1)) != 1
            or persistence.get("source_dialog_id") != first_did
            or persistence.get("source_method_key") != source_method
            or int(persistence.get("source_turn_id", -1)) != 1):
        raise ValueError("resume artifact has invalid persistence record")
    return artifact


def _ensure_run_config(run_dir: Path, config: Mapping[str, Any]) -> None:
    if not run_dir.is_absolute():
        raise ValueError("run-dir must be absolute")
    if run_dir.exists() and run_dir.is_symlink():
        raise ValueError("run-dir may not be a symlink")
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config.json"
    invariant = (
        "schema_version", "experiment_id", "dataset",
        "benchmark_type",
        "dialogues_file_sha256", "source_full_workload_sha256",
        "selected_workload_sha256", "n_dialogs", "n_images", "seed",
        "method_keys", "shard_size", "max_new_tokens", "model_revision",
    )
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise ValueError("existing run config is not a regular file")
        current = json.loads(path.read_text())
        mismatch = {key: (current.get(key), config.get(key)) for key in invariant
                    if current.get(key) != config.get(key)}
        if mismatch:
            raise ValueError(f"existing run config mismatch: {mismatch}")
    else:
        _write_exclusive_json(path, dict(config))
    for name in ("images", "shards"):
        child = run_dir / name
        if child.exists() and child.is_symlink():
            raise ValueError(f"run artifact directory is a symlink: {child}")
        child.mkdir(exist_ok=True)


def _hash_tensor(tensor: torch.Tensor) -> str:
    return _load_visdial_helpers()._hash_tensor(tensor)


def _hash_tensor_mapping(values: Mapping[str, Any]) -> str:
    return _load_visdial_helpers()._hash_tensor_mapping(values)


def _image_input_hash(values: Mapping[str, Any]) -> str:
    return _load_visdial_helpers()._image_input_hash(values)


def _combined_suffix(runner, combined_ids: torch.Tensor) -> torch.Tensor:
    ids = combined_ids[0] if combined_ids.ndim == 2 else combined_ids
    positions = (ids == runner.image_token_id).nonzero(as_tuple=True)[0]
    if positions.numel() < 1:
        raise AssertionError("combined multimodal prompt has no image token")
    expected = torch.arange(positions[0], positions[-1] + 1,
                            device=positions.device)
    if not torch.equal(positions, expected):
        raise AssertionError("combined image-token span is not contiguous")
    return ids[int(positions[-1]) + 1:]


def _tokenized_suffix(runner, prompt: str) -> torch.Tensor:
    tokenized = runner.processor.tokenizer(prompt, return_tensors="pt")
    return _load_visdial_helpers()._suffix_from_tokenized(runner, tokenized)


def _run_normal_request(runner, server, image, dialog: Mapping[str, Any],
                        turn_id: int, expected_prompt: str, *,
                        capture_saliency: bool, capture_cache: bool) -> dict:
    """Normal pixel path with the exact final VisDial timing boundaries."""
    helpers = _load_visdial_helpers()
    capture = VisionForwardCapture(runner, capture_saliency=capture_saliency)
    with capture:
        torch.cuda.synchronize()
        request_started = time.perf_counter()
        prompt_started = time.perf_counter()
        request_prompt = gold_history_prompt(dialog, int(turn_id))
        prompt_build_ms = (time.perf_counter() - prompt_started) * 1e3
        if request_prompt != expected_prompt:
            raise AssertionError("timed prompt differs from canonical reference")
        enc_cpu, processor_timing = helpers._exact_processor_call(
            runner, image, request_prompt)
        h2d_started = time.perf_counter()
        enc_device = runner.to_device(enc_cpu)
        torch.cuda.synchronize()
        input_h2d_ms = (time.perf_counter() - h2d_started) * 1e3
        phase = {
            "prompt_build_ms": float(prompt_build_ms),
            "tokenization_ms": float(processor_timing["tokenization_ms"]),
            "image_preprocess_ms": float(
                processor_timing["image_preprocess_ms"]),
            "input_prepare_ms": float(processor_timing["input_prepare_ms"]),
            "input_h2d_ms": float(input_h2d_ms),
            "processor_total_ms": float(processor_timing["processor_total_ms"]),
        }
        result = server.recompute(
            enc_device, return_past_key_values=capture_cache)
        returned = time.perf_counter()
    capture_finished = time.perf_counter()
    result.update(helpers._timing_fields(
        result, request_started, phase, returned))
    stats = capture.stats()
    result.update({
        "prompt": request_prompt,
        "enc_cpu": enc_cpu,
        "capture": capture,
        "capture_stats": stats,
        "vision_ms": float(stats["vision_ms"]),
        "vision_forward_count": int(capture.call_count),
        "separate_vision_forward_count": 0,
        "capture_cleanup_finished_at_s": float(capture_finished),
        "capture_post_answer_ms": float(max(
            0.0, capture_finished - result["postprocess_finished_at_s"]
        ) * 1e3),
    })
    return result


def _run_stored_request(runner, server, ctx, dialog: Mapping[str, Any],
                        turn_id: int, expected_prompt: str, method_key: str,
                        *, budget: float, cold: bool, seed: int,
                        image_id: str) -> dict:
    """SSD path; page-cache conditioning and guard setup precede TTFT."""
    helpers = _load_visdial_helpers()
    with helpers._NoVisionForward(runner) as guard:
        conditioning_started = time.perf_counter()
        conditioning_method = "none_warm"
        if cold:
            ctx.reader.drop_all()
            conditioning_method = "posix_fadvise_DONTNEED"
        conditioning_finished = time.perf_counter()
        torch.cuda.synchronize()
        request_started = time.perf_counter()
        prompt_started = time.perf_counter()
        request_prompt = gold_history_prompt(dialog, int(turn_id))
        prompt_build_ms = (time.perf_counter() - prompt_started) * 1e3
        if request_prompt != expected_prompt:
            raise AssertionError("timed prompt differs from canonical reference")
        started = time.perf_counter()
        tokenized = runner.processor.tokenizer(
            request_prompt, return_tensors="pt")
        tokenization_ms = (time.perf_counter() - started) * 1e3
        started = time.perf_counter()
        suffix_cpu = helpers._suffix_from_tokenized(runner, tokenized)
        input_prepare_ms = (time.perf_counter() - started) * 1e3
        started = time.perf_counter()
        suffix_device = suffix_cpu.to(runner.model.device)
        torch.cuda.synchronize()
        input_h2d_ms = (time.perf_counter() - started) * 1e3
        phase = {
            "prompt_build_ms": float(prompt_build_ms),
            "tokenization_ms": float(tokenization_ms),
            "image_preprocess_ms": 0.0,
            "input_prepare_ms": float(input_prepare_ms),
            "input_h2d_ms": float(input_h2d_ms),
            "processor_total_ms": None,
        }
        if method_key == "fullload":
            result = server.request(
                ctx, mode="fullload", cold=False, suffix_ids=suffix_device)
        else:
            result = server.request_cvpr25(
                ctx, static=None, budget=float(budget), mode="prefix",
                sep_policy="sidecar", cold=False, seed=int(seed),
                image_id=image_id, suffix_ids=suffix_device,
                expected_prefix_layout="visionzip_image_only")
        returned = time.perf_counter()
    result.update(helpers._timing_fields(
        result, request_started, phase, returned))
    result.update({
        "prompt": request_prompt,
        "suffix_cpu": suffix_cpu,
        "vision_ms": 0.0,
        "vision_forward_count": int(guard.calls),
        "separate_vision_forward_count": 0,
        "page_cache_conditioning_started_at_s": float(conditioning_started),
        "page_cache_conditioning_finished_at_s": float(conditioning_finished),
        "page_cache_conditioning_ms": float(
            (conditioning_finished - conditioning_started) * 1e3),
        "page_cache_conditioning_method": conditioning_method,
        "page_cache_conditioning_excluded_from_ttft": True,
        "cache_conditioning_started_at_s": float(conditioning_started),
        "cache_conditioning_finished_at_s": float(conditioning_finished),
        "cache_conditioning_excluded_from_ttft": True,
    })
    if not conditioning_finished < request_started:
        raise AssertionError("cold-cache conditioning overlaps TTFT")
    return result


def _io_fields(result: Mapping[str, Any], method_key: str,
               full_visual_bytes: int | None) -> dict:
    fields = _load_visdial_helpers()._io_fields(
        result, method_key, full_visual_bytes)
    per_kind = fields.get("io_detail", {})
    fields.update({
        "normal_kv_preads": int(per_kind.get("k", {}).get("preads", 0)
                                + per_kind.get("v", {}).get("preads", 0)),
        "separator_preads": int(per_kind.get("sep", {}).get("preads", 0)),
        "first_k_planning_ms": float(
            result.get("selector_ms", 0.0) or 0.0),
        "actual_selected_normal_chunk_fraction": (
            None if method_key == "recompute" else
            1.0 if method_key == "fullload" else
            result.get("touched_chunk_fraction")),
    })
    # The shared VisDial helper uses ``None`` for FullLoad because no scatter
    # operation exists.  Downstream tabular analysis expects a numeric phase;
    # represent the absent operation as measured zero, not missing data.
    if fields.get("scatter_ms") is None:
        fields["scatter_ms"] = 0.0
    return fields


def _store_manifest(store_dir: Path, meta: Mapping[str, Any],
                    persisted: Mapping[str, Any]) -> dict:
    files = sorted(path for path in store_dir.rglob("*") if path.is_file())
    sizes = {path.relative_to(store_dir).as_posix(): int(path.stat().st_size)
             for path in files}
    hashes = persisted.get("hashes", {})
    file_hashes = hashes.get("files_sha256", {})
    return {
        "physical_layout": "visionzip_image_only",
        "n_files": len(files),
        "file_sizes": sizes,
        "total_store_bytes": int(sum(sizes.values())),
        "meta_sha256": file_hashes.get("meta.json") or sha256_file(
            store_dir / "meta.json"),
        "layout_sha256": file_hashes.get("visionzip_layout.pt") or sha256_file(
            store_dir / "visionzip_layout.pt"),
        "permutation_sha256": str(meta["permutation_sha256"]),
        "inverse_permutation_sha256": str(meta["inverse_permutation_sha256"]),
        "visual_kv_bytes": int(meta["bytes_visual_kv"]),
        "separator_sidecar_bytes": int(meta["bytes_separator_sidecar"]),
        "probe_sidecar_bytes": int(meta["bytes_probe_sidecar"]),
        "prefix_len": int(meta["prefix_len"]),
        "v_token_start": int(meta["v_token_start"]),
        "v_token_num": int(meta["v_token_num"]),
        "n_chunks_per_layer": int(meta["n_chunks_per_layer"]),
        "chunk_size": int(meta["chunk_size"]),
        "num_layers": int(meta["num_layers"]),
        "payload_full_hash_performed": bool(
            hashes.get("full_integrity_hash", False)),
        "prefix_kv_sample_sha256": hashes.get("prefix_kv_sample_sha256"),
    }


def _layout_summary(store_dir: Path, meta: Mapping[str, Any]) -> dict:
    layout = torch.load(store_dir / "visionzip_layout.pt",
                        map_location="cpu", weights_only=False)
    order = [int(v) for v in torch.as_tensor(
        layout["stored_to_original"]).flatten().tolist()]
    inverse = [int(v) for v in torch.as_tensor(
        layout["original_to_stored"]).flatten().tolist()]
    if permutation_sha256(order) != str(meta["permutation_sha256"]):
        raise AssertionError("layout permutation hash disagrees with meta")
    return {
        "physical_layout": "visionzip_image_only",
        "importance_source": layout.get("importance_source"),
        "capture_source": layout.get("capture_source"),
        "stored_to_original_sha256": permutation_sha256(order),
        "original_to_stored_sha256": permutation_sha256(inverse),
        "n_rows": len(order),
        "newline_original": [int(v) for v in torch.as_tensor(
            layout["newline_original"]).flatten().tolist()],
        "layout_uses_dataset_question": bool(
            layout.get("layout_uses_dataset_question", False)),
        "llm_used_for_layout_scoring": bool(
            layout.get("llm_used_for_layout_scoring", False)),
        "calibration_questions": int(layout.get("calibration_questions", 0)),
        "image_input_sha256": layout.get("image_input_sha256"),
    }


def _persistence_record(dialog: Mapping[str, Any], source_method: str,
                        source_execution_id: str, capture,
                        persisted: Mapping[str, Any], persist_started: float,
                        context_ready: float, context_open_ms: float) -> dict:
    timing = persisted["timing_ms"]
    capture_stats = capture.stats()
    saliency_d2h_ms = float(capture_stats.get(
        "saliency_materialize_ms", 0.0))
    total_persist_ms = (saliency_d2h_ms + float(timing["persist_ms"])
                        + float(context_open_ms))
    hashes = persisted.get("hashes", {})
    files = hashes.get("files_sha256", {})
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET,
        "image_id": _image_id(dialog),
        "source_dialog_id": _dialog_id(dialog),
        "source_global_dialog_ordinal": int(dialog["global_dialog_ordinal"]),
        "source_turn_id": 1,
        "source_method_key": source_method,
        "source_method": METHODS[source_method]["label"],
        "source_execution_id": source_execution_id,
        "capture_from_turn1": True,
        "capture_from_same_answer1_forward": True,
        "store_build_count": 1,
        "vision_forward_count": int(capture.call_count),
        "separate_vision_forward_count": 0,
        "separate_prefix_forward_count": 0,
        "saliency_call_count": int(capture.saliency_call_count),
        "saliency_source": (
            "penultimate_cls_to_patch_attention_same_vision_forward"),
        "saliency_extra_ms": float(capture_stats.get(
            "saliency_reduction_ms", 0.0)),
        "saliency_d2h_ms": saliency_d2h_ms,
        "persist_ms": float(total_persist_ms),
        "total_persist_ms": float(total_persist_ms),
        "helper_total_ms": float(timing["helper_total_ms"]),
        "token_mapping_ms": float(timing.get("token_mapping_ms", 0.0)),
        "permutation_ms": float(timing.get("permutation_ms", 0.0)),
        "kv_materialize_ms": float(timing.get("kv_materialize_ms", 0.0)),
        "kv_repack_ms": float(timing.get("kv_repack_ms", 0.0)),
        "repack_ms": float(timing.get("kv_repack_ms", 0.0)),
        "ssd_write_ms": float(timing.get("ssd_write_ms", 0.0)),
        "buffered_write_ms": float(timing.get("ssd_write_ms", 0.0)),
        "fsync_ms": float(timing.get("file_fsync_ms", 0.0)
                          + timing.get("directory_fsync_ms", 0.0)
                          + timing.get("parent_fsync_ms", 0.0)),
        "context_open_ms": float(context_open_ms),
        "persist_started_at_s": float(persist_started),
        "store_ready_at_s": float(context_ready),
        "visual_kv_bytes": int(persisted["bytes"]["visual_kv"]),
        "total_ssd_write_bytes": int(persisted["bytes"]["total"]),
        "bytes_written": int(persisted["bytes"]["total"]),
        "permutation_sha256": hashes.get("permutation_sha256"),
        "meta_sha256": files.get("meta.json"),
        "layout_sha256": files.get("visionzip_layout.pt"),
        "durable_fsync_completed": bool(
            persisted["durability"]["parent_fsynced_after_rename"]),
        "atomic_no_clobber": bool(
            persisted["durability"]["atomic_no_clobber"]),
        "layout_questions_used": 0,
        "layout_answers_used": 0,
        "calibration_questions": 0,
        "future_turns_used_for_layout": 0,
        "layout_uses_dataset_question": False,
    }


def _gold_answers(turn: Mapping[str, Any]) -> list[str]:
    value = turn.get("answers")
    if isinstance(value, (list, tuple)) and value:
        return [str(item) for item in value]
    return [_gold_answer(turn)]


def _prompt_history_fields(runner, dialog: Mapping[str, Any],
                           turn_id: int) -> tuple[str, str, int]:
    prompt = gold_history_prompt(dialog, int(turn_id))
    history = prior_history_text(dialog, int(turn_id))
    history_ids = runner.processor.tokenizer(
        history, add_special_tokens=False).input_ids
    return prompt, history, len(history_ids)


def _make_row(*, workload: Mapping[str, Any], dialog: Mapping[str, Any],
              turn: Mapping[str, Any], method_key: str,
              order: Sequence[str], order_position: int, prompt: str,
              history: str, history_tokens: int, result: Mapping[str, Any],
              suffix_hash: str, combined_suffix_hash: str | None,
              input_hash: str | None, image_input_hash: str | None,
              physical_exists: bool, used_by_request: bool,
              manifest: Mapping[str, Any] | None,
              persistence_source: bool, execution_id: str,
              full_visual_bytes: int | None, dialogue_session_id: str,
              context_instance_id: str | None, suffix_tokens: int) -> dict:
    did = _dialog_id(dialog)
    tid = _turn_id(turn, 0)
    qid = _question_id(turn, did, tid)
    gold = _gold_answers(turn)
    method = METHODS[method_key]
    store_id = (str(manifest["meta_sha256"])
                if used_by_request and manifest is not None else None)
    physical_id = (str(manifest["meta_sha256"])
                   if physical_exists and manifest is not None else None)
    selected_ids = result.get("selected_chunk_ids_per_layer")
    selection_fingerprint = (
        _json_hash(selected_ids) if selected_ids is not None else None)
    row = {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET,
        "benchmark_type": workload["benchmark_type"],
        "official_benchmark_identity_claimed": False,
        "benchmark_disclaimer": workload["benchmark_disclaimer"],
        "dialogues_file_sha256": workload["dialogues_file_sha256"],
        "source_full_workload_sha256": workload[
            "source_full_workload_sha256"],
        "selected_workload_sha256": workload["selected_workload_sha256"],
        "model_revision": workload.get("model_revision"),
        "dialog_id": did,
        "global_dialog_ordinal": int(dialog["global_dialog_ordinal"]),
        "image_id": _image_id(dialog),
        "turn_id": tid,
        "question_id": qid,
        "question": _question(turn),
        "gold": gold,
        "method_key": method_key,
        "method": method["label"],
        "budget": method["budget"],
        "method_order": list(order),
        "method_order_position": int(order_position),
        "execution_id": execution_id,
        "dialogue_session_id": dialogue_session_id,
        "context_instance_id": context_instance_id,
        "gpu_request_cache_fresh": True,
        "text_kv_reused_from_prior_turn": False,
        "request_path": ("ssd_visual_prefix" if used_by_request
                         else "normal_multimodal_pixel"),
        "execution_mode": ("ssd_visual_prefix" if used_by_request
                           else "normal_multimodal_pixel"),
        "history_policy": "gold_teacher_forced",
        "history_tokens": int(history_tokens),
        "history_text_tokens": int(history_tokens),
        "suffix_tokens": int(suffix_tokens),
        "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
        "text_history_sha256": _sha256_bytes(history.encode("utf-8")),
        "question_sha256": _sha256_bytes(_question(turn).encode("utf-8")),
        "prediction": str(result["answer"]),
        "score": float(exact_score(str(result["answer"]), gold)),
        "quality_score": float(exact_score(str(result["answer"]), gold)),
        "quality_metric": "gqa_normalized_exact",
        "first_token_id": int(result["first_token_id"]),
        "generated_tokens": int(result["generated_tokens"]),
        "max_new_tokens": MAX_NEW_TOKENS,
        "suffix_ids_sha256": suffix_hash,
        "combined_recompute_suffix_sha256": combined_suffix_hash,
        "input_tensors_sha256": input_hash,
        "image_input_sha256": image_input_hash,
        "physical_store_exists_at_request_start": bool(physical_exists),
        "used_by_request": bool(used_by_request),
        "store_id": store_id,
        "same_physical_store_id": store_id,
        "physical_store_id_at_request_start": physical_id,
        "persistence_source_request": bool(persistence_source),
        "store_build_count_contribution": int(persistence_source),
        "physical_layout": ("visionzip_image_only" if manifest is not None
                            else None),
        "permutation_sha256": (manifest.get("permutation_sha256")
                               if manifest is not None else None),
        "layout_sha256": (manifest.get("layout_sha256")
                          if manifest is not None else None),
        "selection_fingerprint_sha256": selection_fingerprint,
        "layout_questions_used": 0,
        "layout_answers_used": 0,
        "calibration_questions": 0,
        "future_turns_used_for_layout": 0,
        "layout_uses_dataset_question": False,
        "layout_uses_generated_answer": False,
        "layout_uses_llm_qk": False,
        **_io_fields(result, method_key, full_visual_bytes),
    }
    for key in (
        "prompt_build_ms", "tokenization_ms", "image_preprocess_ms",
        "input_prepare_ms", "input_h2d_ms", "processor_total_ms",
        "pre_core_ms", "pre_core_phase_sum_ms", "pre_core_unattributed_ms",
        "core_ttft_ms", "end_to_end_ttft_ms", "ttft_ms", "decode_ms",
        "model_e2e_ms", "postprocess_ms", "request_e2e_ms", "e2e_ms",
        "request_started_at_s", "core_started_at_s", "first_token_at_s",
        "model_finished_at_s", "server_postprocess_finished_at_s",
        "postprocess_finished_at_s", "request_finished_at_s",
        "caller_returned_at_s", "caller_return_overhead_ms",
        "ttft_identity_error_ms", "model_e2e_identity_error_ms",
        "request_e2e_identity_error_ms", "vision_ms",
        "vision_forward_count", "separate_vision_forward_count",
        "page_cache_conditioning_started_at_s",
        "page_cache_conditioning_finished_at_s",
        "page_cache_conditioning_ms", "page_cache_conditioning_method",
        "page_cache_conditioning_excluded_from_ttft",
    ):
        if key in result:
            row[key] = result[key]
    if not used_by_request:
        # Turn-1 counterfactual labels describe the arm that will be used on
        # later turns; Turn 1 itself is a normal pixel request.  Do not report
        # a fictitious FullLoad=100% or Prefix budget when no store was read.
        row.update({
            "selected_visual_kv_bytes": 0,
            "selected_kv_ratio": None,
            "ssd_payload_ratio_vs_full_visual_kv": None,
            "actual_selected_normal_chunk_fraction": None,
            "selection_fingerprint_sha256": None,
        })
    row["gpu_memory_allocated"] = int(torch.cuda.memory_allocated())
    row["gpu_peak_memory_allocated"] = int(
        torch.cuda.max_memory_allocated())
    row["process_rss_bytes"] = int(psutil.Process().memory_info().rss)
    return row


def _validate_request_rows(rows: Sequence[Mapping[str, Any]], turn_id: int,
                           manifest: Mapping[str, Any] | None) -> None:
    if len(rows) != len(METHOD_KEYS):
        raise AssertionError("request must have exactly four method rows")
    methods = {str(row["method_key"]): row for row in rows}
    if set(methods) != set(METHOD_KEYS):
        raise AssertionError("request method coverage mismatch")
    for field in ("prompt_sha256", "text_history_sha256",
                  "suffix_ids_sha256"):
        if len({row[field] for row in rows}) != 1:
            raise AssertionError(f"method {field} mismatch")
    if len({tuple(row["method_order"]) for row in rows}) != 1:
        raise AssertionError("method order changed within request")
    if int(turn_id) == 1:
        if any(int(row["ssd_read_bytes"]) != 0 for row in rows):
            raise AssertionError("Turn 1 unexpectedly read the SSD store")
        if any(int(row["vision_forward_count"]) != 1 for row in rows):
            raise AssertionError("Turn 1 did not use four normal pixel paths")
        if len({int(row["first_token_id"]) for row in rows}) != 1:
            raise AssertionError("Turn 1 first tokens differ")
        if len({str(row["prediction"]) for row in rows}) != 1:
            raise AssertionError("Turn 1 predictions differ")
        if any(bool(row["used_by_request"]) for row in rows):
            raise AssertionError("Turn 1 used the SSD store")
        return
    if manifest is None:
        raise AssertionError("later turn has no physical store")
    store_id = str(manifest["meta_sha256"])
    recomp = methods["recompute"]
    if (int(recomp["ssd_read_bytes"]) != 0
            or int(recomp["vision_forward_count"]) != 1
            or recomp["used_by_request"]):
        raise AssertionError("later-turn ReComp contract failed")
    for method in ("fullload", "prefix25", "prefix45"):
        row = methods[method]
        if (not row["used_by_request"] or row["store_id"] != store_id
                or int(row["vision_forward_count"]) != 0):
            raise AssertionError(f"{method} did not reuse the one store")
    if int(methods["fullload"]["ssd_read_bytes"]) != int(
            manifest["visual_kv_bytes"]):
        raise AssertionError("FullLoad did not read full visual KV")
    for method in ("prefix25", "prefix45"):
        row = methods[method]
        validate_prefix_selection(row["selected_chunk_ids_per_layer"],
                                  int(row["n_chunks_total"]),
                                  float(row["budget"]))
        if any(int(row[key]) != 0 for key in (
                "static_score_calls", "query_score_calls",
                "diversity_calls")):
            raise AssertionError("Prefix request used an online selector")
    validate_nested_prefixes(methods["prefix25"], methods["prefix45"])


def _validate_image_rows(rows: Sequence[Mapping[str, Any]],
                         group: Mapping[str, Any], manifest: Mapping[str, Any],
                         persistence: Mapping[str, Any], seed: int) -> dict:
    expected = len(group["dialogs"]) * 3 * len(METHOD_KEYS)
    if len(rows) != expected:
        raise AssertionError(f"image row count {len(rows)} != {expected}")
    cursor = 0
    history_by_dialog: dict[str, list[int]] = defaultdict(list)
    all_session_ids: list[str] = []
    all_context_ids: list[str] = []
    for dialog in group["dialogs"]:
        order = list(method_order(int(dialog["global_dialog_ordinal"]), seed))
        dialog_session_ids = set()
        for fallback, turn in enumerate(dialog["turns"], 1):
            block = rows[cursor:cursor + len(METHOD_KEYS)]
            cursor += len(METHOD_KEYS)
            if [row["method_key"] for row in block] != order:
                raise AssertionError("execution order differs from global rotation")
            tid = _turn_id(turn, fallback)
            _validate_request_rows(block, tid, manifest)
            history_by_dialog[_dialog_id(dialog)].append(
                int(block[0]["history_tokens"]))
            dialog_session_ids.update(row["dialogue_session_id"] for row in block)
            if any(row["gpu_request_cache_fresh"] is not True
                   or row["text_kv_reused_from_prior_turn"] is not False
                   for row in block):
                raise AssertionError("dialogue request cache isolation failed")
        if len(dialog_session_ids) != 1:
            raise AssertionError("dialogue session ID changed within dialogue")
        all_session_ids.append(next(iter(dialog_session_ids)))
        context_ids = {
            str(row["context_instance_id"])
            for row in rows if row["dialog_id"] == _dialog_id(dialog)
            and row.get("context_instance_id") is not None
        }
        if len(context_ids) != 1:
            raise AssertionError(
                "physical ImageContext changed within one dialogue")
        all_context_ids.append(next(iter(context_ids)))
    if len(set(all_session_ids)) != len(group["dialogs"]):
        raise AssertionError("dialogue session ID was reused across dialogues")
    if len(set(all_context_ids)) != len(group["dialogs"]):
        raise AssertionError("ImageContext instance was reused across dialogues")
    for did, values in history_by_dialog.items():
        if any(a >= b for a, b in zip(values, values[1:])):
            raise AssertionError(
                f"history tokens do not strictly increase in {did}: {values}")
    cached = [row for row in rows if row["used_by_request"]]
    if {row["store_id"] for row in cached} != {manifest["meta_sha256"]}:
        raise AssertionError("cache requests did not share one physical store")
    if sum(int(row["persistence_source_request"]) for row in rows) != 1:
        raise AssertionError("store was not built exactly once")
    if int(persistence["store_build_count"]) != 1:
        raise AssertionError("persistence record has wrong build count")
    if any(int(row[key]) != 0 for row in rows for key in (
            "layout_questions_used", "layout_answers_used",
            "calibration_questions")):
        raise AssertionError("question/answer/calibration leakage detected")
    if any(float(row["end_to_end_ttft_ms"])
           >= float(row["request_e2e_ms"]) for row in rows):
        raise AssertionError("TTFT must be strictly below request E2E")
    if any(abs(float(row.get("request_e2e_identity_error_ms", 0.0)))
           > 1e-5 for row in rows):
        raise AssertionError("request E2E timing identity failed")
    if {row.get("permutation_sha256") for row in cached} != {
            manifest["permutation_sha256"]}:
        raise AssertionError("cache requests disagree on physical permutation")
    for method in ("prefix25", "prefix45"):
        fingerprints = {
            row.get("selection_fingerprint_sha256") for row in cached
            if row["method_key"] == method
        }
        if len(fingerprints) != 1 or None in fingerprints:
            raise AssertionError(
                f"{method} selection fingerprint changed across requests")
    return {
        "passed": True,
        "expected_rows": expected,
        "observed_rows": len(rows),
        "expected_dialogs": len(group["dialogs"]),
        "observed_dialogs": len({_dialog_id(d) for d in group["dialogs"]}),
        "store_build_count": 1,
        "store_reused_by_all_cache_requests": True,
        "all_method_prompts_and_histories_identical": True,
        "all_recompute_and_cache_suffixes_identical": True,
        "all_turn1_prediction_and_first_token_agree": True,
        "dialogue_sessions_independent": True,
        "context_instance_reopened_per_dialogue": True,
        "gpu_request_cache_fresh_every_request": True,
        "text_kv_reused_from_prior_turn": False,
        "all_prefix_requests_exact_first_k": True,
        "prefix25_subset_prefix45": True,
        "same_permutation_fingerprint_all_cache_requests": True,
        "same_selection_fingerprint_within_prefix_method": True,
        "calibration_questions": 0,
        "layout_questions_used": 0,
        "layout_answers_used": 0,
        "future_turn_leakage": False,
        "all_ttft_lt_e2e": True,
        "all_e2e_timing_identities_hold": True,
    }


def _execute_image_group(*, runner, server, group: Mapping[str, Any],
                         workload: Mapping[str, Any], run_dir: Path,
                         temp_root: Path, experiment_id: str,
                         shard_index: int, seed: int, cold: bool,
                         capacity_before_build: Mapping[str, Any],
                         reserve_bytes: int,
                         recovered_incomplete_temp_store: bool = False) -> dict:
    ImageContext, _ = _runtime_classes()
    image_id = str(group["image_id"])
    dialogs = list(group["dialogs"])
    if not dialogs:
        raise AssertionError("empty image group")
    image_store = temp_root / "payload" / image_id
    assert_owned_temp_path(image_store, temp_root, experiment_id, image_id)
    if os.path.lexists(image_store):
        # A store without its immutable artifact may represent a prior crash.
        # Preserve it rather than silently deleting evidence or overwriting a
        # no-clobber publication.
        raise FileExistsError(
            f"orphan temporary store requires inspection: {image_store}")

    image_path = resolve_image_path(dialogs[0])
    if any(resolve_image_path(dialog) != image_path for dialog in dialogs):
        raise AssertionError("same image ID maps to different image paths")
    image_file_sha = sha256_file(image_path)
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")

    rows: list[dict] = []
    manifest: dict | None = None
    layout_summary: dict | None = None
    persistence: dict | None = None
    persisted: dict | None = None
    ctx = None
    stored_prefix_ids: list[int] | None = None
    captured_prefix_ids: list[int] | None = None
    source_saliency_hashes: dict[str, str] = {}
    store_ready_at: float | None = None
    first_dialog_id = _dialog_id(dialogs[0])
    first_order = method_order(
        int(dialogs[0]["global_dialog_ordinal"]), seed)
    source_method = designated_source_method(first_order)
    store_build_count = 0

    try:
        for dialog_index_within_image, dialog in enumerate(dialogs):
            if _image_id(dialog) != image_id:
                raise AssertionError("image group contains another image")
            if dialog_index_within_image > 0:
                if ctx is None or manifest is None:
                    raise AssertionError("next dialogue started before store exists")
                # Dialogue text/generation state is never carried across
                # sessions.  Reopen only the immutable physical image store;
                # each Server request still creates a fresh GPU DynamicCache.
                ctx.close()
                ctx = ImageContext(
                    image_store, runner.model.device, drop_cache=True,
                    require_v_hidden=False)
                ctx.validate_prefix_layout("visionzip_image_only")
            dialogue_session_id = str(uuid.uuid4())
            context_instance_id = (str(uuid.uuid4())
                                   if ctx is not None else None)
            order = method_order(int(dialog["global_dialog_ordinal"]), seed)
            for fallback_turn, turn in enumerate(dialog["turns"], 1):
                tid = _turn_id(turn, fallback_turn)
                prompt, history, history_tokens = _prompt_history_fields(
                    runner, dialog, tid)
                method_rows: list[dict] = []
                t1_input_hashes: dict[str, str] = {}
                t1_predictions: dict[str, str] = {}
                t1_first_tokens: dict[str, int] = {}
                for order_position, method_key in enumerate(order):
                    torch.cuda.reset_peak_memory_stats()
                    execution_id = str(uuid.uuid4())
                    physical_exists = ctx is not None
                    used_by_request = tid >= 2 and method_key != "recompute"
                    persistence_source = (
                        dialog_index_within_image == 0 and tid == 1
                        and method_key == source_method)
                    if tid == 1 or method_key == "recompute":
                        capture_saliency = (
                            dialog_index_within_image == 0 and tid == 1
                            and method_key.startswith("prefix"))
                        result = _run_normal_request(
                            runner, server, image, dialog, tid, prompt,
                            capture_saliency=capture_saliency,
                            capture_cache=persistence_source)
                        enc_cpu = result.pop("enc_cpu")
                        capture = result.pop("capture")
                        request_prompt = result.pop("prompt")
                        if request_prompt != prompt:
                            raise AssertionError("normal request changed prompt")
                        input_hash = _hash_tensor_mapping(enc_cpu)
                        image_hash = _image_input_hash(enc_cpu)
                        combined_suffix = _combined_suffix(
                            runner, enc_cpu["input_ids"])
                        token_suffix = _tokenized_suffix(runner, prompt)
                        if not torch.equal(combined_suffix.cpu(),
                                           token_suffix.cpu()):
                            raise AssertionError(
                                "combined ReComp suffix != tokenizer cache suffix")
                        suffix_hash = _hash_tensor(token_suffix)
                        suffix_tokens = int(token_suffix.numel())
                        combined_suffix_hash = _hash_tensor(combined_suffix)
                        if combined_suffix_hash != suffix_hash:
                            raise AssertionError("suffix tensor hashes disagree")
                        cache = result.pop("captured_past_key_values", None)
                        if tid == 1:
                            t1_input_hashes[method_key] = input_hash
                            t1_predictions[method_key] = str(result["answer"])
                            t1_first_tokens[method_key] = int(
                                result["first_token_id"])
                        if capture_saliency:
                            source_saliency_hashes[method_key] = _hash_tensor(
                                capture.result_cpu())
                        if persistence_source:
                            if cache is None or store_build_count != 0:
                                raise AssertionError(
                                    "source request did not return one cache")
                            persist_started = time.perf_counter()
                            persisted = persist_captured_visual_prefix(
                                runner, cache, enc_cpu["input_ids"],
                                enc_cpu["image_sizes"][0],
                                capture.result_cpu(), image_store,
                                image_id=image_id, model_id=runner.model_id,
                                chunk_size=CHUNK_SIZE,
                                image_input_sha256=image_hash,
                                capture_stats=capture,
                                extra_metadata={
                                    "dataset": DATASET,
                                    "source_dialog_id": _dialog_id(dialog),
                                    "source_global_dialog_ordinal": int(
                                        dialog["global_dialog_ordinal"]),
                                    "source_turn_id": 1,
                                    "source_execution_id": execution_id,
                                    "history_policy": "gold_teacher_forced",
                                    "future_turns_used_for_layout": 0,
                                    "layout_questions_used": 0,
                                    "layout_answers_used": 0,
                                    "persistence_schedule":
                                        "immediately_after_source_answer1",
                                })
                            store_build_count += 1
                            cache = None
                            torch.cuda.synchronize()
                            context_started = time.perf_counter()
                            ctx = ImageContext(
                                image_store, runner.model.device,
                                drop_cache=True, require_v_hidden=False)
                            ctx.validate_prefix_layout("visionzip_image_only")
                            context_instance_id = str(uuid.uuid4())
                            context_open_ms = (
                                time.perf_counter() - context_started) * 1e3
                            store_ready_at = time.perf_counter()
                            meta = ctx.meta
                            manifest = _store_manifest(
                                image_store, meta, persisted)
                            layout_summary = _layout_summary(image_store, meta)
                            stored_prefix_ids = [int(v) for v in
                                                 meta["prefix_input_ids"]]
                            captured_prefix_ids = [int(v) for v in
                                enc_cpu["input_ids"][0, :int(meta["prefix_len"])]
                                .detach().cpu().tolist()]
                            if captured_prefix_ids != stored_prefix_ids:
                                raise AssertionError(
                                    "stored prefix IDs differ from source forward")
                            persistence = _persistence_record(
                                dialog, source_method, execution_id, capture,
                                persisted, persist_started, store_ready_at,
                                context_open_ms)
                        elif cache is not None:
                            raise AssertionError(
                                "non-source normal request retained a cache")
                    else:
                        if ctx is None or manifest is None:
                            raise AssertionError("cache request preceded persistence")
                        result = _run_stored_request(
                            runner, server, ctx, dialog, tid, prompt, method_key,
                            budget=float(METHODS[method_key]["budget"]),
                            cold=cold, seed=seed, image_id=image_id)
                        suffix_cpu = result.pop("suffix_cpu")
                        request_prompt = result.pop("prompt")
                        if request_prompt != prompt:
                            raise AssertionError("stored request changed prompt")
                        suffix_hash = _hash_tensor(suffix_cpu)
                        suffix_tokens = int(suffix_cpu.numel())
                        combined_suffix_hash = None
                        input_hash = None
                        image_hash = None
                        capture = None

                    row = _make_row(
                        workload=workload, dialog=dialog, turn=turn,
                        method_key=method_key, order=order,
                        order_position=order_position, prompt=prompt,
                        history=history, history_tokens=history_tokens,
                        result=result, suffix_hash=suffix_hash,
                        combined_suffix_hash=combined_suffix_hash,
                        input_hash=input_hash, image_input_hash=image_hash,
                        physical_exists=physical_exists,
                        used_by_request=used_by_request, manifest=manifest,
                        persistence_source=persistence_source,
                        execution_id=execution_id,
                        full_visual_bytes=(manifest["visual_kv_bytes"]
                                           if manifest is not None else None),
                        dialogue_session_id=dialogue_session_id,
                        context_instance_id=context_instance_id,
                        suffix_tokens=suffix_tokens)
                    method_rows.append(row)
                    if capture is not None:
                        del capture
                    gc.collect()

                if tid == 1:
                    if (len(set(t1_input_hashes.values())) != 1
                            or len(set(t1_predictions.values())) != 1
                            or len(set(t1_first_tokens.values())) != 1):
                        raise AssertionError(
                            f"Turn 1 normal arms disagree: {_dialog_id(dialog)}")
                    if dialog_index_within_image == 0:
                        if (set(source_saliency_hashes)
                                != {"prefix25", "prefix45"}
                                or len(set(source_saliency_hashes.values())) != 1):
                            raise AssertionError(
                                "first-dialog Prefix saliency captures disagree")
                _validate_request_rows(method_rows, tid, manifest)
                rows.extend(method_rows)

        if (ctx is None or manifest is None or layout_summary is None
                or persistence is None or persisted is None
                or stored_prefix_ids is None or captured_prefix_ids is None):
            raise AssertionError("image completed without a durable store")
        if store_build_count != 1:
            raise AssertionError(f"store build count is {store_build_count}")
        # Normal rows that executed after persistence carry the same immutable
        # layout fingerprints even though they did not consume that store.
        for row in rows:
            if row["physical_store_exists_at_request_start"]:
                row["physical_store_id_at_request_start"] = manifest[
                    "meta_sha256"]
                row["permutation_sha256"] = manifest["permutation_sha256"]
                row["layout_sha256"] = manifest["layout_sha256"]
            row["stored_prefix_ids_sha256"] = _json_hash(stored_prefix_ids)
            row["captured_prefix_ids_sha256"] = _json_hash(captured_prefix_ids)
            row["stored_prefix_ids_match_source"] = True
            row["full_visual_kv_bytes"] = int(manifest["visual_kv_bytes"])
            row["visual_tokens"] = int(manifest["v_token_num"])
            row["active_visual_tokens"] = int(manifest["v_token_num"])
            row["total_context_tokens"] = int(
                manifest["prefix_len"] + row["suffix_tokens"])

        validation = _validate_image_rows(
            rows, group, manifest, persistence, seed)
        validation.update({
            "stored_prefix_ids_match_source": True,
            "stored_prefix_ids_sha256": _json_hash(stored_prefix_ids),
            "captured_prefix_ids_sha256": _json_hash(captured_prefix_ids),
            "first_dialog_id": first_dialog_id,
            "designated_source_method": source_method,
            "source_saliency_fingerprints_identical": True,
            "source_saliency_sha256": next(iter(
                source_saliency_hashes.values())),
        })
        capacity_after_build = _capacity_guard(
            temp_root, reserve_bytes=int(reserve_bytes))
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": str(experiment_id),
            "dataset": DATASET,
            "benchmark_type": workload["benchmark_type"],
            "official_benchmark_identity_claimed": False,
            "benchmark_disclaimer": workload["benchmark_disclaimer"],
            "image_id": image_id,
            "image_ordinal": int(group["image_ordinal"]),
            "shard_index": int(shard_index),
            "dialogues_file_sha256": workload["dialogues_file_sha256"],
            "source_full_workload_sha256": workload[
                "source_full_workload_sha256"],
            "selected_workload_sha256": workload[
                "selected_workload_sha256"],
            "model_revision": workload.get("model_revision"),
            "source_image_path": str(image_path),
            "source_image_sha256": image_file_sha,
            "dialog_ids": [_dialog_id(dialog) for dialog in dialogs],
            "global_dialog_ordinals": [
                int(dialog["global_dialog_ordinal"]) for dialog in dialogs],
            "n_dialogs": len(dialogs),
            "n_turns": len(dialogs) * 3,
            "n_rows": len(rows),
            "history_policy": "gold_teacher_forced",
            "persistence_overhead": persistence,
            "store_manifest": manifest,
            "layout_artifact": layout_summary,
            "stored_prefix_ids": stored_prefix_ids,
            "stored_prefix_ids_sha256": _json_hash(stored_prefix_ids),
            "captured_prefix_ids_sha256": _json_hash(captured_prefix_ids),
            "store_ready_at_s": store_ready_at,
            "capacity_before_build": dict(capacity_before_build),
            "capacity_after_build": capacity_after_build,
            "recovered_incomplete_temp_store_before_build": bool(
                recovered_incomplete_temp_store),
            "rows": rows,
            "validation": validation,
        }
        artifact["artifact_content_sha256"] = _artifact_body_hash(artifact)
        return artifact
    finally:
        if ctx is not None:
            ctx.close()
        image.close()


def _model_revision() -> str | None:
    ref = (Path.home() / ".cache/huggingface/hub/"
           "models--llava-hf--llava-v1.6-vicuna-7b-hf/refs/main")
    return ref.read_text().strip() if ref.is_file() else None


def _base_config(args, workload: Mapping[str, Any], groups: Sequence[Mapping],
                 n_shards: int, warmup: Mapping[str, Any], runner) -> dict:
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(args.temp_root)
    gpu = torch.cuda.get_device_properties(0)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": str(args.experiment_id),
        "dataset": DATASET,
        "benchmark_type": workload["benchmark_type"],
        "official_benchmark_identity_claimed": False,
        "benchmark_disclaimer": workload["benchmark_disclaimer"],
        "index": str(workload["path"]),
        "dialogues_file_sha256": workload["dialogues_file_sha256"],
        "source_full_workload_sha256": workload[
            "source_full_workload_sha256"],
        "selected_workload_sha256": workload["selected_workload_sha256"],
        "top_artifact_content_sha256": workload.get(
            "artifact_content_sha256"),
        "source_full_n_dialogs": int(workload["source_full_n_dialogs"]),
        "partial_workload": bool(workload["partial_workload"]),
        "n_dialogs": int(workload["n_dialogs"]),
        "n_turns": int(workload["n_turns"]),
        "n_images": len(groups),
        "n_requests": int(workload["n_turns"]) * len(METHOD_KEYS),
        "shard_size": int(args.shard_size),
        "n_shards": int(n_shards),
        "seed": int(args.seed),
        "method_keys": list(METHOD_KEYS),
        "methods": METHODS,
        "method_order_policy": (
            "zero-based cyclic rotation by full-workload global dialogue "
            "ordinal (D1 starts ReComp); identical order across T1-T3; seed "
            "does not phase-shift method order"),
        "history_policy": "gold_teacher_forced",
        "model": runner.model_id,
        "model_revision": _model_revision(),
        "load_4bit": bool(runner.load_4bit),
        "quantization": "4-bit NF4 double-quant",
        "attention": runner.attn,
        "decoding": "greedy",
        "max_new_tokens": int(args.max_new_tokens),
        "chunk_size": CHUNK_SIZE,
        "physical_layout": "visionzip_image_only",
        "retrieval": "sequential physical first-k chunks",
        "separator_policy": "stable_tail_plus_sidecar",
        "first_dialog_turn1_policy": (
            "all four normal pixel paths; last Prefix arm in balanced order "
            "captures same-forward saliency+past and persists immediately"),
        "other_turn1_policy": "all four normal pixel paths; zero SSD reads",
        "later_turn_policy": (
            "ReComp pixel+vision; three cache arms use one physical SSD store"),
        "layout_questions_used": 0,
        "layout_answers_used": 0,
        "calibration_questions": 0,
        "future_turns_used_for_layout": 0,
        "layout_uses_dataset_question": False,
        "ssd_read_api": "buffered os.pread",
        "o_direct": False,
        "ssd_controller_cache_flushed": False,
        "cache_condition": "OS-page-cache-cold",
        "page_cache_conditioning": "posix_fadvise(DONTNEED)",
        "page_cache_conditioning_inside_ttft": False,
        "main_ttft_field": "end_to_end_ttft_ms",
        "ttft_definition": (
            "after page-cache conditioning, before prompt construction -> "
            "tokenization/processor -> H2D -> ReComp vision or SSD pread/"
            "scatter -> prefill -> greedy first token -> CUDA synchronize"),
        "jpeg_file_read_and_decode_timed": False,
        "persistence_in_main_ttft": False,
        "image_at_a_time_temporary_store": True,
        "temporary_payload_deleted_only_after_immutable_image_artifact": True,
        "unmeasured_warmup": dict(warmup),
        "run_dir": str(args.run_dir),
        "temp_root": str(args.temp_root),
        "capacity_guard": {
            "min_free_after_gib": float(args.min_free_after_gib),
            "build_headroom_gib": BUILD_HEADROOM_GIB,
            "max_used_percent_exclusive": 96.0,
        },
        "machine": {
            "hostname": platform.node(),
            "gpu_name": gpu.name,
            "gpu_total_memory_bytes": int(gpu.total_memory),
            "system_ram_total_bytes": int(vm.total),
            "system_ram_available_bytes_at_start": int(vm.available),
            "disk_total_bytes": int(disk.total),
            "disk_free_bytes_at_start": int(disk.free),
            "cpu_count": os.cpu_count(),
            "torch_version": torch.__version__,
        },
        "run_started_at_unix": time.time(),
    }


def _validate_existing_config(run_dir: Path, args,
                              workload: Mapping[str, Any],
                              groups: Sequence[Mapping], n_shards: int) -> dict:
    path = run_dir / "config.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError("completed/resumed run has no regular config.json")
    config = json.loads(path.read_text())
    expected = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": str(args.experiment_id),
        "dataset": DATASET,
        "benchmark_type": workload["benchmark_type"],
        "dialogues_file_sha256": workload["dialogues_file_sha256"],
        "source_full_workload_sha256": workload[
            "source_full_workload_sha256"],
        "selected_workload_sha256": workload["selected_workload_sha256"],
        "n_dialogs": int(workload["n_dialogs"]),
        "n_images": len(groups),
        "seed": int(args.seed),
        "method_keys": list(METHOD_KEYS),
        "shard_size": int(args.shard_size),
        "n_shards": int(n_shards),
        "max_new_tokens": MAX_NEW_TOKENS,
        "model_revision": _model_revision(),
    }
    mismatch = {key: (config.get(key), value) for key, value in expected.items()
                if config.get(key) != value}
    if mismatch:
        raise ValueError(f"existing run config mismatch: {mismatch}")
    return config


def _validate_shard_marker(path: Path, *, args,
                           workload: Mapping[str, Any],
                           shard: Mapping[str, Any], run_dir: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"shard marker is not regular: {path}")
    marker = json.loads(path.read_text())
    groups = list(shard["groups"])
    image_ids = [str(group["image_id"]) for group in groups]
    expected = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": str(args.experiment_id),
        "dataset": DATASET,
        "benchmark_type": workload["benchmark_type"],
        "dialogues_file_sha256": workload["dialogues_file_sha256"],
        "source_full_workload_sha256": workload[
            "source_full_workload_sha256"],
        "selected_workload_sha256": workload["selected_workload_sha256"],
        "shard_index": int(args.shard_index),
        "shard_size": int(args.shard_size),
        "image_start": int(shard["start"]),
        "image_stop": int(shard["stop"]),
        "image_ids": image_ids,
        "completed_image_ids": image_ids,
        "complete": True,
    }
    mismatch = {key: (marker.get(key), value) for key, value in expected.items()
                if marker.get(key) != value}
    if mismatch:
        raise ValueError(f"completed shard marker mismatch: {mismatch}")
    recorded_content = marker.get("artifact_content_sha256")
    if recorded_content != _artifact_body_hash(marker):
        raise ValueError("shard marker content hash mismatch")
    file_hashes = marker.get("image_artifact_file_sha256")
    content_hashes = marker.get("image_artifact_content_sha256")
    if (not isinstance(file_hashes, Mapping)
            or set(file_hashes) != set(image_ids)
            or not isinstance(content_hashes, Mapping)
            or set(content_hashes) != set(image_ids)):
        raise ValueError("shard marker image hash coverage mismatch")
    for group in groups:
        image_id = str(group["image_id"])
        artifact_path = _image_artifact_path(run_dir, image_id)
        artifact = validate_resume_artifact(
            artifact_path, experiment_id=args.experiment_id,
            image_group=group, shard_index=args.shard_index,
            workload=workload, seed=args.seed)
        if sha256_file(artifact_path) != file_hashes[image_id]:
            raise ValueError("published image file hash changed")
        if artifact["artifact_content_sha256"] != content_hashes[image_id]:
            raise ValueError("published image content hash changed")
    return marker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--expected-index-sha256", required=True,
                        help="SHA256 of the complete dialogues JSON file")
    parser.add_argument("--expected-workload-sha256", required=True,
                        help="ordered dialog/turn/question request-key SHA256")
    parser.add_argument("--expected-dialogs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-new-tokens", type=int,
                        default=MAX_NEW_TOKENS)
    parser.add_argument("--min-free-after-gib", type=float,
                        default=MIN_FREE_AFTER_GIB)
    parser.add_argument("--max-dialogs", type=int, choices=(10, 100))
    parser.add_argument("--allow-partial-workload", action="store_true")
    return parser


def _validate_cli(args) -> None:
    if not args.run_dir.is_absolute() or not args.temp_root.is_absolute():
        raise ValueError("run-dir and temp-root must be absolute")
    if ((args.run_dir.exists() and args.run_dir.is_symlink())
            or (args.temp_root.exists() and args.temp_root.is_symlink())):
        raise ValueError("run-dir and temp-root may not be symlinks")
    run_root, temp_root = args.run_dir.resolve(), args.temp_root.resolve()
    if (run_root == temp_root or run_root in temp_root.parents
            or temp_root in run_root.parents):
        raise ValueError("run-dir and temp-root may not overlap")
    if not MIN_SHARD_SIZE <= int(args.shard_size) <= MAX_SHARD_SIZE:
        raise ValueError("shard-size must be in [40, 60]")
    if int(args.shard_index) < 0:
        raise ValueError("shard-index must be nonnegative")
    if int(args.seed) != SEED:
        raise ValueError("the frozen MT-GQA experiment uses seed 1234")
    if int(args.max_new_tokens) != MAX_NEW_TOKENS:
        raise ValueError("the frozen MT-GQA system experiment uses 16 tokens")
    if float(args.min_free_after_gib) < MIN_FREE_AFTER_GIB:
        raise ValueError("min-free-after-gib may not be below 30 GiB")
    if (args.max_dialogs is None) != (not args.allow_partial_workload):
        raise ValueError(
            "--max-dialogs and --allow-partial-workload must be used together")


def main() -> None:
    args = build_parser().parse_args()
    _validate_cli(args)
    mt_helpers = _load_mt_helpers()
    source_expected_dialogs = (
        int(getattr(mt_helpers, "EXPECTED_DIALOGUES", args.expected_dialogs))
        if args.max_dialogs is not None else int(args.expected_dialogs))
    full = resolve_dialogues(
        args.index, expected_dialogs=source_expected_dialogs,
        expected_dialogues_sha256=args.expected_index_sha256,
        expected_workload_sha256=args.expected_workload_sha256)
    workload = select_dialogues(full, args.max_dialogs)
    if int(workload["n_dialogs"]) != int(args.expected_dialogs):
        raise ValueError(
            f"selected dialog count mismatch: expected {args.expected_dialogs}, "
            f"got {workload['n_dialogs']}")
    model_revision = _model_revision()
    if not isinstance(model_revision, str) or not model_revision:
        raise ValueError("local LLaVA-NeXT checkpoint revision is unavailable")
    workload["model_revision"] = model_revision
    groups = group_dialogues_by_image(workload["dialogs"])
    shard = shard_image_groups(groups, args.shard_index, args.shard_size)

    config_path = args.run_dir / "config.json"
    if config_path.exists():
        _validate_existing_config(
            args.run_dir, args, workload, groups, shard["n_shards"])
    elif args.run_dir.exists():
        if args.run_dir.is_symlink() or not args.run_dir.is_dir():
            raise ValueError("existing run-dir is not a regular directory")
        if any(args.run_dir.iterdir()):
            raise ValueError(
                "nonempty output root has no config.json; refusing to mix runs")

    _claim_temp_root(args.temp_root, args.experiment_id, DATASET)
    shard_marker = (args.run_dir / "shards" /
                    f"shard_{args.shard_index:03d}.json")
    if shard_marker.is_file():
        _validate_shard_marker(
            shard_marker, args=args, workload=workload, shard=shard,
            run_dir=args.run_dir)
        print(f"shard {args.shard_index} already complete; nothing to do")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    runner = LlavaRunner().load()
    _, Server = _runtime_classes()
    server = Server(runner, max_new_tokens=MAX_NEW_TOKENS)
    warmup = _load_visdial_helpers()._run_unmeasured_warmup(runner, server)
    config = _base_config(
        args, workload, groups, shard["n_shards"], warmup, runner)
    _ensure_run_config(args.run_dir, config)

    reserve_bytes = int(float(args.min_free_after_gib) * (1024 ** 3))
    headroom_bytes = int(BUILD_HEADROOM_GIB * (1024 ** 3))
    completed: list[str] = []
    started_at = time.time()
    for local_index, group in enumerate(shard["groups"]):
        image_id = str(group["image_id"])
        artifact_path = _image_artifact_path(args.run_dir, image_id)
        image_store = args.temp_root / "payload" / image_id
        assert_owned_temp_path(
            image_store, args.temp_root, args.experiment_id, image_id)
        if artifact_path.is_file():
            validate_resume_artifact(
                artifact_path, experiment_id=args.experiment_id,
                image_group=group, shard_index=args.shard_index,
                workload=workload, seed=args.seed)
            if os.path.lexists(image_store):
                remove_owned_temp_store(
                    image_store, args.temp_root, args.experiment_id, image_id)
            completed.append(image_id)
            print(f"[{local_index + 1}/{len(shard['groups'])}] "
                  f"{image_id}: immutable artifact exists", flush=True)
            continue
        if os.path.lexists(image_store):
            remove_owned_temp_store(
                image_store, args.temp_root, args.experiment_id, image_id)
            recovered_incomplete_temp_store = True
        else:
            recovered_incomplete_temp_store = False
        capacity_before = _capacity_guard(
            args.temp_root, reserve_bytes=reserve_bytes,
            extra_headroom_bytes=headroom_bytes)
        artifact = _execute_image_group(
            runner=runner, server=server, group=group, workload=workload,
            run_dir=args.run_dir, temp_root=args.temp_root,
            experiment_id=args.experiment_id, shard_index=args.shard_index,
            seed=args.seed, cold=True,
            capacity_before_build=capacity_before,
            reserve_bytes=reserve_bytes,
            recovered_incomplete_temp_store=recovered_incomplete_temp_store)
        _write_exclusive_json(artifact_path, artifact)
        # Publication is durable and content hashed before the only recursive
        # deletion in this runner is allowed.
        remove_owned_temp_store(
            image_store, args.temp_root, args.experiment_id, image_id)
        completed.append(image_id)
        print(
            f"[{local_index + 1}/{len(shard['groups'])}] {image_id}: "
            f"{artifact['n_dialogs']} dialogs, {artifact['n_rows']} rows; "
            f"store {artifact['store_manifest']['total_store_bytes']/1e9:.2f} GB",
            flush=True)
        torch.cuda.empty_cache()
        gc.collect()

    image_ids = [str(group["image_id"]) for group in shard["groups"]]
    if completed != image_ids:
        raise AssertionError("shard did not complete images in canonical order")
    marker = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": str(args.experiment_id),
        "dataset": DATASET,
        "benchmark_type": workload["benchmark_type"],
        "dialogues_file_sha256": workload["dialogues_file_sha256"],
        "source_full_workload_sha256": workload[
            "source_full_workload_sha256"],
        "selected_workload_sha256": workload["selected_workload_sha256"],
        "shard_index": int(args.shard_index),
        "shard_size": int(args.shard_size),
        "image_start": int(shard["start"]),
        "image_stop": int(shard["stop"]),
        "image_ids": image_ids,
        "completed_image_ids": completed,
        "complete": True,
        "elapsed_seconds": float(time.time() - started_at),
        "image_artifact_file_sha256": {
            image_id: sha256_file(_image_artifact_path(args.run_dir, image_id))
            for image_id in image_ids
        },
        "image_artifact_content_sha256": {
            image_id: json.loads(_image_artifact_path(
                args.run_dir, image_id).read_text())["artifact_content_sha256"]
            for image_id in image_ids
        },
    }
    marker["artifact_content_sha256"] = _artifact_body_hash(marker)
    _write_exclusive_json(shard_marker, marker)
    print(f"completed shard {args.shard_index}: {len(completed)} images")


if __name__ == "__main__":
    main()

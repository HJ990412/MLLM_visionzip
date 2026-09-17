"""Image-at-a-time ImageOnly-Repack generalization runner.

This runner deliberately does not call the calibration/reorder/static pipeline.
For each image it directly builds one ``visionzip_image_only`` store, evaluates
the frozen questions with ReComp, same-layout FullLoad, Prefix25 and Prefix45,
publishes one immutable result artifact, and removes only that experiment-owned
temporary KV payload.  Independent shard invocations can therefore resume by
skipping already-published image artifacts without retaining hundreds of GB of
Visual KV.

The paper metric is ``end_to_end_ttft_ms``.  Its caller-side timestamp starts
before prompt construction, tokenization/processor work and initial H2D.  The
absolute first-token timestamp returned by :class:`mmimpress.serve.Server`
already follows the greedy decision with ``torch.cuda.synchronize()``.  Cache
conditioning is completed before the caller-side request timestamp.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
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
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import psutil
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mmimpress.config import CHUNK_SIZE, MODEL_ID  # noqa: E402
from mmimpress.cvpr25 import (budget_chunk_count, prefix_chunk_ids,
                              permutation_sha256)  # noqa: E402
from mmimpress.dataset import METRICS, load_index, question_answers  # noqa: E402
from mmimpress.model import LlavaRunner  # noqa: E402
from mmimpress.piggyback import sha256_file, stable_json_sha256  # noqa: E402
from mmimpress.serve import ImageContext, Server  # noqa: E402


SCHEMA_VERSION = "image-only-generalization-e2e-ttft-v1"
METHOD_KEYS = ("recompute", "fullload", "prefix25", "prefix45")
METHODS = {
    "recompute": {"label": "ReComp", "budget": None},
    "fullload": {"label": "ImageOnly FullLoad", "budget": 1.0},
    "prefix25": {"label": "ImageOnly Repack + Prefix25", "budget": 0.25},
    "prefix45": {"label": "ImageOnly Repack + Prefix45", "budget": 0.45},
}
BOUNDARY_QUESTION_ID = "__image_only_fixed_synthetic_boundary_v1__"
BOUNDARY_TEXT = (
    "This is a fixed non-dataset cache-construction boundary. "
    "Describe the image briefly."
)
WARMUP_PROMPT = (
    "USER: <image>\nThis is a synthetic unmeasured serving warm-up. "
    "Describe the image briefly. ASSISTANT:"
)
OWNER_FILE = ".image_only_generalization_owner.json"


def _load_build_module():
    path = Path(__file__).resolve().parent / "01_build_store.py"
    spec = importlib.util.spec_from_file_location(
        "image_only_generalization_build_store", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load direct builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BUILD = _load_build_module()
build_one = _BUILD.build_one


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_hash(value: Any) -> str:
    return stable_json_sha256(value)


def _workload_hash(entries: Sequence[Mapping[str, Any]], skip: int,
                   questions: int) -> str:
    payload = "\n".join(
        f"{entry['image_id']}\t{q['question_id']}"
        for entry in entries
        for q in entry["questions"][skip:skip + questions]
    ).encode("utf-8")
    return _sha256_bytes(payload)


def resolve_workload(index_path: Path | str, *, skip: int, questions: int,
                     expected_index_sha256: str | None = None,
                     expected_workload_sha256: str | None = None,
                     expected_images: int | None = None,
                     expected_questions: int | None = None) -> dict[str, Any]:
    """Resolve and fail-close the full canonical workload before sharding.

    The global request ordinal stored here is later used for method rotation;
    consequently shard size, shard order and resume do not alter method order.
    """
    path = Path(index_path).resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"index must be a regular file: {path}")
    if skip < 0 or questions < 1:
        raise ValueError("skip must be nonnegative and questions must be positive")
    entries = load_index(path)
    if not isinstance(entries, list) or not entries:
        raise ValueError("index is empty or not a list")
    image_ids = [str(entry["image_id"]) for entry in entries]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("image IDs are not unique")

    all_question_ids: set[str] = set()
    all_question_texts: set[str] = set()
    request_rows = []
    per_image_counts = []
    ordinal = 0
    for image_ordinal, entry in enumerate(entries):
        all_questions = entry.get("questions")
        if not isinstance(all_questions, list):
            raise ValueError(f"questions missing for image {entry['image_id']}")
        for question in all_questions:
            qid = str(question["question_id"])
            if qid in all_question_ids:
                raise ValueError(f"duplicate question ID: {qid}")
            all_question_ids.add(qid)
            all_question_texts.add(str(question["question"]))
        selected = all_questions[skip:skip + questions]
        if len(selected) != min(questions, max(0, len(all_questions) - skip)):
            raise AssertionError("unexpected Python slice behavior")
        per_image_counts.append(len(selected))
        for local_question_ordinal, question in enumerate(selected):
            request_rows.append({
                "global_request_ordinal": ordinal,
                "image_ordinal": image_ordinal,
                "local_question_ordinal": local_question_ordinal,
                "image_id": str(entry["image_id"]),
                "question_id": str(question["question_id"]),
            })
            ordinal += 1

    index_sha = sha256_file(path)
    workload_sha = _workload_hash(entries, skip, questions)
    observed_images = len(entries)
    observed_questions = len(request_rows)
    checks = (
        (expected_index_sha256, index_sha, "index SHA256"),
        (expected_workload_sha256, workload_sha, "workload SHA256"),
        (expected_images, observed_images, "image count"),
        (expected_questions, observed_questions, "question count"),
    )
    for expected, observed, label in checks:
        if expected is not None and expected != observed:
            raise ValueError(f"{label} mismatch: expected {expected}, got {observed}")
    if BOUNDARY_QUESTION_ID in all_question_ids:
        raise ValueError("synthetic boundary question ID collides with the index")
    if BOUNDARY_TEXT in all_question_texts:
        raise ValueError("synthetic boundary text collides with the index")

    return {
        "path": path,
        "entries": entries,
        "request_rows": request_rows,
        "ordinal_by_key": {
            (row["image_id"], row["question_id"]):
                int(row["global_request_ordinal"])
            for row in request_rows
        },
        "index_sha256": index_sha,
        "workload_sha256": workload_sha,
        "n_images": observed_images,
        "n_questions": observed_questions,
        "questions_per_image": {
            "min": min(per_image_counts),
            "mean": float(np.mean(per_image_counts)),
            "max": max(per_image_counts),
        },
        "skip": int(skip),
        "questions": int(questions),
        "boundary_question_id": BOUNDARY_QUESTION_ID,
        "boundary_text": BOUNDARY_TEXT,
        "boundary_text_sha256": _sha256_bytes(BOUNDARY_TEXT.encode("utf-8")),
    }


def _partial_workload(full: Mapping[str, Any], *, max_images: int | None,
                      max_questions_per_image: int | None) -> dict[str, Any]:
    """Derive a deterministic smoke view while retaining full-workload proof."""
    image_count = (int(max_images) if max_images is not None
                   else int(full["n_images"]))
    question_count = (int(max_questions_per_image)
                      if max_questions_per_image is not None
                      else int(full["questions"]))
    if not 1 <= image_count <= int(full["n_images"]):
        raise ValueError("max-images is outside the frozen workload")
    if not 1 <= question_count <= int(full["questions"]):
        raise ValueError("max-questions-per-image is outside the frozen slice")
    entries = list(full["entries"][:image_count])
    selected_counts = [len(entry["questions"][
        int(full["skip"]):int(full["skip"]) + question_count])
        for entry in entries]
    request_rows = []
    ordinal_by_key = {}
    # Preserve the full canonical ordinal for ordering, even in a smoke shard.
    full_ordinals = full["ordinal_by_key"]
    for image_ordinal, entry in enumerate(entries):
        selected = entry["questions"][
            int(full["skip"]):int(full["skip"]) + question_count]
        for local_ordinal, question in enumerate(selected):
            key = (str(entry["image_id"]), str(question["question_id"]))
            ordinal = int(full_ordinals[key])
            request_rows.append({
                "global_request_ordinal": ordinal,
                "image_ordinal": image_ordinal,
                "local_question_ordinal": local_ordinal,
                "image_id": key[0],
                "question_id": key[1],
            })
            ordinal_by_key[key] = ordinal
    out = dict(full)
    out.update({
        "entries": entries,
        "request_rows": request_rows,
        "ordinal_by_key": ordinal_by_key,
        "source_full_workload_sha256": full["workload_sha256"],
        "source_full_n_images": int(full["n_images"]),
        "source_full_n_questions": int(full["n_questions"]),
        "workload_sha256": _workload_hash(
            entries, int(full["skip"]), question_count),
        "n_images": len(entries),
        "n_questions": len(request_rows),
        "questions": question_count,
        "questions_per_image": {
            "min": min(selected_counts),
            "mean": float(np.mean(selected_counts)),
            "max": max(selected_counts),
        },
        "partial_workload": (
            image_count != int(full["n_images"])
            or question_count != int(full["questions"])),
    })
    return out


def balanced_method_order(global_request_ordinal: int, seed: int = 1234,
                          methods: Sequence[str] = METHOD_KEYS) -> tuple[str, ...]:
    """Cyclic balanced ordering invariant to sharding and resume."""
    methods = tuple(methods)
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("methods must be a nonempty unique sequence")
    if any(method not in METHODS for method in methods):
        raise ValueError(f"unknown method in {methods}")
    offset = (int(global_request_ordinal) + int(seed)) % len(methods)
    return methods[offset:] + methods[:offset]


def _read_owner(root: Path) -> dict[str, Any]:
    marker = root / OWNER_FILE
    if not marker.is_file() or marker.is_symlink():
        raise ValueError(f"missing regular ownership marker: {marker}")
    return json.loads(marker.read_text())


def assert_owned_temp_path(path: Path | str, root: Path | str,
                           experiment_id: str,
                           expected_image_id: str | None = None) -> Path:
    """Validate exactly ``root/payload/<one image leaf>`` for safe removal."""
    root = Path(root)
    candidate = Path(path)
    if not root.is_absolute() or not candidate.is_absolute():
        raise ValueError("owned temporary paths must be absolute")
    if root.is_symlink() or candidate.is_symlink():
        raise ValueError("owned temporary paths may not be symlinks")
    root_resolved = root.resolve()
    candidate_resolved = candidate.resolve(strict=False)
    payload = root_resolved / "payload"
    if payload.is_symlink() or candidate_resolved.parent != payload:
        raise ValueError(
            f"temporary target is not one direct image leaf: {candidate}")
    if candidate_resolved.name in ("", ".", ".."):
        raise ValueError(f"invalid temporary image leaf: {candidate}")
    if (expected_image_id is not None
            and candidate_resolved.name != str(expected_image_id)):
        raise ValueError(
            f"temporary leaf {candidate_resolved.name!r} does not match "
            f"image {expected_image_id!r}")
    owner = _read_owner(root_resolved)
    if (owner.get("schema_version") != SCHEMA_VERSION
            or owner.get("experiment_id") != str(experiment_id)
            or owner.get("purpose") != "temporary_visual_kv_only"):
        raise ValueError(f"temporary-root ownership mismatch: {root_resolved}")
    return candidate_resolved


def _write_exclusive_json(path: Path, value: Any) -> None:
    """Publish a complete JSON file without replacing an existing name."""
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
        os.link(tmp, path)  # atomic no-clobber publication on this filesystem
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


def _atomic_owned_json(path: Path, value: Any) -> None:
    """Atomically rewrite metadata inside a newly owned temporary store."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with tmp.open("x") as handle:
        json.dump(value, handle, indent=1, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _atomic_owned_torch(path: Path, value: Any) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    torch.save(value, tmp)
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def _claim_temp_root(root: Path, experiment_id: str, dataset: str) -> dict:
    if root.exists() and root.is_symlink():
        raise ValueError(f"temporary root is a symlink: {root}")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    marker = root / OWNER_FILE
    wanted = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": str(experiment_id),
        "purpose": "temporary_visual_kv_only",
        "dataset": str(dataset),
    }
    if marker.exists():
        actual = _read_owner(root)
        if actual != wanted:
            raise ValueError(f"existing temporary root has another owner: {root}")
    else:
        _write_exclusive_json(marker, wanted)
    return wanted


def _ensure_run_config(run_dir: Path, config: dict) -> None:
    if run_dir.exists() and run_dir.is_symlink():
        raise ValueError(f"run directory is a symlink: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config.json"
    if path.exists():
        current = json.loads(path.read_text())
        invariant_keys = (
            "schema_version", "experiment_id", "dataset", "index_sha256",
            "workload_sha256", "n_images", "n_questions", "skip",
            "questions_per_image_requested", "seed", "method_keys",
            "boundary_text_sha256", "model_revision",
        )
        mismatch = {key: (current.get(key), config.get(key))
                    for key in invariant_keys
                    if current.get(key) != config.get(key)}
        if mismatch:
            raise ValueError(f"existing run config mismatch: {mismatch}")
    else:
        _write_exclusive_json(path, config)
    (run_dir / "images").mkdir(exist_ok=True)
    (run_dir / "shards").mkdir(exist_ok=True)


def _validate_resume_artifact(path: Path, *, experiment_id: str,
                              dataset: str, image_id: str,
                              image_ordinal: int, shard_index: int,
                              seed: int,
                              workload: Mapping[str, Any],
                              eval_questions: Sequence[Mapping[str, Any]]) -> dict:
    """Fail closed before treating an immutable image result as complete.

    A process can die after publishing an image artifact but before publishing
    its shard marker.  Resume is allowed to skip the expensive model work in
    that case, but only after proving that the no-clobber artifact belongs to
    this exact workload and contains every method row for every selected
    question.  This is intentionally a pure JSON check: the temporary KV
    payload may already have been safely removed.
    """
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"resume artifact is not a regular file: {path}")
    artifact = json.loads(path.read_text())
    expected_header = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": str(experiment_id),
        "dataset": str(dataset),
        "image_id": str(image_id),
        "image_ordinal": int(image_ordinal),
        "shard_index": int(shard_index),
        "index_sha256": str(workload["index_sha256"]),
        "workload_sha256": str(workload["workload_sha256"]),
    }
    mismatch = {key: (artifact.get(key), expected)
                for key, expected in expected_header.items()
                if artifact.get(key) != expected}
    if mismatch:
        raise ValueError(f"resume artifact identity mismatch: {mismatch}")

    recorded_hash = artifact.get("artifact_content_sha256")
    body = {key: value for key, value in artifact.items()
            if key != "artifact_content_sha256"}
    if not isinstance(recorded_hash, str) or recorded_hash != _json_hash(body):
        raise ValueError(f"resume artifact content hash mismatch: {path}")

    manifest = artifact.get("store_manifest")
    rows = artifact.get("rows")
    if not isinstance(manifest, Mapping) or not isinstance(rows, list):
        raise ValueError(f"resume artifact omits manifest/rows: {path}")
    expected_qids = [str(q["question_id"]) for q in eval_questions]
    expected_rows = len(expected_qids) * len(METHOD_KEYS)
    if len(rows) != expected_rows:
        raise ValueError(
            f"resume artifact row count {len(rows)} != {expected_rows}")
    store_id = manifest.get("meta_sha256")
    if not isinstance(store_id, str) or not store_id:
        raise ValueError("resume artifact has no physical store identity")

    ordinal_by_key = workload["ordinal_by_key"]
    for qid in expected_qids:
        question_rows = [row for row in rows
                         if str(row.get("question_id")) == qid]
        methods = [row.get("method_key") for row in question_rows]
        if len(question_rows) != len(METHOD_KEYS) or set(methods) != set(METHOD_KEYS):
            raise ValueError(
                f"resume artifact method coverage mismatch for {image_id}/{qid}")
        ordinal = int(ordinal_by_key[(str(image_id), qid)])
        wanted_order = list(balanced_method_order(ordinal, int(seed)))
        if methods != wanted_order:
            raise ValueError(
                f"resume artifact execution order mismatch for {image_id}/{qid}")
        suffix_hashes = set()
        for position, row in enumerate(question_rows):
            method = str(row["method_key"])
            expected_budget = METHODS[method]["budget"]
            if (row.get("dataset") != dataset
                    or str(row.get("image_id")) != str(image_id)
                    or int(row.get("image_ordinal", -1)) != int(image_ordinal)
                    or int(row.get("global_request_ordinal", -1)) != ordinal
                    or row.get("method") != METHODS[method]["label"]
                    or row.get("budget") != expected_budget
                    or row.get("method_order") != wanted_order
                    or int(row.get("method_order_position", -1)) != position):
                raise ValueError(
                    f"resume artifact row identity mismatch for {image_id}/{qid}/{method}")
            if not float(row["end_to_end_ttft_ms"]) < float(row["request_e2e_ms"]):
                raise ValueError("resume artifact violates TTFT < E2E")
            suffix_hashes.add(str(row.get("suffix_ids_sha256")))
            if method == "recompute":
                if (row.get("same_physical_store_id") is not None
                        or row.get("store_id") is not None
                        or int(row.get("ssd_read_bytes", -1)) != 0
                        or int(row.get("ssd_read_preads", -1)) != 0
                        or int(row.get("vision_forward_count", -1)) != 1):
                    raise ValueError("resume ReComp row violates the baseline contract")
            else:
                if (row.get("same_physical_store_id") != store_id
                        or row.get("store_id") != store_id
                        or int(row.get("vision_forward_count", -1)) != 0
                        or row.get("page_cache_conditioning_excluded_from_ttft") is not True):
                    raise ValueError("resume cache row violates the shared-store contract")
                if method == "fullload":
                    if int(row.get("ssd_read_bytes", -1)) != int(
                            manifest["visual_kv_bytes"]):
                        raise ValueError("resume FullLoad byte count is incomplete")
                else:
                    selected = row.get("selected_chunk_ids_per_layer")
                    wanted = prefix_chunk_ids(int(row["n_chunks_total"]),
                                              float(expected_budget))
                    if (not selected or any(
                            [int(x) for x in layer] != wanted
                            for layer in selected)):
                        raise ValueError("resume Prefix row is not exact first-k")
                    if (int(row.get("static_score_calls", -1)) != 0
                            or int(row.get("query_score_calls", -1)) != 0
                            or int(row.get("diversity_calls", -1)) != 0):
                        raise ValueError("resume Prefix row contains selector calls")
        if len(suffix_hashes) != 1 or "None" in suffix_hashes:
            raise ValueError(f"resume suffix mismatch for {image_id}/{qid}")
    if {str(row.get("question_id")) for row in rows} != set(expected_qids):
        raise ValueError("resume artifact contains an unexpected question")
    return artifact


def _validate_completed_shard(marker_path: Path, *, experiment_id: str,
                              dataset: str, shard_index: int,
                              shard_size: int, start: int, stop: int,
                              seed: int,
                              workload: Mapping[str, Any],
                              shard_entries: Sequence[Mapping[str, Any]],
                              run_dir: Path) -> None:
    """Validate a completed shard and all of its immutable image artifacts."""
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ValueError(f"completed shard marker is not regular: {marker_path}")
    marker = json.loads(marker_path.read_text())
    image_ids = [str(entry["image_id"]) for entry in shard_entries]
    expected = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": str(experiment_id),
        "shard_index": int(shard_index),
        "shard_size": int(shard_size),
        "image_start": int(start),
        "image_stop": int(stop),
        "image_ids": image_ids,
        "completed_image_ids": image_ids,
        "complete": True,
    }
    mismatch = {key: (marker.get(key), value) for key, value in expected.items()
                if marker.get(key) != value}
    # These redundant identity fields were added after the first successful
    # smoke artifact.  Old markers remain resumable because config + every
    # content-hashed artifact are validated above/below; when present, the
    # redundant fields must still agree.
    optional_identity = {
        "dataset": str(dataset),
        "index_sha256": str(workload["index_sha256"]),
        "workload_sha256": str(workload["workload_sha256"]),
    }
    mismatch.update({
        key: (marker.get(key), value)
        for key, value in optional_identity.items()
        if key in marker and marker.get(key) != value
    })
    if mismatch:
        raise ValueError(f"completed shard marker mismatch: {mismatch}")
    hashes = marker.get("image_artifact_hashes")
    if not isinstance(hashes, Mapping) or set(hashes) != set(image_ids):
        raise ValueError("completed shard artifact hash coverage mismatch")
    for local_i, entry in enumerate(shard_entries):
        image_id = str(entry["image_id"])
        path = run_dir / "images" / f"{image_id}.json"
        eval_questions = entry["questions"][
            int(workload["skip"]):int(workload["skip"])
            + int(workload["questions"])]
        _validate_resume_artifact(
            path, experiment_id=experiment_id, dataset=dataset,
            image_id=image_id, image_ordinal=start + local_i,
            shard_index=shard_index, seed=seed, workload=workload,
            eval_questions=eval_questions)
        if sha256_file(path) != hashes[image_id]:
            raise ValueError(f"completed shard file hash mismatch: {path}")


def _validate_existing_run_config(run_dir: Path, *, args,
                                  workload: Mapping[str, Any],
                                  n_shards: int) -> dict:
    """Check the persisted run identity without loading the GPU model."""
    path = run_dir / "config.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"completed run has no regular config: {path}")
    config = json.loads(path.read_text())
    expected = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": str(args.experiment_id),
        "dataset": str(args.dataset),
        "index_sha256": str(workload["index_sha256"]),
        "workload_sha256": str(workload["workload_sha256"]),
        "n_images": int(workload["n_images"]),
        "n_questions": int(workload["n_questions"]),
        "skip": int(args.skip),
        "questions_per_image_requested": int(workload["questions"]),
        "shard_size": int(args.shard_size),
        "n_shards": int(n_shards),
        "seed": int(args.seed),
        "method_keys": list(METHOD_KEYS),
        "boundary_text_sha256": str(workload["boundary_text_sha256"]),
        "max_new_tokens": int(args.max_new_tokens),
        "physical_layout": "visionzip_image_only",
        "cache_condition": "OS-page-cache-cold",
        "main_ttft_field": "end_to_end_ttft_ms",
    }
    mismatch = {key: (config.get(key), value) for key, value in expected.items()
                if config.get(key) != value}
    if mismatch:
        raise ValueError(f"completed run config mismatch: {mismatch}")
    return config


def _fixed_boundary_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return the only object visible to ``build_one``; no real Q/A survives."""
    return {
        "image_id": str(entry["image_id"]),
        "image_path": str(entry["image_path"]),
        "questions": [{
            "question_id": BOUNDARY_QUESTION_ID,
            "question": BOUNDARY_TEXT,
            "answer": "",
        }],
    }


def _collect_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _collect_strings(key)
            yield from _collect_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _collect_strings(item)


def _patch_and_validate_build_artifacts(image_dir: Path, profile: dict,
                                        eval_questions: Sequence[dict]) -> tuple:
    """Correct legacy boundary provenance and prove layout Q/A independence."""
    meta_path = image_dir / "meta.json"
    layout_path = image_dir / "visionzip_layout.pt"
    meta = json.loads(meta_path.read_text())
    layout = torch.load(layout_path, map_location="cpu", weights_only=True)
    additions = {
        "dataset_question_used_for_prefix_boundary_only": False,
        "synthetic_boundary_used_for_prefix_boundary_only": True,
        "boundary_text_source": "fixed_non_dataset_synthetic",
        "boundary_question_id": BOUNDARY_QUESTION_ID,
        "boundary_text": BOUNDARY_TEXT,
        "boundary_text_sha256": _sha256_bytes(BOUNDARY_TEXT.encode("utf-8")),
        "layout_uses_dataset_question": False,
        "layout_uses_generated_answer": False,
        "layout_uses_llm_qk": False,
        "llm_used_for_layout_scoring": False,
        "calibration_questions": 0,
    }
    meta.update(additions)
    layout.update(additions)
    profile = dict(profile)
    profile.update(additions)
    profile.update({
        "saliency_ms": float(profile["vision_saliency_ms"]),
        "repack_ms": float(profile["kv_materialize_ms"]
                           + profile["kv_repack_ms"]),
        "mapping_metadata_bytes": int(
            meta_path.stat().st_size + layout_path.stat().st_size),
        "visual_kv_bytes": int(meta["bytes_visual_kv"]),
    })
    _atomic_owned_json(meta_path, meta)
    _atomic_owned_torch(layout_path, layout)

    forbidden_questions = {str(q["question"]) for q in eval_questions}
    forbidden_ids = {str(q["question_id"]) for q in eval_questions}
    artifact_strings = (set(_collect_strings(meta))
                        | set(_collect_strings(layout))
                        | set(_collect_strings(profile)))
    if artifact_strings & forbidden_questions:
        raise AssertionError("an evaluation question leaked into layout metadata")
    if artifact_strings & forbidden_ids:
        raise AssertionError("an evaluation question ID leaked into layout metadata")
    required_false = (
        "layout_uses_dataset_question", "layout_uses_generated_answer",
        "layout_uses_llm_qk", "llm_used_for_layout_scoring",
        "dataset_question_used_for_prefix_boundary_only",
    )
    for document_name, document in (("meta", meta), ("layout", layout),
                                    ("profile", profile)):
        if any(document.get(key) is not False for key in required_false):
            raise AssertionError(f"{document_name} has query-dependent provenance")
        if int(document.get("calibration_questions", -1)) != 0:
            raise AssertionError(f"{document_name} has calibration provenance")
        if document.get("boundary_text_source") != "fixed_non_dataset_synthetic":
            raise AssertionError(f"{document_name} has the wrong boundary source")
    if meta.get("physical_layout") != "visionzip_image_only":
        raise AssertionError("direct builder did not create the requested layout")
    return meta, layout, profile


def _fsync_store_tree(image_dir: Path) -> dict[str, Any]:
    """Make recent buffered writes durable before any DONTNEED conditioning."""
    started = time.perf_counter()
    files = sorted(path for path in image_dir.rglob("*") if path.is_file())
    total_bytes = 0
    for path in files:
        if path.is_symlink():
            raise ValueError(f"store payload may not contain symlinks: {path}")
        total_bytes += path.stat().st_size
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    directories = [image_dir] + sorted(
        (path for path in image_dir.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts), reverse=True)
    for path in directories:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return {
        "fsync_ms": float((time.perf_counter() - started) * 1e3),
        "fsynced_files": len(files),
        "fsynced_directories": len(directories),
        "total_store_bytes": int(total_bytes),
        "durable_before_cache_conditioning": True,
    }


def _compact_layout(layout: Mapping[str, Any]) -> dict[str, Any]:
    score = torch.as_tensor(layout["token_score_original"]).float().flatten()
    return {
        "importance_source": str(layout["importance_source"]),
        "token_score_original": [float(x) if math.isfinite(float(x)) else None
                                 for x in score.tolist()],
        "stored_to_original": [int(x) for x in torch.as_tensor(
            layout["stored_to_original"]).flatten().tolist()],
        "original_to_stored": [int(x) for x in torch.as_tensor(
            layout["original_to_stored"]).flatten().tolist()],
        "newline_original": [int(x) for x in torch.as_tensor(
            layout["newline_original"]).flatten().tolist()],
        "layout_uses_dataset_question": False,
        "layout_uses_generated_answer": False,
        "layout_uses_llm_qk": False,
        "llm_used_for_layout_scoring": False,
        "calibration_questions": 0,
        "boundary_text_source": "fixed_non_dataset_synthetic",
        "boundary_text_sha256": _sha256_bytes(BOUNDARY_TEXT.encode("utf-8")),
    }


def _store_manifest(image_dir: Path, meta: Mapping[str, Any],
                    layout: Mapping[str, Any]) -> dict[str, Any]:
    files = sorted(path for path in image_dir.rglob("*") if path.is_file())
    sizes = {path.relative_to(image_dir).as_posix(): int(path.stat().st_size)
             for path in files}
    order = [int(x) for x in torch.as_tensor(
        layout["stored_to_original"]).flatten().tolist()]
    return {
        "physical_layout": "visionzip_image_only",
        "n_files": len(files),
        "file_sizes": sizes,
        "total_store_bytes": int(sum(sizes.values())),
        "meta_sha256": sha256_file(image_dir / "meta.json"),
        "layout_sha256": sha256_file(image_dir / "visionzip_layout.pt"),
        "source_image_sha256": meta.get("source_image_sha256"),
        "permutation_sha256": permutation_sha256(order),
        "visual_kv_bytes": int(meta["bytes_visual_kv"]),
        "separator_sidecar_bytes": int(meta["bytes_separator_sidecar"]),
        "probe_sidecar_bytes": int(meta["bytes_probe_sidecar"]),
        "payload_full_hash_performed": False,
    }


class _TimedCallableProxy:
    def __init__(self, target):
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "elapsed_ms", 0.0)
        object.__setattr__(self, "calls", 0)

    def __call__(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return self.target(*args, **kwargs)
        finally:
            self.elapsed_ms += (time.perf_counter() - started) * 1e3
            self.calls += 1

    def __getattr__(self, name):
        return getattr(self.target, name)

    def __setattr__(self, name, value):
        if name in {"target", "elapsed_ms", "calls"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.target, name, value)


def _combined_processor(runner, image, prompt) -> tuple[dict, dict]:
    processor = runner.processor
    tokenizer = processor.tokenizer
    image_processor = processor.image_processor
    timed_tokenizer = _TimedCallableProxy(tokenizer)
    timed_image = _TimedCallableProxy(image_processor)
    processor.tokenizer = timed_tokenizer
    processor.image_processor = timed_image
    started = time.perf_counter()
    try:
        enc = processor(images=image, text=prompt, return_tensors="pt")
    finally:
        total_ms = (time.perf_counter() - started) * 1e3
        processor.tokenizer = tokenizer
        processor.image_processor = image_processor
    if timed_tokenizer.calls != 1 or timed_image.calls != 1:
        raise AssertionError("combined processor component call count changed")
    component = timed_tokenizer.elapsed_ms + timed_image.elapsed_ms
    return dict(enc), {
        "processor_total_ms": float(total_ms),
        "tokenization_ms": float(timed_tokenizer.elapsed_ms),
        "image_preprocess_ms": float(timed_image.elapsed_ms),
        "input_prepare_ms": float(max(0.0, total_ms - component)),
    }


def _suffix_from_tokenized(runner, tokenized: Mapping[str, torch.Tensor]):
    ids = tokenized["input_ids"][0]
    positions = (ids == runner.image_token_id).nonzero(as_tuple=True)[0]
    if positions.numel() < 1:
        raise AssertionError("single-image prompt has no image placeholder")
    # The combined processor expands one placeholder to a contiguous image span;
    # the tokenizer-only stored path retains exactly one.  Cutting after the last
    # occurrence is correct for both forms and must produce identical suffixes.
    if positions.numel() > 1:
        expected = torch.arange(positions[0], positions[-1] + 1,
                                device=positions.device)
        if not torch.equal(positions, expected):
            raise AssertionError("expanded image token positions are not contiguous")
    return ids[int(positions[-1]) + 1:]


class _VisionCounter:
    def __init__(self, runner, expected: int):
        self.tower = runner.model.model.vision_tower
        self.expected = int(expected)
        self.calls = 0
        self.handle = None

    def __enter__(self):
        self.handle = self.tower.register_forward_pre_hook(self._count)
        return self

    def _count(self, *_args, **_kwargs):
        self.calls += 1

    def __exit__(self, exc_type, exc, traceback):
        self.handle.remove()
        if exc_type is None and self.calls != self.expected:
            raise AssertionError(
                f"vision-forward count {self.calls} != {self.expected}")
        return False


def _timing_fields(result: Mapping[str, Any], request_started: float,
                   returned_at: float, phases: Mapping[str, Any]) -> dict:
    core_started = float(result["core_started_at_s"])
    first = float(result["first_token_at_s"])
    model_finished = float(result["model_finished_at_s"])
    pre_core_ms = (core_started - request_started) * 1e3
    core_ttft_ms = (first - core_started) * 1e3
    end_to_end_ttft_ms = (first - request_started) * 1e3
    decode_ms = (model_finished - first) * 1e3
    model_e2e_ms = (model_finished - request_started) * 1e3
    request_e2e_ms = (returned_at - request_started) * 1e3
    out = {
        **phases,
        "pre_core_ms": float(pre_core_ms),
        "core_ttft_ms": float(core_ttft_ms),
        "end_to_end_ttft_ms": float(end_to_end_ttft_ms),
        "ttft_ms": float(end_to_end_ttft_ms),
        "decode_ms": float(decode_ms),
        "model_e2e_ms": float(model_e2e_ms),
        "request_e2e_ms": float(request_e2e_ms),
        "e2e_ms": float(request_e2e_ms),
        "e2e_latency_ms": float(request_e2e_ms),
        "request_started_at_s": float(request_started),
        "core_started_at_s": core_started,
        "first_token_at_s": first,
        "model_finished_at_s": model_finished,
        "request_returned_at_s": float(returned_at),
        "ttft_identity_error_ms": float(
            end_to_end_ttft_ms - pre_core_ms - core_ttft_ms),
        "model_e2e_identity_error_ms": float(
            model_e2e_ms - end_to_end_ttft_ms - decode_ms),
    }
    if not (request_started <= core_started <= first <= model_finished
            <= returned_at):
        raise AssertionError("request timestamps are not monotonic")
    if not end_to_end_ttft_ms < request_e2e_ms:
        raise AssertionError("TTFT must be smaller than request E2E")
    return out


def _io_fields(result: Mapping[str, Any], method: str,
               full_visual_bytes: int) -> dict[str, Any]:
    io = result.get("io") or {"bytes": 0, "ms": 0.0, "preads": 0,
                              "chunk_units": 0, "per_kind": {}}
    per_kind = io.get("per_kind", {})
    normal_bytes = sum(int(per_kind.get(kind, {}).get("bytes", 0))
                       for kind in ("k", "v"))
    sep_bytes = int(per_kind.get("sep", {}).get("bytes", 0))
    normal_preads = sum(int(per_kind.get(kind, {}).get("preads", 0))
                        for kind in ("k", "v"))
    sep_preads = int(per_kind.get("sep", {}).get("preads", 0))
    selected = normal_bytes + sep_bytes if method != "recompute" else 0
    hook_ms = float(result.get("hook_ms", 0.0) or 0.0)
    if method == "fullload":
        scatter_ms = max(hook_ms - float(io.get("ms", 0.0)), 0.0)
        scatter_semantics = "derived_hook_minus_pread"
    elif method.startswith("prefix"):
        scatter_ms = float(result.get("scatter_ms", 0.0) or 0.0)
        scatter_semantics = "direct_synchronized_scatter_timer"
    else:
        scatter_ms = None
        scatter_semantics = "not_applicable"
    return {
        "ssd_read_ms": float(io.get("ms", 0.0)),
        "ssd_read_bytes": int(io.get("bytes", 0)),
        "ssd_read_preads": int(io.get("preads", 0)),
        "ssd_preads": int(io.get("preads", 0)),
        "ssd_read_chunks": int(io.get("chunk_units", 0)),
        "normal_kv_read_bytes": int(normal_bytes),
        "separator_read_bytes": int(sep_bytes),
        "normal_kv_preads": int(normal_preads),
        "separator_preads": int(sep_preads),
        "selected_visual_kv_bytes": int(selected),
        "full_visual_kv_bytes": int(full_visual_bytes),
        "actual_total_ssd_byte_ratio": (
            float(selected) / full_visual_bytes
            if method != "recompute" and full_visual_bytes else None),
        "selector_ms": float(result.get("selector_ms", 0.0) or 0.0),
        "first_k_planning_ms": float(result.get("selector_ms", 0.0) or 0.0),
        "scatter_ms": scatter_ms,
        "scatter_ms_semantics": scatter_semantics,
        "prefill_ms": float(result.get("prefill_ms", 0.0) or 0.0),
        "hook_ms": (hook_ms if method == "fullload" else None),
        "n_chunks_selected": result.get("n_chunks_selected"),
        "n_chunks_total": result.get("n_chunks_total"),
        "selected_chunk_ids_per_layer": result.get(
            "selected_chunk_ids_per_layer"),
        "touched_chunk_fraction": result.get("touched_chunk_fraction"),
        "logical_kv_ratio": result.get("logical_kv_ratio"),
        "static_score_calls": int(result.get("static_score_calls", 0)),
        "query_score_calls": int(result.get("query_score_calls", 0)),
        "diversity_calls": int(result.get("diversity_calls", 0)),
        "selection_mode": result.get("selection_mode"),
        "separator_policy": result.get("separator_policy"),
    }


def _run_recompute(runner, server, image, question: str) -> tuple[dict, str]:
    with _VisionCounter(runner, expected=1) as vision:
        torch.cuda.synchronize()
        request_started = time.perf_counter()
        started = time.perf_counter()
        prompt = runner.prompt(question)
        prompt_ms = (time.perf_counter() - started) * 1e3
        enc_cpu, processor = _combined_processor(runner, image, prompt)
        started = time.perf_counter()
        enc_device = runner.to_device(enc_cpu)
        torch.cuda.synchronize()
        h2d_ms = (time.perf_counter() - started) * 1e3
        result = server.recompute(enc_device)
        returned = time.perf_counter()
    suffix = _suffix_from_tokenized(runner, enc_cpu)
    phases = {
        "prompt_build_ms": float(prompt_ms),
        "tokenization_ms": processor["tokenization_ms"],
        "image_preprocess_ms": processor["image_preprocess_ms"],
        "input_prepare_ms": processor["input_prepare_ms"],
        "input_h2d_ms": float(h2d_ms),
        "processor_total_ms": processor["processor_total_ms"],
    }
    result = dict(result)
    result.update(_timing_fields(result, request_started, returned, phases))
    result.update({
        "vision_forward_count": int(vision.calls),
        "page_cache_conditioning_method": "not_applicable_recompute",
        "page_cache_conditioning_ms": 0.0,
        "page_cache_conditioning_excluded_from_ttft": True,
    })
    suffix_hash = _sha256_bytes(
        suffix.detach().cpu().contiguous().numpy().tobytes())
    del enc_device, enc_cpu
    return result, suffix_hash


def _run_stored(runner, server, ctx, question: str, method: str,
                budget: float | None, image_id: str, seed: int) -> tuple[dict, str]:
    with _VisionCounter(runner, expected=0) as vision:
        conditioning_started = time.perf_counter()
        ctx.reader.drop_all()
        conditioning_finished = time.perf_counter()
        torch.cuda.synchronize()
        request_started = time.perf_counter()
        started = time.perf_counter()
        prompt = runner.prompt(question)
        prompt_ms = (time.perf_counter() - started) * 1e3
        started = time.perf_counter()
        tokenized = runner.processor.tokenizer(prompt, return_tensors="pt")
        token_ms = (time.perf_counter() - started) * 1e3
        started = time.perf_counter()
        suffix_cpu = _suffix_from_tokenized(runner, tokenized)
        prepare_ms = (time.perf_counter() - started) * 1e3
        started = time.perf_counter()
        suffix_device = suffix_cpu.to(runner.model.device)
        torch.cuda.synchronize()
        h2d_ms = (time.perf_counter() - started) * 1e3
        if method == "fullload":
            result = server.request(ctx, mode="fullload", cold=False,
                                    suffix_ids=suffix_device)
        else:
            result = server.request_cvpr25(
                ctx, static=None, budget=float(budget), mode="prefix",
                sep_policy="sidecar", cold=False, seed=seed,
                image_id=image_id, suffix_ids=suffix_device,
                expected_prefix_layout="visionzip_image_only")
        returned = time.perf_counter()
    phases = {
        "prompt_build_ms": float(prompt_ms),
        "tokenization_ms": float(token_ms),
        "image_preprocess_ms": 0.0,
        "input_prepare_ms": float(prepare_ms),
        "input_h2d_ms": float(h2d_ms),
        "processor_total_ms": None,
    }
    result = dict(result)
    result.update(_timing_fields(result, request_started, returned, phases))
    result.update({
        "vision_forward_count": int(vision.calls),
        "page_cache_conditioning_started_at_s": float(conditioning_started),
        "page_cache_conditioning_finished_at_s": float(conditioning_finished),
        "page_cache_conditioning_ms": float(
            (conditioning_finished - conditioning_started) * 1e3),
        "page_cache_conditioning_method": "posix_fadvise_DONTNEED",
        "page_cache_conditioning_excluded_from_ttft": True,
    })
    if not conditioning_finished < request_started:
        raise AssertionError("page-cache conditioning overlaps the request timer")
    suffix_hash = _sha256_bytes(
        suffix_cpu.detach().cpu().contiguous().numpy().tobytes())
    return result, suffix_hash


def _importance_coverage(layout: Mapping[str, Any], meta: Mapping[str, Any],
                         budget: float) -> dict[str, Any]:
    scores = torch.as_tensor(layout["token_score_original"]).float().flatten()
    order = torch.as_tensor(layout["stored_to_original"]).long().flatten()
    separators = set(int(x) for x in meta["newline_idx"])
    normal = torch.tensor([i for i in range(scores.numel()) if i not in separators])
    n_chunks = int(meta["n_chunks_per_layer"])
    k = budget_chunk_count(n_chunks, budget)
    selected_stored_rows = min(k * int(meta["chunk_size"]), scores.numel())
    selected_original = order[:selected_stored_rows]
    selected_normal = torch.tensor(
        [int(x) for x in selected_original.tolist() if int(x) not in separators])
    total_mass = float(scores[normal].double().sum())
    selected_mass = float(scores[selected_normal].double().sum())
    return {
        "importance_mass_coverage": (
            selected_mass / total_mass if total_mass else None),
        "selected_importance_mass": selected_mass,
        "total_importance_mass": total_mass,
        "nominal_budget": float(budget),
        "actual_selected_normal_chunk_fraction": float(k / n_chunks),
        "selected_normal_tokens_by_layout": int(selected_normal.numel()),
        "total_normal_tokens": int(normal.numel()),
    }


def _synthetic_warmup(runner, server) -> dict[str, Any]:
    height, width = 480, 640
    yy, xx = np.indices((height, width), dtype=np.uint16)
    pixels = np.stack(((3 * xx + yy) % 256, (xx + 5 * yy) % 256,
                       (7 * xx + 11 * yy) % 256), axis=-1).astype(np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    enc = runner.encode_prompt(image, WARMUP_PROMPT)
    result = server.recompute(enc)
    return {
        "enabled": True,
        "count": 1,
        "fixture": "deterministic_in_memory_rgb_pattern_640x480",
        "fixture_sha256": _sha256_bytes(pixels.tobytes()),
        "fixture_is_dataset_image": False,
        "prompt_sha256": _sha256_bytes(WARMUP_PROMPT.encode("utf-8")),
        "consumed_evaluation_request": False,
        "ssd_store_read_or_written": False,
        "excluded_from_all_latency_metrics": True,
        "first_token_id": int(result["first_token_id"]),
    }


def _model_revision() -> str | None:
    ref = (Path.home() / ".cache/huggingface/hub/"
           "models--llava-hf--llava-v1.6-vicuna-7b-hf/refs/main")
    return ref.read_text().strip() if ref.is_file() else None


def _base_config(args, workload: Mapping[str, Any], n_shards: int,
                 warmup: Mapping[str, Any], runner) -> dict[str, Any]:
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(args.temp_root)
    gpu = torch.cuda.get_device_properties(0)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": args.experiment_id,
        "dataset": args.dataset,
        "index": str(workload["path"]),
        "index_sha256": workload["index_sha256"],
        "workload_sha256": workload["workload_sha256"],
        "source_full_workload_sha256": workload.get(
            "source_full_workload_sha256", workload["workload_sha256"]),
        "source_full_n_images": int(workload.get(
            "source_full_n_images", workload["n_images"])),
        "source_full_n_questions": int(workload.get(
            "source_full_n_questions", workload["n_questions"])),
        "partial_workload": bool(workload.get("partial_workload", False)),
        "n_images": workload["n_images"],
        "n_questions": workload["n_questions"],
        "questions_per_image": workload["questions_per_image"],
        "skip": args.skip,
        "questions_per_image_requested": int(workload["questions"]),
        "shard_size": args.shard_size,
        "n_shards": n_shards,
        "seed": args.seed,
        "method_keys": list(METHOD_KEYS),
        "methods": METHODS,
        "method_order_policy": (
            "cyclic deterministic rotation by global canonical request ordinal "
            "plus seed; invariant to shard size/order/resume"),
        "boundary_question_id": BOUNDARY_QUESTION_ID,
        "boundary_text": BOUNDARY_TEXT,
        "boundary_text_sha256": workload["boundary_text_sha256"],
        "boundary_text_source": "fixed_non_dataset_synthetic",
        "boundary_id_and_text_do_not_collide_with_index": True,
        "calibration_questions": 0,
        "layout_uses_dataset_question": False,
        "layout_uses_generated_answer": False,
        "layout_uses_llm_qk": False,
        "query_dependent_importance": False,
        "static_diverse_calls": 0,
        "model": runner.model_id,
        "model_revision": _model_revision(),
        "load_4bit": runner.load_4bit,
        "quantization": "4-bit NF4 double-quant",
        "attention": runner.attn,
        "decoding": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "chunk_size": CHUNK_SIZE,
        "physical_layout": "visionzip_image_only",
        "retrieval": "sequential physical first-k chunks",
        "separator_policy": "stable_tail_plus_sidecar",
        "ssd_read_api": "buffered os.pread",
        "o_direct": False,
        "ssd_controller_cache_flushed": False,
        "cache_condition": "OS-page-cache-cold",
        "page_cache_conditioning": "posix_fadvise(DONTNEED)",
        "page_cache_conditioning_inside_ttft": False,
        "store_fsynced_before_page_cache_conditioning": True,
        "main_ttft_field": "end_to_end_ttft_ms",
        "core_ttft_field": "core_ttft_ms",
        "tokenization_inside_ttft": True,
        "initial_h2d_inside_ttft": True,
        "prompt_construction_inside_ttft": True,
        "recompute_image_processing_inside_ttft": True,
        "ttft_definition": (
            "after page-cache conditioning, before prompt construction -> "
            "tokenization/processor -> input preparation -> initial H2D -> "
            "ReComp vision or SSD pread/scatter -> prefill -> greedy first "
            "token -> CUDA synchronize"),
        "jpeg_file_read_and_decode_timed": False,
        "image_processor_resize_crop_tensorize_timed_for_recompute": True,
        "persistence_in_main_ttft": False,
        "image_at_a_time_temporary_store": True,
        "temporary_payload_deleted_after_immutable_image_artifact": True,
        "unmeasured_warmup": dict(warmup),
        "machine": {
            "hostname": platform.node(),
            "gpu_name": gpu.name,
            "gpu_total_memory_bytes": int(gpu.total_memory),
            "system_ram_total_bytes": int(vm.total),
            "system_ram_available_bytes": int(vm.available),
            "disk_total_bytes": int(disk.total),
            "disk_free_bytes_at_start": int(disk.free),
        },
        "capacity_guard": {
            "min_free_after_gib": float(args.min_free_after_gib),
            "max_used_percent_exclusive": 96.0,
            "build_headroom_gib": 3.0,
        },
    }


def _capacity_guard(path: Path, *, reserve_bytes: int,
                    extra_headroom_bytes: int = 0) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    # Match ``df`` semantics on ext4: reserved blocks are absent from ``free``.
    # ``used/total`` would understate occupancy and weaken the 96% guard.
    used_pct = 100.0 * float(usage.used) / float(usage.used + usage.free)
    required = int(reserve_bytes) + int(extra_headroom_bytes)
    if usage.free < required:
        raise RuntimeError(
            f"free-space guard failed: free={usage.free}, required={required}")
    if used_pct >= 96.0:
        raise RuntimeError(
            f"disk-used guard failed: {used_pct:.3f}% is not below 96%")
    return {
        "disk_total_bytes": int(usage.total),
        "disk_used_bytes": int(usage.used),
        "disk_free_bytes": int(usage.free),
        "disk_used_percent": used_pct,
        "reserve_bytes": int(reserve_bytes),
        "extra_headroom_bytes": int(extra_headroom_bytes),
    }


def _validate_live_row(row: Mapping[str, Any], meta: Mapping[str, Any]) -> None:
    method = row["method_key"]
    if method == "recompute":
        if (row["ssd_read_bytes"] != 0 or row["ssd_read_preads"] != 0
                or row["vision_forward_count"] != 1):
            raise AssertionError("ReComp performed SSD I/O or omitted vision")
        return
    if row["vision_forward_count"] != 0:
        raise AssertionError("stored method invoked the vision tower")
    if not (row["page_cache_conditioning_finished_at_s"]
            < row["request_started_at_s"]):
        raise AssertionError("cache conditioning was not outside TTFT")
    if method == "fullload":
        if row["ssd_read_bytes"] != int(meta["bytes_visual_kv"]):
            raise AssertionError("FullLoad did not read exactly the full visual KV")
        return
    budget = float(row["budget"])
    wanted = prefix_chunk_ids(int(meta["n_chunks_per_layer"]), budget)
    selected = row["selected_chunk_ids_per_layer"]
    if not selected or any([int(x) for x in layer] != wanted for layer in selected):
        raise AssertionError("Prefix request is not exact physical first-k")
    if (row["static_score_calls"] != 0 or row["query_score_calls"] != 0
            or row["diversity_calls"] != 0):
        raise AssertionError("Prefix request called an online selector")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=("gqa_large", "vqav2", "textvqa"))
    parser.add_argument("--metric", required=True, choices=tuple(METRICS))
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--skip", type=int, required=True)
    parser.add_argument("--questions", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-size", type=int, default=16)
    parser.add_argument("--expected-index-sha256", required=True)
    parser.add_argument("--expected-workload-sha256", required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--expected-questions", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--min-free-after-gib", type=float, default=64.0)
    parser.add_argument("--allow-partial-workload", action="store_true",
                        help="explicitly permit deterministic smoke truncation")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-questions-per-image", type=int, default=None)
    args = parser.parse_args()

    if not args.run_dir.is_absolute() or not args.temp_root.is_absolute():
        raise ValueError("run-dir and temp-root must be absolute")
    run_resolved, temp_resolved = (args.run_dir.resolve(),
                                   args.temp_root.resolve())
    if (run_resolved == temp_resolved or run_resolved in temp_resolved.parents
            or temp_resolved in run_resolved.parents):
        raise ValueError("run and temporary roots overlap")
    if args.shard_index < 0 or args.shard_size < 1:
        raise ValueError("invalid shard specification")
    if args.max_new_tokens != 16:
        raise ValueError("the fixed generalization experiment uses 16 tokens")

    full_workload = resolve_workload(
        args.index, skip=args.skip, questions=args.questions,
        expected_index_sha256=args.expected_index_sha256,
        expected_workload_sha256=args.expected_workload_sha256,
        expected_images=args.expected_images,
        expected_questions=args.expected_questions)
    partial_requested = (args.max_images is not None
                         or args.max_questions_per_image is not None)
    if partial_requested and not args.allow_partial_workload:
        raise ValueError(
            "partial workload flags require --allow-partial-workload")
    if args.allow_partial_workload and not partial_requested:
        raise ValueError(
            "--allow-partial-workload requires an explicit truncation flag")
    workload = _partial_workload(
        full_workload, max_images=args.max_images,
        max_questions_per_image=args.max_questions_per_image)
    if not args.allow_partial_workload and workload["partial_workload"]:
        raise AssertionError("main run silently became partial")
    n_shards = math.ceil(workload["n_images"] / args.shard_size)
    if args.shard_index >= n_shards:
        raise ValueError(f"shard index {args.shard_index} >= {n_shards}")
    start = args.shard_index * args.shard_size
    stop = min(start + args.shard_size, workload["n_images"])
    shard_entries = workload["entries"][start:stop]

    _claim_temp_root(args.temp_root, args.experiment_id, args.dataset)
    payload_root = args.temp_root / "payload"
    payload_root.mkdir(exist_ok=True)
    shard_marker = args.run_dir / "shards" / f"shard_{args.shard_index:03d}.json"
    if shard_marker.is_file():
        _validate_existing_run_config(
            args.run_dir, args=args, workload=workload, n_shards=n_shards)
        _validate_completed_shard(
            shard_marker, experiment_id=args.experiment_id,
            dataset=args.dataset, shard_index=args.shard_index,
            shard_size=args.shard_size, start=start, stop=stop,
            seed=args.seed, workload=workload, shard_entries=shard_entries,
            run_dir=args.run_dir)
        print(f"shard {args.shard_index} already complete; nothing to do")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    runner = LlavaRunner().load()
    server = Server(runner, max_new_tokens=args.max_new_tokens)
    warmup = _synthetic_warmup(runner, server)
    config = _base_config(args, workload, n_shards, warmup, runner)
    _ensure_run_config(args.run_dir, config)

    completed = []
    started_at = time.time()
    reserve = int(args.min_free_after_gib * (1024 ** 3))
    for local_i, entry in enumerate(shard_entries):
        image_id = str(entry["image_id"])
        image_ordinal = start + local_i
        artifact_path = args.run_dir / "images" / f"{image_id}.json"
        image_dir = payload_root / image_id
        assert_owned_temp_path(image_dir, args.temp_root, args.experiment_id,
                               image_id)
        if artifact_path.is_file():
            eval_questions = entry["questions"][
                args.skip:args.skip + int(workload["questions"])]
            _validate_resume_artifact(
                artifact_path, experiment_id=args.experiment_id,
                dataset=args.dataset, image_id=image_id,
                image_ordinal=image_ordinal, shard_index=args.shard_index,
                seed=args.seed, workload=workload,
                eval_questions=eval_questions)
            if image_dir.exists():
                shutil.rmtree(image_dir)
            completed.append(image_id)
            print(f"[{local_i + 1}/{len(shard_entries)}] {image_id}: immutable result exists")
            continue
        if image_dir.exists():
            shutil.rmtree(image_dir)
        _capacity_guard(args.temp_root, reserve_bytes=reserve,
                        extra_headroom_bytes=3 * (1024 ** 3))

        eval_questions = entry["questions"][
            args.skip:args.skip + int(workload["questions"])]
        with Image.open(ROOT / entry["image_path"]) as source_image:
            image = source_image.convert("RGB")
        rows = []
        try:
            sanitized = _fixed_boundary_entry(entry)
            meta, profile = build_one(
                runner, sanitized, payload_root, layout="visionzip",
                separator_sidecar=True)
            meta, layout, profile = _patch_and_validate_build_artifacts(
                image_dir, profile, eval_questions)
            fsync_profile = _fsync_store_tree(image_dir)
            profile.update(fsync_profile)
            profile["mapping_metadata_bytes"] = int(
                (image_dir / "meta.json").stat().st_size
                + (image_dir / "visionzip_layout.pt").stat().st_size)
            capacity_after_build = _capacity_guard(
                args.temp_root, reserve_bytes=reserve)
            profile["capacity_after_build"] = capacity_after_build
            manifest = _store_manifest(image_dir, meta, layout)
            compact_layout = _compact_layout(layout)

            ctx = ImageContext(image_dir, runner.model.device,
                               drop_cache=True, require_v_hidden=False)
            try:
                # Layout validation is benchmark setup, not a first-request tax.
                ctx.validate_prefix_layout("visionzip_image_only")
                for q in eval_questions:
                    question_id = str(q["question_id"])
                    ordinal = workload["ordinal_by_key"][(image_id, question_id)]
                    order = balanced_method_order(ordinal, args.seed)
                    suffix_hashes = {}
                    question_rows = []
                    for position, method in enumerate(order):
                        budget = METHODS[method]["budget"]
                        torch.cuda.reset_peak_memory_stats()
                        if method == "recompute":
                            result, suffix_hash = _run_recompute(
                                runner, server, image, str(q["question"]))
                        else:
                            result, suffix_hash = _run_stored(
                                runner, server, ctx, str(q["question"]),
                                method, budget, image_id, args.seed)
                        suffix_hashes[method] = suffix_hash
                        gold = question_answers(q)
                        row = {
                            "schema_version": SCHEMA_VERSION,
                            "dataset": args.dataset,
                            "image_id": image_id,
                            "image_ordinal": image_ordinal,
                            "question_id": question_id,
                            "global_request_ordinal": ordinal,
                            "question": str(q["question"]),
                            "gold": list(gold),
                            "method_key": method,
                            "method": METHODS[method]["label"],
                            "budget": budget,
                            "method_order": list(order),
                            "method_order_position": position,
                            "prediction": result["answer"],
                            "score": float(METRICS[args.metric](
                                result["answer"], gold)),
                            "quality_score": float(METRICS[args.metric](
                                result["answer"], gold)),
                            "quality_metric": args.metric,
                            "first_token_id": int(result["first_token_id"]),
                            "generated_tokens": int(result["generated_tokens"]),
                            "suffix_ids_sha256": suffix_hash,
                            "physical_layout": (
                                "not_applicable_recompute" if method == "recompute"
                                else "visionzip_image_only"),
                            "same_physical_store_id": (
                                None if method == "recompute"
                                else manifest["meta_sha256"]),
                            "store_id": (
                                None if method == "recompute"
                                else manifest["meta_sha256"]),
                            "calibration_questions": 0,
                            "layout_uses_dataset_question": False,
                            "layout_uses_generated_answer": False,
                            "layout_uses_llm_qk": False,
                            **_io_fields(result, method,
                                         int(meta["bytes_visual_kv"])),
                            **{key: value for key, value in result.items()
                               if key in {
                                   "prompt_build_ms", "tokenization_ms",
                                   "image_preprocess_ms", "input_prepare_ms",
                                   "input_h2d_ms", "processor_total_ms",
                                   "pre_core_ms", "core_ttft_ms",
                                   "end_to_end_ttft_ms", "ttft_ms", "decode_ms",
                                   "model_e2e_ms", "request_e2e_ms", "e2e_ms",
                                   "e2e_latency_ms", "request_started_at_s",
                                   "core_started_at_s", "first_token_at_s",
                                   "model_finished_at_s", "request_returned_at_s",
                                   "ttft_identity_error_ms",
                                   "model_e2e_identity_error_ms",
                                   "vision_forward_count",
                                   "page_cache_conditioning_started_at_s",
                                   "page_cache_conditioning_finished_at_s",
                                   "page_cache_conditioning_ms",
                                   "page_cache_conditioning_method",
                                   "page_cache_conditioning_excluded_from_ttft",
                               }},
                            "gpu_memory_allocated": int(
                                torch.cuda.memory_allocated()),
                            "gpu_peak_memory_allocated": int(
                                torch.cuda.max_memory_allocated()),
                            "process_rss_bytes": int(
                                psutil.Process().memory_info().rss),
                        }
                        if method.startswith("prefix"):
                            row.update(_importance_coverage(
                                layout, meta, float(budget)))
                        _validate_live_row(row, meta)
                        question_rows.append(row)
                    if len(set(suffix_hashes.values())) != 1:
                        raise AssertionError(
                            f"method suffix mismatch: {image_id}/{question_id}")
                    rows.extend(question_rows)
            finally:
                ctx.close()

            expected_rows = len(eval_questions) * len(METHOD_KEYS)
            if len(rows) != expected_rows:
                raise AssertionError(f"row count {len(rows)} != {expected_rows}")
            artifact = {
                "schema_version": SCHEMA_VERSION,
                "experiment_id": args.experiment_id,
                "dataset": args.dataset,
                "image_id": image_id,
                "image_ordinal": image_ordinal,
                "shard_index": args.shard_index,
                "index_sha256": workload["index_sha256"],
                "workload_sha256": workload["workload_sha256"],
                "boundary": {
                    "question_id": BOUNDARY_QUESTION_ID,
                    "text": BOUNDARY_TEXT,
                    "text_sha256": workload["boundary_text_sha256"],
                    "source": "fixed_non_dataset_synthetic",
                    "collides_with_index": False,
                },
                "build_profile": profile,
                "store_manifest": manifest,
                "layout_artifact": compact_layout,
                "rows": rows,
                "validation": {
                    "passed": True,
                    "expected_rows": expected_rows,
                    "observed_rows": len(rows),
                    "calibration_questions": 0,
                    "layout_uses_dataset_question": False,
                    "layout_uses_generated_answer": False,
                    "layout_uses_llm_qk": False,
                    "all_cache_methods_same_store": len({
                        row["same_physical_store_id"] for row in rows
                        if row["method_key"] != "recompute"}) == 1,
                    "recompute_has_no_store_id": all(
                        row["same_physical_store_id"] is None for row in rows
                        if row["method_key"] == "recompute"),
                    "recompute_ssd_bytes": sum(
                        row["ssd_read_bytes"] for row in rows
                        if row["method_key"] == "recompute"),
                    "all_ttft_lt_e2e": all(
                        row["end_to_end_ttft_ms"] < row["request_e2e_ms"]
                        for row in rows),
                },
            }
            artifact["artifact_content_sha256"] = _json_hash({
                key: value for key, value in artifact.items()
                if key != "artifact_content_sha256"})
            _write_exclusive_json(artifact_path, artifact)
            completed.append(image_id)
            print(
                f"[{local_i + 1}/{len(shard_entries)}] {image_id}: "
                f"{len(rows)} rows published; store {manifest['total_store_bytes']/1e9:.2f} GB",
                flush=True)
        finally:
            if image_dir.exists():
                assert_owned_temp_path(image_dir, args.temp_root,
                                       args.experiment_id, image_id)
                shutil.rmtree(image_dir)
            image.close()
            del image, rows
            torch.cuda.empty_cache()
            gc.collect()

    marker = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": args.experiment_id,
        "dataset": args.dataset,
        "index_sha256": workload["index_sha256"],
        "workload_sha256": workload["workload_sha256"],
        "shard_index": args.shard_index,
        "shard_size": args.shard_size,
        "image_start": start,
        "image_stop": stop,
        "image_ids": [str(entry["image_id"]) for entry in shard_entries],
        "completed_image_ids": completed,
        "complete": len(completed) == len(shard_entries),
        "elapsed_seconds": time.time() - started_at,
        "image_artifact_hashes": {
            image_id: sha256_file(args.run_dir / "images" / f"{image_id}.json")
            for image_id in completed
        },
    }
    if not marker["complete"]:
        raise AssertionError("shard did not complete every image")
    _write_exclusive_json(shard_marker, marker)
    print(f"completed shard {args.shard_index}: {len(completed)} images")


if __name__ == "__main__":
    main()

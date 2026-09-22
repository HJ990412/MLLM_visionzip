#!/usr/bin/env python3
"""Strict analyzer for the MT-GQA Gold/Generated-history four-arm run.

The evaluator publishes immutable per-image artifacts.  This program is a
CPU-only, read-only consumer of those artifacts: it reconstructs the complete
protocol/method/dialogue/turn matrix, recomputes strict normalized exact-match
quality, audits causal history provenance, and only then publishes derived
results into a new results root.

The primary quality score deliberately does *not* call
``mmimpress.dataset.exact_score``.  That legacy helper accepts a prediction
whose leading words match the gold answer; the MT-GQA reporting contract here
requires equality after the frozen GQA normalization.
"""
from __future__ import annotations

import argparse
import ctypes
import csv
import errno
import gzip
import hashlib
import io
import json
import math
import os
import re
import statistics
import sys
import time
import uuid
import shutil
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mmimpress.mt_gqa import (  # noqa: E402
    IMAGE_MARKER,
    SHORT_ANSWER_INSTRUCTION,
    mt_gqa_prompt,
    mt_gqa_prior_history_text,
)


SCHEMA_VERSION = "mt-gqa-gold-generated-history-analysis-v2"
PROTOCOLS = ("gold_history", "generated_history")
METHOD_KEYS = ("recompute", "fullload", "qa_chunk25", "ours25")
METHOD_LABELS = {
    "recompute": "ReComp",
    "fullload": "FullLoad",
    "qa_chunk25": "QA-Chunk25",
    "ours25": "Ours25",
}
TURNS = (1, 2, 3)
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 1234
FULL_DIALOGUES = 4_061
FULL_IMAGES = 398
FROZEN_MODEL_REVISION = "c916e6cdcd760b4cecd1dd4907f84ac649f93b23"
QA_RATER_ALGORITHM_ID = "sparsevlm_visual_text_mean_threshold_v1"
FROZEN_QA_CONFIGURATION = {
    "physical_layout": "raster",
    "head_reduce": "mean",
    "chunk_aggregation": "mean_valid_spatial_tokens",
    "normal_chunk_budget": 0.25,
    "budget_helper": "budget_chunk_count_round",
    "rater_algorithm_id": QA_RATER_ALGORITHM_ID,
    "rater_scope": "entire_available_causal_suffix",
    "fallback": False,
    "adaptive_budget": False,
}
FROZEN_OURS_CONFIGURATION = {
    "physical_layout": "visionzip_image_only",
    "normal_chunk_budget": 0.25,
    "selection": "fixed_first_k_prefix",
    "online_query_scoring": False,
}


class AnalysisError(RuntimeError):
    """An immutable evaluator artifact violated the frozen contract."""


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _stable_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha_bytes(payload)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite result: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory_noreplace(staging: Path, destination: Path) -> None:
    """Atomically publish a same-filesystem directory without replacement."""
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to replace results root: {destination}")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        at_fdcwd = -100
        rename_noreplace = 1
        result = renameat2(
            ctypes.c_int(at_fdcwd), ctypes.c_char_p(os.fsencode(staging)),
            ctypes.c_int(at_fdcwd), ctypes.c_char_p(os.fsencode(destination)),
            ctypes.c_uint(rename_noreplace))
        if result != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(destination))
    else:
        # Linux environments used by this project expose renameat2.  Retain a
        # guarded fallback for CPU unit-test platforms without that symbol.
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        os.rename(staging, destination)
    _fsync_directory(destination.parent)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(
        value, indent=2, ensure_ascii=False, allow_nan=False,
    ) + "\n")


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    _atomic_text(path, buffer.getvalue())


def normalize_answer(value: Any) -> str:
    """Frozen strict GQA normalization used by the prior MT-GQA analyzer."""
    text = re.sub(r"[^\w\s]", " ", str(value).lower())
    return " ".join(word for word in text.split() if word not in {"a", "an", "the"})


def strict_exact_score(prediction: Any, gold: Any) -> float:
    if isinstance(gold, (list, tuple)):
        if not gold:
            raise AnalysisError("gold answer list is empty")
        gold = gold[0]
    return float(normalize_answer(prediction) == normalize_answer(gold))


def exact_mcnemar_pvalue(qa_only: int, ours_only: int) -> float:
    """Two-sided exact McNemar/binomial p-value without a SciPy dependency."""
    b, c = int(qa_only), int(ours_only)
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be nonnegative")
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    # Keep the binomial tail as an exact rational until the final JSON-facing
    # conversion.  This avoids labeling an lgamma approximation "exact".
    numerator = 2 * sum(math.comb(n, i) for i in range(m + 1))
    probability = Fraction(numerator, 1 << n)
    return float(min(Fraction(1, 1), probability))


def _pick(row: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    result = row.get("result")
    if isinstance(result, Mapping):
        for name in names:
            if name in result and result[name] is not None:
                return result[name]
    return default


def _required(row: Mapping[str, Any], context: str, *names: str) -> Any:
    value = _pick(row, *names, default=None)
    if value is None:
        raise AnalysisError(f"{context}: missing {'/'.join(names)}")
    return value


def _as_int(value: Any, context: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise AnalysisError(f"{context}: bool is not an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{context}: invalid integer {value!r}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise AnalysisError(f"{context}: non-integral value {value!r}")
    if minimum is not None and result < minimum:
        raise AnalysisError(f"{context}: {result} < {minimum}")
    return result


def _as_float(value: Any, context: str, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{context}: invalid number {value!r}") from exc
    if not math.isfinite(result):
        raise AnalysisError(f"{context}: non-finite number")
    if minimum is not None and result < minimum:
        raise AnalysisError(f"{context}: {result} < {minimum}")
    return result


def _canonical_protocol(value: Any) -> str:
    key = re.sub(r"[^a-z]", "", str(value).lower())
    if key in {"gold", "goldhistory", "teacherforced", "goldteacherforced"}:
        return "gold_history"
    if key in {"generated", "generatedhistory", "selfgenerated", "methodgenerated"}:
        return "generated_history"
    raise AnalysisError(f"unknown history protocol: {value!r}")


def _canonical_method(value: Any) -> str:
    key = re.sub(r"[^a-z0-9]", "", str(value).lower())
    aliases = {
        "recompute": "recompute", "recomp": "recompute",
        "fullload": "fullload", "full": "fullload",
        "qachunk25": "qa_chunk25", "qachunk": "qa_chunk25",
        "ours25": "ours25", "ours": "ours25",
        "imageonlyprefix25": "ours25", "prefix25": "ours25",
    }
    if key not in aliases:
        raise AnalysisError(f"unknown method: {value!r}")
    return aliases[key]


def _gold_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        values = [str(item) for item in value]
    else:
        values = [str(value)]
    if not values or any(not item for item in values):
        raise AnalysisError("gold answer must be nonempty")
    return values


def _list_of_strings(value: Any, context: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise AnalysisError(f"{context}: expected a list")
    return [str(item) for item in value]


def _selected_layers(value: Any, context: str) -> list[list[int]]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise AnalysisError(f"{context}: selected chunks must be layer lists")
    layers: list[list[int]] = []
    for layer, items in enumerate(value):
        if not isinstance(items, (list, tuple)):
            raise AnalysisError(f"{context}: layer {layer} is not a list")
        ids = [_as_int(item, f"{context}/layer{layer}", minimum=0) for item in items]
        if len(ids) != len(set(ids)):
            raise AnalysisError(f"{context}: duplicate chunk ID in layer {layer}")
        layers.append(ids)
    return layers


def _metric_float(row: Mapping[str, Any], *names: str, default: float = 0.0) -> float:
    value = _pick(row, *names, default=None)
    return float(default) if value is None else _as_float(value, names[0], minimum=0.0)


def _metric_int(row: Mapping[str, Any], *names: str, default: int = 0) -> int:
    value = _pick(row, *names, default=None)
    return int(default) if value is None else _as_int(value, names[0], minimum=0)


def _io_nested(row: Mapping[str, Any], key: str, default: Any = 0) -> Any:
    io_value = _pick(row, "io", "io_summary", default={})
    if isinstance(io_value, Mapping) and io_value.get(key) is not None:
        return io_value[key]
    return default


def _canonical_row(raw: Mapping[str, Any], source: Path, ordinal: int,
                   inherited_protocol: str | None = None) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise AnalysisError(f"{source}: row {ordinal} is not an object")
    context = f"{source.name}:row{ordinal}"
    protocol_value = _pick(raw, "protocol", "history_protocol", default=inherited_protocol)
    protocol = _canonical_protocol(_required(
        {"protocol": protocol_value}, context, "protocol"))
    method = _canonical_method(_required(raw, context, "method_key", "method", "method_id"))
    dialog_id = str(_required(raw, context, "dialog_id", "dialogue_id"))
    image_id = str(_required(raw, context, "image_id"))
    turn = _as_int(_required(raw, context, "turn_id", "turn"), f"{context}/turn")
    if turn not in TURNS:
        raise AnalysisError(f"{context}: turn must be 1, 2, or 3")
    question_id = str(_required(raw, context, "question_id"))
    question = str(_required(raw, context, "question"))
    gold = _gold_list(_required(raw, context, "gold", "gold_answer", "answers", "reference_answer"))
    prediction = str(_required(raw, context, "prediction", "answer"))
    logical_id = str(_pick(
        raw, "logical_request_id", "request_id",
        default=f"{protocol}:{dialog_id}:t{turn}:{method}"))
    physical_id_value = _pick(
        raw, "physical_execution_id", "execution_id", "inference_id",
        default=None)
    physical_id = str(physical_id_value) if physical_id_value is not None else logical_id

    status_value = _pick(raw, "status", "request_status", default=None)
    runtime_success = _pick(raw, "runtime_success", "success", default=None)
    if status_value is None and runtime_success is None:
        raise AnalysisError(f"{context}: request status is missing")
    status_ok = (str(status_value).lower() in {
        "ok", "pass", "passed", "success", "complete", "completed",
    } if status_value is not None else bool(runtime_success))
    if runtime_success is not None:
        status_ok = status_ok and bool(runtime_success)

    prompt = str(_required(raw, context, "prompt", "prompt_text"))
    prompt_sha = str(_pick(raw, "prompt_sha256", default=_sha_bytes(prompt.encode("utf-8"))))
    if prompt_sha != _sha_bytes(prompt.encode("utf-8")):
        raise AnalysisError(f"{context}: prompt SHA256 mismatch")
    history = str(_required(
        raw, context, "history_text", "prior_history_text",
        "history_diagnostic_text"))
    history_sha = str(_pick(
        raw, "history_sha256", "text_history_sha256",
        default=_sha_bytes(history.encode("utf-8"))))
    if history_sha != _sha_bytes(history.encode("utf-8")):
        raise AnalysisError(f"{context}: history SHA256 mismatch")
    history_source = str(_required(raw, context, "history_source", "history_policy"))
    history_answers_value = _pick(
        raw, "history_answers", "previous_answers", "prior_answers", default=None)
    if history_answers_value is None:
        raise AnalysisError(f"{context}: raw history answer provenance is missing")
    history_answers = _list_of_strings(history_answers_value, f"{context}/history_answers")
    source_ids = _list_of_strings(_pick(
        raw, "history_source_request_ids", "history_source_execution_ids",
        "prior_prediction_execution_ids", "prior_request_ids", default=[]),
        f"{context}/history_source_request_ids")

    first_token = _as_int(_required(
        raw, context, "first_token_id", "first_token"), f"{context}/first_token", minimum=0)
    input_tokens = _as_int(_required(
        raw, context, "input_token_count", "actual_input_tokens",
        "context_input_tokens", "total_context_tokens"),
        f"{context}/input_tokens", minimum=1)
    generated_tokens = _as_int(_required(
        raw, context, "generated_token_count", "generated_tokens"),
        f"{context}/generated_tokens", minimum=1)
    answer_tokens = _metric_int(
        raw, "answer_token_count", "answer_tokens", default=generated_tokens)
    ttft_value = _pick(raw, "end_to_end_ttft_ms", "ttft_ms", "TTFT_ms", default=None)
    if ttft_value is None:
        seconds = _required(raw, context, "ttft_s", "ttft")
        ttft_ms = _as_float(seconds, f"{context}/ttft_s", minimum=0.0) * 1e3
    else:
        ttft_ms = _as_float(ttft_value, f"{context}/ttft_ms", minimum=0.0)

    ssd_bytes = _pick(raw, "ssd_read_bytes", "total_actual_pread_bytes", default=None)
    if ssd_bytes is None:
        ssd_bytes = _io_nested(raw, "bytes", 0)
    preads = _pick(raw, "pread_count", "ssd_preads", default=None)
    if preads is None:
        preads = _io_nested(raw, "preads", 0)
    ssd_read_ms_value = _pick(raw, "ssd_read_ms", "ssd_read_latency_ms", default=None)
    if ssd_read_ms_value is None:
        ssd_read_ms_value = float(_io_nested(raw, "seconds", 0.0)) * 1e3

    selected = _selected_layers(
        _pick(raw, "selected_chunk_ids_per_layer", "selected_chunks_per_layer", default=None),
        f"{context}/selected_chunks")
    store_permutation = _pick(
        raw, "store_permutation_sha256", "permutation_sha256", default=None)
    strict = strict_exact_score(prediction, gold)
    claimed = _pick(raw, "strict_correct", default=None)
    if claimed is None and "correct" in raw:
        claimed = raw["correct"]
    if claimed is not None and float(claimed) != strict:
        raise AnalysisError(
            f"{context}: stored strict correctness {claimed!r} != recomputed {strict}")

    row = dict(raw)
    row.update({
        "analysis_schema_version": SCHEMA_VERSION,
        "protocol": protocol,
        "method_key": method,
        "method": METHOD_LABELS[method],
        "dialog_id": dialog_id,
        "image_id": image_id,
        "turn_id": turn,
        "question_id": question_id,
        "question": question,
        "gold": gold,
        "prediction": prediction,
        "strict_correct": strict,
        "quality_metric": "strict_normalized_exact_match",
        "logical_request_id": logical_id,
        "physical_execution_id": physical_id,
        "status_ok": bool(status_ok),
        "prompt": prompt,
        "prompt_sha256": prompt_sha,
        "history_text": history,
        "history_sha256": history_sha,
        "history_source": history_source,
        "history_answers": history_answers,
        "history_source_request_ids": source_ids,
        "first_token_id": first_token,
        "input_token_count": input_tokens,
        "generated_token_count": generated_tokens,
        "answer_token_count": answer_tokens,
        "ttft_ms": ttft_ms,
        "ssd_read_bytes": _as_int(ssd_bytes, f"{context}/ssd_bytes", minimum=0),
        "pread_count": _as_int(preads, f"{context}/preads", minimum=0),
        "ssd_read_latency_ms": _as_float(
            ssd_read_ms_value, f"{context}/ssd_read_ms", minimum=0.0),
        "probe_read_bytes": _metric_int(raw, "probe_read_bytes", default=0),
        "selected_kv_read_bytes": _metric_int(
            raw, "selected_chunk_payload_bytes", "selected_chunk_payload_read_bytes",
            "normal_kv_read_bytes", default=0),
        "separator_read_bytes": _metric_int(raw, "separator_read_bytes", default=0),
        "contiguous_runs_per_layer_mean": _metric_float(
            raw, "contiguous_runs_per_layer_mean", "mean_runs_per_layer", default=0.0),
        "selector_ms": _metric_float(
            raw, "selector_ms", "online_selector_total_ms", default=0.0),
        "n_raters": _metric_float(raw, "n_raters", "rater_count", default=0.0),
        "rater_ms": _metric_float(raw, "rater_ms", "rater_selection_ms", default=0.0),
        "projection_ms": _metric_float(raw, "projection_ms", "query_projection_ms", default=0.0),
        "probe_io_ms": _metric_float(raw, "probe_io_ms", default=0.0),
        "query_scoring_ms": _metric_float(raw, "query_scoring_ms", default=0.0),
        "chunk_aggregation_ms": _metric_float(raw, "chunk_aggregation_ms", default=0.0),
        "topk_chunk_ms": _metric_float(raw, "topk_chunk_ms", "topk_ms", default=0.0),
        "selected_chunk_ids_per_layer": selected,
        "store_permutation_sha256": (
            str(store_permutation) if store_permutation is not None else None),
        "source_artifact": str(source),
    })
    return row


def _artifact_rows(value: Mapping[str, Any], path: Path) -> list[tuple[Mapping[str, Any], str | None]]:
    direct = value.get("rows")
    if isinstance(direct, list):
        return [(row, None) for row in direct]
    output: list[tuple[Mapping[str, Any], str | None]] = []
    for container_name in ("protocol_rows", "protocols"):
        container = value.get(container_name)
        if isinstance(container, Mapping):
            for protocol, payload in container.items():
                rows = payload.get("rows") if isinstance(payload, Mapping) else payload
                if not isinstance(rows, list):
                    raise AnalysisError(f"{path}: {container_name}/{protocol} has no rows")
                output.extend((row, str(protocol)) for row in rows)
            return output
    for protocol in PROTOCOLS:
        payload = value.get(protocol)
        if payload is None:
            continue
        rows = payload.get("rows") if isinstance(payload, Mapping) else payload
        if not isinstance(rows, list):
            raise AnalysisError(f"{path}: {protocol} has no rows")
        output.extend((row, protocol) for row in rows)
    if output:
        return output
    raise AnalysisError(f"{path}: image artifact has no protocol rows")


def discover_artifacts(run_dir: Path) -> list[Path]:
    candidates = [
        run_dir / "images", run_dir / "image_artifacts",
        run_dir / "artifacts" / "images",
    ]
    populated: list[tuple[Path, list[Path]]] = []
    for directory in candidates:
        paths = sorted(directory.glob("*.json")) if directory.is_dir() else []
        if paths:
            populated.append((directory, paths))
    if not populated:
        raise AnalysisError(f"no per-image JSON artifacts under {run_dir}")
    if len(populated) > 1:
        descriptions = ", ".join(str(item[0]) for item in populated)
        raise AnalysisError(f"ambiguous per-image artifact directories: {descriptions}")
    return populated[0][1]


def load_rows(run_dir: Path, *, expected_protocol: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = discover_artifacts(run_dir)
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    image_ids: set[str] = set()
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise AnalysisError(f"artifact is not a regular file: {path}")
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AnalysisError(f"cannot read artifact {path}") from exc
        if not isinstance(artifact, Mapping):
            raise AnalysisError(f"artifact is not an object: {path}")
        claimed_hash = artifact.get("artifact_content_sha256")
        if claimed_hash is not None:
            body = {key: value for key, value in artifact.items()
                    if key != "artifact_content_sha256"}
            if str(claimed_hash) != _stable_json(body):
                raise AnalysisError(f"artifact content hash mismatch: {path}")
        hashes[path.name] = _sha_file(path)
        inherited_image = artifact.get("image_id")
        manifests = artifact.get("store_manifests", {})
        ours_manifest = manifests.get("image_only", {}) if isinstance(
            manifests, Mapping) else {}
        inherited_ours_permutation = (ours_manifest.get("permutation_sha256")
                                      if isinstance(ours_manifest, Mapping)
                                      else None)
        for ordinal, (raw, protocol) in enumerate(_artifact_rows(artifact, path), 1):
            enriched = dict(raw)
            method_value = enriched.get("method_key", enriched.get("method", ""))
            try:
                is_ours = _canonical_method(method_value) == "ours25"
            except AnalysisError:
                is_ours = False
            if is_ours and inherited_ours_permutation is not None:
                enriched.setdefault(
                    "store_permutation_sha256", inherited_ours_permutation)
            row = _canonical_row(enriched, path, ordinal, protocol)
            if expected_protocol is not None and row["protocol"] != expected_protocol:
                raise AnalysisError(
                    f"{path}: expected {expected_protocol}, observed {row['protocol']}")
            if inherited_image is not None and str(inherited_image) != row["image_id"]:
                raise AnalysisError(f"{path}: row image differs from artifact image")
            rows.append(row)
            image_ids.add(row["image_id"])
    return rows, {
        "artifact_count": len(paths),
        "artifact_sha256": hashes,
        "image_count": len(image_ids),
    }


def load_index(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise AnalysisError(f"index is not a regular file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        dialogs = payload.get("dialogues", payload.get("dialogs"))
        envelope = {key: value for key, value in payload.items()
                    if key not in {"dialogues", "dialogs"}}
    else:
        dialogs, envelope = payload, {}
    if not isinstance(dialogs, list) or not dialogs:
        raise AnalysisError("MT-GQA index contains no dialogues")
    output: dict[str, dict[str, Any]] = {}
    for dialog in dialogs:
        if not isinstance(dialog, Mapping):
            raise AnalysisError("index dialogue is not an object")
        did = str(dialog.get("dialog_id", dialog.get("dialogue_id", "")))
        turns = dialog.get("turns")
        if not did or did in output or not isinstance(turns, list) or len(turns) != 3:
            raise AnalysisError(f"invalid/duplicate index dialogue {did!r}")
        if [int(turn.get("turn_id", 0)) for turn in turns] != [1, 2, 3]:
            raise AnalysisError(f"{did}: invalid turn IDs")
        output[did] = dict(dialog)
    return output, {
        "path": str(path.resolve()), "sha256": _sha_file(path),
        "dialogues": len(output), "envelope": envelope,
    }


def render_history(dialog: Mapping[str, Any], turn_id: int,
                   prior_answers: Sequence[str]) -> str:
    if len(prior_answers) != turn_id - 1:
        raise ValueError("causal history requires exactly turn-1 prior answers")
    lines: list[str] = []
    for index, answer in enumerate(prior_answers, 1):
        turn = dialog["turns"][index - 1]
        lines.extend((f"Q{index}: {turn['question']}", f"A{index}: {answer}"))
    return "\n".join(lines)


def render_prompt(dialog: Mapping[str, Any], turn_id: int,
                  prior_answers: Sequence[str]) -> str:
    history = render_history(dialog, turn_id, prior_answers)
    current = dialog["turns"][turn_id - 1]
    body: list[str] = []
    if history:
        body.extend((history, ""))
    body.append(f"Current question Q{turn_id}: {current['question']}")
    body.append(f"{SHORT_ANSWER_INSTRUCTION} ASSISTANT:")
    return f"USER: {IMAGE_MARKER}\n" + "\n".join(body)


def _history_source_kind(value: str) -> str:
    key = re.sub(r"[^a-z]", "", str(value).lower())
    if key in {"none", "empty", "nohistory"}:
        return "none"
    if "gold" in key or "teacherforced" in key:
        return "gold"
    if "generated" in key or "prediction" in key or "samemethod" in key:
        return "generated"
    raise AnalysisError(f"unknown history source: {value!r}")


def validate_rows(rows: Sequence[dict[str, Any]], dialogs: Mapping[str, dict[str, Any]],
                  expected_dialogs: int, expected_images: int | None = None,
                  require_independent_executions: bool = True) -> tuple[dict[tuple[str, str, int, str], dict], dict]:
    expected_rows = int(expected_dialogs) * len(PROTOCOLS) * len(TURNS) * len(METHOD_KEYS)
    checks: dict[str, bool] = {}
    matrix: dict[tuple[str, str, int, str], dict] = {}
    logical_ids: set[str] = set()
    failures = 0
    for row in rows:
        key = (row["protocol"], row["dialog_id"], row["turn_id"], row["method_key"])
        if key in matrix:
            raise AnalysisError(f"duplicate logical matrix cell: {key}")
        matrix[key] = row
        if row["logical_request_id"] in logical_ids:
            raise AnalysisError(f"duplicate logical request ID: {row['logical_request_id']}")
        logical_ids.add(row["logical_request_id"])
        failures += int(not row["status_ok"])
    checks["exact_logical_row_count"] = len(rows) == expected_rows
    checks["zero_failed_requests"] = failures == 0
    checks["zero_duplicate_cells"] = len(matrix) == len(rows)
    physical_ids = [row["physical_execution_id"] for row in rows]
    checks["physical_execution_ids_independent"] = (
        not require_independent_executions
        or len(physical_ids) == len(set(physical_ids)))
    if not checks["exact_logical_row_count"]:
        raise AnalysisError(f"logical row count {len(rows)} != {expected_rows}")
    if failures:
        raise AnalysisError(f"run contains {failures} failed requests")

    observed_dialog_ids = {row["dialog_id"] for row in rows}
    if len(observed_dialog_ids) != expected_dialogs:
        raise AnalysisError(
            f"dialogue count {len(observed_dialog_ids)} != {expected_dialogs}")
    unknown = observed_dialog_ids - set(dialogs)
    if unknown:
        raise AnalysisError(f"rows contain unknown dialogue IDs: {sorted(unknown)[:5]}")
    expected_cells = {
        (protocol, did, turn, method)
        for protocol in PROTOCOLS for did in observed_dialog_ids
        for turn in TURNS for method in METHOD_KEYS
    }
    if set(matrix) != expected_cells:
        missing = sorted(expected_cells - set(matrix))[:5]
        extra = sorted(set(matrix) - expected_cells)[:5]
        raise AnalysisError(f"incomplete matrix; missing={missing}, extra={extra}")
    checks["complete_protocol_method_turn_matrix"] = True
    image_ids = {row["image_id"] for row in rows}
    if expected_images is not None and len(image_ids) != expected_images:
        raise AnalysisError(f"image count {len(image_ids)} != {expected_images}")
    checks["expected_dialogue_count"] = True
    checks["expected_image_count"] = expected_images is None or len(image_ids) == expected_images

    t1_matches = 0
    t1_shared = 0
    gold_causal = 0
    generated_causal = 0
    generated_provenance = 0
    for did in sorted(observed_dialog_ids):
        dialog = dialogs[did]
        image_id = str(dialog["image_id"])
        for protocol in PROTOCOLS:
            for method in METHOD_KEYS:
                prior_rows: list[dict] = []
                for turn in TURNS:
                    row = matrix[(protocol, did, turn, method)]
                    source_turn = dialog["turns"][turn - 1]
                    source_gold = _gold_list(source_turn.get(
                        "answers", source_turn.get("answer")))
                    if (row["image_id"] != image_id
                            or row["question_id"] != str(source_turn["question_id"])
                            or row["question"] != str(source_turn["question"])
                            or row["gold"] != source_gold):
                        raise AnalysisError(f"{protocol}/{did}/T{turn}/{method}: workload drift")
                    if protocol == "gold_history":
                        answers = [_gold_list(dialog["turns"][i].get(
                            "answers", dialog["turns"][i].get("answer")))[0]
                                   for i in range(turn - 1)]
                        expected_history = mt_gqa_prior_history_text(dialog, turn)
                        expected_prompt = mt_gqa_prompt(dialog, turn)
                        expected_source = "none" if turn == 1 else "gold"
                    else:
                        answers = [prior["prediction"] for prior in prior_rows]
                        expected_history = render_history(dialog, turn, answers)
                        expected_prompt = render_prompt(dialog, turn, answers)
                        expected_source = "none" if turn == 1 else "generated"
                    if row["history_answers"] != answers:
                        raise AnalysisError(
                            f"{protocol}/{did}/T{turn}/{method}: history answers are not causal")
                    if row["history_text"] != expected_history or row["prompt"] != expected_prompt:
                        raise AnalysisError(
                            f"{protocol}/{did}/T{turn}/{method}: prompt/history contamination")
                    actual_source = _history_source_kind(row["history_source"])
                    if turn == 1:
                        if actual_source not in {"none", "gold" if protocol == "gold_history" else "generated"}:
                            raise AnalysisError(f"{protocol}/{did}/T1/{method}: invalid empty-history source")
                    elif actual_source != expected_source:
                        raise AnalysisError(
                            f"{protocol}/{did}/T{turn}/{method}: wrong history source")
                    if protocol == "generated_history" and turn > 1:
                        refs = row["history_source_request_ids"]
                        if len(refs) != turn - 1:
                            raise AnalysisError(
                                f"generated/{did}/T{turn}/{method}: missing source request provenance")
                        for index, (ref, prior) in enumerate(zip(refs, prior_rows), 1):
                            allowed = {
                                prior["logical_request_id"], prior["physical_execution_id"],
                                str(_pick(prior, "request_id", "execution_id", default="")),
                            }
                            if ref not in allowed:
                                raise AnalysisError(
                                    f"generated/{did}/T{turn}/{method}: source {index} is not own prior row")
                        generated_provenance += 1
                    if protocol == "gold_history":
                        gold_causal += 1
                    else:
                        generated_causal += 1
                    prior_rows.append(row)
        for method in METHOD_KEYS:
            gold = matrix[("gold_history", did, 1, method)]
            generated = matrix[("generated_history", did, 1, method)]
            fields = ("prompt", "prompt_sha256", "history_text", "prediction", "first_token_id")
            if any(gold[field] != generated[field] for field in fields):
                raise AnalysisError(f"{did}/T1/{method}: cross-protocol identity failed")
            t1_matches += 1
            if gold["physical_execution_id"] == generated["physical_execution_id"]:
                t1_shared += 1
                if require_independent_executions:
                    raise AnalysisError(
                        f"{did}/T1/{method}: protocols reused one physical execution")

    checks.update({
        "gold_history_causal_and_exact": gold_causal == expected_dialogs * 3 * 4,
        "generated_history_same_method_causal_and_exact": generated_causal == expected_dialogs * 3 * 4,
        "generated_history_raw_provenance_exact": generated_provenance == expected_dialogs * 2 * 4,
        "future_leakage_zero": True,
        "cross_protocol_t1_prompt_prediction_first_token_100pct": t1_matches == expected_dialogs * 4,
        "cross_protocol_t1_physical_executions_independent": (
            not require_independent_executions or t1_shared == 0),
        "strict_score_recomputed": True,
    })
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise AnalysisError("validation checks failed: " + ", ".join(failed_checks))
    validation = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "checks": checks,
        "expected_logical_rows": expected_rows,
        "observed_logical_rows": len(rows),
        "expected_dialogues": expected_dialogs,
        "observed_dialogues": len(observed_dialog_ids),
        "observed_images": len(image_ids),
        "failed_requests": failures,
        "duplicate_cells": len(rows) - len(matrix),
        "t1_pairs_checked": t1_matches,
        "t1_shared_physical_pairs": t1_shared,
    }
    return matrix, validation


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        raise AnalysisError(f"cannot average empty rows for {key}")
    return float(statistics.fmean(float(row[key]) for row in rows))


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size), "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def quality_tables(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, dict[str, dict]], dict[str, list[dict]]]:
    summaries: dict[str, dict[str, dict]] = {}
    csv_rows: dict[str, list[dict]] = {}
    for protocol in PROTOCOLS:
        summaries[protocol] = {}
        csv_rows[protocol] = []
        for method in METHOD_KEYS:
            by_turn: dict[int, float] = {}
            counts: dict[int, int] = {}
            for turn in TURNS:
                selected = [row for row in rows if row["protocol"] == protocol
                            and row["method_key"] == method and row["turn_id"] == turn]
                accuracy = _mean(selected, "strict_correct")
                by_turn[turn], counts[turn] = accuracy, len(selected)
                csv_rows[protocol].append({
                    "method_key": method, "method": METHOD_LABELS[method],
                    "turn": turn, "n": len(selected),
                    "correct": int(sum(row["strict_correct"] for row in selected)),
                    "accuracy": accuracy, "accuracy_percent": accuracy * 100.0,
                })
            avg = statistics.fmean(by_turn.values())
            summaries[protocol][method] = {
                "method_key": method, "method": METHOD_LABELS[method],
                "acc1": by_turn[1], "acc2": by_turn[2], "acc3": by_turn[3],
                "avg": float(avg), "counts": counts,
            }
    return summaries, csv_rows


def latency_and_lengths(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    latency: dict[str, list[dict]] = {protocol: [] for protocol in PROTOCOLS}
    lengths: dict[str, list[dict]] = {protocol: [] for protocol in PROTOCOLS}
    for protocol in PROTOCOLS:
        for method in METHOD_KEYS:
            for population, turns in (("turn2", (2,)), ("turn3", (3,)),
                                      ("pooled_t2_t3", (2, 3))):
                selected = [row for row in rows if row["protocol"] == protocol
                            and row["method_key"] == method and row["turn_id"] in turns]
                ttft = _stats([row["ttft_ms"] for row in selected])
                inputs = _stats([row["input_token_count"] for row in selected])
                latency[protocol].append({
                    "method_key": method, "method": METHOD_LABELS[method],
                    "population": population, "turns": "+".join(map(str, turns)),
                    "n": ttft["n"], "ttft_mean_ms": ttft["mean"],
                    "ttft_p50_ms": ttft["p50"], "ttft_p95_ms": ttft["p95"],
                    "input_tokens_mean": inputs["mean"],
                    "input_tokens_p50": inputs["p50"],
                    "input_tokens_p95": inputs["p95"],
                })
            for turn in TURNS:
                selected = [row for row in rows if row["protocol"] == protocol
                            and row["method_key"] == method and row["turn_id"] == turn]
                generated = _stats([row["generated_token_count"] for row in selected])
                answer = _stats([row["answer_token_count"] for row in selected])
                inputs = _stats([row["input_token_count"] for row in selected])
                lengths[protocol].append({
                    "method_key": method, "method": METHOD_LABELS[method],
                    "turn": turn, "n": len(selected),
                    "input_tokens_mean": inputs["mean"],
                    "input_tokens_p50": inputs["p50"],
                    "input_tokens_p95": inputs["p95"],
                    "generated_tokens_mean": generated["mean"],
                    "generated_tokens_p50": generated["p50"],
                    "generated_tokens_p95": generated["p95"],
                    "answer_tokens_mean": answer["mean"],
                })
    return latency, lengths


IO_FIELDS = (
    "ssd_read_bytes", "probe_read_bytes", "selected_kv_read_bytes",
    "separator_read_bytes", "pread_count", "contiguous_runs_per_layer_mean",
    "ssd_read_latency_ms", "selector_ms", "n_raters", "rater_ms",
    "projection_ms", "probe_io_ms", "query_scoring_ms",
    "chunk_aggregation_ms", "topk_chunk_ms",
)


def io_tables(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {protocol: [] for protocol in PROTOCOLS}
    for protocol in PROTOCOLS:
        for method in METHOD_KEYS:
            for population, turns in (("turn2", (2,)), ("turn3", (3,)),
                                      ("pooled_t2_t3", (2, 3))):
                selected = [row for row in rows if row["protocol"] == protocol
                            and row["method_key"] == method and row["turn_id"] in turns]
                item: dict[str, Any] = {
                    "method_key": method, "method": METHOD_LABELS[method],
                    "population": population, "n": len(selected),
                }
                for field in IO_FIELDS:
                    item[field + "_mean"] = _mean(selected, field)
                for field in ("ssd_read_bytes", "probe_read_bytes",
                              "selected_kv_read_bytes", "separator_read_bytes"):
                    item[field.replace("_bytes", "_mb") + "_mean"] = (
                        item[field + "_mean"] / 1e6)
                output[protocol].append(item)
    return output


def _selection_set(row: Mapping[str, Any]) -> set[tuple[int, int]]:
    return {(layer, int(chunk))
            for layer, chunks in enumerate(row["selected_chunk_ids_per_layer"])
            for chunk in chunks}


def _jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def _selection_pair(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    a, b = _selection_set(left), _selection_set(right)
    if not a or not b:
        raise AnalysisError("cache-hit selection IDs are missing")
    return {
        "jaccard": _jaccard(a, b),
        "identical": a == b,
        "left_selected": len(a), "right_selected": len(b),
        "intersection": len(a & b), "union": len(a | b),
        "changed_chunk_count": len(a ^ b),
        "replacement_count": max(len(a - b), len(b - a)),
    }


def selection_analysis(matrix: Mapping[tuple[str, str, int, str], Mapping[str, Any]],
                       rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, bool]]:
    dialog_ids = sorted({row["dialog_id"] for row in rows})
    within: dict[str, Any] = {}
    for protocol in PROTOCOLS:
        pairs = []
        for did in dialog_ids:
            left = matrix[(protocol, did, 2, "qa_chunk25")]
            right = matrix[(protocol, did, 3, "qa_chunk25")]
            pair = _selection_pair(left, right)
            pair.update({"dialog_id": did, "image_id": left["image_id"],
                         "left_turn": 2, "right_turn": 3})
            pairs.append(pair)
        within[protocol] = {
            "n_pairs": len(pairs),
            "mean_jaccard": statistics.fmean(pair["jaccard"] for pair in pairs),
            "identical_selection_rate": statistics.fmean(
                float(pair["identical"]) for pair in pairs),
            "mean_changed_chunk_count": statistics.fmean(
                pair["changed_chunk_count"] for pair in pairs),
            "mean_replacement_count": statistics.fmean(
                pair["replacement_count"] for pair in pairs),
            "pairs": pairs,
        }
    cross: dict[str, Any] = {}
    for turn in (2, 3):
        pairs = []
        for did in dialog_ids:
            left = matrix[("gold_history", did, turn, "qa_chunk25")]
            right = matrix[("generated_history", did, turn, "qa_chunk25")]
            pair = _selection_pair(left, right)
            pair.update({"dialog_id": did, "image_id": left["image_id"], "turn": turn})
            pairs.append(pair)
        cross[f"turn{turn}"] = {
            "n_pairs": len(pairs),
            "mean_jaccard": statistics.fmean(pair["jaccard"] for pair in pairs),
            "identical_selection_rate": statistics.fmean(
                float(pair["identical"]) for pair in pairs),
            "mean_changed_chunk_count": statistics.fmean(
                pair["changed_chunk_count"] for pair in pairs),
            "pairs": pairs,
        }

    ours_by_image: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["method_key"] == "ours25" and row["turn_id"] in (2, 3):
            ours_by_image[row["image_id"]].append(row)
    ours_images: dict[str, Any] = {}
    ours_invariant = True
    ours_prefix = True
    ours_permutation_invariant = True
    for image_id, selected in sorted(ours_by_image.items()):
        hashes = {_stable_json(row["selected_chunk_ids_per_layer"]) for row in selected}
        permutations = {row.get("store_permutation_sha256") for row in selected}
        invariant = len(hashes) == 1 and all(row["selected_chunk_ids_per_layer"] for row in selected)
        permutation_invariant = len(permutations) == 1 and None not in permutations
        prefix = invariant and all(
            chunks == list(range(len(chunks)))
            for chunks in selected[0]["selected_chunk_ids_per_layer"])
        ours_images[image_id] = {
            "requests": len(selected), "invariant": bool(invariant),
            "storage_permutation_invariant": bool(permutation_invariant),
            "exact_first_k_prefix": bool(prefix),
            "selection_sha256": next(iter(hashes)) if len(hashes) == 1 else None,
            "store_permutation_sha256": (
                next(iter(permutations)) if permutation_invariant else None),
        }
        ours_invariant &= bool(invariant)
        ours_prefix &= bool(prefix)
        ours_permutation_invariant &= bool(permutation_invariant)

    budget_equal = True
    fixed_quarter_budget = True
    qa_configuration_locked = True
    ours_configuration_locked = True
    for row in rows:
        if row["turn_id"] not in (2, 3) or row["method_key"] not in {
                "qa_chunk25", "ours25"}:
            continue
        counts = [len(items) for items in row["selected_chunk_ids_per_layer"]]
        total_value = _pick(row, "n_chunks_total", default=None)
        ratio_value = _pick(
            row, "retention_ratio", "nominal_retention_ratio", "budget",
            default=None)
        if total_value is None or ratio_value is None:
            fixed_quarter_budget = False
        else:
            total = int(total_value)
            expected_k = max(1, min(total, round(total * 0.25)))
            fixed_quarter_budget &= (
                float(ratio_value) == 0.25 and bool(counts)
                and all(count == expected_k for count in counts))
        if row["method_key"] == "ours25":
            ours_configuration_locked &= (
                str(_pick(row, "physical_layout", default=""))
                == "visionzip_image_only"
                and int(_pick(row, "query_score_calls", default=0)) == 0
                and int(_pick(row, "static_score_calls", default=0)) == 0)
            continue
        qa_configuration_locked &= (
            str(_pick(row, "physical_layout", default="")) == "raster"
            and str(_pick(row, "chunk_score", default=""))
            == "mean_valid_spatial_token_importance"
            and str(_pick(row, "head_reduce", default="")) == "mean"
            and str(_pick(row, "rater_algorithm_id", default=""))
            == QA_RATER_ALGORITHM_ID
            and int(_pick(row, "probe_heads_used", default=-1)) == 3
            and float(_pick(row, "fallback_rate", default=-1.0)) == 0.0
            and _pick(row, "adaptive_ratio", default=None) is False)
        ours = matrix[(row["protocol"], row["dialog_id"], row["turn_id"], "ours25")]
        qa_counts = counts
        ours_counts = [len(items) for items in ours["selected_chunk_ids_per_layer"]]
        if not qa_counts or qa_counts != ours_counts:
            budget_equal = False
            break
    checks = {
        "qa_cache_hit_selection_present": all(
            row["selected_chunk_ids_per_layer"] for row in rows
            if row["method_key"] == "qa_chunk25" and row["turn_id"] in (2, 3)),
        "ours_cache_hit_selection_present": all(
            row["selected_chunk_ids_per_layer"] for row in rows
            if row["method_key"] == "ours25" and row["turn_id"] in (2, 3)),
        "qa_ours_normal_chunk_budget_equal": budget_equal,
        "qa_ours_fixed_25pct_budget_exact": fixed_quarter_budget,
        "qa_chunk_validated_configuration_unchanged": qa_configuration_locked,
        "ours_validated_configuration_unchanged": ours_configuration_locked,
        "ours_selection_invariant_across_protocol_query_turn": ours_invariant,
        "ours_storage_permutation_invariant_across_protocol": (
            ours_permutation_invariant),
        "ours_selection_is_exact_first_k_prefix": ours_prefix,
    }
    if not all(checks.values()):
        raise AnalysisError("selection validation failed: " + ", ".join(
            name for name, passed in checks.items() if not passed))
    return {
        "schema_version": SCHEMA_VERSION,
        "qa_t2_t3_within_protocol": within,
        "qa_gold_vs_generated_same_turn": cross,
        "ours_per_image_invariance": ours_images,
        "ours_all_images_invariant": ours_invariant,
        "ours_all_images_storage_permutation_invariant": (
            ours_permutation_invariant),
        "qa_ours_normal_chunk_budget_equal": budget_equal,
    }, checks


def error_propagation(matrix: Mapping[tuple[str, str, int, str], Mapping[str, Any]],
                      dialog_ids: Sequence[str]) -> tuple[list[dict], dict[str, Any]]:
    csv_rows: list[dict] = []
    payload: dict[str, Any] = {}
    for method in METHOD_KEYS:
        triples = [[int(matrix[("generated_history", did, turn, method)]["strict_correct"])
                    for turn in TURNS] for did in dialog_ids]
        pattern_counts = Counter((a, b) for a, b, _ in triples)
        method_payload: dict[str, Any] = {
            "t1_t2_patterns": {
                f"t1_{'correct' if a else 'wrong'}__t2_{'correct' if b else 'wrong'}":
                    pattern_counts[(a, b)]
                for a in (1, 0) for b in (1, 0)
            }
        }
        for a in (1, 0):
            for b in (1, 0):
                label = f"T1{'C' if a else 'W'}_T2{'C' if b else 'W'}"
                count = pattern_counts[(a, b)]
                csv_rows.append({
                    "method_key": method, "method": METHOD_LABELS[method],
                    "metric": "t1_t2_joint_correctness_count",
                    "condition": label, "n": len(triples),
                    "correct": count, "accuracy": count / len(triples),
                })
        for a in (1, 0):
            group = [b for x, b, _ in triples if x == a]
            accuracy = statistics.fmean(group) if group else None
            name = f"t2_accuracy_given_t1_{'correct' if a else 'wrong'}"
            method_payload[name] = {"n": len(group), "accuracy": accuracy}
            csv_rows.append({
                "method_key": method, "method": METHOD_LABELS[method],
                "metric": name, "condition": f"T1={'correct' if a else 'wrong'}",
                "n": len(group), "correct": sum(group), "accuracy": accuracy,
            })
        t3_patterns: dict[str, Any] = {}
        for a in (1, 0):
            for b in (1, 0):
                group = [c for x, y, c in triples if x == a and y == b]
                key = ("C" if a else "W") + ("C" if b else "W")
                t3_patterns[key] = {
                    "n": len(group), "correct": sum(group),
                    "accuracy": statistics.fmean(group) if group else None,
                }
                csv_rows.append({
                    "method_key": method, "method": METHOD_LABELS[method],
                    "metric": "t3_accuracy_by_prior_pattern", "condition": key,
                    **t3_patterns[key],
                })
        both = [c for a, b, c in triples if a and b]
        any_wrong = [c for a, b, c in triples if not (a and b)]
        method_payload["t3_by_previous_correctness"] = {
            "both_correct": {"n": len(both), "correct": sum(both),
                             "accuracy": statistics.fmean(both) if both else None},
            "at_least_one_wrong": {"n": len(any_wrong), "correct": sum(any_wrong),
                                   "accuracy": statistics.fmean(any_wrong) if any_wrong else None},
        }
        method_payload["t3_prior_patterns"] = t3_patterns
        for label, group in (("both_correct", both), ("at_least_one_wrong", any_wrong)):
            csv_rows.append({
                "method_key": method, "method": METHOD_LABELS[method],
                "metric": "t3_accuracy_by_previous_correctness", "condition": label,
                "n": len(group), "correct": sum(group),
                "accuracy": statistics.fmean(group) if group else None,
            })
        payload[method] = method_payload
    return csv_rows, payload


def _bootstrap_difference(differences: np.ndarray, resamples: int = BOOTSTRAP_RESAMPLES,
                          seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    if differences.ndim != 2 or differences.shape[1] != 3 or differences.shape[0] < 1:
        raise ValueError("bootstrap differences must have shape (dialogues, 3)")
    rng = np.random.default_rng(seed)
    n = differences.shape[0]
    turn_samples = np.empty((resamples, 3), dtype=np.float64)
    avg_samples = np.empty(resamples, dtype=np.float64)
    batch = 128
    for start in range(0, resamples, batch):
        stop = min(resamples, start + batch)
        indexes = rng.integers(0, n, size=(stop - start, n), endpoint=False)
        sampled = differences[indexes].mean(axis=1)
        turn_samples[start:stop] = sampled
        avg_samples[start:stop] = sampled.mean(axis=1)
    point = differences.mean(axis=0)
    return {
        "resamples": int(resamples), "seed": int(seed),
        "cluster_unit": "dialogue",
        "turns": {
            f"acc{turn}": {
                "difference": float(point[turn - 1]),
                "ci95_low": float(np.percentile(turn_samples[:, turn - 1], 2.5)),
                "ci95_high": float(np.percentile(turn_samples[:, turn - 1], 97.5)),
            } for turn in TURNS
        },
        "avg": {
            "difference": float(point.mean()),
            "ci95_low": float(np.percentile(avg_samples, 2.5)),
            "ci95_high": float(np.percentile(avg_samples, 97.5)),
        },
    }


def paired_quality(matrix: Mapping[tuple[str, str, int, str], Mapping[str, Any]],
                   dialog_ids: Sequence[str], resamples: int = BOOTSTRAP_RESAMPLES,
                   seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "comparison": "QA-Chunk25 minus Ours25",
    }
    for protocol in PROTOCOLS:
        contingencies: dict[str, Any] = {}
        differences = np.empty((len(dialog_ids), 3), dtype=np.float64)
        for turn in TURNS:
            cells = Counter()
            for index, did in enumerate(dialog_ids):
                qa = int(matrix[(protocol, did, turn, "qa_chunk25")]["strict_correct"])
                ours = int(matrix[(protocol, did, turn, "ours25")]["strict_correct"])
                differences[index, turn - 1] = qa - ours
                cells[(qa, ours)] += 1
            contingencies[f"turn{turn}"] = {
                "both_correct": cells[(1, 1)],
                "qa_only_correct": cells[(1, 0)],
                "ours_only_correct": cells[(0, 1)],
                "both_wrong": cells[(0, 0)],
                "n": len(dialog_ids),
                "qa_minus_ours": float(differences[:, turn - 1].mean()),
                "mcnemar_exact_two_sided_p": exact_mcnemar_pvalue(
                    cells[(1, 0)], cells[(0, 1)]),
            }
        output[protocol] = {
            "per_turn": contingencies,
            "dialogue_cluster_bootstrap_95ci": _bootstrap_difference(
                differences, resamples=resamples, seed=seed),
        }
    return output


def history_comparison(quality: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> list[dict]:
    rows: list[dict] = []
    for method in METHOD_KEYS:
        for metric in ("acc1", "acc2", "acc3", "avg"):
            gold = float(quality["gold_history"][method][metric])
            generated = float(quality["generated_history"][method][metric])
            rows.append({
                "method_key": method, "method": METHOD_LABELS[method],
                "metric": metric, "gold": gold, "generated": generated,
                "delta_generated_minus_gold": generated - gold,
                "gold_percent": gold * 100.0,
                "generated_percent": generated * 100.0,
                "delta_percentage_points": (generated - gold) * 100.0,
            })
    return rows


def _summary_rows(quality: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, list[dict]]:
    return {protocol: [quality[protocol][method] for method in METHOD_KEYS]
            for protocol in PROTOCOLS}


def _lookup(rows: Sequence[Mapping[str, Any]], method: str, population: str) -> Mapping[str, Any]:
    matches = [row for row in rows if row["method_key"] == method
               and row["population"] == population]
    if len(matches) != 1:
        raise AnalysisError(f"missing summary {method}/{population}")
    return matches[0]


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _pct_or_na(value: Any) -> str:
    return "N/A" if value is None else _pct(float(value))


def _num(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def build_analysis_markdown(quality: Mapping[str, Mapping[str, Mapping[str, Any]]],
                            comparisons: Sequence[Mapping[str, Any]],
                            latency: Mapping[str, Sequence[Mapping[str, Any]]],
                            lengths: Mapping[str, Sequence[Mapping[str, Any]]],
                            io_rows: Mapping[str, Sequence[Mapping[str, Any]]],
                            selection: Mapping[str, Any],
                            paired: Mapping[str, Any],
                            propagation: Mapping[str, Any],
                            validation: Mapping[str, Any],
                            index_info: Mapping[str, Any]) -> str:
    lines = ["# MT-GQA Gold + Generated History — Four-Arm Evaluation", ""]
    for protocol, title in (("gold_history", "Gold-History"),
                            ("generated_history", "Generated-History")):
        lines.extend([
            f"## {title}", "",
            "| Method | Acc1 | Acc2 | Acc3 | Avg |",
            "|---|---:|---:|---:|---:|",
        ])
        for method in METHOD_KEYS:
            item = quality[protocol][method]
            lines.append(
                f"| {METHOD_LABELS[method]} | {_pct(item['acc1'])} | "
                f"{_pct(item['acc2'])} | {_pct(item['acc3'])} | {_pct(item['avg'])} |")
        lines.append("")

    lines.extend([
        "## Gold vs Generated", "",
        "| Method | Gold Avg | Generated Avg | Δ Avg | Gold Acc3 | Generated Acc3 | Δ Acc3 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for method in METHOD_KEYS:
        gold = quality["gold_history"][method]
        generated = quality["generated_history"][method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {_pct(gold['avg'])} | {_pct(generated['avg'])} | "
            f"{(generated['avg'] - gold['avg']) * 100:+.2f} pp | {_pct(gold['acc3'])} | "
            f"{_pct(generated['acc3'])} | {(generated['acc3'] - gold['acc3']) * 100:+.2f} pp |")
    lines.append("")

    # The user-facing contract requires Gold, Generated, then Gold-vs-Generated
    # as the first three tables.  Token-length diagnostics intentionally start
    # only after those three paper-style quality tables.
    lines.extend(["## Input/output token lengths", ""])
    for protocol, title in (("gold_history", "Gold-History"),
                            ("generated_history", "Generated-History")):
        lines.extend([
            f"### {title}", "",
            "| Method | T1 generated | T2 generated | T2 input | T3 input |",
            "|---|---:|---:|---:|---:|",
        ])
        for method in METHOD_KEYS:
            by_turn = {int(row["turn"]): row for row in lengths[protocol]
                       if row["method_key"] == method}
            lines.append(
                f"| {METHOD_LABELS[method]} | "
                f"{by_turn[1]['generated_tokens_mean']:.2f} | "
                f"{by_turn[2]['generated_tokens_mean']:.2f} | "
                f"{by_turn[2]['input_tokens_mean']:.2f} | "
                f"{by_turn[3]['input_tokens_mean']:.2f} |")
        lines.append("")
    generated_t3_inputs = {
        METHOD_LABELS[method]: next(
            row["input_tokens_mean"] for row in lengths["generated_history"]
            if row["method_key"] == method and int(row["turn"]) == 3)
        for method in METHOD_KEYS
    }
    lines.extend([
        "Generated-History T3 input lengths are method-specific because each method propagates "
        "its own decoded responses: "
        + ", ".join(f"{method} {value:.2f}" for method, value in generated_t3_inputs.items())
        + " tokens on average. TTFT differences therefore combine Visual-KV path effects with "
        "history-length effects; the report does not conflate the two.", "",
    ])

    for protocol, title in (("gold_history", "Gold-History system performance"),
                            ("generated_history", "Generated-History system performance")):
        lines.extend([
            f"## {title}", "",
            "Main population is cache-hit Turns 2–3. TTFT includes prompt/token preparation, "
            "H2D, online selection, SSD I/O, scatter, prefill, and the synchronized first-token decision.", "",
            "| Method | T2 TTFT mean (p50/p95) | T3 TTFT mean (p50/p95) | "
            "T2–3 mean (p50/p95) | SSD MB | Selector ms | Preads |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for method in METHOD_KEYS:
            t2 = _lookup(latency[protocol], method, "turn2")
            t3 = _lookup(latency[protocol], method, "turn3")
            pooled = _lookup(latency[protocol], method, "pooled_t2_t3")
            io_pool = _lookup(io_rows[protocol], method, "pooled_t2_t3")
            lines.append(
                f"| {METHOD_LABELS[method]} | {_num(t2['ttft_mean_ms'])} "
                f"({_num(t2['ttft_p50_ms'])}/{_num(t2['ttft_p95_ms'])}) | "
                f"{_num(t3['ttft_mean_ms'])} "
                f"({_num(t3['ttft_p50_ms'])}/{_num(t3['ttft_p95_ms'])}) | "
                f"{_num(pooled['ttft_mean_ms'])} "
                f"({_num(pooled['ttft_p50_ms'])}/{_num(pooled['ttft_p95_ms'])}) | "
                f"{_num(io_pool['ssd_read_mb_mean'])} | {_num(io_pool['selector_ms_mean'])} | "
                f"{_num(io_pool['pread_count_mean'])} |")
        lines.append("")

    gold_gap = {metric: quality["gold_history"]["qa_chunk25"][metric]
                - quality["gold_history"]["ours25"][metric]
                for metric in ("acc1", "acc2", "acc3", "avg")}
    gen_gap = {metric: quality["generated_history"]["qa_chunk25"][metric]
               - quality["generated_history"]["ours25"][metric]
               for metric in ("acc1", "acc2", "acc3", "avg")}
    qa_gold_io = _lookup(io_rows["gold_history"], "qa_chunk25", "pooled_t2_t3")
    ours_gold_io = _lookup(io_rows["gold_history"], "ours25", "pooled_t2_t3")
    qa_gold_lat = _lookup(latency["gold_history"], "qa_chunk25", "pooled_t2_t3")
    ours_gold_lat = _lookup(latency["gold_history"], "ours25", "pooled_t2_t3")
    qa_gen_io = _lookup(io_rows["generated_history"], "qa_chunk25", "pooled_t2_t3")
    ours_gen_io = _lookup(io_rows["generated_history"], "ours25", "pooled_t2_t3")
    qa_gen_lat = _lookup(latency["generated_history"], "qa_chunk25", "pooled_t2_t3")
    ours_gen_lat = _lookup(latency["generated_history"], "ours25", "pooled_t2_t3")
    gold_j = selection["qa_t2_t3_within_protocol"]["gold_history"]
    gen_j = selection["qa_t2_t3_within_protocol"]["generated_history"]

    lines.extend(["## Detailed cache-hit I/O and selector costs", ""])
    for protocol, title in (("gold_history", "Gold-History"),
                            ("generated_history", "Generated-History")):
        lines.extend([
            f"### {title} pooled T2–T3 I/O", "",
            "| Method | SSD MB | Probe MB | Selected KV MB | Separator MB | "
            "Preads | Runs/layer | SSD read ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for method in METHOD_KEYS:
            item = _lookup(io_rows[protocol], method, "pooled_t2_t3")
            lines.append(
                f"| {METHOD_LABELS[method]} | {_num(item['ssd_read_mb_mean'])} | "
                f"{_num(item['probe_read_mb_mean'])} | "
                f"{_num(item['selected_kv_read_mb_mean'])} | "
                f"{_num(item['separator_read_mb_mean'])} | "
                f"{_num(item['pread_count_mean'])} | "
                f"{_num(item['contiguous_runs_per_layer_mean_mean'])} | "
                f"{_num(item['ssd_read_latency_ms_mean'])} |")
        lines.extend([
            "", f"### {title} QA-Chunk25 selector by turn", "",
            "| Turn | Raters | Rater ms | Projection ms | Probe I/O ms | "
            "Query score ms | Chunk aggregation ms | Top-k ms | Selector ms |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for population, turn in (("turn2", 2), ("turn3", 3)):
            item = _lookup(io_rows[protocol], "qa_chunk25", population)
            lines.append(
                f"| {turn} | {_num(item['n_raters_mean'])} | "
                f"{_num(item['rater_ms_mean'])} | "
                f"{_num(item['projection_ms_mean'])} | "
                f"{_num(item['probe_io_ms_mean'])} | "
                f"{_num(item['query_scoring_ms_mean'])} | "
                f"{_num(item['chunk_aggregation_ms_mean'])} | "
                f"{_num(item['topk_chunk_ms_mean'])} | "
                f"{_num(item['selector_ms_mean'])} |")
        lines.append("")

    lines.extend([
        "## Generated-minus-Gold degradation", "",
        "| Method | ΔAcc2 | ΔAcc3 | ΔAvg |",
        "|---|---:|---:|---:|",
    ])
    history_deltas: dict[str, dict[str, float]] = {}
    for method in METHOD_KEYS:
        gold = quality["gold_history"][method]
        generated = quality["generated_history"][method]
        history_deltas[method] = {
            metric: float(generated[metric] - gold[metric])
            for metric in ("acc2", "acc3", "avg")
        }
        delta = history_deltas[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {delta['acc2']*100:+.2f} pp | "
            f"{delta['acc3']*100:+.2f} pp | {delta['avg']*100:+.2f} pp |")
    lines.append("")

    lines.extend(["## Paired QA-Chunk25 vs Ours25", ""])
    for protocol, title in (("gold_history", "Gold-History"),
                            ("generated_history", "Generated-History")):
        lines.extend([
            f"### {title}", "",
            "| Turn | Both correct | QA only | Ours only | Both wrong | "
            "QA−Ours | Exact McNemar p | Dialogue-bootstrap 95% CI |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        protocol_paired = paired[protocol]
        bootstrap = protocol_paired["dialogue_cluster_bootstrap_95ci"]
        for turn in TURNS:
            cells = protocol_paired["per_turn"][f"turn{turn}"]
            ci = bootstrap["turns"][f"acc{turn}"]
            lines.append(
                f"| {turn} | {cells['both_correct']} | "
                f"{cells['qa_only_correct']} | {cells['ours_only_correct']} | "
                f"{cells['both_wrong']} | {cells['qa_minus_ours']*100:+.2f} pp | "
                f"{cells['mcnemar_exact_two_sided_p']:.6g} | "
                f"[{ci['ci95_low']*100:+.2f}, {ci['ci95_high']*100:+.2f}] pp |")
        avg_ci = bootstrap["avg"]
        lines.append(
            f"| Avg | — | — | — | — | {avg_ci['difference']*100:+.2f} pp | — | "
            f"[{avg_ci['ci95_low']*100:+.2f}, {avg_ci['ci95_high']*100:+.2f}] pp |")
        lines.append("")

    lines.extend([
        "## Generated-history error propagation", "",
        "| Method | T1C/T2C | T1C/T2W | T1W/T2C | T1W/T2W |",
        "|---|---:|---:|---:|---:|",
    ])
    for method in METHOD_KEYS:
        patterns = propagation[method]["t1_t2_patterns"]
        lines.append(
            f"| {METHOD_LABELS[method]} | "
            f"{patterns['t1_correct__t2_correct']} | "
            f"{patterns['t1_correct__t2_wrong']} | "
            f"{patterns['t1_wrong__t2_correct']} | "
            f"{patterns['t1_wrong__t2_wrong']} |")
    lines.extend([
        "", "T3 accuracy for each exact prior-turn correctness pattern:", "",
        "| Method | CC | CW | WC | WW |",
        "|---|---:|---:|---:|---:|",
    ])
    for method in METHOD_KEYS:
        patterns = propagation[method]["t3_prior_patterns"]
        lines.append(
            f"| {METHOD_LABELS[method]} | "
            + " | ".join(
                f"{_pct_or_na(patterns[key]['accuracy'])} ({patterns[key]['n']})"
                for key in ("CC", "CW", "WC", "WW"))
            + " |")
    lines.extend([
        "", "Conditional summaries:", "",
        "| Method | T2 given T1 correct (n) | T2 given T1 wrong (n) | "
        "T3 given T1,T2 correct (n) | T3 given prior error (n) |",
        "|---|---:|---:|---:|---:|",
    ])
    for method in METHOD_KEYS:
        item = propagation[method]
        t2c = item["t2_accuracy_given_t1_correct"]
        t2w = item["t2_accuracy_given_t1_wrong"]
        t3c = item["t3_by_previous_correctness"]["both_correct"]
        t3w = item["t3_by_previous_correctness"]["at_least_one_wrong"]
        lines.append(
            f"| {METHOD_LABELS[method]} | {_pct_or_na(t2c['accuracy'])} ({t2c['n']}) | "
            f"{_pct_or_na(t2w['accuracy'])} ({t2w['n']}) | "
            f"{_pct_or_na(t3c['accuracy'])} ({t3c['n']}) | "
            f"{_pct_or_na(t3w['accuracy'])} ({t3w['n']}) |")
    lines.append("")

    relative_avg_propagation = (
        history_deltas["ours25"]["avg"]
        - history_deltas["qa_chunk25"]["avg"])
    if relative_avg_propagation < 0:
        propagation_answer = (
            f"Ours changes {abs(relative_avg_propagation)*100:.2f} pp more "
            "negatively than QA in Avg")
    elif relative_avg_propagation > 0:
        propagation_answer = (
            f"Ours changes {relative_avg_propagation*100:.2f} pp more "
            "favorably than QA in Avg")
    else:
        propagation_answer = "Ours and QA have the same Generated−Gold Avg change"

    lines.extend([
        "## Direct answers to Q1–Q8", "",
        "### Q1 — Gold-history QA-Chunk25 vs Ours25", "",
        "The QA−Ours gaps for Acc1/Acc2/Acc3/Avg are "
        + "/".join(f"{gold_gap[key] * 100:+.2f} pp" for key in ("acc1", "acc2", "acc3", "avg")) + ".", "",
        "### Q2 — Generated-history QA-Chunk25 vs Ours25", "",
        "The QA−Ours gaps for Acc1/Acc2/Acc3/Avg are "
        + "/".join(f"{gen_gap[key] * 100:+.2f} pp" for key in ("acc1", "acc2", "acc3", "avg")) + ".", "",
        "### Q3 — Generated minus Gold", "",
    ])
    for method in METHOD_KEYS:
        gold, generated = quality["gold_history"][method], quality["generated_history"][method]
        lines.append(
            f"- {METHOD_LABELS[method]}: ΔAcc2 {100*(generated['acc2']-gold['acc2']):+.2f} pp, "
            f"ΔAcc3 {100*(generated['acc3']-gold['acc3']):+.2f} pp, "
            f"ΔAvg {100*(generated['avg']-gold['avg']):+.2f} pp.")
    lines.extend([
        "", "### Q4 — Method-specific error propagation", "",
        propagation_answer + ". The full ΔAcc2/ΔAcc3/ΔAvg and conditional T2/T3 tables above "
        "show the method-specific pattern. This is a descriptive association under each method's "
        "own generated history, not an equivalence test or a causal estimate.", "",
        "### Q5 — Does the adaptive-selection quality gain persist?", "",
        f"QA−Ours Avg is {gold_gap['avg']*100:+.2f} pp with Gold history and "
        f"{gen_gap['avg']*100:+.2f} pp with Generated history. Per-turn paired cells, exact "
        "McNemar p-values, and dialogue-cluster bootstrap CIs are in `paired_quality.json`.", "",
        "### Q6 — Cost of that gain", "",
        f"In Gold T2–3, QA−Ours mean TTFT is "
        f"{qa_gold_lat['ttft_mean_ms']-ours_gold_lat['ttft_mean_ms']:+.2f} ms, selector cost is "
        f"{qa_gold_io['selector_ms_mean']-ours_gold_io['selector_ms_mean']:+.2f} ms, SSD traffic is "
        f"{qa_gold_io['ssd_read_mb_mean']-ours_gold_io['ssd_read_mb_mean']:+.2f} MB/request, and "
        f"preads differ by {qa_gold_io['pread_count_mean']-ours_gold_io['pread_count_mean']:+.2f}/request.", "",
        f"In Generated T2–3, QA−Ours mean TTFT is "
        f"{qa_gen_lat['ttft_mean_ms']-ours_gen_lat['ttft_mean_ms']:+.2f} ms, selector cost is "
        f"{qa_gen_io['selector_ms_mean']-ours_gen_io['selector_ms_mean']:+.2f} ms, SSD traffic is "
        f"{qa_gen_io['ssd_read_mb_mean']-ours_gen_io['ssd_read_mb_mean']:+.2f} MB/request, and "
        f"preads differ by {qa_gen_io['pread_count_mean']-ours_gen_io['pread_count_mean']:+.2f}/request.", "",
        "### Q7 — QA T2↔T3 selection", "",
        f"Mean layer-tagged chunk Jaccard is {gold_j['mean_jaccard']:.6f} for Gold and "
        f"{gen_j['mean_jaccard']:.6f} for Generated history (difference "
        f"{gen_j['mean_jaccard']-gold_j['mean_jaccard']:+.6f}). Cross-protocol same-turn "
        "comparisons are recorded in `selection_analysis.json`.", "",
        "### Q8 — Ours fixed-prefix invariance", "",
        "Ours selected physical prefix IDs are identical across Gold/Generated, T2/T3, and all "
        f"queries for the same image: **{'YES' if selection['ours_all_images_invariant'] else 'NO'}**. "
        "The underlying image-only storage permutation is also identical across protocols: "
        f"**{'YES' if selection['ours_all_images_storage_permutation_invariant'] else 'NO'}**.", "",
        "## Paired quality and error propagation", "",
        f"Exact paired cells and McNemar tests use {paired['gold_history']['per_turn']['turn1']['n']:,} "
        "dialogues per turn. The bootstrap uses 10,000 dialogue-cluster resamples with seed 1234. "
        "Conditional error-propagation counts and accuracies are in `error_propagation.csv`.", "",
        "## Protocol interpretation", "",
        "Gold-History is a controlled multi-turn evaluation that separates Visual-KV retrieval "
        "effects from prior-generation error propagation. Generated-History propagates each "
        "method's own previous outputs and captures realistic conversational error propagation. "
        "They answer different questions; neither is labeled the uniquely correct metric.", "",
        "## Limitations", "",
        "- This is MT-GQA-reconstructed with MetaCompress-compatible Acc1/Acc2/Acc3/Avg reporting; "
        "exact MetaCompress protocol identity is not claimed.",
        "- Dialogue-cluster bootstrap follows the requested unit and does not additionally cluster "
        "dialogues that share an image.",
        "- OS page cache is conditioned, but SSD controller cache is not flushed.",
        "- FullLoad uses persisted FP16 Visual-KV and the stored-KV manual decode path, while "
        "ReComp uses the ordinary pixel/HF generation path; no byte-identical or output-"
        "equivalence claim is made between them.",
        "- No equivalence claim is made; the report provides differences, CIs, and McNemar tests.", "",
        "## Completion", "",
        "```text",
        "IMPLEMENTATION: PASS",
        "GOLD-HISTORY RUN: PASS",
        "GENERATED-HISTORY RUN: PASS",
        "VALIDATION: PASS",
        "```", "",
        f"- Workload index: `{index_info['sha256']}`",
        f"- Logical rows: {validation['observed_logical_rows']:,}",
        f"- Failed requests: {validation['failed_requests']}",
        f"- Duplicate cells: {validation['duplicate_cells']}", "",
        "MT-GQA GOLD + GENERATED HISTORY 4-ARM EVALUATION VALIDATED: YES", "",
    ])
    return "\n".join(lines)


def _write_raw_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_gzip(path: Path, source: Path) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite result: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with source.open("rb") as src, gzip.open(temporary, "wb", compresslevel=6) as dst:
            while True:
                block = src.read(1024 * 1024)
                if not block:
                    break
                dst.write(block)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_protection_validation(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise AnalysisError(f"protection validation is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read protection validation: {path}") from exc
    if not isinstance(value, Mapping) or value.get("passed") is not True:
        raise AnalysisError("prior-artifact protection validation did not pass")
    for field in ("missing_paths", "changed_paths"):
        if value.get(field) != []:
            raise AnalysisError(f"protection validation has nonempty {field}")
    if not isinstance(value.get("added_paths", []), list):
        raise AnalysisError("protection validation added_paths is malformed")
    if value.get("before_manifest_sha256") != value.get("after_manifest_sha256"):
        raise AnalysisError("protected-artifact before/after fingerprints differ")
    return {**dict(value), "path": str(path), "sha256": _sha_file(path)}


def _manual_history_samples(
    rows: Sequence[Mapping[str, Any]], *, seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """One deterministic T2 and T3 sample for every protocol/method cell."""
    samples: list[dict[str, Any]] = []
    for protocol in PROTOCOLS:
        for method in METHOD_KEYS:
            for turn in (2, 3):
                candidates = [row for row in rows
                              if row["protocol"] == protocol
                              and row["method_key"] == method
                              and row["turn_id"] == turn]
                if not candidates:
                    raise AnalysisError(
                        f"no manual-audit candidate for {protocol}/{method}/T{turn}")
                chosen = min(candidates, key=lambda row: _sha_bytes(
                    f"{seed}\0{protocol}\0{method}\0{turn}\0{row['dialog_id']}"
                    .encode("utf-8")))
                samples.append({
                    "seed": seed,
                    "protocol": protocol,
                    "method_key": method,
                    "method": METHOD_LABELS[method],
                    "dialog_id": chosen["dialog_id"],
                    "image_id": chosen["image_id"],
                    "turn_id": turn,
                    "question_id": chosen["question_id"],
                    "question": chosen["question"],
                    "history_source": chosen["history_source"],
                    "history_text": chosen["history_text"],
                    "history_answers": chosen["history_answers"],
                    "history_source_request_ids": chosen[
                        "history_source_request_ids"],
                    "prompt": chosen["prompt"],
                    "prompt_sha256": chosen["prompt_sha256"],
                    "prediction": chosen["prediction"],
                    "strict_correct": chosen["strict_correct"],
                    "causal_history_and_provenance_exact": True,
                })
    return samples


def _write_derived_results(
    output_root: Path, *, rows: Sequence[Mapping[str, Any]],
    summaries: Mapping[str, Sequence[Mapping[str, Any]]],
    quality_csv: Mapping[str, Sequence[Mapping[str, Any]]],
    latency: Mapping[str, Sequence[Mapping[str, Any]]],
    lengths: Mapping[str, Sequence[Mapping[str, Any]]],
    ios: Mapping[str, Sequence[Mapping[str, Any]]],
    selection: Mapping[str, Any], validation: Mapping[str, Any],
    analysis_config: Mapping[str, Any], comparisons: Sequence[Mapping[str, Any]],
    propagation_csv: Sequence[Mapping[str, Any]], propagation: Mapping[str, Any],
    paired: Mapping[str, Any], manual_samples: Sequence[Mapping[str, Any]],
    report: str, bootstrap_seed: int,
) -> dict[str, Any]:
    output_root.mkdir(parents=False, exist_ok=False)
    for protocol in PROTOCOLS:
        directory = output_root / protocol
        directory.mkdir()
        protocol_rows = sorted(
            (row for row in rows if row["protocol"] == protocol),
            key=lambda row: (row["dialog_id"], row["turn_id"],
                             METHOD_KEYS.index(row["method_key"])))
        _write_raw_jsonl(directory / "raw.jsonl", protocol_rows)
        _write_gzip(directory / "raw.jsonl.gz", directory / "raw.jsonl")
        _atomic_csv(
            directory / "summary.csv",
            ["method_key", "method", "acc1", "acc2", "acc3", "avg", "counts"],
            summaries[protocol])
        _atomic_csv(
            directory / "quality_by_turn.csv",
            ["method_key", "method", "turn", "n", "correct", "accuracy",
             "accuracy_percent"], quality_csv[protocol])
        _atomic_csv(directory / "ttft_by_turn.csv", list(latency[protocol][0]),
                    latency[protocol])
        _atomic_csv(directory / "token_lengths.csv", list(lengths[protocol][0]),
                    lengths[protocol])
        _atomic_csv(directory / "io_breakdown.csv", list(ios[protocol][0]),
                    ios[protocol])
        protocol_selection = {
            "schema_version": SCHEMA_VERSION,
            "protocol": protocol,
            "qa_t2_t3": selection["qa_t2_t3_within_protocol"][protocol],
            "ours_per_image_invariance": selection["ours_per_image_invariance"],
            "ours_all_images_invariant": selection["ours_all_images_invariant"],
            "ours_all_images_storage_permutation_invariant": selection[
                "ours_all_images_storage_permutation_invariant"],
        }
        _atomic_json(directory / "selection_analysis.json", protocol_selection)
        _atomic_json(directory / "validation.json", {
            **validation, "protocol": protocol,
            "logical_rows": len(protocol_rows),
        })
        _atomic_json(directory / "config.json", {**analysis_config, "protocol": protocol})

    comparison_dir = output_root / "comparison"
    comparison_dir.mkdir()
    _atomic_csv(comparison_dir / "history_comparison.csv",
                list(comparisons[0]), comparisons)
    _atomic_csv(
        comparison_dir / "error_propagation.csv",
        ["method_key", "method", "metric", "condition", "n", "correct",
         "accuracy"], propagation_csv)
    _atomic_json(comparison_dir / "error_propagation.json", propagation)
    _atomic_json(comparison_dir / "paired_quality.json", paired)
    _atomic_json(comparison_dir / "selection_analysis.json", selection)
    _atomic_json(comparison_dir / "manual_history_samples.json", {
        "schema_version": SCHEMA_VERSION,
        "seed": bootstrap_seed,
        "samples": manual_samples,
    })
    _atomic_json(comparison_dir / "validation.json", validation)
    _atomic_text(comparison_dir / "ANALYSIS.md", report)
    _atomic_text(comparison_dir / "README.md", report)
    _atomic_json(output_root / "config.json", analysis_config)
    _atomic_json(output_root / "validation.json", validation)
    _atomic_text(output_root / "README.md", report)
    output_sha256 = {
        path.relative_to(output_root).as_posix(): _sha_file(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }
    completion = {
        "schema_version": SCHEMA_VERSION, "passed": True,
        "validation_sha256": _sha_file(output_root / "validation.json"),
        "analysis_sha256": _sha_file(comparison_dir / "ANALYSIS.md"),
        "logical_rows": len(rows),
        "output_sha256": output_sha256,
    }
    # Marker last within the staging tree.
    _atomic_json(output_root / "COMPLETED", completion)
    for directory in sorted(
            (path for path in output_root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts), reverse=True):
        _fsync_directory(directory)
    _fsync_directory(output_root)
    return completion


def validate_frozen_source_configs(
    configs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected = {
        "model": "llava-hf/llava-v1.6-vicuna-7b-hf",
        "model_revision": FROZEN_MODEL_REVISION,
        "load_4bit": True,
        "quantization": "4-bit NF4 double-quant",
        "compute_dtype": "bfloat16",
        "attention": "eager",
        "decoding": "greedy",
        "max_new_tokens": 16,
        "chunk_size": 64,
        "probe_heads": 3,
        "qa_chunk_configuration": FROZEN_QA_CONFIGURATION,
        "ours_configuration": FROZEN_OURS_CONFIGURATION,
        "later_turn_policy": (
            "ReComp pixels; FullLoad/QA use raster SSD; Ours uses independent "
            "image-only repacked SSD"),
    }
    for protocol in PROTOCOLS:
        config = configs.get(protocol)
        if not isinstance(config, Mapping):
            raise AnalysisError(f"missing source config for {protocol}")
        mismatch = {key: (config.get(key), value)
                    for key, value in expected.items()
                    if config.get(key) != value}
        turn1_policy = str(config.get("turn1_policy", ""))
        raster_policy = str(config.get("qa_raster_source_policy", ""))
        if ("FullLoad captures/persists" not in turn1_policy
                or "captured by FullLoad's own T1" not in raster_policy):
            mismatch["raster_source_provenance"] = (
                {"turn1_policy": turn1_policy,
                 "qa_raster_source_policy": raster_policy},
                "FullLoad-own-T1 canonical raster capture")
        if mismatch:
            raise AnalysisError(
                f"{protocol} frozen source configuration mismatch: {mismatch}")
    common_keys = tuple(expected) + (
        "turn1_policy", "qa_raster_source_policy", "methods",
        "method_order_policy", "quality_metric", "normalization",
        "input_token_count_definition", "main_ttft_field",
    )
    gold = configs["gold_history"]
    generated = configs["generated_history"]
    cross_mismatch = {key: (gold.get(key), generated.get(key))
                      for key in common_keys
                      if gold.get(key) != generated.get(key)}
    if cross_mismatch:
        raise AnalysisError(
            f"Gold/Generated frozen configurations differ: {cross_mismatch}")
    return {
        "passed": True,
        "expected": expected,
        "gold_generated_identical_fields": list(common_keys),
    }


def analyze(gold_run_dir: Path, generated_run_dir: Path,
            results_dir: Path, index_path: Path, protection_validation: Path, *,
            expected_dialogs: int | None = None, expected_images: int | None = None,
            require_independent_executions: bool = True,
            bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
            bootstrap_seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    gold_run_dir = gold_run_dir.resolve()
    generated_run_dir = generated_run_dir.resolve()
    results_dir = results_dir.resolve()
    index_path = index_path.resolve()
    for protocol, run_dir in (("gold_history", gold_run_dir),
                              ("generated_history", generated_run_dir)):
        if run_dir.is_symlink() or not run_dir.is_dir():
            raise AnalysisError(f"{protocol} run directory is invalid: {run_dir}")
        if (results_dir == run_dir or run_dir in results_dir.parents
                or results_dir in run_dir.parents):
            raise AnalysisError("run and result directories may not overlap")
    if gold_run_dir == generated_run_dir:
        raise AnalysisError("Gold and Generated protocols require independent run directories")
    if results_dir.is_symlink():
        raise AnalysisError("results directory may not be a symlink")

    configs: dict[str, dict[str, Any]] = {}
    for protocol, run_dir in (("gold_history", gold_run_dir),
                              ("generated_history", generated_run_dir)):
        config_path = run_dir / "config.json"
        if config_path.is_symlink() or not config_path.is_file():
            raise AnalysisError(f"{protocol} run has no regular config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("protocol") != protocol:
            raise AnalysisError(f"{protocol} run config protocol mismatch")
        configs[protocol] = config
    frozen_config_validation = validate_frozen_source_configs(configs)
    dialogs, index_info = load_index(index_path)
    if expected_dialogs is None:
        configured = {int(config.get("n_dialogs", -1))
                      for config in configs.values()}
        if len(configured) != 1 or next(iter(configured)) < 1:
            raise AnalysisError("protocol run configs disagree on dialogue count")
        expected_dialogs = next(iter(configured))
    if expected_images is None:
        configured_images = {int(config.get("n_images", -1))
                             for config in configs.values()}
        if len(configured_images) != 1 or next(iter(configured_images)) < 1:
            raise AnalysisError("protocol run configs disagree on image count")
        expected_images = next(iter(configured_images))
    if expected_dialogs == FULL_DIALOGUES and len(dialogs) != FULL_DIALOGUES:
        raise AnalysisError("full analysis requires the frozen 4,061-dialogue index")
    if bootstrap_resamples != BOOTSTRAP_RESAMPLES or bootstrap_seed != BOOTSTRAP_SEED:
        raise AnalysisError("analysis requires 10,000 bootstrap resamples and seed 1234")

    gold_rows, gold_artifacts = load_rows(
        gold_run_dir, expected_protocol="gold_history")
    generated_rows, generated_artifacts = load_rows(
        generated_run_dir, expected_protocol="generated_history")
    rows = gold_rows + generated_rows
    artifact_info = {
        "gold_history": gold_artifacts,
        "generated_history": generated_artifacts,
    }
    matrix, validation = validate_rows(
        rows, dialogs, expected_dialogs, expected_images,
        require_independent_executions=require_independent_executions)
    protection = _load_protection_validation(protection_validation)
    validation["checks"]["prior_artifacts_unchanged"] = True
    validation["checks"]["frozen_source_configuration"] = True
    validation["protection_validation"] = protection
    validation["frozen_source_configuration"] = frozen_config_validation
    quality, quality_csv = quality_tables(rows)
    latency, lengths = latency_and_lengths(rows)
    ios = io_tables(rows)
    selection, selection_checks = selection_analysis(matrix, rows)
    validation["checks"].update(selection_checks)
    dialog_ids = sorted({row["dialog_id"] for row in rows})
    propagation_csv, propagation = error_propagation(matrix, dialog_ids)
    paired = paired_quality(
        matrix, dialog_ids, resamples=bootstrap_resamples,
        seed=bootstrap_seed)
    comparisons = history_comparison(quality)
    summaries = _summary_rows(quality)
    manual_samples = _manual_history_samples(rows, seed=bootstrap_seed)
    validation["manual_history_samples"] = manual_samples
    validation["checks"]["manual_history_samples_cover_protocol_method_t2_t3"] = (
        len(manual_samples) == len(PROTOCOLS) * len(METHOD_KEYS) * 2)
    validation["passed"] = all(validation["checks"].values())
    if not validation["passed"]:
        raise AnalysisError("post-selection/manual validation failed")

    analysis_config = {
        "schema_version": SCHEMA_VERSION,
        "gold_run_dir": str(gold_run_dir),
        "generated_run_dir": str(generated_run_dir),
        "results_dir": str(results_dir),
        "index": index_info, "source_configs": configs,
        "protocols": list(PROTOCOLS), "method_keys": list(METHOD_KEYS),
        "quality_metric": "strict_normalized_exact_match",
        "normalization": "lowercase; punctuation to spaces; remove a/an/the; collapse whitespace",
        "main_ttft_population": "cache-hit turns 2 and 3",
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "artifact_info": artifact_info,
        "logical_requests": len(rows),
        "physical_execution_ids": len({row["physical_execution_id"] for row in rows}),
        "protocol_physical_executions_independent": require_independent_executions,
        "protection_validation": protection,
        "generated_history_policy": "same-method own prior predictions only",
        "gold_history_policy": "frozen gold answers only",
        "created_at_unix": time.time(),
    }
    report = build_analysis_markdown(
        quality, comparisons, latency, lengths, ios, selection, paired,
        propagation, validation, index_info)

    if os.path.lexists(results_dir):
        raise FileExistsError(f"results root already exists: {results_dir}")
    if results_dir.parent.is_symlink() or not results_dir.parent.is_dir():
        raise AnalysisError(f"results parent is invalid: {results_dir.parent}")
    staging = results_dir.with_name(
        f".{results_dir.name}.analysis-staging-{os.getpid()}-{uuid.uuid4().hex}")
    if os.path.lexists(staging):
        raise FileExistsError(staging)
    try:
        completion = _write_derived_results(
            staging, rows=rows, summaries=summaries,
            quality_csv=quality_csv, latency=latency, lengths=lengths,
            ios=ios, selection=selection, validation=validation,
            analysis_config=analysis_config, comparisons=comparisons,
            propagation_csv=propagation_csv, propagation=propagation,
            paired=paired, manual_samples=manual_samples, report=report,
            bootstrap_seed=bootstrap_seed)
        _publish_directory_noreplace(staging, results_dir)
    except Exception:
        if staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return {
        "validation": validation, "quality": quality,
        "selection": selection, "paired": paired,
        "results_dir": str(results_dir), "completion": completion,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the MT-GQA Gold/Generated-history four-arm run")
    parser.add_argument("--gold-run-dir", type=Path, required=True)
    parser.add_argument("--generated-run-dir", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--index", type=Path,
                        default=ROOT / "data/mt_gqa/dialogues.json")
    parser.add_argument("--protection-validation", type=Path, required=True)
    parser.add_argument("--expected-dialogs", type=int)
    parser.add_argument("--expected-images", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(
        args.gold_run_dir, args.generated_run_dir,
        args.results_root, args.index, args.protection_validation,
        expected_dialogs=args.expected_dialogs,
        expected_images=args.expected_images,
        require_independent_executions=True,
    )
    print(json.dumps({
        "passed": result["validation"]["passed"],
        "logical_rows": result["validation"]["observed_logical_rows"],
        "results_dir": result["results_dir"],
        "verdict": "MT-GQA GOLD + GENERATED HISTORY 4-ARM EVALUATION VALIDATED: YES",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

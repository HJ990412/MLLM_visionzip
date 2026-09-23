#!/usr/bin/env python3
"""Strict CPU-only analyzer for MT-VQA-v2-reconstructed Generated-History.

The GPU evaluator publishes one immutable JSON artifact per image.  This
program treats those artifacts as untrusted input: it reconstructs every
dialogue/request identity and same-method causal history, recomputes the
repository VQAv2 consensus score from all ten annotations, validates the
four serving contracts, and only then atomically publishes a result tree.

The primary quality metric is the *soft* repository VQA score.  Binary
``score == 1`` results are diagnostic only (paired four-cells and generated
history conditioning); they are never substituted for the primary metric.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import statistics
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mmimpress.dataset import vqa_score  # noqa: E402
from mmimpress import mt_vqa_v2  # noqa: E402


def _load_common():
    """Load already-tested atomic/statistical primitives from analyzer 55."""
    path = ROOT / "scripts/55_analyze_mt_gqa_history.py"
    spec = importlib.util.spec_from_file_location("_mt_vqa_analysis_common", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COMMON = _load_common()

SCHEMA_VERSION = "mt-vqa-v2-generated-4arm-analysis-v1"
SHARD_SCHEMA_VERSION = "mt-vqa-v2-generated-4arm-shard-v1"
PROTOCOL = "generated_history"
METHOD_KEYS = ("recompute", "fullload", "qa_chunk25", "ours25")
METHOD_LABELS = {
    "recompute": "ReComp", "fullload": "FullLoad",
    "qa_chunk25": "QA-Chunk25", "ours25": "Ours25",
}
TURNS = (1, 2, 3)
FULL_DIALOGUES = 250
FULL_IMAGES = 250
FULL_ROWS = 3_000
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 1234
FROZEN_MODEL_REVISION = "c916e6cdcd760b4cecd1dd4907f84ac649f93b23"
EXPECTED_INDEX_SHA256 = "89719b2a1187c07e3228cc76cf1e473b3c713a0dcb3d65da0c596ae81898d6ea"
EXPECTED_WORKLOAD_SHA256 = "384e39bad4e2e8d5865fe20bad7661d3cad8fe0b5ce5effbfc43ea896170cbbc"
QUALITY_METRIC = "repository_vqa_consensus_min_matches_over_3"
QA_RATER_ALGORITHM_ID = "sparsevlm_visual_text_mean_threshold_v1"
FROZEN_QA_CONFIGURATION = dict(COMMON.FROZEN_QA_CONFIGURATION)
FROZEN_OURS_CONFIGURATION = dict(COMMON.FROZEN_OURS_CONFIGURATION)

_atomic_text = COMMON._atomic_text
_atomic_json = COMMON._atomic_json
_atomic_csv = COMMON._atomic_csv
_publish_directory_noreplace = COMMON._publish_directory_noreplace
_fsync_directory = COMMON._fsync_directory
_sha_file = COMMON._sha_file
_sha_bytes = COMMON._sha_bytes
_stable_json = COMMON._stable_json
exact_mcnemar_pvalue = COMMON.exact_mcnemar_pvalue


class AnalysisError(RuntimeError):
    """An input artifact violated the frozen evaluation contract."""


def _pick(row: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if row.get(name) is not None:
            return row[name]
    result = row.get("result")
    if isinstance(result, Mapping):
        for name in names:
            if result.get(name) is not None:
                return result[name]
    return default


def _required(row: Mapping[str, Any], context: str, *names: str) -> Any:
    value = _pick(row, *names, default=None)
    if value is None:
        raise AnalysisError(f"{context}: missing {'/'.join(names)}")
    return value


def _integer(value: Any, context: str, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise AnalysisError(f"{context}: boolean is not an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{context}: invalid integer {value!r}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise AnalysisError(f"{context}: non-integral value {value!r}")
    if minimum is not None and result < minimum:
        raise AnalysisError(f"{context}: {result} < {minimum}")
    return result


def _number(value: Any, context: str, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{context}: invalid number {value!r}") from exc
    if not math.isfinite(result):
        raise AnalysisError(f"{context}: non-finite number")
    if minimum is not None and result < minimum:
        raise AnalysisError(f"{context}: {result} < {minimum}")
    return result


def _method(value: Any) -> str:
    key = "".join(ch for ch in str(value).lower() if ch.isalnum())
    aliases = {
        "recomp": "recompute", "recompute": "recompute",
        "fullload": "fullload", "full": "fullload",
        "qachunk25": "qa_chunk25", "qachunk": "qa_chunk25",
        "ours25": "ours25", "ours": "ours25",
        "imageonlyprefix25": "ours25",
    }
    if key not in aliases:
        raise AnalysisError(f"unknown method {value!r}")
    return aliases[key]


def soft_vqa_score(prediction: Any, answers: Any) -> float:
    """Repository VQAv2 score, requiring the complete ten-answer annotation."""
    if not isinstance(answers, (list, tuple)) or len(answers) != 10:
        raise AnalysisError("VQAv2 scoring requires exactly ten answers")
    values = [str(value) for value in answers]
    if any(not value.strip() for value in values):
        raise AnalysisError("VQAv2 answers must be nonempty")
    return float(vqa_score(str(prediction), values))


def render_history(dialog: Mapping[str, Any], turn: int,
                   prior_predictions: Sequence[str]) -> str:
    if len(prior_predictions) != turn - 1:
        raise ValueError("history must contain exactly the available prior turns")
    lines: list[str] = []
    for prior_turn, prediction in enumerate(prior_predictions, 1):
        question = str(dialog["turns"][prior_turn - 1]["question"]).strip()
        lines.extend((f"Q{prior_turn}: {question}",
                      f"A{prior_turn}: {prediction}"))
    return "\n".join(lines)


def render_prompt(dialog: Mapping[str, Any], turn: int,
                  prior_predictions: Sequence[str]) -> str:
    history = render_history(dialog, turn, prior_predictions)
    body: list[str] = []
    if history:
        body.extend((history, ""))
    question = str(dialog["turns"][turn - 1]["question"]).strip()
    body.extend((
        f"Current question Q{turn}: {question}",
        "Answer the current question with a single word or short phrase. ASSISTANT:",
    ))
    return "USER: <image>\n" + "\n".join(body)


def _selected_layers(value: Any, context: str) -> list[list[int]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise AnalysisError(f"{context}: selection must be a layer list")
    output: list[list[int]] = []
    for layer, chunks in enumerate(value):
        if not isinstance(chunks, list):
            raise AnalysisError(f"{context}: layer {layer} is not a list")
        ids = [_integer(item, f"{context}/layer{layer}", 0) for item in chunks]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise AnalysisError(f"{context}: chunk IDs are not sorted/unique")
        output.append(ids)
    return output


def _canonical_row(raw: Mapping[str, Any], source: Path, ordinal: int,
                   manifests: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise AnalysisError(f"{source}: row {ordinal} is not an object")
    context = f"{source.name}:row{ordinal}"
    if str(_required(raw, context, "protocol")) != PROTOCOL:
        raise AnalysisError(f"{context}: only Generated-History is allowed")
    method = _method(_required(raw, context, "method_key", "method"))
    turn = _integer(_required(raw, context, "turn_id", "turn"), context, 1)
    if turn not in TURNS:
        raise AnalysisError(f"{context}: invalid turn")
    answers = _required(raw, context, "gold_answers", "gold", "answers")
    if not isinstance(answers, list) or len(answers) != 10:
        raise AnalysisError(f"{context}: all ten VQAv2 answers were not preserved")
    answers = [str(answer) for answer in answers]
    for alias in ("gold", "gold_answers", "gold_answer"):
        if raw.get(alias) is not None and raw.get(alias) != answers:
            raise AnalysisError(f"{context}: {alias} does not preserve the canonical ten answers")
    prediction = str(_required(raw, context, "prediction", "answer"))
    score = soft_vqa_score(prediction, answers)
    for claimed_name in ("vqa_score", "quality_score", "score", "correct"):
        if raw.get(claimed_name) is not None and not math.isclose(
                float(raw[claimed_name]), score, abs_tol=1e-12):
            raise AnalysisError(
                f"{context}: {claimed_name} differs from recomputed VQA score")
    if (raw.get("quality_metric") != QUALITY_METRIC
            or raw.get("quality_metric_implementation")
            != "mmimpress.dataset.vqa_score"
            or raw.get("official_vqa_evaluator_claimed") is not False
            or float(raw.get("binary_correct_threshold", -1)) != 0.5
            or int(raw.get("binary_correct", -1)) != int(score >= 0.5)
            or int(raw.get("full_credit_correct", -1)) != int(score == 1.0)):
        raise AnalysisError(f"{context}: VQA metric metadata/diagnostics changed")
    prompt = str(_required(raw, context, "prompt"))
    history = str(_required(raw, context, "history_text"))
    prompt_sha = str(_required(raw, context, "prompt_sha256"))
    history_sha = str(_required(
        raw, context, "history_text_sha256", "text_history_sha256"))
    if prompt_sha != _sha_bytes(prompt.encode()) or history_sha != _sha_bytes(history.encode()):
        raise AnalysisError(f"{context}: prompt/history hash mismatch")
    history_answers = _required(raw, context, "history_answers")
    source_ids = _required(raw, context, "history_source_request_ids")
    if not isinstance(history_answers, list) or not isinstance(source_ids, list):
        raise AnalysisError(f"{context}: malformed history provenance")
    selected = _selected_layers(
        _pick(raw, "selected_chunk_ids_per_layer", default=[]), context)
    status = str(_pick(raw, "status", default="")).lower()
    status_ok = status in {"ok", "pass", "passed", "success", "completed"}
    logical = str(_required(raw, context, "logical_request_id"))
    physical = str(_required(
        raw, context, "physical_execution_id", "execution_id"))
    if raw.get("execution_id") is not None and str(raw["execution_id"]) != physical:
        raise AnalysisError(f"{context}: physical/execution ID aliases differ")
    ttft = _number(_required(
        raw, context, "end_to_end_ttft_ms", "ttft_ms"), context, 0.0)
    timestamp_names = (
        "request_started_at_s", "core_started_at_s", "first_token_at_s",
        "model_finished_at_s", "request_returned_at_s",
    )
    timestamps = [_pick(raw, name, default=None) for name in timestamp_names]
    if any(value is None for value in timestamps):
        raise AnalysisError(f"{context}: authoritative timing timestamps are missing")
    timestamp_values = [_number(value, f"{context}/{name}")
                        for name, value in zip(timestamp_names, timestamps)]
    if timestamp_values != sorted(timestamp_values):
        raise AnalysisError(f"{context}: request timestamps are not monotonic")
    timestamp_ttft = (timestamp_values[2] - timestamp_values[0]) * 1e3
    if abs(timestamp_ttft - ttft) > 1e-3:
        raise AnalysisError(f"{context}: TTFT does not match synchronized first-token timestamps")
    if abs(float(_pick(raw, "ttft_identity_error_ms", default=0.0))) > 1e-3:
        raise AnalysisError(f"{context}: TTFT phase identity failed")
    row = dict(raw)
    row.update({
        "analysis_schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "method_key": method,
        "method": METHOD_LABELS[method],
        "dialog_id": str(_required(raw, context, "dialog_id", "dialogue_id")),
        "image_id": str(_required(raw, context, "image_id")),
        "turn_id": turn,
        "question_id": str(_required(raw, context, "question_id")),
        "question": str(_required(raw, context, "question")),
        "gold_answers": answers,
        "gold": answers,
        "prediction": prediction,
        "vqa_score": score,
        "quality_score": score,
        "full_credit_correct": int(score == 1.0),
        "quality_metric": QUALITY_METRIC,
        "logical_request_id": logical,
        "physical_execution_id": physical,
        "status_ok": status_ok,
        "prompt": prompt,
        "prompt_sha256": prompt_sha,
        "history_text": history,
        "history_text_sha256": history_sha,
        "history_answers": [str(item) for item in history_answers],
        "history_source_request_ids": [str(item) for item in source_ids],
        "first_token_id": _integer(_required(raw, context, "first_token_id"), context, 0),
        "input_token_count": _integer(_required(raw, context, "input_token_count"), context, 1),
        "generated_token_count": _integer(_required(
            raw, context, "generated_token_count", "generated_tokens"), context, 1),
        "ttft_ms": ttft,
        "request_e2e_ms": _number(_required(raw, context, "request_e2e_ms"), context, 0.0),
        "ssd_read_bytes": _integer(_pick(raw, "ssd_read_bytes", default=0), context, 0),
        "normal_kv_read_bytes": _integer(_pick(raw, "normal_kv_read_bytes", default=0), context, 0),
        "probe_read_bytes": _integer(_pick(raw, "probe_read_bytes", default=0), context, 0),
        "separator_read_bytes": _integer(_pick(raw, "separator_read_bytes", default=0), context, 0),
        "pread_count": _integer(_pick(raw, "pread_count", "ssd_preads", default=0), context, 0),
        "normal_kv_preads": _integer(_pick(raw, "normal_kv_preads", default=0), context, 0),
        "probe_preads": _integer(_pick(raw, "probe_preads", default=0), context, 0),
        "separator_preads": _integer(_pick(raw, "separator_preads", default=0), context, 0),
        "ssd_read_ms": _number(_pick(raw, "ssd_read_ms", default=0), context, 0.0),
        "selected_chunk_ids_per_layer": selected,
        "source_artifact": str(source),
        "store_manifests": dict(manifests),
    })
    for name in (
        "rater_count", "n_raters", "rater_selection_ms", "rater_ms",
        "query_projection_ms", "projection_ms", "probe_io_ms",
        "query_scoring_ms", "chunk_aggregation_ms", "topk_chunk_ms",
        "selected_id_d2h_ms", "chunk_planning_ms", "selector_wall_ms",
        "selector_ms", "chunk_io_ms", "scatter_ms", "prefill_ms",
        "contiguous_runs_per_layer_mean",
    ):
        value = _pick(raw, name, default=None)
        if value is not None:
            row[name] = _number(value, f"{context}/{name}", 0.0)
    return row


def discover_artifacts(run_dir: Path) -> list[Path]:
    candidates = (run_dir / "images", run_dir / "image_artifacts",
                  run_dir / "artifacts" / "images")
    populated = [(path, sorted(path.glob("*.json"))) for path in candidates
                 if path.is_dir() and list(path.glob("*.json"))]
    if len(populated) != 1:
        raise AnalysisError(f"expected one populated image artifact directory under {run_dir}")
    return populated[0][1]


def load_rows(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    persistence: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    experiment_ids: set[str] = set()
    for path in discover_artifacts(run_dir):
        if path.is_symlink() or not path.is_file():
            raise AnalysisError(f"artifact is not a regular file: {path}")
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AnalysisError(f"cannot parse {path}") from exc
        if not isinstance(artifact, Mapping):
            raise AnalysisError(f"artifact is not an object: {path}")
        claimed = artifact.get("artifact_content_sha256")
        body = {key: value for key, value in artifact.items()
                if key != "artifact_content_sha256"}
        if claimed != _stable_json(body):
            raise AnalysisError(f"artifact content hash mismatch: {path}")
        if artifact.get("schema_version") != SHARD_SCHEMA_VERSION:
            raise AnalysisError(f"artifact schema mismatch: {path}")
        if artifact.get("protocol") != PROTOCOL:
            raise AnalysisError(f"artifact protocol mismatch: {path}")
        experiment_id = str(artifact.get("experiment_id", ""))
        if not experiment_id:
            raise AnalysisError(f"artifact experiment identity is missing: {path}")
        experiment_ids.add(experiment_id)
        manifests = artifact.get("store_manifests")
        if not isinstance(manifests, Mapping) or set(manifests) != {"raster", "image_only"}:
            raise AnalysisError(f"artifact omits dual-store manifests: {path}")
        raster, image_only = manifests["raster"], manifests["image_only"]
        if (not isinstance(raster, Mapping) or not isinstance(image_only, Mapping)
                or raster.get("physical_layout") != "raster"
                or image_only.get("physical_layout") != "visionzip_image_only"
                or int(raster.get("visual_kv_bytes", -1))
                != int(image_only.get("visual_kv_bytes", -2))
                or int(raster.get("n_chunks_per_layer", -1))
                != int(image_only.get("n_chunks_per_layer", -2))
                or not image_only.get("permutation_sha256")):
            raise AnalysisError(f"artifact dual-store manifest contract failed: {path}")
        raw_rows = artifact.get("rows")
        if not isinstance(raw_rows, list):
            raise AnalysisError(f"artifact omits rows: {path}")
        image_id = str(artifact.get("image_id", ""))
        if not image_id or image_id in seen_images:
            raise AnalysisError(f"duplicate/empty image artifact: {image_id!r}")
        seen_images.add(image_id)
        for ordinal, raw in enumerate(raw_rows, 1):
            row = _canonical_row(raw, path, ordinal, manifests)
            if row["image_id"] != image_id:
                raise AnalysisError(f"{path}: row image mismatch")
            for field in (
                    "dialogues_file_sha256", "source_full_workload_sha256",
                    "selected_workload_sha256", "model_revision"):
                if artifact.get(field) != row.get(field):
                    raise AnalysisError(f"{path}: artifact/row {field} mismatch")
            rows.append(row)
        persist = artifact.get("persistence_overhead")
        if not isinstance(persist, Mapping) or set(persist) != {"raster", "image_only"}:
            raise AnalysisError(f"{path}: persistence provenance is missing")
        if artifact.get("store_build_counts") != {"raster": 1, "image_only": 1}:
            raise AnalysisError(f"{path}: each physical store must be built once")
        by_physical = {row["physical_execution_id"]: row for row in rows
                       if row["image_id"] == image_id}
        for store_kind in ("raster", "image_only"):
            value = persist[store_kind]
            if not isinstance(value, Mapping):
                raise AnalysisError(f"{path}: malformed persistence record")
            persistence.append({
                "image_id": image_id,
                "store_kind": store_kind,
                "source_dialog_id": str(value.get("source_dialog_id", "")),
                "source_method_key": str(value.get("source_method_key", "")),
                "source_execution_id": str(value.get("source_execution_id", "")),
                "persist_ms": _number(
                    _required(value.get("timing_ms", {}), str(path), "persist_ms"),
                    f"{path}/persist_ms", 0.0),
                "ssd_write_ms": _number(
                    _required(value.get("timing_ms", {}), str(path), "ssd_write_ms"),
                    f"{path}/ssd_write_ms", 0.0),
                "write_bytes": _integer(
                    _required(value.get("bytes", {}), str(path), "total"),
                    f"{path}/write_bytes", 0),
            })
            expected_method = "fullload" if store_kind == "raster" else "ours25"
            source_id = str(value.get("source_execution_id", ""))
            source_row = by_physical.get(source_id)
            if (str(value.get("source_method_key", "")) != expected_method
                    or source_row is None or source_row["turn_id"] != 1
                    or source_row["method_key"] != expected_method
                    or source_row.get("persistence_source") != store_kind):
                raise AnalysisError(f"{path}: {store_kind} persistence source is not its own T1")
        hashes[path.name] = _sha_file(path)
    if len(experiment_ids) != 1:
        raise AnalysisError("image artifacts span multiple experiments")
    return rows, {
        "artifact_count": len(hashes), "artifact_sha256": hashes,
        "image_count": len(seen_images),
        "experiment_id": next(iter(experiment_ids)),
    }, persistence


def load_index(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise AnalysisError(f"index must be a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot parse index: {path}") from exc
    envelope = dict(payload) if isinstance(payload, Mapping) else {}
    raw_dialogs = payload.get("dialogues") if isinstance(payload, Mapping) else payload
    if not isinstance(raw_dialogs, list) or not raw_dialogs:
        raise AnalysisError("index has no dialogues")
    dialogs: dict[str, dict[str, Any]] = {}
    images: set[str] = set()
    questions: set[str] = set()
    for ordinal, raw in enumerate(raw_dialogs):
        if not isinstance(raw, Mapping):
            raise AnalysisError("index dialogue is not an object")
        dialog = dict(raw)
        did = str(dialog.get("dialog_id", dialog.get("dialogue_id", "")))
        image = str(dialog.get("image_id", ""))
        turns = dialog.get("turns")
        if not did or did in dialogs or not image or image in images:
            raise AnalysisError(f"duplicate dialogue/image in index: {did!r}/{image!r}")
        if not isinstance(turns, list) or len(turns) != 3:
            raise AnalysisError(f"{did}: dialogue does not have three turns")
        for turn_id, turn in enumerate(turns, 1):
            if not isinstance(turn, Mapping) or int(turn.get("turn_id", -1)) != turn_id:
                raise AnalysisError(f"{did}: turn ordering changed")
            qid = str(turn.get("question_id", ""))
            if not qid or qid in questions:
                raise AnalysisError(f"{did}: question was reused")
            soft_vqa_score("", turn.get("answers"))
            questions.add(qid)
        dialogs[did] = dialog
        images.add(image)
    envelope.pop("dialogues", None)
    return dialogs, {
        "path": str(path.resolve()), "sha256": _sha_file(path),
        "workload_sha256": mt_vqa_v2.workload_sha256(list(dialogs.values())),
        "dialogues": len(dialogs), "images": len(images),
        "turns": len(questions), "question_reuse": 0,
        "all_dialogues_exactly_three_turns": True,
        "all_dialogues_one_unique_image": True,
        "envelope": envelope,
    }


def _run_lengths(ids: Sequence[int]) -> list[int]:
    if not ids:
        return []
    output: list[int] = []
    start = previous = int(ids[0])
    for item in ids[1:]:
        current = int(item)
        if current != previous + 1:
            output.append(previous - start + 1)
            start = current
        previous = current
    output.append(previous - start + 1)
    return output


def _expected_selected_bytes(row: Mapping[str, Any], store: Mapping[str, Any]) -> int:
    layers = int(store["num_layers"])
    tokens = int(store["v_token_num"])
    visual = int(store["visual_kv_bytes"])
    if layers <= 0 or tokens <= 0 or visual % (layers * tokens):
        raise AnalysisError("store has non-integral visual token-row bytes")
    chunk = int(store["chunk_size"])
    selected_rows = 0
    for layer_ids in row["selected_chunk_ids_per_layer"]:
        for chunk_id in layer_ids:
            start = int(chunk_id) * chunk
            stop = min(start + chunk, tokens)
            if start < 0 or start >= tokens:
                raise AnalysisError("selection addresses an invalid chunk")
            selected_rows += stop - start
    return selected_rows * (visual // (layers * tokens))


def _validate_io(row: Mapping[str, Any]) -> None:
    method, turn = row["method_key"], row["turn_id"]
    total = row["ssd_read_bytes"]
    normal, probe, separator = (row["normal_kv_read_bytes"],
                                row["probe_read_bytes"],
                                row["separator_read_bytes"])
    preads = row["pread_count"]
    normal_preads, probe_preads, separator_preads = (
        row["normal_kv_preads"], row["probe_preads"], row["separator_preads"])
    if total != normal + probe + separator or preads != (
            normal_preads + probe_preads + separator_preads):
        raise AnalysisError("SSD byte/pread component decomposition failed")
    if turn == 1 or method == "recompute":
        if any((total, preads, normal, probe, separator)):
            raise AnalysisError("pixel/ReComp request performed SSD Visual-KV reads")
        return
    manifests = row["store_manifests"]
    store = manifests["image_only" if method == "ours25" else "raster"]
    layers = int(store["num_layers"])
    if method == "fullload":
        if (total != int(store["visual_kv_bytes"]) or normal != total
                or probe or separator or normal_preads != 2 * layers
                or probe_preads or separator_preads):
            raise AnalysisError("FullLoad exact I/O contract failed")
        return
    expected_normal = _expected_selected_bytes(row, store)
    runs = [len(_run_lengths(ids)) for ids in row["selected_chunk_ids_per_layer"]]
    if method == "qa_chunk25":
        raster = manifests["raster"]
        if (normal != expected_normal
                or probe != int(raster["probe_sidecar_bytes"])
                or separator != int(raster["separator_sidecar_bytes"])
                or normal_preads != 2 * sum(runs)
                or probe_preads != layers or separator_preads != 1):
            raise AnalysisError("QA-Chunk25 exact I/O contract failed")
    elif method == "ours25":
        if (normal != expected_normal or probe
                or separator != int(store["separator_sidecar_bytes"])
                or normal_preads != 2 * layers or probe_preads
                or separator_preads != 1 or runs != [1] * layers):
            raise AnalysisError("Ours25 exact I/O contract failed")


def validate_rows(rows: Sequence[dict[str, Any]],
                  dialogs: Mapping[str, dict[str, Any]],
                  expected_dialogs: int, expected_images: int) -> tuple[
                      dict[tuple[str, int, str], dict[str, Any]], dict[str, Any]]:
    expected_rows = expected_dialogs * 3 * 4
    if len(rows) != expected_rows:
        raise AnalysisError(f"logical rows {len(rows)} != {expected_rows}")
    matrix: dict[tuple[str, int, str], dict[str, Any]] = {}
    logical_ids: set[str] = set()
    physical_ids: set[str] = set()
    failed = 0
    for row in rows:
        key = (row["dialog_id"], row["turn_id"], row["method_key"])
        if key in matrix:
            raise AnalysisError(f"duplicate matrix cell {key}")
        if row["logical_request_id"] in logical_ids:
            raise AnalysisError("duplicate logical request ID")
        if row["physical_execution_id"] in physical_ids:
            raise AnalysisError("physical execution was reused")
        matrix[key] = row
        logical_ids.add(row["logical_request_id"])
        physical_ids.add(row["physical_execution_id"])
        failed += int(not row["status_ok"])
    if failed:
        raise AnalysisError(f"run contains {failed} failed requests")
    observed_dialogs = {row["dialog_id"] for row in rows}
    observed_images = {row["image_id"] for row in rows}
    if len(observed_dialogs) != expected_dialogs or len(observed_images) != expected_images:
        raise AnalysisError("observed workload counts differ from expectation")
    if observed_dialogs != set(dialogs):
        raise AnalysisError("run/index dialogue membership differs")
    expected = {(did, turn, method) for did in dialogs
                for turn in TURNS for method in METHOD_KEYS}
    if set(matrix) != expected:
        raise AnalysisError("run does not contain the complete 4-arm matrix")

    t1_fair = 0
    qa_calls = 0
    ours_fingerprints: dict[str, set[str]] = defaultdict(set)
    ours_permutations: dict[str, set[str]] = defaultdict(set)
    seen_sessions: set[str] = set()
    seen_contexts: set[str] = set()
    for did, dialog in dialogs.items():
        image_id = str(dialog["image_id"])
        dialog_sessions: set[str] = set()
        dialog_contexts: set[str] = set()
        for method in METHOD_KEYS:
            prior: list[dict[str, Any]] = []
            for turn in TURNS:
                row = matrix[(did, turn, method)]
                source = dialog["turns"][turn - 1]
                source_answers = [str(value) for value in source["answers"]]
                if (row["image_id"] != image_id
                        or row["question_id"] != str(source["question_id"])
                        or row["question"] != str(source["question"])
                        or row["gold_answers"] != source_answers):
                    raise AnalysisError(f"{did}/T{turn}/{method}: workload drift")
                prior_predictions = [item["prediction"] for item in prior]
                if (row["history_answers"] != prior_predictions
                        or row["history_text"] != render_history(
                            dialog, turn, prior_predictions)
                        or row["prompt"] != render_prompt(
                            dialog, turn, prior_predictions)):
                    raise AnalysisError(f"{did}/T{turn}/{method}: generated history contamination")
                expected_ids = [item["logical_request_id"] for item in prior]
                if row["history_source_request_ids"] != expected_ids:
                    raise AnalysisError(f"{did}/T{turn}/{method}: cross-method history lineage")
                physical_lineage = row.get("history_source_physical_execution_ids")
                if (not isinstance(physical_lineage, list)
                        or [str(value) for value in physical_lineage]
                        != [item["physical_execution_id"] for item in prior]):
                    raise AnalysisError(f"{did}/T{turn}/{method}: physical history lineage")
                if turn > 1:
                    source_kind = str(row.get("history_source", "")).lower()
                    if "gold" in source_kind or not (
                            "generated" in source_kind or "prediction" in source_kind):
                        raise AnalysisError(f"{did}/T{turn}/{method}: history is not generated")
                    entries = row.get("history_entries")
                    if not isinstance(entries, list) or len(entries) != turn - 1:
                        raise AnalysisError("raw history entry provenance is missing")
                    for index, entry in enumerate(entries):
                        if (entry.get("answer_source") != "generated"
                                or entry.get("source_method_key") != method
                                or str(entry.get("answer")) != prior_predictions[index]
                                or int(entry.get("turn_id", -1)) != index + 1):
                            raise AnalysisError("gold/future/cross-method history contamination")
                if row.get("history_turn_ids") != list(range(1, turn)):
                    raise AnalysisError("history turn IDs contain current/future leakage")
                if int(row.get("future_leakage", -1)) != 0:
                    raise AnalysisError("row reports future leakage")
                if row.get("quality_metric") != QUALITY_METRIC:
                    raise AnalysisError("quality metric changed")
                expected_logical = f"{PROTOCOL}:{did}:t{turn}:{method}"
                if row["logical_request_id"] != expected_logical:
                    raise AnalysisError("logical request identity changed")
                is_cache = turn >= 2 and method != "recompute"
                expected_path = "stored_visual_kv" if is_cache else "normal_multimodal_pixel"
                if (bool(row.get("cache_hit")) != is_cache
                        or row.get("request_path") != expected_path
                        or row.get("execution_mode") != expected_path
                        or int(row.get("vision_forward_count", -1))
                        != (0 if is_cache else 1)):
                    raise AnalysisError("request serving-path contract failed")
                if is_cache and row.get(
                        "page_cache_conditioning_excluded_from_ttft") is not True:
                    raise AnalysisError("page-cache conditioning leaked into TTFT")
                if row["ttft_ms"] <= 0 or row["request_e2e_ms"] < row["ttft_ms"]:
                    raise AnalysisError("request timing boundaries are invalid")
                session = str(row.get("dialogue_session_id", ""))
                context_id = str(row.get("context_instance_id", ""))
                if not session or not context_id:
                    raise AnalysisError("dialogue session/context provenance is missing")
                dialog_sessions.add(session)
                dialog_contexts.add(context_id)
                _validate_io(row)
                if turn >= 2 and method == "qa_chunk25":
                    qa_calls += 1
                    manifest = row["store_manifests"]["raster"]
                    layers = int(manifest["num_layers"])
                    if (int(row.get("query_score_calls", 0)) != layers
                            or int(row.get("chunk_score_calls", 0)) != layers
                            or int(row.get("rater_count", row.get("n_raters", 0))) <= 0
                            or row.get("physical_layout") != "raster"
                            or row.get("chunk_score")
                            != "mean_valid_spatial_token_importance"
                            or row.get("head_reduce") != "mean"
                            or row.get("rater_algorithm_id") != QA_RATER_ALGORITHM_ID
                            or row.get("rater_scope")
                            != "entire_available_causal_suffix"
                            or int(row.get("probe_heads_used", -1)) != 3
                            or float(row.get("fallback_rate", -1)) != 0.0
                            or row.get("adaptive_ratio") is not False):
                        raise AnalysisError("QA frozen selector contract failed")
                if turn >= 2 and method in {"qa_chunk25", "ours25"}:
                    store = row["store_manifests"][
                        "image_only" if method == "ours25" else "raster"]
                    total_chunks = int(store["n_chunks_per_layer"])
                    expected_k = max(1, min(total_chunks, round(total_chunks * 0.25)))
                    selected = row["selected_chunk_ids_per_layer"]
                    if (not selected or any(len(ids) != expected_k for ids in selected)
                            or int(row.get("n_chunks_total", -1)) != total_chunks
                            or float(row.get("retention_ratio", -1)) != 0.25):
                        raise AnalysisError("frozen 25% normal-chunk budget failed")
                    loaded = row.get("actual_loaded_chunk_ids_per_layer")
                    if loaded is not None and loaded != selected:
                        raise AnalysisError("selected and physically loaded chunks differ")
                    if method == "ours25":
                        if (row.get("physical_layout") != "visionzip_image_only"
                                or any(int(row.get(name, 0) or 0) for name in (
                                    "query_score_calls", "static_score_calls", "diversity_calls"))
                                or any(ids != list(range(expected_k)) for ids in selected)):
                            raise AnalysisError("Ours fixed-prefix contract failed")
                        ours_fingerprints[image_id].add(_stable_json(selected))
                        permutation = row.get("store_permutation_sha256")
                        if not permutation:
                            raise AnalysisError("Ours permutation provenance is missing")
                        ours_permutations[image_id].add(str(permutation))
                if turn >= 2 and method == "fullload" and int(
                        row.get("query_score_calls", 0) or 0):
                    raise AnalysisError("FullLoad invoked selector scoring")
                prior.append(row)
        if len(dialog_sessions) != 1 or len(dialog_contexts) != 1:
            raise AnalysisError(f"{did}: session/context changed within dialogue")
        session = next(iter(dialog_sessions))
        context_id = next(iter(dialog_contexts))
        if session in seen_sessions or context_id in seen_contexts:
            raise AnalysisError("session/context leaked across dialogues")
        seen_sessions.add(session)
        seen_contexts.add(context_id)
        t1 = [matrix[(did, 1, method)] for method in METHOD_KEYS]
        fairness_fields = (
            "prompt", "prompt_sha256", "input_tensors_sha256",
            "image_input_sha256", "input_ids_sha256", "prediction",
            "first_token_id",
        )
        for field in fairness_fields:
            values = {row.get(field) for row in t1}
            if len(values) != 1 or None in values:
                raise AnalysisError(f"{did}: Turn-1 fairness failed for {field}")
        if len({row["physical_execution_id"] for row in t1}) != 4:
            raise AnalysisError(f"{did}: Turn-1 executions were not independent")
        t1_fair += 1
    if any(len(value) != 1 for value in ours_fingerprints.values()):
        raise AnalysisError("Ours T2/T3 prefix changed for the same image")
    if any(len(value) != 1 for value in ours_permutations.values()):
        raise AnalysisError("Ours storage permutation changed for the same image")
    checks = {
        "all_dialogues_exactly_three_turns_same_image": True,
        "complete_method_turn_matrix": True,
        "exact_logical_row_count": len(rows) == expected_rows,
        "failed_requests_zero": failed == 0,
        "duplicate_requests_zero": len(matrix) == len(rows),
        "physical_executions_unique": len(physical_ids) == len(rows),
        "vqa_soft_scores_recomputed_from_ten_answers": True,
        "same_method_generated_history_exact": True,
        "gold_history_uses_zero": True,
        "future_leakage_zero": True,
        "turn1_four_arm_prompt_pixel_prediction_first_token_fair": t1_fair == expected_dialogs,
        "qa_query_scoring_calls_positive": qa_calls == expected_dialogs * 2,
        "full_load_selector_calls_zero": True,
        "ours_query_scoring_calls_zero": True,
        "qa_and_ours_fixed_25pct_budget": True,
        "exact_ssd_io_contracts": True,
        "ours_t2_t3_prefix_and_permutation_invariant": True,
    }
    return matrix, {
        "schema_version": SCHEMA_VERSION, "passed": all(checks.values()),
        "checks": checks, "expected_logical_rows": expected_rows,
        "observed_logical_rows": len(rows),
        "expected_dialogues": expected_dialogs,
        "observed_dialogues": len(observed_dialogs),
        "expected_images": expected_images,
        "observed_images": len(observed_images),
        "failed_requests": failed, "duplicate_requests": len(rows) - len(matrix),
        "t1_dialogues_checked": t1_fair,
    }


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {"n": int(array.size), "mean": float(array.mean()),
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95))}


def quality_tables(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict], list[dict]]:
    summary, by_turn = [], []
    for method in METHOD_KEYS:
        accuracies: dict[int, float] = {}
        for turn in TURNS:
            selected = [row for row in rows if row["method_key"] == method
                        and row["turn_id"] == turn]
            score_sum = math.fsum(float(row["vqa_score"]) for row in selected)
            accuracy = score_sum / len(selected)
            accuracies[turn] = accuracy
            by_turn.append({
                "method_key": method, "method": METHOD_LABELS[method],
                "turn": turn, "n": len(selected), "score_sum": score_sum,
                "accuracy": accuracy, "accuracy_percent": accuracy * 100,
                "full_credit_count": sum(row["full_credit_correct"] for row in selected),
            })
        summary.append({
            "method_key": method, "method": METHOD_LABELS[method],
            "acc1": accuracies[1], "acc2": accuracies[2],
            "acc3": accuracies[3],
            "avg": statistics.fmean(accuracies.values()),
        })
    return summary, by_turn


def latency_and_lengths(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict], list[dict]]:
    latency, lengths = [], []
    for method in METHOD_KEYS:
        for population, turns in (("turn2", (2,)), ("turn3", (3,)),
                                  ("pooled_t2_t3", (2, 3))):
            selected = [row for row in rows if row["method_key"] == method
                        and row["turn_id"] in turns]
            stats = _stats([row["ttft_ms"] for row in selected])
            latency.append({
                "method_key": method, "method": METHOD_LABELS[method],
                "population": population, "turns": "+".join(map(str, turns)),
                "n": stats["n"], "ttft_mean_ms": stats["mean"],
                "ttft_p50_ms": stats["p50"], "ttft_p95_ms": stats["p95"],
            })
        for turn in TURNS:
            selected = [row for row in rows if row["method_key"] == method
                        and row["turn_id"] == turn]
            inputs = _stats([row["input_token_count"] for row in selected])
            generated = _stats([row["generated_token_count"] for row in selected])
            lengths.append({
                "method_key": method, "method": METHOD_LABELS[method],
                "turn": turn, "n": len(selected),
                "input_tokens_mean": inputs["mean"],
                "input_tokens_p50": inputs["p50"],
                "input_tokens_p95": inputs["p95"],
                "generated_tokens_mean": generated["mean"],
                "generated_tokens_p50": generated["p50"],
                "generated_tokens_p95": generated["p95"],
            })
    return latency, lengths


def io_tables(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    output: list[dict] = []
    full_by_cell = {(row["dialog_id"], row["turn_id"]): row
                    for row in rows if row["method_key"] == "fullload"}
    for method in METHOD_KEYS:
        for population, turns in (("turn2", (2,)), ("turn3", (3,)),
                                  ("pooled_t2_t3", (2, 3))):
            selected = [row for row in rows if row["method_key"] == method
                        and row["turn_id"] in turns]
            total_bytes = math.fsum(row["ssd_read_bytes"] for row in selected)
            matched_full = math.fsum(full_by_cell[(row["dialog_id"], row["turn_id"])][
                "ssd_read_bytes"] for row in selected)
            run_lengths = [length for row in selected
                           for ids in row["selected_chunk_ids_per_layer"]
                           for length in _run_lengths(ids)]
            selected_chunks = sum(len(ids) for row in selected
                                  for ids in row["selected_chunk_ids_per_layer"])
            total_chunks = sum(
                int(row["store_manifests"][
                    "image_only" if method == "ours25" else "raster"][
                        "n_chunks_per_layer"])
                * len(row["selected_chunk_ids_per_layer"])
                for row in selected if row["selected_chunk_ids_per_layer"])
            full_chunk_counts = [int(row["store_manifests"]["raster"][
                "n_chunks_per_layer"]) for row in selected]
            if method == "fullload":
                mean_run_length: float | None = statistics.fmean(full_chunk_counts)
                max_run_length = max(full_chunk_counts)
            elif method == "recompute":
                mean_run_length, max_run_length = None, 0
            else:
                mean_run_length = statistics.fmean(run_lengths) if run_lengths else 0.0
                max_run_length = max(run_lengths, default=0)
            item = {
                "method_key": method, "method": METHOD_LABELS[method],
                "population": population, "n": len(selected),
                "nominal_normal_chunk_budget": (
                    0.25 if method in {"qa_chunk25", "ours25"}
                    else 1.0 if method == "fullload" else 0.0),
                "actual_normal_touched_chunk_ratio": (
                    selected_chunks / total_chunks if total_chunks else
                    1.0 if method == "fullload" else 0.0),
                "ssd_read_bytes_mean": total_bytes / len(selected),
                "ssd_read_mb_mean": total_bytes / len(selected) / 1e6,
                "ssd_ratio_vs_matched_fullload": (
                    total_bytes / matched_full if matched_full else 0.0),
                "probe_mb_mean": statistics.fmean(row["probe_read_bytes"] for row in selected) / 1e6,
                "selected_kv_mb_mean": statistics.fmean(row["normal_kv_read_bytes"] for row in selected) / 1e6,
                "separator_mb_mean": statistics.fmean(row["separator_read_bytes"] for row in selected) / 1e6,
                "pread_count_mean": statistics.fmean(row["pread_count"] for row in selected),
                "ssd_read_ms_mean": statistics.fmean(row["ssd_read_ms"] for row in selected),
                "contiguous_runs_per_layer_mean": (
                    statistics.fmean(len(_run_lengths(ids)) for row in selected
                                     for ids in row["selected_chunk_ids_per_layer"])
                    if run_lengths else (1.0 if method == "fullload" else 0.0)),
                "global_mean_run_length_chunks": mean_run_length,
                "max_run_length_chunks": max_run_length,
            }
            output.append(item)
    return output


SELECTOR_FIELDS = (
    "rater_count", "rater_selection_ms", "query_projection_ms",
    "probe_io_ms", "query_scoring_ms", "chunk_aggregation_ms",
    "topk_chunk_ms", "selected_id_d2h_ms", "chunk_planning_ms",
    "selector_wall_ms", "chunk_io_ms", "scatter_ms", "prefill_ms", "ttft_ms",
)


def selector_tables(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    output: list[dict] = []
    for population, turns in (("turn2", (2,)), ("turn3", (3,)),
                              ("pooled_t2_t3", (2, 3))):
        selected = [row for row in rows if row["method_key"] == "qa_chunk25"
                    and row["turn_id"] in turns]
        item: dict[str, Any] = {
            "method_key": "qa_chunk25", "method": "QA-Chunk25",
            "population": population, "n": len(selected),
            "component_timings_may_overlap": True,
            "authoritative_wall_fields": "selector_wall_ms;ttft_ms",
        }
        aliases = {
            "rater_count": ("rater_count", "n_raters"),
            "rater_selection_ms": ("rater_selection_ms", "rater_ms"),
            "query_projection_ms": ("query_projection_ms", "projection_ms"),
            "selector_wall_ms": ("selector_wall_ms", "selector_ms"),
        }
        for field in SELECTOR_FIELDS:
            names = aliases.get(field, (field,))
            values = [_number(_pick(row, *names, default=0), field, 0.0)
                      for row in selected]
            item[field + "_mean"] = statistics.fmean(values)
        # Preserve both the evaluator's precise names and the paper-facing
        # aliases requested by the experiment contract.
        item["rater_ms_mean"] = item["rater_selection_ms_mean"]
        item["projection_ms_mean"] = item["query_projection_ms_mean"]
        item["planning_ms_mean"] = item["chunk_planning_ms_mean"]
        item["prefill_interval_ms_mean"] = item["prefill_ms_mean"]
        output.append(item)
    return output


def _selection_set(row: Mapping[str, Any]) -> set[tuple[int, int]]:
    return {(layer, chunk) for layer, ids in enumerate(
        row["selected_chunk_ids_per_layer"]) for chunk in ids}


def selection_analysis(matrix: Mapping[tuple[str, int, str], Mapping[str, Any]],
                       dialogs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for did, dialog in dialogs.items():
        left = _selection_set(matrix[(did, 2, "qa_chunk25")])
        right = _selection_set(matrix[(did, 3, "qa_chunk25")])
        if not left or not right:
            raise AnalysisError("QA selection IDs are missing")
        union = left | right
        qa_t3 = matrix[(did, 3, "qa_chunk25")]["vqa_score"]
        ours_t3 = matrix[(did, 3, "ours25")]["vqa_score"]
        pairs.append({
            "dialog_id": did, "image_id": str(dialog["image_id"]),
            "jaccard": len(left & right) / len(union),
            "identical": left == right,
            "changed_chunk_count": len(left ^ right),
            "replacement_count": max(len(left - right), len(right - left)),
            "qa_minus_ours_t3_soft_score": qa_t3 - ours_t3,
        })
    jaccards = [item["jaccard"] for item in pairs]
    changed = [item for item in pairs if not item["identical"]]
    identical = [item for item in pairs if item["identical"]]
    ours_images: dict[str, Any] = {}
    for did, dialog in dialogs.items():
        image = str(dialog["image_id"])
        rows = [matrix[(did, turn, "ours25")] for turn in (2, 3)]
        fingerprints = {_stable_json(row["selected_chunk_ids_per_layer"]) for row in rows}
        permutations = {row.get("store_permutation_sha256") for row in rows}
        invariant = len(fingerprints) == 1 and len(permutations) == 1 and None not in permutations
        prefix = invariant and all(ids == list(range(len(ids)))
                                   for ids in rows[0]["selected_chunk_ids_per_layer"])
        ours_images[image] = {
            "dialog_id": did, "invariant_t2_t3": invariant,
            "exact_first_k_prefix": prefix,
            "selection_sha256": next(iter(fingerprints)) if len(fingerprints) == 1 else None,
            "store_permutation_sha256": next(iter(permutations)) if len(permutations) == 1 else None,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "qa_t2_t3": {
            "n_pairs": len(pairs),
            "mean_jaccard": statistics.fmean(jaccards),
            "median_jaccard": statistics.median(jaccards),
            "identical_selection_rate": statistics.fmean(float(item["identical"]) for item in pairs),
            "mean_changed_chunk_count": statistics.fmean(item["changed_chunk_count"] for item in pairs),
            "mean_replacement_count": statistics.fmean(item["replacement_count"] for item in pairs),
            "t3_qa_minus_ours_by_selection_change": {
                "identical": {"n": len(identical), "mean_soft_delta": (
                    statistics.fmean(item["qa_minus_ours_t3_soft_score"] for item in identical)
                    if identical else None)},
                "changed": {"n": len(changed), "mean_soft_delta": (
                    statistics.fmean(item["qa_minus_ours_t3_soft_score"] for item in changed)
                    if changed else None)},
                "interpretation": "descriptive association, not a causal effect",
            },
            "pairs": pairs,
        },
        "ours_per_image_invariance": ours_images,
        "ours_all_images_invariant": all(value["invariant_t2_t3"] for value in ours_images.values()),
        "ours_all_images_exact_first_k_prefix": all(value["exact_first_k_prefix"] for value in ours_images.values()),
    }


def clustered_soft_bootstrap(matrix: Mapping[tuple[str, int, str], Mapping[str, Any]],
                             dialogs: Mapping[str, Mapping[str, Any]],
                             resamples: int = BOOTSTRAP_RESAMPLES,
                             seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Image-cluster ratio-estimator bootstrap of QA minus Ours soft scores."""
    clusters: dict[str, list[str]] = defaultdict(list)
    for did, dialog in dialogs.items():
        clusters[str(dialog["image_id"])].append(did)
    images = sorted(clusters)
    sums = np.zeros((len(images), 3), dtype=np.float64)
    counts = np.zeros((len(images), 3), dtype=np.int64)
    for image_index, image in enumerate(images):
        for did in clusters[image]:
            for turn in TURNS:
                sums[image_index, turn - 1] += (
                    matrix[(did, turn, "qa_chunk25")]["vqa_score"]
                    - matrix[(did, turn, "ours25")]["vqa_score"])
                counts[image_index, turn - 1] += 1
    point = sums.sum(axis=0) / counts.sum(axis=0)
    rng = np.random.default_rng(seed)
    turn_samples = np.empty((resamples, 3), dtype=np.float64)
    avg_samples = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 128):
        stop = min(resamples, start + 128)
        indexes = rng.integers(0, len(images), size=(stop - start, len(images)))
        sampled_sums = sums[indexes].sum(axis=1)
        sampled_counts = counts[indexes].sum(axis=1)
        values = sampled_sums / sampled_counts
        turn_samples[start:stop] = values
        avg_samples[start:stop] = values.mean(axis=1)
    return {
        "primary_inference": True, "cluster_unit": "image",
        "estimator": "ratio of sampled score sums to sampled request counts",
        "resamples": resamples, "seed": seed,
        "turns": {f"acc{turn}": {
            "difference": float(point[turn - 1]),
            "ci95_low": float(np.percentile(turn_samples[:, turn - 1], 2.5)),
            "ci95_high": float(np.percentile(turn_samples[:, turn - 1], 97.5)),
        } for turn in TURNS},
        "avg": {"difference": float(point.mean()),
                "ci95_low": float(np.percentile(avg_samples, 2.5)),
                "ci95_high": float(np.percentile(avg_samples, 97.5))},
    }


def paired_quality(matrix: Mapping[tuple[str, int, str], Mapping[str, Any]],
                   dialogs: Mapping[str, Mapping[str, Any]], *,
                   resamples: int, seed: int) -> dict[str, Any]:
    per_turn: dict[str, Any] = {}
    for turn in TURNS:
        cells: Counter[tuple[int, int]] = Counter()
        comparisons = Counter()
        soft: list[float] = []
        for did in dialogs:
            qa = float(matrix[(did, turn, "qa_chunk25")]["vqa_score"])
            ours = float(matrix[(did, turn, "ours25")]["vqa_score"])
            qf, of = int(qa == 1.0), int(ours == 1.0)
            cells[(qf, of)] += 1
            comparisons["qa_greater" if qa > ours else
                        "ours_greater" if ours > qa else "equal"] += 1
            soft.append(qa - ours)
        per_turn[f"turn{turn}"] = {
            "n": len(soft), "qa_minus_ours_soft_accuracy": statistics.fmean(soft),
            "soft_score_comparison": dict(comparisons),
            "full_credit_diagnostic": {
                "definition": "score == 1.0",
                "both_full_credit": cells[(1, 1)],
                "qa_only_full_credit": cells[(1, 0)],
                "ours_only_full_credit": cells[(0, 1)],
                "neither_full_credit": cells[(0, 0)],
                "mcnemar_exact_two_sided_p_diagnostic": exact_mcnemar_pvalue(
                    cells[(1, 0)], cells[(0, 1)]),
                "primary_test": False,
            },
        }
    bootstrap = clustered_soft_bootstrap(
        matrix, dialogs, resamples=resamples, seed=seed)
    includes_zero = bootstrap["avg"]["ci95_low"] <= 0 <= bootstrap["avg"]["ci95_high"]
    return {
        "schema_version": SCHEMA_VERSION,
        "comparison": "QA-Chunk25 minus Ours25",
        "primary_metric": "paired soft VQA score",
        "per_turn": per_turn,
        "image_cluster_bootstrap_95ci": bootstrap,
        "avg_ci_includes_zero": includes_zero,
        "equivalence_claim_permitted": False,
        "full_credit_cells_are_diagnostic_only": True,
    }


def error_propagation(matrix: Mapping[tuple[str, int, str], Mapping[str, Any]],
                      dialogs: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict], dict[str, Any]]:
    csv_rows: list[dict] = []
    payload: dict[str, Any] = {
        "conditioning_definition": "full-credit means soft VQA score == 1.0",
        "outcome": "mean soft VQA score",
        "diagnostic_only": True,
        "methods": {},
    }
    for method in METHOD_KEYS:
        triples = [[float(matrix[(did, turn, method)]["vqa_score"])
                    for turn in TURNS] for did in dialogs]
        groups = {
            "t2_given_t1_full_credit": [b for a, b, _ in triples if a == 1.0],
            "t2_given_t1_non_full_credit": [b for a, b, _ in triples if a != 1.0],
            "t3_given_both_prior_full_credit": [c for a, b, c in triples if a == b == 1.0],
            "t3_given_any_prior_non_full_credit": [c for a, b, c in triples if not (a == b == 1.0)],
        }
        method_payload: dict[str, Any] = {}
        for metric, values in groups.items():
            record = {"n": len(values), "score_sum": math.fsum(values),
                      "mean_soft_score": statistics.fmean(values) if values else None}
            method_payload[metric] = record
            csv_rows.append({"method_key": method, "method": METHOD_LABELS[method],
                             "metric": metric, "condition": metric, **record})
        patterns: dict[str, Any] = {}
        for first in (True, False):
            for second in (True, False):
                label = ("C" if first else "W") + ("C" if second else "W")
                values = [c for a, b, c in triples
                          if (a == 1.0) == first and (b == 1.0) == second]
                record = {"n": len(values), "score_sum": math.fsum(values),
                          "mean_soft_score": statistics.fmean(values) if values else None}
                patterns[label] = record
                csv_rows.append({"method_key": method, "method": METHOD_LABELS[method],
                                 "metric": "t3_by_prior_full_credit_pattern",
                                 "condition": label, **record})
        method_payload["t3_prior_patterns"] = patterns
        payload["methods"][method] = method_payload
    return csv_rows, payload


def persistence_tables(persistence: Sequence[Mapping[str, Any]]) -> list[dict]:
    output: list[dict] = []
    for kind, attribution in (
        ("raster", "shared physical store captured by FullLoad T1; standalone-equivalent attribution to FullLoad and QA"),
        ("image_only", "physical image-only store captured by Ours T1"),
    ):
        selected = [row for row in persistence if row["store_kind"] == kind]
        timing = _stats([row["persist_ms"] for row in selected])
        writes = _stats([row["write_bytes"] / 1e6 for row in selected])
        output.append({
            "store_kind": kind, "n_images": len(selected),
            "persistence_mean_ms": timing["mean"],
            "persistence_p50_ms": timing["p50"],
            "persistence_p95_ms": timing["p95"],
            "ssd_write_mb_per_image_mean": writes["mean"],
            "attribution": attribution,
            "included_in_cache_hit_ttft": False,
        })
    return output


def session_latency_tables(matrix: Mapping[tuple[str, int, str], Mapping[str, Any]],
                           dialogs: Mapping[str, Mapping[str, Any]],
                           persistence: Sequence[Mapping[str, Any]]) -> list[dict]:
    per_image = {(row["image_id"], row["store_kind"]): row
                 for row in persistence}
    output: list[dict] = []
    attribution = {"recompute": None, "fullload": "raster",
                   "qa_chunk25": "raster", "ours25": "image_only"}
    for method in METHOD_KEYS:
        values: list[float] = []
        for did, dialog in dialogs.items():
            request_total = math.fsum(matrix[(did, turn, method)]["request_e2e_ms"]
                                      for turn in TURNS)
            kind = attribution[method]
            persist = 0.0 if kind is None else per_image[(str(dialog["image_id"]), kind)]["persist_ms"]
            values.append(request_total + persist)
        stats = _stats(values)
        output.append({
            "method_key": method, "method": METHOD_LABELS[method],
            "n_dialogues": len(values), "session_mean_ms": stats["mean"],
            "session_p50_ms": stats["p50"], "session_p95_ms": stats["p95"],
            "formula": "T1 request_e2e + attributed persistence + T2 request_e2e + T3 request_e2e",
            "persistence_attribution": kind or "none",
            "derived_standalone_equivalent": method == "qa_chunk25",
        })
    return output


def validate_frozen_config(config: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "protocols": [PROTOCOL],
        "protocol_scope": "generated_history_only",
        "seed": 1234,
        "method_keys": list(METHOD_KEYS),
        "dataset": "vqav2_validation_mt3_reconstructed",
        "benchmark_type": "MT-VQA-v2-reconstructed",
        "history_policy": "method_local_generated",
        "quality_metric": QUALITY_METRIC,
        "quality_metric_implementation": "mmimpress.dataset.vqa_score",
        "official_vqa_evaluator_claimed": False,
        "binary_correct_threshold": 0.5,
        "model": "llava-hf/llava-v1.6-vicuna-7b-hf",
        "model_revision": FROZEN_MODEL_REVISION,
        "load_4bit": True,
        "quantization": "4-bit NF4 double-quant",
        "compute_dtype": "bfloat16",
        "attention": "eager", "decoding": "greedy", "max_new_tokens": 16,
        "chunk_size": 64, "probe_heads": 3,
        "main_ttft_field": "end_to_end_ttft_ms",
        "qa_chunk_configuration": FROZEN_QA_CONFIGURATION,
        "ours_configuration": FROZEN_OURS_CONFIGURATION,
        "dataset_construction": {
            "source_index_sha256": mt_vqa_v2.SOURCE_INDEX_SHA256,
            "source_slice": "questions[1:5]",
            "dialogue_membership": "questions[1:4]",
            "dialogues_per_image": 1,
            "dialogue_overlap": False,
            "question_reuse": False,
        },
        "later_turn_policy": (
            "ReComp pixels; FullLoad/QA use raster SSD; Ours uses independent "
            "image-only repacked SSD"),
    }
    mismatch = {key: (config.get(key), value) for key, value in expected.items()
                if config.get(key) != value}
    if ("FullLoad captures/persists" not in str(config.get("turn1_policy", ""))
            or "captured by FullLoad's own T1" not in str(
                config.get("qa_raster_source_policy", ""))):
        mismatch["raster_source_provenance"] = "FullLoad-own-T1 raster capture"
    if mismatch:
        raise AnalysisError(f"frozen source configuration mismatch: {mismatch}")
    return {"passed": True, "expected": expected}


def _load_protection(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AnalysisError("protection validation must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(value, Mapping) or value.get("passed") is not True
            or value.get("missing_paths") != [] or value.get("changed_paths") != []
            or value.get("before_manifest_sha256") != value.get("after_manifest_sha256")):
        raise AnalysisError("prior-artifact protection validation failed")
    return {**dict(value), "path": str(path.resolve()), "sha256": _sha_file(path)}


def _lookup(rows: Sequence[Mapping[str, Any]], method: str,
            population: str) -> Mapping[str, Any]:
    selected = [row for row in rows if row["method_key"] == method
                and row["population"] == population]
    if len(selected) != 1:
        raise AnalysisError(f"missing aggregate {method}/{population}")
    return selected[0]


def _quality(summary: Sequence[Mapping[str, Any]], method: str) -> Mapping[str, Any]:
    return next(row for row in summary if row["method_key"] == method)


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _reduction(baseline: float, ours: float) -> float:
    if baseline <= 0:
        raise AnalysisError("TTFT reduction baseline must be positive")
    return 100.0 * (baseline - ours) / baseline


def build_analysis_markdown(*, summary: Sequence[Mapping[str, Any]],
                            latency: Sequence[Mapping[str, Any]],
                            lengths: Sequence[Mapping[str, Any]],
                            ios: Sequence[Mapping[str, Any]],
                            selectors: Sequence[Mapping[str, Any]],
                            selection: Mapping[str, Any],
                            paired: Mapping[str, Any],
                            propagation: Mapping[str, Any],
                            persistence: Sequence[Mapping[str, Any]],
                            sessions: Sequence[Mapping[str, Any]],
                            validation: Mapping[str, Any],
                            index_info: Mapping[str, Any]) -> str:
    lines = [
        "# MT-VQA-v2 Generated-History 4-Arm Analysis", "",
        "### MT-VQA-v2 Generated-History Quality", "",
        "| Method | Acc1 | Acc2 | Acc3 | Avg |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHOD_KEYS:
        row = _quality(summary, method)
        lines.append(f"| {row['method']} | {_pct(row['acc1'])} | {_pct(row['acc2'])} | {_pct(row['acc3'])} | {_pct(row['avg'])} |")
    lines.extend(["", "### MT-VQA-v2 Cache-hit TTFT", "",
                  "| Method | T2 mean (p50/p95) | T3 mean (p50/p95) | T2–T3 pooled |",
                  "|---|---:|---:|---:|"])
    for method in METHOD_KEYS:
        t2, t3, pooled = (_lookup(latency, method, name) for name in
                          ("turn2", "turn3", "pooled_t2_t3"))
        lines.append(
            f"| {METHOD_LABELS[method]} | {t2['ttft_mean_ms']:.2f} ({t2['ttft_p50_ms']:.2f}/{t2['ttft_p95_ms']:.2f}) | "
            f"{t3['ttft_mean_ms']:.2f} ({t3['ttft_p50_ms']:.2f}/{t3['ttft_p95_ms']:.2f}) | "
            f"{pooled['ttft_mean_ms']:.2f} ({pooled['ttft_p50_ms']:.2f}/{pooled['ttft_p95_ms']:.2f}) |")
    lines.extend(["", "### Quality–Efficiency Summary", "",
                  "| Method | Avg Acc | T2–T3 TTFT | SSD MB/request | Preads/request |",
                  "|---|---:|---:|---:|---:|"])
    for method in METHOD_KEYS:
        q = _quality(summary, method)
        t = _lookup(latency, method, "pooled_t2_t3")
        io_row = _lookup(ios, method, "pooled_t2_t3")
        lines.append(f"| {METHOD_LABELS[method]} | {_pct(q['avg'])} | {t['ttft_mean_ms']:.2f} ms | {io_row['ssd_read_mb_mean']:.3f} | {io_row['pread_count_mean']:.2f} |")

    ours_t = _lookup(latency, "ours25", "pooled_t2_t3")["ttft_mean_ms"]
    reductions = {method: _reduction(
        _lookup(latency, method, "pooled_t2_t3")["ttft_mean_ms"], ours_t)
        for method in ("recompute", "fullload", "qa_chunk25")}
    gaps = {name: paired["image_cluster_bootstrap_95ci"]["turns"][name]["difference"]
            for name in ("acc1", "acc2", "acc3")}
    gaps["avg"] = paired["image_cluster_bootstrap_95ci"]["avg"]["difference"]
    qa_sel = selection["qa_t2_t3"]
    full = _lookup(latency, "fullload", "pooled_t2_t3")["ttft_mean_ms"]
    recomp = _lookup(latency, "recompute", "pooled_t2_t3")["ttft_mean_ms"]
    ours_prop = propagation["methods"]["ours25"]
    qa_prop = propagation["methods"]["qa_chunk25"]
    avg_bootstrap = paired["image_cluster_bootstrap_95ci"]["avg"]
    if avg_bootstrap["ci95_low"] > 0:
        quality_gain_answer = (
            "Yes for this reconstructed workload: QA has a positive paired "
            "soft-score difference and its 95% image-cluster bootstrap CI "
            "excludes zero.")
    elif avg_bootstrap["ci95_high"] < 0:
        quality_gain_answer = (
            "No: the paired difference favors Ours, and the 95% "
            "image-cluster bootstrap CI excludes zero.")
    else:
        quality_gain_answer = (
            "No clear positive gain is established: the primary Avg "
            "bootstrap CI includes zero. This is not evidence of equivalence.")
    same_qualitative_conclusion = (
        reductions["qa_chunk25"] > 0 and avg_bootstrap["ci95_low"] <= 0)
    selection_split = qa_sel["t3_qa_minus_ours_by_selection_change"]
    ours_full = ours_prop["t3_given_both_prior_full_credit"]["mean_soft_score"]
    ours_error = ours_prop["t3_given_any_prior_non_full_credit"]["mean_soft_score"]
    qa_full = qa_prop["t3_given_both_prior_full_credit"]["mean_soft_score"]
    qa_error = qa_prop["t3_given_any_prior_non_full_credit"]["mean_soft_score"]
    ours_error_drop = (None if ours_full is None or ours_error is None
                       else ours_full - ours_error)
    qa_error_drop = (None if qa_full is None or qa_error is None
                     else qa_full - qa_error)

    lines.extend([
        "", "## Generated-history input lengths", "",
        "| Method | T2 input tokens | T3 input tokens | T2 generated tokens | T3 generated tokens |",
        "|---|---:|---:|---:|---:|",
    ])
    for method in METHOD_KEYS:
        t2 = next(row for row in lengths if row["method_key"] == method and row["turn"] == 2)
        t3 = next(row for row in lengths if row["method_key"] == method and row["turn"] == 3)
        lines.append(f"| {METHOD_LABELS[method]} | {t2['input_tokens_mean']:.2f} | {t3['input_tokens_mean']:.2f} | {t2['generated_tokens_mean']:.2f} | {t3['generated_tokens_mean']:.2f} |")
    lines.extend([
        "", "## TTFT reductions", "",
        f"- Ours vs ReComp: **{reductions['recompute']:.2f}%**.",
        f"- Ours vs FullLoad: **{reductions['fullload']:.2f}%**.",
        f"- Ours vs QA-Chunk25: **{reductions['qa_chunk25']:.2f}%**.",
        "- T2–T3 is the comparison population. Stored methods are cache hits; ReComp intentionally reprocesses pixels.",
        "", "## Selector, SSD I/O, and locality", "",
        "QA component timings may overlap. `selector_wall_ms` and request TTFT are the authoritative wall clocks; component means must not be summed.", "",
        "| Method | SSD MB | ratio vs FullLoad | probe MB | selected K/V MB | separator MB | preads | read ms | runs/layer | mean run | max run |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for method in METHOD_KEYS:
        row = _lookup(ios, method, "pooled_t2_t3")
        mean_run = "N/A" if row["global_mean_run_length_chunks"] is None else f"{row['global_mean_run_length_chunks']:.2f}"
        lines.append(f"| {METHOD_LABELS[method]} | {row['ssd_read_mb_mean']:.3f} | {row['ssd_ratio_vs_matched_fullload']:.4f} | {row['probe_mb_mean']:.3f} | {row['selected_kv_mb_mean']:.3f} | {row['separator_mb_mean']:.3f} | {row['pread_count_mean']:.2f} | {row['ssd_read_ms_mean']:.2f} | {row['contiguous_runs_per_layer_mean']:.2f} | {mean_run} | {row['max_run_length_chunks']} |")
    selector = _lookup(selectors, "qa_chunk25", "pooled_t2_t3")
    lines.extend([
        "", f"QA pooled selector wall: {selector['selector_wall_ms_mean']:.2f} ms; rater count: {selector['rater_count_mean']:.2f}; probe I/O: {selector['probe_io_ms_mean']:.2f} ms.",
        "", "## QA selection change", "",
        f"Across {qa_sel['n_pairs']} dialogues, T2↔T3 layer-tagged chunk Jaccard is mean **{qa_sel['mean_jaccard']:.6f}**, median **{qa_sel['median_jaccard']:.6f}**; identical rate **{100*qa_sel['identical_selection_rate']:.2f}%**; mean changed chunks **{qa_sel['mean_changed_chunk_count']:.2f}**.",
        f"Ours fixed-prefix T2/T3 invariance: **{'PASS' if selection['ours_all_images_invariant'] else 'FAIL'}**.",
        "", "## Paired soft quality", "",
        f"QA−Ours Acc1/Acc2/Acc3/Avg: {gaps['acc1']*100:+.2f}/{gaps['acc2']*100:+.2f}/{gaps['acc3']*100:+.2f}/{gaps['avg']*100:+.2f} pp.",
        f"The primary 10,000-resample image-cluster bootstrap Avg CI is [{paired['image_cluster_bootstrap_95ci']['avg']['ci95_low']*100:+.2f}, {paired['image_cluster_bootstrap_95ci']['avg']['ci95_high']*100:+.2f}] pp (seed 1234). Full-credit four-cells and McNemar values are diagnostic only.",
        "", "## Error propagation", "",
        "Conditioning uses full credit (`score == 1`), while every conditional outcome is mean soft VQA score. Populations are method-specific, so these are descriptive associations, not causal effects.", "",
        "| Method | T2 given T1 full | T2 given T1 non-full | T3 given both prior full | T3 given any prior non-full |",
        "|---|---:|---:|---:|---:|",
    ])
    for method in METHOD_KEYS:
        p = propagation["methods"][method]
        values = [p[name]["mean_soft_score"] for name in (
            "t2_given_t1_full_credit", "t2_given_t1_non_full_credit",
            "t3_given_both_prior_full_credit", "t3_given_any_prior_non_full_credit")]
        fmt = ["N/A" if value is None else _pct(value) for value in values]
        lines.append(f"| {METHOD_LABELS[method]} | {' | '.join(fmt)} |")
    lines.extend([
        "", "## Persistence and derived session latency", "",
        "Persistence is excluded from cache-hit TTFT. QA's raster persistence in the secondary session table is a derived standalone-equivalent attribution of the shared FullLoad-captured raster store, not a separately measured QA write.", "",
        "| Store | persist mean (p50/p95) | write MB/image |",
        "|---|---:|---:|",
    ])
    for row in persistence:
        lines.append(f"| {row['store_kind']} | {row['persistence_mean_ms']:.2f} ({row['persistence_p50_ms']:.2f}/{row['persistence_p95_ms']:.2f}) | {row['ssd_write_mb_per_image_mean']:.3f} |")
    lines.extend(["", "## MT-GQA Generated-History comparison", "",
                  "| Dataset | ReComp Avg | FullLoad Avg | QA Avg | Ours Avg | ReComp TTFT | FullLoad TTFT | QA TTFT | Ours TTFT |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                  "| MT-GQA reconstructed | 66.72% | 66.65% | 64.93% | 65.13% | 529.23 | 688.85 | 451.20 | 279.16 |",
                  f"| MT-VQA-v2 reconstructed | {_pct(_quality(summary, 'recompute')['avg'])} | {_pct(_quality(summary, 'fullload')['avg'])} | {_pct(_quality(summary, 'qa_chunk25')['avg'])} | {_pct(_quality(summary, 'ours25')['avg'])} | {recomp:.2f} | {full:.2f} | {_lookup(latency, 'qa_chunk25', 'pooled_t2_t3')['ttft_mean_ms']:.2f} | {ours_t:.2f} |",
                  "", "## Direct answers (Q1–Q9)", "",
                  "### Q1 — QA vs Ours quality gap", "",
                  f"QA−Ours Acc1/Acc2/Acc3/Avg is {gaps['acc1']*100:+.2f}/{gaps['acc2']*100:+.2f}/{gaps['acc3']*100:+.2f}/{gaps['avg']*100:+.2f} pp.", "",
                  "### Q2 — Clear quality gain from per-turn adaptation?", "",
                  quality_gain_answer, "",
                  "### Q3 — Ours TTFT reduction vs QA", "", f"**{reductions['qa_chunk25']:.2f}%**.", "",
                  "### Q4 — Ours TTFT reduction vs ReComp", "", f"**{reductions['recompute']:.2f}%**.", "",
                  "### Q5 — Is FullLoad faster than ReComp?", "",
                  f"FullLoad is {'faster' if full < recomp else 'slower'} by {abs(full-recomp):.2f} ms on pooled T2–T3 mean TTFT.", "",
                  "### Q6 — How much does QA selection change?", "",
                  f"Mean/median Jaccard is {qa_sel['mean_jaccard']:.6f}/{qa_sel['median_jaccard']:.6f}, identical rate {100*qa_sel['identical_selection_rate']:.2f}%, and mean symmetric-difference count {qa_sel['mean_changed_chunk_count']:.2f}.", "",
                  "### Q7 — Does selection change yield quality gain?", "",
                  f"When QA selection changed, the descriptive T3 QA−Ours soft-score delta is {selection_split['changed']['mean_soft_delta'] if selection_split['changed']['mean_soft_delta'] is not None else 'N/A'} (n={selection_split['changed']['n']}); when identical it is {selection_split['identical']['mean_soft_delta'] if selection_split['identical']['mean_soft_delta'] is not None else 'N/A'} (n={selection_split['identical']['n']}). This split is not randomized, so selection change alone does not identify a causal quality gain.", "",
                  "### Q8 — Is Ours unusually vulnerable to error propagation?", "",
                  f"The descriptive T3 drop from both-prior-full to any-prior-non-full is {_pct(ours_error_drop) if ours_error_drop is not None else 'N/A'} for Ours and {_pct(qa_error_drop) if qa_error_drop is not None else 'N/A'} for QA. Because method-specific conditioning sets differ, this comparison alone cannot establish that Ours is specially vulnerable.", "",
                  "### Q9 — Same qualitative conclusion as MT-GQA?", "",
                  f"{'Yes' if same_qualitative_conclusion else 'No'} under the preregistered qualitative criterion (no clear positive QA quality gain plus a positive Ours-vs-QA TTFT reduction). MT-GQA showed QA−Ours Avg −0.20 pp and Ours-vs-QA TTFT reduction 38.13%; MT-VQA-v2 shows {gaps['avg']*100:+.2f} pp with CI [{avg_bootstrap['ci95_low']*100:+.2f}, {avg_bootstrap['ci95_high']*100:+.2f}] pp and {reductions['qa_chunk25']:.2f}% TTFT reduction. No cross-dataset equivalence is claimed.",
                  "", "## Limitations", "",
                  "- No official/released MT-VQA-v2 dialogue artifact was available locally. This is a deterministic 250-image `MT-VQA-v2-reconstructed` subset and does not claim official benchmark or exact MetaCompress identity.",
                  "- The repository scorer preserves its established `min(matches/3, 1)` semantics but is not claimed byte-identical to the official VQA evaluation package.",
                  "- The frozen local VQAv2 index establishes this workload, but its original upstream stream-prefix length is not recoverable from the local config.",
                  "- OS page cache is conditioned; SSD controller cache is not flushed.",
                  "- ReComp and persisted Visual-KV paths are operationally different; no byte/output-equivalence claim is made.",
                  "- Full-credit conditioning and McNemar are diagnostic; primary quality and inference remain soft-score means and image-cluster bootstrap.",
                  "", "## Completion", "", "```text",
                  "DATASET CONSTRUCTION: PASS", "IMPLEMENTATION: PASS",
                  "FULL RUN: PASS", "VALIDATION: PASS", "```", "",
                  f"- Images: {validation['observed_images']}",
                  f"- Dialogues: {validation['observed_dialogues']}",
                  f"- Total requests: {validation['observed_logical_rows']}",
                  f"- Failed / duplicates: {validation['failed_requests']} / {validation['duplicate_requests']}",
                  f"- Index SHA256: `{index_info['sha256']}`", "",
                  "MT-VQA-v2 GENERATED-HISTORY 4-ARM EVALUATION VALIDATED: YES", ""])
    return "\n".join(lines)


def _write_raw(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    buffer = io.StringIO()
    for row in rows:
        # Source manifests are repeated only as an analysis aid internally.
        value = {key: item for key, item in row.items()
                 if key != "store_manifests"}
        buffer.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    _atomic_text(path, buffer.getvalue())


def _write_gzip(path: Path, source: Path) -> None:
    if os.path.lexists(path):
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with source.open("rb") as src, gzip.open(temporary, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_outputs(output: Path, *, rows: Sequence[Mapping[str, Any]],
                   summary: Sequence[Mapping[str, Any]], quality: Sequence[Mapping[str, Any]],
                   latency: Sequence[Mapping[str, Any]], lengths: Sequence[Mapping[str, Any]],
                   ios: Sequence[Mapping[str, Any]], selectors: Sequence[Mapping[str, Any]],
                   persistence: Sequence[Mapping[str, Any]], sessions: Sequence[Mapping[str, Any]],
                   selection: Mapping[str, Any], propagation_rows: Sequence[Mapping[str, Any]],
                   propagation: Mapping[str, Any], paired: Mapping[str, Any],
                   dataset_construction: Mapping[str, Any], validation: Mapping[str, Any],
                   config: Mapping[str, Any], report: str) -> dict[str, Any]:
    output.mkdir(parents=False, exist_ok=False)
    ordered = sorted(rows, key=lambda row: (
        row["dialog_id"], row["turn_id"], METHOD_KEYS.index(row["method_key"])))
    _write_raw(output / "raw.jsonl", ordered)
    _write_gzip(output / "raw.jsonl.gz", output / "raw.jsonl")
    _atomic_csv(output / "summary.csv", list(summary[0]), summary)
    _atomic_csv(output / "quality_by_turn.csv", list(quality[0]), quality)
    _atomic_csv(output / "ttft_by_turn.csv", list(latency[0]), latency)
    _atomic_csv(output / "token_lengths.csv", list(lengths[0]), lengths)
    _atomic_csv(output / "io_breakdown.csv", list(ios[0]), ios)
    _atomic_csv(output / "selector_breakdown.csv", list(selectors[0]), selectors)
    _atomic_csv(output / "persistence_summary.csv", list(persistence[0]), persistence)
    _atomic_csv(output / "session_latency.csv", list(sessions[0]), sessions)
    _atomic_json(output / "selection_analysis.json", selection)
    _atomic_csv(output / "error_propagation.csv", list(propagation_rows[0]), propagation_rows)
    _atomic_json(output / "error_propagation.json", propagation)
    _atomic_json(output / "paired_quality.json", paired)
    _atomic_json(output / "dataset_construction.json", dataset_construction)
    _atomic_json(output / "validation.json", validation)
    _atomic_json(output / "config.json", config)
    _atomic_text(output / "ANALYSIS.md", report)
    _atomic_text(output / "README.md", report)
    expected_names = {
        "raw.jsonl", "raw.jsonl.gz", "summary.csv", "quality_by_turn.csv",
        "ttft_by_turn.csv", "token_lengths.csv", "io_breakdown.csv",
        "selector_breakdown.csv", "persistence_summary.csv", "session_latency.csv",
        "selection_analysis.json", "error_propagation.csv", "error_propagation.json",
        "paired_quality.json", "dataset_construction.json", "validation.json",
        "config.json", "ANALYSIS.md", "README.md",
    }
    observed = {path.name for path in output.iterdir() if path.is_file()}
    if observed != expected_names:
        raise AnalysisError(f"result tree mismatch: {observed ^ expected_names}")
    hashes = {name: _sha_file(output / name) for name in sorted(expected_names)}
    completion = {
        "schema_version": SCHEMA_VERSION, "passed": True,
        "logical_rows": len(rows),
        "experiment_id": config["experiment_id"],
        "index_sha256": config["index_sha256"],
        "workload_sha256": config["workload_sha256"],
        "model_revision": config["model_revision"],
        "seed": config["seed"],
        "methods": config["methods"],
        "protocol": config["protocol"],
        "bootstrap": config["bootstrap"],
        "validation_sha256": hashes["validation.json"],
        "analysis_sha256": hashes["ANALYSIS.md"],
        "output_sha256": hashes,
    }
    _atomic_json(output / "COMPLETED", completion)
    _fsync_directory(output)
    return completion


def analyze(run_dir: Path, results_root: Path, index: Path,
            protection_validation: Path, *, expected_dialogs: int = FULL_DIALOGUES,
            expected_images: int = FULL_IMAGES,
            bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
            bootstrap_seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    run_dir, results_root, index = (Path(value).resolve()
                                    for value in (run_dir, results_root, index))
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise AnalysisError(f"invalid run directory: {run_dir}")
    if (results_root == run_dir or results_root in run_dir.parents
            or run_dir in results_root.parents or results_root.is_symlink()):
        raise AnalysisError("run/results directories overlap or result is a symlink")
    if bootstrap_resamples < 1 or bootstrap_seed < 0:
        raise AnalysisError("invalid bootstrap configuration")
    if expected_dialogs == FULL_DIALOGUES and (
            bootstrap_resamples != BOOTSTRAP_RESAMPLES
            or bootstrap_seed != BOOTSTRAP_SEED):
        raise AnalysisError("full analysis requires 10,000 resamples and seed 1234")
    config_path = run_dir / "config.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise AnalysisError("run has no regular config.json")
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    frozen = validate_frozen_config(source_config)
    dialogs, index_info = load_index(index)
    if len(dialogs) != expected_dialogs or index_info["images"] != expected_images:
        raise AnalysisError("index size differs from requested full workload")
    expected_requests = expected_dialogs * len(TURNS) * len(METHOD_KEYS)
    config_identity = {
        "n_dialogs": expected_dialogs,
        "n_turns": expected_dialogs * len(TURNS),
        "n_images": expected_images,
        "n_requests": expected_requests,
        "planned_logical_requests_this_protocol": expected_requests,
        "planned_physical_executions_this_protocol": expected_requests,
        "planned_logical_requests_generated_only": expected_requests,
        "planned_physical_executions_generated_only": expected_requests,
        "dialogues_file_sha256": index_info["sha256"],
        "source_full_workload_sha256": index_info["workload_sha256"],
        "selected_workload_sha256": index_info["workload_sha256"],
    }
    config_mismatch = {key: (source_config.get(key), value)
                       for key, value in config_identity.items()
                       if source_config.get(key) != value}
    if config_mismatch:
        raise AnalysisError(f"run config/workload identity mismatch: {config_mismatch}")
    if expected_dialogs == FULL_DIALOGUES and (
            index_info["sha256"] != EXPECTED_INDEX_SHA256
            or index_info["workload_sha256"] != EXPECTED_WORKLOAD_SHA256):
        raise AnalysisError("canonical MT-VQA-v2 index/workload hash changed")
    rows, artifacts, raw_persistence = load_rows(run_dir)
    if artifacts["experiment_id"] != str(source_config["experiment_id"]):
        raise AnalysisError("artifact/config experiment identity mismatch")
    identity_fields = (
        "dialogues_file_sha256", "source_full_workload_sha256",
        "selected_workload_sha256", "model_revision",
    )
    for row in rows:
        mismatch = {field: (row.get(field), source_config.get(field))
                    for field in identity_fields
                    if row.get(field) != source_config.get(field)}
        if mismatch:
            raise AnalysisError(f"row/config immutable identity mismatch: {mismatch}")
    matrix, validation = validate_rows(
        rows, dialogs, expected_dialogs, expected_images)
    protection = _load_protection(Path(protection_validation).resolve())
    validation["checks"].update({
        "frozen_source_configuration": True,
        "prior_artifacts_unchanged": True,
        "dataset_construction_validated": True,
    })
    validation["frozen_source_configuration"] = frozen
    validation["protection_validation"] = protection
    validation.update({
        "experiment_id": str(source_config["experiment_id"]),
        "index_sha256": index_info["sha256"],
        "workload_sha256": index_info["workload_sha256"],
        "model_revision": source_config["model_revision"],
        "seed": int(source_config["seed"]),
        "methods": list(METHOD_KEYS),
        "protocol": PROTOCOL,
        "bootstrap": {
            "resamples": int(bootstrap_resamples),
            "seed": int(bootstrap_seed), "cluster_unit": "image",
        },
    })
    validation["passed"] = all(validation["checks"].values())
    if not validation["passed"]:
        raise AnalysisError("validation failed")

    summary, quality = quality_tables(rows)
    latency, lengths = latency_and_lengths(rows)
    ios = io_tables(rows)
    selectors = selector_tables(rows)
    selection = selection_analysis(matrix, dialogs)
    propagation_rows, propagation = error_propagation(matrix, dialogs)
    paired = paired_quality(matrix, dialogs, resamples=bootstrap_resamples,
                            seed=bootstrap_seed)
    persistence = persistence_tables(raw_persistence)
    sessions = session_latency_tables(matrix, dialogs, raw_persistence)
    dataset_construction = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_type": "MT-VQA-v2-reconstructed",
        "official_released_dialogue_artifact_found_locally": False,
        "official_benchmark_identity_claimed": False,
        "seed": 1234,
        "source_dataset": "VQAv2 validation local frozen slice",
        "source_index_sha256": mt_vqa_v2.SOURCE_INDEX_SHA256,
        "evaluation_slice": "questions[1:5]",
        "dialogue_membership": "first source-order triple questions[1:4]",
        "dialogues_per_image": 1,
        "turns_per_dialogue": 3,
        "dialogue_overlap": False,
        "question_reuse": False,
        "images": expected_images, "dialogues": expected_dialogs,
        "selected_questions": expected_dialogs * 3,
        "results_used_to_choose_membership": False,
        "index": index_info,
        "passed": True,
        "disclaimer": mt_vqa_v2.DISCLAIMER,
    }
    analysis_config = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": str(source_config["experiment_id"]),
        "index_sha256": index_info["sha256"],
        "workload_sha256": index_info["workload_sha256"],
        "model_revision": source_config["model_revision"],
        "seed": int(source_config["seed"]),
        "run_dir": str(run_dir), "results_root": str(results_root),
        "index": index_info, "source_config": source_config,
        "artifact_info": artifacts, "protocol": PROTOCOL,
        "methods": list(METHOD_KEYS), "logical_requests": len(rows),
        "quality_metric": QUALITY_METRIC,
        "quality_metric_implementation": "mmimpress.dataset.vqa_score",
        "quality_is_soft": True,
        "full_credit_definition": "score == 1.0 (diagnostic only)",
        "main_ttft_population": "turns 2 and 3; stored methods cache-hit, ReComp pixels",
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed, "bootstrap_cluster": "image",
        "bootstrap": {
            "resamples": int(bootstrap_resamples),
            "seed": int(bootstrap_seed), "cluster_unit": "image",
        },
        "protection_validation": protection,
        "created_at_unix": time.time(),
    }
    report = build_analysis_markdown(
        summary=summary, latency=latency, lengths=lengths, ios=ios,
        selectors=selectors, selection=selection, paired=paired,
        propagation=propagation, persistence=persistence, sessions=sessions,
        validation=validation, index_info=index_info)
    if os.path.lexists(results_root):
        raise FileExistsError(f"results root already exists: {results_root}")
    if results_root.parent.is_symlink() or not results_root.parent.is_dir():
        raise AnalysisError("results parent is invalid")
    staging = results_root.with_name(
        f".{results_root.name}.analysis-staging-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        completion = _write_outputs(
            staging, rows=rows, summary=summary, quality=quality,
            latency=latency, lengths=lengths, ios=ios, selectors=selectors,
            persistence=persistence, sessions=sessions, selection=selection,
            propagation_rows=propagation_rows, propagation=propagation,
            paired=paired, dataset_construction=dataset_construction,
            validation=validation, config=analysis_config, report=report)
        _publish_directory_noreplace(staging, results_root)
    except Exception:
        if staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return {
        "results_root": str(results_root), "validation": validation,
        "quality": summary, "latency": latency, "selection": selection,
        "paired": paired, "completion": completion,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--index", type=Path,
                        default=ROOT / "data/mt_vqa_v2/dialogues.json")
    parser.add_argument("--protection-validation", type=Path, required=True)
    parser.add_argument("--expected-dialogs", type=int, default=FULL_DIALOGUES)
    parser.add_argument("--expected-images", type=int, default=FULL_IMAGES)
    parser.add_argument("--bootstrap-resamples", type=int,
                        default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(
        args.run_dir, args.results_root, args.index,
        args.protection_validation,
        expected_dialogs=args.expected_dialogs,
        expected_images=args.expected_images,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed)
    print(json.dumps({
        "passed": result["validation"]["passed"],
        "logical_rows": result["validation"]["observed_logical_rows"],
        "results_root": result["results_root"],
        "verdict": "MT-VQA-v2 GENERATED-HISTORY 4-ARM EVALUATION VALIDATED: YES",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

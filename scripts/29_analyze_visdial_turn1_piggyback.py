"""Analyze the VisDial Turn-1-piggyback end-to-end TTFT experiment.

The raw run is deliberately treated as immutable evidence.  This program
validates the complete 100-dialog workload before deriving any table, writes
analysis artifacts beside the run without replacing existing files, and
publishes the requested result directory with a single atomic rename.  The
main latency throughout this file is ``end_to_end_ttft_ms``; the older
``core_ttft_ms`` is diagnostic only.

Cumulative curves are paired: a cumulative value is formed independently for
each dialog first, and only then are mean/p50/p95 computed across dialogs.
For an SSD-backed method, persistence is uncharged at N=1.  At N>=2 the
conservative (``worst``) scenario charges that dialog's one persistence event
exactly once, whereas ``hidden`` is the explicitly ideal hidden-persistence
reference.
"""
from __future__ import annotations

import argparse
import ctypes
import csv
import errno
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
EXPECTED_INDEX_SHA256 = (
    "8c3dd7e983cb39e61d26362a0353b86ac84845bd7537a6331078ab7707777383"
)
EXPECTED_REQUEST_KEYS_SHA256 = (
    "395ba928a15eb45bca905aa91d1ec89617981f18b9b22338341b2a189b3f141a"
)
BASE_METHODS = ("ReComp", "Prefix25", "Prefix45")
METHOD_ORDER = ("ReComp", "FullLoad", "Prefix25", "Prefix45")
CACHE_METHODS = ("FullLoad", "Prefix25", "Prefix45")
PREFIX_METHODS = ("Prefix25", "Prefix45")
METHOD_BUDGET = {"ReComp": None, "FullLoad": None,
                 "Prefix25": 0.25, "Prefix45": 0.45}
MAX_TURN = 10
TIMING_TOLERANCE_MS = 1.0
QUALITY_TOLERANCE = 1e-9
EXPECTED_SCHEMA_VERSION = "visdial-turn1-piggyback-e2e-ttft-v1"
EXPECTED_CORRECTNESS_SCHEMA = "visdial-turn1-piggyback-correctness-v2"
REQUIRED_CORRECTNESS_CHECKS = {
    "smoke_config_complete",
    "exact_two_dialog_workload",
    "source_execution_links_to_turn1",
    "captured_provenance_no_second_forward",
    "visual_token_span_equal",
    "kv_shape_equal",
    "system_kv_equal",
    "layer0_visual_kv_bitwise_equal",
    "full_visual_kv_payload_numerically_close",
    "permutation_equal",
    "prefix25_payload_numerically_close",
    "prefix45_payload_numerically_close",
    "prefix25_first_token_equal",
    "prefix25_prediction_equal",
    "prefix45_first_token_equal",
    "prefix45_prediction_equal",
    "prefix_reader_has_no_scorer_calls",
    "prefix_loaded_chunk_ids_and_bytes_equal",
}

RESULT_NAMES = (
    "config.json",
    "summary.csv",
    "per_turn.csv",
    "persistence_per_image.csv",
    "per_dialog.csv",
    "cumulative_ttft.csv",
    "cumulative_e2e.csv",
    "break_even_ttft.csv",
    "break_even_e2e.csv",
    "quality_by_turn.csv",
    "validation.json",
    "README.md",
    "fig_cumulative_ttft.csv",
    "fig_cumulative_e2e.csv",
    "fig_per_turn_ttft.csv",
    "fig_break_even_distribution.csv",
)
RUN_OUTPUT_NAMES = tuple(
    name for name in RESULT_NAMES
    if name not in {"config.json", "persistence_per_image.csv"}
)


class AnalysisError(RuntimeError):
    """A fail-closed input or publication error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise AnalysisError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise AnalysisError(
                    f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise AnalysisError(
                    f"JSONL row is not an object at {path}:{line_number}")
            row = dict(row)
            row["_line_number"] = line_number
            rows.append(row)
    if not rows:
        raise AnalysisError(f"empty raw input: {path}")
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise AnalysisError(f"CSV has no header: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise AnalysisError(f"empty CSV input: {path}")
    return rows


def persistence_input(run_dir: Path) -> tuple[Path, list[dict[str, Any]], str]:
    """Prefer the completed CSV and fail over to the append-only JSONL.

    A killed runner may have durable ``persistence.jsonl`` but no final CSV.
    Supporting that state does not relax validation: the same one-row/image
    schema is canonicalized and the requested CSV is materialized as a new
    artifact, never by changing either source file.
    """
    csv_path = run_dir / "persistence_per_image.csv"
    jsonl_path = run_dir / "persistence.jsonl"
    if csv_path.is_file():
        return csv_path, read_csv(csv_path), "csv"
    if jsonl_path.is_file():
        rows = read_jsonl(jsonl_path)
        for row in rows:
            row.pop("_line_number", None)
        return jsonl_path, rows, "jsonl_fallback"
    raise AnalysisError(
        f"missing persistence input: expected {csv_path} or {jsonl_path}")


def field(row: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def number(value: Any, context: str, *, allow_none: bool = False) -> float | None:
    if value in (None, ""):
        if allow_none:
            return None
        raise AnalysisError(f"missing numeric field: {context}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"invalid number {context}: {value!r}") from exc
    if not math.isfinite(result):
        raise AnalysisError(f"non-finite number {context}: {value!r}")
    return result


def integer(value: Any, context: str, *, allow_none: bool = False) -> int | None:
    parsed = number(value, context, allow_none=allow_none)
    if parsed is None:
        return None
    if not float(parsed).is_integer():
        raise AnalysisError(f"not an integer {context}: {value!r}")
    return int(parsed)


def boolean(value: Any, context: str, *, allow_none: bool = False) -> bool | None:
    if value in (None, ""):
        if allow_none:
            return None
        raise AnalysisError(f"missing boolean field: {context}")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "1", "pass", "passed"}:
        return True
    if text in {"false", "no", "0", "fail", "failed"}:
        return False
    raise AnalysisError(f"invalid boolean {context}: {value!r}")


def close(left: float, right: float, tolerance: float = TIMING_TOLERANCE_MS) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def canonical_method(row: dict[str, Any]) -> str:
    raw_key = str(field(row, "method_key", default=""))
    raw_label = str(field(row, "method", "method_label", default=""))
    text = re.sub(r"[^a-z0-9]+", "", (raw_key + " " + raw_label).lower())
    if "recompute" in text or "recomp" in text:
        return "ReComp"
    if "fullload" in text:
        return "FullLoad"
    if "prefix25" in text or ("prefix" in text and "025" in text):
        return "Prefix25"
    if "prefix45" in text or ("prefix" in text and "045" in text):
        return "Prefix45"
    raise AnalysisError(
        f"unknown method at raw line {row.get('_line_number')}: "
        f"key={raw_key!r}, label={raw_label!r}")


def method_sort_key(method: str) -> int:
    try:
        return METHOD_ORDER.index(method)
    except ValueError:
        return len(METHOD_ORDER)


def parse_selection(value: Any, context: str) -> list[list[int]] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception as exc:
            raise AnalysisError(f"invalid selection JSON {context}") from exc
    if isinstance(value, dict):
        def layer_key(item: tuple[Any, Any]) -> tuple[int, str]:
            match = re.search(r"\d+", str(item[0]))
            return (int(match.group()) if match else 10**9, str(item[0]))
        value = [layer for _, layer in sorted(value.items(), key=layer_key)]
    if not isinstance(value, list) or not value:
        raise AnalysisError(f"selection is not a non-empty layer list: {context}")
    if value and all(isinstance(x, (int, float)) for x in value):
        value = [value]
    parsed: list[list[int]] = []
    for layer_index, layer in enumerate(value):
        if not isinstance(layer, list):
            raise AnalysisError(f"invalid selection layer {context}/{layer_index}")
        parsed.append([
            int(integer(x, f"{context}/layer{layer_index}")) for x in layer
        ])
    return parsed


def parse_json_object(value: Any, context: str) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception as exc:
            raise AnalysisError(f"invalid object JSON {context}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"expected object at {context}")
    return value


def normalized_answer(text: Any) -> str:
    normalized = re.sub(r"[^\w\s]", " ", str(text).lower())
    return " ".join(
        token for token in normalized.split() if token not in {"a", "an", "the"}
    )


def generative_match(prediction: Any, gold: Any) -> float:
    pred = normalized_answer(prediction)
    target = normalized_answer(gold)
    return float(
        pred == target
        or (target and pred.split()[:len(target.split())] == target.split())
    )


def percentile(values: Iterable[float], p: float) -> float | None:
    array = np.asarray(list(values), dtype=float)
    return float(np.percentile(array, p)) if array.size else None


def stats(values: Iterable[float], prefix: str) -> dict[str, Any]:
    vals = [float(value) for value in values]
    return {
        f"{prefix}_mean": float(np.mean(vals)) if vals else None,
        f"{prefix}_p50": percentile(vals, 50),
        f"{prefix}_p95": percentile(vals, 95),
    }


def sum_field(rows: Iterable[dict[str, Any]], name: str) -> float:
    return float(sum(float(row[name]) for row in rows))


def mean_field(rows: Iterable[dict[str, Any]], name: str) -> float | None:
    values = [float(row[name]) for row in rows if row.get(name) is not None]
    return float(np.mean(values)) if values else None


def check_budget(method: str, raw_budget: Any, context: str) -> float | None:
    expected = METHOD_BUDGET[method]
    actual = number(raw_budget, context, allow_none=True)
    if expected is None:
        if actual not in (None, 0.0, 1.0):
            raise AnalysisError(f"unexpected {method} budget at {context}: {actual}")
        return None
    if actual is None or abs(actual - expected) > 1e-12:
        raise AnalysisError(
            f"wrong {method} budget at {context}: expected={expected} actual={actual}")
    return expected


def canonicalize_raw(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        line = raw["_line_number"]
        context = f"raw line {line}"
        method = canonical_method(raw)
        turn = integer(field(raw, "turn_id"), context + "/turn_id")
        dialog_id = str(field(raw, "dialog_id", default=""))
        if not dialog_id or turn is None or not 1 <= turn <= MAX_TURN:
            raise AnalysisError(f"invalid request key at {context}")
        budget = check_budget(method, field(raw, "budget"), context + "/budget")
        prediction = field(raw, "prediction", "answer", default="")
        gold = field(raw, "gold", "gold_answer", "reference_answer", default="")
        if gold == "":
            raise AnalysisError(f"missing gold answer at {context}")
        quality = number(field(raw, "quality_score", "quality"),
                         context + "/quality")
        row: dict[str, Any] = {
            "schema_version": str(field(raw, "schema_version", default="")),
            "dialog_id": dialog_id,
            "image_id": str(field(raw, "image_id", default=dialog_id)),
            "turn_id": int(turn),
            "method": method,
            "budget": budget,
            "active_images": integer(field(raw, "active_images"),
                                     context + "/active_images"),
            "active_image_ids": field(raw, "active_image_ids", default=[]),
            "prediction": str(prediction),
            "gold": str(gold),
            "quality_score": quality,
            "question": str(field(raw, "question", default="")),
            "history_tokens": integer(
                field(raw, "history_tokens", "history_text_tokens"),
                context + "/history_tokens"),
            "history_sha256": str(field(
                raw, "text_history_sha256", "gold_history_sha256",
                "history_sha256", default="")),
            "suffix_ids_sha256": str(field(raw, "suffix_ids_sha256", default="")),
            "prompt_sha256": str(field(raw, "prompt_sha256", default="")),
            "input_ids_sha256": str(field(
                raw, "input_ids_sha256", "tokenized_prompt_sha256",
                "input_tensors_sha256", default="")),
            "pixel_values_sha256": str(field(
                raw, "pixel_values_sha256", "image_tensor_sha256",
                "image_input_sha256", default="")),
            "first_token_id": integer(field(raw, "first_token_id"),
                                      context + "/first_token_id",
                                      allow_none=True),
            "execution_id": str(field(raw, "execution_id", "request_id",
                                      default="")),
            "shared_persist_id": str(field(raw, "shared_persist_id",
                                           default="")),
            "store_available_at_request_start": boolean(field(
                raw, "store_available_at_request_start"),
                context + "/store_available_at_request_start",
                allow_none=True),
            "physical_store_shared": boolean(field(
                raw, "physical_store_shared_by_prefix_budgets"),
                context + "/physical_store_shared", allow_none=True),
            "execution_mode": str(field(raw, "execution_mode", "request_path",
                                        default="")),
            "prompt_build_ms": number(field(raw, "prompt_build_ms"),
                                      context + "/prompt_build_ms"),
            "tokenization_ms": number(field(raw, "tokenization_ms"),
                                      context + "/tokenization_ms"),
            "image_preprocess_ms": number(field(raw, "image_preprocess_ms"),
                                           context + "/image_preprocess_ms"),
            "input_prepare_ms": number(field(raw, "input_prepare_ms"),
                                        context + "/input_prepare_ms"),
            "input_h2d_ms": number(field(raw, "input_h2d_ms"),
                                    context + "/input_h2d_ms"),
            "pre_core_ms": number(field(raw, "pre_core_ms"),
                                  context + "/pre_core_ms"),
            "vision_ms": number(field(raw, "vision_ms"), context + "/vision_ms"),
            "saliency_capture_enabled": boolean(field(
                raw, "saliency_capture_enabled"),
                context + "/saliency_capture_enabled", allow_none=True),
            "saliency_extra_ms": number(field(raw, "saliency_extra_ms", default=0),
                                        context + "/saliency_extra_ms"),
            "core_ttft_ms": number(field(raw, "core_ttft_ms"),
                                   context + "/core_ttft_ms"),
            "end_to_end_ttft_ms": number(field(raw, "end_to_end_ttft_ms"),
                                          context + "/end_to_end_ttft_ms"),
            "decode_ms": number(field(raw, "decode_ms"), context + "/decode_ms"),
            "model_e2e_ms": number(field(raw, "model_e2e_ms"),
                                   context + "/model_e2e_ms"),
            "postprocess_ms": number(field(raw, "postprocess_ms"),
                                     context + "/postprocess_ms"),
            "request_e2e_ms": number(
                field(raw, "request_e2e_ms", "e2e_ms"),
                context + "/request_e2e_ms"),
            "e2e_ms": number(field(raw, "e2e_ms", "request_e2e_ms"),
                             context + "/e2e_ms"),
            "ssd_read_ms": number(field(raw, "ssd_read_ms", default=0),
                                  context + "/ssd_read_ms"),
            "ssd_read_bytes": integer(field(raw, "ssd_read_bytes", default=0),
                                      context + "/ssd_read_bytes"),
            "ssd_preads": integer(field(
                raw, "ssd_preads", "actual_pread_count", "ssd_read_preads",
                default=0), context + "/ssd_preads"),
            "normal_kv_read_bytes": integer(field(
                raw, "normal_kv_read_bytes", default=0),
                context + "/normal_kv_read_bytes"),
            "separator_read_bytes": integer(field(
                raw, "separator_read_bytes", default=0),
                context + "/separator_read_bytes"),
            "full_visual_kv_bytes": integer(field(
                raw, "full_visual_kv_bytes"),
                context + "/full_visual_kv_bytes"),
            "selected_visual_kv_bytes": integer(field(
                raw, "selected_visual_kv_bytes", default=0),
                context + "/selected_visual_kv_bytes"),
            "selected_kv_ratio": number(field(raw, "selected_kv_ratio"),
                                        context + "/selected_kv_ratio",
                                        allow_none=True),
            "touched_chunk_fraction": number(field(
                raw, "touched_chunk_fraction"),
                context + "/touched_chunk_fraction", allow_none=True),
            "scatter_ms": number(field(raw, "scatter_ms", default=0),
                                 context + "/scatter_ms"),
            "prefill_ms": number(field(raw, "prefill_ms", default=0),
                                 context + "/prefill_ms"),
            "vision_forward_count": integer(
                field(raw, "vision_forward_count"),
                context + "/vision_forward_count"),
            "cache_started_s": number(field(
                raw, "cache_conditioning_started_at_s",
                "page_cache_conditioning_started_at_s"),
                context + "/cache_conditioning_started_at_s", allow_none=True),
            "cache_finished_s": number(field(
                raw, "cache_conditioning_finished_at_s",
                "page_cache_conditioning_finished_at_s"),
                context + "/cache_conditioning_finished_at_s", allow_none=True),
            "cache_conditioning_method": str(field(
                raw, "page_cache_conditioning_method",
                "cache_conditioning_method", default="")),
            "cache_conditioning_excluded": boolean(field(
                raw, "page_cache_conditioning_excluded_from_ttft",
                "cache_conditioning_excluded_from_ttft"),
                context + "/cache_conditioning_excluded", allow_none=True),
            "request_started_s": number(field(raw, "request_started_at_s"),
                                        context + "/request_started_at_s"),
            "core_started_s": number(field(raw, "core_started_at_s"),
                                     context + "/core_started_at_s"),
            "first_token_s": number(field(raw, "first_token_at_s"),
                                     context + "/first_token_at_s"),
            "model_finished_s": number(field(raw, "model_finished_at_s"),
                                        context + "/model_finished_at_s"),
            "request_finished_s": number(field(
                raw, "request_finished_at_s", "postprocess_finished_at_s"),
                                          context + "/request_finished_at_s"),
            "selected_layers": parse_selection(
                field(raw, "selected_chunk_ids_per_layer"),
                context + "/selected_chunk_ids_per_layer"),
            "io_detail": parse_json_object(field(raw, "io_detail"),
                                           context + "/io_detail"),
            "n_chunks_selected": integer(
                field(raw, "n_chunks_selected"),
                context + "/n_chunks_selected", allow_none=True),
            "n_chunks_total": integer(field(raw, "n_chunks_total"),
                                      context + "/n_chunks_total",
                                      allow_none=True),
            "selection_mode": field(raw, "selection_mode"),
            "static_score_calls": integer(field(raw, "static_score_calls"),
                                          context + "/static_score_calls",
                                          allow_none=True),
            "query_score_calls": integer(field(raw, "query_score_calls"),
                                         context + "/query_score_calls",
                                         allow_none=True),
            "diversity_calls": integer(field(raw, "diversity_calls"),
                                       context + "/diversity_calls",
                                       allow_none=True),
            "raw": raw,
            "line_number": line,
        }
        phase_sum = sum(float(row[name]) for name in (
            "prompt_build_ms", "tokenization_ms", "image_preprocess_ms",
            "input_prepare_ms", "input_h2d_ms"))
        row["pre_core_phase_sum_ms"] = phase_sum
        row["pre_core_unattributed_ms"] = float(row["pre_core_ms"]) - phase_sum
        rows.append(row)
    return rows


def canonicalize_persistence(
        raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows, 2):
        context = f"persistence CSV row {index}"
        persist_start = number(field(raw, "persist_started_at_s"),
                               context + "/persist_started_at_s")
        write_finished = number(field(raw, "write_finished_at_s"),
                                context + "/write_finished_at_s")
        fsync_finished = number(field(raw, "fsync_finished_at_s"),
                                context + "/fsync_finished_at_s")
        ready = number(field(raw, "store_ready_at_s"),
                       context + "/store_ready_at_s")
        row: dict[str, Any] = {
            "schema_version": str(field(raw, "schema_version", default="")),
            "dialog_id": str(field(raw, "dialog_id", default="")),
            "image_id": str(field(raw, "image_id", default="")),
            "persist_id": str(field(raw, "persist_id", default="")),
            "source_turn_id": integer(field(raw, "source_turn_id"),
                                      context + "/source_turn_id"),
            "source_method": str(field(raw, "source_method", default="")),
            "source_execution_id": str(field(
                raw, "source_execution_id", default="")),
            "capture_from_turn1": boolean(field(raw, "capture_from_turn1"),
                                          context + "/capture_from_turn1"),
            "separate_vision_forward_count": integer(field(
                raw, "separate_vision_forward_count"),
                context + "/separate_vision_forward_count"),
            "vision_forward_count": integer(field(raw, "vision_forward_count"),
                                            context + "/vision_forward_count"),
            "saliency_capture_mode": str(field(
                raw, "saliency_capture_mode", default="")),
            "saliency_extra_ms": number(field(raw, "saliency_extra_ms"),
                                        context + "/saliency_extra_ms"),
            "saliency_postprocess_ms": number(field(
                raw, "saliency_postprocess_ms", "saliency_post_response_ms",
                "saliency_d2h_ms", default=0),
                context + "/saliency_postprocess_ms"),
            "token_mapping_ms": number(field(raw, "token_mapping_ms", default=0),
                                       context + "/token_mapping_ms"),
            "permutation_ms": number(field(raw, "permutation_ms"),
                                     context + "/permutation_ms"),
            "kv_materialize_ms": number(field(raw, "kv_materialize_ms"),
                                        context + "/kv_materialize_ms"),
            "kv_repack_ms": number(field(raw, "kv_repack_ms", "repack_ms"),
                                   context + "/kv_repack_ms"),
            "repack_ms": number(field(raw, "repack_ms", "kv_repack_ms"),
                                context + "/repack_ms"),
            "buffered_write_ms": number(field(
                raw, "buffered_write_ms", "ssd_write_ms"),
                context + "/buffered_write_ms"),
            "fsync_ms": number(field(raw, "fsync_ms"), context + "/fsync_ms"),
            "atomic_rename_ms": number(field(raw, "atomic_rename_ms", default=0),
                                       context + "/atomic_rename_ms"),
            "context_open_ms": number(field(raw, "context_open_ms"),
                                      context + "/context_open_ms"),
            "persist_unattributed_ms": number(field(
                raw, "persist_unattributed_ms", default=0),
                context + "/persist_unattributed_ms"),
            "persist_ms": number(field(raw, "persist_ms"),
                                 context + "/persist_ms"),
            "visual_kv_bytes": integer(field(raw, "visual_kv_bytes"),
                                       context + "/visual_kv_bytes"),
            "separator_sidecar_bytes": integer(field(
                raw, "separator_sidecar_bytes", default=0),
                context + "/separator_sidecar_bytes"),
            "sys_kv_bytes": integer(field(raw, "sys_kv_bytes", default=0),
                                    context + "/sys_kv_bytes"),
            "layout_metadata_bytes": integer(field(
                raw, "layout_metadata_bytes", default=0),
                context + "/layout_metadata_bytes"),
            "total_ssd_write_bytes": integer(field(
                raw, "total_ssd_write_bytes", "ssd_write_bytes"),
                context + "/total_ssd_write_bytes"),
            "permutation_sha256": str(field(
                raw, "permutation_sha256", default="")),
            "prefix_kv_sample_sha256": str(field(
                raw, "prefix_kv_sample_sha256", default="")),
            "persist_started_at_s": persist_start,
            "write_finished_at_s": write_finished,
            "fsync_finished_at_s": fsync_finished,
            "store_ready_at_s": ready,
            "durable_fsync_completed": boolean(field(
                raw, "durable_fsync_completed"),
                context + "/durable_fsync_completed"),
            "raw": raw,
        }
        if not row["dialog_id"] or not row["image_id"] or not row["persist_id"]:
            raise AnalysisError(f"missing persistence identity at {context}")
        exclusive_phase_sum = sum(float(row[name]) for name in (
            "saliency_postprocess_ms", "permutation_ms", "repack_ms",
            "buffered_write_ms", "fsync_ms", "atomic_rename_ms",
            "context_open_ms",
        ))
        row["persist_phase_sum_ms"] = exclusive_phase_sum
        row["persist_phase_identity_residual_ms"] = (
            row["persist_ms"] - exclusive_phase_sum
            - row["persist_unattributed_ms"]
        )
        rows.append(row)
    return rows


def persistence_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key != "raw"}
        for row in sorted(rows, key=lambda item: item["dialog_id"])
    ]


class Validator:
    def __init__(self) -> None:
        self.checks: dict[str, bool] = {}
        self.details: dict[str, Any] = {}
        self.failures: list[str] = []

    def check(self, name: str, condition: bool, detail: Any = None) -> None:
        passed = bool(condition)
        self.checks[name] = passed
        if detail is not None:
            self.details[name] = detail
        if not passed:
            self.failures.append(name if detail is None else f"{name}: {detail}")

    def result(self) -> dict[str, Any]:
        return {
            "passed": not self.failures and all(self.checks.values()),
            "checks": self.checks,
            "details": self.details,
            "failures": self.failures,
        }


def configured_method_order(config: dict[str, Any]) -> dict[str, list[str]] | None:
    value = field(config, "method_order_by_dialog", "dialog_method_order",
                  "method_orders_by_dialog", "resolved_method_order_by_dialog",
                  "method_order_per_dialog", "dialog_method_orders")
    if not isinstance(value, dict):
        return None
    result: dict[str, list[str]] = {}
    for dialog_id, methods in value.items():
        if not isinstance(methods, list):
            raise AnalysisError(f"invalid method order for dialog {dialog_id}")
        result[str(dialog_id)] = [canonical_method(
            {"method_key": method, "method": method, "_line_number": "config"}
        ) for method in methods]
    return result


def correctness_validation_source(config: dict[str, Any]) \
        -> tuple[Path | None, str | None]:
    source = config.get("correctness_validation_source")
    if isinstance(source, dict):
        raw_path = field(source, "path", "json", "validation_json")
        expected_hash = field(source, "sha256", "json_sha256")
    else:
        raw_path = source
        expected_hash = field(config, "correctness_validation_sha256",
                              "captured_kv_validation_sha256")
    if not raw_path:
        return None, str(expected_hash) if expected_hash else None
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve(), str(expected_hash) if expected_hash else None


def frozen_request_content(index_path: Path) -> dict[tuple[str, int], dict[str, str]]:
    payload = read_json(index_path)
    dialogs = payload.get("dialogs")
    if not isinstance(dialogs, list) or len(dialogs) != 100:
        raise AnalysisError("frozen VisDial index does not contain 100 dialogs")
    result: dict[tuple[str, int], dict[str, str]] = {}
    for dialog in dialogs:
        dialog_id = str(dialog["dialog_id"])
        image_ids = dialog.get("image_ids")
        turns = dialog.get("turns")
        if not isinstance(image_ids, list) or len(image_ids) != 1:
            raise AnalysisError(f"not a single-image dialog: {dialog_id}")
        if not isinstance(turns, list) or len(turns) != MAX_TURN:
            raise AnalysisError(f"not a 10-turn dialog: {dialog_id}")
        for turn_index, turn in enumerate(turns, 1):
            if int(turn["turn_id"]) != turn_index:
                raise AnalysisError(f"non-canonical turn order: {dialog_id}")
            history_lines = [f"Image caption: {dialog['caption']}"]
            prompt_lines = [f"Image caption: {dialog['caption']}", ""]
            for prior in turns[:turn_index - 1]:
                pair = [f"Q{prior['turn_id']}: {prior['question']}",
                        f"A{prior['turn_id']}: {prior['gold_answer']}"]
                history_lines.extend(pair)
                prompt_lines.extend(pair)
            prompt_lines.extend(
                ["", f"Current question Q{turn_index}: {turn['question']}"])
            history = "\n".join(history_lines)
            prompt_body = "\n".join(prompt_lines)
            prompt = (
                "USER: <image>\n" + prompt_body
                + "\nAnswer the current question concisely. ASSISTANT:"
            )
            result[(dialog_id, turn_index)] = {
                "image_id": str(image_ids[0]),
                "question": str(turn["question"]),
                "gold": str(turn["gold_answer"]),
                "history_sha256": hashlib.sha256(history.encode("utf-8")).hexdigest(),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
    return result


def validate_inputs(
        rows: list[dict[str, Any]], persistence: list[dict[str, Any]],
        config: dict[str, Any], input_hashes: dict[str, str]) -> dict[str, Any]:
    validator = Validator()
    methods = tuple(sorted({row["method"] for row in rows}, key=method_sort_key))
    expected_methods = BASE_METHODS + (("FullLoad",) if "FullLoad" in methods else ())
    expected_methods = tuple(sorted(expected_methods, key=method_sort_key))
    validator.check("exact_method_set", methods == expected_methods,
                    {"expected": expected_methods, "actual": methods})
    config_method_values = config.get("method_keys", config.get("methods", []))
    configured_methods: tuple[str, ...] = ()
    if isinstance(config_method_values, list):
        try:
            configured_methods = tuple(sorted((canonical_method({
                "method_key": value, "method": value,
                "_line_number": "config methods",
            }) for value in config_method_values), key=method_sort_key))
        except AnalysisError:
            configured_methods = ()
    validator.check("config_and_raw_method_sets_exact",
                    configured_methods == methods,
                    {"config": configured_methods, "raw": methods})

    request_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    exact_keys = Counter()
    dialog_method: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["dialog_id"], row["turn_id"])
        request_groups[key].append(row)
        exact_keys[(row["dialog_id"], row["turn_id"], row["method"])] += 1
        dialog_method[(row["dialog_id"], row["method"])].append(row)

    dialogs = sorted({row["dialog_id"] for row in rows})
    images = {row["image_id"] for row in rows}
    expected_keys = {(dialog, turn) for dialog in dialogs
                     for turn in range(1, MAX_TURN + 1)}
    matrix_ok = (
        len(dialogs) == 100
        and len(images) == 100
        and len(request_groups) == 1000
        and set(request_groups) == expected_keys
        and len(rows) == 1000 * len(methods)
        and all(count == 1 for count in exact_keys.values())
        and all({row["method"] for row in group} == set(methods)
                for group in request_groups.values())
    )
    validator.check("exact_100_dialog_1000_turn_complete_matrix", matrix_ok, {
        "dialogs": len(dialogs), "images": len(images),
        "requests": len(request_groups), "records": len(rows),
        "methods": methods,
    })
    validator.check("config_workload_counts_exact",
                    int(config.get("n_dialogs", -1)) == 100
                    and int(config.get("n_unique_images", -1)) == 100
                    and int(config.get("turns_per_dialog", -1)) == 10
                    and int(config.get("n_turns", -1)) == 1000
                    and int(config.get("n_method_requests", -1))
                    == 1000 * len(methods)
                    and int(config.get("observed_rows", -1)) == len(rows)
                    and int(config.get("observed_persistence_rows", -1)) == 100
                    and config.get("status") == "complete")
    validator.check("frozen_visdial_index_sha256",
                    config.get("index_sha256") == EXPECTED_INDEX_SHA256,
                    config.get("index_sha256"))
    validator.check("frozen_request_keys_sha256",
                    config.get("expected_request_keys_sha256")
                    == EXPECTED_REQUEST_KEYS_SHA256,
                    config.get("expected_request_keys_sha256"))
    configured_keys = config.get("expected_request_keys")
    configured_set = set()
    if isinstance(configured_keys, list):
        try:
            configured_set = {(str(key[0]), int(key[1])) for key in configured_keys}
        except Exception:
            configured_set = set()
    validator.check("raw_request_keys_equal_config", configured_set == set(request_groups),
                    {"configured": len(configured_set), "raw": len(request_groups)})
    validator.check("configured_request_key_payload_hash_exact",
                    isinstance(configured_keys, list)
                    and stable_json_sha256(configured_keys)
                    == EXPECTED_REQUEST_KEYS_SHA256)
    index_path = Path(str(config.get("index", "")))
    if not index_path.is_absolute():
        index_path = ROOT / index_path
    actual_index_hash = sha256_file(index_path) if index_path.is_file() else None
    validator.check("actual_index_file_hash_exact",
                    actual_index_hash == EXPECTED_INDEX_SHA256,
                    {"path": str(index_path), "sha256": actual_index_hash})
    frozen_content = (frozen_request_content(index_path)
                      if actual_index_hash == EXPECTED_INDEX_SHA256 else {})
    content_matches = bool(frozen_content)
    for row in rows:
        expected = frozen_content.get((row["dialog_id"], row["turn_id"]))
        content_matches &= expected is not None
        if expected is not None:
            content_matches &= all(
                row[name] == expected[name] for name in (
                    "image_id", "question", "gold", "history_sha256",
                    "prompt_sha256")
            )
    validator.check("raw_content_matches_frozen_canonical_index", content_matches)
    one_image_per_dialog = all(
        len({row["image_id"] for row in rows if row["dialog_id"] == dialog}) == 1
        for dialog in dialogs
    )
    validator.check("exactly_one_fixed_image_per_dialog", one_image_per_dialog)
    validator.check("single_active_image_exact",
                    all(row["active_images"] == 1
                            and row["active_image_ids"] == [row["image_id"]]
                            for row in rows))
    validator.check("dataset_and_seed_exact",
                    config.get("dataset") == "visdial_v1.0_val"
                    and int(config.get("seed", -1)) == 1234)
    validator.check("schema_version_exact_everywhere",
                    config.get("schema_version") == EXPECTED_SCHEMA_VERSION
                    and all(row["schema_version"] == EXPECTED_SCHEMA_VERSION
                            for row in rows)
                    and all(row["schema_version"] == EXPECTED_SCHEMA_VERSION
                            for row in persistence))
    validator.check("gold_teacher_forced_history",
                    config.get("history_policy") == "gold_teacher_forced")
    validator.check("same_model_and_decoding_contract",
                    config.get("model") == "llava-hf/llava-v1.6-vicuna-7b-hf"
                    and config.get("decoding") == "greedy"
                    and bool(config.get("load_4bit"))
                    and config.get("attention") == "eager")
    validator.check("generation_and_chunk_contract",
                    int(config.get("max_new_tokens", 16)) == 16
                    and int(config.get("chunk_size", 64)) == 64)
    validator.check("imageonly_repack_sequential_prefix_contract",
                    config.get("physical_layout") == "visionzip_image_only"
                    and config.get("retrieval")
                    == "sequential_first_k_repacked_chunks"
                    and config.get("separator_policy") == "sidecar")
    correctness_path, expected_correctness_hash = \
        correctness_validation_source(config)
    correctness_payload = None
    actual_correctness_hash = None
    if correctness_path is not None and correctness_path.is_file():
        actual_correctness_hash = sha256_file(correctness_path)
        correctness_payload = read_json(correctness_path)
    correctness_checks = (
        correctness_payload.get("checks", {})
        if isinstance(correctness_payload, dict) else {}
    )
    validator.check("captured_kv_correctness_source_hash_exact",
                    correctness_payload is not None
                    and bool(expected_correctness_hash)
                    and actual_correctness_hash == expected_correctness_hash, {
                        "path": str(correctness_path) if correctness_path else None,
                        "expected_sha256": expected_correctness_hash,
                        "actual_sha256": actual_correctness_hash,
                    })
    validator.check("captured_kv_vs_direct_builder_smoke_passed",
                    isinstance(correctness_payload, dict)
                    and correctness_payload.get("schema_version")
                    == EXPECTED_CORRECTNESS_SCHEMA
                    and correctness_payload.get("passed") is True
                    and isinstance(correctness_checks, dict)
                    and set(correctness_checks) == REQUIRED_CORRECTNESS_CHECKS
                    and all(value is True for value in correctness_checks.values()),
                    correctness_checks)

    identical_history = True
    identical_inputs = True
    identical_gold = True
    history_monotonic = True
    for key, group in request_groups.items():
        identical_history &= bool(group[0]["history_sha256"])
        identical_history &= len({row["history_sha256"] for row in group}) == 1
        identical_history &= bool(group[0]["suffix_ids_sha256"])
        identical_history &= len({row["suffix_ids_sha256"] for row in group}) == 1
        identical_gold &= len({(row["gold"], row["question"], row["prompt_sha256"])
                               for row in group}) == 1
        if key[1] == 1:
            for hash_name in ("input_ids_sha256", "pixel_values_sha256"):
                identical_inputs &= bool(group[0][hash_name])
                identical_inputs &= len({row[hash_name] for row in group}) == 1
    for (dialog, method), group in dialog_method.items():
        ordered = sorted(group, key=lambda row: row["turn_id"])
        counts = [int(row["history_tokens"]) for row in ordered]
        history_monotonic &= all(right >= left
                                 for left, right in zip(counts, counts[1:]))
    validator.check("gold_history_and_suffix_hashes_identical", identical_history)
    validator.check("questions_gold_and_prompt_hashes_identical", identical_gold)
    validator.check("turn1_tokenized_inputs_and_pixels_identical", identical_inputs)
    validator.check("history_tokens_monotonic", history_monotonic)

    quality_residuals = [
        abs(float(row["quality_score"])
            - generative_match(row["prediction"], row["gold"]))
        for row in rows
    ]
    validator.check("auxiliary_quality_recomputed_exactly",
                    max(quality_residuals, default=math.inf) <= QUALITY_TOLERANCE,
                    {"max_abs_residual": max(quality_residuals, default=None)})

    residuals = defaultdict(list)
    timestamp_order = True
    cache_outside = True
    timing_positive = True
    for row in rows:
        residuals["ttft_pre_core_plus_core"].append(abs(
            row["end_to_end_ttft_ms"] - row["pre_core_ms"]
            - row["core_ttft_ms"]))
        residuals["model_e2e_ttft_plus_decode"].append(abs(
            row["model_e2e_ms"] - row["end_to_end_ttft_ms"]
            - row["decode_ms"]))
        residuals["request_e2e_model_plus_postprocess"].append(abs(
            row["request_e2e_ms"] - row["model_e2e_ms"]
            - row["postprocess_ms"]))
        residuals["e2e_alias"].append(abs(row["e2e_ms"] - row["request_e2e_ms"]))
        residuals["timestamp_ttft"].append(abs(
            1000.0 * (row["first_token_s"] - row["request_started_s"])
            - row["end_to_end_ttft_ms"]))
        residuals["timestamp_pre_core"].append(abs(
            1000.0 * (row["core_started_s"] - row["request_started_s"])
            - row["pre_core_ms"]))
        residuals["timestamp_model_e2e"].append(abs(
            1000.0 * (row["model_finished_s"] - row["request_started_s"])
            - row["model_e2e_ms"]))
        residuals["timestamp_request_e2e"].append(abs(
            1000.0 * (row["request_finished_s"] - row["request_started_s"])
            - row["request_e2e_ms"]))
        timestamp_order &= (
            row["request_started_s"] <= row["core_started_s"]
            <= row["first_token_s"] <= row["model_finished_s"]
            <= row["request_finished_s"]
        )
        timing_positive &= (
            row["end_to_end_ttft_ms"] > 0
            and row["core_ttft_ms"] > 0
            and row["decode_ms"] >= 0
            and row["request_e2e_ms"] >= row["end_to_end_ttft_ms"]
            and row["pre_core_ms"] >= row["pre_core_phase_sum_ms"] - 1.0
        )
        if row["method"] in CACHE_METHODS and row["turn_id"] >= 2:
            cache_outside &= row["cache_started_s"] is not None
            cache_outside &= row["cache_finished_s"] is not None
            if row["cache_started_s"] is not None and row["cache_finished_s"] is not None:
                cache_outside &= (
                    row["cache_started_s"] <= row["cache_finished_s"]
                    <= row["request_started_s"]
                )
            cache_outside &= (
                row["cache_conditioning_method"]
                == "posix_fadvise_DONTNEED"
                and row["cache_conditioning_excluded"] is True
            )
    max_residuals = {name: max(values, default=math.inf)
                     for name, values in residuals.items()}
    validator.check("timing_values_and_phase_partition_valid", timing_positive)
    validator.check("exact_timing_identities_within_1ms",
                    all(value <= TIMING_TOLERANCE_MS
                        for value in max_residuals.values()), max_residuals)
    validator.check("request_timestamp_order_valid", timestamp_order)
    validator.check("page_cache_conditioning_before_request_timer", cache_outside)
    validator.check("config_main_metric_end_to_end_ttft",
                    config.get("main_ttft_metric") == "end_to_end_ttft_ms"
                    and bool(config.get("input_preprocessing_timed")))
    validator.check("config_cache_conditioning_excluded",
                    config.get("page_cache_conditioning_in_ttft") is False)
    validator.check("cold_buffered_pread_not_odirect",
                    bool(config.get("cold_page_cache"))
                    and str(config.get("ssd_read_api", "")).lower() in {
                        "buffered_pread", "os.pread", "pread", "buffered os.pread"
                    }
                    and config.get("o_direct") is False
                    and config.get("ssd_controller_cache_flushed") is False)

    turn1_valid = True
    reuse_valid = True
    first_k_valid = True
    nested_valid = True
    score_calls_valid = True
    real_pread_valid = True
    io_geometry_valid = True
    persistence_geometry = {row["dialog_id"]: row for row in persistence}
    selection_fingerprints: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        raw = row["raw"]
        persisted = persistence_geometry.get(row["dialog_id"])
        io_geometry_valid &= persisted is not None
        if persisted is not None:
            io_geometry_valid &= (
                row["full_visual_kv_bytes"] == persisted["visual_kv_bytes"])
        if row["turn_id"] == 1:
            turn1_valid &= row["execution_mode"] == "normal_multimodal_turn1"
            turn1_valid &= row["vision_forward_count"] == 1
            turn1_valid &= row["vision_ms"] > 0
            turn1_valid &= row["ssd_read_bytes"] == 0 and row["ssd_preads"] == 0
            turn1_valid &= abs(row["ssd_read_ms"]) <= 1e-9
            turn1_valid &= row["selected_layers"] is None
            turn1_valid &= (
                row["saliency_capture_enabled"]
                is (row["method"] in PREFIX_METHODS)
            )
            io_geometry_valid &= row["selected_visual_kv_bytes"] == 0
        elif row["method"] == "ReComp":
            reuse_valid &= row["execution_mode"] in {
                "recompute", "normal_multimodal_recompute"}
            reuse_valid &= row["vision_forward_count"] == 1
            reuse_valid &= row["vision_ms"] > 0
            reuse_valid &= row["ssd_read_bytes"] == 0 and row["ssd_preads"] == 0
            reuse_valid &= abs(row["ssd_read_ms"]) <= 1e-9
            reuse_valid &= row["selected_layers"] is None
            reuse_valid &= row["saliency_capture_enabled"] is False
            io_geometry_valid &= row["selected_visual_kv_bytes"] == 0
        elif row["method"] in CACHE_METHODS:
            wanted_modes = (
                {"ssd_fullload", "ssd_full_load"}
                if row["method"] == "FullLoad"
                else {"ssd_prefix", "ssd_sequential_prefix"}
            )
            reuse_valid &= row["execution_mode"] in wanted_modes
            reuse_valid &= row["vision_forward_count"] == 0
            reuse_valid &= abs(row["vision_ms"]) <= 1e-9
            reuse_valid &= row["saliency_capture_enabled"] is False
            real_pread_valid &= row["ssd_read_bytes"] > 0 and row["ssd_preads"] > 0
            real_pread_valid &= row["ssd_read_ms"] > 0
            actual_bytes = integer(field(
                raw, "total_actual_pread_bytes", "actual_pread_bytes",
                default=row["ssd_read_bytes"]), "actual pread bytes")
            real_pread_valid &= actual_bytes == row["ssd_read_bytes"]
            io_detail = row["io_detail"]
            if not io_detail:
                real_pread_valid = False
            else:
                try:
                    detailed_bytes = sum(int(item.get("bytes", 0))
                                         for item in io_detail.values())
                    detailed_preads = sum(int(item.get("preads", 0))
                                          for item in io_detail.values())
                except (AttributeError, TypeError, ValueError):
                    real_pread_valid = False
                else:
                    real_pread_valid &= detailed_bytes == row["ssd_read_bytes"]
                    real_pread_valid &= detailed_preads == row["ssd_preads"]
            io_geometry_valid &= (
                row["normal_kv_read_bytes"] + row["separator_read_bytes"]
                == row["ssd_read_bytes"]
                and row["selected_visual_kv_bytes"] == row["ssd_read_bytes"]
            )
            if row["method"] == "FullLoad":
                io_geometry_valid &= (
                    row["ssd_read_bytes"] == row["full_visual_kv_bytes"]
                    and row["normal_kv_read_bytes"]
                    == row["full_visual_kv_bytes"]
                    and row["separator_read_bytes"] == 0
                    and row["ssd_preads"] == 64
                )
            if row["method"] in PREFIX_METHODS:
                score_calls = (row["static_score_calls"], row["query_score_calls"],
                               row["diversity_calls"])
                score_calls_valid &= score_calls == (0, 0, 0)
                score_calls_valid &= row["selection_mode"] == "prefix"
                layers = row["selected_layers"]
                total = row["n_chunks_total"]
                selected = row["n_chunks_selected"]
                if layers is None or total is None or selected is None:
                    first_k_valid = False
                else:
                    expected_k = max(1, min(total, round(float(row["budget"]) * total)))
                    wanted = list(range(expected_k))
                    first_k_valid &= selected == expected_k
                    first_k_valid &= all(layer == wanted for layer in layers)
                    io_geometry_valid &= len(layers) == 32
                    io_geometry_valid &= row["ssd_preads"] == 2 * len(layers) + 1
                    io_geometry_valid &= (
                        row["touched_chunk_fraction"] is not None
                        and close(row["touched_chunk_fraction"], expected_k / total,
                                  tolerance=1e-9)
                        and row["selected_kv_ratio"] is not None
                        and 0.0 < row["selected_kv_ratio"] < 1.0
                    )
                    if persisted is not None:
                        io_geometry_valid &= (
                            row["separator_read_bytes"]
                            == persisted["separator_sidecar_bytes"])
                    fingerprint = stable_json_sha256(layers)
                    prior = selection_fingerprints[row["method"]].setdefault(
                        row["image_id"], fingerprint)
                    first_k_valid &= prior == fingerprint

    by_key_method = {(row["dialog_id"], row["turn_id"], row["method"]): row
                     for row in rows}
    for dialog, turn in request_groups:
        if turn == 1:
            continue
        p25 = by_key_method.get((dialog, turn, "Prefix25"))
        p45 = by_key_method.get((dialog, turn, "Prefix45"))
        if p25 is None or p45 is None:
            nested_valid = False
            continue
        if p25["selected_layers"] is None or p45["selected_layers"] is None:
            nested_valid = False
            continue
        nested_valid &= len(p25["selected_layers"]) == len(p45["selected_layers"])
        if len(p25["selected_layers"]) == len(p45["selected_layers"]):
            nested_valid &= all(set(left).issubset(set(right))
                                for left, right in zip(p25["selected_layers"],
                                                       p45["selected_layers"]))
        io_geometry_valid &= p25["ssd_read_bytes"] < p45["ssd_read_bytes"]
        io_geometry_valid &= p45["ssd_read_bytes"] < p45["full_visual_kv_bytes"]
    validator.check("turn1_all_methods_normal_inference_vision1_ssd0", turn1_valid)
    validator.check("turn2_10_recomp_vision1_and_cache_methods_vision0", reuse_valid)
    validator.check("prefix_reads_are_real_measured_preads", real_pread_valid)
    validator.check("ssd_bytes_preads_and_persisted_geometry_exact",
                    io_geometry_valid)
    validator.check("prefix_exact_first_k_budget", first_k_valid)
    validator.check("prefix25_nested_in_prefix45", nested_valid)
    validator.check("no_static_query_or_diversity_score_calls", score_calls_valid)

    turn1_groups = [group for (dialog, turn), group in request_groups.items()
                    if turn == 1]
    prediction_agreements = [
        len({row["prediction"] for row in group}) == 1 for group in turn1_groups
    ]
    first_token_agreements = [
        None not in {row["first_token_id"] for row in group}
        and len({row["first_token_id"] for row in group}) == 1
        for group in turn1_groups
    ]
    validator.check("turn1_prediction_and_first_token_agreement",
                    all(prediction_agreements) and all(first_token_agreements), {
                        "prediction_agreement_rate": float(np.mean(prediction_agreements)),
                        "first_token_agreement_rate": float(np.mean(first_token_agreements)),
                    })

    persistence_by_dialog = {row["dialog_id"]: row for row in persistence}
    persistence_ids = [row["persist_id"] for row in persistence]
    one_persist = (
        len(persistence) == 100
        and len(persistence_by_dialog) == 100
        and len(set(persistence_ids)) == 100
        and set(persistence_by_dialog) == set(dialogs)
        and len({row["image_id"] for row in persistence}) == 100
    )
    validator.check("one_shared_persistence_event_per_dialog_image", one_persist)
    persistence_links_valid = True
    for row in rows:
        persisted = persistence_by_dialog.get(row["dialog_id"])
        persistence_links_valid &= persisted is not None
        if persisted is not None:
            persistence_links_valid &= row["image_id"] == persisted["image_id"]
            persistence_links_valid &= row["shared_persist_id"] == persisted["persist_id"]
        persistence_links_valid &= row["physical_store_shared"] is True
        expected_available = row["turn_id"] >= 2 and row["method"] in CACHE_METHODS
        persistence_links_valid &= (
            row["store_available_at_request_start"] is expected_available)
    validator.check("raw_rows_link_one_shared_store_with_correct_availability",
                    persistence_links_valid)
    persist_source_valid = all(
        row["source_turn_id"] == 1
        and row["capture_from_turn1"] is True
        and row["separate_vision_forward_count"] == 0
        and row["vision_forward_count"] == 1
        and row["saliency_capture_mode"] in {
            "piggyback_penultimate_attention",
            "penultimate_cls_to_patch_attention_same_vision_forward",
        }
        and bool(row["source_execution_id"])
        for row in persistence
    )
    validator.check("persistence_uses_turn1_capture_without_second_forward",
                    persist_source_valid)
    persist_timestamps = all(
        row["persist_started_at_s"] <= row["write_finished_at_s"]
        <= row["fsync_finished_at_s"] <= row["store_ready_at_s"]
        and close(1000.0 * (row["store_ready_at_s"]
                            - row["persist_started_at_s"]),
                  row["persist_ms"])
        for row in persistence
    )
    persist_durable = all(
        row["durable_fsync_completed"] is True
        and row["fsync_ms"] >= 0
        and row["context_open_ms"] >= 0
        and row["total_ssd_write_bytes"] > 0
        and len(row["permutation_sha256"]) == 64
        and len(row["prefix_kv_sample_sha256"]) == 64
        for row in persistence
    )
    validator.check("persistence_timestamp_identity_and_store_ready", persist_timestamps)
    validator.check("persistence_fsync_bytes_and_hashes_recorded", persist_durable)
    persistence_phase_residual = max(
        (abs(row["persist_phase_identity_residual_ms"]) for row in persistence),
        default=math.inf)
    validator.check("persistence_exclusive_phase_identity_within_1ms",
                    persistence_phase_residual <= TIMING_TOLERANCE_MS,
                    {"max_abs_residual_ms": persistence_phase_residual})
    persist_after_answer = True
    source_method_matches = True
    saliency_capture_matches = True
    for dialog, persist in persistence_by_dialog.items():
        turn1 = request_groups[(dialog, 1)]
        matching_source = [
            row for row in turn1
            if row["execution_id"] == persist["source_execution_id"]
        ]
        persist_after_answer &= len(matching_source) == 1
        if matching_source:
            persist_after_answer &= (
                persist["persist_started_at_s"]
                >= matching_source[0]["request_finished_s"]
            )
            try:
                persisted_method = canonical_method({
                    "method_key": persist["source_method"],
                    "method": persist["source_method"],
                    "_line_number": f"persistence/{dialog}",
                })
            except AnalysisError:
                persisted_method = ""
            source_method_matches &= persisted_method == matching_source[0]["method"]
            saliency_capture_matches &= close(
                persist["saliency_extra_ms"],
                matching_source[0]["saliency_extra_ms"], tolerance=1e-6)
    validator.check("persistence_starts_after_turn1_response", persist_after_answer)
    validator.check("persistence_source_method_and_execution_match_raw",
                    source_method_matches)
    validator.check("persistence_saliency_matches_source_turn1_capture",
                    saliency_capture_matches)
    validator.details["persistence_source_method_counts"] = dict(Counter(
        canonical_method({
            "method_key": row["source_method"],
            "method": row["source_method"],
            "_line_number": "persistence source",
        }) for row in persistence
    ))

    # The source CSV has one persistence event shared by both Prefix budgets;
    # no per-method duplicate is permitted.
    validator.check("turn1_persist_is_not_double_counted_by_budget",
                    all(not row["raw"].get("budget") for row in persistence))
    no_leakage = (
        int(config.get("future_turn_calibration_count", 0)) == 0
        and int(config.get("calibration_questions", 0)) == 0
        and config.get("layout_uses_dataset_question") is False
        and str(config.get("saliency_source", "")).lower() in {
            "image_only_turn1_vision_attention",
            "turn1_vision_penultimate_cls_to_patch_attention_head_sum",
            "piggyback_penultimate_attention",
        }
        and all(
            boolean(row["raw"].get("layout_uses_dataset_question"),
                    "persistence/layout_uses_dataset_question") is False
            and integer(row["raw"].get("future_turns_used_for_layout"),
                        "persistence/future_turns_used_for_layout") == 0
            and integer(row["raw"].get("calibration_questions"),
                        "persistence/calibration_questions") == 0
            for row in persistence
        )
    )
    validator.check("image_only_saliency_without_question_or_future_leakage",
                    no_leakage)

    configured_order = configured_method_order(config)
    order_valid = configured_order is not None and set(configured_order) == set(dialogs)
    position_counts = {method: Counter() for method in methods}
    if configured_order is not None:
        for dialog in dialogs:
            wanted = configured_order.get(dialog, [])
            order_valid &= set(wanted) == set(methods) and len(wanted) == len(methods)
            for position, method in enumerate(wanted):
                position_counts[method][position] += 1
            for turn in range(1, MAX_TURN + 1):
                observed = [row["method"] for row in request_groups[(dialog, turn)]]
                order_valid &= observed == wanted
    balanced = all(
        max(counts.values(), default=0) - min(
            (counts.get(i, 0) for i in range(len(methods))), default=0) <= 1
        for counts in position_counts.values()
    )
    validator.check("configured_method_order_matches_raw", order_valid)
    validator.check("deterministically_rotated_order_balances_positions", balanced,
                    {method: dict(counts) for method, counts in position_counts.items()})

    validator.details["input_sha256"] = input_hashes
    validator.details["method_set"] = list(methods)
    validator.details["selection_fingerprints"] = selection_fingerprints
    validator.details["timing_tolerance_ms"] = TIMING_TOLERANCE_MS
    return validator.result()


def aggregate_rows(rows: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in sorted({row["method"] for row in rows}, key=method_sort_key):
        selected = [row for row in rows if row["method"] == method]
        if scope == "reuse_only_turns_2_10":
            selected = [row for row in selected if row["turn_id"] >= 2]
        if not selected:
            continue
        result: dict[str, Any] = {
            "scope": scope,
            "method": method,
            "budget": METHOD_BUDGET[method],
            "n_requests": len(selected),
            "n_dialogs": len({row["dialog_id"] for row in selected}),
            "quality_mean": mean_field(selected, "quality_score"),
            "ssd_read_bytes_mean": mean_field(selected, "ssd_read_bytes"),
            "ssd_read_bytes_total": int(sum_field(selected, "ssd_read_bytes")),
            "ssd_preads_mean": mean_field(selected, "ssd_preads"),
            "pre_core_unattributed_ms_mean": mean_field(
                selected, "pre_core_unattributed_ms"),
        }
        for name in (
            "end_to_end_ttft_ms", "core_ttft_ms", "pre_core_ms",
            "prompt_build_ms", "tokenization_ms", "image_preprocess_ms",
            "input_prepare_ms", "input_h2d_ms", "vision_ms", "ssd_read_ms",
            "scatter_ms", "prefill_ms", "decode_ms", "model_e2e_ms",
            "postprocess_ms", "request_e2e_ms",
        ):
            result.update(stats((row[name] for row in selected), name))
        output.append(result)
    baseline = next((row for row in output if row["method"] == "ReComp"), None)
    if baseline:
        for row in output:
            base_ttft = baseline["end_to_end_ttft_ms_mean"]
            base_e2e = baseline["request_e2e_ms_mean"]
            row["end_to_end_ttft_delta_vs_recomp_ms"] = (
                row["end_to_end_ttft_ms_mean"] - base_ttft)
            row["end_to_end_ttft_reduction_vs_recomp_pct"] = (
                100.0 * (1.0 - row["end_to_end_ttft_ms_mean"] / base_ttft))
            row["request_e2e_delta_vs_recomp_ms"] = (
                row["request_e2e_ms_mean"] - base_e2e)
            row["quality_delta_vs_recomp"] = (
                row["quality_mean"] - baseline["quality_mean"])
    return output


def build_per_turn(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    methods = sorted({row["method"] for row in rows}, key=method_sort_key)
    for turn in range(1, MAX_TURN + 1):
        turn_rows = [row for row in rows if row["turn_id"] == turn]
        baseline_values = [row["end_to_end_ttft_ms"] for row in turn_rows
                           if row["method"] == "ReComp"]
        baseline_mean = float(np.mean(baseline_values))
        for method in methods:
            selected = [row for row in turn_rows if row["method"] == method]
            result: dict[str, Any] = {
                "turn_id": turn,
                "method": method,
                "budget": METHOD_BUDGET[method],
                "n_dialogs": len(selected),
                "quality_mean": mean_field(selected, "quality_score"),
                "history_tokens_mean": mean_field(selected, "history_tokens"),
                "ssd_read_bytes_mean": mean_field(selected, "ssd_read_bytes"),
                "ssd_read_bytes_total": int(sum_field(selected, "ssd_read_bytes")),
                "ssd_preads_mean": mean_field(selected, "ssd_preads"),
            }
            for name in (
                "end_to_end_ttft_ms", "core_ttft_ms", "pre_core_ms",
                "vision_ms", "ssd_read_ms", "scatter_ms", "prefill_ms",
                "decode_ms", "request_e2e_ms",
            ):
                result.update(stats((row[name] for row in selected), name))
            result["end_to_end_ttft_delta_vs_recomp_ms"] = (
                result["end_to_end_ttft_ms_mean"] - baseline_mean)
            result["end_to_end_ttft_reduction_vs_recomp_pct"] = (
                100.0 * (1.0 - result["end_to_end_ttft_ms_mean"] / baseline_mean))
            output.append(result)
    return output


def build_per_dialog(
        rows: list[dict[str, Any]], persistence_by_dialog: dict[str, dict[str, Any]]) \
        -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dialog_id"], row["method"])].append(row)
    for (dialog, method), group in sorted(
            groups.items(), key=lambda item: (item[0][0], method_sort_key(item[0][1]))):
        group = sorted(group, key=lambda row: row["turn_id"])
        persist = persistence_by_dialog[dialog]
        ttft_hidden_2 = sum_field(group[:2], "end_to_end_ttft_ms")
        e2e_hidden_2 = sum_field(group[:2], "request_e2e_ms")
        is_cache = method in CACHE_METHODS
        output.append({
            "dialog_id": dialog,
            "image_id": group[0]["image_id"],
            "method": method,
            "budget": METHOD_BUDGET[method],
            "n_turns": len(group),
            "quality_mean": mean_field(group, "quality_score"),
            "turn1_end_to_end_ttft_ms": group[0]["end_to_end_ttft_ms"],
            "reuse_turns_end_to_end_ttft_ms_mean": mean_field(group[1:],
                                                               "end_to_end_ttft_ms"),
            "request_ttft_sum_10turn_ms": sum_field(group,
                                                     "end_to_end_ttft_ms"),
            "request_e2e_sum_10turn_ms": sum_field(group, "request_e2e_ms"),
            "ssd_read_bytes_turn2_10": int(sum_field(group[1:],
                                                     "ssd_read_bytes")),
            "one_time_ssd_write_bytes": (
                persist["total_ssd_write_bytes"] if is_cache else 0),
            "persist_ms_once": persist["persist_ms"] if is_cache else 0.0,
            "two_turn_ttft_hidden_ms": ttft_hidden_2,
            "two_turn_ttft_worst_ms": (
                ttft_hidden_2 + persist["persist_ms"] if is_cache
                else ttft_hidden_2),
            "two_turn_e2e_hidden_ms": e2e_hidden_2,
            "two_turn_e2e_worst_ms": (
                e2e_hidden_2 + persist["persist_ms"] if is_cache
                else e2e_hidden_2),
            "ten_turn_ttft_hidden_ms": sum_field(group, "end_to_end_ttft_ms"),
            "ten_turn_ttft_worst_ms": (
                sum_field(group, "end_to_end_ttft_ms") + persist["persist_ms"]
                if is_cache else sum_field(group, "end_to_end_ttft_ms")),
            "ten_turn_e2e_hidden_ms": sum_field(group, "request_e2e_ms"),
            "ten_turn_e2e_worst_ms": (
                sum_field(group, "request_e2e_ms") + persist["persist_ms"]
                if is_cache else sum_field(group, "request_e2e_ms")),
        })
    return output


def per_dialog_cumulative(
        rows: list[dict[str, Any]], persistence_by_dialog: dict[str, dict[str, Any]],
        latency_field: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dialog_id"], row["method"])].append(row)
    output: list[dict[str, Any]] = []
    for (dialog, method), group in groups.items():
        ordered = sorted(group, key=lambda row: row["turn_id"])
        running = 0.0
        for turn_count, row in enumerate(ordered, 1):
            running += float(row[latency_field])
            if method == "ReComp":
                scenarios = (("recompute_no_persistence", running, False),)
            else:
                persist_ms = float(persistence_by_dialog[dialog]["persist_ms"])
                scenarios = (
                    ("worst", running + (persist_ms if turn_count >= 2 else 0.0),
                     turn_count >= 2),
                    ("hidden", running, False),
                )
            for scenario, cumulative, charged in scenarios:
                output.append({
                    "dialog_id": dialog,
                    "method": method,
                    "budget": METHOD_BUDGET[method],
                    "scenario": scenario,
                    "turns_n": turn_count,
                    "cumulative_ms": cumulative,
                    "persistence_charged": charged,
                })
    return output


def aggregate_cumulative(dialog_rows: list[dict[str, Any]], metric_name: str) \
        -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in dialog_rows:
        groups[(row["turns_n"], row["method"], row["scenario"])].append(row)
    baseline = {
        (row["dialog_id"], row["turns_n"]): row["cumulative_ms"]
        for row in dialog_rows if row["method"] == "ReComp"
    }
    output: list[dict[str, Any]] = []
    for (turns_n, method, scenario), rows in sorted(
            groups.items(), key=lambda item: (
                item[0][0], method_sort_key(item[0][1]), item[0][2])):
        values = [row["cumulative_ms"] for row in rows]
        deltas = [row["cumulative_ms"]
                  - baseline[(row["dialog_id"], turns_n)] for row in rows]
        base_values = [baseline[(row["dialog_id"], turns_n)] for row in rows]
        result = {
            "metric": metric_name,
            "turns_n": turns_n,
            "method": method,
            "budget": METHOD_BUDGET[method],
            "scenario": scenario,
            "n_dialogs": len(rows),
            "persistence_charged": method != "ReComp" and turns_n >= 2
                                     and scenario == "worst",
            **stats(values, "cumulative_ms"),
            **stats(deltas, "delta_vs_recomp_ms"),
            "reduction_vs_recomp_pct_of_means": (
                100.0 * (1.0 - float(np.mean(values)) / float(np.mean(base_values)))
            ),
            "strictly_better_dialog_fraction": float(np.mean(
                [delta < 0 for delta in deltas])),
            "is_direct_two_turn_case": turns_n == 2,
        }
        output.append(result)
    return output


def build_break_even(
        dialog_cumulative: list[dict[str, Any]],
        aggregate_cumulative_rows: list[dict[str, Any]], metric_name: str) \
        -> list[dict[str, Any]]:
    baseline = {
        (row["dialog_id"], row["turns_n"]): float(row["cumulative_ms"])
        for row in dialog_cumulative if row["method"] == "ReComp"
    }
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in dialog_cumulative:
        if row["method"] in PREFIX_METHODS:
            grouped[(row["dialog_id"], row["method"], row["scenario"])].append(row)
    output: list[dict[str, Any]] = []
    for (dialog, method, scenario), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: row["turns_n"])
        comparisons = {
            row["turns_n"]: float(row["cumulative_ms"])
            < baseline[(dialog, row["turns_n"])] for row in rows
        }
        eligible = [turn for turn in range(2, MAX_TURN + 1)
                    if comparisons.get(turn, False)]
        break_even = min(eligible) if eligible else None
        stays = (all(comparisons[turn]
                     for turn in range(break_even, MAX_TURN + 1))
                 if break_even is not None else None)
        ours = {row["turns_n"]: float(row["cumulative_ms"]) for row in rows}
        output.append({
            "metric": metric_name,
            "level": "dialog",
            "dialog_id": dialog,
            "method": method,
            "budget": METHOD_BUDGET[method],
            "scenario": scenario,
            "break_even_turn": break_even,
            "censored_at_turn_10": break_even is None,
            "stays_better_through_turn_10": stays,
            "better_at_turn_2": comparisons[2],
            "two_turn_ours_ms": ours[2],
            "two_turn_recomp_ms": baseline[(dialog, 2)],
            "two_turn_delta_ms": ours[2] - baseline[(dialog, 2)],
            "ten_turn_ours_ms": ours[10],
            "ten_turn_recomp_ms": baseline[(dialog, 10)],
            "ten_turn_delta_ms": ours[10] - baseline[(dialog, 10)],
        })

    mean_lookup = {
        (row["method"], row["scenario"], row["turns_n"]):
        float(row["cumulative_ms_mean"])
        for row in aggregate_cumulative_rows
    }
    for method in PREFIX_METHODS:
        for scenario in ("worst", "hidden"):
            comparisons = {
                turn: mean_lookup[(method, scenario, turn)]
                < mean_lookup[("ReComp", "recompute_no_persistence", turn)]
                for turn in range(1, MAX_TURN + 1)
            }
            eligible = [turn for turn in range(2, MAX_TURN + 1)
                        if comparisons[turn]]
            break_even = min(eligible) if eligible else None
            stays = (all(comparisons[turn]
                         for turn in range(break_even, MAX_TURN + 1))
                     if break_even is not None else None)
            ours2 = mean_lookup[(method, scenario, 2)]
            recomp2 = mean_lookup[("ReComp", "recompute_no_persistence", 2)]
            ours10 = mean_lookup[(method, scenario, 10)]
            recomp10 = mean_lookup[("ReComp", "recompute_no_persistence", 10)]
            output.append({
                "metric": metric_name,
                "level": "aggregate_mean_curve",
                "dialog_id": None,
                "method": method,
                "budget": METHOD_BUDGET[method],
                "scenario": scenario,
                "break_even_turn": break_even,
                "censored_at_turn_10": break_even is None,
                "stays_better_through_turn_10": stays,
                "better_at_turn_2": comparisons[2],
                "two_turn_ours_ms": ours2,
                "two_turn_recomp_ms": recomp2,
                "two_turn_delta_ms": ours2 - recomp2,
                "ten_turn_ours_ms": ours10,
                "ten_turn_recomp_ms": recomp10,
                "ten_turn_delta_ms": ours10 - recomp10,
            })
    return output


def build_quality_by_turn(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    methods = sorted({row["method"] for row in rows}, key=method_sort_key)
    for turn in range(1, MAX_TURN + 1):
        baseline = [row for row in rows
                    if row["turn_id"] == turn and row["method"] == "ReComp"]
        baseline_mean = mean_field(baseline, "quality_score")
        for method in methods:
            selected = [row for row in rows
                        if row["turn_id"] == turn and row["method"] == method]
            values = [row["quality_score"] for row in selected]
            mean = float(np.mean(values))
            output.append({
                "turn_id": turn,
                "method": method,
                "budget": METHOD_BUDGET[method],
                "n_dialogs": len(selected),
                "quality_metric": "generative_auxiliary_normalized_match_not_official_visdial",
                "quality_mean": mean,
                "quality_p50": percentile(values, 50),
                "quality_p95": percentile(values, 95),
                "quality_delta_vs_recomp": mean - float(baseline_mean),
            })
    return output


def add_experiment_level_fields(
        summary_rows: list[dict[str, Any]],
        persistence: list[dict[str, Any]],
        cumulative_ttft: list[dict[str, Any]],
        cumulative_e2e: list[dict[str, Any]],
        break_even_ttft: list[dict[str, Any]],
        break_even_e2e: list[dict[str, Any]]) -> None:
    """Attach persistence, direct N=2, N=10 and break-even to summary rows."""
    persist_stats: dict[str, Any] = {}
    for name in (
        "saliency_extra_ms", "saliency_postprocess_ms", "permutation_ms",
        "repack_ms", "buffered_write_ms", "fsync_ms", "context_open_ms",
        "persist_ms", "total_ssd_write_bytes",
    ):
        persist_stats.update(stats((row[name] for row in persistence), name))

    def curve_lookup(source: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
        return {(row["method"], row["scenario"], row["turns_n"]): row
                for row in source}

    ttft = curve_lookup(cumulative_ttft)
    e2e = curve_lookup(cumulative_e2e)
    be_ttft = {(row["method"], row["scenario"]): row for row in break_even_ttft
               if row["level"] == "aggregate_mean_curve"}
    be_e2e = {(row["method"], row["scenario"]): row for row in break_even_e2e
              if row["level"] == "aggregate_mean_curve"}
    for row in summary_rows:
        method = row["method"]
        is_cache = method in CACHE_METHODS
        for name, value in persist_stats.items():
            row[name] = value if is_cache else 0.0
        if method == "ReComp":
            scenarios = ("recompute_no_persistence",)
        else:
            scenarios = ("worst", "hidden")
        for scenario in scenarios:
            for turns_n in (2, 10):
                ttft_row = ttft[(method, scenario, turns_n)]
                e2e_row = e2e[(method, scenario, turns_n)]
                stem = f"n{turns_n}_{scenario}"
                row[f"{stem}_cumulative_ttft_ms_mean"] = \
                    ttft_row["cumulative_ms_mean"]
                row[f"{stem}_cumulative_e2e_ms_mean"] = \
                    e2e_row["cumulative_ms_mean"]
                row[f"{stem}_ttft_delta_vs_recomp_ms"] = \
                    ttft_row["delta_vs_recomp_ms_mean"]
                row[f"{stem}_e2e_delta_vs_recomp_ms"] = \
                    e2e_row["delta_vs_recomp_ms_mean"]
            if method in PREFIX_METHODS:
                row[f"{scenario}_ttft_break_even_turn"] = \
                    be_ttft[(method, scenario)]["break_even_turn"]
                row[f"{scenario}_e2e_break_even_turn"] = \
                    be_e2e[(method, scenario)]["break_even_turn"]


def build_break_even_distribution(break_even_rows: list[dict[str, Any]]) \
        -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in break_even_rows:
        if row["level"] == "dialog":
            groups[(row["method"], row["scenario"])].append(row)
    for (method, scenario), rows in sorted(groups.items()):
        counts = Counter(
            "censored>10" if row["break_even_turn"] is None
            else str(row["break_even_turn"]) for row in rows
        )
        for label in [str(turn) for turn in range(2, MAX_TURN + 1)] + ["censored>10"]:
            output.append({
                "method": method,
                "budget": METHOD_BUDGET[method],
                "scenario": scenario,
                "break_even_turn_or_censored": label,
                "n_dialogs": counts[label],
                "fraction": counts[label] / len(rows),
            })
    return output


def validate_derived(
        rows: list[dict[str, Any]], persistence_by_dialog: dict[str, dict[str, Any]],
        cumulative_ttft_dialog: list[dict[str, Any]],
        cumulative_e2e_dialog: list[dict[str, Any]],
        cumulative_ttft_aggregate: list[dict[str, Any]],
        cumulative_e2e_aggregate: list[dict[str, Any]],
        break_even_ttft: list[dict[str, Any]],
        break_even_e2e: list[dict[str, Any]]) -> dict[str, Any]:
    validator = Validator()
    for name, values, raw_field in (
        ("ttft", cumulative_ttft_dialog, "end_to_end_ttft_ms"),
        ("e2e", cumulative_e2e_dialog, "request_e2e_ms"),
    ):
        raw_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            raw_groups[(row["dialog_id"], row["method"])].append(row)
        ok = True
        n1_ok = True
        once_ok = True
        for value in values:
            ordered = sorted(raw_groups[(value["dialog_id"], value["method"])],
                             key=lambda row: row["turn_id"])
            hidden = sum(float(row[raw_field])
                         for row in ordered[:value["turns_n"]])
            expected = hidden
            if value["scenario"] == "worst" and value["turns_n"] >= 2:
                expected += persistence_by_dialog[value["dialog_id"]]["persist_ms"]
            ok &= close(value["cumulative_ms"], expected, tolerance=1e-8)
            if value["turns_n"] == 1 and value["method"] in CACHE_METHODS:
                n1_ok &= close(value["cumulative_ms"], hidden, tolerance=1e-8)
                n1_ok &= value["persistence_charged"] is False
            if value["scenario"] == "worst" and value["turns_n"] >= 2:
                once_ok &= close(value["cumulative_ms"] - hidden,
                                 persistence_by_dialog[value["dialog_id"]]["persist_ms"],
                                 tolerance=1e-8)
        validator.check(f"{name}_cumulative_dialog_first_algebra", ok)
        validator.check(f"{name}_n1_persistence_uncharged", n1_ok)
        validator.check(f"{name}_worst_charges_persistence_exactly_once", once_ok)

    def validate_aggregate(dialog_values: list[dict[str, Any]],
                           aggregate_values: list[dict[str, Any]]) -> bool:
        groups: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in dialog_values:
            groups[(row["turns_n"], row["method"], row["scenario"])].append(row)
        baseline = {
            (row["dialog_id"], row["turns_n"]): row["cumulative_ms"]
            for row in dialog_values if row["method"] == "ReComp"
        }
        ok = len(aggregate_values) == len(groups)
        for row in aggregate_values:
            key = (row["turns_n"], row["method"], row["scenario"])
            members = groups.get(key, [])
            if not members:
                ok = False
                continue
            values = [member["cumulative_ms"] for member in members]
            deltas = [member["cumulative_ms"]
                      - baseline[(member["dialog_id"], member["turns_n"])]
                      for member in members]
            bases = [baseline[(member["dialog_id"], member["turns_n"])]
                     for member in members]
            expected = {
                "cumulative_ms_mean": float(np.mean(values)),
                "cumulative_ms_p50": percentile(values, 50),
                "cumulative_ms_p95": percentile(values, 95),
                "delta_vs_recomp_ms_mean": float(np.mean(deltas)),
                "delta_vs_recomp_ms_p50": percentile(deltas, 50),
                "delta_vs_recomp_ms_p95": percentile(deltas, 95),
                "reduction_vs_recomp_pct_of_means": 100.0 * (
                    1.0 - float(np.mean(values)) / float(np.mean(bases))),
                "strictly_better_dialog_fraction": float(np.mean(
                    [delta < 0 for delta in deltas])),
            }
            ok &= row["n_dialogs"] == len(members)
            ok &= all(close(row[name], value, tolerance=1e-10)
                      for name, value in expected.items())
        return ok

    validator.check("ttft_aggregate_mean_p50_p95_recomputed_dialog_first",
                    validate_aggregate(cumulative_ttft_dialog,
                                       cumulative_ttft_aggregate))
    validator.check("e2e_aggregate_mean_p50_p95_recomputed_dialog_first",
                    validate_aggregate(cumulative_e2e_dialog,
                                       cumulative_e2e_aggregate))

    def validate_break_even(rows_be: list[dict[str, Any]], cumulative: list[dict[str, Any]]) -> bool:
        lookup = {(row["dialog_id"], row["method"], row["scenario"],
                   row["turns_n"]): row["cumulative_ms"] for row in cumulative}
        baseline = {(row["dialog_id"], row["turns_n"]): row["cumulative_ms"]
                    for row in cumulative if row["method"] == "ReComp"}
        ok = True
        for row in rows_be:
            if row["level"] != "dialog":
                continue
            eligible = [turn for turn in range(2, MAX_TURN + 1)
                        if lookup[(row["dialog_id"], row["method"],
                                   row["scenario"], turn)]
                        < baseline[(row["dialog_id"], turn)]]
            expected = min(eligible) if eligible else None
            ok &= row["break_even_turn"] == expected
            ok &= row["censored_at_turn_10"] == (expected is None)
            if expected is not None:
                expected_stays = all(
                    lookup[(row["dialog_id"], row["method"], row["scenario"], turn)]
                    < baseline[(row["dialog_id"], turn)]
                    for turn in range(expected, MAX_TURN + 1))
                ok &= row["stays_better_through_turn_10"] == expected_stays
        return ok

    validator.check("ttft_break_even_strict_and_censoring_valid",
                    validate_break_even(break_even_ttft, cumulative_ttft_dialog))
    validator.check("e2e_break_even_strict_and_censoring_valid",
                    validate_break_even(break_even_e2e, cumulative_e2e_dialog))

    def validate_aggregate_break_even(
            rows_be: list[dict[str, Any]],
            aggregate_values: list[dict[str, Any]]) -> bool:
        lookup = {(row["method"], row["scenario"], row["turns_n"]):
                  row["cumulative_ms_mean"] for row in aggregate_values}
        ok = True
        selected = [row for row in rows_be
                    if row["level"] == "aggregate_mean_curve"]
        ok &= len(selected) == len(PREFIX_METHODS) * 2
        for row in selected:
            comparisons = {
                turn: lookup[(row["method"], row["scenario"], turn)]
                < lookup[("ReComp", "recompute_no_persistence", turn)]
                for turn in range(2, MAX_TURN + 1)
            }
            eligible = [turn for turn, better in comparisons.items() if better]
            expected = min(eligible) if eligible else None
            expected_stays = (
                all(comparisons[turn]
                    for turn in range(expected, MAX_TURN + 1))
                if expected is not None else None
            )
            ok &= row["break_even_turn"] == expected
            ok &= row["censored_at_turn_10"] == (expected is None)
            ok &= row["stays_better_through_turn_10"] == expected_stays
        return ok

    validator.check("ttft_aggregate_mean_curve_break_even_recomputed",
                    validate_aggregate_break_even(
                        break_even_ttft, cumulative_ttft_aggregate))
    validator.check("e2e_aggregate_mean_curve_break_even_recomputed",
                    validate_aggregate_break_even(
                        break_even_e2e, cumulative_e2e_aggregate))
    return validator.result()


def claim_verdict(summary_rows: list[dict[str, Any]],
                  break_even_ttft: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    reuse = {row["method"]: row for row in summary_rows
             if row["scope"] == "reuse_only_turns_2_10"}
    recomp = reuse["ReComp"]
    aggregate_worst = {
        row["method"]: row for row in break_even_ttft
        if row["level"] == "aggregate_mean_curve" and row["scenario"] == "worst"
    }
    latency_win = {
        method: reuse[method]["end_to_end_ttft_ms_mean"]
        < recomp["end_to_end_ttft_ms_mean"] for method in PREFIX_METHODS
    }
    worst_break_even = {
        method: aggregate_worst[method]["break_even_turn"]
        for method in PREFIX_METHODS
    }
    tradeoff = (
        reuse["Prefix45"]["quality_mean"] > reuse["Prefix25"]["quality_mean"]
        and reuse["Prefix45"]["end_to_end_ttft_ms_mean"]
        > reuse["Prefix25"]["end_to_end_ttft_ms_mean"]
    )
    if any(latency_win.values()) and any(value is not None
                                         for value in worst_break_even.values()) and tradeoff:
        verdict = "SUPPORTED"
    elif any(latency_win.values()):
        verdict = "PARTIALLY SUPPORTED"
    else:
        verdict = "NOT SUPPORTED"
    return verdict, {
        "reuse_end_to_end_ttft_lower_than_recomp": latency_win,
        "aggregate_worst_case_break_even_turn": worst_break_even,
        "ordered_quality_latency_tradeoff_25_to_45": tradeoff,
    }


def csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise AnalysisError("refusing to emit a headerless empty CSV")
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def fmt(value: Any, digits: int = 2) -> str:
    return "—" if value in (None, "") else f"{float(value):.{digits}f}"


def build_readme(
        config: dict[str, Any], summary_rows: list[dict[str, Any]],
        persistence: list[dict[str, Any]], per_turn: list[dict[str, Any]],
        cumulative_ttft: list[dict[str, Any]],
        cumulative_e2e: list[dict[str, Any]], break_even_ttft: list[dict[str, Any]],
        break_even_e2e: list[dict[str, Any]], verdict: str,
        verdict_details: dict[str, Any], validation: dict[str, Any]) -> str:
    overall = {row["method"]: row for row in summary_rows
               if row["scope"] == "overall_all_turns"}
    reuse = {row["method"]: row for row in summary_rows
             if row["scope"] == "reuse_only_turns_2_10"}
    methods = sorted(overall, key=method_sort_key)
    overall_lines = [
        "| Method | Quality† | End-to-end TTFT mean/p50/p95 (ms) | Core TTFT mean (ms) | Request E2E mean (ms) | SSD read/request (MB) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        row = overall[method]
        overall_lines.append(
            f"| {method} | {fmt(row['quality_mean'], 3)} | "
            f"{fmt(row['end_to_end_ttft_ms_mean'])}/"
            f"{fmt(row['end_to_end_ttft_ms_p50'])}/"
            f"{fmt(row['end_to_end_ttft_ms_p95'])} | "
            f"{fmt(row['core_ttft_ms_mean'])} | "
            f"{fmt(row['request_e2e_ms_mean'])} | "
            f"{fmt(row['ssd_read_bytes_mean'] / 1e6)} |"
        )

    persistence_lines = []
    for name in ("saliency_extra_ms", "saliency_postprocess_ms", "permutation_ms",
                 "repack_ms", "buffered_write_ms", "fsync_ms", "context_open_ms",
                 "persist_ms"):
        values = [row[name] for row in persistence]
        persistence_lines.append(
            f"| {name} | "
            f"{'inside Turn-1 TTFT; not added to persist' if name == 'saliency_extra_ms' else 'post-response persistence path'} | "
            f"{fmt(np.mean(values))} | {fmt(percentile(values, 50))} | "
            f"{fmt(percentile(values, 95))} |"
        )

    turn_lookup = {(row["turn_id"], row["method"]): row for row in per_turn}
    turn1_lines = []
    for method in methods:
        row = turn_lookup[(1, method)]
        turn1_lines.append(
            f"| {method} | {fmt(row['end_to_end_ttft_ms_mean'])} | "
            f"{fmt(row['end_to_end_ttft_ms_p50'])} | "
            f"{fmt(row['end_to_end_ttft_ms_p95'])} | "
            f"{fmt(row['vision_ms_mean'])} |"
        )
    trend_lines = []
    for turn in range(1, MAX_TURN + 1):
        values = {method: turn_lookup[(turn, method)]["end_to_end_ttft_ms_mean"]
                  for method in ("ReComp", "Prefix25", "Prefix45")}
        trend_lines.append(
            f"| {turn} | {fmt(values['ReComp'])} | {fmt(values['Prefix25'])} | "
            f"{fmt(values['Prefix45'])} | "
            f"{fmt(values['ReComp'] - values['Prefix25'])} | "
            f"{fmt(values['ReComp'] - values['Prefix45'])} |"
        )

    def cumulative_at(source: list[dict[str, Any]], turn: int) -> list[str]:
        selected = [row for row in source if row["turns_n"] == turn]
        order = {("ReComp", "recompute_no_persistence"): 0,
                 ("Prefix25", "worst"): 1, ("Prefix45", "worst"): 2,
                 ("Prefix25", "hidden"): 3, ("Prefix45", "hidden"): 4,
                 ("FullLoad", "worst"): 5, ("FullLoad", "hidden"): 6}
        selected.sort(key=lambda row: order.get((row["method"], row["scenario"]), 99))
        return [
            f"| {row['method']} | {row['scenario']} | "
            f"{fmt(row['cumulative_ms_mean'])} | {fmt(row['cumulative_ms_p50'])} | "
            f"{fmt(row['cumulative_ms_p95'])} | "
            f"{fmt(row['reduction_vs_recomp_pct_of_means'])}% |"
            for row in selected
            if row["method"] in {"ReComp", "Prefix25", "Prefix45"}
        ]

    def break_even_lines(source: list[dict[str, Any]]) -> list[str]:
        rows = [row for row in source if row["level"] == "aggregate_mean_curve"]
        return [
            f"| {row['method']} | {row['scenario']} | "
            f"{row['break_even_turn'] if row['break_even_turn'] is not None else 'censored >10'} | "
            f"{row['stays_better_through_turn_10'] if row['stays_better_through_turn_10'] is not None else '—'} | "
            f"{fmt(row['two_turn_delta_ms'])} | {fmt(row['ten_turn_delta_ms'])} |"
            for row in sorted(rows, key=lambda row: (method_sort_key(row["method"]),
                                                       row["scenario"]), reverse=False)
        ]

    index_hash = config.get("index_sha256")
    request_hash = config.get("expected_request_keys_sha256")
    order_policy = field(config, "method_order_policy", "method_ordering",
                         default="deterministically rotated; see config.json")
    write_mb = np.mean([row["total_ssd_write_bytes"] for row in persistence]) / 1e6
    source_counts = Counter(canonical_method({
        "method_key": row["source_method"], "method": row["source_method"],
        "_line_number": "persistence source",
    }) for row in persistence)
    text = f"""# VisDial Turn-1 Piggyback — End-to-End TTFT

Validation: **{'PASS' if validation['passed'] else 'FAIL'}**  
Claim verdict: **{verdict}**

## Timing boundary (main metric)

`end_to_end_ttft_ms` is the main TTFT metric.  OS page-cache conditioning is
completed first and is outside the request timer.  The timer then starts
immediately before prompt construction/tokenization and includes input
preparation, initial CPU→GPU H2D, either vision recomputation or SSD KV loading,
cache reconstruction/scatter, prefill, first-token selection, and the
first-token CUDA synchronization.  `core_ttft_ms` begins only after initial
request preparation/H2D and is retained as a diagnostic for comparison with
older runs.  `request_e2e_ms` continues through decode and the common
postprocessing boundary. Per-phase `vision_ms` is a CUDA-event device-timeline
diagnostic on GPU runs, whereas the server-side TTFT boundary is the synchronized
wall-clock request timestamp; phase sums therefore remain diagnostic.

Cold-cache experiments evict the OS page cache before request timing using
`posix_fadvise(DONTNEED)`. This cache-conditioning step is excluded from TTFT
because it is benchmark setup rather than request processing. Reads use
buffered `pread`; `O_DIRECT` is not used, and the SSD controller cache is not
forcibly flushed.  Consequently this is an OS-page-cache cold condition, not a
claim of a physically “true cold SSD”.

## Workload and controls

- Dataset: VisDial v1.0 validation, exactly 100 dialogs / 100 unique images /
  10 turns per dialog / 1,000 requests per method.
- Frozen index SHA256: `{index_hash}`
- Frozen request-key SHA256: `{request_hash}`
- History: gold teacher-forced; per-request history and token hashes are equal
  across methods.
- Method order: `{order_policy}`. Position balance and the exact recorded order
  are validated.
- Methods: {', '.join(methods)}. No query-dependent selector, Static+Diverse,
  SparseVLM, or MaxMin is present.

## Overall (all turns)

{chr(10).join(overall_lines)}

† Quality is the existing normalized generative-match auxiliary metric, not an
official VisDial MRR/R@K/Mean Rank/NDCG score.

## Turn 1 and shared persistence semantics

Every method performs normal pixel-based multimodal inference at Turn 1.  The
Prefix methods do not start with a pre-existing SSD cache. Visual KV and
penultimate image-only saliency are captured from that same Turn-1 forward;
there is no second vision forward and no separate cache-build inference. The
one persistence event per image is shared by Prefix25 and Prefix45—it is not
duplicated per budget. `saliency_extra_ms` is Turn-1 capture instrumentation
overhead and is not added again to `persist_ms`.

Only one Prefix execution per dialog retains the already-created cache object
for the physical persistence event (source counts: `{dict(source_counts)}`).
Both Prefix arms use the same saliency instrumentation, and each counterfactual
cumulative curve uses that method's own measured normal Turn-1 request plus the
one shared persistence measurement. Cache-object export occurs after the
first-token boundary; this source choice therefore cannot improve reported
TTFT, while any small post-first-token bookkeeping difference remains visible
in request E2E.

| Method | Turn-1 E2E TTFT mean | p50 | p95 | vision mean |
|---|---:|---:|---:|---:|
{chr(10).join(turn1_lines)}

| Accounting component | Boundary | mean (ms) | p50 (ms) | p95 (ms) |
|---|---|---:|---:|---:|
{chr(10).join(persistence_lines)}

Mean one-time SSD write: {write_mb:.2f} MB/image. ReComp performs no SSD read or
write. The correct claim is that the proposed path spends SSD traffic to avoid
repeated vision recomputation—not that it uses less SSD I/O than ReComp.

## Reuse turns (Turn 2–10)

| Method | E2E TTFT mean (ms) | Δ vs ReComp (ms) | reduction vs ReComp | quality† | SSD read/request (MB) |
|---|---:|---:|---:|---:|---:|
"""
    for method in methods:
        row = reuse[method]
        text += (
            f"| {method} | {fmt(row['end_to_end_ttft_ms_mean'])} | "
            f"{fmt(row['end_to_end_ttft_delta_vs_recomp_ms'])} | "
            f"{fmt(row['end_to_end_ttft_reduction_vs_recomp_pct'])}% | "
            f"{fmt(row['quality_mean'], 3)} | "
            f"{fmt(row['ssd_read_bytes_mean'] / 1e6)} |\n"
        )

    text += f"""

### Per-turn end-to-end TTFT trend

| Turn | ReComp | Prefix25 | Prefix45 | ReComp−P25 | ReComp−P45 |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(trend_lines)}

## Two-turn and ten-turn cumulative latency

Cumulative TTFT below is explicitly the **sum of per-request TTFTs**, not the
wall-clock completion time of a multi-request session. Cumulative E2E is also
formed as a per-request service-time sum: `worst` is the back-to-back completion
case, while `hidden` is an ideal service-latency reference and is not a measured
session wall clock (unmeasured user think time is not added). Values are
calculated per dialog first and then aggregated across dialogs. At N=1, both
Prefix scenarios leave persistence uncharged. At N≥2, `worst` adds `persist_ms`
exactly once; `hidden` excludes it.

### Cumulative end-to-end TTFT — N=2

| Method | Scenario | mean (ms) | p50 | p95 | reduction vs ReComp |
|---|---|---:|---:|---:|---:|
{chr(10).join(cumulative_at(cumulative_ttft, 2))}

### Cumulative end-to-end TTFT — N=10

| Method | Scenario | mean (ms) | p50 | p95 | reduction vs ReComp |
|---|---|---:|---:|---:|---:|
{chr(10).join(cumulative_at(cumulative_ttft, 10))}

### Scenario-adjusted cumulative E2E — N=2

| Method | Scenario | mean (ms) | p50 | p95 | reduction vs ReComp |
|---|---|---:|---:|---:|---:|
{chr(10).join(cumulative_at(cumulative_e2e, 2))}

### Scenario-adjusted cumulative E2E — N=10

| Method | Scenario | mean (ms) | p50 | p95 | reduction vs ReComp |
|---|---|---:|---:|---:|---:|
{chr(10).join(cumulative_at(cumulative_e2e, 10))}

## Break-even on aggregate mean curves

The break-even turn is the smallest strict `N >= 2` for which Prefix is lower
than paired ReComp. `censored >10` means no crossing was observed within this
10-turn workload. Per-dialog distributions are preserved in the CSV files.

### TTFT sum

| Method | Scenario | break-even N | stays better through N=10 | N=2 delta (ms) | N=10 delta (ms) |
|---|---|---:|---:|---:|---:|
{chr(10).join(break_even_lines(break_even_ttft))}

### Scenario-adjusted E2E sum

| Method | Scenario | break-even N | stays better through N=10 | N=2 delta (ms) | N=10 delta (ms) |
|---|---|---:|---:|---:|---:|
{chr(10).join(break_even_lines(break_even_e2e))}

## Claim verdict

**{verdict}.** Machine-readable grounds: `{json.dumps(verdict_details, sort_keys=True)}`.
This verdict follows the measured data, including persistence on the
conservative path; the ideal hidden scenario is kept separate.

## Artifact semantics

- `summary.csv`: overall and reuse-only method summaries.
- `per_turn.csv` / `quality_by_turn.csv`: Turn 1…10 statistics.
- `per_dialog.csv`: paired dialog-level request sums and direct N=2/N=10 values.
- `cumulative_ttft.csv`: per-request TTFT sums after dialog-first aggregation.
- `cumulative_e2e.csv`: scenario-adjusted sums of per-request E2E after
  dialog-first aggregation (`worst` is back-to-back; `hidden` is ideal, not
  session wall clock).
- `break_even_*.csv`: dialog-level and separate aggregate-mean-curve crossings.
- `persistence_per_image.csv`: the immutable measured CSV copied byte-for-byte,
  or a lossless CSV materialization when the append-only JSONL fallback is the
  only completed source.
"""
    return text


def write_stage(
        stage: Path, run_dir: Path, config: dict[str, Any],
        persistence_csv: bytes, artifacts: dict[str, str]) -> None:
    shutil.copyfile(run_dir / "config.json", stage / "config.json")
    (stage / "persistence_per_image.csv").write_bytes(persistence_csv)
    for name, content in artifacts.items():
        (stage / name).write_text(content)
    actual = {path.name for path in stage.iterdir() if path.is_file()}
    if actual != set(RESULT_NAMES):
        raise AnalysisError(
            f"staged artifact set mismatch: missing={set(RESULT_NAMES)-actual}, "
            f"extra={actual-set(RESULT_NAMES)}")


def publish_run_outputs(run_dir: Path, stage: Path,
                        names: Iterable[str] = RUN_OUTPUT_NAMES) -> None:
    destinations = [run_dir / name for name in names]
    existing = [str(path) for path in destinations if path.exists()]
    if existing:
        raise AnalysisError(
            "refusing to overwrite run analysis artifacts: " + ", ".join(existing))
    # O_EXCL makes every destination a no-overwrite publication.  All source
    # bytes have already been validated and staged before this point.
    created: list[Path] = []
    try:
        for destination in destinations:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            fd = os.open(destination, flags, 0o644)
            # Track the destination as soon as O_EXCL creates it.  If copying
            # or fsync fails, the outer rollback must also remove this partial
            # file so a clean retry is not blocked by our own artifact.
            created.append(destination)
            try:
                with os.fdopen(fd, "wb") as output, (stage / destination.name).open("rb") as source:
                    shutil.copyfileobj(source, output)
                    output.flush()
                    os.fsync(output.fileno())
            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise
    except Exception:
        # Roll back only files created by this invocation.  Pre-existing files
        # can never enter ``created`` because O_EXCL rejects them.
        for destination in created:
            destination.unlink(missing_ok=True)
        raise


def rename_noreplace(source: Path, destination: Path) -> None:
    """Linux atomic directory publication that cannot replace any target."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise AnalysisError("atomic no-overwrite publication requires renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100,
                       os.fsencode(destination), 1)  # RENAME_NOREPLACE
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise AnalysisError(
                f"refusing to overwrite results directory: {destination}")
        raise OSError(error, os.strerror(error), destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze VisDial Turn-1 piggyback E2E TTFT results")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    results_dir = args.results_dir.resolve()
    required = {
        "config": run_dir / "config.json",
        "raw": run_dir / "raw.jsonl",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise AnalysisError("missing run inputs: " + ", ".join(missing))
    persistence_source, persistence_source_rows, persistence_format = \
        persistence_input(run_dir)
    required["persistence"] = persistence_source
    run_output_names = list(RUN_OUTPUT_NAMES)
    if persistence_format == "jsonl_fallback":
        run_output_names.append("persistence_per_image.csv")
    if results_dir.exists():
        raise AnalysisError(f"refusing to overwrite results directory: {results_dir}")
    results_dir.parent.mkdir(parents=True, exist_ok=True)
    for name in run_output_names:
        if (run_dir / name).exists():
            raise AnalysisError(f"refusing to overwrite run artifact: {run_dir / name}")

    config = read_json(required["config"])
    correctness_path, _ = correctness_validation_source(config)
    if correctness_path is None or not correctness_path.is_file():
        raise AnalysisError(
            "missing captured-KV correctness validation referenced by config: "
            f"{correctness_path}")
    required["correctness_validation"] = correctness_path
    persistence_jsonl_path = run_dir / "persistence.jsonl"
    if persistence_jsonl_path.is_file() and persistence_jsonl_path != persistence_source:
        required["persistence_jsonl"] = persistence_jsonl_path
    before_hashes = {name: sha256_file(path) for name, path in required.items()}
    raw_rows = read_jsonl(required["raw"])
    rows = canonicalize_raw(raw_rows)
    persistence = canonicalize_persistence(persistence_source_rows)
    persistence_crosscheck = None
    if "persistence_jsonl" in required:
        jsonl_rows = read_jsonl(required["persistence_jsonl"])
        for row in jsonl_rows:
            row.pop("_line_number", None)
        persistence_from_jsonl = canonicalize_persistence(jsonl_rows)
        persistence_crosscheck = (
            persistence_projection(persistence)
            == persistence_projection(persistence_from_jsonl)
        )
    validation = validate_inputs(rows, persistence, config, before_hashes)
    validation["checks"]["persistence_csv_jsonl_crosscheck"] = (
        persistence_crosscheck is not False)
    validation["details"]["persistence_csv_jsonl_crosscheck"] = {
        "jsonl_present": "persistence_jsonl" in required,
        "equal": persistence_crosscheck,
    }
    if persistence_crosscheck is False:
        validation["passed"] = False
        validation["failures"].append("persistence_csv_jsonl_crosscheck")
    if not validation["passed"]:
        raise AnalysisError("input validation failed:\n" +
                            json.dumps(validation, indent=2, default=str))

    persistence_by_dialog = {row["dialog_id"]: row for row in persistence}
    summary_rows = (
        aggregate_rows(rows, "overall_all_turns")
        + aggregate_rows(rows, "reuse_only_turns_2_10")
    )
    per_turn = build_per_turn(rows)
    per_dialog = build_per_dialog(rows, persistence_by_dialog)
    ttft_dialog = per_dialog_cumulative(
        rows, persistence_by_dialog, "end_to_end_ttft_ms")
    e2e_dialog = per_dialog_cumulative(
        rows, persistence_by_dialog, "request_e2e_ms")
    cumulative_ttft = aggregate_cumulative(
        ttft_dialog, "sum_of_per_request_end_to_end_ttft_ms")
    cumulative_e2e = aggregate_cumulative(
        e2e_dialog, "sum_of_per_request_e2e_ms_with_persistence_scenario")
    break_even_ttft = build_break_even(
        ttft_dialog, cumulative_ttft, "sum_of_per_request_end_to_end_ttft_ms")
    break_even_e2e = build_break_even(
        e2e_dialog, cumulative_e2e,
        "sum_of_per_request_e2e_ms_with_persistence_scenario")
    quality_by_turn = build_quality_by_turn(rows)
    add_experiment_level_fields(
        summary_rows, persistence, cumulative_ttft, cumulative_e2e,
        break_even_ttft, break_even_e2e)
    derived_validation = validate_derived(
        rows, persistence_by_dialog, ttft_dialog, e2e_dialog,
        cumulative_ttft, cumulative_e2e,
        break_even_ttft, break_even_e2e)
    validation["checks"].update(derived_validation["checks"])
    validation["details"]["derived"] = derived_validation["details"]
    validation["failures"].extend(derived_validation["failures"])
    validation["passed"] = not validation["failures"] and all(
        validation["checks"].values())
    if not validation["passed"]:
        raise AnalysisError("derived validation failed:\n" +
                            json.dumps(validation, indent=2, default=str))

    verdict, verdict_details = claim_verdict(summary_rows, break_even_ttft)
    validation["claim_verdict"] = verdict
    validation["claim_verdict_details"] = verdict_details
    validation["main_metric"] = "end_to_end_ttft_ms"
    validation["cumulative_ttft_semantics"] = "sum_of_per_request_ttft_per_dialog_first"
    validation["cumulative_e2e_semantics"] = {
        "aggregation": "sum_of_per_request_e2e_per_dialog_first",
        "worst": "back_to_back completion with persistence charged once",
        "hidden": "ideal service-latency sum with persistence hidden; not wall clock",
    }
    validation["persistence_semantics"] = {
        "turn_1": "uncharged for both worst and hidden",
        "turn_2_to_10_worst": "persist_ms charged exactly once",
        "turn_2_to_10_hidden": "ideal hidden-persistence reference; uncharged",
        "shared_across_budgets": True,
    }
    validation["persistence_input_format"] = persistence_format
    validation["analyzer"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256_file(Path(__file__).resolve()),
    }

    fig_ttft = [dict(row) for row in cumulative_ttft
                if row["method"] in {"ReComp", "Prefix25", "Prefix45"}]
    fig_e2e = [dict(row) for row in cumulative_e2e
               if row["method"] in {"ReComp", "Prefix25", "Prefix45"}]
    fig_turn = [{
        "turn_id": row["turn_id"], "method": row["method"],
        "budget": row["budget"],
        "end_to_end_ttft_ms_mean": row["end_to_end_ttft_ms_mean"],
        "end_to_end_ttft_ms_p50": row["end_to_end_ttft_ms_p50"],
        "end_to_end_ttft_ms_p95": row["end_to_end_ttft_ms_p95"],
        "delta_vs_recomp_ms": row["end_to_end_ttft_delta_vs_recomp_ms"],
    } for row in per_turn if row["method"] in {"ReComp", "Prefix25", "Prefix45"}]
    fig_break_even = build_break_even_distribution(break_even_ttft)

    artifacts: dict[str, str] = {
        "summary.csv": csv_text(summary_rows),
        "per_turn.csv": csv_text(per_turn),
        "per_dialog.csv": csv_text(per_dialog),
        "cumulative_ttft.csv": csv_text(cumulative_ttft),
        "cumulative_e2e.csv": csv_text(cumulative_e2e),
        "break_even_ttft.csv": csv_text(break_even_ttft),
        "break_even_e2e.csv": csv_text(break_even_e2e),
        "quality_by_turn.csv": csv_text(quality_by_turn),
        "fig_cumulative_ttft.csv": csv_text(fig_ttft),
        "fig_cumulative_e2e.csv": csv_text(fig_e2e),
        "fig_per_turn_ttft.csv": csv_text(fig_turn),
        "fig_break_even_distribution.csv": csv_text(fig_break_even),
    }
    artifacts["README.md"] = build_readme(
        config, summary_rows, persistence, per_turn, cumulative_ttft, cumulative_e2e,
        break_even_ttft, break_even_e2e, verdict, verdict_details, validation)
    persistence_csv = (
        required["persistence"].read_bytes()
        if persistence_format == "csv"
        else csv_text(persistence_source_rows).encode("utf-8")
    )

    stage = Path(tempfile.mkdtemp(
        prefix=f".{results_dir.name}.stage.", dir=results_dir.parent))
    try:
        # Stage first, then verify both the sources and staged copies.  The
        # source config cannot be swapped between a pre-copy hash and copying
        # without that race being detected here.
        artifacts["validation.json"] = json.dumps(
            validation, indent=2, sort_keys=True, default=str) + "\n"
        write_stage(stage, run_dir, config, persistence_csv, artifacts)
        after_hashes = {name: sha256_file(path) for name, path in required.items()}
        unchanged = before_hashes == after_hashes
        staged_config_matches = (
            sha256_file(stage / "config.json") == before_hashes["config"])
        if persistence_format == "csv":
            staged_persistence_matches = (
                sha256_file(stage / "persistence_per_image.csv")
                == before_hashes["persistence"])
        else:
            staged_rows = canonicalize_persistence(
                read_csv(stage / "persistence_per_image.csv"))
            staged_persistence_matches = (
                persistence_projection(staged_rows)
                == persistence_projection(persistence)
            )
        validation["checks"]["all_source_inputs_unchanged"] = unchanged
        validation["checks"]["staged_config_matches_validated_source"] = \
            staged_config_matches
        validation["checks"]["staged_persistence_matches_validated_source"] = \
            staged_persistence_matches
        validation["details"]["input_sha256_after_staging"] = after_hashes
        validation["details"]["staged_config_sha256"] = sha256_file(
            stage / "config.json")
        if not (unchanged and staged_config_matches and staged_persistence_matches):
            validation["passed"] = False
            for name, passed in (
                ("all_source_inputs_unchanged", unchanged),
                ("staged_config_matches_validated_source", staged_config_matches),
                ("staged_persistence_matches_validated_source",
                 staged_persistence_matches),
            ):
                if not passed:
                    validation["failures"].append(name)
            raise AnalysisError("source or staged evidence changed during analysis")
        (stage / "validation.json").write_text(json.dumps(
            validation, indent=2, sort_keys=True, default=str) + "\n")
        # Check destinations before either publication.  The result directory
        # itself is then committed atomically and never replaces an old result.
        conflicts = [str(run_dir / name) for name in run_output_names
                     if (run_dir / name).exists()]
        if conflicts:
            raise AnalysisError("run artifacts appeared during analysis: "
                                + ", ".join(conflicts))
        rename_noreplace(stage, results_dir)
        stage = None  # type: ignore[assignment]
        try:
            publish_run_outputs(run_dir, results_dir, run_output_names)
        except Exception:
            # Results were already atomically committed and remain an auditable
            # publication.  Surface the incomplete run mirroring loudly.
            raise AnalysisError(
                f"results published at {results_dir}, but run mirroring failed")
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)

    print(json.dumps({
        "status": "PASS",
        "results_dir": str(results_dir),
        "run_dir": str(run_dir),
        "records": len(rows),
        "methods": sorted({row["method"] for row in rows}, key=method_sort_key),
        "claim_verdict": verdict,
        "input_sha256": before_hashes,
    }, indent=2))


if __name__ == "__main__":
    main()

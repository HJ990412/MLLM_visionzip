"""Paper-facing cache-hit analysis for the completed VisDial experiment.

This script is intentionally analysis-only.  It treats the completed run as
immutable evidence, recomputes every reported statistic from ``raw.jsonl``,
and publishes a new result directory without touching the source run or its
existing analysis.  The main population is strictly Turn 2--10: Turn 1 and
one-time persistence are reported separately.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN = ROOT / "runs/visdial_turn1_piggyback_e2e_ttft/main_seed1234"
DEFAULT_SOURCE_RESULTS = (
    ROOT / "results/visdial_turn1_piggyback_e2e_ttft/main_seed1234"
)
DEFAULT_RESULTS = ROOT / "results/visdial_cache_hit_analysis"

SCHEMA_VERSION = "visdial-cache-hit-analysis-v1"
SOURCE_SCHEMA_VERSION = "visdial-turn1-piggyback-e2e-ttft-v1"
EXPECTED_INDEX_SHA256 = (
    "8c3dd7e983cb39e61d26362a0353b86ac84845bd7537a6331078ab7707777383"
)
EXPECTED_REQUEST_KEYS_SHA256 = (
    "395ba928a15eb45bca905aa91d1ec89617981f18b9b22338341b2a189b3f141a"
)
METHODS = ("ReComp", "FullLoad", "Prefix25", "Prefix45")
METHOD_KEYS = {
    "recompute": "ReComp",
    "fullload": "FullLoad",
    "prefix25": "Prefix25",
    "prefix45": "Prefix45",
}
METHOD_BUDGETS = {
    "ReComp": None,
    "FullLoad": None,
    "Prefix25": 0.25,
    "Prefix45": 0.45,
}
CACHE_METHODS = {"FullLoad", "Prefix25", "Prefix45"}
PREFIX_METHODS = {"Prefix25", "Prefix45"}
MAIN_TURNS = tuple(range(2, 11))
MB = 1_000_000.0
TOLERANCE_MS = 1.0

RESULT_NAMES = {
    "config.json",
    "main_cache_hit_table.csv",
    "turn1_sanity.csv",
    "persistence_overhead.csv",
    "ttft_by_turn_cache_hit.csv",
    "quality_by_turn_cache_hit.csv",
    "io_breakdown_cache_hit.csv",
    "fig_cache_hit_ttft_by_turn.csv",
    "summary.json",
    "validation.json",
    "README.md",
}


class AnalysisError(RuntimeError):
    """Raised when immutable evidence or a derived artifact fails validation."""


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    """Hash relative names and contents of all regular files below ``path``."""
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
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
                    f"invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise AnalysisError(f"non-object row at {path}:{line_number}")
            row["_line_number"] = line_number
            rows.append(row)
    if not rows:
        raise AnalysisError(f"empty JSONL input: {path}")
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise AnalysisError(f"empty CSV input: {path}")
    return rows


def as_float(value: Any, context: str, *, default: float | None = None) -> float:
    if value in (None, ""):
        if default is not None:
            return float(default)
        raise AnalysisError(f"missing numeric field: {context}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"invalid numeric field {context}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise AnalysisError(f"non-finite numeric field {context}: {value!r}")
    return parsed


def as_int(value: Any, context: str, *, default: int | None = None) -> int:
    if value in (None, "") and default is not None:
        return int(default)
    parsed = as_float(value, context)
    if not parsed.is_integer():
        raise AnalysisError(f"non-integer field {context}: {value!r}")
    return int(parsed)


def as_bool(value: Any, context: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (1, "1", "true", "True", "yes", "YES"):
        return True
    if value in (0, "0", "false", "False", "no", "NO"):
        return False
    raise AnalysisError(f"invalid boolean field {context}: {value!r}")


def percentile(values: Iterable[float], percent: float) -> float | None:
    """NumPy-compatible linear percentile without an analysis dependency."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * float(percent) / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def metric_stats(rows: Iterable[dict[str, Any]], name: str,
                 *, null_as_zero: bool = False) -> dict[str, float | None]:
    values: list[float] = []
    for row in rows:
        value = row.get(name)
        if value is None:
            if null_as_zero:
                values.append(0.0)
            continue
        values.append(float(value))
    return {
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
    }


def normalize_answer(text: Any) -> str:
    import re
    normalized = re.sub(r"[^\w\s]", " ", str(text).lower())
    return " ".join(
        token for token in normalized.split() if token not in {"a", "an", "the"}
    )


def generative_match(prediction: Any, gold: Any) -> float:
    pred = normalize_answer(prediction)
    target = normalize_answer(gold)
    return float(
        pred == target
        or (target and pred.split()[:len(target.split())] == target.split())
    )


def canonical_method(row: dict[str, Any]) -> str:
    key = str(row.get("method_key", ""))
    if key not in METHOD_KEYS:
        raise AnalysisError(
            f"unknown method_key at raw line {row.get('_line_number')}: {key!r}"
        )
    return METHOD_KEYS[key]


def canonicalize_raw(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        line = raw["_line_number"]
        method = canonical_method(raw)
        turn = as_int(raw.get("turn_id"), f"raw line {line}/turn_id")
        if turn not in range(1, 11):
            raise AnalysisError(f"turn out of range at raw line {line}: {turn}")
        budget_raw = raw.get("budget")
        if method == "Prefix25" and abs(as_float(
                budget_raw, f"raw line {line}/budget") - 0.25) > 1e-12:
            raise AnalysisError(f"wrong Prefix25 budget at raw line {line}")
        if method == "Prefix45" and abs(as_float(
                budget_raw, f"raw line {line}/budget") - 0.45) > 1e-12:
            raise AnalysisError(f"wrong Prefix45 budget at raw line {line}")
        if method == "ReComp" and budget_raw not in (None, ""):
            raise AnalysisError(f"ReComp unexpectedly has a budget at line {line}")
        if method == "FullLoad" and abs(as_float(
                budget_raw, f"raw line {line}/budget") - 1.0) > 1e-12:
            raise AnalysisError(f"wrong FullLoad budget at raw line {line}")

        dialog_id = str(raw.get("dialog_id", ""))
        image_id = str(raw.get("image_id", ""))
        if not dialog_id or not image_id:
            raise AnalysisError(f"missing dialog/image identity at raw line {line}")
        row = {
            "line_number": line,
            "schema_version": str(raw.get("schema_version", "")),
            "dataset": str(raw.get("dataset", "")),
            "dialog_id": dialog_id,
            "image_id": image_id,
            "turn_id": turn,
            "method": method,
            "method_key": str(raw.get("method_key")),
            "budget": METHOD_BUDGETS[method],
            "history_policy": str(raw.get("history_policy", "")),
            "history_tokens": as_int(
                raw.get("history_tokens"), f"raw line {line}/history_tokens"
            ),
            "question": str(raw.get("question", "")),
            "gold": str(raw.get("gold", "")),
            "prediction": str(raw.get("prediction", "")),
            "quality_score": as_float(
                raw.get("quality_score"), f"raw line {line}/quality_score"
            ),
            "quality_metric": str(raw.get("quality_metric", "")),
            "prompt_sha256": str(raw.get("prompt_sha256", "")),
            "history_sha256": str(raw.get("text_history_sha256", "")),
            "suffix_ids_sha256": str(raw.get("suffix_ids_sha256", "")),
            "input_ids_sha256": str(raw.get("input_ids_sha256", "")),
            "pixel_values_sha256": str(raw.get("pixel_values_sha256", "")),
            "first_token_id": as_int(
                raw.get("first_token_id"), f"raw line {line}/first_token_id"
            ),
            "execution_mode": str(raw.get("execution_mode", "")),
            "turn1_normal_inference": as_bool(
                raw.get("turn1_normal_inference"),
                f"raw line {line}/turn1_normal_inference",
            ),
            "store_available_at_request_start": as_bool(
                raw.get("store_available_at_request_start"),
                f"raw line {line}/store_available_at_request_start",
            ),
            "vision_forward_count": as_int(
                raw.get("vision_forward_count"),
                f"raw line {line}/vision_forward_count",
            ),
            "separate_vision_forward_count": as_int(
                raw.get("separate_vision_forward_count", 0),
                f"raw line {line}/separate_vision_forward_count",
                default=0,
            ),
            "prompt_build_ms": as_float(
                raw.get("prompt_build_ms"), f"raw line {line}/prompt_build_ms"
            ),
            "tokenization_ms": as_float(
                raw.get("tokenization_ms"), f"raw line {line}/tokenization_ms"
            ),
            "image_preprocess_ms": as_float(
                raw.get("image_preprocess_ms"),
                f"raw line {line}/image_preprocess_ms",
            ),
            "input_prepare_ms": as_float(
                raw.get("input_prepare_ms"), f"raw line {line}/input_prepare_ms"
            ),
            "input_h2d_ms": as_float(
                raw.get("input_h2d_ms"), f"raw line {line}/input_h2d_ms"
            ),
            "vision_ms": as_float(
                raw.get("vision_ms"), f"raw line {line}/vision_ms"
            ),
            "core_ttft_ms": as_float(
                raw.get("core_ttft_ms"), f"raw line {line}/core_ttft_ms"
            ),
            "end_to_end_ttft_ms": as_float(
                raw.get("end_to_end_ttft_ms"),
                f"raw line {line}/end_to_end_ttft_ms",
            ),
            "request_e2e_ms": as_float(
                raw.get("request_e2e_ms"), f"raw line {line}/request_e2e_ms"
            ),
            "decode_ms": as_float(
                raw.get("decode_ms"), f"raw line {line}/decode_ms"
            ),
            "ssd_read_ms": as_float(
                raw.get("ssd_read_ms", 0), f"raw line {line}/ssd_read_ms",
                default=0,
            ),
            "ssd_read_bytes": as_int(
                raw.get("ssd_read_bytes", 0), f"raw line {line}/ssd_read_bytes",
                default=0,
            ),
            "total_actual_pread_bytes": as_int(
                raw.get("total_actual_pread_bytes", 0),
                f"raw line {line}/total_actual_pread_bytes",
                default=0,
            ),
            "ssd_preads": as_int(
                raw.get("ssd_read_preads", 0),
                f"raw line {line}/ssd_read_preads",
                default=0,
            ),
            "scatter_ms": (
                None if raw.get("scatter_ms") is None else as_float(
                    raw.get("scatter_ms"), f"raw line {line}/scatter_ms"
                )
            ),
            "prefill_ms": as_float(
                raw.get("prefill_ms"), f"raw line {line}/prefill_ms"
            ),
            "selector_ms": as_float(
                raw.get("selector_ms", 0), f"raw line {line}/selector_ms",
                default=0,
            ),
            "hook_total_ms": (
                None if raw.get("hook_total_ms") is None else as_float(
                    raw.get("hook_total_ms"),
                    f"raw line {line}/hook_total_ms",
                )
            ),
            "selected_kv_ratio": (
                None if raw.get("selected_kv_ratio") is None else as_float(
                    raw.get("selected_kv_ratio"),
                    f"raw line {line}/selected_kv_ratio",
                )
            ),
            "full_visual_kv_bytes": as_int(
                raw.get("full_visual_kv_bytes"),
                f"raw line {line}/full_visual_kv_bytes",
            ),
            "static_score_calls": as_int(
                raw.get("static_score_calls", 0),
                f"raw line {line}/static_score_calls", default=0,
            ),
            "query_score_calls": as_int(
                raw.get("query_score_calls", 0),
                f"raw line {line}/query_score_calls", default=0,
            ),
            "diversity_calls": as_int(
                raw.get("diversity_calls", 0),
                f"raw line {line}/diversity_calls", default=0,
            ),
            "selection_mode": raw.get("selection_mode"),
            "request_started_at_s": as_float(
                raw.get("request_started_at_s"),
                f"raw line {line}/request_started_at_s",
            ),
            "first_token_at_s": as_float(
                raw.get("first_token_at_s"), f"raw line {line}/first_token_at_s"
            ),
            "request_finished_at_s": as_float(
                raw.get("request_finished_at_s"),
                f"raw line {line}/request_finished_at_s",
            ),
            "cache_started_at_s": (
                None if raw.get("page_cache_conditioning_started_at_s") is None
                else as_float(
                    raw.get("page_cache_conditioning_started_at_s"),
                    f"raw line {line}/cache_started",
                )
            ),
            "cache_finished_at_s": (
                None if raw.get("page_cache_conditioning_finished_at_s") is None
                else as_float(
                    raw.get("page_cache_conditioning_finished_at_s"),
                    f"raw line {line}/cache_finished",
                )
            ),
            "cache_conditioning_method": str(
                raw.get("page_cache_conditioning_method", "")
            ),
            "cache_conditioning_excluded": as_bool(
                raw.get("page_cache_conditioning_excluded_from_ttft"),
                f"raw line {line}/cache_conditioning_excluded",
            ),
            "raw": raw,
        }
        rows.append(row)
    return rows


def method_rows(rows: Iterable[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["method"] == method]


def mean(rows: Iterable[dict[str, Any]], name: str) -> float:
    values = [float(row[name]) for row in rows]
    if not values:
        raise AnalysisError(f"cannot average empty metric {name}")
    return sum(values) / len(values)


def build_main_table(cache_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method = {method: method_rows(cache_rows, method) for method in METHODS}
    recomp_ttft = mean(by_method["ReComp"], "end_to_end_ttft_ms")
    recomp_e2e = mean(by_method["ReComp"], "request_e2e_ms")
    recomp_quality = mean(by_method["ReComp"], "quality_score")
    full_ssd = mean(by_method["FullLoad"], "ssd_read_bytes")
    output: list[dict[str, Any]] = []
    for method in METHODS:
        selected = by_method[method]
        ttft = metric_stats(selected, "end_to_end_ttft_ms")
        e2e = metric_stats(selected, "request_e2e_ms")
        quality = mean(selected, "quality_score")
        ssd_mean = mean(selected, "ssd_read_bytes")
        ratio = None if method == "ReComp" else 100.0 * ssd_mean / full_ssd
        output.append({
            "method": method,
            "budget_fraction": METHOD_BUDGETS[method],
            "turn_range": "2-10",
            "n_dialogs": len({row["dialog_id"] for row in selected}),
            "n_requests": len(selected),
            "quality_metric": "normalized_generative_match_auxiliary_not_official_visdial",
            "aux_quality_mean": quality,
            "aux_quality_delta_vs_recomp": quality - recomp_quality,
            "end_to_end_ttft_ms_mean": ttft["mean"],
            "end_to_end_ttft_ms_p50": ttft["p50"],
            "end_to_end_ttft_ms_p95": ttft["p95"],
            "ttft_delta_vs_recomp_ms": ttft["mean"] - recomp_ttft,
            "ttft_reduction_vs_recomp_pct": (
                100.0 * (1.0 - ttft["mean"] / recomp_ttft)
            ),
            "request_e2e_ms_mean": e2e["mean"],
            "request_e2e_ms_p50": e2e["p50"],
            "request_e2e_ms_p95": e2e["p95"],
            "e2e_delta_vs_recomp_ms": e2e["mean"] - recomp_e2e,
            "e2e_reduction_vs_recomp_pct": (
                100.0 * (1.0 - e2e["mean"] / recomp_e2e)
            ),
            "ssd_read_bytes_per_request": ssd_mean,
            "ssd_read_mb_per_request": ssd_mean / MB,
            "ssd_read_bytes_total": sum(row["ssd_read_bytes"] for row in selected),
            "ssd_ratio_vs_fullload_pct": ratio,
            "ssd_reduction_vs_fullload_pct": (
                None if ratio is None else 100.0 - ratio
            ),
        })
    return output


def build_turn1_sanity(turn1_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in turn1_rows:
        groups[row["dialog_id"]][row["method"]] = row
    output: list[dict[str, Any]] = []
    for method in METHODS:
        selected = method_rows(turn1_rows, method)
        prediction_matches = 0
        token_matches = 0
        prompt_matches = 0
        input_matches = 0
        pixel_matches = 0
        for dialog, records in groups.items():
            reference = records["ReComp"]
            row = records[method]
            prediction_matches += row["prediction"] == reference["prediction"]
            token_matches += row["first_token_id"] == reference["first_token_id"]
            prompt_matches += row["prompt_sha256"] == reference["prompt_sha256"]
            input_matches += row["input_ids_sha256"] == reference["input_ids_sha256"]
            pixel_matches += (
                row["pixel_values_sha256"] == reference["pixel_values_sha256"]
            )
        ttft = metric_stats(selected, "end_to_end_ttft_ms")
        e2e = metric_stats(selected, "request_e2e_ms")
        count = len(selected)
        output.append({
            "method": method,
            "budget_fraction": METHOD_BUDGETS[method],
            "n_dialogs": count,
            "end_to_end_ttft_ms_mean": ttft["mean"],
            "end_to_end_ttft_ms_p50": ttft["p50"],
            "end_to_end_ttft_ms_p95": ttft["p95"],
            "request_e2e_ms_mean": e2e["mean"],
            "vision_ms_mean": mean(selected, "vision_ms"),
            "vision_forward_count_mean": mean(selected, "vision_forward_count"),
            "separate_vision_forward_count_total": sum(
                row["separate_vision_forward_count"] for row in selected
            ),
            "ssd_read_bytes_total": sum(row["ssd_read_bytes"] for row in selected),
            "prediction_agreement_vs_recomp_pct": 100.0 * prediction_matches / count,
            "first_token_agreement_vs_recomp_pct": 100.0 * token_matches / count,
            "prompt_agreement_vs_recomp_pct": 100.0 * prompt_matches / count,
            "tokenized_input_agreement_vs_recomp_pct": 100.0 * input_matches / count,
            "pixel_tensor_agreement_vs_recomp_pct": 100.0 * pixel_matches / count,
            "saliency_extra_ms_mean": mean(selected, "raw_saliency_extra_ms"),
        })
    return output


def build_persistence_overhead(
        persistence_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    definitions = (
        ("saliency_extra", "saliency_extra_ms", "inside_turn1_ttft_not_readded"),
        ("saliency_postprocess", "saliency_postprocess_ms", "post_answer_persistence"),
        ("permutation", "permutation_ms", "post_answer_persistence"),
        ("kv_repack", "repack_ms", "post_answer_persistence"),
        ("buffered_ssd_write", "buffered_write_ms", "post_answer_persistence"),
        ("fsync", "fsync_ms", "post_answer_persistence"),
        ("total_persistence", "persist_ms", "one_time_total_outside_cache_hit_ttft"),
    )
    output = []
    for component, field, scope in definitions:
        values = [as_float(row.get(field), f"persistence/{field}")
                  for row in persistence_rows]
        output.append({
            "component": component,
            "source_field": field,
            "n_images": len(values),
            "mean_ms": sum(values) / len(values),
            "p50_ms": percentile(values, 50),
            "p95_ms": percentile(values, 95),
            "accounting_scope": scope,
        })
    return output


def build_turn_tables(cache_rows: list[dict[str, Any]]) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    long_rows: list[dict[str, Any]] = []
    quality_wide: list[dict[str, Any]] = []
    figure_wide: list[dict[str, Any]] = []
    for turn in MAIN_TURNS:
        turn_rows = [row for row in cache_rows if row["turn_id"] == turn]
        recomp_ttft = mean(method_rows(turn_rows, "ReComp"),
                           "end_to_end_ttft_ms")
        history = mean(method_rows(turn_rows, "ReComp"), "history_tokens")
        quality_row: dict[str, Any] = {
            "turn_id": turn,
            "n_dialogs": 100,
            "history_tokens_mean": history,
            "quality_metric": "normalized_generative_match_auxiliary_not_official_visdial",
        }
        figure_row: dict[str, Any] = {
            "turn_id": turn,
            "history_tokens_mean": history,
        }
        for method in METHODS:
            selected = method_rows(turn_rows, method)
            ttft = metric_stats(selected, "end_to_end_ttft_ms")
            e2e = metric_stats(selected, "request_e2e_ms")
            long_rows.append({
                "turn_id": turn,
                "history_tokens_mean": mean(selected, "history_tokens"),
                "method": method,
                "budget_fraction": METHOD_BUDGETS[method],
                "n_dialogs": len(selected),
                "end_to_end_ttft_ms_mean": ttft["mean"],
                "end_to_end_ttft_ms_p50": ttft["p50"],
                "end_to_end_ttft_ms_p95": ttft["p95"],
                "ttft_delta_vs_recomp_ms": ttft["mean"] - recomp_ttft,
                "ttft_reduction_vs_recomp_pct": (
                    100.0 * (1.0 - ttft["mean"] / recomp_ttft)
                ),
                "request_e2e_ms_mean": e2e["mean"],
                "quality_mean": mean(selected, "quality_score"),
                "ssd_read_mb_per_request": mean(selected, "ssd_read_bytes") / MB,
            })
            quality_row[method] = mean(selected, "quality_score")
            figure_row[method] = ttft["mean"]
        quality_row["Prefix25_delta_vs_ReComp"] = (
            quality_row["Prefix25"] - quality_row["ReComp"]
        )
        quality_row["Prefix45_delta_vs_ReComp"] = (
            quality_row["Prefix45"] - quality_row["ReComp"]
        )
        figure_row["ReComp_minus_Prefix25_ms"] = (
            figure_row["ReComp"] - figure_row["Prefix25"]
        )
        figure_row["ReComp_minus_Prefix45_ms"] = (
            figure_row["ReComp"] - figure_row["Prefix45"]
        )
        quality_wide.append(quality_row)
        figure_wide.append(figure_row)
    return long_rows, quality_wide, figure_wide


def build_io_breakdown(cache_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in METHODS:
        selected = method_rows(cache_rows, method)
        read = metric_stats(selected, "ssd_read_ms")
        preads = metric_stats(selected, "ssd_preads")
        scatter = metric_stats(selected, "scatter_ms")
        prefill = metric_stats(selected, "prefill_ms")
        planning = metric_stats(selected, "selector_ms")
        hook = metric_stats(selected, "hook_total_ms")
        ratios = [row["selected_kv_ratio"] for row in selected
                  if row["selected_kv_ratio"] is not None]
        full_forward_after_hook = None
        if method == "FullLoad":
            residuals = [
                row["prefill_ms"] - row["hook_total_ms"]
                for row in selected if row["hook_total_ms"] is not None
            ]
            full_forward_after_hook = sum(residuals) / len(residuals)
        prefill_semantics = {
            "ReComp": "inclusive multimodal-forward interval containing vision",
            "FullLoad": "inclusive interval containing hook SSD read/cache write; not phase-comparable",
            "Prefix25": "forward interval after SSD read and scatter",
            "Prefix45": "forward interval after SSD read and scatter",
        }[method]
        scatter_semantics = {
            "ReComp": "not applicable (reported raw zero)",
            "FullLoad": "not separately measured; cache write occurs inside hook",
            "Prefix25": "separately measured GPU cache reconstruction/scatter",
            "Prefix45": "separately measured GPU cache reconstruction/scatter",
        }[method]
        output.append({
            "method": method,
            "budget_fraction": METHOD_BUDGETS[method],
            "n_requests": len(selected),
            "ssd_read_mb_per_request": mean(selected, "ssd_read_bytes") / MB,
            "ssd_read_ms_mean": read["mean"],
            "ssd_read_ms_p50": read["p50"],
            "ssd_read_ms_p95": read["p95"],
            "preads_per_request_mean": preads["mean"],
            "preads_per_request_p50": preads["p50"],
            "preads_per_request_p95": preads["p95"],
            "scatter_ms_mean": scatter["mean"],
            "scatter_ms_p50": scatter["p50"],
            "scatter_ms_p95": scatter["p95"],
            "prefill_ms_mean": prefill["mean"],
            "prefill_ms_p50": prefill["p50"],
            "prefill_ms_p95": prefill["p95"],
            "prefill_semantics": prefill_semantics,
            "hook_total_ms_mean": hook["mean"],
            "fullload_prefill_minus_hook_ms_mean_diagnostic":
                full_forward_after_hook,
            "scatter_semantics": scatter_semantics,
            "first_k_planning_ms_mean": planning["mean"],
            "first_k_planning_ms_p50": planning["p50"],
            "first_k_planning_ms_p95": planning["p95"],
            "selected_visual_kv_ratio_mean": (
                sum(ratios) / len(ratios) if ratios else None
            ),
            "static_score_calls_total": sum(
                row["static_score_calls"] for row in selected
            ),
            "query_score_calls_total": sum(
                row["query_score_calls"] for row in selected
            ),
            "diversity_calls_total": sum(
                row["diversity_calls"] for row in selected
            ),
            "planning_semantics": (
                "sequential first-k index planning; no online scorer"
                if method in PREFIX_METHODS else "not applicable"
            ),
        })
    return output


def build_amortization(
        rows: list[dict[str, Any]], persistence_rows: list[dict[str, str]]) \
        -> dict[str, Any]:
    persist = {
        row["dialog_id"]: as_float(row["persist_ms"], "persistence/persist_ms")
        for row in persistence_rows
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dialog_id"], row["method"])].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda row: row["turn_id"])
    dialogs = sorted({row["dialog_id"] for row in rows})
    output: dict[str, Any] = {
        "scope": "secondary persistence overhead / amortization analysis",
        "worst_semantics": "one measured persist_ms charged once for N>=2",
        "hidden_semantics": "ideal reference with persistence excluded",
        "metrics": {},
    }
    for metric_name, field in (
            ("end_to_end_ttft_ms", "end_to_end_ttft_ms"),
            ("request_e2e_ms", "request_e2e_ms")):
        metric: dict[str, Any] = {}
        recomp_curve = {}
        for n in range(1, 11):
            recomp_curve[n] = sum(
                sum(row[field] for row in grouped[(dialog, "ReComp")][:n])
                for dialog in dialogs
            ) / len(dialogs)
        for method in ("Prefix25", "Prefix45"):
            curves = {"hidden": {}, "worst": {}}
            for n in range(1, 11):
                hidden_values = [
                    sum(row[field] for row in grouped[(dialog, method)][:n])
                    for dialog in dialogs
                ]
                worst_values = [
                    value + (persist[dialog] if n >= 2 else 0.0)
                    for value, dialog in zip(hidden_values, dialogs)
                ]
                curves["hidden"][n] = sum(hidden_values) / len(hidden_values)
                curves["worst"][n] = sum(worst_values) / len(worst_values)
            method_result: dict[str, Any] = {"curves": curves}
            for scenario in ("hidden", "worst"):
                crossing = next((
                    n for n in range(2, 11)
                    if curves[scenario][n] < recomp_curve[n]
                ), None)
                method_result[scenario] = {
                    "strict_break_even_turn": crossing,
                    "censored_at_turn_10": crossing is None,
                    "n2_ours_ms": curves[scenario][2],
                    "n2_recomp_ms": recomp_curve[2],
                    "n2_delta_ms": curves[scenario][2] - recomp_curve[2],
                    "n10_ours_ms": curves[scenario][10],
                    "n10_recomp_ms": recomp_curve[10],
                    "n10_delta_ms": curves[scenario][10] - recomp_curve[10],
                    "n10_reduction_pct": 100.0 * (
                        1.0 - curves[scenario][10] / recomp_curve[10]
                    ),
                }
            metric[method] = method_result
        metric["ReComp_curve"] = recomp_curve
        output["metrics"][metric_name] = metric
    return output


def csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise AnalysisError("cannot serialize an empty CSV")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                fieldnames.append(name)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


class Validator:
    def __init__(self) -> None:
        self.checks: dict[str, bool] = {}
        self.details: dict[str, Any] = {}
        self.failures: list[str] = []

    def check(self, name: str, condition: Any, detail: Any = None) -> None:
        passed = bool(condition)
        self.checks[name] = passed
        if detail is not None:
            self.details[name] = detail
        if not passed:
            self.failures.append(name)

    def result(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "passed": not self.failures and all(self.checks.values()),
            "checks": self.checks,
            "details": self.details,
            "failures": self.failures,
        }


def validate_inputs(
        rows: list[dict[str, Any]], persistence: list[dict[str, str]],
        config: dict[str, Any], source_validation: dict[str, Any],
        input_hashes: dict[str, str]) -> Validator:
    validator = Validator()
    cache_rows = [row for row in rows if row["turn_id"] in MAIN_TURNS]
    turn1_rows = [row for row in rows if row["turn_id"] == 1]
    method_counts = Counter(row["method"] for row in cache_rows)
    turn1_counts = Counter(row["method"] for row in turn1_rows)

    validator.check("source_validation_passed",
                    source_validation.get("passed") is True)
    validator.check("source_validation_checks_all_true",
                    bool(source_validation.get("checks")) and
                    all(source_validation["checks"].values()))
    source_input_hashes = source_validation.get("details", {}).get(
        "input_sha256", {})
    validator.check("raw_sha_matches_validated_source",
                    source_input_hashes.get("raw") == input_hashes["raw"])
    validator.check("config_sha_matches_validated_source",
                    source_input_hashes.get("config") == input_hashes["config"])
    validator.check("persistence_sha_matches_validated_source",
                    source_input_hashes.get("persistence") ==
                    input_hashes["persistence"])
    validator.check("source_schema_exact",
                    config.get("schema_version") == SOURCE_SCHEMA_VERSION and
                    all(row["schema_version"] == SOURCE_SCHEMA_VERSION
                        for row in rows))
    validator.check("completed_source_config", config.get("status") == "complete")
    validator.check("workload_hash_exact",
                    config.get("index_sha256") == EXPECTED_INDEX_SHA256 and
                    input_hashes["index"] == EXPECTED_INDEX_SHA256)
    validator.check("ordered_request_hash_exact",
                    config.get("expected_request_keys_sha256") ==
                    EXPECTED_REQUEST_KEYS_SHA256 and
                    stable_json_sha256(config.get("expected_request_keys")) ==
                    EXPECTED_REQUEST_KEYS_SHA256)
    validator.check("exact_workload_counts",
                    len(rows) == 4000 and len(cache_rows) == 3600 and
                    len(turn1_rows) == 400 and
                    len({row["dialog_id"] for row in rows}) == 100 and
                    len({row["image_id"] for row in rows}) == 100)
    validator.check("method_set_exact", set(method_counts) == set(METHODS))
    validator.check("cache_hit_900_rows_per_method",
                    method_counts == Counter({method: 900 for method in METHODS}),
                    dict(method_counts))
    validator.check("turn1_100_rows_per_method",
                    turn1_counts == Counter({method: 100 for method in METHODS}),
                    dict(turn1_counts))
    validator.check("main_turn_range_exact_2_10",
                    {row["turn_id"] for row in cache_rows} == set(MAIN_TURNS))
    validator.check("turn1_absent_from_main_population",
                    not any(row["turn_id"] == 1 for row in cache_rows))

    keys = Counter((row["dialog_id"], row["turn_id"], row["method"])
                   for row in rows)
    validator.check("complete_unique_request_matrix",
                    len(keys) == 4000 and all(value == 1 for value in keys.values()))
    observed_request_keys: list[list[Any]] = []
    seen_request_keys: set[tuple[str, int]] = set()
    for row in rows:
        key = (row["dialog_id"], row["turn_id"])
        if key not in seen_request_keys:
            seen_request_keys.add(key)
            observed_request_keys.append([key[0], key[1]])
    validator.check(
        "raw_ordered_request_keys_match_config",
        observed_request_keys == config.get("expected_request_keys") and
        stable_json_sha256(observed_request_keys) == EXPECTED_REQUEST_KEYS_SHA256,
    )

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dialog_id"], row["turn_id"])].append(row)
    prompt_equal = True
    for group in grouped.values():
        prompt_equal &= len({(
            row["question"], row["gold"], row["history_tokens"],
            row["history_sha256"], row["suffix_ids_sha256"],
            row["prompt_sha256"],
        ) for row in group}) == 1
    validator.check("prompt_gold_history_identical_across_methods", prompt_equal)
    validator.check("gold_teacher_forced_history",
                    all(row["history_policy"] == "gold_teacher_forced"
                        for row in rows))

    quality_residual = max(abs(
        row["quality_score"] - generative_match(row["prediction"], row["gold"])
    ) for row in rows)
    validator.check("aux_quality_recomputed_exactly", quality_residual <= 1e-12,
                    {"max_abs_residual": quality_residual})
    validator.check("quality_is_explicitly_not_official",
                    all("auxiliary" in row["quality_metric"] and
                        "not_official" in row["quality_metric"]
                        for row in rows))

    ttft_identity = max(abs(
        row["end_to_end_ttft_ms"] -
        (row["first_token_at_s"] - row["request_started_at_s"]) * 1000.0
    ) for row in rows)
    e2e_identity = max(abs(
        row["request_e2e_ms"] -
        (row["request_finished_at_s"] - row["request_started_at_s"]) * 1000.0
    ) for row in rows)
    validator.check("end_to_end_ttft_timestamp_identity",
                    ttft_identity <= TOLERANCE_MS,
                    {"max_abs_residual_ms": ttft_identity})
    validator.check("request_e2e_timestamp_identity",
                    e2e_identity <= TOLERANCE_MS,
                    {"max_abs_residual_ms": e2e_identity})
    validator.check("ttft_strictly_below_request_e2e",
                    all(row["end_to_end_ttft_ms"] < row["request_e2e_ms"]
                        for row in rows))
    validator.check("config_main_metric_is_end_to_end_ttft",
                    config.get("main_ttft_metric") == "end_to_end_ttft_ms" and
                    config.get("input_preprocessing_timed") is True)

    cached = [row for row in cache_rows if row["method"] in CACHE_METHODS]
    recomp = method_rows(cache_rows, "ReComp")
    validator.check("cache_hit_store_available_and_no_vision",
                    all(row["store_available_at_request_start"] and
                        row["vision_forward_count"] == 0 and
                        row["vision_ms"] == 0.0 and row["ssd_read_bytes"] > 0
                        for row in cached))
    validator.check("recomp_recomputes_vision_without_ssd",
                    all(row["vision_forward_count"] == 1 and
                        row["ssd_read_bytes"] == 0 and row["ssd_preads"] == 0
                        for row in recomp))
    validator.check("actual_pread_bytes_match",
                    all(row["ssd_read_bytes"] == row["total_actual_pread_bytes"]
                        for row in cache_rows))
    validator.check("cache_conditioning_before_timer",
                    all(row["cache_started_at_s"] is not None and
                        row["cache_finished_at_s"] is not None and
                        row["cache_started_at_s"] <= row["cache_finished_at_s"] <=
                        row["request_started_at_s"] and
                        row["cache_conditioning_excluded"] and
                        row["cache_conditioning_method"] ==
                        "posix_fadvise_DONTNEED" for row in cached))
    validator.check("buffered_pread_not_odirect",
                    config.get("ssd_read_api") == "buffered_pread" and
                    config.get("o_direct") is False and
                    config.get("ssd_controller_cache_flushed") is False)
    validator.check("no_online_scorers",
                    all(row["static_score_calls"] == 0 and
                        row["query_score_calls"] == 0 and
                        row["diversity_calls"] == 0 for row in cache_rows))
    validator.check("prefix_mode_is_sequential_first_k",
                    all(row["selection_mode"] == "prefix"
                        for row in cache_rows if row["method"] in PREFIX_METHODS))

    turn1_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in turn1_rows:
        turn1_grouped[row["dialog_id"]].append(row)
    turn1_equal = True
    turn1_outputs_equal = True
    for group in turn1_grouped.values():
        turn1_equal &= len({(
            row["prompt_sha256"], row["input_ids_sha256"],
            row["pixel_values_sha256"], row["question"], row["gold"],
        ) for row in group}) == 1
        turn1_outputs_equal &= len({(
            row["prediction"], row["first_token_id"]
        ) for row in group}) == 1
    validator.check("turn1_inputs_identical", turn1_equal)
    validator.check("turn1_input_hashes_nonempty",
                    all(row["prompt_sha256"] and row["input_ids_sha256"] and
                        row["pixel_values_sha256"] for row in turn1_rows))
    validator.check("turn1_predictions_and_first_tokens_identical",
                    turn1_outputs_equal)
    validator.check("turn1_normal_inference_no_ssd",
                    all(row["turn1_normal_inference"] and
                        not row["store_available_at_request_start"] and
                        row["vision_forward_count"] == 1 and
                        row["ssd_read_bytes"] == 0 for row in turn1_rows))

    validator.check("one_persistence_row_per_dialog_image",
                    len(persistence) == 100 and
                    len({row["dialog_id"] for row in persistence}) == 100 and
                    len({row["image_id"] for row in persistence}) == 100)
    validator.check("persistence_is_turn1_piggyback_without_extra_forward",
                    all(row.get("source_turn_id") == "1" and
                        row.get("capture_from_same_answer1_forward") == "True" and
                        row.get("separate_vision_forward_count") == "0" and
                        row.get("separate_prefix_forward_count") == "0"
                        for row in persistence))
    validator.check("persistence_has_no_calibration_or_future_leakage",
                    all(row.get("layout_uses_dataset_question") == "False" and
                        row.get("future_turns_used_for_layout") == "0" and
                        row.get("calibration_questions") == "0"
                        for row in persistence))
    return validator


def compare_to_prior_summary(
        main_table: list[dict[str, Any]], prior_summary: list[dict[str, str]]) \
        -> tuple[bool, dict[str, float]]:
    prior = {
        row["method"]: row for row in prior_summary
        if row.get("scope") == "reuse_only_turns_2_10"
    }
    maximum = 0.0
    for row in main_table:
        old = prior.get(row["method"])
        if old is None:
            return False, {"missing_method": row["method"]}  # type: ignore[dict-item]
        pairs = (
            ("aux_quality_mean", "quality_mean"),
            ("end_to_end_ttft_ms_mean", "end_to_end_ttft_ms_mean"),
            ("end_to_end_ttft_ms_p50", "end_to_end_ttft_ms_p50"),
            ("end_to_end_ttft_ms_p95", "end_to_end_ttft_ms_p95"),
            ("request_e2e_ms_mean", "request_e2e_ms_mean"),
            ("ssd_read_bytes_per_request", "ssd_read_bytes_mean"),
        )
        for new_name, old_name in pairs:
            residual = abs(float(row[new_name]) - float(old[old_name]))
            maximum = max(maximum, residual)
    return maximum <= 1e-9, {"maximum_abs_residual": maximum}


def main_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["method"]: row for row in rows}


def persistence_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["component"]: row for row in rows}


def make_summary(
        config: dict[str, Any], source_paths: dict[str, Path],
        input_hashes: dict[str, str], main_table: list[dict[str, Any]],
        turn1: list[dict[str, Any]], persistence: list[dict[str, Any]],
        figure: list[dict[str, Any]], io_rows: list[dict[str, Any]],
        persistence_source: list[dict[str, str]], amortization: dict[str, Any]) \
        -> dict[str, Any]:
    main = main_lookup(main_table)
    turns = {row["turn_id"]: row for row in figure}
    persist = persistence_lookup(persistence)
    total_write = sum(as_int(
        row["total_ssd_write_bytes"], "persistence/total_ssd_write_bytes"
    ) for row in persistence_source)
    paper_ready = (
        "On VisDial cache-hit turns 2--10, ImageOnly-Repack Prefix25 reduced "
        f"end-to-end TTFT by {main['Prefix25']['ttft_reduction_vs_recomp_pct']:.2f}% "
        "relative to recomputing the visual context, while reducing SSD bytes "
        f"read by {main['Prefix25']['ssd_reduction_vs_fullload_pct']:.2f}% "
        "relative to FullLoad, with an auxiliary-quality delta of "
        f"{main['Prefix25']['aux_quality_delta_vs_recomp']:.3f}. "
        "Prefix45 retained comparable normalized generative-match auxiliary "
        f"quality and reduced end-to-end TTFT by "
        f"{main['Prefix45']['ttft_reduction_vs_recomp_pct']:.2f}% versus "
        "ReComp. FullLoad was slower than ReComp, showing that SSD caching "
        "alone is insufficient when the entire Visual KV is read per request. "
        "The one-time cache-persistence cost is reported separately and is "
        "not included in these cache-hit request latencies."
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_scope": {
            "main": "Turn 2-10 cache-hit requests only",
            "turn1": "separate cache-miss / construction-opportunity sanity check",
            "persistence": "separate one-time overhead",
            "amortization": "secondary persistence-inclusive analysis",
        },
        "source": {
            name: {"path": str(path), "sha256": input_hashes.get(name)}
            for name, path in source_paths.items()
        },
        "workload": {
            "dataset": config["dataset"],
            "dialogs": 100,
            "images": 100,
            "turns_per_dialog": 10,
            "main_turns": list(MAIN_TURNS),
            "cache_hit_requests_per_method": 900,
            "index_sha256": config["index_sha256"],
            "ordered_request_keys_sha256": config["expected_request_keys_sha256"],
            "history_policy": config["history_policy"],
        },
        "timing": {
            "main_metric": "end_to_end_ttft_ms",
            "boundary": config["ttft_definition"],
            "core_ttft_role": "diagnostic only; excluded from the main table",
            "page_cache_conditioning": "posix_fadvise(DONTNEED) before timer",
            "io_api": "buffered pread",
            "o_direct": False,
            "ssd_controller_cache_flushed": False,
        },
        "main_cache_hit_results": main,
        "comparisons": {
            "Prefix25_vs_ReComp": {
                "ttft_reduction_pct": main["Prefix25"]["ttft_reduction_vs_recomp_pct"],
                "e2e_reduction_pct": main["Prefix25"]["e2e_reduction_vs_recomp_pct"],
                "quality_delta": main["Prefix25"]["aux_quality_delta_vs_recomp"],
            },
            "Prefix45_vs_ReComp": {
                "ttft_reduction_pct": main["Prefix45"]["ttft_reduction_vs_recomp_pct"],
                "e2e_reduction_pct": main["Prefix45"]["e2e_reduction_vs_recomp_pct"],
                "quality_delta": main["Prefix45"]["aux_quality_delta_vs_recomp"],
            },
            "Prefix25_vs_FullLoad": {
                "ttft_reduction_pct": 100.0 * (
                    1.0 - main["Prefix25"]["end_to_end_ttft_ms_mean"] /
                    main["FullLoad"]["end_to_end_ttft_ms_mean"]
                ),
                "ssd_read_reduction_pct": main["Prefix25"][
                    "ssd_reduction_vs_fullload_pct"],
            },
            "Prefix45_vs_FullLoad": {
                "ttft_reduction_pct": 100.0 * (
                    1.0 - main["Prefix45"]["end_to_end_ttft_ms_mean"] /
                    main["FullLoad"]["end_to_end_ttft_ms_mean"]
                ),
                "ssd_read_reduction_pct": main["Prefix45"][
                    "ssd_reduction_vs_fullload_pct"],
            },
        },
        "turn1_sanity": main_lookup(turn1),
        "persistence_overhead": {
            "components": persistence_lookup(persistence),
            "one_time_ssd_write_bytes_total": total_write,
            "one_time_ssd_write_mb_per_image_mean": total_write / 100 / MB,
            "semantics": (
                "executed synchronously immediately after the source Answer 1; "
                "reported outside subsequent cache-hit request timing"
            ),
        },
        "turn_wise_trend": {
            "history_tokens_mean_turn2": turns[2]["history_tokens_mean"],
            "history_tokens_mean_turn10": turns[10]["history_tokens_mean"],
            "recomp_minus_prefix25_ms_turn2": turns[2]["ReComp_minus_Prefix25_ms"],
            "recomp_minus_prefix25_ms_turn10": turns[10]["ReComp_minus_Prefix25_ms"],
            "recomp_minus_prefix45_ms_turn2": turns[2]["ReComp_minus_Prefix45_ms"],
            "recomp_minus_prefix45_ms_turn10": turns[10]["ReComp_minus_Prefix45_ms"],
        },
        "io_breakdown": main_lookup(io_rows),
        "secondary_amortization": amortization,
        "paper_ready_conclusion": paper_ready,
        "limitations": [
            "Quality is normalized generative-match auxiliary score, not official VisDial MRR/R@K/NDCG.",
            "Cold means OS page-cache conditioning; O_DIRECT and SSD-controller cache flush were not used.",
            "JPEG file read/decode was outside the request timer; resize/crop/tensorization was inside.",
            "Cache-hit TTFT excludes one-time persistence, which is reported separately rather than claimed cost-free.",
        ],
        "persistence_total_ms_mean": persist["total_persistence"]["mean_ms"],
    }


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def build_readme(
        config: dict[str, Any], main_table: list[dict[str, Any]],
        turn1: list[dict[str, Any]], persistence: list[dict[str, Any]],
        figure: list[dict[str, Any]], summary: dict[str, Any],
        validation: dict[str, Any]) -> str:
    main = main_lookup(main_table)
    amort = summary["secondary_amortization"]["metrics"]
    lines = [
        "# VisDial Cache-Hit Serving Analysis",
        "",
        f"Validation: **{'PASS' if validation['passed'] else 'FAIL'}**",
        "",
        "## Analysis scope",
        "",
        "The paper-facing main population is strictly VisDial Turns 2--10: "
        "100 dialogs x 9 cache-hit turns = 900 requests per method. Turn 1 is "
        "a cache-miss/construction-opportunity sanity check and is not mixed "
        "into the main average. One-time persistence is reported separately; "
        "it is neither added to cache-hit request TTFT nor described as free.",
        "",
        f"- Frozen index SHA256: `{config['index_sha256']}`",
        f"- Ordered request-key SHA256: `{config['expected_request_keys_sha256']}`",
        "- History policy: gold teacher-forced",
        "- Main metric: `end_to_end_ttft_ms`",
        "",
        "## Timing boundary",
        "",
        "OS page-cache conditioning via `posix_fadvise(DONTNEED)` completes "
        "outside the timer. The request timer starts before prompt construction "
        "and tokenization, then includes input preparation, initial H2D, vision "
        "recomputation or SSD read/scatter, prefill, first-token selection, and "
        "the first-token CUDA synchronization. `core_ttft_ms` is diagnostic only.",
        "",
        "Reads use buffered `pread`; neither `O_DIRECT` nor an SSD-controller "
        "cache flush is used. This is an OS-page-cache-cold condition, not a "
        "physically true-cold-SSD claim. JPEG read/decode is outside the timer; "
        "model resize/crop/tensorization is inside.",
        "",
        "## Main cache-hit result: Turns 2--10",
        "",
        "| Method | Aux quality† | TTFT mean | p50 | p95 | Δ TTFT vs ReComp | TTFT reduction | E2E mean | SSD MB/req | SSD ratio vs FullLoad |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = main[method]
        lines.append(
            f"| {method} | {fmt(row['aux_quality_mean'], 3)} | "
            f"{fmt(row['end_to_end_ttft_ms_mean'])} | "
            f"{fmt(row['end_to_end_ttft_ms_p50'])} | "
            f"{fmt(row['end_to_end_ttft_ms_p95'])} | "
            f"{fmt(row['ttft_delta_vs_recomp_ms'])} | "
            f"{fmt(row['ttft_reduction_vs_recomp_pct'])}% | "
            f"{fmt(row['request_e2e_ms_mean'])} | "
            f"{fmt(row['ssd_read_mb_per_request'])} | "
            f"{('—' if row['ssd_ratio_vs_fullload_pct'] is None else fmt(row['ssd_ratio_vs_fullload_pct']) + '%')} |"
        )
    lines.extend([
        "",
        "† Normalized generative-match auxiliary score; not official VisDial "
        "MRR/R@K/Mean Rank/NDCG.",
        "",
        "### Main comparisons",
        "",
        f"- Prefix25 vs ReComp: TTFT {fmt(main['Prefix25']['ttft_reduction_vs_recomp_pct'])}% lower, E2E {fmt(main['Prefix25']['e2e_reduction_vs_recomp_pct'])}% lower, quality delta {fmt(main['Prefix25']['aux_quality_delta_vs_recomp'], 3)}.",
        f"- Prefix45 vs ReComp: TTFT {fmt(main['Prefix45']['ttft_reduction_vs_recomp_pct'])}% lower, E2E {fmt(main['Prefix45']['e2e_reduction_vs_recomp_pct'])}% lower, quality delta {fmt(main['Prefix45']['aux_quality_delta_vs_recomp'], 3)}.",
        f"- Prefix25/45 reduce SSD bytes versus FullLoad by {fmt(main['Prefix25']['ssd_reduction_vs_fullload_pct'])}% / {fmt(main['Prefix45']['ssd_reduction_vs_fullload_pct'])}%.",
        "",
        "## Cache-hit I/O breakdown",
        "",
        "| Method | SSD MB/req | SSD read ms | OS preads/req | Scatter ms | Raw prefill ms | First-k planning ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    io_by_method = summary["io_breakdown"]
    for method in METHODS:
        row = io_by_method[method]
        lines.append(
            f"| {method} | {fmt(row['ssd_read_mb_per_request'])} | "
            f"{fmt(row['ssd_read_ms_mean'])} | "
            f"{fmt(row['preads_per_request_mean'])} | "
            f"{fmt(row['scatter_ms_mean'])} | "
            f"{fmt(row['prefill_ms_mean'])} | "
            f"{fmt(row['first_k_planning_ms_mean'])} |"
        )
    lines.extend([
        "",
        "`preads/req` is the measured `ssd_read_preads` count; raw chunk-unit "
        "counts are not treated as system calls. The inherited `selector_ms` "
        "field is deterministic first-k index planning only: static/query/"
        "diversity scorer calls are all zero.",
        "Main-table SSD ratios use actual aggregate read bytes divided by "
        "FullLoad aggregate read bytes. They are not nominal budgets or the "
        "unweighted mean of per-image selected-KV ratios.",
        "",
        "Phase caveat: ReComp raw `prefill_ms` includes vision, FullLoad raw "
        "`prefill_ms` includes its hook-based SSD read/cache write, and Prefix "
        "prefill starts after separately measured read/scatter. FullLoad scatter "
        "is therefore N/A rather than zero. These raw phase columns are "
        "diagnostic and must not be compared as mutually exclusive compute phases.",
        "",
        "## Turn-1 sanity",
        "",
        "| Method | TTFT mean/p50/p95 (ms) | Prediction agreement | First-token agreement | Vision forwards | SSD read |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in turn1:
        lines.append(
            f"| {row['method']} | {fmt(row['end_to_end_ttft_ms_mean'])}/"
            f"{fmt(row['end_to_end_ttft_ms_p50'])}/"
            f"{fmt(row['end_to_end_ttft_ms_p95'])} | "
            f"{fmt(row['prediction_agreement_vs_recomp_pct'])}% | "
            f"{fmt(row['first_token_agreement_vs_recomp_pct'])}% | "
            f"{fmt(row['vision_forward_count_mean'])} | "
            f"{row['ssd_read_bytes_total']} B |"
        )
    lines.extend([
        "",
        "Every arm performs one normal pixel-based multimodal inference at Turn "
        "1. Prefix saliency and Visual KV are captured from that same forward; "
        "there is no separate vision or prefix recomputation.",
        "",
        "## One-time persistence overhead",
        "",
        "| Component | Mean | p50 | p95 |",
        "|---|---:|---:|---:|",
    ])
    for row in persistence:
        lines.append(
            f"| {row['component']} | {fmt(row['mean_ms'])} | "
            f"{fmt(row['p50_ms'])} | {fmt(row['p95_ms'])} |"
        )
    lines.extend([
        "",
        f"The measured one-time write is {summary['persistence_overhead']['one_time_ssd_write_mb_per_image_mean']:.2f} MB/image on average. This cost is outside subsequent cache-hit request timing, but is retained here and in the secondary amortization analysis.",
        "In the completed implementation it ran synchronously immediately "
        "after the source Answer 1; separating it here is an analysis boundary, "
        "not an overlap or background-execution claim.",
        "The total also includes postprocessing, atomic publication, context "
        "open, and measured residual time not all shown as headline components; "
        "component percentiles must not be summed to reconstruct the total percentile.",
        "",
        "## Turn-wise cache-hit TTFT",
        "",
        "| Turn | History tokens | ReComp | FullLoad | Prefix25 | Prefix45 | ReComp-P25 | ReComp-P45 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in figure:
        lines.append(
            f"| {row['turn_id']} | {fmt(row['history_tokens_mean'])} | "
            f"{fmt(row['ReComp'])} | {fmt(row['FullLoad'])} | "
            f"{fmt(row['Prefix25'])} | {fmt(row['Prefix45'])} | "
            f"{fmt(row['ReComp_minus_Prefix25_ms'])} | "
            f"{fmt(row['ReComp_minus_Prefix45_ms'])} |"
        )
    p25_ttft = amort["end_to_end_ttft_ms"]["Prefix25"]["worst"]
    p45_ttft = amort["end_to_end_ttft_ms"]["Prefix45"]["worst"]
    p25_e2e = amort["request_e2e_ms"]["Prefix25"]["worst"]
    p45_e2e = amort["request_e2e_ms"]["Prefix45"]["worst"]
    lines.extend([
        "",
        "The absolute advantage remains present as history grows: ReComp-Prefix25 "
        f"changes from {figure[0]['ReComp_minus_Prefix25_ms']:.2f} ms at Turn 2 "
        f"to {figure[-1]['ReComp_minus_Prefix25_ms']:.2f} ms at Turn 10; the "
        "corresponding Prefix45 values are "
        f"{figure[0]['ReComp_minus_Prefix45_ms']:.2f} and "
        f"{figure[-1]['ReComp_minus_Prefix45_ms']:.2f} ms.",
        "",
        "## Storage interpretation",
        "",
        "FullLoad reads the complete Visual KV from SSD and is slower than "
        "ReComp on cache-hit turns. SSD caching alone therefore does not "
        "guarantee lower TTFT; reducing bytes read is necessary in this setup. "
        "ReComp performs zero SSD read/write, so no claim is made that Prefix "
        "uses less SSD I/O than ReComp.",
        "",
        "## Secondary persistence/amortization result",
        "",
        "This is not the main cache-hit table. In the conservative back-to-back "
        "case, one measured persistence event is charged once for N>=2.",
        "",
        f"- Prefix25 strict break-even: TTFT N={p25_ttft['strict_break_even_turn']}, E2E N={p25_e2e['strict_break_even_turn']}.",
        f"- Prefix45 strict break-even: TTFT N={p45_ttft['strict_break_even_turn']}, E2E N={p45_e2e['strict_break_even_turn']}.",
        "",
        "## Paper-ready conclusion",
        "",
        summary["paper_ready_conclusion"],
        "",
        "## Validation and artifact semantics",
        "",
        f"All {len(validation['checks'])} fail-closed checks pass. The input raw "
        "run and prior result tree are hashed before and after analysis. Main "
        "tables are recomputed from raw rows rather than copied or hardcoded.",
        "",
        "- `main_cache_hit_table.csv`: paper main table, Turns 2--10 only.",
        "- `turn1_sanity.csv`: separate Turn-1 fairness/capture evidence.",
        "- `persistence_overhead.csv`: separate one-time cost.",
        "- `ttft_by_turn_cache_hit.csv`: detailed long-form trend.",
        "- `quality_by_turn_cache_hit.csv`: wide auxiliary-quality trend.",
        "- `io_breakdown_cache_hit.csv`: actual pread/scatter/prefill accounting.",
        "- `fig_cache_hit_ttft_by_turn.csv`: paper-figure-ready wide TTFT data.",
    ])
    return "\n".join(lines) + "\n"


def write_fsynced(path: Path, content: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o644)
    with os.fdopen(fd, "w") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def rename_noreplace(source: Path, destination: Path) -> None:
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
            raise AnalysisError(f"refusing to overwrite results: {destination}")
        raise OSError(error, os.strerror(error), destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze immutable VisDial Turn 2-10 cache-hit results"
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--source-results-dir", type=Path,
                        default=DEFAULT_SOURCE_RESULTS)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    source_results = args.source_results_dir.resolve()
    results_dir = args.results_dir.resolve()
    if os.path.lexists(results_dir):
        raise AnalysisError(f"refusing to overwrite results: {results_dir}")
    required = {
        "config": run_dir / "config.json",
        "raw": run_dir / "raw.jsonl",
        "persistence": run_dir / "persistence_per_image.csv",
        "persistence_jsonl": run_dir / "persistence.jsonl",
        "source_validation": source_results / "validation.json",
        "source_summary": source_results / "summary.csv",
        "source_readme": source_results / "README.md",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise AnalysisError("missing immutable source artifacts: " + ", ".join(missing))
    config = read_json(required["config"])
    index_path = Path(config.get("index", ""))
    if not index_path.is_file():
        raise AnalysisError(f"missing frozen index: {index_path}")
    required["index"] = index_path

    before_hashes = {name: sha256_file(path) for name, path in required.items()}
    before_run_tree = tree_sha256(run_dir)
    before_result_tree = tree_sha256(source_results)

    rows = canonicalize_raw(read_jsonl(required["raw"]))
    for row in rows:
        row["raw_saliency_extra_ms"] = as_float(
            row["raw"].get("saliency_extra_ms", 0),
            f"raw line {row['line_number']}/saliency_extra_ms", default=0,
        )
    persistence_source = read_csv(required["persistence"])
    source_validation = read_json(required["source_validation"])
    prior_summary = read_csv(required["source_summary"])
    validator = validate_inputs(
        rows, persistence_source, config, source_validation, before_hashes
    )
    if not validator.result()["passed"]:
        raise AnalysisError("source validation failed:\n" + json.dumps(
            validator.result(), indent=2, ensure_ascii=False
        ))

    cache_rows = [row for row in rows if row["turn_id"] in MAIN_TURNS]
    turn1_rows = [row for row in rows if row["turn_id"] == 1]
    main_table = build_main_table(cache_rows)
    turn1 = build_turn1_sanity(turn1_rows)
    persistence = build_persistence_overhead(persistence_source)
    turn_table, quality_turn, figure = build_turn_tables(cache_rows)
    io_rows = build_io_breakdown(cache_rows)
    amortization = build_amortization(rows, persistence_source)

    prior_equal, prior_detail = compare_to_prior_summary(main_table, prior_summary)
    validator.check("raw_reaggregation_matches_prior_validated_summary",
                    prior_equal, prior_detail)
    validator.check("main_table_uses_exactly_900_cache_hits_per_method",
                    all(row["n_requests"] == 900 and row["turn_range"] == "2-10"
                        for row in main_table))
    validator.check("main_table_excludes_persistence_columns",
                    all(not any("persist" in key for key in row)
                        for row in main_table))
    recomputed_means = {
        method: mean(method_rows(cache_rows, method), "end_to_end_ttft_ms")
        for method in METHODS
    }
    validator.check("main_ttft_is_raw_end_to_end_field_without_persistence",
                    all(abs(row["end_to_end_ttft_ms_mean"] -
                            recomputed_means[row["method"]]) <= 1e-12
                        for row in main_table))
    validator.check("persistence_overhead_reported_separately",
                    {row["component"] for row in persistence} >= {
                        "permutation", "kv_repack", "buffered_ssd_write",
                        "fsync", "total_persistence",
                    })
    validator.check("turn_tables_cover_only_turns_2_10",
                    len(turn_table) == 36 and len(quality_turn) == 9 and
                    len(figure) == 9 and
                    {row["turn_id"] for row in turn_table} == set(MAIN_TURNS))
    validator.check("figure_values_match_detailed_turn_table",
                    all(abs(
                        figure_row[method] - next(
                            row["end_to_end_ttft_ms_mean"] for row in turn_table
                            if row["turn_id"] == figure_row["turn_id"] and
                            row["method"] == method
                        )
                    ) <= 1e-12 for figure_row in figure for method in METHODS))
    validator.check("quality_turn_values_match_raw",
                    all(abs(
                        row[method] - mean([
                            item for item in cache_rows
                            if item["turn_id"] == row["turn_id"] and
                            item["method"] == method
                        ], "quality_score")
                    ) <= 1e-12 for row in quality_turn for method in METHODS))
    source_paths = dict(required)
    summary = make_summary(
        config, source_paths, before_hashes, main_table, turn1, persistence,
        figure, io_rows, persistence_source, amortization,
    )
    readme_wording_probe = build_readme(
        config, main_table, turn1, persistence, figure, summary,
        validator.result(),
    )
    wording_payload = readme_wording_probe + json.dumps(
        summary, ensure_ascii=False, allow_nan=False
    )
    validator.check(
        "persistence_wording_matches_synchronous_execution",
        "ran synchronously immediately" in readme_wording_probe and
        "asynchronous" not in wording_payload.lower() and
        "cache construction has no cost" not in wording_payload.lower(),
        {"required": "ran synchronously immediately",
         "forbidden": ["asynchronous", "cache construction has no cost"]},
    )
    analyzer_hash = sha256_file(Path(__file__).resolve())
    analysis_config = {
        "schema_version": SCHEMA_VERSION,
        "analysis_only": True,
        "new_inference_executed": False,
        "source_run_dir": str(run_dir),
        "source_results_dir": str(source_results),
        "results_dir": str(results_dir),
        "analyzer": {"path": str(Path(__file__).resolve()),
                     "sha256": analyzer_hash},
        "source_file_sha256": before_hashes,
        "source_run_tree_sha256_before": before_run_tree,
        "source_results_tree_sha256_before": before_result_tree,
        "dataset": config["dataset"],
        "index_sha256": config["index_sha256"],
        "ordered_request_keys_sha256": config["expected_request_keys_sha256"],
        "main_turns": list(MAIN_TURNS),
        "main_rows": len(cache_rows),
        "main_rows_per_method": 900,
        "turn1_in_main_rows": 0,
        "methods": list(METHODS),
        "main_metric": "end_to_end_ttft_ms",
        "core_ttft_role": "diagnostic_only",
        "persistence_in_main_ttft": False,
        "persistence_reported_separately": True,
        "history_policy": "gold_teacher_forced",
        "quality_metric": "normalized_generative_match_auxiliary_not_official_visdial",
        "ssd_units": {"MB": 1_000_000, "GB": 1_000_000_000},
        "page_cache_conditioning": {
            "method": "posix_fadvise(DONTNEED)",
            "inside_request_timer": False,
            "io_api": "buffered pread",
            "o_direct": False,
            "ssd_controller_cache_flushed": False,
        },
    }

    artifacts = {
        "config.json": json.dumps(analysis_config, indent=2, ensure_ascii=False,
                                  allow_nan=False) + "\n",
        "main_cache_hit_table.csv": csv_text(main_table),
        "turn1_sanity.csv": csv_text(turn1),
        "persistence_overhead.csv": csv_text(persistence),
        "ttft_by_turn_cache_hit.csv": csv_text(turn_table),
        "quality_by_turn_cache_hit.csv": csv_text(quality_turn),
        "io_breakdown_cache_hit.csv": csv_text(io_rows),
        "fig_cache_hit_ttft_by_turn.csv": csv_text(figure),
        "summary.json": json.dumps(summary, indent=2, ensure_ascii=False,
                                   allow_nan=False) + "\n",
    }
    validator.check("required_artifact_set_exact_before_validation",
                    set(artifacts) | {"validation.json", "README.md"} == RESULT_NAMES)

    after_hashes = {name: sha256_file(path) for name, path in required.items()}
    after_run_tree = tree_sha256(run_dir)
    after_result_tree = tree_sha256(source_results)
    validator.check("all_source_files_unchanged",
                    before_hashes == after_hashes,
                    {"before": before_hashes, "after": after_hashes})
    validator.check("source_run_tree_unchanged",
                    before_run_tree == after_run_tree,
                    {"before": before_run_tree, "after": after_run_tree})
    validator.check("source_results_tree_unchanged",
                    before_result_tree == after_result_tree,
                    {"before": before_result_tree, "after": after_result_tree})
    validation = validator.result()
    validation["input_sha256_before"] = before_hashes
    validation["input_sha256_after"] = after_hashes
    validation["source_run_tree_sha256_before"] = before_run_tree
    validation["source_run_tree_sha256_after"] = after_run_tree
    validation["source_results_tree_sha256_before"] = before_result_tree
    validation["source_results_tree_sha256_after"] = after_result_tree
    validation["analyzer"] = analysis_config["analyzer"]
    if not validation["passed"]:
        raise AnalysisError("derived validation failed:\n" + json.dumps(
            validation, indent=2, ensure_ascii=False
        ))
    artifacts["validation.json"] = json.dumps(
        validation, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"
    artifacts["README.md"] = build_readme(
        config, main_table, turn1, persistence, figure, summary, validation
    )

    results_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(
        prefix=f".{results_dir.name}.staging-", dir=results_dir.parent
    ))
    try:
        for name in sorted(RESULT_NAMES):
            write_fsynced(stage / name, artifacts[name])
        staged_names = {path.name for path in stage.iterdir() if path.is_file()}
        if staged_names != RESULT_NAMES:
            raise AnalysisError(
                f"staged artifact mismatch: {staged_names ^ RESULT_NAMES}"
            )
        directory_fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        rename_noreplace(stage, results_dir)
        parent_fd = os.open(results_dir.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise

    print(json.dumps({
        "status": "complete",
        "analysis_only": True,
        "new_inference_executed": False,
        "results_dir": str(results_dir),
        "cache_hit_rows": len(cache_rows),
        "validation_checks": len(validation["checks"]),
        "validation_passed": validation["passed"],
        "main": {
            row["method"]: {
                "quality": row["aux_quality_mean"],
                "ttft_ms": row["end_to_end_ttft_ms_mean"],
                "e2e_ms": row["request_e2e_ms_mean"],
                "ssd_mb_per_request": row["ssd_read_mb_per_request"],
            } for row in main_table
        },
    }, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

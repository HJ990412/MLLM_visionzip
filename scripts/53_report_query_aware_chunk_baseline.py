#!/usr/bin/env python3
"""Validate and publish the final QA-Chunk25 paper-facing report bundle.

The GPU runner deliberately owns measurement and the first-pass machine
artifacts.  This script is a CPU-only second reader: it parses the durable
``results_final.jsonl`` again, independently recomputes the important metrics,
checks every runner export, checks old-artifact/source-store protection, and
then publishes the numbered 1--26 analysis and a compact README.

Existing runner artifacts are never silently replaced.  A byte-identical file
is retained, a missing export is copied atomically from the run directory, and
any conflicting file makes publication fail closed.  Re-running the reporter
is therefore safe and idempotent.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import stat
import sys
import uuid
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent.parent
RUN_SCHEMA_VERSION = "qa-chunk-gqa-pilot-v1"
REPORT_SCHEMA_VERSION = "qa-chunk-final-report-v1"
EXPECTED_INDEX_SHA256 = (
    "514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a"
)
EXPECTED_WORKLOAD_SHA256 = (
    "97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66"
)
METHOD_KEYS = (
    "recompute", "fullload", "qa_token25", "qa_chunk25", "ours25",
)
METHOD_IDS = {
    "recompute": "recompute",
    "fullload": "fullload",
    "qa_token25": "qa_token25",
    "qa_chunk25": "qa_chunk25",
    "ours25": "imageonly_prefix25",
}
DISPLAY_LABELS = {
    "recompute": "ReComp",
    "fullload": "FullLoad",
    "qa_token25": "QA-Token25",
    "qa_chunk25": "QA-Chunk25",
    "ours25": "Ours25",
}
RUNNER_EXPORTS = (
    "results_final.jsonl",
    "summary.json",
    "summary.csv",
    "selection_analysis.json",
    "latency_breakdown.csv",
    "io_breakdown.csv",
    "validation.json",
    "RUN_ANALYSIS.md",
)
SOURCE_FILES = (
    "mmimpress/sparsevlm.py",
    "mmimpress/serve.py",
    "mmimpress/store.py",
    "mmimpress/cvpr25.py",
    "scripts/49_eval_query_aware_baseline.py",
    "scripts/50_protect_query_aware_artifacts.py",
    "scripts/51_protect_qa_chunk_source_stores.py",
    "scripts/52_eval_query_aware_chunk_baseline.py",
    "scripts/53_report_query_aware_chunk_baseline.py",
)


class ReportValidationError(RuntimeError):
    """The supplied evidence cannot safely support a final report."""


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _regular_file(path: Path) -> os.stat_result:
    try:
        value = path.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise ReportValidationError(f"missing required file: {path}") from error
    if not stat.S_ISREG(value.st_mode):
        raise ReportValidationError(f"required path is not a regular file: {path}")
    return value


def _sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    before = _regular_file(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if ((before.st_dev, before.st_ino, before.st_mode)
                != (opened.st_dev, opened.st_ino, opened.st_mode)):
            raise ReportValidationError(f"file changed while opening: {path}")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, block_size)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    for field in ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns"):
        if getattr(before, field) != getattr(after, field):
            raise ReportValidationError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    digest = _sha256_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReportValidationError(f"cannot parse JSON: {path}") from error
    if not isinstance(value, dict):
        raise ReportValidationError(f"JSON root is not an object: {path}")
    if _sha256_file(path) != digest:
        raise ReportValidationError(f"file changed while reading: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    digest = _sha256_file(path)
    rows: list[dict[str, Any]] = []
    line_number = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ReportValidationError(
                        f"blank JSONL line {line_number}: {path}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("row is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise ReportValidationError(
            f"cannot parse JSONL {path} at/near line {line_number}") from error
    if _sha256_file(path) != digest:
        raise ReportValidationError(f"file changed while reading: {path}")
    return rows


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportValidationError(f"{label} is not a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ReportValidationError(f"{label} is not a sequence")
    return value


def _finite(value: Any, label: str, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool):
        raise ReportValidationError(f"{label} is not numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ReportValidationError(f"{label} is not numeric") from error
    if not math.isfinite(number):
        raise ReportValidationError(f"{label} is not finite")
    return number


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [_finite(row.get(key), key) for row in rows
              if row.get(key) is not None]
    numbers = [float(value) for value in values if value is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ReportValidationError("percentile of empty values")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    a, b = set(map(int, left)), set(map(int, right))
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _layer_jaccards(left: Any, right: Any, label: str) -> list[float]:
    left_layers = _sequence(left, f"{label}.left")
    right_layers = _sequence(right, f"{label}.right")
    if len(left_layers) != len(right_layers) or not left_layers:
        raise ReportValidationError(f"{label} layer count mismatch/empty")
    return [_jaccard(
        _sequence(a, f"{label}.left_layer"),
        _sequence(b, f"{label}.right_layer"),
    ) for a, b in zip(left_layers, right_layers)]


def _max_run_length(row: Mapping[str, Any]) -> int:
    maximum = 0
    for layer in row.get("selected_chunk_ids_per_layer") or []:
        ordered = sorted(set(map(int, layer)))
        run = 0
        previous = None
        for value in ordered:
            run = run + 1 if previous is not None and value == previous + 1 else 1
            maximum = max(maximum, run)
            previous = value
    return maximum


def _recompute_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for method in METHOD_KEYS:
        all_rows = [row for row in rows if row.get("method_key") == method]
        hits = [row for row in all_rows if int(row.get("turn_id", 0)) > 1]
        ttft = [float(_finite(row.get("end_to_end_ttft_ms"), "TTFT"))
                for row in hits]
        output[method] = {
            "n_requests": len(all_rows),
            "n_cache_hit_requests": len(hits),
            "accuracy_all_turns": _mean(all_rows, "correct"),
            "accuracy_cache_hits": _mean(hits, "correct"),
            "ttft_cache_hit_mean_ms": sum(ttft) / len(ttft) if ttft else None,
            "ttft_cache_hit_p50_ms": _percentile(ttft, 50) if ttft else None,
            "ttft_cache_hit_p95_ms": _percentile(ttft, 95) if ttft else None,
            "turn1_ttft_mean_ms": _mean(
                [row for row in all_rows if int(row.get("turn_id", 0)) == 1],
                "end_to_end_ttft_ms"),
            "actual_ssd_mb_per_cache_hit": _mean(hits, "actual_ssd_mb"),
            "actual_ssd_ratio_vs_fullload": _mean(
                hits, "actual_ssd_ratio_vs_fullload"),
            "normal_selected_chunk_ratio": (
                _mean(hits, "normal_selected_chunk_ratio")
                if _mean(hits, "normal_selected_chunk_ratio") is not None
                else _mean(hits, "touched_chunk_fraction")),
            "total_touched_chunk_ratio": _mean(hits, "touched_chunk_fraction"),
            "touched_chunk_fraction": _mean(hits, "touched_chunk_fraction"),
            "selected_chunk_payload_mb": (
                (_mean(hits, "normal_kv_read_bytes") or 0.0) / 1e6),
            "probe_io_mb": (_mean(hits, "probe_read_bytes") or 0.0) / 1e6,
            "separator_io_mb": (
                (_mean(hits, "separator_read_bytes") or 0.0) / 1e6),
            "ssd_preads_per_cache_hit": _mean(hits, "ssd_preads"),
            "ssd_read_latency_ms": _mean(hits, "ssd_read_ms"),
            "contiguous_runs_per_layer": _mean(
                hits, "contiguous_runs_per_layer_mean"),
            "mean_contiguous_run_length": _mean(
                hits, "mean_contiguous_run_length"),
            "max_contiguous_run_length": float(max(
                (_max_run_length(row) for row in hits), default=0)),
            "query_score_calls_total": sum(
                int(row.get("query_score_calls", 0)) for row in hits),
            "chunk_score_calls_total": sum(
                int(row.get("chunk_score_calls", 0)) for row in hits),
        }
        for field in (
            "selector_ms", "online_selector_total_ms",
            "selector_decision_host_wall_ms", "rater_selection_ms",
            "query_projection_ms", "probe_h2d_ms", "probe_io_ms",
            "normal_kv_read_ms", "selected_chunk_io_ms",
            "separator_read_ms",
            "query_scoring_ms", "chunk_aggregation_ms", "topk_chunk_ms",
            "selected_id_d2h_ms", "chunk_planning_ms", "chunk_io_ms",
            "scatter_ms", "prefill_ms",
        ):
            output[method][field] = _mean(hits, field)
    return output


def _recompute_selection(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    chunks = [row for row in rows if row.get("method_key") == "qa_chunk25"
              and int(row.get("turn_id", 0)) > 1]
    tokens = {(str(row["image_id"]), str(row["question_id"])): row
              for row in rows if row.get("method_key") == "qa_token25"
              and int(row.get("turn_id", 0)) > 1}
    by_image: dict[str, list[Mapping[str, Any]]] = {}
    requests: list[dict[str, Any]] = []
    overlaps: list[dict[str, Any]] = []
    for row in chunks:
        image_id, question_id = str(row["image_id"]), str(row["question_id"])
        by_image.setdefault(image_id, []).append(row)
        selected = row.get("selected_chunk_ids_per_layer") or []
        requests.append({
            "image_id": row["image_id"], "question_id": row["question_id"],
            "turn_id": int(row["turn_id"]),
            "selected_chunk_ids_per_layer": selected,
            "selected_chunks_sha256": _canonical_hash(selected),
        })
        token = tokens.get((image_id, question_id))
        if token is None:
            raise ReportValidationError(
                f"missing QA-Token peer for {image_id}/{question_id}")
        touched = token.get("selected_chunk_ids_per_layer") or []
        layer_values = _layer_jaccards(touched, selected, "token/chunk overlap")
        intersections = [len(set(map(int, a)) & set(map(int, b)))
                         for a, b in zip(touched, selected)]
        unions = [len(set(map(int, a)) | set(map(int, b)))
                  for a, b in zip(touched, selected)]
        overlaps.append({
            "image_id": row["image_id"], "question_id": row["question_id"],
            "qa_token_touched_chunks_per_layer": [len(value) for value in touched],
            "qa_chunk_selected_chunks_per_layer": [len(value) for value in selected],
            "intersection_per_layer": intersections,
            "union_per_layer": unions,
            "jaccard_per_layer": layer_values,
            "mean_layer_jaccard": sum(layer_values) / len(layer_values),
        })
    pairs: list[dict[str, Any]] = []
    for image_id, image_rows in sorted(by_image.items()):
        image_rows.sort(key=lambda row: int(row["turn_id"]))
        for left, right in combinations(image_rows, 2):
            values = _layer_jaccards(
                left.get("selected_chunk_ids_per_layer") or [],
                right.get("selected_chunk_ids_per_layer") or [],
                "QA-Chunk pair")
            identical = all(value == 1.0 for value in values)
            pairs.append({
                "image_id": image_id,
                "left_turn": int(left["turn_id"]),
                "right_turn": int(right["turn_id"]),
                "left_question_id": left["question_id"],
                "right_question_id": right["question_id"],
                "chunk_jaccard": sum(values) / len(values),
                "chunk_jaccard_per_layer": values,
                "identical": identical,
                "consecutive": int(right["turn_id"]) == int(left["turn_id"]) + 1,
            })
    pair_values = [float(row["chunk_jaccard"]) for row in pairs]
    consecutive = [float(row["chunk_jaccard"]) for row in pairs
                   if row["consecutive"]]
    overlap_values = [float(row["mean_layer_jaccard"]) for row in overlaps]
    token_counts = [value for row in overlaps
                    for value in row["qa_token_touched_chunks_per_layer"]]
    chunk_counts = [value for row in overlaps
                    for value in row["qa_chunk_selected_chunks_per_layer"]]
    intersections = [value for row in overlaps
                     for value in row["intersection_per_layer"]]
    return {
        "scope": "cache-hit turns 2..6",
        "n_query_requests": len(chunks), "n_images": len(by_image),
        "requests": sorted(requests, key=lambda row: (
            str(row["image_id"]), row["turn_id"], str(row["question_id"]))),
        "pairs": pairs, "n_pairs": len(pairs),
        "mean_pairwise_chunk_jaccard": (
            sum(pair_values) / len(pair_values) if pair_values else None),
        "mean_consecutive_chunk_jaccard": (
            sum(consecutive) / len(consecutive) if consecutive else None),
        "identical_chunk_selection_rate": (
            sum(bool(row["identical"]) for row in pairs) / len(pairs)
            if pairs else None),
        "different_selection_pairs": sum(
            not bool(row["identical"]) for row in pairs),
        "n_consecutive_pairs": len(consecutive),
        "qa_token_overlap": {
            "requests": overlaps, "n_requests": len(overlaps),
            "mean_layer_jaccard": (
                sum(overlap_values) / len(overlap_values)
                if overlap_values else None),
            "mean_qa_token_touched_chunks_per_layer": (
                sum(token_counts) / len(token_counts) if token_counts else None),
            "mean_qa_chunk_selected_chunks_per_layer": (
                sum(chunk_counts) / len(chunk_counts) if chunk_counts else None),
            "mean_intersection_chunks_per_layer": (
                sum(intersections) / len(intersections)
                if intersections else None),
        },
    }


def _equivalent(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise ReportValidationError(f"{label} is not a mapping")
        for key, value in expected.items():
            if key not in actual:
                raise ReportValidationError(f"{label} missing {key}")
            _equivalent(actual[key], value, f"{label}.{key}")
        return
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        if (not isinstance(actual, Sequence) or isinstance(actual, (str, bytes))
                or len(actual) != len(expected)):
            raise ReportValidationError(f"{label} sequence mismatch")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _equivalent(left, right, f"{label}[{index}]")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        observed = _finite(actual, label)
        if observed is None or not math.isclose(
                observed, float(expected), rel_tol=1e-10, abs_tol=1e-10):
            raise ReportValidationError(
                f"{label} differs: {observed!r} != {expected!r}")
        return
    if actual != expected:
        raise ReportValidationError(f"{label} differs: {actual!r} != {expected!r}")


def _verify_runner_artifacts(
    run_dir: Path, results_dir: Path, summary: Mapping[str, Any],
    recomputed: Mapping[str, Mapping[str, Any]],
    selection: Mapping[str, Any], recomputed_selection: Mapping[str, Any],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in RUNNER_EXPORTS:
        source = run_dir / name
        target = results_dir / name
        source_hash = _sha256_file(source)
        hashes[name] = source_hash
        if os.path.lexists(target):
            if _sha256_file(target) != source_hash:
                raise ReportValidationError(
                    f"runner export differs between run/results: {name}")
        else:
            _publish_bytes(target, source.read_bytes())

    partial = run_dir / "results_partial.jsonl"
    if _sha256_file(partial) != hashes["results_final.jsonl"]:
        raise ReportValidationError(
            "results_partial.jsonl and results_final.jsonl differ")

    methods = _mapping(summary.get("per_method"), "summary.per_method")
    if set(methods) != set(METHOD_KEYS):
        raise ReportValidationError("summary method set mismatch")
    for method in METHOD_KEYS:
        row = _mapping(methods[method], f"summary.{method}")
        if row.get("method_id") != METHOD_IDS[method]:
            raise ReportValidationError(f"{method} method_id mismatch")
        if row.get("display_label") != DISPLAY_LABELS[method]:
            raise ReportValidationError(f"{method} display label mismatch")
        _equivalent(row, recomputed[method], f"summary.{method}")
    _equivalent(selection, recomputed_selection, "selection_analysis")

    artifacts_path = run_dir / "run_artifacts.json"
    artifacts = _read_json(artifacts_path)
    if artifacts.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ReportValidationError("run_artifacts schema mismatch")
    recorded = _mapping(artifacts.get("files_sha256"), "run_artifacts.files")
    for name, digest in recorded.items():
        if _sha256_file(run_dir / str(name)) != digest:
            raise ReportValidationError(f"run_artifacts hash mismatch: {name}")
    result_artifacts = results_dir / "run_artifacts.json"
    if os.path.lexists(result_artifacts):
        if _sha256_file(result_artifacts) != _sha256_file(artifacts_path):
            raise ReportValidationError("run_artifacts run/results mismatch")
    else:
        _publish_bytes(result_artifacts, artifacts_path.read_bytes())

    # CSVs must be parseable, have one row per method in stable order, and
    # expose values consistent with summary.json.  Exact bytes remain runner-owned.
    for filename in ("summary.csv", "latency_breakdown.csv", "io_breakdown.csv"):
        with (run_dir / filename).open(newline="", encoding="utf-8") as handle:
            csv_rows = list(csv.DictReader(handle))
        if [row.get("method_key") for row in csv_rows] != list(METHOD_KEYS):
            raise ReportValidationError(f"{filename} method rows mismatch")
    return hashes


def _validate_protection_file(path: Path, label: str) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("passed") is not True:
        raise ReportValidationError(f"{label} did not pass")
    for field in ("missing_paths", "added_paths", "changed_paths"):
        if field in value and value.get(field) != []:
            raise ReportValidationError(f"{label} reports {field}")
    if ("read_only_source_reuse_validated" in value
            and value.get("read_only_source_reuse_validated") is not True):
        raise ReportValidationError(f"{label} read-only gate failed")
    return value


def _validate_general_protection_pair(
    run_dir: Path, results_dir: Path, validation_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_json(run_dir.parent / "protected_artifacts_before.json")
    report = _validate_protection_file(
        validation_path, "prior-artifact protection")
    entries = manifest.get("entries")
    if not isinstance(entries, dict):
        raise ReportValidationError("prior-artifact manifest entries missing")
    if _canonical_hash(entries) != manifest.get("manifest_sha256"):
        raise ReportValidationError("prior-artifact manifest was modified")
    exclusions = manifest.get("excluded_new_roots")
    if exclusions != [str(run_dir.parent), str(results_dir.parent)]:
        raise ReportValidationError("prior-artifact exclusion roots mismatch")
    digest = manifest.get("manifest_sha256")
    if (report.get("before_manifest_sha256") != digest
            or report.get("after_manifest_sha256") != digest):
        raise ReportValidationError("prior-artifact before/after hash mismatch")
    for stem in ("entry_count", "file_count", "directory_count",
                 "symlink_count", "total_bytes"):
        if (report.get(f"{stem}_before") != manifest.get(stem)
                or report.get(f"{stem}_after") != manifest.get(stem)):
            raise ReportValidationError(
                f"prior-artifact {stem} before/after mismatch")
    return manifest, report


def _validate_source_protection_pair(
    run_dir: Path, validation_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_json(run_dir.parent / "source_stores_before.json")
    report = _validate_protection_file(
        validation_path, "source-store protection")
    unsigned = dict(manifest)
    recorded = unsigned.pop("manifest_sha256", None)
    if _canonical_hash(unsigned) != recorded:
        raise ReportValidationError("source-store manifest was modified")
    if report.get("before_manifest_sha256") != recorded:
        raise ReportValidationError("source-store before manifest mismatch")
    if (report.get("before_store_fingerprint_sha256")
            != manifest.get("source_store_fingerprint_sha256")):
        raise ReportValidationError("source-store before fingerprint mismatch")
    if (report.get("before_store_fingerprint_sha256")
            != report.get("after_store_fingerprint_sha256")):
        raise ReportValidationError("source-store fingerprint changed")
    if (report.get("before_store_tree_sha256")
            != report.get("after_store_tree_sha256")):
        raise ReportValidationError("source-store tree hash changed")
    return manifest, report


def _default_general_protection(results_dir: Path) -> Path:
    return results_dir.parent / "protected_artifacts_validation.json"


def _default_source_protection(results_dir: Path) -> Path:
    candidates = (
        results_dir / "source_stores_validation.json",
        results_dir.parent / "source_stores_validation.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def _independent_checks(
    config: Mapping[str, Any], manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]], selection: Mapping[str, Any],
    runner_validation: Mapping[str, Any], completed: Mapping[str, Any] | None,
    protection_ok: bool, source_protection_ok: bool, tests_ok: bool,
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks["runner_schema"] = all(
        value.get("schema_version") == RUN_SCHEMA_VERSION
        for value in (config, manifest, runner_validation))
    checks["runner_status_complete"] = config.get("status") == "complete"
    checks["frozen_index"] = config.get("index_sha256") == EXPECTED_INDEX_SHA256
    checks["frozen_workload"] = (
        config.get("full_workload_sha256") == EXPECTED_WORKLOAD_SHA256
        and config.get("selected_workload_sha256") == EXPECTED_WORKLOAD_SHA256)
    checks["full_40_image_240_question_contract"] = (
        config.get("n_images") == 40 and config.get("n_questions") == 240
        and config.get("selected_images") == 40
        and config.get("selected_questions") == 240
        and config.get("skip") == 4 and config.get("questions_per_image") == 6)
    checks["five_methods_exact"] = (
        tuple(config.get("method_keys", ())) == METHOD_KEYS
        and tuple(manifest.get("method_keys", ())) == METHOD_KEYS)
    identities = [str(row.get("request_id")) for row in rows]
    checks["expected_completed_count"] = len(rows) == 1200
    checks["duplicates_zero"] = len(identities) == len(set(identities))
    checks["per_method_counts"] = all(sum(
        row.get("method_key") == method for row in rows) == 240
        for method in METHOD_KEYS)
    checks["per_method_cache_hits"] = all(sum(
        row.get("method_key") == method and int(row.get("turn_id", 0)) > 1
        for row in rows) == 200 for method in METHOD_KEYS)
    checks["request_ids_canonical"] = all(
        row.get("request_id") == "gqa:{image}:{question}:{method}".format(
            image=row.get("image_id"), question=row.get("question_id"),
            method=row.get("method_key")) for row in rows)
    checks["method_metadata_stable"] = all(
        row.get("method_id") == METHOD_IDS.get(str(row.get("method_key")))
        and row.get("display_label") == DISPLAY_LABELS.get(
            str(row.get("method_key"))) for row in rows)
    checks["future_question_leakage_zero"] = (
        config.get("future_question_leakage") == 0 and all(
            row.get("future_question_ids_used") == []
            and int(row.get("future_questions_in_prompt", -1)) == 0
            for row in rows))
    chunk = [row for row in rows if row.get("method_key") == "qa_chunk25"
             and int(row.get("turn_id", 0)) > 1]
    token = [row for row in rows if row.get("method_key") == "qa_token25"
             and int(row.get("turn_id", 0)) > 1]
    ours_by_request = {
        (str(row["image_id"]), str(row["question_id"])): row for row in rows
        if row.get("method_key") == "ours25" and int(row.get("turn_id", 0)) > 1}
    checks["qa_chunk_query_scoring_called"] = bool(chunk) and all(
        int(row.get("query_score_calls", 0))
        == int(row.get("expected_layers", -1)) > 0 for row in chunk)
    checks["qa_chunk_scores_created"] = bool(chunk) and all(
        int(row.get("chunk_score_calls", 0))
        == int(row.get("expected_layers", -1)) > 0 for row in chunk)
    checks["qa_chunk_algorithm_fixed"] = bool(chunk) and all(
        row.get("selection_granularity") == "ssd_chunk"
        and row.get("chunk_score_aggregation")
        == "mean_valid_spatial_token_importance"
        and row.get("physical_layout") == "raster"
        and row.get("repacking") is False
        and row.get("adaptive_ratio") is False
        and int(row.get("full_load_fallback_count", -1)) == 0
        for row in chunk)
    checks["actual_loaded_equals_selected"] = bool(chunk) and all(
        row.get("actual_loaded_chunk_ids_per_layer")
        == row.get("selected_chunk_ids_per_layer") for row in chunk)
    checks["ours_matched_chunk_count"] = bool(chunk) and all(
        [len(layer) for layer in row.get("selected_chunk_ids_per_layer", [])]
        == [len(layer) for layer in ours_by_request[
            (str(row["image_id"]), str(row["question_id"]))
        ].get("selected_chunk_ids_per_layer", [])] for row in chunk)
    checks["ssd_bytes_include_probe_selected_separator"] = all(
        int(row.get("ssd_read_bytes", -1))
        == int(row.get("normal_kv_read_bytes", 0))
        + int(row.get("probe_read_bytes", 0))
        + int(row.get("separator_read_bytes", 0)) for row in chunk + token)
    checks["selection_evidence_complete"] = (
        selection.get("n_query_requests") == 200
        and selection.get("n_images") == 40
        and selection.get("n_pairs") == 400
        and selection.get("n_consecutive_pairs") == 160
        and selection.get("qa_token_overlap", {}).get("n_requests") == 200)
    checks["query_dependence_observed"] = int(
        selection.get("different_selection_pairs", 0)) > 0
    runner_checks = runner_validation.get("checks")
    checks["runner_validation_passed"] = (
        runner_validation.get("passed") is True
        and isinstance(runner_checks, Mapping) and bool(runner_checks)
        and all(value is True for value in runner_checks.values()))
    checks["completed_marker_valid"] = (
        completed is not None and completed.get("passed") is True
        and completed.get("completed") == 1200
        and completed.get("_validation_hash_matches") is True)
    checks["prior_artifacts_unchanged"] = protection_ok
    checks["source_stores_unchanged"] = source_protection_ok
    checks["cpu_tests_passed"] = tests_ok
    return checks


def _publish_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ReportValidationError(f"output parent is unsafe: {path.parent}")
    digest = hashlib.sha256(payload).hexdigest()
    if os.path.lexists(path):
        if _sha256_file(path) != digest:
            raise ReportValidationError(
                f"refusing to replace conflicting existing artifact: {path}")
        return "preserved"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return "published"


def _preflight_bytes(path: Path, payload: bytes) -> None:
    """Reject a conflicting artifact before any report output is published."""
    if os.path.lexists(path):
        digest = hashlib.sha256(payload).hexdigest()
        if _sha256_file(path) != digest:
            raise ReportValidationError(
                f"refusing to replace conflicting existing artifact: {path}")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def _pct(value: Any, digits: int = 2) -> str:
    number = _finite(value, "percentage", nullable=True)
    return "—" if number is None else f"{100.0 * number:.{digits}f}%"


def _num(value: Any, digits: int = 2) -> str:
    number = _finite(value, "number", nullable=True)
    return "—" if number is None else f"{number:.{digits}f}"


def _signed(value: Any, digits: int = 2) -> str:
    number = _finite(value, "signed")
    assert number is not None
    return f"{number:+.{digits}f}"


def _comparisons(methods: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    chunk, token, ours = (methods[key] for key in (
        "qa_chunk25", "qa_token25", "ours25"))
    token_mb = float(token["actual_ssd_mb_per_cache_hit"])
    return {
        "chunk_ssd_reduction_vs_token_percent": 100.0 * (
            1.0 - float(chunk["actual_ssd_mb_per_cache_hit"]) / token_mb),
        "chunk_minus_token_accuracy_pp": 100.0 * (
            float(chunk["accuracy_all_turns"])
            - float(token["accuracy_all_turns"])),
        "chunk_minus_ours_accuracy_pp": 100.0 * (
            float(chunk["accuracy_all_turns"])
            - float(ours["accuracy_all_turns"])),
        "chunk_minus_ours_ttft_ms": (
            float(chunk["ttft_cache_hit_mean_ms"])
            - float(ours["ttft_cache_hit_mean_ms"])),
        "chunk_over_ours_ttft_ratio": (
            float(chunk["ttft_cache_hit_mean_ms"])
            / float(ours["ttft_cache_hit_mean_ms"])),
        "chunk_minus_token_ttft_ms": (
            float(chunk["ttft_cache_hit_mean_ms"])
            - float(token["ttft_cache_hit_mean_ms"])),
    }


def _dominant_latency_explanation(
    chunk: Mapping[str, Any], ours: Mapping[str, Any]) -> str:
    candidates = {
        "selector wall": max(0.0, float(chunk.get("online_selector_total_ms") or 0)
                             - float(ours.get("online_selector_total_ms") or 0)),
        "scattered chunk I/O": max(0.0, float(chunk.get("chunk_io_ms") or 0)
                                   - float(ours.get("chunk_io_ms") or 0)),
        "prefill": max(0.0, float(chunk.get("prefill_ms") or 0)
                       - float(ours.get("prefill_ms") or 0)),
    }
    ordered = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
    return ", ".join(f"{name} Δ≈{value:.2f} ms" for name, value in ordered)


def _build_analysis(
    run_dir: Path, results_dir: Path, config: Mapping[str, Any],
    summary: Mapping[str, Any], selection: Mapping[str, Any],
    runner_validation: Mapping[str, Any], report_validation: Mapping[str, Any],
    runner_hashes: Mapping[str, str], general_protection: Mapping[str, Any],
    source_protection: Mapping[str, Any], test_result: str,
) -> str:
    methods = _mapping(summary["per_method"], "summary.per_method")
    chunk, token, ours = (methods[key] for key in (
        "qa_chunk25", "qa_token25", "ours25"))
    overlap = _mapping(selection["qa_token_overlap"], "qa_token_overlap")
    comparison = _comparisons(methods)
    passed = bool(report_validation["passed"])
    verdict = "YES" if passed else "NO"
    checks = _mapping(report_validation["checks"], "report checks")
    failed_checks = [key for key, value in checks.items() if value is not True]
    runner_checks = _mapping(runner_validation.get("checks", {}), "runner checks")
    limitations = runner_validation.get("limitations", [])
    if not isinstance(limitations, list):
        raise ReportValidationError("validation limitations are malformed")

    lines = [
        "# QA-Chunk25 GQA 40/240 최종 분석", "",
        ("지속 저장된 five-arm raw 결과를 CPU-only reporter가 다시 읽어 "
         "독립 집계했고, runner 산출물·provenance·artifact protection을 교차 "
         f"검증했다. 최종 판정은 **{verdict}**다."), "",
        "## Main pilot table", "",
        ("Accuracy는 240개 전체 질문, TTFT/I/O는 cache-hit turns 2–6의 "
         "request mean이다. MB는 decimal MB(10^6 bytes)다."), "",
        "| Method | Accuracy | TTFT | Query-aware | Physical Chunk Budget | SSD MB | SSD Ratio | Selector ms | Touched Chunks | Preads |",
        "|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in METHOD_KEYS:
        row = methods[key]
        budget = ("—" if key == "recompute" else
                  "100.00%" if key == "fullload" else
                  _pct(row.get("normal_selected_chunk_ratio")
                       if key in {"qa_chunk25", "ours25"}
                       else row.get("touched_chunk_fraction")))
        lines.append(
            f"| {DISPLAY_LABELS[key]} | {_pct(row['accuracy_all_turns'])} | "
            f"{_num(row['ttft_cache_hit_mean_ms'])} ms | "
            f"{'Yes' if row.get('query_dependent') else 'No'} | {budget} | "
            f"{_num(row['actual_ssd_mb_per_cache_hit'])} | "
            f"{_pct(row['actual_ssd_ratio_vs_fullload'])} | "
            f"{_num(row['online_selector_total_ms'])} | "
            f"{_pct(row.get('touched_chunk_fraction'))} | "
            f"{_num(row['ssd_preads_per_cache_hit'])} |")

    lines.extend([
        "", "## A–F 핵심 답변", "",
        "### A. QA-Token25 대비 SSD I/O 감소", "",
        (f"QA-Chunk25는 {_num(token['actual_ssd_mb_per_cache_hit'])}에서 "
         f"{_num(chunk['actual_ssd_mb_per_cache_hit'])} MB/request로 "
         f"{comparison['chunk_ssd_reduction_vs_token_percent']:.2f}% 줄였다."), "",
        "### B. 그 대가의 accuracy 변화", "",
        (f"QA-Chunk25−QA-Token25 accuracy는 "
         f"{_signed(comparison['chunk_minus_token_accuracy_pp'])} pp다."), "",
        "### C. 약 25% physical budget에서 Ours 대비 quality", "",
        (f"QA-Chunk25−Ours25 accuracy는 "
         f"{_signed(comparison['chunk_minus_ours_accuracy_pp'])} pp다."), "",
        "### D. Ours 대비 속도", "",
        (f"QA-Chunk25는 Ours25보다 "
         f"{_signed(comparison['chunk_minus_ours_ttft_ms'])} ms 차이가 나며 "
         f"TTFT ratio는 {comparison['chunk_over_ours_ttft_ratio']:.4f}×다."), "",
        "### E. latency gap의 관측 구성", "",
        (_dominant_latency_explanation(chunk, ours)
         + ". Component는 overlap 가능하므로 이 순위는 진단용이고, TTFT와 "
           "observed selector wall이 권위 있는 수치다."), "",
        "### F. 질문별 chunk 선택 변화", "",
        (f"400 within-image query pairs 중 "
         f"{selection['different_selection_pairs']} pairs가 달랐고, mean "
         f"pairwise Jaccard={_num(selection['mean_pairwise_chunk_jaccard'], 6)}, "
         f"consecutive={_num(selection['mean_consecutive_chunk_jaccard'], 6)}, "
         f"identical rate={_pct(selection['identical_chunk_selection_rate'], 4)}다."), "",
        "## 요청된 1–26 항목", "",
        "### 1. QA-Chunk25 exact algorithm", "",
        ("현재 질문과 causal history에서 SparseVLM text raters를 고르고, 저장된 "
         "probe K와 query Q로 layer별 visual-token importance를 계산한다. 이를 "
         "physical raster chunk score로 집계해 Top chunks만 K/V pread하고 GPU "
         "scatter/mask 후 prefill·generation한다."), "",
        "### 2. Chunk score aggregation 정의", "",
        ("Main baseline은 `mean_valid_spatial_token_importance` 하나로 고정했다. "
         "separator/newline/structural rows와 final-chunk padding은 numerator와 "
         "denominator에서 제외하며 실제 valid spatial row 수만 분모로 쓴다."), "",
        "### 3. 25% chunk budget 계산 방식", "",
        ("`cvpr25.budget_chunk_count(n_chunks, 0.25)`의 "
         "`round(n_chunks × 0.25)` semantics를 Ours25와 공유한다. 따라서 token "
         "ceil이 아니라 이미지별 Ours와 정확히 같은 normal chunk count다."), "",
        "### 4. Physical layout", "",
        ("QA-Chunk25와 QA-Token25/FullLoad는 canonical original raster layout을 "
         "사용하며 repacking=false다. Ours25만 image-only saliency로 importance-aware "
         "repacked layout의 first-k prefix를 사용한다."), "",
        "### 5. Query-dependent scoring 호출 횟수", "",
        (f"QA-Chunk25 query_score_calls 총합은 "
         f"{int(chunk['query_score_calls_total']):,}, chunk_score_calls 총합은 "
         f"{int(chunk['chunk_score_calls_total']):,}다. Cache-hit request마다 모든 "
         "decoder layer에서 한 번씩 실행됐다."), "",
        "### 6. Selected chunk ratio", "",
        (f"normal selected chunk ratio={_pct(chunk['normal_selected_chunk_ratio'], 4)}, "
         f"total touched chunk ratio={_pct(chunk['total_touched_chunk_ratio'], 4)}다. "
         "Probe와 separator sidecar는 ratio 밖이지만 byte/latency/pread에는 포함된다."), "",
        "### 7. Query-pair chunk Jaccard", "",
        (f"전체 pair mean={_num(selection['mean_pairwise_chunk_jaccard'], 6)}, "
         f"consecutive mean={_num(selection['mean_consecutive_chunk_jaccard'], 6)} "
         f"(pairs={selection['n_pairs']}, consecutive={selection['n_consecutive_pairs']})."), "",
        "### 8. Identical selection rate", "",
        (f"Exact layerwise chunk selection identical rate는 "
         f"{_pct(selection['identical_chunk_selection_rate'], 4)}이며, 다른 pair는 "
         f"{selection['different_selection_pairs']}/{selection['n_pairs']}다."), "",
        "### 9. QA-Token25와 selection overlap", "",
        (f"동일 query/layer에서 QA-Token touched chunks와 QA-Chunk selected chunks의 "
         f"mean Jaccard={_num(overlap['mean_layer_jaccard'], 6)}다. 평균 chunk 수는 "
         f"{_num(overlap['mean_qa_token_touched_chunks_per_layer'])} 대 "
         f"{_num(overlap['mean_qa_chunk_selected_chunks_per_layer'])}, intersection은 "
         f"{_num(overlap['mean_intersection_chunks_per_layer'])}/layer다. Exact IDs와 "
         "per-request intersection은 `selection_analysis.json`에 있다."), "",
        "### 10. Accuracy", "",
    ])
    lines.extend(
        f"- {DISPLAY_LABELS[key]}: {_pct(methods[key]['accuracy_all_turns'])} "
        f"(cache-hit {_pct(methods[key]['accuracy_cache_hits'])})"
        for key in METHOD_KEYS)
    lines.extend([
        "", "### 11. TTFT", "",
    ])
    lines.extend(
        f"- {DISPLAY_LABELS[key]}: mean {_num(methods[key]['ttft_cache_hit_mean_ms'])} ms, "
        f"p50 {_num(methods[key]['ttft_cache_hit_p50_ms'])}, "
        f"p95 {_num(methods[key]['ttft_cache_hit_p95_ms'])} ms"
        for key in METHOD_KEYS)
    lines.extend([
        "", "TTFT는 request start부터 synchronized first output token까지며 prompt, "
        "tokenization, initial H2D, selector/probe/I/O/scatter, prefill을 포함한다.", "",
        "### 12. Selector wall-clock", "",
        (f"QA-Chunk25 online_selector_total_ms={_num(chunk['online_selector_total_ms'])} "
         f"ms, decision-host wall={_num(chunk['selector_decision_host_wall_ms'])} ms다. "
         "Component 합은 overlap 때문에 TTFT decomposition이 아니다."), "",
        "### 13. Probe I/O", "",
        (f"QA-Chunk25 probe={_num(chunk['probe_io_mb'], 6)} MB/request, "
         f"probe_io={_num(chunk['probe_io_ms'])} ms/request다. Probe는 offline으로 "
         "숨기지 않고 total SSD/TTFT critical path에 포함했다."), "",
        "### 14. Selected-chunk I/O", "",
        (f"Selected K/V payload={_num(chunk['selected_chunk_payload_mb'], 6)} "
         f"MB/request, selected K/V raw pread="
         f"{_num(chunk['selected_chunk_io_ms'])} ms/request다. "
         f"Separator raw pread={_num(chunk['separator_read_ms'])} ms이며, "
         f"더 넓은 host chunk-I/O interval(두 K/V read, separator/setup·변환 포함)은 "
         f"{_num(chunk['chunk_io_ms'])} ms다."), "",
        "### 15. Total SSD MB", "",
        (f"QA-Chunk25 total={_num(chunk['actual_ssd_mb_per_cache_hit'], 6)} "
         f"MB/request이며 selected K/V + probe + separator sidecar를 모두 포함한다."), "",
        "### 16. FullLoad 대비 SSD ratio", "",
        (f"QA-Chunk25/FullLoad actual SSD ratio="
         f"{_pct(chunk['actual_ssd_ratio_vs_fullload'], 4)}다."), "",
        "### 17. Touched chunks", "",
        (f"QA-Chunk25={_pct(chunk['touched_chunk_fraction'], 4)}, "
         f"QA-Token25={_pct(token['touched_chunk_fraction'], 4)}, "
         f"Ours25={_pct(ours['touched_chunk_fraction'], 4)}다."), "",
        "### 18. Contiguous runs/layer", "",
        (f"QA-Chunk25는 {_num(chunk['contiguous_runs_per_layer'])} runs/layer, "
         f"mean run length {_num(chunk['mean_contiguous_run_length'])}, max "
         f"{_num(chunk['max_contiguous_run_length'])} chunks다. Ours25는 "
         f"{_num(ours['contiguous_runs_per_layer'])} run/layer의 prefix다."), "",
        "### 19. Preads/request", "",
        (f"QA-Chunk25={_num(chunk['ssd_preads_per_cache_hit'])}, "
         f"QA-Token25={_num(token['ssd_preads_per_cache_hit'])}, "
         f"Ours25={_num(ours['ssd_preads_per_cache_hit'])} preads/request다."), "",
        "### 20. Scatter/prefill breakdown", "",
        (f"QA-Chunk25 rater/projection/probe/scoring/aggregation/top-k/ID-D2H/"
         f"planning/host-chunk-I/O/scatter/prefill은 각각 "
         f"{_num(chunk['rater_selection_ms'])}/{_num(chunk['query_projection_ms'])}/"
         f"{_num(chunk['probe_io_ms'])}/{_num(chunk['query_scoring_ms'])}/"
         f"{_num(chunk['chunk_aggregation_ms'])}/{_num(chunk['topk_chunk_ms'])}/"
         f"{_num(chunk['selected_id_d2h_ms'])}/{_num(chunk['chunk_planning_ms'])}/"
         f"{_num(chunk['chunk_io_ms'])}/{_num(chunk['scatter_ms'])}/"
         f"{_num(chunk['prefill_ms'])} ms다. Host chunk-I/O 중 계측된 "
         f"selected-K/V/separator raw pread는 "
         f"{_num(chunk['selected_chunk_io_ms'])}/"
         f"{_num(chunk['separator_read_ms'])} ms다."), "",
        "### 21. QA-Token25 대비 변화", "",
        (f"Accuracy 변화={_signed(comparison['chunk_minus_token_accuracy_pp'])} pp, "
         f"SSD 감소={comparison['chunk_ssd_reduction_vs_token_percent']:.2f}%, "
         f"TTFT 변화={_signed(comparison['chunk_minus_token_ttft_ms'])} ms다."), "",
        "### 22. Ours25 대비 quality gap", "",
        f"QA-Chunk25−Ours25={_signed(comparison['chunk_minus_ours_accuracy_pp'])} pp다.", "",
        "### 23. Ours25 대비 TTFT gap", "",
        (f"QA-Chunk25−Ours25={_signed(comparison['chunk_minus_ours_ttft_ms'])} "
         f"ms, ratio={comparison['chunk_over_ours_ttft_ratio']:.4f}×다."), "",
        "### 24. Tests", "",
        f"- CPU test result (verbatim CLI evidence): `{test_result}`",
        (f"- Runner validation: {sum(value is True for value in runner_checks.values())}/"
         f"{len(runner_checks)} PASS."),
        (f"- Reporter independent gates: "
         f"{sum(value is True for value in checks.values())}/{len(checks)} PASS."), "",
        "### 25. Artifact protection", "",
        (f"Prior-artifact verification passed={general_protection.get('passed')}; "
         f"before/after manifest={general_protection.get('before_manifest_sha256')} / "
         f"{general_protection.get('after_manifest_sha256')}. "
         f"Source-store read-only verification passed="
         f"{source_protection.get('read_only_source_reuse_validated')}; "
         f"fingerprint={source_protection.get('after_store_fingerprint_sha256')}. "
         "Runner/result exports are byte-identical and existing files were not clobbered."), "",
        "### 26. Limitations", "",
    ])
    lines.extend(f"- {item}" for item in limitations)
    lines.extend([
        "- 이 결과는 고정 GQA 40-image/240-question pilot이며 independent questions를 "
        "cache-hit turns처럼 평가했다. 장기 conversational history로 일반화하지 않는다.",
        "- Buffered pread와 `POSIX_FADV_DONTNEED`는 SSD controller cache까지 제거하지 못한다.",
        "- Mean aggregation과 25% budget은 결과를 본 뒤 튜닝하지 않았다.",
    ])
    if failed_checks:
        lines.extend(["", "## Failed validation gates", ""])
        lines.extend(f"- `{name}`" for name in failed_checks)
    lines.extend([
        "", "## Provenance", "",
        f"- Run directory: `{run_dir}`",
        f"- Results directory: `{results_dir}`",
        f"- GQA index SHA256: `{config['index_sha256']}`",
        f"- Workload SHA256: `{config['selected_workload_sha256']}`",
        "", "### Runner evidence SHA256", "",
        "| File | SHA256 |", "|---|---|",
    ])
    lines.extend(f"| `{name}` | `{digest}` |"
                 for name, digest in runner_hashes.items())
    lines.extend(["", "### Report-time source SHA256", "",
                  "| File | SHA256 |", "|---|---|"])
    for relative in SOURCE_FILES:
        lines.append(f"| `{relative}` | `{_sha256_file(ROOT / relative)}` |")
    lines.extend([
        "", "## Reproduction", "",
        "GPU pilot는 detached launcher로 실행하고, 완료 후 protection verify와 "
        "이 CPU-only reporter를 실행한다.", "",
        "```bash",
        "bash scripts/run_qa_chunk25_gqa_background.sh --help",
        ("python scripts/53_report_query_aware_chunk_baseline.py "
         f"--run-dir {run_dir} --results-dir {results_dir} "
         "--test-result '<observed test summary>'"),
        "```", "",
        f"QA-CHUNK25 BASELINE VALIDATED: {verdict}",
    ])
    return "\n".join(lines) + "\n"


def _build_readme(
    run_dir: Path, config: Mapping[str, Any], summary: Mapping[str, Any],
    report_validation: Mapping[str, Any],
) -> str:
    methods = _mapping(summary["per_method"], "summary.per_method")
    verdict = "YES" if report_validation["passed"] else "NO"
    return "\n".join([
        "# QA-Chunk25 GQA pilot artifact", "",
        f"Final validation: **{verdict}**", "",
        ("Five methods were evaluated in the same frozen GQA 40-image/240-question "
         "run. Accuracy uses all turns; latency and I/O use cache-hit turns 2–6."), "",
        "| Method | Accuracy | TTFT (ms) | SSD MB/request | Touched chunks |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {DISPLAY_LABELS[key]} | {_pct(methods[key]['accuracy_all_turns'])} | "
            f"{_num(methods[key]['ttft_cache_hit_mean_ms'])} | "
            f"{_num(methods[key]['actual_ssd_mb_per_cache_hit'])} | "
            f"{_pct(methods[key].get('touched_chunk_fraction'))} |"
            for key in METHOD_KEYS
        ],
        "", "Artifacts:", "",
        "- `ANALYSIS.md`: numbered 1–26 report and A–F research answers.",
        "- `results_final.jsonl`: durable raw request evidence.",
        "- `summary.csv`: five-method headline results.",
        "- `selection_analysis.json`: exact chunk IDs, pairwise Jaccard, and QA-Token overlap.",
        "- `latency_breakdown.csv` / `io_breakdown.csv`: stage and storage metrics.",
        "- `validation.json`: runner validation gates.",
        "- `report_validation.json`: independent reporter/protection gates.",
        "", f"Run evidence: `{run_dir}`", "",
        f"Index SHA256: `{config['index_sha256']}`", "",
        f"Workload SHA256: `{config['selected_workload_sha256']}`", "",
        f"QA-CHUNK25 BASELINE VALIDATED: {verdict}", "",
    ])


def generate_report(
    run_dir: Path | str,
    results_dir: Path | str,
    *,
    test_result: str,
    protection_validation: Path | str | None = None,
    source_protection_validation: Path | str | None = None,
) -> dict[str, Any]:
    run_arg, results_arg = Path(run_dir), Path(results_dir)
    if run_arg.is_symlink() or results_arg.is_symlink():
        raise ReportValidationError("run/results arguments may not be symlinks")
    run, results = run_arg.resolve(), results_arg.resolve()
    if not run.is_dir() or not results.is_dir():
        raise ReportValidationError("run/results must be existing real directories")
    if run == results or run in results.parents or results in run.parents:
        raise ReportValidationError("run and results directories overlap")

    config = _read_json(run / "config.json")
    manifest = _read_json(run / "manifest.json")
    summary = _read_json(run / "summary.json")
    selection = _read_json(run / "selection_analysis.json")
    runner_validation = _read_json(run / "validation.json")
    rows = _read_jsonl(run / "results_final.jsonl")
    completed_path = run / "COMPLETED"
    completed = _read_json(completed_path) if completed_path.is_file() else None
    if completed is not None:
        completed = dict(completed)
        completed["_validation_hash_matches"] = (
            completed.get("validation_sha256")
            == _sha256_file(run / "validation.json"))

    if summary.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ReportValidationError("summary schema mismatch")
    if summary.get("config") != config:
        raise ReportValidationError("summary embedded config differs from config.json")
    if Path(str(config.get("run_dir"))).resolve() != run:
        raise ReportValidationError("config run_dir mismatch")
    if Path(str(config.get("results_dir"))).resolve() != results:
        raise ReportValidationError("config results_dir mismatch")
    if manifest.get("run_id") != config.get("run_id"):
        raise ReportValidationError("manifest/config run_id mismatch")

    recomputed = _recompute_summaries(rows)
    recomputed_selection = _recompute_selection(rows)
    runner_hashes = _verify_runner_artifacts(
        run, results, summary, recomputed, selection, recomputed_selection)

    general_path = Path(protection_validation).resolve() if (
        protection_validation is not None) else _default_general_protection(results)
    source_path = Path(source_protection_validation).resolve() if (
        source_protection_validation is not None) else _default_source_protection(results)
    protection_error = source_error = None
    try:
        general_manifest, general_protection = _validate_general_protection_pair(
            run, results, general_path)
        protection_ok = True
    except ReportValidationError as error:
        general_manifest = {"manifest_sha256": None}
        general_protection = {"passed": False, "error": str(error)}
        protection_ok, protection_error = False, str(error)
    try:
        source_manifest, source_protection = _validate_source_protection_pair(
            run, source_path)
        source_ok = True
    except ReportValidationError as error:
        source_manifest = {"manifest_sha256": None}
        source_protection = {
            "passed": False, "read_only_source_reuse_validated": False,
            "error": str(error),
        }
        source_ok, source_error = False, str(error)

    tests_ok = bool(test_result.strip()) and bool(re.search(
        r"\b(?:passed|ok)\b", test_result, flags=re.IGNORECASE)) \
        and not bool(re.search(r"\b(?:failed|error)\b", test_result,
                               flags=re.IGNORECASE))
    checks = _independent_checks(
        config, manifest, rows, selection, runner_validation, completed,
        protection_ok, source_ok, tests_ok)
    report_validation = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "passed": all(checks.values()),
        "terminal_verdict": (
            "QA-CHUNK25 BASELINE VALIDATED: YES" if all(checks.values())
            else "QA-CHUNK25 BASELINE VALIDATED: NO"),
        "run_dir": str(run), "results_dir": str(results),
        "raw_sha256": runner_hashes["results_final.jsonl"],
        "raw_rows": len(rows), "checks": checks,
        "failed_checks": [key for key, value in checks.items()
                          if value is not True],
        "runner_validation_passed": runner_validation.get("passed") is True,
        "protection_validation": str(general_path),
        "source_protection_validation": str(source_path),
        "protection_manifest_sha256": general_manifest.get("manifest_sha256"),
        "source_protection_manifest_sha256": source_manifest.get(
            "manifest_sha256"),
        "protection_error": protection_error,
        "source_protection_error": source_error,
        "test_result": test_result,
    }
    analysis = _build_analysis(
        run, results, config, summary, selection, runner_validation,
        report_validation, runner_hashes, general_protection,
        source_protection, test_result)
    readme = _build_readme(run, config, summary, report_validation)
    terminal = report_validation["terminal_verdict"]
    if not analysis.rstrip().endswith(terminal):
        raise AssertionError("terminal verdict invariant failed")

    payloads = {
        "report_validation.json": _json_bytes(report_validation),
        "README.md": readme.encode("utf-8"),
        "ANALYSIS.md": analysis.encode("utf-8"),
    }
    for name, payload in payloads.items():
        _preflight_bytes(results / name, payload)
    statuses = {name: _publish_bytes(results / name, payload)
                for name, payload in payloads.items()}
    return {
        "passed": report_validation["passed"], "terminal_verdict": terminal,
        "analysis": str(results / "ANALYSIS.md"),
        "readme": str(results / "README.md"),
        "report_validation": str(results / "report_validation.json"),
        "publish_status": statuses,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--test-result", required=True,
        help="verbatim observed CPU test summary; never inferred",
    )
    parser.add_argument("--protection-validation", type=Path)
    parser.add_argument("--source-protection-validation", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = generate_report(
            args.run_dir, args.results_dir, test_result=args.test_result,
            protection_validation=args.protection_validation,
            source_protection_validation=args.source_protection_validation)
    except ReportValidationError as error:
        print(f"QA-Chunk25 report refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

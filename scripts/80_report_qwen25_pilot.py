#!/usr/bin/env python3
"""Audit and report one completed Qwen2.5-VL three-arm pilot.

Reports only measured values present in raw.jsonl and persistence.jsonl.
Incomplete runs, stale histories, duplicated requests, and missing method
coverage fail rather than yielding a partial performance comparison.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "qwen25-image-only-pilot-v1"
REPORT_SCHEMA = "qwen25-image-only-report-v1"
METHODS = ("recompute", "fullload", "ours25")
STRUCTURAL_GATES = (
    "cpu_score_reference", "runtime_load", "source_capture", "geometry",
    "gpu_score_reference", "query_independence", "persistence", "roundtrip",
    "repacked_full100", "capture", "io", "request_isolation", "history",
)
METRIC_KEYS = (
    "visual_read_bytes", "structural_read_bytes", "metadata_read_bytes",
    "pread_calls", "read_spans", "kept_tokens", "total_visual_tokens",
    "kept_ratio", "selected_chunks", "padding_rows_read",
    "visual_payload_read_ratio", "total_payload_read_ratio",
    "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes",
    "vision_calls", "online_query_score_calls", "selector_ms", "planning_ms",
    "raw_pread_ms", "h2d_ms", "assembly_ms", "prefill_ms",
    "store_load_inclusive_ms", "h2d_and_assembly_ms", "suffix_prefill_ms",
    "multimodal_prefill_ms", "prompt_preparation_ms", "decode_ms",
    "score_extra_ms", "capture_clone_ms", "score_peak_extra_gpu_bytes",
)
PERSISTENCE_KEYS = (
    "persistence_ms", "writer_persistence_ms", "capture_overhead_ms",
    "score_ms", "permutation_ms", "kv_materialize_ms", "repack_ms",
    "write_ms", "fsync_ms", "visual_file_bytes",
    "structural_file_bytes", "metadata_file_bytes",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True,
                  ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_text_new(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row: {path}:{line_number}")
            rows.append(value)
    return rows


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _metric(record: Mapping[str, Any], key: str) -> float | None:
    # A pixel request does not read this experiment's Visual-KV SSD store.
    # This says nothing about ordinary image/model filesystem traffic.
    if record.get("request_path") == "normal_pixels" and key in {
        "visual_read_bytes", "structural_read_bytes", "metadata_read_bytes",
        "pread_calls", "read_spans", "raw_pread_ms", "store_load_inclusive_ms",
    }:
        return 0.0
    sources = [record, record.get("result", {})]
    result = record.get("result", {})
    if isinstance(result, Mapping):
        if key == "raw_pread_ms":
            read_io = result.get("read_io", {})
            if isinstance(read_io, Mapping) and read_io.get("ms") is not None:
                value = float(read_io["ms"])
                if not math.isfinite(value) or value < 0:
                    raise ValueError("invalid raw pread time")
                return value
        sources.extend([result.get("io", {}), result.get("timing_ms", {}),
                        result.get("memory", {})])
    aliases = {
        "kept_ratio": ("actual_kept_ratio",),
        "store_load_inclusive_ms": ("store_load_inclusive", "ssd_read"),
        "h2d_and_assembly_ms": ("h2d_and_assembly",),
        "suffix_prefill_ms": ("suffix_prefill",),
        "multimodal_prefill_ms": ("multimodal_prefill",),
        "prompt_preparation_ms": ("prompt_preparation",),
        "decode_ms": ("decode",),
    }
    for name in (key, *aliases.get(key, ())):
      for source in sources:
        if isinstance(source, Mapping) and source.get(name) is not None:
            value = source[name]
            if isinstance(value, bool):
                raise ValueError(f"boolean metric: {key}")
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"invalid metric {key}: {value}")
            return number
    return None


def _persist_metric(record: Mapping[str, Any], key: str) -> float | None:
    payload = record.get("persistence", {})
    if not isinstance(payload, Mapping):
        raise ValueError("persistence row lacks object payload")
    sources = [payload, payload.get("timing_ms", {}),
               payload.get("metadata", {}), payload.get("bytes", {})]
    aliases = {"persistence_ms": ("total_ms", "total_persistence_ms"),
               "score_ms": ("capture_score_ms", "saliency_ms", "score_compute_ms"),
               "capture_overhead_ms": ("capture_clone_ms",),
               "repack_ms": ("kv_repack_ms",),
               "write_ms": ("ssd_write_ms",),
               "visual_file_bytes": ("bytes_visual_kv", "visual_bytes"),
               "structural_file_bytes": ("bytes_structural_kv", "structural_bytes"),
               "metadata_file_bytes": ("bytes_metadata_file", "metadata_bytes",)}
    for name in (key, *aliases.get(key, ())):
        for source in sources:
            if isinstance(source, Mapping) and source.get(name) is not None:
                number = float(source[name])
                if not math.isfinite(number) or number < 0:
                    raise ValueError(f"invalid persistence metric {name}")
                return number
    return None


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * probability
    lo = math.floor(position)
    hi = math.ceil(position)
    weight = position - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _scope_summary(rows: Sequence[Mapping[str, Any]], method: str,
                   scope: str) -> dict[str, Any]:
    selected = [row for row in rows if row["method"] == method and (
        scope == "all" or
        (scope == "hit" and int(row["turn_id"]) > 1) or
        (scope.startswith("turn_") and int(row["turn_id"]) == int(scope[5:]))
    )]
    ttft = [float(row["ttft_ms"]) for row in selected]
    e2e = [float(row["request_e2e_ms"]) for row in selected]
    output: dict[str, Any] = {
        "method": method, "scope": scope, "requests": len(selected),
        "accuracy": _mean([float(row["correct"]) for row in selected]),
        "ttft_mean_ms": _mean(ttft), "ttft_p50_ms": _quantile(ttft, 0.50),
        "ttft_p95_ms": _quantile(ttft, 0.95),
        "request_e2e_mean_ms": _mean(e2e),
        "request_e2e_p50_ms": _quantile(e2e, 0.50),
        "request_e2e_p95_ms": _quantile(e2e, 0.95),
        "generated_token_mean": _mean([
            float(row["generated_token_count"]) for row in selected]),
        "truncated_count": sum(bool(row.get("truncated")) for row in selected),
    }
    for key in METRIC_KEYS:
        values = [_metric(row, key) for row in selected]
        known = [value for value in values if value is not None]
        output[f"{key}_mean"] = _mean(known) if len(known) == len(values) else None
        output[f"{key}_measured"] = len(known)
    return output


def _bootstrap_pairs(rows: Sequence[Mapping[str, Any]], *, draws: int = 4000,
                     seed: int = 1234) -> dict[str, Any]:
    by_key = {(row["image_id"], row["dialog_id"], row["turn_id"], row["method"]): row
              for row in rows if int(row["turn_id"]) > 1}
    image_ids = sorted({str(row["image_id"]) for row in rows})
    rng = random.Random(seed)
    estimates: dict[str, Any] = {}
    for comparator in ("recompute", "fullload"):
        paired_by_image: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list))
        for (image_id, dialog_id, turn_id, method), left in by_key.items():
            if method != "ours25":
                continue
            right = by_key[(image_id, dialog_id, turn_id, comparator)]
            for key, left_value, right_value in (
                ("ttft_ms", float(left["ttft_ms"]), float(right["ttft_ms"])),
                ("accuracy", float(left["correct"]), float(right["correct"])),
            ):
                paired_by_image[image_id][key].append(left_value - right_value)
            for key in ("visual_read_bytes", "structural_read_bytes",
                        "metadata_read_bytes"):
                left_value = _metric(left, key)
                right_value = _metric(right, key)
                if left_value is not None and right_value is not None:
                    paired_by_image[image_id][key].append(left_value - right_value)
        method_report = {}
        for key in ("ttft_ms", "accuracy", "visual_read_bytes",
                    "structural_read_bytes", "metadata_read_bytes"):
            cluster_values = [statistics.fmean(paired_by_image[iid][key])
                              for iid in image_ids if paired_by_image[iid][key]]
            if len(cluster_values) != len(image_ids):
                method_report[key] = None
                continue
            samples = [statistics.fmean(rng.choices(cluster_values,
                                                   k=len(cluster_values)))
                       for _ in range(draws)]
            method_report[key] = {
                "ours_minus_comparator": statistics.fmean(cluster_values),
                "ci_95_low": _quantile(samples, 0.025),
                "ci_95_high": _quantile(samples, 0.975),
                "image_clusters": len(cluster_values), "bootstrap_draws": draws,
            }
        estimates[comparator] = method_report
    return estimates


def _score(dataset: str, prediction: str, gold: str) -> float:
    clean = lambda value: " ".join(
        word for word in re.sub(r"[^\w\s]", " ", str(value).lower()).split()
        if word not in {"a", "an", "the"})
    pred, answer = clean(prediction), clean(gold)
    if dataset == "gqa":
        return float(pred == answer or (answer and
                     pred.split()[:len(answer.split())] == answer.split()))
    return float(pred == answer)


def _t1_capture_overhead(rows: Sequence[Mapping[str, Any]], *,
                         draws: int = 4000, seed: int = 1234) -> dict[str, Any]:
    """Observational per-image T1 TTFT differences, with balanced rotation."""
    t1 = {(row["image_id"], row["method"]): row for row in rows
          if int(row["turn_id"]) == 1}
    image_ids = sorted({str(row["image_id"]) for row in rows})
    output = {}
    for method in ("fullload", "ours25"):
        differences = [float(t1[(iid, method)]["ttft_ms"])
                       - float(t1[(iid, "recompute")]["ttft_ms"])
                       for iid in image_ids]
        rng = random.Random(seed)
        samples = [statistics.fmean(rng.choices(differences,
                                               k=len(differences)))
                   for _ in range(draws)]
        score_ms = [_metric(t1[(iid, method)], "score_extra_ms")
                    for iid in image_ids]
        clone_ms = [_metric(t1[(iid, method)], "capture_clone_ms")
                    for iid in image_ids]
        output[method] = {
            "t1_ttft_minus_recompute_mean_ms": statistics.fmean(differences),
            "ci_95_low_ms": _quantile(samples, 0.025),
            "ci_95_high_ms": _quantile(samples, 0.975),
            "image_clusters": len(image_ids), "bootstrap_draws": draws,
            "score_extra_mean_ms": (_mean(score_ms)
                                    if None not in score_ms else None),
            "capture_clone_mean_ms": (_mean(clone_ms)
                                      if None not in clone_ms else None),
            "interpretation": (
                "observational noisy T1 TTFT difference under method rotation; "
                "post-response score/clone are charged to persistence, not TTFT"),
        }
    return output


def _audit(manifest: Mapping[str, Any], raw: list[dict[str, Any]],
           persistence: list[dict[str, Any]],
           gpu_inventory: list[dict[str, Any]]) -> dict[str, Any]:
    unsigned = dict(manifest)
    expected_manifest_hash = unsigned.pop("manifest_sha256", None)
    require(expected_manifest_hash == canonical_hash(unsigned),
            "manifest content hash mismatch")
    require(manifest.get("schema_version") == SCHEMA_VERSION,
            "wrong manifest schema")
    image_rows = manifest["images"]
    expected_count = sum(len(image["turns"]) for image in image_rows) * 3
    require(len(raw) == expected_count, "raw request count mismatch")
    require(len(persistence) == 2 * len(image_rows),
            "persistence store count mismatch")
    expected_gpu_events = 2 + 2 * len(image_rows)
    require(len(gpu_inventory) == expected_gpu_events,
            "GPU process inventory event count mismatch")
    expected_phases = [("pilot_start", None)]
    for image in image_rows:
        expected_phases.extend((("image_start", image["image_id"]),
                                ("image_end", image["image_id"])))
    expected_phases.append(("pilot_end", None))
    require([(row.get("phase"), row.get("image_id")) for row in gpu_inventory]
            == expected_phases, "GPU process inventory phase/order mismatch")
    require(all(row.get("foreign_process_count") == 0 and
                row.get("foreign_processes") == [] for row in gpu_inventory),
            "concurrent GPU compute process observed")
    require(len({row.get("request_id") for row in raw}) == len(raw),
            "duplicate request ID")
    row_map = {(row.get("image_id"), row.get("turn_id"), row.get("method")): row
               for row in raw}
    require(len(row_map) == len(raw), "duplicate image/turn/method row")
    persistence_map = {(row.get("image_id"), row.get("method")): row
                       for row in persistence}
    require(len(persistence_map) == len(persistence), "duplicate persistence row")
    for image in image_rows:
        iid = image["image_id"]
        expected_order = image["method_order"]
        require(sorted(expected_order) == sorted(METHODS),
                f"bad method rotation: {iid}")
        histories: dict[str, list[dict[str, str]]] = {key: [] for key in METHODS}
        for turn in image["turns"]:
            tid = turn["turn_id"]
            for method in expected_order:
                key = (iid, tid, method)
                require(key in row_map, f"missing request {key}")
                row = row_map[key]
                require(row.get("schema_version") == SCHEMA_VERSION,
                        f"wrong request schema {key}")
                require(row.get("question_id") == turn["question_id"] and
                        row.get("question") == turn["question"] and
                        row.get("gold") == turn["gold"],
                        f"question/gold differs from manifest: {key}")
                require(row.get("image_sha256") == image["image_sha256"],
                        f"image hash differs from manifest: {key}")
                require(row.get("method_order") == expected_order and
                        row.get("method_order_position") == expected_order.index(method),
                        f"method order mismatch: {key}")
                expected_history = histories[method] if manifest["dataset"] != "gqa" else []
                require(row.get("history") == expected_history and
                        row.get("history_sha256") == canonical_hash(expected_history),
                        f"history isolation failure: {key}")
                expected_path = ("normal_pixels" if tid == 1 or method == "recompute"
                                 else "ssd_cache_hit")
                require(row.get("request_path") == expected_path,
                        f"request path mismatch: {key}")
                require(float(row["ttft_ms"]) >= 0 and
                        float(row["request_e2e_ms"]) >= float(row["ttft_ms"]),
                        f"invalid timing: {key}")
                require(int(row["generated_token_count"]) >= 1,
                        f"no generated token: {key}")
                require(float(row["correct"]) == _score(
                    manifest["dataset"], row["prediction"], turn["gold"]),
                    f"scorer mismatch: {key}")
                histories[method].append({
                    "question_id": turn["question_id"],
                    "question": turn["question"],
                    "prediction": row["prediction"]})
        for method in ("fullload", "ours25"):
            require((iid, method) in persistence_map,
                    f"missing method-owned store: {iid}/{method}")
            persist = persistence_map[(iid, method)]
            require(persist.get("image_sha256") == image["image_sha256"] and
                    persist.get("source_turn_id") == 1,
                    f"store provenance mismatch: {iid}/{method}")
    return {"passed": True, "expected_requests": expected_count,
            "observed_requests": len(raw),
            "expected_stores": 2 * len(image_rows),
            "observed_stores": len(persistence),
            "history_isolation": True, "method_coverage": True,
            "gpu_inventory_events": len(gpu_inventory),
            "concurrent_gpu_compute_processes": 0,
            "manifest_content_hash": expected_manifest_hash}


def _persistence_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in ("fullload", "ours25"):
        selected = [row for row in rows if row["method"] == method]
        item: dict[str, Any] = {"method": method, "stores": len(selected)}
        for key in PERSISTENCE_KEYS:
            values = [_persist_metric(row, key) for row in selected]
            known = [value for value in values if value is not None]
            item[f"{key}_mean"] = _mean(known) if len(known) == len(values) else None
            item[f"{key}_measured"] = len(known)
        output.append(item)
    return output


def _session_summary(manifest: Mapping[str, Any],
                     raw: Sequence[Mapping[str, Any]],
                     persistence: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_image_method: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw:
        by_image_method[(row["image_id"], row["method"])].append(row)
    persisted = {(row["image_id"], row["method"]): row for row in persistence}
    output = []
    for method in METHODS:
        totals = []
        request_only = []
        activation_times = []
        activation_read_bytes = []
        activation_pread_calls = []
        metadata_resident_bytes = []
        incomplete = 0
        for image in manifest["images"]:
            iid = image["image_id"]
            request_ms = sum(float(row["request_e2e_ms"])
                             for row in by_image_method[(iid, method)])
            request_only.append(request_ms)
            if method == "recompute":
                totals.append(request_ms)
                activation_times.append(0.0)
                activation_read_bytes.append(0.0)
                activation_pread_calls.append(0.0)
                metadata_resident_bytes.append(0.0)
                continue
            persist_ms = _persist_metric(persisted[(iid, method)], "persistence_ms")
            hits = sorted((row for row in by_image_method[(iid, method)]
                           if int(row["turn_id"]) > 1),
                          key=lambda row: int(row["turn_id"]))
            first_condition = hits[0].get("conditioning", {}) if hits else {}
            activation_ms = (first_condition.get("activation_ms")
                             if isinstance(first_condition, Mapping) else None)
            activation_io = (first_condition.get("activation_io", {})
                             if isinstance(first_condition, Mapping) else {})
            if activation_ms is not None:
                activation_times.append(float(activation_ms))
            if isinstance(activation_io, Mapping) and activation_io.get("bytes") is not None:
                activation_read_bytes.append(float(activation_io["bytes"]))
                activation_pread_calls.append(float(activation_io["preads"]))
            if isinstance(first_condition, Mapping) and first_condition.get("metadata_resident_bytes") is not None:
                metadata_resident_bytes.append(float(first_condition["metadata_resident_bytes"]))
            if persist_ms is None or activation_ms is None:
                incomplete += 1
                continue
            totals.append(request_ms + persist_ms + float(activation_ms))
        output.append({"method": method, "images": len(manifest["images"]),
                       "request_only_mean_ms": _mean(request_only),
                       "aggregation_unit": (
                           "six_independent_requests_per_image"
                           if manifest["dataset"] == "gqa" else
                           "three_turn_generated_history_session"),
                       "session_e2e_mean_ms": (
                           _mean(totals) if not incomplete and
                           manifest["dataset"] != "gqa" else None),
                       "image_independent_requests_total_mean_ms": (
                           _mean(totals) if not incomplete and
                           manifest["dataset"] == "gqa" else None),
                       "missing_provisioning_or_activation_timing_images": incomplete,
                       "activation_mean_ms": (_mean(activation_times)
                                              if len(activation_times) == len(manifest["images"]) else None),
                       "activation_read_bytes_mean": (_mean(activation_read_bytes)
                                                      if len(activation_read_bytes) == len(manifest["images"]) else None),
                       "activation_pread_calls_mean": (_mean(activation_pread_calls)
                                                       if len(activation_pread_calls) == len(manifest["images"]) else None),
                       "metadata_resident_bytes_mean": (_mean(metadata_resident_bytes)
                                                        if len(metadata_resident_bytes) == len(manifest["images"]) else None),
                       "activation_timing_policy": (
                           "first per-store activation outside TTFT included"
                           if method != "recompute" else "not applicable")})
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no CSV rows for {path}")
    fields = list(rows[0])
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _fmt(value: Any, digits: int = 3) -> str:
    return "NOT MEASURED" if value is None else f"{float(value):.{digits}f}"


def _analysis(report: Mapping[str, Any]) -> str:
    by_scope = {(row["method"], row["scope"]): row for row in report["summary"]}
    heading = ("# Qwen2.5-VL 소규모 포팅 pilot 분석" if report["benchmark_validated"]
               else "# DIAGNOSTIC ONLY — Qwen2.5-VL 수치 불일치 조사")
    lines = [heading, "",
             f"- Workload: {report['workload_name']} ({report['dataset']})",
             f"- Images: {report['images']}; requests: {report['audit']['observed_requests']}",
             "- T1은 전 방법 정상 pixel inference입니다. GQA의 Q2 이후는 서로 독립적인 질문이며, MT의 이전 답변은 각 방법의 생성 결과만 사용합니다.",
             "- TTFT는 첫 output token materialization 및 CUDA 동기화까지입니다. 이미지 파일 읽기와 RGB decode, page-cache conditioning은 timer 밖입니다. ReComp의 processor 전처리와 vision/full visual prefill은 timer 안입니다.",
             "- MT-GQA는 reconstructed subset이며 공식 MetaCompress benchmark 결과가 아닙니다.", "",
             "## Validation contract", "",
             f"- validation_status: {report['validation_status']}",
             f"- benchmark_validated: {str(report['benchmark_validated']).lower()}",
             f"- run_mode: {report['run_mode']}",
             f"- FullLoad gate: {report['validation']['numerical_gate_statuses']['fullload']}",
             f"- Prefix25 gate: {report['validation']['numerical_gate_statuses']['prefix25']}",
             "- 두 수치 gate의 원본 per-question 결과와 출력 token 일치 여부는 validation_evidence.json에 그대로 보존했습니다.", "",
             "## Quality 및 cache-hit TTFT", "",
             "| Method | All accuracy | Hit accuracy | Hit TTFT mean ms | p50 ms | p95 ms | Hit E2E mean ms |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    if not report["benchmark_validated"]:
        lines.insert(2, "**진단용 결과입니다. GPU correctness validation이 FAIL이므로 유효한 benchmark 성능/품질 주장으로 사용할 수 없습니다.**")
        lines.insert(3, "")
    for method in METHODS:
        all_row, hit = by_scope[(method, "all")], by_scope[(method, "hit")]
        lines.append("| " + " | ".join((
            method, _fmt(all_row["accuracy"]), _fmt(hit["accuracy"]),
            _fmt(hit["ttft_mean_ms"]), _fmt(hit["ttft_p50_ms"]),
            _fmt(hit["ttft_p95_ms"]), _fmt(hit["request_e2e_mean_ms"]),
        )) + " |")
    lines += ["", "## Cache-hit I/O", "",
              "| Method | Visual bytes/request | Structural bytes/request | Metadata bytes/request | pread calls/request | read spans/request | kept tokens/request |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        hit = by_scope[(method, "hit")]
        lines.append("| " + " | ".join((
            method, _fmt(hit["visual_read_bytes_mean"], 0),
            _fmt(hit["structural_read_bytes_mean"], 0),
            _fmt(hit["metadata_read_bytes_mean"], 0),
            _fmt(hit["pread_calls_mean"]), _fmt(hit["read_spans_mean"]),
            _fmt(hit["kept_tokens_mean"]),
        )) + " |")
    aggregate_label = ("6 independent requests/image total mean ms"
                       if report["dataset"] == "gqa" else "3-turn session E2E mean ms")
    lines += ["", "## Persistence 및 session", "",
              "| Method | Persistence mean ms | Score mean ms | Repack mean ms | Write mean ms | Activation mean ms | " + aggregate_label + " |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    persist = {row["method"]: row for row in report["persistence_summary"]}
    sessions = {row["method"]: row for row in report["session_summary"]}
    for method in METHODS:
        p = persist.get(method, {})
        lines.append("| " + " | ".join((
            method, _fmt(p.get("persistence_ms_mean")),
            _fmt(p.get("score_ms_mean")), _fmt(p.get("repack_ms_mean")),
            _fmt(p.get("write_ms_mean")),
            _fmt(sessions[method]["activation_mean_ms"]),
            _fmt(sessions[method]["image_independent_requests_total_mean_ms"]
                 if report["dataset"] == "gqa" else
                 sessions[method]["session_e2e_mean_ms"]),
        )) + " |")
    lines += ["", "Activation I/O는 첫 store activation에서만 발생하며 cache-hit read bytes와 분리했습니다. 세부 값은 sessions.csv에 있습니다.", "",
              "## T1 capture timing", "",
              "T1 TTFT 차이는 image-paired 관측치이며 method rotation에도 잡음이 큽니다. score/clone은 응답 이후 persistence에 귀속됩니다.", ""]
    for method, item in report["t1_capture_overhead"].items():
        lines.append(
            f"- {method} − ReComp T1 TTFT: {_fmt(item['t1_ttft_minus_recompute_mean_ms'])} ms "
            f"(95% CI [{_fmt(item['ci_95_low_ms'])}, {_fmt(item['ci_95_high_ms'])}]); "
            f"score_extra {_fmt(item['score_extra_mean_ms'])} ms, "
            f"capture_clone {_fmt(item['capture_clone_mean_ms'])} ms")
    lines += ["", "## Paired image-cluster bootstrap", "",
              "Ours25 − comparator 차이입니다. 95% percentile CI가 0을 포함해도 두 방법의 동등성을 뜻하지 않습니다. 작은 pilot의 추정 불확실성이 큽니다.", ""]
    for comparator, metrics in report["paired_bootstrap"].items():
        lines.append(f"- vs {comparator}: " + ", ".join(
            f"{key}={_fmt(item['ours_minus_comparator'])} "
            f"[{_fmt(item['ci_95_low'])}, {_fmt(item['ci_95_high'])}]"
            for key, item in metrics.items() if item is not None))
    lines += ["", "성능 수치는 raw JSONL에 기록된 실제 측정값에서 계산했습니다. 빈 지표는 측정되지 않은 값입니다.", ""]
    return "\n".join(lines)


def _audit_validation_mode(manifest: Mapping[str, Any],
                           final_status: Mapping[str, Any], *,
                           diagnostic_after_numerical_fail: bool) -> dict[str, Any]:
    validation = manifest.get("validation")
    require(isinstance(validation, dict), "run manifest lacks bound GPU validation")
    path = Path(validation.get("path", ""))
    require(path.is_file() and sha256_file(path) == validation.get("sha256"),
            "bound GPU validation file is missing or changed")
    source = json.loads(path.read_text(encoding="utf-8"))
    gates = source.get("gates", {})
    require(source.get("schema_version") == "qwen25-gpu-correctness-v1" and
            isinstance(gates, dict), "wrong GPU validator schema")
    structural = validation.get("structural_gate_statuses", {})
    numerical = validation.get("numerical_gate_statuses", {})
    require(isinstance(structural, dict) and set(structural) == set(STRUCTURAL_GATES) and
            all(value == "PASS" for value in structural.values()),
            "structural validation status is incomplete or failed")
    require(isinstance(numerical, dict) and set(numerical) == {"fullload", "prefix25"},
            "numerical validation gates are missing")
    require(all(gates.get(name, {}).get("status") == status
                for name, status in {**structural, **numerical}.items()),
            "manifest gate statuses differ from bound validator")
    require(validation.get("numerical_gate_evidence") ==
            {name: gates[name] for name in ("fullload", "prefix25")},
            "numerical gate evidence differs from bound validator")
    require(all(manifest.get(key) == validation.get(key) == final_status.get(key)
                for key in ("validation_status", "benchmark_validated", "run_mode")),
            "manifest/status validation labels differ")
    require(final_status.get("validation") == validation and
            final_status.get("execution_status") == "PASS",
            "final status lacks matching validation or completed execution")
    if diagnostic_after_numerical_fail:
        require(validation.get("run_mode") == "diagnostic_numerical_mismatch" and
                validation.get("validation_status") == source.get("status") == "FAIL" and
                validation.get("benchmark_validated") is False and
                final_status.get("status") == "DIAGNOSTIC COMPLETE" and
                any(value == "FAIL" for value in numerical.values()) and
                all(value in {"PASS", "FAIL"} for value in numerical.values()),
                "run is not a completed numerical-only diagnostic")
    else:
        require(validation.get("run_mode") == "validated_benchmark" and
                validation.get("validation_status") == source.get("status") == "PASS" and
                validation.get("benchmark_validated") is True and
                final_status.get("status") == "PASS" and
                all(value == "PASS" for value in numerical.values()),
                "failed validation requires explicit diagnostic report option")
    return validation


def report_run(run_dir: Path, results_dir: Path, *,
               diagnostic_after_numerical_fail: bool = False) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    raw_path = run_dir / "raw.jsonl"
    persistence_path = run_dir / "persistence.jsonl"
    gpu_inventory_path = run_dir / "gpu_inventory.jsonl"
    runtime_path = run_dir / "runtime.json"
    final_status_path = run_dir / "final_status.json"
    status = json.loads(final_status_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = _audit_validation_mode(
        manifest, status,
        diagnostic_after_numerical_fail=diagnostic_after_numerical_fail)
    raw = read_jsonl(raw_path)
    persistence = read_jsonl(persistence_path)
    gpu_inventory = read_jsonl(gpu_inventory_path)
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    require(runtime.get("model_id") == "Qwen/Qwen2.5-VL-7B-Instruct" and
            runtime.get("weight_offload") is False and
            int(runtime.get("quantized_module_count", 0)) > 0 and
            int(runtime.get("kv_heads", 0)) > 0 and
            runtime.get("checkpoint_revision") == runtime.get("processor_revision")
            == runtime.get("tokenizer_revision"),
            "runtime fingerprint does not prove pinned quantized resident Qwen")
    audit = _audit(manifest, raw, persistence, gpu_inventory)
    max_turn = max(len(image["turns"]) for image in manifest["images"])
    scopes = ["all", "hit", *(f"turn_{tid}" for tid in range(1, max_turn + 1))]
    summary = [_scope_summary(raw, method, scope)
               for method in METHODS for scope in scopes]
    persistence_summary = _persistence_summary(persistence)
    session_summary = _session_summary(manifest, raw, persistence)
    for row in [*summary, *persistence_summary, *session_summary]:
        row.update({"validation_status": validation["validation_status"],
                    "benchmark_validated": validation["benchmark_validated"],
                    "run_mode": validation["run_mode"]})
    report = {"schema_version": REPORT_SCHEMA,
              "result_label": ("VALIDATED BENCHMARK" if validation["benchmark_validated"]
                               else "DIAGNOSTIC ONLY — VALIDATION FAIL"),
              "validation_status": validation["validation_status"],
              "benchmark_validated": validation["benchmark_validated"],
              "run_mode": validation["run_mode"],
              "validation": validation,
              "numerical_gate_evidence": validation["numerical_gate_evidence"],
              "dataset": manifest["dataset"],
              "workload_name": manifest["workload_name"],
              "images": len(manifest["images"]),
              "source_run_dir": str(run_dir.resolve()),
              "source_hashes": {"manifest.json": sha256_file(manifest_path),
                                "raw.jsonl": sha256_file(raw_path),
                                "persistence.jsonl": sha256_file(persistence_path),
                                "gpu_inventory.jsonl": sha256_file(gpu_inventory_path),
                                "runtime.json": sha256_file(runtime_path),
                                "final_status.json": sha256_file(final_status_path),
                                "validation.json": validation["sha256"]},
              "runtime_fingerprint": runtime,
              "manifest_sha256": manifest["manifest_sha256"],
              "audit": audit, "summary": summary,
              "persistence_summary": persistence_summary,
              "session_summary": session_summary,
              "paired_bootstrap": _bootstrap_pairs(raw),
              "t1_capture_overhead": _t1_capture_overhead(raw),
              "notes": [
                  "A cache-hit TTFT excludes one-time persistence; session E2E includes it when measured.",
                  "Raw pread time is the OS syscall interval; store_load_inclusive includes pread, CPU BF16 decode, and reorder. Components can overlap or be inclusive and must not be summed to derive total TTFT.",
                  "Page-cache conditioning is not an SSD controller cold guarantee or O_DIRECT measurement.",
                  "A confidence interval containing zero is not equivalence evidence.",
              ]}
    results_dir.mkdir(parents=True, exist_ok=False)
    write_json_new(results_dir / "summary.json", report)
    _write_csv(results_dir / "summary.csv", summary)
    _write_csv(results_dir / "persistence.csv", persistence_summary)
    _write_csv(results_dir / "sessions.csv", session_summary)
    write_text_new(results_dir / "ANALYSIS.md", _analysis(report))
    write_json_new(results_dir / "source_hashes.json", report["source_hashes"])
    write_json_new(results_dir / "validation_evidence.json", {
        "validation_status": validation["validation_status"],
        "benchmark_validated": validation["benchmark_validated"],
        "numerical_gate_evidence": validation["numerical_gate_evidence"],
        "validation_file_sha256": validation["sha256"]})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-after-numerical-fail", action="store_true",
                        help="required to report a completed numerical-only diagnostic run")
    args = parser.parse_args()
    report = report_run(
        args.run_dir.resolve(), args.results_dir.resolve(),
        diagnostic_after_numerical_fail=args.diagnostic_after_numerical_fail)
    print(json.dumps({"results_dir": str(args.results_dir.resolve()),
                      "dataset": report["dataset"],
                      "images": report["images"],
                      "requests": report["audit"]["observed_requests"],
                      "audit": "PASS",
                      "validation_status": report["validation_status"],
                      "benchmark_validated": report["benchmark_validated"],
                      "result_label": report["result_label"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

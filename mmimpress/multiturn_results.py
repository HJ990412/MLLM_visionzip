"""Turn-aware aggregation and validation for multi-turn run artifacts."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from mmimpress.cvpr25 import budget_chunk_count


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _num(rows, key):
    return np.asarray([float(r[key]) for r in rows
                       if r.get(key) is not None and math.isfinite(float(r[key]))])


def _mean(rows, key):
    x = _num(rows, key)
    return float(x.mean()) if x.size else None


def _pct(rows, key, p):
    x = _num(rows, key)
    return float(np.percentile(x, p)) if x.size else None


def _one_group(rows, aggregation, group_value):
    method = rows[0]["method"]
    out = {
        "aggregation": aggregation,
        "group": group_value,
        "method": method,
        "budget": rows[0].get("budget"),
        "n": len(rows),
        "quality_mean": _mean(rows, "quality_score"),
        "ttft_ms_mean": _mean(rows, "ttft_ms"),
        "ttft_ms_p50": _pct(rows, "ttft_ms", 50),
        "ttft_ms_p95": _pct(rows, "ttft_ms", 95),
        "decode_ms_mean": _mean(rows, "decode_ms"),
        "decode_ms_p50": _pct(rows, "decode_ms", 50),
        "decode_ms_p95": _pct(rows, "decode_ms", 95),
        "e2e_ms_mean": _mean(rows, "e2e_ms"),
        "e2e_ms_p50": _pct(rows, "e2e_ms", 50),
        "e2e_ms_p95": _pct(rows, "e2e_ms", 95),
        "ssd_read_mb_mean": (_mean(rows, "ssd_read_bytes") or 0.0) / 1e6,
        "ssd_read_mb_total": sum(float(r.get("ssd_read_bytes") or 0) for r in rows) / 1e6,
        "selector_ms_mean": _mean(rows, "selector_ms"),
        "scatter_ms_mean": _mean(rows, "scatter_ms"),
        "prefill_ms_mean": _mean(rows, "prefill_ms"),
        "selected_kv_ratio_mean": _mean(rows, "selected_kv_ratio"),
        "logical_visual_token_ratio_mean": _mean(rows, "selected_kv_ratio"),
        "ssd_payload_ratio_vs_full_visual_kv_mean": _mean(
            rows, "ssd_payload_ratio_vs_full_visual_kv"),
        "history_tokens_mean": _mean(rows, "history_tokens"),
        "active_visual_tokens_mean": _mean(rows, "active_visual_tokens"),
        "total_context_tokens_mean": _mean(rows, "total_context_tokens"),
        "full_visual_kv_gb_mean": (_mean(rows, "full_visual_kv_bytes") or 0.0) / 1e9,
        "selected_visual_kv_gb_mean": (
            _mean(rows, "selected_visual_kv_bytes") or 0.0) / 1e9,
        "gpu_memory_allocated_gb_mean": (
            _mean(rows, "gpu_memory_allocated") or 0.0) / 1e9,
        "gpu_peak_memory_allocated_gb_mean": (
            _mean(rows, "gpu_peak_memory_allocated") or 0.0) / 1e9,
        "process_rss_gb_mean": (_mean(rows, "process_rss_bytes") or 0.0) / 1e9,
    }
    return out


def aggregate(rows, aggregation, key_fn):
    groups = defaultdict(list)
    for r in rows:
        groups[(key_fn(r), r["method"])].append(r)
    out = [_one_group(rs, aggregation, g) for (g, _), rs in groups.items()]
    def group_sort(row):
        group = row["group"]
        return ((0, float(group)) if isinstance(group, (int, float)) else
                (1, str(group))) + (row["method"],)
    out.sort(key=group_sort)
    _add_comparisons(out)
    return out


def _add_comparisons(summary):
    by_group = defaultdict(dict)
    for row in summary:
        by_group[(row["aggregation"], str(row["group"]))][row["method"]] = row
    for methods in by_group.values():
        full, sparse = methods.get("FullLoad"), methods.get("SparseVLM 25%")
        for row in methods.values():
            if full and full["ssd_read_mb_mean"]:
                row["ssd_read_ratio_vs_fullload"] = \
                    row["ssd_read_mb_mean"] / full["ssd_read_mb_mean"]
            else:
                row["ssd_read_ratio_vs_fullload"] = None
            if (row["method"].startswith("Static+Diverse") and full and
                    full["ttft_ms_mean"] and full["ssd_read_mb_mean"]):
                row["ttft_reduction_vs_fullload_pct"] = 100.0 * (
                    1.0 - row["ttft_ms_mean"] / full["ttft_ms_mean"])
                row["ssd_reduction_vs_fullload_pct"] = 100.0 * (
                    1.0 - row["ssd_read_mb_mean"] / full["ssd_read_mb_mean"])
                row["quality_delta_vs_fullload"] = (
                    row["quality_mean"] - full["quality_mean"]
                    if row["quality_mean"] is not None and
                    full["quality_mean"] is not None else None)
            if (row["method"].startswith("Static+Diverse") and sparse and
                    sparse["ttft_ms_mean"] and sparse["ssd_read_mb_mean"]):
                row["ttft_reduction_vs_sparsevlm_pct"] = 100.0 * (
                    1.0 - row["ttft_ms_mean"] / sparse["ttft_ms_mean"])
                row["ssd_reduction_vs_sparsevlm_pct"] = 100.0 * (
                    1.0 - row["ssd_read_mb_mean"] / sparse["ssd_read_mb_mean"])


def _write_csv(path, rows):
    path = Path(path)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def validate_records(rows, config=None):
    failures = []
    by_request = defaultdict(list)
    by_dialog_method = defaultdict(list)
    exact_keys = defaultdict(int)
    for r in rows:
        exact_keys[(r["dialog_id"], int(r["turn_id"]), r["method"])] += 1
        by_request[(r["dialog_id"], int(r["turn_id"]))].append(r)
        by_dialog_method[(r["dialog_id"], r["method"])].append(r)
        if not float(r["ttft_ms"]) < float(r["e2e_ms"]):
            failures.append(f"TTFT>=E2E: {r['dialog_id']} t{r['turn_id']} {r['method']}")
        residual = abs(float(r["ttft_ms"]) + float(r["decode_ms"])
                       - float(r["e2e_ms"]))
        if residual > 2.0:
            failures.append(f"timing residual {residual:.3f}ms: "
                            f"{r['dialog_id']} t{r['turn_id']} {r['method']}")
        if r.get("image_kv_build_count_dialog") is None:
            failures.append(f"missing image KV build count: {r['dialog_id']}")
        elif int(r["image_kv_build_count_dialog"]) != 1:
            failures.append(f"image KV build count != 1: {r['dialog_id']}")
        if r.get("static_metadata_build_count_dialog") is None:
            failures.append(f"missing static metadata build count: {r['dialog_id']}")
        elif int(r["static_metadata_build_count_dialog"]) != 1:
            failures.append(f"static metadata build count != 1: {r['dialog_id']}")
        if r.get("method") == "FullLoad" and int(r.get("ssd_read_bytes", 0)) != \
                int(r.get("full_visual_kv_bytes", 0)):
            failures.append(f"FullLoad did not read full visual KV: "
                            f"{r['dialog_id']} t{r['turn_id']}")
        if str(r.get("method", "")).startswith("Static+Diverse"):
            if (r.get("n_chunks_total") is None or
                    r.get("n_chunks_selected") is None):
                failures.append(f"missing chunk counts: {r['dialog_id']} "
                                f"t{r['turn_id']} {r['method']}")
                continue
            budget = float(r["budget"])
            total = int(r["n_chunks_total"])
            expected = budget_chunk_count(total, budget)
            selected = float(r["n_chunks_selected"])
            if abs(selected - expected) > 1e-6:
                failures.append(f"chunk budget mismatch: {r['dialog_id']} "
                                f"t{r['turn_id']} {r['method']} "
                                f"selected={selected} expected={expected}")
    for key, request_rows in by_request.items():
        hashes = {r["text_history_sha256"] for r in request_rows}
        if len(hashes) != 1:
            failures.append(f"method histories differ: {key}")
        suffix = {r["suffix_ids_sha256"] for r in request_rows}
        if len(suffix) != 1:
            failures.append(f"method suffix token ids differ: {key}")
        active = {tuple(r.get("active_image_ids", [])) for r in request_rows}
        if len(active) != 1:
            failures.append(f"method active images differ: {key}")
        expected_methods = set((config or {}).get("methods", []))
        if expected_methods and {r["method"] for r in request_rows} != expected_methods:
            failures.append(f"request method set mismatch: {key}")
    duplicates = [k for k, n in exact_keys.items() if n != 1]
    if duplicates:
        failures.append(f"duplicate result keys: {duplicates[:10]}")
    expected_n = int((config or {}).get("n_turns", 0)) * len(
        (config or {}).get("methods", []))
    if expected_n and len(rows) != expected_n:
        failures.append(f"result row count mismatch: expected={expected_n} got={len(rows)}")
    expected_requests = {
        (str(dialog_id), int(turn_id))
        for dialog_id, turn_id in (config or {}).get("expected_request_keys", [])
    }
    if expected_requests and set(by_request) != expected_requests:
        missing = sorted(expected_requests - set(by_request))[:10]
        extra = sorted(set(by_request) - expected_requests)[:10]
        failures.append(f"request key mismatch: missing={missing} extra={extra}")
    for key, group in by_dialog_method.items():
        group.sort(key=lambda x: int(x["turn_id"]))
        hist = [int(r["history_tokens"]) for r in group]
        if any(b < a for a, b in zip(hist, hist[1:])):
            failures.append(f"history length decreased: {key} {hist}")
        active = [int(r.get("active_images", 0)) for r in group]
        if any(b < a for a, b in zip(active, active[1:])):
            failures.append(f"active images decreased: {key} {active}")

    expected_methods = set((config or {}).get("methods", []))
    if expected_methods:
        got = {r["method"] for r in rows}
        if got != expected_methods:
            failures.append(f"method set mismatch: expected={expected_methods} got={sorted(got)}")
    no_leak = ((config or {}).get("calibration_policy")
               in (None, "caption_only_pre_dialog", "none_correctness_gate_only"))
    if not no_leak:
        failures.append("unrecognized calibration policy")
    if int((config or {}).get("future_turn_calibration_count", 0)) != 0:
        failures.append("future-turn calibration count is nonzero")
    return {
        "passed": not failures,
        "n_records": len(rows),
        "n_dialogs": len({r["dialog_id"] for r in rows}),
        "n_turns": len(by_request),
        "checks": {
            "ttft_lt_e2e": not any("TTFT>=E2E" in x for x in failures),
            "e2e_approximately_ttft_plus_decode": not any("timing residual" in x for x in failures),
            "identical_text_history_across_methods": not any("histories differ" in x for x in failures),
            "identical_suffix_ids_across_methods": not any("suffix token" in x for x in failures),
            "history_tokens_monotonic": not any("history length" in x for x in failures),
            "active_images_monotonic": not any("active images decreased" in x for x in failures),
            "image_kv_built_once_per_dialog": not any("image KV build count" in x for x in failures),
            "static_metadata_built_once_per_dialog": not any("static metadata build count" in x for x in failures),
            "full_load_reads_full_visual_kv": not any("FullLoad did not" in x for x in failures),
            "selected_chunk_count_obeys_budget": not any("chunk budget mismatch" in x for x in failures),
            "selected_chunk_counts_present": not any("missing chunk counts" in x for x in failures),
            "identical_active_images_across_methods": not any("active images differ" in x for x in failures),
            "unique_dialog_turn_method_keys": not any("duplicate result keys" in x for x in failures),
            "complete_result_matrix": not any("result row count mismatch" in x for x in failures),
            "expected_request_keys_exact": not any("request key mismatch" in x for x in failures),
            "no_future_turn_calibration": no_leak and
                int((config or {}).get("future_turn_calibration_count", 0)) == 0 and
                not any("future-turn calibration count" in x for x in failures),
        },
        "failures": failures[:200],
    }


def build_artifacts(run_dir, config=None):
    run_dir = Path(run_dir)
    rows = read_jsonl(run_dir / "raw.jsonl")
    overall = aggregate(rows, "overall", lambda _: "all")
    by_turn = aggregate(rows, "turn", lambda r: int(r["turn_id"]))
    by_active = aggregate(rows, "active_images", lambda r: int(r["active_images"]))
    _write_csv(run_dir / "summary.csv", overall + by_active)
    _write_csv(run_dir / "per_turn.csv", by_turn)
    _write_csv(run_dir / "per_active_images.csv", by_active)

    per_dialog = []
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dialog_id"], r["method"])].append(r)
    for (dialog_id, _), rs in groups.items():
        row = _one_group(rs, "dialog", dialog_id)
        row["dialog_id"] = dialog_id
        per_dialog.append(row)
    per_dialog.sort(key=lambda x: (x["dialog_id"], x["method"]))
    _write_csv(run_dir / "per_dialog.csv", per_dialog)

    validation = validate_records(rows, config)
    with open(run_dir / "validation.json", "w") as f:
        json.dump(validation, f, indent=1)
    return overall, by_turn, by_active, per_dialog, validation

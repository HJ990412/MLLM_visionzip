#!/usr/bin/env python3
"""Supplement a completed frozen v2 pilot with per-image and paired audits.

This script reads the immutable raw rows, store metadata, and the report made by
85_report_qwen25_v2_pilot.py. It never reruns inference or changes that report.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
METHODS = ("recompute", "fullload", "ours25")
PAIRS = (("fullload", "recompute"), ("ours25", "recompute"),
         ("ours25", "fullload"))
DRAW_COUNT = 4000
SEED = 1234


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPORT = _load("qwen25_v2_report_for_audit", ROOT / "scripts/85_report_qwen25_v2_pilot.py")
V1 = REPORT.V1_REPORT


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _finite(value: Any, name: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"invalid numeric value: {name}")
    result = float(value)
    _require(math.isfinite(result) and result >= 0, f"nonfinite/negative: {name}")
    return result


def _mean(values: list[float]) -> float:
    _require(bool(values), "empty mean")
    return statistics.fmean(values)


def _paired_bootstrap(raw: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(str(row["image_id"]), str(row.get("dialog_id")),
               int(row["turn_id"]), row["method"]): row for row in raw}
    _require(len(by_key) == len(raw), "duplicate paired request key")
    image_ids = sorted({str(row["image_id"]) for row in raw})

    def value(row: dict[str, Any], metric: str) -> float:
        if metric == "accuracy":
            return _finite(row["correct"], metric)
        if metric == "total_ssd_read_bytes":
            return sum(_finite(V1._metric(row, key), key) for key in (
                "visual_read_bytes", "structural_read_bytes", "metadata_read_bytes"))
        if metric == "visual_read_bytes":
            return _finite(V1._metric(row, metric), metric)
        return _finite(row[metric], metric)

    output: dict[str, Any] = {}
    for scope in ("all", "hit"):
        scope_report: dict[str, Any] = {}
        for left_method, right_method in PAIRS:
            pair_label = f"{left_method}_minus_{right_method}"
            pair_report: dict[str, Any] = {}
            for metric in ("accuracy", "ttft_ms", "request_e2e_ms",
                           "visual_read_bytes", "total_ssd_read_bytes"):
                by_image: dict[str, list[float]] = defaultdict(list)
                paired_count = 0
                for (iid, did, turn_id, method), left in by_key.items():
                    if method != left_method or (scope == "hit" and turn_id == 1):
                        continue
                    right = by_key.get((iid, did, turn_id, right_method))
                    _require(right is not None,
                             f"missing paired {right_method}: {iid}/{did}/{turn_id}")
                    by_image[iid].append(value(left, metric) - value(right, metric))
                    paired_count += 1
                _require(sorted(by_image) == image_ids, f"missing image cluster: {pair_label}")
                cluster_values = [_mean(by_image[iid]) for iid in image_ids]
                rng = random.Random(SEED)
                draws = [_mean(rng.choices(cluster_values, k=len(cluster_values)))
                         for _ in range(DRAW_COUNT)]
                pair_report[metric] = {
                    "paired_requests": paired_count,
                    "image_clusters": len(image_ids),
                    "mean_difference": _mean(cluster_values),
                    "ci_95_low": V1._quantile(draws, 0.025),
                    "ci_95_high": V1._quantile(draws, 0.975),
                    "draws": DRAW_COUNT, "seed": SEED,
                    "cluster_unit": "image",
                }
            scope_report[pair_label] = pair_report
        output[scope] = scope_report
    return output


def _image_metrics(manifest: dict[str, Any], raw: list[dict[str, Any]],
                   persistence: list[dict[str, Any]],
                   summary: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        rows[(str(row["image_id"]), row["method"])].append(row)
    stores = {(str(row["image_id"]), row["method"]): row for row in persistence}
    _require(len(stores) == len(persistence), "duplicate persistence image/method")
    geometry: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    for image in manifest["images"]:
        iid = str(image["image_id"])
        for method in METHODS:
            method_rows = sorted(rows[(iid, method)], key=lambda row: int(row["turn_id"]))
            _require(len(method_rows) == len(image["turns"]),
                     f"missing request for image/method: {iid}/{method}")
            request_ms = sum(_finite(row["request_e2e_ms"], "request_e2e_ms")
                             for row in method_rows)
            hits = method_rows[1:]
            if method == "recompute":
                _require((iid, method) not in stores, "ReComp unexpectedly persisted KV")
                sessions.append({
                    "image_id": iid, "method": method, "request_count": len(method_rows),
                    "request_e2e_sum_ms": request_ms, "score_after_request_ms": 0.0,
                    "capture_clone_after_request_ms": 0.0, "writer_ms": 0.0,
                    "activation_outside_request_ms": 0.0,
                    "image_total_ms": request_ms,
                })
                continue

            persisted = stores[(iid, method)]
            payload = persisted["persistence"]
            writer = _finite(payload["writer_persistence_ms"], "writer_persistence_ms")
            score = _finite(payload["capture_score_ms"], "capture_score_ms")
            clone = _finite(payload["capture_clone_ms"], "capture_clone_ms")
            declared = _finite(payload["persistence_ms"], "persistence_ms")
            _require(math.isclose(declared, writer + score + clone, abs_tol=1e-6),
                     f"persistence components do not reproduce total: {iid}/{method}")
            _require(bool(hits) and isinstance(hits[0].get("conditioning"), dict),
                     f"missing first-hit conditioning: {iid}/{method}")
            first_cond = hits[0]["conditioning"]
            activation = _finite(first_cond["activation_ms"], "activation_ms")
            _require(isinstance(first_cond.get("activation_io"), dict),
                     f"missing activation I/O: {iid}/{method}")
            _require(all(_finite(hit["conditioning"]["activation_ms"],
                                 "later activation_ms") == 0.0
                         and hit["conditioning"].get("activation_io") is None
                         for hit in hits[1:]),
                     f"store activated more than once: {iid}/{method}")
            sessions.append({
                "image_id": iid, "method": method, "request_count": len(method_rows),
                "request_e2e_sum_ms": request_ms, "score_after_request_ms": score,
                "capture_clone_after_request_ms": clone, "writer_ms": writer,
                "activation_outside_request_ms": activation,
                "image_total_ms": request_ms + score + clone + writer + activation,
            })

            meta_path = Path(persisted["store_dir"]) / "meta.json"
            meta = _read(meta_path)
            _require(meta == payload["metadata"],
                     f"persisted metadata differs from store: {iid}/{method}")
            total_tokens = int(meta["visual_count"])
            chunk_size = int(meta["chunk_size"])
            full_chunks = int(meta["n_chunks"])
            _require(chunk_size == 64 and full_chunks == math.ceil(total_tokens / 64),
                     f"frozen chunk policy changed: {iid}/{method}")
            _require(len(meta["stored_to_original"]) == total_tokens and
                     sorted(meta["stored_to_original"]) == list(range(total_tokens)),
                     f"visual permutation invalid: {iid}/{method}")
            first = hits[0]["result"]
            kept = int(first["kept_tokens"])
            selected_chunks = int(first["selected_chunks"])
            selected_original = sorted(int(x) for x in meta["stored_to_original"][:kept])
            _require(kept == min(total_tokens, selected_chunks * chunk_size),
                     f"selected token/chunk count mismatch: {iid}/{method}")
            _require(method != "fullload" or
                     (kept == total_tokens and selected_chunks == full_chunks),
                     f"FullLoad did not read full Visual KV: {iid}")
            row_bytes_all_layers = (2 * int(meta["num_layers"]) * int(meta["row_bytes"]))
            valid_visual = kept * row_bytes_all_layers
            visual_read = selected_chunks * chunk_size * row_bytes_all_layers
            structural_read = int(meta["bytes_structural_kv"])
            _require(int(meta["bytes_visual_kv"]) ==
                     full_chunks * chunk_size * row_bytes_all_layers,
                     f"store lacks complete padded Visual KV: {iid}/{method}")
            hit_metrics = []
            for hit in hits:
                result = hit["result"]
                _require(int(result["kept_tokens"]) == kept and
                         int(result["total_visual_tokens"]) == total_tokens and
                         int(result["selected_chunks"]) == selected_chunks,
                         f"selected set geometry changed across turns: {iid}/{method}")
                _require(int(result["visual_read_bytes"]) == visual_read and
                         int(result["structural_read_bytes"]) == structural_read and
                         int(result["metadata_read_bytes"]) == 0,
                         f"SSD read bytes changed across turns: {iid}/{method}")
                _require(int(result["padding_rows_read"]) ==
                         selected_chunks * chunk_size - kept,
                         f"padding rows mismatch: {iid}/{method}")
                read_io = result["read_io"]
                _require(int(read_io["bytes"]) == visual_read + structural_read and
                         int(read_io["preads"]) == int(result["pread_calls"]) and
                         int(read_io["spans"]) == int(result["read_spans"]),
                         f"actual I/O counters mismatch: {iid}/{method}")
                hit_metrics.append((int(result["pread_calls"]), int(result["read_spans"])))
            geometry.append({
                "image_id": iid, "method": method, "hit_requests": len(hits),
                "nominal_budget_ratio": 1.0 if method == "fullload" else 0.25,
                "chunk_size": chunk_size, "visual_tokens": total_tokens,
                "full_chunks": full_chunks, "selected_chunks": selected_chunks,
                "kept_tokens": kept, "actual_token_retention": kept / total_tokens,
                "selected_chunk_ids_in_store_order": list(range(selected_chunks)),
                "selected_original_visual_ids_sha256": V1.canonical_hash(selected_original),
                "valid_visual_bytes_per_hit": valid_visual,
                "visual_read_bytes_per_hit": visual_read,
                "padding_visual_bytes_per_hit": visual_read - valid_visual,
                "structural_read_bytes_per_hit": structural_read,
                "metadata_read_bytes_per_hit": 0,
                "total_ssd_read_bytes_per_hit": visual_read + structural_read,
                "actual_preads_per_hit": [pair[0] for pair in hit_metrics],
                "actual_spans_per_hit": [pair[1] for pair in hit_metrics],
                "activation_read_bytes_once": int(first_cond["activation_io"]["bytes"]),
                "activation_preads_once": int(first_cond["activation_io"]["preads"]),
                "metadata_resident_bytes": int(first_cond["metadata_resident_bytes"]),
                "whole_visual_store_bytes": int(meta["bytes_visual_kv"]),
                "structural_store_bytes": structural_read,
                "metadata_store_bytes": int(meta["bytes_metadata_file"]),
                "store_meta_sha256": REPORT.V2_PILOT.sha256_file(meta_path),
            })

    old_sessions = {row["method"]: row for row in summary["session_summary"]}
    for method in METHODS:
        selected = [row["image_total_ms"] for row in sessions if row["method"] == method]
        name = ("image_independent_requests_total_mean_ms" if manifest["dataset"] == "gqa"
                else "session_e2e_mean_ms")
        _require(math.isclose(_mean(selected), _finite(old_sessions[method][name], name),
                              rel_tol=1e-9, abs_tol=1e-5),
                 f"existing session summary differs from disjoint components: {method}")
    return geometry, sessions


def audit(run_dir: Path, results_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve(strict=True)
    results_dir = results_dir.resolve(strict=True)
    manifest = _read(run_dir / "manifest.json")
    final_status = _read(run_dir / "final_status.json")
    binding = REPORT._audit_v2_binding(manifest, final_status)
    REPORT._audit_frozen_workload(manifest)
    summary = _read(results_dir / "summary.json")
    _require(summary.get("validation") == binding and
             summary.get("source_run_dir") == str(run_dir),
             "report does not match the completed v2 pilot")
    raw_path = run_dir / "raw.jsonl"
    persistence_path = run_dir / "persistence.jsonl"
    _require(summary["source_hashes"]["raw.jsonl"] ==
             REPORT.V2_PILOT.sha256_file(raw_path) and
             summary["source_hashes"]["persistence.jsonl"] ==
             REPORT.V2_PILOT.sha256_file(persistence_path),
             "raw data changed after the primary report")
    raw = V1.read_jsonl(raw_path)
    persistence = V1.read_jsonl(persistence_path)
    V1._audit(manifest, raw, persistence,
              V1.read_jsonl(run_dir / "gpu_inventory.jsonl"))
    geometry, sessions = _image_metrics(manifest, raw, persistence, summary)
    paired = _paired_bootstrap(raw)
    output = {
        "schema_version": "qwen25-v2-pilot-supplement-v1",
        "dataset": manifest["dataset"],
        "validation_sha256": binding["sha256"],
        "pilot_manifest_sha256": manifest["manifest_sha256"],
        "raw_sha256": summary["source_hashes"]["raw.jsonl"],
        "persistence_sha256": summary["source_hashes"]["persistence.jsonl"],
        "primary_summary_sha256": REPORT.V2_PILOT.sha256_file(results_dir / "summary.json"),
        "images": len(manifest["images"]), "requests": len(raw),
        "nominal_ours_budget_ratio": 0.25, "chunk_size": 64,
        "timing_derivation": (
            "T1 request E2E is measured before VisionScoreCapture.__exit__; its score "
            "computation and KV capture clone occur after request E2E. The store writer "
            "runs after that. First-hit activation runs outside request timing. Therefore "
            "per-image total is sum of measured request E2E + score + clone + writer "
            "+ one activation, with no overlapping subcomponents added."),
        "file_read_and_rgb_decode_in_request_timers": False,
        "os_page_cache_conditioning_in_request_timers": False,
        "ssd_controller_or_nand_cold_guaranteed": False,
        "per_image_geometry": geometry,
        "per_image_totals": sessions,
        "paired_image_cluster_bootstrap": paired,
    }
    V1.write_json_new(results_dir / "pilot_supplement.json", output)
    lines = [f"# v2 {manifest['dataset']} pilot supplement", "",
             "The primary report and frozen raw data remain unchanged. This audit verifies "
             "per-image SSD geometry and disjoint session timing against the primary summary.",
             "", "All image-cluster intervals use 4,000 draws and seed 1234. "
             "Intervals containing zero do not establish equivalence.", "",
             "| Scope | Left minus right | Metric | Mean difference | 95% low | 95% high |",
             "|---|---|---|---:|---:|---:|"]
    for scope, comparisons in paired.items():
        for label, metrics in comparisons.items():
            for metric, value in metrics.items():
                lines.append(f"| {scope} | {label} | {metric} | "
                             f"{value['mean_difference']:.6f} | "
                             f"{value['ci_95_low']:.6f} | {value['ci_95_high']:.6f} |")
    lines += ["", "Per-image visual token counts, full and selected chunk counts, "
              "valid/padded visual bytes, actual preads/spans, store sizes, and "
              "metadata activation are in `pilot_supplement.json`.", "",
              output["timing_derivation"], ""]
    V1.write_text_new(results_dir / "PILOT_SUPPLEMENT.md", "\n".join(lines))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run_dir, args.results_dir)
    print(json.dumps({"dataset": result["dataset"], "images": result["images"],
                      "requests": result["requests"], "status": "PASS"},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

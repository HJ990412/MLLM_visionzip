#!/usr/bin/env python3
"""Audit and summarize a v2-gated frozen Qwen GQA/MT pilot.

This reuses the original raw-row scorer, coverage audit, timing aggregation,
and image-cluster bootstrap without accepting the v1 validator contract.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V2_PILOT = _load_script("qwen25_v2_pilot", ROOT / "scripts/84_eval_qwen25_v2_pilot.py")
V1_REPORT = _load_script("qwen25_frozen_report_v1", ROOT / "scripts/80_report_qwen25_pilot.py")
METHODS = ("recompute", "fullload", "ours25")
NATIVE_BF16_VISUAL_KV_BYTES_PER_TOKEN = 28 * 2 * 4 * 128 * 2


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _audit_v2_binding(manifest: Mapping[str, Any],
                      final_status: Mapping[str, Any]) -> dict[str, Any]:
    bound = manifest.get("validation")
    _require(isinstance(bound, dict), "pilot lacks a v2 validation binding")
    _require(bound.get("schema_version") == V2_PILOT.SCHEMA,
             "pilot was bound to another validator schema")
    path = Path(bound.get("path", ""))
    _require(path.is_file() and V2_PILOT.sha256_file(path) == bound.get("sha256"),
             "bound v2 validation was removed or changed")
    current = V2_PILOT.bind_v2_validation(path)
    _require(bound == current, "v2 validation binding differs from current evidence")
    _require(all(manifest.get(key) == final_status.get(key) == bound.get(key)
                 for key in ("validation_status", "benchmark_validated", "run_mode")),
             "manifest and final status disagree about v2 validation")
    _require(final_status.get("validation") == bound and
             final_status.get("status") == "PASS" and
             final_status.get("execution_status") == "PASS",
             "pilot did not finish all measured requests")
    _require(bound["validation_status"] == "PASS" and
             bound["gpu_system_correctness"] == "PASS" and
             bound["benchmark_validated"] is True and
             bound["gate_statuses"] == {name: "PASS" for name in V2_PILOT.GATES},
             "v2 system correctness is incomplete")
    return bound


def _audit_frozen_workload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    dataset = manifest.get("dataset")
    _require(dataset in ("gqa", "mt_gqa_reconstructed"),
             "pilot dataset is not frozen GQA or reconstructed MT")
    key = "gqa" if dataset == "gqa" else "mt"
    old, provenance = V2_PILOT.frozen_workload(key)
    _require(manifest.get("frozen_workload") == provenance,
             "frozen workload provenance changed")
    unbound = dict(manifest)
    for field in ("validation", "validation_status", "benchmark_validated",
                  "run_mode", "frozen_workload"):
        unbound.pop(field, None)
    unbound["manifest_sha256"] = old["manifest_sha256"]
    _require(unbound == old,
             "pilot image, question, gold, history, or method schedule differs from frozen original")
    return provenance


def _agreement(raw: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    selected = [row for row in raw if scope == "all" or int(row["turn_id"]) > 1]
    paired: dict[tuple[str, str | None, int], dict[str, dict[str, Any]]] = {}
    for row in selected:
        key = (str(row["image_id"]), row["dialog_id"], int(row["turn_id"]))
        paired.setdefault(key, {})[row["method"]] = row
    _require(all(set(group) == set(METHODS) for group in paired.values()),
             "agreement calculation lacks a paired method")
    output = []
    for left_method, right_method in (("recompute", "fullload"),
                                      ("recompute", "ours25"),
                                      ("fullload", "ours25")):
        left = [group[left_method] for group in paired.values()]
        right = [group[right_method] for group in paired.values()]
        first_same = sum(a["first_token_id"] == b["first_token_id"]
                         for a, b in zip(left, right))
        prediction_same = sum(a["prediction"] == b["prediction"]
                              for a, b in zip(left, right))
        generated_available = all("generated_token_ids" in a.get("result", {}) and
                                  "generated_token_ids" in b.get("result", {})
                                  for a, b in zip(left, right))
        generated_same = (sum(a["result"]["generated_token_ids"] ==
                              b["result"]["generated_token_ids"]
                              for a, b in zip(left, right))
                          if generated_available else None)
        n = len(left)
        output.append({
            "scope": scope, "left": left_method, "right": right_method,
            "paired_requests": n,
            "first_token_agreement_count": first_same,
            "first_token_agreement_rate": first_same / n if n else None,
            "prediction_agreement_count": prediction_same,
            "prediction_agreement_rate": prediction_same / n if n else None,
            "generated_sequence_agreement_count": generated_same,
            "generated_sequence_agreement_rate": (
                generated_same / n if n and generated_same is not None else None),
        })
    return output



def _raw_integrity(raw: list[dict[str, Any]],
                   persistence: list[dict[str, Any]]) -> dict[str, Any]:
    def nonfinite(value: Any) -> int:
        if isinstance(value, float):
            return int(not math.isfinite(value))
        if isinstance(value, list):
            return sum(nonfinite(item) for item in value)
        if isinstance(value, dict):
            return sum(nonfinite(item) for item in value.values())
        return 0

    error_rows = [row.get("request_id") for row in raw
                  if "error" in row or "error" in row.get("result", {})]
    duplicate_request_ids = len(raw) - len({row.get("request_id") for row in raw})
    nonfinite_count = nonfinite(raw) + nonfinite(persistence)
    logits_missing = 0
    logits_nonfinite = 0
    for row in raw:
        result = row.get("result", {})
        digest = result.get("first_logits_sha256")
        if not (result.get("first_logits_finite") is True and
                result.get("first_logits_dtype") == "float32" and
                result.get("first_logits_shape") == [152064] and
                isinstance(digest, str) and len(digest) == 64 and
                all(char in "0123456789abcdef" for char in digest)):
            logits_missing += 1
        value = result.get("first_logits_nonfinite_count")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            logits_nonfinite += value
        else:
            logits_missing += 1
    _require(not error_rows and not duplicate_request_ids and
             not nonfinite_count and not logits_missing and not logits_nonfinite,
             "raw pilot has request errors, duplicates, missing logits evidence, "
             "or nonfinite values")
    return {
        "request_error_count": len(error_rows),
        "duplicate_request_id_count": duplicate_request_ids,
        "nonfinite_value_count": nonfinite_count,
        "nonfinite_first_logit_count": logits_nonfinite,
        "missing_first_logit_evidence_count": logits_missing,
        "first_logit_vectors_verified": len(raw),
        "truncated_request_count": sum(bool(row.get("truncated")) for row in raw),
        "raw_request_count": len(raw),
        "persistence_store_count": len(persistence),
    }


def _storage_metrics(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {
        method: [row for row in raw if row["method"] == method and
                 int(row["turn_id"]) > 1]
        for method in METHODS
    }
    rows = []
    for method, hits in by_method.items():
        _require(len(hits) == 40 * (5 if raw[0]["dataset"] == "gqa" else 2),
                 f"incomplete {method} hit coverage")
        image_values: dict[str, tuple[int, int]] = {}
        for row in hits:
            result = row["result"]
            total = int(result.get("total_visual_tokens", 0))
            kept = int(result.get("kept_tokens", 0))
            if method == "recompute":
                continue
            _require(total > 0 and 0 < kept <= total,
                     f"bad visual retention geometry: {row['request_id']}")
            pair = (kept, total)
            prior = image_values.setdefault(str(row["image_id"]), pair)
            _require(prior == pair, f"visual retention changed within image: {row['image_id']}")
        if method != "recompute":
            _require(len(image_values) == 40, "missing image retention rows")
        visual = [float(V1_REPORT._metric(row, "visual_read_bytes") or 0) for row in hits]
        structural = [float(V1_REPORT._metric(row, "structural_read_bytes") or 0)
                      for row in hits]
        metadata = [float(V1_REPORT._metric(row, "metadata_read_bytes") or 0)
                    for row in hits]
        kept_values = [int(row["result"]["kept_tokens"]) for row in hits] if method != "recompute" else []
        valid_visual = ([kept * NATIVE_BF16_VISUAL_KV_BYTES_PER_TOKEN
                         for kept in kept_values] if method != "recompute"
                        else [0] * len(hits))
        _require(all(read >= valid for read, valid in zip(visual, valid_visual)),
                 f"visual read bytes less than valid BF16 KV: {method}")
        row = {
            "method": method, "scope": "hit", "requests": len(hits),
            "mean_image_token_retention": (
                sum(kept / total for kept, total in image_values.values()) /
                len(image_values) if image_values else None),
            "aggregate_token_retention": (
                sum(kept for kept, _ in image_values.values()) /
                sum(total for _, total in image_values.values())
                if image_values else None),
            "mean_kept_visual_tokens": (
                sum(kept for kept, _ in image_values.values()) /
                len(image_values) if image_values else None),
            "mean_total_visual_tokens": (
                sum(total for _, total in image_values.values()) /
                len(image_values) if image_values else None),
            "valid_visual_bytes_mean": sum(valid_visual) / len(hits),
            "visual_read_bytes_mean": sum(visual) / len(hits),
            "padding_visual_read_bytes_mean": (
                sum(read - valid for read, valid in zip(visual, valid_visual)) /
                len(hits)),
            "structural_read_bytes_mean": sum(structural) / len(hits),
            "metadata_read_bytes_mean": sum(metadata) / len(hits),
            "total_ssd_read_bytes_mean": (
                sum(v + s + m for v, s, m in zip(visual, structural, metadata)) /
                len(hits)),
        }
        rows.append(row)
    full = next(row for row in rows if row["method"] == "fullload")
    for row in rows:
        row["visual_read_ratio_vs_fullload"] = (
            row["visual_read_bytes_mean"] / full["visual_read_bytes_mean"])
        row["total_ssd_read_ratio_vs_fullload"] = (
            row["total_ssd_read_bytes_mean"] / full["total_ssd_read_bytes_mean"])
    return rows


def _fmt(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _analysis(report: Mapping[str, Any]) -> str:
    dataset = report["dataset"]
    lines = [
        f"# Qwen2.5-VL v2 validated {dataset} pilot",
        "",
        f"GPU system correctness: **{report['validation_status']}** "
        f"(G1–G15 all PASS). Raw coverage audit: **PASS**.",
        f"Frozen images: {report['images']}; requests: {report['audit']['observed_requests']}; "
        f"stores: {report['audit']['observed_stores']}.",
        f"Request errors: {report['raw_integrity']['request_error_count']}; "
        f"duplicate IDs: {report['raw_integrity']['duplicate_request_id_count']}; "
        f"nonfinite JSON values: {report['raw_integrity']['nonfinite_value_count']}; "
        f"nonfinite first logits: {report['raw_integrity']['nonfinite_first_logit_count']}; "
        f"verified logit vectors: {report['raw_integrity']['first_logit_vectors_verified']}; "
        f"truncated outputs: {report['raw_integrity']['truncated_request_count']}.",
        "",
        "The workload is the original frozen GQA 40×6 or reconstructed MT 40×3 schedule. "
        "GQA questions are independent; MT history uses each method's own generated answers.",
        "",
        "## Measured methods",
        "",
        "| Method | Scope | Requests | Accuracy | TTFT mean ms | TTFT p50 ms | TTFT p95 ms | "
        "Visual read B | Structural read B | Metadata read B | preads | kept visual tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary"]:
        if row["scope"] not in ("all", "hit"):
            continue
        fields = (
            row["method"], row["scope"], row["requests"],
            row["accuracy"], row["ttft_mean_ms"], row["ttft_p50_ms"],
            row["ttft_p95_ms"], row["visual_read_bytes_mean"],
            row["structural_read_bytes_mean"], row["metadata_read_bytes_mean"],
            row["pread_calls_mean"], row["kept_tokens_mean"],
        )
        lines.append("| " + " | ".join(_fmt(value) for value in fields) + " |")
    lines += ["", "## Paired output agreement", "",
              "| Scope | Left | Right | Pairs | First token | Prediction | Generated sequence |",
              "|---|---|---|---:|---:|---:|---:|"]
    for row in report["agreement"]:
        lines.append("| " + " | ".join((
            row["scope"], row["left"], row["right"], str(row["paired_requests"]),
            _fmt(row["first_token_agreement_rate"]),
            _fmt(row["prediction_agreement_rate"]),
            _fmt(row["generated_sequence_agreement_rate"]),
        )) + " |")
    lines += [
        "", "## SSD and visual retention (cache hits)", "",
        "MB below is MiB (1,048,576 bytes). Valid visual bytes count native BF16 K/V "
        "for retained tokens; read bytes include 64-token chunk padding.",
        "",
        "| Method | Mean image retention % | Aggregate token retention % | "
        "Valid visual MiB | Visual read MiB | Padding MiB | Structural MiB | "
        "Metadata MiB | Total SSD ratio vs FullLoad |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["storage_metrics"]:
        mean_retention = row["mean_image_token_retention"]
        aggregate_retention = row["aggregate_token_retention"]
        fields = (
            row["method"],
            None if mean_retention is None else 100 * mean_retention,
            None if aggregate_retention is None else 100 * aggregate_retention,
            row["valid_visual_bytes_mean"] / 1048576,
            row["visual_read_bytes_mean"] / 1048576,
            row["padding_visual_read_bytes_mean"] / 1048576,
            row["structural_read_bytes_mean"] / 1048576,
            row["metadata_read_bytes_mean"] / 1048576,
            row["total_ssd_read_ratio_vs_fullload"],
        )
        lines.append("| " + " | ".join(_fmt(value) for value in fields) + " |")
    persistence = {row["method"]: row for row in report["persistence_summary"]}
    sessions = {row["method"]: row for row in report["session_summary"]}
    lines += [
        "", "## One-time and session costs", "",
        "GQA total is six independent requests per image plus one-time persistence "
        "and activation. MT E2E is a three-turn generated-history session.",
        "",
        "| Method | Score ms | Repack ms | Write ms | fsync ms | "
        "Persistence ms | Activation ms | Per-image total ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        cost = persistence.get(method, {})
        session = sessions[method]
        total = (session["image_independent_requests_total_mean_ms"]
                 if dataset == "gqa" else session["session_e2e_mean_ms"])
        fields = (
            method, cost.get("score_ms_mean"), cost.get("repack_ms_mean"),
            cost.get("write_ms_mean"), cost.get("fsync_ms_mean"),
            cost.get("persistence_ms_mean"), session.get("activation_mean_ms"),
            total,
        )
        lines.append("| " + " | ".join(_fmt(value) for value in fields) + " |")
    lines += [
        "", "## Paired image-cluster intervals", "",
        "4,000 bootstrap draws with seed 1234 and image as the cluster unit.",
        "",
        "| Ours25 minus | Metric | Mean | 95% low | 95% high |",
        "|---|---|---:|---:|---:|",
    ]
    for comparator in ("recompute", "fullload"):
        for metric in ("ttft_ms", "accuracy", "visual_read_bytes",
                       "structural_read_bytes", "metadata_read_bytes"):
            estimate = report["paired_bootstrap"][comparator][metric]
            if estimate is None:
                continue
            lines.append("| " + " | ".join((
                comparator, metric,
                _fmt(estimate["ours_minus_comparator"]),
                _fmt(estimate["ci_95_low"]), _fmt(estimate["ci_95_high"]),
            )) + " |")
    lines += [
        "A confidence interval crossing zero does not establish equivalence.",
        "",
        f"Validation SHA256: `{report['validation']['sha256']}`. "
        f"Frozen workload content SHA256: `{report['frozen_workload']['content_sha256']}`.",
        "",
    ]
    return "\n".join(lines)


def report_run(run_dir: Path, results_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve(strict=True)
    manifest_path = run_dir / "manifest.json"
    final_status_path = run_dir / "final_status.json"
    manifest = _read_json(manifest_path)
    final_status = _read_json(final_status_path)
    validation = _audit_v2_binding(manifest, final_status)
    frozen = _audit_frozen_workload(manifest)
    raw_path = run_dir / "raw.jsonl"
    persistence_path = run_dir / "persistence.jsonl"
    inventory_path = run_dir / "gpu_inventory.jsonl"
    runtime_path = run_dir / "runtime.json"
    raw = V1_REPORT.read_jsonl(raw_path)
    persistence = V1_REPORT.read_jsonl(persistence_path)
    inventory = V1_REPORT.read_jsonl(inventory_path)
    raw_integrity = _raw_integrity(raw, persistence)
    storage_metrics = _storage_metrics(raw)
    runtime = _read_json(runtime_path)
    _require(runtime.get("model_id") == V2_PILOT.V1_PILOT.MODEL_ID and
             runtime.get("weight_offload") is False and
             int(runtime.get("quantized_module_count", 0)) > 0 and
             runtime.get("checkpoint_revision") == validation["model_revision"] and
             runtime.get("processor_revision") == validation["model_revision"] and
             runtime.get("tokenizer_revision") == validation["model_revision"],
             "runtime fingerprint is not the pinned quantized Qwen checkpoint")
    audit = V1_REPORT._audit(manifest, raw, persistence, inventory)
    expected = 720 if manifest["dataset"] == "gqa" else 360
    _require(audit["observed_requests"] == expected and
             audit["observed_stores"] == 80,
             "frozen full-pilot request or store coverage mismatch")
    max_turn = 6 if manifest["dataset"] == "gqa" else 3
    scopes = ("all", "hit", *(f"turn_{index}" for index in range(1, max_turn + 1)))
    summary = [V1_REPORT._scope_summary(raw, method, scope)
               for method in METHODS for scope in scopes]
    persistence_summary = V1_REPORT._persistence_summary(persistence)
    session_summary = V1_REPORT._session_summary(manifest, raw, persistence)
    for row in [*summary, *persistence_summary, *session_summary]:
        row.update({"validation_status": "PASS", "benchmark_validated": True,
                    "run_mode": "validated_benchmark_v2"})
    source_hashes = {name: V2_PILOT.sha256_file(path) for name, path in {
        "manifest.json": manifest_path,
        "raw.jsonl": raw_path,
        "persistence.jsonl": persistence_path,
        "gpu_inventory.jsonl": inventory_path,
        "runtime.json": runtime_path,
        "final_status.json": final_status_path,
        "scripts/84_eval_qwen25_v2_pilot.py": ROOT / "scripts/84_eval_qwen25_v2_pilot.py",
        "scripts/85_report_qwen25_v2_pilot.py": ROOT / "scripts/85_report_qwen25_v2_pilot.py",
    }.items()}
    source_hashes["validation.json"] = validation["sha256"]
    source_hashes["frozen_workload_manifest.json"] = frozen["file_sha256"]
    result: dict[str, Any] = {
        "schema_version": "qwen25-image-only-report-v2",
        "result_label": "VALIDATED BENCHMARK — V2 MATCHED-COMPUTATION CONTRACT",
        "validation_status": "PASS", "gpu_system_correctness": "PASS",
        "benchmark_validated": True, "run_mode": "validated_benchmark_v2",
        "validation": validation, "frozen_workload": frozen,
        "dataset": manifest["dataset"], "workload_name": manifest["workload_name"],
        "images": 40, "source_run_dir": str(run_dir),
        "source_hashes": source_hashes, "runtime_fingerprint": runtime,
        "manifest_sha256": manifest["manifest_sha256"], "audit": audit,
        "summary": summary, "agreement": _agreement(raw, "all") + _agreement(raw, "hit"),
        "raw_integrity": raw_integrity, "storage_metrics": storage_metrics,
        "persistence_summary": persistence_summary,
        "session_summary": session_summary,
        "bootstrap_config": {"draws": 4000, "seed": 1234,
                             "cluster_unit": "image"},
        "paired_bootstrap": V1_REPORT._bootstrap_pairs(raw, draws=4000, seed=1234),
        "t1_capture_overhead": V1_REPORT._t1_capture_overhead(
            raw, draws=4000, seed=1234),
        "notes": [
            "v1 numerical mismatches remain preserved as diagnostics in the old validation.",
            "The v2 correctness gate compares matched-shape in-memory and SSD serving paths.",
            "MT-GQA is a reconstructed subset and is not an official benchmark result.",
            "TTFT is recorded before first-logit CPU cloning; request E2E includes that diagnostic clone.",
        ],
    }
    results_dir = results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=False)
    V1_REPORT.write_json_new(results_dir / "summary.json", result)
    V1_REPORT._write_csv(results_dir / "summary.csv", summary)
    V1_REPORT._write_csv(results_dir / "persistence.csv", persistence_summary)
    V1_REPORT._write_csv(results_dir / "sessions.csv", session_summary)
    V1_REPORT._write_csv(results_dir / "agreement.csv", result["agreement"])
    V1_REPORT._write_csv(results_dir / "storage_metrics.csv", storage_metrics)
    V1_REPORT.write_text_new(results_dir / "ANALYSIS.md", _analysis(result))
    V1_REPORT.write_json_new(results_dir / "source_hashes.json", source_hashes)
    V1_REPORT.write_json_new(results_dir / "validation_evidence.json", {
        "validation_file_sha256": validation["sha256"],
        "fixed_validation_manifest_content_sha256":
            validation["fixed_validation_manifest_content_sha256"],
        "v2_gate_statuses": validation["gate_statuses"],
        "frozen_inputs_sha256": validation["frozen_inputs_sha256"],
        "frozen_workload_file_sha256": frozen["file_sha256"],
        "frozen_workload_content_sha256": frozen["content_sha256"],
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    result = report_run(args.run_dir, args.results_dir)
    print(json.dumps({
        "results_dir": str(args.results_dir.resolve()),
        "dataset": result["dataset"], "images": result["images"],
        "requests": result["audit"]["observed_requests"],
        "audit": "PASS", "gpu_system_correctness": "PASS",
        "benchmark_validated": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

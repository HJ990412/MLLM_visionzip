"""Publish the ImageOnly Prefix budget sweep with a same-run ReComp arm.

This analyzer is intentionally a thin extension of
``26_analyze_image_only_budget_sweep.py``.  The cache-path validation,
coverage, Pareto, sensitivity, operating-point, and reproducibility logic is
reused unchanged.  This file adds only the semantics that are different for
pixel recomputation:

* exactly one same-run ``recompute`` arm is required;
* ReComp must perform no selector, KV-store read, or cache scatter;
* accuracy and true TTFT are reported against both FullLoad and ReComp; and
* SSD and cache-policy Pareto membership remains scoped to comparable
  SSD-backed paths (FullLoad plus Prefix budgets), while a separate global
  true-TTFT frontier includes the same-run ReComp baseline.

Existing runs and results are read-only.  Publication is atomic and refuses
to replace the fixed output directory.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
BASE_PATH = ROOT / "scripts/26_analyze_image_only_budget_sweep.py"
_BASE_MODULE_NAME = "image_only_budget_sweep_analysis_for_recomp"
_SPEC = importlib.util.spec_from_file_location(_BASE_MODULE_NAME, BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load base analyzer: {BASE_PATH}")
BASE = importlib.util.module_from_spec(_SPEC)
sys.modules[_BASE_MODULE_NAME] = BASE
try:
    _SPEC.loader.exec_module(BASE)
except Exception:
    sys.modules.pop(_BASE_MODULE_NAME, None)
    raise


SweepError = BASE.SweepError
CORE = BASE.CORE
BUDGETS = BASE.BUDGETS
PREFIX_BASE = BASE.PREFIX_BASE
PREFIX_METHODS = tuple(BASE.method_for_budget(budget) for budget in BUDGETS)
EXPECTED_METHODS = ("fullload", "recompute") + PREFIX_METHODS
CACHE_METHODS = ("fullload",) + PREFIX_METHODS

DEFAULT_RUN = ROOT / "runs/image_only_repack_budget_sweep_with_recomp/main_20_50"
DEFAULT_OUTPUT = ROOT / "results/image_only_repack_budget_sweep_with_recomp"
DEFAULT_STORE = BASE.DEFAULT_STORE


def _option(tokens: list[str], name: str) -> str:
    positions = [i for i, value in enumerate(tokens) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(tokens):
        raise SweepError(f"run command must contain exactly one {name}")
    return tokens[positions[0] + 1]


def _resolved_argument(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def validate_command(command: str) -> None:
    """Fail closed on the fixed integrated-run command contract."""
    tokens = shlex.split(str(command))
    if "--no-recompute" in tokens:
        raise SweepError("integrated run disabled ReComp with --no-recompute")
    expected_values = {
        "--budgets": "0.20,0.25,0.30,0.35,0.40,0.45,0.50",
        "--selectors": PREFIX_BASE,
        "--prefix-layout": "visionzip_image_only",
        "--sep-policy": "sidecar",
        "--metric": "gqa",
        "--dataset": "gqa",
        "--limit": "40",
        "--questions": "6",
        "--skip": "4",
        "--expect-images": "40",
        "--expect-questions": "240",
    }
    mismatches = {}
    for flag, expected in expected_values.items():
        actual = _option(tokens, flag)
        if actual != expected:
            mismatches[flag] = {"expected": expected, "actual": actual}
    ratio_raw = _option(tokens, "--ratio")
    try:
        ratio = float(ratio_raw)
    except ValueError as exc:
        raise SweepError(f"invalid --ratio value: {ratio_raw!r}") from exc
    if ratio != 0.25:
        mismatches["--ratio"] = {
            "expected": 0.25, "actual": ratio_raw}
    expected_paths = {
        "--index": ROOT / "data/index.json",
        "--store": DEFAULT_STORE,
        "--run-dir": DEFAULT_RUN,
    }
    for flag, expected in expected_paths.items():
        actual = _resolved_argument(_option(tokens, flag))
        if actual != expected.resolve():
            mismatches[flag] = {
                "expected": str(expected.resolve()), "actual": str(actual)}
    if mismatches:
        raise SweepError(f"integrated run command mismatch: {mismatches}")


def load_frozen_inputs(run_dir: Path) -> tuple:
    run_dir = Path(run_dir).resolve()
    if run_dir != DEFAULT_RUN.resolve():
        raise SweepError(f"run directory must be exactly {DEFAULT_RUN}")
    ordered_keys, frozen = CORE._frozen_workload()
    try:
        run = CORE._load_run(
            run_dir, "budget_sweep_with_recomp", ordered_keys, frozen,
            new_schema=True, expected_layout="visionzip_image_only")
        prior = CORE._load_run(
            BASE.OLD_RUN, "prior_image_only", ordered_keys, frozen,
            new_schema=True, expected_layout="visionzip_image_only")
        profile = CORE._load_profile(
            BASE.BUILD_PROFILE, "visionzip_image_only",
            {image for image, _ in ordered_keys})
    except Exception as exc:
        raise SweepError(str(exc)) from exc
    if tuple(run["groups"]) != EXPECTED_METHODS:
        raise SweepError(
            f"method/order mismatch: {tuple(run['groups'])} != "
            f"{EXPECTED_METHODS}")
    if [float(value) for value in run["summary"].get("budgets", [])] != \
            list(BUDGETS):
        raise SweepError("run budget list differs from the frozen sweep")
    if run["summary"].get("ratio") != 0.25:
        raise SweepError("base retention ratio changed")
    validate_command(str(run["summary"].get("command", "")))
    return ordered_keys, frozen, run, prior, profile


def validate_recompute(rows: list[dict],
                       ordered_keys: list[tuple[str, str]]) -> dict:
    """Validate that ReComp really is pixel recomputation with no KV I/O."""
    raw_keys = [(str(row.get("image_id")), str(row.get("question_id")))
                for row in rows]
    expected_keys = set(ordered_keys)
    if (len(rows) != len(ordered_keys) or len(set(raw_keys)) != len(raw_keys)
            or set(raw_keys) != expected_keys):
        raise SweepError(
            "ReComp rows must contain the exact workload once: "
            f"rows={len(rows)}, unique={len(set(raw_keys))}, "
            f"expected={len(ordered_keys)}")
    by_key = BASE.rows_by_key(rows)
    failures = []
    max_e2e_residual = 0.0
    max_ttft_prefill_residual = 0.0
    layout_annotations = set()
    calibration_annotations = set()
    zero_fields = (
        "ssd_read_ms", "ssd_read_bytes", "ssd_read_chunks", "ssd_preads",
        "normal_kv_read_bytes", "separator_read_bytes",
        "normal_kv_preads", "separator_preads", "total_actual_pread_bytes",
        "mean_bytes_per_pread", "normal_mean_bytes_per_pread",
        "separator_mean_bytes_per_pread",
    )
    blank_fields = (
        "selector_ms", "scatter_ms", "prepare_ms", "n_chunks_selected",
        "n_chunks_total", "touched_chunk_fraction", "logical_kv_ratio",
        "normal_chunk_count_total", "static_score_calls",
        "query_score_calls", "diversity_calls",
    )
    for key in ordered_keys:
        row = by_key[key]
        context = f"recompute/{key[0]}/{key[1]}"
        if (row.get("method_key") != "recompute"
                or row.get("method") != "ReComp"
                or BASE.number(row.get("retention"), context + "/retention",
                        blank=True) is not None
                or row.get("retention_kind") != "none"
                or row.get("retrieval") != "recompute"):
            failures.append(context + ": provenance")
        for field in zero_fields:
            value = BASE.number(row.get(field), context + "/" + field,
                                blank=True)
            if value is None or not BASE.close(value, 0.0, atol=1e-9):
                failures.append(context + f": nonzero/missing {field}")
        for field in blank_fields:
            if BASE.number(row.get(field), context + "/" + field,
                           blank=True) is not None:
                failures.append(context + f": unexpected {field}")
        try:
            selection = CORE._json_cell(
                row.get("selected_chunk_ids_per_layer"), context + "/selection")
        except Exception as exc:
            raise SweepError(str(exc)) from exc
        if selection is not None:
            failures.append(context + ": selected chunks present")
        for field in ("selection_mode", "separator_policy",
                      "validated_prefix_layout",
                      "reordered_prefix_store_validated"):
            if row.get(field) not in (None, ""):
                failures.append(context + f": unexpected {field}")
        ttft = BASE.number(row.get("ttft_ms"), context + "/ttft")
        prefill = BASE.number(row.get("prefill_ms"), context + "/prefill")
        decode = BASE.number(row.get("decode_ms"), context + "/decode")
        e2e = BASE.number(row.get("e2e_latency_ms"), context + "/e2e")
        max_ttft_prefill_residual = max(
            max_ttft_prefill_residual, abs(ttft - prefill))
        max_e2e_residual = max(max_e2e_residual, abs(e2e - ttft - decode))
        if ttft <= 0 or prefill <= 0 or e2e <= 0:
            failures.append(context + ": non-positive timing")
        if not ttft < e2e:
            failures.append(context + ": TTFT !< E2E")
        if BASE.integer(row.get("first_token_id"), context + "/first",
                        blank=True) is None:
            failures.append(context + ": missing first token")
        generated = BASE.integer(row.get("generated_tokens"),
                                 context + "/generated")
        if not 1 <= generated <= 16:
            failures.append(context + ": generated-token cap")
        layout_annotations.add(str(row.get("physical_layout") or ""))
        calibration_annotations.add(
            str(row.get("calibration_questions") or ""))
    if failures:
        raise SweepError(f"ReComp validation failed: {failures[:20]}")
    if max_e2e_residual > 1.0 or max_ttft_prefill_residual > 1.0:
        raise SweepError(
            "ReComp latency algebra exceeds 1 ms: "
            f"e2e={max_e2e_residual}, prefill={max_ttft_prefill_residual}")
    return {
        "passed": True,
        "requests": len(rows),
        "execution": "pixel recomputation; no KV-store access",
        "timed_scope": (
            "after processor/H2D through vision tower, multimodal prefill, "
            "and first output token"),
        "raw_physical_layout_annotations": sorted(layout_annotations),
        "raw_calibration_annotations": sorted(calibration_annotations),
        "physical_layout_semantics": (
            "not applicable to ReComp; the raw CSV field is a run-global "
            "04_eval annotation"),
        "max_abs_e2e_minus_ttft_decode_ms": max_e2e_residual,
        "max_abs_ttft_minus_prefill_ms": max_ttft_prefill_residual,
        "ssd_read_bytes": 0,
        "ssd_preads": 0,
    }


def validate_run_semantics(run: dict, store: Path,
                           ordered_keys: list[tuple[str, str]]) -> dict:
    cache_run = cache_projection(run)
    cache = BASE.validate_run_semantics(cache_run, store, ordered_keys)
    recomp = validate_recompute(run["groups"]["recompute"], ordered_keys)
    return {"passed": True, "cache_paths": cache, "recompute": recomp}


def cache_projection(run: dict) -> dict:
    """Return an immutable-view projection accepted by the base analyzer."""
    projected = dict(run)
    projected["groups"] = {
        method: run["groups"][method] for method in CACHE_METHODS}
    return projected


def _summarize_recompute(run: dict,
                         ordered_keys: list[tuple[str, str]]) -> tuple[dict, dict]:
    images = [key[0] for key in ordered_keys]
    maps = {method: BASE.rows_by_key(rows)
            for method, rows in run["groups"].items()}
    rows = [maps["recompute"][key] for key in ordered_keys]
    scores = np.asarray([
        BASE.number(row["correct"], "recompute/correct") for row in rows])
    full_scores = np.asarray([
        BASE.number(maps["fullload"][key]["correct"], "fullload/correct")
        for key in ordered_keys])
    comparison = CORE._paired_bootstrap(
        scores, full_scores, images, n=BASE.BOOTSTRAP_RESAMPLES,
        seed=BASE.BOOTSTRAP_SEED)
    accuracy_ci = BASE.cluster_accuracy_ci(scores, images)
    result = {
        "method_key": "recompute",
        "method": "ReComp (pixel recomputation)",
        "budget": "",
        "budget_pct": "",
        "n_requests": len(rows),
        "n_images": len(set(images)),
        "accuracy": float(scores.mean()),
        "accuracy_pct": float(scores.mean() * 100),
        "accuracy_image_cluster_ci95_lo_pct": accuracy_ci[0],
        "accuracy_image_cluster_ci95_hi_pct": accuracy_ci[1],
        "accuracy_delta_vs_fullload_pp":
            float((scores.mean() - full_scores.mean()) * 100),
        "accuracy_delta_image_cluster_ci95_lo_pp":
            comparison["image_cluster_delta_ci95_pp"][0],
        "accuracy_delta_image_cluster_ci95_hi_pp":
            comparison["image_cluster_delta_ci95_pp"][1],
        "mcnemar_budget_only_correct": comparison["mcnemar"]["a_only"],
        "mcnemar_fullload_only_correct": comparison["mcnemar"]["b_only"],
        "mcnemar_p_exact_two_sided":
            comparison["mcnemar"]["p_exact_two_sided"],
    }
    for field, prefix in (
            ("ttft_ms", "ttft"), ("decode_ms", "decode"),
            ("e2e_latency_ms", "e2e"), ("selector_ms", "selector"),
            ("ssd_read_ms", "ssd_read"), ("scatter_ms", "scatter"),
            ("prepare_ms", "prepare"), ("prefill_ms", "prefill")):
        values = [BASE.number(row.get(field), field, blank=True) for row in rows]
        values = [value for value in values if value is not None]
        if values:
            mean, p50, p95 = CORE._percentiles(values)
            result[f"{prefix}_mean_ms"] = mean
            result[f"{prefix}_p50_ms"] = p50
            result[f"{prefix}_p95_ms"] = p95
        else:
            result[f"{prefix}_mean_ms"] = ""
            result[f"{prefix}_p50_ms"] = ""
            result[f"{prefix}_p95_ms"] = ""
    io_fields = (
        "normal_kv_read_bytes", "separator_read_bytes", "ssd_read_bytes",
        "normal_kv_preads", "separator_preads", "ssd_preads",
        "mean_bytes_per_pread", "ssd_read_chunks", "n_chunks_selected",
        "n_chunks_total", "touched_chunk_fraction", "logical_kv_ratio",
    )
    for field in io_fields:
        values = [BASE.number(row.get(field), field, blank=True) for row in rows]
        values = [value for value in values if value is not None]
        result[field + "_mean"] = float(np.mean(values)) if values else ""
        if field in ("n_chunks_selected", "n_chunks_total"):
            result[field + "_min"] = min(values) if values else ""
            result[field + "_max"] = max(values) if values else ""
    result["ssd_read_mb_mean"] = 0.0
    result["ssd_read_bytes_total"] = 0
    result["normal_kv_read_mb_mean"] = 0.0
    result["separator_read_mb_mean"] = 0.0
    return result, comparison


def summarize(run: dict, ordered_keys: list[tuple[str, str]]) -> tuple:
    cache_run = cache_projection(run)
    cache_summary, paired_full, maps = BASE.summarize(cache_run, ordered_keys)
    recomp, recomp_vs_full = _summarize_recompute(run, ordered_keys)
    paired_full = dict(paired_full)
    paired_full["recompute"] = recomp_vs_full
    maps = dict(maps)
    maps["recompute"] = BASE.rows_by_key(run["groups"]["recompute"])

    summary_rows = [recomp, *cache_summary]
    images = [key[0] for key in ordered_keys]
    recomp_scores = np.asarray([
        BASE.number(maps["recompute"][key]["correct"], "recompute/correct")
        for key in ordered_keys])
    full = next(row for row in summary_rows if row["method_key"] == "fullload")
    paired_recompute = {}
    for row in summary_rows:
        method = row["method_key"]
        scores = np.asarray([
            BASE.number(maps[method][key]["correct"], method + "/correct")
            for key in ordered_keys])
        comparison = CORE._paired_bootstrap(
            scores, recomp_scores, images, n=BASE.BOOTSTRAP_RESAMPLES,
            seed=BASE.BOOTSTRAP_SEED)
        paired_recompute[method] = comparison
        row.update({
            "accuracy_delta_vs_fullload_image_cluster_ci95_lo_pp":
                row["accuracy_delta_image_cluster_ci95_lo_pp"],
            "accuracy_delta_vs_fullload_image_cluster_ci95_hi_pp":
                row["accuracy_delta_image_cluster_ci95_hi_pp"],
            "mcnemar_method_only_correct_vs_fullload":
                row["mcnemar_budget_only_correct"],
            "accuracy_delta_vs_recompute_pp":
                float((scores.mean() - recomp_scores.mean()) * 100),
            "accuracy_delta_vs_recompute_image_cluster_ci95_lo_pp":
                comparison["image_cluster_delta_ci95_pp"][0],
            "accuracy_delta_vs_recompute_image_cluster_ci95_hi_pp":
                comparison["image_cluster_delta_ci95_pp"][1],
            "mcnemar_method_only_correct_vs_recompute":
                comparison["mcnemar"]["a_only"],
            "mcnemar_recompute_only_correct":
                comparison["mcnemar"]["b_only"],
            "mcnemar_p_exact_two_sided_vs_recompute":
                comparison["mcnemar"]["p_exact_two_sided"],
            "ttft_reduction_vs_recompute_pct": 100 * (
                1 - row["ttft_mean_ms"] / recomp["ttft_mean_ms"]),
            "ttft_speedup_vs_recompute":
                recomp["ttft_mean_ms"] / row["ttft_mean_ms"],
            "ttft_speedup_vs_fullload":
                full["ttft_mean_ms"] / row["ttft_mean_ms"],
        })
        # BASE.summarize computes these for cache paths.  ReComp receives the
        # same explicitly named FullLoad comparisons without pretending it is
        # an SSD selection budget.
        if method == "recompute":
            row.update({
                "ttft_reduction_vs_fullload_pct": 100 * (
                    1 - row["ttft_mean_ms"] / full["ttft_mean_ms"]),
                "ssd_reduction_vs_fullload_pct": "",
                "ssd_read_ratio_vs_fullload": "",
                "accuracy_loss_vs_fullload_pp":
                    full["accuracy_pct"] - row["accuracy_pct"],
            })
    return summary_rows, cache_summary, paired_full, paired_recompute, maps


def augment_sensitivity(maps: dict, ordered_keys: list[tuple[str, str]],
                        frozen: dict) -> tuple[list[dict], list[dict], list[dict]]:
    question_rows, image_rows, error_rows = BASE.sensitivity_rows(
        maps, ordered_keys, frozen)
    question_map = {(row["image_id"], row["question_id"]): row
                    for row in question_rows}
    for key in ordered_keys:
        out = question_map[key]
        recomp = maps["recompute"][key]
        recomp_correct = BASE.integer(recomp["correct"], "recompute/correct")
        out["recompute_prediction"] = recomp["prediction"]
        out["recompute_correct"] = recomp_correct
        out["fullload_prediction_equals_recompute"] = bool(
            out["fullload_prediction"] == recomp["prediction"])
        for budget in BUDGETS:
            tag = int(round(100 * budget))
            current = int(out[f"correct_{tag}"])
            out[f"budget_only_correct_vs_recompute_{tag}"] = bool(
                current and not recomp_correct)
            out[f"recompute_only_correct_vs_budget_{tag}"] = bool(
                recomp_correct and not current)

    image_map = {row["image_id"]: row for row in image_rows}
    for image_id, out in image_map.items():
        group = [row for row in question_rows if row["image_id"] == image_id]
        recomp_accuracy = float(np.mean([
            int(row["recompute_correct"]) for row in group]))
        out["recompute_accuracy"] = recomp_accuracy
        out["fullload_delta_vs_recompute_pp"] = (
            out["fullload_accuracy"] - recomp_accuracy) * 100
        for budget in BUDGETS:
            tag = int(round(100 * budget))
            out[f"delta_vs_recompute_pp_{tag}"] = (
                out[f"accuracy_{tag}"] - recomp_accuracy) * 100

    for out in error_rows:
        category = out["question_type"]
        group = (question_rows if category == "all" else [
            row for row in question_rows if row["question_type"] == category])
        tag = int(round(float(out["budget"]) * 100))
        current = np.asarray([int(row[f"correct_{tag}"]) for row in group])
        recomp = np.asarray([int(row["recompute_correct"]) for row in group])
        out.update({
            "recompute_accuracy": float(recomp.mean()),
            "delta_vs_recompute_pp":
                float((current.mean() - recomp.mean()) * 100),
            "both_correct_vs_recompute":
                int(((current == 1) & (recomp == 1)).sum()),
            "budget_only_correct_vs_recompute":
                int(((current == 1) & (recomp == 0)).sum()),
            "recompute_only_correct":
                int(((current == 0) & (recomp == 1)).sum()),
            "neither_correct_vs_recompute":
                int(((current == 0) & (recomp == 0)).sum()),
        })
    return question_rows, image_rows, error_rows


def baseline_rows(summary_rows: list[dict]) -> list[dict]:
    fields = (
        "method_key", "method", "budget", "budget_pct", "n_requests",
        "accuracy", "accuracy_pct", "accuracy_delta_vs_fullload_pp",
        "accuracy_delta_vs_fullload_image_cluster_ci95_lo_pp",
        "accuracy_delta_vs_fullload_image_cluster_ci95_hi_pp",
        "mcnemar_method_only_correct_vs_fullload",
        "mcnemar_fullload_only_correct", "mcnemar_p_exact_two_sided",
        "accuracy_delta_vs_recompute_pp",
        "accuracy_delta_vs_recompute_image_cluster_ci95_lo_pp",
        "accuracy_delta_vs_recompute_image_cluster_ci95_hi_pp",
        "mcnemar_method_only_correct_vs_recompute",
        "mcnemar_recompute_only_correct",
        "mcnemar_p_exact_two_sided_vs_recompute",
        "ttft_mean_ms", "ttft_p50_ms", "ttft_p95_ms",
        "ttft_reduction_vs_fullload_pct",
        "ttft_reduction_vs_recompute_pct", "ssd_read_mb_mean",
        "ssd_read_bytes_mean", "ssd_preads_mean",
    )
    return [{field: row.get(field, "") for field in fields}
            for row in summary_rows]


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise SweepError(f"base README template changed around {old[:40]!r}")
    return text.replace(old, new, 1)


def make_readme(summary_rows: list[dict], cache_summary: list[dict],
                tradeoff: list[dict], coverage: list[dict],
                pareto_ssd: list[dict], pareto_ttft: list[dict],
                pareto_ttft_all: list[dict],
                operating: dict, correlations: dict, validation: dict,
                question_rows: list[dict], run_command: str) -> str:
    text = BASE.make_readme(
        cache_summary, tradeoff, coverage, pareto_ssd, pareto_ttft,
        operating, correlations, validation, question_rows, run_command)
    text = _replace_once(
        text, "# ImageOnly-Repack sequential-Prefix budget sweep",
        "# ImageOnly-Repack sequential-Prefix budget sweep with ReComp")
    text = _replace_once(
        text,
        "All arms ran in one process with the\n"
        "same model, prompt, decoding, physical SSD store, and schema-v2 true-TTFT\n"
        "contract.",
        "All arms ran in one process with the same model, prompt, decoding, and\n"
        "schema-v2 true-TTFT contract. FullLoad and Prefix arms used the same\n"
        "physical SSD store; ReComp used pixels and did not access that store.")

    table = [
        "| Method | Accuracy | Delta vs FullLoad | Delta vs ReComp | TTFT | "
        "TTFT reduction vs FullLoad | TTFT reduction vs ReComp | SSD MB | Preads |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        method = row["method_key"]
        label = ("ReComp" if method == "recompute" else
                 "FullLoad" if method == "fullload" else
                 f"Prefix {float(row['budget'])*100:.0f}%")
        d_full = "--" if method == "fullload" else \
            f"{row['accuracy_delta_vs_fullload_pp']:+.2f} pp"
        d_recomp = "--" if method == "recompute" else \
            f"{row['accuracy_delta_vs_recompute_pp']:+.2f} pp"
        t_full = "--" if method == "fullload" else \
            f"{row['ttft_reduction_vs_fullload_pct']:+.2f}%"
        t_recomp = "--" if method == "recompute" else \
            f"{row['ttft_reduction_vs_recompute_pct']:+.2f}%"
        table.append(
            f"| {label} | {row['accuracy_pct']:.2f}% | {d_full} | "
            f"{d_recomp} | {row['ttft_mean_ms']:.2f} ms | {t_full} | "
            f"{t_recomp} | {row['ssd_read_mb_mean']:.2f} | "
            f"{row['ssd_preads_mean']:.1f} |")
    start = text.index("## C. Main table\n\n") + len("## C. Main table\n\n")
    end = text.index("\n\nAccuracy is binary normalized GQA match.", start)
    text = text[:start] + "\n".join(table) + text[end:]
    text = _replace_once(
        text,
        "Accuracy is binary normalized GQA match.  `summary.csv` also reports the\n"
        "image-cluster bootstrap 95% CI for each accuracy and each paired FullLoad\n"
        "delta.",
        "Accuracy is binary normalized GQA match.  `summary.csv` reports the\n"
        "image-cluster bootstrap 95% CI for each accuracy and paired deltas against\n"
        "both same-run FullLoad and same-run ReComp.")
    text = _replace_once(
        text,
        "## D. Pareto frontier\n\n",
        "## D. Pareto frontier\n\n"
        "The required SSD and cache-path TTFT frontiers remain restricted to\n"
        "FullLoad plus Prefix. ReComp is excluded from the SSD frontier because its\n"
        "zero SSD traffic comes from doing pixel recomputation, not from a better SSD\n"
        "cache policy. A separate all-method TTFT frontier includes ReComp because\n"
        "true TTFT and accuracy are directly comparable across the nine arms.\n\n")
    def pareto_label(row: dict) -> str:
        if row["method_key"] == "recompute":
            return "ReComp"
        if row["method_key"] == "fullload":
            return "FullLoad"
        return f"{float(row['budget'])*100:.0f}%"

    cache_ttft_line = "- Accuracy vs true TTFT: " + ", ".join(
        pareto_label(row) for row in pareto_ttft
        if row["is_pareto_optimal"])
    all_ttft_line = "- Accuracy vs true TTFT (all methods): " + ", ".join(
        pareto_label(row) for row in pareto_ttft_all
        if row["is_pareto_optimal"])
    text = _replace_once(text, cache_ttft_line,
                         cache_ttft_line + "\n" + all_ttft_line)
    text = _replace_once(
        text,
        "Dominated points and their dominators are explicit in the two Pareto CSVs.",
        "Dominated points and their dominators are explicit in the three Pareto CSVs.")

    recomp = next(row for row in summary_rows
                  if row["method_key"] == "recompute")
    full = next(row for row in summary_rows
                if row["method_key"] == "fullload")
    baseline_note = f"""ReComp is measured in the same process and request loop. It reads pixels and
recomputes the vision tower plus multimodal prefill for every question, with
zero KV-store bytes and zero preads. Its mean true TTFT is
{recomp['ttft_mean_ms']:.2f} ms and accuracy is {recomp['accuracy_pct']:.2f}%.
FullLoad is {full['accuracy_delta_vs_recompute_pp']:+.2f} pp relative to ReComp
and its TTFT reduction relative to ReComp is
{full['ttft_reduction_vs_recompute_pct']:+.2f}% (negative means slower).

The ReComp timer follows the existing schema-v2 implementation: image
processor work and host-to-device transfer occur before `t0`; the measured
interval includes the vision tower, multimodal prefill, and first output token.
The raw `physical_layout` CSV cell is a run-global annotation emitted by
`04_eval.py`; ReComp never accesses that layout, as verified by zero I/O and
`retrieval=recompute`.

"""
    text = _replace_once(
        text, "## F. SSD/TTFT cost\n\n",
        "## F. SSD/TTFT cost\n\n" + baseline_note)
    text = _replace_once(
        text,
        "**Claim assessment:",
        "The ReComp arm is an additional compute baseline; cache-path operating-point\n"
        "thresholds, cache-only Pareto membership, and recovery analysis remain defined\n"
        "against FullLoad. The separate all-method true-TTFT Pareto includes ReComp.\n"
        "`baseline_comparison.csv` provides every arm's paired quality and TTFT\n"
        "comparison against both baselines.\n\n**Claim assessment:")
    return text


def publish(out_dir: Path, source_records: list[dict],
            files: dict[str, object]) -> None:
    out_dir = Path(out_dir).resolve()
    if out_dir != DEFAULT_OUTPUT.resolve():
        raise SweepError(f"output must be exactly {DEFAULT_OUTPUT}")
    if out_dir.exists() or out_dir.is_symlink():
        raise SweepError(f"refusing to overwrite existing output: {out_dir}")
    parent = out_dir.parent
    if not parent.is_dir() or parent.is_symlink():
        raise SweepError(f"bad output parent: {parent}")
    stage = Path(tempfile.mkdtemp(
        prefix=".image_only_budget_sweep_with_recomp.stage.", dir=parent))
    try:
        for name, value in files.items():
            path = stage / name
            if name == "per_request.csv":
                shutil.copyfile(value, path)
            elif name.endswith(".json"):
                BASE.write_json(path, value)
            elif name.endswith(".csv"):
                BASE.write_csv(path, value)
            elif name == "README.md":
                path.write_text(str(value))
            else:
                raise SweepError(f"unknown output type: {name}")
        if {path.name for path in stage.iterdir()} != set(files):
            raise SweepError("staged output file set mismatch")
        BASE.recheck_sources(source_records)
        parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            CORE._rename_noreplace(parent_fd, stage.name,
                                   parent_fd, out_dir.name)
        finally:
            os.close(parent_fd)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def analyze(run_dir: Path, out_dir: Path, store: Path) -> dict:
    ordered_keys, frozen, run, prior, profile = load_frozen_inputs(run_dir)
    store_info, store_sources = BASE.validate_store(
        store, profile, {image for image, _ in ordered_keys})
    runtime = validate_run_semantics(run, store, ordered_keys)
    (summary_rows, cache_summary, paired_full, paired_recompute,
     maps) = summarize(run, ordered_keys)
    coverage_rows, correlations, old_coverage = BASE.compute_coverage(
        store, ordered_keys, cache_summary)

    # Keep ReComp out of cache-policy frontiers. The separate global true-TTFT
    # frontier below includes it because all nine arms share timing semantics.
    pareto_ssd = BASE.pareto_rows(
        cache_summary, "ssd_read_mb_mean", "ssd_read_mb_mean")
    pareto_ttft = BASE.pareto_rows(
        cache_summary, "ttft_mean_ms", "ttft_mean_ms")
    for rows in (pareto_ssd, pareto_ttft):
        for row in rows:
            row["frontier_population"] = "fullload_and_prefix_only"
            row["recompute_excluded"] = True
            row["recompute_exclusion_reason"] = (
                "zero-SSD pixel-recompute baseline; not an SSD cache path")
    pareto_ttft_all = BASE.pareto_rows(
        summary_rows, "ttft_mean_ms", "ttft_mean_ms")
    for row in pareto_ttft_all:
        row["frontier_population"] = "all_nine_same_run_methods"
        row["recompute_excluded"] = False
        row["recompute_exclusion_reason"] = ""
    tradeoff = BASE.tradeoff_rows(cache_summary, pareto_ssd, pareto_ttft)
    question_rows, image_rows, error_rows = augment_sensitivity(
        maps, ordered_keys, frozen)
    operating = BASE.operating_points(
        cache_summary, pareto_ssd, pareto_ttft)
    reproducibility = BASE.cross_run_reproducibility(
        run, prior, ordered_keys)

    with BASE.OLD_VALIDATION.open() as handle:
        old_validation = json.load(handle)
    identity = old_validation.get("full_load_identity", {})
    agreement = identity.get("prediction_agreement")
    historical_warning = {
        "strict_full_load_exact_pass": bool(
            old_validation.get("strict_full_load_exact_pass", False)),
        "agreement_requests": int(round(float(agreement) * BASE.N_QUESTIONS)
                                  if agreement is not None else 238),
        "n_requests": BASE.N_QUESTIONS,
        "structural_integrity_pass": bool(
            old_validation.get("structural_integrity_pass", False)),
        "scope": "historical raster-vs-image-only cross-layout comparison",
    }

    source_records = []
    for label, path in (
            ("frozen_index", CORE.INDEX),
            ("evaluation_code", ROOT / "scripts/04_eval.py"),
            ("serve_code", ROOT / "mmimpress/serve.py"),
            ("model_code", ROOT / "mmimpress/model.py"),
            ("dataset_metric_code", ROOT / "mmimpress/dataset.py"),
            ("selector_code", ROOT / "mmimpress/cvpr25.py"),
            ("reader_code", ROOT / "mmimpress/store.py"),
            ("analysis_core", BASE.CORE_PATH),
            ("budget_sweep_analysis_base", BASE_PATH),
            ("analyzer", Path(__file__)),
            ("build_profile", BASE.BUILD_PROFILE),
            ("raw_calibration_analysis_only", BASE.RAW_CALIBRATION),
            ("old_coverage", BASE.OLD_COVERAGE),
            ("old_config", BASE.OLD_CONFIG),
            ("old_validation", BASE.OLD_VALIDATION)):
        source_records.append(BASE.source_record(label, path))
    for run_name, loaded in (("sweep_with_recomp", run), ("prior", prior)):
        for kind, path in loaded["files"].items():
            source_records.append(BASE.source_record(
                f"run.{run_name}.{kind}", path))
    source_records.extend(store_sources)
    unique = {}
    for record in source_records:
        unique.setdefault(record["path"], record)
    source_records = list(unique.values())
    input_manifest = {
        record["label"]: {
            "path": str(record["path"]), "sha256": record["sha256"]}
        for record in source_records
    }

    config = {
        "schema_version": 1,
        "analysis": "image_only_repack_prefix_budget_sweep_with_recomp",
        "generated_unix_time": time.time(),
        "cpu_only_analyzer": True,
        "frozen_workload": {
            "dataset": "gqa", "images": BASE.N_IMAGES,
            "questions": BASE.N_QUESTIONS, "questions_per_image": 6,
            "question_slice": [4, 10], "index_sha256": BASE.INDEX_SHA256,
            "workload_sha256": BASE.WORKLOAD_SHA256,
        },
        "budgets": list(BUDGETS),
        "same_run_fullload": True,
        "same_run_recompute": True,
        "per_question_execution_order": [
            "recompute", "fullload", *PREFIX_METHODS],
        "model": {
            "id": "llava-hf/llava-v1.6-vicuna-7b-hf",
            "cached_revision": BASE.local_model_revision(),
            "quantization": "4-bit NF4 double-quant",
            "compute_dtype": "bfloat16", "attention": "eager",
            "decoding": "greedy", "max_new_tokens": 16,
        },
        "methods": {
            "recompute": {
                "semantics": "pixel recomputation for every question",
                "kv_store_access": False,
                "timed_scope": runtime["recompute"]["timed_scope"],
            },
            "fullload": {
                "semantics": "read the complete visual KV from SSD"},
            "prefix": {
                "layout": "visionzip_image_only",
                "retrieval": "sequential first-k Prefix",
                "budget_rounding":
                    "Python round(total_chunks * budget), clamped",
                "chunk_size": BASE.CHUNK_SIZE,
                "separator_policy": "sidecar",
                "prefix_io":
                    "adjacent chunk ranges coalesced to one span per file",
                "layout_uses_dataset_question": False,
                "calibration_questions": 0,
                "online_scoring": False,
            },
        },
        "pareto": {
            "ssd_and_cache_ttft": {
                "population": "FullLoad plus seven Prefix budgets",
                "recompute_included": False,
                "reason": (
                    "ReComp has zero SSD traffic by construction and is a "
                    "pixel-compute baseline rather than an SSD cache path"),
            },
            "all_method_ttft": {
                "population": "ReComp, FullLoad, and seven Prefix budgets",
                "recompute_included": True,
                "reason": "true TTFT and accuracy are directly comparable",
                "pareto_methods": [
                    row["method_key"] for row in pareto_ttft_all
                    if row["is_pareto_optimal"]],
            },
        },
        "store": store_info,
        "run_command": run["summary"]["command"],
        "bootstrap": {
            "primary_unit": "image cluster", "clusters": BASE.N_IMAGES,
            "questions_per_cluster": 6,
            "resamples": BASE.BOOTSTRAP_RESAMPLES,
            "seed": BASE.BOOTSTRAP_SEED,
            "paired_vs_same_run_fullload": paired_full,
            "paired_vs_same_run_recompute": paired_recompute,
        },
        "coverage": {
            "sparsevlm_scores_analysis_only": True,
            "used_for_layout_or_serving": False,
            "correlations": correlations,
            "frozen_25pct_reference": old_coverage,
        },
        "operating_points": operating,
        "operating_point_reference": "same-run FullLoad",
        "cross_run_25_50_reproducibility": reproducibility,
        "known_historical_cross_layout_warning": historical_warning,
        "inputs": input_manifest,
    }
    cross_run_exact = bool(reproducibility["all_predictions_exact"] and
                           reproducibility["all_first_tokens_exact"])
    validation = {
        "schema_version": 1,
        "all_passed": cross_run_exact,
        "checks": {
            "exact_frozen_40_image_240_question_workload": {"passed": True},
            "exact_recompute_fullload_seven_prefix_arms_2160_rows": {
                "passed": True},
            "same_process_same_run_recompute_and_fullload": {"passed": True},
            "recompute_is_pixel_path_with_zero_ssd_and_no_selection": {
                "passed": True},
            "same_immutable_image_only_store_all_cache_arms": {
                "passed": True,
                "manifest_sha256": store_info["manifest_sha256"]},
            "model_decoding_schema_v2_cold_chunk64_sidecar": {
                "passed": True},
            "zero_calibration_query_static_diversity_prefix_selection": {
                "passed": True},
            "exact_first_k_python_rounding_and_budget_nesting": {
                "passed": True},
            "actual_cache_pread_bytes_and_split_exact": {"passed": True},
            "contiguous_prefix_preads_64_normal_plus_1_separator": {
                "passed": True},
            "ttft_and_e2e_timing_algebra_all_arms": {"passed": True},
            "paired_bootstrap_against_both_same_run_baselines": {
                "passed": True},
            "cache_pareto_excludes_recompute_and_global_ttft_includes_it": {
                "passed": True},
            "coverage_complete_monotonic_and_25pct_reproduced": {
                "passed": True},
            "cross_run_25_50_predictions_and_first_tokens_exact": {
                "passed": cross_run_exact},
            "source_artifacts_rechecked_before_atomic_publication": {
                "passed": True},
        },
        "runtime": runtime,
        "pareto_scope": config["pareto"],
        "cross_run_reproducibility": reproducibility,
        "known_historical_cross_layout_warning": historical_warning,
    }
    readme = make_readme(
        summary_rows, cache_summary, tradeoff, coverage_rows, pareto_ssd,
        pareto_ttft, pareto_ttft_all, operating, correlations, validation,
        question_rows, run["summary"]["command"])
    files = {
        "config.json": config,
        "per_request.csv": run["files"]["csv"],
        "summary.csv": summary_rows,
        "baseline_comparison.csv": baseline_rows(summary_rows),
        "budget_tradeoff.csv": tradeoff,
        "importance_coverage.csv": coverage_rows,
        "pareto_accuracy_vs_ssd.csv": pareto_ssd,
        "pareto_accuracy_vs_ttft.csv": pareto_ttft,
        "pareto_accuracy_vs_ttft_all_methods.csv": pareto_ttft_all,
        "validation.json": validation,
        "README.md": readme,
        "per_question_sensitivity.csv": question_rows,
        "per_image_sensitivity.csv": image_rows,
        "error_analysis.csv": error_rows,
    }
    publish(out_dir, source_records, files)
    return {
        "output": str(Path(out_dir).resolve()),
        "recommended_budget": operating["recommended_budget"],
        "summary": summary_rows,
        "operating_points": operating,
        "validation": validation,
    }


def self_test() -> None:
    assert DEFAULT_RUN == ROOT / \
        "runs/image_only_repack_budget_sweep_with_recomp/main_20_50"
    assert DEFAULT_OUTPUT == ROOT / \
        "results/image_only_repack_budget_sweep_with_recomp"
    assert EXPECTED_METHODS[:2] == ("fullload", "recompute")
    assert len(EXPECTED_METHODS) == 9

    command = (
        "python scripts/04_eval.py --index data/index.json "
        "--store kvstore_image_only_visionzip --limit 40 --questions 6 "
        "--skip 4 --ratio 0.25 "
        "--budgets 0.20,0.25,0.30,0.35,0.40,0.45,0.50 "
        "--selectors visionzip_repack_prefix "
        "--prefix-layout visionzip_image_only --sep-policy sidecar "
        "--metric gqa --dataset gqa --expect-images 40 "
        "--expect-questions 240 "
        "--run-dir runs/image_only_repack_budget_sweep_with_recomp/main_20_50")
    validate_command(command)
    try:
        validate_command(command + " --no-recompute")
    except SweepError:
        pass
    else:
        raise AssertionError("--no-recompute was not rejected")

    # A zero-cost, high-quality ReComp row must not erase the cache frontier.
    cache = [
        {"method_key": "fullload", "budget": "", "budget_pct": "",
         "accuracy": 0.8, "accuracy_pct": 80.0, "cost": 10.0},
        {"method_key": "prefix@25", "budget": 0.25, "budget_pct": 25,
         "accuracy": 0.7, "accuracy_pct": 70.0, "cost": 2.0},
        {"method_key": "prefix@50", "budget": 0.50, "budget_pct": 50,
         "accuracy": 0.75, "accuracy_pct": 75.0, "cost": 5.0},
    ]
    recomp = {"method_key": "recompute", "budget": "", "budget_pct": "",
              "accuracy": 0.9, "accuracy_pct": 90.0, "cost": 0.0}
    cache_frontier = BASE.pareto_rows(cache, "cost", "cost")
    assert all(row["is_pareto_optimal"] for row in cache_frontier)
    incorrect_frontier = BASE.pareto_rows(
        [recomp, *cache], "cost", "cost")
    assert not any(row["is_pareto_optimal"]
                   for row in incorrect_frontier if row["method_key"] !=
                   "recompute")
    assert any(row["method_key"] == "recompute" and
               row["is_pareto_optimal"] for row in incorrect_frontier)
    assert not next(row for row in incorrect_frontier
                    if row["method_key"] == "fullload")["is_pareto_optimal"]

    keys = [("image", "question")]
    valid = {
        "image_id": "image", "question_id": "question",
        "method_key": "recompute", "method": "ReComp",
        "retention": "", "retention_kind": "none",
        "retrieval": "recompute", "physical_layout": "visionzip_image_only",
        "ssd_read_ms": 0, "ssd_read_bytes": 0, "ssd_read_chunks": 0,
        "ssd_preads": 0, "normal_kv_read_bytes": 0,
        "separator_read_bytes": 0, "normal_kv_preads": 0,
        "separator_preads": 0, "total_actual_pread_bytes": 0,
        "mean_bytes_per_pread": 0, "normal_mean_bytes_per_pread": 0,
        "separator_mean_bytes_per_pread": 0, "selector_ms": "",
        "scatter_ms": "", "prepare_ms": "", "n_chunks_selected": "",
        "n_chunks_total": "", "touched_chunk_fraction": "",
        "logical_kv_ratio": "", "normal_chunk_count_total": "",
        "static_score_calls": "", "query_score_calls": "",
        "diversity_calls": "", "selected_chunk_ids_per_layer": "null",
        "selection_mode": "", "separator_policy": "",
        "validated_prefix_layout": "",
        "reordered_prefix_store_validated": "", "ttft_ms": 10.0,
        "prefill_ms": 10.0, "decode_ms": 2.0, "e2e_latency_ms": 12.0,
        "first_token_id": 7, "generated_tokens": 2,
    }
    assert validate_recompute([valid], keys)["passed"]
    invalid = dict(valid, ssd_read_bytes=1)
    try:
        validate_recompute([invalid], keys)
    except SweepError:
        pass
    else:
        raise AssertionError("nonzero ReComp SSD bytes were not rejected")

    # Keep the signs of both baseline comparisons explicit.
    recomp_acc, full_acc, prefix_acc = 0.62, 0.60, 0.58
    recomp_ttft, full_ttft, prefix_ttft = 1000.0, 700.0, 400.0
    assert abs((prefix_acc - recomp_acc) * 100 - (-4.0)) < 1e-12
    assert abs((prefix_acc - full_acc) * 100 - (-2.0)) < 1e-12
    assert abs(100 * (1 - prefix_ttft / recomp_ttft) - 60.0) < 1e-12
    assert abs(100 * (1 - prefix_ttft / full_ttft) - 42.8571428571) < 1e-9
    print("budget-sweep-with-ReComp analyzer self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    result = analyze(Path(args.run_dir), Path(args.out_dir), Path(args.store))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

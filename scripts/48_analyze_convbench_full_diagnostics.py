#!/usr/bin/env python3
"""Read-only final ConvBench diagnostics from frozen answer and judge artifacts.

All rates use the complete conversation population. A missing or unresolved
judge decision contributes zero model wins; a missing decision is also reported
as an integrity failure. The optional overall judge is diagnostic only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("recompute", "fullload", "prefix25", "prefix45")
LABELS = dict(zip(METHODS, ("ReComp", "FullLoad", "Prefix25", "Prefix45")))
STAGES = ("_first_turn", "_second_turn", "_third_turn")
STAGE_NAMES = ("S1", "S2", "S3")


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise ImportError(filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def describe(values: list[float]) -> dict:
    if not values:
        raise ValueError("empty statistics population")
    values = sorted(float(value) for value in values)
    def quantile(q: float) -> float:
        at = (len(values) - 1) * q
        lo = int(at)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (at - lo)
    return {"n": len(values), "mean": statistics.fmean(values),
            "p50": quantile(.50), "p95": quantile(.95), "p99": quantile(.99),
            "max": values[-1]}


def classify_repetition(row: dict, sanity) -> dict:
    signals = sanity.repeated_segments(row["prediction"])
    clear = any(
        item["kind"] == "identical_character_run" or
        (item["kind"] == "repeated_token_phrase" and
         any(char.isalpha() for char in item["unit"]))
        for item in signals)
    return {"degenerate_repetition": clear,
            "numeric_repetition_review": bool(signals) and not clear,
            "repetition_evidence": signals}


def validate_judge(judge_run: Path, answer_run: Path, index_path: Path,
                   dialogs: list[dict], analysis, *, verify_checkpoints: bool = True
                   ) -> tuple[dict, dict, dict]:
    config = json.loads((judge_run / "config.json").read_text(encoding="utf-8"))
    if (config["index_sha256"] != file_hash(index_path) or
            config["answer_sha256"] != analysis._judge_answer_input_hash(answer_run, dialogs) or
            config["selected_source_row_indices"] !=
            [d["source_row_index"] for d in dialogs]):
        raise ValueError("judge config is not tied to these immutable answer artifacts")
    if (config.get("model_id") != "meta-llama/Meta-Llama-3.1-8B-Instruct" or
            config.get("second_pass_model_id") != config.get("model_id") or
            config.get("commercial_api_used") is not False or
            config.get("decoding") != "greedy; temperature=0; do_sample=False"):
        raise ValueError("judge config violates frozen local Llama greedy protocol")
    raw_path = judge_run / "judge_raw.jsonl"
    records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    expected = {(LABELS[method], d["conversation_id"], stage)
                for method in METHODS for d in dialogs for stage in STAGES}
    source_rows = {d["conversation_id"]: d["source_row_index"] for d in dialogs}
    keyed = {}
    overall = 0
    expected_overall = {(LABELS[method], d["conversation_id"], "_overall_conversation")
                        for method in METHODS for d in dialogs}
    for record in records:
        key = (record.get("method"), record.get("conversation_id"), record.get("judge_turn"))
        if key[2] == "_overall_conversation":
            if key not in expected_overall:
                raise ValueError(f"unexpected overall judge record: {key}")
            overall += 1
        elif key not in expected:
            raise ValueError(f"unexpected stage judge record: {key}")
        if key in keyed:
            raise ValueError(f"duplicate judge record: {key}")
        if (record.get("source_row_index") != source_rows.get(key[1]) or
                record.get("position") not in (0, 1)):
            raise ValueError(f"judge source or pairwise position mismatch: {key}")
        winner = record.get("winner")
        stage = record.get("parse_stage")
        if winner not in ("A", "B", "C", None) or stage not in (
                "first_pass", "second_pass", "unresolved"):
            raise ValueError(f"unknown judge winner or parse stage: {key}")
        if (winner in ("A", "B")) != (stage in ("first_pass", "second_pass")):
            raise ValueError(f"judge label and parser stage disagree: {key}")
        if stage != "first_pass" and not isinstance(record.get("extraction_raw_response"), str):
            raise ValueError(f"second-pass extraction artifact absent: {key}")
        win = winner == ("A" if record["position"] == 0 else "B")
        observed = record.get("model_wins")
        if winner in ("C", None):
            if observed not in (None, False):
                raise ValueError(f"unresolved judge marked win: {key}")
        elif type(observed) is not bool or observed != win:
            raise ValueError(f"judge win does not follow pairwise position: {key}")
        keyed[key] = record
    missing = expected.difference(keyed)
    # A full run with partial judge output must fail before publishing scores.
    if missing:
        raise ValueError(f"missing {len(missing)} of {len(expected)} stage judgments")
    if overall not in (0, len(dialogs) * len(METHODS)):
        raise ValueError("partial overall judge population")
    judge_turns = tuple(config.get("judge_turns", ()))
    if judge_turns not in (STAGES, STAGES + ("_overall_conversation",)):
        raise ValueError("judge config has unexpected ordered stage population")
    if (overall == 0) != (judge_turns == STAGES):
        raise ValueError("judge raw overall population differs from config")
    checkpoint_count = 0
    if verify_checkpoints:
        judge_runner = load_script("convbench_42_diagnostics_dependency",
                                   "42_judge_convbench.py")
        expected_paths = set()
        for dialog in dialogs:
            for method in LABELS.values():
                path = (judge_run / "judgments" / method /
                        f"{dialog['source_row_index']:04d}.json")
                expected_paths.add(path)
                if not path.is_file() or path.is_symlink():
                    raise FileNotFoundError(f"missing or linked judge checkpoint: {path}")
                first = keyed[(method, dialog["conversation_id"], STAGES[0])]
                records = judge_runner.validate_checkpoint(
                    path, method, dialog, first["position"], judge_turns)
                if len(records) != len(judge_turns):
                    raise ValueError(f"incomplete judge checkpoint: {path}")
                for record in records:
                    key = (record["method"], record["conversation_id"], record["judge_turn"])
                    if record != keyed.get(key):
                        raise ValueError(f"judge checkpoint/raw content mismatch: {path}/{key}")
                checkpoint_count += 1
        actual_paths = {path for path in (judge_run / "judgments").rglob("*")
                        if path.is_file() or path.is_symlink()}
        if actual_paths != expected_paths:
            raise ValueError("unexpected or missing judge checkpoint files")
    return keyed, {"stage_expected": len(expected), "stage_completed": len(expected),
                   "overall_diagnostic_count": overall,
                   "judge_checkpoints_verified": checkpoint_count,
                   "judge_raw_sha256": file_hash(raw_path)}, config


def scores_and_judge(keyed: dict, dialogs: list[dict]) -> tuple[list[dict], list[dict], dict]:
    ids = [d["conversation_id"] for d in dialogs]
    n = len(ids)
    quality = []
    diagnostic = []
    vectors = {}
    for method in METHODS:
        label = LABELS[method]
        stage_vectors = []
        scores = {}
        total_unresolved = 0
        for stage, score_name in zip(STAGES, STAGE_NAMES):
            records = [keyed[(label, cid, stage)] for cid in ids]
            wins = np.array([int(record["winner"] == (
                "A" if record["position"] == 0 else "B")) for record in records],
                dtype=np.float64)
            unresolved = sum(record["winner"] in ("C", None) for record in records)
            first = sum(record["parse_stage"] == "first_pass" for record in records)
            second = sum(record["parse_stage"] == "second_pass" for record in records)
            if first + second + unresolved != n:
                raise ValueError("judge parser accounting differs from fixed denominator")
            scores[score_name] = 100 * float(wins.mean())
            total_unresolved += unresolved
            stage_vectors.append(wins)
            diagnostic.append({"method": label, "stage": score_name,
                               "denominator": n, "model_wins": int(wins.sum()),
                               "first_pass_parsed": first,
                               "second_pass_recovered": second,
                               "unresolved": unresolved,
                               "unresolved_rate": unresolved / n})
        vectors[label] = np.stack(stage_vectors, axis=1)
        scores["Avg"] = sum(scores[name] for name in STAGE_NAMES) / 3
        quality.append({"method": label, **scores,
                        "delta_Avg_vs_ReComp": None,
                        "unresolved": total_unresolved,
                        "fixed_stage_denominator": n})
    recomp_avg = quality[0]["Avg"]
    for item in quality:
        item["delta_Avg_vs_ReComp"] = item["Avg"] - recomp_avg
    return quality, diagnostic, vectors


def paired_bootstrap(rows: list[dict], ids: list[str], vectors: dict,
                     *, resamples: int = 10_000, seed: int = 1234) -> list[dict]:
    """Resample complete conversations, retaining all three turns and methods."""
    by_key = {(row["conversation_id"], row["method_key"], row["turn_id"]): row
              for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("duplicate request in bootstrap population")
    n = len(ids)
    ttft = {LABELS[method]: np.array([
        np.mean([float(by_key[(cid, method, turn)]["end_to_end_ttft_ms"])
                 for turn in (2, 3)]) for cid in ids], dtype=np.float64)
        for method in METHODS}
    sample = np.random.default_rng(seed).integers(0, n, size=(resamples, n))
    out = []
    baseline_quality = vectors["ReComp"].mean(axis=1)
    base_ttft = ttft["ReComp"]
    for method in METHODS:
        label = LABELS[method]
        delta = vectors[label].mean(axis=1) - baseline_quality
        delta_estimate = 100 * float(delta.mean())
        delta_samples = 100 * delta[sample].mean(axis=1)
        lower, upper = np.percentile(delta_samples, (2.5, 97.5))
        out.append({"method": label, "metric": "delta_Avg_vs_ReComp_pp",
                    "estimate": delta_estimate, "ci95_low": float(lower),
                    "ci95_high": float(upper), "cluster_unit": "conversation",
                    "resamples": resamples, "seed": seed})
        # Resample both arms with the same conversation indices. The two
        # independent request TTFT observations inside each cluster stay paired.
        reduction = 100 * (1 - float(ttft[label].mean()) / float(base_ttft.mean()))
        samples = 100 * (1 - ttft[label][sample].mean(axis=1) /
                         base_ttft[sample].mean(axis=1))
        lower, upper = np.percentile(samples, (2.5, 97.5))
        out.append({"method": label, "metric": "cache_hit_TTFT_reduction_vs_ReComp_pct",
                    "estimate": reduction, "ci95_low": float(lower),
                    "ci95_high": float(upper), "cluster_unit": "conversation",
                    "resamples": resamples, "seed": seed})
    return out


def generation_tables(rows: list[dict], dialogs: list[dict], sanity) -> dict:
    ids = [d["conversation_id"] for d in dialogs]
    by_key = {(r["conversation_id"], r["method_key"], int(r["turn_id"])): r for r in rows}
    if len(by_key) != len(rows) or len(rows) != 12 * len(ids):
        raise ValueError("generation coverage or uniqueness failure")
    request_table = []
    repetition_table = []
    context_table = []
    system_table = []
    turn1_table = []
    repetition_by_key = {}
    for row in rows:
        flags = classify_repetition(row, sanity)
        key = (row["conversation_id"], row["method_key"], row["turn_id"])
        repetition_by_key[key] = flags
        request_table.append({
            "conversation_id": row["conversation_id"], "image_id": row["image_id"],
            "method": row["method"], "turn_id": row["turn_id"],
            "ttft_ms": row["end_to_end_ttft_ms"],
            "request_e2e_ms": row["request_e2e_ms"],
            "prompt_tokens": row["prompt_tokens"],
            "history_tokens": row["history_tokens"],
            "previous_answer_tokens": row["previous_answer_tokens"],
            "actual_input_tokens": row["context_input_tokens"],
            "ssd_read_bytes": row["ssd_read_bytes"],
            "ssd_read_mb": row["ssd_read_bytes"] / 1e6,
            "ssd_read_MB": row["ssd_read_bytes"] / 1e6,
            "pread_count": row["ssd_read_preads"],
            "ssd_read_ms": row["ssd_read_ms"],
            "scatter_ms": row["scatter_ms"], "prefill_ms": row["prefill_ms"],
            "planning_ms": row["first_k_planning_ms"],
            "context_overflow_tokens": row["input_overflow_tokens"],
            "nominal_context_limit": row["text_context_limit"],
            "degenerate_repetition": flags["degenerate_repetition"],
            "numeric_repetition_review": flags["numeric_repetition_review"],
            "first_token_id": row["first_token_id"]})
    for method in METHODS:
        for turn in (1, 2, 3):
            group = [by_key[(cid, method, turn)] for cid in ids]
            repeat = sum(repetition_by_key[(cid, method, turn)]["degenerate_repetition"]
                         for cid in ids)
            numeric = sum(repetition_by_key[(cid, method, turn)]["numeric_repetition_review"]
                          for cid in ids)
            repetition_table.append({"method": LABELS[method], "turn_id": turn,
                                     "requests": len(group), "repetition_count": repeat,
                                     "repetition_rate": repeat / len(group),
                                     "numeric_review_count": numeric})
            lengths = describe([r["context_input_tokens"] for r in group])
            context_table.append({"method": LABELS[method], "turn_id": turn,
                                  "requests": len(group),
                                  "above_nominal_4096_count": sum(
                                      r["context_input_tokens"] > 4096 for r in group),
                                  "max_input_tokens": lengths["max"],
                                  "p50_input_tokens": lengths["p50"],
                                  "p95_input_tokens": lengths["p95"],
                                  "p99_input_tokens": lengths["p99"]})
        hits = [by_key[(cid, method, turn)] for cid in ids for turn in (2, 3)]
        t2 = describe([r["end_to_end_ttft_ms"] for r in hits if r["turn_id"] == 2])
        t3 = describe([r["end_to_end_ttft_ms"] for r in hits if r["turn_id"] == 3])
        pooled = describe([r["end_to_end_ttft_ms"] for r in hits])
        system_table.append({"method": LABELS[method],
                             "T2_TTFT_mean_ms": t2["mean"],
                             "T2_TTFT_p50_ms": t2["p50"],
                             "T2_TTFT_p95_ms": t2["p95"],
                             "T3_TTFT_mean_ms": t3["mean"],
                             "T3_TTFT_p50_ms": t3["p50"],
                             "T3_TTFT_p95_ms": t3["p95"],
                             "cache_hit_TTFT_mean_ms": pooled["mean"],
                             "cache_hit_TTFT_p50_ms": pooled["p50"],
                             "cache_hit_TTFT_p95_ms": pooled["p95"],
                             "cache_hit_requests": pooled["n"],
                             "TTFT_reduction_vs_ReComp_pct": None,
                             "SSD_MB_per_request": statistics.fmean(
                                 r["ssd_read_bytes"] / 1e6 for r in hits),
                             "SSD_reduction_vs_FullLoad_pct": None,
                             "pread_count_mean": statistics.fmean(
                                 r["ssd_read_preads"] for r in hits),
                             "ssd_read_mean_ms": statistics.fmean(
                                 r["ssd_read_ms"] for r in hits),
                             "scatter_mean_ms": statistics.fmean(
                                 r["scatter_ms"] for r in hits),
                             "prefill_mean_ms": statistics.fmean(
                                 r["prefill_ms"] for r in hits),
                             "prompt_tokens_mean": statistics.fmean(
                                 r["prompt_tokens"] for r in hits),
                             "history_tokens_mean": statistics.fmean(
                                 r["history_tokens"] for r in hits)})
    base_ttft = system_table[0]["cache_hit_TTFT_mean_ms"]
    base_ssd = system_table[1]["SSD_MB_per_request"]
    if base_ttft <= 0 or base_ssd <= 0:
        raise ValueError("invalid system comparison baseline")
    for record in system_table:
        record["TTFT_reduction_vs_ReComp_pct"] = 100 * (
            1 - record["cache_hit_TTFT_mean_ms"] / base_ttft)
        record["SSD_reduction_vs_FullLoad_pct"] = 100 * (
            1 - record["SSD_MB_per_request"] / base_ssd)
    for cid in ids:
        t1 = [by_key[(cid, method, 1)] for method in METHODS]
        tests = {
            "pixel_input_agreement": len({r["input_tensors_sha256"] for r in t1}) == 1,
            "prompt_agreement": len({r["prompt_sha256"] for r in t1}) == 1,
            "A1_text_agreement": len({r["prediction"] for r in t1}) == 1,
            "first_token_agreement": len({r["first_token_id"] for r in t1}) == 1,
            "generated_token_count_agreement": len({r["generated_tokens"] for r in t1}) == 1,
        }
        turn1_table.append({"conversation_id": cid,
                            "source_row_index": t1[0]["source_row_index"],
                            **tests, "all_agree": all(tests.values()),
                            "ReComp_prompt_sha256": t1[0]["prompt_sha256"],
                            "ReComp_A1_text_sha256": text_hash(t1[0]["prediction"]),
                            "method_A1_hashes_json": json.dumps({
                                LABELS[method]: text_hash(by_key[(cid, method, 1)]["prediction"])
                                for method in METHODS}, sort_keys=True)})
    return {"per_request": request_table, "repetition": repetition_table,
            "context": context_table, "system": system_table, "turn1": turn1_table}


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        raise ValueError(f"empty output table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
        handle.flush()
        os.fsync(handle.fileno())


def analyze(index_path: Path, answer_run: Path, judge_run: Path, n: int,
            *, full: bool = False, bootstrap_resamples: int = 10_000) -> dict:
    if full and n != 577:
        raise ValueError("full-run validation requires exactly 577 conversations")
    if not 1 <= n <= 577:
        raise ValueError("expected conversations must be 1..577")
    analysis = load_script("convbench_45_diagnostics_dependency", "45_analyze_convbench_full.py")
    sanity = load_script("convbench_47_diagnostics_dependency", "47_check_convbench_quality_sanity.py")
    rows, persistence, config = analysis.validate_and_load(index_path, answer_run, n)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    dialogs = analysis.selected_dialogs(index, config, n)
    ids = [d["conversation_id"] for d in dialogs]
    keyed, judge_coverage, judge_config = validate_judge(
        judge_run, answer_run, index_path, dialogs, analysis)
    quality, judge, vectors = scores_and_judge(keyed, dialogs)
    generation = generation_tables(rows, dialogs, sanity)
    bootstrap = paired_bootstrap(rows, ids, vectors, resamples=bootstrap_resamples)
    persistence_table = analysis.summarize_persistence(persistence)
    sessions = analysis.summarize(rows, persistence)[2]
    summary = {
        "expected_conversations": n, "completed_conversations": len(dialogs),
        "expected_generation_requests": 12 * n,
        "completed_generation_requests": len(rows),
        "expected_stage_judgments": 12 * n,
        "completed_stage_judgments": judge_coverage["stage_completed"],
        "overall_diagnostic_judgments": judge_coverage["overall_diagnostic_count"],
        "judge_checkpoints_verified": judge_coverage["judge_checkpoints_verified"],
        "first_pass_parsed": sum(r["first_pass_parsed"] for r in judge),
        "second_pass_recovered": sum(r["second_pass_recovered"] for r in judge),
        "unresolved": sum(r["unresolved"] for r in judge),
        "unresolved_rate": sum(r["unresolved"] for r in judge) / (12 * n),
        "turn1_full_agreement": sum(r["all_agree"] for r in generation["turn1"]),
        "turn1_generation_agreement": sum(r["A1_text_agreement"] for r in generation["turn1"]),
        "degenerate_repetition_requests": sum(
            r["repetition_count"] for r in generation["repetition"]),
        "above_nominal_4096_requests": sum(
            r["above_nominal_4096_count"] for r in generation["context"]),
        "max_input_tokens": max(r["context_input_tokens"] for r in rows),
        "unique_image_ids": len({r["image_id"] for r in rows}),
        "persistence_store_instances": len(persistence),
        "technical_failures_in_completed_artifacts": 0,
        "retry_count": None,
        "fixed_stage_denominator": n,
        "answer_index_sha256": config["index_sha256"],
        "judge_raw_sha256": judge_coverage["judge_raw_sha256"],
        "judge_model_id": judge_config.get("judge_model_id", judge_config.get("checkpoint")),
        "cache_hit_condition": "OS-page-cache-cold via posix_fadvise(DONTNEED) with buffered pread; SSD controller cache not explicitly flushed",
        "score_semantics": "fixed denominator; unresolved zero wins; overall excluded from Avg",
        "stage_score_interpretation": "ConvBench stage-wise conversation quality; S1 prompt includes all three responses",
        "passed": True,
    }
    if (len(rows) != 12 * n or judge_coverage["stage_completed"] != 12 * n or
            any(r["fixed_stage_denominator"] != n for r in quality) or
            any(r["denominator"] != n for r in judge) or
            any(r["cache_hit_requests"] != 2 * n for r in generation["system"])):
        raise ValueError("full logical size or fixed denominator validation failed")
    return {"validation": summary, "quality": quality, "judge": judge,
            "generation": generation, "bootstrap": bootstrap,
            "persistence": persistence_table,
            "persistence_per_conversation": persistence,
            "sessions": sessions}


def publish(result: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to replace analysis artifact: {output}")
    with tempfile.TemporaryDirectory(prefix=".convbench-final-", dir=output.parent) as tmp:
        staging = Path(tmp)
        for filename, records in (
            ("quality.csv", result["quality"]), ("judge_diagnostics.csv", result["judge"]),
            ("repetition.csv", result["generation"]["repetition"]),
            ("context.csv", result["generation"]["context"]),
            ("system.csv", result["generation"]["system"]),
            ("turn1_agreement.csv", result["generation"]["turn1"]),
            ("per_request_diagnostics.csv", result["generation"]["per_request"]),
            ("paired_bootstrap.csv", result["bootstrap"]),
            ("persistence.csv", result["persistence"]),
            ("persistence_per_conversation.csv", result["persistence_per_conversation"]),
            ("session_latency.csv", result["sessions"]),
        ):
            write_csv(staging / filename, records)
        (staging / "validation.json").write_text(
            json.dumps(result["validation"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        v = result["validation"]
        lines = ["# ConvBench final diagnostics", "",
                 f"Validated {v['completed_conversations']}/{v['expected_conversations']} conversations, "
                 f"{v['completed_generation_requests']}/{v['expected_generation_requests']} generation "
                 f"requests, and {v['completed_stage_judgments']}/{v['expected_stage_judgments']} "
                 "stage judgments.", "",
                 "The S1/S2/S3 denominator is fixed at the full conversation count per method. "
                 "Unresolved decisions remain in the denominator and contribute zero model wins. "
                 "Avg excludes any overall-conversation judgment. S1 is an official stage-wise "
                 "conversation score whose prompt includes all three responses; it is not pure A1 accuracy.", "",
                 "## Quality (0–100)", "",
                 "| Method | S1 | S2 | S3 | Avg | ΔAvg vs ReComp | Unresolved |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for row in result["quality"]:
            lines.append(f"| {row['method']} | {row['S1']:.2f} | {row['S2']:.2f} | "
                         f"{row['S3']:.2f} | {row['Avg']:.2f} | "
                         f"{row['delta_Avg_vs_ReComp']:+.2f} | {row['unresolved']} |")
        lines += ["", "## System", "",
                  "| Method | T2 mean | T3 mean | Pooled mean | Pooled p50 | Pooled p95 | Δ TTFT vs ReComp | SSD MB/request | SSD ↓ vs FullLoad |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in result["generation"]["system"]:
            lines.append(f"| {row['method']} | {row['T2_TTFT_mean_ms']:.2f} | "
                         f"{row['T3_TTFT_mean_ms']:.2f} | "
                         f"{row['cache_hit_TTFT_mean_ms']:.2f} | "
                         f"{row['cache_hit_TTFT_p50_ms']:.2f} | "
                         f"{row['cache_hit_TTFT_p95_ms']:.2f} | "
                         f"{row['TTFT_reduction_vs_ReComp_pct']:+.2f}% | "
                         f"{row['SSD_MB_per_request']:.2f} | "
                         f"{row['SSD_reduction_vs_FullLoad_pct']:.2f}% |")
        lines += ["", "## Turn-specific TTFT distribution (ms)", "",
                  "| Method | T2 p50 | T2 p95 | T3 p50 | T3 p95 |",
                  "|---|---:|---:|---:|---:|"]
        for row in result["generation"]["system"]:
            lines.append(f"| {row['method']} | {row['T2_TTFT_p50_ms']:.2f} | "
                         f"{row['T2_TTFT_p95_ms']:.2f} | "
                         f"{row['T3_TTFT_p50_ms']:.2f} | "
                         f"{row['T3_TTFT_p95_ms']:.2f} |")
        lines += ["", "T2 and T3 timings are independent request start-to-first-token intervals; "
                  "the pooled cache-hit statistic combines their individual observations. "
                  "Persistence is a separate one-time synchronous cost and is included in the "
                  "3-turn cumulative session latency table. The OS page cache was conditioned "
                  "with posix_fadvise(DONTNEED) outside the timer, using buffered pread. "
                  "SSD controller cache was not explicitly flushed.", "",
                  "## Diagnostics", "",
                  f"First-pass parsed: {v['first_pass_parsed']}; second-pass recovered: "
                  f"{v['second_pass_recovered']}; unresolved: {v['unresolved']} "
                  f"({v['unresolved_rate']:.2%}).",
                  f"Turn-1 generation agreement: {v['turn1_generation_agreement']}/"
                  f"{v['expected_conversations']}. Full pixel/prompt/token agreement: "
                  f"{v['turn1_full_agreement']}/{v['expected_conversations']}.",
                  f"Conservative clear-repetition flags: {v['degenerate_repetition_requests']} "
                  "requests. Outputs were not altered or regenerated.",
                  f"Inputs above 4096 tokens: {v['above_nominal_4096_requests']}; "
                  f"maximum input: {v['max_input_tokens']} tokens.", "",
                  f"Unique images: {v['unique_image_ids']}; persisted conversation-store "
                  f"instances: {v['persistence_store_instances']}. The latter counts "
                  "independent conversation executions when an image appears more than once.",
                  f"Judge checkpoint files verified against raw records: "
                  f"{v['judge_checkpoints_verified']}.", "",
                  f"The paired bootstrap uses complete conversation clusters, with "
                  f"{result['bootstrap'][0]['resamples']:,} resamples. Quality ΔAvg and "
                  "TTFT reductions are relative to ReComp. "
                  "The 3-turn session table includes synchronous persistence for cache methods. "
                  "Detailed per-turn TTFT, judge, repetition, context, persistence, and paired "
                  "confidence interval tables are adjacent CSV files.", ""]
        (staging / "README.md").write_text("\n".join(lines), encoding="utf-8")
        manifest = {p.name: {"bytes": p.stat().st_size, "sha256": file_hash(p)}
                    for p in sorted(staging.iterdir()) if p.is_file()}
        (staging / "artifact_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"analysis output appeared during publication: {output}")
        staging.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "data/convbench/index.json")
    parser.add_argument("--answer-run", type=Path, required=True)
    parser.add_argument("--judge-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-conversations", type=int, required=True)
    parser.add_argument("--require-full", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    args = parser.parse_args()
    if args.bootstrap_resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    result = analyze(args.index, args.answer_run, args.judge_run,
                     args.expected_conversations, full=args.require_full,
                     bootstrap_resamples=args.bootstrap_resamples)
    publish(result, args.out)
    print(json.dumps(result["validation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

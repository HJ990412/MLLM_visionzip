#!/usr/bin/env python3
"""Validate and summarize ConvBench per-request serving and judge artifacts.

The cache-hit population is the union of independent Turn-2 and Turn-3
requests.  Session cumulative latency is a separate secondary quantity.
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
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mmimpress.cvpr25 import budget_chunk_count  # noqa: E402

METHODS = ("recompute", "fullload", "prefix25", "prefix45")
LABELS = {"recompute": "ReComp", "fullload": "FullLoad",
          "prefix25": "Prefix25", "prefix45": "Prefix45"}
NATIVE_CONTEXT_POLICY = "native_overflow_no_truncation_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def content_hash(value: dict) -> str:
    body = {k: v for k, v in value.items() if k != "artifact_content_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def stats(values: list[float]) -> dict:
    if not values:
        raise ValueError("cannot summarize empty population")
    return {"n": len(values), "mean": statistics.fmean(values),
            "p50": quantile(values, 0.50), "p95": quantile(values, 0.95)}


def _first_k(row: dict, budget: float) -> None:
    n = int(row["n_chunks_total"])
    wanted = list(range(budget_chunk_count(n, budget)))
    selected = row["selected_chunk_ids_per_layer"]
    if not isinstance(selected, list) or not selected:
        raise ValueError("prefix chunk selection missing")
    if any(ids != wanted for ids in selected):
        raise ValueError("prefix request did not read physical first-k chunks")
    if row.get("static_score_calls", 0) or row.get("query_score_calls", 0) or row.get("diversity_calls", 0):
        raise ValueError("online score/diversity call in prefix path")


def _answer_prompt_renderer():
    source = ROOT / "scripts/41_eval_convbench_answers.py"
    spec = importlib.util.spec_from_file_location("convbench_answer_renderer", source)
    if spec is None or spec.loader is None:
        raise ImportError(source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.render_prompt


def selected_dialogs(index: dict, config: dict, n_conversations: int) -> list[dict]:
    all_dialogs = index["conversations"]
    selected_ids = config.get("selected_conversation_ids")
    if selected_ids is None:
        # Runs created before explicit smoke-ID selection used the first N rows.
        return all_dialogs[:n_conversations]
    if len(selected_ids) != n_conversations or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("answer-run selected conversation count/uniqueness mismatch")
    by_id = {dialog["conversation_id"]: dialog for dialog in all_dialogs}
    if len(by_id) != len(all_dialogs) or any(cid not in by_id for cid in selected_ids):
        raise ValueError("answer-run selected conversation ID absent from index")
    selected = [by_id[cid] for cid in selected_ids]
    if config.get("selected_source_ids") is not None and [
        str(dialog["source_id"]) for dialog in selected
    ] != config["selected_source_ids"]:
        raise ValueError("answer-run selected source IDs mismatch")
    return selected


def validate_context_caps(rows: list[dict], config: dict) -> bool:
    """Check the recorded generation budget against each actual request input."""
    policy_id = config.get("context_policy_id")
    if policy_id not in (None, NATIVE_CONTEXT_POLICY):
        raise ValueError(f"unknown context policy: {policy_id}")
    present = ["effective_max_new_tokens" in row for row in rows]
    if not any(present):
        if policy_id is not None:
            raise ValueError("new context policy lacks per-request cap records")
        return False  # Legacy smoke artifacts predate the dynamic cap fields.
    if not all(present):
        raise ValueError("mixed context-cap recording in one answer population")
    for row in rows:
        limit = int(row["text_context_limit"])
        input_tokens = int(row["context_input_tokens"])
        nominal = int(row["nominal_max_new_tokens"])
        effective = int(row["effective_max_new_tokens"])
        if row["turn_id"] == 1 or row["method_key"] == "recompute":
            path_tokens = row["expanded_input_tokens"]
        else:
            path_tokens = row["context_tokens_for_cache_path"]
        expected_effective = (min(nominal, limit - input_tokens + 1)
                              if input_tokens <= limit else nominal)
        if (path_tokens is None or input_tokens != int(path_tokens) or
                limit != int(config["text_context_limit"]) or
                not 1 <= input_tokens or
                (input_tokens > limit and policy_id is None) or
                nominal != int(config["max_new_tokens"]) or
                effective != expected_effective or
                int(row["context_remaining_positions_before_request"]) !=
                limit - input_tokens or
                int(row["context_available_output_tokens"]) !=
                limit - input_tokens + 1 or
                bool(row["generation_cap_clamped_by_context"]) !=
                (effective < nominal) or
                not 1 <= int(row["generated_tokens"]) <= effective or
                bool(row["generation_cap_reached"]) !=
                (int(row["generated_tokens"]) == effective)):
            raise ValueError("inconsistent context cap or output length in answer row")
        if policy_id == NATIVE_CONTEXT_POLICY:
            generated = int(row["generated_tokens"])
            overflow = max(0, input_tokens - limit)
            required = (
                all(field in row for field in (
                    "original_prompt_tokens", "final_prompt_tokens",
                    "input_overflow_tokens", "native_overflow_execution",
                    "truncation_applied", "truncated_tokens",
                    "truncation_source", "preserved_current_question",
                    "preserved_image_marker", "runtime_success",
                    "first_token_success", "last_generated_sequence_position",
                    "last_expected_executed_position",
                    "generation_position_overflow_tokens")) and
                row.get("context_policy_id") == policy_id and
                type(row.get("original_prompt_tokens")) is int and
                row["original_prompt_tokens"] == input_tokens and
                type(row.get("final_prompt_tokens")) is int and
                row["final_prompt_tokens"] == input_tokens and
                type(row.get("input_overflow_tokens")) is int and
                row["input_overflow_tokens"] == overflow and
                row.get("native_overflow_execution") is (overflow > 0) and
                row.get("truncation_applied") is False and
                type(row.get("truncated_tokens")) is int and
                row["truncated_tokens"] == 0 and
                row.get("truncation_source") is None and
                row.get("preserved_current_question") is True and
                row.get("preserved_image_marker") is True and
                row.get("runtime_success") is True and
                row.get("first_token_success") is True and
                row.get("last_generated_sequence_position") ==
                input_tokens + generated - 1 and
                row.get("last_expected_executed_position") ==
                input_tokens + generated - 2 and
                row.get("generation_position_overflow_tokens") ==
                max(0, input_tokens + generated - limit)
            )
            if not required:
                raise ValueError("inconsistent native overflow / zero-truncation request record")
        elif row.get("context_policy_id") is not None:
            raise ValueError("legacy config contains a new-policy answer row")
    return True


def validate_and_load(index_path: Path, run_dir: Path,
                      n_conversations: int) -> tuple[list[dict], list[dict], dict]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config["index_sha256"] != sha256_file(index_path):
        raise ValueError("answer-run index hash mismatch")
    if n_conversations > int(config["limit"]):
        raise ValueError("analysis asks for more conversations than generation config")
    dialogs = selected_dialogs(index, config, n_conversations)
    all_rows: list[dict] = []
    persistence: list[dict] = []
    render_prompt = _answer_prompt_renderer()
    for dialog in dialogs:
        cid = dialog["conversation_id"]
        path = run_dir / "conversations" / f"{cid}.json"
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if artifact.get("artifact_content_sha256") != content_hash(artifact):
            raise ValueError(f"corrupt answer artifact: {path}")
        if artifact["conversation_id"] != cid or artifact["source_row_index"] != dialog["source_row_index"]:
            raise ValueError("answer artifact does not match source conversation")
        rows = artifact["rows"]
        if len(rows) != 12 or {(r["turn_id"], r["method_key"]) for r in rows} != {
            (turn, method) for turn in (1, 2, 3) for method in METHODS
        }:
            raise ValueError(f"incomplete or duplicate method-turn rows: {cid}")
        by_key = {(r["turn_id"], r["method_key"]): r for r in rows}
        store_id = artifact["store_manifest"]["meta_sha256"]
        persisted = artifact["persistence"]
        if (persisted["conversation_id"] != cid or
                persisted["vision_forward_count"] != 1 or
                persisted["separate_visual_prefix_forward_count"] != 0 or
                persisted["saliency_call_count"] != 1 or
                not persisted["capture_from_same_answer1_forward"] or
                not persisted["durable_fsync_completed"] or
                persisted["total_ssd_write_bytes"] <= 0):
            raise ValueError("Turn-1 piggyback persistence validation failed")
        source_method = artifact["source_method_key"]
        source_t1 = by_key[(1, source_method)]
        if not (source_t1["request_finished_at_s"] <=
                persisted["persist_started_at_s"] <
                persisted["store_ready_at_s"] <=
                min(by_key[(2, method)]["request_started_at_s"]
                    for method in METHODS)):
            raise ValueError("persistence timing overlaps cache-hit requests")
        for method in METHODS:
            previous = []
            previous_finished = None
            for turn in (1, 2, 3):
                row = by_key[(turn, method)]
                if row["conversation_id"] != cid or row["method_key"] != method:
                    raise ValueError("answer row identity mismatch")
                if row["previous_answers"] != previous or row["history_policy"] != "same_method_generated_answers":
                    raise ValueError(f"generated-history contamination: {cid}/{method}/T{turn}")
                prompt = row["prompt"]
                questions = [t["question"] for t in dialog["turns"][:turn]]
                if prompt != render_prompt(questions, previous):
                    raise ValueError("generated-history prompt differs from official-formatted causal prompt")
                start = float(row["request_started_at_s"])
                first = float(row["first_token_at_s"])
                finished = float(row["request_finished_at_s"])
                if not (start < first <= finished):
                    raise ValueError("invalid per-request timer ordering")
                if previous_finished is not None and not previous_finished <= start:
                    raise ValueError("method turn timers overlap")
                if not math.isclose(float(row["end_to_end_ttft_ms"]),
                                    (first - start) * 1000, abs_tol=0.1):
                    raise ValueError("TTFT is not this request's first-token interval")
                if not math.isclose(float(row["request_e2e_ms"]),
                                    (finished - start) * 1000, abs_tol=0.1):
                    raise ValueError("request E2E boundary mismatch")
                if row["end_to_end_ttft_ms"] > row["request_e2e_ms"]:
                    raise ValueError("TTFT exceeds full generation time")
                if turn == 1 or method == "recompute":
                    if row["ssd_read_bytes"] != 0 or row["ssd_read_preads"] != 0:
                        raise ValueError("pixel path read SSD KV")
                else:
                    if row["same_physical_store_id"] != store_id:
                        raise ValueError("cache methods used different physical layouts")
                    if row["ssd_read_bytes"] <= 0:
                        raise ValueError("cache request did not read SSD KV")
                    if method == "prefix25":
                        _first_k(row, 0.25)
                    elif method == "prefix45":
                        _first_k(row, 0.45)
                if turn == 1 and (row["vision_forward_count"] != 1 or
                                  row["separate_vision_forward_count"] != 0):
                    raise ValueError("Turn 1 piggyback/vision-forward contract failed")
                previous.append(row["prediction"])
                previous_finished = finished
        for turn in (2, 3):
            p25 = by_key[(turn, "prefix25")]
            p45 = by_key[(turn, "prefix45")]
            if p25["ssd_read_bytes"] >= p45["ssd_read_bytes"]:
                raise ValueError("Prefix25 SSD bytes are not below Prefix45")
        all_rows.extend(rows)
        persistence.append(persisted)
    if len(all_rows) != 12 * n_conversations:
        raise ValueError("logical request population mismatch")
    validate_context_caps(all_rows, config)
    return all_rows, persistence, config


def summarize(rows: list[dict], persistence: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    system, by_turn, session = [], [], []
    persist_by_id = {p["conversation_id"]: p for p in persistence}
    for method in METHODS:
        method_rows = [r for r in rows if r["method_key"] == method]
        hits = [r for r in method_rows if r["turn_id"] in (2, 3)]
        if len(hits) * 3 != len(method_rows) * 2:
            raise ValueError("pooled cache-hit count mismatch")
        ttft = stats([float(r["end_to_end_ttft_ms"]) for r in hits])
        e2e = stats([float(r["request_e2e_ms"]) for r in hits])
        ssd = stats([float(r["ssd_read_bytes"]) / 1e6 for r in hits])
        system.append({"method": LABELS[method], "method_key": method,
                       "cache_hit_requests": ttft["n"],
                       "cache_hit_ttft_mean_ms": ttft["mean"],
                       "cache_hit_ttft_p50_ms": ttft["p50"],
                       "cache_hit_ttft_p95_ms": ttft["p95"],
                       "cache_hit_e2e_mean_ms": e2e["mean"],
                       "ssd_MB_per_request": ssd["mean"],
                       "preads_per_request": statistics.fmean(r["ssd_read_preads"] for r in hits),
                       "ssd_read_mean_ms": statistics.fmean(r["ssd_read_ms"] for r in hits),
                       "scatter_mean_ms": statistics.fmean(r["scatter_ms"] for r in hits),
                       "core_prefill_inclusive_mean_ms": (
                           statistics.fmean(r["prefill_ms"] for r in hits)
                           if method in ("recompute", "fullload") else None),
                       "text_history_prefill_mean_ms": (
                           statistics.fmean(r["prefill_ms"] for r in hits)
                           if method in ("prefix25", "prefix45") else None),
                       "planning_mean_ms": statistics.fmean(r["first_k_planning_ms"] for r in hits)})
        for turn in (1, 2, 3):
            group = [r for r in method_rows if r["turn_id"] == turn]
            value = stats([float(r["end_to_end_ttft_ms"]) for r in group])
            by_turn.append({"method": LABELS[method], "turn_id": turn,
                            "requests": value["n"], "ttft_mean_ms": value["mean"],
                            "ttft_p50_ms": value["p50"], "ttft_p95_ms": value["p95"],
                            "e2e_mean_ms": statistics.fmean(r["request_e2e_ms"] for r in group),
                            "prompt_tokens_mean": statistics.fmean(r["prompt_tokens"] for r in group),
                            "history_tokens_mean": statistics.fmean(r["history_tokens"] for r in group),
                            "previous_answer_tokens_mean": statistics.fmean(r["previous_answer_tokens"] for r in group)})
        by_dialog = defaultdict(list)
        for row in method_rows:
            by_dialog[row["conversation_id"]].append(row)
        per_session = []
        for cid, group in by_dialog.items():
            if len(group) != 3:
                raise ValueError("session has fewer than three method requests")
            if method == "recompute":
                total = sum(float(r["request_e2e_ms"]) for r in group)
            else:
                # All cache arms share one physical store captured from a
                # designated counterfactual Turn-1 pixel request.  Charge its
                # measured capture path to each cache-method session.
                source = next(r for r in rows if r["conversation_id"] == cid
                              and r["persistence_source_request"])
                total = float(source["request_e2e_ms"]) + sum(
                    float(r["request_e2e_ms"]) for r in group
                    if r["turn_id"] in (2, 3))
                total += float(persist_by_id[cid]["persist_ms"])
            per_session.append(total)
        session.append({"method": LABELS[method], "conversations": len(per_session),
                        "cumulative_session_e2e_mean_ms": statistics.fmean(per_session),
                        "includes_one_time_persistence": method != "recompute",
                        "cache_method_turn1_policy": (
                            "designated_same_forward_capture_source_T1"
                            if method != "recompute" else "own_normal_pixel_T1")})
    recomp_ttft = next(item["cache_hit_ttft_mean_ms"] for item in system
                       if item["method_key"] == "recompute")
    full_ssd = next(item["ssd_MB_per_request"] for item in system
                    if item["method_key"] == "fullload")
    for item in system:
        item["ttft_reduction_vs_recomp_pct"] = (
            100 * (1 - item["cache_hit_ttft_mean_ms"] / recomp_ttft))
        item["ssd_reduction_vs_fullload_pct"] = (
            100 * (1 - item["ssd_MB_per_request"] / full_ssd))
    return system, by_turn, session


def summarize_persistence(persistence: list[dict]) -> list[dict]:
    fields = (
        "saliency_reduction_ms", "saliency_d2h_ms", "permutation_ms",
        "kv_repack_ms", "ssd_write_ms", "fsync_ms", "persist_ms",
        "total_ssd_write_bytes",
    )
    rows = []
    for field in fields:
        values = [float(item[field]) for item in persistence]
        if field == "total_ssd_write_bytes":
            values = [value / 1e6 for value in values]
        s = stats(values)
        rows.append({"component": field, "unit": "MB" if field.endswith("bytes") else "ms",
                     "images": len(values), "mean": s["mean"],
                     "p50": s["p50"], "p95": s["p95"]})
    return rows


def summarize_context_caps(rows: list[dict], policy_id: str | None = None) -> list[dict]:
    summary = []
    for method in METHODS:
        for turn in (1, 2, 3):
            group = [r for r in rows if r["method_key"] == method and r["turn_id"] == turn]
            item = {
                "method": LABELS[method], "turn_id": turn, "requests": len(group),
                "context_input_tokens_max": max(r["context_input_tokens"] for r in group),
                "context_remaining_positions_min": min(
                    r["context_remaining_positions_before_request"] for r in group),
                "effective_max_new_tokens_min": min(r["effective_max_new_tokens"] for r in group),
                "context_clamped_requests": sum(
                    bool(r["generation_cap_clamped_by_context"]) for r in group),
                "cap_reached_requests": sum(
                    bool(r["generation_cap_reached"]) for r in group),
            }
            if policy_id == NATIVE_CONTEXT_POLICY:
                item.update({
                    "native_input_overflow_requests": sum(
                        r["native_overflow_execution"] for r in group),
                    "max_input_overflow_tokens": max(
                        r["input_overflow_tokens"] for r in group),
                    "max_generation_position_overflow_tokens": max(
                        r["generation_position_overflow_tokens"] for r in group),
                    "truncation_applied_requests": sum(
                        r["truncation_applied"] for r in group),
                    "truncated_tokens_total": sum(
                        r["truncated_tokens"] for r in group),
                })
            summary.append(item)
    return summary


def context_policy_request_rows(rows: list[dict], policy_id: str) -> list[dict]:
    """Expose validated original/final input and overflow for every request."""
    if policy_id != NATIVE_CONTEXT_POLICY:
        raise ValueError("per-request context policy export requires native policy")
    fields = (
        "conversation_id", "method", "method_key", "turn_id",
        "context_policy_id", "original_prompt_tokens", "text_context_limit",
        "input_overflow_tokens", "native_overflow_execution",
        "truncation_applied", "truncated_tokens", "truncation_source",
        "final_prompt_tokens", "preserved_current_question",
        "preserved_image_marker", "nominal_max_new_tokens",
        "effective_max_new_tokens", "generation_cap_clamped_by_context",
        "generated_tokens", "last_generated_sequence_position",
        "last_expected_executed_position",
        "generation_position_overflow_tokens", "runtime_success",
        "first_token_success", "end_to_end_ttft_ms",
    )
    return [{field: row[field] for field in fields} for row in rows]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def _judge_answer_input_hash(run_dir: Path, dialogs: list[dict]) -> str:
    digest = hashlib.sha256()
    digest.update(sha256_file(run_dir / "config.json").encode())
    for dialog in dialogs:
        cid = dialog["conversation_id"]
        path = run_dir / "conversations" / f"{cid}.json"
        digest.update(cid.encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def judge_analysis(judge_run: Path, dialogs: list[dict], quality: dict,
                   *, bootstrap_resamples: int = 10_000,
                   judge_turns: tuple[str, ...] = ("_first_turn", "_second_turn", "_third_turn",
                                                   "_overall_conversation")) -> tuple[list[dict], list[dict], dict, dict]:
    """Recompute fixed-population scores from immutable raw judge decisions.

    The archived judge quality_summary may have used a valid-A/B denominator.
    Its scores are deliberately not used for this analysis.
    """
    official_turns = ("_first_turn", "_second_turn", "_third_turn", "_overall_conversation")
    if judge_turns not in (official_turns[:3], official_turns):
        raise ValueError("judge turns must be the three stages or all four official turns")
    turns = judge_turns
    metrics = ("S1", "S2", "S3", "overall_official_secondary")[:len(turns)]
    ordered_ids = [d["conversation_id"] for d in dialogs]
    source_by_id = {d["conversation_id"]: d["source_row_index"] for d in dialogs}
    records = [json.loads(line) for line in
               (judge_run / "judge_raw.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    expected_decisions = len(dialogs) * len(METHODS) * len(turns)
    if len(records) > expected_decisions:
        raise ValueError("judge raw population exceeds expected size")
    legacy = any(record.get("parse_stage") not in
                 ("first_pass", "second_pass", "unresolved") for record in records)
    stages = {stage: sum(record.get("parse_stage") == stage for record in records)
              for stage in ("first_pass", "second_pass", "unresolved")}
    if not legacy and any(record.get("parse_stage") not in stages for record in records):
        raise ValueError("unknown new judge parse stage")
    if not legacy and any(
        (record.get("winner") in ("A", "B")) !=
        (record.get("parse_stage") in ("first_pass", "second_pass")) or
        (record.get("winner") in (None, "C")) !=
        (record.get("parse_stage") == "unresolved")
        for record in records
    ):
        raise ValueError("winner and judge parse stage disagree")
    expected_keys = {(method, cid, turn) for method in LABELS.values()
                     for cid in ordered_ids for turn in turns}
    keyed = {}
    for record in records:
        key = (record["method"], record["conversation_id"], record["judge_turn"])
        if (key in keyed or key not in expected_keys or
                record["source_row_index"] != source_by_id.get(record["conversation_id"])):
            raise ValueError("duplicate/unexpected judge record")
        keyed[key] = record
    missing_total = len(expected_keys - set(keyed))
    rows = []
    matrix = {}
    summaries = {}
    for method in LABELS.values():
        per_turn = []
        summary = {"n_conversations": len(dialogs)}
        valid_by_turn = {}
        unresolved_by_turn = {}
        for turn, metric in zip(turns, metrics, strict=True):
            wins = []
            unresolved = 0
            legacy_c = 0
            first_pass = 0
            second_pass = 0
            missing = 0
            for cid in ordered_ids:
                record = keyed.get((method, cid, turn))
                if record is None:
                    wins.append(0)
                    missing += 1
                    continue
                winner = record["winner"]
                if winner not in ("A", "B", "C", None):
                    raise ValueError("invalid judge winner")
                if record["position"] not in (0, 1):
                    raise ValueError("invalid judge position")
                model_letter = "A" if record["position"] == 0 else "B"
                won = int(winner == model_letter)
                recorded_win = record["model_wins"]
                if ((recorded_win not in (None, False) if winner in (None, "C") else
                     type(recorded_win) is not bool or recorded_win != bool(won))):
                    raise ValueError("judge raw model_wins field inconsistent")
                wins.append(won)
                unresolved += int(winner in (None, "C"))
                legacy_c += int(winner == "C")
                first_pass += int(record.get("parse_stage") == "first_pass")
                second_pass += int(record.get("parse_stage") == "second_pass")
            valid = len(wins) - unresolved - missing
            denominator = len(dialogs)
            observed = 100 * sum(wins) / denominator
            if not legacy and not missing_total:
                if (quality[method]["valid_decisions_by_turn"][metric] != valid or
                        quality[method]["unresolved_by_turn"][metric] != unresolved):
                    raise ValueError("judge decision counts disagree with raw records")
            summary[metric] = observed
            valid_by_turn[metric] = valid
            unresolved_by_turn[metric] = unresolved
            if metric != "overall_official_secondary":
                per_turn.append(np.array(wins, dtype=np.float64))
                rows.append({"method": method, "metric": metric,
                             "conversations": len(wins), "valid_decisions": valid,
                             "score_denominator": denominator,
                             "model_wins": sum(wins),
                             "first_pass_parsed": first_pass,
                             "second_pass_recovered": second_pass,
                             "unresolved": unresolved, "missing": missing,
                             "legacy_C": legacy_c,
                             "unresolved_rate": unresolved / denominator,
                             "score": observed})
        matrix[method] = per_turn
        summary["Avg"] = sum(summary[m] for m in ("S1", "S2", "S3")) / 3
        if len(turns) == 3:
            summary["overall_official_secondary"] = None
        summary["valid_decisions_by_turn"] = valid_by_turn
        summary["unresolved_by_turn"] = unresolved_by_turn
        summary["total_valid_decisions"] = sum(valid_by_turn.values())
        summary["total_unresolved"] = sum(unresolved_by_turn.values())
        summary["main_unresolved"] = sum(unresolved_by_turn[m] for m in ("S1", "S2", "S3"))
        summary["main_unresolved_rate"] = summary["main_unresolved"] / (3 * denominator)
        summary["missing_by_turn"] = {
            metric: len(dialogs) - valid_by_turn[metric] - unresolved_by_turn[metric]
            for metric in metrics}
        summary["total_missing"] = sum(summary["missing_by_turn"].values())
        summary["main_missing"] = sum(summary["missing_by_turn"][m]
                                      for m in ("S1", "S2", "S3"))
        summaries[method] = summary
        if not legacy and not missing_total and (
            quality[method]["total_valid_decisions"] !=
                summary["total_valid_decisions"] or
            quality[method]["total_unresolved"] !=
                summary["total_unresolved"]
        ):
            raise ValueError("judge total decision counts disagree with raw records")
    unresolved_total = sum(record["winner"] in (None, "C") for record in records)
    main_unresolved = sum(s["main_unresolved"] for s in summaries.values())
    main_expected = len(dialogs) * len(METHODS) * 3
    main_missing = sum(s["main_missing"] for s in summaries.values())
    decision_counts = {
        "total_expected_judge_decisions": expected_decisions,
        "first_pass_parsed": stages["first_pass"],
        "second_pass_recovered": stages["second_pass"],
        "unresolved": unresolved_total,
        "unresolved_rate": unresolved_total / expected_decisions,
        "missing": missing_total,
        "missing_rate": missing_total / expected_decisions,
        "main_expected_judge_decisions": main_expected,
        "main_unresolved": main_unresolved,
        "main_unresolved_rate": main_unresolved / main_expected,
        "main_missing": main_missing,
        "main_missing_rate": main_missing / main_expected,
        "legacy_C": sum(record["winner"] == "C" for record in records),
        "judge_decision_protocol": "fixed_denominator_unresolved_zero_win",
        "quality_complete": main_missing == 0,
        "fixed_denominator_scores_complete": True,
        "all_labels_resolved": not (unresolved_total or missing_total),
        "bootstrap_suppressed_for_unresolved": False,
    }
    rng = np.random.default_rng(1234)
    sampled_indices = rng.integers(0, len(dialogs),
                                   size=(bootstrap_resamples, len(dialogs)))
    bootstrap = []
    full_matrix = matrix["FullLoad"]
    for method in LABELS.values():
        for metric_index, metric in enumerate(("S1", "S2", "S3", "Avg")):
            values = (matrix[method][metric_index] if metric_index < 3 else
                      sum(matrix[method]) / 3)
            full_values = (full_matrix[metric_index] if metric_index < 3 else
                           sum(full_matrix) / 3)
            estimates = 100 * values[sampled_indices].mean(axis=1)
            deltas = 100 * (values - full_values)[sampled_indices].mean(axis=1)
            low, high = np.percentile(estimates, [2.5, 97.5])
            delta_low, delta_high = np.percentile(deltas, [2.5, 97.5])
            bootstrap.append({
                "method": method, "metric": metric,
                "cluster_unit": "conversation", "resamples": bootstrap_resamples,
                "seed": 1234, "score_estimate": 100 * float(values.mean()),
                "score_ci95_low": float(low), "score_ci95_high": float(high),
                "delta_vs_FullLoad_estimate": 100 * float((values - full_values).mean()),
                "delta_vs_FullLoad_ci95_low": float(delta_low),
                "delta_vs_FullLoad_ci95_high": float(delta_high),
            })
    return rows, bootstrap, decision_counts, summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "data/convbench/index.json")
    parser.add_argument("--answer-run", type=Path, required=True)
    parser.add_argument("--judge-run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-conversations", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.expected_conversations <= 577:
        raise ValueError("expected conversations must be 1..577")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("analysis output exists; refusing to replace it")
    rows, persistence, config = validate_and_load(
        args.index, args.answer_run, args.expected_conversations)
    system, per_turn, session = summarize(rows, persistence)
    persistence_summary = summarize_persistence(persistence)
    context_caps_available = validate_context_caps(rows, config)
    context_policy_id = config.get("context_policy_id")
    context_caps = (summarize_context_caps(rows, context_policy_id)
                    if context_caps_available else None)
    context_requests = (context_policy_request_rows(rows, context_policy_id)
                        if context_policy_id == NATIVE_CONTEXT_POLICY else None)
    quality = None
    judge_rows = None
    bootstrap_rows = None
    judge_counts = None
    score_provenance = None
    if args.judge_run is not None:
        judge_config = json.loads((args.judge_run / "config.json").read_text())
        selected = selected_dialogs(
            json.loads(args.index.read_text()), config, args.expected_conversations)
        if (judge_config["index_sha256"] != sha256_file(args.index) or
                judge_config["answer_sha256"] != _judge_answer_input_hash(args.answer_run, selected) or
                judge_config["selected_source_row_indices"] !=
                [d["source_row_index"] for d in selected]):
            raise ValueError("judge run does not belong to this answer population")
        quality = json.loads((args.judge_run / "quality_summary.json").read_text())
        judge_turns = tuple(judge_config.get("judge_turns", (
            "_first_turn", "_second_turn", "_third_turn", "_overall_conversation")))
        if set(quality) != set(LABELS.values()):
            raise ValueError("judge method set mismatch")
        if any(int(quality[m]["n_conversations"]) != args.expected_conversations
               for m in quality):
            raise ValueError("judge score denominator mismatch")
        judge_rows, bootstrap_rows, judge_counts, fixed_quality = judge_analysis(
            args.judge_run, selected, quality, judge_turns=judge_turns)
        if judge_counts["main_missing"]:
            raise ValueError(f"Missing stage judge decisions: {judge_counts['main_missing']}")
        quality_rows = []
        for method in LABELS.values():
            summary = fixed_quality[method]
            row = {"method": method, **{
                key: summary[key] for key in (
                    "S1", "S2", "S3", "Avg", "overall_official_secondary",
                    "n_conversations")}}
            for metric in (("S1", "S2", "S3", "overall_official_secondary")[:len(judge_turns)]):
                row[f"{metric}_score_denominator"] = args.expected_conversations
                row[f"{metric}_valid_decisions"] = summary["valid_decisions_by_turn"][metric]
                row[f"{metric}_unresolved"] = summary["unresolved_by_turn"][metric]
                row[f"{metric}_missing"] = summary["missing_by_turn"][metric]
            row["total_valid_decisions"] = summary["total_valid_decisions"]
            row["total_unresolved"] = summary["total_unresolved"]
            row["total_missing"] = summary["total_missing"]
            row["main_unresolved_rate"] = summary["main_unresolved_rate"]
            quality_rows.append(row)
        score_provenance = {
            "scoring_policy": "fixed_conversation_denominator_unresolved_or_missing_zero_win_v1",
            "judge_raw_sha256": sha256_file(args.judge_run / "judge_raw.jsonl"),
            "archived_judge_quality_summary_sha256": sha256_file(
                args.judge_run / "quality_summary.json"),
            "score_source": "judge_raw.jsonl; archived quality_summary scores ignored",
            "stage_denominator_per_method": args.expected_conversations,
            "avg_formula": "(S1 + S2 + S3) / 3",
            "overall_conversation_role": (
                "not_executed" if len(judge_turns) == 3 else "diagnostic_only"),
            "unresolved_or_missing_model_win": 0,
        }
    validation = {
        "passed": True,
        "expected_conversations": args.expected_conversations,
        "logical_method_turn_requests": len(rows),
        "cache_hit_requests_per_method": args.expected_conversations * 2,
        "total_cache_hit_requests": args.expected_conversations * 2 * 4,
        "independent_T2_T3_timers_verified": True,
        "cache_hit_statistic": "pooled individual T2 and T3 request TTFT samples",
        "one_time_persistence_separate_from_cache_hit_ttft": True,
        "quality_available": quality is not None,
        "judge_decision_counts": judge_counts,
        "context_caps_recorded_and_verified": context_caps_available,
        "context_clamped_requests": (
            sum(row["context_clamped_requests"] for row in context_caps)
            if context_caps is not None else None),
        "generation_cap_reached_requests": (
            sum(row["cap_reached_requests"] for row in context_caps)
            if context_caps is not None else None),
        "context_policy_id": context_policy_id,
        "native_input_overflow_requests": (
            sum(row["native_input_overflow_requests"] for row in context_caps)
            if context_requests is not None else None),
        "max_input_overflow_tokens": (
            max(row["input_overflow_tokens"] for row in context_requests)
            if context_requests is not None else None),
        "max_generation_position_overflow_tokens": (
            max(row["generation_position_overflow_tokens"] for row in context_requests)
            if context_requests is not None else None),
        "truncation_applied_requests": (
            sum(row["truncation_applied"] for row in context_requests)
            if context_requests is not None else None),
        "truncated_tokens_total": (
            sum(row["truncated_tokens"] for row in context_requests)
            if context_requests is not None else None),
        "native_overflow_first_token_successes": (
            sum(row["first_token_success"] for row in context_requests
                if row["native_overflow_execution"])
            if context_requests is not None else None),
        "answer_index_sha256": config["index_sha256"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".convbench-analysis-",
                                     dir=args.out.parent) as temporary:
        staging = Path(temporary)
        write_csv(staging / "system_summary.csv", system)
        write_csv(staging / "per_turn_system.csv", per_turn)
        write_csv(staging / "session_summary.csv", session)
        write_csv(staging / "persistence_summary.csv", persistence_summary)
        write_csv(staging / "persistence_per_conversation.csv", persistence)
        if context_caps is not None:
            write_csv(staging / "context_cap_summary.csv", context_caps)
        if context_requests is not None:
            write_csv(staging / "context_policy_per_request.csv", context_requests)
        if quality is not None:
            write_csv(staging / "quality_summary.csv", quality_rows)
            write_csv(staging / "per_turn_quality.csv", judge_rows)
            write_csv(staging / "judge_summary.csv", judge_rows)
            (staging / "score_provenance.json").write_text(
                json.dumps(score_provenance, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            if bootstrap_rows:
                write_csv(staging / "bootstrap.csv", bootstrap_rows)
        with (staging / "raw.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        (staging / "validation.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        by_turn = {(r["method"], r["turn_id"]): r for r in per_turn}
        lines = [
            "# ConvBench evaluation analysis",
            "",
            f"Validated {args.expected_conversations} native three-turn conversations, "
            f"{len(rows)} logical method-turn requests, and "
            f"{args.expected_conversations * 2} individual cache-hit requests per method.",
            "",
            "The cache-hit TTFT is the pooled distribution of independent Turn-2 "
            "and Turn-3 request start-to-first-token intervals. It is not session "
            "cumulative latency. OS page-cache conditioning via "
            "posix_fadvise(DONTNEED) is outside each timer; buffered pread is used "
            "and SSD controller cache is not explicitly flushed.",
            "",
            "## System results",
            "",
            "| Method | T2 mean (ms) | T3 mean (ms) | Pooled cache-hit mean (ms) | p50 | p95 | SSD MB/request |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for item in system:
            method = item["method"]
            lines.append(
                f"| {method} | {by_turn[(method, 2)]['ttft_mean_ms']:.2f} | "
                f"{by_turn[(method, 3)]['ttft_mean_ms']:.2f} | "
                f"{item['cache_hit_ttft_mean_ms']:.2f} | "
                f"{item['cache_hit_ttft_p50_ms']:.2f} | "
                f"{item['cache_hit_ttft_p95_ms']:.2f} | "
                f"{item['ssd_MB_per_request']:.2f} |"
            )
        if quality is not None:
            lines.extend(("", "## Pairwise judge quality (0–100 model win rate)",
                          "", "| Method | S1 | S2 | S3 | Avg |",
                          "|---|---:|---:|---:|---:|"))
            for method in LABELS.values():
                q = fixed_quality[method]
                formatted = ["NA" if q[k] is None else f"{q[k]:.2f}"
                             for k in ("S1", "S2", "S3", "Avg")]
                lines.append(f"| {method} | {' | '.join(formatted)} |")
            if not judge_counts["legacy_C"]:
                lines.extend((
                    "", f"Expected judge decisions: {judge_counts['total_expected_judge_decisions']}; "
                    f"first-pass parsed: {judge_counts['first_pass_parsed']}; "
                    f"second-pass recovered: {judge_counts['second_pass_recovered']}; "
                    f"unresolved: {judge_counts['unresolved']} "
                    f"({judge_counts['unresolved_rate']:.2%}); "
                    f"missing: {judge_counts['missing']}.",
                    f"Main S1/S2/S3 unresolved: {judge_counts['main_unresolved']}/"
                    f"{judge_counts['main_expected_judge_decisions']} "
                    f"({judge_counts['main_unresolved_rate']:.2%}).",
                    f"Each stage score divides model wins by the fixed "
                    f"{args.expected_conversations} conversations per method. "
                    "Unresolved or missing decisions count as zero wins. "
                    + ("Overall-conversation judgment was not executed. "
                       if len(judge_turns) == 3 else
                       "Overall-conversation judgment is diagnostic and excluded from Avg. ") +
                    "Per-method counts are in per_turn_quality.csv.",
                ))
            if judge_counts["legacy_C"]:
                lines.extend((
                    f"This legacy artifact has {judge_counts['legacy_C']} C fallback "
                    "decisions; these contribute zero wins under the fixed "
                    "denominator and remain separately identified.",
                ))
        lines.extend(("", "## One-time persistence", "",
                      f"Mean synchronous persistence: "
                      f"{next(x['mean'] for x in persistence_summary if x['component'] == 'persist_ms'):.2f} ms/image. "
                      "It is excluded from cache-hit TTFT and included once in "
                      "the secondary session cumulative E2E table.",
                      "", "Generated answers form each method's later-turn history. "
                      "Their lengths may differ and affect TTFT. The cache methods "
                      "share one physical image-only layout captured during a "
                      "designated normal Turn-1 request. The session table charges "
                      "that measured capture request plus persistence to each "
                      "cache-method counterfactual.",
                      "", "Source and cap details are in the run configs, dataset "
                      "provenance, raw per-request rows, and validation.json.", ""))
        if context_caps is not None:
            lines.extend(("## Context cap", "",
                          f"The nominal output cap was reduced in "
                          f"{validation['context_clamped_requests']} of "
                          f"{len(rows)} requests; "
                          f"{validation['generation_cap_reached_requests']} "
                          "requests reached their effective cap. "
                          "See context_cap_summary.csv by method and turn.", ""))
            if context_requests is not None:
                lines.extend((
                    f"Frozen policy `{context_policy_id}` processed "
                    f"{validation['native_input_overflow_requests']} inputs above "
                    f"the nominal {config['text_context_limit']}-token limit "
                    "without truncation or RoPE configuration changes. "
                    "Inputs within the nominal limit retained the dynamic output "
                    "cap; longer inputs used the nominal generation cap. "
                    "Successful first-token records and zero text truncation were "
                    "validated for every request. See context_policy_per_request.csv "
                    "for original/final input lengths and positional overflow.", ""))
        else:
            lines.extend(("## Context cap", "",
                          "This legacy smoke run predates per-request dynamic "
                          "context-cap recording.", ""))
        (staging / "README.md").write_text("\n".join(lines), encoding="utf-8")
        manifest = {p.name: {"size_bytes": p.stat().st_size, "sha256": sha256_file(p)}
                    for p in sorted(staging.iterdir()) if p.is_file()}
        (staging / "artifact_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.out.exists():
            raise FileExistsError("analysis output appeared during publication")
        staging.rename(args.out)
    print(json.dumps({"status": "complete", **validation}, indent=2))


if __name__ == "__main__":
    main()

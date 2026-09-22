"""Isolated, untruncated native-context probe for ConvBench source IDs 494/418.

This diagnostic imports the frozen answer runner but changes its context guard
only in memory.  It never changes the checkpoint, RoPE configuration, prompt,
tokenizer, KV layout, or the production answer runner.  Non-overflow requests
retain the production 1024/context-aware cap.  Overflow requests use a small
explicit output cap by default so the first-token and decode capability can be
checked before considering longer positional extrapolation.

The forward hook checks the logits used for each next-token decision.  This
extra GPU work affects diagnostic latency; recorded TTFTs are validity checks,
not performance estimates.  A fresh, otherwise empty --run-dir is required.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _answer_module():
    path = ROOT / "scripts/41_eval_convbench_answers.py"
    spec = importlib.util.spec_from_file_location("_convbench_native_probe_answers", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def native_cap(production_cap, runner, server, input_tokens: int,
               overflow_max_new_tokens: int) -> dict:
    """Bypass only the production guard, without altering any model setting."""
    limit = int(runner.model.config.text_config.max_position_embeddings)
    if input_tokens <= limit:
        return production_cap(runner, server, input_tokens)
    nominal = int(server.max_new_tokens)
    if not 1 <= overflow_max_new_tokens <= nominal:
        raise ValueError("overflow output cap must be within nominal cap")
    return {
        "nominal_max_new_tokens": nominal,
        "effective_max_new_tokens": overflow_max_new_tokens,
        "generation_cap_clamped_by_context": False,
        "context_input_tokens": input_tokens,
        "context_remaining_positions_before_request": limit - input_tokens,
        "context_available_output_tokens": limit - input_tokens + 1,
    }


class LogitProbe:
    """Retain small GPU booleans until a request ends, then materialize them."""

    def __init__(self):
        self.records: list[dict[str, Any]] = []

    def hook(self, module, args, kwargs, output):
        logits = getattr(output, "logits", None)
        if logits is None or logits.ndim != 3 or logits.shape[-1] < 2:
            self.records.append({"finite": None, "position": None,
                                 "vocab_size": None})
            return
        pos = kwargs.get("cache_position")
        if pos is None:
            pos = kwargs.get("position_ids")
        self.records.append({
            "finite": torch.isfinite(logits[:, -1, :]).all(),
            "position": pos.max().detach() if torch.is_tensor(pos)
            and pos.numel() else None,
            "vocab_size": int(logits.shape[-1]),
        })

    def summary(self, start: int) -> dict:
        values = self.records[start:]
        finite = [None if r["finite"] is None else bool(r["finite"].item())
                  for r in values]
        positions = [int(r["position"].item()) for r in values
                     if r["position"] is not None]
        vocabs = {r["vocab_size"] for r in values if r["vocab_size"] is not None}
        return {
            "model_forward_outputs": len(values),
            "logit_decisions_checked": sum(v is not None for v in finite),
            "all_last_token_logits_finite": bool(values) and
            all(v is True for v in finite),
            "nonfinite_or_missing_logits_at_forwards": [
                i for i, value in enumerate(finite) if value is not True],
            "max_explicit_executed_position": max(positions) if positions
            else None,
            "vocab_sizes_seen": sorted(vocabs),
        }


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()


def _append_journal(path: Path, value: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _install_request_journal(answer, runner, probe: LogitProbe,
                             dialog: dict, journal: Path,
                             cap_events: list[dict]):
    order = answer.method_order(int(dialog["global_conversation_ordinal"]))
    sequence = {"attempts": 0}
    normal = answer._run_normal
    stored = answer._run_stored

    def invoke(original, positional, keywords):
        number = sequence["attempts"]
        sequence["attempts"] += 1
        turn_id, method_key = number // 4 + 1, order[number % 4]
        questions = positional[3]
        prior_answers = positional[4]
        prompt = answer.render_prompt(questions, prior_answers)
        record = {
            "conversation_id": dialog["conversation_id"],
            "source_id": dialog["source_id"],
            "turn_id": turn_id,
            "method_key": method_key,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "original_untruncated_prompt": prompt,
            "original_prompt_tokens": len(runner.processor.tokenizer(
                prompt, truncation=False).input_ids),
            "native_prompt_unchanged": True,
            "truncation_applied": False,
            "rope_scaling_modified": False,
            "max_position_embeddings_modified": False,
        }
        start = len(probe.records)
        start_caps = len(cap_events)
        captured_warnings = []
        try:
            with warnings.catch_warnings(record=True) as found:
                warnings.simplefilter("always")
                try:
                    result = original(*positional, **keywords)
                finally:
                    captured_warnings = [str(w.message) for w in found]
            record["runtime_success"] = True
            record["generated_tokens"] = int(result["generated_tokens"])
            record["first_token_id"] = int(result["first_token_id"])
            record["first_token_success"] = record["generated_tokens"] >= 1
            record["answer"] = str(result["answer"])
            record["nonempty_answer"] = bool(record["answer"].strip())
            record["ttft_ms"] = float(result["ttft_ms"])
            record["request_started_at_s"] = float(result[
                "request_started_at_s"])
            record["first_token_at_s"] = float(result["first_token_at_s"])
            record["effective_max_new_tokens"] = int(result[
                "effective_max_new_tokens"])
            return result
        except Exception as exc:
            record["runtime_success"] = False
            record["first_token_success"] = None
            record["error_type"] = type(exc).__name__
            record["error"] = str(exc)
            record["traceback"] = traceback.format_exc()
            raise
        finally:
            record["warnings"] = captured_warnings
            events = cap_events[start_caps:]
            if len(events) == 1:
                tokens = int(events[0]["context_input_tokens"])
                limit = int(runner.model.config.text_config.max_position_embeddings)
                record.update({
                    "actual_input_tokens": tokens,
                    "nominal_max_position_embeddings": limit,
                    "input_overflow_tokens": max(0, tokens - limit),
                    "final_input_tokens": tokens,
                    "context_cap": events[0],
                })
                if record.get("generated_tokens") is not None:
                    generated = int(record["generated_tokens"])
                    record["last_generated_sequence_position"] = (
                        tokens + generated - 1)
                    record["last_expected_executed_position"] = (
                        tokens + generated - 2)
                    record["generation_position_overflow_tokens"] = max(
                        0, tokens + generated - limit)
            else:
                record["context_cap_event_count"] = len(events)
            record.update(probe.summary(start))
            if record.get("first_token_id") is not None:
                vocab = record["vocab_sizes_seen"]
                record["first_token_id_valid"] = len(vocab) == 1 and (
                    0 <= record["first_token_id"] < vocab[0])
            _append_journal(journal, record)
            print(f"{dialog['conversation_id']} T{turn_id} {method_key}: "
                  f"input={record.get('actual_input_tokens')} "
                  f"overflow={record.get('input_overflow_tokens')} "
                  f"success={record['runtime_success']} "
                  f"finite={record['all_last_token_logits_finite']}",
                  flush=True)

    def wrapped_normal(*positional, **keywords):
        return invoke(normal, positional, keywords)

    def wrapped_stored(*positional, **keywords):
        return invoke(stored, positional, keywords)

    answer._run_normal = wrapped_normal
    answer._run_stored = wrapped_stored
    return normal, stored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-ids", default="494,418")
    parser.add_argument("--expected-index-sha256", required=True)
    parser.add_argument("--overflow-max-new-tokens", type=int, default=32,
                        help="Diagnostic cap only on input>4096 (1..1024)")
    args = parser.parse_args()
    if not 1 <= args.overflow_max_new_tokens <= 1024:
        parser.error("--overflow-max-new-tokens must be 1..1024")
    answer = _answer_module()
    dialogs, index_sha = answer.load_index(ROOT / "data/convbench/index.json")
    if index_sha != args.expected_index_sha256:
        raise ValueError("official index SHA256 mismatch")
    selected, _ = answer.select_workload(
        dialogs, 2, False, args.source_ids)
    if len(selected) != 2:
        raise ValueError("native overflow probe requires exactly two source IDs")
    source = answer.load_source_provenance()
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and (not run_dir.is_dir() or run_dir.is_symlink()
                             or any(run_dir.iterdir())):
        raise FileExistsError("probe run directory must be new or empty")
    run_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = run_dir / "_temporary_visual_kv"
    temp_dir.mkdir()
    journal = run_dir / "request_journal.jsonl"
    journal.touch(exist_ok=False)

    import transformers
    from mmimpress.serve import Server

    runner = answer.LlavaRunner().load()
    text_cfg = runner.model.config.text_config
    config = {
        "kind": "diagnostic_native_overflow_probe",
        "source_ids": [str(d["source_id"]) for d in selected],
        "index_sha256": index_sha,
        **source,
        "model_id": runner.model_id,
        "transformers_version": transformers.__version__,
        "max_position_embeddings": int(text_cfg.max_position_embeddings),
        "rope_scaling": getattr(text_cfg, "rope_scaling", None),
        "tokenizer_model_max_length": int(
            runner.processor.tokenizer.model_max_length),
        "attention_implementation": runner.model.config._attn_implementation,
        "max_new_tokens_normal": 1024,
        "max_new_tokens_on_input_overflow": args.overflow_max_new_tokens,
        "prompt_truncation": False,
        "model_or_rope_configuration_change": False,
        "note": "Finite-logit diagnostic hook adds overhead to TTFT",
    }
    _write_new(run_dir / "config.json", config)
    server = Server(runner, max_new_tokens=1024)
    answer._visdial()._run_unmeasured_warmup(runner, server)

    production_cap = answer._effective_cap
    cap_events: list[dict] = []

    def cap(runner_arg, server_arg, input_tokens):
        result = native_cap(production_cap, runner_arg, server_arg,
                            input_tokens, args.overflow_max_new_tokens)
        cap_events.append(result)
        return result

    answer._effective_cap = cap
    probe = LogitProbe()
    hook = runner.model.register_forward_hook(probe.hook, with_kwargs=True)
    failures = []
    try:
        for dialog in selected:
            cid = answer._safe_id(dialog["conversation_id"])
            original_normal, original_stored = _install_request_journal(
                answer, runner, probe, dialog, journal, cap_events)
            started = time.time()
            try:
                artifact = answer._run_conversation(
                    runner, server, dialog, index_sha, temp_dir)
                _write_new(run_dir / "conversations" / f"{cid}.json", artifact)
                print(f"{cid}: 12 native probe requests complete", flush=True)
            except Exception as exc:
                failure = {"conversation_id": cid,
                           "error_type": type(exc).__name__,
                           "error": str(exc),
                           "traceback": traceback.format_exc(),
                           "elapsed_s": time.time() - started}
                failures.append(failure)
                _write_new(run_dir / f"failure_{cid}.json", failure)
                print(f"{cid}: native probe failed: {exc}", flush=True)
            finally:
                answer._run_normal = original_normal
                answer._run_stored = original_stored
                # This is a diagnostic-only directory.  The entire transient
                # Visual KV is removed even when the native forward fails.
                store = temp_dir / cid
                if store.exists():
                    shutil.rmtree(store)
    finally:
        hook.remove()
    request_records = [json.loads(line) for line in journal.read_text().splitlines()]
    expected_grid = {
        (str(dialog["conversation_id"]), turn, method)
        for dialog in selected for turn in (1, 2, 3)
        for method in answer.METHOD_KEYS
    }
    actual_grid = {
        (str(r["conversation_id"]), int(r["turn_id"]), str(r["method_key"]))
        for r in request_records
    }
    summary = {
        "source_ids": config["source_ids"],
        "attempted_requests": len(request_records),
        "successful_requests": sum(bool(r["runtime_success"])
                                   for r in request_records),
        "overflow_requests": sum(r.get("input_overflow_tokens", 0) > 0
                                 for r in request_records),
        "all_successful_logits_finite": all(
            r["all_last_token_logits_finite"] for r in request_records
            if r["runtime_success"]),
        "all_successful_first_tokens_valid": all(
            r.get("first_token_success") is True and
            r.get("first_token_id_valid") is True
            for r in request_records if r["runtime_success"]),
        "all_successful_answers_nonempty": all(
            r.get("nonempty_answer") is True for r in request_records
            if r["runtime_success"]),
        "all_successful_ttft_positive": all(
            r.get("ttft_ms", 0) > 0 and
            r.get("first_token_at_s", 0) > r.get("request_started_at_s", 0)
            for r in request_records if r["runtime_success"]),
        "complete_method_turn_grid": actual_grid == expected_grid and
        len(request_records) == len(actual_grid),
        "max_observed_input_tokens": max(
            (r.get("actual_input_tokens", 0) for r in request_records),
            default=None),
        "max_observed_generated_sequence_position": max(
            (r.get("last_generated_sequence_position", 0)
             for r in request_records), default=None),
        "failures": failures,
        "all_24_requests_completed": len(request_records) == 24 and
        not failures and all(r["runtime_success"] for r in request_records),
    }
    summary["probe_valid"] = all(summary[key] for key in (
        "all_24_requests_completed", "all_successful_logits_finite",
        "all_successful_first_tokens_valid", "all_successful_answers_nonempty",
        "all_successful_ttft_positive", "complete_method_turn_grid"))
    _write_new(run_dir / "summary.json", summary)
    return 0 if summary["probe_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

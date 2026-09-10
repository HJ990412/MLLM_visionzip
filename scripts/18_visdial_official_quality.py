"""VisDial v1.0 candidate-ranking quality evaluation (no TTFT timing).

This runner is deliberately separate from ``14_eval_multiturn.py``.  It
computes the official sparse retrieval metrics from the 100 answer candidates
at every evaluated round and NDCG at the densely annotated validation round.
It does not report TTFT, decode latency, E2E latency, or SSD timing.

For one (method, dialog, turn), the multimodal prompt is selected/prefilled
exactly once.  Each candidate is then scored as

    sum log p(candidate tokens + EOS | image, caption, gold history, question)

without length normalisation.  Candidate branches are evaluated sequentially
and ``DynamicCache.crop(prompt_length)`` restores the common prompt cache after
every branch.  This avoids copying the very large visual prefix cache 100
times.

The stored-KV arms reuse the production ImageContext/PrefixCache and the exact
LayerSelector/CVPR25ChunkSelector implementations used by the system runner.
The ReComp arm prefills the prompt from pixels once, then uses the same
candidate-branching procedure.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmimpress.model import LlavaRunner
from mmimpress.cvpr25 import budget_chunk_count
from mmimpress.multiturn import (load_canonical, ndcg, resolve_image_path,
                                 retrieval_metrics, sha256_file,
                                 visdial_prompt)
from mmimpress.serve import (BIAS, CVPR25ChunkSelector, ImageContext,
                             LayerSelector, PrefixCache, Server, load_static,
                             suffix_ids_from_prompt)
from mmimpress.store import IOCounter


SCHEMA_VERSION = "visdial-official-quality-v1"
OFFICIAL_STARTER_COMMIT = "5844f3d5a575e9ec1c1684feb760e7de5c912beb"
OFFICIAL_STARTER_URL = (
    "https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch/"
    f"tree/{OFFICIAL_STARTER_COMMIT}"
)
METHODS = {
    "recompute": ("ReComp", None),
    "fullload": ("FullLoad", 1.0),
    "sparsevlm": ("SparseVLM 25%", 0.25),
    "static_diverse@25": ("Static+Diverse 25%", 0.25),
    "static_diverse@50": ("Static+Diverse 50%", 0.50),
}
DEFAULT_METHODS = ",".join(METHODS)
_FORBIDDEN_LATENCY_FIELDS = {
    "ttft", "ttft_ms", "decode_latency", "decode_ms", "e2e_latency",
    "e2e_ms", "prefill_ms", "selector_ms", "ssd_read_ms", "model_ms",
}


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _stable_json_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return _sha_text(payload)


def _store_content_snapshot(manifest):
    """Immutable build/reorder/static semantics from a mutable manifest.

    ``pipeline_manifest.json`` also receives timestamps from later launcher
    stages.  Comparing its whole-file digest would reject a safe resume after
    such an update, while comparing no store identity could silently mix
    different KV layouts.  This mirrors the system runner's content subset.
    """
    stages = manifest["stages"]
    build = stages["build"]
    static = stages.get("static", {})
    return {
        "index_sha256": manifest["index_sha256"],
        "store_index_sha256": manifest["store_index_sha256"],
        "image_ids": manifest["image_ids"],
        "calibration_policy": manifest["calibration_policy"],
        "future_turn_calibration_count":
            manifest["future_turn_calibration_count"],
        "calibration_records": manifest["calibration_records"],
        "reorder": manifest["reorder"],
        "static_selector": manifest.get("static_selector"),
        "build": {
            "image_kv_build_count": build["image_kv_build_count"],
            "full_visual_kv_bytes": build["full_visual_kv_bytes"],
        },
        "static": {
            "done": bool(static.get("done", False)),
            "static_metadata_build_count":
                static.get("static_metadata_build_count"),
            "artifacts": static.get("artifacts"),
        },
    }


def _sha_ids(ids: torch.Tensor) -> str:
    arr = ids.detach().cpu().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _candidate_sha(candidates) -> str:
    payload = json.dumps(list(candidates), ensure_ascii=False,
                         separators=(",", ":"))
    return _sha_text(payload)


def _source_image_id(dialog) -> int:
    """Official VisDial JSON expects the numeric validation image id."""
    raw = str(dialog["image_ids"][0]).rsplit(":", 1)[-1]
    return int(raw)


def _selected_turns(dialogs, max_turns):
    for dialog in dialogs:
        n = min(len(dialog["turns"]),
                max_turns if max_turns is not None else len(dialog["turns"]))
        for turn in dialog["turns"][:n]:
            yield dialog, turn


def _validate_visdial(dialogs, max_turns=None):
    """CPU-only validation of the official candidate/dense representation."""
    assert dialogs, "empty canonical index"
    n_turns = n_dense = 0
    dense_by_round = defaultdict(int)
    for dialog in dialogs:
        assert dialog["dataset"] == "visdial_v1.0_val", dialog["dataset"]
        assert len(dialog["images"]) == len(dialog["image_ids"]) == 1
        assert len(dialog["turns"]) == 10, dialog["dialog_id"]
        assert _source_image_id(dialog) >= 0
        dense_turns = [turn for turn in dialog["turns"]
                       if "dense_relevance" in turn]
        assert len(dense_turns) == 1, \
            f"expected one dense validation round for {dialog['dialog_id']}"
        for _, turn in _selected_turns([dialog], max_turns):
            ti = int(turn["turn_id"])
            candidates = turn["candidate_answers"]
            candidate_indices = turn.get("candidate_answer_indices")
            assert len(candidates) == 100, (dialog["dialog_id"], ti)
            assert len(set(candidates)) == 100, \
                f"duplicate candidate text at {dialog['dialog_id']} turn {ti}"
            if candidate_indices is not None:
                assert len(candidate_indices) == len(set(candidate_indices)) == 100
            gt = int(turn["gt_index"])
            assert 0 <= gt < 100
            assert candidates[gt] == turn["gold_answer"]
            if "dense_relevance" in turn:
                assert len(turn["dense_relevance"]) == 100
                assert all(math.isfinite(float(x)) and 0 <= float(x) <= 1
                           for x in turn["dense_relevance"])
                assert any(float(x) > 0 for x in turn["dense_relevance"]), \
                    f"empty dense relevance at {dialog['dialog_id']} turn {ti}"
                n_dense += 1
                dense_by_round[ti] += 1
            _assert_causal_prompt(dialog, ti)
            n_turns += 1
    return {
        "passed": True,
        "n_dialogs": len(dialogs),
        "n_turns": n_turns,
        "n_candidates": n_turns * 100,
        "n_dense_rounds": n_dense,
        "dense_round_counts": dict(sorted(dense_by_round.items())),
        "causal_prompt_assertions": n_turns,
    }


def _assert_causal_prompt(dialog, turn_id: int) -> str:
    """Prove that selector-visible text cannot depend on forbidden fields.

    Previous gold answers are intentionally preserved because teacher-forced
    history is the primary protocol.  Candidate lists at every observed turn,
    the current gold answer, and every future question/answer are poisoned;
    the current prompt must remain byte-for-byte identical.
    """
    prompt = visdial_prompt(dialog, turn_id)
    poisoned = copy.deepcopy(dialog)
    marker = f"__VISDIAL_FORBIDDEN_FUTURE_{dialog['dialog_id']}_{turn_id}__"
    for turn in poisoned["turns"]:
        ti = int(turn["turn_id"])
        # Candidates, gt indices, and dense labels are never prompt inputs,
        # including for prior turns.
        turn["candidate_answers"] = [f"{marker}:candidate:{ti}:{j}"
                                     for j in range(100)]
        if "candidate_answer_indices" in turn:
            turn["candidate_answer_indices"] = list(range(100000, 100100))
        turn["gt_index"] = 99
        if "dense_relevance" in turn:
            turn["dense_relevance"] = [0.0] * 100
        if ti == turn_id:
            turn["gold_answer"] = f"{marker}:current-gold"
        elif ti > turn_id:
            turn["question"] = f"{marker}:future-question:{ti}"
            turn["gold_answer"] = f"{marker}:future-answer:{ti}"
    assert visdial_prompt(poisoned, turn_id) == prompt, \
        "current gold/candidates/future turns leaked into selector prompt"
    assert marker not in prompt
    return prompt


def _candidate_token_ids(tokenizer, prompt: str, candidates):
    """Tokenise answer continuations at the exact prompt boundary.

    Tokenising the concatenated string, rather than each answer in isolation,
    preserves SentencePiece's word-boundary semantics.  The explicit prefix
    assertion is essential: if tokenisation changed a prompt token, branching
    from a prefilled cache would not represent the concatenated sequence.
    """
    base = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    eos = tokenizer.eos_token_id
    assert eos is not None, "candidate likelihood requires an EOS token"
    assert not base or base[-1] != eos, \
        "tokenizer unexpectedly appends EOS to the bare prompt"
    result = []
    for ci, candidate in enumerate(candidates):
        full = tokenizer(prompt + " " + str(candidate),
                         return_tensors="pt").input_ids[0].tolist()
        assert full[:len(base)] == base, \
            f"candidate {ci} changes tokenisation before the cache boundary"
        continuation = full[len(base):]
        assert eos not in continuation, \
            f"candidate {ci} contains an EOS special token"
        continuation.append(int(eos))
        assert continuation and continuation[-1] == eos
        result.append(continuation)
    assert len(result) == 100
    return result


def _ranks_from_scores(scores):
    """Original-option-order 1-based ranks with stable index tie-breaking."""
    assert len(scores) == 100
    assert all(math.isfinite(float(x)) for x in scores)
    order = sorted(range(100), key=lambda i: (-float(scores[i]), i))
    ranks = [0] * 100
    for rank, original_index in enumerate(order, 1):
        ranks[original_index] = rank
    assert sorted(ranks) == list(range(1, 101))
    return ranks


def _cache_length(cache) -> int:
    assert hasattr(cache, "get_seq_length") and hasattr(cache, "crop"), \
        "this scorer requires transformers.DynamicCache with crop()"
    return int(cache.get_seq_length())


def _selection_summary(method_key, selector):
    if method_key == "recompute":
        return None
    stats = selector.stats()
    keep = ("n_chunks_selected", "n_chunks_total", "touched_chunk_fraction",
            "logical_kv_ratio", "fallback_rate", "probe_rate",
            "mean_jaccard", "k_keep")
    return {key: stats.get(key) for key in keep if key in stats}


@torch.no_grad()
def _stored_prompt_prefill(method_key, runner, server, ctx, static,
                           suffix_ids, seed, image_id):
    """Select/load and prefill one stored prompt; candidates are not accepted."""
    assert method_key in METHODS and method_key != "recompute"
    device = runner.model.device
    suffix_ids = suffix_ids.to(device)
    assert suffix_ids.ndim == 1 and suffix_ids.numel() > 0
    counter = IOCounter()  # Reads are needed, but timings are never exported.
    assert isinstance(ctx.cache, PrefixCache)
    cache = ctx.cache.new_request()
    prefix_len = int(ctx.meta["prefix_len"])

    if method_key == "fullload":
        empty = torch.empty(0, dtype=torch.long, device=device)
        selector = LayerSelector(runner, ctx, empty, ratio=server.ratio,
                                 mode="fullload", counter=counter)
        with selector:
            output = _text_prefill(runner, cache, suffix_ids, prefix_len)
    elif method_key == "sparsevlm":
        rater_rows = server.raters(ctx, suffix_ids)
        selector = LayerSelector(runner, ctx, rater_rows, ratio=0.25,
                                 mode="impress", counter=counter)
        with selector:
            output = _text_prefill(runner, cache, suffix_ids, prefix_len)
    else:
        budget = METHODS[method_key][1]
        assert budget in (0.25, 0.50)
        selector = CVPR25ChunkSelector(
            runner, ctx, static, budget=budget, mode="static_diverse",
            sep_policy="sidecar", diverse_frac=0.25, counter=counter,
            seed=seed, image_id=image_id)
        # static_diverse is query-independent.  Passing no text embedding is
        # both the production behaviour and a structural leakage barrier.
        selector.prepare(None)
        expected_chunks = budget_chunk_count(
            int(ctx.meta["n_chunks_per_layer"]), float(budget))
        assert len(selector.chunks_per_layer) == int(ctx.meta["num_layers"])
        assert all(int(value) == expected_chunks
                   for value in selector.chunks_per_layer), \
            (method_key, selector.chunks_per_layer, expected_chunks)
        output = _text_prefill(runner, cache, suffix_ids, prefix_len)

    returned = output.past_key_values
    assert returned is cache, "model replaced the mutable prompt DynamicCache"
    expected = prefix_len + int(suffix_ids.numel())
    assert _cache_length(cache) == expected, \
        (_cache_length(cache), expected, method_key)
    # Clone the single retained row so the view cannot keep the full
    # prompt-by-vocabulary logits allocation alive through 100 branches.
    return (cache, output.logits[0, -1].clone(),
            _selection_summary(method_key, selector))


@torch.no_grad()
def _text_prefill(runner, cache, suffix_ids, prefix_len):
    device = runner.model.device
    n = int(suffix_ids.numel())
    positions = torch.arange(prefix_len, prefix_len + n, device=device)
    return runner.model(
        input_ids=suffix_ids.unsqueeze(0),
        attention_mask=torch.ones(1, prefix_len + n, dtype=torch.long,
                                  device=device),
        position_ids=positions.unsqueeze(0), cache_position=positions,
        past_key_values=cache, use_cache=True, return_dict=True,
        logits_to_keep=1)


@torch.no_grad()
def _recompute_prompt_prefill(runner, encoded):
    """Pixel-to-prompt full recomputation once, without candidate tokens."""
    encoded = runner.to_device(encoded)
    output = runner.model(**encoded, use_cache=True, return_dict=True,
                          logits_to_keep=1)
    cache = output.past_key_values
    assert hasattr(cache, "crop"), \
        "ReComp must return DynamicCache for candidate branching"
    expected = int(encoded["input_ids"].shape[1])
    assert _cache_length(cache) == expected, (_cache_length(cache), expected)
    return cache, output.logits[0, -1].clone(), None


@torch.no_grad()
def _score_candidates(runner, cache, prompt_last_logits, candidate_ids,
                      max_context, progress=None):
    """Branch 100 candidates from one prompt cache and restore after each."""
    device = runner.model.device
    base_len = _cache_length(cache)
    base_log_probs = F.log_softmax(prompt_last_logits.float(), dim=-1)
    scores = []
    for ci, ids in enumerate(candidate_ids):
        assert _cache_length(cache) == base_len
        target = torch.tensor(ids, dtype=torch.long, device=device)
        assert target.ndim == 1 and target.numel() >= 1
        assert base_len + int(target.numel()) <= max_context, \
            (base_len, int(target.numel()), max_context, ci)
        score = base_log_probs[target[0]]
        try:
            if target.numel() > 1:
                branch_input = target[:-1]
                n = int(branch_input.numel())
                positions = torch.arange(base_len, base_len + n,
                                         device=device)
                output = runner.model(
                    input_ids=branch_input.unsqueeze(0),
                    attention_mask=torch.ones(
                        1, base_len + n, dtype=torch.long, device=device),
                    position_ids=positions.unsqueeze(0),
                    cache_position=positions, past_key_values=cache,
                    use_cache=True, return_dict=True)
                assert output.past_key_values is cache, \
                    "candidate forward replaced the mutable DynamicCache"
                log_probs = F.log_softmax(output.logits[0].float(), dim=-1)
                score = score + log_probs.gather(
                    1, target[1:].unsqueeze(1)).sum()
            value = float(score.item())
            assert math.isfinite(value), (ci, value)
            scores.append(value)
        finally:
            # This is the branch point: never let candidate i become history
            # for candidate i+1.
            cache.crop(base_len)
            assert _cache_length(cache) == base_len
        if progress is not None:
            progress(ci + 1, len(candidate_ids))
    assert len(scores) == len(candidate_ids) == 100
    return scores


def _load_existing(path):
    rows = []
    if path.exists():
        with open(path) as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid {path}:{line_no}: {exc}") from exc
    return rows


def _metrics_for_rows(rows):
    gt_ranks = [int(row["gt_rank"]) for row in rows]
    dense = [float(row["ndcg"]) for row in rows
             if row["ndcg"] is not None]
    output = retrieval_metrics(gt_ranks)
    output.update({
        "ndcg": float(np.mean(dense)) if dense else None,
        "n_sparse_rounds": len(gt_ranks),
        "n_dense_rounds": len(dense),
    })
    selections = [row.get("selection") for row in rows
                  if isinstance(row.get("selection"), dict)]
    for field in ("logical_kv_ratio", "touched_chunk_fraction"):
        values = [float(item[field]) for item in selections
                  if item.get(field) is not None]
        output[f"{field}_mean"] = (
            float(np.mean(values)) if values else None)
    return output


def _summarise(rows, method_keys):
    output = {}
    for method_key in method_keys:
        selected = [r for r in rows if r["method_key"] == method_key]
        overall = _metrics_for_rows(selected)
        by_turn = {}
        for turn_id in sorted({int(r["turn_id"]) for r in selected}):
            group = [r for r in selected if int(r["turn_id"]) == turn_id]
            by_turn[str(turn_id)] = _metrics_for_rows(group)
        output[method_key] = {
            "method": METHODS[method_key][0],
            "budget": METHODS[method_key][1],
            "overall": overall,
            "by_turn": by_turn,
        }
    if "fullload" in output:
        quality_fields = ("mrr", "r@1", "r@5", "r@10", "mean_rank", "ndcg")

        def add_deltas(metric, baseline):
            for field in quality_fields:
                value, reference = metric.get(field), baseline.get(field)
                metric[f"{field}_delta_vs_fullload"] = (
                    float(value) - float(reference)
                    if value is not None and reference is not None else None)

        full = output["fullload"]
        for item in output.values():
            add_deltas(item["overall"], full["overall"])
            for turn_id, metric in item["by_turn"].items():
                baseline = full["by_turn"].get(turn_id)
                if baseline is not None:
                    add_deltas(metric, baseline)
    return output


def _format_metric(value, digits=4):
    return "--" if value is None else f"{float(value):.{digits}f}"


def _selection_errors(method_key, selection):
    """Validate the method-specific selector provenance stored in one row."""
    if method_key == "recompute":
        return [] if selection is None else ["ReComp selection must be null"]
    if not isinstance(selection, dict):
        return ["stored-arm selection must be an object"]

    errors = []

    def finite(name):
        value = selection.get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(float(value))):
            errors.append(f"{name} must be finite numeric")
            return None
        return float(value)

    touched = finite("touched_chunk_fraction")
    logical = finite("logical_kv_ratio")
    fallback = finite("fallback_rate")
    if touched is not None and not 0.0 < touched <= 1.0:
        errors.append("touched_chunk_fraction must be in (0, 1]")
    if logical is not None and not 0.0 < logical <= 1.0:
        errors.append("logical_kv_ratio must be in (0, 1]")
    if fallback is not None and not 0.0 <= fallback <= 1.0:
        errors.append("fallback_rate must be in [0, 1]")

    if method_key == "fullload":
        if touched is not None and not math.isclose(touched, 1.0):
            errors.append("FullLoad touched_chunk_fraction must equal 1")
        if logical is not None and not math.isclose(logical, 1.0):
            errors.append("FullLoad logical_kv_ratio must equal 1")
        if fallback is not None and not math.isclose(fallback, 0.0):
            errors.append("FullLoad fallback_rate must equal 0")
    elif method_key == "sparsevlm":
        probe = finite("probe_rate")
        k_keep = finite("k_keep")
        if probe is not None and not 0.0 <= probe <= 1.0:
            errors.append("SparseVLM probe_rate must be in [0, 1]")
        if (probe is not None and fallback is not None and
                not math.isclose(probe + fallback, 1.0, abs_tol=1e-9)):
            errors.append("SparseVLM probe_rate + fallback_rate must equal 1")
        if k_keep is not None and (k_keep < 1 or not k_keep.is_integer()):
            errors.append("SparseVLM k_keep must be a positive integer")
        if logical is not None and logical > 0.25 + 1e-9:
            errors.append("SparseVLM logical_kv_ratio exceeds its 25% budget")
        if (logical is not None and touched is not None and
                touched + 1e-9 < logical):
            errors.append("SparseVLM touched chunks cannot trail logical keep")
    elif method_key.startswith("static_diverse"):
        selected = finite("n_chunks_selected")
        total = finite("n_chunks_total")
        budget = METHODS[method_key][1]
        if total is not None and (total < 1 or not total.is_integer()):
            errors.append("Static+Diverse n_chunks_total must be a positive integer")
        if selected is not None and (selected < 1 or not selected.is_integer()):
            errors.append("Static+Diverse n_chunks_selected must be a positive integer")
        if selected is not None and total is not None and total >= 1:
            expected = budget_chunk_count(int(total), float(budget))
            if int(selected) != expected:
                errors.append(
                    f"Static+Diverse selected {int(selected)} chunks; expected {expected}")
            expected_fraction = expected / int(total)
            if (touched is not None and
                    not math.isclose(touched, expected_fraction,
                                     rel_tol=0.0, abs_tol=1e-9)):
                errors.append("Static+Diverse touched fraction disagrees with budget")
        if fallback is not None and not math.isclose(fallback, 0.0):
            errors.append("Static+Diverse fallback_rate must equal 0")
    else:
        errors.append(f"unknown method key {method_key}")
    return errors


def _write_readme(run_dir, config, summary, validation):
    lines = [
        "# VisDial v1.0 official candidate-ranking quality run",
        "",
        f"- Schema: `{SCHEMA_VERSION}`",
        f"- Dialogs / evaluated rounds: {config['n_dialogs']} / {config['n_turns']}",
        f"- History: `{config['history_policy']}`",
        f"- Calibration: `{config['calibration_policy']}`",
        f"- Candidate count per round: {config['candidate_count_per_turn']}",
        f"- Validation: `{'PASS' if validation['passed'] else 'FAIL'}`",
        "- System latency measurements: none (quality-only run)",
        "",
        "## Overall",
        "",
        "| Method | MRR | R@1 | R@5 | R@10 | Mean Rank | NDCG | Logical KV | Touched chunks | Sparse / dense rounds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method_key in config["method_keys"]:
        item = summary[method_key]
        metric = item["overall"]
        lines.append(
            f"| {item['method']} | {_format_metric(metric['mrr'])} | "
            f"{_format_metric(metric['r@1'])} | "
            f"{_format_metric(metric['r@5'])} | "
            f"{_format_metric(metric['r@10'])} | "
            f"{_format_metric(metric['mean_rank'], 2)} | "
            f"{_format_metric(metric['ndcg'])} | "
            f"{_format_metric(metric['logical_kv_ratio_mean'])} | "
            f"{_format_metric(metric['touched_chunk_fraction_mean'])} | "
            f"{metric['n_sparse_rounds']} / {metric['n_dense_rounds']} |"
        )
    lines.extend([
        "",
        "## Protocol",
        "",
        "Every round uses the image, caption, and identical gold teacher-forced "
        "history. Each of the 100 source-order candidates is scored by the "
        "unnormalized sum of token log-probabilities, including EOS. The common "
        "prompt cache is cropped back to its exact branch point after every "
        "candidate. Ranks are 1-based and aligned with the original candidate "
        "order.",
        "",
        "MRR, R@1/5/10, and Mean Rank are the standard VisDial sparse retrieval "
        "metrics. NDCG is emitted only for validation rounds carrying dense "
        "relevance annotations. Scoring, rank conversion, and NDCG were checked "
        f"against the official starter at commit `{OFFICIAL_STARTER_COMMIT}`.",
        "",
        "## Artifacts",
        "",
        "- `raw.jsonl`: scores, ranks, provenance hashes, and selection summary "
        "for every method/dialog/round.",
        "- `official_ranks/*.json`: VisDial evaluator-schema-compatible rank "
        "fragments for the configured deterministic subset.",
        "- `summary.csv`: overall metrics by method.",
        "- `per_turn.csv`: metrics grouped by dialogue round.",
        "- `per_dialog.csv`: metrics grouped by dialog and method.",
        "- `validation.json`: completeness, rank, score, and cross-method "
        "identity checks.",
        "",
        "## Limitations",
        "",
        "- Candidate likelihood is a model scoring policy, not a prescribed "
        "VisDial model architecture. Because it is not length-normalized, it can "
        "prefer shorter answers; this policy is fixed across all methods.",
        "- The rank metrics and rank-file format are official-style retrieval "
        "outputs; no claim is made that this generative checkpoint reproduces a "
        "published VisDial baseline.",
        "- Rank files cover exactly the configured index and turn limit. Do not "
        "treat a subset file as a complete full-validation EvalAI submission.",
        "- This run intentionally contains no TTFT, decode, E2E, or SSD timing. "
        "Those belong to the separate system runner.",
        "- Stored arms use caption-only pre-dialog calibration. Current/future "
        "answers and candidates are excluded from selector inputs.",
        "",
    ])
    with open(run_dir / "README.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _validate_output_rows(rows, expected_keys, cpu_validation, source_turns):
    """Validate raw rows before publishing summaries or official rank files."""
    seen = set()
    duplicate_keys = []
    rank_failures = []
    rank_score_failures = []
    score_failures = []
    gt_alignment_failures = []
    ndcg_failures = []
    selection_failures = []
    source_failures = []
    forbidden_fields = []
    prompt_hashes = defaultdict(set)
    selector_hashes = defaultdict(set)
    candidate_hashes = defaultdict(set)
    for row in rows:
        key = (row["dialog_id"], int(row["turn_id"]), row["method_key"])
        if key in seen:
            duplicate_keys.append(key)
        seen.add(key)
        request = key[:2]

        try:
            ranks = [int(value) for value in row["ranks"]]
        except (KeyError, TypeError, ValueError):
            ranks = []
        if sorted(ranks) != list(range(1, 101)):
            rank_failures.append(key)

        scores = row.get("scores", [])
        scores_valid = (
            isinstance(scores, list) and len(scores) == 100 and
            all(not isinstance(value, bool) and isinstance(value, (int, float))
                and math.isfinite(float(value)) for value in scores)
        )
        if not scores_valid:
            score_failures.append(key)
        elif ranks != _ranks_from_scores(scores):
            rank_score_failures.append(key)

        try:
            gt = int(row["gt_index"])
            gt_aligned = (0 <= gt < 100 and len(ranks) == 100 and
                          int(row["gt_rank"]) == ranks[gt])
        except (KeyError, TypeError, ValueError, IndexError):
            gt_aligned = False
        if not gt_aligned:
            gt_alignment_failures.append(key)

        source = source_turns.get(request)
        if source is None:
            source_failures.append({"key": key, "reason": "missing source turn"})
        else:
            dialog, turn, dialog_order = source
            expected_source = {
                "schema_version": SCHEMA_VERSION,
                "dataset": "visdial_v1.0_val",
                "dialog_order": int(dialog_order),
                "image_id": _source_image_id(dialog),
                "gt_index": int(turn["gt_index"]),
                "method": METHODS[key[2]][0],
                "budget": METHODS[key[2]][1],
                "prompt_sha256": _sha_text(visdial_prompt(dialog, key[1])),
                "candidate_answers_sha256":
                    _candidate_sha(turn["candidate_answers"]),
                "score_definition":
                    "unnormalized sum log-probability including EOS",
            }
            mismatched = [name for name, value in expected_source.items()
                          if row.get(name) != value]
            if mismatched:
                source_failures.append({"key": key, "fields": mismatched})

            has_dense = "dense_relevance" in turn
            expected_ndcg = (ndcg(scores, turn["dense_relevance"])
                             if scores_valid and has_dense else None)
            actual_ndcg = row.get("ndcg")
            ndcg_valid = scores_valid and (
                actual_ndcg is None and expected_ndcg is None
            ) or (
                scores_valid and
                actual_ndcg is not None and expected_ndcg is not None and
                isinstance(actual_ndcg, (int, float)) and
                not isinstance(actual_ndcg, bool) and
                math.isfinite(float(actual_ndcg)) and
                0.0 <= float(actual_ndcg) <= 1.0 + 1e-12 and
                math.isclose(float(actual_ndcg), float(expected_ndcg),
                             rel_tol=0.0, abs_tol=1e-12)
            )
            if not ndcg_valid:
                ndcg_failures.append(key)

        selection_error = _selection_errors(key[2], row.get("selection"))
        if selection_error:
            selection_failures.append({"key": key, "errors": selection_error})
        present = sorted(_FORBIDDEN_LATENCY_FIELDS.intersection(row))
        if present:
            forbidden_fields.append({"key": key, "fields": present})
        prompt_hashes[request].add(row.get("prompt_sha256"))
        selector_hashes[request].add(row.get("selector_input_ids_sha256"))
        candidate_hashes[request].add(row.get("candidate_answers_sha256"))

    missing = sorted(expected_keys - seen)
    unexpected = sorted(seen - expected_keys)
    cross_method_prompt_failures = [key for key, values in prompt_hashes.items()
                                    if len(values) != 1]
    cross_method_selector_failures = [
        key for key, values in selector_hashes.items() if len(values) != 1]
    cross_method_candidate_failures = [
        key for key, values in candidate_hashes.items() if len(values) != 1]
    failure_groups = (
        duplicate_keys, rank_failures, rank_score_failures, score_failures,
        gt_alignment_failures, ndcg_failures, selection_failures,
        source_failures, forbidden_fields, missing, unexpected,
        cross_method_prompt_failures, cross_method_selector_failures,
        cross_method_candidate_failures,
    )
    return {
        "passed": not any(failure_groups),
        "cpu_index_validation": cpu_validation,
        "expected_records": len(expected_keys),
        "actual_records": len(rows),
        "duplicate_keys": duplicate_keys[:20],
        "missing_keys": missing[:20],
        "unexpected_keys": unexpected[:20],
        "rank_permutation_failures": rank_failures[:20],
        "rank_score_alignment_failures": rank_score_failures[:20],
        "score_vector_failures": score_failures[:20],
        "ground_truth_rank_alignment_failures": gt_alignment_failures[:20],
        "ndcg_alignment_failures": ndcg_failures[:20],
        "selection_semantic_failures": selection_failures[:20],
        "source_alignment_failures": source_failures[:20],
        "forbidden_latency_fields": forbidden_fields[:20],
        "cross_method_prompt_hash_failures": cross_method_prompt_failures[:20],
        "cross_method_selector_input_hash_failures":
            cross_method_selector_failures[:20],
        "cross_method_candidate_hash_failures":
            cross_method_candidate_failures[:20],
        "candidate_branching":
            "one prompt prefill; DynamicCache.crop after every candidate",
        "score_definition": "unnormalized sum log-probability including EOS",
        "selector_forbidden_inputs": [
            "candidate_answers", "candidate_answer_indices", "gt_index",
            "current_gold_answer", "future_questions", "future_answers",
        ],
        "contains_system_latency_measurements": False,
    }


def _write_outputs(run_dir, rows, method_keys, summary, expected_keys,
                   cpu_validation, config, source_turns):
    validation = _validate_output_rows(
        rows, expected_keys, cpu_validation, source_turns)
    with open(run_dir / "validation.json", "w") as f:
        json.dump(validation, f, indent=1)
    if not validation["passed"]:
        return validation

    ranks_dir = run_dir / "official_ranks"
    ranks_dir.mkdir(parents=True, exist_ok=True)
    for method_key in method_keys:
        selected = [r for r in rows if r["method_key"] == method_key]
        selected.sort(key=lambda r: (int(r["dialog_order"]),
                                     int(r["turn_id"])))
        official = [{"image_id": int(r["image_id"]),
                     "round_id": int(r["turn_id"]),
                     "ranks": [int(x) for x in r["ranks"]]}
                    for r in selected]
        with open(ranks_dir / f"{method_key.replace('@', '_')}.json", "w") as f:
            json.dump(official, f)

    with open(run_dir / "summary.json", "w") as f:
        json.dump({"schema_version": SCHEMA_VERSION,
                   "metrics": summary}, f, indent=1)
    columns = [
        "method_key", "method", "budget", "n_sparse_rounds",
        "n_dense_rounds", "mrr", "r@1", "r@5", "r@10", "mean_rank",
        "ndcg", "logical_kv_ratio_mean", "touched_chunk_fraction_mean",
        "mrr_delta_vs_fullload", "r@1_delta_vs_fullload",
        "r@5_delta_vs_fullload", "r@10_delta_vs_fullload",
        "mean_rank_delta_vs_fullload", "ndcg_delta_vs_fullload",
    ]
    with open(run_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for key in method_keys:
            item, metric = summary[key], summary[key]["overall"]
            writer.writerow({"method_key": key, "method": item["method"],
                             "budget": item["budget"], **metric})

    grouped_columns = ["aggregation", "group", *columns]
    with open(run_dir / "per_turn.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=grouped_columns)
        writer.writeheader()
        for key in method_keys:
            item = summary[key]
            for turn_id in sorted(item["by_turn"], key=int):
                writer.writerow({
                    "aggregation": "turn", "group": int(turn_id),
                    "method_key": key, "method": item["method"],
                    "budget": item["budget"], **item["by_turn"][turn_id],
                })

    dialog_order = {
        row["dialog_id"]: int(row["dialog_order"]) for row in rows
    }
    dialog_ids = sorted(dialog_order, key=lambda value: dialog_order[value])
    with open(run_dir / "per_dialog.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=grouped_columns)
        writer.writeheader()
        for dialog_id in dialog_ids:
            for key in method_keys:
                group = [row for row in rows
                         if row["dialog_id"] == dialog_id and
                         row["method_key"] == key]
                metric = _metrics_for_rows(group)
                writer.writerow({
                    "aggregation": "dialog", "group": dialog_id,
                    "method_key": key, "method": METHODS[key][0],
                    "budget": METHODS[key][1], **metric,
                })

    _write_readme(run_dir, config, summary, validation)
    return validation


def _validate_store_manifest(store_dir, index_path, dialogs, needs_static):
    path = store_dir / "pipeline_manifest.json"
    assert path.exists(), f"missing causal store provenance manifest: {path}"
    with open(path) as f:
        manifest = json.load(f)
    assert manifest["index_sha256"] == sha256_file(index_path)
    assert manifest["calibration_policy"] == "caption_only_pre_dialog"
    assert int(manifest["future_turn_calibration_count"]) == 0
    assert manifest["stages"]["build"]["done"]
    assert manifest["stages"]["reorder"]["done"]
    if needs_static:
        assert manifest["stages"]["static"]["done"]
    available = set(manifest["image_ids"])
    requested = {d["image_ids"][0] for d in dialogs}
    assert requested <= available, \
        f"store is missing canonical images: {sorted(requested - available)}"

    static_stage = manifest["stages"].get("static", {})
    if static_stage.get("done"):
        artifact_rows = static_stage.get("artifacts", [])
    else:
        artifact_rows = manifest["stages"]["reorder"].get(
            "artifacts_after_reorder", [])
    artifacts = {row["image_id"]: row for row in artifact_rows}
    assert requested <= set(artifacts), \
        f"manifest has no final artifacts for {sorted(requested - set(artifacts))}"
    for image_id in requested:
        image_dir = store_dir / image_id
        meta_path = image_dir / "meta.json"
        assert meta_path.exists(), meta_path
        artifact = artifacts[image_id]
        assert sha256_file(meta_path) == artifact["meta_sha256"], \
            f"store metadata hash mismatch for {image_id}"
        with open(meta_path) as handle:
            meta = json.load(handle)
        assert meta["image_id"] == image_id
        assert int(meta["chunk_size"]) == 64
        item_size = np.dtype(meta["dtype"]).itemsize
        full_file_bytes = (int(meta["v_token_num"]) *
                           int(meta["num_heads"]) *
                           int(meta["head_dim"]) * item_size)
        probe_file_bytes = (int(meta["v_token_num"]) *
                            int(meta["probe_heads"]) *
                            int(meta["head_dim"]) * item_size)
        for layer in range(int(meta["num_layers"])):
            layer_dir = image_dir / f"layer_{layer:02d}"
            for name, expected_size in (
                    ("k.bin", full_file_bytes),
                    ("v.bin", full_file_bytes),
                    ("probe_k.bin", probe_file_bytes)):
                path_for_file = layer_dir / name
                assert path_for_file.stat().st_size == expected_size, \
                    f"store file size mismatch: {path_for_file}"
        assert (image_dir / "sys_kv.pt").is_file()
        assert (image_dir / "v_hidden.pt").is_file()
        if needs_static:
            assert sha256_file(image_dir / "static.pt") == \
                artifact["static_pt_sha256"], \
                f"static metadata hash mismatch for {image_id}"
            assert (image_dir / "sep_kv.bin").stat().st_size == \
                int(artifact["sep_kv_bytes"]), \
                f"separator sidecar size mismatch for {image_id}"
    return path, manifest


_RESUME_SEMANTIC_KEYS = (
    "schema_version", "dataset", "quality_mode", "index", "index_sha256",
    "store", "store_content_fingerprint", "calibration_policy",
    "future_turn_calibration_count", "method_keys", "methods", "n_dialogs",
    "n_turns", "expected_request_keys_sha256", "max_dialogs", "max_turns",
    "seed", "history_policy", "prompt_template", "model",
    "model_config_name_or_path", "model_checkpoint_revision", "load_4bit",
    "quantization", "attention", "decoding", "candidate_count_per_turn",
    "candidate_branch_batch_size", "candidate_cache_restore",
    "candidate_tokenization", "score_definition", "length_normalized",
    "eos_token_id", "bos_token_id", "pad_token_id", "tokenizer_class",
    "tokenizer_name_or_path", "image_processor_class", "image_token_id",
    "max_position_embeddings", "transformers_version", "torch_version",
    "sparsevlm_budget", "static_diverse_budgets", "diverse_frac",
    "separator_policy", "official_starter_commit",
)


def _config_compatible(old, new):
    mismatches = {
        key: {"existing": old.get(key), "requested": new.get(key)}
        for key in _RESUME_SEMANTIC_KEYS
        if old.get(key) != new.get(key)
    }
    if mismatches:
        raise RuntimeError(
            "unsafe resume: semantic config mismatch:\n" +
            json.dumps(mismatches, indent=1, ensure_ascii=False)
        )


def _validate_resume_record(row, dialog, turn, dialog_order, method_key,
                            prompt_sha, suffix_sha, candidate_sha,
                            candidate_lengths, prompt_tokens):
    """Reject a stale/corrupt completed row before allowing it to be skipped."""
    required = {
        "schema_version", "dataset", "dialog_id", "dialog_order", "image_id",
        "turn_id", "method", "method_key", "budget", "gt_index", "gt_rank",
        "ndcg", "scores", "ranks", "candidate_token_lengths_including_eos",
        "prompt_cache_tokens", "prompt_sha256", "selector_input_ids_sha256",
        "candidate_answers_sha256", "selection", "score_definition",
    }
    missing = sorted(required - set(row))
    if missing:
        raise RuntimeError(f"unsafe resume: raw record is missing {missing}")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "visdial_v1.0_val",
        "dialog_id": dialog["dialog_id"],
        "dialog_order": int(dialog_order),
        "image_id": _source_image_id(dialog),
        "turn_id": int(turn["turn_id"]),
        "method": METHODS[method_key][0],
        "method_key": method_key,
        "budget": METHODS[method_key][1],
        "gt_index": int(turn["gt_index"]),
        "prompt_cache_tokens": int(prompt_tokens),
        "prompt_sha256": prompt_sha,
        "selector_input_ids_sha256": suffix_sha,
        "candidate_answers_sha256": candidate_sha,
        "score_definition": "unnormalized sum log-probability including EOS",
    }
    mismatches = {
        key: {"existing": row.get(key), "expected": value}
        for key, value in expected.items() if row.get(key) != value
    }
    lengths = [int(value) for value in
               row["candidate_token_lengths_including_eos"]]
    if lengths != [int(value) for value in candidate_lengths]:
        mismatches["candidate_token_lengths_including_eos"] = {
            "existing": lengths, "expected": candidate_lengths,
        }
    forbidden = sorted(_FORBIDDEN_LATENCY_FIELDS.intersection(row))
    if forbidden:
        mismatches["forbidden_latency_fields"] = {
            "existing": forbidden, "expected": [],
        }
    scores = row["scores"]
    if (len(scores) != 100 or
            not all(math.isfinite(float(value)) for value in scores)):
        mismatches["scores"] = {
            "existing": "invalid/non-finite score vector",
            "expected": "100 finite values",
        }
    else:
        expected_ranks = _ranks_from_scores(scores)
        ranks = [int(value) for value in row["ranks"]]
        if ranks != expected_ranks:
            mismatches["ranks"] = {
                "existing": ranks, "expected": expected_ranks,
            }
        gt = int(turn["gt_index"])
        expected_gt_rank = expected_ranks[gt]
        if int(row["gt_rank"]) != expected_gt_rank:
            mismatches["gt_rank"] = {
                "existing": row["gt_rank"], "expected": expected_gt_rank,
            }
        expected_ndcg = (ndcg(scores, turn["dense_relevance"])
                         if "dense_relevance" in turn else None)
        actual_ndcg = row["ndcg"]
        ndcg_matches = (
            actual_ndcg is None and expected_ndcg is None
        ) or (
            actual_ndcg is not None and expected_ndcg is not None and
            math.isclose(float(actual_ndcg), float(expected_ndcg),
                         rel_tol=0.0, abs_tol=1e-12)
        )
        if not ndcg_matches:
            mismatches["ndcg"] = {
                "existing": actual_ndcg, "expected": expected_ndcg,
            }
    selection_errors = _selection_errors(method_key, row["selection"])
    if selection_errors:
        mismatches["selection_semantics"] = {
            "existing": row["selection"], "errors": selection_errors,
        }
    if mismatches:
        key = (dialog["dialog_id"], int(turn["turn_id"]), method_key)
        raise RuntimeError(
            f"unsafe resume: raw record mismatch for {key}:\n" +
            json.dumps(mismatches, indent=1, ensure_ascii=False)
        )


def main():
    parser = argparse.ArgumentParser(
        description="VisDial official 100-candidate quality scorer; no TTFT")
    parser.add_argument("--index", required=True,
                        help="canonical VisDial validation index.json")
    parser.add_argument("--store", default=None,
                        help="causal VisDial KV store (required for stored arms)")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--methods", default=DEFAULT_METHODS,
                        help=f"comma-separated keys: {','.join(METHODS)}")
    parser.add_argument("--max-dialogs", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10,
                        help="print candidate progress every N branches; 0 disables")
    parser.add_argument("--validate-only", action="store_true",
                        help="CPU-only canonical/leakage checks; load no model/store")
    args = parser.parse_args()

    assert args.max_dialogs is None or args.max_dialogs >= 1
    assert args.max_turns is None or 1 <= args.max_turns <= 10
    assert args.progress_every >= 0
    method_keys = [x.strip() for x in args.methods.split(",") if x.strip()]
    assert method_keys and len(method_keys) == len(set(method_keys))
    assert all(key in METHODS for key in method_keys), method_keys

    index_path = Path(args.index)
    dialogs = load_canonical(index_path)
    if args.max_dialogs is not None:
        dialogs = dialogs[:args.max_dialogs]
    cpu_validation = _validate_visdial(dialogs, args.max_turns)
    if args.validate_only:
        print(json.dumps(cpu_validation, indent=1))
        return

    assert args.run_dir, "--run-dir is required unless --validate-only is used"
    stored_keys = [key for key in method_keys if key != "recompute"]
    store_dir = Path(args.store) if args.store else None
    if stored_keys:
        assert store_dir is not None, "--store is required for stored-KV methods"
        manifest_path, store_manifest = _validate_store_manifest(
            store_dir, index_path, dialogs,
            needs_static=any(key.startswith("static_diverse")
                             for key in stored_keys))
        manifest_sha = sha256_file(manifest_path)
        store_snapshot = _store_content_snapshot(store_manifest)
        store_fingerprint = _stable_json_hash(store_snapshot)
    else:
        manifest_path = store_manifest = None
        manifest_sha = None
        store_snapshot = store_fingerprint = None

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "raw.jsonl"
    owned_outputs = [raw_path, run_dir / "config.json",
                     run_dir / "summary.json", run_dir / "summary.csv",
                     run_dir / "per_turn.csv", run_dir / "per_dialog.csv",
                     run_dir / "validation.json", run_dir / "README.md",
                     run_dir / "official_ranks"]
    existing_outputs = [str(path) for path in owned_outputs if path.exists()]
    if existing_outputs and not args.resume:
        raise FileExistsError(
            "quality outputs already exist; use a new directory or pass "
            f"--resume: {existing_outputs}")
    if existing_outputs and args.resume and not (run_dir / "config.json").exists():
        raise FileNotFoundError(
            "cannot safely resume outputs without their config.json")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    runner = LlavaRunner().load()
    server = Server(runner, ratio=0.25)
    max_context = int(runner.cfg.text_config.max_position_embeddings)

    selected = list(_selected_turns(dialogs, args.max_turns))
    dialog_order_by_id = {
        dialog["dialog_id"]: order for order, dialog in enumerate(dialogs)
    }
    source_turns = {
        (dialog["dialog_id"], int(turn["turn_id"])):
            (dialog, turn, dialog_order_by_id[dialog["dialog_id"]])
        for dialog, turn in selected
    }
    expected_key_rows = [
        [dialog["dialog_id"], int(turn["turn_id"]), method_key]
        for dialog, turn in selected for method_key in method_keys
    ]
    tokenizer = runner.processor.tokenizer
    config = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "visdial_v1.0_val",
        "quality_mode": "official_candidate_ranking",
        "index": str(index_path.resolve()),
        "index_sha256": sha256_file(index_path),
        "store": str(store_dir.resolve()) if store_dir else None,
        "store_manifest": (str(manifest_path.resolve())
                           if manifest_path else None),
        # The whole manifest also contains mutable launcher timestamps.  Keep
        # that audit digest, but resume against the immutable content subset.
        "store_manifest_sha256_at_quality_start": manifest_sha,
        "store_content_snapshot": store_snapshot,
        "store_content_fingerprint": store_fingerprint,
        "calibration_policy": (store_manifest["calibration_policy"]
                               if store_manifest else None),
        "future_turn_calibration_count": (
            int(store_manifest["future_turn_calibration_count"])
            if store_manifest else 0),
        "method_keys": method_keys,
        "methods": [METHODS[key][0] for key in method_keys],
        "n_dialogs": len(dialogs),
        "n_turns": len(selected),
        "expected_request_keys": expected_key_rows,
        "expected_request_keys_sha256": _stable_json_hash(expected_key_rows),
        "max_dialogs": args.max_dialogs,
        "max_turns": args.max_turns,
        "seed": args.seed,
        "history_policy": "gold_teacher_forced",
        "prompt_template": "visdial-gold-history-v1",
        "model": runner.model_id,
        "model_config_name_or_path": runner.cfg._name_or_path,
        "model_checkpoint_revision": getattr(runner.cfg, "_commit_hash", None),
        "load_4bit": runner.load_4bit,
        "quantization": "NF4 double-quant",
        "attention": runner.attn,
        "decoding": None,
        "candidate_count_per_turn": 100,
        "candidate_branch_batch_size": 1,
        "candidate_cache_restore": "DynamicCache.crop(prompt_length)",
        "candidate_tokenization": (
            "tokenize(prompt + single ASCII space + candidate); require exact "
            "prompt-token prefix; append one EOS token"),
        "score_definition": "unnormalized sum log-probability including EOS",
        "length_normalized": False,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_name_or_path": tokenizer.name_or_path,
        "image_processor_class":
            type(runner.processor.image_processor).__name__,
        "image_token_id": int(runner.image_token_id),
        "max_position_embeddings": max_context,
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "sparsevlm_budget": 0.25,
        "static_diverse_budgets": [0.25, 0.50],
        "diverse_frac": 0.25,
        "separator_policy": "sidecar",
        "metric_definitions": ["MRR", "R@1", "R@5", "R@10",
                               "Mean Rank", "NDCG (dense round only)"],
        "official_starter_repository": OFFICIAL_STARTER_URL,
        "official_starter_commit": OFFICIAL_STARTER_COMMIT,
        "official_reference_files": [
            "visdialch/decoders/gen.py", "visdialch/metrics.py", "evaluate.py"],
        "official_rank_files": {
            key: f"official_ranks/{key.replace('@', '_')}.json"
            for key in method_keys
        },
        "official_rank_scope": (
            "schema-compatible fragment for the configured deterministic subset; "
            "not a complete full-validation EvalAI submission"),
        "rank_format": ("VisDial list[{image_id, round_id, ranks}]; ranks "
                        "are 1-based and aligned to original candidate order"),
        "contains_system_latency_measurements": False,
        "page_cache_policy": "not controlled; quality-only run",
    }
    config_path = run_dir / "config.json"
    prior_elapsed = 0.0
    if config_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"{config_path} exists; use a new directory or pass --resume")
        with open(config_path) as f:
            old_config = json.load(f)
        _config_compatible(old_config, config)
        prior_elapsed = float(old_config.get("elapsed_seconds", 0.0))
        config["store_manifest_sha256_at_quality_start"] = old_config.get(
            "store_manifest_sha256_at_quality_start",
            config["store_manifest_sha256_at_quality_start"])
        config["store_content_snapshot"] = old_config.get(
            "store_content_snapshot", config["store_content_snapshot"])
        config["run_started_at_unix"] = old_config.get(
            "run_started_at_unix", time.time())
        config["resume_count"] = int(old_config.get("resume_count", 0)) + 1
    else:
        config["run_started_at_unix"] = time.time()
        config["resume_count"] = 0
    config["elapsed_seconds"] = prior_elapsed
    with open(config_path, "w") as f:
        json.dump(config, f, indent=1)

    old_rows = _load_existing(raw_path) if args.resume else []
    old_by_key = {
        (r["dialog_id"], int(r["turn_id"]), r["method_key"]): r
        for r in old_rows
    }
    completed = set(old_by_key)
    if len(completed) != len(old_rows):
        raise RuntimeError("unsafe resume: duplicate records in raw.jsonl")
    expected_keys = {(d["dialog_id"], int(t["turn_id"]), key)
                     for d, t in selected for key in method_keys}
    unexpected_completed = sorted(completed - expected_keys)
    if unexpected_completed:
        raise RuntimeError(
            "unsafe resume: raw.jsonl contains records outside the configured "
            f"workload: {unexpected_completed[:20]}")

    total_jobs = len(expected_keys)
    finished_jobs = len(completed)
    elapsed_start = time.time()
    with open(raw_path, "a") as raw:
        for dialog_order, dialog in enumerate(dialogs):
            image_id = dialog["image_ids"][0]
            needs_store = bool(stored_keys)
            ctx = static = None
            if needs_store:
                image_store = store_dir / image_id
                assert (image_store / "meta.json").exists(), image_store
                ctx = ImageContext(image_store, runner.model.device,
                                   drop_cache=False)
                if any(key.startswith("static_diverse") for key in stored_keys):
                    static = load_static(ctx)
            image = Image.open(resolve_image_path(
                dialog["images"][0]["image_path"])).convert("RGB")
            try:
                n_turns = min(len(dialog["turns"]),
                              args.max_turns or len(dialog["turns"]))
                for turn in dialog["turns"][:n_turns]:
                    turn_id = int(turn["turn_id"])
                    prompt = _assert_causal_prompt(dialog, turn_id)
                    prompt_sha = _sha_text(prompt)
                    suffix_ids = suffix_ids_from_prompt(runner, prompt)
                    suffix_sha = _sha_ids(suffix_ids)

                    # This processor call contains the bare prompt only.  It
                    # proves that every method sees the exact reusable prefix.
                    encoded = runner.encode_prompt(image, prompt)
                    if ctx is not None:
                        expected_len = int(ctx.meta["prefix_len"]) + len(suffix_ids)
                        assert int(encoded["input_ids"].shape[1]) == expected_len
                        prefix = torch.tensor(ctx.meta["prefix_input_ids"],
                                              dtype=encoded["input_ids"].dtype)
                        assert torch.equal(
                            encoded["input_ids"][0, :ctx.meta["prefix_len"]],
                            prefix)
                        assert torch.equal(
                            encoded["input_ids"][0, ctx.meta["prefix_len"]:],
                            suffix_ids.to(encoded["input_ids"].device))

                    candidates = list(turn["candidate_answers"])
                    candidate_sha = _candidate_sha(candidates)
                    candidate_ids = _candidate_token_ids(
                        runner.processor.tokenizer, prompt, candidates)
                    candidate_lengths = [len(ids) for ids in candidate_ids]
                    assert _sha_ids(suffix_ids) == suffix_sha, \
                        "candidate tokenisation mutated selector input"

                    for method_key in method_keys:
                        key = (dialog["dialog_id"], turn_id, method_key)
                        if key in completed:
                            _validate_resume_record(
                                old_by_key[key], dialog, turn, dialog_order,
                                method_key, prompt_sha, suffix_sha,
                                candidate_sha, candidate_lengths,
                                int(encoded["input_ids"].shape[1]))
                            continue
                        finished_jobs += 1
                        label, budget = METHODS[method_key]
                        print(f"[{finished_jobs}/{total_jobs}] "
                              f"{dialog['dialog_id']} turn {turn_id} {label}",
                              flush=True)
                        BIAS.clear()
                        cache = prompt_logits = selection_summary = None
                        try:
                            # Neither preparation function accepts a turn,
                            # gold answer, candidate list, or future history.
                            if method_key == "recompute":
                                cache, prompt_logits, selection_summary = \
                                    _recompute_prompt_prefill(runner, encoded)
                            else:
                                cache, prompt_logits, selection_summary = \
                                    _stored_prompt_prefill(
                                        method_key, runner, server, ctx, static,
                                        suffix_ids, args.seed, image_id)

                            def progress(done, total):
                                if (args.progress_every and
                                        (done % args.progress_every == 0 or
                                         done == total)):
                                    print(f"  candidates {done}/{total}",
                                          flush=True)

                            scores = _score_candidates(
                                runner, cache, prompt_logits, candidate_ids,
                                max_context, progress=progress)
                        finally:
                            BIAS.clear()

                        # Ground-truth indices and dense relevance labels are
                        # consulted only after selection and scoring finish.
                        ranks = _ranks_from_scores(scores)
                        gt_index = int(turn["gt_index"])
                        dense_score = (ndcg(scores, turn["dense_relevance"])
                                       if "dense_relevance" in turn else None)
                        record = {
                            "schema_version": SCHEMA_VERSION,
                            "dataset": "visdial_v1.0_val",
                            "dialog_id": dialog["dialog_id"],
                            "dialog_order": dialog_order,
                            "image_id": _source_image_id(dialog),
                            "turn_id": turn_id,
                            "method": label,
                            "method_key": method_key,
                            "budget": budget,
                            "gt_index": gt_index,
                            "gt_rank": int(ranks[gt_index]),
                            "ndcg": (float(dense_score)
                                     if dense_score is not None else None),
                            "scores": scores,
                            # Required official format alignment: rank i is for
                            # candidate_answers[i] in the untouched source order.
                            "ranks": ranks,
                            "candidate_token_lengths_including_eos":
                                candidate_lengths,
                            "prompt_cache_tokens": int(_cache_length(cache)),
                            "prompt_sha256": prompt_sha,
                            "selector_input_ids_sha256": suffix_sha,
                            "candidate_answers_sha256": candidate_sha,
                            "selection": selection_summary,
                            "score_definition":
                                "unnormalized sum log-probability including EOS",
                        }
                        assert not _FORBIDDEN_LATENCY_FIELDS.intersection(record)
                        raw.write(json.dumps(record) + "\n")
                        raw.flush()
                        completed.add(key)
                        del cache, prompt_logits
            finally:
                image.close()
                if ctx is not None:
                    ctx.close()
                BIAS.clear()
                torch.cuda.empty_cache()

    rows = _load_existing(raw_path)
    config["elapsed_seconds"] = prior_elapsed + (time.time() - elapsed_start)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=1)
    summary = _summarise(rows, method_keys)
    validation = _write_outputs(run_dir, rows, method_keys, summary,
                                expected_keys, cpu_validation, config,
                                source_turns)
    print(json.dumps(summary, indent=1))
    print(json.dumps(validation, indent=1))
    if not validation["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

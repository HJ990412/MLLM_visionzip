#!/usr/bin/env python3
"""ConvBench judge for the baseline alignment (ReComp answers only; stage judgments S1/S2/S3).

Unchanged from the official protocol / the September run: judge prompts (official ``generate_request``),
A/B placement (pairwise.npy), first-pass rule ('response x is better', exactly one), denominator 577,
unresolved = zero wins, judge model Llama-3.1-8B-Instruct (BF16, SDPA, greedy).

Changed (notes.md, stage 2):
  J1  second pass = the official extraction: official ``generate_parsing_answer_request`` messages
      (system prompt + 3 examples), output cap 512 as in the official call, winner = exactly one of
      'Final Answer: Response A' / 'Final Answer: Response B' (convbencheval.py:221-229); then the official
      scoring fallback 'Rating:(.*)' -> A/B, otherwise C (convbencheval.py:67-77, 397-404).
  J2  first-pass output cap 512 -> --cap (2048).

Every judgment is generated once, greedily, up to --cap tokens and its token ids are kept.  The text of the
first 512 tokens is the "cap 512" view (what a run with the official cap produces under greedy decoding);
both views are parsed, so one run yields the scores for "J1" (cap 512 + official extraction) and for
"J1+J2" (cap --cap + official extraction).

Results are cached by the SHA-256 of the judge messages (same judge, same prompt, greedy -> same output), so
conversations whose three answers are unchanged between runs are not judged again.

Output: <out>/config.json, <out>/judgments/<source_row_index>.json, <out>/progress.json, <out>/summary.json,
<out>/judge_raw.jsonl, <out>/DONE.  Re-running continues from the finished conversations.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mmimpress.convbench_judge import (  # noqa: E402
    MODEL_ID, LocalLlamaJudge, build_messages, extract_official_rate, extract_official_winner,
    load_official_prompt_builder, load_pairwise, normalize_row, official_revision, score_records, sha256_file,
)

SOURCE = ROOT / "data/convbench_source"
INDEX = ROOT / "data/convbench/index.json"
SEPT_ANSWERS = ROOT / "runs/convbench_context/full_577_native"
SEPT_JUDGE = ROOT / "runs/convbench_context/judge_full_577_stage_only/judge_raw.jsonl"
CHECKPOINT = Path("/home/dblab/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct"
                  "/snapshots/0e9e39f249a16976918f6564b8830bc894c89659")
TURNS = ("_first_turn", "_second_turn", "_third_turn")
OFFICIAL_CAP = 512
EXTRACTION_CAP = 512


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=1)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def load_official_extraction_builder(source_root: Path):
    """The upstream ``generate_parsing_answer_request`` function, loaded without running upstream top-level code."""
    evaluator = source_root / "ConvBenchEval" / "convbencheval.py"
    _, prompts = load_official_prompt_builder(source_root)
    tree = ast.parse(evaluator.read_text(encoding="utf-8"), filename=str(evaluator))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "generate_parsing_answer_request"]
    if len(functions) != 1:
        raise ValueError("official evaluator must contain one generate_parsing_answer_request")
    namespace = {"_PROMPT_SYSTEM_ANSWER_EXTRACTION_V1": prompts._PROMPT_SYSTEM_ANSWER_EXTRACTION_V1,
                 "_ICL_SYSTEM_ANSWER_EXTRACTION_V1": prompts._ICL_SYSTEM_ANSWER_EXTRACTION_V1}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(evaluator), "exec"), namespace)
    return namespace["generate_parsing_answer_request"]


def official_second_pass_winner(result: str) -> str | None:
    """convbencheval.py:224-229 (case-sensitive substring, exactly one)."""
    selected = [ch for ch in "AB" if f"Final Answer: Response {ch}" in result]
    return selected[0] if len(selected) == 1 else None


class Judge(LocalLlamaJudge):
    def generate_ids(self, messages, cap: int) -> tuple[list[int], int]:
        tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_tensors="pt")
        prompt_tokens = int(tokens.shape[-1])
        if prompt_tokens + cap > int(self.model.config.max_position_embeddings):
            raise ValueError(f"judge prompt exceeds context: {prompt_tokens}+{cap}")
        tokens = tokens.to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                input_ids=tokens, attention_mask=self.torch.ones_like(tokens, dtype=self.torch.long),
                do_sample=False, temperature=None, top_p=None, max_new_tokens=cap,
                pad_token_id=self.tokenizer.eos_token_id)
        return output[0, prompt_tokens:].tolist(), prompt_tokens

    def text(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)


def read_answers(path: Path, rows: list[dict]) -> dict[str, list[str]]:
    """ReComp answers from a September run directory or from a 213 generation directory."""
    answers = {}
    for row in rows:
        payload = json.loads((path / "conversations" / f"{row['conversation_id']}.json").read_text(encoding="utf-8"))
        recomp = sorted((r for r in payload["rows"] if r["method"] == "ReComp"), key=lambda r: r["turn_id"])
        if [r["turn_id"] for r in recomp] != [1, 2, 3]:
            raise ValueError(f"three ReComp answers expected for {row['conversation_id']}")
        answers[row["conversation_id"]] = [str(r["prediction"]) for r in recomp]
    return answers


def parse_view(text: str, extraction: dict | None) -> dict:
    """Winner of one view: first pass -> official extraction -> official Rating fallback -> C."""
    first = extract_official_winner(text)
    if first is not None:
        return {"winner": first, "parse_stage": "first_pass"}
    second = official_second_pass_winner(extraction["text"]) if extraction else None
    if second is not None:
        return {"winner": second, "parse_stage": "second_pass"}
    third = extract_official_rate(text)
    if third in ("A", "B"):
        return {"winner": third, "parse_stage": "rating_fallback"}
    return {"winner": None, "parse_stage": "unresolved"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", type=Path, required=True, help="213 output dir, or 'sept' for the September run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--limit", type=int, help="first N conversations only (smoke)")
    parser.add_argument("--compare-sept-judge", action="store_true",
                        help="compare with the September ReComp judge records (only meaningful with --answers sept)")
    args = parser.parse_args()
    answers_dir = SEPT_ANSWERS if str(args.answers) == "sept" else args.answers
    if args.cap < OFFICIAL_CAP:
        parser.error("--cap must be at least the official 512")

    raw_rows = json.loads(INDEX.read_text(encoding="utf-8"))["conversations"]
    rows = [normalize_row(r) for r in raw_rows]
    if args.limit:
        rows = rows[:args.limit]
    positions = load_pairwise(SOURCE, [r["source_row_index"] for r in rows])
    builder, _ = load_official_prompt_builder(SOURCE)
    extraction_builder = load_official_extraction_builder(SOURCE)
    answers = read_answers(answers_dir, rows)

    config = {
        "schema": "convbench-alignment-judge-v1", "answers": str(answers_dir), "cap": args.cap,
        "official_cap_view": OFFICIAL_CAP, "extraction_cap": EXTRACTION_CAP,
        "judge_model_id": MODEL_ID, "judge_checkpoint": str(CHECKPOINT), "judge_dtype": "bfloat16",
        "judge_attention": "sdpa", "judge_decoding": "greedy",
        "official_commit": official_revision(SOURCE),
        "official_evaluator_sha256": sha256_file(SOURCE / "ConvBenchEval/convbencheval.py"),
        "official_prompts_sha256": sha256_file(SOURCE / "ConvBenchEval/prompts.py"),
        "official_pairwise_sha256": sha256_file(SOURCE / "ConvBenchEval/pairwise.npy"),
        "index_sha256": sha256_file(INDEX), "judge_turns": list(TURNS), "denominator": 577,
        "first_pass": "official: exactly one of 'response a is better' / 'response b is better' (lower-cased)",
        "second_pass": "official generate_parsing_answer_request + exactly one 'Final Answer: Response X'",
        "third_pass": "official extract_rate ('Rating:'); otherwise C = unresolved = zero wins",
        "conversations": len(rows),
    }
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists():
        previous = json.loads((out / "config.json").read_text(encoding="utf-8"))
        if previous != config:
            raise SystemExit(f"judge config mismatch; refusing to resume into {out}")
    else:
        atomic_json(out / "config.json", config)
    args.cache.mkdir(parents=True, exist_ok=True)

    judge: Judge | None = None
    counters = {"judged_now": 0, "cache_hits": 0, "extractions_now": 0}
    started = time.time()
    fresh_conversations = 0

    def get_judge() -> Judge:
        nonlocal judge
        if judge is None:
            judge = Judge(CHECKPOINT, max_new_tokens=args.cap)
        return judge

    def cached(messages: list[dict], cap: int, kind: str) -> dict:
        key = hashlib.sha256(json.dumps([kind, cap, messages], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        path = args.cache / key[:2] / f"{key}.json"
        if path.exists():
            counters["cache_hits"] += 1
            return json.loads(path.read_text(encoding="utf-8"))
        j = get_judge()
        ids, prompt_tokens = j.generate_ids(messages, cap)
        entry = {"key": key, "kind": kind, "cap": cap, "prompt_tokens": prompt_tokens, "output_ids": ids,
                 "output_tokens": len(ids), "text": j.text(ids)}
        if kind == "judgment" and len(ids) > OFFICIAL_CAP:
            entry["text_512"] = j.text(ids[:OFFICIAL_CAP])
        atomic_json(path, entry)
        counters["judged_now" if kind == "judgment" else "extractions_now"] += 1
        return entry

    def progress(done: int, note: str = "") -> None:
        atomic_json(out / "progress.json", {
            "done": done, "total": len(rows), "updated_unix": time.time(), "note": note, **counters,
            "seconds_per_fresh_conversation": ((time.time() - started) / fresh_conversations
                                               if fresh_conversations else None)})

    all_records: list[dict] = []
    for ordinal, row in enumerate(rows):
        cid, source_index = row["conversation_id"], row["source_row_index"]
        position = positions[source_index]
        checkpoint = out / "judgments" / f"{source_index:04d}.json"
        if checkpoint.exists():
            records = json.loads(checkpoint.read_text(encoding="utf-8"))["records"]
            all_records.extend(records)
            continue
        before = counters["judged_now"]
        records = []
        for turn in TURNS:
            messages = build_messages(builder, row, answers[cid], position, turn, None)
            entry = cached(messages, args.cap, "judgment")
            text_full = entry["text"]
            text_512 = entry.get("text_512", text_full)
            views = {}
            for name, text in (("cap512", text_512), ("capN", text_full)):
                extraction = None
                if extract_official_winner(text) is None:
                    extraction = cached(extraction_builder(text), EXTRACTION_CAP, "extraction")
                parsed = parse_view(text, extraction)
                views[name] = {**parsed,
                               "extraction_text": extraction["text"] if extraction else None,
                               "extraction_output_tokens": extraction["output_tokens"] if extraction else None,
                               "model_wins": (parsed["winner"] == ("A" if position == 0 else "B")
                                              if parsed["winner"] is not None else None)}
            records.append({
                "conversation_id": cid, "source_row_index": source_index, "method": "ReComp", "judge_turn": turn,
                "position": position, "judgment_cache_key": entry["key"], "prompt_tokens": entry["prompt_tokens"],
                "output_tokens": entry["output_tokens"], "hit_cap_512": entry["output_tokens"] >= OFFICIAL_CAP,
                "hit_cap_N": entry["output_tokens"] >= args.cap, "raw_response": text_full,
                "raw_response_512": text_512 if text_512 != text_full else None, "views": views})
        atomic_json(checkpoint, {"records": records})
        all_records.extend(records)
        if counters["judged_now"] > before:
            fresh_conversations += 1
        progress(ordinal + 1, cid)
        if (ordinal + 1) % 10 == 0:
            print(f"[{ordinal + 1}/{len(rows)}] {cid} {counters}", flush=True)

    # ------------------------------------------------------------------ scores for both views
    def view_records(name: str) -> list[dict]:
        return [{"conversation_id": r["conversation_id"], "source_row_index": r["source_row_index"],
                 "method": "ReComp", "judge_turn": r["judge_turn"], "position": r["position"],
                 "winner": r["views"][name]["winner"]} for r in all_records]

    summary = {"conversations": len(rows), "judgments": len(all_records), "cap": args.cap, **counters, "views": {}}
    for name in ("cap512", "capN"):
        score = score_records(view_records(name), expected_per_method=577 if not args.limit else len(rows),
                              judge_turns=TURNS)["ReComp"]
        stages = {}
        for turn, label in zip(TURNS, ("S1", "S2", "S3")):
            group = [r for r in all_records if r["judge_turn"] == turn]
            stages[label] = {stage: sum(r["views"][name]["parse_stage"] == stage for r in group)
                             for stage in ("first_pass", "second_pass", "rating_fallback", "unresolved")}
        summary["views"][name] = {
            "S1": score["S1"], "S2": score["S2"], "S3": score["S3"], "Avg": score["Avg"],
            "model_wins": score["model_wins_by_turn"], "unresolved": score["unresolved_by_turn"],
            "unresolved_total": score["total_unresolved"], "parse_stages": stages}
    summary["output_tokens"] = {
        "hit_cap_512": {label: sum(r["hit_cap_512"] for r in all_records if r["judge_turn"] == turn)
                        for turn, label in zip(TURNS, ("S1", "S2", "S3"))},
        "hit_cap_N": {label: sum(r["hit_cap_N"] for r in all_records if r["judge_turn"] == turn)
                      for turn, label in zip(TURNS, ("S1", "S2", "S3"))},
        "max": max(r["output_tokens"] for r in all_records),
    }

    if args.compare_sept_judge:
        sept = {}
        with SEPT_JUDGE.open(encoding="utf-8") as handle:
            for line in handle:
                r = json.loads(line)
                if r["method"] == "ReComp":
                    sept[(r["conversation_id"], r["judge_turn"])] = r
        comparison = {"records": 0, "text512_identical_to_september": 0,
                      "text512_differs": [], "september_resolved": 0}
        for name in ("cap512", "capN"):
            comparison[name] = {"same_winner": 0, "winner_changed": [], "became_unresolved": [],
                                "september_unresolved_now_resolved": 0, "september_unresolved_now_model_win": 0,
                                "september_unresolved_still_unresolved": 0}
        for r in all_records:
            old = sept.get((r["conversation_id"], r["judge_turn"]))
            if old is None:
                continue
            comparison["records"] += 1
            text_512 = r["raw_response_512"] if r["raw_response_512"] is not None else r["raw_response"]
            same_text = text_512 == old["raw_response"]
            comparison["text512_identical_to_september"] += same_text
            if not same_text:
                comparison["text512_differs"].append([r["conversation_id"], r["judge_turn"]])
            old_winner = old["winner"] if old["winner"] in ("A", "B") else None
            comparison["september_resolved"] += old_winner is not None
            for name in ("cap512", "capN"):
                new = r["views"][name]["winner"]
                c = comparison[name]
                if old_winner is not None:
                    if new == old_winner:
                        c["same_winner"] += 1
                    elif new is None:
                        c["became_unresolved"].append([r["conversation_id"], r["judge_turn"], old_winner,
                                                       old["parse_stage"]])
                    else:
                        c["winner_changed"].append([r["conversation_id"], r["judge_turn"], old_winner, new,
                                                    old["parse_stage"], r["views"][name]["parse_stage"], same_text])
                elif new is None:
                    c["september_unresolved_still_unresolved"] += 1
                else:
                    c["september_unresolved_now_resolved"] += 1
                    c["september_unresolved_now_model_win"] += bool(r["views"][name]["model_wins"])
        summary["september_comparison"] = comparison

    atomic_json(out / "summary.json", summary)
    with open(out / "judge_raw.jsonl.tmp", "w", encoding="utf-8") as handle:
        for r in all_records:
            handle.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(out / "judge_raw.jsonl.tmp", out / "judge_raw.jsonl")
    progress(len(rows), "complete")
    (out / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    brief = {k: {m: (round(v[m], 2) if isinstance(v[m], float) else v[m])
                 for m in ("S1", "S2", "S3", "Avg", "unresolved")} for k, v in summary["views"].items()}
    print(json.dumps({"scores": brief, **counters, "output_tokens": summary["output_tokens"]}, indent=1))


if __name__ == "__main__":
    main()

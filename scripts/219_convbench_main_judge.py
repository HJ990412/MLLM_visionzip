#!/usr/bin/env python3
"""ConvBench main experiment: the frozen judge applied to the four methods' answers, and the judge check (gate 6).

The judge is the one frozen in docs/convbench_baseline_alignment.md and implemented in
scripts/214_convbench_alignment_judge.py; that module is imported and its pieces are used as they are
(``Judge``, ``load_official_extraction_builder``, ``parse_view``, the caps and the checkpoint path):
Llama-3.1-8B-Instruct (BF16, SDPA, greedy), official ``generate_request`` prompts, official pairwise.npy
positions, first-pass cap 2048, official second-pass extraction, official ``Rating:`` fallback,
unresolved = zero wins, denominator 577.  The view reported is ``capN`` (the adopted one).

  --model M              judge the committed main-phase answers of the four methods of model M
                         -> <res>/judge/<M>/{judgments/<role>/<row>.json, summary.json, judge_raw.jsonl, DONE}
  --check                gate 6: judge again, with an EMPTY cache, a subset of the alignment run's final ReComp
                         answers of both models and compare with the judgments recorded then
                         -> <res>/gates/judge_check.json

Cache (same key as scripts/214: SHA-256 of [kind, cap, messages]): <res>/judge_cache.  With
--reuse-alignment-cache a missing entry is looked up, read-only, in the alignment run's cache before the
judge is called (same judge, same prompt, greedy decoding; the driver passes this only after gate 6 has
shown that the judge reproduces those records).

A method whose request failed has no answer for that turn: the stage of that turn and the later stages are
recorded as ``generation_failed`` (zero wins, not counted as unresolved); an earlier stage is judged with an
empty string in place of the missing later answer(s) and flagged ``judged_with_missing_later_answer``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import convbench_main_common as C  # noqa: E402
import llava_next_main_status as STATUS  # noqa: E402
from mmimpress.convbench_judge import (build_messages, extract_official_winner, load_official_prompt_builder,  # noqa: E402
                                       load_pairwise, normalize_row, score_records)

CHECK_CONVERSATIONS = 8      # per model, gate 6
STAGE = {"_first_turn": "S1", "_second_turn": "S2", "_third_turn": "S3"}


def load_214():
    spec = importlib.util.spec_from_file_location("_convbench_main_judge_214", ROOT / "scripts/214_convbench_alignment_judge.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class Runtime:
    """Judge + cache.  The model is loaded on first use."""

    def __init__(self, J, cache: Path, fallback: Path | None, cap: int):
        self.J, self.cache, self.fallback, self.cap = J, cache, fallback, cap
        self.judge = None
        self.counters = {"judged_now": 0, "extractions_now": 0, "cache_hits": 0, "alignment_cache_hits": 0}
        cache.mkdir(parents=True, exist_ok=True)

    def _judge(self):
        if self.judge is None:
            self.judge = self.J.Judge(self.J.CHECKPOINT, max_new_tokens=self.cap)
        return self.judge

    def cached(self, messages, cap, kind):
        key = hashlib.sha256(json.dumps([kind, cap, messages], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        path = self.cache / key[:2] / f"{key}.json"
        if path.exists():
            self.counters["cache_hits"] += 1
            return json.loads(path.read_text(encoding="utf-8"))
        if self.fallback is not None:
            other = self.fallback / key[:2] / f"{key}.json"
            if other.exists():
                entry = json.loads(other.read_text(encoding="utf-8"))
                entry["origin"] = "alignment_cache"
                self.J.atomic_json(path, entry)
                self.counters["alignment_cache_hits"] += 1
                return entry
        j = self._judge()
        ids, prompt_tokens = j.generate_ids(messages, cap)
        entry = {"key": key, "kind": kind, "cap": cap, "prompt_tokens": prompt_tokens, "output_ids": ids, "output_tokens": len(ids),
                 "text": j.text(ids), "origin": "generated"}
        if kind == "judgment" and len(ids) > self.J.OFFICIAL_CAP:
            entry["text_512"] = j.text(ids[:self.J.OFFICIAL_CAP])
        self.J.atomic_json(path, entry)
        self.counters["judged_now" if kind == "judgment" else "extractions_now"] += 1
        return entry

    def stage(self, builder, extraction_builder, row, answers, position, turn):
        """One stage judgment in the adopted view (first-pass cap N + official extraction + Rating fallback)."""
        messages = build_messages(builder, row, answers, position, turn, None)
        entry = self.cached(messages, self.cap, "judgment")
        text = entry["text"]
        extraction = None
        if extract_official_winner(text) is None:
            extraction = self.cached(extraction_builder(text), self.J.EXTRACTION_CAP, "extraction")
        parsed = self.J.parse_view(text, extraction)
        return {"judge_turn": turn, "position": position, "judgment_cache_key": entry["key"], "origin": entry.get("origin", "generated"),
                "prompt_tokens": entry["prompt_tokens"], "output_tokens": entry["output_tokens"],
                "hit_cap_N": entry["output_tokens"] >= self.cap, "raw_response": text,
                "winner": parsed["winner"], "parse_stage": parsed["parse_stage"],
                "extraction_text": extraction["text"] if extraction else None,
                "model_wins": (parsed["winner"] == ("A" if position == 0 else "B")) if parsed["winner"] is not None else None}


def plan_stages(predictions):
    """What to do with each stage given the answers that exist (None = the request failed).

    -> list of (turn name, action): "judge", "judge_with_missing_later_answer" or "generation_failed".
    """
    out = []
    for k, turn in enumerate(C_TURNS, start=1):
        if any(predictions[i] is None for i in range(k)):
            out.append((turn, "generation_failed"))
        elif any(p is None for p in predictions[k:]):
            out.append((turn, "judge_with_missing_later_answer"))
        else:
            out.append((turn, "judge"))
    return out


C_TURNS = ("_first_turn", "_second_turn", "_third_turn")


def load_answers(res, model, index):
    """{conversation id: {role: [answer or None] * 3}} from the committed main-phase conversations."""
    docs = C.committed(res, model, "main", index)
    out = {}
    for cid, doc in docs.items():
        by = C.rows_by_key(doc)
        out[cid] = {role: [by[(role, t)]["prediction"] if by[(role, t)]["status"] == "ok" else None for t in (1, 2, 3)] for role in C.MAIN_ROLES}
    return out


def judge_model(args, J):
    res, model = args.res.resolve(), args.model
    C.verify_freeze(res)
    raw_rows, _ = C.load_index()
    limit = C.dev_limit(res)
    answers = load_answers(res, model, raw_rows)
    if not C.phase_complete(res, model, "main"):
        raise SystemExit(f"the main phase of {model} is not complete")
    conversations = [r for r in raw_rows if r["conversation_id"] in answers]
    if not limit and len(conversations) != C.N_CONV:
        raise SystemExit(f"{len(conversations)} committed conversations, {C.N_CONV} expected")
    rows = {r["conversation_id"]: normalize_row(r) for r in conversations}
    positions = load_pairwise(J.SOURCE, [rows[c]["source_row_index"] for c in rows])
    builder, _ = load_official_prompt_builder(J.SOURCE)
    extraction_builder = J.load_official_extraction_builder(J.SOURCE)
    out = res / "judge" / model
    out.mkdir(parents=True, exist_ok=True)
    config = {"schema": "convbench-main-judge-v1", "model": model, "cap": args.cap, "extraction_cap": J.EXTRACTION_CAP,
              "judge_checkpoint": str(J.CHECKPOINT), "official_evaluator_sha256": C.sha(J.SOURCE / "ConvBenchEval/convbencheval.py"),
              "official_prompts_sha256": C.sha(J.SOURCE / "ConvBenchEval/prompts.py"),
              "official_pairwise_sha256": C.sha(J.SOURCE / "ConvBenchEval/pairwise.npy"), "index_sha256": C.sha(C.INDEX),
              "judge_script_214_sha256": C.sha(ROOT / "scripts/214_convbench_alignment_judge.py"),
              "roles": C.MAIN_ROLES, "conversations": len(conversations), "denominator": C.N_CONV if not limit else len(conversations),
              "reuse_alignment_cache": bool(args.reuse_alignment_cache)}
    previous = C.read_json(out / "config.json")
    if previous is not None and previous != config:
        raise SystemExit(f"judge config mismatch; refusing to resume into {out}")
    C.atomic_json(out / "config.json", config)
    rt = Runtime(J, res / "judge_cache", C.ALIGN_CACHE if args.reuse_alignment_cache else None, args.cap)
    rel = C.res_rel(res)
    total = len(conversations) * len(C.MAIN_ROLES)
    started = time.time(); fresh_units = 0; done = 0
    records = []
    for conv in conversations:
        cid = conv["conversation_id"]; row = rows[cid]; position = positions[row["source_row_index"]]
        for role in C.MAIN_ROLES:
            checkpoint = out / "judgments" / role / f"{row['source_row_index']:04d}.json"
            if checkpoint.exists():
                records.extend(json.loads(checkpoint.read_text(encoding="utf-8"))["records"])
                done += 1
                continue
            before = rt.counters["judged_now"]
            preds = answers[cid][role]
            recs = []
            for turn, action in plan_stages(preds):
                base = {"conversation_id": cid, "source_row_index": row["source_row_index"], "role": role, "method": C.DISPLAY[role],
                        "judge_turn": turn, "position": position, "action": action}
                if action == "generation_failed":
                    recs.append({**base, "winner": None, "parse_stage": "generation_failed", "model_wins": None, "origin": None,
                                 "output_tokens": None, "hit_cap_N": None, "raw_response": None})
                    continue
                filled = [p if p is not None else "" for p in preds]
                recs.append({**base, **rt.stage(builder, extraction_builder, row, filled, position, turn)})
            J.atomic_json(checkpoint, {"records": recs})
            records.extend(recs)
            done += 1
            if rt.counters["judged_now"] > before:
                fresh_units += 1
            if done % 4 == 0 or done == total:
                now = time.time()
                rate = (now - started) / fresh_units if fresh_units else None
                st = STATUS.read(rel)
                after = float(st.get("estimated_seconds_after_this_stage") or 0.0)
                eta = now + rate * (total - done) if rate else None      # upper bound: assumes every remaining unit needs the judge
                STATUS.update(rel, stage=f"judge_{model}", model=C.MODEL_NAME[model], phase="judge", judge_units_done=done, judge_units_total=total,
                              judge_counters=rt.counters, stage_eta_utc=C.utc(eta) if eta else None,
                              eta_utc=C.utc(eta + after) if eta else st.get("eta_utc"), python_pid=__import__("os").getpid())
        if (conv["ordinal"] + 1) % 20 == 0:
            print(json.dumps({"judge": model, "done": done, "of": total, **rt.counters}), flush=True)

    denominator = config["denominator"]
    summary = {"model": model, "conversations": len(conversations), "denominator": denominator, "cap": args.cap, **rt.counters, "roles": {}}
    for role in C.MAIN_ROLES:
        mine = [r for r in records if r["role"] == role]
        view = [{"conversation_id": r["conversation_id"], "source_row_index": r["source_row_index"], "method": role,
                 "judge_turn": r["judge_turn"], "position": r["position"], "winner": r["winner"]} for r in mine]
        score = score_records(view, expected_per_method=denominator, judge_turns=C_TURNS)[role]
        stages = {}
        for turn, label in STAGE.items():
            group = [r for r in mine if r["judge_turn"] == turn]
            stages[label] = {s: sum(r["parse_stage"] == s for r in group)
                             for s in ("first_pass", "second_pass", "rating_fallback", "unresolved", "generation_failed")}
            stages[label]["judged_with_missing_later_answer"] = sum(r["action"] == "judge_with_missing_later_answer" for r in group)
            stages[label]["hit_cap_N"] = sum(bool(r["hit_cap_N"]) for r in group)
            stages[label]["from_alignment_cache"] = sum(r.get("origin") == "alignment_cache" for r in group)
        summary["roles"][role] = {"method": C.DISPLAY[role], "S1": score["S1"], "S2": score["S2"], "S3": score["S3"], "Avg": score["Avg"],
                                  "model_wins": score["model_wins_by_turn"], "parse_stages": stages,
                                  "unresolved": {k: v["unresolved"] for k, v in stages.items()},
                                  "generation_failed": {k: v["generation_failed"] for k, v in stages.items()}}
    C.atomic_json(out / "summary.json", summary)
    with open(out / "judge_raw.jsonl.tmp", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    (out / "judge_raw.jsonl.tmp").replace(out / "judge_raw.jsonl")
    (out / "DONE").write_text(C.utc() + "\n")
    print(json.dumps({"judge": model, "scores": {r: {k: round(v[k], 2) for k in ("S1", "S2", "S3", "Avg")} for r, v in summary["roles"].items()},
                      **rt.counters}, ensure_ascii=False), flush=True)


def check(args, J):
    """Gate 6: fresh judgments (empty cache) of alignment ReComp answers == the judgments recorded by the alignment run."""
    res = args.res.resolve()
    config = C.verify_freeze(res)
    raw_rows, _ = C.load_index()
    ordinals = (list(C.GATE_ORDINALS) + list(C.PILOT_ORDINALS))[:CHECK_CONVERSATIONS]
    limit = C.dev_limit(res)
    if limit:
        ordinals = ordinals[:limit]
    builder, _ = load_official_prompt_builder(J.SOURCE)
    extraction_builder = J.load_official_extraction_builder(J.SOURCE)
    cache = res / "gates" / "judge_check_cache"
    if cache.exists() and any(cache.iterdir()) and not (res / "gates" / "judge_check.partial").exists():
        raise SystemExit("judge check cache is not empty")
    (res / "gates").mkdir(exist_ok=True)
    (res / "gates" / "judge_check.partial").write_text(C.utc() + "\n")      # an interrupted check continues from its own outputs only
    rt = Runtime(J, cache, None, args.cap)
    out = {"gate": "gate6_judge_reproduces_alignment_judgments", "models": {}, "conversations_per_model": len(ordinals),
           "criterion": "for every re-judged stage: winner and parse stage equal to the alignment record (view capN); "
                        "identical judgment text is reported as well",
           "cache": "empty at the start (nothing read from the alignment cache)"}
    for model in C.MODELS:
        rows = [normalize_row(raw_rows[o]) for o in ordinals]
        positions = load_pairwise(J.SOURCE, [r["source_row_index"] for r in rows])
        compared = same_winner = same_stage = same_text = 0
        differences = []
        for row in rows:
            answers = [str(r["prediction"]) for r in C.alignment_rows(model, row["conversation_id"])]
            old = json.loads((C.ALIGN_JUDGE[model] / "judgments" / f"{row['source_row_index']:04d}.json").read_text(encoding="utf-8"))["records"]
            old = {r["judge_turn"]: r for r in old}
            for turn in C_TURNS:
                new = rt.stage(builder, extraction_builder, row, answers, positions[row["source_row_index"]], turn)
                o = old[turn]
                compared += 1
                w = new["winner"] == o["views"]["capN"]["winner"]; s = new["parse_stage"] == o["views"]["capN"]["parse_stage"]
                t = new["raw_response"] == o["raw_response"]
                same_winner += w; same_stage += s; same_text += t
                if not (w and s and t):
                    differences.append({"conversation_id": row["conversation_id"], "judge_turn": turn, "winner": [o["views"]["capN"]["winner"], new["winner"]],
                                        "parse_stage": [o["views"]["capN"]["parse_stage"], new["parse_stage"]], "text_identical": bool(t),
                                        "output_tokens": [o["output_tokens"], new["output_tokens"]]})
        out["models"][model] = {"status": "PASS" if compared and same_winner == compared and same_stage == compared else "FAIL",
                                "stage_judgments_compared": compared, "same_winner": same_winner, "same_parse_stage": same_stage,
                                "identical_text": same_text, "differences": differences,
                                "conversation_ids": [r["conversation_id"] for r in rows]}
    out["judged_now"] = rt.counters["judged_now"]; out["cache_hits"] = rt.counters["cache_hits"]
    out["status"] = "PASS" if all(m["status"] == "PASS" for m in out["models"].values()) else "FAIL"
    out["config_sha256"] = config["_file_sha256"]; out["at_utc"] = C.utc()
    C.atomic_json(res / "gates" / "judge_check.json", out)
    (res / "gates" / "judge_check.partial").unlink()
    print(json.dumps({"gate6": out["status"], **{m: {k: v for k, v in d.items() if k != "differences"} for m, d in out["models"].items()}},
                     ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", type=Path, required=True)
    p.add_argument("--model", choices=C.MODELS)
    p.add_argument("--check", action="store_true")
    p.add_argument("--cap", type=int, default=C.JUDGE_CAP)
    p.add_argument("--reuse-alignment-cache", action="store_true")
    args = p.parse_args()
    J = load_214()
    if args.cap != C.JUDGE_CAP:
        p.error("the frozen first-pass cap is 2048")
    if args.check:
        return check(args, J)
    if args.model is None:
        p.error("--model or --check")
    judge_model(args, J)


if __name__ == "__main__":
    main()

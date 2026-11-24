#!/usr/bin/env python3
"""ConvBench alignment, stage 1: analysis of the stored September artifacts (no generation, no judging).

Reads only:
  runs/convbench_context/judge_full_577_stage_only/judge_raw.jsonl   (September judge records, 4 methods)
  results/convbench_full/generation_577/raw.jsonl                    (September generation rows)
  results/convbench_full/full_577/per_request_diagnostics.csv        (September repetition flags)
  data/convbench/reference_token_lengths.json                        (reference answer token lengths)
Writes <out>/stage1/*.{json,csv,md}.  Nothing outside <out> is touched.

Unresolved classification (543 records, parse_stage == "unresolved"):
  (a) the first-pass judgment used the whole 512-token output cap and contains no conclusion
  (b) a conclusion naming one winner is present, but not in a form the September parser accepted
  (c) the judgment ended by itself (below the cap) without choosing a winner (tie / cannot decide)
  (d) everything else (read by hand; listed in full in unresolved_other.md)
The rules are heuristics over the judge text; they classify, they never change a score.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JUDGE_RAW = ROOT / "runs/convbench_context/judge_full_577_stage_only/judge_raw.jsonl"
GEN_RAW = ROOT / "results/convbench_full/generation_577/raw.jsonl"
DIAG = ROOT / "results/convbench_full/full_577/per_request_diagnostics.csv"
REF_LEN = ROOT / "data/convbench/reference_token_lengths.json"
JUDGE_CAP = 512
METHODS = ("ReComp", "FullLoad", "Prefix25", "Prefix45")
STAGES = {"_first_turn": "S1", "_second_turn": "S2", "_third_turn": "S3"}

CUE = re.compile(
    r"\b(overall|conclu\w+|therefore|thus|in summary|to summarize|based on|i would say|i think|i believe|"
    r"i choose|i prefer|i would choose|i would rate|my (?:final )?(?:judg\w+|answer|choice|decision|verdict)|"
    r"ultimately|final answer|in the end|taking (?:all|everything)|considering (?:all|these|the above)|"
    r"given (?:these|the above|this)|as a result|hence)\b", re.I)
WIN = re.compile(
    r"\b(better|superior|stronger|preferable|preferred|prefer|winner|wins|outperforms?|more (?:accurate|suitable|"
    r"appropriate|effective|comprehensive|relevant)|the best|choose|select)\b", re.I)
WHO = re.compile(r"\b(?:assistant|response)\s+([ab])\b", re.I)
TIE = re.compile(
    r"\b(tie\b|equally (?:good|valid|strong|accurate|effective)|both (?:responses|assistants)[^.\n]{0,80}"
    r"(?:equal|same quality|similar(?:ly)? (?:good|accurate))|neither (?:response|assistant)[^.\n]{0,60}(?:better|"
    r"correct|accurate)|cannot (?:determine|decide|conclude)|can't (?:determine|decide)|no clear winner|"
    r"difficult to (?:determine|say|decide) which|not possible to (?:determine|say|decide))", re.I)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


DIRECT = re.compile(
    r"\b(?:assistant|response)\s+([ab])(?:'s|\u2019s)?(?:\s+[\w-]+){0,4}?\s+(?:is|was|are|were|would be|seems|appears|"
    r"provides|offers|gives|does|did|has|performs)\s+(?:[\w-]+\s+){0,4}?(?:better|superior|stronger|preferable|"
    r"the best|the winner|the stronger|the more)\b", re.I)
CHOOSE = re.compile(
    r"\b(?:choose|prefer|select|favor|favour|go with|side with|better (?:response|answer|one) is|winner is)\s+"
    r"(?:the\s+)?(?:[\w-]+\s+){0,3}?(?:assistant|response)\s+([ab])\b", re.I)


def verdict(text: str) -> tuple[str | None, str | None]:
    """Last conclusion sentence (cue + winning word) that names one winner.

    Order of evidence inside such a sentence: "<side> ... is ... better" / "choose <side>" (last such match);
    otherwise a sentence that names only one side; otherwise "<side> ... better than ...".
    """
    found = (None, None)
    for sentence in sentences(text):
        if not (CUE.search(sentence) and WIN.search(sentence)):
            continue
        direct = [m.group(1).upper() for m in DIRECT.finditer(sentence)] + \
                 [m.group(1).upper() for m in CHOOSE.finditer(sentence)]
        if direct:
            if len(set(direct)) == 1:
                found = (direct[0], sentence)
            continue
        who = [m.group(1).upper() for m in WHO.finditer(sentence)]
        if not who:
            continue
        if len(set(who)) == 1:
            found = (who[0], sentence)
            continue
        win_at = WIN.search(sentence).start()
        before = [m.group(1).upper() for m in WHO.finditer(sentence) if m.start() < win_at]
        negated = re.search(r"\b(not|n't|less|worse|inferior)\b[^.]{0,40}$", sentence[:win_at], re.I)
        if before and not negated and re.search(r"\bthan\b", sentence[win_at:], re.I):
            found = (before[-1], sentence)
    return found


def classify(record: dict) -> dict:
    raw = record["raw_response"]
    truncated = int(record["output_tokens"]) >= JUDGE_CAP
    low = raw.lower()
    both_phrases = ("response a is better" in low) and ("response b is better" in low)
    winner, sentence = verdict(raw)
    tie = TIE.search(raw)
    if winner is not None:
        cls = "b"
    elif truncated:          # the text stops at the cap before any conclusion
        cls = "a"
    elif tie:
        cls = "c"
    else:
        cls = "d"
    return {"class": cls, "truncated_at_cap": truncated, "both_official_phrases": both_phrases,
            "heuristic_winner": winner, "evidence": sentence or (tie.group(0) if tie else None)}


def pct(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def dist(values: list[float]) -> dict:
    return {"n": len(values), "mean": statistics.mean(values), "p50": pct(values, .5), "p90": pct(values, .9),
            "p95": pct(values, .95), "p99": pct(values, .99), "max": max(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out / "stage1"
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 1. unresolved classification
    records = [json.loads(line) for line in JUDGE_RAW.open(encoding="utf-8")]
    unresolved = [r for r in records if r["parse_stage"] == "unresolved"]
    rows = []
    for r in unresolved:
        info = classify(r)
        rows.append({"method": r["method"], "stage": STAGES[r["judge_turn"]], "conversation_id": r["conversation_id"],
                     "output_tokens": r["output_tokens"], "prompt_tokens": r["prompt_tokens"], **info,
                     "second_pass_raw": r["extraction_raw_response"], "position": r["position"],
                     "raw_tail": r["raw_response"][-500:]})
    with (out / "unresolved_classification.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = collections.Counter((r["method"], r["stage"], r["class"]) for r in rows)
    lines = ["| 방법 | 단계 | 판정 불가 | (a) 상한에서 잘림 | (b) 결론 있음·해석 실패 | (c) 승자 미선택 | (d) 기타 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        for stage in ("S1", "S2", "S3"):
            c = [table[(method, stage, k)] for k in "abcd"]
            lines.append(f"| {method} | {stage} | {sum(c)} | {c[0]} | {c[1]} | {c[2]} | {c[3]} |")
        c = [sum(table[(method, s, k)] for s in ("S1", "S2", "S3")) for k in "abcd"]
        lines.append(f"| {method} | 합계 | {sum(c)} | {c[0]} | {c[1]} | {c[2]} | {c[3]} |")
    c = [sum(1 for r in rows if r["class"] == k) for k in "abcd"]
    lines.append(f"| 전체 | 합계 | {sum(c)} | {c[0]} | {c[1]} | {c[2]} | {c[3]} |")
    extra = {
        "unresolved_total": len(rows),
        "by_class": dict(zip("abcd", c)),
        "truncated_at_cap_among_unresolved": sum(r["truncated_at_cap"] for r in rows),
        "class_b_truncated": sum(r["class"] == "b" and r["truncated_at_cap"] for r in rows),
        "class_b_not_truncated": sum(r["class"] == "b" and not r["truncated_at_cap"] for r in rows),
        "class_b_second_pass_named_a_label": sum(
            r["class"] == "b" and bool(re.search(r"\b(?:is|label is|preference is)?:?\s*\n*\s*[AB]\b",
                                                 r["second_pass_raw"] or "")) for r in rows),
        "class_b_second_pass_starts_with_label": sum(
            r["class"] == "b" and bool(re.match(r"\s*[AB]\b", r["second_pass_raw"] or "")) for r in rows),
        "both_official_phrases_present": sum(r["both_official_phrases"] for r in rows),
        "all_records": len(records),
        "all_records_at_cap": sum(int(r["output_tokens"]) >= JUDGE_CAP for r in records),
        "first_pass_parsed_at_cap": sum(r["parse_stage"] == "first_pass" and int(r["output_tokens"]) >= JUDGE_CAP
                                        for r in records),
        "second_pass_recovered": sum(r["parse_stage"] == "second_pass" for r in records),
        "second_pass_output_cap_tokens": 16,
        "second_pass_outputs_at_16_tokens": sum((r.get("extraction_output_tokens") or 0) >= 16 for r in records),
    }
    # how the heuristic conclusion of class (b) relates to the model side (diagnostic only; not a score)
    b_rows = [r for r in rows if r["class"] == "b"]
    extra["class_b_heuristic_model_wins"] = {
        f"{m}/{s}": [sum(r["heuristic_winner"] == ("A" if r["position"] == 0 else "B")
                         for r in b_rows if r["method"] == m and r["stage"] == s),
                     sum(1 for r in b_rows if r["method"] == m and r["stage"] == s)]
        for m in METHODS for s in ("S1", "S2", "S3")}
    (out / "unresolved_summary.json").write_text(json.dumps(extra, indent=1, ensure_ascii=False) + "\n")
    (out / "unresolved_table.md").write_text("\n".join(lines) + "\n")

    rng = random.Random(20261121)
    ex = ["# 판정 불가 예시 (방법별·단계별·분류별 최대 3건, 고정 시드 무작위)\n",
          "판정 원문의 끝 500자와 9월 2차 추출 출력을 그대로 적는다.\n"]
    for method in METHODS:
        for stage in ("S1", "S2", "S3"):
            for cls in "abcd":
                group = [r for r in rows if (r["method"], r["stage"], r["class"]) == (method, stage, cls)]
                if not group:
                    continue
                ex.append(f"\n## {method} / {stage} / ({cls}) — {len(group)}건\n")
                for r in rng.sample(group, min(3, len(group))):
                    ex.append(f"- `{r['conversation_id']}` 출력 {r['output_tokens']}토큰, 모델 답 위치 "
                              f"{'A' if r['position'] == 0 else 'B'}, 규칙이 읽은 결론: {r['heuristic_winner']}")
                    ex.append(f"  - 판정 원문 끝: `{json.dumps(r['raw_tail'][-320:], ensure_ascii=False)}`")
                    ex.append(f"  - 9월 2차 추출 출력: `{json.dumps(r['second_pass_raw'], ensure_ascii=False)}`")
    (out / "unresolved_examples.md").write_text("\n".join(ex) + "\n")
    other = ["# (c)·(d)로 분류된 판정 불가 전체 (직접 확인용)\n"]
    for r in rows:
        if r["class"] in "cd":
            other.append(f"- ({r['class']}) {r['method']} {r['stage']} `{r['conversation_id']}` "
                         f"{r['output_tokens']}토큰: `{json.dumps(r['raw_tail'][-400:], ensure_ascii=False)}`")
    (out / "unresolved_other.md").write_text("\n".join(other) + "\n")

    # ------------------------------------------------------------------ 2. ReComp generation check
    gen = []
    with GEN_RAW.open(encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            if r["method"] == "ReComp":
                gen.append(r)
    flags = {}
    with DIAG.open(encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            if r["method"] == "ReComp":
                flags[(r["conversation_id"], int(r["turn_id"]))] = (r["degenerate_repetition"] == "True",
                                                                   r["numeric_repetition_review"] == "True")
    ref = {row["conversation_id"]: row for row in json.loads(REF_LEN.read_text())["rows"]}
    check = {"requests": len(gen), "turns": {}}
    md = ["| 턴 | 요청 | 답 길이(토큰) 평균 / p50 / p90 / p95 / p99 / 최대 | 기준 답 길이(토큰) 평균 / p50 / p90 / p95 / p99 / 최대 | "
          "상한 도달(1024) | 상한 도달(문맥으로 낮춘 상한) | 문맥으로 상한이 낮아진 요청 | 반복 표시 | 숫자 반복 검토 표시 | 입력 > 4096 | 입력 p50 / 최대 |",
          "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for turn in (1, 2, 3):
        group = [r for r in gen if r["turn_id"] == turn]
        g = [r["generated_tokens"] for r in group]
        rf = [ref[r["conversation_id"]][f"reference_A{turn}_tokens"] for r in group]
        inp = [r["context_input_tokens"] for r in group]
        item = {
            "requests": len(group), "generated_tokens": dist(g), "reference_tokens": dist(rf),
            "cap_reached_at_1024": sum(r["generation_cap_reached"] and not r["generation_cap_clamped_by_context"]
                                       for r in group),
            "cap_reached_at_context_clamped_cap": sum(r["generation_cap_reached"] and
                                                      r["generation_cap_clamped_by_context"] for r in group),
            "cap_clamped_by_context": sum(r["generation_cap_clamped_by_context"] for r in group),
            "repetition_flag": sum(flags[(r["conversation_id"], turn)][0] for r in group),
            "numeric_repetition_review_flag": sum(flags[(r["conversation_id"], turn)][1] for r in group),
            "input_above_4096": sum(v > 4096 for v in inp), "input_tokens": dist(inp),
            "cap_reached_and_repetition_flag": sum(r["generation_cap_reached"] and flags[(r["conversation_id"], turn)][0]
                                                   for r in group),
            "clamped_cap_reached_ids": [[r["conversation_id"], r["context_input_tokens"], r["effective_max_new_tokens"]]
                                        for r in group if r["generation_cap_reached"] and
                                        r["generation_cap_clamped_by_context"]],
        }
        check["turns"][f"T{turn}"] = item
        f = lambda d: (f"{d['mean']:.0f} / {d['p50']:.0f} / {d['p90']:.0f} / {d['p95']:.0f} / {d['p99']:.0f} / "
                       f"{d['max']:.0f}")
        md.append(f"| T{turn} | {len(group)} | {f(item['generated_tokens'])} | {f(item['reference_tokens'])} | "
                  f"{item['cap_reached_at_1024']} | {item['cap_reached_at_context_clamped_cap']} | "
                  f"{item['cap_clamped_by_context']} | {item['repetition_flag']} | "
                  f"{item['numeric_repetition_review_flag']} | {item['input_above_4096']} | "
                  f"{item['input_tokens']['p50']:.0f} / {item['input_tokens']['max']:.0f} |")
    total_cap = sum(r["generation_cap_reached"] for r in gen)
    check["cap_reached_total"] = total_cap
    check["cap_reached_rate"] = total_cap / len(gen)
    check["input_above_4096_total"] = sum(r["context_input_tokens"] > 4096 for r in gen)
    check["repetition_flag_total"] = sum(v[0] for v in flags.values())
    check["conversations_with_any_cap_reached"] = len({r["conversation_id"] for r in gen if r["generation_cap_reached"]})
    md.append(f"\n합계: 상한 도달 {total_cap}/{len(gen)} ({100 * total_cap / len(gen):.2f}%), "
              f"그중 문맥으로 낮춘 상한 {sum(v['cap_reached_at_context_clamped_cap'] for v in check['turns'].values())}건; "
              f"반복 표시 {check['repetition_flag_total']}건; 4096 초과 입력 {check['input_above_4096_total']}건; "
              f"상한에 한 번이라도 도달한 대화 {check['conversations_with_any_cap_reached']}개.")
    (out / "generation_check.json").write_text(json.dumps(check, indent=1, ensure_ascii=False) + "\n")
    (out / "generation_check.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"unresolved": extra, "generation": {k: v for k, v in check.items() if k != "turns"}},
                     indent=1, ensure_ascii=False))
    print("\n".join(lines))
    print("\n".join(md))


if __name__ == "__main__":
    main()

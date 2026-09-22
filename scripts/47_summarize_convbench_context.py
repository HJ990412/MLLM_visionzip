#!/usr/bin/env python3
"""Summarize the complete CPU ConvBench context preflight without model execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "data/convbench/context_preflight.json"
INDEX = ROOT / "data/convbench/index.json"
OUTPUT_JSON = ROOT / "data/convbench/context_risk_report.json"
OUTPUT_MD = ROOT / "data/convbench/context_risk_report.md"


def percentile_linear(values: list[int], percentile: int) -> float:
    """Match the linear interpolation convention used by numpy.percentile."""
    if not values or not 0 <= percentile <= 100:
        raise ValueError("nonempty values and a percentile in [0, 100] required")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    fraction = position - lower
    return round(ordered[lower] * (1 - fraction) +
                 ordered[min(lower + 1, len(ordered) - 1)] * fraction, 2)


def summarize(rows: list[dict], context_limit: int, nominal_cap: int) -> dict:
    if not rows or context_limit <= 0 or nominal_cap <= 0:
        raise ValueError("rows, context limit, and nominal cap must be positive")
    if len({row["conversation_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate conversation ID in preflight")
    turns = []
    for turn_index in range(3):
        samples = [(row["conversation_id"],
                    row["reference_history_input_tokens"][turn_index])
                   for row in rows]
        if any(not isinstance(value, int) or value < 0 for _, value in samples):
            raise ValueError("invalid input token count")
        input_over = [{"conversation_id": conv_id, "input_tokens": value}
                      for conv_id, value in samples if value > context_limit]
        nominal_over = [{"conversation_id": conv_id, "input_tokens": value}
                        for conv_id, value in samples
                        if value + nominal_cap > context_limit]
        observed_flags = [row["input_plus_1024_exceeds_4096"][turn_index]
                          for row in rows]
        expected_flags = [value + nominal_cap > context_limit
                          for _, value in samples]
        if observed_flags != expected_flags:
            raise ValueError("preflight nominal-cap flags disagree with token counts")
        lengths = [value for _, value in samples]
        turns.append({
            "turn": turn_index + 1,
            "measurement": ("exact combined processor with image" if turn_index == 0
                            else "reference-history proxy: tokenized text plus T1 image expansion"),
            "n": len(samples),
            "input_tokens": {
                "p50": percentile_linear(lengths, 50),
                "p90": percentile_linear(lengths, 90),
                "p95": percentile_linear(lengths, 95),
                "p99": percentile_linear(lengths, 99),
                "max": max(lengths),
            },
            "input_exceeds_context_count": len(input_over),
            "input_exceeds_context_cases": input_over,
            "input_plus_nominal_cap_exceeds_context_count": len(nominal_over),
            "input_plus_nominal_cap_exceeds_context_cases": nominal_over,
        })
    return {
        "schema_version": "convbench-context-risk-report-v1",
        "n_conversations": len(rows),
        "context_limit": context_limit,
        "nominal_generation_cap": nominal_cap,
        "percentile_method": "linear interpolation at (n-1)*p/100, rounded to 2 decimals",
        "scope": "T1 exact; T2/T3 official reference-answer history proxy, not generated-history measurements or failure predictions",
        "turns": turns,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# ConvBench 577개 대화 문맥 위험 사전 점검",
        "",
        "이 보고서는 `context_preflight.json`의 전체 577개 대화를 재집계한다. "
        "T1은 이미지와 프롬프트를 함께 넣은 processor의 **정확한 입력 길이**다. "
        "T2·T3는 **공식 참조 답변 이력**을 토큰화하고 같은 이미지의 T1 확장분을 더한 "
        "**대리 추정치**다. 실제 실험의 생성 답변 이력과 다르므로 T2·T3 수치를 "
        "실제 실패 건수로 해석하면 안 된다.",
        "",
        f"모델 문맥 한도: **{report['context_limit']:,}토큰**. "
        f"명목 생성 상한: **{report['nominal_generation_cap']:,}토큰**. "
        "백분위는 정렬한 577개 값에서 `(n-1)×p/100` 위치를 선형 보간했다.",
        "",
        "| 턴 | 측정 | p50 | p90 | p95 | p99 | 최대 | 입력 > 4,096 | 입력 + 1,024 > 4,096 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["turns"]:
        p = item["input_tokens"]
        measurement = "정확" if item["turn"] == 1 else "참조 이력 대리치"
        lines.append(
            f"| T{item['turn']} | {measurement} | {p['p50']:g} | {p['p90']:g} | "
            f"{p['p95']:g} | {p['p99']:g} | {p['max']:,} | "
            f"{item['input_exceeds_context_count']} | "
            f"{item['input_plus_nominal_cap_exceeds_context_count']} |")
    lines += [
        "",
        "`입력 > 4,096`은 입력 자체가 한도를 넘는 사례다. "
        "`입력 + 1,024 > 4,096`은 명목 생성 상한을 그대로 쓸 수 없는 사례다. "
        "후자는 요청별 유효 생성 상한을 줄이면 실행 가능할 수도 있다.",
        "",
        "## 입력 자체가 문맥 한도를 넘는 사례",
        "",
    ]
    for item in report["turns"]:
        cases = item["input_exceeds_context_cases"]
        cases_text = ", ".join(f"{c['conversation_id']} ({c['input_tokens']:,})"
                               for c in cases) or "없음"
        lines.append(f"- T{item['turn']}: {cases_text}")
    lines += [
        "",
        "참조 이력의 특히 긴 답변은 대리치 최대값을 크게 만든다. "
        "실제 생성 이력에서는 같은 대화의 길이와 초과 여부가 달라질 수 있다. "
        "전체 실행 전에는 실제 생성 답변 기반 스트레스 검증과 문맥 정책 확인이 필요하다.",
        "",
        "## 재현",
        "",
        "`python scripts/47_summarize_convbench_context.py`",
        "",
        f"입력 SHA-256: `{report['source_sha256']['preflight']}` "
        f"(`context_preflight.json`), `{report['source_sha256']['index']}` (`index.json`).",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    preflight_bytes = PREFLIGHT.read_bytes()
    index_bytes = INDEX.read_bytes()
    source = json.loads(preflight_bytes)
    index = json.loads(index_bytes)
    rows = source["rows"]
    dialogs = index["conversations"]
    if len(rows) != 577 or source["n_conversations"] != len(rows) or len(dialogs) != len(rows):
        raise ValueError("expected exactly 577 conversations in both inputs")
    if [(row["conversation_id"], row["source_row_index"]) for row in rows] != [
        (dialog["conversation_id"], dialog["source_row_index"])
        for dialog in dialogs
    ]:
        raise ValueError("preflight row order does not match official index")
    context = source["model_context_tokens"]
    cap = source["nominal_generation_cap"]
    if (context, cap) != (4096, 1024):
        raise ValueError("unexpected model context or nominal generation cap")
    report = summarize(rows, context, cap)
    report["source_sha256"] = {
        "preflight": hashlib.sha256(preflight_bytes).hexdigest(),
        "index": hashlib.sha256(index_bytes).hexdigest(),
    }
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    OUTPUT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {OUTPUT_JSON.relative_to(ROOT)} and {OUTPUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

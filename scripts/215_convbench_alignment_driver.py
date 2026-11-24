#!/usr/bin/env python3
"""ConvBench baseline alignment: orchestrator for stages 2-5 and the diagnostics (notes.md has the plan).

Started by scripts/run_convbench_alignment.sh (detached, locked).  Runs the steps one after another as
subprocesses, never waits for input, and keeps <res>/status.json current.  Every step is resumable and a
finished step is skipped, so the same command continues after an interruption or a reboot.

  stage 2   re-judge the stored September ReComp answers with the fixed judge (no generation)
  replay    new ReComp-only generator vs September answers on 12 conversations (must be identical)
  stage 3A  generation cap always 1024 (G1), September tokenization; judge
  stage 3B  G1 + official token ids (G2): the combined fix, all 577 x 3 regenerated; judge
  stage 4   LLaVA-1.5 ReComp with the stage-3B settings; judge
  stage 5   docs/convbench_baseline_alignment.md + report.md
  D1        FP16 weights (diagnostic, never adopted); skipped on GPU out-of-memory
  D2        sampling temperature 0.2 (diagnostic) - only if LLaVA-NeXT is outside the target range
  D3        LLaVA-1.5 official 'pad' preprocessing (diagnostic) - only if LLaVA-1.5 is outside the range
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = "/home/dblab/anaconda3/envs/mllm_ft/bin/python"
GEN = "scripts/213_convbench_recomp_generate.py"
JUDGE = "scripts/214_convbench_alignment_judge.py"
TARGET = {"next": {"S1": 7.95, "S2": 11.46, "S3": 7.58, "Avg": 9.00},
          "llava15": {"S1": 4.33, "S2": 5.72, "S3": 5.55, "Avg": 5.20}}
STAGE_TOL, AVG_TOL = 3.0, 2.0
REPLAY_IDS = "1,2,3,4,5,6,7,8,9,10,379,418"
JUDGE_CAP = 2048
STOP = {"requested": False}


def utc(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts if ts is not None else time.time()))


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def in_range(scores: dict | None, model: str) -> bool | None:
    if scores is None:
        return None
    t = TARGET[model]
    return (all(abs(scores[k] - t[k]) <= STAGE_TOL for k in ("S1", "S2", "S3"))
            and abs(scores["Avg"] - t["Avg"]) <= AVG_TOL)


def view(res: Path, judge_name: str, name: str = "capN") -> dict | None:
    summary = read_json(res / "judge" / judge_name / "summary.json")
    if not summary or not (res / "judge" / judge_name / "DONE").exists():
        return None
    return summary["views"][name]


# ---------------------------------------------------------------------------------------------- steps
def build_steps(res: Path) -> list[dict]:
    g, j, cache = res / "gen", res / "judge", res / "judge_cache"

    def gen(name: str, *extra: str) -> list[str]:
        return [PY, GEN, "--out", str(g / name), *extra]

    def judge(name: str, answers: str, *extra: str) -> list[str]:
        return [PY, JUDGE, "--answers", answers, "--out", str(j / name), "--cache", str(cache),
                "--cap", str(JUDGE_CAP), *extra]

    next_out = lambda: in_range(view(res, "next_3B_cap_tok"), "next") is False
    l15_out = lambda: in_range(view(res, "llava15_final"), "llava15") is False
    core = lambda: True
    return [
        dict(id="stage2_judge_sept", title="2단계: 9월 ReComp 답 재판정(고친 판정기)", core=True, when=core,
             cmd=judge("sept_answers", "sept", "--compare-sept-judge"), out=j / "sept_answers", est=4.6 * 3600),
        dict(id="replay_check", title="3단계 사전 검사: 새 생성 스크립트와 9월 답 일치(12대화)", core=True, when=core,
             cmd=gen("replay_sept", "--model", "next", "--cap-policy", "sept_clamp", "--tokenization", "sept",
                     "--source-ids", REPLAY_IDS, "--compare-sept"), out=g / "replay_sept", est=300),
        dict(id="stage3A_gen", title="3단계 A 생성: 상한 항상 1024(G1)", core=True, when=core,
             cmd=gen("next_3A_cap", "--model", "next", "--cap-policy", "official_1024", "--tokenization", "sept",
                     "--reuse-from-sept"), out=g / "next_3A_cap", est=600),
        dict(id="stage3A_judge", title="3단계 A 판정", core=True, when=core,
             cmd=judge("next_3A_cap", str(g / "next_3A_cap")), out=j / "next_3A_cap", est=900),
        dict(id="stage3B_gen", title="3단계 B 생성: G1 + 공식 토큰열(G2), 577대화 x 3턴", core=True, when=core,
             cmd=gen("next_3B_cap_tok", "--model", "next", "--cap-policy", "official_1024", "--tokenization",
                     "official"), out=g / "next_3B_cap_tok", est=2.7 * 3600),
        dict(id="stage3B_judge", title="3단계 B 판정", core=True, when=core,
             cmd=judge("next_3B_cap_tok", str(g / "next_3B_cap_tok")), out=j / "next_3B_cap_tok", est=4.6 * 3600),
        dict(id="stage4_gen", title="4단계 생성: LLaVA-1.5 ReComp 577대화 x 3턴", core=True, when=core,
             cmd=gen("llava15_final", "--model", "llava15", "--cap-policy", "official_1024", "--tokenization",
                     "official"), out=g / "llava15_final", est=1.5 * 3600),
        dict(id="stage4_judge", title="4단계 판정", core=True, when=core,
             cmd=judge("llava15_final", str(g / "llava15_final")), out=j / "llava15_final", est=4.6 * 3600),
        dict(id="stage5_freeze", title="5단계: 설정 동결 문서와 report.md", core=True, when=core, internal="freeze",
             est=30),
        dict(id="D1_fp16_gen", title="진단 D1 생성: FP16 가중치", core=False, when=core, oom_skip=True,
             cmd=gen("next_D1_fp16", "--model", "next", "--cap-policy", "official_1024", "--tokenization", "official",
                     "--precision", "fp16"), out=g / "next_D1_fp16", est=2.2 * 3600),
        dict(id="D1_fp16_judge", title="진단 D1 판정", core=False,
             when=lambda: (g / "next_D1_fp16" / "DONE").exists(),
             cmd=judge("next_D1_fp16", str(g / "next_D1_fp16")), out=j / "next_D1_fp16", est=4.6 * 3600),
        dict(id="D2_sample_gen", title="진단 D2 생성: temperature 0.2 샘플링(LLaVA-NeXT가 범위 밖일 때만)", core=False,
             when=next_out, cmd=gen("next_D2_sample", "--model", "next", "--cap-policy", "official_1024",
                                    "--tokenization", "official", "--decoding", "sample_t02"),
             out=g / "next_D2_sample", est=2.7 * 3600),
        dict(id="D2_sample_judge", title="진단 D2 판정", core=False,
             when=lambda: (g / "next_D2_sample" / "DONE").exists(),
             cmd=judge("next_D2_sample", str(g / "next_D2_sample")), out=j / "next_D2_sample", est=4.6 * 3600),
        dict(id="D3_pad_gen", title="진단 D3 생성: LLaVA-1.5 공식 pad 전처리(LLaVA-1.5가 범위 밖일 때만)", core=False,
             when=l15_out, cmd=gen("llava15_D3_pad", "--model", "llava15", "--cap-policy", "official_1024",
                                   "--tokenization", "official", "--image-pad", "official"),
             out=g / "llava15_D3_pad", est=1.5 * 3600),
        dict(id="D3_pad_judge", title="진단 D3 판정", core=False,
             when=lambda: (g / "llava15_D3_pad" / "DONE").exists(),
             cmd=judge("llava15_D3_pad", str(g / "llava15_D3_pad")), out=j / "llava15_D3_pad", est=4.6 * 3600),
        dict(id="final_report", title="report.md 최종 갱신", core=True, when=core, internal="report", est=30),
    ]


# ---------------------------------------------------------------------------------------------- report
def fmt(x) -> str:
    return "—" if x is None else f"{x:.2f}"


def sept_reference() -> dict:
    out = {}
    with (ROOT / "results/convbench_full/full_577/quality.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["method"] == "ReComp":
                out.update({k: float(row[k]) for k in ("S1", "S2", "S3", "Avg")})
    out["unresolved"] = {}
    out["model_wins"] = {}
    with (ROOT / "results/convbench_full/full_577/judge_diagnostics.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["method"] == "ReComp":
                out["unresolved"][row["stage"]] = int(row["unresolved"])
                out["model_wins"][row["stage"]] = int(row["model_wins"])
    return out


def row_cells(scores: dict | None) -> str:
    if scores is None:
        return "— | — | — | — | —"
    u = scores["unresolved"]
    return (f"{scores['S1']:.2f} | {scores['S2']:.2f} | {scores['S3']:.2f} | {scores['Avg']:.2f} | "
            f"{u['S1']} / {u['S2']} / {u['S3']}")


def diff_cells(a: dict | None, b: dict | None) -> str:
    if a is None or b is None:
        return "— | — | — | —"
    return " | ".join(f"{a[k] - b[k]:+.2f}" for k in ("S1", "S2", "S3", "Avg"))


def gen_line(res: Path, name: str) -> str:
    s = read_json(res / "gen" / name / "summary.json")
    if not s or not (res / "gen" / name / "DONE").exists():
        return f"| `{name}` | — | — | — | — | — | — |"
    c, k, a = s["cap_reached"], s["cap_clamped_by_context"], s["input_above_4096"]
    return (f"| `{name}` | {s['generated']} / {s['copied_from_september']} | "
            f"{c['1']} / {c['2']} / {c['3']} | {k['1']} / {k['2']} / {k['3']} | {a['1']} / {a['2']} / {a['3']} | "
            f"{s['max_input_tokens']} | {s['official_token_ids_mismatch']} |")


def verdict_text(scores: dict | None, model: str) -> str:
    if scores is None:
        return "아직 결과 없음"
    t = TARGET[model]
    parts = [f"{k} {scores[k] - t[k]:+.2f}" for k in ("S1", "S2", "S3")] + [f"평균 {scores['Avg'] - t['Avg']:+.2f}"]
    bad = [k for k in ("S1", "S2", "S3") if abs(scores[k] - t[k]) > STAGE_TOL]
    if abs(scores["Avg"] - t["Avg"]) > AVG_TOL:
        bad.append("평균")
    state = "**충족**" if not bad else f"**미충족** (범위를 벗어난 항목: {', '.join(bad)})"
    return f"{state} — MetaCompress 대비 차이 {', '.join(parts)}"


def se(p_percent: float, n: int = 577) -> float:
    p = p_percent / 100
    return 100 * math.sqrt(max(p * (1 - p), 0) / n)


def build_report(res: Path, status: dict) -> str:
    sept = sept_reference()
    V = lambda name, v="capN": view(res, name, v)
    s2_512, s2_N = V("sept_answers", "cap512"), V("sept_answers")
    a3, b3, l15 = V("next_3A_cap"), V("next_3B_cap_tok"), V("llava15_final")
    d1, d2, d3 = V("next_D1_fp16"), V("next_D2_sample"), V("llava15_D3_pad")
    s2_summary = read_json(res / "judge" / "sept_answers" / "summary.json") or {}
    stage1 = res / "stage1"
    steps = {s["id"]: s for s in status.get("steps", [])}
    L: list[str] = []
    A = L.append
    A("# ConvBench 기준 점수 점검 보고 — ReComp vs MetaCompress 기본 모델\n")
    A(f"결과 디렉터리 `{res.relative_to(ROOT)}` · 작성 {utc()} · 드라이버 상태 `{status.get('stage')}`\n")
    pending = [s["id"] for s in status.get("steps", []) if s["state"] in ("pending", "running")]
    if pending:
        A(f"> **아직 끝나지 않은 단계가 있다**: {', '.join(pending)}. 아래 표의 —는 결과가 아직 없다는 뜻이다. "
          "이 파일은 단계가 끝날 때마다 다시 쓰인다.\n")
    failed = [f"{s['id']} ({s.get('note', '')})" for s in status.get("steps", []) if s["state"] in ("failed", "skipped")]
    if failed:
        A(f"> 실행하지 않았거나 실패한 단계: {'; '.join(failed)}\n")

    # ------------------------------------------------------------------ 1
    A("## 1. 최종 대조표\n")
    A(f"목표(실행 전 고정): 단계별 차이 ±{STAGE_TOL:.1f}점, 평균 차이 ±{AVG_TOL:.1f}점. 분모 577, 판정 불가는 0승. "
      "판정 불가는 S1 / S2 / S3 건수.\n")
    A("| 모델 | 행 | S1 | S2 | S3 | 평균 | 판정 불가 |")
    A("|---|---|---:|---:|---:|---:|---|")
    t = TARGET["next"]
    A(f"| LLaVA-NeXT-7b | MetaCompress 기본 모델 | {t['S1']:.2f} | {t['S2']:.2f} | {t['S3']:.2f} | {t['Avg']:.2f} | 알 수 없음 |")
    A(f"| LLaVA-NeXT-7b | 9월 ReComp (9월 판정기) | {row_cells(sept)} |")
    A(f"| LLaVA-NeXT-7b | **최종 ReComp** (3B 생성 + 확정 판정기) | {row_cells(b3)} |")
    t = TARGET["llava15"]
    A(f"| LLaVA-1.5-7b | MetaCompress 기본 모델 | {t['S1']:.2f} | {t['S2']:.2f} | {t['S3']:.2f} | {t['Avg']:.2f} | 알 수 없음 |")
    A("| LLaVA-1.5-7b | 9월 ReComp | 실행한 적 없음 | | | | |")
    A(f"| LLaVA-1.5-7b | **최종 ReComp** (4단계) | {row_cells(l15)} |")
    A("")
    A(f"- LLaVA-NeXT-7b: {verdict_text(b3, 'next')}")
    A(f"- LLaVA-1.5-7b: {verdict_text(l15, 'llava15')}")
    A(f"- 참고(판정 기준에는 쓰지 않음): 승률 8% 근처에서 577대화의 이항 표준오차는 약 {se(8):.2f}점이다. "
      "우리 쪽과 MetaCompress 쪽 모두 단일 실행이다.\n")

    # ------------------------------------------------------------------ 2
    A("## 2. 1단계 결과 (저장된 자료 분석)\n")
    A("### 2.1 판정 불가 543건 분류 (9월 실행, 네 방법)\n")
    A((stage1 / "unresolved_table.md").read_text(encoding="utf-8"))
    summ = read_json(stage1 / "unresolved_summary.json") or {}
    A(f"- (a) 판정문이 512토큰 상한을 다 쓰고 결론 없이 끝남. (b) 결론 문장은 있으나 9월 해석기가 읽지 못함. "
      f"(c) 승자를 고르지 않음, (d) 기타는 0건.")
    A(f"- 상한에 닿은 판정 불가는 {summ.get('truncated_at_cap_among_unresolved')}건이고, 그중 "
      f"{summ.get('class_b_truncated')}건은 잘리기 전에 결론처럼 보이는 문장이 있어 (b)로 셌다(규칙 기반 분류라 항목별 중간 결론이 "
      f"섞여 있다). 상한에 닿지 않은 {summ.get('class_b_not_truncated')}건은 전부 (b)다.")
    A(f"- (b)의 원인: Llama는 \"Assistant A's response is better\"라고 써서 공식 1차 문구에 걸리지 않고, 9월의 2차 추출은 "
      f"공식 추출이 아닌 자체 프롬프트에 출력 16토큰·한 글자 완전 일치 규칙이었다. 2차 출력이 A/B로 시작했는데도 탈락한 (b)가 "
      f"{summ.get('class_b_second_pass_starts_with_label')}건이다.")
    A("- 방법별·단계별·분류별 예시 3건씩: `stage1/unresolved_examples.md`. 건별 분류: `stage1/unresolved_classification.csv`.\n")
    A("### 2.2 9월 ReComp 생성 답 점검\n")
    A((stage1 / "generation_check.md").read_text(encoding="utf-8"))
    A("- \"문맥으로 낮춘 상한\"은 9월 규칙 `min(1024, 4096 - 입력 + 1)` 때문에 1024보다 작아진 상한이다. "
      "반복 표시는 9월 `per_request_diagnostics.csv`의 표시를 그대로 셌다.\n")
    A("### 2.3 설정 대조표\n")
    A((stage1 / "config_comparison.md").read_text(encoding="utf-8").split("\n", 1)[1])

    # ------------------------------------------------------------------ 3
    A("\n## 3. 시도한 설정 전체의 점수표\n")
    A("판정기 열: `9월` = 9월 판정기(상한 512, 자체 2차 추출), `J1` = 상한 512 + 공식 2차 추출, "
      f"`J1+J2` = 상한 {JUDGE_CAP} + 공식 2차 추출(**확정 판정기**).\n")
    A("| # | 모델 | 생성 설정 (바꾼 항목) | 판정기 | S1 | S2 | S3 | 평균 | 판정 불가 | 채택 |")
    A("|---|---|---|---|---:|---:|---:|---:|---|---|")
    rows = [
        ("0", "NeXT", "9월 답 (바꾼 것 없음)", "9월", sept, "기준"),
        ("1", "NeXT", "9월 답", "J1", s2_512, "아니오 (참고: 공식 상한 유지)"),
        ("2", "NeXT", "9월 답", "J1+J2", s2_N, "판정기 확정"),
        ("3", "NeXT", "3A: 생성 상한 항상 1024 (G1)", "J1+J2", a3, "아니오 (항목별 효과용)"),
        ("4", "NeXT", "3B: G1 + 공식 토큰열 (G2)", "J1+J2", b3, "**예 (확정 생성 설정)**"),
        ("5", "NeXT", "D1: 3B + FP16 가중치", "J1+J2", d1, "아니오 (진단)"),
        ("6", "NeXT", "D2: 3B + temperature 0.2 샘플링", "J1+J2", d2, "아니오 (진단)"),
        ("7", "1.5", "4단계: 3B와 같은 설정", "J1+J2", l15, "**예**"),
        ("8", "1.5", "D3: 4단계 + 공식 pad 전처리", "J1+J2", d3, "아니오 (진단)"),
    ]
    not_run = {"5": "D1_fp16_judge", "6": "D2_sample_judge", "8": "D3_pad_judge"}
    for num, model, setting, judge_name, scores, adopt in rows:
        note = ""
        if scores is None and num in not_run:
            st = steps.get(not_run[num], {})
            gen_st = steps.get(not_run[num].replace("_judge", "_gen"), {})
            state = gen_st.get("state") if gen_st.get("state") in ("skipped", "not_needed", "failed") else st.get("state")
            note = {"not_needed": " (조건에 해당하지 않아 실행하지 않음)", "skipped": f" (건너뜀: {gen_st.get('note') or st.get('note')})",
                    "failed": f" (실패: {gen_st.get('note') or st.get('note')})"}.get(state, " (아직 실행 전)")
        A(f"| {num} | {model} | {setting}{note} | {judge_name} | {row_cells(scores)} | {adopt} |")
    A("\n같은 답을 공식 상한 512 보기(J1)로 읽은 점수 (참고):\n")
    A("| 실행 | S1 | S2 | S3 | 평균 | 판정 불가 |")
    A("|---|---:|---:|---:|---:|---|")
    for name, label in (("sept_answers", "9월 답"), ("next_3A_cap", "3A"), ("next_3B_cap_tok", "3B"),
                        ("next_D1_fp16", "D1"), ("next_D2_sample", "D2"), ("llava15_final", "LLaVA-1.5 4단계"),
                        ("llava15_D3_pad", "D3")):
        if V(name, "cap512") is not None:
            A(f"| {label} | {row_cells(V(name, 'cap512'))} |")
    A("\n생성 실행별 점검 (T1 / T2 / T3):\n")
    A("| 실행 | 새로 생성 / 9월 답 재사용 (대화) | 상한 도달 | 문맥으로 상한이 낮아진 요청 | 입력 > 4096 | 최대 입력 토큰 | 공식 토큰열 불일치 요청 |")
    A("|---|---|---|---|---|---:|---:|")
    for name in ("next_3A_cap", "next_3B_cap_tok", "llava15_final", "next_D1_fp16", "next_D2_sample", "llava15_D3_pad"):
        if (res / "gen" / name).exists():
            A(gen_line(res, name))
    A("\n- 3A의 \"공식 토큰열 불일치\"는 9월 토큰열을 일부러 유지한 실행이므로 새로 생성한 요청 전부가 불일치로 세어진다"
      "(9월 답을 재사용한 대화는 세지 않는다).")
    replay = read_json(res / "gen" / "replay_sept" / "summary.json")
    if replay and "september_comparison" in replay:
        c = replay["september_comparison"]
        A(f"- 사전 검사(replay): 새 생성 스크립트를 9월 설정으로 12대화에 돌린 답이 9월 답과 글자 단위로 같은 요청 "
          f"{c['identical']}/{c['requests_compared']}.")
    cmp_ = s2_summary.get("september_comparison")
    if cmp_:
        A("\n2단계 확인 — 고친 판정기가 9월에 판정 가능했던 건의 승패를 바꾸었는가 (ReComp 1,731건):\n")
        A(f"- 앞 512토큰 판정문이 9월 원문과 글자 단위로 같은 건: {cmp_['text512_identical_to_september']}/{cmp_['records']}"
          f" (다른 건 {len(cmp_['text512_differs'])}건).")
        A(f"- 9월에 판정 가능했던 건: {cmp_['september_resolved']}건.")
        for name, label in (("cap512", "J1 (상한 512)"), ("capN", f"J1+J2 (상한 {JUDGE_CAP}, 확정)")):
            c = cmp_[name]
            A(f"- {label}: 승자 같음 {c['same_winner']}건, **승자 바뀜 {len(c['winner_changed'])}건**, "
              f"판정 불가로 바뀜 {len(c['became_unresolved'])}건. 9월 판정 불가 중 이번에 판정된 건 "
              f"{c['september_unresolved_now_resolved']}건(그중 모델 승 {c['september_unresolved_now_model_win']}건), "
              f"여전히 판정 불가 {c['september_unresolved_still_unresolved']}건.")
            for item in c["winner_changed"][:20]:
                A(f"  - 바뀐 건: `{item[0]}` {item[1]} {item[2]}→{item[3]} (9월 {item[4]}, 이번 {item[5]}, "
                  f"앞 512토큰 원문 동일: {item[6]})")
            for item in c["became_unresolved"][:20]:
                A(f"  - 판정 불가로 바뀐 건: `{item[0]}` {item[1]} (9월 {item[2]}, {item[3]})")
        ot = s2_summary.get("output_tokens", {})
        A(f"- 판정문이 상한 {JUDGE_CAP}에도 닿은 건(S1 / S2 / S3): "
          f"{ot.get('hit_cap_N', {}).get('S1')} / {ot.get('hit_cap_N', {}).get('S2')} / {ot.get('hit_cap_N', {}).get('S3')}. "
          "사전에 정한 대로 상한을 더 올리지 않았다.")

    # ------------------------------------------------------------------ 4
    A("\n## 4. 바꾼 항목별 효과 (점, LLaVA-NeXT-7b ReComp)\n")
    A("| 항목 | 비교 | ΔS1 | ΔS2 | ΔS3 | Δ평균 |")
    A("|---|---|---:|---:|---:|---:|")
    A(f"| J1 판정 2차 추출을 공식대로 | (9월 답, J1) − (9월 답, 9월 판정기) | {diff_cells(s2_512, sept)} |")
    A(f"| J2 판정 출력 상한 512 → {JUDGE_CAP} | (9월 답, J1+J2) − (9월 답, J1) | {diff_cells(s2_N, s2_512)} |")
    A(f"| G1 생성 상한 항상 1024 | 3A − (9월 답, J1+J2) | {diff_cells(a3, s2_N)} |")
    A(f"| G2 공식 토큰열 | 3B − 3A | {diff_cells(b3, a3)} |")
    A(f"| 전체 (J1+J2+G1+G2) | 3B − (9월 답, 9월 판정기) | {diff_cells(b3, sept)} |")
    A("\n- G1과 G2는 순차 분해다(G2만 고친 전체 실행은 돌리지 않았다). 한 대화의 승패가 한 번 바뀌면 0.17점이다.")
    for name, label in (("sept_answers", "9월 답"), ("next_3A_cap", "3A"), ("next_3B_cap_tok", "3B")):
        v = V(name)
        if v:
            w = v["model_wins"]
            A(f"- 모델 승수 {label} (J1+J2): S1 {w['S1']}, S2 {w['S2']}, S3 {w['S3']} / 577.")
    A(f"- 모델 승수 9월(9월 판정기): S1 {sept['model_wins']['S1']}, S2 {sept['model_wins']['S2']}, "
      f"S3 {sept['model_wins']['S3']} / 577.")

    # ------------------------------------------------------------------ 5
    A("\n## 5. 양자화 진단 (D1, 채택하지 않음)\n")
    if d1 is not None:
        A("| | S1 | S2 | S3 | 평균 | 판정 불가 |")
        A("|---|---:|---:|---:|---:|---|")
        A(f"| 3B (NF4, 확정) | {row_cells(b3)} |")
        A(f"| D1 (FP16 가중치, 그 외 3B와 같음) | {row_cells(d1)} |")
        A(f"| D1 − 3B | {diff_cells(d1, b3)} | |")
        A("\n- FP16은 `torch.float16` 비양자화 가중치·FP16 계산, eager attention, greedy. 단일 실행이라 이 차이에는 "
          f"생성 내용이 달라져 생기는 판정 변동(표준오차 약 {se(8):.1f}점 규모)이 섞여 있다.")
    else:
        st = steps.get("D1_fp16_gen", {})
        A(f"- 결과 없음. 상태: `{st.get('state', 'pending')}` {st.get('note', '')}")

    # ------------------------------------------------------------------ 6
    A("\n## 6. 남은 차이에 대한 설명\n")
    A("**확인된 원인(고친 것)**\n")
    A(f"- J1·J2(판정 쪽): 판정 불가가 S1/S2/S3 {sept['unresolved']['S1']}/{sept['unresolved']['S2']}/{sept['unresolved']['S3']}건에서 "
      + (f"{s2_N['unresolved']['S1']}/{s2_N['unresolved']['S2']}/{s2_N['unresolved']['S3']}건으로 줄었다. 점수 효과는 4절."
         if s2_N else "(2단계 결과 대기)."))
    A("- G1(생성 상한의 문맥 조정)과 G2(LLaVA-NeXT 프롬프트 토큰열이 공식 LLaVA와 다름): 효과는 4절.")
    A("\n**추정이지만 확인하지 못한 원인**\n")
    if d2 is not None:
        A(f"- 생성 디코딩: 공식 래퍼는 temperature 0.2 샘플링이다. 진단 D2 − 3B = {diff_cells(d2, b3)} (S1 | S2 | S3 | 평균). 채택하지 않았다.")
    else:
        A("- 생성 디코딩: 공식 래퍼는 temperature 0.2 샘플링, 우리는 greedy. "
          + ("LLaVA-NeXT가 범위 안이라 진단 D2는 돌리지 않았다." if in_range(b3, "next") else "진단 D2 결과는 아직 없다."))
    if d3 is not None:
        A(f"- LLaVA-1.5 이미지 전처리: 공식은 pad, HF 기본값은 가운데 자르기. 진단 D3 − 4단계 = {diff_cells(d3, l15)}. 채택하지 않았다.")
    else:
        A("- LLaVA-1.5 이미지 전처리: 공식은 pad, HF 기본값은 가운데 자르기. "
          + ("LLaVA-1.5가 범위 안이라 진단 D3는 돌리지 않았다." if in_range(l15, "llava15") else "진단 D3 결과는 아직 없다."))
    A("- 양자화(NF4 vs FP16): 5절의 진단 값. 논문 전체 설정이라 바꾸지 않았다.")
    A("- 판정 모델 디코딩: 공식 호출은 temperature를 넘기지 않아 API 기본 샘플링이고 우리는 greedy다. "
      "MetaCompress가 Llama를 어떻게 디코딩했는지는 알 수 없다. 모델 답이 이기는 비율이 10% 미만인 상황에서 판정이 샘플링되면 "
      "승패가 뒤집히는 건이 생기므로 점수가 달라질 수 있다는 것은 추정일 뿐, 재지 않았다.")
    A("- 반복 출력: greedy에서 1024토큰 상한까지 반복하는 답이 남아 있다(3절 표의 상한 도달 수). 모델의 실제 출력이라 고치지 않았다.")
    A("\n**MetaCompress 쪽 설정을 알 수 없어 대조하지 못한 항목**\n")
    A("- 공개 저장소(commit `70a620e`)에는 README만 있다. 생성 길이 상한, 프롬프트 형식, 대화 이력 구성(생성 답인지 기준 답인지), "
      "디코딩, 모델 정밀도, 체크포인트 출처, 판정 모델 로딩·디코딩·출력 상한, 2차 추출 방식 전부 확인할 수 없다.")
    A("- **분모**: Table 1의 LLaVA-NeXT-7b Base 7.95 / 11.46 / 7.58은 k/577로 만들 수 없고, 세 값을 동시에 만드는 분모는 "
      "400..578 중 541뿐이다(43, 62, 41승). LLaVA-1.5-7b·13b와 LLaVA-NeXT-13b의 Base는 577로만 맞는다. "
      "즉 MetaCompress의 LLaVA-NeXT-7b 기본 모델 행은 36개 대화가 빠진 채 계산된 것으로 보이지만, 어떤 대화인지는 알 수 없다.")
    if b3:
        w = b3["model_wins"]
        A(f"  - 참고 환산(판정에는 쓰지 않음): MetaCompress 쪽을 분모 577로 바꾸면 43/62/41승 = 7.45 / 10.75 / 7.11 (평균 8.44). "
          f"우리 최종 승수는 {w['S1']} / {w['S2']} / {w['S3']}승이다.")
    A("- 판정 모델이 같은 Llama-3.1-8B-Instruct라도 revision, 정밀도, 채팅 템플릿이 다르면 판정문이 달라진다. 확인할 수 없다.")

    # ------------------------------------------------------------------ 7
    A("\n## 7. 동결한 설정과 근거\n")
    doc = ROOT / "docs/convbench_baseline_alignment.md"
    A(f"`docs/convbench_baseline_alignment.md`에 적었다" + ("." if doc.exists() else " (5단계가 끝나면 생성된다)."))
    A("요약: 생성 = NF4/BF16/eager, greedy, 상한 항상 1024, 공식 LLaVA 토큰열, 생성 답 이력, 자르지 않음. "
      f"판정 = Llama-3.1-8B-Instruct BF16 greedy, 공식 프롬프트·A/B 배치, 1차 출력 상한 {JUDGE_CAP}, 공식 2차 추출(상한 512), "
      "공식 `Rating:` 3차 규칙, 분모 577, 판정 불가 0승.")

    # ------------------------------------------------------------------ 8
    A("\n## 8. 코드, 지시와 다르게 한 것, 한계\n")
    A("**추가한 코드** (기존 파일은 고치지 않았다)\n")
    A("- `scripts/212_convbench_alignment_stage1.py` — 1단계 분석(판정 불가 분류, 생성 답 점검).")
    A("- `scripts/213_convbench_recomp_generate.py` — ReComp 전용 생성(SSD store 없음). 상한 정책·토큰열·디코딩·정밀도·pad 옵션.")
    A("- `scripts/214_convbench_alignment_judge.py` — 고친 판정기(공식 2차 추출, 상한 2048, 프롬프트 해시 캐시, 상한 512 보기 동시 산출).")
    A("- `scripts/215_convbench_alignment_driver.py`, `scripts/run_convbench_alignment.sh` — 분리 실행 드라이버, status.json, 보고서.")
    A("- `docs/convbench_baseline_alignment.md` — 동결 설정(5단계).")
    A("- 방법 코드(`mmimpress/`), 9월 스크립트(`scripts/41`, `42` 등), 기존 store·결과 디렉터리·동결 파일은 수정하지 않았다. "
      "`mmimpress/convbench_judge.py`의 함수는 불러 쓰기만 했다.")
    A("\n**지시와 다르게 했거나 지시에 없던 것**\n")
    A("- 생성 디코딩(greedy)과 LLaVA-1.5 이미지 전처리(HF 기본값)는 공식 코드와 다르지만, 정밀도처럼 논문 전체가 공유하는 설정으로 보고 "
      "채택 대상에서 빼고 조건부 진단(D2, D3)으로만 두었다. 지시문이 명시한 것은 정밀도뿐이므로 이것은 내 판단이다.")
    A("- 판정 디코딩(greedy)도 공식(API 기본 샘플링)과 다르지만 바꾸지 않았다. 재현 가능한 판정을 유지하기 위해서다.")
    A("- 3단계에서 \"항목별로 나눠 돌린다\"를 G2만 고친 전체 실행 대신 순차 분해(3A = G1만, 3B = G1+G2)로 대신했다. "
      "3A는 범위 충족 여부와 관계없이 항상 돌렸다(9월 답 재사용으로 비용이 작다).")
    A("- 3A는 577대화를 모두 새로 생성하지 않았다. greedy에서 문맥으로 낮춘 상한에 닿지 않은 대화는 답이 9월과 같으므로 9월 답을 "
      "그대로 썼다(근거: replay 검사). 3B·4단계·진단은 577대화 × 3턴을 모두 새로 생성했다.")
    A("- 판정 캐시: 같은 판정 프롬프트는 다시 판정하지 않고 저장된 출력을 썼다(같은 판정기·greedy이므로 출력이 같다. "
      "2단계에서 9월 원문과의 일치 수로 검증).")
    A("- 2단계의 J1 점수는 상한 512로 따로 돌린 것이 아니라 상한 2048 출력의 앞 512토큰으로 계산했다.")
    A("- 양자화 진단은 지시대로 FP16(`torch.float16`)으로 했다. 계산 dtype도 BF16에서 FP16으로 함께 바뀐다.")
    A("- MetaCompress 논문 PDF와 저장소는 대조 목적으로만 내려받았고 프로젝트에 복사하지 않았다.")
    A("\n**한계**\n")
    A("- 모든 점수는 단일 실행이다. 577대화에서 한 대화가 0.17점이고, 승률 5–10%의 표준오차가 1점 안팎이므로 "
      "1점 이하의 항목별 효과는 방향을 단정하기 어렵다.")
    A("- 1단계의 (a)/(b) 분류는 규칙 기반이다. 상한에 닿은 건 중 (b)로 센 96건에는 항목별 중간 결론이 섞여 있다.")
    A("- 판정 모델은 MetaCompress를 따라 Llama-3.1-8B-Instruct를 썼다. 공식 ConvBench의 GPT-3.5 판정 점수와는 비교할 수 없다.")
    A("- TTFT 등 시스템 지표는 이 작업에서 재지 않았다. `scripts/213`은 품질 점검용이며 시간 측정 경계가 없다.")
    A("- 본실험의 네 방법 실행기(`scripts/41` 계열)는 9월 정책(문맥 조정 상한, 9월 토큰열)을 코드에 고정하고 있다. "
      "동결 설정을 본실험에 쓰려면 그 실행기에 같은 두 항목을 반영해야 한다(이 작업에서는 방법 코드를 바꾸지 않았다).")
    A("\n## 단계 실행 기록\n")
    A("| 단계 | 상태 | 시작 (UTC) | 종료 (UTC) | 소요 | 비고 |")
    A("|---|---|---|---|---:|---|")
    for s in status.get("steps", []):
        dur = f"{s['seconds'] / 3600:.2f} h" if s.get("seconds") else ""
        A(f"| {s['id']} | {s['state']} | {s.get('started_utc') or ''} | {s.get('finished_utc') or ''} | {dur} | {s.get('note') or ''} |")
    return "\n".join(L) + "\n"


def build_freeze_doc(res: Path) -> str:
    V = lambda name, v="capN": view(res, name, v)
    b3, l15 = V("next_3B_cap_tok"), V("llava15_final")
    sept = sept_reference()
    L: list[str] = []
    A = L.append
    A("# ConvBench 기준 설정 동결 (baseline alignment)\n")
    A(f"작성 {utc()} · 근거 실행 `{res.relative_to(ROOT)}` (`report.md`, `notes.md`)\n")
    A("이 문서의 생성 설정과 판정 설정을 이후 ConvBench 본실험(네 방법 × 두 모델)에 그대로 쓴다. "
      "9월 실행(`results/convbench_full/full_577`)에서 바꾼 항목에는 근거를 적었다.\n")
    A("## 1. 생성 설정\n")
    A("| 항목 | 값 | 9월 대비 | 근거 |")
    A("|---|---|---|---|")
    A("| 모델 | LLaVA-NeXT-7b `llava-hf/llava-v1.6-vicuna-7b-hf` rev `c916e6c`; LLaVA-1.5-7b `llava-hf/llava-1.5-7b-hf` rev `b234b80` | 같음 / 신규 | |")
    A("| 정밀도 | NF4 4-bit(double quant), BF16 계산, eager attention | 같음 | 논문 전체 설정 |")
    A("| 디코딩 | greedy (`do_sample=False`) | 같음 | 논문 전체 설정. 공식 래퍼는 temperature 0.2 샘플링 |")
    A("| 생성 길이 상한 | `max_new_tokens = 1024`, 문맥 길이에 따른 조정 없음 | **바뀜** (9월: 입력 ≤ 4096이면 `min(1024, 4096 − 입력 + 1)`) | 결함: 낮춘 상한에 닿아 잘린 답 15건(9월 ReComp). 공식 `llava.py:53`에는 조정이 없음 |")
    A("| 프롬프트 | `llava_v1` 템플릿: `{system} USER: <image>\\n{Q1} ASSISTANT: {A1}</s>USER: {Q2} ASSISTANT: …` | 문자열 같음 | 공식 `llava.py:51, 134-179` |")
    A("| 프롬프트 토큰열 | 원본 LLaVA `tokenizer_image_token`(느린 legacy 토크나이저)과 같은 토큰 ID. LLaVA-NeXT HF 토크나이저에서는 `<image> \\n`, `</s> USER`로 렌더링해야 같아진다. LLaVA-1.5 HF 토크나이저는 템플릿 문자열 그대로 같다. 요청마다 일치를 검사한다 | **바뀜** (LLaVA-NeXT만: 9월은 `<image>` 뒤 `▁` 없음, `</s>` 뒤 `USER` 한 토큰) | 공식 코드와의 불일치. `scripts/213_convbench_recomp_generate.py`의 `OFFICIAL_RENDER`, `official_token_ids` |")
    A("| 대화 이력 | 같은 방법이 생성한 A1, A2. 이미지는 첫 턴에만 | 같음 | 공식 `run_multi_turn.py:178-188` |")
    A("| 답 후처리 | 특수 토큰 제거 후 `.strip()` | 같음 | 공식 `llava.py:145` |")
    A("| 문맥 한계 | 프롬프트·이력을 자르지 않는다. 입력이 4096을 넘어도 그대로 실행한다 | 같음 | 9월 `native_overflow_no_truncation_v1` |")
    A("| 이미지 전처리 | HF 프로세서 기본값(LLaVA-NeXT AnyRes, LLaVA-1.5 가운데 자르기) | 같음 | 논문 전체 설정. 공식 래퍼는 pad |")
    A("\n## 2. 판정 설정\n")
    A("| 항목 | 값 | 9월 대비 | 근거 |")
    A("|---|---|---|---|")
    A("| 판정 모델 | `meta-llama/Meta-Llama-3.1-8B-Instruct` rev `0e9e39f`, BF16, 비양자화, SDPA, greedy | 같음 | MetaCompress §6.1 각주 2 |")
    A("| 판정 프롬프트 | 공식 `generate_request` (S1·S2·S3), 이미지는 넣지 않음 | 같음 | 공식 `convbencheval.py:108-152` |")
    A("| A/B 배치 | 공식 `pairwise.npy` | 같음 | |")
    A(f"| 1차 판정 출력 상한 | {JUDGE_CAP}토큰 | **바뀜** (9월·공식: 512) | 결함: Llama 판정문 6,924건 중 506건이 512에 닿았고 판정 불가 543건 중 428건이 그 경우 |")
    A("| 승자 해석 1차 | 소문자 판정문에 `response a is better` / `response b is better` 중 정확히 하나 | 같음 | 공식 `convbencheval.py:214-219` |")
    A("| 승자 해석 2차 | 공식 `generate_parsing_answer_request`(공식 system 프롬프트 + 예시 3개)를 같은 Llama에 보내고, 출력에 `Final Answer: Response A` / `B` 중 정확히 하나. 출력 상한 512 | **바뀜** (9월: 자체 프롬프트, 16토큰, 한 글자 완전 일치) | 공식 코드와의 불일치. 공식 `convbencheval.py:192-229` |")
    A("| 승자 해석 3차 | 판정문의 `Rating:` → A/B, 아니면 판정 불가 | 공식 규칙 추가(결과 영향 없음) | 공식 `convbencheval.py:67-77, 397-404` |")
    A("| 점수 | 단계별 모델 승수 / 577, 평균 = (S1+S2+S3)/3, 판정 불가는 0승 | 같음 | 공식 `convbencheval.py:376-435` |")
    A("\n## 3. 이 설정으로 얻은 ReComp 점수\n")
    A("| 모델 | | S1 | S2 | S3 | 평균 | 판정 불가 |")
    A("|---|---|---:|---:|---:|---:|---|")
    for model, label, scores in (("next", "LLaVA-NeXT-7b", b3), ("llava15", "LLaVA-1.5-7b", l15)):
        t = TARGET[model]
        A(f"| {label} | MetaCompress 기본 모델 | {t['S1']:.2f} | {t['S2']:.2f} | {t['S3']:.2f} | {t['Avg']:.2f} | |")
        A(f"| {label} | ReComp (동결 설정) | {row_cells(scores)} |")
        A(f"| {label} | 목표 범위(단계 ±{STAGE_TOL:.1f}, 평균 ±{AVG_TOL:.1f}) | {verdict_text(scores, model)} | | | | |")
    A(f"| LLaVA-NeXT-7b | 9월 ReComp (9월 설정) | {row_cells(sept)} |")
    A("\n## 4. 실행 방법\n")
    A("```")
    A("# ReComp 생성 (SSD store 없음)")
    A("python scripts/213_convbench_recomp_generate.py --out <dir> --model next|llava15 --cap-policy official_1024 --tokenization official")
    A("# 판정")
    A(f"python scripts/214_convbench_alignment_judge.py --answers <dir> --out <dir> --cache <cache dir> --cap {JUDGE_CAP}")
    A("```")
    A("\n## 5. 본실험에 옮길 때 주의\n")
    A("- 네 방법 실행기(`scripts/41_eval_convbench_answers.py` 계열)는 9월 정책을 코드에 고정하고 있다"
      "(`_effective_cap`, `render_prompt`). 본실험 실행기에 ① 상한 항상 1024, ② 공식 토큰열 렌더링(LLaVA-NeXT: `<image> \\n`, `</s> USER`)과 "
      "요청별 일치 검사를 반영해야 한다. 이 작업에서는 방법 코드를 바꾸지 않았다.")
    A("- ② 는 이미지 뒤 접미부 토큰을 바꾼다. 이미지 prefix KV(시스템 문장 ~ 이미지 토큰)는 그대로다.")
    A("- 판정은 `scripts/214`를 방법별 답에 그대로 쓰면 된다(현재는 `method == \"ReComp\"` 행만 읽으므로 방법 이름 선택 인자를 추가해야 한다).")
    A("- 채택하지 않은 설정과 진단 결과는 근거 실행의 `report.md` 3–6절에 있다.")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------- driver
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--res", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    res = args.res.resolve()
    steps = build_steps(res)
    status_path = res / "status.json"
    previous = read_json(status_path, {}) or {}
    prior_steps = {s["id"]: s for s in previous.get("steps", [])}
    state = [{"id": s["id"], "title": s["title"],
              **{k: prior_steps.get(s["id"], {}).get(k) for k in ("started_utc", "finished_utc", "seconds", "note")},
              "state": (prior_steps.get(s["id"], {}).get("state")
                        if prior_steps.get(s["id"], {}).get("state") in ("done", "not_needed") else "pending")}
             for s in steps]
    rel = res.relative_to(ROOT)
    status = {
        "run": str(rel), "stage": "starting", "stage_title": None, "progress": None, "eta_utc": None,
        "last_error": previous.get("last_error") if args.report_only else None,
        "driver_pid": os.getpid(), "driver_started_utc": utc(),
        "first_started_utc": previous.get("first_started_utc") or utc(),
        "driver_invocations": int(previous.get("driver_invocations") or 0) + (0 if args.report_only else 1),
        "updated_utc": utc(), "steps": state,
        "commands": {"progress": f"cat {rel}/status.json", "log": f"tail -n 30 {rel}/driver.log",
                     "stop": f"kill -TERM -- -$(cat {rel}/driver.pid)",
                     "continue": f"scripts/run_convbench_alignment.sh {rel}"},
    }

    def save() -> None:
        status["updated_utc"] = utc()
        atomic_json(status_path, status)

    def write_report() -> None:
        try:
            (res / "report.md").write_text(build_report(res, status), encoding="utf-8")
        except Exception as exc:  # the report must never stop the experiment
            print(f"{utc()} report failed: {exc!r}", flush=True)
            status["last_error"] = {"stage": "report", "error": repr(exc), "at_utc": utc()}

    if args.report_only:
        status["steps"] = previous.get("steps", state)
        status["stage"] = previous.get("stage")
        (res / "report.md").write_text(build_report(res, status), encoding="utf-8")
        print("report written")
        return

    def on_signal(signum, _frame):
        STOP["requested"] = True
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    (res / "logs").mkdir(exist_ok=True)

    def remaining_estimate(index: int, current_left: float) -> float:
        total = current_left
        for later, later_state in zip(steps[index + 1:], state[index + 1:]):
            if later_state["state"] in ("done", "not_needed", "skipped", "failed"):
                continue
            try:
                needed = bool(later["when"]())
            except Exception:
                needed = False
            # conditional diagnostics are counted only once their condition is known to hold
            if later["core"] or needed or later["id"].startswith("D1"):
                total += later["est"]
        return total

    save()
    for index, (step, st) in enumerate(zip(steps, state)):
        if STOP["requested"]:
            break
        done_marker = (step["out"] / "DONE") if "out" in step else None
        if "internal" not in step and done_marker.exists():
            st["state"] = "done"
            st["note"] = st.get("note") or "이미 완료"
            continue
        if not step["when"]():
            st["state"] = "not_needed"
            st["note"] = "조건에 해당하지 않음"
            save()
            continue
        st.update(state="running", started_utc=utc(), note=None)
        status.update(stage=step["id"], stage_title=step["title"], progress=None)
        save()
        t0 = time.time()
        print(f"=== {utc()} START {step['id']}: {step['title']}", flush=True)
        if "internal" in step:
            try:
                if step["internal"] == "freeze":
                    (ROOT / "docs/convbench_baseline_alignment.md").write_text(build_freeze_doc(res), encoding="utf-8")
                st["state"] = "done"
            except Exception as exc:
                st.update(state="failed", note=repr(exc))
                status["last_error"] = {"stage": step["id"], "error": repr(exc), "at_utc": utc()}
            st.update(finished_utc=utc(), seconds=time.time() - t0)
            write_report()
            save()
            print(f"=== {utc()} END {step['id']} {st['state']}", flush=True)
            continue

        log_path = res / "logs" / f"{step['id']}.log"
        rc, attempts, oom = None, 0, False
        while attempts < 3 and not STOP["requested"]:
            attempts += 1
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(f"\n=== {utc()} attempt {attempts}: {' '.join(step['cmd'])}\n")
                log.flush()
                proc = subprocess.Popen(step["cmd"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                while proc.poll() is None:
                    time.sleep(20)
                    if STOP["requested"]:
                        proc.terminate()
                        try:
                            proc.wait(timeout=120)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        break
                    p = read_json(step["out"] / "progress.json")
                    left = step["est"] - (time.time() - t0)
                    if p and p.get("total"):
                        rate = p.get("seconds_per_conversation_this_process") or p.get("seconds_per_fresh_conversation")
                        status["progress"] = {"done": p["done"], "total": p["total"], "unit": "conversations",
                                              "note": p.get("note")}
                        if rate and p["done"] < p["total"]:
                            left = rate * (p["total"] - p["done"])
                    status["eta_utc"] = utc(time.time() + remaining_estimate(index, max(left, 60)))
                    save()
                rc = proc.returncode
            if rc == 0 and done_marker.exists():
                break
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            oom = "out of memory" in tail.lower()
            if STOP["requested"] or (oom and step.get("oom_skip")) or "replay check failed" in tail:
                break
            print(f"{utc()} {step['id']} attempt {attempts} rc={rc}; retrying in 60 s", flush=True)
            time.sleep(60)
        st.update(finished_utc=utc(), seconds=(st.get("seconds") or 0) + time.time() - t0)
        if STOP["requested"]:
            st["state"] = "pending"
            st["note"] = "신호로 중단됨; 같은 명령으로 이어서 실행"
            break
        if rc == 0 and done_marker.exists():
            st["state"] = "done"
            p = read_json(step["out"] / "progress.json") or {}
            status["progress"] = {"done": p.get("done"), "total": p.get("total"), "unit": "conversations"}
        elif not step["core"]:
            st["state"] = "skipped"
            st["note"] = ("GPU 메모리 부족으로 건너뜀" if oom else f"실패 rc={rc} (logs/{step['id']}.log); 진단이므로 건너뜀")
            status["last_error"] = {"stage": step["id"], "error": st["note"], "at_utc": utc()}
        else:
            st["state"] = "failed"
            st["note"] = f"rc={rc} after {attempts} attempts (logs/{step['id']}.log)"
            status["last_error"] = {"stage": step["id"], "error": st["note"], "at_utc": utc()}
            status["stage"] = "stopped_on_error"
            write_report()
            save()
            print(f"=== {utc()} STOP {step['id']}: {st['note']}", flush=True)
            sys.exit(1)
        write_report()
        save()
        print(f"=== {utc()} END {step['id']} {st['state']} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    if STOP["requested"]:
        status["stage"] = "stopped_by_signal"
        status["last_error"] = {"stage": "signal", "error": "driver received TERM/INT; continue with the same command",
                                "at_utc": utc()}
        write_report()
        save()
        print(f"=== {utc()} stopped by signal", flush=True)
        sys.exit(130)
    status.update(stage="done", stage_title="모든 단계 종료", eta_utc=None, finished_utc=utc(), driver_pid=None)
    write_report()
    save()
    print(f"=== {utc()} DRIVER DONE", flush=True)


if __name__ == "__main__":
    main()

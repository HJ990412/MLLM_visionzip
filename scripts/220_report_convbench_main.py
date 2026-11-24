#!/usr/bin/env python3
"""ConvBench main experiment: report.md (and the pilot table of gate 7).

Reads committed conversations, gate files and judge outputs only; never touches the GPU or the stores.
Not part of the measurement freeze (file name matches NNN_report_*.py).

  --res RES                      results/<run>/report.md (+ report_tables.json); whatever exists is reported
  --res RES --pilot MODEL        results/<run>/gates/<MODEL>/gate7_pilot_table.{md,json}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import convbench_main_common as C  # noqa: E402

MB = 1e6
QUALITY_PAIRS = [(C.OURS, C.IMP), (C.OURS, C.FULL), (C.OURS, C.RECOMP), (C.IMP, C.FULL), (C.FULL, C.RECOMP)]
TTFT_PAIRS = [(C.OURS, C.FULL), (C.OURS, C.RECOMP), (C.OURS, C.IMP), (C.IMP, C.FULL), (C.FULL, C.RECOMP)]
TTFT_RATIOS = [(C.OURS, C.FULL), (C.OURS, C.RECOMP), (C.OURS, C.IMP), (C.IMP, C.FULL)]
STAGES = (("_first_turn", "S1"), ("_second_turn", "S2"), ("_third_turn", "S3"))
N = C.DISPLAY


# ------------------------------------------------------------------ helpers
def fx(x, d=2, unit=""):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{d}f}{unit}"


def table(header, rows, align=None):
    align = align or ["---"] + ["---:"] * (len(header) - 1)
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def bootstrap_mean(values, resamples=C.RESAMPLES, seed=C.SEED):
    """Conversation-level bootstrap of the mean of per-conversation values (paired differences)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size < 2:
        return {"mean": float(v.mean()) if v.size else None, "ci_low": None, "ci_high": None, "n": int(v.size)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(resamples, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(v.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": int(v.size)}


def bootstrap_ratio(a, b, resamples=C.RESAMPLES, seed=C.SEED):
    """Conversation-level bootstrap of sum(a) / sum(b) (per-conversation sums over the follow-up turns)."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.size < 2:
        return {"ratio": float(a.sum() / b.sum()) if a.size else None, "ci_low": None, "ci_high": None, "n": int(a.size)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(resamples, a.size))
    r = a[idx].sum(axis=1) / b[idx].sum(axis=1)
    lo, hi = np.percentile(r, [2.5, 97.5])
    return {"ratio": float(a.sum() / b.sum()), "ci_low": float(lo), "ci_high": float(hi), "n": int(a.size)}


def dist(values):
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if not v.size:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    return {"n": int(v.size), "mean": float(v.mean()), "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
            "min": float(v.min()), "max": float(v.max())}


def ok_rows(docs, role, turns):
    return [r for d in docs.values() for r in d["rows"] if r["role"] == role and r["turn_id"] in turns and r["status"] == "ok"]


# ------------------------------------------------------------ system tables
def system_tables(docs, roles=C.MAIN_ROLES):
    """Everything that does not need the judge, from a set of committed conversations."""
    out = {"conversations": len(docs), "roles": {}}
    by = {cid: C.rows_by_key(d) for cid, d in docs.items()}
    for role in roles:
        ttft = {}
        for label, turns in (("T2", (2,)), ("T3", (3,)), ("T2-T3", (2, 3))):
            ttft[label] = dist([r["ttft_ms"] for r in ok_rows(docs, role, turns)])
        fu = ok_rows(docs, role, (2, 3))
        allr = ok_rows(docs, role, (1, 2, 3))
        failed = [r for d in docs.values() for r in d["rows"] if r["role"] == role and r["status"] != "ok"]
        entry = {"method": N[role], "ttft": ttft, "t1_ttft": dist([r["ttft_ms"] for r in ok_rows(docs, role, (1,))]),
                 "ssd_mb_per_followup": float(np.mean([r["ssd_read_bytes"] for r in fu]) / MB) if fu else None,
                 "read_calls_per_followup": float(np.mean([r["read_calls"] for r in fu])) if fu else None,
                 "followup_requests": len(fu), "requests_ok": len(allr), "requests_failed": len(failed),
                 "input_tokens": {f"T{t}": dist([r["input_tokens"] for r in ok_rows(docs, role, (t,))]) for t in (1, 2, 3)},
                 "requests_above_4096": {f"T{t}": sum(r["input_above_4096"] for r in ok_rows(docs, role, (t,))) for t in (1, 2, 3)},
                 "generated_tokens": {f"T{t}": dist([r["generated_tokens"] for r in ok_rows(docs, role, (t,))]) for t in (1, 2, 3)},
                 "generated_tokens_mean": float(np.mean([r["generated_tokens"] for r in allr])) if allr else None,
                 "cap_reached": {f"T{t}": sum(r["cap_reached"] for r in ok_rows(docs, role, (t,))) for t in (1, 2, 3)},
                 "e2e_ms_followup_mean": float(np.mean([r["request_e2e_ms"] for r in fu])) if fu else None}
        if role != C.RECOMP and fu:
            blocked = np.array([r["read_blocked_ms"] if r["read_blocked_ms"] is not None else np.nan for r in fu], dtype=float)
            hook = np.array([r["hook_ms"] if r["hook_ms"] is not None else np.nan for r in fu], dtype=float)
            prefill = np.array([r["prefill_ms"] for r in fu], dtype=float)
            scatter = [r["scatter_ms"] for r in fu]
            d = {"ttft_mean_ms": float(np.mean([r["ttft_ms"] for r in fu])), "prefill_mean_ms": float(prefill.mean()),
                 "read_wait_mean_ms": float(np.nanmean(blocked)), "read_wait_p95_ms": float(np.nanpercentile(blocked, 95)),
                 "hook_mean_ms": float(np.nanmean(hook)), "text_compute_mean_ms": float(np.nanmean(prefill - hook)),
                 "outside_prefill_mean_ms": float(np.mean([r["ttft_ms"] for r in fu]) - prefill.mean())}
            if all(s is not None for s in scatter):
                d["gpu_load_mean_ms"] = float(np.mean(scatter))
                d["gpu_load_source"] = "적재 호출 구간의 합(읽은 행을 GPU cache의 원래 위치에 쓰고 mask를 만드는 시간)"
            else:
                d["gpu_load_mean_ms"] = float(np.nanmean(hook - blocked))
                d["gpu_load_source"] = ("layer hook 시간 − 읽기 대기 (모든 행의 host FP16 → GPU BF16 복사)" if role == C.FULL else
                                        "layer hook 시간 − 읽기 대기 (GPU 복사에 probe 중요도 계산·선택이 함께 들어 있어 분리되지 않음)")
            workers = [r.get("worker_read_ms") for r in fu]
            if all(w is not None for w in workers):
                d["worker_read_mean_ms"] = float(np.mean(workers))
            entry["decomposition"] = d
        elif fu:
            entry["decomposition"] = {"ttft_mean_ms": float(np.mean([r["ttft_ms"] for r in fu])),
                                      "prefill_mean_ms": float(np.mean([r["prefill_ms"] for r in fu]))}
        out["roles"][role] = entry

    def paired(a, b):
        da, sa, sb = [], [], []
        for cid in docs:
            ra = [by[cid].get((a, t)) for t in (2, 3)]; rb = [by[cid].get((b, t)) for t in (2, 3)]
            if any(r is None or r["status"] != "ok" for r in ra + rb):
                continue
            xa = sum(r["ttft_ms"] for r in ra); xb = sum(r["ttft_ms"] for r in rb)
            da.append((xa - xb) / 2); sa.append(xa); sb.append(xb)
        return da, sa, sb

    out["ttft_pairs"] = []
    for a, b in TTFT_PAIRS:
        if a in roles and b in roles:
            da, sa, sb = paired(a, b)
            out["ttft_pairs"].append({"pair": f"{N[a]} − {N[b]}", **bootstrap_mean(da)})
    out["ttft_ratios"] = []
    for a, b in TTFT_RATIOS:
        if a in roles and b in roles:
            da, sa, sb = paired(a, b)
            out["ttft_ratios"].append({"ratio_name": f"{N[a]} / {N[b]}", **bootstrap_ratio(sa, sb)})

    # IMPRESS internals
    imp = [r["impress"] for r in ok_rows(docs, C.IMP, (2, 3)) if "impress" in r] if C.IMP in roles else []
    if imp:
        fb = np.array([x["fb"] for x in imp], dtype=float); sim = np.array([x["sim"] for x in imp], dtype=float)
        kc = np.array([x["kc"] for x in imp], dtype=float); vc = np.array([x["vc"] for x in imp], dtype=float)
        nc = np.array([x["nc"] for x in imp], dtype=float)[:, None]
        raster = np.array([x["raster"] for x in imp], dtype=float)
        share = np.array([[s if s is not None else np.nan for s in x["share"]] for x in imp], dtype=float)
        cons = fb == 0
        kv = (kc + vc) / (2 * nc)
        out["impress"] = {"followup_requests": int(fb.shape[0]), "fallback_rate": float(fb.mean()), "mean_jaccard": float(sim.mean()),
                          "threshold": 0.3, "kv_chunk_fraction": float(kv.mean()),
                          "kv_chunk_fraction_non_fallback_layers": float(kv[cons].mean()) if cons.any() else None,
                          "raster_chunk_fraction_computed": float((raster / nc).mean()),
                          "raster_chunk_fraction_computed_non_fallback_layers": float((raster / nc)[cons].mean()) if cons.any() else None,
                          "prefix_share": float(np.nanmean(share)),
                          "requests_with_any_fallback": float((fb.sum(axis=1) > 0).mean()),
                          "requests_all_layers_fallback": float((fb.sum(axis=1) == fb.shape[1]).mean())}
        if all(x.get("only") is not None for x in imp):
            only = np.array([x["only"] for x in imp], dtype=float)
            out["impress"]["importance_only_chunk_fraction_computed"] = float((only / nc).mean())

    # turn-1 store cost
    store_of = {}
    for d in docs.values():
        for kind, s in d["stores"].items():
            store_of.setdefault(s.get("source_role"), []).append(s)
    out["stores"] = {}
    for role in roles:
        s = store_of.get(role)
        if not s:
            continue
        after = [(x.get("official_turn1") or {}).get("total_after_request_ms") or 0.0 for x in s]
        capture = [(x.get("official_turn1") or {}).get("capture_ms") for x in s]
        writer = np.array([x["native_writer_wall_ms"] for x in s], dtype=float)
        out["stores"][role] = {"method": N[role], "conversations": len(s),
                               "store_mb_mean": float(np.mean([x["file_logical_bytes"] for x in s]) / MB),
                               "store_mb_max": float(max(x["file_logical_bytes"] for x in s) / MB),
                               "store_seconds_mean": float(writer.mean() / 1000), "store_seconds_p95": float(np.percentile(writer, 95) / 1000),
                               "impress_reorder_seconds_mean_included": float(np.mean(after) / 1000) if any(after) else None,
                               "impress_capture_ms_inside_turn1": float(np.mean([c for c in capture if c is not None])) if any(c is not None for c in capture) else None,
                               "integrity_hash_seconds_mean_harness": float(np.mean([x["completion_full_hash_ms"] for x in s]) / 1000),
                               "written_mb_mean": float(np.mean([x["os_write_bytes"] for x in s]) / MB),
                               "written_bytes_total": int(sum(x["os_write_bytes"] for x in s)),
                               "activation_seconds_mean": float(np.mean([x["activation_ms"] for x in s if x.get("activation_ms") is not None]) / 1000)
                               if any(x.get("activation_ms") is not None for x in s) else None}
    out["failures"] = [{k: f.get(k) for k in ("conversation_id", "method", "turn_id", "stage", "attempt", "error_type", "error", "oom")}
                       for d in docs.values() for f in d["failures"]]
    out["failed_rows"] = [{"conversation_id": cid, "method": r["method"], "turn_id": r["turn_id"], "cause": r.get("failure"), "executed": r.get("executed")}
                          for cid, d in docs.items() for r in d["rows"] if r["status"] != "ok"]
    out["t1_identical_conversations"] = sum(bool(d["t1_tokens_identical_main_methods"]) for d in docs.values())
    out["conversation_seconds_mean"] = float(np.mean([d["seconds"] for d in docs.values()])) if docs else None
    out["conversation_seconds_total"] = float(np.sum([d["seconds"] for d in docs.values()])) if docs else None
    return out


def recomp_vs_alignment(model, docs):
    same = total = 0; diff = []
    for cid, d in docs.items():
        by = C.rows_by_key(d); ref = C.alignment_rows(model, cid)
        for t in (1, 2, 3):
            r = by.get((C.RECOMP, t)); a = ref[t - 1]
            total += 1
            ok = r is not None and r["status"] == "ok" and r["prediction"] == a["prediction"] and r["generated_tokens"] == a["generated_tokens"]
            same += ok
            if not ok and len(diff) < 20:
                diff.append({"conversation_id": cid, "turn": t})
    return {"requests": total, "identical": same, "examples_of_differences": diff}


def agreement_with_recomp(docs):
    """How often a method's generated tokens equal ReComp's on the same turn (context for the noise reference)."""
    out = {}
    for role in (C.FULL, C.IMP, C.OURS):
        row = {}
        for t in (1, 2, 3):
            n = same = 0
            for d in docs.values():
                by = C.rows_by_key(d); a = by.get((role, t)); b = by.get((C.RECOMP, t))
                if a and b and a["status"] == b["status"] == "ok":
                    n += 1; same += a["generated_token_ids"] == b["generated_token_ids"]
            row[f"T{t}"] = [same, n]
        out[role] = row
    return out


# ------------------------------------------------------------------ quality
def quality(res, model, n_conv):
    out_dir = Path(res) / "judge" / model
    if not (out_dir / "DONE").exists():
        return None
    summary = C.read_json(out_dir / "summary.json")
    records = C.read_jsonl(out_dir / "judge_raw.jsonl")
    wins = {}
    for r in records:
        wins.setdefault(r["role"], {}).setdefault(r["conversation_id"], {})[r["judge_turn"]] = 1.0 if r["model_wins"] else 0.0
    conv_ids = sorted({r["conversation_id"] for r in records})
    per = {role: np.array([[wins[role][c].get(t, 0.0) for t, _ in STAGES] for c in conv_ids]) for role in wins}
    pairs = []
    for a, b in QUALITY_PAIRS:
        if a in per and b in per:
            d = (per[a].mean(axis=1) - per[b].mean(axis=1)) * 100
            entry = {"pair": f"{N[a]} − {N[b]}", **bootstrap_mean(d)}
            entry["stage_differences"] = {label: float((per[a][:, i].mean() - per[b][:, i].mean()) * 100) for i, (_, label) in enumerate(STAGES)}
            entry["conversations_with_different_outcome"] = int((np.abs(per[a] - per[b]).sum(axis=1) > 0).sum())
            pairs.append(entry)
    align = (C.read_json(C.ALIGN_JUDGE[model] / "summary.json") or {}).get("views", {}).get("capN")
    return {"summary": summary, "pairs": pairs, "alignment_recomp": align, "conversations": len(conv_ids), "denominator": summary["denominator"]}


# ---------------------------------------------------------------- markdown
def ci(e, d=2, unit=""):
    if e.get("ci_low") is None:
        return fx(e.get("mean", e.get("ratio")), d, unit)
    return f"{fx(e.get('mean', e.get('ratio')), d, unit)} [{fx(e['ci_low'], d)}, {fx(e['ci_high'], d)}]"


def system_md(t, level="###", pilot=False):
    L = []
    roles = [r for r in C.MAIN_ROLES if r in t["roles"]]
    R = t["roles"]
    L += [f"{level} TTFT (재사용 턴 T2–T3, ms)", "",
          table(["방법", "T2 평균", "T2 p50", "T2 p95", "T3 평균", "T3 p50", "T3 p95", "T2–T3 평균", "T2–T3 p50", "T2–T3 p95", "요청 수"],
                [[N[r]] + [fx(R[r]["ttft"][k][s], 1) for k in ("T2", "T3", "T2-T3") for s in ("mean", "p50", "p95")] + [R[r]["followup_requests"]] for r in roles]), ""]
    L += ["짝지은 차이(같은 대화·같은 턴, 대화 단위 paired bootstrap 10,000회, seed 1234, 95% CI):", "",
          table(["비교", "TTFT 차이 (ms) [95% CI]", "대화 수"], [[e["pair"], ci(e, 1), e["n"]] for e in t["ttft_pairs"]]), "",
          table(["비율", "TTFT 평균의 비 [95% CI]", "대화 수"], [[e["ratio_name"], ci(e, 3), e["n"]] for e in t["ttft_ratios"]]), ""]
    L += [f"{level} SSD 읽기량과 읽기 호출 수 (재사용 턴 요청당)", "",
          table(["방법", "SSD MB/followup", "읽기 호출 수/followup", "T1 TTFT 평균 (ms)"],
                [[N[r], fx(R[r]["ssd_mb_per_followup"], 1), fx(R[r]["read_calls_per_followup"], 1), fx(R[r]["t1_ttft"]["mean"], 1)] for r in roles]), ""]
    if "impress" in t:
        i = t["impress"]
        L += [f"{level} IMPRESS 통계 (재사용 턴 {i['followup_requests']:,}요청 × 32 layer)", "",
              table(["항목", "값"], [
                  ["fallback 비율 (layer 단위; 평균 Jaccard < 0.3이면 그 layer 전체를 읽음)", fx(100 * i["fallback_rate"], 1, " %")],
                  ["fallback이 하나라도 있는 요청 / 모든 layer가 fallback인 요청", f"{fx(100 * i['requests_with_any_fallback'], 1, ' %')} / {fx(100 * i['requests_all_layers_fallback'], 1, ' %')}"],
                  ["평균 Jaccard (probe head 쌍)", fx(i["mean_jaccard"], 3)],
                  ["K·V chunk 읽기 비율 — 전체", fx(100 * i["kv_chunk_fraction"], 1, " %")],
                  ["K·V chunk 읽기 비율 — fallback이 아닌 layer만", fx(100 * i["kv_chunk_fraction_non_fallback_layers"], 1, " %") if i["kv_chunk_fraction_non_fallback_layers"] is not None else "—"],
                  ["K·V chunk 읽기 비율 — 재배치하지 않았을 경우의 계산값 (전체 / fallback이 아닌 layer만)",
                   f"{fx(100 * i['raster_chunk_fraction_computed'], 1, ' %')} / " + (fx(100 * i["raster_chunk_fraction_computed_non_fallback_layers"], 1, " %") if i["raster_chunk_fraction_computed_non_fallback_layers"] is not None else "—")],
                  ["중요도 중 prefix 행이 차지하는 비율", fx(100 * i["prefix_share"], 1, " %")]], ["---", "---:"]), ""]
    L += [f"{level} 재사용 턴 시간 분해 (T2–T3 평균, ms)", "",
          table(["방법", "TTFT", "읽기 대기", "GPU 적재", "이미지 뒤 텍스트의 계산", "prefill 구간 전체", "prefill 밖(프롬프트·토큰화 등)"],
                [[N[r], fx(R[r]["decomposition"]["ttft_mean_ms"], 1), fx(R[r]["decomposition"].get("read_wait_mean_ms"), 1),
                  fx(R[r]["decomposition"].get("gpu_load_mean_ms"), 1), fx(R[r]["decomposition"].get("text_compute_mean_ms"), 1),
                  fx(R[r]["decomposition"]["prefill_mean_ms"], 1), fx(R[r]["decomposition"].get("outside_prefill_mean_ms"), 1)]
                 for r in roles if "decomposition" in R[r]]), "",
          "- 읽기 대기: 요청 thread가 SSD 읽기를 기다린 시간. 이미지 뒤 텍스트의 계산 = prefill 구간 − layer hook 시간(모델이 접미부 토큰을 계산한 시간). ReComp의 prefill 구간에는 이미지 인코딩과 전체 입력 계산이 들어 있다.",
          *[f"- {N[r]}의 GPU 적재: {R[r]['decomposition']['gpu_load_source']}" for r in roles if "decomposition" in R[r] and "gpu_load_source" in R[r]["decomposition"]],
          "- 읽기 대기 + GPU 적재 + 텍스트 계산이 prefill 구간이고, TTFT는 여기에 prefill 밖 시간(프롬프트 조립, 토큰화, cache 할당)을 더한 값이다. 항목마다 계측 구간이 달라 합이 정확히 맞지는 않을 수 있다.", ""]
    if t["stores"]:
        L += [f"{level} T1 저장 비용 (대화당)", "",
              table(["방법", "저장 시간 평균 (s)", "저장 시간 p95 (s)", "저장 크기 평균 (MB)", "저장 크기 최대 (MB)", "쓴 양 평균 (MB)", "대화 수"],
                    [[N[r], fx(s["store_seconds_mean"], 2), fx(s["store_seconds_p95"], 2), fx(s["store_mb_mean"], 1), fx(s["store_mb_max"], 1),
                      fx(s["written_mb_mean"], 1), s["conversations"]] for r, s in t["stores"].items()]), "",
              "- 저장 시간은 store를 쓰는 호출의 벽시계 시간이다(T1 요청 뒤, TTFT 밖). IMPRESS는 전체 KV를 쓴 뒤 중요도 순으로 다시 쓰는 시간을 포함한다"
              + (f"(재배치 평균 {fx(t['stores'][C.IMP]['impress_reorder_seconds_mean_included'], 2)} s; T1 요청 안에서 attention 통계를 모으는 시간 {fx(t['stores'][C.IMP]['impress_capture_ms_inside_turn1'], 1)} ms는 T1 TTFT에 들어 있다)" if C.IMP in t["stores"] and t["stores"][C.IMP]["impress_reorder_seconds_mean_included"] else "") + ".",
              "- 실행기가 저장 뒤에 하는 무결성 해시(전체 파일 재읽기)는 저장 시간에 넣지 않았다. ReComp는 저장하지 않는다.", ""]
    L += [f"{level} 입력 길이와 생성 길이", "",
          table(["방법", "T1 입력 평균", "T2 입력 평균", "T3 입력 평균", "T3 입력 p95", "T3 입력 최대", "4096 초과 요청 (T1/T2/T3)", "평균 생성 길이", "상한(1024) 도달 (T1/T2/T3)"],
                [[N[r], fx(R[r]["input_tokens"]["T1"]["mean"], 0), fx(R[r]["input_tokens"]["T2"]["mean"], 0), fx(R[r]["input_tokens"]["T3"]["mean"], 0),
                  fx(R[r]["input_tokens"]["T3"]["p95"], 0), fx(R[r]["input_tokens"]["T3"]["max"], 0),
                  "/".join(str(R[r]["requests_above_4096"][f"T{k}"]) for k in (1, 2, 3)) + f" (합 {sum(R[r]['requests_above_4096'].values())})",
                  fx(R[r]["generated_tokens_mean"], 1), "/".join(str(R[r]["cap_reached"][f"T{k}"]) for k in (1, 2, 3)) + f" (합 {sum(R[r]['cap_reached'].values())})"]
                 for r in roles]), "",
          table(["방법", "T1 입력 min/p50/p95/max", "T2 입력 min/p50/p95/max", "T3 입력 min/p50/p95/max", "생성 길이 T1/T2/T3 평균"],
                [[N[r]] + ["/".join(fx(R[r]["input_tokens"][f"T{k}"][s], 0) for s in ("min", "p50", "p95", "max")) for k in (1, 2, 3)]
                 + ["/".join(fx(R[r]["generated_tokens"][f"T{k}"]["mean"], 1) for k in (1, 2, 3))] for r in roles]), ""]
    return L


def failures_md(t, n_conv_expected):
    L = []
    if not t["failed_rows"]:
        return [f"실패한 요청: 0건 (요청 {sum(v['requests_ok'] for v in t['roles'].values()):,}건 모두 완료).", ""]
    L += [f"실패한 요청: {len(t['failed_rows'])}건. 분모는 그대로이며 실패한 요청은 0승으로 계산했다.", "",
          table(["대화", "방법", "턴", "원인", "실행됨"], [[f["conversation_id"], f["method"], f["turn_id"], str(f["cause"])[:160], "예" if f["executed"] else "아니오(앞 턴 실패)"]
                                                 for f in t["failed_rows"][:60]], ["---"] * 5), ""]
    if len(t["failed_rows"]) > 60:
        L += [f"(처음 60건만 표시; 전체는 report_tables.json의 failed_rows)", ""]
    return L


def quality_md(q, model):
    L = []
    s = q["summary"]["roles"]
    mc = C.METACOMPRESS[model]
    rows = []
    for role in C.MAIN_ROLES:
        v = s[role]
        fail = sum(v["generation_failed"].values())
        rows.append([N[role], fx(v["S1"]), fx(v["S2"]), fx(v["S3"]), fx(v["Avg"]),
                     "/".join(str(v["unresolved"][k]) for k in ("S1", "S2", "S3")) + f" (합 {sum(v['unresolved'].values())})",
                     "/".join(str(v["generation_failed"][k]) for k in ("S1", "S2", "S3")) if fail else "0"])
    rows.append(["참고: MetaCompress 기본 모델 (논문 값)", fx(mc["S1"]), fx(mc["S2"]), fx(mc["S3"]), fx(mc["Avg"]), "—", "—"])
    a = q["alignment_recomp"]
    if a:
        rows.append(["참고: 점검 실행의 ReComp", fx(a["S1"]), fx(a["S2"]), fx(a["S3"]), fx(a["Avg"]),
                     "/".join(str(a["unresolved"][k]) for k in ("S1", "S2", "S3")) + f" (합 {a['unresolved_total']})", "—"])
    L += ["### 품질 (ConvBench, 판정기 Llama-3.1-8B-Instruct; 승률 %, 분모 " + str(q["denominator"]) + ")", "",
          table(["방법", "S1", "S2", "S3", "평균", "판정 불가 (S1/S2/S3)", "생성 실패 (S1/S2/S3)"], rows), "",
          "- 판정 불가와 생성 실패는 분모에 남기고 0승으로 계산했다.", ""]
    L += ["### 품질의 짝지은 비교 (평균 점수 기준, 대화 단위 paired bootstrap 10,000회, seed 1234)", "",
          table(["비교", "평균 점수 차이 (pp) [95% CI]", "S1 차이", "S2 차이", "S3 차이", "판정 결과가 다른 대화 수"],
                [[e["pair"], ci(e, 2), fx(e["stage_differences"]["S1"]), fx(e["stage_differences"]["S2"]), fx(e["stage_differences"]["S3"]),
                  e["conversations_with_different_outcome"]] for e in q["pairs"]]), ""]
    noise = next((e for e in q["pairs"] if e["pair"] == f"{N[C.FULL]} − {N[C.RECOMP]}"), None)
    if noise:
        L += [f"- 잡음 수준 참고값: FullLoad − ReComp = {ci(noise, 2)} pp. FullLoad는 이미지 KV를 모두 읽으므로 ReComp와의 차이는 "
              "FP16 저장과 계산 순서에서 오는 생성 토큰의 갈림, 그리고 판정기가 그 차이에 반응하는 정도를 보여 준다. 다른 비교의 차이는 이 값의 크기와 CI 폭에 비추어 읽어야 한다.", ""]
    return L


def judge_diag_md(q):
    s = q["summary"]["roles"]
    rows = []
    for role in C.MAIN_ROLES:
        st = s[role]["parse_stages"]
        tot = {k: sum(st[x][k] for x in ("S1", "S2", "S3")) for k in ("first_pass", "second_pass", "rating_fallback", "unresolved", "generation_failed",
                                                                    "hit_cap_N", "from_alignment_cache", "judged_with_missing_later_answer")}
        n = 3 * q["denominator"]
        rows.append([N[role], tot["first_pass"], tot["second_pass"], tot["rating_fallback"], tot["unresolved"], fx(100 * tot["unresolved"] / n, 2, " %"),
                     tot["hit_cap_N"], tot["from_alignment_cache"], tot["judged_with_missing_later_answer"]])
    return ["### 판정 진단 (단계 판정 " + f"{3 * q['denominator']:,}" + "건 / 방법)", "",
            table(["방법", "1차에서 해석", "2차(공식 추출)에서 복구", "3차(Rating)에서 복구", "판정 불가", "판정 불가 비율", "출력 상한(2048) 도달",
                   "점검 실행 cache에서 읽음", "뒤 턴 답 없이 판정"], rows), "",
            f"- 이번 실행에서 새로 판정한 판정문 {q['summary'].get('judged_now', 0):,}건, 2차 추출 호출 {q['summary'].get('extractions_now', 0):,}건, "
            f"점검 실행 cache에서 읽은 것 {q['summary'].get('alignment_cache_hits', 0):,}건(같은 판정 프롬프트).", ""]


def gates_md(res, model):
    g = Path(res) / "gates"
    L = []
    tests = C.read_json(g / "tests.json"); check = C.read_json(g / "judge_check.json")
    gate4 = C.read_json(g / model / "gate_gate.json"); gate40 = C.read_json(g / model / "gate_pilot.json")
    admit = C.read_json(g / model / "admission.json"); over = C.read_json(g / model / "gate7_overflow.json")

    def st(doc, key=None):
        if doc is None:
            return "미실행"
        if key is None:
            return doc.get("status", "?")
        return "PASS" if doc["gates"].get(key, {}).get("ok") else ("FAIL" if key in doc.get("gates", {}) else "—")
    rows = []
    suite = (tests or {}).get("suite") or {}
    rows.append(["1. 기존 테스트에 새 실패 0", st(tests), "—",
                 f"{suite.get('ran', '—')}개 실행, 새 실패 {len(suite.get('new_failures', []))}개, 이전부터 실패 {len(suite.get('failures_that_existed_before_this_work', []))}개" if suite else "—"])
    for key, title in (("gate2_recomp_equals_alignment", "2. ReComp = 점검 실행의 최종 ReComp"),):
        d4 = (gate4 or {}).get("gates", {}).get(key, {}); d40 = (gate40 or {}).get("gates", {}).get(key, {})
        rows.append([title, st(gate4, key), st(gate40, key), f"4대화: {d4.get('requests', '—')}요청 중 불일치 {d4.get('mismatches', '—')}; 40대화: {d40.get('requests', '—')}요청 중 불일치 {d40.get('mismatches', '—')}"])
    d = (gate4 or {}).get("gates", {}).get("gate3_ours", {})
    rows.append(["3. Ours: 동기 = 비동기(토큰·읽기 바이트), 두 번 실행 = 같음", st(gate4, "gate3_ours"), "(4대화에서만)",
                 (f"동기/비동기 {d['sync_vs_async']['requests']}요청: 토큰 불일치 {d['sync_vs_async']['token_mismatches']}, 바이트 불일치 {d['sync_vs_async']['byte_mismatches']}; "
                  f"두 번 실행 {d['same_request_twice']['requests']}요청: 토큰 불일치 {d['same_request_twice']['token_mismatches']}") if d else "—"])
    d = (gate4 or {}).get("gates", {}).get("gate4_impress", {})
    rows.append(["4. IMPRESS: 유지 100% = 같은 store의 FullLoad loader, 동기 = 비동기", st(gate4, "gate4_impress"), "(4대화에서만)",
                 (f"유지 100% {d['retention_100_vs_fullload_loader_on_same_store']['requests']}요청: 토큰 불일치 {d['retention_100_vs_fullload_loader_on_same_store']['token_mismatches']}; "
                  f"동기/비동기 {d['sync_vs_async']['requests']}요청: 토큰 불일치 {d['sync_vs_async']['token_mismatches']}, 바이트 불일치 {d['sync_vs_async']['byte_mismatches']}") if d else "—"])
    key = "gate5_turn1_tokens_identical_across_four_methods"
    d4 = (gate4 or {}).get("gates", {}).get(key, {}); d40 = (gate40 or {}).get("gates", {}).get(key, {})
    rows.append(["5. T1 생성 토큰이 네 방법에서 같음", st(gate4, key), st(gate40, key),
                 f"4대화 중 다른 대화 {d4.get('differing', '—')}; 40대화 중 다른 대화 {d40.get('differing', '—')}"])
    m = (check or {}).get("models", {}).get(model)
    rows.append(["6. 판정기가 점검 실행의 판정을 재현", m["status"] if m else "미실행", "—",
                 f"{m['stage_judgments_compared']}건 재판정: 승자 일치 {m['same_winner']}, 해석 단계 일치 {m['same_parse_stage']}, 판정문 글자까지 일치 {m['identical_text']}" if m else "—"])
    rows.append(["7. 파일럿 표, 4096 초과 입력 확인 (기록용)", "—", (over or {}).get("status", "미실행"),
                 ("파일럿 안에 4096 초과 입력이 있음" if (over or {}).get("status") == "COVERED_BY_PILOT" else
                  f"따로 돌린 2대화: {over.get('requests', '—')}요청, 실패 {len(over.get('failed_requests', []))}, 4096 초과 {over.get('requests_above_4096', '—')}" if over else "—")])
    L += [table(["Gate", "4대화", "파일럿 40대화", "내용"], rows, ["---"] * 4), "",
          f"본실험 진입 판정: **{admit['status'] if admit else '미기록'}**" + (f" ({admit['at_utc']})" if admit else ""), ""]
    if admit and admit["status"] != "PASS":
        L += ["실패한 gate와 원인:", "", "```json", json.dumps({"parts": admit["parts"], "failed_gates": admit["failed_gates"]}, ensure_ascii=False, indent=1)[:6000], "```", ""]
    sel = C.read_json(C.model_dir(res, model) / "overflow_selection.json")
    if over and over.get("status") not in (None, "COVERED_BY_PILOT") and sel and sel.get("note"):
        L += [f"- 4096 초과 확인의 대화 선택: {sel['note']}", ""]
    return L


def ssd_md(res):
    res = Path(res)
    pre = C.read_json(res / "prerun.json") or {}
    counter = C.read_json(res / "ssd_counter.json") or {}
    before = C.read_json(res / "smart_before.json") or {}; after = C.read_json(res / "smart_after.json") or {}
    L = ["## 9. SSD 쓰기량, SMART, 소요 시간", ""]
    exp = pre.get("expected_ssd_write", {})
    L += [table(["항목", "값"], [
        ["예상 쓰기량 (store 쓰기, 두 모델 합)", fx((exp.get("total_bytes") or 0) / 1e12, 2, " TB")],
        ["  └ 모델별 예상", ", ".join(f"{C.MODEL_NAME[m]} {fx(v, 2)} TB" for m, v in (exp.get("per_model_tb") or {}).items())],
        ["실측: 장치 전체 쓰기량 (커널 카운터, 드라이버 첫 시작 이후, 모든 프로세스)", fx((counter.get("written_bytes") or 0) / 1e12, 3, " TB")],
        ["실측: 장치 전체 읽기량 (같은 카운터)", fx((counter.get("read_bytes") or 0) / 1e12, 3, " TB")],
        ["재부팅 횟수(카운터 구간 수 − 1)", max(0, int(counter.get("boots", 1)) - 1)]], ["---", "---:"]), ""]
    L += ["- 커널 카운터는 파일 시스템이 장치에 보낸 쓰기량이며 NAND에 실제로 쓴 양(쓰기 증폭 포함)이 아니다. 재부팅 직전 마지막 표본 이후의 쓰기는 잡히지 않는다.", ""]

    def smart_line(label, d):
        if not d:
            return [label, "기록 없음", "—", "—"]
        if d.get("status") == "MEASURED":
            return [label, "측정", d.get("percentage_used"), d.get("data_units_written")]
        t = (d.get("available_without_root") or {}).get("hwmon_temperature", {}).get("composite_c")
        return [label, f"측정하지 못함 ({d.get('reason', '')[:90]})", "—", f"온도 {t} °C" if t is not None else "—"]
    L += [table(["SMART", "상태", "percentage used", "data units written / 비고"], [smart_line("실행 전", before), smart_line("실행 후", after)], ["---"] * 4), ""]
    if before.get("status") == "MEASURED" and after.get("status") == "MEASURED":
        dw = (after.get("data_units_written_bytes") or 0) - (before.get("data_units_written_bytes") or 0)
        L += [f"- SMART 전후 차이: data units written {fx(dw / 1e12, 3)} TB, percentage used {before.get('percentage_used')} → {after.get('percentage_used')}.", ""]
    else:
        manual = [p.name for p in sorted(res.glob("smart_manual_*.txt"))]
        L += ["- SMART 값(percentage used, data units written)은 root 권한이 필요한 NVMe admin 명령이라 이 계정으로는 읽지 못했다. "
              "`sudo nvme smart-log /dev/nvme0` 출력을 `smart_manual_before.txt` / `smart_manual_after.txt`로 이 디렉터리에 두면 사람이 비교할 수 있다"
              + (f" (현재 있는 파일: {', '.join(manual)})." if manual else "."), ""]
    steps = C.read_jsonl(res / "driver_steps.jsonl")
    if steps:
        agg = {}
        for s in steps:
            if s.get("seconds") is not None:
                agg.setdefault(s["step"], [0.0, 0])
                agg[s["step"]][0] += s["seconds"]; agg[s["step"]][1] += 1
        L += [table(["단계", "소요 시간 (h)", "실행 횟수"], [[k, fx(v[0] / 3600, 2), v[1]] for k, v in agg.items()]), "",
              f"- 단계 합계 {fx(sum(v[0] for v in agg.values()) / 3600, 2)} h. 예상은 {fx(pre.get('expected_duration', {}).get('total_hours_low'), 1)}–{fx(pre.get('expected_duration', {}).get('total_hours_high'), 1)} h였다.", ""]
    return L


def resume_md(res):
    res = Path(res)
    inv = C.read_jsonl(res / "driver_invocations.jsonl"); log = C.read_jsonl(res / "resume_log.jsonl")
    L = ["## 실행 이력 (이어서 실행)", ""]
    resumes = [i for i in inv if i.get("kind") == "resume"]
    L += [f"- 드라이버 시작 {len(inv)}회 (처음 1회 + 이어서 실행 {len(resumes)}회)."
          + (" 이어서 실행한 시점: " + ", ".join(i["at_utc"] for i in resumes) + "." if resumes else "")]
    inter = [e for e in log if e["mode"].startswith("interrupted")]
    retry = [e for e in log if e["mode"].startswith("request_failed")]
    gpu = [e for e in log if e["mode"].startswith("gpu_used")]
    L += [f"- 끊겨서 T1부터 다시 실행한 대화 {len(inter)}건" + (": " + ", ".join(f"{C.MODEL_NAME[e['model']]} {e['phase']} {e['conversation_id']}({e['at_utc']}, 버린 행 {e['rows_discarded']})" for e in inter) if inter else "") + ".",
          f"- 요청 실패로 T1부터 한 번 더 실행한 대화 {len(retry)}건" + (": " + ", ".join(f"{C.MODEL_NAME[e['model']]} {e['phase']} {e['conversation_id']}" for e in retry[:40]) if retry else "") + ".",
          f"- 다른 프로세스가 GPU를 써서 다시 실행한 대화 {len(gpu)}건.", ""]
    return L


def stage0_md(res):
    p = Path(res) / "stage0" / "stage0.md"
    L = ["## 11. 0단계 결과: 토큰열 결함의 영향 범위", ""]
    if not p.exists():
        return L + ["0단계 산출물이 없다.", ""]
    L += ["점검에서 고친 결함: LLaVA-NeXT의 HF 토크나이저는 원본 LLaVA의 토큰화 절차(`tokenizer_image_token`, 느린 legacy 토크나이저)와 두 곳에서 "
          "다른 토큰을 냈다. `<image>` 바로 뒤에 원본은 `▁`(29871)를 넣는데 HF는 넣지 않았고, `</s>` 뒤의 `USER`를 원본은 `▁US`,`ER`로, HF는 `USER` 한 토큰으로 냈다. "
          "LLaVA-1.5의 HF 토크나이저는 원본과 같았다.", "",
          "끝난 실행이 실제로 보낸 프롬프트(모델·데이터셋마다 T1·T2·T3 각 20개)를 기존 경로의 토큰열과 원본 절차의 토큰열로 비교했다:", "",
          p.read_text(encoding="utf-8").strip(), "",
          "- LLaVA-NeXT의 MT-GQA·MT-VQA 프롬프트는 뽑은 120개 모두에서 토큰 하나(`<image>` 직후의 `▁`)가 빠져 있다. `</s>` 뒤 `USER` 결함은 해당하지 않는다(이 프롬프트에는 `</s>`가 없다).",
          "- LLaVA-1.5는 두 데이터셋 모두 불일치 0건.",
          "- 비교한 기존 경로 토큰열은 각 실행이 기록한 접미부 토큰열과 240개 모두 같았다. 이 차이가 MT-GQA·MT-VQA 점수에 주는 영향은 측정하지 않았다. 코드는 고치지 않았고 실험도 다시 돌리지 않았다.",
          "- 산출물: `stage0/stage0.json`, 스크립트 `scripts/216_convbench_stage0_tokens.py`.", ""]
    return L


CODE_LIST = [
    ("scripts/216_convbench_stage0_tokens.py", "0단계: MT-GQA·MT-VQA 프롬프트의 토큰열 비교(읽기 전용)"),
    ("scripts/convbench_main_common.py", "공통 정의(arm, 단계별 대화 목록, 파일 형식, 동결 검사)"),
    ("scripts/217_eval_convbench_main.py", "본실험 실행기: ConvBench 요청 루프, 대화 단위 기록·이어서 실행·실패 기록. 방법 호출은 scripts/209(LLaVA-NeXT), scripts/206(LLaVA-1.5)을 import해 그대로 부른다"),
    ("scripts/218_validate_convbench_main.py", "gate 1–5, 7과 본실험 진입 판정"),
    ("scripts/219_convbench_main_judge.py", "동결한 판정기(scripts/214의 함수)를 네 방법의 답에 적용, gate 6"),
    ("scripts/220_report_convbench_main.py", "이 보고서와 파일럿 표"),
    ("scripts/221_convbench_main_driver.py, scripts/run_convbench_main.sh", "드라이버(분리 실행, 잠금, 상태 기록, 이어서 실행)"),
    ("tests/test_convbench_main.py", "단위 테스트"),
    ("docs/convbench_main_contract.md", "방법 정의와 gate(실행 전 동결)"),
]

DEVIATIONS = [
    "방법 코드(`mmimpress/`)와 기존 스크립트는 바꾸지 않았다. 바뀐 것은 서버 객체의 생성 상한(16 → 1024)과 프롬프트 문자열, store 메타데이터에 적히는 데이터셋 이름뿐이다.",
    "gate 2의 비교 대상: 점검 실행은 생성 토큰 ID 목록을 남기지 않았다(답 문자열, 생성 토큰 수, 첫 토큰 ID만 있음). 그래서 '생성 토큰이 모두 일치'를 답 문자열·생성 토큰 수·첫 토큰 ID·입력 토큰 수의 일치로 판정했다.",
    "gate 3·4는 추가 arm(동기 읽기, 두 번째 실행, 유지 100%, 같은 store의 FullLoad loader)이 필요해 4대화(9 arm)에서만 판정했다. 파일럿 40대화는 지시대로 4 arm이며 gate 2·5를 다시 판정했다.",
    "gate가 실패한 모델은 본실험을 시작하지 않고 실패 내용을 기록한 뒤, 드라이버는 다른 모델과 판정·보고서까지 진행하고 종료하도록 했다(gate가 모델마다 있으므로 그렇게 해석했다).",
    "'warm-up 없음'은 미리 만들어 둔 KV 없이 모든 방법이 T1을 픽셀부터 계산한다는 뜻으로 읽었다. 기준 실행과 같이 모델을 올린 직후의 측정 제외 요청 1건(프로세스 warm-up)은 그대로 있다.",
    "요청이 실패하면 그 대화를 T1부터 한 번 더 실행하고, 다시 실패하면 실패를 기록한 채 대화를 확정한다(그 방법의 뒤 턴도 실패 처리). 실패가 연속 5대화 또는 누적 40대화를 넘으면 그 단계를 멈춘다. 지시에 없던 세부 규칙이다.",
    "실패한 요청의 판정: 그 턴과 뒤 단계는 '생성 실패'로 0승, 앞 단계는 빠진 뒤 답을 빈 문자열로 두고 판정했다(공식 판정 프롬프트가 세 턴의 답을 모두 넣기 때문).",
    "판정 cache: gate 6에서 판정기가 점검 실행의 판정을 재현함을 확인한 뒤, 판정 프롬프트가 점검 실행과 글자까지 같은 경우(ReComp의 답이 같을 때) 점검 실행의 cache를 읽기 전용으로 읽었다. 건수는 판정 진단 표에 있다.",
    "판정 불가와 별도로 '생성 실패' 열을 두었다.",
    "4096 초과 확인(gate 7): 파일럿에 초과 입력이 없으면 점검 실행에서 ReComp 입력이 가장 길었던 2대화를 골랐다. LLaVA-1.5는 점검 실행에 4096 초과 입력이 없어 가장 긴 2대화를 대신 돌렸다.",
    "대화가 끝날 때마다 GPU 메모리 cache를 비운다(측정 구간 밖). 기준 실행에는 없던 동작으로, 다음 대화의 T1 시간에만 영향을 줄 수 있다.",
    "TTFT 비율에도 대화 단위 bootstrap CI를 붙였다. 품질 비교에는 단계별 차이와 판정 결과가 다른 대화 수를 함께 적었다.",
    "본실험은 577대화 전부를 새로 돌린다(4대화·파일럿 40대화의 결과는 본실험 표에 쓰지 않는다).",
]

LIMITS = [
    "SMART 값(percentage used, data units written)은 권한이 없어 측정하지 못했다. 쓰기량은 커널 카운터(장치 전체)로 대신했다.",
    "실행은 모델마다 한 번이다. 같은 설정을 반복했을 때의 변동은 재지 않았다. greedy 디코딩이므로 같은 장비·같은 코드에서는 결정적이다.",
    "판정기는 Llama-3.1-8B-Instruct 하나이며 이미지를 보지 않는다(공식 절차와 같음). 판정기 자체의 잡음은 FullLoad − ReComp 차이로만 가늠했다.",
    "TTFT는 이 장비(RTX 4090, NVMe SSD, page cache를 비운 상태)의 값이다. 장치 내부 cache는 비우지 않았다.",
    "IMPRESS의 GPU 적재 시간은 probe 중요도 계산·선택과 한 구간에 들어 있어 분리되지 않는다.",
    "0단계에서 확인한 LLaVA-NeXT 토큰 하나의 차이가 MT-GQA·MT-VQA 결과에 주는 영향은 측정하지 않았다.",
    "정밀도(NF4/BF16), greedy 디코딩, HF 이미지 전처리는 점검 실행에서 동결한 대로이며 MetaCompress 공식 설정(샘플링, pad 전처리)과 다르다.",
]


def model_section(res, model, index, tables_out):
    L = [f"# {C.MODEL_NAME[model]}", ""]
    main_done = C.phase_complete(res, model, "main")
    docs = C.committed(res, model, "main", index)
    admit = C.read_json(Path(res) / "gates" / model / "admission.json")
    L += ["## Gate", ""] + gates_md(res, model)
    if not docs:
        L += [f"본실험: 실행되지 않음" + (" (gate 실패)" if admit and admit["status"] != "PASS" else "") + ".", ""]
        pilot = C.committed(res, model, "pilot", index)
        if pilot:
            L += [f"## 파일럿 {len(pilot)}대화 (본실험 아님, 참고)", ""] + system_md(system_tables(pilot), "###", pilot=True)
        return L
    t = system_tables(docs)
    tables_out[model] = {"system": t}
    L += [f"본실험: {len(docs)} / {C.N_CONV if not C.dev_limit(res) else len(docs)} 대화 확정" + ("" if main_done else " — **미완료(중간 집계)**")
          + f", 요청 {sum(len(d['rows']) for d in docs.values()):,}건.", ""]
    q = quality(res, model, len(docs))
    L += ["## 1–2. 품질", ""]
    if q is None:
        L += ["판정이 아직 끝나지 않았다(생성이 모두 끝난 뒤 실행).", ""]
    else:
        tables_out[model]["quality"] = {"summary": q["summary"], "pairs": q["pairs"], "alignment_recomp": q["alignment_recomp"]}
        L += quality_md(q, model)
    rv = recomp_vs_alignment(model, docs); ag = agreement_with_recomp(docs)
    tables_out[model]["recomp_vs_alignment"] = rv; tables_out[model]["token_agreement_with_recomp"] = ag
    L += [f"- 본실험 ReComp와 점검 실행 ReComp의 일치(답 문자열·생성 토큰 수): {rv['identical']:,} / {rv['requests']:,} 요청.",
          f"- T1 생성 토큰이 네 방법에서 같은 대화: {t['t1_identical_conversations']} / {len(docs)}.",
          "- 생성 토큰이 ReComp와 완전히 같은 요청 수(T1/T2/T3): " + "; ".join(
              f"{N[r]} " + "/".join(f"{v[k][0]}" for k in ("T1", "T2", "T3")) + f" (각 {v['T1'][1]}요청 중)" for r, v in ag.items()) + ".", ""]
    L += ["## 3–8. 시스템 지표", ""] + system_md(t)
    L += ["## 실패한 요청", ""] + failures_md(t, C.N_CONV)
    if q is not None:
        L += ["## 10. 판정 진단", ""] + judge_diag_md(q)
    pt = Path(res) / "gates" / model / "gate7_pilot_table.md"
    if pt.exists():
        L += ["## 파일럿 40대화 표 (gate 7, 본실험 아님)", "", pt.read_text(encoding="utf-8").strip(), ""]
    return L


def pilot_table(res, model):
    index, _ = C.load_index()
    docs = C.committed(res, model, "pilot", index)
    t = system_tables(docs)
    rv = recomp_vs_alignment(model, docs)
    R = t["roles"]
    rows = [[N[r], fx(R[r]["ttft"]["T2-T3"]["mean"], 1), fx(R[r]["ttft"]["T2-T3"]["p50"], 1), fx(R[r]["ttft"]["T2-T3"]["p95"], 1),
             fx(R[r]["ssd_mb_per_followup"], 1), fx(R[r]["read_calls_per_followup"], 1), fx(R[r]["generated_tokens_mean"], 1),
             sum(R[r]["cap_reached"].values()), sum(R[r]["requests_above_4096"].values()), R[r]["requests_failed"]] for r in C.MAIN_ROLES]
    md = [f"{C.MODEL_NAME[model]}, 파일럿 {len(docs)}대화 × 4 arm × 3턴 (판정 점수는 두 모델의 생성이 끝난 뒤에만 계산한다).", "",
          table(["방법", "TTFT T2–T3 평균 (ms)", "p50", "p95", "SSD MB/followup", "읽기 호출 수", "평균 생성 길이", "상한 도달", "4096 초과 입력", "실패한 요청"], rows), "",
          table(["비율", "TTFT 평균의 비 [95% CI]"], [[e["ratio_name"], ci(e, 3)] for e in t["ttft_ratios"]]), "",
          f"- ReComp가 점검 실행과 같은 요청: {rv['identical']} / {rv['requests']}. T1 생성 토큰이 네 방법에서 같은 대화: {t['t1_identical_conversations']} / {len(docs)}. "
          f"대화당 평균 {fx(t['conversation_seconds_mean'], 1)} s."]
    out = Path(res) / "gates" / model
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate7_pilot_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    C.atomic_json(out / "gate7_pilot_table.json", {"system": t, "recomp_vs_alignment": rv, "at_utc": C.utc()})
    print("\n".join(md))


def report(res):
    res = Path(res)
    index, _ = C.load_index()
    status = C.read_json(res / "status.json") or {}
    config = C.read_json(res / "config.json") or {}
    pre = C.read_json(res / "prerun.json") or {}
    tables_out = {}
    L = [f"# ConvBench 본실험 보고서 — {res.name}", "",
         f"작성 {C.utc()} · 상태 `{status.get('outcome') if status.get('stage') == 'report' and status.get('outcome') else status.get('stage', '?')}` · 계약 `docs/convbench_main_contract.md` (SHA256 {str(config.get('contract_sha256', ''))[:12]}…) · "
         f"설정 `docs/convbench_baseline_alignment.md`", "",
         "측정 보고이며 GO/NO-GO 판정은 없다. 방법 이름은 ReComp, FullLoad, IMPRESS, Ours.", "",
         "- ReComp: 매 턴 이미지부터 다시 계산. FullLoad: T1에서 저장한 이미지 KV 전체를 읽음. IMPRESS: 공개 구현의 선택 절차(유지 25 %, 재배치 store). "
         "Ours: 중요도 순 재배열 없이 총 KV 예산 25 %(LLaVA-NeXT는 PYR-beta-raw, LLaVA-1.5는 PYR-b20 layer 배분).",
         "- 공통: NF4/BF16/eager, FP16 SSD KV, greedy, 생성 상한 1024, 모든 방법이 T1을 픽셀부터 계산, T2–T3는 자기 방법의 생성 이력, 요청 전 page cache drop, 저장된 KV를 읽는 세 방법은 비동기 읽기.",
         f"- 데이터: ConvBench 577 대화 × 3턴. 모델 2 × 방법 4 × 577 × 3 = 13,848 요청. arm 순서는 대화마다 회전.", ""]
    last = status.get("last_error")
    if last:
        L += [f"마지막 오류 기록: `{json.dumps(last, ensure_ascii=False)[:600]}`", ""]
    for model in C.MODELS:
        L += model_section(res, model, index, tables_out)
    L += ssd_md(res)
    L += resume_md(res)
    L += stage0_md(res)
    L += ["## 12. 코드, 지시와 다르게 했거나 지시에 없던 것, 측정하지 않은 것과 한계", "",
          "추가한 코드(기존 파일은 바꾸지 않음):", "", table(["파일", "내용"], [[f"`{a}`", b] for a, b in CODE_LIST], ["---", "---"]), "",
          "지시와 다르게 했거나 지시에 없던 것:", ""] + [f"{i}. {s}" for i, s in enumerate(DEVIATIONS, 1)] + [
          "", "측정하지 않은 것과 한계:", ""] + [f"- {s}" for s in LIMITS] + [""]
    notes = res / "notes.md"
    if notes.exists():
        L += ["## 추가 메모 (notes.md)", "", notes.read_text(encoding="utf-8").strip(), ""]
    L += ["## 실행 전 점검 (prerun.json 요약)", "",
          f"- 예상 SSD 쓰기량 {fx((pre.get('expected_ssd_write', {}).get('total_bytes') or 0) / 1e12, 2)} TB, 예상 소요 시간 "
          f"{fx(pre.get('expected_duration', {}).get('total_hours_low'), 1)}–{fx(pre.get('expected_duration', {}).get('total_hours_high'), 1)} h, "
          f"여유 공간 {fx(pre.get('free_space', {}).get('free_gib'), 1)} GiB, SMART {pre.get('smart', {}).get('status', '?')}.",
          "- 대화 하나를 확정하면 그 대화의 store를 지웠다.", ""]
    (res / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    C.atomic_json(res / "report_tables.json", {"at_utc": C.utc(), "models": tables_out})
    print(json.dumps({"report": str((res / "report.md").relative_to(ROOT)), "models_with_main_rows": list(tables_out)}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", type=Path, required=True)
    p.add_argument("--pilot", choices=C.MODELS)
    args = p.parse_args()
    if args.pilot:
        return pilot_table(args.res.resolve(), args.pilot)
    report(args.res.resolve())


if __name__ == "__main__":
    main()

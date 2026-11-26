#!/usr/bin/env python3
"""Report of the MT-GQA HyperInfer run (contract docs/mt_gqa_hyperinfer_contract.md, section 7).

Reads committed rows only: HyperInfer from this run, ReComp / FullLoad / IMPRESS / Ours from the
earlier main runs (same image, dialogue and turn).  Writes results/<run>/report.md and report.json.
Not part of the measurement freeze.  Works on a run that is still going or that stopped at a gate:
it reports what exists.

  python scripts/227_report_mt_gqa_hyperinfer.py --run runs/mt_gqa_hyperinfer_<UTC>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mmimpress.mtvqa25k.adapter import read_rows  # noqa: E402

MODELS = ("next", "l15")
MODEL_DIR = {"next": "llava_next", "l15": "llava15"}
MODEL_LABEL = {"next": "LLaVA-NeXT-7B", "l15": "LLaVA-1.5-7B"}
REF_RUN = {"next": ROOT / "runs/mt_gqa_main_v2_20261124T051939Z", "l15": ROOT / "runs/llava15_mt_gqa_main_20261120T165001Z"}
REF_DATE = {"next": "2026-11-24", "l15": "2026-11-21"}
FULL, IMPRESS, HYPER = "fullload_async", "impress_official_async", "hyperinfer_async"
RESAMPLES, SEED = 10000, 1234
MB = 1e6
STAT_KEYS = ("prefetch_wait_ms", "probe_sync_ms", "select_ms", "sync_read_ms", "copy_ms", "gpu_load_ms", "critical_io_ms", "total_io_ms", "io_ratio",
             "prefetched_tokens", "hit_tokens", "used_tokens_after_layer0", "prefetched_bytes", "wasted_bytes", "sync_bytes", "predicted_chunks",
             "cut_chunks", "first_job_chunks", "first_job_rule", "half_rule_jobs", "chunks_per_layer")


# ------------------------------------------------------------------ reading
def _slim(r):
    res = r["result"]
    out = {"score": r["score"], "ttft": r["ttft_ms"], "bytes": r["ssd_read_bytes"], "reads": r["ssd_preads"], "tokens": r["generated_token_ids"],
           "blocked": r.get("read_blocked_ms"), "k": r["k_read_bytes"], "v": r["v_read_bytes"], "probe": r["probe_read_bytes"],
           "io_ms": (res.get("io") or {}).get("ms"), "hook_ms": res.get("hook_ms"), "fallback_rate": res.get("fallback_rate"),
           "mean_jaccard": res.get("mean_jaccard"), "k_frac": res.get("touched_k_chunk_fraction"), "v_frac": res.get("touched_v_chunk_fraction"),
           "probe_wait_ms": res.get("prefetch_wait_ms"), "pid": r.get("process_id"), "at": r.get("completed_at_unix")}
    h = res.get("hyperinfer")
    if h and h.get("kv_prefetch") not in (None, "off"):
        out["h"] = {k: h.get(k) for k in STAT_KEYS}
        out["hl"] = h["layers"]
    return out


def _image(args):
    run, phase, image, methods = args
    cp = Path(run) / phase / "images" / image / "COMMITTED.json"
    if not cp.exists():
        return None
    doc = json.loads(cp.read_text())
    folder = cp.parent / doc["attempt_id"]
    by, dialogs = {}, []
    for r in read_rows(folder / "raw.jsonl"):
        if methods is not None and r["method_id"] not in methods:
            continue
        by[(r["method_id"], r["dialog_id"], r["turn_id"])] = _slim(r)
        if r["dialog_id"] not in dialogs:
            dialogs.append(r["dialog_id"])
    extra = {}
    p = folder / "impress_official_persistence.json"
    if p.exists():
        pj = json.loads(p.read_text())
        extra["persist"] = {"ms": pj.get("service_persistence_ms"), "write_bytes": pj.get("os_write_bytes"), "file_bytes": pj.get("file_logical_bytes")}
    return {"image": image, "dialogs": dialogs, "by": by, "resumed": doc.get("resumed_inside_image", 0), **extra}


def collect(run, phase, images, methods, workers=12):
    with Pool(workers) as pool:
        out = pool.map(_image, [(str(run), phase, i, methods) for i in images], chunksize=4)
    return [r for r in out if r is not None]


# -------------------------------------------------------------------- stats
def cells(records, method, turns=(2, 3)):
    return [r["by"][(method, d, t)] for r in records for d in r["dialogs"] for t in turns]


def arm_row(records, method, name, date):
    sc = {t: np.array([c["score"] for c in cells(records, method, (t,))]) for t in (1, 2, 3)}
    fu = cells(records, method)
    ttft = np.array([c["ttft"] for c in fu])
    return {"method": name, "run_date": date, "images": len(records), "dialogues": int(sc[1].size), "followup_requests": len(fu),
            "em_followup": 100 * np.concatenate([sc[2], sc[3]]).mean(), "em_T1": 100 * sc[1].mean(), "em_T2": 100 * sc[2].mean(), "em_T3": 100 * sc[3].mean(),
            "ttft_mean_ms": ttft.mean(), "ttft_p50_ms": np.percentile(ttft, 50), "ttft_p95_ms": np.percentile(ttft, 95),
            "ssd_mb_per_followup": np.mean([c["bytes"] for c in fu]) / MB, "read_calls_per_followup": np.mean([c["reads"] for c in fu]),
            "t1_ttft_mean_ms": np.mean([c["ttft"] for c in cells(records, method, (1,))])}


def per_image(ra, rb, a, b, key, scale=1.0):
    """[(sum_a, sum_b, count)] per image over the follow-up requests; ra / rb are index-aligned record lists."""
    out = []
    for x, y in zip(ra, rb):
        assert x["image"] == y["image"] and x["dialogs"] == y["dialogs"]
        sa = sum(x["by"][(a, d, t)][key] for d in x["dialogs"] for t in (2, 3)) * scale
        sb = sum(y["by"][(b, d, t)][key] for d in y["dialogs"] for t in (2, 3)) * scale
        out.append((sa, sb, 2 * len(x["dialogs"])))
    return np.asarray(out, dtype=np.float64)


def boot_diff(per):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(per), size=(RESAMPLES, len(per)))
    s = per[idx]
    boot = s[..., 0].sum(1) / s[..., 2].sum(1) - s[..., 1].sum(1) / s[..., 2].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"diff": float(per[:, 0].sum() / per[:, 2].sum() - per[:, 1].sum() / per[:, 2].sum()), "ci_low": float(lo), "ci_high": float(hi)}


def boot_ratio(per):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(per), size=(RESAMPLES, len(per)))
    s = per[idx]
    boot = s[..., 0].sum(1) / s[..., 1].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"ratio": float(per[:, 0].sum() / per[:, 1].sum()), "ci_low": float(lo), "ci_high": float(hi)}


def fx(x, d=1, unit=""):
    return "—" if x is None else f"{x:,.{d}f}{unit}"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def prefetch_summary(fu):
    """Request means and per-layer means over HyperInfer follow-up cells."""
    hs = [c["h"] for c in fu]
    L = len(fu[0]["hl"]["fallback"])
    lay = {k: np.array([[np.nan if v is None else float(v) for v in c["hl"][k]] for c in fu]) for k in fu[0]["hl"] if k not in ("rule", "ready")}
    tot = lambda k: float(sum(h[k] or 0 for h in hs))      # noqa: E731
    jac = lay["jaccard_next"]; both = lay["jaccard_next_both_consensus"] > 0.5
    out = {
        "requests": len(fu),
        "precision": tot("hit_tokens") / tot("prefetched_tokens") if tot("prefetched_tokens") else None,
        "coverage": tot("hit_tokens") / tot("used_tokens_after_layer0") if tot("used_tokens_after_layer0") else None,
        "jaccard_consensus_pairs": float(np.nanmean(jac[both])) if both.any() else None,
        "jaccard_all_pairs": float(np.nanmean(jac)),
        "consensus_pair_share": float(both.mean()),
        "cut_chunk_share": tot("cut_chunks") / tot("predicted_chunks") if tot("predicted_chunks") else None,
        "first_job_chunks_mean": mean([h["first_job_chunks"] for h in hs]),
        "half_rule_requests": int(sum((h["half_rule_jobs"] or 0) > 0 for h in hs)),
        "wasted_mb_per_followup": tot("wasted_bytes") / len(fu) / MB,
        "prefetched_mb_per_followup": tot("prefetched_bytes") / len(fu) / MB,
        "sync_mb_per_followup": tot("sync_bytes") / len(fu) / MB,
        "chunk_read_fraction": float((np.nansum(lay["prefetched_chunks"], 1) + np.nansum(lay["missing_chunks"], 1)).mean() / (L * hs[0]["chunks_per_layer"])) if len({h["chunks_per_layer"] for h in hs}) == 1 else
                               float(np.mean([(np.nansum(lay["prefetched_chunks"][i]) + np.nansum(lay["missing_chunks"][i])) / (L * hs[i]["chunks_per_layer"]) for i in range(len(hs))])),
        "io_ratio_pooled": tot("critical_io_ms") / tot("total_io_ms") if tot("total_io_ms") else None,
        "io_ratio_request_mean": mean([h["io_ratio"] for h in hs]),
    }
    for k in ("prefetch_wait_ms", "probe_sync_ms", "select_ms", "sync_read_ms", "copy_ms", "gpu_load_ms", "critical_io_ms", "total_io_ms"):
        out[k] = mean([h[k] for h in hs])
    chunks = float(np.nansum(lay["prefetched_chunks"]))
    out["worker_k_ms_per_chunk"] = float(np.nansum(lay["worker_k_ms"])) / chunks if chunks else None
    out["worker_v_ms_per_chunk"] = float(np.nansum(lay["worker_v_ms"])) / chunks if chunks else None
    out["worker_probe_ms_per_layer"] = float(np.nanmean(lay["worker_probe_ms"][:, 1:]))
    out["budget_ms_mean"] = float(np.nanmean(lay["budget_ms"][:, 1:]))
    out["compute_prev_ms_mean"] = float(np.nanmean(lay["compute_prev_ms"][:, 1:]))
    import warnings
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_layer = []
        for li in range(L):
            pt, ht, ut = np.nansum(lay["prefetched_tokens"][:, li]), np.nansum(lay["hit_tokens"][:, li]), np.nansum(lay["used_tokens"][:, li])
            per_layer.append({
                "layer": li, "fallback_rate": float(np.nanmean(lay["fallback"][:, li])),
                "wait_ms": float(np.nanmean(lay["wait_ms"][:, li])), "select_ms": float(np.nanmean(lay["select_ms"][:, li])),
                "sync_read_ms": float(np.nanmean(lay["sync_read_ms"][:, li]) + np.nanmean(lay["probe_sync_ms"][:, li])),
                "gpu_load_ms": float(np.nanmean(lay["gpu_load_ms"][:, li])),
                "budget_ms": float(np.nanmean(lay["budget_ms"][:, li])) if li else None,
                "predicted_chunks": float(np.nanmean(lay["predicted_chunks"][:, li])), "prefetched_chunks": float(np.nanmean(lay["prefetched_chunks"][:, li])),
                "needed_chunks": float(np.nanmean(lay["needed_chunks"][:, li])), "missing_chunks": float(np.nanmean(lay["missing_chunks"][:, li])),
                "wasted_chunks": float(np.nanmean(lay["wasted_chunks"][:, li])),
                "precision": float(ht / pt) if pt else None, "coverage": float(ht / ut) if (ut and li) else None,
                "jaccard_next": float(np.nanmean(jac[:, li])) if li < L - 1 else None})
    out["per_layer"] = [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()} for row in per_layer]
    return out


# ---------------------------------------------------------------- per model
def model_report(top, model, config, manifest):
    sub = top / MODEL_DIR[model]
    ref_cfg = json.loads((REF_RUN[model] / "config.json").read_text())
    recomp, full, imp, ours = ref_cfg["main_methods"]
    names = {recomp: "ReComp", full: "FullLoad", imp: "IMPRESS", ours: "Ours", HYPER: "HyperInfer"}
    out = {"model": model, "label": MODEL_LABEL[model], "names": names}
    gates = {}
    for name in ("tests", "logic", f"{model}_gate", f"{model}_dev", f"{model}_admission"):
        p = top / "gates" / f"{name}.json"
        gates[name] = json.loads(p.read_text()) if p.exists() else None
    out["gates"] = gates
    main = collect(sub, "main", manifest["phase_images"]["main"], None) if (sub / "main").exists() else []
    out["main_images"], out["main_complete"] = len(main), (sub / "main_execution_complete.json").exists()
    if main:
        images = [r["image"] for r in main]
        ref = collect(REF_RUN[model], "main", images, None)
        assert [r["image"] for r in ref] == images
        out["arms"] = [arm_row(ref, m, names[m], REF_DATE[model]) for m in (recomp, full, imp)] + \
                      [arm_row(main, HYPER, "HyperInfer", time.strftime("%Y-%m-%d", time.gmtime(min(c["at"] for c in cells(main, HYPER)))))] + \
                      [arm_row(ref, ours, "Ours", REF_DATE[model])]
        pairs = [("Ours − HyperInfer", ref, main, ours, HYPER), ("HyperInfer − IMPRESS", main, ref, HYPER, imp),
                 ("HyperInfer − FullLoad", main, ref, HYPER, full), ("HyperInfer − ReComp", main, ref, HYPER, recomp)]
        out["paired"] = []
        for label, ra, rb, a, b in pairs:
            em = boot_diff(per_image(ra, rb, a, b, "score", 100.0)) if len(main) > 1 else None
            tt = boot_diff(per_image(ra, rb, a, b, "ttft")) if len(main) > 1 else None
            out["paired"].append({"comparison": label, "em": em, "ttft": tt})
        ratios = [("Ours / HyperInfer", ref, main, ours, HYPER), ("HyperInfer / IMPRESS", main, ref, HYPER, imp),
                  ("HyperInfer / FullLoad", main, ref, HYPER, full), ("HyperInfer / ReComp", main, ref, HYPER, recomp)]
        out["ratios"] = [dict(label=label, **(boot_ratio(per_image(ra, rb, a, b, "ttft")) if len(main) > 1 else {})) for label, ra, rb, a, b in ratios]
        # item 4: tokens against the recorded IMPRESS rows
        bad, n = [], 0
        for x, y in zip(main, ref):
            for d in x["dialogs"]:
                for t in (1, 2, 3):
                    n += 1
                    if x["by"][(HYPER, d, t)]["tokens"] != y["by"][(imp, d, t)]["tokens"]:
                        bad.append({"image": x["image"], "dialog_id": d, "turn_id": t})
        out["identity"] = {"requests": n, "equal": n - len(bad), "mismatches": bad[:50], "mismatch_count": len(bad)}
        fu = cells(main, HYPER)
        out["prefetch"] = prefetch_summary(fu)
        imp_fu = cells(ref, imp)
        out["bytes_vs_impress"] = {"hyperinfer_mb": np.mean([c["bytes"] for c in fu]) / MB, "impress_mb": np.mean([c["bytes"] for c in imp_fu]) / MB,
                                   "ratio": float(np.sum([c["bytes"] for c in fu]) / np.sum([c["bytes"] for c in imp_fu]))}
        out["selection"] = {"hyperinfer": {"fallback_rate": mean([c["fallback_rate"] for c in fu]), "mean_jaccard": mean([c["mean_jaccard"] for c in fu]),
                                           "kv_chunk_read_fraction": out["prefetch"]["chunk_read_fraction"],
                                           "kv_chunk_needed_fraction": mean([c["k_frac"] for c in fu])},
                            "impress": {"fallback_rate": mean([c["fallback_rate"] for c in imp_fu]), "mean_jaccard": mean([c["mean_jaccard"] for c in imp_fu]),
                                        "k_chunk_read_fraction": mean([c["k_frac"] for c in imp_fu]), "v_chunk_read_fraction": mean([c["v_frac"] for c in imp_fu])}}
        out["decomposition"] = {
            "HyperInfer": {"read_wait_ms": mean([c["blocked"] for c in fu]), "prefetch_wait_ms": out["prefetch"]["prefetch_wait_ms"],
                           "sync_read_ms": out["prefetch"]["sync_read_ms"], "probe_sync_ms": out["prefetch"]["probe_sync_ms"],
                           "host_copy_ms": out["prefetch"]["copy_ms"], "gpu_load_ms": out["prefetch"]["gpu_load_ms"],
                           "select_ms": out["prefetch"]["select_ms"], "hook_ms": mean([c["hook_ms"] for c in fu]), "ttft_ms": mean([c["ttft"] for c in fu])},
            "IMPRESS": {"read_wait_ms": mean([c["blocked"] for c in imp_fu]), "hook_ms": mean([c["hook_ms"] for c in imp_fu]),
                        "probe_wait_ms": mean([c["probe_wait_ms"] for c in imp_fu]), "ttft_ms": mean([c["ttft"] for c in imp_fu])},
            "FullLoad": {"read_wait_ms": mean([c["blocked"] for c in cells(ref, full)]), "ttft_ms": mean([c["ttft"] for c in cells(ref, full)])},
            "Ours": {"read_wait_ms": mean([c["blocked"] for c in cells(ref, ours)]), "ttft_ms": mean([c["ttft"] for c in cells(ref, ours)])}}
        ps = [r["persist"] for r in main if r.get("persist")]
        out["t1_store"] = {"images": len(ps), "persist_ms_mean": mean([p["ms"] for p in ps]), "write_mb_mean": mean([p["write_bytes"] for p in ps]) / MB if ps else None,
                           "store_mb_mean": mean([p["file_bytes"] for p in ps]) / MB if ps else None,
                           "write_gb_total": sum(p["write_bytes"] or 0 for p in ps) / 1e9}
    # DEV: IMPRESS and HyperInfer measured together (I/O ratio), and FullLoad
    if (sub / "dev").exists():
        dev = collect(sub, "dev", manifest["phase_images"]["dev"], {FULL, IMPRESS, HYPER})
        if dev:
            d_imp, d_hyp, d_full = cells(dev, IMPRESS), cells(dev, HYPER), cells(dev, FULL)
            ratio = lambda cs: float(sum(c["blocked"] for c in cs) / sum(c["io_ms"] for c in cs))      # noqa: E731
            out["dev"] = {"images": len(dev), "followups": len(d_imp),
                          "impress": {"ttft_ms": mean([c["ttft"] for c in d_imp]), "io_ratio": ratio(d_imp), "read_wait_ms": mean([c["blocked"] for c in d_imp]),
                                      "io_ms": mean([c["io_ms"] for c in d_imp]), "em": 100 * mean([c["score"] for c in d_imp])},
                          "hyperinfer": {"ttft_ms": mean([c["ttft"] for c in d_hyp]), "read_wait_ms": mean([c["blocked"] for c in d_hyp]),
                                         "io_ms": mean([c["io_ms"] for c in d_hyp]), "em": 100 * mean([c["score"] for c in d_hyp]),
                                         "summary": {k: v for k, v in prefetch_summary(d_hyp).items() if k != "per_layer"}},
                          "fullload": {"ttft_ms": mean([c["ttft"] for c in d_full]), "em": 100 * mean([c["score"] for c in d_full])}}
            out["dev"]["hyperinfer"]["io_ratio"] = out["dev"]["hyperinfer"]["summary"]["io_ratio_pooled"]
            out["dev"]["ttft_ratio_hyperinfer_over_impress"] = out["dev"]["hyperinfer"]["ttft_ms"] / out["dev"]["impress"]["ttft_ms"]
    # execution record
    ex = {}
    for ph in ("gate", "dev", "main"):
        starts = sorted(sub.glob(f"{ph}_start_*.json"))
        done = sub / f"{ph}_execution_complete.json"
        if starts:
            first = json.loads(starts[0].read_text())["started_unix"]
            end = json.loads(done.read_text())["finished_unix"] if done.exists() else None
            ex[ph] = {"process_starts": len(starts), "complete": done.exists(), "wall_hours": (end - first) / 3600 if end else None,
                      "half_rule_note": [json.loads(s.read_text()).get("resume") for s in starts]}
    out["execution"] = ex
    log = sub / "resume_log.jsonl"
    out["resume_events"] = [json.loads(l) for l in log.read_text().splitlines() if l.strip()] if log.exists() else []
    return out


# ----------------------------------------------------------------- markdown
def gate_md(r):
    lines = ["| gate | 4장 | DEV 40장 |", "|---|---|---|"]
    g4, gd = r["gates"].get(f"{r['model']}_gate"), r["gates"].get(f"{r['model']}_dev")
    rows = [("2. prefetch 끔 ≡ IMPRESS", "gate2_prefetch_off_equals_impress"), ("3. 출력 동일성 (HyperInfer ≡ IMPRESS)", "gate3_output_equals_impress"),
            ("4. 기존 IMPRESS 기록과 일치", "gate4_tokens_equal_recorded_impress"), ("5. 읽지 않은 KV를 쓰지 않음", "gate5_unread_rows_not_used"),
            ("6. prefetch 동기 실행 ≡ 비동기", "gate6_sync_equals_async"), ("6. 두 번 실행", "gate6_repeat")]

    def cell(doc, key):
        if not doc:
            return "미실행"
        if doc.get("status") == "FAIL" and "gates" not in doc:
            return "단계 미완료"
        g = doc["gates"].get(key)
        if g is None:
            return "해당 없음"
        n = g.get("requests")
        bad = g.get("mismatches", g.get("guard_failures", 0)) + (g.get("against_impress", {}).get("mismatches", 0) if key.startswith("gate5") else 0)
        return f"{g['status']} ({n - bad:,} / {n:,})" if n is not None else g["status"]

    for label, key in rows:
        lines.append(f"| {label} | {cell(g4, key)} | {cell(gd, key)} |")
    adm = r["gates"].get(f"{r['model']}_admission")
    lines += ["", f"admission: **{adm['status'] if adm else '미판정'}**" + (f" (실패: {', '.join(adm['failed'])})" if adm and adm.get("failed") else "")]
    for doc in (g4, gd):
        if doc and doc.get("failed"):
            for key in doc["failed"]:
                g = (doc.get("gates") or {}).get(key, {})
                first = (g.get("first_mismatches") or g.get("first_guard_failures") or (g.get("against_impress") or {}).get("first_mismatches") or [])[:5]
                lines += ["", f"실패한 gate `{key}` ({doc['phase']}): 불일치 {g.get('mismatches', g.get('guard_failures'))}건. 처음 몇 건:", "```json",
                          json.dumps(first, ensure_ascii=False, indent=1)[:3000], "```"]
    return "\n".join(lines)


def model_md(r, constants):
    m = r["model"]
    L = [f"## {r['label']}", "", "### 구현 검증 (gate)", "", gate_md(r), ""]
    if not r.get("arms"):
        L += ["본실험 행이 없다(시작하지 않았거나 아직 커밋된 이미지가 없다).", ""]
    else:
        partial = "" if r["main_complete"] else f" **진행 중 또는 중단: {r['main_images']} / 398 이미지 기준의 중간 값이다.**"
        L += [f"### 1. 결과 표 ({r['main_images']} images){partial}", "",
              "| 방법 | 실행일 | followup EM | T1 EM | T2 EM | T3 EM | TTFT 평균 (ms) | p50 | p95 | SSD MB/followup | 읽기 호출 수/followup |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for a in r["arms"]:
            L.append(f"| {a['method']} | {a['run_date']} | {a['em_followup']:.2f} | {a['em_T1']:.2f} | {a['em_T2']:.2f} | {a['em_T3']:.2f} | "
                     f"{a['ttft_mean_ms']:.1f} | {a['ttft_p50_ms']:.1f} | {a['ttft_p95_ms']:.1f} | {a['ssd_mb_per_followup']:.1f} | {a['read_calls_per_followup']:.1f} |")
        L += ["", "HyperInfer만 이번 실행의 값이고 나머지는 기존 본실험의 같은 이미지·대화·턴 기록이다. TTFT와 읽기량은 재사용 턴(T2–T3) 기준이다. ReComp는 store를 읽지 않는다.", "",
              "### 2. 짝지은 비교 (이미지 단위 paired bootstrap 10,000회, seed 1234, 95% CI)", "",
              "| 비교 | EM 차이 (pp) | 95% CI | TTFT 차이 (ms) | 95% CI |", "|---|---|---|---|---|"]
        for p in r["paired"]:
            if p["em"]:
                L.append(f"| {p['comparison']} | {p['em']['diff']:+.2f} | [{p['em']['ci_low']:+.2f}, {p['em']['ci_high']:+.2f}] | "
                         f"{p['ttft']['diff']:+.1f} | [{p['ttft']['ci_low']:+.1f}, {p['ttft']['ci_high']:+.1f}] |")
        L += ["", "TTFT 차이는 서로 다른 실행에서 잰 값끼리의 차이다(HyperInfer는 이번 실행, 상대는 기존 본실험). 같은 실행에서 함께 잰 비교는 아래 DEV 40장 값을 본다.", "",
              "### 3. TTFT 비율 (평균의 비, 95% CI)", "", "| 비율 | 값 | 95% CI |", "|---|---|---|"]
        for x in r["ratios"]:
            L.append(f"| {x['label']} | {fx(x.get('ratio'), 3)} | [{fx(x.get('ci_low'), 3)}, {fx(x.get('ci_high'), 3)}] |")
        i = r["identity"]
        L += ["", "### 4. 출력 동일성", "",
              f"본실험 {i['requests']:,}요청(T1–T3) 중 HyperInfer의 생성 토큰이 기존 IMPRESS 기록과 일치한 요청: **{i['equal']:,} / {i['requests']:,}**."]
        if i["mismatch_count"]:
            L += ["", f"불일치 {i['mismatch_count']}건(처음 50건): " + ", ".join(f"{b['dialog_id']} T{b['turn_id']}" for b in i["mismatches"]),
                  "원인은 요청별 행(`runs/<run>/" + MODEL_DIR[m] + "/main/…/raw.jsonl.gz`)의 layer별 사용 집합 요약과 기존 기록을 대조해 확인해야 한다. 이 보고서 스크립트는 원인을 판정하지 않는다."]
    gd = r["gates"].get(f"{m}_dev")
    if gd and gd.get("gate8_ttft_continuity"):
        L += ["", "### 5. TTFT 연속성 (gate 8, DEV 40장; gate 아님)", "",
              "| 방법 | 턴 | 요청 | 이번 DEV 평균 (ms) | 기존 본실험 평균 (ms) | 비율 | 토큰 일치 |", "|---|---|---|---|---|---|---|"]
        for method, label in ((FULL, "FullLoad"), (IMPRESS, "IMPRESS")):
            e = gd["gate8_ttft_continuity"]["methods"][method]
            tk = e["tokens_equal_to_earlier_run"]
            for key, turn in (("followup", "T2–T3"), ("turn1", "T1")):
                x = e[key]
                L.append(f"| {label} | {turn} | {x['requests']:,} | {x['this_run_mean_ms']:.1f} | {x['earlier_run_mean_ms']:.1f} | {x['ratio_of_means']:.3f} | "
                         f"{tk['equal']:,} / {tk['requests']:,} |")
        L += ["", "이번 DEV는 7개 arm을 회전시켰고 기존 본실험은 4개 arm을 회전시켰다."]
    if r.get("dev"):
        d = r["dev"]
        L += ["", f"### 같은 실행에서 함께 잰 값 (DEV {d['images']}장, followup {d['followups']:,}요청)", "",
              "| 방법 | followup EM | TTFT 평균 (ms) | 읽기 대기 (ms) | 전체 pread 시간 (ms) | I/O ratio |", "|---|---|---|---|---|---|",
              f"| FullLoad | {d['fullload']['em']:.2f} | {d['fullload']['ttft_ms']:.1f} | — | — | — |",
              f"| IMPRESS | {d['impress']['em']:.2f} | {d['impress']['ttft_ms']:.1f} | {d['impress']['read_wait_ms']:.1f} | {d['impress']['io_ms']:.1f} | {d['impress']['io_ratio']:.3f} |",
              f"| HyperInfer | {d['hyperinfer']['em']:.2f} | {d['hyperinfer']['ttft_ms']:.1f} | {d['hyperinfer']['read_wait_ms']:.1f} | {d['hyperinfer']['io_ms']:.1f} | {fx(d['hyperinfer']['io_ratio'], 3)} |",
              "", f"DEV에서 TTFT 비율 HyperInfer / IMPRESS = **{d['ttft_ratio_hyperinfer_over_impress']:.3f}** (같은 실행, 같은 요청, arm 순서 회전)."]
    if r.get("prefetch"):
        p = r["prefetch"]
        b = r["bytes_vs_impress"]
        L += ["", "### 6. prefetch 통계 (본실험, 재사용 턴 요청 평균)", "",
              "| 항목 | 값 |", "|---|---|",
              f"| prefetch된 토큰 중 실제 사용 집합에 든 비율 | {fx(100 * p['precision'] if p['precision'] is not None else None, 1, '%')} |",
              f"| 사용 집합 중 prefetch로 채워진 비율 (layer 1 이후) | {fx(100 * p['coverage'] if p['coverage'] is not None else None, 1, '%')} |",
              f"| 인접 layer 선택 집합 Jaccard, fallback이 아닌 layer 쌍 | {fx(p['jaccard_consensus_pairs'], 3)} (전체 쌍의 {100 * p['consensus_pair_share']:.1f}%) |",
              f"| 인접 layer 선택 집합 Jaccard, 전체 쌍 | {fx(p['jaccard_all_pairs'], 3)} |",
              f"| 예산에 걸려 잘린 chunk 비율 (예측 chunk 대비) | {fx(100 * p['cut_chunk_share'] if p['cut_chunk_share'] is not None else None, 1, '%')} |",
              f"| 첫 prefetch(layer 0 → 1)에서 읽은 chunk 수 | {fx(p['first_job_chunks_mean'], 2)} |",
              f"| 앞 절반 규칙이 적용된 요청 수 | {p['half_rule_requests']:,} / {p['requests']:,} |",
              f"| prefetch로 읽은 양 | {p['prefetched_mb_per_followup']:.1f} MB/followup |",
              f"| 읽었지만 쓰지 않은 양 | {p['wasted_mb_per_followup']:.1f} MB/followup |",
              f"| 동기로 읽은 양 (K·V) | {p['sync_mb_per_followup']:.1f} MB/followup |",
              f"| 총 읽기 바이트, IMPRESS 대비 | {b['hyperinfer_mb']:.1f} MB 대 {b['impress_mb']:.1f} MB (×{b['ratio']:.3f}) |",
              f"| 임계 경로: prefetch 대기 | {p['prefetch_wait_ms']:.1f} ms |",
              f"| 임계 경로: layer 0의 probe key 읽기 | {p['probe_sync_ms']:.2f} ms |",
              f"| 임계 경로: 선택 계산 | {p['select_ms']:.1f} ms |",
              f"| 임계 경로: 빠진 KV 동기 읽기 | {p['sync_read_ms']:.1f} ms |",
              f"| I/O ratio (임계 경로 I/O / 전체 pread 시간) | {fx(p['io_ratio_pooled'], 3)} (요청별 평균 {fx(p['io_ratio_request_mean'], 3)}) |",
              f"| layer당 예산 평균 (직전 layer의 계산 + 선택 시간) | {fx(p['budget_ms_mean'], 2)} ms (그중 계산 {fx(p['compute_prev_ms_mean'], 2)} ms) |",
              f"| worker의 실측 chunk당 읽기 시간 (동결 상수 K {constants['models'][m]['k_chunk_s'] * 1e3:.3f} / V {constants['models'][m]['v_chunk_s'] * 1e3:.3f} ms) | "
              f"K {fx(p['worker_k_ms_per_chunk'], 3)} ms, V {fx(p['worker_v_ms_per_chunk'], 3)} ms |",
              f"| worker의 layer당 probe key 읽기 시간 | {fx(p['worker_probe_ms_per_layer'], 3)} ms |"]
        if r.get("dev"):
            L.append(f"| I/O ratio, DEV 40장에서 함께 잰 값 | HyperInfer {fx(r['dev']['hyperinfer']['io_ratio'], 3)}, IMPRESS {r['dev']['impress']['io_ratio']:.3f} |")
        L += ["", "사용 집합은 선택된 이미지 토큰 + separator다. Jaccard는 separator를 뺀 이미지 토큰 집합으로 계산했다. I/O ratio의 분모는 worker가 읽은 시간을 포함한 모든 pread 시간의 합이다.", "",
              "layer별 평균:", "",
              "| layer | fallback | 예산 (ms) | 예측 chunk | prefetch chunk | 필요 chunk | 빠진 chunk | 버린 chunk | 적중 비율 | 채움 비율 | 다음 layer와 Jaccard | 대기 (ms) | 선택 (ms) | 동기 읽기 (ms) | GPU 적재 (ms) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in p["per_layer"]:
            L.append(f"| {x['layer']} | {100 * x['fallback_rate']:.0f}% | {fx(x['budget_ms'], 2)} | {x['predicted_chunks']:.1f} | {x['prefetched_chunks']:.1f} | {x['needed_chunks']:.1f} | "
                     f"{x['missing_chunks']:.1f} | {x['wasted_chunks']:.1f} | {fx(100 * x['precision'] if x['precision'] is not None else None, 0, '%')} | "
                     f"{fx(100 * x['coverage'] if x['coverage'] is not None else None, 0, '%')} | {fx(x['jaccard_next'], 2)} | {x['wait_ms']:.2f} | {x['select_ms']:.2f} | "
                     f"{x['sync_read_ms']:.2f} | {x['gpu_load_ms']:.2f} |")
        s = r["selection"]
        L += ["", "### 7. fallback, Jaccard, chunk 읽기 비율", "",
              "| 방법 | fallback 비율 | probe head 평균 Jaccard | K·V chunk 읽기 비율 |", "|---|---|---|---|",
              f"| HyperInfer (이번 실행) | {100 * s['hyperinfer']['fallback_rate']:.1f}% | {s['hyperinfer']['mean_jaccard']:.3f} | "
              f"{100 * s['hyperinfer']['kv_chunk_read_fraction']:.1f}% (선택에 필요한 chunk {100 * s['hyperinfer']['kv_chunk_needed_fraction']:.1f}%) |",
              f"| IMPRESS (기존 본실험) | {100 * s['impress']['fallback_rate']:.1f}% | {s['impress']['mean_jaccard']:.3f} | "
              f"K {100 * s['impress']['k_chunk_read_fraction']:.1f}%, V {100 * s['impress']['v_chunk_read_fraction']:.1f}% |"]
        d = r["decomposition"]
        L += ["", "### 8. 재사용 턴 시간 분해 (요청 평균, ms)", "",
              "| 방법 | TTFT | 읽기 대기 합계 | prefetch 대기 | 동기 읽기 (KV + layer 0 probe) | host 복사 | GPU 적재 | 선택 계산 | hook 합계 |", "|---|---|---|---|---|---|---|---|---|",
              f"| HyperInfer | {d['HyperInfer']['ttft_ms']:.1f} | {fx(d['HyperInfer']['read_wait_ms'])} | {d['HyperInfer']['prefetch_wait_ms']:.1f} | "
              f"{d['HyperInfer']['sync_read_ms'] + d['HyperInfer']['probe_sync_ms']:.1f} | {d['HyperInfer']['host_copy_ms']:.1f} | {d['HyperInfer']['gpu_load_ms']:.1f} | "
              f"{d['HyperInfer']['select_ms']:.1f} | {fx(d['HyperInfer']['hook_ms'])} |",
              f"| IMPRESS (기존) | {d['IMPRESS']['ttft_ms']:.1f} | {fx(d['IMPRESS']['read_wait_ms'])} | probe 대기 {fx(d['IMPRESS']['probe_wait_ms'])} | 미측정 | 미측정 | 미측정 | 미측정 | {fx(d['IMPRESS']['hook_ms'])} |",
              f"| FullLoad (기존) | {d['FullLoad']['ttft_ms']:.1f} | {fx(d['FullLoad']['read_wait_ms'])} | — | — | — | — | — | — |",
              f"| Ours (기존) | {d['Ours']['ttft_ms']:.1f} | {fx(d['Ours']['read_wait_ms'])} | — | — | — | — | — | — |",
              "", "HyperInfer의 읽기 대기 합계 = prefetch 대기 + layer 0의 probe 읽기 + 빠진 KV 읽기 호출(prefetch buffer에서 옮기는 host 복사 포함). "
                  "GPU 적재는 cache 기록 호출의 시간이다. 기존 IMPRESS 행에는 GPU 적재와 선택 계산이 따로 기록돼 있지 않다."]
        t = r["t1_store"]
        L += ["", "### 9. T1 저장 비용과 소요 시간", "",
              f"- T1 store 기록(IMPRESS 형식, 이미지당 한 번): 평균 {fx(t['persist_ms_mean'], 0)} ms, 쓰기 {fx(t['write_mb_mean'], 0)} MB/이미지, store 크기 {fx(t['store_mb_mean'], 0)} MB, "
              f"본실험 합계 {t['write_gb_total']:.1f} GB ({t['images']} images)."]
    ex = r.get("execution", {})
    for ph in ("gate", "dev", "main"):
        if ph in ex:
            e = ex[ph]
            L.append(f"- `{ph}` 단계: 프로세스 시작 {e['process_starts']}회(이어서 실행 {e['process_starts'] - 1}회), "
                     + (f"소요 {e['wall_hours']:.2f}시간." if e["wall_hours"] is not None else "미완료."))
    ev = [e for e in r.get("resume_events", []) if e.get("rerun_dialogues") or e.get("restart_dialogues")]
    if ev:
        L += ["", "이어서 실행하며 다시 실행한 대화:"]
        for e in ev:
            L.append(f"- {e.get('phase')} / {e.get('image_id')} ({e.get('mode')}): " + ", ".join(e.get("rerun_dialogues") or e.get("restart_dialogues") or []))
    else:
        L.append("- 이어서 실행하며 다시 실행한 대화: 없음.")
    return "\n".join(L)


CODE_MD = """### 추가한 코드 (기존 파일은 수정하지 않았다)

- `mmimpress/hyperinfer.py`: HyperInfer. IMPRESS 선택기를 상속하고 store 읽기 지점만 prefetch buffer를 거치게 한다. 순수 함수(위치 표, 예측 chunk, 예산 자르기), worker 작업, 디버그용 NaN 검사 래퍼, 요청 함수.
- `tests/test_hyperinfer.py`: 15개 테스트. 공개 코드 `prefetch_by_chunk` 직접 호출 대조, prefetch 끔 ≡ IMPRESS, prefetch 켬에서 선택·mask·cache 내용이 IMPRESS와 같음, 바이트·chunk 회계, 예산 상태의 요청 간 이어짐, NaN 검사 래퍼.
- `scripts/225_eval_mt_gqa_hyperinfer.py`: 실행(상수 측정, freeze, gate·dev·main 단계). 두 모델의 기존 본실험 하네스(`scripts/222`, `scripts/206`)를 import해 그대로 쓴다.
- `scripts/226_validate_mt_gqa_hyperinfer.py`: gate 판정.
- `scripts/227_report_mt_gqa_hyperinfer.py`: 이 보고서.
- `scripts/run_mt_gqa_hyperinfer.sh`: 드라이버.
- 문서: `docs/hyperinfer_port_contract.md`(원본 대조표), `docs/mt_gqa_hyperinfer_contract.md`(실행 계약), `docs/mt_gqa_hyperinfer_chunk_times.json`(동결한 상수).

### 측정하지 않은 것과 한계

- ReComp, FullLoad, IMPRESS, Ours는 다시 돌리지 않았다. 표의 값은 기존 본실험의 것이고, HyperInfer와의 TTFT 차이는 서로 다른 실행 사이의 차이다. 같은 실행에서 함께 잰 값은 DEV 40장의 FullLoad, IMPRESS, HyperInfer뿐이다.
- 캐시 계층(GPU·CPU 캐시, CKLFU), 주기적 재배치, warm-up은 이식하지 않았다. 모든 요청이 cold SSD다.
- chunk 읽기 시간 상수는 GPU가 쉬는 상태에서 잰 평균이다. 요청 중에는 GPU 계산과 Python 실행이 겹치므로 worker의 실제 읽기 시간은 다를 수 있다. 차이는 prefetch 대기 시간으로 드러난다.
- 기존 IMPRESS 행에는 GPU 적재 시간과 선택 계산 시간이 따로 없어 8절에서 HyperInfer와 나란히 놓지 못했다.
- Oracle, History 방식(논문 VI-D의 비교 대상)은 만들지 않았다.
- SSD 쓰기량은 장치 전체의 커널 카운터다(이 실행 외의 쓰기 포함, NAND 쓰기량 아님). SMART 값은 root 권한이 없어 읽지 못했다.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    top = args.run.resolve()
    out_root = ROOT / "results" / top.name
    config = json.loads((top / "config.json").read_text())
    manifest = json.loads((top / "manifest.json").read_text())
    status = json.loads((out_root / "status.json").read_text()) if (out_root / "status.json").exists() else {}
    constants = json.loads((top / "mt_gqa_hyperinfer_chunk_times.json").read_text())
    reports, errors = {}, {}
    for m in MODELS:
        try:
            reports[m] = model_report(top, m, config, manifest)
        except Exception:      # noqa: BLE001  (the other model's section is still written)
            errors[m] = traceback.format_exc()
    L = [f"# MT-GQA × HyperInfer 측정 보고 ({top.name})", "",
         f"작성 {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}. 계약: `docs/mt_gqa_hyperinfer_contract.md` (SHA256 `{config['contract_sha256'][:16]}…`). "
         "측정 보고이며 GO/NO-GO 판정은 없다.", "",
         "## 요약", ""]
    for m in MODELS:
        r = reports.get(m)
        if r is None:
            L.append(f"- **{MODEL_LABEL[m]}**: 보고서 생성 중 오류(아래 참조).")
            continue
        adm = r["gates"].get(f"{m}_admission")
        if not r.get("arms"):
            L.append(f"- **{MODEL_LABEL[m]}**: 본실험 값 없음. admission {adm['status'] if adm else '미판정'}"
                     + (f" (실패한 단계: {', '.join(adm['failed'])})." if adm and adm.get("failed") else "."))
            continue
        a = {x["method"]: x for x in r["arms"]}
        rv = {x["label"]: x for x in r["ratios"]}
        note = "" if r["main_complete"] else f" (중간 값: {r['main_images']} / 398 이미지)"
        L.append(f"- **{MODEL_LABEL[m]}**{note}: HyperInfer followup EM {a['HyperInfer']['em_followup']:.2f}%, TTFT {a['HyperInfer']['ttft_mean_ms']:.1f} ms. "
                 f"IMPRESS {a['IMPRESS']['em_followup']:.2f}% / {a['IMPRESS']['ttft_mean_ms']:.1f} ms, Ours {a['Ours']['em_followup']:.2f}% / {a['Ours']['ttft_mean_ms']:.1f} ms. "
                 f"TTFT 비율 HyperInfer / IMPRESS {fx(rv['HyperInfer / IMPRESS'].get('ratio'), 3)}, Ours / HyperInfer {fx(rv['Ours / HyperInfer'].get('ratio'), 3)}. "
                 f"생성 토큰이 기존 IMPRESS 기록과 일치한 요청 {r['identity']['equal']:,} / {r['identity']['requests']:,}.")
    tests, logic = (reports.get("next") or reports.get("l15") or {"gates": {}})["gates"].get("tests"), (reports.get("next") or reports.get("l15") or {"gates": {}})["gates"].get("logic")
    L += ["", "## 두 모델 공통 gate", "",
          f"- gate 1 (로직 대조, 공개 코드 직접 호출): **{logic['status'] if logic else '미실행'}**"
          + (f". 테스트 {logic['ran']}개, 공개 코드 대조 테스트 {len(logic['public_comparison_tests'])}개, 건너뜀 {logic['skipped']}개. 직접 호출한 것: {logic['called_directly']}. "
             f"직접 호출하지 못한 것: {logic['not_callable']}." if logic else "."),
          f"- gate 7 (기존 테스트, 새 실패 0개): **{tests['status'] if tests else '미실행'}**"]
    if tests:
        for s in tests["suites"]:
            L.append(f"  - transformers {s['transformers']}: {s['ran']}개 실행, 새 실패 {len(s['new_failures'])}개, 이전부터 실패하던 테스트 {len(s['baseline_failures_for_this_environment'])}개: "
                     + "; ".join(s["baseline_failures_for_this_environment"]))
    L.append("")
    for m in MODELS:
        if m in reports:
            try:
                L += [model_md(reports[m], constants), ""]
            except Exception:      # noqa: BLE001
                errors[m] = traceback.format_exc()
        if m in errors:
            L += [f"## {MODEL_LABEL[m]}", "", "이 모델의 절을 만드는 중 오류가 났다:", "```", errors[m][-3000:], "```", ""]
    counter = top / "ssd_counter.json"
    written = json.loads(counter.read_text()).get("written_bytes") if counter.exists() else None
    prerun = json.loads((out_root / "prerun.json").read_text()) if (out_root / "prerun.json").exists() else {}
    inv = (top / "driver_invocations.jsonl").read_text().splitlines() if (top / "driver_invocations.jsonl").exists() else []
    post = json.loads((top / "gates" / "posthoc.json").read_text()) if (top / "gates" / "posthoc.json").exists() else None
    if post is None:
        protection = "미확인(posthoc 미실행)."
    else:
        if post["existing_files_unchanged"]:
            protection = "`results/`, `runs/`, `kvstore*`의 기존 파일 목록(크기·수정 시각)이 freeze 때와 같다."
        else:
            ch = post["changed"]
            protection = (f"목록이 달라졌다. 수정 {len(ch['modified'])}건, 삭제 {len(ch['removed'])}건, 추가 {len(ch['added'])}건"
                          "(추가는 다른 작업의 새 파일).")
        protection += " 동결한 소스 파일 변경 없음." if not post["frozen_source_changed"] else f" 동결 소스 변경: {post['frozen_source_changed']}"
    L += ["## 실행 전체", "",
          f"- 드라이버 시작 {len(inv)}회(이어서 실행 {max(0, len(inv) - 1)}회). 보고서를 쓰기 직전의 드라이버 단계: `{status.get('stage')}`."
          + (f" 마지막 오류: {json.dumps(status.get('last_error'), ensure_ascii=False)}" if status.get("last_error") else ""),
          f"- SSD 쓰기량(장치 전체 커널 카운터, 드라이버 시작 이후): {fx(written / 1e12 if written else None, 3)} TB. 사전 예상 {fx((prerun.get('expected_ssd_write') or {}).get('total_tb'), 3)} TB.",
          f"- 요청: 완료 {status.get('completed_requests')} / 전체 {status.get('expected_requests')}.",
          "- 기존 파일 보호: " + protection,
          "", "## 10. 동결한 상수와 원본 대조표", "", "### 동결한 chunk 읽기 시간 상수", "",
          "| 모델 | K chunk (ms) | V chunk (ms) | 표본 수 |", "|---|---|---|---|"]
    for m in MODELS:
        c = constants["models"][m]
        L.append(f"| {MODEL_LABEL[m]} | {1e3 * c['k_chunk_s']:.5f} | {1e3 * c['v_chunk_s']:.5f} | {c['samples']:,} |")
    L += ["", "측정 방법은 계약 4절, 측정 결과 전체는 `results/" + top.name + "/calibration_<model>.json`. 공개 코드의 디스크 상수는 0.40818 ms다.", "",
          "### 원본 대조표 (0단계 최종본)", "", "아래는 `docs/hyperinfer_port_contract.md`의 내용이다.", ""]
    table = (top / "PORT_TABLE.md").read_text().splitlines()
    L += ["> " + line if line.strip() and not line.startswith("|") else line for line in table[1:]]
    L += ["", "## 11. 코드, 지시와 다르게 한 것, 한계", "", CODE_MD, "### 지시와 다르게 했거나 지시에 없던 것", "",
          "계약 9절의 10개 항목이 전부다(`docs/mt_gqa_hyperinfer_contract.md`). 요약:", ""]
    contract = (top / "CONTRACT.md").read_text()
    sec = contract[contract.index("## 9. "):]
    L += [line for line in sec.splitlines()[2:] if line.strip()]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "report.md").write_text("\n".join(L) + "\n")

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o

    slim = {m: {k: v for k, v in r.items() if k not in ("names",)} for m, r in reports.items()}
    (out_root / "report.json").write_text(json.dumps(clean({"run": top.name, "models": slim, "errors": errors}), ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"report": str((out_root / "report.md").relative_to(ROOT)), "models": list(reports), "errors": list(errors)}))
    sys.exit(0)


if __name__ == "__main__":
    main()

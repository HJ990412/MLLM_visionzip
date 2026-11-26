#!/usr/bin/env python3
"""Report of the MT-GQA HyperInfer v2 run (contract docs/mt_gqa_hyperinfer_v2_contract.md, section 6).

Reads committed rows only: HyperInfer from this run; ReComp / FullLoad / IMPRESS / Ours from the
earlier main runs; the earlier HyperInfer run (public-code time budget) for the comparison of item 3.
Writes results/<run>/report.md and report.json.  Not part of the measurement freeze.  Works on a run
that is still going or that stopped at a gate: it reports what exists.

  python scripts/230_report_mt_gqa_hyperinfer_v2.py --run runs/mt_gqa_hyperinfer_v2_<UTC>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mmimpress.mtvqa25k.adapter import read_rows  # noqa: E402


def load_script(name, filename):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R1 = load_script("_mt_gqa_hyperinfer_v1_report", "227_report_mt_gqa_hyperinfer.py")      # row readers and bootstrap of the earlier report
cells, arm_row, per_image, boot_diff, boot_ratio, fx, mean = R1.cells, R1.arm_row, R1.per_image, R1.boot_diff, R1.boot_ratio, R1.fx, R1.mean
MODELS, MODEL_DIR, MODEL_LABEL, REF_RUN, REF_DATE = R1.MODELS, R1.MODEL_DIR, R1.MODEL_LABEL, R1.REF_RUN, R1.REF_DATE
FULL, IMPRESS, HYPER = R1.FULL, R1.IMPRESS, R1.HYPER
V1_RUN = ROOT / "runs/mt_gqa_hyperinfer_20261124T171039Z"
V1_DATE = "2026-11-24"
MB = 1e6
SCALARS = ("window_ms", "sync_wait_ms", "probe_wait_ms", "inflight_wait_ms", "probe_sync_ms", "select_ms", "sync_read_ms", "copy_ms", "gpu_load_ms",
           "critical_io_ms", "total_io_ms", "io_ratio", "late_starts", "jobs", "jobs_stopped", "jobs_exhausted", "reads_done_in_window",
           "reads_done_after_stop", "prefetched_tokens", "hit_tokens", "used_tokens_after_layer0", "prefetched_bytes", "wasted_bytes", "sync_bytes",
           "predicted_chunks", "chunks_per_layer")
LAYER_COLS = ("fallback", "window_ms", "sync_wait_ms", "probe_wait_ms", "inflight_wait_ms", "probe_sync_ms", "select_ms", "sync_read_ms", "gpu_load_ms",
              "predicted_chunks", "k_done", "v_done", "prefetched_chunks", "reads_done_in_window", "chunks_done_in_window", "reads_done_after_stop",
              "needed_chunks", "k_missing", "v_missing", "k_wasted", "v_wasted", "prefetched_tokens", "used_tokens", "hit_tokens", "worker_probe_ms",
              "probe_ready", "worker_done", "jaccard_next", "jaccard_next_both_consensus")


# ------------------------------------------------------------------ reading
def _arr(values, n):
    out = np.full(n, np.nan, dtype=np.float32)
    for i, v in enumerate(values):
        if v is not None:
            out[i] = float(v)
    return out


def _slim(r):
    res = r["result"]
    out = {"score": r["score"], "ttft": r["ttft_ms"], "bytes": r["ssd_read_bytes"], "reads": r["ssd_preads"], "tokens": r["generated_token_ids"],
           "blocked": r.get("read_blocked_ms"), "k": r["k_read_bytes"], "v": r["v_read_bytes"], "probe": r["probe_read_bytes"],
           "io_ms": (res.get("io") or {}).get("ms"), "hook_ms": res.get("hook_ms"), "fallback_rate": res.get("fallback_rate"),
           "mean_jaccard": res.get("mean_jaccard"), "k_frac": res.get("touched_k_chunk_fraction"), "v_frac": res.get("touched_v_chunk_fraction"),
           "probe_wait_ms": res.get("prefetch_wait_ms"), "pid": r.get("process_id"), "at": r.get("completed_at_unix")}
    h = res.get("hyperinfer")
    if h and h.get("truncation") == "reactive":
        L = len(h["layers"]["fallback"])
        out["h"] = {k: h.get(k) for k in SCALARS}
        out["hl"] = {k: _arr(h["layers"][k], L) for k in LAYER_COLS}
        out["k_ms"] = np.asarray(h["k_read_ms"], dtype=np.float32)
        out["v_ms"] = np.asarray(h["v_read_ms"], dtype=np.float32)
        out["p_ms"] = np.asarray(h["probe_read_ms"], dtype=np.float32)
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
def dist(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not x.size:
        return None
    p = np.percentile(x, [10, 50, 90, 99])
    return {"n": int(x.size), "mean": float(x.mean()), "p10": float(p[0]), "p50": float(p[1]), "p90": float(p[2]), "p99": float(p[3])}


def prefetch_summary(fu):
    """Request means, distributions and per-layer means over HyperInfer follow-up cells of this run."""
    hs = [c["h"] for c in fu]
    lay = {k: np.stack([c["hl"][k] for c in fu]).astype(np.float64) if len({len(c["hl"][k]) for c in fu}) == 1 else None for k in LAYER_COLS}
    L = lay["fallback"].shape[1]
    tot = lambda k: float(sum(h[k] or 0 for h in hs))      # noqa: E731
    n = len(fu)
    jac, both = lay["jaccard_next"], lay["jaccard_next_both_consensus"] > 0.5
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = {
            "requests": n,
            "coverage": tot("hit_tokens") / tot("used_tokens_after_layer0") if tot("used_tokens_after_layer0") else None,
            "precision": tot("hit_tokens") / tot("prefetched_tokens") if tot("prefetched_tokens") else None,
            "jaccard_consensus_pairs": float(np.nanmean(jac[both])) if both.any() else None,
            "jaccard_all_pairs": float(np.nanmean(jac)), "consensus_pair_share": float(np.nanmean(both[:, :L - 1])),
            "prefetched_mb_per_followup": tot("prefetched_bytes") / n / MB, "wasted_mb_per_followup": tot("wasted_bytes") / n / MB,
            "sync_mb_per_followup": tot("sync_bytes") / n / MB,
            "predicted_chunks_per_layer": tot("predicted_chunks") / max(1.0, tot("jobs")),
            "reads_per_job": float((np.nansum(lay["k_done"]) + np.nansum(lay["v_done"])) / max(1.0, tot("jobs"))),
            "reads_in_window_per_job": tot("reads_done_in_window") / max(1.0, tot("jobs")),
            "reads_after_stop_per_job": tot("reads_done_after_stop") / max(1.0, tot("jobs")),
            "chunks_in_window_per_job": float(np.nansum(lay["chunks_done_in_window"]) / max(1.0, tot("jobs"))),
            "chunks_prefetched_per_job": float(np.nansum(lay["prefetched_chunks"]) / max(1.0, tot("jobs"))),
            "jobs_exhausted_share": tot("jobs_exhausted") / max(1.0, tot("jobs")),
            "late_starts": int(tot("late_starts")), "jobs": int(tot("jobs")),
            "probe_ready_share": float(np.nanmean(lay["probe_ready"][:, 1:])), "worker_done_share": float(np.nanmean(lay["worker_done"][:, 1:])),
            "io_ratio_pooled": tot("critical_io_ms") / tot("total_io_ms") if tot("total_io_ms") else None,
            "io_ratio_request_mean": mean([h["io_ratio"] for h in hs]),
            "dist": {"window_ms": dist(lay["window_ms"][:, 1:]), "sync_wait_ms": dist(lay["sync_wait_ms"][:, 1:]),
                     "probe_read_ms": dist(np.concatenate([c["p_ms"] for c in fu])),
                     "k_read_ms": dist(np.concatenate([c["k_ms"] for c in fu])), "v_read_ms": dist(np.concatenate([c["v_ms"] for c in fu]))},
        }
        for k in ("window_ms", "sync_wait_ms", "probe_wait_ms", "inflight_wait_ms", "probe_sync_ms", "select_ms", "sync_read_ms", "copy_ms", "gpu_load_ms",
                  "critical_io_ms", "total_io_ms"):
            out[k] = mean([h[k] for h in hs])
        per_layer = []
        for li in range(L):
            pt, ht, ut = np.nansum(lay["prefetched_tokens"][:, li]), np.nansum(lay["hit_tokens"][:, li]), np.nansum(lay["used_tokens"][:, li])
            col = lambda k: float(np.nanmean(lay[k][:, li]))      # noqa: E731
            per_layer.append({
                "layer": li, "fallback_rate": col("fallback"), "window_ms": col("window_ms") if li else None,
                "worker_probe_ms": col("worker_probe_ms") if li else None, "predicted_chunks": col("predicted_chunks"),
                "reads_in_window": col("reads_done_in_window"), "chunks_in_window": col("chunks_done_in_window"),
                "reads_after_stop": col("reads_done_after_stop"), "k_done": col("k_done"), "v_done": col("v_done"),
                "needed_chunks": col("needed_chunks"), "k_missing": col("k_missing"), "v_missing": col("v_missing"),
                "wasted_reads": col("k_wasted") + col("v_wasted"),
                "precision": float(ht / pt) if pt else None, "coverage": float(ht / ut) if (ut and li) else None,
                "probe_wait_ms": col("probe_wait_ms"), "inflight_wait_ms": col("inflight_wait_ms"), "select_ms": col("select_ms"),
                "sync_read_ms": col("sync_read_ms") + col("probe_sync_ms"), "gpu_load_ms": col("gpu_load_ms"),
                "jaccard_next": float(np.nanmean(jac[:, li])) if li < L - 1 else None})
    out["per_layer"] = [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()} for row in per_layer]
    return out


def v1_summary(fu):
    hs = [c["h"] for c in fu]
    tot = lambda k: float(sum(h[k] or 0 for h in hs))      # noqa: E731
    return {"ttft_ms": mean([c["ttft"] for c in fu]), "prefetch_wait_ms": mean([h["prefetch_wait_ms"] for h in hs]),
            "coverage": tot("hit_tokens") / tot("used_tokens_after_layer0"), "precision": tot("hit_tokens") / tot("prefetched_tokens") if tot("prefetched_tokens") else None,
            "sync_read_ms": mean([h["sync_read_ms"] for h in hs]), "select_ms": mean([h["select_ms"] for h in hs]),
            "prefetched_mb": tot("prefetched_bytes") / len(fu) / MB, "wasted_mb": tot("wasted_bytes") / len(fu) / MB,
            "read_wait_ms": mean([c["blocked"] for c in fu]), "reads": mean([c["reads"] for c in fu]), "bytes_mb": mean([c["bytes"] for c in fu]) / MB}


# ---------------------------------------------------------------- per model
def model_report(top, model, config, manifest):
    sub = top / MODEL_DIR[model]
    ref_cfg = json.loads((REF_RUN[model] / "config.json").read_text())
    recomp, full, imp, ours = ref_cfg["main_methods"]
    names = {recomp: "ReComp", full: "FullLoad", imp: "IMPRESS", ours: "Ours", HYPER: "HyperInfer"}
    out = {"model": model, "label": MODEL_LABEL[model], "names": names}
    gates = {}
    for name in ("tests", f"{model}_gate", f"{model}_dev", f"{model}_admission"):
        p = top / "gates" / f"{name}.json"
        gates[name] = json.loads(p.read_text()) if p.exists() else None
    out["gates"] = gates
    v1_sub = V1_RUN / MODEL_DIR[model]
    main = collect(sub, "main", manifest["phase_images"]["main"], None) if (sub / "main").exists() else []
    out["main_images"], out["main_complete"] = len(main), (sub / "main_execution_complete.json").exists()
    if main:
        images = [r["image"] for r in main]
        ref = collect(REF_RUN[model], "main", images, None)
        old = R1.collect(v1_sub, "main", images, {HYPER})
        assert [r["image"] for r in ref] == images and [r["image"] for r in old] == images
        many = len(main) > 1
        out["arms"] = [arm_row(ref, m, names[m], REF_DATE[model]) for m in (recomp, full, imp)] + \
                      [arm_row(main, HYPER, "HyperInfer", time.strftime("%Y-%m-%d", time.gmtime(min(c["at"] for c in cells(main, HYPER)))))] + \
                      [arm_row(ref, ours, "Ours", REF_DATE[model])]
        out["arm_v1"] = arm_row(old, HYPER, "HyperInfer (이전 실행)", V1_DATE)
        pairs = [("Ours 대 HyperInfer", "Ours − HyperInfer", "Ours / HyperInfer", ref, main, ours, HYPER),
                 ("HyperInfer 대 IMPRESS", "HyperInfer − IMPRESS", "HyperInfer / IMPRESS", main, ref, HYPER, imp),
                 ("HyperInfer 대 FullLoad", "HyperInfer − FullLoad", "HyperInfer / FullLoad", main, ref, HYPER, full),
                 ("HyperInfer 대 ReComp", "HyperInfer − ReComp", "HyperInfer / ReComp", main, ref, HYPER, recomp)]
        out["paired"] = []
        for label, dl, rl, ra, rb, a, b in pairs:
            out["paired"].append({"comparison": label, "diff_label": dl, "ratio_label": rl,
                                  "em": boot_diff(per_image(ra, rb, a, b, "score", 100.0)) if many else None,
                                  "ttft": boot_diff(per_image(ra, rb, a, b, "ttft")) if many else None,
                                  "ratio": boot_ratio(per_image(ra, rb, a, b, "ttft")) if many else None})
        bad, n = [], 0
        for x, y in zip(main, ref):
            for d in x["dialogs"]:
                for t in (1, 2, 3):
                    n += 1
                    if x["by"][(HYPER, d, t)]["tokens"] != y["by"][(imp, d, t)]["tokens"]:
                        bad.append({"image": x["image"], "dialog_id": d, "turn_id": t})
        out["identity"] = {"requests": n, "equal": n - len(bad), "mismatches": bad[:50], "mismatch_count": len(bad)}
        fu, imp_fu, old_fu = cells(main, HYPER), cells(ref, imp), cells(old, HYPER)
        out["prefetch"] = prefetch_summary(fu)
        out["bytes_vs_impress"] = {"hyperinfer_mb": np.mean([c["bytes"] for c in fu]) / MB, "impress_mb": np.mean([c["bytes"] for c in imp_fu]) / MB,
                                   "ratio": float(np.sum([c["bytes"] for c in fu]) / np.sum([c["bytes"] for c in imp_fu]))}
        # item 3: against the earlier HyperInfer run, same requests
        v1 = v1_summary(old_fu)
        pf = out["prefetch"]
        out["vs_v1"] = {
            "v1": v1,
            "v2": {"ttft_ms": mean([c["ttft"] for c in fu]), "prefetch_wait_ms": pf["probe_wait_ms"] + pf["inflight_wait_ms"], "coverage": pf["coverage"],
                   "precision": pf["precision"], "sync_read_ms": pf["sync_read_ms"], "select_ms": pf["select_ms"],
                   "prefetched_mb": pf["prefetched_mb_per_followup"], "wasted_mb": pf["wasted_mb_per_followup"],
                   "read_wait_ms": mean([c["blocked"] for c in fu]), "reads": mean([c["reads"] for c in fu]), "bytes_mb": mean([c["bytes"] for c in fu]) / MB},
            "ttft_diff": boot_diff(per_image(main, old, HYPER, HYPER, "ttft")) if many else None,
            "ttft_ratio": boot_ratio(per_image(main, old, HYPER, HYPER, "ttft")) if many else None,
            "v1_over_impress": boot_ratio(per_image(old, ref, HYPER, imp, "ttft")) if many else None,
            "tokens_equal": sum(x["by"][(HYPER, d, t)]["tokens"] == y["by"][(HYPER, d, t)]["tokens"] for x, y in zip(main, old) for d in x["dialogs"] for t in (1, 2, 3)),
            "requests": n}
        out["decomposition"] = {
            "HyperInfer": {"ttft_ms": mean([c["ttft"] for c in fu]), "read_wait_ms": mean([c["blocked"] for c in fu]), "hook_ms": mean([c["hook_ms"] for c in fu]),
                           "probe_wait_ms": pf["probe_wait_ms"], "inflight_wait_ms": pf["inflight_wait_ms"], "probe_sync_ms": pf["probe_sync_ms"],
                           "select_ms": pf["select_ms"], "sync_read_ms": pf["sync_read_ms"], "copy_ms": pf["copy_ms"], "gpu_load_ms": pf["gpu_load_ms"],
                           "sync_wait_ms": pf["sync_wait_ms"]},
            "IMPRESS": {"ttft_ms": mean([c["ttft"] for c in imp_fu]), "read_wait_ms": mean([c["blocked"] for c in imp_fu]), "hook_ms": mean([c["hook_ms"] for c in imp_fu]),
                        "probe_wait_ms": mean([c["probe_wait_ms"] for c in imp_fu])}}
        ps = [r["persist"] for r in main if r.get("persist")]
        out["t1_store"] = {"images": len(ps), "persist_ms_mean": mean([p["ms"] for p in ps]), "write_mb_mean": mean([p["write_bytes"] for p in ps]) / MB if ps else None,
                           "store_mb_mean": mean([p["file_bytes"] for p in ps]) / MB if ps else None, "write_gb_total": sum(p["write_bytes"] or 0 for p in ps) / 1e9}
    # DEV: FullLoad, IMPRESS and HyperInfer measured together
    if (sub / "dev").exists():
        dev = collect(sub, "dev", manifest["phase_images"]["dev"], {FULL, IMPRESS, HYPER})
        if dev:
            d_imp, d_hyp, d_full = cells(dev, IMPRESS), cells(dev, HYPER), cells(dev, FULL)
            ratio = lambda cs: float(sum(c["blocked"] for c in cs) / sum(c["io_ms"] for c in cs))      # noqa: E731

            def row(cs):
                t = np.array([c["ttft"] for c in cs])
                return {"em": 100 * mean([c["score"] for c in cs]), "ttft_ms": float(t.mean()), "ttft_p50_ms": float(np.median(t)), "ttft_p95_ms": float(np.percentile(t, 95)),
                        "mb": mean([c["bytes"] for c in cs]) / MB, "reads": mean([c["reads"] for c in cs]), "read_wait_ms": mean([c["blocked"] for c in cs]),
                        "io_ms": mean([c["io_ms"] for c in cs])}

            summ = prefetch_summary(d_hyp)
            out["dev"] = {"images": len(dev), "followups": len(d_imp), "fullload": row(d_full), "impress": dict(row(d_imp), io_ratio=ratio(d_imp)),
                          "hyperinfer": dict(row(d_hyp), io_ratio=summ["io_ratio_pooled"], summary={k: v for k, v in summ.items() if k != "per_layer"}),
                          "ratio": boot_ratio(per_image(dev, dev, HYPER, IMPRESS, "ttft")) if len(dev) > 1 else
                                   {"ratio": mean([c["ttft"] for c in d_hyp]) / mean([c["ttft"] for c in d_imp])},
                          "diff": boot_diff(per_image(dev, dev, HYPER, IMPRESS, "ttft")) if len(dev) > 1 else None}
            v1_dev = R1.collect(v1_sub, "dev", [r["image"] for r in dev], {IMPRESS, HYPER})
            if len(v1_dev) == len(dev):
                out["dev"]["v1_ratio"] = (boot_ratio(per_image(v1_dev, v1_dev, HYPER, IMPRESS, "ttft")) if len(dev) > 1 else
                                          {"ratio": mean([c["ttft"] for c in cells(v1_dev, HYPER)]) / mean([c["ttft"] for c in cells(v1_dev, IMPRESS)])})
    ex, sync_cost = {}, []
    for ph in ("gate", "dev", "main"):
        starts = sorted(sub.glob(f"{ph}_start_*.json"))
        done = sub / f"{ph}_execution_complete.json"
        if starts:
            docs = [json.loads(s.read_text()) for s in starts]
            end = json.loads(done.read_text())["finished_unix"] if done.exists() else None
            ex[ph] = {"process_starts": len(starts), "complete": done.exists(), "wall_hours": (end - docs[0]["started_unix"]) / 3600 if end else None}
            sync_cost += [dict(d["idle_synchronize_cost"], phase=ph) for d in docs if d.get("idle_synchronize_cost")]
    out["execution"], out["idle_sync_cost"] = ex, sync_cost
    log = sub / "resume_log.jsonl"
    out["resume_events"] = [json.loads(l) for l in log.read_text().splitlines() if l.strip()] if log.exists() else []
    return out


# ----------------------------------------------------------------- markdown
GATE_ROWS = [("1. 출력 동일성 (HyperInfer ≡ IMPRESS)", "gate1_output_equals_impress"), ("2. prefetch 끔 ≡ IMPRESS", "gate2_prefetch_off_equals_impress"),
             ("3. 읽지 않은 KV를 쓰지 않음", "gate3_unread_rows_not_used"), ("4. 두 번 실행", "gate4_repeat"),
             ("5. 멈춤 확인 (계산 종료 뒤 시작된 읽기 0건)", "gate5_no_read_started_after_stop")]


def gate_md(r):
    m = r["model"]
    g4, gd = r["gates"].get(f"{m}_gate"), r["gates"].get(f"{m}_dev")
    lines = ["| gate | 4장 | DEV 40장 |", "|---|---|---|"]

    def cell(doc, key):
        if not doc:
            return "미실행"
        if "gates" not in doc:
            return "단계 미완료"
        g = doc["gates"].get(key)
        if g is None:
            return "해당 없음"
        if key.startswith("gate5"):
            jobs = sum(a["layers_checked"] for a in g["arms"].values())
            late = sum(a["reads_started_after_stop"] for a in g["arms"].values())
            return f"{g['status']} (layer {jobs:,}개 중 위반 {late:,}건)"
        n = g.get("requests")
        bad = g.get("mismatches", g.get("guard_failures", 0)) + (g.get("against_impress", {}).get("mismatches", 0) if key.startswith("gate3") else 0)
        return f"{g['status']} ({n - bad:,} / {n:,})" if n is not None else g["status"]

    for label, key in GATE_ROWS:
        lines.append(f"| {label} | {cell(g4, key)} | {cell(gd, key)} |")
    adm = r["gates"].get(f"{m}_admission")
    lines += ["", f"admission: **{adm['status'] if adm else '미판정'}**" + (f" (실패: {', '.join(adm['failed'])})" if adm and adm.get("failed") else "")]
    for name, doc in (("4장", g4), ("DEV 40장", gd)):
        if not doc or "gates" not in doc:
            if doc and doc.get("failed"):
                lines += ["", f"{name} 단계가 완료되지 않았다: `{json.dumps(doc.get('execution'), ensure_ascii=False)[:600]}`"]
            continue
        g5, g4r = doc["gates"]["gate5_no_read_started_after_stop"], doc["gates"]["gate4_repeat"]
        a = g5["arms"][HYPER]
        pb = g4r.get("prefetched_bytes") or {}
        lines += ["", f"- {name}, gate 5 (HyperInfer arm): prefetch 작업 {a['jobs']:,}개(layer), prefetch 읽기 {a['prefetch_reads']:,}건, 그중 멈춤 뒤에 시작된 읽기 "
                      f"{a['reads_started_after_stop']}건, 멈춤 시점에 진행 중이던 읽기 {a['reads_in_flight_at_stop']:,}건. 읽기 호출이 나간 시각 − 멈춤 시각의 최댓값 "
                      f"{fx(a['latest_read_call_relative_to_stop_ms'], 3)} ms.",
                  f"- {name}, gate 4: 두 실행에서 prefetch된 양이 달랐던 요청 {pb.get('requests_with_a_different_amount')} / {pb.get('requests')} "
                  f"(평균 {fx((pb.get('first_mean') or 0) / MB, 1)} MB 대 {fx((pb.get('second_mean') or 0) / MB, 1)} MB). 생성 토큰은 위 표대로다."]
        if doc.get("tokens_against_recorded_impress"):
            t = doc["tokens_against_recorded_impress"]
            lines.append(f"- {name}, 참고: HyperInfer의 생성 토큰이 기존 본실험의 IMPRESS 기록과 일치한 요청 {t['requests'] - t['mismatches']:,} / {t['requests']:,}.")
        for key in doc.get("failed") or []:
            g = doc["gates"].get(key, {})
            first = (g.get("first_mismatches") or g.get("first_guard_failures") or g.get("first_violations") or (g.get("against_impress") or {}).get("first_mismatches") or [])[:5]
            lines += ["", f"실패한 gate `{key}` ({name}). 기준: {g.get('criterion')}. 처음 몇 건:", "```json", json.dumps(first, ensure_ascii=False, indent=1)[:3000], "```"]
    return "\n".join(lines)


def ci(d, digits=1, sign=True):
    if not d:
        return "—"
    f = f"{{:{'+' if sign else ''}.{digits}f}}"
    return f"[{f.format(d['ci_low'])}, {f.format(d['ci_high'])}]"


def dist_row(label, d, digits=3):
    if not d:
        return f"| {label} | — | — | — | — | — | — |"
    return f"| {label} | {d['n']:,} | {d['mean']:.{digits}f} | {d['p10']:.{digits}f} | {d['p50']:.{digits}f} | {d['p90']:.{digits}f} | {d['p99']:.{digits}f} |"


def model_md(r):
    m = r["model"]
    L = [f"## {r['label']}", "", "### 구현 검증 (gate 1–5)", "", gate_md(r), ""]
    dev = r.get("dev")
    if not r.get("arms"):
        L += ["본실험 행이 없다(시작하지 않았거나 아직 커밋된 이미지가 없다).", ""]
    else:
        partial = "" if r["main_complete"] else f" **진행 중 또는 중단: {r['main_images']} / 398 이미지 기준의 중간 값이다.**"
        L += [f"### 1. 결과 표 ({r['main_images']} images){partial}", "",
              "| 방법 | 실행일 | followup EM | TTFT 평균 (ms) | p50 | p95 | SSD MB/followup | 읽기 호출 수/followup |", "|---|---|---|---|---|---|---|---|"]
        for a in r["arms"]:
            L.append(f"| {a['method']} | {a['run_date']} | {a['em_followup']:.2f} | {a['ttft_mean_ms']:.1f} | {a['ttft_p50_ms']:.1f} | {a['ttft_p95_ms']:.1f} | "
                     f"{a['ssd_mb_per_followup']:.1f} | {a['read_calls_per_followup']:.1f} |")
        hy = next(a for a in r["arms"] if a["method"] == "HyperInfer")
        L += ["", f"HyperInfer만 이번 실행의 값이고 나머지는 기존 본실험의 같은 이미지·대화·턴 기록이다. TTFT와 읽기량은 재사용 턴(T2–T3) 기준이다. "
                  f"HyperInfer의 턴별 EM은 T1 {hy['em_T1']:.2f} / T2 {hy['em_T2']:.2f} / T3 {hy['em_T3']:.2f}이다.", "",
              "### 2. 짝지은 비교와 비율 (이미지 단위 paired bootstrap 10,000회, seed 1234, 95% CI)", "",
              "| 비교 | EM 차이 (pp) | 95% CI | TTFT 차이 (ms) | 95% CI | TTFT 비율 | 95% CI |", "|---|---|---|---|---|---|---|"]
        for p in r["paired"]:
            em, tt, ra = p["em"], p["ttft"], p["ratio"]
            L.append(f"| {p['comparison']} ({p['diff_label']}, {p['ratio_label']}) | {fx(em and em['diff'], 2) if not em else format(em['diff'], '+.2f')} | {ci(em, 2)} | "
                     f"{'—' if not tt else format(tt['diff'], '+.1f')} | {ci(tt, 1)} | {'—' if not ra else format(ra['ratio'], '.3f')} | {ci(ra, 3, sign=False)} |")
        L += ["", "위 표의 TTFT 차이와 비율은 서로 다른 실행에서 잰 값끼리의 비교다(HyperInfer는 이번 실행, 상대는 기존 본실험)."]
        if dev:
            dr = dev["ratio"]
            L += ["", f"**같은 실행에서 잰 HyperInfer / IMPRESS TTFT 비율 (DEV {dev['images']}장, followup {dev['followups']:,}요청, arm 순서 회전): "
                      f"{dr['ratio']:.3f}** {ci(dr, 3, sign=False) if 'ci_low' in dr else ''}"
                      + (f", 차이 {dev['diff']['diff']:+.1f} ms {ci(dev['diff'], 1)}." if dev.get("diff") else "."), "",
                  "| 방법 (DEV, 같은 실행) | followup EM | TTFT 평균 (ms) | p50 | p95 | SSD MB/followup | 읽기 호출 수 | 읽기 대기 (ms) | 전체 pread 시간 (ms) | I/O ratio |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for name, key in (("FullLoad", "fullload"), ("IMPRESS", "impress"), ("HyperInfer", "hyperinfer")):
                d = dev[key]
                L.append(f"| {name} | {d['em']:.2f} | {d['ttft_ms']:.1f} | {d['ttft_p50_ms']:.1f} | {d['ttft_p95_ms']:.1f} | {d['mb']:.1f} | {d['reads']:.1f} | "
                         f"{fx(d['read_wait_ms'], 1)} | {fx(d['io_ms'], 1)} | {fx(d.get('io_ratio'), 3)} |")
            gd = r["gates"].get(f"{m}_dev") or {}
            cont = (gd.get("gate7_dev_table") or {}).get("continuity_with_earlier_main_run")
            if cont:
                parts = [f"{name} {cont[k]['followup']['this_run_mean_ms']:.1f} ms 대 {cont[k]['followup']['earlier_run_mean_ms']:.1f} ms "
                         f"(×{cont[k]['followup']['ratio_of_means']:.3f})" for name, k in (("FullLoad", FULL), ("IMPRESS", IMPRESS))]
                L += ["", "기존 본실험의 같은 요청과의 TTFT 연속성(이번 DEV 대 기존): " + ", ".join(parts) + ". 이번 DEV는 6개 arm을, 기존 본실험은 4개 arm을 회전시켰다."]
        v = r["vs_v1"]
        a, b = v["v1"], v["v2"]
        o = r["arm_v1"]
        hi_imp = next(p for p in r["paired"] if p["comparison"] == "HyperInfer 대 IMPRESS")["ratio"]
        L += ["", "### 3. 이전 실행과의 비교 (같은 요청끼리, 본실험)", "",
              f"이전 실행은 `{V1_RUN.name}` ({V1_DATE}, 공개 코드의 시간 예산으로 미리 자르고 prefetch 전체를 기다림)이다.", "",
              "| 항목 | 이전 실행 | 이번 실행 |", "|---|---|---|",
              f"| TTFT 평균 (ms) | {a['ttft_ms']:.1f} | {b['ttft_ms']:.1f} |",
              f"| TTFT p50 / p95 (ms) | {o['ttft_p50_ms']:.1f} / {o['ttft_p95_ms']:.1f} | {hy['ttft_p50_ms']:.1f} / {hy['ttft_p95_ms']:.1f} |",
              f"| HyperInfer / IMPRESS (본실험, 다른 실행끼리) | {fx(v['v1_over_impress'] and v['v1_over_impress']['ratio'], 3)} | {fx(hi_imp and hi_imp['ratio'], 3)} |",
              f"| HyperInfer / IMPRESS (DEV, 같은 실행) | {fx(((dev or {}).get('v1_ratio') or {}).get('ratio'), 3)} | {fx(((dev or {}).get('ratio') or {}).get('ratio'), 3)} |",
              f"| prefetch 대기 (ms/요청) | {a['prefetch_wait_ms']:.1f} | {b['prefetch_wait_ms']:.2f} |",
              f"| 사용 집합 중 prefetch로 채워진 비율 (layer 1 이후) | {100 * a['coverage']:.1f}% | {100 * b['coverage']:.1f}% |",
              f"| prefetch된 토큰 중 실제로 쓴 비율 | {fx(a['precision'] and 100 * a['precision'], 1, '%')} | {fx(b['precision'] and 100 * b['precision'], 1, '%')} |",
              f"| prefetch로 읽은 양 (MB/요청) | {a['prefetched_mb']:.1f} | {b['prefetched_mb']:.1f} |",
              f"| 빠진 KV 동기 읽기 (ms/요청) | {a['sync_read_ms']:.1f} | {b['sync_read_ms']:.1f} |",
              f"| 요청 thread의 읽기 대기 합계 (ms/요청) | {fx(a['read_wait_ms'], 1)} | {fx(b['read_wait_ms'], 1)} |",
              f"| 읽기 호출 수 / 총 읽기량 (MB) | {a['reads']:.1f} / {a['bytes_mb']:.1f} | {b['reads']:.1f} / {b['bytes_mb']:.1f} |", "",
              f"TTFT 차이(이번 − 이전): {'—' if not v['ttft_diff'] else format(v['ttft_diff']['diff'], '+.1f')} ms {ci(v['ttft_diff'], 1)}, "
              f"비율 {'—' if not v['ttft_ratio'] else format(v['ttft_ratio']['ratio'], '.3f')} {ci(v['ttft_ratio'], 3, sign=False)}. "
              f"서로 다른 날의 실행이라 실행 간 변동이 섞여 있다. 생성 토큰이 이전 실행과 같은 요청: {v['tokens_equal']:,} / {v['requests']:,}.",
              "이전 실행의 prefetch 대기는 예측 chunk를 다 읽을 때까지 기다린 시간이고, 이번 실행의 것은 probe key 대기와 진행 중이던 읽기의 마무리 대기의 합이다."]
        pf = r["prefetch"]
        d = pf["dist"]
        L += ["", "### 4. 계산 시간 창과 읽기 시간의 실측 분포 (본실험, 재사용 턴)", "",
              "| 항목 (ms) | 표본 수 | 평균 | p10 | p50 | p90 | p99 |", "|---|---|---|---|---|---|---|",
              dist_row("layer당 GPU 계산 시간 창 (prefetch 제출 ~ 다음 layer에서 GPU 동기화 완료)", d["window_ms"]),
              dist_row("probe key 읽기 (layer당 한 번, worker)", d["probe_read_ms"]),
              dist_row("K chunk 읽기 하나 (worker)", d["k_read_ms"]),
              dist_row("V chunk 읽기 하나 (worker)", d["v_read_ms"]),
              dist_row("요청 thread가 GPU 동기화에서 막힌 시간 (layer당)", d["sync_wait_ms"]), "",
              f"- 한 layer의 계산 시간 안에 **읽기를 마친** chunk 읽기: 평균 {pf['reads_in_window_per_job']:.3f}건(K chunk 또는 V chunk 하나가 1건), "
              f"K와 V가 모두 끝난 chunk로는 {pf['chunks_in_window_per_job']:.3f}개. 멈춤 시점에 진행 중이어서 그 뒤에 끝난 읽기: layer당 {pf['reads_after_stop_per_job']:.3f}건.",
              f"- 합쳐서 layer당 prefetch 읽기 {pf['reads_per_job']:.3f}건, K·V가 모두 prefetch된 chunk {pf['chunks_prefetched_per_job']:.3f}개. "
              f"예측 chunk는 layer당 평균 {pf['predicted_chunks_per_layer']:.1f}개였다. 예측 chunk를 전부 읽고 끝난 작업의 비율 {100 * pf['jobs_exhausted_share']:.2f}%.",
              f"- 다음 layer가 probe key를 필요로 할 때 이미 읽혀 있던 비율 {100 * pf['probe_ready_share']:.1f}%, 첫 KV 읽기 직전에 worker가 이미 끝나 있던 비율 {100 * pf['worker_done_share']:.1f}%."]
        sc = r.get("idle_sync_cost") or []
        if sc:
            L.append(f"- **GPU 계산 완료의 판단:** layer l+1의 hook 첫머리에서 `torch.cuda.synchronize()`가 돌아온 시점을 layer l의 완료로 보고, 곧바로 lock 안에서 멈춤을 표시한다. "
                     f"비용: 할 일이 없을 때 동기화 호출 한 번은 평균 {1e6 * np.mean([s['mean_s'] for s in sc]):.1f} µs(p99 {1e6 * np.max([s['p99_s'] for s in sc]):.1f} µs, "
                     f"단계 시작 때 2,000번 측정). 요청 thread가 동기화에서 실제로 막힌 시간은 layer당 평균 {d['sync_wait_ms']['mean']:.3f} ms, 요청당 {pf['sync_wait_ms']:.1f} ms다. "
                     "이 시간은 직전 layer의 GPU 계산이 끝나기를 기다린 시간이다. 선택이 그 출력을 필요로 하므로 IMPRESS도 선택 도중(probe key를 GPU로 옮기는 지점)에 같은 계산을 기다리지만, "
                     "IMPRESS 쪽의 그 시간은 따로 재지 않았다. 판단 자체에 드는 비용은 동기화 호출 한 번과 lock 한 번이다.")
        L += ["", "layer별 평균:", "",
              "| layer | 계산 시간 창 (ms) | probe key 읽기 (ms) | 예측 chunk | 창 안에 끝난 읽기 | 창 안에 K·V가 다 끝난 chunk | 멈춤 뒤 끝난 읽기 | 필요 chunk | 빠진 K chunk | 빠진 V chunk | 채움 비율 | fallback | 다음 layer와 Jaccard |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        pct = lambda x: "—" if x is None else f"{100 * x:.1f}%"      # noqa: E731
        for row in pf["per_layer"]:
            L.append(f"| {row['layer']} | {fx(row['window_ms'], 2)} | {fx(row['worker_probe_ms'], 2)} | {row['predicted_chunks']:.1f} | {row['reads_in_window']:.2f} | "
                     f"{row['chunks_in_window']:.2f} | {row['reads_after_stop']:.2f} | {row['needed_chunks']:.1f} | {row['k_missing']:.1f} | {row['v_missing']:.1f} | "
                     f"{pct(row['coverage'])} | {100 * row['fallback_rate']:.0f}% | {fx(row['jaccard_next'], 2)} |")
        bv = r["bytes_vs_impress"]
        L += ["", "### 5. prefetch 통계 (본실험, 재사용 턴 요청 평균)", "", "| 항목 | 값 |", "|---|---|",
              f"| 사용 집합 중 prefetch로 채워진 비율 (layer 1 이후) | {pct(pf['coverage'])} |",
              f"| prefetch된 것 중 실제로 쓴 비율 (토큰 기준) | {pct(pf['precision'])} |",
              f"| prefetch로 읽은 양 | {pf['prefetched_mb_per_followup']:.1f} MB/followup |",
              f"| 읽었지만 쓰지 않은 양 | {pf['wasted_mb_per_followup']:.2f} MB/followup |",
              f"| 동기로 읽은 양 (K·V) | {pf['sync_mb_per_followup']:.1f} MB/followup |",
              f"| 총 읽기 바이트, IMPRESS 대비 | {bv['hyperinfer_mb']:.1f} MB 대 {bv['impress_mb']:.1f} MB (×{bv['ratio']:.4f}) |",
              f"| 인접 layer 선택 집합 Jaccard, fallback이 아닌 layer 쌍 | {fx(pf['jaccard_consensus_pairs'], 3)} (전체 쌍의 {100 * pf['consensus_pair_share']:.1f}%) |",
              f"| 인접 layer 선택 집합 Jaccard, 전체 쌍 | {fx(pf['jaccard_all_pairs'], 3)} |", "",
              "사용 집합은 선택된 이미지 토큰 + separator다. 채워진 토큰은 K와 V가 모두 prefetch된 chunk의 토큰으로 센다. Jaccard는 separator를 뺀 이미지 토큰 집합으로 계산했다."]
        dc, di = r["decomposition"]["HyperInfer"], r["decomposition"]["IMPRESS"]
        L += ["", "### 6. 임계 경로에 남은 시간 (본실험, 재사용 턴 요청 평균, ms)", "", "| 항목 | HyperInfer |", "|---|---|",
              f"| probe key 대기 (worker의 probe key 읽기가 끝나기를 기다림) | {dc['probe_wait_ms']:.3f} |",
              f"| 진행 중이던 읽기의 마무리 대기 | {dc['inflight_wait_ms']:.3f} |",
              f"| layer 0의 probe key 읽기 (동기) | {dc['probe_sync_ms']:.2f} |",
              f"| 선택 계산 | {dc['select_ms']:.1f} |",
              f"| 빠진 KV 동기 읽기 | {dc['sync_read_ms']:.1f} |",
              f"| prefetch buffer에서 옮기는 host 복사 | {dc['copy_ms']:.2f} |",
              f"| GPU 적재 (cache 기록 호출) | {dc['gpu_load_ms']:.1f} |",
              f"| 요청 thread의 읽기 대기 합계 | {fx(dc['read_wait_ms'], 1)} |",
              f"| hook 합계 | {fx(dc['hook_ms'], 1)} |",
              f"| TTFT | {dc['ttft_ms']:.1f} |", "",
              f"기존 본실험의 IMPRESS: 읽기 대기 합계 {fx(di['read_wait_ms'], 1)} ms(그중 probe 대기 {fx(di['probe_wait_ms'], 2)} ms), hook 합계 {fx(di['hook_ms'], 1)} ms, TTFT {di['ttft_ms']:.1f} ms. "
              "기존 IMPRESS 행에는 선택 계산과 GPU 적재가 따로 기록돼 있지 않다.", "",
              f"I/O ratio = (probe key 대기 + 진행 중이던 읽기의 마무리 대기 + layer 0의 probe key 읽기 + 빠진 KV 동기 읽기) / (worker 포함 전체 pread 시간): "
              f"**{fx(pf['io_ratio_pooled'], 3)}** (요청별 평균 {fx(pf['io_ratio_request_mean'], 3)})."
              + (f" DEV {dev['images']}장에서 함께 잰 값: HyperInfer {fx(dev['hyperinfer']['io_ratio'], 3)}, IMPRESS {fx(dev['impress']['io_ratio'], 3)} "
                 "(IMPRESS는 요청 thread가 읽기로 막힌 시간 / 전체 pread 시간)." if dev else "")]
        idn = r["identity"]
        L += ["", "### 7. 출력 동일성", "",
              f"본실험 {idn['requests']:,}요청(T1–T3) 중 HyperInfer의 생성 토큰이 기존 IMPRESS 기록과 일치한 요청: **{idn['equal']:,} / {idn['requests']:,}**."]
        if idn["mismatch_count"]:
            L += ["", "불일치(처음 50건):", "```json", json.dumps(idn["mismatches"], ensure_ascii=False)[:4000], "```"]
        t1 = r["t1_store"]
        L += ["", "### 실행 기록", "",
              f"- T1 store 기록(IMPRESS 형식, 이미지당 한 번): 평균 {fx(t1['persist_ms_mean'], 0)} ms, 쓰기 {fx(t1['write_mb_mean'], 0)} MB/이미지, "
              f"본실험 합계 {t1['write_gb_total']:.1f} GB ({t1['images']} images)."]
    for ph, e in (r.get("execution") or {}).items():
        L.append(f"- `{ph}` 단계: 프로세스 시작 {e['process_starts']}회(이어서 실행 {e['process_starts'] - 1}회), "
                 + (f"소요 {e['wall_hours']:.2f}시간." if e["wall_hours"] is not None else "미완료."))
    ev = r.get("resume_events") or []
    redo = [(e.get("phase"), e.get("image_id"), e.get("mode"), e.get("rerun_dialogues") or e.get("restart_dialogues") or []) for e in ev]
    if redo:
        L += ["", "이어서 실행하며 다시 실행한 대화:"] + [f"- {ph} / {img} ({mode}): {', '.join(ds) if ds else '이미지 전체를 처음부터'}" for ph, img, mode, ds in redo]
    else:
        L.append("- 이어서 실행하며 다시 실행한 대화: 없음.")
    return "\n".join(L)


CODE_MD = """### 추가한 코드 (기존 파일은 원본 대조표 문서 외에는 수정하지 않았다)

- `mmimpress/hyperinfer_v2.py`: prefetch를 멈추는 방식을 바꾼 HyperInfer. worker 작업(`PrefetchJob`, `run_prefetch_job`), store 읽기 대리 객체, 선택기, 요청 함수. 예측 chunk 계산, chunk 하나를 읽는 함수, NaN 검사 래퍼는 `mmimpress/hyperinfer.py`(수정하지 않음)에서 불러 쓴다.
- `tests/test_hyperinfer_v2.py`: 12개 테스트. 멈춤 뒤에는 읽기를 시작하지 않고 진행 중이던 읽기는 끝내는지, 멈춤이 probe key 읽기보다 먼저 와도 probe key는 읽는지, 멈춤이 없으면 예측 chunk를 K → V 오름차순으로 전부 읽는지, probe key 읽기가 실패해도 기다리는 쪽이 풀리는지, prefetch를 끄면 IMPRESS와 같은지, prefetch가 0건이든 전부든 K만 읽힌 chunk가 있든 선택·mask·cache 내용이 IMPRESS와 같은지, 바이트·chunk 회계.
- `scripts/228_eval_mt_gqa_hyperinfer_v2.py`: 실행(freeze, gate·dev·main 단계). 이전 실행의 `scripts/225`를 이 프로세스 안에서만 불러 arm 표와 단계 표를 바꿔 쓴다.
- `scripts/229_validate_mt_gqa_hyperinfer_v2.py`: gate 판정. `scripts/230_report_mt_gqa_hyperinfer_v2.py`: 이 보고서. `scripts/run_mt_gqa_hyperinfer_v2.sh`: 드라이버.
- 문서: `docs/mt_gqa_hyperinfer_v2_contract.md`(실행 계약), `docs/hyperinfer_port_contract.md`(원본 대조표, 2026-11-25 개정).

### 한계

- ReComp, FullLoad, IMPRESS, Ours는 다시 돌리지 않았다. 표의 값은 기존 본실험의 것이고, HyperInfer와의 TTFT 차이는 서로 다른 실행 사이의 차이다. 같은 실행에서 함께 잰 값은 DEV 40장의 FullLoad, IMPRESS, HyperInfer뿐이다.
- 이전 실행과의 비교도 서로 다른 날의 실행 사이의 비교다.
- 계산 시간 창은 prefetch 제출 시각부터 다음 layer에서 GPU 동기화가 끝난 시각까지의 실측 시간이다. layer의 연산을 GPU에 넘기는 Python 시간이 포함된다.
- worker는 Python thread다. 읽기(pread) 중에는 GIL을 놓지만, 읽기 사이의 처리는 요청 thread와 GIL을 나눠 쓴다.
- 캐시 계층(GPU·CPU 캐시, CKLFU), 주기적 재배치, warm-up은 이식하지 않았다. 모든 요청이 cold SSD다. worker는 디스크 읽기만 하고 GPU 복사는 요청 thread가 한다(원본 대조표 j, m).
- 기존 IMPRESS 행에는 선택 계산 시간과 GPU 적재 시간이 따로 없어 6절에서 나란히 놓지 못했다.
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
    reports, errors = {}, {}
    for m in MODELS:
        try:
            reports[m] = model_report(top, m, config, manifest)
        except Exception:      # noqa: BLE001  (the other model's section is still written)
            errors[m] = traceback.format_exc()
    L = [f"# MT-GQA × HyperInfer 재측정 보고: prefetch를 논문 IV-C대로 ({top.name})", "",
         f"작성 {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}. 계약: `docs/mt_gqa_hyperinfer_v2_contract.md` (SHA256 `{config['contract_sha256'][:16]}…`). "
         "측정 보고이며 GO/NO-GO 판정은 없다.", "",
         "바꾼 것은 prefetch를 멈추는 방식 하나다. layer l의 GPU 계산이 끝나면 worker는 새 chunk 읽기를 시작하지 않고(이미 시작한 읽기 하나만 끝낸다), "
         "그때까지 읽은 것이 prefetch된 집합이 된다. 시간 예산과 chunk당 읽기 시간 상수는 쓰지 않는다.", "",
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
        rv = {x["comparison"]: x["ratio"] for x in r["paired"]}
        v, dev = r["vs_v1"], r.get("dev") or {}
        note = "" if r["main_complete"] else f" (중간 값: {r['main_images']} / 398 이미지)"
        L.append(f"- **{MODEL_LABEL[m]}**{note}: HyperInfer followup EM {a['HyperInfer']['em_followup']:.2f}%, TTFT {a['HyperInfer']['ttft_mean_ms']:.1f} ms "
                 f"(이전 실행 {v['v1']['ttft_ms']:.1f} ms). IMPRESS {a['IMPRESS']['em_followup']:.2f}% / {a['IMPRESS']['ttft_mean_ms']:.1f} ms, "
                 f"Ours {a['Ours']['em_followup']:.2f}% / {a['Ours']['ttft_mean_ms']:.1f} ms. "
                 f"TTFT 비율 HyperInfer / IMPRESS {fx(rv['HyperInfer 대 IMPRESS'] and rv['HyperInfer 대 IMPRESS']['ratio'], 3)}"
                 f"(같은 실행의 DEV {fx((dev.get('ratio') or {}).get('ratio'), 3)}), Ours / HyperInfer {fx(rv['Ours 대 HyperInfer'] and rv['Ours 대 HyperInfer']['ratio'], 3)}. "
                 f"prefetch 대기 {v['v1']['prefetch_wait_ms']:.1f} → {v['v2']['prefetch_wait_ms']:.2f} ms, prefetch로 채워진 비율 {100 * v['v1']['coverage']:.1f}% → {100 * v['v2']['coverage']:.1f}%. "
                 f"생성 토큰이 기존 IMPRESS 기록과 일치한 요청 {r['identity']['equal']:,} / {r['identity']['requests']:,}.")
    tests = (reports.get("next") or reports.get("l15") or {"gates": {}})["gates"].get("tests")
    L += ["", "## 두 모델 공통 gate", "", f"- gate 6 (기존 테스트, 새 실패 0개): **{tests['status'] if tests else '미실행'}**"]
    if tests:
        for s in tests["suites"]:
            L.append(f"  - transformers {s['transformers']}: {s['ran']}개 실행, 새 실패 {len(s['new_failures'])}개"
                     + (f" ({'; '.join(s['new_failures'])})" if s["new_failures"] else "")
                     + f", 이전부터 실패하던 테스트 {len(s['baseline_failures_for_this_environment'])}개: " + "; ".join(s["baseline_failures_for_this_environment"]))
    L.append("")
    for m in MODELS:
        if m in reports:
            try:
                L += [model_md(reports[m]), ""]
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
            protection = f"목록이 달라졌다. 수정 {len(ch['modified'])}건, 삭제 {len(ch['removed'])}건, 추가 {len(ch['added'])}건(추가는 다른 작업의 새 파일)."
        protection += " 동결한 소스 파일 변경 없음." if not post["frozen_source_changed"] else f" 동결 소스 변경: {post['frozen_source_changed']}"
    L += ["## 실행 전체", "",
          f"- 드라이버 시작 {len(inv)}회(이어서 실행 {max(0, len(inv) - 1)}회). 보고서를 쓰기 직전의 드라이버 단계: `{status.get('stage')}`."
          + (f" 마지막 오류: {json.dumps(status.get('last_error'), ensure_ascii=False)}" if status.get("last_error") else ""),
          f"- SSD 쓰기량(장치 전체 커널 카운터, 드라이버 시작 이후): {fx(written / 1e12 if written else None, 3)} TB. 사전 예상 {fx((prerun.get('expected_ssd_write') or {}).get('total_tb'), 3)} TB.",
          f"- 요청: 완료 {status.get('completed_requests')} / 전체 {status.get('expected_requests')}.",
          "- 기존 파일 보호: " + protection,
          "", "## 8. 코드, 지시와 다르게 한 것, 한계", "", CODE_MD, "### 지시와 다르게 했거나 지시에 없던 것", "",
          "계약 8절의 항목이 전부다(`docs/mt_gqa_hyperinfer_v2_contract.md`).", ""]
    contract = (top / "CONTRACT.md").read_text()
    sec = contract[contract.index("## 8. "):]
    L += [line for line in sec.splitlines()[2:] if line.strip()]
    L += ["", "### 원본 대조표에서 바뀐 행", "", "전체 표는 `docs/hyperinfer_port_contract.md`(이 실행의 사본: `runs/" + top.name + "/PORT_TABLE.md`)에 있다. 이번에 바뀌거나 더해진 행:", "",
          "| # | 항목 | 논문의 서술 (절) | 공개 코드의 동작 (파일·줄) | 우리 이식본 | 차이와 방향 |", "|---|---|---|---|---|---|"]
    table = (top / "PORT_TABLE.md").read_text().splitlines()
    L += [line for line in table if line[:6] in ("| b | ", "| c | ", "| d | ", "| f | ", "| i | ", "| k | ", "| t | ", "| u | ")]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "report.md").write_text("\n".join(L) + "\n")

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        if isinstance(o, (np.floating, np.integer)):
            return clean(o.item())
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o

    slim = {m: {k: v for k, v in r.items() if k not in ("names",)} for m, r in reports.items()}
    (out_root / "report.json").write_text(json.dumps(clean({"run": top.name, "models": slim, "errors": errors}), ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"report": str((out_root / "report.md").relative_to(ROOT)), "models": list(reports), "errors": list(errors)}))
    sys.exit(0)


if __name__ == "__main__":
    main()

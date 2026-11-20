#!/usr/bin/env python3
"""Tables and report of experiment B: LLaVA-1.5 x MT-VQA 25k, ReComp / FullLoad /
IMPRESS / Ours with the IMPRESS arm being the public-implementation path
(contract docs/llava15_mtvqa_impress_code_contract.md, section 10).

Reads committed rows only; takes no part in any measurement (not in the source
freeze).  ``--phase pilot`` writes the pilot table; ``--phase main`` writes the
full report; ``--phase validation`` the report when the main run was not
admitted.  ``--allow-partial`` reports the committed prefix of an unfinished
phase and says so.  ``--notes`` (hand-written: stopped run, disk / SSD writes,
surprises, things not done) is appended verbatim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mmimpress.mtvqa25k.adapter import read_rows  # noqa: E402

RECOMP, FULL, IMP = "recompute", "fullload_async", "impress_official_async"
OURS = "llava15_kv_PYR-b20_packed_async"
METHODS = [RECOMP, FULL, IMP, OURS]
NAME = {RECOMP: "ReComp", FULL: "FullLoad", IMP: "IMPRESS", OURS: "Ours"}
KIND = {FULL: "raster", IMP: "impress_official", OURS: "packed_PYR-b20"}
PAIRS = [(OURS, IMP), (OURS, FULL), (OURS, RECOMP), (IMP, FULL)]
EVAL = "199_eval_llava15_mtvqa_impress_code.py"
CONTRACT_DOC = "docs/llava15_mtvqa_impress_code_contract.md"
MB = 1e6


# ------------------------------------------------------------- per image io
def image_record(folder):
    """One committed image -> compact record (worker; never raises)."""
    folder = Path(folder)
    try:
        commit = json.loads((folder / "COMMITTED.json").read_text())
        attempt = folder / commit["attempt_id"]
        rows = read_rows(attempt / "raw.jsonl")
        out = {"image_id": folder.name, "status": "OK", "requests": len(rows), "rows": {}, "t1_identical": commit.get("t1_tokens_identical_across_arms"),
               "ordinal": rows[0]["global_dialog_ordinal"]}
        for r in rows:
            res = r["result"]
            c = {"score": r["score"], "ttft": r["ttft_ms"], "e2e": r["request_e2e_ms"], "bytes": r["ssd_read_bytes"], "reads": r["ssd_preads"],
                 "probe": r["probe_read_bytes"], "k": r["k_read_bytes"], "v": r["v_read_bytes"], "packed": r["packed_read_bytes"],
                 "blocked": r["read_blocked_ms"], "atype": r["annotation"]["answer_type"], "tokens": len(r["generated_token_ids"]),
                 "t1": r["generated_token_ids"] if r["turn_id"] == 1 else None, "prefill": res.get("prefill_ms"),
                 "pos": r["method_order_position"]}
            if r["turn_id"] > 1 and r["family"] == "impress_official":
                log = res["layer_log"]
                c["imp"] = {"fb": [int(x["mode"] == "fallback") for x in log], "sim": [x["sim"] for x in log],
                            "kc": [x["k_chunks"] for x in log], "vc": [x["v_chunks"] for x in log],
                            "un": res["raster_chunks"]["chunk_fraction"], "share": [d["prefix_share"] for d in res["layer_diagnostics"]],
                            "blocked_probe": res.get("probe_hook_blocked_ms"),
                            "ready": res.get("prefetch_ready_on_arrival"), "prefetched": res.get("prefetched_probe_reads"),
                            "wait": res.get("prefetch_wait_ms"), "hook": res.get("hook_ms"), "probe_read": res.get("probe_read_ms")}
            if r["turn_id"] > 1 and r["family"] in ("fullload", "ours"):
                ahead = (res.get("read_wait") or {}).get("readahead") or {}
                c["ahead"] = {"wait": ahead.get("wait_ms"), "worker": ahead.get("worker_read_ms"), "peak": ahead.get("peak_outstanding"),
                              "ready": ahead.get("ready_on_arrival"), "scatter": res.get("scatter_ms")}
            out["rows"][f"{r['method_id']}|{r['turn_id']}"] = c
        store = {}
        for method, kind in KIND.items():
            p = json.loads((attempt / f"{kind}_persistence.json").read_text())
            a = json.loads((attempt / f"{kind}_activation.json").read_text())
            s = {"logical_bytes": p["file_logical_bytes"], "kv_bytes": p["bytes"]["visual_kv"], "probe_bytes": p["bytes"].get("probe_sidecar", 0),
                 "store_ready_ms": p["timing_ms"]["persist_ms"], "writer_wall_ms": p["native_writer_wall_ms"], "service_ms": p["service_persistence_ms"],
                 "hash_ms": p["completion_full_hash_ms"], "ssd_write_ms": p["timing_ms"]["ssd_write_ms"], "durability_ms": p["timing_ms"]["durability_ms"],
                 "os_write_bytes": p["os_write_bytes"], "activation_ms": a["outer_wall_ms"]}
            if kind != "packed_PYR-b20":
                s["probe_write_ms"] = p["write_by_file_name"]["probe_k.bin"]["ms"]
            if kind == "impress_official":
                t = p["official_turn1"]; pk = t["layout"]["per_kind"]
                s.update(capture_ms=t["capture_ms"], materialize_ms=t["materialize_ms"], stats_save_ms=t["stats_save_ms"], order_ms=t["layout"]["order_ms"],
                         layout_ms=t["layout_ms"], after_request_ms=t["total_after_request_ms"],
                         reorder_read_ms=sum(v["read_ms"] for v in pk.values()), reorder_write_ms=sum(v["write_ms"] for v in pk.values()),
                         reorder_bytes=sum(v["bytes"] for v in pk.values()),
                         trailer_bytes=t["trailer_bytes_per_layer"] * len(t["trailer_sha256_per_layer"]))
                src_row = next(r for r in rows if r["physical_execution_id"] == p["source"]["physical_execution_id"])
                s["capture_request_t1_ttft_ms"] = src_row["ttft_ms"]
                s["other_arms_t1_ttft_ms"] = float(np.mean([r["ttft_ms"] for r in rows if r["turn_id"] == 1 and r["method_id"] == "recompute"]))
            store[method] = s
        out["store"] = store
        out["peak_scratch"] = json.loads((attempt / "disk_usage.json").read_text())["peak_scratch_bytes"]
        return out
    except Exception as exc:      # noqa: BLE001  (reported by the caller)
        return {"image_id": folder.name, "status": "ERROR", "error": repr(exc)}


def collect(run, phase, manifest, allow_partial, workers=12):
    images = manifest["phase_images"][phase]
    folders = [run / phase / "images" / i for i in images]
    present = [f for f in folders if (f / "COMMITTED.json").exists()]
    if len(present) != len(folders) and not allow_partial:
        raise RuntimeError(f"{phase}: {len(present)}/{len(folders)} images committed; use --allow-partial for an interim table")
    with Pool(workers) as pool:
        records = pool.map(image_record, [str(f) for f in present], chunksize=32)
    bad = [r for r in records if r["status"] != "OK"]
    if bad:
        raise RuntimeError("unreadable committed images: " + repr(bad[:3]))
    return records, len(folders)


# ---------------------------------------------------------------- analysis
def bootstrap(values, counts, resamples=10000):
    """scripts/142 ``bootstrap`` (image clusters, RandomState 1234, pooled sums / counts)."""
    import llava15_support as S
    return S.load_script("_mtvqa_analysis_142", "142_analyze_mtvqa25k.py").bootstrap(values, counts, resamples)


def cell(rec, method, turn, key):
    return rec["rows"][f"{method}|{turn}"][key]


def arm_table(records):
    rows = []
    for m in METHODS:
        sc = {t: np.array([cell(r, m, t, "score") for r in records]) for t in (1, 2, 3)}
        fu = np.concatenate([sc[2], sc[3]])
        ttft = np.array([cell(r, m, t, "ttft") for r in records for t in (2, 3)])
        e2e = np.array([cell(r, m, t, "e2e") for r in records for t in (2, 3)])
        t1 = np.array([cell(r, m, 1, "ttft") for r in records])
        by = {k: np.array([cell(r, m, t, k) for r in records for t in (2, 3)], dtype=float) for k in ("bytes", "probe", "k", "v", "packed", "reads")}
        blocked = [cell(r, m, t, "blocked") for r in records for t in (2, 3)]
        rows.append({"method": NAME[m], "method_id": m, "images": len(records), "followup_requests": int(fu.size),
                     "score_T1": 100 * sc[1].mean(), "score_T2": 100 * sc[2].mean(), "score_T3": 100 * sc[3].mean(),
                     "score_all": 100 * np.concatenate([sc[1], fu]).mean(), "score_followup": 100 * fu.mean(),
                     "followup_ttft_mean_ms": ttft.mean(), "followup_ttft_p50_ms": np.percentile(ttft, 50), "followup_ttft_p95_ms": np.percentile(ttft, 95),
                     "ssd_mb_per_followup": by["bytes"].mean() / MB, "probe_mb": by["probe"].mean() / MB, "k_mb": by["k"].mean() / MB,
                     "v_mb": by["v"].mean() / MB, "packed_mb": by["packed"].mean() / MB, "read_calls_per_followup": by["reads"].mean(),
                     "followup_e2e_mean_ms": e2e.mean(), "t1_ttft_mean_ms": t1.mean(),
                     "read_blocked_ms_mean": float(np.mean(blocked)) if m != RECOMP else None})
    base = {r["method_id"]: r["followup_ttft_mean_ms"] for r in rows}
    for r in rows:
        r["ttft_vs_fullload"] = r["followup_ttft_mean_ms"] / base[FULL]
        r["ttft_vs_recomp"] = r["followup_ttft_mean_ms"] / base[RECOMP]
    return rows


def ttft_table(records, label):
    stats = {}
    for m in METHODS:
        v = np.array([cell(r, m, t, "ttft") for r in records for t in (2, 3)])
        stats[m] = {"mean": v.mean(), "p50": np.percentile(v, 50), "p95": np.percentile(v, 95), "n": int(v.size)}
    return [{"segment": label, "method": NAME[m], "images": len(records), "followup_requests": stats[m]["n"],
             "followup_ttft_mean_ms": stats[m]["mean"], "followup_ttft_p50_ms": stats[m]["p50"], "followup_ttft_p95_ms": stats[m]["p95"],
             "ratio_vs_FullLoad": stats[m]["mean"] / stats[FULL]["mean"], "ratio_vs_ReComp": stats[m]["mean"] / stats[RECOMP]["mean"]} for m in METHODS]


def paired_table(records, resamples=10000):
    n = len(records)
    values = np.zeros((n, 2 * len(PAIRS))); counts = np.full((n, 2 * len(PAIRS)), 2.0)
    for i, r in enumerate(records):
        for j, (a, b) in enumerate(PAIRS):
            values[i, 2 * j] = 100 * sum(cell(r, a, t, "score") - cell(r, b, t, "score") for t in (2, 3))
            values[i, 2 * j + 1] = sum(cell(r, a, t, "ttft") - cell(r, b, t, "ttft") for t in (2, 3))
    point = values.sum(axis=0) / counts.sum(axis=0)
    ci = bootstrap(values, counts, resamples) if n > 1 else np.full((2, values.shape[1]), np.nan)
    out = []
    for j, (a, b) in enumerate(PAIRS):
        lo, hi = ci[0, 2 * j], ci[1, 2 * j]
        out.append({"comparison": f"{NAME[a]} − {NAME[b]}", "images": n,
                    "score_diff_pp": point[2 * j], "score_ci_low": lo, "score_ci_high": hi,
                    "score_reading": "차이 미입증" if lo <= 0 <= hi else ("높음" if lo > 0 else "낮음"),
                    "ttft_diff_ms": point[2 * j + 1], "ttft_ci_low": ci[0, 2 * j + 1], "ttft_ci_high": ci[1, 2 * j + 1],
                    "basis": "same run, image-cluster paired bootstrap 10,000, seed 1234; no reused record"})
    return out


def impress_internal(records):
    fb = np.array([cell(r, IMP, t, "imp")["fb"] for r in records for t in (2, 3)], dtype=float)          # (requests, 32)
    sim = np.array([cell(r, IMP, t, "imp")["sim"] for r in records for t in (2, 3)], dtype=float)
    kc = np.array([cell(r, IMP, t, "imp")["kc"] for r in records for t in (2, 3)], dtype=float)
    vc = np.array([cell(r, IMP, t, "imp")["vc"] for r in records for t in (2, 3)], dtype=float)
    imp = [cell(r, IMP, t, "imp") for r in records for t in (2, 3)]
    nc = 9.0
    share = np.array([x["share"] for x in imp], dtype=float)
    layers = [{"layer": li, "fallback_rate": fb[:, li].mean(), "mean_jaccard": sim[:, li].mean(),
               "k_chunk_fraction": kc[:, li].mean() / nc, "v_chunk_fraction": vc[:, li].mean() / nc,
               "prefix_share": share[:, li].mean()} for li in range(fb.shape[1])]
    blocked = {NAME[m]: {"mean_ms": float(np.mean([cell(r, m, t, "blocked") for r in records for t in (2, 3)])),
                         "p95_ms": float(np.percentile([cell(r, m, t, "blocked") for r in records for t in (2, 3)], 95))} for m in (FULL, IMP, OURS)}
    ahead = {}
    for m in (FULL, OURS):
        a = [cell(r, m, t, "ahead") for r in records for t in (2, 3)]
        ahead[NAME[m]] = {"readahead_wait_ms_mean": float(np.mean([x["wait"] for x in a])), "worker_read_ms_mean": float(np.mean([x["worker"] for x in a])),
                          "peak_outstanding_mean": float(np.mean([x["peak"] for x in a])), "peak_outstanding_max": int(max(x["peak"] for x in a)),
                          "items_ready_on_arrival_mean": float(np.mean([x["ready"] for x in a]))}
    return {"followup_requests": int(fb.shape[0]), "fallback_rate": float(fb.mean()), "mean_jaccard": float(sim.mean()),
            "threshold": 0.3, "prefix_share": float(share.mean()),
            "consensus_kv_chunk_fraction": float((((kc + vc) / (2 * nc))[fb == 0]).mean()) if (fb == 0).any() else None,
            "probe_hook_blocked_ms_mean": float(np.mean([x["blocked_probe"] for x in imp])), "k_chunk_fraction": float(kc.mean() / nc), "v_chunk_fraction": float(vc.mean() / nc),
            "kv_chunk_fraction": float((kc.mean() + vc.mean()) / (2 * nc)),
            "requests_with_any_fallback": float((fb.sum(axis=1) > 0).mean()), "requests_all_layers_fallback": float((fb.sum(axis=1) == fb.shape[1]).mean()),
            "raster_counterfactual_kv_chunk_fraction": float(np.mean([x["un"] for x in imp])),
            "raster_counterfactual_note": "computed from the request keep masks mapped back to raster positions; never read",
            "probe_prefetch": {"prefetched_reads_mean": float(np.mean([x["prefetched"] for x in imp])), "ready_on_arrival_mean": float(np.mean([x["ready"] for x in imp])),
                               "wait_ms_mean": float(np.mean([x["wait"] for x in imp])), "hook_ms_mean": float(np.mean([x["hook"] for x in imp])),
                               "probe_read_ms_mean": float(np.mean([x["probe_read"] for x in imp]))},
            "read_blocked_ms": blocked, "readahead": ahead, "layers": layers}


def storage_table(records):
    out = []
    for m in (FULL, IMP, OURS):
        s = [r["store"][m] for r in records]

        def mean(key):
            return float(np.mean([x[key] for x in s])) if key in s[0] else None
        out.append({"method": NAME[m], "store_kind": KIND[m], "images": len(s), "store_mb_per_image": mean("logical_bytes") / MB,
                    "visual_kv_mb": mean("kv_bytes") / MB, "probe_copy_mb": mean("probe_bytes") / MB,
                    "write_ms_store_ready_mean": mean("store_ready_ms"), "write_ms_store_ready_p95": float(np.percentile([x["store_ready_ms"] for x in s], 95)),
                    "probe_key_write_ms": mean("probe_write_ms"), "integrity_hash_ms": mean("hash_ms"),
                    "impress_capture_ms_inside_t1": mean("capture_ms"), "impress_materialize_ms": mean("materialize_ms"),
                    "impress_stats_save_ms": mean("stats_save_ms"), "impress_order_ms": mean("order_ms"), "impress_layout_ms": mean("layout_ms"),
                    "impress_reorder_read_ms": mean("reorder_read_ms"), "impress_reorder_write_ms": mean("reorder_write_ms"),
                    "impress_after_request_ms": mean("after_request_ms"), "impress_prefix_row_bytes": mean("trailer_bytes"),
                    "impress_capture_request_t1_ttft_ms": mean("capture_request_t1_ttft_ms"), "recomp_t1_ttft_ms_same_images": mean("other_arms_t1_ttft_ms"),
                    "service_persistence_ms_incl_hash": mean("service_ms"),
                    "activation_ms": mean("activation_ms"), "os_write_mb_per_image": mean("os_write_bytes") / MB})
    return out


def by_answer_type(records):
    out = []
    for atype in ("yes/no", "number", "other"):
        row = {"answer_type": atype}
        for m in METHODS:
            v = [cell(r, m, t, "score") for r in records for t in (2, 3) if cell(r, m, t, "atype") == atype]
            row[NAME[m]] = 100 * float(np.mean(v)) if v else None
            row["requests"] = len(v)
        out.append(row)
    return out


def t1_identity(records):
    same = 0
    for r in records:
        same += len({tuple(cell(r, m, 1, "t1")) for m in METHODS}) == 1
    return {"images": len(records), "identical_across_arms": same, "mismatched": len(records) - same}


# ------------------------------------------------------------------ output
def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def f1(x, d=1):
    return "미실행" if x is None else f"{x:,.{d}f}"


def tables_md(arms, paired, imp, storage, t1, segments, atypes):
    L = []
    L += ["| 방법 | Followup score | TTFT 평균(ms) | p50 | p95 | FullLoad 대비 | ReComp 대비 | SSD MB/followup |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for a in arms:
        L.append(f"| {a['method']} | {a['score_followup']:.4f} | {a['followup_ttft_mean_ms']:.1f} | {a['followup_ttft_p50_ms']:.1f} | {a['followup_ttft_p95_ms']:.1f} | "
                 f"{a['ttft_vs_fullload']:.3f}× | {a['ttft_vs_recomp']:.3f}× | {a['ssd_mb_per_followup']:.2f} |")
    L += ["", "턴별 VQA score:", "", "| 방법 | T1 | T2 | T3 | 전체 | Followup |", "|---|---:|---:|---:|---:|---:|"]
    for a in arms:
        L.append(f"| {a['method']} | {a['score_T1']:.4f} | {a['score_T2']:.4f} | {a['score_T3']:.4f} | {a['score_all']:.4f} | {a['score_followup']:.4f} |")
    i = next(a for a in arms if a["method_id"] == IMP)
    L += ["", f"IMPRESS의 종류별 읽기량(followup 평균): probe {i['probe_mb']:.2f} MB / K {i['k_mb']:.2f} MB / V {i['v_mb']:.2f} MB (합 {i['ssd_mb_per_followup']:.2f} MB). probe에는 prefix 행 기여분 0.22 MB(32 layer × 6,912 bytes)가 포함된다.", ""]
    L += ["참고(턴 1 TTFT, followup E2E, read 호출 수, 요청 스레드가 읽기를 기다린 시간):", "",
          "| 방법 | T1 TTFT 평균(ms) | followup E2E 평균(ms) | read 호출/followup | 읽기 대기 평균(ms) |", "|---|---:|---:|---:|---:|"]
    for a in arms:
        wait = "—" if a["method_id"] == RECOMP else f1(a["read_blocked_ms_mean"], 2)
        L.append(f"| {a['method']} | {a['t1_ttft_mean_ms']:.1f} | {a['followup_e2e_mean_ms']:.1f} | {a['read_calls_per_followup']:.1f} | {wait} |")
    for label, seg in segments:
        L += ["", f"### {label}", "", "| 방법 | followup TTFT 평균(ms) | p50 | p95 | FullLoad 대비 | ReComp 대비 |", "|---|---:|---:|---:|---:|---:|"]
        for s in seg:
            L.append(f"| {s['method']} | {s['followup_ttft_mean_ms']:.1f} | {s['followup_ttft_p50_ms']:.1f} | {s['followup_ttft_p95_ms']:.1f} | "
                     f"{s['ratio_vs_FullLoad']:.3f}× | {s['ratio_vs_ReComp']:.3f}× |")
    L += ["", "### 이미지 단위 paired bootstrap 95% CI (10,000회, seed 1234, followup; 모두 같은 실행, 기록 재사용 없음)", "",
          "| 비교 | VQA score 차이 (pp) | 해석 | TTFT 차이 (ms) |", "|---|---:|---|---:|"]
    for p in paired:
        L.append(f"| {p['comparison']} | {p['score_diff_pp']:+.4f} [{p['score_ci_low']:+.4f}, {p['score_ci_high']:+.4f}] | {p['score_reading']} | "
                 f"{p['ttft_diff_ms']:+.1f} [{p['ttft_ci_low']:+.1f}, {p['ttft_ci_high']:+.1f}] |")
    L += ["", "### IMPRESS 내부 통계 (followup)", "",
          f"- fallback 비율(layer·요청 평균): {100 * imp['fallback_rate']:.2f}% · probe head 간 평균 Jaccard: {imp['mean_jaccard']:.4f} (임계값 {imp['threshold']}, 이상이면 합의)",
          f"- fallback이 한 layer라도 있었던 요청: {100 * imp['requests_with_any_fallback']:.2f}% · 32개 layer 모두 fallback인 요청: {100 * imp['requests_all_layers_fallback']:.2f}%",
          f"- K·V chunk 비율(읽은 chunk / layer당 9 chunk, probe sidecar 제외): K {100 * imp['k_chunk_fraction']:.2f}% · V {100 * imp['v_chunk_fraction']:.2f}% · K·V 평균 {100 * imp['kv_chunk_fraction']:.2f}%"
          + ("" if imp["consensus_kv_chunk_fraction"] is None else f" · 합의 layer만 {100 * imp['consensus_kv_chunk_fraction']:.2f}%"),
          f"- 중요도 합에서 prefix 행이 차지하는 비중(probe head 3개 × 이미지 토큰 576개, layer·요청 평균): {100 * imp['prefix_share']:.2f}%",
          f"- 재배치하지 않았다면(계산값, 읽지 않음) K·V chunk 비율: {100 * imp['raster_counterfactual_kv_chunk_fraction']:.2f}%",
          f"- probe sidecar 선행 읽기: 요청당 {imp['probe_prefetch']['prefetched_reads_mean']:.1f}개 중 {imp['probe_prefetch']['ready_on_arrival_mean']:.2f}개가 사용 시점에 완료, 대기 {imp['probe_prefetch']['wait_ms_mean']:.2f} ms",
          "- GPU(요청 스레드)가 읽기를 기다린 시간, followup 평균 / p95: "
          + " · ".join(f"{k} {v['mean_ms']:.2f} / {v['p95_ms']:.2f} ms" for k, v in imp["read_blocked_ms"].items()),
          "- 읽기와 연산이 겹치므로 hook 시간과 디스크 시간의 합을 TTFT로 해석하지 않는다.", "",
          "| layer | fallback 비율 | 평균 Jaccard | K chunk 비율 | V chunk 비율 | prefix 행 비중 |", "|---:|---:|---:|---:|---:|---:|"]
    for x in imp["layers"]:
        L.append(f"| {x['layer']} | {100 * x['fallback_rate']:.2f}% | {x['mean_jaccard']:.4f} | {100 * x['k_chunk_fraction']:.2f}% | {100 * x['v_chunk_fraction']:.2f}% | "
                 f"{100 * x['prefix_share']:.2f}% |")
    L += ["", "### 저장·턴 1 비용 (이미지당 평균)", "",
          "| 방법 | store MB | Visual KV MB | probe 사본 MB | 쓰기(store 준비까지) ms | p95 | 무결성 해시 ms | 재배치 쓰기 ms | activation ms |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in storage:
        na = (lambda x: f1(x)) if s["method"] == "IMPRESS" else (lambda x: "—")      # only IMPRESS has a turn-1 layout step
        L.append(f"| {s['method']} | {s['store_mb_per_image']:.2f} | {s['visual_kv_mb']:.2f} | {s['probe_copy_mb']:.2f} | {s['write_ms_store_ready_mean']:.1f} | "
                 f"{s['write_ms_store_ready_p95']:.1f} | {f1(s['integrity_hash_ms'])} | {na(s['impress_layout_ms'])} | {s['activation_ms']:.1f} |")
    st = next(s for s in storage if s["method"] == "IMPRESS")
    L += ["", "IMPRESS의 턴 1 추가 비용(이미지당 평균, followup TTFT에는 들어 있지 않다):", "",
          "| 항목 | 시간(ms) | 어디서 |", "|---|---:|---|",
          f"| prefix 행 기여분 + 재배치 통계 캡처 (32 layer) | {f1(st['impress_capture_ms_inside_t1'])} | 턴 1 요청 안 (그 요청의 TTFT에 포함) |",
          f"| 캡처 결과를 host로 옮기기 | {f1(st['impress_materialize_ms'])} | 요청 밖 |",
          f"| 증거 파일(.npz) 저장 | {f1(st['impress_stats_save_ms'])} | 요청 밖 (실험 기록용) |",
          f"| 재배치 순서 계산 (32 layer 정렬) | {f1(st['impress_order_ms'])} | 요청 밖 |",
          f"| 재배치 쓰기 전체 (K / V / probe sidecar 읽기+치환+쓰기, meta) | {f1(st['impress_layout_ms'])} | 요청 밖 |",
          f"|   그중 읽기 / 쓰기 | {f1(st['impress_reorder_read_ms'])} / {f1(st['impress_reorder_write_ms'])} | |",
          f"| 요청 밖 합계 | {f1(st['impress_after_request_ms'])} | |", "",
          f"캡처가 걸린 턴 1 요청의 TTFT는 평균 {f1(st['impress_capture_request_t1_ttft_ms'])} ms, 같은 이미지의 ReComp 턴 1은 평균 {f1(st['recomp_t1_ttft_ms_same_images'])} ms다"
          "(캡처 요청에는 KV 캡처도 함께 걸려 있다). probe key 사본 쓰기는 " + f"{f1(st['probe_key_write_ms'])} ms(raster writer 안)다.",
          "\"쓰기(store 준비까지)\"는 raster writer(쓰기+fsync)의 시간이고 IMPRESS의 재배치 쓰기는 별도 열이다. ReComp는 store가 없다. "
          "FullLoad store에도 쓰이지 않는 probe 사본이 들어 있다(같은 writer).", "",
          "### answer type별 followup VQA score (탐색적, 다중비교 보정 없음)", "", "| answer type | 요청 수 | " + " | ".join(NAME[m] for m in METHODS) + " |", "|---|---:|" + "---:|" * 4]
    for a in atypes:
        L.append(f"| {a['answer_type']} | {a['requests']:,} | " + " | ".join(f"{a[NAME[m]]:.4f}" for m in METHODS) + " |")
    L += ["", f"턴 1 생성 토큰이 네 arm에서 모두 같은 이미지: {t1['identical_across_arms']:,} / {t1['images']:,}."]
    return L


def gate_md(run):
    L = ["| 단계 | 결과 | 내용 |", "|---|---|---|"]
    g = {}
    for name in ("cpu", "prefix", "gate", "pilot", "admission"):
        p = run / "gates" / f"{name}.json"
        g[name] = json.loads(p.read_text()) if p.exists() else None
    if g["cpu"]:
        c = g["cpu"]; s1 = c["step1_public_equivalence"]; s6 = c["step6_cpu_tests"]
        L.append(f"| 1. 공개 구현과의 동등성 (CPU) | {s1['status']} | 공개 함수 호출 테스트 {len(s1['public_cases'])}개 모두 실행·통과(skip 0), 커밋 `{s1['public_head'][:7]}` |")
    else:
        L.append("| 1. 공개 구현과의 동등성 | 미실행 | |")
    if g["prefix"]:
        c = g["prefix"]
        L.append(f"| 2. prefix 행 기여분 (gate 44장, GPU) | {c['status']} | {c['images_checked']}장 중 저장값 ≠ 재계산값인 이미지 {len(c['failed_images'])}장 |")
    else:
        L.append("| 2. prefix 행 기여분 | 미실행 | |")
    if g["gate"] and "step4_async_consistency" in g["gate"]:
        c = g["gate"]; s3 = c["step3_impress_r100_vs_fullload_loader_on_reordered_store"]; s4 = c["step4_async_consistency"]; s5 = c["step5_turn1_identity"]
        ref = c["reference_reordered_fullload_vs_raster_fullload"]
        L.append(f"| 3. 정합성 (유지 100% IMPRESS vs 같은 재배치 store의 FullLoad loader) | {s3['status']} | {s3['compared']}건 중 생성 토큰 불일치 {s3['different_tokens']}건 "
                 f"(입력이 달라진 요청 {s3['different_input']}건). 참고(판정 아님): 재배치 store FullLoad와 raster store FullLoad의 생성 토큰이 다른 요청 "
                 f"{ref['different_tokens']}건 / {ref['compared']}건 (그중 입력이 같은 요청 {ref['different_tokens_same_input']}건) |")
        L.append(f"| 4. 비동기 정합성 (gate 44장) | {s4['status']} | " + "; ".join(f"{k}: {s4[k]['compared']}건 중 불일치 {s4[k]['mismatched']}건" for k in ("FullLoad", "IMPRESS", "Ours")) + " |")
        L.append(f"| 5. 턴 1 일치 (gate 9 arm) | {s5['status']} | {s5['images']}장 중 {s5['identical']}장 일치 |")
    else:
        L += ["| 3. 정합성 | 미실행 | |", "| 4. 비동기 정합성 | 미실행 | |", "| 5. 턴 1 일치 | 미실행 | |"]
    if g["cpu"]:
        s6 = g["cpu"]["step6_cpu_tests"]
        L.append(f"| 6. 기존 테스트 | {s6['status']} | " + "; ".join(f"transformers {s['transformers']}: {s['ran']}개 실행, 실패 {len(s['failed'])}개(모두 환경 실패), 새 실패 {len(s['new_failures'])}개"
                                                              for s in s6["suites"]) + " |")
    else:
        L.append("| 6. 기존 테스트 | 미실행 | |")
    if g["pilot"] and "measured" in g["pilot"]:
        c = g["pilot"]
        L.append(f"| 7. Pilot 40장 (본조건 4 arm) | {c['status']} | {c['phase']['requests']}건 감사 PASS, 턴 1 일치 {c['step5_turn1_identity']['identical']}/{c['step5_turn1_identity']['images']}장, "
                 f"이미지당 {c['measured']['seconds_per_image_mean']:.2f} s → main 예상 {c['projection_main']['hours']:.1f} h, 디스크 {c['projection_main']['storage']} |")
    else:
        L.append("| 7. Pilot 40장 | 미실행 | |")
    a = g["admission"]
    L.append(f"| 8. 본실행 진입 | {a['status'] if a else '미실행'} | 1–7 모두 PASS일 때만 실행 |")
    return L, g


def settings_md(run, config):
    env = config["environment"]
    return ["## 설정", "",
            f"- 실행: `runs/{run.name}` (요청 단위 원본), 이 디렉터리(`results/{run.name}`)는 집계와 보고서.",
            f"- 환경: Python {env['python'].split()[0]}, torch {env['torch']}, **transformers {env['transformers']}** (`{env['executable']}`).",
            f"- 모델 `{config['model']}` revision `{config['revision'][:12]}…`, NF4/BF16/eager, FP16 SSD KV, greedy max 16, batch 1, CPU threads 2. visual token 576개 고정, separator 없음.",
            "- 모집단·절차: MT-VQA-v2-reconstructed-25k 25,000 images × 3턴(이미지당 dialogue 1개), method-local generated history, ten-answer leave-one-out 공식 VQA score, dialogue 단위 arm 순서 회전. "
            "턴 1은 모든 방법이 픽셀부터 계산, 캐시 예열 없음, 저장 KV 읽기는 요청 직전에 page cache를 비운 cold 조건.",
            "- IMPRESS(공개 구현 방식, `mmimpress/impress_official.py` — LLaVA-NeXT 실험과 같은 코드; 이 실험의 유일한 IMPRESS): probe head 0·1·2, layer마다 25% 유지(576개 중 144개), "
            "유사도 임계값 0.3(이상이면 합의), chunk 64 토큰. 중요도 = 턴 1에서 저장한 prefix 행 기여분 + 요청 때 계산한 suffix 행 기여분(전체 행의 attention 열 합). "
            "fallback은 그 layer의 이미지 토큰 K·V 전부를 읽어 mask 없이 사용. 재배치는 턴 1 요청 하나의 통계(모든 head·모든 행 열 합)로 한 번. 다음 layer probe sidecar 선행 읽기. "
            f"공개 구현 {config['official_checkout']['repository']} 커밋 `{config['official_checkout']['head']}`.",
            "- FullLoad: 전체 Visual KV(raster 순서), 워커가 layer 순서로 K·V 전체를 이어 읽음.",
            "- Ours: 34절의 PYR-b20 packed store(image-only CLIP 순서, layer별 배분 그대로), 워커가 packed 파일을 layer 순서로 이어 읽음(hit당 preadv 32회).",
            "- 비동기 공통: 백그라운드 워커 1개, 디스크 읽기만 워커가 하고 GPU 적재는 요청 스레드, 선행 읽기는 TTFT 타이머 시작 뒤에만, 앞서 읽는 깊이 상한 없음.",
            "- 캐시 계층 없음: 모든 방법이 cold SSD. ReComp·FullLoad의 완결된 이전 기록이 없어 네 arm 모두 이 실행 안에서 잰다.", ""]


def phase_times(run):
    times = {}
    for ph in ("gate", "pilot", "main"):
        starts = sorted(run.glob(f"{ph}_start_*.json")); done = run / f"{ph}_execution_complete.json"
        if starts and done.exists():
            segs = [json.loads(x.read_text())["started_unix"] for x in starts]
            times[ph] = {"process_starts": len(starts), "first_start_unix": segs[0], "finished_unix": json.loads(done.read_text())["finished_unix"]}
    return times


def protection_check(run, out_root):
    import llava15_support as S
    E = S.load_script("_l15_mtvqa_impress_code_eval_report", EVAL)
    before = json.loads((run / "protection_before.json").read_text())
    after = E.stat_inventory(run, out_root / "protection_after.tsv.gz")
    protection = {"before": before, "after": after, "unchanged": before["sha256"] == after["sha256"] and before["files"] == after["files"]}
    (out_root / "protection_after.json").write_text(json.dumps(protection, indent=1) + "\n")
    return protection, E.source_hashes()


def step4_md(g):
    """Details of validation step 3 and of the reordered-vs-raster reference count (gate phase)."""
    gate = g.get("gate") or {}
    s3 = gate.get("step3_impress_r100_vs_fullload_loader_on_reordered_store")
    ref = gate.get("reference_reordered_fullload_vs_raster_fullload")
    if not s3 or not ref:
        return []
    L = ["### 검증 3의 내용 (gate 44장, 132건)", "",
         f"- 판정: 유지 100% IMPRESS와, 같은 재배치 store를 FullLoad loader(선택·mask 없음)로 읽은 결과의 생성 토큰이 다른 요청 {s3['different_tokens']}건 / {s3['compared']}건.",
         f"- 참고(판정에 쓰지 않음): 재배치 store FullLoad와 raster store FullLoad의 생성 토큰이 다른 요청 {ref['different_tokens']}건 / {ref['compared']}건. "
         f"그중 두 요청의 입력이 같은 것은 {ref['different_tokens_same_input']}건이고, 나머지는 앞 턴의 답이 달라져 history가 달라진 요청이다. "
         "두 store의 K·V 바이트는 같고 행 순서만 다르다.", ""]
    for title, block, a, b in (("판정 대상 불일치", s3, "impress_official_r100", "fullload_reordered"), ("참고: 재배치 vs raster FullLoad", ref, "fullload_reordered", "fullload")):
        if not block["requests"]:
            continue
        L += [f"{title}:", "", f"| image | dialogue | 턴 | 입력 동일 | `{a}` 토큰 | `{b}` 토큰 | 달라진 위치 | raster FullLoad 상위 두 후보 logit |", "|---|---|---:|---|---|---|---:|---|"]
        for x in block["requests"]:
            st = x.get("logits_at_position")
            top = f"{st['top2'][0]} (id {st['top2_ids'][0]}) / {st['top2'][1]} (id {st['top2_ids'][1]})" if st else "—"
            L.append(f"| {x['image_id']} | {x['dialog_id']} | {x['turn_id']} | {'예' if x['same_input'] else '아니오'} | {x[a]} | {x[b]} | {x['position']} | {top} |")
        L.append("")
    return L


def sync_async_md(g):
    ref = (g.get("gate") or {}).get("reference_sync_vs_async_followup_ttft")
    if not ref:
        return []
    L = ["### 참고: gate 44장의 동기·비동기 followup TTFT (88요청/arm, 판정 기준 아님)", "",
         "| 방법 | 동기 평균(ms) | 비동기 평균(ms) | paired 평균 차이(ms) |", "|---|---:|---:|---:|"]
    for k in ("FullLoad", "IMPRESS", "Ours"):
        L.append(f"| {k} | {ref[k]['sync_mean_ms']:.1f} | {ref[k]['async_mean_ms']:.1f} | {ref[k]['paired_mean_difference_ms']:+.1f} |")
    return L + [""]


def report_without_main(run, manifest, config, out_root, notes):
    """Top-level report when the main run was not admitted: validation outcome, failure details, pilot table."""
    out_root.mkdir(parents=True, exist_ok=True)
    gates, g = gate_md(run)
    admitted = (g.get("admission") or {}).get("status") == "PASS"
    failed = [k for k in ("cpu", "prefix", "gate", "pilot") if (g.get(k) or {}).get("status") != "PASS"]
    lines = ["# 실험 B — LLaVA-1.5-7B × MT-VQA-v2-reconstructed-25k: ReComp / FullLoad / IMPRESS / Ours (IMPRESS = 공개 구현 방식)", "",
             "측정값만 적는다. 측정하지 않은 값은 \"미실행\"이다. 계약: `docs/llava15_mtvqa_impress_code_contract.md` (사본 `CONTRACT.md`).", ""]
    if admitted:
        lines += ["검증 단계 1–7이 모두 PASS다. 본실행 결과는 `--phase main`으로 만든 보고서를 본다.", ""]
    else:
        labels = {"cpu": ["1·6(공개 구현 동등성·기존 테스트)"], "prefix": ["2(prefix 행 기여분)"], "pilot": ["7(Pilot 40장)"]}
        gate = g.get("gate") or {}
        labels["gate"] = [name for key, name in (("step3_impress_r100_vs_fullload_loader_on_reordered_store", "3(정합성)"), ("step4_async_consistency", "4(비동기 정합성)"),
                                                 ("step5_turn1_identity", "5(턴 1 일치)")) if (gate.get(key) or {}).get("status") != "PASS"] or ["3–5(gate)"]
        names = [x for k in failed for x in labels[k]]
        lines += [f"**본실행(25,000장)은 미실행이다.** 검증 단계 {', '.join(names)}이(가) 사전 고정 기준으로 PASS가 아니어서, 계약 9절대로 본실행 없이 실패 내용만 보고한다.", ""]
    lines += settings_md(run, config) + ["## 검증 단계", ""] + gates + [""] + step4_md(g) + sync_async_md(g)
    summary = {"run": run.name, "main_executed": False, "admission": (g.get("admission") or {}).get("status"), "failed_steps": failed,
               "gates": {k: (v or {}).get("status") for k, v in g.items()}}
    pilot_done = all((run / "pilot" / "images" / i / "COMMITTED.json").exists() for i in manifest["phase_images"]["pilot"])
    if pilot_done:
        records, _ = collect(run, "pilot", manifest, False)
        arms = arm_table(records); paired = paired_table(records); imp = impress_internal(records); storage = storage_table(records)
        body = tables_md(arms, paired, imp, storage, t1_identity(records), [("Pilot 40장의 TTFT 비율", ttft_table(records, "pilot"))], by_answer_type(records))
        lines += ["## Pilot 40장 (본조건 4 arm, 480요청) — 본실행이 아니다", "",
                  "검증 7의 소규모 확인이다. 구현 확인용 40장이며 본실행 분모가 아니다. 40장에서의 score 차이는 구간이 넓으므로 방법 간 정확도 우열의 근거로 쓰지 않는다.",
                  "VQA score는 ten-answer leave-one-out 공식 VQA score(%). SSD MB는 read syscall이 실제로 돌려준 바이트(1 MB = 1e6 bytes)의 요청 평균.", ""] + body + [""]
        summary["pilot"] = {"arms": arms, "paired": paired, "impress_internal": imp, "storage": storage}
    lines += ["## 본실행 보고 항목", "",
              "| 항목 | 값 |", "|---|---|",
              "| 본실행 표 (25,000장, arm별 VQA score·TTFT·SSD MB, IMPRESS probe/K/V 바이트) | 미실행 |",
              "| 이미지 단위 paired bootstrap 95% CI (25,000장) | 미실행 |",
              "| IMPRESS 내부 통계 (25,000장) | 미실행 |",
              "| 저장·턴 1 비용 (25,000장) | 미실행 |", ""]
    times = phase_times(run)
    plan = json.loads((run / "storage_plan.json").read_text())
    protection, sources = protection_check(run, out_root)
    interruptions = sorted(x.name for x in run.glob("interruption_*.json"))
    lines += ["## 실행 기록", "",
              f"- 요청 수: gate {manifest['expected_requests']['gate']:,} · pilot {manifest['expected_requests']['pilot']:,} · main 0 / {manifest['expected_requests']['main']:,} (미실행).",
              "- 소요 시간: " + " · ".join(f"{ph} {(t['finished_unix'] - t['first_start_unix']) / 60:.1f}분" for ph, t in times.items()) + ".",
              f"- 중단·재개 기록: {len(interruptions)}건.",
              f"- 실행 전 디스크 여유 {plan['free_bytes_at_freeze'] / 2**30:.1f} GiB, 보고 시점 {shutil.disk_usage(run).free / 2**30:.1f} GiB. scratch store는 이미지마다 감사·commit 뒤 삭제했다.",
              f"- 기존 `results/`·`runs/`·`kvstore*` 파일 목록(경로, 크기, mtime) 비교: {'변화 없음' if protection['unchanged'] else '변화 있음 — protection_after.json 참고'} "
              f"({protection['before']['files']:,}개 → {protection['after']['files']:,}개).",
              f"- 소스 freeze 대조(보고 시점): {'일치' if sources == config['source_sha256'] else '불일치'}."]
    if notes is not None:
        lines += ["", notes.read_text().rstrip()]
    (out_root / "report.md").write_text("\n".join(lines) + "\n")
    summary.update(times=times, protection_unchanged=protection["unchanged"], source_freeze_matches=sources == config["source_sha256"],
                   environment=config["environment"], generated_unix=time.time())
    (out_root / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    for name in ("config.json", "CONTRACT.md", "storage_plan.json"):
        shutil.copyfile(run / name, out_root / name)
    if (run / "gates").is_dir():
        shutil.copytree(run / "gates", out_root / "gates", dirs_exist_ok=True)
    print(json.dumps({"phase": "validation", "admission": summary["admission"], "report": str(out_root / "report.md")}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--phase", choices=("pilot", "main", "validation"), required=True,
                   help="validation: the top-level report when the main run was not admitted")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--notes", type=Path, default=None, help="markdown appended verbatim to the main report (deviations, not done)")
    p.add_argument("--out", type=Path, default=None, help="output directory (default results/<run name>)")
    args = p.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    config = json.loads((run / "config.json").read_text())
    out_root = args.out.resolve() if args.out is not None else ROOT / "results" / run.name
    if args.phase == "validation":
        return report_without_main(run, manifest, config, out_root, args.notes)
    out = out_root / args.phase
    out.mkdir(parents=True, exist_ok=True)
    records, expected = collect(run, args.phase, manifest, args.allow_partial)
    partial = len(records) != expected
    arms = arm_table(records); paired = paired_table(records); imp = impress_internal(records); storage = storage_table(records)
    t1 = t1_identity(records); atypes = by_answer_type(records)
    segments = []
    seg_rows = []
    if args.phase == "main":
        first = set(manifest["first_segment_image_ids"])
        head = [r for r in records if r["image_id"] in first]
        if head:
            seg = ttft_table(head, f"first_{len(head)}"); segments.append((f"같은 실행 1,000장 구간 (main manifest 순서 앞쪽 {len(head):,}장)", seg)); seg_rows += seg
        whole = ttft_table(records, "all"); segments.append((f"참고: 전체 {len(records):,}장의 같은 비율", whole)); seg_rows += whole
    else:
        whole = ttft_table(records, "pilot"); segments.append((f"Pilot {len(records)}장의 TTFT 비율", whole)); seg_rows += whole
    write_csv(out / "arms.csv", arms); write_csv(out / "paired.csv", paired); write_csv(out / "storage.csv", storage)
    write_csv(out / "ttft_segments.csv", seg_rows); write_csv(out / "impress_layers.csv", imp["layers"]); write_csv(out / "answer_types.csv", atypes)
    requests = sum(r["requests"] for r in records)
    summary = {"run": run.name, "phase": args.phase, "partial": partial, "images": len(records), "expected_images": expected, "requests": requests,
               "arms": arms, "ttft_segments": seg_rows, "paired": paired, "impress_internal": imp, "storage": storage, "turn1_identity": t1,
               "answer_types": atypes, "peak_scratch_bytes_max": int(max(r["peak_scratch"] for r in records)),
               "config_sha256": hashlib.sha256((run / "config.json").read_bytes()).hexdigest(),
               "generated_unix": time.time()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, allow_nan=True) + "\n")
    body = tables_md(arms, paired, imp, storage, t1, segments, atypes)
    title = {"pilot": "Pilot 40장 (본조건 4 arm) — 검증 7의 소규모 확인", "main": "본실행"}[args.phase]
    head_note = f"요청 수: {requests:,} (이미지 {len(records):,}장 × 3턴 × 4 arm)." + (f" **INTERIM: {expected:,}장 중 {len(records):,}장만 commit된 중간 집계이며 최종 결과가 아니다.**" if partial else "")
    if args.phase == "pilot":
        lines = [f"# LLaVA-1.5-7B × MT-VQA 25k — {title}", "", "구현 확인용이며 본실행 분모에 넣지 않는다. 계약: `docs/llava15_mtvqa_impress_code_contract.md`.", "", head_note, "",
                 "VQA score는 ten-answer leave-one-out 공식 VQA score(%). SSD MB는 read syscall이 실제로 돌려준 바이트(1 MB = 1e6 bytes)의 요청 평균.", ""] + body
        (out / "report.md").write_text("\n".join(lines) + "\n")
        print(json.dumps({"phase": "pilot", "images": len(records), "report": str(out / "report.md")}), flush=True)
        return
    gates, g = gate_md(run)
    env = config["environment"]
    plan = json.loads((run / "storage_plan.json").read_text())
    times = {}
    for ph in ("gate", "pilot", "main"):
        starts = sorted(run.glob(f"{ph}_start_*.json")); done = run / f"{ph}_execution_complete.json"
        if starts and done.exists():
            segs = [json.loads(s.read_text())["started_unix"] for s in starts]
            times[ph] = {"process_starts": len(starts), "first_start_unix": segs[0], "finished_unix": json.loads(done.read_text())["finished_unix"]}
    interruptions = sorted(p.name for p in run.glob("interruption_*.json"))
    import llava15_support as S
    E = S.load_script("_l15_mtvqa_impress_code_eval_report", EVAL)
    before = json.loads((run / "protection_before.json").read_text())
    after = E.stat_inventory(run, out_root / "protection_after.tsv.gz")
    protection = {"before": before, "after": after, "unchanged": before["sha256"] == after["sha256"] and before["files"] == after["files"]}
    (out_root / "protection_after.json").write_text(json.dumps(protection, indent=1) + "\n")
    source_ok = E.source_hashes() == config["source_sha256"]
    lines = (["# 실험 B — LLaVA-1.5-7B × MT-VQA-v2-reconstructed-25k: ReComp / FullLoad / IMPRESS / Ours (IMPRESS = 공개 구현 방식)", "",
              "측정값만 적는다. 측정하지 않은 값은 \"미실행\"이다. 계약: `docs/llava15_mtvqa_impress_code_contract.md` (사본 `CONTRACT.md`).", ""]
             + settings_md(run, config) + ["## 검증 단계", ""] + gates + [""] + step4_md(g) + sync_async_md(g))
    lines += [
             f"## {title} (25,000장)", "", head_note, "",
             "VQA score는 ten-answer leave-one-out 공식 VQA score(%). SSD MB는 read syscall이 실제로 돌려준 바이트(1 MB = 1e6 bytes)의 요청 평균. "
             "FullLoad·ReComp 대비는 같은 실행 안의 TTFT 평균 비율이다.", ""] + body + ["",
             "## 실행 기록", "",
             f"- 요청 수: gate {manifest['expected_requests']['gate']:,} · pilot {manifest['expected_requests']['pilot']:,} · main {requests:,} / {manifest['expected_requests']['main']:,}"
             + (" (미완료)" if partial else "") + ". gate·pilot 요청은 main 분모에 넣지 않았다.",
             "- 소요 시간: " + " · ".join(f"{ph} {(t['finished_unix'] - t['first_start_unix']) / 3600:.2f} h (프로세스 시작 {t['process_starts']}회)" for ph, t in times.items()) + ".",
             f"- 중단·재개 기록: {len(interruptions)}건" + (" (" + ", ".join(interruptions) + ")" if interruptions else "") + ".",
             f"- 이미지당 scratch 최대 사용량: {summary['peak_scratch_bytes_max'] / MB:,.1f} MB. 실행 전 여유 {plan['free_bytes_at_freeze'] / 2**30:.1f} GiB, 보고 시점 여유 {shutil.disk_usage(run).free / 2**30:.1f} GiB.",
             f"- 기존 `results/`·`runs/`·`kvstore*` 파일 목록(경로, 크기, mtime) 비교: {'변화 없음' if protection['unchanged'] else '변화 있음 — protection_after.json 참고'} "
             f"({before['files']:,}개 → {after['files']:,}개).",
             f"- 소스 freeze 대조(보고 시점): {'일치' if source_ok else '불일치'}."]
    if args.notes is not None:
        lines += ["", args.notes.read_text().rstrip()]
    (out_root / "report.md").write_text("\n".join(lines) + "\n")
    summary.update(gates={k: (v or {}).get("status") for k, v in g.items()}, times=times, interruptions=interruptions, protection_unchanged=protection["unchanged"],
                   source_freeze_matches=source_ok, environment=env)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    for name in ("config.json", "CONTRACT.md", "storage_plan.json"):
        shutil.copyfile(run / name, out_root / name)
    if (run / "gates").is_dir():
        shutil.copytree(run / "gates", out_root / "gates", dirs_exist_ok=True)
    print(json.dumps({"phase": "main", "images": len(records), "partial": partial, "report": str(out_root / "report.md")}), flush=True)


if __name__ == "__main__":
    main()

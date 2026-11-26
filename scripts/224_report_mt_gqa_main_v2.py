#!/usr/bin/env python3
"""Tables and report of the LLaVA-NeXT x MT-GQA main experiment v2: ReComp / FullLoad / IMPRESS / Ours with the
original-LLaVA token sequence (contract docs/mt_gqa_main_v2_contract.md, section 8).

Reads committed rows only; takes no part in any measurement (not in the source freeze).

  --phase dev    the DEV pilot table (gate 8)
  --phase auto   results/<run>/report.md: the main tables when the main phase is complete, an interim main table
                 when it was admitted but is not complete, otherwise the gate outcome (and the DEV pilot table
                 if it exists)

A measurement report: no GO / NO-GO verdict is produced.  Methods are called ReComp, FullLoad, IMPRESS, Ours.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
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
from mmimpress.recompute_kv25 import paired_bootstrap  # noqa: E402


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# numeric helpers of the LLaVA-1.5 x MT-GQA report (per-image sums, bootstrap of ratios, storage and time tables,
# execution record); they read the names below from their module, which are set to this run's arms
R = load_script("_mtgqa_main_v2_report_helpers", "208_report_llava15_mt_gqa_main.py")

RECOMP, FULL, IMP = "recompute", "fullload_async", "impress_official_async"
OURS = "ours_origorder_kv25_PYR-beta-raw_packed_async"
METHODS = [RECOMP, FULL, IMP, OURS]
NAME = {RECOMP: "ReComp", FULL: "FullLoad", IMP: "IMPRESS", OURS: "Ours"}
KIND = {FULL: "raster", IMP: "impress_official", OURS: "packedorig_PYR-beta-raw"}
PAIRS = [(OURS, IMP), (OURS, FULL), (OURS, RECOMP), (IMP, FULL), (FULL, RECOMP)]
RATIOS = [(OURS, FULL), (OURS, RECOMP), (OURS, IMP), (IMP, FULL)]
PHASES = ("legacy", "gate", "devgate", "dev", "main")
CONTRACT_DOC = "docs/mt_gqa_main_v2_contract.md"
DRIVER = "scripts/run_mt_gqa_main_v2.sh"
RESAMPLES, SEED = 10000, 1234
MB = 1e6
PREVIOUS_RUN = {RECOMP: "mt_gqa_ours_order_20261120T052725Z", FULL: "mt_gqa_ours_order_20261120T052725Z",
                OURS: "mt_gqa_ours_order_20261120T052725Z", IMP: "mt_gqa_impress_code_20261117T185832Z"}
PREVIOUS_DATE = {RECOMP: "11/20", FULL: "11/20", OURS: "11/20", IMP: "11/17"}
# MetaCompress base model (LLaVA-NeXT-7B, MT-GQA), per-turn accuracy as given in the task statement
METACOMPRESS = {"Acc1": 63.83, "Acc2": 66.68, "Acc3": 67.94, "avg": 66.15}
for _k, _v in (("RECOMP", RECOMP), ("FULL", FULL), ("IMP", IMP), ("OURS", OURS), ("METHODS", METHODS), ("NAME", NAME), ("KIND", KIND),
               ("PAIRS", PAIRS), ("RATIOS", RATIOS), ("PHASES", PHASES), ("RESAMPLES", RESAMPLES), ("SEED", SEED)):
    setattr(R, _k, _v)
fx, utc, jsonl, write_csv = R.fx, R.utc, R.jsonl, R.write_csv


# ------------------------------------------------------------- per image io
def image_record(args):
    """One committed image -> compact record (worker; never raises)."""
    folder, methods = Path(args[0]), args[1]
    try:
        commit = json.loads((folder / "COMMITTED.json").read_text())
        attempt = folder / commit["attempt_id"]
        rows = read_rows(attempt / "raw.jsonl")
        out = {"image_id": folder.name, "status": "OK", "requests": len(rows), "rows": [], "resumed_inside_image": commit.get("resumed_inside_image", 0),
               "t1_identical": commit.get("t1_tokens_identical_across_arms"), "attempt": commit["attempt_id"],
               "token_match": sum(bool(r["official_token_ids_match"]) for r in rows), "tokenization": sorted({r["tokenization"] for r in rows})}
        for r in rows:
            if r["method_id"] not in methods:
                continue
            res = r["result"]
            c = {"m": r["method_id"], "d": r["dialog_id"], "t": r["turn_id"], "score": r["score"], "ttft": r["ttft_ms"], "e2e": r["request_e2e_ms"],
                 "bytes": r["ssd_read_bytes"], "reads": r["ssd_preads"], "probe": r["probe_read_bytes"], "k": r["k_read_bytes"], "v": r["v_read_bytes"],
                 "packed": r["packed_read_bytes"], "blocked": r["read_blocked_ms"], "prefill": res.get("prefill_ms"), "hook": res.get("hook_ms"),
                 "scatter": res.get("scatter_ms"), "pid": r.get("process_id"), "at": r.get("completed_at_unix"), "tok": r["generated_token_ids"],
                 "pos": r["method_order_position"], "ntok": r.get("input_token_count")}
            ahead = (res.get("read_wait") or {}).get("readahead") or {}
            if ahead:
                c["worker"] = ahead.get("worker_read_ms"); c["ready"] = ahead.get("ready_on_arrival")
            if r["turn_id"] > 1 and r["family"] == "impress_official":
                log = res["layer_log"]
                c["imp"] = {"fb": [int(x["mode"] == "fallback") for x in log], "sim": [x["sim"] for x in log],
                            "kc": [x["k_chunks"] for x in log], "vc": [x["v_chunks"] for x in log],
                            "share": [d["prefix_share"] for d in res["layer_diagnostics"]],
                            "raster": res["raster_chunks"]["chunks"], "only": res["importance_only_chunks"]["chunks"],
                            "nc": (r["geometry"]["v_token_num"] + 63) // 64,
                            "blocked_probe": res.get("probe_hook_blocked_ms"), "probe_read": res.get("probe_read_ms"),
                            "ready": res.get("prefetch_ready_on_arrival"), "prefetched": res.get("prefetched_probe_reads")}
            out["rows"].append(c)
        store = {}
        for method, kind in KIND.items():
            pp = attempt / f"{kind}_persistence.json"
            if method not in methods or not pp.exists():
                continue
            p = json.loads(pp.read_text())
            a = json.loads((attempt / f"{kind}_activation.json").read_text())
            s = {"logical_bytes": p["file_logical_bytes"], "kv_bytes": p["bytes"]["visual_kv"], "probe_bytes": p["bytes"].get("probe_sidecar", 0),
                 "meta_bytes": p["bytes"].get("meta", 0), "store_ready_ms": p["timing_ms"]["persist_ms"], "writer_wall_ms": p["native_writer_wall_ms"],
                 "service_ms": p["service_persistence_ms"], "hash_ms": p["completion_full_hash_ms"], "ssd_write_ms": p["timing_ms"]["ssd_write_ms"],
                 "durability_ms": p["timing_ms"]["durability_ms"], "repack_ms": p["timing_ms"].get("kv_repack_ms"),
                 "row_plan_ms": p["timing_ms"].get("row_plan_ms"), "permutation_ms": p["timing_ms"].get("permutation_ms"),
                 "os_write_bytes": p["os_write_bytes"], "activation_ms": a["outer_wall_ms"], "after_request_ms": 0.0}
            if kind == "impress_official":
                t = p["official_turn1"]
                s.update(capture_ms=t["capture_ms"], layout_ms=t["layout_ms"], after_request_ms=t["total_after_request_ms"],
                         trailer_bytes=t["trailer_bytes_per_layer"] * len(t["trailer_sha256_per_layer"]))
            s["total_store_ms"] = s["store_ready_ms"] + s["after_request_ms"]
            store[method] = s
        out["store"] = store
        out["peak_scratch"] = json.loads((attempt / "disk_usage.json").read_text())["peak_scratch_bytes"]
        meta = attempt / f"{KIND[OURS]}_meta.json"
        if meta.exists():
            m = json.loads(meta.read_text())
            out["ours_table"] = {"idx": m["stored_original_indices"], "thr": m["stored_min_keep_count"], "counts": m["layer_keep_counts"],
                                 "v_token_num": m["v_token_num"], "newline_idx": m["newline_idx"]}
        return out
    except Exception as exc:      # noqa: BLE001  (reported by the caller)
        return {"image_id": folder.name, "status": "ERROR", "error": repr(exc)}


def collect(run, phase, manifest, methods, allow_partial, workers=12):
    images = manifest["phase_images"][phase]
    folders = [run / phase / "images" / i for i in images]
    present = [f for f in folders if (f / "COMMITTED.json").exists()]
    if len(present) != len(folders) and not allow_partial:
        raise RuntimeError(f"{phase}: {len(present)}/{len(folders)} images committed")
    with Pool(workers) as pool:
        records = pool.map(image_record, [(str(f), methods) for f in present], chunksize=8)
    bad = [r for r in records if r["status"] != "OK"]
    if bad:
        raise RuntimeError("unreadable committed images: " + repr(bad[:3]))
    for r in records:
        r["by"] = {(c["m"], c["d"], c["t"]): c for c in r["rows"]}
        r["dialogs"] = sorted({c["d"] for c in r["rows"]})
    return records, len(folders)


# ---------------------------------------------------------------- analysis
def impress_internal(records):
    imp = [c["imp"] for c in R.cells(records, IMP)]
    fb = np.array([x["fb"] for x in imp], dtype=float); sim = np.array([x["sim"] for x in imp], dtype=float)
    kc = np.array([x["kc"] for x in imp], dtype=float); vc = np.array([x["vc"] for x in imp], dtype=float)
    share = np.array([x["share"] for x in imp], dtype=float)
    raster = np.array([x["raster"] for x in imp], dtype=float); only = np.array([x["only"] for x in imp], dtype=float)
    nc = np.array([x["nc"] for x in imp], dtype=float)[:, None]
    cons = fb == 0
    kv = (kc + vc) / (2 * nc)
    layers = [{"layer": li, "fallback_rate": fb[:, li].mean(), "mean_jaccard": sim[:, li].mean(), "kv_chunk_fraction": kv[:, li].mean(),
               "raster_chunk_fraction_computed": (raster[:, li:li + 1] / nc).mean(), "prefix_share": share[:, li].mean()} for li in range(fb.shape[1])]
    return {"followup_requests": int(fb.shape[0]), "fallback_rate": float(fb.mean()), "mean_jaccard": float(sim.mean()), "threshold": 0.3,
            "k_chunk_fraction": float((kc / nc).mean()), "v_chunk_fraction": float((vc / nc).mean()), "kv_chunk_fraction": float(kv.mean()),
            "consensus_kv_chunk_fraction": float(kv[cons].mean()) if cons.any() else None,
            "raster_chunk_fraction_computed": float((raster / nc).mean()),
            "consensus_raster_chunk_fraction_computed": float(((raster / nc)[cons]).mean()) if cons.any() else None,
            "importance_only_chunk_fraction_computed": float((only / nc).mean()),
            "consensus_importance_only_chunk_fraction_computed": float(((only / nc)[cons]).mean()) if cons.any() else None,
            "prefix_share": float(share.mean()),
            "requests_with_any_fallback": float((fb.sum(axis=1) > 0).mean()), "requests_all_layers_fallback": float((fb.sum(axis=1) == fb.shape[1]).mean()),
            "probe_prefetched_reads_mean": float(np.mean([x["prefetched"] for x in imp])), "probe_ready_on_arrival_mean": float(np.mean([x["ready"] for x in imp])),
            "layers": layers}


def previous_comparison(run, records):
    """Report item 4: this run against the earlier main runs, request by request (same method, dialogue, turn)."""
    ref = run / "reference"
    if not (ref / "previous_rows.json.gz").exists():
        return None
    with gzip.open(ref / "previous_rows.json.gz", "rt", encoding="utf-8") as f:
        old = {(r["m"], r["d"], r["t"]): r for r in json.load(f)}
    with gzip.open(ref / "previous_ours_tables.json.gz", "rt", encoding="utf-8") as f:
        tables = json.load(f)
    out = {"methods": [], "sources": json.loads((ref / "previous_sources.json").read_text())["sources"]}
    for m in METHODS:
        per_image = []; row = {"method": NAME[m], "method_id": m, "previous_run": PREVIOUS_RUN[m], "previous_date": PREVIOUS_DATE[m],
                               "requests": 0, "missing_in_previous": 0, "tokens_differ": 0, "tokens_differ_by_turn": {1: 0, 2: 0, 3: 0},
                               "prompt_differs_by_turn": {1: 0, 2: 0, 3: 0},
                               "correctness_changed": 0, "only_previous_correct": 0, "only_this_run_correct": 0,
                               "followup_only_previous_correct": 0, "followup_only_this_run_correct": 0,
                               "correctness_changed_by_turn": {1: 0, 2: 0, 3: 0}}
        sums = {"old": {1: [], 2: [], 3: []}, "new": {1: [], 2: [], 3: []}}
        ttft = {"old": [], "new": []}
        for r in records:
            a = b = n = 0
            for d in r["dialogs"]:
                for t in (1, 2, 3):
                    new = r["by"][(m, d, t)]; prev = old.get((m, d, t))
                    if prev is None:
                        row["missing_in_previous"] += 1
                        continue
                    row["requests"] += 1
                    sums["old"][t].append(prev["score"]); sums["new"][t].append(new["score"])
                    if new["tok"] != prev["tok"]:
                        row["tokens_differ"] += 1; row["tokens_differ_by_turn"][t] += 1
                    if new["score"] != prev["score"]:
                        row["correctness_changed"] += 1; row["correctness_changed_by_turn"][t] += 1
                        row["only_previous_correct"] += prev["score"] > new["score"]; row["only_this_run_correct"] += new["score"] > prev["score"]
                        if t > 1:
                            row["followup_only_previous_correct"] += prev["score"] > new["score"]; row["followup_only_this_run_correct"] += new["score"] > prev["score"]
                    if t > 1:
                        a += new["score"]; b += prev["score"]; n += 1
                        ttft["old"].append(prev["ttft"]); ttft["new"].append(new["ttft"])
            if n:
                per_image.append((100.0 * a, 100.0 * b, n))
        fu = lambda key: 100 * float(np.mean(sums[key][2] + sums[key][3])) if sums[key][2] else None      # noqa: E731
        row.update(em_followup_previous=fu("old"), em_followup_this_run=fu("new"),
                   em_by_turn_previous={t: 100 * float(np.mean(sums["old"][t])) for t in (1, 2, 3) if sums["old"][t]},
                   em_by_turn_this_run={t: 100 * float(np.mean(sums["new"][t])) for t in (1, 2, 3) if sums["new"][t]},
                   ttft_followup_previous_ms=float(np.mean(ttft["old"])) if ttft["old"] else None,
                   ttft_followup_this_run_ms=float(np.mean(ttft["new"])) if ttft["new"] else None)
        if len(per_image) > 1:
            bs = paired_bootstrap(per_image, RESAMPLES, SEED)
            row.update(em_followup_diff_pp=bs["mean"], em_followup_diff_ci=[bs["ci_low"], bs["ci_high"]])
        out["methods"].append(row)
    same = differ = missing = 0; examples = []
    for r in records:
        now, prev = r.get("ours_table"), tables.get(r["image_id"])
        if now is None or prev is None:
            missing += 1
            continue
        if now == prev:
            same += 1
        else:
            differ += 1
            if len(examples) < 5:
                examples.append({"image_id": r["image_id"], "counts_equal": now["counts"] == prev["counts"],
                                 "tokens_in_common": len(set(now["idx"]) & set(prev["idx"])), "tokens": len(now["idx"])})
    out["ours_token_lists"] = {"images": len(records), "identical": same, "different": differ, "not_compared": missing, "examples": examples,
                               "what_is_compared": "per image, the Ours store's index tables (stored_original_indices, stored_min_keep_count) and the per-layer row "
                                                   "counts -- they determine every layer's list of stored original token indices"}
    return out


def metacompress_table(arms, previous):
    a = next(x for x in arms if x["method_id"] == RECOMP)
    row = {"Acc1": a["em_T1"], "Acc2": a["em_T2"], "Acc3": a["em_T3"]}
    row["avg"] = float(np.mean([row["Acc1"], row["Acc2"], row["Acc3"]]))
    out = {"metacompress_base_model": METACOMPRESS, "recomp_this_run": row, "difference_pp": {k: row[k] - METACOMPRESS[k] for k in METACOMPRESS}}
    if previous:
        p = next((x for x in previous["methods"] if x["method_id"] == RECOMP), None)
        if p and p.get("em_by_turn_previous"):
            e = p["em_by_turn_previous"]
            out["recomp_previous_run"] = {"Acc1": e.get(1), "Acc2": e.get(2), "Acc3": e.get(3), "avg": float(np.mean([e[1], e[2], e[3]])) if len(e) == 3 else None}
    return out


# ------------------------------------------------------------------ output
def main_tables_md(arms, paired, ratios, imp, storage, decomp, t1, previous, meta):
    L = ["### 1. 본실험 표", "",
         "| 방법 | Followup EM (%) | T1 EM | T2 EM | T3 EM | 전체 EM | TTFT 평균 (ms) | p50 | p95 | SSD MB/followup | 읽기 호출 수/followup |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a in arms:
        L.append(f"| {a['method']} | {a['em_followup']:.3f} | {a['em_T1']:.3f} | {a['em_T2']:.3f} | {a['em_T3']:.3f} | {a['em_all']:.3f} | "
                 f"{a['followup_ttft_mean_ms']:.1f} | {a['followup_ttft_p50_ms']:.1f} | {a['followup_ttft_p95_ms']:.1f} | "
                 f"{a['ssd_mb_per_followup']:.2f} | {a['read_calls_per_followup']:.1f} |")
    L += ["", "Followup = T2–T3, strict normalized EM. 전체 EM은 T1–T3 세 턴 요청 전체의 EM이다. TTFT·SSD·읽기 호출 수는 followup 요청의 값이다. "
              "SSD MB는 read syscall이 실제로 돌려준 바이트(1 MB = 1e6 bytes)의 요청 평균, 읽기 호출 수는 pread + preadv 횟수의 요청 평균.", "",
          "| 방법 | T1 TTFT 평균 (ms) | followup E2E 평균 (ms) |", "|---|---:|---:|"]
    for a in arms:
        L.append(f"| {a['method']} | {a['t1_ttft_mean_ms']:.1f} | {a['followup_e2e_mean_ms']:.1f} |")
    i = next((a for a in arms if a["method_id"] == IMP), None)
    if i:
        L += ["", f"IMPRESS의 종류별 읽기량(followup 평균): probe key 사본 {i['probe_mb']:.2f} MB / K {i['k_mb']:.2f} MB / V {i['v_mb']:.2f} MB. "
                  f"읽기 호출 수는 요청마다 다르다(최소 {i['read_calls_min']:.0f}, 최대 {i['read_calls_max']:.0f})."]
    L += ["", f"### 2. 짝지은 비교 (이미지 단위 paired bootstrap {RESAMPLES:,}회, seed {SEED}, 95% CI, followup)", "",
          "| 비교 | EM 차이 (pp) | TTFT 차이 (ms) |", "|---|---:|---:|"]
    for p in paired:
        if "em_diff_pp" in p:
            L.append(f"| {p['comparison']} | {p['em_diff_pp']:+.3f} [{p['em_ci_low']:+.3f}, {p['em_ci_high']:+.3f}] | "
                     f"{p['ttft_diff_ms']:+.1f} [{p['ttft_ci_low']:+.1f}, {p['ttft_ci_high']:+.1f}] |")
    L += ["", "CI가 0을 포함하는 비교는 차이가 입증되지 않은 것이다(같다는 뜻이 아니다).",
          "", "### 3. TTFT 비율 (followup 평균의 비, 같은 실행; 괄호는 이미지 단위 bootstrap 95% CI)", "", "| 비율 | 값 |", "|---|---:|"]
    for r in ratios:
        L.append(f"| {r['ratio']} | {fx(r['ttft_ratio_of_means'], 3)}× [{fx(r['ci_low'], 3)}, {fx(r['ci_high'], 3)}] |")
    L += ["", "### 4. 기존 결과와의 대조", ""]
    if previous:
        L += ["기존 실행은 `<image>` 직후의 `▁` 토큰이 빠진 토큰열로 돌았다(ReComp·FullLoad·Ours는 11/20 `mt_gqa_ours_order_20261120T052725Z`, "
              "IMPRESS는 11/17 `mt_gqa_impress_code_20261117T185832Z`). 같은 (방법, 대화, 턴) 요청끼리 비교한다. 생성 토큰이 다른 것은 gate 실패가 아니다.", "",
              "| 방법 | 기존 실행 | Followup EM 기존 → 이번 (%) | 차이 (pp) [95% CI] | Followup TTFT 기존 → 이번 (ms) | 생성 토큰이 달라진 요청 | 정오가 바뀐 요청 (기존만 정답 / 이번만 정답) |",
              "|---|---|---:|---:|---:|---:|---:|"]
        for p in previous["methods"]:
            ci = (f" [{p['em_followup_diff_ci'][0]:+.3f}, {p['em_followup_diff_ci'][1]:+.3f}]" if "em_followup_diff_ci" in p else "")
            diff = f"{p['em_followup_diff_pp']:+.3f}{ci}" if "em_followup_diff_pp" in p else "—"
            L.append(f"| {p['method']} | {p['previous_date']} | {fx(p['em_followup_previous'], 3)} → {fx(p['em_followup_this_run'], 3)} | {diff} | "
                     f"{fx(p['ttft_followup_previous_ms'])} → {fx(p['ttft_followup_this_run_ms'])} | {p['tokens_differ']:,} / {p['requests']:,} | "
                     f"{p['correctness_changed']:,} ({p['only_previous_correct']:,} / {p['only_this_run_correct']:,}) |")
        L += ["", "턴별:", "", "| 방법 | 토큰이 달라진 요청 T1 / T2 / T3 | 정오가 바뀐 요청 T1 / T2 / T3 | EM 기존 T1 / T2 / T3 | EM 이번 T1 / T2 / T3 |", "|---|---:|---:|---:|---:|"]
        for p in previous["methods"]:
            td, cd = p["tokens_differ_by_turn"], p["correctness_changed_by_turn"]
            eo, en = p["em_by_turn_previous"], p["em_by_turn_this_run"]
            L.append(f"| {p['method']} | {td[1]:,} / {td[2]:,} / {td[3]:,} | {cd[1]:,} / {cd[2]:,} / {cd[3]:,} | "
                     + " / ".join(fx(eo.get(t), 2) for t in (1, 2, 3)) + " | " + " / ".join(fx(en.get(t), 2) for t in (1, 2, 3)) + " |")
        o = previous["ours_token_lists"]
        n_all = previous["methods"][0]["requests"]
        L += ["", f"- 요청 수는 방법당 T1–T3 전체({n_all:,}건)이고 EM·TTFT는 followup({n_all * 2 // 3:,}건)이다. 정오가 바뀐 요청 수는 T1–T3 전체에서 센 값이다"
                  "(followup만: " + " · ".join(f"{p['method']} {p['followup_only_previous_correct']:,} / {p['followup_only_this_run_correct']:,}" for p in previous["methods"]) + ").",
              "- EM 차이의 CI는 이번 실행 − 기존 실행을 이미지 단위로 짝지어 bootstrap한 값이다(서로 다른 실행이지만 요청과 채점이 같다). "
              "TTFT는 서로 다른 날의 실행 값이라 나란히 적기만 했다. 이번 실행은 suffix가 토큰 하나 길다.",
              "- T2·T3는 자기 방법의 생성 이력을 쓰므로, 앞 턴의 답이 달라지면 프롬프트도 달라진다.",
              f"- **Ours가 고른 layer별 토큰 목록**: 이미지 {o['images']:,}장 중 기존(11/20)과 같은 이미지 {o['identical']:,}장, 다른 이미지 {o['different']:,}장"
              + (f", 비교하지 못한 이미지 {o['not_compared']:,}장" if o["not_compared"] else "") + ". "
              "비교한 것은 이미지마다 Ours store의 인덱스 표와 layer별 행 수(각 layer가 저장하는 원래 토큰 인덱스 목록을 정한다)다."
              + (" 다른 이미지의 예: " + json.dumps(o["examples"], ensure_ascii=False) if o["examples"] else "")]
    else:
        L += ["기존 실행의 요청 기록 사본(`reference/previous_rows.json.gz`)이 없어 대조하지 못했다(미실행)."]
    L += ["", "### 5. MetaCompress 기본 모델과 ReComp의 턴별 EM", "",
          "| | Acc1 (T1) | Acc2 (T2) | Acc3 (T3) | 평균 |", "|---|---:|---:|---:|---:|"]
    mc = meta["metacompress_base_model"]; rc = meta["recomp_this_run"]; df = meta["difference_pp"]
    L.append(f"| MetaCompress 기본 모델 | {mc['Acc1']:.2f} | {mc['Acc2']:.2f} | {mc['Acc3']:.2f} | {mc['avg']:.2f} |")
    L.append(f"| ReComp (이번 실행) | {rc['Acc1']:.2f} | {rc['Acc2']:.2f} | {rc['Acc3']:.2f} | {rc['avg']:.2f} |")
    L.append(f"| 차이 (ReComp − MetaCompress, pp) | {df['Acc1']:+.2f} | {df['Acc2']:+.2f} | {df['Acc3']:+.2f} | {df['avg']:+.2f} |")
    if meta.get("recomp_previous_run"):
        pr = meta["recomp_previous_run"]
        L.append(f"| ReComp (11/20 실행, 참고) | {fx(pr['Acc1'], 2)} | {fx(pr['Acc2'], 2)} | {fx(pr['Acc3'], 2)} | {fx(pr['avg'], 2)} |")
    L += ["", "ReComp의 평균은 세 턴 EM의 산술 평균이다. MetaCompress의 MT-GQA 대화 파일은 공개되지 않아 이 실험은 같은 원본(GQA testdev-balanced)에서 "
              "재구성한 4,061개 대화를 쓴다. 대화 구성과 채점(strict normalized EM)이 MetaCompress와 같다는 보장은 없다."]
    if imp:
        L += ["", "### 6. IMPRESS 통계 (followup)", "",
              f"- fallback 비율(layer·요청 평균): {100 * imp['fallback_rate']:.2f}% — 한 layer라도 fallback인 요청 {100 * imp['requests_with_any_fallback']:.2f}%, "
              f"32개 layer 모두 fallback인 요청 {100 * imp['requests_all_layers_fallback']:.2f}%",
              f"- probe head 간 평균 Jaccard: {imp['mean_jaccard']:.4f} (임계값 {imp['threshold']}, 이상이면 투표 선택)",
              f"- K·V chunk 읽기 비율(읽은 chunk 수 / layer의 chunk 수, layer·요청 평균): 전체 {100 * imp['kv_chunk_fraction']:.2f}% "
              f"(K {100 * imp['k_chunk_fraction']:.2f}% · V {100 * imp['v_chunk_fraction']:.2f}%)",
              f"- fallback이 아닌 layer만: {fx(None if imp['consensus_kv_chunk_fraction'] is None else 100 * imp['consensus_kv_chunk_fraction'], 2)}%",
              f"- 재배치하지 않았을 경우(raster 순서)의 **계산값**(읽지 않음): 전체 {100 * imp['raster_chunk_fraction_computed']:.2f}% / fallback이 아닌 layer만 "
              f"{fx(None if imp['consensus_raster_chunk_fraction_computed'] is None else 100 * imp['consensus_raster_chunk_fraction_computed'], 2)}%",
              f"- 중요도 중 prefix 행이 차지하는 비율(probe head 3개 × 이미지 토큰, layer·요청 평균): {100 * imp['prefix_share']:.2f}%",
              f"- 다음 layer probe key 미리 읽기: 요청당 {imp['probe_prefetched_reads_mean']:.1f}회 중 {imp['probe_ready_on_arrival_mean']:.2f}회가 사용 시점에 완료", "",
              "계산값은 요청의 layer별 keep mask(투표 선택된 토큰 + separator)를 raster 순서의 위치로 옮겨 64토큰 chunk로 센 것이다. fallback layer는 어느 순서에서나 100%다.", "",
              "| layer | fallback 비율 | 평균 Jaccard | K·V chunk 비율 | raster였다면 (계산값) | prefix 행 비율 |", "|---:|---:|---:|---:|---:|---:|"]
        for x in imp["layers"]:
            L.append(f"| {x['layer']} | {100 * x['fallback_rate']:.2f}% | {x['mean_jaccard']:.4f} | {100 * x['kv_chunk_fraction']:.2f}% | "
                     f"{100 * x['raster_chunk_fraction_computed']:.2f}% | {100 * x['prefix_share']:.2f}% |")
    L += ["", "### 7. 재사용 턴(followup) 시간 분해", "",
          "| 방법 | TTFT 평균 (ms) | 읽기 대기 평균 (ms) | p95 | GPU 적재 평균 (ms) | layer hook 합 (ms) | 워커 읽기 합 (ms) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for d in decomp:
        if d["method"] == "ReComp":
            L.append(f"| ReComp | {d['ttft_mean_ms']:.1f} | — | — | — | — | — |")
        else:
            L.append(f"| {d['method']} | {d['ttft_mean_ms']:.1f} | {d['read_wait_mean_ms']:.2f} | {d['read_wait_p95_ms']:.2f} | {fx(d['gpu_load_mean_ms'], 2)} | "
                     f"{d['hook_mean_ms']:.2f} | {fx(d.get('worker_read_mean_ms'), 2)} |")
    L += ["", "- 읽기 대기: 요청 스레드가 읽기를 기다리며 멈춘 시간(비동기 읽기에서는 워커가 아직 못 읽은 구간을 기다린 시간, IMPRESS의 K·V 읽기는 hook 안의 동기 읽기 시간).",
          "- GPU 적재: Ours는 selector의 scatter 타이머(GPU cache에 행 쓰기 + mask), FullLoad는 layer hook 시간에서 읽기 대기를 뺀 값(전체 행의 host→GPU 복사).",
          "- IMPRESS는 layer hook 안에서 probe 점수 계산·선택·K/V 읽기·적재를 함께 하므로 GPU 적재만 따로 잰 값이 없다"
          + (f"(hook 합 − 읽기 대기 = {next((d['hook_minus_read_wait_mean_ms'] for d in decomp if d['method'] == 'IMPRESS'), 0):.2f} ms가 선택 계산 + 적재)." if any(d["method"] == "IMPRESS" for d in decomp) else "."),
          "- 워커 읽기 합은 연산과 겹쳐 진행되므로 TTFT에 더해지는 값이 아니다. 항목들의 합을 TTFT로 해석하지 않는다.",
          "", "### 8. T1 저장 비용 (이미지당 평균; store는 이미지의 첫 대화 T1에서 한 번 만든다)", "",
          "| 방법 | 저장 크기 (MB) | Visual KV (MB) | probe key 사본 (MB) | 쓰기 (ms) | 재배치 추가 (ms) | 저장 시간 합 (ms) | p95 | OS 기록량 (MB) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in storage:
        L.append(f"| {s['method']} | {s['store_mb_per_image']:.2f} | {s['visual_kv_mb']:.2f} | {s['probe_copy_mb']:.2f} | {s['writer_ms_store_ready_mean']:.1f} | "
                 f"{fx(s['after_request_layout_ms']) if s['method'] == 'IMPRESS' else '—'} | {s['total_store_ms_mean']:.1f} | {s['total_store_ms_p95']:.1f} | {s['os_write_mb_per_image']:.2f} |")
    L += ["", "- ReComp는 store가 없다. \"쓰기\"는 writer가 파일을 쓰고 fsync·rename까지 마친 시간이다. \"재배치 추가\"는 IMPRESS가 T1 통계로 순서를 정해 K·V·probe 파일을 다시 쓰는 시간이다.",
          "- FullLoad store에도 probe key 사본이 들어 있다(11/17 구현의 writer가 같다). FullLoad는 그 파일을 읽지 않는다.",
          "- 실험 하니스가 증거용으로 하는 전체 파일 해시(무결성 봉인)는 위 시간에 넣지 않았다: "
          + " · ".join(f"{s['method']} {fx(s['integrity_hash_ms_harness'])} ms" for s in storage) + ".",
          "- IMPRESS의 중요도 통계 캡처는 T1 요청 안에서 일어난다(이미지당 평균 "
          + fx(next((s["impress_capture_ms_inside_t1"] for s in storage if s["method"] == "IMPRESS"), None)) + " ms, 그 요청의 TTFT에 포함).",
          "- 실험 전체의 SSD 쓰기량은 아래 \"실행 기록\"에 있다.",
          "", f"T1 생성 토큰이 네 방법에서 모두 같은 대화: {t1['identical_across_arms']:,} / {t1['dialogues']:,}."]
    return L


def gate_md(run, manifest):
    n4, nd = len(manifest["phase_images"]["gate"]), len(manifest["phase_images"]["dev"])
    g = {}
    for name in ("tests", "tokens", "legacy", "gate", "devgate", "dev", "admission", "posthoc_main"):
        p = run / "gates" / f"{name}.json"
        g[name] = json.loads(p.read_text()) if p.exists() else None

    def st(doc, *keys):
        for k in keys:
            doc = (doc or {}).get(k)
        return (doc or {}).get("status", "미실행") if isinstance(doc, dict) else "미실행"

    def cmp_txt(doc):
        if not doc:
            return "미실행"
        return f"{doc['requests']:,}건 중 토큰 불일치 {doc['tokens_differ']}건" + (
            ", 읽기 바이트 불일치 " + str(max(doc["fields_differ"].values())) + "건" if doc.get("fields_differ") else "")

    tk, lg, a, dg, d = g["tokens"] or {}, g["legacy"] or {}, g["gate"] or {}, g["devgate"] or {}, g["dev"] or {}
    L = [f"| gate | {n4}장 | DEV {nd}장 | 내용 |", "|---|---|---|---|"]
    g1 = tk.get("gate1_fixed_path_equals_original_llava")
    if g1:
        pm = g1["per_method"]
        txt = (f"기존 본실험이 실제로 보낸 프롬프트 {g1['prompts_total']:,}개(대화·턴 {g1['prompts_per_method_expected']:,}개 × 네 방법의 생성 이력) 중 "
               f"고친 경로 = 원본 LLaVA 절차 {sum(s['fixed_path_equal_original'] for s in pm.values()):,}개 "
               f"(ReComp의 프롬프트 {pm.get(RECOMP, {}).get('prompts', 0):,}개 중 {pm.get(RECOMP, {}).get('fixed_path_equal_original', 0):,}개). "
               f"기존 경로는 {sum(s['legacy_path_equal_original'] for s in pm.values()):,}개 일치")
    else:
        txt = ""
    L.append(f"| 1. 토큰열 = 원본 LLaVA 절차 | {st(g1)} | (공통, CPU) | {txt} |")
    g2 = tk.get("gate2_llava15_path_unchanged")
    L.append(f"| 2. LLaVA-1.5 경로 토큰열 불변 | {st(g2)} | (공통, CPU) | "
             + (f"프롬프트 {g2['compared']}개 중 기록과 같은 것 {g2['suffix_ids_equal_recorded']}개, 그 실행이 동결한 소스 {g2['frozen_files_of_that_run']}개 중 바뀐 파일 {len(g2['frozen_files_changed'])}개" if g2 else "") + " |")
    g3 = lg.get("gate3_legacy_path_reproduces_recomp")
    other = lg.get("reported_other_methods_under_legacy") or {}
    L.append(f"| 3. 기존 경로 옵션으로 11/20 ReComp 재현 | {st(g3)} | — | "
             + (f"ReComp {g3['requests']:,}건 중 토큰 불일치 {g3['tokens_differ']}건" if g3 else "")
             + ("; 참고(판정 아님): " + " · ".join(f"{NAME[m]} {other[m]['requests']:,}건 중 {other[m]['tokens_differ']}건" for m in (FULL, IMP, OURS) if m in other) if other else "") + " |")
    g4a, g4d = a.get("gate4_ours") or {}, dg.get("gate4_ours") or {}
    L.append(f"| 4. Ours 동기 = 비동기, 두 번 실행 | {st(a, 'gate4_ours')} | {st(dg, 'gate4_ours')} | 동기 vs 비동기: 4장 {cmp_txt(g4a.get('sync_vs_async'))} · DEV {cmp_txt(g4d.get('sync_vs_async'))}; "
             f"두 번 실행: 4장 {cmp_txt(g4a.get('executed_twice'))} · DEV {cmp_txt(g4d.get('executed_twice'))} |")
    g5a, g5d = a.get("gate5_impress") or {}, dg.get("gate5_impress") or {}
    L.append(f"| 5. IMPRESS | {st(a, 'gate5_impress')} | {st(dg, 'gate5_impress')} | 유지 100% vs 같은 재배치 store의 FullLoad loader: "
             f"4장 {cmp_txt(g5a.get('retention_100_vs_fullload_loader_on_reordered_store'))} · DEV {cmp_txt(g5d.get('retention_100_vs_fullload_loader_on_reordered_store'))}; "
             f"동기 vs 비동기: 4장 {cmp_txt(g5a.get('sync_vs_async'))} · DEV {cmp_txt(g5d.get('sync_vs_async'))} |")
    g6a, g6d = a.get("gate6_turn1_identical_across_methods"), d.get("gate6_turn1_identical_across_methods")
    L.append(f"| 6. T1 생성 토큰이 네 방법에서 같음 | {st(g6a)} | {st(g6d)} | "
             + " · ".join(f"{lab} 대화 {x['dialogues']:,}개 중 불일치 {x['mismatch_count']}개" for lab, x in (("4장", g6a), ("DEV", g6d)) if x) + " |")
    t = g["tests"]
    if t:
        L.append(f"| 7. 기존 테스트 | {t['status']} | (공통) | " + "; ".join(
            f"transformers {s['transformers']}: {s['ran']}개 실행, 실패 {len(s['failed'])}개(이 작업 전부터 실패하던 것 {len(s['failed']) - len(s['new_failures'])}개), 새 실패 {len(s['new_failures'])}개"
            for s in t["suites"]) + f"; 새 테스트 파일 {t['new_unit_tests']['ran']}개 통과 |")
    else:
        L.append("| 7. 기존 테스트 | 미실행 | | |")
    ex = (d.get("execution") or {})
    L.append(f"| 8. DEV {nd}장 파일럿 | — | {'실행됨' if ex.get('status') == 'PASS' else '미실행'} | "
             + (f"{ex.get('requests', 0):,}건 감사 PASS (판정 항목 아님, 표는 아래)" if ex.get("status") == "PASS" else "") + " |")
    adm = g["admission"]
    L += ["", f"본실험 진입 판정: **{adm['status'] if adm else '미실행'}** (gate 1–7이 모두 PASS이고 DEV 파일럿이 실행됐을 때만 본실험을 시작한다)."]
    if t:
        for s in t["suites"]:
            if s["failed"]:
                L += ["", f"transformers {s['transformers']} 환경에서 이 작업 전부터 실패하던 테스트(목록만 적음): " + "; ".join(f"`{x}`" for x in s["failed"] if x not in s["new_failures"]) + "."]
                if s["new_failures"]:
                    L += ["새로 실패한 테스트: " + "; ".join(f"`{x}`" for x in s["new_failures"]) + "."]
    return L, g


def gate_failure_md(g, status):
    """What failed and the recorded cause, for a run whose main phase was not started."""
    L = []
    adm = g.get("admission") or {}
    steps = adm.get("steps") or {name: {"status": (g.get(name) or {}).get("status", "MISSING"), "ok": (g.get(name) or {}).get("status") == "PASS"}
                                 for name in ("tests", "tokens", "legacy", "gate", "devgate", "dev")}
    for name, step in steps.items():
        if step.get("ok"):
            continue
        doc = g.get(name) or {}
        if step.get("status") in ("MISSING", None):
            L.append(f"- `{name}`: 실행되지 않음(앞 단계가 통과하지 못해 시작하지 않았거나, 요청 단계·검사가 결과를 내지 못했다).")
            continue
        L.append(f"- `{name}`: {step.get('status')}" + (f" — 실패 항목: {', '.join(doc.get('failed') or [])}" if doc.get("failed") else ""))

        def walk(part, depth=0):
            if not isinstance(part, dict) or part.get("status") in ("PASS", "REPORTED") or depth > 2:
                return
            if part.get("criterion"):
                L.append(f"  - 기준: {part['criterion']}")
            counts = {k: part[k] for k in ("requests", "expected", "tokens_differ", "tokens_differ_same_prompt", "prompt_differs", "fields_differ", "missing",
                                           "mismatch_count", "dialogues", "compared", "suffix_ids_equal_recorded", "prompt_rebuilt_equal",
                                           "frozen_files_changed", "prompts_total") if k in part}
            if counts:
                L.append(f"  - 집계: {json.dumps(counts, ensure_ascii=False)}")
            if "per_method" in part:
                L.append("  - 방법별: " + json.dumps({NAME.get(m, m): {k: s[k] for k in ("prompts", "fixed_path_equal_original", "legacy_path_suffix_equal_recorded_suffix")}
                                                    for m, s in part["per_method"].items()}, ensure_ascii=False))
            for x in (part.get("examples") or part.get("mismatches") or part.get("mismatch_examples") or [])[:5]:
                L.append(f"  - 예: {json.dumps(x, ensure_ascii=False)[:700]}")
            for v in part.values():
                if isinstance(v, dict) and "status" in v:
                    walk(v, depth + 1)
        for key in doc.get("failed") or []:
            walk(doc.get(key))
        if name == "tests":
            for s in doc.get("suites", []):
                if s.get("new_failures"):
                    L.append(f"  - transformers {s['transformers']}: 새 실패 {s['new_failures']}")
    err = status.get("last_error")
    if err:
        L.append(f"- 드라이버가 기록한 마지막 오류: `{json.dumps(err, ensure_ascii=False)[:900]}`")
    return L


def settings_md(run, config):
    env = config["environment"]
    return ["## 설정", "",
            f"- 실행: `runs/{run.name}`(요청 단위 원본), `results/{run.name}`(집계·보고서·status.json·driver.log). 계약: `{CONTRACT_DOC}`(사본 `CONTRACT.md`).",
            f"- 환경: Python {env['python'].split()[0]}, torch {env['torch']}, transformers {env['transformers']} (`{env['executable']}`) — 11/17·11/20 실행과 같다(freeze에서 대조).",
            f"- 모델 `{config['model']}` revision `{config['revision'][:8]}`, NF4/BF16/eager, FP16 SSD KV, greedy cap16, batch 1, seed 1234.",
            "- 프롬프트 토큰화: 프롬프트 문자열은 기존과 같고, 토크나이저에 넘기는 문자열만 `<image>` 뒤에 공백 하나를 둔다(`<image> \\n`). 그러면 HF 토크나이저가 "
            "원본 LLaVA 절차(`tokenizer_image_token`, 느린 legacy 토크나이저)와 같은 토큰 ID를 낸다(`<image>` 직후의 `▁` 하나가 더 생긴다). "
            "요청마다 실제로 쓴 토큰 ID와 원본 절차의 토큰 ID를 비교했다. 기존 경로는 옵션(`tokenization=\"legacy\"`)으로 남아 있다.",
            "- 데이터: MT-GQA 398 images / 4,061 dialogues × 3턴, method-local generated history, strict normalized EM, 대화 단위 arm 순서 회전. "
            "모든 방법이 매 대화 T1을 픽셀부터 계산하고(warm-up 없음), 저장 KV 읽기는 요청 직전에 page cache를 비운 cold 조건이다.",
            "- ReComp: 매 턴 이미지부터 다시 계산. FullLoad: Visual KV 전체 저장, 워커가 layer 순서로 전부 읽음(11/17 실행의 구현).",
            "- IMPRESS: 11/17 실행의 구현·설정 그대로(probe head 0·1·2, 전체 행 열 합 중요도, 상위 25%, 평균 Jaccard ≥ 0.3이면 투표 선택 아니면 fallback, "
            "K/V 분리, 64토큰 chunk, 첫 대화 T1 기준 layer별 재배치, 다음 layer probe key 미리 읽기).",
            "- Ours: 11/20 실행의 \"Ours(원래 순서)\" 구현 그대로. image-only CLIP CLS→patch attention 중요도, PYR-beta-raw 벡터(총 KV 예산 25%), "
            "layer l은 상위 k_l개 토큰을 원래 토큰 인덱스 오름차순으로 저장, 재사용 턴에는 각 행을 GPU cache의 원래 위치에 올리고 나머지는 mask. 중요도 순 재배열 없음. 비동기 읽기.", ""]


def execution_md(run, manifest, rec, resume_ttft, done_requests, plan, protection, source_ok, storage, prerun, status):
    L = ["## 실행 기록", ""]
    exp = manifest["expected_requests"]
    L.append("- 요청 수: " + " · ".join(f"{ph} {done_requests.get(ph, 0):,} / {exp[ph]:,}" for ph in PHASES) + ". legacy·gate·devgate·dev 요청은 본실험 분모에 넣지 않았다.")
    for ph in PHASES:
        p = rec["phases"][ph]
        if p["process_starts"]:
            end = p["finished_unix"]
            L.append(f"- {ph}: 프로세스 시작 {p['process_starts']}회 (" + ", ".join(utc(s["at_unix"]) for s in p["starts"]) + ")"
                     + (f", 종료 {utc(end)}, 처음 시작부터 {(end - p['starts'][0]['at_unix']) / 3600:.2f} h" if end else ", 미완료") + ".")
    inv = rec["driver_invocations"]
    resumed = [x for x in inv if x["kind"] == "resume"]
    extra_starts = sum(max(0, rec["phases"][ph]["process_starts"] - 1) for ph in PHASES)
    L.append(f"- **이어서 실행한 횟수**: 드라이버를 다시 실행한 횟수 {len(resumed)}회"
             + (" (" + ", ".join(x["at_utc"] for x in resumed) + ")" if resumed else "")
             + f"; 요청 단계의 프로세스가 다시 시작된 횟수 {extra_starts}회(드라이버를 다시 실행해서 이어진 것과 드라이버가 스스로 다시 시작한 것을 합한 값). "
               f"드라이버 실행은 모두 {len(inv)}회.")
    ev = [e for e in rec["resume_events"] if e.get("mode") in ("continue", "fresh")]
    if ev:
        L += ["- **다시 실행한 대화**(중단 전에 시작됐다가 다시 실행된 대화):", "",
              "| 단계 | 이미지 | 방식 | 유지한 대화 수 | 끊긴 대화 | 다시 실행한 대화 | 버린 기록 (요청 수) | store 확인 |", "|---|---|---|---:|---|---|---:|---|"]
        for e in ev:
            again = e.get("rerun_dialogues")
            if again is None:
                again = e.get("restart_dialogues") or []
            dropped = e.get("dropped_rows") if e["mode"] == "continue" else e.get("rows_found")
            L.append(f"| {e['phase']} | {e['image_id']} | {'끊긴 대화의 T1부터' if e['mode'] == 'continue' else '이미지의 첫 대화부터'} | {e.get('complete_dialogues_kept', 0)} | "
                     f"{e.get('interrupted_dialogue') or '(대화 사이)'} | {len(again)}개: {', '.join(again[:12])}{' …' if len(again) > 12 else ''} | {fx(dropped, 0)} | {e.get('store_check', '')} |")
        L += ["", "끊긴 대화의 불완전한 기록은 결과에서 뺐고(이어 간 경우 `discarded_rows_resume_*.jsonl`, 이미지를 다시 한 경우 `ABANDONED.json`이 있는 attempt 디렉터리에 보관), 그 대화는 T1부터 다시 실행했다. "
                  "\"이미지의 첫 대화부터\"는 끊긴 대화가 store를 만드는 첫 대화였거나 디스크의 store가 기록된 SHA256과 맞지 않은 경우다."]
    else:
        L.append("- 다시 실행한 대화: 없음.")
    if rec["interruptions"]:
        L.append("- 중단 기록: " + "; ".join(f"`{x['file']}` {str(x['error'])[:160]}" for x in rec["interruptions"]))
    if resume_ttft:
        L += ["", "프로세스 시작 직후 요청의 TTFT (본실험, followup; 이어서 실행한 직후의 영향 확인):", "",
              "| 프로세스 | 구분 | 방법 | 처음 N건 평균 (ms) | 첫 요청 (ms) | 나머지 평균 (ms) | 나머지 p95 (ms) | 처음 N건 중 나머지 p95 초과 |", "|---|---|---|---:|---:|---:|---:|---:|"]
        for e in resume_ttft:
            for name, m in e["methods"].items():
                L.append(f"| {e['process_index']} ({e['first_request_utc']}) | {'처음 시작' if e['kind'] == 'first start' else '이어서 실행'} | {name} | "
                         f"{m['first_mean_ms']:.1f} (N={m['first_n']}) | {m['first_request_ms']:.1f} | {m['rest_mean_ms']:.1f} | {m['rest_p95_ms']:.1f} | {m['first_over_p95_of_rest']} |")
    ssd = rec["ssd"]
    store_total = sum(s["os_write_bytes_total"] for s in storage) if storage else None
    expect = (prerun or {}).get("expected_ssd_write", {})
    L += ["", f"- **실험 전체의 SSD 쓰기량**(장치 `nvme0n1`의 쓰기 섹터 수 차이, 다른 프로세스의 쓰기 포함): {ssd['bytes_written_between_samples'] / 1e9:,.1f} GB "
              f"(실행 전 예상 {fx(expect.get('total_tb'), 3)} TB; store 쓰기만 센 값). status.json의 누적값: {fx(status.get('ssd_written_gb'), 1)} GB."
          + (f" 본실험의 store 쓰기(프로세스 I/O 계수 합): {store_total / 1e9:,.1f} GB." if store_total else ""),
          "- SMART(percentage used, data units written)는 root 권한이 필요해 읽지 못했다(미실행). `sudo nvme smart-log /dev/nvme0` 출력을 "
          f"`results/{run.name}/smart_manual_after.txt`로 두면 값을 확인할 수 있다."]
    if ssd["by_driver_stage"]:
        L.append("- 단계별 SSD 쓰기량: " + " · ".join(f"{s['until']} {s['bytes'] / 1e9:,.1f} GB" for s in ssd["by_driver_stage"]) + ".")
    L += [f"- 디스크 여유: freeze 시점 {plan['free_bytes_at_freeze'] / 2**30:,.1f} GiB, 보고 시점 {shutil.disk_usage(run).free / 2**30:,.1f} GiB.",
          f"- 기존 `results/`·`runs/`·`kvstore*` 파일 목록(경로, 크기, mtime): 실행 전 {protection['before']['files']:,}개 → 보고 시점 {protection['after']['files']:,}개, "
          + ("**변화 없음**." if protection["unchanged"] else "**목록 지문이 다르다** — `protection_before.tsv.gz` / `protection_after.tsv.gz`를 대조해야 한다(다른 작업이 파일을 만들었을 수 있다)."),
          f"- freeze 때 기록한 소스 파일: " + ("모두 그대로다." if source_ok else "**바뀐 파일이 있다**(다음 단계는 FROZEN_SOURCE_CONFIG_MISMATCH로 멈춘다).")]
    return L


CODE_AND_LIMITS = """## 9. 추가하거나 바꾼 코드, 지시와 다르게 했거나 지시에 없던 것, 측정하지 않은 것과 한계

### 추가한 코드 (기존 파일은 하나도 수정하지 않았다)

- `mmimpress/mt_gqa_tokens.py` — 토큰열 옵션. `tokenization="official"`은 토크나이저에 넘기는 문자열의 `<image>` 뒤에 공백 하나를 두어 HF 토크나이저가 원본 LLaVA 절차의 토큰 ID를 내게 하고,
  `"legacy"`는 기존 경로 그대로다. 원본 절차의 토큰 ID 계산(`official_token_ids`)과 비교 함수가 함께 있다. ConvBench 정렬에서 쓴 렌더링(`scripts/213`의 `OFFICIAL_RENDER["next"]`)과 같은 방식이다.
- `scripts/222_eval_mt_gqa_main_v2.py` — 실행기. `scripts/209`(= 11/17의 `scripts/196` + 11/20의 `scripts/202` 호출)를 import해 요청·캡처·저장·활성화 함수를 그대로 부른다.
  새로 쓴 부분은 프롬프트를 토크나이저 입력으로 바꾸는 한 줄, 요청 뒤 토큰 ID 비교, MT-GQA용 이미지 감사, 대화 단위 이어 실행, status.json 갱신이다.
- `scripts/223_validate_mt_gqa_main_v2.py`(gate), `scripts/224_report_mt_gqa_main_v2.py`(이 보고서, 측정에 관여하지 않아 동결 대상이 아님), `scripts/run_mt_gqa_main_v2.sh`(드라이버),
  `tests/test_mt_gqa_main_v2.py`, `docs/mt_gqa_main_v2_contract.md`.
- 방법 코드(`mmimpress/serve.py`, `packed_serve.py`, `packed_store.py`, `piggyback.py`, `impress_official.py`)와 LLaVA-1.5 경로(`mmimpress/llava15/`), 기존 스크립트·테스트·store·결과 디렉터리·동결 파일은 수정하지 않았다.
  freeze가 11/17·11/20 실행의 소스 해시와 대조했고, gate 2가 LLaVA-1.5 본실험이 동결한 파일이 모두 그대로임을 확인했다.

### 지시와 다르게 했거나 지시에 없던 것

- **gate 1의 프롬프트 집합.** 본실험의 T2·T3 프롬프트는 그 실행에서 각 방법이 생성한 답에 따라 정해지므로 본실험 전에는 알 수 없다. 그래서 본실험 전 gate 1은 기존 본실험(11/20, 11/17)이
  실제로 보낸 프롬프트 — 대화·턴 12,183개 × 네 방법의 생성 이력 = 48,732개 — 로 판정했고, 본실험에서는 요청마다(48,732건) 실제로 쓴 토큰 ID를 원본 절차의 토큰 ID와 비교해 하나라도 다르면 멈추게 했다.
  본실험 요청 전체의 일치 건수는 위 gate 표 아래에 적었다.
- **gate 2의 판정 내용.** LLaVA-1.5 경로의 코드를 건드리지 않았으므로, 프롬프트 60개의 토큰 ID가 LLaVA-1.5 본실험(11/20)의 기록과 같은지와 그 실행이 동결한 소스 파일이 모두 그대로인지를 함께 판정했다.
- **gate 3의 단계.** 기존 경로 옵션 단계(`legacy`)는 네 방법을 모두 돌렸다. 판정은 지시대로 ReComp만 했고 나머지 세 방법의 재현 여부는 참고로 적었다.
- **gate 4·5·6의 DEV 40장 판정.** 4장에서 판정한 뒤 DEV 40장에서도 판정했다(gate 4·5는 별도 단계 `devgate`의 7 arm, gate 6은 파일럿의 4 arm). gate 판정용 arm(동기 읽기, 두 번째 실행, 유지 100%, FullLoad loader)은 본실험 arm이 아니다.
- **결과 디렉터리.** 보고서·status.json·driver.log·PID 파일은 `results/mt_gqa_main_v2_<UTC>/`에, 요청 단위 원본과 증거 파일은 기존 실행들과 같이 `runs/mt_gqa_main_v2_<UTC>/`에 있다.
- **이어 실행.** 끊긴 대화가 이미지의 첫 대화(그 이미지의 store를 만드는 대화)였거나 디스크의 store가 만들 때 기록한 SHA256과 맞지 않으면 그 이미지를 첫 대화부터 다시 한다(그 이미지에서 이미 끝난 대화도 다시 실행된다).
  요청 단계가 오류로 끝나면 드라이버가 같은 실행 안에서 최대 3회까지 스스로 다시 시작한다. 토큰 ID 불일치·이미지 감사 실패·디스크 부족처럼 다시 해도 같은 곳에서 멈출 오류는 다시 시작하지 않는다.
- 드라이버는 GPU를 쓰는 단계 전에 다른 프로세스가 GPU를 쓰고 있으면 비워질 때까지 기다린다(30초 간격, 최대 2시간).
- 모델을 올린 직후 합성 이미지 요청 1건으로 프로세스를 예열한다(기존 실행과 같음, 모든 지표에서 제외). 저장 KV의 warm-up은 없다.
- TTFT 비율에 이미지 단위 bootstrap 95% CI를 함께 적었다. 기존 결과와의 대조(4절)에 EM 차이의 이미지 단위 paired bootstrap CI를 더했다.
- gate 7에서 "이 작업 전부터 실패하던 테스트"로 본 것은 transformers 4.57.6 환경의 GLM-4.6V 테스트 3개와 5.13.1 환경의 4개다. 5.13.1의 4개 중 2개(`tests/test_convbench_main.py`, 11/22 작성)는
  기존 실행들의 목록에 없던 것으로, 실행 환경이 4.57.6 환경임을 assert하거나 4.x의 느린 토크나이저 동작에 기대는 테스트다. 이 작업의 새 파일을 모두 치운 상태에서도 같은 실패가 나는 것을 확인하고 목록에 넣었다.
- freeze 전에 개발 리허설(단계당 이미지 1장, 측정 아님)을 두 번 돌려 스크립트와 이어 실행을 확인했고, 그 디렉터리는 삭제했다. 리허설에서 미리 본 것은 계약 9절 14번에 적었다.
- status.json의 예상 종료 시각은 근사값이다.

### 측정하지 않은 것과 한계

- 본실험은 1회 실행이다. 반복 실행에 따른 변동은 재지 않았다.
- 토큰 하나가 더 들어간 것 외의 요인이 기존 실행과의 차이에 섞였는지는 4장(gate 3)에서만 확인했다: 기존 경로 옵션으로 4장의 ReComp 생성 토큰이 11/20과 같았다. 398장 전체를 기존 경로로 다시 돌리지는 않았다.
- 기존 실행과의 TTFT 대조는 서로 다른 날의 값이다. 같은 실행 안에서 두 토큰열의 TTFT를 비교하지 않았다.
- MetaCompress와의 대조는 수치를 나란히 적은 것이다. 대화 파일·프롬프트·채점이 같다는 보장이 없고, MetaCompress의 값을 이 환경에서 재현하지 않았다.
- LLaVA-NeXT × MT-VQA 경로에도 같은 토큰 결함이 있다(ConvBench 본실험 0단계). 그 경로는 이 작업에서 고치지도 다시 돌리지도 않았다.
- SMART 수준의 SSD 마모(percentage used, NAND 쓰기량)는 root 권한이 없어 읽지 못했다. SSD 쓰기량은 커널의 장치 단위 쓰기 섹터 수다.
- IMPRESS의 "재배치하지 않았을 경우" chunk 비율은 계산값이며 실제로 읽어서 잰 값이 아니다. raster 순서의 IMPRESS store로 TTFT를 재지 않았다.
- warm cache, 25% 외 예산, 다른 벡터, O_DIRECT·io_uring, 워커 여러 개는 재지 않았다.
"""


def dev_md(run, manifest):
    records, _ = collect(run, "dev", manifest, METHODS, allow_partial=False)
    arms = R.arm_table(records, METHODS)
    paired = R.paired_table(records, PAIRS)
    ratios = R.ratio_table(records, RATIOS)
    L = ["| 방법 | Followup EM (%) | T1 EM | T2 EM | T3 EM | TTFT 평균 (ms) | p50 | p95 | SSD MB/followup | 읽기 호출 수 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a in arms:
        L.append(f"| {a['method']} | {a['em_followup']:.3f} | {a['em_T1']:.3f} | {a['em_T2']:.3f} | {a['em_T3']:.3f} | {a['followup_ttft_mean_ms']:.1f} | "
                 f"{a['followup_ttft_p50_ms']:.1f} | {a['followup_ttft_p95_ms']:.1f} | {a['ssd_mb_per_followup']:.2f} | {a['read_calls_per_followup']:.1f} |")
    L += ["", "| 비교 | EM 차이 (pp) [95% CI] | TTFT 차이 (ms) [95% CI] |", "|---|---:|---:|"]
    for p in paired:
        if "em_diff_pp" in p:
            L.append(f"| {p['comparison']} | {p['em_diff_pp']:+.3f} [{p['em_ci_low']:+.3f}, {p['em_ci_high']:+.3f}] | {p['ttft_diff_ms']:+.1f} [{p['ttft_ci_low']:+.1f}, {p['ttft_ci_high']:+.1f}] |")
    L += ["", "TTFT 비율: " + " · ".join(f"{r['ratio']} {fx(r['ttft_ratio_of_means'], 3)}×" for r in ratios) + ".",
          f"DEV {len(records)}장, followup {arms[0]['followup_requests']:,}건/방법. 파일럿은 구현 확인용이며 본실험의 arm이나 설정을 고르는 데 쓰지 않았다."]
    return L, {"arms": arms, "paired": paired, "ratios": ratios}


def protection_check(run, out_root, config):
    E = load_script("_mtgqa_main_v2_eval_for_report", "222_eval_mt_gqa_main_v2.py")
    before = json.loads((run / "protection_before.json").read_text())
    after = E.stat_inventory(run, out_root / "protection_after.tsv.gz")
    protection = {"before": before, "after": after, "unchanged": before["sha256"] == after["sha256"] and before["files"] == after["files"]}
    source_ok = not E.freeze_violations(config["source_sha256"])
    return protection, source_ok


def committed_counts(run):
    out = {}
    for ph in PHASES:
        out[ph] = sum(json.loads(cp.read_text())["requests"] for cp in (run / ph / "images").glob("*/COMMITTED.json")) if (run / ph).exists() else 0
    return out


def report_auto(run, manifest, config, out_root):
    out_root.mkdir(parents=True, exist_ok=True)
    gate_lines, g = gate_md(run, manifest)
    admitted = (g.get("admission") or {}).get("status") == "PASS"
    main_done = (run / "main_execution_complete.json").exists()
    counts = committed_counts(run)
    status = {}
    try:
        status = json.loads((out_root / "status.json").read_text())
    except (OSError, ValueError):
        pass
    prerun = None
    try:
        prerun = json.loads((out_root / "prerun.json").read_text())
    except (OSError, ValueError):
        pass
    lines = ["# LLaVA-NeXT-7B × MT-GQA 본실험 v2 (공식 토큰열): ReComp / FullLoad / IMPRESS / Ours", "",
             f"작성 {utc(time.time())} · 실행 `{run.name}` · 측정 보고(GO/NO-GO 판정 없음)", ""]
    summary = {"run": run.name, "admission": (g.get("admission") or {}).get("status"), "main_complete": main_done, "committed_requests": counts,
               "written_utc": utc(time.time())}
    if main_done:
        lines += [f"**상태: 본실험 완료** — {counts['main']:,} / {manifest['expected_requests']['main']:,} 요청.", ""]
    elif admitted:
        lines += [f"**상태: 본실험 미완료** — {counts['main']:,} / {manifest['expected_requests']['main']:,} 요청이 commit됐다. 아래 본실험 표는 commit된 이미지만으로 만든 **중간값**이다. "
                  f"이어서 실행: `{DRIVER} runs/{run.name}`.", ""]
    else:
        lines += ["**상태: 본실험을 시작하지 않았다** — gate가 통과하지 않았거나 gate 단계가 끝나지 않았다. 실패 내용과 원인은 아래에 있다. 기준은 바꾸지 않았다.", ""]
    if status.get("last_error"):
        lines += [f"마지막 오류(status.json): `{json.dumps(status['last_error'], ensure_ascii=False)[:900]}`", ""]
    lines += ["## Gate", ""] + gate_lines + [""]
    post = g.get("posthoc_main")
    if post:
        tr, t1p = post["token_record_main"], post["turn1_identity_main"]
        lines += [f"본실험 요청의 토큰열 확인(요청마다 실제로 쓴 토큰 ID = 원본 LLaVA 절차의 토큰 ID): {tr['requests_whose_ids_equal_original_llava']:,} / {tr['requests']:,}건 일치. "
                  f"이미지 감사: {json.dumps(post['image_audits'], ensure_ascii=False)}. 본실험에서 T1 생성 토큰이 네 방법에서 다른 대화: {t1p['mismatch_count']:,} / {t1p['dialogues']:,}.", ""]
    if not admitted:
        lines += ["### 실패 내용과 원인", ""] + gate_failure_md(g, status) + [""]
    storage = []
    records = []
    if admitted and counts["main"]:
        records, total_images = collect(run, "main", manifest, METHODS, allow_partial=not main_done)
        arms = R.arm_table(records, METHODS)
        paired = R.paired_table(records, PAIRS)
        ratios = R.ratio_table(records, RATIOS)
        imp = impress_internal(records)
        storage = R.storage_table(records, METHODS)
        decomp = R.decomposition(records, METHODS)
        t1 = R.t1_identity(records, METHODS)
        previous = previous_comparison(run, records)
        meta = metacompress_table(arms, previous)
        tok = {"requests": sum(r["requests"] for r in records), "ids_equal_original": sum(r["token_match"] for r in records),
               "tokenizations": sorted({t for r in records for t in r["tokenization"]})}
        lines += [f"## 본실험 결과 ({len(records):,} / {total_images:,} images, 방법당 followup {arms[0]['followup_requests']:,}건"
                  + ("" if main_done else "; **중간값**") + ")", ""]
        lines += main_tables_md(arms, paired, ratios, imp, storage, decomp, t1, previous, meta) + [""]
        lines += [f"토큰열: 이 표의 요청 {tok['requests']:,}건 중 {tok['ids_equal_original']:,}건에서 실제로 쓴 토큰 ID가 원본 LLaVA 절차의 토큰 ID와 같았다.", ""]
        write_csv(out_root / "main_arms.csv", arms); write_csv(out_root / "main_paired.csv", paired); write_csv(out_root / "main_ttft_ratios.csv", ratios)
        write_csv(out_root / "main_storage.csv", storage); write_csv(out_root / "main_followup_decomposition.csv", decomp)
        write_csv(out_root / "main_impress_layers.csv", imp["layers"])
        if previous:
            write_csv(out_root / "main_vs_previous.csv", [{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in p.items()} for p in previous["methods"]])
        summary.update(arms=arms, paired=paired, ratios=ratios, impress={k: v for k, v in imp.items() if k != "layers"}, storage=storage,
                       decomposition=decomp, t1_identity=t1, previous=previous, metacompress=meta, token_record=tok, images=len(records))
    if (run / "dev_execution_complete.json").exists():
        dev_lines, dev_summary = dev_md(run, manifest)
        lines += ["## DEV 40장 파일럿 (gate 8; 본실험 표가 아님)", ""] + dev_lines + [""]
        summary["dev"] = dev_summary
    lines += settings_md(run, config)
    rec = R.execution_record(run, manifest)
    plan = json.loads((run / "storage_plan.json").read_text())
    protection, source_ok = protection_check(run, out_root, config)
    resume_ttft = R.after_start_ttft(records, rec) if records else []
    lines += execution_md(run, manifest, rec, resume_ttft, counts, plan, protection, source_ok, storage, prerun, status) + [""]
    if prerun:
        d = prerun["expected_duration"]
        lines += ["## 실행 전 예상 (prerun.json)", "",
                  f"- 예상 SSD 쓰기량 {prerun['expected_ssd_write']['total_tb']:.3f} TB(본실험 {prerun['expected_ssd_write']['main_tb']:.3f} TB), 예상 소요 {d['total_hours']:.1f} h, "
                  f"디스크 여유 {prerun['free_space']['free_gib']:.1f} GiB, SMART {prerun['smart']['status']}.", ""]
    lines += [CODE_AND_LIMITS]
    notes = out_root / "notes.md"      # optional hand-written additions; appended verbatim
    if notes.exists():
        lines += ["", "## 추가 메모", "", notes.read_text().rstrip(), ""]
    summary.update(execution=rec, protection=protection, frozen_source_unchanged=source_ok)
    (out_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_root / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    if (run / "gates").is_dir():
        shutil.copytree(run / "gates", out_root / "gates", dirs_exist_ok=True)
    print(json.dumps({"report": str((out_root / "report.md").relative_to(ROOT)), "admission": summary["admission"], "main_complete": main_done,
                      "committed": counts}, ensure_ascii=False))


def report_dev(run, manifest, out_root):
    out = out_root / "dev"; out.mkdir(parents=True, exist_ok=True)
    lines, summary = dev_md(run, manifest)
    write_csv(out / "dev_arms.csv", summary["arms"]); write_csv(out / "dev_paired.csv", summary["paired"])
    text = ["# LLaVA-NeXT-7B × MT-GQA v2 — DEV 40장 파일럿 (gate 8)", "", f"작성 {utc(time.time())} · 실행 `{run.name}` · 본실험 표가 아니다.", ""] + lines
    (out / "report_dev.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--phase", choices=("dev", "auto"), default="auto")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    config = json.loads((run / "config.json").read_text())
    out_root = args.out.resolve() if args.out is not None else ROOT / "results" / run.name
    if args.phase == "dev":
        return report_dev(run, manifest, out_root)
    report_auto(run, manifest, config, out_root)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Tables and report of the LLaVA-1.5 x MT-GQA main experiment: ReComp / FullLoad / IMPRESS / Ours
(contract docs/llava15_mt_gqa_main_contract.md, section 9).

Reads committed rows only; takes no part in any measurement (not in the source freeze).

  --phase dev    the DEV pilot table (gate 8) incl. the reference comparison with the reordered packed arm
  --phase auto   results/<run>/report.md: the main tables when the main phase is complete, an interim main
                 table when it was admitted but is not complete, otherwise the gate outcome (and the DEV
                 pilot table if it exists)

A measurement report: no GO / NO-GO verdict is produced.
"""
from __future__ import annotations

import argparse
import csv
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

RECOMP, FULL, IMP = "recompute", "fullload_async", "impress_official_async"
OURS = "llava15_kv_PYR-b20_origorder_packed_async"
REORDER = "llava15_kv_PYR-b20_packed_async"
METHODS = [RECOMP, FULL, IMP, OURS]
NAME = {RECOMP: "ReComp", FULL: "FullLoad", IMP: "IMPRESS", OURS: "Ours", REORDER: "Ours(재배열)"}
KIND = {FULL: "raster", IMP: "impress_official", OURS: "packedorig_PYR-b20", REORDER: "packed_PYR-b20"}
PAIRS = [(OURS, IMP), (OURS, FULL), (OURS, RECOMP), (IMP, FULL), (FULL, RECOMP)]
RATIOS = [(OURS, FULL), (OURS, RECOMP), (OURS, IMP), (IMP, FULL)]
EVAL = "206_eval_llava15_mt_gqa_main.py"
CONTRACT_DOC = "docs/llava15_mt_gqa_main_contract.md"
PHASES = ("gate", "devgate", "dev", "main")
RESAMPLES, SEED = 10000, 1234
AFTER_RESUME_REQUESTS = 40      # follow-up requests per method looked at right after a process start
MB = 1e6


# ------------------------------------------------------------- per image io
def image_record(args):
    """One committed image -> compact record (worker; never raises)."""
    folder, methods = Path(args[0]), args[1]
    try:
        commit = json.loads((folder / "COMMITTED.json").read_text())
        attempt = folder / commit["attempt_id"]
        rows = read_rows(attempt / "raw.jsonl")
        out = {"image_id": folder.name, "status": "OK", "requests": len(rows), "rows": [], "resumed_inside_image": commit.get("resumed_inside_image", 0),
               "t1_identical": commit.get("t1_tokens_identical_across_arms"), "attempt": commit["attempt_id"]}
        for r in rows:
            if r["method_id"] not in methods:
                continue
            res = r["result"]
            c = {"m": r["method_id"], "d": r["dialog_id"], "t": r["turn_id"], "score": r["score"], "ttft": r["ttft_ms"], "e2e": r["request_e2e_ms"],
                 "bytes": r["ssd_read_bytes"], "reads": r["ssd_preads"], "probe": r["probe_read_bytes"], "k": r["k_read_bytes"], "v": r["v_read_bytes"],
                 "packed": r["packed_read_bytes"], "blocked": r["read_blocked_ms"], "prefill": res.get("prefill_ms"), "hook": res.get("hook_ms"),
                 "scatter": res.get("scatter_ms"), "pid": r.get("process_id"), "at": r.get("completed_at_unix"), "tok": r["generated_token_ids"],
                 "pos": r["method_order_position"]}
            ahead = (res.get("read_wait") or {}).get("readahead") or {}
            if ahead:
                c["worker"] = ahead.get("worker_read_ms"); c["ready"] = ahead.get("ready_on_arrival")
            if r["turn_id"] > 1 and r["family"] == "impress_official":
                log = res["layer_log"]
                c["imp"] = {"fb": [int(x["mode"] == "fallback") for x in log], "sim": [x["sim"] for x in log],
                            "kc": [x["k_chunks"] for x in log], "vc": [x["v_chunks"] for x in log],
                            "share": [d["prefix_share"] for d in res["layer_diagnostics"]],
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
        eq = attempt / "store_equivalence.json"
        if eq.exists():
            out["store_equivalence"] = json.loads(eq.read_text())
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
def cells(records, method, turns=(2, 3)):
    return [r["by"][(method, d, t)] for r in records for d in r["dialogs"] for t in turns]


def arm_table(records, methods):
    rows = []
    for m in methods:
        sc = {t: np.array([c["score"] for c in cells(records, m, (t,))]) for t in (1, 2, 3)}
        fu = cells(records, m)
        ttft = np.array([c["ttft"] for c in fu]); e2e = np.array([c["e2e"] for c in fu])
        t1 = np.array([c["ttft"] for c in cells(records, m, (1,))])
        by = {k: np.array([c[k] for c in fu], dtype=float) for k in ("bytes", "probe", "k", "v", "packed", "reads")}
        rows.append({"method": NAME[m], "method_id": m, "images": len(records), "dialogues": int(sc[1].size), "followup_requests": len(fu),
                     "em_T1": 100 * sc[1].mean(), "em_T2": 100 * sc[2].mean(), "em_T3": 100 * sc[3].mean(),
                     "em_followup": 100 * np.concatenate([sc[2], sc[3]]).mean(), "em_all": 100 * np.concatenate([sc[1], sc[2], sc[3]]).mean(),
                     "followup_ttft_mean_ms": ttft.mean(), "followup_ttft_p50_ms": np.percentile(ttft, 50), "followup_ttft_p95_ms": np.percentile(ttft, 95),
                     "ssd_mb_per_followup": by["bytes"].mean() / MB, "read_calls_per_followup": by["reads"].mean(),
                     "probe_mb": by["probe"].mean() / MB, "k_mb": by["k"].mean() / MB, "v_mb": by["v"].mean() / MB, "packed_mb": by["packed"].mean() / MB,
                     "read_calls_min": by["reads"].min(), "read_calls_max": by["reads"].max(),
                     "followup_e2e_mean_ms": e2e.mean(), "t1_ttft_mean_ms": t1.mean()})
    return rows


def per_image(records, a, b, key, scale=1.0):
    """[(sum_a, sum_b, count)] per image over the follow-up requests."""
    out = []
    for r in records:
        sa = sum(r["by"][(a, d, t)][key] for d in r["dialogs"] for t in (2, 3)) * scale
        sb = sum(r["by"][(b, d, t)][key] for d in r["dialogs"] for t in (2, 3)) * scale
        out.append((sa, sb, 2 * len(r["dialogs"])))
    return out


def ratio_bootstrap(per):
    """Image-cluster bootstrap of mean(a) / mean(b) (the resampling of the paired difference)."""
    arr = np.asarray(per, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(arr), size=(RESAMPLES, len(arr)))
    s = arr[idx]
    boot = s[..., 0].sum(1) / s[..., 1].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"ratio": float(arr[:, 0].sum() / arr[:, 1].sum()), "ci_low": float(lo), "ci_high": float(hi)}


def paired_table(records, pairs):
    out = []
    for a, b in pairs:
        em = paired_bootstrap(per_image(records, a, b, "score", 100.0), RESAMPLES, SEED) if len(records) > 1 else None
        tt = paired_bootstrap(per_image(records, a, b, "ttft"), RESAMPLES, SEED) if len(records) > 1 else None
        row = {"comparison": f"{NAME[a]} − {NAME[b]}", "images": len(records)}
        if em:
            row.update(em_diff_pp=em["mean"], em_ci_low=em["ci_low"], em_ci_high=em["ci_high"],
                       em_ci_excludes_zero=not (em["ci_low"] <= 0 <= em["ci_high"]),
                       ttft_diff_ms=tt["mean"], ttft_ci_low=tt["ci_low"], ttft_ci_high=tt["ci_high"],
                       basis=f"same run, image-cluster paired bootstrap {RESAMPLES:,}, seed {SEED}, follow-up (T2-T3)")
        out.append(row)
    return out


def ratio_table(records, ratios):
    out = []
    for a, b in ratios:
        rb = ratio_bootstrap(per_image(records, a, b, "ttft")) if len(records) > 1 else {"ratio": None, "ci_low": None, "ci_high": None}
        out.append({"ratio": f"{NAME[a]} / {NAME[b]}", "ttft_ratio_of_means": rb["ratio"], "ci_low": rb["ci_low"], "ci_high": rb["ci_high"]})
    return out


def impress_internal(records):
    imp = [c["imp"] for c in cells(records, IMP)]
    fb = np.array([x["fb"] for x in imp], dtype=float); sim = np.array([x["sim"] for x in imp], dtype=float)
    kc = np.array([x["kc"] for x in imp], dtype=float); vc = np.array([x["vc"] for x in imp], dtype=float)
    share = np.array([x["share"] for x in imp], dtype=float)
    nc = 9.0
    layers = [{"layer": li, "fallback_rate": fb[:, li].mean(), "mean_jaccard": sim[:, li].mean(), "k_chunk_fraction": kc[:, li].mean() / nc,
               "v_chunk_fraction": vc[:, li].mean() / nc, "prefix_share": share[:, li].mean()} for li in range(fb.shape[1])]
    return {"followup_requests": int(fb.shape[0]), "fallback_rate": float(fb.mean()), "mean_jaccard": float(sim.mean()), "threshold": 0.3,
            "k_chunk_fraction": float(kc.mean() / nc), "v_chunk_fraction": float(vc.mean() / nc), "kv_chunk_fraction": float((kc.mean() + vc.mean()) / (2 * nc)),
            "consensus_kv_chunk_fraction": float((((kc + vc) / (2 * nc))[fb == 0]).mean()) if (fb == 0).any() else None,
            "prefix_share": float(share.mean()),
            "requests_with_any_fallback": float((fb.sum(axis=1) > 0).mean()), "requests_all_layers_fallback": float((fb.sum(axis=1) == fb.shape[1]).mean()),
            "probe_prefetched_reads_mean": float(np.mean([x["prefetched"] for x in imp])), "probe_ready_on_arrival_mean": float(np.mean([x["ready"] for x in imp])),
            "layers": layers}


def storage_table(records, methods):
    out = []
    for m in methods:
        s = [r["store"][m] for r in records if m in r["store"]]
        if not s:
            continue

        def mean(key):
            v = [x[key] for x in s if x.get(key) is not None]
            return float(np.mean(v)) if v else None
        out.append({"method": NAME[m], "store_kind": KIND[m], "images": len(s), "store_mb_per_image": mean("logical_bytes") / MB,
                    "visual_kv_mb": mean("kv_bytes") / MB, "probe_copy_mb": mean("probe_bytes") / MB,
                    "writer_ms_store_ready_mean": mean("store_ready_ms"), "writer_ms_store_ready_p95": float(np.percentile([x["store_ready_ms"] for x in s], 95)),
                    "after_request_layout_ms": mean("after_request_ms"), "total_store_ms_mean": mean("total_store_ms"),
                    "total_store_ms_p95": float(np.percentile([x["total_store_ms"] for x in s], 95)),
                    "impress_capture_ms_inside_t1": mean("capture_ms"), "row_select_ms": mean("repack_ms"), "row_plan_ms": mean("row_plan_ms"),
                    "integrity_hash_ms_harness": mean("hash_ms"), "service_persistence_ms_incl_hash": mean("service_ms"),
                    "activation_ms": mean("activation_ms"), "os_write_mb_per_image": mean("os_write_bytes") / MB,
                    "os_write_bytes_total": int(sum(x["os_write_bytes"] for x in s))})
    return out


def decomposition(records, methods):
    """Follow-up turns: time the request thread waited for reads, and time spent loading rows to the GPU."""
    out = []
    for m in methods:
        fu = cells(records, m)
        row = {"method": NAME[m], "followup_requests": len(fu), "ttft_mean_ms": float(np.mean([c["ttft"] for c in fu])),
               "prefill_mean_ms": float(np.mean([c["prefill"] for c in fu]))}
        if m != RECOMP:
            blocked = np.array([c["blocked"] for c in fu], dtype=float)
            hook = np.array([c["hook"] for c in fu], dtype=float)
            row.update(read_wait_mean_ms=float(blocked.mean()), read_wait_p95_ms=float(np.percentile(blocked, 95)),
                       hook_mean_ms=float(hook.mean()), hook_minus_read_wait_mean_ms=float((hook - blocked).mean()))
            if all(c.get("scatter") is not None for c in fu):
                row["gpu_load_mean_ms"] = float(np.mean([c["scatter"] for c in fu]))
                row["gpu_load_source"] = "scatter timer of the packed selector (index write to the GPU cache + mask)"
            elif m == FULL:
                row["gpu_load_mean_ms"] = row["hook_minus_read_wait_mean_ms"]
                row["gpu_load_source"] = "layer hook time minus read wait (host FP16 -> GPU BF16 copy of every row)"
            else:
                row["gpu_load_mean_ms"] = None
                row["gpu_load_source"] = "not separable: the layer hook time also holds the probe scoring and the selection"
            if all(c.get("worker") is not None for c in fu):
                row["worker_read_mean_ms"] = float(np.mean([c["worker"] for c in fu]))
            if m == IMP:
                row["probe_wait_mean_ms"] = float(np.mean([c["imp"]["blocked_probe"] for c in fu]))
        out.append(row)
    return out


def t1_identity(records, methods):
    n = same = 0
    for r in records:
        for d in r["dialogs"]:
            n += 1
            same += len({tuple(r["by"][(m, d, 1)]["tok"]) for m in methods}) == 1
    return {"dialogues": n, "identical_across_arms": same, "mismatched": n - same}


def order_reference(records):
    """DEV reference: Ours vs the reordered packed arm (reported only)."""
    em = paired_bootstrap(per_image(records, OURS, REORDER, "score", 100.0), RESAMPLES, SEED)
    tt = paired_bootstrap(per_image(records, OURS, REORDER, "ttft"), RESAMPLES, SEED)
    rb = ratio_bootstrap(per_image(records, OURS, REORDER, "ttft"))
    a, b = cells(records, OURS), cells(records, REORDER)
    return {"images": len(records), "followup_requests": len(a),
            "em_ours": 100 * float(np.mean([c["score"] for c in a])), "em_reordered": 100 * float(np.mean([c["score"] for c in b])),
            "em_diff_pp": em["mean"], "em_ci": [em["ci_low"], em["ci_high"]],
            "ttft_ours_ms": float(np.mean([c["ttft"] for c in a])), "ttft_reordered_ms": float(np.mean([c["ttft"] for c in b])),
            "ttft_diff_ms": tt["mean"], "ttft_ci": [tt["ci_low"], tt["ci_high"]], "ttft_ratio": rb["ratio"], "ttft_ratio_ci": [rb["ci_low"], rb["ci_high"]],
            "tokens_differ": sum(x["tok"] != y["tok"] for x, y in zip(a, b)),
            "bytes_or_read_calls_differ": sum(x["bytes"] != y["bytes"] or x["reads"] != y["reads"] for x, y in zip(a, b)),
            "store_rows_compared": sum(r.get("store_equivalence", {}).get("kv_rows_compared", 0) for r in records),
            "store_rows_different": sum(r.get("store_equivalence", {}).get("kv_rows_different", 0) for r in records)}


# ------------------------------------------------------- execution records
def jsonl(path):
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def utc(ts):
    return time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(ts))


def execution_record(run, manifest):
    starts = {}
    for ph in PHASES:
        files = sorted(run.glob(f"{ph}_start_*.json"))
        docs = [json.loads(f.read_text()) for f in files]
        done = run / f"{ph}_execution_complete.json"
        starts[ph] = {"process_starts": len(docs), "starts": [{"at_unix": d["started_unix"], "pid": d["pid"], "resume": d.get("resume"), "boot_id": d.get("boot_id"),
                                                               "committed_requests_at_start": d.get("committed_requests_at_start")} for d in docs],
                      "finished_unix": json.loads(done.read_text())["finished_unix"] if done.exists() else None}
    invocations = jsonl(run / "driver_invocations.jsonl")
    resumes = jsonl(run / "resume_log.jsonl")
    interruptions = []
    for f in sorted(run.glob("interruption_*.json")):
        d = json.loads(f.read_text())
        interruptions.append({"file": f.name, "error": d.get("error"), "at_unix": d.get("at_unix")})
    samples = [s for s in jsonl(run / "ssd_samples.jsonl") if "sectors_written" in s]
    by_boot = {}
    for s in samples:
        by_boot.setdefault(s["boot_id"], []).append(s["sectors_written"])
    written = sum((max(v) - min(v)) * 512 for v in by_boot.values())
    stages = []; prev = None      # bytes written between consecutive driver-level samples of the same boot
    for s in samples:
        if prev is not None and prev["boot_id"] == s["boot_id"] and s["label"] != "image_committed" and not s["label"].endswith("_process_start"):
            stages.append({"until": s["label"], "bytes": (s["sectors_written"] - prev["sectors_written"]) * 512, "at_unix": s["at_unix"]})
        if s["label"] != "image_committed" and not s["label"].endswith("_process_start"):
            prev = s
    return {"phases": starts, "driver_invocations": invocations, "resume_events": resumes, "interruptions": interruptions,
            "ssd": {"device": "nvme0n1", "boots": len(by_boot), "samples": len(samples), "bytes_written_between_samples": int(written),
                    "by_driver_stage": stages,
                    "note": "device-level sectors written between the first and the last sample of each boot (includes every other writer on the device; "
                            "writes after the last sample before a crash are not counted)"}}


def after_start_ttft(records, starts):
    """TTFT of the first follow-up requests of every evaluation process of the main phase, against the rest."""
    rows = [c for r in records for c in r["rows"] if c["t"] > 1 and c.get("pid") is not None and c.get("at") is not None]
    by_pid = {}
    for c in rows:
        by_pid.setdefault(c["pid"], []).append(c)
    order = sorted(by_pid, key=lambda p: min(c["at"] for c in by_pid[p]))
    out = []
    for i, pid in enumerate(order):
        mine = sorted(by_pid[pid], key=lambda c: c["at"])
        entry = {"process_index": i + 1, "pid": pid, "first_request_utc": utc(mine[0]["at"]), "kind": "first start" if i == 0 else "continued", "methods": {}}
        for m in METHODS:
            seq = [c for c in mine if c["m"] == m]
            head = seq[:AFTER_RESUME_REQUESTS]; head_ids = {id(c) for c in head}
            rest = [c["ttft"] for c in rows if c["m"] == m and id(c) not in head_ids]
            if not head or not rest:
                continue
            entry["methods"][NAME[m]] = {"first_n": len(head), "first_mean_ms": float(np.mean([c["ttft"] for c in head])),
                                         "first_request_ms": head[0]["ttft"], "first_p50_ms": float(np.median([c["ttft"] for c in head])),
                                         "rest_mean_ms": float(np.mean(rest)), "rest_p50_ms": float(np.median(rest)), "rest_p95_ms": float(np.percentile(rest, 95)),
                                         "first_over_p95_of_rest": int(sum(c["ttft"] > np.percentile(rest, 95) for c in head))}
        out.append(entry)
    return out


# ------------------------------------------------------------------ output
def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def fx(x, d=1, unit=""):
    return "—" if x is None else f"{x:,.{d}f}{unit}"


def main_tables_md(arms, paired, ratios, imp, storage, decomp, t1):
    L = ["| 방법 | Followup EM (%) | TTFT 평균 (ms) | p50 | p95 | SSD MB/followup | 읽기 호출 수/followup |", "|---|---:|---:|---:|---:|---:|---:|"]
    for a in arms:
        L.append(f"| {a['method']} | {a['em_followup']:.3f} | {a['followup_ttft_mean_ms']:.1f} | {a['followup_ttft_p50_ms']:.1f} | {a['followup_ttft_p95_ms']:.1f} | "
                 f"{a['ssd_mb_per_followup']:.2f} | {a['read_calls_per_followup']:.1f} |")
    L += ["", "Followup = T2–T3, strict normalized EM. SSD MB는 read syscall이 실제로 돌려준 바이트(1 MB = 1e6 bytes)의 요청 평균, 읽기 호출 수는 pread + preadv 횟수의 요청 평균.", "",
          "턴별 EM과 참고값:", "", "| 방법 | T1 EM | T2 EM | T3 EM | T1 TTFT 평균 (ms) | followup E2E 평균 (ms) |", "|---|---:|---:|---:|---:|---:|"]
    for a in arms:
        L.append(f"| {a['method']} | {a['em_T1']:.3f} | {a['em_T2']:.3f} | {a['em_T3']:.3f} | {a['t1_ttft_mean_ms']:.1f} | {a['followup_e2e_mean_ms']:.1f} |")
    i = next((a for a in arms if a["method_id"] == IMP), None)
    if i:
        L += ["", f"IMPRESS의 종류별 읽기량(followup 평균): probe key 사본 {i['probe_mb']:.2f} MB / K {i['k_mb']:.2f} MB / V {i['v_mb']:.2f} MB. "
                  f"읽기 호출 수는 요청마다 다르다(최소 {i['read_calls_min']:.0f}, 최대 {i['read_calls_max']:.0f})."]
    L += ["", f"### 짝지은 비교 (이미지 단위 paired bootstrap {RESAMPLES:,}회, seed {SEED}, 95% CI, followup)", "",
          "| 비교 | EM 차이 (pp) | TTFT 차이 (ms) |", "|---|---:|---:|"]
    for p in paired:
        if "em_diff_pp" in p:
            L.append(f"| {p['comparison']} | {p['em_diff_pp']:+.3f} [{p['em_ci_low']:+.3f}, {p['em_ci_high']:+.3f}] | "
                     f"{p['ttft_diff_ms']:+.1f} [{p['ttft_ci_low']:+.1f}, {p['ttft_ci_high']:+.1f}] |")
    L += ["", "### TTFT 비율 (followup 평균의 비, 같은 실행; 괄호는 이미지 단위 bootstrap 95% CI)", "", "| 비율 | 값 |", "|---|---:|"]
    for r in ratios:
        L.append(f"| {r['ratio']} | {fx(r['ttft_ratio_of_means'], 3)}× [{fx(r['ci_low'], 3)}, {fx(r['ci_high'], 3)}] |")
    if imp:
        L += ["", "### IMPRESS 통계 (followup)", "",
              f"- fallback 비율(layer·요청 평균): {100 * imp['fallback_rate']:.2f}% — 한 layer라도 fallback인 요청 {100 * imp['requests_with_any_fallback']:.2f}%, "
              f"32개 layer 모두 fallback인 요청 {100 * imp['requests_all_layers_fallback']:.2f}%",
              f"- probe head 간 평균 Jaccard: {imp['mean_jaccard']:.4f} (임계값 {imp['threshold']}, 이상이면 투표 선택)",
              f"- K·V chunk 읽기 비율(읽은 chunk / layer당 9 chunk): K {100 * imp['k_chunk_fraction']:.2f}% · V {100 * imp['v_chunk_fraction']:.2f}% · 평균 {100 * imp['kv_chunk_fraction']:.2f}%"
              + ("" if imp["consensus_kv_chunk_fraction"] is None else f" (투표 선택 layer만 {100 * imp['consensus_kv_chunk_fraction']:.2f}%)"),
              f"- 중요도 중 prefix 행이 차지하는 비율(probe head 3개 × 이미지 토큰 576개, layer·요청 평균): {100 * imp['prefix_share']:.2f}%",
              f"- 다음 layer probe key 미리 읽기: 요청당 {imp['probe_prefetched_reads_mean']:.1f}회 중 {imp['probe_ready_on_arrival_mean']:.2f}회가 사용 시점에 완료", "",
              "| layer | fallback 비율 | 평균 Jaccard | K chunk 비율 | V chunk 비율 | prefix 행 비율 |", "|---:|---:|---:|---:|---:|---:|"]
        for x in imp["layers"]:
            L.append(f"| {x['layer']} | {100 * x['fallback_rate']:.2f}% | {x['mean_jaccard']:.4f} | {100 * x['k_chunk_fraction']:.2f}% | {100 * x['v_chunk_fraction']:.2f}% | "
                     f"{100 * x['prefix_share']:.2f}% |")
    L += ["", "### T1 저장 비용 (이미지당 평균; store는 이미지의 첫 대화 T1에서 한 번 만든다)", "",
          "| 방법 | 저장 크기 (MB) | Visual KV (MB) | probe key 사본 (MB) | 쓰기 (ms) | 재배치 추가 (ms) | 저장 시간 합 (ms) | p95 | OS 기록량 (MB) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in storage:
        L.append(f"| {s['method']} | {s['store_mb_per_image']:.2f} | {s['visual_kv_mb']:.2f} | {s['probe_copy_mb']:.2f} | {s['writer_ms_store_ready_mean']:.1f} | "
                 f"{fx(s['after_request_layout_ms']) if s['method'] == 'IMPRESS' else '—'} | {s['total_store_ms_mean']:.1f} | {s['total_store_ms_p95']:.1f} | {s['os_write_mb_per_image']:.2f} |")
    L += ["", "- ReComp는 store가 없다. \"쓰기\"는 writer가 파일을 쓰고 fsync·rename까지 마친 시간이다. \"재배치 추가\"는 IMPRESS가 T1 통계로 순서를 정해 K·V·probe 파일을 다시 쓰는 시간이다.",
          "- FullLoad store에도 probe key 사본이 들어 있다(11/18 구현의 writer가 같다). FullLoad는 그 파일을 읽지 않는다.",
          "- 실험 하니스가 증거용으로 하는 전체 파일 해시(무결성 봉인)는 위 시간에 넣지 않았다: "
          + " · ".join(f"{s['method']} {fx(s['integrity_hash_ms_harness'])} ms" for s in storage) + ".",
          "- IMPRESS의 중요도 통계 캡처는 T1 요청 안에서 일어난다(이미지당 평균 "
          + fx(next((s["impress_capture_ms_inside_t1"] for s in storage if s["method"] == "IMPRESS"), None)) + " ms, 그 요청의 TTFT에 포함)."]
    L += ["", "### 재사용 턴(followup) 시간 분해", "",
          "| 방법 | TTFT 평균 (ms) | 읽기 대기 평균 (ms) | p95 | GPU 적재 평균 (ms) | layer hook 합 (ms) | 워커 읽기 합 (ms) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for d in decomp:
        if d["method"] == "ReComp":
            L.append(f"| ReComp | {d['ttft_mean_ms']:.1f} | — | — | — | — | — |")
        else:
            L.append(f"| {d['method']} | {d['ttft_mean_ms']:.1f} | {d['read_wait_mean_ms']:.2f} | {d['read_wait_p95_ms']:.2f} | {fx(d['gpu_load_mean_ms'], 2)} | "
                     f"{d['hook_mean_ms']:.2f} | {fx(d.get('worker_read_mean_ms'), 2)} |")
    L += ["", "- 읽기 대기: 요청 스레드가 읽기를 기다리며 멈춘 시간(동기 읽기는 읽는 시간 전체, 비동기 읽기는 워커가 아직 못 읽은 구간을 기다린 시간).",
          "- GPU 적재: Ours는 selector의 scatter 타이머(GPU cache에 행 쓰기 + mask), FullLoad는 layer hook 시간에서 읽기 대기를 뺀 값(전체 행의 host→GPU 복사).",
          "- IMPRESS는 layer hook 안에서 probe 점수 계산·선택·K/V 읽기·적재를 함께 하므로 GPU 적재만 따로 잰 값이 없다"
          + (f"(hook 합 − 읽기 대기 = {next((d['hook_minus_read_wait_mean_ms'] for d in decomp if d['method'] == 'IMPRESS'), 0):.2f} ms가 선택 계산 + 적재)." if any(d["method"] == "IMPRESS" for d in decomp) else "."),
          "- 워커 읽기 합은 연산과 겹쳐 진행되므로 TTFT에 더해지는 값이 아니다. 항목들의 합을 TTFT로 해석하지 않는다.",
          "", f"T1 생성 토큰이 네 방법에서 모두 같은 대화: {t1['identical_across_arms']:,} / {t1['dialogues']:,}."]
    return L


def gate_md(run, manifest):
    n4, nd = len(manifest["phase_images"]["gate"]), len(manifest["phase_images"]["dev"])
    g = {}
    for name in ("tests", "gate", "qindep_gate", "devgate", "qindep_devgate", "dev", "admission", "posthoc_main"):
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

    L = [f"| gate | {n4}장 | DEV {nd}장 | 내용 |", "|---|---|---|---|"]
    a, d, dg = g["gate"] or {}, g["dev"] or {}, g["devgate"] or {}
    g1a, g1d = a.get("gate1_store_contents"), d.get("gate1_store_contents")
    L.append(f"| 1. Ours 저장 내용 일치 | {st(g1a)} | {st(g1d)} | "
             + " · ".join(f"{lab} {x['pass_count']}/{x['image_count']}장, 비교한 K·V 행 {x['kv_rows_compared']:,}개 중 다른 행 {x['kv_rows_different']}개"
                          for lab, x in (("4장", g1a), ("DEV", g1d)) if x) + " |")
    q4, qd = g["qindep_gate"], g["qindep_devgate"]
    L.append(f"| 2. Ours 선택·배치가 질문과 무관 | {st(q4)} | {st(qd)} | "
             + " · ".join(f"{lab} {x['pass_count']}/{x['image_count']}장(질문 3개씩)" for lab, x in (("4장", q4), ("DEV", qd)) if x and "pass_count" in x) + " |")
    L.append(f"| 3. Ours 동기 = 비동기 | {st(a, 'gate3_ours_sync_equals_async')} | {st(dg, 'gate3_ours_sync_equals_async')} | "
             f"4장 {cmp_txt(a.get('gate3_ours_sync_equals_async'))} · DEV {cmp_txt(dg.get('gate3_ours_sync_equals_async'))} |")
    L.append(f"| 4. Ours 두 번 실행 | {st(a, 'gate4_ours_repeat')} | {st(dg, 'gate4_ours_repeat')} | "
             f"4장 {cmp_txt(a.get('gate4_ours_repeat'))} · DEV {cmp_txt(dg.get('gate4_ours_repeat'))} |")
    g5a, g5d = a.get("gate5_impress") or {}, dg.get("gate5_impress") or {}
    L.append(f"| 5. IMPRESS | {st(a, 'gate5_impress')} | {st(dg, 'gate5_impress')} | 유지 100% vs 같은 재배치 store의 FullLoad loader: "
             f"4장 {cmp_txt(g5a.get('retention_100_vs_fullload_loader_on_reordered_store'))} · DEV {cmp_txt(g5d.get('retention_100_vs_fullload_loader_on_reordered_store'))}; "
             f"동기 vs 비동기: 4장 {cmp_txt(g5a.get('sync_vs_async'))} · DEV {cmp_txt(g5d.get('sync_vs_async'))} |")
    g6a, g6d = a.get("gate6_recomp_continuity"), d.get("gate6_recomp_continuity")
    L.append(f"| 6. ReComp = 11/12 실행 | {st(g6a)} | {st(g6d)} | "
             + " · ".join(f"{lab} {x['requests']:,}건 중 토큰 불일치 {x['mismatch_count']}건" for lab, x in (("4장", g6a), ("DEV", g6d)) if x) + " |")
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
    L += ["", f"본실험 진입 판정: **{adm['status'] if adm else '미실행'}** (gate 1–7이 {n4}장과 DEV {nd}장에서 모두 PASS이고 DEV 파일럿이 실행됐을 때만 본실험을 시작한다)."]
    if t:
        for s in t["suites"]:
            if s["failed"]:
                L += ["", f"transformers {s['transformers']} 환경에서 이 작업 전부터 실패하던 테스트(목록만 적음): " + "; ".join(f"`{x}`" for x in s["failed"] if x not in s["new_failures"]) + "."]
                if s["new_failures"]:
                    L += [f"새로 실패한 테스트: " + "; ".join(f"`{x}`" for x in s["new_failures"]) + "."]
    return L, g


def gate_failure_md(g):
    """What failed and the recorded cause, for a run whose main phase was not started."""
    L = []
    adm = g.get("admission") or {}
    for name, step in (adm.get("steps") or {}).items():
        if step.get("ok"):
            continue
        doc = g.get(name) or {}
        if step.get("status") == "MISSING":
            L.append(f"- `{name}`: 실행되지 않음(앞 단계가 통과하지 못해 시작하지 않았거나, 검사가 결과를 내지 못했다).")
            continue
        L.append(f"- `{name}`: {step.get('status')}" + (f" — 실패 항목: {', '.join(doc.get('failed') or [])}" if doc.get("failed") else ""))
        for key in doc.get("failed") or []:
            part = doc.get(key) or {}
            if key == "execution":
                L.append(f"  - 요청 단계가 끝나지 않았다: {json.dumps({k: part.get(k) for k in ('status', 'requests', 'expected')}, ensure_ascii=False)}")
            for sub in ([part] if "examples" in part or "mismatches" in part or "images" in part else [v for v in part.values() if isinstance(v, dict)]):
                if sub.get("status") == "PASS":
                    continue
                if sub.get("criterion"):
                    L.append(f"  - 기준: {sub['criterion']}")
                counts = {k: sub[k] for k in ("requests", "tokens_differ", "tokens_differ_same_prompt", "prompt_differs", "fields_differ", "missing",
                                              "mismatch_count", "image_count", "pass_count", "kv_rows_different") if k in sub}
                L.append(f"  - 집계: {json.dumps(counts, ensure_ascii=False)}")
                for x in (sub.get("examples") or sub.get("mismatches") or [im for im in sub.get("images", []) if im.get("status") != "PASS"])[:5]:
                    L.append(f"  - 예: {json.dumps(x, ensure_ascii=False)[:600]}")
        if name == "tests":
            for s in doc.get("suites", []):
                if s.get("new_failures"):
                    L.append(f"  - transformers {s['transformers']}: 새 실패 {s['new_failures']}")
        if name.startswith("qindep"):
            for im in doc.get("images", []):
                if im.get("status") != "PASS":
                    L.append(f"  - {im['image_id']}: " + ", ".join(k for k, v in im.items() if v is False))
    return L


def settings_md(run, config):
    env = config["environment"]
    return ["## 설정", "",
            f"- 실행: `runs/{run.name}`(요청 단위 원본), `results/{run.name}`(집계·보고서·status.json·driver.log). 계약: `{CONTRACT_DOC}`(사본 `CONTRACT.md`).",
            f"- 환경: Python {env['python'].split()[0]}, torch {env['torch']}, transformers {env['transformers']} (`{env['executable']}`) — 11/18 LLaVA-1.5 실행과 같다(freeze에서 대조).",
            f"- 모델 `{config['model']}` revision `{config['revision'][:8]}`, NF4/BF16/eager, FP16 SSD KV, greedy cap16, HF 기본 336×336 전처리, visual token 576개, separator 없음.",
            "- 데이터: MT-GQA 398 images / 4,061 dialogues × 3턴, method-local generated history, strict normalized EM, 대화 단위 arm 순서 회전. "
            "모든 방법이 매 대화 T1을 픽셀부터 계산하고, 저장 KV 읽기는 요청 직전에 page cache를 비운 cold 조건이다.",
            "- ReComp: 매 턴 이미지부터 다시 계산. FullLoad: Visual KV 전체 저장, 워커가 layer 순서로 전부 읽음(11/18 구현).",
            "- IMPRESS: 11/18 실행의 구현·설정 그대로(probe head 0·1·2, 전체 행 열 합 중요도, 상위 25%, 평균 Jaccard ≥ 0.3이면 투표 선택 아니면 fallback, "
            "K/V 분리, 64토큰 chunk, 첫 대화 T1 기준 layer별 재배치, 다음 layer probe key 미리 읽기).",
            "- Ours: image-only CLIP CLS→patch attention 중요도, PYR-b20 layer별 배분(총 KV 예산 25%), layer l은 상위 k_l개 토큰을 원래 토큰 인덱스 오름차순으로 저장, "
            "재사용 턴에는 각 행을 GPU cache의 원래 위치에 올리고 나머지는 mask. 파일 구성·비동기 읽기 순서·pinned buffer·읽기 호출 수는 기존 LLaVA-1.5 비동기 경로와 같다.", ""]


def execution_md(run, manifest, config, rec, resume_ttft, done_requests, plan, protection, source_ok, storage):
    L = ["## 실행 기록", ""]
    exp = manifest["expected_requests"]
    L.append("- 요청 수: " + " · ".join(f"{ph} {done_requests.get(ph, 0):,} / {exp[ph]:,}" for ph in PHASES) + ". gate·devgate·dev 요청은 본실험 분모에 넣지 않았다.")
    for ph in PHASES:
        p = rec["phases"][ph]
        if p["process_starts"]:
            end = p["finished_unix"]
            L.append(f"- {ph}: 프로세스 시작 {p['process_starts']}회 (" + ", ".join(utc(s["at_unix"]) for s in p["starts"]) + ")"
                     + (f", 종료 {utc(end)}, 처음 시작부터 {(end - p['starts'][0]['at_unix']) / 3600:.2f} h" if end else ", 미완료") + ".")
    inv = rec["driver_invocations"]
    L.append(f"- 드라이버 실행 {len(inv)}회: " + ", ".join(f"{x['at_utc']} ({x['kind']}, 당시 단계 {x.get('stage_at_start')})" for x in inv) + ".")
    resumed = [x for x in inv if x["kind"] == "resume"]
    extra_starts = sum(max(0, rec["phases"][ph]["process_starts"] - 1) for ph in PHASES)
    L.append(f"- 이어서 실행한 횟수: 드라이버 재실행 {len(resumed)}회, 요청 단계 프로세스 재시작 {extra_starts}회"
             + (" (중단 없이 한 번에 끝났다)." if not resumed and not extra_starts else "."))
    ev = [e for e in rec["resume_events"] if e.get("mode") in ("continue", "fresh")]
    if ev:
        L += ["", "| 시점 (UTC) | 단계 | 이미지 | 처리 | 유지한 완료 대화 | 끊긴 대화 | 다시 실행한 대화 |", "|---|---|---|---|---:|---|---|"]
        for e in ev:
            how = "완료 대화 유지, 끊긴 대화부터" if e["mode"] == "continue" else f"이미지 처음부터 ({e.get('store_check')})"
            L.append(f"| {utc(e['at_unix'])} | {e['phase']} | {e['image_id']} | {how} | {e.get('complete_dialogues_kept', 0)} | {e.get('interrupted_dialogue')} | "
                     f"{', '.join(e.get('restart_dialogues', [])[:12])}{' …' if len(e.get('restart_dialogues', [])) > 12 else ''} |")
        L.append("")
    else:
        L.append("- 다시 실행한 대화: 없음.")
    if rec["interruptions"]:
        L.append("- 중단 기록: " + "; ".join(f"{x['file']} ({(x['error'] or '')[:120]})" for x in rec["interruptions"]) + ".")
    if resume_ttft:
        L += ["", f"프로세스 시작 직후의 followup 요청(방법당 처음 {AFTER_RESUME_REQUESTS}건)과 나머지 요청의 TTFT (본실험):", "",
              "| 프로세스 | 첫 요청 시각 (UTC) | 방법 | 처음 n건 평균 (ms) | 그 방법의 첫 요청 (ms) | 나머지 평균 (ms) | 평균 차이 (ms) | 나머지 p95 (ms) | 처음 n건 중 p95 초과 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for e in resume_ttft:
            for name, v in e["methods"].items():
                L.append(f"| {e['process_index']} ({'처음 시작' if e['kind'] == 'first start' else '이어서 실행'}) | {e['first_request_utc']} | {name} | {v['first_mean_ms']:.1f} | "
                         f"{v['first_request_ms']:.1f} | {v['rest_mean_ms']:.1f} | {v['first_mean_ms'] - v['rest_mean_ms']:+.1f} | {v['rest_p95_ms']:.1f} | "
                         f"{v['first_over_p95_of_rest']} / {v['first_n']} |")
        L.append("")
        L.append("\"나머지\"는 그 방법의 본실험 followup 요청 중 해당 처음 n건을 뺀 전부다. \"p95 초과\"는 처음 n건 중 나머지의 p95보다 느린 요청 수다(우연히도 n의 5%쯤은 넘는다).")
        if len(resume_ttft) == 1:
            L.append("이어서 실행한 적이 없으므로 \"이어서 실행한 직후\"의 영향은 측정 대상이 없다. 위 표는 처음 시작 직후의 값이다(모델 로드 뒤 합성 요청 1건으로 예열, 측정에서 제외).")
    ssd = rec["ssd"]
    store_total = sum(s["os_write_bytes_total"] for s in storage) if storage else None
    L += ["", f"- SSD 기록량(실험 전체, `/sys/block/{ssd['device']}/stat`의 기록 섹터 차이, 부팅 {ssd['boots']}회 구간 합): {ssd['bytes_written_between_samples'] / 1e9:,.1f} GB. "
              "이 값은 장치 전체의 기록량이라 같은 SSD에 쓴 다른 프로세스의 기록도 포함한다.",
          f"- 실행 전 예상 기록량(store 크기 × 이미지 수): {plan['expected_ssd_write_bytes']['total'] / 1e9:,.1f} GB "
          f"(gate {plan['expected_ssd_write_bytes']['gate'] / 1e9:.1f} · devgate {plan['expected_ssd_write_bytes']['devgate'] / 1e9:.1f} · dev {plan['expected_ssd_write_bytes']['dev'] / 1e9:.1f} · "
          f"main {plan['expected_ssd_write_bytes']['main'] / 1e9:.1f} · 질문 무관 확인 {plan['expected_ssd_write_bytes']['question_independence'] / 1e9:.1f})."
          + (f" 본실험에서 store 쓰기 때 프로세스가 OS에 넘긴 바이트의 합(FullLoad·IMPRESS·Ours store): {store_total / 1e9:,.1f} GB." if store_total else "")
          + " 예상값에는 단위 테스트가 임시로 쓰는 store, 요청 기록, 이어 실행으로 다시 만든 store가 들어 있지 않다.",
          "- 드라이버 단계별 장치 기록량: " + " · ".join(
              f"{'(중단된 구간)' if x['until'].startswith('driver_start') else x['until'].replace('after_', '').replace('driver_', '')} {x['bytes'] / 1e9:.1f} GB"
              for x in ssd.get("by_driver_stage", [])) + ".",
          f"- 디스크 여유: 실행 전 {plan['free_bytes_at_freeze'] / 2**30:.1f} GiB, 보고 시점 {shutil.disk_usage(run).free / 2**30:.1f} GiB. scratch store는 이미지마다 감사·commit 뒤 삭제했다.",
          f"- 기존 `results/`·`runs/`·`kvstore*` 파일 목록(경로, 크기, mtime) 비교: {'변화 없음' if protection['unchanged'] else '변화 있음 — protection_after.json 참고'} "
          f"({protection['before']['files']:,}개 → {protection['after']['files']:,}개).",
          f"- 소스 freeze 대조(보고 시점): {'일치' if source_ok else '불일치'} (freeze 때 기록한 파일의 SHA256 비교; 그 뒤 새로 생긴 파일과 보고서 스크립트는 대상이 아니다)."]
    return L


CODE_MD = [
    "## 코드에서 추가하거나 바꾼 부분", "",
    "바꾼 파일 (기본 동작은 그대로, 옵션 추가):", "",
    "- `mmimpress/llava15/store.py`: `write_llava15_packed_store(..., row_order=\"importance\")`, `persist_captured_llava15_prefix(..., row_order=\"importance\")`에 "
    "`row_order=\"original\"` 옵션. layer l이 중요도 순서의 앞 k_l개 토큰(동점 처리 포함, 기존 `visionzip_repack_order`의 순서)을 원래 토큰 인덱스 오름차순으로 저장한다. "
    "메타데이터에 중요도 순열을 남기지 않고(`stored_original_indices`, `stored_min_keep_count`만), `visionzip_layout.pt`를 쓰지 않는다. "
    "새 store format `llava15_prefixstore_ratio_packed_original_v1`, 새 검증 함수 `validate_llava15_packed_original_meta`.",
    "- `mmimpress/llava15/serve.py`: `Llava15PackedContext`가 store의 `row_order`를 읽어 layer별 원래 인덱스를 준비하고, "
    "`Llava15PrefixSelector._scatter_layer`가 원래 순서 store의 행을 GPU cache의 원래 토큰 위치에 쓰고 그 위치만 보이게 mask를 만든다. "
    "읽기 계획(`PackedReader.stream_plan`), 워커, pinned buffer, hook은 기존 코드 그대로다.", "",
    "바꾸지 않은 것: LLaVA-NeXT 경로(`mmimpress/packed_store.py`, `packed_serve.py`, `piggyback.py`), `mmimpress/impress_official.py`, `mmimpress/llava15/raster.py`, "
    "`mmimpress/serve.py`, 11/18 실행의 하니스 `scripts/199_eval_llava15_mtvqa_impress_code.py`(이 실험이 import해서 그대로 호출), 기존 store·결과 디렉터리·동결 파일.", "",
    "새 파일:", "",
    "- `scripts/206_eval_llava15_mt_gqa_main.py`: 이 실험의 실행 하니스(gate / devgate / dev / main, 대화 단위 이어 실행).",
    "- `scripts/207_validate_llava15_mt_gqa_main.py`: gate 1–8 판정과 본실험 진입 판정.",
    "- `scripts/208_report_llava15_mt_gqa_main.py`: 이 보고서(측정에 관여하지 않으며 소스 freeze 대상이 아니다).",
    "- `scripts/llava15_main_status.py`: status.json, SSD 기록량 표본, 드라이버 실행 기록.",
    "- `scripts/run_llava15_mt_gqa_main.sh`: 드라이버(터미널과 분리, 잠금, 단계별 자동 판정, 이어 실행).",
    "- `tests/test_llava15_original_order.py`: 새 테스트.",
    "- `docs/llava15_mt_gqa_main_contract.md`: 실행 전 동결한 방법 정의와 gate.", "",
    "새 테스트 (`tests/test_llava15_original_order.py`):", "",
    "- 저장: 원래 순서 store의 layer별 집합이 재배열 store의 앞 k_l개와 같다(동점 포함), 같은 토큰의 K·V 바이트·byte table·payload 크기가 같다, 메타데이터에 중요도 순열이 없고 "
    "`visionzip_layout.pt`가 없다, 기본 경로의 메타데이터·바이트가 그대로다, 변조·잘못된 옵션은 거부된다.",
    "- 적재: 각 행이 GPU cache의 원래 위치에 올라가고 저장하지 않은 위치는 mask된다, 두 store가 layer별로 같은 토큰을 같은 K/V 값으로 보여 준다, "
    "비동기 읽기가 동기 읽기와 같은 buffer·cache·mask를 만들고 읽기 구간·순서·호출 수(32회)가 재배열 store의 비동기 읽기와 같다.",
    "- 질문 무관: 서로 다른 프롬프트의 캡처 세 개로 만든 store의 layer별 인덱스 목록·byte table·파일 구성이 같다.",
    "- 하니스: arm·단계·요청 수, gate 비교 함수, 이어 실행 계획(완료 대화 유지, 끊긴 대화의 행 버리기, store 검증 실패 시 이미지 재시작).", ""]

LIMITS_MD = [
    "## 지시와 다르게 했거나 지시에 없던 것, 측정하지 않은 것과 한계", "",
    "지시와 다르게 했거나 지시에 없던 것:", "",
    "- 이어 실행: 완료된 대화는 건너뛰고 끊긴 대화는 T1부터 다시 한다. 다만 store는 이미지의 첫 대화 T1에서 한 번 만들어 그 이미지의 모든 대화가 쓰므로, "
    "끊긴 지점이 첫 대화이거나 디스크의 store가 만들 때 기록한 SHA256과 맞지 않으면 그 이미지의 store와 기록을 모두 버리고 이미지의 첫 대화부터 다시 한다"
    "(이 경우 그 이미지에서 이미 끝났던 대화도 다시 실행된다). 버린 기록은 삭제하지 않고 `discarded_rows_resume_*.jsonl` / `ABANDONED.json`으로 남겨 결과에서만 제외했다.",
    "- 드라이버는 요청 단계가 오류로 끝나면 같은 실행 안에서 최대 3회까지 스스로 다시 시작한다(지시에 없던 것). 각 재시작은 위 \"실행 기록\"의 프로세스 시작 횟수에 들어 있다. "
    "GPU를 쓰는 단계는 다른 프로세스가 GPU를 쓰고 있으면 비워질 때까지(최대 2시간) 기다렸다가 시작한다.",
    "- 소스 freeze 대조는 freeze 때 기록한 파일만 비교한다. 그 뒤 새로 생긴 파일과 보고서 스크립트는 대상이 아니다(저장소의 다른 작업이 실험을 멈추지 않게 하기 위함).",
    "- gate 판정용 arm(Ours 동기 읽기, Ours 두 번째 실행, IMPRESS 동기 읽기, IMPRESS 유지 100%, 재배치 store를 읽는 FullLoad loader)을 4장에서는 본실험 arm과 한 단계에서, "
    "DEV 40장에서는 별도 단계(devgate, 7 arm)로 돌렸다. gate 1과 6의 DEV 판정은 파일럿 단계의 기록으로 했다.",
    "- DEV 파일럿은 네 방법에 Ours(재배열)을 더한 5 arm을 한 단계에서 돌렸다. 따라서 파일럿의 arm 회전은 5 arm 기준이고 본실험은 4 arm 기준이다.",
    "- gate 2는 Ours store를 질문 3개로 새로 만들어 layer별 인덱스 목록·행 수·byte table·파일 구성·메타데이터(요청 기록과 payload 해시 제외)를 비교했다. payload 바이트는 보고만 한다.",
    "- gate 7은 실행 환경(transformers 4.57.6)과 교차 확인용 5.13.1 환경 두 곳에서 돌렸고 두 곳 모두 새 실패 0개를 요구했다.",
    "- gate 검사 프로그램이 결과 파일을 만들지 못하고 끝난 경우(판정이 아닌 실행 오류)는 FAIL로 확정하지 않고 멈추며, 같은 명령으로 그 검사부터 다시 한다. 판정이 기록된 gate는 다시 판정하지 않는다.",
    "- 모델을 올린 직후 합성 이미지 요청 1건으로 예열한다(11/18 실행과 같음, 모든 지표에서 제외). 저장 KV의 예열은 없다.",
    "- status.json의 예상 종료 시각은 현재 단계의 요청당 시간을 남은 요청 수에 곱한 근사값이다.",
    "- TTFT 비율에 이미지 단위 bootstrap 95% CI를 함께 적었다(지시에 없던 것).", "",
    "측정하지 않은 것과 한계:", "",
    "- 한 번의 실행이다. 실행 간 TTFT 변동은 재지 않았고, 다른 실행의 절대 TTFT와 비교하지 않는다.",
    "- FullLoad의 동기 읽기 arm은 돌리지 않았다(지시의 gate에 없음). FullLoad store의 probe key 사본은 저장만 되고 읽히지 않는다.",
    "- IMPRESS의 GPU 적재 시간은 선택 계산과 한 타이머 안에 있어 따로 재지 못했다.",
    "- SSD 기록량은 장치 전체의 값이며 이 실험만의 기록량을 분리한 값이 아니다. 강제 종료 직전 마지막 표본 이후의 기록은 빠진다.",
    "- 다른 KV 예산(25% 외), 다른 layer 배분, warm page cache 조건, 다른 데이터셋·모델은 이 실험의 범위가 아니다.",
    "- GO/NO-GO 판정은 두지 않았다. 이 문서는 측정 보고다.", ""]


def dev_md(run, manifest):
    """The DEV pilot table (gate 8) with the reordered packed arm as a reference."""
    methods = METHODS + [REORDER]
    records, expected = collect(run, "dev", manifest, methods, False)
    arms = arm_table(records, methods)
    paired = paired_table(records, PAIRS)
    ratios = ratio_table(records, RATIOS)
    ref = order_reference(records)
    L = ["| 방법 | Followup EM (%) | TTFT 평균 (ms) | p50 | p95 | SSD MB/followup | 읽기 호출 수/followup |", "|---|---:|---:|---:|---:|---:|---:|"]
    for a in arms:
        L.append(f"| {a['method']} | {a['em_followup']:.3f} | {a['followup_ttft_mean_ms']:.1f} | {a['followup_ttft_p50_ms']:.1f} | {a['followup_ttft_p95_ms']:.1f} | "
                 f"{a['ssd_mb_per_followup']:.2f} | {a['read_calls_per_followup']:.1f} |")
    L += ["", f"DEV {len(records)}장({ref['followup_requests']:,} followup/arm)의 짝지은 비교(참고, 구간이 넓다):", "", "| 비교 | EM 차이 (pp) | TTFT 차이 (ms) |", "|---|---:|---:|"]
    for p in paired:
        L.append(f"| {p['comparison']} | {p['em_diff_pp']:+.3f} [{p['em_ci_low']:+.3f}, {p['em_ci_high']:+.3f}] | {p['ttft_diff_ms']:+.1f} [{p['ttft_ci_low']:+.1f}, {p['ttft_ci_high']:+.1f}] |")
    L += ["", "### DEV 참고값: Ours(재배열)과 Ours의 차이 (gate가 아니며 본실험 arm에도 없다)", "",
          f"- Followup EM: Ours {ref['em_ours']:.3f}% · Ours(재배열) {ref['em_reordered']:.3f}% · 차이(Ours − Ours(재배열)) {ref['em_diff_pp']:+.3f} pp "
          f"[{ref['em_ci'][0]:+.3f}, {ref['em_ci'][1]:+.3f}]",
          f"- Followup TTFT 평균: Ours {ref['ttft_ours_ms']:.2f} ms · Ours(재배열) {ref['ttft_reordered_ms']:.2f} ms · 차이 {ref['ttft_diff_ms']:+.2f} ms "
          f"[{ref['ttft_ci'][0]:+.2f}, {ref['ttft_ci'][1]:+.2f}] · 비율 {ref['ttft_ratio']:.4f}× [{ref['ttft_ratio_ci'][0]:.4f}, {ref['ttft_ratio_ci'][1]:.4f}]",
          f"- 생성 토큰이 다른 followup 요청: {ref['tokens_differ']:,} / {ref['followup_requests']:,} · 읽기 바이트나 호출 수가 다른 요청: {ref['bytes_or_read_calls_differ']:,}",
          f"- 두 store의 같은 토큰 K·V 행 비교: {ref['store_rows_compared']:,}행 중 다른 행 {ref['store_rows_different']}개"]
    return L, {"arms": arms, "paired": paired, "ratios": ratios, "order_reference": ref, "images": len(records), "storage": storage_table(records, methods)}


def protection_check(run, out_root, config):
    import llava15_support as S
    E = S.load_script("_l15_mtgqa_main_eval_report", EVAL)
    before = json.loads((run / "protection_before.json").read_text())
    after = E.stat_inventory(run, out_root / "protection_after.tsv.gz")
    protection = {"before": before, "after": after, "unchanged": before["sha256"] == after["sha256"] and before["files"] == after["files"]}
    (out_root / "protection_after.json").write_text(json.dumps(protection, indent=1) + "\n")
    return protection, not E.freeze_violations(config["source_sha256"])


def committed_counts(run):
    out = {}
    for ph in PHASES:
        out[ph] = sum(json.loads(cp.read_text())["requests"] for cp in (run / ph / "images").glob("*/COMMITTED.json")) if (run / ph).exists() else 0
    return out


def report_auto(run, manifest, config, out_root):
    out_root.mkdir(parents=True, exist_ok=True)
    gates, g = gate_md(run, manifest)
    admitted = (g.get("admission") or {}).get("status") == "PASS"
    main_done = (run / "main_execution_complete.json").exists()
    counts = committed_counts(run)
    status = {}
    try:
        status = json.loads((out_root / "status.json").read_text())
    except (OSError, ValueError):
        pass
    lines = ["# LLaVA-1.5-7B × MT-GQA 본실험: ReComp / FullLoad / IMPRESS / Ours", "",
             "측정 보고다. GO/NO-GO 판정은 두지 않는다. 측정하지 않은 값은 \"미실행\" 또는 \"—\"로 적는다.", ""]
    summary = {"run": run.name, "admission": (g.get("admission") or {}).get("status"), "main_complete": main_done, "committed_requests": counts,
               "gates": {k: (v or {}).get("status") for k, v in g.items()}, "generated_unix": time.time()}
    if main_done:
        lines += [f"**본실험 완료**: {counts['main']:,} / {manifest['expected_requests']['main']:,} 요청 "
                  f"({len(manifest['phase_images']['main']):,} images, {len(manifest['phase_dialogue_ids']['main']):,} dialogues × 3턴 × 4 arm).", ""]
    elif admitted:
        lines += [f"**본실험 미완료(중간 집계)**: {counts['main']:,} / {manifest['expected_requests']['main']:,} 요청까지 commit됐다. 아래 본실험 표는 commit된 이미지만의 값이며 최종 결과가 아니다. "
                  f"이어서 실행: `scripts/run_llava15_mt_gqa_main.sh runs/{run.name}`", ""]
    else:
        lines += ["**본실험은 시작하지 않았다.** gate가 사전 고정 기준을 통과하지 못했거나 gate 단계가 끝나지 않았다. 기준은 바꾸지 않았다.", ""]
    if status.get("last_error"):
        lines += [f"마지막 오류(status.json): `{json.dumps(status['last_error'], ensure_ascii=False)}`", ""]
    lines += settings_md(run, config) + ["## Gate", ""] + gates + [""]
    if not admitted:
        failure = gate_failure_md(g)
        lines += ["### 실패 내용과 원인", "",
                  "아래는 통과하지 못한 gate의 기준, 집계, 어긋난 요청의 예다(전체는 `gates/*.json`). 드라이버는 기준에 어긋난 사실과 그 증거까지만 기록한다."] + (failure or ["- 본실험 진입 판정 기록이 없다(gate 단계가 실행 오류로 중단됨). status.json의 마지막 오류와 driver.log를 본다."]) + [""]
    storage = []
    records = []
    if admitted and counts["main"]:
        records, expected = collect(run, "main", manifest, METHODS, True)
        partial = len(records) != expected
        arms = arm_table(records, METHODS); paired = paired_table(records, PAIRS); ratios = ratio_table(records, RATIOS)
        imp = impress_internal(records); storage = storage_table(records, [FULL, IMP, OURS]); decomp = decomposition(records, METHODS)
        t1 = t1_identity(records, METHODS)
        main_dir = out_root / "main"; main_dir.mkdir(exist_ok=True)
        write_csv(main_dir / "arms.csv", arms); write_csv(main_dir / "paired.csv", paired); write_csv(main_dir / "ttft_ratios.csv", ratios)
        write_csv(main_dir / "storage.csv", storage); write_csv(main_dir / "impress_layers.csv", imp["layers"]); write_csv(main_dir / "decomposition.csv", decomp)
        lines += [f"## 본실험 표 ({len(records):,}장" + (f" / {expected:,}장, 중간 집계" if partial else "") + ")", ""]
        lines += main_tables_md(arms, paired, ratios, imp, storage, decomp, t1) + [""]
        summary.update(main={"images": len(records), "expected_images": expected, "partial": partial, "arms": arms, "paired": paired, "ttft_ratios": ratios,
                             "impress_internal": imp, "storage": storage, "decomposition": decomp, "turn1_identity": t1,
                             "peak_scratch_bytes_max": int(max(r["peak_scratch"] for r in records)),
                             "images_resumed_inside": sum(1 for r in records if r["resumed_inside_image"])})
        post = g.get("posthoc_main")
        if post:
            c = post["recomp_continuity_main"]
            lines += [f"참고(판정 아님): 본실험 ReComp의 생성 토큰을 11/12 실행의 같은 요청과 비교하면 {c['requests']:,}건 중 {c['mismatch_count']}건이 다르다.", ""]
    else:
        lines += ["## 본실험 표", "", "미실행.", ""]
    dev_done = (run / "dev_execution_complete.json").exists()
    if dev_done:
        body, dev = dev_md(run, manifest)
        lines += [f"## DEV {len(manifest['phase_images']['dev'])}장 파일럿 (gate 8; 본실험 분모가 아니다)", ""] + body + [""]
        summary["dev"] = dev
    else:
        lines += ["## DEV 40장 파일럿", "", "미실행.", ""]
    rec = execution_record(run, manifest)
    plan = json.loads((run / "storage_plan.json").read_text())
    protection, source_ok = protection_check(run, out_root, config)
    resume_ttft = after_start_ttft(records, rec) if records else []
    lines += execution_md(run, manifest, config, rec, resume_ttft, counts, plan, protection, source_ok, storage) + [""]
    lines += CODE_MD + LIMITS_MD
    notes = out_root / "notes.md"      # optional hand-written additions; appended verbatim
    if notes.exists():
        lines += [notes.read_text().rstrip(), ""]
    (out_root / "report.md").write_text("\n".join(lines) + "\n")
    summary.update(execution=rec, after_start_ttft=resume_ttft, protection_unchanged=protection["unchanged"], source_freeze_matches=source_ok,
                   environment=config["environment"])
    (out_root / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str) + "\n")
    for name in ("config.json", "CONTRACT.md", "storage_plan.json"):
        shutil.copyfile(run / name, out_root / name)
    if (run / "gates").is_dir():
        shutil.copytree(run / "gates", out_root / "gates", dirs_exist_ok=True)
    print(json.dumps({"report": str(out_root / "report.md"), "admission": summary["admission"], "main_complete": main_done,
                      "main_requests": counts["main"]}), flush=True)


def report_dev(run, manifest, out_root):
    out = out_root / "dev"; out.mkdir(parents=True, exist_ok=True)
    body, dev = dev_md(run, manifest)
    write_csv(out / "arms.csv", dev["arms"]); write_csv(out / "paired.csv", dev["paired"])
    (out / "summary.json").write_text(json.dumps(dev, indent=1, ensure_ascii=False) + "\n")
    lines = ["# LLaVA-1.5-7B × MT-GQA — DEV 40장 파일럿 (gate 8)", "",
             f"구현 확인용이며 본실험 분모에 넣지 않는다. 계약: `{CONTRACT_DOC}`. 방법 5개(본실험 4개 + 참고용 Ours(재배열))를 한 단계에서 돌렸다.", ""] + body
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"phase": "dev", "images": dev["images"], "report": str(out / "report.md")}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--phase", choices=("dev", "auto"), required=True)
    p.add_argument("--out", type=Path, default=None, help="output directory (default results/<run name>)")
    args = p.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    config = json.loads((run / "config.json").read_text())
    out_root = args.out.resolve() if args.out is not None else ROOT / "results" / run.name
    if args.phase == "dev":
        return report_dev(run, manifest, out_root)
    return report_auto(run, manifest, config, out_root)


if __name__ == "__main__":
    main()

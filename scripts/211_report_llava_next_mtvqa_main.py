#!/usr/bin/env python3
"""Tables and report of the LLaVA-NeXT x MT-VQA main experiment: ReComp (v3 record) / FullLoad / IMPRESS / Ours
(contract docs/llava_next_mtvqa_main_contract.md, section 9).

Reads committed rows only; takes no part in any measurement (not in the source freeze).

  --phase pilot  results/<run>/pilot_table.md: the 40-image pilot table (FullLoad, IMPRESS, Ours, ReComp) and the
                 ReComp TTFT of this run against the v3 record of the same requests
  --phase auto   results/<run>/report.md: the main tables when the main phase is complete, an interim main table
                 when it was admitted but is not complete, otherwise the outcome of the pre-run checks and gates

A measurement report: no GO / NO-GO verdict is produced.
"""
from __future__ import annotations

import argparse
import csv
import gzip
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

from mmimpress.recompute_kv25 import paired_bootstrap  # noqa: E402

RECOMP, FULL, IMP = "recompute", "fullload_async", "impress_official_async"
OURS = "ours_origorder_kv25_PYR-beta-raw_packed_async"
V3_RECOMP, V3_FULL = "recompute", "fullload"
MAIN = [FULL, IMP, OURS]
NAME = {RECOMP: "ReComp", FULL: "FullLoad", IMP: "IMPRESS", OURS: "Ours"}
KIND = {FULL: "raster", IMP: "impress_official", OURS: "packedorig_PYR-beta-raw"}
V3 = "v3"                                    # pseudo method id of the v3 ReComp record inside the paired tables
SCORE_PAIRS = [(OURS, IMP), (OURS, FULL), (OURS, V3), (IMP, FULL), (FULL, V3)]
TTFT_PAIRS = [(OURS, IMP), (OURS, FULL), (IMP, FULL)]
RATIOS = [(OURS, FULL), (OURS, IMP), (IMP, FULL)]
CROSS_RATIOS = [(OURS, V3), (FULL, V3)]
CONTRACT_DOC = "docs/llava_next_mtvqa_main_contract.md"
PHASES = ("gate", "pilot", "main")
RESAMPLES, SEED = 10000, 1234
SEGMENTS = 10
MB = 1e6
V3_NOTE = "v3 기록(2026-10 실행)의 값 — 이번 실행에서 잰 값이 아님"


# ------------------------------------------------------------- per image io
def raw_rows(folder, commit):
    """Rows of a committed image; the journal is checked against the SHA256 of its commit."""
    attempt = folder / commit["attempt_id"]
    archive = commit.get("journal_archives", {}).get("raw.jsonl")
    path = attempt / (archive["path"] if archive else "raw.jsonl")
    data = gzip.open(path, "rb").read() if path.suffix == ".gz" else path.read_bytes()
    if hashlib.sha256(data).hexdigest() != commit["raw_sha256"]:
        raise ValueError("committed raw SHA256 mismatch")
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


def image_record(args):
    """One committed image -> compact record (worker; never raises)."""
    folder, methods = Path(args[0]), args[1]
    try:
        commit = json.loads((folder / "COMMITTED.json").read_text())
        attempt = folder / commit["attempt_id"]
        rows = raw_rows(folder, commit)
        out = {"image_id": folder.name, "status": "OK", "requests": len(rows), "rows": [], "attempts_before": commit.get("attempts_before", 0),
               "image_seconds": commit.get("image_seconds"), "committed_at": commit.get("committed_at_unix"),
               "t1_equal_v3": commit.get("t1_equal_v3", {}), "pid": commit.get("process_id")}
        for r in rows:
            if r["method_id"] not in methods:
                continue
            res = r["result"]
            c = {"m": r["method_id"], "t": r["turn_id"], "score": r["score"], "ttft": r["ttft_ms"], "e2e": r["request_e2e_ms"],
                 "bytes": r["ssd_read_bytes"], "reads": r["ssd_preads"], "probe": r["probe_read_bytes"], "k": r["k_read_bytes"], "v": r["v_read_bytes"],
                 "packed": r["packed_read_bytes"], "blocked": r["read_blocked_ms"], "prefill": res.get("prefill_ms"), "hook": res.get("hook_ms"),
                 "scatter": res.get("scatter_ms"), "at": r.get("completed_at_unix"), "tok": r["generated_token_ids"],
                 "pos": r["method_order_position"], "atype": r["annotation"]["answer_type"], "n": r["N_content"]}
            ahead = (res.get("read_wait") or {}).get("readahead") or {}
            if ahead:
                c["worker"] = ahead.get("worker_read_ms"); c["ready"] = ahead.get("ready_on_arrival")
            if r["turn_id"] > 1 and r["family"] == "impress_official":
                log = res["layer_log"]; g = r["geometry"]
                c["imp"] = {"fb": [int(x["mode"] == "fallback") for x in log], "sim": [x["sim"] for x in log],
                            "kc": [x["k_chunks"] for x in log], "vc": [x["v_chunks"] for x in log], "nc": (g["v_token_num"] + 63) // 64,
                            "raster": res["raster_chunks"]["chunks"], "only": res["importance_only_chunks"]["chunks"],
                            "share": [d["prefix_share"] for d in res["layer_diagnostics"]],
                            "blocked_probe": res.get("probe_hook_blocked_ms"), "ready": res.get("prefetch_ready_on_arrival")}
            out["rows"].append(c)
        store = {}
        for method, kind in KIND.items():
            pp = attempt / f"{kind}_persistence.json"
            if method not in methods or not pp.exists():
                continue
            p = json.loads(pp.read_text())
            a = json.loads((attempt / f"{kind}_activation.json").read_text())
            t = p.get("timing_ms") or {}
            s = {"logical_bytes": p["file_logical_bytes"], "kv_bytes": p["bytes"]["visual_kv"], "probe_bytes": p["bytes"].get("probe_sidecar", 0),
                 "sep_bytes": p["bytes"].get("separator_sidecar", 0), "writer_wall_ms": p["native_writer_wall_ms"],
                 "service_ms": p["service_persistence_ms"], "hash_ms": p["completion_full_hash_ms"], "persist_ms": t.get("persist_ms"),
                 "ssd_write_ms": t.get("ssd_write_ms"), "durability_ms": t.get("durability_ms"), "os_write_bytes": p["os_write_bytes"],
                 "activation_ms": a["outer_wall_ms"], "after_request_ms": 0.0, "capture_ms": None}
            if kind == "impress_official":
                o = p["official_turn1"]
                s.update(capture_ms=o["capture_ms"], after_request_ms=o["total_after_request_ms"], layout_ms=o["layout_ms"])
            store[method] = s
        out["store"] = store
        return out
    except Exception as exc:      # noqa: BLE001  (reported by the caller)
        return {"image_id": folder.name, "status": "ERROR", "error": repr(exc)}


def collect(run, phase, manifest, methods, workers=12):
    images = manifest["phase_images"][phase]
    present = [run / phase / "images" / i for i in images if (run / phase / "images" / i / "COMMITTED.json").exists()]
    if not present:
        return [], len(images)
    with Pool(workers) as pool:
        records = pool.map(image_record, [(str(f), methods) for f in present], chunksize=32)
    bad = [r for r in records if r["status"] != "OK"]
    if bad:
        raise RuntimeError("unreadable committed images: " + repr(bad[:3]))
    for r in records:
        r["by"] = {(c["m"], c["t"]): c for c in r["rows"]}
    return records, len(images)


def load_v3(run, v3_phase="main"):
    from mmimpress.mtvqa25k_packed import reference as REF
    return REF.load_index(run / "reference" / f"v3_{v3_phase}_index.jsonl.gz")


def attach_v3(records, v3):
    """Add the v3 ReComp row of the same image and turn to every record as pseudo method ``V3``."""
    for r in records:
        ref = v3[r["image_id"]]
        for t in (1, 2, 3):
            x = ref[(V3_RECOMP, t)]
            r["by"][(V3, t)] = {"m": V3, "t": t, "score": x["score"], "ttft": x["ttft_ms"], "e2e": x["request_e2e_ms"], "bytes": 0, "reads": 0,
                                "tok": x["generated_token_ids"]}
            f = ref.get((V3_FULL, t))
            if f is not None:
                r["by"][("v3_full", t)] = {"score": f["score"], "ttft": f["ttft_ms"], "tok": f["generated_token_ids"]}
    return records


# ---------------------------------------------------------------- analysis
def cells(records, method, turns=(2, 3)):
    return [r["by"][(method, t)] for r in records for t in turns]


def arm_row(records, m, name, note=""):
    sc = {t: np.array([c["score"] for c in cells(records, m, (t,))]) for t in (1, 2, 3)}
    fu = cells(records, m)
    ttft = np.array([c["ttft"] for c in fu]); t1 = np.array([c["ttft"] for c in cells(records, m, (1,))])
    by = {k: np.array([c.get(k, 0) for c in fu], dtype=float) for k in ("bytes", "probe", "k", "v", "packed", "reads")}
    return {"method": name, "method_id": m, "note": note, "images": len(records), "followup_requests": len(fu),
            "score_followup": 100 * np.concatenate([sc[2], sc[3]]).mean(), "score_T1": 100 * sc[1].mean(), "score_T2": 100 * sc[2].mean(),
            "score_T3": 100 * sc[3].mean(), "followup_ttft_mean_ms": ttft.mean(), "followup_ttft_p50_ms": np.percentile(ttft, 50),
            "followup_ttft_p95_ms": np.percentile(ttft, 95), "ssd_mb_per_followup": by["bytes"].mean() / MB,
            "read_calls_per_followup": by["reads"].mean(), "read_calls_min": by["reads"].min(), "read_calls_max": by["reads"].max(),
            "probe_mb": by["probe"].mean() / MB, "k_mb": by["k"].mean() / MB, "v_mb": by["v"].mean() / MB, "packed_mb": by["packed"].mean() / MB,
            "t1_ttft_mean_ms": t1.mean()}


def arm_table(records, methods, with_v3=True):
    rows = []
    if with_v3:
        rows.append(arm_row(records, V3, "ReComp", V3_NOTE))
    for m in methods:
        rows.append(arm_row(records, m, NAME[m]))
    return rows


def per_image(records, a, b, key, scale=1.0):
    """[(sum_a, sum_b, count)] per image over the follow-up requests (same image, same turn)."""
    return [((r["by"][(a, 2)][key] + r["by"][(a, 3)][key]) * scale, (r["by"][(b, 2)][key] + r["by"][(b, 3)][key]) * scale, 2) for r in records]


def ratio_bootstrap(per):
    """Image-cluster bootstrap of mean(a) / mean(b) (the resampling of the paired difference)."""
    arr = np.asarray(per, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(arr), size=(RESAMPLES, len(arr)))
    sa = arr[:, 0][idx].sum(1); sb = arr[:, 1][idx].sum(1)
    lo, hi = np.percentile(sa / sb, [2.5, 97.5])
    return {"ratio": float(arr[:, 0].sum() / arr[:, 1].sum()), "ci_low": float(lo), "ci_high": float(hi)}


def label(m):
    return "ReComp" if m == V3 else NAME[m]


def paired_table(records, pairs, key, scale, unit):
    out = []
    for a, b in pairs:
        cross = V3 in (a, b)
        row = {"comparison": f"{label(a)} − {label(b)}", "images": len(records), "unit": unit,
               "basis": ("ReComp = v3 기록, 같은 이미지·같은 턴끼리 짝지음 (다른 실행)" if cross else "같은 실행")
                        + f"; 이미지 단위 paired bootstrap {RESAMPLES:,}회, seed {SEED}, followup(T2–T3)"}
        if len(records) > 1:
            pb = paired_bootstrap(per_image(records, a, b, key, scale), RESAMPLES, SEED)
            row.update(diff=pb["mean"], ci_low=pb["ci_low"], ci_high=pb["ci_high"], ci_excludes_zero=not (pb["ci_low"] <= 0 <= pb["ci_high"]))
        out.append(row)
    return out


def ratio_table(records, ratios):
    out = []
    for a, b in ratios:
        cross = V3 in (a, b)
        rb = ratio_bootstrap(per_image(records, a, b, "ttft")) if len(records) > 1 else {"ratio": None, "ci_low": None, "ci_high": None}
        out.append({"ratio": f"{label(a)} / {label(b)}", "ttft_ratio_of_means": rb["ratio"], "ci_low": rb["ci_low"], "ci_high": rb["ci_high"],
                    "note": "ReComp TTFT = v3 기록 (다른 실행에서 잰 값)" if cross else "같은 실행"})
    return out


def impress_internal(records):
    imp = [c["imp"] for c in cells(records, IMP)]
    fb = np.array([x["fb"] for x in imp], dtype=float); sim = np.array([x["sim"] for x in imp], dtype=float)
    kc = np.array([x["kc"] for x in imp], dtype=float); vc = np.array([x["vc"] for x in imp], dtype=float)
    nc = np.array([x["nc"] for x in imp], dtype=float)[:, None]
    raster = np.array([x["raster"] for x in imp], dtype=float); only = np.array([x["only"] for x in imp], dtype=float)
    share = np.array([x["share"] for x in imp], dtype=float)
    cons = fb == 0
    kv = (kc + vc) / (2 * nc)

    def frac(x, mask=None):
        v = x / nc
        return float(v.mean() if mask is None else v[mask].mean()) if (mask is None or mask.any()) else None
    layers = [{"layer": li, "fallback_rate": float(fb[:, li].mean()), "mean_jaccard": float(sim[:, li].mean()), "kv_chunk_fraction": float(kv[:, li].mean()),
               "raster_chunk_fraction_computed": float((raster[:, li:li + 1] / nc).mean()), "prefix_share": float(share[:, li].mean())}
              for li in range(fb.shape[1])]
    return {"followup_requests": int(fb.shape[0]), "fallback_rate": float(fb.mean()), "mean_jaccard": float(sim.mean()), "threshold": 0.3,
            "kv_chunk_fraction": float(kv.mean()), "consensus_kv_chunk_fraction": float(kv[cons].mean()) if cons.any() else None,
            "k_chunk_fraction": frac(kc), "v_chunk_fraction": frac(vc),
            "raster_chunk_fraction_computed": frac(raster), "consensus_raster_chunk_fraction_computed": frac(raster, cons),
            "importance_only_chunk_fraction_computed": frac(only), "consensus_importance_only_chunk_fraction_computed": frac(only, cons),
            "prefix_share": float(share.mean()),
            "requests_with_any_fallback": float((fb.sum(axis=1) > 0).mean()), "requests_all_layers_fallback": float((fb.sum(axis=1) == fb.shape[1]).mean()),
            "probe_ready_on_arrival_mean": float(np.mean([x["ready"] for x in imp])),
            "probe_hook_blocked_ms": float(np.mean([x["blocked_probe"] for x in imp])), "layers": layers}


def decomposition(records, methods):
    """Follow-up turns: time the request thread waited for reads, and time spent loading rows to the GPU."""
    out = []
    for m in methods:
        fu = cells(records, m)
        blocked = np.array([c["blocked"] for c in fu], dtype=float); hook = np.array([c["hook"] for c in fu], dtype=float)
        row = {"method": NAME[m], "followup_requests": len(fu), "ttft_mean_ms": float(np.mean([c["ttft"] for c in fu])),
               "prefill_mean_ms": float(np.mean([c["prefill"] for c in fu])),
               "read_wait_mean_ms": float(blocked.mean()), "read_wait_p95_ms": float(np.percentile(blocked, 95)),
               "hook_mean_ms": float(hook.mean()), "hook_minus_read_wait_mean_ms": float((hook - blocked).mean())}
        if all(c.get("scatter") is not None for c in fu):
            row["gpu_load_mean_ms"] = float(np.mean([c["scatter"] for c in fu]))
            row["gpu_load_source"] = "적재 호출 구간의 합(행과 separator를 GPU cache의 원래 위치에 쓰고 mask를 만듦)"
        elif m == FULL:
            row["gpu_load_mean_ms"] = row["hook_minus_read_wait_mean_ms"]
            row["gpu_load_source"] = "layer hook 시간 − 읽기 대기 (모든 행의 host FP16 → GPU BF16 복사)"
        else:
            row["gpu_load_mean_ms"] = row["hook_minus_read_wait_mean_ms"]
            row["gpu_load_source"] = "layer hook 시간 − 읽기 대기 (GPU 복사에 probe 중요도 계산·선택이 함께 들어 있어 분리되지 않음)"
        if all(c.get("worker") is not None for c in fu):
            row["worker_read_mean_ms"] = float(np.mean([c["worker"] for c in fu]))
        out.append(row)
    return out


def storage_table(records, methods):
    out = []
    for m in methods:
        s = [r["store"][m] for r in records if m in r["store"]]
        if not s:
            continue

        def mean(key):
            v = [x[key] for x in s if x.get(key) is not None]
            return float(np.mean(v)) if v else None
        total = np.array([x["writer_wall_ms"] for x in s])
        out.append({"method": NAME[m], "store_kind": KIND[m], "images": len(s), "store_mb_per_image": mean("logical_bytes") / MB,
                    "store_mb_max": max(x["logical_bytes"] for x in s) / MB, "visual_kv_mb": mean("kv_bytes") / MB, "probe_copy_mb": mean("probe_bytes") / MB,
                    "writer_ms_mean": float(total.mean()), "writer_ms_p95": float(np.percentile(total, 95)),
                    "impress_layout_after_request_ms": mean("after_request_ms") if m == IMP else None,
                    "impress_capture_ms_inside_t1": mean("capture_ms"),
                    "integrity_hash_ms_harness": mean("hash_ms"), "service_ms_incl_hash": mean("service_ms"), "activation_ms": mean("activation_ms"),
                    "os_write_mb_per_image": mean("os_write_bytes") / MB, "os_write_bytes_total": int(sum(x["os_write_bytes"] for x in s))})
    return out


def continuity(records, v3):
    """Turn-1 tokens of the three methods against the v3 turn-1 tokens; FullLoad follow-ups against the v3 FullLoad record."""
    out = {"images": len(records), "t1": {}, "t1_mismatch_images": []}
    for m in MAIN:
        same = 0
        for r in records:
            ok = r["by"][(m, 1)]["tok"] == r["by"][(V3, 1)]["tok"]
            same += ok
            if not ok and len(out["t1_mismatch_images"]) < 200:
                out["t1_mismatch_images"].append({"image_id": r["image_id"], "method": NAME[m]})
        out["t1"][NAME[m]] = {"equal": same, "of": len(records)}
    out["t1_all_three_equal_images"] = sum(all(r["by"][(m, 1)]["tok"] == r["by"][(V3, 1)]["tok"] for m in MAIN) for r in records)
    full = {"requests": 0, "tokens_equal": 0, "by_turn": {}}
    for t in (2, 3):
        pairs = [(r["by"][(FULL, t)], r["by"][("v3_full", t)]) for r in records if ("v3_full", t) in r["by"]]
        eq = sum(a["tok"] == b["tok"] for a, b in pairs)
        full["by_turn"][f"T{t}"] = {"requests": len(pairs), "tokens_equal": eq}
        full["requests"] += len(pairs); full["tokens_equal"] += eq
    a = np.array([c["score"] for c in cells(records, FULL)]); b = np.array([r["by"][("v3_full", t)]["score"] for r in records for t in (2, 3)])
    tb = np.array([r["by"][("v3_full", t)]["ttft"] for r in records for t in (2, 3)]); ta = np.array([c["ttft"] for c in cells(records, FULL)])
    full.update(score_this_run=100 * float(a.mean()), score_v3=100 * float(b.mean()), score_diff_pp=100 * float(a.mean() - b.mean()),
                ttft_this_run_ms=float(ta.mean()), ttft_v3_ms=float(tb.mean()),
                note="v3의 FullLoad는 동기 읽기였고 이번 FullLoad는 비동기 읽기다. 생성 토큰은 읽기 방식과 무관해야 하므로 대조하고, TTFT는 참고로만 적는다.")
    out["fullload_vs_v3_fullload"] = full
    return out


def segments(records, samples):
    """The main phase in ``SEGMENTS`` equal parts by commit order: per-method mean follow-up TTFT and the SSD temperature."""
    order = sorted(records, key=lambda r: r["committed_at"] or 0)
    temp = {s["image_id"]: s for s in samples if s.get("label") == "image_committed" and s.get("phase") == "main"}
    out = []
    n = len(order)
    for i in range(SEGMENTS):
        part = order[i * n // SEGMENTS:(i + 1) * n // SEGMENTS]
        if not part:
            continue
        temps = [temp[r["image_id"]]["ssd_temperature_c"] for r in part if r["image_id"] in temp and temp[r["image_id"]].get("ssd_temperature_c") is not None]
        row = {"segment": i + 1, "images": len(part), "from_utc": utc(part[0]["committed_at"]), "to_utc": utc(part[-1]["committed_at"]),
               "ssd_temperature_mean_c": float(np.mean(temps)) if temps else None, "ssd_temperature_max_c": float(np.max(temps)) if temps else None,
               "seconds_per_image": float(np.mean([r["image_seconds"] for r in part if r["image_seconds"] is not None]))}
        for m in MAIN:
            row[NAME[m]] = float(np.mean([c["ttft"] for c in cells(part, m)]))
            row[NAME[m] + "_T1"] = float(np.mean([c["ttft"] for c in cells(part, m, (1,))]))
        out.append(row)
    return out


def recomp_ttft_check(pilot_records, v3):
    """Pilot: the ReComp TTFT measured now against the v3 record of the same requests."""
    out = {"images": len(pilot_records)}
    for name, turns in (("all_turns", (1, 2, 3)), ("followup", (2, 3)), ("T1", (1,))):
        now = np.array([r["by"][(RECOMP, t)]["ttft"] for r in pilot_records for t in turns])
        ref = np.array([v3[r["image_id"]][(V3_RECOMP, t)]["ttft_ms"] for r in pilot_records for t in turns])
        out[name] = {"requests": int(now.size), "this_run_mean_ms": float(now.mean()), "v3_mean_ms": float(ref.mean()),
                     "ratio_this_run_over_v3": float(now.mean() / ref.mean()), "paired_mean_difference_ms": float((now - ref).mean())}
    tok = sum(r["by"][(RECOMP, t)]["tok"] == v3[r["image_id"]][(V3_RECOMP, t)]["generated_token_ids"] for r in pilot_records for t in (1, 2, 3))
    out["tokens_equal"] = {"equal": int(tok), "of": 3 * len(pilot_records)}
    return out


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
    return time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(ts)) if ts else "—"


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def fx(x, d=1, unit=""):
    return "—" if x is None else f"{x:,.{d}f}{unit}"


def pct(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def table(header, rows, align=None):
    align = align or ["---"] + ["---:"] * (len(header) - 1)
    return ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def smart_md(out_root):
    L = []
    docs = {k: read_json(out_root / f"smart_{k}.json") for k in ("before", "after")}
    pre = read_json(out_root / "prerun.json", {}) or {}
    docs["before"] = docs["before"] or pre.get("smart")
    rows = []
    for key in ("percentage_used", "data_units_written", "data_units_written_bytes", "temperature", "available_spare", "media_errors", "power_on_hours"):
        b = (docs["before"] or {}).get(key); a = (docs["after"] or {}).get(key)
        diff = (a - b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
        rows.append([key, "측정 못함" if b is None else f"{b:,}", "측정 못함" if a is None else f"{a:,}", "—" if diff is None else f"{diff:+,}"])
    def hw(d):
        d = d or {}
        t = (d.get("hwmon_temperature") or (d.get("available_without_root") or {}).get("hwmon_temperature") or {}).get("composite_c")
        return t
    tb, ta = hw(docs["before"]), hw(docs["after"])
    rows.append(["온도 — NVMe 센서(composite, root 불필요) °C", fx(tb), fx(ta), "—" if None in (tb, ta) else f"{ta - tb:+.1f}"])
    L += table(["SMART 항목", "실행 전", "실행 후", "차이"], rows)
    for k in ("before", "after"):
        d = docs[k] or {}
        if d.get("status") == "NOT_MEASURED":
            L += ["", "SMART 값(percentage used, data units written)은 읽지 못했다: NVMe SMART 로그는 root 권한이 필요한 admin 명령인데 이 계정에는 비밀번호 없는 sudo가 없고 "
                      "SMART 도구(`nvme-cli`, `smartmontools`)도 설치돼 있지 않다. 온도는 root 없이 읽히는 NVMe 센서 값을 적었고, 쓰기량은 위의 커널 카운터 값을 쓴다."]
            break
    for k in ("before", "after"):
        manual = out_root / f"smart_manual_{k}.txt"
        if manual.exists():
            sys.path.insert(0, str(ROOT / "scripts"))
            import llava_next_main_status as STATUS
            L += ["", f"사람이 넣어 준 `{manual.name}`의 값: `{json.dumps(STATUS.parse_smart_text(manual.read_text()), ensure_ascii=False)}`"]
    return L


def ssd_md(run, out_root, storage, main_complete):
    pre = read_json(out_root / "prerun.json", {}) or {}
    samples = jsonl(run / "ssd_samples.jsonl")
    counter = read_json(run / "ssd_counter.json", {}) or {}
    expected = (pre.get("expected_ssd_write") or {}).get("bytes", {})
    by_phase = {}
    for s in samples:
        if s.get("label") == "image_committed":
            e = by_phase.setdefault(s["phase"], {"images": 0, "store_write_bytes": 0})
            e["images"] += 1; e["store_write_bytes"] += s.get("store_write_bytes") or 0
    rows = []
    for ph in PHASES:
        e = by_phase.get(ph, {"images": 0, "store_write_bytes": 0})
        rows.append([{"gate": "gate (4장)", "pilot": "파일럿 (40장)", "main": "본실험"}[ph], fx(expected.get(ph, 0) / 1e9, 1), fx(e["store_write_bytes"] / 1e9, 1), f"{e['images']:,}"])
    total_measured = sum(e["store_write_bytes"] for e in by_phase.values())
    rows.append(["합계", fx(expected.get("total", 0) / 1e9, 1), fx(total_measured / 1e9, 1), ""])
    L = table(["단계", "예상 store 쓰기량 (GB)", "실측 store 쓰기량 (GB)", "commit된 이미지"], rows)
    L += ["", f"- 예상값: 실행 전 v3 기록의 이미지별 visual token 수로 계산한 store 쓰기량(이미지당 FullLoad store + IMPRESS store 쓰기와 재배치 재기록 + Ours store). "
              f"실측값: commit된 이미지의 store를 쓴 프로세스의 write 바이트 합(`os_write_bytes`).",
          f"- 장치 전체 쓰기량(커널 카운터 `/sys/block/{counter.get('device', 'nvme0n1')}/stat`, 드라이버를 처음 띄운 뒤 누적, 모든 프로세스 포함): "
          f"**{fx((counter.get('written_bytes') or 0) / 1e12, 3)} TB**"
          + (f" (부팅 {counter.get('boots')}회에 걸쳐 합산; 재부팅 직전 마지막 표본 뒤의 쓰기는 포함되지 않는다)" if (counter.get("boots") or 1) > 1 else "")
          + ". 요청 기록·증거 파일·다시 실행한 이미지의 store가 포함된다. NAND에 실제로 쓰인 양(write amplification)은 측정하지 않았다.",
          ""]
    L += smart_md(out_root)
    return L, {"expected": expected, "measured_store_write_bytes": by_phase, "device_written_bytes": counter.get("written_bytes"), "boots": counter.get("boots")}


def execution_md(run, records_main, n_main):
    inv = jsonl(run / "driver_invocations.jsonl"); resume = jsonl(run / "resume_log.jsonl")
    failed = jsonl(run / "main_failed_images.jsonl"); errors = jsonl(run / "main_image_errors.jsonl")
    starts = sorted(run.glob("main_start_*.json"))
    L = [f"- 드라이버를 띄운 횟수: {len(inv)}회 (처음 1회 + 이어서 실행 {max(0, len(inv) - 1)}회)."]
    for i, x in enumerate(inv):
        L.append(f"  - {i + 1}번째: {x.get('at_utc')} · {'처음 실행' if x.get('kind') == 'fresh' else '이어서 실행'} · 그때 단계 `{x.get('stage_at_start')}` · boot {str(x.get('boot_id'))[:8]}")
    L.append(f"- 본실험 프로세스를 시작한 횟수: {len(starts)}회.")
    redo = [x for x in resume if x.get("mode") == "image_restarted_from_turn1"]
    gpu = [x for x in resume if x.get("mode") == "gpu_contamination_image_will_restart"]
    L.append(f"- T1부터 다시 실행한 이미지: {len(redo)}건" + (" — " + ", ".join(f"{x['phase']}:{x['image_id']}({x.get('at_utc')}, 버린 행 {x.get('rows_discarded')})" for x in redo[:60])
                                                  + (" …" if len(redo) > 60 else "") if redo else "") + ".")
    if gpu:
        L.append(f"- 다른 프로세스가 GPU를 써서 이미지를 중단하고 기다린 횟수: {len(gpu)}건.")
    L.append(f"- 본실험에서 두 번 시도해도 끝나지 않아 건너뛴 이미지: {len(failed)}건" + (" — " + ", ".join(x["image_id"] for x in failed) if failed else "") + f" (오류 기록 {len(errors)}건).")
    if records_main:
        secs = np.array([r["image_seconds"] for r in records_main if r["image_seconds"] is not None])
        first = min(r["committed_at"] for r in records_main if r["committed_at"]); last = max(r["committed_at"] for r in records_main if r["committed_at"])
        L.append(f"- 본실험: 이미지 {len(records_main):,}/{n_main:,}장, 이미지당 평균 {secs.mean():.2f} s (p95 {np.percentile(secs, 95):.2f} s), "
                 f"이미지 실행 시간 합 {secs.sum() / 3600:.2f} h, 첫 commit {utc(first)} ~ 마지막 commit {utc(last)} (벽시계 {(last - first) / 3600:.2f} h).")
    if inv:
        st = read_json(ROOT / "results" / run.name / "status.json", {}) or {}
        L.append(f"- 전체: 드라이버 첫 시작 {inv[0].get('at_utc')}, 보고서 작성 {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}"
                 + (f", 종료 기록 {st.get('finished_utc')}" if st.get("finished_utc") else "") + ".")
    return L, {"driver_invocations": inv, "restarted_images": redo, "failed_images": failed, "main_process_starts": len(starts)}


# ------------------------------------------------------------------ tables
def main_tables_md(arms, score_pairs, ttft_pairs, ratios, cross):
    L = []
    L += table(["방법", "followup VQA score (%)", "T2 (%)", "T3 (%)", "TTFT 평균 (ms)", "p50", "p95", "SSD MB/followup", "읽기 호출 수/followup"],
               [[a["method"] + (" †" if a["note"] else ""), fx(a["score_followup"], 3), fx(a["score_T2"], 3), fx(a["score_T3"], 3), fx(a["followup_ttft_mean_ms"]),
                 fx(a["followup_ttft_p50_ms"]), fx(a["followup_ttft_p95_ms"]), fx(a["ssd_mb_per_followup"], 2),
                 fx(a["read_calls_per_followup"], 1)] for a in arms])
    L += ["", f"† ReComp 행은 {V3_NOTE}. score와 TTFT 모두 v3 기록의 요청별 값을 같은 이미지에서 집계했다. "
              "FullLoad·IMPRESS·Ours는 이번 실행에서 잰 값이다. 읽기 호출 수는 read syscall 수, SSD MB는 read syscall이 돌려준 바이트(1 MB = 10^6 B)다.", ""]
    imp = next((a for a in arms if a["method_id"] == IMP), None)
    if imp:
        L += [f"IMPRESS의 종류별 읽기량(followup 평균): probe key 사본 {imp['probe_mb']:.2f} MB, K {imp['k_mb']:.2f} MB, V {imp['v_mb']:.2f} MB. "
              f"읽기 호출 수 범위: " + ", ".join(f"{a['method']} {a['read_calls_min']:.0f}–{a['read_calls_max']:.0f}" for a in arms if not a["note"]) + ".", ""]
    L += ["### 2. 짝지은 비교 (이미지 단위 paired bootstrap 95% CI, followup)", "", "score (pp):", ""]
    L += table(["비교", "차이 (pp)", "95% CI", "CI가 0을 제외", "기준"],
               [[p["comparison"], fx(p.get("diff"), 3), f"[{fx(p.get('ci_low'), 3)}, {fx(p.get('ci_high'), 3)}]",
                 "예" if p.get("ci_excludes_zero") else "아니오", p["basis"].split(";")[0]] for p in score_pairs], ["---", "---:", "---:", "---", "---"])
    L += ["", "TTFT (ms; 같은 실행의 세 방법만):", ""]
    L += table(["비교", "차이 (ms)", "95% CI"],
               [[p["comparison"], fx(p.get("diff"), 2), f"[{fx(p.get('ci_low'), 2)}, {fx(p.get('ci_high'), 2)}]"] for p in ttft_pairs])
    L += ["", f"모든 CI는 이미지 단위 paired bootstrap {RESAMPLES:,}회, seed {SEED}이다. CI가 0을 포함하면 '차이 미입증'이며 동등하다는 뜻이 아니다. "
              "ReComp가 들어간 비교는 v3 기록의 같은 이미지·같은 턴과 짝지은 것으로, 두 값이 서로 다른 실행에서 나왔다.", "",
          "### 3. TTFT 비율 (followup 평균의 비)", ""]
    L += table(["비율", "값", "95% CI (이미지 bootstrap)", "비고"],
               [[r["ratio"], fx(r["ttft_ratio_of_means"], 4), f"[{fx(r['ci_low'], 4)}, {fx(r['ci_high'], 4)}]", r["note"]] for r in ratios + cross],
               ["---", "---:", "---:", "---"])
    L += ["", "Ours/ReComp와 FullLoad/ReComp의 분모는 v3 기록의 ReComp TTFT다(다른 실행에서 잰 값). 이번 장비 상태에서의 ReComp TTFT는 4번(파일럿 40장)에서 따로 확인한다."]
    return L


def impress_md(imp):
    L = table(["항목", "값"], [
        ["followup 요청 수", f"{imp['followup_requests']:,}"],
        ["fallback 비율 (layer·요청 평균)", pct(imp["fallback_rate"])],
        ["fallback layer가 하나라도 있는 요청 / 모든 layer가 fallback인 요청", f"{pct(imp['requests_with_any_fallback'])} / {pct(imp['requests_all_layers_fallback'])}"],
        ["평균 Jaccard (probe head 3개 쌍 평균; 임계값 0.3)", f"{imp['mean_jaccard']:.4f}"],
        ["K·V chunk 읽기 비율 — 전체", pct(imp["kv_chunk_fraction"])],
        ["K·V chunk 읽기 비율 — fallback이 아닌 layer만", pct(imp["consensus_kv_chunk_fraction"])],
        ["재배치하지 않았을 경우(raster 순서)의 K·V chunk 비율 — **계산값**, 전체 / fallback이 아닌 layer만",
         f"{pct(imp['raster_chunk_fraction_computed'])} / {pct(imp['consensus_raster_chunk_fraction_computed'])}"],
        ["separator도 중요도 순서에 두었을 경우 — **계산값**, 전체 / fallback이 아닌 layer만",
         f"{pct(imp['importance_only_chunk_fraction_computed'])} / {pct(imp['consensus_importance_only_chunk_fraction_computed'])}"],
        ["중요도 합에서 prefix 행이 차지하는 비율", pct(imp["prefix_share"])],
        ["미리 읽은 probe key 31개 중 도착 시 이미 읽혀 있던 수 (요청 평균)", f"{imp['probe_ready_on_arrival_mean']:.1f}"],
    ], ["---", "---:"])
    L += ["", "chunk 비율은 layer당 읽은 64토큰 chunk 수 / 그 layer의 chunk 수를 layer·요청으로 평균한 것이다(probe key 사본 제외, fallback layer는 100%). "
              "계산값은 요청의 layer별 유지 mask를 다른 저장 순서의 위치로 옮겨 chunk를 센 것이며 실제로 읽은 값이 아니다.", "", "layer별:", ""]
    L += table(["layer", "fallback", "Jaccard", "K·V chunk 실측", "raster 순서 계산값", "prefix 행 비율"],
               [[x["layer"], pct(x["fallback_rate"]), f"{x['mean_jaccard']:.3f}", pct(x["kv_chunk_fraction"]), pct(x["raster_chunk_fraction_computed"]),
                 pct(x["prefix_share"])] for x in imp["layers"]])
    return L


def decomposition_md(decomp):
    L = table(["방법", "TTFT 평균 (ms)", "읽기 대기 (ms)", "읽기 대기 p95", "GPU 적재 (ms)", "워커의 읽기 시간 (ms)", "layer hook 시간 (ms)", "prefill (ms)"],
              [[d["method"], fx(d["ttft_mean_ms"]), fx(d["read_wait_mean_ms"], 2), fx(d["read_wait_p95_ms"], 2), fx(d.get("gpu_load_mean_ms"), 2),
                fx(d.get("worker_read_mean_ms"), 1), fx(d["hook_mean_ms"], 1), fx(d["prefill_mean_ms"], 1)] for d in decomp])
    L += ["", "- 읽기 대기: 요청 스레드가 읽기 호출 안에 있었거나 워커의 읽기를 기다린 벽시계 시간(`read_blocked_ms`). 읽기와 GPU 연산이 겹치므로 항목의 합이 TTFT가 되지 않는다."]
    for d in decomp:
        L.append(f"- GPU 적재({d['method']}): {d.get('gpu_load_source')}.")
    return L


def storage_md(storage):
    L = table(["방법", "이미지당 store 크기 (MB)", "최대 (MB)", "그중 Visual KV", "probe key 사본", "저장 시간 평균 (ms)", "p95", "요청 뒤 재배치 (ms)", "T1 안의 통계 수집 (ms)", "이미지당 실제 write (MB)"],
              [[s["method"], fx(s["store_mb_per_image"], 1), fx(s["store_mb_max"], 1), fx(s["visual_kv_mb"], 1), fx(s["probe_copy_mb"], 1), fx(s["writer_ms_mean"], 0),
                fx(s["writer_ms_p95"], 0), fx(s["impress_layout_after_request_ms"], 0), fx(s["impress_capture_ms_inside_t1"], 1), fx(s["os_write_mb_per_image"], 1)] for s in storage])
    L += ["", "- 저장 시간: store writer를 부른 시점부터 돌아올 때까지의 벽시계 시간(파일 쓰기와 fsync 포함; IMPRESS는 raster 쓰기 뒤의 재배치 재기록 포함). "
              "하니스가 무결성 확인을 위해 하는 전체 해시(" + ", ".join(f"{s['method']} {fx(s['integrity_hash_ms_harness'], 0)} ms" for s in storage) + ")와 "
              "활성화 때의 확인(" + ", ".join(f"{s['method']} {fx(s['activation_ms'], 0)} ms" for s in storage) + ")은 포함하지 않았다.",
          "- T1 요청의 TTFT·E2E에는 캡처 비용이 들어 있고(그 요청 하나), 저장 시간은 요청 밖이다. 둘 다 followup TTFT에는 들어가지 않는다.",
          "- IMPRESS의 실제 write가 store 크기의 약 2배인 것은 raster 순서로 쓴 뒤 재배치 순서로 다시 쓰기 때문이다(11/17 구현 그대로)."]
    return L


def segments_md(seg):
    L = table(["구간", "이미지", "기간 (UTC)", "FullLoad TTFT (ms)", "IMPRESS", "Ours", "T1 TTFT (세 방법 평균)", "SSD 온도 평균 (°C)", "최고", "이미지당 시간 (s)"],
              [[s["segment"], f"{s['images']:,}", f"{s['from_utc'][5:16]} ~ {s['to_utc'][5:16]}", fx(s["FullLoad"]), fx(s["IMPRESS"]), fx(s["Ours"]),
                fx(np.mean([s["FullLoad_T1"], s["IMPRESS_T1"], s["Ours_T1"]])), fx(s["ssd_temperature_mean_c"]), fx(s["ssd_temperature_max_c"]),
                fx(s["seconds_per_image"], 2)] for s in seg])
    L += ["", "구간은 commit 순서로 나눈 10등분이다. TTFT는 followup(T2–T3) 평균, SSD 온도는 이미지를 commit할 때마다 읽은 NVMe 센서(composite) 값이다."]
    return L


def continuity_md(cont, pilot_gate):
    L = []
    g4 = (pilot_gate or {}).get("gate4_continuity") or {}
    rc = g4.get("recomp_vs_v3") or {}
    L.append(f"- 파일럿 40장: 이번에 돌린 ReComp의 생성 토큰이 v3 기록의 같은 요청과 일치한 건수 "
             f"**{rc.get('requests', 0) - rc.get('mismatch_count', 0)}/{rc.get('requests', 0)}**"
             + (" (gate 4)." if rc else " (파일럿 미실행)."))
    if cont:
        L.append("- 본실험: T1 생성 토큰이 v3 기록의 같은 이미지 T1 토큰과 일치한 건수 — "
                 + ", ".join(f"{k} **{v['equal']:,}/{v['of']:,}**" for k, v in cont["t1"].items())
                 + f"; 세 방법 모두 일치한 이미지 {cont['t1_all_three_equal_images']:,}/{cont['images']:,}.")
        if cont["t1_mismatch_images"]:
            L.append("  - 불일치 목록(앞 200건까지; 전체는 `continuity.json`과 `runs/<run>/main_t1_v3_mismatch.jsonl`): "
                     + ", ".join(f"{x['image_id']}({x['method']})" for x in cont["t1_mismatch_images"][:200]))
        f = cont["fullload_vs_v3_fullload"]
        L.append(f"- 참고 — FullLoad followup을 v3 기록의 FullLoad와 대조: 생성 토큰 일치 {f['tokens_equal']:,}/{f['requests']:,} "
                 f"(T2 {f['by_turn']['T2']['tokens_equal']:,}/{f['by_turn']['T2']['requests']:,}, T3 {f['by_turn']['T3']['tokens_equal']:,}/{f['by_turn']['T3']['requests']:,}); "
                 f"followup VQA score 이번 {f['score_this_run']:.3f}% / v3 {f['score_v3']:.3f}% (차이 {f['score_diff_pp']:+.3f} pp); "
                 f"TTFT 이번 {f['ttft_this_run_ms']:.1f} ms / v3 {f['ttft_v3_ms']:.1f} ms. {f['note']}")
    return L


def gate_md(run, out_root):
    g = {name: read_json(run / "gates" / f"{name}.json") for name in ("tests", "scorer", "gate", "pilot", "admission")}
    pre = read_json(out_root / "prerun.json")

    def st(doc):
        return (doc or {}).get("status", "미실행")

    def cmp_txt(doc):
        if not doc:
            return "미실행"
        txt = f"{doc['requests']}건(followup {doc.get('followup_requests', '?')}건) 중 토큰 불일치 {doc['tokens_differ']}건"
        if doc.get("fields_differ"):
            txt += f", 읽기 바이트 불일치 {max(doc['fields_differ'].values())}건"
        return txt
    L = table(["gate", "결과", "내용"], [], ["---", "---", "---"])
    t = g["tests"]
    L.append(f"| 1. 기존 테스트 | {st(t)} | " + ("; ".join(
        f"transformers {s['transformers']}: {s['ran']}개 실행, 실패 {len(s['failed'])}개(이전부터 실패하던 것 {len(s['failed']) - len(s['new_failures'])}개), 새 실패 {len(s['new_failures'])}개"
        for s in t["suites"]) + f"; 새 테스트 파일 {t['new_unit_tests']['ran']}개 " + ("통과" if t["new_unit_tests"]["ok"] else "실패") if t else "") + " |")
    a = g["gate"] or {}
    g2, g3 = a.get("gate2_ours") or {}, a.get("gate3_impress") or {}
    L.append(f"| 2. Ours 동기 = 비동기, 두 번 실행 | {st(g2) if g2 else st(g['gate'])} | 동기 vs 비동기: {cmp_txt(g2.get('sync_vs_async'))}; 같은 요청 두 번: {cmp_txt(g2.get('executed_twice'))} |")
    L.append(f"| 3. IMPRESS | {st(g3) if g3 else st(g['gate'])} | 유지 100% vs 같은 재배치 store의 FullLoad loader: {cmp_txt(g3.get('retention_100_vs_fullload_loader_on_reordered_store'))}; "
             f"비동기 vs 동기: {cmp_txt(g3.get('sync_vs_async'))} |")
    p = g["pilot"] or {}
    g4 = p.get("gate4_continuity") or {}
    rc, t1 = g4.get("recomp_vs_v3") or {}, g4.get("turn1_of_the_three_methods_vs_v3") or {}
    L.append(f"| 4. 연속성 (파일럿 40장) | {st(g4) if g4 else st(g['pilot'])} | "
             + (f"ReComp 토큰 = v3 기록: {rc.get('requests', 0) - rc.get('mismatch_count', 0)}/{rc.get('requests', 0)}; T1 토큰 = v3 T1: "
                + ", ".join(f"{NAME[m]} {v['requests'] - v['mismatch_count']}/{v['requests']}" for m, v in t1.items() if isinstance(v, dict) and "requests" in v) if g4 else "") + " |")
    s = g["scorer"]
    L.append(f"| 5. 채점기 검증 | {st(s)} | " + (f"`scripts/135_validate_mtvqa25k_scorer.py` 종료 코드 {s['returncode']}, fixture {s['result'].get('synthetic_fixtures')}개, "
                                              f"기존 기록 {s['result'].get('actual_existing_raw_requests')}건, 불일치 {s['result'].get('per_question_mismatches')}건" if s else "") + " |")
    ex = p.get("execution") or {}
    L.append(f"| 6. 파일럿 40장 표 | {'작성됨' if ex.get('status') == 'PASS' else '미실행'} | "
             + (f"{ex.get('requests', 0)}건 감사 PASS (판정 항목 아님; 표는 아래)" if ex.get("status") == "PASS" else "") + " |")
    adm = g["admission"]
    L += ["", f"실행 전 점검: **{(pre or {}).get('status', '미실행')}**" + (f" — {'; '.join(pre['blocking'])}" if pre and pre.get("blocking") else "")
          + f" · 본실험 진입 판정: **{adm['status'] if adm else '미실행'}** (실행 전 점검과 gate 1–5가 모두 통과했을 때만 본실험을 시작한다)."]
    if t:
        for su in t["suites"]:
            old = [x for x in su["failed"] if x not in su["new_failures"]]
            if old:
                L += ["", f"transformers {su['transformers']} 환경에서 이전부터 실패하던 테스트(목록만 적음): " + "; ".join(f"`{x}`" for x in old) + "."]
            if su["new_failures"]:
                L += ["", "새로 실패한 테스트: " + "; ".join(f"`{x}`" for x in su["new_failures"]) + "."]
    if a.get("images"):
        rep = a.get("reported_not_judged") or {}
        r = rep.get("recomp_vs_v3_main") or {}
        L += ["", f"참고(판정 아님, gate 4장): ReComp 토큰 = v3 기록 {r.get('requests', 0) - r.get('mismatch_count', 0)}/{r.get('requests', 0)}; "
                  f"재배치 store를 읽은 FullLoad loader와 raster store의 FullLoad의 토큰이 다른 요청 {rep.get('fullload_reordered_vs_raster_fullload_tokens_differ')}건."]
    return L, g


def gate_failure_md(g, pre):
    L = []
    if pre and pre.get("blocking"):
        L += [f"- 실행 전 점검에서 멈춤: {x}" for x in pre["blocking"]]
    adm = g.get("admission") or {}
    for name, step in (adm.get("steps") or {}).items():
        if step.get("ok") or name == "prerun":
            continue
        doc = g.get(name) or {}
        if step.get("status") == "MISSING":
            L.append(f"- `{name}`: 실행되지 않음(앞 단계가 통과하지 못해 시작하지 않았거나 검사가 결과를 내지 못했다).")
            continue
        L.append(f"- `{name}`: {step.get('status')}" + (f" — 실패 항목: {', '.join(doc.get('failed') or [])}" if doc.get("failed") else ""))

        def walk(part, depth=0):
            if not isinstance(part, dict) or depth > 3:
                return
            if part.get("status") == "FAIL" and part.get("criterion"):
                L.append(f"  - 기준: {part['criterion']}")
                for key in ("examples", "mismatches"):
                    for ex in (part.get(key) or [])[:5]:
                        L.append(f"    - 예: `{json.dumps(ex, ensure_ascii=False)[:400]}`")
                if "tokens_differ" in part:
                    L.append(f"    - 요청 {part.get('requests')}건 중 토큰 불일치 {part.get('tokens_differ')}건(그중 프롬프트가 같은 요청 {part.get('tokens_differ_same_prompt')}건), "
                             f"읽기 바이트 불일치 {part.get('fields_differ')}, 빠진 요청 {part.get('missing')}건")
                if "mismatch_count" in part:
                    L.append(f"    - 요청 {part.get('requests')}건 중 불일치 {part.get('mismatch_count')}건, 빠진 요청 {part.get('missing')}건")
            for v in part.values():
                walk(v, depth + 1)
        if name == "tests":
            for su in doc.get("suites", []):
                if su.get("new_failures"):
                    L.append(f"  - transformers {su['transformers']}: 새 실패 " + "; ".join(su["new_failures"]))
            if not (doc.get("new_unit_tests") or {}).get("ok"):
                L.append("  - 새 테스트 파일이 통과하지 못했다.")
        elif name == "scorer":
            L.append(f"  - 종료 코드 {doc.get('returncode')}: `{(doc.get('stderr_tail') or '')[-400:]}`")
        else:
            if "execution" in (doc.get("failed") or []):
                L.append(f"  - 요청 단계가 끝나지 않았다: `{json.dumps({k: (doc.get('execution') or {}).get(k) for k in ('status', 'requests', 'expected', 'failed_images')}, ensure_ascii=False)}`")
            walk(doc)
    return L or ["- 기록된 실패 항목이 없다(판정 전 단계에서 멈췄다)."]


def pilot_md(run, manifest, v3):
    records, n = collect(run, "pilot", manifest, [RECOMP] + MAIN)
    if len(records) != n or not records:
        return None, None
    attach_v3(records, v3)
    arms = [arm_row(records, m, NAME[m]) for m in MAIN + [RECOMP]]
    check = recomp_ttft_check(records, v3)
    L = table(["방법", "followup VQA score (%)", "T2 (%)", "T3 (%)", "TTFT 평균 (ms)", "p50", "p95", "SSD MB/followup", "읽기 호출 수"],
              [[a["method"], fx(a["score_followup"], 2), fx(a["score_T2"], 2), fx(a["score_T3"], 2), fx(a["followup_ttft_mean_ms"]), fx(a["followup_ttft_p50_ms"]),
                fx(a["followup_ttft_p95_ms"]), fx(a["ssd_mb_per_followup"], 2), fx(a["read_calls_per_followup"], 1)] for a in arms])
    L += ["", f"파일럿 {n}장 × 3턴 × 4 arm = {4 * 3 * n}요청(본실험 분모에 넣지 않는다). 네 방법 모두 이번 실행에서 잰 값이다. 이미지 40장이라 score의 표본 오차가 크다."]
    return L, {"arms": arms, "recomp_ttft_check": check, "images": n}


def recomp_check_md(check):
    if not check:
        return ["파일럿이 실행되지 않아 확인하지 못했다."]
    L = table(["구간", "요청 수", "이번 실행 ReComp TTFT 평균 (ms)", "v3 기록 같은 요청 평균 (ms)", "비율 (이번 / v3)", "짝지은 평균 차이 (ms)"],
              [[{"all_turns": "T1–T3 전체", "followup": "followup (T2–T3)", "T1": "T1"}[k], check[k]["requests"], fx(check[k]["this_run_mean_ms"]), fx(check[k]["v3_mean_ms"]),
                fx(check[k]["ratio_this_run_over_v3"], 4), fx(check[k]["paired_mean_difference_ms"], 2)] for k in ("all_turns", "followup", "T1")])
    L += ["", f"파일럿 40장에서 ReComp를 함께 돌려 잰 값과 v3 기록의 같은 요청(같은 이미지·같은 턴)의 값이다. 생성 토큰은 {check['tokens_equal']['equal']}/{check['tokens_equal']['of']}건 일치했다. "
              "이 비율이 1에서 벗어난 만큼 3번의 Ours/ReComp, FullLoad/ReComp 비율에는 실행 간 차이가 섞여 있다."]
    return L


CODE_MD = [
    "| 파일 | 내용 |", "|---|---|",
    "| `scripts/209_eval_llava_next_mtvqa_main.py` (새 파일) | 실행기. freeze, 실행 전 점검, gate·파일럿·본실험 단계, 이미지 단위 commit·store 삭제·이어 실행. "
    "FullLoad·IMPRESS는 `scripts/196`(11/17 실행)을 import해 캡처·재배치·읽기 추적 함수를 그대로 부르고 요청 호출도 그 스크립트의 호출과 같다. "
    "Ours는 `scripts/202`(11/20 실행)의 writer 호출(`row_order=\"original\"`)과 요청 호출(`request_packed`)과 같다. |",
    "| `scripts/210_validate_llava_next_mtvqa_main.py` (새 파일) | gate 1–6 판정과 본실험 진입 판정 |",
    "| `scripts/211_report_llava_next_mtvqa_main.py` (새 파일) | 이 보고서와 파일럿 표 (측정에 관여하지 않음) |",
    "| `scripts/llava_next_main_status.py` (새 파일) | status.json, SSD 쓰기량·온도·SMART 기록 |",
    "| `scripts/run_llava_next_mtvqa_main.sh` (새 파일) | 드라이버: 실행 전 점검부터 보고서까지, 터미널과 분리, 잠금 파일, 이어 실행 |",
    "| `tests/test_llava_next_mtvqa_main.py` (새 파일) | 실행기·보고기의 단위 테스트 |",
    "| `docs/llava_next_mtvqa_main_contract.md` (새 파일) | 계약 문서 (실행 전 동결) |",
    "| `mmimpress/**`, 기존 스크립트·테스트 | 수정 없음. freeze가 11/17·11/20 실행의 소스 해시와 대조해 확인했다. |",
]


def notes_md(out_root):
    p = out_root / "notes.md"
    return ["", p.read_text().rstrip()] if p.exists() else []


def deviations_md(config, pre):
    smart = (pre or {}).get("smart") or {}
    L = [
        "지시와 다르게 했거나 지시에 없던 것:",
        "",
        "- gate 4장 단계의 arm은 9개다: gate 2·3에 필요한 7개(IMPRESS 동기·비동기·유지 100%, 재배치 store를 읽는 FullLoad loader, Ours 동기·비동기·두 번째 실행)에 "
        "ReComp와 FullLoad를 더했다(본실험 방법 세 가지의 실행 경로를 파일럿 전에 한 번 통과시키기 위함; 판정에는 쓰지 않는다).",
        "- gate 4의 ReComp 대조 기준은 v3 기록의 본실험(main) 단계다(보고서의 ReComp 행이 쓰는 기록). v3의 pilot40 단계 기록과의 대조는 참고로 gate 파일에 적었다.",
        "- arm 순서 회전은 단계 안에서의 이미지 순번 기준이다(순번 % arm 수). 본실험은 3 arm, 파일럿은 4 arm 기준으로 돈다.",
        "- 본실험에서 한 이미지가 실행 오류나 감사 실패로 끝나지 않으면 T1부터 한 번 더 시도하고, 그래도 안 되면 그 이미지를 기록하고 건너뛴다"
        f"(건너뛴 이미지가 {config.get('main_failure_policy', {}).get('max_failed_images', 25)}장을 넘거나 연속으로 실패하면 멈춘다). 며칠짜리 실행이 이미지 한 장 때문에 멈추지 않게 하기 위함이며, 건너뛴 이미지는 모든 표에서 빠진다.",
        "- 다른 프로세스가 GPU를 쓰면 실행을 멈추지 않고 그 프로세스가 끝날 때까지 기다린다(최대 6시간). 그때 진행 중이던 이미지는 T1부터 다시 한다.",
        "- 본실험에서는 요청이 끝난 뒤 store가 바뀌지 않았는지를 파일 크기·수정 시각·inode로 확인한다(11/17 실행은 전체 재해시). gate·파일럿에서는 전체 재해시도 한다. "
        "저장 직후의 전체 해시와 활성화 때의 확인은 11/17 실행과 같다.",
        "- 본실험에서는 이미지 감사를 통과한 뒤 IMPRESS의 T1 통계 증거 파일(이미지당 약 1.2 MB)을 지우고 store 메타데이터 사본을 gzip으로 보관한다(SHA256은 남긴다). gate·파일럿은 그대로 둔다.",
        "- TTFT 비율에 이미지 단위 bootstrap 95% CI를 함께 적었다.",
        "- 실행 전 예상 소요 시간은 개발 리허설(이미지 몇 장) 값으로 적었고, 파일럿이 끝난 뒤 파일럿 실측값으로 `prerun_pilot_estimate.json`과 status.json에 다시 적었다.",
    ]
    if smart.get("status") == "NOT_MEASURED":
        L.append("- **SMART 값(percentage used, data units written)을 읽지 못했다.** NVMe SMART 로그는 root 권한이 필요한데 이 계정에는 비밀번호 없는 sudo가 없고 "
                 "`nvme-cli`/`smartmontools`도 설치돼 있지 않다. 대신 root 없이 읽히는 값(NVMe 센서 온도, 커널의 장치 쓰기 카운터)을 기록했다. "
                 "`sudo nvme smart-log /dev/nvme0` 출력을 `results/<run>/smart_manual_before.txt` / `smart_manual_after.txt`로 넣으면 보고서를 다시 만들 때 반영된다.")
    L += [
        "",
        "측정하지 않은 것과 한계:",
        "",
        "- ReComp는 이번 본실험에서 돌리지 않았다. ReComp 행과 ReComp가 들어간 비교·비율은 v3 기록(2026-10)의 값이며, 장비 상태(SSD 사용량·온도, 드라이버, 동시에 도는 arm 구성)가 다른 실행에서 잰 값이다. "
        "TTFT의 실행 간 차이는 파일럿 40장의 ReComp로만 확인했다.",
        "- v3의 ReComp는 한 이미지에서 5개 arm이 함께 도는 조건에서, 이번 세 방법은 3개 arm이 도는 조건에서 측정됐다.",
        "- MT-VQA 채점기는 공식 evaluator의 조건부 정규화를 그대로 따른다(사람 답 10개가 모두 같으면 소문자화 등을 하지 않음). 대문자로 시작하는 답은 그런 문항에서 0점이 되므로 절대 점수가 낮게 나오고 방법 간 차이가 압축될 수 있다(기존 MT-VQA 실행과 같은 지표).",
        "- cold SSD는 요청 직전 `posix_fadvise(DONTNEED)`로 page cache를 비운 조건이다. NAND 수준의 cold 상태나 O_DIRECT는 아니다.",
        "- SSD 쓰기량은 커널 카운터와 프로세스 write 바이트다. NAND에 실제로 쓰인 양(write amplification)과 SSD 수명 소모는 SMART 없이는 알 수 없다.",
        "- 반복 실행 없음(한 번 측정). 다른 유지 비율·임계값·probe head, warm cache, 다른 모델·데이터셋은 이 실행의 범위가 아니다.",
        "- IMPRESS의 GPU 적재 시간은 probe 중요도 계산·선택과 분리해 재지 못했다(6번 표의 설명 참조).",
    ]
    return L


def settings_md(run, config, manifest):
    env = config["environment"]
    return [
        f"- 모델 `{config['model']}` revision `{config['revision']}`, NF4 / BF16 / eager, FP16 SSD KV, greedy cap 16, seed 1234, batch 1, CPU thread 2.",
        f"- 환경: `{env['executable']}`, transformers {env['transformers']}, torch {env['torch']}, numpy {env['numpy']} (11/17·11/20 실행과 같음을 freeze가 확인).",
        "- 데이터: MT-VQA-v2-reconstructed-25k, 이미지 25,000장 × 대화 1개 × 3턴. 지표: 공식 ten-answer leave-one-out VQA score, followup(T2–T3).",
        "- 모든 방법이 T1을 픽셀부터 계산한다(warm-up 없음). T2–T3는 자기 방법의 생성 이력을 쓴다. 저장 KV를 읽는 요청 직전에 그 store의 page cache를 비운다(TTFT 밖). 세 방법 모두 비동기 읽기다.",
        f"- FullLoad·IMPRESS 구현: `{config['implementation_reference']['fullload_impress']['run']}`의 것 그대로. Ours: `{config['implementation_reference']['ours']['run']}`의 \"Ours(원래 순서)\" 그대로"
        " (PYR-beta-raw 벡터, 총 KV 예산 25%, 중요도 순 재배열 없음).",
        f"- 계약 문서 `{CONTRACT_DOC}` SHA256 `{config['contract_sha256'][:16]}…` (실행 전 동결). 요청 수: gate {manifest['expected_requests']['gate']}, "
        f"파일럿 {manifest['expected_requests']['pilot']}, 본실험 {manifest['expected_requests']['main']:,}.",
    ]


def stat_inventory(run):
    """Fingerprint of (path, size, mtime_ns) of every file under results/, runs/, kvstore* outside this run
    (the same walk as the freeze's ``protection_before.json``)."""
    import os
    own = {run.resolve(), (ROOT / "results" / run.name).resolve()}
    digest = hashlib.sha256(); files = 0; total = 0
    roots = [ROOT / "results", ROOT / "runs"] + sorted(p for p in ROOT.glob("kvstore*") if p.is_dir())
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            here = Path(dirpath)
            dirnames[:] = sorted(d for d in dirnames if (here / d).resolve() not in own)
            for name in sorted(filenames):
                p = here / name
                try:
                    st = p.lstat()
                except FileNotFoundError:
                    continue
                digest.update(f"{p.relative_to(ROOT)}\t{st.st_size}\t{st.st_mtime_ns}\n".encode())
                files += 1; total += st.st_size
    return {"files": files, "bytes": total, "sha256": digest.hexdigest()}


def protection_check(run, out_root):
    before = read_json(run / "protection_before.json")
    if not before:
        return None
    after = stat_inventory(run)
    doc = {"status": "UNCHANGED" if after["sha256"] == before["sha256"] else "CHANGED", "before": {k: before[k] for k in ("files", "bytes", "sha256")},
           "after": after,
           "note": "fingerprint of (path, size, mtime) of every file under results/, runs/, kvstore* outside this run; other sessions writing new runs change it"}
    (out_root / "protection_after.json").write_text(json.dumps(doc, indent=1) + "\n")
    return doc


# ------------------------------------------------------------------ report
def report_auto(run, manifest, config, out_root, check_protection=True):
    pre = read_json(out_root / "prerun.json")
    gate_lines, g = gate_md(run, out_root)
    admitted = (g["admission"] or {}).get("status") == "PASS"
    main_done = (run / "main_execution_complete.json").exists()
    st = read_json(out_root / "status.json", {}) or {}
    v3 = load_v3(run) if (run / "reference" / "v3_main_index.jsonl.gz").exists() else None
    records, n_main = (collect(run, "main", manifest, MAIN) if admitted else ([], len(manifest["phase_images"]["main"])))
    L = [f"# LLaVA-NeXT × MT-VQA 25k 본실험 — ReComp / FullLoad / IMPRESS / Ours", "",
         f"run `{run.name}` · 보고서 작성 {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())} · 계약 `{CONTRACT_DOC}`", ""]
    if main_done:
        L += [f"**상태: 본실험 완료** — 이미지 {len(records):,}/{n_main:,}장, {3 * 3 * len(records):,}요청(FullLoad·IMPRESS·Ours). 측정 보고이며 GO/NO-GO 판정은 두지 않는다.", ""]
    elif admitted:
        L += [f"**상태: 본실험 진행 중 또는 중단됨(중간 보고)** — commit된 이미지 {len(records):,}/{n_main:,}장. 아래 표는 그때까지 commit된 이미지만의 값이며 최종 결과가 아니다. "
              f"마지막 오류: `{json.dumps(st.get('last_error'), ensure_ascii=False)}`. 같은 명령으로 이어서 실행된다: `scripts/run_llava_next_mtvqa_main.sh runs/{run.name}`", ""]
    else:
        L += ["**상태: 본실험을 시작하지 않았다.** 실행 전 점검 또는 gate가 통과하지 못했다(또는 아직 끝나지 않았다). 기준은 바꾸지 않았다.", "",
              "## 실패 내용과 원인", ""] + gate_failure_md(g, pre) + [""]
        if st.get("last_error"):
            L += [f"드라이버가 기록한 마지막 오류: `{json.dumps(st.get('last_error'), ensure_ascii=False)}`", ""]
    L += ["## 설정", ""] + settings_md(run, config, manifest) + ["", "## 실행 전 점검과 gate", ""] + gate_lines + [""]
    if pre:
        e = pre["expected_ssd_write"]; f = pre["free_space"]; rec = pre["v3_recomp_record"]; b = pre["ttft_interval_v3_vs_this_run"]
        L += ["실행 전 점검(`prerun.json`):", "",
              f"- 예상 SSD 쓰기량: **{e['total_tb']:.1f} TB** (본실험 {e['main_tb']:.1f} TB; 이미지당 평균 {e['per_image_mean_bytes']['written'] / 1e9:.2f} GB = "
              f"FullLoad store {e['per_image_mean_bytes']['fullload_store'] / 1e9:.2f} GB + IMPRESS store {e['per_image_mean_bytes']['impress_store'] / 1e9:.2f} GB × 약 2(재배치 재기록) + "
              f"Ours store {e['per_image_mean_bytes']['ours_store'] / 1e9:.2f} GB).",
              f"- 예상 소요 시간: 본실험 {pre['expected_duration']['main_hours']:.0f} h (리허설 기준 이미지당 {pre['expected_duration']['seconds_per_main_image']:.1f} s)"
              + (f"; 파일럿 실측 기준 {read_json(out_root / 'prerun_pilot_estimate.json', {}).get('main_hours_estimate', 0):.0f} h"
                 if (out_root / "prerun_pilot_estimate.json").exists() else "") + ".",
              f"- 여유 공간 {f['free_gib']:.1f} GiB, 이미지 하나의 store 묶음 최대 {f['largest_image_bundle_bytes'] / 2**30:.2f} GiB의 {f['ratio_free_to_bundle']:.0f}배 (기준 20배 이상: {'충족' if f['ok'] else '미달'}).",
              f"- v3 기록의 ReComp 요청별 결과: {rec['recomp_requests_complete']:,}/{rec['requests_expected']:,}건(생성 토큰·score·TTFT) — {rec['status']}. "
              f"v3 안에서 T1 토큰이 모든 arm에서 같은 이미지 {rec['v3_turn1_tokens_identical_across_v3_arms_images']:,}장.",
              f"- TTFT 측정 구간이 v3 기록과 같은가: **{'같다' if b['same_interval'] else '다르다'}**. "
              "시작 = GPU 동기화 직후, 프롬프트를 조립하기 직전(이력 구성, 이미지 파일 열기·디코드, 전처리, host→GPU 복사가 구간 안에 있다). "
              "끝 = 첫 출력 토큰이 생성된 시각(`Server.recompute`가 generate 안에서 찍는 시각). "
              "확인 방법: v3 실행이 동결해 둔 소스 사본과 지금 소스를 문장 단위로 대조했다 — "
              f"`Server.recompute` 동일 {b['checks']['serve.recompute_identical']}, 첫 토큰 시각 기록기 동일 {b['checks']['serve._FirstTokenTimestamp_identical']}, "
              f"`normal_request`의 타이머 시작 문장 동일 {b['checks']['timer_start_statement_identical']}, "
              f"`normal_request`에서 다른 문장은 캡처 대상을 고르는 {len(b['normal_request_differing_statements'])}개뿐이고 ReComp 요청에서는 두 쪽 값이 같음 "
              f"{b['checks']['normal_request_differs_only_in_capture_selection'] and b['checks']['capture_selection_identical_for_a_recompute_request']}, "
              f"요청 기록의 TTFT 필드 동일 {b['checks']['row_ttft_field_identical']}.",
              "  - 구간은 같지만 조건이 다른 점: v3는 한 이미지에서 5개 arm(ReComp, FullLoad, MPIC-32, AllHead, [B])을 한 프로세스에서 돌렸고 이번에는 3개(파일럿은 4개)를 돌린다 / "
              "v3는 `Server`를 기본 인자로 만들었고 이번에는 11/17 실행의 인자로 만들며 첫 요청 전에 읽기 워커 스레드를 만들어 둔다(픽셀 요청은 둘 다 쓰지 않는다) / "
              "v3의 FullLoad는 동기 읽기였다(ReComp와는 무관).", ""]
    # pilot
    pl, pj = (pilot_md(run, manifest, load_v3(run, "main")) if v3 is not None and (run / "pilot_execution_complete.json").exists() else (None, None))
    if pl:
        L += ["### 파일럿 40장 표 (gate 6)", ""] + pl + [""]
    summary = {"run": run.name, "main_complete": main_done, "admitted": admitted, "images": len(records), "images_expected": n_main, "pilot": pj}
    if records and v3 is not None:
        attach_v3(records, v3)
        arms = arm_table(records, MAIN)
        sp = paired_table(records, SCORE_PAIRS, "score", 100.0, "pp"); tp = paired_table(records, TTFT_PAIRS, "ttft", 1.0, "ms")
        ratios = ratio_table(records, RATIOS); cross = ratio_table(records, CROSS_RATIOS)
        imp = impress_internal(records); decomp = decomposition(records, MAIN); storage = storage_table(records, MAIN)
        cont = continuity(records, v3)
        samples = jsonl(run / "ssd_samples.jsonl"); seg = segments(records, samples)
        title = "본실험 결과" if main_done else f"본실험 중간 결과 (commit된 {len(records):,}장)"
        L += [f"## {title}", "", "### 1. 본실험 표", ""] + main_tables_md(arms, sp, tp, ratios, cross)
        L += ["", "### 4. ReComp TTFT 확인 (파일럿 40장)", ""] + recomp_check_md((pj or {}).get("recomp_ttft_check"))
        L += ["", "### 5. IMPRESS 통계 (본실험 followup)", ""] + impress_md(imp)
        L += ["", "### 6. 재사용 턴 시간 분해 (followup)", ""] + decomposition_md(decomp)
        L += ["", "### 7. T1 저장 비용", ""] + storage_md(storage)
        ssd_lines, ssd_json = ssd_md(run, out_root, storage, main_done)
        ex_lines, ex_json = execution_md(run, records, n_main)
        L += ["", "### 8. SSD 쓰기량, SMART, 소요 시간", ""] + ssd_lines + ["", "소요 시간과 이어 실행:", ""] + ex_lines
        L += ["", "### 9. 시간에 따른 TTFT 변화 (본실험 10구간)", ""] + segments_md(seg)
        L += ["", "### 10. 연속성", ""] + continuity_md(cont, g["pilot"])
        summary.update(arms=arms, paired_score=sp, paired_ttft=tp, ttft_ratios=ratios, ttft_ratios_cross_run=cross, impress=imp, decomposition=decomp,
                       storage=storage, continuity=cont, segments=seg, ssd=ssd_json, execution=ex_json)
        write_csv(out_root / "main_table.csv", arms); write_csv(out_root / "paired_score.csv", sp); write_csv(out_root / "paired_ttft.csv", tp)
        write_csv(out_root / "ttft_ratios.csv", ratios + cross); write_csv(out_root / "segments.csv", seg); write_csv(out_root / "storage.csv", storage)
        write_csv(out_root / "impress_layers.csv", imp["layers"])
        (out_root / "continuity.json").write_text(json.dumps(cont, indent=1, ensure_ascii=False) + "\n")
    else:
        L += ["## 본실험 결과", "", "본실험 표(1–3), IMPRESS 통계(5), 시간 분해(6), 저장 비용(7), 10구간 TTFT(9): **미실행**.", "",
              "### 4. ReComp TTFT 확인 (파일럿 40장)", ""] + recomp_check_md((pj or {}).get("recomp_ttft_check"))
        ssd_lines, ssd_json = ssd_md(run, out_root, [], False)
        ex_lines, ex_json = execution_md(run, [], n_main)
        L += ["", "### 8. SSD 쓰기량, SMART, 소요 시간", ""] + ssd_lines + ["", "소요 시간과 이어 실행:", ""] + ex_lines
        L += ["", "### 10. 연속성", ""] + continuity_md(None, g["pilot"])
        summary.update(ssd=ssd_json, execution=ex_json)
    L += ["", "### 11. 코드, 지시와 다른 점, 한계", "", "추가하거나 바꾼 코드:", ""] + CODE_MD + [""] + deviations_md(config, pre)
    if check_protection:
        prot = protection_check(run, out_root)
        if prot:
            L += ["", f"기존 `results/`·`runs/`·`kvstore*` 파일 목록(경로·크기·수정 시각)의 실행 전후 비교: **{prot['status']}**"
                      + (" (이 실행 밖의 파일이 바뀌거나 새로 생겼다 — 다른 작업이 같은 저장소에서 돌았는지 확인 필요; 이 실행은 자기 디렉터리 밖에 쓰지 않는다)" if prot["status"] == "CHANGED" else "") + "."]
    L += notes_md(out_root)
    (out_root / "report.md").write_text("\n".join(L) + "\n")
    (out_root / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float) + "\n")
    print(json.dumps({"report": str((out_root / "report.md").relative_to(ROOT)), "main_complete": main_done, "admitted": admitted, "images": len(records)}))


def report_pilot(run, manifest, out_root):
    pl, pj = pilot_md(run, manifest, load_v3(run, "main"))
    if pl is None:
        raise SystemExit("the pilot phase is not complete")
    L = [f"# 파일럿 40장 표 — run `{run.name}`", ""] + pl + ["", "## ReComp TTFT 확인", ""] + recomp_check_md(pj["recomp_ttft_check"])
    (out_root / "pilot_table.md").write_text("\n".join(L) + "\n")
    (out_root / "pilot_table.json").write_text(json.dumps(pj, indent=1, ensure_ascii=False, default=float) + "\n")
    print(json.dumps({"pilot_table": str((out_root / "pilot_table.md").relative_to(ROOT))}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--phase", choices=("pilot", "auto"), default="auto")
    p.add_argument("--no-protection-check", action="store_true")
    args = p.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    config = json.loads((run / "config.json").read_text())
    out_root = ROOT / "results" / run.name
    out_root.mkdir(parents=True, exist_ok=True)
    if args.phase == "pilot":
        return report_pilot(run, manifest, out_root)
    report_auto(run, manifest, config, out_root, check_protection=not args.no_protection_check)


if __name__ == "__main__":
    main()

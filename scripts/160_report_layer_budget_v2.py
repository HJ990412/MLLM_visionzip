#!/usr/bin/env python3
"""v2 report: independent HOLD audit, paired analysis, mechanical decision
(v1 A10 rule on A1-2_packed vs U_packed), secondary comparisons (packed vs
unpacked, prior-work vectors), protection after, REPORT.md / REPRODUCE.md.

Usage: --run RUN [--skip-hold]
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from mmimpress.prefix25 import sha
from mmimpress.recompute_kv25 import paired_bootstrap, sha_manifest, stat_inventory

PK = "ours_prefixstore_kv25_"
SEED = 1234
V1_RES = ROOT / "results/layer_budget_kv25_20261105T125754Z"
NEW = ["docs/layer_budget_kv25_v2_contract.md", "mmimpress/packed_store.py", "mmimpress/packed_serve.py", "tests/test_packed_store.py",
       "scripts/158_allocate_prior_work_vectors.py", "scripts/159_eval_layer_budget_v2_dev.py", "scripts/160_report_layer_budget_v2.py",
       "scripts/161_validate_packed_store_v2.py", "scripts/162_eval_layer_budget_v2_hold.py", "scripts/packed_support_v2.py", "papers/layer_budget_kv25_v2_packed_prompt.md"]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def jl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _json_default(o):
    import numpy as np
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(f"not serializable: {type(o).__name__}")


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=1, allow_nan=False, default=_json_default); f.write("\n"); f.flush(); os.fsync(f.fileno())


def csv_write(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def name(method):
    if method == "recompute":
        return "ReComp"
    if method == "fullload":
        return "FullLoad"
    if method.startswith(PK):
        return method[len(PK):]
    return method


def vec_of(method):
    return method[len(PK):-len("_packed")]


def pct(x):
    return f"{100 * x:.3f}"


def stats(v):
    a = np.asarray(v, dtype=float)
    return float(a.mean()), float(np.median(a)), float(np.percentile(a, 95))


def md_table(head, rows):
    return "| " + " | ".join(head) + " |\n|" + "|".join("---" for _ in head) + "|\n" + "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in rows) + "\n"


# ------------------------------------------------------------------- HOLD
def hold_rows(hold):
    rows = []; attempts = {}
    for cp in sorted((hold / "main" / "images").glob("*/COMMITTED.json")):
        doc = json.loads(cp.read_text()); folder = cp.parent / doc["attempt_id"]
        if sha(folder / "raw.jsonl") != doc["raw_sha256"]:
            raise RuntimeError("committed raw hash mismatch: " + str(folder))
        rows += jl(folder / "raw.jsonl"); attempts[cp.parent.name] = folder
    return rows, attempts


def hold_audit(hold, config, manifest, attempts, H):
    out = {"images": 0, "requests": 0, "failures": [], "per_image_status": {}}
    for image_id, folder in attempts.items():
        rows = jl(folder / "raw.jsonl")
        receipt = H.audit_image_rows(rows, manifest, config, "main", image_id, folder)
        out["images"] += 1; out["requests"] += len(rows)
        out["per_image_status"][image_id] = receipt["status"]
        if receipt["status"] != "PASS":
            out["failures"].append({"image_id": image_id, "failures": receipt["failures"][:5]})
    out["expected_requests"] = manifest["expected_requests"]["main"]
    out["expected_images"] = 398
    out["status"] = "PASS" if not out["failures"] and out["requests"] == out["expected_requests"] and out["images"] == 398 else "FAIL"
    return out


def hold_analysis(hold, rows, config, res, H):
    methods = config["methods"]
    by = defaultdict(dict)
    for r in rows:
        by[r["method_id"]][(r["dialog_id"], r["turn_id"])] = r
    images = sorted({r["image_id"] for r in rows})
    img_of = {r["dialog_id"]: r["image_id"] for r in rows}
    quality = []
    for m in methods:
        d = by[m]
        turns = {t: [v["score"] for k, v in d.items() if k[1] == t] for t in (1, 2, 3)}
        follow = turns[2] + turns[3]; allv = turns[1] + follow
        quality.append({"method": m, "name": name(m), "T1_pct": 100 * np.mean(turns[1]), "T2_pct": 100 * np.mean(turns[2]), "T3_pct": 100 * np.mean(turns[3]),
                        "all_pct": 100 * np.mean(allv), "all_correct": int(sum(allv)), "all_n": len(allv),
                        "followup_pct": 100 * np.mean(follow), "followup_correct": int(sum(follow)), "followup_n": len(follow)})

    def paired(a, b, turn_filter, key="score"):
        per = []
        for img in images:
            ka = [v[key] for k, v in by[a].items() if img_of[k[0]] == img and turn_filter(k[1])]
            kb = [v[key] for k, v in by[b].items() if img_of[k[0]] == img and turn_filter(k[1])]
            assert len(ka) == len(kb)
            per.append((sum(ka), sum(kb), len(ka)))
        return paired_bootstrap(per, 10000, SEED)

    packed = [m for m in methods if m.startswith(PK)]
    u = f"{PK}U_packed"; sel = f"{PK}A1-2_packed"
    comparisons = [(sel, u)] + [(sel, m) for m in packed if m not in (sel, u)] + [(m, u) for m in packed if m not in (sel, u)]
    comparisons += [(m, b) for m in packed for b in ("recompute", "fullload")] + [("fullload", "recompute")]
    pairs = []
    for a, b in comparisons:
        for scope, flt in (("followup", lambda t: t > 1), ("all", lambda t: True)):
            d = paired(a, b, flt)
            agree = np.mean([by[a][k]["prediction"] == by[b][k]["prediction"] for k in by[a] if flt(k[1])])
            cells = {"both": 0, "a_only": 0, "b_only": 0, "neither": 0}
            for k in by[a]:
                if not flt(k[1]):
                    continue
                x, y = by[a][k]["score"], by[b][k]["score"]
                cells["both" if x and y else "a_only" if x else "b_only" if y else "neither"] += 1
            pairs.append({"a": a, "b": b, "scope": scope, "delta_pp": 100 * d["mean"], "ci95_lo_pp": 100 * d["ci_low"], "ci95_hi_pp": 100 * d["ci_high"],
                          "prediction_agreement": agree, **cells, "resamples": d["resamples"], "seed": d["seed"]})
        for metric in ("ttft_ms", "request_e2e_ms", "ssd_read_bytes"):
            d = paired(a, b, lambda t: t > 1, key=metric)
            pairs.append({"a": a, "b": b, "scope": "followup", "metric": metric, "delta": d["mean"], "ci95_lo": d["ci_low"], "ci95_hi": d["ci_high"]})
    latency = []
    for m in methods:
        f = [v for k, v in by[m].items() if k[1] > 1]
        t = stats([v["ttft_ms"] for v in f]); e = stats([v["request_e2e_ms"] for v in f])
        b = np.array([v["ssd_read_bytes"] for v in f]); p = np.array([v["ssd_preads"] for v in f])
        entry = {"method": m, "name": name(m), "ttft_mean_ms": t[0], "ttft_p50_ms": t[1], "ttft_p95_ms": t[2],
                 "e2e_mean_ms": e[0], "e2e_p50_ms": e[1], "e2e_p95_ms": e[2], "ssd_mb_per_followup": b.mean() / 1e6,
                 "read_calls_mean": p.mean(), "pread_calls_mean": float(np.mean([v["pread_calls"] for v in f])), "preadv_calls_mean": float(np.mean([v["preadv_calls"] for v in f])),
                 "followups": len(f)}
        for key in ("chunk_io_ms", "scatter_ms", "packed_read_ms", "prefill_ms"):
            vals = [v["result"].get(key) for v in f if v["result"].get(key) is not None]
            entry[f"{key}_mean"] = float(np.mean(vals)) if vals else None
        latency.append(entry)
    storage = []; setup = {}
    for m in methods[1:]:
        kind = H.kind_of(m)
        rs = []
        for img in images:
            folder = next(p for p in (hold / "main" / "images" / img).glob("attempt_*") if (p / "independent_audit.json").exists() and json.loads((p / "independent_audit.json").read_text())["status"] == "PASS")
            per = json.loads((folder / f"{kind}_persistence.json").read_text()); act = json.loads((folder / f"{m}_activation.json").read_text())
            meta = json.loads((folder / f"{kind}_meta.json").read_text())
            n = meta["v_token_num"] - len(meta["newline_idx"])
            stored = meta.get("stored_content_count_total", n * 32)
            rs.append({"bytes": per["file_logical_bytes"], "allocated": per["allocated_bytes"], "os_write": per["os_write_bytes"],
                       "persist_ms": per["service_persistence_ms"], "writer_ms": per["native_writer_wall_ms"], "act_ms": act["outer_wall_ms"], "N": n, "content_rows_total": stored,
                       "sha_ms": per.get("timing_ms", {}).get("packed_sha256_ms", 0.0)})
            setup[(m, img)] = per["service_persistence_ms"] + act["outer_wall_ms"]
        b = stats([x["bytes"] for x in rs]); pm = stats([x["persist_ms"] for x in rs]); am = stats([x["act_ms"] for x in rs])
        storage.append({"method": m, "name": name(m), "images": len(rs), "stored_content_pct": 100 * np.mean([x["content_rows_total"] / (32 * x["N"]) for x in rs]),
                        "store_mb_mean": b[0] / 1e6, "store_mb_p50": b[1] / 1e6, "store_mb_p95": b[2] / 1e6,
                        "allocated_mb_mean": np.mean([x["allocated"] for x in rs]) / 1e6, "os_write_mb_mean": np.mean([x["os_write"] for x in rs]) / 1e6,
                        "persistence_ms_mean": pm[0], "persistence_ms_p50": pm[1], "persistence_ms_p95": pm[2], "native_writer_ms_mean": np.mean([x["writer_ms"] for x in rs]),
                        "packed_sha256_ms_mean": np.mean([x["sha_ms"] for x in rs]),
                        "activation_ms_mean": am[0], "activation_ms_p50": am[1], "activation_ms_p95": am[2], "provenance": "MEASURED"})
    dialogs = sorted({r["dialog_id"] for r in rows})
    first_dialog = {}
    for r in sorted(rows, key=lambda x: x["global_dialog_ordinal"]):
        first_dialog.setdefault(r["image_id"], r["dialog_id"])
    sessions = []; sess_vals = {}
    for m in methods:
        shared, standalone = [], []
        for d in dialogs:
            e2e = sum(by[m][(d, t)]["request_e2e_ms"] for t in (1, 2, 3))
            img = img_of[d]
            su = setup.get((m, img), 0.0) if m != "recompute" else 0.0
            shared.append(e2e + (su if first_dialog.get(img) == d else 0.0)); standalone.append(e2e + su)
        sess_vals[m] = (shared, standalone)
        s1, s2 = stats(shared), stats(standalone)
        sessions.append({"method": m, "name": name(m), "shared_mean_ms": s1[0], "shared_p50_ms": s1[1], "shared_p95_ms": s1[2],
                         "standalone_mean_ms": s2[0], "standalone_p50_ms": s2[1], "standalone_p95_ms": s2[2],
                         "provenance": "MEASURED_NO_KV_SETUP" if m == "recompute" else "MEASURED (shared) / DERIVED (standalone)"})
    for m in methods[1:]:
        for kind_i, kind in ((0, "shared"), (1, "standalone")):
            per = []
            for img in images:
                idx = [i for i, d in enumerate(dialogs) if img_of[d] == img]
                per.append((sum(sess_vals[m][kind_i][i] for i in idx), sum(sess_vals["recompute"][kind_i][i] for i in idx), len(idx)))
            d = paired_bootstrap(per, 10000, SEED)
            pairs.append({"a": m, "b": "recompute", "scope": f"session_{kind}", "metric": "session_ms", "delta": d["mean"], "ci95_lo": d["ci_low"], "ci95_hi": d["ci_high"]})
    lat = {l["method"]: l for l in latency}
    spread = {name(m): lat[m]["ttft_mean_ms"] / lat[u]["ttft_mean_ms"] - 1 for m in packed if m != u}
    csv_write(res / "hold" / "quality.csv", quality); csv_write(res / "hold" / "latency_io.csv", latency)
    csv_write(res / "hold" / "storage.csv", storage); csv_write(res / "hold" / "paired_analysis.csv", pairs); csv_write(res / "hold" / "sessions.csv", sessions)
    write_json(res / "hold" / "paired_analysis.json", pairs)
    return {"quality": quality, "latency": latency, "storage": storage, "pairs": pairs, "sessions": sessions, "images": len(images), "dialogs": len(dialogs),
            "packed_ttft_spread_vs_U_packed": spread, "max_abs_spread": max(abs(v) for v in spread.values())}


def decide(h, config, v1_decision, v1_latency):
    u = f"{PK}U_packed"; sel = f"{PK}A1-2_packed"
    q = {x["method"]: x for x in h["quality"]}; lat = {x["method"]: x for x in h["latency"]}
    d = next(p for p in h["pairs"] if p["a"] == sel and p["b"] == u and p["scope"] == "followup" and "metric" not in p)
    delta, lo, hi = d["delta_pp"], d["ci95_lo_pp"], d["ci95_hi_pp"]
    delta, lo, hi = float(delta), float(lo), float(hi)
    ci_excludes_zero = bool(lo > 0 or hi < 0)
    mb_ok = bool(lat[sel]["ssd_mb_per_followup"] <= lat[u]["ssd_mb_per_followup"] * 1.02)
    ttft_ok = bool(lat[sel]["ttft_mean_ms"] <= lat[u]["ttft_mean_ms"] * 1.05)
    if delta >= 1.0 and ci_excludes_zero and mb_ok and ttft_ok:
        verdict = "GO"
    elif (delta >= 0.5 and not ci_excludes_zero) or (delta >= 1.0 and ci_excludes_zero and (mb_ok != ttft_ok)):
        verdict = "PARTIAL"
    else:
        verdict = "NO-GO"
    v1_u_ttft = next(float(r["ttft_mean_ms"]) for r in v1_latency if r["method"] == "ours_prefixstore_kv25")
    v1_a_ttft = next(float(r["ttft_mean_ms"]) for r in v1_latency if r["method"] == "ours_prefixstore_kv25_layerwise_A1-2")
    out = {"verdict": verdict, "selected": sel, "control": u, "delta_followup_pp": delta, "ci95_pp": [lo, hi], "ci_excludes_zero": ci_excludes_zero,
           "ssd_mb_selected": lat[sel]["ssd_mb_per_followup"], "ssd_mb_uniform": lat[u]["ssd_mb_per_followup"], "ssd_ok_1.02": mb_ok,
           "ttft_mean_selected": lat[sel]["ttft_mean_ms"], "ttft_mean_uniform": lat[u]["ttft_mean_ms"], "ttft_ratio": lat[sel]["ttft_mean_ms"] / lat[u]["ttft_mean_ms"], "ttft_ok_1.05": ttft_ok,
           "followup_pct": {m: q[m]["followup_pct"] for m in config["methods"]},
           "rule": "GO: delta>=+1.0pp & CI excludes 0 & MB<=1.02x & TTFT<=1.05x; PARTIAL: delta>=+0.5pp with CI incl. 0, or GO with exactly one of MB/TTFT violated; else NO-GO",
           "secondary": {"v1_verdict": v1_decision["verdict"], "v1_delta_followup_pp": v1_decision["delta_followup_pp"], "v1_ci95_pp": v1_decision["ci95_pp"], "v1_ttft_ratio": v1_decision["ttft_mean_selected"] / v1_decision["ttft_mean_uniform"],
                         "U_packed_vs_v1_U_unpacked_ttft_ratio_cross_run": lat[u]["ttft_mean_ms"] / v1_u_ttft, "v1_U_unpacked_ttft_mean_ms": v1_u_ttft,
                         "A1-2_packed_vs_v1_A1-2_unpacked_ttft_ratio_cross_run": lat[sel]["ttft_mean_ms"] / v1_a_ttft, "v1_A1-2_unpacked_ttft_mean_ms": v1_a_ttft,
                         "packed_ttft_spread_vs_U_packed": h["packed_ttft_spread_vs_U_packed"], "spread_within_2pct": bool(h["max_abs_spread"] <= 0.02),
                         "prior_work_vs_A1-2": {name(p["b"]): {"delta_pp_A12_minus": p["delta_pp"], "ci95": [p["ci95_lo_pp"], p["ci95_hi_pp"]]}
                                               for p in h["pairs"] if p["a"] == sel and p["b"] not in (u, "recompute", "fullload") and p["scope"] == "followup" and "metric" not in p}}}
    return out


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--skip-hold", action="store_true")
    args = ap.parse_args()
    run = Path(args.run).resolve(); res = ROOT / "results" / run.name
    res.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / "config.json").read_text())
    shutil.copy2(run / "CONTRACT.md", res / "CONTRACT.md")
    for p in sorted((run / "phase2").glob("*")):
        if p.is_file():
            shutil.copy2(p, res / ("phase2_" + p.name if not p.name.startswith("layer_budget") else p.name))
    for f in ("data_split.json", "io_microbench.csv", "source_modifications.json"):
        if (run / f).exists():
            shutil.copy2(run / f, res / f)
    for f in ("storage_plan.json", "prior_work_sources.json", "v1_reuse.json"):
        shutil.copy2(run / "phase0" / f, res / ("phase0_" + f))
    for p in sorted((run / "gates").glob("*.json")):
        shutil.copy2(p, res / ("gate_" + p.name))
    gates = {p.stem: json.loads(p.read_text())["status"] for p in (run / "gates").glob("*.json") if "attempt" not in p.stem}
    failed_attempts = sorted(p.stem for p in (run / "gates").glob("*attempt*_failed.json"))
    p2 = json.loads((run / "phase2" / "summary.json").read_text())
    vectors = list(csv.DictReader((run / "phase2" / "vectors.csv").open()))
    dev = json.loads((res / "dev" / "summary.json").read_text())
    sel = json.loads((run / "dev_selection.json").read_text())
    g9 = json.loads((run / "gates" / "g9_io_microbench.json").read_text())
    v1_decision = json.loads((V1_RES / "decision.json").read_text())
    v1_latency = list(csv.DictReader((V1_RES / "hold" / "latency_io.csv").open()))
    v1_quality = list(csv.DictReader((V1_RES / "hold" / "quality.csv").open()))
    hold = run / "hold"
    H = load("layer_budget_v2_hold", "162_eval_layer_budget_v2_hold.py")
    Hres = None; audit = None; decision = None
    if not args.skip_hold and (hold / "main").exists():
        hconfig = json.loads((hold / "config.json").read_text()); hmanifest = json.loads((hold / "manifest.json").read_text())
        hconfig.update(_file_sha256=sha(hold / "config.json"), _manifest_sha256=sha(hold / "manifest.json"))
        rows, attempts = hold_rows(hold)
        audit = hold_audit(hold, hconfig, hmanifest, attempts, H)
        write_json(res / "independent_audit.json", audit)
        Hres = hold_analysis(hold, rows, hconfig, res, H)
        decision = decide(Hres, hconfig, v1_decision, v1_latency)
        decision["audit_status"] = audit["status"]; decision["final"] = audit["status"] == "PASS"
        write_json(res / "decision.json", decision)
    # io microbench figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        io = list(csv.DictReader((run / "io_microbench.csv").open()))
        sizes = []
        for r in io:
            if r["size_label"] not in sizes:
                sizes.append(r["size_label"])
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for mode, c in (("buffered_cold", "#0077bb"), ("direct", "#cc3311")):
            ys = [np.mean([float(r["dd_gbps"]) for r in io if r["size_label"] == s and r["mode"] == mode]) for s in sizes]
            es = [np.std([float(r["dd_gbps"]) for r in io if r["size_label"] == s and r["mode"] == mode], ddof=1) for s in sizes]
            ax.errorbar(range(len(sizes)), ys, yerr=es, marker="o", color=c, label=mode, capsize=3)
        ax.set_xticks(range(len(sizes))); ax.set_xticklabels(sizes); ax.set_ylabel("GB/s"); ax.set_xlabel("file size")
        ax.set_title(f"Cold sequential read throughput by file size (dd, 5 files each; readahead {g9['read_ahead_kb']} KB)"); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(res / "fig_io_microbench.png", dpi=130); plt.close(fig)
    except Exception as exc:
        write_json(res / "fig_io_microbench_error.json", {"error": repr(exc)})
    # protection after
    before = json.loads((run / "protection_before.json").read_text())
    excl = [f"results/{run.name}"] + NEW
    shas = {k: v for k, v in sha_manifest([ROOT / x for x in ["results", "docs", "mmimpress", "scripts", "tests", "README.md"]], ROOT, exclude=excl).items() if "__pycache__" not in k}
    kv = stat_inventory(sorted(ROOT.glob("kvstore*")), ROOT)
    diff = {"sha_changed": sorted(k for k in before["sha256"] if k in shas and shas[k] != before["sha256"][k]),
            "sha_missing": sorted(set(before["sha256"]) - set(shas)), "sha_added": sorted(set(shas) - set(before["sha256"])),
            "kv_changed": sorted(k for k in before["kvstore_inventory"] if k in kv and kv[k] != before["kvstore_inventory"][k]),
            "kv_missing": sorted(set(before["kvstore_inventory"]) - set(kv)), "kv_added": sorted(set(kv) - set(before["kvstore_inventory"]))}
    amendments = {}
    for p in sorted(run.glob("freeze_amendment_*.json")):
        a = json.loads(p.read_text())
        if a.get("sha256_now"):                      # scope-only amendments carry no file hash
            amendments[a["file"]] = a["sha256_now"]
    unexplained = [k for k in diff["sha_changed"] if shas[k]["sha256"] != amendments.get(k)]
    explained = [k for k in diff["sha_changed"] if k not in unexplained]
    hard = {**diff, "sha_changed": unexplained}
    protection = {"status": "PASS" if not any(hard.values()) else "FAIL", "files_before": len(before["sha256"]), "files_after": len(shas), "differences": diff,
                  "changes_explained_by_recorded_amendments": explained, "amendment_files": [p.name for p in sorted(run.glob("freeze_amendment_*.json"))],
                  "source_modifications": json.loads((run / "source_modifications.json").read_text())}
    write_json(res / "protection_result.json", protection)
    write_json(run / "protection_after.json", {"stage": "after", "utc": time.time(), "sha256": shas, "kvstore_inventory": kv})
    # ---------------------------------------------------------------- REPORT
    L = []
    L.append("# Layer-wise budget v2 — Packed PrefixStore-KV25 + prior-work allocation baselines (LLaVA-NeXT) — results\n")
    L.append(f"Run `{run}`. Contract [CONTRACT.md](CONTRACT.md) (SHA `{config['contract_sha256'][:16]}…`, frozen before measurement; v1 contract SHA `{config['v1_contract_sha256'][:16]}…`). "
             "Model llava-v1.6-vicuna-7b NF4/BF16/eager, FP16 SSD KV, greedy max 16, seed 1234. v1 run: `results/layer_budget_kv25_20261105T125754Z` (PARTIAL).\n")
    if decision is not None:
        s = decision["secondary"]
        L.append(f"## Decision: **{decision['verdict']}**\n")
        L.append(f"HOLD followup accuracy, A1-2_packed − U_packed: **{decision['delta_followup_pp']:+.3f} pp** [{decision['ci95_pp'][0]:+.3f}, {decision['ci95_pp'][1]:+.3f}] "
                 f"(image-cluster paired bootstrap, 10,000 resamples, seed 1234; CI {'excludes' if decision['ci_excludes_zero'] else 'includes'} 0). "
                 f"SSD MB/followup {decision['ssd_mb_selected']:.3f} vs uniform {decision['ssd_mb_uniform']:.3f} ({'OK' if decision['ssd_ok_1.02'] else 'VIOLATED'} ≤1.02×); "
                 f"TTFT mean {decision['ttft_mean_selected']:.2f} vs {decision['ttft_mean_uniform']:.2f} ms = {decision['ttft_ratio']:.3f}× ({'OK' if decision['ttft_ok_1.05'] else 'VIOLATED'} ≤1.05×). "
                 f"Independent audit: {decision['audit_status']}; final = {decision['final']}.\n")
        L.append("### v1 → v2\n")
        L.append(md_table(["", "v1 (unpacked, 65 preads)", "v2 (packed, 1 preadv)"],
                          [["Verdict", s["v1_verdict"], decision["verdict"]],
                           ["A1-2 − U followup Δ pp [CI]", f"{s['v1_delta_followup_pp']:+.3f} [{s['v1_ci95_pp'][0]:+.3f}, {s['v1_ci95_pp'][1]:+.3f}]", f"{decision['delta_followup_pp']:+.3f} [{decision['ci95_pp'][0]:+.3f}, {decision['ci95_pp'][1]:+.3f}]"],
                           ["TTFT A1-2 / U", f"{s['v1_ttft_ratio']:.3f}×", f"{decision['ttft_ratio']:.3f}×"],
                           ["U TTFT mean ms", f"{s['v1_U_unpacked_ttft_mean_ms']:.2f}", f"{decision['ttft_mean_uniform']:.2f} ({s['U_packed_vs_v1_U_unpacked_ttft_ratio_cross_run']:.3f}× v1, cross-run)"],
                           ["A1-2 TTFT mean ms", f"{s['v1_A1-2_unpacked_ttft_mean_ms']:.2f}", f"{decision['ttft_mean_selected']:.2f} ({s['A1-2_packed_vs_v1_A1-2_unpacked_ttft_ratio_cross_run']:.3f}× v1, cross-run)"],
                           ["Packed TTFT spread across vectors vs U_packed", "—", ", ".join(f"{k} {v:+.1%}" for k, v in s["packed_ttft_spread_vs_U_packed"].items()) + (" (within ±2 %)" if s["spread_within_2pct"] else " (outside ±2 %)")]]))
    L.append("## Gates\n")
    L.append(md_table(["Gate", "Result"], [[k, v] for k, v in sorted(gates.items())] +
                      [["G3 DEV byte invariance (max |ratio−1|)", f"{dev['g3']['max_abs_ratio_deviation']:.6f} (tol 0.02) {dev['g3']['status']}"],
                       ["G6 DEV packed == unpacked outputs (U, A1-2, directcopy)", f"{dev['g6']['checks']} checks, all identical = {dev['g6']['all_identical']} ({dev['g6']['status']})"],
                       ["Artifact protection before/after", protection["status"]]]))
    if failed_attempts:
        L.append("Failed gate attempts preserved: " + ", ".join(failed_attempts) + ".\n")
    L.append("## G9 — cold sequential read throughput by file size (GNU dd; fio not installed)\n")
    L.append(md_table(["size", "buffered cold GB/s (mean, min–max)", "O_DIRECT GB/s (mean, min–max)"],
                      [[sz, f"{g9['summary'][sz + '/buffered_cold']['gbps_mean']:.2f} ({g9['summary'][sz + '/buffered_cold']['gbps_min']:.2f}–{g9['summary'][sz + '/buffered_cold']['gbps_max']:.2f})",
                        f"{g9['summary'][sz + '/direct']['gbps_mean']:.2f} ({g9['summary'][sz + '/direct']['gbps_min']:.2f}–{g9['summary'][sz + '/direct']['gbps_max']:.2f})"] for sz in g9["sizes"]]))
    L.append("![io](fig_io_microbench.png)\n")
    L.append("## Allocation vectors (Σ r_l = 8.0, clip [0.10, 0.60])\n")
    ids = p2["alloc_ids"]
    L.append(md_table(["layer"] + ids + ["γ_l (VL-Cache)", "cos_l (SA)"], [[r["layer"]] + [f"{float(r[a]):.3f}" for a in ids] + [f"{float(r['gamma_all_mean']):.3f}", f"{float(r['cos_mean']):.4f}"] for r in vectors]))
    L.append("Ratio SHA256: " + ", ".join(f"{a} `{p2['sha256'][a][:12]}…`" for a in ids) + f". Clipped layers: {json.dumps(p2['clipped_layers'])}. ![vectors](phase2_fig_vectors.png)\n")
    L.append("VLC_offline and SA_offline freeze the CAL mean of quantities that VL-Cache and SqueezeAttention compute online per request; they are allocation criteria moved offline, **not reproductions of the original systems**. "
             "PYR is the fixed-shape PyramidKV schedule (v1 F-front ratios); PYR-beta is the PyramidKV code schedule with β = 20. "
             f"SA-2g uses the SqueezeAttention 3-cluster rule with class-3 budget = {p2['diag']['sa_percent_used']:.3f} (ASSUMPTION from the README example ratio; clipped to r_min). "
             "All comparisons share bytes, the packed I/O path and the VisionZip token order and differ only in the per-layer cut point.\n")
    L.append("## Phase 3 — DEV (40 images × 5 cache hits per arm = 200 hits; T1 shared; every arm from its own real store)\n")
    L.append(md_table(["Arm", "Store", "Hit acc. %", "Δ vs A1-2_packed (pp) [95% CI]", "Δ vs U_packed (pp) [95% CI]", "Δ vs FullLoad (pp) [95% CI]"],
                      [[q["arm"], q["store_kind"], pct(q["hit_accuracy"]),
                        (f"{q['delta_pp_vs_A1-2_packed']:+.2f} [{q['ci95_lo_vs_A1-2_packed']:+.2f}, {q['ci95_hi_vs_A1-2_packed']:+.2f}]" if "delta_pp_vs_A1-2_packed" in q else "—"),
                        (f"{q['delta_pp_vs_U_packed']:+.2f} [{q['ci95_lo_vs_U_packed']:+.2f}, {q['ci95_hi_vs_U_packed']:+.2f}]" if "delta_pp_vs_U_packed" in q else "—"),
                        (f"{q['delta_pp_vs_fullload']:+.2f} [{q['ci95_lo_vs_fullload']:+.2f}, {q['ci95_hi_vs_fullload']:+.2f}]" if "delta_pp_vs_fullload" in q else "—")] for q in dev["quality"]]))
    L.append(md_table(["Arm", "Repeats", "TTFT mean ± SD(repeat means) / p50 / p95 ms", "read ms", "scatter ms", "SSD MB/hit", "pread / preadv calls"],
                      [[l["arm"], l["repeats"], f"{l['ttft_mean_ms']:.1f} ± {l['ttft_mean_sd_across_repeats_ms']:.1f} / {l['ttft_p50_ms']:.1f} / {l['ttft_p95_ms']:.1f}",
                        f"{l['chunk_io_ms_mean']:.1f}" if l.get("chunk_io_ms_mean") is not None else "—", f"{l['scatter_ms_mean']:.1f}" if l.get("scatter_ms_mean") is not None else "—",
                        f"{l['ssd_mb_per_hit']:.2f}", f"{l['pread_calls_mean']:.0f} / {l['preadv_calls_mean']:.0f}"] for l in dev["latency_io"]]))
    r = dev["packed_vs_unpacked_ttft_ratio"]
    L.append(f"Packed / unpacked TTFT: U {r['U']:.3f}×, A1-2 {r['A1-2']:.3f}×; A1-2 / U: unpacked {r['A1-2_vs_U_unpacked']:.3f}×, packed {r['A1-2_vs_U_packed']:.3f}×; directcopy / scatter (A1-2 packed) {r['directcopy_vs_scatter_A1-2']:.3f}×. "
             "Packed TTFT spread vs U_packed: " + ", ".join(f"{k} {v:+.1%}" for k, v in dev["packed_ttft_spread_vs_U_packed"].items()) + ".\n")
    L.append(md_table(["Store", "Store MB/image", "Persistence ms mean / p95", "Native writer SSD write ms", "packed SHA256 ms", "Activation ms", "OS write MB"],
                      [[s_["store_key"], f"{s_['store_mb_mean']:.2f}", f"{s_['persistence_ms_mean']:.1f} / {s_['persistence_ms_p95']:.1f}",
                        f"{s_['native_writer_ssd_write_ms_mean']:.1f}" if s_.get("native_writer_ssd_write_ms_mean") is not None else "—", f"{s_['packed_sha256_ms_mean']:.1f}",
                        f"{s_['activation_ms_mean']:.1f}", f"{s_['os_write_mb_mean']:.2f}"] for s_ in dev["storage"]]))
    err = list(csv.DictReader((res / "dev" / "error_analysis.csv").open()))
    types, arms = [], []
    for row in err:
        if row["question_type"] not in types:
            types.append(row["question_type"])
        if row["arm"] not in arms:
            arms.append(row["arm"])
    L.append("Question-type Δ vs FullLoad on hits (pp), spatial highlighted:\n")
    L.append(md_table(["type", "n"] + arms, [[("**spatial**" if t == "spatial" else t), next(x["n_questions"] for x in err if x["question_type"] == t)] +
                                             [f"{float(next(x['delta_vs_fullload_pp'] for x in err if x['arm'] == a and x['question_type'] == t)):+.1f}" for a in arms] for t in types]))
    L.append(f"SELECTED = {sel['selected']} (fixed before measurement); best DEV vector = {sel['best_dev_vector']} (comparison only). HOLD arms: {sel['hold_arms']}.\n")
    if Hres is not None:
        L.append("## Phase 4 — HOLD (MT-GQA-reconstructed, 398 images, 4,061 dialogues × 3 turns, generated history, 7 arms)\n")
        L.append(f"Independent audit {audit['status']}: {audit['requests']} / {audit['expected_requests']} requests, {audit['images']} / 398 images.\n")
        L.append(md_table(["Method", "T1 %", "T2 %", "T3 %", "All % (correct/N)", "Followup % (correct/N)"],
                          [[q["name"], f"{q['T1_pct']:.3f}", f"{q['T2_pct']:.3f}", f"{q['T3_pct']:.3f}", f"{q['all_pct']:.3f} ({q['all_correct']}/{q['all_n']})",
                            f"{q['followup_pct']:.3f} ({q['followup_correct']}/{q['followup_n']})"] for q in Hres["quality"]]))
        L.append("### Prior-work comparison (HOLD followup, same bytes, same packed path)\n")
        lat = {l["method"]: l for l in Hres["latency"]}; qq = {q["method"]: q for q in Hres["quality"]}
        u = f"{PK}U_packed"; selm = f"{PK}A1-2_packed"
        rows_pw = []
        for m in [x for x in hconfig["methods"] if x.startswith(PK)]:
            dA = next((p for p in Hres["pairs"] if p["a"] == selm and p["b"] == m and p["scope"] == "followup" and "metric" not in p), None)
            dU = next((p for p in Hres["pairs"] if p["a"] == m and p["b"] == u and p["scope"] == "followup" and "metric" not in p), None)
            rows_pw.append([name(m), f"{qq[m]['followup_pct']:.3f}",
                            (f"{dU['delta_pp']:+.3f} [{dU['ci95_lo_pp']:+.3f}, {dU['ci95_hi_pp']:+.3f}]" if dU else ("0 (control)" if m == u else "—")),
                            (f"{-dA['delta_pp']:+.3f} [{-dA['ci95_hi_pp']:+.3f}, {-dA['ci95_lo_pp']:+.3f}]" if dA else ("0 (SELECTED)" if m == selm else "—")),
                            f"{lat[m]['ttft_mean_ms']:.2f} ({lat[m]['ttft_mean_ms'] / lat[u]['ttft_mean_ms']:.3f}×)", f"{lat[m]['ssd_mb_per_followup']:.2f}", f"{lat[m]['read_calls_mean']:.0f}"])
        L.append(md_table(["Vector (packed)", "Followup %", "Δ vs U_packed pp [CI]", "Δ vs A1-2_packed pp [CI]", "TTFT mean ms (× U_packed)", "SSD MB/followup", "read calls"], rows_pw))
        L.append(md_table(["a − b", "Scope", "Δ pp [95% CI]", "a-only / b-only / both / neither", "Pred. agreement"],
                          [[f"{name(p['a'])} − {name(p['b'])}", p["scope"], f"{p['delta_pp']:+.3f} [{p['ci95_lo_pp']:+.3f}, {p['ci95_hi_pp']:+.3f}]",
                            f"{p['a_only']} / {p['b_only']} / {p['both']} / {p['neither']}", pct(p["prediction_agreement"])] for p in Hres["pairs"] if "metric" not in p]))
        L.append(md_table(["Method", "TTFT mean / p50 / p95 ms", "E2E mean / p50 / p95 ms", "read ms", "scatter ms", "SSD MB/followup", "pread / preadv"],
                          [[l["name"], f"{l['ttft_mean_ms']:.2f} / {l['ttft_p50_ms']:.2f} / {l['ttft_p95_ms']:.2f}", f"{l['e2e_mean_ms']:.2f} / {l['e2e_p50_ms']:.2f} / {l['e2e_p95_ms']:.2f}",
                            f"{l['chunk_io_ms_mean']:.1f}" if l.get("chunk_io_ms_mean") is not None else "—", f"{l['scatter_ms_mean']:.1f}" if l.get("scatter_ms_mean") is not None else "—",
                            f"{l['ssd_mb_per_followup']:.2f}", f"{l['pread_calls_mean']:.0f} / {l['preadv_calls_mean']:.0f}"] for l in Hres["latency"]]))
        L.append(md_table(["Method", "Stored content %", "Store MB mean / p50 / p95", "OS write MB", "Persistence ms mean / p95 (native writer)", "Activation ms mean / p95"],
                          [[s_["name"], f"{s_['stored_content_pct']:.3f}", f"{s_['store_mb_mean']:.2f} / {s_['store_mb_p50']:.2f} / {s_['store_mb_p95']:.2f}", f"{s_['os_write_mb_mean']:.2f}",
                            f"{s_['persistence_ms_mean']:.1f} / {s_['persistence_ms_p95']:.1f} ({s_['native_writer_ms_mean']:.1f})", f"{s_['activation_ms_mean']:.1f} / {s_['activation_ms_p95']:.1f}"] for s_ in Hres["storage"]]))
        L.append(md_table(["Method", "Shared-stream session mean / p50 / p95 ms", "Standalone-equivalent mean / p50 / p95 ms", "Provenance"],
                          [[s_["name"], f"{s_['shared_mean_ms']:.1f} / {s_['shared_p50_ms']:.1f} / {s_['shared_p95_ms']:.1f}", f"{s_['standalone_mean_ms']:.1f} / {s_['standalone_p50_ms']:.1f} / {s_['standalone_p95_ms']:.1f}", s_["provenance"]] for s_ in Hres["sessions"]]))
        L.append(md_table(["a − b", "Scope / metric", "Δ [95% CI]"],
                          [[f"{name(p['a'])} − {name(p['b'])}", f"{p['scope']} / {p['metric']}", f"{p['delta']:+.3f} [{p['ci95_lo']:+.3f}, {p['ci95_hi']:+.3f}]"] for p in Hres["pairs"] if "metric" in p]))
        v1q = {r["method"]: r for r in v1_quality}
        L.append("### v1 HOLD reference (cross-run, unpacked)\n")
        L.append(md_table(["v1 method", "Followup %", "TTFT mean ms", "SSD MB/followup", "preads"],
                          [[r["name"], f"{float(v1q[r['method']]['followup_pct']):.3f}", f"{float(r['ttft_mean_ms']):.2f}", f"{float(r['ssd_mb_per_followup']):.2f}", f"{float(r['preads_mean']):.0f}"] for r in v1_latency]))
    L.append("## Limits and NOT_MEASURED\n")
    L.append("- Paired CIs: seed 1234, 10,000 image-cluster bootstrap resamples; a CI containing 0 is not equivalence. DEV has 40 image clusters and all DEV images are inside HOLD (different questions).\n"
             "- VLC_offline / SA_offline are offline CAL-mean vectorisations of online criteria, not reproductions of VL-Cache / SqueezeAttention; no official VL-Cache code was available (definitions from the paper). PYR / PYR-beta reproduce fixed-shape schedules only.\n"
             "- Packed serving keeps LLaVA's dense GPU buffer and the v1 scatter; no GPU-memory reduction is claimed. H2D bytes, independent GPU footprint, NAND-level coldness, O_DIRECT serving (G9 shows it would be faster) are NOT_MEASURED / not implemented. Qwen2.5-VL and GLM were not run.\n"
             "- v1 → v2 TTFT comparisons are cross-run (different days, same machine); only within-run paired comparisons carry CIs.\n"
             f"- Existing sources: {json.dumps(protection['source_modifications'].get('changed_existing_files'))} changed vs the v1 HOLD freeze (recorded: {json.dumps(protection['source_modifications'].get('known_v1_post_hold_edits'))}); v2 only adds modules.\n")
    (res / "REPORT.md").write_text("\n".join(L))
    (res / "REPRODUCE.md").write_text(f"""# Reproduce (v2)

```bash
cd {ROOT}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.:scripts:tests PYTHONDONTWRITEBYTECODE=1
PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python
RUN={run}
$PY scripts/161_validate_packed_store_v2.py --run $RUN --stage freeze     # Phase 0 (contract SHA, v1 reuse, storage plan, protection before)
$PY scripts/158_allocate_prior_work_vectors.py --run $RUN                 # Phase 1 vectors + G8
$PY scripts/161_validate_packed_store_v2.py --run $RUN --stage g9          # dd microbench
$PY scripts/161_validate_packed_store_v2.py --run $RUN --stage cpu
$PY scripts/161_validate_packed_store_v2.py --run $RUN --stage g3
$PY scripts/161_validate_packed_store_v2.py --run $RUN --stage gpu         # G1/G2/G4 + G6/G7
$PY scripts/159_eval_layer_budget_v2_dev.py --run $RUN --freeze
$PY scripts/159_eval_layer_budget_v2_dev.py --run $RUN --gpu smoke
$PY scripts/159_eval_layer_budget_v2_dev.py --run $RUN --gpu gqa
$PY scripts/159_eval_layer_budget_v2_dev.py --run $RUN --analyze
$PY scripts/162_eval_layer_budget_v2_hold.py --run $RUN --stage freeze
$PY scripts/162_eval_layer_budget_v2_hold.py --run $RUN --stage smoke
$PY scripts/162_eval_layer_budget_v2_hold.py --run $RUN --stage main       # --resume after an interruption
$PY scripts/160_report_layer_budget_v2.py --run $RUN
```
A new reproduction must use a new run root; existing runs/results/stores are never modified. HOLD payloads are deleted after each image's audit and commit (run-local scratch only) and must be regenerated to re-serve.
""")
    print(json.dumps({"report": str(res / "REPORT.md"), "decision": decision and {k: decision[k] for k in ("verdict", "delta_followup_pp", "ci95_pp", "ttft_ratio", "final")}, "protection": protection["status"]}, indent=1))


if __name__ == "__main__":
    main()

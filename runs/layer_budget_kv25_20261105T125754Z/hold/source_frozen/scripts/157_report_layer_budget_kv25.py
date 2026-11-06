#!/usr/bin/env python3
"""Final report, independent HOLD audit, decision and preservation (contract A10/A11).

Reads only run artifacts (never re-runs inference):
  phase1/  sensitivity + coverage,  phase2/ vectors,  gates/,  DEV results
  (scripts/155 --analyze output),  hold/<phase>/images/*/COMMITTED attempts.
Writes results/<RUN>/{REPORT.md, CONTRACT.md, decision.json, hold/*.csv,
independent_audit.json, REPRODUCE.md, protection_result.json, ...}.
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

LW = "ours_prefixstore_kv25_layerwise_"
NAMES = {"recompute": "ReComp", "fullload": "FullLoad", "ours_prefixstore_kv25": "B-uniform", "U": "U (Ours-KV25)"}
SEED = 1234


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def jl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=1, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))
        f.write("\n")


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
    if method.startswith(LW):
        return f"B-{method[len(LW):]}"
    return NAMES.get(method, method)


def pct(x):
    return f"{100 * x:.3f}"


def pp(x):
    return f"{100 * x:+.3f} pp"


def stats(v):
    v = np.asarray(v, dtype=float)
    return v.mean(), np.median(v), np.percentile(v, 95)


# ------------------------------------------------------------------- HOLD
def hold_rows(hold):
    rows = []; attempts = {}
    for cp in sorted((hold / "main" / "images").glob("*/COMMITTED.json")):
        doc = json.loads(cp.read_text()); folder = cp.parent / doc["attempt_id"]
        if sha(folder / "raw.jsonl") != doc["raw_sha256"]:
            raise RuntimeError("committed raw hash mismatch: " + str(folder))
        rows += jl(folder / "raw.jsonl"); attempts[cp.parent.name] = folder
    return rows, attempts


def hold_audit(hold, config, manifest, attempts):
    H = load("layer_budget_hold", "156_eval_layer_budget_hold.py")
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


def hold_analysis(hold, rows, config, res):
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
        follow = turns[2] + turns[3]
        allv = turns[1] + follow
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

    pairs = []
    lw = [m for m in methods if m.startswith(LW)]
    comparisons = [(m, "ours_prefixstore_kv25") for m in lw] + [(m, b) for m in lw for b in ("recompute", "fullload")] + \
                  [("ours_prefixstore_kv25", b) for b in ("recompute", "fullload")]
    if len(lw) == 2:
        comparisons.append((lw[1], lw[0]))
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
        kb = np.array([sum(c["returned"] for c in v["result"].get("os_pread_trace", []) if c["path"].endswith("/k.bin")) for v in f])
        vb = np.array([sum(c["returned"] for c in v["result"].get("os_pread_trace", []) if c["path"].endswith("/v.bin")) for v in f])
        sb = np.array([sum(c["returned"] for c in v["result"].get("os_pread_trace", []) if c["path"].endswith("sep_kv.bin")) for v in f])
        latency.append({"method": m, "name": name(m), "ttft_mean_ms": t[0], "ttft_p50_ms": t[1], "ttft_p95_ms": t[2],
                        "e2e_mean_ms": e[0], "e2e_p50_ms": e[1], "e2e_p95_ms": e[2], "ssd_mb_per_followup": b.mean() / 1e6,
                        "k_mb": kb.mean() / 1e6, "v_mb": vb.mean() / 1e6, "struct_mb": sb.mean() / 1e6, "preads_mean": p.mean(), "followups": len(f)})
    # storage / setup per image
    storage = []; setup = {}
    for m in methods[1:]:
        kind = "raster" if m == "fullload" else "image_only" if m == "ours_prefixstore_kv25" else "layerwise_" + m[len(LW):]
        rs = []
        for img in images:
            folder = next(p for p in (hold / "main" / "images" / img).glob("attempt_*") if (p / "independent_audit.json").exists() and json.loads((p / "independent_audit.json").read_text())["status"] == "PASS")
            per = json.loads((folder / f"{kind}_persistence.json").read_text()); act = json.loads((folder / f"{m}_activation.json").read_text())
            meta = json.loads((folder / f"{kind}_meta.json").read_text())
            n = meta["v_token_num"] - len(meta["newline_idx"])
            stored = meta.get("stored_content_count_total", meta.get("stored_content_count", n) * (32 if "stored_content_count" in meta else 1))
            rs.append({"image_id": img, "bytes": per["file_logical_bytes"], "allocated": per["allocated_bytes"], "os_write": per["os_write_bytes"],
                       "persist_ms": per["service_persistence_ms"], "act_ms": act["outer_wall_ms"], "N": n,
                       "content_rows_total": stored if "stored_content_count_total" in meta else (meta["stored_content_count"] * 32 if "stored_content_count" in meta else n * 32)})
            setup[(m, img)] = per["service_persistence_ms"] + act["outer_wall_ms"]
        b = stats([x["bytes"] for x in rs]); pm = stats([x["persist_ms"] for x in rs]); am = stats([x["act_ms"] for x in rs])
        storage.append({"method": m, "name": name(m), "images": len(rs), "stored_content_pct": 100 * np.mean([x["content_rows_total"] / (32 * x["N"]) for x in rs]),
                        "store_mb_mean": b[0] / 1e6, "store_mb_p50": b[1] / 1e6, "store_mb_p95": b[2] / 1e6,
                        "allocated_mb_mean": np.mean([x["allocated"] for x in rs]) / 1e6, "os_write_mb_mean": np.mean([x["os_write"] for x in rs]) / 1e6,
                        "persistence_ms_mean": pm[0], "persistence_ms_p50": pm[1], "persistence_ms_p95": pm[2],
                        "activation_ms_mean": am[0], "activation_ms_p50": am[1], "activation_ms_p95": am[2], "provenance": "MEASURED"})
    # sessions: physical shared stream (setup once per image, on its first dialogue) and standalone-equivalent (setup every dialogue)
    dialogs = sorted({r["dialog_id"] for r in rows})
    first_dialog = {}
    for r in rows:
        first_dialog.setdefault(r["image_id"], r["dialog_id"]) if r["global_dialog_ordinal"] == min(x["global_dialog_ordinal"] for x in rows if x["image_id"] == r["image_id"]) else None
    sessions = []
    sess_vals = {}
    for m in methods:
        shared, standalone = [], []
        for d in dialogs:
            e2e = sum(by[m][(d, t)]["request_e2e_ms"] for t in (1, 2, 3))
            img = img_of[d]
            su = setup.get((m, img), 0.0) if m != "recompute" else 0.0
            shared.append(e2e + (su if first_dialog.get(img) == d else 0.0))
            standalone.append(e2e + su)
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
    csv_write(res / "hold" / "quality.csv", quality); csv_write(res / "hold" / "latency_io.csv", latency)
    csv_write(res / "hold" / "storage.csv", storage); csv_write(res / "hold" / "paired_analysis.csv", pairs); csv_write(res / "hold" / "sessions.csv", sessions)
    write_json(res / "hold" / "paired_analysis.json", pairs)
    return {"quality": quality, "latency": latency, "storage": storage, "pairs": pairs, "sessions": sessions, "images": len(images), "dialogs": len(dialogs)}


def decide(h, config):
    methods = config["methods"]
    lw = [m for m in methods if m.startswith(LW)]
    sel = lw[0]
    q = {x["method"]: x for x in h["quality"]}; lat = {x["method"]: x for x in h["latency"]}
    d = next(p for p in h["pairs"] if p["a"] == sel and p["b"] == "ours_prefixstore_kv25" and p["scope"] == "followup" and "metric" not in p)
    delta, lo, hi = d["delta_pp"], d["ci95_lo_pp"], d["ci95_hi_pp"]
    ci_excludes_zero = lo > 0 or hi < 0
    mb_ok = lat[sel]["ssd_mb_per_followup"] <= lat["ours_prefixstore_kv25"]["ssd_mb_per_followup"] * 1.02
    ttft_ok = lat[sel]["ttft_mean_ms"] <= lat["ours_prefixstore_kv25"]["ttft_mean_ms"] * 1.05
    if delta >= 1.0 and ci_excludes_zero and mb_ok and ttft_ok:
        verdict = "GO"
    elif (delta >= 0.5 and not ci_excludes_zero) or (delta >= 1.0 and ci_excludes_zero and (mb_ok != ttft_ok)):
        verdict = "PARTIAL"
    else:
        verdict = "NO-GO"
    out = {"verdict": verdict, "selected": sel, "delta_followup_pp": delta, "ci95_pp": [lo, hi], "ci_excludes_zero": ci_excludes_zero,
           "ssd_mb_selected": lat[sel]["ssd_mb_per_followup"], "ssd_mb_uniform": lat["ours_prefixstore_kv25"]["ssd_mb_per_followup"], "ssd_ok_1.02": mb_ok,
           "ttft_mean_selected": lat[sel]["ttft_mean_ms"], "ttft_mean_uniform": lat["ours_prefixstore_kv25"]["ttft_mean_ms"], "ttft_ok_1.05": ttft_ok,
           "followup_pct": {m: q[m]["followup_pct"] for m in methods},
           "rule": "GO: delta>=+1.0pp & CI excludes 0 & MB<=1.02x & TTFT<=1.05x; PARTIAL: delta>=+0.5pp with CI incl. 0, or GO with exactly one of MB/TTFT violated; else NO-GO"}
    if len(lw) == 2:
        ff = lw[1]
        out["f_front_followup_pct"] = q[ff]["followup_pct"]
        out["f_front_minus_selected_pp"] = q[ff]["followup_pct"] - q[sel]["followup_pct"]
        out["front_heavy_schedule_sufficient"] = out["f_front_minus_selected_pp"] >= -0.5
    return out


# ------------------------------------------------------------------ report
def md_table(head, rows):
    return "| " + " | ".join(head) + " |\n|" + "|".join("---" for _ in head) + "|\n" + "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in rows) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--skip-hold", action="store_true")
    args = ap.parse_args()
    run = Path(args.run).resolve(); res = ROOT / "results" / run.name
    res.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / "config.json").read_text())
    shutil.copy2(run / "CONTRACT.md", res / "CONTRACT.md")
    for f in ("sensitivity.csv", "fig_sensitivity.png", "fig_coverage.png", "summary.json", "timing_probe.json", "cal_manifest.json"):
        if (run / "phase1" / f).exists():
            shutil.copy2(run / "phase1" / f, res / ("phase1_" + f))
    for p in sorted((run / "phase2").glob("*")):
        if p.is_file():
            shutil.copy2(p, res / ("phase2_" + p.name if not p.name.startswith("layer_budget") else p.name))
    shutil.copy2(run / "data_split.json", res / "data_split.json")
    for p in sorted((run / "gates").glob("*.json")):
        shutil.copy2(p, res / ("gate_" + p.name))
    p1 = json.loads((run / "phase1" / "summary.json").read_text())
    p2 = json.loads((run / "phase2" / "summary.json").read_text())
    gates = {p.stem: json.loads(p.read_text())["status"] for p in (run / "gates").glob("*.json")}
    dev = json.loads((res / "dev" / "summary.json").read_text())
    sel = json.loads((run / "dev_selection.json").read_text())
    sens = list(csv.DictReader((run / "phase1" / "sensitivity.csv").open()))
    vectors = list(csv.DictReader((run / "phase2" / "vectors.csv").open()))
    hold = run / "hold"
    H = None; audit = None; decision = None
    if not args.skip_hold and (hold / "main").exists():
        hconfig = json.loads((hold / "config.json").read_text()); hmanifest = json.loads((hold / "manifest.json").read_text())
        hconfig.update(_file_sha256=sha(hold / "config.json"), _manifest_sha256=sha(hold / "manifest.json"))
        rows, attempts = hold_rows(hold)
        audit = hold_audit(hold, hconfig, hmanifest, attempts)
        write_json(res / "independent_audit.json", audit)
        H = hold_analysis(hold, rows, hconfig, res)
        decision = decide(H, hconfig)
        decision["audit_status"] = audit["status"]; decision["final"] = audit["status"] == "PASS"
        write_json(res / "decision.json", decision)
    # protection after
    NEW = ["mmimpress/layer_budget.py", "scripts/152_profile_layer_sensitivity.py", "scripts/153_allocate_layer_budget.py", "scripts/154_validate_layer_budget_kv25.py",
           "scripts/155_eval_layer_budget_kv25.py", "scripts/156_eval_layer_budget_hold.py", "scripts/157_report_layer_budget_kv25.py", "tests/test_layer_budget_kv25.py", "docs/layer_budget_kv25_contract.md"]
    before = json.loads((run / "protection_before.json").read_text())
    excl = [f"results/{run.name}"] + NEW
    shas = {k: v for k, v in sha_manifest([ROOT / x for x in ["results", "docs", "mmimpress", "scripts", "tests", "README.md"]], ROOT, exclude=excl).items() if "__pycache__" not in k}
    kv = stat_inventory(sorted(ROOT.glob("kvstore*")), ROOT)
    diff = {"sha_changed": sorted(k for k in before["sha256"] if k in shas and shas[k] != before["sha256"][k]),
            "sha_missing": sorted(set(before["sha256"]) - set(shas)), "sha_added": sorted(set(shas) - set(before["sha256"])),
            "kv_changed": sorted(k for k in before["kvstore_inventory"] if k in kv and kv[k] != before["kvstore_inventory"][k]),
            "kv_missing": sorted(set(before["kvstore_inventory"]) - set(kv)), "kv_added": sorted(set(kv) - set(before["kvstore_inventory"]))}
    protection = {"status": "PASS" if not any(diff.values()) else "FAIL", "files_before": len(before["sha256"]), "files_after": len(shas), "differences": diff,
                  "modified_existing_sources_before_run": json.loads((run / "source_modifications.json").read_text())["changed_existing_files"]}
    write_json(res / "protection_result.json", protection)
    write_json(run / "protection_after.json", {"stage": "after", "utc": time.time(), "sha256": shas, "kvstore_inventory": kv})
    # ---------------------------------------------------------------- REPORT
    L = []
    L.append(f"# Layer-wise budget allocation for PrefixStore-KV25 (LLaVA-NeXT) — results\n")
    L.append(f"Run `{run}`. Contract [CONTRACT.md](CONTRACT.md) (SHA `{config['contract_sha256'][:16]}…`, frozen before measurement). "
             "Model llava-v1.6-vicuna-7b NF4/BF16/eager, FP16 SSD KV, 64-row chunks, greedy max 16, seed 1234.\n")
    if decision is not None:
        L.append(f"## Decision: **{decision['verdict']}**\n")
        L.append(f"HOLD followup accuracy, B-{decision['selected'][len(LW):]} − B-uniform: **{decision['delta_followup_pp']:+.3f} pp** "
                 f"[{decision['ci95_pp'][0]:+.3f}, {decision['ci95_pp'][1]:+.3f}] (image-cluster paired bootstrap, 10,000 resamples, seed 1234; "
                 f"CI {'excludes' if decision['ci_excludes_zero'] else 'includes'} 0). SSD MB/followup {decision['ssd_mb_selected']:.3f} vs uniform "
                 f"{decision['ssd_mb_uniform']:.3f} ({'OK' if decision['ssd_ok_1.02'] else 'VIOLATED'} ≤1.02×); TTFT mean {decision['ttft_mean_selected']:.2f} vs "
                 f"{decision['ttft_mean_uniform']:.2f} ms ({'OK' if decision['ttft_ok_1.05'] else 'VIOLATED'} ≤1.05×). Independent audit: {decision['audit_status']}; "
                 f"final = {decision['final']}.\n")
        if "front_heavy_schedule_sufficient" in decision:
            L.append(f"F-front − SELECTED (HOLD followup) = {decision['f_front_minus_selected_pp']:+.3f} pp → "
                     + ("**a front-heavy layer schedule is sufficient; the contribution is the schedule, not profiling.**" if decision["front_heavy_schedule_sufficient"] else "profiling adds beyond the fixed front-heavy schedule.") + "\n")
    L.append("## Gates\n")
    L.append(md_table(["Gate", "Result"], [[k, v] for k, v in sorted(gates.items())] + [["G3 DEV byte invariance (max |ratio−1|)", f"{dev['g3']['max_abs_ratio_deviation']:.6f} (tol 0.02) {dev['g3']['status']}"],
                                                                                         ["Artifact protection before/after", protection["status"]]]))
    L.append("## Data split\n")
    ds = json.loads((run / "data_split.json").read_text())
    L.append(md_table(["Role", "Source", "Images / questions", "Overlap"],
                      [["CAL", "VQAv2 250-image workload", f"{ds['cal']['images']} / {ds['cal']['questions']}", f"CAL∩DEV {ds['cal']['overlap_with_dev']}, CAL∩HOLD {ds['cal']['overlap_with_hold']}"],
                       ["DEV", "GQA 40 × questions[4:10]", f"{ds['dev']['images']} / {ds['dev']['questions']}", f"DEV∩HOLD {ds['dev_hold_overlap']}"],
                       ["HOLD", "MT-GQA-reconstructed", f"{ds['hold']['images']} / {ds['hold']['dialogues']} dialogues, {ds['hold']['requests_per_arm']} req/arm", "—"],
                       ["(rejected) GQA-large", "data/gqa_large", f"{ds['cal_candidate_gqa_large']['images']}", f"CAL∩HOLD {ds['cal_candidate_gqa_large']['overlap_with_hold']} → 0 remain"]]))
    L.append("## Phase 1 — layer sensitivity (CAL)\n")
    L.append(f"{p1['n_questions']} questions / {p1['n_images']} images; per-variant forward {p1['timing_ms']['per_variant_forward_mean']:.1f} ms. "
             f"Mean KL(FULL‖U) = {p1['kl_U_mean']:.4f}. Spearman ρ between s_l (restore) and KL(D_l) (remove) rankings = {p1['spearman_rho_sR_vs_klD']:.3f}; "
             f"layers with s_l > 0: {p1['s_positive_layers']}/32. CAL accuracy FULL EM {pct(p1['accuracy']['FULL_em'])} %, U EM {pct(p1['accuracy']['U_em'])} % "
             f"(VQA soft {pct(p1['accuracy']['FULL_vqa'])} / {pct(p1['accuracy']['U_vqa'])} %).\n")
    L.append("![sensitivity](phase1_fig_sensitivity.png)\n\n![coverage](phase1_fig_coverage.png)\n")
    L.append(md_table(["layer", "s_l", "KL(R_l)", "KL(D_l)", "Δ first-token gold match R−U (pp)"],
                      [[r["layer"], f"{float(r['s_l']):.4f}", f"{float(r['kl_R_mean']):.4f}", f"{float(r['kl_D_mean']):.4f}", r["delta_first_match_gold_R_minus_U_pp"]] for r in sens]))
    L.append("## Phase 2 — allocation vectors (Σ r_l = 8.0, clip [0.10, 0.60])\n")
    ids = [k for k in vectors[0] if k not in ("layer", "s_l")]
    L.append(md_table(["layer"] + ids, [[r["layer"]] + [r[a] for a in ids] for r in vectors]))
    L.append("SHA256: " + ", ".join(f"{a} `{p2['sha256'][a][:12]}…`" for a in ids) + ". ![ratios](phase2_fig_ratios.png)\n")
    L.append("## Phase 3 — DEV (40 images × 5 cache hits per arm = 200 hits; T1 shared)\n")
    L.append(md_table(["Arm", "Hit acc. %", "All acc. %", "Δ vs U (pp) [95% CI]", "Δ vs FullLoad (pp) [95% CI]", "Pred. agreement vs FullLoad"],
                      [[q["arm"], pct(q["hit_accuracy"]), pct(q["all_accuracy"]),
                        (f"{q['delta_pp_vs_U']:+.2f} [{q['ci95_lo_vs_U']:+.2f}, {q['ci95_hi_vs_U']:+.2f}]" if "delta_pp_vs_U" in q else "—"),
                        (f"{q['delta_pp_vs_fullload']:+.2f} [{q['ci95_lo_vs_fullload']:+.2f}, {q['ci95_hi_vs_fullload']:+.2f}]" if "delta_pp_vs_fullload" in q else "—"),
                        pct(q["prediction_agreement_fullload_hits"])] for q in dev["quality"]]))
    L.append(md_table(["Arm", "TTFT mean / p50 / p95 ms", "E2E mean ms", "SSD MB/hit", "preads"],
                      [[l["arm"], f"{l['ttft_mean_ms']:.1f} / {l['ttft_p50_ms']:.1f} / {l['ttft_p95_ms']:.1f}", f"{l['e2e_mean_ms']:.1f}", f"{l['ssd_mb_per_hit']:.2f}", f"{l['preads_mean']:.1f}"] for l in dev["latency_io"]]))
    L.append(md_table(["Arm", "Store MB/image", "Persistence ms", "Activation ms", "OS write MB"],
                      [[s["arm"], f"{s['store_mb_mean']:.2f}", f"{s['persistence_ms_mean']:.1f}", f"{s['activation_ms_mean']:.1f}", f"{s['os_write_mb_mean']:.2f}"] for s in dev["storage"]]))
    err = list(csv.DictReader((res / "dev" / "error_analysis.csv").open()))
    L.append("Question-type Δ vs FullLoad on hits (pp), spatial highlighted:\n")
    types = []
    for r in err:
        if r["question_type"] not in types:
            types.append(r["question_type"])
    arms = []
    for r in err:
        if r["arm"] not in arms:
            arms.append(r["arm"])
    tab = [[("**spatial**" if t == "spatial" else t), next(r["n_questions"] for r in err if r["question_type"] == t)] +
           [f"{float(next(r['delta_vs_fullload_pp'] for r in err if r['arm'] == a and r['question_type'] == t)):+.1f}" for a in arms] for t in types]
    L.append(md_table(["type", "n"] + arms, tab))
    L.append(f"**SELECTED = {sel['selected']}** (rule: {sel['rule']}); F-front − SELECTED = {sel['f_front_minus_selected_pp']:+.2f} pp → F-front "
             + ("admitted to HOLD" if sel["f_front_admitted_to_hold"] else "not admitted") + f". HOLD arms: {sel['hold_arms']}.\n")
    if H is not None:
        L.append("## Phase 4 — HOLD (MT-GQA-reconstructed, 398 images, 4,061 dialogues × 3 turns, generated history)\n")
        L.append(f"Independent audit {audit['status']}: {audit['requests']} / {audit['expected_requests']} requests, {audit['images']} / 398 images.\n")
        L.append(md_table(["Method", "T1 %", "T2 %", "T3 %", "All % (correct/N)", "Followup % (correct/N)"],
                          [[q["name"], f"{q['T1_pct']:.3f}", f"{q['T2_pct']:.3f}", f"{q['T3_pct']:.3f}", f"{q['all_pct']:.3f} ({q['all_correct']}/{q['all_n']})",
                            f"{q['followup_pct']:.3f} ({q['followup_correct']}/{q['followup_n']})"] for q in H["quality"]]))
        L.append(md_table(["a − b", "Scope", "Δ pp [95% CI]", "a-only / b-only / both / neither", "Pred. agreement"],
                          [[f"{name(p['a'])} − {name(p['b'])}", p["scope"], f"{p['delta_pp']:+.3f} [{p['ci95_lo_pp']:+.3f}, {p['ci95_hi_pp']:+.3f}]",
                            f"{p['a_only']} / {p['b_only']} / {p['both']} / {p['neither']}", pct(p["prediction_agreement"])] for p in H["pairs"] if "metric" not in p]))
        L.append(md_table(["Method", "TTFT mean / p50 / p95 ms", "E2E mean / p50 / p95 ms", "K / V / struct MB", "SSD MB/followup", "preads"],
                          [[l["name"], f"{l['ttft_mean_ms']:.2f} / {l['ttft_p50_ms']:.2f} / {l['ttft_p95_ms']:.2f}", f"{l['e2e_mean_ms']:.2f} / {l['e2e_p50_ms']:.2f} / {l['e2e_p95_ms']:.2f}",
                            f"{l['k_mb']:.2f} / {l['v_mb']:.2f} / {l['struct_mb']:.2f}", f"{l['ssd_mb_per_followup']:.2f}", f"{l['preads_mean']:.2f}"] for l in H["latency"]]))
        L.append(md_table(["Method", "Stored content %", "Store MB mean / p50 / p95", "OS write MB", "Persistence ms mean / p95", "Activation ms mean / p95"],
                          [[s["name"], f"{s['stored_content_pct']:.3f}", f"{s['store_mb_mean']:.2f} / {s['store_mb_p50']:.2f} / {s['store_mb_p95']:.2f}", f"{s['os_write_mb_mean']:.2f}",
                            f"{s['persistence_ms_mean']:.1f} / {s['persistence_ms_p95']:.1f}", f"{s['activation_ms_mean']:.1f} / {s['activation_ms_p95']:.1f}"] for s in H["storage"]]))
        L.append(md_table(["Method", "Shared-stream session mean / p50 / p95 ms", "Standalone-equivalent mean / p50 / p95 ms", "Provenance"],
                          [[s["name"], f"{s['shared_mean_ms']:.1f} / {s['shared_p50_ms']:.1f} / {s['shared_p95_ms']:.1f}", f"{s['standalone_mean_ms']:.1f} / {s['standalone_p50_ms']:.1f} / {s['standalone_p95_ms']:.1f}", s["provenance"]] for s in H["sessions"]]))
        L.append(md_table(["a − b", "Scope / metric", "Δ [95% CI]"],
                          [[f"{name(p['a'])} − {name(p['b'])}", f"{p['scope']} / {p['metric']}", f"{p['delta']:+.3f} [{p['ci95_lo']:+.3f}, {p['ci95_hi']:+.3f}]"] for p in H["pairs"] if "metric" in p]))
    L.append("## Limits and NOT_MEASURED\n")
    L.append("- Paired CIs: seed 1234, 10,000 image-cluster bootstrap resamples; a CI containing 0 is not equivalence. DEV has 40 image clusters and all DEV images are inside HOLD (different questions).\n"
             "- Phase 1 R/D variants use first-token KL/NLL/argmax proxies; only FULL and U were fully decoded. CAL gold is the VQAv2 majority answer.\n"
             "- Layerwise stores keep LLaVA's dense GPU buffer; no GPU-memory reduction is claimed. H2D bytes, independent GPU footprint, non-source T1 capture increment and NAND-level coldness are NOT_MEASURED. Qwen2.5-VL and GLM were not run.\n"
             "- Existing sources modified before the run (recorded with diffs in the run): " + ", ".join(protection["modified_existing_sources_before_run"]) + ".\n")
    (res / "REPORT.md").write_text("\n".join(L))
    (res / "REPRODUCE.md").write_text(f"""# Reproduce

```bash
cd {ROOT}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.:scripts:tests PYTHONDONTWRITEBYTECODE=1
PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python
RUN={run}
# Phase 0/1: data split, N distribution, storage plan (see run/phase0), contract freeze (config.json, phase1_freeze.json)
$PY scripts/152_profile_layer_sensitivity.py --run $RUN --max-images 10 --probe-report $RUN/phase1/timing_probe.json
$PY scripts/152_profile_layer_sensitivity.py --run $RUN
$PY scripts/153_allocate_layer_budget.py --run $RUN
$PY scripts/154_validate_layer_budget_kv25.py --run $RUN --stage cpu
$PY scripts/154_validate_layer_budget_kv25.py --run $RUN --stage g3
$PY scripts/154_validate_layer_budget_kv25.py --run $RUN --stage gpu
$PY scripts/155_eval_layer_budget_kv25.py --run $RUN --freeze
$PY scripts/155_eval_layer_budget_kv25.py --run $RUN --gpu smoke
$PY scripts/155_eval_layer_budget_kv25.py --run $RUN --gpu gqa
$PY scripts/155_eval_layer_budget_kv25.py --run $RUN --analyze
$PY scripts/156_eval_layer_budget_hold.py --run $RUN --stage freeze
$PY scripts/156_eval_layer_budget_hold.py --run $RUN --stage smoke
$PY scripts/156_eval_layer_budget_hold.py --run $RUN --stage main        # --resume after an interruption
$PY scripts/157_report_layer_budget_kv25.py --run $RUN
```
A new reproduction must use a new run root; existing runs/results/stores are never modified. HOLD payloads are deleted after each image's audit and commit and must be regenerated to re-serve.
""")
    print(json.dumps({"report": str(res / "REPORT.md"), "decision": decision and {k: decision[k] for k in ("verdict", "delta_followup_pp", "ci95_pp", "final")}, "protection": protection["status"]}, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Final audit, decision and report for the Qwen layer-wise budget (PyramidKV) packed experiment.

Runs the independent auditor (170) into results/<run>/final, then computes the pre-registered
decision, the U_packed-vs-B integrity check, secondary tables, protection after, the deletion
log and REPORT.md.  Post-hoc analysis only: nothing here feeds back into a measured request.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import qwen_pyr_support as Q
from qwen_pyr_support import B, LABEL, METHODS, PACKED, PRIMARY, ROOT, SENSITIVITY, U_PACKED, load_script, read, sha, write
from mmimpress.recompute_kv25 import paired_bootstrap, sha_manifest, stat_inventory

SEED = 1234
QTYPE = (      # identical to scripts/165 / 159 / 155 (LLaVA reports)
    ("yes/no", re.compile(r"^(is|are|do|does|was|were|has|have|can|could|did|will|would)\b", re.I)),
    ("color", re.compile(r"\bcolou?r\b", re.I)),
    ("count", re.compile(r"\bhow many\b|\bnumber of\b", re.I)),
    ("spatial", re.compile(r"\b(left|right|above|below|behind|front|under|near|beside|top|bottom|side|between|next to)\b", re.I)),
    ("material/attribute", re.compile(r"\b(material|made of|shape|size|texture|large|small|tall|short|thin|thick)\b", re.I)),
    ("object", re.compile(r"^(what|which|who)\b", re.I)),
)
INT = {"turn_id", "ssd_read_bytes", "ssd_preads", "ssd_pread_calls", "ssd_preadv_calls", "N_content", "S_structural", "k", "gpu_cache_kv_bytes",
       "h2d_kv_bytes", "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes", "generated_token_count", "visual_read_bytes", "structural_read_bytes"}
FLOAT = {"score", "ttft_ms", "request_e2e_ms", "core_ttft_ms"}
FOLLOW = lambda k, v: k[1] > 1
ALL = lambda k, v: True


def qtype(question):
    for name, pattern in QTYPE:
        if pattern.search(question):
            return name
    return "other"


def load_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def load_rows(final):
    rows = []
    for r in load_csv(final / "request_diagnostics.csv"):
        for k in INT:
            r[k] = int(float(r[k]))
        for k in FLOAT:
            r[k] = float(r[k])
        for k in list(r):
            if k.startswith("t_"):
                r[k] = float(r[k]) if r[k] not in ("", None) else None
        r["cap_reached"] = r["cap_reached"] == "True"
        rows.append(r)
    return rows


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def md_table(head, rows):
    out = ["| " + " | ".join(str(h) for h in head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out) + "\n"


def paired(by, a, b, images, img_of, flt, key="score", scale=100.0):
    acc = {img: [0.0, 0.0, 0] for img in images}
    for k, v in by[a].items():
        if flt(k, v):
            e = acc[img_of[k[0]]]
            e[0] += v[key]
            e[1] += by[b][k][key]
            e[2] += 1
    per = [tuple(e) for e in acc.values() if e[2] > 0]
    d = paired_bootstrap(per, 10000, SEED)
    return {"a": a, "b": b, "delta": d["mean"] * scale, "ci_lo": d["ci_low"] * scale, "ci_hi": d["ci_high"] * scale,
            "ci_excludes_zero": bool(d["ci_low"] > 0 or d["ci_high"] < 0), "n": int(sum(e[2] for e in per)), "images": len(per), "resamples": 10000, "seed": SEED}


def cells(by, a, b, flt):
    c = Counter()
    agree = []
    for k, v in by[a].items():
        if not flt(k, v):
            continue
        x, y = v["score"], by[b][k]["score"]
        c["both" if x and y else "a_only" if x else "b_only" if y else "neither"] += 1
        agree.append(v["prediction"] == by[b][k]["prediction"])
    return {k: c[k] for k in ("both", "a_only", "b_only", "neither")}, float(np.mean(agree))


def stats(values):
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95))}


def decide(quality_pair, lat):
    delta, lo, hi = quality_pair["delta"], quality_pair["ci_lo"], quality_pair["ci_hi"]
    excl = bool(lo > 0 or hi < 0)
    mb_ok = bool(lat[PRIMARY]["ssd_mb_per_followup"] <= lat[U_PACKED]["ssd_mb_per_followup"] * 1.02)
    ttft_ok = bool(lat[PRIMARY]["ttft_mean_ms"] <= lat[U_PACKED]["ttft_mean_ms"] * 1.05)
    if delta >= 1.0 and excl and mb_ok and ttft_ok:
        verdict = "GO"
    elif (delta >= 0.5 and not excl) or (delta >= 1.0 and excl and (mb_ok != ttft_ok)):
        verdict = "PARTIAL"
    else:
        verdict = "NO-GO"
    note = None
    if verdict == "NO-GO" and excl and delta > 0:
        note = f"규칙상 NO-GO지만 CI는 0 제외: Δ = {delta:+.3f} pp [{lo:+.3f}, {hi:+.3f}]"
    elif verdict == "NO-GO" and excl and delta < 0:
        note = (f"CI 전체가 0보다 작다: Δ = {delta:+.3f} pp [{lo:+.3f}, {hi:+.3f}]. 이 HOLD에서 PYR-b20은 같은 바이트의 균등 배분(U_packed)보다 "
                f"followup 정확도가 낮다. 개선이 아니라 저하이다.")
    elif verdict == "NO-GO":
        note = f"CI가 0을 포함한다: Δ = {delta:+.3f} pp [{lo:+.3f}, {hi:+.3f}]. 차이 미입증이며 동등하다는 뜻이 아니다."
    return {"verdict": verdict, "vector": "PYR-b20_packed", "control": "U_packed", "delta_followup_pp": delta, "ci95_pp": [lo, hi], "ci_excludes_zero": excl,
            "ssd_mb_vector": lat[PRIMARY]["ssd_mb_per_followup"], "ssd_mb_uniform": lat[U_PACKED]["ssd_mb_per_followup"], "ssd_ok_1.02": mb_ok,
            "ttft_mean_vector": lat[PRIMARY]["ttft_mean_ms"], "ttft_mean_uniform": lat[U_PACKED]["ttft_mean_ms"],
            "ttft_ratio": lat[PRIMARY]["ttft_mean_ms"] / lat[U_PACKED]["ttft_mean_ms"], "ttft_ok_1.05": ttft_ok, "note": note}


def integrity(by, res):
    """U_packed and B serve the same KV: all 12,183 requests must agree token by token."""
    keys = sorted(by[B])
    mism = []
    for k in keys:
        a, b = by[B][k], by[U_PACKED][k]
        if a["generated_token_ids"] != b["generated_token_ids"]:
            mism.append({"dialog_id": k[0], "turn_id": k[1], "image_id": a["image_id"], "same_history": a["history_sha256"] == b["history_sha256"],
                         "B_tokens": json.loads(a["generated_token_ids"]), "U_packed_tokens": json.loads(b["generated_token_ids"]),
                         "B_prediction": a["prediction"], "U_packed_prediction": b["prediction"], "B_score": a["score"], "U_packed_score": b["score"],
                         "first_token_equal": a["first_token_id"] == b["first_token_id"]})
    first = [m for m in mism if m["same_history"]]
    doc = {"definition": "generated_token_ids of qwen_ours_prefixstore_kv25 (B, 57 files) vs U_packed (1 file), every (dialogue, turn) of the main",
           "requests": len(keys), "token_exact": len(keys) - len(mism), "token_exact_rate": (len(keys) - len(mism)) / len(keys),
           "mismatches": len(mism), "first_divergence_points_same_history": len(first),
           "first_divergence_by_turn": dict(Counter(m["turn_id"] for m in first)),
           "propagated_through_history": len(mism) - len(first), "status": "PASS" if not mism else "PENDING_REVIEW", "mismatch_dump": mism}
    write(res / "u_packed_vs_B_integrity.json", doc, exclusive=False)
    return doc


def determinism_vs_parent(by, res):
    """Reference only: same requests of the 2026-10-02 Qwen main (recompute, fullload, B)."""
    run = Q.PARENT_RUN
    commits = sorted((run / "main/images").glob("*/COMMITTED.json"))
    if len(commits) != 398:
        doc = {"status": "NOT_MEASURED", "reason": "parent main raw not available locally"}
        write(res / "determinism_vs_parent.json", doc, exclusive=False)
        return doc
    agree = {m: Counter() for m in METHODS[:3]}
    for cp in commits:
        com = json.loads(cp.read_text())
        with (cp.parent / com["attempt_id"] / "raw.jsonl").open() as f:
            for line in f:
                r = json.loads(line)
                m = r["method_id"]
                cur = by[m].get((r["dialog_id"], r["turn_id"]))
                if cur is None:
                    agree[m]["missing"] += 1
                    continue
                agree[m]["n"] += 1
                agree[m]["token_exact"] += json.loads(cur["generated_token_ids"]) == r["result"]["generated_token_ids"]
                agree[m]["prediction_equal"] += cur["prediction"] == r["prediction"]
                agree[m]["score_equal"] += cur["score"] == r["score"]
                if r["turn_id"] > 1:
                    agree[m]["followup_n"] += 1
                    agree[m]["followup_token_exact"] += json.loads(cur["generated_token_ids"]) == r["result"]["generated_token_ids"]
    doc = {"status": "MEASURED", "parent_run": str(run.relative_to(ROOT)), "scope": "reference only; absolute TTFT is never compared across runs",
           "methods": {m: {**dict(c), "token_exact_rate": c["token_exact"] / c["n"], "prediction_agreement": c["prediction_equal"] / c["n"],
                           "followup_token_exact_rate": c["followup_token_exact"] / c["followup_n"]} for m, c in agree.items()}}
    write(res / "determinism_vs_parent.json", doc, exclusive=False)
    return doc


def figure(res, docs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor=surface)
    ax.set_facecolor(surface)
    x = list(range(28))
    ax.plot(x, docs["U"]["layer_keep_ratios"], color=muted, lw=1.5, ls=":", label="U (uniform 0.25, control)")
    ax.plot(x, docs["PYR-b20"]["layer_keep_ratios"], color="#2a78d6", lw=2, marker="o", ms=4, markevery=3, label="PYR-b20 (PyramidKV β=20, primary)")
    ax.plot(x, docs["PYR-b2p5"]["layer_keep_ratios"], color="#eb6834", lw=2, ls="--", marker="s", ms=4, markevery=3, label="PYR-b2p5 (PyramidKV β=2.5, sensitivity)")
    for name, dy in (("PYR-b20", -13), ("PYR-b2p5", 7)):
        r = docs[name]["layer_keep_ratios"]
        ax.annotate(f"{r[0]:.4g}", (0, r[0]), textcoords="offset points", xytext=(6, 3), fontsize=8, color=ink)
        ax.annotate(f"{r[-1]:.4g}", (27, r[-1]), textcoords="offset points", xytext=(-2, dy), ha="right", fontsize=8, color=ink)
    ax.set_xlabel("decoder layer (Qwen2.5-VL-7B, 28 layers)", color=ink)
    ax.set_ylabel("keep ratio r_l  (Σ r_l = 7.0)", color=ink)
    ax.set_title("Layer-wise keep ratios on Qwen2.5-VL: PyramidKV schedules vs uniform", color=ink, fontsize=11, loc="left")
    ax.set_xlim(-0.5, 27.5)
    ax.set_ylim(0, 0.52)
    ax.grid(color=grid, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=muted, labelsize=8)
    leg = ax.legend(frameon=False, fontsize=8, loc="upper right")
    for t in leg.get_texts():
        t.set_color(ink)
    fig.tight_layout()
    fig.savefig(res / "fig_vectors_qwen.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def protection_after(run, res):
    before = read(run / "protection_before.json")
    excl = before["excluded_new_paths"]
    shas = {k: v for k, v in sha_manifest([ROOT / x for x in ["results", "docs", "mmimpress", "scripts", "tests", "README.md"]], ROOT, exclude=excl).items() if "__pycache__" not in k}
    kv = stat_inventory(sorted(ROOT.glob("kvstore*")), ROOT)
    runs = stat_inventory(sorted(p for p in (ROOT / "runs").iterdir() if p.name != run.name), ROOT)

    def inv_diff(b, a):
        return {"changed": sorted(k for k in b if k in a and a[k] != b[k]), "missing": sorted(set(b) - set(a)), "added": sorted(set(a) - set(b))}
    diff = {"sha_changed": sorted(k for k in before["sha256"] if k in shas and shas[k] != before["sha256"][k]),
            "sha_missing": sorted(set(before["sha256"]) - set(shas)), "sha_added": sorted(set(shas) - set(before["sha256"]))}
    kvd, rund = inv_diff(before["kvstore_inventory"], kv), inv_diff(before["runs_inventory"], runs)
    # other sessions may add new run roots / result files while this run is in progress: additions are reported, changes and deletions are failures
    hard = [diff["sha_changed"], diff["sha_missing"], kvd["changed"], kvd["missing"], kvd["added"], rund["changed"], rund["missing"]]
    new_ok = all((ROOT / p).exists() for p in Q.NEW_PATHS)
    config = read(run / "config.json")
    frozen_changed = sorted(k for k, d in config["source_sha256"].items() if sha(ROOT / k) != d)
    protection = {"status": "PASS" if not any(hard) and not frozen_changed and new_ok else "FAIL", "files_before": len(before["sha256"]), "files_after": len(shas),
                  "kvstore_files_before": len(before["kvstore_inventory"]), "kvstore_files_after": len(kv), "runs_files_before": len(before["runs_inventory"]), "runs_files_after": len(runs),
                  "differences": {"sha": diff, "kvstore": kvd, "runs": {"changed": rund["changed"], "missing": rund["missing"], "added_count": len(rund["added"])}},
                  "sha_added_reported_only": diff["sha_added"][:50], "runs_added_reported_only": rund["added"][:50],
                  "frozen_runtime_sources_changed_since_freeze": frozen_changed, "new_files_of_this_run": Q.NEW_PATHS,
                  "amendment_files": [p.name for p in sorted(run.glob("freeze_amendment_*.json"))], "source_modifications": read(run / "source_modifications.json"),
                  "scope": "full SHA256 of results/docs/mmimpress/scripts/tests/README.md (new paths of this run excluded); (size, mtime_ns, inode) of kvstore* and of every other runs/* root"}
    write(res / "protection_result.json", protection, exclusive=False)
    write(run / "protection_after.json", {"stage": "after", "utc": time.time(), "sha256": shas, "kvstore_inventory": kv, "runs_inventory": runs}, exclusive=False)
    import hashlib

    def digest(inv):
        return hashlib.sha256(json.dumps(inv, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for stage, sh, kvi, ri, src in (("before", before["sha256"], before["kvstore_inventory"], before["runs_inventory"], "protection_before.json"), ("after", shas, kv, runs, "protection_after.json")):
        write(res / f"protection_{stage}.json", {"stage": stage, "full_file": str((run / src).relative_to(ROOT)), "full_file_sha256": sha(run / src), "sha256": sh, "kvstore_inventory": kvi,
                                                 "runs_inventory_files": len(ri), "runs_inventory_digest_sha256": digest(ri)}, exclusive=False)
    return protection


def deletion_log(run, res):
    images, files = [], 0
    for phase in ("integration", "smoke", "main"):
        for cp in sorted((run / phase / "images").glob("*/COMMITTED.json")):
            folder = cp.parent / json.loads(cp.read_text())["attempt_id"]
            rc = read(folder / "cleanup_receipt.json")
            allow = read(folder / "cleanup_allowlist.json")
            ok = rc["status"] == "PASS" and rc["removed_new_files"] == len(allow["file_sha256"]) and not Path(allow["scratch"]).exists() \
                and Path(allow["scratch"]).is_relative_to(run / "scratch")
            images.append({"phase": phase, "image_id": cp.parent.name, "scratch": allow["scratch"], "removed_files": rc["removed_new_files"],
                           "allowlist_sha256": sha(folder / "cleanup_allowlist.json"), "ok": ok})
            files += rc["removed_new_files"]
    gate = [{"case": p.parent.name, "removed_files": read(p)["removed_gate_store_files"]} for p in sorted((run / "gates/gpu").glob("*/evidence.json"))]
    left = [str(p.relative_to(run)) for p in (run / "scratch").rglob("*") if p.is_file()]
    left_gate = [str(p.relative_to(run)) for p in (run / "gates/gpu").glob("*/stores")]
    doc = {"status": "PASS" if all(x["ok"] for x in images) and not left and not left_gate else "FAIL", "image_bundles": len(images), "removed_scratch_files": files,
           "gate_cases": gate, "removed_gate_store_files": sum(g["removed_files"] for g in gate), "files_left_in_scratch": left, "gate_store_dirs_left": left_gate,
           "scope": "only run-local scratch stores (exact SHA allowlist after independent audit + commit) and run-local gate stores; no existing store, run or result was deleted",
           "images": images}
    write(res / "deletion_log.json", doc, exclusive=False)
    return doc


def f(x, n=3):
    return f"{x:.{n}f}"


def pm(d, n=3):
    return f"{d['delta']:+.{n}f} [{d['ci_lo']:+.{n}f}, {d['ci_hi']:+.{n}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    a = ap.parse_args()
    run = a.run.resolve()
    res = ROOT / "results" / run.name
    res.mkdir(parents=True, exist_ok=True)
    final = res / "final"
    config = read(run / "config.json")
    docs = Q.load_vectors(run)
    if not (final / "independent_audit.json").exists():
        A = load_script("_qpyr_final_auditor", "170_audit_qwen_layer_budget_pyr.py")
        A.run_audit(run, final)
    audit = read(final / "independent_audit.json")
    rows = load_rows(final)
    by = defaultdict(dict)
    for r in rows:
        by[r["method_id"]][(r["dialog_id"], r["turn_id"])] = r
    images = sorted({r["image_id"] for r in rows})
    img_of = {r["dialog_id"]: r["image_id"] for r in rows}
    assert audit["status"] == "PASS" and audit["main_requests"] == 73098 and len(images) == 398 and all(len(by[m]) == 12183 for m in METHODS)

    # ---------------------------------------------------------------- quality
    quality = []
    for m in METHODS:
        d = by[m]
        turns = {t: [v["score"] for k, v in d.items() if k[1] == t] for t in (1, 2, 3)}
        follow = turns[2] + turns[3]
        allv = turns[1] + follow
        quality.append({"method": m, "name": LABEL[m], "T1_pct": 100 * np.mean(turns[1]), "T2_pct": 100 * np.mean(turns[2]), "T3_pct": 100 * np.mean(turns[3]),
                        "all_pct": 100 * np.mean(allv), "all_correct": int(sum(allv)), "all_n": len(allv),
                        "followup_pct": 100 * np.mean(follow), "followup_correct": int(sum(follow)), "followup_n": len(follow)})
    write_csv(res / "quality.csv", quality)
    qual = {q["method"]: q for q in quality}

    # ------------------------------------------------------- paired analysis
    pairs = []
    plan = [(PRIMARY, U_PACKED), (SENSITIVITY, U_PACKED), (PRIMARY, SENSITIVITY), (U_PACKED, B)] + [(m, base) for base in ("recompute", "fullload") for m in (B,) + tuple(PACKED)]
    for a_, b_ in plan:
        for scope, flt in (("followup", FOLLOW), ("all", ALL), ("T2", lambda k, v: k[1] == 2), ("T3", lambda k, v: k[1] == 3)):
            d = paired(by, a_, b_, images, img_of, flt)
            c, agree = cells(by, a_, b_, flt)
            pairs.append({**d, "a_name": LABEL[a_], "b_name": LABEL[b_], "scope": scope, "metric": "accuracy_pp", **c, "prediction_agreement": agree})
    for a_, b_ in [(U_PACKED, B), (PRIMARY, U_PACKED), (SENSITIVITY, U_PACKED)] + [(m, base) for base in ("recompute", "fullload") for m in (B,) + tuple(PACKED)]:
        d = paired(by, a_, b_, images, img_of, FOLLOW, key="ttft_ms", scale=1.0)
        pairs.append({**d, "a_name": LABEL[a_], "b_name": LABEL[b_], "scope": "followup", "metric": "ttft_ms"})
    write_csv(res / "paired_analysis.csv", pairs)
    write(res / "paired_analysis.json", pairs, exclusive=False)

    def pair(a_, b_, scope="followup", metric="accuracy_pp"):
        return next(p for p in pairs if p["a"] == a_ and p["b"] == b_ and p["scope"] == scope and p["metric"] == metric)

    # --------------------------------------------------------------- latency
    seg = ["t_prompt_preparation", "t_store_load_inclusive", "t_raw_pread", "t_fixed_first_k_planning", "t_layerwise_planning", "t_h2d_and_assembly",
           "t_mask_hook_install", "t_suffix_prefill", "t_multimodal_prefill", "t_decode"]
    latency = []
    for m in METHODS:
        hit = [v for k, v in by[m].items() if k[1] > 1]
        one = [v for k, v in by[m].items() if k[1] == 1]
        t, e = stats([r["ttft_ms"] for r in hit]), stats([r["request_e2e_ms"] for r in hit])
        row = {"method": m, "name": LABEL[m], "followups": len(hit), "ttft_mean_ms": t["mean"], "ttft_p50_ms": t["p50"], "ttft_p95_ms": t["p95"],
               "e2e_mean_ms": e["mean"], "e2e_p95_ms": e["p95"], "T1_ttft_mean_ms": float(np.mean([r["ttft_ms"] for r in one])),
               "ssd_mb_per_followup": float(np.mean([r["ssd_read_bytes"] for r in hit])) / 1e6, "read_calls_per_followup": float(np.mean([r["ssd_preads"] for r in hit])),
               "pread_per_followup": float(np.mean([r["ssd_pread_calls"] for r in hit])), "preadv_per_followup": float(np.mean([r["ssd_preadv_calls"] for r in hit])),
               "gpu_cache_kv_mb_per_followup": float(np.mean([r["gpu_cache_kv_bytes"] for r in hit])) / 1e6,
               "peak_gpu_allocated_max_mb": max(r["peak_gpu_allocated_bytes"] for r in hit + one) / 1e6,
               "peak_gpu_allocated_followup_mean_mb": float(np.mean([r["peak_gpu_allocated_bytes"] for r in hit])) / 1e6}
        for s in seg:
            vals = [r[s] for r in hit if r[s] is not None]
            row["seg_" + s[2:] + "_mean_ms"] = float(np.mean(vals)) if vals else None
        latency.append(row)
    lat = {x["method"]: x for x in latency}
    for x in latency:
        x["ttft_ratio_vs_U_packed"] = x["ttft_mean_ms"] / lat[U_PACKED]["ttft_mean_ms"]
        x["ttft_ratio_vs_ReComp"] = x["ttft_mean_ms"] / lat["recompute"]["ttft_mean_ms"]
        x["ttft_ratio_vs_FullLoad"] = x["ttft_mean_ms"] / lat["fullload"]["ttft_mean_ms"]
    write_csv(res / "latency_io.csv", latency)

    # ---------------------------------------------------- storage / sessions
    stages = load_csv(final / "persistence_stage_scope.csv")
    storage = []
    for m in METHODS[1:]:
        ss = [s for s in stages if s["method_id"] == m]
        w, act = stats([float(s["writer_ms"]) for s in ss]), stats([float(s["activation_ms"]) for s in ss])
        nat = stats([float(s["native_writer_wall_ms"]) for s in ss])
        storage.append({"method": m, "name": LABEL[m], "builds": len(ss), "write_mb_mean": float(np.mean([float(s["write_bytes"]) for s in ss])) / 1e6,
                        "allocated_mb_mean": float(np.mean([float(s["allocated_bytes"]) for s in ss])) / 1e6,
                        "outer_persistence_mean_ms": w["mean"], "outer_persistence_p50_ms": w["p50"], "outer_persistence_p95_ms": w["p95"],
                        "native_writer_mean_ms": nat["mean"], "ssd_write_mean_ms": float(np.mean([float(s["ssd_write_ms"]) for s in ss])),
                        "fsync_mean_ms": float(np.mean([float(s["fsync_ms"]) for s in ss])), "activation_mean_ms": act["mean"], "activation_p50_ms": act["p50"],
                        "activation_p95_ms": act["p95"], "files_per_store": 59 if m in ("fullload", B) else 3})
    write_csv(res / "storage.csv", storage)
    sess_rows = load_csv(final / "session_summary.csv")
    per_dialogue = load_csv(final / "session_per_dialogue.csv")
    sby = defaultdict(dict)
    for r in per_dialogue:
        sby[r["method_id"]][(r["dialog_id"], 0)] = {"shared": float(r["shared_stream_session_E2E_ms"]), "standalone": float(r["derived_standalone_E2E_ms"])}
    sess_pairs = []
    for m in (B,) + tuple(PACKED):
        for base in ("recompute", "fullload", B):
            if m == base:
                continue
            for key in ("shared", "standalone"):
                d = paired(sby, m, base, images, img_of, ALL, key=key, scale=1.0)
                sess_pairs.append({"a_name": LABEL[m], "b_name": LABEL[base], "accounting": "shared_stream_session_ms" if key == "shared" else "derived_standalone_session_ms",
                                   "delta_ms": d["delta"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"], "dialogues": d["n"]})
    sessions = [{"method": r["method_id"], "name": r["label"], **{k: float(r[k]) for k in r if k.endswith("_ms") or k.endswith("_image")}} for r in sess_rows]
    write_csv(res / "sessions.csv", sessions)
    write(res / "sessions_paired.json", sess_pairs, exclusive=False)
    sess = {s["method"]: s for s in sessions}

    # --------------------------------------------------------- question types
    types = ["all"] + [t for t, _ in QTYPE] + ["other"]
    qt = []
    for t in types:
        flt = (lambda k, v: k[1] > 1) if t == "all" else (lambda k, v, t=t: k[1] > 1 and qtype(v["question"]) == t)
        n = sum(1 for k, v in by[U_PACKED].items() if flt(k, v))
        if n == 0:
            continue
        row = {"question_type": t, "n_followup": n}
        for m in METHODS:
            row[f"acc_{LABEL[m]}_pct"] = 100 * float(np.mean([v["score"] for k, v in by[m].items() if flt(k, v)]))
        for m in (PRIMARY, SENSITIVITY):
            d = paired(by, m, U_PACKED, images, img_of, flt)
            row[f"delta_{LABEL[m]}_minus_U_pp"] = d["delta"]
            row[f"ci_lo_{LABEL[m]}"] = d["ci_lo"]
            row[f"ci_hi_{LABEL[m]}"] = d["ci_hi"]
        qt.append(row)
    write_csv(res / "question_types.csv", qt)

    # ------------------------------------------------ layer k_l / bytes table
    ktab = read(run / "phase0" / "k_tables.json")
    layer_rows = []
    for li in range(28):
        row = {"layer": li}
        for a_ in Q.ALLOCS:
            row[f"r_{a_}"] = docs[a_]["layer_keep_ratios"][li]
        for n in (260, 345, 529):
            for a_ in Q.ALLOCS:
                k = ktab["k_per_layer_by_N"][str(n)][a_][li]
                row[f"k_{a_}_N{n}"] = k
                row[f"KV_bytes_{a_}_N{n}"] = 2 * k * 1024
        layer_rows.append(row)
    write_csv(res / "layer_k_bytes.csv", layer_rows)
    gpu_equal = all(by[m][k]["gpu_cache_kv_bytes"] == v["gpu_cache_kv_bytes"] and by[m][k]["ssd_read_bytes"] == v["ssd_read_bytes"]
                    for k, v in by[B].items() if k[1] > 1 for m in PACKED)

    # ------------------------------------------------ integrity, determinism
    integ = integrity(by, res)
    det = determinism_vs_parent(by, res)

    # -------------------------------------------------------------- decision
    primary = decide(pair(PRIMARY, U_PACKED), lat)
    sens = pair(SENSITIVITY, U_PACKED)
    b20_vs_b2p5 = pair(PRIMARY, SENSITIVITY)
    started = read(run / "main_started_at.json")["at_unix"]
    finished = read(run / "main_finished_at.json")["at_unix"]
    interruptions = sorted(p.name for p in run.glob("interruption_*.json"))
    failed_attempts = sorted(str(p.parent.relative_to(run)) for p in run.glob("*/images/*/attempt_*/failure.json"))
    gates = {"QG1 CPU (vectors, row budget, validator, no LLaVA defaults)": read(run / "gates/qg1_cpu.json")["status"],
             "기존 CPU 테스트 (non-GLM 전체)": read(run / "gates/cpu_regression.json")["status"]}
    gpu = read(run / "gates/qg_gpu.json")
    names = {"QG2": "QG2 U 회귀 (bytes·텐서·SDPA 입력·logits·ID bitwise)", "QG3": "QG3 layerwise 구현 (PYR-b20, PYR-b2p5)", "QG4": "QG4 FP64 의미 검증 1e-12",
             "QG5": "QG5 negative fixture", "QG6": "QG6 I/O (preadv 1회, vision 0, score 0)"}
    for g, label in names.items():
        gates[label] = "PASS" if gpu["per_gate"][g] else "FAIL"
    integ_val = read(run / "integration_validation.json")
    gates["QG7 통합 (6 arm × 12 요청 × 4 dialogue + 독립 감사)"] = integ_val["status"]
    gates["Smoke 72 요청"] = read(run / "smoke/validation.json")["status"]
    final_ok = audit["status"] == "PASS" and audit["main_requests"] == 73098 and audit["failed_or_skipped_requests"] == 0
    decision_value = primary["verdict"] if integ["status"] == "PASS" and final_ok else "PENDING_REVIEW"
    protection = protection_after(run, res)
    dele = deletion_log(run, res)
    decision = {"decision": decision_value, "primary": primary, "primary_rule_verdict_before_integrity_check": primary["verdict"],
                "integrity_u_packed_vs_B": {k: v for k, v in integ.items() if k != "mismatch_dump"},
                "sensitivity_reported_only": {"PYR-b2p5_minus_U_packed": {k: sens[k] for k in ("delta", "ci_lo", "ci_hi", "ci_excludes_zero")},
                                              "PYR-b20_minus_PYR-b2p5": {k: b20_vs_b2p5[k] for k in ("delta", "ci_lo", "ci_hi", "ci_excludes_zero")},
                                              "note": "β=2.5 수치는 보고만 한다. 판정에 쓰지 않으며, β=20보다 좋아 보이더라도 HOLD에서 본 뒤의 사후 관찰이므로 다른 데이터에서 확인이 필요하다."},
                "audit_status": audit["status"], "final": final_ok, "requests": audit["main_requests"], "expected_requests": 73098, "images": len(images),
                "failed_or_skipped_requests": audit["failed_or_skipped_requests"], "failed_attempt_records": failed_attempts, "interruptions": interruptions,
                "main_wall_hours_first_start_to_finish": (finished - started) / 3600,
                "followup_pct": {LABEL[m]: qual[m]["followup_pct"] for m in METHODS}, "gates": gates,
                "rule": config["decision_rule"], "contract_sha256": config["contract_sha256"], "config_sha256": sha(run / "config.json"),
                "protection": protection["status"], "deletion_log": dele["status"],
                "not_run": ["MT-VQA", "GLM", "LLaVA"]}
    write(res / "decision.json", decision, exclusive=False)

    # --------------------------------------------------------------- copies
    figure(res, docs)
    for name in ("CONTRACT.md", "design_ragged_cache.md", "config.json", "storage_plan.json", "integration_validation.json", "source_modifications.json"):
        shutil.copy2(run / name, res / name)
    shutil.copy2(final / "independent_audit.json", res / "independent_audit.json")
    (res / "vectors").mkdir(exist_ok=True)
    for p in (run / "vectors").iterdir():
        shutil.copy2(p, res / "vectors" / p.name)
    (res / "phase0").mkdir(exist_ok=True)
    for p in (run / "phase0").iterdir():
        shutil.copy2(p, res / "phase0" / p.name)
    (res / "gates").mkdir(exist_ok=True)
    for name in ("qg1_cpu.json", "cpu_regression.json", "qg_gpu.json"):
        shutil.copy2(run / "gates" / name, res / "gates" / name)
    (res / "gates" / "gpu_cases").mkdir(exist_ok=True)
    for p in sorted((run / "gates/gpu").glob("*/evidence.json")):
        shutil.copy2(p, res / "gates" / "gpu_cases" / f"{p.parent.name}.json")
    shutil.copy2(run / "smoke/validation.json", res / "gates" / "smoke_validation.json")

    # ---------------------------------------------------------------- report
    L = []
    P, S_, U_ = PRIMARY, SENSITIVITY, U_PACKED
    d_main = pair(P, U_)
    L.append("# Qwen2.5-VL Layer-wise Budget — PyramidKV(β=20) + PackedPrefixStore, MT-GQA HOLD\n")
    L.append(f"Run `{run.relative_to(ROOT)}`. 계약 [CONTRACT.md](CONTRACT.md) (SHA `{config['contract_sha256'][:16]}…`, gate·smoke·main 전에 freeze). "
             f"설계 [design_ragged_cache.md](design_ragged_cache.md). 단위: MB = 10^6 B, ms, 정확도 차이는 pp. followup = T2–T3.\n")
    L.append("## 판정\n")
    L.append(f"**{decision_value}** — 주 판정(사전 고정): Δ = followup 정확도(PYR-b20_packed − U_packed) = **{d_main['delta']:+.3f} pp** "
             f"[{d_main['ci_lo']:+.3f}, {d_main['ci_hi']:+.3f}] (이미지 단위 paired bootstrap 95% CI, 10,000회, seed 1234).\n")
    L.append(md_table(["조건", "값", "기준", "충족"],
                      [["Δ followup (pp)", f"{primary['delta_followup_pp']:+.3f}", "≥ +1.0 (GO) / ≥ +0.5 (PARTIAL)", "예" if primary["delta_followup_pp"] >= 1.0 else "아니오"],
                       ["95% CI가 0 제외", f"[{primary['ci95_pp'][0]:+.3f}, {primary['ci95_pp'][1]:+.3f}]", "0 제외 (양의 방향)", ("예" if primary["ci95_pp"][0] > 0 else "아니오 — 0 제외이지만 음의 방향") if primary["ci_excludes_zero"] else "아니오"],
                       ["SSD MB/followup", f"{primary['ssd_mb_vector']:.3f} vs U {primary['ssd_mb_uniform']:.3f}", "≤ U × 1.02", "예" if primary["ssd_ok_1.02"] else "아니오"],
                       ["TTFT 평균 (ms)", f"{primary['ttft_mean_vector']:.3f} vs U {primary['ttft_mean_uniform']:.3f} ({primary['ttft_ratio']:.3f}×)", "≤ U × 1.05", "예" if primary["ttft_ok_1.05"] else "아니오"],
                       ["U_packed ≡ B (토큰 단위)", f"{integ['token_exact']:,} / {integ['requests']:,}", "100%", "예" if integ["status"] == "PASS" else "아니오 → PENDING_REVIEW"],
                       ["요청 완주·실패 0·감사", f"{audit['main_requests']:,} / 73,098, 실패·스킵 {audit['failed_or_skipped_requests']}, 감사 {audit['status']}", "전부", "예" if final_ok else "아니오"]]))
    if primary["note"]:
        L.append(primary["note"] + "\n")
    if integ["status"] != "PASS":
        L.append(f"규칙상 판정은 {primary['verdict']}이지만 U_packed와 B의 예측이 100% 일치하지 않아 decision은 PENDING_REVIEW이다. 불일치 분석은 아래 '정합성 사후 확인' 참조.\n")
    L.append("이 실험에서 반드시 알아야 할 것:\n")
    L.append("- **PYR-b20 = PyramidKV 공개 코드 식(β=20)의 layer 비율 그대로이며 하한이 없다.** 원 PyramidKV의 attention 기반 token 선택은 쓰지 않고 "
             "image-only saliency 순서에서 layer별 자르는 지점만 따랐다.\n"
             "- **Qwen은 N이 작아(260–529) β=20의 마지막 layer가 3–7행이다.** (LLaVA는 17–36행)\n"
             "- **모든 packed arm은 같은 바이트·같은 I/O 경로·같은 token 순서를 쓰고 layer별 자르는 지점만 다르다.**\n"
             "- ragged cache는 설계 A(layer별 compact cache + 각 layer self-attention forward pre-hook으로 mask를 오른쪽 정렬)이다. "
             "검증 범위는 고정 core pair 10개(hit prompt 2종)와 integration dialogue 4개이다.\n"
             "- MT-VQA·GLM·LLaVA는 실행하지 않았다.\n")

    L.append("## Gate\n")
    L.append(md_table(["Gate", "결과"], [[k, v] for k, v in gates.items()] + [["독립 최종 감사 (73,098 요청)", audit["status"]], ["Artifact protection before/after", protection["status"]],
                                                                             ["Deletion log (run-local scratch·gate store만)", dele["status"]]]))
    creg = read(run / "gates/cpu_regression.json")
    L.append(f"기존 CPU 테스트: {creg['ran']} (새 테스트 14개 포함), returncode 0. GLM-4.6V 테스트는 분리 실행했고 {len(creg['glm_errors'])}건이 이 변경 전과 똑같이 실패한다"
             f"(설치된 transformers API 문제, 새 파일을 import하지 않음). QG4의 legacy FP32 1e-5 비교는 진단값으로만 기록했다(gate 아님).\n")

    L.append("## 정확도\n")
    L.append(md_table(["Arm", "T1 %", "T2 %", "T3 %", "All % (정답/N)", "Followup % (정답/N)"],
                      [[q["name"], f(q["T1_pct"]), f(q["T2_pct"]), f(q["T3_pct"]), f"{q['all_pct']:.3f} ({q['all_correct']}/{q['all_n']})",
                        f"{q['followup_pct']:.3f} ({q['followup_correct']}/{q['followup_n']})"] for q in quality]))
    L.append(md_table(["비교 (A − B)", "Followup Δpp [95% CI]", "All Δpp [95% CI]", "T2 Δpp", "T3 Δpp", "A만 / B만 정답 (followup)", "예측 일치율"],
                      [[f"{LABEL[x]} − {LABEL[y]}", pm(pair(x, y)), pm(pair(x, y, "all")), f"{pair(x, y, 'T2')['delta']:+.3f}", f"{pair(x, y, 'T3')['delta']:+.3f}",
                        f"{pair(x, y)['a_only']} / {pair(x, y)['b_only']}", f"{100 * pair(x, y)['prediction_agreement']:.2f}%"] for x, y in plan]))
    L.append("T1은 모든 arm이 full-pixel이다. 차이는 generated history가 갈라지는 효과를 포함한 method 수준 비교이다. CI가 0을 포함하는 차이는 '차이 미입증'이며 동등하다는 뜻이 아니다.\n")
    L.append(f"β 민감도(보고만, 판정에 쓰지 않음): PYR-b2p5 − U_packed = {pm(sens)} pp, PYR-b20 − PYR-b2p5 = {pm(b20_vs_b2p5)} pp.\n")
    if sens["delta"] > d_main["delta"] and decision_value != "GO":
        L.append("β=2.5가 β=20보다 좋게 나왔지만 판정은 바꾸지 않는다. 이것은 HOLD에서 본 뒤의 사후 관찰이므로 다른 데이터에서 확인이 필요하다.\n")

    L.append("## 손실 회복 (ReComp·FullLoad 대비, followup)\n")
    L.append(md_table(["Arm", "vs ReComp Δpp [95% CI]", "vs FullLoad Δpp [95% CI]", "TTFT / ReComp", "TTFT / FullLoad"],
                      [[LABEL[m], pm(pair(m, "recompute")), pm(pair(m, "fullload")), f"{lat[m]['ttft_ratio_vs_ReComp']:.3f}×", f"{lat[m]['ttft_ratio_vs_FullLoad']:.3f}×"]
                       for m in (B,) + tuple(PACKED)]))
    gap = -pair(U_, "recompute")["delta"]
    rec = d_main["delta"]
    if gap > 0 and rec >= 0:
        L.append(f"균등 25%(U_packed)의 ReComp 대비 손실은 {gap:.3f} pp이다. PYR-b20은 그중 {rec:+.3f} pp를 되찾았다(손실의 {100 * rec / gap:.1f}%).")
    elif gap > 0:
        L.append(f"균등 25%(U_packed)의 ReComp 대비 손실은 {gap:.3f} pp이다. PYR-b20은 손실을 되찾지 못하고 {-rec:.3f} pp 더 키웠다"
                 f"(ReComp 대비 손실 {gap - rec:.3f} pp, 균등의 {100 * (gap - rec) / gap:.1f}%).")
    L.append("\n")

    L.append("## Followup 성능과 I/O\n")
    L.append(md_table(["Arm", "TTFT mean / p50 / p95 ms", "E2E mean / p95 ms", "SSD MB", "읽기 호출 (pread / preadv)", "GPU cache KV MB", "peak GPU alloc MB (followup 평균 / 전체 최대)"],
                      [[x["name"], f"{x['ttft_mean_ms']:.3f} / {x['ttft_p50_ms']:.3f} / {x['ttft_p95_ms']:.3f}", f"{x['e2e_mean_ms']:.3f} / {x['e2e_p95_ms']:.3f}",
                        f"{x['ssd_mb_per_followup']:.3f}", f"{x['pread_per_followup']:.0f} / {x['preadv_per_followup']:.0f}", f"{x['gpu_cache_kv_mb_per_followup']:.3f}",
                        f"{x['peak_gpu_allocated_followup_mean_mb']:.1f} / {x['peak_gpu_allocated_max_mb']:.1f}"] for x in latency]))

    def sg(m, key):
        v = lat[m].get(f"seg_{key}_mean_ms")
        return "–" if v is None else f"{v:.3f}"
    L.append("구간별 평균 ms (followup; 구간은 서로 겹칠 수 있어 합산하지 않는다. `읽기 포함`은 계획 + 읽기 + (B/FullLoad는) CPU 조립):\n")
    L.append(md_table(["Arm", "prompt 준비", "읽기 포함", "raw read", "계획", "H2D·조립", "hook 설치", "prefill", "decode"],
                      [[LABEL[m], sg(m, "prompt_preparation"), sg(m, "store_load_inclusive"), sg(m, "raw_pread"),
                        sg(m, "layerwise_planning") if m in PACKED else sg(m, "fixed_first_k_planning"), sg(m, "h2d_and_assembly"), sg(m, "mask_hook_install"),
                        sg(m, "suffix_prefill") if m != "recompute" else sg(m, "multimodal_prefill"), sg(m, "decode")] for m in METHODS]))
    pk = pair(U_, B, "followup", "ttft_ms")
    L.append(f"**packed 효과** (같은 KV, 저장 형식만 다름): U_packed − B의 TTFT = {pk['delta']:+.3f} ms [{pk['ci_lo']:+.3f}, {pk['ci_hi']:+.3f}] "
             f"({lat[U_]['ttft_mean_ms'] / lat[B]['ttft_mean_ms']:.3f}×; p50 {lat[U_]['ttft_p50_ms']:.3f} vs {lat[B]['ttft_p50_ms']:.3f}, p95 {lat[U_]['ttft_p95_ms']:.3f} vs {lat[B]['ttft_p95_ms']:.3f}). "
             f"읽기 호출은 57회 pread → 1회 preadv, 읽은 바이트는 같다({lat[U_]['ssd_mb_per_followup']:.3f} MB). "
             f"packed 경로는 buffer를 한 번 H2D한 뒤 GPU에서 조립하므로 B(CPU 조립 후 layer·종류별 56회 H2D)와 조립 방식도 다르다 — 이 차이는 저장 형식 하나의 효과가 아니라 새 hit 경로 전체의 효과다.\n")
    pt = pair(P, U_, "followup", "ttft_ms")
    L.append(f"PYR-b20 − U_packed의 TTFT = {pt['delta']:+.3f} ms [{pt['ci_lo']:+.3f}, {pt['ci_hi']:+.3f}]. 세 packed arm의 SSD 바이트와 GPU cache KV 바이트는 요청마다 서로 같고 B와도 같다"
             f"({'확인' if gpu_equal else '불일치'}; Σ k_l = 28·ceil(N/4)). DONTNEED는 OS page-cache hint이며 NAND cold나 O_DIRECT가 아니다. 절대 TTFT는 다른 실행과 비교하지 않는다.\n")

    L.append("## 저장과 setup (이미지당 1회)\n")
    L.append(md_table(["Arm", "파일 수", "기록 MB", "outer persistence mean / p50 / p95 ms", "writer ms", "write / fsync ms", "activation mean / p50 / p95 ms"],
                      [[s["name"], s["files_per_store"], f"{s['write_mb_mean']:.3f}", f"{s['outer_persistence_mean_ms']:.3f} / {s['outer_persistence_p50_ms']:.3f} / {s['outer_persistence_p95_ms']:.3f}",
                        f"{s['native_writer_mean_ms']:.3f}", f"{s['ssd_write_mean_ms']:.3f} / {s['fsync_mean_ms']:.3f}",
                        f"{s['activation_mean_ms']:.3f} / {s['activation_p50_ms']:.3f} / {s['activation_p95_ms']:.3f}"] for s in storage]))
    L.append("persistence는 writer wall + completion seal(기존 B와 같은 경계), activation은 메타 검증 + payload SHA256 전체 검증 + (packed) pinned buffer 할당이다. 파일 수는 payload + meta.json + completion.json.\n")

    L.append("## 3-turn session (shared-stream / standalone-equivalent)\n")
    L.append(md_table(["Arm", "setup ms/image", "shared-stream mean / p50 / p95 ms", "standalone-equivalent (DERIVED) mean / p50 / p95 ms"],
                      [[sess[m]["name"], f(sess[m]["setup_ms_per_image"]),
                        f"{sess[m]['shared_stream_session_E2E_mean_ms']:.3f} / {sess[m]['shared_stream_session_E2E_p50_ms']:.3f} / {sess[m]['shared_stream_session_E2E_p95_ms']:.3f}",
                        f"{sess[m]['derived_standalone_E2E_mean_ms']:.3f} / {sess[m]['derived_standalone_E2E_p50_ms']:.3f} / {sess[m]['derived_standalone_E2E_p95_ms']:.3f}"] for m in METHODS]))
    L.append(md_table(["A − B", "accounting", "Δ ms [95% CI]"], [[f"{s['a_name']} − {s['b_name']}", s["accounting"], f"{s['delta_ms']:+.3f} [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]"]
                                                               for s in sess_pairs if s["b_name"] in ("ReComp", LABEL[B])]))
    L.append("shared-stream은 이미지 setup을 source dialogue에 한 번 부과한 실측 합이다. standalone-equivalent는 같은 실측 setup을 dialogue마다 다시 부과한 **DERIVED** 값이며 "
             "non-source T1에 capture hook을 걸었을 때의 증분은 NOT_MEASURED이다.\n")

    L.append("## 질문 유형별 Δ (followup, LLaVA 보고서와 같은 regex 분류; 탐색적 수치)\n")
    L.append(md_table(["유형", "n", "U %", "PYR-b20 %", "PYR-b2p5 %", "ReComp %", "PYR-b20 − U [95% CI]", "PYR-b2p5 − U [95% CI]"],
                      [[e["question_type"], e["n_followup"], f(e["acc_U_packed_pct"], 2), f(e["acc_PYR-b20_packed_pct"], 2),
                        f(e["acc_PYR-b2p5_packed_pct"], 2), f(e["acc_ReComp_pct"], 2),
                        f"{e['delta_PYR-b20_packed_minus_U_pp']:+.3f} [{e['ci_lo_PYR-b20_packed']:+.3f}, {e['ci_hi_PYR-b20_packed']:+.3f}]",
                        f"{e['delta_PYR-b2p5_packed_minus_U_pp']:+.3f} [{e['ci_lo_PYR-b2p5_packed']:+.3f}, {e['ci_hi_PYR-b2p5_packed']:+.3f}]"] for e in qt]))
    L.append("유형별 수치는 사전 등록한 가설 검정이 아니다. 다중 비교 보정을 하지 않았고 표본이 작은 유형은 CI가 넓다.\n")

    L.append("## layer별 k_l·바이트 (세 벡터)\n")
    L.append(md_table(["N", "균등 k", "PYR-b20 첫 / 마지막", "PYR-b2p5 첫 / 마지막", "Σ k_l (세 벡터 동일)", "payload MB (구조 포함, 세 벡터·B 동일)"],
                      [[n, (n + 3) // 4, f"{ktab['k_per_layer_by_N'][str(n)]['PYR-b20'][0]} / {ktab['k_per_layer_by_N'][str(n)]['PYR-b20'][-1]}",
                        f"{ktab['k_per_layer_by_N'][str(n)]['PYR-b2p5'][0]} / {ktab['k_per_layer_by_N'][str(n)]['PYR-b2p5'][-1]}", 28 * ((n + 3) // 4),
                        f"{((n + 3) // 4 + 21) * 57344 / 1e6:.3f}"] for n in (260, 345, 529)]))
    L.append(f"HOLD 398장의 N은 {ktab['N_min']} / {ktab['N_median_low_actual_image']} / {ktab['N_max']} (최소 / 실제 이미지 중간값 / 최대), 서로 다른 N {ktab['distinct_N']}개. "
             "layer 28개 전체의 비율·k_l·바이트는 [layer_k_bytes.csv](layer_k_bytes.csv), 그림은 [fig_vectors_qwen.png](fig_vectors_qwen.png). "
             "k_l = 0인 layer는 없다(최소 PYR-b20 3행).\n")

    L.append("## 정합성 사후 확인 — U_packed ≡ B\n")
    L.append(f"U_packed와 기존 B는 같은 KV를 쓰므로 예측이 토큰 단위로 같아야 한다. main 12,183 요청 중 **{integ['token_exact']:,}건 일치 ({100 * integ['token_exact_rate']:.3f}%)**, 불일치 {integ['mismatches']}건.\n")
    if integ["mismatches"]:
        L.append(f"불일치 {integ['mismatches']}건 중 같은 history에서 처음 갈라진 요청은 {integ['first_divergence_points_same_history']}건(turn별 {integ['first_divergence_by_turn']}), "
                 f"나머지 {integ['propagated_through_history']}건은 앞 turn의 답이 달라져 history가 달라진 뒤의 요청이다. 전체 덤프는 [u_packed_vs_B_integrity.json](u_packed_vs_B_integrity.json).\n")
    else:
        L.append("gate에서도 같은 capture로 만든 두 store의 payload·조립 텐서·28 layer SDPA 입력·첫 logits이 bitwise 같았고(QG2), integration에서 scored arm 4개의 T1 capture가 비트 단위로 같았다.\n")

    L.append("## 결정성 참고 (10/02 Qwen 본실험의 같은 요청과 비교)\n")
    if det["status"] == "MEASURED":
        L.append(md_table(["Arm", "요청", "토큰 일치", "예측 일치", "followup 토큰 일치"],
                          [[LABEL[m], x["n"], f"{100 * x['token_exact_rate']:.3f}%", f"{100 * x['prediction_agreement']:.3f}%", f"{100 * x['followup_token_exact_rate']:.3f}%"]
                           for m, x in det["methods"].items()]))
        L.append("참고 수치일 뿐이다. 절대 TTFT는 실행 간 비교하지 않는다.\n")
    else:
        L.append("NOT_MEASURED (부모 run의 raw가 로컬에 없음).\n")

    L.append("## 실행 기록\n")
    hours = (finished - started) / 3600
    L.append(f"- 실제 요청 수: main **{audit['main_requests']:,} / 73,098** (398 images, 4,061 dialogues × 3 turns × 6 arms), smoke {audit['phase_counts']['smoke']}, integration {audit['phase_counts']['integration']}.\n"
             f"- 실패·스킵: {audit['failed_or_skipped_requests']}. 실패한 image attempt 기록: {failed_attempts if failed_attempts else '없음'}. 중단 기록: {interruptions if interruptions else '없음'}.\n"
             f"- 소요 시간: main 첫 시작부터 종료까지 {hours:.2f}시간 (계약의 예상 2.7–3.4시간).\n"
             f"- 독립 감사: {audit['status']} ({audit['checks']:,} checks). Artifact protection: {protection['status']} "
             f"(SHA {protection['files_before']} files, kvstore inventory {protection['kvstore_files_before']}, 다른 run inventory {protection['runs_files_before']}). "
             f"삭제: run-local scratch {dele['removed_scratch_files']} files / {dele['image_bundles']} image bundles, gate store {dele['removed_gate_store_files']} files — 기존 산출물 삭제 없음.\n"
             f"- freeze amendment: {', '.join(protection['amendment_files']) if protection['amendment_files'] else '없음'}. 기존 소스 수정 없음(새 파일만 추가).\n")
    L.append("### 프롬프트와 다르게 한 점\n")
    L.append("1. 설계 문서를 디스크에 쓰기 전에 설계 A의 기능 점검을 이미지 1장으로 한 번 실행했다(판정·수치에 쓰지 않음). 'GPU 측정 전에 문서로 남긴다'의 순서를 이 1회만큼 지키지 못했다.\n"
             "2. freeze 전에 별도 run root(`runs/qwen_layer_budget_pyr_dev_20261111T0900Z`)에서 gate·integration·smoke를 한 번 리허설했다(정합성 fixture만 사용, 개발 기록으로 보존). "
             "공식 gate·integration·smoke는 freeze 뒤 이 run에서 처음부터 다시 실행했고 수정 없이 통과했다.\n"
             "3. packed hit의 조립은 '파일 전체를 pinned buffer로 preadv → buffer 1회 H2D → GPU에서 행 index 1회'로 구현했다. 결과 텐서는 기존 `load_prefix`와 bitwise 같지만(QG2) "
             "조립 연산 위치(CPU → GPU)가 B와 다르다. 그래서 'U_packed vs B'의 TTFT 차이는 읽기 호출 수뿐 아니라 조립 방식의 차이도 포함한다.\n"
             "4. 기존 CPU 테스트 중 GLM-4.6V 3건은 이 변경 전부터 실패하므로 분리 보고하고 PASS 기준에서 뺐다(계약 8절).\n"
             "5. 보고 스크립트(171)는 측정 뒤 실행하는 분석 코드라서 freeze 대상 runtime source에서 제외했다. 결과를 본 뒤 음수 Δ에 맞게 문구 세 곳(판정 표의 CI 행, 판정 설명 한 줄, 손실 회복 문장)과 유형 표의 강조를 고쳐 보고서를 다시 생성했다 — 수치·판정·감사 결과는 첫 생성과 같다. 그림 색 팔레트의 자동 검증 스크립트는 node가 없어 실행하지 못했다(선 모양·marker·직접 표기를 함께 썼다).\n"
             "6. QG2·QG4는 요구보다 넓게 했다(QG2에 28 layer SDPA 입력 비교 추가, QG4를 U에도 적용).\n")
    L.append("### NOT_MEASURED\n")
    L.append("- PyramidKV의 attention 기반 token 선택(여기서는 layer별 자르는 지점만 사용).\n- H2D 시간 단독(조립과 함께 측정), NAND/controller cold, O_DIRECT.\n"
             "- standalone session의 non-source T1 capture 증분(DERIVED 값에 포함되지 않음).\n- 동시 요청 throughput·메모리 포화.\n- MT-VQA, GLM, LLaVA (이번 범위 아님).\n")
    (res / "REPORT.md").write_text("\n".join(L))

    env_lines = " \\\n       ".join(Q.ENV.split(" "))
    (res / "REPRODUCE.md").write_text(f"""# 재현 절차

```bash
cd {ROOT}
export {env_lines}
PY={Q.PY}
RUN={run.relative_to(ROOT)}
$PY scripts/167_freeze_qwen_layer_budget_pyr.py --stage vectors                # 새 run root, 벡터 3개, 검산, k_l 표
#   design_ragged_cache.md 를 run root에 두고 docs/qwen_layer_budget_pyr_contract.md 를 확정한 뒤:
$PY scripts/167_freeze_qwen_layer_budget_pyr.py --stage freeze --run $RUN       # storage plan, protection before, config/manifest
$PY scripts/168_validate_qwen_layerwise_packed.py --run $RUN --stage cpu        # QG1 + 기존 CPU 테스트
$PY scripts/168_validate_qwen_layerwise_packed.py --run $RUN --stage gpu        # QG2-QG6 (core pair 10개)
$PY scripts/169_eval_qwen_layer_budget_pyr.py --run $RUN --phase integration    # QG7
$PY scripts/169_eval_qwen_layer_budget_pyr.py --run $RUN --phase all            # smoke 72 -> main 73,098 (중단 시 --resume)
$PY scripts/171_report_qwen_layer_budget_pyr.py --run $RUN                      # 독립 감사(170), 판정, protection after, REPORT
```

고정값: Qwen/Qwen2.5-VL-7B-Instruct `cc594898137f460bfe9f0759e9844b3ce807cfb5`, NF4 double / BF16 / SDPA, seed 1234, greedy max 16, batch 1, CPU threads 2,
transformers {config['environment']['transformers']}, torch {config['environment']['torch']}. 벡터 SHA256은 `vectors/vectors_sha256.json`, 소스 SHA256은 `config.json`의 `source_sha256`.
절대 TTFT는 실행마다 달라지므로 같은 실행 안의 arm끼리만 비교한다.
""")
    # the runner leaves CURRENT_STATUS at AWAITING_INDEPENDENT_FINAL_AUDIT; record the terminal state once the audit and the report exist
    terminal = {"execution_state": "FINISHED" if final_ok else "AUDIT_FAILED", "decision": decision_value, "main_final_requests": audit["main_requests"],
                "expected_main_requests": 73098, "failed_or_skipped_requests": audit["failed_or_skipped_requests"], "independent_audit": audit["status"],
                "audit_sha256": sha(final / "independent_audit.json"), "protection": protection["status"], "deletion_log": dele["status"],
                "runner_last_status": read(run / "CURRENT_STATUS.json") if read(run / "CURRENT_STATUS.json").get("execution_state") != "FINISHED" and "runner_last_status" not in read(run / "CURRENT_STATUS.json") else read(run / "CURRENT_STATUS.json").get("runner_last_status"),
                "at_unix": time.time()}
    for target in (run / "CURRENT_STATUS.json", res / "CURRENT_STATUS.json"):
        write(target, terminal, exclusive=False)
    print(json.dumps({"decision": decision_value, "delta_pp": primary["delta_followup_pp"], "ci": primary["ci95_pp"], "ttft_ratio": primary["ttft_ratio"],
                      "integrity": integ["status"], "token_exact": integ["token_exact"], "audit": audit["status"], "protection": protection["status"],
                      "deletion": dele["status"], "main_hours": hours}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

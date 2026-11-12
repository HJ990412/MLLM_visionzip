#!/usr/bin/env python3
"""Final audit, curve and report for the Qwen2.5-VL retention-ratio sweep (25/30/40/50/100 %).

Runs the independent auditor (175) into results/<run>/final, then computes the pre-registered
curve table, the two descriptive baselines, the U100_packed-vs-FullLoad integrity check, the
determinism reference against the PyramidKV run, protection after, the deletion log, the two
trade-off figures and REPORT.md.  There is no verdict.  Post-hoc analysis only: nothing here
feeds back into a measured request.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import qwen_sweep_support as Q
from qwen_sweep_support import ALLOCS, LABEL, METHODS, PACKED, ROOT, U25, U100, load_script, read, sha, write
from mmimpress.recompute_kv25 import paired_bootstrap, sha_manifest, stat_inventory

SEED = 1234
ROW_ALL = 28 * 2 * 4 * 128 * 2
INT = {"turn_id", "ssd_read_bytes", "ssd_preads", "ssd_pread_calls", "ssd_preadv_calls", "N_content", "S_structural", "k", "gpu_cache_kv_bytes",
       "h2d_kv_bytes", "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes", "generated_token_count", "visual_read_bytes", "structural_read_bytes"}
FLOAT = {"score", "ttft_ms", "request_e2e_ms", "core_ttft_ms", "retention_ratio"}
FOLLOW = lambda k, v: k[1] > 1
ALL = lambda k, v: True
SCOPES = (("followup", FOLLOW), ("all", ALL), ("T1", lambda k, v: k[1] == 1), ("T2", lambda k, v: k[1] == 2), ("T3", lambda k, v: k[1] == 3))
SEG = ("prompt_preparation", "store_load_inclusive", "raw_pread", "fixed_first_k_planning", "layerwise_planning", "h2d_and_assembly",
       "mask_hook_install", "suffix_prefill", "multimodal_prefill", "decode")
PYR_METHOD = {"recompute": "recompute", "fullload": "fullload", U25: "qwen_ours_prefixstore_kv25_U_packed"}


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
    """Image-cluster paired bootstrap of mean(a) - mean(b); b=None gives the CI of a alone."""
    acc = {img: [0.0, 0.0, 0] for img in images}
    for k, v in by[a].items():
        if flt(k, v):
            e = acc[img_of[k[0]]]
            e[0] += v[key]
            e[1] += by[b][k][key] if b is not None else 0.0
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


def f(x, n=3):
    return "–" if x is None else f"{x:.{n}f}"


def pm(d, n=3):
    return f"{d['delta']:+.{n}f} [{d['ci_lo']:+.{n}f}, {d['ci_hi']:+.{n}f}]"


# ------------------------------------------------------------ integrity / determinism
def integrity(by, res):
    """U100_packed and FullLoad serve the same KV: all 12,183 requests must agree token by token."""
    keys = sorted(by["fullload"])
    mism = []
    for k in keys:
        a, b = by["fullload"][k], by[U100][k]
        if a["generated_token_ids"] != b["generated_token_ids"]:
            mism.append({"dialog_id": k[0], "turn_id": k[1], "image_id": a["image_id"], "same_history": a["history_sha256"] == b["history_sha256"],
                         "fullload_tokens": json.loads(a["generated_token_ids"]), "U100_packed_tokens": json.loads(b["generated_token_ids"]),
                         "fullload_prediction": a["prediction"], "U100_packed_prediction": b["prediction"], "fullload_score": a["score"], "U100_packed_score": b["score"],
                         "first_token_equal": a["first_token_id"] == b["first_token_id"]})
    first = [m for m in mism if m["same_history"]]
    doc = {"definition": "generated_token_ids of fullload (57 files, 57 pread, CPU assembly) vs qwen_ours_kv_U100_packed (1 file, 1 preadv, GPU assembly), every (dialogue, turn) of the main",
           "requests": len(keys), "token_exact": len(keys) - len(mism), "token_exact_rate": (len(keys) - len(mism)) / len(keys),
           "mismatches": len(mism), "first_divergence_points_same_history": len(first),
           "first_divergence_by_turn": dict(Counter(m["turn_id"] for m in first)),
           "propagated_through_history": len(mism) - len(first), "status": "PASS" if not mism else "PENDING_REVIEW", "mismatch_dump": mism}
    write(res / "u100_packed_vs_fullload_integrity.json", doc, exclusive=False)
    return doc


def determinism_vs_pyramidkv(by, res):
    """Reference only: the same requests of the PyramidKV run (recompute, fullload, U_packed)."""
    run = Q.PYR_RUN
    commits = sorted((run / "main/images").glob("*/COMMITTED.json"))
    if len(commits) != 398:
        doc = {"status": "NOT_MEASURED", "reason": "PyramidKV-run main raw not available locally"}
        write(res / "determinism_vs_pyramidkv_run.json", doc, exclusive=False)
        return doc
    rev = {v: k for k, v in PYR_METHOD.items()}
    needles = [f'"method_id":"{m}"' for m in rev]          # cheap pre-filter on the compact JSON lines; the parsed method_id decides
    agree = {m: Counter() for m in PYR_METHOD}
    for cp in commits:
        com = json.loads(cp.read_text())
        with (cp.parent / com["attempt_id"] / "raw.jsonl").open() as fh:
            for line in fh:
                if not any(needle in line for needle in needles):
                    continue
                r = json.loads(line)
                if r["method_id"] not in rev:
                    continue
                m = rev[r["method_id"]]
                cur = by[m].get((r["dialog_id"], r["turn_id"]))
                if cur is None:
                    agree[m]["missing"] += 1
                    continue
                c = agree[m]
                same = json.loads(cur["generated_token_ids"]) == r["result"]["generated_token_ids"]
                c["n"] += 1
                c["token_exact"] += same
                c["prediction_equal"] += cur["prediction"] == r["prediction"]
                c["score_equal"] += cur["score"] == r["score"]
                if r["turn_id"] > 1:
                    c["followup_n"] += 1
                    c["followup_token_exact"] += same
    if any(c["n"] == 0 or c["followup_n"] == 0 for c in agree.values()):
        doc = {"status": "NOT_MEASURED", "reason": "no matching request rows in the PyramidKV-run raw", "counts": {m: dict(c) for m, c in agree.items()}}
        write(res / "determinism_vs_pyramidkv_run.json", doc, exclusive=False)
        return doc
    doc = {"status": "MEASURED", "pyramidkv_run": str(run.relative_to(Q.ROOT)), "method_map": PYR_METHOD,
           "scope": "reference only; absolute TTFT is never compared across runs",
           "methods": {m: {**dict(c), "token_exact_rate": c["token_exact"] / c["n"], "prediction_agreement": c["prediction_equal"] / c["n"],
                           "followup_token_exact_rate": c["followup_token_exact"] / c["followup_n"]} for m, c in agree.items()}}
    write(res / "determinism_vs_pyramidkv_run.json", doc, exclusive=False)
    return doc


# ----------------------------------------------------------------------- figures
def figures(res, curve):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"            # reference palette slots 1-3 (validate all-pairs)
    packed = [c for c in curve if c["method"] in PACKED]
    others = {c["method"]: c for c in curve if c["method"] not in PACKED}
    for xkey, xlabel, name, title in (("ttft_mean_ms", "follow-up TTFT, mean (ms)", "fig_tradeoff_qwen.png", "Qwen2.5-VL: follow-up accuracy vs TTFT by retention ratio (MT-GQA HOLD)"),
                                      ("ssd_mb_per_followup", "SSD bytes read per follow-up (MB)", "fig_tradeoff_qwen_ssd.png", "Qwen2.5-VL: follow-up accuracy vs SSD bytes read by retention ratio (MT-GQA HOLD)")):
        fig, ax = plt.subplots(figsize=(8.4, 4.8), facecolor=surface)
        ax.set_facecolor(surface)
        xs = [c[xkey] for c in packed]
        ys = [c["followup_pct"] for c in packed]
        err = [[c["followup_pct"] - c["followup_ci_lo"] for c in packed], [c["followup_ci_hi"] - c["followup_pct"] for c in packed]]
        ax.plot(xs, ys, color=blue, lw=2, zorder=2)
        ax.errorbar(xs, ys, yerr=err, fmt="o", color=blue, ms=7, mec=surface, mew=1.5, ecolor=blue, elinewidth=1.2, capsize=3, zorder=3,
                    label="packed, uniform ratio (1 file, 1 preadv)")
        span = max(c[xkey] for c in curve) - min(c[xkey] for c in curve)
        offsets = [((-7, -14), "right"), ((-7, 8), "right"), ((6, -14), "left"), ((8, -16), "left"), ((9, -4), "left")]      # neighbours alternate sides so close points stay legible
        for c, (xy, ha) in zip(packed, offsets):
            ax.annotate(f"{round(100 * c['ratio'])}%", (c[xkey], c["followup_pct"]), textcoords="offset points", xytext=xy, ha=ha, fontsize=9, color=ink)
        for m, color, marker, label in (("fullload", orange, "s", "FullLoad (57 files, 100%)"), ("recompute", aqua, "D", "ReComp (no cache)")):
            c = others[m]
            ax.errorbar([c[xkey]], [c["followup_pct"]], yerr=[[c["followup_pct"] - c["followup_ci_lo"]], [c["followup_ci_hi"] - c["followup_pct"]]],
                        fmt=marker, color=color, ms=7, mec=surface, mew=1.5, ecolor=color, elinewidth=1.2, capsize=3, zorder=3, label=label)
            ax.annotate(LABEL[m], (c[xkey], c["followup_pct"]), textcoords="offset points", xytext=(7, -13) if (m == "fullload" or xkey != "ttft_mean_ms") else (-8, -14),
                        ha="left" if (m == "fullload" or xkey != "ttft_mean_ms") else "right", fontsize=9, color=ink)
        ax.set_xlabel(xlabel, color=ink)
        ax.set_ylabel("follow-up (T2–T3) accuracy, %  (95% CI, image bootstrap)", color=ink)
        ax.set_title(title, color=ink, fontsize=11, loc="left")
        ax.set_xlim(min(0 if xkey != "ttft_mean_ms" else min(c[xkey] for c in curve) - 0.08 * span, min(c[xkey] for c in curve) - 0.08 * span), max(c[xkey] for c in curve) + 0.10 * span)
        ax.grid(color=grid, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.tick_params(colors=muted, labelsize=8)
        leg = ax.legend(frameon=False, fontsize=8, loc="lower right")
        for t in leg.get_texts():
            t.set_color(ink)
        fig.tight_layout()
        fig.savefig(res / name, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)


# ------------------------------------------------------------ protection / deletion
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


# -------------------------------------------------------------------------- main
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
        A = load_script("_qsweep_final_auditor", Q.AUDIT_SCRIPT)
        A.run_audit(run, final)
    audit = read(final / "independent_audit.json")
    rows = load_rows(final)
    by = defaultdict(dict)
    for r in rows:
        by[r["method_id"]][(r["dialog_id"], r["turn_id"])] = r
    images = sorted({r["image_id"] for r in rows})
    img_of = {r["dialog_id"]: r["image_id"] for r in rows}
    E = Q.EXPECT
    assert audit["status"] == "PASS" and audit["main_requests"] == E["main"] and len(images) == E["images"] and all(len(by[m]) == E["per_method"] for m in METHODS)

    # ---------------------------------------------------------------- quality
    quality = []
    for m in METHODS:
        d = by[m]
        turns = {t: [v["score"] for k, v in d.items() if k[1] == t] for t in (1, 2, 3)}
        follow = turns[2] + turns[3]
        allv = turns[1] + follow
        ci = paired(by, m, None, images, img_of, FOLLOW)
        quality.append({"method": m, "name": LABEL[m], "ratio": Q.RATIO[PACKED[m]] if m in PACKED else 1.0,
                        "T1_pct": 100 * np.mean(turns[1]), "T2_pct": 100 * np.mean(turns[2]), "T3_pct": 100 * np.mean(turns[3]),
                        "all_pct": 100 * np.mean(allv), "all_correct": int(sum(allv)), "all_n": len(allv),
                        "followup_pct": 100 * np.mean(follow), "followup_correct": int(sum(follow)), "followup_n": len(follow),
                        "followup_ci_lo": ci["ci_lo"], "followup_ci_hi": ci["ci_hi"]})
    write_csv(res / "quality.csv", quality)
    qual = {q["method"]: q for q in quality}

    # ------------------------------------------------------- paired analysis
    pairs = []
    packed = list(PACKED)
    plan = [(m, base) for base in ("recompute", "fullload") for m in packed] + [(m, U25) for m in packed[1:]] \
        + [(packed[i + 1], packed[i]) for i in range(1, len(packed) - 1)] + [("fullload", "recompute")]
    for a_, b_ in plan:
        for scope, flt in SCOPES:
            d = paired(by, a_, b_, images, img_of, flt)
            c, agree = cells(by, a_, b_, flt)
            pairs.append({**d, "a_name": LABEL[a_], "b_name": LABEL[b_], "scope": scope, "metric": "accuracy_pp", **c, "prediction_agreement": agree})
    for a_, b_ in [(m, base) for base in ("recompute", "fullload") for m in packed] + [(m, U25) for m in packed[1:]] + [("fullload", "recompute")]:
        d = paired(by, a_, b_, images, img_of, FOLLOW, key="ttft_ms", scale=1.0)
        pairs.append({**d, "a_name": LABEL[a_], "b_name": LABEL[b_], "scope": "followup", "metric": "ttft_ms"})
    write_csv(res / "paired_analysis.csv", pairs)
    write(res / "paired_analysis.json", pairs, exclusive=False)

    def pair(a_, b_, scope="followup", metric="accuracy_pp"):
        return next(p for p in pairs if p["a"] == a_ and p["b"] == b_ and p["scope"] == scope and p["metric"] == metric)

    # --------------------------------------------------------------- latency
    latency = []
    for m in METHODS:
        hit = [v for k, v in by[m].items() if k[1] > 1]
        one = [v for k, v in by[m].items() if k[1] == 1]
        t, e = stats([r["ttft_ms"] for r in hit]), stats([r["request_e2e_ms"] for r in hit])
        row = {"method": m, "name": LABEL[m], "ratio": qual[m]["ratio"], "followups": len(hit), "ttft_mean_ms": t["mean"], "ttft_p50_ms": t["p50"], "ttft_p95_ms": t["p95"],
               "e2e_mean_ms": e["mean"], "e2e_p95_ms": e["p95"], "T1_ttft_mean_ms": float(np.mean([r["ttft_ms"] for r in one])),
               "ssd_mb_per_followup": float(np.mean([r["ssd_read_bytes"] for r in hit])) / 1e6, "read_calls_per_followup": float(np.mean([r["ssd_preads"] for r in hit])),
               "pread_per_followup": float(np.mean([r["ssd_pread_calls"] for r in hit])), "preadv_per_followup": float(np.mean([r["ssd_preadv_calls"] for r in hit])),
               "gpu_cache_kv_mb_per_followup": float(np.mean([r["gpu_cache_kv_bytes"] for r in hit])) / 1e6,
               "k_mean_per_followup": float(np.mean([r["k"] for r in hit])),
               "peak_gpu_allocated_max_mb": max(r["peak_gpu_allocated_bytes"] for r in hit + one) / 1e6,
               "peak_gpu_allocated_followup_mean_mb": float(np.mean([r["peak_gpu_allocated_bytes"] for r in hit])) / 1e6}
        for s in SEG:
            vals = [r["t_" + s] for r in hit if r["t_" + s] is not None]
            row["seg_" + s + "_mean_ms"] = float(np.mean(vals)) if vals else None
        latency.append(row)
    lat = {x["method"]: x for x in latency}
    for x in latency:
        x["ttft_ratio_vs_ReComp"] = x["ttft_mean_ms"] / lat["recompute"]["ttft_mean_ms"]
        x["ttft_reduction_vs_ReComp_pct"] = 100 * (1 - x["ttft_ratio_vs_ReComp"])
        x["ttft_ratio_vs_FullLoad"] = x["ttft_mean_ms"] / lat["fullload"]["ttft_mean_ms"]
        x["ttft_reduction_vs_FullLoad_pct"] = 100 * (1 - x["ttft_ratio_vs_FullLoad"])
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
                        "activation_p95_ms": act["p95"], "files_per_store": 59 if m == "fullload" else 3})
    write_csv(res / "storage.csv", storage)
    sto = {s["method"]: s for s in storage}
    sess_rows = load_csv(final / "session_summary.csv")
    per_dialogue = load_csv(final / "session_per_dialogue.csv")
    sby = defaultdict(dict)
    for r in per_dialogue:
        sby[r["method_id"]][(r["dialog_id"], 0)] = {"shared": float(r["shared_stream_session_E2E_ms"]), "standalone": float(r["derived_standalone_E2E_ms"])}
    sess_pairs = []
    for m in packed:
        for base in ("recompute", "fullload"):
            for key in ("shared", "standalone"):
                d = paired(sby, m, base, images, img_of, ALL, key=key, scale=1.0)
                sess_pairs.append({"a_name": LABEL[m], "b_name": LABEL[base], "accounting": "shared_stream_session_ms" if key == "shared" else "derived_standalone_session_ms",
                                   "delta_ms": d["delta"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"], "dialogues": d["n"]})
    sessions = [{"method": r["method_id"], "name": r["label"], **{k: float(r[k]) for k in r if k.endswith("_ms") or k.endswith("_image")}} for r in sess_rows]
    write_csv(res / "sessions.csv", sessions)
    write(res / "sessions_paired.json", sess_pairs, exclusive=False)
    sess = {s["method"]: s for s in sessions}

    # ------------------------------------------------ integrity, determinism
    integ = integrity(by, res)
    det = determinism_vs_pyramidkv(by, res)

    # ------------------------------------------------------- curve, baselines
    ktab = read(run / "phase0" / "k_tables.json")
    n_med = ktab["N_median_low_actual_image"]
    curve = []
    for m in METHODS:
        alloc = PACKED.get(m)
        entry = {"method": m, "name": LABEL[m], "ratio": qual[m]["ratio"], "packed": m in PACKED,
                 "k_per_layer_median_image": ktab["k_by_N"][str(n_med)][alloc] if alloc else n_med,
                 "k_per_layer_min_max_image": [ktab["k_by_N"][str(ktab["N_min"])][alloc], ktab["k_by_N"][str(ktab["N_max"])][alloc]] if alloc else [ktab["N_min"], ktab["N_max"]],
                 **{k: qual[m][k] for k in ("T1_pct", "T2_pct", "T3_pct", "all_pct", "followup_pct", "followup_correct", "followup_n", "followup_ci_lo", "followup_ci_hi")},
                 **{k: lat[m][k] for k in ("ttft_mean_ms", "ttft_p50_ms", "ttft_p95_ms", "e2e_mean_ms", "ttft_ratio_vs_ReComp", "ttft_reduction_vs_ReComp_pct", "ttft_ratio_vs_FullLoad",
                                           "ttft_reduction_vs_FullLoad_pct", "ssd_mb_per_followup", "read_calls_per_followup", "gpu_cache_kv_mb_per_followup")}}
        for base, tag in (("recompute", "vs_ReComp"), ("fullload", "vs_FullLoad"), (U25, "vs_U25")):
            if m != base and (m in PACKED or (m, base) == ("fullload", "recompute")):
                p = pair(m, base)
                entry[f"delta_{tag}_pp"] = p["delta"]
                entry[f"ci_{tag}_pp"] = [p["ci_lo"], p["ci_hi"]]
                entry[f"ci_{tag}_includes_zero"] = not p["ci_excludes_zero"]
        if m in sto:
            entry.update(persistence_ms_per_image=sto[m]["outer_persistence_mean_ms"], store_mb_per_image=sto[m]["write_mb_mean"], activation_ms_per_image=sto[m]["activation_mean_ms"])
        entry.update(shared_stream_session_ms=sess[m]["shared_stream_session_E2E_mean_ms"], standalone_equivalent_session_ms_DERIVED=sess[m]["derived_standalone_E2E_mean_ms"],
                     setup_ms_per_image=sess[m]["setup_ms_per_image"])
        curve.append(entry)
    cur = {c["method"]: c for c in curve}
    base_a = next((m for m in packed if cur[m]["ci_vs_ReComp_includes_zero"]), None)
    base_b = next((m for m in packed if -cur[m]["delta_vs_ReComp_pp"] <= Q.LLAVA_U25_LOSS_PP), None)
    steps = []
    for lo_m, hi_m in zip(packed, packed[1:]):
        p = pair(hi_m, lo_m) if lo_m != U25 else pair(hi_m, U25)
        steps.append({"from": LABEL[lo_m], "to": LABEL[hi_m], "delta_followup_pp": p["delta"], "ci95_pp": [p["ci_lo"], p["ci_hi"]], "ci_excludes_zero": p["ci_excludes_zero"],
                      "accuracy_drops_as_ratio_grows": p["delta"] < 0})
    baselines = {
        "a_loss_not_detected": {"rule": config["report_rule"]["baseline_a_loss_not_detected"], "result": LABEL[base_a] if base_a else "NONE",
                                "ratio": cur[base_a]["ratio"] if base_a else None,
                                "evidence": {LABEL[m]: {"delta_vs_ReComp_pp": cur[m]["delta_vs_ReComp_pp"], "ci95_pp": cur[m]["ci_vs_ReComp_pp"], "ci_includes_zero": cur[m]["ci_vs_ReComp_includes_zero"]} for m in packed}},
        "b_llava_25_level": {"rule": config["report_rule"]["baseline_b_llava_25_level"], "threshold_loss_pp": Q.LLAVA_U25_LOSS_PP, "result": LABEL[base_b] if base_b else "NONE",
                             "ratio": cur[base_b]["ratio"] if base_b else None,
                             "evidence": {LABEL[m]: {"loss_vs_ReComp_pp": -cur[m]["delta_vs_ReComp_pp"], "within_threshold": -cur[m]["delta_vs_ReComp_pp"] <= Q.LLAVA_U25_LOSS_PP} for m in packed}},
        "note": "둘 다 사전 고정한 기술 통계이며 HOLD에서 관찰한 값이다. 다른 데이터에서 확인되지 않았다. CI가 0을 포함하는 것은 '차이 미입증'이며 동등하다는 뜻이 아니다."}

    # ------------------------------------------------- gates, protection, status
    started = read(run / "main_started_at.json")["at_unix"]
    finished = read(run / "main_finished_at.json")["at_unix"]
    interruptions = sorted(p.name for p in run.glob("interruption_*.json"))
    failed_attempts = sorted(str(p.parent.relative_to(run)) for p in run.glob("*/images/*/attempt_*/failure.json"))
    gpu = read(run / "gates/sg_gpu.json")
    integ_val = read(run / "integration_validation.json")
    gates = {"SG1 CPU (벡터 SHA·합, N=1–3,000 정수 분수 k, validator, 기본값 미사용)": read(run / "gates/sg1_cpu.json")["status"],
             "SG2 25% 회귀 (payload·K/V·SDPA 입력·logits·ID bitwise, PyramidKV run 코드 대비)": "PASS" if gpu["per_gate"]["SG2"] else "FAIL",
             "SG3 100% 동치 (U100_packed ≡ FullLoad: K/V·SDPA 입력·logits·ID bitwise)": "PASS" if gpu["per_gate"]["SG3"] else "FAIL",
             "SG4 포함 관계 (HOLD 398장, U25 ⊂ U30 ⊂ U40 ⊂ U50 ⊂ U100)": read(run / "gates/sg4_inclusion.json")["status"],
             "SG5 I/O (preadv 1회·pread 0회, 바이트, vision 0·score 0, 질문 불변)": "PASS" if gpu["per_gate"]["SG5"] else "FAIL",
             "SGR 참조 검증 (추가: 독립 in-memory 참조 bitwise, 30/40/50% FP64 1e-12)": "PASS" if gpu["per_gate"]["SGR"] else "FAIL",
             f"SG6 통합 (7 arm × 12 요청 × 4 dialogue = {E['integration']} 요청 + 독립 감사)": integ_val["status"],
             "기존 CPU 테스트 (non-GLM 전체)": read(run / "gates/cpu_regression.json")["status"],
             f"Smoke {E['smoke']} 요청": read(run / "smoke/validation.json")["status"]}
    final_ok = audit["status"] == "PASS" and audit["main_requests"] == E["main"] and audit["failed_or_skipped_requests"] == 0
    status = "FINAL" if integ["status"] == "PASS" and final_ok else "PENDING_REVIEW"
    protection = protection_after(run, res)
    dele = deletion_log(run, res)
    hours = (finished - started) / 3600
    summary = {"status": status, "verdict": "none (curve reported; no GO/NO-GO)", "run": str(run.relative_to(ROOT)),
               "curve": curve, "baselines": baselines, "adjacent_ratio_steps": steps,
               "non_monotone_segments": [s for s in steps if s["accuracy_drops_as_ratio_grows"]],
               "integrity_u100_packed_vs_fullload": {k: v for k, v in integ.items() if k != "mismatch_dump"},
               "packed_path_effect_at_100pct": {"definition": "U100_packed - fullload, same KV; file layout / read syscall / assembly differ",
                                                "ttft_delta_ms": pair(U100, "fullload", metric="ttft_ms")["delta"],
                                                "ttft_ci95_ms": [pair(U100, "fullload", metric="ttft_ms")["ci_lo"], pair(U100, "fullload", metric="ttft_ms")["ci_hi"]],
                                                "ttft_ratio": lat[U100]["ttft_mean_ms"] / lat["fullload"]["ttft_mean_ms"],
                                                "ssd_mb": [lat[U100]["ssd_mb_per_followup"], lat["fullload"]["ssd_mb_per_followup"]]},
               "determinism_vs_pyramidkv_run": det, "audit_status": audit["status"], "final": final_ok, "requests": audit["main_requests"],
               "expected_requests": E["main"], "images": len(images), "failed_or_skipped_requests": audit["failed_or_skipped_requests"],
               "failed_attempt_records": failed_attempts, "interruptions": interruptions, "main_wall_hours_first_start_to_finish": hours,
               "gates": gates, "rule": config["report_rule"], "llava_reference": {"uniform_25_loss_vs_ReComp_pp": Q.LLAVA_U25_LOSS_PP, "median_image_rows_per_layer_at_25pct": Q.LLAVA_U25_MEDIAN_ROWS,
                                                                                 "source": "prompt (LLaVA v2 HOLD, same MT-GQA); not re-measured here"},
               "contract_sha256": config["contract_sha256"], "config_sha256": sha(run / "config.json"),
               "protection": protection["status"], "deletion_log": dele["status"], "not_run": ["MT-VQA", "GLM", "LLaVA"]}
    write(res / "summary.json", summary, exclusive=False)
    figures(res, curve)

    # --------------------------------------------------------------- copies
    for name in ("CONTRACT.md", "config.json", "storage_plan.json", "integration_validation.json", "source_modifications.json"):
        shutil.copy2(run / name, res / name)
    shutil.copy2(final / "independent_audit.json", res / "independent_audit.json")
    for sub in ("vectors", "phase0"):
        (res / sub).mkdir(exist_ok=True)
        for p in (run / sub).iterdir():
            shutil.copy2(p, res / sub / p.name)
    (res / "gates").mkdir(exist_ok=True)
    for name in ("sg1_cpu.json", "sg4_inclusion.json", "cpu_regression.json", "sg_gpu.json"):
        shutil.copy2(run / "gates" / name, res / "gates" / name)
    (res / "gates" / "gpu_cases").mkdir(exist_ok=True)
    for p in sorted((run / "gates/gpu").glob("*/evidence.json")):
        shutil.copy2(p, res / "gates" / "gpu_cases" / f"{p.parent.name}.json")
    shutil.copy2(run / "smoke/validation.json", res / "gates" / "smoke_validation.json")
    for p in sorted(run.glob("freeze_amendment_*.json")):
        shutil.copy2(p, res / p.name)

    # ---------------------------------------------------------------- report
    L = []
    rc, fl = cur["recompute"], cur["fullload"]
    L.append("# Qwen2.5-VL 보존 비율 sweep (25 / 30 / 40 / 50 / 100%) — MT-GQA HOLD\n")
    L.append(f"Run `{run.relative_to(ROOT)}`. 계약 [CONTRACT.md](CONTRACT.md) (SHA `{config['contract_sha256'][:16]}…`, gate·smoke·main 전에 freeze). "
             f"단위: MB = 10^6 B, ms, 정확도 차이는 pp. followup = T2–T3. CI = 이미지 단위 paired bootstrap 95% (10,000회, seed 1234).\n")
    L.append(f"**판정은 없다. 곡선을 보고한다.** 상태: **{status}** — main {audit['main_requests']:,} / {E['main']:,} 요청, 실패·스킵 {audit['failed_or_skipped_requests']}, "
             f"독립 감사 {audit['status']}, U100_packed ≡ FullLoad {integ['token_exact']:,} / {integ['requests']:,}.\n")
    if integ["status"] != "PASS":
        L.append("U100_packed와 FullLoad의 예측이 100% 일치하지 않아 상태를 PENDING_REVIEW로 둔다. 불일치 분석은 아래 '정합성 사후 확인' 참조.\n")
    L.append("## 곡선 (HOLD followup)\n")
    L.append(md_table(["Arm", "비율", f"layer당 content 행 (N={n_med})", "Followup % (정답/N)", "vs ReComp Δpp [CI]", "vs FullLoad Δpp [CI]", "vs U25 Δpp [CI]",
                       "TTFT mean ms", "TTFT / ReComp", "ReComp 대비 단축", "SSD MB / followup"],
                      [[c["name"], f"{round(100 * c['ratio'])}%" if c["packed"] else "100% (전체)" if c["method"] == "fullload" else "–", c["k_per_layer_median_image"] if c["method"] != "recompute" else "–",
                        f"{c['followup_pct']:.3f} ({c['followup_correct']}/{c['followup_n']})",
                        pm({"delta": c["delta_vs_ReComp_pp"], "ci_lo": c["ci_vs_ReComp_pp"][0], "ci_hi": c["ci_vs_ReComp_pp"][1]}) if "delta_vs_ReComp_pp" in c else "–",
                        pm({"delta": c["delta_vs_FullLoad_pp"], "ci_lo": c["ci_vs_FullLoad_pp"][0], "ci_hi": c["ci_vs_FullLoad_pp"][1]}) if "delta_vs_FullLoad_pp" in c else "–",
                        pm({"delta": c["delta_vs_U25_pp"], "ci_lo": c["ci_vs_U25_pp"][0], "ci_hi": c["ci_vs_U25_pp"][1]}) if "delta_vs_U25_pp" in c else "–",
                        f(c["ttft_mean_ms"]), f"{c['ttft_ratio_vs_ReComp']:.3f}×", f"{c['ttft_reduction_vs_ReComp_pct']:.1f}%", f(c["ssd_mb_per_followup"])] for c in curve]))
    L.append("Δ는 (행의 arm − 기준 arm)이다. CI가 0을 포함하는 차이는 '차이 미입증'이며 동등하다는 뜻이 아니다. T1은 모든 arm이 full-pixel이다. "
             "차이는 generated history가 갈라지는 효과를 포함한 method 수준 비교이다.\n")
    L.append("그림: [fig_tradeoff_qwen.png](fig_tradeoff_qwen.png) (가로축 TTFT), [fig_tradeoff_qwen_ssd.png](fig_tradeoff_qwen_ssd.png) (가로축 SSD MB). 수치 전체는 [summary.json](summary.json).\n")
    L.append("이 실험에서 반드시 알아야 할 것:\n")
    L.append("- **모든 packed arm은 같은 중요도 순서(image-only saliency)의 앞 k개를 모든 layer에 똑같이 쓰고 비율만 다르다.** k = ceil(비율 × N). 구조 행 21개는 항상 전부 남긴다. "
             "큰 비율의 집합은 작은 비율의 집합을 포함한다(HOLD 398장 전부에서 감사).\n"
             "- **U100_packed와 FullLoad의 TTFT 차이는 packed 경로(파일 1개·preadv 1회·GPU 조립)의 효과이며 토큰 선택과 무관하다.** 두 arm은 같은 KV를 쓴다.\n"
             f"- **Qwen의 N은 {ktab['N_min']}–{ktab['N_max']}이다(중간값 이미지 {n_med}).** layer당 남는 content 토큰 수(중간값 이미지): "
             + ", ".join(f"{round(100 * Q.RATIO[x])}% → {ktab['k_by_N'][str(n_med)][x]}개" for x in ALLOCS)
             + f". LLaVA 25%는 중간값 이미지에서 {Q.LLAVA_U25_MEDIAN_ROWS}개다(N 1,344–2,880; 이 실행에서 다시 재지 않은 선행 값).\n"
             "- MT-VQA·GLM·LLaVA는 실행하지 않았다.\n")

    L.append("## 사전 고정한 요약 기준선 (기술 통계)\n")
    L.append(md_table(["기준선", "규칙", "결과"],
                      [["(a) 손실 미검출", "ReComp 대비 Δ의 95% CI가 0을 포함하는 가장 작은 비율", f"**{baselines['a_loss_not_detected']['result']}**"],
                       ["(b) LLaVA 25% 수준", f"ReComp 대비 손실 점추정이 {Q.LLAVA_U25_LOSS_PP:.2f} pp 이하인 가장 작은 비율", f"**{baselines['b_llava_25_level']['result']}**"]]))
    L.append(md_table(["비율", "ReComp 대비 Δpp [95% CI]", "CI가 0 포함 (a)", f"손실 ≤ {Q.LLAVA_U25_LOSS_PP:.2f} pp (b)"],
                      [[LABEL[m], pm({"delta": cur[m]["delta_vs_ReComp_pp"], "ci_lo": cur[m]["ci_vs_ReComp_pp"][0], "ci_hi": cur[m]["ci_vs_ReComp_pp"][1]}),
                        "예 (차이 미입증)" if cur[m]["ci_vs_ReComp_includes_zero"] else "아니오", "예" if -cur[m]["delta_vs_ReComp_pp"] <= Q.LLAVA_U25_LOSS_PP else "아니오"] for m in packed]))
    L.append("두 기준선은 실행 전에 고정했고 HOLD 결과를 보고 바꾸지 않았다. **둘 다 HOLD에서 관찰한 값이며 다른 데이터에서 확인되지 않았다.** "
             "(a)는 손실이 없다는 뜻이 아니라 이 표본 크기에서 손실을 검출하지 못했다는 뜻이다.\n")
    if base_a and base_a != Q.U100 and not cur[base_a]["ci_vs_FullLoad_includes_zero"]:
        L.append(f"**(a)는 경계선 결과다.** {LABEL[base_a]}의 ReComp 대비 CI 상한은 {cur[base_a]['ci_vs_ReComp_pp'][1]:+.3f} pp로 0을 겨우 넘고, "
                 f"같은 arm을 FullLoad(= U100_packed와 같은 KV)와 비교하면 Δ {cur[base_a]['delta_vs_FullLoad_pp']:+.3f} pp, "
                 f"CI [{cur[base_a]['ci_vs_FullLoad_pp'][0]:+.3f}, {cur[base_a]['ci_vs_FullLoad_pp'][1]:+.3f}]로 0을 포함하지 않는다. "
                 f"점추정으로는 {LABEL[base_a]}도 ReComp보다 {-cur[base_a]['delta_vs_ReComp_pp']:.3f} pp 낮다. "
                 "기준선 (a)의 규칙(ReComp 대비 CI)은 그대로 적용했고, 이 문장은 결과를 본 뒤 해석을 돕기 위해 덧붙인 것이다.\n")

    L.append("## 비율을 올릴 때의 변화 (인접 비율 비교)\n")
    L.append(md_table(["구간", "Followup Δpp [95% CI]", "T2 Δpp", "T3 Δpp", "TTFT Δ ms", "SSD MB Δ", "비고"],
                      [[f"{LABEL[lo_m]} → {LABEL[hi_m]}", pm(pair(hi_m, lo_m)), f"{pair(hi_m, lo_m, 'T2')['delta']:+.3f}", f"{pair(hi_m, lo_m, 'T3')['delta']:+.3f}",
                        f"{lat[hi_m]['ttft_mean_ms'] - lat[lo_m]['ttft_mean_ms']:+.3f}", f"{lat[hi_m]['ssd_mb_per_followup'] - lat[lo_m]['ssd_mb_per_followup']:+.3f}",
                        ("**정확도 하락**" + (" (CI 0 제외)" if pair(hi_m, lo_m)["ci_excludes_zero"] else " (CI 0 포함: 차이 미입증)")) if pair(hi_m, lo_m)["delta"] < 0
                        else ("상승, CI 0 제외" if pair(hi_m, lo_m)["ci_excludes_zero"] else "CI 0 포함: 차이 미입증")] for lo_m, hi_m in zip(packed, packed[1:])]))
    drops = [s for s in steps if s["accuracy_drops_as_ratio_grows"]]
    if drops:
        L.append("비율이 커지는데 followup 정확도 점추정이 떨어진 구간: " + "; ".join(
            f"{s['from']} → {s['to']} {s['delta_followup_pp']:+.3f} pp [{s['ci95_pp'][0]:+.3f}, {s['ci95_pp'][1]:+.3f}]" for s in drops) + ". 숨기지 않고 그대로 적는다.\n")
    else:
        L.append("비율이 커지는데 followup 정확도 점추정이 떨어진 구간은 없다.\n")

    L.append("## 정확도 (turn별)\n")
    L.append(md_table(["Arm", "T1 %", "T2 %", "T3 %", "All % (정답/N)", "Followup % [95% CI]"],
                      [[q["name"], f(q["T1_pct"]), f(q["T2_pct"]), f(q["T3_pct"]), f"{q['all_pct']:.3f} ({q['all_correct']}/{q['all_n']})",
                        f"{q['followup_pct']:.3f} [{q['followup_ci_lo']:.3f}, {q['followup_ci_hi']:.3f}]"] for q in quality]))
    L.append(md_table(["비교 (A − B)", "Followup Δpp [95% CI]", "All Δpp [95% CI]", "T2 Δpp", "T3 Δpp", "A만 / B만 정답 (followup)", "예측 일치율 (followup)"],
                      [[f"{LABEL[a_]} − {LABEL[b_]}", pm(pair(a_, b_)), pm(pair(a_, b_, "all")), f"{pair(a_, b_, 'T2')['delta']:+.3f}", f"{pair(a_, b_, 'T3')['delta']:+.3f}",
                        f"{pair(a_, b_)['a_only']} / {pair(a_, b_)['b_only']}", f"{100 * pair(a_, b_)['prediction_agreement']:.2f}%"]
                       for a_, b_ in [(m, "recompute") for m in packed] + [(m, "fullload") for m in packed] + [("fullload", "recompute")]]))

    L.append("## Followup 성능과 I/O\n")
    L.append(md_table(["Arm", "TTFT mean / p50 / p95 ms", "E2E mean / p95 ms", "TTFT / ReComp (단축)", "TTFT / FullLoad (단축)", "SSD MB", "읽기 호출 (pread / preadv)", "GPU cache KV MB",
                       "peak GPU alloc MB (followup 평균 / 전체 최대)"],
                      [[x["name"], f"{f(x['ttft_mean_ms'])} / {f(x['ttft_p50_ms'])} / {f(x['ttft_p95_ms'])}", f"{f(x['e2e_mean_ms'])} / {f(x['e2e_p95_ms'])}",
                        f"{x['ttft_ratio_vs_ReComp']:.3f}× ({x['ttft_reduction_vs_ReComp_pct']:.1f}%)", f"{x['ttft_ratio_vs_FullLoad']:.3f}× ({x['ttft_reduction_vs_FullLoad_pct']:.1f}%)",
                        f(x["ssd_mb_per_followup"]), f"{x['pread_per_followup']:.0f} / {x['preadv_per_followup']:.0f}", f(x["gpu_cache_kv_mb_per_followup"]),
                        f"{x['peak_gpu_allocated_followup_mean_mb']:.1f} / {x['peak_gpu_allocated_max_mb']:.1f}"] for x in latency]))
    L.append("구간별 평균 ms (followup; 구간은 서로 겹칠 수 있어 합산하지 않는다. `읽기 포함`은 계획 + 읽기 + (FullLoad는) CPU 조립):\n")

    def sg(m, key):
        return f(lat[m].get("seg_" + key + "_mean_ms"))
    L.append(md_table(["Arm", "prompt 준비", "읽기 포함", "raw read", "계획", "H2D·조립", "hook 설치", "suffix prefill", "decode"],
                      [[LABEL[m], sg(m, "prompt_preparation"), sg(m, "store_load_inclusive"), sg(m, "raw_pread"),
                        sg(m, "layerwise_planning") if m in PACKED else sg(m, "fixed_first_k_planning"), sg(m, "h2d_and_assembly"), sg(m, "mask_hook_install"),
                        sg(m, "suffix_prefill") if m != "recompute" else sg(m, "multimodal_prefill") + " (full prefill)", sg(m, "decode")] for m in METHODS]))
    d100 = pair(U100, "fullload", metric="ttft_ms")
    L.append(f"**packed 경로 효과 (100%에서, 같은 KV)**: U100_packed − FullLoad의 TTFT = {d100['delta']:+.3f} ms [{d100['ci_lo']:+.3f}, {d100['ci_hi']:+.3f}] "
             f"({lat[U100]['ttft_mean_ms'] / lat['fullload']['ttft_mean_ms']:.3f}×; p50 {f(lat[U100]['ttft_p50_ms'])} vs {f(lat['fullload']['ttft_p50_ms'])}, "
             f"p95 {f(lat[U100]['ttft_p95_ms'])} vs {f(lat['fullload']['ttft_p95_ms'])}). 읽기 호출은 57회 pread → 1회 preadv. "
             f"읽은 바이트는 U100_packed {f(lat[U100]['ssd_mb_per_followup'])} MB, FullLoad {f(lat['fullload']['ssd_mb_per_followup'])} MB이다"
             "(FullLoad는 64행 chunk 단위로 padding 행까지 읽는다). 이 차이는 파일 배치·읽기 호출 수·조립 위치(CPU → GPU)를 합친 packed 경로 전체의 효과이며, 어느 토큰을 남기는지와 무관하다.\n")
    L.append("DONTNEED는 OS page-cache hint이며 NAND cold나 O_DIRECT가 아니다. 절대 TTFT는 다른 실행과 비교하지 않는다.\n")

    L.append("## 저장과 setup (이미지당 1회)\n")
    L.append(md_table(["Arm", "파일 수", "store MB / 이미지", "persistence mean / p50 / p95 ms", "writer ms", "write / fsync ms", "activation mean / p50 / p95 ms"],
                      [[s["name"], s["files_per_store"], f(s["write_mb_mean"]), f"{f(s['outer_persistence_mean_ms'])} / {f(s['outer_persistence_p50_ms'])} / {f(s['outer_persistence_p95_ms'])}",
                        f(s["native_writer_mean_ms"]), f"{f(s['ssd_write_mean_ms'])} / {f(s['fsync_mean_ms'])}",
                        f"{f(s['activation_mean_ms'])} / {f(s['activation_p50_ms'])} / {f(s['activation_p95_ms'])}"] for s in storage]))
    L.append("persistence는 writer wall + completion seal, activation은 메타 검증 + payload SHA256 전체 검증 + (packed) pinned buffer 할당이다. 파일 수는 payload + meta.json + completion.json.\n")

    L.append("## 3-turn session (shared-stream / standalone-equivalent)\n")
    L.append(md_table(["Arm", "setup ms/image", "shared-stream mean / p50 / p95 ms", "standalone-equivalent (DERIVED) mean / p50 / p95 ms"],
                      [[LABEL[m], f(sess[m]["setup_ms_per_image"]),
                        f"{f(sess[m]['shared_stream_session_E2E_mean_ms'])} / {f(sess[m]['shared_stream_session_E2E_p50_ms'])} / {f(sess[m]['shared_stream_session_E2E_p95_ms'])}",
                        f"{f(sess[m]['derived_standalone_E2E_mean_ms'])} / {f(sess[m]['derived_standalone_E2E_p50_ms'])} / {f(sess[m]['derived_standalone_E2E_p95_ms'])}"] for m in METHODS]))
    L.append(md_table(["A − B", "accounting", "Δ ms [95% CI]"], [[f"{p['a_name']} − {p['b_name']}", p["accounting"], f"{p['delta_ms']:+.3f} [{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}]"] for p in sess_pairs]))
    L.append("shared-stream은 이미지 setup을 source dialogue에 한 번 부과한 실측 합이다. standalone-equivalent는 같은 실측 setup을 dialogue마다 다시 부과한 **DERIVED** 값이며 "
             "non-source T1에 capture hook을 걸었을 때의 증분은 NOT_MEASURED이다.\n")

    L.append("## 비율별 k와 바이트\n")
    L.append(md_table(["N", *[f"{x} k" for x in ALLOCS], *[f"{x} MB" for x in ALLOCS]],
                      [[n, *[ktab["k_by_N"][str(n)][x] for x in ALLOCS], *[f"{ktab['packed_bytes_by_N'][str(n)][x] / 1e6:.3f}" for x in ALLOCS]] for n in (ktab["N_min"], n_med, ktab["N_max"])]))
    L.append(f"HOLD 398장의 N은 {ktab['N_min']} / {n_med} / {ktab['N_max']} (최소 / 실제 이미지 중간값 / 최대), 서로 다른 N {ktab['distinct_N']}개. "
             "packed 파일 크기 = (k + 21) × 57,344 B. k는 부동소수 `layer_keep_counts`와 정수 분수 `ceil(p·N/q)` 두 가지로 계산해 N = 1–3,000 전부에서 일치함을 확인했다. "
             "전체 표는 [phase0/k_tables.csv](phase0/k_tables.csv).\n")

    L.append("## Gate\n")
    L.append(md_table(["Gate", "결과"], [[k, v] for k, v in gates.items()] + [[f"독립 최종 감사 ({audit['main_requests']:,} 요청)", audit["status"]],
                                                                        ["Artifact protection before/after", protection["status"]], ["Deletion log (run-local scratch·gate store만)", dele["status"]]]))
    reg = read(run / "gates/cpu_regression.json")
    L.append(f"기존 CPU 테스트: {reg['ran']} (새 테스트 14개 포함), returncode {reg['returncode']}. GLM-4.6V 테스트는 분리 실행했고 {len(reg['glm_errors'])}건이 "
             f"PyramidKV run 때와 {'똑같이' if reg['glm_errors_same_as_pyramidkv_run'] else '다르게'} 실패한다(설치된 transformers API 문제, 새 파일을 import하지 않음). "
             "SG2의 meta 차이는 문서 필드뿐이며 목록은 `gates/gpu_cases/*.json`의 `SG2.meta_differs_only_in_document_fields`에 있다.\n")

    L.append("## 정합성 사후 확인 — U100_packed ≡ FullLoad\n")
    L.append(f"U100_packed와 FullLoad는 같은 KV를 쓰므로 예측이 토큰 단위로 같아야 한다. main {integ['requests']:,} 요청 중 **{integ['token_exact']:,}건 일치 "
             f"({100 * integ['token_exact_rate']:.3f}%)**, 불일치 {integ['mismatches']}건.\n")
    if integ["mismatches"]:
        L.append(f"처음 갈라진 지점(같은 history에서 토큰이 달라진 요청): {integ['first_divergence_points_same_history']}건 (turn별 {integ['first_divergence_by_turn']}). "
                 f"나머지 {integ['propagated_through_history']}건은 앞 turn의 답이 달라져 history가 갈라진 뒤의 요청이다. 전체 덤프는 "
                 "[u100_packed_vs_fullload_integrity.json](u100_packed_vs_fullload_integrity.json).\n")
        L.append(md_table(["dialogue", "turn", "같은 history", "FullLoad 예측", "U100_packed 예측", "첫 토큰 같음"],
                          [[m["dialog_id"], m["turn_id"], m["same_history"], m["fullload_prediction"], m["U100_packed_prediction"], m["first_token_equal"]] for m in integ["mismatch_dump"][:30]]))
    else:
        L.append("gate에서도 같은 capture로 만든 두 store의 layer별 K/V·28 layer SDPA 입력·첫 logits이 bitwise 같았고(SG3), integration에서 stored arm 6개의 T1 capture KV가 비트 단위로 같았다.\n")

    L.append("## 결정성 참고 (PyramidKV run의 같은 요청과 비교)\n")
    if det["status"] == "MEASURED":
        L.append(md_table(["이 실행의 arm", "PyramidKV run의 arm", "요청", "토큰 일치", "예측 일치", "followup 토큰 일치"],
                          [[LABEL[m], PYR_METHOD[m], v["n"], f"{100 * v['token_exact_rate']:.3f}%", f"{100 * v['prediction_agreement']:.3f}%", f"{100 * v['followup_token_exact_rate']:.3f}%"]
                           for m, v in det["methods"].items()]))
        L.append("참고 수치일 뿐이다. 절대 TTFT는 실행 간 비교하지 않는다.\n")
    else:
        L.append("NOT_MEASURED: " + det["reason"] + "\n")

    L.append("## 실행 기록\n")
    L.append(f"- 실제 요청 수: main **{audit['main_requests']:,} / {E['main']:,}** ({E['images']} images, {E['dialogues']:,} dialogues × 3 turns × 7 arms), smoke {E['smoke']}, integration {E['integration']}.\n"
             f"- 실패·스킵: {audit['failed_or_skipped_requests']}. 실패한 image attempt 기록: {failed_attempts or '없음'}. 중단 기록: {interruptions or '없음'}.\n"
             f"- 소요 시간: main 첫 시작부터 종료까지 {hours:.2f}시간 (계약의 예상 {config['expected_duration_after_smoke']['estimate_hours'][0]}–{config['expected_duration_after_smoke']['estimate_hours'][1]}시간).\n"
             f"- 독립 감사: {audit['status']} ({audit['checks']:,} checks). Artifact protection: {protection['status']} (SHA {protection['files_before']} files, kvstore inventory {protection['kvstore_files_before']}, "
             f"다른 run inventory {protection['runs_files_before']}). 삭제: run-local scratch {dele['removed_scratch_files']} files / {dele['image_bundles']} image bundles, gate store {dele['removed_gate_store_files']} files — 기존 산출물 삭제 없음.\n"
             f"- freeze amendment: {', '.join(protection['amendment_files']) or '없음'}. 기존 소스 수정 없음(새 파일만 추가).\n")
    L.append("### 프롬프트와 다르게 한 점\n")
    L.append("1. freeze 전에 별도 run root(`runs/qwen_budget_sweep_dev_20261111T1530Z`)에서 gate·integration·smoke를 한 번 리허설했다(정합성 fixture만 사용, 개발 기록으로 보존). "
             "공식 gate·integration·smoke는 freeze 뒤 이 run에서 처음부터 다시 실행했다. 자세한 내용은 계약 15절.\n"
             "2. SG4(CPU)의 점수는 PyramidKV run이 각 HOLD 이미지의 T1에서 capture한 image-only saliency를 썼다(CPU gate에서는 모델을 돌리지 않으므로). "
             "이 실행에서 실제로 capture한 점수로는 main의 이미지별 독립 감사가 398장 전부에서 같은 포함 관계를 다시 확인했다.\n"
             "3. 요구된 gate에 더해 SGR(다섯 비율 각각의 독립 in-memory 참조 bitwise 비교, 30/40/50%의 FP64 의미 검증)을 넣었다. SG2·SG3에는 28 layer SDPA 입력 비교를 더했다.\n"
             "4. 기존 CPU 테스트 중 GLM-4.6V 3건은 이 변경 전부터 실패하므로 PyramidKV run과 같이 분리 보고하고 PASS 기준에서 뺐다.\n"
             "5. 보고 스크립트(176)는 측정 뒤 실행하는 분석 코드라서 freeze 대상 runtime source에서 제외했다. 그림 색 팔레트의 자동 검증 스크립트는 node가 없어 실행하지 못했다"
             "(검증된 기준 팔레트의 앞 세 색을 쓰고 marker 모양과 점별 직접 표기를 함께 썼다). "
             "main 결과로 보고서를 한 번 만든 뒤 176을 두 군데 고쳤다: 그림의 글자 위치(50%·ReComp label이 선·축과 겹침)와 기준선 (a)의 경계선 설명 문장. "
             "수치·규칙·기준선 결과는 바뀌지 않았다.\n"
             "6. 프롬프트 문서가 저장소 파일이 아니라 대화로 전달되어 '결과 기록'을 그 문서에 직접 적지 못했다. 같은 내용을 [RESULT_RECORD.md](RESULT_RECORD.md)에 두었다.\n")
    L.append("### NOT_MEASURED\n")
    L.append("- H2D 시간 단독(조립과 함께 측정), NAND/controller cold, O_DIRECT.\n"
             "- standalone session의 non-source T1 capture 증분(DERIVED 값에 포함되지 않음).\n"
             "- 동시 요청 throughput·메모리 포화.\n"
             "- 25–100% 사이의 다른 비율(60–90% 등)과 25% 미만.\n"
             "- LLaVA 25%의 손실(−2.30 pp)과 토큰 수(552)는 선행 실행의 값이며 이 실행에서 다시 재지 않았다.\n"
             "- MT-VQA, GLM, LLaVA (이번 범위 아님).\n")
    (res / "REPORT.md").write_text("\n".join(L))

    env_lines = " \\\n       ".join(Q.ENV.split(" "))
    (res / "REPRODUCE.md").write_text(f"""# 재현 절차

```bash
cd {ROOT}
export {env_lines}
PY={Q.PY}
RUN={run.relative_to(ROOT)}
$PY scripts/172_freeze_qwen_budget_sweep.py --stage vectors                    # 새 run root, 벡터 5개, k 표
#   docs/qwen_budget_sweep_contract.md 를 확정한 뒤:
$PY scripts/172_freeze_qwen_budget_sweep.py --stage freeze --run $RUN           # storage plan, protection before, config/manifest
$PY scripts/173_validate_qwen_budget_sweep.py --run $RUN --stage cpu            # SG1, SG4 + 기존 CPU 테스트
$PY scripts/173_validate_qwen_budget_sweep.py --run $RUN --stage gpu            # SG2, SG3, SG5, SGR (core pair 10개)
$PY scripts/174_eval_qwen_budget_sweep.py --run $RUN --phase integration        # SG6
$PY scripts/174_eval_qwen_budget_sweep.py --run $RUN --phase all                # smoke {E['smoke']} -> main {E['main']:,} (중단 시 --resume)
$PY scripts/176_report_qwen_budget_sweep.py --run $RUN                          # 독립 감사(175), 곡선, protection after, REPORT
```

고정값: Qwen/Qwen2.5-VL-7B-Instruct `{config['model']['revision']}`, NF4 double / BF16 / SDPA, seed 1234, greedy max 16, batch 1, CPU threads 2,
transformers {config['environment']['transformers']}, torch {config['environment']['torch']}. 벡터 SHA256은 `vectors/vectors_sha256.json`, 소스 SHA256은 `config.json`의 `source_sha256`.
packed hit 경로는 PyramidKV run의 `mmimpress/qwen25/layerwise.py::run_cache_layerwise`를 그대로 쓴다(`mmimpress/qwen25/ratio_packed.py`가 비율만 일반화).
절대 TTFT는 실행마다 달라지므로 같은 실행 안의 arm끼리만 비교한다.
""")
    # the runner leaves CURRENT_STATUS at AWAITING_INDEPENDENT_FINAL_AUDIT; record the terminal state once the audit and the report exist
    previous = read(run / "CURRENT_STATUS.json")
    terminal = {"execution_state": "FINISHED" if final_ok else "AUDIT_FAILED", "status": status, "main_final_requests": audit["main_requests"],
                "expected_main_requests": E["main"], "failed_or_skipped_requests": audit["failed_or_skipped_requests"], "independent_audit": audit["status"],
                "audit_sha256": sha(final / "independent_audit.json"), "protection": protection["status"], "deletion_log": dele["status"],
                "runner_last_status": previous.get("runner_last_status", previous), "at_unix": time.time()}
    for target in (run / "CURRENT_STATUS.json", res / "CURRENT_STATUS.json"):
        write(target, terminal, exclusive=False)
    print(json.dumps({"status": status, "baseline_a": baselines["a_loss_not_detected"]["result"], "baseline_b": baselines["b_llava_25_level"]["result"],
                      "followup_pct": {LABEL[m]: round(qual[m]["followup_pct"], 3) for m in METHODS}, "ttft_ms": {LABEL[m]: round(lat[m]["ttft_mean_ms"], 3) for m in METHODS},
                      "integrity": integ["status"], "token_exact": integ["token_exact"], "audit": audit["status"], "protection": protection["status"],
                      "deletion": dele["status"], "main_hours": hours}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

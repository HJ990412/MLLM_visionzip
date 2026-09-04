"""Tables, confidence intervals, error analysis and figures for the
large-scale / cross-dataset validation.

Accuracy differences of one or two points on a few hundred questions are the
kind of number that flips with a different sample, so everything here is
reported with a PAIRED bootstrap: the same questions are resampled for both
methods at once, which is what makes the difference interval meaningful.  A
McNemar count is reported alongside for the binary view.

  python scripts/12_analysis.py --datasets gqa_large,vqav2,textvqa
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mmimpress.config import RESULTS_DIR

OUT = RESULTS_DIR / "generalization_summary"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dcdbd6"
REF, OTHER = "#8a8985", "#52514e"
# validated all-pairs slots (worst CVD dE 13.0, worst normal-vision dE 16.3)
STYLE = {
    "static_diverse_chunk@25": ("Static+Diverse 25%", "#2a78d6", "o"),
    "static_diverse_chunk@50": ("Static+Diverse 50%", "#008300", "D"),
    "sparsevlm":               ("SparseVLM",          "#e87ba4", "^"),
    "fullload":                ("FullLoad",           REF,       "s"),
    "recompute":               ("ReComp",             OTHER,     "v"),
}
ORDER = ["recompute", "fullload", "sparsevlm",
         "static_diverse_chunk@25", "static_diverse_chunk@50"]


def store_stats(ds):
    """Per-image visual-KV size and chunk count, parsed from the build log.

    Shard stores are deleted as the run advances (a whole dataset would not fit
    on the SSD at once), so the only durable record of what was written is the
    builder's own output.
    """
    log = RESULTS_DIR / ds / "run.log"
    if not log.exists():
        return {}
    rx = re.compile(r"v_num=(\d+) chunks=(\d+) ([\d.]+)GB\+([\d.]+)")
    v, c, kv, side = [], [], [], []
    for line in log.read_text(errors="ignore").splitlines():
        m = rx.search(line)
        if m:
            v.append(int(m.group(1)))
            c.append(int(m.group(2)))
            kv.append(float(m.group(3)))
            side.append(float(m.group(4)))
    if not kv:
        return {}
    return {"n_images_built": len(kv),
            "visual_tokens_mean": float(np.mean(v)),
            "chunks_per_layer_mean": float(np.mean(c)),
            "kv_gb_per_image_mean": float(np.mean(kv)),
            "sidecar_gb_per_image_mean": float(np.mean(side)),
            "total_gb_written": float(np.sum(kv) + np.sum(side))}


def load_rows(ds):
    """{method: {question_id: record}} merged over shards."""
    d = RESULTS_DIR / ds
    per, qorder = defaultdict(dict), []
    for p in sorted(d.glob("shard_*.json")):
        with open(p) as f:
            for r in json.load(f)["rows"]:
                qid = r["question_id"]
                qorder.append((r["image_id"], qid, r["question"], r["gold"]))
                for m, v in r.items():
                    if isinstance(v, dict) and "answer" in v:
                        per[m][qid] = v
    return per, qorder


def bootstrap_ci(a, n_boot=10000, seed=0):
    """95% CI of one method's mean score, same resampling as the paired test."""
    rng = np.random.RandomState(seed)
    a = np.asarray(a, float)
    m = a[rng.randint(0, len(a), size=(n_boot, len(a)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_bootstrap(a, b, n_boot=10000, seed=0):
    """CI for each mean and for the paired difference a - b."""
    rng = np.random.RandomState(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = rng.randint(0, len(a), size=(n_boot, len(a)))
    sa, sb = a[idx].mean(1), b[idx].mean(1)
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return {"a_mean": float(a.mean()), "a_ci": q(sa),
            "b_mean": float(b.mean()), "b_ci": q(sb),
            "diff_mean": float((a - b).mean()), "diff_ci": q(sa - sb)}


def mcnemar(a, b, thresh=0.5):
    """Discordant counts on binarised scores (+ exact-binomial p)."""
    a = np.asarray(a) >= thresh
    b = np.asarray(b) >= thresh
    n01 = int((~a & b).sum())      # only b correct
    n10 = int((a & ~b).sum())      # only a correct
    both = int((a & b).sum())
    neither = int((~a & ~b).sum())
    n = n01 + n10
    if n == 0:
        p = 1.0
    else:
        from math import comb
        k = min(n01, n10)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only": n10, "b_only": n01, "both": both, "neither": neither,
            "p_exact": p}


QTYPE = [
    ("yes/no", re.compile(r"^(is|are|do|does|was|were|has|have|can|did)\b", re.I)),
    ("color", re.compile(r"\bcolou?r|what colou?r\b", re.I)),
    ("count", re.compile(r"\bhow many|number of\b", re.I)),
    ("spatial", re.compile(r"\b(left|right|above|below|behind|front|under|"
                           r"near|beside|top|bottom|side)\b", re.I)),
    ("material/attr", re.compile(r"\b(material|made of|shape|size|texture|"
                                 r"large|small|tall|short|thin|thick)\b", re.I)),
    ("object", re.compile(r"^(what|which|who)\b", re.I)),
]


def qtype(q):
    for name, rx in QTYPE:
        if rx.search(q):
            return name
    return "other"


# ------------------------------------------------------------------ tables
def dataset_table(ds, per, qorder):
    ids = [q[1] for q in qorder]
    ids = list(dict.fromkeys(ids))
    present = [m for m in ORDER if m in per]
    fl = np.array([per["fullload"][i]["acc"] for i in ids])
    rows = []
    for m in present:
        v = per[m]
        sc = np.array([v[i]["acc"] for i in ids])
        tt = np.array([v[i]["ttft"] for i in ids])
        row = {"dataset": ds, "method": STYLE.get(m, (m,))[0],
               "key": m, "n": len(ids),
               "score": 100 * sc.mean(),
               "d_vs_fullload_pp": 100 * (sc.mean() - fl.mean()),
               "ttft_mean_ms": 1e3 * tt.mean(),
               "ttft_p50_ms": 1e3 * float(np.percentile(tt, 50)),
               "ttft_p95_ms": 1e3 * float(np.percentile(tt, 95))}
        lo, hi = bootstrap_ci(sc)
        row["score_ci_lo"], row["score_ci_hi"] = 100 * lo, 100 * hi
        row["speedup_vs_fullload"] = (
            1e3 * np.array([per["fullload"][i]["ttft"] for i in ids]).mean()
            / row["ttft_mean_ms"])
        for k, out in (("disk_mb", "ssd_mb"), ("disk_ms", "ssd_ms"),
                       ("selector_ms", "selector_ms"),
                       ("scatter_ms", "scatter_ms"),
                       ("touched_chunk_fraction", "touched_chunk"),
                       ("n_chunks_selected", "chunks_selected"),
                       ("model_ms", "model_ms"), ("hook_ms", "hook_ms")):
            vals = [v[i].get(k) for i in ids if v[i].get(k) is not None]
            if vals:
                row[out] = float(np.mean(vals))
                if out in ("ssd_ms", "selector_ms"):
                    row[out + "_p50"] = float(np.percentile(vals, 50))
                    row[out + "_p95"] = float(np.percentile(vals, 95))
        if "ssd_mb" in row and "fullload" in per:
            base = np.mean([per["fullload"][i]["disk_mb"] for i in ids])
            row["ssd_ratio"] = row["ssd_mb"] / base
        rows.append(row)
    return rows, ids


def stats_block(ds, per, ids):
    out = {}
    fl = [per["fullload"][i]["acc"] for i in ids]
    for m in ("static_diverse_chunk@25", "static_diverse_chunk@50",
              "sparsevlm"):
        if m not in per:
            continue
        a = [per[m][i]["acc"] for i in ids]
        out[m] = {"vs_fullload": paired_bootstrap(a, fl),
                  "mcnemar_vs_fullload": mcnemar(a, fl)}
        if "sparsevlm" in per and m != "sparsevlm":
            sv = [per["sparsevlm"][i]["acc"] for i in ids]
            out[m]["vs_sparsevlm"] = paired_bootstrap(a, sv)
    return out


def error_analysis(per, qorder, ids, m="static_diverse_chunk@25"):
    meta = {q[1]: q for q in qorder}
    groups = {"fullload_only": [], "ours_only": [], "both": [], "neither": []}
    by_type = defaultdict(lambda: {"fullload_only": 0, "ours_only": 0,
                                   "both": 0, "neither": 0, "n": 0})
    for i in ids:
        f = per["fullload"][i]["acc"] >= 0.5
        o = per[m][i]["acc"] >= 0.5
        g = ("both" if f and o else "fullload_only" if f else
             "ours_only" if o else "neither")
        t = qtype(meta[i][2])
        by_type[t][g] += 1
        by_type[t]["n"] += 1
        if g in ("fullload_only", "ours_only"):
            groups[g].append({"question_id": i, "image_id": meta[i][0],
                              "question": meta[i][2], "gold": meta[i][3],
                              "fullload": per["fullload"][i]["answer"],
                              "ours": per[m][i]["answer"], "qtype": t})
        else:
            groups[g].append(i)
    return groups, dict(by_type)


# ----------------------------------------------------------------- figures
def _style(ax, xl, yl, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.set_xlabel(xl, color=INK2, fontsize=9)
    ax.set_ylabel(yl, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)


def fig_scatter(all_rows, xkey, xlabel, title, out):
    """One facet per dataset, five methods, shared legend.

    Five series is past the point where direct labels stay legible next to tall
    CI whiskers, so identity comes from the legend plus a distinct marker shape
    rather than from labels crowded around the markers.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dss = sorted({r["dataset"] for r in all_rows})
    fig, axes = plt.subplots(1, len(dss), figsize=(4.4 * len(dss), 5.0),
                             dpi=170, facecolor=SURFACE, squeeze=False)
    handles, seen = [], set()
    for ax, ds in zip(axes[0], dss):
        rows = [r for r in all_rows if r["dataset"] == ds and xkey in r]
        _style(ax, xlabel, "Score (%)", ds)
        if not rows:
            continue
        xs = [r[xkey] for r in rows]
        ylo = min(r.get("score_ci_lo", r["score"]) for r in rows)
        yhi = max(r.get("score_ci_hi", r["score"]) for r in rows)
        px = (max(xs) - min(xs)) * 0.18 or 1
        py = (yhi - ylo) * 0.12 or 1
        ax.set_xlim(min(xs) - px, max(xs) + px)
        ax.set_ylim(ylo - py, yhi + py)
        for key in ORDER:
            r = next((r for r in rows if r["key"] == key), None)
            if r is None:
                continue
            name, col, mk = STYLE.get(key, (key, OTHER, "o"))
            if "score_ci_lo" in r:
                ax.errorbar(r[xkey], r["score"],
                            yerr=[[r["score"] - r["score_ci_lo"]],
                                  [r["score_ci_hi"] - r["score"]]],
                            fmt="none", ecolor=col, elinewidth=1.6,
                            capsize=4, capthick=1.6, zorder=2, alpha=0.85)
            h = ax.scatter(r[xkey], r["score"], s=105, color=col, marker=mk,
                           zorder=3, edgecolors=SURFACE, linewidths=1.8,
                           label=name)
            if name not in seen:
                handles.append(h)
                seen.add(name)
    fig.suptitle(f"{title}   (whiskers: 95% bootstrap CI)", color=INK,
                 fontsize=12, x=0.01, ha="left")
    fig.legend(handles=handles, frameon=False, fontsize=8.5, ncol=5,
               labelcolor=INK2, loc="lower center", bbox_to_anchor=(0.5, 0.012))
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def fig_reuse(per, qorder, out, static_ms):
    """Per-question online cost vs how many questions the image has served.

    The static metadata is built once per image; this shows what a question
    actually pays online and how the one-off build amortises as an image
    answers more questions.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = "static_diverse_chunk@25"
    seen = defaultdict(int)
    by_k = defaultdict(list)
    for img, qid, _q, _g in qorder:
        if qid not in per.get(m, {}):
            continue
        seen[img] += 1
        by_k[seen[img]].append(per[m][qid]["selector_ms"])
    ks = sorted(by_k)
    online = [float(np.mean(by_k[k])) for k in ks]
    amort = [static_ms / k for k in ks]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=170, facecolor=SURFACE)
    _style(ax, "question index for the same image",
           "per-question overhead (ms)",
           "Static metadata is built once and reused")
    ax.plot(ks, online, color="#2a78d6", lw=2, marker="o", ms=7,
            markeredgecolor=SURFACE, markeredgewidth=1.6,
            label="online selector (measured)")
    ax.plot(ks, amort, color="#eda100", lw=2, marker="s", ms=7, ls=(0, (5, 3)),
            markeredgecolor=SURFACE, markeredgewidth=1.6,
            label=f"one-off metadata build / k  ({static_ms:.0f} ms once)")
    ax.set_xticks(ks)
    ax.set_yscale("log")
    import matplotlib.ticker as mt
    lo = min(min(online), min(amort)) * 0.5
    hi = max(max(online), max(amort)) * 2.2
    ax.set_ylim(lo, hi)
    ticks = [t for t in (5, 10, 20, 50, 100, 200, 500, 1000) if lo <= t <= hi]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(mt.NullFormatter())
    ax.annotate(f"{online[0]:.0f} ms, flat", (ks[0], online[0]), color=INK,
                fontsize=8.5, xytext=(6, 8), textcoords="offset points")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="gqa_large,vqav2,textvqa")
    ap.add_argument("--static-ms", type=float, default=591.4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    all_rows, all_stats, all_err = [], {}, {}
    first_per = first_qorder = None
    for ds in args.datasets.split(","):
        d = RESULTS_DIR / ds
        if not list(d.glob("shard_*.json")):
            print(f"{ds}: no shards, skipping")
            continue
        per, qorder = load_rows(ds)
        rows, ids = dataset_table(ds, per, qorder)
        all_rows += rows
        all_stats[ds] = stats_block(ds, per, ids)
        groups, by_type = error_analysis(per, qorder, ids)
        all_err[ds] = {"counts": {k: len(v) for k, v in groups.items()},
                       "by_qtype": by_type}
        st = store_stats(ds)
        if st:
            all_stats.setdefault(ds, {})["store"] = st
            print(f"  store: {st['n_images_built']} images built, "
                  f"{st['kv_gb_per_image_mean']:.2f} GB KV + "
                  f"{st['sidecar_gb_per_image_mean']:.2f} GB sidecar each, "
                  f"{st['chunks_per_layer_mean']:.1f} chunks/layer, "
                  f"{st['visual_tokens_mean']:.0f} visual tokens, "
                  f"{st['total_gb_written']:.0f} GB written in total")
        with open(OUT / f"errors_{ds}.json", "w") as f:
            json.dump({"fullload_only": groups["fullload_only"],
                       "ours_only": groups["ours_only"],
                       "by_qtype": by_type}, f, indent=1)
        if first_per is None:
            first_per, first_qorder = per, qorder
        print(f"\n=== {ds} (n={rows[0]['n']}) ===")
        hdr = (f"{'method':<22}{'score':>8}{'dFL':>7}{'TTFT':>8}{'p50':>8}"
               f"{'p95':>8}{'x':>6}{'sel ms':>8}{'MB':>8}{'ratio':>7}"
               f"{'chunkf':>8}")
        print(hdr)
        for r in rows:
            g = lambda k, f="{:>8.1f}", w=8: (f.format(r[k]) if k in r
                                              else " " * w)
            print(f"{r['method']:<22}{r['score']:>7.1f}%{r['d_vs_fullload_pp']:>7.1f}"
                  f"{r['ttft_mean_ms']:>8.1f}{r['ttft_p50_ms']:>8.1f}"
                  f"{r['ttft_p95_ms']:>8.1f}{r['speedup_vs_fullload']:>6.2f}"
                  f"{g('selector_ms')}{g('ssd_mb')}{g('ssd_ratio','{:>7.3f}',7)}"
                  f"{g('touched_chunk','{:>8.3f}')}")

    cols = sorted({k for r in all_rows for k in r})
    with open(OUT / "main_tables.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(all_rows)
    # timing broken out separately, as requested: metadata lookup and selection
    # are offline/online separated, SSD read and reconstruction are separate
    # stages, and model_ms is the forward up to the first token
    tcols = ["dataset", "method", "ttft_mean_ms", "ttft_p50_ms", "ttft_p95_ms",
             "selector_ms", "selector_ms_p50", "selector_ms_p95",
             "ssd_ms", "ssd_ms_p50", "ssd_ms_p95", "scatter_ms", "model_ms",
             "hook_ms", "speedup_vs_fullload"]
    with open(OUT / "timing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, tcols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    with open(OUT / "stats.json", "w") as f:
        json.dump(all_stats, f, indent=1)
    with open(OUT / "errors_summary.json", "w") as f:
        json.dump(all_err, f, indent=1)

    fig_scatter(all_rows, "ssd_ratio", "SSD bytes read / FullLoad",
                "Score vs SSD read", OUT / "fig1_ssd_vs_score.png")
    fig_scatter(all_rows, "ttft_mean_ms", "TTFT (ms, mean)",
                "Score vs TTFT", OUT / "fig2_ttft_vs_score.png")
    if first_per is not None:
        fig_reuse(first_per, first_qorder, OUT / "fig3_metadata_reuse.png",
                  args.static_ms)

    print("\n=== paired bootstrap (10k resamples) ===")
    for ds, st in all_stats.items():
        for m, v in st.items():
            if m == "store":
                continue
            b = v["vs_fullload"]
            mc = v["mcnemar_vs_fullload"]
            print(f"{ds:<11}{m:<26} score {b['a_mean']*100:5.1f}% "
                  f"[{b['a_ci'][0]*100:.1f}, {b['a_ci'][1]*100:.1f}]  "
                  f"vs FullLoad {b['diff_mean']*100:+.1f}pp "
                  f"[{b['diff_ci'][0]*100:+.1f}, {b['diff_ci'][1]*100:+.1f}]  "
                  f"McNemar ours-only {mc['a_only']} / FL-only {mc['b_only']} "
                  f"p={mc['p_exact']:.3g}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

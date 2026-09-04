"""Ablation tables + figures: what actually makes chunk selection work?

Reads the eval JSONs produced by scripts/04_eval.py, splits them into the
per-method files, writes summary CSVs, and renders three figures.

Figures are light-surface PNGs committed to the repo, so they deliberately
commit to a single look and paint surface/ink explicitly rather than inheriting
matplotlib defaults.  The categorical hues are the five slots that clear the
data-viz validator on the ALL-PAIRS pairlist (scatter needs all pairs, not just
adjacent): worst CVD dE 13.0, worst normal-vision dE 16.3, both above their
gates.  Two of the five sit below 3:1 against the surface, which obliges the
relief rule -- every point carries a visible direct label and every number also
exists in the README tables, so identity is never colour-alone.

  python scripts/07_ablation_report.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mmimpress.config import RESULTS_DIR

ABL = RESULTS_DIR / "ablation_25"
SWEEP = RESULTS_DIR / "budget_sweep"
FIGS = RESULTS_DIR / "figures"

# display name -> (key in the eval json, colour slot)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dcdbd6"
# name -> (json key, hue, marker).  The five hues are the slots that clear the
# validator on the ALL-PAIRS pairlist; the marker is a second, redundant
# encoding so series that land on identical values stay distinguishable.
METHODS = {
    "Static + Diverse":   ("static_diverse_chunk",   "#2a78d6", "o"),
    "Random":             ("random_chunk",           "#eda100", "s"),
    "VisionZip Static":   ("visionzip_static_chunk", "#e87ba4", "^"),
    "Diverse Only":       ("diverse_chunk",          "#008300", "D"),
    "Hybrid + Diverse":   ("cvpr25_hybrid_diverse",  "#4a3aa7", "v"),
}
EXTRA = {"Hybrid": "cvpr25_hybrid_chunk", "SparseVLM": "sparsevlm"}
REF = "#8a8985"          # FullLoad reference, deliberately not a series hue
OTHER = "#52514e"        # "other" bucket for diagnostic points, also neutral


def load(p):
    with open(p) as f:
        return json.load(f)


def random_seeds(d):
    """Every random_chunk_s* method in one summary."""
    return {k: v for k, v in d["summary"]["per_method"].items()
            if k.startswith("random_chunk_s")}


def mean_std(vals):
    n = len(vals)
    if n == 0:
        return None, None
    m = sum(vals) / n
    if n == 1:
        return m, 0.0
    return m, (sum((v - m) ** 2 for v in vals) / (n - 1)) ** 0.5


def collapse_random(d):
    """Random across seeds -> one pseudo-method with mean and std."""
    seeds = random_seeds(d)
    if not seeds:
        return None
    out = {"n_seeds": len(seeds)}
    for k in ("acc", "ttft_mean_ms", "disk_mb", "touched_chunk_fraction",
              "selector_ms", "disk_ms", "ttft_p50_ms", "ttft_p95_ms",
              "n_chunks_selected", "acc_delta_vs_fullload_pp",
              "ttft_speedup_vs_fullload", "bytes_vs_fullload"):
        vals = [v[k] for v in seeds.values() if v.get(k) is not None]
        m, s = mean_std(vals)
        if m is not None:
            out[k] = m
            out[k + "_std"] = s
    out["per_seed_acc"] = {k: v["acc"] for k, v in seeds.items()}
    return out


def method_row(summary, key, d_random):
    if key == "random_chunk":
        return d_random
    return summary["per_method"].get(key)


# ------------------------------------------------------------------ tables
def write_table_a(d25):
    ABL.mkdir(parents=True, exist_ok=True)
    rnd = collapse_random(d25)
    pm = d25["summary"]["per_method"]

    # per-method files (13)
    for k, v in random_seeds(d25).items():
        with open(ABL / f"random_seed{k[-1]}.json", "w") as f:
            json.dump(v, f, indent=1)
    for name, key in [("static", "visionzip_static_chunk"),
                      ("diverse", "diverse_chunk"),
                      ("static_diverse", "static_diverse_chunk"),
                      ("hybrid", "cvpr25_hybrid_chunk"),
                      ("hybrid_diverse", "cvpr25_hybrid_diverse"),
                      ("fullload", "fullload")]:
        if key in pm:
            with open(ABL / f"{name}_25.json", "w") as f:
                json.dump(pm[key], f, indent=1)
    with open(ABL / "random_summary.json", "w") as f:
        json.dump(rnd, f, indent=1)

    order = [("FullLoad", "fullload"), ("Random Chunk", "random_chunk"),
             ("VisionZip Static", "visionzip_static_chunk"),
             ("Diverse Only", "diverse_chunk"),
             ("Static + Diverse", "static_diverse_chunk"),
             ("Hybrid", "cvpr25_hybrid_chunk"),
             ("Hybrid + Diverse", "cvpr25_hybrid_diverse")]
    cols = ["method", "accuracy", "d_acc_vs_fullload_pp", "ttft_mean_ms",
            "ttft_p50_ms", "ttft_p95_ms", "speedup_vs_fullload",
            "selector_ms", "selector_p50_ms", "selector_p95_ms", "ssd_mb",
            "ssd_ratio", "ssd_ms", "touched_chunk_fraction",
            "n_chunks_selected", "scatter_ms", "fallback_rate"]
    rows = []
    for name, key in order:
        v = method_row(d25["summary"], key, rnd)
        if v is None:
            continue
        rows.append({
            "method": name, "accuracy": round(v["acc"] * 100, 2),
            "d_acc_vs_fullload_pp": round(v.get("acc_delta_vs_fullload_pp", 0), 2),
            "ttft_mean_ms": round(v["ttft_mean_ms"], 1),
            "ttft_p50_ms": round(v.get("ttft_p50_ms", 0), 1),
            "ttft_p95_ms": round(v.get("ttft_p95_ms", 0), 1),
            "speedup_vs_fullload": round(v.get("ttft_speedup_vs_fullload", 1), 3),
            "selector_ms": round(v.get("selector_ms", v.get("hook_ms", 0)), 2),
            "selector_p50_ms": round(v.get("selector_ms_p50", 0), 2),
            "selector_p95_ms": round(v.get("selector_ms_p95", 0), 2),
            "ssd_mb": round(v.get("disk_mb", 0), 1),
            "ssd_ratio": round(v.get("bytes_vs_fullload", 1), 3),
            "ssd_ms": round(v.get("disk_ms", 0), 1),
            "touched_chunk_fraction": round(v.get("touched_chunk_fraction", 1), 3),
            "n_chunks_selected": round(v.get("n_chunks_selected", 0), 1),
            "scatter_ms": round(v.get("scatter_ms", 0), 1),
            "fallback_rate": round(v.get("fallback_rate") or 0, 3)})
    with open(ABL / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)
    return rows, rnd


def write_table_b(bydg):
    SWEEP.mkdir(parents=True, exist_ok=True)
    cols = ["budget", "method", "accuracy", "ttft_mean_ms", "ssd_mb",
            "ssd_ratio", "touched_chunk_fraction", "selector_ms",
            "n_chunks_selected"]
    rows = []
    for b in sorted(bydg):
        d = bydg[b]
        rnd = collapse_random(d)
        pm = d["summary"]["per_method"]
        for name, key in [("FullLoad", "fullload"),
                          ("Random", "random_chunk"),
                          ("VisionZip Static", "visionzip_static_chunk"),
                          ("Diverse Only", "diverse_chunk"),
                          ("Static + Diverse", "static_diverse_chunk"),
                          ("Hybrid + Diverse", "cvpr25_hybrid_diverse"),
                          ("SparseVLM", "sparsevlm")]:
            v = method_row(d["summary"], key, rnd)
            if v is None:
                continue
            if key not in ("fullload", "random_chunk") and key in pm:
                tag = {"visionzip_static_chunk": "static",
                       "diverse_chunk": "diverse",
                       "static_diverse_chunk": "static_diverse",
                       "cvpr25_hybrid_diverse": "hybrid_diverse",
                       "sparsevlm": "sparsevlm"}[key]
                with open(SWEEP / f"{tag}_{int(b*1000)}.json", "w") as f:
                    json.dump(pm[key], f, indent=1)
            rows.append({
                "budget": b, "method": name,
                "accuracy": round(v["acc"] * 100, 2),
                "ttft_mean_ms": round(v["ttft_mean_ms"], 1),
                "ssd_mb": round(v.get("disk_mb", 0), 1),
                "ssd_ratio": round(v.get("bytes_vs_fullload", 1), 3),
                "touched_chunk_fraction": round(
                    v.get("touched_chunk_fraction", 1), 3),
                "selector_ms": round(v.get("selector_ms",
                                           v.get("hook_ms", 0)), 2),
                "n_chunks_selected": round(v.get("n_chunks_selected", 0), 1)})
    with open(SWEEP / "budget_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)
    return rows


# ----------------------------------------------------------------- figures
def _style(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)


def scatter_fig(rows25, xkey, xlabel, title, out, fullload, xlim=None):
    """Every method as one labelled point, FullLoad included.

    FullLoad is plotted rather than drawn as a bare reference line: at a fixed
    chunk budget every selector reads almost exactly the same bytes, so an axis
    scaled only to the selectors would magnify a 0.008 spread into the full
    width and imply a difference that is not there.  Showing the 1.0 reference
    keeps the axis honest.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=170, facecolor=SURFACE)
    _style(ax, xlabel, "GQA accuracy (%)", title)

    label_to_colour = {n: c for n, (_, c, _m) in METHODS.items()}
    label_to_marker = {n: m for n, (_, _c, m) in METHODS.items()}
    label_to_colour["Random Chunk"] = METHODS["Random"][1]
    label_to_marker["Random Chunk"] = METHODS["Random"][2]
    # A 6th categorical hue fails the all-pairs gates with this set (CVD dE 3.2,
    # normal 12.9), so the diagnostic "Hybrid" point folds into a neutral
    # "other" ink instead of inventing a hue -- it is directly labelled, and
    # the neutral never enters the categorical pairlist.
    label_to_colour["Hybrid"] = OTHER
    label_to_colour["FullLoad"] = REF

    xs = [r[xkey] for r in rows25]
    lo, hi = min(xs), max(xs)
    pad = (hi - lo) * 0.18 or 1.0
    if xlim:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(lo - pad, hi + pad)
    x0, x1 = ax.get_xlim()
    mid = x0 + 0.62 * (x1 - x0)
    ys = [r["accuracy"] for r in rows25]
    y0, y1 = min(ys) - 2.0, max(ys) + 2.0
    ax.set_ylim(y0, y1)
    # At a fixed chunk budget the selectors sit on top of each other in x by
    # construction, so labels have to be de-collided explicitly rather than
    # nudged apart by jittering the data.
    placed = []
    xtol, ytol = (x1 - x0) * 0.08, (y1 - y0) * 0.055
    for r in sorted(rows25, key=lambda r: -r["accuracy"]):
        col = label_to_colour.get(r["method"], OTHER)
        ax.scatter(r[xkey], r["accuracy"], s=115, color=col, zorder=3,
                   marker=label_to_marker.get(r["method"], "o"),
                   edgecolors=SURFACE, linewidths=2)
        right = r[xkey] < mid
        dy = 4
        while any(abs(r[xkey] - px) < xtol
                  and abs((r["accuracy"] + dy * (y1 - y0) / 300) - py) < ytol
                  for px, py in placed):
            dy -= 13
        placed.append((r[xkey], r["accuracy"] + dy * (y1 - y0) / 300))
        ax.annotate(r["method"], (r[xkey], r["accuracy"]), color=INK,
                    fontsize=8.5, ha="left" if right else "right",
                    xytext=(9 if right else -9, dy),
                    textcoords="offset points")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def budget_fig(rowsB, out):
    """Five series: legend only, no per-line end labels.

    The rule is a legend for >= 2 series and direct labels only up to 4; at five
    the end labels collided with each other and with the FullLoad reference, so
    the legend carries identity and marker shape backs up the hue.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 4.8), dpi=170, facecolor=SURFACE)
    _style(ax, "SSD chunk budget (%)", "GQA accuracy (%)",
           "Accuracy vs budget")
    fl = [r for r in rowsB if r["method"] == "FullLoad"]
    if fl:
        ax.axhline(fl[0]["accuracy"], color=REF, lw=1.6, ls=(0, (5, 4)),
                   zorder=1, label=f"FullLoad, 100% ({fl[0]['accuracy']:.1f}%)")
    for name, (key, col, mk) in METHODS.items():
        pts = sorted([(r["budget"] * 100, r["accuracy"]) for r in rowsB
                      if r["method"] == name])
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=col, lw=2, marker=mk, ms=7, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.6, label=name)
    ax.set_xlim(22, 53)
    ax.set_xticks([25, 30, 35, 37.5, 40, 45, 50])
    ax.legend(frameon=False, fontsize=8.5, ncol=3, labelcolor=INK2,
              loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    d25 = load(ABL / "all_25.json")
    rows25, rnd = write_table_a(d25)
    bydg = {0.25: d25}
    for p, b in ((SWEEP / "all_375.json", 0.375), (SWEEP / "all_50.json", 0.5)):
        if p.exists():
            bydg[b] = load(p)
    rowsB = write_table_b(bydg)

    FIGS.mkdir(parents=True, exist_ok=True)
    fl = [r for r in rows25 if r["method"] == "FullLoad"][0]
    scatter_fig(rows25, "ssd_ratio", "SSD bytes read / FullLoad",
                "Accuracy vs SSD read (25% chunk budget)",
                FIGS / "fig1_ssd_vs_accuracy.png", fl, xlim=(0.15, 1.12))
    scatter_fig(rows25, "ttft_mean_ms", "TTFT (ms, mean)",
                "Accuracy vs TTFT (25% chunk budget)",
                FIGS / "fig2_ttft_vs_accuracy.png", fl, xlim=(280, 790))
    budget_fig(rowsB, FIGS / "fig3_budget_vs_accuracy.png")

    print("=== Table A (25% budget) ===")
    hdr = f"{'method':<20}{'acc%':>7}{'dFL':>7}{'TTFT':>8}{'x':>6}{'sel ms':>8}{'MB':>8}{'ratio':>7}{'chunkf':>8}"
    print(hdr)
    for r in rows25:
        print(f"{r['method']:<20}{r['accuracy']:>7.1f}{r['d_acc_vs_fullload_pp']:>7.1f}"
              f"{r['ttft_mean_ms']:>8.1f}{r['speedup_vs_fullload']:>6.2f}"
              f"{r['selector_ms']:>8.1f}{r['ssd_mb']:>8.1f}{r['ssd_ratio']:>7.3f}"
              f"{r['touched_chunk_fraction']:>8.3f}")
    if rnd:
        print(f"\nRandom over {rnd['n_seeds']} seeds: acc {rnd['acc']*100:.1f}%"
              f" +/- {rnd['acc_std']*100:.1f}  TTFT {rnd['ttft_mean_ms']:.1f}"
              f" +/- {rnd['ttft_mean_ms_std']:.1f} ms")
        print("  per seed:", {k: round(v * 100, 1)
                              for k, v in rnd["per_seed_acc"].items()})
    print("\n=== Table B (budget sweep) ===")
    print(f"{'budget':>8}{'method':<20}{'acc%':>7}{'TTFT':>8}{'MB':>8}{'chunkf':>8}")
    for r in rowsB:
        print(f"{r['budget']*100:>7.1f}%{r['method']:<20}{r['accuracy']:>7.1f}"
              f"{r['ttft_mean_ms']:>8.1f}{r['ssd_mb']:>8.1f}"
              f"{r['touched_chunk_fraction']:>8.3f}")
    print(f"\nwrote {ABL/'summary.csv'}, {SWEEP/'budget_summary.csv'}, {FIGS}/*.png")


if __name__ == "__main__":
    main()

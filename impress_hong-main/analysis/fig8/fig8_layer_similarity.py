"""Reproduce IMPRESS Figure 8: average head-pair Jaccard similarity of
important-token index sets, PER LAYER, at top 10% and top 40%.

Paper (§4.2 Observation II): "we study the average similarity of token
index sets across different heads for the OPT-6.7B, OPT-13B, and OPT-30B
models, when selecting the top 10% and 40% most important tokens (Figure
8). Each bar ... represents the average value for a single transformer
layer." Findings to check: (1) higher ratio -> greater similarity;
(2) smaller models / deeper layers -> lower, but "still significantly
higher than the expected value from random selection in most cases".

This machine has OPT-6.7B only -> one-model panel; the paper's OPT-30B
all-layer averages (0.68 @40%, 0.48 @10%) and the random-selection
expectation E(J) = r/(2-r) are drawn as reference lines.

Reuses the fig7b importance-score cache (no GPU needed if present).
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/dblab/hj/FlexGen")
sys.path.insert(0, "/home/dblab/hj/analysis/fig7b")
from fig7b_jaccard import (topk_sets, jaccard_matrix, offdiag_mean,
    capture_scores, BLUE_RAMP, C_SURFACE, C_TEXT, C_TEXT2)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
RATIOS = [0.10, 0.40]                     # paper Figure 8 (a), (b)
PAPER_30B = {0.40: 0.68, 0.10: 0.48}      # §4.2: OPT-30B all-layer averages
C_BAR = "#2a78d6"
C_REF = "#52514e"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-6.7b")
    parser.add_argument("--path", type=str, default="~/opt_weights")
    parser.add_argument("--offload-dir", type=str,
                        default="~/flexllmgen_offload_dir")
    parser.add_argument("--scores-cache", type=str,
        default="/home/dblab/hj/analysis/fig7b/output/scores_21shot_seed0.pt",
        help="fig7b importance cache; if missing, captured on GPU")
    parser.add_argument("--num-shots", type=int, default=21)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--query-cache", type=str, default=None,
        help="use a QUERY-conditioned score cache (fig7b_query .pt, a "
             "per-query list) instead of the prefix-only cache")
    parser.add_argument("--query-idx", type=int, default=0,
        help="with --query-cache: 0 = mean over queries, N = query N only")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    mode_tag = "prefix-only"
    if args.query_cache:
        per_query = torch.load(args.query_cache)
        use_q = (per_query if args.query_idx == 0
                 else [per_query[args.query_idx - 1]])
        s = per_query[0][0].shape[1]
        mode_tag = ("query-conditioned, mean of %d queries" % len(use_q)
                    if args.query_idx == 0
                    else "query-conditioned, query %d only" % args.query_idx)
        scores = None
        print(f"query-conditioned cache: {args.query_cache} ({mode_tag}); "
              f"prefix {s} tokens — no GPU needed")
    elif os.path.exists(args.scores_cache):
        blob = torch.load(args.scores_cache)
        scores, s = blob["scores"], blob["prefix_len"]
        print(f"scores loaded from cache ({args.scores_cache}); "
              f"prefix {s} tokens — no GPU needed")
    else:
        from transformers import AutoTokenizer
        from flexllmgen.impress.verify_impress_e2e import build_workload
        tok = AutoTokenizer.from_pretrained("facebook/opt-30b",
                                            padding_side="left")
        prefix_texts, _ = build_workload(4, args.num_shots, args.seed)
        ids = tok(prefix_texts[0]).input_ids
        scores, _ = capture_scores(args.model, args.path, args.offload_dir,
                                   ids)
        s = len(ids)
    num_layers = 32 if scores is None else len(scores)

    per_layer = {r: [] for r in RATIOS}
    for j in range(num_layers):
        for r in RATIOS:
            if scores is None:  # query-conditioned: mean over queries
                vals = []
                for sq in use_q:
                    sets, _ = topk_sets(sq[j], r)
                    M = jaccard_matrix(sets)
                    assert np.allclose(np.diag(M), 1.0)
                    vals.append(offdiag_mean(M))
                per_layer[r].append(float(np.mean(vals)))
            else:
                sets, _ = topk_sets(scores[j], r)
                M = jaccard_matrix(sets)
                assert np.allclose(np.diag(M), 1.0)
                per_layer[r].append(offdiag_mean(M))

    with open(os.path.join(OUT_DIR, "fig8_per_layer.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer_1idx"] + [f"avg_r{int(r*100)}" for r in RATIOS])
        for j in range(num_layers):
            w.writerow([j + 1] + [round(per_layer[r][j], 4) for r in RATIOS])

    print(f"\nOPT-6.7B ({num_layers} layers), prefix {s} tokens:")
    for r in RATIOS:
        vals = per_layer[r]
        exp_j = r / (2.0 - r)
        print(f"  top {r:4.0%}: all-layer avg {np.mean(vals):.3f} "
              f"(min L{int(np.argmin(vals))+1}={min(vals):.3f}, "
              f"max L{int(np.argmax(vals))+1}={max(vals):.3f}; "
              f"E(J)_random={exp_j:.3f}; paper OPT-30B avg="
              f"{PAPER_30B[r]:.2f})")

    # ---- Figure 8-style two-panel bar chart ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"text.color": C_TEXT, "axes.labelcolor": C_TEXT2,
                         "font.size": 10, "figure.facecolor": C_SURFACE,
                         "savefig.facecolor": C_SURFACE})
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.0), sharex=True)
    for ax, r, tag in zip(axes, RATIOS, ("(a)", "(b)")):
        vals = per_layer[r]
        ax.set_facecolor(C_SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(True, axis="y", color="#e3e2dd", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.bar(np.arange(1, num_layers + 1), vals, color=C_BAR,
               edgecolor=C_SURFACE, linewidth=0.8)
        ax.axhline(PAPER_30B[r], color=C_REF, linestyle="--", linewidth=1.2)
        ax.text(num_layers + 0.4, PAPER_30B[r], " paper 30B avg",
                color=C_TEXT2, fontsize=8, va="center")
        ax.axhline(float(np.mean(vals)), color=C_BAR, linestyle="--",
                   linewidth=1.0)
        ax.text(num_layers + 0.4, float(np.mean(vals)), " 6.7B avg (ours)",
                color=C_BAR, fontsize=8, va="center")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Similarity (Jaccard)", fontsize=10)
        ax.set_title(f"{tag} select the top {r:.0%} most important tokens",
                     loc="left", fontsize=11, color=C_TEXT)
        ax.tick_params(colors=C_TEXT2, labelsize=8)
    axes[1].set_xlabel("Layer (1-indexed)")
    axes[1].set_xticks(np.arange(0, num_layers + 1, 4))
    fig.suptitle(f"Per-layer head-pair similarity ({mode_tag}) — "
                 f"OPT-6.7B, RTE prefix {s} tokens (cf. IMPRESS Fig. 8)",
                 fontsize=11, color=C_TEXT)
    fig.tight_layout()
    qtag = ("" if not args.query_cache
            else ("_query" if args.query_idx == 0
                  else f"_query_q{args.query_idx}"))
    out = os.path.join(OUT_DIR, f"fig8_opt6.7b{qtag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    print("DONE")


if __name__ == "__main__":
    main()

"""Reproduce IMPRESS Figure 7(b): head-pair Jaccard similarity heatmap of
important-token index sets, middle transformer layer of OPT-6.7B.

Paper grounding (fast25-chen-weijian-impress.txt):
- §4.1: importance = column sums of the post-softmax attention weight
  matrix (H2O). Computed here for ALL heads (no probe approximation).
- §4.2 Observation I / Figure 7: J(A,B) = |A n B| / |A u B|; Figure 7(b)
  is "the important token sets in the keys produced by the middle
  transformer layer of the OPT-6.7B model", average > 0.95; colorbar
  0.8-1.0.

Approved assumptions:
- layers: 1, 16, 32 in 1-indexed naming = 0-indexed 0, 15, 31 (all-layer
  averages also reported, captured in the same forward)
- ratios: primary 50% (Figure 7(a) basis); 10% / 40% (Figure 8 values)
  reported alongside
- input: one real RTE 21-shot prefix (~1.65k tokens, seed 0) from the
  existing reproduction pipeline; <= 2048 so NO position interpolation
  is involved (pure trained-model attention)

Read-only reuse of flexllmgen.impress.importance (Stage 3 hook); no
IMPRESS code is modified.
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/dblab/hj/FlexGen")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
LAYERS_1IDX = [1, 16, 32]           # user-selected; 0-indexed: 0, 15, 31
RATIOS = [0.50, 0.10, 0.40]         # primary first

# dataviz reference palette: sequential blue ramp (light -> dark)
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
C_SURFACE, C_TEXT, C_TEXT2 = "#fcfcfb", "#0b0b0b", "#52514e"


def capture_scores(model_name, weights_path, offload_dir, prefix_ids):
    """One prefill; returns {layer: (n_head, s) float32} importance."""
    from flexllmgen.compression import CompressionConfig
    from flexllmgen.flex_opt import Policy, OptLM, get_opt_config
    from flexllmgen.pytorch_backend import (TorchDevice, TorchDisk,
        TorchMixedDevice)
    from flexllmgen.utils import ExecutionEnv
    from flexllmgen.impress.importance import (install_importance_hook,
        uninstall_importance_hook)

    config = get_opt_config(model_name)
    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk,
                       mixed=TorchMixedDevice([gpu, cpu, disk]))
    policy = Policy(1, 1, 100, 0, 100, 0, 100, 0,
                    overlap=False, sep_layer=True, pin_weight=True,
                    cpu_cache_compute=False, attn_sparsity=1.0,
                    compress_weight=False,
                    comp_weight_config=CompressionConfig(
                        num_bits=4, group_size=64, group_dim=0, symmetric=False),
                    compress_cache=False,
                    comp_cache_config=CompressionConfig(
                        num_bits=4, group_size=64, group_dim=2, symmetric=False))
    print(f"init weights: {model_name} ...")
    model = OptLM(config, env, weights_path, policy)
    collector = install_importance_hook()
    try:
        collector.enabled = True
        model.generate((list(prefix_ids),), max_new_tokens=1, do_sample=False)
        collector.enabled = False
        scores = {j: s[0].float() for j, s in collector.scores.items()}
        assert len(scores) == config.num_hidden_layers
        return scores, config.n_head
    finally:
        uninstall_importance_hook()
        env.close_copy_threads()


def topk_sets(layer_scores, ratio):
    """(n_head, s) scores -> list of top-k index sets per head."""
    n_head, s = layer_scores.shape
    k = max(1, int(round(s * ratio)))
    return [set(torch.topk(layer_scores[h], k).indices.tolist())
            for h in range(n_head)], k


def jaccard_matrix(sets):
    H = len(sets)
    M = np.zeros((H, H))
    for i in range(H):
        for j in range(H):
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            M[i, j] = inter / union if union else 1.0
    return M


def offdiag_mean(M):
    H = M.shape[0]
    mask = ~np.eye(H, dtype=bool)
    return float(M[mask].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-6.7b")
    parser.add_argument("--path", type=str, default="~/opt_weights")
    parser.add_argument("--offload-dir", type=str,
                        default="~/flexllmgen_offload_dir")
    parser.add_argument("--num-shots", type=int, default=21)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ratios", type=str, default="0.5,0.1,0.4",
        help="comma list; first = primary (used for figures)")
    parser.add_argument("--vmin", type=float, default=0.8)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--grid-all", action="store_true",
        help="also render a grid heatmap of ALL layers at the primary ratio")
    parser.add_argument("--prefix-file", type=str, default=None,
        help="use this text file as the prefix instead of the built-in "
             "RTE few-shot prompt (for CoT / document-augmented variants); "
             "truncated to 2048 tokens (no position extension here)")
    parser.add_argument("--scores-cache", type=str, default="auto",
        help="path to cache the captured importance scores; 'auto' = "
             "output/scores_<shots>shot_seed<seed>.pt; 'off' = disable. "
             "With a cache hit the model is NOT loaded (ratio/scale "
             "changes run in seconds on CPU)")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    # input: real RTE 21-shot prefix from the reproduction pipeline
    from transformers import AutoTokenizer
    from flexllmgen.impress.verify_impress_e2e import build_workload
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b",
                                              padding_side="left")
    if args.prefix_file:
        import hashlib
        text = open(args.prefix_file).read()
        prefix_ids = tokenizer(text).input_ids
        if len(prefix_ids) > 2048:
            print(f"[warn] prefix {len(prefix_ids)} tokens > 2048; "
                  f"truncating (no position extension in this analysis)")
            prefix_ids = prefix_ids[:2048]
        tag = hashlib.md5(text.encode()).hexdigest()[:8]
        s = len(prefix_ids)
        print(f"input: custom prefix from {args.prefix_file}, {s} tokens "
              f"(md5 {tag})")
    else:
        prefix_texts, _ = build_workload(4, args.num_shots, args.seed)
        prefix_ids = tokenizer(prefix_texts[0]).input_ids
        s = len(prefix_ids)
        print(f"input: RTE {args.num_shots}-shot prefix, {s} tokens "
              f"(seed {args.seed}; <= 2048, no position interpolation)")

    ratios = [float(r) for r in args.ratios.split(",")]
    primary = ratios[0]
    if args.scores_cache == "auto":
        name = (f"scores_file_{tag}.pt" if args.prefix_file
                else f"scores_{args.num_shots}shot_seed{args.seed}.pt")
        cache_path = os.path.join(OUT_DIR, name)
    else:
        cache_path = args.scores_cache
    if cache_path != "off" and os.path.exists(cache_path):
        blob = torch.load(cache_path)
        assert blob["prefix_len"] == s, "cache from a different prefix"
        scores, n_head = blob["scores"], blob["n_head"]
        print(f"scores loaded from cache: {cache_path} (no GPU needed)")
    else:
        scores, n_head = capture_scores(args.model, args.path,
                                        args.offload_dir, prefix_ids)
        if cache_path != "off":
            torch.save(dict(scores=scores, n_head=n_head, prefix_len=s,
                            num_shots=args.num_shots, seed=args.seed),
                       cache_path)
            print(f"scores cached to {cache_path}")
    num_layers = len(scores)

    # ---- all-layer / all-ratio average table ----
    rows = []
    mats = {}   # (layer0idx, ratio) -> matrix
    sel_0idx = [l - 1 for l in LAYERS_1IDX]
    for j in range(num_layers):
        row = {"layer_1idx": j + 1}
        for r in ratios:
            sets, k = topk_sets(scores[j], r)
            M = jaccard_matrix(sets)
            assert np.allclose(np.diag(M), 1.0), f"diagonal != 1 at L{j}"
            row[f"avg_r{int(r*100)}"] = round(offdiag_mean(M), 4)
            if j in sel_0idx or r == primary:
                mats[(j, r)] = M
        rows.append(row)
    with open(os.path.join(OUT_DIR, "avg_jaccard_all_layers.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    hdr = " ".join(f"r={int(r*100)}%".rjust(7) for r in ratios)
    print(f"\nper-layer mean off-diagonal Jaccard (n_head={n_head}):")
    print(f"{'layer(1idx)':>12s} {hdr}")
    for row in rows:
        mark = " <-- selected" if row["layer_1idx"] in LAYERS_1IDX else ""
        vals = " ".join(f"{row[f'avg_r{int(r*100)}']:7.3f}" for r in ratios)
        print(f"{row['layer_1idx']:>12d} {vals}{mark}")
    for r in ratios:
        grand = float(np.mean([row[f"avg_r{int(r*100)}"] for row in rows]))
        exp_j = r / (2.0 - r)
        print(f"  r={int(r*100)}%: all-layer avg {grand:.3f} "
              f"(random-selection expectation E(J)={exp_j:.3f})")

    # ---- heatmaps: selected layers at the primary ratio (50%) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("impress_blue", BLUE_RAMP)
    plt.rcParams.update({"text.color": C_TEXT, "axes.labelcolor": C_TEXT2,
                         "font.size": 10, "figure.facecolor": C_SURFACE,
                         "savefig.facecolor": C_SURFACE})

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))
    for ax, l1 in zip(axes, LAYERS_1IDX):
        M = mats[(l1 - 1, primary)]
        im = ax.imshow(M, cmap=cmap, vmin=args.vmin, vmax=args.vmax, origin="upper")
        ax.set_title(f"Layer {l1} (idx {l1 - 1})  "
                     f"avg={offdiag_mean(M):.3f}", fontsize=11, color=C_TEXT)
        ax.set_xlabel("Head Index")
        ax.set_ylabel("Head Index")
        ax.tick_params(colors=C_TEXT2, labelsize=8)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("Jaccard similarity", color=C_TEXT2)
    cb.ax.tick_params(colors=C_TEXT2)
    fig.suptitle(f"Important-token index-set similarity across heads "
                 f"(OPT-6.7B, top {primary:.0%}, RTE prefix "
                 f"{s} tokens)",
                 fontsize=12, color=C_TEXT, y=1.02)
    out = os.path.join(OUT_DIR, f"fig7b_layers_1_16_32_r{int(primary*100)}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # single-layer primary figure (middle layer, paper-style)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    M = mats[(15, primary)]
    im = ax.imshow(M, cmap=cmap, vmin=args.vmin, vmax=args.vmax, origin="upper")
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Head Index")
    ax.set_title(f"OPT-6.7B middle layer (L16), top {primary:.0%} — "
                 f"avg={offdiag_mean(M):.3f}", fontsize=11, color=C_TEXT)
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label("Jaccard similarity", color=C_TEXT2)
    fig.tight_layout()
    out2 = os.path.join(OUT_DIR, f"fig7b_L16_r{int(primary*100)}.png")
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"saved {out2}")

    if args.grid_all:
        fig, axes = plt.subplots(4, 8, figsize=(22, 11.5))
        for j in range(num_layers):
            ax = axes[j // 8][j % 8]
            M = mats[(j, primary)]
            im = ax.imshow(M, cmap=cmap, vmin=args.vmin, vmax=args.vmax,
                           origin="upper")
            ax.set_title(f"L{j + 1}  avg={offdiag_mean(M):.3f}",
                         fontsize=9, color=C_TEXT)
            ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01)
        cb.set_label("Jaccard similarity", color=C_TEXT2)
        cb.ax.tick_params(colors=C_TEXT2)
        fig.suptitle(f"All 32 layers, top {primary:.0%}, scale "
                     f"[{args.vmin:g}, {args.vmax:g}] — OPT-6.7B, RTE "
                     f"prefix {s} tokens", fontsize=13, color=C_TEXT)
        out3 = os.path.join(OUT_DIR,
                            f"fig7b_all_layers_r{int(primary*100)}.png")
        fig.savefig(out3, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out3}")
    print("DONE")


if __name__ == "__main__":
    main()

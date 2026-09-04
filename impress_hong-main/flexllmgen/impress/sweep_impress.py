"""Hyperparameter sweep over the integrated IMPRESS serving mode (Stage 9).

Sweeps, one dimension at a time around the defaults (alpha 0.6, 3 probe
heads, chunk_size 64, retention 25%):
  alpha            0.0 .. 1.2   (paper Figure 22: TTFT + accuracy vs alpha;
                                 alpha 0 -> threshold 1 -> always fallback)
  probe_head_count 1 .. 5       (paper §4.3 hyperparameter discussion)
  chunk_size       16 .. 256    (paper Figure 23: TTFT vs chunk size,
                                 IMPRESS vs a full-disk-load baseline)

Per configuration (fresh radix tree; 2 cold requests populate it, warm
requests are measured): accuracy, warm mean TTFT, ITF-loaded vectors and
their ratio vs the naive all-heads count (paper Figure 11's "load ratio"),
fallback rate, chunk-tier traffic. ReComp is measured once as a reference.

Outputs (to --out-dir):
  sweep_results.csv                 raw table
  sweep_alpha.png                   Figure 22/11-style panels
  sweep_probe.png                   probe-head-count panels
  sweep_chunk.png                   Figure 23-style TTFT bars
  (single-axis panels are used instead of the paper's dual-axis layout)

Usage:
  python -m flexllmgen.impress.sweep_impress --num-samples 20
"""

import argparse
import csv
import os
import shutil
import time

import numpy as np
import torch

from flexllmgen.compression import CompressionConfig
from flexllmgen.flex_opt import Policy, OptLM, get_opt_config
from flexllmgen.pytorch_backend import (TorchDevice, TorchDisk,
    TorchMixedDevice)
from flexllmgen.utils import ExecutionEnv
from flexllmgen.impress.impress_serving import ImpressConfig, ImpressServer
from flexllmgen.impress.verify_impress_e2e import build_workload

DEFAULTS = dict(alpha=0.6, probe=3, chunk=64)
ALPHA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
PROBE_GRID = [1, 2, 3, 4, 5]
CHUNK_GRID = [16, 32, 64, 128, 256]

# dataviz reference palette (validated order; adjacent slots only)
C_SERIES1 = "#2a78d6"   # IMPRESS (blue, slot 1)
C_SERIES2 = "#1baf7a"   # FullLoad (aqua, slot 2)
C_REF = "#52514e"       # ReComp reference line (secondary ink, dashed)
C_SURFACE = "#fcfcfb"
C_GRID = "#e3e2dd"
C_TEXT = "#0b0b0b"
C_TEXT2 = "#52514e"


def styled_axes(ax):
    ax.set_facecolor(C_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_GRID)
    ax.grid(True, axis="y", color=C_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=C_TEXT2, labelsize=9)


def run_config(server, requests, true_id, false_id, fullload_requests=0):
    """Serve the workload on one configuration; measure warm requests."""
    ctrl = server.controller
    n_head = server.opt_config.n_head
    num_layers = server.opt_config.num_hidden_layers

    def pred(logits):
        return 0 if logits[true_id] > logits[false_id] else 1

    preds, warm_ttfts, naive_vectors = [], [], 0
    for i, (p_ids, q_ids, _) in enumerate(requests):
        logits, ttft, r_len, nr_len = server.request(p_ids, q_ids)
        preds.append(pred(logits))
        if i == 1:
            # both prefixes are now inserted: measure only warm requests
            ctrl.io.reset()
            for k in server.manager.stats:
                server.manager.stats[k] = 0
        if nr_len == 0:
            warm_ttfts.append(ttft)
            k_keep = max(1, int(round(r_len * ctrl.retention_ratio)))
            naive_vectors += num_layers * (n_head * r_len + n_head * k_keep)

    acc = float(np.mean([p == r[2] for p, r in zip(preds, requests)]))
    res = dict(
        acc=acc,
        ttft_ms=float(np.mean(warm_ttfts)) * 1000,
        ttft_std_ms=float(np.std(warm_ttfts)) * 1000,
        vectors=ctrl.io.total_vectors,
        load_ratio=ctrl.io.total_vectors / max(naive_vectors, 1),
        fallback=ctrl.io.fallback_rate(),
        disk_chunks=server.manager.stats["disk_chunk_loads"],
        pcie_chunks=server.manager.stats["pcie_chunk_transfers"],
        fullload_ttft_ms=None,
    )

    if fullload_requests > 0:  # Figure 23 baseline on this chunk size
        gpu_dev = server.dev
        ctrl.selected_by_layer = None
        ctrl.capture_new_kv = False
        old_ratio = ctrl.retention_ratio
        ctrl.retention_ratio = 1.0
        ts = []
        for p_ids, q_ids, _ in requests[:fullload_requests]:
            t0 = time.perf_counter()
            m = server.tree.match(p_ids)
            ctrl.prefix_kv = server.tree.load_prefix_kv(m, device=gpu_dev)
            ctrl.prefix_len = m.r_len
            ctrl.enabled = True
            server.model.generate((list(q_ids),), max_new_tokens=1,
                                  do_sample=False)
            ts.append(time.perf_counter() - t0)
            ctrl.enabled = False
        ctrl.retention_ratio = old_ratio
        res["fullload_ttft_ms"] = float(np.mean(ts)) * 1000
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-6.7b")
    parser.add_argument("--w-gpu-percent", type=int, default=100,
        help="weight %% resident on GPU (13B: ~70, 30B: ~30 on a 24GB GPU)")
    parser.add_argument("--w-cpu-percent", type=int, default=0,
        help="weight %% on CPU (rest goes to disk)")
    parser.add_argument("--path", type=str, default="~/opt_weights")
    parser.add_argument("--offload-dir", type=str,
                        default="~/flexllmgen_offload_dir")
    parser.add_argument("--work-dir", type=str,
                        default="/home/dblab/hj/impress_kv_store/sweep")
    parser.add_argument("--out-dir", type=str,
                        default="/home/dblab/hj/impress_sweep")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--num-shots", type=int, default=21)
    parser.add_argument("--retention", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b",
                                              padding_side="left")
    true_id = tokenizer(" True", add_special_tokens=False).input_ids[0]
    false_id = tokenizer(" False", add_special_tokens=False).input_ids[0]
    prefix_texts, samples = build_workload(args.num_samples, args.num_shots,
                                           args.seed)
    prefixes = [tokenizer(t).input_ids for t in prefix_texts]
    from flexllmgen.impress.verify_selection_rte import format_rte
    requests = []
    for i, ex in enumerate(samples):
        q = tokenizer("\n\n" + format_rte(ex),
                      add_special_tokens=False).input_ids
        requests.append((prefixes[i % 2], q, ex["label"]))
    print(f"workload: prefixes {len(prefixes[0])}/{len(prefixes[1])} tokens, "
          f"{len(requests)} requests; retention {args.retention:.0%}")

    config = get_opt_config(args.model)
    total_vectors = ((len(prefixes[0]) + len(prefixes[1]))
                     * config.num_hidden_layers * 2)
    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk,
                       mixed=TorchMixedDevice([gpu, cpu, disk]))
    policy = Policy(1, 1, args.w_gpu_percent, args.w_cpu_percent,
                    100, 0, 100, 0,
                    overlap=False, sep_layer=True, pin_weight=True,
                    cpu_cache_compute=False, attn_sparsity=1.0,
                    compress_weight=False,
                    comp_weight_config=CompressionConfig(
                        num_bits=4, group_size=64, group_dim=0, symmetric=False),
                    compress_cache=False,
                    comp_cache_config=CompressionConfig(
                        num_bits=4, group_size=64, group_dim=2, symmetric=False))
    print(f"init weights: {args.model} ...")
    model = OptLM(config, env, args.path, policy)

    def make_server(alpha, probe, chunk, tag):
        tree_dir = os.path.join(args.work_dir, f"tree_{tag}")
        shutil.rmtree(tree_dir, ignore_errors=True)
        cfg = ImpressConfig(
            retention_ratio=args.retention, alpha=alpha,
            probe_head_count=probe, chunk_size=chunk,
            gpu_cache_capacity=int(total_vectors * 0.6),
            cpu_cache_capacity=total_vectors,
            reorder_every_n_requests=10)
        return ImpressServer(model, config, tree_dir, cfg,
                             gpu_device=gpu.dev), tree_dir

    rows = []
    try:
        # ---------------- ReComp reference (measured once) ----------------
        server, tree_dir = make_server(**{"alpha": DEFAULTS["alpha"],
                                          "probe": DEFAULTS["probe"],
                                          "chunk": DEFAULTS["chunk"],
                                          "tag": "recomp"})
        ctrl = server.controller
        ts, preds = [], []
        for p_ids, q_ids, label in requests:
            t0 = time.perf_counter()
            model.generate((list(p_ids) + list(q_ids),), max_new_tokens=1,
                           do_sample=False)
            ts.append(time.perf_counter() - t0)
            preds.append(0 if ctrl.last_logits[0][true_id]
                         > ctrl.last_logits[0][false_id] else 1)
        recomp = dict(ttft_ms=float(np.mean(ts)) * 1000,
                      acc=float(np.mean([p == r[2]
                                         for p, r in zip(preds, requests)])))
        server.close()
        shutil.rmtree(tree_dir, ignore_errors=True)
        print(f"ReComp reference: acc {recomp['acc']*100:.1f}%, "
              f"TTFT {recomp['ttft_ms']:.1f} ms")

        # --------------------------- sweeps -------------------------------
        sweeps = ([("alpha", a, a, DEFAULTS["probe"], DEFAULTS["chunk"])
                   for a in ALPHA_GRID]
                  + [("probe", p, DEFAULTS["alpha"], p, DEFAULTS["chunk"])
                     for p in PROBE_GRID]
                  + [("chunk", c, DEFAULTS["alpha"], DEFAULTS["probe"], c)
                     for c in CHUNK_GRID])
        for sweep, value, alpha, probe, chunk in sweeps:
            server, tree_dir = make_server(alpha, probe, chunk,
                                           f"{sweep}_{value}")
            fl = 8 if sweep == "chunk" else 0
            r = run_config(server, requests, true_id, false_id,
                           fullload_requests=fl)
            server.close()
            shutil.rmtree(tree_dir, ignore_errors=True)
            r.update(sweep=sweep, value=value)
            rows.append(r)
            extra = (f", fullload {r['fullload_ttft_ms']:.1f} ms"
                     if r["fullload_ttft_ms"] else "")
            print(f"  {sweep}={value:<5}: acc {r['acc']*100:5.1f}%, "
                  f"TTFT {r['ttft_ms']:6.1f} ms, load ratio "
                  f"{r['load_ratio']*100:5.1f}%, fallback "
                  f"{r['fallback']*100:5.1f}%{extra}")
    finally:
        env.close_copy_threads()

    # ------------------------------ CSV ---------------------------------
    csv_path = os.path.join(args.out_dir, "sweep_results.csv")
    cols = ["sweep", "value", "acc", "ttft_ms", "ttft_std_ms", "vectors",
            "load_ratio", "fallback", "disk_chunks", "pcie_chunks",
            "fullload_ttft_ms"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow({"sweep": "recomp", "value": "",
                    "acc": recomp["acc"], "ttft_ms": recomp["ttft_ms"]})
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})

    # ------------------------------ plots --------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"text.color": C_TEXT, "axes.labelcolor": C_TEXT2,
                         "font.size": 10, "figure.facecolor": C_SURFACE,
                         "savefig.facecolor": C_SURFACE})

    def panel_plot(sub, xs, series_label, fname, title, xlabel,
                   load_panel=True):
        n_panels = 3 if load_panel else 2
        fig, axes = plt.subplots(n_panels, 1, figsize=(5.4, 2.1 * n_panels),
                                 sharex=True)
        # TTFT panel
        ax = axes[0]
        styled_axes(ax)
        ax.plot(xs, [r["ttft_ms"] for r in sub], color=C_SERIES1,
                linewidth=2, marker="o", markersize=6)
        ax.axhline(recomp["ttft_ms"], color=C_REF, linestyle="--",
                   linewidth=1.2)
        ax.text(xs[-1], recomp["ttft_ms"], " ReComp", color=C_TEXT2,
                va="bottom", ha="right", fontsize=9)
        ax.text(xs[0], sub[0]["ttft_ms"], "IMPRESS ", color=C_SERIES1,
                va="bottom", ha="left", fontsize=9)
        ax.set_ylabel("warm TTFT (ms)")
        ax.set_ylim(bottom=0)
        ax.set_title(title, loc="left", fontsize=11, color=C_TEXT)
        # accuracy panel
        ax = axes[1]
        styled_axes(ax)
        ax.plot(xs, [r["acc"] * 100 for r in sub], color=C_SERIES1,
                linewidth=2, marker="o", markersize=6)
        ax.axhline(recomp["acc"] * 100, color=C_REF, linestyle="--",
                   linewidth=1.2)
        ax.text(xs[-1], recomp["acc"] * 100, " ReComp", color=C_TEXT2,
                va="bottom", ha="right", fontsize=9)
        ax.set_ylabel("accuracy (%)")
        lo = min(min(r["acc"] for r in sub), recomp["acc"]) * 100 - 8
        ax.set_ylim(max(0, lo), 100)
        if load_panel:
            ax = axes[2]
            styled_axes(ax)
            ax.plot(xs, [r["load_ratio"] * 100 for r in sub],
                    color=C_SERIES1, linewidth=2, marker="o", markersize=6)
            ax.set_ylabel("keys+values loaded\nvs naive (%)")
            ax.set_ylim(0, 110)
        axes[-1].set_xlabel(xlabel)
        axes[-1].set_xticks(xs)
        fig.tight_layout()
        out = os.path.join(args.out_dir, fname)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"saved {out}")

    alpha_rows = [r for r in rows if r["sweep"] == "alpha"]
    panel_plot(alpha_rows, [r["value"] for r in alpha_rows], "IMPRESS",
               "sweep_alpha.png",
               "Alpha sweep (threshold = j^alpha) — cf. paper Fig. 22/11",
               "alpha")

    probe_rows = [r for r in rows if r["sweep"] == "probe"]
    panel_plot(probe_rows, [r["value"] for r in probe_rows], "IMPRESS",
               "sweep_probe.png",
               "Probe head count sweep — cf. paper §4.3",
               "probe heads per layer")

    # Figure 23-style grouped bars: TTFT vs chunk size
    chunk_rows = [r for r in rows if r["sweep"] == "chunk"]
    xs = np.arange(len(chunk_rows))
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    styled_axes(ax)
    width = 0.38
    b1 = ax.bar(xs - width / 2, [r["fullload_ttft_ms"] for r in chunk_rows],
                width, color=C_SERIES2, label="FullLoad (no filtering)",
                edgecolor=C_SURFACE, linewidth=2)
    b2 = ax.bar(xs + width / 2, [r["ttft_ms"] for r in chunk_rows],
                width, color=C_SERIES1, label="IMPRESS",
                edgecolor=C_SURFACE, linewidth=2)
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.0f", fontsize=8, color=C_TEXT2, padding=2)
    ax.axhline(recomp["ttft_ms"], color=C_REF, linestyle="--", linewidth=1.2)
    ax.text(xs[-1] + width, recomp["ttft_ms"], " ReComp", color=C_TEXT2,
            va="bottom", ha="right", fontsize=9)
    ax.set_ylabel("warm TTFT (ms)")
    ax.set_title("Chunk size sweep — cf. paper Fig. 23", loc="left",
                 fontsize=11, color=C_TEXT)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax = axes[1]
    styled_axes(ax)
    ax.plot(xs, [r["acc"] * 100 for r in chunk_rows], color=C_SERIES1,
            linewidth=2, marker="o", markersize=6)
    ax.axhline(recomp["acc"] * 100, color=C_REF, linestyle="--",
               linewidth=1.2)
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(max(0, min(min(r["acc"] for r in chunk_rows),
                           recomp["acc"]) * 100 - 8), 100)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(r["value"]) for r in chunk_rows])
    ax.set_xlabel("chunk size (tokens)")
    fig.tight_layout()
    out = os.path.join(args.out_dir, "sweep_chunk.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")
    print(f"saved {csv_path}")
    print("DONE")


if __name__ == "__main__":
    main()

"""Stage 10 verification: selective chunk loading.

Part 1 (CPU, synthetic): the chunk-skip path must be NUMERICALLY EQUIVALENT
  to the Stage-8 path (load all of R, filter on GPU) — same selected data,
  different route — and chunks without selected tokens must never be opened.
Part 2 (GPU): physical I/O vs retention 50/25/10% against FullLoad, with
  caches disabled so every needed chunk is read from disk. Files opened,
  bytes read and disk time must decrease monotonically with retention.
Part 3 (GPU): alpha sweep on the new path — disk read time should now be
  monotone in alpha (the old sweep's TTFT jittered 60-103 ms with no trend).
Part 4 (GPU): Stage-8 3-mode TTFT comparison redone with a Figure-17-style
  I/O-time breakdown.

Measurements run with reorder_every_n_requests=8 and the measured window
starting at request 13 — after the background KV reorderer has packed
important tokens into dense chunks, which (paper Figure 12) is what makes
chunk skipping possible at all. Disk reads are page-cache warm, so absolute
disk times understate a cold SSD; relative comparisons are the point.

Usage:
  python -m flexllmgen.impress.verify_selective_loading            # all
  python -m flexllmgen.impress.verify_selective_loading --parts 1
"""

import argparse
import os
import shutil
import tempfile
import time

import numpy as np
import torch

from flexllmgen.impress.impress_serving import (ImpressConfig, ImpressServer,
    ChunkKVManager)
from flexllmgen.impress.radix_tree import PrefixRadixTree
from flexllmgen.impress.selective_loading import (PathKVProvider,
    mha_selective)
from flexllmgen.impress.similarity_guided import (SimGuidedConfig, IOCounter,
    mha_prefix_reuse_sim)
from flexllmgen.pytorch_backend import TorchDevice, TorchTensor

MEASURE_FROM = 12  # first measured request (post-reorder; reorder every 8)


# ======================================================= Part 1: equivalence
def tt(dev, t):
    return TorchTensor.create_from_torch(t, dev)


def _p1_weights(dev, g, n_head, head_dim, identity_kq):
    H = n_head * head_dim

    def w(*shape):
        return tt(dev, torch.randn(*shape, generator=g) * 0.1)
    wd = dict(w_q=w(H, H), b_q=tt(dev, torch.zeros(H)),
              w_k=w(H, H), b_k=tt(dev, torch.zeros(H)),
              w_v=w(H, H), b_v=tt(dev, torch.zeros(H)),
              w_out=w(H, H), b_out=tt(dev, torch.zeros(H)),
              w_ln=tt(dev, torch.ones(H)), b_ln=tt(dev, torch.zeros(H)))
    if identity_kq:
        wd["w_q"] = tt(dev, torch.eye(H))
        wd["w_k"] = tt(dev, torch.eye(H))
    return wd


def part1():
    N_HEAD, HD, M, N, CS = 4, 8, 20, 3, 4
    H = N_HEAD * HD
    dev = TorchDevice("cpu")
    cfg = SimGuidedConfig(probe_head_count=3, alpha=0.6)

    def build_case(case, seed):
        """Returns (tree, manager, provider, prefix kv tensors, x_new)."""
        g = torch.Generator().manual_seed(seed)
        tmp = tempfile.mkdtemp(prefix=f"impress_sel_{case}_")
        tree = PrefixRadixTree(tmp, chunk_size=CS)
        w_vec = torch.tensor([1.0, -1.0]).repeat(H // 2)
        w_heads = w_vec.view(N_HEAD, HD)
        kv = {}
        for j in range(2):
            pk = torch.randn(M, N_HEAD, HD, generator=g) * 0.01
            if case == "agree":     # tokens 0..4 dominate in EVERY head
                for t in range(5):
                    pk[t] = (5.0 - 0.5 * t) * w_heads
            elif case == "disagree":  # head h dominates tokens [5h, 5h+5)
                for h in range(N_HEAD):
                    for t in range(5 * h, min(5 * h + 5, M)):
                        pk[t, h] = (5.0 - 0.5 * (t - 5 * h)) * w_heads[h]
            pv = torch.randn(M, N_HEAD, HD, generator=g)
            kv[j] = (pk, pv)
        tree.insert(list(range(M)), kv)
        icfg = ImpressConfig(probe_head_count=3, alpha=0.6,
                             gpu_cache_capacity=0, cpu_cache_capacity=0,
                             selective=True)
        mgr = ChunkKVManager(tree, torch.device("cpu"), icfg, n_head=N_HEAD)
        provider = PathKVProvider(mgr, tree.match(list(range(M))))
        if case == "random":
            x_new = torch.randn(1, N, H, generator=g)
            wd = _p1_weights(dev, g, N_HEAD, HD, identity_kq=False)
        else:
            x_new = w_vec.expand(1, N, H).clone()
            wd = _p1_weights(dev, g, N_HEAD, HD, identity_kq=True)
        return tmp, tree, mgr, provider, kv, x_new, wd

    def run_both(tree, mgr, provider, kv, x_new, wd, ratio):
        mask_old = tt(dev, torch.ones(1, M + N, dtype=torch.bool))
        pk, pv = tree.load_prefix_kv(tree.match(list(range(M))))
        io_old = IOCounter()
        v_old, ko, vo = mha_prefix_reuse_sim(
            dev, tt(dev, x_new.clone()), mask_old, kv[0][0], kv[0][1],
            wd["w_q"], wd["b_q"], wd["w_k"], wd["b_k"], wd["w_v"], wd["b_v"],
            wd["w_out"], wd["b_out"], wd["w_ln"], wd["b_ln"],
            N_HEAD, ratio, cfg, io_counter=io_old, use_probe=True,
            layer_id=0)
        mgr.phys.reset()
        io_new = IOCounter()
        mask_new = tt(dev, torch.ones(1, M + N, dtype=torch.bool))
        v_new, kn, vn = mha_selective(
            dev, tt(dev, x_new.clone()), mask_new, provider, 0,
            wd["w_q"], wd["b_q"], wd["w_k"], wd["b_k"], wd["w_v"], wd["b_v"],
            wd["w_out"], wd["b_out"], wd["w_ln"], wd["b_ln"],
            N_HEAD, ratio, cfg, io_counter=io_new)
        assert torch.allclose(v_old.data, v_new.data, atol=1e-5), (
            (v_old.data - v_new.data).abs().max())
        assert torch.allclose(ko.data, kn.data, atol=1e-6)
        assert torch.allclose(vo.data, vn.data, atol=1e-6)
        return io_old, io_new

    # ---- agree: probe path, chunk skip ----
    tmp, tree, mgr, provider, kv, x_new, wd = build_case("agree", 0)
    io_old, io_new = run_both(tree, mgr, provider, kv, x_new, wd, 0.25)
    assert io_old.records[0]["mode"] == io_new.records[0]["mode"] == "probe"
    opened = {os.path.basename(p) for p in mgr.phys.opened_files}
    # selected = tokens 0..4 -> chunks 0 (tokens 0-3) and 1 (token 4) only
    expected = {"probe_3.k.npy",
                "chunk_00000.k.npy", "chunk_00000.v.npy",
                "chunk_00001.k.npy", "chunk_00001.v.npy"}
    assert opened == expected, opened
    for skipped in ("chunk_00002", "chunk_00003", "chunk_00004"):
        assert not any(skipped in p for p in opened)
    shutil.rmtree(tmp, ignore_errors=True)
    print("  1a. probe path: output == Stage-8 path (atol 1e-5); opened "
          f"files = probe sidecar + 2 selected chunks only ({sorted(opened)})"
          "; chunks 2-4 never read")

    # ---- disagree: fallback loads the full range ----
    tmp, tree, mgr, provider, kv, x_new, wd = build_case("disagree", 1)
    io_old, io_new = run_both(tree, mgr, provider, kv, x_new, wd, 0.25)
    assert io_old.records[0]["mode"] == io_new.records[0]["mode"] == "fallback"
    opened = {os.path.basename(p) for p in mgr.phys.opened_files}
    assert sum(1 for p in opened if p.startswith("chunk_")) == 10, opened
    shutil.rmtree(tmp, ignore_errors=True)
    print("  1b. fallback path: output == Stage-8 path; all 5 chunk pairs "
          "read (wider range by nature)")

    # ---- random weights + full-load ratio ----
    tmp, tree, mgr, provider, kv, x_new, wd = build_case("random", 2)
    run_both(tree, mgr, provider, kv, x_new, wd, 0.25)
    run_both(tree, mgr, provider, kv, x_new, wd, 1.0)
    shutil.rmtree(tmp, ignore_errors=True)
    print("  1c. random weights: equivalent at retention 0.25 and 1.0")
    print("Part 1 PASSED")


# ==================================================== GPU parts (2, 3, 4)
def build_gpu_env(args):
    from flexllmgen.compression import CompressionConfig
    from flexllmgen.flex_opt import Policy, OptLM, get_opt_config
    from flexllmgen.pytorch_backend import TorchDisk, TorchMixedDevice
    from flexllmgen.utils import ExecutionEnv
    from flexllmgen.impress.verify_impress_e2e import build_workload
    from flexllmgen.impress.verify_selection_rte import format_rte
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b",
                                              padding_side="left")
    ids = dict(true=tokenizer(" True", add_special_tokens=False).input_ids[0],
               false=tokenizer(" False", add_special_tokens=False).input_ids[0])
    prefix_texts, samples = build_workload(args.num_samples, args.num_shots,
                                           args.seed)
    prefixes = [tokenizer(t).input_ids for t in prefix_texts]
    requests = []
    for i, ex in enumerate(samples):
        q = tokenizer("\n\n" + format_rte(ex),
                      add_special_tokens=False).input_ids
        requests.append((prefixes[i % 2], q, ex["label"]))

    config = get_opt_config(args.model)
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
    print(f"init weights: {args.model} ... (prefixes "
          f"{len(prefixes[0])}/{len(prefixes[1])} tokens, "
          f"{len(requests)} requests)")
    model = OptLM(config, env, args.path, policy)
    total_vectors = ((len(prefixes[0]) + len(prefixes[1]))
                     * config.num_hidden_layers * 2)
    return dict(model=model, config=config, env=env, gpu=gpu,
                requests=requests, ids=ids, total_vectors=total_vectors)


def run_serving(G, args, retention, alpha=0.6, gpu_cache=0, cpu_cache=0,
                tag=""):
    """One server over the workload; metrics from request MEASURE_FROM on."""
    tree_dir = os.path.join(args.work_dir, f"tree_{tag}")
    shutil.rmtree(tree_dir, ignore_errors=True)
    cfg = ImpressConfig(retention_ratio=retention, alpha=alpha,
                        gpu_cache_capacity=gpu_cache,
                        cpu_cache_capacity=cpu_cache,
                        reorder_every_n_requests=8, selective=True)
    server = ImpressServer(G["model"], G["config"], tree_dir, cfg,
                           gpu_device=G["gpu"].dev)
    ctrl = server.controller
    phys = server.manager.phys
    preds, ttfts = [], []
    n_meas = 0
    try:
        for i, (p_ids, q_ids, _) in enumerate(G["requests"]):
            if i == MEASURE_FROM:
                phys.reset()
                ctrl.io.reset()
            logits, ttft, r_len, nr_len = server.request(p_ids, q_ids)
            preds.append(0 if logits[G["ids"]["true"]]
                         > logits[G["ids"]["false"]] else 1)
            if i >= MEASURE_FROM and nr_len == 0:
                ttfts.append(ttft)
                n_meas += 1
        acc = float(np.mean([p == r[2]
                             for p, r in zip(preds, G["requests"])]))
        s = phys.summary()
        return dict(acc=acc,
                    ttft_ms=float(np.median(ttfts)) * 1000,
                    files_per_req=s["files"] / n_meas,
                    disk_mb_per_req=s["disk_mb"] / n_meas,
                    disk_ms_per_req=s["disk_ms"] / n_meas,
                    pcie_mb_per_req=s["pcie_mb"] / n_meas,
                    fallback=ctrl.io.fallback_rate())
    finally:
        server.close()
        shutil.rmtree(tree_dir, ignore_errors=True)


def part2(G, args):
    print("\nPart 2: physical I/O vs retention (caches OFF -> every needed "
          "chunk is a real disk read; measured post-reorder)")
    grid = [("FullLoad", 1.0), ("50%", 0.5), ("25%", 0.25), ("10%", 0.1)]
    rows = []
    for name, r in grid:
        res = run_serving(G, args, retention=r, tag=f"p2_{name}")
        rows.append((name, res))
        print(f"  {name:8s}: files/req {res['files_per_req']:7.1f}, "
              f"disk {res['disk_mb_per_req']:7.1f} MB/req, "
              f"disk {res['disk_ms_per_req']:6.1f} ms/req, "
              f"TTFT {res['ttft_ms']:6.1f} ms, acc {res['acc']*100:.0f}%, "
              f"fallback {res['fallback']*100:.0f}%")
    for metric in ("files_per_req", "disk_mb_per_req", "disk_ms_per_req"):
        vals = [r[metric] for _, r in rows]
        assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)), (
            f"{metric} not monotonically decreasing: {vals}")
    fl, r10 = rows[0][1], rows[-1][1]
    print(f"  PASS: files/bytes/disk-time all monotone; at 10% retention "
          f"disk bytes are {fl['disk_mb_per_req']/r10['disk_mb_per_req']:.1f}x"
          f" lower and disk time {fl['disk_ms_per_req']/r10['disk_ms_per_req']:.1f}x lower than FullLoad")
    return rows


def part3(G, args):
    print("\nPart 3: alpha sweep on the selective path (caches OFF, "
          "retention 25%) — disk time should now be monotone in alpha")
    grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    rows = []
    for a in grid:
        res = run_serving(G, args, retention=0.25, alpha=a, tag=f"p3_{a}")
        rows.append((a, res))
        print(f"  alpha={a:3.1f}: disk {res['disk_ms_per_req']:6.1f} ms/req "
              f"({res['disk_mb_per_req']:6.1f} MB), TTFT "
              f"{res['ttft_ms']:6.1f} ms, fallback {res['fallback']*100:5.1f}%,"
              f" acc {res['acc']*100:.0f}%")
    dms = [r["disk_ms_per_req"] for _, r in rows]
    # monotone non-increasing with small measurement tolerance
    ok = all(dms[i + 1] <= dms[i] * 1.10 + 0.5 for i in range(len(dms) - 1))
    print(f"  disk time {dms[0]:.1f} -> {dms[-1]:.1f} ms/req across alpha "
          f"({'PASS: monotone within tolerance' if ok else 'WARN: non-monotone'});"
          " previous sweep's TTFT jittered 60-103 ms with no alpha trend")
    # small chart: alpha vs disk time and TTFT (single-axis panels)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from flexllmgen.impress.sweep_impress import (styled_axes, C_SERIES1,
        C_SURFACE, C_TEXT, C_TEXT2)
    plt.rcParams.update({"text.color": C_TEXT, "axes.labelcolor": C_TEXT2,
                         "font.size": 10, "figure.facecolor": C_SURFACE,
                         "savefig.facecolor": C_SURFACE})
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 4.4), sharex=True)
    for ax, key, label in ((axes[0], "disk_ms_per_req", "disk read (ms/req)"),
                           (axes[1], "ttft_ms", "TTFT (ms, median)")):
        styled_axes(ax)
        ax.plot(grid, [r[key] for _, r in rows], color=C_SERIES1,
                linewidth=2, marker="o", markersize=6)
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
    axes[0].set_title("Selective loading: alpha vs physical disk I/O",
                      loc="left", fontsize=11, color=C_TEXT)
    axes[1].set_xlabel("alpha (threshold = j^alpha)")
    axes[1].set_xticks(grid)
    fig.tight_layout()
    out = os.path.join(args.out_dir, "selective_alpha.png")
    os.makedirs(args.out_dir, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")
    return rows


def part4(G, args):
    print("\nPart 4: 3-mode TTFT with I/O breakdown (cf. paper Figure 17)")
    # IMPRESS with working caches, IMPRESS with caches off, FullLoad, ReComp
    imp = run_serving(G, args, retention=0.25,
                      gpu_cache=int(G["total_vectors"] * 0.6),
                      cpu_cache=G["total_vectors"], tag="p4_impress")
    imp_nc = run_serving(G, args, retention=0.25, tag="p4_impress_nc")
    full = run_serving(G, args, retention=1.0, tag="p4_fullload")

    # ReComp (needs a controller only for logits capture)
    from flexllmgen.impress.selective_loading import (install_selective_hook,
        uninstall_selective_hook)
    ctrl = install_selective_hook()
    ctrl.capture_logits = True
    preds, ts = [], []
    try:
        for p_ids, q_ids, _ in G["requests"]:
            t0 = time.perf_counter()
            G["model"].generate((list(p_ids) + list(q_ids),),
                                max_new_tokens=1, do_sample=False)
            ts.append(time.perf_counter() - t0)
            preds.append(0 if ctrl.last_logits[0][G["ids"]["true"]]
                         > ctrl.last_logits[0][G["ids"]["false"]] else 1)
    finally:
        uninstall_selective_hook()
    recomp = dict(ttft_ms=float(np.median(ts)) * 1000,
                  acc=float(np.mean([p == r[2]
                                     for p, r in zip(preds, G["requests"])])),
                  disk_ms_per_req=0.0)

    print(f"  {'mode':22s} {'TTFT':>9s} {'disk I/O':>10s} {'rest':>9s} "
          f"{'acc':>6s}")
    for name, r in (("IMPRESS (cache on)", imp),
                    ("IMPRESS (cache off)", imp_nc),
                    ("FullLoad (cache off)", full),
                    ("ReComp", recomp)):
        rest = r["ttft_ms"] - r["disk_ms_per_req"]
        print(f"  {name:22s} {r['ttft_ms']:7.1f}ms {r['disk_ms_per_req']:8.1f}ms "
              f"{rest:7.1f}ms {r['acc']*100:5.1f}%")
    io_cut = full["disk_ms_per_req"] - imp_nc["disk_ms_per_req"]
    ttft_cut = full["ttft_ms"] - imp_nc["ttft_ms"]
    print(f"  cache-off comparison: selective loading removes "
          f"{io_cut:.1f} ms/req of disk I/O ({io_cut/max(ttft_cut,1e-9)*100:.0f}% "
          f"of the {ttft_cut:.1f} ms TTFT gap to FullLoad)")
    assert imp["ttft_ms"] < recomp["ttft_ms"]
    assert imp["ttft_ms"] < full["ttft_ms"]
    print("  PASS: IMPRESS fastest; I/O breakdown reported")


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
                        default="/home/dblab/hj/impress_kv_store/selective")
    parser.add_argument("--out-dir", type=str,
                        default="/home/dblab/hj/impress_sweep")
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--num-shots", type=int, default=21)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pos-extend", type=int, default=0,
        help="install PI position extension to this length (0 = off); needed for prefixes beyond OPT's trained 2048")
    parser.add_argument("--parts", type=str, default="1,2,3,4")
    args = parser.parse_args()
    if args.pos_extend:
        from flexllmgen.impress.pos_extend import install_pos_extension
        install_pos_extension(args.pos_extend)
        print(f"position extension installed: target_len="
              f"{args.pos_extend} (PI interpolation)")
    parts = {p.strip() for p in args.parts.split(",")}

    if "1" in parts:
        print("Part 1: numerical equivalence + chunk skip (CPU, synthetic)")
        part1()
    if parts & {"2", "3", "4"}:
        G = build_gpu_env(args)
        try:
            if "2" in parts:
                part2(G, args)
            if "3" in parts:
                part3(G, args)
            if "4" in parts:
                part4(G, args)
        finally:
            G["env"].close_copy_threads()
    print("DONE")


if __name__ == "__main__":
    main()

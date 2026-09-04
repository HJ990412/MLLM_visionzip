"""Query-conditioned variant of the Fig 7(b) analysis.

Importance here is computed the way IMPRESS serving does (§4.1 / Figure 9
step 2): the QUERY tokens' q vectors attend over the prefix keys, and a
prefix token's importance is the column sum over the QUERY ROWS ONLY —
instead of the prefix's full causal self-attention used in fig7b_jaccard.

Outputs, per ratio (default 50%):
  1. head-pair Jaccard per layer (as in Fig 7(b)) under query-conditioned
     importance, vs the prefix-only baseline from the fig7b cache;
  2. CROSS-QUERY agreement: for each head, Jaccard between the important
     sets induced by different questions (quantifies §3.2 Challenge 1).

Runs one prefill of (prefix + query) per query; by causality the prefix
block of the attention matrix is identical to the serving path (verified
equivalence in Stage 4).
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/dblab/hj/FlexGen")
sys.path.insert(0, "/home/dblab/hj/analysis/fig7b")
from fig7b_jaccard import (topk_sets, jaccard_matrix, offdiag_mean,
    BLUE_RAMP, C_SURFACE, C_TEXT, C_TEXT2)

from flexllmgen.pytorch_backend import TorchDevice, TorchTensor, DeviceType
from flexllmgen.flex_opt import SelfAttention

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


class QueryRowCollector:
    def __init__(self):
        self.enabled = False
        self.prefix_len = 0
        self.current_layer = None
        self.scores = {}   # layer -> (n_head, m) float32 cpu


def make_hooked_mha(col):
    """Copy of TorchDevice.mha (pytorch_backend.py:298-365) capturing
    column sums over QUERY ROWS x PREFIX COLUMNS after softmax."""
    def mha(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
            w_out, b_out, w_ln, b_ln, n_head, donate, compress_cache,
            comp_config):
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)
        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5
        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data,
                              bias=b_ln.data)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        q = q.view(b, s, n_head, head_dim).permute(0, 2, 1, 3) \
             .reshape(b * n_head, s, head_dim)
        k = k.view(b, s, n_head, head_dim).permute(0, 2, 3, 1) \
             .reshape(b * n_head, head_dim, s)
        v = v.view(b, s, n_head, head_dim).permute(0, 2, 1, 3) \
             .reshape(b * n_head, s, head_dim)
        attn = torch.bmm(q, k)
        idx = torch.arange(s, device=self.dev)
        causal = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal
        attn = attn.view(b, n_head, s, s)
        attn = torch.where(mask, attn, -1e4)
        attn = F.softmax(attn.view(b * n_head, s, s), dim=2)
        if col.enabled:
            m0 = col.prefix_len
            aw = attn.view(b, n_head, s, s)
            # query rows (m0:) x prefix columns (:m0) -> column sums
            col.scores[col.current_layer] = \
                aw[:, :, m0:, :m0].float().sum(dim=2)[0].cpu()
        value = torch.bmm(attn, v).view(b, n_head, s, head_dim)
        value = value.transpose(1, 2).reshape(b, s, h)
        value = F.linear(value, w_out.data, bias=b_out.data)
        value.add_(inputs.data)
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)
        k = TorchTensor.create_from_torch(k, self)
        v = TorchTensor.create_from_torch(v, self)
        return TorchTensor.create_from_torch(value, self), k, v
    return mha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-6.7b")
    parser.add_argument("--path", type=str, default="~/opt_weights")
    parser.add_argument("--offload-dir", type=str,
                        default="~/flexllmgen_offload_dir")
    parser.add_argument("--num-shots", type=int, default=21)
    parser.add_argument("--num-queries", type=int, default=3)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot-sets", action="store_true",
        help="render a head x token membership heatmap (layers 1/16/32, "
             "query 1) to output/sets_heatmap_q1.png")
    parser.add_argument("--pos-extend", type=int, default=0,
        help="install PI position extension (required for prefixes >2048)")
    parser.add_argument("--prefix-style", choices=("qa", "ctx", "dctx"),
        default="qa",
        help="qa = N-shot QA-only; ctx = wikitext-passage style; dctx = "
             "N shots each with DATASET-internal context (unused RTE train "
             "premises), total kept under 2k tokens (no PI needed)")
    parser.add_argument("--query-idx", type=int, default=0,
        help="0 = average over all queries (default); 1..N = use only "
             "that query for the head-pair Jaccard heatmap")
    parser.add_argument("--show-sets", type=int, default=0,
        help="print the query texts and, per layer (1/16/32), the top-N "
             "decoded tokens of heads 0/1/2 plus their pairwise Jaccard")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    style_tag = ("" if args.prefix_style == "qa" and args.num_shots == 21
                 else f"_{args.prefix_style}{args.num_shots}")
    cache = os.path.join(OUT_DIR,
                         f"scores_query_{args.num_queries}q_seed{args.seed}"
                         f"{style_tag}.pt")

    if args.pos_extend:
        from flexllmgen.impress.pos_extend import install_pos_extension
        install_pos_extension(args.pos_extend)
        print(f"position extension installed: {args.pos_extend} (PI)")
    from transformers import AutoTokenizer
    from flexllmgen.impress.verify_impress_e2e import build_workload
    from flexllmgen.impress.verify_selection_rte import format_rte
    tok = AutoTokenizer.from_pretrained("facebook/opt-30b",
                                        padding_side="left")
    prefix_texts, samples = build_workload(max(4, args.num_queries),
                                           args.num_shots, args.seed)
    if args.prefix_style == "dctx":
        # 10-shot (paper's upper bound) + dataset-internal context filler,
        # total <= ~1840 tokens so prefix+query stays inside 2048
        from datasets import load_dataset
        d = load_dataset("glue", "rte")
        rngp = np.random.RandomState(args.seed + 5)
        order = rngp.permutation(len(d["train"])).tolist()
        shots_ex = [d["train"][int(i)] for i in order[:args.num_shots]]
        pool = [d["train"][int(i)]["sentence1"]
                for i in order[args.num_shots:]]
        qa_texts = [format_rte(ex) + (" True" if ex["label"] == 0
                                      else " False") for ex in shots_ex]
        qa_tok = sum(len(tok(t, add_special_tokens=False).input_ids)
                     for t in qa_texts)
        target = 1840
        per_shot = max(0, target - qa_tok - 4 * args.num_shots) \
            // args.num_shots
        parts, pi = [], 0
        for k in range(args.num_shots):
            ctx, used = [], 0
            while used < per_shot and pi < len(pool):
                t = pool[pi].strip()
                pi += 1
                used += len(tok(" " + t,
                                add_special_tokens=False).input_ids)
                ctx.append(t)
            parts.append("Context: " + " ".join(ctx) + "\n" + qa_texts[k])
        prefix_ids = tok("\n\n".join(parts)).input_ids[:1840]
    elif args.prefix_style == "ctx":
        # paper-condition prefix: 2-10 shots each with a wikitext passage
        from flexllmgen.impress.verify_paper_condition import (
            build_workload as build_ctx)
        ctx_prefixes, _, _ = build_ctx("rte", tok, args.seed)
        prefix_ids = ctx_prefixes[0]
    else:
        prefix_ids = tok(prefix_texts[0]).input_ids
    m = len(prefix_ids)
    queries = [tok("\n\n" + format_rte(samples[i]),
                   add_special_tokens=False).input_ids
               for i in range(args.num_queries)]
    print(f"prefix {m} tokens + {args.num_queries} RTE questions "
          f"({[len(q) for q in queries]} tokens); ratio {args.ratio:.0%}")

    if os.path.exists(cache):
        per_query = torch.load(cache)
        print(f"scores loaded from cache: {cache}")
    else:
        from flexllmgen.compression import CompressionConfig
        from flexllmgen.flex_opt import Policy, OptLM, get_opt_config
        from flexllmgen.pytorch_backend import TorchDisk, TorchMixedDevice
        from flexllmgen.utils import ExecutionEnv
        config = get_opt_config(args.model)
        gpu, cpu = TorchDevice("cuda:0"), TorchDevice("cpu")
        disk = TorchDisk(args.offload_dir)
        env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk,
                           mixed=TorchMixedDevice([gpu, cpu, disk]))
        policy = Policy(1, 1, 100, 0, 100, 0, 100, 0,
                        overlap=False, sep_layer=True, pin_weight=True,
                        cpu_cache_compute=False, attn_sparsity=1.0,
                        compress_weight=False,
                        comp_weight_config=CompressionConfig(
                            num_bits=4, group_size=64, group_dim=0,
                            symmetric=False),
                        compress_cache=False,
                        comp_cache_config=CompressionConfig(
                            num_bits=4, group_size=64, group_dim=2,
                            symmetric=False))
        print(f"init weights: {args.model} ...")
        model = OptLM(config, env, args.path, policy)

        col = QueryRowCollector()
        col.prefix_len = m
        orig_mha = TorchDevice.mha
        orig_fw = SelfAttention.forward
        TorchDevice.mha = make_hooked_mha(col)

        def fw(self, *a, **kw):
            col.current_layer = self.layer_id
            return orig_fw(self, *a, **kw)
        SelfAttention.forward = fw
        per_query = []
        try:
            for qi, q_ids in enumerate(queries):
                col.scores = {}
                col.enabled = True
                model.generate((prefix_ids + q_ids,), max_new_tokens=1,
                               do_sample=False)
                col.enabled = False
                per_query.append(dict(col.scores))
                print(f"  query {qi + 1}: captured "
                      f"{len(col.scores)} layers")
        finally:
            TorchDevice.mha = orig_mha
            SelfAttention.forward = orig_fw
            env.close_copy_threads()
        torch.save(per_query, cache)
        print(f"scores cached to {cache}")

    num_layers = len(per_query[0])
    r = args.ratio

    if args.show_sets > 0:
        N = args.show_sets
        print("\n=== queries used (RTE validation, lm-eval template, "
              "no answer) ===")
        for qi in range(args.num_queries):
            txt = "\n\n" + format_rte(samples[qi])
            print(f"--- query {qi + 1} ({len(queries[qi])} tokens, "
                  f"label={samples[qi]['label']}) ---")
            print(txt.strip()[:400])
        print("\n=== important-token sets (query 1, top "
              f"{args.ratio:.0%}; showing top {N} of "
              f"{max(1, int(round(m * args.ratio)))} per head) ===")
        sq = per_query[0]
        for j in (0, 15, 31):
            print(f"\n[layer {j + 1}]")
            sets3 = []
            for h in (0, 1, 2):
                sc = sq[j][h]
                k_keep = max(1, int(round(m * args.ratio)))
                top_full = torch.topk(sc, k_keep).indices
                sets3.append(set(top_full.tolist()))
                shown = ", ".join(
                    f"{int(i)}:{tok.decode([prefix_ids[int(i)]])!r}"
                    for i in top_full[:N])
                print(f"  head {h}: {shown}")
            for a, b in ((0, 1), (0, 2), (1, 2)):
                inter = len(sets3[a] & sets3[b])
                union = len(sets3[a] | sets3[b])
                print(f"  J(head{a}, head{b}) = {inter / union:.3f}  "
                      f"(교집합 {inter}/{union})")
            core = sets3[0] & sets3[1] & sets3[2]
            print(f"  3-head 교집합: {len(core)}개 "
                  f"({len(core) / max(1, int(round(m * args.ratio))):.0%} of k)")

    if args.plot_sets:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        toks = [tok.decode([t]) for t in prefix_ids]
        nl_pos = [i for i, t in enumerate(toks) if "\n" in t]
        ans_pos = [i for i, t in enumerate(toks) if t.strip() == "Answer"]
        k_keep = max(1, int(round(m * args.ratio)))
        sq = per_query[0]
        layers = (0, 15, 31)
        fig, axes = plt.subplots(
            len(layers) + 1, 1, figsize=(13, 8.2), sharex=True,
            gridspec_kw={"height_ratios": [0.5] + [3] * len(layers)})
        ax = axes[0]
        ax.scatter(nl_pos, [0.6] * len(nl_pos), marker="|", s=48,
                   color="#52514e", label="newline (shot boundary)")
        ax.scatter(ans_pos, [0.25] * len(ans_pos), marker="|", s=48,
                   color="#e34948", label="'Answer'")
        ax.scatter([0], [0.6], marker="D", s=24, color="#0b0b0b")
        ax.set_ylim(0, 1); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
        ax.set_title(f"structural-token track (BOS=◆ idx 0)  |  query 1, "
                     f"top {args.ratio:.0%} (k={k_keep})",
                     loc="left", fontsize=10)
        cmap = ListedColormap(["#fcfcfb", "#2a78d6"])
        for ax, j in zip(axes[1:], layers):
            n_head = sq[j].shape[0]
            Mm = np.zeros((n_head, m), dtype=bool)
            for h in range(n_head):
                Mm[h, torch.topk(sq[j][h], k_keep).indices.numpy()] = True
            frac_all = float(Mm.all(axis=0).sum())
            ax.imshow(Mm, cmap=cmap, aspect="auto", interpolation="none")
            ax.set_ylabel(f"L{j + 1}\nhead")
            ax.set_title(f"layer {j + 1}: tokens selected by ALL 32 heads "
                         f"= {frac_all:.0f} (vertical stripes)", loc="left",
                         fontsize=10)
            ax.tick_params(labelsize=8)
        axes[-1].set_xlabel("prefix token index (0..1641)")
        fig.tight_layout()
        out = os.path.join(OUT_DIR, "sets_heatmap_q1.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"saved {out}")

    # ---- 1) head-pair Jaccard per layer, query-conditioned ----
    print(f"\n[1] head-pair Jaccard (query-conditioned importance), "
          f"r={r:.0%}, mean over {len(per_query)} queries:")
    base = None
    base_path = os.path.join(OUT_DIR, "scores_21shot_seed0.pt")
    if os.path.exists(base_path):
        base = torch.load(base_path)["scores"]
    print(f"{'layer':>6s} {'query-cond':>11s} {'prefix-only':>12s}")
    hp_by_layer = []
    for j in range(num_layers):
        vals = []
        for sq in per_query:
            sets, _ = topk_sets(sq[j], r)
            vals.append(offdiag_mean(jaccard_matrix(sets)))
        hp_by_layer.append(float(np.mean(vals)))
        b_str = ""
        if base is not None:
            bs, _ = topk_sets(base[j], r)
            b_str = f"{offdiag_mean(jaccard_matrix(bs)):12.3f}"
        if j in (0, 15, 31):
            print(f"{j + 1:>6d} {hp_by_layer[-1]:11.3f} {b_str}")
    print(f"  all-layer avg: query-cond {np.mean(hp_by_layer):.3f} "
          f"(E(J) random = {r / (2 - r):.3f})")

    # ---- head x head Jaccard heatmaps (fig7b style, query-conditioned) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("impress_blue", BLUE_RAMP)
    plt.rcParams.update({"text.color": C_TEXT, "axes.labelcolor": C_TEXT2,
                         "font.size": 10, "figure.facecolor": C_SURFACE,
                         "savefig.facecolor": C_SURFACE})
    use_q = (per_query if args.query_idx == 0
             else [per_query[args.query_idx - 1]])
    q_tag = ("mean of %d queries" % len(per_query) if args.query_idx == 0
             else "query %d only" % args.query_idx)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))
    for ax, j in zip(axes, (0, 15, 31)):
        Ms = []
        for sq in use_q:
            sets, _ = topk_sets(sq[j], r)
            Ms.append(jaccard_matrix(sets))
        M = np.mean(Ms, axis=0)
        im = ax.imshow(M, cmap=cmap, vmin=0.0, vmax=1.0, origin="upper")
        ax.set_title(f"Layer {j + 1}  avg={offdiag_mean(M):.3f}",
                     fontsize=11, color=C_TEXT)
        ax.set_xlabel("Head Index")
        ax.set_ylabel("Head Index")
        ax.tick_params(colors=C_TEXT2, labelsize=8)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("Jaccard similarity", color=C_TEXT2)
    cb.ax.tick_params(colors=C_TEXT2)
    fig.suptitle(f"QUERY-conditioned importance ({q_tag}), "
                 f"top {r:.0%} — OPT-6.7B, RTE prefix {m} tokens",
                 fontsize=12, color=C_TEXT, y=1.02)
    suffix = "" if args.query_idx == 0 else f"_q{args.query_idx}"
    out = os.path.join(OUT_DIR,
                       f"fig7b_query_layers_1_16_32_r{int(r*100)}"
                       f"{style_tag}{suffix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # ---- 2) CROSS-QUERY agreement per head (same head, different query) ----
    print(f"\n[2] cross-query Jaccard (same head, different questions), "
          f"r={r:.0%}:")
    for j in (0, 15, 31):
        n_head = per_query[0][j].shape[0]
        sets_q = [topk_sets(sq[j], r)[0] for sq in per_query]
        vals = []
        for h in range(n_head):
            for a in range(len(per_query)):
                for b2 in range(a + 1, len(per_query)):
                    A, B = sets_q[a][h], sets_q[b2][h]
                    vals.append(len(A & B) / len(A | B))
        print(f"  layer {j + 1:2d}: {np.mean(vals):.3f}")
    print("DONE")


if __name__ == "__main__":
    main()

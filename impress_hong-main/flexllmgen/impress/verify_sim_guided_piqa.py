"""Stage 5 verification: similarity-guided identification on PIQA (OPT-6.7B).

Compares, at the same retention ratio (default 25%, as in the paper's
Figure 11 PIQA setting):
  reuse@1.0        : full prefix reuse, no selection      — accuracy reference
  naive (Stage 4)  : all heads' keys, per-head selection  — I/O baseline
  probe (§4.3)     : 3 probe heads + Jaccard threshold, per-layer fallback

Measured:
  (a) fallback rate over probe-attempted layers (paper §6.3: < 20% on avg)
  (b) total loaded vectors, probe vs naive (must clearly decrease)
  (c) accuracy difference probe vs naive (paper: within ~1-2%p)

Scoring: PIQA is 2-choice with multi-token answers; per choice we run one
prefill over (question + answer) as the new tokens and sum the answer
tokens' log-softmax scores from the captured full logits.

Usage:
  python -m flexllmgen.impress.verify_sim_guided_piqa \
      --num-samples 50 --retention 0.25 --alpha 0.6 --probe-heads 3
"""

import argparse

import numpy as np
import torch

from flexllmgen.compression import CompressionConfig
from flexllmgen.flex_opt import Policy, OptLM, SelfAttention, get_opt_config
from flexllmgen.pytorch_backend import (TorchDevice, TorchDisk,
    TorchMixedDevice)
from flexllmgen.utils import ExecutionEnv
from flexllmgen.impress.prefix_kv import PrefixKVStore
from flexllmgen.impress.similarity_guided import (SimGuidedConfig,
    install_sim_guided_hook, uninstall_sim_guided_hook)
from flexllmgen.impress.verify_selection_rte import capture_prefix_kv


def load_piqa():
    from datasets import load_dataset
    try:
        return load_dataset("ybisk/piqa", trust_remote_code=True)
    except Exception:
        return load_dataset("ybisk/piqa", revision="refs/convert/parquet")


def format_piqa(ex):
    return f"Question: {ex['goal']}\nAnswer:"


def build_piqa(num_shots, num_samples, seed):
    d = load_piqa()
    rng = np.random.RandomState(seed)
    shot_idx = rng.choice(len(d["train"]), num_shots, replace=False)
    shots = []
    for i in shot_idx:
        ex = d["train"][int(i)]
        sol = ex["sol1"] if int(ex["label"]) == 0 else ex["sol2"]
        shots.append(format_piqa(ex) + " " + sol)
    prefix_text = "\n\n".join(shots)
    val_idx = rng.choice(len(d["validation"]), num_samples, replace=False)
    samples = [d["validation"][int(i)] for i in val_idx]
    return prefix_text, samples


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
    parser.add_argument("--kv-dir", type=str,
                        default="/home/dblab/hj/impress_kv_store/stage5_piqa")
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--retention", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--probe-heads", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b",
                                              padding_side="left")
    prefix_text, samples = build_piqa(args.num_shots, args.num_samples,
                                      args.seed)
    prefix_ids = tokenizer(prefix_text).input_ids
    m = len(prefix_ids)

    items = []  # (q_ids, [choice1_ids, choice2_ids], label)
    for ex in samples:
        q_ids = tokenizer("\n\n" + format_piqa(ex),
                          add_special_tokens=False).input_ids
        chs = [tokenizer(" " + ex[f"sol{c}"],
                         add_special_tokens=False).input_ids for c in (1, 2)]
        items.append((q_ids, chs, int(ex["label"])))
    print(f"prefix: {args.num_shots}-shot, {m} tokens; "
          f"{len(items)} PIQA validation samples; retention "
          f"{args.retention:.0%}, alpha {args.alpha}, "
          f"probe heads {args.probe_heads}")

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
    print(f"init weights: {args.model} ...")
    model = OptLM(config, env, args.path, policy)
    num_layers = config.num_hidden_layers

    try:
        # ---- prefix KV: prefill once -> Stage-1 store -> GPU-resident ----
        captured = capture_prefix_kv(model, prefix_ids, num_layers)
        store = PrefixKVStore(args.kv_dir)
        for j in range(num_layers):
            store.store_layer(j, *captured[j])
        store.finalize(args.model, num_layers, m)
        del captured
        prefix_kv = {}
        for j in range(num_layers):
            k_t, v_t = store.load_layer(j)
            prefix_kv[j] = (k_t.to(gpu.dev), v_t.to(gpu.dev))
        print(f"prefix KV via Stage-1 store: {store.total_bytes() / 1e6:.1f} MB")

        controller = install_sim_guided_hook(
            SimGuidedConfig(probe_head_count=args.probe_heads,
                            alpha=args.alpha))
        controller.prefix_kv = prefix_kv
        controller.prefix_len = m
        controller.capture_full_logits = True
        controller.enabled = True

        def choice_loglik(q_ids, choice_ids):
            input_ids = q_ids + choice_ids
            model.generate((input_ids,), max_new_tokens=1, do_sample=False)
            logits = controller.full_logits[0]          # (n, vocab)
            lp = torch.log_softmax(logits, dim=-1)
            n, L = len(input_ids), len(choice_ids)
            return float(sum(lp[n - L - 1 + i, choice_ids[i]]
                             for i in range(L)))

        results = {}
        modes = [("reuse@1.0", 1.0, True),
                 ("naive", args.retention, False),
                 ("probe", args.retention, True)]
        for name, ratio, use_probe in modes:
            controller.retention_ratio = ratio
            controller.use_probe = use_probe
            controller.io.reset()
            preds = []
            for q_ids, chs, _ in items:
                scores = [choice_loglik(q_ids, c) for c in chs]
                preds.append(int(np.argmax(scores)))
            acc = float(np.mean([p == it[2] for p, it in zip(preds, items)]))
            io = controller.io
            n_calls = len(io.records)
            print(f"  {name:9s}: accuracy = {acc*100:5.1f}%, "
                  f"vectors = {io.total_vectors:,} "
                  f"({io.total_vectors / max(n_calls,1):,.0f}/layer-call), "
                  f"modes = {io.mode_counts()}")
            results[name] = dict(acc=acc, vectors=io.total_vectors,
                                 fallback=io.fallback_rate())

        # -------------------------------------------------- (a) (b) (c)
        fb = results["probe"]["fallback"]
        print(f"\n(a) fallback rate = {fb*100:.1f}% of probe-attempted "
              f"layer-calls "
              f"({'PASS: <= ~20%, matches paper §6.3' if fb <= 0.20 else 'WARN: above the paper-reported ~20%'})")
        # per-layer breakdown (probe mode ran last, records still in io)
        by_layer = {}
        for r in controller.io.records:
            if r["mode"] in ("probe", "fallback"):
                by_layer.setdefault(r["layer"], []).append(
                    r["mode"] == "fallback")
        rates = [float(np.mean(v)) for _, v in sorted(by_layer.items())]
        high = [j for j, (_, v) in enumerate(sorted(by_layer.items()))
                if np.mean(v) > 0.5]
        print(f"    per-layer fallback rate: "
              f"first half avg {np.mean(rates[:len(rates)//2])*100:.0f}%, "
              f"second half avg {np.mean(rates[len(rates)//2:])*100:.0f}%; "
              f"layers mostly falling back (>50%): {high}")

        nv, pv = results["naive"]["vectors"], results["probe"]["vectors"]
        assert pv < nv, "probe mode must load fewer vectors than naive"
        print(f"(b) loaded vectors: naive {nv:,} -> probe {pv:,} "
              f"({nv / pv:.2f}x reduction) PASS")

        diff = results["probe"]["acc"] - results["naive"]["acc"]
        ok = abs(diff) <= 0.04
        print(f"(c) accuracy: naive {results['naive']['acc']*100:.1f}% vs "
              f"probe {results['probe']['acc']*100:.1f}% "
              f"(diff {diff*100:+.1f}%p) "
              f"({'PASS: within paper-reported 1-2%p band (50-sample noise allowed)' if ok else 'WARN: larger than expected'})")
        print(f"    reference reuse@1.0 accuracy = "
              f"{results['reuse@1.0']['acc']*100:.1f}%")
        print("DONE")
    finally:
        uninstall_sim_guided_hook()
        env.close_copy_threads()


if __name__ == "__main__":
    main()

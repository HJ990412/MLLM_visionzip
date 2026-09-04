"""Stage 4 verification: naive per-head selection accuracy on RTE (few-shot).

Reproduces the direction of the paper's Figure 4 trend (lower KV retention
-> mildly lower accuracy) with the Figure 9(a) naive identification: all
heads' prefix keys are loaded from the Stage-1 store, importance is computed
per head (H2O column sum), and only the top retention_ratio tokens' K/V per
head take part in the remaining prefill (§4.1 dataflow). Exact numbers need
not match the paper — only the direction is checked.

Pipeline per evaluated mode:
  ReComp        : full prefill of (prefix + question), no reuse   — reference
  reuse @ 1.00  : prefix KV loaded from disk, no filtering        — sanity
  reuse @ 0.50 / 0.25 / 0.10 : per-head top-k selection

Scoring (multiple choice, batch 1, gen_len=1): capture last-token logits and
compare the logits of " True" vs " False" (lm-eval-style RTE prompt).

Usage:
  python -m flexllmgen.impress.verify_selection_rte \
      --num-samples 50 --num-shots 5 --ratios 1.0,0.5,0.25,0.1
"""

import argparse
import os

import numpy as np
import torch

from flexllmgen.compression import CompressionConfig
from flexllmgen.flex_opt import Policy, OptLM, SelfAttention, get_opt_config
from flexllmgen.pytorch_backend import (TorchDevice, TorchDisk,
    TorchMixedDevice)
from flexllmgen.utils import ExecutionEnv
from flexllmgen.impress.prefix_kv import PrefixKVStore
from flexllmgen.impress.prefix_reuse import (install_prefix_reuse_hook,
    uninstall_prefix_reuse_hook)


# --------------------------------------------------------------- RTE data
def format_rte(ex):
    """lm-eval-harness style RTE prompt; label 0 -> ' True', 1 -> ' False'."""
    return (f"{ex['sentence1']}\nQuestion: {ex['sentence2']} "
            f"True or False?\nAnswer:")


def build_rte(num_shots, num_samples, seed):
    from datasets import load_dataset
    d = load_dataset("glue", "rte")
    rng = np.random.RandomState(seed)
    shot_idx = rng.choice(len(d["train"]), num_shots, replace=False)
    shots = []
    for i in shot_idx:
        ex = d["train"][int(i)]
        ans = " True" if ex["label"] == 0 else " False"
        shots.append(format_rte(ex) + ans)
    prefix_text = "\n\n".join(shots)

    val_idx = rng.choice(len(d["validation"]), num_samples, replace=False)
    samples = [d["validation"][int(i)] for i in val_idx]
    return prefix_text, samples


# ------------------------------------------------------- prefill KV capture
def capture_prefix_kv(model, prefix_ids, num_layers):
    """Run one normal prefill on the prefix alone and grab per-layer KV
    (same technique as verify_stage1; patch is removed afterwards)."""
    captured = {}
    orig_forward = SelfAttention.forward

    def hooked(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
               cache_write_buf, i, k):
        orig_forward(self, hidden, cache_read_buf, weight_read_buf,
                     attention_mask, cache_write_buf, i, k)
        if i == 0:
            k_t, v_t = cache_write_buf.val
            captured[self.layer_id] = (
                k_t.data.detach().to("cpu").contiguous().clone(),
                v_t.data.detach().to("cpu").contiguous().clone())

    SelfAttention.forward = hooked
    try:
        model.generate((prefix_ids,), max_new_tokens=1, do_sample=False)
    finally:
        SelfAttention.forward = orig_forward
    assert len(captured) == num_layers
    return captured


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
                        default="/home/dblab/hj/impress_kv_store/stage4_rte")
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--ratios", type=str, default="1.0,0.5,0.25,0.1")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    ratios = [float(r) for r in args.ratios.split(",")]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b",
                                              padding_side="left")
    true_id = tokenizer(" True", add_special_tokens=False).input_ids[0]
    false_id = tokenizer(" False", add_special_tokens=False).input_ids[0]

    prefix_text, samples = build_rte(args.num_shots, args.num_samples,
                                     args.seed)
    prefix_ids = tokenizer(prefix_text).input_ids       # includes BOS
    queries, labels = [], []
    for ex in samples:
        q_text = "\n\n" + format_rte(ex)
        queries.append(tokenizer(q_text, add_special_tokens=False).input_ids)
        labels.append(ex["label"])
    m = len(prefix_ids)
    print(f"prefix: {args.num_shots}-shot, {m} tokens; "
          f"{len(samples)} RTE validation samples; "
          f"avg query len {np.mean([len(q) for q in queries]):.1f}")

    # ------------------------------------------------ model & environment
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
        # ---- prefix KV: prefill once -> Stage-1 store -> load all heads ----
        captured = capture_prefix_kv(model, prefix_ids, num_layers)
        store = PrefixKVStore(args.kv_dir)
        for j in range(num_layers):
            store.store_layer(j, *captured[j])
        store.finalize(args.model, num_layers, m)
        del captured
        prefix_kv = {}
        for j in range(num_layers):
            k_t, v_t = store.load_layer(j)   # all heads' keys AND values
            prefix_kv[j] = (k_t.to(gpu.dev), v_t.to(gpu.dev))
        print(f"prefix KV stored+loaded via Stage-1 store "
              f"({store.total_bytes() / 1e6:.1f} MB, chunk_size "
              f"{store.chunk_size})")

        controller = install_prefix_reuse_hook()
        controller.prefix_kv = prefix_kv
        controller.prefix_len = m
        controller.capture_logits = True

        def predict(input_ids):
            model.generate((input_ids,), max_new_tokens=1, do_sample=False)
            logits = controller.last_logits[0]
            return 0 if logits[true_id] > logits[false_id] else 1

        results = {}   # mode -> (accuracy, predictions)
        modes = [("ReComp", None)] + [(f"reuse@{r:.2f}", r) for r in ratios]
        for name, ratio in modes:
            preds = []
            controller.enabled = ratio is not None
            if ratio is not None:
                controller.retention_ratio = ratio
            for q_ids in queries:
                inputs = prefix_ids + q_ids if ratio is None else q_ids
                preds.append(predict(inputs))
            acc = float(np.mean([p == l for p, l in zip(preds, labels)]))
            kept = (f", kept {controller.last_kept[0]}/{controller.last_kept[1]}"
                    f" tokens/head" if ratio is not None and ratio < 1.0 else "")
            print(f"  {name:12s}: accuracy = {acc*100:5.1f}%{kept}")
            results[name] = (acc, preds)

        # ------------------------------------------------------ conclusions
        recomp_acc, recomp_preds = results["ReComp"]
        full_name = f"reuse@{ratios[0]:.2f}"
        full_acc, full_preds = results[full_name]
        agree = float(np.mean([a == b for a, b in
                               zip(recomp_preds, full_preds)]))
        print(f"\nagreement(ReComp vs {full_name}) = {agree*100:.1f}%")
        assert agree >= 0.9, (
            "full reuse should be numerically ~identical to recompute")

        accs = [results[f"reuse@{r:.2f}"][0] for r in ratios]
        drops = [f"{r:.0%}: {(a - recomp_acc)*100:+.1f}%"
                 for r, a in zip(ratios, accs)]
        print("accuracy delta vs ReComp:", ", ".join(drops))
        min_acc = min(accs)
        print(f"Figure 4 direction check: max drop "
              f"{(recomp_acc - min_acc)*100:.1f}%p at low retention "
              f"({'consistent with paper trend' if min_acc <= full_acc + 1e-9 else 'no drop observed'})")
        print("DONE")
    finally:
        uninstall_prefix_reuse_hook()
        env.close_copy_threads()


if __name__ == "__main__":
    main()

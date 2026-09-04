"""Stage 3 verification: token importance metric (paper §4.1, H2O column sum).

Part 1 (CPU, synthetic): a hand-built 3-head x 4-token attention weight
tensor; function output must equal hand-computed column sums, for every
accepted input layout (2D / 3D per-head / FlexGen's (b*n_head, q, k) / 4D).

Part 2 (GPU, OPT-6.7B): run prefill on a ~32-token prefix with the
importance hook installed; must produce a finite (num_layers, n_head, s)
score tensor with per-head total ~= s (each softmax row sums to 1), without
crashing. No token filtering — scores are only computed and inspected.

Usage:
  python -m flexllmgen.impress.verify_importance             # both parts
  python -m flexllmgen.impress.verify_importance --skip-model
"""

import argparse

import numpy as np
import torch

from flexllmgen.impress.importance import (column_sum_importance,
    to_token_score_pairs, install_importance_hook, uninstall_importance_hook)


# ------------------------------------------------------------- synthetic
def test_synthetic():
    # 3 heads x (4 query x 4 key), row-stochastic like real softmax output
    h0 = [[1.0, 0.0, 0.0, 0.0],
          [0.5, 0.5, 0.0, 0.0],
          [0.2, 0.3, 0.5, 0.0],
          [0.1, 0.1, 0.4, 0.4]]
    h1 = [[1.00, 0.00, 0.00, 0.00],
          [0.25, 0.75, 0.00, 0.00],
          [0.25, 0.25, 0.50, 0.00],
          [0.25, 0.25, 0.25, 0.25]]
    h2 = [[1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 1.0]]
    w = torch.tensor([h0, h1, h2])                    # (n_head=3, q=4, k=4)
    # hand-computed column sums per head
    expected = torch.tensor([
        [1.80, 0.90, 0.90, 0.40],
        [1.75, 1.25, 0.75, 0.25],
        [1.00, 1.00, 1.00, 1.00],
    ])

    # (n_head, q, k) -> (n_head, k)
    out = column_sum_importance(w)
    assert torch.allclose(out, expected, atol=1e-6), out
    # (q, k) -> (k,)
    out2d = column_sum_importance(w[0])
    assert torch.allclose(out2d, expected[0], atol=1e-6), out2d
    # (token_id, score) pairs for one head
    pairs = to_token_score_pairs(out[0])
    assert [i for i, _ in pairs] == [0, 1, 2, 3], pairs
    assert all(abs(s - e) < 1e-6
               for (_, s), e in zip(pairs, [1.80, 0.90, 0.90, 0.40])), pairs

    # batched: b=2, second batch = 10x the first (distinct values per batch)
    w4 = torch.stack([w, w * 10.0])                    # (b=2, n_head, q, k)
    exp4 = torch.stack([expected, expected * 10.0])    # (b=2, n_head, k)
    out4 = column_sum_importance(w4)
    assert torch.allclose(out4, exp4, atol=1e-5), out4
    # FlexGen layout: (b * n_head, q, k) + n_head — must invert the view
    # exactly as pytorch_backend.py:341 does (b, n_head, s, s) -> (b*n_head, s, s)
    out_bh = column_sum_importance(w4.view(6, 4, 4), n_head=3)
    assert torch.allclose(out_bh, exp4, atol=1e-5), out_bh

    print("Part 1 (synthetic 4-token x 3-head): PASSED "
          "(2D / 3D / 4D / FlexGen b*n_head layouts, hand-computed sums)")


# ----------------------------------------------------------------- model
def build_inputs(prompt_len, vocab_size):
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/opt-30b", padding_side="left")
        prompt = "Paris is the capital city of France. " * 8
        ids = tokenizer([prompt], truncation=True, padding="max_length",
                        max_length=prompt_len).input_ids[0][:prompt_len]
        return ids
    except Exception as e:
        print(f"[warn] tokenizer unavailable ({e}); using random token ids")
        rng = np.random.RandomState(0)
        return rng.randint(10, min(vocab_size - 1, 30000),
                           size=prompt_len).tolist()


def test_opt(model_name, weights_path, offload_dir, prompt_len,
             w_gpu=100, w_cpu=0):
    from flexllmgen.compression import CompressionConfig
    from flexllmgen.flex_opt import Policy, OptLM, get_opt_config
    from flexllmgen.pytorch_backend import (TorchDevice, TorchDisk,
        TorchMixedDevice)
    from flexllmgen.utils import ExecutionEnv

    config = get_opt_config(model_name)
    inputs = (build_inputs(prompt_len, config.vocab_size),)

    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk,
                       mixed=TorchMixedDevice([gpu, cpu, disk]))
    policy = Policy(1, 1, w_gpu, w_cpu, 100, 0, 100, 0,
                    overlap=False, sep_layer=True, pin_weight=True,
                    cpu_cache_compute=False, attn_sparsity=1.0,
                    compress_weight=False,
                    comp_weight_config=CompressionConfig(
                        num_bits=4, group_size=64, group_dim=0, symmetric=False),
                    compress_cache=False,
                    comp_cache_config=CompressionConfig(
                        num_bits=4, group_size=64, group_dim=2, symmetric=False))

    collector = install_importance_hook()
    try:
        print(f"init weights: {model_name} ...")
        model = OptLM(config, env, weights_path, policy)
        collector.enabled = True
        model.generate(inputs, max_new_tokens=1, do_sample=False)
        collector.enabled = False

        num_layers, n_head, s = (config.num_hidden_layers, config.n_head,
                                 prompt_len)
        assert sorted(collector.scores) == list(range(num_layers)), (
            f"captured layers: {sorted(collector.scores)}")
        stacked = collector.stacked()          # (num_layers, b, n_head, s)
        assert stacked.shape == (num_layers, 1, n_head, s), stacked.shape
        scores = stacked[:, 0]                 # (num_layers, n_head, s)

        assert torch.isfinite(scores).all(), "non-finite importance scores"
        assert (scores >= 0).all(), "negative importance scores"
        # each softmax row sums to 1 -> per-head column sums total ~= s
        head_totals = scores.sum(dim=-1)       # (num_layers, n_head)
        assert torch.allclose(head_totals,
                              torch.full_like(head_totals, float(s)),
                              atol=0.5), (
            f"head totals deviate from s={s}: "
            f"min={head_totals.min():.3f}, max={head_totals.max():.3f}")

        for j in (0, num_layers // 2):
            top = torch.topk(scores[j, 0], k=5).indices.tolist()
            print(f"  layer {j:2d} head 0: top-5 important token idx = {top}")
        print(f"Part 2 (OPT): PASSED — score tensor "
              f"(layers={num_layers}, heads={n_head}, tokens={s}), finite, "
              f"per-head totals ~= {s} "
              f"(range [{head_totals.min():.3f}, {head_totals.max():.3f}])")
    finally:
        uninstall_importance_hook()
        env.close_copy_threads()


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
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()

    test_synthetic()
    if not args.skip_model:
        test_opt(args.model, args.path, args.offload_dir, args.prompt_len,
                 args.w_gpu_percent, args.w_cpu_percent)
    print("ALL PASSED")


if __name__ == "__main__":
    main()

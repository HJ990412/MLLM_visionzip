"""Stage 1 end-to-end verification of the IMPRESS data plane on OPT.

Goal (no importance filtering yet):
  (a) the prefill K/V of every layer, stored to disk in 64-token chunks via
      PrefixKVStore and loaded back, is bit-exact vs the original tensors;
  (b) decoding that continues from the disk-loaded KV produces exactly the
      same output tokens as the recompute baseline.

Method — FlexGen pipeline code is NOT modified; SelfAttention.forward is
wrapped (monkey-patched) from this script:
  Run A (baseline / capture): normal generate(). At prefill (i == 0) the
      freshly computed per-layer KV in cache_write_buf is copied to CPU and
      kept. This run's output is the recompute reference.
  Store/Load: captured KV -> PrefixKVStore (chunked .npy on disk) -> load
      back -> bit-exact comparison per layer.                      -> (a)
  Run B (replace): generate() again on the same inputs, but at prefill the
      KV in cache_write_buf is REPLACED by the disk-loaded tensors, so all
      decoding steps read disk-roundtripped KV. Output must equal Run A.
                                                                   -> (b)

Usage:
  python -m flexllmgen.impress.verify_stage1 \
      --model facebook/opt-6.7b --prompt-len 512 --gen-len 8
"""

import argparse
import os

import numpy as np
import torch

from flexllmgen.compression import CompressionConfig
from flexllmgen.flex_opt import Policy, OptLM, SelfAttention, get_opt_config
from flexllmgen.pytorch_backend import (TorchDevice, TorchDisk,
    TorchMixedDevice, TorchTensor)
from flexllmgen.utils import ExecutionEnv
from flexllmgen.impress.prefix_kv import PrefixKVStore, DEFAULT_CHUNK_SIZE


class PrefillKVHook:
    """State shared with the wrapped SelfAttention.forward."""

    def __init__(self):
        self.mode = None          # None | "capture" | "replace"
        self.captured = {}        # layer_id -> (k_cpu, v_cpu) torch tensors
        self.replacement = {}     # layer_id -> (k_cpu, v_cpu) torch tensors


HOOK = PrefillKVHook()
_ORIG_FORWARD = SelfAttention.forward


def _hooked_forward(self, hidden, cache_read_buf, weight_read_buf,
                    attention_mask, cache_write_buf, i, k):
    _ORIG_FORWARD(self, hidden, cache_read_buf, weight_read_buf,
                  attention_mask, cache_write_buf, i, k)
    if i != 0 or HOOK.mode is None:
        return

    # Prefill just ran: cache_write_buf holds this layer's full prefix KV,
    # shape (prompt_len, b * n_head, head_dim), fp16, on the compute device.
    k_t, v_t = cache_write_buf.pop()

    if HOOK.mode == "capture":
        HOOK.captured[self.layer_id] = (
            k_t.data.detach().to("cpu").contiguous().clone(),
            v_t.data.detach().to("cpu").contiguous().clone(),
        )
        cache_write_buf.store((k_t, v_t))
    elif HOOK.mode == "replace":
        lk, lv = HOOK.replacement[self.layer_id]
        dev = k_t.device  # TorchDevice of the compute device (GPU)
        assert tuple(lk.shape) == tuple(k_t.shape), (
            f"layer {self.layer_id}: loaded {tuple(lk.shape)} "
            f"vs computed {tuple(k_t.shape)}")
        new_k = TorchTensor.create_from_torch(lk.to(dev.dev), dev)
        new_v = TorchTensor.create_from_torch(lv.to(dev.dev), dev)
        cache_write_buf.store((new_k, new_v))
    else:
        raise ValueError(HOOK.mode)


SelfAttention.forward = _hooked_forward


def build_inputs(prompt_len, vocab_size, pad_token_id):
    """One prompt of exactly prompt_len tokens. Tries the OPT tokenizer
    (left-padded, as in flex_opt.get_test_inputs); falls back to
    deterministic random ids if the tokenizer is unavailable (offline)."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/opt-30b", padding_side="left")
        prompt = ("Paris is the capital city of France. " * 200)
        ids = tokenizer([prompt], truncation=True, padding="max_length",
                        max_length=prompt_len).input_ids[0]
        ids = ids[:prompt_len]
        return ids, tokenizer
    except Exception as e:
        print(f"[warn] tokenizer unavailable ({e}); using random token ids")
        rng = np.random.RandomState(0)
        ids = rng.randint(10, min(vocab_size - 1, 30000),
                          size=prompt_len).tolist()
        return ids, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-6.7b")
    parser.add_argument("--w-gpu-percent", type=int, default=100,
        help="weight %% resident on GPU (13B: ~70, 30B: ~30 on a 24GB GPU)")
    parser.add_argument("--w-cpu-percent", type=int, default=0,
        help="weight %% on CPU (rest goes to disk)")
    parser.add_argument("--path", type=str, default="~/opt_weights",
        help="Directory with converted numpy weights (<model>-np).")
    parser.add_argument("--offload-dir", type=str,
        default="~/flexllmgen_offload_dir")
    parser.add_argument("--kv-dir", type=str,
        default="/home/dblab/hj/impress_kv_store/stage1",
        help="Where prefix KV chunks are written.")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()

    config = get_opt_config(args.model)
    input_ids, tokenizer = build_inputs(
        args.prompt_len, config.vocab_size, config.pad_token_id)
    inputs = (input_ids,)  # batch size 1

    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk,
                       mixed=TorchMixedDevice([gpu, cpu, disk]))

    # Everything on GPU, no overlap: deterministic, simplest code path
    # (generation_loop_normal), batch size 1.
    policy = Policy(1, 1,
                    args.w_gpu_percent, args.w_cpu_percent,  # weights
                    100, 0,   # cache: GPU
                    100, 0,   # activations: GPU
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

    try:
        # ---- Run A: recompute baseline + capture prefill KV ----
        print(f"Run A (baseline+capture): prompt_len={args.prompt_len}, "
              f"gen_len={args.gen_len}")
        HOOK.mode = "capture"
        out_a = model.generate(inputs, max_new_tokens=args.gen_len,
                               do_sample=False)
        HOOK.mode = None
        num_layers = config.num_hidden_layers
        assert len(HOOK.captured) == num_layers, (
            f"captured {len(HOOK.captured)} layers, expected {num_layers}")

        # ---- Store to disk in chunks ----
        store = PrefixKVStore(args.kv_dir, chunk_size=args.chunk_size)
        for j in range(num_layers):
            store.store_layer(j, *HOOK.captured[j])
        store.finalize(args.model, num_layers, args.prompt_len,
                       extra={"n_head": config.n_head,
                              "head_dim": config.input_dim // config.n_head})
        meta0 = store.layer(0).meta
        print(f"stored {num_layers} layers to {store.root}: "
              f"{meta0['num_chunks']} chunks/tensor/layer "
              f"(chunk_size={args.chunk_size}, "
              f"last chunk={meta0['chunks'][-1]['num_tokens']} tokens), "
              f"total {store.total_bytes() / 1e9:.3f} GB")

        # ---- (a) load back and compare bit-exact ----
        mismatched = []
        for j in range(num_layers):
            lk, lv = store.load_layer(j)
            ck, cv = HOOK.captured[j]
            if not (torch.equal(lk, ck) and torch.equal(lv, cv)):
                mismatched.append(j)
            HOOK.replacement[j] = (lk, lv)
        if mismatched:
            print(f"(a) FAILED: KV mismatch in layers {mismatched}")
        else:
            print(f"(a) PASSED: all {num_layers} layers bit-exact "
                  f"after disk roundtrip")

        # ---- Run B: decode from disk-loaded KV ----
        print("Run B (replace prefill KV with disk-loaded KV)")
        HOOK.mode = "replace"
        out_b = model.generate(inputs, max_new_tokens=args.gen_len,
                               do_sample=False)
        HOOK.mode = None

        gen_a = out_a[0, args.prompt_len:]
        gen_b = out_b[0, args.prompt_len:]
        same = np.array_equal(out_a, out_b)
        print(f"  recompute  tokens: {gen_a.tolist()}")
        print(f"  disk-load  tokens: {gen_b.tolist()}")
        if tokenizer is not None:
            print(f"  recompute  text: {tokenizer.decode(gen_a)!r}")
            print(f"  disk-load  text: {tokenizer.decode(gen_b)!r}")
        print(f"(b) {'PASSED' if same else 'FAILED'}: decoding from "
              f"disk-loaded KV {'==' if same else '!='} recompute baseline")

        ok = not mismatched and same
        print("RESULT:", "ALL PASSED" if ok else "FAILED")
        return 0 if ok else 1
    finally:
        env.close_copy_threads()


if __name__ == "__main__":
    raise SystemExit(main())

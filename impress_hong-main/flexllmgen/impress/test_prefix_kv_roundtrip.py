"""Quick CPU-only sanity test for PrefixKVLayer/PrefixKVStore.

Random tensors only, no model. Checks:
- bit-exact roundtrip (fp16/fp32), including a non-multiple-of-chunk-size
  token count (last chunk is partial)
- chunk metadata (count, per-chunk token counts, file existence)
- reopening a store from disk (fresh objects, metadata-driven load)

Usage: python -m flexllmgen.impress.test_prefix_kv_roundtrip
"""

import os
import shutil
import tempfile

import numpy as np
import torch

from flexllmgen.impress.prefix_kv import PrefixKVLayer, PrefixKVStore


def test_layer_roundtrip(tmp, num_tokens, chunk_size, dtype):
    torch.manual_seed(0)
    k = torch.randn(num_tokens, 32, 128, dtype=dtype)
    v = torch.randn(num_tokens, 32, 128, dtype=dtype)

    layer_dir = os.path.join(tmp, f"layer_t{num_tokens}_c{chunk_size}_{dtype}")
    layer = PrefixKVLayer(0, layer_dir, chunk_size)
    meta = layer.store(k, v)

    expected_chunks = (num_tokens + chunk_size - 1) // chunk_size
    assert meta["num_chunks"] == expected_chunks, meta["num_chunks"]
    assert sum(c["num_tokens"] for c in meta["chunks"]) == num_tokens
    last = meta["chunks"][-1]
    assert last["num_tokens"] == num_tokens - last["start"]
    for c in meta["chunks"]:
        assert os.path.exists(os.path.join(layer_dir, c["k_file"]))
        assert os.path.exists(os.path.join(layer_dir, c["v_file"]))

    # load through a *fresh* object so only disk state is used
    k2, v2 = PrefixKVLayer(0, layer_dir).load()
    assert k2.dtype == dtype and v2.dtype == dtype
    assert torch.equal(k, k2), "K roundtrip not bit-exact"
    assert torch.equal(v, v2), "V roundtrip not bit-exact"

    # single-chunk load matches the corresponding slice
    kc, vc = layer.load_chunk(expected_chunks - 1)
    start = last["start"]
    assert np.array_equal(kc, k[start:].numpy())
    assert np.array_equal(vc, v[start:].numpy())
    print(f"  OK: tokens={num_tokens} chunk_size={chunk_size} dtype={dtype} "
          f"chunks={expected_chunks} (last={last['num_tokens']})")


def test_store_roundtrip(tmp):
    torch.manual_seed(1)
    root = os.path.join(tmp, "store")
    num_layers, num_tokens = 4, 130  # 130 = 2 full chunks + 2 tokens
    store = PrefixKVStore(root, chunk_size=64)
    ref = {}
    for j in range(num_layers):
        k = torch.randn(num_tokens, 8, 16, dtype=torch.float16)
        v = torch.randn(num_tokens, 8, 16, dtype=torch.float16)
        store.store_layer(j, k, v)
        ref[j] = (k, v)
    store.finalize("dummy-model", num_layers, num_tokens)

    reopened = PrefixKVStore(root)
    assert reopened.meta["num_layers"] == num_layers
    assert reopened.meta["chunk_size"] == 64
    for j in range(num_layers):
        k2, v2 = reopened.load_layer(j)
        assert torch.equal(ref[j][0], k2)
        assert torch.equal(ref[j][1], v2)
    assert reopened.total_bytes() > 0
    print(f"  OK: store with {num_layers} layers x {num_tokens} tokens, "
          f"{reopened.total_bytes() / 1e6:.2f} MB on disk")


if __name__ == "__main__":
    tmp = tempfile.mkdtemp(prefix="impress_kv_test_")
    try:
        test_layer_roundtrip(tmp, num_tokens=512, chunk_size=64, dtype=torch.float16)
        test_layer_roundtrip(tmp, num_tokens=517, chunk_size=64, dtype=torch.float16)
        test_layer_roundtrip(tmp, num_tokens=30, chunk_size=64, dtype=torch.float32)
        test_store_roundtrip(tmp)
        print("ALL PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

"""Chunked disk storage for prefix KV (IMPRESS data plane, Stage 1).

Follows the IMPRESS paper (FAST'25):
- §6.1: "Uniformly, each chunk holds keys or values from 64 tokens" — a chunk
  contains the keys OR the values of `chunk_size` consecutive prefix tokens;
  K chunks and V chunks are separate objects (files).
- §4.4.1: per-node metadata keeps pointer lists to key/value chunks (k_ptr /
  v_ptr) plus a token-order mapping list. At this stage there is no radix
  tree and no reordering, so the mapping is identity (`token_order = None`)
  and metadata lives in one JSON file per layer.
- §5: "For KV reordering, we implemented the PrefixKVLayer class to store
  reordered KVs and mapping lists per layer" — PrefixKVLayer is the per-layer
  storage unit; reordering support comes in a later stage.

Tensor layout matches FlexGen's KV cache layout produced by TorchDevice.mha
(pytorch_backend.py:354-365): (num_tokens, b * n_head, head_dim). Only dim 0
(token) is chunked; trailing dims are stored as-is, so any (tokens, ...) tensor
works.

Chunk files are plain .npy files (same on-disk format FlexGen's TorchDisk
uses), so a later stage can wrap them in TorchTensor/general_copy for async
3-tier movement without changing the format.
"""

import json
import os

import numpy as np
import torch

DEFAULT_CHUNK_SIZE = 64  # tokens per chunk, paper §6.1
LAYER_META_NAME = "meta.json"
STORE_META_NAME = "meta.json"

_STR_TO_TORCH_DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _to_numpy(tensor):
    """torch.Tensor (any device) or np.ndarray -> contiguous np.ndarray."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().to("cpu").contiguous().numpy()
    return np.ascontiguousarray(tensor)


class PrefixKVLayer:
    """Chunked disk storage for one transformer layer's prefix K/V.

    Directory layout (one directory per layer):
        <dir>/meta.json
        <dir>/chunk_00000.k.npy   # keys  of tokens [0, chunk_size)
        <dir>/chunk_00000.v.npy   # values of tokens [0, chunk_size)
        <dir>/chunk_00001.k.npy
        ...
    The last chunk may hold fewer than chunk_size tokens; per-chunk token
    counts are recorded in meta.json.
    """

    def __init__(self, layer_id, storage_dir, chunk_size=DEFAULT_CHUNK_SIZE):
        assert chunk_size > 0
        self.layer_id = layer_id
        self.dir = os.path.abspath(os.path.expanduser(storage_dir))
        self.chunk_size = chunk_size
        self.meta = None
        self._maybe_load_meta()

    def _meta_path(self):
        return os.path.join(self.dir, LAYER_META_NAME)

    def _maybe_load_meta(self):
        if os.path.exists(self._meta_path()):
            with open(self._meta_path()) as f:
                self.meta = json.load(f)
            assert self.meta["layer_id"] == self.layer_id, (
                f"layer dir {self.dir} holds layer {self.meta['layer_id']}, "
                f"expected {self.layer_id}")
            self.chunk_size = self.meta["chunk_size"]

    # ------------------------------------------------------------------ store
    def store(self, k, v, token_order=None):
        """Chunk K/V along the token dim (dim 0) and write to disk.

        k, v: (num_tokens, ...) with identical shape/dtype.
        token_order: the §4.4.1 mapping list m when k/v are stored in
        reordered form (stored[m] recovers the original order); None means
        identity (original order).
        Returns the written metadata dict.
        """
        k = _to_numpy(k)
        v = _to_numpy(v)
        assert k.shape == v.shape, f"K/V shape mismatch: {k.shape} vs {v.shape}"
        assert k.dtype == v.dtype, f"K/V dtype mismatch: {k.dtype} vs {v.dtype}"
        num_tokens = k.shape[0]
        assert num_tokens > 0

        os.makedirs(self.dir, exist_ok=True)

        chunks = []
        for idx, start in enumerate(range(0, num_tokens, self.chunk_size)):
            end = min(start + self.chunk_size, num_tokens)
            k_file = f"chunk_{idx:05d}.k.npy"
            v_file = f"chunk_{idx:05d}.v.npy"
            np.save(os.path.join(self.dir, k_file), k[start:end])
            np.save(os.path.join(self.dir, v_file), v[start:end])
            chunks.append({
                "idx": idx,
                "start": start,
                "num_tokens": end - start,
                "k_file": k_file,
                "v_file": v_file,
            })

        self.meta = {
            "layer_id": self.layer_id,
            "chunk_size": self.chunk_size,
            "num_tokens": num_tokens,
            "tail_shape": list(k.shape[1:]),
            "dtype": str(k.dtype),
            "num_chunks": len(chunks),
            # Token-order mapping list (paper §4.4.1). None = identity order.
            "token_order": (list(token_order) if token_order is not None
                            else None),
            "chunks": chunks,
        }
        with open(self._meta_path(), "w") as f:
            json.dump(self.meta, f, indent=1)
        return self.meta

    # ------------------------------------------------------------------- load
    def load_chunk(self, idx):
        """Load one chunk. Returns (k_chunk, v_chunk) as np.ndarray."""
        assert self.meta is not None, f"no metadata in {self.dir}"
        info = self.meta["chunks"][idx]
        k = np.load(os.path.join(self.dir, info["k_file"]))
        v = np.load(os.path.join(self.dir, info["v_file"]))
        assert k.shape[0] == info["num_tokens"], (
            f"chunk {idx}: file holds {k.shape[0]} tokens, "
            f"metadata says {info['num_tokens']}")
        return k, v

    def load(self, device=None):
        """Reassemble all chunks into full (num_tokens, ...) K/V tensors.

        Returns (k, v) as torch.Tensor on `device` (default: cpu).
        """
        assert self.meta is not None, f"no metadata in {self.dir}"
        num_tokens = self.meta["num_tokens"]
        full_shape = (num_tokens, *self.meta["tail_shape"])
        np_dtype = np.dtype(self.meta["dtype"])

        k_full = np.empty(full_shape, dtype=np_dtype)
        v_full = np.empty(full_shape, dtype=np_dtype)
        covered = 0
        for info in self.meta["chunks"]:
            k_c, v_c = self.load_chunk(info["idx"])
            start, n = info["start"], info["num_tokens"]
            k_full[start:start + n] = k_c
            v_full[start:start + n] = v_c
            covered += n
        assert covered == num_tokens, (
            f"chunks cover {covered} tokens, expected {num_tokens}")

        k_t = torch.from_numpy(k_full)
        v_t = torch.from_numpy(v_full)
        if device is not None:
            k_t = k_t.to(device)
            v_t = v_t.to(device)
        return k_t, v_t


class PrefixKVStore:
    """All layers of one stored prefix.

    Directory layout:
        <root>/meta.json          # prefix-level metadata
        <root>/layer_00/          # PrefixKVLayer directory
        <root>/layer_01/
        ...
    """

    def __init__(self, root, chunk_size=DEFAULT_CHUNK_SIZE):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.chunk_size = chunk_size
        self._layers = {}
        self.meta = None
        meta_path = os.path.join(self.root, STORE_META_NAME)
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.meta = json.load(f)
            self.chunk_size = self.meta["chunk_size"]

    def _layer_dir(self, layer_id):
        return os.path.join(self.root, f"layer_{layer_id:02d}")

    def layer(self, layer_id):
        if layer_id not in self._layers:
            self._layers[layer_id] = PrefixKVLayer(
                layer_id, self._layer_dir(layer_id), self.chunk_size)
        return self._layers[layer_id]

    def store_layer(self, layer_id, k, v, token_order=None):
        return self.layer(layer_id).store(k, v, token_order=token_order)

    def load_layer(self, layer_id, device=None):
        return self.layer(layer_id).load(device=device)

    def finalize(self, model_name, num_layers, num_tokens, extra=None):
        """Write prefix-level metadata after all layers are stored."""
        self.meta = {
            "model_name": model_name,
            "num_layers": num_layers,
            "num_tokens": num_tokens,
            "chunk_size": self.chunk_size,
        }
        if extra:
            self.meta.update(extra)
        os.makedirs(self.root, exist_ok=True)
        with open(os.path.join(self.root, STORE_META_NAME), "w") as f:
            json.dump(self.meta, f, indent=1)
        return self.meta

    def total_bytes(self):
        total = 0
        for dirpath, _, files in os.walk(self.root):
            for name in files:
                if name.endswith(".npy"):
                    total += os.path.getsize(os.path.join(dirpath, name))
        return total

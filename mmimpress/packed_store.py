"""PackedPrefixStore for PrefixStore-KV25 (``prefixstore_kv25_packed_v1``).

One image == one ``kv_packed.bin``::

    [L0 K rows 0..k_0) [L0 V rows 0..k_0) [L1 K] [L1 V] ... [L31 V] [separator K/V sidecar]

FP16, token-major ``(rows, num_heads, head_dim)`` per block, in the global
image-only VisionZip order.  Every byte of every block is produced by the same
operations as the unpacked ``prefix25`` / ``prefix25_layerwise`` writer in
``mmimpress.store.write_image_store`` (materialise, permute, slice ``[:k_l]``,
``tobytes``), so the packed slices are bit-identical to ``layer_xx/{k,v}.bin``
and the trailing block is bit-identical to ``sep_kv.bin``.  ``sys_kv.pt``,
``visionzip_layout.pt``, ``meta.json`` and ``integrity.json`` stay separate
files, read at activation only.

The per-layer row counts ``k_l`` always come from a frozen
``mmimpress.layer_budget`` document (the uniform vector reproduces the
``prefix25`` rows exactly), so every packed store carries ``budget_policy =
"layerwise_v1"`` metadata and passes the v1 layerwise geometry validator.

Reading: ``PackedReader.read_all`` issues exactly ONE ``os.preadv`` of the whole
file into a page-locked host buffer allocated at activation (outside every
request timer); layers are zero-copy views into that buffer.  ``os.pread`` is
not used on the hit path (it would return a fresh ``bytes`` object and force a
second host copy into pinned memory).

Contract: docs/layer_budget_kv25_v2_contract.md (A3).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from mmimpress.store import META_NAME, n_chunks

STORE_FORMAT = "prefixstore_kv25_packed_v1"
STORAGE_POLICY = "prefix25_packed"
META_FORMAT = "llava_fp16_importance_prefix25_packed_v1"
LAYERWISE_META_FORMAT = "llava_fp16_importance_prefix25_layerwise_v1"
PACKED_NAME = "kv_packed.bin"
_TORCH_DTYPE = {"float16": torch.float16}
_NP_DTYPE = {"float16": np.float16}


# ------------------------------------------------------------------ layout
def packed_layout(counts, row_bytes: int, num_layers: int, n_sep: int):
    """Byte table of the packed file for per-layer row counts ``counts``."""
    counts = [int(c) for c in counts]
    if len(counts) != int(num_layers) or any(c < 0 for c in counts):
        raise ValueError("invalid per-layer row counts")
    off = 0
    table = []
    for li, k in enumerate(counts):
        k_off = off
        off += k * row_bytes
        v_off = off
        off += k * row_bytes
        table.append({"layer": li, "k_offset": k_off, "v_offset": v_off, "rows": k,
                      "bytes": k * row_bytes})
    sep_bytes = 2 * int(num_layers) * int(n_sep) * row_bytes
    return table, off, sep_bytes, off + sep_bytes


# ------------------------------------------------------------------ writer
def write_packed_store(out_dir: Path, layers, v_start: int, v_num: int,
                       prefix_input_ids, newline_idx, extra=None,
                       chunk_size: int = 64, dtype: str = "float16",
                       probe_heads: int = 0, stored_to_original=None,
                       separator_sidecar: bool = True,
                       timing_out: dict | None = None,
                       storage_policy: str = STORAGE_POLICY,
                       layer_budget: dict | None = None):
    """Write one packed image prefix.  Same signature/semantics as
    ``mmimpress.store.write_image_store`` so it can stand in for it inside
    the Turn-1 piggyback persistence flow; returns the meta dict."""
    from mmimpress.layer_budget import (POLICY, budget_sha256, layer_keep_counts,
                                        total_row_budget, validate_ratios)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if storage_policy != STORAGE_POLICY:
        raise ValueError("packed writer serves only the prefix25_packed policy")
    if layer_budget is None:
        raise ValueError("packed stores require a frozen layer budget document")
    if probe_heads or not separator_sidecar or chunk_size != 64 or dtype != "float16":
        raise ValueError("packed store requires probe_heads=0, separator sidecar, chunk 64, float16")
    td = _TORCH_DTYPE[dtype]
    prefix_len = v_start + v_num
    H, hd = layers[0][0].shape[1], layers[0][0].shape[3]
    assert layers[0][0].shape[2] >= prefix_len, layers[0][0].shape
    L = len(layers)
    n_content = v_num - len(newline_idx)
    if n_content < 1 or stored_to_original is None:
        raise ValueError("packed store requires nonempty repacked content and a permutation")
    if list(stored_to_original)[n_content:] != list(newline_idx):
        raise ValueError("packed store requires a stable structural tail")
    ratios = validate_ratios(layer_budget["layer_keep_ratios"], L,
                             layer_budget.get("total_ratio", 0.25),
                             layer_budget["r_min"], layer_budget["r_max"])
    if layer_budget.get("layer_budget_sha256") != budget_sha256(ratios):
        raise ValueError("layer budget SHA256 does not match its ratios")
    if layer_budget.get("policy", POLICY) != POLICY:
        raise ValueError("unsupported layer budget policy")
    rows_per_layer = layer_keep_counts(ratios, n_content, L,
                                       layer_budget.get("total_ratio", 0.25),
                                       layer_budget["r_min"], layer_budget["r_max"])
    stored_content = (n_content + 3) // 4
    assert sum(rows_per_layer) == total_row_budget(n_content, L) == L * stored_content

    order = [int(i) for i in stored_to_original]
    assert len(order) == v_num and sorted(order) == list(range(v_num)), \
        "stored_to_original is not a visual-token permutation"
    perm = torch.tensor(order, dtype=torch.long)
    inverse = [0] * v_num
    for stored, original in enumerate(order):
        inverse[original] = stored
    newline_original = [int(i) for i in newline_idx]
    newline_stored = [inverse[i] for i in newline_original]
    sep_pos = torch.tensor(sorted(newline_stored), dtype=torch.long)
    n_sep = len(newline_original)
    row_bytes = H * hd * 2
    table, sep_offset, sep_bytes, total_bytes = packed_layout(rows_per_layer, row_bytes, L, n_sep)

    materialize_seconds = repack_seconds = write_seconds = serialize_seconds = hash_seconds = 0.0
    t0 = time.perf_counter()
    sys_k = torch.stack([k[0, :, :v_start].to(td).cpu() for k, _ in layers])
    sys_v = torch.stack([v[0, :, :v_start].to(td).cpu() for _, v in layers])
    materialize_seconds += time.perf_counter() - t0
    t0 = time.perf_counter()
    torch.save({"k": sys_k, "v": sys_v}, out_dir / "sys_kv.pt")
    write_seconds += time.perf_counter() - t0

    digest = hashlib.sha256()
    written = 0
    sep_k, sep_v = [], []
    path = out_dir / PACKED_NAME
    with path.open("wb") as f:
        for li, (k, v) in enumerate(layers):
            for kind, t in (("k", k), ("v", v)):
                t0 = time.perf_counter()
                block = t[0, :, v_start:prefix_len].permute(1, 0, 2).to(td).contiguous().cpu()
                assert block.shape == (v_num, H, hd), block.shape
                materialize_seconds += time.perf_counter() - t0
                t0 = time.perf_counter()
                block = block.index_select(0, perm).contiguous()
                repack_seconds += time.perf_counter() - t0
                t0 = time.perf_counter()
                buf = block[:rows_per_layer[li]].numpy().tobytes()
                serialize_seconds += time.perf_counter() - t0
                expected_off = table[li]["k_offset" if kind == "k" else "v_offset"]
                assert written == expected_off, (li, kind, written, expected_off)
                t0 = time.perf_counter()
                f.write(buf)
                write_seconds += time.perf_counter() - t0
                t0 = time.perf_counter()
                digest.update(buf)
                hash_seconds += time.perf_counter() - t0
                written += len(buf)
                selected = block.index_select(0, sep_pos).clone()
                (sep_k if kind == "k" else sep_v).append(selected)
        assert written == sep_offset
        sep_blob = torch.stack([torch.stack(sep_k), torch.stack(sep_v)])
        assert tuple(sep_blob.shape) == (2, L, n_sep, H, hd)
        payload = sep_blob.numpy().tobytes()
        assert len(payload) == sep_bytes
        t0 = time.perf_counter()
        f.write(payload)
        write_seconds += time.perf_counter() - t0
        t0 = time.perf_counter()
        digest.update(payload)
        hash_seconds += time.perf_counter() - t0
        written += len(payload)
        f.flush()
    assert written == total_bytes == path.stat().st_size
    visual_bytes = total_bytes - sep_bytes
    max_rows = max(rows_per_layer)
    meta = {
        "v_token_start": v_start, "v_token_num": v_num, "prefix_len": prefix_len,
        "num_layers": L, "num_heads": H, "head_dim": hd, "probe_heads": 0,
        "dtype": dtype, "chunk_size": chunk_size,
        "n_chunks_per_layer": n_chunks(v_num, chunk_size),
        "newline_idx": list(newline_original), "n_spatial": n_content,
        "prefix_input_ids": list(prefix_input_ids),
        "layout": "packed: one kv_packed.bin = [L0 K rows 0..k_0)[L0 V][L1 K][L1 V]...[L31 V][separator K/V sidecar]; "
                  "each block token-major (rows, num_heads, head_dim) fp16 in the global importance order",
        "bytes_visual_kv": visual_bytes, "bytes_probe_sidecar": 0,
        "bytes_separator_sidecar": sep_bytes,
        "order": list(order), "order_is_per_layer": False,
        "newline_stored": newline_stored, "reordered": True,
        "format": META_FORMAT, "storage_policy": STORAGE_POLICY,
        "store_format": STORE_FORMAT, "packed_file": PACKED_NAME,
        "budget_policy": "layerwise_v1",
        "original_content_count": n_content,
        "uniform_equivalent_content_count": stored_content,
        "stored_content_count_per_layer": list(rows_per_layer),
        "stored_content_count_total": int(sum(rows_per_layer)),
        "payload_rows_per_layer": list(rows_per_layer),
        "retention_ratio": 0.25,
        "rounding_policy": "hamilton_largest_remainder_total_ceil_original_content",
        "layer_budget_id": str(layer_budget["alloc_id"]),
        "layer_keep_ratios": [float(x) for x in layer_budget["layer_keep_ratios"]],
        "layer_keep_counts": list(rows_per_layer),
        "layer_budget_sha256": str(layer_budget["layer_budget_sha256"]),
        "layer_budget_r_min": float(layer_budget["r_min"]),
        "layer_budget_r_max": float(layer_budget["r_max"]),
        "stored_row_to_original": list(order)[:max_rows],
        "full_importance_permutation": list(order),
        "order_semantics": "full virtual GPU permutation; layer l stores stored_row_to_original[:layer_keep_counts[l]]",
        "payload_chunks_per_layer": [n_chunks(r, chunk_size) for r in rows_per_layer],
        "valid_rows_last_chunk_per_layer": [((r - 1) % chunk_size + 1) if r else 0 for r in rows_per_layer],
        "padding_rows": 0,
        "packed_row_bytes": row_bytes, "packed_layout": table,
        "packed_sep_offset": sep_offset, "packed_sep_bytes": sep_bytes,
        "packed_total_bytes": total_bytes, "packed_sha256": digest.hexdigest(),
    }
    meta["packed_writer_timing_ms"] = {
        "kv_materialize_ms": materialize_seconds * 1e3, "kv_repack_ms": repack_seconds * 1e3,
        "ssd_write_ms": write_seconds * 1e3, "prefix_slice_serialization_ms": serialize_seconds * 1e3,
        "packed_sha256_ms": hash_seconds * 1e3,
    }
    if extra:
        meta.update(extra)
    assert meta["order"] == order and meta["order_is_per_layer"] is False
    assert meta["newline_stored"] == newline_stored
    t0 = time.perf_counter()
    with open(out_dir / META_NAME, "w") as f:
        json.dump(meta, f, indent=1)
    write_seconds += time.perf_counter() - t0
    if timing_out is not None:
        timing_out.update({
            "kv_materialize_ms": materialize_seconds * 1e3,
            "kv_repack_ms": repack_seconds * 1e3,
            "ssd_write_ms": write_seconds * 1e3,
            "prefix_slice_serialization_ms": serialize_seconds * 1e3,
            "packed_sha256_ms": hash_seconds * 1e3,
        })
    return meta


# --------------------------------------------------------------- validator
def validate_packed_meta(m: dict) -> None:
    """Fail-closed check of a packed store's metadata (geometry, vector,
    counts, byte table).  Delegates every shared layerwise check to the v1
    validator on a format-relabelled copy."""
    from mmimpress.prefix25 import validate_llava_prefix_layerwise_meta
    if (m.get("format") != META_FORMAT or m.get("storage_policy") != STORAGE_POLICY
            or m.get("store_format") != STORE_FORMAT or m.get("packed_file") != PACKED_NAME
            or m.get("probe_heads") != 0 or m.get("bytes_probe_sidecar") != 0
            or m.get("dtype") != "float16" or 'payload_rows' in m):
        raise ValueError("invalid packed store format")
    relabelled = dict(m, format=LAYERWISE_META_FORMAT, storage_policy="prefix25_layerwise")
    validate_llava_prefix_layerwise_meta(relabelled)
    counts = [int(x) for x in m["layer_keep_counts"]]
    L = int(m["num_layers"])
    row_bytes = int(m["num_heads"]) * int(m["head_dim"]) * 2
    n_sep = len(m["newline_idx"])
    table, sep_offset, sep_bytes, total = packed_layout(counts, row_bytes, L, n_sep)
    if (m.get("packed_row_bytes") != row_bytes or m.get("packed_layout") != table
            or m.get("packed_sep_offset") != sep_offset or m.get("packed_sep_bytes") != sep_bytes
            or m.get("packed_total_bytes") != total or m.get("bytes_separator_sidecar") != sep_bytes
            or m.get("bytes_visual_kv") != total - sep_bytes
            or not isinstance(m.get("packed_sha256"), str) or len(m["packed_sha256"]) != 64):
        raise ValueError("packed byte table disagrees with the row counts")


def packed_file_sha256(path: Path, block: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ reader
class PackedReader:
    """One ``os.preadv`` of the whole packed file into a pinned host buffer.

    The buffer is allocated in ``__init__`` (activation time).  ``read_all``
    is the only thing that runs inside a request timer.
    """

    def __init__(self, store_dir: Path, meta: dict, pin: bool = True):
        self.dir = Path(store_dir)
        self.meta = meta
        self.path = self.dir / meta["packed_file"]
        self.total = int(meta["packed_total_bytes"])
        self.row_bytes = int(meta["packed_row_bytes"])
        self.H, self.hd, self.L = int(meta["num_heads"]), int(meta["head_dim"]), int(meta["num_layers"])
        self.n_sep = len(meta["newline_idx"])
        self.table = meta["packed_layout"]
        self.sep_offset = int(meta["packed_sep_offset"])
        if self.path.stat().st_size != self.total:
            raise ValueError("packed file length disagrees with metadata")
        self.buf = torch.empty(self.total, dtype=torch.uint8)
        self.pinned = False
        if pin and torch.cuda.is_available():
            self.buf = self.buf.pin_memory()
            self.pinned = True
        self._np = self.buf.numpy()
        self._fd = None
        self.read_calls = 0

    def _open(self):
        if self._fd is None:
            self._fd = os.open(self.path, os.O_RDONLY)
        return self._fd

    def drop_all(self):
        """Evict the packed file (and any other .bin) from the page cache."""
        for p in sorted(self.dir.rglob("*.bin")):
            fd = os.open(p, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)

    def read_all(self, counter=None):
        """The single timed read: whole file -> pinned buffer, one syscall."""
        fd = self._open()
        t0 = time.perf_counter()
        n = os.preadv(fd, [memoryview(self._np)], 0)
        dt = time.perf_counter() - t0
        if n != self.total:
            raise AssertionError(f"short packed read: {n} != {self.total}")
        self.read_calls += 1
        if counter is not None:
            counter.record("packed", n, dt, preads=1, units=0)
        return n, dt

    def layer_view(self, layer: int, kind: str) -> torch.Tensor:
        row = self.table[layer]
        off = int(row["k_offset" if kind == "k" else "v_offset"])
        k = int(row["rows"])
        return self.buf[off:off + k * self.row_bytes].view(torch.float16).view(k, self.H, self.hd)

    def sep_view(self) -> torch.Tensor:
        n = 2 * self.L * self.n_sep * self.H * self.hd
        return self.buf[self.sep_offset:self.sep_offset + n * 2].view(torch.float16).view(2, self.L, self.n_sep, self.H, self.hd)

    def close(self):
        if self._fd is not None:
            os.posix_fadvise(self._fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(self._fd)
            self._fd = None
        self._np = None
        self.buf = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ----------------------------------------------------------- persistence
def persist_captured_packed_prefix(runner, captured_past_key_values, expanded_input_ids,
                                   image_size, per_sub_scores, out_dir, *, layer_budget,
                                   **kw):
    """Turn-1 piggyback persistence of a packed store.

    Runs the unmodified ``mmimpress.piggyback.persist_captured_visual_prefix``
    flow (capture-provenance checks, image-only VisionZip permutation,
    staging, fsync, atomic no-clobber publication, receipts) with the packed
    writer standing in for ``write_image_store`` for the duration of the
    call.  The unpacked writer and every existing code path are untouched.
    """
    import mmimpress.piggyback as pb
    assert pb.write_image_store.__module__ == "mmimpress.store"
    original = pb.write_image_store
    pb.write_image_store = write_packed_store
    try:
        receipt = pb.persist_captured_visual_prefix(
            runner, captured_past_key_values, expanded_input_ids, image_size,
            per_sub_scores, out_dir, storage_policy=STORAGE_POLICY,
            layer_budget=dict(layer_budget), **kw)
    finally:
        pb.write_image_store = original
    meta = receipt["meta"]
    assert meta["store_format"] == STORE_FORMAT
    return receipt

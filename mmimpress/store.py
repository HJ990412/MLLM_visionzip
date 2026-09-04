"""Head-major chunked prefix-KV store for image prefixes.

One image == one prefix.  Its visual K/V is written per layer as

    layer_{l:02d}/k.bin        (v_num, num_heads, head_dim) fp16, C-contiguous
    layer_{l:02d}/v.bin        same
    layer_{l:02d}/probe_k.bin  (v_num, probe_heads, head_dim) fp16

TOKEN-MAJOR, because of how IMPRESS actually reads.  SparseVLM scores tokens
per head, but IMPRESS's probe-head consensus then applies ONE token set to
EVERY head, so the unit that gets fetched is "these tokens, all heads" -- which
token-major makes a single contiguous span per chunk.  A head-major layout was
tried first and measured worse: it turned 64 preads per request into ~3100,
and the syscall cost ate the byte savings.  Head-major only pays off when whole
heads can be skipped, which a uniform per-head budget never does.

Identification is the one access that wants a few heads and all tokens, so the
probe heads' keys are stored redundantly in their own file (IMPRESS 6.5), at
probe/(2*num_heads) = 4.7% extra space for 3 of 32 heads.  Without it the probe
phase would have to read every head's chunk.

The small system-prompt KV (tokens before the image block) is stored as one
torch file and always loaded in full.

Every read goes through ChunkReader, which uses os.pread (no mmap, no implicit
readahead beyond the range) and can drop the page cache per file descriptor, so
byte counts and wall-clock are attributable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from mmimpress.config import CHUNK_SIZE, PROBE_HEADS

META_NAME = "meta.json"
_NP_DTYPE = {"float16": np.float16, "float32": np.float32}
_TORCH_DTYPE = {"float16": torch.float16, "float32": torch.float32}


# --------------------------------------------------------------- addressing
def n_chunks(v_num: int, chunk_size: int = CHUNK_SIZE) -> int:
    return (v_num + chunk_size - 1) // chunk_size


def chunk_span(ci: int, v_num: int, chunk_size: int = CHUNK_SIZE):
    """Token range [start, end) of chunk `ci`."""
    s = ci * chunk_size
    return s, min(s + chunk_size, v_num)


def byte_range(tok_start: int, tok_end: int, width: int, head_dim: int,
               itemsize: int = 2):
    """Byte (offset, length) of token rows [tok_start, tok_end).

    `width` is the number of heads stored per token: num_heads for k/v.bin,
    probe_heads for the sidecar.
    """
    row = width * head_dim * itemsize
    return tok_start * row, (tok_end - tok_start) * row


def chunks_for_tokens(tokens, v_num: int, chunk_size: int = CHUNK_SIZE):
    """Sorted unique chunk ids covering the given STORED token positions."""
    return sorted({int(t) // chunk_size for t in tokens})


def merge_ranges(ranges, max_gap: int = 0):
    """Coalesce (offset, length) pairs that touch or overlap into fewer preads.

    max_gap > 0 also merges ranges separated by at most that many bytes, which
    trades a little read amplification for far fewer syscalls; the extra bytes
    are still counted, so measurements stay honest.
    """
    if not ranges:
        return []
    ranges = sorted(ranges)
    out = [list(ranges[0])]
    for off, ln in ranges[1:]:
        cur = out[-1]
        end = cur[0] + cur[1]
        if off <= end + max_gap:
            cur[1] = max(end, off + ln) - cur[0]
        else:
            out.append([off, ln])
    return [tuple(r) for r in out]


# ------------------------------------------------------------------- meta
def load_meta(store_dir: Path) -> dict:
    with open(Path(store_dir) / META_NAME) as f:
        return json.load(f)


# ------------------------------------------------------------------ writer
def write_image_store(out_dir: Path, layers, v_start: int, v_num: int,
                      prefix_input_ids, newline_idx, extra=None,
                      chunk_size: int = CHUNK_SIZE, dtype: str = "float16",
                      probe_heads: int = PROBE_HEADS):
    """Persist one image prefix (token-major + probe sidecar).

    layers: list over transformer layers of (k, v), each (1, H, prefix_len, hd)
            -- the HF DynamicCache layout.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    td = _TORCH_DTYPE[dtype]
    prefix_len = v_start + v_num
    H, hd = layers[0][0].shape[1], layers[0][0].shape[3]
    assert layers[0][0].shape[2] >= prefix_len, layers[0][0].shape

    sys_k = torch.stack([k[0, :, :v_start].to(td).cpu() for k, _ in layers])
    sys_v = torch.stack([v[0, :, :v_start].to(td).cpu() for _, v in layers])
    torch.save({"k": sys_k, "v": sys_v}, out_dir / "sys_kv.pt")

    total = probe_bytes = 0
    for li, (k, v) in enumerate(layers):
        ldir = out_dir / f"layer_{li:02d}"
        ldir.mkdir(exist_ok=True)
        for kind, t in (("k", k), ("v", v)):
            # (H, v_num, hd) -> (v_num, H, hd): token-major
            block = t[0, :, v_start:prefix_len].permute(1, 0, 2).to(td) \
                     .contiguous().cpu()
            assert block.shape == (v_num, H, hd), block.shape
            buf = block.numpy().tobytes()
            (ldir / f"{kind}.bin").write_bytes(buf)
            total += len(buf)
            if kind == "k":
                pb = block[:, :probe_heads].contiguous().numpy().tobytes()
                (ldir / "probe_k.bin").write_bytes(pb)
                probe_bytes += len(pb)

    meta = {
        "v_token_start": v_start,
        "v_token_num": v_num,
        "prefix_len": prefix_len,
        "num_layers": len(layers),
        "num_heads": H,
        "head_dim": hd,
        "probe_heads": probe_heads,
        "dtype": dtype,
        "chunk_size": chunk_size,
        "n_chunks_per_layer": n_chunks(v_num, chunk_size),
        "newline_idx": list(newline_idx),
        "n_spatial": v_num - len(newline_idx),
        "prefix_input_ids": list(prefix_input_ids),
        "layout": "token-major (v_token_num, num_heads, head_dim) fp16; chunk "
                  "ci = rows [ci*chunk_size, ...) = one contiguous span across "
                  "all heads. probe_k.bin mirrors the first probe_heads heads.",
        "bytes_visual_kv": total,
        "bytes_probe_sidecar": probe_bytes,
    }
    if extra:
        meta.update(extra)
    with open(out_dir / META_NAME, "w") as f:
        json.dump(meta, f, indent=1)
    return meta


# ------------------------------------------------------------------ reader
class IOCounter:
    """Physical read accounting, split by kind."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.preads = 0
        self.bytes = 0
        self.seconds = 0.0
        self.chunk_units = 0
        self.per_kind = {}

    def record(self, kind, nbytes, seconds, preads=1, units=0):
        self.preads += preads
        self.bytes += nbytes
        self.seconds += seconds
        self.chunk_units += units
        d = self.per_kind.setdefault(kind, {"bytes": 0, "seconds": 0.0,
                                            "preads": 0, "units": 0})
        d["bytes"] += nbytes
        d["seconds"] += seconds
        d["preads"] += preads
        d["units"] += units

    def summary(self):
        return {"preads": self.preads, "bytes": self.bytes,
                "mb": self.bytes / 1e6, "ms": self.seconds * 1e3,
                "chunk_units": self.chunk_units, "per_kind": self.per_kind}


class ChunkReader:
    """os.pread-based reader over one image's store directory."""

    def __init__(self, store_dir: Path, meta: dict, drop_cache: bool = True):
        self.dir = Path(store_dir)
        self.meta = meta
        self.drop_cache = drop_cache
        self.np_dtype = _NP_DTYPE[meta["dtype"]]
        self.itemsize = np.dtype(self.np_dtype).itemsize
        self._fds = {}

    def _fd(self, layer: int, name: str):
        key = (layer, name)
        if key not in self._fds:
            self._fds[key] = os.open(
                self.dir / f"layer_{layer:02d}" / f"{name}.bin", os.O_RDONLY)
        return self._fds[key]

    def close(self):
        for fd in self._fds.values():
            if self.drop_cache:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        self._fds.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def drop_all(self):
        """Evict this store's files from the page cache (cold-run setup)."""
        for p in sorted(self.dir.rglob("*.bin")):
            fd = os.open(p, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)

    # -------------------------------------------------------------- reads
    def _read(self, layer, name, ranges, counter, kind, units, max_gap=0):
        import time
        fd = self._fd(layer, name)
        merged = merge_ranges(ranges, max_gap=max_gap)
        t0 = time.perf_counter()
        blobs = [(off, os.pread(fd, ln, off)) for off, ln in merged]
        dt = time.perf_counter() - t0
        if counter is not None:
            counter.record(kind, sum(ln for _, ln in merged), dt,
                           preads=len(merged), units=units)
        return blobs

    def read_chunks(self, layer, kind, cids, counter=None, max_gap=0):
        """Chunks `cids` of one layer, all heads.

        Returns (rows LongTensor, vals (n_rows, num_heads, head_dim)).
        """
        m = self.meta
        v_num, H, hd = m["v_token_num"], m["num_heads"], m["head_dim"]
        cs = m["chunk_size"]
        spans = [chunk_span(ci, v_num, cs) for ci in sorted(set(cids))]
        ranges = [byte_range(s, e, H, hd, self.itemsize) for s, e in spans]
        blobs = self._read(layer, kind, ranges, counter, kind, len(spans),
                           max_gap)
        rows, vals = [], []
        row_bytes = H * hd * self.itemsize
        for off, buf in blobs:
            start = off // row_bytes
            n = len(buf) // row_bytes
            a = np.frombuffer(buf, dtype=self.np_dtype).reshape(n, H, hd)
            rows.append(torch.arange(start, start + n))
            vals.append(torch.from_numpy(a.copy()))
        return torch.cat(rows), torch.cat(vals, dim=0)

    def read_probe(self, layer, counter=None):
        """The probe-head key sidecar: (v_num, probe_heads, head_dim)."""
        m = self.meta
        v_num, P, hd = m["v_token_num"], m["probe_heads"], m["head_dim"]
        nbytes = v_num * P * hd * self.itemsize
        blobs = self._read(layer, "probe_k", [(0, nbytes)], counter, "probe",
                           m["n_chunks_per_layer"])
        a = np.frombuffer(blobs[0][1], dtype=self.np_dtype)
        return torch.from_numpy(a.copy()).view(v_num, P, hd)

    def read_full(self, layer, kind, counter=None):
        """Whole (v_num, num_heads, head_dim) block for one layer."""
        m = self.meta
        v_num, H, hd = m["v_token_num"], m["num_heads"], m["head_dim"]
        nbytes = v_num * H * hd * self.itemsize
        blobs = self._read(layer, kind, [(0, nbytes)], counter, kind,
                           m["n_chunks_per_layer"])
        a = np.frombuffer(blobs[0][1], dtype=self.np_dtype)
        return torch.from_numpy(a.copy()).view(v_num, H, hd)

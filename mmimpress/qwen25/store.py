"""Qwen2.5-VL native BF16 visual KV store.

One image has one global, stable image-only order. Each decoder layer has a
token-major K file and V file, so the first ``k`` physical chunks are one
contiguous ``pread`` span per file. Nonvisual prefix rows (both sides of the
image token run) live in a separate structural file and are never budgeted.

The writer consumes the Turn-1 cache in memory and writes its final physical
order directly. Activation checks every payload hash once. A cache hit reads
only its selected visual spans and the structural file; no payload is retained
by this object between calls.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mmimpress.cvpr25 import (budget_chunk_count, permutation_sha256,
                              visual_kv_budget_count)


FORMAT = "qwen25_bf16_visual_kv_v1"
DEFAULT_CHUNK_SIZE = 64
_ROW_ITEMSIZE = 2
_REQUIRED_EXTRA = (
    "image_sha256", "checkpoint_revision", "processor_settings",
    "image_grid_thw", "geometry", "position_policy", "logical_position_ids",
    "processor_revision", "code_revision", "environment_revision",
    "key_rope_state",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError(f"metadata value is not JSON compatible: {type(value)!r}")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _deep_size(value: Any, seen: set[int] | None = None) -> int:
    """Approximate resident Python metadata bytes, counting shared objects once."""
    if seen is None:
        seen = set()
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(_deep_size(k, seen) + _deep_size(v, seen)
                    for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        size += sum(_deep_size(v, seen) for v in value)
    return size


def _normalise_scores(scores: Any, n: int) -> list[float]:
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().flatten().cpu().tolist()
    values = [struct.unpack("<f", struct.pack("<f", float(x)))[0]
              for x in scores]
    if len(values) != n:
        raise ValueError(f"expected {n} visual scores, got {len(values)}")
    if not all(math.isfinite(x) for x in values):
        raise ValueError("every visual score must be finite")
    return values


def stable_visual_order(scores: Any, visual_count: int | None = None) -> list[int]:
    """Stored-to-original order; equal float32 scores keep original index."""
    if visual_count is None:
        visual_count = int(scores.numel()) if isinstance(scores, torch.Tensor) \
            else len(scores)
    values = _normalise_scores(scores, int(visual_count))
    return sorted(range(len(values)), key=lambda i: (-values[i], i))


def inverse_permutation(stored_to_original: Any) -> list[int]:
    order = [int(x) for x in stored_to_original]
    if sorted(order) != list(range(len(order))):
        raise ValueError("stored_to_original is not a permutation")
    inverse = [0] * len(order)
    for stored, original in enumerate(order):
        inverse[original] = stored
    return inverse


def _bf16_bytes(tensor: torch.Tensor) -> bytes:
    """Copy raw BF16 bits without a BF16 -> FP16 conversion or NumPy bridge."""
    if tensor.dtype != torch.bfloat16 or tensor.device.type != "cpu" \
            or not tensor.is_contiguous():
        raise ValueError("raw BF16 serialization requires contiguous CPU BF16")
    return ctypes.string_at(tensor.data_ptr(), tensor.numel() * _ROW_ITEMSIZE)


def _bf16_tensor(payload: bytes, shape: tuple[int, ...]) -> torch.Tensor:
    count = math.prod(shape)
    if len(payload) != count * _ROW_ITEMSIZE:
        raise ValueError("BF16 payload byte count does not match shape")
    out = torch.empty(shape, dtype=torch.bfloat16, device="cpu")
    if payload:
        ctypes.memmove(out.data_ptr(), payload, len(payload))
    return out


def _position_ids(value: Any, prefix_len: int) -> list[list[int]]:
    value = _jsonable(value)
    if len(value) != 3:
        raise ValueError("logical_position_ids must have three MRoPE axes")
    result = []
    for axis in value:
        if len(axis) == 1 and isinstance(axis[0], list):
            axis = axis[0]
        if len(axis) != prefix_len:
            raise ValueError("logical_position_ids length differs from prefix")
        result.append([int(x) for x in axis])
    return result


def _normalise_layers(layers: Any, prefix_len: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(layers, "key_cache") and hasattr(layers, "value_cache"):
        layers = zip(layers.key_cache, layers.value_cache)
    layers = list(layers)
    if not layers:
        raise ValueError("the KV cache has no decoder layers")
    first = layers[0][0]
    if first.ndim != 4 or first.shape[0] != 1:
        raise ValueError("native decoder KV must have shape [1, H_kv, seq, D]")
    heads, head_dim = int(first.shape[1]), int(first.shape[3])
    if heads < 1 or head_dim < 1:
        raise ValueError("invalid native KV shape")
    for li, pair in enumerate(layers):
        if len(pair) != 2:
            raise ValueError(f"layer {li} does not contain K and V")
        for kind, tensor in zip(("K", "V"), pair):
            if tensor.dtype != torch.bfloat16:
                raise ValueError(f"layer {li} {kind} is {tensor.dtype}, expected native BF16")
            if tensor.ndim != 4 or tensor.shape[0] != 1 \
                    or tensor.shape[1] != heads or tensor.shape[3] != head_dim \
                    or tensor.shape[2] < prefix_len:
                raise ValueError(f"layer {li} {kind} has inconsistent native KV shape")
    return layers


def _write_file(path: Path, payload: bytes, stats: dict) -> dict:
    started = time.perf_counter()
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        write_end = time.perf_counter()
        os.fsync(handle.fileno())
    ended = time.perf_counter()
    stats["write_seconds"] += write_end - started
    stats["fsync_seconds"] += ended - write_end
    return {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def write_qwen_store(out_dir: str | Path, layers: Any, visual_start: int,
                     visual_count: int, prefix_ids: Any, scores: Any = None,
                     extra: dict | None = None, *, chunk_size: int = DEFAULT_CHUNK_SIZE,
                     timing_out: dict | None = None) -> dict:
    """Write a canonical or saliency-repacked Qwen prefix directly from Turn-1 KV.

    ``layers`` contains native BF16 pairs in HF cache shape [1,H_kv,seq,D].
    ``scores=None`` writes canonical original visual order for FullLoad; a score
    vector writes the stable descending order for Ours. ``extra`` supplies the
    image/model/processor/geometry/position identity and MRoPE positions.
    The destination must not already exist, which protects previous artifacts.
    """
    started = time.perf_counter()
    visual_start, visual_count, chunk_size = int(visual_start), int(visual_count), int(chunk_size)
    prefix_ids = _jsonable(prefix_ids)
    if len(prefix_ids) == 1 and isinstance(prefix_ids[0], list):
        prefix_ids = prefix_ids[0]
    prefix_ids = [int(x) for x in prefix_ids]
    prefix_len = len(prefix_ids)
    if visual_start < 0 or visual_count < 1 or visual_start + visual_count > prefix_len:
        raise ValueError("invalid expanded image-token interval in prefix")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    extra = _jsonable(extra or {})
    missing = [key for key in _REQUIRED_EXTRA if key not in extra]
    if missing:
        raise ValueError(f"missing Qwen store metadata: {missing}")
    for key in ("image_sha256", "checkpoint_revision", "processor_revision",
                "code_revision", "environment_revision", "position_policy",
                "key_rope_state"):
        if not isinstance(extra[key], str) or not extra[key]:
            raise ValueError(f"{key} must be a nonempty string")
    if extra["key_rope_state"] != "post_mrope":
        raise ValueError("this store requires post-MRoPE cached keys")
    if len(extra["image_sha256"]) != 64 or any(
            ch not in "0123456789abcdefABCDEF" for ch in extra["image_sha256"]):
        raise ValueError("image_sha256 must be a 64-digit content digest")
    if not isinstance(extra["processor_settings"], dict) \
            or not isinstance(extra["geometry"], dict):
        raise ValueError("processor_settings and geometry must be objects")
    position_ids = _position_ids(extra["logical_position_ids"], prefix_len)
    layers = _normalise_layers(layers, prefix_len)
    num_layers = len(layers)
    heads, head_dim = int(layers[0][0].shape[1]), int(layers[0][0].shape[3])
    if scores is None:
        order = list(range(visual_count))
        score_values = None
    else:
        score_values = _normalise_scores(scores, visual_count)
        order = stable_visual_order(score_values, visual_count)
        if extra.get("score_source") is None or extra.get("score_layer") is None:
            raise ValueError("repacked store requires score_source and score_layer")
    inverse = inverse_permutation(order)
    permutation_end = time.perf_counter()

    n_chunks = (visual_count + chunk_size - 1) // chunk_size
    stored_rows = n_chunks * chunk_size
    padding_rows = stored_rows - visual_count
    structural_indices = [i for i in range(prefix_len)
                          if i < visual_start or i >= visual_start + visual_count]
    structural_count = len(structural_indices)
    row_bytes = heads * head_dim * _ROW_ITEMSIZE
    identity = {key: extra[key] for key in (
        "image_sha256", "checkpoint_revision", "processor_settings",
        "image_grid_thw", "geometry", "position_policy")}
    identity.update({"prefix_input_ids": prefix_ids, "kv_dtype": "bfloat16",
                     "processor_revision": extra["processor_revision"],
                     "code_revision": extra["code_revision"],
                     "environment_revision": extra["environment_revision"],
                     "key_rope_state": extra["key_rope_state"],
                     "logical_position_ids_sha256": hashlib.sha256(
                         _canonical_json(position_ids)).hexdigest()})
    identity_hash = hashlib.sha256(_canonical_json(identity)).hexdigest()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    stats = {"materialize_seconds": 0.0, "repack_seconds": 0.0,
             "write_seconds": 0.0, "fsync_seconds": 0.0}
    file_records = {}
    try:
        # All nonvisual rows, including vision_end after the image, are stored
        # in original logical order and never count against the visual budget.
        t0 = time.perf_counter()
        struct_per_kind = []
        for kind_idx in range(2):
            struct_per_layer = []
            for pair in layers:
                tensor = pair[kind_idx]
                indices = torch.tensor(structural_indices, dtype=torch.long,
                                       device=tensor.device)
                rows = tensor[0].index_select(1, indices).permute(1, 0, 2)
                struct_per_layer.append(rows.to("cpu").contiguous())
            struct_per_kind.append(torch.stack(struct_per_layer))
        structural = torch.stack(struct_per_kind).contiguous()
        stats["materialize_seconds"] += time.perf_counter() - t0
        structural_payload = _bf16_bytes(structural)
        file_records["structural_kv.bin"] = _write_file(
            out_dir / "structural_kv.bin", structural_payload, stats)

        order_cpu = torch.tensor(order, dtype=torch.long)
        for li, pair in enumerate(layers):
            layer_dir = out_dir / f"layer_{li:03d}"
            layer_dir.mkdir()
            for kind, tensor in zip(("k", "v"), pair):
                t0 = time.perf_counter()
                rows = tensor[0, :, visual_start:visual_start + visual_count, :]
                rows = rows.permute(1, 0, 2).to("cpu").contiguous()
                stats["materialize_seconds"] += time.perf_counter() - t0
                if score_values is not None:
                    t0 = time.perf_counter()
                    rows = rows.index_select(0, order_cpu)
                    stats["repack_seconds"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                if padding_rows:
                    padded = torch.zeros((stored_rows, heads, head_dim),
                                         dtype=torch.bfloat16)
                    padded[:visual_count] = rows
                    rows = padded
                rows = rows.contiguous()
                stats["materialize_seconds"] += time.perf_counter() - t0
                rel = f"layer_{li:03d}/{kind}.bin"
                file_records[rel] = _write_file(
                    layer_dir / f"{kind}.bin", _bf16_bytes(rows), stats)
            layer_fsync_start = time.perf_counter()
            layer_fd = os.open(layer_dir, os.O_RDONLY)
            try:
                os.fsync(layer_fd)
            finally:
                os.close(layer_fd)
            stats["fsync_seconds"] += time.perf_counter() - layer_fsync_start

        visual_bytes = sum(record["size"] for name, record in file_records.items()
                           if name != "structural_kv.bin")
        meta = {
            "format": FORMAT,
            "model_family": "Qwen2.5-VL",
            "layout": "token_major_repacked_bf16" if scores is not None else "token_major_canonical_bf16",
            "dtype": "bfloat16",
            "byte_order": sys.byteorder,
            "num_layers": num_layers,
            "num_kv_heads": heads,
            "head_dim": head_dim,
            "native_kv_shape": [1, heads, prefix_len, head_dim],
            "visual_start": visual_start,
            "visual_count": visual_count,
            "prefix_len": prefix_len,
            "prefix_input_ids": prefix_ids,
            "prefix_sha256": hashlib.sha256(_canonical_json(prefix_ids)).hexdigest(),
            "structural_indices": structural_indices,
            "structural_count": structural_count,
            "chunk_size": chunk_size,
            "n_chunks": n_chunks,
            "stored_rows": stored_rows,
            "valid_rows_last_chunk": visual_count - (n_chunks - 1) * chunk_size,
            "padding_rows": padding_rows,
            "row_bytes": row_bytes,
            "stored_to_original": order,
            "original_to_stored": inverse,
            "permutation_sha256": permutation_sha256(order),
            "global_order_all_layers": True,
            "score_source": extra.get("score_source") if score_values is not None else None,
            "score_layer": extra.get("score_layer") if score_values is not None else None,
            "saliency_sha256": None if score_values is None else hashlib.sha256(
                b"".join(struct.pack("<f", x) for x in score_values)).hexdigest(),
            "image_grid_thw": extra["image_grid_thw"],
            "geometry": extra["geometry"],
            "logical_position_ids": position_ids,
            "key_rope_state": extra.get("key_rope_state"),
            "rope_deltas": extra.get("rope_deltas"),
            "identity": identity,
            "identity_sha256": identity_hash,
            "code_revision": extra.get("code_revision"),
            "environment_revision": extra.get("environment_revision"),
            "processor_revision": extra.get("processor_revision"),
            "files": file_records,
            "bytes_visual_kv": visual_bytes,
            "bytes_structural_kv": len(structural_payload),
            "bytes_metadata_file": 0,
            "extra": {key: value for key, value in extra.items()
                      if key not in _REQUIRED_EXTRA},
        }
        # The metadata file records its own size; decimal digit growth settles
        # in at most a few iterations. It cannot contain its own SHA digest.
        for _ in range(10):
            encoded = (json.dumps(meta, indent=2, sort_keys=True,
                                  ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            if len(encoded) == meta["bytes_metadata_file"]:
                break
            meta["bytes_metadata_file"] = len(encoded)
        else:
            raise RuntimeError("metadata byte count did not converge")
        meta_start = time.perf_counter()
        with (out_dir / "meta.json").open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            meta_write_end = time.perf_counter()
            os.fsync(handle.fileno())
        meta_end = time.perf_counter()
        stats["write_seconds"] += meta_write_end - meta_start
        stats["fsync_seconds"] += meta_end - meta_write_end
        dir_fsync_start = time.perf_counter()
        dir_fd = os.open(out_dir, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        stats["fsync_seconds"] += time.perf_counter() - dir_fsync_start
    except Exception:
        # A failed build has no meta.json completion marker. Preserve the
        # partial directory for diagnosis; later builds cannot overwrite it.
        raise
    if timing_out is not None:
        timing_out.update({
            "permutation_ms": (permutation_end - started) * 1e3,
            "kv_materialize_ms": stats["materialize_seconds"] * 1e3,
            "kv_repack_ms": stats["repack_seconds"] * 1e3,
            "ssd_write_ms": stats["write_seconds"] * 1e3,
            "fsync_ms": stats["fsync_seconds"] * 1e3,
            "total_persistence_ms": (time.perf_counter() - started) * 1e3,
        })
    return meta


class IOCounter:
    """Actual bytes returned and OS pread calls, by payload class."""

    def __init__(self):
        self.bytes = 0
        self.preads = 0
        self.spans = 0
        self.seconds = 0.0
        self.per_kind: dict[str, dict[str, float | int]] = {}
        self.span_details: list[dict[str, Any]] = []

    def span(self, kind: str, source: str, offset: int, requested: int) -> None:
        self.spans += 1
        self.span_details.append({"kind": kind, "source": source,
                                  "offset": offset, "requested_bytes": requested})
        entry = self.per_kind.setdefault(kind, {"bytes": 0, "preads": 0,
                                                "spans": 0, "seconds": 0.0})
        entry["spans"] += 1

    def call(self, kind: str, actual_bytes: int, seconds: float) -> None:
        self.bytes += actual_bytes
        self.preads += 1
        self.seconds += seconds
        entry = self.per_kind.setdefault(kind, {"bytes": 0, "preads": 0,
                                                "spans": 0, "seconds": 0.0})
        entry["bytes"] += actual_bytes
        entry["preads"] += 1
        entry["seconds"] += seconds

    def summary(self) -> dict:
        return {"bytes": self.bytes, "preads": self.preads,
                "spans": self.spans, "ms": self.seconds * 1e3,
                "span_details": list(self.span_details),
                "per_kind": {key: dict(value)
                             for key, value in self.per_kind.items()}}


def _pread_exact(fd: int, count: int, offset: int, io: IOCounter,
                 kind: str, source: str) -> bytes:
    if count == 0:
        return b""
    io.span(kind, source, offset, count)
    chunks = []
    remaining = count
    cursor = offset
    while remaining:
        started = time.perf_counter()
        chunk = os.pread(fd, remaining, cursor)
        io.call(kind, len(chunk), time.perf_counter() - started)
        if not chunk:
            raise EOFError(f"short pread at offset {cursor}: {remaining} bytes missing")
        chunks.append(chunk)
        cursor += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


@dataclass
class LoadedKV:
    layers: list[tuple[torch.Tensor, torch.Tensor]]
    logical_indices: tuple[int, ...]
    position_ids: torch.Tensor
    selected_visual_original: tuple[int, ...]
    selected_chunks: int
    planning_ms: float
    kept_visual_tokens: int
    kept_visual_ratio: float
    padding_rows_read: int
    visual_payload_read_ratio: float
    total_payload_read_ratio: float
    io: IOCounter
    budget_unit: str
    selected_visual_stored: tuple[int, ...]
    loaded_valid_visual_rows: int
    extra_valid_visual_rows: int
    structural_rows: int
    h2d_kv_bytes: int


def plan_prefix_budget(meta: dict, budget: float = 0.25,
                       *, budget_unit: str = "chunk") -> dict:
    """Return read geometry and attended rows without touching any payload.

    The legacy rounded chunk path is unchanged. Visual-KV mode uses the
    post-merger content count and reads whole 64-row chunks while retaining
    exactly the first ``ceil(budget * N)`` real rows.
    """
    n = int(meta["visual_count"])
    cs = int(meta["chunk_size"])
    if budget_unit == "chunk":
        chunks = budget_chunk_count(int(meta["n_chunks"]), float(budget))
        kept = min(n, chunks * cs)
    elif budget_unit == "visual_kv":
        if n < 1:
            raise ValueError("visual-KV serving requires nonempty image content")
        if cs != DEFAULT_CHUNK_SIZE:
            raise ValueError("visual-KV serving requires 64-row chunks")
        if meta.get("layout") != "token_major_repacked_bf16" \
                or meta.get("global_order_all_layers") is not True:
            raise ValueError("visual-KV serving requires globally repacked BF16 layout")
        if meta.get("score_source") != "last_fullatt_ViT_received_attention" \
                or not isinstance(meta.get("score_layer"), int) \
                or not isinstance(meta.get("saliency_sha256"), str):
            raise ValueError("visual-KV serving requires frozen image-only saliency provenance")
        kept = visual_kv_budget_count(n, budget)
        if kept == 0:
            raise ValueError("visual-KV serving requires a positive retained budget")
        chunks = (kept + cs - 1) // cs
    else:
        raise ValueError(f"unknown budget unit: {budget_unit}")
    disk_rows = chunks * cs
    valid_rows = min(n, disk_rows)
    if not 0 <= kept <= valid_rows or not 0 <= chunks <= meta["n_chunks"]:
        raise AssertionError("budget plan exceeds valid store geometry")
    return {"budget_unit": budget_unit, "selected_chunks": chunks,
            "kept_visual_tokens": kept, "disk_rows": disk_rows,
            "loaded_valid_visual_rows": valid_rows,
            "extra_valid_visual_rows": valid_rows - kept,
            "padding_rows_read": disk_rows - valid_rows}


class QwenStore:
    """Activated image store; keeps metadata and FDs, never decoded KV rows."""

    def __init__(self, path: str | Path, expected_identity: dict | None = None,
                 verify: bool = True):
        self.path = Path(path)
        self.activation_io = IOCounter()
        self.activation_started = time.perf_counter()
        self._fds: dict[str, int] = {}
        try:
            fd = os.open(self.path / "meta.json", os.O_RDONLY)
            try:
                size = os.fstat(fd).st_size
                raw = _pread_exact(fd, size, 0, self.activation_io, "metadata", "meta.json")
            finally:
                os.close(fd)
            self.metadata_file_bytes = len(raw)
            self.metadata_file_sha256 = hashlib.sha256(raw).hexdigest()
            self.meta = json.loads(raw)
            self.metadata_resident_bytes = _deep_size(self.meta)
            self._validate_meta(expected_identity)
            if verify:
                self._verify_payload_hashes()
            self.activation_ms = (time.perf_counter() - self.activation_started) * 1e3
        except Exception:
            self.close()
            raise

    def _expected_paths(self) -> set[str]:
        return {"structural_kv.bin"} | {
            f"layer_{li:03d}/{kind}.bin"
            for li in range(self.meta["num_layers"]) for kind in ("k", "v")}

    def _validate_meta(self, expected_identity: dict | None) -> None:
        m = self.meta
        if m.get("format") != FORMAT or m.get("dtype") != "bfloat16" \
                or m.get("byte_order") != sys.byteorder \
                or m.get("key_rope_state") != "post_mrope":
            raise ValueError("unsupported or mislabeled Qwen KV store")
        n = int(m["visual_count"])
        p = int(m["prefix_len"])
        start = int(m["visual_start"])
        cs = int(m["chunk_size"])
        heads = int(m["num_kv_heads"])
        hd = int(m["head_dim"])
        if min(n, p, cs, heads, hd, int(m["num_layers"])) < 1 \
                or start < 0 or start + n > p:
            raise ValueError("invalid store dimensions")
        if len(m["prefix_input_ids"]) != p or len(m["stored_to_original"]) != n:
            raise ValueError("invalid prefix IDs or permutation length")
        if inverse_permutation(m["stored_to_original"]) != m["original_to_stored"]:
            raise ValueError("permutation inverse mismatch")
        if permutation_sha256(m["stored_to_original"]) != m["permutation_sha256"]:
            raise ValueError("permutation digest mismatch")
        if hashlib.sha256(_canonical_json(m["prefix_input_ids"])).hexdigest() != m["prefix_sha256"]:
            raise ValueError("prefix token digest mismatch")
        expected_struct = [i for i in range(p) if i < start or i >= start + n]
        if m["structural_indices"] != expected_struct \
                or m["structural_count"] != len(expected_struct):
            raise ValueError("structural prefix index mismatch")
        nc = (n + cs - 1) // cs
        if m["n_chunks"] != nc or m["stored_rows"] != nc * cs \
                or m["padding_rows"] != nc * cs - n \
                or m["valid_rows_last_chunk"] != n - (nc - 1) * cs:
            raise ValueError("chunk geometry mismatch")
        if m["row_bytes"] != heads * hd * _ROW_ITEMSIZE:
            raise ValueError("row byte width mismatch")
        if m["native_kv_shape"] != [1, heads, p, hd] \
                or m["global_order_all_layers"] is not True:
            raise ValueError("native KV shape or global order mismatch")
        if m["layout"] not in ("token_major_repacked_bf16",
                               "token_major_canonical_bf16"):
            raise ValueError("unknown physical layout")
        if m["layout"] == "token_major_canonical_bf16" and (
                m["stored_to_original"] != list(range(n))
                or m["saliency_sha256"] is not None):
            raise ValueError("canonical layout contradicts permutation or score")
        if m["layout"] == "token_major_repacked_bf16" and (
                m["saliency_sha256"] is None or m["score_source"] is None
                or m["score_layer"] is None):
            raise ValueError("repacked layout lacks saliency provenance")
        if _position_ids(m["logical_position_ids"], p) != m["logical_position_ids"]:
            raise ValueError("logical MRoPE position shape mismatch")
        identity = m["identity"]
        if identity.get("prefix_input_ids") != m["prefix_input_ids"] \
                or identity.get("kv_dtype") != "bfloat16" \
                or identity.get("image_grid_thw") != m["image_grid_thw"] \
                or identity.get("geometry") != m["geometry"] \
                or identity.get("key_rope_state") != m["key_rope_state"] \
                or identity.get("code_revision") != m["code_revision"] \
                or identity.get("environment_revision") != m["environment_revision"] \
                or identity.get("logical_position_ids_sha256") != hashlib.sha256(
                    _canonical_json(m["logical_position_ids"])).hexdigest():
            raise ValueError("cache identity contradicts store metadata")
        if hashlib.sha256(_canonical_json(identity)).hexdigest() != m["identity_sha256"]:
            raise ValueError("cache identity digest mismatch")
        if expected_identity is not None:
            for key, expected in _jsonable(expected_identity).items():
                if identity.get(key) != expected:
                    raise ValueError(f"cache identity mismatch: {key}")
        if set(m["files"]) != self._expected_paths():
            raise ValueError("payload file list mismatch")
        visual_expected = 2 * m["num_layers"] * m["stored_rows"] * m["row_bytes"]
        struct_expected = 2 * m["num_layers"] * m["structural_count"] * m["row_bytes"]
        if m["bytes_visual_kv"] != visual_expected \
                or m["bytes_structural_kv"] != struct_expected \
                or m["bytes_metadata_file"] != self.metadata_file_bytes:
            raise ValueError("payload or metadata byte count mismatch")
        for rel, record in m["files"].items():
            expected_size = struct_expected if rel == "structural_kv.bin" \
                else m["stored_rows"] * m["row_bytes"]
            if record["size"] != expected_size:
                raise ValueError(f"payload size mismatch in metadata: {rel}")
            if (self.path / rel).stat().st_size != expected_size:
                raise ValueError(f"payload file size mismatch: {rel}")

    def _fd(self, rel: str) -> int:
        if rel not in self._fds:
            self._fds[rel] = os.open(self.path / rel, os.O_RDONLY)
        return self._fds[rel]

    def _verify_payload_hashes(self) -> None:
        for rel, record in self.meta["files"].items():
            fd = self._fd(rel)
            digest = hashlib.sha256()
            cursor = 0
            while cursor < record["size"]:
                count = min(4 * 1024 * 1024, record["size"] - cursor)
                payload = _pread_exact(fd, count, cursor, self.activation_io,
                                       "activation_visual" if rel != "structural_kv.bin"
                                       else "activation_structural", rel)
                digest.update(payload)
                cursor += len(payload)
            if digest.hexdigest() != record["sha256"]:
                raise ValueError(f"payload hash mismatch: {rel}")

    def load_prefix(self, budget: float = 0.25,
                    *, budget_unit: str = "chunk") -> LoadedKV:
        """Read whole prefix chunks; compact only the budgeted content rows."""
        planning_start = time.perf_counter()
        m = self.meta
        plan = plan_prefix_budget(m, budget, budget_unit=budget_unit)
        chunks = plan["selected_chunks"]
        selected_rows = plan["kept_visual_tokens"]
        disk_rows = plan["disk_rows"]
        selected = m["stored_to_original"][:selected_rows]
        combined_positions = m["structural_indices"] + [m["visual_start"] + i
                                                         for i in selected]
        sort_order = sorted(range(len(combined_positions)),
                            key=combined_positions.__getitem__)
        logical_indices = tuple(combined_positions[i] for i in sort_order)
        sort_tensor = torch.tensor(sort_order, dtype=torch.long)
        io = IOCounter()
        structural_size = m["bytes_structural_kv"]
        planning_ms = (time.perf_counter() - planning_start) * 1e3
        structural_payload = _pread_exact(self._fd("structural_kv.bin"),
                                          structural_size, 0, io, "structural",
                                          "structural_kv.bin")
        structural = _bf16_tensor(structural_payload,
                                  (2, m["num_layers"], m["structural_count"],
                                   m["num_kv_heads"], m["head_dim"]))
        layers = []
        for li in range(m["num_layers"]):
            pair = []
            for kind_idx, kind in enumerate(("k", "v")):
                rel = f"layer_{li:03d}/{kind}.bin"
                payload = _pread_exact(self._fd(rel), disk_rows * m["row_bytes"],
                                       0, io, "visual", rel)
                visual = _bf16_tensor(payload,
                                      (disk_rows, m["num_kv_heads"], m["head_dim"]))
                rows = torch.cat((structural[kind_idx, li],
                                  visual[:selected_rows]), dim=0)
                rows = rows.index_select(0, sort_tensor)
                pair.append(rows.permute(1, 0, 2).unsqueeze(0).contiguous())
            layers.append((pair[0], pair[1]))
        positions = torch.tensor([[axis[i] for i in logical_indices]
                                  for axis in m["logical_position_ids"]],
                                 dtype=torch.long).unsqueeze(1)
        return LoadedKV(
            layers=layers,
            logical_indices=logical_indices,
            position_ids=positions,
            selected_visual_original=tuple(sorted(selected)),
            selected_chunks=chunks,
            planning_ms=planning_ms,
            kept_visual_tokens=selected_rows,
            kept_visual_ratio=selected_rows / m["visual_count"],
            padding_rows_read=plan["padding_rows_read"],
            visual_payload_read_ratio=(disk_rows / m["stored_rows"]),
            total_payload_read_ratio=(io.bytes / (m["bytes_visual_kv"]
                                                  + m["bytes_structural_kv"])),
            io=io,
            budget_unit=budget_unit,
            selected_visual_stored=tuple(range(selected_rows)),
            loaded_valid_visual_rows=plan["loaded_valid_visual_rows"],
            extra_valid_visual_rows=plan["extra_valid_visual_rows"],
            structural_rows=m["structural_count"],
            h2d_kv_bytes=(selected_rows + m["structural_count"])
                         * m["row_bytes"] * 2 * m["num_layers"],
        )

    def drop_payload_cache(self) -> dict:
        """Request cold-page conditioning; caller times this outside TTFT."""
        attempted = 0
        failed = []
        if not callable(getattr(os, "posix_fadvise", None)) \
                or not hasattr(os, "POSIX_FADV_DONTNEED"):
            paths = list(self.meta["files"])
            return {"attempted": len(paths),
                    "failed": [{"path": rel, "error": "posix_fadvise unavailable"}
                               for rel in paths]}
        for rel in self.meta["files"]:
            attempted += 1
            try:
                os.posix_fadvise(self._fd(rel), 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError as exc:
                failed.append({"path": rel, "error": str(exc)})
        return {"attempted": attempted, "failed": failed}

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def __enter__(self) -> "QwenStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def open_qwen_store(path: str | Path, expected_identity: dict | None = None,
                    verify: bool = True) -> QwenStore:
    """Activate a store. Hash checks and metadata reads are activation costs."""
    return QwenStore(path, expected_identity=expected_identity, verify=verify)

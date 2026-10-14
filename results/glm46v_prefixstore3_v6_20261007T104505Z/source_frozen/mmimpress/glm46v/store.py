"""Native BF16 GLM-4.6V-Flash PrefixStore-KV25 [B].

The store receives a capture from the normal source T1 forward. Both arms copy
the complete prefix to CPU. B additionally repacks *all* original content rows
before serialization writes only ceil(original N / 4). Content files contain
token-major [row, native KV head, head dimension] bits with 64-row logical
chunks and an unpadded final chunk. FullLoad never computes or stores scores.

Activation requires the expected prefix identity, a durable completion seal,
and full metadata/payload hash verification. An activated reader keeps only
metadata and file descriptors; every load performs actual OS pread calls.
Keys are post-RoPE bits: this module never rotates keys or derives positions
from compact cache length. Native suffix/causal semantics belong to the runner.
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


MODEL_ID = "zai-org/GLM-4.6V-Flash"
FORMAT = "glm46v_native_bf16_prefixstore_v1"
SEAL_FORMAT = "glm46v_prefixstore_completion_v1"
DEFAULT_CHUNK_SIZE = 64
_ITEM_BYTES = 2
_REQUIRED_EXTRA = (
    "model_id", "image_sha256", "checkpoint_revision", "processor_revision",
    "processor_settings", "geometry", "position_policy", "logical_position_ids",
    "key_rope_state", "rope_config", "rope_offset", "code_revision",
    "environment_revision",
)


@dataclass(frozen=True)
class NativeKVSpec:
    num_layers: int
    num_kv_heads: int
    head_dim: int

    def __post_init__(self):
        for field in (self.num_layers, self.num_kv_heads, self.head_dim):
            if type(field) is not int or field < 1:
                raise ValueError("native KV geometry must contain positive integers")

    @property
    def bytes_per_token(self) -> int:
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * _ITEM_BYTES


# Official checkpoint expectation; the writer validates actual tensors and
# enforces this only when the caller supplies it as shape_spec.
EXPECTED_NATIVE_KV_SPEC = NativeKVSpec(40, 2, 128)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("metadata object keys must be strings")
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError(f"metadata is not JSON compatible: {type(value)!r}")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _ints(value: Any, name: str, *, nonnegative: bool = True) -> list[int]:
    value = _jsonable(value)
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError(f"{name} must contain integers")
    if nonnegative and any(item < 0 for item in value):
        raise ValueError(f"{name} must contain nonnegative integers")
    return value


def _prefix_ids(value: Any) -> list[int]:
    value = _jsonable(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    result = _ints(value, "prefix input IDs")
    if not result:
        raise ValueError("the native prefix is empty")
    return result


def _positions(value: Any, prefix_length: int) -> list[list[int]]:
    value = _jsonable(value)
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("native position IDs require three axes")
    result = []
    for axis in value:
        if isinstance(axis, list) and len(axis) == 1 and isinstance(axis[0], list):
            axis = axis[0]
        axis = _ints(axis, "native position axis")
        if len(axis) != prefix_length:
            raise ValueError("native position length differs from original prefix")
        result.append(axis)
    return result


def _content(value: Any, prefix_length: int) -> list[int]:
    content = _ints(value, "content logical IDs")
    if not content or content != sorted(set(content)) or content[-1] >= prefix_length:
        raise ValueError("content logical IDs must be nonempty, unique, sorted and in range")
    return content


def _validate_extra(extra: Any, prefix_length: int) -> dict:
    extra = _jsonable(extra)
    if not isinstance(extra, dict):
        raise ValueError("native prefix identity metadata is required")
    missing = [key for key in _REQUIRED_EXTRA if key not in extra]
    if missing:
        raise ValueError(f"missing native prefix identity metadata: {missing}")
    if extra["model_id"] != MODEL_ID or not _sha(extra["image_sha256"]):
        raise ValueError("invalid GLM model/image identity")
    for key in ("checkpoint_revision", "processor_revision", "position_policy",
                "code_revision", "environment_revision"):
        if not isinstance(extra[key], str) or not extra[key]:
            raise ValueError(f"{key} must be a nonempty string")
    if extra["key_rope_state"] != "post_rope":
        raise ValueError("native GLM store requires post-RoPE keys")
    for key in ("processor_settings", "geometry", "rope_config"):
        if not isinstance(extra[key], dict) or not extra[key]:
            raise ValueError(f"{key} must be a nonempty native configuration object")
    def valid_offset(value):
        return type(value) is int or (isinstance(value, list) and bool(value)
                                     and all(valid_offset(item) for item in value))
    if not valid_offset(extra["rope_offset"]):
        raise ValueError("native RoPE offset must contain integers")
    extra["logical_position_ids"] = _positions(extra["logical_position_ids"], prefix_length)
    _canonical(extra)  # Reject NaN/infinite values in all other identity fields.
    return extra


def _scores(scores: Any, count: int) -> list[float]:
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().flatten().cpu().tolist()
    if not isinstance(scores, (list, tuple)) or len(scores) != count:
        raise ValueError(f"expected {count} image-only scores")
    values = []
    for item in scores:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("image-only scores must be numeric")
        try:
            item = struct.unpack("<f", struct.pack("<f", float(item)))[0]
        except (OverflowError, struct.error) as exc:
            raise ValueError("image-only scores must be finite float32") from exc
        if not math.isfinite(item):
            raise ValueError("image-only scores must be finite float32")
        values.append(item)
    return values


def stable_visual_order(scores: Any, visual_count: int | None = None) -> list[int]:
    """One descending float32 image-only order, ties resolved by original ID."""
    if visual_count is None:
        visual_count = scores.numel() if isinstance(scores, torch.Tensor) else len(scores)
    values = _scores(scores, visual_count)
    return sorted(range(visual_count), key=lambda index: (-values[index], index))


def inverse_permutation(order: Any) -> list[int]:
    order = _ints(order, "importance permutation")
    if sorted(order) != list(range(len(order))):
        raise ValueError("importance order is not a permutation")
    inverse = [0] * len(order)
    for rank, original in enumerate(order):
        inverse[original] = rank
    return inverse


def _native_layers(layers: Any, length: int, spec: NativeKVSpec | None):
    if hasattr(layers, "key_cache") and hasattr(layers, "value_cache"):
        keys, values = layers.key_cache, layers.value_cache
        if len(keys) != len(values):
            raise ValueError("native cache K/V layer counts differ")
        layers = list(zip(keys, values))
    else:
        layers = list(layers)
    if not layers or any(not isinstance(pair, (tuple, list)) or len(pair) != 2
                         for pair in layers):
        raise ValueError("native cache requires decoder K/V pairs")
    first = layers[0][0]
    if not isinstance(first, torch.Tensor) or first.ndim != 4 or first.shape[0] != 1:
        raise ValueError("native KV shape must be [1, KV heads, sequence, head dimension]")
    actual = NativeKVSpec(len(layers), int(first.shape[1]), int(first.shape[3]))
    if spec is not None and actual != spec:
        raise ValueError(f"actual native KV geometry {actual} differs from required {spec}")
    for pair in layers:
        for tensor in pair:
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.bfloat16:
                raise ValueError("native KV must be BF16 without conversion/head expansion")
            if (tensor.ndim != 4 or tensor.shape[0] != 1
                    or tensor.shape[1] != actual.num_kv_heads
                    or tensor.shape[3] != actual.head_dim or tensor.shape[2] < length):
                raise ValueError("inconsistent native KV shape or missing prefix rows")
    return layers, actual


def _raw(tensor: torch.Tensor) -> bytes:
    if tensor.dtype != torch.bfloat16 or tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("serialization requires contiguous CPU native BF16 bits")
    return ctypes.string_at(tensor.data_ptr(), tensor.numel() * _ITEM_BYTES)


def _tensor(payload: bytes, shape: tuple[int, ...]) -> torch.Tensor:
    if len(payload) != math.prod(shape) * _ITEM_BYTES:
        raise ValueError("BF16 byte count differs from declared shape")
    result = torch.empty(shape, dtype=torch.bfloat16, device="cpu")
    if payload:
        ctypes.memmove(result.data_ptr(), payload, len(payload))
    return result


def _identity(extra: dict, prefix: list[int], content: list[int], spec: NativeKVSpec) -> dict:
    identity = {key: value for key, value in extra.items()
                if key not in ("score_source", "score_layer", "logical_position_ids")}
    identity.update({"prefix_input_ids": prefix, "content_logical_indices": content,
                     "logical_position_ids_sha256": _digest(extra["logical_position_ids"]),
                     "kv_dtype": "bfloat16", "num_layers": spec.num_layers,
                     "num_kv_heads": spec.num_kv_heads, "head_dim": spec.head_dim})
    return identity


def make_store_identity(extra: dict, prefix_ids: Any, content_indices: Any,
                        shape_spec: NativeKVSpec) -> dict:
    """Build expected identity from a frozen native capture contract, not disk."""
    prefix = _prefix_ids(prefix_ids)
    return _identity(_validate_extra(extra, len(prefix)), prefix,
                     _content(content_indices, len(prefix)), shape_spec)


def _fsync_dir(path: Path, stats: dict) -> None:
    started = time.perf_counter()
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    stats["fsync_seconds"] += time.perf_counter() - started


def _write(path: Path, payload: bytes, stats: dict, kind: str) -> dict:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        started = time.perf_counter()
        view = memoryview(payload)
        while offset < len(payload):
            count = os.write(fd, view[offset:])
            stats["os_write_calls"] += 1
            stats["os_write_bytes"] += count
            stats["write_ranges"].append({"kind": kind, "source": str(path),
                                          "offset": offset, "actual_bytes": count})
            if count <= 0:
                raise OSError("zero-length store write")
            offset += count
        stats["write_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        os.fsync(fd)
        stats["fsync_seconds"] += time.perf_counter() - started
    finally:
        os.close(fd)
    return {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def write_glm_store(out_dir: str | Path, layers: Any, content_indices: Any,
                    prefix_ids: Any, scores: Any = None, extra: dict | None = None,
                    *, storage_policy: str = "full", chunk_size: int = DEFAULT_CHUNK_SIZE,
                    shape_spec: NativeKVSpec | None = None,
                    timing_out: dict | None = None) -> dict:
    """Persist complete native T1 prefix capture; never overwrite artifacts.

    ``content_indices`` are original logical rows in native merged-token order.
    Their complement comprises all system, structural and boundary prefix rows.
    ``prefix25`` needs question-independent vision scores and always writes
    ceil(N/4), without recomputing a budget from chunks or selected rows.
    A failed build retains its directory without a valid completion seal.
    """
    started = time.perf_counter()
    if storage_policy not in ("full", "prefix25"):
        raise ValueError("unsupported GLM storage policy")
    if type(chunk_size) is not int or chunk_size != DEFAULT_CHUNK_SIZE:
        raise ValueError("GLM PrefixStore requires 64-row chunks")
    if (storage_policy == "full" and scores is not None
            or storage_policy == "prefix25" and scores is None):
        raise ValueError("FullLoad is canonical; prefix25 requires image-only scores")
    prefix = _prefix_ids(prefix_ids)
    content = _content(content_indices, len(prefix))
    extra = _validate_extra(extra, len(prefix))
    if storage_policy == "full":
        extra = {key: value for key, value in extra.items()
                 if key not in ("score_source", "score_layer")}
    layers, spec = _native_layers(layers, len(prefix), shape_spec)
    n = len(content)
    values = None if scores is None else _scores(scores, n)
    order = list(range(n)) if values is None else stable_visual_order(values, n)
    if values is not None and (not isinstance(extra.get("score_source"), str)
                              or not extra["score_source"]
                              or type(extra.get("score_layer")) is not int
                              or extra["score_layer"] < 0):
        raise ValueError("prefix25 requires frozen image-only score provenance")
    k = n if storage_policy == "full" else (n + 3) // 4
    content_set = set(content)
    structural = [index for index in range(len(prefix)) if index not in content_set]
    identity = _identity(extra, prefix, content, spec)
    stats = {"materialize_seconds": 0.0, "repack_seconds": 0.0,
             "serialize_seconds": 0.0, "write_seconds": 0.0, "fsync_seconds": 0.0,
             "os_write_bytes": 0, "os_write_calls": 0, "write_ranges": []}
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=False)
    try:
        t0 = time.perf_counter()
        # Copy the full prefix for every layer/kind before selecting any row.
        # clone() also isolates a CPU fixture/capture from later caller mutation.
        cpu = [(key[..., :len(prefix), :].detach().to("cpu").clone().contiguous(),
                value[..., :len(prefix), :].detach().to("cpu").clone().contiguous())
               for key, value in layers]
        stats["materialize_seconds"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        original = torch.tensor(content, dtype=torch.long)
        rank = torch.tensor(order, dtype=torch.long)
        structural_index = torch.tensor(structural, dtype=torch.long)
        # Repack all N content rows in every native KV head and decoder layer.
        # This intentionally keeps full CPU materialization/repacking in [B].
        repacked = [(key[0].index_select(1, original).permute(1, 0, 2)
                     .index_select(0, rank).contiguous(),
                     value[0].index_select(1, original).permute(1, 0, 2)
                     .index_select(0, rank).contiguous()) for key, value in cpu]
        structural_tensor = torch.stack([torch.stack([pair[kind][0].index_select(1, structural_index)
                                                      .permute(1, 0, 2).contiguous()
                                                      for pair in cpu]) for kind in range(2)]).contiguous()
        stats["repack_seconds"] = time.perf_counter() - t0
        files = {}
        t0 = time.perf_counter()
        struct_payload = _raw(structural_tensor)
        stats["serialize_seconds"] += time.perf_counter() - t0
        files["structural_kv.bin"] = _write(path / "structural_kv.bin", struct_payload,
                                           stats, "structural")
        for li, pair in enumerate(repacked):
            layer_path = path / f"layer_{li:03d}"
            layer_path.mkdir()
            for kind, rows in zip(("k", "v"), pair):
                t0 = time.perf_counter()
                payload = _raw(rows[:k].contiguous())
                stats["serialize_seconds"] += time.perf_counter() - t0
                rel = f"layer_{li:03d}/{kind}.bin"
                files[rel] = _write(path / rel, payload, stats, "content")
            _fsync_dir(layer_path, stats)
        row_bytes = spec.num_kv_heads * spec.head_dim * _ITEM_BYTES
        chunks = (k + chunk_size - 1) // chunk_size
        meta = {
            "format": FORMAT, "model_family": "GLM-4.6V-Flash",
            "dtype": "bfloat16", "byte_order": sys.byteorder,
            "storage_policy": storage_policy, "chunk_size": chunk_size,
            "num_layers": spec.num_layers, "num_kv_heads": spec.num_kv_heads,
            "head_dim": spec.head_dim, "native_kv_shape": [1, spec.num_kv_heads,
                                                            len(prefix), spec.head_dim],
            "bytes_per_native_token": spec.bytes_per_token,
            "layout": "token_major_canonical_bf16" if values is None else "token_major_repacked_bf16",
            "structural_layout": "kv_kind_layer_token_head_dimension",
            "prefix_len": len(prefix), "prefix_input_ids": prefix,
            "prefix_sha256": _digest(prefix), "content_logical_indices": content,
            "structural_indices": structural, "structural_count": len(structural),
            "original_content_count": n, "stored_content_count": k,
            "stored_rows": k, "n_chunks": chunks, "padding_rows": 0,
            "valid_rows_last_chunk": k - (chunks - 1) * chunk_size,
            "row_bytes": row_bytes, "full_importance_permutation": order,
            "original_to_stored": inverse_permutation(order),
            "stored_row_to_original": order[:k],
            "stored_to_original": order[:k], "permutation_sha256": _digest(order),
            "global_order_all_layers": True,
            "retention_ratio": 1.0 if values is None else 0.25,
            "rounding_policy": "all_original_content" if values is None else "ceil_original_content",
            "score_source": None if values is None else extra["score_source"],
            "score_layer": None if values is None else extra["score_layer"],
            "saliency_sha256": None if values is None else hashlib.sha256(
                b"".join(struct.pack("<f", value) for value in values)).hexdigest(),
            "logical_position_ids": extra["logical_position_ids"],
            "key_rope_state": extra["key_rope_state"],
            "rope_config": extra["rope_config"], "rope_offset": extra["rope_offset"],
            "geometry": extra["geometry"], "identity": identity,
            "identity_sha256": _digest(identity), "extra": extra, "files": files,
            "bytes_content_kv": k * spec.bytes_per_token,
            "bytes_structural_kv": len(struct_payload), "bytes_metadata_file": 0,
            "bytes_seal_file": 0, "store_logical_bytes": 0,
        }
        payload_bytes = meta["bytes_content_kv"] + meta["bytes_structural_kv"]
        for _ in range(20):
            encoded = _canonical(meta) + b"\n"
            seal = {"format": SEAL_FORMAT, "completed": True,
                    "metadata_sha256": hashlib.sha256(encoded).hexdigest(),
                    "metadata_bytes": len(encoded), "files_manifest_sha256": _digest(files),
                    "identity_sha256": meta["identity_sha256"]}
            seal_bytes = _canonical(seal) + b"\n"
            logical = payload_bytes + len(encoded) + len(seal_bytes)
            if (len(encoded), len(seal_bytes), logical) == (
                    meta["bytes_metadata_file"], meta["bytes_seal_file"], meta["store_logical_bytes"]):
                break
            meta.update(bytes_metadata_file=len(encoded), bytes_seal_file=len(seal_bytes),
                        store_logical_bytes=logical)
        else:
            raise RuntimeError("completion byte counts did not converge")
        _write(path / "meta.json", encoded, stats, "metadata")
        _fsync_dir(path, stats)
        _write(path / ".seal.pending", seal_bytes, stats, "seal")
        os.replace(path / ".seal.pending", path / "seal.json")
        _fsync_dir(path, stats)
        _fsync_dir(path.parent, stats)
    except Exception:
        # Even a failure after atomic rename (e.g. directory fsync) must not
        # leave an apparently usable store. Preserve the failed seal as
        # evidence and make activation fail on the missing completion marker.
        if (path / "seal.json").exists():
            os.replace(path / "seal.json", path / ".seal.failed")
            try:
                _fsync_dir(path, stats)
            except OSError:
                pass
        raise
    finally:
        if timing_out is not None:
            timing_out.update({
                "kv_materialize_ms": stats["materialize_seconds"] * 1e3,
                "kv_repack_ms": stats["repack_seconds"] * 1e3,
                "serialization_ms": stats["serialize_seconds"] * 1e3,
                "ssd_write_ms": stats["write_seconds"] * 1e3,
                "fsync_ms": stats["fsync_seconds"] * 1e3,
                "total_persistence_ms": (time.perf_counter() - started) * 1e3,
                "full_cpu_materialized_bytes": len(prefix) * spec.bytes_per_token,
                "full_cpu_content_repacked_bytes": n * spec.bytes_per_token,
                "os_write_bytes": stats["os_write_bytes"], "os_write_calls": stats["os_write_calls"],
                "write_ranges": stats["write_ranges"],
            })
    if timing_out is not None:
        timing_out["store_allocated_bytes"] = sum(item.stat().st_blocks * 512
                                                  for item in [path, *path.rglob("*")])
        timing_out["total_persistence_ms"] = (time.perf_counter() - started) * 1e3
    return meta


class IOCounter:
    """Actual returned OS bytes, including short reads and zero-byte EOF calls."""
    def __init__(self):
        self.bytes = self.preads = self.spans = 0
        self.seconds = 0.0
        self.per_kind: dict[str, dict] = {}
        self.span_details: list[dict] = []
        self.call_details: list[dict] = []

    def span(self, kind: str, source: str, offset: int, count: int):
        self.spans += 1
        self.span_details.append({"kind": kind, "source": source, "offset": offset,
                                  "requested_bytes": count})
        self.per_kind.setdefault(kind, {"bytes": 0, "preads": 0, "spans": 0, "ms": 0.0})["spans"] += 1

    def call(self, kind: str, source: str, offset: int, requested: int,
             actual: int, seconds: float):
        self.bytes += actual
        self.preads += 1
        self.seconds += seconds
        entry = self.per_kind[kind]
        entry["bytes"] += actual
        entry["preads"] += 1
        entry["ms"] += seconds * 1e3
        self.call_details.append({"kind": kind, "source": source, "offset": offset,
                                  "requested_bytes": requested, "actual_bytes": actual,
                                  "ms": seconds * 1e3})

    def summary(self) -> dict:
        return {"bytes": self.bytes, "preads": self.preads, "spans": self.spans,
                "ms": self.seconds * 1e3, "span_details": list(self.span_details),
                "call_details": list(self.call_details),
                "per_kind": {key: dict(value) for key, value in self.per_kind.items()}}


def _pread(fd: int, count: int, offset: int, io: IOCounter, kind: str, source: str) -> bytes:
    if not count:
        return b""
    io.span(kind, source, offset, count)
    blocks = []
    while count:
        started = time.perf_counter()
        block = os.pread(fd, count, offset)
        io.call(kind, source, offset, count, len(block), time.perf_counter() - started)
        if not block:
            raise EOFError(f"short store pread {source} at {offset}: {count} bytes missing")
        blocks.append(block)
        count -= len(block)
        offset += len(block)
    return b"".join(blocks)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate metadata key: {key}")
        result[key] = value
    return result


def _deep_size(value: Any, seen=None) -> int:
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(_deep_size(key, seen) + _deep_size(item, seen) for key, item in value.items())
    elif isinstance(value, (list, tuple)):
        size += sum(_deep_size(item, seen) for item in value)
    return size


@dataclass
class LoadedKV:
    layers: list[tuple[torch.Tensor, torch.Tensor]]
    logical_indices: tuple[int, ...]
    position_ids: torch.Tensor
    prefix_input_ids: tuple[int, ...]
    selected_content_original: tuple[int, ...]
    selected_content_logical: tuple[int, ...]
    native_prefix_length: int
    rope_offset: Any
    original_content_count: int
    kept_visual_tokens: int
    selected_chunks: int
    structural_rows: int
    planning_ms: float
    assembly_ms: float
    io: IOCounter
    h2d_kv_bytes: int
    padding_rows_read: int = 0

    @property
    def selected_visual_original(self):
        return self.selected_content_original


class GLMStore:
    """Sealed activated store. No resident tensors or payload byte buffers."""
    def __init__(self, path: str | Path, *, expected_identity: dict):
        started = time.perf_counter()
        self.path = Path(path)
        self._fds: dict[str, int] = {}
        self.activation_io = IOCounter()
        self._closed = False
        try:
            seal_raw = self._read_aux("seal.json")
            seal = json.loads(seal_raw, object_pairs_hook=_unique_object)
            raw = self._read_aux("meta.json")
            if (not isinstance(seal, dict) or seal.get("format") != SEAL_FORMAT
                    or seal.get("completed") is not True
                    or seal.get("metadata_bytes") != len(raw)
                    or seal.get("metadata_sha256") != hashlib.sha256(raw).hexdigest()):
                raise ValueError("invalid/incomplete store completion seal or metadata hash")
            self.meta = json.loads(raw, object_pairs_hook=_unique_object)
            self.metadata_file_sha256 = hashlib.sha256(raw).hexdigest()
            self._validate_meta(expected_identity, len(raw), len(seal_raw))
            if (seal.get("files_manifest_sha256") != _digest(self.meta["files"])
                    or seal.get("identity_sha256") != self.meta["identity_sha256"]):
                raise ValueError("completion seal contradicts payload/identity manifest")
            self._verify_payloads()
            self.metadata_resident_bytes = _deep_size(self.meta)
            self.store_allocated_bytes = sum(item.stat().st_blocks * 512
                                              for item in [self.path, *self.path.rglob("*")])
            self.activation_ms = (time.perf_counter() - started) * 1e3
        except Exception:
            self.close()
            raise

    def _open(self, path: Path) -> int:
        # Refuse symlink redirection; expected paths are fixed by native shape.
        if path.parent.is_symlink() or path.is_symlink():
            raise ValueError("store paths must not be symbolic links")
        return os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                       | getattr(os, "O_CLOEXEC", 0))

    def _read_aux(self, rel: str) -> bytes:
        fd = self._open(self.path / rel)
        try:
            return _pread(fd, os.fstat(fd).st_size, 0, self.activation_io,
                          "activation_metadata", rel)
        finally:
            os.close(fd)

    def _fd(self, rel: str) -> int:
        if self._closed:
            raise ValueError("store reader is closed")
        if rel not in self._fds:
            self._fds[rel] = self._open(self.path / rel)
        return self._fds[rel]

    def _validate_meta(self, expected: dict, metadata_bytes: int, seal_bytes: int):
        m = self.meta
        if not isinstance(m, dict) or (m.get("format"), m.get("dtype"), m.get("byte_order")) != (
                FORMAT, "bfloat16", sys.byteorder):
            raise ValueError("unsupported or mislabeled native GLM store")
        integer_fields = ("num_layers", "num_kv_heads", "head_dim", "prefix_len",
                          "original_content_count", "stored_content_count", "stored_rows",
                          "n_chunks", "padding_rows", "valid_rows_last_chunk", "chunk_size",
                          "row_bytes", "structural_count", "bytes_content_kv",
                          "bytes_structural_kv", "bytes_metadata_file", "bytes_seal_file",
                          "store_logical_bytes", "bytes_per_native_token")
        if any(type(m.get(key)) is not int or m[key] < 0 for key in integer_fields):
            raise ValueError("invalid store integer geometry")
        spec = NativeKVSpec(m["num_layers"], m["num_kv_heads"], m["head_dim"])
        prefix = _prefix_ids(m["prefix_input_ids"])
        if len(prefix) != m["prefix_len"] or _digest(prefix) != m["prefix_sha256"]:
            raise ValueError("prefix token IDs/length/hash mismatch")
        content = _content(m["content_logical_indices"], len(prefix))
        n = len(content)
        content_set = set(content)
        structural = [index for index in range(len(prefix)) if index not in content_set]
        if m["structural_indices"] != structural or m["structural_count"] != len(structural):
            raise ValueError("duplicate, missing or invalid structure logical IDs")
        order = _ints(m["full_importance_permutation"], "full importance permutation")
        if len(order) != n or inverse_permutation(order) != m["original_to_stored"]:
            raise ValueError("invalid full original-content permutation/inverse")
        if m["permutation_sha256"] != _digest(order):
            raise ValueError("importance permutation hash mismatch")
        policy = m["storage_policy"]
        if policy not in ("full", "prefix25"):
            raise ValueError("invalid storage policy")
        k = n if policy == "full" else (n + 3) // 4
        chunks = (k + 63) // 64
        if (m["original_content_count"] != n or m["stored_content_count"] != k
                or m["stored_rows"] != k or m["chunk_size"] != 64
                or m["n_chunks"] != chunks or m["padding_rows"] != 0
                or m["valid_rows_last_chunk"] != k - (chunks - 1) * 64
                or m["stored_row_to_original"] != order[:k]
                or m["stored_to_original"] != order[:k]):
            raise ValueError("invalid original-N budget/mapping/chunk geometry")
        if (m["native_kv_shape"] != [1, spec.num_kv_heads, len(prefix), spec.head_dim]
                or m["row_bytes"] != spec.num_kv_heads * spec.head_dim * _ITEM_BYTES
                or m["bytes_per_native_token"] != spec.bytes_per_token
                or m["global_order_all_layers"] is not True
                or m["structural_layout"] != "kv_kind_layer_token_head_dimension"):
            raise ValueError("native KV layout/geometry mismatch")
        if policy == "full":
            if (order != list(range(n)) or m["layout"] != "token_major_canonical_bf16"
                    or any(m[key] is not None for key in ("score_source", "score_layer", "saliency_sha256"))
                    or m["retention_ratio"] != 1.0 or m["rounding_policy"] != "all_original_content"):
                raise ValueError("FullLoad must be canonical without score/importance sidecar")
        elif (m["layout"] != "token_major_repacked_bf16" or not _sha(m["saliency_sha256"])
              or not isinstance(m["score_source"], str) or not m["score_source"]
              or type(m["score_layer"]) is not int or m["score_layer"] < 0
              or m["retention_ratio"] != .25 or m["rounding_policy"] != "ceil_original_content"):
            raise ValueError("prefix25 lacks frozen image-only saliency/budget provenance")
        extra = _validate_extra(m["extra"], len(prefix))
        if policy == "prefix25" and (extra.get("score_source") != m["score_source"]
                                     or extra.get("score_layer") != m["score_layer"]):
            raise ValueError("frozen saliency provenance contradicts store metadata")
        if policy == "full" and any(key in extra for key in ("score_source", "score_layer")):
            raise ValueError("FullLoad must not persist importance provenance")
        if (m["logical_position_ids"] != extra["logical_position_ids"]
                or m["key_rope_state"] != extra["key_rope_state"]
                or m["rope_config"] != extra["rope_config"]
                or m["rope_offset"] != extra["rope_offset"]
                or m["geometry"] != extra["geometry"]):
            raise ValueError("native position/RoPE/geometry identity mismatch")
        identity = _identity(extra, prefix, content, spec)
        if m["identity"] != identity or m["identity_sha256"] != _digest(identity):
            raise ValueError("store identity contradicts prefix metadata")
        if not isinstance(expected, dict) or not expected or _jsonable(expected) != identity:
            raise ValueError("cache identity mismatch: complete expected native prefix identity required")
        paths = {"structural_kv.bin"} | {f"layer_{li:03d}/{kind}.bin"
                                          for li in range(spec.num_layers) for kind in ("k", "v")}
        if not isinstance(m["files"], dict) or set(m["files"]) != paths:
            raise ValueError("payload file manifest mismatch")
        top = {"meta.json", "seal.json", "structural_kv.bin"} | {
            f"layer_{li:03d}" for li in range(spec.num_layers)}
        if {item.name for item in self.path.iterdir()} != top:
            raise ValueError("unexpected/incomplete store files")
        for li in range(spec.num_layers):
            layer_path = self.path / f"layer_{li:03d}"
            if layer_path.is_symlink() or {item.name for item in layer_path.iterdir()} != {"k.bin", "v.bin"}:
                raise ValueError("unexpected layer payload paths")
        content_bytes = k * spec.bytes_per_token
        structural_bytes = len(structural) * spec.bytes_per_token
        if (m["bytes_content_kv"] != content_bytes or m["bytes_structural_kv"] != structural_bytes
                or m["bytes_metadata_file"] != metadata_bytes or m["bytes_seal_file"] != seal_bytes
                or m["store_logical_bytes"] != content_bytes + structural_bytes + metadata_bytes + seal_bytes):
            raise ValueError("payload/metadata/seal byte count mismatch")
        for rel, record in m["files"].items():
            size = structural_bytes if rel == "structural_kv.bin" else k * m["row_bytes"]
            if (not isinstance(record, dict) or type(record.get("size")) is not int
                    or record["size"] != size or not _sha(record.get("sha256"))
                    or os.fstat(self._fd(rel)).st_size != size):
                raise ValueError(f"short/oversized or malformed payload: {rel}")

    def _verify_payloads(self):
        for rel, record in self.meta["files"].items():
            digest = hashlib.sha256()
            offset = 0
            while offset < record["size"]:
                count = min(4 * 1024 * 1024, record["size"] - offset)
                payload = _pread(self._fd(rel), count, offset, self.activation_io,
                                 "activation_structure" if rel == "structural_kv.bin" else "activation_content", rel)
                digest.update(payload)
                offset += len(payload)
            if digest.hexdigest() != record["sha256"]:
                raise ValueError(f"payload hash mismatch: {rel}")

    def load_prefix(self) -> LoadedKV:
        """Read all sealed rows once per request; B has frozen first-k only."""
        started = time.perf_counter()
        m = self.meta
        if self._closed:
            raise ValueError("store reader is closed")
        content = [m["content_logical_indices"][index] for index in m["stored_row_to_original"]]
        combined = m["structural_indices"] + content
        sorting = sorted(range(len(combined)), key=combined.__getitem__)
        logical = tuple(combined[index] for index in sorting)
        sorting_tensor = torch.tensor(sorting, dtype=torch.long)
        planning_ms = (time.perf_counter() - started) * 1e3
        io = IOCounter()
        assembly_seconds = 0.0
        payload = _pread(self._fd("structural_kv.bin"), m["bytes_structural_kv"], 0,
                         io, "structure", "structural_kv.bin")
        t0 = time.perf_counter()
        structural = _tensor(payload, (2, m["num_layers"], m["structural_count"],
                                        m["num_kv_heads"], m["head_dim"]))
        assembly_seconds += time.perf_counter() - t0
        layers = []
        for li in range(m["num_layers"]):
            pair = []
            for ki, kind in enumerate(("k", "v")):
                rel = f"layer_{li:03d}/{kind}.bin"
                payload = _pread(self._fd(rel), m["stored_rows"] * m["row_bytes"], 0,
                                 io, "content", rel)
                t0 = time.perf_counter()
                rows = _tensor(payload, (m["stored_rows"], m["num_kv_heads"], m["head_dim"]))
                rows = torch.cat((structural[ki, li], rows), dim=0).index_select(0, sorting_tensor)
                pair.append(rows.permute(1, 0, 2).unsqueeze(0).contiguous())
                assembly_seconds += time.perf_counter() - t0
            layers.append((pair[0], pair[1]))
        t0 = time.perf_counter()
        positions = torch.tensor([[axis[index] for index in logical]
                                  for axis in m["logical_position_ids"]], dtype=torch.long).unsqueeze(1)
        prefix = tuple(m["prefix_input_ids"][index] for index in logical)
        assembly_seconds += time.perf_counter() - t0
        return LoadedKV(layers, logical, positions, prefix,
                        tuple(sorted(m["stored_row_to_original"])), tuple(sorted(content)),
                        m["prefix_len"], _jsonable(m["rope_offset"]), m["original_content_count"],
                        m["stored_content_count"], m["n_chunks"], m["structural_count"],
                        planning_ms, assembly_seconds * 1e3, io,
                        m["bytes_content_kv"] + m["bytes_structural_kv"])

    def drop_payload_cache(self) -> dict:
        """OS page-cache hint only; success does not establish cold NAND I/O."""
        failed = []
        for rel in self.meta["files"]:
            try:
                if not callable(getattr(os, "posix_fadvise", None)) or not hasattr(os, "POSIX_FADV_DONTNEED"):
                    raise OSError("posix_fadvise unavailable")
                os.posix_fadvise(self._fd(rel), 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError as exc:
                failed.append({"path": rel, "error": str(exc)})
        return {"attempted": len(self.meta["files"]), "failed": failed}

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def open_glm_store(path: str | Path, *, expected_identity: dict) -> GLMStore:
    """Activate after writer return; verification cannot be disabled."""
    return GLMStore(path, expected_identity=expected_identity)

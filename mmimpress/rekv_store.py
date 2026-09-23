"""Same-forward pre-RoPE capture and an isolated SSD store for ReKV-Chunk25.

The existing image stores contain rotated keys and serve different algorithms.
This module records the *projection outputs* of the ordinary Turn-1 pixel
prefill, then writes raw keys and values in canonical visual-token order.
Only small BF16 block representatives are activated on the GPU for cache hits;
the scoring path casts both vectors to FP32 as in the pinned official code.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mmimpress.config import CHUNK_SIZE
from mmimpress.model import cache_layers
from mmimpress.piggyback import (
    _fsync_directory, _fsync_staging_tree, _rename_noreplace,
    stable_json_sha256,
)
from mmimpress.store import ChunkReader, IOCounter, merge_ranges, n_chunks


STORE_SCHEMA = "rekv-pre-rope-ssd-v1"
KEY_REPRESENTATION = "pre_rope_k_projection"
VALUE_REPRESENTATION = "pre_rope_v_projection"
REPRESENTATIVE_DTYPE = "bfloat16"
PAYLOAD_DTYPE = "float16"


def _sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _payload_hash(root: Path, paths: Sequence[str]) -> str:
    """Hash the named payload files and their relative paths in fixed order."""
    digest = hashlib.sha256()
    for relative in sorted(paths):
        path = root / relative
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _input_ids_1d(input_ids: torch.Tensor) -> torch.Tensor:
    ids = torch.as_tensor(input_ids)
    if ids.ndim == 2:
        if ids.shape[0] != 1:
            raise ValueError("ReKV requires one image and batch size one")
        ids = ids[0]
    if ids.ndim != 1:
        raise ValueError(f"expected one token sequence, got {tuple(ids.shape)}")
    return ids.detach().to(device="cpu", dtype=torch.long)


class ReKVCapture:
    """Capture raw K and V directly from every decoder projection in Turn 1.

    Projection forward hooks see values before LLaMA reshapes or applies RoPE.
    Only the system-plus-image prefix is copied to CPU; question and generated
    tokens never enter the persistent source. Hooks and active markers are
    removed even when model generation fails.
    """

    def __init__(self, runner, visual_start: int, visual_count: int):
        self.runner = runner
        self.visual_start = int(visual_start)
        self.visual_count = int(visual_count)
        self.prefix_len = self.visual_start + self.visual_count
        if self.visual_start < 0 or self.visual_count < 1:
            raise ValueError("invalid visual span for ReKV capture")
        self.num_heads = int(runner.n_heads)
        self.head_dim = int(runner.head_dim)
        self._layers: list[tuple[torch.Tensor | None, torch.Tensor | None]] = [
            (None, None) for _ in runner.layers
        ]
        self._full_calls = [[0, 0] for _ in runner.layers]
        self._short_calls = [[0, 0] for _ in runner.layers]
        self._handles = []
        self._entered = False
        self._completed = False
        self.materialize_ms = 0.0

    def _hook(self, layer_index: int, kind_index: int):
        def hook(_module, _args, output):
            if not torch.is_tensor(output) or output.ndim != 3:
                raise AssertionError("K/V projection did not return [B,S,H*D]")
            if output.shape[0] != 1 or output.shape[-1] != self.num_heads * self.head_dim:
                raise AssertionError(
                    f"unexpected ReKV projection shape {tuple(output.shape)}")
            if int(output.shape[1]) < self.prefix_len:
                self._short_calls[layer_index][kind_index] += 1
                return
            self._full_calls[layer_index][kind_index] += 1
            if self._full_calls[layer_index][kind_index] != 1:
                raise AssertionError("multiple full image prefills in one Turn-1 capture")
            started = time.perf_counter()
            raw = output[0, :self.prefix_len].detach().reshape(
                self.prefix_len, self.num_heads, self.head_dim)
            # Preserve native BF16 for mean-before-storage-quantization.
            cpu = raw.to(device="cpu").contiguous()
            self.materialize_ms += (time.perf_counter() - started) * 1e3
            prior = self._layers[layer_index]
            self._layers[layer_index] = (
                cpu if kind_index == 0 else prior[0],
                cpu if kind_index == 1 else prior[1],
            )
        return hook

    def __enter__(self):
        if self._entered:
            raise RuntimeError("ReKVCapture is single-use")
        self._entered = True
        if getattr(self.runner, "_mmimpress_active_rekv_capture", None) is not None:
            raise RuntimeError("concurrent ReKV capture is unsafe")
        setattr(self.runner, "_mmimpress_active_rekv_capture", self)
        try:
            for li, layer in enumerate(self.runner.layers):
                self._handles.append(layer.self_attn.k_proj.register_forward_hook(
                    self._hook(li, 0)))
                self._handles.append(layer.self_attn.v_proj.register_forward_hook(
                    self._hook(li, 1)))
        except BaseException:
            self._remove()
            raise
        return self

    def _remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if getattr(self.runner, "_mmimpress_active_rekv_capture", None) is self:
            delattr(self.runner, "_mmimpress_active_rekv_capture")

    def __exit__(self, exc_type, _exc, _traceback):
        self._remove()
        if exc_type is not None:
            self._layers = [(None, None) for _ in self._layers]
            return False
        for li, (key, value) in enumerate(self._layers):
            if self._full_calls[li] != [1, 1] or key is None or value is None:
                raise AssertionError(
                    f"layer {li} lacks one raw K/V image prefill: "
                    f"{self._full_calls[li]}")
            if key.shape != value.shape or key.shape != (
                    self.prefix_len, self.num_heads, self.head_dim):
                raise AssertionError(f"invalid raw K/V shape in layer {li}")
            if not torch.isfinite(key).all() or not torch.isfinite(value).all():
                raise AssertionError(f"non-finite captured raw K/V in layer {li}")
        self._completed = True
        return False

    def result_cpu(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        if not self._completed:
            raise RuntimeError("ReKV K/V is available after successful context exit")
        return tuple((key, value) for key, value in self._layers)

    def stats(self) -> dict[str, Any]:
        return {
            "capture_source": "same_turn1_normal_multimodal_prefill",
            "key_capture_point": "k_proj_output_before_rope",
            "value_capture_point": "v_proj_output",
            "capture_complete": self._completed,
            "full_prefill_calls_per_layer": [list(row) for row in self._full_calls],
            "short_decode_calls_per_layer": [list(row) for row in self._short_calls],
            "num_layers": len(self._layers),
            "visual_start": self.visual_start,
            "visual_count": self.visual_count,
            "prefix_len": self.prefix_len,
            "capture_dtype": (str(self._layers[0][0].dtype)
                              if self._completed else None),
            "capture_cpu_bytes": int(sum(
                key.numel() * key.element_size()
                + value.numel() * value.element_size()
                for key, value in self._layers if key is not None
                and value is not None)),
            "materialize_ms": float(self.materialize_ms),
            "separate_model_forward_count": 0,
            "separate_vision_forward_count": 0,
            "hooks_removed": not self._handles,
        }


def representative_keys(
    visual_keys: torch.Tensor, separators: Sequence[int], chunk_size: int,
    *, device=None,
) -> tuple[torch.Tensor, list[int]]:
    """Native-dtype torch.mean per block, then all-head BF16 metadata."""
    if visual_keys.ndim != 3 or visual_keys.shape[0] < 1:
        raise ValueError("visual keys must have shape [tokens, heads, dim]")
    v_num = int(visual_keys.shape[0])
    separator_set = {int(pos) for pos in separators}
    if any(pos < 0 or pos >= v_num for pos in separator_set):
        raise ValueError("separator outside visual span")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    width = int(visual_keys.shape[1] * visual_keys.shape[2])
    compute_device = torch.device(device) if device is not None else visual_keys.device
    keys = visual_keys.to(device=compute_device)
    reps = torch.zeros(n_chunks(v_num, chunk_size), width,
                       dtype=torch.bfloat16, device=compute_device)
    counts: list[int] = []
    for ci in range(reps.shape[0]):
        start = ci * int(chunk_size)
        stop = min(start + int(chunk_size), v_num)
        valid = [pos for pos in range(start, stop) if pos not in separator_set]
        counts.append(len(valid))
        if valid:
            # The pinned original calls K.mean(dim=token_axis) in native dtype.
            reps[ci] = keys[valid].mean(dim=0).reshape(-1).to(torch.bfloat16)
    return reps.to(device="cpu"), counts


@torch.no_grad()
def validate_pre_rope_capture_against_cache(
    runner, raw_capture: ReKVCapture, captured_past_key_values, *,
    atol: float = 0.03125, rtol: float = 0.015625,
) -> dict[str, Any]:
    """Small original-position RoPE diagnostic against the same Turn-1 cache.

    The tolerance is predeclared for BF16 projection and RoPE arithmetic.
    This is meant for the actual-model smoke, outside the full store writer.
    """
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    captured = raw_capture.result_cpu()
    normal = cache_layers(captured_past_key_values)
    if len(captured) != len(normal):
        raise ValueError("raw and post-RoPE cache layer counts differ")
    length = int(raw_capture.prefix_len)
    sample = sorted({
        0, min(1, length - 1), raw_capture.visual_start,
        raw_capture.visual_start + raw_capture.visual_count // 2,
        length - 1,
    })
    position_cpu = torch.tensor(sample, dtype=torch.long)
    rotary = runner.model.model.language_model.rotary_emb
    results = []
    for li, ((raw_k, raw_v), (post_k, post_v)) in enumerate(zip(captured, normal)):
        device = post_k.device
        positions = position_cpu.to(device)
        unrotated = raw_k.index_select(0, position_cpu).to(
            device=device, dtype=post_k.dtype).permute(1, 0, 2).unsqueeze(0)
        cos, sin = rotary(unrotated, positions.unsqueeze(0))
        _, expected = apply_rotary_pos_emb(
            torch.zeros_like(unrotated), unrotated, cos, sin)
        observed = post_k.index_select(2, positions)
        if expected.shape != observed.shape:
            raise AssertionError("pre-RoPE diagnostic K shape mismatch")
        difference = (expected.float() - observed.float()).abs()
        scale = observed.float().abs().clamp(min=1e-12)
        key_ok = bool(torch.allclose(
            expected.float(), observed.float(), atol=atol, rtol=rtol))
        expected_v = raw_v.index_select(0, position_cpu).to(
            device=device, dtype=post_v.dtype).permute(1, 0, 2).unsqueeze(0)
        observed_v = post_v.index_select(2, positions)
        value_equal = bool(torch.equal(expected_v, observed_v))
        results.append({
            "layer": li,
            "sample_source_positions": sample,
            "max_abs_k_error": float(difference.max().item()),
            "max_relative_k_error": float((difference / scale).max().item()),
            "raw_v_equals_cache_v": value_equal,
            "key_allclose": key_ok,
            "passed": key_ok and value_equal,
        })
    return {
        "passed": all(row["passed"] for row in results),
        "layers": results,
        "sample_count_per_layer": len(sample),
        "atol": float(atol),
        "rtol": float(rtol),
        "capture_key_representation": KEY_REPRESENTATION,
        "cache_key_representation": "post_rope_original_positions",
        "position_policy": "original_prefix_positions_for_diagnostic_only",
    }


def _size_and_shape(meta: Mapping[str, Any], root: Path) -> None:
    v_num = int(meta["v_token_num"])
    layers = int(meta["num_layers"])
    heads = int(meta["num_heads"])
    head_dim = int(meta["head_dim"])
    itemsize = 2
    expected_visual = v_num * heads * head_dim * itemsize
    for li in range(layers):
        for kind in ("k", "v"):
            path = root / f"layer_{li:02d}" / f"{kind}.bin"
            if path.stat().st_size != expected_visual:
                raise ValueError(f"ReKV payload size mismatch: {path}")
    n_sep = len(meta["newline_idx"])
    expected_sep = 2 * layers * n_sep * heads * head_dim * itemsize
    if (root / "sep_kv.bin").stat().st_size != expected_sep:
        raise ValueError("ReKV separator sidecar size mismatch")


@torch.no_grad()
def persist_captured_rekv_prefix(
    runner,
    captured_past_key_values,
    input_ids: torch.Tensor,
    image_size: Sequence[int] | torch.Tensor,
    raw_capture: ReKVCapture,
    destination: Path,
    *,
    image_id: str | int,
    model_id: str | None = None,
    chunk_size: int = CHUNK_SIZE,
    image_input_sha256: str | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    full_integrity_hash: bool = False,
) -> dict[str, Any]:
    """Publish a canonical raw-K image store with atomic no-clobber rename."""
    persist_started = time.perf_counter()
    target = Path(destination)
    if target.name in ("", ".", ".."):
        raise ValueError("unsafe ReKV store destination")
    target.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(target):
        raise FileExistsError(errno.EEXIST, "refusing to overwrite ReKV store", target)
    if not isinstance(raw_capture, ReKVCapture) or raw_capture.runner is not runner:
        raise TypeError("raw_capture must be the completed same-run ReKVCapture")
    captured = raw_capture.result_cpu()
    capture_stats = raw_capture.stats()
    if not capture_stats["hooks_removed"]:
        raise AssertionError("ReKV capture hooks remain active")

    ids = _input_ids_1d(input_ids)
    v_start, v_num = runner.visual_span(ids)
    prefix_len = v_start + v_num
    if (v_start, v_num) != (raw_capture.visual_start, raw_capture.visual_count):
        raise AssertionError("ReKV capture and input visual spans differ")
    if torch.is_tensor(image_size):
        size = [int(x) for x in image_size.detach().cpu().reshape(-1)]
    else:
        size = [int(x) for x in image_size]
    if len(size) != 2:
        raise ValueError("image size must have two dimensions")
    base, hi_h, hi_w, separators = runner.anyres_layout(size, v_num)
    separators = sorted(int(x) for x in separators)
    if len(separators) != len(set(separators)):
        raise AssertionError("duplicate image separators")

    normal_cache = cache_layers(captured_past_key_values)
    if len(normal_cache) != len(captured) or not captured:
        raise AssertionError("normal Turn-1 cache and raw capture layer counts differ")
    heads, head_dim = int(runner.n_heads), int(runner.head_dim)
    for li, ((raw_k, raw_v), (normal_k, normal_v)) in enumerate(
            zip(captured, normal_cache)):
        if raw_k.shape != (prefix_len, heads, head_dim) or raw_v.shape != raw_k.shape:
            raise AssertionError(f"bad raw prefix shape at layer {li}")
        if normal_k.shape[2] < prefix_len or normal_v.shape[2] < prefix_len:
            raise AssertionError(f"normal Turn-1 cache too short at layer {li}")
        observed_v = normal_v[0, :, :prefix_len].detach().to(
            device="cpu", dtype=raw_v.dtype).permute(1, 0, 2).contiguous()
        if not torch.equal(raw_v, observed_v):
            raise AssertionError(
                f"raw-capture V differs from normal attention cache at layer {li}")

    n_physical = n_chunks(v_num, int(chunk_size))
    rep_layers = []
    valid_counts = None
    started = time.perf_counter()
    for key, _ in captured:
        rep, counts = representative_keys(
            key[v_start:prefix_len], separators, int(chunk_size),
            device=getattr(getattr(runner, "model", None), "device", None))
        rep_layers.append(rep)
        if valid_counts is None:
            valid_counts = counts
        elif valid_counts != counts:
            raise AssertionError("normal chunk validity varies by layer")
    representatives = torch.stack(rep_layers).contiguous()
    representative_build_ms = (time.perf_counter() - started) * 1e3
    assert representatives.shape == (len(captured), n_physical, heads * head_dim)
    assert valid_counts is not None
    candidate_ids = [ci for ci, count in enumerate(valid_counts) if count > 0]
    if not candidate_ids:
        raise AssertionError("image has no rankable ReKV chunk")

    model_config_hash = stable_json_sha256(runner.cfg.to_dict())
    prefix_ids = ids[:prefix_len].tolist()
    metadata = {
        "schema_version": STORE_SCHEMA,
        "method_id": "rekv_chunk25",
        "image_id": str(image_id),
        "model_id": str(model_id or runner.model_id),
        "model_revision": getattr(runner.cfg, "_commit_hash", None),
        "model_config_sha256": model_config_hash,
        "source_prefix_sha256": stable_json_sha256(prefix_ids),
        "source_prefix_input_ids": prefix_ids,
        "image_input_sha256": image_input_sha256,
        "v_token_start": int(v_start),
        "v_token_num": int(v_num),
        "prefix_len": int(prefix_len),
        "num_layers": len(captured),
        "num_heads": heads,
        "head_dim": head_dim,
        "dtype": PAYLOAD_DTYPE,
        "key_representation": KEY_REPRESENTATION,
        "value_representation": VALUE_REPRESENTATION,
        "physical_layout": "canonical_raster",
        "reordered": False,
        "chunk_size": int(chunk_size),
        "n_chunks_per_layer": n_physical,
        "normal_chunk_count": n_physical,
        "normal_candidate_chunk_ids": candidate_ids,
        "valid_spatial_counts": valid_counts,
        "n_spatial": int(sum(valid_counts)),
        "newline_idx": separators,
        "newline_stored": separators,
        "original_visual_token_positions": list(range(v_start, prefix_len)),
        "separator_source_positions": [v_start + pos for pos in separators],
        "representative_dtype": REPRESENTATIVE_DTYPE,
        "representative_metadata_bytes": int(representatives.numel() * 2),
        "valid_counts_metadata_bytes": int(len(valid_counts) * 8),
        "metadata_gpu_bytes_total": int(
            representatives.numel() * 2 + len(valid_counts) * 8),
        "representative_residency": "gpu_active_image",
        "representative_build_ms": float(representative_build_ms),
        "capture_stats": capture_stats,
        "anyres_layout": {"base": base, "hi_h": hi_h, "hi_w": hi_w},
    }
    if extra_metadata:
        reserved = set(metadata)
        if reserved.intersection(extra_metadata):
            raise ValueError("extra ReKV metadata would override invariants")
        metadata.update(dict(extra_metadata))

    staging = Path(tempfile.mkdtemp(
        prefix=f".{target.name}.staging-", dir=target.parent))
    published = False
    try:
        write_started = time.perf_counter()
        payload_paths: list[str] = []
        sys_k = torch.stack([key[:v_start].permute(1, 0, 2).to(torch.float16)
                             for key, _ in captured])
        sys_v = torch.stack([value[:v_start].permute(1, 0, 2).to(torch.float16)
                             for _, value in captured])
        torch.save({"k": sys_k, "v": sys_v}, staging / "sys_kv.pt")
        payload_paths.append("sys_kv.pt")
        torch.save(representatives, staging / "k_rep.pt")
        sep_index = torch.tensor(separators, dtype=torch.long)
        sidecar = torch.stack([
            torch.stack([key[v_start:prefix_len].index_select(0, sep_index)
                         for key, _ in captured]),
            torch.stack([value[v_start:prefix_len].index_select(0, sep_index)
                         for _, value in captured]),
        ]).to(torch.float16).contiguous()
        (staging / "sep_kv.bin").write_bytes(sidecar.numpy().tobytes())
        payload_paths.append("sep_kv.bin")
        visual_bytes = 0
        for li, (key, value) in enumerate(captured):
            layer_dir = staging / f"layer_{li:02d}"
            layer_dir.mkdir()
            for kind, source in (("k", key), ("v", value)):
                visual = source[v_start:prefix_len].to(torch.float16).contiguous()
                if visual.shape != (v_num, heads, head_dim):
                    raise AssertionError("raw visual payload geometry changed")
                payload = visual.numpy().tobytes()
                relative = f"layer_{li:02d}/{kind}.bin"
                (staging / relative).write_bytes(payload)
                payload_paths.append(relative)
                visual_bytes += len(payload)
        write_ms = (time.perf_counter() - write_started) * 1e3
        metadata["bytes_visual_kv"] = int(visual_bytes)
        metadata["bytes_separator_sidecar"] = int(sidecar.numel() * 2)
        metadata["bytes_initial_kv"] = int(sys_k.numel() * 2 + sys_v.numel() * 2)
        metadata["source_payload_sha256"] = _payload_hash(staging, payload_paths)
        metadata["representative_file_sha256"] = _sha256_file(staging / "k_rep.pt")
        (staging / "meta.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _size_and_shape(metadata, staging)
        file_sizes = {
            path.relative_to(staging).as_posix(): int(path.stat().st_size)
            for path in sorted(staging.rglob("*")) if path.is_file()
        }
        file_fsync_ms, directory_fsync_ms, n_files, n_dirs = \
            _fsync_staging_tree(staging)
        rename_started = time.perf_counter()
        _rename_noreplace(staging, target)
        rename_ms = (time.perf_counter() - rename_started) * 1e3
        published = True
        parent_fsync_started = time.perf_counter()
        _fsync_directory(target.parent)
        parent_fsync_ms = (time.perf_counter() - parent_fsync_started) * 1e3
        persist_ms = (time.perf_counter() - persist_started) * 1e3
        if full_integrity_hash:
            if _payload_hash(target, payload_paths) != metadata["source_payload_sha256"]:
                raise AssertionError("published ReKV source payload hash changed")
        return {
            "store_dir": str(target.resolve()),
            "image_id": str(image_id),
            "meta": metadata,
            "file_sizes": file_sizes,
            "bytes": {
                "total": sum(file_sizes.values()),
                "visual_kv": int(visual_bytes),
                "separator": int(metadata["bytes_separator_sidecar"]),
                "representatives": int(metadata["representative_metadata_bytes"]),
                "initial_kv": int(metadata["bytes_initial_kv"]),
            },
            "hashes": {
                "source_payload_sha256": metadata["source_payload_sha256"],
                "representative_file_sha256": metadata["representative_file_sha256"],
            },
            "timing_ms": {
                "capture_materialize_ms": float(raw_capture.materialize_ms),
                "representative_build_ms": float(representative_build_ms),
                "store_write_ms": float(write_ms),
                "file_fsync_ms": float(file_fsync_ms),
                "directory_fsync_ms": float(directory_fsync_ms),
                "atomic_rename_ms": float(rename_ms),
                "parent_fsync_ms": float(parent_fsync_ms),
                "persist_ms": float(persist_ms),
            },
            "durability": {
                "same_filesystem_staging": True,
                "atomic_no_clobber": True,
                "files_fsynced": int(n_files),
                "directories_fsynced_before_rename": int(n_dirs),
                "parent_fsynced_after_rename": True,
            },
        }
    finally:
        if not published and os.path.lexists(staging):
            shutil.rmtree(staging)


class ReKVRawReader(ChunkReader):
    """Selected canonical raw-K/V chunks only, coalesced by contiguous run."""

    def __init__(self, store_dir, meta, drop_cache=True):
        super().__init__(store_dir, meta, drop_cache=drop_cache)
        self.read_trace: list[tuple[int, str, int, int]] = []
        self.last_drop_statuses: list[dict[str, Any]] = []
        self.last_drop_ms = 0.0

    def reset_read_trace(self):
        self.read_trace.clear()

    def _read(self, layer, name, ranges, counter, kind, units, max_gap=0):
        actual = merge_ranges(ranges, max_gap=max_gap)
        blobs = super()._read(layer, name, ranges, counter, kind, units, max_gap)
        self.read_trace.extend(
            (int(layer), str(name), int(offset), int(length))
            for offset, length in actual)
        return blobs

    def drop_all(self):
        """Evict payload files and expose each actual fadvise call result."""
        started = time.perf_counter()
        statuses = []
        try:
            for path in sorted(self.dir.rglob("*.bin")):
                fd = os.open(path, os.O_RDONLY)
                try:
                    returned = os.posix_fadvise(
                        fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    statuses.append({
                        "path": path.relative_to(self.dir).as_posix(),
                        "advice": "POSIX_FADV_DONTNEED",
                        "return_value": returned,
                        "status": "ok",
                    })
                finally:
                    os.close(fd)
        finally:
            self.last_drop_statuses = statuses
            self.last_drop_ms = (time.perf_counter() - started) * 1e3
            self.reset_read_trace()
        return statuses

    def read_chunks(self, layer, kind, cids, counter=None, max_gap=0):
        if kind not in ("k", "v"):
            raise ValueError("ReKV reader accepts only raw K or V")
        if not 0 <= int(layer) < int(self.meta["num_layers"]):
            raise ValueError("ReKV layer index outside store")
        unique = sorted({int(cid) for cid in cids})
        candidates = set(int(x) for x in self.meta["normal_candidate_chunk_ids"])
        if not unique or any(cid not in candidates for cid in unique):
            raise ValueError("ReKV reader received empty or noncandidate chunks")
        if max_gap != 0:
            raise ValueError("ReKV selected reads cannot bridge unselected chunks")
        return super().read_chunks(layer, kind, unique, counter, max_gap=0)

    def read_full(self, *_args, **_kwargs):
        raise RuntimeError("ReKV retrieval must not load the full image payload")

    def read_probe(self, *_args, **_kwargs):
        raise RuntimeError("ReKV retrieval uses resident all-head metadata")


class ReKVContext:
    """Metadata-ready active image; selected visual payload remains on SSD."""

    def __init__(self, store_path: Path, device, runner=None, drop_cache=True):
        self.dir = Path(store_path)
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        meta = self.meta
        if meta.get("schema_version") != STORE_SCHEMA:
            raise ValueError("unsupported ReKV store schema")
        if meta.get("key_representation") != KEY_REPRESENTATION:
            raise ValueError("ReKV store keys are not pre-RoPE")
        if meta.get("physical_layout") != "canonical_raster" or meta.get("reordered"):
            raise ValueError("ReKV store is not in canonical visual order")
        if meta.get("dtype") != PAYLOAD_DTYPE or meta.get("representative_dtype") != REPRESENTATIVE_DTYPE:
            raise ValueError("ReKV store dtype mismatch")
        if len(meta["valid_spatial_counts"]) != int(meta["normal_chunk_count"]):
            raise ValueError("invalid ReKV chunk validity metadata")
        if sum(meta["valid_spatial_counts"]) != int(meta["n_spatial"]):
            raise ValueError("invalid ReKV spatial count")
        _size_and_shape(meta, self.dir)
        if runner is not None:
            if str(meta["model_id"]) != str(runner.model_id):
                raise ValueError("ReKV store model ID differs from runtime")
            if meta["model_config_sha256"] != stable_json_sha256(runner.cfg.to_dict()):
                raise ValueError("ReKV store model configuration differs")
            if int(meta["num_layers"]) != len(runner.layers):
                raise ValueError("ReKV store layer count differs")
        self.device = torch.device(device)
        activation_started = time.perf_counter()
        self.reader = ReKVRawReader(self.dir, meta, drop_cache=drop_cache)
        initial_started = time.perf_counter()
        sys_cpu = torch.load(
            self.dir / "sys_kv.pt", weights_only=True, map_location="cpu")
        expected_sys = (int(meta["num_layers"]), int(meta["num_heads"]),
                        int(meta["v_token_start"]), int(meta["head_dim"]))
        if set(sys_cpu) != {"k", "v"} or any(
                tuple(sys_cpu[kind].shape) != expected_sys
                or sys_cpu[kind].dtype != torch.float16
                for kind in ("k", "v")):
            raise ValueError("ReKV raw initial context shape/dtype mismatch")
        self.sys_kv = {
            kind: sys_cpu[kind].to(device=self.device, dtype=torch.bfloat16)
            for kind in ("k", "v")}
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.initial_context_activation_ms = (
            time.perf_counter() - initial_started) * 1e3
        self.initial_context_gpu_bytes = int(sum(
            value.numel() * value.element_size()
            for value in self.sys_kv.values()))
        self.initial_context_cpu_bytes = 0
        del sys_cpu

        metadata_started = time.perf_counter()
        rep_cpu = torch.load(
            self.dir / "k_rep.pt", weights_only=True, map_location="cpu")
        expected = (int(meta["num_layers"]), int(meta["normal_chunk_count"]),
                    int(meta["num_heads"] * meta["head_dim"]))
        if rep_cpu.dtype != torch.bfloat16 or tuple(rep_cpu.shape) != expected:
            raise ValueError("ReKV representative metadata shape/dtype mismatch")
        self.k_rep = rep_cpu.to(device=self.device, dtype=torch.bfloat16)
        self.valid_counts_gpu = torch.tensor(
            meta["valid_spatial_counts"], dtype=torch.long,
            device=self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.metadata_activation_ms = (
            time.perf_counter() - metadata_started) * 1e3
        self.activation_total_ms = (
            time.perf_counter() - activation_started) * 1e3
        self.representative_metadata_gpu_bytes = int(
            self.k_rep.numel() * self.k_rep.element_size())
        self.representative_metadata_cpu_bytes = 0
        self.metadata_gpu_bytes_total = (
            self.representative_metadata_gpu_bytes
            + self.valid_counts_gpu.numel()
            * self.valid_counts_gpu.element_size())
        del rep_cpu

    def read_sep_kv(self, counter: IOCounter | None = None) -> torch.Tensor:
        meta = self.meta
        shape = (2, int(meta["num_layers"]), len(meta["newline_idx"]),
                 int(meta["num_heads"]), int(meta["head_dim"]))
        expected = int(np.prod(shape)) * 2
        fd = os.open(self.dir / "sep_kv.bin", os.O_RDONLY)
        try:
            started = time.perf_counter()
            blob = os.pread(fd, expected, 0)
            elapsed = time.perf_counter() - started
            if len(blob) != expected:
                raise IOError("short ReKV separator pread")
            if counter is not None:
                counter.record("sep", expected, elapsed, preads=1)
            self.reader.read_trace.append((-1, "sep", 0, expected))
            return torch.from_numpy(np.frombuffer(
                blob, dtype=np.float16).copy().reshape(shape))
        finally:
            os.close(fd)

    def drop_payload_cache(self):
        self.reader.drop_all()

    @property
    def source_payload_hash(self) -> str:
        paths = ["sys_kv.pt", "sep_kv.bin"]
        for li in range(int(self.meta["num_layers"])):
            paths.extend([f"layer_{li:02d}/k.bin", f"layer_{li:02d}/v.bin"])
        return _payload_hash(self.dir, paths)

    def close(self):
        self.reader.close()
        self.k_rep = None
        self.valid_counts_gpu = None
        self.sys_kv = None


__all__ = [
    "ReKVCapture", "ReKVContext", "ReKVRawReader",
    "persist_captured_rekv_prefix", "representative_keys",
    "validate_pre_rope_capture_against_cache",
]

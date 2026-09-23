"""MPIC-style selective recomputation over an SSD-resident image cache.

This is an isolated Transformers adaptation of MPIC's selective-attention
mechanism.  It intentionally does not add a mode to the existing IMPRESS
selectors: those paths assume a contiguous prefix, whereas MPIC propagates a
non-contiguous set of active rows through every decoder layer.

The paper-facing contract lives in ``docs/mpic_baseline_contract.md``.  The
important invariants enforced here are:

* all current text rows and the first ``k`` canonical image rows are active;
* active rows retain their original logical positions;
* cached image K/V remains the complete attention context and active K/V
  replaces, rather than appends to, its logical slots;
* stored keys are post-RoPE.  They are untouched at the same position and an
  explicitly labelled phase relocation is used only by shifted diagnostics;
* the first token is produced by one decoder-layer traversal over active rows;
* the raw visual-input sidecar and every reused KV byte are read and accounted
  inside the request boundary.

This module never monkey-patches model attention and never consults the query
scoring or image-only repacking code.
"""
from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from mmimpress.config import CHUNK_SIZE
from mmimpress.model import cache_layers
from mmimpress.piggyback import DecoderVisualHiddenCapture, stable_json_sha256
from mmimpress.store import IOCounter


METHOD_ID = "mpic32_ssd"
PAPER_LABEL = "MPIC-32 (SSD adaptation)"
STORE_SCHEMA = "mpic-ssd-store-v1"
POSITION_POLICY_SAME = "post_rope_cached_k_reused_at_identical_logical_position"
POSITION_POLICY_SHIFTED = (
    "implementation_choice_post_rope_phase_relocation_only;"
    "does_not_repair_preceding_context_hidden_state_change"
)

_NP_DTYPES = {"float16": np.float16, "float32": np.float32}
_TORCH_DTYPES = {"float16": torch.float16, "float32": torch.float32}


def _runtime_provenance(runner) -> dict[str, Any]:
    """Stable compatibility identity for a persisted MPIC payload."""
    import transformers

    config = runner.model.config
    config_dict = config.to_dict()
    quantization = config_dict.get("quantization_config")
    processor = runner.processor
    image_processor = getattr(processor, "image_processor", None)
    tokenizer = getattr(processor, "tokenizer", None)
    return {
        "model": str(getattr(runner, "model_id", config_dict.get(
            "_name_or_path", "unknown"))),
        "model_revision": getattr(config, "_commit_hash", None),
        "model_config_sha256": stable_json_sha256(config_dict),
        "processor_class": (
            f"{type(processor).__module__}.{type(processor).__qualname__}"),
        "image_processor_class": (
            None if image_processor is None else
            f"{type(image_processor).__module__}."
            f"{type(image_processor).__qualname__}"),
        "tokenizer_class": (
            None if tokenizer is None else
            f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"),
        "tokenizer_vocab_size": (
            None if tokenizer is None else int(len(tokenizer))),
        "transformers_version": str(transformers.__version__),
        "torch_version": str(torch.__version__),
        "attention_implementation": getattr(
            config, "_attn_implementation", None),
        "load_4bit": bool(getattr(runner, "load_4bit", False)),
        "quantization_config_sha256": (
            None if quantization is None
            else stable_json_sha256(quantization)),
    }


# ---------------------------------------------------------------------------
# Pure active-row, mask, cache, and RoPE primitives

@dataclass(frozen=True)
class ActiveRowPlan:
    """Logical row sets used by one selective prefill."""

    total_tokens: int
    visual_start: int
    visual_tokens: int
    k_recompute: int
    active_positions: torch.Tensor
    image_positions: torch.Tensor
    text_positions: torch.Tensor
    reused_image_positions: torch.Tensor

    @property
    def n_active(self) -> int:
        return int(self.active_positions.numel())


def build_active_row_plan(total_tokens: int, visual_start: int,
                          visual_tokens: int,
                          k_recompute: int) -> ActiveRowPlan:
    """Select all text rows plus the first ``k`` canonical image rows."""
    total = int(total_tokens)
    start = int(visual_start)
    count = int(visual_tokens)
    if total < 1 or start < 0 or count < 1 or start + count > total:
        raise ValueError("invalid visual span/total length")
    k = min(max(int(k_recompute), 0), count)
    prefix_text = torch.arange(0, start, dtype=torch.long)
    image = torch.arange(start, start + k, dtype=torch.long)
    suffix_text = torch.arange(start + count, total, dtype=torch.long)
    text = torch.cat((prefix_text, suffix_text))
    active = torch.cat((prefix_text, image, suffix_text))
    reused = torch.arange(start + k, start + count, dtype=torch.long)
    if active.numel():
        assert bool(torch.all(active[1:] > active[:-1]))
    return ActiveRowPlan(
        total_tokens=total, visual_start=start, visual_tokens=count,
        k_recompute=k, active_positions=active,
        image_positions=image, text_positions=text,
        reused_image_positions=reused,
    )


def build_causal_mask(query_positions: torch.Tensor, key_length: int,
                      dtype: torch.dtype,
                      valid_key_mask: torch.Tensor | None = None) \
        -> torch.Tensor:
    """Return additive ``[1,1,Q,K]`` mask using original logical positions."""
    query = torch.as_tensor(query_positions, dtype=torch.long)
    if query.ndim != 1:
        raise ValueError("query_positions must be one-dimensional")
    length = int(key_length)
    if length < 1 or (query.numel() and
                      (int(query.min()) < 0 or int(query.max()) >= length)):
        raise ValueError("query positions outside key space")
    keys = torch.arange(length, device=query.device)
    allowed = keys.unsqueeze(0) <= query.unsqueeze(1)
    if valid_key_mask is not None:
        valid = torch.as_tensor(valid_key_mask, dtype=torch.bool,
                                device=query.device)
        if valid.shape != (length,):
            raise ValueError("valid_key_mask has wrong shape")
        allowed &= valid.unsqueeze(0)
    result = torch.full(
        (query.numel(), length), torch.finfo(dtype).min,
        dtype=dtype, device=query.device)
    result.masked_fill_(allowed, 0.0)
    return result.unsqueeze(0).unsqueeze(0)


def _as_hf_image_cache(value: torch.Tensor, visual_tokens: int) \
        -> torch.Tensor:
    """Accept HF ``[1,H,N,D]`` or token-major ``[N,H,D]`` image K/V."""
    if value.ndim == 4:
        if value.shape[0] != 1 or value.shape[2] != visual_tokens:
            raise ValueError("cached image tensor has wrong HF shape")
        return value
    if value.ndim == 3 and value.shape[0] == visual_tokens:
        return value.permute(1, 0, 2).unsqueeze(0)
    raise ValueError("cached image tensor must be [1,H,N,D] or [N,H,D]")


def assemble_linked_kv(cached_image_k: torch.Tensor,
                       cached_image_v: torch.Tensor,
                       active_k: torch.Tensor, active_v: torch.Tensor,
                       active_positions: torch.Tensor, visual_start: int,
                       total_tokens: int, dummy_value: float = 0.0) \
        -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble full logical K/V and indexed-replace every active row.

    ``cached_image_*`` contains every canonical image row.  Rows for the
    recomputed image prefix may contain any sentinel because ``active_*``
    replaces them before this function returns.  A coverage assertion proves
    no text dummy remains in the attention context.
    """
    n_image = (cached_image_k.shape[2] if cached_image_k.ndim == 4
               else cached_image_k.shape[0])
    image_k = _as_hf_image_cache(cached_image_k, int(n_image))
    image_v = _as_hf_image_cache(cached_image_v, int(n_image))
    if image_k.shape != image_v.shape:
        raise ValueError("cached K/V shape mismatch")
    if active_k.shape != active_v.shape or active_k.ndim != 4:
        raise ValueError("active K/V must be matching [1,H,A,D] tensors")
    if (active_k.shape[0] != 1 or active_k.shape[1] != image_k.shape[1]
            or active_k.shape[3] != image_k.shape[3]):
        raise ValueError("active and cached K/V dimensions disagree")
    positions = torch.as_tensor(active_positions, dtype=torch.long,
                                device=active_k.device)
    if positions.ndim != 1 or positions.numel() != active_k.shape[2]:
        raise ValueError("active position count does not match active K/V")
    start, total = int(visual_start), int(total_tokens)
    if start < 0 or start + n_image > total:
        raise ValueError("image span outside full cache")
    if positions.numel() and (
            int(positions.min()) < 0 or int(positions.max()) >= total):
        raise ValueError("active position outside full cache")

    shape = (1, image_k.shape[1], total, image_k.shape[3])
    full_k = torch.full(shape, float(dummy_value), dtype=active_k.dtype,
                         device=active_k.device)
    full_v = torch.full_like(full_k, float(dummy_value))
    image_k = image_k.to(device=active_k.device, dtype=active_k.dtype)
    image_v = image_v.to(device=active_v.device, dtype=active_v.dtype)
    full_k[:, :, start:start + n_image, :].copy_(image_k)
    full_v[:, :, start:start + n_image, :].copy_(image_v)
    full_k.index_copy_(2, positions, active_k)
    full_v.index_copy_(2, positions, active_v)

    covered = torch.zeros(total, dtype=torch.bool, device=positions.device)
    covered[start:start + n_image] = True
    if positions.numel():
        covered[positions] = True
    if not bool(covered.all()):
        missing = (~covered).nonzero(as_tuple=True)[0].tolist()
        raise AssertionError(f"dummy K/V would be exposed at rows {missing}")
    return full_k, full_v


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def relocate_post_rope_keys(keys: torch.Tensor,
                            source_cos: torch.Tensor,
                            source_sin: torch.Tensor,
                            target_cos: torch.Tensor,
                            target_sin: torch.Tensor) -> torch.Tensor:
    """Phase-relocate post-RoPE keys without mutating the source tensor.

    The operation implements ``a_t R(target) R(source)^-1 / a_s`` and also
    handles a rotary implementation whose source/target attention scaling is
    not exactly one.  This corrects positional phase only; it cannot correct a
    hidden-state change caused by different preceding content.
    """
    if keys.ndim != 4:
        raise ValueError("keys must be [batch, heads, sequence, head_dim]")

    def expanded(value: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(value, dtype=keys.dtype, device=keys.device)
        if value.ndim == 3:
            value = value.unsqueeze(1)
        if value.ndim != 4:
            raise ValueError("RoPE cos/sin must be [B,S,D] or [B,1,S,D]")
        return value

    sc, ss = expanded(source_cos), expanded(source_sin)
    tc, ts = expanded(target_cos), expanded(target_sin)
    if any(value.shape[-2:] != keys.shape[-2:]
           for value in (sc, ss, tc, ts)):
        raise ValueError("RoPE tensors do not match key sequence/head dim")
    denominator = (sc.square() + ss.square()).clamp_min(
        torch.finfo(keys.dtype).tiny)
    delta_cos = (tc * sc + ts * ss) / denominator
    delta_sin = (ts * sc - tc * ss) / denominator
    return keys * delta_cos + _rotate_half(keys) * delta_sin


def seed_dynamic_cache(
        layer_keys_values: Iterable[tuple[torch.Tensor, torch.Tensor]],
        config=None):
    """Create a decode cache with one exact, full-length tensor per layer."""
    from transformers import DynamicCache

    layers = list(layer_keys_values)
    if not layers:
        raise ValueError("cannot seed an empty decode cache")
    length = None
    for index, (key, value) in enumerate(layers):
        if key.shape != value.shape or key.ndim != 4:
            raise ValueError(f"invalid layer {index} K/V")
        if length is None:
            length = int(key.shape[-2])
        elif int(key.shape[-2]) != length:
            raise ValueError("cache length varies by layer")
    cache = DynamicCache(ddp_cache_data=layers, config=config)
    assert all(int(layer.keys.shape[-2]) == length for layer in cache.layers)
    return cache


# ---------------------------------------------------------------------------
# Dedicated raw SSD store and reader

class MPICRawReader:
    """Attributed raw reads for canonical image KV and visual inputs."""

    def __init__(self, store_dir: Path, meta: Mapping[str, Any],
                 drop_cache: bool = True):
        self.dir = Path(store_dir)
        self.meta = dict(meta)
        self.drop_cache = bool(drop_cache)
        self.np_dtype = _NP_DTYPES[self.meta["dtype"]]
        self.itemsize = np.dtype(self.np_dtype).itemsize
        self._fds: dict[tuple[Any, ...], int] = {}

    def _fd(self, *parts: Any) -> int:
        key = tuple(parts)
        if key not in self._fds:
            if parts[0] == "visual":
                path = self.dir / "visual_input.bin"
            else:
                layer, kind = int(parts[0]), str(parts[1])
                path = self.dir / f"layer_{layer:02d}" / f"{kind}.bin"
            self._fds[key] = os.open(path, os.O_RDONLY)
        return self._fds[key]

    @staticmethod
    def _pread(fd: int, offset: int, length: int) -> tuple[bytes, float]:
        started = time.perf_counter()
        payload = os.pread(fd, int(length), int(offset))
        elapsed = time.perf_counter() - started
        if len(payload) != int(length):
            raise IOError(f"short pread: {len(payload)} != {length}")
        return payload, elapsed

    def read_visual_inputs(self, k: int, counter: IOCounter | None = None) \
            -> torch.Tensor:
        count = min(max(int(k), 0), int(self.meta["v_token_num"]))
        width = int(self.meta["hidden_size"])
        if count == 0:
            return torch.empty((0, width), dtype=_TORCH_DTYPES[
                self.meta["dtype"]])
        nbytes = count * width * self.itemsize
        payload, elapsed = self._pread(self._fd("visual"), 0, nbytes)
        if counter is not None:
            counter.record("embedding", len(payload), elapsed, preads=1,
                           units=count)
        array = np.frombuffer(payload, dtype=self.np_dtype).reshape(
            count, width)
        return torch.from_numpy(array.copy())

    def read_reused_layer(self, layer: int, kind: str, k_recompute: int,
                          counter: IOCounter | None = None) \
            -> tuple[torch.Tensor, torch.Tensor]:
        """Read the chunk-aligned range covering logical reused rows ``[k,N)``."""
        if kind not in ("k", "v"):
            raise ValueError("kind must be k or v")
        n = int(self.meta["v_token_num"])
        heads = int(self.meta["num_heads"])
        dim = int(self.meta["head_dim"])
        k = min(max(int(k_recompute), 0), n)
        if k == n:
            return (torch.empty(0, dtype=torch.long),
                    torch.empty((0, heads, dim),
                                dtype=_TORCH_DTYPES[self.meta["dtype"]]))
        chunk = int(self.meta["chunk_size"])
        row_start = (k // chunk) * chunk
        row_bytes = heads * dim * self.itemsize
        offset = row_start * row_bytes
        length = (n - row_start) * row_bytes
        payload, elapsed = self._pread(
            self._fd(int(layer), kind), offset, length)
        units = math.ceil((n - row_start) / chunk)
        if counter is not None:
            counter.record(f"kv_{kind}", len(payload), elapsed, preads=1,
                           units=units)
        array = np.frombuffer(payload, dtype=self.np_dtype).reshape(
            n - row_start, heads, dim)
        rows = torch.arange(row_start, n, dtype=torch.long)
        return rows, torch.from_numpy(array.copy())

    def drop_all(self) -> None:
        """Best-effort page-cache conditioning, deliberately outside TTFT."""
        for path in sorted(self.dir.rglob("*.bin")):
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)

    def close(self) -> None:
        for fd in self._fds.values():
            if self.drop_cache:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        self._fds.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class MPICContext:
    """One validated, canonical, SSD-resident MPIC image payload."""

    def __init__(self, store_dir: Path, device, drop_cache: bool = True,
                 runner=None):
        self.dir = Path(store_dir)
        with (self.dir / "meta.json").open(encoding="utf-8") as handle:
            self.meta = json.load(handle)
        with (self.dir / "integrity.json").open(encoding="utf-8") as handle:
            self.integrity = json.load(handle)
        self.device = torch.device(device)
        self.reader = MPICRawReader(self.dir, self.meta,
                                    drop_cache=drop_cache)
        self.metadata_resident_bytes = (
            (self.dir / "meta.json").stat().st_size
            + (self.dir / "integrity.json").stat().st_size)
        self.validate()
        if runner is not None:
            self.validate_runtime(runner)

    def validate(self) -> None:
        m = self.meta
        required = {
            "schema_version": STORE_SCHEMA,
            "physical_layout": "canonical_raster",
            "mpic_compatible": True,
            "reordered": False,
            "layout_uses_dataset_question": False,
            "visual_kv_source": "turn1_captured_past_key_values",
            "visual_input_source": "same_turn1_decoder_layer0_input",
            "turn1_normal_inference": True,
            "separate_vision_forward": False,
            "separate_prefix_forward": False,
            "cached_key_representation": "post_rope",
        }
        for key, expected in required.items():
            if m.get(key) != expected:
                raise ValueError(
                    f"invalid MPIC store metadata {key}: {m.get(key)!r}")
        n, heads, dim = (int(m["v_token_num"]), int(m["num_heads"]),
                         int(m["head_dim"]))
        itemsize = np.dtype(_NP_DTYPES[m["dtype"]]).itemsize
        expected_layer_bytes = n * heads * dim * itemsize
        for layer in range(int(m["num_layers"])):
            for kind in ("k", "v"):
                path = self.dir / f"layer_{layer:02d}" / f"{kind}.bin"
                if path.stat().st_size != expected_layer_bytes:
                    raise ValueError(f"MPIC payload size mismatch: {path}")
        expected_input = n * int(m["hidden_size"]) * itemsize
        if (self.dir / "visual_input.bin").stat().st_size != expected_input:
            raise ValueError("visual-input sidecar size mismatch")
        prefix = [int(value) for value in m["prefix_input_ids"]]
        if len(prefix) != int(m["prefix_len"]):
            raise ValueError("source prefix IDs have wrong length")
        start = int(m["v_token_start"])
        if prefix[start:] != [int(m["image_token_id"])] * n:
            raise ValueError("source image span is not canonical placeholders")
        if self.integrity.get("payload_sample_sha256") != m.get(
                "payload_sample_sha256"):
            raise ValueError("metadata/integrity payload hash mismatch")
        expected_payloads = [
            f"layer_{layer:02d}/{kind}.bin"
            for layer in range(int(m["num_layers"]))
            for kind in ("k", "v")
        ] + ["visual_input.bin"]
        recorded_payloads = self.integrity.get("payload_files")
        if recorded_payloads != expected_payloads:
            raise ValueError("integrity payload inventory mismatch")
        for relative in recorded_payloads:
            pure = Path(relative)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"unsafe integrity payload path: {relative!r}")
            payload = self.dir / pure
            if payload.is_symlink() or not payload.is_file():
                raise ValueError(f"invalid integrity payload file: {payload}")
        # Do not trust two copies of the same recorded string.  Recompute the
        # documented first/last-block fingerprint from every payload when a
        # context is opened, so corruption and incomplete resumed stores fail
        # before any request is measured.
        observed_hash = _payload_sample_hash(self.dir, recorded_payloads)
        if observed_hash != m.get("payload_sample_sha256"):
            raise ValueError("MPIC payload sample hash mismatch")
        self._payload_paths = tuple(recorded_payloads)
        self._validated_payload_hash = observed_hash

    @property
    def source_payload_hash(self) -> str:
        """Recompute the live payload fingerprint, rather than echo metadata."""
        return _payload_sample_hash(self.dir, self._payload_paths)

    @property
    def validated_payload_hash(self) -> str:
        """Fingerprint independently verified when this context was opened."""
        return str(self._validated_payload_hash)

    def close(self) -> None:
        self.reader.close()

    def validate_runtime(self, runner) -> None:
        """Reject a store produced by a different model/runtime identity."""
        current = _runtime_provenance(runner)
        for key in (
            "model", "model_revision", "model_config_sha256",
            "processor_class", "image_processor_class", "tokenizer_class",
            "tokenizer_vocab_size", "transformers_version", "torch_version",
            "attention_implementation", "load_4bit",
            "quantization_config_sha256",
        ):
            if self.meta.get(key) != current.get(key):
                raise ValueError(
                    f"MPIC store/runtime mismatch for {key}: "
                    f"{self.meta.get(key)!r} != {current.get(key)!r}")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, indent=1, ensure_ascii=False,
                      allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_timed(path: Path, payload: bytes) -> float:
    started = time.perf_counter()
    path.write_bytes(payload)
    return (time.perf_counter() - started) * 1e3


def _fsync_tree(root: Path) -> tuple[float, int, int]:
    started = time.perf_counter()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    directories = [root] + sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda value: len(value.parts), reverse=True)
    for path in directories:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return ((time.perf_counter() - started) * 1e3, len(files),
            len(directories))


def _payload_sample_hash(root: Path, relative_paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        path = root / relative
        size = path.stat().st_size
        with path.open("rb") as handle:
            head = handle.read(min(size, 4096))
            if size > 4096:
                handle.seek(max(0, size - 4096))
                tail = handle.read(4096)
            else:
                tail = b""
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(size.to_bytes(8, "big"))
        digest.update(len(head).to_bytes(4, "big"))
        digest.update(head)
        digest.update(len(tail).to_bytes(4, "big"))
        digest.update(tail)
    return digest.hexdigest()


@torch.no_grad()
def persist_captured_mpic_prefix(
        runner, captured_past_key_values,
        expanded_input_ids: torch.Tensor,
        image_size: Sequence[int] | torch.Tensor,
        visual_input_states: torch.Tensor, out_dir: Path, *,
        image_id: str | int,
        hidden_capture_stats: DecoderVisualHiddenCapture,
        model_id: str | None = None, chunk_size: int = CHUNK_SIZE,
        image_input_sha256: str | None = None,
        extra_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Atomically persist MPIC image KV and layer-0 visual inputs from Turn 1.

    This helper performs no model or vision forward.  The live capture object
    and captured cache must come from the same normal request.
    """
    total_started = time.perf_counter()
    destination = Path(out_dir)
    if destination.name in ("", ".", ".."):
        raise ValueError("unsafe MPIC store destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise FileExistsError(errno.EEXIST, "refusing to overwrite store",
                              destination)
    if not isinstance(hidden_capture_stats, DecoderVisualHiddenCapture):
        raise TypeError("MPIC persistence requires the live hidden capture")
    capture = hidden_capture_stats
    if capture.runner is not runner:
        raise ValueError("hidden capture belongs to another runner")
    captured_visual = capture.result_cpu()
    supplied_visual = torch.as_tensor(visual_input_states).detach().cpu()
    if not torch.equal(captured_visual.float(), supplied_visual.float()):
        raise ValueError("visual inputs differ from same-request capture")
    capture_doc = capture.stats()
    for key, expected in {
        "visual_hidden_capture_count": 1,
        "separate_model_forward_count": 0,
        "separate_vision_forward_count": 0,
        "capture_source": "same_turn1_normal_multimodal_prefill",
    }.items():
        if capture_doc.get(key) != expected:
            raise ValueError(f"invalid visual-input provenance: {key}")

    ids = torch.as_tensor(expanded_input_ids).detach().cpu()
    if ids.ndim == 2:
        if ids.shape[0] != 1:
            raise ValueError("MPIC persistence supports batch size one")
        ids = ids[0]
    if ids.ndim != 1:
        raise ValueError("expanded input IDs must be one-dimensional")
    v_start, v_num = runner.visual_span(ids)
    prefix_len = int(v_start + v_num)
    prefix_ids = [int(value) for value in ids[:prefix_len].tolist()]
    if torch.is_tensor(image_size):
        canonical_image_size = [
            int(value) for value in image_size.detach().cpu().reshape(-1)]
    else:
        canonical_image_size = [int(value) for value in image_size]
    if len(canonical_image_size) != 2:
        raise ValueError("image_size must contain height and width")
    base, hi_h, hi_w, newline_idx = runner.anyres_layout(
        canonical_image_size, int(v_num))
    visual = supplied_visual.to(torch.float16).contiguous()
    if visual.ndim != 2 or visual.shape[0] != v_num:
        raise ValueError("visual input sidecar has wrong shape")
    if int(capture_doc["visual_start"]) != v_start or int(
            capture_doc["visual_tokens"]) != v_num:
        raise ValueError("hidden-capture span differs from expanded prompt")
    if not bool(torch.isfinite(visual).all()):
        raise ValueError("visual input contains NaN/Inf")

    layers = cache_layers(captured_past_key_values)
    if not layers:
        raise ValueError("captured cache is empty")
    for index, (key, value) in enumerate(layers):
        if key.shape != value.shape or key.ndim != 4:
            raise ValueError(f"invalid captured layer {index}")
        if key.shape[0] != 1 or key.shape[2] < prefix_len:
            raise ValueError(f"captured layer {index} is too short")
    heads, head_dim = int(layers[0][0].shape[1]), int(
        layers[0][0].shape[3])

    source_positions = list(range(v_start, v_start + v_num))
    provenance = _runtime_provenance(runner)
    if model_id is not None and str(model_id) != provenance["model"]:
        raise ValueError("explicit model_id disagrees with the active runner")
    metadata = {
        "schema_version": STORE_SCHEMA,
        "image_id": str(image_id),
        **provenance,
        "dtype": "float16",
        "v_token_start": int(v_start), "v_token_num": int(v_num),
        "prefix_len": int(prefix_len), "num_layers": len(layers),
        "num_heads": heads, "head_dim": head_dim,
        "hidden_size": int(visual.shape[1]),
        "chunk_size": int(chunk_size),
        "n_chunks_per_layer": math.ceil(v_num / int(chunk_size)),
        "image_token_id": int(runner.image_token_id),
        "prefix_input_ids": prefix_ids,
        "source_prefix_ids_sha256": stable_json_sha256(prefix_ids),
        "source_pre_image_ids_sha256": stable_json_sha256(
            prefix_ids[:v_start]),
        "source_positions": source_positions,
        "source_position_hash": stable_json_sha256(source_positions),
        "newline_idx": [int(value) for value in newline_idx],
        "base_grid": int(base), "hires_grid": [int(hi_h), int(hi_w)],
        "physical_layout": "canonical_raster",
        "mpic_compatible": True, "reordered": False,
        "order_is_per_layer": False,
        "layout_uses_dataset_question": False,
        "query_dependent_selection": False,
        "visual_kv_source": "turn1_captured_past_key_values",
        "visual_input_source": "same_turn1_decoder_layer0_input",
        "turn1_normal_inference": True,
        "turn1_question_present_after_image": True,
        "image_kv_causally_excludes_turn1_question": True,
        "separate_vision_forward": False,
        "separate_prefix_forward": False,
        "cached_key_representation": "post_rope",
        "system_text_policy": "recompute_all_text",
        "separator_policy": "ordinary_canonical_image_rows",
        "hidden_capture": capture_doc,
        "image_input_sha256": image_input_sha256,
        "bytes_visual_kv": int(
            len(layers) * 2 * v_num * heads * head_dim * 2),
        "bytes_visual_input": int(visual.numel() * visual.element_size()),
        "bytes_separator_sidecar": 0,
        "bytes_probe_sidecar": 0,
    }
    if extra_metadata:
        overlap = set(metadata) & set(extra_metadata)
        if overlap:
            raise ValueError("extra metadata overrides invariants: "
                             + ", ".join(sorted(overlap)))
        metadata.update(dict(extra_metadata))

    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.staging-", dir=destination.parent))
    published = False
    write_ms = materialize_ms = 0.0
    payload_paths: list[str] = []
    try:
        for index, (key, value) in enumerate(layers):
            layer_dir = staging / f"layer_{index:02d}"
            layer_dir.mkdir()
            for kind, tensor in (("k", key), ("v", value)):
                started = time.perf_counter()
                block = tensor[
                    0, :, v_start:prefix_len, :].permute(
                        1, 0, 2).to(torch.float16).contiguous().cpu()
                materialize_ms += (time.perf_counter() - started) * 1e3
                relative = f"layer_{index:02d}/{kind}.bin"
                write_ms += _write_bytes_timed(
                    staging / relative, block.numpy().tobytes())
                payload_paths.append(relative)
        write_ms += _write_bytes_timed(
            staging / "visual_input.bin", visual.numpy().tobytes())
        payload_paths.append("visual_input.bin")
        sample_hash = _payload_sample_hash(staging, payload_paths)
        metadata["payload_sample_sha256"] = sample_hash
        metadata["payload_sample_hash_policy"] = (
            "sorted path,size,first4096,last4096 for every KV/input payload")
        write_ms += _write_bytes_timed(
            staging / "meta.json", _json_bytes(metadata))
        integrity = {
            "schema_version": STORE_SCHEMA,
            "payload_sample_sha256": sample_hash,
            "payload_files": payload_paths,
            "visual_input_sha256": _sha256_bytes(
                visual.numpy().tobytes()),
            "source_prefix_ids_sha256": metadata[
                "source_prefix_ids_sha256"],
        }
        write_ms += _write_bytes_timed(
            staging / "integrity.json", _json_bytes(integrity))
        fsync_ms, synced_files, synced_dirs = _fsync_tree(staging)

        from mmimpress.piggyback import _fsync_directory, _rename_noreplace
        started = time.perf_counter()
        _rename_noreplace(staging, destination)
        rename_ms = (time.perf_counter() - started) * 1e3
        published = True
        started = time.perf_counter()
        _fsync_directory(destination.parent)
        parent_fsync_ms = (time.perf_counter() - started) * 1e3
        persistence_ms = (time.perf_counter() - total_started) * 1e3
        file_sizes = {
            path.relative_to(destination).as_posix(): path.stat().st_size
            for path in sorted(destination.rglob("*")) if path.is_file()
        }
        return {
            "store_dir": str(destination.resolve()),
            "image_id": str(image_id), "meta": metadata,
            "integrity": integrity, "file_sizes": file_sizes,
            "bytes": {
                "visual_kv": int(metadata["bytes_visual_kv"]),
                "visual_input": int(metadata["bytes_visual_input"]),
                "metadata": int(file_sizes["meta.json"]
                                + file_sizes["integrity.json"]),
                "total": int(sum(file_sizes.values())),
            },
            "timing_ms": {
                "kv_materialize_ms": float(materialize_ms),
                "visual_input_capture_materialize_ms": float(
                    capture_doc["materialize_ms"]),
                "ssd_write_ms": float(write_ms),
                "fsync_ms": float(fsync_ms),
                "atomic_rename_ms": float(rename_ms),
                "parent_fsync_ms": float(parent_fsync_ms),
                "persist_ms": float(persistence_ms),
                "provisioning_post_response_ms": float(
                    persistence_ms + capture_doc["materialize_ms"]),
            },
            "timing_semantics": {
                "persist_ms": (
                    "starts when persist_captured_mpic_prefix is called and "
                    "therefore excludes visual-input D2H performed when the "
                    "Turn-1 capture context exits"),
                "provisioning_post_response_ms": (
                    "visual-input capture materialization plus persist_ms; "
                    "the capture materialization also remains recorded in "
                    "meta.hidden_capture.materialize_ms"),
            },
            "durability": {
                "same_filesystem_staging": True,
                "atomic_no_clobber": True,
                "files_fsynced": synced_files,
                "directories_fsynced": synced_dirs,
                "parent_fsynced": True,
            },
        }
    finally:
        if not published and os.path.lexists(staging):
            shutil.rmtree(staging)


# ---------------------------------------------------------------------------
# Selective layer traversal and generation

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _clear_foreign_attention_bias() -> None:
    """Prevent the legacy global hook state from crossing method boundaries."""
    module = sys.modules.get("mmimpress.serve")
    if module is not None and hasattr(module, "BIAS"):
        module.BIAS.clear()


def expand_prompt_without_pixels(runner, prompt: str, visual_tokens: int) \
        -> tuple[torch.Tensor, int]:
    """Tokenize one placeholder and expand it to the stored image length."""
    ids = runner.processor.tokenizer(
        prompt, return_tensors="pt").input_ids[0].detach().cpu()
    locations = (ids == int(runner.image_token_id)).nonzero(as_tuple=True)[0]
    if locations.numel() != 1:
        raise ValueError("MPIC supports exactly one image placeholder")
    start = int(locations[0])
    expanded = torch.cat((
        ids[:start],
        torch.full((int(visual_tokens),), int(runner.image_token_id),
                   dtype=ids.dtype),
        ids[start + 1:],
    ))
    return expanded, start


class MPICSelectivePrefill:
    """Run one active-row traversal and return first-token logits/full cache."""

    def __init__(self, runner, context: MPICContext, k_recompute: int = 32,
                 dummy_value: float = 0.0):
        self.runner = runner
        self.context = context
        self.k_recompute = int(k_recompute)
        self.dummy_value = float(dummy_value)

    @torch.no_grad()
    def run(self, expanded_ids: torch.Tensor, target_visual_start: int,
            counter: IOCounter) -> tuple[torch.Tensor, list, ActiveRowPlan,
                                         dict[str, Any]]:
        runner, ctx = self.runner, self.context
        model = runner.model
        device = torch.device(model.device)
        lm = model.model.language_model
        meta = ctx.meta
        n_image = int(meta["v_token_num"])
        total = int(expanded_ids.numel())
        plan = build_active_row_plan(
            total, int(target_visual_start), n_image, self.k_recompute)
        interval_started = time.perf_counter()
        visual_inputs = ctx.reader.read_visual_inputs(
            plan.k_recompute, counter)
        embedding_read_done = time.perf_counter()
        _sync(device)
        h2d_started = time.perf_counter()
        active_pos = plan.active_positions.to(device)
        active_ids = expanded_ids.index_select(
            0, plan.active_positions).to(device)
        visual_device = visual_inputs.to(
            device=device, dtype=model.get_input_embeddings().weight.dtype)
        _sync(device)
        h2d_ms = (time.perf_counter() - h2d_started) * 1e3

        _sync(device)
        embedding_started = time.perf_counter()
        active_hidden = model.get_input_embeddings()(
            active_ids.unsqueeze(0))
        if plan.k_recompute:
            if visual_device.dtype != active_hidden.dtype:
                visual_device = visual_device.to(dtype=active_hidden.dtype)
            image_mask = ((active_pos >= plan.visual_start)
                          & (active_pos < plan.visual_start
                             + plan.k_recompute))
            if int(image_mask.sum()) != plan.k_recompute:
                raise AssertionError("active image-row mapping mismatch")
            active_hidden[:, image_mask, :] = visual_device.unsqueeze(0)
        _sync(device)
        input_embedding_ms = (
            time.perf_counter() - embedding_started) * 1e3

        source_start = int(meta["v_token_start"])
        source_positions = torch.arange(
            source_start, source_start + n_image,
            dtype=torch.long, device=device).unsqueeze(0)
        target_positions = torch.arange(
            plan.visual_start, plan.visual_start + n_image,
            dtype=torch.long, device=device).unsqueeze(0)
        same_positions = bool(torch.equal(source_positions,
                                          target_positions))
        position_policy = (POSITION_POLICY_SAME if same_positions
                           else POSITION_POLICY_SHIFTED)
        causal_mask = build_causal_mask(
            active_pos, total, active_hidden.dtype)

        layer_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        layer_active: list[int] = []
        layer_image: list[int] = []
        layer_text: list[int] = []
        layer_keys: list[int] = []
        layer_valid_image: list[int] = []
        kv_h2d_ms = 0.0
        cache_assembly_ms = 0.0
        position_ms = 0.0

        from transformers.models.llama.modeling_llama import (
            apply_rotary_pos_emb, repeat_kv)

        for layer_index, layer in enumerate(runner.layers):
            read_rows_k, cached_k_cpu = ctx.reader.read_reused_layer(
                layer_index, "k", plan.k_recompute, counter)
            read_rows_v, cached_v_cpu = ctx.reader.read_reused_layer(
                layer_index, "v", plan.k_recompute, counter)
            if not torch.equal(read_rows_k, read_rows_v):
                raise AssertionError("K/V SSD ranges disagree")

            _sync(device)
            started = time.perf_counter()
            if read_rows_k.numel():
                rows_device = read_rows_k.to(device)
                k_device = cached_k_cpu.to(
                    device=device, dtype=active_hidden.dtype).permute(
                        1, 0, 2).unsqueeze(0)
                v_device = cached_v_cpu.to(
                    device=device, dtype=active_hidden.dtype).permute(
                        1, 0, 2).unsqueeze(0)
            else:
                rows_device = torch.empty(0, dtype=torch.long, device=device)
                k_device = v_device = None
            _sync(device)
            kv_h2d_ms += (time.perf_counter() - started) * 1e3

            _sync(device)
            started = time.perf_counter()
            cached_k = torch.full(
                (1, int(meta["num_heads"]), n_image,
                 int(meta["head_dim"])), self.dummy_value,
                device=device, dtype=active_hidden.dtype)
            cached_v = torch.full_like(cached_k, self.dummy_value)
            if read_rows_k.numel():
                cached_k.index_copy_(2, rows_device, k_device)
                cached_v.index_copy_(2, rows_device, v_device)
            _sync(device)
            cache_assembly_ms += (time.perf_counter() - started) * 1e3

            if not same_positions:
                _sync(device)
                started = time.perf_counter()
                source_cos, source_sin = lm.rotary_emb(
                    cached_k, source_positions)
                target_cos, target_sin = lm.rotary_emb(
                    cached_k, target_positions)
                cached_k = relocate_post_rope_keys(
                    cached_k, source_cos, source_sin,
                    target_cos, target_sin)
                _sync(device)
                position_ms += (time.perf_counter() - started) * 1e3

            residual = active_hidden
            normalized = layer.input_layernorm(active_hidden)
            attention = layer.self_attn
            input_shape = normalized.shape[:-1]
            hidden_shape = (*input_shape, -1, attention.head_dim)
            query = attention.q_proj(normalized).view(
                hidden_shape).transpose(1, 2)
            active_k = attention.k_proj(normalized).view(
                hidden_shape).transpose(1, 2)
            active_v = attention.v_proj(normalized).view(
                hidden_shape).transpose(1, 2)
            cos, sin = lm.rotary_emb(
                normalized, active_pos.unsqueeze(0))
            query, active_k = apply_rotary_pos_emb(
                query, active_k, cos, sin)

            _sync(device)
            started = time.perf_counter()
            full_k, full_v = assemble_linked_kv(
                cached_k, cached_v, active_k, active_v, active_pos,
                plan.visual_start, total, self.dummy_value)
            _sync(device)
            cache_assembly_ms += (time.perf_counter() - started) * 1e3

            key_for_attention = repeat_kv(
                full_k, attention.num_key_value_groups)
            value_for_attention = repeat_kv(
                full_v, attention.num_key_value_groups)
            weights = torch.matmul(
                query, key_for_attention.transpose(2, 3)) * attention.scaling
            weights = weights + causal_mask
            weights = F.softmax(
                weights, dim=-1, dtype=torch.float32).to(query.dtype)
            output = torch.matmul(weights, value_for_attention)
            output = output.transpose(1, 2).contiguous().reshape(
                *input_shape, -1)
            output = attention.o_proj(output)
            active_hidden = residual + output
            residual = active_hidden
            active_hidden = layer.post_attention_layernorm(active_hidden)
            active_hidden = residual + layer.mlp(active_hidden)

            layer_cache.append((full_k, full_v))
            layer_active.append(plan.n_active)
            layer_image.append(plan.k_recompute)
            layer_text.append(int(plan.text_positions.numel()))
            layer_keys.append(total)
            layer_valid_image.append(n_image)

        active_hidden = lm.norm(active_hidden)
        if int(active_pos[-1]) != total - 1:
            raise AssertionError("final prompt row is not active")
        logits = model.lm_head(active_hidden[:, -1:, :])
        _sync(device)
        interval_ms = (time.perf_counter() - interval_started) * 1e3
        stats = {
            "active_rows_per_layer": layer_active,
            "recomputed_image_rows_per_layer": layer_image,
            "recomputed_text_rows_per_layer": layer_text,
            "attention_key_length_per_layer": layer_keys,
            "valid_image_key_count_per_layer": layer_valid_image,
            "decoder_prefill_pass_count": 1,
            "vision_forward_count": 0,
            "embedding_read_ms": (
                embedding_read_done - interval_started) * 1e3,
            "h2d_ms": float(h2d_ms + kv_h2d_ms),
            "input_embedding_and_assignment_ms": float(input_embedding_ms),
            "h2d_timing_semantics": (
                "synchronized host-to-device transfers for active IDs, "
                "visual-input rows, cached K/V rows, and row indices; cache "
                "allocation/indexed assembly is excluded"),
            "cache_assembly_ms": float(cache_assembly_ms),
            "position_processing_ms": float(position_ms),
            "selective_prefill_interval_ms": float(interval_ms),
            "selective_prefill_interval_semantics": (
                "inclusive of embedding/KV pread, H2D, position processing, "
                "cache assembly, all active-row decoder layers, final norm "
                "and LM head; not pure GPU compute"),
            "position_handling_policy": position_policy,
            "same_source_target_positions": same_positions,
            "source_positions": source_positions[0].tolist(),
            "target_positions": target_positions[0].tolist(),
            "source_position_hash": stable_json_sha256(
                source_positions[0].tolist()),
            "target_position_hash": stable_json_sha256(
                target_positions[0].tolist()),
        }
        return logits, layer_cache, plan, stats


class MPICServer:
    """Request-level MPIC-32 generation with inclusive TTFT accounting."""

    def __init__(self, runner, k_recompute: int = 32,
                 max_new_tokens: int = 16):
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.runner = runner
        self.k_recompute = int(k_recompute)
        self.max_new_tokens = int(max_new_tokens)

    @torch.no_grad()
    def request(self, context: MPICContext, question: str | None = None, *,
                prompt_text: str | None = None, cold: bool = True,
                k_recompute: int | None = None,
                dummy_value: float = 0.0) -> dict[str, Any]:
        if (question is None) == (prompt_text is None):
            raise ValueError("provide exactly one of question or prompt_text")
        runner = self.runner
        model = runner.model
        device = torch.device(model.device)
        tokenizer = runner.processor.tokenizer
        k_requested = (self.k_recompute if k_recompute is None
                       else int(k_recompute))

        # Payload bytes were independently fingerprinted when the image
        # context opened.  Do not reread sample blocks immediately before each
        # timed SSD request: POSIX_DONTNEED can evict OS pages but cannot prove
        # eviction from the drive/controller cache.  The serving runner audits
        # the live payload again at the image boundary after all requests.
        source_hash_before = context.validated_payload_hash
        integrity_before_ms = 0.0
        conditioning_started = time.perf_counter()
        if cold:
            context.reader.drop_all()
        conditioning_done = time.perf_counter()
        _clear_foreign_attention_bias()
        _sync(device)
        request_started = time.perf_counter()
        prompt = (prompt_text if prompt_text is not None
                  else runner.prompt(str(question)))
        expanded_ids, target_start = expand_prompt_without_pixels(
            runner, prompt, int(context.meta["v_token_num"]))
        prompt_ready = time.perf_counter()
        counter = IOCounter()
        engine = MPICSelectivePrefill(
            runner, context, k_requested, dummy_value=dummy_value)
        logits, layer_cache, plan, stats = engine.run(
            expanded_ids, target_start, counter)

        _sync(device)
        cache_started = time.perf_counter()
        cache = seed_dynamic_cache(layer_cache, config=model.config)
        del layer_cache
        _sync(device)
        prefill_cache_lengths = [
            int(layer.keys.shape[-2]) for layer in cache.layers]
        decode_cache_assembly_ms = (
            time.perf_counter() - cache_started) * 1e3
        first = int(logits[0, -1].argmax())
        _sync(device)
        first_token_at = time.perf_counter()

        tokens = [first]
        current = int(expanded_ids.numel())
        while (tokens[-1] != tokenizer.eos_token_id
               and len(tokens) < self.max_new_tokens):
            cache_position = torch.tensor([current], device=device)
            output = model(
                input_ids=torch.tensor([[tokens[-1]]], device=device),
                attention_mask=torch.ones(
                    1, current + 1, dtype=torch.long, device=device),
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position,
                past_key_values=cache, use_cache=True)
            tokens.append(int(output.logits[0, -1].argmax()))
            current += 1
        _sync(device)
        finished_at = time.perf_counter()
        answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        final_cache_lengths = [
            int(layer.keys.shape[-2]) for layer in cache.layers]
        io = counter.summary()
        kv_bytes = sum(int(io["per_kind"].get(kind, {}).get("bytes", 0))
                       for kind in ("kv_k", "kv_v"))
        embedding_bytes = int(io["per_kind"].get(
            "embedding", {}).get("bytes", 0))
        kv_ms = sum(float(io["per_kind"].get(kind, {}).get("seconds", 0.0))
                    for kind in ("kv_k", "kv_v")) * 1e3
        embedding_ms = float(io["per_kind"].get(
            "embedding", {}).get("seconds", 0.0)) * 1e3
        n_image = int(context.meta["v_token_num"])
        source_positions = [int(value) for value in context.meta[
            "source_positions"]]
        target_positions = list(range(
            target_start, target_start + n_image))
        source_pre = [int(value) for value in context.meta[
            "prefix_input_ids"][:int(context.meta["v_token_start"])]]
        target_pre = [int(value) for value in expanded_ids[:target_start]]
        same_context = (source_pre == target_pre
                        and source_positions == target_positions)
        full_visual_bytes = int(context.meta["bytes_visual_kv"])
        k = plan.k_recompute
        stats["cache_assembly_ms"] += decode_cache_assembly_ms
        stats.update({
            "method": PAPER_LABEL, "method_id": METHOD_ID,
            "k_recompute": k, "n_image_tokens": n_image,
            "n_recomputed_image_tokens": k,
            "n_reused_image_tokens": n_image - k,
            "n_recomputed_text_tokens": int(plan.text_positions.numel()),
            "recomputed_image_token_ratio": k / n_image,
            "reused_image_token_ratio": (n_image - k) / n_image,
            "retained_image_context_ratio": 1.0,
            "actual_kv_read_ratio": (
                kv_bytes / full_visual_bytes if full_visual_bytes else 0.0),
            "actual_ssd_read_ratio": (
                (kv_bytes + embedding_bytes) / full_visual_bytes
                if full_visual_bytes else 0.0),
            "actual_ssd_read_ratio_denominator": (
                "full canonical visual K/V bytes; numerator includes reused "
                "K/V plus recomputation-input embedding sidecar"),
            "selected_image_local_rows": list(range(k)),
            "selected_image_logical_rows": plan.image_positions.tolist(),
            "source_position_hash": context.meta["source_position_hash"],
            "target_position_hash": stable_json_sha256(target_positions),
            "same_source_target_context": same_context,
            "ssd_kv_bytes": kv_bytes,
            "ssd_embedding_bytes": embedding_bytes,
            "ssd_separator_bytes": 0,
            "ssd_metadata_bytes": 0,
            "ssd_total_bytes": kv_bytes + embedding_bytes,
            "metadata_resident_bytes": context.metadata_resident_bytes,
            "metadata_residency_policy": (
                "validated meta/integrity retained in host memory while "
                "image context is open; zero per-request metadata read"),
            "pread_count": int(io["preads"]),
            "kv_read_ms": float(kv_ms),
            "embedding_read_ms": float(embedding_ms),
            "decode_cache_assembly_ms": float(decode_cache_assembly_ms),
            "prompt_and_tokenization_ms": (
                prompt_ready - request_started) * 1e3,
            "ttft_ms": (first_token_at - request_started) * 1e3,
            "model_generation_e2e_ms": (
                finished_at - request_started) * 1e3,
            "decode_ms": (finished_at - first_token_at) * 1e3,
            "prediction": answer, "first_token_id": first,
            "generated_token_ids": tokens,
            "generated_token_count": len(tokens),
            "prefill_cache_lengths": prefill_cache_lengths,
            "final_cache_lengths": final_cache_lengths,
            "expected_final_cache_length": (
                int(expanded_ids.numel()) + len(tokens) - 1),
            "decode_cache_append_exact": (
                all(length == int(expanded_ids.numel())
                    for length in prefill_cache_lengths)
                and all(length == int(expanded_ids.numel()) + len(tokens) - 1
                        for length in final_cache_lengths)),
            "status": "ok", "retry_count": 0,
            "page_cache_conditioning_ms": (
                conditioning_done - conditioning_started) * 1e3,
            "page_cache_conditioning_method": (
                "posix_fadvise_DONTNEED" if cold else "disabled"),
            "page_cache_conditioning_excluded_from_ttft": True,
            "mixed_hit_miss_parallelism": "not_applicable_single_image_all_hit",
            "layerwise_prefetch": False,
            "io": io,
        })
        # Match the other server wrappers: the authoritative request E2E
        # boundary includes response decoding and result construction, while
        # the source-integrity audit remains an explicitly out-of-band check.
        response_ready_at = time.perf_counter()
        stats["server_postprocess_ms"] = (
            response_ready_at - finished_at) * 1e3
        stats["request_e2e_ms"] = (
            response_ready_at - request_started) * 1e3

        source_hash_after = context.validated_payload_hash
        integrity_after_ms = 0.0
        stats.update({
            "source_payload_hash_before": source_hash_before,
            "source_payload_hash_after": source_hash_after,
            "payload_integrity_check_ms": float(
                integrity_before_ms + integrity_after_ms),
            "payload_integrity_timing_semantics": (
                "payload sampled and verified when the image context opened; "
                "no per-request sample read; serving runner performs a live "
                "image-boundary audit after the final hit"),
            "request_return_wall_ms": (
                time.perf_counter() - request_started) * 1e3,
        })
        _clear_foreign_attention_bias()
        del cache
        return stats


__all__ = [
    "ActiveRowPlan", "METHOD_ID", "MPICContext", "MPICRawReader",
    "MPICSelectivePrefill", "MPICServer", "PAPER_LABEL",
    "assemble_linked_kv", "build_active_row_plan", "build_causal_mask",
    "expand_prompt_without_pixels", "persist_captured_mpic_prefix",
    "relocate_post_rope_keys", "seed_dynamic_cache",
]

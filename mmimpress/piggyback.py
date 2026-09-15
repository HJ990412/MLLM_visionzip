"""Turn-1 piggyback capture and durable Visual-KV persistence.

This module deliberately does not expose a model-forward helper.  A caller
wraps its *normal* first-turn multimodal request in :class:`VisionForwardCapture`
and passes the cache returned by that same request to
:func:`persist_captured_visual_prefix`.  Consequently neither saliency capture
nor persistence can perform a second vision or language-model prefix forward.

Only the penultimate vision encoder layer's ``self_attn`` is asked for
attention weights.  All other vision layers, and the top-level model, retain
their ordinary ``output_attentions=False`` behaviour.
"""
from __future__ import annotations

import ctypes
import errno
import hashlib
import importlib.metadata
import inspect
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from mmimpress.config import CHUNK_SIZE
from mmimpress.cvpr25 import (anyres_token_scores, permutation_sha256,
                              visionzip_repack_order)
from mmimpress.model import cache_layers
from mmimpress.reorder import mapping_from_perm
from mmimpress.store import write_image_store


def sha256_file(path: Path, block_bytes: int = 8 << 20) -> str:
    """Return the SHA256 of one regular file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    """Hash a JSON value using a stable, whitespace-free canonical encoding."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def deterministic_method_rotation(methods: Sequence[str], dialog_index: int,
                                  seed: int = 0) -> tuple[str, ...]:
    """Rotate a fixed method list deterministically for one dialog.

    ``dialog_index`` is zero-based.  With seed zero, dialogs 0, 1, and 2 start
    at methods 0, 1, and 2 respectively.  This balances first/last execution
    positions without changing the workload or using process-global RNG state.
    """
    ordered = tuple(str(method) for method in methods)
    if not ordered:
        raise ValueError("at least one method is required")
    if len(set(ordered)) != len(ordered):
        raise ValueError("method rotation requires unique method names")
    index = int(dialog_index)
    if index < 0:
        raise ValueError("dialog_index must be non-negative")
    offset = (index + int(seed)) % len(ordered)
    return ordered[offset:] + ordered[:offset]


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


class VisionForwardCapture:
    """Instrument exactly one normal vision-tower invocation.

    ``capture_saliency=False`` installs timing-only hooks on the vision tower;
    it never changes any forward arguments.  ``capture_saliency=True`` also
    installs a pre-hook and a forward hook on *only* the penultimate encoder
    layer's ``self_attn``.  That pre-hook requests eager attention weights and
    the forward hook records exactly

    ``attentions[:, :, 0, 1:].sum(dim=1)``.

    CUDA events measure the full vision invocation and saliency-reduction
    kernel when inputs are on CUDA.  CPU wall time is used by synthetic tests
    and other CPU execution.  Attention scores are copied to CPU on context
    exit, after the wrapped normal request has returned.  The context fails
    closed unless exactly one vision invocation (and, in capture mode, exactly
    one penultimate-attention invocation) occurred.  Hooks are always removed.
    """

    def __init__(self, runner, capture_saliency: bool = False):
        self.runner = runner
        self.capture_saliency = bool(capture_saliency)
        self.call_count = 0
        self.saliency_call_count = 0
        self.vision_elapsed_ms: float | None = None
        self.saliency_reduction_ms: float | None = None
        self.saliency_materialize_ms: float = 0.0
        self.saliency_hook_submit_ms: float = 0.0
        self.saliency_scores: torch.Tensor | None = None
        self.timing_backend: str | None = None
        self._handles = []
        self._entered = False
        self._vision_cpu_t0: float | None = None
        self._reduction_cpu_t0: float | None = None
        self._vision_events = None
        self._reduction_events = None
        self._prepared_vision_events = None
        self._prepared_reduction_events = None
        self._saliency_device: torch.Tensor | None = None

        try:
            self.vision_tower = runner.model.model.vision_tower
            encoder_layers = self.vision_tower.vision_model.encoder.layers
        except AttributeError as exc:
            raise AssertionError(
                "expected LLaVA-NeXT vision_tower.vision_model.encoder.layers"
            ) from exc
        if len(encoder_layers) < 2:
            raise AssertionError("vision tower needs a penultimate layer")
        assert not self.vision_tower.training, \
            "vision capture requires the normal eval-mode serving model"
        self.penultimate_self_attn = encoder_layers[-2].self_attn
        self.saliency_layer_index = len(encoder_layers) - 2
        self._attention_signature = inspect.signature(
            self.penultimate_self_attn.forward)
        assert "output_attentions" in self._attention_signature.parameters, (
            "vision self-attention has no output_attentions parameter"
        )

        vision_config = getattr(self.vision_tower, "config", None)
        configured_output = getattr(vision_config, "output_attentions", False)
        assert configured_output is False, (
            "vision config must keep global output_attentions=False"
        )
        model_config = getattr(runner.model, "config", None)
        model_output = getattr(model_config, "output_attentions", False)
        assert model_output is False, (
            "top-level model config must keep output_attentions=False"
        )

        if self.capture_saliency:
            config = getattr(self.penultimate_self_attn, "config", None)
            implementation = getattr(config, "_attn_implementation", None)
            assert implementation == "eager", (
                "penultimate vision self-attention must use eager attention "
                f"for exact weights, got {implementation!r}"
            )

    @property
    def per_sub_scores(self) -> torch.Tensor | None:
        """Alias matching :func:`mmimpress.cvpr25.anyres_token_scores`."""
        return self.saliency_scores

    def __enter__(self):
        if self._entered:
            raise RuntimeError("VisionForwardCapture instances are single-use")
        self._entered = True
        active_name = "_mmimpress_active_vision_forward_capture"
        if getattr(self.vision_tower, active_name, None) is not None:
            raise RuntimeError(
                "concurrent/nested capture on one vision tower is unsafe"
            )
        setattr(self.vision_tower, active_name, self)
        try:
            model_device = getattr(self.runner.model, "device", None)
            if model_device is None:
                try:
                    model_device = next(self.vision_tower.parameters()).device
                except StopIteration:
                    pass
            if model_device is not None:
                model_device = torch.device(model_device)
            if (model_device is not None and model_device.type == "cuda"
                    and torch.cuda.is_available()):
                with torch.cuda.device(model_device):
                    self._prepared_vision_events = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    if self.capture_saliency:
                        self._prepared_reduction_events = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                    # CUDA events are lazily created on first record.  Warm
                    # them here, before the wrapped request starts its timer.
                    stream = torch.cuda.current_stream(model_device)
                    for event in self._prepared_vision_events:
                        event.record(stream)
                    if self._prepared_reduction_events is not None:
                        for event in self._prepared_reduction_events:
                            event.record(stream)
                    stream.synchronize()
            self._handles.append(self.vision_tower.register_forward_pre_hook(
                self._vision_pre_hook, with_kwargs=True))
            self._handles.append(self.vision_tower.register_forward_hook(
                self._vision_forward_hook, with_kwargs=True))
            if self.capture_saliency:
                self._handles.append(
                    self.penultimate_self_attn.register_forward_pre_hook(
                        self._attention_pre_hook, with_kwargs=True))
                self._handles.append(
                    self.penultimate_self_attn.register_forward_hook(
                        self._attention_forward_hook, with_kwargs=True))
        except BaseException:
            self._remove_hooks()
            raise
        return self

    def _remove_hooks(self):
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        active_name = "_mmimpress_active_vision_forward_capture"
        if getattr(self.vision_tower, active_name, None) is self:
            delattr(self.vision_tower, active_name)

    def _vision_pre_hook(self, module, args, kwargs):
        self.call_count += 1
        assert self.call_count == 1, (
            "wrapped request invoked the vision tower more than once"
        )
        tensor = _first_tensor((args, kwargs))
        use_cuda = bool(tensor is not None and tensor.is_cuda
                        and torch.cuda.is_available())
        if use_cuda:
            assert self._prepared_vision_events is not None, (
                "CUDA vision input disagrees with runner.model.device"
            )
            start, end = self._prepared_vision_events
            start.record(torch.cuda.current_stream(tensor.device))
            self._vision_events = (start, end)
            self.timing_backend = "cuda_event"
        else:
            self._vision_cpu_t0 = time.perf_counter()
            self.timing_backend = "cpu_perf_counter"

    def _vision_forward_hook(self, module, args, kwargs, output):
        if self._vision_events is not None:
            tensor = _first_tensor(output)
            assert tensor is not None and tensor.is_cuda
            self._vision_events[1].record(
                torch.cuda.current_stream(tensor.device))
        else:
            assert self._vision_cpu_t0 is not None
            self.vision_elapsed_ms = (
                time.perf_counter() - self._vision_cpu_t0) * 1e3

    def _attention_pre_hook(self, module, args, kwargs):
        self.saliency_call_count += 1
        assert self.saliency_call_count == 1, (
            "penultimate vision self-attention ran more than once"
        )
        # Bind against the installed CLIPAttention signature so both current
        # keyword calls and older positional calls change exactly one named
        # argument without relying on a version-specific positional index.
        bound = self._attention_signature.bind_partial(*args, **kwargs)
        bound.arguments["output_attentions"] = True
        return bound.args, bound.kwargs

    def _attention_forward_hook(self, module, args, kwargs, output):
        assert isinstance(output, (tuple, list)) and len(output) >= 2, (
            "eager vision self-attention did not return attention weights"
        )
        attentions = output[1]
        assert torch.is_tensor(attentions) and attentions.ndim == 4, (
            "expected vision attention shape (batch, heads, tokens, tokens)"
        )
        assert attentions.shape[-2] >= 1 and attentions.shape[-1] >= 2, (
            "vision attention has no CLS-to-patch row"
        )
        use_cuda = bool(attentions.is_cuda and torch.cuda.is_available())
        submit_t0 = time.perf_counter()
        if use_cuda:
            assert self._prepared_reduction_events is not None
            start, end = self._prepared_reduction_events
            stream = torch.cuda.current_stream(attentions.device)
            start.record(stream)
            scores = attentions[:, :, 0, 1:].sum(dim=1).float()
            end.record(stream)
            self._reduction_events = (start, end)
        else:
            self._reduction_cpu_t0 = time.perf_counter()
            scores = attentions[:, :, 0, 1:].sum(dim=1).float()
            self.saliency_reduction_ms = (
                time.perf_counter() - self._reduction_cpu_t0) * 1e3
        self.saliency_hook_submit_ms = (
            time.perf_counter() - submit_t0) * 1e3
        self._saliency_device = scores.detach()

    def _finish_timings_and_scores(self):
        if self._vision_events is not None:
            start, end = self._vision_events
            end.synchronize()
            self.vision_elapsed_ms = float(start.elapsed_time(end))
        if self._reduction_events is not None:
            start, end = self._reduction_events
            end.synchronize()
            self.saliency_reduction_ms = float(start.elapsed_time(end))
        if self.capture_saliency and self._saliency_device is not None:
            t0 = time.perf_counter()
            self.saliency_scores = self._saliency_device.cpu()
            self.saliency_materialize_ms = (time.perf_counter() - t0) * 1e3
            self._saliency_device = None

    def __exit__(self, exc_type, exc, traceback):
        # Removal comes first so even timing/materialisation failures cannot
        # leak instrumentation into later methods or requests.
        self._remove_hooks()
        if exc_type is not None:
            self._saliency_device = None
            return False
        self._finish_timings_and_scores()
        assert self.call_count == 1, (
            f"expected one vision-tower invocation, got {self.call_count}"
        )
        if self.capture_saliency:
            assert self.saliency_call_count == 1, (
                "expected one penultimate self-attention invocation, got "
                f"{self.saliency_call_count}"
            )
            assert self.saliency_scores is not None
            assert torch.isfinite(self.saliency_scores).all(), \
                "captured vision saliency contains NaN/Inf"
        return False

    def result_cpu(self) -> torch.Tensor:
        """Return captured CPU scores after successful context exit."""
        if not self.capture_saliency:
            raise RuntimeError("timing-only capture has no saliency result")
        if self.saliency_scores is None:
            raise RuntimeError("saliency result is not ready until context exit")
        return self.saliency_scores

    def stats(self) -> dict[str, Any]:
        """JSON-ready instrumentation summary."""
        reduction = (float(self.saliency_reduction_ms)
                     if self.saliency_reduction_ms is not None else 0.0)
        materialize = float(self.saliency_materialize_ms)
        return {
            "capture_saliency": self.capture_saliency,
            "vision_call_count": int(self.call_count),
            "saliency_call_count": int(self.saliency_call_count),
            "vision_ms": (None if self.vision_elapsed_ms is None
                          else float(self.vision_elapsed_ms)),
            "saliency_reduction_ms": reduction,
            "saliency_reduce_cuda_ms": (
                reduction if self._reduction_events is not None else None),
            "saliency_materialize_ms": materialize,
            "saliency_d2h_ms": materialize,
            "saliency_post_response_ms": materialize,
            "saliency_hook_submit_ms": float(self.saliency_hook_submit_ms),
            # Reduction is already inside Turn-1 request/vision timing.  D2H
            # happens on context exit after the response, so keep it separate
            # and never add the combined number to Turn-1 again.
            "saliency_extra_ms": reduction,
            "saliency_total_instrumentation_ms": reduction + materialize,
            "extra_saliency_forward_ms": 0.0,
            "extra_vision_forward_calls": max(0, int(self.call_count) - 1),
            "vision_num_layers": int(self.saliency_layer_index + 2),
            "saliency_layer_index": int(self.saliency_layer_index),
            "saliency_layer_from_end": 2,
            "vision_attention_backend": getattr(
                getattr(self.penultimate_self_attn, "config", None),
                "_attn_implementation", None),
            "global_vision_output_attentions": False,
            "global_model_output_attentions": False,
            "transformers_version": _package_version("transformers"),
            "timing_backend": self.timing_backend,
            "vision_timing_semantics": (
                "CUDA-event device timeline when timing_backend=cuda_event; "
                "server end-to-end TTFT wall timestamps remain authoritative"
            ),
        }


class VisionSaliencyCapture(VisionForwardCapture):
    """Convenience spelling for exact penultimate-attention capture."""

    def __init__(self, runner):
        super().__init__(runner, capture_saliency=True)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_staging_tree(root: Path) -> tuple[float, float, int, int]:
    """fsync every file and then every directory, deepest first."""
    files = sorted(path for path in root.rglob("*") if path.is_file())
    t0 = time.perf_counter()
    for path in files:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    file_ms = (time.perf_counter() - t0) * 1e3

    directories = [root] + [path for path in root.rglob("*")
                            if path.is_dir()]
    directories.sort(key=lambda path: len(path.parts), reverse=True)
    t0 = time.perf_counter()
    for path in directories:
        _fsync_directory(path)
    directory_ms = (time.perf_counter() - t0) * 1e3
    return file_ms, directory_ms, len(files), len(directories)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory with Linux RENAME_NOREPLACE."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError(
            "atomic no-clobber publication requires Linux renameat2"
        )
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd, os.fsencode(source), at_fdcwd, os.fsencode(destination),
        rename_noreplace,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), destination)
        raise OSError(error, os.strerror(error), destination)


def _store_file_manifest(root: Path) -> tuple[dict[str, int],
                                               dict[str, str], str]:
    sizes: dict[str, int] = {}
    hashes: dict[str, str] = {}
    tree = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = sha256_file(path)
        sizes[relative] = int(size)
        hashes[relative] = digest
        # Documented stable tree-hash framing: UTF-8 relative path, NUL,
        # eight-byte unsigned size, raw SHA256, then NUL.
        tree.update(relative.encode("utf-8"))
        tree.update(b"\0")
        tree.update(int(size).to_bytes(8, "big", signed=False))
        tree.update(bytes.fromhex(digest))
        tree.update(b"\0")
    return sizes, hashes, tree.hexdigest()


def _store_file_sizes(root: Path) -> dict[str, int]:
    return {
        path.relative_to(root).as_posix(): int(path.stat().st_size)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def _sampled_store_sha256(root: Path, sizes: Mapping[str, int],
                          sample_bytes: int = 4096) -> str:
    """Hash file identities plus bounded head/tail samples from every file.

    This provides a cheap provenance fingerprint over real persisted KV bytes;
    unlike a full-store hash it does not reread roughly 1.2 GB per image on the
    latency experiment's persistence path.
    """
    digest = hashlib.sha256()
    for relative in sorted(sizes):
        size = int(sizes[relative])
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(size.to_bytes(8, "big", signed=False))
        path = root / relative
        fd = os.open(path, os.O_RDONLY)
        try:
            take = min(int(sample_bytes), size)
            head = os.pread(fd, take, 0)
            tail_offset = max(0, size - take)
            tail = os.pread(fd, take, tail_offset) if size > take else b""
        finally:
            os.close(fd)
        digest.update(len(head).to_bytes(4, "big"))
        digest.update(head)
        digest.update(len(tail).to_bytes(4, "big"))
        digest.update(tail)
        digest.update(b"\0")
    return digest.hexdigest()


def _byte_breakdown(sizes: Mapping[str, int], meta: Mapping[str, Any]) \
        -> dict[str, int]:
    system_prefix = int(sizes.get("sys_kv.pt", 0))
    layout_metadata = int(sizes.get("visionzip_layout.pt", 0))
    meta_bytes = int(sizes.get("meta.json", 0))
    visual = int(meta["bytes_visual_kv"])
    probe = int(meta["bytes_probe_sidecar"])
    separator = int(meta["bytes_separator_sidecar"])
    total = int(sum(sizes.values()))
    known = (system_prefix + layout_metadata + meta_bytes + visual + probe
             + separator)
    return {
        "visual_kv": visual,
        "probe_sidecar": probe,
        "separator_sidecar": separator,
        "system_prefix": system_prefix,
        "layout_metadata": layout_metadata,
        "meta": meta_bytes,
        "other": total - known,
        "total": total,
    }


@torch.no_grad()
def persist_captured_visual_prefix(
    runner,
    captured_past_key_values,
    expanded_input_ids: torch.Tensor,
    image_size: Sequence[int] | torch.Tensor,
    per_sub_scores: torch.Tensor,
    out_dir: Path,
    *,
    image_id: str | int,
    model_id: str | None = None,
    chunk_size: int = CHUNK_SIZE,
    image_input_sha256: str | None = None,
    capture_stats: Mapping[str, Any] | VisionForwardCapture | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    full_integrity_hash: bool = False,
) -> dict[str, Any]:
    """Persist the reusable prefix from a completed normal Turn-1 request.

    Parameters
    ----------
    captured_past_key_values:
        The ``DynamicCache`` returned by the same normal multimodal inference
        that produced Turn 1's answer.  It may contain question/answer rows;
        only ``[system | expanded visual span]`` is written.
    expanded_input_ids:
        The processor-produced input IDs used for that request.  They must
        contain exactly one contiguous expanded image-token span.
    per_sub_scores:
        Penultimate CLS-to-patch scores captured by
        :class:`VisionForwardCapture`, one row per AnyRes sub-image.
    capture_stats:
        Prefer passing the completed :class:`VisionForwardCapture` object
        itself.  A JSON-ready mapping from ``capture.stats()`` is also
        accepted, but it must prove one saliency-enabled eager vision call,
        zero extra forwards, penultimate-layer capture, and globally disabled
        top-level/full-tower attentions.
    out_dir:
        Final per-image store directory.  Publication is atomic and refuses to
        replace any existing path, including a broken symlink.

    The function performs geometry, permutation, repacking, writes and fsyncs;
    it never calls ``runner.model``, ``runner.prefix_forward`` or the vision
    tower.  A same-filesystem hidden staging directory is fully synced and
    atomically renamed with ``RENAME_NOREPLACE``; its parent is then fsynced.
    By default only bounded KV samples and small metadata are hashed after the
    store-ready ``persist_ms`` boundary.  ``full_integrity_hash=True`` performs
    a full post-publication diagnostic hash, still outside that boundary.
    """
    persist_t0 = time.perf_counter()
    destination = Path(out_dir)
    if destination.name in ("", ".", ".."):
        raise ValueError(f"unsafe output directory: {destination}")
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise FileExistsError(errno.EEXIST, "refusing to overwrite store",
                              destination)

    capture_object = (capture_stats if isinstance(
        capture_stats, VisionForwardCapture) else None)
    if capture_object is not None:
        assert capture_object.runner is runner, \
            "capture object belongs to a different runner/model"
        capture_document = capture_object.stats()
        assert torch.equal(
            torch.as_tensor(per_sub_scores).float().cpu(),
            capture_object.result_cpu(),
        ), "per_sub_scores differ from the completed capture object"
    else:
        assert capture_stats is not None, (
            "capture_stats are required to prove same-request piggyback "
            "provenance"
        )
        capture_document = dict(capture_stats)
    capture_requirements = {
        "capture_saliency": True,
        "vision_call_count": 1,
        "saliency_call_count": 1,
        "extra_vision_forward_calls": 0,
        "extra_saliency_forward_ms": 0.0,
        "saliency_layer_from_end": 2,
        "vision_attention_backend": "eager",
        "global_vision_output_attentions": False,
        "global_model_output_attentions": False,
    }
    for key, expected in capture_requirements.items():
        assert capture_document.get(key) == expected, (
            f"invalid Turn-1 capture provenance {key}: "
            f"{capture_document.get(key)!r} != {expected!r}"
        )
    assert capture_document.get("vision_ms") is not None, \
        "capture provenance is missing vision timing"
    captured_layer_count = int(capture_document.get("vision_num_layers", -1))
    captured_layer_index = int(capture_document.get(
        "saliency_layer_index", -1))
    assert captured_layer_count >= 2, \
        "capture provenance is missing the vision layer count"
    assert captured_layer_index == captured_layer_count - 2, (
        "capture provenance does not identify the penultimate vision layer"
    )

    ids = expanded_input_ids
    if ids.ndim == 2:
        assert ids.shape[0] == 1, ids.shape
        ids = ids[0]
    assert ids.ndim == 1, ids.shape
    v_start, v_num = runner.visual_span(ids)
    prefix_len = int(v_start + v_num)
    if torch.is_tensor(image_size):
        size = [int(item) for item in image_size.detach().cpu().reshape(-1)]
    else:
        size = [int(item) for item in image_size]
    assert len(size) == 2, f"expected image size [height, width], got {size}"
    base, hi_h, hi_w, newline_idx = runner.anyres_layout(size, v_num)

    layers = cache_layers(captured_past_key_values)
    assert layers, "captured cache has no decoder layers"
    for layer_index, (key, value) in enumerate(layers):
        assert key.ndim == value.ndim == 4, (layer_index, key.shape,
                                             value.shape)
        assert key.shape == value.shape, (layer_index, key.shape, value.shape)
        assert key.shape[0] == 1 and key.shape[2] >= prefix_len, (
            f"captured layer {layer_index} is shorter than visual prefix: "
            f"{key.shape[2]} < {prefix_len}"
        )

    t0 = time.perf_counter()
    token_scores = anyres_token_scores(
        runner, torch.as_tensor(per_sub_scores).float().cpu(), size, v_num,
        base_side=base,
    )
    token_mapping_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    stored_to_original = visionzip_repack_order(token_scores, newline_idx)
    original_to_stored = mapping_from_perm(stored_to_original)
    permutation_ms = (time.perf_counter() - t0) * 1e3
    assert stored_to_original[-len(newline_idx):] == list(newline_idx)

    invariant_metadata = {
        "image_id": str(image_id),
        "model": str(model_id or getattr(runner, "model_id", "unknown")),
        "base_grid": int(base),
        "hires_grid": [int(hi_h), int(hi_w)],
        "physical_layout": "visionzip_image_only",
        "layout_method": "visionzip_image_only",
        "layout_source": "turn1_normal_inference_piggyback",
        "visual_kv_source": "turn1_captured_past_key_values",
        "saliency_source": (
            "turn1_vision_penultimate_cls_to_patch_attention_head_sum"
        ),
        "turn1_normal_inference": True,
        "separate_vision_forward": False,
        "separate_prefix_forward": False,
        "composed_from_store": False,
        "layout_uses_dataset_question": False,
        "llm_used_for_layout_scoring": False,
        "calibration_questions": 0,
        "turn1_question_present_in_source_forward": True,
        "visual_prefix_causally_precedes_question": True,
        "global_order_all_layers": True,
        "separator_policy": "stable_tail_plus_sidecar",
        "separator_tail": True,
        "probe_heads_required_for_serving": 0,
        "permutation_sha256": permutation_sha256(stored_to_original),
        "inverse_permutation_sha256": permutation_sha256(
            original_to_stored),
    }
    if image_input_sha256 is not None:
        invariant_metadata["image_input_sha256"] = str(image_input_sha256)
    invariant_metadata["turn1_capture"] = capture_document
    invariant_metadata["capture_provenance_validated"] = True
    if extra_metadata:
        writer_reserved = {
            "v_token_start", "v_token_num", "prefix_len", "num_layers",
            "num_heads", "head_dim", "probe_heads", "dtype", "chunk_size",
            "n_chunks_per_layer", "newline_idx", "newline_stored",
            "n_spatial", "prefix_input_ids", "layout", "order",
            "order_is_per_layer", "reordered", "bytes_visual_kv",
            "bytes_probe_sidecar", "bytes_separator_sidecar",
        }
        conflicts = (set(extra_metadata) & set(invariant_metadata)
                     | set(extra_metadata) & writer_reserved)
        if conflicts:
            raise ValueError(
                "extra_metadata cannot override invariants: "
                + ", ".join(sorted(conflicts))
            )
        invariant_metadata.update(dict(extra_metadata))

    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.staging-", dir=parent))
    published = False
    try:
        writer_timing: dict[str, float] = {}
        meta = write_image_store(
            staging, layers, int(v_start), int(v_num),
            ids[:prefix_len].detach().cpu().tolist(), list(newline_idx),
            extra=invariant_metadata, chunk_size=int(chunk_size),
            probe_heads=0, stored_to_original=stored_to_original,
            separator_sidecar=True, timing_out=writer_timing,
        )
        assert meta["probe_heads"] == 0
        assert meta["bytes_probe_sidecar"] == 0
        assert meta["bytes_separator_sidecar"] > 0

        layout_artifact = {
            "schema_version": 2,
            "image_id": str(image_id),
            "importance_source": invariant_metadata["saliency_source"],
            "capture_source": "same_turn1_normal_inference",
            "token_score_original": token_scores,
            "stored_to_original": torch.tensor(stored_to_original,
                                                dtype=torch.int32),
            "original_to_stored": torch.tensor(original_to_stored,
                                                dtype=torch.int32),
            "newline_original": torch.tensor(newline_idx,
                                             dtype=torch.int32),
            "layout_uses_dataset_question": False,
            "llm_used_for_layout_scoring": False,
            "calibration_questions": 0,
            "separate_vision_forward": False,
            "separate_prefix_forward": False,
            "image_input_sha256": image_input_sha256,
        }
        t0 = time.perf_counter()
        torch.save(layout_artifact, staging / "visionzip_layout.pt")
        layout_write_ms = (time.perf_counter() - t0) * 1e3

        file_sizes = _store_file_sizes(staging)
        byte_breakdown = _byte_breakdown(file_sizes, meta)

        file_fsync_ms, directory_fsync_ms, synced_files, synced_dirs = \
            _fsync_staging_tree(staging)
        t0 = time.perf_counter()
        _rename_noreplace(staging, destination)
        atomic_rename_ms = (time.perf_counter() - t0) * 1e3
        published = True
        t0 = time.perf_counter()
        try:
            _fsync_directory(parent)
        except Exception as exc:
            raise RuntimeError(
                f"store was atomically published at {destination}, but "
                "parent-directory fsync failed; durability is uncertain and "
                "the no-clobber destination must be inspected, not retried"
            ) from exc
        parent_fsync_ms = (time.perf_counter() - t0) * 1e3

        durability_ms = (file_fsync_ms + directory_fsync_ms
                         + atomic_rename_ms + parent_fsync_ms)
        # Store-ready is the critical persistence boundary.  Integrity hashes
        # below are diagnostic result bookkeeping and explicitly excluded.
        persist_ms = (time.perf_counter() - persist_t0) * 1e3

        t0 = time.perf_counter()
        integrity_ok = True
        integrity_error = None
        sample_hash = None
        file_hashes: dict[str, str] = {}
        tree_hash = None
        try:
            sample_hash = _sampled_store_sha256(destination, file_sizes)
            if full_integrity_hash:
                checked_sizes, file_hashes, tree_hash = \
                    _store_file_manifest(destination)
                assert checked_sizes == file_sizes, \
                    "store sizes changed after durable publication"
            else:
                file_hashes = {
                    relative: sha256_file(destination / relative)
                    for relative in ("meta.json", "visionzip_layout.pt")
                }
        except Exception as exc:
            # Publication and its parent fsync already succeeded.  Never turn
            # a post-commit diagnostic failure into an apparent persist
            # failure that a no-clobber retry cannot recover from.
            integrity_ok = False
            integrity_error = f"{type(exc).__name__}: {exc}"
        hash_ms = (time.perf_counter() - t0) * 1e3
        helper_total_ms = (time.perf_counter() - persist_t0) * 1e3
        total_ssd_write_ms = (float(writer_timing["ssd_write_ms"])
                              + float(layout_write_ms))
        total_repack_ms = (float(writer_timing["kv_materialize_ms"])
                           + float(writer_timing["kv_repack_ms"]))
        timing_ms = {
            "token_mapping_ms": float(token_mapping_ms),
            "permutation_ms": float(permutation_ms),
            "kv_materialize_ms": float(writer_timing["kv_materialize_ms"]),
            "kv_repack_ms": float(writer_timing["kv_repack_ms"]),
            "repack_ms": total_repack_ms,
            "store_write_ms": float(writer_timing["ssd_write_ms"]),
            "layout_write_ms": float(layout_write_ms),
            "ssd_write_ms": total_ssd_write_ms,
            "hash_ms": float(hash_ms),
            "integrity_hash_ms": float(hash_ms),
            "file_fsync_ms": float(file_fsync_ms),
            "directory_fsync_ms": float(directory_fsync_ms),
            "atomic_rename_ms": float(atomic_rename_ms),
            "parent_fsync_ms": float(parent_fsync_ms),
            "durability_ms": float(durability_ms),
            "persist_ms": float(persist_ms),
            "helper_total_ms": float(helper_total_ms),
        }
        return {
            "store_dir": str(destination.resolve()),
            "image_id": str(image_id),
            "meta": meta,
            "timing_ms": timing_ms,
            "bytes": byte_breakdown,
            "file_sizes": file_sizes,
            "hashes": {
                "files_sha256": file_hashes,
                "tree_sha256": tree_hash,
                "prefix_kv_sample_sha256": sample_hash,
                "sample_hash_framing": (
                    "each sorted file: path_utf8 + NUL + uint64be(size) + "
                    "len/head4096 + len/tail4096 + NUL"
                ),
                "full_integrity_hash": bool(full_integrity_hash),
                "permutation_sha256": meta["permutation_sha256"],
                "inverse_permutation_sha256": meta[
                    "inverse_permutation_sha256"],
                "tree_hash_framing": (
                    "path_utf8 + NUL + uint64be(size) + raw_sha256 + NUL"
                    if full_integrity_hash else None
                ),
            },
            "durability": {
                "same_filesystem_staging": True,
                "atomic_no_clobber": True,
                "files_fsynced": int(synced_files),
                "directories_fsynced_before_rename": int(synced_dirs),
                "parent_fsynced_after_rename": True,
            },
            "integrity": {
                "ok": integrity_ok,
                "error": integrity_error,
                "excluded_from_persist_ms": True,
            },
        }
    finally:
        if not published and os.path.lexists(staging):
            shutil.rmtree(staging)


# Short spelling for callers that organise all persistence around Turn 1.
persist_turn1_prefix = persist_captured_visual_prefix


__all__ = [
    "VisionForwardCapture",
    "VisionSaliencyCapture",
    "deterministic_method_rotation",
    "persist_captured_visual_prefix",
    "persist_turn1_prefix",
    "sha256_file",
    "stable_json_sha256",
]

"""Image-only scoring captured from an unchanged GLM-4.6V-Flash vision pass.

The checkpoint uses Glm4vVisionModel and Glm46VProcessor. The frozen formula
and native Conv2d merge mapping are in docs/glm46v_architecture.md. This is a
VisionZip-style adaptation, not an official GLM implementation of VisionZip.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch


CHECKPOINT_REVISION = "411bb4d77144a3f03accbf4b780f5acb8b7cde4e"
SCORE_SOURCE = "glm4v_last_vision_block_postsoftmax_headmean_querysum_groupmean"


def _validated_grid(grid_thw: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    grid = torch.as_tensor(grid_thw).detach().cpu()
    if grid.dtype.is_floating_point or grid.dtype == torch.bool:
        raise ValueError("image_grid_thw must contain integer geometry")
    grid = grid.long()
    if grid.ndim != 2 or grid.shape[0] < 1 or grid.shape[1] != 3 or (grid <= 0).any():
        raise ValueError("image_grid_thw must have positive [T,H,W] rows")
    if spatial_merge_size < 1 or (grid[:, 1:] % spatial_merge_size != 0).any():
        raise ValueError("spatial grid dimensions must divide spatial_merge_size")
    return grid


def native_patch_coordinates(grid_thw: torch.Tensor, spatial_merge_size: int = 2) -> torch.Tensor:
    """Return [patch, T/H/W] in native spatial-block-major processor order."""
    grid = _validated_grid(grid_thw, spatial_merge_size)
    coordinates: list[list[int]] = []
    for temporal, height, width in grid.tolist():
        for t in range(temporal):
            for block_h in range(height // spatial_merge_size):
                for block_w in range(width // spatial_merge_size):
                    for local_h in range(spatial_merge_size):
                        for local_w in range(spatial_merge_size):
                            coordinates.append([
                                t,
                                block_h * spatial_merge_size + local_h,
                                block_w * spatial_merge_size + local_w,
                            ])
    return torch.tensor(coordinates, dtype=torch.long)


def received_attention_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    query_block_size: int = 128,
    _allocation_observer: Callable[[], None] | None = None,
) -> torch.Tensor:
    """Head mean of query-summed post-softmax attention received by each patch.

    Q/K are [patch, head, dim], already transformed by native vision RoPE.
    Only query rows are blocked. Every block uses ALL keys from its frame,
    preserving the full-key softmax denominator. The stock model is untouched.
    """
    if query.ndim != 3 or query.shape != key.shape or any(d == 0 for d in query.shape):
        raise ValueError("vision query/key must have identical nonempty [patch, head, dim] shape")
    if query.dtype != key.dtype or query.device != key.device or not query.dtype.is_floating_point:
        raise ValueError("vision query/key must share floating dtype and device")
    if not torch.isfinite(query).all() or not torch.isfinite(key).all():
        raise ValueError("vision query/key must be finite")
    if query_block_size < 1:
        raise ValueError("query_block_size must be positive")
    boundaries_tensor = torch.as_tensor(cu_seqlens).detach().cpu()
    if boundaries_tensor.ndim != 1 or boundaries_tensor.dtype.is_floating_point:
        raise ValueError("cu_seqlens must be one-dimensional integer boundaries")
    boundaries = boundaries_tensor.long().tolist()
    if len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != query.shape[0]:
        raise ValueError("cu_seqlens must cover exactly the vision patch sequence")
    if any(b <= a for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("cu_seqlens must increase strictly")

    scores = torch.empty(query.shape[0], dtype=torch.float32, device=query.device)
    scaling = query.shape[-1] ** -0.5
    for start, end in zip(boundaries, boundaries[1:]):
        all_keys = key[start:end].transpose(0, 1)
        received = torch.zeros((query.shape[1], end - start), dtype=torch.float32, device=query.device)
        for row_start in range(start, end, query_block_size):
            q_rows = query[row_start:min(row_start + query_block_size, end)].transpose(0, 1)
            logits = torch.matmul(q_rows, all_keys.transpose(-1, -2)) * scaling
            # Match the native GLM eager score convention while its real forward
            # remains SDPA. Explicit probabilities are solely this image score.
            probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
            if _allocation_observer is not None:
                _allocation_observer()
            received += probabilities.float().sum(dim=1)
            del logits, probabilities
        scores[start:end] = received.mean(dim=0)
    return scores


def merge_native_scores(
    patch_scores: torch.Tensor,
    grid_thw: torch.Tensor,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    """Mean each native Conv2d receptive field in final visual-token order."""
    grid = _validated_grid(grid_thw, spatial_merge_size)
    if patch_scores.ndim != 1 or not patch_scores.dtype.is_floating_point:
        raise ValueError("patch scores must be a one-dimensional floating tensor")
    expected_patches = int(grid.prod(dim=1).sum().item())
    if patch_scores.numel() != expected_patches or not torch.isfinite(patch_scores).all():
        raise ValueError("patch scores must cover the full finite native patch sequence")
    group_size = spatial_merge_size**2
    return patch_scores.reshape(expected_patches // group_size, group_size).float().mean(dim=-1)


def stable_importance_order(scores: torch.Tensor) -> torch.Tensor:
    """One descending rank; equal scores retain their original native IDs."""
    if scores.ndim != 1 or scores.numel() < 1 or not torch.isfinite(scores).all():
        raise ValueError("importance scores must be a nonempty finite vector")
    return torch.argsort(scores, descending=True, stable=True)


class VisionScoreCapture:
    """Observe one normal GLM vision forward without replacing its output.

    ``scores`` is CPU FP32 in native expanded image-token order. Hooks observe
    the stock final QKV projection and remove themselves even after failure.
    ``score_peak_extra_gpu_bytes`` is the largest sampled live allocation above
    the score baseline, NOT a complete kernel/allocator peak. Global peak stats
    are never reset; the runner measures the entire request independently.
    """

    def __init__(self, model: Any, *, query_block_size: int = 128) -> None:
        self.model = model
        self.visual = getattr(model, "visual", None)
        if self.visual is None:
            self.visual = getattr(getattr(model, "model", None), "visual", None)
        if self.visual is None and hasattr(model, "blocks") and hasattr(model, "downsample"):
            self.visual = model  # small native vision-only correctness fixture
        if self.visual is None:
            raise ValueError("model does not expose native GLM visual tower")
        if not hasattr(self.visual, "downsample") or not hasattr(self.visual, "merger"):
            raise ValueError("native GLM Conv2d downsample/merger is required")
        self.query_block_size = int(query_block_size)
        if self.query_block_size < 1 or len(self.visual.blocks) < 1:
            raise ValueError("positive query blocking and a nonempty vision tower are required")
        self._last_block = len(self.visual.blocks) - 1
        self.call_count = 0
        self.score_seconds = 0.0
        self.hook_cpu_seconds = 0.0
        self.qk_copy_seconds = 0.0
        self.qk_capture_bytes = 0
        self.score_peak_extra_gpu_bytes = 0
        self.score_peak_gpu_allocated_bytes = 0
        self.score_memory_scope = "sampled_live_allocation; request_peak_measured_separately"
        self.scores: torch.Tensor | None = None
        self.geometry: dict[str, Any] = {}
        self.cu_seqlens: torch.Tensor | None = None
        self._grid_thw: torch.Tensor | None = None
        self._position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None
        self._qk: torch.Tensor | None = None
        self._vision_output_rows: int | None = None
        self._copy_events: tuple[Any, Any] | None = None
        self._handles: list[Any] = []

    def __enter__(self) -> "VisionScoreCapture":
        if self._handles or self.call_count:
            raise RuntimeError("VisionScoreCapture instances can be used only once")
        if self.model.training or self.visual.training:
            raise ValueError("vision score capture requires eval mode")
        attention = self.visual.blocks[self._last_block].attn
        if attention.training or getattr(attention, "is_causal", False):
            raise ValueError("native GLM vision attention must be noncausal and eval")
        self._handles = [
            self.visual.register_forward_pre_hook(self._vision_pre_hook, with_kwargs=True),
            self.visual.register_forward_hook(self._vision_post_hook, with_kwargs=True),
            attention.register_forward_pre_hook(self._attention_pre_hook, with_kwargs=True),
            attention.qkv.register_forward_hook(self._qkv_hook),
        ]
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if exc_type is not None:
            self._release_capture()
            return
        try:
            if self.call_count != 1 or self._qk is None or self._position_embeddings is None:
                raise RuntimeError("expected one complete native GLM vision forward")
            self._compute_scores()
        finally:
            self._release_capture()

    def _vision_pre_hook(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        started = time.perf_counter()
        self.call_count += 1
        if self.call_count != 1:
            raise RuntimeError("more than one vision call during GLM score capture")
        grid = kwargs.get("grid_thw", args[1] if len(args) > 1 else None)
        if grid is None:
            raise ValueError("vision forward did not receive grid_thw")
        self._grid_thw = _validated_grid(grid, int(self.visual.spatial_merge_size)).clone()
        self.hook_cpu_seconds += time.perf_counter() - started

    def _vision_post_hook(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
        started = time.perf_counter()
        merged = output.pooler_output if hasattr(output, "pooler_output") else output
        if not isinstance(merged, torch.Tensor) or merged.ndim != 2:
            raise ValueError("GLM stock merger must return [visual_token, hidden] output")
        self._vision_output_rows = int(merged.shape[0])
        self.hook_cpu_seconds += time.perf_counter() - started

    def _attention_pre_hook(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        started = time.perf_counter()
        positions = kwargs.get("position_embeddings", args[2] if len(args) > 2 else None)
        boundaries = kwargs.get("cu_seqlens", args[1] if len(args) > 1 else None)
        if not isinstance(positions, tuple) or len(positions) != 2 or boundaries is None:
            raise ValueError("native GLM attention RoPE embeddings and frame boundaries are required")
        if self._position_embeddings is not None:
            raise RuntimeError("last GLM vision attention ran more than once")
        self._position_embeddings = positions
        self.cu_seqlens = boundaries.detach().cpu().long().clone()
        self.hook_cpu_seconds += time.perf_counter() - started

    def _qkv_hook(self, module: Any, args: tuple[Any, ...], output: torch.Tensor) -> None:
        started = time.perf_counter()
        if self._qk is not None:
            raise RuntimeError("last GLM vision QKV projection ran more than once")
        heads = int(self.visual.blocks[self._last_block].attn.num_heads)
        if output.ndim != 2 or output.shape[1] % (3 * heads):
            raise ValueError("native GLM QKV projection has unexpected shape")
        if output.device.type == "cuda":
            copy_start = torch.cuda.Event(enable_timing=True)
            copy_end = torch.cuda.Event(enable_timing=True)
            copy_start.record()
        self._qk = output.detach().reshape(output.shape[0], 3, heads, -1)[:, :2].contiguous()
        self.qk_capture_bytes = self._qk.numel() * self._qk.element_size()
        if output.device.type == "cuda":
            copy_end.record()
            self._copy_events = (copy_start, copy_end)
        else:
            self.qk_copy_seconds = time.perf_counter() - started
        self.hook_cpu_seconds += time.perf_counter() - started

    @torch.no_grad()
    def _compute_scores(self) -> None:
        if self._qk is None or self._grid_thw is None or self.cu_seqlens is None:
            raise RuntimeError("incomplete GLM score capture")
        qk = self._qk
        device = qk.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            if self._copy_events is not None:
                self.qk_copy_seconds = self._copy_events[0].elapsed_time(self._copy_events[1]) / 1000.0
            score_baseline = torch.cuda.memory_allocated(device)
        else:
            score_baseline = 0
        started = time.perf_counter()
        grid = self._grid_thw
        merge_size = int(self.visual.spatial_merge_size)
        patches = int(grid.prod(dim=1).sum().item())
        expected_cu = torch.cat((torch.zeros(1, dtype=torch.long), torch.repeat_interleave(
            grid[:, 1] * grid[:, 2], grid[:, 0]
        ).cumsum(0)))
        if qk.shape[0] != patches or not torch.equal(self.cu_seqlens, expected_cu):
            raise ValueError("captured GLM patches/full-frame attention disagree with native geometry")
        if self._vision_output_rows != patches // merge_size**2:
            raise ValueError("stock GLM merger rows disagree with native geometry")
        downsample = self.visual.downsample
        if tuple(downsample.kernel_size) != (merge_size, merge_size) or tuple(downsample.stride) != (merge_size, merge_size):
            raise ValueError("native GLM downsample kernel/stride differs from spatial merge contract")
        if tuple(downsample.padding) != (0, 0) or tuple(downsample.dilation) != (1, 1):
            raise ValueError("native GLM downsample receptive field differs from merge contract")

        def observe_allocation() -> None:
            if device.type == "cuda":
                allocated = torch.cuda.memory_allocated(device)
                self.score_peak_gpu_allocated_bytes = max(self.score_peak_gpu_allocated_bytes, allocated)
                self.score_peak_extra_gpu_bytes = max(self.score_peak_extra_gpu_bytes, allocated - score_baseline)

        from transformers.models.glm4v.modeling_glm4v import apply_rotary_pos_emb_vision

        cos, sin = self._position_embeddings  # type: ignore[misc]
        query, key = apply_rotary_pos_emb_vision(qk[:, 0], qk[:, 1], cos, sin)
        observe_allocation()
        patch_scores = received_attention_scores(
            query, key, self.cu_seqlens,
            query_block_size=self.query_block_size,
            _allocation_observer=observe_allocation,
        )
        merged_scores = merge_native_scores(patch_scores, grid, merge_size)
        self.scores = merged_scores.detach().float().cpu()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.score_seconds = time.perf_counter() - started
        self.geometry = {
            "image_grid_thw": grid.tolist(),
            "premerge_patch_count": patches,
            "merged_visual_token_count": patches // merge_size**2,
            "score_count": self.scores.numel(),
            "spatial_merge_size": merge_size,
            "merge_group_size": merge_size**2,
            "patch_order": "frame_spatial_block_local_hw",
            "visual_token_order": "native_frame_raster",
            "last_vision_block_index": self._last_block,
            "last_block_is_full_attention": True,
            "frame_count": int(grid[:, 0].sum().item()),
            "score_source": SCORE_SOURCE,
            "query_block_size": self.query_block_size,
        }

    def _release_capture(self) -> None:
        self._qk = None
        self._position_embeddings = None
        self._copy_events = None

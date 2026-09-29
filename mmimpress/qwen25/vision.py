"""Image-only VisionZip scoring for a stock Qwen2.5-VL vision forward.

The reference is JIA-Lab-research/VisionZip, commit
``8f86b55c6f000eb033e6912538af2dd7dcb30502``, file
``Qwen2_5_VL/qwen2_5vl_visionzip.py`` (SHA256
``26f828971ac9d4058768e07d9cb5c329b792d13206c64e6bf66f5e4b8c026004``).
Its last-block post-softmax attention is averaged over heads, summed over
query rows, averaged within each spatial merger group, and restored from
window order with ``argsort(window_index)``. We use the same score and leave
the stock vision forward, merger, and all visual tokens intact.
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch


VISIONZIP_COMMIT = "8f86b55c6f000eb033e6912538af2dd7dcb30502"
VISIONZIP_FILE_SHA256 = "26f828971ac9d4058768e07d9cb5c329b792d13206c64e6bf66f5e4b8c026004"
SCORE_SOURCE = "last_full_vision_block_post_softmax_received_attention"


def received_attention_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    query_block_size: int = 128,
) -> torch.Tensor:
    """Return one received-attention score per patch in *window order*.

    ``query`` and ``key`` have shape ``[patch, vision_head, head_dim]`` and
    already include vision RoPE. Each interval in ``cu_seqlens`` is a valid
    full-attention sequence (one frame); queries attend to every key in that
    interval. Query blocking bounds the temporary attention allocation while
    preserving the exact softmax denominator over all valid keys.
    """
    if query.ndim != 3 or query.shape != key.shape:
        raise ValueError("vision query/key must have identical [patch, head, dim] shape")
    if query.shape[0] == 0 or query.shape[1] == 0 or query.shape[2] == 0:
        raise ValueError("vision query/key dimensions must be nonzero")
    if query_block_size < 1:
        raise ValueError("query_block_size must be positive")
    boundaries = torch.as_tensor(cu_seqlens, dtype=torch.long).tolist()
    if len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != query.shape[0]:
        raise ValueError("cu_seqlens must cover exactly the vision patch sequence")
    if any(b <= a for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("cu_seqlens must have strictly increasing boundaries")

    scores = torch.empty(query.shape[0], dtype=torch.float32, device=query.device)
    denominator = math.sqrt(query.shape[-1])
    for start, end in zip(boundaries, boundaries[1:]):
        # Key axis spans the entire valid frame, even when query rows are blocked.
        all_keys = key[start:end].transpose(0, 1)
        received = torch.zeros((query.shape[1], end - start), dtype=torch.float32, device=query.device)
        for row_start in range(start, end, query_block_size):
            row_end = min(row_start + query_block_size, end)
            q_rows = query[row_start:row_end].transpose(0, 1)
            logits = torch.matmul(q_rows, all_keys.transpose(-1, -2)) / denominator
            probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
            # Float32 accumulation avoids block-size-dependent BF16 rounding.
            received += probabilities.float().sum(dim=1)
        scores[start:end] = received.mean(dim=0)
    return scores


def merge_window_scores(
    patch_scores: torch.Tensor,
    window_index: torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Map window-ordered patch scores to original LLM visual-token order."""
    if patch_scores.ndim != 1 or window_index.ndim != 1:
        raise ValueError("patch_scores and window_index must be one-dimensional")
    if spatial_merge_size < 1:
        raise ValueError("spatial_merge_size must be positive")
    merge_unit = spatial_merge_size**2
    n_patch = patch_scores.numel()
    if n_patch % merge_unit:
        raise ValueError("patch count is not divisible by spatial merge group size")
    n_visual = n_patch // merge_unit
    if window_index.numel() != n_visual:
        raise ValueError("window_index does not cover all merged visual tokens")
    indices = window_index.to(device=patch_scores.device, dtype=torch.long)
    if not torch.equal(torch.sort(indices).values, torch.arange(n_visual, device=indices.device)):
        raise ValueError("window_index must be a permutation of merged visual-token positions")
    merged_window_order = patch_scores.reshape(n_visual, merge_unit).mean(dim=-1)
    return merged_window_order[torch.argsort(indices)]


class VisionScoreCapture:
    """Capture one stock Qwen2.5-VL vision pass and derive image-only scores.

    Use ``with VisionScoreCapture(model) as capture:`` around the normal
    multimodal model forward. On exit, ``scores`` is a CPU float32 tensor in
    the same order as the full vision merger output and expanded image tokens.
    ``score_seconds`` includes synchronized score computation and CPU transfer;
    ``hook_cpu_seconds`` counts Python hook work only. No second vision forward
    or LLM prefix forward is performed. Hooks are removed even on failure.
    """

    def __init__(self, model: Any, *, query_block_size: int = 128) -> None:
        self.model = model
        self.visual = model.visual
        self.query_block_size = int(query_block_size)
        self.call_count = 0
        self.score_seconds = 0.0
        self.score_peak_extra_gpu_bytes = 0
        self.score_peak_gpu_allocated_bytes = 0
        self.hook_cpu_seconds = 0.0
        self.scores: torch.Tensor | None = None
        self.geometry: dict[str, Any] = {}
        self.window_index: torch.Tensor | None = None
        self.cu_seqlens: torch.Tensor | None = None
        self._handles: list[Any] = []
        self._grid_thw: torch.Tensor | None = None
        self._position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None
        self._qk: torch.Tensor | None = None
        self._vision_output_rows: int | None = None
        self._last_block = len(self.visual.blocks) - 1
        if self._last_block < 0 or self._last_block not in self.visual.fullatt_block_indexes:
            raise ValueError("Qwen2.5-VL last vision block must be a full-attention block")
        if self.query_block_size < 1:
            raise ValueError("query_block_size must be positive")

    def __enter__(self) -> "VisionScoreCapture":
        if self._handles:
            raise RuntimeError("VisionScoreCapture cannot be entered twice")
        if self.model.training or self.visual.training:
            raise ValueError("vision score capture requires eval mode")
        attn = self.visual.blocks[self._last_block].attn
        self._handles = [
            self.visual.register_forward_pre_hook(self._vision_pre_hook, with_kwargs=True),
            self.visual.register_forward_hook(self._vision_post_hook, with_kwargs=True),
            attn.register_forward_pre_hook(self._attention_pre_hook, with_kwargs=True),
            attn.qkv.register_forward_hook(self._qkv_hook),
        ]
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if exc_type is not None:
            self._release_capture()
            return
        if self.call_count != 1 or self._qk is None or self._position_embeddings is None:
            self._release_capture()
            raise RuntimeError("expected exactly one complete Qwen2.5-VL vision forward")
        try:
            self._compute_scores()
        finally:
            self._release_capture()

    def _vision_pre_hook(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        started = time.perf_counter()
        self.call_count += 1
        if self.call_count != 1:
            raise RuntimeError("more than one vision tower call during score capture")
        grid = kwargs.get("grid_thw", args[1] if len(args) > 1 else None)
        if grid is None:
            raise ValueError("vision forward did not receive image_grid_thw")
        self._grid_thw = torch.as_tensor(grid).detach().cpu().long().clone()
        self.hook_cpu_seconds += time.perf_counter() - started

    def _vision_post_hook(
        self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any
    ) -> None:
        started = time.perf_counter()
        merged = output.pooler_output if hasattr(output, "pooler_output") else output
        self._vision_output_rows = int(merged.shape[0])
        self.hook_cpu_seconds += time.perf_counter() - started

    def _attention_pre_hook(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        started = time.perf_counter()
        positions = kwargs.get("position_embeddings", args[3] if len(args) > 3 else None)
        boundaries = kwargs.get("cu_seqlens", args[1] if len(args) > 1 else None)
        if positions is None or boundaries is None:
            raise ValueError("last vision attention did not receive RoPE embeddings and cu_seqlens")
        self._position_embeddings = positions
        self.cu_seqlens = boundaries.detach().cpu().long().clone()
        self.hook_cpu_seconds += time.perf_counter() - started

    def _qkv_hook(self, module: Any, args: tuple[Any, ...], output: torch.Tensor) -> None:
        started = time.perf_counter()
        if self._qk is not None:
            raise RuntimeError("last vision QKV projection ran more than once")
        num_heads = int(self.visual.blocks[self._last_block].attn.num_heads)
        if output.ndim != 2 or output.shape[1] % (3 * num_heads):
            raise ValueError("last vision QKV projection has an unexpected shape")
        # Keep Q and K only. This is a copy from the stock QKV projection output,
        # not a repeat projection or second tower pass.
        self._qk = output.detach().reshape(output.shape[0], 3, num_heads, -1)[:, :2].contiguous()
        self.hook_cpu_seconds += time.perf_counter() - started

    def _compute_scores(self) -> None:
        if self._qk is None or self._grid_thw is None or self.cu_seqlens is None:
            raise RuntimeError("incomplete vision score capture")
        qk = self._qk
        device = qk.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            score_baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            score_baseline = 0
        started = time.perf_counter()

        grid = self._grid_thw
        merge_size = int(self.visual.spatial_merge_size)
        merge_unit = merge_size**2
        if grid.ndim != 2 or grid.shape[1] != 3 or (grid <= 0).any():
            raise ValueError("image_grid_thw must have positive [T,H,W] rows")
        if (grid[:, 1:] % merge_size != 0).any():
            raise ValueError("image grid dimensions must divide spatial_merge_size")
        expected_patches = int(torch.prod(grid, dim=1).sum().item())
        if qk.shape[0] != expected_patches:
            raise ValueError("vision Q/K patch count differs from image_grid_thw")
        expected_cu = torch.cat((torch.zeros(1, dtype=torch.long), torch.repeat_interleave(
            grid[:, 1] * grid[:, 2], grid[:, 0]
        ).cumsum(0)))
        if not torch.equal(self.cu_seqlens, expected_cu):
            raise ValueError("last vision block did not use expected per-frame full-attention boundaries")

        window_index, _ = self.visual.get_window_index(grid)
        self.window_index = torch.as_tensor(window_index).detach().cpu().long().clone()
        expected_merged = expected_patches // merge_unit
        if self._vision_output_rows != expected_merged:
            raise ValueError("stock vision merger output count differs from image geometry")

        # Match stock Transformers vision attention: apply vision RoPE to the
        # Q/K projection before evaluating the exact post-softmax received score.
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision

        cos, sin = self._position_embeddings  # type: ignore[misc]
        query, key = apply_rotary_pos_emb_vision(qk[:, 0], qk[:, 1], cos, sin)
        patch_scores = received_attention_scores(
            query, key, self.cu_seqlens, query_block_size=self.query_block_size
        )
        merged_scores = merge_window_scores(patch_scores, self.window_index, merge_size)
        if merged_scores.numel() != expected_merged:
            raise ValueError("received score count differs from stock merger output")
        self.scores = merged_scores.detach().float().cpu()
        self.geometry = {
            "image_grid_thw": grid.tolist(),
            "premerge_patch_count": expected_patches,
            "merged_visual_token_count": expected_merged,
            "score_count": int(self.scores.numel()),
            "spatial_merge_size": merge_size,
            "merge_group_size": merge_unit,
            "last_vision_block_index": self._last_block,
            "last_block_is_full_attention": True,
            "frame_count": int(grid[:, 0].sum().item()),
            "score_source": SCORE_SOURCE,
        }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            self.score_peak_gpu_allocated_bytes = torch.cuda.max_memory_allocated(device)
            self.score_peak_extra_gpu_bytes = max(
                0, self.score_peak_gpu_allocated_bytes - score_baseline)
        self.score_seconds = time.perf_counter() - started

    def _release_capture(self) -> None:
        self._qk = None
        self._position_embeddings = None
        self._grid_thw = None

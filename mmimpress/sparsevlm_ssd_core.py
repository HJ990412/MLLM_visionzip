"""Explicit, fixed-budget SparseVLM SSD scoring contracts (LLaVA MHA only).

This is an adaptation of original SparseVLM's rater and attention-mean metric,
not its progressive hidden-token pruning, scheduling or recycling algorithm.
There are deliberately no imports from the legacy serving implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


METHOD_POLICIES = {
    "sparsevlm_ssd_kv25_probe3": "fixed_first_3",
    "sparsevlm_ssd_kv25_allhead": "all",
}


@dataclass(frozen=True)
class RaterSelection:
    ids: torch.Tensor
    weights: torch.Tensor
    fallback: bool


def _finite(value: torch.Tensor, name: str):
    finite = torch.isfinite(value).all()
    if value.device.type == "cuda":
        torch._assert_async(finite, f"{name} contains NaN or Inf")
    elif not bool(finite):
        raise ValueError(f"{name} contains NaN or Inf")


def _ids(values: Iterable[int], size: int, name: str) -> list[int]:
    raw = list(values)
    if any(isinstance(x, bool) or int(x) != x for x in raw):
        raise ValueError(f"{name} must contain integer indices")
    result = [int(x) for x in raw]
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate indices")
    if any(x < 0 or x >= size for x in result):
        raise ValueError(f"{name} outside [0, {size})")
    return sorted(result)


def scoring_head_ids(policy: str, query_heads: int, kv_heads: int) -> tuple[int, ...]:
    """Reject GQA and implicit zero-probe policies rather than reshape them."""
    if query_heads <= 0 or query_heads != kv_heads:
        raise ValueError("SparseVLM SSD requires MHA: Q heads must equal KV heads")
    if policy == "fixed_first_3":
        if query_heads < 3:
            raise ValueError("fixed_first_3 requires at least three MHA heads")
        return (0, 1, 2)
    if policy == "all":
        return tuple(range(query_heads))
    raise ValueError(f"unknown explicit scoring head policy: {policy!r}")


def select_raters(visual_embeddings: torch.Tensor, suffix_embeddings: torch.Tensor,
                  suffix_valid_mask: torch.Tensor | None = None) -> RaterSelection:
    """One request's FP32 rater selection over the entire actual text suffix.

    Visual embeddings include the expanded image block's structural newline
    rows. Returned IDs are suffix-local; structural rows are excluded only
    from the later real-token ranking budget. Empty suffixes fail closed.
    """
    v, t = visual_embeddings, suffix_embeddings
    if v.ndim != 2 or t.ndim != 2 or v.shape[1] != t.shape[1]:
        raise ValueError("rater embeddings must be [visual/text rows, hidden]")
    if v.shape[0] <= 0 or t.shape[0] <= 0 or v.shape[1] <= 0:
        raise ValueError("rater selection requires nonempty visual and text rows")
    if v.device != t.device:
        raise ValueError("visual and suffix embeddings must share a device")
    _finite(v, "visual embeddings")
    _finite(t, "suffix embeddings")
    valid = torch.ones(t.shape[0], dtype=torch.bool, device=t.device)
    if suffix_valid_mask is not None:
        if suffix_valid_mask.dtype != torch.bool or suffix_valid_mask.shape != valid.shape:
            raise ValueError("suffix_valid_mask must be a text-length boolean mask")
        valid = suffix_valid_mask.to(t.device)
    positions = torch.where(valid)[0]
    if positions.numel() == 0:
        raise ValueError("empty valid text suffix")
    weights = (v.float() @ t[positions].float().T).softmax(dim=-1).mean(dim=0)
    _finite(weights, "rater weights")
    chosen = positions[weights > weights.mean()]
    fallback = chosen.numel() == 0
    if fallback:
        chosen = positions
    all_weights = torch.zeros(t.shape[0], device=t.device, dtype=torch.float32)
    all_weights[positions] = weights
    return RaterSelection(chosen, all_weights, fallback)


def score_visual(query: torch.Tensor, system_key: torch.Tensor,
                 visual_key: torch.Tensor, suffix_key: torch.Tensor,
                 rater_rows: Iterable[int], head_policy: str,
                 query_block_size: int = 128,
                 visual_valid_mask: torch.Tensor | None = None,
                 suffix_valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Exact causal attention importance, with a whole-context denominator.

    Q, system K and suffix K have shape [all heads, rows, head_dim]. Visual K
    has [scoring heads, original visual rows, head_dim] in the explicit policy
    order. All operands are already RoPE-rotated and share the fixed compute
    dtype. QK matmul preserves that dtype; softmax/reduction use FP32. Blocks
    split query rows only, never the set of keys in a softmax denominator.
    """
    tensors = (query, system_key, visual_key, suffix_key)
    if any(t.ndim != 3 for t in tensors):
        raise ValueError("Q/K operands must be [heads, rows, head_dim]")
    h, suffix_len, d = query.shape
    head_ids = scoring_head_ids(head_policy, h, suffix_key.shape[0])
    if (suffix_len <= 0 or d <= 0 or visual_key.shape[1] <= 0
            or system_key.shape[0] != h or suffix_key.shape[1] != suffix_len
            or visual_key.shape[0] != len(head_ids)
            or any(t.shape[-1] != d for t in tensors)):
        raise ValueError("invalid Q/K head policy, sequence or head_dim geometry")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("unsupported fixed compute dtype")
    if any(t.dtype != query.dtype or t.device != query.device for t in tensors):
        raise ValueError("all Q/K operands must share compute dtype and device")
    if query_block_size <= 0:
        raise ValueError("query_block_size must be positive")
    for value, name in zip(tensors, ("query", "system K", "visual K", "suffix K")):
        _finite(value, name)
    raters = _ids(rater_rows, suffix_len, "rater rows")
    if not raters:
        raise ValueError("empty raters: callers must use explicit rater fallback")
    vn, pn = visual_key.shape[1], system_key.shape[1]
    vv = torch.ones(vn, dtype=torch.bool, device=query.device)
    sv = torch.ones(suffix_len, dtype=torch.bool, device=query.device)
    for supplied, expected, name in ((visual_valid_mask, vv, "visual"),
                                     (suffix_valid_mask, sv, "suffix")):
        if supplied is not None and (supplied.dtype != torch.bool or supplied.shape != expected.shape):
            raise ValueError(f"{name} validity mask must be a sequence-length bool tensor")
    if visual_valid_mask is not None:
        vv = visual_valid_mask.to(query.device)
    if suffix_valid_mask is not None:
        sv = suffix_valid_mask.to(query.device)
    valid_geometry = sv[raters].all() & vv.any()
    if query.device.type == "cuda":
        torch._assert_async(valid_geometry, "raters must be valid and visual candidates nonempty")
    elif not bool(valid_geometry):
        raise ValueError("raters must be valid and visual candidates nonempty")
    hi = torch.tensor(head_ids, device=query.device)
    keys = torch.cat((system_key[hi], visual_key, suffix_key[hi]), dim=1)
    valid_keys = torch.cat((torch.ones(pn, dtype=torch.bool, device=query.device), vv, sv))
    per_head_sum = torch.zeros((len(head_ids), vn), dtype=torch.float32, device=query.device)
    result = None
    key_positions = torch.arange(keys.shape[1], device=query.device)
    for start in range(0, len(raters), query_block_size):
        rows = torch.tensor(raters[start:start + query_block_size], device=query.device)
        logits = torch.matmul(query[hi][:, rows], keys.transpose(1, 2)) * (d ** -0.5)
        _finite(logits, "QK logits before causal masking")
        allowed = ((key_positions[None, :] <= (pn + vn + rows[:, None]))
                   & valid_keys[None, :])
        probability = logits.float().masked_fill(~allowed[None], -torch.inf).softmax(dim=-1)
        visual_probability = probability[:, :, pn:pn + vn]
        if len(raters) <= query_block_size:
            result = visual_probability.mean(dim=1).mean(dim=0)
        else:
            per_head_sum += visual_probability.sum(dim=1)
    if result is None:
        result = (per_head_sum / len(raters)).mean(dim=0)
    _finite(result, "visual scores")
    return result


def exact_topk(scores: torch.Tensor, structural_ids: Iterable[int] = (),
               padding_ids: Iterable[int] = ()) -> torch.Tensor:
    """Exactly ceil(real rows / 4), ties by ascending original token index."""
    if scores.ndim != 1 or scores.numel() <= 0:
        raise ValueError("scores must be a nonempty visual vector")
    _finite(scores, "scores")
    structural = _ids(structural_ids, scores.numel(), "structural IDs")
    padding = _ids(padding_ids, scores.numel(), "padding IDs")
    if set(structural) & set(padding):
        raise ValueError("structural and padding IDs must be disjoint")
    valid = torch.ones(scores.numel(), device=scores.device, dtype=torch.bool)
    valid[structural + padding] = False
    real = torch.where(valid)[0]
    if real.numel() <= 0:
        raise ValueError("N_content must be positive")
    ranked = torch.argsort(scores[real], descending=True, stable=True)
    selected = real[ranked[:(real.numel() + 3) // 4]]
    return selected.sort().values


def canonical_chunk_plan(selected_ids: Iterable[int], v_num: int,
                         structural_ids: Iterable[int] = (),
                         padding_ids: Iterable[int] = (),
                         chunk_size: int = 64) -> dict:
    """Validate exact token budget and map identity positions to whole chunks."""
    if v_num <= 0 or chunk_size != 64:
        raise ValueError("canonical SSD contract requires positive rows and chunk_size=64")
    structural = _ids(structural_ids, v_num, "structural IDs")
    padding = _ids(padding_ids, v_num, "padding IDs")
    selected = _ids(selected_ids, v_num, "selected IDs")
    if set(structural) & set(padding):
        raise ValueError("structural and padding rows overlap")
    excluded = set(structural + padding)
    n_content = v_num - len(excluded)
    if n_content <= 0 or len(selected) != (n_content + 3) // 4:
        raise ValueError("selection violates exact ceil(N_content/4) budget")
    if set(selected) & excluded:
        raise ValueError("selection contains structural or padding rows")
    chunks = sorted({i // chunk_size for i in selected})
    spans = [(i * chunk_size, min((i + 1) * chunk_size, v_num)) for i in chunks]
    ranges = []
    for lo, hi in spans:
        if ranges and ranges[-1][1] == lo:
            ranges[-1][1] = hi
        else:
            ranges.append([lo, hi])
    read_rows = [i for lo, hi in spans for i in range(lo, hi)]
    real_read = sum(i not in excluded for i in read_rows)
    return {
        "N_content": n_content, "k": len(selected), "v_num": v_num,
        "chunk_size": chunk_size, "selected_tokens": selected,
        "selected_chunks": chunks, "chunk_spans": [list(x) for x in spans],
        "chunk_runs": ranges, "contiguous_runs": len(ranges),
        "read_rows": read_rows, "read_real_rows": real_read,
        "extra_real_rows": real_read - len(selected),
        "read_structural_rows": sum(i in set(structural) for i in read_rows),
        "read_padding_rows": sum(i in set(padding) for i in read_rows),
        "eof_short_chunk_rows": sum(hi - lo for lo, hi in spans if hi - lo < chunk_size),
        "keep_tokens": sorted(selected + structural),
        "structural_ids": structural, "padding_ids": padding,
    }

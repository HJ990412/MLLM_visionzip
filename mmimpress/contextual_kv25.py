"""Image-only, original-token Visual-KV25 selection for LLaVA-NeXT.

The functions in this module only build an image-level plan.  They never run a
model, average decoder K/V, inspect a question, or read an SSD store.  The
plan's first ``k`` physical rows are the selected original visual tokens.
"""
from __future__ import annotations

import hashlib
import math
import random
from numbers import Integral, Real
from typing import Any, Sequence

import torch

from mmimpress.cvpr25 import (visionzip_repack_order,
                              visual_kv_budget_count)


LAYOUT_POLICY = "visionzip_contextual_original_v1"
VARIANTS = frozenset(("dominant", "contextual", "random", "uniform"))


def _finite_matrix(value: Any, name: str) -> torch.Tensor:
    matrix = torch.as_tensor(value).detach().to(dtype=torch.float32,
                                                 device="cpu")
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must have shape (tokens, positive_dim)")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    return matrix


def normalize_descriptors(value: Any) -> torch.Tensor:
    """Return FP32 unit vectors, retaining exact zero vectors as zeros."""
    vectors = _finite_matrix(value, "descriptors")
    # Match the Turn-1 capture hook's FP32 L2 rule exactly.  A finite input
    # whose FP32 norm overflows is rejected rather than treated as zero.
    lengths = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    if not bool(torch.isfinite(lengths).all()):
        raise ValueError("descriptor L2 norm contains NaN or Inf")
    normalized = torch.where(
        lengths > 0,
        vectors / lengths.clamp_min(torch.finfo(vectors.dtype).tiny),
        torch.zeros_like(vectors),
    )
    if not bool(torch.isfinite(normalized).all()):
        raise ValueError("normalized descriptors contain NaN or Inf")
    return normalized


def vision_key_patch_descriptors(raw_k_proj: Any,
                                 num_heads: int) -> torch.Tensor:
    """Reduce one CLIP ``k_proj`` output to normalized patch descriptors.

    ``raw_k_proj`` has shape ``(sub_images, 1 + patches, heads * head_dim)``.
    The first token is CLS.  Head reduction is an FP32 arithmetic mean, then
    each patch vector is L2 normalized.  Zero vectors stay zero.
    """
    if isinstance(num_heads, bool) or not isinstance(num_heads, Integral) \
            or num_heads < 1:
        raise ValueError("num_heads must be a positive integer")
    raw = torch.as_tensor(raw_k_proj).detach()
    if raw.ndim != 3 or raw.shape[0] < 1 or raw.shape[1] < 2 \
            or raw.shape[2] < 1 or raw.shape[2] % num_heads:
        raise ValueError("invalid CLIP k_proj output shape")
    if not bool(torch.isfinite(raw).all()):
        raise ValueError("CLIP k_proj output contains NaN or Inf")
    patches = raw[:, 1:, :].float().reshape(
        raw.shape[0], raw.shape[1] - 1, num_heads, raw.shape[2] // num_heads)
    reduced = patches.mean(dim=2).cpu()
    return normalize_descriptors(reduced.reshape(-1, reduced.shape[-1])).view(
        raw.shape[0], raw.shape[1] - 1, reduced.shape[-1])


def anyres_token_descriptors(runner, per_sub_descriptors: Any,
                             image_size: Sequence[int] | torch.Tensor,
                             v_num: int, base_side: int | None = None
                             ) -> torch.Tensor:
    """Map CLIP patch vectors to LLaVA's expanded visual-token coordinates.

    The mapping follows the model's base image, tiled high-resolution crops,
    and ``unpad_image`` geometry.  Structural newline rows are zero vectors;
    their positions are separately determined by ``runner.anyres_layout``.
    Input vectors are copied without averaging across spatial tokens.
    """
    from transformers.models.llava_next.modeling_llava_next import (
        get_anyres_image_grid_shape, unpad_image)

    per_sub = torch.as_tensor(per_sub_descriptors).detach().to(
        dtype=torch.float32, device="cpu")
    if per_sub.ndim != 3 or per_sub.shape[2] < 1:
        raise ValueError("per_sub_descriptors must be (sub_images, patches, dim)")
    if not bool(torch.isfinite(per_sub).all()):
        raise ValueError("per_sub_descriptors contain NaN or Inf")
    if torch.is_tensor(image_size):
        size = [int(x) for x in image_size.detach().cpu().reshape(-1)]
    else:
        size = [int(x) for x in image_size]
    if len(size) != 2 or min(size) < 1:
        raise ValueError("image_size must contain positive height and width")
    if isinstance(v_num, bool) or not isinstance(v_num, Integral) or v_num < 1:
        raise ValueError("v_num must be a positive integer")
    cfg = runner.cfg
    vc = cfg.vision_config
    side = (int(base_side) if base_side is not None
            else int(vc.image_size // vc.patch_size))
    if side < 1:
        raise ValueError("base_side must be positive")
    nph, npw = get_anyres_image_grid_shape(
        size, cfg.image_grid_pinpoints, vc.image_size)
    nph, npw = int(nph), int(npw)
    if per_sub.shape[:2] != (1 + nph * npw, side * side):
        raise ValueError("per-sub-image patch descriptors disagree with AnyRes grid")

    dim = int(per_sub.shape[2])
    tiled = per_sub[1:].reshape(nph, npw, side, side, dim)
    tiled = tiled.permute(0, 2, 1, 3, 4).reshape(nph * side,
                                                 npw * side, dim)
    kept = unpad_image(tiled.permute(2, 0, 1).contiguous(), size)
    if kept.ndim != 3 or kept.shape[0] != dim:
        raise AssertionError("unpad_image changed descriptor channels")
    hi_h, hi_w = int(kept.shape[1]), int(kept.shape[2])
    if side * side + hi_h * (hi_w + 1) != v_num:
        raise ValueError("AnyRes descriptor geometry disagrees with visual span")

    mapped = torch.zeros((v_num, dim), dtype=torch.float32)
    mapped[:side * side] = per_sub[0]
    body = mapped[side * side:].view(hi_h, hi_w + 1, dim)
    body[:, :hi_w] = kept.permute(1, 2, 0)
    return mapped


def split_visual_kv_budget(n_content: int, alpha: float) -> tuple[int, int, int]:
    """Return ``(k, k_dominant, k_context)`` under the fixed 25% budget."""
    if isinstance(n_content, bool) or not isinstance(n_content, Integral) \
            or n_content < 0:
        raise ValueError("n_content must be a non-negative integer")
    if isinstance(alpha, bool) or not isinstance(alpha, Real) \
            or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1]")
    k = visual_kv_budget_count(n_content, .25)
    assert k == (n_content + 3) // 4
    contextual = math.floor(float(alpha) * k)
    return k, k - contextual, contextual


def uniform_target_ids(remaining_ids: Sequence[int], count: int) -> list[int]:
    """Choose mid-bin positions from original-ID sorted remaining tokens."""
    remaining = [int(x) for x in remaining_ids]
    if remaining != sorted(set(remaining)):
        raise ValueError("remaining_ids must be strictly increasing")
    if isinstance(count, bool) or not isinstance(count, Integral) \
            or not 0 <= count <= len(remaining):
        raise ValueError("invalid target count")
    if count == 0:
        return []
    size = len(remaining)
    targets = [remaining[((2 * j + 1) * size) // (2 * count)]
               for j in range(count)]
    if len(set(targets)) != count:
        raise AssertionError("uniform target rule repeated an ID")
    return targets


def contextual_representatives(remaining_ids: Sequence[int],
                               target_ids: Sequence[int],
                               descriptors: Any) -> dict[str, Any]:
    """Assign by cosine to targets and choose one original member per cluster."""
    remaining = [int(x) for x in remaining_ids]
    targets = [int(x) for x in target_ids]
    if remaining != sorted(set(remaining)) or targets != sorted(set(targets)) \
            or not targets or not set(targets).issubset(remaining):
        raise ValueError("invalid remaining IDs or target IDs")
    vectors = normalize_descriptors(descriptors)
    if remaining[-1] >= vectors.shape[0] or remaining[0] < 0:
        raise ValueError("descriptor rows do not cover remaining IDs")
    target_vectors = vectors[targets]
    similarities = vectors[remaining] @ target_vectors.T
    target_set = set(targets)
    assignment: dict[int, int] = {}
    clusters = {target: [] for target in targets}
    for row, original in enumerate(remaining):
        # Force every target into its own cluster even when an identical or
        # zero descriptor would otherwise tie with an earlier target.
        target = original if original in target_set else targets[int(
            torch.argmax(similarities[row]).item())]
        assignment[original] = target
        clusters[target].append(original)

    representatives = []
    for target in targets:
        members = clusters[target]  # ascending original IDs
        centroid = vectors[members].mean(dim=0)
        length = float(torch.linalg.vector_norm(centroid).item())
        if not math.isfinite(length):
            raise ValueError("cluster centroid has NaN or Inf norm")
        if length == 0:
            chosen = target
        else:
            cosine = vectors[members] @ (centroid / length)
            chosen = members[int(torch.argmax(cosine).item())]
        representatives.append(chosen)
    if len(set(representatives)) != len(targets):
        raise AssertionError("cluster representatives are not unique")
    return {
        "assignments": {str(original): target
                        for original, target in assignment.items()},
        "representative_ids": representatives,
        "cluster_sizes": [len(clusters[target]) for target in targets],
        "cluster_size_by_target": {str(target): len(clusters[target])
                                   for target in targets},
        "cluster_members": {str(target): clusters[target]
                            for target in targets},
    }


def _random_seed(image_id: str | int, seed: int) -> tuple[int, str]:
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    digest = hashlib.sha256(f"{int(seed)}\0{image_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big"), digest.hex()


def build_selection_plan(token_scores: Any, descriptors: Any,
                         newline_idx: Sequence[int], alpha: float,
                         variant: str, image_id: str | int,
                         seed: int = 1234) -> dict[str, Any]:
    """Build a full, deterministic original-token Visual-KV25 layout plan.

    ``descriptors`` has one row per original visual token, with zero rows at
    structural separators.  It is required only for ``contextual`` with a
    positive contextual count.  The method has no question-dependent input.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown selection variant: {variant!r}")
    scores = torch.as_tensor(token_scores, dtype=torch.float32).flatten().cpu()
    vn = int(scores.numel())
    separators = sorted(int(x) for x in newline_idx)
    if len(separators) != len(set(separators)) \
            or any(x < 0 or x >= vn for x in separators):
        raise ValueError("invalid structural separator IDs")
    n_content = vn - len(separators)
    k, k_dominant, k_context = split_visual_kv_budget(n_content, alpha)
    if variant == "dominant" and k_context != 0:
        raise ValueError("dominant variant requires k_context=0")

    # Reuse the frozen dominant-only rank, including stable original-ID ties.
    ranking = visionzip_repack_order(scores, separators)
    content_rank = ranking[:n_content]
    dominant = content_rank[:k_dominant]
    dominant_set = set(dominant)
    remaining = sorted(x for x in content_rank if x not in dominant_set)

    targets: list[int] = []
    contextual: list[int] = []
    assignments: dict[str, int] = {}
    cluster_sizes: list[int] = []
    cluster_size_by_target: dict[str, int] = {}
    cluster_members: dict[str, list[int]] = {}
    random_seed_digest: str | None = None
    random_seed_integer: int | None = None
    if k_context:
        if variant in ("contextual", "uniform"):
            targets = uniform_target_ids(remaining, k_context)
        if variant == "contextual":
            if descriptors is None:
                raise ValueError("contextual variant requires vision-key descriptors")
            vectors = _finite_matrix(descriptors, "descriptors")
            if vectors.shape[0] != vn:
                raise ValueError("descriptor row count disagrees with visual span")
            clusters = contextual_representatives(remaining, targets, vectors)
            contextual = clusters["representative_ids"]
            assignments = clusters["assignments"]
            cluster_sizes = clusters["cluster_sizes"]
            cluster_size_by_target = clusters["cluster_size_by_target"]
            cluster_members = clusters["cluster_members"]
        elif variant == "uniform":
            contextual = list(targets)
        elif variant == "random":
            random_seed_integer, random_seed_digest = _random_seed(image_id, seed)
            contextual = sorted(random.Random(random_seed_integer).sample(
                remaining, k_context))
        else:
            raise AssertionError("dominant variant has a contextual budget")

    selected_set = dominant_set | set(contextual)
    if len(selected_set) != k or len(set(contextual)) != k_context \
            or dominant_set.intersection(contextual):
        raise AssertionError("dominant/contextual selection violates fixed budget")
    selected_order = [x for x in content_rank if x in selected_set]
    unselected_order = [x for x in content_rank if x not in selected_set]
    order = selected_order + unselected_order + separators
    if len(order) != vn or sorted(order) != list(range(vn)):
        raise AssertionError("physical order is not a full permutation")
    inverse = [0] * vn
    for stored, original in enumerate(order):
        inverse[original] = stored
    if k_context == 0 and order != ranking:
        raise AssertionError("alpha=0 must equal dominant-only VisionZip order")
    return {
        "layout_policy": LAYOUT_POLICY,
        "selection_source": (
            "turn1_image_only_saliency_and_vision_keys"
            if variant == "contextual" and k_context > 0
            else "turn1_image_only_saliency"),
        "variant": variant,
        "alpha": float(alpha),
        "image_id": str(image_id),
        "n_content": n_content,
        "n_structural": len(separators),
        "k": k,
        "k_dominant": k_dominant,
        "k_context": k_context,
        "actual_dominant_fraction_of_content": (
            k_dominant / n_content if n_content else 0.0),
        "actual_context_fraction_of_content": (
            k_context / n_content if n_content else 0.0),
        "actual_content_retention_fraction": (k / n_content
                                               if n_content else 0.0),
        "content_saliency_ranking": content_rank,
        "dominant_ids": dominant,
        "remaining_ids": remaining,
        "contextual_ids": contextual,
        "target_ids": targets,
        "assignments": assignments,
        "representative_ids": contextual if variant == "contextual" else [],
        "cluster_sizes": cluster_sizes,
        "cluster_size_by_target": cluster_size_by_target,
        "cluster_members": cluster_members,
        "selected_original_ids": selected_order,
        "structural_original_ids": separators,
        "stored_to_original": order,
        "original_to_stored": inverse,
        "random_seed": int(seed) if variant == "random" else None,
        "random_seed_digest": random_seed_digest,
        "random_seed_integer": random_seed_integer,
    }

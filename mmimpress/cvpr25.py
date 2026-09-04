"""Storage-aware visual KV selection: static saliency + chunk-level ranking.

Root cause this module addresses (measured, see results/profile_sparsevlm.json
and results/diagnose.json): SparseVLM-style identification runs inside the LLM,
per layer, and produces a TOKEN-level selection; the storage system then has to
turn that into CHUNK reads.  At 25% token retention the touched chunk fraction
is 0.49, so half the bytes are read for tokens that were not selected, and the
per-layer identification machinery costs more than the I/O it saves.

The fix has two halves:
  * make importance REUSABLE across the questions of one image, so it is paid
    once per image instead of once per request;
  * rank STORAGE CHUNKS directly, so the budget is a byte budget and the
    touched fraction equals the budget by construction.

What is borrowed, and from where (exact files in
/home/dblab/hj/reference_repos/cvpr25):

  VisionZip  visionzip/clip_encoder.py:50-53
             cls_idx=0; cls_attention = attn_weights[:, :, 0, 1:];
             cls_attention_sum = cls_attention.sum(dim=1); topk(...)
             -> the penultimate CLIP layer's CLS-to-patch attention, summed
             over heads, is a query-independent per-patch saliency.  We use the
             SCORE but not the action: VisionZip prunes/merges the visual token
             sequence immediately, which would destroy the token-index-to-SSD
             -offset mapping our store depends on.  We keep every token and
             only rank chunks.
             visionzip/utils.py:89,124
             raw_key_states.mean(1) stored as layer.metric -> a per-patch
             descriptor; we reuse the same idea for chunk descriptors.

  PACT       transformers/PACT/utils.py:650-696 custom_pruning(k_image, q_image)
             importance from a small K.Q reduction over projections that the
             model already computes, averaged over heads, WITHOUT ever forming
             the N_q x N_k attention matrix (this is what makes it
             FlashAttention-compatible).  We keep that shape but push the image
             side offline: k_image is precomputed per chunk at store build time,
             so the online cost is one q_proj over the question tokens plus a
             (n_chunks x hidden) dot product.

  DivPrune   LLaVA/llava/model/llava_arch.py:151-170 DivPrune()
             greedy MaxMin selection on a cosine-distance matrix: repeatedly add
             the candidate whose minimum distance to the chosen set is largest.
             Applied here ONLY among chunks, and only to break ties / fill the
             tail of the budget -- running it over all visual tokens would
             deliberately spread the selection across the image and inflate the
             touched chunk fraction, which is the opposite of what we need.

  PyramidDrop  llava/model/modeling_llama_pdrop.py:414 materialises
             attn_weights = Q @ K^T inside the LLM to rank visual tokens at
             stage boundaries.  Referenced and NOT used: it requires the full
             attention matrix and eager attention, i.e. exactly the cost we are
             removing.  Its layer-progressive idea is also a poor fit here,
             because a KV store must decide what to READ before any layer runs.
"""
from __future__ import annotations

import torch

# --------------------------------------------------- VisionZip static score
@torch.no_grad()
def clip_cls_patch_saliency(runner, pixel_values, layer_from_end: int = 2):
    """Per-patch saliency from the CLIP encoder alone (no LLM involved).

    Returns (n_sub_images, num_patches) float32 on CPU.

    VisionZip reads `image_forward_outs.attentions[-2]` -- the PENULTIMATE
    encoder layer -- because the last layer's CLS row is already specialised for
    the contrastive objective.  We reproduce that indexing.  The vision tower is
    asked for attentions directly; nothing is hooked and the LLM never runs.
    """
    vt = runner.model.model.vision_tower
    px = pixel_values
    if px.dim() == 5:                      # (1, n_sub, 3, H, W)
        px = px[0]
    out = vt(px.to(vt.device, vt.dtype), output_attentions=True,
             output_hidden_states=False, return_dict=True)
    attn = out.attentions[-layer_from_end]           # (n_sub, heads, S, S)
    cls_attention = attn[:, :, 0, 1:]                # CLS row -> patches
    return cls_attention.sum(dim=1).float().cpu()    # sum over heads


def anyres_token_scores(runner, per_sub_scores, image_size, v_num,
                        base_side=None):
    """Map per-sub-image CLIP patch scores onto LLaVA-NeXT visual token order.

    The visual block is [base 24x24 patches | unpadded high-res grid + one
    row-separator per row].  The high-res sub-images tile a
    (num_patch_height x num_patch_width) grid, each contributing 24x24 patches,
    and the tiled grid is then unpadded to the image's aspect ratio.  This
    reproduces that geometry so a patch score lands on the token that carries
    the same patch, which is what lets the score be reused as chunk metadata.

    Returns (v_num,) float32; separator tokens get +inf (always retained).
    """
    from transformers.models.llava_next.modeling_llava_next import (
        get_anyres_image_grid_shape, unpad_image)
    cfg = runner.cfg
    vc = cfg.vision_config
    b = base_side or (vc.image_size // vc.patch_size)
    nph, npw = get_anyres_image_grid_shape(image_size, cfg.image_grid_pinpoints,
                                           vc.image_size)
    assert per_sub_scores.shape[0] == 1 + nph * npw, \
        (per_sub_scores.shape, nph, npw)
    assert per_sub_scores.shape[1] == b * b, per_sub_scores.shape

    scores = torch.full((v_num,), float("inf"), dtype=torch.float32)
    scores[:b * b] = per_sub_scores[0]                     # base image

    # tile the high-res sub-images into one (nph*b, npw*b) grid of scores
    hi = per_sub_scores[1:].view(nph, npw, b, b)
    grid = hi.permute(0, 2, 1, 3).reshape(nph * b, npw * b)
    kept = unpad_image(grid.unsqueeze(0), image_size)[0]    # (hi_h, hi_w)
    hi_h, hi_w = kept.shape
    assert b * b + hi_h * (hi_w + 1) == v_num, (b, hi_h, hi_w, v_num)
    body = scores[b * b:].view(hi_h, hi_w + 1)
    body[:, :hi_w] = kept
    body[:, hi_w] = float("inf")                           # row separators
    return scores


# ----------------------------------------------------- chunk-level ranking
def aggregate_chunk_scores(token_scores, chunk_size, mode="topk_mean",
                           alpha=0.7, top_frac=0.25):
    """Token scores -> one score per SSD chunk.

    A chunk is what actually gets read, so it should be judged by how much of
    the budget it earns as a whole.  `max` over-rewards a chunk holding a single
    hot patch; plain `mean` under-rewards a chunk holding a few very hot ones.
    The default blends them:

        alpha * mean(top `top_frac` scores in chunk) + (1-alpha) * mean(all)

    modes: "topk_mean" (default), "mean", "max", "sum".  Separator tokens carry
    +inf and are excluded from the statistics (they are always retained and are
    free), so a chunk is never selected merely for containing one.
    """
    n = token_scores.numel()
    nc = (n + chunk_size - 1) // chunk_size
    out = torch.zeros(nc, dtype=torch.float32)
    for c in range(nc):
        seg = token_scores[c * chunk_size:(c + 1) * chunk_size]
        seg = seg[torch.isfinite(seg)]
        if seg.numel() == 0:
            out[c] = 0.0
            continue
        if mode == "mean":
            out[c] = seg.mean()
        elif mode == "max":
            out[c] = seg.max()
        elif mode == "sum":
            out[c] = seg.sum()
        else:                                   # topk_mean
            k = max(1, int(round(top_frac * seg.numel())))
            out[c] = alpha * seg.topk(k).values.mean() + (1 - alpha) * seg.mean()
    return out


def select_chunks_by_budget(chunk_scores, budget_frac, force=None):
    """Top chunks under a BYTE budget, not a token budget.

    The unit of I/O is the chunk, so taking the top `budget_frac` of chunks
    makes the touched chunk fraction equal the budget by construction -- which
    is exactly what token-level top-k could not guarantee (measured 0.49 touched
    at a 0.25 token budget).

    force: chunk ids that must be included (e.g. chunks holding separators).
    """
    nc = chunk_scores.numel()
    k = max(1, min(nc, int(round(budget_frac * nc))))
    order = torch.argsort(chunk_scores, descending=True, stable=True)
    chosen = []
    seen = set()
    for c in (force or []):
        c = int(c)
        if c not in seen:
            chosen.append(c)
            seen.add(c)
    for c in order.tolist():
        if len(chosen) >= k:
            break
        if c not in seen:
            chosen.append(c)
            seen.add(c)
    return sorted(chosen[:max(k, len(force or []))])


# --------------------------------------------- PACT-inspired query correction
@torch.no_grad()
def image_chunk_keys(runner, v_hidden, chunk_size, layer_idx=0):
    """Precomputed per-chunk image KEY descriptor (offline, once per image).

    PACT's custom_pruning(k_image, q_image) scores tokens from the layer's key
    and query projections instead of from attention weights.  We keep the key
    side but compute it once, at store build time, on the pre-layer visual
    hidden states, and average it within each chunk:

        kdesc[c] = mean_{t in chunk c} k_proj(input_layernorm(v_hidden[t]))

    Returns (n_chunks, hidden) float16 on CPU.  No rotary embedding is applied
    to either side, so the online query is scored in the same space.
    """
    layer = runner.layers[layer_idx]
    dev = runner.model.device
    h = v_hidden.to(dev, torch.bfloat16).unsqueeze(0)
    k = layer.self_attn.k_proj(layer.input_layernorm(h))[0]     # (v_num, hid)
    n, hid = k.shape
    nc = (n + chunk_size - 1) // chunk_size
    pad = nc * chunk_size - n
    if pad:
        k = torch.cat([k, k.new_zeros(pad, hid)], dim=0)
    return k.view(nc, chunk_size, hid).mean(dim=1).to(torch.float16).cpu()


@torch.no_grad()
def query_chunk_scores(runner, text_emb, chunk_keys, n_head, layer_idx=0):
    """Online query correction: one q_proj plus a (n_chunks x hidden) dot.

    Mirrors PACT's reduction -- dot product between image keys and the query,
    averaged over heads and query positions -- but never builds an
    N_q x N_k attention matrix, so it is compatible with any attention backend
    and costs a few hundred microseconds.

    text_emb: (n_q, hidden) question token embeddings (already available).
    Returns (n_chunks,) float32 on CPU, z-normalised.
    """
    layer = runner.layers[layer_idx]
    dev = runner.model.device
    hid = text_emb.shape[-1]
    hd = hid // n_head
    q = layer.self_attn.q_proj(
        layer.input_layernorm(text_emb.to(dev, torch.bfloat16).unsqueeze(0)))[0]
    q = q.view(-1, n_head, hd)                                  # (n_q, H, hd)
    kk = chunk_keys.to(dev, torch.bfloat16).view(-1, n_head, hd)  # (C, H, hd)
    # (H, n_q, hd) x (H, hd, C) -> (H, n_q, C); mean over heads and queries
    s = torch.bmm(q.permute(1, 0, 2), kk.permute(1, 2, 0)) * (hd ** -0.5)
    s = s.float().mean(dim=(0, 1)).cpu()
    return (s - s.mean()) / s.std().clamp(min=1e-6)


def zscore(x):
    x = x.float()
    return (x - x.mean()) / x.std().clamp(min=1e-6)


# ------------------------------------------- DivPrune-inspired diversity
def maxmin_diverse(desc, k, seeds=None):
    """Greedy MaxMin selection on cosine distance (DivPrune's DivPrune()).

    Port of divprune LLaVA/llava/model/llava_arch.py:151-170: start from the
    seed set, then repeatedly add the candidate whose MINIMUM cosine distance to
    the already-selected set is largest.  Here `desc` holds CHUNK descriptors,
    not token descriptors -- selecting diverse tokens would scatter reads across
    the image, which is the failure mode this project is trying to remove.
    """
    n = desc.shape[0]
    d = desc.float()
    d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-6)
    dist = 1.0 - d @ d.t()
    chosen = list(seeds or [])
    if not chosen:
        chosen = [int(dist.sum(dim=1).argmax())]
    while len(chosen) < k:
        m = dist[torch.tensor(chosen, dtype=torch.long)].min(dim=0).values
        m[torch.tensor(chosen, dtype=torch.long)] = -1.0
        nxt = int(m.argmax())
        if m[nxt] < 0:
            break
        chosen.append(nxt)
    return sorted(chosen)


# ==================================================================== ablation
# One place that turns (static score, optional query score) into a chunk id
# list.  Everything downstream -- SSD read, cache reconstruction, masking -- is
# shared by every strategy, so a difference in the results can only come from
# WHICH chunks were chosen, never from how they were fetched.

STRATEGIES = ("static", "hybrid", "diverse", "static_diverse", "diverse_only",
              "random")


def _rng_for(seed, image_id, layer):
    """Deterministic per (seed, image, layer) generator.

    Random selection must not depend on the order requests happen to arrive,
    or repeated runs would disagree for reasons that have nothing to do with
    the method.
    """
    import zlib
    import numpy as np
    h = zlib.crc32(str(image_id).encode()) & 0xFFFFFFFF
    return np.random.RandomState((int(seed) * 1000003 + layer * 7919 + h)
                                 % (2 ** 31 - 1))


def budget_chunk_count(n_chunks, budget):
    """round(total_chunks * budget), clamped -- the one rounding convention.

    Shared by every strategy and every budget (25 / 37.5 / 50%), so all methods
    read exactly the same number of chunks and the comparison is about content,
    not volume.
    """
    return max(1, min(n_chunks, int(round(budget * n_chunks))))


def choose_chunks(mode, layer, static, budget, force=(), query_score=None,
                  lam_static=1.0, lam_query=1.0, diverse_frac=0.25,
                  seed=0, image_id=""):
    """Chunk ids for one layer.  Returns (cids, score_or_None).

    static          VisionZip image saliency only, top-k by chunk score
    hybrid          + PACT-shaped query correction
    diverse         hybrid, then DivPrune MaxMin over the tail
    static_diverse  static, then DivPrune MaxMin over the tail  (no query)
    diverse_only    DivPrune MaxMin alone, no saliency, no query
    random          uniform chunks, no scoring of any kind
    """
    chunk_scores = static["chunk_score"][layer]
    nc = chunk_scores.numel()
    k = budget_chunk_count(nc, budget)
    force = sorted({int(c) for c in force})

    if mode == "random":
        rng = _rng_for(seed, image_id, layer)
        pool = [c for c in range(nc) if c not in set(force)]
        take = max(0, k - len(force))
        picked = rng.choice(len(pool), size=min(take, len(pool)),
                            replace=False) if take else []
        return sorted(force + [pool[int(i)] for i in picked]), None

    if mode == "diverse_only":
        # no saliency at all: start MaxMin from the chunk that is furthest from
        # the rest and keep adding the most dissimilar chunk
        cids = maxmin_diverse(static["chunk_keys"][layer], k,
                              seeds=force or None)
        return sorted(cids), None

    score = lam_static * zscore(chunk_scores)
    if query_score is not None and mode in ("hybrid", "diverse"):
        score = score + lam_query * query_score
    cids = select_chunks_by_budget(score, budget, force=force)

    if mode in ("diverse", "static_diverse"):
        keep_n = max(1, int(round((1 - diverse_frac) * len(cids))))
        seeds = sorted(cids, key=lambda c: -float(score[c]))[:keep_n]
        cids = maxmin_diverse(static["chunk_keys"][layer], len(cids),
                              seeds=seeds)
    return sorted(cids), score

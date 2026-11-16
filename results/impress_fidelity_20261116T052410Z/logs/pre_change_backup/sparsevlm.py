"""SparseVLM visual-token importance -- the replacement for IMPRESS's H2O metric.

IMPRESS (FAST'25, 4.1) scores a prefix token by the column sum of the
post-softmax attention matrix (H2O).  For a text prefix every column is a word,
so summing over all query rows is sensible.  For an image prefix that metric is
wrong in two ways: (1) most query rows are *other visual tokens*, so the score
is dominated by intra-image attention that says nothing about the question, and
(2) it has no notion of which text tokens actually interrogate the image.

SparseVLM fixes exactly that.  Two steps, both ported here:

  1. Rater selection.  Not every text token is a useful judge of visual
     relevance ("the", "?" are not).  SparseVLM picks *raters*: with v_t the
     visual hidden states and t_t the text hidden states,

         m_v_t = softmax(v_t @ t_t^T, dim=text).mean(dim=visual)     -> (t,)
         raters = { text tokens : m_v_t > mean(m_v_t) }

     computed ONCE per request on the pre-layer input hidden states.

  2. Visual scoring.  Take the rater ROWS x visual COLUMNS block of the
     post-softmax attention and average over the rater axis.

DELIBERATE DEVIATION: SparseVLM averages over heads before ranking.  We do not.
IMPRESS identifies important tokens per head and only then unifies them through
probe-head Jaccard voting (4.3), so collapsing the head axis first would
destroy the signal that machinery consumes.  Head collapse is available as
`head_reduce="mean"` for an apples-to-apples SparseVLM ablation.

Not ported (SparseVLM extras, orthogonal to prefix-KV storage):
  * token recycling / merging of pruned tokens -- would require writing merged
    KV back into the store; noted in docs/DESIGN.md as future work.
  * text-visual attention gravity (RoPE) correction from SparseVLM+ V2.
"""
from __future__ import annotations

import math

import torch


# ------------------------------------------------------------ 1. raters
def select_raters(hidden_states: torch.Tensor, v_start: int, v_num: int):
    """Text tokens that judge visual relevance (SparseVLM 3.2).

    hidden_states: (s, h) or (1, s, h) -- the model input embeddings, i.e.
    BEFORE the decoder layers, matching SparseVLM's placement.
    Returns ABSOLUTE sequence indices, ascending.  If no token beats the mean
    (degenerate, all-equal case) every text token becomes a rater.
    """
    h = hidden_states
    if h.dim() == 2:
        h = h.unsqueeze(0)
    assert h.dim() == 3 and h.shape[0] == 1, tuple(hidden_states.shape)
    t_start = v_start + v_num
    assert t_start < h.shape[1], "no text tokens after the visual block"

    h = h.float()
    v_t = h[:, v_start:t_start, :]                  # (1, v, d)
    t_t = h[:, t_start:, :]                         # (1, t, d)
    m_v_t = (v_t @ t_t.transpose(1, 2)).softmax(dim=2).mean(dim=1)   # (1, t)
    local = torch.where(m_v_t > m_v_t.mean())[1]
    if local.numel() == 0:
        local = torch.arange(t_t.shape[1], device=h.device)
    return local + t_start


# ---------------------------------------------------------- 2. scoring
def rater_visual_scores(attn: torch.Tensor, rater_rows, v_start: int,
                        v_num: int, head_reduce: str = "none"):
    """Per-head visual-token importance from the rater rows of `attn`.

    attn: post-softmax weights, (H, q, k) or (1, H, q, k).
    rater_rows: indices into the QUERY axis of `attn`.  When the query axis is
        the whole sequence these are the absolute indices from select_raters();
        when it covers only the suffix (prefix-reuse prefill) the caller must
        rebase them first.
    Returns (H, v_num) scores, or (v_num,) with head_reduce="mean" (the
    original SparseVLM behaviour).  Visual indices are LOCAL (0..v_num-1).
    """
    if attn.dim() == 4:
        assert attn.shape[0] == 1, "batch 1 only"
        attn = attn[0]
    assert attn.dim() == 3, tuple(attn.shape)
    r = torch.as_tensor(rater_rows, dtype=torch.long, device=attn.device)
    assert r.numel() > 0, "empty rater set"
    s = attn.float()[:, r, v_start:v_start + v_num].mean(dim=1)   # (H, v_num)
    return s.mean(dim=0) if head_reduce == "mean" else s


def rater_visual_scores_from_qk(q, k, rater_rows, v_start, v_num,
                                head_reduce="none", causal_from=None):
    """Same score without materialising the full attention matrix.

    q: (H, s, hd) rotated queries, k: (H, s, hd) rotated keys of the SAME rows
    the attention would see.  Only the rater rows are softmaxed, so cost is
    O(|raters| x s) instead of O(s^2) -- this is what the serving path uses.
    causal_from: index in the key axis where suffix keys begin; rater rows are
    positions within that suffix, so keys past a rater are masked out.
    """
    r = torch.as_tensor(rater_rows, dtype=torch.long, device=q.device)
    logits = (q[:, r] @ k.transpose(1, 2)) * (q.shape[-1] ** -0.5)  # (H,R,s)
    if causal_from is not None:
        # keys at/after `causal_from` belong to the suffix the rater itself
        # lives in, so a rater may only attend to suffix keys up to its own
        # position; prefix keys are always visible.
        n_new = k.shape[1] - causal_from
        j = torch.arange(n_new, device=k.device).view(1, 1, -1)
        row = r.view(1, -1, 1)
        logits[:, :, causal_from:] = torch.where(
            j <= row, logits[:, :, causal_from:], float("-inf"))
    p = logits.float().softmax(dim=-1)
    s = p[:, :, v_start:v_start + v_num].mean(dim=1)
    return s.mean(dim=0) if head_reduce == "mean" else s


# --------------------------------------------------------- 3. selection
def topk_budget(n_candidates: int, ratio: float) -> int:
    return max(1, min(n_candidates, math.ceil(ratio * n_candidates)))


def select_topk(scores: torch.Tensor, ratio: float, forbid=None):
    """Top-k over the last (visual) axis, independently per head.

    forbid: LOCAL visual indices excluded from the ranking (LLaVA's row
    separators -- structural tokens that are always retained, so they must not
    consume budget).  Returns LOCAL indices, shape scores.shape[:-1] + (k,).
    """
    s = scores.clone()
    if forbid is not None and len(forbid) > 0:
        idx = torch.as_tensor(list(forbid), dtype=torch.long, device=s.device)
        s.index_fill_(-1, idx, float("-inf"))
        k = topk_budget(s.shape[-1] - idx.numel(), ratio)
    else:
        k = topk_budget(s.shape[-1], ratio)
    return s.topk(k, dim=-1).indices


# ------------------------------- 4. SparseVLM adaptive per-layer ratio
def redundancy_ratio(logits: torch.Tensor) -> float:
    """SparseVLM's rank-based redundancy estimate for one layer.

    SparseVLM sets the per-layer sparsification level from how low-rank the
    text->visual attention logit matrix is: a near-rank-deficient block means
    the raters cannot tell the visual tokens apart, i.e. the layer is redundant
    and can be pruned harder.  Returned value is 1 - rank/min(dim) in [0, 1).

    APPROXIMATION: the paper does not fix a rank estimator; we use torch's
    tolerance-based matrix_rank on the (raters x visual) logits.  Off by
    default -- the main path uses IMPRESS's fixed retention ratio so the
    comparison against the text baseline stays controlled.
    """
    m = logits.float()
    if m.dim() > 2:
        m = m.reshape(-1, m.shape[-1])
    r = int(torch.linalg.matrix_rank(m))
    return 1.0 - r / max(1, min(m.shape))


def adaptive_ratio(base_ratio: float, redundancy: float,
                   lo: float = 0.5, hi: float = 1.5) -> float:
    """Scale the retention ratio by how non-redundant the layer looks."""
    scale = lo + (hi - lo) * (1.0 - redundancy)
    return float(min(1.0, max(1e-3, base_ratio * scale)))

"""Serving: identify-then-load prefill over a disk-resident image prefix.

Per request the flow is IMPRESS's 4.1 dataflow with an image as the prefix:

  1. the image id selects the stored prefix (image prefixes are either fully
     shared or not shared at all, so IMPRESS's radix tree degenerates to a
     lookup -- there is no R/NR split to compute);
  2. SparseVLM raters are chosen once, from the stored visual hidden states and
     the question's token embeddings (no vision tower on the critical path);
  3. per layer: read ONLY the probe heads' keys, score, Jaccard-vs-threshold,
     then either load the consensus tokens' chunks for every head, or fall back
     to loading the layer in full and selecting per head;
  4. loaded rows are scattered into a GPU prefix cache and everything not
     loaded is masked out, so the attention the model runs is exactly the
     attention the loaded bytes support;
  5. decoding proceeds with the masks frozen (IMPRESS leaves decoding alone).

Modes: "impress" (the above), "fullload" (read every chunk, no masking -- the
AS-like baseline) and "recompute" (no store at all, prefill from pixels).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
import transformers.models.llama.modeling_llama as ml
from transformers import StoppingCriteria

from mmimpress import reorder as ro
from mmimpress import sparsevlm as sv
from mmimpress.config import ALPHA, PROBE_HEADS, RETENTION_RATIO
from mmimpress.store import ChunkReader, IOCounter, chunks_for_tokens, load_meta

# ------------------------------------------------- per-layer additive bias
_ORIG_EAGER = ml.eager_attention_forward
BIAS = {}


class _FirstTokenTimestamp(StoppingCriteria):
    """Observe generation's first selected token without stopping it."""

    def __init__(self):
        self.at = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.at is None:
            # Materialize the selected token on the host, matching the explicit
            # ``int(argmax)`` boundary in the stored-KV paths.
            _ = int(input_ids[0, -1])
            torch.cuda.synchronize()
            self.at = time.perf_counter()
        return torch.zeros(input_ids.shape[0], dtype=torch.bool,
                           device=input_ids.device)


def _eager_with_bias(module, query, key, value, attention_mask, **kw):
    b = BIAS.get(getattr(module, "layer_idx", None))
    if b is not None:
        klen = key.shape[-2]
        b = F.pad(b, (0, klen - b.shape[-1]))     # suffix keys stay unmasked
        attention_mask = b if attention_mask is None else \
            attention_mask + b.to(attention_mask.dtype)
    return _ORIG_EAGER(module, query, key, value, attention_mask, **kw)


ml.eager_attention_forward = _eager_with_bias


def _min_val(dtype):
    return torch.finfo(dtype).min


# ------------------------------------------------------------ prefix cache
class PrefixCache:
    """GPU-resident prefix K/V that the model actually reads.

    DynamicCache.update() COPIES the tensors it is handed, so seeding a cache
    and then mutating the originals is a silent no-op -- the model would attend
    to zeros.  new_request() therefore seeds the cache first and then binds
    self.k/self.v to the cache's own tensors, so every later write from the
    per-layer selector lands where attention will read it.

    System-prompt rows are real and always present; visual rows start as zeros
    and are filled only where the selector actually reads bytes.  Unfilled rows
    are always masked, so their contents never reach the softmax.
    """

    def __init__(self, meta, sys_kv, device, dtype=torch.bfloat16):
        self.meta = meta
        self.device = device
        self.dtype = dtype
        self.sys_kv = sys_kv
        self.v_start = meta["v_token_start"]
        self.shape = (1, meta["num_heads"], meta["prefix_len"],
                      meta["head_dim"])
        self.k = self.v = None

    def new_request(self):
        """Fresh zero-filled cache with the system prompt restored."""
        from transformers import DynamicCache
        c = DynamicCache()
        scratch = torch.zeros(self.shape, device=self.device, dtype=self.dtype)
        L = self.meta["num_layers"]
        for li in range(L):
            c.update(scratch, scratch, li)     # copied, so one scratch is enough
        del scratch
        self.k = [c.layers[li].keys for li in range(L)]
        self.v = [c.layers[li].values for li in range(L)]
        for li in range(L):
            self.k[li][0, :, :self.v_start] = self.sys_kv["k"][li].to(
                self.device, self.dtype)
            self.v[li][0, :, :self.v_start] = self.sys_kv["v"][li].to(
                self.device, self.dtype)
        return c

    def write(self, layer, kind, rows, vals):
        """rows: stored visual positions; vals: (n, num_heads, head_dim)."""
        buf = (self.k if kind == "k" else self.v)[layer]
        idx = rows.to(self.device) + self.v_start
        buf[0, :, idx, :] = vals.to(self.device, self.dtype).permute(1, 0, 2)

    def write_full(self, layer, kind, block):
        """block: (v_num, num_heads, head_dim) as stored."""
        buf = (self.k if kind == "k" else self.v)[layer]
        buf[0, :, self.v_start:, :] = block.to(self.device,
                                               self.dtype).permute(1, 0, 2)


# -------------------------------------------------------------- selection
def similarity_threshold(k_keep, n, alpha=ALPHA):
    """IMPRESS 4.3: t = j ** alpha with j = E(Jaccard) of two random picks."""
    r = k_keep / n
    return (r / (2.0 - r)) ** alpha


def mean_pairwise_jaccard(masks):
    P = masks.shape[0]
    vals = []
    for a in range(P):
        for b in range(a + 1, P):
            inter = (masks[a] & masks[b]).sum().float()
            union = (masks[a] | masks[b]).sum().float().clamp(min=1)
            vals.append(inter / union)
    return float(torch.stack(vals).mean()) if vals else 1.0


def contiguous_runs(indices):
    """Number and lengths of maximal consecutive runs in integer indices.

    QA-Select ranks logical tokens, but the SSD serves token chunks.  Keeping
    this small storage-locality primitive independent of the model makes the
    physical-I/O contract directly unit-testable.
    """
    values = sorted({int(value) for value in indices})
    if not values:
        return 0, []
    lengths = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            lengths.append(previous - start + 1)
            start = value
        previous = value
    lengths.append(previous - start + 1)
    return len(lengths), lengths


def qa_select_plan(scores, ratio, separators, v_num, chunk_size):
    """Fixed-budget SparseVLM-style logical selection and physical plan.

    ``scores`` may be per-head ``(H, V)`` or already head-reduced ``(V,)``.
    SparseVLM's original head-mean ranking is used.  Structural separators are
    always retained through their sidecar and therefore neither consume the
    25% spatial-token budget nor force their containing normal chunks to be
    read.  Returned token positions are in the store's physical coordinate
    system; QA-Select validates that this system is canonical/raster order.
    """
    value = torch.as_tensor(scores)
    if value.dim() == 2:
        value = value.mean(dim=0)
    assert value.dim() == 1 and value.numel() == int(v_num), value.shape
    separator_list = sorted({int(item) for item in separators})
    assert all(0 <= item < int(v_num) for item in separator_list)
    selected = sv.select_topk(value, float(ratio), forbid=separator_list)
    selected = selected.reshape(-1)
    selected_cpu = sorted(int(item) for item in selected.detach().cpu())
    assert not (set(selected_cpu) & set(separator_list))
    chunks = chunks_for_tokens(selected_cpu, int(v_num), int(chunk_size))
    run_count, run_lengths = contiguous_runs(chunks)
    return {
        "selected_tokens": selected_cpu,
        "selected_chunks": chunks,
        "contiguous_runs": int(run_count),
        "contiguous_run_lengths": run_lengths,
    }


def mean_valid_spatial_chunk_scores(scores, separators, v_num, chunk_size):
    """Aggregate query-dependent token importance in physical SSD chunks.

    QA-Chunk uses exactly the same SparseVLM token importance as QA-Select and
    changes only the selection unit.  Structural row separators are supplied
    by a sidecar, so they contribute neither to a chunk's numerator nor its
    denominator.  Padding in the final physical chunk is handled in the same
    way: only real, non-structural visual rows enter the mean.

    Returns ``(chunk_scores, valid_counts)`` on the input device.  A chunk with
    no spatial rows receives ``-inf`` and can therefore never win Top-k.
    """
    value = torch.as_tensor(scores)
    if value.dim() == 2:
        value = value.mean(dim=0)
    assert value.dim() == 1 and value.numel() == int(v_num), value.shape
    v_num, chunk_size = int(v_num), int(chunk_size)
    assert v_num > 0 and chunk_size > 0
    separator_list = sorted({int(item) for item in separators})
    assert all(0 <= item < v_num for item in separator_list)

    valid = torch.ones(v_num, dtype=torch.bool, device=value.device)
    if separator_list:
        valid[torch.as_tensor(separator_list, dtype=torch.long,
                              device=value.device)] = False
    n_chunks = (v_num + chunk_size - 1) // chunk_size
    pad = n_chunks * chunk_size - v_num
    weighted = torch.where(valid, value.float(), torch.zeros(
        (), dtype=torch.float32, device=value.device))
    if pad:
        weighted = F.pad(weighted, (0, pad), value=0.0)
        valid = F.pad(valid, (0, pad), value=False)
    sums = weighted.view(n_chunks, chunk_size).sum(dim=1)
    counts = valid.view(n_chunks, chunk_size).sum(dim=1)
    chunk_scores = sums / counts.clamp(min=1).to(sums.dtype)
    chunk_scores = chunk_scores.masked_fill(counts == 0, float("-inf"))
    return chunk_scores, counts


def select_qa_chunks_ours_budget(chunk_scores, valid_counts, ratio=0.25):
    """Stable Top-k chunks with the exact rounding used by Ours Prefix25.

    In particular this intentionally does *not* use SparseVLM's ceil-based
    token budget.  ``budget_chunk_count`` uses ``round(n_chunks * ratio)``;
    sharing that helper is what makes QA-Chunk's physical K/V chunk count
    identical to Ours for every image.
    """
    from mmimpress.cvpr25 import budget_chunk_count

    values = torch.as_tensor(chunk_scores).float()
    counts = torch.as_tensor(valid_counts, device=values.device)
    assert values.dim() == counts.dim() == 1
    assert values.numel() == counts.numel() and values.numel() > 0
    k = budget_chunk_count(int(values.numel()), float(ratio))
    rankable = values.masked_fill(counts <= 0, float("-inf"))
    # Stable descending order gives the lower physical chunk id precedence for
    # exact ties, matching the deterministic tie policy used by cvpr25.py.
    order = torch.argsort(rankable, descending=True, stable=True)
    # Filter, rather than merely assigning -inf, so a malformed layout with
    # fewer than k spatial-bearing chunks can never return a structural-only
    # chunk.  The selector's exact-budget assertion then fails closed.
    order = order[counts.index_select(0, order) > 0]
    return order[:k]


def qa_chunk_plan(scores, ratio, separators, v_num, chunk_size):
    """Pure QA-Chunk score/selection plan used by CPU contract tests."""
    chunk_scores, counts = mean_valid_spatial_chunk_scores(
        scores, separators, v_num, chunk_size)
    selected = select_qa_chunks_ours_budget(chunk_scores, counts, ratio)
    ranked = [int(item) for item in selected.detach().cpu().reshape(-1)]
    selected_chunks = sorted(ranked)
    run_count, run_lengths = contiguous_runs(selected_chunks)
    return {
        "chunk_scores": chunk_scores.detach().cpu().tolist(),
        "valid_spatial_counts": counts.detach().cpu().tolist(),
        "selected_chunks_ranked": ranked,
        "selected_chunks": selected_chunks,
        "contiguous_runs": int(run_count),
        "contiguous_run_lengths": [int(item) for item in run_lengths],
    }


def bias_from_keep(keep, meta, device, dtype=torch.bfloat16):
    """keep: (v_num,) or (H, v_num) bool -> additive (1, h, 1, prefix_len)."""
    if keep.dim() == 1:
        keep = keep.unsqueeze(0)
    h = keep.shape[0]
    b = torch.zeros(1, h, 1, meta["prefix_len"], device=device, dtype=dtype)
    v0, vn = meta["v_token_start"], meta["v_token_num"]
    b[0, :, 0, v0:v0 + vn] = torch.where(
        keep.to(device), torch.zeros((), dtype=dtype, device=device),
        torch.tensor(_min_val(dtype), dtype=dtype, device=device))
    return b


class LayerSelector:
    """Forward pre-hooks that do the per-layer identify-then-load step."""

    def __init__(self, runner, ctx, rater_rows, ratio=RETENTION_RATIO,
                 probe=PROBE_HEADS, alpha=ALPHA, mode="impress",
                 counter=None):
        self.runner = runner
        self.ctx = ctx
        self.rater_rows = rater_rows          # suffix-relative
        self.ratio = ratio
        self.probe = probe
        self.alpha = alpha
        self.mode = mode
        self.io = counter if counter is not None else IOCounter()
        self.log = []
        self.done = set()
        self.hook_seconds = 0.0
        self._handles = []
        m = ctx.meta
        self.k_keep = sv.topk_budget(m["n_spatial"], ratio)
        self.thr = similarity_threshold(self.k_keep, m["n_spatial"], alpha)
        # Everything below works in STORED positions.  Attention is invariant
        # to the order of the keys it attends over (RoPE is already baked into
        # the stored K), so a reordered store needs no un-permute on the
        # serving path -- only the structural separators have to be located.
        nl = m.get("newline_stored", m["newline_idx"])
        self._nl_per_layer = bool(nl) and isinstance(nl[0], list)
        self._nl = ([torch.tensor(x, dtype=torch.long) for x in nl]
                    if self._nl_per_layer
                    else torch.tensor(nl, dtype=torch.long))

    def nl(self, layer):
        """Stored positions of the structural separators for one layer."""
        return self._nl[layer] if self._nl_per_layer else self._nl

    def __enter__(self):
        for li, layer in enumerate(self.runner.layers):
            self._handles.append(layer.register_forward_pre_hook(
                self._hook(li, layer), with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()

    # ------------------------------------------------------------ helpers
    def _keep_mask(self, layer, sel_stored, n_heads=None):
        """Stored-position selection (+ separators) -> bool keep mask."""
        m = self.ctx.meta
        shape = ((n_heads, m["v_token_num"]) if n_heads
                 else (m["v_token_num"],))
        keep = torch.zeros(shape, dtype=torch.bool)
        keep.scatter_(-1, sel_stored.cpu(), True)
        keep[..., self.nl(layer)] = True
        return keep

    def _load_rows(self, li, stored_positions):
        """Read the chunks covering `stored_positions` -- one contiguous span
        per chunk, all heads at once -- into the prefix cache."""
        m = self.ctx.meta
        cids = chunks_for_tokens(stored_positions, m["v_token_num"],
                                 m["chunk_size"])
        for kind in ("k", "v"):
            rows, vals = self.ctx.reader.read_chunks(li, kind, cids, self.io)
            self.ctx.cache.write(li, kind, rows, vals)
        return cids

    def _load_layer_full(self, li):
        for kind in ("k", "v"):
            block = self.ctx.reader.read_full(li, kind, self.io)
            self.ctx.cache.write_full(li, kind, block)

    # --------------------------------------------------------------- hook
    def _hook(self, li, layer):
        def hook(module, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            if h.shape[1] == 1 or li in self.done:
                return                     # decoding: selection stays frozen
            t0 = time.perf_counter()
            self._run(li, layer, h, kwargs)
            self.done.add(li)
            self.hook_seconds += time.perf_counter() - t0
        return hook

    def _run(self, li, layer, h, kwargs):
        m, ctx = self.ctx.meta, self.ctx
        dev = h.device
        H, hd = m["num_heads"], m["head_dim"]
        v0, vn = m["v_token_start"], m["v_token_num"]
        n = h.shape[1]

        if self.mode == "fullload":
            self._load_layer_full(li)
            self.log.append({"layer": li, "mode": "full", "sim": None,
                             "chunks": m["n_chunks_per_layer"]})
            return

        attn = layer.self_attn
        hn = layer.input_layernorm(h)
        q = attn.q_proj(hn).view(1, n, H, hd).transpose(1, 2)
        kn = attn.k_proj(hn).view(1, n, H, hd).transpose(1, 2)
        cos, sin = kwargs["position_embeddings"]
        q, kn = ml.apply_rotary_pos_emb(q, kn, cos, sin)
        q, kn = q[0], kn[0]                                  # (H, n, hd)

        P = min(self.probe, H)
        sys_k = ctx.cache.k[li][0, :, :v0]                   # (H, v0, hd)

        # --- probe phase: only the probe heads' visual keys leave the disk ---
        pk = ctx.reader.read_probe(li, self.io).to(dev, q.dtype) \
            .permute(1, 0, 2)[:P]                            # (P, v_num, hd)
        keys_p = torch.cat([sys_k[:P], pk, kn[:P]], dim=1)
        sp = sv.rater_visual_scores_from_qk(
            q[:P], keys_p, self.rater_rows, v0, vn, causal_from=v0 + vn)
        sp[:, self.nl(li).to(dev)] = float("-inf")           # separators are free
        top = sp.topk(self.k_keep, dim=-1).indices
        masks = torch.zeros(P, vn, dtype=torch.bool, device=dev)
        masks.scatter_(1, top, True)
        sim = mean_pairwise_jaccard(masks) if P >= 2 else 1.0

        if sim > self.thr:
            votes = masks.sum(0).float()
            big = 10.0 * (float(sp.max()) + 1.0)
            sel = (votes * big + sp.mean(0)).topk(self.k_keep).indices
            cids = self._load_rows(li,
                                   sel.tolist() + self.nl(li).tolist())
            keep = self._keep_mask(li, sel)
            self.log.append({"layer": li, "mode": "probe", "sim": sim,
                             "chunks": len(cids)})
        else:
            # fallback (IMPRESS 4.3 step 6): all heads' keys, per-head choice
            self._load_layer_full(li)
            kv = ctx.cache.k[li][0, :, v0:]                  # stored order
            keys_a = torch.cat([sys_k, kv, kn], dim=1)
            sa = sv.rater_visual_scores_from_qk(
                q, keys_a, self.rater_rows, v0, vn, causal_from=v0 + vn)
            sa[:, self.nl(li).to(dev)] = float("-inf")
            top_h = sa.topk(self.k_keep, dim=-1).indices     # (H, k)
            keep = self._keep_mask(li, top_h, n_heads=H)
            self.log.append({"layer": li, "mode": "fallback", "sim": sim,
                             "chunks": m["n_chunks_per_layer"]})

        BIAS[li] = bias_from_keep(keep, m, dev)

    # ------------------------------------------------------------ summary
    def stats(self):
        modes = [r["mode"] for r in self.log]
        n = max(1, len(modes))
        sims = [r["sim"] for r in self.log if r["sim"] is not None]
        nc = self.ctx.meta["n_chunks_per_layer"]
        ch = [r.get("chunks", nc) for r in self.log]
        is_full = self.mode == "fullload"
        return {"touched_chunk_fraction": (1.0 if is_full else
                                             float(np.mean(ch)) / nc),
                "logical_kv_ratio": (1.0 if is_full else self.k_keep
                                      / self.ctx.meta["v_token_num"]),
                "fallback_rate": modes.count("fallback") / n,
                "probe_rate": modes.count("probe") / n,
                "mean_jaccard": (sum(sims) / len(sims)) if sims else None,
                "threshold": self.thr, "k_keep": self.k_keep,
                "hook_ms": self.hook_seconds * 1e3}


class QASelectLayerSelector:
    """SparseVLM-based per-query Top-k selection over a raster SSD store.

    This is intentionally narrower than the historical :class:`LayerSelector`:
    every decoder layer reads the configured probe heads, averages their
    rater-to-visual scores as SparseVLM does, keeps exactly the fixed spatial
    token budget, and reads precisely the unique normal chunks containing
    those logical tokens.  There is no probe-head voting, Jaccard threshold,
    adaptive budget, or full-layer fallback.

    Extra rows brought in by a touched chunk are written to the reconstruction
    buffer but remain masked.  Thus logical retention and physical SSD traffic
    are both honest and independently measurable.
    """

    def __init__(self, runner, ctx, rater_rows, ratio=0.25, probe=PROBE_HEADS,
                 counter=None):
        self.runner = runner
        self.ctx = ctx
        self.rater_rows = rater_rows
        self.ratio = float(ratio)
        self.probe = int(probe)
        self.io = counter if counter is not None else IOCounter()
        self.log = []
        self.done = set()
        self._handles = []
        self._cpu_ms = {
            "query_projection": 0.0,
            "probe_h2d": 0.0,
            "query_scoring": 0.0,
            "topk": 0.0,
            "selected_id_d2h": 0.0,
            "chunk_planning": 0.0,
            "probe_read_pipeline": 0.0,
            "chunk_io": 0.0,
            "scatter": 0.0,
        }
        self._cuda_events = {key: [] for key in (
            "query_projection", "probe_h2d", "query_scoring", "topk",
            "scatter")}
        self.hook_seconds = 0.0
        self.decision_host_seconds = 0.0
        self.query_score_calls = 0
        m = ctx.meta
        assert math.isclose(self.ratio, 0.25, rel_tol=0.0, abs_tol=1e-12), \
            f"QA-Select25 requires fixed ratio 0.25, got {self.ratio}"
        # This implementation and on-disk format target the repository's
        # Vicuna/LLaMA MHA model.  Under GQA, q_proj and k_proj have different
        # head counts and silently viewing both with the stored KV-head count
        # would be wrong; fail explicitly instead of implying portability.
        configured_q_heads = int(getattr(runner, "n_heads", m["num_heads"]))
        text_config = getattr(getattr(runner, "cfg", None), "text_config", None)
        configured_kv_heads = int(getattr(
            text_config, "num_key_value_heads", configured_q_heads))
        assert configured_q_heads == configured_kv_heads == int(m["num_heads"]), \
            ("QA-Select currently requires MHA with matching stored/query/KV "
             f"heads, got q={configured_q_heads}, kv={configured_kv_heads}, "
             f"stored={m['num_heads']}")
        assert 0 < self.probe <= int(m["probe_heads"]), \
            (self.probe, m["probe_heads"])
        self.k_keep = sv.topk_budget(int(m["n_spatial"]), self.ratio)
        self._nl = [torch.tensor(ctx.separator_positions(li), dtype=torch.long)
                    for li in range(int(m["num_layers"]))]
        self._separator_blob = None

    def __enter__(self):
        for li, layer in enumerate(self.runner.layers):
            self._handles.append(layer.register_forward_pre_hook(
                self._hook(li, layer), with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()

    def _timed(self, name, device, operation):
        """Time GPU work with events without adding per-stage synchronizes."""
        device = torch.device(device)
        if device.type == "cuda" and torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream(device))
            value = operation()
            end.record(torch.cuda.current_stream(device))
            self._cuda_events[name].append((start, end))
            return value
        started = time.perf_counter()
        value = operation()
        self._cpu_ms[name] += (time.perf_counter() - started) * 1e3
        return value

    def prepare_structural(self):
        """Read/scatter the shared separator sidecar once per request."""
        started = time.perf_counter()
        self._separator_blob = self.ctx.read_sep_kv(self.io)
        self._cpu_ms["chunk_io"] += (time.perf_counter() - started) * 1e3
        dev = self.runner.model.device

        def scatter_all():
            for li in range(int(self.ctx.meta["num_layers"])):
                positions = self._nl[li]
                self.ctx.cache.write(
                    li, "k", positions, self._separator_blob[0][li])
                self.ctx.cache.write(
                    li, "v", positions, self._separator_blob[1][li])

        self._timed("scatter", dev, scatter_all)

    def _hook(self, li, layer):
        def hook(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            if hidden.shape[1] == 1 or li in self.done:
                return
            started = time.perf_counter()
            self._run(li, layer, hidden, kwargs)
            self.done.add(li)
            self.hook_seconds += time.perf_counter() - started
        return hook

    def _run(self, li, layer, hidden, kwargs):
        decision_started = time.perf_counter()
        m, ctx = self.ctx.meta, self.ctx
        dev = hidden.device
        heads, head_dim = int(m["num_heads"]), int(m["head_dim"])
        v_start, v_num = int(m["v_token_start"]), int(m["v_token_num"])
        n_suffix = int(hidden.shape[1])
        attention = layer.self_attn

        def project_query_and_suffix_keys():
            normalised = layer.input_layernorm(hidden)
            query = attention.q_proj(normalised).view(
                1, n_suffix, heads, head_dim).transpose(1, 2)
            suffix_key = attention.k_proj(normalised).view(
                1, n_suffix, heads, head_dim).transpose(1, 2)
            cos, sin = kwargs["position_embeddings"]
            query, suffix_key = ml.apply_rotary_pos_emb(
                query, suffix_key, cos, sin)
            return query[0], suffix_key[0]

        query, suffix_key = self._timed(
            "query_projection", dev, project_query_and_suffix_keys)

        probe_started = time.perf_counter()
        probe_cpu = ctx.reader.read_probe(li, self.io)
        self._cpu_ms["probe_read_pipeline"] += (
            time.perf_counter() - probe_started) * 1e3
        probe_heads = min(self.probe, heads)
        probe_key = self._timed(
            "probe_h2d", dev,
            lambda: probe_cpu.to(dev, query.dtype).permute(1, 0, 2)[
                :probe_heads])
        system_key = ctx.cache.k[li][0, :probe_heads, :v_start]
        keys = torch.cat(
            [system_key, probe_key, suffix_key[:probe_heads]], dim=1)

        scores = self._timed(
            "query_scoring", dev,
            lambda: sv.rater_visual_scores_from_qk(
                query[:probe_heads], keys, self.rater_rows,
                v_start, v_num, head_reduce="mean",
                causal_from=v_start + v_num))
        self.query_score_calls += 1
        selected = self._timed(
            "topk", dev,
            lambda: sv.select_topk(
                scores, self.ratio, forbid=self._nl[li].tolist()))

        # This transfer is required to turn logical token ids into host-side
        # SSD byte ranges.  It also naturally completes all preceding selector
        # kernels, without artificial synchronizes between individual stages.
        d2h_started = time.perf_counter()
        selected_tokens = sorted(
            int(item) for item in selected.detach().cpu().reshape(-1))
        self._cpu_ms["selected_id_d2h"] += (
            time.perf_counter() - d2h_started) * 1e3
        planning_started = time.perf_counter()
        chunks = chunks_for_tokens(
            selected_tokens, v_num, int(m["chunk_size"]))
        run_count, run_lengths = contiguous_runs(chunks)
        self._cpu_ms["chunk_planning"] += (
            time.perf_counter() - planning_started) * 1e3
        # This is the non-additive, observed host interval for query
        # projection -> probe acquisition -> scoring -> Top-k -> selected-ID
        # materialisation -> chunk planning.  It excludes selected chunk reads
        # and scatter, which are reported separately.
        self.decision_host_seconds += time.perf_counter() - decision_started

        io_started = time.perf_counter()
        loaded = {}
        for kind in ("k", "v"):
            loaded[kind] = ctx.reader.read_chunks(
                li, kind, chunks, self.io)
        self._cpu_ms["chunk_io"] += (
            time.perf_counter() - io_started) * 1e3

        def scatter_and_mask():
            for kind in ("k", "v"):
                rows, values = loaded[kind]
                ctx.cache.write(li, kind, rows, values)
            keep = torch.zeros(v_num, dtype=torch.bool)
            keep[selected_tokens] = True
            keep[self._nl[li]] = True
            BIAS[li] = bias_from_keep(keep, m, dev)

        self._timed("scatter", dev, scatter_and_mask)
        self.log.append({
            "layer": int(li),
            "selected_tokens": selected_tokens,
            "selected_chunks": [int(item) for item in chunks],
            "contiguous_runs": int(run_count),
            "contiguous_run_lengths": [int(item) for item in run_lengths],
        })

    def _stage_ms(self, name):
        value = float(self._cpu_ms.get(name, 0.0))
        for start, end in self._cuda_events.get(name, []):
            value += float(start.elapsed_time(end))
        return value

    def stats(self):
        m = self.ctx.meta
        layers = max(1, int(m["num_layers"]))
        total_chunks = int(m["n_chunks_per_layer"])
        chunk_counts = [len(row["selected_chunks"]) for row in self.log]
        run_counts = [row["contiguous_runs"] for row in self.log]
        run_lengths = [length for row in self.log
                       for length in row["contiguous_run_lengths"]]
        io_summary = self.io.summary()
        probe_detail = io_summary.get("per_kind", {}).get("probe", {})
        stages = {
            "query_projection_ms": self._stage_ms("query_projection"),
            "probe_h2d_ms": self._stage_ms("probe_h2d"),
            "query_scoring_ms": self._stage_ms("query_scoring"),
            "topk_ms": self._stage_ms("topk"),
            "selected_id_d2h_ms": float(
                self._cpu_ms["selected_id_d2h"]),
            "chunk_planning_ms": float(self._cpu_ms["chunk_planning"]),
            "chunk_io_ms": float(self._cpu_ms["chunk_io"]),
            "scatter_ms": self._stage_ms("scatter"),
            "probe_read_pipeline_ms": float(
                self._cpu_ms["probe_read_pipeline"]),
            "probe_io_ms": float(probe_detail.get("seconds", 0.0)) * 1e3,
        }
        selector_component_sum = sum(stages[key] for key in (
            "query_projection_ms", "probe_h2d_ms", "query_scoring_ms",
            "topk_ms", "selected_id_d2h_ms", "chunk_planning_ms",
            "probe_read_pipeline_ms"))
        return {
            **stages,
            "selector_decision_host_wall_ms": float(
                self.decision_host_seconds * 1e3),
            "online_selector_without_raters_ms": float(
                self.decision_host_seconds * 1e3),
            "online_selector_component_sum_without_raters_ms": float(
                selector_component_sum),
            "selector_hook_host_wall_ms": float(self.hook_seconds * 1e3),
            "selector_component_timing_semantics": (
                "online_selector_without_raters_ms is the observed host wall "
                "for per-layer decision stages through chunk planning; the "
                "separate component sum can overlap and is diagnostic only"),
            "selection_mode": "qa_select25",
            "nominal_retention_ratio": float(self.ratio),
            "importance_source": (
                "SparseVLM-style text-guided rater-to-visual attention"),
            "query_dependent": True,
            "physical_layout": "raster",
            "repacking": False,
            "online_selection": True,
            "fallback_rate": 0.0,
            "adaptive_ratio": False,
            "query_score_calls": int(self.query_score_calls),
            "static_score_calls": 0,
            "diversity_calls": 0,
            "k_keep": int(self.k_keep),
            "logical_selected_tokens": int(self.k_keep),
            "logical_selected_tokens_per_layer": [
                len(row["selected_tokens"]) for row in self.log],
            "logical_selected_token_ratio": (
                float(self.k_keep) / float(m["n_spatial"])),
            "logical_kv_ratio": (
                float(self.k_keep) / float(m["n_spatial"])),
            "attended_kv_ratio_including_structural": (
                float(self.k_keep + len(self._nl[0])) / float(m["v_token_num"])),
            "selected_token_ids_per_layer": [
                row["selected_tokens"] for row in self.log],
            "selected_chunk_ids_per_layer": [
                row["selected_chunks"] for row in self.log],
            "n_chunks_selected": (
                float(np.mean(chunk_counts)) if chunk_counts else 0.0),
            "n_chunks_total": total_chunks,
            "selected_unique_chunks_total": int(sum(chunk_counts)),
            "touched_chunk_fraction": (
                float(np.mean(chunk_counts)) / total_chunks
                if chunk_counts else 0.0),
            "contiguous_runs_per_layer": run_counts,
            "contiguous_runs_per_layer_mean": (
                float(np.mean(run_counts)) if run_counts else 0.0),
            "mean_contiguous_run_length": (
                float(np.mean(run_lengths)) if run_lengths else 0.0),
            "normal_chunk_count_total": int(sum(chunk_counts)),
            "probe_read_bytes": int(probe_detail.get("bytes", 0)),
            "probe_preads": int(probe_detail.get("preads", 0)),
            "probe_heads_used": int(self.probe),
            "decoder_heads_total": int(m["num_heads"]),
            "probe_head_policy": "fixed_first_p_heads_mean",
            "separator_policy": "sidecar",
            "hook_ms": float(self.hook_seconds * 1e3),
            "layers_completed": len(self.log),
            "expected_layers": layers,
        }


class QAChunkLayerSelector(QASelectLayerSelector):
    """Query-aware mean-importance selection at physical SSD-chunk granularity.

    The probe-Q/K path and SparseVLM head-mean token scores are deliberately
    identical to :class:`QASelectLayerSelector`.  The only algorithmic change
    is that valid spatial token scores are averaged within their canonical
    raster chunks and the Ours-matched number of chunks is selected directly.
    Whole selected chunks are attended; unselected chunks remain masked.
    """

    def __init__(self, runner, ctx, rater_rows, ratio=0.25,
                 probe=PROBE_HEADS, counter=None):
        super().__init__(runner, ctx, rater_rows, ratio=ratio, probe=probe,
                         counter=counter)
        from mmimpress.cvpr25 import budget_chunk_count

        total_chunks = int(ctx.meta["n_chunks_per_layer"])
        self.k_chunks = budget_chunk_count(total_chunks, self.ratio)
        self.chunk_score_calls = 0
        for name in ("chunk_aggregation", "topk_chunk"):
            self._cpu_ms[name] = 0.0
            self._cuda_events[name] = []

    def _run(self, li, layer, hidden, kwargs):
        decision_started = time.perf_counter()
        m, ctx = self.ctx.meta, self.ctx
        dev = hidden.device
        heads, head_dim = int(m["num_heads"]), int(m["head_dim"])
        v_start, v_num = int(m["v_token_start"]), int(m["v_token_num"])
        chunk_size = int(m["chunk_size"])
        n_suffix = int(hidden.shape[1])
        attention = layer.self_attn

        def project_query_and_suffix_keys():
            normalised = layer.input_layernorm(hidden)
            query = attention.q_proj(normalised).view(
                1, n_suffix, heads, head_dim).transpose(1, 2)
            suffix_key = attention.k_proj(normalised).view(
                1, n_suffix, heads, head_dim).transpose(1, 2)
            cos, sin = kwargs["position_embeddings"]
            query, suffix_key = ml.apply_rotary_pos_emb(
                query, suffix_key, cos, sin)
            return query[0], suffix_key[0]

        query, suffix_key = self._timed(
            "query_projection", dev, project_query_and_suffix_keys)

        probe_started = time.perf_counter()
        probe_cpu = ctx.reader.read_probe(li, self.io)
        self._cpu_ms["probe_read_pipeline"] += (
            time.perf_counter() - probe_started) * 1e3
        probe_heads = min(self.probe, heads)
        probe_key = self._timed(
            "probe_h2d", dev,
            lambda: probe_cpu.to(dev, query.dtype).permute(1, 0, 2)[
                :probe_heads])
        system_key = ctx.cache.k[li][0, :probe_heads, :v_start]
        keys = torch.cat(
            [system_key, probe_key, suffix_key[:probe_heads]], dim=1)

        token_scores = self._timed(
            "query_scoring", dev,
            lambda: sv.rater_visual_scores_from_qk(
                query[:probe_heads], keys, self.rater_rows,
                v_start, v_num, head_reduce="mean",
                causal_from=v_start + v_num))
        self.query_score_calls += 1
        chunk_scores, valid_counts = self._timed(
            "chunk_aggregation", dev,
            lambda: mean_valid_spatial_chunk_scores(
                token_scores, self._nl[li].tolist(), v_num, chunk_size))
        self.chunk_score_calls += 1
        selected = self._timed(
            "topk_chunk", dev,
            lambda: select_qa_chunks_ours_budget(
                chunk_scores, valid_counts, self.ratio))

        # The host needs exact chunk ids to issue preads.  This one D2H boundary
        # completes every preceding selector kernel without adding artificial
        # synchronizations between projection, scoring, aggregation, and Top-k.
        d2h_started = time.perf_counter()
        ranked_chunks = [
            int(item) for item in selected.detach().cpu().reshape(-1)]
        self._cpu_ms["selected_id_d2h"] += (
            time.perf_counter() - d2h_started) * 1e3

        planning_started = time.perf_counter()
        chunks = sorted(set(ranked_chunks))
        assert len(chunks) == self.k_chunks, (
            f"QA-Chunk selected {len(chunks)} chunks, expected {self.k_chunks}")
        separators = set(int(item) for item in self._nl[li].tolist())
        selected_spatial = 0
        for chunk in chunks:
            start = chunk * chunk_size
            end = min(start + chunk_size, v_num)
            count = sum(row not in separators for row in range(start, end))
            assert count > 0, \
                f"QA-Chunk selected structural-only chunk {chunk}"
            selected_spatial += count
        run_count, run_lengths = contiguous_runs(chunks)
        self._cpu_ms["chunk_planning"] += (
            time.perf_counter() - planning_started) * 1e3
        # Observed, non-additive host interval for the decision path.  Selected
        # K/V reads and scatter are intentionally reported as separate stages.
        self.decision_host_seconds += time.perf_counter() - decision_started

        io_started = time.perf_counter()
        loaded = {}
        actual_loaded_chunks = None
        for kind in ("k", "v"):
            rows, values = ctx.reader.read_chunks(
                li, kind, chunks, self.io)
            loaded[kind] = (rows, values)
            actual = chunks_for_tokens(rows.tolist(), v_num, chunk_size)
            assert actual == chunks, (
                f"QA-Chunk loaded {actual} for selected chunks {chunks}")
            if actual_loaded_chunks is None:
                actual_loaded_chunks = actual
            else:
                assert actual == actual_loaded_chunks
        self._cpu_ms["chunk_io"] += (
            time.perf_counter() - io_started) * 1e3

        def scatter_and_mask():
            keep = torch.zeros(v_num, dtype=torch.bool)
            for kind in ("k", "v"):
                rows, values = loaded[kind]
                ctx.cache.write(li, kind, rows, values)
                keep[rows] = True
            # Every structural row remains available from the shared sidecar,
            # independent of whether its containing raster chunk was selected.
            keep[self._nl[li]] = True
            BIAS[li] = bias_from_keep(keep, m, dev)

        self._timed("scatter", dev, scatter_and_mask)
        self.log.append({
            "layer": int(li),
            "selected_chunks_ranked": ranked_chunks,
            "selected_chunks": [int(item) for item in chunks],
            "actual_loaded_chunks": [int(item)
                                     for item in actual_loaded_chunks],
            "selected_spatial_tokens": int(selected_spatial),
            "chunk_scores_generated": int(chunk_scores.numel()),
            "contiguous_runs": int(run_count),
            "contiguous_run_lengths": [int(item) for item in run_lengths],
        })

    def stats(self):
        m = self.ctx.meta
        layers = max(1, int(m["num_layers"]))
        total_chunks = int(m["n_chunks_per_layer"])
        chunk_counts = [len(row["selected_chunks"]) for row in self.log]
        spatial_counts = [row["selected_spatial_tokens"] for row in self.log]
        run_counts = [row["contiguous_runs"] for row in self.log]
        run_lengths_per_layer = [row["contiguous_run_lengths"]
                                 for row in self.log]
        run_lengths = [length for row in run_lengths_per_layer for length in row]
        io_summary = self.io.summary()
        probe_detail = io_summary.get("per_kind", {}).get("probe", {})
        stages = {
            "query_projection_ms": self._stage_ms("query_projection"),
            "probe_h2d_ms": self._stage_ms("probe_h2d"),
            "query_scoring_ms": self._stage_ms("query_scoring"),
            "chunk_aggregation_ms": self._stage_ms("chunk_aggregation"),
            "topk_chunk_ms": self._stage_ms("topk_chunk"),
            # Common result consumers historically call this field topk_ms.
            "topk_ms": self._stage_ms("topk_chunk"),
            "selected_id_d2h_ms": float(self._cpu_ms["selected_id_d2h"]),
            "chunk_planning_ms": float(self._cpu_ms["chunk_planning"]),
            "chunk_io_ms": float(self._cpu_ms["chunk_io"]),
            "scatter_ms": self._stage_ms("scatter"),
            "probe_read_pipeline_ms": float(
                self._cpu_ms["probe_read_pipeline"]),
            "probe_io_ms": float(probe_detail.get("seconds", 0.0)) * 1e3,
        }
        selector_component_sum = sum(stages[key] for key in (
            "query_projection_ms", "probe_h2d_ms", "query_scoring_ms",
            "chunk_aggregation_ms", "topk_chunk_ms",
            "selected_id_d2h_ms", "chunk_planning_ms",
            "probe_read_pipeline_ms"))
        mean_spatial = (float(np.mean(spatial_counts))
                        if spatial_counts else 0.0)
        attended = mean_spatial + len(self._nl[0])
        return {
            **stages,
            "selector_decision_host_wall_ms": float(
                self.decision_host_seconds * 1e3),
            "online_selector_without_raters_ms": float(
                self.decision_host_seconds * 1e3),
            "online_selector_component_sum_without_raters_ms": float(
                selector_component_sum),
            "selector_hook_host_wall_ms": float(self.hook_seconds * 1e3),
            "selector_component_timing_semantics": (
                "online_selector_without_raters_ms is the observed host wall "
                "for per-layer decision stages through chunk planning; the "
                "separate component sum can overlap and is diagnostic only"),
            "selection_mode": "qa_chunk25",
            "selection_granularity": "ssd_chunk",
            "chunk_score_aggregation": (
                "mean_valid_spatial_token_importance"),
            "chunk_budget_rounding": (
                "cvpr25.budget_chunk_count=round(total_chunks*ratio)"),
            "nominal_retention_ratio": float(self.ratio),
            "importance_source": (
                "SparseVLM-style text-guided rater-to-visual attention"),
            "query_dependent": True,
            "physical_layout": "raster",
            "repacking": False,
            "online_selection": True,
            "fallback_rate": 0.0,
            "full_load_fallback_count": 0,
            "adaptive_ratio": False,
            "query_score_calls": int(self.query_score_calls),
            "chunk_score_calls": int(self.chunk_score_calls),
            "static_score_calls": 0,
            "diversity_calls": 0,
            "k_chunks": int(self.k_chunks),
            "normal_chunk_budget_count": int(self.k_chunks),
            "normal_selected_chunk_count": int(self.k_chunks),
            "selected_chunk_count_per_layer": chunk_counts,
            "normal_chunk_candidate_count": total_chunks,
            "n_chunks_selected": (
                float(np.mean(chunk_counts)) if chunk_counts else 0.0),
            "n_chunks_total": total_chunks,
            "normal_selected_chunk_ratio": (
                float(np.mean(chunk_counts)) / total_chunks
                if chunk_counts else 0.0),
            "touched_chunk_fraction": (
                float(np.mean(chunk_counts)) / total_chunks
                if chunk_counts else 0.0),
            "total_touched_chunk_ratio": (
                float(np.mean(chunk_counts)) / total_chunks
                if chunk_counts else 0.0),
            "touched_chunk_ratio_semantics": (
                "selected normal K/V chunks divided by all normal K/V chunks; "
                "probe and separator sidecars are accounted separately in "
                "bytes, latency, and preads"),
            "selected_unique_chunks_total": int(sum(chunk_counts)),
            "normal_chunk_count_total": int(sum(chunk_counts)),
            "selected_chunk_ids_per_layer": [
                row["selected_chunks"] for row in self.log],
            "selected_chunk_ids_ranked_per_layer": [
                row["selected_chunks_ranked"] for row in self.log],
            "actual_loaded_chunk_ids_per_layer": [
                row["actual_loaded_chunks"] for row in self.log],
            "actual_loaded_chunks_match_selected": all(
                row["actual_loaded_chunks"] == row["selected_chunks"]
                for row in self.log),
            "chunk_scores_generated_per_layer": [
                row["chunk_scores_generated"] for row in self.log],
            "logical_selected_spatial_tokens_per_layer": spatial_counts,
            "logical_selected_spatial_token_ratio": (
                mean_spatial / float(m["n_spatial"])),
            "logical_selected_token_ratio": (
                mean_spatial / float(m["n_spatial"])),
            "logical_kv_ratio": attended / float(m["v_token_num"]),
            "attended_kv_ratio_including_structural": (
                attended / float(m["v_token_num"])),
            "contiguous_runs_per_layer": run_counts,
            "contiguous_runs_per_layer_mean": (
                float(np.mean(run_counts)) if run_counts else 0.0),
            "contiguous_run_lengths_per_layer": run_lengths_per_layer,
            "mean_contiguous_run_length": (
                float(np.mean(run_lengths)) if run_lengths else 0.0),
            "max_contiguous_run_length": (
                int(max(run_lengths)) if run_lengths else 0),
            "max_contiguous_run_length_per_layer": [
                int(max(row)) if row else 0 for row in run_lengths_per_layer],
            "probe_read_bytes": int(probe_detail.get("bytes", 0)),
            "probe_preads": int(probe_detail.get("preads", 0)),
            "probe_heads_used": int(self.probe),
            "decoder_heads_total": int(m["num_heads"]),
            "probe_head_policy": "fixed_first_p_heads_mean",
            "separator_policy": "sidecar",
            "hook_ms": float(self.hook_seconds * 1e3),
            "layers_completed": len(self.log),
            "expected_layers": layers,
        }


# ------------------------------------------------------------------ context
class ImageContext:
    """One stored image prefix, opened for serving.

    The store may be in raster order or reordered (4.4.1); either way the
    serving path reads and reasons in stored positions, so nothing here depends
    on which.  meta["order"] records the permutation for analysis only.
    """

    def __init__(self, store_dir, device, drop_cache=True,
                 require_v_hidden=True):
        from pathlib import Path
        self.dir = Path(store_dir)
        self.meta = load_meta(self.dir)
        nl = self.meta.get("newline_stored", self.meta["newline_idx"])
        if nl and isinstance(nl[0], list):
            self._separator_positions = [
                sorted(int(x) for x in row) for row in nl]
        else:
            row = sorted(int(x) for x in nl)
            self._separator_positions = [
                list(row) for _ in range(self.meta["num_layers"])]
        n_sep = len(self._separator_positions[0])
        assert all(len(row) == n_sep
                   for row in self._separator_positions), \
            "separator count varies across layers"
        self.sep_kv_shape = (2, self.meta["num_layers"], n_sep,
                             self.meta["num_heads"], self.meta["head_dim"])
        sep_path = self.dir / "sep_kv.bin"
        assert sep_path.stat().st_size == int(np.prod(self.sep_kv_shape)) * 2, \
            f"separator sidecar size mismatch: {sep_path}"
        self.reader = ChunkReader(self.dir, self.meta, drop_cache=drop_cache)
        sys_kv = torch.load(self.dir / "sys_kv.pt", weights_only=True)
        self.cache = PrefixCache(self.meta, sys_kv, device)
        v_hidden_path = self.dir / "v_hidden.pt"
        if require_v_hidden:
            self.v_hidden = torch.load(v_hidden_path, weights_only=True)
        else:
            # FullLoad and the sequential-Prefix path never consult the
            # SparseVLM rater input.  Turn-1 piggyback stores therefore need
            # not persist this otherwise-unused tensor.
            self.v_hidden = None

    def separator_positions(self, layer):
        """Stored row-separator positions without loading VisionZip metadata."""
        return self._separator_positions[layer]

    def validate_reordered_prefix_store(self):
        """Validate the on-disk layout required by the Prefix baseline.

        ``reorder_prefix_chunk`` is meaningful only for an importance-reordered
        store with an independent permutation at every decoder layer.  Keep
        this check on the context, rather than in the timed selector, so a bad
        raster/Morton store fails before it can produce a misleading result and
        valid requests do not pay validation overhead in TTFT.

        The legacy store metadata does not record the reorder algorithm name,
        but its importance layout is distinguishable from the shared Morton
        layout used by this repository via ``order_is_per_layer``.  The checks
        below also prove that every recorded order is a full permutation and
        that the stored separator positions agree with it.
        """
        if getattr(self, "_reordered_prefix_store_validated", False):
            return
        m = self.meta
        assert m.get("reordered") is True, \
            "prefix baseline requires an already-reordered KV store"
        assert m.get("order_is_per_layer") is True, \
            ("prefix baseline requires the per-layer importance layout; "
             "shared/raster layouts are not valid")
        orders = m.get("order")
        L, vn = m["num_layers"], m["v_token_num"]
        assert isinstance(orders, list) and len(orders) == L, \
            "missing per-layer stored-to-original permutations"
        expected = list(range(vn))
        original_sep = sorted(int(x) for x in m["newline_idx"])
        for li, order in enumerate(orders):
            assert isinstance(order, list) and len(order) == vn, \
                f"invalid permutation length at layer {li}"
            assert sorted(int(x) for x in order) == expected, \
                f"invalid stored-to-original permutation at layer {li}"
            stored_sep = self.separator_positions(li)
            assert sorted(int(order[p]) for p in stored_sep) == original_sep, \
                f"separator mapping disagrees with permutation at layer {li}"
        self._reordered_prefix_store_validated = True

    def validate_prefix_layout(self, expected_layout):
        """Validate an explicitly named physical layout for Prefix loading.

        The historical validator above intentionally remains pinned to the
        calib=4 per-layer importance store.  New image-only experiments use a
        single permutation for every layer and therefore need a separate,
        provenance-aware gate rather than weakening that old contract.
        """
        expected_layout = str(expected_layout)
        if getattr(self, "_prefix_layout_validated", None) == expected_layout:
            return
        m = self.meta
        actual = m.get("physical_layout", m.get("layout_method"))
        assert actual == expected_layout, \
            f"expected Prefix layout {expected_layout!r}, got {actual!r}"
        L, vn = int(m["num_layers"]), int(m["v_token_num"])
        identity = list(range(vn))
        raw = m.get("order")
        if raw:
            orders = raw if m.get("order_is_per_layer") else [raw] * L
        else:
            orders = [identity] * L
        assert len(orders) == L
        original_sep = sorted(int(x) for x in m["newline_idx"])
        for li, order in enumerate(orders):
            assert len(order) == vn and sorted(int(x) for x in order) == identity, \
                f"invalid {expected_layout} permutation at layer {li}"
            stored_sep = self.separator_positions(li)
            assert sorted(int(order[p]) for p in stored_sep) == original_sep, \
                f"separator mapping disagrees at layer {li}"

        if expected_layout == "visionzip_image_only":
            assert m.get("reordered") is True
            assert m.get("order_is_per_layer") is False, \
                "image-only VisionZip must use one global layer-independent order"
            assert m.get("global_order_all_layers") is True
            assert m.get("layout_uses_dataset_question") is False
            assert m.get("llm_used_for_layout_scoring") is False
            assert int(m.get("calibration_questions", -1)) == 0
            assert m.get("separator_tail") is True
            n_sep = len(original_sep)
            expected_tail = list(range(vn - n_sep, vn))
            assert self.separator_positions(0) == expected_tail
            assert all(self.separator_positions(li) == expected_tail
                       for li in range(L))
            assert (self.dir / "visionzip_layout.pt").is_file(), \
                "missing question-independent layout artifact"
        elif expected_layout == "raster":
            assert all([int(x) for x in order] == identity for order in orders)
            assert m.get("layout_uses_dataset_question") is False
            assert int(m.get("calibration_questions", -1)) == 0
        elif expected_layout == "morton":
            assert m.get("order_is_per_layer") is False
            assert m.get("separator_tail") is True
        elif expected_layout == "calib_importance_sep_tail":
            assert m.get("order_is_per_layer") is True
            assert int(m.get("calibration_questions", -1)) > 0
            assert m.get("separator_tail") is True
        else:
            raise AssertionError(f"unsupported explicit Prefix layout: {expected_layout}")
        self._prefix_layout_validated = expected_layout
        self._reordered_prefix_store_validated = True

    def validate_qa_select_layout(self):
        """Fail closed unless this is QA-Select's canonical raster store.

        Query-aware selection must not inherit the locality advantage of the
        image-only repacked store.  It also needs the small probe-key sidecar,
        the original-order visual input states used by SparseVLM rater
        selection, and the shared structural-token sidecar.  Validation is
        deliberately completed before the request timer starts.
        """
        if getattr(self, "_qa_select_layout_validated", False):
            return
        m = self.meta
        actual = m.get("physical_layout", m.get("layout_method"))
        assert actual == "raster", \
            f"QA-Select requires canonical raster layout, got {actual!r}"
        assert m.get("reordered", False) is False, \
            "QA-Select must not use a physically repacked store"
        assert m.get("order_is_per_layer", False) is False, \
            "QA-Select must not use a per-layer physical permutation"
        raw_order = m.get("order")
        if raw_order is not None:
            identity = list(range(int(m["v_token_num"])))
            assert [int(item) for item in raw_order] == identity, \
                "QA-Select raster store has a non-identity permutation"
        assert m.get("layout_uses_dataset_question") is False
        assert m.get("layout_uses_generated_answer") is False
        assert int(m.get("calibration_questions", -1)) == 0
        assert int(m.get("future_questions_used", -1)) == 0
        assert m.get("qa_select_compatible") is True
        assert m.get("turn1_normal_inference") is True
        assert m.get("layout_source") == "turn1_normal_inference_piggyback"
        assert m.get("visual_kv_source") == \
            "turn1_captured_past_key_values"
        assert m.get("visual_hidden_source") == \
            "same_turn1_decoder_layer0_input"
        assert m.get("separate_vision_forward") is False
        assert m.get("separate_prefix_forward") is False
        assert m.get("separate_model_forward_for_visual_hidden") is False
        assert m.get("capture_provenance_validated") is True
        hidden_capture = m.get("hidden_capture") or {}
        assert hidden_capture.get("capture_source") == \
            "same_turn1_normal_multimodal_prefill"
        assert int(hidden_capture.get("visual_hidden_capture_count", -1)) == 1
        original_separators = sorted(int(item) for item in m["newline_idx"])
        assert len(original_separators) == len(set(original_separators))
        assert all(0 <= item < int(m["v_token_num"])
                   for item in original_separators)
        assert all(self.separator_positions(li) == original_separators
                   for li in range(int(m["num_layers"]))), \
            "QA-Select raster separators are not in original positions"
        assert int(m.get("n_spatial", -1)) == (
            int(m["v_token_num"]) - len(original_separators))
        assert int(m.get("probe_heads", 0)) > 0, \
            "QA-Select requires a non-empty probe-key sidecar"
        assert int(m.get("probe_heads_required_for_serving", -1)) == int(
            m["probe_heads"])
        assert int(m.get("bytes_probe_sidecar", 0)) > 0
        assert int(m.get("bytes_separator_sidecar", 0)) > 0
        assert self.v_hidden is not None, \
            "QA-Select requires same-Turn-1 visual hidden states"
        assert self.v_hidden.ndim == 2
        assert int(self.v_hidden.shape[0]) == int(m["v_token_num"])
        self._qa_select_layout_validated = True

    def read_sep_kv(self, counter=None):
        """Always-loaded row-separator KV: (2, L, n_sep, H, hd), one read.

        Prefix-chunk loading shares this exact timed path with scored methods;
        all metadata derivation and validation happened in ``__init__`` before
        the request timer.
        """
        import os
        import time as _t
        import numpy as _np
        shape = self.sep_kv_shape
        n = int(_np.prod(shape))
        path = self.dir / "sep_kv.bin"
        fd = os.open(path, os.O_RDONLY)
        try:
            t0 = _t.perf_counter()
            buf = os.pread(fd, n * 2, 0)
            dt = _t.perf_counter() - t0
            # Do not evict here: this function runs inside the online request
            # timer.  ChunkReader.drop_all() already evicts sep_kv.bin before
            # the next cold request, outside TTFT/E2E.
        finally:
            os.close(fd)
        if counter is not None:
            counter.record("sep", len(buf), dt, preads=1, units=0)
        a = _np.frombuffer(buf, dtype=_np.float16).reshape(shape)
        return torch.from_numpy(a.copy())

    def close(self):
        self.reader.close()
        self.cache = None


def suffix_ids_for(runner, question):
    """Question-side token ids, without touching the vision tower.

    The stored prefix already covers [system tokens | expanded image block], so
    a request only needs the text after <image>: one tokenizer call, no pixels.
    """
    return suffix_ids_from_prompt(runner, runner.prompt(question))


def suffix_ids_from_prompt(runner, prompt):
    """Token ids after the sole image span in an arbitrary formatted prompt.

    The SSD single-image path is valid only when the image is the first and
    only multimodal block.  Explicit assertions prevent this helper from being
    accidentally used to concatenate independent MMDU image-prefix stores.
    """
    tok = runner.processor.tokenizer
    ids = tok(prompt, return_tensors="pt").input_ids[0]
    image_pos = (ids == runner.image_token_id).nonzero(as_tuple=True)[0]
    assert image_pos.numel() == 1, \
        f"stored single-image path needs one <image>, got {image_pos.numel()}"
    i = int(image_pos[0])
    return ids[i + 1:]


# ------------------------------------------------------------------- server
class Server:
    def __init__(self, runner, ratio=RETENTION_RATIO, probe=PROBE_HEADS,
                 alpha=ALPHA, max_new_tokens=16):
        self.runner = runner
        self.ratio = ratio
        self.probe = probe
        self.alpha = alpha
        self.max_new_tokens = max_new_tokens

    # ---------------------------------------------------------- raters
    def raters(self, ctx, suffix_ids):
        dev = self.runner.model.device
        assert ctx.v_hidden is not None, \
            "SparseVLM rater selection requires v_hidden.pt"
        emb = self.runner.model.get_input_embeddings()(suffix_ids.to(dev))
        pair = torch.cat([ctx.v_hidden.to(dev).float().unsqueeze(0),
                          emb.float().unsqueeze(0)], dim=1)
        r = sv.select_raters(pair, 0, ctx.meta["v_token_num"])
        return (r - ctx.meta["v_token_num"]).to(dev)

    # --------------------------------------------------------- generate
    @torch.no_grad()
    def _decode(self, cache, suffix_ids, prefix_len):
        """Prefill, expose the first token, then finish greedy decoding.

        The first CUDA synchronization is deliberately immediately after the
        first-token argmax.  It is the TTFT boundary used by every stored-KV
        method.  ``decode_ms`` starts at that boundary and covers only the
        autoregressive work needed for tokens 2..N (N <= max_new_tokens).

        The old loop returned at most ``max_new_tokens`` tokens but executed
        one unused forward after token N.  Stopping before that forward keeps
        the generated answer semantics while making every method perform the
        same maximum of 16 output-token decisions.
        """
        assert self.max_new_tokens >= 1
        model = self.runner.model
        dev = model.device
        tok = self.runner.processor.tokenizer
        n = suffix_ids.shape[0]
        pos = torch.arange(prefix_len, prefix_len + n, device=dev)

        prefill_t0 = time.perf_counter()
        out = model(input_ids=suffix_ids.to(dev).unsqueeze(0),
                    attention_mask=torch.ones(1, prefix_len + n,
                                              dtype=torch.long, device=dev),
                    position_ids=pos.unsqueeze(0), cache_position=pos,
                    past_key_values=cache, use_cache=True)
        first = int(out.logits[0, -1].argmax())
        torch.cuda.synchronize()
        first_token_at = time.perf_counter()
        prefill_ms = (first_token_at - prefill_t0) * 1e3

        decode_t0 = first_token_at
        # Count generated token decisions in the standard HF sense, including
        # EOS when it is produced.  Decoding the answer below removes specials.
        toks, cur = [first], prefix_len + n
        while toks[-1] != tok.eos_token_id and len(toks) < self.max_new_tokens:
            prev = toks[-1]
            cp = torch.tensor([cur], device=dev)
            out = model(input_ids=torch.tensor([[prev]], device=dev),
                        attention_mask=torch.ones(1, cur + 1, dtype=torch.long,
                                                  device=dev),
                        position_ids=cp.unsqueeze(0), cache_position=cp,
                        past_key_values=cache, use_cache=True)
            nxt = int(out.logits[0, -1].argmax())
            cur += 1
            toks.append(nxt)
        torch.cuda.synchronize()
        finished_at = time.perf_counter()
        answer = tok.decode(toks, skip_special_tokens=True).strip()
        return answer, first, {
            "first_token_at": first_token_at,
            "finished_at": finished_at,
            "prefill_ms": prefill_ms,
            "decode_ms": (finished_at - decode_t0) * 1e3,
            "generated_tokens": len(toks),
        }

    @torch.no_grad()
    def request(self, ctx, question=None, mode="impress", cold=True,
                prompt_text=None, suffix_ids=None):
        """Serve one question with true-TTFT and end-to-end timing.

        ``ttft`` ends immediately after the first output token is available;
        ``decode_latency`` covers subsequent autoregressive generation;
        ``e2e_latency`` ends after the final generated token.  Tokenization,
        host-to-device input preparation and cold-cache eviction stay outside
        all three timers, matching the original experiment boundary.
        """
        BIAS.clear()
        if cold:
            ctx.reader.drop_all()
        counter = IOCounter()
        dev = self.runner.model.device
        # Tokenization and request-input H2D are outside the online timers.
        if suffix_ids is None:
            suffix_ids = (suffix_ids_from_prompt(self.runner, prompt_text)
                          if prompt_text is not None
                          else suffix_ids_for(self.runner, question))
        suffix_ids = suffix_ids.to(dev)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        # FullLoad has no selector and must not pay a history-length-dependent
        # SparseVLM rater cost.  Existing artifacts remain untouched; this is
        # the causal multi-turn measurement path used by new runs.
        rr = (self.raters(ctx, suffix_ids) if mode != "fullload"
              else torch.empty(0, dtype=torch.long, device=dev))
        sel = LayerSelector(self.runner, ctx, rr, self.ratio, self.probe,
                            self.alpha, mode=mode, counter=counter)
        cache = ctx.cache.new_request()
        with sel:
            answer, first_token_id, timing = self._decode(
                cache, suffix_ids, ctx.meta["prefix_len"])
        ttft = timing["first_token_at"] - t0
        e2e = timing["finished_at"] - t0
        BIAS.clear()
        del cache
        return {
            "answer": answer,
            "first_token_id": first_token_id,
            "ttft": ttft,
            "decode_latency": timing["decode_ms"] / 1e3,
            "e2e_latency": e2e,
            # For hook-based modes this interval includes the layer hooks
            # (selection/read/scatter) as well as the model's prompt prefill.
            "prefill_ms": timing["prefill_ms"],
            "generated_tokens": timing["generated_tokens"],
            "n_raters": int(rr.numel()),
            "io": counter.summary(),
            "core_started_at_s": t0,
            "first_token_at_s": timing["first_token_at"],
            "model_finished_at_s": timing["finished_at"],
            "postprocess_finished_at_s": time.perf_counter(),
            **sel.stats(),
        }

    @torch.no_grad()
    def request_qa_select(self, ctx, question=None, *, cold=True,
                          prompt_text=None, suffix_ids=None):
        """Serve one cache hit with fixed-budget query-aware selection.

        The timer starts after the optional caller-supplied token preparation,
        matching the other core serving methods.  Paper-facing runners place a
        common outer timestamp before prompt construction/tokenization/H2D and
        use the absolute first-token timestamp returned here.  All online
        rater selection, probe reads, scoring, Top-k/chunk planning, selected
        reads, scatter, prefill, and first-token work occurs after ``t0``.
        """
        # Layout/provenance checks are invariant across requests and must not
        # be charged selectively to the first measured query.
        ctx.validate_qa_select_layout()
        BIAS.clear()
        if cold:
            ctx.reader.drop_all()
        counter = IOCounter()
        dev = self.runner.model.device
        if suffix_ids is None:
            suffix_ids = (suffix_ids_from_prompt(self.runner, prompt_text)
                          if prompt_text is not None
                          else suffix_ids_for(self.runner, question))
        suffix_ids = suffix_ids.to(dev)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rater_cpu_ms = 0.0
        rater_events = None
        if torch.device(dev).type == "cuda" and torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream(dev))
            rater_rows = self.raters(ctx, suffix_ids)
            end.record(torch.cuda.current_stream(dev))
            rater_events = (start, end)
        else:
            started = time.perf_counter()
            rater_rows = self.raters(ctx, suffix_ids)
            rater_cpu_ms = (time.perf_counter() - started) * 1e3

        cache = ctx.cache.new_request()
        selector = QASelectLayerSelector(
            self.runner, ctx, rater_rows, ratio=self.ratio,
            probe=self.probe, counter=counter)
        selector.prepare_structural()
        predecode_setup_wall_ms = (time.perf_counter() - t0) * 1e3
        with selector:
            answer, first_token_id, timing = self._decode(
                cache, suffix_ids, ctx.meta["prefix_len"])
        assert len(selector.log) == int(ctx.meta["num_layers"]), (
            "QA-Select did not run exactly once for every decoder layer: "
            f"{len(selector.log)} != {ctx.meta['num_layers']}")
        assert selector.query_score_calls == int(ctx.meta["num_layers"])
        ttft = timing["first_token_at"] - t0
        e2e = timing["finished_at"] - t0
        rater_ms = (float(rater_events[0].elapsed_time(rater_events[1]))
                     if rater_events is not None else float(rater_cpu_ms))
        stats = selector.stats()
        stats["rater_selection_ms"] = rater_ms
        stats["online_selector_total_ms"] = float(
            rater_ms + stats["online_selector_without_raters_ms"])
        stats["online_selector_component_sum_ms"] = float(
            rater_ms
            + stats["online_selector_component_sum_without_raters_ms"])
        # Existing result consumers use selector_ms.  Keep it as a precise
        # alias of the explicitly defined online selector total.
        stats["selector_ms"] = stats["online_selector_total_ms"]
        stats["predecode_setup_host_wall_ms"] = float(
            predecode_setup_wall_ms)
        stats["online_selector_host_wall_proxy_ms"] = float(
            predecode_setup_wall_ms + stats["selector_hook_host_wall_ms"])
        io_summary = counter.summary()
        per_kind = io_summary.get("per_kind", {})
        normal_kv_bytes = int(
            per_kind.get("k", {}).get("bytes", 0)
            + per_kind.get("v", {}).get("bytes", 0))
        normal_kv_preads = int(
            per_kind.get("k", {}).get("preads", 0)
            + per_kind.get("v", {}).get("preads", 0))
        separator_bytes = int(per_kind.get("sep", {}).get("bytes", 0))
        separator_preads = int(per_kind.get("sep", {}).get("preads", 0))
        probe_bytes = int(per_kind.get("probe", {}).get("bytes", 0))
        probe_preads = int(per_kind.get("probe", {}).get("preads", 0))

        BIAS.clear()
        del cache
        return {
            "answer": answer,
            "first_token_id": first_token_id,
            "ttft": ttft,
            "decode_latency": timing["decode_ms"] / 1e3,
            "e2e_latency": e2e,
            "prefill_ms": timing["prefill_ms"],
            "decode_ms": timing["decode_ms"],
            "generated_tokens": timing["generated_tokens"],
            "n_raters": int(rater_rows.numel()),
            "io": io_summary,
            "normal_kv_read_bytes": normal_kv_bytes,
            "separator_read_bytes": separator_bytes,
            "probe_read_bytes": probe_bytes,
            "normal_kv_preads": normal_kv_preads,
            "separator_preads": separator_preads,
            "probe_preads": probe_preads,
            "total_actual_pread_bytes": int(io_summary.get("bytes", 0)),
            "qa_select_layout_validated": bool(getattr(
                ctx, "_qa_select_layout_validated", False)),
            "core_started_at_s": t0,
            "first_token_at_s": timing["first_token_at"],
            "model_finished_at_s": timing["finished_at"],
            "postprocess_finished_at_s": time.perf_counter(),
            **stats,
        }

    @torch.no_grad()
    def request_qa_chunk(self, ctx, question=None, *, cold=True,
                         prompt_text=None, suffix_ids=None):
        """Serve one cache hit with Ours-budgeted query-aware SSD chunks.

        The raster store, SparseVLM raters, probe heads, Q/K scoring, TTFT
        boundary, and I/O counter are the validated QA-Select path.  Only the
        post-score decision changes: valid spatial scores are mean-aggregated
        by physical chunk and exactly the Ours Prefix25 chunk count is loaded.
        """
        ctx.validate_qa_select_layout()
        BIAS.clear()
        if cold:
            ctx.reader.drop_all()
        counter = IOCounter()
        dev = self.runner.model.device
        if suffix_ids is None:
            suffix_ids = (suffix_ids_from_prompt(self.runner, prompt_text)
                          if prompt_text is not None
                          else suffix_ids_for(self.runner, question))
        suffix_ids = suffix_ids.to(dev)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rater_cpu_ms = 0.0
        rater_events = None
        if torch.device(dev).type == "cuda" and torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream(dev))
            rater_rows = self.raters(ctx, suffix_ids)
            end.record(torch.cuda.current_stream(dev))
            rater_events = (start, end)
        else:
            started = time.perf_counter()
            rater_rows = self.raters(ctx, suffix_ids)
            rater_cpu_ms = (time.perf_counter() - started) * 1e3

        cache = ctx.cache.new_request()
        selector = QAChunkLayerSelector(
            self.runner, ctx, rater_rows, ratio=self.ratio,
            probe=self.probe, counter=counter)
        selector.prepare_structural()
        predecode_setup_wall_ms = (time.perf_counter() - t0) * 1e3
        with selector:
            answer, first_token_id, timing = self._decode(
                cache, suffix_ids, ctx.meta["prefix_len"])
        assert len(selector.log) == int(ctx.meta["num_layers"]), (
            "QA-Chunk did not run exactly once for every decoder layer: "
            f"{len(selector.log)} != {ctx.meta['num_layers']}")
        assert selector.query_score_calls == int(ctx.meta["num_layers"])
        assert selector.chunk_score_calls == int(ctx.meta["num_layers"])
        ttft = timing["first_token_at"] - t0
        e2e = timing["finished_at"] - t0
        rater_ms = (float(rater_events[0].elapsed_time(rater_events[1]))
                     if rater_events is not None else float(rater_cpu_ms))
        stats = selector.stats()
        stats["rater_selection_ms"] = rater_ms
        stats["online_selector_total_ms"] = float(
            rater_ms + stats["online_selector_without_raters_ms"])
        stats["online_selector_component_sum_ms"] = float(
            rater_ms
            + stats["online_selector_component_sum_without_raters_ms"])
        stats["selector_ms"] = stats["online_selector_total_ms"]
        stats["predecode_setup_host_wall_ms"] = float(
            predecode_setup_wall_ms)
        stats["online_selector_host_wall_proxy_ms"] = float(
            predecode_setup_wall_ms + stats["selector_hook_host_wall_ms"])

        io_summary = counter.summary()
        per_kind = io_summary.get("per_kind", {})
        normal_kv_bytes = int(
            per_kind.get("k", {}).get("bytes", 0)
            + per_kind.get("v", {}).get("bytes", 0))
        normal_kv_preads = int(
            per_kind.get("k", {}).get("preads", 0)
            + per_kind.get("v", {}).get("preads", 0))
        normal_kv_read_ms = float(
            per_kind.get("k", {}).get("seconds", 0.0)
            + per_kind.get("v", {}).get("seconds", 0.0)) * 1e3
        separator_bytes = int(per_kind.get("sep", {}).get("bytes", 0))
        separator_preads = int(per_kind.get("sep", {}).get("preads", 0))
        separator_read_ms = float(
            per_kind.get("sep", {}).get("seconds", 0.0)) * 1e3
        probe_bytes = int(per_kind.get("probe", {}).get("bytes", 0))
        probe_preads = int(per_kind.get("probe", {}).get("preads", 0))

        BIAS.clear()
        del cache
        return {
            "answer": answer,
            "first_token_id": first_token_id,
            "ttft": ttft,
            "decode_latency": timing["decode_ms"] / 1e3,
            "e2e_latency": e2e,
            "prefill_ms": timing["prefill_ms"],
            "decode_ms": timing["decode_ms"],
            "generated_tokens": timing["generated_tokens"],
            "n_raters": int(rater_rows.numel()),
            "io": io_summary,
            "normal_kv_read_bytes": normal_kv_bytes,
            "selected_chunk_payload_bytes": normal_kv_bytes,
            "separator_read_bytes": separator_bytes,
            "probe_read_bytes": probe_bytes,
            "normal_kv_preads": normal_kv_preads,
            "separator_preads": separator_preads,
            "probe_preads": probe_preads,
            "normal_kv_read_ms": normal_kv_read_ms,
            "selected_chunk_io_ms": normal_kv_read_ms,
            "separator_read_ms": separator_read_ms,
            "total_actual_pread_bytes": int(io_summary.get("bytes", 0)),
            "qa_chunk_layout_validated": bool(getattr(
                ctx, "_qa_select_layout_validated", False)),
            "qa_select_layout_validated": bool(getattr(
                ctx, "_qa_select_layout_validated", False)),
            "core_started_at_s": t0,
            "first_token_at_s": timing["first_token_at"],
            "model_finished_at_s": timing["finished_at"],
            "postprocess_finished_at_s": time.perf_counter(),
            **stats,
        }

    @torch.no_grad()
    def request_cvpr25(self, ctx, question=None, static=None, budget=0.25,
                       mode="static", lam_static=1.0, lam_query=1.0,
                       sep_policy="force", diverse_frac=0.25, cold=True,
                       seed=0, image_id="", prompt_text=None,
                       suffix_ids=None, expected_prefix_layout=None):
        """Hook-free chunk-first request.

        Selection, reads and cache fill all happen BEFORE the forward, so the
        model runs with no hooks and no attention weights are ever produced for
        identification.  ``prefill_ms`` ends at the first-token boundary,
        ``decode_ms`` is subsequent autoregressive generation, and
        ``model_ms`` remains their sum for backwards-compatible diagnostics.
        """
        # This is deliberately before cache eviction, token preparation and
        # the request timer.  A Prefix result from a raster/shared-order store
        # would answer a different research question and must fail closed.
        if mode == "prefix":
            if expected_prefix_layout is None:
                ctx.validate_reordered_prefix_store()
            else:
                ctx.validate_prefix_layout(expected_prefix_layout)
        BIAS.clear()
        if cold:
            ctx.reader.drop_all()
        counter = IOCounter()
        dev = self.runner.model.device
        # Tokenization and request-input H2D are outside the online timers.
        if suffix_ids is None:
            suffix_ids = (suffix_ids_from_prompt(self.runner, prompt_text)
                          if prompt_text is not None
                          else suffix_ids_for(self.runner, question))
        suffix_ids = suffix_ids.to(dev)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        # The final static_diverse method is query-independent.  Avoid an
        # otherwise unused O(history length) embedding lookup so its measured
        # selector overhead remains the intended image-level constant.
        text_emb = (self.runner.model.get_input_embeddings()(suffix_ids)
                    if mode in ("hybrid", "diverse") else None)
        cache = ctx.cache.new_request()
        sel = CVPR25ChunkSelector(self.runner, ctx, static, budget, mode,
                                  lam_static, lam_query, sep_policy,
                                  diverse_frac, counter, seed=seed,
                                  image_id=image_id)
        sel.prepare(text_emb)
        torch.cuda.synchronize()
        t_prep = time.perf_counter() - t0

        answer, first_token_id, timing = self._decode(
            cache, suffix_ids, ctx.meta["prefix_len"])
        ttft = timing["first_token_at"] - t0
        e2e = timing["finished_at"] - t0
        t_model_ms = timing["prefill_ms"] + timing["decode_ms"]

        BIAS.clear()
        del cache
        st = sel.stats()
        io_summary = counter.summary()
        per_kind = io_summary.get("per_kind", {})
        separator_bytes = int(per_kind.get("sep", {}).get("bytes", 0))
        normal_kv_bytes = int(per_kind.get("k", {}).get("bytes", 0) +
                              per_kind.get("v", {}).get("bytes", 0))
        separator_preads = int(per_kind.get("sep", {}).get("preads", 0))
        normal_kv_preads = int(per_kind.get("k", {}).get("preads", 0) +
                               per_kind.get("v", {}).get("preads", 0))
        st.update({
            "answer": answer,
            "first_token_id": first_token_id,
            "ttft": ttft,
            "decode_latency": timing["decode_ms"] / 1e3,
            "e2e_latency": e2e,
            "prepare_ms": t_prep * 1e3,
            "prefill_ms": timing["prefill_ms"],
            "decode_ms": timing["decode_ms"],
            "model_ms": t_model_ms,
            "generated_tokens": timing["generated_tokens"],
            "io": io_summary,
            "normal_kv_read_bytes": normal_kv_bytes,
            "separator_read_bytes": separator_bytes,
            "normal_kv_preads": normal_kv_preads,
            "separator_preads": separator_preads,
            "total_actual_pread_bytes": int(io_summary.get("bytes", 0)),
            "reordered_prefix_store_validated": (
                bool(getattr(ctx, "_reordered_prefix_store_validated", False))
                if mode == "prefix" else None),
            "validated_prefix_layout": (
                getattr(ctx, "_prefix_layout_validated", None)
                if mode == "prefix" else None),
            "n_raters": 0,
            "core_started_at_s": t0,
            "first_token_at_s": timing["first_token_at"],
            "model_finished_at_s": timing["finished_at"],
            "postprocess_finished_at_s": time.perf_counter(),
        })
        st["selector_ms"] = st["select_ms"] + st["query_ms"]
        return st

    @torch.no_grad()
    def recompute(self, enc, question_unused=None,
                  return_past_key_values=False):
        """ReComp baseline with the same true-TTFT/decode boundary.

        Input processing and host-to-device transfer happen before ``t0``.
        The timed prefill includes the vision tower and multimodal prompt.
        ``generate`` is retained so prediction semantics remain byte-for-byte
        comparable with the original ReComp baseline; a non-stopping criterion
        timestamps the first generated token without ending generation.
        """
        runner = self.runner
        enc = runner.to_device(enc)
        tok = runner.processor.tokenizer
        assert self.max_new_tokens >= 1
        # Timing instrumentation itself is prepared before the online timer.
        marker = _FirstTokenTimestamp()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prefill_t0 = time.perf_counter()
        out = runner.model.generate(
            **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id, stopping_criteria=[marker],
            return_dict_in_generate=return_past_key_values,
            use_cache=True)
        torch.cuda.synchronize()
        finished_at = time.perf_counter()
        assert marker.at is not None, "generate returned without an output token"
        first_token_at = marker.at
        decode_t0 = first_token_at
        sequences = out.sequences if return_past_key_values else out
        toks = sequences[0, enc["input_ids"].shape[1]:]
        text = tok.decode(toks, skip_special_tokens=True).strip()
        result = {
            "answer": text,
            "first_token_id": int(toks[0]),
            "ttft": first_token_at - t0,
            "decode_latency": finished_at - decode_t0,
            "e2e_latency": finished_at - t0,
            "prefill_ms": (first_token_at - prefill_t0) * 1e3,
            "decode_ms": (finished_at - decode_t0) * 1e3,
            "generated_tokens": int(toks.numel()),
            "core_started_at_s": t0,
            "first_token_at_s": first_token_at,
            "model_finished_at_s": finished_at,
            "postprocess_finished_at_s": time.perf_counter(),
        }
        if return_past_key_values:
            result["captured_past_key_values"] = out.past_key_values
        return result


# --------------------------------------------------------------- calibration
class Calibrator:
    """Collect per-layer SparseVLM importance over a fully loaded prefix.

    Feeds reorder.py: IMPRESS 4.4.1 repacks by AVERAGE importance, so the
    statistic has to be accumulated over several questions about the same image
    before any order is committed.  Scores are averaged over heads because the
    consensus path -- the one that decides which chunks get read -- applies a
    single token set to every head, so a single per-layer order is what the
    read pattern actually wants.
    """

    def __init__(self, runner, ctx, rater_rows):
        self.runner = runner
        self.ctx = ctx
        self.rater_rows = rater_rows
        self.scores = {}
        self._handles = []

    def __enter__(self):
        for li, layer in enumerate(self.runner.layers):
            self._handles.append(layer.register_forward_pre_hook(
                self._hook(li, layer), with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()

    def _hook(self, li, layer):
        def hook(module, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            if h.shape[1] == 1 or li in self.scores:
                return
            m = self.ctx.meta
            H, hd = m["num_heads"], m["head_dim"]
            v0, vn = m["v_token_start"], m["v_token_num"]
            n = h.shape[1]
            attn = layer.self_attn
            hn = layer.input_layernorm(h)
            q = attn.q_proj(hn).view(1, n, H, hd).transpose(1, 2)
            kn = attn.k_proj(hn).view(1, n, H, hd).transpose(1, 2)
            cos, sin = kwargs["position_embeddings"]
            q, kn = ml.apply_rotary_pos_emb(q, kn, cos, sin)
            keys = torch.cat([self.ctx.cache.k[li][0], kn[0]], dim=1)
            s = sv.rater_visual_scores_from_qk(
                q[0], keys, self.rater_rows, v0, vn, causal_from=v0 + vn)
            self.scores[li] = s.mean(dim=0).float().cpu()   # (v_num,)
        return hook


@torch.no_grad()
def calibrate_image(server, ctx, questions):
    """Mean per-layer visual importance over several questions (original order).

    Returns (num_layers, v_num) float tensor.
    """
    acc, n = None, 0
    for q in questions:
        suffix_ids = suffix_ids_for(server.runner, q)
        rr = server.raters(ctx, suffix_ids)
        cache = ctx.cache.new_request()
        for li in range(ctx.meta["num_layers"]):
            for kind in ("k", "v"):
                ctx.cache.write_full(li, kind,
                                     ctx.reader.read_full(li, kind))
        cal = Calibrator(server.runner, ctx, rr)
        with cal:
            dev = server.runner.model.device
            P, nn = ctx.meta["prefix_len"], suffix_ids.shape[0]
            pos = torch.arange(P, P + nn, device=dev)
            server.runner.model(
                input_ids=suffix_ids.to(dev).unsqueeze(0),
                attention_mask=torch.ones(1, P + nn, dtype=torch.long,
                                          device=dev),
                position_ids=pos.unsqueeze(0), cache_position=pos,
                past_key_values=cache, use_cache=True)
        s = torch.stack([cal.scores[li]
                         for li in range(ctx.meta["num_layers"])])
        acc = s if acc is None else acc + s
        n += 1
        del cache
    return acc / max(n, 1)


# ===================================================================== CVPR25
# Storage-aware, chunk-first selection.  Deliberately NOT a LayerSelector
# subclass and deliberately hook-free: the whole point is that nothing runs
# inside the LLM for identification.  The chunk set for every layer is decided
# BEFORE the forward, the bytes are read, the cache is filled, and then the
# model runs untouched.  Identification therefore costs no LLM compute, needs
# no attention weights, and does not force eager attention.

class CVPR25ChunkSelector:
    """Static (VisionZip) + optional query correction (PACT) over SSD chunks.

    modes:
      "prefix"   first-k chunks in the calibrated, reordered SSD layout;
                 no static/query/diversity score is loaded or evaluated
      "static"   VisionZip saliency only -- zero per-question identification
      "hybrid"   + a PACT-shaped query correction (one q_proj, one dot product)
      "diverse"  + DivPrune MaxMin tie-breaking among near-equal chunks

    The budget is a CHUNK budget, so touched-chunk-fraction == budget by
    construction.  Whole chunks are read, so every token inside a selected
    chunk is kept (masking them out would cost accuracy and save no bytes);
    the realised logical KV ratio is reported rather than assumed.
    """

    def __init__(self, runner, ctx, static, budget=0.25, mode="static",
                 lam_static=1.0, lam_query=1.0, sep_policy="force",
                 diverse_frac=0.25, counter=None, seed=0, image_id=""):
        self.runner = runner
        self.ctx = ctx
        self.static = static
        self.budget = budget
        self.mode = mode
        self.lam_static = lam_static
        self.lam_query = lam_query
        self.sep_policy = sep_policy
        self.diverse_frac = diverse_frac
        self.io = counter if counter is not None else IOCounter()
        self.t = {"select": 0.0, "chunk_io": 0.0, "scatter": 0.0,
                  "query": 0.0}
        self.seed = seed
        self.image_id = image_id
        self.chunks_per_layer = []
        self.selected_chunk_ids_per_layer = []
        self.kept_tokens_per_layer = []
        self.static_score_calls = 0
        self.query_score_calls = 0
        self.diversity_calls = 0
        assert self.mode == "prefix" or self.static is not None, \
            f"{self.mode} requires static metadata"
        if self.mode == "prefix":
            assert self.static is None, \
                "prefix baseline must not receive VisionZip/static metadata"
            assert self.sep_policy == "sidecar", \
                "prefix baseline is defined with the shared separator sidecar"

    # -------------------------------------------------------- query score
    def _query_scores(self, text_emb):
        """(L, n_chunks) PACT-shaped correction from precomputed image keys.

        One q_proj over the question tokens, then a single batched dot product
        against the per-chunk key descriptors.  No attention matrix, no LLM
        layer executed, no hook.
        """
        from mmimpress.cvpr25 import zscore
        runner, dev = self.runner, self.runner.model.device
        layer = runner.layers[0]
        H = runner.n_heads
        hd = runner.head_dim
        ck = self.static["chunk_keys"]                     # (L, C, hid)
        L, C, hid = ck.shape
        q = layer.self_attn.q_proj(layer.input_layernorm(
            text_emb.to(dev, torch.bfloat16).unsqueeze(0)))[0]
        q = q.view(-1, H, hd).permute(1, 0, 2)             # (H, n_q, hd)
        kk = ck.to(dev, torch.bfloat16).view(L * C, H, hd).permute(1, 2, 0)
        s = torch.bmm(q, kk) * (hd ** -0.5)               # (H, n_q, L*C)
        s = s.float().mean(dim=(0, 1)).view(L, C).cpu()
        return torch.stack([zscore(s[li]) for li in range(L)])

    # ------------------------------------------------------------ prepare
    @torch.no_grad()
    def prepare(self, text_emb):
        """Decide, read and install everything for this request."""
        from mmimpress.cvpr25 import choose_chunks
        m, ctx = self.ctx.meta, self.ctx
        dev = self.runner.model.device
        L, nc = m["num_layers"], m["n_chunks_per_layer"]
        cs, vn = m["chunk_size"], m["v_token_num"]

        self._sep = None
        if self.sep_policy == "sidecar":
            t0 = time.perf_counter()
            self._sep = ctx.read_sep_kv(self.io)
            self.t["chunk_io"] += time.perf_counter() - t0

        qs = None
        if self.mode in ("hybrid", "diverse"):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            qs = self._query_scores(text_emb)
            self.query_score_calls += 1
            torch.cuda.synchronize()
            self.t["query"] += time.perf_counter() - t0

        for li in range(L):
            t0 = time.perf_counter()
            # "sidecar" separator policy: separators come from their own tiny
            # file, so their chunks are not bought out of the budget.
            force = (self.static["sep_chunks"][li]
                     if self.sep_policy == "force" else [])
            cids, _ = choose_chunks(
                self.mode, li, self.static, self.budget, force=force,
                query_score=(qs[li] if qs is not None else None),
                lam_static=self.lam_static, lam_query=self.lam_query,
                diverse_frac=self.diverse_frac, seed=self.seed,
                image_id=self.image_id, n_chunks=nc)
            if self.mode in ("static", "hybrid", "diverse",
                             "static_diverse"):
                self.static_score_calls += 1
            if self.mode in ("diverse", "static_diverse", "diverse_only"):
                self.diversity_calls += 1
            self.t["select"] += time.perf_counter() - t0
            self.chunks_per_layer.append(len(cids))
            self.selected_chunk_ids_per_layer.append([int(c) for c in cids])

            t0 = time.perf_counter()
            loaded = {}
            for kind in ("k", "v"):
                loaded[kind] = ctx.reader.read_chunks(li, kind, cids, self.io)
            self.t["chunk_io"] += time.perf_counter() - t0

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for kind in ("k", "v"):
                rows, vals = loaded[kind]
                ctx.cache.write(li, kind, rows, vals)
            keep = torch.zeros(vn, dtype=torch.bool)
            keep[loaded["k"][0]] = True          # whole chunks are readable
            if self.sep_policy == "sidecar":
                sp = torch.tensor(ctx.separator_positions(li),
                                  dtype=torch.long)
                ctx.cache.write(li, "k", sp, self._sep[0][li])
                ctx.cache.write(li, "v", sp, self._sep[1][li])
                keep[sp] = True
            self.kept_tokens_per_layer.append(int(keep.sum()))
            BIAS[li] = bias_from_keep(keep, m, dev)
            torch.cuda.synchronize()
            self.t["scatter"] += time.perf_counter() - t0

    def stats(self):
        nc = self.ctx.meta["n_chunks_per_layer"]
        run_rows = [contiguous_runs(row)
                    for row in self.selected_chunk_ids_per_layer]
        run_counts = [count for count, _ in run_rows]
        run_lengths = [length for _, lengths in run_rows for length in lengths]
        return {"n_chunks_selected": float(np.mean(self.chunks_per_layer)),
                "n_chunks_total": nc,
                "touched_chunk_fraction": float(np.mean(self.chunks_per_layer))
                / nc,
                "logical_kv_ratio": float(np.mean(self.kept_tokens_per_layer))
                / self.ctx.meta["v_token_num"],
                "logical_kv_ratio_per_layer": [
                    n / self.ctx.meta["v_token_num"]
                    for n in self.kept_tokens_per_layer],
                "fallback_rate": 0.0,
                "mean_jaccard": None,
                "selection_mode": self.mode,
                "selected_chunk_ids_per_layer":
                    self.selected_chunk_ids_per_layer,
                "contiguous_runs_per_layer": run_counts,
                "contiguous_runs_per_layer_mean": (
                    float(np.mean(run_counts)) if run_counts else 0.0),
                "mean_contiguous_run_length": (
                    float(np.mean(run_lengths)) if run_lengths else 0.0),
                "normal_chunk_count_total": int(sum(self.chunks_per_layer)),
                "static_score_calls": int(self.static_score_calls),
                "query_score_calls": int(self.query_score_calls),
                "diversity_calls": int(self.diversity_calls),
                "separator_policy": self.sep_policy,
                "select_ms": self.t["select"] * 1e3,
                "query_ms": self.t["query"] * 1e3,
                "chunk_io_ms": self.t["chunk_io"] * 1e3,
                "scatter_ms": self.t["scatter"] * 1e3,
                "hook_ms": 0.0}


def load_static(ctx):
    p = ctx.dir / "static.pt"
    assert p.exists(), f"missing static sidecar: run scripts/06_build_static.py"
    static = torch.load(p, weights_only=True)
    assert tuple(static["sep_kv_shape"]) == ctx.sep_kv_shape
    assert [[int(x) for x in row] for row in static["sep_pos"]] == \
        ctx._separator_positions
    return static

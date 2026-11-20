"""IMPRESS selection path written from the authors' public implementation.

Reference: https://github.com/ISCS-ZJU/HyperInfer, commit 91df7a2
(``h2o_flexgen/flexgen``).  Nothing is copied from that repository; the logic
below is a fresh implementation, and ``tests/test_impress_official.py`` feeds
the same tensors to the public functions and to these ones.

Public function                      -> here
  sele_tokenid (attention of the        causal_attention, column_importance,
    probe heads, every row)             prefix_row_importance,
                                        suffix_row_importance(_from_qk)
  _sele_important_tokens                head_topk, vote_consensus,
                                        select_tokens
  cal_jaccard_sim                       mean_pairwise_jaccard
  mha_with_sele_percent_prefixkv        descending_order, reverse_index,
    (generate_mapping_list)             layer_order, Turn1AttentionStats
  mha_with_filled_selected_prefixkv     OfficialImpressSelector (additive mask
    (del_tokenids)                      on the tokens that were not selected)
  my_pcache_fast (KV_Division, 64-      write_official_layout,
    token chunks, head_token, reorder)  read_probe_block, the selector's loads

What the public code does, as read at that commit:

  importance   attention of the probe heads (0, 1, 2) over the WHOLE sequence
               (prefix rows and suffix rows), causal softmax, column sum; only
               prefix columns are candidates.
  selection    per head top ``ceil(n * percent / 100)``; mean pairwise Jaccard
               of the per-head sets; ``sim >= sim_thred`` -> the tokens with
               the most votes, that many again; otherwise every prefix token.
  fallback     every prefix token's K and V is loaded and used, no mask.
  mask         ``mha_with_filled_selected_prefixkv`` puts ``finfo.min`` on the
               columns of the tokens that were not selected.
  reorder      one request per prefix: column sums over every row, summed over
               ALL heads, descending; the store keeps that order and reads
               64-token chunks, K and V in separate chunk files.

MLLM adaptation (the contract lists these as differences):

  * candidates are the image tokens; system-prompt tokens and the row
    separators are always kept and never consume budget;
  * the prefix rows' share of the column sums does not depend on the question,
    so it is taken once from the image's first turn-1 inference
    (``Turn1AttentionStats``) and stored next to the probe keys; a request
    computes only the suffix rows' share and adds the stored one;
  * the stored order is [separators | image tokens by descending turn-1
    importance], from that single turn-1 request and never updated.

Everything on the serving path works in STORED positions, like the existing
selectors: attention does not depend on the order of the keys.

Contract: docs/mt_gqa_impress_official_contract.md.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import transformers.models.llama.modeling_llama as ml

from mmimpress.reorder import mapping_from_perm
from mmimpress.store import (META_NAME, IOCounter, chunks_for_tokens,
                             load_meta, write_separator_sidecar_from_store)

OFFICIAL_REPOSITORY = "https://github.com/ISCS-ZJU/HyperInfer"
OFFICIAL_COMMIT = "91df7a2fc12581c6780a6485c5b50e7fd9ef9d0e"
PROBE_HEADS = (0, 1, 2)            # --sele-head 0 1 2
SIM_THRESHOLD = 0.3                # --sim-thred 0.3 at --sele-percent 25
KEEP_RATIO = 0.25
CHUNK_TOKENS = 64                  # --chunk-size 64
LAYOUT = "impress_official_turn1"
TRAILER = "prefix_row_importance_f32"
PREFETCH = ("none", "next_layer")

_NP = {"float16": np.float16, "float32": np.float32}


# ===================================================================== pure
def keep_count(n_candidates, ratio=KEEP_RATIO):
    """``math.ceil(prefix_len * cur_sele_percent / 100)`` of the public code."""
    return max(1, min(int(n_candidates), math.ceil(n_candidates * ratio)))


def causal_attention(q, k, dtype=torch.float32):
    """Attention probabilities of every row: (P, s, hd) x (P, s, hd) -> (P, s, s).

    ``q`` already carries the 1/sqrt(hd) scaling.  Row i sees keys 0..i.
    """
    s = q.shape[1]
    logits = q @ k.transpose(1, 2)
    idx = torch.arange(s, device=q.device)
    hidden = idx.view(1, 1, s) > idx.view(1, s, 1)
    return torch.softmax(logits.masked_fill(hidden, float("-inf")), dim=2,
                         dtype=dtype)


def column_importance(attn, n_prefix, row_start=0, row_stop=None):
    """Column sums over rows [row_start, row_stop), prefix columns only."""
    return attn[:, row_start:row_stop].sum(dim=1)[:, :n_prefix]


def prefix_row_importance(attn, n_prefix):
    """The share of the column sums contributed by the prefix rows."""
    return column_importance(attn, n_prefix, 0, n_prefix)


def suffix_row_importance(attn, n_prefix):
    """The share of the column sums contributed by the suffix rows."""
    return column_importance(attn, n_prefix, n_prefix, None)


def suffix_row_importance_from_qk(q, keys, n_prefix, scale):
    """Suffix rows' share without building the prefix rows.

    q: (P, n, hd) queries of the n suffix rows (not scaled); keys:
    (P, n_prefix + n, hd), the prefix keys followed by the suffix rows' own
    keys.  Suffix row i sees every prefix key and suffix keys 0..i.  Returns
    (P, n_prefix) float32.
    """
    n = q.shape[1]
    assert keys.shape[1] == n_prefix + n, (keys.shape, n_prefix, n)
    logits = (q @ keys.transpose(1, 2)) * scale
    idx = torch.arange(n, device=q.device)
    hidden = idx.view(1, 1, n) > idx.view(1, n, 1)
    logits[:, :, n_prefix:] = logits[:, :, n_prefix:].masked_fill(
        hidden, float("-inf"))
    return logits.float().softmax(dim=-1)[:, :, :n_prefix].sum(dim=1)


def head_topk(importance, k):
    """Per head, the k most important candidates: (P, N) -> (P, k) indices."""
    return torch.topk(importance, k, dim=1, largest=True).indices


def mean_pairwise_jaccard(indices, n=None):
    """Mean Jaccard of the per-head index sets over every pair of heads.

    Same arithmetic as the public ``cal_jaccard_sim`` (integer set sizes,
    Python float division, plain mean), so the comparison with the threshold
    gives the same answer on a boundary value.
    """
    heads, k = indices.shape
    n = int(indices.max()) + 1 if n is None else int(n)
    masks = torch.zeros(heads, n, dtype=torch.bool, device=indices.device)
    masks.scatter_(1, indices, True)
    pairs = [(a, b) for a in range(heads) for b in range(a + 1, heads)]
    if not pairs:
        return 1.0, masks
    inter = torch.stack([(masks[a] & masks[b]).sum() for a, b in pairs])
    sizes = masks.sum(dim=1)
    union = torch.stack([sizes[a] + sizes[b] for a, b in pairs]) - inter
    vals = [(i / u) if u > 0 else 0
            for i, u in zip(inter.tolist(), union.tolist())]
    return sum(vals) / len(vals), masks


def vote_consensus(masks, k):
    """The k candidates with the most votes: (P, N) bool -> sorted (k,).

    The public code sorts the vote counts with an unstable ``argsort`` and
    takes the first k, so the order inside one vote count is not defined
    there.  Here it is: lower index first.
    """
    n = masks.shape[1]
    votes = masks.sum(dim=0).to(torch.int64)
    index = torch.arange(n, device=masks.device, dtype=torch.int64)
    rank_key = votes * n + (n - 1 - index)
    chosen = torch.topk(rank_key, k).indices
    return torch.sort(chosen).values, votes


def select_tokens(importance, k, threshold=SIM_THRESHOLD):
    """Per-head top-k -> mean Jaccard -> vote consensus or fallback.

    importance: (P, N); candidates that must not be chosen carry ``-inf`` (k
    never exceeds the number of finite candidates).  Returns a dict:
    ``head_topk`` (P, k), ``similarity``, ``fallback``, ``selected`` (sorted
    indices: k of them on consensus, every index 0..N-1 on fallback) and
    ``votes`` (N,).
    """
    top = head_topk(importance, k)
    sim, masks = mean_pairwise_jaccard(top, importance.shape[1])
    if sim >= threshold:
        selected, votes = vote_consensus(masks, k)
        return {"head_topk": top, "similarity": sim, "fallback": False,
                "selected": selected, "votes": votes}
    return {"head_topk": top, "similarity": sim, "fallback": True,
            "selected": torch.arange(importance.shape[1],
                                     device=importance.device),
            "votes": masks.sum(dim=0).to(torch.int64)}


# ------------------------------------------------------------------- order
def descending_order(summed):
    """stored -> original: descending importance, lower index first on ties."""
    return torch.argsort(torch.as_tensor(summed), descending=True,
                         stable=True)


def reverse_index(order):
    """original -> stored position (the public ``reverse_indices``)."""
    order = torch.as_tensor(order, dtype=torch.long)
    out = torch.full_like(order, -1)
    out[order] = torch.arange(order.numel(), dtype=torch.long)
    return out


def layer_order(summed, separators):
    """Stored order of one layer: [separators | image tokens, descending].

    summed: (v_num,) importance of every visual token of the layer (column
    sums over every row, summed over all heads).  Separators are always
    loaded, so they are kept together at the front (raster order); the image
    tokens follow in the public code's descending order.
    """
    summed = torch.as_tensor(summed, dtype=torch.float32)
    sep = sorted(int(i) for i in separators)
    is_sep = torch.zeros(summed.numel(), dtype=torch.bool)
    if sep:
        is_sep[torch.tensor(sep, dtype=torch.long)] = True
    order = descending_order(summed)
    order = order[~is_sep[order]]
    return sep + [int(i) for i in order]


# ============================================================ turn-1 capture
class Turn1AttentionStats:
    """Column sums of the image tokens taken from a normal turn-1 inference.

    Wraps the decoder's eager attention function while the request runs and
    reads the attention probabilities the model itself computed -- no extra
    forward.  Per layer, from the prefill only:

      prefix_rows   (probe, v_num): probe heads, rows of the prefix positions
                    (system prompt + image), column sums over the image
                    columns.  Question-independent; a later request adds the
                    suffix rows' share to this.
      order_scores  (v_num,): every head, every row (prefix + this turn-1
                    suffix), column sums -- the statistic the public
                    ``--generate-mapping-list`` path sorts by.
    """

    def __init__(self, v_start, v_num, probe=len(PROBE_HEADS)):
        self.v_start, self.v_num, self.probe = int(v_start), int(v_num), int(probe)
        self._prefix = {}
        self._order = {}
        self.rows = None
        self.seconds = 0.0
        self.calls = 0
        self._prev = None

    def __enter__(self):
        if self._prev is not None:
            raise RuntimeError("Turn1AttentionStats is single-use")
        self._prev = ml.eager_attention_forward
        ml.eager_attention_forward = self._forward
        return self

    def __exit__(self, *exc):
        assert ml.eager_attention_forward == self._forward, \
            "eager attention was replaced while the capture was active"
        ml.eager_attention_forward = self._prev

    def _forward(self, module, query, key, value, attention_mask, **kw):
        out = self._prev(module, query, key, value, attention_mask, **kw)
        li = getattr(module, "layer_idx", None)
        if li is None or li in self._prefix:
            return out                      # decoding steps: already captured
        weights = out[1]
        cuda = weights.is_cuda
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        v0, p0 = self.v_start, self.v_start + self.v_num
        assert weights.dim() == 4 and weights.shape[0] == 1 \
            and weights.shape[-2] == weights.shape[-1] >= p0, tuple(weights.shape)
        self._prefix[li] = weights[0, :self.probe, :p0, v0:p0].sum(
            dim=1, dtype=torch.float32)
        self._order[li] = weights[0, :, :, v0:p0].sum(
            dim=(0, 1), dtype=torch.float32)
        self.rows = int(weights.shape[-1])
        if cuda:
            torch.cuda.synchronize()
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        return out

    def result(self):
        """(prefix_rows (L, probe, v_num), order_scores (L, v_num)) on the host."""
        layers = sorted(self._prefix)
        assert layers == list(range(len(layers))) and layers, layers
        prefix = torch.stack([self._prefix[li] for li in layers]).cpu()
        order = torch.stack([self._order[li] for li in layers]).cpu()
        return prefix, order


# ================================================================== storage
def probe_block_bytes(meta):
    """(probe-key bytes, prefix-row trailer bytes) of one layer's sidecar."""
    vn, probe, hd = (int(meta["v_token_num"]), int(meta["probe_heads"]),
                     int(meta["head_dim"]))
    item = np.dtype(_NP[meta["dtype"]]).itemsize
    return vn * probe * hd * item, probe * vn * 4


def write_official_layout(store_dir, prefix_rows, order_scores,
                          metadata_update=None):
    """Turn a FRESH raster store into the IMPRESS layout; return timings.

    Per layer: K, V and the probe-key sidecar are rewritten in
    ``layer_order(order_scores[layer], separators)``; the layer's prefix-row
    importance (probe, v_num), in the same stored order, is appended to the
    probe sidecar as little-endian float32 so that one read returns both.
    ``meta.json`` records the per-layer order; ``sep_kv.bin`` is rebuilt.
    """
    store_dir = Path(store_dir)
    meta = load_meta(store_dir)
    if meta.get("reordered") or meta.get("order"):
        raise ValueError("the IMPRESS layout is built from a fresh raster store")
    L, vn, hd = int(meta["num_layers"]), int(meta["v_token_num"]), int(meta["head_dim"])
    H, probe = int(meta["num_heads"]), int(meta["probe_heads"])
    prefix_rows = torch.as_tensor(prefix_rows, dtype=torch.float32)
    order_scores = torch.as_tensor(order_scores, dtype=torch.float32)
    if tuple(prefix_rows.shape) != (L, probe, vn) or tuple(order_scores.shape) != (L, vn):
        raise ValueError(f"expected ({L}, {probe}, {vn}) and ({L}, {vn}), got "
                         f"{tuple(prefix_rows.shape)} and {tuple(order_scores.shape)}")
    if not (torch.isfinite(prefix_rows).all() and torch.isfinite(order_scores).all()):
        raise ValueError("turn-1 attention statistics are not finite")
    dt = _NP[meta["dtype"]]
    separators = [int(i) for i in meta["newline_idx"]]
    widths = (("k", H), ("v", H), ("probe_k", probe))
    timing = {name: {"read_ms": 0.0, "permute_ms": 0.0, "write_ms": 0.0, "bytes": 0}
              for name, _ in widths}
    orders = []
    order_ms = 0.0
    started = time.perf_counter()
    for li in range(L):
        t0 = time.perf_counter()
        perm = layer_order(order_scores[li], separators)
        order_ms += (time.perf_counter() - t0) * 1e3
        if sorted(perm) != list(range(vn)):
            raise ValueError("layer order is not a permutation")
        perm_np = np.asarray(perm)
        trailer = np.ascontiguousarray(
            prefix_rows[li].numpy()[:, perm_np].astype("<f4"))
        ldir = store_dir / f"layer_{li:02d}"
        for name, width in widths:
            p = ldir / f"{name}.bin"
            t0 = time.perf_counter()
            a = np.fromfile(p, dtype=dt).reshape(vn, width, hd)
            t1 = time.perf_counter()
            rewritten = np.ascontiguousarray(a[perm_np])
            t2 = time.perf_counter()
            tmp = p.with_name(p.name + ".layout-tmp")
            if tmp.exists():
                raise FileExistsError(f"stale layout temp file: {tmp}")
            with open(tmp, "wb") as f:
                f.write(rewritten.tobytes())
                if name == "probe_k":
                    f.write(trailer.tobytes())
            os.replace(tmp, p)
            t3 = time.perf_counter()
            entry = timing[name]
            entry["read_ms"] += (t1 - t0) * 1e3
            entry["permute_ms"] += (t2 - t1) * 1e3
            entry["write_ms"] += (t3 - t2) * 1e3
            entry["bytes"] += int(rewritten.nbytes) + (
                int(trailer.nbytes) if name == "probe_k" else 0)
        orders.append([int(x) for x in perm])
    rewrite_ms = (time.perf_counter() - started) * 1e3

    key_bytes, trailer_bytes = probe_block_bytes(meta)
    meta["order"] = orders
    meta["order_is_per_layer"] = True
    meta["newline_stored"] = [list(range(len(separators))) for _ in range(L)]
    meta["reordered"] = True
    meta["physical_layout"] = LAYOUT
    meta["probe_sidecar_trailer"] = {
        "content": TRAILER, "dtype": "<f4", "shape": [probe, vn],
        "offset_bytes": key_bytes, "bytes": trailer_bytes,
        "order": "stored", "rows": "system prompt + image positions",
    }
    if metadata_update:
        meta.update(metadata_update)
    t0 = time.perf_counter()
    with open(store_dir / META_NAME, "w") as f:
        json.dump(meta, f, indent=1)
    meta_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    write_separator_sidecar_from_store(store_dir, load_meta(store_dir))
    sidecar_ms = (time.perf_counter() - t0) * 1e3
    return {"per_kind": timing, "rewrite_ms": rewrite_ms, "order_ms": order_ms,
            "meta_write_ms": meta_ms, "separator_sidecar_ms": sidecar_ms,
            "total_ms": rewrite_ms + meta_ms + sidecar_ms,
            "trailer_bytes_per_layer": trailer_bytes}


def validate_official_store(meta):
    """Fail closed unless ``meta`` describes a store this path can serve."""
    L, vn = int(meta["num_layers"]), int(meta["v_token_num"])
    seps = sorted(int(i) for i in meta["newline_idx"])
    ok = (meta.get("physical_layout") == LAYOUT and meta.get("reordered") is True
          and meta.get("order_is_per_layer") is True
          and meta.get("storage_policy", "full") == "full"
          and meta.get("probe_sidecar_trailer", {}).get("content") == TRAILER
          and int(meta["probe_heads"]) >= len(PROBE_HEADS)
          and len(meta.get("order", [])) == L
          and meta.get("newline_stored") == [list(range(len(seps)))] * L)
    if ok:
        for order in meta["order"]:
            ok = ok and len(order) == vn and sorted(order[:len(seps)]) == seps
    if not ok:
        raise ValueError("store is not an IMPRESS (public-implementation) layout")
    return True


def read_probe_block(reader, layer, counter=None):
    """One read of a layer's probe sidecar.

    Returns (probe keys (v_num, probe, hd) as stored, prefix-row importance
    (probe, v_num) float32, stored order).  Counted under kind ``probe``.
    """
    m = reader.meta
    vn, probe, hd = int(m["v_token_num"]), int(m["probe_heads"]), int(m["head_dim"])
    key_bytes, trailer_bytes = probe_block_bytes(m)
    blobs = reader._read(layer, "probe_k", [(0, key_bytes + trailer_bytes)],
                         counter, "probe", m["n_chunks_per_layer"])
    buf = blobs[0][1]
    keys = np.frombuffer(buf, dtype=reader.np_dtype, count=vn * probe * hd)
    rows = np.frombuffer(buf, dtype="<f4", count=probe * vn, offset=key_bytes)
    return (torch.from_numpy(keys.copy()).view(vn, probe, hd),
            torch.from_numpy(rows.copy()).view(probe, vn))


# ================================================================= selector
class OfficialImpressSelector:
    """Per-layer identify-then-load following the public implementation.

    Layer by layer, as a forward pre-hook on the decoder layer:

      1. read the probe sidecar (probe heads' keys + stored prefix-row
         importance) -- one read; with ``prefetch="next_layer"`` the read of
         layer i+1 is issued by one background worker when the hook of layer i
         ends (layer 0 is read in the hook);
      2. suffix rows' importance for the probe heads, plus the stored
         prefix-row importance; separators are not candidates;
      3. ``select_tokens``;
      4. consensus: read the 64-token K and V chunks that hold the selected
         tokens or a separator, mask every image token that was not selected;
         fallback: read the layer's whole K and V and mask nothing.

    Decoding steps do not load or select anything.
    """

    def __init__(self, runner, ctx, ratio=KEEP_RATIO, probe=len(PROBE_HEADS),
                 sim_threshold=SIM_THRESHOLD, counter=None, prefetch="none",
                 prefetch_executor=None):
        if prefetch not in PREFETCH:
            raise ValueError(f"prefetch must be one of {PREFETCH}, got {prefetch!r}")
        m = ctx.meta
        validate_official_store(m)
        self.runner, self.ctx = runner, ctx
        self.ratio, self.probe = float(ratio), int(probe)
        self.threshold = float(sim_threshold)
        self.prefetch = prefetch
        self.io = counter if counter is not None else IOCounter()
        self.k_keep = keep_count(m["n_spatial"], ratio)
        self.n_sep = len(m["newline_idx"])
        self.sep = torch.arange(self.n_sep, dtype=torch.long)   # stored positions
        self.log = []
        self.probe_log = []
        self.keep_masks = {}            # layer -> (v_num,) bool, host
        self.done = set()
        self.hook_seconds = 0.0
        self.read_blocked_seconds = 0.0
        self._diag = {}                 # layer -> tensors, read after the request
        self._handles = []
        self._executor = prefetch_executor
        self._owns_executor = False
        self._pending = None            # (layer, Future) or None

    # --------------------------------------------------------------- hooks
    def __enter__(self):
        if self.prefetch == "next_layer" and self._executor is None:
            from concurrent.futures import ThreadPoolExecutor
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="impress-official-prefetch")
            self._owns_executor = True
        for li, layer in enumerate(self.runner.layers):
            self._handles.append(layer.register_forward_pre_hook(
                self._hook(li, layer), with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        self._discard_pending()
        if self._owns_executor:
            self._executor.shutdown(wait=True)
            self._executor = None
            self._owns_executor = False

    def _hook(self, li, layer):
        def hook(module, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            if li in self.done:
                return                         # decoding: selection is frozen
            t0 = time.perf_counter()
            self._run(li, layer, h, kwargs)
            self.done.add(li)
            if self.prefetch == "next_layer":
                self._submit_next(li)
            self.hook_seconds += time.perf_counter() - t0
        return hook

    # ------------------------------------------------------------ prefetch
    def _prefetch_read(self, li):
        counter = IOCounter()
        t0 = time.perf_counter()
        block = read_probe_block(self.ctx.reader, li, counter)
        return block, counter, time.perf_counter() - t0

    def _submit_next(self, li):
        nxt = li + 1
        if nxt >= self.ctx.meta["num_layers"] or nxt in self.done:
            return
        self._pending = (nxt, self._executor.submit(self._prefetch_read, nxt))

    def _discard_pending(self):
        pending, self._pending = self._pending, None
        if pending is not None and not pending[1].cancel():
            pending[1].exception()             # wait; never raise from here

    def _fold(self, counter):
        for kind, d in counter.per_kind.items():
            self.io.record(kind, d["bytes"], d["seconds"], preads=d["preads"],
                           units=d["units"])

    def _probe_block(self, li):
        t0 = time.perf_counter()
        pending, self._pending = self._pending, None
        if pending is not None and pending[0] != li:
            if not pending[1].cancel():
                pending[1].exception()
            pending = None
        if pending is not None:
            future = pending[1]
            ready = future.done()
            block, counter, read_s = future.result()
            blocked = time.perf_counter() - t0
            self.read_blocked_seconds += blocked
            self._fold(counter)
            self.probe_log.append({"layer": li, "source": "prefetch",
                                   "read_s": read_s, "blocked_s": blocked,
                                   "wait_s": blocked,
                                   "ready_on_arrival": bool(ready)})
            return block
        block = read_probe_block(self.ctx.reader, li, self.io)
        blocked = time.perf_counter() - t0
        self.read_blocked_seconds += blocked
        self.probe_log.append({"layer": li, "source": "hook", "read_s": blocked,
                               "blocked_s": blocked, "wait_s": 0.0,
                               "ready_on_arrival": None})
        return block

    def _blocking(self, fn, *args):
        t0 = time.perf_counter()
        try:
            return fn(*args)
        finally:
            self.read_blocked_seconds += time.perf_counter() - t0

    # ----------------------------------------------------------- one layer
    def _run(self, li, layer, h, kwargs):
        from mmimpress.serve import BIAS, bias_from_keep
        m, ctx = self.ctx.meta, self.ctx
        dev = h.device
        H, hd = m["num_heads"], m["head_dim"]
        v0, vn = m["v_token_start"], m["v_token_num"]
        nc = m["n_chunks_per_layer"]
        n = h.shape[1]
        P = self.probe

        attn = layer.self_attn
        hn = layer.input_layernorm(h)
        q = attn.q_proj(hn).view(1, n, H, hd).transpose(1, 2)
        kn = attn.k_proj(hn).view(1, n, H, hd).transpose(1, 2)
        cos, sin = kwargs["position_embeddings"]
        q, kn = ml.apply_rotary_pos_emb(q, kn, cos, sin)
        q, kn = q[0, :P], kn[0, :P]                         # (P, n, hd)

        probe_keys, prefix_rows = self._probe_block(li)
        pk = probe_keys.to(dev, q.dtype).permute(1, 0, 2)[:P]      # (P, vn, hd)
        sys_k = ctx.cache.k[li][0, :P, :v0]
        keys = torch.cat([sys_k, pk, kn], dim=1)
        suffix = suffix_row_importance_from_qk(
            q, keys, v0 + vn, hd ** -0.5)[:, v0:]                  # (P, vn)
        importance = prefix_rows[:P].to(dev) + suffix
        sep = self.sep.to(dev)
        importance[:, sep] = float("-inf")
        sel = select_tokens(importance, self.k_keep, self.threshold)

        if sel["fallback"]:
            for kind in ("k", "v"):
                block = self._blocking(ctx.reader.read_full, li, kind, self.io)
                ctx.cache.write_full(li, kind, block)
            keep = torch.ones(vn, dtype=torch.bool)
            self.log.append({"layer": li, "mode": "fallback",
                             "sim": sel["similarity"], "chunks": nc,
                             "k_chunks": nc, "v_chunks": nc})
            # nothing is masked: no BIAS entry for this layer
        else:
            chosen = sel["selected"].cpu()
            cids = chunks_for_tokens(chosen.tolist() + self.sep.tolist(), vn,
                                     m["chunk_size"])
            for kind in ("k", "v"):
                rows, vals = self._blocking(ctx.reader.read_chunks, li, kind,
                                            cids, self.io)
                ctx.cache.write(li, kind, rows, vals)
            keep = torch.zeros(vn, dtype=torch.bool)
            keep[chosen] = True
            keep[self.sep] = True
            self.log.append({"layer": li, "mode": "consensus",
                             "sim": sel["similarity"], "chunks": len(cids),
                             "k_chunks": len(cids), "v_chunks": len(cids)})
            BIAS[li] = bias_from_keep(keep, m, dev)
        self.keep_masks[li] = keep
        self._diag[li] = (prefix_rows[:P], suffix, sel["head_topk"])

    # -------------------------------------------------------------- summary
    def read_wait_stats(self):
        return {"read_blocked_ms": self.read_blocked_seconds * 1e3,
                "readahead": None}

    def stats(self):
        nc = self.ctx.meta["n_chunks_per_layer"]
        n = max(1, len(self.log))
        sims = [r["sim"] for r in self.log]
        fb = [r for r in self.log if r["mode"] == "fallback"]
        return {
            "selection_path": "impress_official",
            "fallback_rate": len(fb) / n,
            "consensus_rate": (len(self.log) - len(fb)) / n,
            "mean_jaccard": (sum(sims) / len(sims)) if sims else None,
            "threshold": self.threshold, "k_keep": self.k_keep,
            "keep_ratio": self.ratio, "probe_heads": self.probe,
            "hook_ms": self.hook_seconds * 1e3,
            "touched_k_chunk_fraction": float(np.mean(
                [r["k_chunks"] for r in self.log])) / nc if self.log else None,
            "touched_v_chunk_fraction": float(np.mean(
                [r["v_chunks"] for r in self.log])) / nc if self.log else None,
            "fallback_layers": len(fb),
            "prefetch": self.prefetch,
            "probe_read_ms": 1e3 * sum(r["read_s"] for r in self.probe_log),
            "probe_pread_ms": 1e3 * float(
                self.io.per_kind.get("probe", {}).get("seconds", 0.0)),
            "prefetch_wait_ms": 1e3 * sum(r["wait_s"] for r in self.probe_log),
            "probe_hook_blocked_ms": 1e3 * sum(
                r["blocked_s"] for r in self.probe_log),
            "prefetched_probe_reads": sum(
                r["source"] == "prefetch" for r in self.probe_log),
            "prefetch_ready_on_arrival": sum(
                bool(r["ready_on_arrival"]) for r in self.probe_log),
        }

    def diagnostics(self):
        """Per-layer figures for the report; call after the request.

        prefix_share   prefix rows' part of the probe heads' importance over
                       the image candidates: sum(prefix) / sum(prefix + suffix)
        order_overlap_head   mean over probe heads of |head top-k intersect
                       the first k image tokens of the stored order| / k
        order_overlap_selected   the same for the consensus selection (None on
                       a fallback layer)
        """
        lo, hi = self.n_sep, self.n_sep + self.k_keep
        out = []
        for row in self.log:
            li = row["layer"]
            prefix, suffix, top = self._diag[li]
            prefix = prefix.to(suffix.device)
            p = float(prefix[:, self.n_sep:].sum())
            s = float(suffix[:, self.n_sep:].sum())
            head = float(((top >= lo) & (top < hi)).float().mean())
            chosen = None
            if row["mode"] == "consensus":
                keep = self.keep_masks[li]
                chosen = float(keep[lo:hi].sum()) / self.k_keep
            out.append({"layer": li,
                        "prefix_share": p / (p + s) if (p + s) > 0 else None,
                        "order_overlap_head": head,
                        "order_overlap_selected": chosen})
        return out


def alternative_chunk_counts(keep_masks, layer_log, meta, position):
    """Chunks the request's loads would touch under another stored order.

    Computed, never read.  ``position(layer)`` returns, for every stored
    position of that layer, the position the same token would have in the
    alternative order.  A fallback layer reads every chunk in any order.
    """
    cs, nc = int(meta["chunk_size"]), int(meta["n_chunks_per_layer"])
    counts = []
    for row in layer_log:
        li = int(row["layer"])
        if row["mode"] == "fallback":
            counts.append(nc)
            continue
        where = torch.as_tensor(position(li), dtype=torch.long)
        loaded = keep_masks[li].nonzero(as_tuple=True)[0]
        counts.append(int(torch.unique(where[loaded] // cs).numel()))
    n = max(1, len(layer_log))
    return {"chunks": counts, "chunk_fraction": float(sum(counts)) / (n * nc)}


# ================================================================== request
@torch.no_grad()
def request_official(server, ctx, suffix_ids, *, ratio=KEEP_RATIO,
                     probe=len(PROBE_HEADS), sim_threshold=SIM_THRESHOLD,
                     prefetch="none", cold=True, selector_out=None):
    """Serve one follow-up turn through ``OfficialImpressSelector``.

    Same timer boundaries as ``Server.request``: cold-cache eviction, the
    request counter and the prefetch worker are prepared before the timer;
    TTFT ends right after the first output token.
    """
    from mmimpress.serve import BIAS
    BIAS.clear()
    if cold:
        ctx.reader.drop_all()
    counter = IOCounter()
    dev = server.runner.model.device
    suffix_ids = suffix_ids.to(dev)
    executor = server._prefetch_pool() if prefetch == "next_layer" else None

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    sel = OfficialImpressSelector(
        server.runner, ctx, ratio=ratio, probe=probe,
        sim_threshold=sim_threshold, counter=counter, prefetch=prefetch,
        prefetch_executor=executor)
    if selector_out is not None:
        selector_out.append(sel)
    cache = ctx.cache.new_request()
    with sel:
        answer, first_token_id, timing = server._decode(
            cache, suffix_ids, ctx.meta["prefix_len"])
    BIAS.clear()
    del cache
    return {
        "answer": answer,
        "first_token_id": first_token_id,
        "ttft": timing["first_token_at"] - t0,
        "decode_latency": timing["decode_ms"] / 1e3,
        "e2e_latency": timing["finished_at"] - t0,
        "prefill_ms": timing["prefill_ms"],
        "generated_tokens": timing["generated_tokens"],
        "generated_token_ids": timing["generated_token_ids"],
        "generated_token_count": timing["generated_token_count"],
        "n_score_rows": int(suffix_ids.numel()),
        "io": counter.summary(),
        "core_started_at_s": t0,
        "first_token_at_s": timing["first_token_at"],
        "model_finished_at_s": timing["finished_at"],
        "postprocess_finished_at_s": time.perf_counter(),
        "read_wait": sel.read_wait_stats(),
        **sel.stats(),
    }

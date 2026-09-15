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

"""Serving of LLaVA-1.5 packed prefix stores (contract docs/llava15_mt_gqa_contract.md).

The request-time work is the LLaVA-NeXT packed path (``mmimpress.packed_serve``)
with the separator scatter removed (LLaVA-1.5 has no separator) and the total
ratio taken from the store's own budget:

* ``Llava15PackedContext`` (activation, outside every request timer): metadata
  validation, integrity envelope, layout artifact, system KV, dense GPU
  ``PrefixCache``, pinned host buffer of the packed file size;
* ``Llava15PrefixSelector.prepare`` (inside the timer): ONE ``os.preadv`` of
  ``kv_packed.bin`` into the pinned buffer, then per layer the v1 scatter
  (``PrefixCache.write`` with stored rows ``[0, k_l)``) and the v1
  ``bias_from_keep`` mask with rows ``[0, k_l)`` visible;
* ``request_llava15_packed``: the v1 ``Server._decode`` path and timing
  boundaries of ``mmimpress.packed_serve.request_packed``;
* ``Llava15StreamSelector`` / ``request_llava15_packed(readahead="layer_stream")``
  (docs/llava15_mtvqa_impress_contract.md): the asynchronous variant, the
  LLaVA-1.5 counterpart of ``mmimpress.packed_serve.PackedStreamSelector``.
  One background worker ``preadv``s the packed file one layer range at a time,
  in layer order, into the same pinned buffer while the model computes; a
  forward pre-hook on decoder layer i waits for layer i's range and then runs
  the same per-layer scatter + mask as the synchronous path.  The default
  ``readahead="none"`` is the unchanged synchronous single-``preadv`` path.

The reader (``PackedReader``), the dense cache (``PrefixCache``), the mask
builder (``bias_from_keep`` / ``BIAS``), the decode loop and the syscall trace
(``PackedReadTrace``) are the existing objects, imported unchanged.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from mmimpress.llava15 import N_VISUAL, NUM_LAYERS
from mmimpress.llava15.store import (LAYOUT_POLICY, keep_counts, meta_budget, uniform_equivalent,
                                     validate_llava15_packed_meta)
from mmimpress.packed_serve import PackedReadTrace  # noqa: F401  (re-exported for harnesses)
from mmimpress.packed_store import PackedReader
from mmimpress.serve import BIAS, IOCounter, PrefixCache, bias_from_keep, suffix_ids_from_prompt
from mmimpress.store import load_meta


class Llava15PackedContext:
    def __init__(self, store_dir, device, pin: bool = True, expected_budget=None):
        from mmimpress.prefix25 import verify_integrity
        self.dir = Path(store_dir)
        self.meta = load_meta(self.dir)
        validate_llava15_packed_meta(self.meta, expected_budget)
        if not (self.dir / "integrity.json").exists():
            raise ValueError("packed store lacks its integrity completion envelope")
        self.integrity_activation = verify_integrity(self.dir)
        m = self.meta
        self.n_sep = 0
        self.reader = PackedReader(self.dir, m, pin=pin)
        sys_kv = torch.load(self.dir / "sys_kv.pt", weights_only=True)
        self.cache = PrefixCache(m, sys_kv, device)
        self.v_hidden = None
        self._layout_validated = False

    def separator_positions(self, layer):
        return []

    def validate_packed_layout(self):
        """Activation-time permutation / artifact / file-length checks."""
        if self._layout_validated:
            return
        from mmimpress.cvpr25 import permutation_sha256, visionzip_repack_order
        m = self.meta
        vn = int(m["v_token_num"])
        assert vn == N_VISUAL and int(m["num_layers"]) == NUM_LAYERS and m["newline_idx"] == []
        assert int(m["prefix_len"]) == int(m["v_token_start"]) + vn
        assert m.get("physical_layout") == LAYOUT_POLICY and m.get("separator_policy") == "none_llava15_has_no_separator"
        order = [int(x) for x in m["order"]]
        assert sorted(order) == list(range(vn))
        assert m.get("permutation_sha256") == permutation_sha256(order)
        artifact = torch.load(self.dir / "visionzip_layout.pt", map_location="cpu", weights_only=True)
        scores = torch.as_tensor(artifact["token_score_original"]).flatten()
        assert scores.numel() == vn and bool(torch.isfinite(scores).all())
        assert [int(x) for x in artifact["newline_original"]] == []
        assert [int(x) for x in artifact["stored_to_original"]] == order
        inverse = [0] * vn
        for stored, original in enumerate(order):
            inverse[original] = stored
        assert [int(x) for x in artifact["original_to_stored"]] == inverse
        assert m.get("inverse_permutation_sha256") == permutation_sha256(inverse)
        assert visionzip_repack_order(scores, []) == order
        assert artifact.get("layout_uses_dataset_question") is False
        assert artifact.get("llm_used_for_layout_scoring") is False
        assert int(artifact.get("calibration_questions", -1)) == 0
        assert (self.dir / m["packed_file"]).stat().st_size == int(m["packed_total_bytes"])
        self._layout_validated = True

    def close(self):
        self.reader.close()
        self.cache = None


class Llava15PrefixSelector:
    """Query-independent layer-wise prefix selection served from a packed store."""

    def __init__(self, runner, ctx: Llava15PackedContext, counter=None):
        self.runner, self.ctx = runner, ctx
        self.io = counter if counter is not None else IOCounter()
        self.t = {"select": 0.0, "chunk_io": 0.0, "scatter": 0.0, "query": 0.0}
        m = ctx.meta
        # Full metadata validation ran at activation; inside the request timer only the
        # budget document is re-validated and k_l recomputed (as the LLaVA-NeXT selector does).
        self.budget = meta_budget(m)
        vn = int(m["v_token_num"])
        n_content = int(m["n_spatial"])
        if n_content != vn or n_content != N_VISUAL:
            raise ValueError("visual-content count disagrees with the LLaVA-1.5 geometry")
        counts = keep_counts(self.budget, n_content)
        if counts != [int(x) for x in m["layer_keep_counts"]]:
            raise ValueError("layerwise keep counts disagree with store")
        self.n_content = n_content
        self.k_per_layer = counts
        self.k_max = max(counts)
        self.k_target = uniform_equivalent(self.budget, n_content)
        self.selected_stored_ids = list(range(self.k_max))
        self.selected_original_ids = [int(x) for x in m["order"][:self.k_max]]
        self.kept_tokens_per_layer = []
        self.read_bytes = 0
        self.read_calls = 0
        self.read_ms = 0.0

    @torch.no_grad()
    def prepare(self):
        ctx, m = self.ctx, self.ctx.meta
        dev = self.runner.model.device
        L = int(m["num_layers"])
        t0 = time.perf_counter()
        n, dt = ctx.reader.read_all(self.io)
        self.t["chunk_io"] += time.perf_counter() - t0
        self.read_bytes, self.read_calls, self.read_ms = n, 1, dt * 1e3
        for li in range(L):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            self._scatter_layer(li, dev)
            torch.cuda.synchronize()
            self.t["scatter"] += time.perf_counter() - t0

    def _scatter_layer(self, li, dev):
        """Layer ``li``: stored rows ``[0, k_l)`` -> GPU cache, and the mask.

        Shared verbatim by the synchronous ``prepare`` loop and the
        asynchronous per-layer hook, so both issue the same GPU writes.
        """
        ctx, m = self.ctx, self.ctx.meta
        vn = int(m["v_token_num"])
        k = self.k_per_layer[li]
        for kind in ("k", "v"):
            vals = ctx.reader.layer_view(li, kind)
            assert vals.shape[0] == k
            if k:
                ctx.cache.write(li, kind, torch.arange(k, dtype=torch.long), vals)
        keep = torch.zeros(vn, dtype=torch.bool)
        keep[:k] = True
        assert int(keep.sum()) == k
        self.kept_tokens_per_layer.append(int(keep.sum()))
        BIAS[li] = bias_from_keep(keep, m, dev)

    def stats(self):
        m = self.ctx.meta
        L = int(m["num_layers"])
        row_bytes = int(m["packed_row_bytes"])
        per_layer_bytes = [2 * k * row_bytes for k in self.k_per_layer]
        assert self.kept_tokens_per_layer == list(self.k_per_layer)
        return {
            "store_format": m["store_format"], "selection_mode": "prefix", "budget_unit": "visual_kv",
            "ratio": float(self.budget["total_ratio"]), "budget_policy": m["budget_policy"],
            "layer_budget_id": str(m["layer_budget_id"]), "layer_budget_sha256": str(m["layer_budget_sha256"]),
            "layer_keep_ratios": [float(x) for x in m["layer_keep_ratios"]],
            "N_content": self.n_content, "N_structural": 0, "k_uniform_equivalent": self.k_target,
            "k_total": int(sum(self.k_per_layer)), "k_per_layer": list(self.k_per_layer), "k_max": int(self.k_max),
            "attended_content_kv_count_per_layer": list(self.k_per_layer),
            "selected_stored_ids": self.selected_stored_ids, "selected_original_ids": self.selected_original_ids,
            "selection_rule": "layer l attends stored rows [0, k_l) = selected_original_ids[:k_per_layer[l]]",
            "keep_count_per_layer": self.kept_tokens_per_layer,
            "planned_normal_read_bytes_per_layer": per_layer_bytes,
            "planned_packed_read": {"offset_bytes": 0, "length_bytes": int(m["packed_total_bytes"])},
            "logical_content_retention_ratio": sum(self.k_per_layer) / (L * self.n_content),
            "logical_content_retention_ratio_per_layer": [k / self.n_content for k in self.k_per_layer],
            "logical_required_content_bytes": int(sum(per_layer_bytes)),
            "static_score_calls": 0, "query_score_calls": 0, "diversity_calls": 0, "separator_policy": "none",
            "select_ms": self.t["select"] * 1e3, "query_ms": self.t["query"] * 1e3,
            "chunk_io_ms": self.t["chunk_io"] * 1e3, "scatter_ms": self.t["scatter"] * 1e3, "hook_ms": 0.0,
            "packed_read_ms": self.read_ms, "read_calls": self.read_calls, "read_syscall": "preadv",
            "directcopy": False, "fallback_rate": 0.0,
        }


READAHEAD_POLICIES = ("none", "layer_stream")


class Llava15StreamSelector(Llava15PrefixSelector):
    """Asynchronous variant: the packed file is read in layer order by one
    background worker while the model computes.

    Same store, same bytes, same per-layer GPU writes and masks as
    ``Llava15PrefixSelector``.  The differences are who reads and when:

    * ``begin()`` (called after the TTFT timer started) hands one ``[K | V]``
      byte range per layer -- ``PackedReader.stream_plan()`` without its
      separator block, which is empty for LLaVA-1.5 -- to a
      ``mmimpress.readahead.LayerReadAhead`` worker, which ``preadv``s each
      range into its own place in the pinned buffer;
    * a forward pre-hook on decoder layer i waits until layer i's range has
      arrived, runs the shared ``_scatter_layer`` and lets the layer compute.

    The pinned buffer is still allocated at activation, outside TTFT.
    """

    def __init__(self, runner, ctx: Llava15PackedContext, counter=None, executor=None, max_ahead=None):
        super().__init__(runner, ctx, counter)
        self._executor = executor
        self._owns_executor = False
        self.max_ahead = max_ahead
        plan = ctx.reader.stream_plan()
        if plan[0][0] != "sep" or int(plan[0][2]) != 0:
            raise ValueError("LLaVA-1.5 packed stores have no separator block")
        self._order = [key for key, _, _ in plan[1:]]
        self._plan = {key: (int(off), int(ln)) for key, off, ln in plan[1:]}
        if self._order != list(range(int(ctx.meta["num_layers"]))):
            raise ValueError("stream plan does not cover every layer in order")
        self._stream = None
        self._handles = []
        self.done = set()
        self.hook_seconds = 0.0
        self.read_blocked_seconds = 0.0

    def _read_item(self, key):
        """Worker body: one preadv into the pinned buffer, private counter."""
        offset, length = self._plan[key]
        counter = IOCounter()
        n, dt = self.ctx.reader.read_range(offset, length, counter)
        return n, dt, counter

    def begin(self):
        """Start the read-ahead.  Call after the request timer started."""
        if self._stream is not None:
            return
        from mmimpress.readahead import LayerReadAhead
        if self._executor is None:
            from concurrent.futures import ThreadPoolExecutor
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llava15-packed-readahead")
            self._owns_executor = True
        self._stream = LayerReadAhead(self._executor, self._read_item, self._order, max_ahead=self.max_ahead).start()

    def close(self):
        """Stop the worker loop and leave nothing in flight.  Idempotent."""
        if self._stream is not None:
            self._stream.close()
        if self._owns_executor:
            self._executor.shutdown(wait=True)
            self._executor = None
            self._owns_executor = False

    def __enter__(self):
        self.begin()
        for li, layer in enumerate(self.runner.layers):
            self._handles.append(layer.register_forward_pre_hook(self._hook(li), with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        self.close()

    def _take(self, key):
        t0 = time.perf_counter()
        n, dt, counter = self._stream.take(key)
        self.read_blocked_seconds += time.perf_counter() - t0
        for kind, d in counter.per_kind.items():
            self.io.record(kind, d["bytes"], d["seconds"], preads=d["preads"], units=d["units"])
        self.read_bytes += n
        self.read_calls += 1 if n else 0
        self.read_ms += dt * 1e3

    def _hook(self, li):
        def hook(module, args, kwargs):
            if li in self.done:
                return                     # decoding: nothing left to load
            t0 = time.perf_counter()
            self._take(li)
            t1 = time.perf_counter()
            self._scatter_layer(li, self.runner.model.device)
            self.done.add(li)
            now = time.perf_counter()
            self.t["scatter"] += now - t1
            self.hook_seconds += now - t0
        return hook

    def prepare(self):
        raise RuntimeError("Llava15StreamSelector loads inside layer hooks; use it as a context manager")

    def stats(self):
        st = super().stats()
        m = self.ctx.meta
        ranges = [{"key": str(key), "offset_bytes": int(self._plan[key][0]), "length_bytes": int(self._plan[key][1])}
                  for key in self._order]
        st.update({
            "readahead_policy": "layer_stream", "planned_packed_read": ranges,
            "read_calls": int(self.read_calls), "packed_read_ms": self.read_ms,
            # worker reads overlap computation: neither chunk_io_ms nor hook_ms adds up to TTFT with anything
            "chunk_io_ms": self.read_ms, "hook_ms": self.hook_seconds * 1e3,
            "read_wait": {"read_blocked_ms": self.read_blocked_seconds * 1e3,
                          "readahead": self._stream.stats() if self._stream is not None else None},
        })
        assert self.read_bytes == int(m["packed_total_bytes"])
        return st


@torch.no_grad()
def _request_llava15_packed_stream(server, ctx: Llava15PackedContext, *, suffix_ids=None, prompt_text=None, cold=True,
                                   max_ahead=None):
    """Packed cache hit with the layer-ordered background read-ahead (same timer boundaries)."""
    ctx.validate_packed_layout()
    BIAS.clear()
    if cold:
        ctx.reader.drop_all()
    counter = IOCounter()
    dev = server.runner.model.device
    if suffix_ids is None:
        suffix_ids = suffix_ids_from_prompt(server.runner, prompt_text)
    suffix_ids = suffix_ids.to(dev)
    executor = server._prefetch_pool()            # worker thread exists before the timer
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    sel = Llava15StreamSelector(server.runner, ctx, counter, executor=executor, max_ahead=max_ahead)
    try:
        sel.begin()                                # reads start here, after t0
        cache = ctx.cache.new_request()
        with sel:
            answer, first_token_id, timing = server._decode(cache, suffix_ids, ctx.meta["prefix_len"])
    finally:
        sel.close()
    ttft = timing["first_token_at"] - t0
    e2e = timing["finished_at"] - t0
    BIAS.clear()
    del cache
    st = sel.stats()
    io_summary = counter.summary()
    m = ctx.meta
    st.update({
        "answer": answer, "first_token_id": first_token_id, "ttft": ttft,
        "decode_latency": timing["decode_ms"] / 1e3, "e2e_latency": e2e, "prepare_ms": None,
        "prefill_ms": timing["prefill_ms"], "decode_ms": timing["decode_ms"],
        "model_ms": timing["prefill_ms"] + timing["decode_ms"],
        "generated_tokens": timing["generated_tokens"], "generated_token_ids": timing["generated_token_ids"],
        "generated_token_count": timing["generated_token_count"], "io": io_summary,
        "normal_kv_read_bytes": int(m["packed_total_bytes"]), "separator_read_bytes": 0,
        "normal_kv_preads": sum(1 for _, ln in sel._plan.values() if ln), "separator_preads": 0,
        "read_calls": int(io_summary.get("preads", 0)),
        "total_actual_pread_bytes": int(io_summary.get("bytes", 0)),
        "validated_prefix_layout": "visionzip_image_only_packed", "n_raters": 0,
        "core_started_at_s": t0, "first_token_at_s": timing["first_token_at"],
        "model_finished_at_s": timing["finished_at"], "postprocess_finished_at_s": time.perf_counter(),
    })
    st["selector_ms"] = st["select_ms"] + st["query_ms"]
    return st


@torch.no_grad()
def request_llava15_packed(server, ctx: Llava15PackedContext, *, suffix_ids=None, prompt_text=None, cold=True, image_id="",
                           readahead="none", max_ahead=None):
    """Hook-free packed cache hit with the v1 request timing boundaries.

    ``readahead="layer_stream"`` serves the same store through
    ``Llava15StreamSelector`` (background layer-ordered read, per-layer hooks);
    the timer boundaries are unchanged.  The default is the synchronous path.
    """
    if readahead not in READAHEAD_POLICIES:
        raise ValueError(f"readahead must be one of {READAHEAD_POLICIES}, got {readahead!r}")
    if readahead == "layer_stream":
        return _request_llava15_packed_stream(server, ctx, suffix_ids=suffix_ids, prompt_text=prompt_text, cold=cold,
                                              max_ahead=max_ahead)
    ctx.validate_packed_layout()
    BIAS.clear()
    if cold:
        ctx.reader.drop_all()
    counter = IOCounter()
    dev = server.runner.model.device
    if suffix_ids is None:
        suffix_ids = suffix_ids_from_prompt(server.runner, prompt_text)
    suffix_ids = suffix_ids.to(dev)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    cache = ctx.cache.new_request()
    sel = Llava15PrefixSelector(server.runner, ctx, counter)
    sel.prepare()
    torch.cuda.synchronize()
    t_prep = time.perf_counter() - t0
    answer, first_token_id, timing = server._decode(cache, suffix_ids, ctx.meta["prefix_len"])
    ttft = timing["first_token_at"] - t0
    e2e = timing["finished_at"] - t0
    BIAS.clear()
    del cache
    st = sel.stats()
    io_summary = counter.summary()
    m = ctx.meta
    st.update({
        "answer": answer, "first_token_id": first_token_id, "ttft": ttft,
        "decode_latency": timing["decode_ms"] / 1e3, "e2e_latency": e2e, "prepare_ms": t_prep * 1e3,
        "prefill_ms": timing["prefill_ms"], "decode_ms": timing["decode_ms"],
        "model_ms": timing["prefill_ms"] + timing["decode_ms"],
        "generated_tokens": timing["generated_tokens"], "generated_token_ids": timing["generated_token_ids"],
        "generated_token_count": timing["generated_token_count"], "io": io_summary,
        "normal_kv_read_bytes": int(m["packed_total_bytes"]), "separator_read_bytes": 0,
        "normal_kv_preads": 1, "separator_preads": 0, "read_calls": 1,
        "total_actual_pread_bytes": int(io_summary.get("bytes", 0)),
        "validated_prefix_layout": "visionzip_image_only_packed", "n_raters": 0,
        "core_started_at_s": t0, "first_token_at_s": timing["first_token_at"],
        "model_finished_at_s": timing["finished_at"], "postprocess_finished_at_s": time.perf_counter(),
    })
    st["selector_ms"] = st["select_ms"] + st["query_ms"]
    return st

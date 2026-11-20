"""Serving of PackedPrefixStore images (contract v2, A3).

``PackedImageContext`` opens a packed store (metadata validation, integrity
envelope, system KV, dense GPU ``PrefixCache``, pinned host buffer) at
activation time.  ``PackedPrefixSelector.prepare`` performs the request-time
work: ONE ``os.preadv`` of ``kv_packed.bin`` into the pinned buffer, then per
layer the v1 scatter (``PrefixCache.write`` with the stored-row index), the
separator scatter, and the v1 ``bias_from_keep`` mask with rows ``[0, k_l)``
visible.  ``directcopy=True`` (DEV diagnostic only) replaces the index scatter
by one contiguous slice copy per layer/kind.  Everything after ``prepare`` is
the v1 ``Server._decode`` path.

Original-order stores (``meta["row_order"] == "original"``, see
``mmimpress.packed_store``) are served by the same classes: the reads, the
pinned buffer and the per-layer hooks are identical; a layer's rows go to
their ORIGINAL token positions of the GPU cache, separators to their original
positions, and every position that was not stored is masked.

``PackedReadTrace`` records every ``os.pread`` AND ``os.preadv`` call so the
harness can prove "one read syscall per hit" independently of the IOCounter.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from mmimpress.packed_store import (ROW_ORDER_IMPORTANCE, ROW_ORDER_ORIGINAL, PackedReader,
                                    layer_original_indices, validate_packed_meta)
from mmimpress.serve import BIAS, IOCounter, PrefixCache, bias_from_keep, suffix_ids_from_prompt
from mmimpress.store import load_meta


class PackedImageContext:
    def __init__(self, store_dir, device, pin: bool = True):
        from mmimpress.prefix25 import verify_integrity
        self.dir = Path(store_dir)
        self.meta = load_meta(self.dir)
        validate_packed_meta(self.meta)
        if not (self.dir / "integrity.json").exists():
            raise ValueError("packed store lacks its integrity completion envelope")
        self.integrity_activation = verify_integrity(self.dir)
        m = self.meta
        self.row_order = m.get("row_order", ROW_ORDER_IMPORTANCE)
        self.layer_rows = None
        if self.row_order == ROW_ORDER_ORIGINAL:
            # separators and stored rows sit at their original token positions
            row = sorted(int(x) for x in m["newline_idx"])
            self.layer_rows = [torch.tensor(x, dtype=torch.long) for x in layer_original_indices(m)]
        else:
            row = sorted(int(x) for x in m["newline_stored"])
        self._separator_positions = [list(row) for _ in range(int(m["num_layers"]))]
        self.n_sep = len(row)
        self.reader = PackedReader(self.dir, m, pin=pin)
        sys_kv = torch.load(self.dir / "sys_kv.pt", weights_only=True)
        self.cache = PrefixCache(m, sys_kv, device)
        self.v_hidden = None
        self._layout_validated = False

    def separator_positions(self, layer):
        return self._separator_positions[layer]

    def validate_packed_layout(self):
        """Activation-time checks mirroring ``ImageContext.validate_visual_kv_layout``
        for the packed file (permutation artifact, separator tail, byte table)."""
        if self._layout_validated:
            return
        if self.row_order == ROW_ORDER_ORIGINAL:
            return self._validate_original_layout()
        from mmimpress.cvpr25 import permutation_sha256, visionzip_repack_order
        m = self.meta
        vn = int(m["v_token_num"]); L = int(m["num_layers"]); cs = int(m["chunk_size"])
        original_sep = [int(x) for x in m["newline_idx"]]
        assert original_sep == sorted(set(original_sep)) and all(0 <= p < vn for p in original_sep)
        n_content = vn - len(original_sep)
        assert n_content > 0 and int(m.get("n_spatial", -1)) == n_content
        assert cs == 64 and int(m["n_chunks_per_layer"]) == (vn + cs - 1) // cs
        assert int(m["prefix_len"]) == int(m["v_token_start"]) + vn
        assert m.get("separator_policy") == "stable_tail_plus_sidecar" and m.get("physical_layout") == "visionzip_image_only"
        order = [int(x) for x in m["order"]]
        assert len(order) == vn and order[n_content:] == original_sep
        assert m.get("permutation_sha256") == permutation_sha256(order)
        assert all(self.separator_positions(li) == list(range(n_content, vn)) for li in range(L))
        artifact = torch.load(self.dir / "visionzip_layout.pt", map_location="cpu", weights_only=True)
        scores = torch.as_tensor(artifact["token_score_original"]).flatten()
        assert scores.numel() == vn
        assert [int(x) for x in artifact["newline_original"]] == original_sep
        assert [int(x) for x in artifact["stored_to_original"]] == order
        inverse = [0] * vn
        for stored, original in enumerate(order):
            inverse[original] = stored
        assert [int(x) for x in artifact["original_to_stored"]] == inverse
        assert m.get("inverse_permutation_sha256") == permutation_sha256(inverse)
        assert visionzip_repack_order(scores, original_sep) == order
        assert artifact.get("layout_uses_dataset_question") is False
        assert artifact.get("llm_used_for_layout_scoring") is False
        assert int(artifact.get("calibration_questions", -1)) == 0
        assert (self.dir / m["packed_file"]).stat().st_size == int(m["packed_total_bytes"])
        self._layout_validated = True

    def _validate_original_layout(self):
        """Activation-time checks of an original-order store: no importance
        artifact, per-layer rows ascending / nested / content-only, separators
        at their original positions, byte table."""
        m = self.meta
        vn = int(m["v_token_num"]); L = int(m["num_layers"])
        original_sep = [int(x) for x in m["newline_idx"]]
        assert original_sep == sorted(set(original_sep)) and all(0 <= p < vn for p in original_sep)
        n_content = vn - len(original_sep)
        assert n_content > 0 and int(m.get("n_spatial", -1)) == n_content
        assert int(m["prefix_len"]) == int(m["v_token_start"]) + vn
        assert m.get("separator_policy") == "original_position_plus_sidecar" and m.get("physical_layout") == "original_order_layer_subsets"
        assert not (self.dir / "visionzip_layout.pt").exists(), "original-order store must not carry the importance artifact"
        assert "order" not in m and "full_importance_permutation" not in m and "permutation_sha256" not in m
        assert all(self.separator_positions(li) == original_sep for li in range(L))
        counts = [int(c) for c in m["layer_keep_counts"]]
        sep = set(original_sep); by_count = {}
        for li, rows in enumerate(self.layer_rows):
            rows = rows.tolist()
            assert len(rows) == counts[li] and rows == sorted(set(rows)) and not sep & set(rows)
            assert all(0 <= r < vn for r in rows)
            assert by_count.setdefault(counts[li], rows) == rows       # equal counts -> equal sets
        nested = [set(by_count[c]) for c in sorted(by_count)]
        assert all(a <= b for a, b in zip(nested, nested[1:]))
        assert m.get("layout_uses_dataset_question") is False and m.get("llm_used_for_layout_scoring") is False
        assert int(m.get("calibration_questions", -1)) == 0
        assert (self.dir / m["packed_file"]).stat().st_size == int(m["packed_total_bytes"])
        self._layout_validated = True

    def close(self):
        self.reader.close()
        self.cache = None


class PackedReadTrace:
    """Independent trace of real ``os.pread``/``os.preadv`` returns, with an
    optional store-directory guard for ``.bin`` payloads."""

    def __init__(self, allowed=None):
        self.allowed = None if allowed is None else Path(allowed).resolve()
        self.calls = []

    def _path(self, fd):
        return Path(os.readlink(f"/proc/self/fd/{fd}")).resolve()

    def _guard(self, path):
        if self.allowed is not None and path.suffix == ".bin" and not path.is_relative_to(self.allowed):
            raise AssertionError(f"foreign KV payload access: {path}")

    def __enter__(self):
        self._pread, self._preadv = os.pread, os.preadv
        trace = self

        def pread(fd, count, offset):
            path = trace._path(fd); trace._guard(path)
            blob = trace._pread(fd, count, offset)
            trace.calls.append({"syscall": "pread", "path": str(path), "offset": offset, "requested": count, "returned": len(blob)})
            return blob

        def preadv(fd, buffers, offset, *flags):
            path = trace._path(fd); trace._guard(path)
            requested = sum(memoryview(b).nbytes for b in buffers)
            n = trace._preadv(fd, buffers, offset, *flags)
            trace.calls.append({"syscall": "preadv", "path": str(path), "offset": offset, "requested": requested, "returned": n})
            return n

        os.pread, os.preadv = pread, preadv
        return self

    def __exit__(self, *exc):
        os.pread, os.preadv = self._pread, self._preadv

    def summary(self):
        return {"bytes": sum(c["returned"] for c in self.calls), "preads": len(self.calls),
                "pread_calls": sum(c["syscall"] == "pread" for c in self.calls),
                "preadv_calls": sum(c["syscall"] == "preadv" for c in self.calls), "calls": self.calls}


class PackedPrefixSelector:
    """Query-independent prefix selection served from a packed store."""

    def __init__(self, runner, ctx: PackedImageContext, budget=0.25, counter=None, directcopy=False):
        from mmimpress.layer_budget import layer_keep_counts
        self.runner, self.ctx = runner, ctx
        self.io = counter if counter is not None else IOCounter()
        self.directcopy = bool(directcopy)
        self.t = {"select": 0.0, "chunk_io": 0.0, "scatter": 0.0, "query": 0.0}
        m = ctx.meta
        if float(budget) != 0.25:
            raise ValueError("packed stores serve only the frozen 0.25 visual_kv budget")
        vn = int(m["v_token_num"]); n_content = int(m["n_spatial"])
        if n_content <= 0 or n_content != vn - len(ctx.separator_positions(0)):
            raise ValueError("visual-content count disagrees with separators")
        self.original_rows = ctx.row_order == ROW_ORDER_ORIGINAL
        if self.original_rows and self.directcopy:
            raise ValueError("directcopy needs the importance row order")
        counts = layer_keep_counts(m["layer_keep_ratios"], n_content, int(m["num_layers"]), 0.25,
                                   float(m["layer_budget_r_min"]), float(m["layer_budget_r_max"]))
        if counts != [int(x) for x in m["layer_keep_counts"]]:
            raise ValueError("layerwise keep counts disagree with store")
        k_u = (n_content + 3) // 4
        if sum(counts) != int(m["num_layers"]) * k_u:
            raise ValueError("layerwise budget differs from uniform total")
        self.n_content, self.n_structural = n_content, vn - n_content
        self.k_per_layer = counts; self.k_max = max(counts); self.k_target = k_u
        if self.original_rows:
            # no global stored-row id: layer l's rows are ctx.layer_rows[l] (ascending original indices)
            self.selected_stored_ids = None
            self.selected_original_ids = [int(x) for x in m["stored_original_indices"]]
        else:
            self.selected_stored_ids = list(range(self.k_max))
            self.selected_original_ids = [int(x) for x in m["order"][:self.k_max]]
        self.kept_tokens_per_layer = []
        self.read_bytes = 0; self.read_calls = 0; self.read_ms = 0.0

    @torch.no_grad()
    def prepare(self):
        ctx, m = self.ctx, self.ctx.meta
        dev = self.runner.model.device
        L, vn = int(m["num_layers"]), int(m["v_token_num"])
        vstart = int(m["v_token_start"])
        t0 = time.perf_counter()
        n, dt = ctx.reader.read_all(self.io)
        self.t["chunk_io"] += time.perf_counter() - t0
        self.read_bytes, self.read_calls, self.read_ms = n, 1, dt * 1e3
        sep = ctx.reader.sep_view()
        for li in range(L):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            self._scatter_layer(li, sep, dev)
            torch.cuda.synchronize()
            self.t["scatter"] += time.perf_counter() - t0

    def _scatter_layer(self, li, sep, dev):
        """Layer ``li``: stored rows + separator rows -> GPU cache, and mask.

        Shared verbatim by the synchronous ``prepare`` loop and the
        asynchronous per-layer hook, so both issue the same GPU writes.
        """
        ctx, m = self.ctx, self.ctx.meta
        vn, vstart = int(m["v_token_num"]), int(m["v_token_start"])
        k = self.k_per_layer[li]
        rows = ctx.layer_rows[li] if self.original_rows else None      # original token positions of the stored rows
        for kind in ("k", "v"):
            vals = ctx.reader.layer_view(li, kind)
            assert vals.shape[0] == k
            if k:
                if self.original_rows:
                    ctx.cache.write(li, kind, rows, vals)
                elif self.directcopy:
                    buf = (ctx.cache.k if kind == "k" else ctx.cache.v)[li]
                    buf[0, :, vstart:vstart + k, :] = vals.to(dev, ctx.cache.dtype).permute(1, 0, 2)
                else:
                    ctx.cache.write(li, kind, torch.arange(k, dtype=torch.long), vals)
        keep = torch.zeros(vn, dtype=torch.bool)
        if self.original_rows:
            keep[rows] = True
        else:
            keep[:k] = True
        sp = torch.tensor(ctx.separator_positions(li), dtype=torch.long)
        ctx.cache.write(li, "k", sp, sep[0][li])
        ctx.cache.write(li, "v", sp, sep[1][li])
        if self.original_rows:
            assert int(keep.sum()) == k
        keep[sp] = True
        if self.original_rows:
            assert int(keep.sum()) == k + self.n_structural      # separators are disjoint from the stored rows
        else:
            assert int(keep[:self.n_content].sum()) == k and bool(keep[self.n_content:].all())
        self.kept_tokens_per_layer.append(int(keep.sum()))
        BIAS[li] = bias_from_keep(keep, m, dev)

    def stats(self):
        m = self.ctx.meta
        L = int(m["num_layers"])
        row_bytes = int(m["packed_row_bytes"])
        per_layer_bytes = [2 * k * row_bytes for k in self.k_per_layer]
        assert self.kept_tokens_per_layer == [k + self.n_structural for k in self.k_per_layer]
        if self.original_rows:
            rule = ("layer l attends the original token indices {selected_original_ids[j] : stored_min_keep_count[j] <= k_per_layer[l]}, "
                    "each at its original cache position")
        else:
            rule = "layer l attends stored rows [0, k_l) = selected_original_ids[:k_per_layer[l]]"
        return {
            "row_order": ROW_ORDER_ORIGINAL if self.original_rows else ROW_ORDER_IMPORTANCE,
            "store_format": m["store_format"], "selection_mode": "prefix", "budget_unit": "visual_kv", "ratio": 0.25,
            "budget_policy": "layerwise_v1", "layer_budget_id": str(m["layer_budget_id"]), "layer_budget_sha256": str(m["layer_budget_sha256"]),
            "layer_keep_ratios": [float(x) for x in m["layer_keep_ratios"]],
            "N_content": self.n_content, "N_structural": self.n_structural, "k_target": None, "k_uniform_equivalent": self.k_target,
            "k_total": int(sum(self.k_per_layer)), "k_per_layer": list(self.k_per_layer), "k_max": int(self.k_max),
            "attended_content_kv_count": None, "attended_content_kv_count_per_layer": list(self.k_per_layer),
            "selected_stored_ids": self.selected_stored_ids, "selected_original_ids": self.selected_original_ids,
            "selection_rule": rule,
            "keep_count_per_layer": self.kept_tokens_per_layer,
            "planned_normal_read_bytes_per_layer": per_layer_bytes,
            "planned_packed_read": {"offset_bytes": 0, "length_bytes": int(m["packed_total_bytes"])},
            "logical_content_retention_ratio": sum(self.k_per_layer) / (L * self.n_content),
            "logical_content_retention_ratio_per_layer": [k / self.n_content for k in self.k_per_layer],
            "structural_included_visual_retention_ratio": (sum(self.k_per_layer) / L + self.n_structural) / int(m["v_token_num"]),
            "logical_required_content_bytes": int(sum(per_layer_bytes)),
            "unused_loaded_real_rows": 0, "unused_loaded_real_bytes": 0, "actual_loaded_structural_rows": 0, "structural_bytes_in_normal_payload": 0,
            "normal_chunks_read": None, "normal_chunks_read_per_layer": [int(np.ceil(k / 64)) for k in self.k_per_layer],
            "static_score_calls": 0, "query_score_calls": 0, "diversity_calls": 0, "separator_policy": "sidecar",
            "select_ms": self.t["select"] * 1e3, "query_ms": self.t["query"] * 1e3, "chunk_io_ms": self.t["chunk_io"] * 1e3,
            "scatter_ms": self.t["scatter"] * 1e3, "hook_ms": 0.0, "packed_read_ms": self.read_ms, "read_calls": self.read_calls,
            "read_syscall": "preadv", "directcopy": self.directcopy, "fallback_rate": 0.0, "mean_jaccard": None,
            "logical_kv_ratio": float(np.mean(self.kept_tokens_per_layer)) / int(m["v_token_num"]),
        }


class PackedStreamSelector(PackedPrefixSelector):
    """Asynchronous variant: the packed file is read in layer order by one
    background worker while the model computes.

    Same store, same bytes, same per-layer GPU writes and masks as
    ``PackedPrefixSelector``.  The differences are who reads and when:

    * ``begin()`` (called after the TTFT timer started) hands the byte ranges
      of ``PackedReader.stream_plan()`` -- separator block, then one ``[K|V]``
      range per layer -- to a ``LayerReadAhead`` worker, which ``preadv``s
      each range into its own place in the pinned buffer;
    * a forward pre-hook on decoder layer i waits until layer i's range (and,
      for layer 0, the separator block) has arrived, runs the shared
      ``_scatter_layer`` and lets the layer compute.

    The pinned buffer is still allocated at activation, outside TTFT.
    """

    def __init__(self, runner, ctx: PackedImageContext, budget=0.25, counter=None, executor=None, max_ahead=None):
        super().__init__(runner, ctx, budget, counter, directcopy=False)
        self._executor = executor
        self._owns_executor = False
        self.max_ahead = max_ahead
        self._plan = {key: (off, ln) for key, off, ln in ctx.reader.stream_plan()}
        self._order = [key for key, _, _ in ctx.reader.stream_plan()]
        self._stream = None
        self._sep = None
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
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="packed-readahead")
            self._owns_executor = True
        self._stream = LayerReadAhead(self._executor, self._read_item, self._order, max_ahead=self.max_ahead).start()

    def close(self):
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
            if self._sep is None:
                self._take("sep")
                self._sep = self.ctx.reader.sep_view()
            self._take(li)
            t1 = time.perf_counter()
            self._scatter_layer(li, self._sep, self.runner.model.device)
            self.done.add(li)
            now = time.perf_counter()
            self.t["scatter"] += now - t1
            self.hook_seconds += now - t0
        return hook

    def prepare(self):
        raise RuntimeError("PackedStreamSelector loads inside layer hooks; use it as a context manager")

    def stats(self):
        st = super().stats()
        m = self.ctx.meta
        ranges = [{"key": str(key), "offset_bytes": int(self._plan[key][0]), "length_bytes": int(self._plan[key][1])} for key in self._order]
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
def request_packed(server, ctx: PackedImageContext, *, suffix_ids=None, prompt_text=None, question=None,
                   cold=True, image_id="", directcopy=False, readahead="none", max_ahead=None):
    """Hook-free packed cache hit with the v1 request timing boundaries.

    ``readahead="layer_stream"`` serves the same store through
    ``PackedStreamSelector`` (background layer-ordered read, per-layer hooks);
    the timer boundaries are unchanged.
    """
    from mmimpress.serve import suffix_ids_for
    if readahead not in ("none", "layer_stream"):
        raise ValueError(f"readahead must be 'none' or 'layer_stream', got {readahead!r}")
    if readahead == "layer_stream":
        if directcopy:
            raise ValueError("directcopy is a synchronous-path diagnostic")
        return _request_packed_stream(server, ctx, suffix_ids=suffix_ids, prompt_text=prompt_text, question=question,
                                      cold=cold, max_ahead=max_ahead)
    ctx.validate_packed_layout()
    BIAS.clear()
    if cold:
        ctx.reader.drop_all()
    counter = IOCounter()
    dev = server.runner.model.device
    if suffix_ids is None:
        suffix_ids = (suffix_ids_from_prompt(server.runner, prompt_text) if prompt_text is not None
                      else suffix_ids_for(server.runner, question))
    suffix_ids = suffix_ids.to(dev)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    cache = ctx.cache.new_request()
    sel = PackedPrefixSelector(server.runner, ctx, 0.25, counter, directcopy=directcopy)
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
        "prefill_ms": timing["prefill_ms"], "decode_ms": timing["decode_ms"], "model_ms": timing["prefill_ms"] + timing["decode_ms"],
        "generated_tokens": timing["generated_tokens"], "generated_token_ids": timing["generated_token_ids"],
        "generated_token_count": timing["generated_token_count"], "io": io_summary,
        "normal_kv_read_bytes": int(m["packed_total_bytes"]) - int(m["packed_sep_bytes"]),
        "separator_read_bytes": int(m["packed_sep_bytes"]),
        "normal_kv_preads": 1, "separator_preads": 0, "read_calls": 1,
        "total_actual_pread_bytes": int(io_summary.get("bytes", 0)),
        "validated_prefix_layout": ("original_order_packed" if ctx.row_order == ROW_ORDER_ORIGINAL
                                    else "visionzip_image_only_packed"), "n_raters": 0,
        "core_started_at_s": t0, "first_token_at_s": timing["first_token_at"], "model_finished_at_s": timing["finished_at"],
        "postprocess_finished_at_s": time.perf_counter(),
    })
    st["selector_ms"] = st["select_ms"] + st["query_ms"]
    return st


@torch.no_grad()
def _request_packed_stream(server, ctx: PackedImageContext, *, suffix_ids=None, prompt_text=None, question=None,
                           cold=True, max_ahead=None):
    """Packed cache hit with the layer-ordered background read-ahead."""
    from mmimpress.serve import suffix_ids_for
    ctx.validate_packed_layout()
    BIAS.clear()
    if cold:
        ctx.reader.drop_all()
    counter = IOCounter()
    dev = server.runner.model.device
    if suffix_ids is None:
        suffix_ids = (suffix_ids_from_prompt(server.runner, prompt_text) if prompt_text is not None
                      else suffix_ids_for(server.runner, question))
    suffix_ids = suffix_ids.to(dev)
    executor = server._prefetch_pool()            # worker thread exists before the timer
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    sel = PackedStreamSelector(server.runner, ctx, 0.25, counter, executor=executor, max_ahead=max_ahead)
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
    layer_reads = sum(1 for key, (_, ln) in sel._plan.items() if key != "sep" and ln)
    st.update({
        "answer": answer, "first_token_id": first_token_id, "ttft": ttft,
        "decode_latency": timing["decode_ms"] / 1e3, "e2e_latency": e2e, "prepare_ms": None,
        "prefill_ms": timing["prefill_ms"], "decode_ms": timing["decode_ms"], "model_ms": timing["prefill_ms"] + timing["decode_ms"],
        "generated_tokens": timing["generated_tokens"], "generated_token_ids": timing["generated_token_ids"],
        "generated_token_count": timing["generated_token_count"], "io": io_summary,
        "normal_kv_read_bytes": int(m["packed_total_bytes"]) - int(m["packed_sep_bytes"]),
        "separator_read_bytes": int(m["packed_sep_bytes"]),
        "normal_kv_preads": layer_reads, "separator_preads": 1 if int(m["packed_sep_bytes"]) else 0,
        "read_calls": int(io_summary.get("preads", 0)),
        "total_actual_pread_bytes": int(io_summary.get("bytes", 0)),
        "validated_prefix_layout": ("original_order_packed" if ctx.row_order == ROW_ORDER_ORIGINAL
                                    else "visionzip_image_only_packed"), "n_raters": 0,
        "core_started_at_s": t0, "first_token_at_s": timing["first_token_at"], "model_finished_at_s": timing["finished_at"],
        "postprocess_finished_at_s": time.perf_counter(),
    })
    st["selector_ms"] = st["select_ms"] + st["query_ms"]
    return st

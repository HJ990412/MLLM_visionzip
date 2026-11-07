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

``PackedReadTrace`` records every ``os.pread`` AND ``os.preadv`` call so the
harness can prove "one read syscall per hit" independently of the IOCounter.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from mmimpress.packed_store import PackedReader, validate_packed_meta
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
        order = m["order"]
        counts = layer_keep_counts(m["layer_keep_ratios"], n_content, int(m["num_layers"]), 0.25,
                                   float(m["layer_budget_r_min"]), float(m["layer_budget_r_max"]))
        if counts != [int(x) for x in m["layer_keep_counts"]]:
            raise ValueError("layerwise keep counts disagree with store")
        k_u = (n_content + 3) // 4
        if sum(counts) != int(m["num_layers"]) * k_u:
            raise ValueError("layerwise budget differs from uniform total")
        self.n_content, self.n_structural = n_content, vn - n_content
        self.k_per_layer = counts; self.k_max = max(counts); self.k_target = k_u
        self.selected_stored_ids = list(range(self.k_max))
        self.selected_original_ids = [int(x) for x in order[:self.k_max]]
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
            k = self.k_per_layer[li]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for kind in ("k", "v"):
                vals = ctx.reader.layer_view(li, kind)
                assert vals.shape[0] == k
                if k:
                    if self.directcopy:
                        buf = (ctx.cache.k if kind == "k" else ctx.cache.v)[li]
                        buf[0, :, vstart:vstart + k, :] = vals.to(dev, ctx.cache.dtype).permute(1, 0, 2)
                    else:
                        ctx.cache.write(li, kind, torch.arange(k, dtype=torch.long), vals)
            keep = torch.zeros(vn, dtype=torch.bool)
            keep[:k] = True
            sp = torch.tensor(ctx.separator_positions(li), dtype=torch.long)
            ctx.cache.write(li, "k", sp, sep[0][li])
            ctx.cache.write(li, "v", sp, sep[1][li])
            keep[sp] = True
            assert int(keep[:self.n_content].sum()) == k and bool(keep[self.n_content:].all())
            self.kept_tokens_per_layer.append(int(keep.sum()))
            BIAS[li] = bias_from_keep(keep, m, dev)
            torch.cuda.synchronize()
            self.t["scatter"] += time.perf_counter() - t0

    def stats(self):
        m = self.ctx.meta
        L = int(m["num_layers"])
        row_bytes = int(m["packed_row_bytes"])
        per_layer_bytes = [2 * k * row_bytes for k in self.k_per_layer]
        assert self.kept_tokens_per_layer == [k + self.n_structural for k in self.k_per_layer]
        return {
            "store_format": m["store_format"], "selection_mode": "prefix", "budget_unit": "visual_kv", "ratio": 0.25,
            "budget_policy": "layerwise_v1", "layer_budget_id": str(m["layer_budget_id"]), "layer_budget_sha256": str(m["layer_budget_sha256"]),
            "layer_keep_ratios": [float(x) for x in m["layer_keep_ratios"]],
            "N_content": self.n_content, "N_structural": self.n_structural, "k_target": None, "k_uniform_equivalent": self.k_target,
            "k_total": int(sum(self.k_per_layer)), "k_per_layer": list(self.k_per_layer), "k_max": int(self.k_max),
            "attended_content_kv_count": None, "attended_content_kv_count_per_layer": list(self.k_per_layer),
            "selected_stored_ids": self.selected_stored_ids, "selected_original_ids": self.selected_original_ids,
            "selection_rule": "layer l attends stored rows [0, k_l) = selected_original_ids[:k_per_layer[l]]",
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


@torch.no_grad()
def request_packed(server, ctx: PackedImageContext, *, suffix_ids=None, prompt_text=None, question=None,
                   cold=True, image_id="", directcopy=False):
    """Hook-free packed cache hit with the v1 request timing boundaries."""
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
        "validated_prefix_layout": "visionzip_image_only_packed", "n_raters": 0,
        "core_started_at_s": t0, "first_token_at_s": timing["first_token_at"], "model_finished_at_s": timing["finished_at"],
        "postprocess_finished_at_s": time.perf_counter(),
    })
    st["selector_ms"] = st["select_ms"] + st["query_ms"]
    return st

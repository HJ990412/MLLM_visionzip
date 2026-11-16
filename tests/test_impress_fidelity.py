"""CPU-only contracts for the paper-faithful IMPRESS options.

Covers the two switches added to the IMPRESS port (``LayerSelector`` with
``mode="impress"``):

  importance_metric  "sparsevlm" (default) | "h2o" (IMPRESS 4.1 column sum)
  fallback_v         "full" (default)      | "selected" (IMPRESS 4.3)

Everything runs on tiny synthetic tensors with a fake reader/cache in the
style of ``tests/test_qa_select.py``: no model, no CUDA, no real store.

The default-equivalence test compares the current ``LayerSelector`` against
``_LegacyLayerSelector``, a frozen verbatim copy of ``_run``/``stats`` as they
were before the options existed.  The workspace is not a Git worktree, so the
reference has to live here.
"""
from __future__ import annotations

import copy
import importlib.util
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch import nn

import transformers.models.llama.modeling_llama as ml

from mmimpress import serve
from mmimpress import sparsevlm as sv
from mmimpress.serve import (
    BIAS,
    Calibrator,
    LayerSelector,
    Server,
    bias_from_keep,
    calibrate_image,
    mean_pairwise_jaccard,
    similarity_threshold,
)
from mmimpress.store import IOCounter, chunks_for_tokens

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name, alias):
    spec = importlib.util.spec_from_file_location(alias, ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ===================================================================== fakes
META = {
    "num_layers": 4,
    "num_heads": 4,
    "head_dim": 8,
    "v_token_start": 2,
    "v_token_num": 24,
    "prefix_len": 26,
    "n_spatial": 22,
    "chunk_size": 4,
    "n_chunks_per_layer": 6,
    "probe_heads": 3,
    "dtype": "float16",
    "newline_idx": [5, 23],
}
N_SUFFIX = 5
ROW_BYTES = META["num_heads"] * META["head_dim"] * 2          # fp16


class _FakeReader:
    """Holds real (random) K/V so loaded contents can be checked.

    Accounting mirrors ``ChunkReader``: bytes are what a pread would return,
    one pread per contiguous run of chunks, units = number of chunk spans.
    Every call is recorded in order.
    """

    def __init__(self, meta, k, v):
        self.meta = meta
        self.k = k                      # [layer] -> (v_num, H, hd) fp16
        self.v = v
        self.calls = []

    def _record(self, counter, kind, tensor, preads, units):
        if counter is not None:
            counter.record(kind, tensor.numel() * tensor.element_size(),
                           0.001, preads=preads, units=units)

    def read_probe(self, layer, counter=None):
        self.calls.append(("probe", layer))
        value = self.k[layer][:, :self.meta["probe_heads"]].clone()
        self._record(counter, "probe", value, 1,
                     self.meta["n_chunks_per_layer"])
        return value

    def read_full(self, layer, kind, counter=None):
        self.calls.append(("full", layer, kind))
        value = (self.k if kind == "k" else self.v)[layer].clone()
        self._record(counter, kind, value, 1,
                     self.meta["n_chunks_per_layer"])
        return value

    def read_chunks(self, layer, kind, cids, counter=None):
        m = self.meta
        cids = sorted(set(int(c) for c in cids))
        self.calls.append(("chunks", layer, kind, tuple(cids)))
        rows = []
        for c in cids:
            rows.extend(range(c * m["chunk_size"],
                              min((c + 1) * m["chunk_size"],
                                  m["v_token_num"])))
        rows = torch.tensor(rows, dtype=torch.long)
        value = (self.k if kind == "k" else self.v)[layer][rows].clone()
        runs, _ = serve.contiguous_runs(cids)
        self._record(counter, kind, value, runs, len(cids))
        return rows, value


class _FakeCache:
    """PrefixCache stand-in: zero visual rows, real system rows."""

    def __init__(self, meta, sys_k):
        shape = (1, meta["num_heads"], meta["prefix_len"], meta["head_dim"])
        self.v_start = meta["v_token_start"]
        self.k = [torch.zeros(shape) for _ in range(meta["num_layers"])]
        self.v = [torch.zeros(shape) for _ in range(meta["num_layers"])]
        for li in range(meta["num_layers"]):
            self.k[li][0, :, :self.v_start] = sys_k[li]

    def write(self, layer, kind, rows, vals):
        buf = (self.k if kind == "k" else self.v)[layer]
        buf[0, :, rows + self.v_start, :] = vals.to(buf.dtype).permute(1, 0, 2)

    def write_full(self, layer, kind, block):
        buf = (self.k if kind == "k" else self.v)[layer]
        buf[0, :, self.v_start:, :] = block.to(buf.dtype).permute(1, 0, 2)


class _FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Identity()
        self.k_proj = nn.Identity()


class _FakeDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.Identity()
        self.self_attn = _FakeAttention()

    def forward(self, hidden_states, position_embeddings=None):
        return hidden_states


class _World:
    """One synthetic image prefix plus one request's suffix hidden states.

    Layers 0 and 2 are built so the three probe heads agree (identical probe
    keys and queries -> Jaccard 1 -> consensus path); layers 1 and 3 are
    independent noise (Jaccard near chance -> fallback).  Both paths are
    therefore exercised by every test that walks the layers.
    """

    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        m = self.meta = copy.deepcopy(META)
        L, H, hd = m["num_layers"], m["num_heads"], m["head_dim"]
        vn, v0, P = m["v_token_num"], m["v_token_start"], m["probe_heads"]

        def fp16(*shape, scale=1.0):
            return (torch.randn(*shape, generator=g) * scale).to(
                torch.float16)

        self.k = [fp16(vn, H, hd, scale=2.0) for _ in range(L)]
        self.v = [fp16(vn, H, hd) + 3.0 for _ in range(L)]   # never ~0
        self.sys_k = [fp16(H, v0, hd).float() for _ in range(L)]
        # suffix hidden: (1, n, H * hd); identity projections make it both
        # the query and the suffix key of every head
        self.hidden = [torch.randn(1, N_SUFFIX, H * hd, generator=g) * 2.0
                       for _ in range(L)]
        for li in (0, 2):
            self.k[li][:, :P] = self.k[li][:, :1]
            head0 = self.hidden[li].view(1, N_SUFFIX, H, hd)[:, :, :1]
            self.hidden[li].view(1, N_SUFFIX, H, hd)[:, :, :P] = head0
            self.sys_k[li][:P] = self.sys_k[li][:1]
        self.layers = [_FakeDecoderLayer() for _ in range(L)]
        self.runner = SimpleNamespace(
            layers=self.layers,
            model=SimpleNamespace(device=torch.device("cpu")))

    def context(self):
        reader = _FakeReader(self.meta, self.k, self.v)
        cache = _FakeCache(self.meta, self.sys_k)
        return SimpleNamespace(meta=self.meta, reader=reader, cache=cache,
                               v_hidden=None)

    def run(self, selector_cls=LayerSelector, rater_rows=(0, 2), **kwargs):
        """Drive the selector's hooks once over every layer."""
        BIAS.clear()
        ctx = self.context()
        counter = IOCounter()
        selector = selector_cls(
            self.runner, ctx, torch.tensor(list(rater_rows)), ratio=0.25,
            probe=3, alpha=0.6, mode=kwargs.pop("mode", "impress"),
            counter=counter, **kwargs)
        with mock.patch(
                "mmimpress.serve.ml.apply_rotary_pos_emb",
                side_effect=lambda query, key, cos, sin: (query, key)):
            with selector:
                for li, layer in enumerate(self.layers):
                    layer(hidden_states=self.hidden[li],
                          position_embeddings=(None, None))
        bias = {li: BIAS[li].clone() for li in sorted(BIAS)}
        BIAS.clear()
        return SimpleNamespace(selector=selector, ctx=ctx, counter=counter,
                               bias=bias)


def _keep_from_bias(bias, meta):
    """(heads, v_num) bool: which visual rows a layer's bias leaves visible."""
    v0, vn = meta["v_token_start"], meta["v_token_num"]
    keep = bias[0, :, 0, v0:v0 + vn] == 0
    if keep.shape[0] == 1:
        keep = keep.expand(meta["num_heads"], vn)
    return keep


def _io(counter):
    """IOCounter contents without wall-clock noise."""
    summary = counter.summary()
    return {
        "preads": summary["preads"], "bytes": summary["bytes"],
        "chunk_units": summary["chunk_units"],
        "per_kind": {kind: {key: value for key, value in detail.items()
                            if key != "seconds"}
                     for kind, detail in summary["per_kind"].items()},
    }


# ============================================ frozen pre-change reference
class _LegacyLayerSelector(LayerSelector):
    """``LayerSelector._run`` / ``.stats`` exactly as before the options.

    Verbatim copy of the previous implementation.  Do not "fix" or modernise
    it: its only job is to pin the historical behaviour.
    """

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

        pk = ctx.reader.read_probe(li, self.io).to(dev, q.dtype) \
            .permute(1, 0, 2)[:P]                            # (P, v_num, hd)
        keys_p = torch.cat([sys_k[:P], pk, kn[:P]], dim=1)
        sp = sv.rater_visual_scores_from_qk(
            q[:P], keys_p, self.rater_rows, v0, vn, causal_from=v0 + vn)
        sp[:, self.nl(li).to(dev)] = float("-inf")
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


LEGACY_LOG_KEYS = ("layer", "mode", "sim", "chunks")
LEGACY_STAT_KEYS = ("touched_chunk_fraction", "logical_kv_ratio",
                    "fallback_rate", "probe_rate", "mean_jaccard",
                    "threshold", "k_keep")          # hook_ms is wall-clock


# ====================================================== explicit attention
def _explicit_attention(q, sys_k, vis_k, suf_k):
    """Suffix rows of the causal attention matrix, built the slow way.

    Independent of the production code path: float64, an explicit loop over
    heads and rows, and an explicit list of the keys each row may see.
    q, suf_k: (H, n, hd); sys_k: (H, v0, hd); vis_k: (H, vn, hd).
    Returns (H, n, v0 + vn + n) post-softmax probabilities.
    """
    H, n, hd = q.shape
    v0, vn = sys_k.shape[1], vis_k.shape[1]
    keys = torch.cat([sys_k, vis_k, suf_k], dim=1).double()
    attn = torch.zeros(H, n, v0 + vn + n, dtype=torch.float64)
    for h in range(H):
        for i in range(n):
            visible = v0 + vn + i + 1          # prefix + suffix keys 0..i
            logits = (keys[h, :visible] @ q[h, i].double()) / math.sqrt(hd)
            attn[h, i, :visible] = torch.softmax(logits, dim=0)
    return attn


class H2OScoreTests(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(7)
        self.H, self.n, self.hd, self.v0, self.vn = 3, 6, 8, 2, 11
        self.q = torch.randn(self.H, self.n, self.hd, generator=g)
        self.sys_k = torch.randn(self.H, self.v0, self.hd, generator=g)
        self.vis_k = torch.randn(self.H, self.vn, self.hd, generator=g)
        self.suf_k = torch.randn(self.H, self.n, self.hd, generator=g)
        self.keys = torch.cat([self.sys_k, self.vis_k, self.suf_k], dim=1)
        self.attn = _explicit_attention(
            self.q, self.sys_k, self.vis_k, self.suf_k)

    def test_h2o_equals_column_sum_of_explicit_attention_matrix(self):
        expected = self.attn[:, :, self.v0:self.v0 + self.vn].sum(dim=1)
        got = sv.h2o_visual_scores_from_qk(
            self.q, self.keys, self.v0, self.vn,
            causal_from=self.v0 + self.vn)
        self.assertEqual(tuple(got.shape), (self.H, self.vn))
        self.assertTrue(torch.allclose(got.double(), expected,
                                       rtol=1e-5, atol=1e-6))
        # every row of the explicit matrix is a probability distribution, so
        # the reference really is post-softmax over system+visual+suffix
        self.assertTrue(torch.allclose(
            self.attn.sum(dim=-1), torch.ones(self.H, self.n,
                                              dtype=torch.float64)))
        # matrix-form helper agrees with the Q/K form
        from_matrix = sv.h2o_visual_scores(self.attn, self.v0, self.vn)
        self.assertTrue(torch.allclose(from_matrix.double(), expected,
                                       rtol=1e-5, atol=1e-6))
        # head_reduce="mean" is the head average of the per-head score
        reduced = sv.h2o_visual_scores_from_qk(
            self.q, self.keys, self.v0, self.vn, head_reduce="mean",
            causal_from=self.v0 + self.vn)
        self.assertTrue(torch.allclose(reduced, got.mean(dim=0)))

    def test_causal_mask_matters_and_suffix_rows_are_all_used(self):
        causal = sv.h2o_visual_scores_from_qk(
            self.q, self.keys, self.v0, self.vn,
            causal_from=self.v0 + self.vn)
        unmasked = sv.h2o_visual_scores_from_qk(
            self.q, self.keys, self.v0, self.vn)
        self.assertFalse(torch.allclose(causal, unmasked))
        # dropping a query row changes the score: every suffix row counts
        fewer = sv.h2o_visual_scores_from_qk(
            self.q[:, :-1], self.keys[:, :-1], self.v0, self.vn,
            causal_from=self.v0 + self.vn)
        self.assertFalse(torch.allclose(causal, fewer))

    def test_h2o_differs_from_sparsevlm_only_in_the_row_set(self):
        h2o = sv.h2o_visual_scores_from_qk(
            self.q, self.keys, self.v0, self.vn,
            causal_from=self.v0 + self.vn)
        all_rows = sv.rater_visual_scores_from_qk(
            self.q, self.keys, list(range(self.n)), self.v0, self.vn,
            causal_from=self.v0 + self.vn)
        # sum over all rows == n * mean over all rows
        self.assertTrue(torch.allclose(h2o, all_rows * self.n,
                                       rtol=1e-5, atol=1e-6))
        some_rows = sv.rater_visual_scores_from_qk(
            self.q, self.keys, [1, 4], self.v0, self.vn,
            causal_from=self.v0 + self.vn)
        expected = self.attn[:, [1, 4], self.v0:self.v0 + self.vn].mean(1)
        self.assertTrue(torch.allclose(some_rows.double(), expected,
                                       rtol=1e-5, atol=1e-6))
        self.assertFalse(torch.allclose(h2o, some_rows * self.n))

    def test_suffix_key_row_mismatch_and_bad_metric_fail(self):
        with self.assertRaises(AssertionError):
            sv.h2o_visual_scores_from_qk(
                self.q[:, :3], self.keys, self.v0, self.vn,
                causal_from=self.v0 + self.vn)
        with self.assertRaises(ValueError):
            sv.validate_importance_metric("attention")
        self.assertEqual(sv.IMPORTANCE_METRICS, ("sparsevlm", "h2o"))
        self.assertEqual(sv.DEFAULT_IMPORTANCE_METRIC, "sparsevlm")


class ThresholdTests(unittest.TestCase):
    def test_similarity_threshold_matches_paper_formula(self):
        # ratio 0.25: j = 0.25 / (2 - 0.25) = 1/7, t = j ** 0.6
        self.assertAlmostEqual(similarity_threshold(576, 2304, 0.6),
                               0.3111, places=4)
        self.assertAlmostEqual(similarity_threshold(576, 2304, 0.6),
                               (1.0 / 7.0) ** 0.6, places=12)
        self.assertAlmostEqual(similarity_threshold(1, 4), 0.3111, places=4)
        # LLaVA-NeXT geometry used in the pilot: 2304 spatial tokens
        self.assertEqual(sv.topk_budget(2304, 0.25), 576)


# ============================================================ LayerSelector
class DefaultEquivalenceTests(unittest.TestCase):
    """Default options reproduce the pre-change selector exactly."""

    def tearDown(self):
        BIAS.clear()

    def _assert_same(self, new, old, meta):
        self.assertEqual(sorted(new.bias), sorted(old.bias))
        for li in new.bias:
            self.assertTrue(torch.equal(new.bias[li], old.bias[li]), li)
        for li in range(meta["num_layers"]):
            self.assertTrue(torch.equal(new.ctx.cache.k[li],
                                        old.ctx.cache.k[li]), li)
            self.assertTrue(torch.equal(new.ctx.cache.v[li],
                                        old.ctx.cache.v[li]), li)
        # identical reads, in the identical order, with identical accounting
        self.assertEqual(new.ctx.reader.calls, old.ctx.reader.calls)
        self.assertEqual(_io(new.counter), _io(old.counter))
        # every historical log field is unchanged; new fields are additive
        self.assertEqual(
            [{k: r[k] for k in LEGACY_LOG_KEYS} for r in new.selector.log],
            old.selector.log)
        for record in new.selector.log:
            self.assertTrue(set(LEGACY_LOG_KEYS) <= set(record))
        new_stats, old_stats = new.selector.stats(), old.selector.stats()
        for key in LEGACY_STAT_KEYS:
            self.assertEqual(new_stats[key], old_stats[key], key)
        self.assertEqual(set(old_stats) - set(new_stats), set())

    def test_default_options_match_frozen_legacy_implementation(self):
        for seed in range(6):
            with self.subTest(seed=seed):
                world = _World(seed)
                new = world.run(LayerSelector)
                old = world.run(_LegacyLayerSelector)
                self._assert_same(new, old, world.meta)
                modes = [r["mode"] for r in new.selector.log]
                # both branches are really exercised
                self.assertEqual(modes[0], "probe")
                self.assertEqual(modes[2], "probe")
                self.assertIn("fallback", modes)
                self.assertEqual(new.selector.importance_metric, "sparsevlm")
                self.assertEqual(new.selector.fallback_v, "full")

    def test_explicit_default_values_are_the_defaults(self):
        world = _World(1)
        implicit = world.run(LayerSelector)
        explicit = world.run(LayerSelector, importance_metric="sparsevlm",
                             fallback_v="full")
        self._assert_same(explicit, world.run(_LegacyLayerSelector),
                          world.meta)
        self.assertEqual(implicit.selector.log, explicit.selector.log)

    def test_fullload_mode_is_unchanged(self):
        world = _World(2)
        new = world.run(LayerSelector, mode="fullload")
        old = world.run(_LegacyLayerSelector, mode="fullload")
        self._assert_same(new, old, world.meta)
        stats = new.selector.stats()
        self.assertEqual(stats["touched_chunk_fraction"], 1.0)
        self.assertEqual(stats["touched_k_chunk_fraction"], 1.0)
        self.assertEqual(stats["touched_v_chunk_fraction"], 1.0)
        self.assertEqual(stats["fallback_v_token_exact_bytes"], 0)
        self.assertIsNone(stats["importance_metric"])
        self.assertIsNone(stats["fallback_v"])

    def test_default_k_and_v_fractions_equal_the_historical_fraction(self):
        world = _World(3)
        stats = world.run(LayerSelector).selector.stats()
        self.assertAlmostEqual(stats["touched_k_chunk_fraction"],
                               stats["touched_chunk_fraction"], places=12)
        self.assertAlmostEqual(stats["touched_v_chunk_fraction"],
                               stats["touched_chunk_fraction"], places=12)

    def test_invalid_options_are_rejected(self):
        world = _World(0)
        with self.assertRaises(ValueError):
            world.run(LayerSelector, importance_metric="attention")
        with self.assertRaises(ValueError):
            world.run(LayerSelector, fallback_v="per_head")


class FallbackSelectedVTests(unittest.TestCase):
    def tearDown(self):
        BIAS.clear()

    def _check(self, metric, seed):
        world = _World(seed)
        m = world.meta
        v0, vn, nc = m["v_token_start"], m["v_token_num"], m["n_chunks_per_layer"]
        full = world.run(LayerSelector, importance_metric=metric,
                         fallback_v="full")
        sel = world.run(LayerSelector, importance_metric=metric,
                        fallback_v="selected")

        # the per-head keep masks do not depend on the V policy
        self.assertEqual(sorted(full.bias), sorted(sel.bias))
        for li in full.bias:
            self.assertTrue(torch.equal(full.bias[li], sel.bias[li]), li)
        self.assertEqual([r["mode"] for r in full.selector.log],
                         [r["mode"] for r in sel.selector.log])
        self.assertEqual([r["sim"] for r in full.selector.log],
                         [r["sim"] for r in sel.selector.log])

        fallback_layers = [r["layer"] for r in sel.selector.log
                           if r["mode"] == "fallback"]
        self.assertTrue(fallback_layers)
        exact_total = v_read_total = 0
        for record in sel.selector.log:
            li = record["layer"]
            keep = _keep_from_bias(sel.bias[li], m)          # (H, vn)
            calls = [c for c in sel.ctx.reader.calls if c[1] == li]
            stored_v = world.v[li].float().permute(1, 0, 2)  # (H, vn, hd)
            cache_v = sel.ctx.cache.v[li][0, :, v0:]
            # K is identical to the full policy in every layer
            self.assertTrue(torch.equal(sel.ctx.cache.k[li],
                                        full.ctx.cache.k[li]), li)
            if record["mode"] != "fallback":
                self.assertEqual(record["k_chunks"], record["v_chunks"])
                self.assertEqual(record["chunks"], record["k_chunks"])
                continue

            union = keep.any(dim=0).nonzero(as_tuple=True)[0].tolist()
            for separator in m["newline_idx"]:
                self.assertIn(separator, union)
            want = tuple(chunks_for_tokens(union, vn, m["chunk_size"]))
            # K: probe sidecar, then the WHOLE key block; V: union chunks only
            self.assertEqual(calls, [("probe", li), ("full", li, "k"),
                                     ("chunks", li, "v", want)])
            self.assertNotIn(("full", li, "v"), sel.ctx.reader.calls)
            self.assertEqual(record["k_chunks"], nc)
            self.assertEqual(record["v_chunks"], len(want))
            self.assertEqual(record["chunks"], (nc + len(want)) / 2)

            loaded = torch.zeros(vn, dtype=torch.bool)
            for c in want:
                loaded[c * m["chunk_size"]:(c + 1) * m["chunk_size"]] = True
            # every (head, token) the mask keeps has its real V in the cache
            self.assertTrue(bool((keep & ~loaded.unsqueeze(0)).sum() == 0))
            self.assertTrue(torch.equal(cache_v[keep], stored_v[keep]))
            self.assertTrue(bool((cache_v[keep].abs().sum(-1) > 0).all()))
            # rows of read chunks are loaded for all heads, the rest never
            self.assertTrue(torch.equal(cache_v[:, loaded],
                                        stored_v[:, loaded]))
            self.assertTrue(bool((cache_v[:, ~loaded] == 0).all()))
            # kept rows agree with the full policy's V, so the attention
            # output cannot differ
            full_v = full.ctx.cache.v[li][0, :, v0:]
            self.assertTrue(torch.equal(cache_v[keep], full_v[keep]))

            self.assertEqual(record["v_read_bytes"],
                             int(loaded.sum()) * ROW_BYTES)
            self.assertEqual(record["v_token_exact_bytes"],
                             int(keep.sum()) * m["head_dim"] * 2)
            self.assertLessEqual(record["v_token_exact_bytes"],
                                 record["v_read_bytes"])
            exact_total += record["v_token_exact_bytes"]
            v_read_total += record["v_read_bytes"]

        # ---- accounting: K bytes equal, V bytes never larger, probe equal
        io_full, io_sel = _io(full.counter), _io(sel.counter)
        self.assertEqual(io_sel["per_kind"]["k"], io_full["per_kind"]["k"])
        self.assertEqual(io_sel["per_kind"]["probe"],
                         io_full["per_kind"]["probe"])
        self.assertLessEqual(io_sel["per_kind"]["v"]["bytes"],
                             io_full["per_kind"]["v"]["bytes"])
        saved = sum((nc - r["v_chunks"]) for r in sel.selector.log
                    if r["mode"] == "fallback")
        self.assertEqual(
            io_full["per_kind"]["v"]["bytes"] - io_sel["per_kind"]["v"]["bytes"],
            saved * m["chunk_size"] * ROW_BYTES)

        stats_full, stats_sel = full.selector.stats(), sel.selector.stats()
        self.assertEqual(stats_sel["fallback_v"], "selected")
        self.assertEqual(stats_sel["importance_metric"], metric)
        self.assertEqual(stats_sel["fallback_rate"],
                         stats_full["fallback_rate"])
        self.assertEqual(stats_sel["touched_k_chunk_fraction"],
                         stats_full["touched_k_chunk_fraction"])
        self.assertLessEqual(stats_sel["touched_v_chunk_fraction"],
                             stats_full["touched_v_chunk_fraction"])
        self.assertAlmostEqual(
            stats_sel["touched_chunk_fraction"],
            (stats_sel["touched_k_chunk_fraction"]
             + stats_sel["touched_v_chunk_fraction"]) / 2, places=12)
        # measured and hypothetical V volumes are separate fields
        self.assertEqual(stats_sel["fallback_v_read_bytes"], v_read_total)
        self.assertEqual(stats_sel["fallback_v_token_exact_bytes"],
                         exact_total)
        self.assertEqual(stats_full["fallback_v_token_exact_bytes"],
                         exact_total)
        self.assertEqual(stats_full["fallback_v_read_bytes"],
                         len(fallback_layers) * vn * ROW_BYTES)
        self.assertEqual(stats_sel["fallback_layers"], len(fallback_layers))
        self.assertEqual(stats_sel["fallback_k_chunks"],
                         len(fallback_layers) * nc)
        return full, sel, world

    def test_selected_reads_all_k_and_only_union_v_chunks(self):
        for metric in ("sparsevlm", "h2o"):
            for seed in range(4):
                with self.subTest(metric=metric, seed=seed):
                    self._check(metric, seed)

    def test_some_v_chunk_is_actually_skipped(self):
        # Sparse per-head selections: 1 spatial token per head, so the union
        # cannot cover the layer and "selected" must read fewer V bytes.
        world = _World(5)
        world.meta["n_spatial"] = 4          # ceil(4 * .25) == 1 token/head
        full = world.run(LayerSelector, fallback_v="full")
        sel = world.run(LayerSelector, fallback_v="selected")
        v_full = _io(full.counter)["per_kind"]["v"]["bytes"]
        v_sel = _io(sel.counter)["per_kind"]["v"]["bytes"]
        self.assertLess(v_sel, v_full)
        self.assertLess(sel.selector.stats()["touched_v_chunk_fraction"],
                        full.selector.stats()["touched_v_chunk_fraction"])
        for li in full.bias:
            self.assertTrue(torch.equal(full.bias[li], sel.bias[li]))

    def test_masked_attention_output_is_identical_under_both_policies(self):
        # CPU analogue of the GPU criterion: V rows that were not read are
        # masked for every head, so attention over the prefix is unchanged.
        full, sel, world = self._check("sparsevlm", 0)
        m = world.meta
        H, hd, v0 = m["num_heads"], m["head_dim"], m["v_token_start"]
        for li in range(m["num_layers"]):
            q = world.hidden[li].view(1, N_SUFFIX, H, hd).transpose(1, 2)[0]
            outputs = []
            for run in (full, sel):
                keys = run.ctx.cache.k[li][0]                 # (H, P, hd)
                values = run.ctx.cache.v[li][0]
                bias = run.bias[li][0, :, 0].float()          # (h|1, P)
                logits = q @ keys.transpose(1, 2) / math.sqrt(hd)
                weights = (logits + bias.unsqueeze(1)).softmax(dim=-1)
                outputs.append(weights @ values)
            self.assertTrue(torch.equal(outputs[0], outputs[1]), li)


class H2OSelectorTests(unittest.TestCase):
    def tearDown(self):
        BIAS.clear()

    def test_h2o_scores_reach_selection_and_ignore_rater_rows(self):
        world = _World(4)
        m = world.meta
        forbidden = AssertionError("rater scoring must not run under h2o")
        with mock.patch("mmimpress.sparsevlm.rater_visual_scores_from_qk",
                        side_effect=forbidden):
            a = world.run(LayerSelector, rater_rows=(0, 2),
                          importance_metric="h2o")
            b = world.run(LayerSelector, rater_rows=(),
                          importance_metric="h2o")
        for li in a.bias:
            self.assertTrue(torch.equal(a.bias[li], b.bias[li]))
        self.assertEqual(a.selector.log, b.selector.log)

        # fallback layers keep exactly the per-head top-k of the explicit
        # column sums (separators excluded from ranking, always kept)
        H, hd = m["num_heads"], m["head_dim"]
        v0, vn = m["v_token_start"], m["v_token_num"]
        checked = 0
        for record in a.selector.log:
            if record["mode"] != "fallback":
                continue
            li = record["layer"]
            q = world.hidden[li].view(1, N_SUFFIX, H, hd).transpose(1, 2)[0]
            attn = _explicit_attention(
                q, world.sys_k[li], world.k[li].float().permute(1, 0, 2), q)
            score = attn[:, :, v0:v0 + vn].sum(dim=1)
            score[:, m["newline_idx"]] = float("-inf")
            expected = torch.zeros(H, vn, dtype=torch.bool)
            expected.scatter_(1, score.topk(a.selector.k_keep, -1).indices,
                              True)
            expected[:, m["newline_idx"]] = True
            self.assertTrue(torch.equal(_keep_from_bias(a.bias[li], m),
                                        expected), li)
            checked += 1
        self.assertGreater(checked, 0)

    def test_metric_changes_the_selection(self):
        world = _World(4)
        h2o = world.run(LayerSelector, rater_rows=(0,),
                        importance_metric="h2o")
        rater = world.run(LayerSelector, rater_rows=(0,),
                          importance_metric="sparsevlm")
        self.assertTrue(any(not torch.equal(h2o.bias[li], rater.bias[li])
                            for li in h2o.bias))


# ================================================== Server / calibration
class _FakeModel:
    """Runs the decoder layers once so forward pre-hooks fire."""

    def __init__(self, world):
        self.world = world
        self.device = torch.device("cpu")
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        for li, layer in enumerate(self.world.layers):
            layer(hidden_states=self.world.hidden[li],
                  position_embeddings=(None, None))
        return SimpleNamespace()


def _server_world(seed=0):
    world = _World(seed)
    world.runner.model = _FakeModel(world)
    return world


class ServerMetricTests(unittest.TestCase):
    def tearDown(self):
        BIAS.clear()

    def _request(self, world, server, **kwargs):
        ctx = world.context()
        ctx.reader.drop_all = lambda: None
        ctx.cache.new_request = lambda: object()
        timing = {"first_token_at": 1.0, "finished_at": 2.0,
                  "prefill_ms": 1.0, "decode_ms": 1.0, "generated_tokens": 1,
                  "generated_token_ids": [3], "generated_token_count": 1}

        def decode(cache, suffix_ids, prefix_len):
            # stands in for the prefill: fires every layer's selector hook
            world.runner.model()
            return "yes", 3, timing

        with mock.patch("torch.cuda.synchronize", return_value=None), \
             mock.patch(
                 "mmimpress.serve.ml.apply_rotary_pos_emb",
                 side_effect=lambda query, key, cos, sin: (query, key)), \
             mock.patch.object(Server, "_decode", side_effect=decode):
            result = server.request(ctx, suffix_ids=torch.arange(N_SUFFIX),
                                    **kwargs)
        self.assertEqual(world.runner.model.calls, 1)
        world.runner.model.calls = 0
        return result

    def test_h2o_request_never_selects_raters_or_needs_v_hidden(self):
        world = _server_world()
        forbidden = AssertionError("raters() must not be called under h2o")
        with mock.patch.object(Server, "raters", side_effect=forbidden), \
             mock.patch("mmimpress.sparsevlm.select_raters",
                        side_effect=forbidden):
            server = Server(world.runner, importance_metric="h2o")
            result = self._request(world, server)
            # per-request override of a default server
            override = self._request(world, Server(world.runner),
                                     importance_metric="h2o",
                                     fallback_v="selected")
        self.assertEqual(result["n_raters"], 0)
        self.assertEqual(result["n_score_rows"], N_SUFFIX)
        self.assertEqual(result["importance_metric"], "h2o")
        self.assertEqual(result["fallback_v"], "full")
        # selection really ran in every layer, with no rater set at all
        self.assertEqual(result["probe_rate"] + result["fallback_rate"], 1.0)
        self.assertGreater(result["io"]["per_kind"]["probe"]["bytes"], 0)
        self.assertEqual(override["importance_metric"], "h2o")
        self.assertEqual(override["fallback_v"], "selected")

    def test_default_request_still_selects_raters_exactly_once(self):
        world = _server_world()
        raters = torch.tensor([0, 2])
        with mock.patch.object(Server, "raters",
                               return_value=raters) as mocked:
            result = self._request(world, Server(world.runner))
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(result["n_raters"], 2)
        self.assertEqual(result["n_score_rows"], 2)
        self.assertEqual(result["importance_metric"], "sparsevlm")
        self.assertEqual(result["fallback_v"], "full")

    def test_server_rejects_unknown_options(self):
        world = _server_world()
        with self.assertRaises(ValueError):
            Server(world.runner, importance_metric="attention")
        with self.assertRaises(ValueError):
            Server(world.runner, fallback_v="per_head")


class CalibrationMetricTests(unittest.TestCase):
    def tearDown(self):
        BIAS.clear()

    def _calibrate(self, world, server, **kwargs):
        ctx = world.context()
        ctx.cache.new_request = lambda: object()
        with mock.patch(
                "mmimpress.serve.ml.apply_rotary_pos_emb",
                side_effect=lambda query, key, cos, sin: (query, key)), \
             mock.patch("mmimpress.serve.suffix_ids_for",
                        return_value=torch.arange(N_SUFFIX)):
            return calibrate_image(server, ctx, ["q1", "q2"], **kwargs)

    def _expected(self, world, rows=None):
        m = world.meta
        H, hd = m["num_heads"], m["head_dim"]
        v0, vn = m["v_token_start"], m["v_token_num"]
        out = []
        for li in range(m["num_layers"]):
            q = world.hidden[li].view(1, N_SUFFIX, H, hd).transpose(1, 2)[0]
            attn = _explicit_attention(
                q, world.sys_k[li], world.k[li].float().permute(1, 0, 2), q)
            block = attn[:, :, v0:v0 + vn]
            per_head = (block.sum(dim=1) if rows is None
                        else block[:, rows].mean(dim=1))
            out.append(per_head.mean(dim=0))
        return torch.stack(out)

    def test_h2o_calibration_is_head_mean_column_sum_without_raters(self):
        world = _server_world(3)
        forbidden = AssertionError("raters() must not be called under h2o")
        with mock.patch.object(Server, "raters", side_effect=forbidden):
            server = Server(world.runner, importance_metric="h2o")
            scores = self._calibrate(world, server)
            # explicit argument on a default server takes precedence
            explicit = self._calibrate(world, Server(world.runner),
                                       importance_metric="h2o")
        self.assertEqual(tuple(scores.shape),
                         (world.meta["num_layers"], world.meta["v_token_num"]))
        self.assertTrue(torch.allclose(scores.double(), self._expected(world),
                                       rtol=1e-4, atol=1e-5))
        self.assertTrue(torch.equal(scores, explicit))

    def test_default_calibration_still_uses_rater_rows(self):
        world = _server_world(3)
        with mock.patch.object(Server, "raters",
                               return_value=torch.tensor([1, 3])) as mocked:
            scores = self._calibrate(world, Server(world.runner))
        self.assertEqual(mocked.call_count, 2)          # once per question
        self.assertTrue(torch.allclose(
            scores.double(), self._expected(world, rows=[1, 3]),
            rtol=1e-4, atol=1e-5))

    def test_calibrator_default_metric_is_sparsevlm(self):
        world = _server_world(0)
        calibrator = Calibrator(world.runner, world.context(),
                                torch.tensor([0]))
        self.assertEqual(calibrator.importance_metric, "sparsevlm")
        with self.assertRaises(ValueError):
            Calibrator(world.runner, world.context(), torch.tensor([0]),
                       importance_metric="attention")


# ================================================================== scripts
class EvalScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eval04 = _load_script("04_eval.py", "eval04_for_impress_fidelity")

    def test_selectors_and_default_cli_values(self):
        selectors = self.eval04.SELECTORS
        self.assertEqual(selectors["sparsevlm"], ("impress", {}))
        self.assertEqual(selectors["sparsevlm_fallback_v_selected"],
                         ("impress", {"fallback_v": "selected"}))

    def test_pack_carries_options_breakdown_and_token_ids(self):
        result = {
            "answer": "yes", "ttft": 0.01, "decode_latency": 0.02,
            "e2e_latency": 0.03, "generated_tokens": 2,
            "generated_token_ids": [3, 2],
            "importance_metric": "h2o", "fallback_v": "selected",
            "touched_chunk_fraction": 0.7, "touched_k_chunk_fraction": 0.8,
            "touched_v_chunk_fraction": 0.6, "fallback_rate": 0.25,
            "mean_jaccard": 0.4, "fallback_v_read_bytes": 50,
            "fallback_v_token_exact_bytes": 12, "n_score_rows": 5,
            "io": {"bytes": 130, "mb": 0.00013, "ms": 1.0, "preads": 5,
                   "chunk_units": 2,
                   "per_kind": {"k": {"bytes": 60, "preads": 2, "units": 1},
                                "v": {"bytes": 50, "preads": 2, "units": 1},
                                "probe": {"bytes": 20, "preads": 1,
                                          "units": 0}}},
        }
        packed = self.eval04._pack(result, {"answer": "yes"}, "gqa")
        self.assertEqual(packed["generated_token_ids"], [3, 2])
        self.assertEqual(packed["importance_metric"], "h2o")
        self.assertEqual(packed["fallback_v"], "selected")
        self.assertEqual(packed["probe_read_bytes"], 20)
        self.assertEqual(packed["k_read_bytes"], 60)
        self.assertEqual(packed["v_read_bytes"], 50)
        self.assertEqual(packed["touched_k_chunk_fraction"], 0.8)
        self.assertEqual(packed["touched_v_chunk_fraction"], 0.6)
        # measured and hypothetical stay separate, and neither is the total
        self.assertEqual(packed["fallback_v_read_bytes"], 50)
        self.assertEqual(packed["fallback_v_token_exact_bytes"], 12)
        self.assertEqual(packed["total_actual_pread_bytes"], 130)
        self.assertEqual(packed["ssd_read_bytes"], 130)

    def test_store_importance_metric_inference(self):
        infer = self.eval04._store_importance_metric
        self.assertEqual(infer({"importance_metric": "h2o",
                                "reordered": True}), "h2o")
        # legacy calibrated importance layout: only sparsevlm existed
        self.assertEqual(infer({"reordered": True,
                                "order_is_per_layer": True}), "sparsevlm")
        self.assertEqual(infer({"reordered": True, "order_is_per_layer": True,
                                "layout_method": "calib_importance_sep_tail"}),
                         "sparsevlm")
        # layouts that use no importance statistic
        self.assertIsNone(infer({"reordered": False}))
        self.assertIsNone(infer({"reordered": True, "order_is_per_layer": False,
                                 "layout_method": "visionzip_image_only"}))
        self.assertIsNone(infer({"importance_metric": None,
                                 "reordered": False}))


class ReorderScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reorder = _load_script("02_reorder.py",
                                   "reorder02_for_impress_fidelity")

    def _main(self, *argv):
        with mock.patch("sys.argv", ["02_reorder.py", *argv]), \
             mock.patch("sys.stderr"):
            self.reorder.main()

    def test_metric_is_rejected_where_no_calibration_runs(self):
        for argv in (("--order", "morton", "--importance-metric", "h2o"),
                     ("--order", "importance", "--separator-tail",
                      "--reference-order-store", "/nonexistent",
                      "--importance-metric", "h2o")):
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit) as caught:
                    self._main(*argv)
                self.assertEqual(caught.exception.code, 2)
        with self.assertRaises(SystemExit):
            self._main("--order", "importance", "--importance-metric", "bad")


if __name__ == "__main__":
    unittest.main()

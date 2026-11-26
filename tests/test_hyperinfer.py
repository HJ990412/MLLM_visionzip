"""HyperInfer's next-layer KV prefetch on top of the IMPRESS port.

Contract: docs/mt_gqa_hyperinfer_contract.md; source table: docs/hyperinfer_port_contract.md.

``PublicPrefetchTests`` call the public ``PrefixKVLayer.prefetch_by_chunk``
(https://github.com/ISCS-ZJU/HyperInfer, commit 91df7a2,
``h2o_flexgen/flexgen/my_pcache_fast.py``) and compare it with
``mmimpress.hyperinfer`` on the same inputs.  They need a checkout and are
skipped without it:

    IMPRESS_OFFICIAL_REPO=/path/to/HyperInfer python -m unittest tests.test_hyperinfer

How the public method is called: the class constructor writes chunk files and
builds caches, so the method is called unbound on a plain object that carries
the attributes it reads (``reorder``, ``reverse_reorder``, ``chunk_size``,
``device_map`` ...).  ``_prefetch_chunks_direct_pinned`` -- the public loop
over the chunks -- is the real one; only its two data movers
(``_chunk_source_readonly``, ``_copy_readonly_source_to_gpu``) are replaced by
recorders, and ``torch.empty`` is kept on the host.  So the order conversion,
the chunk ids, the budget cut, the half rule, the K-then-V order and the
prefetched token set are the public code's.

Not callable without the model and its global state (both are single lines
inside the 1,300-line layer function of ``flex_opt_fast_pcache.py``): the
prediction (``prefetch_tokenids = sele_tokenids.clone()``, line 1108) and the
missing tokens (``sele_tokenids[~torch.isin(sele_tokenids, prefetched)]``,
lines 1125-1126).  The same two operations are written out in
``PublicPrefetchTests._public_lines`` and compared.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_impress_official import _World  # noqa: E402

from mmimpress import hyperinfer as hi  # noqa: E402
from mmimpress import impress_official as io_  # noqa: E402
from mmimpress.serve import BIAS, bias_from_keep  # noqa: E402
from mmimpress.store import IOCounter, chunk_span  # noqa: E402

OFFICIAL = os.environ.get("IMPRESS_OFFICIAL_REPO")
_PC = None


def _public():
    global _PC
    if _PC is None:
        flexgen = Path(OFFICIAL) / "h2o_flexgen" / "flexgen"
        saved_path = list(sys.path)
        saved = {name: sys.modules.get(name) for name in ("myheap", "my_pcache_fast")}
        sys.path.insert(0, str(flexgen))
        try:
            import my_pcache_fast
            _PC = my_pcache_fast
        finally:
            sys.path[:] = saved_path
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
    return _PC


def _orders(seed, layers, vn):
    g = torch.Generator().manual_seed(seed)
    return [torch.randperm(vn, generator=g).tolist() for _ in range(layers)]


# ===================================================================== pure
class PureTests(unittest.TestCase):
    def test_position_map_moves_a_token_to_its_place_in_the_next_layer(self):
        orders = _orders(0, 4, 37)
        maps = hi.layer_position_maps(orders)
        self.assertEqual(len(maps), 3)
        for li, m in enumerate(maps):
            for r in range(37):
                self.assertEqual(orders[li + 1][int(m[r])], orders[li][r])

    def test_predicted_chunks_are_sorted_unique_chunks_of_the_used_tokens(self):
        orders = _orders(1, 2, 50)
        m = hi.layer_position_maps(orders)[0]
        keep = torch.zeros(50, dtype=torch.bool)
        keep[[0, 3, 4, 17, 49]] = True
        want = sorted({orders[1].index(orders[0][r]) // 8 for r in (0, 3, 4, 17, 49)})
        self.assertEqual(hi.predicted_chunks(keep, m, 8), want)
        self.assertEqual(hi.predicted_chunks(torch.ones(50, dtype=torch.bool), m, 8), list(range(7)))
        self.assertEqual(hi.predicted_chunks(torch.zeros(50, dtype=torch.bool), m, 8), [])

    def test_budget_cut(self):
        ids = [0, 1, 4, 5, 9]
        self.assertEqual(hi.truncate_chunks(ids, 0.0, 1.0, 2.0), ([0, 1], "half"))
        self.assertEqual(hi.truncate_chunks([7], 0.0, 1.0, 2.0), ([], "half"))
        self.assertEqual(hi.truncate_chunks(ids, 9.0, 1.0, 2.0), ([0, 1, 4], "budget"))      # exactly three pairs fit
        self.assertEqual(hi.truncate_chunks(ids, 8.999, 1.0, 2.0), ([0, 1], "budget"))
        self.assertEqual(hi.truncate_chunks(ids, 2.0, 1.0, 2.0), ([], "budget"))             # positive but below one pair
        self.assertEqual(hi.truncate_chunks(ids, 1e9, 1.0, 2.0), (ids, "budget"))
        self.assertEqual(hi.truncate_chunks([], 5.0, 1.0, 2.0), ([], "budget"))

    def test_chunk_token_mask_and_missing(self):
        mask = hi.chunk_token_mask([0, 4], 37, 8)
        self.assertEqual(mask.nonzero().flatten().tolist(), list(range(8)) + list(range(32, 37)))
        self.assertEqual(hi.missing_chunks([5, 1, 3, 1], [3, 9]), [1, 5])
        self.assertEqual(hi.missing_chunks([2], []), [2])

    def test_state_rejects_bad_constants(self):
        with self.assertRaises(ValueError):
            hi.PrefetchState(0.0, 1e-3)
        s = hi.PrefetchState(1e-3, 2e-3)
        self.assertEqual((s.budget_s, s.requests), (0.0, 0))


# =================================================================== public
@unittest.skipUnless(OFFICIAL, "set IMPRESS_OFFICIAL_REPO to the public HyperInfer checkout")
class PublicPrefetchTests(unittest.TestCase):
    H, HD = 2, 4

    def _public_call(self, order_next, token_ids, vn, cs, budget, k_s, v_s):
        """The public ``prefetch_by_chunk``: (chunks in read order with kind, prefetched ORIGINAL token ids)."""
        pc = _public()
        nc = (vn + cs - 1) // cs
        calls = []
        stub = SimpleNamespace(
            reorder=io_.reverse_index(order_next), reverse_reorder=torch.as_tensor(order_next, dtype=torch.long),
            chunk_size=cs, layer_id=1, disk_type="KV_Division", chunk_num=nc,
            device_map=["k_disk"] * nc + ["v_disk"] * nc, shape=(1, vn, self.H, self.HD), torch_dtype=torch.float16,
            key_tokens=[("k", c) for c in range(nc)], value_tokens=[("v", c) for c in range(nc)])
        stub._chunk_source_readonly = lambda chunk, want_key: calls.append((chunk[1], "k" if want_key else "v")) or chunk
        stub._copy_readonly_source_to_gpu = lambda target, out_pos, take, source, pinned: None
        stub._prefetch_chunks_direct_pinned = types.MethodType(pc.PrefixKVLayer._prefetch_chunks_direct_pinned, stub)
        real_empty = torch.empty

        def host_empty(*args, **kwargs):
            kwargs.pop("device", None)
            return real_empty(*args, **kwargs)

        with mock.patch.dict(pc.AVG_CHUNK_TIME, {"k_disk": k_s, "v_disk": v_s}), mock.patch.object(torch, "empty", host_empty):
            key, value, tokenids, pinned = pc.PrefixKVLayer.prefetch_by_chunk(
                stub, pos_id=torch.as_tensor(token_ids, dtype=torch.long), need_key=True, need_value=True,
                k_v_num_tokens=vn, time_budget=budget)
        self.assertEqual(key.shape[0], tokenids.numel())
        return calls, sorted(tokenids.tolist())

    @staticmethod
    def _public_lines(sele_tokenids, prefetched_tokenids):
        """flex_opt_fast_pcache.py line 1108 and lines 1125-1126, written out."""
        prefetch_tokenids = sele_tokenids.clone()
        prefetch_mask = ~torch.isin(sele_tokenids, prefetched_tokenids)
        return prefetch_tokenids, sele_tokenids[prefetch_mask]

    def test_chunks_cut_and_token_set_match_the_public_code(self):
        g = torch.Generator().manual_seed(7)
        checked = rules = 0
        for vn, cs in ((37, 8), (64, 8), (576, 64), (2144, 64), (130, 64)):
            orders = _orders(vn, 2, vn)
            m = hi.layer_position_maps(orders)[0]
            nc = (vn + cs - 1) // cs
            for trial in range(12):
                n_keep = [1, max(1, vn // 4), vn][trial % 3]
                keep = torch.zeros(vn, dtype=torch.bool)
                keep[torch.randperm(vn, generator=g)[:n_keep]] = True
                tokens = torch.tensor(orders[0])[keep]                       # the public code works in original token ids
                k_s, v_s = 0.3e-3, 0.5e-3
                for budget in (0.0, 0.1e-3, 0.8e-3, 0.8e-3 * 3, 0.8e-3 * 3 - 1e-9, 2.5e-3, 1.0):
                    predicted = hi.predicted_chunks(keep, m, cs)
                    chosen, rule = hi.truncate_chunks(predicted, budget, k_s, v_s)
                    calls, public_tokens = self._public_call(orders[1], tokens, vn, cs, budget, k_s, v_s)
                    self.assertEqual(calls, [(c, kind) for c in chosen for kind in ("k", "v")])
                    ours = torch.tensor(orders[1])[hi.chunk_token_mask(chosen, vn, cs)]
                    self.assertEqual(sorted(ours.tolist()), public_tokens)
                    self.assertEqual(rule, "budget" if budget > 0.0 else "half")
                    # prediction and missing tokens (the two lines that cannot be called)
                    prediction, missing = self._public_lines(tokens, torch.tensor(public_tokens, dtype=torch.long))
                    self.assertEqual(sorted(prediction.tolist()), sorted(tokens.tolist()))
                    needed = sorted({orders[1].index(int(t)) // cs for t in tokens})
                    self.assertEqual(needed, predicted)
                    miss_chunks = hi.missing_chunks(needed, chosen)
                    self.assertEqual(sorted({orders[1].index(int(t)) // cs for t in missing}), miss_chunks)
                    self.assertTrue(all(c < nc for c in chosen))
                    checked += 1
                    rules += rule == "half"
        self.assertGreater(checked, 400)
        self.assertGreater(rules, 50)

    def test_public_constants_and_single_worker(self):
        pc = _public()
        self.assertEqual(pc.AVG_CHUNK_TIME["disk"], 0.00040818)
        import inspect
        src = inspect.getsource(pc.Pcache.__init__)
        self.assertIn("ThreadPoolExecutor(max_workers=1)", src)


# ================================================================= selector
class _Run:
    """One request through a selector on the synthetic store of ``test_impress_official._World``."""

    def __init__(self, world, make):
        BIAS.clear()
        self.ctx = world.context()
        self.counter = IOCounter()
        self.selector = make(world.runner, self.ctx, self.counter)
        with mock.patch("mmimpress.impress_official.ml.apply_rotary_pos_emb",
                        side_effect=lambda query, key, cos, sin: (query, key)):
            with self.selector:
                for li, layer in enumerate(world.layers):
                    layer(hidden_states=world.hidden[li], position_embeddings=(None, None))
                for li, layer in enumerate(world.layers):      # decoding steps
                    layer(hidden_states=world.hidden[li][:, :1], position_embeddings=(None, None))
        self.bias = {li: BIAS[li].clone() for li in sorted(BIAS)}
        BIAS.clear()
        self.ctx.reader.close()


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.pool = ThreadPoolExecutor(max_workers=1)

    def tearDown(self):
        self.pool.shutdown(wait=True)
        BIAS.clear()

    def _impress(self, w):
        return _Run(w, lambda runner, ctx, counter: io_.OfficialImpressSelector(
            runner, ctx, counter=counter, prefetch="next_layer", prefetch_executor=self.pool))

    def _hyper(self, w, mode, state=None):
        return _Run(w, lambda runner, ctx, counter: hi.HyperInferSelector(
            runner, ctx, state=state, kv_prefetch=mode, counter=counter, prefetch_executor=self.pool))

    def _same_outcome(self, a, b):
        self.assertEqual([r["mode"] for r in a.selector.log], [r["mode"] for r in b.selector.log])
        self.assertEqual([r["sim"] for r in a.selector.log], [r["sim"] for r in b.selector.log])
        for li in a.selector.keep_masks:
            self.assertTrue(torch.equal(a.selector.keep_masks[li], b.selector.keep_masks[li]))
        self.assertEqual(sorted(a.bias), sorted(b.bias))
        for li in a.bias:
            self.assertTrue(torch.equal(a.bias[li], b.bias[li]))

    def test_prefetch_off_is_the_impress_selector(self):
        with tempfile.TemporaryDirectory() as td:
            w = _World(td)
            a, b = self._impress(w), self._hyper(w, "off")
            self._same_outcome(a, b)
            self.assertEqual(a.counter.summary()["bytes"], b.counter.summary()["bytes"])
            self.assertEqual(a.counter.summary()["preads"], b.counter.summary()["preads"])
            self.assertEqual({k: (v["bytes"], v["preads"]) for k, v in a.counter.per_kind.items()},
                             {k: (v["bytes"], v["preads"]) for k, v in b.counter.per_kind.items()})
            for li in range(w.L):
                self.assertTrue(torch.equal(a.ctx.cache.k[li], b.ctx.cache.k[li]))
                self.assertTrue(torch.equal(a.ctx.cache.v[li], b.ctx.cache.v[li]))
            self.assertEqual(b.selector.hyperinfer_stats(), {"kv_prefetch": "off"})
            self.assertEqual(b.selector.stats()["prefetch"], "next_layer")

    def test_prefetch_on_selects_masks_and_loads_what_impress_does(self):
        for mode in ("async", "sync"):
            for k_s in (1e-12, 1e3):          # after the first job: everything fits / nothing fits
                with tempfile.TemporaryDirectory() as td:
                    w = _World(td)
                    a = self._impress(w)
                    state = hi.PrefetchState(k_s, k_s)
                    b = self._hyper(w, mode, state)
                    self._same_outcome(a, b)
                    for li in range(w.L):     # the rows moved to the cache are IMPRESS's rows: the needed chunks, nothing else
                        self.assertTrue(torch.equal(a.ctx.cache.k[li], b.ctx.cache.k[li]), (mode, k_s, li))
                        self.assertTrue(torch.equal(a.ctx.cache.v[li], b.ctx.cache.v[li]), (mode, k_s, li))
                    self.assertEqual({r["mode"] for r in b.selector.log}, {"consensus", "fallback"})
                    s = b.selector.hyperinfer_stats()
                    cols = s["layers"]
                    m = w.meta
                    nc, cs, vn = m["n_chunks_per_layer"], m["chunk_size"], w.vn
                    row = w.H * w.hd * 2
                    self.assertEqual(cols["rule"], [None, "half"] + ["budget"] * (w.L - 2))
                    self.assertEqual(cols["prefetched_chunks"][0], 0)
                    for li in range(w.L):
                        needed = nc if cols["fallback"][li] else len(
                            {int(p) // cs for p in b.selector.keep_masks[li].nonzero().flatten()})
                        self.assertEqual(cols["needed_chunks"][li], needed)
                        self.assertEqual(cols["hit_chunks"][li] + cols["missing_chunks"][li], needed)
                        self.assertEqual(cols["hit_chunks"][li] + cols["wasted_chunks"][li], cols["prefetched_chunks"][li])
                        self.assertEqual(cols["predicted_chunks"][li] - cols["cut_chunks"][li], cols["prefetched_chunks"][li])
                        self.assertEqual(cols["used_tokens"][li], int(b.selector.keep_masks[li].sum()))
                    # the first job (budget 0) reads the first half of the predicted chunks
                    self.assertEqual(cols["prefetched_chunks"][1], cols["predicted_chunks"][1] // 2)
                    if k_s > 1:
                        self.assertEqual(cols["prefetched_chunks"][2:], [0] * (w.L - 2))
                    else:
                        self.assertEqual(cols["cut_chunks"][2:], [0] * (w.L - 2))
                        for li in range(2, w.L):          # the prediction is what the previous layer used
                            prev = b.selector.keep_masks[li - 1]
                            want = hi.predicted_chunks(prev, hi.layer_position_maps(m["order"])[li - 1], cs)
                            self.assertEqual(cols["predicted_chunks"][li], len(want))
                    # bytes: probe blocks + every prefetched chunk + every synchronously read chunk, K and V
                    kv = b.counter.per_kind["k"]["bytes"] + b.counter.per_kind["v"]["bytes"]
                    self.assertEqual(kv, s["prefetched_bytes"] + s["sync_bytes"])
                    self.assertEqual(b.counter.per_kind["k"]["bytes"], b.counter.per_kind["v"]["bytes"])
                    self.assertEqual(b.counter.per_kind["probe"]["bytes"], a.counter.per_kind["probe"]["bytes"])
                    impress_kv = a.counter.per_kind["k"]["bytes"] + a.counter.per_kind["v"]["bytes"]
                    self.assertEqual(kv - s["wasted_bytes"], impress_kv)
                    self.assertGreaterEqual(s["wasted_bytes"], 0)
                    self.assertEqual(sum(cols["missing_chunks"]) * 0 + s["sync_bytes"],
                                     sum(sum(chunk_span(c, vn, cs)[1] - chunk_span(c, vn, cs)[0] for c in b.selector._reads[li]["missing"])
                                         for li in range(w.L)) * row * 2)
                    self.assertEqual(state.requests, 0)          # counted by the request function, not the selector
                    self.assertGreater(state.budget_s, 0.0)       # the last measured (compute + selection)
                    self.assertEqual([r["source"] for r in b.selector.probe_log], ["hook"] + ["prefetch"] * (w.L - 1))

    def test_budget_state_carries_into_the_next_request(self):
        with tempfile.TemporaryDirectory() as td:
            w = _World(td)
            state = hi.PrefetchState(1e-12, 1e-12)
            first = self._hyper(w, "sync", state).selector.hyperinfer_stats()
            second = self._hyper(w, "sync", state).selector.hyperinfer_stats()
            self.assertEqual(first["layers"]["rule"][1], "half")
            self.assertEqual(first["half_rule_jobs"], 1)
            self.assertEqual(second["layers"]["rule"][1], "budget")          # the budget of the previous request, as in the public code
            self.assertEqual(second["half_rule_jobs"], 0)
            self.assertEqual(second["layers"]["cut_chunks"], [0] * w.L)

    def test_adjacent_layer_similarity_is_reported_on_image_tokens(self):
        with tempfile.TemporaryDirectory() as td:
            w = _World(td)
            b = self._hyper(w, "sync", hi.PrefetchState(1e-12, 1e-12))
            cols = b.selector.hyperinfer_stats()["layers"]
            m = w.meta
            n_sep = len(m["newline_idx"])
            for li in range(w.L - 1):
                sets = []
                for l in (li, li + 1):
                    keep = b.selector.keep_masks[l].clone()
                    keep[:n_sep] = False
                    sets.append({m["order"][l][int(p)] for p in keep.nonzero().flatten()})
                self.assertAlmostEqual(cols["jaccard_next"][li], len(sets[0] & sets[1]) / len(sets[0] | sets[1]))
                self.assertEqual(cols["jaccard_next_both_consensus"][li],
                                 not cols["fallback"][li] and not cols["fallback"][li + 1])

    def test_worker_read_is_one_chunk_per_pread_k_then_v(self):
        with tempfile.TemporaryDirectory() as td:
            w = _World(td)
            ctx = w.context()
            counter = IOCounter()
            k = hi.read_one_chunk(ctx.reader, 1, "k", 3, counter)
            rows, vals = ctx.reader.read_chunks(1, "k", [3])
            self.assertTrue(np.array_equal(k, vals.numpy()))
            self.assertEqual((counter.preads, counter.bytes), (1, k.nbytes))
            last = hi.read_one_chunk(ctx.reader, 0, "v", w.meta["n_chunks_per_layer"] - 1)
            self.assertEqual(last.shape[0], w.vn - (w.meta["n_chunks_per_layer"] - 1) * w.meta["chunk_size"])
            ctx.reader.close()

    def test_constructor_checks(self):
        with tempfile.TemporaryDirectory() as td:
            w = _World(td)
            ctx = w.context()
            with self.assertRaises(ValueError):
                hi.HyperInferSelector(w.runner, ctx, kv_prefetch="next_layer")
            with self.assertRaises(ValueError):
                hi.HyperInferSelector(w.runner, ctx, kv_prefetch="async", prefetch_executor=self.pool)       # no state
            with self.assertRaises(ValueError):
                hi.HyperInferSelector(w.runner, ctx, kv_prefetch="async", state=hi.PrefetchState(1, 1))      # no worker
            ctx.reader.close()


# ==================================================================== guard
class UnreadRowGuardTests(unittest.TestCase):
    META = {"v_token_start": 2, "v_token_num": 6, "prefix_len": 8, "num_layers": 1}

    def tearDown(self):
        BIAS.clear()

    @staticmethod
    def _attention(module, query, key, value, attention_mask, **kw):
        from mmimpress.serve import _min_val
        logits = query @ key.transpose(-1, -2)
        b = BIAS.get(module.layer_idx)
        if b is not None:
            logits = logits + torch.nn.functional.pad(b, (0, key.shape[-2] - b.shape[-1]))
        w = torch.softmax(logits, dim=-1)
        return w @ value, w

    def _call(self, key, value, keep, query):
        import transformers.models.llama.modeling_llama as ml
        module = SimpleNamespace(layer_idx=0)
        BIAS.clear()
        if keep is not None:
            BIAS[0] = bias_from_keep(keep, self.META, torch.device("cpu"), dtype=torch.float32)
        guard = hi.UnreadRowGuard(self.META)
        with mock.patch.object(ml, "eager_attention_forward", self._attention):
            with guard:
                out = ml.eager_attention_forward(module, query, key, value, None)
        return out[0], guard

    def test_masked_unread_rows_do_not_change_the_output_and_used_unread_rows_break_it(self):
        g = torch.Generator().manual_seed(3)
        H, hd, n = 2, 4, 10                                    # 8 prefix + 2 suffix keys
        query = torch.randn(1, H, 2, hd, generator=g)
        key = torch.randn(1, H, n, hd, generator=g)
        value = torch.randn(1, H, n, hd, generator=g)
        keep = torch.tensor([True, False, True, False, False, True])
        zero_k, zero_v, nan_k, nan_v = key.clone(), value.clone(), key.clone(), value.clone()
        unread = [3, 5, 6]                                       # absolute positions of the masked visual rows 1, 3, 4
        for t in (zero_k, zero_v):
            t[:, :, unread] = 0.0
        for t in (nan_k, nan_v):
            t[:, :, unread] = float("nan")
        want, _ = self._call(zero_k, zero_v, keep, query)
        got, guard = self._call(nan_k, nan_v, keep, query)
        self.assertTrue(bool(torch.isfinite(got).all()))
        self.assertTrue(torch.equal(got, want))
        r = guard.result({0: keep})
        self.assertEqual((r["all_outputs_finite"], r["used_equals_selection"], r["used_rows_all_read"]), (True, True, True))
        self.assertEqual((r["unread_rows"], r["used_rows"]), (3, 3))
        # one row that the selection uses was not read: the output is not finite
        bad_k = nan_k.clone()
        bad_k[:, :, 2] = float("nan")
        got, guard = self._call(bad_k, nan_v, keep, query)
        self.assertFalse(bool(torch.isfinite(got).all()))
        r = guard.result({0: keep})
        self.assertEqual((r["all_outputs_finite"], r["used_rows_all_read"]), (False, False))
        # a fallback layer has no mask: every row must have been read
        got, guard = self._call(key, value, None, query)
        self.assertTrue(guard.result({0: torch.ones(6, dtype=torch.bool)})["used_equals_selection"])
        got, guard = self._call(nan_k, nan_v, None, query)
        self.assertFalse(guard.result({0: torch.ones(6, dtype=torch.bool)})["all_outputs_finite"])

    def test_guard_restores_the_attention_function(self):
        import transformers.models.llama.modeling_llama as ml
        before = ml.eager_attention_forward
        with hi.UnreadRowGuard(self.META):
            self.assertIsNot(ml.eager_attention_forward, before)
        self.assertIs(ml.eager_attention_forward, before)


if __name__ == "__main__":
    unittest.main()

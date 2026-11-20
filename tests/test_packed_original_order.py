"""CPU contracts for the original-order packed store (``row_order="original"``).

Contract: docs/mt_gqa_ours_order_contract.md.  On the 2-layer FP16 fixture of
test_llava_kv25 (separators at original positions 0 and v_num-1):

* the original-order store keeps, per layer, exactly the token set the
  importance layout stores (``order[:k_l]``), in ascending original index;
  every K/V row of a token is bit-identical between the two stores, the
  separator block and the byte table are identical, the total size is equal;
* its metadata holds no importance permutation and no ``visionzip_layout.pt``;
  the per-layer index lists are recovered from the two small tables;
* serving puts each row at its ORIGINAL cache position, separators at their
  original positions and masks everything else; the visible K/V equal the
  raster values; the read ranges, read count and bytes equal the importance
  store's, synchronously and through the layer stream;
* the Turn-1 persistence flow with ``row_order="original"`` takes no question:
  captures of different lengths give byte-identical stores;
* the default (importance) path is unchanged and tampered metadata fails closed.
"""
from __future__ import annotations

import copy
import inspect
import json
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mmimpress.layer_budget import budget_document, layer_keep_counts
from mmimpress.packed_serve import (PackedImageContext, PackedPrefixSelector, PackedReadTrace,
                                    PackedStreamSelector)
from mmimpress.packed_store import (IMPORTANCE_ORDER_KEYS, ORIGINAL_STORE_FORMAT, PACKED_NAME, STORE_FORMAT,
                                    layer_original_indices, original_order_plan, persist_captured_packed_prefix,
                                    validate_packed_meta, write_packed_store)
from mmimpress.prefix25 import seal_integrity
from mmimpress.serve import BIAS, IOCounter
from test_impress_fidelity import _FakeDecoderLayer
from test_packed_store import RATIOS, _stores
from test_visdial_turn1_piggyback_core import _GeometryRunner, _capture_provenance

CASES = (1, 63, 64, 65, 127, 129, 349, 2200)
ROW = 2 * 2 * 2          # fixture: heads 2, head_dim 2, fp16


def _both(root, n, alloc):
    """The importance store ``p`` of test_packed_store plus its original-order twin ``o``."""
    meta_a, rank, layers, u, p, doc, _ = _stores(root, n, alloc)
    extra = {"layout_source": "turn1_normal_inference_piggyback", "visual_kv_source": "turn1_captured_past_key_values",
             "layout_uses_dataset_question": False, "llm_used_for_layout_scoring": False, "calibration_questions": 0,
             # must be stripped by the writer
             "permutation_sha256": meta_a["permutation_sha256"], "physical_layout": "visionzip_image_only"}
    timing = {}
    o = write_packed_store(root / "o", layers, meta_a["v_token_start"], meta_a["v_token_num"], meta_a["prefix_input_ids"],
                           meta_a["newline_idx"], extra=extra, stored_to_original=meta_a["order"], separator_sidecar=True,
                           probe_heads=0, storage_policy="prefix25_packed", layer_budget=doc, timing_out=timing,
                           row_order="original")
    seal_integrity(root / "o")
    return meta_a, rank, layers, p, o, doc


def _runner(layers=2):
    return SimpleNamespace(layers=[_FakeDecoderLayer() for _ in range(layers)], model=SimpleNamespace(device=torch.device("cpu")))


class OriginalOrderPlan(unittest.TestCase):
    def test_plan_is_the_sorted_prefix_sets(self):
        order = [7, 2, 9, 4, 1, 8, 3, 0, 5, 6]
        counts = [3, 5, 3, 0, 8]
        indices, min_keep, per_layer = original_order_plan(order, counts)
        self.assertEqual(indices, sorted(order[:8]))
        self.assertEqual(per_layer, [sorted(order[:c]) for c in counts])
        self.assertEqual(dict(zip(indices, min_keep)), {7: 3, 2: 3, 9: 3, 4: 5, 1: 5, 8: 8, 3: 8, 0: 8})
        meta = {"layer_keep_counts": counts, "stored_original_indices": indices, "stored_min_keep_count": min_keep}
        self.assertEqual(layer_original_indices(meta), per_layer)
        with self.assertRaises(ValueError):
            original_order_plan(order[:4], counts)
        with self.assertRaises(ValueError):
            original_order_plan([1, 1, 2], [3])

    def test_only_order_and_counts_are_inputs(self):
        self.assertEqual(list(inspect.signature(original_order_plan).parameters), ["order", "counts"])


class OriginalOrderBytes(unittest.TestCase):
    def test_same_sets_same_token_bytes_same_separators_same_size(self):
        for alloc in ("U", "T"):
            for n in CASES:
                with self.subTest(alloc=alloc, n=n), tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    meta_a, rank, layers, p, o, doc = _both(root, n, alloc)
                    k = layer_keep_counts(RATIOS[alloc], n, num_layers=2)
                    validate_packed_meta(o); validate_packed_meta(p)
                    self.assertEqual(o["store_format"], ORIGINAL_STORE_FORMAT)
                    self.assertEqual(p["store_format"], STORE_FORMAT)
                    self.assertEqual(o["layer_keep_counts"], k)
                    rows = layer_original_indices(o)
                    for li in range(2):
                        self.assertEqual(rows[li], sorted(p["order"][:k[li]]))
                        self.assertEqual(set(rows[li]), set(rank[:k[li]]))
                    # identical byte table, separator block and size
                    for key in ("packed_layout", "packed_sep_offset", "packed_sep_bytes", "packed_total_bytes", "packed_row_bytes",
                                "bytes_visual_kv", "bytes_separator_sidecar", "newline_idx", "layer_keep_ratios", "layer_budget_sha256"):
                        self.assertEqual(o[key], p[key], key)
                    bo = (root / "o" / PACKED_NAME).read_bytes(); bp = (root / "p" / PACKED_NAME).read_bytes()
                    self.assertEqual(len(bo), len(bp))
                    self.assertEqual(bo[o["packed_sep_offset"]:], bp[p["packed_sep_offset"]:])
                    # every stored token: the same K and V bytes in both stores
                    for li in range(2):
                        for kind in ("k", "v"):
                            off = o["packed_layout"][li][f"{kind}_offset"]
                            by_token_o = {t: bo[off + j * ROW: off + (j + 1) * ROW] for j, t in enumerate(rows[li])}
                            by_token_p = {t: bp[off + j * ROW: off + (j + 1) * ROW] for j, t in enumerate(p["order"][:k[li]])}
                            self.assertEqual(by_token_o, by_token_p)
                            # and they are the captured raster values
                            src = layers[li][0 if kind == "k" else 1][0, :, meta_a["v_token_start"]:, :].permute(1, 0, 2).contiguous()
                            for t, blob in by_token_o.items():
                                self.assertEqual(blob, src[t].numpy().tobytes())
                    self.assertEqual((root / "o" / "sys_kv.pt").read_bytes(), (root / "p" / "sys_kv.pt").read_bytes())

    def test_metadata_holds_no_importance_permutation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            meta_a, rank, layers, p, o, doc = _both(root, 349, "T")
            on_disk = json.loads((root / "o" / "meta.json").read_text())
            self.assertEqual(on_disk, o)
            for key in IMPORTANCE_ORDER_KEYS:
                self.assertNotIn(key, on_disk)
            self.assertEqual(on_disk["row_order"], "original")
            self.assertIs(on_disk["reordered"], False)
            self.assertEqual(on_disk["physical_layout"], "original_order_layer_subsets")
            self.assertEqual(sorted(x.name for x in (root / "o").iterdir()), ["integrity.json", PACKED_NAME, "meta.json", "sys_kv.pt"])
            # the tables are smaller than one full permutation
            self.assertEqual(len(on_disk["stored_original_indices"]), max(o["layer_keep_counts"]))
            self.assertLess(len(on_disk["stored_original_indices"]), len(p["order"]))
            # the default writer is untouched: the importance store still records its permutation
            for key in ("order", "full_importance_permutation", "stored_row_to_original", "newline_stored"):
                self.assertIn(key, p)
            self.assertNotIn("row_order", p)

    def test_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            meta_a, rank, layers, p, o, doc = _both(root, 349, "T")
            k = o["layer_keep_counts"]; idx = o["stored_original_indices"]
            swapped = list(idx); swapped[0], swapped[1] = swapped[1], swapped[0]
            for field, value in [("store_format", STORE_FORMAT), ("row_order", "importance"), ("reordered", True),
                                 ("order", p["order"]), ("permutation_sha256", "0" * 64), ("full_importance_permutation", p["order"]),
                                 ("stored_original_indices", swapped), ("stored_original_indices", idx[:-1]),
                                 ("stored_original_indices", [0] + idx[1:]),                 # a separator position
                                 ("stored_min_keep_count", [max(k)] * len(idx)), ("stored_min_keep_count", [k[0] + 1] * len(idx)),
                                 ("layer_keep_counts", [k[0] + 1, k[1] - 1]), ("layer_budget_sha256", "0" * 64),
                                 ("packed_total_bytes", o["packed_total_bytes"] + 1), ("packed_layout", o["packed_layout"][::-1]),
                                 ("physical_layout", "visionzip_image_only"), ("payload_rows", k[0])]:
                bad = copy.deepcopy(o); bad[field] = value
                with self.subTest(field=field, value=str(value)[:30]), self.assertRaises(ValueError):
                    validate_packed_meta(bad)
            # an importance artifact next to an original-order store is refused at activation
            (root / "o" / "visionzip_layout.pt").write_bytes(b"x")
            (root / "o" / "integrity.json").unlink(); seal_integrity(root / "o")
            ctx = PackedImageContext(root / "o", "cpu", pin=False)
            with self.assertRaises(AssertionError):
                ctx.validate_packed_layout()
            ctx.close()
            with self.assertRaises(ValueError):
                write_packed_store(root / "x", layers, meta_a["v_token_start"], meta_a["v_token_num"], meta_a["prefix_input_ids"],
                                   meta_a["newline_idx"], stored_to_original=meta_a["order"], separator_sidecar=True, probe_heads=0,
                                   layer_budget=doc, row_order="raster")


class OriginalOrderServing(unittest.TestCase):
    def tearDown(self):
        BIAS.clear()

    def _prepare(self, root, name, runner):
        ctx = PackedImageContext(root / name, "cpu", pin=False)
        ctx.validate_packed_layout()
        cache = ctx.cache.new_request()
        counter = IOCounter()
        sel = PackedPrefixSelector(runner, ctx, .25, counter)
        with mock.patch("torch.cuda.synchronize"), PackedReadTrace(root / name) as trace:
            sel.prepare()
        bias = {li: BIAS[li].clone() for li in BIAS}
        BIAS.clear()
        return ctx, cache, sel, trace.summary(), bias

    def test_rows_land_on_original_positions_and_the_rest_is_masked(self):
        runner = _runner()
        for alloc in ("U", "T"):
            for n in CASES:
                with self.subTest(alloc=alloc, n=n), tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    meta_a, rank, layers, p, o, doc = _both(root, n, alloc)
                    k = layer_keep_counts(RATIOS[alloc], n, num_layers=2)
                    v0, vn = meta_a["v_token_start"], meta_a["v_token_num"]
                    sep = meta_a["newline_idx"]
                    ctx_o, cache_o, sel_o, so, bias_o = self._prepare(root, "o", runner)
                    ctx_p, cache_p, sel_p, sp, bias_p = self._prepare(root, "p", runner)
                    try:
                        # reads: one preadv of the whole file, identical to the importance store
                        self.assertEqual((so["preads"], so["preadv_calls"], so["pread_calls"], so["bytes"]),
                                         (sp["preads"], sp["preadv_calls"], sp["pread_calls"], sp["bytes"]))
                        self.assertEqual(so["preads"], 1)
                        self.assertEqual([(c["offset"], c["returned"]) for c in so["calls"]], [(c["offset"], c["returned"]) for c in sp["calls"]])
                        st_o, st_p = sel_o.stats(), sel_p.stats()
                        for key in ("k_per_layer", "keep_count_per_layer", "k_total", "layer_budget_sha256", "logical_kv_ratio",
                                    "planned_packed_read", "planned_normal_read_bytes_per_layer"):
                            self.assertEqual(st_o[key], st_p[key], key)
                        self.assertEqual(st_o["row_order"], "original"); self.assertEqual(st_p["row_order"], "importance")
                        self.assertEqual(st_o["selected_original_ids"], sorted(st_p["selected_original_ids"]))
                        self.assertIsNone(st_o["selected_stored_ids"])
                        self.assertEqual(st_p["selected_stored_ids"], list(range(max(k))))
                        for li in range(2):
                            keep = sorted(rank[:k[li]])
                            visible = (bias_o[li][0, 0, 0, v0:v0 + vn] == 0).nonzero().flatten().tolist()
                            self.assertEqual(visible, sorted(keep + sep))
                            self.assertTrue(bool((bias_o[li][0, 0, 0, :v0] == 0).all()))
                            for kind, tensors in (("keys", 0), ("values", 1)):
                                got = getattr(cache_o.layers[li], kind)
                                raster = layers[li][tensors].to(torch.float16).to(torch.bfloat16)
                                for t in keep + sep:           # visible positions hold the raster values
                                    self.assertTrue(torch.equal(got[0, :, v0 + t], raster[0, :, v0 + t]))
                                hidden = [t for t in range(vn) if t not in set(keep + sep)]
                                if hidden:                     # never written
                                    self.assertEqual(float(got[0, :, torch.tensor(hidden) + v0].abs().sum()), 0.0)
                                self.assertTrue(torch.equal(got[0, :, :v0], getattr(cache_p.layers[li], kind)[0, :, :v0]))
                                # the importance cache holds the same rows at their stored rank
                                ref = getattr(cache_p.layers[li], kind)
                                for stored, t in enumerate(rank[:k[li]]):
                                    self.assertTrue(torch.equal(got[0, :, v0 + t], ref[0, :, v0 + stored]))
                        with self.assertRaises(ValueError):
                            PackedPrefixSelector(runner, ctx_o, .25, directcopy=True)
                    finally:
                        ctx_o.close(); ctx_p.close()

    def test_layer_stream_equals_synchronous_and_the_importance_read_plan(self):
        runner = _runner()
        pool = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(pool.shutdown, wait=True)
        for alloc in ("U", "T"):
            for n in (1, 64, 129, 349):
                with self.subTest(alloc=alloc, n=n), tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    _both(root, n, alloc)
                    ctx_s, cache_s, sync, ss, bias_s = self._prepare(root, "o", runner)
                    ctx_p = PackedImageContext(root / "p", "cpu", pin=False)
                    ctx_a = PackedImageContext(root / "o", "cpu", pin=False)
                    self.assertEqual(ctx_a.reader.stream_plan(), ctx_p.reader.stream_plan())
                    threads = []
                    real = ctx_a.reader.read_range

                    def read_range(offset, length, counter=None):
                        threads.append(threading.get_ident())
                        return real(offset, length, counter)

                    ctx_a.reader.read_range = read_range
                    counter = IOCounter()
                    stream = PackedStreamSelector(runner, ctx_a, 0.25, counter, executor=pool)
                    with PackedReadTrace(root / "o") as trace:
                        stream.begin()
                        cache_a = ctx_a.cache.new_request()
                        with stream:
                            for layer in runner.layers:
                                layer(hidden_states=torch.zeros(1, 3, 4))
                            for layer in runner.layers:      # decode step: nothing more is read
                                layer(hidden_states=torch.zeros(1, 1, 4))
                    sa = trace.summary()
                    self.assertEqual(sorted(BIAS), sorted(bias_s))
                    for li in bias_s:
                        self.assertTrue(torch.equal(BIAS[li], bias_s[li]))
                        self.assertTrue(torch.equal(ctx_a.cache.k[li], ctx_s.cache.k[li]))
                        self.assertTrue(torch.equal(ctx_a.cache.v[li], ctx_s.cache.v[li]))
                    self.assertTrue(torch.equal(ctx_a.reader.buf, ctx_s.reader.buf))
                    nonzero = [(off, ln) for _, off, ln in ctx_p.reader.stream_plan() if ln]
                    self.assertEqual([(c["offset"], c["returned"]) for c in sa["calls"]], nonzero)
                    self.assertEqual(sa["bytes"], ss["bytes"]); self.assertEqual(sa["pread_calls"], 0)
                    self.assertEqual(counter.preads, len(nonzero))
                    main = threading.get_ident()
                    self.assertTrue(threads and all(t != main for t in threads))
                    a, s = stream.stats(), sync.stats()
                    for key in ("k_per_layer", "keep_count_per_layer", "selected_original_ids", "row_order", "store_format", "k_total"):
                        self.assertEqual(a[key], s[key], key)
                    self.assertEqual(a["read_wait"]["readahead"]["items_taken"], 3)
                    ctx_s.close(); ctx_a.close(); ctx_p.close()
                    del cache_s, cache_a
                    BIAS.clear()


class OriginalOrderPersistence(unittest.TestCase):
    """The Turn-1 flow: 7 visual tokens (separator at 6), system prefix 2."""

    SCORES = [0.2, 0.8, 0.8, 0.1, 0.6, 0.3, float("inf")]

    def _capture(self, sequence, seed):
        g = torch.Generator().manual_seed(1)          # the visual prefix does not depend on the question
        prefix = [(torch.randn(1, 2, 9, 4, generator=g), torch.randn(1, 2, 9, 4, generator=g)) for _ in range(2)]
        gq = torch.Generator().manual_seed(seed)      # question / answer rows differ per request
        return [(torch.cat([k, torch.randn(1, 2, sequence - 9, 4, generator=gq)], dim=2).to(torch.float16),
                 torch.cat([v, torch.randn(1, 2, sequence - 9, 4, generator=gq)], dim=2).to(torch.float16)) for k, v in prefix]

    def _persist(self, root, name, sequence, seed, row_order):
        doc = budget_document("T", [0.35, 0.15], model="fixture", source={}, num_layers=2)
        with mock.patch("mmimpress.piggyback.anyres_token_scores", side_effect=lambda *a, **k: torch.tensor(self.SCORES)):
            return persist_captured_packed_prefix(
                _GeometryRunner(), self._capture(sequence, seed), torch.arange(sequence).unsqueeze(0), [480, 640], torch.ones(2, 4),
                root / name, layer_budget=doc, image_id="img", capture_stats=_capture_provenance(),
                **({"row_order": row_order} if row_order else {}))

    def test_three_questions_give_the_same_store_and_no_importance_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            receipts = [self._persist(root, f"q{i}", seq, 10 + i, "original") for i, seq in enumerate((12, 15, 23))]
            ref = self._persist(root, "imp", 12, 10, None)
            blobs = [(root / f"q{i}" / PACKED_NAME).read_bytes() for i in range(3)]
            self.assertEqual(blobs[0], blobs[1]); self.assertEqual(blobs[0], blobs[2])
            metas = [json.loads((root / f"q{i}" / "meta.json").read_text()) for i in range(3)]
            for m in metas:
                validate_packed_meta(m)
                self.assertFalse((root / "q0" / "visionzip_layout.pt").exists())
                for key in IMPORTANCE_ORDER_KEYS:
                    self.assertNotIn(key, m)
                self.assertEqual(m["packed_sha256"], metas[0]["packed_sha256"])
                self.assertEqual(layer_original_indices(m), layer_original_indices(metas[0]))
                self.assertIs(m["layout_uses_dataset_question"], False)
                self.assertIs(m["llm_used_for_layout_scoring"], False)
                self.assertEqual(m["calibration_questions"], 0)
            imp = ref["meta"]
            self.assertTrue((root / "imp" / "visionzip_layout.pt").exists())       # default path unchanged
            self.assertEqual(imp["store_format"], STORE_FORMAT)
            self.assertIn("permutation_sha256", imp)
            counts = imp["layer_keep_counts"]
            self.assertEqual(layer_original_indices(metas[0]), [sorted(imp["order"][:c]) for c in counts])
            self.assertEqual(metas[0]["packed_total_bytes"], imp["packed_total_bytes"])
            # ties (0.8, 0.8) resolve to the lower original index, as in the importance order
            self.assertEqual(imp["order"][:2], [1, 2])
            self.assertIsNone(receipts[0]["hashes"]["permutation_sha256"])
            self.assertEqual(receipts[0]["timing_ms"]["layout_write_ms"], 0.0)
            self.assertEqual(sorted(receipts[0]["hashes"]["files_sha256"]), ["meta.json"])
            self.assertTrue(receipts[0]["integrity"]["ok"])
            self.assertEqual(sorted(ref["hashes"]["files_sha256"]), ["meta.json", "visionzip_layout.pt"])

    def test_option_is_rejected_outside_the_packed_visionzip_prefix(self):
        from mmimpress.piggyback import persist_captured_visual_prefix
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                persist_captured_visual_prefix(_GeometryRunner(), self._capture(12, 1), torch.arange(12), [480, 640], torch.ones(2, 4),
                                               Path(td) / "x", image_id="x", capture_stats=_capture_provenance(), row_order="original")
            with self.assertRaises(ValueError):
                self._persist(Path(td), "y", 12, 1, "raster")


class HarnessTests(unittest.TestCase):
    """scripts/202: arm table, arm order, and the on-disk comparison of the two packed stores."""

    @classmethod
    def setUpClass(cls):
        scripts = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        import llava15_support as S
        cls.E = S.load_script("_mtgqa_ours_order_eval_test", "202_eval_mt_gqa_ours_order.py")

    def test_arms_names_and_request_counts(self):
        E = self.E
        main = E.PHASE_METHODS["main"]
        self.assertEqual(main, ["recompute", "fullload_async", E.OURS_ASYNC, E.ORIG_ASYNC])
        self.assertEqual(E.PHASE_METHODS["dev"], main)
        self.assertEqual([E.ARMS[m]["display"] for m in main], ["ReComp", "FullLoad", "Ours(재배열)", "Ours(원래 순서)"])
        self.assertEqual(E.OURS_ASYNC, "ours_prefixstore_kv25_PYR-beta-raw_packed_async")      # the method id of the previous runs
        self.assertEqual(E.PHASE_METHODS["gate"], main[:3] + [E.ORIG, E.ORIG_ASYNC, E.ORIG_REPEAT])
        self.assertEqual(E.PHASE_METHODS["devgate"], [E.ORIG, E.ORIG_ASYNC, E.ORIG_REPEAT])
        self.assertEqual({E.kind_of(m) for m in (E.ORIG, E.ORIG_ASYNC, E.ORIG_REPEAT)}, {E.ORIG_KIND})
        self.assertEqual(E.PACKED_KINDS, {E.PACKED_KIND: "importance", E.ORIG_KIND: "original"})
        self.assertEqual(E.store_kinds(main), ["raster", E.PACKED_KIND, E.ORIG_KIND])
        self.assertEqual(4061 * 3 * len(main), 48732)
        self.assertNotIn("202_eval_mt_gqa_ours_order.py", E.REPORT_ONLY)
        self.assertNotIn("203_validate_mt_gqa_ours_order.py", E.REPORT_ONLY)
        self.assertIn("204_report_mt_gqa_ours_order.py", E.REPORT_ONLY)

    def test_arm_order_balances_position_and_predecessor_of_the_two_packed_arms(self):
        from collections import Counter
        E = self.E
        main = E.PHASE_METHODS["main"]; A, B = E.OURS_ASYNC, E.ORIG_ASYNC
        position = {m: Counter() for m in main}; before = {m: Counter() for m in main}
        for ordinal in range(8 * 5):
            order = E.method_order(main, ordinal)
            self.assertEqual(sorted(order), sorted(main))
            for i, m in enumerate(order):
                position[m][i] += 1; before[m][order[i - 1]] += 1      # cyclic: the first arm follows the previous turn's last
        self.assertEqual(position[A], position[B])
        self.assertEqual(set(position[A].values()), {10})
        self.assertEqual(before[A], Counter({"fullload_async": 20, B: 20}))
        self.assertEqual(before[B], Counter({"fullload_async": 20, A: 20}))
        for m in ("recompute", "fullload_async"):      # unchanged slots: the plain rotation of the previous runs
            self.assertEqual(set(position[m].values()), {10})
        for ordinal in range(4):                        # the first block is exactly the previous rotation
            self.assertEqual(E.method_order(main, ordinal), main[ordinal:] + main[:ordinal])
        gate = E.PHASE_METHODS["devgate"]               # without both packed arms: plain rotation
        for ordinal in range(9):
            r = ordinal % 3
            self.assertEqual(E.method_order(gate, ordinal), gate[r:] + gate[:r])

    def test_store_equivalence_reads_both_stores_back(self):
        E = self.E
        for alloc, n in (("U", 65), ("T", 349)):
            with self.subTest(alloc=alloc, n=n), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                meta_a, rank, layers, p, o, doc = _both(root, n, alloc)
                eq = E.store_equivalence(root / "p", root / "o")
                self.assertEqual(eq["status"], "PASS")
                self.assertEqual(eq["kv_rows_compared"], 2 * sum(o["layer_keep_counts"]))
                self.assertEqual(eq["kv_rows_different"], 0)
                self.assertEqual(eq["layer_sets_equal"], 2)
                self.assertTrue(eq["separator_block_equal"] and eq["payload_bytes_equal"] and eq["system_kv_equal"])
                self.assertEqual(eq["metadata_bytes"]["original"]["meta.json"], (root / "o" / "meta.json").stat().st_size)
                # one flipped byte in one K row of the original-order payload
                f = root / "o" / PACKED_NAME
                blob = bytearray(f.read_bytes()); blob[3] ^= 1; f.write_bytes(bytes(blob))
                bad = E.store_equivalence(root / "p", root / "o")
                self.assertEqual((bad["status"], bad["kv_rows_different"]), ("FAIL", 1))
                blob[3] ^= 1; blob[-1] ^= 1; f.write_bytes(bytes(blob))      # one flipped byte in the separator block
                bad = E.store_equivalence(root / "p", root / "o")
                self.assertEqual((bad["status"], bad["kv_rows_different"], bad["separator_block_equal"]), ("FAIL", 0, False))


if __name__ == "__main__":
    unittest.main()

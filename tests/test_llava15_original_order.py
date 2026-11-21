"""CPU contracts of the original row order on the LLaVA-1.5 packed path
(docs/llava15_mt_gqa_main_contract.md).

``row_order="original"`` in ``mmimpress.llava15.store`` / ``mmimpress.llava15.serve``:

  * the store holds, per layer, the SAME token set as the importance store
    (``order[:k_l]``, ties included) in ascending original token index; every
    stored token's K and V bytes, the byte table and the payload size are
    those of the importance store;
  * the metadata holds no importance permutation and the importance artifact
    ``visionzip_layout.pt`` is not written;
  * at request time each row lands on its ORIGINAL cache position and every
    position that was not stored is masked;
  * the asynchronous read issues the same ranges, in the same order, as the
    importance store's asynchronous read, and restores the same buffer, GPU
    rows and masks as the synchronous read of the same store;
  * the selection and layout do not depend on the question;
  * the default (importance) path is unchanged: same metadata keys, same
    statistics keys, same bytes.

The run harness (scripts 206 / 207): arms, phases, gate comparisons, resume
planning.
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(Path(__file__).resolve().parent), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from test_llava15_mtvqa_impress import IDS, V0, _layers, _pool, _stream_runner  # noqa: E402
from test_llava15_store import ROW, STATS, Runner, budgets  # noqa: E402

from mmimpress.cvpr25 import visionzip_repack_order  # noqa: E402
from mmimpress.llava15 import store as ST  # noqa: E402
from mmimpress.llava15.serve import (Llava15PackedContext, Llava15PrefixSelector, Llava15StreamSelector,  # noqa: E402
                                     PackedReadTrace)
from mmimpress.packed_store import IMPORTANCE_ORDER_KEYS, layer_original_indices, original_order_plan  # noqa: E402
from mmimpress.prefix25 import seal_integrity  # noqa: E402
from mmimpress.serve import BIAS  # noqa: E402
from mmimpress.store import IOCounter  # noqa: E402

VEC = "PYR-b20"
TOTAL = 75_497_472


def _persist(path, layers, scores, doc, row_order=None, ids=IDS, extra=None):
    kw = {} if row_order is None else {"row_order": row_order}
    receipt = ST.persist_captured_llava15_prefix(Runner(), layers, ids, scores, path, budget=doc, image_id="t",
                                                 capture_stats=STATS, image_input_sha256="0" * 64, extra_metadata=extra, **kw)
    seal_integrity(path)
    return receipt


class _Stores(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="llava15_origorder_test_"))
        cls.layers = _layers(777)
        cls.scores = torch.rand((1, 576), generator=torch.Generator().manual_seed(9))
        # ties: inside the largest layer set, across a layer boundary, and outside every set
        ranked = torch.argsort(cls.scores[0], descending=True, stable=True).tolist()
        cls.doc = budgets()[VEC]
        cls.counts = ST.keep_counts(cls.doc, 576)
        cls.scores[0, ranked[3]] = cls.scores[0, ranked[2]]
        boundary = cls.counts[5]
        cls.scores[0, ranked[boundary]] = cls.scores[0, ranked[boundary - 1]]
        cls.scores[0, ranked[500]] = cls.scores[0, ranked[499]]
        cls.order = visionzip_repack_order(cls.scores[0], [])
        cls.r_imp = _persist(cls.tmp / "imp", cls.layers, cls.scores, cls.doc)
        cls.r_org = _persist(cls.tmp / "org", cls.layers, cls.scores, cls.doc, "original")
        cls.m_imp, cls.m_org = cls.r_imp["meta"], cls.r_org["meta"]

    @classmethod
    def tearDownClass(cls):
        BIAS.clear()
        shutil.rmtree(cls.tmp)

    def tearDown(self):
        BIAS.clear()

    def _ctx(self, name):
        ctx = Llava15PackedContext(self.tmp / name, "cpu", pin=False, expected_budget=self.doc)
        ctx.validate_packed_layout()
        self.addCleanup(ctx.close)
        return ctx


class OriginalOrderBytes(_Stores):
    def test_same_sets_same_token_bytes_same_table_same_size(self):
        counts = self.counts
        self.assertEqual(self.m_imp["order"], self.order)
        self.assertEqual(self.m_org["layer_keep_counts"], counts)
        rows = layer_original_indices(self.m_org)
        for li, k in enumerate(counts):
            self.assertEqual(rows[li], sorted(self.order[:k]))                       # the set, ties included
        self.assertEqual(self.m_org["packed_layout"], self.m_imp["packed_layout"])
        self.assertEqual((self.m_org["packed_total_bytes"], self.m_org["packed_sep_bytes"]), (TOTAL, 0))
        a = np.fromfile(self.tmp / "imp" / "kv_packed.bin", dtype=np.uint8)
        b = np.fromfile(self.tmp / "org" / "kv_packed.bin", dtype=np.uint8)
        self.assertEqual((a.size, b.size), (TOTAL, TOTAL))
        self.assertFalse(np.array_equal(a, b))                                       # the row order did change
        for li, k in enumerate(counts):
            to_ascending = np.argsort(np.asarray(self.order[:k]), kind="stable")
            for kind, src in (("k", self.layers[li][0]), ("v", self.layers[li][1])):
                off = self.m_imp["packed_layout"][li][f"{kind}_offset"]
                ra = a[off:off + k * ROW].reshape(k, ROW)[to_ascending]
                rb = b[off:off + k * ROW].reshape(k, ROW)
                self.assertTrue(np.array_equal(ra, rb), (li, kind))                  # same token -> same bytes
                if li in (0, 13, 31):                                                # and they are the capture's bytes
                    ids = torch.tensor(rows[li]) + V0
                    want = src[0][:, ids].to(torch.float16).transpose(0, 1).contiguous().numpy().tobytes()
                    self.assertEqual(rb.tobytes(), want)
        self.assertTrue(torch.equal(torch.load(self.tmp / "imp" / "sys_kv.pt", weights_only=True)["k"],
                                    torch.load(self.tmp / "org" / "sys_kv.pt", weights_only=True)["k"]))

    def test_plan_inputs_are_the_order_and_the_counts_only(self):
        union, min_keep, per_layer = original_order_plan(self.order, self.counts)
        self.assertEqual(self.m_org["stored_original_indices"], union)
        self.assertEqual(self.m_org["stored_min_keep_count"], min_keep)
        self.assertEqual(union, sorted(self.order[:max(self.counts)]))
        self.assertEqual(per_layer, layer_original_indices(self.m_org))

    def test_metadata_holds_no_importance_permutation(self):
        m = self.m_org
        for key in IMPORTANCE_ORDER_KEYS:
            self.assertNotIn(key, m)
        self.assertEqual((m["row_order"], m["reordered"], m["store_format"], m["format"], m["physical_layout"]),
                         ("original", False, ST.ORIGINAL_STORE_FORMAT, ST.ORIGINAL_META_FORMAT, "original_order_layer_subsets"))
        self.assertFalse((self.tmp / "org" / "visionzip_layout.pt").exists())
        self.assertTrue((self.tmp / "imp" / "visionzip_layout.pt").exists())
        self.assertNotIn("permutation_sha256", self.r_org["hashes"])
        self.assertEqual(sorted(self.r_org["hashes"]["files_sha256"]), ["meta.json"])
        self.assertIn("row_plan_ms", self.r_org["timing_ms"])
        ST.validate_llava15_packed_meta(m, self.doc)
        ST.validate_llava15_packed_original_meta(m, self.doc)

    def test_default_path_is_unchanged(self):
        m = self.m_imp
        self.assertNotIn("row_order", m)
        self.assertNotIn("stored_original_indices", m)
        self.assertEqual((m["store_format"], m["format"], m["physical_layout"], m["reordered"]),
                         (ST.STORE_FORMAT, ST.META_FORMAT, "visionzip_image_only", True))
        self.assertNotIn("row_plan_ms", self.r_imp["timing_ms"])
        self.assertEqual(sorted(self.r_imp["hashes"]["files_sha256"]), ["meta.json", "visionzip_layout.pt"])
        explicit = _persist(self.tmp / "imp_explicit", self.layers, self.scores, self.doc, "importance")
        self.assertEqual(explicit["meta"]["packed_sha256"], m["packed_sha256"])
        self.assertEqual(set(explicit["meta"]), set(m))
        shutil.rmtree(self.tmp / "imp_explicit")

    def test_fail_closed(self):
        m = self.m_org
        bad_idx = copy.deepcopy(m); bad_idx["stored_original_indices"][0], bad_idx["stored_original_indices"][1] = (
            bad_idx["stored_original_indices"][1], bad_idx["stored_original_indices"][0])
        bad_min = copy.deepcopy(m); bad_min["stored_min_keep_count"][0] = 1
        bad_cnt = copy.deepcopy(m); bad_cnt["layer_keep_counts"][0] += 1
        cases = [bad_idx, bad_min, bad_cnt, dict(m, order=self.order), dict(m, permutation_sha256="0" * 64), dict(m, row_order="importance"),
                 dict(m, reordered=True), dict(m, store_format=ST.STORE_FORMAT), dict(m, physical_layout="visionzip_image_only"),
                 dict(m, newline_idx=[575]), dict(m, packed_total_bytes=1), dict(m, layer_budget_total_ratio=0.5),
                 {k: v for k, v in m.items() if k != "stored_min_keep_count"}]
        for i, case in enumerate(cases):
            with self.assertRaises(ValueError, msg=i):
                ST.validate_llava15_packed_meta(case)
        with self.assertRaises(ValueError):
            ST.validate_llava15_packed_meta(m, budgets()["U25"])
        with self.assertRaises(ValueError):
            ST.validate_llava15_packed_meta(dict(self.m_imp, row_order="original"))      # importance store relabelled
        with self.assertRaises(ValueError):
            ST.write_llava15_packed_store(self.tmp / "w", self.layers, V0, 576, [], self.order, budget=self.doc, row_order="raster")
        with self.assertRaises(ValueError):
            ST.persist_captured_llava15_prefix(Runner(), self.layers, IDS, self.scores, self.tmp / "p", budget=self.doc, image_id="t",
                                               capture_stats=STATS, row_order="raster")
        with self.assertRaises(ValueError):      # extra metadata cannot smuggle the permutation back in
            ST.persist_captured_llava15_prefix(Runner(), self.layers, IDS, self.scores, self.tmp / "p", budget=self.doc, image_id="t",
                                               capture_stats=STATS, row_order="original", extra_metadata={"order": self.order})
        self.assertFalse((self.tmp / "p").exists())
        d = self.tmp / "with_artifact"
        shutil.copytree(self.tmp / "org", d)
        shutil.copy2(self.tmp / "imp" / "visionzip_layout.pt", d / "visionzip_layout.pt")
        (d / "integrity.json").unlink(); seal_integrity(d)
        ctx = Llava15PackedContext(d, "cpu", pin=False)
        with self.assertRaises(AssertionError):
            ctx.validate_packed_layout()
        ctx.close(); shutil.rmtree(d)


class OriginalOrderServing(_Stores):
    def test_rows_land_on_original_positions_and_the_rest_is_masked(self):
        counts = self.counts
        rows = layer_original_indices(self.m_org)
        ctx = self._ctx("org")
        self.assertEqual(ctx.row_order, "original")
        cache = ctx.cache.new_request()
        sel = Llava15PrefixSelector(Runner(), ctx)
        with mock.patch("torch.cuda.synchronize"), PackedReadTrace(self.tmp / "org") as trace:
            sel.prepare()
        s = trace.summary()
        self.assertEqual((s["preadv_calls"], s["pread_calls"], s["bytes"]), (1, 0, TOTAL))
        for li in range(32):
            k = counts[li]
            stored = torch.zeros(576, dtype=torch.bool); stored[torch.tensor(rows[li])] = True
            b = BIAS[li][0, 0, 0]
            self.assertTrue(bool((b[:V0] == 0).all()))
            self.assertTrue(torch.equal(b[V0:] == 0, stored))                        # visible == stored, at the original index
            self.assertTrue(bool((b[V0:][~stored] == torch.finfo(b.dtype).min).all()))
            for name, src in (("keys", self.layers[li][0]), ("values", self.layers[li][1])):
                got = getattr(cache.layers[li], name)[0, :, V0:V0 + 576]
                want = src[0][:, V0:V0 + 576].to(torch.float16).to(torch.bfloat16)
                self.assertTrue(torch.equal(got[:, stored], want[:, stored]))        # the token's own K/V at its own position
                self.assertTrue(bool((got[:, ~stored] == 0).all()))                  # nothing else was written
            self.assertEqual(int(stored.sum()), k)
        st = sel.stats()
        self.assertEqual((st["row_order"], st["selected_stored_ids"], st["store_format"]), ("original", None, ST.ORIGINAL_STORE_FORMAT))
        self.assertEqual(st["selected_original_ids"], sorted(self.order[:max(counts)]))
        self.assertEqual((st["k_per_layer"], st["keep_count_per_layer"], st["read_calls"]), (counts, counts, 1))

    def test_same_tokens_visible_as_the_importance_store(self):
        """Layer by layer, both stores expose the same original tokens with the same K/V values."""
        seen = {}
        for name in ("imp", "org"):
            ctx = self._ctx(name)
            cache = ctx.cache.new_request()
            sel = Llava15PrefixSelector(Runner(), ctx)
            with mock.patch("torch.cuda.synchronize"):
                sel.prepare()
            per_layer = []
            for li in (0, 7, 31):
                visible = (BIAS[li][0, 0, 0, V0:] == 0).nonzero(as_tuple=True)[0]
                original = visible if name == "org" else torch.tensor(self.order)[visible]
                keys = cache.layers[li].keys[0, :, V0 + visible]
                srt = torch.argsort(original)
                per_layer.append((original[srt].tolist(), keys[:, srt].clone()))
            seen[name] = per_layer
            self.assertEqual("row_order" in sel.stats(), name == "org")              # default statistics keep their key set
            BIAS.clear()
        for (ia, ka), (ib, kb) in zip(seen["imp"], seen["org"]):
            self.assertEqual(ia, ib)
            self.assertTrue(torch.equal(ka, kb))

    def test_layer_stream_equals_synchronous_and_the_importance_read_plan(self):
        runner = _stream_runner()
        ctx_s = self._ctx("org")
        cache_s = ctx_s.cache.new_request()
        sync = Llava15PrefixSelector(runner, ctx_s, IOCounter())
        with mock.patch("torch.cuda.synchronize", return_value=None):
            sync.prepare()
        bias_s = {li: BIAS[li].clone() for li in BIAS}
        BIAS.clear()

        traces = {}
        for name in ("org", "imp"):
            ctx = self._ctx(name)
            ctx.reader.buf.zero_()
            threads = []
            real = ctx.reader.read_range

            def read_range(offset, length, counter=None, real=real, threads=threads):
                threads.append(threading.get_ident())
                return real(offset, length, counter)

            ctx.reader.read_range = read_range
            stream = Llava15StreamSelector(runner, ctx, IOCounter(), executor=_pool(self))
            self.assertEqual(threads, [])                                            # the buffer exists, nothing read before begin()
            self.assertEqual(int(ctx.reader.buf.numel()), TOTAL)
            hooks_before = [len(l._forward_pre_hooks) for l in runner.layers]
            with PackedReadTrace(self.tmp / name) as trace:
                stream.begin()
                cache = ctx.cache.new_request()
                with stream:
                    for layer in runner.layers:
                        layer(hidden_states=torch.zeros(1, 3, 4))
                    for layer in runner.layers:                                      # a decode step loads nothing
                        layer(hidden_states=torch.zeros(1, 1, 4))
            self.assertEqual([len(l._forward_pre_hooks) for l in runner.layers], hooks_before)
            main = threading.get_ident()
            self.assertTrue(len(threads) == 32 and all(t != main for t in threads))
            traces[name] = [(c["syscall"], c["offset"], c["requested"], c["returned"]) for c in trace.summary()["calls"]]
            st = stream.stats()
            self.assertEqual((st["read_calls"], st["readahead_policy"]), (32, "layer_stream"))
            if name == "org":
                for li in range(32):
                    self.assertTrue(torch.equal(BIAS[li], bias_s[li]))
                    self.assertTrue(torch.equal(ctx.cache.k[li], ctx_s.cache.k[li]))
                    self.assertTrue(torch.equal(ctx.cache.v[li], ctx_s.cache.v[li]))
                self.assertTrue(torch.equal(ctx.reader.buf, ctx_s.reader.buf))
                self.assertEqual(st["row_order"], "original")
                for key in ("k_per_layer", "keep_count_per_layer", "selected_original_ids", "store_format", "k_total"):
                    self.assertEqual(st[key], sync.stats()[key], key)
            del cache
            BIAS.clear()
        plan, off = [], 0
        for k in self.counts:
            plan.append(("preadv", off, 2 * k * ROW, 2 * k * ROW)); off += 2 * k * ROW
        self.assertEqual(traces["org"], plan)                                        # layer order, [K|V] per layer, tiles the file
        self.assertEqual(traces["org"], traces["imp"])                               # same ranges, order and call count as before
        del cache_s


class QuestionIndependence(unittest.TestCase):
    def test_three_questions_give_the_same_selection_and_layout(self):
        """Different prompts change the captured K/V (and the ids after the image), never what is stored where."""
        tmp = Path(tempfile.mkdtemp(prefix="llava15_origorder_q_"))
        try:
            scores = torch.rand((1, 576), generator=torch.Generator().manual_seed(21))
            doc = budgets()[VEC]
            metas = []
            for i, tail in enumerate(([9, 9], [7, 7, 7, 7], [8])):
                ids = torch.tensor([1, 2, 3, 4, 5] + [32000] * 576 + tail)
                g = torch.Generator().manual_seed(100 + i)
                layers = [(torch.randn((1, 32, V0 + 576 + len(tail), 128), generator=g).to(torch.bfloat16),
                           torch.randn((1, 32, V0 + 576 + len(tail), 128), generator=g).to(torch.bfloat16)) for _ in range(32)]
                r = _persist(tmp / f"q{i}", layers, scores, doc, "original", ids=ids, extra={"source_dialog_id": f"d{i}"})
                metas.append(r["meta"])
                self.assertEqual(sorted(p.name for p in (tmp / f"q{i}").iterdir()), ["integrity.json", "kv_packed.bin", "meta.json", "sys_kv.pt"])
            ref = metas[0]
            for m in metas[1:]:
                self.assertEqual(layer_original_indices(m), layer_original_indices(ref))
                self.assertEqual(m["stored_original_indices"], ref["stored_original_indices"])
                self.assertEqual(m["stored_min_keep_count"], ref["stored_min_keep_count"])
                self.assertEqual(m["packed_layout"], ref["packed_layout"])
                self.assertNotEqual(m["packed_sha256"], ref["packed_sha256"])          # payload follows the capture
                differing = {k for k in set(m) | set(ref) if m.get(k) != ref.get(k)}
                self.assertEqual(differing, {"packed_sha256", "packed_writer_timing_ms", "source_dialog_id"})
            for m in metas:
                self.assertIs(m["layout_uses_dataset_question"], False)
                self.assertIs(m["llm_used_for_layout_scoring"], False)
                self.assertEqual(m["calibration_questions"], 0)
        finally:
            shutil.rmtree(tmp)


# ================================================================== harness
import llava15_support as S  # noqa: E402

H = S.load_script("_l15_mtgqa_main_eval", "206_eval_llava15_mt_gqa_main.py")
G = S.load_script("_l15_mtgqa_main_validate", "207_validate_llava15_mt_gqa_main.py")


def _row(method, dialog, turn, tokens=(5, 2), prompt="p", **extra):
    return dict({"method_id": method, "dialog_id": dialog, "turn_id": turn, "image_id": "img", "prompt": prompt, "prediction": "x", "score": 1.0,
                 "ttft_ms": 10.0, "ssd_read_bytes": 0, "probe_read_bytes": 0, "k_read_bytes": 0, "v_read_bytes": 0, "packed_read_bytes": 0,
                 "result": {"generated_token_ids": list(tokens)}}, **extra)


class HarnessTests(unittest.TestCase):
    def test_arms_names_and_request_counts(self):
        self.assertEqual([H.ARMS[m]["display"] for m in H.MAIN_METHODS], ["ReComp", "FullLoad", "IMPRESS", "Ours"])
        self.assertEqual(H.PHASE_METHODS["main"], H.MAIN_METHODS)
        self.assertEqual(H.PHASE_METHODS["dev"], H.MAIN_METHODS + [H.REORDER_ASYNC])
        self.assertEqual(H.EXPECTED_REQUESTS, {"gate": 1470, "devgate": 7938, "dev": 5670, "main": 48732})
        self.assertEqual({ph: 3 * len(H.PHASE_METHODS[ph]) for ph in H.PHASES}, {"gate": 30, "devgate": 21, "dev": 15, "main": 12})
        # every stored main arm reads asynchronously; Ours is the original-order store, the DEV reference the importance-order store
        self.assertTrue(all(H.ARMS[m]["async"] for m in H.MAIN_METHODS[1:]))
        self.assertEqual((H.kind_of(H.OURS_ASYNC), H.PACKED_KINDS[H.ORIG_KIND]), (H.ORIG_KIND, "original"))
        self.assertEqual((H.kind_of(H.REORDER_ASYNC), H.PACKED_KINDS[H.PACKED_KIND]), (H.PACKED_KIND, "importance"))
        self.assertEqual({H.kind_of(m) for m in (H.OURS, H.OURS_ASYNC, H.OURS_REPEAT)}, {H.ORIG_KIND})
        self.assertEqual({H.kind_of(m) for m in (H.NEW, H.NEW_ASYNC, H.NEW_R100, H.FULL_REORDERED)}, {H.NEW_KIND})
        self.assertEqual(H.REORDER_ASYNC, "llava15_kv_PYR-b20_packed_async")              # the method id of the 2026-11-18 run
        for ph in ("gate", "devgate"):                                                    # every gate comparison has both of its arms
            for a, b in ((H.OURS, H.OURS_ASYNC), (H.OURS_ASYNC, H.OURS_REPEAT), (H.NEW_R100, H.FULL_REORDERED), (H.NEW, H.NEW_ASYNC)):
                self.assertIn(a, H.PHASE_METHODS[ph]); self.assertIn(b, H.PHASE_METHODS[ph])
        self.assertEqual(G.REQUIRED, ("tests", "gate", "qindep_gate", "devgate", "qindep_devgate", "dev"))

    def test_fullload_and_impress_are_the_20261118_functions_and_settings(self):
        E = H.E
        self.assertIs(H.normal_request, E.normal_request)
        self.assertIs(H.build_official_layout, E.build_official_layout)
        self.assertIs(H.load_servers, E.load_servers)
        self.assertEqual((H.SERVER_ARGS, H.OFFICIAL), (E.SERVER_ARGS, E.OFFICIAL))
        self.assertEqual((H.OFFICIAL["probe_head_ids"], H.OFFICIAL["sim_threshold"], H.OFFICIAL["ratio"], H.OFFICIAL["chunk_size"], H.OFFICIAL["prefetch_async"]),
                         ([0, 1, 2], 0.3, 0.25, 64, "next_layer"))
        for m in ("recompute", "fullload_async", H.NEW_ASYNC):
            self.assertEqual({k: v for k, v in H.ARMS[m].items() if k != "display"}, {k: v for k, v in E.ARMS[m].items() if k != "display"})
        self.assertEqual((H.VECTOR, H.PACKED_BYTES), ("PYR-b20", 75_497_472))

    def test_freeze_check_ignores_new_files_and_report_scripts_only(self):
        frozen = {"mmimpress/llava15/store.py": "a", "scripts/206_eval_llava15_mt_gqa_main.py": "b", "scripts/180_report_llava15_mt_gqa.py": "c",
                  "tests/test_llava15_store.py": "d"}
        with mock.patch.object(H, "source_hashes", return_value=dict(frozen, **{"scripts/999_new_tool.py": "x", "tests/test_new.py": "y"})):
            self.assertEqual(H.freeze_violations(frozen), [])                              # files created later are not compared
        with mock.patch.object(H, "source_hashes", return_value=dict(frozen, **{"scripts/180_report_llava15_mt_gqa.py": "changed"})):
            self.assertEqual(H.freeze_violations(frozen), [])                              # a report script only reads rows
        with mock.patch.object(H, "source_hashes", return_value=dict(frozen, **{"mmimpress/llava15/store.py": "changed"})):
            self.assertEqual(H.freeze_violations(frozen), ["mmimpress/llava15/store.py"])
        gone = {k: v for k, v in frozen.items() if k != "tests/test_llava15_store.py"}
        with mock.patch.object(H, "source_hashes", return_value=gone):
            self.assertEqual(H.freeze_violations(frozen), ["tests/test_llava15_store.py"])
        self.assertEqual(H.freeze_violations(H.source_hashes()), [])

    def test_rotation_is_cyclic(self):
        m = H.MAIN_METHODS
        self.assertEqual([H.method_order(m, i)[0] for i in range(5)], [m[0], m[1], m[2], m[3], m[0]])
        self.assertEqual(sorted(H.method_order(m, 7)), sorted(m))

    def test_gate_comparison(self):
        keys = [("d1", 1), ("d1", 2)]
        rows = {("a", "d1", 1): _row("a", "d1", 1), ("b", "d1", 1): _row("b", "d1", 1),
                ("a", "d1", 2): _row("a", "d1", 2, ssd_read_bytes=10), ("b", "d1", 2): _row("b", "d1", 2, ssd_read_bytes=10)}
        self.assertEqual(G.compare(rows, "a", "b", keys, ("ssd_read_bytes",))["status"], "PASS")
        rows[("b", "d1", 2)] = _row("b", "d1", 2, tokens=(5, 3), ssd_read_bytes=10)
        out = G.compare(rows, "a", "b", keys, ("ssd_read_bytes",))
        self.assertEqual((out["status"], out["tokens_differ"], out["tokens_differ_same_prompt"]), ("FAIL", 1, 1))
        rows[("b", "d1", 2)] = _row("b", "d1", 2, ssd_read_bytes=11)
        out = G.compare(rows, "a", "b", keys, ("ssd_read_bytes",))
        self.assertEqual((out["status"], out["tokens_differ"], out["fields_differ"]["ssd_read_bytes"]), ("FAIL", 0, 1))
        del rows[("b", "d1", 2)]
        self.assertEqual(G.compare(rows, "a", "b", keys)["status"], "FAIL")                # a missing request is never a pass

    def test_store_equivalence_reads_both_stores_back(self):
        tmp = Path(tempfile.mkdtemp(prefix="llava15_origorder_eq_"))
        try:
            layers = _layers(31); scores = torch.rand((1, 576), generator=torch.Generator().manual_seed(3)); doc = budgets()[VEC]
            _persist(tmp / "imp", layers, scores, doc); _persist(tmp / "org", layers, scores, doc, "original")
            eq = H.store_equivalence(tmp / "imp", tmp / "org")
            self.assertEqual((eq["status"], eq["layer_sets_equal"], eq["kv_rows_compared"], eq["kv_rows_different"]), ("PASS", 32, 2 * 4608, 0))
            self.assertEqual((eq["payload_bytes"], eq["original_payload_bytes"]), (TOTAL, TOTAL))
            with (tmp / "org" / "kv_packed.bin").open("r+b") as f:                         # one flipped byte in one row
                f.seek(ROW * 3 + 7); b = f.read(1); f.seek(ROW * 3 + 7); f.write(bytes([b[0] ^ 1]))
            bad = H.store_equivalence(tmp / "imp", tmp / "org")
            self.assertEqual((bad["status"], bad["kv_rows_different"]), ("FAIL", 1))
        finally:
            shutil.rmtree(tmp)


class ResumeTests(unittest.TestCase):
    """Dialogue-level resume planning (no model, no GPU)."""
    METHODS = ["recompute", H.OURS_ASYNC]

    def setUp(self):
        self.run_dir = Path(tempfile.mkdtemp(prefix="llava15_main_resume_"))
        self.addCleanup(shutil.rmtree, self.run_dir)
        self.dialogs = [{"dialog_id": f"d{i}", "image_id": "img"} for i in range(3)]

    def _attempt(self, rows, store=True, name="attempt_0001"):
        art = self.run_dir / "main" / "images" / "img" / name
        art.mkdir(parents=True)
        with (art / "raw.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        scratch = self.run_dir / "scratch" / "main" / "img" / name
        if store:
            d = scratch / H.ORIG_KIND
            d.mkdir(parents=True)
            (d / "kv_packed.bin").write_bytes(b"payload"); (d / "meta.json").write_text("{}"); (d / "integrity.json").write_text("{}")
            hashes = H.file_hashes(d)
            seal = hashes.pop("integrity.json")
            (art / f"{H.ORIG_KIND}_hashes.json").write_text(json.dumps(hashes))
            (art / f"{H.ORIG_KIND}_meta.json").write_text("{}")
            (art / f"{H.ORIG_KIND}_persistence.json").write_text(json.dumps({"files": {"integrity.json": {"sha256": seal}}}))
        return art, scratch

    def _rows(self, n_dialogs, partial=0):
        rows = [_row(m, f"d{i}", t) for i in range(n_dialogs) for t in (1, 2, 3) for m in self.METHODS]
        return rows + [_row(self.METHODS[0], f"d{n_dialogs}", 1)] * bool(partial)

    def test_no_attempt_starts_fresh(self):
        plan = H.plan_image(self.run_dir, "main", "img", self.dialogs, self.METHODS)
        self.assertEqual((plan["mode"], plan["attempt"], plan["resumed"]), ("fresh", "attempt_0001", False))

    def test_complete_dialogues_are_kept_and_the_interrupted_one_restarts(self):
        self._attempt(self._rows(1, partial=1))
        plan = H.plan_image(self.run_dir, "main", "img", self.dialogs, self.METHODS)
        self.assertEqual((plan["mode"], plan["attempt"], plan["n_complete"]), ("continue", "attempt_0001", 1))
        self.assertEqual((len(plan["kept_rows"]), len(plan["dropped_rows"])), (6, 1))
        self.assertEqual((plan["interrupted_dialogue"], plan["restart_dialogues"]), ("d1", ["d1", "d2"]))
        self.assertTrue(all(r["dialog_id"] == "d0" for r in plan["kept_rows"]))

    def test_interrupted_first_dialogue_restarts_the_image(self):
        self._attempt(self._rows(0, partial=1))
        plan = H.plan_image(self.run_dir, "main", "img", self.dialogs, self.METHODS)
        self.assertEqual((plan["mode"], plan["attempt"], plan["discard_attempt"]), ("fresh", "attempt_0002", "attempt_0001"))
        self.assertEqual(plan["restart_dialogues"], ["d0", "d1", "d2"])

    def test_unverifiable_store_restarts_the_image(self):
        art, scratch = self._attempt(self._rows(2))
        (scratch / H.ORIG_KIND / "kv_packed.bin").write_bytes(b"changed")
        plan = H.plan_image(self.run_dir, "main", "img", self.dialogs, self.METHODS)
        self.assertEqual((plan["mode"], plan["attempt"]), ("fresh", "attempt_0002"))
        self.assertIn("differ", plan["store_check"])
        self.assertEqual(H.remove_scratch(self.run_dir, Path(plan["discard_scratch"])), len(b"changed") + 4)
        self.assertFalse(scratch.exists())
        with self.assertRaises(ValueError):                                              # never outside this run's scratch
            H.remove_scratch(self.run_dir, self.run_dir / "main")

    def test_missing_store_or_leftover_staging_restarts_the_image(self):
        self._attempt(self._rows(1), store=False)
        self.assertEqual(H.plan_image(self.run_dir, "main", "img", self.dialogs, self.METHODS)["mode"], "fresh")
        shutil.rmtree(self.run_dir / "main")
        art, scratch = self._attempt(self._rows(1))
        (scratch / ".x.staging-1").mkdir()
        self.assertEqual(H.plan_image(self.run_dir, "main", "img", self.dialogs, self.METHODS)["mode"], "fresh")

    def test_torn_last_line_and_all_dialogues_complete(self):
        art, _ = self._attempt(self._rows(3))
        with (art / "raw.jsonl").open("a") as f:
            f.write('{"method_id": "recompute", "dialog')                                 # power loss in the middle of a line
        rows, torn = H.tolerant_rows(art / "raw.jsonl")
        self.assertEqual((len(rows), torn), (18, 1))
        plan = H.plan_image(self.run_dir, "main", "img", self.dialogs, self.METHODS)
        self.assertEqual((plan["mode"], plan["n_complete"], plan["restart_dialogues"], plan["interrupted_dialogue"]), ("continue", 3, [], None))

    def test_a_dialogue_missing_one_arm_turn_is_not_complete(self):
        rows = [r for r in self._rows(2) if not (r["dialog_id"] == "d0" and r["turn_id"] == 3 and r["method_id"] == H.OURS_ASYNC)]
        n, kept, dropped = H.complete_prefix(rows, self.dialogs, self.METHODS)
        self.assertEqual((n, len(kept), len(dropped)), (0, 0, 11))                         # later rows cannot be complete either


if __name__ == "__main__":
    unittest.main()

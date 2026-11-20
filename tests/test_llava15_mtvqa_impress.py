"""CPU contracts of the LLaVA-1.5 x MT-VQA IMPRESS run (docs/llava15_mtvqa_impress_v2_contract.md).

Pins what this task adds to the LLaVA-1.5 path:

  * ``mmimpress.llava15.raster``: the full raster store (FullLoad / IMPRESS):
    bytes equal the capture, empty separator sidecar, no hidden-state file,
    served by the unchanged ``ImageContext``; the turn-1 importance repack of
    ``mmimpress.impress_mt`` works on a store without separators;
  * ``Llava15StreamSelector`` / ``request_llava15_packed(readahead=...)``: the
    asynchronous packed read restores the same pinned buffer, GPU rows and
    masks as the synchronous single ``preadv``, reads on the worker, tiles the
    file in layer order with 32 ``preadv`` calls; the synchronous default is
    untouched;
  * the run harness (scripts 193 / 194): arms, phases, logit evidence and the
    step-4 comparison of ``impress_r100`` with the FullLoad loader on the same
    reordered store (plus the reordered-vs-raster reference count).
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(Path(__file__).resolve().parent), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from test_impress_fidelity import _FakeDecoderLayer  # noqa: E402
from test_llava15_store import ROW, STATS, Runner, budgets  # noqa: E402

import llava15_support as S  # noqa: E402
from mmimpress import impress_mt  # noqa: E402
from mmimpress.llava15 import raster as RS  # noqa: E402
from mmimpress.llava15 import store as ST  # noqa: E402
from mmimpress.llava15.serve import (  # noqa: E402
    READAHEAD_POLICIES, Llava15PackedContext, Llava15PrefixSelector, Llava15StreamSelector, PackedReadTrace,
    request_llava15_packed)
from mmimpress.prefix25 import seal_integrity  # noqa: E402
from mmimpress.serve import BIAS, ImageContext, LayerSelector  # noqa: E402
from mmimpress.store import IOCounter, load_meta  # noqa: E402

E = S.load_script("_l15_mtvqa_impress_eval", "193_eval_llava15_mtvqa_impress.py")
G = S.load_script("_l15_mtvqa_impress_validate", "194_validate_llava15_mtvqa_impress.py")

V0 = 5
PROBE = 3


def _layers(seed=1234):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn((1, 32, V0 + 576 + 2, 128), generator=g).to(torch.bfloat16),
             torch.randn((1, 32, V0 + 576 + 2, 128), generator=g).to(torch.bfloat16)) for _ in range(32)]


IDS = torch.tensor([1, 2, 3, 4, 5] + [32000] * 576 + [9, 9])


def _pool(test):
    pool = ThreadPoolExecutor(max_workers=1)
    pool.submit(lambda: None).result()
    test.addCleanup(pool.shutdown, True)
    return pool


def _stream_runner():
    return SimpleNamespace(layers=[_FakeDecoderLayer() for _ in range(32)], model=SimpleNamespace(device=torch.device("cpu")))


# =============================================================== raster store
class RasterStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="llava15_raster_test_"))
        cls.layers = _layers()
        cls.receipt = RS.persist_captured_llava15_raster_prefix(
            Runner(), cls.layers, IDS, cls.tmp / "raster", image_id="t", probe_heads=PROBE, chunk_size=64,
            image_input_sha256="0" * 64, extra_metadata={"dataset": "unit"})
        seal_integrity(cls.tmp / "raster")
        cls.meta = load_meta(cls.tmp / "raster")

    @classmethod
    def tearDownClass(cls):
        BIAS.clear()
        shutil.rmtree(cls.tmp)

    def _block(self, li, kind):
        src = self.layers[li][0 if kind == "k" else 1]
        return src[0][:, V0:V0 + 576].to(torch.float16).transpose(0, 1).contiguous()      # (576, 32, 128)

    def test_files_are_the_capture_in_raster_order(self):
        m, root = self.meta, self.tmp / "raster"
        self.assertEqual((m["v_token_num"], m["n_spatial"], m["newline_idx"], m["probe_heads"], m["chunk_size"], m["n_chunks_per_layer"]),
                         (576, 576, [], PROBE, 64, 9))
        self.assertEqual((m["reordered"], m["physical_layout"], m["storage_policy"], m["dataset"]), (False, "raster", "full", "unit"))
        self.assertNotIn("order", m)
        self.assertEqual(m["bytes_visual_kv"], 301_989_888)
        self.assertEqual(m["bytes_probe_sidecar"], 32 * 576 * PROBE * 128 * 2)
        self.assertEqual((root / "sep_kv.bin").stat().st_size, 0)
        self.assertFalse((root / "v_hidden.pt").exists())
        for li in (0, 17, 31):
            for kind in ("k", "v"):
                want = self._block(li, kind)
                self.assertEqual((root / f"layer_{li:02d}/{kind}.bin").read_bytes(), want.numpy().tobytes())
            self.assertEqual((root / f"layer_{li:02d}/probe_k.bin").read_bytes(),
                             self._block(li, "k")[:, :PROBE].contiguous().numpy().tobytes())
        self.assertEqual(self.receipt["bytes"]["visual_kv"], 301_989_888)
        self.assertEqual(self.receipt["bytes"]["separator_sidecar"], 0)
        self.assertEqual(self.receipt["bytes"]["total"], sum(self.receipt["file_sizes"].values()))
        RS.validate_llava15_raster_meta(m, PROBE)

    def test_image_context_serves_it_without_hidden_states(self):
        ctx = ImageContext(self.tmp / "raster", "cpu", drop_cache=False, require_v_hidden=False)
        try:
            self.assertEqual(ctx.separator_positions(0), [])
            self.assertEqual(ctx.sep_kv_shape, (2, 32, 0, 32, 128))
            self.assertIsNone(ctx.v_hidden)
            counter = IOCounter()
            self.assertTrue(torch.equal(ctx.reader.read_full(9, "v", counter), self._block(9, "v")))
            self.assertTrue(torch.equal(ctx.reader.read_probe(9, counter), self._block(9, "k")[:, :PROBE]))
            rows, vals = ctx.reader.read_chunks(9, "k", [0, 8], counter)
            self.assertEqual(rows.tolist(), list(range(64)) + list(range(512, 576)))
            self.assertTrue(torch.equal(vals[64:], self._block(9, "k")[512:]))
            self.assertEqual(counter.bytes, 576 * ROW + 576 * PROBE * 256 + 128 * ROW)
            ctx.reader.drop_all()                      # includes the empty separator file
            self.assertGreater(ctx.integrity_activation["integrity_read_bytes"], 301_989_888)
        finally:
            ctx.close()

    def test_validator_and_writer_fail_closed(self):
        m = self.meta
        for bad in (dict(m, newline_idx=[575]), dict(m, v_token_num=575), dict(m, bytes_visual_kv=1), dict(m, storage_policy="prefix25"),
                    dict(m, bytes_separator_sidecar=8), dict(m, separator_policy="original_positions_plus_sidecar"),
                    dict(m, bytes_probe_sidecar=0), dict(m, num_layers=31)):
            with self.assertRaises(ValueError):
                RS.validate_llava15_raster_meta(bad, PROBE)
        with self.assertRaises(ValueError):
            RS.validate_llava15_raster_meta(m, 4)
        with self.assertRaises(FileExistsError):
            RS.persist_captured_llava15_raster_prefix(Runner(), self.layers, IDS, self.tmp / "raster", image_id="t", probe_heads=PROBE)
        with self.assertRaises(ValueError):
            RS.persist_captured_llava15_raster_prefix(Runner(), self.layers, IDS, self.tmp / "p0", image_id="t", probe_heads=0)
        with self.assertRaises(ValueError):
            RS.persist_captured_llava15_raster_prefix(Runner(), self.layers, IDS, self.tmp / "ex", image_id="t", probe_heads=PROBE,
                                                      extra_metadata={"newline_idx": [1]})
        with self.assertRaises(ValueError):
            RS.persist_captured_llava15_raster_prefix(Runner(), self.layers[:31], IDS, self.tmp / "l31", image_id="t", probe_heads=PROBE)
        self.assertFalse(list(self.tmp.glob(".*staging*")))          # failed attempts leave nothing behind

    def test_turn1_importance_repack_without_separators(self):
        root = self.tmp / "impress"
        shutil.copytree(self.tmp / "raster", root)
        (root / "integrity.json").unlink()
        scores = torch.rand(32, 576, generator=torch.Generator().manual_seed(7))
        scores[3, 10] = scores[3, 400]                              # a tie: stable order
        timing = impress_mt.reorder_store_by_importance(root, scores, metadata_update={"importance_metric": "h2o", "calibration_questions": 1})
        m = load_meta(root)
        self.assertTrue(m["reordered"] and m["order_is_per_layer"])
        self.assertEqual(m["newline_stored"], [[] for _ in range(32)])
        self.assertEqual((root / "sep_kv.bin").stat().st_size, 0)
        RS.validate_llava15_raster_meta(m, PROBE)                    # still a valid full store
        for li in (0, 3, 31):
            order = m["order"][li]
            self.assertEqual(order, np.argsort(-scores[li].numpy(), kind="stable").tolist())
            self.assertEqual(order, torch.argsort(scores[li], descending=True, stable=True).tolist())
            stored = np.fromfile(root / f"layer_{li:02d}/k.bin", dtype=np.float16).reshape(576, 32, 128)
            self.assertTrue(np.array_equal(stored, self._block(li, "k").numpy()[order]))
            probe = np.fromfile(root / f"layer_{li:02d}/probe_k.bin", dtype=np.float16).reshape(576, PROBE, 128)
            self.assertTrue(np.array_equal(probe, stored[:, :PROBE]))
        self.assertLess(m["order"][3].index(10), m["order"][3].index(400))
        self.assertEqual(timing["per_kind"]["k"]["bytes"], 32 * 576 * ROW)
        self.assertEqual(timing["per_kind"]["probe_k"]["bytes"], m["bytes_probe_sidecar"])
        # the serving path locates no separators and the raster counterfactual maps stored rows back
        ctx = ImageContext(root, "cpu", drop_cache=False, require_v_hidden=False)
        try:
            sel = LayerSelector(SimpleNamespace(layers=[]), ctx, torch.empty(0, dtype=torch.long), ratio=0.25, probe=PROBE, alpha=0.6,
                                importance_metric="h2o", fallback_v="selected")
            self.assertEqual((sel.k_keep, sel.nl(0).numel(), sel.nl(31).numel()), (144, 0, 0))
            self.assertAlmostEqual(sel.thr, (0.25 / 1.75) ** 0.6, places=12)
            keep = torch.zeros(576, dtype=torch.bool)
            keep[:64] = True                                         # the first stored chunk of layer 0
            counts = impress_mt.unreordered_chunk_counts({0: keep}, [{"layer": 0, "mode": "probe"}], ctx.meta)
            want = len({o // 64 for o in m["order"][0][:64]})
            self.assertEqual((counts["k_chunks"], counts["v_chunks"]), ([want], [want]))
        finally:
            ctx.close()


# ============================================================ async packed read
class StreamSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="llava15_stream_test_"))
        cls.layers = _layers(4321)
        cls.scores = torch.rand((1, 576), generator=torch.Generator().manual_seed(5))
        cls.docs = budgets()
        for name in ("U25", "PYR-b20"):
            ST.persist_captured_llava15_prefix(Runner(), cls.layers, IDS, cls.scores, cls.tmp / name, budget=cls.docs[name],
                                               image_id="t", capture_stats=STATS, image_input_sha256="0" * 64)
            seal_integrity(cls.tmp / name)

    @classmethod
    def tearDownClass(cls):
        BIAS.clear()
        shutil.rmtree(cls.tmp)

    def tearDown(self):
        BIAS.clear()

    def _ctx(self, name):
        ctx = Llava15PackedContext(self.tmp / name, "cpu", pin=False, expected_budget=self.docs[name])
        ctx.validate_packed_layout()
        self.addCleanup(ctx.close)
        return ctx

    def test_stream_restores_the_synchronous_buffer_cache_and_masks(self):
        runner = _stream_runner()
        for name in ("U25", "PYR-b20"):
            with self.subTest(vector=name):
                counts = ST.keep_counts(self.docs[name], 576)
                ctx_s = self._ctx(name)
                cache_s = ctx_s.cache.new_request()
                counter_s = IOCounter()
                sync = Llava15PrefixSelector(runner, ctx_s, counter_s)
                with mock.patch("torch.cuda.synchronize", return_value=None), PackedReadTrace(self.tmp / name) as trace_s:
                    sync.prepare()
                bias_s = {li: BIAS[li].clone() for li in BIAS}
                BIAS.clear()

                ctx_a = self._ctx(name)
                ctx_a.reader.buf.zero_()
                threads = []
                real = ctx_a.reader.read_range

                def read_range(offset, length, counter=None):
                    threads.append(threading.get_ident())
                    return real(offset, length, counter)

                ctx_a.reader.read_range = read_range
                counter_a = IOCounter()
                stream = Llava15StreamSelector(runner, ctx_a, counter_a, executor=_pool(self))
                self.assertEqual(threads, [])                                  # nothing is read before begin()
                hooks_before = [len(l._forward_pre_hooks) for l in runner.layers]
                with PackedReadTrace(self.tmp / name) as trace_a:
                    stream.begin()
                    cache_a = ctx_a.cache.new_request()
                    with stream:
                        for layer in runner.layers:
                            layer(hidden_states=torch.zeros(1, 3, 4))
                        for layer in runner.layers:                           # a decode step loads nothing
                            layer(hidden_states=torch.zeros(1, 1, 4))
                self.assertEqual([len(l._forward_pre_hooks) for l in runner.layers], hooks_before)
                self.assertEqual(sorted(BIAS), list(range(32)))
                for li in range(32):
                    self.assertTrue(torch.equal(BIAS[li], bias_s[li]))
                    self.assertTrue(torch.equal(ctx_a.cache.k[li], ctx_s.cache.k[li]))
                    self.assertTrue(torch.equal(ctx_a.cache.v[li], ctx_s.cache.v[li]))
                    self.assertEqual(int((BIAS[li][0, 0, 0, V0:] == 0).sum()), counts[li])
                self.assertTrue(torch.equal(ctx_a.reader.buf, ctx_s.reader.buf))
                sa, ss = trace_a.summary(), trace_s.summary()
                self.assertEqual((ss["preadv_calls"], ss["pread_calls"], ss["bytes"]), (1, 0, 75_497_472))
                self.assertEqual((sa["preadv_calls"], sa["pread_calls"], sa["bytes"]), (32, 0, 75_497_472))
                plan, off = [], 0
                for k in counts:
                    plan.append((off, 2 * k * ROW)); off += 2 * k * ROW
                self.assertEqual([(c["offset"], c["returned"]) for c in sa["calls"]], plan)      # layer order, tiles the file
                self.assertEqual(off, 75_497_472)
                self.assertEqual((counter_a.bytes, counter_a.preads), (counter_s.bytes, 32))
                main = threading.get_ident()
                self.assertTrue(len(threads) == 32 and all(t != main for t in threads))
                a, s = stream.stats(), sync.stats()
                for key in ("k_per_layer", "keep_count_per_layer", "selected_stored_ids", "selected_original_ids",
                            "layer_budget_sha256", "store_format", "k_total", "ratio", "N_content"):
                    self.assertEqual(a[key], s[key], key)
                self.assertEqual((a["read_calls"], a["readahead_policy"], s["read_calls"]), (32, "layer_stream", 1))
                self.assertNotIn("readahead_policy", s)
                ahead = a["read_wait"]["readahead"]
                self.assertEqual((ahead["items_read"], ahead["items_taken"], ahead["max_ahead"]), (32, 32, None))
                self.assertGreaterEqual(a["read_wait"]["read_blocked_ms"], 0.0)
                with self.assertRaises(RuntimeError):
                    stream.prepare()
                del cache_s, cache_a
                BIAS.clear()

    def test_worker_error_surfaces_and_nothing_stays_behind(self):
        runner = _stream_runner()
        ctx = self._ctx("PYR-b20")
        real = ctx.reader.read_range

        def failing(offset, length, counter=None):
            if offset > 0:
                raise OSError("injected read failure")
            return real(offset, length, counter)

        ctx.reader.read_range = failing
        stream = Llava15StreamSelector(runner, ctx, IOCounter())            # owns its executor
        cache = ctx.cache.new_request()
        with self.assertRaises(OSError):
            with stream:
                for layer in runner.layers:
                    layer(hidden_states=torch.zeros(1, 3, 4))
        self.assertTrue(all(len(l._forward_pre_hooks) == 0 for l in runner.layers))
        self.assertIsNone(stream._executor)
        self.assertEqual(stream.done, {0})
        del cache

    def test_request_argument_and_bounded_depth(self):
        self.assertEqual(READAHEAD_POLICIES, ("none", "layer_stream"))
        with self.assertRaises(ValueError):
            request_llava15_packed(None, None, readahead="all_layers")
        runner = _stream_runner()
        ctx = self._ctx("U25")
        stream = Llava15StreamSelector(runner, ctx, IOCounter(), executor=_pool(self), max_ahead=2)
        cache = ctx.cache.new_request()
        with stream:
            for layer in runner.layers:
                layer(hidden_states=torch.zeros(1, 3, 4))
        st = stream.stats()
        self.assertEqual(st["read_wait"]["readahead"]["max_ahead"], 2)
        self.assertLessEqual(st["read_wait"]["readahead"]["peak_outstanding"], 2)
        del cache


# ==================================================================== harness
def _row(method, turn, toks, prompt="p", ties=None, **bytes_):
    row = {"method_id": method, "turn_id": turn, "dialog_id": "d", "prompt": prompt, "ttft_ms": 1.0,
           "result": {"generated_token_ids": list(toks)}}
    if ties is not None:
        row["result"]["logit_ties"] = ties
    for f in G.BYTE_FIELDS:
        row[f] = bytes_.get(f, 0)
    return row


def _step(tied=()):
    return {"max": 1.0, "n_tied": len(tied) if tied else 1, "tied_ids": list(tied) or [0], "top2_ids": [0, 1], "top2": [1.0, 0.5]}


class HarnessTests(unittest.TestCase):
    def test_arms_phases_and_store_kinds(self):
        self.assertEqual(E.PHASE_METHODS["main"], ["recompute", "fullload_async", "impress_async", "llava15_kv_PYR-b20_packed_async"])
        self.assertEqual(E.PHASE_METHODS["pilot"], E.PHASE_METHODS["main"])
        self.assertEqual(len(E.PHASE_METHODS["gate"]), 9)
        self.assertEqual(E.ARMS["fullload_reordered"], {"family": "fullload", "kind": "impress", "async": False,
                                                        "display": "FullLoad loader (재배치 store)"})
        self.assertNotIn("fullload_reordered", E.PHASE_METHODS["main"])
        self.assertNotIn("impress_r100", E.PHASE_METHODS["main"])
        self.assertEqual(E.store_kinds(E.PHASE_METHODS["gate"]), ["raster", "impress", "packed_PYR-b20"])
        self.assertEqual(E.store_kinds(E.PHASE_METHODS["main"]), ["raster", "impress", "packed_PYR-b20"])
        self.assertEqual([E.ARMS[m]["display"] for m in E.PHASE_METHODS["main"]], ["ReComp", "FullLoad", "IMPRESS", "Ours"])
        self.assertTrue(all(E.ARMS[m]["async"] for m in E.PHASE_METHODS["main"][1:]))
        self.assertEqual(E.ARMS["impress_r100"]["ratio"], 1.0)
        self.assertEqual(E.IMPRESS, {"probe_heads": 3, "alpha": 0.6, "ratio": 0.25, "chunk_size": 64, "importance_metric": "h2o",
                                     "fallback_v": "selected", "prefetch_async": "next_layer", "calibration_questions": 1})
        self.assertEqual(E.LOGIT_ARMS, {"gate": ("fullload",)})
        self.assertEqual(E.OURS, S.PYR)                                       # the README-34 arm id
        self.assertEqual(E.request_id("r", "main", "recompute", "d1", 2),
                         "llava15-mtvqa-impress-v2:r:main:llava-hf/llava-1.5-7b-hf:recompute:d1:t2")
        order = E.method_order(E.PHASE_METHODS["main"], 6)
        self.assertEqual(order[0], "impress_async")

    def test_vector_is_the_readme34_document(self):
        doc = ST.load_budget(E.VECTOR_SOURCE)
        self.assertEqual(E.sha(E.VECTOR_SOURCE), E.VECTOR_FILE_SHA)
        self.assertEqual(doc["layer_budget_sha256"], E.VECTOR_RATIO_SHA)
        counts = ST.keep_counts(doc, 576)
        self.assertEqual(counts, S.independent_counts(doc["layer_keep_ratios"], 576, E.VECTOR_FRACTION))
        self.assertEqual(sum(2 * k * ROW for k in counts), E.PACKED_BYTES)

    def test_logit_ties_and_trace_helpers(self):
        head = torch.nn.Linear(4, 6, bias=False)
        with torch.no_grad():
            head.weight.zero_()
            head.weight[2, 0] = head.weight[5, 0] = 3.0
            head.weight[1, 0] = 1.0
        runner = SimpleNamespace(model=SimpleNamespace(lm_head=head))
        with E.LogitTies(runner) as ties:
            head(torch.tensor([[[0.0, 0, 0, 0], [1.0, 0, 0, 0]]]))
            head(torch.tensor([[[1.0, 0, 0, 0], [-1.0, 0, 0, 0]]]))
        self.assertEqual(len(head._forward_hooks), 0)
        self.assertEqual((ties.steps[0]["n_tied"], ties.steps[0]["tied_ids"], ties.steps[0]["max"]), (2, [2, 5], 3.0))
        self.assertEqual((ties.steps[1]["n_tied"], ties.steps[1]["max"]), (3, 0.0))
        self.assertEqual(E._chunks_of(0, 64 * ROW, ROW), [0])
        self.assertEqual(E._chunks_of(64 * ROW, 128 * ROW, ROW), [1, 2])
        self.assertEqual(E._chunks_of(512 * ROW, 64 * ROW, ROW), [8])
        calls = [{"syscall": "pread", "path": "/s/layer_00/k.bin", "offset": 0, "requested": 8, "returned": 8, "t_s": 10.5, "main_thread": False}]
        self.assertEqual(E.compact_trace(calls, "/s", 10.0), [["pread", "layer_00/k.bin", 0, 8, 8, 500.0, 0]])

    def test_unittest_log_parsing(self):
        log = ("ERROR: setUpClass (test_glm46v_runner.NativeProcessorRunnerTests)\n----\nFAIL: test_x (m.C)\n----\n"
               "Ran 12 tests in 1.0s\n\nFAILED (errors=1, failures=1)\n")
        parsed = G.parse_unittest_log(log)
        self.assertEqual((parsed["ran"], parsed["ok"]), (12, False))
        self.assertEqual(parsed["failing"], ["setUpClass (test_glm46v_runner.NativeProcessorRunnerTests)", "test_x (m.C)"])
        self.assertEqual(G.new_failures(parsed["failing"], ["setUpClass (test_glm46v_runner.NativeProcessorRunnerTests)"]), ["test_x (m.C)"])
        self.assertTrue(G.parse_unittest_log("Ran 3 tests in 0.1s\n\nOK\n")["ok"])

    def test_sync_async_comparison(self):
        rows = {"i1": {}, "i2": {}}
        for image in rows:
            for t in (1, 2, 3):
                rows[image][("impress", t)] = _row("impress", t, [5, 2], ssd_read_bytes=10 * (t > 1), k_read_bytes=4 * (t > 1))
                rows[image][("impress_async", t)] = _row("impress_async", t, [5, 2], ssd_read_bytes=10 * (t > 1), k_read_bytes=4 * (t > 1))
        self.assertEqual(G.compare_sync_async(rows, "impress", "impress_async"), {"compared": 6, "mismatched": 0, "examples": []})
        rows["i2"][("impress_async", 3)]["result"]["generated_token_ids"] = [5, 3]
        rows["i1"][("impress_async", 2)]["k_read_bytes"] = 5
        out = G.compare_sync_async(rows, "impress", "impress_async")
        self.assertEqual(out["mismatched"], 2)
        self.assertEqual({(e["image_id"], e["turn_id"], tuple(e["fields"])) for e in out["examples"]},
                         {("i1", 2, ("k_read_bytes",)), ("i2", 3, ("generated_token_ids",))})

    def test_step4_token_comparison_and_reference_count(self):
        def image(t2, t3, p3=None, ties=None):
            """t2 / t3: {arm: tokens}; p3: {arm: prompt of turn 3}."""
            rows = {}
            for arm in ("fullload", "fullload_reordered", "impress_r100"):
                rows[(arm, 1)] = _row(arm, 1, [4, 2])
                rows[(arm, 2)] = _row(arm, 2, t2[arm], ties=ties if arm == "fullload" else None)
                rows[(arm, 3)] = _row(arm, 3, t3[arm], prompt=(p3 or {}).get(arm, "p3"), ties=[_step(), _step()] if arm == "fullload" else None)
            return rows
        same = {"fullload": [9, 2], "fullload_reordered": [9, 2], "impress_r100": [9, 2]}
        flip = {"fullload": [9, 2], "fullload_reordered": [8, 2], "impress_r100": [8, 2]}       # key order flips the raster answer
        rows = {"equal": image(same, same),
                "order": image(flip, {"fullload": [7, 2], "fullload_reordered": [6, 2], "impress_r100": [6, 2]},
                               p3={"fullload": "A2: nine", "fullload_reordered": "A2: eight", "impress_r100": "A2: eight"},
                               ties=[_step((8, 9)), _step()])}
        step4 = G.compare_tokens(rows, "impress_r100", "fullload_reordered")
        self.assertEqual((step4["compared"], step4["different_tokens"], step4["different_input"], step4["requests"]), (6, 0, 0, []))
        ref = G.compare_tokens(rows, "fullload_reordered", "fullload", logits_of="fullload")
        self.assertEqual((ref["compared"], ref["different_tokens"], ref["different_tokens_same_input"], ref["different_input"]), (6, 2, 1, 1))
        first = ref["requests"][0]
        self.assertEqual((first["image_id"], first["turn_id"], first["same_input"], first["position"]), ("order", 2, True, 0))
        self.assertEqual((first["fullload_reordered"], first["fullload"], first["logits_at_position"]["tied_ids"]), ([8, 2], [9, 2], [8, 9]))
        self.assertEqual((ref["requests"][1]["turn_id"], ref["requests"][1]["same_input"]), (3, False))
        self.assertNotIn("logits_at_position", ref["requests"][1])
        # a real difference between the IMPRESS path and the loader on the same store is counted
        rows["bad"] = image({"fullload": [9, 2], "fullload_reordered": [9, 2], "impress_r100": [5, 2]}, same)
        step4 = G.compare_tokens(rows, "impress_r100", "fullload_reordered")
        self.assertEqual((step4["compared"], step4["different_tokens"], step4["different_tokens_same_input"]), (9, 1, 1))
        self.assertEqual((step4["requests"][0]["image_id"], step4["requests"][0]["impress_r100"]), ("bad", [5, 2]))

    def test_turn1_identity(self):
        methods = ["recompute", "fullload_async"]
        rows = {"a": {("recompute", 1): _row("recompute", 1, [1, 2]), ("fullload_async", 1): _row("fullload_async", 1, [1, 2])},
                "b": {("recompute", 1): _row("recompute", 1, [1, 2]), ("fullload_async", 1): _row("fullload_async", 1, [3, 2])}}
        out = G.t1_identity(rows, methods)
        self.assertEqual((out["images"], out["identical"], out["mismatched"]), (2, 1, 1))
        self.assertEqual(out["examples"][0]["image_id"], "b")


if __name__ == "__main__":
    unittest.main()

"""CPU contracts for the asynchronous stored-KV read paths (MT-GQA async run).

Contract: docs/mt_gqa_async_contract.md.  Pins:

  * ``LayerReadAhead``: layer-ordered reads on one worker, bounded depth,
    exception surfacing, clean shutdown;
  * FullLoad ``prefetch="layer_stream"``: the same reads, bytes, pread counts
    and cache contents as the synchronous FullLoad, issued by the worker and
    only after ``begin_readahead``; the synchronous default is untouched;
  * packed ``readahead="layer_stream"``: ranges tile the file, the pinned
    buffer / GPU rows / masks equal the single-``preadv`` path;
  * the turn-1 IMPRESS store: one-question H2O importance equals
    ``calibrate_image``, the repack equals ``scripts/02_reorder.py``'s, and the
    raster counterfactual chunk count is the chunk arithmetic of the masks.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_impress_fidelity import (  # noqa: E402
    N_SUFFIX,
    _FakeDecoderLayer,
    _FakeReader,
    _World,
    _io,
    _server_world,
)
from test_packed_store import _stores  # noqa: E402

from mmimpress import impress_mt  # noqa: E402
from mmimpress.packed_serve import (  # noqa: E402
    PackedImageContext, PackedPrefixSelector, PackedReadTrace,
    PackedStreamSelector)
from mmimpress.readahead import LayerReadAhead  # noqa: E402
from mmimpress.serve import (  # noqa: E402
    BIAS, PREFETCH_POLICIES, LayerSelector, Server, calibrate_image)
from mmimpress.store import (  # noqa: E402
    IOCounter, chunks_for_tokens, load_meta, write_image_store)

ROOT = Path(__file__).resolve().parents[1]


def _pool(test):
    pool = ThreadPoolExecutor(max_workers=1)
    pool.submit(lambda: None).result()
    test.addCleanup(pool.shutdown, True)
    return pool


# ============================================================ LayerReadAhead
class LayerReadAheadTests(unittest.TestCase):
    def test_reads_in_order_on_the_worker_and_hands_out_payloads(self):
        pool = _pool(self)
        seen = []

        def read(key):
            seen.append((key, threading.get_ident()))
            return key * 10

        stream = LayerReadAhead(pool, read, range(6)).start()
        self.assertEqual([stream.take(i) for i in range(6)],
                         [0, 10, 20, 30, 40, 50])
        stream.close()
        self.assertEqual([k for k, _ in seen], list(range(6)))
        self.assertTrue(all(t != threading.get_ident() for _, t in seen))
        stats = stream.stats()
        self.assertEqual(stats["items_read"], 6)
        self.assertEqual(stats["items_taken"], 6)
        self.assertIsNone(stats["max_ahead"])
        self.assertGreaterEqual(stats["first_read_start_s"],
                                stats["started_at_s"])
        self.assertLessEqual(stats["ready_on_arrival"], 6)

    def test_nothing_is_read_before_start(self):
        pool = _pool(self)
        seen = []
        stream = LayerReadAhead(pool, seen.append, range(3))
        time.sleep(0.05)
        self.assertEqual(seen, [])
        with self.assertRaises(RuntimeError):
            stream.take(0)
        stream.start()
        with self.assertRaises(RuntimeError):
            stream.start()
        stream.take(2)                       # waits for 0, 1, 2 in order
        self.assertEqual(seen, [0, 1, 2])
        stream.close()

    def test_max_ahead_bounds_unconsumed_items(self):
        pool = _pool(self)
        seen = []
        stream = LayerReadAhead(pool, lambda k: seen.append(k) or k,
                                range(8), max_ahead=2).start()
        deadline = time.time() + 2.0
        while len(seen) < 2 and time.time() < deadline:
            time.sleep(0.005)
        time.sleep(0.05)                     # the worker must now be parked
        self.assertEqual(seen, [0, 1])
        for i in range(8):
            self.assertEqual(stream.take(i), i)
        stream.close()
        self.assertEqual(seen, list(range(8)))
        self.assertLessEqual(stream.stats()["peak_outstanding"], 2)
        self.assertEqual(stream.stats()["max_ahead"], 2)
        with self.assertRaises(ValueError):
            LayerReadAhead(pool, lambda k: k, range(2), max_ahead=0)

    def test_unbounded_worker_runs_ahead_of_the_consumer(self):
        pool = _pool(self)
        stream = LayerReadAhead(pool, lambda k: k, range(5)).start()
        deadline = time.time() + 2.0
        while stream.stats()["items_read"] < 5 and time.time() < deadline:
            time.sleep(0.005)
        self.assertEqual(stream.stats()["peak_outstanding"], 5)
        self.assertEqual([stream.take(i) for i in range(5)], list(range(5)))
        self.assertEqual(stream.stats()["ready_on_arrival"], 5)
        stream.close()

    def test_worker_exception_surfaces_in_take(self):
        pool = _pool(self)

        def read(key):
            if key == 2:
                raise OSError("disk gone")
            return key

        stream = LayerReadAhead(pool, read, range(4)).start()
        self.assertEqual(stream.take(0), 0)
        self.assertEqual(stream.take(1), 1)
        with self.assertRaises(OSError):
            stream.take(2)
        stream.close()                       # does not raise
        self.assertEqual(pool.submit(lambda: 7).result(), 7)   # worker is free

    def test_close_stops_the_loop_and_rejects_unknown_or_repeated_keys(self):
        pool = _pool(self)
        gate = threading.Event()
        seen = []

        def read(key):
            seen.append(key)
            gate.wait(2.0)
            return key

        stream = LayerReadAhead(pool, read, range(50)).start()
        while not seen:
            time.sleep(0.001)
        closer = threading.Thread(target=stream.close)
        closer.start()
        gate.set()
        closer.join(5.0)
        self.assertFalse(closer.is_alive())
        self.assertLess(len(seen), 50)
        with self.assertRaises(ValueError):
            LayerReadAhead(pool, read, [1, 1])
        other = LayerReadAhead(pool, lambda k: k, range(2)).start()
        other.take(0)
        with self.assertRaises(KeyError):
            other.take(0)
        with self.assertRaises(KeyError):
            other.take(9)
        other.close()


# ======================================================= FullLoad layer_stream
class _ThreadReader(_FakeReader):
    def __init__(self, meta, k, v):
        super().__init__(meta, k, v)
        self.threads = []
        self.times = []

    def read_full(self, layer, kind, counter=None):
        self.threads.append(threading.get_ident())
        self.times.append(time.perf_counter())
        return super().read_full(layer, kind, counter)


def _drive(world, selector):
    with mock.patch("mmimpress.serve.ml.apply_rotary_pos_emb",
                    side_effect=lambda query, key, cos, sin: (query, key)):
        with selector:
            for li, layer in enumerate(world.layers):
                layer(hidden_states=world.hidden[li],
                      position_embeddings=(None, None))
            # decoding steps must not load anything again
            for li, layer in enumerate(world.layers):
                layer(hidden_states=world.hidden[li][:, :1],
                      position_embeddings=(None, None))


def _full(world, reader_cls=_FakeReader, **kwargs):
    ctx = world.context()
    ctx.reader = reader_cls(world.meta, world.k, world.v)
    counter = IOCounter()
    selector = LayerSelector(world.runner, ctx, torch.empty(0, dtype=torch.long),
                             mode="fullload", counter=counter, **kwargs)
    return ctx, counter, selector


class FullLoadStreamTests(unittest.TestCase):
    def tearDown(self):
        BIAS.clear()

    def test_policy_names(self):
        self.assertEqual(PREFETCH_POLICIES,
                         ("none", "next_layer", "layer_stream"))

    def test_same_reads_bytes_cache_and_log_as_synchronous(self):
        for seed in range(3):
            world = _World(seed)
            sync_ctx, sync_counter, sync = _full(world)
            _drive(world, sync)
            ctx, counter, stream = _full(world, reader_cls=_ThreadReader,
                                         prefetch="layer_stream",
                                         prefetch_executor=_pool(self))
            _drive(world, stream)
            L = world.meta["num_layers"]
            self.assertEqual(ctx.reader.calls, sync_ctx.reader.calls)
            self.assertEqual(ctx.reader.calls,
                             [("full", li, kind) for li in range(L)
                              for kind in ("k", "v")])
            self.assertEqual(_io(counter), _io(sync_counter))
            self.assertEqual(list(counter.summary()["per_kind"]),
                             list(sync_counter.summary()["per_kind"]))
            self.assertEqual(stream.log, sync.log)
            for li in range(L):
                self.assertTrue(torch.equal(ctx.cache.k[li],
                                            sync_ctx.cache.k[li]))
                self.assertTrue(torch.equal(ctx.cache.v[li],
                                            sync_ctx.cache.v[li]))
            self.assertEqual(BIAS, {})                     # FullLoad: no mask
            main = threading.get_ident()
            self.assertTrue(all(t != main for t in ctx.reader.threads))
            a, b = stream.stats(), sync.stats()
            self.assertEqual(a["prefetch"], "layer_stream")
            self.assertIsNone(b["prefetch"])
            for key in a:
                if key not in ("prefetch", "hook_ms"):
                    self.assertEqual(a[key], b[key], key)
            wait = stream.read_wait_stats()
            self.assertEqual(wait["readahead"]["items_read"], L)
            self.assertEqual(wait["readahead"]["items_taken"], L)
            self.assertIsNone(wait["readahead"]["max_ahead"])
            self.assertGreaterEqual(wait["read_blocked_ms"], 0.0)
            self.assertIsNone(sync.read_wait_stats()["readahead"])
            self.assertGreater(sync.read_wait_stats()["read_blocked_ms"], 0.0)

    def test_reads_start_only_at_begin_readahead(self):
        world = _World(0)
        ctx, counter, stream = _full(world, reader_cls=_ThreadReader,
                                     prefetch="layer_stream",
                                     prefetch_executor=_pool(self))
        time.sleep(0.05)
        self.assertEqual(ctx.reader.calls, [])             # constructed only
        begun = time.perf_counter()
        stream.begin_readahead()
        first = stream._stream
        stream.begin_readahead()                           # idempotent
        self.assertIs(stream._stream, first)
        _drive(world, stream)
        self.assertTrue(all(t >= begun for t in ctx.reader.times))
        stream.close_readahead()                           # idempotent

    def test_max_ahead_is_passed_to_the_stream(self):
        world = _World(1)
        ctx, counter, stream = _full(world, prefetch="layer_stream",
                                     prefetch_executor=_pool(self),
                                     max_ahead=1)
        _drive(world, stream)
        stats = stream.read_wait_stats()["readahead"]
        self.assertEqual(stats["max_ahead"], 1)
        self.assertLessEqual(stats["peak_outstanding"], 1)

    def test_impress_rejects_layer_stream_and_next_layer_is_unchanged(self):
        world = _World(0)
        with self.assertRaises(ValueError):
            LayerSelector(world.runner, world.context(), torch.tensor([0]),
                          mode="impress", prefetch="layer_stream")
        ctx, counter, selector = _full(world, prefetch="next_layer")
        _drive(world, selector)
        self.assertIsNone(selector._stream)
        self.assertIsNone(selector.stats()["prefetch"])

    def test_owns_and_releases_an_executor_when_none_is_given(self):
        world = _World(2)
        ctx, counter, stream = _full(world, prefetch="layer_stream")
        before = threading.active_count()
        _drive(world, stream)
        self.assertIsNone(stream._executor)
        self.assertEqual(threading.active_count(), before)

    def test_worker_exception_surfaces_and_nothing_stays_in_flight(self):
        world = _World(0)

        class _Broken(_FakeReader):
            def read_full(self, layer, kind, counter=None):
                if layer == 2:
                    raise OSError("short read")
                return super().read_full(layer, kind, counter)

        pool = _pool(self)
        ctx, counter, stream = _full(world, reader_cls=_Broken,
                                     prefetch="layer_stream",
                                     prefetch_executor=pool)
        with self.assertRaises(OSError):
            _drive(world, stream)
        self.assertEqual(pool.submit(lambda: 3).result(), 3)

    def test_keep_masks_and_blocked_time_are_recorded_for_impress(self):
        world = _World(0)
        run = world.run(LayerSelector, importance_metric="h2o",
                        fallback_v="selected")
        sel = run.selector
        self.assertEqual(sorted(sel.keep_masks),
                         list(range(world.meta["num_layers"])))
        for row in sel.log:
            keep = sel.keep_masks[row["layer"]]
            self.assertEqual(keep.dim(), 2 if row["mode"] == "fallback" else 1)
        self.assertGreater(sel.read_wait_stats()["read_blocked_ms"], 0.0)
        self.assertNotIn("read_blocked_ms", sel.stats())


class ServerStreamTests(unittest.TestCase):
    def tearDown(self):
        BIAS.clear()

    def _request(self, world, server, **kwargs):
        ctx = world.context()
        ctx.reader = _ThreadReader(world.meta, world.k, world.v)
        ctx.reader.drop_all = lambda: None
        order = []
        ctx.cache.new_request = lambda: order.append(
            ("new_request", len(ctx.reader.calls))) or object()
        timing = {"first_token_at": 1.0, "finished_at": 2.0,
                  "prefill_ms": 1.0, "decode_ms": 1.0, "generated_tokens": 1,
                  "generated_token_ids": [3], "generated_token_count": 1}

        def decode(cache, suffix_ids, prefix_len):
            world.runner.model()
            return "yes", 3, timing

        with mock.patch("torch.cuda.synchronize", return_value=None), \
             mock.patch("mmimpress.serve.validate_full_load_contract",
                        return_value="ok"), \
             mock.patch(
                 "mmimpress.serve.ml.apply_rotary_pos_emb",
                 side_effect=lambda query, key, cos, sin: (query, key)), \
             mock.patch.object(Server, "_decode", side_effect=decode):
            result = server.request(ctx, suffix_ids=torch.arange(N_SUFFIX),
                                    **kwargs)
        world.runner.model.calls = 0
        return result, ctx

    def test_fullload_layer_stream_request_matches_synchronous(self):
        world = _server_world(0)
        server = Server(world.runner)
        self.addCleanup(server.close)
        sync, sync_ctx = self._request(world, server, mode="fullload")
        self.assertIsNone(server._prefetch_executor)
        out = []
        pre, ctx = self._request(world, server, mode="fullload",
                                 prefetch="layer_stream", selector_out=out)
        self.assertIsNotNone(server._prefetch_executor)
        self.assertEqual(len(out), 1)
        self.assertIsInstance(out[0], LayerSelector)
        clean = lambda io: {                                    # noqa: E731
            "preads": io["preads"], "bytes": io["bytes"],
            "chunk_units": io["chunk_units"],
            "per_kind": {k: {a: b for a, b in v.items() if a != "seconds"}
                         for k, v in io["per_kind"].items()}}
        self.assertEqual(clean(pre["io"]), clean(sync["io"]))
        self.assertEqual(ctx.reader.calls, sync_ctx.reader.calls)
        self.assertEqual(pre["generated_token_ids"],
                         sync["generated_token_ids"])
        self.assertEqual(pre["prefetch"], "layer_stream")
        self.assertIsNone(sync["prefetch"])
        self.assertIsNone(sync["read_wait"]["readahead"])
        self.assertEqual(pre["read_wait"]["readahead"]["items_taken"],
                         world.meta["num_layers"])
        main = threading.get_ident()
        self.assertTrue(all(t != main for t in ctx.reader.threads))
        self.assertTrue(all(t == main for t in sync_ctx.reader.threads))

    def test_impress_request_reports_read_wait_and_rejects_layer_stream(self):
        world = _server_world(1)
        server = Server(world.runner, importance_metric="h2o",
                        fallback_v="selected")
        self.addCleanup(server.close)
        result, _ = self._request(world, server, mode="impress",
                                  prefetch="next_layer")
        self.assertIsNone(result["read_wait"]["readahead"])
        self.assertGreater(result["read_wait"]["read_blocked_ms"], 0.0)
        with self.assertRaises(ValueError):
            self._request(world, server, mode="impress",
                          prefetch="layer_stream")


# ========================================================= packed layer_stream
class PackedStreamTests(unittest.TestCase):
    def tearDown(self):
        BIAS.clear()

    def _runner(self):
        layers = [_FakeDecoderLayer() for _ in range(2)]
        return SimpleNamespace(layers=layers,
                               model=SimpleNamespace(device=torch.device("cpu")))

    def test_stream_plan_tiles_the_file_and_ranges_match_read_all(self):
        for alloc in ("U", "T"):
            for n in (1, 63, 65, 349):
                with self.subTest(alloc=alloc, n=n), \
                        tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    _stores(root, n, alloc)
                    whole = PackedImageContext(root / "p", "cpu", pin=False)
                    whole.reader.read_all()
                    expected = whole.reader.buf.clone()
                    ctx = PackedImageContext(root / "p", "cpu", pin=False)
                    ctx.reader.buf.zero_()
                    plan = ctx.reader.stream_plan()
                    self.assertEqual([key for key, _, _ in plan],
                                     ["sep", 0, 1])
                    spans = sorted((off, off + ln) for _, off, ln in plan)
                    self.assertEqual(spans[0][0], 0)
                    self.assertEqual(spans[-1][1], ctx.reader.total)
                    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
                        self.assertEqual(a1, b0)
                    counter = IOCounter()
                    for _, off, ln in plan:
                        got, _ = ctx.reader.read_range(off, ln, counter)
                        self.assertEqual(got, ln)
                    self.assertTrue(torch.equal(ctx.reader.buf, expected))
                    self.assertEqual(counter.bytes, ctx.reader.total)
                    with self.assertRaises(ValueError):
                        ctx.reader.read_range(ctx.reader.total, 1)
                    whole.close(); ctx.close()

    def test_stream_selector_restores_the_synchronous_cache_and_masks(self):
        runner = self._runner()
        for alloc in ("U", "T"):
            for n in (1, 64, 129, 349):
                with self.subTest(alloc=alloc, n=n), \
                        tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    _stores(root, n, alloc)
                    # synchronous reference
                    ctx_s = PackedImageContext(root / "p", "cpu", pin=False)
                    cache_s = ctx_s.cache.new_request()
                    counter_s = IOCounter()
                    with PackedReadTrace(root / "p") as trace_s:
                        sync = PackedPrefixSelector(runner, ctx_s, 0.25,
                                                    counter_s)
                        with mock.patch("torch.cuda.synchronize",
                                        return_value=None):
                            sync.prepare()
                    bias_s = {li: BIAS[li].clone() for li in BIAS}
                    BIAS.clear()
                    # asynchronous
                    ctx_a = PackedImageContext(root / "p", "cpu", pin=False)
                    counter_a = IOCounter()
                    threads = []
                    real = ctx_a.reader.read_range

                    def read_range(offset, length, counter=None):
                        threads.append(threading.get_ident())
                        return real(offset, length, counter)

                    ctx_a.reader.read_range = read_range
                    stream = PackedStreamSelector(runner, ctx_a, 0.25,
                                                  counter_a,
                                                  executor=_pool(self))
                    with PackedReadTrace(root / "p") as trace_a:
                        stream.begin()
                        cache_a = ctx_a.cache.new_request()
                        with stream:
                            for layer in runner.layers:
                                layer(hidden_states=torch.zeros(1, 3, 4))
                            for layer in runner.layers:      # decode step
                                layer(hidden_states=torch.zeros(1, 1, 4))
                    self.assertEqual(sorted(BIAS), sorted(bias_s))
                    for li in bias_s:
                        self.assertTrue(torch.equal(BIAS[li], bias_s[li]))
                    for li in range(2):
                        self.assertTrue(torch.equal(ctx_a.cache.k[li],
                                                    ctx_s.cache.k[li]))
                        self.assertTrue(torch.equal(ctx_a.cache.v[li],
                                                    ctx_s.cache.v[li]))
                    self.assertTrue(torch.equal(ctx_a.reader.buf,
                                                ctx_s.reader.buf))
                    total = ctx_a.meta["packed_total_bytes"]
                    sa, ss = trace_a.summary(), trace_s.summary()
                    self.assertEqual(ss["preadv_calls"], 1)
                    self.assertEqual(sa["bytes"], total)
                    self.assertEqual(ss["bytes"], total)
                    self.assertEqual(sa["pread_calls"], 0)
                    nonzero = [(key, off, ln) for key, off, ln
                               in ctx_a.reader.stream_plan() if ln]
                    self.assertEqual(
                        [(c["offset"], c["returned"]) for c in sa["calls"]],
                        [(off, ln) for _, off, ln in nonzero])
                    self.assertEqual(counter_a.bytes, counter_s.bytes)
                    self.assertEqual(counter_a.preads, len(nonzero))
                    main = threading.get_ident()
                    self.assertTrue(threads and all(t != main for t in threads))
                    a, s = stream.stats(), sync.stats()
                    for key in ("k_per_layer", "keep_count_per_layer",
                                "selected_stored_ids", "selected_original_ids",
                                "layer_budget_sha256", "store_format",
                                "logical_kv_ratio", "k_total"):
                        self.assertEqual(a[key], s[key], key)
                    self.assertEqual(a["read_calls"], len(nonzero))
                    self.assertEqual(a["readahead_policy"], "layer_stream")
                    self.assertEqual(
                        a["read_wait"]["readahead"]["items_taken"], 3)
                    with self.assertRaises(RuntimeError):
                        stream.prepare()
                    ctx_s.close(); ctx_a.close()
                    del cache_s, cache_a
                    BIAS.clear()

    def test_request_packed_validates_readahead_argument(self):
        from mmimpress.packed_serve import request_packed
        with self.assertRaises(ValueError):
            request_packed(None, None, readahead="all_layers")
        with self.assertRaises(ValueError):
            request_packed(None, None, readahead="layer_stream",
                           directcopy=True)


# ============================================================ IMPRESS turn-1
def _raster_store(root, seed=0, v_num=24, layers=3, heads=4, hd=8):
    g = torch.Generator().manual_seed(seed)
    v0 = 2
    prefix = v0 + v_num
    kv = [(torch.randn(1, heads, prefix, hd, generator=g),
           torch.randn(1, heads, prefix, hd, generator=g))
          for _ in range(layers)]
    newline = [5, 23]
    meta = write_image_store(root, kv, v0, v_num, list(range(prefix)), newline,
                             chunk_size=4, probe_heads=3,
                             separator_sidecar=True)
    return meta, kv, newline


class ImpressTurn1StoreTests(unittest.TestCase):
    def tearDown(self):
        BIAS.clear()

    def test_repack_equals_02_reorder_commit(self):
        spec = importlib.util.spec_from_file_location(
            "reorder02_for_mt_gqa_async", ROOT / "scripts/02_reorder.py")
        reorder02 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reorder02)
        with tempfile.TemporaryDirectory() as td:
            a, b = Path(td) / "a", Path(td) / "b"
            meta, kv, newline = _raster_store(a, seed=3)
            _raster_store(b, seed=3)
            L, vn = meta["num_layers"], meta["v_token_num"]
            scores = torch.rand(L, vn, generator=torch.Generator().manual_seed(9))
            scores[1, 4] = scores[1, 9]                    # a tie: stable order
            raster_k = [np.fromfile(a / f"layer_{li:02d}/k.bin", dtype=np.float16)
                        .reshape(vn, 4, 8).copy() for li in range(L)]
            timing = impress_mt.reorder_store_by_importance(
                a, scores, metadata_update={"importance_metric": "h2o"})
            perms = {li: torch.argsort(scores[li], descending=True,
                                       stable=True).tolist() for li in range(L)}
            reorder02.commit(b, load_meta(b), perms)
            ma, mb = load_meta(a), load_meta(b)
            for key in ("order", "order_is_per_layer", "newline_stored",
                        "reordered"):
                self.assertEqual(ma[key], mb[key], key)
            self.assertTrue(ma["order_is_per_layer"])
            self.assertEqual(ma["importance_metric"], "h2o")
            for li in range(L):
                for name in ("k", "v", "probe_k"):
                    rel = f"layer_{li:02d}/{name}.bin"
                    self.assertEqual((a / rel).read_bytes(),
                                     (b / rel).read_bytes(), rel)
                order = ma["order"][li]
                self.assertEqual(order, perms[li])
                # descending importance, and the stored rows are the raster
                # rows at those original positions
                values = scores[li][torch.tensor(order)]
                self.assertTrue(bool((values[:-1] >= values[1:]).all()))
                stored = np.fromfile(a / f"layer_{li:02d}/k.bin",
                                     dtype=np.float16).reshape(vn, 4, 8)
                self.assertTrue(np.array_equal(stored, raster_k[li][order]))
                probe = np.fromfile(a / f"layer_{li:02d}/probe_k.bin",
                                    dtype=np.float16).reshape(vn, 3, 8)
                self.assertTrue(np.array_equal(probe, stored[:, :3]))
                # separators: located per layer, sidecar rebuilt in that order
                self.assertEqual(sorted(order[p] for p in ma["newline_stored"][li]),
                                 newline)
            sep = np.fromfile(a / "sep_kv.bin", dtype=np.float16).reshape(
                2, L, len(newline), 4, 8)
            for li in range(L):
                stored = np.fromfile(a / f"layer_{li:02d}/k.bin",
                                     dtype=np.float16).reshape(vn, 4, 8)
                self.assertTrue(np.array_equal(
                    sep[0, li], stored[sorted(ma["newline_stored"][li])]))
            for name in ("k", "v", "probe_k"):
                self.assertGreater(timing["per_kind"][name]["bytes"], 0)
                self.assertGreater(timing["per_kind"][name]["write_ms"], 0.0)
            self.assertEqual(timing["per_kind"]["probe_k"]["bytes"],
                             ma["bytes_probe_sidecar"])
            self.assertFalse(timing["shared_order"])
            self.assertFalse(list(a.rglob("*.reorder-tmp")))
            # a store that is already reordered is refused
            with self.assertRaises(ValueError):
                impress_mt.reorder_store_by_importance(a, scores)
            with self.assertRaises(ValueError):
                impress_mt.reorder_store_by_importance(b, scores[:, :5])

    def test_one_suffix_importance_equals_calibrate_image(self):
        world = _server_world(2)
        server = Server(world.runner, importance_metric="h2o")
        self.addCleanup(server.close)
        suffix = torch.arange(N_SUFFIX)

        def context():
            ctx = world.context()
            ctx.cache.new_request = lambda: object()
            return ctx

        with mock.patch("mmimpress.serve.ml.apply_rotary_pos_emb",
                        side_effect=lambda q, k, cos, sin: (q, k)), \
             mock.patch("mmimpress.serve.suffix_ids_for",
                        return_value=suffix):
            reference = calibrate_image(server, context(), ["one question"],
                                        importance_metric="h2o")
            world.runner.model.calls = 0
            ctx = context()
            got = impress_mt.importance_for_suffix(server, ctx, suffix)
        self.assertEqual(tuple(got.shape), (world.meta["num_layers"],
                                            world.meta["v_token_num"]))
        self.assertTrue(torch.equal(got, reference))
        self.assertEqual(ctx.reader.calls,
                         [("full", li, kind)
                          for li in range(world.meta["num_layers"])
                          for kind in ("k", "v")])
        with self.assertRaises(ValueError):
            impress_mt.importance_for_suffix(server, ctx, suffix,
                                             importance_metric="sparsevlm")

    def test_raster_counterfactual_equals_measured_chunks_on_a_raster_store(self):
        for seed in range(3):
            world = _World(seed)
            run = world.run(LayerSelector, importance_metric="h2o",
                            fallback_v="selected")
            sel = run.selector
            self.assertIn("fallback", [r["mode"] for r in sel.log])
            self.assertIn("probe", [r["mode"] for r in sel.log])
            got = impress_mt.unreordered_chunk_counts(
                sel.keep_masks, sel.log, world.meta)      # no order: raster
            self.assertEqual(got["k_chunks"], [r["k_chunks"] for r in sel.log])
            self.assertEqual(got["v_chunks"], [r["v_chunks"] for r in sel.log])
            stats = sel.stats()
            self.assertAlmostEqual(got["k_chunk_fraction"],
                                   stats["touched_k_chunk_fraction"])
            self.assertAlmostEqual(got["v_chunk_fraction"],
                                   stats["touched_v_chunk_fraction"])

    def test_raster_counterfactual_maps_stored_rows_to_original_positions(self):
        meta = {"chunk_size": 4, "n_chunks_per_layer": 3, "v_token_num": 12,
                "order_is_per_layer": True,
                "order": [[0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11],
                          list(range(12))]}
        keep0 = torch.zeros(12, dtype=torch.bool)
        keep0[:3] = True                 # one stored chunk; originals 0, 4, 8
        keep1 = torch.zeros(2, 12, dtype=torch.bool)
        keep1[0, 0] = keep1[1, 9] = True  # union touches chunks 0 and 2
        log = [{"layer": 0, "mode": "probe"}, {"layer": 1, "mode": "fallback"}]
        got = impress_mt.unreordered_chunk_counts({0: keep0, 1: keep1}, log,
                                                  meta)
        self.assertEqual(chunks_for_tokens([0, 1, 2], 12, 4), [0])
        self.assertEqual(got["k_chunks"], [3, 3])
        self.assertEqual(got["v_chunks"], [3, 2])
        self.assertAlmostEqual(got["k_chunk_fraction"], 1.0)
        self.assertAlmostEqual(got["v_chunk_fraction"], 5 / 6)
        self.assertEqual(impress_mt.stored_order(meta, 1), list(range(12)))
        self.assertEqual(impress_mt.stored_order({"v_token_num": 3}, 0),
                         [0, 1, 2])


if __name__ == "__main__":
    unittest.main()

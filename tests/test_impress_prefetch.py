"""CPU-only contracts for next-layer probe-key prefetching (IMPRESS Fig. 10b).

``LayerSelector(prefetch="next_layer")`` hands the probe-key read of layer
i+1 to one background worker when the hook of layer i ends.  These tests pin:

  * identical selection, log, bytes and pread counts versus ``prefetch="none"``;
  * the exact moment a read is queued, and that none is queued after the last
    layer, during decoding, or for FullLoad;
  * that the worker only reads/converts and the main thread does the rest;
  * that a worker exception surfaces in the request and nothing stays queued.

Fakes (reader, cache, layers, world) are shared with
``tests/test_impress_fidelity.py``.
"""
from __future__ import annotations

import importlib.util
import sys
import threading
import time
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_impress_fidelity import (  # noqa: E402
    LEGACY_STAT_KEYS,
    N_SUFFIX,
    _FakeReader,
    _LegacyLayerSelector,
    _World,
    _io,
    _server_world,
)

from mmimpress import serve  # noqa: E402
from mmimpress.serve import BIAS, LayerSelector, Server  # noqa: E402
from mmimpress.store import IOCounter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

TIMING_STATS = ("hook_ms", "probe_read_ms", "probe_pread_ms",
                "prefetch_wait_ms", "probe_hook_blocked_ms",
                "prefetch", "prefetched_probe_reads",
                "prefetch_ready_on_arrival")


class _ThreadRecordingReader(_FakeReader):
    """Fake reader that also records which thread issued every read."""

    def __init__(self, meta, k, v):
        super().__init__(meta, k, v)
        self.threads = []                    # (call, thread ident)

    def read_probe(self, layer, counter=None):
        self.threads.append((("probe", layer), threading.get_ident()))
        return super().read_probe(layer, counter)

    def read_full(self, layer, kind, counter=None):
        self.threads.append((("full", layer, kind), threading.get_ident()))
        return super().read_full(layer, kind, counter)

    def read_chunks(self, layer, kind, cids, counter=None):
        self.threads.append((("chunks", layer, kind), threading.get_ident()))
        return super().read_chunks(layer, kind, cids, counter)


class _DeferredExecutor:
    """Executor whose jobs run only when their result is requested.

    A queued read therefore cannot have happened "early": if the selector
    asks for layer i+1's keys it must be consuming the job it queued.  Every
    submit is recorded together with a snapshot of the selector's state.
    """

    def __init__(self):
        self.submits = []
        self.snapshot = lambda: None

    def submit(self, fn, *args):
        outer = self
        record = {"args": args, "state": self.snapshot(), "ran": False}
        self.submits.append(record)

        class _Lazy(Future):
            def _run_once(self):
                if not record["ran"]:
                    record["ran"] = True
                    try:
                        self.set_result(fn(*args))
                    except BaseException as error:          # noqa: BLE001
                        self.set_exception(error)

            def done(self):
                return record["ran"]

            def result(self, timeout=None):
                self._run_once()
                return Future.result(self, timeout)

            def exception(self, timeout=None):
                self._run_once()
                return Future.exception(self, timeout)

            def cancel(self):
                if record["ran"]:
                    return False
                record["cancelled"] = True
                return Future.cancel(self)

        outer.last = _Lazy()
        return outer.last


def _drive(world, selector, layers=None):
    """Run the prefill once over every layer with the selector's hooks on."""
    with mock.patch(
            "mmimpress.serve.ml.apply_rotary_pos_emb",
            side_effect=lambda query, key, cos, sin: (query, key)):
        with selector:
            for li, layer in enumerate(layers or world.layers):
                layer(hidden_states=world.hidden[li],
                      position_embeddings=(None, None))


def _selector(world, ctx, counter, **kwargs):
    kwargs.setdefault("importance_metric", "h2o")
    kwargs.setdefault("fallback_v", "selected")
    return LayerSelector(world.runner, ctx, torch.tensor([0, 2]), ratio=0.25,
                         probe=3, alpha=0.6,
                         mode=kwargs.pop("mode", "impress"),
                         counter=counter, **kwargs)


def _non_timing(stats):
    return {k: v for k, v in stats.items() if k not in TIMING_STATS}


class PrefetchEquivalenceTests(unittest.TestCase):
    def tearDown(self):
        BIAS.clear()

    def test_next_layer_matches_none_in_selection_log_bytes_and_preads(self):
        for metric, fallback_v in (("h2o", "selected"), ("sparsevlm", "full"),
                                   ("h2o", "full"), ("sparsevlm", "selected")):
            for seed in range(4):
                with self.subTest(metric=metric, fallback_v=fallback_v,
                                  seed=seed):
                    world = _World(seed)
                    sync = world.run(LayerSelector, importance_metric=metric,
                                     fallback_v=fallback_v, prefetch="none")
                    pre = world.run(LayerSelector, importance_metric=metric,
                                    fallback_v=fallback_v,
                                    prefetch="next_layer")
                    # per-layer selection
                    self.assertEqual(sorted(pre.bias), sorted(sync.bias))
                    for li in sync.bias:
                        self.assertTrue(torch.equal(pre.bias[li],
                                                    sync.bias[li]), li)
                    # the log is untouched by prefetching -- every key
                    self.assertEqual(pre.selector.log, sync.selector.log)
                    modes = [r["mode"] for r in pre.selector.log]
                    self.assertIn("probe", modes)
                    self.assertIn("fallback", modes)
                    # bytes, pread counts, chunk units -- total and per kind
                    self.assertEqual(_io(pre.counter), _io(sync.counter))
                    self.assertEqual(
                        list(pre.counter.summary()["per_kind"]),
                        list(sync.counter.summary()["per_kind"]))
                    # the very same reads, in the very same order
                    self.assertEqual(pre.ctx.reader.calls,
                                     sync.ctx.reader.calls)
                    # loaded cache contents
                    for li in range(world.meta["num_layers"]):
                        self.assertTrue(torch.equal(pre.ctx.cache.k[li],
                                                    sync.ctx.cache.k[li]))
                        self.assertTrue(torch.equal(pre.ctx.cache.v[li],
                                                    sync.ctx.cache.v[li]))
                    self.assertEqual(_non_timing(pre.selector.stats()),
                                     _non_timing(sync.selector.stats()))

    def test_none_is_the_default_and_matches_the_frozen_legacy_selector(self):
        world = _World(1)
        implicit = world.run(LayerSelector)
        explicit = world.run(LayerSelector, prefetch="none")
        legacy = world.run(_LegacyLayerSelector)
        self.assertEqual(implicit.selector.prefetch, "none")
        self.assertEqual(implicit.selector.log, explicit.selector.log)
        self.assertEqual(implicit.ctx.reader.calls, legacy.ctx.reader.calls)
        self.assertEqual(_io(implicit.counter), _io(legacy.counter))
        for li in legacy.bias:
            self.assertTrue(torch.equal(implicit.bias[li], legacy.bias[li]))
        new_stats, old_stats = implicit.selector.stats(), legacy.selector.stats()
        for key in LEGACY_STAT_KEYS:
            self.assertEqual(new_stats[key], old_stats[key], key)
        # no worker exists and no read was prefetched
        self.assertIsNone(implicit.selector._executor)
        self.assertEqual(new_stats["prefetch"], "none")
        self.assertEqual(new_stats["prefetch_wait_ms"], 0.0)
        self.assertEqual(new_stats["prefetched_probe_reads"], 0)
        self.assertEqual([r["source"] for r in implicit.selector.probe_log],
                         ["hook"] * world.meta["num_layers"])
        # with synchronous reads the hook is blocked for the whole read
        self.assertEqual(new_stats["probe_hook_blocked_ms"],
                         new_stats["probe_read_ms"])

    def test_invalid_prefetch_value_is_rejected(self):
        world = _World(0)
        with self.assertRaises(ValueError):
            world.run(LayerSelector, prefetch="all_layers")
        with self.assertRaises(ValueError):
            Server(world.runner, prefetch="all_layers")


class PrefetchSchedulingTests(unittest.TestCase):
    def tearDown(self):
        BIAS.clear()

    def test_read_is_queued_when_the_previous_hook_ends_and_never_after_last(
            self):
        world = _World(2)
        L = world.meta["num_layers"]
        ctx, counter = world.context(), IOCounter()
        executor = _DeferredExecutor()
        selector = _selector(world, ctx, counter, prefetch="next_layer",
                             prefetch_executor=executor)
        executor.snapshot = lambda: {
            "done": sorted(selector.done),
            "bias": sorted(BIAS),
            "logged": [r["layer"] for r in selector.log],
            "calls": list(ctx.reader.calls),
        }
        BIAS.clear()
        _drive(world, selector)

        # one queued read per layer 1..L-1, in order; nothing after the last
        self.assertEqual([r["args"] for r in executor.submits],
                         [(li,) for li in range(1, L)])
        for record in executor.submits:
            nxt = record["args"][0]
            state = record["state"]
            # queued as hook nxt-1 finishes: that layer is selected, masked,
            # logged, and all of its own reads are already issued ...
            self.assertEqual(state["done"], list(range(nxt)))
            self.assertEqual(state["bias"], list(range(nxt)))
            self.assertEqual(state["logged"], list(range(nxt)))
            self.assertTrue(any(c[1] == nxt - 1 and c[0] != "probe"
                                for c in state["calls"]))
            # ... and nothing of the next layer has been read yet
            self.assertFalse(any(c[1] == nxt for c in state["calls"]))
            self.assertTrue(record["ran"])          # consumed by hook nxt
        # layer 0 was read in the hook, the rest came from the queue
        self.assertEqual([(r["layer"], r["source"])
                          for r in selector.probe_log],
                         [(0, "hook")] + [(li, "prefetch")
                                          for li in range(1, L)])
        self.assertIsNone(selector._pending)
        stats = selector.stats()
        self.assertEqual(stats["prefetch"], "next_layer")
        self.assertEqual(stats["prefetched_probe_reads"], L - 1)
        self.assertGreaterEqual(stats["prefetch_wait_ms"], 0.0)
        self.assertGreaterEqual(stats["probe_hook_blocked_ms"],
                                stats["prefetch_wait_ms"])
        # an injected executor belongs to the caller: it is not shut down
        self.assertIs(selector._executor, executor)

    def test_decoding_steps_issue_no_reads_and_queue_nothing(self):
        world = _World(3)
        ctx, counter = world.context(), IOCounter()
        executor = _DeferredExecutor()
        selector = _selector(world, ctx, counter, prefetch="next_layer",
                             prefetch_executor=executor)
        with mock.patch(
                "mmimpress.serve.ml.apply_rotary_pos_emb",
                side_effect=lambda query, key, cos, sin: (query, key)):
            with selector:
                for li, layer in enumerate(world.layers):       # prefill
                    layer(hidden_states=world.hidden[li],
                          position_embeddings=(None, None))
                calls = list(ctx.reader.calls)
                submits = len(executor.submits)
                log = [dict(r) for r in selector.log]
                io_before = _io(counter)
                for _ in range(3):                              # decode steps
                    for li, layer in enumerate(world.layers):
                        layer(hidden_states=world.hidden[li][:, :1],
                              position_embeddings=(None, None))
                self.assertEqual(ctx.reader.calls, calls)
                self.assertEqual(len(executor.submits), submits)
                self.assertEqual(selector.log, log)
                self.assertEqual(_io(counter), io_before)
                self.assertIsNone(selector._pending)

    def test_fullload_never_prefetches(self):
        world = _World(0)
        ctx, counter = world.context(), IOCounter()
        executor = _DeferredExecutor()
        selector = _selector(world, ctx, counter, mode="fullload",
                             prefetch="next_layer",
                             prefetch_executor=executor)
        _drive(world, selector)
        self.assertEqual(executor.submits, [])
        self.assertEqual(selector.probe_log, [])
        self.assertTrue(all(c[0] == "full" for c in ctx.reader.calls))
        stats = selector.stats()
        self.assertIsNone(stats["prefetch"])
        self.assertEqual(stats["prefetch_wait_ms"], 0.0)
        self.assertEqual(stats["probe_read_ms"], 0.0)

    def test_worker_only_reads_probe_keys_and_main_thread_does_the_rest(self):
        world = _World(4)
        L = world.meta["num_layers"]
        ctx, counter = world.context(), IOCounter()
        ctx.reader = _ThreadRecordingReader(world.meta, world.k, world.v)
        selector = _selector(world, ctx, counter, prefetch="next_layer")
        writers = []
        original_write = ctx.cache.write
        original_write_full = ctx.cache.write_full
        ctx.cache.write = lambda *a: (writers.append(threading.get_ident()),
                                      original_write(*a))[1]
        ctx.cache.write_full = lambda *a: (
            writers.append(threading.get_ident()), original_write_full(*a))[1]
        recorders = []
        original_record = counter.record

        def record(*args, **kwargs):
            recorders.append(threading.get_ident())
            return original_record(*args, **kwargs)

        counter.record = record
        BIAS.clear()
        _drive(world, selector)

        main = threading.get_ident()
        by_call = dict(ctx.reader.threads)
        self.assertEqual(by_call[("probe", 0)], main)       # layer 0: in hook
        workers = {by_call[("probe", li)] for li in range(1, L)}
        self.assertEqual(len(workers), 1)                   # ONE worker
        self.assertNotIn(main, workers)
        for call, thread in ctx.reader.threads:             # K/V stay sync
            if call[0] != "probe":
                self.assertEqual(thread, main, call)
        # cache writes and the request's IOCounter stay on the main thread
        self.assertTrue(writers)
        self.assertEqual(set(writers), {main})
        self.assertEqual(set(recorders), {main})
        # the selector created the worker itself and shut it down on exit
        self.assertIsNone(selector._executor)
        self.assertFalse(selector._owns_executor)


class _SlowProbeReader(_FakeReader):
    """Probe reads of one layer take a while (to race with the main thread)."""

    def __init__(self, meta, k, v, slow_layer, delay=0.05):
        super().__init__(meta, k, v)
        self.slow_layer, self.delay = slow_layer, delay
        self.finished = []
        self.started = threading.Event()

    def read_probe(self, layer, counter=None):
        if layer == self.slow_layer:
            self.started.set()
            time.sleep(self.delay)
        out = super().read_probe(layer, counter)
        self.finished.append(layer)
        return out


class _FailingProbeReader(_FakeReader):
    def __init__(self, meta, k, v, fail_layer, error):
        super().__init__(meta, k, v)
        self.fail_layer, self.error = fail_layer, error

    def read_probe(self, layer, counter=None):
        if layer == self.fail_layer:
            raise self.error
        return super().read_probe(layer, counter)


class PrefetchFailureTests(unittest.TestCase):
    def tearDown(self):
        BIAS.clear()

    def test_worker_exception_reaches_the_request_unchanged(self):
        world = _World(0)
        error = OSError("short pread for probe_k layer 2")
        ctx, counter = world.context(), IOCounter()
        ctx.reader = _FailingProbeReader(world.meta, world.k, world.v, 2,
                                         error)
        pool = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(pool.shutdown)
        selector = _selector(world, ctx, counter, prefetch="next_layer",
                             prefetch_executor=pool)
        BIAS.clear()
        with self.assertRaises(OSError) as caught:
            _drive(world, selector)
        self.assertIs(caught.exception, error)
        # layers 0 and 1 completed; layer 2 never selected anything
        self.assertEqual(sorted(selector.done), [0, 1])
        self.assertEqual([r["layer"] for r in selector.log], [0, 1])
        self.assertFalse(any(c[1] >= 2 for c in ctx.reader.calls))
        # nothing queued, hooks removed, caller's worker still usable
        self.assertIsNone(selector._pending)
        self.assertTrue(all(len(layer._forward_pre_hooks) == 0
                            for layer in world.layers))
        self.assertEqual(pool.submit(lambda: 7).result(), 7)

    def test_own_worker_is_shut_down_when_the_worker_read_fails(self):
        world = _World(0)
        ctx, counter = world.context(), IOCounter()
        ctx.reader = _FailingProbeReader(world.meta, world.k, world.v, 1,
                                         ValueError("bad probe sidecar"))
        selector = _selector(world, ctx, counter, prefetch="next_layer")
        before = threading.active_count()
        with self.assertRaisesRegex(ValueError, "bad probe sidecar"):
            _drive(world, selector)
        self.assertIsNone(selector._executor)
        self.assertIsNone(selector._pending)
        self.assertEqual(threading.active_count(), before)

    def test_main_thread_exception_cleans_up_a_queued_read(self):
        # Layer 1's forward raises AFTER hook 1 queued layer 2's read, which
        # is still running.  The original exception must propagate and the
        # read must be finished or cancelled -- never left in flight.
        world = _World(1)
        boom = RuntimeError("layer forward failed")
        ctx, counter = world.context(), IOCounter()
        ctx.reader = _SlowProbeReader(world.meta, world.k, world.v,
                                      slow_layer=2)
        reader = ctx.reader

        class _Raising(type(world.layers[0])):
            def forward(self, hidden_states, position_embeddings=None):
                # fail only once the worker is in the middle of the read
                assert reader.started.wait(timeout=5)
                raise boom

        layers = list(world.layers)
        layers[1] = _Raising()
        world.runner.layers = layers
        pool = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(pool.shutdown)
        selector = _selector(world, ctx, counter, prefetch="next_layer",
                             prefetch_executor=pool)
        BIAS.clear()
        with self.assertRaises(RuntimeError) as caught:
            _drive(world, selector, layers=layers)
        self.assertIs(caught.exception, boom)
        self.assertIsNone(selector._pending)
        # the queued read ran to completion before the selector returned;
        # being unconsumed, it is not billed to the request
        self.assertEqual(ctx.reader.finished, [0, 1, 2])
        self.assertEqual(
            counter.summary()["per_kind"]["probe"]["preads"], 2)
        self.assertTrue(all(len(layer._forward_pre_hooks) == 0
                            for layer in layers))

    def test_queued_but_unstarted_read_is_cancelled(self):
        world = _World(1)
        ctx, counter = world.context(), IOCounter()
        executor = _DeferredExecutor()
        selector = _selector(world, ctx, counter, prefetch="next_layer",
                             prefetch_executor=executor)
        with mock.patch(
                "mmimpress.serve.ml.apply_rotary_pos_emb",
                side_effect=lambda query, key, cos, sin: (query, key)):
            with selector:
                world.layers[0](hidden_states=world.hidden[0],
                                position_embeddings=(None, None))
                self.assertIsNotNone(selector._pending)     # layer 1 queued
        self.assertIsNone(selector._pending)
        self.assertEqual(len(executor.submits), 1)
        self.assertFalse(executor.submits[0]["ran"])
        self.assertTrue(executor.submits[0].get("cancelled"))
        self.assertFalse(any(c == ("probe", 1) for c in ctx.reader.calls))


class ServerPrefetchTests(unittest.TestCase):
    def tearDown(self):
        BIAS.clear()

    def _request(self, world, server, reader_cls=None, **kwargs):
        ctx = world.context()
        if reader_cls is not None:
            ctx.reader = reader_cls(world.meta, world.k, world.v)
        ctx.reader.drop_all = lambda: None
        ctx.cache.new_request = lambda: object()
        timing = {"first_token_at": 1.0, "finished_at": 2.0,
                  "prefill_ms": 1.0, "decode_ms": 1.0, "generated_tokens": 1,
                  "generated_token_ids": [3], "generated_token_count": 1}
        events = []

        def synchronize():
            events.append(("sync", len(ctx.reader.calls)))

        def decode(cache, suffix_ids, prefix_len):
            events.append(("prefill", len(ctx.reader.calls),
                           threading.active_count()))
            world.runner.model()
            return "yes", 3, timing

        with mock.patch("torch.cuda.synchronize", side_effect=synchronize), \
             mock.patch(
                 "mmimpress.serve.ml.apply_rotary_pos_emb",
                 side_effect=lambda query, key, cos, sin: (query, key)), \
             mock.patch.object(Server, "_decode", side_effect=decode):
            result = server.request(ctx, suffix_ids=torch.arange(N_SUFFIX),
                                    importance_metric="h2o",
                                    fallback_v="selected", **kwargs)
        return result, ctx, events

    def test_request_prefetch_matches_sync_and_starts_worker_before_timer(
            self):
        world = _server_world(2)
        L = world.meta["num_layers"]
        server = Server(world.runner)
        self.addCleanup(server.close)
        self.assertEqual(server.prefetch, "none")
        sync, sync_ctx, _ = self._request(world, server)
        self.assertIsNone(server._prefetch_executor)       # never created
        threads_before = threading.active_count()
        pre, pre_ctx, events = self._request(
            world, server, reader_cls=_ThreadRecordingReader,
            prefetch="next_layer")
        # the worker thread exists before the timer's synchronize, and no
        # read at all has happened by then
        self.assertEqual(events[0], ("sync", 0))
        self.assertEqual(events[1][:2], ("prefill", 0))
        self.assertEqual(events[1][2], threads_before + 1)
        self.assertIsNotNone(server._prefetch_executor)

        for key in ("answer", "first_token_id", "generated_token_ids",
                    "fallback_rate", "probe_rate", "mean_jaccard",
                    "touched_chunk_fraction", "touched_k_chunk_fraction",
                    "touched_v_chunk_fraction", "importance_metric",
                    "fallback_v", "n_score_rows"):
            self.assertEqual(pre[key], sync[key], key)
        clean = lambda io: {                                    # noqa: E731
            "preads": io["preads"], "bytes": io["bytes"],
            "chunk_units": io["chunk_units"],
            "per_kind": {k: {a: b for a, b in v.items() if a != "seconds"}
                         for k, v in io["per_kind"].items()}}
        self.assertEqual(clean(pre["io"]), clean(sync["io"]))
        self.assertEqual(pre_ctx.reader.calls, sync_ctx.reader.calls)
        self.assertEqual(sync["prefetch"], "none")
        self.assertEqual(sync["prefetched_probe_reads"], 0)
        self.assertEqual(pre["prefetch"], "next_layer")
        self.assertEqual(pre["prefetched_probe_reads"], L - 1)

        main = threading.get_ident()
        by_call = dict(pre_ctx.reader.threads)
        self.assertEqual(by_call[("probe", 0)], main)
        self.assertTrue(all(by_call[("probe", li)] != main
                            for li in range(1, L)))
        # the same worker serves the next request
        executor = server._prefetch_executor
        self._request(world, server, prefetch="next_layer")
        self.assertIs(server._prefetch_executor, executor)
        server.close()
        self.assertIsNone(server._prefetch_executor)

    def test_server_default_and_fullload_stay_synchronous(self):
        world = _server_world(0)
        server = Server(world.runner, prefetch="next_layer")
        self.addCleanup(server.close)
        self.assertEqual(server.prefetch, "next_layer")
        with self.assertRaises(ValueError):
            self._request(world, server, prefetch="all_layers")
        with mock.patch("mmimpress.serve.validate_full_load_contract",
                        return_value="ok"):
            full, ctx, _ = self._request(world, server, mode="fullload")
        self.assertIsNone(server._prefetch_executor)       # FullLoad: none
        self.assertIsNone(full["prefetch"])
        self.assertTrue(all(c[0] == "full" for c in ctx.reader.calls))
        pre, _, _ = self._request(world, server)           # server default
        self.assertEqual(pre["prefetch"], "next_layer")
        self.assertIsNotNone(server._prefetch_executor)


class EvalScriptPrefetchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "eval04_for_impress_prefetch", ROOT / "scripts/04_eval.py")
        cls.eval04 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.eval04)

    def test_selectors_pin_the_prefetch_policy(self):
        selectors = self.eval04.SELECTORS
        self.assertEqual(selectors["impress_sync"],
                         ("impress", {"prefetch": "none"}))
        self.assertEqual(selectors["impress_prefetch"],
                         ("impress", {"prefetch": "next_layer"}))
        # pre-existing selectors are unchanged
        self.assertEqual(selectors["sparsevlm"], ("impress", {}))
        self.assertEqual(self.eval04._display_method("impress_sync"),
                         "IMPRESS (동기)")
        self.assertEqual(self.eval04._display_method("impress_prefetch"),
                         "IMPRESS (선행 로딩)")

    def test_pack_carries_prefetch_timing(self):
        result = {
            "answer": "yes", "ttft": 0.01, "decode_latency": 0.02,
            "e2e_latency": 0.03, "generated_tokens": 1,
            "generated_token_ids": [3], "prefetch": "next_layer",
            "probe_read_ms": 20.0, "probe_pread_ms": 15.0,
            "prefetch_wait_ms": 4.0, "probe_hook_blocked_ms": 5.0,
            "prefetched_probe_reads": 31, "prefetch_ready_on_arrival": 20,
            "hook_ms": 700.0,
            "io": {"bytes": 20, "mb": 2e-5, "ms": 15.0, "preads": 1,
                   "chunk_units": 0,
                   "per_kind": {"probe": {"bytes": 20, "preads": 1,
                                          "units": 0}}},
        }
        packed = self.eval04._pack(result, {"answer": "yes"}, "gqa")
        for key in ("prefetch", "probe_read_ms", "probe_pread_ms",
                    "prefetch_wait_ms", "probe_hook_blocked_ms",
                    "prefetched_probe_reads", "prefetch_ready_on_arrival"):
            self.assertEqual(packed[key], result[key], key)
        self.assertEqual(packed["preads"], 1)
        self.assertEqual(packed["ssd_read_bytes"], 20)


if __name__ == "__main__":
    unittest.main()

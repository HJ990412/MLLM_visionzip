"""HyperInfer's next-layer KV prefetch with the paper's reactive truncation.

Contract: docs/mt_gqa_hyperinfer_v2_contract.md; source table: docs/hyperinfer_port_contract.md.

The prefetch of layer l+1 stops when layer l's GPU work is done: no read is started after the
stop, a read already started is finished, and what was read by then is the prefetched set.
These tests run on the host; the stop is driven directly (``JobTests``) or held at the start of
the next layer's hook (``_Held``) so that both ends are covered: nothing prefetched beyond the
probe block, and every predicted chunk prefetched.
"""
from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_hyperinfer import _Run  # noqa: E402
from test_impress_official import _World  # noqa: E402

from mmimpress import hyperinfer_v2 as h2  # noqa: E402
from mmimpress import impress_official as io_  # noqa: E402
from mmimpress.serve import BIAS  # noqa: E402
from mmimpress.store import chunk_span  # noqa: E402


# ====================================================================== job
class _Gate:
    """A fake store read that blocks until released, one read at a time."""

    def __init__(self):
        self.calls = []
        self.started = threading.Semaphore(0)
        self.release = threading.Semaphore(0)

    def probe(self, reader, layer, counter):
        self.calls.append(("probe", layer))
        self.started.release()
        self.release.acquire()
        return ("keys", "rows")

    def chunk(self, reader, layer, kind, cid, counter):
        self.calls.append((kind, cid))
        self.started.release()
        self.release.acquire()
        return np.full((1,), cid)


class JobTests(unittest.TestCase):
    def setUp(self):
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.keep = torch.ones(256, dtype=torch.bool)
        self.map = torch.arange(256)

    def tearDown(self):
        self.pool.shutdown(wait=True)

    def _start(self, gate):
        job = h2.PrefetchJob(3, self.keep)
        job.future = self.pool.submit(h2.run_prefetch_job, job, None, self.map, 64, gate.probe, gate.chunk)
        return job

    def test_no_read_starts_after_the_stop_and_the_read_in_flight_ends(self):
        gate = _Gate()
        job = self._start(gate)
        for _ in range(4):                  # probe, k0, v0, k1 started; k1 is in flight
            gate.started.acquire()
            if len(gate.calls) < 4:
                gate.release.release()
        self.assertEqual(gate.calls, [("probe", 3), ("k", 0), ("v", 0), ("k", 1)])
        stop_at = job.request_stop()
        self.assertFalse(job.future.done())         # the read in flight is not abandoned
        gate.release.release()
        job.future.result(timeout=10)
        self.assertEqual(gate.calls, [("probe", 3), ("k", 0), ("v", 0), ("k", 1)])     # v1 was never started
        self.assertEqual(job.ended, h2.STOPPED)
        self.assertEqual(sorted(job.done["k"]), [0, 1])
        self.assertEqual(sorted(job.done["v"]), [0])
        self.assertEqual([(r[0], r[1]) for r in job.reads], [("k", 0), ("v", 0), ("k", 1)])
        self.assertTrue(all(r[2] < stop_at for r in job.reads))
        self.assertEqual(sum(r[4] <= stop_at for r in job.reads), 2)                    # two finished inside the window
        self.assertGreater(job.refused_at, stop_at)
        self.assertEqual(job.predicted, [0, 1, 2, 3])

    def test_stop_before_the_probe_read_ends_reads_the_probe_and_nothing_else(self):
        gate = _Gate()
        job = self._start(gate)
        gate.started.acquire()
        job.request_stop()
        self.assertFalse(job.probe_ready.is_set())
        gate.release.release()
        job.future.result(timeout=10)
        self.assertTrue(job.probe_ready.is_set())
        self.assertEqual(job.block, ("keys", "rows"))
        self.assertEqual(gate.calls, [("probe", 3)])
        self.assertEqual((job.done, job.reads, job.ended), ({"k": {}, "v": {}}, [], h2.STOPPED))

    def test_without_a_stop_every_predicted_chunk_is_read_k_then_v_ascending(self):
        gate = _Gate()
        keep = torch.zeros(256, dtype=torch.bool)
        keep[[200, 3, 70]] = True
        job = h2.PrefetchJob(1, keep)
        job.future = self.pool.submit(h2.run_prefetch_job, job, None, self.map, 64, gate.probe, gate.chunk)
        for _ in range(7):
            gate.started.acquire()
            gate.release.release()
        job.future.result(timeout=10)
        self.assertEqual(gate.calls, [("probe", 1), ("k", 0), ("v", 0), ("k", 1), ("v", 1), ("k", 3), ("v", 3)])
        self.assertEqual(job.ended, h2.EXHAUSTED)
        self.assertIsNone(job.stop_at)

    def test_stop_is_recorded_once(self):
        job = h2.PrefetchJob(1, self.keep)
        first = job.request_stop()
        time.sleep(0.002)
        self.assertEqual(job.request_stop(), first)
        self.assertIsNone(job.may_start())

    def test_a_failed_probe_read_still_releases_the_waiter(self):
        def broken(reader, layer, counter):
            raise OSError("probe read failed")

        job = h2.PrefetchJob(1, self.keep)
        job.future = self.pool.submit(h2.run_prefetch_job, job, None, self.map, 64, broken, None)
        self.assertTrue(job.probe_ready.wait(timeout=10))
        self.assertIsNone(job.block)
        with self.assertRaises(OSError):
            job.future.result(timeout=10)


# ================================================================= selector
class _Held(h2.HyperInferSelector):
    """The stop of every job is held until the worker has read everything it predicted."""

    def _before_stop(self, li, job):
        deadline = time.time() + 30
        while job.finished_at is None and time.time() < deadline:
            time.sleep(0.0005)


class _HeldAfter(_Held):
    """The stop arrives while the worker's ``n``-th chunk read of every job is in flight."""

    n = 3

    def _submit(self, li):
        nxt = li + 1
        if nxt >= self._base_ctx.meta["num_layers"]:
            return
        job = h2.PrefetchJob(nxt, self.keep_masks[li])
        seen = []

        def chunk(reader, layer, kind, cid, counter):
            rows = h2.read_one_chunk(reader, layer, kind, cid, counter)
            seen.append(cid)
            if len(seen) == self.n:
                job.request_stop()
            return rows

        job.future = self._executor.submit(h2.run_prefetch_job, job, self._base_ctx.reader, self._maps[nxt - 1],
                                           self._base_ctx.meta["chunk_size"], io_.read_probe_block, chunk)
        self._job = job
        self._submit_at = time.perf_counter()
        self.layers.setdefault(li, {})["submitted_at"] = self._submit_at


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.pool = ThreadPoolExecutor(max_workers=1)

    def tearDown(self):
        self.pool.shutdown(wait=True)
        BIAS.clear()

    def _impress(self, w):
        return _Run(w, lambda runner, ctx, counter: io_.OfficialImpressSelector(
            runner, ctx, counter=counter, prefetch="next_layer", prefetch_executor=self.pool))

    def _hyper(self, w, cls=h2.HyperInferSelector, mode="async"):
        return _Run(w, lambda runner, ctx, counter: cls(
            runner, ctx, kv_prefetch=mode, counter=counter, prefetch_executor=self.pool))

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
            a, b = self._impress(w), self._hyper(w, mode="off")
            self._same_outcome(a, b)
            self.assertEqual({k: (v["bytes"], v["preads"]) for k, v in a.counter.per_kind.items()},
                             {k: (v["bytes"], v["preads"]) for k, v in b.counter.per_kind.items()})
            for li in range(w.L):
                self.assertTrue(torch.equal(a.ctx.cache.k[li], b.ctx.cache.k[li]))
                self.assertTrue(torch.equal(a.ctx.cache.v[li], b.ctx.cache.v[li]))
            self.assertEqual(b.selector.hyperinfer_stats(), {"kv_prefetch": "off"})
            self.assertEqual(b.selector.stats()["prefetch"], "next_layer")

    def _check(self, w, a, b):
        """Whatever the worker managed to read: IMPRESS's selection, mask and cache rows; consistent accounting."""
        self._same_outcome(a, b)
        for li in range(w.L):
            self.assertTrue(torch.equal(a.ctx.cache.k[li], b.ctx.cache.k[li]), li)
            self.assertTrue(torch.equal(a.ctx.cache.v[li], b.ctx.cache.v[li]), li)
        s = b.selector.hyperinfer_stats()
        cols = s["layers"]
        m = w.meta
        nc, cs, vn = m["n_chunks_per_layer"], m["chunk_size"], w.vn
        row = w.H * w.hd * 2
        self.assertEqual(s["late_starts"], 0)
        self.assertEqual(cols["late_starts"], [0] * w.L)
        self.assertEqual((cols["k_done"][0], cols["v_done"][0], cols["ended"][0], cols["window_ms"][0]), (0, 0, None, None))
        self.assertEqual(s["jobs"], w.L - 1)
        maps = h2.layer_position_maps(m["order"])
        for li in range(w.L):
            needed = nc if cols["fallback"][li] else len({int(p) // cs for p in b.selector.keep_masks[li].nonzero().flatten()})
            self.assertEqual(cols["needed_chunks"][li], needed)
            for kind in "kv":
                self.assertEqual(cols[f"{kind}_hit"][li] + cols[f"{kind}_missing"][li], needed)
                self.assertEqual(cols[f"{kind}_hit"][li] + cols[f"{kind}_wasted"][li], cols[f"{kind}_done"][li])
            self.assertIn(cols["k_done"][li] - cols["v_done"][li], (0, 1))          # K then V, one read at a time
            self.assertEqual(cols["prefetched_chunks"][li], cols["v_done"][li])
            self.assertEqual(cols["reads_done_in_window"][li] + cols["reads_done_after_stop"][li], cols["k_done"][li] + cols["v_done"][li])
            self.assertLessEqual(cols["reads_done_after_stop"][li], 1)              # only the read in flight
            self.assertEqual(cols["used_tokens"][li], int(b.selector.keep_masks[li].sum()))
            if li:
                want = h2.predicted_chunks(b.selector.keep_masks[li - 1], maps[li - 1], cs)
                self.assertEqual(cols["predicted_chunks"][li], len(want))
                job = b.selector.jobs[li]
                self.assertEqual(sorted(job.done["k"]), want[:cols["k_done"][li]])   # a prefix of the prediction, ascending
                self.assertEqual(sorted(job.done["v"]), want[:cols["v_done"][li]])
                self.assertEqual(cols["ended"][li] == h2.EXHAUSTED, cols["v_done"][li] == len(want))
        kv = b.counter.per_kind["k"]["bytes"] + b.counter.per_kind["v"]["bytes"]
        self.assertEqual(kv, s["prefetched_bytes"] + s["sync_bytes"])
        self.assertEqual(b.counter.per_kind["probe"]["bytes"], a.counter.per_kind["probe"]["bytes"])
        impress_kv = a.counter.per_kind["k"]["bytes"] + a.counter.per_kind["v"]["bytes"]
        self.assertEqual(kv - s["wasted_bytes"], impress_kv)
        sync = sum(sum(chunk_span(c, vn, cs)[1] - chunk_span(c, vn, cs)[0] for c in b.selector._reads[li][kind]["missing"])
                   for li in range(w.L) for kind in "kv") * row
        self.assertEqual(s["sync_bytes"], sync)
        self.assertEqual([r["source"] for r in b.selector.probe_log], ["hook"] + ["prefetch"] * (w.L - 1))
        self.assertEqual(len(s["probe_read_ms"]), w.L - 1)
        self.assertEqual(len(s["k_read_ms"]), sum(cols["k_done"]))
        self.assertEqual(len(s["v_read_ms"]), sum(cols["v_done"]))
        return s

    def test_stop_at_the_next_layer_keeps_impress_output(self):
        with tempfile.TemporaryDirectory() as td:
            w = _World(td)
            a, b = self._impress(w), self._hyper(w)
            s = self._check(w, a, b)
            self.assertEqual({r["mode"] for r in b.selector.log}, {"consensus", "fallback"})
            self.assertEqual(s["truncation"], "reactive")

    def test_everything_prefetched_keeps_impress_output_and_reads_the_rest(self):
        with tempfile.TemporaryDirectory() as td:
            w = _World(td)
            a, b = self._impress(w), self._hyper(w, _Held)
            s = self._check(w, a, b)
            cols = s["layers"]
            self.assertEqual(cols["ended"][1:], [h2.EXHAUSTED] * (w.L - 1))
            self.assertEqual(cols["v_done"][1:], cols["predicted_chunks"][1:])
            self.assertEqual(s["reads_done_after_stop"], 0)
            self.assertGreater(s["hit_tokens"], 0)
            self.assertGreater(s["prefetched_bytes"], 0)
            # a layer whose prediction covers what it needs reads nothing synchronously
            self.assertTrue(any(cols["k_missing"][li] == 0 and cols["v_missing"][li] == 0 for li in range(1, w.L)))

    def test_partial_prefetch_reads_the_missing_kind_only(self):
        with tempfile.TemporaryDirectory() as td:
            w = _World(td)
            a, b = self._impress(w), self._hyper(w, _HeldAfter)
            s = self._check(w, a, b)
            cols = s["layers"]
            self.assertEqual(cols["k_done"][1:], [2] * (w.L - 1))             # K0, V0, K1: a K chunk without its V chunk
            self.assertEqual(cols["v_done"][1:], [1] * (w.L - 1))
            self.assertEqual(cols["ended"][1:], [h2.STOPPED] * (w.L - 1))
            self.assertEqual(cols["reads_done_after_stop"][1:], [1] * (w.L - 1))        # the read in flight
            self.assertEqual(cols["reads_done_in_window"][1:], [2] * (w.L - 1))
            self.assertEqual(cols["chunks_done_in_window"][1:], [1] * (w.L - 1))
            for li in range(1, w.L):
                if cols["k_wasted"][li] == 0:                                 # the second K chunk is needed: only its V is read
                    self.assertEqual(cols["v_missing"][li] - cols["k_missing"][li], 1)
            self.assertTrue(any(cols["k_wasted"][li] == 0 for li in range(1, w.L)))

    def test_two_runs_give_the_same_output(self):
        with tempfile.TemporaryDirectory() as td:
            w = _World(td)
            self._same_outcome(self._hyper(w), self._hyper(w, _Held))

    def test_constructor_checks(self):
        with tempfile.TemporaryDirectory() as td:
            w = _World(td)
            ctx = w.context()
            with self.assertRaises(ValueError):
                h2.HyperInferSelector(w.runner, ctx, kv_prefetch="sync", prefetch_executor=self.pool)
            with self.assertRaises(ValueError):
                h2.HyperInferSelector(w.runner, ctx, kv_prefetch="async")       # no worker
            ctx.reader.close()

    def test_a_request_that_stops_early_leaves_no_job_running(self):
        with tempfile.TemporaryDirectory() as td:
            w = _World(td)
            ctx = w.context()
            sel = h2.HyperInferSelector(w.runner, ctx, kv_prefetch="async", prefetch_executor=self.pool)
            from unittest import mock
            with mock.patch("mmimpress.impress_official.ml.apply_rotary_pos_emb", side_effect=lambda query, key, cos, sin: (query, key)):
                with sel:
                    w.layers[0](hidden_states=w.hidden[0], position_embeddings=(None, None))
                    job = sel._job
                    self.assertIsNotNone(job)
            self.assertTrue(job.future.done())
            self.assertTrue(job.stop)
            self.assertIsNone(sel._job)
            BIAS.clear()
            ctx.reader.close()


if __name__ == "__main__":
    unittest.main()

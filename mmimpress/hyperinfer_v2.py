"""HyperInfer's layer-aware KV prefetching with the paper's reactive truncation.

Paper: HyperInfer (IEEE TC 2026), IV-C and Fig. 10(c).  The prefetch of layer
l+1 lives inside layer l's compute time: when the GPU finishes layer l no new
read is issued, the token selection of layer l+1 runs, and only the KV the
selection needs and the prefetch did not bring is read.

``mmimpress/hyperinfer.py`` (the port measured on 2026-11-24, not modified)
cut the prefetch the way the public code does: before reading, with a time
budget and an estimated read time per chunk, and the next layer then waited
for the whole job.  This module changes only how the prefetch stops:

  worker       for layer l+1, started when layer l's hook ends: (1) the probe
               block, (2) the predicted chunks in ascending chunk id (stored
               order = descending importance), one read at a time, K chunk
               then V chunk.  Before every chunk read the worker asks, under
               the job's lock, whether the stop was set.
  stop         at the start of layer l+1's hook the request thread waits for
               the GPU (``torch.cuda.synchronize()`` -- the real completion of
               layer l, not an estimate) and sets the stop under the same
               lock.  A read that was already started is finished; no read is
               started afterwards.
  P            the K chunks and the V chunks whose read finished.
  wait         the probe block is always read by the worker; if it is not
               there when the selection needs it, the request thread waits
               for it (and only for it).  The read in flight ends while the
               selection runs; the request thread joins the worker right
               before its own first K/V read of the layer.  Both waits are
               inside the TTFT and recorded separately.
  recovery     the IMPRESS reads, for the chunks the selection needs minus P
               (per kind: a chunk whose K was prefetched and whose V was not
               reads only V).

No time budget, no chunk time constant, no half rule; the first job (layer 1)
follows the same rule.  Prediction, read unit, selection, fallback, store
format, mask and output (only G_l is used) are those of the earlier port and
of ``mmimpress.impress_official`` (not modified); the pure functions and the
debug guard are imported from ``mmimpress.hyperinfer``.

``kv_prefetch="off"`` runs ``OfficialImpressSelector`` with its next-layer
probe read-ahead, untouched.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import torch

from mmimpress import hyperinfer as h1
from mmimpress import impress_official as io_
from mmimpress.store import IOCounter, byte_range, chunk_span

KV_PREFETCH = ("off", "async")
KINDS = ("k", "v")
STOPPED, EXHAUSTED = "stop", "exhausted"

layer_position_maps = h1.layer_position_maps
predicted_chunks = h1.predicted_chunks
chunk_token_mask = h1.chunk_token_mask
read_one_chunk = h1.read_one_chunk
position_maps_for = h1.position_maps_for
UnreadRowGuard = h1.UnreadRowGuard
poison_visual_rows = h1.poison_visual_rows


# ===================================================================== job
class PrefetchJob:
    """One layer's prefetch: what the worker read and when it was told to stop.

    ``may_start`` (worker) and ``request_stop`` (request thread) take the same
    lock and read the clock inside it, so a read that was allowed to start has
    a decision time earlier than the stop time, and no read is allowed after
    the stop.
    """

    def __init__(self, layer, keep):
        self.layer, self.keep = int(layer), keep
        self.lock = threading.Lock()
        self.stop = False
        self.stop_at = None
        self.probe_ready = threading.Event()
        self.block = None
        self.counter = IOCounter()
        self.predicted = None
        self.done = {"k": {}, "v": {}}       # kind -> {chunk id: rows}
        self.reads = []                      # (kind, chunk id, decided_at, begin_at, end_at)
        self.refused_at = None               # when the worker saw the stop
        self.started_at = self.probe_done_at = self.finished_at = None
        self.ended = None                    # STOPPED | EXHAUSTED
        self.future = None

    def request_stop(self):
        with self.lock:
            if not self.stop:
                self.stop = True
                self.stop_at = time.perf_counter()
            return self.stop_at

    def may_start(self):
        """Decision time of a read that may start, or None after the stop."""
        with self.lock:
            if self.stop:
                return None
            return time.perf_counter()


def run_prefetch_job(job, reader, position_map, chunk_size, read_probe=io_.read_probe_block, read_chunk=read_one_chunk):
    """Worker body.  The probe block is read whatever the stop says (the
    selection cannot run without it); chunk reads are started one at a time
    and only while the stop is not set."""
    job.started_at = time.perf_counter()
    try:
        job.block = read_probe(reader, job.layer, job.counter)
    finally:
        job.probe_done_at = time.perf_counter()
        job.probe_ready.set()
    job.predicted = predicted_chunks(job.keep, position_map, chunk_size)
    job.ended = EXHAUSTED
    for cid in job.predicted:
        for kind in KINDS:
            decided = job.may_start()
            if decided is None:
                job.ended = STOPPED
                job.refused_at = time.perf_counter()
                break
            begin = time.perf_counter()
            rows = read_chunk(reader, job.layer, kind, cid, job.counter)
            end = time.perf_counter()
            job.done[kind][int(cid)] = rows
            job.reads.append((kind, int(cid), decided, begin, end))
        if job.ended == STOPPED:
            break
    job.finished_at = time.perf_counter()
    return job


# ================================================================== reading
class _Reader:
    """The store reader as the IMPRESS selector sees it.  The first K/V read
    of a layer joins that layer's prefetch job; chunks the job brought are
    copied from its buffer, the others are read from the store (adjacent ones
    in one pread, like the base reader).  Without a buffer for the kind the
    base reader is called as is."""

    def __init__(self, base, owner):
        self._base = base
        self._owner = owner

    def __getattr__(self, name):
        return getattr(self._base, name)

    def _assemble(self, layer, kind, cids, buf, counter, max_gap=0):
        base, m = self._base, self._base.meta
        vn, H, hd, cs = (int(m["v_token_num"]), int(m["num_heads"]),
                         int(m["head_dim"]), int(m["chunk_size"]))
        spans = [chunk_span(c, vn, cs) for c in cids]
        start, total = {}, 0
        for c, (s, e) in zip(cids, spans):
            start[c] = total
            total += e - s
        out = np.empty((total, H, hd), dtype=base.np_dtype)
        t0 = time.perf_counter()
        miss = [c for c in cids if c not in buf]
        for c, (s, e) in zip(cids, spans):
            if c in buf:
                out[start[c]:start[c] + (e - s)] = buf[c]
        t1 = time.perf_counter()
        if miss:
            row_bytes = H * hd * base.itemsize
            ranges = [byte_range(*chunk_span(c, vn, cs), H, hd, base.itemsize) for c in miss]
            blobs = base._read(layer, kind, ranges, counter, kind, len(miss), max_gap)
            for off, blob in blobs:
                first = (off // row_bytes) // cs
                n = len(blob) // row_bytes
                out[start[first]:start[first] + n] = np.frombuffer(
                    blob, dtype=base.np_dtype).reshape(n, H, hd)
        t2 = time.perf_counter()
        self._owner._note_read(layer, kind, cids, miss, t1 - t0, t2 - t1)
        return spans, out

    def read_chunks(self, layer, kind, cids, counter=None, max_gap=0):
        buf = self._owner._begin_reads(layer).get(kind)
        if not buf:
            t0 = time.perf_counter()
            out = self._base.read_chunks(layer, kind, cids, counter, max_gap)
            ids = sorted(set(int(c) for c in cids))
            self._owner._note_read(layer, kind, ids, ids, 0.0, time.perf_counter() - t0)
            return out
        cids = sorted(set(int(c) for c in cids))
        spans, out = self._assemble(layer, kind, cids, buf, counter, max_gap)
        rows = torch.cat([torch.arange(s, e) for s, e in spans])
        return rows, torch.from_numpy(out)

    def read_full(self, layer, kind, counter=None):
        m = self._base.meta
        every = list(range(int(m["n_chunks_per_layer"])))
        buf = self._owner._begin_reads(layer).get(kind)
        if not buf:
            t0 = time.perf_counter()
            out = self._base.read_full(layer, kind, counter)
            self._owner._note_read(layer, kind, every, every, 0.0, time.perf_counter() - t0)
            return out
        _, out = self._assemble(layer, kind, every, buf, counter)
        return torch.from_numpy(out)


class _Context:
    def __init__(self, ctx, owner):
        self._ctx = ctx
        self.meta = ctx.meta
        self.reader = _Reader(ctx.reader, owner)
        self.cache = h1._Cache(ctx.cache, owner)

    def __getattr__(self, name):
        return getattr(self._ctx, name)


# ================================================================= selector
class HyperInferSelector(io_.OfficialImpressSelector):
    """IMPRESS's per-layer selector plus the next-layer KV prefetch that stops
    when the current layer's GPU work is done.

    kv_prefetch
      "off"    the parent class, with its next-layer probe read-ahead; nothing
               here runs.
      "async"  layer l+1's job is submitted to the worker when layer l's hook
               ends and stopped at the start of layer l+1's hook.
    """

    def __init__(self, runner, ctx, *, kv_prefetch="async", position_maps=None,
                 ratio=io_.KEEP_RATIO, probe=len(io_.PROBE_HEADS), sim_threshold=io_.SIM_THRESHOLD,
                 counter=None, prefetch_executor=None):
        if kv_prefetch not in KV_PREFETCH:
            raise ValueError(f"kv_prefetch must be one of {KV_PREFETCH}, got {kv_prefetch!r}")
        self.kv_prefetch = kv_prefetch
        on = kv_prefetch != "off"
        super().__init__(runner, ctx, ratio=ratio, probe=probe, sim_threshold=sim_threshold,
                         counter=counter, prefetch="none" if on else "next_layer",
                         prefetch_executor=prefetch_executor)
        if not on:
            return
        if prefetch_executor is None:
            raise ValueError("kv_prefetch='async' needs the server's prefetch worker")
        self._base_ctx = ctx
        self._maps = position_maps if position_maps is not None else layer_position_maps(ctx.meta["order"])
        self.ctx = _Context(ctx, self)
        self._job = None                # the job submitted for the next layer
        self._cur = None                # the job of the layer whose hook is running, until joined
        self._buffers = {}              # layer -> {"k": {chunk id: rows}, "v": {...}}
        self._submit_at = None
        self._gpu_load_s = {}
        self._reads = {}                # layer -> kind -> {"needed", "missing", "copy_s", "sync_s"}
        self._probe_done_at = None
        self._first_read_at = {}
        self.layers = {}                # layer -> per-layer record
        self.jobs = {}                  # layer -> joined PrefetchJob

    # ------------------------------------------------------------ plumbing
    def __exit__(self, *exc):
        if self.kv_prefetch != "off":
            for job in (self._job, self._cur):      # only when the request did not reach the layer
                if job is not None:
                    job.request_stop()
                    try:
                        job.future.result()
                        self._fold(job.counter)
                    except Exception:               # noqa: BLE001  (the request's own error is the one to report)
                        pass
            self._job = self._cur = None
            self._buffers.clear()
        return super().__exit__(*exc)

    def _note_read(self, layer, kind, needed, missing, copy_s, sync_s):
        self._reads.setdefault(layer, {})[kind] = {"needed": list(needed), "missing": list(missing),
                                                   "copy_s": copy_s, "sync_s": sync_s}

    # --------------------------------------------------------------- stages
    def _submit(self, li):
        nxt = li + 1
        if nxt >= self._base_ctx.meta["num_layers"]:
            return
        job = PrefetchJob(nxt, self.keep_masks[li])
        job.future = self._executor.submit(run_prefetch_job, job, self._base_ctx.reader, self._maps[nxt - 1],
                                           self._base_ctx.meta["chunk_size"])
        self._job = job
        self._submit_at = time.perf_counter()
        self.layers.setdefault(li, {})["submitted_at"] = self._submit_at

    def _before_stop(self, li, job):
        """Nothing in a request.  Tests hold the stop here to drive the worker."""

    def _stop(self, li, h):
        """Start of layer li > 0: the previous layer's GPU work is waited for,
        then the job of this layer is told to start no further read."""
        job, self._job = self._job, None
        rec = self.layers.setdefault(li, {})
        if job is None:
            return
        assert job.layer == li, (job.layer, li)
        self._before_stop(li, job)
        t0 = time.perf_counter()
        if h.is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        stop_at = job.request_stop()
        rec.update(window_s=t1 - self._submit_at, sync_wait_s=t1 - t0, stop_at=stop_at, stop_call_s=time.perf_counter() - t1)
        self._cur = job

    def _begin_reads(self, li):
        """First K/V read of the layer: the worker is joined (its read in
        flight, if any, ends here) and what it read becomes the buffer."""
        if li not in self._first_read_at:
            self._first_read_at[li] = time.perf_counter()
        job = self._cur
        if job is not None and job.layer == li:
            t0 = time.perf_counter()
            done = job.future.done()
            job.future.result()
            wait = time.perf_counter() - t0
            self.read_blocked_seconds += wait
            self._fold(job.counter)
            self._buffers[li] = job.done
            self.jobs[li] = job
            self._cur = None
            self.layers.setdefault(li, {}).update(inflight_wait_s=wait, worker_done_on_arrival=bool(done))
        return self._buffers.get(li) or {}

    # ------------------------------------------------- parent's entry points
    def _probe_block(self, li):
        if self.kv_prefetch == "off":
            return super()._probe_block(li)
        job = self._cur
        rec = self.layers.setdefault(li, {})
        if job is not None and job.layer == li:
            t0 = time.perf_counter()
            ready = job.probe_ready.is_set()
            job.probe_ready.wait()
            wait = time.perf_counter() - t0
            if job.block is None:       # the probe read failed: raise the worker's error here
                job.future.result()
                raise RuntimeError("prefetch job ended without a probe block")
            self.read_blocked_seconds += wait
            rec.update(probe_wait_s=wait, probe_ready_on_arrival=bool(ready))
            self.probe_log.append({"layer": li, "source": "prefetch", "read_s": job.probe_done_at - job.started_at,
                                   "blocked_s": wait, "wait_s": wait, "ready_on_arrival": bool(ready)})
            block = job.block
        else:                           # layer 0: read in the hook, like the parent
            t0 = time.perf_counter()
            block = io_.read_probe_block(self._base_ctx.reader, li, self.io)
            blocked = time.perf_counter() - t0
            self.read_blocked_seconds += blocked
            self.probe_log.append({"layer": li, "source": "hook", "read_s": blocked,
                                   "blocked_s": blocked, "wait_s": 0.0, "ready_on_arrival": None})
            rec.update(probe_sync_s=blocked)
        self._probe_done_at = time.perf_counter()
        return block

    def _run(self, li, layer, h, kwargs):
        if self.kv_prefetch == "off":
            return super()._run(li, layer, h, kwargs)
        self._stop(li, h)
        super()._run(li, layer, h, kwargs)
        self._begin_reads(li)           # no-op: the parent has read; kept so that a job is never left unjoined
        rec = self.layers.setdefault(li, {})
        # selection = from the probe block's arrival to the first K/V read request
        rec["select_s"] = self._first_read_at[li] - self._probe_done_at
        self._buffers.pop(li, None)     # prefetched chunks the selection did not need are dropped here
        self._submit(li)

    # -------------------------------------------------------------- summary
    def hyperinfer_stats(self):
        """Per-layer and per-request figures; call after the request (outside every timer)."""
        if self.kv_prefetch == "off":
            return {"kv_prefetch": "off"}
        m = self._base_ctx.meta
        L, vn, cs, nc = int(m["num_layers"]), int(m["v_token_num"]), int(m["chunk_size"]), int(m["n_chunks_per_layer"])
        row_bytes = int(m["num_heads"]) * int(m["head_dim"]) * self._base_ctx.reader.itemsize
        n_sep = self.n_sep
        modes = {r["layer"]: r["mode"] for r in self.log}
        orders = [torch.as_tensor(o, dtype=torch.long) for o in m["order"]]

        def rows_of(ids):
            return sum(chunk_span(c, vn, cs)[1] - chunk_span(c, vn, cs)[0] for c in ids)

        def image_set(li):              # selected image tokens of layer li, original indices
            keep = self.keep_masks[li].clone()
            keep[:n_sep] = False
            out = torch.zeros(vn, dtype=torch.bool)
            out[orders[li][keep]] = True
            return out

        names = ("fallback", "window_ms", "sync_wait_ms", "probe_wait_ms", "inflight_wait_ms", "probe_sync_ms", "select_ms",
                 "sync_read_ms", "copy_ms", "gpu_load_ms", "probe_ready", "worker_done", "ended", "predicted_chunks",
                 "k_done", "v_done", "prefetched_chunks", "reads_done_in_window", "chunks_done_in_window", "reads_done_after_stop",
                 "late_starts", "begin_after_stop_ms", "worker_start_delay_ms", "needed_chunks", "k_hit", "v_hit", "k_missing",
                 "v_missing", "k_wasted", "v_wasted", "prefetched_tokens", "used_tokens", "hit_tokens", "prefetched_bytes",
                 "wasted_bytes", "sync_bytes", "worker_probe_ms", "worker_k_ms", "worker_v_ms", "jaccard_next",
                 "jaccard_next_both_consensus")
        cols = {k: [] for k in names}
        k_ms, v_ms, probe_ms = [], [], []
        sets = [image_set(li) for li in range(L)]
        for li in range(L):
            rec, job, reads = self.layers.get(li, {}), self.jobs.get(li), self._reads[li]
            keep = self.keep_masks[li]
            have = {kind: sorted(job.done[kind]) if job else [] for kind in KINDS}
            both = sorted(set(have["k"]) & set(have["v"]))
            needed = reads["k"]["needed"]
            assert needed == reads["v"]["needed"], li
            pmask = chunk_token_mask(both, vn, cs)
            stop_at = rec.get("stop_at")
            job_reads = job.reads if job else []
            in_window = [r for r in job_reads if r[4] <= stop_at]
            ended_k = {r[1] for r in in_window if r[0] == "k"}
            ended_v = {r[1] for r in in_window if r[0] == "v"}
            cols["fallback"].append(modes[li] == "fallback")
            cols["window_ms"].append(1e3 * rec["window_s"] if job else None)
            cols["sync_wait_ms"].append(1e3 * rec["sync_wait_s"] if job else None)
            cols["probe_wait_ms"].append(1e3 * rec.get("probe_wait_s", 0.0))
            cols["inflight_wait_ms"].append(1e3 * rec.get("inflight_wait_s", 0.0))
            cols["probe_sync_ms"].append(1e3 * rec.get("probe_sync_s", 0.0))
            cols["select_ms"].append(1e3 * rec["select_s"])
            cols["sync_read_ms"].append(1e3 * sum(reads[k]["sync_s"] for k in KINDS))
            cols["copy_ms"].append(1e3 * sum(reads[k]["copy_s"] for k in KINDS))
            cols["gpu_load_ms"].append(1e3 * self._gpu_load_s.get(li, 0.0))
            cols["probe_ready"].append(rec.get("probe_ready_on_arrival"))
            cols["worker_done"].append(rec.get("worker_done_on_arrival"))
            cols["ended"].append(job.ended if job else None)
            cols["predicted_chunks"].append(len(job.predicted) if job else 0)
            cols["k_done"].append(len(have["k"]))
            cols["v_done"].append(len(have["v"]))
            cols["prefetched_chunks"].append(len(both))
            cols["reads_done_in_window"].append(len(in_window))
            cols["chunks_done_in_window"].append(len(ended_k & ended_v))
            cols["reads_done_after_stop"].append(len(job_reads) - len(in_window))
            cols["late_starts"].append(sum(r[2] > stop_at for r in job_reads) if job else 0)
            cols["begin_after_stop_ms"].append(max((1e3 * (r[3] - stop_at) for r in job_reads), default=None) if job else None)
            cols["worker_start_delay_ms"].append(1e3 * (job.started_at - self.layers[li - 1]["submitted_at"]) if job else None)
            cols["needed_chunks"].append(len(needed))
            for kind in KINDS:
                missing = reads[kind]["missing"]
                cols[f"{kind}_hit"].append(len(needed) - len(missing))
                cols[f"{kind}_missing"].append(len(missing))
                cols[f"{kind}_wasted"].append(len([c for c in have[kind] if c not in set(needed)]))
            cols["prefetched_tokens"].append(int(pmask.sum()))
            cols["used_tokens"].append(int(keep.sum()))
            cols["hit_tokens"].append(int((pmask & keep).sum()))
            cols["prefetched_bytes"].append(sum(rows_of(have[k]) for k in KINDS) * row_bytes)
            cols["wasted_bytes"].append(sum(rows_of([c for c in have[k] if c not in set(needed)]) for k in KINDS) * row_bytes)
            cols["sync_bytes"].append(sum(rows_of(reads[k]["missing"]) for k in KINDS) * row_bytes)
            cols["worker_probe_ms"].append(1e3 * (job.probe_done_at - job.started_at) if job else None)
            cols["worker_k_ms"].append(1e3 * sum(r[4] - r[3] for r in job_reads if r[0] == "k") if job else None)
            cols["worker_v_ms"].append(1e3 * sum(r[4] - r[3] for r in job_reads if r[0] == "v") if job else None)
            if job:
                probe_ms.append(1e3 * (job.probe_done_at - job.started_at))
                k_ms += [1e3 * (r[4] - r[3]) for r in job_reads if r[0] == "k"]
                v_ms += [1e3 * (r[4] - r[3]) for r in job_reads if r[0] == "v"]
            if li + 1 < L:
                a, b = sets[li], sets[li + 1]
                union = int((a | b).sum())
                cols["jaccard_next"].append(int((a & b).sum()) / union if union else None)
                cols["jaccard_next_both_consensus"].append(modes[li] != "fallback" and modes[li + 1] != "fallback")
        total = lambda k: float(sum(x for x in cols[k] if x is not None))      # noqa: E731
        io_ms = 1e3 * float(self.io.seconds)
        critical = total("probe_wait_ms") + total("inflight_wait_ms") + total("probe_sync_ms") + total("sync_read_ms")
        return {
            "kv_prefetch": self.kv_prefetch, "truncation": "reactive", "layers": cols, "chunks_per_layer": nc,
            "k_read_ms": k_ms, "v_read_ms": v_ms, "probe_read_ms": probe_ms,
            "window_ms": total("window_ms"), "sync_wait_ms": total("sync_wait_ms"),
            "probe_wait_ms": total("probe_wait_ms"), "inflight_wait_ms": total("inflight_wait_ms"),
            "probe_sync_ms": total("probe_sync_ms"), "select_ms": total("select_ms"), "sync_read_ms": total("sync_read_ms"),
            "copy_ms": total("copy_ms"), "gpu_load_ms": total("gpu_load_ms"),
            "critical_io_ms": critical, "total_io_ms": io_ms, "io_ratio": critical / io_ms if io_ms > 0 else None,
            "late_starts": int(total("late_starts")), "jobs": len(self.jobs),
            "jobs_stopped": sum(e == STOPPED for e in cols["ended"]), "jobs_exhausted": sum(e == EXHAUSTED for e in cols["ended"]),
            "reads_done_in_window": int(total("reads_done_in_window")), "reads_done_after_stop": int(total("reads_done_after_stop")),
            "prefetched_tokens": int(total("prefetched_tokens")), "hit_tokens": int(total("hit_tokens")),
            "used_tokens_after_layer0": int(sum(cols["used_tokens"][1:])),
            "prefetched_bytes": int(total("prefetched_bytes")), "wasted_bytes": int(total("wasted_bytes")),
            "sync_bytes": int(total("sync_bytes")), "predicted_chunks": int(total("predicted_chunks")),
        }


def idle_synchronize_cost(n=2000):
    """Seconds one ``torch.cuda.synchronize()`` takes when the GPU has nothing queued
    (mean, p50, p99 over n calls) -- the price of asking, apart from the waiting itself."""
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    out = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        torch.cuda.synchronize()
        out[i] = time.perf_counter() - t0
    return {"calls": n, "mean_s": float(out.mean()), "p50_s": float(np.median(out)), "p99_s": float(np.percentile(out, 99))}


# ================================================================== request
@torch.no_grad()
def request_hyperinfer(server, ctx, suffix_ids, *, kv_prefetch="async", ratio=io_.KEEP_RATIO,
                       probe=len(io_.PROBE_HEADS), sim_threshold=io_.SIM_THRESHOLD, cold=True,
                       selector_out=None, debug_unread_nan=False, guard_out=None):
    """Serve one follow-up turn with HyperInfer.

    Timer boundaries are those of ``impress_official.request_official`` and of
    the earlier port: cold-cache eviction, the request counter, the prefetch
    worker and the store's position tables are prepared before the timer; TTFT
    ends right after the first output token.
    """
    from mmimpress.serve import BIAS
    BIAS.clear()
    if cold:
        ctx.reader.drop_all()
    counter = IOCounter()
    dev = server.runner.model.device
    suffix_ids = suffix_ids.to(dev)
    executor = server._prefetch_pool()
    maps = position_maps_for(ctx) if kv_prefetch != "off" else None
    guard = UnreadRowGuard(ctx.meta) if debug_unread_nan else None

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    sel = HyperInferSelector(server.runner, ctx, kv_prefetch=kv_prefetch, position_maps=maps,
                             ratio=ratio, probe=probe, sim_threshold=sim_threshold, counter=counter,
                             prefetch_executor=executor)
    if selector_out is not None:
        selector_out.append(sel)
    cache = ctx.cache.new_request()
    if guard is not None:
        poison_visual_rows(ctx.cache, ctx.meta)
        guard.__enter__()
    try:
        with sel:
            answer, first_token_id, timing = server._decode(cache, suffix_ids, ctx.meta["prefix_len"])
    finally:
        if guard is not None:
            guard.__exit__(None, None, None)
    BIAS.clear()
    del cache
    if guard is not None and guard_out is not None:
        guard_out.append(guard)
    return {
        "answer": answer,
        "first_token_id": first_token_id,
        "ttft": timing["first_token_at"] - t0,
        "decode_latency": timing["decode_ms"] / 1e3,
        "e2e_latency": timing["finished_at"] - t0,
        "prefill_ms": timing["prefill_ms"],
        "generated_tokens": timing["generated_tokens"],
        "generated_token_ids": timing["generated_token_ids"],
        "generated_token_count": timing["generated_token_count"],
        "n_score_rows": int(suffix_ids.numel()),
        "io": counter.summary(),
        "core_started_at_s": t0,
        "first_token_at_s": timing["first_token_at"],
        "model_finished_at_s": timing["finished_at"],
        "postprocess_finished_at_s": time.perf_counter(),
        "read_wait": sel.read_wait_stats(),
        **sel.stats(),
    }

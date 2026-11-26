"""HyperInfer's layer-aware KV prefetching on top of the IMPRESS port.

Reference: https://github.com/ISCS-ZJU/HyperInfer, commit 91df7a2
(``h2o_flexgen/flexgen/flex_opt_fast_pcache.py`` and ``my_pcache_fast.py``).
Nothing is copied from that repository; ``tests/test_hyperinfer.py`` calls the
public ``PrefixKVLayer.prefetch_by_chunk`` with the same inputs as the
functions below.  Source comparison table: docs/hyperinfer_port_contract.md.

What is added to ``mmimpress.impress_official`` (which is not modified):

  prediction   the tokens layer l uses (selected image tokens + separators)
               are the prediction for layer l+1 -- every one of them;
  unit         the prediction is moved to layer l+1's stored order, turned
               into chunk ids, sorted ascending; whole K and V chunks are
               read, one chunk at a time, K then V;
  truncation   the public time budget: (compute time + selection time) of
               the previous layer, minus an estimated time per K chunk and
               per V chunk, cut before the first chunk that does not fit; a
               budget of 0 (only the first request of a process) reads the
               first half of the predicted chunks.  The budget lives in a
               ``PrefetchState`` that outlives the request, as in the public
               code;
  verify       at layer l+1 the IMPRESS selection runs unchanged; chunks it
               needs that were prefetched are taken from the buffer, the rest
               is read synchronously with the IMPRESS reads;
  discard      prefetched chunks the selection does not need are dropped
               (never moved to the GPU); the mask is the IMPRESS mask, so the
               attention sees exactly what IMPRESS's attention sees;
  worker       one background worker; it reads layer l+1's probe block first
               (the IMPRESS port's read-ahead, kept), then the chunks.  It does
               disk reads and host arrays only -- device transfer and the
               cache writes stay on the request thread, the rule of every
               asynchronous arm of this repository;
  wait         layer l+1 waits for its job before anything else; the wait is
               inside the request's TTFT.

``kv_prefetch="off"`` runs ``OfficialImpressSelector`` with the next-layer
probe read-ahead, untouched: no override below is active.

The selection itself is not re-implemented: ``HyperInferSelector`` runs the
parent's ``_run`` and only changes where its reads come from (``_Reader``).
"""
from __future__ import annotations

import time

import numpy as np
import torch
import transformers.models.llama.modeling_llama as ml

from mmimpress import impress_official as io_
from mmimpress.store import IOCounter, byte_range, chunk_span, merge_ranges

KV_PREFETCH = ("off", "async", "sync")
RULE_BUDGET, RULE_HALF = "budget", "half"


# ===================================================================== pure
def layer_position_maps(orders):
    """Per layer l < L-1: stored position at layer l -> stored position at l+1.

    orders[l][r] is the original index of the token stored at position r of
    layer l (``meta["order"]``).
    """
    out = []
    for cur, nxt in zip(orders[:-1], orders[1:]):
        cur = torch.as_tensor(cur, dtype=torch.long)
        out.append(io_.reverse_index(nxt)[cur])
    return out


def predicted_chunks(keep, position_map, chunk_size):
    """Chunk ids of layer l+1 that hold a token layer l used, ascending.

    keep: (v_num,) bool over layer l's stored positions.
    """
    keep = torch.as_tensor(keep, dtype=torch.bool)
    pos = position_map[keep]
    return torch.unique(pos // int(chunk_size)).tolist()


def truncate_chunks(chunk_ids, budget_s, k_chunk_s, v_chunk_s):
    """The public cut: (chunks to read, rule).

    budget > 0: walk the ascending ids, subtract one K and one V estimate per
    chunk, stop before the first chunk that does not fit.  Otherwise: the
    first half of the ids.
    """
    ids = [int(c) for c in chunk_ids]
    if budget_s > 0.0:
        remaining = float(budget_s)
        for i in range(len(ids)):
            est = 0.0
            est += k_chunk_s
            est += v_chunk_s
            if remaining - est >= 0.0:
                remaining -= est
            else:
                return ids[:i], RULE_BUDGET
        return ids, RULE_BUDGET
    return ids[:len(ids) // 2], RULE_HALF


def chunk_token_mask(chunk_ids, v_num, chunk_size):
    """(v_num,) bool: every stored position inside the given chunks."""
    mask = torch.zeros(int(v_num), dtype=torch.bool)
    for c in chunk_ids:
        s, e = chunk_span(int(c), int(v_num), int(chunk_size))
        mask[s:e] = True
    return mask


def missing_chunks(needed, have):
    """Chunks the selection needs that the prefetch did not bring."""
    have = set(int(c) for c in have)
    return [int(c) for c in sorted(set(int(c) for c in needed)) if int(c) not in have]


class PrefetchState:
    """What survives a request: the time budget (seconds) and the constants.

    One per arm and per process.  ``budget_s`` starts at 0 -- the first
    request's first prefetch uses the half rule -- and afterwards always holds
    (compute time + selection time) of the last layer that measured them.
    """

    def __init__(self, k_chunk_s, v_chunk_s):
        self.k_chunk_s = float(k_chunk_s)
        self.v_chunk_s = float(v_chunk_s)
        if not (self.k_chunk_s > 0.0 and self.v_chunk_s > 0.0):
            raise ValueError("chunk read time constants must be positive")
        self.budget_s = 0.0
        self.requests = 0


# ================================================================== reading
def read_one_chunk(reader, layer, kind, cid, counter=None):
    """One whole chunk of one layer's K or V file, one pread: (rows, H, hd)."""
    m = reader.meta
    H, hd = int(m["num_heads"]), int(m["head_dim"])
    s, e = chunk_span(int(cid), int(m["v_token_num"]), int(m["chunk_size"]))
    off, ln = byte_range(s, e, H, hd, reader.itemsize)
    blobs = reader._read(layer, kind, [(off, ln)], counter, kind, 1)
    return np.frombuffer(blobs[0][1], dtype=reader.np_dtype).reshape(e - s, H, hd)


class _Reader:
    """The store reader as the IMPRESS selector sees it, with a prefetch buffer.

    ``read_chunks`` / ``read_full`` return what the base reader returns; the
    chunks present in the buffer of that layer are copied from it, the others
    are read from the store (adjacent ones in one pread, like the base
    reader).  Without a buffer for the layer the base reader is called as is.
    """

    def __init__(self, base, owner):
        self._base = base
        self._owner = owner

    def __getattr__(self, name):
        return getattr(self._base, name)

    def _assemble(self, layer, kind, cids, counter, max_gap=0):
        base, m = self._base, self._base.meta
        vn, H, hd, cs = (int(m["v_token_num"]), int(m["num_heads"]),
                         int(m["head_dim"]), int(m["chunk_size"]))
        buf = self._owner._buffers.get(layer) or {}
        which = 0 if kind == "k" else 1
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
                out[start[c]:start[c] + (e - s)] = buf[c][which]
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
        if not self._owner._buffers.get(layer):
            t0 = time.perf_counter()
            out = self._base.read_chunks(layer, kind, cids, counter, max_gap)
            ids = sorted(set(int(c) for c in cids))
            self._owner._note_read(layer, kind, ids, ids, 0.0, time.perf_counter() - t0)
            return out
        cids = sorted(set(int(c) for c in cids))
        spans, out = self._assemble(layer, kind, cids, counter, max_gap)
        rows = torch.cat([torch.arange(s, e) for s, e in spans])
        return rows, torch.from_numpy(out)

    def read_full(self, layer, kind, counter=None):
        m = self._base.meta
        every = list(range(int(m["n_chunks_per_layer"])))
        if not self._owner._buffers.get(layer):
            t0 = time.perf_counter()
            out = self._base.read_full(layer, kind, counter)
            self._owner._note_read(layer, kind, every, every, 0.0, time.perf_counter() - t0)
            return out
        _, out = self._assemble(layer, kind, every, counter)
        return torch.from_numpy(out)


class _Cache:
    """The prefix cache with its two write calls timed (host clock, no extra
    device synchronisation: the copy from pageable host memory is blocking)."""

    def __init__(self, base, owner):
        self._base = base
        self._owner = owner

    def __getattr__(self, name):
        return getattr(self._base, name)

    def write(self, layer, kind, rows, vals):
        t0 = time.perf_counter()
        self._base.write(layer, kind, rows, vals)
        self._owner._gpu_load_s[layer] = self._owner._gpu_load_s.get(layer, 0.0) + time.perf_counter() - t0

    def write_full(self, layer, kind, block):
        t0 = time.perf_counter()
        self._base.write_full(layer, kind, block)
        self._owner._gpu_load_s[layer] = self._owner._gpu_load_s.get(layer, 0.0) + time.perf_counter() - t0


class _Context:
    def __init__(self, ctx, owner):
        self._ctx = ctx
        self.meta = ctx.meta
        self.reader = _Reader(ctx.reader, owner)
        self.cache = _Cache(ctx.cache, owner)

    def __getattr__(self, name):
        return getattr(self._ctx, name)


# ================================================================= selector
class HyperInferSelector(io_.OfficialImpressSelector):
    """IMPRESS's per-layer selector plus the next-layer KV prefetch.

    kv_prefetch
      "off"    the parent class, with its next-layer probe read-ahead; nothing
               here runs.
      "async"  the prefetch job of layer l+1 is submitted to the worker when
               layer l's hook ends.
      "sync"   the same job is executed on the request thread at that point
               (a check of the logic without thread timing).
    """

    def __init__(self, runner, ctx, *, state=None, kv_prefetch="async", position_maps=None,
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
        if state is None:
            raise ValueError("kv_prefetch needs a PrefetchState")
        if kv_prefetch == "async" and prefetch_executor is None:
            raise ValueError("kv_prefetch='async' needs the server's prefetch worker")
        self.state = state
        self._base_ctx = ctx
        self._maps = position_maps if position_maps is not None else layer_position_maps(ctx.meta["order"])
        self.ctx = _Context(ctx, self)
        self._buffers = {}              # layer -> {chunk id: (k rows, v rows)}
        self._job = None                # (layer, Future or result dict)
        self._resolved = {}             # layer -> probe block
        self._submit_at = None
        self._gpu_load_s = {}
        self._reads = {}                # layer -> {"needed": [...], "missing": [...], "copy_s", "sync_s"}
        self._probe_done_at = None
        self._first_read_at = {}
        self.layers = {}                # layer -> per-layer record
        self.jobs = {}                  # target layer -> job record

    # ------------------------------------------------------------ plumbing
    def __exit__(self, *exc):
        if self.kv_prefetch != "off":
            job, self._job = self._job, None
            if job is not None:         # only when the request did not reach the layer
                try:
                    result = job[1].result() if hasattr(job[1], "result") else job[1]
                    self._fold(result["counter"])
                except Exception:       # noqa: BLE001  (the request's own error is the one to report)
                    pass
            self._buffers.clear()
        return super().__exit__(*exc)

    def _note_read(self, layer, kind, needed, missing, copy_s, sync_s):
        if layer not in self._first_read_at:
            self._first_read_at[layer] = time.perf_counter() - copy_s - sync_s
        r = self._reads.setdefault(layer, {"needed": list(needed), "missing": list(missing),
                                           "copy_s": 0.0, "sync_s": 0.0})
        r["copy_s"] += copy_s
        r["sync_s"] += sync_s

    # ------------------------------------------------------------ the job
    def _prefetch_job(self, nxt, keep, budget_s):
        """Worker body: layer ``nxt``'s probe block, then its predicted chunks."""
        base = self._base_ctx.reader
        counter = IOCounter()
        t0 = time.perf_counter()
        block = io_.read_probe_block(base, nxt, counter)
        t1 = time.perf_counter()
        m = self._base_ctx.meta
        predicted = predicted_chunks(keep, self._maps[nxt - 1], m["chunk_size"])
        chosen, rule = truncate_chunks(predicted, budget_s, self.state.k_chunk_s, self.state.v_chunk_s)
        chunks = {}
        k_s = v_s = 0.0
        for cid in chosen:
            a = time.perf_counter()
            k = read_one_chunk(base, nxt, "k", cid, counter)
            b = time.perf_counter()
            v = read_one_chunk(base, nxt, "v", cid, counter)
            c = time.perf_counter()
            chunks[cid] = (k, v)
            k_s += b - a
            v_s += c - b
        t2 = time.perf_counter()
        return {"layer": nxt, "block": block, "chunks": chunks, "counter": counter,
                "predicted": predicted, "chosen": chosen, "rule": rule, "budget_s": float(budget_s),
                "probe_s": t1 - t0, "k_s": k_s, "v_s": v_s, "start_s": t0, "end_s": t2}

    def _submit_kv(self, li):
        nxt = li + 1
        if nxt >= self._base_ctx.meta["num_layers"]:
            return
        keep = self.keep_masks[li]
        budget = self.state.budget_s
        if self.kv_prefetch == "sync":
            self._job = (nxt, self._prefetch_job(nxt, keep, budget))
        else:
            self._job = (nxt, self._executor.submit(self._prefetch_job, nxt, keep, budget))
        self._submit_at = time.perf_counter()

    def _resolve(self, li, h):
        """Start of layer li > 0: previous layer's compute time, then the job."""
        job, self._job = self._job, None
        rec = self.layers.setdefault(li, {})
        if job is None:
            return
        assert job[0] == li, (job[0], li)
        if h.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        compute_s = t0 - self._submit_at
        future = job[1]
        if hasattr(future, "result"):
            ready = future.done()
            result = future.result()
        else:
            ready, result = True, future
        wait_s = time.perf_counter() - t0
        self.read_blocked_seconds += wait_s
        self._fold(result["counter"])
        self._buffers[li] = result["chunks"]
        self._resolved[li] = result["block"]
        # the public budget: (compute + selection) of the layer that just finished
        self.state.budget_s = compute_s + self.layers[li - 1]["select_s"]
        rec.update(compute_prev_s=compute_s, wait_s=wait_s, ready_on_arrival=bool(ready))
        self.jobs[li] = {k: result[k] for k in ("predicted", "chosen", "rule", "budget_s", "probe_s",
                                                "k_s", "v_s", "start_s", "end_s")}
        self.probe_log.append({"layer": li, "source": "prefetch", "read_s": result["probe_s"],
                               "blocked_s": wait_s, "wait_s": wait_s, "ready_on_arrival": bool(ready)})

    # ------------------------------------------------- parent's entry points
    def _probe_block(self, li):
        if self.kv_prefetch == "off":
            return super()._probe_block(li)
        if li in self._resolved:
            block = self._resolved.pop(li)
        else:                           # layer 0: read in the hook, like the parent
            t0 = time.perf_counter()
            block = io_.read_probe_block(self._base_ctx.reader, li, self.io)
            blocked = time.perf_counter() - t0
            self.read_blocked_seconds += blocked
            self.probe_log.append({"layer": li, "source": "hook", "read_s": blocked,
                                   "blocked_s": blocked, "wait_s": 0.0, "ready_on_arrival": None})
            self.layers.setdefault(li, {}).update(probe_sync_s=blocked)
        self._probe_done_at = time.perf_counter()
        return block

    def _run(self, li, layer, h, kwargs):
        if self.kv_prefetch == "off":
            return super()._run(li, layer, h, kwargs)
        self._resolve(li, h)
        super()._run(li, layer, h, kwargs)
        rec = self.layers.setdefault(li, {})
        # selection = from the probe block's arrival to the first K/V read request
        rec["select_s"] = self._first_read_at[li] - self._probe_done_at
        self._buffers.pop(li, None)     # prefetched chunks the selection did not need are dropped here
        self._submit_kv(li)

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

        def chunk_bytes(ids):
            return sum((chunk_span(c, vn, cs)[1] - chunk_span(c, vn, cs)[0]) for c in ids) * row_bytes * 2

        def image_set(li):              # selected image tokens of layer li, original indices
            keep = self.keep_masks[li].clone()
            keep[:n_sep] = False
            out = torch.zeros(vn, dtype=torch.bool)
            out[orders[li][keep]] = True
            return out

        cols = {k: [] for k in (
            "fallback", "wait_ms", "compute_prev_ms", "select_ms", "probe_sync_ms", "sync_read_ms", "copy_ms", "gpu_load_ms",
            "ready", "rule", "budget_ms", "predicted_chunks", "prefetched_chunks", "cut_chunks", "needed_chunks",
            "hit_chunks", "missing_chunks", "wasted_chunks", "prefetched_tokens", "used_tokens", "hit_tokens",
            "prefetched_bytes", "wasted_bytes", "sync_bytes", "worker_probe_ms", "worker_k_ms", "worker_v_ms",
            "jaccard_next", "jaccard_next_both_consensus")}
        sets = [image_set(li) for li in range(L)]
        for li in range(L):
            rec, job, reads = self.layers.get(li, {}), self.jobs.get(li), self._reads[li]
            keep = self.keep_masks[li]
            have = list(job["chosen"]) if job else []
            needed, missing = reads["needed"], reads["missing"]
            hit = [c for c in needed if c in set(have)]
            wasted = [c for c in have if c not in set(needed)]
            pmask = chunk_token_mask(have, vn, cs)
            cols["fallback"].append(modes[li] == "fallback")
            cols["wait_ms"].append(1e3 * rec.get("wait_s", 0.0))
            cols["compute_prev_ms"].append(1e3 * rec["compute_prev_s"] if "compute_prev_s" in rec else None)
            cols["select_ms"].append(1e3 * rec["select_s"])
            cols["probe_sync_ms"].append(1e3 * rec.get("probe_sync_s", 0.0))
            cols["sync_read_ms"].append(1e3 * reads["sync_s"])
            cols["copy_ms"].append(1e3 * reads["copy_s"])
            cols["gpu_load_ms"].append(1e3 * self._gpu_load_s.get(li, 0.0))
            cols["ready"].append(rec.get("ready_on_arrival"))
            cols["rule"].append(job["rule"] if job else None)
            cols["budget_ms"].append(1e3 * job["budget_s"] if job else None)
            cols["predicted_chunks"].append(len(job["predicted"]) if job else 0)
            cols["prefetched_chunks"].append(len(have))
            cols["cut_chunks"].append(len(job["predicted"]) - len(have) if job else 0)
            cols["needed_chunks"].append(len(needed))
            cols["hit_chunks"].append(len(hit))
            cols["missing_chunks"].append(len(missing))
            cols["wasted_chunks"].append(len(wasted))
            cols["prefetched_tokens"].append(int(pmask.sum()))
            cols["used_tokens"].append(int(keep.sum()))
            cols["hit_tokens"].append(int((pmask & keep).sum()))
            cols["prefetched_bytes"].append(chunk_bytes(have))
            cols["wasted_bytes"].append(chunk_bytes(wasted))
            cols["sync_bytes"].append(chunk_bytes(missing))
            cols["worker_probe_ms"].append(1e3 * job["probe_s"] if job else None)
            cols["worker_k_ms"].append(1e3 * job["k_s"] if job else None)
            cols["worker_v_ms"].append(1e3 * job["v_s"] if job else None)
            if li + 1 < L:
                a, b = sets[li], sets[li + 1]
                union = int((a | b).sum())
                cols["jaccard_next"].append(int((a & b).sum()) / union if union else None)
                cols["jaccard_next_both_consensus"].append(modes[li] != "fallback" and modes[li + 1] != "fallback")
        total = lambda k: float(sum(x for x in cols[k] if x is not None))      # noqa: E731
        io_ms = 1e3 * float(self.io.seconds)
        critical = total("wait_ms") + total("probe_sync_ms") + total("sync_read_ms")
        return {
            "kv_prefetch": self.kv_prefetch, "layers": cols, "chunks_per_layer": nc,
            "k_chunk_ms": 1e3 * self.state.k_chunk_s, "v_chunk_ms": 1e3 * self.state.v_chunk_s,
            "budget_after_request_ms": 1e3 * self.state.budget_s,
            "half_rule_jobs": sum(r == RULE_HALF for r in cols["rule"]),
            "prefetch_wait_ms": total("wait_ms"), "probe_sync_ms": total("probe_sync_ms"),
            "select_ms": total("select_ms"), "sync_read_ms": total("sync_read_ms"),
            "copy_ms": total("copy_ms"), "gpu_load_ms": total("gpu_load_ms"),
            "critical_io_ms": critical, "total_io_ms": io_ms,
            "io_ratio": critical / io_ms if io_ms > 0 else None,
            "prefetched_tokens": int(total("prefetched_tokens")), "hit_tokens": int(total("hit_tokens")),
            "used_tokens_after_layer0": int(sum(cols["used_tokens"][1:])),
            "prefetched_bytes": int(total("prefetched_bytes")), "wasted_bytes": int(total("wasted_bytes")),
            "sync_bytes": int(total("sync_bytes")),
            "predicted_chunks": int(total("predicted_chunks")), "cut_chunks": int(total("cut_chunks")),
            "first_job_chunks": cols["prefetched_chunks"][1] if L > 1 else 0,
            "first_job_rule": cols["rule"][1] if L > 1 else None,
        }


# ================================================================ debugging
class UnreadRowGuard:
    """Debug check that no visual row is used unless it was read.

    The request fills every visual row of the prefix cache with NaN before
    the first layer; reads overwrite the rows they bring.  This wraps the
    eager attention function: at the masked visual columns (the columns the
    selection excluded -- their attention weight is exactly 0) K and V are
    replaced by 0, which is what those rows hold in a normal request when
    they were not read; nothing else is touched.  A NaN row that is NOT
    masked therefore reaches the softmax and the output is not finite.

    Recorded per layer at the prefill call: ``used`` (unmasked visual columns),
    ``read`` (visual rows without NaN) and whether the attention output is
    finite.
    """

    def __init__(self, meta):
        self.v0, self.vn = int(meta["v_token_start"]), int(meta["v_token_num"])
        self.used, self.read, self.finite = {}, {}, {}
        self.calls = 0
        self._prev = None

    def __enter__(self):
        self._prev = ml.eager_attention_forward
        ml.eager_attention_forward = self._forward
        return self

    def __exit__(self, *exc):
        assert ml.eager_attention_forward == self._forward, "eager attention was replaced while the guard was active"
        ml.eager_attention_forward = self._prev

    def _forward(self, module, query, key, value, attention_mask, **kw):
        from mmimpress.serve import BIAS
        li = getattr(module, "layer_idx", None)
        v0, p0 = self.v0, self.v0 + self.vn
        bias = BIAS.get(li)
        if bias is not None:
            masked = bias[0, 0, 0, v0:p0] != 0
        else:
            masked = torch.zeros(self.vn, dtype=torch.bool, device=key.device)
        first = li not in self.used
        if first:
            ok = torch.isfinite(key[0, :, v0:p0]).all(dim=2).all(dim=0) & torch.isfinite(value[0, :, v0:p0]).all(dim=2).all(dim=0)
            self.used[li], self.read[li] = (~masked).cpu(), ok.cpu()
        if bool(masked.any()):
            m = masked.view(1, 1, self.vn, 1)
            key, value = key.clone(), value.clone()
            key[:, :, v0:p0] = torch.where(m, torch.zeros((), dtype=key.dtype, device=key.device), key[:, :, v0:p0])
            value[:, :, v0:p0] = torch.where(m, torch.zeros((), dtype=value.dtype, device=value.device), value[:, :, v0:p0])
        out = self._prev(module, query, key, value, attention_mask, **kw)
        fin = bool(torch.isfinite(out[0]).all())
        self.finite[li] = fin and self.finite.get(li, True)
        self.calls += 1
        return out

    def result(self, keep_masks):
        layers = sorted(self.used)
        return {
            "layers": len(layers), "attention_calls": self.calls,
            "all_outputs_finite": all(self.finite[li] for li in layers),
            "used_equals_selection": all(torch.equal(self.used[li], keep_masks[li]) for li in layers),
            "used_rows_all_read": all(bool((self.read[li] | ~self.used[li]).all()) for li in layers),
            "unread_rows": int(sum(int((~self.read[li]).sum()) for li in layers)),
            "used_rows": int(sum(int(self.used[li].sum()) for li in layers)),
        }


def poison_visual_rows(cache_owner, meta):
    """NaN in every visual row of every layer of the request's prefix cache."""
    v0, p0 = int(meta["v_token_start"]), int(meta["v_token_start"]) + int(meta["v_token_num"])
    for li in range(int(meta["num_layers"])):
        cache_owner.k[li][0, :, v0:p0] = float("nan")
        cache_owner.v[li][0, :, v0:p0] = float("nan")


# ================================================================== request
def position_maps_for(ctx):
    """The layer -> next layer position tables of a store, built once per context."""
    maps = getattr(ctx, "_hyperinfer_position_maps", None)
    if maps is None:
        maps = layer_position_maps(ctx.meta["order"])
        ctx._hyperinfer_position_maps = maps
    return maps


@torch.no_grad()
def request_hyperinfer(server, ctx, suffix_ids, *, state=None, kv_prefetch="async", ratio=io_.KEEP_RATIO,
                       probe=len(io_.PROBE_HEADS), sim_threshold=io_.SIM_THRESHOLD, cold=True,
                       selector_out=None, debug_unread_nan=False, guard_out=None):
    """Serve one follow-up turn with HyperInfer.

    Timer boundaries are those of ``impress_official.request_official``:
    cold-cache eviction, the request counter, the prefetch worker and the
    store's position tables are prepared before the timer; TTFT ends right
    after the first output token.
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
    sel = HyperInferSelector(server.runner, ctx, state=state, kv_prefetch=kv_prefetch, position_maps=maps,
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
    if state is not None and kv_prefetch != "off":
        state.requests += 1
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

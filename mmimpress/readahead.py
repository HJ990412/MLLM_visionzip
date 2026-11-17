"""Layer-ordered background read-ahead for the asynchronous stored-KV arms.

One background worker reads a fixed sequence of items (normally "layer 0,
layer 1, ...") in order, and the main thread -- the one that drives the GPU --
takes each item when the corresponding layer is about to run, waiting only if
the item has not arrived yet.  The worker does disk reads (and host buffer
conversion) only; device transfer, scatter and attention stay on the main
thread.

Only reads whose location is known without the question's own computation can
go through here: FullLoad's whole K/V blocks and the packed store's per-layer
byte ranges.  IMPRESS's probe-key read-ahead is the older, one-layer-ahead
mechanism inside ``LayerSelector`` (``prefetch="next_layer"``).

``start()`` must be called after the request's TTFT timer started; the
executor (and its thread) must exist before it.

Contract: docs/mt_gqa_async_contract.md.
"""
from __future__ import annotations

import threading
import time


class LayerReadAhead:
    """Read ``keys`` in order on ``executor``; hand them out with ``take``.

    read_item(key) -> payload runs on the worker thread.
    max_ahead: None = no bound.  Otherwise the worker does not start another
    read while ``max_ahead`` items are read but not yet taken.
    """

    def __init__(self, executor, read_item, keys, max_ahead=None):
        self._executor = executor
        self._read = read_item
        self._keys = list(keys)
        if len(set(self._keys)) != len(self._keys):
            raise ValueError("read-ahead keys must be unique")
        if max_ahead is not None and int(max_ahead) < 1:
            raise ValueError("max_ahead must be None or >= 1")
        self.max_ahead = None if max_ahead is None else int(max_ahead)
        self._cond = threading.Condition()
        self._results = {}
        self._taken = set()
        self._error = None
        self._stop = False
        self._produced = 0
        self._consumed = 0
        self._future = None
        self.peak_outstanding = 0
        self.started_at = None
        self.read_log = []                   # worker: one record per read
        self.take_log = []                   # main: one record per take

    # ------------------------------------------------------------- worker
    def start(self):
        if self._future is not None:
            raise RuntimeError("read-ahead already started")
        self.started_at = time.perf_counter()
        self._future = self._executor.submit(self._loop)
        return self

    def _loop(self):
        try:
            for key in self._keys:
                with self._cond:
                    while (not self._stop and self.max_ahead is not None
                           and self._produced - self._consumed
                           >= self.max_ahead):
                        self._cond.wait()
                    if self._stop:
                        return
                t0 = time.perf_counter()
                payload = self._read(key)
                t1 = time.perf_counter()
                with self._cond:
                    self._results[key] = payload
                    self._produced += 1
                    self.peak_outstanding = max(
                        self.peak_outstanding,
                        self._produced - self._consumed)
                    self.read_log.append({"key": key, "start_s": t0,
                                          "end_s": t1})
                    self._cond.notify_all()
        except BaseException as error:              # noqa: BLE001
            with self._cond:
                self._error = error
                self._cond.notify_all()

    # --------------------------------------------------------------- main
    def take(self, key):
        """Payload of ``key``; blocks until the worker delivered it.

        A worker exception is re-raised here, i.e. inside the request.
        """
        if self._future is None:
            raise RuntimeError("read-ahead was not started")
        if key not in self._keys or key in self._taken:
            raise KeyError(f"read-ahead item {key!r} is unknown or was "
                           "already taken")
        t0 = time.perf_counter()
        with self._cond:
            ready = key in self._results
            while key not in self._results:
                if self._error is not None:
                    raise self._error
                if self._stop:
                    raise RuntimeError("read-ahead was closed")
                self._cond.wait()
            payload = self._results.pop(key)
            self._taken.add(key)
            self._consumed += 1
            self._cond.notify_all()
        self.take_log.append({"key": key,
                              "wait_s": time.perf_counter() - t0,
                              "ready_on_arrival": bool(ready)})
        return payload

    def close(self):
        """Stop the worker loop and wait for it; never raises."""
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        future = self._future
        if future is not None and not future.cancel():
            future.exception()
        with self._cond:
            self._results.clear()

    # ------------------------------------------------------------ summary
    def stats(self):
        reads = list(self.read_log)
        return {
            "items_read": len(reads),
            "items_taken": len(self.take_log),
            "max_ahead": self.max_ahead,
            "peak_outstanding": int(self.peak_outstanding),
            # time the main thread was blocked waiting for an item
            "wait_ms": 1e3 * sum(r["wait_s"] for r in self.take_log),
            "ready_on_arrival": sum(r["ready_on_arrival"]
                                    for r in self.take_log),
            # wall time of the worker's reads (they overlap computation, so
            # this does NOT add to hook time or TTFT)
            "worker_read_ms": 1e3 * sum(r["end_s"] - r["start_s"]
                                        for r in reads),
            "started_at_s": self.started_at,
            "first_read_start_s": reads[0]["start_s"] if reads else None,
            "last_read_end_s": reads[-1]["end_s"] if reads else None,
        }

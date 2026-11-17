"""IMPRESS store construction and diagnostics for the multi-turn condition.

The IMPRESS arm stores the whole visual KV plus the probe-key sidecar at the
first dialogue's turn 1 (the ordinary raster piggyback writer), and then
repacks each layer in descending importance for THAT turn-1 question:

  importance_for_suffix   H2O column sums over every suffix row, head-mean,
                          for one explicit suffix (the turn-1 prompt after
                          ``<image>``).  Same forward and same ``Calibrator``
                          as ``serve.calibrate_image`` with one question; the
                          only difference is that the suffix ids are given
                          instead of being rebuilt from ``runner.prompt``.
  reorder_store_by_importance
                          per-layer stable descending order, K / V / probe
                          sidecar rewritten in place (the logic of
                          ``scripts/02_reorder.py`` ``rewrite_layer`` +
                          ``commit`` for ``--order importance``), timed per
                          file kind, separator sidecar rebuilt.

``unreordered_chunk_counts`` is the offline counterfactual used by the report:
which chunks the same per-layer selection would have touched had the store
stayed in raster order.  It is computed from the request's keep masks after
the request; nothing is read.

Contract: docs/mt_gqa_async_contract.md.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from mmimpress.reorder import mapping_from_perm
from mmimpress.store import (META_NAME, load_meta,
                             write_separator_sidecar_from_store)

_NP = {"float16": np.float16, "float32": np.float32}


@torch.no_grad()
def importance_for_suffix(server, ctx, suffix_ids, importance_metric="h2o"):
    """Per-layer visual importance (num_layers, v_num) for one suffix.

    Loads the whole stored prefix (original order) into a fresh cache and
    runs the suffix once with the ``Calibrator`` hooks, exactly like one
    iteration of ``serve.calibrate_image``.
    """
    from mmimpress import sparsevlm as sv
    from mmimpress.serve import Calibrator
    metric = sv.validate_importance_metric(importance_metric)
    if metric != "h2o":
        raise ValueError("the multi-turn IMPRESS arm is defined with the "
                         "h2o importance metric")
    dev = server.runner.model.device
    suffix_ids = torch.as_tensor(suffix_ids).to(dev)
    rr = torch.empty(0, dtype=torch.long, device=dev)
    cache = ctx.cache.new_request()
    L = ctx.meta["num_layers"]
    for li in range(L):
        for kind in ("k", "v"):
            ctx.cache.write_full(li, kind, ctx.reader.read_full(li, kind))
    cal = Calibrator(server.runner, ctx, rr, importance_metric=metric)
    with cal:
        P, nn = ctx.meta["prefix_len"], suffix_ids.shape[0]
        pos = torch.arange(P, P + nn, device=dev)
        server.runner.model(
            input_ids=suffix_ids.unsqueeze(0),
            attention_mask=torch.ones(1, P + nn, dtype=torch.long,
                                      device=dev),
            position_ids=pos.unsqueeze(0), cache_position=pos,
            past_key_values=cache, use_cache=True)
    scores = torch.stack([cal.scores[li] for li in range(L)])
    del cache
    return scores


def importance_perms(scores):
    """{layer -> stable descending-importance permutation} (IMPRESS 4.4.1)."""
    return {li: torch.argsort(scores[li], descending=True,
                              stable=True).tolist()
            for li in range(scores.shape[0])}


def reorder_store_by_importance(store_dir, scores, metadata_update=None):
    """Repack a FRESH raster store per layer by ``scores``; return timings.

    Rewrites ``layer_*/{k,v,probe_k}.bin`` in place (temp file + rename),
    records the per-layer order and separator positions in ``meta.json`` and
    rebuilds ``sep_kv.bin``.  The resulting files and order metadata are the
    ones ``scripts/02_reorder.py --order importance`` would produce from the
    same scores.
    """
    store_dir = Path(store_dir)
    meta = load_meta(store_dir)
    if meta.get("reordered") or meta.get("order"):
        raise ValueError("importance reorder expects a fresh raster store")
    L, vn, hd = int(meta["num_layers"]), int(meta["v_token_num"]), int(meta["head_dim"])
    scores = torch.as_tensor(scores)
    if tuple(scores.shape) != (L, vn):
        raise ValueError(f"scores must be ({L}, {vn}), got {tuple(scores.shape)}")
    dt = _NP[meta["dtype"]]
    widths = (("k", int(meta["num_heads"])), ("v", int(meta["num_heads"])),
              ("probe_k", int(meta["probe_heads"])))
    timing = {name: {"read_ms": 0.0, "permute_ms": 0.0, "write_ms": 0.0,
                     "bytes": 0} for name, _ in widths}
    perms = importance_perms(scores)
    orders, newline_stored = [], []
    started = time.perf_counter()
    for li in range(L):
        perm = perms[li]
        if sorted(perm) != list(range(vn)):
            raise ValueError("importance order is not a permutation")
        perm_np = np.asarray(perm)
        ldir = store_dir / f"layer_{li:02d}"
        for name, width in widths:
            p = ldir / f"{name}.bin"
            t0 = time.perf_counter()
            a = np.fromfile(p, dtype=dt).reshape(vn, width, hd)
            t1 = time.perf_counter()
            rewritten = np.ascontiguousarray(a[perm_np])
            t2 = time.perf_counter()
            tmp = p.with_name(p.name + ".reorder-tmp")
            if tmp.exists():
                raise FileExistsError(f"stale reorder temp file: {tmp}")
            rewritten.tofile(tmp)
            os.replace(tmp, p)
            t3 = time.perf_counter()
            entry = timing[name]
            entry["read_ms"] += (t1 - t0) * 1e3
            entry["permute_ms"] += (t2 - t1) * 1e3
            entry["write_ms"] += (t3 - t2) * 1e3
            entry["bytes"] += int(rewritten.nbytes)
        inverse = mapping_from_perm(perm)          # original -> stored
        orders.append([int(x) for x in perm])      # stored -> original
        newline_stored.append([inverse[int(i)] for i in meta["newline_idx"]])
    rewrite_ms = (time.perf_counter() - started) * 1e3

    same = all(order == orders[0] for order in orders)
    meta["order"] = orders[0] if same else orders
    meta["order_is_per_layer"] = not same
    meta["newline_stored"] = newline_stored[0] if same else newline_stored
    meta["reordered"] = True
    if metadata_update:
        meta.update(metadata_update)
    t0 = time.perf_counter()
    with open(store_dir / META_NAME, "w") as f:
        json.dump(meta, f, indent=1)
    meta_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    write_separator_sidecar_from_store(store_dir, load_meta(store_dir))
    sidecar_ms = (time.perf_counter() - t0) * 1e3
    return {"per_kind": timing, "rewrite_ms": rewrite_ms,
            "meta_write_ms": meta_ms, "separator_sidecar_ms": sidecar_ms,
            "total_ms": rewrite_ms + meta_ms + sidecar_ms,
            "shared_order": bool(same)}


def stored_order(meta, layer):
    """stored -> original positions of one layer."""
    order = meta.get("order")
    if not order:
        return list(range(int(meta["v_token_num"])))
    return order[layer] if meta.get("order_is_per_layer") else order


def unreordered_chunk_counts(keep_masks, layer_log, meta):
    """Chunks the request's selection would touch in RASTER order.

    Computed, never read.  For every layer the tokens that were loaded for
    attention -- the consensus selection plus separators on the probe path,
    the union of the per-head selections plus separators for V on the
    fallback path -- are mapped back to original positions and counted in
    64-token raster chunks.  K on a fallback layer is the whole layer in any
    order.  Returns per-layer ``k_chunks`` / ``v_chunks`` lists and the same
    share-of-layer fractions as ``touched_{k,v}_chunk_fraction``.
    """
    cs = int(meta["chunk_size"])
    nc = int(meta["n_chunks_per_layer"])
    k_chunks, v_chunks = [], []
    for row in layer_log:
        li = int(row["layer"])
        keep = keep_masks[li]
        needed = keep.any(dim=0) if keep.dim() == 2 else keep
        order = torch.as_tensor(stored_order(meta, li), dtype=torch.long)
        original = order[needed.nonzero(as_tuple=True)[0]]
        count = int(torch.unique(original // cs).numel())
        v_chunks.append(count)
        k_chunks.append(nc if row["mode"] == "fallback" else count)
    n = max(1, len(layer_log))
    return {"k_chunks": k_chunks, "v_chunks": v_chunks,
            "k_chunk_fraction": float(sum(k_chunks)) / (n * nc),
            "v_chunk_fraction": float(sum(v_chunks)) / (n * nc)}

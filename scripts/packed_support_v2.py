"""Shared harness helpers for the v2 packed-store experiments (scripts 159/161/162).

persist_packed   : Turn-1 piggyback persistence of a packed store + integrity seal + receipt
activate_packed  : PackedImageContext activation (pinned buffer, layout validation) with timing
packed_hit       : one cold cache hit through ``request_packed`` under the independent
                   pread/preadv trace, the no-vision guard and optional all-step logits
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import torch

from mmimpress.packed_serve import PackedImageContext, PackedReadTrace, request_packed
from mmimpress.packed_store import PACKED_NAME, persist_captured_packed_prefix
from mmimpress.prefix25 import process_io, seal_integrity
from mmimpress.serve import suffix_ids_from_prompt

ABLATION = "layer_budget_kv25_v2_packed"


def persist_packed(exp, capture, path, image, doc, extra=None):
    before = process_io()
    started = time.perf_counter()
    enc = capture["enc"]
    receipt = persist_captured_packed_prefix(
        exp.runner, capture["cache"], enc["input_ids"], enc["image_sizes"][0], capture["vision"].result_cpu(), path,
        layer_budget=doc, capture_stats=capture["vision"], image_id=image["image_id"], model_id=exp.runner.model_id,
        image_input_sha256=image["image_sha256"],
        extra_metadata={"checkpoint_revision": exp.runner.cfg._commit_hash, "ablation": ABLATION, **(extra or {})})
    t = time.perf_counter()
    files = seal_integrity(path)
    hash_ms = (time.perf_counter() - t) * 1e3
    total = (time.perf_counter() - started) * 1e3
    after = process_io()
    meta = receipt["meta"]
    timing = dict(receipt["timing_ms"], packed_sha256_ms=meta["packed_writer_timing_ms"]["packed_sha256_ms"])
    return {"physical_write_id": str(uuid.uuid4()), "store_path": str(path), "arm": f"{doc['alloc_id']}_packed", "image_id": image["image_id"],
            "metadata": meta, "files": files, "timing_ms": timing, "hash_and_seal_ms": hash_ms, "persistence_ms": total,
            "os_write_bytes": after["wchar"] - before["wchar"], "os_write_calls": after["syscw"] - before["syscw"],
            "file_logical_bytes": sum(x["size"] for x in files.values()), "allocated_bytes": sum(x["allocated_bytes"] for x in files.values()),
            "N": meta["original_content_count"], "k_per_layer": meta["layer_keep_counts"], "layer_budget_sha256": meta["layer_budget_sha256"],
            "packed_total_bytes": meta["packed_total_bytes"], "packed_sha256": meta["packed_sha256"], "store_format": meta["store_format"]}


def activate_packed(exp, path):
    start = time.perf_counter()
    ctx = PackedImageContext(path, exp.runner.model.device)
    ctx.validate_packed_layout()
    torch.cuda.synchronize()
    return ctx, {"activation_ms": (time.perf_counter() - start) * 1e3, **ctx.integrity_activation,
                 "pinned": ctx.reader.pinned, "pinned_host_bytes": int(ctx.reader.total), "store_format": ctx.meta["store_format"]}


def packed_hit(exp, ctx, path, prompt, logits=False, directcopy=False):
    runner, server = exp.runner, exp.server
    conditioning = time.perf_counter()
    ctx.reader.drop_all()
    cond = {"policy": "DONTNEED", "ms": (time.perf_counter() - conditioning) * 1e3}
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
    captured = []; handle = None
    if logits:
        handle = runner.model.register_forward_hook(lambda m, a, o: captured.append(o.logits[0, -1].detach().float().cpu()) if hasattr(o, "logits") else None)
    suffix = suffix_ids_from_prompt(runner, prompt).to(runner.model.device)
    with exp.e.QA._NoVisionForward(runner) as guard, PackedReadTrace(path) as trace:
        value = request_packed(server, ctx, suffix_ids=suffix, cold=False, directcopy=directcopy)
    value["ttft_ms"] = (value["first_token_at_s"] - start) * 1e3
    value["request_e2e_ms"] = (time.perf_counter() - start) * 1e3
    value["prediction"] = value["answer"]; value["vision_calls"] = guard.calls; value["online_query_score_calls"] = 0
    if handle:
        handle.remove(); value["all_logits"] = captured
    s = trace.summary()
    assert s["bytes"] == value["io"]["bytes"] and s["preads"] == value["io"]["preads"] == 1
    assert s["pread_calls"] == 0 and s["preadv_calls"] == 1
    call = s["calls"][0]
    assert call["path"].endswith("/" + PACKED_NAME) and Path(call["path"]).resolve().parent == Path(path).resolve()
    assert call["offset"] == 0 and call["returned"] == call["requested"] == ctx.meta["packed_total_bytes"]
    assert value["vision_calls"] == 0
    value["actual_io"] = s; value["conditioning"] = cond
    value["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(); value["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
    return value

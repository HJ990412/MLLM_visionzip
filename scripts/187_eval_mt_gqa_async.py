#!/usr/bin/env python3
"""MT-GQA multi-turn: ReComp / FullLoad / IMPRESS / Ours with asynchronous
stored-KV reads (contract docs/mt_gqa_async_contract.md).

Lifecycle is the LLaVA-NeXT MT-GQA HOLD one (scripts/164, itself the 162/156
code): request generation, dialogue-wise arm rotation (``method_order``),
method-local generated history, strict normalized EM, image-atomic attempts,
independent per-image audit, atomic commit, exact-allowlist cleanup of the
run-local scratch stores.  The shared pieces are imported from 164/156 and not
copied.  What is new here:

* arms.  ``recompute``; ``fullload`` / ``fullload_async``; ``impress`` /
  ``impress_async`` (+ ``impress_r100``, gate phase only);
  ``ours_prefixstore_kv25_PYR-beta-raw_packed`` / ``..._packed_async``.
  A synchronous arm and its asynchronous twin serve the SAME store, which is
  built from the turn 1 of whichever of the two comes first in the first
  dialogue's rotation.
* store kinds.  ``raster`` (FullLoad: the HOLD raster writer),
  ``impress`` (the same writer, then per-layer repack by the turn-1
  question's H2O importance, ``mmimpress.impress_mt``),
  ``packed_PYR-beta-raw`` (the pyrbeta run's vector document, unchanged).
* read trace with thread and time, so the audit can prove that read-ahead
  reads are issued by the worker and only after the timer started.
* phases ``gate`` (4 smoke images, 8 arms), ``dev`` (40 DEV images, 7 arms),
  ``main`` (398 images, 4 arms).

Stages: --stage freeze | gate | dev | main [--resume]
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mmimpress import impress_mt
from mmimpress.layer_budget import load_budget_document
from mmimpress.packed_serve import PackedImageContext, PackedReadTrace, request_packed
from mmimpress.packed_store import PACKED_NAME, STORE_FORMAT, persist_captured_packed_prefix

SCHEMA = "llava-mtgqa-async-v1"
CONTRACT = ROOT / "docs/mt_gqa_async_contract.md"
PYRBETA_RUN = ROOT / "runs/layer_budget_kv25_pyrbeta_20261110T073604Z"
V2_RUN = ROOT / "runs/layer_budget_kv25_v2_20261106T073502Z"
V2_SPLIT = ROOT / "results/layer_budget_kv25_v2_20261106T073502Z/data_split.json"
VECTOR = "PYR-beta-raw"
VECTOR_RATIO_SHA = "51f9b0db19a93f8ca8b34b8beb8af21285721c5ea460566afbd9a3185b331724"
VECTOR_FILE_SHA = "8bfc4cbc55eb778a7ef0433ade4bf13f7a24107b76515f23a72ac01e0740ea5d"
OURS = f"ours_prefixstore_kv25_{VECTOR}_packed"
OURS_ASYNC = OURS + "_async"
PACKED_KIND = f"packed_{VECTOR}"
IMPRESS = {"probe_heads": 3, "alpha": 0.6, "ratio": 0.25, "chunk_size": 64, "importance_metric": "h2o",
           "fallback_v": "selected", "prefetch_async": "next_layer", "calibration_questions": 1}
READAHEAD = {"fullload_async": "layer_stream", OURS_ASYNC: "layer_stream", "impress_async": "next_layer", "max_ahead": None}

# family, store kind, asynchronous?, IMPRESS keep ratio (None = not IMPRESS)
ARMS = OrderedDict([
    ("recompute",      {"family": "recompute", "kind": "none", "async": False, "display": "ReComp"}),
    ("fullload",       {"family": "fullload", "kind": "raster", "async": False, "display": "FullLoad"}),
    ("fullload_async", {"family": "fullload", "kind": "raster", "async": True, "display": "FullLoad"}),
    ("impress",        {"family": "impress", "kind": "impress", "async": False, "display": "IMPRESS", "ratio": 0.25}),
    ("impress_async",  {"family": "impress", "kind": "impress", "async": True, "display": "IMPRESS", "ratio": 0.25}),
    ("impress_r100",   {"family": "impress", "kind": "impress", "async": False, "display": "IMPRESS (유지 100%)", "ratio": 1.0}),
    (OURS,             {"family": "ours", "kind": PACKED_KIND, "async": False, "display": "Ours"}),
    (OURS_ASYNC,       {"family": "ours", "kind": PACKED_KIND, "async": True, "display": "Ours"}),
])
PHASE_METHODS = {
    "gate": ["recompute", "fullload", "fullload_async", "impress", "impress_async", "impress_r100", OURS, OURS_ASYNC],
    "dev": ["recompute", "fullload", "fullload_async", "impress", "impress_async", OURS, OURS_ASYNC],
    "main": ["recompute", "fullload_async", "impress_async", OURS_ASYNC],
}
PHASES = ("gate", "dev", "main")


def helper(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


H = helper("164_eval_layer_budget_pyrbeta_hold.py", "_mtgqa_async_hold_lifecycle")
V, QA, H1 = H.V, H.QA, H.H1
import main_b_support as support

MODEL_ID, REVISION, INDEX_SHA, WORKLOAD_SHA, PRIOR, SHORT_ANSWER = (
    H.MODEL_ID, H.REVISION, H.INDEX_SHA, H.WORKLOAD_SHA, H.PRIOR, H.SHORT_ANSWER)
sha, canonical_hash, safe, sync_dir, atomic_json, append, load_jsonl = (
    H.sha, H.canonical_hash, H.safe, H.sync_dir, H.atomic_json, H.append, H.load_jsonl)
frozen_dialogues, method_order, strict_score, assert_gpu_exclusive, conditioning_trace, PilotCounts = (
    H.frozen_dialogues, H.method_order, H.strict_score, H.assert_gpu_exclusive, H.conditioning_trace, H.PilotCounts)
normal_request, file_hashes, Checks, capacity_guard, cleanup_committed = (
    H.normal_request, H.file_hashes, H.Checks, H.capacity_guard, H.cleanup_committed)
doc_counts, independent_counts = H.doc_counts, H.independent_counts


REPORT_ONLY = ("189_report_mt_gqa_async.py",)      # reads committed rows only; not part of the measurement freeze


def source_hashes():
    paths = (list((ROOT / "mmimpress").rglob("*.py")) + [p for p in (ROOT / "scripts").glob("*.py") if p.name not in REPORT_ONLY]
             + list((ROOT / "tests").glob("test_*.py")))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def request_id(run_id, phase, method, dialog_id, turn):
    return f"{SCHEMA}:{run_id}:{phase}:{MODEL_ID}:{method}:{dialog_id}:t{turn}"


def kind_of(method):
    return ARMS[method]["kind"]


def stored_methods(methods):
    return [m for m in methods if kind_of(m) != "none"]


def store_kinds(methods):
    kinds = []
    for m in stored_methods(methods):
        if kind_of(m) not in kinds:
            kinds.append(kind_of(m))
    return kinds


def prompt_factory(methods, dialog, turn, method, generated):
    return H1.prompt_factory(methods, dialog, turn, method, generated)


# ------------------------------------------------------------------ tracing
class ThreadedReadTrace(PackedReadTrace):
    """``PackedReadTrace`` plus, per call, the start time and whether the
    calling thread is the one that entered the trace (the request thread)."""

    def __enter__(self):
        self._pread, self._preadv = os.pread, os.preadv
        trace = self
        main = threading.get_ident()

        def pread(fd, count, offset):
            path = trace._path(fd); trace._guard(path)
            t = time.perf_counter()
            blob = trace._pread(fd, count, offset)
            trace.calls.append({"syscall": "pread", "path": str(path), "offset": offset, "requested": count, "returned": len(blob),
                                "t_s": t, "main_thread": threading.get_ident() == main})
            return blob

        def preadv(fd, buffers, offset, *flags):
            path = trace._path(fd); trace._guard(path)
            requested = sum(memoryview(b).nbytes for b in buffers)
            t = time.perf_counter()
            n = trace._preadv(fd, buffers, offset, *flags)
            trace.calls.append({"syscall": "preadv", "path": str(path), "offset": offset, "requested": requested, "returned": n,
                                "t_s": t, "main_thread": threading.get_ident() == main})
            return n

        os.pread, os.preadv = pread, preadv
        return self


class WriteTimer:
    """Times ``Path.write_bytes`` per file name while the store writer runs.

    The raster writer does not time the probe sidecar separately; this
    observes its own writes without changing them.
    """

    def __enter__(self):
        self.by_name = {}
        self._orig = Path.write_bytes
        timer = self

        def write_bytes(path_self, data):
            t0 = time.perf_counter()
            n = timer._orig(path_self, data)
            dt = time.perf_counter() - t0
            e = timer.by_name.setdefault(path_self.name, {"ms": 0.0, "bytes": 0, "files": 0})
            e["ms"] += dt * 1e3; e["bytes"] += len(data); e["files"] += 1
            return n

        Path.write_bytes = write_bytes
        return self

    def __exit__(self, *exc):
        Path.write_bytes = self._orig


def compact_trace(calls, store_path, t0):
    root = Path(store_path).resolve()
    out = []
    for c in calls:
        p = Path(c["path"])
        rel = p.relative_to(root).as_posix() if p.is_relative_to(root) else str(p)
        out.append([c["syscall"], rel, int(c["offset"]), int(c["requested"]), int(c["returned"]),
                    round((c["t_s"] - t0) * 1e3, 4), 1 if c["main_thread"] else 0])
    return out


# ---------------------------------------------------------------- requests
def stored_hit(runner, servers, ctx, factory, method, image_id, store_path):
    arm = ARMS[method]
    family = arm["family"]
    with QA._NoVisionForward(runner) as guard:
        c0 = time.perf_counter(); ctx.reader.drop_all(); conditioning = (time.perf_counter() - c0) * 1000
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
        prompt, history, entries = factory()
        tokenized = runner.processor.tokenizer(prompt, return_tensors="pt")
        suffix = QA._suffix_from_tokenized(runner, tokenized).to(runner.model.device)
        selectors = []
        with PilotCounts(runner) as counts, ThreadedReadTrace() as reads:
            if family == "fullload":
                result = servers["base"].request(ctx, mode="fullload", suffix_ids=suffix, cold=False, prompt_text=prompt, cache_id=image_id,
                                                 prefetch="layer_stream" if arm["async"] else "none",
                                                 max_ahead=READAHEAD["max_ahead"], selector_out=selectors)
            elif family == "impress":
                server = servers["r100"] if arm["ratio"] == 1.0 else servers["base"]
                result = server.request(ctx, mode="impress", suffix_ids=suffix, cold=False,
                                        importance_metric=IMPRESS["importance_metric"], fallback_v=IMPRESS["fallback_v"],
                                        prefetch=IMPRESS["prefetch_async"] if arm["async"] else "none", selector_out=selectors)
            else:
                result = request_packed(servers["base"], ctx, suffix_ids=suffix, cold=False, image_id=image_id, directcopy=False,
                                        readahead="layer_stream" if arm["async"] else "none", max_ahead=READAHEAD["max_ahead"])
        returned = time.perf_counter()
    s = reads.summary()
    assert s["bytes"] == result["io"]["bytes"] and s["preads"] == result["io"]["preads"]
    if family == "impress":      # after the request, outside every timer
        sel = selectors[0]
        result["layer_log"] = [dict(r) for r in sel.log]
        result["probe_sources"] = [[r["layer"], r["source"], r["ready_on_arrival"]] for r in sel.probe_log]
        result["unreordered_chunks"] = impress_mt.unreordered_chunk_counts(sel.keep_masks, sel.log, ctx.meta)
        result["impress_params"] = {"ratio": float(server.ratio), "probe": int(server.probe), "alpha": float(server.alpha)}
    del selectors
    result.update(projection_calls=counts.result(), scoring_calls=0,
                  end_to_end_ttft_ms=(result["first_token_at_s"] - started) * 1000, request_e2e_ms=(returned - started) * 1000,
                  request_started_at_s=started, vision_forward_count=guard.calls, page_cache_conditioning_ms=conditioning,
                  cap_reached=(len(result["generated_token_ids"]) == 16 and result["generated_token_ids"][-1] != runner.processor.tokenizer.eos_token_id),
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                  os_read_trace=compact_trace(s["calls"], store_path, result["core_started_at_s"]),
                  pread_calls=s["pread_calls"], preadv_calls=s["preadv_calls"])
    ctx.cache.k = None; ctx.cache.v = None
    return result, {"prompt": prompt, "history": history, "history_entries": entries, "suffix_input_ids": suffix.cpu().tolist()}


def stored_request(runner, servers, ctx, factory, method, image_id, store_path):
    original = [l.self_attn.forward for l in runner.layers]
    hooks_before = [len(l._forward_pre_hooks) for l in runner.layers]
    with conditioning_trace() as conditioning:
        result, diag = stored_hit(runner, servers, ctx, factory, method, image_id, store_path)
    assert all(l.self_attn.forward == f for l, f in zip(runner.layers, original)), "adapter not restored"
    assert [len(l._forward_pre_hooks) for l in runner.layers] == hooks_before, "layer hook left behind"
    from mmimpress.serve import BIAS
    assert not BIAS, "stale mask after request"
    assert conditioning and all(e["status"] == "SUCCESS" for e in conditioning), "page cache conditioning not verified"
    root = Path(store_path).resolve()
    result.update(conditioned_files=sorted(Path(e["path"]).relative_to(root).as_posix() for e in conditioning),
                  adapter_restored=True, page_cache_conditioning_method="posix_fadvise_DONTNEED",
                  page_cache_conditioning_excluded_from_ttft=True)
    return result, diag


# -------------------------------------------------------------- persistence
def build_impress_layout(runner, server, path, diag, scores_path):
    """Turn-1 side of the IMPRESS arm: importance of the turn-1 question from
    the store just written, then the per-layer repack.  Not in any TTFT."""
    from mmimpress.serve import ImageContext
    t0 = time.perf_counter()
    ctx = ImageContext(path, runner.model.device, drop_cache=False, require_v_hidden=False)
    suffix = QA._suffix_from_tokenized(runner, runner.processor.tokenizer(diag["prompt"], return_tensors="pt"))
    torch.cuda.synchronize(); t1 = time.perf_counter()
    scores = impress_mt.importance_for_suffix(server, ctx, suffix, IMPRESS["importance_metric"])
    torch.cuda.synchronize(); t2 = time.perf_counter()
    ctx.close(); del ctx
    array = scores.numpy().astype(np.float32)
    assert np.isfinite(array).all()
    np.save(scores_path, array)
    t3 = time.perf_counter()
    reorder = impress_mt.reorder_store_by_importance(path, scores, metadata_update={
        "physical_layout": "impress_turn1_importance", "layout_method": "importance",
        "layout_source": "first_dialogue_turn1_question_importance", "layout_uses_dataset_question": True,
        "llm_used_for_layout_scoring": True, "calibration_questions": 1, "importance_metric": IMPRESS["importance_metric"],
        "importance_metric_calibration_questions": 1, "qa_select_compatible": False, "global_order_all_layers": False,
        "composed_from_store": True, "impress_turn1_suffix_ids_sha256": canonical_hash(suffix.tolist())})
    t4 = time.perf_counter()
    gc.collect(); torch.cuda.empty_cache()
    return {"context_open_ms": (t1 - t0) * 1e3, "importance_ms": (t2 - t1) * 1e3, "scores_save_ms": (t3 - t2) * 1e3,
            "reorder_ms": (t4 - t3) * 1e3, "reorder": reorder, "total_ms": (t4 - t0) * 1e3,
            "importance_definition": "H2O column sum over every suffix row of the turn-1 prompt, head mean; one separate suffix forward over the stored FP16 prefix",
            "suffix_tokens": int(suffix.numel()), "suffix_ids_sha256": canonical_hash(suffix.tolist()),
            "scores_file": Path(scores_path).name, "scores_sha256": sha(scores_path)}


def _native_persist(runner, server, result, diag, path, image_id, kind, source, docs, scores_path):
    from mmimpress.piggyback import persist_captured_raster_prefix
    enc = diag["enc_cpu"]
    args = (runner, result["captured_past_key_values"], enc["input_ids"], enc["image_sizes"][0])
    common = dict(image_id=image_id, model_id=MODEL_ID, chunk_size=64, image_input_sha256=diag["image_input_sha256"],
                  extra_metadata={"dataset": "MT-GQA-reconstructed", "source_dialog_id": source["dialog_id"], "source_turn_id": 1,
                                  "source_method_id": source["method_id"], "source_execution_id": source["physical_execution_id"],
                                  "source_logical_request_id": source["logical_request_id"], "run_schema": SCHEMA})
    if kind in ("raster", "impress"):
        with WriteTimer() as timer:
            receipt = persist_captured_raster_prefix(*args, diag["hidden"].result_cpu(), path, probe_heads=IMPRESS["probe_heads"],
                                                     hidden_capture_stats=diag["hidden"], full_integrity_hash=False, **common)
        receipt["write_by_file_name"] = timer.by_name
        if kind == "impress":
            receipt["impress_turn1"] = build_impress_layout(runner, server, path, diag, scores_path)
        return receipt
    vec = kind[len("packed_"):]
    return persist_captured_packed_prefix(*args, diag["vision"].result_cpu(), path, layer_budget=docs[vec],
                                          capture_stats=diag["vision"], full_integrity_hash=False, **common)


def persist(runner, server, result, diag, path, image_id, kind, source, docs, scores_path):
    return support.persist_llava(lambda: _native_persist(runner, server, result, diag, path, image_id, kind, source, docs, scores_path), path, kind, source)


def activate(runner, kind, path):
    started = time.perf_counter()
    if kind in ("raster", "impress"):
        from mmimpress.serve import ImageContext
        ctx = ImageContext(path, runner.model.device, drop_cache=False, require_v_hidden=False)
        completion = support.verify_completion(path, "integrity.json")
        pinned = 0
    else:
        ctx = PackedImageContext(path, runner.model.device)
        ctx.validate_packed_layout()
        completion = ctx.integrity_activation
        pinned = int(ctx.reader.total)
    ctx.activation = {"kind": kind, "host_tensor_bytes": sum(t.numel() * t.element_size() for t in ctx.cache.sys_kv.values()),
                      "pinned_host_buffer_bytes": pinned, "gpu_tensor_bytes": 0, "visual_payload_resident_bytes": 0,
                      "file_bytes": sum((path / n).stat().st_size for n in ("meta.json", "sys_kv.pt")),
                      "completion_integrity": completion, "event_id": str(uuid.uuid4()), "event_kind": "activation"}
    ctx.activation["outer_wall_ms"] = (time.perf_counter() - started) * 1000
    return ctx


def make_row(runner, result, diag, meta, *, methods, method, dialog, turn, phase, run, attempt, physical_id, ordinal, store_info, config):
    arm = ARMS[method]
    question = dialog["turns"][turn - 1]
    n = int(meta["n_spatial"]); hit = turn > 1 and arm["family"] != "recompute"
    logical = request_id(run.name, phase, method, dialog["dialog_id"], turn)
    suffix = QA._suffix_from_tokenized(runner, runner.processor.tokenizer(diag["prompt"], return_tensors="pt")).cpu().tolist()
    if diag.get("suffix_input_ids") is not None:
        assert diag["suffix_input_ids"] == suffix
    value = {a: b for a, b in result.items() if a not in ("captured_past_key_values", "first_logits", "logits", "processor")}
    g = {key: meta.get(key, []) if key == "padding_idx" else meta[key] for key in
         ("num_heads", "head_dim", "num_layers", "v_token_start", "v_token_num", "n_spatial", "prefix_len", "dtype", "newline_idx", "padding_idx")}
    order = method_order(methods, ordinal)
    io = result.get("io", {}) if hit else {}
    per_kind = io.get("per_kind", {})
    wait = result.get("read_wait") or {}
    return {"schema_version": SCHEMA, "experiment_id": run.name, "phase": phase, "attempt_id": attempt,
            "logical_request_id": logical, "request_id": logical, "physical_execution_id": physical_id,
            "model_id": MODEL_ID, "model_revision": REVISION, "method_id": method, "family": arm["family"], "async_read": bool(arm["async"]),
            "store_kind": arm["kind"], "image_id": dialog["image_id"],
            "dialog_id": dialog["dialog_id"], "turn_id": turn, "question_id": str(question["question_id"]),
            "question": question["question"], "gold": question["answers"], "prediction": result["answer"],
            "score": strict_score(result["answer"], question["answers"][0]), "prompt": diag["prompt"],
            "history": diag["history"], "history_entries": diag["history_entries"],
            "prompt_sha256": hashlib.sha256(diag["prompt"].encode()).hexdigest(), "suffix_input_ids": suffix,
            "suffix_ids_sha256": canonical_hash(suffix), "history_sha256": hashlib.sha256(diag["history"].encode()).hexdigest(),
            "config_sha256": config["_file_sha256"], "manifest_sha256": config["_manifest_sha256"],
            "code_sha256": canonical_hash(config["source_sha256"]), "global_dialog_ordinal": ordinal,
            "method_order": order, "method_order_position": order.index(method),
            "request_started_at_s": result["request_started_at_s"], "first_token_at_s": result["first_token_at_s"],
            "request_finished_at_s": result["request_started_at_s"] + result["request_e2e_ms"] / 1000,
            "ttft_ms": result["end_to_end_ttft_ms"], "request_e2e_ms": result["request_e2e_ms"],
            "ssd_read_bytes": int(io.get("bytes", 0)), "ssd_preads": int(io.get("preads", 0)),
            "pread_calls": int(result.get("pread_calls", 0)), "preadv_calls": int(result.get("preadv_calls", 0)),
            "probe_read_bytes": int(per_kind.get("probe", {}).get("bytes", 0)), "k_read_bytes": int(per_kind.get("k", {}).get("bytes", 0)),
            "v_read_bytes": int(per_kind.get("v", {}).get("bytes", 0)), "packed_read_bytes": int(per_kind.get("packed", {}).get("bytes", 0)),
            "read_blocked_ms": wait.get("read_blocked_ms") if hit else None,
            "N_content": n, "geometry": g, "source_store": store_info if hit else None,
            "provisioning_mode": "FRESH_IMAGE_STREAMING", "result": safe(value)}


# ------------------------------------------------------------------- audit
def _chunks_of(offset, length, width, vn):
    row0, rows = offset // width, length // width
    return list(range(row0 // 64, (row0 + rows - 1) // 64 + 1))


def audit_image_rows(rows, manifest, config, phase, image_id, artifact=None):
    methods = config["phases"][phase]["methods"]
    c = Checks()
    ids = set(manifest["phase_dialogue_ids"][phase])
    ds = [d for d in manifest["dialogues"] if d["image_id"] == image_id and d["dialog_id"] in ids]
    c.require(bool(ds), "known_image", image_id)
    expected = {(d["dialog_id"], m, t): d["turns"][t - 1] for d in ds for m in methods for t in (1, 2, 3)}
    bykey = {(r["dialog_id"], r["method_id"], r["turn_id"]): r for r in rows}
    c.require(set(bykey) == set(expected) and len(rows) == len(expected), "image_completeness", image_id)
    c.require(len({r["logical_request_id"] for r in rows}) == len(rows), "unique_logical_ids", image_id)
    c.require(len({r["physical_execution_id"] for r in rows}) == len(rows), "unique_physical_ids", image_id)
    ordinals = {d["dialog_id"]: i for i, d in enumerate(manifest["dialogues"])}
    doc = config["layer_budgets"][VECTOR]
    metas, sources = {}, {}
    if artifact is not None:
        artifact = Path(artifact)
        builds = []
        for kind in store_kinds(methods):
            p = json.loads((artifact / f"{kind}_persistence.json").read_text())
            meta = json.loads((artifact / f"{kind}_meta.json").read_text())
            hashes = json.loads((artifact / f"{kind}_hashes.json").read_text())
            files = p["files"]; builds.append(p["event_id"]); metas[kind] = meta; sources[kind] = p["source"]
            c.require(hashes == {n: f["sha256"] for n, f in files.items()}, "complete_store_hash_inventory", kind)
            c.require(p["file_logical_bytes"] == sum(f["size"] for f in files.values()), "file_occupancy_sums", kind)
            c.require(p["service_persistence_ms"] >= p["native_writer_wall_ms"] > 0, "measured_store_completion", kind)
            c.require(p["source"]["dialog_id"] == ds[0]["dialog_id"] and kind_of(p["source"]["method_id"]) == kind
                      and p["source"]["method_id"] in methods, "first_dialogue_T1_build_same_kind", kind)
            a = json.loads((artifact / f"{kind}_activation.json").read_text())
            c.require(a["event_kind"] == "activation" and a["outer_wall_ms"] > 0 and a["kind"] == kind, "actual_activation_event", kind)
            n = meta["v_token_num"] - len(meta["newline_idx"]); width = meta["num_heads"] * meta["head_dim"] * 2
            vn, nl = meta["v_token_num"], meta["num_layers"]
            if kind.startswith("packed_"):
                counts = doc_counts(doc, n)
                c.require(counts == independent_counts(doc["layer_keep_ratios"], n, doc["r_max"]) and min(counts) >= 1, "PK_counts_independent_nonzero", kind)
                c.require(meta["layer_budget_r_min"] == doc["r_min"] and meta["layer_budget_r_max"] == doc["r_max"]
                          and meta["layer_keep_ratios"] == doc["layer_keep_ratios"], "PK_clip_range_and_ratios_from_document", kind)
                c.require(meta["storage_policy"] == "prefix25_packed" and meta["store_format"] == STORE_FORMAT and meta["budget_policy"] == "layerwise_v1", "PK_policy", kind)
                c.require(meta["layer_keep_counts"] == counts == meta["payload_rows_per_layer"], "PK_counts_recomputed", kind)
                c.require(sum(counts) == 32 * ((n + 3) // 4), "PK_budget_equals_uniform_total", kind)
                c.require(meta["layer_budget_sha256"] == doc["layer_budget_sha256"] == VECTOR_RATIO_SHA and meta["layer_budget_id"] == VECTOR, "PK_vector_identity", kind)
                sep = 2 * nl * len(meta["newline_idx"]) * width
                total = sum(2 * kk * width for kk in counts) + sep
                c.require(meta["packed_total_bytes"] == total == files[PACKED_NAME]["size"] and meta["packed_sep_bytes"] == sep, "PK_exact_file_length", kind)
                c.require(files[PACKED_NAME]["sha256"] == meta["packed_sha256"], "PK_packed_sha256_matches_inventory", kind)
                c.require(not any(f.startswith("layer_") or f == "sep_kv.bin" for f in files), "PK_no_unpacked_payload_files", kind)
            else:
                c.require(meta.get("storage_policy", "full") == "full" and meta["probe_heads"] == IMPRESS["probe_heads"] and meta["chunk_size"] == 64, "full_store_with_probe_sidecar", kind)
                for li in range(nl):
                    ok = (files[f"layer_{li:02d}/k.bin"]["size"] == files[f"layer_{li:02d}/v.bin"]["size"] == vn * width
                          and files[f"layer_{li:02d}/probe_k.bin"]["size"] == vn * IMPRESS["probe_heads"] * meta["head_dim"] * 2)
                    c.require(ok, "full_kv_and_probe_file_lengths", f"{kind}/layer{li}")
                if kind == "raster":
                    c.require(not meta.get("reordered", False) and "order" not in meta, "raster_not_reordered", kind)
                else:
                    t1 = p.get("impress_turn1", {})
                    first = bykey.get((ds[0]["dialog_id"], p["source"]["method_id"], 1), {})
                    c.require(meta.get("reordered") is True and meta.get("importance_metric") == "h2o" and meta.get("calibration_questions") == 1, "impress_reordered_by_one_question_h2o", kind)
                    c.require(t1.get("suffix_ids_sha256") == first.get("suffix_ids_sha256") == meta.get("impress_turn1_suffix_ids_sha256"), "impress_importance_from_first_dialogue_T1_prompt", kind)
                    c.require(first.get("physical_execution_id") == p["source"]["physical_execution_id"], "impress_importance_source_execution", kind)
                    scores_file = artifact / t1.get("scores_file", "missing")
                    ok = scores_file.is_file() and sha(scores_file) == t1.get("scores_sha256")
                    c.require(ok, "impress_scores_evidence", kind)
                    if ok:
                        scores = np.load(scores_file)
                        orders = meta["order"] if meta.get("order_is_per_layer") else [meta["order"]] * nl
                        nls = meta["newline_stored"] if meta.get("order_is_per_layer") else [meta["newline_stored"]] * nl
                        good = scores.shape == (nl, vn)
                        for li in range(nl if good else 0):
                            expect = np.argsort(-scores[li], kind="stable").tolist()
                            good = good and orders[li] == expect and sorted(orders[li][q] for q in nls[li]) == sorted(meta["newline_idx"])
                        c.require(good, "impress_order_is_descending_turn1_importance_per_layer", kind)
        c.require(len(set(builds)) == len(store_kinds(methods)), "distinct_physical_build_events", image_id)
    packed_sets = {}
    for key, r in bykey.items():
        rid = r["request_id"]; did, m, t = key; result = r["result"]; question = expected.get(key)
        if question is None:
            continue
        arm = ARMS[m]; family = arm["family"]
        c.require(r["schema_version"] == SCHEMA and r["model_id"] == MODEL_ID, "schema_model", rid)
        c.require(rid == r["logical_request_id"] == f"{SCHEMA}:{r['experiment_id']}:{phase}:{MODEL_ID}:{m}:{did}:t{t}", "logical_identity", rid)
        c.require(r["phase"] == phase and r["image_id"] == image_id and r["family"] == family and r["async_read"] == arm["async"], "phase_image_arm", rid)
        c.require(r["question_id"] == str(question["question_id"]) and r["question"] == question["question"] and r["gold"] == question["answers"], "frozen_question_gold", rid)
        c.require(r["score"] == strict_score(r["prediction"], r["gold"][0]), "strict_em_recalculation", rid)
        c.require(r["prediction"] == result["answer"], "raw_decoded_prediction", rid)
        c.require(r["config_sha256"] == config["_file_sha256"] and r["manifest_sha256"] == config["_manifest_sha256"], "config_manifest_hash", rid)
        c.require(r["code_sha256"] == canonical_hash(config["source_sha256"]), "code_hash", rid)
        ordinal = ordinals[did]; order = method_order(methods, ordinal)
        c.require(r["global_dialog_ordinal"] == ordinal and r["method_order"] == order and r["method_order_position"] == order.index(m), "deterministic_rotation", rid)
        c.require(r["image_sha256"] == manifest["image_identity"][image_id]["sha256"], "frozen_image_hash", rid)
        history = []; entries = []
        for pt in range(1, t):
            prev = bykey.get((did, m, pt)); q = expected.get((did, m, pt))
            if prev is None:
                c.require(False, "missing_history_source", rid); continue
            history += [f"Q{pt}: {q['question'].strip()}", f"A{pt}: {prev['prediction']}"]
            entries.append({"turn_id": pt, "question_id": str(q["question_id"]), "question": q["question"].strip(), "answer": prev["prediction"],
                            "answer_source": "method_local_generated", "source_model": MODEL_ID, "source_method_id": m, "source_dialog_id": did,
                            "source_logical_request_id": prev["logical_request_id"], "source_physical_execution_id": prev["physical_execution_id"]})
        hist = "\n".join(history); body = ([hist, ""] if hist else [])
        body += [f"Current question Q{t}: {question['question'].strip()}", f"{SHORT_ANSWER} ASSISTANT:"]
        prompt = "USER: <image>\n" + "\n".join(body)
        c.require(r["history"] == hist and r["history_entries"] == entries, "exact_method_local_history_lineage", rid)
        c.require(r["prompt"] == prompt, "causal_prompt_reconstruction", rid)
        c.require(r["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest(), "prompt_hash", rid)
        c.require(r["suffix_ids_sha256"] == canonical_hash(r["suffix_input_ids"]), "suffix_hash", rid)
        c.require(0 < r["ttft_ms"] <= r["request_e2e_ms"], "valid_ttft_e2e", rid)
        c.require(math.isclose(r["ttft_ms"], 1000 * (r["first_token_at_s"] - r["request_started_at_s"]), abs_tol=1e-7), "true_ttft_recalculation", rid)
        tokens = result["generated_token_ids"]
        c.require(1 <= len(tokens) <= 16 and tokens[0] == result["first_token_id"], "actual_generated_tokens", rid)
        pixel = t == 1 or family == "recompute"
        c.require(result["vision_forward_count"] == int(pixel), "vision_call_contract", rid)
        trace = result.get("os_read_trace", [])
        c.require(r["ssd_read_bytes"] == sum(e[4] for e in trace), "actual_returned_bytes", rid)
        c.require(r["ssd_preads"] == len(trace) == r["pread_calls"] + r["preadv_calls"], "actual_read_count", rid)
        c.require(all(e[3] == e[4] for e in trace), "successful_full_requested_reads", rid)
        g = r["geometry"]; n = g["v_token_num"] - len(set(g["newline_idx"]) | set(g["padding_idx"]))
        c.require(n == r["N_content"] and n > 0, "actual_content_geometry", rid)
        calls = result.get("projection_calls", {})
        prefill = {"q": 2, "k": 2, "v": 1} if (family == "impress" and not pixel) else {"q": 1, "k": 1, "v": 1}
        for li in range(g["num_layers"]):
            observed = calls.get(str(li), {})
            c.require(observed.get("prefill") == prefill, "prefill_projection_counts_each_layer", rid)
            c.require(observed.get("decode", {"q": 0, "k": 0, "v": 0}) == {"q": len(tokens) - 1, "k": len(tokens) - 1, "v": len(tokens) - 1}, "decode_projection_counts", rid)
        if pixel:
            c.require(r["ssd_read_bytes"] == 0 and not trace and r["source_store"] is None, "pixel_request_reads_nothing", rid)
            continue
        c.require(result.get("adapter_restored") is True, "adapter_restored", rid)
        conditioned = set(result.get("conditioned_files", []))
        c.require(bool(conditioned) and all(e[1] in conditioned for e in trace), "every_read_file_conditioned_cold", rid)
        c.require(all(e[5] >= 0 for e in trace), "every_read_after_timer_start", rid)
        src = r["source_store"]; c.require(src is not None and src["kind"] == arm["kind"], "store_provenance", rid)
        if src is not None:
            c.require(src["source"]["dialog_id"] == ds[0]["dialog_id"] and kind_of(src["source"]["method_id"]) == arm["kind"], "first_dialogue_T1_source_same_kind", rid)
            source_row = bykey.get((ds[0]["dialog_id"], src["source"]["method_id"], 1), {})
            c.require(src["source"]["physical_execution_id"] == source_row.get("physical_execution_id"), "actual_source_physical_id", rid)
        width = g["num_heads"] * g["head_dim"] * 2; vn = g["v_token_num"]; nl = g["num_layers"]; structural = set(g["newline_idx"])
        wait = result.get("read_wait") or {}
        ahead = wait.get("readahead")
        main_flags = [e[6] for e in trace]
        if family == "fullload":
            exp_reads = Counter()
            for li in range(nl):
                for f in ("k.bin", "v.bin"):
                    exp_reads[(f"layer_{li:02d}/{f}", 0, vn * width, vn * width)] += 1
            c.require(Counter((e[1], e[2], e[3], e[4]) for e in trace) == exp_reads, "fullload_exact_whole_block_ranges", rid)
            c.require(r["preadv_calls"] == 0 and r["pread_calls"] == 2 * nl, "fullload_pread_count", rid)
            c.require([e[1] for e in trace] == [f"layer_{li:02d}/{f}" for li in range(nl) for f in ("k.bin", "v.bin")], "fullload_layer_order", rid)
            if arm["async"]:
                c.require(not any(main_flags), "fullload_async_reads_on_worker", rid)
                c.require(result.get("prefetch") == "layer_stream" and ahead is not None and ahead["items_read"] == ahead["items_taken"] == nl
                          and ahead["max_ahead"] == READAHEAD["max_ahead"], "fullload_async_layer_stream", rid)
            else:
                c.require(all(main_flags) and ahead is None and result.get("prefetch") is None, "fullload_sync_reads_on_request_thread", rid)
        elif family == "ours":
            counts = doc_counts(doc, n)
            c.require(counts == independent_counts(doc["layer_keep_ratios"], n, doc["r_max"]), "PK_k_per_layer_independent", rid)
            c.require(result.get("k_per_layer") == counts, "PK_k_per_layer_recomputed", rid)
            c.require(result.get("attended_content_kv_count_per_layer") == counts and result.get("k_total") == sum(counts) == nl * ((n + 3) // 4), "PK_budget_total", rid)
            c.require(result.get("layer_budget_sha256") == doc["layer_budget_sha256"] and result.get("layer_budget_id") == VECTOR, "PK_vector_identity_hit", rid)
            c.require(result.get("selected_stored_ids") == list(range(max(counts))), "PK_prefix_contract", rid)
            sep = 2 * nl * len(structural) * width
            body = sum(2 * kk * width for kk in counts); total = body + sep
            c.require(r["ssd_read_bytes"] == total and r["pread_calls"] == 0, "PK_bytes_equal_file_no_pread", rid)
            c.require(all(e[0] == "preadv" and e[1] == PACKED_NAME for e in trace), "PK_preadv_on_packed_file_only", rid)
            if arm["async"]:
                plan = [(body, sep)]; off = 0
                for kk in counts:
                    plan.append((off, 2 * kk * width)); off += 2 * kk * width
                plan = [p for p in plan if p[1]]
                c.require([(e[2], e[4]) for e in trace] == plan, "PK_async_ranges_sep_then_layers_tile_file", rid)
                c.require(r["preadv_calls"] == len(plan), "PK_async_read_call_count", rid)
                c.require(not any(main_flags), "PK_async_reads_on_worker", rid)
                c.require(result.get("readahead_policy") == "layer_stream" and ahead is not None and ahead["items_taken"] == nl + 1
                          and ahead["max_ahead"] == READAHEAD["max_ahead"], "PK_async_layer_stream", rid)
            else:
                c.require(len(trace) == 1 and trace[0][2] == 0 and trace[0][4] == trace[0][3] == total, "PK_single_preadv_whole_file", rid)
                c.require(r["preadv_calls"] == 1 and result.get("read_calls") == 1 and all(main_flags), "PK_sync_read_call_counts", rid)
            original = result.get("selected_original_ids", [])
            c.require(len(original) == max(counts) and len(set(original)) == len(original) and all(0 <= i < vn and i not in structural for i in original), "original_ids_real", rid)
            c.require(result.get("keep_count_per_layer") == [kk + len(structural) for kk in counts], "structural_keep", rid)
            c.require(result.get("store_format") == STORE_FORMAT and result.get("directcopy") is False, "PK_scatter_path", rid)
            packed_sets.setdefault(m, []).append((original, counts, total))
            c.require(result.get("query_score_calls", 0) == 0, "online_scoring_zero", rid)
        else:      # impress
            ratio = arm["ratio"]
            k_keep = max(1, min(n, math.ceil(ratio * n)))
            thr = ((k_keep / n) / (2.0 - k_keep / n)) ** IMPRESS["alpha"]
            c.require(result.get("k_keep") == k_keep and math.isclose(result.get("threshold", -1), thr, rel_tol=1e-12), "impress_budget_and_threshold", rid)
            c.require(result.get("importance_metric") == "h2o" and result.get("fallback_v") == "selected"
                      and result.get("impress_params") == {"ratio": ratio, "probe": IMPRESS["probe_heads"], "alpha": IMPRESS["alpha"]}, "impress_parameters", rid)
            c.require(result.get("n_score_rows") == len(r["suffix_input_ids"]), "impress_query_rows_are_whole_suffix", rid)
            log = result.get("layer_log", [])
            c.require([x["layer"] for x in log] == list(range(nl)), "impress_layer_log_complete", rid)
            probe_bytes = vn * IMPRESS["probe_heads"] * g["head_dim"] * 2
            probes = [e for e in trace if e[1].endswith("/probe_k.bin")]
            c.require([e[1] for e in probes] == [f"layer_{li:02d}/probe_k.bin" for li in range(nl)] and all(e[2] == 0 and e[4] == probe_bytes for e in probes), "impress_probe_reads_one_per_layer", rid)
            c.require(r["probe_read_bytes"] == nl * probe_bytes, "impress_probe_bytes", rid)
            c.require(r["probe_read_bytes"] + r["k_read_bytes"] + r["v_read_bytes"] == r["ssd_read_bytes"] and r["preadv_calls"] == 0, "impress_bytes_by_kind_sum", rid)
            ok_layers = True; kbytes = vbytes = 0
            for x in log:
                li = x["layer"]; got = {}
                for name in ("k", "v"):
                    es = [e for e in trace if e[1] == f"layer_{li:02d}/{name}.bin"]
                    ok_layers = ok_layers and all(e[2] % (64 * width) == 0 and e[4] % width == 0 and e[2] + e[4] <= vn * width for e in es)
                    got[name] = sorted({ch for e in es for ch in _chunks_of(e[2], e[4], width, vn)})
                    b = sum(e[4] for e in es)
                    kbytes += b if name == "k" else 0; vbytes += b if name == "v" else 0
                    ok_layers = ok_layers and b == sum(min(64, vn - ch * 64) for ch in got[name]) * width
                ok_layers = ok_layers and len(got["k"]) == x["k_chunks"] and len(got["v"]) == x["v_chunks"]
                if x["mode"] == "probe":
                    ok_layers = ok_layers and got["k"] == got["v"] and x["sim"] > thr
                else:
                    ok_layers = ok_layers and x["k_chunks"] == (vn + 63) // 64 and x["sim"] <= thr
            c.require(ok_layers, "impress_kv_reads_match_layer_decisions", rid)
            c.require(kbytes == r["k_read_bytes"] and vbytes == r["v_read_bytes"], "impress_kv_bytes_from_trace", rid)
            fb = sum(x["mode"] == "fallback" for x in log)
            c.require(math.isclose(result.get("fallback_rate", -1), fb / nl), "impress_fallback_rate_recomputed", rid)
            un = result.get("unreordered_chunks", {})
            c.require(len(un.get("k_chunks", [])) == nl and len(un.get("v_chunks", [])) == nl, "impress_raster_counterfactual_present", rid)
            sources = result.get("probe_sources", [])
            kv_main = all(e[6] == 1 for e in trace if not e[1].endswith("/probe_k.bin"))
            if arm["async"]:
                c.require(result.get("prefetch") == "next_layer" and [s[1] for s in sources] == ["hook"] + ["prefetch"] * (nl - 1), "impress_async_next_layer_probe_prefetch", rid)
                c.require(probes and probes[0][6] == 1 and not any(e[6] for e in probes[1:]) and kv_main, "impress_async_only_probe_reads_on_worker", rid)
            else:
                c.require(result.get("prefetch") == "none" and all(s[1] == "hook" for s in sources) and all(main_flags), "impress_sync_reads_on_request_thread", rid)
            if ratio == 1.0:
                c.require(fb == nl and all(x["v_chunks"] == (vn + 63) // 64 for x in log), "impress_r100_loads_everything", rid)
    for m, sets in packed_sets.items():
        c.require(all(s == sets[0] for s in sets), "same_image_selection_invariant", f"{image_id}/{m}")
        if metas:
            meta = metas[kind_of(m)]
            c.require(sets[0][0] == meta["order"][:len(sets[0][0])] and sets[0][2] == meta["packed_total_bytes"], "meta_to_actual_selection_and_bytes", f"{image_id}/{m}")
    return dict(c.result(), image_id=image_id, phase=phase, requests=len(rows), expected_requests=len(expected), dialogs=len(ds))


# ---------------------------------------------------------------- lifecycle
def completed_images(run, phase, config, manifest):
    done = {}
    for cp in sorted((run / phase / "images").glob("*/COMMITTED.json")):
        doc = json.loads(cp.read_text()); folder = cp.parent / doc["attempt_id"]; raw = folder / "raw.jsonl"
        if doc["config_sha256"] != config["_file_sha256"] or doc["manifest_sha256"] != config["_manifest_sha256"]:
            raise RuntimeError("resume config/manifest mismatch")
        if sha(raw) != doc["raw_sha256"]:
            raise RuntimeError("committed raw hash mismatch")
        for name, digest in doc["evidence_sha256"].items():
            if sha(folder / name) != digest:
                raise RuntimeError("committed evidence changed")
        done[cp.parent.name] = doc
    return done


def scratch_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file() and not p.is_symlink())


def run_image(runner, servers, dialogs, ordinals, phase, run, manifest, config):
    methods = config["phases"][phase]["methods"]; docs = config["layer_budgets"]
    assert len({d["image_id"] for d in dialogs}) == 1
    image_id = dialogs[0]["image_id"]; parent = run / phase / "images" / image_id
    parent.mkdir(parents=True, exist_ok=True)
    if (parent / "COMMITTED.json").exists():
        raise FileExistsError("image already committed")
    attempt = f"attempt_{len(list(parent.glob('attempt_*'))) + 1:04d}"
    artifact = parent / attempt; artifact.mkdir(exist_ok=False)
    scratch = run / "scratch" / phase / image_id / attempt
    atomic_json(artifact / "capacity_before.json", capacity_guard(run))
    scratch.mkdir(parents=True, exist_ok=False)
    kinds = {m: kind_of(m) for m in stored_methods(methods)}
    all_kinds = store_kinds(methods)
    ref_kind = all_kinds[0]
    paths = {k: scratch / k for k in all_kinds}
    metas, hashes, persistence, contexts, store_info = {}, {}, {}, {}, {}
    rows = []; pending = []; success = False; capturing = set(); peak_scratch = 0
    source_image_sha = sha(ROOT / dialogs[0]["image_path"])
    atomic_json(artifact / "image_identity.json", {"image_id": image_id, "image_path": str((ROOT / dialogs[0]["image_path"]).resolve()), "sha256": source_image_sha})
    try:
        for di, dialog in enumerate(dialogs):
            ordinal = ordinals[dialog["dialog_id"]]; generated = {m: {} for m in methods}
            for turn in (1, 2, 3):
                for method in method_order(methods, ordinal):
                    assert_gpu_exclusive(); physical = str(uuid.uuid4())
                    logical = request_id(run.name, phase, method, dialog["dialog_id"], turn)
                    source = {"dialog_id": dialog["dialog_id"], "method_id": method, "physical_execution_id": physical, "logical_request_id": logical}
                    append(artifact / "attempts.jsonl", dict(source, turn_id=turn, status="STARTED", at_unix=time.time()))
                    factory = lambda m=method: prompt_factory(methods, dialog, turn, m, generated[m])
                    kind = kinds.get(method, "none")
                    # the first arm of a store kind in the first dialogue's rotation builds that kind's store
                    capture = kind if di == 0 and turn == 1 and kind != "none" and kind not in capturing else "none"
                    if capture != "none":
                        capturing.add(capture)
                    if turn == 1 or method == "recompute":
                        result, diag = normal_request(runner, servers["base"], ROOT / dialog["image_path"], factory,
                                                      "raster" if capture in ("raster", "impress") else ("image_only" if capture != "none" else "none"))
                        append(artifact / "execution_events.jsonl", dict(source, turn_id=turn, status="EXECUTED", prediction=result["answer"],
                                                                        generated_token_ids=result["generated_token_ids"], ttft_ms=result["end_to_end_ttft_ms"],
                                                                        request_e2e_ms=result["request_e2e_ms"]))
                        if capture != "none":
                            start = time.perf_counter()
                            p = persist(runner, servers["base"], result, diag, paths[capture], image_id, capture, source, docs,
                                        artifact / f"{capture}_turn1_importance.npy")
                            outer_ms = (time.perf_counter() - start) * 1000
                            persistence[capture] = dict(p, outer_helper_wall_ms=outer_ms, source=source,
                                                        capture_accounting="capture and exit materialization already within T1 request E2E; not added again")
                            persistence[capture].pop("meta", None)
                            metas[capture] = json.loads((paths[capture] / "meta.json").read_text())
                            hashes[capture] = file_hashes(paths[capture])
                            store_info[capture] = {"kind": capture, "path": str(paths[capture]), "meta_sha256": sha(paths[capture] / "meta.json"),
                                                   "source": source, "physical_build_event_id": p["event_id"]}
                            atomic_json(artifact / f"{capture}_persistence.json", persistence[capture])
                            atomic_json(artifact / f"{capture}_meta.json", metas[capture])
                            atomic_json(artifact / f"{capture}_hashes.json", hashes[capture])
                            peak_scratch = max(peak_scratch, scratch_bytes(scratch))
                        result.pop("captured_past_key_values", None)
                        saved = {a: b for a, b in diag.items() if a not in ("enc_cpu", "hidden", "vision")}
                        if di == 0 and turn == 1:
                            pending.append((method, result, saved, physical))
                            del diag, result
                            continue
                        diag = saved
                    else:
                        if kind not in contexts:
                            contexts[kind] = activate(runner, kind, paths[kind])
                            atomic_json(artifact / f"{kind}_activation.json", contexts[kind].activation)
                        result, diag = stored_request(runner, servers, contexts[kind], factory, method, image_id, paths[kind])
                    meta_kind = kind if kind != "none" else ref_kind
                    row = make_row(runner, result, diag, metas[meta_kind], methods=methods, method=method, dialog=dialog, turn=turn, phase=phase, run=run,
                                   attempt=attempt, physical_id=physical, ordinal=ordinal, store_info=store_info.get(kind), config=config)
                    row["image_sha256"] = source_image_sha
                    append(artifact / "raw.jsonl", row); rows.append(row); generated[method][turn] = row
                    append(artifact / "attempts.jsonl", dict(source, turn_id=turn, status="COMPLETED", at_unix=time.time()))
                    del result, diag
                if di == 0 and turn == 1:
                    assert set(metas) == set(all_kinds)
                    for method, result, diag, physical in pending:
                        meta_kind = kinds.get(method, ref_kind)
                        row = make_row(runner, result, diag, metas[meta_kind], methods=methods, method=method, dialog=dialog, turn=turn, phase=phase, run=run,
                                       attempt=attempt, physical_id=physical, ordinal=ordinal, store_info=None, config=config)
                        row["image_sha256"] = source_image_sha
                        append(artifact / "raw.jsonl", row); rows.append(row); generated[method][1] = row
                        append(artifact / "attempts.jsonl", {"physical_execution_id": physical, "logical_request_id": row["logical_request_id"], "status": "COMPLETED", "at_unix": time.time()})
                    pending.clear(); del result, diag
                    for kind in metas:
                        assert metas[kind]["prefix_input_ids"] == metas[ref_kind]["prefix_input_ids"]
                        assert metas[kind]["image_input_sha256"] == metas[ref_kind]["image_input_sha256"]
        for kind in all_kinds:
            assert file_hashes(paths[kind]) == hashes[kind], "store mutated during requests"
        peak_scratch = max(peak_scratch, scratch_bytes(scratch))
        atomic_json(artifact / "store_integrity_after.json", {"status": "PASS", "file_sha256": hashes})
        atomic_json(artifact / "disk_usage.json", {"peak_scratch_bytes": peak_scratch, "free_bytes_after_requests": shutil.disk_usage(run).free,
                                                   "store_bytes": {k: scratch_bytes(paths[k]) for k in all_kinds}})
        audit = audit_image_rows(load_jsonl(artifact / "raw.jsonl"), manifest, config, phase, image_id, artifact)
        audit["raw_sha256"] = sha(artifact / "raw.jsonl"); atomic_json(artifact / "independent_audit.json", audit)
        if audit["status"] != "PASS":
            raise RuntimeError("INDEPENDENT_IMAGE_AUDIT_FAIL: " + repr(audit["failures"][:10]))
        allowlist = file_hashes(scratch)
        atomic_json(artifact / "cleanup_allowlist.json", {"scratch": str(scratch), "file_sha256": allowlist, "ownership_run": run.name})
        commit = {"status": "COMMITTED", "image_id": image_id, "attempt_id": attempt, "requests": len(rows),
                  "raw_sha256": audit["raw_sha256"], "audit_sha256": sha(artifact / "independent_audit.json"),
                  "config_sha256": config["_file_sha256"], "manifest_sha256": config["_manifest_sha256"]}
        commit["evidence_sha256"] = {p.name: sha(p) for p in artifact.iterdir() if p.is_file()}
        atomic_json(parent / "COMMITTED.json", commit); success = True
    except BaseException as exc:
        atomic_json(artifact / "failure.json", {"status": "PARTIAL", "error": repr(exc), "traceback": traceback.format_exc(),
                                                "completed_uncommitted_rows": len(rows), "main_final_rows_from_this_attempt": 0})
        raise
    finally:
        for ctx in contexts.values():
            ctx.close()
        contexts.clear(); gc.collect()
        if success:
            cleanup_committed(scratch, allowlist, audit, commit, run)
            atomic_json(artifact / "cleanup_receipt.json", {"status": "PASS", "removed_new_files": len(allowlist)})
    return len(rows)


def verify_freeze(run):
    config = json.loads((run / "config.json").read_text()); manifest = json.loads((run / "manifest.json").read_text())
    if source_hashes() != config["source_sha256"]:
        raise RuntimeError("FROZEN_SOURCE_CONFIG_MISMATCH")
    if sha(CONTRACT) != config["contract_sha256"]:
        raise RuntimeError("FROZEN_CONTRACT_MISMATCH")
    if frozen_dialogues() != manifest["dialogues"]:
        raise RuntimeError("FROZEN_DATASET_MISMATCH")
    for image_id, item in manifest["image_identity"].items():
        if sha(ROOT / item["image_path"]) != item["sha256"]:
            raise RuntimeError("FROZEN_IMAGE_MISMATCH: " + image_id)
    for alloc, doc in config["layer_budgets"].items():
        fresh = load_budget_document(doc["path"])
        assert fresh["layer_budget_sha256"] == doc["layer_budget_sha256"] and fresh["layer_keep_ratios"] == doc["layer_keep_ratios"]
    import transformers
    if transformers.__version__ != config["environment"]["transformers"] or torch.__version__ != config["environment"]["torch"]:
        raise RuntimeError("FROZEN_ENVIRONMENT_MISMATCH")
    config.update(_file_sha256=sha(run / "config.json"), _manifest_sha256=sha(run / "manifest.json"))
    return config, manifest


def run_phase(runner, servers, run, phase, config, manifest):
    done = completed_images(run, phase, config, manifest)
    ids = manifest["phase_dialogue_ids"][phase]
    keep = set(ids); groups = OrderedDict(); ordinals = {}
    for ordinal, d in enumerate(manifest["dialogues"]):
        ordinals[d["dialog_id"]] = ordinal
        if d["dialog_id"] in keep:
            groups.setdefault(d["image_id"], []).append(d)
    per_dialog = 3 * len(config["phases"][phase]["methods"])
    expected = len(ids) * per_dialog; completed = sum(x["requests"] for x in done.values())
    assert expected == manifest["expected_requests"][phase]
    started = time.time()
    for index, (image, ds) in enumerate(groups.items()):
        if image in done:
            continue
        if (run / "STOP_REQUESTED").exists():
            raise RuntimeError("STOP_REQUESTED")
        t0 = time.time()
        completed += run_image(runner, servers, ds, ordinals, phase, run, manifest, config)
        done[image] = {"requests": len(ds) * per_dialog}
        progress = {"phase": phase, "status": "RUNNING", "completed_final_requests": completed, "expected_requests": expected,
                    "completed_images": len(done), "images": len(groups), "image_id": image, "image_seconds": time.time() - t0,
                    "at_unix": time.time(), "elapsed_s_this_process": time.time() - started}
        atomic_json(run / f"{phase}_progress.json", progress, replace=True); print(json.dumps(progress), flush=True)
    assert completed == expected, (completed, expected)
    receipt = {"status": "PASS", "requests": completed, "expected_requests": expected, "images": len(groups),
               "independent_image_audit": "PASS", "config_sha256": config["_file_sha256"]}
    path = run / phase / "validation.json"
    if path.exists():
        if json.loads(path.read_text()) != receipt:
            raise RuntimeError("final phase validation mismatch")
    else:
        atomic_json(path, receipt)
    return receipt


# ------------------------------------------------------------------- freeze
def stat_inventory(run, listing=None):
    """(size, mtime_ns) of every existing file under results/, runs/, kvstore*
    outside this run: the before/after evidence that nothing was touched."""
    own = {run.resolve(), (ROOT / "results" / run.name).resolve()}
    digest = hashlib.sha256(); files = 0; total = 0; lines = []
    roots = [ROOT / "results", ROOT / "runs"] + sorted(p for p in ROOT.glob("kvstore*") if p.is_dir())
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            here = Path(dirpath)
            dirnames[:] = sorted(d for d in dirnames if (here / d).resolve() not in own)
            for name in sorted(filenames):
                p = here / name
                try:
                    st = p.lstat()
                except FileNotFoundError:
                    continue
                line = f"{p.relative_to(ROOT)}\t{st.st_size}\t{st.st_mtime_ns}\n"
                digest.update(line.encode()); lines.append(line)
                files += 1; total += st.st_size
    if listing is not None:
        import gzip
        with gzip.open(listing, "wt") as f:
            f.writelines(lines)
    return {"files": files, "bytes": total, "sha256": digest.hexdigest(), "roots": [str(r.relative_to(ROOT)) for r in roots],
            "fingerprint": "sha256 over lines 'relative path<TAB>size<TAB>mtime_ns' of every file, sorted walk; this run's own directories excluded"}


def freeze(run):
    import transformers
    run.mkdir(parents=True, exist_ok=False)
    assert run.name.startswith("mt_gqa_async_") and CONTRACT.is_file()
    ds = frozen_dialogues()
    prior_manifest = json.loads((PRIOR / "manifest.json").read_text())
    assert prior_manifest["dialogues"] == ds
    for image, identity in prior_manifest["image_identity"].items():
        assert sha(ROOT / identity["image_path"]) == identity["sha256"]
    hold_manifest = json.loads((PYRBETA_RUN / "hold" / "manifest.json").read_text())
    assert hold_manifest["dialogues"] == ds and hold_manifest["image_identity"] == prior_manifest["image_identity"]
    assert hold_manifest["index_sha256"] == INDEX_SHA and hold_manifest["workload_sha256"] == WORKLOAD_SHA
    smoke_ids = hold_manifest["smoke_dialogue_ids"]
    smoke_images = [d["image_id"] for d in ds if d["dialog_id"] in set(smoke_ids)]
    dev_images = json.loads(V2_SPLIT.read_text())["dev"]["image_ids"]
    all_images = {d["image_id"] for d in ds}
    assert len(dev_images) == 40 and set(dev_images) <= all_images and len(set(smoke_images)) == 4
    phase_ids = {"gate": [d["dialog_id"] for d in ds if d["image_id"] in set(smoke_images)],
                 "dev": [d["dialog_id"] for d in ds if d["image_id"] in set(dev_images)],
                 "main": [d["dialog_id"] for d in ds]}
    manifest = {"dataset": "MT-GQA-reconstructed", "schema_version": SCHEMA, "index_sha256": INDEX_SHA, "workload_sha256": WORKLOAD_SHA,
                "dialogues": ds, "image_identity": prior_manifest["image_identity"], "phase_dialogue_ids": phase_ids,
                "phase_images": {"gate": smoke_images, "dev": dev_images, "main": sorted(all_images)},
                "phase_methods": PHASE_METHODS,
                "expected_requests": {p: len(phase_ids[p]) * 3 * len(PHASE_METHODS[p]) for p in PHASES},
                "prior_manifest_sha256": sha(PRIOR / "manifest.json"), "pyrbeta_hold_manifest_sha256": sha(PYRBETA_RUN / "hold" / "manifest.json"),
                "gate_policy": "every dialogue of the four images of the HOLD smoke dialogue ids",
                "dev_policy": "every MT-GQA dialogue of the 40 DEV images of results/layer_budget_kv25_v2_20261106T073502Z/data_split.json"}
    assert manifest["expected_requests"]["main"] == 12183 * 4
    support.publish(run / "manifest.json", manifest)
    free = shutil.disk_usage(run).free
    # per image: raster + IMPRESS full stores (K/V + probe + sidecar + hidden) and one packed store, at the largest geometry (N = 2880)
    full_store = 2 * 32 * 2928 * 8192 + 32 * 2928 * 3 * 128 * 2 + 2 * 32 * 48 * 8192 + (64 << 20)
    packed_store = 411_041_792
    bundle = 2 * full_store + packed_store + 2 * 2928 * 8192      # + one layer file in flight during the IMPRESS repack
    plan = {"policy": "FRESH_IMAGE_STREAMING", "minimum_reserve_bytes": 30 << 30, "peak_working_set_bytes": bundle + (4 << 30),
            "largest_image_bundle_bound_bytes": bundle, "maximum_new_payload_bytes": 20 << 30, "raw_and_evidence_growth_bound_bytes": 8 << 30,
            "free_bytes_at_freeze": free, "full_store_bound_bytes": full_store, "packed_store_bound_bytes": packed_store,
            "deletion_scope": "runs/<RUN>/scratch only (exact SHA allowlist after audit+commit); existing stores never deleted"}
    plan["required_free_bytes"] = plan["minimum_reserve_bytes"] + plan["peak_working_set_bytes"] + plan["raw_and_evidence_growth_bound_bytes"]
    plan["status"] = "PASS" if free >= plan["required_free_bytes"] else "BLOCKED_STORAGE"
    support.publish(run / "storage_plan.json", plan)
    if plan["status"] != "PASS":
        raise RuntimeError(f"BLOCKED_STORAGE: free {free} < required {plan['required_free_bytes']}")
    p = PYRBETA_RUN / "vectors" / f"layer_budget_llava_{VECTOR}.json"
    doc = load_budget_document(p)
    hold_doc = json.loads((PYRBETA_RUN / "hold" / "config.json").read_text())["layer_budgets"][VECTOR]
    assert doc["layer_budget_sha256"] == VECTOR_RATIO_SHA == hold_doc["layer_budget_sha256"] and sha(p) == VECTOR_FILE_SHA
    assert doc["alloc_id"] == VECTOR and doc["layer_keep_ratios"] == hold_doc["layer_keep_ratios"] and doc["r_min"] == 0.0 == hold_doc["r_min"]
    budgets = {VECTOR: {"path": str(p), "layer_budget_sha256": doc["layer_budget_sha256"], "layer_keep_ratios": doc["layer_keep_ratios"],
                        "r_min": doc["r_min"], "r_max": doc["r_max"], "alloc_id": VECTOR, "policy": doc["policy"], "schema": doc["schema"],
                        "num_layers": doc["num_layers"], "total_ratio": doc["total_ratio"], "source": doc["source"]}}
    config = {"schema_version": SCHEMA, "model": MODEL_ID, "revision": REVISION, "seed": 1234, "greedy": True, "max_new_tokens": 16,
              "batch_size": 1, "cpu_threads": 2, "history_policy": "method_local_generated",
              "phases": {ph: {"methods": PHASE_METHODS[ph], "dialogues": len(phase_ids[ph])} for ph in PHASES},
              "arms": {m: dict(a) for m, a in ARMS.items()}, "impress": IMPRESS, "readahead": READAHEAD,
              "packed_storage_policy": "prefix25_packed", "store_format": STORE_FORMAT, "layer_budgets": budgets,
              "source_sha256": source_hashes(), "contract_sha256": sha(CONTRACT),
              "manifest_sha256": sha(run / "manifest.json"), "storage_plan_sha256": sha(run / "storage_plan.json"),
              "source_T1_capture": "first frozen dialogue per image; per store kind, the first arm of that kind in the dialogue's rotation; every arm still runs its own full-pixel T1",
              "reference_runs": {"recompute_fullload": str((V2_RUN / "hold").relative_to(ROOT)), "ours": str((PYRBETA_RUN / "hold").relative_to(ROOT))},
              "environment": {"python": sys.version, "executable": sys.executable, "torch": torch.__version__, "transformers": transformers.__version__,
                              "numpy": np.__version__},
              "lifecycle_source": "scripts/164_eval_layer_budget_pyrbeta_hold.py (imported, not copied)",
              "lifecycle_source_sha256": sha(ROOT / "scripts/164_eval_layer_budget_pyrbeta_hold.py")}
    support.publish(run / "config.json", config)
    shutil.copy2(CONTRACT, run / "CONTRACT.md")
    env = {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
           "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
           "pip_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)}
    support.publish(run / "environment_before.json", env)
    support.publish(run / "protection_before.json", stat_inventory(run, run / "protection_before.tsv.gz"))
    print(json.dumps({"freeze": "PASS", "run": str(run), "expected": manifest["expected_requests"], "storage": plan["status"],
                      "free_gib": free / 2**30, "transformers": transformers.__version__}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True, help="runs/mt_gqa_async_<UTC>")
    p.add_argument("--stage", choices=("freeze",) + PHASES, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    run = args.run.resolve()
    if args.stage == "freeze":
        return freeze(run)
    config, manifest = verify_freeze(run)
    assert json.loads((run / "storage_plan.json").read_text())["status"] == "PASS"
    phase = args.stage
    if phase == "main":      # the main run is admitted only by the recorded gate decision
        admit = json.loads((run / "gates" / "admission.json").read_text())
        if admit["status"] != "PASS" or admit["config_sha256"] != config["_file_sha256"]:
            raise RuntimeError("MAIN_NOT_ADMITTED")
    torch.set_num_threads(config["cpu_threads"]); torch.manual_seed(1234); np.random.seed(1234)
    torch.set_grad_enabled(False)
    assert_gpu_exclusive(); runner = V.load_runner()
    from mmimpress.serve import Server
    servers = {"base": Server(runner, ratio=IMPRESS["ratio"], probe=IMPRESS["probe_heads"], alpha=IMPRESS["alpha"], max_new_tokens=16),
               "r100": Server(runner, ratio=1.0, probe=IMPRESS["probe_heads"], alpha=IMPRESS["alpha"], max_new_tokens=16)}
    QA._warmup(runner, servers["base"])
    servers["base"]._prefetch_pool()      # the single read-ahead worker thread exists before any request
    try:
        atomic_json(run / f"{phase}_start_{time.strftime('%Y%m%dT%H%M%S')}.json",
                    {"started_unix": time.time(), "capacity": capacity_guard(run), "pid": os.getpid()})
        if (run / phase).exists() and not args.resume:
            raise FileExistsError("use --resume; previous artifacts preserved")
        receipt = run_phase(runner, servers, run, phase, config, manifest)
        atomic_json(run / f"{phase}_execution_complete.json", {"status": "PASS", **receipt, "finished_unix": time.time()}, replace=True)
    except BaseException as exc:
        error = {"status": "PARTIAL", "error": repr(exc), "traceback": traceback.format_exc(),
                 "resume_command": f"{sys.executable} scripts/187_eval_mt_gqa_async.py --run {run} --stage {phase} --resume"}
        atomic_json(run / ("interruption_" + phase + "_" + time.strftime("%Y%m%dT%H%M%S") + ".json"), error)
        print(json.dumps({k: v for k, v in error.items() if k != "traceback"}), flush=True)
        raise
    finally:
        for server in servers.values():
            server.close()


if __name__ == "__main__":
    main()

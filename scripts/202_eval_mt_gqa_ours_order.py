#!/usr/bin/env python3
"""MT-GQA multi-turn: ReComp / FullLoad / Ours(재배열) / Ours(원래 순서)
(contract docs/mt_gqa_ours_order_contract.md).

This is scripts/196_eval_mt_gqa_impress_code.py (the 2026-11-17 harness, itself
scripts/187 + scripts/164's lifecycle) with the arms of this run:

* the IMPRESS arms and every branch that served them are removed;
* ``ours_prefixstore_kv25_PYR-beta-raw_packed_async`` is unchanged: the packed
  store in VisionZip (importance) row order, the default of
  ``mmimpress.packed_store``;
* ``ours_origorder_kv25_PYR-beta-raw_packed*`` serve the SAME per-layer token
  sets from a store written with ``row_order="original"`` (rows in ascending
  original token index, loaded to their original cache positions);
* gate-only arms: the synchronous twin of the original-order arm and a second
  asynchronous execution of it (reproducibility);
* after an image's requests, the two packed stores are compared byte for byte
  (``store_equivalence``) and the image audit requires that comparison to pass;
* arm rotation: the cyclic rotation of the previous runs, plus the two packed
  arms exchanging their slots every ``len(arms)`` dialogues, so that neither
  always runs right after the other.

Stages: --stage freeze | gate | devgate | dev | main [--resume]
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

from mmimpress.layer_budget import load_budget_document
from mmimpress.packed_serve import PackedImageContext, PackedReadTrace, request_packed
from mmimpress.packed_store import (IMPORTANCE_ORDER_KEYS, ORIGINAL_STORE_FORMAT, PACKED_NAME, ROW_ORDER_IMPORTANCE,
                                    ROW_ORDER_ORIGINAL, STORE_FORMAT, layer_original_indices,
                                    persist_captured_packed_prefix)

SCHEMA = "llava-mtgqa-ours-order-v1"
CONTRACT = ROOT / "docs/mt_gqa_ours_order_contract.md"
RUN_PREFIX = "mt_gqa_ours_order_"
PRIOR_MAIN_RUN = ROOT / "runs/mt_gqa_impress_code_20261117T185832Z"      # continuity reference and the stored IMPRESS rows
PRIOR_IMPRESS = "impress_official_async"
PYRBETA_RUN = ROOT / "runs/layer_budget_kv25_pyrbeta_20261110T073604Z"
V2_SPLIT = ROOT / "results/layer_budget_kv25_v2_20261106T073502Z/data_split.json"
VECTOR = "PYR-beta-raw"
VECTOR_RATIO_SHA = "51f9b0db19a93f8ca8b34b8beb8af21285721c5ea460566afbd9a3185b331724"
VECTOR_FILE_SHA = "8bfc4cbc55eb778a7ef0433ade4bf13f7a24107b76515f23a72ac01e0740ea5d"
OURS_ASYNC = f"ours_prefixstore_kv25_{VECTOR}_packed_async"            # Ours(재배열); the method id of the previous runs
ORIG = f"ours_origorder_kv25_{VECTOR}_packed"                          # Ours(원래 순서), synchronous (gate only)
ORIG_ASYNC = ORIG + "_async"                                           # Ours(원래 순서)
ORIG_REPEAT = ORIG + "_async_repeat"                                   # the same arm executed a second time (gate only)
PACKED_KIND = f"packed_{VECTOR}"
ORIG_KIND = f"packedorig_{VECTOR}"
PACKED_KINDS = {PACKED_KIND: ROW_ORDER_IMPORTANCE, ORIG_KIND: ROW_ORDER_ORIGINAL}
STORE_FORMATS = {PACKED_KIND: STORE_FORMAT, ORIG_KIND: ORIGINAL_STORE_FORMAT}
# constructor arguments of mmimpress.serve.Server and the raster writer's probe sidecar width, as in the 2026-11-17 run
SERVER_ARGS = {"probe_heads": 3, "alpha": 0.6, "ratio": 0.25, "chunk_size": 64}
READAHEAD = {"fullload_async": "layer_stream", OURS_ASYNC: "layer_stream", ORIG_ASYNC: "layer_stream", ORIG_REPEAT: "layer_stream",
             "max_ahead": None}

ARMS = OrderedDict([
    ("recompute",      {"family": "recompute", "kind": "none", "async": False, "display": "ReComp"}),
    ("fullload_async", {"family": "fullload", "kind": "raster", "async": True, "display": "FullLoad"}),
    (OURS_ASYNC,       {"family": "ours", "kind": PACKED_KIND, "async": True, "display": "Ours(재배열)"}),
    (ORIG,             {"family": "ours", "kind": ORIG_KIND, "async": False, "display": "Ours(원래 순서), 동기 읽기"}),
    (ORIG_ASYNC,       {"family": "ours", "kind": ORIG_KIND, "async": True, "display": "Ours(원래 순서)"}),
    (ORIG_REPEAT,      {"family": "ours", "kind": ORIG_KIND, "async": True, "display": "Ours(원래 순서), 두 번째 실행"}),
])
MAIN_METHODS = ["recompute", "fullload_async", OURS_ASYNC, ORIG_ASYNC]
PHASE_METHODS = {
    "gate": ["recompute", "fullload_async", OURS_ASYNC, ORIG, ORIG_ASYNC, ORIG_REPEAT],
    "devgate": [ORIG, ORIG_ASYNC, ORIG_REPEAT],
    "dev": list(MAIN_METHODS),
    "main": list(MAIN_METHODS),
}
PHASES = ("gate", "devgate", "dev", "main")
PHASE_IMAGES = {"gate": "gate", "devgate": "dev", "dev": "dev", "main": "main"}


def helper(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


H = helper("164_eval_layer_budget_pyrbeta_hold.py", "_mtgqa_ours_order_hold_lifecycle")
V, QA, H1 = H.V, H.QA, H.H1
import main_b_support as support

MODEL_ID, REVISION, INDEX_SHA, WORKLOAD_SHA, PRIOR, SHORT_ANSWER = (
    H.MODEL_ID, H.REVISION, H.INDEX_SHA, H.WORKLOAD_SHA, H.PRIOR, H.SHORT_ANSWER)
sha, canonical_hash, safe, sync_dir, atomic_json, append, load_jsonl = (
    H.sha, H.canonical_hash, H.safe, H.sync_dir, H.atomic_json, H.append, H.load_jsonl)
frozen_dialogues, strict_score, assert_gpu_exclusive, conditioning_trace, PilotCounts = (
    H.frozen_dialogues, H.strict_score, H.assert_gpu_exclusive, H.conditioning_trace, H.PilotCounts)
normal_request, file_hashes, Checks, capacity_guard, cleanup_committed = (
    H.normal_request, H.file_hashes, H.Checks, H.capacity_guard, H.cleanup_committed)
doc_counts, independent_counts = H.doc_counts, H.independent_counts


# report scripts read committed rows only and are not part of the measurement freeze
REPORT_ONLY = ("198_report_mt_gqa_impress_code.py", "201_report_llava15_mtvqa_impress_code.py", "204_report_mt_gqa_ours_order.py")


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


def method_order(methods, ordinal):
    """Arm order of one dialogue.

    The cyclic rotation of the previous runs (start at ``ordinal % n``).  A
    cyclic rotation keeps every arm's predecessor fixed, so the two packed arms
    additionally exchange their slots in every other block of ``n`` dialogues:
    over two blocks each of them takes every (position, predecessor) once.
    """
    base = list(methods)
    if OURS_ASYNC in base and ORIG_ASYNC in base and (int(ordinal) // len(base)) % 2 == 1:
        i, j = base.index(OURS_ASYNC), base.index(ORIG_ASYNC)
        base[i], base[j] = base[j], base[i]
    r = int(ordinal) % len(base)
    return base[r:] + base[:r]


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
    """Times ``Path.write_bytes`` per file name while the raster writer runs."""

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
        with PilotCounts(runner) as counts, ThreadedReadTrace() as reads:
            if family == "fullload":
                result = servers["base"].request(ctx, mode="fullload", suffix_ids=suffix, cold=False, prompt_text=prompt, cache_id=image_id,
                                                 prefetch="layer_stream" if arm["async"] else "none",
                                                 max_ahead=READAHEAD["max_ahead"], selector_out=[])
            else:      # both packed arms: the same call, the store decides the row order
                result = request_packed(servers["base"], ctx, suffix_ids=suffix, cold=False, image_id=image_id, directcopy=False,
                                        readahead="layer_stream" if arm["async"] else "none", max_ahead=READAHEAD["max_ahead"])
        returned = time.perf_counter()
    s = reads.summary()
    assert s["bytes"] == result["io"]["bytes"] and s["preads"] == result["io"]["preads"]
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
def _native_persist(runner, result, diag, path, image_id, kind, source, docs):
    from mmimpress.piggyback import persist_captured_raster_prefix
    enc = diag["enc_cpu"]
    args = (runner, result["captured_past_key_values"], enc["input_ids"], enc["image_sizes"][0])
    common = dict(image_id=image_id, model_id=MODEL_ID, chunk_size=64, image_input_sha256=diag["image_input_sha256"],
                  extra_metadata={"dataset": "MT-GQA-reconstructed", "source_dialog_id": source["dialog_id"], "source_turn_id": 1,
                                  "source_method_id": source["method_id"], "source_execution_id": source["physical_execution_id"],
                                  "source_logical_request_id": source["logical_request_id"], "run_schema": SCHEMA})
    if kind == "raster":
        with WriteTimer() as timer:
            receipt = persist_captured_raster_prefix(*args, diag["hidden"].result_cpu(), path, probe_heads=SERVER_ARGS["probe_heads"],
                                                     hidden_capture_stats=diag["hidden"], full_integrity_hash=False, **common)
        receipt["write_by_file_name"] = timer.by_name
        return receipt
    # the two packed kinds differ in this one argument; the importance order is the writer's default and is not passed
    order = {} if PACKED_KINDS[kind] == ROW_ORDER_IMPORTANCE else {"row_order": PACKED_KINDS[kind]}
    return persist_captured_packed_prefix(*args, diag["vision"].result_cpu(), path, layer_budget=docs[VECTOR],
                                          capture_stats=diag["vision"], full_integrity_hash=False, **order, **common)


def persist(runner, result, diag, path, image_id, kind, source, docs):
    return support.persist_llava(lambda: _native_persist(runner, result, diag, path, image_id, kind, source, docs), path, kind, source)


def activate(runner, kind, path):
    started = time.perf_counter()
    if kind == "raster":
        from mmimpress.serve import ImageContext
        ctx = ImageContext(path, runner.model.device, drop_cache=False, require_v_hidden=False)
        completion = support.verify_completion(path, "integrity.json")
        pinned = 0
    else:
        ctx = PackedImageContext(path, runner.model.device)
        ctx.validate_packed_layout()
        assert ctx.row_order == PACKED_KINDS[kind]
        completion = ctx.integrity_activation
        pinned = int(ctx.reader.total)
    ctx.activation = {"kind": kind, "host_tensor_bytes": sum(t.numel() * t.element_size() for t in ctx.cache.sys_kv.values()),
                      "pinned_host_buffer_bytes": pinned, "gpu_tensor_bytes": 0, "visual_payload_resident_bytes": 0,
                      "file_bytes": sum((path / n).stat().st_size for n in ("meta.json", "sys_kv.pt")),
                      "completion_integrity": completion, "event_id": str(uuid.uuid4()), "event_kind": "activation"}
    ctx.activation["outer_wall_ms"] = (time.perf_counter() - started) * 1000
    return ctx


def store_equivalence(reorder_path, original_path):
    """Gate 1 for one image: the original-order store against the importance
    store, read back from disk.

    * layer l's original indices == sorted(order[:k_l]) of the importance store;
    * every stored token's K and V rows are bit-identical in the two files;
    * the separator blocks are bit-identical;
    * the byte tables and the payload sizes are equal;
    * the system-prompt KV tensors are equal;
    * the original-order metadata carries no importance permutation.
    """
    reorder_path, original_path = Path(reorder_path), Path(original_path)
    mr = json.loads((reorder_path / "meta.json").read_text()); mo = json.loads((original_path / "meta.json").read_text())
    counts = [int(c) for c in mr["layer_keep_counts"]]
    row_bytes = int(mr["packed_row_bytes"])
    rows_o = layer_original_indices(mo)
    order = [int(x) for x in mr["order"]]
    sets_equal = [rows_o[li] == sorted(order[:counts[li]]) for li in range(len(counts))]
    table_equal = (mo["layer_keep_counts"] == mr["layer_keep_counts"] and mo["packed_layout"] == mr["packed_layout"]
                   and mo["packed_sep_offset"] == mr["packed_sep_offset"] and mo["packed_sep_bytes"] == mr["packed_sep_bytes"]
                   and mo["packed_row_bytes"] == mr["packed_row_bytes"] and mo["newline_idx"] == mr["newline_idx"]
                   and mo["v_token_num"] == mr["v_token_num"] and mo["v_token_start"] == mr["v_token_start"])
    br = np.fromfile(reorder_path / PACKED_NAME, dtype=np.uint8); bo = np.fromfile(original_path / PACKED_NAME, dtype=np.uint8)
    size_equal = int(br.size) == int(bo.size) == int(mr["packed_total_bytes"]) == int(mo["packed_total_bytes"])
    rows_checked = rows_different = 0
    if table_equal and size_equal and all(sets_equal):
        for li, k in enumerate(counts):
            if not k:
                continue
            to_ascending = np.argsort(np.asarray(order[:k]), kind="stable")      # importance row of each ascending original index
            for kind in ("k", "v"):
                off = int(mr["packed_layout"][li][f"{kind}_offset"])
                a = br[off:off + k * row_bytes].reshape(k, row_bytes)[to_ascending]
                b = bo[off:off + k * row_bytes].reshape(k, row_bytes)
                rows_checked += k; rows_different += int((a != b).any(axis=1).sum())
    sep_off = int(mr["packed_sep_offset"])
    sep_equal = size_equal and bool(np.array_equal(br[sep_off:], bo[sep_off:]))
    del br, bo
    sr = torch.load(reorder_path / "sys_kv.pt", weights_only=True); so = torch.load(original_path / "sys_kv.pt", weights_only=True)
    sys_equal = all(torch.equal(sr[n], so[n]) for n in ("k", "v"))
    no_permutation = (not any(key in mo for key in IMPORTANCE_ORDER_KEYS) and not (original_path / "visionzip_layout.pt").exists()
                      and mo.get("row_order") == ROW_ORDER_ORIGINAL)
    ok = (all(sets_equal) and table_equal and size_equal and rows_checked == 2 * sum(counts) and rows_different == 0
          and sep_equal and sys_equal and no_permutation)
    return {"status": "PASS" if ok else "FAIL", "layers": len(counts), "layer_sets_equal": int(sum(sets_equal)),
            "byte_table_equal": bool(table_equal), "payload_bytes_equal": bool(size_equal), "payload_bytes": int(mr["packed_total_bytes"]),
            "kv_rows_compared": int(rows_checked), "kv_rows_different": int(rows_different), "separator_block_equal": bool(sep_equal),
            "separator_bytes": int(mr["packed_sep_bytes"]), "system_kv_equal": bool(sys_equal),
            "original_metadata_has_no_importance_permutation": bool(no_permutation),
            "reorder_packed_sha256": mr["packed_sha256"], "original_packed_sha256": mo["packed_sha256"],
            "metadata_bytes": {"reorder": {"meta.json": (reorder_path / "meta.json").stat().st_size,
                                           "visionzip_layout.pt": (reorder_path / "visionzip_layout.pt").stat().st_size},
                               "original": {"meta.json": (original_path / "meta.json").stat().st_size}}}


def make_row(runner, result, diag, meta, *, methods, method, dialog, turn, phase, run, attempt, physical_id, ordinal, store_info, config,
             preceding):
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
            "store_kind": arm["kind"], "row_order": PACKED_KINDS.get(arm["kind"]), "image_id": dialog["image_id"],
            "dialog_id": dialog["dialog_id"], "turn_id": turn, "question_id": str(question["question_id"]),
            "question": question["question"], "gold": question["answers"], "prediction": result["answer"],
            "score": strict_score(result["answer"], question["answers"][0]), "prompt": diag["prompt"],
            "history": diag["history"], "history_entries": diag["history_entries"],
            "prompt_sha256": hashlib.sha256(diag["prompt"].encode()).hexdigest(), "suffix_input_ids": suffix,
            "suffix_ids_sha256": canonical_hash(suffix), "history_sha256": hashlib.sha256(diag["history"].encode()).hexdigest(),
            "config_sha256": config["_file_sha256"], "manifest_sha256": config["_manifest_sha256"],
            "code_sha256": canonical_hash(config["source_sha256"]), "global_dialog_ordinal": ordinal,
            "method_order": order, "method_order_position": order.index(method), "preceding_method_id": preceding,
            "request_started_at_s": result["request_started_at_s"], "first_token_at_s": result["first_token_at_s"],
            "request_finished_at_s": result["request_started_at_s"] + result["request_e2e_ms"] / 1000,
            "ttft_ms": result["end_to_end_ttft_ms"], "request_e2e_ms": result["request_e2e_ms"],
            "ssd_read_bytes": int(io.get("bytes", 0)), "ssd_preads": int(io.get("preads", 0)),
            "pread_calls": int(result.get("pread_calls", 0)), "preadv_calls": int(result.get("preadv_calls", 0)),
            "k_read_bytes": int(per_kind.get("k", {}).get("bytes", 0)),
            "v_read_bytes": int(per_kind.get("v", {}).get("bytes", 0)), "packed_read_bytes": int(per_kind.get("packed", {}).get("bytes", 0)),
            "read_blocked_ms": wait.get("read_blocked_ms") if hit else None,
            "N_content": n, "geometry": g, "source_store": store_info if hit else None,
            "provisioning_mode": "FRESH_IMAGE_STREAMING", "result": safe(value)}


# ------------------------------------------------------------------- audit
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
    kinds_here = store_kinds(methods)
    if artifact is not None:
        artifact = Path(artifact)
        builds = []
        for kind in kinds_here:
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
            if kind in PACKED_KINDS:
                counts = doc_counts(doc, n)
                c.require(counts == independent_counts(doc["layer_keep_ratios"], n, doc["r_max"]) and min(counts) >= 1, "PK_counts_independent_nonzero", kind)
                c.require(meta["layer_budget_r_min"] == doc["r_min"] and meta["layer_budget_r_max"] == doc["r_max"]
                          and meta["layer_keep_ratios"] == doc["layer_keep_ratios"], "PK_clip_range_and_ratios_from_document", kind)
                c.require(meta["storage_policy"] == "prefix25_packed" and meta["store_format"] == STORE_FORMATS[kind] and meta["budget_policy"] == "layerwise_v1", "PK_policy", kind)
                c.require(meta["layer_keep_counts"] == counts == meta["payload_rows_per_layer"], "PK_counts_recomputed", kind)
                c.require(sum(counts) == 32 * ((n + 3) // 4), "PK_budget_equals_uniform_total", kind)
                c.require(meta["layer_budget_sha256"] == doc["layer_budget_sha256"] == VECTOR_RATIO_SHA and meta["layer_budget_id"] == VECTOR, "PK_vector_identity", kind)
                sep = 2 * nl * len(meta["newline_idx"]) * width
                total = sum(2 * kk * width for kk in counts) + sep
                c.require(meta["packed_total_bytes"] == total == files[PACKED_NAME]["size"] and meta["packed_sep_bytes"] == sep, "PK_exact_file_length", kind)
                c.require(files[PACKED_NAME]["sha256"] == meta["packed_sha256"], "PK_packed_sha256_matches_inventory", kind)
                c.require(not any(f.startswith("layer_") or f == "sep_kv.bin" for f in files), "PK_no_unpacked_payload_files", kind)
                c.require(meta.get("layout_uses_dataset_question") is False and meta.get("llm_used_for_layout_scoring") is False
                          and meta.get("calibration_questions") == 0, "PK_layout_takes_no_question_llm_or_calibration", kind)
                if kind == ORIG_KIND:
                    structural = set(meta["newline_idx"])
                    idx = meta.get("stored_original_indices", []); thr = meta.get("stored_min_keep_count", [])
                    per_layer = [[i for i, t in zip(idx, thr) if t <= kk] for kk in counts]
                    c.require(meta.get("row_order") == ROW_ORDER_ORIGINAL and meta.get("reordered") is False
                              and not any(key in meta for key in IMPORTANCE_ORDER_KEYS) and "visionzip_layout.pt" not in files,
                              "ORIG_no_importance_permutation_in_store", kind)
                    c.require(len(idx) == len(thr) == max(counts) and idx == sorted(set(idx)) and not structural & set(idx)
                              and all(0 <= i < vn for i in idx) and [len(x) for x in per_layer] == counts, "ORIG_index_table_valid", kind)
                else:
                    c.require("row_order" not in meta and meta.get("reordered") is True and "visionzip_layout.pt" in files
                              and len(meta.get("order", [])) == vn, "PK_importance_layout_unchanged", kind)
            else:
                c.require(meta.get("storage_policy", "full") == "full" and meta["probe_heads"] == SERVER_ARGS["probe_heads"] and meta["chunk_size"] == 64, "full_store_with_probe_sidecar", kind)
                for li in range(nl):
                    ok = (files[f"layer_{li:02d}/k.bin"]["size"] == files[f"layer_{li:02d}/v.bin"]["size"] == vn * width
                          and files[f"layer_{li:02d}/probe_k.bin"]["size"] == vn * SERVER_ARGS["probe_heads"] * meta["head_dim"] * 2)
                    c.require(ok, "full_kv_and_probe_file_lengths", f"{kind}/layer{li}")
                c.require(not meta.get("reordered", False) and "order" not in meta, "raster_not_reordered", kind)
        c.require(len(set(builds)) == len(kinds_here), "distinct_physical_build_events", image_id)
        if PACKED_KIND in kinds_here and ORIG_KIND in kinds_here:      # gate 1, per image
            eq_path = artifact / "store_equivalence.json"
            eq = json.loads(eq_path.read_text()) if eq_path.exists() else {}
            c.require(eq.get("status") == "PASS", "two_packed_stores_hold_the_same_tokens_bit_for_bit", image_id)
            c.require(eq.get("reorder_packed_sha256") == metas[PACKED_KIND]["packed_sha256"]
                      and eq.get("original_packed_sha256") == metas[ORIG_KIND]["packed_sha256"], "store_equivalence_is_about_these_stores", image_id)
            order = metas[PACKED_KIND]["order"]; cts = metas[PACKED_KIND]["layer_keep_counts"]
            idx = metas[ORIG_KIND]["stored_original_indices"]; thr = metas[ORIG_KIND]["stored_min_keep_count"]
            c.require(all([i for i, t in zip(idx, thr) if t <= kk] == sorted(order[:kk]) for kk in cts), "original_index_sets_equal_importance_prefixes", image_id)
            c.require(metas[ORIG_KIND]["packed_total_bytes"] == metas[PACKED_KIND]["packed_total_bytes"], "two_packed_payloads_same_size", image_id)
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
        for li in range(g["num_layers"]):
            observed = calls.get(str(li), {})
            c.require(observed.get("prefill") == {"q": 1, "k": 1, "v": 1}, "prefill_projection_counts_each_layer", rid)
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
            c.require(arm["async"] and not any(main_flags), "fullload_async_reads_on_worker", rid)
            c.require(result.get("prefetch") == "layer_stream" and ahead is not None and ahead["items_read"] == ahead["items_taken"] == nl
                      and ahead["max_ahead"] == READAHEAD["max_ahead"], "fullload_async_layer_stream", rid)
        else:      # both packed arms: identical read contract
            row_order = PACKED_KINDS[arm["kind"]]
            counts = doc_counts(doc, n)
            c.require(counts == independent_counts(doc["layer_keep_ratios"], n, doc["r_max"]), "PK_k_per_layer_independent", rid)
            c.require(result.get("k_per_layer") == counts, "PK_k_per_layer_recomputed", rid)
            c.require(result.get("attended_content_kv_count_per_layer") == counts and result.get("k_total") == sum(counts) == nl * ((n + 3) // 4), "PK_budget_total", rid)
            c.require(result.get("layer_budget_sha256") == doc["layer_budget_sha256"] and result.get("layer_budget_id") == VECTOR, "PK_vector_identity_hit", rid)
            c.require(result.get("row_order") == row_order == r["row_order"], "PK_row_order_of_the_arm", rid)
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
            if row_order == ROW_ORDER_ORIGINAL:
                c.require(result.get("selected_stored_ids") is None and original == sorted(original), "ORIG_rows_addressed_by_original_index", rid)
            else:
                c.require(result.get("selected_stored_ids") == list(range(max(counts))), "PK_prefix_contract", rid)
            c.require(result.get("keep_count_per_layer") == [kk + len(structural) for kk in counts], "structural_keep", rid)
            c.require(result.get("store_format") == STORE_FORMATS[arm["kind"]] and result.get("directcopy") is False, "PK_scatter_path", rid)
            packed_sets.setdefault(m, []).append((original, counts, total))
            c.require(result.get("query_score_calls", 0) == 0, "online_scoring_zero", rid)
    for m, sets in packed_sets.items():
        c.require(all(s == sets[0] for s in sets), "same_image_selection_invariant", f"{image_id}/{m}")
        if metas:
            meta = metas[kind_of(m)]
            ref = meta["stored_original_indices"] if kind_of(m) == ORIG_KIND else meta["order"][:len(sets[0][0])]
            c.require(sets[0][0] == ref and sets[0][2] == meta["packed_total_bytes"], "meta_to_actual_selection_and_bytes", f"{image_id}/{m}")
    if OURS_ASYNC in packed_sets and ORIG_ASYNC in packed_sets:      # criterion c, per image: same tokens visible, same bytes
        a, b = packed_sets[OURS_ASYNC][0], packed_sets[ORIG_ASYNC][0]
        c.require(sorted(a[0]) == b[0] and a[1] == b[1] and a[2] == b[2], "two_packed_arms_same_tokens_counts_and_bytes", image_id)
    # recorded, not a commit condition
    t1 = {}
    for (did, m, t), r in bykey.items():
        if t == 1:
            t1.setdefault(did, set()).add(tuple(r["result"]["generated_token_ids"]))
    return dict(c.result(), image_id=image_id, phase=phase, requests=len(rows), expected_requests=len(expected), dialogs=len(ds),
                t1_tokens_identical_across_arms=all(len(v) == 1 for v in t1.values()))


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
    rows = []; pending = []; success = False; capturing = set(); peak_scratch = 0; preceding = None
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
                    before = preceding; preceding = method
                    # the first arm of a store kind in the first dialogue's rotation builds that kind's store
                    capture = kind if di == 0 and turn == 1 and kind != "none" and kind not in capturing else "none"
                    if capture != "none":
                        capturing.add(capture)
                    if turn == 1 or method == "recompute":
                        result, diag = normal_request(runner, servers["base"], ROOT / dialog["image_path"], factory,
                                                      "raster" if capture == "raster" else ("image_only" if capture != "none" else "none"))
                        append(artifact / "execution_events.jsonl", dict(source, turn_id=turn, status="EXECUTED", prediction=result["answer"],
                                                                        generated_token_ids=result["generated_token_ids"], ttft_ms=result["end_to_end_ttft_ms"],
                                                                        request_e2e_ms=result["request_e2e_ms"]))
                        if capture != "none":
                            start = time.perf_counter()
                            p = persist(runner, result, diag, paths[capture], image_id, capture, source, docs)
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
                            pending.append((method, result, saved, physical, before))
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
                                   attempt=attempt, physical_id=physical, ordinal=ordinal, store_info=store_info.get(kind), config=config, preceding=before)
                    row["image_sha256"] = source_image_sha
                    append(artifact / "raw.jsonl", row); rows.append(row); generated[method][turn] = row
                    append(artifact / "attempts.jsonl", dict(source, turn_id=turn, status="COMPLETED", at_unix=time.time()))
                    del result, diag
                if di == 0 and turn == 1:
                    assert set(metas) == set(all_kinds)
                    for method, result, diag, physical, before in pending:
                        meta_kind = kinds.get(method, ref_kind)
                        row = make_row(runner, result, diag, metas[meta_kind], methods=methods, method=method, dialog=dialog, turn=turn, phase=phase, run=run,
                                       attempt=attempt, physical_id=physical, ordinal=ordinal, store_info=None, config=config, preceding=before)
                        row["image_sha256"] = source_image_sha
                        append(artifact / "raw.jsonl", row); rows.append(row); generated[method][1] = row
                        append(artifact / "attempts.jsonl", {"physical_execution_id": physical, "logical_request_id": row["logical_request_id"], "status": "COMPLETED", "at_unix": time.time()})
                    pending.clear(); del result, diag
                    for kind in metas:
                        assert metas[kind]["prefix_input_ids"] == metas[ref_kind]["prefix_input_ids"]
                        assert metas[kind]["image_input_sha256"] == metas[ref_kind]["image_input_sha256"]
        for kind in all_kinds:
            assert file_hashes(paths[kind]) == hashes[kind], "store mutated during requests"
        if PACKED_KIND in paths and ORIG_KIND in paths:      # gate 1: after every request of the image, outside every timer
            t0 = time.perf_counter()
            eq = store_equivalence(paths[PACKED_KIND], paths[ORIG_KIND])
            eq["compare_ms"] = (time.perf_counter() - t0) * 1e3
            atomic_json(artifact / "store_equivalence.json", eq)
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
    assert run.name.startswith(RUN_PREFIX) and CONTRACT.is_file()
    ds = frozen_dialogues()
    prior_manifest = json.loads((PRIOR / "manifest.json").read_text())
    assert prior_manifest["dialogues"] == ds
    for image, identity in prior_manifest["image_identity"].items():
        assert sha(ROOT / identity["image_path"]) == identity["sha256"]
    hold_manifest = json.loads((PYRBETA_RUN / "hold" / "manifest.json").read_text())
    assert hold_manifest["dialogues"] == ds and hold_manifest["image_identity"] == prior_manifest["image_identity"]
    assert hold_manifest["index_sha256"] == INDEX_SHA and hold_manifest["workload_sha256"] == WORKLOAD_SHA
    previous = json.loads((PRIOR_MAIN_RUN / "manifest.json").read_text())      # the 2026-11-17 main run: same dialogues, same images
    assert previous["dialogues"] == ds and previous["image_identity"] == prior_manifest["image_identity"]
    previous_config = json.loads((PRIOR_MAIN_RUN / "config.json").read_text())
    assert (PRIOR_MAIN_RUN / "main_execution_complete.json").exists()
    assert all(m in previous_config["phases"]["main"]["methods"] for m in ("recompute", "fullload_async", OURS_ASYNC, PRIOR_IMPRESS))
    smoke_ids = hold_manifest["smoke_dialogue_ids"]
    smoke_images = [d["image_id"] for d in ds if d["dialog_id"] in set(smoke_ids)]
    dev_images = json.loads(V2_SPLIT.read_text())["dev"]["image_ids"]
    all_images = {d["image_id"] for d in ds}
    assert len(dev_images) == 40 and set(dev_images) <= all_images and len(set(smoke_images)) == 4
    assert previous["phase_images"]["gate"] == smoke_images and previous["phase_images"]["dev"] == dev_images
    image_sets = {"gate": set(smoke_images), "dev": set(dev_images), "main": all_images}
    phase_ids = {ph: [d["dialog_id"] for d in ds if d["image_id"] in image_sets[PHASE_IMAGES[ph]]] for ph in PHASES}
    manifest = {"dataset": "MT-GQA-reconstructed", "schema_version": SCHEMA, "index_sha256": INDEX_SHA, "workload_sha256": WORKLOAD_SHA,
                "dialogues": ds, "image_identity": prior_manifest["image_identity"], "phase_dialogue_ids": phase_ids,
                "phase_images": {"gate": smoke_images, "devgate": dev_images, "dev": dev_images, "main": sorted(all_images)},
                "phase_methods": PHASE_METHODS,
                "expected_requests": {p: len(phase_ids[p]) * 3 * len(PHASE_METHODS[p]) for p in PHASES},
                "prior_manifest_sha256": sha(PRIOR / "manifest.json"), "pyrbeta_hold_manifest_sha256": sha(PYRBETA_RUN / "hold" / "manifest.json"),
                "previous_main_manifest_sha256": sha(PRIOR_MAIN_RUN / "manifest.json"),
                "gate_policy": "every dialogue of the four images of the HOLD smoke dialogue ids (the gate images of the 2026-11-17 run)",
                "dev_policy": "every MT-GQA dialogue of the 40 DEV images of results/layer_budget_kv25_v2_20261106T073502Z/data_split.json"}
    assert manifest["expected_requests"] == {"gate": 49 * 3 * 6, "devgate": 378 * 3 * 3, "dev": 378 * 3 * 4, "main": 12183 * 4}
    support.publish(run / "manifest.json", manifest)
    free = shutil.disk_usage(run).free
    # per image: one full store (FullLoad: K/V + probe sidecar + separator sidecar + hidden) and two packed stores, at the
    # largest geometry (N = 2880); the question-independence check holds six packed stores of one image at a time
    full_store = 2 * 32 * 2928 * 8192 + 32 * 2928 * 3 * 128 * 2 + 2 * 32 * 48 * 8192 + (64 << 20)
    packed_store = 411_041_792
    bundle = full_store + 6 * packed_store
    plan = {"policy": "FRESH_IMAGE_STREAMING", "minimum_reserve_bytes": 30 << 30, "peak_working_set_bytes": bundle + (4 << 30),
            "largest_image_bundle_bound_bytes": bundle, "maximum_new_payload_bytes": 20 << 30, "raw_and_evidence_growth_bound_bytes": 8 << 30,
            "free_bytes_at_freeze": free, "full_store_bound_bytes": full_store, "packed_store_bound_bytes": packed_store,
            "deletion_scope": "runs/<RUN>/scratch only (exact SHA allowlist after audit+commit; the question-independence stores "
                              "after their comparison); existing stores never deleted"}
    # expected SSD writes, from the store sizes the 2026-11-17 main run measured on the same images
    raster_b, packed_b = {}, {}
    for image in all_images:
        doc_ = json.loads((PRIOR_MAIN_RUN / "main" / "images" / image / "COMMITTED.json").read_text())
        usage = json.loads((PRIOR_MAIN_RUN / "main" / "images" / image / doc_["attempt_id"] / "disk_usage.json").read_text())["store_bytes"]
        raster_b[image] = usage["raster"]; packed_b[image] = usage[PACKED_KIND]
    both = lambda imgs: sum(raster_b[i] + 2 * packed_b[i] for i in imgs)
    plan["expected_ssd_write_bytes"] = {
        "gate": both(set(smoke_images)), "devgate": sum(packed_b[i] for i in dev_images), "dev": both(dev_images), "main": both(all_images),
        "question_independence": 6 * sum(packed_b[i] for i in set(smoke_images) | set(dev_images))}
    plan["expected_ssd_write_bytes"]["total"] = sum(plan["expected_ssd_write_bytes"].values())
    plan["expected_ssd_write_note"] = ("per image: raster store + two packed stores (gate, dev, main); one packed store (devgate); six packed "
                                       "stores per gate/DEV image for the question-independence check (three questions x two row orders); sizes from "
                                       "runs/mt_gqa_impress_code_20261117T185832Z/main disk_usage.json; raw rows and evidence not included")
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
    assert previous_config["layer_budgets"][VECTOR]["layer_budget_sha256"] == VECTOR_RATIO_SHA
    budgets = {VECTOR: {"path": str(p), "layer_budget_sha256": doc["layer_budget_sha256"], "layer_keep_ratios": doc["layer_keep_ratios"],
                        "r_min": doc["r_min"], "r_max": doc["r_max"], "alloc_id": VECTOR, "policy": doc["policy"], "schema": doc["schema"],
                        "num_layers": doc["num_layers"], "total_ratio": doc["total_ratio"], "source": doc["source"]}}
    # the three files this work edited, against the 2026-11-17 freeze (every other frozen source file must be unchanged)
    now = source_hashes(); before = previous_config["source_sha256"]
    changed = sorted(k for k in before if k in now and now[k] != before[k])
    added = sorted(k for k in now if k not in before)
    config = {"schema_version": SCHEMA, "model": MODEL_ID, "revision": REVISION, "seed": 1234, "greedy": True, "max_new_tokens": 16,
              "batch_size": 1, "cpu_threads": 2, "history_policy": "method_local_generated",
              "phases": {ph: {"methods": PHASE_METHODS[ph], "dialogues": len(phase_ids[ph])} for ph in PHASES},
              "arms": {m: dict(a) for m, a in ARMS.items()}, "server_args": SERVER_ARGS, "readahead": READAHEAD,
              "store_kinds": {"raster": "full visual KV, raster order", PACKED_KIND: "packed, row_order=importance (writer default)",
                              ORIG_KIND: "packed, row_order=original"},
              "arm_order": "cyclic rotation by dialogue ordinal; the two packed arms exchange slots when (ordinal // n_arms) is odd",
              "packed_storage_policy": "prefix25_packed", "store_formats": STORE_FORMATS, "layer_budgets": budgets,
              "source_sha256": now, "source_changed_since_20261117_freeze": changed, "source_added_since_20261117_freeze": added,
              "contract_sha256": sha(CONTRACT),
              "manifest_sha256": sha(run / "manifest.json"), "storage_plan_sha256": sha(run / "storage_plan.json"),
              "source_T1_capture": "first frozen dialogue per image; per store kind, the first arm of that kind in the dialogue's order; every arm still runs its own full-pixel T1",
              "reference_runs": {"previous_main": str(PRIOR_MAIN_RUN.relative_to(ROOT)), "previous_impress_method": PRIOR_IMPRESS},
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
                      "free_gib": free / 2**30, "transformers": transformers.__version__, "source_changed": changed, "source_added": added}), flush=True)


def load_model(config):
    torch.set_num_threads(config["cpu_threads"]); torch.manual_seed(1234); np.random.seed(1234)
    torch.set_grad_enabled(False)
    assert_gpu_exclusive(); runner = V.load_runner()
    from mmimpress.serve import Server
    servers = {"base": Server(runner, ratio=SERVER_ARGS["ratio"], probe=SERVER_ARGS["probe_heads"], alpha=SERVER_ARGS["alpha"], max_new_tokens=16)}
    QA._warmup(runner, servers["base"])
    servers["base"]._prefetch_pool()      # the single read-ahead worker thread exists before any request
    return runner, servers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True, help="runs/mt_gqa_ours_order_<UTC>")
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
    runner, servers = load_model(config)
    try:
        atomic_json(run / f"{phase}_start_{time.strftime('%Y%m%dT%H%M%S')}.json",
                    {"started_unix": time.time(), "capacity": capacity_guard(run), "pid": os.getpid()})
        if (run / phase).exists() and not args.resume:
            raise FileExistsError("use --resume; previous artifacts preserved")
        receipt = run_phase(runner, servers, run, phase, config, manifest)
        atomic_json(run / f"{phase}_execution_complete.json", {"status": "PASS", **receipt, "finished_unix": time.time()}, replace=True)
    except BaseException as exc:
        error = {"status": "PARTIAL", "error": repr(exc), "traceback": traceback.format_exc(),
                 "resume_command": f"{sys.executable} scripts/202_eval_mt_gqa_ours_order.py --run {run} --stage {phase} --resume"}
        atomic_json(run / ("interruption_" + phase + "_" + time.strftime("%Y%m%dT%H%M%S") + ".json"), error)
        print(json.dumps({k: v for k, v in error.items() if k != "traceback"}), flush=True)
        raise
    finally:
        for server in servers.values():
            server.close()


if __name__ == "__main__":
    main()

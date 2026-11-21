#!/usr/bin/env python3
"""LLaVA-1.5-7B x MT-GQA main experiment: ReComp / FullLoad / IMPRESS / Ours
(contract docs/llava15_mt_gqa_main_contract.md).

Where the code comes from:

* ReComp, FullLoad and IMPRESS are the arms of the 2026-11-18 run
  (scripts/199_eval_llava15_mtvqa_impress_code.py).  That module is imported
  and its request, capture, layout and tracing functions are called as they
  are (``normal_request``, ``build_official_layout``, ``ThreadedReadTrace``,
  ``WriteTimer``, ``LogitTies``, ``load_servers`` ...); the FullLoad and
  IMPRESS request calls below are the calls of that script.
* Ours is the original-row-order packed store (``row_order="original"`` of
  ``mmimpress.llava15.store`` / ``serve``): image-only CLIP CLS->patch
  importance, the PYR-b20 layer vector, layer l = the top k_l tokens stored in
  ascending original token index and loaded to their original cache positions.
* Data, prompt and scoring: docs/llava15_mt_gqa_contract.md (MT-GQA
  reconstructed, 398 images / 4,061 dialogues x 3 turns, method-local
  generated history, strict normalized EM).

Phases
  gate     4 images, 10 arms   (gates 3, 4, 5, 6 and the store comparison of gate 1)
  devgate  DEV 40,   7 arms    (gates 3, 4, 5 on the DEV images)
  dev      DEV 40,   5 arms    (the pilot of gate 8: the four main arms + the
                                reordered packed arm as a reference; gates 1, 6)
  main     398 images, 4 arms

Resume (``--resume``): committed images are skipped.  Inside the image that
was interrupted, complete dialogues are kept, the interrupted dialogue restarts
at its turn 1, and its partial rows are dropped; if the image's stores cannot
be verified against the hashes recorded when they were built (or the first
dialogue itself was interrupted), rows and stores of that image are discarded
and the image starts again.

Stages: --stage freeze | gate | devgate | dev | main [--resume]
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import llava15_support as S  # noqa: E402
from llava15_support import MODEL_ID, REVISION, SHORT_ANSWER  # noqa: E402
from mmimpress import impress_official  # noqa: E402
from mmimpress.llava15 import N_VISUAL, NUM_LAYERS  # noqa: E402
from mmimpress.llava15.raster import persist_captured_llava15_raster_prefix, validate_llava15_raster_meta  # noqa: E402
from mmimpress.llava15.store import (ORIGINAL_STORE_FORMAT, STORE_FORMAT, keep_counts, load_budget,  # noqa: E402
                                     persist_captured_llava15_prefix)
from mmimpress.mtvqa25k.adapter import archive_before_commit, read_rows, remove_archived_duplicates  # noqa: E402
from mmimpress.packed_store import (IMPORTANCE_ORDER_KEYS, PACKED_NAME, ROW_ORDER_IMPORTANCE, ROW_ORDER_ORIGINAL,  # noqa: E402
                                    layer_original_indices)

E = S.load_script("_l15_mtvqa_impress_code_eval", "199_eval_llava15_mtvqa_impress_code.py")      # the 2026-11-18 harness, used as is
import llava15_main_status as STATUS  # noqa: E402

SCHEMA = "llava15-mtgqa-main-v1"
DATASET = "MT-GQA-reconstructed"
RUN_PREFIX = "llava15_mt_gqa_main_"
CONTRACT = ROOT / "docs/llava15_mt_gqa_main_contract.md"
RECOMP_REF_RUN = ROOT / "runs/llava15_mt_gqa_20261112T062708Z"                    # gate 6 reference (ReComp tokens) and the vector
IMPL_REF_RUN = ROOT / "runs/llava15_mtvqa_impress_code_20261118T053221Z"          # FullLoad / IMPRESS implementation reference
V2_SPLIT = ROOT / "results/layer_budget_kv25_v2_20261106T073502Z/data_split.json"
VECTOR, VECTOR_SOURCE = E.VECTOR, E.VECTOR_SOURCE
VECTOR_FILE_SHA, VECTOR_RATIO_SHA, VECTOR_FRACTION, PACKED_BYTES = E.VECTOR_FILE_SHA, E.VECTOR_RATIO_SHA, E.VECTOR_FRACTION, E.PACKED_BYTES
SERVER_ARGS, OFFICIAL = E.SERVER_ARGS, E.OFFICIAL                                  # the settings of the 2026-11-18 run
NEW_KIND, NEW, NEW_ASYNC, NEW_R100, FULL_REORDERED = E.NEW_KIND, E.NEW, E.NEW_ASYNC, E.NEW_R100, E.FULL_REORDERED
FULL_KINDS = E.FULL_KINDS
REORDER_ASYNC = E.OURS_ASYNC                                                       # the reordered packed arm of the 2026-11-18 run (DEV reference only)
OURS = f"llava15_kv_{VECTOR}_origorder_packed"                                     # Ours, synchronous read (gate only)
OURS_ASYNC = OURS + "_async"                                                       # Ours
OURS_REPEAT = OURS + "_async_repeat"                                               # Ours executed a second time (gate only)
PACKED_KIND = E.PACKED_KIND                                                        # packed, importance row order (existing default)
ORIG_KIND = f"packedorig_{VECTOR}"                                                 # packed, original row order
PACKED_KINDS = {PACKED_KIND: ROW_ORDER_IMPORTANCE, ORIG_KIND: ROW_ORDER_ORIGINAL}
STORE_FORMATS = {PACKED_KIND: STORE_FORMAT, ORIG_KIND: ORIGINAL_STORE_FORMAT}
READAHEAD = {"fullload_async": "layer_stream", NEW_ASYNC: "next_layer", OURS_ASYNC: "layer_stream", OURS_REPEAT: "layer_stream",
             REORDER_ASYNC: "layer_stream", "max_ahead": None}
GiB = 1 << 30

ARMS = OrderedDict([
    ("recompute",      {"family": "recompute", "kind": "none", "async": False, "display": "ReComp"}),
    ("fullload_async", {"family": "fullload", "kind": "raster", "async": True, "display": "FullLoad"}),
    (NEW,              {"family": "impress_official", "kind": NEW_KIND, "async": False, "display": "IMPRESS, 동기 읽기", "ratio": 0.25}),
    (NEW_ASYNC,        {"family": "impress_official", "kind": NEW_KIND, "async": True, "display": "IMPRESS", "ratio": 0.25}),
    # gate only: the IMPRESS path at 100 % retention, and the FullLoad loader (no selection, no mask) on the SAME reordered store
    (NEW_R100,         {"family": "impress_official", "kind": NEW_KIND, "async": False, "display": "IMPRESS (유지 100%)", "ratio": 1.0}),
    (FULL_REORDERED,   {"family": "fullload", "kind": NEW_KIND, "async": False, "display": "FullLoad loader (재배치 store)"}),
    (OURS,             {"family": "ours", "kind": ORIG_KIND, "async": False, "display": "Ours, 동기 읽기"}),
    (OURS_ASYNC,       {"family": "ours", "kind": ORIG_KIND, "async": True, "display": "Ours"}),
    (OURS_REPEAT,      {"family": "ours", "kind": ORIG_KIND, "async": True, "display": "Ours, 두 번째 실행"}),
    (REORDER_ASYNC,    {"family": "ours", "kind": PACKED_KIND, "async": True, "display": "Ours(재배열)"}),
])
MAIN_METHODS = ["recompute", "fullload_async", NEW_ASYNC, OURS_ASYNC]
PHASE_METHODS = {
    "gate": ["recompute", "fullload_async", NEW, NEW_ASYNC, NEW_R100, FULL_REORDERED, OURS, OURS_ASYNC, OURS_REPEAT, REORDER_ASYNC],
    "devgate": [NEW, NEW_ASYNC, NEW_R100, FULL_REORDERED, OURS, OURS_ASYNC, OURS_REPEAT],
    "dev": MAIN_METHODS + [REORDER_ASYNC],
    "main": list(MAIN_METHODS),
}
PHASES = ("gate", "devgate", "dev", "main")
PHASE_IMAGES = {"gate": "gate", "devgate": "dev", "dev": "dev", "main": "main"}
EXPECTED_REQUESTS = {"gate": 49 * 3 * 10, "devgate": 378 * 3 * 7, "dev": 378 * 3 * 5, "main": 12183 * 4}
# report scripts read committed rows only and are not part of the measurement freeze
REPORT_ONLY = E.REPORT_ONLY + ("204_report_mt_gqa_ours_order.py", "205_report_mt_gqa_ours_swap.py",
                              "208_report_llava15_mt_gqa_main.py")
# source files that may differ from the 2026-11-18 freeze: this work's two edits and the LLaVA-NeXT row-order option of 2026-11-20
CHANGED_SINCE_IMPL_REF = ("mmimpress/llava15/serve.py", "mmimpress/llava15/store.py",
                          "mmimpress/packed_serve.py", "mmimpress/packed_store.py", "mmimpress/piggyback.py")

H1, QA = E.H1, E.QA
import main_b_support as support  # noqa: E402

sha, canonical_hash, safe, atomic_json, append, load_jsonl = H1.sha, H1.canonical_hash, H1.safe, H1.atomic_json, H1.append, H1.load_jsonl
method_order, assert_gpu_exclusive, conditioning_trace, PilotCounts = H1.method_order, H1.assert_gpu_exclusive, H1.conditioning_trace, H1.PilotCounts
Checks, capacity_guard, cleanup_committed, file_hashes = H1.Checks, H1.capacity_guard, H1.cleanup_committed, H1.file_hashes
frozen_dialogues, strict_score = H1.frozen_dialogues, H1.strict_score
normal_request, build_official_layout = E.normal_request, E.build_official_layout
ThreadedReadTrace, WriteTimer, LogitTies, compact_trace, compact_json, scratch_bytes = (
    E.ThreadedReadTrace, E.WriteTimer, E.LogitTies, E.compact_trace, E.compact_json, E.scratch_bytes)
stat_inventory, load_servers, environment_sample, _chunks_of = E.stat_inventory, E.load_servers, E.environment_sample, E._chunks_of
LOGIT_ARMS = {}


def source_hashes():
    paths = (list((ROOT / "mmimpress").rglob("*.py")) + [p for p in (ROOT / "scripts").glob("*.py") if p.name not in REPORT_ONLY]
             + list((ROOT / "tests").glob("test_*.py")))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


REPORT_SCRIPT = re.compile(r"^scripts/\d+_report_.*\.py$")


def freeze_violations(frozen):
    """Frozen source files that changed or disappeared.

    Files created after the freeze are not compared (nothing frozen imports
    them), and neither are report scripts (they only read committed rows):
    unrelated work in the repository must not stop a running experiment.
    """
    now = source_hashes()
    return sorted(k for k, v in frozen.items() if now.get(k) != v and not REPORT_SCRIPT.match(k))


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


# ---------------------------------------------------------------- requests
def stored_hit(runner, servers, ctx, factory, method, image_id, store_path, logits=False):
    """scripts/199 ``stored_hit`` with this run's arm table.  The FullLoad and
    IMPRESS calls are unchanged; both packed kinds go through the same
    ``request_llava15_packed`` call (the store decides the row order)."""
    from mmimpress.llava15.serve import request_llava15_packed
    arm = ARMS[method]
    family = arm["family"]
    ties = LogitTies(runner) if logits else None
    with QA._NoVisionForward(runner) as guard:
        c0 = time.perf_counter(); ctx.reader.drop_all(); conditioning = (time.perf_counter() - c0) * 1000
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
        prompt, history, entries = factory()
        tokenized = runner.processor.tokenizer(prompt, return_tensors="pt")
        suffix = QA._suffix_from_tokenized(runner, tokenized).to(runner.model.device)
        selectors = []
        with PilotCounts(runner) as counts, ThreadedReadTrace() as reads:
            if ties is not None:
                ties.__enter__()
            try:
                if family == "fullload":
                    result = servers["base"].request(ctx, mode="fullload", suffix_ids=suffix, cold=False, prompt_text=prompt, cache_id=image_id,
                                                     prefetch="layer_stream" if arm["async"] else "none",
                                                     max_ahead=READAHEAD["max_ahead"], selector_out=selectors)
                elif family == "impress_official":
                    result = impress_official.request_official(
                        servers["base"], ctx, suffix, ratio=arm["ratio"], probe=OFFICIAL["probe_heads"],
                        sim_threshold=OFFICIAL["sim_threshold"], prefetch=OFFICIAL["prefetch_async"] if arm["async"] else "none",
                        cold=False, selector_out=selectors)
                else:
                    result = request_llava15_packed(servers["base"], ctx, suffix_ids=suffix, cold=False, image_id=image_id,
                                                    readahead="layer_stream" if arm["async"] else "none", max_ahead=READAHEAD["max_ahead"])
            finally:
                if ties is not None:
                    ties.__exit__(None, None, None)
        returned = time.perf_counter()
    s = reads.summary()
    assert s["bytes"] == result["io"]["bytes"] and s["preads"] == result["io"]["preads"]
    if family == "impress_official":      # after the request, outside every timer
        sel = selectors[0]
        result["layer_log"] = [dict(r) for r in sel.log]
        result["probe_sources"] = [[r["layer"], r["source"], r["ready_on_arrival"]] for r in sel.probe_log]
        result["layer_diagnostics"] = sel.diagnostics()
        result["raster_chunks"] = impress_official.alternative_chunk_counts(
            sel.keep_masks, sel.log, ctx.meta, lambda li: ctx.meta["order"][li])
        result["official_params"] = {"ratio": float(arm["ratio"]), "probe": OFFICIAL["probe_heads"], "sim_threshold": OFFICIAL["sim_threshold"]}
    del selectors
    if ties is not None:
        result["logit_ties"] = ties.steps
    result.update(projection_calls=counts.result(), scoring_calls=0,
                  end_to_end_ttft_ms=(result["first_token_at_s"] - started) * 1000, request_e2e_ms=(returned - started) * 1000,
                  request_started_at_s=started, vision_forward_count=guard.calls, page_cache_conditioning_ms=conditioning,
                  cap_reached=(len(result["generated_token_ids"]) == 16 and result["generated_token_ids"][-1] != runner.processor.tokenizer.eos_token_id),
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                  os_read_trace=compact_trace(s["calls"], store_path, result["core_started_at_s"]),
                  pread_calls=s["pread_calls"], preadv_calls=s["preadv_calls"])
    ctx.cache.k = None; ctx.cache.v = None
    return result, {"prompt": prompt, "history": history, "history_entries": entries, "suffix_input_ids": suffix.cpu().tolist()}


def stored_request(runner, servers, ctx, factory, method, image_id, store_path, logits=False):
    from mmimpress.serve import BIAS
    original = [l.self_attn.forward for l in runner.layers]
    hooks_before = [len(l._forward_pre_hooks) for l in runner.layers]
    with conditioning_trace() as conditioning:
        result, diag = stored_hit(runner, servers, ctx, factory, method, image_id, store_path, logits)
    assert all(l.self_attn.forward == f for l, f in zip(runner.layers, original)), "adapter not restored"
    assert [len(l._forward_pre_hooks) for l in runner.layers] == hooks_before, "layer hook left behind"
    assert not BIAS, "stale mask after request"
    assert conditioning and all(e["status"] == "SUCCESS" for e in conditioning), "page cache conditioning not verified"
    root = Path(store_path).resolve()
    result.update(conditioned_files=sorted(Path(e["path"]).relative_to(root).as_posix() for e in conditioning),
                  adapter_restored=True, page_cache_conditioning_method="posix_fadvise_DONTNEED",
                  page_cache_conditioning_excluded_from_ttft=True)
    return result, diag


# -------------------------------------------------------------- persistence
def _native_persist(runner, server, result, diag, path, image_id, kind, source, docs, scores_path):
    extra = {"dataset": DATASET, "source_dialog_id": source["dialog_id"], "source_turn_id": 1,
             "source_method_id": source["method_id"], "source_execution_id": source["physical_execution_id"],
             "source_logical_request_id": source["logical_request_id"], "run_schema": SCHEMA, "checkpoint_revision": REVISION}
    if kind in FULL_KINDS:
        with WriteTimer() as timer:
            receipt = persist_captured_llava15_raster_prefix(
                runner, result["captured_past_key_values"], diag["enc_cpu"]["input_ids"], path, image_id=image_id,
                probe_heads=SERVER_ARGS["probe_heads"], chunk_size=SERVER_ARGS["chunk_size"],
                image_input_sha256=diag["image_input_sha256"], extra_metadata=extra)
        receipt["write_by_file_name"] = timer.by_name
        if kind == NEW_KIND:
            receipt["official_turn1"] = build_official_layout(runner, path, diag, scores_path)
        return receipt
    if kind == PACKED_KIND:      # the existing importance-order writer call, row order not passed
        return S.native_persist(runner, result["captured_past_key_values"], diag["enc_cpu"], diag["vision"], path, image_id,
                                docs[VECTOR], diag["image_input_sha256"], extra)
    assert kind == ORIG_KIND
    return persist_captured_llava15_prefix(runner, result["captured_past_key_values"], diag["enc_cpu"]["input_ids"], diag["vision"].result_cpu(),
                                           path, budget=docs[VECTOR], image_id=image_id, capture_stats=diag["vision"],
                                           image_input_sha256=diag["image_input_sha256"], extra_metadata=extra, row_order=ROW_ORDER_ORIGINAL)


def persist(runner, server, result, diag, path, image_id, kind, source, docs, scores_path):
    return support.persist_llava(lambda: _native_persist(runner, server, result, diag, path, image_id, kind, source, docs, scores_path),
                                 path, kind, source)


def activate(runner, kind, path, docs):
    started = time.perf_counter()
    if kind in FULL_KINDS:
        from mmimpress.serve import ImageContext
        ctx = ImageContext(path, runner.model.device, drop_cache=False, require_v_hidden=False)
        validate_llava15_raster_meta(ctx.meta, SERVER_ARGS["probe_heads"])
        if kind == NEW_KIND:
            impress_official.validate_official_store(ctx.meta)
        completion = ctx.integrity_activation          # full checksum scan of the completion envelope
        pinned = 0
    else:
        from mmimpress.llava15.serve import Llava15PackedContext
        ctx = Llava15PackedContext(path, runner.model.device, expected_budget=docs[VECTOR])
        ctx.validate_packed_layout()
        assert ctx.row_order == PACKED_KINDS[kind] and ctx.meta["store_format"] == STORE_FORMATS[kind]
        completion = ctx.integrity_activation
        pinned = int(ctx.reader.total)
    ctx.activation = {"kind": kind, "host_tensor_bytes": sum(t.numel() * t.element_size() for t in ctx.cache.sys_kv.values()),
                      "pinned_host_buffer_bytes": pinned, "gpu_tensor_bytes": 0, "visual_payload_resident_bytes": 0,
                      "file_bytes": sum((Path(path) / n).stat().st_size for n in ("meta.json", "sys_kv.pt")),
                      "completion_integrity": completion, "event_id": str(uuid.uuid4()), "event_kind": "activation"}
    ctx.activation["outer_wall_ms"] = (time.perf_counter() - started) * 1000
    return ctx


def store_equivalence(reorder_path, original_path):
    """Gate 1 for one image: the original-order store against the importance
    store, both read back from disk.

    * layer l's original indices == sorted(order[:k_l]) of the importance store;
    * every stored token's K and V rows are bit-identical in the two files;
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
                   and mo["packed_sep_offset"] == mr["packed_sep_offset"] and mo["packed_sep_bytes"] == mr["packed_sep_bytes"] == 0
                   and mo["packed_row_bytes"] == mr["packed_row_bytes"] and mo["newline_idx"] == mr["newline_idx"] == []
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
    del br, bo
    sr = torch.load(reorder_path / "sys_kv.pt", weights_only=True); so = torch.load(original_path / "sys_kv.pt", weights_only=True)
    sys_equal = all(torch.equal(sr[n], so[n]) for n in ("k", "v"))
    no_permutation = (not any(key in mo for key in IMPORTANCE_ORDER_KEYS) and not (original_path / "visionzip_layout.pt").exists()
                      and mo.get("row_order") == ROW_ORDER_ORIGINAL)
    ok = (all(sets_equal) and table_equal and size_equal and rows_checked == 2 * sum(counts) and rows_different == 0
          and sys_equal and no_permutation)
    return {"status": "PASS" if ok else "FAIL", "layers": len(counts), "layer_sets_equal": int(sum(sets_equal)),
            "byte_table_equal": bool(table_equal), "payload_bytes_equal": bool(size_equal), "payload_bytes": int(mr["packed_total_bytes"]),
            "original_payload_bytes": int(mo["packed_total_bytes"]),
            "kv_rows_compared": int(rows_checked), "kv_rows_different": int(rows_different), "system_kv_equal": bool(sys_equal),
            "original_metadata_has_no_importance_permutation": bool(no_permutation),
            "reorder_packed_sha256": mr["packed_sha256"], "original_packed_sha256": mo["packed_sha256"],
            "metadata_bytes": {"reorder": {"meta.json": (reorder_path / "meta.json").stat().st_size,
                                           "visionzip_layout.pt": (reorder_path / "visionzip_layout.pt").stat().st_size},
                               "original": {"meta.json": (original_path / "meta.json").stat().st_size}}}


def make_row(runner, result, diag, meta, *, methods, method, dialog, turn, phase, run, attempt, physical_id, ordinal, store_info, config):
    arm = ARMS[method]
    question = dialog["turns"][turn - 1]
    n = int(meta["n_spatial"]); hit = turn > 1 and arm["family"] != "recompute"
    logical = request_id(run.name, phase, method, dialog["dialog_id"], turn)
    suffix = QA._suffix_from_tokenized(runner, runner.processor.tokenizer(diag["prompt"], return_tensors="pt")).cpu().tolist()
    if diag.get("suffix_input_ids") is not None:
        assert diag["suffix_input_ids"] == suffix
    value = {a: b for a, b in result.items() if a not in ("captured_past_key_values", "first_logits", "logits")}
    g = {key: meta.get(key, []) if key == "padding_idx" else meta[key] for key in
         ("num_heads", "head_dim", "num_layers", "v_token_start", "v_token_num", "n_spatial", "prefix_len", "dtype", "newline_idx", "padding_idx")}
    order = method_order(methods, ordinal)
    io = result.get("io", {}) if hit else {}
    per_kind = io.get("per_kind", {})
    wait = result.get("read_wait") or {}
    blocked = wait.get("read_blocked_ms")
    if hit and blocked is None and arm["family"] == "ours":
        blocked = result.get("packed_read_ms")          # synchronous packed read: the request thread is blocked for the whole read
    return {"schema_version": SCHEMA, "experiment_id": run.name, "phase": phase, "attempt_id": attempt, "dataset": DATASET,
            "logical_request_id": logical, "request_id": logical, "physical_execution_id": physical_id,
            "model_id": MODEL_ID, "model_revision": REVISION, "method_id": method, "family": arm["family"], "async_read": bool(arm["async"]),
            "store_kind": arm["kind"], "row_order": PACKED_KINDS.get(arm["kind"]), "image_id": dialog["image_id"],
            "dialog_id": dialog["dialog_id"], "turn_id": turn, "question_id": str(question["question_id"]),
            "question": question["question"], "gold": question["answers"], "prediction": result["answer"],
            "score": strict_score(result["answer"], question["answers"][0]), "quality_metric": "strict normalized EM",
            "generated_token_ids": list(result["generated_token_ids"]),
            "prompt": diag["prompt"], "history": diag["history"], "history_entries": diag["history_entries"],
            "prompt_sha256": hashlib.sha256(diag["prompt"].encode()).hexdigest(), "suffix_input_ids": suffix,
            "suffix_ids_sha256": canonical_hash(suffix), "history_sha256": hashlib.sha256(diag["history"].encode()).hexdigest(),
            "config_sha256": config["_file_sha256"], "manifest_sha256": config["_manifest_sha256"],
            "code_sha256": canonical_hash(config["source_sha256"]), "global_dialog_ordinal": ordinal,
            "method_order": order, "method_order_position": order.index(method),
            "request_started_at_s": result["request_started_at_s"], "first_token_at_s": result["first_token_at_s"],
            "request_finished_at_s": result["request_started_at_s"] + result["request_e2e_ms"] / 1000,
            "completed_at_unix": time.time(), "process_id": os.getpid(),
            "ttft_ms": result["end_to_end_ttft_ms"], "request_e2e_ms": result["request_e2e_ms"],
            "ssd_read_bytes": int(io.get("bytes", 0)), "ssd_preads": int(io.get("preads", 0)),
            "pread_calls": int(result.get("pread_calls", 0)), "preadv_calls": int(result.get("preadv_calls", 0)),
            "probe_read_bytes": int(per_kind.get("probe", {}).get("bytes", 0)), "k_read_bytes": int(per_kind.get("k", {}).get("bytes", 0)),
            "v_read_bytes": int(per_kind.get("v", {}).get("bytes", 0)), "packed_read_bytes": int(per_kind.get("packed", {}).get("bytes", 0)),
            "read_blocked_ms": blocked if hit else None,
            "N_content": n, "geometry": g, "source_store": store_info if hit else None,
            "provisioning_mode": "FRESH_IMAGE_STREAMING", "result": safe(value)}


# ------------------------------------------------------------------- audit
def audit_image_rows(rows, manifest, config, phase, image_id, artifact=None):
    """scripts/199 ``audit_image_rows`` for MT-GQA: strict EM instead of the VQA
    score, several dialogues per image, and both packed row orders."""
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
    ordinals = manifest["_ordinals"] if "_ordinals" in manifest else {d["dialog_id"]: i for i, d in enumerate(manifest["dialogues"])}
    doc = config["layer_budgets"][VECTOR]
    metas = {}
    if artifact is not None:
        artifact = Path(artifact)
        builds = []
        for kind in store_kinds(methods):
            p = json.loads((artifact / f"{kind}_persistence.json").read_text())
            meta = json.loads((artifact / f"{kind}_meta.json").read_text())
            hashes = json.loads((artifact / f"{kind}_hashes.json").read_text())
            files = p["files"]; builds.append(p["event_id"]); metas[kind] = meta
            c.require(hashes == {n: f["sha256"] for n, f in files.items() if n != "integrity.json"}, "complete_store_hash_inventory", kind)
            c.require(p["file_logical_bytes"] == sum(f["size"] for f in files.values()), "file_occupancy_sums", kind)
            c.require(p["service_persistence_ms"] >= p["native_writer_wall_ms"] > 0, "measured_store_completion", kind)
            c.require(p["source"]["dialog_id"] == ds[0]["dialog_id"] and kind_of(p["source"]["method_id"]) == kind
                      and p["source"]["method_id"] in methods, "first_dialogue_T1_build_same_kind", kind)
            acts = sorted(artifact.glob(f"{kind}_activation*.json"))
            c.require(bool(acts), "actual_activation_event", kind)
            for ap in acts:
                a = json.loads(ap.read_text())
                c.require(a["event_kind"] == "activation" and a["outer_wall_ms"] > 0 and a["kind"] == kind, "actual_activation_event", kind)
            n = meta["v_token_num"] - len(meta["newline_idx"]); width = meta["num_heads"] * meta["head_dim"] * 2
            vn, nl = meta["v_token_num"], meta["num_layers"]
            c.require(n == vn == N_VISUAL and nl == NUM_LAYERS and meta["newline_idx"] == [] and meta.get("dataset") == DATASET, "llava15_geometry_576_no_separator", kind)
            if kind in PACKED_KINDS:
                counts = keep_counts(doc, n)
                c.require(counts == S.independent_counts(doc["layer_keep_ratios"], n, VECTOR_FRACTION) and min(counts) >= 1, "PK_counts_independent_nonzero", kind)
                c.require(meta["layer_budget_r_min"] == doc["r_min"] == 0.0 and meta["layer_budget_r_max"] == doc["r_max"] == 1.0
                          and meta["layer_budget_total_ratio"] == doc["total_ratio"] == 0.25
                          and meta["layer_keep_ratios"] == doc["layer_keep_ratios"], "PK_clip_range_total_ratio_and_ratios_from_document", kind)
                c.require(meta["storage_policy"] == "prefix_ratio_packed" and meta["store_format"] == STORE_FORMATS[kind] and meta["budget_policy"] == "layerwise_v1", "PK_policy", kind)
                c.require(meta["layer_keep_counts"] == counts == meta["payload_rows_per_layer"], "PK_counts_recomputed", kind)
                c.require(sum(counts) == nl * ((n + 3) // 4), "PK_budget_equals_32_ceil_quarter_N", kind)
                c.require(meta["layer_budget_sha256"] == doc["layer_budget_sha256"] == VECTOR_RATIO_SHA and meta["layer_budget_id"] == VECTOR, "PK_vector_identity", kind)
                if PACKED_KINDS[kind] == ROW_ORDER_IMPORTANCE:
                    c.require(meta["full_importance_permutation"] == meta["order"] and sorted(meta["order"]) == list(range(vn))
                              and "row_order" not in meta, "PK_full_mapping", kind)
                else:
                    per_layer = layer_original_indices(meta)
                    c.require(meta.get("row_order") == ROW_ORDER_ORIGINAL and not any(key in meta for key in IMPORTANCE_ORDER_KEYS)
                              and "visionzip_layout.pt" not in files, "PK_original_order_has_no_importance_permutation", kind)
                    c.require([len(x) for x in per_layer] == counts and all(x == sorted(set(x)) and all(0 <= i < vn for i in x) for x in per_layer),
                              "PK_original_order_rows_ascending_exact_counts", kind)
                    c.require(meta.get("layout_uses_dataset_question") is False and meta.get("llm_used_for_layout_scoring") is False
                              and meta.get("calibration_questions") == 0, "PK_original_order_image_only_selection", kind)
                total = sum(2 * kk * width for kk in counts)
                c.require(meta["packed_total_bytes"] == total == files[PACKED_NAME]["size"] == PACKED_BYTES and meta["packed_sep_bytes"] == 0, "PK_exact_file_length", kind)
                c.require(files[PACKED_NAME]["sha256"] == meta["packed_sha256"], "PK_packed_sha256_matches_inventory", kind)
                c.require(not any(f.startswith("layer_") or f == "sep_kv.bin" for f in files), "PK_no_unpacked_payload_files", kind)
            else:
                c.require(meta.get("storage_policy", "full") == "full" and meta["probe_heads"] == SERVER_ARGS["probe_heads"] and meta["chunk_size"] == 64
                          and meta["n_chunks_per_layer"] == 9, "full_store_with_probe_sidecar", kind)
                c.require(files.get("sep_kv.bin", {}).get("size") == 0 and "v_hidden.pt" not in files, "no_separator_no_hidden", kind)
                for li in range(nl):
                    ok = (files[f"layer_{li:02d}/k.bin"]["size"] == files[f"layer_{li:02d}/v.bin"]["size"] == vn * width
                          and files[f"layer_{li:02d}/probe_k.bin"]["size"] == vn * SERVER_ARGS["probe_heads"] * meta["head_dim"] * 2
                          + (OFFICIAL["probe_heads"] * vn * 4 if kind == NEW_KIND else 0))
                    c.require(ok, "full_kv_and_probe_file_lengths", f"{kind}/layer{li}")
                if kind == "raster":
                    c.require(not meta.get("reordered", False) and "order" not in meta, "raster_not_reordered", kind)
                else:
                    t1 = p.get("official_turn1", {})
                    first = bykey.get((ds[0]["dialog_id"], p["source"]["method_id"], 1), {})
                    c.require(meta.get("physical_layout") == impress_official.LAYOUT and meta.get("reordered") is True and meta.get("order_is_per_layer") is True
                              and meta.get("order_requests") == 1 and meta.get("official_reference_commit") == OFFICIAL["reference_commit"], "official_layout_metadata", kind)
                    c.require(t1.get("suffix_ids_sha256") == first.get("suffix_ids_sha256") == meta.get("official_turn1_suffix_ids_sha256"), "official_statistics_from_T1_prompt", kind)
                    c.require(first.get("physical_execution_id") == p["source"]["physical_execution_id"], "official_statistics_source_execution", kind)
                    c.require(t1.get("capture_layers") == nl and t1.get("capture_rows") == meta["prefix_len"] + len(first.get("suffix_input_ids", [])),
                              "official_capture_is_the_T1_prefill", kind)
                    stats_file = artifact / t1.get("stats_file", "missing")
                    ok = stats_file.is_file() and sha(stats_file) == t1.get("stats_sha256")
                    c.require(ok, "official_statistics_evidence", kind)
                    if ok:
                        z = np.load(stats_file); pr, sc = z["prefix_rows"], z["order_scores"]
                        good = (pr.shape == (nl, OFFICIAL["probe_heads"], vn) and sc.shape == (nl, vn) and pr.dtype == sc.dtype == np.float32
                                and bool(np.isfinite(pr).all() and np.isfinite(sc).all()) and len(t1.get("trailer_sha256_per_layer", [])) == nl)
                        for li in range(nl if good else 0):
                            order = meta["order"][li]
                            good = good and order == np.argsort(-sc[li], kind="stable").tolist() and meta["newline_stored"][li] == []
                            stored = np.ascontiguousarray(pr[li][:, order].astype("<f4")).tobytes()
                            good = good and hashlib.sha256(stored).hexdigest() == t1["trailer_sha256_per_layer"][li]
                        c.require(good, "official_order_and_stored_prefix_rows_recomputed", kind)
        c.require(len(set(builds)) == len(store_kinds(methods)), "distinct_physical_build_events", image_id)
        if PACKED_KIND in metas and ORIG_KIND in metas:      # gate 1 evidence: the two packed stores of this image, compared on disk
            eq_path = artifact / "store_equivalence.json"
            c.require(eq_path.is_file(), "store_equivalence_recorded", image_id)
            if eq_path.is_file():
                eq = json.loads(eq_path.read_text())
                c.require(eq.get("reorder_packed_sha256") == metas[PACKED_KIND]["packed_sha256"]
                          and eq.get("original_packed_sha256") == metas[ORIG_KIND]["packed_sha256"], "store_equivalence_is_about_these_stores", image_id)
                # its PASS/FAIL is gate 1, judged by scripts/207; an unequal store does not block the image commit
    packed_sets = {}
    for key, r in bykey.items():
        rid = r["request_id"]; did, m, t = key; result = r["result"]; question = expected.get(key)
        if question is None:
            continue
        arm = ARMS[m]; family = arm["family"]
        c.require(r["schema_version"] == SCHEMA and r["model_id"] == MODEL_ID and r["model_revision"] == REVISION and r["dataset"] == DATASET, "schema_model", rid)
        c.require(rid == r["logical_request_id"] == f"{SCHEMA}:{r['experiment_id']}:{phase}:{MODEL_ID}:{m}:{did}:t{t}", "logical_identity", rid)
        c.require(r["phase"] == phase and r["image_id"] == image_id and r["family"] == family and r["async_read"] == arm["async"]
                  and r["store_kind"] == arm["kind"] and r["row_order"] == PACKED_KINDS.get(arm["kind"]), "phase_image_arm", rid)
        c.require(r["question_id"] == str(question["question_id"]) and r["question"] == question["question"] and r["gold"] == question["answers"], "frozen_question_gold", rid)
        c.require(r["score"] == strict_score(r["prediction"], r["gold"][0]), "strict_em_recalculation", rid)
        c.require(r["generated_token_ids"] == result["generated_token_ids"], "generated_ids_copied", rid)
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
        c.require(n == r["N_content"] == N_VISUAL and g["newline_idx"] == [] and g["prefix_len"] == g["v_token_start"] + N_VISUAL, "actual_content_geometry", rid)
        calls = result.get("projection_calls", {})
        prefill = {"q": 2, "k": 2, "v": 1} if (family == "impress_official" and not pixel) else {"q": 1, "k": 1, "v": 1}
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
        width = g["num_heads"] * g["head_dim"] * 2; vn = g["v_token_num"]; nl = g["num_layers"]
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
            original_rows = PACKED_KINDS[arm["kind"]] == ROW_ORDER_ORIGINAL
            counts = keep_counts(doc, n)
            c.require(counts == S.independent_counts(doc["layer_keep_ratios"], n, VECTOR_FRACTION), "PK_k_per_layer_independent", rid)
            c.require(result.get("k_per_layer") == counts, "PK_k_per_layer_recomputed", rid)
            c.require(result.get("attended_content_kv_count_per_layer") == counts and result.get("k_total") == sum(counts) == nl * ((n + 3) // 4), "PK_budget_total", rid)
            c.require(result.get("layer_budget_sha256") == doc["layer_budget_sha256"] and result.get("layer_budget_id") == VECTOR and result.get("ratio") == 0.25, "PK_vector_identity_hit", rid)
            if original_rows:
                c.require(result.get("selected_stored_ids") is None and result.get("row_order") == ROW_ORDER_ORIGINAL
                          and result.get("validated_prefix_layout") == "original_order_packed", "PK_original_order_contract", rid)
            else:
                c.require(result.get("selected_stored_ids") == list(range(max(counts))) and "row_order" not in result
                          and result.get("validated_prefix_layout") == "visionzip_image_only_packed", "PK_prefix_contract", rid)
            total = sum(2 * kk * width for kk in counts)
            c.require(r["ssd_read_bytes"] == total == PACKED_BYTES and r["pread_calls"] == 0, "PK_bytes_equal_file_no_pread", rid)
            c.require(all(e[0] == "preadv" and e[1] == PACKED_NAME for e in trace), "PK_preadv_on_packed_file_only", rid)
            if arm["async"]:
                plan = []; off = 0
                for kk in counts:
                    plan.append((off, 2 * kk * width)); off += 2 * kk * width
                plan = [x for x in plan if x[1]]
                c.require([(e[2], e[4]) for e in trace] == plan and off == total, "PK_async_ranges_layers_tile_file", rid)
                c.require(r["preadv_calls"] == len(plan) == nl, "PK_async_read_call_count", rid)
                c.require(not any(main_flags), "PK_async_reads_on_worker", rid)
                c.require(result.get("readahead_policy") == "layer_stream" and ahead is not None and ahead["items_read"] == ahead["items_taken"] == nl
                          and ahead["max_ahead"] == READAHEAD["max_ahead"], "PK_async_layer_stream", rid)
            else:
                c.require(len(trace) == 1 and trace[0][2] == 0 and trace[0][4] == trace[0][3] == total, "PK_single_preadv_whole_file", rid)
                c.require(r["preadv_calls"] == 1 and result.get("read_calls") == 1 and all(main_flags) and "readahead_policy" not in result, "PK_sync_read_call_counts", rid)
            original = result.get("selected_original_ids", [])
            c.require(len(original) == max(counts) and len(set(original)) == len(original) and all(0 <= i < vn for i in original), "original_ids_real", rid)
            if original_rows:
                c.require(original == sorted(original), "PK_original_order_ids_ascending", rid)
            c.require(result.get("keep_count_per_layer") == counts, "keep_equals_k_l_no_structural_rows", rid)
            c.require(result.get("store_format") == STORE_FORMATS[arm["kind"]] and result.get("directcopy") is False, "PK_scatter_path", rid)
            packed_sets.setdefault(m, []).append((original, counts, total))
            c.require(result.get("query_score_calls", 0) == 0 and result.get("static_score_calls", 0) == 0, "online_scoring_zero", rid)
        else:      # impress_official
            ratio = arm["ratio"]; thr = OFFICIAL["sim_threshold"]
            k_keep = max(1, min(n, math.ceil(ratio * n))); nc = (vn + 63) // 64
            c.require(result.get("k_keep") == k_keep and result.get("threshold") == thr and result.get("selection_path") == "impress_official"
                      and result.get("official_params") == {"ratio": ratio, "probe": OFFICIAL["probe_heads"], "sim_threshold": thr}, "official_parameters", rid)
            c.require(result.get("n_score_rows") == len(r["suffix_input_ids"]), "official_suffix_rows_are_whole_suffix", rid)
            log = result.get("layer_log", [])
            c.require([x["layer"] for x in log] == list(range(nl)), "official_layer_log_complete", rid)
            probe_bytes = vn * SERVER_ARGS["probe_heads"] * g["head_dim"] * 2 + OFFICIAL["probe_heads"] * vn * 4
            probes = [e for e in trace if e[1].endswith("/probe_k.bin")]
            c.require([e[1] for e in probes] == [f"layer_{li:02d}/probe_k.bin" for li in range(nl)] and all(e[2] == 0 and e[4] == probe_bytes for e in probes),
                      "official_probe_block_one_read_per_layer", rid)
            c.require(r["probe_read_bytes"] == nl * probe_bytes, "official_probe_bytes", rid)
            c.require(r["probe_read_bytes"] + r["k_read_bytes"] + r["v_read_bytes"] == r["ssd_read_bytes"] and r["preadv_calls"] == 0, "official_bytes_by_kind_sum", rid)
            ok_layers = True; kbytes = vbytes = 0
            for x in log:
                li = x["layer"]; got = {}
                for name in ("k", "v"):
                    es = [e for e in trace if e[1] == f"layer_{li:02d}/{name}.bin"]
                    ok_layers = ok_layers and all(e[2] % (64 * width) == 0 and e[4] % width == 0 and e[2] + e[4] <= vn * width for e in es)
                    got[name] = sorted({ch for e in es for ch in _chunks_of(e[2], e[4], width)})
                    b = sum(e[4] for e in es)
                    kbytes += b if name == "k" else 0; vbytes += b if name == "v" else 0
                    ok_layers = ok_layers and b == sum(min(64, vn - ch * 64) for ch in got[name]) * width
                    if x["mode"] == "fallback":      # the whole layer, one read
                        ok_layers = ok_layers and len(es) == 1 and es[0][2] == 0 and es[0][4] == vn * width
                ok_layers = ok_layers and got["k"] == got["v"] and len(got["k"]) == x["k_chunks"] == x["v_chunks"]
                if x["mode"] == "consensus":
                    ok_layers = ok_layers and x["sim"] >= thr and len(got["k"]) >= (k_keep + 63) // 64
                else:
                    ok_layers = ok_layers and x["mode"] == "fallback" and x["sim"] < thr and len(got["k"]) == nc
            c.require(ok_layers, "official_kv_reads_match_layer_decisions", rid)
            c.require(kbytes == r["k_read_bytes"] and vbytes == r["v_read_bytes"], "official_kv_bytes_from_trace", rid)
            fb = sum(x["mode"] == "fallback" for x in log)
            c.require(math.isclose(result.get("fallback_rate", -1), fb / nl), "official_fallback_rate_recomputed", rid)
            diag = result.get("layer_diagnostics", [])
            c.require(len(diag) == nl and all((d["order_overlap_selected"] is None) == (x["mode"] == "fallback") for d, x in zip(diag, log)),
                      "official_layer_diagnostics_present", rid)
            c.require(len(result.get("raster_chunks", {}).get("chunks", [])) == nl, "official_computed_raster_chunks_present", rid)
            sources = result.get("probe_sources", [])
            kv_main = all(e[6] == 1 for e in trace if not e[1].endswith("/probe_k.bin"))
            if arm["async"]:
                c.require(result.get("prefetch") == "next_layer" and [s[1] for s in sources] == ["hook"] + ["prefetch"] * (nl - 1), "official_async_next_layer_probe_prefetch", rid)
                c.require(bool(probes) and probes[0][6] == 1 and not any(e[6] for e in probes[1:]) and kv_main, "official_async_only_probe_reads_on_worker", rid)
            else:
                c.require(result.get("prefetch") == "none" and all(s[1] == "hook" for s in sources) and all(main_flags), "official_sync_reads_on_request_thread", rid)
            if ratio == 1.0:
                c.require(fb == 0 and all(x["sim"] == 1.0 and x["k_chunks"] == nc for x in log) and r["k_read_bytes"] == r["v_read_bytes"] == nl * vn * width,
                          "official_r100_loads_everything", rid)
    for m, sets in packed_sets.items():
        c.require(all(s == sets[0] for s in sets), "same_image_selection_invariant", f"{image_id}/{m}")
        if metas:
            meta = metas[kind_of(m)]
            stored_ids = (meta["stored_original_indices"] if PACKED_KINDS[kind_of(m)] == ROW_ORDER_ORIGINAL else meta["order"][:len(sets[0][0])])
            c.require(sets[0][0] == stored_ids and sets[0][2] == meta["packed_total_bytes"], "meta_to_actual_selection_and_bytes", f"{image_id}/{m}")
    # recorded, not a commit condition
    t1 = {}
    for (did, m, t), r in bykey.items():
        if t == 1:
            t1.setdefault(did, set()).add(tuple(r["result"]["generated_token_ids"]))
    return dict(c.result(), image_id=image_id, phase=phase, requests=len(rows), expected_requests=len(expected), dialogs=len(ds),
                t1_tokens_identical_across_arms=all(len(v) == 1 for v in t1.values()),
                dialogues_with_t1_mismatch=sum(len(v) != 1 for v in t1.values()))


# ---------------------------------------------------------------- lifecycle
def completed_images(run, phase, config, manifest):
    done = {}
    for cp in sorted((run / phase / "images").glob("*/COMMITTED.json")):
        doc = json.loads(cp.read_text()); folder = cp.parent / doc["attempt_id"]
        if doc["config_sha256"] != config["_file_sha256"] or doc["manifest_sha256"] != config["_manifest_sha256"]:
            raise RuntimeError("resume config/manifest mismatch")
        for name, digest in doc["evidence_sha256"].items():
            if sha(folder / name) != digest:
                raise RuntimeError("committed evidence changed: " + str(folder / name))
        done[cp.parent.name] = doc
    return done


def tolerant_rows(path):
    """Rows of an interrupted journal; a torn final line (power loss) is dropped."""
    rows = []; torn = 0
    if not Path(path).exists():
        return rows, torn
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                torn += 1
    return rows, torn


def complete_prefix(rows, dialogs, methods):
    """(number of leading complete dialogues, kept rows, dropped rows).

    Dialogues run one after another, so the complete ones are a prefix of the
    image's dialogue list; a dialogue is complete when every (arm, turn) row
    of it was written."""
    by_dialog = {}
    for r in rows:
        by_dialog.setdefault(r["dialog_id"], []).append(r)
    want = {(m, t) for m in methods for t in (1, 2, 3)}
    n = 0
    for d in dialogs:
        got = by_dialog.get(d["dialog_id"], [])
        if len(got) != len(want) or {(r["method_id"], r["turn_id"]) for r in got} != want:
            break
        n += 1
    keep_ids = {d["dialog_id"] for d in dialogs[:n]}
    return n, [r for r in rows if r["dialog_id"] in keep_ids], [r for r in rows if r["dialog_id"] not in keep_ids]


def verify_scratch_stores(artifact, scratch, all_kinds):
    """Stores of an interrupted image against the hashes recorded when they
    were built.  Returns (ok, reason)."""
    if not scratch.is_dir() or scratch.is_symlink():
        return False, "scratch directory missing"
    if sorted(p.name for p in scratch.iterdir()) != sorted(all_kinds):
        return False, "scratch does not hold exactly the image's stores"
    for kind in all_kinds:
        names = [f"{kind}_persistence.json", f"{kind}_meta.json", f"{kind}_hashes.json"]
        if not all((artifact / n).is_file() for n in names):
            return False, f"{kind}: build evidence incomplete"
        try:
            p = json.loads((artifact / names[0]).read_text()); recorded = json.loads((artifact / names[2]).read_text())
            found = file_hashes(scratch / kind)
        except (ValueError, OSError) as exc:
            return False, f"{kind}: {exc!r}"
        seal = found.pop("integrity.json", None)
        if found != recorded or seal is None or seal != p["files"].get("integrity.json", {}).get("sha256"):
            return False, f"{kind}: store files differ from the recorded hashes"
    return True, "stores verified against the recorded SHA256 of every file"


def remove_scratch(run, scratch):
    scratch = Path(scratch)
    if not scratch.exists():
        return 0
    if scratch.is_symlink() or not scratch.resolve().is_relative_to((run / "scratch").resolve()) or scratch.resolve() == (run / "scratch").resolve():
        raise ValueError("refusing to remove a path outside this run's scratch")
    n = scratch_bytes(scratch)
    shutil.rmtree(scratch)
    return n


def plan_image(run, phase, image_id, dialogs, methods):
    """How an uncommitted image starts: fresh, or continued after its last complete dialogue."""
    parent = run / phase / "images" / image_id
    attempts = sorted(p for p in parent.glob("attempt_*") if p.is_dir())
    if not attempts:
        return {"mode": "fresh", "attempt": "attempt_0001", "resumed": False}
    last = attempts[-1]
    scratch = run / "scratch" / phase / image_id / last.name
    rows, torn = tolerant_rows(last / "raw.jsonl")
    n, kept, dropped = complete_prefix(rows, dialogs, methods)
    ok, reason = (False, "the image's first dialogue was not complete") if n == 0 else verify_scratch_stores(last, scratch, store_kinds(methods))
    base = {"resumed": True, "previous_attempt": last.name, "rows_found": len(rows), "torn_lines": torn, "complete_dialogues_found": n,
            "store_check": reason}
    if ok:
        return dict(base, mode="continue", attempt=last.name, n_complete=n, kept_rows=kept, dropped_rows=dropped,
                    restart_dialogues=[d["dialog_id"] for d in dialogs[n:]],
                    interrupted_dialogue=dialogs[n]["dialog_id"] if n < len(dialogs) else None)
    return dict(base, mode="fresh", attempt=f"attempt_{len(attempts) + 1:04d}", discard_attempt=last.name, discard_scratch=str(scratch),
                restart_dialogues=[d["dialog_id"] for d in dialogs],
                interrupted_dialogue=dialogs[n]["dialog_id"] if n < len(dialogs) else None)


def run_image(runner, servers, dialogs, ordinals, phase, run, manifest, config, tick=None):
    methods = config["phases"][phase]["methods"]; docs = config["layer_budgets"]
    assert len({d["image_id"] for d in dialogs}) == 1
    image_id = dialogs[0]["image_id"]; parent = run / phase / "images" / image_id
    parent.mkdir(parents=True, exist_ok=True)
    if (parent / "COMMITTED.json").exists():
        raise FileExistsError("image already committed")
    plan = plan_image(run, phase, image_id, dialogs, methods)
    attempt = plan["attempt"]
    artifact = parent / attempt
    scratch = run / "scratch" / phase / image_id / attempt
    kinds = {m: kind_of(m) for m in stored_methods(methods)}
    all_kinds = store_kinds(methods)
    ref_kind = all_kinds[0]
    paths = {k: scratch / k for k in all_kinds}
    metas, hashes, persistence, contexts, store_info = {}, {}, {}, {}, {}
    rows = []; pending = []; success = False; capturing = set(); peak_scratch = 0; first_dialog = 0
    resume_event = None
    if plan["resumed"]:
        resume_event = {"at_unix": time.time(), "phase": phase, "image_id": image_id, "mode": plan["mode"], "attempt": attempt,
                        "previous_attempt": plan["previous_attempt"], "rows_found": plan["rows_found"], "torn_lines": plan["torn_lines"],
                        "complete_dialogues_kept": plan.get("n_complete", 0), "store_check": plan["store_check"],
                        "interrupted_dialogue": plan["interrupted_dialogue"], "restart_dialogues": plan["restart_dialogues"],
                        "process_id": os.getpid()}
    if plan["mode"] == "continue":
        n_resume = len(list(artifact.glob("resume_*.json"))) + 1
        if (artifact / "failure.json").exists():
            os.replace(artifact / "failure.json", artifact / f"failure_before_resume_{n_resume:02d}.json")
        if plan["dropped_rows"] or plan["torn_lines"]:      # the interrupted dialogue's partial rows leave the journal
            with (artifact / f"discarded_rows_resume_{n_resume:02d}.jsonl").open("x") as f:
                for r in plan["dropped_rows"]:
                    f.write(json.dumps(r, allow_nan=False) + "\n")
                f.flush(); os.fsync(f.fileno())
            tmp = artifact / ".raw.jsonl.tmp"
            with tmp.open("w") as f:
                for r in plan["kept_rows"]:
                    f.write(json.dumps(r, allow_nan=False) + "\n")
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp, artifact / "raw.jsonl")
        rows = list(plan["kept_rows"]); first_dialog = plan["n_complete"]; capturing = set(all_kinds)
        for kind in all_kinds:
            persistence[kind] = json.loads((artifact / f"{kind}_persistence.json").read_text())
            metas[kind] = json.loads((artifact / f"{kind}_meta.json").read_text())
            hashes[kind] = json.loads((artifact / f"{kind}_hashes.json").read_text())
            store_info[kind] = {"kind": kind, "path": str(paths[kind]), "meta_sha256": persistence[kind]["files"]["meta.json"]["sha256"],
                                "source": persistence[kind]["source"], "physical_build_event_id": persistence[kind]["event_id"]}
        resume_event["dropped_rows"] = len(plan["dropped_rows"])
        atomic_json(artifact / f"resume_{n_resume:02d}.json", resume_event)
        append(artifact / "attempts.jsonl", {"status": "RESUMED", "at_unix": time.time(), "kept_dialogues": first_dialog,
                                             "restart_dialogue": plan["interrupted_dialogue"]})
    else:
        if plan["resumed"]:      # nothing of the interrupted attempt is reused: its stores are removed, its journal stays as evidence
            removed = remove_scratch(run, Path(plan["discard_scratch"]))
            resume_event["discarded_scratch_bytes"] = removed
            atomic_json(parent / plan["discard_attempt"] / "ABANDONED.json", dict(resume_event, note="rows of this attempt are not part of any result"))
        artifact.mkdir(exist_ok=False)
        atomic_json(artifact / "capacity_before.json", capacity_guard(run))
        scratch.mkdir(parents=True, exist_ok=False)
    if resume_event is not None:
        append(run / "resume_log.jsonl", resume_event)
    logit_arms = LOGIT_ARMS.get(phase, ())
    source_image_sha = sha(ROOT / dialogs[0]["image_path"])
    if not (artifact / "image_identity.json").exists():
        atomic_json(artifact / "image_identity.json", {"image_id": image_id, "image_path": str((ROOT / dialogs[0]["image_path"]).resolve()), "sha256": source_image_sha})
    try:
        assert_gpu_exclusive()
        for di, dialog in enumerate(dialogs):
            if di < first_dialog:
                continue
            ordinal = ordinals[dialog["dialog_id"]]; generated = {m: {} for m in methods}
            for turn in (1, 2, 3):
                for method in method_order(methods, ordinal):
                    physical = str(uuid.uuid4())
                    logical = request_id(run.name, phase, method, dialog["dialog_id"], turn)
                    source = {"dialog_id": dialog["dialog_id"], "method_id": method, "physical_execution_id": physical, "logical_request_id": logical}
                    append(artifact / "attempts.jsonl", dict(source, turn_id=turn, status="STARTED", at_unix=time.time()))
                    factory = lambda m=method: S.prompt_factory(methods, dialog, turn, m, generated[m])
                    kind = kinds.get(method, "none")
                    # the first arm of a store kind in the first dialogue's rotation builds that kind's store
                    capture = kind if di == 0 and turn == 1 and kind != "none" and kind not in capturing else "none"
                    if capture != "none":
                        capturing.add(capture)
                    if turn == 1 or method == "recompute":
                        result, diag = normal_request(runner, servers["base"], ROOT / dialog["image_path"], factory,
                                                      "raster" if capture in FULL_KINDS else ("packed" if capture != "none" else "none"),
                                                      with_stats=capture == NEW_KIND)
                        append(artifact / "execution_events.jsonl", dict(source, turn_id=turn, status="EXECUTED", prediction=result["answer"],
                                                                        generated_token_ids=result["generated_token_ids"], ttft_ms=result["end_to_end_ttft_ms"],
                                                                        request_e2e_ms=result["request_e2e_ms"]))
                        if capture != "none":
                            start = time.perf_counter()
                            p = persist(runner, servers["base"], result, diag, paths[capture], image_id, capture, source, docs,
                                        artifact / f"{capture}_turn1_stats.npz")
                            outer_ms = (time.perf_counter() - start) * 1000
                            persistence[capture] = dict(p, outer_helper_wall_ms=outer_ms, source=source,
                                                        capture_accounting="capture and exit materialization already within T1 request E2E; not added again")
                            persistence[capture].pop("meta", None)
                            metas[capture] = json.loads((paths[capture] / "meta.json").read_text())
                            # the completion scan of persist_llava hashed every payload file once; reuse it
                            hashes[capture] = {n: f["sha256"] for n, f in p["files"].items() if n != "integrity.json"}
                            store_info[capture] = {"kind": capture, "path": str(paths[capture]), "meta_sha256": p["files"]["meta.json"]["sha256"],
                                                   "source": source, "physical_build_event_id": p["event_id"]}
                            atomic_json(artifact / f"{capture}_persistence.json", persistence[capture])
                            compact_json(artifact / f"{capture}_meta.json", metas[capture])
                            atomic_json(artifact / f"{capture}_hashes.json", hashes[capture])
                            peak_scratch = max(peak_scratch, scratch_bytes(scratch))
                        result.pop("captured_past_key_values", None)
                        saved = {a: b for a, b in diag.items() if a not in ("enc_cpu", "vision", "turn1_stats")}
                        if di == 0 and turn == 1:
                            pending.append((method, result, saved, physical))
                            del diag, result
                            continue
                        diag = saved
                    else:
                        if kind not in contexts:
                            contexts[kind] = activate(runner, kind, paths[kind], docs)
                            name = f"{kind}_activation.json"
                            if (artifact / name).exists():      # an activation after a resume is a second event of the same store
                                name = f"{kind}_activation_resume_{len(list(artifact.glob(kind + '_activation*.json'))):02d}.json"
                            atomic_json(artifact / name, contexts[kind].activation)
                        result, diag = stored_request(runner, servers, contexts[kind], factory, method, image_id, paths[kind],
                                                      logits=method in logit_arms)
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
            if tick is not None:      # between dialogues, outside every request timer
                tick(len(rows))
        assert_gpu_exclusive()
        for ctx in contexts.values():      # file descriptors closed before the stores are re-hashed
            ctx.close()
        contexts.clear()
        peak_scratch = max(peak_scratch, scratch_bytes(scratch))
        # one full re-hash of every scratch file: proves the stores did not change while they were served,
        # and is the exact allowlist of the cleanup
        allowlist = file_hashes(scratch)
        for kind in all_kinds:
            got = {n[len(kind) + 1:]: h for n, h in allowlist.items() if n.startswith(kind + "/") and n != f"{kind}/integrity.json"}
            assert got == hashes[kind], "store mutated during requests"
        if PACKED_KIND in paths and ORIG_KIND in paths:      # gate 1: after every request of the image, outside every timer
            t0 = time.perf_counter()
            eq = store_equivalence(paths[PACKED_KIND], paths[ORIG_KIND])
            eq["compare_ms"] = (time.perf_counter() - t0) * 1e3
            atomic_json(artifact / "store_equivalence.json", eq, replace=True)
        atomic_json(artifact / "store_integrity_after.json", {"status": "PASS", "file_sha256": hashes}, replace=True)
        atomic_json(artifact / "disk_usage.json", {"peak_scratch_bytes": peak_scratch, "free_bytes_after_requests": shutil.disk_usage(run).free,
                                                   "store_bytes": {k: scratch_bytes(paths[k]) for k in all_kinds}}, replace=True)
        audit = audit_image_rows(load_jsonl(artifact / "raw.jsonl"), manifest, config, phase, image_id, artifact)
        audit["raw_sha256"] = sha(artifact / "raw.jsonl"); atomic_json(artifact / "independent_audit.json", audit, replace=True)
        if audit["status"] != "PASS":
            raise RuntimeError("INDEPENDENT_IMAGE_AUDIT_FAIL: " + repr(audit["failures"][:10]))
        atomic_json(artifact / "cleanup_allowlist.json", {"scratch": str(scratch), "file_sha256": allowlist, "ownership_run": run.name}, replace=True)
        commit = {"status": "COMMITTED", "image_id": image_id, "attempt_id": attempt, "requests": len(rows),
                  "raw_sha256": audit["raw_sha256"], "audit_sha256": sha(artifact / "independent_audit.json"),
                  "config_sha256": config["_file_sha256"], "manifest_sha256": config["_manifest_sha256"],
                  "t1_tokens_identical_across_arms": audit["t1_tokens_identical_across_arms"],
                  "resumed_inside_image": len(list(artifact.glob("resume_*.json")))}
        for stale in artifact.glob("*.jsonl.gz"):      # an archive left by a commit that was interrupted before it was published
            stale.unlink()
        commit["evidence_sha256"] = {p.name: sha(p) for p in artifact.iterdir() if p.is_file()}
        # journals -> lossless gzip (decompression SHA checked), then the atomic commit
        archive_before_commit(parent, commit, lambda path, value: atomic_json(path, value)); success = True
    except BaseException as exc:
        atomic_json(artifact / "failure.json", {"status": "PARTIAL", "error": repr(exc), "traceback": traceback.format_exc(),
                                                "completed_uncommitted_rows": len(rows), "main_final_rows_from_this_attempt": 0}, replace=True)
        raise
    finally:
        for ctx in contexts.values():
            ctx.close()
        contexts.clear(); gc.collect()
        if success:
            cleanup_committed(scratch, allowlist, audit, commit, run)
            if not any(scratch.parent.iterdir()):      # the now-empty per-image scratch directory
                scratch.parent.rmdir()
            remove_archived_duplicates(parent)
            atomic_json(artifact / "cleanup_receipt.json", {"status": "PASS", "removed_new_files": len(allowlist)})
    return len(rows)


def verify_freeze(run):
    import transformers
    config = json.loads((run / "config.json").read_text())
    changed = freeze_violations(config["source_sha256"])
    if changed:
        # development rehearsals only (run name contains "_dev_"; never a measurement): code may still change
        if not ("_dev_" in run.name and os.environ.get("L15_MAIN_DEV_ALLOW_SOURCE_DRIFT") == "1"):
            raise RuntimeError("FROZEN_SOURCE_CONFIG_MISMATCH: " + repr(changed[:10]))
        config["source_sha256"] = source_hashes()
    if sha(CONTRACT) != config["contract_sha256"]:
        raise RuntimeError("FROZEN_CONTRACT_MISMATCH")
    if sha(run / "manifest.json") != config["manifest_sha256"] or sha(run / "storage_plan.json") != config["storage_plan_sha256"]:
        raise RuntimeError("FROZEN_MANIFEST_OR_STORAGE_PLAN_MISMATCH")
    manifest = json.loads((run / "manifest.json").read_text())
    if frozen_dialogues() != manifest["dialogues"]:
        raise RuntimeError("FROZEN_DATASET_MISMATCH")
    doc = load_budget(run / "vectors" / f"layer_budget_llava15_{VECTOR}.json")
    frozen = config["layer_budgets"][VECTOR]
    if (sha(run / "vectors" / f"layer_budget_llava15_{VECTOR}.json") != VECTOR_FILE_SHA or doc["layer_budget_sha256"] != VECTOR_RATIO_SHA
            or doc["layer_keep_ratios"] != frozen["layer_keep_ratios"] or frozen["layer_budget_sha256"] != VECTOR_RATIO_SHA):
        raise RuntimeError("FROZEN_VECTOR_MISMATCH")
    if config["model"] != MODEL_ID or config["revision"] != REVISION or config["phase_methods"] != PHASE_METHODS:
        raise RuntimeError("FROZEN_ARMS_OR_MODEL_MISMATCH")
    if transformers.__version__ != config["environment"]["transformers"] or torch.__version__ != config["environment"]["torch"]:
        raise RuntimeError("FROZEN_ENVIRONMENT_MISMATCH")
    config.update(_file_sha256=sha(run / "config.json"), _manifest_sha256=sha(run / "manifest.json"))
    manifest["_ordinals"] = {d["dialog_id"]: i for i, d in enumerate(manifest["dialogues"])}
    return config, manifest


def committed_request_counts(run, manifest):
    out = {}
    for ph in PHASES:
        out[ph] = sum(json.loads(cp.read_text())["requests"] for cp in (run / ph / "images").glob("*/COMMITTED.json")) if (run / ph).exists() else 0
    return out


def run_phase(runner, servers, run, phase, config, manifest):
    done = completed_images(run, phase, config, manifest)
    ids = manifest["phase_dialogue_ids"][phase]
    keep = set(ids); groups = OrderedDict(); ordinals = manifest["_ordinals"]
    for d in manifest["dialogues"]:                      # manifest order, as the previous MT-GQA runs
        if d["dialog_id"] in keep:
            groups.setdefault(d["image_id"], []).append(d)
    assert sum(len(v) for v in groups.values()) == len(ids)
    per_dialog = 3 * len(config["phases"][phase]["methods"])
    expected = len(ids) * per_dialog; completed = sum(x["requests"] for x in done.values())
    assert expected == manifest["expected_requests"][phase]
    others = committed_request_counts(run, manifest)
    total_expected = sum(manifest["expected_requests"].values())
    started = time.time(); completed_at_start = completed

    def status(in_image=0, image=None):
        now = time.time(); done_now = completed + in_image
        new = done_now - completed_at_start
        rate = (now - started) / new if new > 0 else None
        later = [p for p in PHASES[PHASES.index(phase) + 1:]]
        remaining_later = sum(manifest["expected_requests"][p] - others[p] for p in later)
        STATUS.update(run, stage=f"{phase}_phase", phase=phase, phase_completed_requests=done_now, phase_expected_requests=expected,
                      completed_requests=sum(v for k, v in others.items() if k != phase) + done_now, expected_requests=total_expected,
                      phase_images_committed=len(done), phase_images=len(groups), current_image=image,
                      seconds_per_request_this_process=rate,
                      phase_eta_utc=STATUS.utc(now + rate * (expected - done_now)) if rate else None,
                      eta_utc=STATUS.utc(now + rate * (expected - done_now + remaining_later)) if rate else None,
                      eta_note="phase rate of this process applied to the remaining requests of this and the later request phases; "
                               "gate checks, the question-independence builds and the report are not included",
                      python_pid=os.getpid())

    status()
    STATUS.ssd_sample(run, f"{phase}_process_start", phase=phase, committed_images=len(done))
    for image in done:      # a commit that was interrupted before its scratch cleanup: the stores of a committed image are not needed
        leftover = run / "scratch" / phase / image
        if leftover.exists():
            append(run / "resume_log.jsonl", {"at_unix": time.time(), "phase": phase, "image_id": image, "mode": "committed_scratch_removed",
                                              "discarded_scratch_bytes": remove_scratch(run, leftover), "restart_dialogues": [],
                                              "process_id": os.getpid()})
    for index, (image, ds) in enumerate(groups.items()):
        if image in done:
            continue
        if (run / "STOP_REQUESTED").exists():
            raise RuntimeError("STOP_REQUESTED")
        t0 = time.time()
        completed += run_image(runner, servers, ds, ordinals, phase, run, manifest, config,
                               tick=lambda n, image=image: status(n, image))
        done[image] = {"requests": len(ds) * per_dialog}
        progress = {"phase": phase, "status": "RUNNING", "completed_final_requests": completed, "expected_requests": expected,
                    "completed_images": len(done), "images": len(groups), "image_id": image, "image_seconds": time.time() - t0,
                    "at_unix": time.time(), "elapsed_s_this_process": time.time() - started}
        atomic_json(run / f"{phase}_progress.json", progress, replace=True)
        status(0, image)
        STATUS.ssd_sample(run, "image_committed", phase=phase, image_id=image, committed_images=len(done))
        if len(done) % 10 == 0 or len(done) == len(groups) or phase != "main":
            print(json.dumps(progress), flush=True)
        if len(done) % 50 == 0:
            append(run / "environment_log.jsonl", dict(environment_sample(), phase=phase, completed_images=len(done),
                                                      free_bytes=shutil.disk_usage(run).free))
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
def build_manifest(limit=0):
    """``limit`` > 0 (development rehearsals only): the first ``limit`` gate / DEV images and ``limit + 1`` main images."""
    ds = frozen_dialogues()
    prior = json.loads((S.PRIOR / "manifest.json").read_text())
    assert prior["dialogues"] == ds
    for image, identity in prior["image_identity"].items():
        assert sha(ROOT / identity["image_path"]) == identity["sha256"], image
    ref = json.loads((RECOMP_REF_RUN / "manifest.json").read_text())      # the 2026-11-12 LLaVA-1.5 MT-GQA run: same dialogues, same images
    assert ref["dialogues"] == ds and ref["image_identity"] == prior["image_identity"]
    assert ref["index_sha256"] == S.INDEX_SHA and ref["workload_sha256"] == S.WORKLOAD_SHA
    smoke_ids = ref["smoke_dialogue_ids"]
    smoke_images = [d["image_id"] for d in ds if d["dialog_id"] in set(smoke_ids)]
    dev_images = json.loads(V2_SPLIT.read_text())["dev"]["image_ids"]
    all_images = {d["image_id"] for d in ds}
    assert len(dev_images) == 40 == len(set(dev_images)) and set(dev_images) <= all_images and len(set(smoke_images)) == 4 == len(smoke_images)
    image_sets = {"gate": set(smoke_images), "dev": set(dev_images), "main": all_images}
    if limit:
        first = list(dict.fromkeys(d["image_id"] for d in ds))
        image_sets = {"gate": set([i for i in first if i in image_sets["gate"]][:limit]),
                      "dev": set([i for i in first if i in image_sets["dev"]][:limit]), "main": set(first[:limit + 1])}
    phase_ids = {ph: [d["dialog_id"] for d in ds if d["image_id"] in image_sets[PHASE_IMAGES[ph]]] for ph in PHASES}
    order = list(dict.fromkeys(d["image_id"] for d in ds))                 # execution order: first appearance in the manifest
    phase_images = {ph: [i for i in order if i in image_sets[PHASE_IMAGES[ph]]] for ph in PHASES}
    manifest = {"dataset": DATASET, "schema_version": SCHEMA, "index_sha256": S.INDEX_SHA, "workload_sha256": S.WORKLOAD_SHA,
                "dialogues": ds, "image_identity": prior["image_identity"], "phase_dialogue_ids": phase_ids, "phase_images": phase_images,
                "phase_methods": PHASE_METHODS,
                "expected_requests": {p: len(phase_ids[p]) * 3 * len(PHASE_METHODS[p]) for p in PHASES},
                "prior_manifest_sha256": sha(S.PRIOR / "manifest.json"), "recomp_reference_manifest_sha256": sha(RECOMP_REF_RUN / "manifest.json"),
                "gate_policy": "every dialogue of the four images of the HOLD smoke dialogue ids (the gate images of the earlier MT-GQA runs)",
                "dev_policy": "every MT-GQA dialogue of the 40 DEV images of results/layer_budget_kv25_v2_20261106T073502Z/data_split.json",
                "main_policy": "all 398 images / 4,061 dialogues, manifest order"}
    assert limit or manifest["expected_requests"] == EXPECTED_REQUESTS, manifest["expected_requests"]
    if limit:
        manifest["rehearsal_image_limit"] = limit
    return manifest


def freeze(run, source_before=None):
    import transformers
    assert run.parent == ROOT / "runs" and run.name.startswith(RUN_PREFIX) and CONTRACT.is_file()
    ref_config = json.loads((IMPL_REF_RUN / "config.json").read_text())
    ref_env = ref_config["environment"]
    assert (transformers.__version__, torch.__version__, np.__version__, sys.executable) == (
        ref_env["transformers"], ref_env["torch"], ref_env["numpy"], ref_env["executable"]), "not the environment of the 2026-11-18 run"
    assert ref_config["server_args"] == SERVER_ARGS and ref_config["impress_official"] == OFFICIAL
    assert all(ref_config["readahead"][k] == READAHEAD[k] for k in ("fullload_async", NEW_ASYNC, REORDER_ASYNC, "max_ahead"))
    # FullLoad / IMPRESS code == the 2026-11-18 freeze: every file of that freeze is byte-identical except the listed ones
    now = source_hashes(); before = ref_config["source_sha256"]
    changed = sorted(k for k in before if k in now and now[k] != before[k])
    removed = sorted(k for k in before if k not in now)
    added = sorted(k for k in now if k not in before)
    assert set(changed) <= set(CHANGED_SINCE_IMPL_REF) and not removed, (changed, removed)
    checkout = os.environ.get("IMPRESS_OFFICIAL_REPO")
    official_head = subprocess.check_output(["git", "-C", checkout, "rev-parse", "HEAD"], text=True).strip() if checkout else None
    assert official_head in (None, OFFICIAL["reference_commit"]), official_head
    run.mkdir(parents=True, exist_ok=False)
    # development rehearsals only (run name contains "_dev_"; never a measurement): a few images per phase
    limit = int(os.environ.get("L15_MAIN_DEV_LIMIT", "0")) if "_dev_" in run.name else 0
    manifest = build_manifest(limit)
    support.publish(run / "manifest.json", manifest)
    (run / "vectors").mkdir()
    target = run / "vectors" / f"layer_budget_llava15_{VECTOR}.json"
    shutil.copyfile(VECTOR_SOURCE, target)
    assert sha(target) == VECTOR_FILE_SHA == sha(VECTOR_SOURCE)
    doc = load_budget(target)
    old = ref_config["layer_budgets"][VECTOR]
    assert doc["layer_budget_sha256"] == VECTOR_RATIO_SHA == old["layer_budget_sha256"] and doc["layer_keep_ratios"] == old["layer_keep_ratios"]
    assert doc["alloc_id"] == VECTOR and doc["r_min"] == 0.0 and doc["r_max"] == 1.0 and doc["total_ratio"] == 0.25
    counts = keep_counts(doc, N_VISUAL)
    assert counts == S.independent_counts(doc["layer_keep_ratios"], N_VISUAL, VECTOR_FRACTION)
    assert counts[:3] == S.EXPECTED_PYR_HEAD and counts[-4:] == S.EXPECTED_PYR_TAIL and sum(counts) == S.EXPECTED_PYR_SUM
    assert sum(2 * k * S.ROW_BYTES for k in counts) == PACKED_BYTES
    free = shutil.disk_usage(run).free
    full_store = (2 * NUM_LAYERS * N_VISUAL * S.ROW_BYTES + NUM_LAYERS * N_VISUAL * SERVER_ARGS["probe_heads"] * 128 * 2
                  + NUM_LAYERS * OFFICIAL["probe_heads"] * N_VISUAL * 4 + (2 << 20))      # K/V + probe keys + prefix-row trailer + sys/meta
    packed_store = PACKED_BYTES + (2 << 20)
    bundle = 2 * full_store + 2 * packed_store + N_VISUAL * S.ROW_BYTES      # gate phase: four stores + one layer file in flight during the IMPRESS repack
    assert bundle < 2 * GiB
    n_img = {ph: len(manifest["phase_images"][ph]) for ph in PHASES}
    writes = {"gate": n_img["gate"] * (3 * full_store + 2 * packed_store),
              "devgate": n_img["devgate"] * (2 * full_store + packed_store),
              "dev": n_img["dev"] * (3 * full_store + 2 * packed_store),
              "main": n_img["main"] * (3 * full_store + packed_store),
              "question_independence": (n_img["gate"] + n_img["devgate"]) * 6 * packed_store}
    writes["total"] = sum(writes.values())
    plan = {"policy": "FRESH_IMAGE_STREAMING (image-major): one live image; stores built from the arms' own T1 of the image's first dialogue -> "
                      "served -> audited -> committed -> run-local scratch deleted by exact SHA256 allowlist",
            "minimum_reserve_bytes": 30 * GiB, "peak_working_set_bytes": 4 * GiB, "largest_image_bundle_bound_bytes": 2 * GiB,
            "maximum_new_payload_bytes": 20 * GiB, "raw_and_evidence_growth_bound_bytes": 16 * GiB,
            "per_image_bundle_bytes": bundle, "full_store_bytes": full_store, "packed_store_bytes": packed_store,
            "expected_ssd_write_bytes": writes,
            "expected_ssd_write_note": "per image: FullLoad store + IMPRESS store (written, then rewritten in the reordered layout) + packed store(s); "
                                       "devgate has no raster FullLoad store; six packed stores per gate/DEV image for the question-independence check; "
                                       "raw rows and evidence not included",
            "free_bytes_at_freeze": free,
            "deletion_scope": "runs/<RUN>/scratch only (exact SHA allowlist after audit+commit; the stores of an interrupted image that cannot be "
                              "verified on resume; the question-independence stores after their comparison); existing stores never deleted"}
    plan["required_free_bytes"] = plan["minimum_reserve_bytes"] + plan["peak_working_set_bytes"] + plan["raw_and_evidence_growth_bound_bytes"]
    plan["status"] = "PASS" if free >= plan["required_free_bytes"] else "BLOCKED_STORAGE"
    support.publish(run / "storage_plan.json", plan)
    if plan["status"] != "PASS":
        raise RuntimeError(f"BLOCKED_STORAGE: free {free} < required {plan['required_free_bytes']}")
    budgets = {VECTOR: dict(doc, path=str(target.relative_to(ROOT)))}
    config = {"schema_version": SCHEMA, "dataset": DATASET, "model": MODEL_ID, "revision": REVISION, "seed": 1234, "greedy": True, "max_new_tokens": 16,
              "batch_size": 1, "cpu_threads": 2, "history_policy": "method_local_generated", "metric": "strict normalized EM",
              "phase_methods": PHASE_METHODS,
              "phases": {ph: {"methods": PHASE_METHODS[ph], "dialogues": len(manifest["phase_dialogue_ids"][ph]),
                              "images": len(manifest["phase_images"][ph])} for ph in PHASES},
              "arms": {m: dict(a) for m, a in ARMS.items()}, "main_methods": MAIN_METHODS,
              "server_args": SERVER_ARGS, "impress_official": OFFICIAL, "readahead": READAHEAD,
              "official_checkout": {"path": str(Path(checkout).resolve()) if checkout else None, "head": official_head,
                                    "repository": OFFICIAL["reference_repository"]},
              "store_kinds": {"raster": "full visual KV, raster order (FullLoad)", NEW_KIND: "full visual KV + probe key copy, per-layer reordered (IMPRESS)",
                              ORIG_KIND: "packed, row_order=original (Ours)", PACKED_KIND: "packed, row_order=importance (writer default; DEV reference)"},
              "store_formats": STORE_FORMATS, "layer_budgets": budgets, "vector_file_sha256": VECTOR_FILE_SHA,
              "expected_rows_N576": counts, "packed_bytes": PACKED_BYTES,
              "arm_order": "cyclic rotation by dialogue ordinal (ordinal % number of arms of the phase)",
              "source_sha256": now, "source_changed_since_20261118_freeze": changed, "source_added_since_20261118_freeze": added,
              "implementation_reference": {"run": str(IMPL_REF_RUN.relative_to(ROOT)), "config_sha256": sha(IMPL_REF_RUN / "config.json"),
                                           "harness": "scripts/199_eval_llava15_mtvqa_impress_code.py (imported)",
                                           "harness_sha256": sha(ROOT / "scripts/199_eval_llava15_mtvqa_impress_code.py")},
              "recomp_reference": {"run": str(RECOMP_REF_RUN.relative_to(ROOT)), "method": "recompute", "phase": "main"},
              "contract_sha256": sha(CONTRACT),
              "manifest_sha256": sha(run / "manifest.json"), "storage_plan_sha256": sha(run / "storage_plan.json"),
              "source_T1_capture": "the image's first dialogue; per store kind, the first arm of that kind in the rotation; every arm still runs its own full-pixel T1 in every dialogue",
              "resume_policy": "committed images skipped; inside an interrupted image complete dialogues are kept and the interrupted dialogue restarts "
                               "at turn 1; unverifiable stores -> the image restarts",
              "environment": {"python": sys.version, "executable": sys.executable, "torch": torch.__version__, "transformers": transformers.__version__,
                              "numpy": np.__version__},
              "lifecycle_source": "scripts/156_eval_layer_budget_hold.py helpers + scripts/llava15_support.py + scripts/199 (imported, not copied)"}
    support.publish(run / "config.json", config)
    shutil.copy2(CONTRACT, run / "CONTRACT.md")
    if source_before is not None:
        shutil.copytree(source_before, run / "source_before")
    env = {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
           "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
           "pip_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)}
    support.publish(run / "environment_before.json", env)
    support.publish(run / "protection_before.json", stat_inventory(run, run / "protection_before.tsv.gz"))
    print(json.dumps({"freeze": "PASS", "run": str(run), "expected": manifest["expected_requests"], "storage": plan["status"],
                      "free_gib": free / 2**30, "transformers": transformers.__version__, "source_changed": changed, "source_added": added}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True, help="runs/llava15_mt_gqa_main_<UTC>")
    p.add_argument("--stage", choices=("freeze",) + PHASES, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--source-before", type=Path, default=None, help="freeze only: directory with the pre-change source evidence to copy")
    args = p.parse_args()
    run = args.run.resolve()
    if args.stage == "freeze":
        return freeze(run, args.source_before)
    assert run.parent == ROOT / "runs" and run.name.startswith(RUN_PREFIX)
    config, manifest = verify_freeze(run)
    assert json.loads((run / "storage_plan.json").read_text())["status"] == "PASS"
    phase = args.stage
    if phase == "main":      # the main run is admitted only by the recorded gate decision
        admit = json.loads((run / "gates" / "admission.json").read_text())
        if admit["status"] != "PASS" or admit["config_sha256"] != config["_file_sha256"]:
            raise RuntimeError("MAIN_NOT_ADMITTED")
    torch.set_num_threads(config["cpu_threads"]); torch.manual_seed(1234); np.random.seed(1234)
    torch.set_grad_enabled(False)
    assert_gpu_exclusive()
    runner, servers, warm = load_servers()
    try:
        atomic_json(run / f"{phase}_start_{time.strftime('%Y%m%dT%H%M%S')}.json",
                    {"started_unix": time.time(), "capacity": capacity_guard(run), "pid": os.getpid(), "warmup": warm,
                     "model": runner.describe(), "resume": bool(args.resume), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                     "committed_requests_at_start": committed_request_counts(run, manifest)[phase]})
        if (run / phase).exists() and not args.resume:
            raise FileExistsError("use --resume; previous artifacts preserved")
        receipt = run_phase(runner, servers, run, phase, config, manifest)
        atomic_json(run / f"{phase}_execution_complete.json", {"status": "PASS", **receipt, "finished_unix": time.time()}, replace=True)
    except BaseException as exc:
        error = {"status": "PARTIAL", "error": repr(exc), "traceback": traceback.format_exc(), "at_unix": time.time(),
                 "resume_command": f"scripts/run_llava15_mt_gqa_main.sh {run.relative_to(ROOT)}"}
        atomic_json(run / ("interruption_" + phase + "_" + time.strftime("%Y%m%dT%H%M%S") + ".json"), error)
        STATUS.update(run, last_error={"stage": f"{phase}_phase", "error": repr(exc), "at_utc": STATUS.utc(time.time())})
        print(json.dumps({k: v for k, v in error.items() if k != "traceback"}), flush=True)
        raise
    finally:
        for server in servers.values():
            server.close()


if __name__ == "__main__":
    main()

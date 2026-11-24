#!/usr/bin/env python3
"""LLaVA-NeXT-7B x MT-VQA-v2-reconstructed-25k main experiment: FullLoad / IMPRESS / Ours
measured in one run; ReComp is taken from the per-request record of the v3 run
(contract docs/llava_next_mtvqa_main_contract.md).

No method code is changed or added here.  Where the code comes from:

* FullLoad and IMPRESS are the arms of the 2026-11-17 run
  (scripts/196_eval_mt_gqa_impress_code.py, results/mt_gqa_impress_code_20261117T185832Z).
  That module is imported and its capture, layout and tracing functions are called as
  they are (``normal_request``, ``turn1_request_with_stats``, ``build_official_layout``,
  ``ThreadedReadTrace``, ``WriteTimer`` ...); the FullLoad and IMPRESS request calls
  below are the calls of that script.
* Ours is "Ours(원래 순서)" of the 2026-11-20 run (scripts/202_eval_mt_gqa_ours_order.py,
  results/mt_gqa_ours_order_20261120T052725Z): the PYR-beta-raw vector, 25 % of the KV
  budget, ``row_order="original"`` of ``mmimpress.packed_store`` / ``packed_serve``; the
  writer and request calls below are the calls of that script.
* Data, prompt and scoring: docs/mt_vqa_v2_reconstructed_25k_contract.md and
  docs/llava_next_mtvqa_pyr_contract.md -- 25,000 images, one dialogue x 3 turns per
  image, method-local generated history, the official ten-answer leave-one-out VQA
  score (``mmimpress.mtvqa25k``).

Phases
  gate    4 images (v3 smoke),   9 arms  (gates 2 and 3)
  pilot   40 images (v3 pilot40), 4 arms  (FullLoad, IMPRESS, Ours + ReComp; gates 4 and 6)
  main    25,000 images,          3 arms  (FullLoad, IMPRESS, Ours)

One image at a time: every arm computes its turn 1 from pixels, the first arm of a
store kind writes that store, turns 2-3 read it from the cold SSD, the image is audited
and committed, and its stores are deleted.

Resume (``--resume``): committed images are skipped; an image that was interrupted
starts again at turn 1 for every arm -- its partial rows are not part of any result and
its stores are removed and rebuilt.

Stages: --stage freeze | prerun | gate | pilot | main [--resume]
"""
from __future__ import annotations

import argparse
import ast
import gc
import gzip
import hashlib
import importlib.util
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

from mmimpress import impress_official  # noqa: E402
from mmimpress.layer_budget import load_budget_document  # noqa: E402
from mmimpress.mtvqa25k.adapter import (archive_before_commit, dialogues_for_base, enrich, read_rows,  # noqa: E402
                                        remove_archived_duplicates, score_from_strings)
from mmimpress.mtvqa25k.scorer import independent_score  # noqa: E402
from mmimpress.packed_serve import PackedImageContext, request_packed  # noqa: E402
from mmimpress.packed_store import (IMPORTANCE_ORDER_KEYS, ORIGINAL_STORE_FORMAT, PACKED_NAME, ROW_ORDER_ORIGINAL,  # noqa: E402
                                    layer_original_indices, persist_captured_packed_prefix)


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


E = load_script("_lnext_mtvqa_main_impress_code_eval", "196_eval_mt_gqa_impress_code.py")      # the 2026-11-17 harness, used as is
import llava_next_main_status as STATUS  # noqa: E402

SCHEMA = "llava-next-mtvqa-main-v1"
DATASET = "MT-VQA-v2-reconstructed-25k"
RUN_PREFIX = "llava_next_mtvqa_main_"
CONTRACT = ROOT / "docs/llava_next_mtvqa_main_contract.md"
DATA = ROOT / "data/mt_vqa_v2_reconstructed_25k"
WORKLOAD_SHA = "eb3c5e812fd6937cc0741cac6dd85d6839a7d00288dffbfa7ea8294272269cc8"
V3_RUN = ROOT / "runs/mt_vqa_v2_reconstructed_25k_v3_20261021T100500Z"
V3_LLAVA = V3_RUN / "llava"
V3_RECOMP, V3_FULL = "recompute", "fullload"
IMPL_REF_RUN = ROOT / "runs/mt_gqa_impress_code_20261117T185832Z"        # FullLoad / IMPRESS implementation reference
OURS_REF_RUN = ROOT / "runs/mt_gqa_ours_order_20261120T052725Z"          # Ours (original row order) implementation reference
PYRBETA_RUN = E.PYRBETA_RUN
VECTOR, VECTOR_RATIO_SHA, VECTOR_FILE_SHA = E.VECTOR, E.VECTOR_RATIO_SHA, E.VECTOR_FILE_SHA
VECTOR_SOURCE = PYRBETA_RUN / "vectors" / f"layer_budget_llava_{VECTOR}.json"
SERVER_ARGS, OFFICIAL = E.SERVER_ARGS, E.OFFICIAL                          # the settings of the 2026-11-17 run
NEW_KIND, NEW, NEW_ASYNC, NEW_R100, FULL_REORDERED = E.NEW_KIND, E.NEW, E.NEW_ASYNC, E.NEW_R100, E.FULL_REORDERED
FULL_KINDS = E.FULL_KINDS
RECOMP, FULL_ASYNC = "recompute", "fullload_async"
ORIG = f"ours_origorder_kv25_{VECTOR}_packed"                              # Ours, synchronous read (gate only); the ids of scripts/202
ORIG_ASYNC = ORIG + "_async"                                               # Ours
ORIG_REPEAT = ORIG + "_async_repeat"                                       # Ours executed a second time (gate only)
ORIG_KIND = f"packedorig_{VECTOR}"
READAHEAD = {FULL_ASYNC: "layer_stream", NEW_ASYNC: "next_layer", ORIG_ASYNC: "layer_stream", ORIG_REPEAT: "layer_stream", "max_ahead": None}
GiB = 1 << 30
ROW_BYTES = 8192                                                           # one token's K (or V) of one layer: 32 heads x 128 x FP16
NUM_LAYERS = 32
SYSTEM_PREFIX_BYTES = 2_622_802                                            # sys_kv.pt of this prompt template (measured, 2026-11-17 run)
FREE_SPACE_FACTOR = 20                                                     # free space >= 20 x the largest per-image store bundle
MAIN_FAILED_IMAGES_MAX = 25                                                # main only: more failed images than this stop the run
MAIN_FAILED_CONSECUTIVE_MAX = 3
GPU_WAIT_SECONDS = 6 * 3600                                                # how long a phase waits for another process to leave the GPU
# seconds per main image used for the time estimate before the pilot has run (development rehearsal of 2026-11-21: 3 main images,
# 12.6 - 16.3 s each, mean 14.2 s; not a measurement)
REHEARSAL_SECONDS_PER_IMAGE = 14.2

ARMS = OrderedDict([
    (RECOMP,         {"family": "recompute", "kind": "none", "async": False, "display": "ReComp"}),
    (FULL_ASYNC,     {"family": "fullload", "kind": "raster", "async": True, "display": "FullLoad"}),
    (NEW,            {"family": "impress_official", "kind": NEW_KIND, "async": False, "display": "IMPRESS, 동기 읽기", "ratio": 0.25}),
    (NEW_ASYNC,      {"family": "impress_official", "kind": NEW_KIND, "async": True, "display": "IMPRESS", "ratio": 0.25}),
    # gate only: the IMPRESS path at 100 % retention, and the FullLoad loader (no selection, no mask) on the SAME reordered store
    (NEW_R100,       {"family": "impress_official", "kind": NEW_KIND, "async": False, "display": "IMPRESS (유지 100%)", "ratio": 1.0}),
    (FULL_REORDERED, {"family": "fullload", "kind": NEW_KIND, "async": False, "display": "FullLoad loader (재배치 store)"}),
    (ORIG,           {"family": "ours", "kind": ORIG_KIND, "async": False, "display": "Ours, 동기 읽기"}),
    (ORIG_ASYNC,     {"family": "ours", "kind": ORIG_KIND, "async": True, "display": "Ours"}),
    (ORIG_REPEAT,    {"family": "ours", "kind": ORIG_KIND, "async": True, "display": "Ours, 두 번째 실행"}),
])
MAIN_METHODS = [FULL_ASYNC, NEW_ASYNC, ORIG_ASYNC]
PHASE_METHODS = {
    "gate": [RECOMP, FULL_ASYNC, NEW, NEW_ASYNC, NEW_R100, FULL_REORDERED, ORIG, ORIG_ASYNC, ORIG_REPEAT],
    "pilot": MAIN_METHODS + [RECOMP],
    "main": list(MAIN_METHODS),
}
PHASES = ("gate", "pilot", "main")
V3_PHASE = {"gate": "smoke", "pilot": "pilot40", "main": "main"}
EXPECTED_REQUESTS = {"gate": 4 * 3 * 9, "pilot": 40 * 3 * 4, "main": 25000 * 3 * 3}
# source files that may differ from the 2026-11-17 freeze: the row-order option of 2026-11-20 and the LLaVA-1.5 port of 2026-11-21
CHANGED_SINCE_IMPL_REF = ("mmimpress/llava15/serve.py", "mmimpress/llava15/store.py",
                          "mmimpress/packed_serve.py", "mmimpress/packed_store.py", "mmimpress/piggyback.py")
CHANGED_SINCE_OURS_REF = ("mmimpress/llava15/serve.py", "mmimpress/llava15/store.py")

H, V, QA, H1 = E.H, E.V, E.QA, E.H1
support = E.support
MODEL_ID, REVISION, SHORT_ANSWER = E.MODEL_ID, E.REVISION, E.SHORT_ANSWER
sha, canonical_hash, safe, atomic_json, append = H.sha, H.canonical_hash, H.safe, H.atomic_json, H.append
method_order, strict_score, assert_gpu_exclusive, conditioning_trace, PilotCounts = (
    H.method_order, H.strict_score, H.assert_gpu_exclusive, H.conditioning_trace, H.PilotCounts)
normal_request, file_hashes, Checks = H.normal_request, H.file_hashes, H.Checks
doc_counts, independent_counts = H.doc_counts, H.independent_counts
prompt_factory = H1.prompt_factory
turn1_request_with_stats, build_official_layout = E.turn1_request_with_stats, E.build_official_layout
ThreadedReadTrace, WriteTimer, compact_trace, stat_inventory, _chunks_of = (
    E.ThreadedReadTrace, E.WriteTimer, E.compact_trace, E.stat_inventory, E._chunks_of)

REPORT_SCRIPT = re.compile(r"^scripts/\d+_report_.*\.py$")


def source_hashes():
    paths = (list((ROOT / "mmimpress").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")) + list((ROOT / "tests").glob("test_*.py")))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths) if not REPORT_SCRIPT.match(str(p.relative_to(ROOT)))}


def freeze_violations(frozen):
    """Frozen source files that changed or disappeared.

    Files created after the freeze are not compared (nothing frozen imports
    them) and report scripts are not frozen (they only read committed rows):
    unrelated work in the repository must not stop a running experiment.
    """
    now = source_hashes()
    return sorted(k for k, v in frozen.items() if now.get(k) != v)


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


def scratch_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file() and not p.is_symlink())


def compact_json(path, value, replace=False):
    """JSON without indentation, fsynced (the manifest and the per-layer IMPRESS order are large)."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as f:
        json.dump(value, f, separators=(",", ":"), allow_nan=False, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    if path.exists() and not replace:
        tmp.unlink()
        raise FileExistsError(path)
    os.replace(tmp, path)


# ---------------------------------------------------------------- requests
def stored_hit(runner, servers, ctx, factory, method, image_id, store_path):
    """``stored_hit`` of scripts/196 (FullLoad, IMPRESS) and scripts/202 (Ours) with this
    run's arm table.  The three request calls are the calls of those scripts."""
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
            elif family == "impress_official":
                result = impress_official.request_official(
                    servers["base"], ctx, suffix, ratio=arm["ratio"], probe=OFFICIAL["probe_heads"],
                    sim_threshold=OFFICIAL["sim_threshold"], prefetch=OFFICIAL["prefetch_async"] if arm["async"] else "none",
                    cold=False, selector_out=selectors)
            else:      # Ours: the packed request; the store (written in original row order) decides where the rows go
                result = request_packed(servers["base"], ctx, suffix_ids=suffix, cold=False, image_id=image_id, directcopy=False,
                                        readahead="layer_stream" if arm["async"] else "none", max_ahead=READAHEAD["max_ahead"])
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
        result["importance_only_chunks"] = impress_official.alternative_chunk_counts(
            sel.keep_masks, sel.log, ctx.meta, lambda li: ctx.importance_only_position[li])
        result["official_params"] = {"ratio": float(arm["ratio"]), "probe": OFFICIAL["probe_heads"], "sim_threshold": OFFICIAL["sim_threshold"]}
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
    from mmimpress.serve import BIAS
    original = [l.self_attn.forward for l in runner.layers]
    hooks_before = [len(l._forward_pre_hooks) for l in runner.layers]
    with conditioning_trace() as conditioning:
        result, diag = stored_hit(runner, servers, ctx, factory, method, image_id, store_path)
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
def _native_persist(runner, result, diag, path, image_id, kind, source, docs, scores_path):
    """The writer calls of scripts/196 (raster, IMPRESS) and scripts/202 (packed,
    ``row_order="original"``); only the dataset identity written to the store metadata differs."""
    from mmimpress.piggyback import persist_captured_raster_prefix
    enc = diag["enc_cpu"]
    args = (runner, result["captured_past_key_values"], enc["input_ids"], enc["image_sizes"][0])
    common = dict(image_id=image_id, model_id=MODEL_ID, chunk_size=64, image_input_sha256=diag["image_input_sha256"],
                  extra_metadata={"dataset": DATASET, "source_dialog_id": source["dialog_id"], "source_turn_id": 1,
                                  "source_method_id": source["method_id"], "source_execution_id": source["physical_execution_id"],
                                  "source_logical_request_id": source["logical_request_id"], "run_schema": SCHEMA})
    if kind in FULL_KINDS:
        with WriteTimer() as timer:
            receipt = persist_captured_raster_prefix(*args, diag["hidden"].result_cpu(), path, probe_heads=SERVER_ARGS["probe_heads"],
                                                     hidden_capture_stats=diag["hidden"], full_integrity_hash=False, **common)
        receipt["write_by_file_name"] = timer.by_name
        if kind == NEW_KIND:
            receipt["official_turn1"] = build_official_layout(runner, path, diag, scores_path)
        return receipt
    assert kind == ORIG_KIND
    return persist_captured_packed_prefix(*args, diag["vision"].result_cpu(), path, layer_budget=docs[VECTOR],
                                          capture_stats=diag["vision"], full_integrity_hash=False, row_order=ROW_ORDER_ORIGINAL, **common)


def persist(runner, result, diag, path, image_id, kind, source, docs, scores_path):
    return support.persist_llava(lambda: _native_persist(runner, result, diag, path, image_id, kind, source, docs, scores_path), path, kind, source)


def activate(runner, kind, path):
    """``activate`` of scripts/196 (full stores) and scripts/202 (packed, original row order)."""
    started = time.perf_counter()
    if kind in FULL_KINDS:
        from mmimpress.serve import ImageContext
        ctx = ImageContext(path, runner.model.device, drop_cache=False, require_v_hidden=False)
        completion = support.verify_completion(path, "integrity.json")
        pinned = 0
        if kind == NEW_KIND:
            impress_official.validate_official_store(ctx.meta)
            # where each stored token would sit if every visual token, separators included, were
            # ordered by importance alone (a computed comparison of the report; nothing is read with it)
            scores = E.TURN1_ORDER_SCORES[str(Path(path))]
            ctx.importance_only_position = [
                impress_official.reverse_index(impress_official.descending_order(scores[li]))[
                    torch.tensor(ctx.meta["order"][li])].tolist() for li in range(ctx.meta["num_layers"])]
    else:
        ctx = PackedImageContext(path, runner.model.device)
        ctx.validate_packed_layout()
        assert ctx.row_order == ROW_ORDER_ORIGINAL
        completion = ctx.integrity_activation
        pinned = int(ctx.reader.total)
    ctx.activation = {"kind": kind, "host_tensor_bytes": sum(t.numel() * t.element_size() for t in ctx.cache.sys_kv.values()),
                      "pinned_host_buffer_bytes": pinned, "gpu_tensor_bytes": 0, "visual_payload_resident_bytes": 0,
                      "file_bytes": sum((Path(path) / n).stat().st_size for n in ("meta.json", "sys_kv.pt")),
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
    blocked = wait.get("read_blocked_ms")
    if hit and blocked is None and arm["family"] == "ours":
        blocked = result.get("packed_read_ms")          # synchronous packed read: the request thread is blocked for the whole read
    row = {"schema_version": SCHEMA, "experiment_id": run.name, "phase": phase, "attempt_id": attempt,
           "logical_request_id": logical, "request_id": logical, "physical_execution_id": physical_id,
           "model_id": MODEL_ID, "model_revision": REVISION, "method_id": method, "family": arm["family"], "async_read": bool(arm["async"]),
           "store_kind": arm["kind"], "row_order": ROW_ORDER_ORIGINAL if arm["kind"] == ORIG_KIND else None, "image_id": dialog["image_id"],
           "dialog_id": dialog["dialog_id"], "turn_id": turn, "question_id": str(question["question_id"]),
           "question": question["question"], "gold": question["answers"], "prediction": result["answer"],
           "prompt": diag["prompt"], "history": diag["history"], "history_entries": diag["history_entries"],
           "prompt_sha256": hashlib.sha256(diag["prompt"].encode()).hexdigest(), "suffix_input_ids": suffix,
           "suffix_ids_sha256": canonical_hash(suffix), "history_sha256": hashlib.sha256(diag["history"].encode()).hexdigest(),
           "config_sha256": config["_file_sha256"], "manifest_sha256": config["_manifest_sha256"],
           "code_sha256": canonical_hash(config["source_sha256"]), "phase_ordinal": ordinal,
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
    return enrich(row, question, runner.processor.tokenizer, "llava")       # official VQA score (+ annotation, native decode fields)


# ------------------------------------------------------------------- audit
def audit_image_rows(rows, manifest, config, phase, image_id, artifact=None):
    """The per-image audit of scripts/196 (FullLoad, IMPRESS) and scripts/202 (Ours, original
    row order), with the official VQA score in place of the strict EM."""
    methods = config["phases"][phase]["methods"]
    c = Checks()
    by_image = manifest["_by_image"] if "_by_image" in manifest else {d["image_id"]: d for d in manifest["dialogues"]}
    dialog = by_image.get(image_id)
    cache = manifest.setdefault("_phase_ordinals", {})
    if phase not in cache:
        cache[phase] = {did: i for i, did in enumerate(manifest["phase_dialogue_ids"][phase])}
    ordinals = cache[phase]
    c.require(dialog is not None and dialog["dialog_id"] in ordinals, "known_image", image_id)
    if dialog is None or dialog["dialog_id"] not in ordinals:
        return dict(c.result(), image_id=image_id, phase=phase, requests=len(rows), expected_requests=0)
    ds = [dialog]
    expected = {(d["dialog_id"], m, t): d["turns"][t - 1] for d in ds for m in methods for t in (1, 2, 3)}
    bykey = {(r["dialog_id"], r["method_id"], r["turn_id"]): r for r in rows}
    c.require(set(bykey) == set(expected) and len(rows) == len(expected), "image_completeness", image_id)
    c.require(len({r["logical_request_id"] for r in rows}) == len(rows), "unique_logical_ids", image_id)
    c.require(len({r["physical_execution_id"] for r in rows}) == len(rows), "unique_physical_ids", image_id)
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
                      and p["source"]["method_id"] in methods, "T1_build_same_kind", kind)
            a = json.loads((artifact / f"{kind}_activation.json").read_text())
            c.require(a["event_kind"] == "activation" and a["outer_wall_ms"] > 0 and a["kind"] == kind, "actual_activation_event", kind)
            n = meta["v_token_num"] - len(meta["newline_idx"]); width = meta["num_heads"] * meta["head_dim"] * 2
            vn, nl = meta["v_token_num"], meta["num_layers"]
            c.require(meta.get("dataset") == DATASET and width == ROW_BYTES and nl == NUM_LAYERS, "store_dataset_and_geometry", kind)
            if kind == ORIG_KIND:
                counts = doc_counts(doc, n)
                c.require(counts == independent_counts(doc["layer_keep_ratios"], n, doc["r_max"]) and min(counts) >= 1, "PK_counts_independent_nonzero", kind)
                c.require(meta["layer_budget_r_min"] == doc["r_min"] and meta["layer_budget_r_max"] == doc["r_max"]
                          and meta["layer_keep_ratios"] == doc["layer_keep_ratios"], "PK_clip_range_and_ratios_from_document", kind)
                c.require(meta["storage_policy"] == "prefix25_packed" and meta["store_format"] == ORIGINAL_STORE_FORMAT and meta["budget_policy"] == "layerwise_v1", "PK_policy", kind)
                c.require(meta["layer_keep_counts"] == counts == meta["payload_rows_per_layer"], "PK_counts_recomputed", kind)
                c.require(sum(counts) == nl * ((n + 3) // 4), "PK_budget_equals_25pct_total", kind)
                c.require(meta["layer_budget_sha256"] == doc["layer_budget_sha256"] == VECTOR_RATIO_SHA and meta["layer_budget_id"] == VECTOR, "PK_vector_identity", kind)
                sep = 2 * nl * len(meta["newline_idx"]) * width
                total = sum(2 * kk * width for kk in counts) + sep
                c.require(meta["packed_total_bytes"] == total == files[PACKED_NAME]["size"] and meta["packed_sep_bytes"] == sep, "PK_exact_file_length", kind)
                c.require(files[PACKED_NAME]["sha256"] == meta["packed_sha256"], "PK_packed_sha256_matches_inventory", kind)
                c.require(not any(f.startswith("layer_") or f == "sep_kv.bin" for f in files), "PK_no_unpacked_payload_files", kind)
                c.require(meta.get("layout_uses_dataset_question") is False and meta.get("llm_used_for_layout_scoring") is False
                          and meta.get("calibration_questions") == 0, "PK_layout_takes_no_question_llm_or_calibration", kind)
                structural = set(meta["newline_idx"])
                idx = meta.get("stored_original_indices", []); thr = meta.get("stored_min_keep_count", [])
                per_layer = layer_original_indices(meta) if len(idx) == len(thr) else []
                c.require(meta.get("row_order") == ROW_ORDER_ORIGINAL and meta.get("reordered") is False
                          and not any(key in meta for key in IMPORTANCE_ORDER_KEYS) and "visionzip_layout.pt" not in files,
                          "ORIG_no_importance_permutation_in_store", kind)
                c.require(len(idx) == len(thr) == max(counts) and idx == sorted(set(idx)) and not structural & set(idx)
                          and all(0 <= i < vn for i in idx) and [len(x) for x in per_layer] == counts, "ORIG_index_table_valid", kind)
            else:
                c.require(meta.get("storage_policy", "full") == "full" and meta["probe_heads"] == SERVER_ARGS["probe_heads"] and meta["chunk_size"] == 64, "full_store_with_probe_sidecar", kind)
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
                        seps = sorted(meta["newline_idx"]); sepset = set(seps)
                        for li in range(nl if good else 0):
                            order = meta["order"][li]
                            expect = seps + [i for i in np.argsort(-sc[li], kind="stable").tolist() if i not in sepset]
                            good = good and order == expect and meta["newline_stored"][li] == list(range(len(seps)))
                            stored = np.ascontiguousarray(pr[li][:, order].astype("<f4")).tobytes()
                            good = good and hashlib.sha256(stored).hexdigest() == t1["trailer_sha256_per_layer"][li]
                        c.require(good, "official_order_and_stored_prefix_rows_recomputed", kind)
        c.require(len(set(builds)) == len(store_kinds(methods)), "distinct_physical_build_events", image_id)
    packed_sets = {}
    for key, r in bykey.items():
        rid = r["request_id"]; did, m, t = key; result = r["result"]; question = expected.get(key)
        if question is None:
            continue
        arm = ARMS[m]; family = arm["family"]
        c.require(r["schema_version"] == SCHEMA and r["model_id"] == MODEL_ID and r["model_revision"] == REVISION and r.get("dataset") == DATASET, "schema_model", rid)
        c.require(rid == r["logical_request_id"] == f"{SCHEMA}:{r['experiment_id']}:{phase}:{MODEL_ID}:{m}:{did}:t{t}", "logical_identity", rid)
        c.require(r["phase"] == phase and r["image_id"] == image_id and r["family"] == family and r["async_read"] == arm["async"]
                  and r["store_kind"] == arm["kind"], "phase_image_arm", rid)
        c.require(r["question_id"] == str(question["question_id"]) and r["question"] == question["question"] and r["gold"] == question["answers"], "frozen_question_gold", rid)
        # official VQA score, recomputed from the frozen annotation by two independent routes
        ann = question["annotation"]
        c.require(r.get("annotation") == ann and r.get("scorer_answer") == r["prediction"], "official_annotation", rid)
        c.require(abs(independent_score(r["prediction"], ann)["vqa_score"] - r["score"]) <= 1e-10, "official_vqa_score_recalculation", rid)
        c.require(abs(score_from_strings(r["prediction"], [a["answer"] for a in ann["answers"]]) - r["score"]) <= 1e-10, "official_vqa_score_from_strings", rid)
        c.require(r.get("generated_token_ids") == result["generated_token_ids"], "generated_ids_copied", rid)
        c.require(r["prediction"] == result["answer"], "raw_decoded_prediction", rid)
        c.require(r["config_sha256"] == config["_file_sha256"] and r["manifest_sha256"] == config["_manifest_sha256"], "config_manifest_hash", rid)
        c.require(r["code_sha256"] == canonical_hash(config["source_sha256"]), "code_hash", rid)
        ordinal = ordinals[did]; order = method_order(methods, ordinal)
        c.require(r["phase_ordinal"] == ordinal and r["method_order"] == order and r["method_order_position"] == order.index(m), "deterministic_rotation", rid)
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
            c.require(src["source"]["dialog_id"] == ds[0]["dialog_id"] and kind_of(src["source"]["method_id"]) == arm["kind"], "T1_source_same_kind", rid)
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
            c.require(result.get("row_order") == ROW_ORDER_ORIGINAL == r["row_order"], "PK_row_order_of_the_arm", rid)
            sep = 2 * nl * len(structural) * width
            body_bytes = sum(2 * kk * width for kk in counts); total = body_bytes + sep
            c.require(r["ssd_read_bytes"] == total and r["pread_calls"] == 0, "PK_bytes_equal_file_no_pread", rid)
            c.require(all(e[0] == "preadv" and e[1] == PACKED_NAME for e in trace), "PK_preadv_on_packed_file_only", rid)
            if arm["async"]:
                plan = [(body_bytes, sep)]; off = 0
                for kk in counts:
                    plan.append((off, 2 * kk * width)); off += 2 * kk * width
                plan = [x for x in plan if x[1]]
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
            c.require(result.get("selected_stored_ids") is None and original == sorted(original), "ORIG_rows_addressed_by_original_index", rid)
            c.require(result.get("keep_count_per_layer") == [kk + len(structural) for kk in counts], "structural_keep", rid)
            c.require(result.get("store_format") == ORIGINAL_STORE_FORMAT and result.get("directcopy") is False, "PK_scatter_path", rid)
            packed_sets.setdefault(m, []).append((original, counts, total))
            c.require(result.get("query_score_calls", 0) == 0, "online_scoring_zero", rid)
        else:      # impress_official
            ratio = arm["ratio"]; thr = OFFICIAL["sim_threshold"]
            k_keep = max(1, min(n, math.ceil(ratio * n))); nc = (vn + 63) // 64
            c.require(result.get("k_keep") == k_keep and result.get("threshold") == thr and result.get("selection_path") == "impress_official"
                      and result.get("official_params") == {"ratio": ratio, "probe": OFFICIAL["probe_heads"], "sim_threshold": thr}, "official_parameters", rid)
            c.require(result.get("n_score_rows") == len(r["suffix_input_ids"]), "official_suffix_rows_are_whole_suffix", rid)
            log = result.get("layer_log", [])
            c.require([x["layer"] for x in log] == list(range(nl)), "official_layer_log_complete", rid)
            probe_bytes = vn * OFFICIAL["probe_heads"] * g["head_dim"] * 2 + OFFICIAL["probe_heads"] * vn * 4
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
                    got[name] = sorted({ch for e in es for ch in _chunks_of(e[2], e[4], width, vn)})
                    b = sum(e[4] for e in es)
                    kbytes += b if name == "k" else 0; vbytes += b if name == "v" else 0
                    ok_layers = ok_layers and b == sum(min(64, vn - ch * 64) for ch in got[name]) * width
                    if x["mode"] == "fallback":      # the whole layer, one read
                        ok_layers = ok_layers and len(es) == 1 and es[0][2] == 0 and es[0][4] == vn * width
                ok_layers = ok_layers and got["k"] == got["v"] and len(got["k"]) == x["k_chunks"] == x["v_chunks"]
                if x["mode"] == "consensus":         # the separators sit in the first chunk and are always read
                    ok_layers = ok_layers and x["sim"] >= thr and len(got["k"]) >= (k_keep + 63) // 64 and (not structural or 0 in got["k"])
                else:
                    ok_layers = ok_layers and x["mode"] == "fallback" and x["sim"] < thr and len(got["k"]) == nc
            c.require(ok_layers, "official_kv_reads_match_layer_decisions", rid)
            c.require(kbytes == r["k_read_bytes"] and vbytes == r["v_read_bytes"], "official_kv_bytes_from_trace", rid)
            fb = sum(x["mode"] == "fallback" for x in log)
            c.require(math.isclose(result.get("fallback_rate", -1), fb / nl), "official_fallback_rate_recomputed", rid)
            diag = result.get("layer_diagnostics", [])
            c.require(len(diag) == nl and all((d["order_overlap_selected"] is None) == (x["mode"] == "fallback") for d, x in zip(diag, log)),
                      "official_layer_diagnostics_present", rid)
            c.require(len(result.get("raster_chunks", {}).get("chunks", [])) == nl and len(result.get("importance_only_chunks", {}).get("chunks", [])) == nl,
                      "official_computed_chunk_comparisons_present", rid)
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
            c.require(sets[0][0] == meta["stored_original_indices"] and sets[0][2] == meta["packed_total_bytes"], "meta_to_actual_selection_and_bytes", f"{image_id}/{m}")
    # recorded, not a commit condition
    t1 = {tuple(r["result"]["generated_token_ids"]) for (did, m, t), r in bykey.items() if t == 1}
    return dict(c.result(), image_id=image_id, phase=phase, requests=len(rows), expected_requests=len(expected), dialogs=len(ds),
                t1_tokens_identical_across_arms=len(t1) == 1)


# ---------------------------------------------------------------- lifecycle
def completed_images(run, phase, config):
    """{image id: commit} of the phase.  The raw journal of every committed image is
    re-verified against its SHA256 when the report reads it."""
    done = {}
    root = run / phase / "images"
    if not root.is_dir():
        return done
    for folder in sorted(root.iterdir()):
        cp = folder / "COMMITTED.json"
        if not cp.exists():
            continue
        doc = json.loads(cp.read_text())
        if doc["config_sha256"] != config["_file_sha256"] or doc["manifest_sha256"] != config["_manifest_sha256"]:
            raise RuntimeError("resume config/manifest mismatch: " + folder.name)
        done[folder.name] = doc
    return done


def failed_images(run, phase):
    path = run / f"{phase}_failed_images.jsonl"
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def remove_scratch(run, scratch):
    scratch = Path(scratch)
    if not scratch.exists():
        return 0
    base = (run / "scratch").resolve()
    if scratch.is_symlink() or not scratch.resolve().is_relative_to(base) or scratch.resolve() == base:
        raise ValueError("refusing to remove a path outside this run's scratch")
    n = scratch_bytes(scratch)
    shutil.rmtree(scratch)
    return n


def store_signature(path):
    """{relative file: [size, mtime_ns, inode]} of a store: any write through the file system changes it."""
    out = {}
    for p in sorted(Path(path).rglob("*")):
        if p.is_symlink():
            raise ValueError("symlink in a store")
        if p.is_file():
            st = p.stat()
            if st.st_nlink != 1:
                raise ValueError("hardlink in a store")
            out[p.relative_to(path).as_posix()] = [st.st_size, st.st_mtime_ns, st.st_ino]
    return out


def cleanup_scratch(run, scratch, signature, audit, commit):
    """Delete the stores of a committed image: exactly the files recorded for it, nothing else."""
    scratch = Path(scratch)
    if audit.get("status") != "PASS" or commit.get("status") != "COMMITTED":
        raise ValueError("independent audit and atomic image commit required")
    if scratch.is_symlink() or not scratch.is_dir():
        raise ValueError("unsafe scratch root")
    resolved = scratch.resolve()
    if not resolved.is_relative_to((run / "scratch").resolve()):
        raise ValueError("outside run scratch")
    if store_signature(scratch) != signature:
        raise ValueError("cleanup allowlist mismatch")
    total = 0
    for relative, (size, _, _) in signature.items():
        (scratch / relative).unlink(); total += size
    for p in sorted((p for p in scratch.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        p.rmdir()
    scratch.rmdir()
    if not any(scratch.parent.iterdir()):      # the now-empty per-image scratch directory
        scratch.parent.rmdir()
    return {"removed_files": len(signature), "removed_bytes": total}


def slim_evidence(artifact, phase, audit):
    """Main phase only, after the audit passed: the two large evidence files of the IMPRESS
    store (25,000 images would hold ~45 GB of them).  The turn-1 statistics the audit just
    re-derived the stored order from are removed (their SHA256 stays in the persistence
    record); the store metadata copies are kept as gzip."""
    if phase != "main":
        return {"policy": "kept as written"}
    out = {"policy": "main phase: IMPRESS turn-1 statistics removed after the audit verified them; store metadata copies gzip-compressed",
           "removed": [], "compressed": []}
    for p in sorted(artifact.glob("*_turn1_stats.npz")):
        out["removed"].append({"file": p.name, "sha256": sha(p), "bytes": p.stat().st_size, "verified_by_audit": audit["status"] == "PASS"})
        p.unlink()
    for p in sorted(artifact.glob("*_meta.json")):
        data = p.read_bytes()
        target = p.with_suffix(".json.gz")
        with target.open("wb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0, compresslevel=6) as z:
            z.write(data)
        assert gzip.decompress(target.read_bytes()) == data
        out["compressed"].append({"file": p.name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "gzip_bytes": target.stat().st_size})
        p.unlink()
    return out


def run_image(runner, servers, dialog, ordinal, phase, run, manifest, config, t1_reference):
    """One image, one dialogue, three turns, every arm of the phase."""
    methods = config["phases"][phase]["methods"]; docs = config["layer_budgets"]
    image_id = dialog["image_id"]; parent = run / phase / "images" / image_id
    parent.mkdir(parents=True, exist_ok=True)
    if (parent / "COMMITTED.json").exists():
        raise FileExistsError("image already committed")
    # an earlier attempt of this image was interrupted (or failed): nothing of it is reused
    previous = sorted(p for p in parent.glob("attempt_*") if p.is_dir())
    attempt = f"attempt_{len(previous) + 1:04d}"
    restart = None
    if previous:
        removed = remove_scratch(run, run / "scratch" / phase / image_id)
        last = previous[-1]
        rows_found = sum(1 for _ in (last / "raw.jsonl").open()) if (last / "raw.jsonl").exists() else 0
        restart = {"at_unix": time.time(), "at_utc": STATUS.utc(time.time()), "phase": phase, "image_id": image_id, "mode": "image_restarted_from_turn1",
                   "previous_attempt": last.name, "new_attempt": attempt, "rows_discarded": rows_found, "discarded_scratch_bytes": removed,
                   "process_id": os.getpid(), "boot_id": STATUS.boot_id()}
        if not (last / "ABANDONED.json").exists():
            atomic_json(last / "ABANDONED.json", dict(restart, note="rows of this attempt are not part of any result"))
        append(run / "resume_log.jsonl", restart)
    artifact = parent / attempt; artifact.mkdir(exist_ok=False)
    scratch = run / "scratch" / phase / image_id / attempt
    atomic_json(artifact / "capacity_before.json", capacity_guard(run))
    scratch.mkdir(parents=True, exist_ok=False)
    kinds = {m: kind_of(m) for m in stored_methods(methods)}
    all_kinds = store_kinds(methods)
    ref_kind = all_kinds[0]
    paths = {k: scratch / k for k in all_kinds}
    metas, hashes, persistence, contexts, store_info, signatures = {}, {}, {}, {}, {}, {}
    rows = []; pending = []; success = False; capturing = set(); peak_scratch = 0
    source_image_sha = sha(ROOT / dialog["image_path"])
    atomic_json(artifact / "image_identity.json", {"image_id": image_id, "image_path": str((ROOT / dialog["image_path"]).resolve()), "sha256": source_image_sha})
    started_unix = time.time()
    try:
        assert_gpu_exclusive()
        generated = {m: {} for m in methods}
        for turn in (1, 2, 3):
            for method in method_order(methods, ordinal):
                physical = str(uuid.uuid4())
                logical = request_id(run.name, phase, method, dialog["dialog_id"], turn)
                source = {"dialog_id": dialog["dialog_id"], "method_id": method, "physical_execution_id": physical, "logical_request_id": logical}
                append(artifact / "attempts.jsonl", dict(source, turn_id=turn, status="STARTED", at_unix=time.time()))
                factory = lambda m=method: prompt_factory(methods, dialog, turn, m, generated[m])
                kind = kinds.get(method, "none")
                # the first arm of a store kind in the rotation builds that kind's store from its own turn 1
                capture = kind if turn == 1 and kind != "none" and kind not in capturing else "none"
                if capture != "none":
                    capturing.add(capture)
                if turn == 1 or method == RECOMP:
                    if capture == NEW_KIND:      # the same pixel request, with the attention capture attached
                        result, diag = turn1_request_with_stats(runner, servers["base"], ROOT / dialog["image_path"], factory)
                    else:
                        result, diag = normal_request(runner, servers["base"], ROOT / dialog["image_path"], factory,
                                                      "raster" if capture == "raster" else ("image_only" if capture != "none" else "none"))
                    append(artifact / "execution_events.jsonl", dict(source, turn_id=turn, status="EXECUTED", prediction=result["answer"],
                                                                    generated_token_ids=result["generated_token_ids"], ttft_ms=result["end_to_end_ttft_ms"],
                                                                    request_e2e_ms=result["request_e2e_ms"]))
                    if capture != "none":
                        start = time.perf_counter()
                        p = persist(runner, result, diag, paths[capture], image_id, capture, source, docs, artifact / f"{capture}_turn1_stats.npz")
                        outer_ms = (time.perf_counter() - start) * 1000
                        persistence[capture] = dict(p, outer_helper_wall_ms=outer_ms, source=source,
                                                    capture_accounting="capture and exit materialization already within T1 request E2E; not added again")
                        persistence[capture].pop("meta", None)
                        metas[capture] = json.loads((paths[capture] / "meta.json").read_text())
                        # the completion scan of persist_llava hashed every payload file once; that inventory is the store's identity
                        hashes[capture] = {n: f["sha256"] for n, f in p["files"].items() if n != "integrity.json"}
                        signatures[capture] = store_signature(paths[capture])
                        store_info[capture] = {"kind": capture, "path": str(paths[capture]), "meta_sha256": p["files"]["meta.json"]["sha256"],
                                               "source": source, "physical_build_event_id": p["event_id"]}
                        atomic_json(artifact / f"{capture}_persistence.json", persistence[capture])
                        compact_json(artifact / f"{capture}_meta.json", metas[capture])
                        atomic_json(artifact / f"{capture}_hashes.json", hashes[capture])
                        peak_scratch = max(peak_scratch, scratch_bytes(scratch))
                    result.pop("captured_past_key_values", None)
                    saved = {a: b for a, b in diag.items() if a not in ("enc_cpu", "hidden", "vision", "turn1_stats")}
                    if turn == 1:
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
            if turn == 1:
                assert set(metas) == set(all_kinds)
                for method, result, diag, physical in pending:
                    meta_kind = kinds.get(method, ref_kind)
                    row = make_row(runner, result, diag, metas[meta_kind], methods=methods, method=method, dialog=dialog, turn=turn, phase=phase, run=run,
                                   attempt=attempt, physical_id=physical, ordinal=ordinal, store_info=None, config=config)
                    row["image_sha256"] = source_image_sha
                    append(artifact / "raw.jsonl", row); rows.append(row); generated[method][1] = row
                    append(artifact / "attempts.jsonl", {"physical_execution_id": physical, "logical_request_id": row["logical_request_id"], "status": "COMPLETED", "at_unix": time.time()})
                pending.clear()
                for kind in metas:
                    assert metas[kind]["prefix_input_ids"] == metas[ref_kind]["prefix_input_ids"]
                    assert metas[kind]["image_input_sha256"] == metas[ref_kind]["image_input_sha256"]
        assert_gpu_exclusive()
        for ctx in contexts.values():      # file descriptors closed before the stores are checked and removed
            ctx.close()
        contexts.clear()
        peak_scratch = max(peak_scratch, scratch_bytes(scratch))
        # the stores did not change while they were served: size / mtime / inode of every file as recorded right after the build;
        # outside the main phase every file is additionally re-hashed
        for kind in all_kinds:
            assert store_signature(paths[kind]) == signatures[kind], "store changed during requests"
            if phase != "main":
                got = file_hashes(paths[kind]); got.pop("integrity.json", None)
                assert got == hashes[kind], "store content changed during requests"
        signature = store_signature(scratch)
        atomic_json(artifact / "store_integrity_after.json", {"status": "PASS", "check": "size, mtime_ns and inode of every store file unchanged since the build"
                                                                                       + ("" if phase == "main" else "; full SHA256 re-hash equal"),
                                                              "file_sha256": hashes})
        atomic_json(artifact / "disk_usage.json", {"peak_scratch_bytes": peak_scratch, "free_bytes_after_requests": shutil.disk_usage(run).free,
                                                   "store_bytes": {k: scratch_bytes(paths[k]) for k in all_kinds}})
        # continuity with the v3 record: every arm's turn-1 tokens against the v3 turn-1 tokens of this image (recorded, never a commit condition)
        reference = t1_reference.get(image_id)
        t1_equal = {r["method_id"]: (reference is not None and r["generated_token_ids"] == reference) for r in rows if r["turn_id"] == 1}
        atomic_json(artifact / "t1_v3.json", {"image_id": image_id, "v3_turn1_tokens": reference, "equal": t1_equal,
                                              "tokens": {r["method_id"]: r["generated_token_ids"] for r in rows if r["turn_id"] == 1}})
        audit = audit_image_rows(read_rows(artifact / "raw.jsonl"), manifest, config, phase, image_id, artifact)
        audit["raw_sha256"] = sha(artifact / "raw.jsonl"); audit["evidence"] = slim_evidence(artifact, phase, audit) if audit["status"] == "PASS" else None
        atomic_json(artifact / "independent_audit.json", audit)
        if audit["status"] != "PASS":
            raise RuntimeError("INDEPENDENT_IMAGE_AUDIT_FAIL: " + repr(audit["failures"][:10]))
        atomic_json(artifact / "cleanup_allowlist.json", {"scratch": str(scratch), "files": signature, "ownership_run": run.name})
        commit = {"status": "COMMITTED", "image_id": image_id, "attempt_id": attempt, "requests": len(rows),
                  "raw_sha256": audit["raw_sha256"], "audit_sha256": sha(artifact / "independent_audit.json"),
                  "config_sha256": config["_file_sha256"], "manifest_sha256": config["_manifest_sha256"],
                  "t1_tokens_identical_across_arms": audit["t1_tokens_identical_across_arms"], "t1_equal_v3": t1_equal,
                  "attempts_before": len(previous), "image_seconds": time.time() - started_unix, "committed_at_unix": time.time(),
                  "process_id": os.getpid()}
        commit["evidence_sha256"] = {p.name: sha(p) for p in artifact.iterdir() if p.is_file()}
        # journals -> lossless gzip (decompression SHA checked), then the atomic commit
        archive_before_commit(parent, commit, lambda path, value: atomic_json(path, value)); success = True
    except BaseException as exc:
        atomic_json(artifact / "failure.json", {"status": "PARTIAL", "error": repr(exc), "traceback": traceback.format_exc(),
                                                "completed_uncommitted_rows": len(rows), "final_rows_from_this_attempt": 0}, replace=True)
        raise
    finally:
        for ctx in contexts.values():
            ctx.close()
        contexts.clear()
        for kind in all_kinds:      # per-image state of the imported harness
            E.TURN1_ORDER_SCORES.pop(str(paths[kind]), None)
        pending.clear(); gc.collect()
        if success:
            receipt = cleanup_scratch(run, scratch, signature, audit, commit)
            remove_archived_duplicates(parent)
            atomic_json(artifact / "cleanup_receipt.json", dict(receipt, status="PASS"))
    return {"requests": len(rows), "t1_equal_v3": t1_equal, "store_bytes": sum(scratch_bytes_recorded(persistence).values()),
            "write_bytes": sum(int(p.get("os_write_bytes", 0)) for p in persistence.values()), "restart": restart}


def scratch_bytes_recorded(persistence):
    return {k: int(p["file_logical_bytes"]) for k, p in persistence.items()}


def capacity_guard(run):
    """Free space must stay above 20 x the largest per-image store bundle (and the fixed reserve)."""
    plan = json.loads((run / "storage_plan.json").read_text())
    free = shutil.disk_usage(run).free
    need = max(plan["required_free_bytes"], plan["minimum_reserve_bytes"])
    if free < need:
        raise RuntimeError(f"BLOCKED_STORAGE: free={free}; required={need}")
    scratch = run / "scratch"
    owned = scratch_bytes(scratch) if scratch.exists() else 0
    if owned > plan["largest_image_bundle_bytes"]:
        raise RuntimeError("BLOCKED_STORAGE: store files of an earlier image are still present")
    return {"free_bytes": free, "required_free_bytes": need, "retained_scratch_bytes": owned}


def verify_freeze(run):
    import transformers
    config = json.loads((run / "config.json").read_text())
    changed = freeze_violations(config["source_sha256"])
    if changed:
        # development rehearsals only (run name contains "_dev_"; never a measurement): code may still change
        if not ("_dev_" in run.name and os.environ.get("LNEXT_MAIN_DEV_ALLOW_SOURCE_DRIFT") == "1"):
            raise RuntimeError("FROZEN_SOURCE_CONFIG_MISMATCH: " + repr(changed[:10]))
        config["source_sha256"] = source_hashes()
    if sha(CONTRACT) != config["contract_sha256"]:
        raise RuntimeError("FROZEN_CONTRACT_MISMATCH")
    if sha(run / "manifest.json") != config["manifest_sha256"] or sha(run / "storage_plan.json") != config["storage_plan_sha256"]:
        raise RuntimeError("FROZEN_MANIFEST_OR_STORAGE_PLAN_MISMATCH")
    manifest = json.loads((run / "manifest.json").read_text())
    target = run / "vectors" / VECTOR_SOURCE.name
    doc = load_budget_document(target)
    frozen = config["layer_budgets"][VECTOR]
    if (sha(target) != VECTOR_FILE_SHA or doc["layer_budget_sha256"] != VECTOR_RATIO_SHA or doc["layer_keep_ratios"] != frozen["layer_keep_ratios"]
            or frozen["layer_budget_sha256"] != VECTOR_RATIO_SHA):
        raise RuntimeError("FROZEN_VECTOR_MISMATCH")
    if config["model"] != MODEL_ID or config["revision"] != REVISION or config["phase_methods"] != PHASE_METHODS:
        raise RuntimeError("FROZEN_ARMS_OR_MODEL_MISMATCH")
    if transformers.__version__ != config["environment"]["transformers"] or torch.__version__ != config["environment"]["torch"]:
        raise RuntimeError("FROZEN_ENVIRONMENT_MISMATCH")
    config.update(_file_sha256=sha(run / "config.json"), _manifest_sha256=sha(run / "manifest.json"))
    manifest["_by_image"] = {d["image_id"]: d for d in manifest["dialogues"]}
    return config, manifest


def committed_request_counts(run):
    out = {}
    for ph in PHASES:
        root = run / ph / "images"
        out[ph] = sum(json.loads(cp.read_text())["requests"] for cp in root.glob("*/COMMITTED.json")) if root.is_dir() else 0
    return out


def wait_for_gpu(run, phase):
    """Another process on the GPU would contaminate the timings: wait until it leaves (it is never terminated)."""
    waited = 0
    while True:
        try:
            assert_gpu_exclusive()
            return waited
        except RuntimeError as exc:
            if waited >= GPU_WAIT_SECONDS:
                raise
            if waited == 0:
                STATUS.update(run, stage=f"{phase}_waiting_for_gpu", last_error={"stage": f"{phase}_phase", "error": repr(exc), "at_utc": STATUS.utc(time.time()),
                                                                                "note": "waiting for the other process to leave the GPU"})
            time.sleep(30); waited += 30


def run_phase(runner, servers, run, phase, config, manifest):
    done = completed_images(run, phase, config)
    ids = manifest["phase_dialogue_ids"][phase]
    by_id = {d["dialog_id"]: d for d in manifest["dialogues"]}
    dialogs = [by_id[did] for did in ids]                # phase order: the frozen list (main = manifest order)
    per_image = 3 * len(config["phases"][phase]["methods"])
    expected = len(ids) * per_image
    assert expected == manifest["expected_requests"][phase]
    t1_reference = load_t1_reference(run)
    failed = {f["image_id"]: f for f in failed_images(run, phase)}
    started = time.time(); new_images = 0; consecutive_failures = 0
    counts = committed_request_counts(run)
    mismatch = sum(1 for c in done.values() if not all(c.get("t1_equal_v3", {}).values()))

    def status(image=None):
        now = time.time()
        rate = (now - started) / new_images if new_images else None
        left = len(dialogs) - len(done) - len(failed)
        later = [p for p in PHASES[PHASES.index(phase) + 1:]]
        later_images = sum(len(manifest["phase_dialogue_ids"][p]) for p in later)
        ssd = STATUS.ssd_counter(run)
        temp = STATUS.ssd_temperature()
        fields = dict(stage=f"{phase}_phase", phase=phase, phase_images_done=len(done), phase_images_total=len(dialogs),
                      phase_images_failed=len(failed), phase_requests_done=sum(c["requests"] for c in done.values()), phase_requests_total=expected,
                      current_image=image, seconds_per_image_this_process=rate,
                      phase_eta_utc=STATUS.utc(now + rate * left) if rate else None,
                      t1_tokens_differ_from_v3_images=mismatch,
                      ssd_written_bytes=ssd.get("written_bytes"), ssd_written_gb=round((ssd.get("written_bytes") or 0) / 1e9, 2),
                      ssd_written_note="device-wide kernel counter since the driver first started (all processes); not NAND writes",
                      ssd_temperature_c=temp["composite_c"], free_gib=round(shutil.disk_usage(run).free / GiB, 1), python_pid=os.getpid())
        if phase == "main":
            fields.update(images_done=len(done), images_total=len(dialogs), eta_utc=fields["phase_eta_utc"],
                          eta_note="rate of this process applied to the remaining main images; the report (minutes) is not included")
        elif rate:
            fields.update(eta_utc=None, eta_note=f"the main phase has not started; {later_images} images remain after this phase "
                                                 "(the estimate is written when the pilot has finished)")
        STATUS.update(run, **fields)
        return ssd, temp

    status()
    STATUS.ssd_sample(run, f"{phase}_process_start", phase=phase, committed_images=len(done))
    leftovers = run / "scratch" / phase
    if leftovers.is_dir():      # stores of an image that was committed but not cleaned up, or of an interrupted image that will be redone
        for folder in sorted(leftovers.iterdir()):
            if folder.name in done:
                append(run / "resume_log.jsonl", {"at_unix": time.time(), "at_utc": STATUS.utc(time.time()), "phase": phase, "image_id": folder.name,
                                                  "mode": "committed_scratch_removed", "discarded_scratch_bytes": remove_scratch(run, folder),
                                                  "process_id": os.getpid(), "boot_id": STATUS.boot_id()})
    index = 0
    while index < len(dialogs):
        dialog = dialogs[index]; image = dialog["image_id"]
        if image in done or image in failed:
            index += 1
            continue
        if (run / "STOP_REQUESTED").exists():
            raise RuntimeError("STOP_REQUESTED")
        t0 = time.time()
        try:
            out = run_image(runner, servers, dialog, index, phase, run, manifest, config, t1_reference)
        except Exception as exc:      # noqa: BLE001
            text = repr(exc)
            if "GPU_TIMING_CONTAMINATION" in text:      # not the image's fault: wait, then the image restarts at turn 1
                append(run / "resume_log.jsonl", {"at_unix": time.time(), "at_utc": STATUS.utc(time.time()), "phase": phase, "image_id": image,
                                                  "mode": "gpu_contamination_image_will_restart", "error": text, "process_id": os.getpid()})
                torch.cuda.empty_cache()
                wait_for_gpu(run, phase)
                status(image)
                continue
            if "BLOCKED_STORAGE" in text or "STOP_REQUESTED" in text or phase != "main":
                raise
            # main phase: one image may fail (audit or execution); it is retried once from turn 1, then recorded and skipped
            attempts = len([p for p in (run / phase / "images" / image).glob("attempt_*") if p.is_dir()])
            consecutive_failures += 1
            event = {"at_unix": time.time(), "at_utc": STATUS.utc(time.time()), "phase": phase, "image_id": image, "index": index,
                     "attempts": attempts, "error": text[:2000], "process_id": os.getpid()}
            append(run / f"{phase}_image_errors.jsonl", event)
            STATUS.update(run, last_error={"stage": f"{phase}_phase", "image_id": image, "error": text[:500], "at_utc": STATUS.utc(time.time())})
            torch.cuda.empty_cache(); gc.collect()
            if attempts < 2:
                continue
            failed[image] = event
            append(run / f"{phase}_failed_images.jsonl", event)
            remove_scratch(run, run / "scratch" / phase / image)
            if len(failed) > MAIN_FAILED_IMAGES_MAX or consecutive_failures >= 2 * MAIN_FAILED_CONSECUTIVE_MAX:
                raise RuntimeError(f"TOO_MANY_FAILED_IMAGES: {len(failed)} failed, {consecutive_failures} consecutive errors; last: {text[:300]}")
            index += 1
            continue
        consecutive_failures = 0; new_images += 1
        done[image] = {"requests": out["requests"], "t1_equal_v3": out["t1_equal_v3"]}
        if not all(out["t1_equal_v3"].values()):
            mismatch += 1
            append(run / f"{phase}_t1_v3_mismatch.jsonl", {"image_id": image, "index": index, "equal": out["t1_equal_v3"], "at_unix": time.time()})
        ssd, temp = status(image)
        STATUS.ssd_sample(run, "image_committed", phase=phase, image_id=image, index=index, committed_images=len(done),
                          image_seconds=time.time() - t0, store_bytes=out["store_bytes"], store_write_bytes=out["write_bytes"])
        if len(done) % 100 == 0 or len(done) == len(dialogs) or phase != "main":
            print(json.dumps({"phase": phase, "images": len(done), "of": len(dialogs), "image_id": image, "image_seconds": round(time.time() - t0, 2),
                              "ssd_written_gb": round((ssd.get("written_bytes") or 0) / 1e9, 1), "ssd_temperature_c": temp["composite_c"],
                              "failed": len(failed), "t1_differs_from_v3": mismatch}), flush=True)
        if len(done) % 500 == 0:
            append(run / "environment_log.jsonl", dict(environment_sample(), phase=phase, completed_images=len(done), free_bytes=shutil.disk_usage(run).free))
        index += 1
    completed = sum(c["requests"] for c in done.values())
    assert len(done) + len(failed) == len(dialogs) and completed == len(done) * per_image
    if phase != "main":
        assert not failed and completed == expected, (completed, expected)
    receipt = {"status": "PASS", "requests": completed, "expected_requests": expected, "images": len(done), "images_expected": len(dialogs),
               "failed_images": sorted(failed), "independent_image_audit": "PASS (every committed image)", "config_sha256": config["_file_sha256"]}
    atomic_json(run / phase / "validation.json", receipt, replace=True)
    return receipt


def environment_sample():
    out = {"at_unix": time.time(), "ssd_temperature": STATUS.ssd_temperature()}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           check=True, capture_output=True, text=True).stdout.strip()
        out["gpu_temperature_c"], out["gpu_utilization_pct"], out["gpu_memory_used_mib"] = [float(x) for x in q.split(",")]
    except Exception as exc:      # noqa: BLE001  (monitoring only)
        out["nvidia_smi_error"] = repr(exc)
    return out


# ---------------------------------------------------------- v3 reference
def reference_dir(run):
    return run / "reference"


def load_t1_reference(run):
    """image id -> the v3 record's turn-1 generated tokens (the v3 ReComp row)."""
    return json.loads((reference_dir(run) / "v3_turn1_tokens.json").read_text())


def build_reference(run, images_by_phase):
    """Compact index of the v3 per-request record (read only) and its completeness for ReComp.

    Every committed v3 image is opened and its journal checked against the SHA256 of its
    commit (``mmimpress.mtvqa25k_packed.reference``).  Required of the ReComp rows of the v3
    main phase, for every one of the 25,000 images and each of the three turns: generated
    tokens (non-empty), an official VQA score in [0, 1] and a positive TTFT.
    """
    from mmimpress.mtvqa25k_packed import reference as REF
    out_dir = reference_dir(run); out_dir.mkdir(exist_ok=True)
    summary = {}
    for phase, v3_phase in V3_PHASE.items():
        images = images_by_phase[v3_phase]
        path = out_dir / f"v3_{v3_phase}_index.jsonl.gz"
        s = REF.build_index(V3_LLAVA / v3_phase / v3_phase / "images", images, path, full=False)
        s["path"] = str(path.relative_to(ROOT)); s["file_sha256"] = sha(path)
        summary[v3_phase] = s
    index = REF.load_index(out_dir / "v3_main_index.jsonl.gz")
    missing = []; cross_arm = 0; t1 = {}; geometry = {}
    for image in images_by_phase["main"]:
        rows = index.get(image)
        if rows is None:
            missing.append({"image_id": image, "reason": "image not readable in the v3 record"}); continue
        for turn in (1, 2, 3):
            r = rows.get((V3_RECOMP, turn))
            ok = (r is not None and isinstance(r.get("generated_token_ids"), list) and len(r["generated_token_ids"]) >= 1
                  and isinstance(r.get("score"), (int, float)) and 0.0 <= r["score"] <= 1.0
                  and isinstance(r.get("ttft_ms"), (int, float)) and r["ttft_ms"] > 0)
            if not ok:
                missing.append({"image_id": image, "turn_id": turn, "reason": "ReComp row missing or without tokens / score / TTFT"})
        r1 = rows.get((V3_RECOMP, 1))
        if r1 is not None:
            t1[image] = r1["generated_token_ids"]
            cross_arm += len({tuple(r["generated_token_ids"]) for (m, t), r in rows.items() if t == 1}) == 1
        full = rows.get((V3_FULL, 2))
        if full is not None:      # FullLoad reads every visual K and V: its bytes give the number of visual tokens
            geometry[image] = {"N": int(full["N_content"]), "newline": int(full["n_newline"]), "kv_bytes": int(full["ssd_read_bytes"])}
    compact_json(out_dir / "v3_turn1_tokens.json", t1, replace=True)
    compact_json(out_dir / "v3_geometry.json", geometry, replace=True)
    n = len(images_by_phase["main"])
    return {"status": "PASS" if not missing and len(t1) == n and len(geometry) == n and all(s["error_count"] == 0 for s in summary.values()) else "FAIL",
            "v3_run": str(V3_RUN.relative_to(ROOT)), "method": V3_RECOMP, "images_expected": n, "requests_expected": 3 * n,
            "recomp_requests_complete": 3 * n - sum(1 for m in missing if "turn_id" in m) - 3 * sum(1 for m in missing if "turn_id" not in m),
            "missing_count": len(missing), "missing": missing[:100],
            "v3_turn1_tokens_identical_across_v3_arms_images": cross_arm,
            "criterion": "ReComp row of the v3 main phase present for each of the 25,000 images x 3 turns with generated tokens, score and TTFT; "
                         "every v3 journal equal to the SHA256 recorded by its commit",
            "indexes": summary, "turn1_tokens_file": str((out_dir / "v3_turn1_tokens.json").relative_to(ROOT))}


# ------------------------------------------------------- TTFT boundary
def _function(path, name):
    src = Path(path).read_text(); tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return node, src
    raise KeyError(f"{name} not found in {path}")


def _statements(node):
    """Top-level statements of a function body, docstring dropped."""
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    return body


def ttft_boundary_check(v3_source=None, now_source=None):
    """Is the TTFT of a v3 ReComp row measured over the same interval as a pixel request of this harness?

    Static comparison of the code that timed the v3 record (the copy frozen with that run)
    with the code this run executes:
      * ``Server.recompute`` and ``_FirstTokenTimestamp`` (``mmimpress/serve.py``): where the
        first-token time is taken -- the syntax trees must be identical;
      * ``normal_request`` (v3: scripts/108, bound by scripts/138; now: scripts/156, the
        function scripts/196 and this script call): where the timer starts -- every
        statement must be identical except the two that choose what a *capturing* request
        records, and those two must evaluate the same for a ReComp request
        (``capture_kind == "none"``);
      * the row field: ``ttft_ms`` is ``end_to_end_ttft_ms`` in both.
    """
    v3_source = Path(v3_source or V3_RUN / "source_frozen"); now_source = Path(now_source or ROOT)
    out = {"v3_source": str(v3_source), "checks": {}}
    for name in ("recompute", "_FirstTokenTimestamp"):
        a, _ = _function(v3_source / "mmimpress/serve.py", name); b, _ = _function(now_source / "mmimpress/serve.py", name)
        out["checks"][f"serve.{name}_identical"] = ast.dump(a) == ast.dump(b)
    a, src_a = _function(v3_source / "scripts/108_eval_llava_mt_gqa_B.py", "normal_request")
    b, src_b = _function(now_source / "scripts/156_eval_layer_budget_hold.py", "normal_request")
    sa, sb = _statements(a), _statements(b)
    differing = [i for i, (x, y) in enumerate(zip(sa, sb)) if ast.dump(x) != ast.dump(y)]
    capture_only = True; same_for_recomp = True; detail = []
    for i in differing:
        x, y = sa[i], sb[i]
        tx, ty = ast.get_source_segment(src_a, x), ast.get_source_segment(src_b, y)
        detail.append({"statement": i, "v3": tx, "now": ty})
        capture_only = capture_only and "capture_kind" in tx and "capture_kind" in ty
        exprs = []
        for node in (x, y):      # the expression that depends on capture_kind: the keyword of the Assign, or the test of the If
            if isinstance(node, ast.If):
                exprs.append(node.test)
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and node.value.keywords:
                exprs.append(node.value.keywords[0].value)
            else:
                exprs.append(None)
        if None in exprs:
            same_for_recomp = False
        else:
            values = [eval(compile(ast.Expression(e), "<boundary>", "eval"), {"capture_kind": "none"}) for e in exprs]      # noqa: S307
            same_for_recomp = same_for_recomp and values[0] == values[1] and not values[0]
            detail[-1]["value_for_recompute_request"] = [bool(v) for v in values]
    start = [ast.get_source_segment(s, n) for s, body in ((src_a, sa), (src_b, sb)) for n in body
             if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "started"]
    out["checks"]["normal_request_same_statement_count"] = len(sa) == len(sb)
    out["checks"]["normal_request_differs_only_in_capture_selection"] = capture_only and len(differing) <= 2
    out["checks"]["capture_selection_identical_for_a_recompute_request"] = same_for_recomp
    out["checks"]["timer_start_statement_identical"] = len(start) == 2 and start[0].replace(" ", "") == start[1].replace(" ", "")
    row_v3 = re.search(r"['\"]ttft_ms['\"]\s*:\s*result\[['\"]end_to_end_ttft_ms['\"]\]", (v3_source / "scripts/108_eval_llava_mt_gqa_B.py").read_text())
    row_now = re.search(r"['\"]ttft_ms['\"]\s*:\s*result\[['\"]end_to_end_ttft_ms['\"]\]", Path(__file__).read_text())
    out["checks"]["row_ttft_field_identical"] = bool(row_v3 and row_now)
    out["normal_request_differing_statements"] = detail
    out["same_interval"] = all(out["checks"].values())
    out["timer_start"] = ("after torch.cuda.synchronize() and reset_peak_memory_stats(), immediately before the prompt is assembled "
                          "(history, image file open and decode, processor, host-to-device copy are inside)")
    out["timer_end"] = ("the first generated token: the time stamp taken by the stopping-criterion marker inside generate "
                        "(Server.recompute, first_token_at_s)")
    out["not_part_of_the_interval_but_different"] = [
        "v3 ran five arms per image in one process (ReComp, FullLoad, MPIC-32, AllHead, [B]); this run runs three (four in the pilot)",
        "v3 constructed Server(runner, max_new_tokens=16) with default selection arguments; this run passes the 2026-11-17 arguments and "
        "creates the read-ahead worker thread before the first request (neither is used by a pixel request)",
        "the follow-up requests of the stored methods of this run use the asynchronous read paths added on 2026-11-16; v3's FullLoad read synchronously",
    ]
    return out


# ------------------------------------------------------------------- freeze
def store_bytes(n_content, newline):
    """Bytes of the three stores of one image and what building them writes (formulas checked
    against the store sizes the 2026-11-17 and 2026-11-20 runs recorded)."""
    vn = n_content + newline
    kv = 2 * NUM_LAYERS * vn * ROW_BYTES
    probe = NUM_LAYERS * vn * SERVER_ARGS["probe_heads"] * 128 * 2
    sep = 2 * NUM_LAYERS * newline * ROW_BYTES
    hidden = vn * 8192
    raster = kv + probe + sep + hidden + SYSTEM_PREFIX_BYTES + (64 << 10)
    trailer = NUM_LAYERS * OFFICIAL["probe_heads"] * vn * 4
    impress = raster + trailer + vn * 300                                   # + per-layer order in the metadata
    impress_written = impress + kv + probe + sep + SYSTEM_PREFIX_BYTES + trailer      # written, then rewritten in the reordered layout
    packed = 2 * ROW_BYTES * NUM_LAYERS * ((n_content + 3) // 4) + sep + SYSTEM_PREFIX_BYTES + (128 << 10)
    return {"raster": raster, NEW_KIND: impress, ORIG_KIND: packed, "bundle": raster + impress + packed + vn * ROW_BYTES,
            "written": raster + impress_written + packed}


def build_manifest(limit=0):
    """``limit`` > 0 (development rehearsals only): the first ``limit`` gate / pilot images and ``limit + 1`` main images."""
    from concurrent.futures import ThreadPoolExecutor
    from mmimpress.mtvqa25k.dataset import workload_hash
    raw = json.loads((DATA / "dialogues.json").read_text())
    assert raw["dataset"] == DATASET and len(raw["dialogues"]) == 25000
    assert workload_hash(raw["dialogues"]) == WORKLOAD_SHA == json.loads((DATA / "config.json").read_text())["ordered_workload_sha256"]
    ds = dialogues_for_base(raw["dialogues"])
    v3 = {ph: json.loads((V3_LLAVA / ph / "manifest.json").read_text()) for ph in ("smoke", "pilot40", "main")}
    assert v3["main"]["dialogues"] == ds, "dataset dialogues differ from the v3 MT-VQA main manifest"
    assert [d["global_dialog_ordinal"] for d in ds] == list(range(25000)) and len({d["image_id"] for d in ds}) == 25000
    for d in ds:
        assert len(d["turns"]) == 3 and all(len(t["answers"]) == 10 and "annotation" in t for t in d["turns"])

    def check(d):
        return sha(ROOT / d["image_path"]) == d["image_sha256"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(check, ds)), "an image file differs from the frozen dataset"
    identity = {d["image_id"]: {"image_path": d["image_path"], "sha256": d["image_sha256"]} for d in ds}
    by_image = {d["image_id"]: d for d in ds}
    smoke = [d["image_id"] for d in v3["smoke"]["dialogues"]]
    pilot = [d["image_id"] for d in v3["pilot40"]["dialogues"]]
    assert len(smoke) == 4 and len(pilot) == 40 and set(smoke) <= set(by_image) and set(pilot) <= set(by_image)
    phase_images = {"gate": smoke, "pilot": pilot, "main": [d["image_id"] for d in ds]}
    if limit:
        phase_images = {"gate": smoke[:limit], "pilot": pilot[:limit], "main": phase_images["main"][:limit + 1]}
    phase_ids = {ph: [by_image[i]["dialog_id"] for i in imgs] for ph, imgs in phase_images.items()}
    manifest = {"dataset": DATASET, "schema_version": SCHEMA, "ordered_workload_sha256": WORKLOAD_SHA,
                "dialogues": ds, "image_identity": identity, "phase_dialogue_ids": phase_ids, "phase_images": phase_images,
                "phase_methods": PHASE_METHODS,
                "expected_requests": {p: len(phase_ids[p]) * 3 * len(PHASE_METHODS[p]) for p in PHASES},
                "v3_manifest_sha256": {ph: sha(V3_LLAVA / ph / "manifest.json") for ph in v3},
                "v3_phase_images": {"smoke": smoke, "pilot40": pilot},
                "gate_policy": "the four images of the v3 smoke manifest, in that order",
                "pilot_policy": "the 40 images of the v3 pilot40 manifest, in that order",
                "main_policy": "all 25,000 images, manifest order (global_dialog_ordinal 0..24999)",
                "arm_rotation": "method order of an image = the phase's arm list rotated by (index of the image in the phase's list) % (number of arms)"}
    assert limit or manifest["expected_requests"] == EXPECTED_REQUESTS, manifest["expected_requests"]
    if limit:
        manifest["rehearsal_image_limit"] = limit
    return manifest


def freeze(run):
    import transformers
    assert run.parent == ROOT / "runs" and run.name.startswith(RUN_PREFIX) and CONTRACT.is_file()
    impl = json.loads((IMPL_REF_RUN / "config.json").read_text()); ours = json.loads((OURS_REF_RUN / "config.json").read_text())
    env = impl["environment"]
    assert (transformers.__version__, torch.__version__, np.__version__, sys.executable) == (
        env["transformers"], env["torch"], env["numpy"], env["executable"]), "not the environment of the 2026-11-17 run"
    assert ours["environment"] == env
    assert impl["server_args"] == SERVER_ARGS == ours["server_args"] and impl["impress_official"] == OFFICIAL
    assert impl["readahead"][FULL_ASYNC] == READAHEAD[FULL_ASYNC] and impl["readahead"][NEW_ASYNC] == READAHEAD[NEW_ASYNC]
    assert ours["readahead"][ORIG_ASYNC] == READAHEAD[ORIG_ASYNC] and impl["readahead"]["max_ahead"] == ours["readahead"]["max_ahead"] == READAHEAD["max_ahead"]
    assert impl["model"] == ours["model"] == MODEL_ID and impl["revision"] == ours["revision"] == REVISION
    assert {k: v for k, v in ours["arms"][ORIG_ASYNC].items() if k != "display"} == {k: v for k, v in ARMS[ORIG_ASYNC].items() if k != "display"}
    for m in (FULL_ASYNC, NEW, NEW_ASYNC, NEW_R100, FULL_REORDERED):
        assert {k: v for k, v in impl["arms"][m].items() if k != "display"} == {k: v for k, v in ARMS[m].items() if k != "display"}, m
    # method code == the two reference freezes: every file of those freezes is byte-identical except the listed ones
    now = source_hashes()
    diffs = {}
    for name, ref, allowed in (("impl_20261117", impl, CHANGED_SINCE_IMPL_REF), ("ours_20261120", ours, CHANGED_SINCE_OURS_REF)):
        before = {k: v for k, v in ref["source_sha256"].items() if not REPORT_SCRIPT.match(k)}
        changed = sorted(k for k in before if k in now and now[k] != before[k])
        removed = sorted(k for k in before if k not in now)
        assert set(changed) <= set(allowed) and not removed, (name, changed, removed)
        diffs[name] = {"changed": changed, "removed": removed, "added": sorted(k for k in now if k not in before)}
    checkout = os.environ.get("IMPRESS_OFFICIAL_REPO")
    official_head = subprocess.check_output(["git", "-C", checkout, "rev-parse", "HEAD"], text=True).strip() if checkout else None
    assert official_head in (None, OFFICIAL["reference_commit"]), official_head
    run.mkdir(parents=True, exist_ok=False)
    # development rehearsals only (run name contains "_dev_"; never a measurement): a few images per phase
    limit = int(os.environ.get("LNEXT_MAIN_DEV_LIMIT", "0")) if "_dev_" in run.name else 0
    manifest = build_manifest(limit)
    compact_json(run / "manifest.json", manifest)
    (run / "vectors").mkdir()
    target = run / "vectors" / VECTOR_SOURCE.name
    shutil.copyfile(VECTOR_SOURCE, target)
    assert sha(target) == VECTOR_FILE_SHA == sha(VECTOR_SOURCE)
    doc = load_budget_document(target)
    for ref in (impl, ours):
        old = ref["layer_budgets"][VECTOR]
        assert doc["layer_budget_sha256"] == VECTOR_RATIO_SHA == old["layer_budget_sha256"] and doc["layer_keep_ratios"] == old["layer_keep_ratios"]
        assert doc["r_min"] == old["r_min"] == 0.0 and doc["r_max"] == old["r_max"]
    assert doc["alloc_id"] == VECTOR
    budgets = {VECTOR: {"path": str(target.relative_to(ROOT)), "layer_budget_sha256": doc["layer_budget_sha256"], "layer_keep_ratios": doc["layer_keep_ratios"],
                        "r_min": doc["r_min"], "r_max": doc["r_max"], "alloc_id": VECTOR, "policy": doc["policy"], "schema": doc["schema"],
                        "num_layers": doc["num_layers"], "total_ratio": doc["total_ratio"], "source": doc["source"]}}
    # the largest image LLaVA-NeXT's AnyRes can produce here: N = 2880 content tokens (+ separators); refined from the v3 geometry by the prerun stage
    bound = store_bytes(2880, 48)
    free = shutil.disk_usage(run).free
    plan = {"policy": "FRESH_IMAGE_STREAMING (image-major): one live image; stores built from the arms' own turn 1 -> served -> audited -> "
                      "committed -> the image's stores deleted",
            "minimum_reserve_bytes": 30 * GiB, "largest_image_bundle_bytes": bound["bundle"],
            "largest_image_bundle_note": "FullLoad store + IMPRESS store + Ours store of the largest geometry (N = 2880 + 48 separators) "
                                         "+ one layer file in flight during the IMPRESS rewrite",
            "free_space_factor": FREE_SPACE_FACTOR, "required_free_bytes": FREE_SPACE_FACTOR * bound["bundle"],
            "free_bytes_at_freeze": free,
            "deletion_scope": "runs/<RUN>/scratch only (the stores of a committed image, by the exact file list recorded for it; the stores of an "
                              "interrupted or failed image before it is redone); existing stores are never deleted"}
    plan["status"] = "PASS" if free >= max(plan["required_free_bytes"], plan["minimum_reserve_bytes"]) else "BLOCKED_STORAGE"
    support.publish(run / "storage_plan.json", plan)
    config = {"schema_version": SCHEMA, "dataset": DATASET, "model": MODEL_ID, "revision": REVISION, "seed": 1234, "greedy": True, "max_new_tokens": 16,
              "batch_size": 1, "cpu_threads": 2, "history_policy": "method_local_generated",
              "metric": "official ten-answer leave-one-out VQA score (GT-Vision-Lab/VQA a013f0043c1e2cdc995922dfe257f7149aa9af06)",
              "phase_methods": PHASE_METHODS,
              "phases": {ph: {"methods": PHASE_METHODS[ph], "dialogues": len(manifest["phase_dialogue_ids"][ph]),
                              "images": len(manifest["phase_images"][ph])} for ph in PHASES},
              "arms": {m: dict(a) for m, a in ARMS.items()}, "main_methods": MAIN_METHODS,
              "server_args": SERVER_ARGS, "impress_official": OFFICIAL, "readahead": READAHEAD,
              "official_checkout": {"path": str(Path(checkout).resolve()) if checkout else None, "head": official_head,
                                    "repository": OFFICIAL["reference_repository"]},
              "store_kinds": {"raster": "full visual KV, raster order (FullLoad)", NEW_KIND: "full visual KV + probe key copy, per-layer reordered (IMPRESS)",
                              ORIG_KIND: "packed, PYR-beta-raw 25 %, row_order=original (Ours)"},
              "store_format_ours": ORIGINAL_STORE_FORMAT, "layer_budgets": budgets, "vector_file_sha256": VECTOR_FILE_SHA,
              "arm_order": manifest["arm_rotation"],
              "source_sha256": now, "source_difference_from_reference_freezes": diffs,
              "implementation_reference": {
                  "fullload_impress": {"run": str(IMPL_REF_RUN.relative_to(ROOT)), "config_sha256": sha(IMPL_REF_RUN / "config.json"),
                                       "harness": "scripts/196_eval_mt_gqa_impress_code.py (imported)",
                                       "harness_sha256": sha(ROOT / "scripts/196_eval_mt_gqa_impress_code.py")},
                  "ours": {"run": str(OURS_REF_RUN.relative_to(ROOT)), "config_sha256": sha(OURS_REF_RUN / "config.json"),
                           "harness": "scripts/202_eval_mt_gqa_ours_order.py (the same writer and request calls; not imported)",
                           "harness_sha256": sha(ROOT / "scripts/202_eval_mt_gqa_ours_order.py"), "method_id": ORIG_ASYNC}},
              "recomp_reference": {"run": str(V3_RUN.relative_to(ROOT)), "method": V3_RECOMP, "phase": "main",
                                   "use": "ReComp score and TTFT of the report; turn-1 token continuity"},
              "contract_sha256": sha(CONTRACT),
              "manifest_sha256": sha(run / "manifest.json"), "storage_plan_sha256": sha(run / "storage_plan.json"),
              "source_T1_capture": "per store kind, the first arm of that kind in the image's rotation; every arm still runs its own full-pixel turn 1",
              "resume_policy": "committed images skipped; an interrupted image restarts at turn 1 for every arm, its partial rows and stores are discarded",
              "main_failure_policy": {"retry_once_then_skip": True, "max_failed_images": MAIN_FAILED_IMAGES_MAX,
                                      "max_consecutive_errors": 2 * MAIN_FAILED_CONSECUTIVE_MAX},
              "rehearsal_seconds_per_image": REHEARSAL_SECONDS_PER_IMAGE,
              "environment": {"python": sys.version, "executable": sys.executable, "torch": torch.__version__, "transformers": transformers.__version__,
                              "numpy": np.__version__},
              "lifecycle_source": "scripts/164 + scripts/156 helpers and scripts/196 (imported, not copied)"}
    support.publish(run / "config.json", config)
    shutil.copy2(CONTRACT, run / "CONTRACT.md")
    envdoc = {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
              "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
              "pip_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)}
    support.publish(run / "environment_before.json", envdoc)
    support.publish(run / "protection_before.json", stat_inventory(run, run / "protection_before.tsv.gz"))
    print(json.dumps({"freeze": "PASS", "run": str(run), "expected": manifest["expected_requests"], "storage": plan["status"],
                      "free_gib": free / 2**30, "transformers": transformers.__version__, "source_difference": {k: v["changed"] for k, v in diffs.items()}}), flush=True)
    if plan["status"] != "PASS":
        raise RuntimeError(f"BLOCKED_STORAGE: free {free} < required {plan['required_free_bytes']}")


# ------------------------------------------------------------------- prerun
def prerun(run):
    """The checks before any GPU measurement; writes results/<run name>/prerun.json.

    Blocking (the driver does not start the experiment): free space below 20 x the
    largest per-image store bundle; a ReComp row missing in the v3 record.
    """
    config, manifest = verify_freeze(run)
    out_root = ROOT / "results" / run.name; out_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    v3_images = dict(manifest["v3_phase_images"], main=[d["image_id"] for d in manifest["dialogues"]])
    reference = build_reference(run, v3_images)
    geometry = json.loads((reference_dir(run) / "v3_geometry.json").read_text())
    per_image = {}
    bad_geometry = []
    for image, g in geometry.items():
        if g["kv_bytes"] != 2 * NUM_LAYERS * (g["N"] + g["newline"]) * ROW_BYTES:
            bad_geometry.append(image)
        per_image[image] = store_bytes(g["N"], g["newline"])
    writes = {}
    for phase in PHASES:
        arms = PHASE_METHODS[phase]; kinds = store_kinds(arms)
        total = 0
        for image in manifest["phase_images"][phase]:
            b = per_image.get(image)
            if b is None:
                continue
            total += b["written"] if set(kinds) == {"raster", NEW_KIND, ORIG_KIND} else 0
        writes[phase] = total
    writes["total"] = sum(writes.values())
    main_images = manifest["phase_images"]["main"]
    sizes = [per_image[i] for i in main_images if i in per_image]
    largest = max(sizes, key=lambda b: b["bundle"]) if sizes else store_bytes(2880, 48)
    mean = lambda key: float(np.mean([b[key] for b in sizes])) if sizes else None      # noqa: E731
    free = shutil.disk_usage(run).free
    required = FREE_SPACE_FACTOR * largest["bundle"]
    plan = json.loads((run / "storage_plan.json").read_text())
    evidence = 0.4e6 * len(main_images) + 3e6 * (len(manifest["phase_images"]["gate"]) + len(manifest["phase_images"]["pilot"]))
    smart = STATUS.smart_snapshot()
    boundary = ttft_boundary_check()
    seconds = config["rehearsal_seconds_per_image"]
    blocking = []
    if free < max(required, plan["required_free_bytes"]):
        blocking.append(f"free space {free} B is below 20 x the largest per-image store bundle ({required} B)")
    if reference["status"] != "PASS":
        blocking.append(f"v3 ReComp record incomplete: {reference['missing_count']} missing rows / images")
    doc = {
        "written_before_any_gpu_measurement_utc": STATUS.utc(time.time()), "run": run.name,
        "status": "PASS" if not blocking else "BLOCKED", "blocking": blocking,
        "expected_ssd_write": {
            "definition": "per image, the bytes the three store builds write: FullLoad store + IMPRESS store (written, then rewritten in the "
                          "reordered layout) + Ours store; summed over the images of each phase from the v3 record's per-image geometry "
                          "(number of visual tokens and separators). Request rows and evidence files (see evidence_bytes_estimate) are not included.",
            "bytes": writes, "total_tb": writes["total"] / 1e12, "main_tb": writes["main"] / 1e12,
            "per_image_mean_bytes": {"fullload_store": mean("raster"), "impress_store": mean(NEW_KIND), "ours_store": mean(ORIG_KIND),
                                     "written": mean("written")},
            "per_image_largest_bundle_bytes": largest["bundle"], "images_with_unexpected_geometry": len(bad_geometry)},
        "evidence_bytes_estimate": evidence,
        "expected_duration": {
            "seconds_per_main_image": seconds, "main_hours": seconds * len(main_images) / 3600, "main_days": seconds * len(main_images) / 86400,
            "basis": "development rehearsal of 2026-11-21 (a few images, not a measurement); replaced by the pilot's measured seconds per image "
                     "in prerun_pilot_estimate.json and status.json when the pilot has finished",
            "gate_and_pilot_minutes_estimate": 25},
        "free_space": {"free_bytes": free, "free_gib": free / GiB, "largest_image_bundle_bytes": largest["bundle"],
                       "required_free_bytes_20x_bundle": required, "ratio_free_to_bundle": free / largest["bundle"],
                       "rule": "do not start when free space < 20 x the largest per-image store bundle; rechecked before every image",
                       "ok": free >= required},
        "smart": smart,
        "ssd_counters_at_prerun": STATUS._safe(STATUS.sectors), "ssd_temperature_at_prerun": STATUS.ssd_temperature(),
        "v3_recomp_record": reference,
        "ttft_interval_v3_vs_this_run": boundary,
        "store_deletion": "the stores of an image are deleted as soon as the image is committed (one live image at a time)",
        "seconds": time.time() - started}
    tmp = out_root / ".prerun.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n"); os.replace(tmp, out_root / "prerun.json")
    atomic_json(run / "prerun_receipt.json", {"status": doc["status"], "blocking": blocking, "prerun_sha256": sha(out_root / "prerun.json"),
                                              "config_sha256": config["_file_sha256"], "at_unix": time.time()}, replace=True)
    STATUS.update(run, expected_ssd_write_tb=round(writes["total"] / 1e12, 2), expected_main_hours=round(seconds * len(main_images) / 3600, 1),
                  smart=smart["status"], prerun=doc["status"])
    print(json.dumps({"prerun": doc["status"], "blocking": blocking, "expected_ssd_write_tb": round(writes["total"] / 1e12, 2),
                      "free_gib": round(free / GiB, 1), "bundle_gib": round(largest["bundle"] / GiB, 2), "smart": smart["status"],
                      "v3_recomp": reference["status"], "ttft_same_interval": boundary["same_interval"]}), flush=True)
    return doc


def load_model(config):
    torch.set_num_threads(config["cpu_threads"]); torch.manual_seed(1234); np.random.seed(1234)
    torch.set_grad_enabled(False)
    runner = V.load_runner()
    from mmimpress.serve import Server
    servers = {"base": Server(runner, ratio=SERVER_ARGS["ratio"], probe=SERVER_ARGS["probe_heads"], alpha=SERVER_ARGS["alpha"], max_new_tokens=16)}
    QA._warmup(runner, servers["base"])
    servers["base"]._prefetch_pool()      # the single read-ahead worker thread exists before any request
    return runner, servers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True, help="runs/llava_next_mtvqa_main_<UTC>")
    p.add_argument("--stage", choices=("freeze", "prerun") + PHASES, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    run = args.run.resolve()
    if args.stage == "freeze":
        return freeze(run)
    assert run.parent == ROOT / "runs" and run.name.startswith(RUN_PREFIX)
    if args.stage == "prerun":
        doc = prerun(run)
        sys.exit(0 if doc["status"] == "PASS" else 1)
    config, manifest = verify_freeze(run)
    assert json.loads((run / "storage_plan.json").read_text())["status"] == "PASS"
    receipt = json.loads((run / "prerun_receipt.json").read_text())
    if receipt["status"] != "PASS" or receipt["config_sha256"] != config["_file_sha256"]:
        raise RuntimeError("PRERUN_NOT_PASSED")
    phase = args.stage
    if phase == "main":      # the main run is admitted only by the recorded gate decision
        admit = json.loads((run / "gates" / "admission.json").read_text())
        if admit["status"] != "PASS" or admit["config_sha256"] != config["_file_sha256"]:
            raise RuntimeError("MAIN_NOT_ADMITTED")
    if (run / phase).exists() and not args.resume:
        raise FileExistsError("use --resume; previous artifacts preserved")
    wait_for_gpu(run, phase)
    runner, servers = load_model(config)
    try:
        atomic_json(run / f"{phase}_start_{time.strftime('%Y%m%dT%H%M%S')}.json",
                    {"started_unix": time.time(), "capacity": capacity_guard(run), "pid": os.getpid(), "resume": bool(args.resume),
                     "boot_id": STATUS.boot_id(), "committed_requests_at_start": committed_request_counts(run)[phase],
                     "ssd_temperature": STATUS.ssd_temperature()})
        receipt = run_phase(runner, servers, run, phase, config, manifest)
        atomic_json(run / f"{phase}_execution_complete.json", {"status": "PASS", **receipt, "finished_unix": time.time()}, replace=True)
    except BaseException as exc:
        error = {"status": "PARTIAL", "error": repr(exc), "traceback": traceback.format_exc(), "at_unix": time.time(),
                 "resume_command": f"scripts/run_llava_next_mtvqa_main.sh {run.relative_to(ROOT)}"}
        atomic_json(run / ("interruption_" + phase + "_" + time.strftime("%Y%m%dT%H%M%S") + ".json"), error)
        STATUS.update(run, last_error={"stage": f"{phase}_phase", "error": repr(exc)[:500], "at_utc": STATUS.utc(time.time())})
        print(json.dumps({k: v for k, v in error.items() if k != "traceback"}), flush=True)
        raise
    finally:
        for server in servers.values():
            server.close()


if __name__ == "__main__":
    main()

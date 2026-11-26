#!/usr/bin/env python3
"""LLaVA-NeXT-7B x MT-GQA main experiment, second version: the prompt is tokenised to the
original-LLaVA token sequence, and ReComp / FullLoad / IMPRESS / Ours are measured in one run
(contract docs/mt_gqa_main_v2_contract.md).

What changes against the earlier LLaVA-NeXT MT-GQA runs is one thing: the string handed to the
tokenizer (``mmimpress.mt_gqa_tokens``, ``tokenization="official"``: ``<image> \\n`` instead of
``<image>\\n``), so that the fast HF tokenizer emits the ids of the original LLaVA procedure
(one ``▁`` right after the image).  The prompt string, the question order, the history, the
generation settings and the scoring are the ones of the earlier runs.  The earlier path stays
available as ``tokenization="legacy"`` (phase ``legacy``).

No method code is changed or added.  Where the code comes from:

* scripts/209_eval_llava_next_mtvqa_main.py is imported and its request, capture, writer and
  activation functions are called as they are.  Those are the calls of the reference runs:
  FullLoad and IMPRESS of 2026-11-17 (scripts/196, results/mt_gqa_impress_code_20261117T185832Z),
  Ours = "Ours(원래 순서)" of 2026-11-20 (scripts/202, results/mt_gqa_ours_order_20261120T052725Z):
  PYR-beta-raw vector, 25 % of the KV budget, ``row_order="original"``.
* Data, prompt and scoring: the MT-GQA lifecycle of scripts/156 / 164 (398 images, 4,061
  dialogues x 3 turns, method-local generated history, strict normalized EM).

Phases
  legacy    4 gate images, the 4 main arms, tokenization "legacy"   (gate 3)
  gate      4 gate images, 9 arms                                    (gates 4, 5, 6)
  devgate   40 DEV images, the 7 arms of gates 4 and 5               (gates 4, 5)
  dev       40 DEV images, the 4 main arms                           (gates 6, 8: the pilot)
  main      398 images,    the 4 main arms

One image at a time: in the image's first dialogue the first arm of a store kind writes that
store from its own turn 1; every arm computes turn 1 of every dialogue from pixels; turns 2-3
read the stores from the cold SSD; the image is audited, committed, and its stores deleted.

After every request, outside every timer, the ids the request really used are compared with the
original-LLaVA ids of its prompt (slow legacy tokenizer).  Under "official" a mismatch stops the
image (and so the phase).

Resume (``--resume``): committed images are skipped.  Inside the image that was interrupted,
complete dialogues are kept, the interrupted dialogue restarts at turn 1 and its partial rows
leave the journal -- provided the image's stores still match the SHA256 recorded when they were
built; otherwise (or when the image's first dialogue, which builds the stores, was the
interrupted one) the image restarts from its first dialogue.

Stages: --stage freeze | legacy | gate | devgate | dev | main [--resume]
"""
from __future__ import annotations

import argparse
import gc
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
from mmimpress import mt_gqa_tokens as TOK  # noqa: E402
from mmimpress.layer_budget import load_budget_document  # noqa: E402
from mmimpress.mtvqa25k.adapter import archive_before_commit, read_rows, remove_archived_duplicates  # noqa: E402
from mmimpress.packed_store import (IMPORTANCE_ORDER_KEYS, ORIGINAL_STORE_FORMAT, PACKED_NAME, ROW_ORDER_ORIGINAL,  # noqa: E402
                                    layer_original_indices)


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# the LLaVA-NeXT harness of 2026-11-21 (itself scripts/196 + the calls of scripts/202), used as is
N = load_script("_mtgqa_main_v2_backend", "209_eval_llava_next_mtvqa_main.py")
import llava_next_main_status as STATUS  # noqa: E402  (generic status.json / SSD bookkeeping helpers)

SCHEMA = "llava-next-mtgqa-main-v2"
DATASET = "MT-GQA-reconstructed"
RUN_PREFIX = "mt_gqa_main_v2_"
CONTRACT = ROOT / "docs/mt_gqa_main_v2_contract.md"
DRIVER = "scripts/run_mt_gqa_main_v2.sh"
# the imported writer calls put the dataset identity and the schema of the calling run into the store metadata
N.DATASET = DATASET
N.SCHEMA = SCHEMA

E, H, V, QA, H1, support = N.E, N.H, N.V, N.QA, N.H1, N.support
IMPL_REF_RUN = N.IMPL_REF_RUN            # runs/mt_gqa_impress_code_20261117T185832Z: FullLoad / IMPRESS implementation; previous IMPRESS rows
OURS_REF_RUN = N.OURS_REF_RUN            # runs/mt_gqa_ours_order_20261120T052725Z: Ours implementation; previous ReComp / FullLoad / Ours rows
PRIOR = H.PRIOR
V2_SPLIT = E.V2_SPLIT
VECTOR, VECTOR_RATIO_SHA, VECTOR_FILE_SHA, VECTOR_SOURCE = N.VECTOR, N.VECTOR_RATIO_SHA, N.VECTOR_FILE_SHA, N.VECTOR_SOURCE
SERVER_ARGS, OFFICIAL, READAHEAD = N.SERVER_ARGS, N.OFFICIAL, N.READAHEAD
MODEL_ID, REVISION, SHORT_ANSWER = N.MODEL_ID, N.REVISION, N.SHORT_ANSWER
INDEX_SHA, WORKLOAD_SHA = H.INDEX_SHA, H.WORKLOAD_SHA
NEW_KIND, ORIG_KIND, FULL_KINDS = N.NEW_KIND, N.ORIG_KIND, N.FULL_KINDS
RECOMP, FULL_ASYNC = N.RECOMP, N.FULL_ASYNC
NEW, NEW_ASYNC, NEW_R100, FULL_REORDERED = N.NEW, N.NEW_ASYNC, N.NEW_R100, N.FULL_REORDERED
ORIG, ORIG_ASYNC, ORIG_REPEAT = N.ORIG, N.ORIG_ASYNC, N.ORIG_REPEAT
ARMS = N.ARMS
GiB = 1 << 30
ROW_BYTES, NUM_LAYERS = N.ROW_BYTES, N.NUM_LAYERS

MAIN_METHODS = [RECOMP, FULL_ASYNC, NEW_ASYNC, ORIG_ASYNC]          # ReComp, FullLoad, IMPRESS, Ours (the arm order of the 2026-11-17 run)
GATE_ONLY = [NEW, NEW_R100, FULL_REORDERED, ORIG, ORIG_REPEAT]
PHASE_METHODS = {
    "legacy": list(MAIN_METHODS),
    "gate": [RECOMP, FULL_ASYNC, NEW, NEW_ASYNC, NEW_R100, FULL_REORDERED, ORIG, ORIG_ASYNC, ORIG_REPEAT],
    "devgate": [NEW, NEW_ASYNC, NEW_R100, FULL_REORDERED, ORIG, ORIG_ASYNC, ORIG_REPEAT],
    "dev": list(MAIN_METHODS),
    "main": list(MAIN_METHODS),
}
PHASES = ("legacy", "gate", "devgate", "dev", "main")
PHASE_IMAGES = {"legacy": "gate", "gate": "gate", "devgate": "dev", "dev": "dev", "main": "main"}
PHASE_TOKENIZATION = {"legacy": TOK.TOKENIZATION_LEGACY, "gate": TOK.TOKENIZATION_OFFICIAL, "devgate": TOK.TOKENIZATION_OFFICIAL,
                      "dev": TOK.TOKENIZATION_OFFICIAL, "main": TOK.TOKENIZATION_OFFICIAL}
EXPECTED_REQUESTS = {"legacy": 49 * 3 * 4, "gate": 49 * 3 * 9, "devgate": 378 * 3 * 7, "dev": 378 * 3 * 4, "main": 12183 * 4}
# source files that may differ from the two reference freezes (the same lists as scripts/209: the row-order option of
# 2026-11-20 and the LLaVA-1.5 port of 2026-11-21); everything else those freezes hashed must be byte-identical
CHANGED_SINCE_IMPL_REF = N.CHANGED_SINCE_IMPL_REF
CHANGED_SINCE_OURS_REF = N.CHANGED_SINCE_OURS_REF
# seconds per request used for the estimated end before this run has measured its own rate (reference: the 2026-11-17 main phase;
# the gate arms are dominated by the IMPRESS family)
# errors of a request phase that a restart would repeat
NO_RETRY_MARKERS = ("OFFICIAL_TOKEN_IDS_MISMATCH", "REQUEST_IDS_DIFFER_FROM_TOKENIZER_INPUT_IDS", "INDEPENDENT_IMAGE_AUDIT_FAIL", "BLOCKED_STORAGE")
SECONDS_PER_REQUEST_PRIOR = {"legacy": 0.65, "gate": 0.80, "devgate": 0.85, "dev": 0.65, "main": 0.65}

sha, canonical_hash, safe, atomic_json, append, load_jsonl = H.sha, H.canonical_hash, H.safe, H.atomic_json, H.append, H.load_jsonl
method_order, strict_score, assert_gpu_exclusive = H.method_order, H.strict_score, H.assert_gpu_exclusive
file_hashes, Checks, capacity_guard, cleanup_committed, frozen_dialogues = (
    H.file_hashes, H.Checks, H.capacity_guard, H.cleanup_committed, H.frozen_dialogues)
doc_counts, independent_counts = H.doc_counts, H.independent_counts
source_hashes, freeze_violations, REPORT_SCRIPT = N.source_hashes, N.freeze_violations, N.REPORT_SCRIPT
kind_of, stored_methods, store_kinds, scratch_bytes, compact_json = N.kind_of, N.stored_methods, N.store_kinds, N.scratch_bytes, N.compact_json
stat_inventory, _chunks_of = N.stat_inventory, N._chunks_of

_TOKENIZERS = {}


def tokenizers():
    """(fast, slow legacy, image token id) of the checkpoint snapshot the runner loads, loaded once (CPU)."""
    if not _TOKENIZERS:
        assert V.SNAPSHOT.name == REVISION
        _TOKENIZERS["value"] = TOK.load_tokenizers(str(V.SNAPSHOT))
    return _TOKENIZERS["value"]


def request_id(run_id, phase, method, dialog_id, turn):
    return f"{SCHEMA}:{run_id}:{phase}:{MODEL_ID}:{method}:{dialog_id}:t{turn}"


# ------------------------------------------------------------------ prompt
class PromptFactory:
    """The prompt of one request.

    ``H1.prompt_factory`` (scripts/156) builds the prompt string exactly as in the earlier runs.
    What the request functions receive -- and tokenise inside the TTFT timer -- is
    ``TOK.tokenizer_input(prompt, tokenization)``: the same string under "legacy", the string
    with one space after ``<image>`` under "official".
    """

    def __init__(self, methods, dialog, turn, method, generated, tokenization):
        self.args = (methods, dialog, turn, method, generated)
        self.tokenization = tokenization
        self.prompt = self.text = None

    def __call__(self):
        prompt, history, entries = H1.prompt_factory(*self.args)
        self.prompt = prompt
        self.text = TOK.tokenizer_input(prompt, self.tokenization)
        return self.text, history, entries


def close_diag(runner, diag, factory):
    """After a request, outside every timer: the request functions call the string they tokenised
    ``prompt``; from here on ``prompt`` is the prompt string and ``tokenizer_input`` what was tokenised."""
    assert diag["prompt"] == factory.text and factory.prompt is not None
    out = {a: b for a, b in diag.items() if a not in ("enc_cpu", "hidden", "vision", "turn1_stats")}
    out["tokenizer_input"] = diag["prompt"]
    out["prompt"] = factory.prompt
    out["tokenization"] = factory.tokenization
    if "enc_cpu" in diag:      # a pixel request: the ids the model received (image block expanded)
        out["request_ids"] = TOK.collapse_image(diag["enc_cpu"]["input_ids"][0].tolist(), runner.image_token_id)
    return out


def pixel_request(runner, servers, image_path, factory, capture):
    """The pixel request calls of scripts/209 (= scripts/196)."""
    if capture == NEW_KIND:      # the same pixel request, with the attention capture attached
        return N.turn1_request_with_stats(runner, servers["base"], image_path, factory)
    return N.normal_request(runner, servers["base"], image_path, factory,
                            "raster" if capture == "raster" else ("image_only" if capture != "none" else "none"))


def make_row(runner, result, diag, meta, *, methods, method, dialog, turn, phase, run, attempt, physical_id, ordinal, store_info, config):
    arm = ARMS[method]
    question = dialog["turns"][turn - 1]
    n = int(meta["n_spatial"]); hit = turn > 1 and arm["family"] != "recompute"
    logical = request_id(run.name, phase, method, dialog["dialog_id"], turn)
    tokenization = config["phases"][phase]["tokenization"]
    assert diag["tokenization"] == tokenization and diag["tokenizer_input"] == TOK.tokenizer_input(diag["prompt"], tokenization)
    tokenized = runner.processor.tokenizer(diag["tokenizer_input"], return_tensors="pt")
    suffix = QA._suffix_from_tokenized(runner, tokenized).cpu().tolist()
    if diag.get("suffix_input_ids") is not None:
        assert diag["suffix_input_ids"] == suffix
    # the ids this request used: a pixel request recorded them; a stored request used the store's prefix ids + this suffix
    path_ids = TOK.collapse_image(tokenized["input_ids"][0].tolist(), runner.image_token_id)
    used = diag.get("request_ids")
    if used is None:
        used = TOK.collapse_image(list(meta["prefix_input_ids"]) + suffix, runner.image_token_id)
    if used != path_ids:
        raise RuntimeError(f"REQUEST_IDS_DIFFER_FROM_TOKENIZER_INPUT_IDS: {logical}")
    official = TOK.official_token_ids(tokenizers()[1], diag["prompt"])
    match = path_ids == official
    if tokenization == TOK.TOKENIZATION_OFFICIAL and not match:
        raise RuntimeError(f"OFFICIAL_TOKEN_IDS_MISMATCH: {logical}: request {len(path_ids)} ids, original LLaVA {len(official)} ids")
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
    return {"schema_version": SCHEMA, "experiment_id": run.name, "phase": phase, "attempt_id": attempt, "dataset": DATASET,
            "logical_request_id": logical, "request_id": logical, "physical_execution_id": physical_id,
            "model_id": MODEL_ID, "model_revision": REVISION, "method_id": method, "family": arm["family"], "async_read": bool(arm["async"]),
            "store_kind": arm["kind"], "row_order": ROW_ORDER_ORIGINAL if arm["kind"] == ORIG_KIND else None, "image_id": dialog["image_id"],
            "dialog_id": dialog["dialog_id"], "turn_id": turn, "question_id": str(question["question_id"]),
            "question": question["question"], "gold": question["answers"], "prediction": result["answer"],
            "score": strict_score(result["answer"], question["answers"][0]), "quality_metric": "strict normalized EM",
            "generated_token_ids": list(result["generated_token_ids"]),
            "prompt": diag["prompt"], "history": diag["history"], "history_entries": diag["history_entries"],
            "prompt_sha256": hashlib.sha256(diag["prompt"].encode()).hexdigest(),
            "tokenization": tokenization, "tokenizer_input": diag["tokenizer_input"],
            "official_token_ids_match": bool(match), "input_token_count": len(path_ids), "official_token_count": len(official),
            "input_ids_sha256": canonical_hash(path_ids), "official_ids_sha256": canonical_hash(official),
            "suffix_input_ids": suffix,
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
    """The per-image audit of scripts/209 (FullLoad, IMPRESS: scripts/196; Ours, original row order:
    scripts/202) for the MT-GQA lifecycle (several dialogues per image, strict normalized EM), plus the
    token-sequence checks of this run: the recorded tokenizer input is the phase's rendering of the
    prompt, the recorded suffix ids are the fast tokenizer's ids of it, and -- under "official" -- the
    ids equal the original-LLaVA ids of the prompt (re-derived here with independently loaded tokenizers)."""
    methods = config["phases"][phase]["methods"]
    tokenization = config["phases"][phase]["tokenization"]
    fast, slow, image_token = tokenizers()
    c = Checks()
    ids = set(manifest["phase_dialogue_ids"][phase])
    ds = [d for d in manifest["dialogues"] if d["image_id"] == image_id and d["dialog_id"] in ids]
    c.require(bool(ds), "known_image", image_id)
    if not ds:
        return dict(c.result(), image_id=image_id, phase=phase, requests=len(rows), expected_requests=0)
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
            a = json.loads((artifact / f"{kind}_activation.json").read_text())
            c.require(a["event_kind"] == "activation" and a["outer_wall_ms"] > 0 and a["kind"] == kind, "actual_activation_event", kind)
            n = meta["v_token_num"] - len(meta["newline_idx"]); width = meta["num_heads"] * meta["head_dim"] * 2
            vn, nl = meta["v_token_num"], meta["num_layers"]
            c.require(meta.get("dataset") == DATASET and meta.get("run_schema") == SCHEMA and width == ROW_BYTES and nl == NUM_LAYERS,
                      "store_dataset_and_geometry", kind)
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
        c.require(r["score"] == strict_score(r["prediction"], r["gold"][0]), "strict_em_recalculation", rid)
        c.require(r.get("generated_token_ids") == result["generated_token_ids"], "generated_ids_copied", rid)
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
        # ---- token sequence (this run)
        text = TOK.tokenizer_input(prompt, tokenization)
        c.require(r.get("tokenization") == tokenization and r.get("tokenizer_input") == text, "tokenizer_input_is_the_phase_rendering_of_the_prompt", rid)
        full = fast(text).input_ids
        position = full.index(image_token) if full.count(image_token) == 1 else -1
        c.require(position >= 0 and full[position + 1:] == r["suffix_input_ids"], "recorded_suffix_ids_are_the_ids_of_the_tokenizer_input", rid)
        path_ids = TOK.collapse_image(full, image_token); official = TOK.official_token_ids(slow, prompt)
        c.require(r.get("input_ids_sha256") == canonical_hash(path_ids) and r.get("official_ids_sha256") == canonical_hash(official)
                  and r.get("official_token_ids_match") == (path_ids == official), "token_id_record_recomputed", rid)
        if tokenization == TOK.TOKENIZATION_OFFICIAL:
            c.require(path_ids == official, "request_ids_equal_original_llava_ids", rid)
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
    # recorded, not a commit condition (gate 6 judges it on the gate and DEV images)
    t1 = {}
    for (did, m, t), r in bykey.items():
        if t == 1:
            t1.setdefault(did, set()).add(tuple(r["result"]["generated_token_ids"]))
    return dict(c.result(), image_id=image_id, phase=phase, requests=len(rows), expected_requests=len(expected), dialogs=len(ds),
                tokenization=tokenization, t1_tokens_identical_across_arms=all(len(v) == 1 for v in t1.values()),
                dialogues_with_t1_mismatch=sum(len(v) != 1 for v in t1.values()))


# ---------------------------------------------------------------- lifecycle
def completed_images(run, phase, config):
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

    Dialogues run one after another, so the complete ones are a prefix of the image's dialogue
    list; a dialogue is complete when every (arm, turn) row of it was written."""
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
    """Stores of an interrupted image against the hashes recorded when they were built.  Returns (ok, reason)."""
    if not scratch.is_dir() or scratch.is_symlink():
        return False, "scratch directory missing"
    if sorted(p.name for p in scratch.iterdir()) != sorted(all_kinds):
        return False, "scratch does not hold exactly the image's stores"
    for kind in all_kinds:
        names = [f"{kind}_persistence.json", f"{kind}_meta.json", f"{kind}_hashes.json", f"{kind}_activation.json"]
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
        if kind == NEW_KIND:      # the turn-1 statistics the IMPRESS layout was built from (reloaded for the report's computed comparison)
            t1 = p.get("official_turn1", {})
            stats = artifact / t1.get("stats_file", "missing")
            if not stats.is_file() or sha(stats) != t1.get("stats_sha256"):
                return False, f"{kind}: turn-1 statistics file differs from the recorded hash"
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
    interrupted = dialogs[n]["dialog_id"] if n < len(dialogs) else None
    # restart_dialogues: what is (still) to be executed; rerun_dialogues: the ones that had started before the interruption and run again
    if ok:
        return dict(base, mode="continue", attempt=last.name, n_complete=n, kept_rows=kept, dropped_rows=dropped,
                    restart_dialogues=[d["dialog_id"] for d in dialogs[n:]], rerun_dialogues=[interrupted] if interrupted else [],
                    interrupted_dialogue=interrupted)
    return dict(base, mode="fresh", attempt=f"attempt_{len(attempts) + 1:04d}", discard_attempt=last.name, discard_scratch=str(scratch),
                restart_dialogues=[d["dialog_id"] for d in dialogs], rerun_dialogues=[d["dialog_id"] for d in dialogs[:n + 1]],
                interrupted_dialogue=interrupted)


def run_image(runner, servers, dialogs, ordinals, phase, run, manifest, config, tick=None):
    methods = config["phases"][phase]["methods"]; docs = config["layer_budgets"]
    tokenization = config["phases"][phase]["tokenization"]
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
        resume_event = {"at_unix": time.time(), "at_utc": STATUS.utc(time.time()), "phase": phase, "image_id": image_id, "mode": plan["mode"], "attempt": attempt,
                        "previous_attempt": plan["previous_attempt"], "rows_found": plan["rows_found"], "torn_lines": plan["torn_lines"],
                        "complete_dialogues_kept": plan.get("n_complete", 0), "store_check": plan["store_check"],
                        "interrupted_dialogue": plan["interrupted_dialogue"], "restart_dialogues": plan["restart_dialogues"],
                        "rerun_dialogues": plan["rerun_dialogues"], "process_id": os.getpid(), "boot_id": STATUS.boot_id()}
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
        if NEW_KIND in all_kinds:      # per-image state of the imported harness: the order scores of the store built before the interruption
            stats = np.load(artifact / persistence[NEW_KIND]["official_turn1"]["stats_file"])
            E.TURN1_ORDER_SCORES[str(paths[NEW_KIND])] = torch.from_numpy(stats["order_scores"])
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
    source_image_sha = sha(ROOT / dialogs[0]["image_path"])
    if not (artifact / "image_identity.json").exists():
        atomic_json(artifact / "image_identity.json", {"image_id": image_id, "image_path": str((ROOT / dialogs[0]["image_path"]).resolve()), "sha256": source_image_sha})
    started_unix = time.time()
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
                    factory = PromptFactory(methods, dialog, turn, method, generated[method], tokenization)
                    kind = kinds.get(method, "none")
                    # the first arm of a store kind in the first dialogue's rotation builds that kind's store
                    capture = kind if di == 0 and turn == 1 and kind != "none" and kind not in capturing else "none"
                    if capture != "none":
                        capturing.add(capture)
                    if turn == 1 or method == RECOMP:
                        result, diag = pixel_request(runner, servers, ROOT / dialog["image_path"], factory, capture)
                        append(artifact / "execution_events.jsonl", dict(source, turn_id=turn, status="EXECUTED", prediction=result["answer"],
                                                                        generated_token_ids=result["generated_token_ids"], ttft_ms=result["end_to_end_ttft_ms"],
                                                                        request_e2e_ms=result["request_e2e_ms"]))
                        if capture != "none":
                            start = time.perf_counter()
                            p = N.persist(runner, result, diag, paths[capture], image_id, capture, source, docs, artifact / f"{capture}_turn1_stats.npz")
                            outer_ms = (time.perf_counter() - start) * 1000
                            persistence[capture] = dict(p, outer_helper_wall_ms=outer_ms, source=source,
                                                        capture_accounting="capture and exit materialization already within T1 request E2E; not added again")
                            persistence[capture].pop("meta", None)
                            metas[capture] = json.loads((paths[capture] / "meta.json").read_text())
                            # the completion scan of persist_llava hashed every payload file once; that inventory is the store's identity
                            hashes[capture] = {n: f["sha256"] for n, f in p["files"].items() if n != "integrity.json"}
                            store_info[capture] = {"kind": capture, "path": str(paths[capture]), "meta_sha256": p["files"]["meta.json"]["sha256"],
                                                   "source": source, "physical_build_event_id": p["event_id"]}
                            atomic_json(artifact / f"{capture}_persistence.json", persistence[capture])
                            compact_json(artifact / f"{capture}_meta.json", metas[capture])
                            atomic_json(artifact / f"{capture}_hashes.json", hashes[capture])
                            peak_scratch = max(peak_scratch, scratch_bytes(scratch))
                        result.pop("captured_past_key_values", None)
                        saved = close_diag(runner, diag, factory)
                        if di == 0 and turn == 1:
                            pending.append((method, result, saved, physical))
                            del diag, result
                            continue
                        diag = saved
                    else:
                        if kind not in contexts:
                            contexts[kind] = N.activate(runner, kind, paths[kind])
                            name = f"{kind}_activation.json"
                            if (artifact / name).exists():      # an activation after a resume is a second event of the same store
                                name = f"{kind}_activation_resume_{len(list(artifact.glob(kind + '_activation*.json'))):02d}.json"
                            atomic_json(artifact / name, contexts[kind].activation)
                        result, diag = N.stored_request(runner, servers, contexts[kind], factory, method, image_id, paths[kind])
                        diag = close_diag(runner, diag, factory)
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
                    pending.clear()
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
                  "tokenization": tokenization,
                  "t1_tokens_identical_across_arms": audit["t1_tokens_identical_across_arms"],
                  "resumed_inside_image": len(list(artifact.glob("resume_*.json"))),
                  "image_seconds_this_process": time.time() - started_unix, "committed_at_unix": time.time(), "process_id": os.getpid()}
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
        contexts.clear()
        for kind in all_kinds:      # per-image state of the imported harness
            E.TURN1_ORDER_SCORES.pop(str(paths[kind]), None)
        pending.clear(); gc.collect()
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
        if not ("_dev_" in run.name and os.environ.get("MTGQA_V2_DEV_ALLOW_SOURCE_DRIFT") == "1"):
            raise RuntimeError("FROZEN_SOURCE_CONFIG_MISMATCH: " + repr(changed[:10]))
        config["source_sha256"] = source_hashes()
    if sha(CONTRACT) != config["contract_sha256"] and "_dev_" not in run.name:
        raise RuntimeError("FROZEN_CONTRACT_MISMATCH")
    if sha(run / "manifest.json") != config["manifest_sha256"] or sha(run / "storage_plan.json") != config["storage_plan_sha256"]:
        raise RuntimeError("FROZEN_MANIFEST_OR_STORAGE_PLAN_MISMATCH")
    manifest = json.loads((run / "manifest.json").read_text())
    if frozen_dialogues() != manifest["dialogues"]:
        raise RuntimeError("FROZEN_DATASET_MISMATCH")
    target = run / "vectors" / VECTOR_SOURCE.name
    doc = load_budget_document(target)
    frozen = config["layer_budgets"][VECTOR]
    if (sha(target) != VECTOR_FILE_SHA or doc["layer_budget_sha256"] != VECTOR_RATIO_SHA or doc["layer_keep_ratios"] != frozen["layer_keep_ratios"]
            or frozen["layer_budget_sha256"] != VECTOR_RATIO_SHA):
        raise RuntimeError("FROZEN_VECTOR_MISMATCH")
    if (config["model"] != MODEL_ID or config["revision"] != REVISION or config["phase_methods"] != PHASE_METHODS
            or {ph: config["phases"][ph]["tokenization"] for ph in PHASES} != PHASE_TOKENIZATION):
        raise RuntimeError("FROZEN_ARMS_OR_MODEL_MISMATCH")
    if transformers.__version__ != config["environment"]["transformers"] or torch.__version__ != config["environment"]["torch"]:
        raise RuntimeError("FROZEN_ENVIRONMENT_MISMATCH")
    config.update(_file_sha256=sha(run / "config.json"), _manifest_sha256=sha(run / "manifest.json"))
    manifest["_ordinals"] = {d["dialog_id"]: i for i, d in enumerate(manifest["dialogues"])}
    return config, manifest


def committed_request_counts(run):
    out = {}
    for ph in PHASES:
        out[ph] = sum(json.loads(cp.read_text())["requests"] for cp in (run / ph / "images").glob("*/COMMITTED.json")) if (run / ph).exists() else 0
    return out


def measured_rates(run):
    """seconds per request each finished phase measured (written by the phase when it completes)."""
    try:
        return json.loads((run / "phase_rates.json").read_text())
    except (OSError, ValueError):
        return {}


def run_phase(runner, servers, run, phase, config, manifest):
    done = completed_images(run, phase, config)
    ids = manifest["phase_dialogue_ids"][phase]
    keep = set(ids); groups = OrderedDict(); ordinals = manifest["_ordinals"]
    for d in manifest["dialogues"]:                      # manifest order, as the previous MT-GQA runs
        if d["dialog_id"] in keep:
            groups.setdefault(d["image_id"], []).append(d)
    assert sum(len(v) for v in groups.values()) == len(ids)
    per_dialog = 3 * len(config["phases"][phase]["methods"])
    expected = len(ids) * per_dialog; completed = sum(x["requests"] for x in done.values())
    assert expected == manifest["expected_requests"][phase]
    others = committed_request_counts(run)
    total_expected = sum(manifest["expected_requests"].values())
    started = time.time(); completed_at_start = completed
    rates = measured_rates(run)

    def status(in_image=0, image=None):
        now = time.time(); done_now = completed + in_image
        new = done_now - completed_at_start
        rate = (now - started) / new if new > 0 else None
        use = rate if rate else SECONDS_PER_REQUEST_PRIOR[phase]
        later = [p for p in PHASES[PHASES.index(phase) + 1:]]
        # a later phase with the main arms runs at the rate the DEV pilot (or this phase, when it has the same arms) measured
        same_arms = rate if (rate and config["phases"][phase]["methods"] == MAIN_METHODS) else rates.get("dev")
        remaining_later = sum((manifest["expected_requests"][p] - others[p])
                              * ((same_arms if (same_arms and PHASE_METHODS[p] == MAIN_METHODS) else None) or SECONDS_PER_REQUEST_PRIOR[p]) for p in later)
        ssd = STATUS.ssd_counter(run)
        STATUS.update(run, stage=f"{phase}_phase", phase=phase, phase_completed_requests=done_now, phase_expected_requests=expected,
                      completed_requests=sum(v for k, v in others.items() if k != phase) + done_now, expected_requests=total_expected,
                      main_completed_requests=(done_now if phase == "main" else others["main"]), main_expected_requests=manifest["expected_requests"]["main"],
                      phase_images_committed=len(done), phase_images=len(groups), current_image=image,
                      seconds_per_request_this_process=rate,
                      phase_eta_utc=STATUS.utc(now + use * (expected - done_now)),
                      eta_utc=STATUS.utc(now + use * (expected - done_now) + remaining_later + 900),
                      eta_note="rate of this process for this phase; later request phases at the rate the DEV pilot measured (before that, at the rate of the "
                               "2026-11-17 run); + 15 min for the gate checks and the report",
                      ssd_written_bytes=ssd.get("written_bytes"), ssd_written_gb=round((ssd.get("written_bytes") or 0) / 1e9, 2),
                      ssd_written_note="device-wide kernel counter since the driver first started (all processes); not NAND writes",
                      ssd_temperature_c=STATUS.ssd_temperature()["composite_c"], free_gib=round(shutil.disk_usage(run).free / GiB, 1),
                      python_pid=os.getpid())

    status()
    STATUS.ssd_sample(run, f"{phase}_process_start", phase=phase, committed_images=len(done))
    for image in done:      # a commit that was interrupted before its scratch cleanup: the stores of a committed image are not needed
        leftover = run / "scratch" / phase / image
        if leftover.exists():
            append(run / "resume_log.jsonl", {"at_unix": time.time(), "at_utc": STATUS.utc(time.time()), "phase": phase, "image_id": image,
                                              "mode": "committed_scratch_removed", "discarded_scratch_bytes": remove_scratch(run, leftover),
                                              "restart_dialogues": [], "process_id": os.getpid(), "boot_id": STATUS.boot_id()})
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
            append(run / "environment_log.jsonl", dict(N.environment_sample(), phase=phase, completed_images=len(done),
                                                      free_bytes=shutil.disk_usage(run).free))
    assert completed == expected, (completed, expected)
    new = completed - completed_at_start
    if new > 0:
        rates = measured_rates(run)
        previous = rates.get(phase)
        rates[phase] = (time.time() - started) / new if previous is None else previous      # the first (longest) process of the phase
        atomic_json(run / "phase_rates.json", rates, replace=True)
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
    prior = json.loads((PRIOR / "manifest.json").read_text())
    assert prior["dialogues"] == ds
    for image, identity in prior["image_identity"].items():
        assert sha(ROOT / identity["image_path"]) == identity["sha256"], image
    refs = {"impl_20261117": json.loads((IMPL_REF_RUN / "manifest.json").read_text()), "ours_20261120": json.loads((OURS_REF_RUN / "manifest.json").read_text())}
    for name, ref in refs.items():      # the two earlier main runs: same dialogues, same images, same gate / DEV images
        assert ref["dialogues"] == ds and ref["image_identity"] == prior["image_identity"], name
        assert ref["index_sha256"] == INDEX_SHA and ref["workload_sha256"] == WORKLOAD_SHA, name
    smoke_images = refs["ours_20261120"]["phase_images"]["gate"]
    dev_images = json.loads(V2_SPLIT.read_text())["dev"]["image_ids"]
    all_images = {d["image_id"] for d in ds}
    assert len(dev_images) == 40 == len(set(dev_images)) and set(dev_images) <= all_images and len(set(smoke_images)) == 4 == len(smoke_images)
    for ref in refs.values():
        assert ref["phase_images"]["gate"] == smoke_images and ref["phase_images"]["dev"] == dev_images
    image_sets = {"gate": set(smoke_images), "dev": set(dev_images), "main": all_images}
    if limit:
        first = list(dict.fromkeys(d["image_id"] for d in ds))
        image_sets = {"gate": set([i for i in first if i in image_sets["gate"]][:limit]),
                      "dev": set([i for i in first if i in image_sets["dev"]][:limit]), "main": set(first[:limit + 1])}
    phase_ids = {ph: [d["dialog_id"] for d in ds if d["image_id"] in image_sets[PHASE_IMAGES[ph]]] for ph in PHASES}
    order = list(dict.fromkeys(d["image_id"] for d in ds))                 # execution order: first appearance in the manifest
    phase_images = {ph: [i for i in order if i in image_sets[PHASE_IMAGES[ph]]] for ph in PHASES}
    manifest = {"dataset": DATASET, "schema_version": SCHEMA, "index_sha256": INDEX_SHA, "workload_sha256": WORKLOAD_SHA,
                "dialogues": ds, "image_identity": prior["image_identity"], "phase_dialogue_ids": phase_ids, "phase_images": phase_images,
                "phase_methods": PHASE_METHODS, "phase_tokenization": PHASE_TOKENIZATION,
                "expected_requests": {p: len(phase_ids[p]) * 3 * len(PHASE_METHODS[p]) for p in PHASES},
                "prior_manifest_sha256": sha(PRIOR / "manifest.json"),
                "reference_manifest_sha256": {"impl_20261117": sha(IMPL_REF_RUN / "manifest.json"), "ours_20261120": sha(OURS_REF_RUN / "manifest.json")},
                "gate_policy": "every dialogue of the four gate images of the 2026-11-17 and 2026-11-20 runs",
                "dev_policy": "every MT-GQA dialogue of the 40 DEV images of results/layer_budget_kv25_v2_20261106T073502Z/data_split.json",
                "main_policy": "all 398 images / 4,061 dialogues, manifest order"}
    assert limit or manifest["expected_requests"] == EXPECTED_REQUESTS, manifest["expected_requests"]
    if limit:
        manifest["rehearsal_image_limit"] = limit
    return manifest


def reference_store_bytes(images):
    """Per image, the store sizes the 2026-11-17 main run measured (FullLoad store, IMPRESS store, packed store)."""
    out = {}
    for image in images:
        doc = json.loads((IMPL_REF_RUN / "main" / "images" / image / "COMMITTED.json").read_text())
        usage = json.loads((IMPL_REF_RUN / "main" / "images" / image / doc["attempt_id"] / "disk_usage.json").read_text())["store_bytes"]
        out[image] = {"raster": usage["raster"], NEW_KIND: usage[NEW_KIND], "packed": usage[E.PACKED_KIND]}
    return out


def reference_seconds_per_request():
    """Wall seconds per request of the 2026-11-17 main phase (same four-arm composition, IMPRESS included)."""
    starts = sorted(IMPL_REF_RUN.glob("main_start_*.json"))
    done = json.loads((IMPL_REF_RUN / "main_execution_complete.json").read_text())
    first = json.loads(starts[0].read_text())["started_unix"]
    return (done["finished_unix"] - first) / done["requests"], len(starts)


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
    assert impl["max_new_tokens"] == ours["max_new_tokens"] == 16 and impl["seed"] == ours["seed"] == 1234
    for m in (ORIG, ORIG_ASYNC, ORIG_REPEAT):
        assert {k: v for k, v in ours["arms"][m].items() if k != "display"} == {k: v for k, v in ARMS[m].items() if k != "display"}, m
    for m in (RECOMP, FULL_ASYNC, NEW, NEW_ASYNC, NEW_R100, FULL_REORDERED):
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
    limit = int(os.environ.get("MTGQA_V2_DEV_LIMIT", "0")) if "_dev_" in run.name else 0
    manifest = build_manifest(limit)
    support.publish(run / "manifest.json", manifest)
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
    # ---- storage plan: store sizes the 2026-11-17 main run measured on the same images
    images = sorted({i for ph in PHASES for i in manifest["phase_images"][ph]})
    size = reference_store_bytes(images)
    write = {"raster": lambda i: size[i]["raster"], NEW_KIND: lambda i: 2 * size[i][NEW_KIND], ORIG_KIND: lambda i: size[i]["packed"]}
    writes = {ph: int(sum(write[k](i) for i in manifest["phase_images"][ph] for k in store_kinds(PHASE_METHODS[ph]))) for ph in PHASES}
    writes["total"] = sum(writes.values())
    bound = N.store_bytes(2880, 48)                      # the largest geometry LLaVA-NeXT's AnyRes produces here
    free = shutil.disk_usage(run).free
    plan = {"policy": "FRESH_IMAGE_STREAMING (image-major): one live image; stores built from the arms' own T1 of the image's first dialogue -> "
                      "served -> audited -> committed -> run-local scratch deleted by exact SHA256 allowlist",
            "minimum_reserve_bytes": 30 * GiB, "peak_working_set_bytes": bound["bundle"] + 4 * GiB, "largest_image_bundle_bound_bytes": bound["bundle"],
            "maximum_new_payload_bytes": 20 * GiB, "raw_and_evidence_growth_bound_bytes": 16 * GiB,
            "expected_ssd_write_bytes": writes,
            "expected_ssd_write_note": "per image: FullLoad store + IMPRESS store (written, then rewritten in the reordered layout) + Ours store; the devgate "
                                       "phase has no FullLoad store; sizes from the 2026-11-17 main run's disk_usage.json of the same image (the Ours store has "
                                       "the size of that run's packed store); raw rows and evidence not included",
            "free_bytes_at_freeze": free,
            "deletion_scope": "runs/<RUN>/scratch only (exact SHA allowlist after audit+commit; the stores of an interrupted image that cannot be "
                              "verified on resume); existing stores never deleted"}
    plan["required_free_bytes"] = plan["minimum_reserve_bytes"] + plan["peak_working_set_bytes"] + plan["raw_and_evidence_growth_bound_bytes"]
    plan["status"] = "PASS" if free >= plan["required_free_bytes"] else "BLOCKED_STORAGE"
    support.publish(run / "storage_plan.json", plan)
    if plan["status"] != "PASS":
        raise RuntimeError(f"BLOCKED_STORAGE: free {free} < required {plan['required_free_bytes']}")
    config = {"schema_version": SCHEMA, "dataset": DATASET, "model": MODEL_ID, "revision": REVISION, "seed": 1234, "greedy": True, "max_new_tokens": 16,
              "batch_size": 1, "cpu_threads": 2, "history_policy": "method_local_generated", "metric": "strict normalized EM",
              "phase_methods": PHASE_METHODS,
              "phases": {ph: {"methods": PHASE_METHODS[ph], "dialogues": len(manifest["phase_dialogue_ids"][ph]),
                              "images": len(manifest["phase_images"][ph]), "tokenization": PHASE_TOKENIZATION[ph]} for ph in PHASES},
              "tokenization": {"module": "mmimpress/mt_gqa_tokens.py", "module_sha256": sha(ROOT / "mmimpress/mt_gqa_tokens.py"),
                               "official": "tokenizer input = prompt with one space after <image>; the fast tokenizer then emits the original-LLaVA ids "
                                           "(tokenizer_image_token, slow legacy tokenizer); compared after every request",
                               "legacy": "tokenizer input = prompt (the path of every LLaVA-NeXT MT-GQA run up to 2026-11-22)",
                               "after_image_official": TOK.OFFICIAL_AFTER_IMAGE},
              "arms": {m: dict(a) for m, a in ARMS.items()}, "main_methods": MAIN_METHODS,
              "server_args": SERVER_ARGS, "impress_official": OFFICIAL, "readahead": READAHEAD,
              "official_checkout": {"path": str(Path(checkout).resolve()) if checkout else None, "head": official_head,
                                    "repository": OFFICIAL["reference_repository"]},
              "store_kinds": {"raster": "full visual KV, raster order (FullLoad)", NEW_KIND: "full visual KV + probe key copy, per-layer reordered (IMPRESS)",
                              ORIG_KIND: "packed, PYR-beta-raw 25 %, row_order=original (Ours)"},
              "store_format_ours": ORIGINAL_STORE_FORMAT, "layer_budgets": budgets, "vector_file_sha256": VECTOR_FILE_SHA,
              "arm_order": "cyclic rotation by the dialogue's ordinal in the dataset (ordinal % number of arms of the phase)",
              "source_sha256": now, "source_difference_from_reference_freezes": diffs,
              "implementation_reference": {
                  "fullload_impress": {"run": str(IMPL_REF_RUN.relative_to(ROOT)), "config_sha256": sha(IMPL_REF_RUN / "config.json"),
                                       "harness": "scripts/196_eval_mt_gqa_impress_code.py (imported through scripts/209)",
                                       "harness_sha256": sha(ROOT / "scripts/196_eval_mt_gqa_impress_code.py")},
                  "ours": {"run": str(OURS_REF_RUN.relative_to(ROOT)), "config_sha256": sha(OURS_REF_RUN / "config.json"),
                           "harness": "scripts/202_eval_mt_gqa_ours_order.py (the same writer and request calls, through scripts/209)",
                           "harness_sha256": sha(ROOT / "scripts/202_eval_mt_gqa_ours_order.py"), "method_id": ORIG_ASYNC},
                  "backend": {"harness": "scripts/209_eval_llava_next_mtvqa_main.py (imported)",
                              "harness_sha256": sha(ROOT / "scripts/209_eval_llava_next_mtvqa_main.py")}},
              "previous_results": {RECOMP: str(OURS_REF_RUN.relative_to(ROOT)), FULL_ASYNC: str(OURS_REF_RUN.relative_to(ROOT)),
                                   ORIG_ASYNC: str(OURS_REF_RUN.relative_to(ROOT)), NEW_ASYNC: str(IMPL_REF_RUN.relative_to(ROOT))},
              "contract_sha256": sha(CONTRACT),
              "manifest_sha256": sha(run / "manifest.json"), "storage_plan_sha256": sha(run / "storage_plan.json"),
              "source_T1_capture": "the image's first dialogue; per store kind, the first arm of that kind in the rotation; every arm still runs its own full-pixel T1 in every dialogue",
              "resume_policy": "committed images skipped; inside an interrupted image complete dialogues are kept and the interrupted dialogue restarts "
                               "at turn 1; unverifiable stores -> the image restarts",
              "environment": {"python": sys.version, "executable": sys.executable, "torch": torch.__version__, "transformers": transformers.__version__,
                              "numpy": np.__version__},
              "lifecycle_source": "scripts/156 + 164 helpers, scripts/196 and scripts/209 (imported, not copied)"}
    support.publish(run / "config.json", config)
    shutil.copy2(CONTRACT, run / "CONTRACT.md")
    envdoc = {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
              "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
              "pip_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)}
    support.publish(run / "environment_before.json", envdoc)
    support.publish(run / "protection_before.json", stat_inventory(run, run / "protection_before.tsv.gz"))
    # ---- results/<run name>/prerun.json: what is expected, written before any GPU measurement
    per_request, ref_starts = reference_seconds_per_request()
    seconds = {ph: manifest["expected_requests"][ph] * (per_request if PHASE_METHODS[ph] == MAIN_METHODS else SECONDS_PER_REQUEST_PRIOR[ph]) for ph in PHASES}
    out_root = ROOT / "results" / run.name; out_root.mkdir(parents=True, exist_ok=True)
    prerun = {"written_before_any_gpu_measurement_utc": STATUS.utc(time.time()), "run": run.name,
              "expected_requests": manifest["expected_requests"], "expected_requests_total": sum(manifest["expected_requests"].values()),
              "expected_ssd_write": {"bytes": writes, "total_tb": writes["total"] / 1e12, "main_tb": writes["main"] / 1e12, "note": plan["expected_ssd_write_note"]},
              "expected_duration": {"seconds_per_request_reference": per_request,
                                    "basis": f"wall time of the 2026-11-17 main phase / its 48,732 requests ({ref_starts} process start(s)), applied to the phases "
                                             "with the four main arms; the gate arms use a rounded prior; unit tests (two environments) and gate checks add about 30 min",
                                    "phase_hours": {ph: seconds[ph] / 3600 for ph in PHASES},
                                    "total_hours": sum(seconds.values()) / 3600 + 0.5},
              "free_space": {"free_bytes": free, "free_gib": free / GiB, "required_free_bytes": plan["required_free_bytes"],
                             "largest_image_bundle_bytes": bound["bundle"]},
              "smart": STATUS.smart_snapshot(), "ssd_counters_at_freeze": STATUS._safe(STATUS.sectors), "ssd_temperature_at_freeze": STATUS.ssd_temperature()}
    tmp = out_root / ".prerun.json.tmp"
    tmp.write_text(json.dumps(prerun, indent=1, ensure_ascii=False) + "\n"); os.replace(tmp, out_root / "prerun.json")
    STATUS.update(run, expected_requests=prerun["expected_requests_total"], completed_requests=0, main_expected_requests=manifest["expected_requests"]["main"],
                  expected_ssd_write_tb=round(writes["total"] / 1e12, 3), expected_total_hours=round(prerun["expected_duration"]["total_hours"], 1),
                  eta_utc=STATUS.utc(time.time() + sum(seconds.values()) + 1800), eta_note="pre-run estimate (see prerun.json)")
    print(json.dumps({"freeze": "PASS", "run": str(run), "expected": manifest["expected_requests"], "storage": plan["status"],
                      "free_gib": free / 2**30, "expected_ssd_write_tb": writes["total"] / 1e12, "expected_hours": prerun["expected_duration"]["total_hours"],
                      "transformers": transformers.__version__, "source_difference": {k: v["changed"] for k, v in diffs.items()}}), flush=True)


def load_model(config):
    """The loading sequence of scripts/209 (= the reference runs)."""
    runner, servers = N.load_model(config)
    fast, slow, image_token = tokenizers()
    assert image_token == runner.image_token_id
    probe = "USER: <image>\nCurrent question Q1: What is this?\n" + SHORT_ANSWER + " ASSISTANT:"
    for mode in TOK.TOKENIZATIONS:      # the independently loaded fast tokenizer is the one the runner's processor holds
        text = TOK.tokenizer_input(probe, mode)
        assert runner.processor.tokenizer(text).input_ids == fast(text).input_ids
    return runner, servers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True, help="runs/mt_gqa_main_v2_<UTC>")
    p.add_argument("--stage", choices=("freeze",) + PHASES, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    run = args.run.resolve()
    if args.stage == "freeze":
        return freeze(run)
    assert run.parent == ROOT / "runs" and run.name.startswith(RUN_PREFIX)
    config, manifest = verify_freeze(run)
    assert json.loads((run / "storage_plan.json").read_text())["status"] == "PASS"
    phase = args.stage
    if phase == "main":      # the main run is admitted only by the recorded gate decision
        admit = json.loads((run / "gates" / "admission.json").read_text())
        if admit["status"] != "PASS" or admit["config_sha256"] != config["_file_sha256"]:
            raise RuntimeError("MAIN_NOT_ADMITTED")
    if (run / phase).exists() and not args.resume:
        raise FileExistsError("use --resume; previous artifacts preserved")
    assert_gpu_exclusive()
    runner, servers = load_model(config)
    try:
        atomic_json(run / f"{phase}_start_{time.strftime('%Y%m%dT%H%M%S')}.json",
                    {"started_unix": time.time(), "capacity": capacity_guard(run), "pid": os.getpid(), "resume": bool(args.resume),
                     "boot_id": STATUS.boot_id(), "committed_requests_at_start": committed_request_counts(run)[phase],
                     "tokenization": config["phases"][phase]["tokenization"], "ssd_temperature": STATUS.ssd_temperature()})
        receipt = run_phase(runner, servers, run, phase, config, manifest)
        atomic_json(run / f"{phase}_execution_complete.json", {"status": "PASS", **receipt, "finished_unix": time.time()}, replace=True)
    except BaseException as exc:
        error = {"status": "PARTIAL", "error": repr(exc), "traceback": traceback.format_exc(), "at_unix": time.time(),
                 "resume_command": f"{DRIVER} {run.relative_to(ROOT)}"}
        atomic_json(run / ("interruption_" + phase + "_" + time.strftime("%Y%m%dT%H%M%S") + ".json"), error)
        STATUS.update(run, last_error={"stage": f"{phase}_phase", "error": repr(exc)[:800], "at_utc": STATUS.utc(time.time())})
        if any(marker in repr(exc) for marker in NO_RETRY_MARKERS):      # a restart would stop at the same place: the driver does not restart the phase
            atomic_json(run / f"NO_RETRY_{phase}.json", {"error": repr(exc)[:2000], "at_unix": time.time()}, replace=True)
        print(json.dumps({k: v for k, v in error.items() if k != "traceback"}), flush=True)
        raise
    finally:
        for server in servers.values():
            server.close()


if __name__ == "__main__":
    main()

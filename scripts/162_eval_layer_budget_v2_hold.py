#!/usr/bin/env python3
"""Phase 4 (v2) — HOLD main for packed PrefixStore-KV25 (contract v2, A5/A8).

Same lifecycle as the v1 HOLD runner (scripts/156): MT-GQA-reconstructed
398 images / 4,061 dialogues x 3 turns, generated method-local history,
strict normalized EM, image-atomic attempts, independent per-image audit,
atomic commit, exact-allowlist cleanup, one live image at a time.  Arms come
from ``dev_selection.json``: recompute, fullload and
``ours_prefixstore_kv25_<vec>_packed`` for vec in U, A1-2, VLC_offline,
SA_offline, PYR (seven arms).  Every method executes its own full-pixel T1
for every dialogue; the first dialogue's T1 of each stored method builds that
method's store.  Packed hits are served by ``mmimpress.packed_serve`` with one
``preadv`` per hit; nothing of v1 or the prior B main is reused as a number.

Stages: --stage freeze | --stage smoke | --stage main [--resume]
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
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mmimpress.layer_budget import layer_keep_counts, load_budget_document
from mmimpress.packed_serve import PackedImageContext, PackedReadTrace, request_packed
from mmimpress.packed_store import PACKED_NAME, STORE_FORMAT, persist_captured_packed_prefix

SCHEMA = "llava-mtgqa-layerbudget-v2-hold-v1"
MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"
REVISION = "c916e6cdcd760b4cecd1dd4907f84ac649f93b23"
INDEX_SHA = "2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224"
WORKLOAD_SHA = "0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62"
PRIOR = ROOT / "runs/mt_gqa_B_llava5_qwen3_20261001T093921Z/llava"
CONTRACT = ROOT / "docs/layer_budget_kv25_v2_contract.md"
B_CONTRACT = ROOT / "docs/mt_gqa_B_main_contract.md"
SHORT_ANSWER = "Answer the current question with a single word or short phrase."
PK = "ours_prefixstore_kv25_"
PK_SUFFIX = "_packed"


def helper(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


V = helper("97_validate_sparsevlm_ssd_kv25.py", "_hold_allhead_gpu_helpers")
QA = helper("49_eval_query_aware_baseline.py", "_hold_common_io")
H1 = helper("156_eval_layer_budget_hold.py", "_hold_v1_lifecycle")
import main_b_support as support

sha, canonical_hash, safe, sync_dir, atomic_json, append, load_jsonl = (H1.sha, H1.canonical_hash, H1.safe, H1.sync_dir, H1.atomic_json, H1.append, H1.load_jsonl)
frozen_dialogues, method_order, strict_score, assert_gpu_exclusive, conditioning_trace, PilotCounts = (
    H1.frozen_dialogues, H1.method_order, H1.strict_score, H1.assert_gpu_exclusive, H1.conditioning_trace, H1.PilotCounts)
normal_request, file_hashes, Checks, event_key, capacity_guard, cleanup_committed = (
    H1.normal_request, H1.file_hashes, H1.Checks, H1.event_key, H1.capacity_guard, H1.cleanup_committed)


def request_id(run_id, phase, method, dialog_id, turn):
    return f"{SCHEMA}:{run_id}:{phase}:{MODEL_ID}:{method}:{dialog_id}:t{turn}"


def prompt_factory(methods, dialog, turn, method, generated):
    return H1.prompt_factory(methods, dialog, turn, method, generated)


def vector_of(method):
    assert method.startswith(PK) and method.endswith(PK_SUFFIX), method
    return method[len(PK):-len(PK_SUFFIX)]


def kind_of(method):
    if method == "fullload":
        return "raster"
    if method.startswith(PK):
        return "packed_" + vector_of(method)
    return "none"


# ---------------------------------------------------------------- requests
def stored_hit(runner, server, ctx, factory, method, image_id):
    with QA._NoVisionForward(runner) as guard:
        c0 = time.perf_counter(); ctx.reader.drop_all(); conditioning = (time.perf_counter() - c0) * 1000
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
        prompt, history, entries = factory()
        tokenized = runner.processor.tokenizer(prompt, return_tensors="pt")
        suffix = QA._suffix_from_tokenized(runner, tokenized).to(runner.model.device)
        with PilotCounts(runner) as counts, PackedReadTrace() as reads:
            if method == "fullload":
                result = server.request(ctx, mode="fullload", suffix_ids=suffix, cold=False, prompt_text=prompt, cache_id=image_id)
            else:
                result = request_packed(server, ctx, suffix_ids=suffix, cold=False, image_id=image_id, directcopy=False)
        returned = time.perf_counter()
    s = reads.summary()
    assert s["bytes"] == result["io"]["bytes"] and s["preads"] == result["io"]["preads"]
    result.update(projection_calls=counts.result(), scoring_calls=0,
                  end_to_end_ttft_ms=(result["first_token_at_s"] - started) * 1000, request_e2e_ms=(returned - started) * 1000,
                  request_started_at_s=started, vision_forward_count=guard.calls, page_cache_conditioning_ms=conditioning,
                  cap_reached=(len(result["generated_token_ids"]) == 16 and result["generated_token_ids"][-1] != runner.processor.tokenizer.eos_token_id),
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                  os_pread_trace=s["calls"], pread_calls=s["pread_calls"], preadv_calls=s["preadv_calls"])
    ctx.cache.k = None; ctx.cache.v = None
    return result, {"prompt": prompt, "history": history, "history_entries": entries, "suffix_input_ids": suffix.cpu().tolist()}


def stored_request(runner, server, ctx, factory, method, image_id):
    original = [l.self_attn.forward for l in runner.layers]
    with conditioning_trace() as conditioning:
        result, diag = stored_hit(runner, server, ctx, factory, method, image_id)
    assert all(l.self_attn.forward == f for l, f in zip(runner.layers, original)), "adapter not restored"
    from mmimpress.serve import BIAS
    assert not BIAS, "stale mask after request"
    assert conditioning and all(e["status"] == "SUCCESS" for e in conditioning), "page cache conditioning not verified"
    result.update(conditioning_events=conditioning, adapter_restored=True,
                  page_cache_conditioning_method="posix_fadvise_DONTNEED", page_cache_conditioning_excluded_from_ttft=True)
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
        return persist_captured_raster_prefix(*args, diag["hidden"].result_cpu(), path, probe_heads=3,
                                              hidden_capture_stats=diag["hidden"], full_integrity_hash=False, **common)
    vec = kind[len("packed_"):]
    return persist_captured_packed_prefix(*args, diag["vision"].result_cpu(), path, layer_budget=docs[vec],
                                          capture_stats=diag["vision"], full_integrity_hash=False, **common)


def persist(runner, result, diag, path, image_id, kind, source, docs):
    return support.persist_llava(lambda: _native_persist(runner, result, diag, path, image_id, kind, source, docs), path, kind, source)


def activate(runner, method, path):
    started = time.perf_counter()
    if method == "fullload":
        from mmimpress.serve import ImageContext
        ctx = ImageContext(path, runner.model.device, drop_cache=False, require_v_hidden=False)
        completion = support.verify_completion(path, "integrity.json")
        pinned = 0
    else:
        ctx = PackedImageContext(path, runner.model.device)
        ctx.validate_packed_layout()
        completion = ctx.integrity_activation
        pinned = int(ctx.reader.total)
    ctx.activation = {"host_tensor_bytes": sum(t.numel() * t.element_size() for t in ctx.cache.sys_kv.values()),
                      "pinned_host_buffer_bytes": pinned, "gpu_tensor_bytes": 0, "visual_payload_resident_bytes": 0,
                      "file_bytes": sum((path / n).stat().st_size for n in ("meta.json", "sys_kv.pt")),
                      "completion_integrity": completion, "event_id": str(uuid.uuid4()), "event_kind": "activation"}
    ctx.activation["outer_wall_ms"] = (time.perf_counter() - started) * 1000
    return ctx


def make_row(runner, result, diag, meta, *, methods, method, dialog, turn, phase, run, attempt, physical_id, ordinal, store_info, config):
    question = dialog["turns"][turn - 1]
    n = int(meta["n_spatial"]); hit = turn > 1
    stored = method.startswith(PK)
    k = (n + 3) // 4 if hit and stored else n
    logical = request_id(run.name, phase, method, dialog["dialog_id"], turn)
    suffix = QA._suffix_from_tokenized(runner, runner.processor.tokenizer(diag["prompt"], return_tensors="pt")).cpu().tolist()
    if result.get("suffix_input_ids") is not None:
        assert result["suffix_input_ids"] == suffix
    value = {a: b for a, b in result.items() if a not in ("captured_past_key_values", "first_logits", "logits")}
    g = {key: meta.get(key, []) if key == "padding_idx" else meta[key] for key in
         ("num_heads", "head_dim", "num_layers", "v_token_start", "v_token_num", "n_spatial", "prefix_len", "dtype", "newline_idx", "padding_idx")}
    order = method_order(methods, ordinal)
    return {"schema_version": SCHEMA, "experiment_id": run.name, "phase": phase, "attempt_id": attempt,
            "logical_request_id": logical, "request_id": logical, "physical_execution_id": physical_id,
            "model_id": MODEL_ID, "model_revision": REVISION, "method_id": method, "image_id": dialog["image_id"],
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
            "ssd_read_bytes": int(result.get("io", {}).get("bytes", 0)), "ssd_preads": int(result.get("io", {}).get("preads", 0)),
            "read_calls": int(result.get("io", {}).get("preads", 0)), "pread_calls": int(result.get("pread_calls", 0)), "preadv_calls": int(result.get("preadv_calls", 0)),
            "N_content": n, "k": k, "k_per_layer": result.get("k_per_layer") if hit and stored else None,
            "k_total": result.get("k_total") if hit and stored else None,
            "content_kv_fraction": k / n, "geometry": g,
            "budget_unit": "visual_kv" if stored else "full_context",
            "budget_policy": "layerwise_v1" if stored else None, "vector": vector_of(method) if stored else None,
            "store_format": STORE_FORMAT if stored else ("raster" if method == "fullload" else None),
            "source_store": store_info if hit and method != "recompute" else None,
            "provisioning_mode": "FRESH_IMAGE_STREAMING", "result": safe(value)}


# ------------------------------------------------------------------- audit
def audit_image_rows(rows, manifest, config, phase, image_id, artifact=None):
    methods = config["methods"]
    c = Checks()
    ds = [d for d in manifest["dialogues"] if d["image_id"] == image_id]
    if phase in ("smoke", "integration"):
        ids = manifest["smoke_dialogue_ids"]
        ds = [d for d in ds if d["dialog_id"] in ids]
    c.require(bool(ds), "known_image", image_id)
    expected = {(d["dialog_id"], m, t): d["turns"][t - 1] for d in ds for m in methods for t in (1, 2, 3)}
    bykey = {(r["dialog_id"], r["method_id"], r["turn_id"]): r for r in rows}
    c.require(set(bykey) == set(expected) and len(rows) == len(expected), "image_completeness", image_id)
    c.require(len({r["logical_request_id"] for r in rows}) == len(rows), "unique_logical_ids", image_id)
    c.require(len({r["physical_execution_id"] for r in rows}) == len(rows), "unique_physical_ids", image_id)
    ordinals = {d["dialog_id"]: i for i, d in enumerate(manifest["dialogues"])}
    metas = {}
    if artifact is not None:
        artifact = Path(artifact)
        builds = []
        for m in methods[1:]:
            kind = kind_of(m)
            p = json.loads((artifact / f"{kind}_persistence.json").read_text())
            meta = json.loads((artifact / f"{kind}_meta.json").read_text())
            hashes = json.loads((artifact / f"{kind}_hashes.json").read_text())
            files = p["files"]; builds.append(p["event_id"]); metas[kind] = meta
            c.require(hashes == {n: f["sha256"] for n, f in files.items()}, "complete_store_hash_inventory", kind)
            c.require(p["file_logical_bytes"] == sum(f["size"] for f in files.values()), "file_occupancy_sums", kind)
            c.require(p["service_persistence_ms"] >= p["native_writer_wall_ms"] > 0 and p["os_write_bytes"] >= p["file_logical_bytes"], "measured_store_completion", kind)
            c.require(p["source"]["dialog_id"] == ds[0]["dialog_id"] and p["source"]["method_id"] == m, "first_source_build_own_method", kind)
            n = meta["v_token_num"] - len(meta["newline_idx"]); width = meta["num_heads"] * meta["head_dim"] * 2
            if kind.startswith("packed_"):
                vec = kind[len("packed_"):]
                doc = config["layer_budgets"][vec]
                counts = layer_keep_counts(doc["layer_keep_ratios"], n)
                c.require(meta["storage_policy"] == "prefix25_packed" and meta["store_format"] == STORE_FORMAT and meta["budget_policy"] == "layerwise_v1", "PK_policy", kind)
                c.require(meta["layer_keep_counts"] == counts == meta["payload_rows_per_layer"], "PK_counts_recomputed", kind)
                c.require(sum(counts) == 32 * ((n + 3) // 4), "PK_budget_equals_uniform_total", kind)
                c.require(meta["layer_budget_sha256"] == doc["layer_budget_sha256"] and meta["layer_budget_id"] == vec, "PK_vector_identity", kind)
                c.require(meta["full_importance_permutation"] == meta["order"] and sorted(meta["order"]) == list(range(meta["v_token_num"])), "PK_full_mapping", kind)
                sep = 2 * meta["num_layers"] * len(meta["newline_idx"]) * width
                total = sum(2 * kk * width for kk in counts) + sep
                c.require(meta["packed_total_bytes"] == total == files[PACKED_NAME]["size"] and meta["packed_sep_bytes"] == sep, "PK_exact_file_length", kind)
                c.require(files[PACKED_NAME]["sha256"] == meta["packed_sha256"], "PK_packed_sha256_matches_inventory", kind)
                c.require(not any(f.startswith("layer_") or f == "sep_kv.bin" for f in files), "PK_no_unpacked_payload_files", kind)
                if vec == "U":
                    c.require(counts == [(n + 3) // 4] * meta["num_layers"], "PK_uniform_rows", kind)
            else:
                c.require(meta.get("storage_policy", "full") == "full", "raster_full_store", kind)
        c.require(len(set(builds)) == len(methods) - 1, "distinct_physical_build_events", image_id)
        for m in methods[1:]:
            a = json.loads((artifact / f"{m}_activation.json").read_text())
            c.require(a["event_kind"] == "activation" and a["outer_wall_ms"] > 0, "actual_activation_event", m)
        orders = {k: json.dumps(v["order"]) for k, v in metas.items() if k != "raster"}
        c.require(len(set(orders.values())) <= 1, "same_image_same_permutation", image_id)
    sel_sets = {}
    for key, r in bykey.items():
        rid = r["request_id"]; did, m, t = key; result = r["result"]; question = expected.get(key)
        if question is None:
            continue
        c.require(r["schema_version"] == SCHEMA and r["model_id"] == MODEL_ID, "schema_model", rid)
        c.require(rid == r["logical_request_id"] == f"{SCHEMA}:{r['experiment_id']}:{phase}:{MODEL_ID}:{m}:{did}:t{t}", "logical_identity", rid)
        c.require(r["phase"] == phase and r["image_id"] == image_id, "phase_image", rid)
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
        c.require(result["vision_forward_count"] == int(t == 1 or m == "recompute"), "vision_call_contract", rid)
        trace = result.get("os_pread_trace", [])
        c.require(r["ssd_read_bytes"] == sum(e["returned"] for e in trace), "actual_returned_bytes", rid)
        c.require(r["ssd_preads"] == len(trace) == r["read_calls"], "actual_read_count", rid)
        c.require(all(e["returned"] == e["requested"] for e in trace), "successful_full_requested_reads", rid)
        c.require(not any("probe_k.bin" in e["path"] for e in trace), "no_probe_read", rid)
        g = r["geometry"]; n = g["v_token_num"] - len(set(g["newline_idx"]) | set(g["padding_idx"]))
        c.require(n == r["N_content"] and n > 0, "actual_content_geometry", rid)
        if t == 1 or m == "recompute":
            c.require(r["k"] == n and r["ssd_read_bytes"] == 0, "full_pixel_request", rid)
            c.require(r["source_store"] is None, "pixel_not_cache_hit", rid)
        else:
            c.require(result.get("adapter_restored") is True, "adapter_restored", rid)
            c.require(bool(result.get("conditioning_events")) and all(x["status"] == "SUCCESS" for x in result["conditioning_events"]), "conditioning_success", rid)
            conditioned = {x["path"] for x in result["conditioning_events"]}
            c.require(all(e["path"] in conditioned for e in trace), "every_read_file_conditioned", rid)
            src = r["source_store"]; c.require(src is not None, "store_provenance", rid)
            c.require(src["source"]["dialog_id"] == ds[0]["dialog_id"] and src["source"]["method_id"] == m, "first_source_T1_own_method", rid)
            source_row = bykey.get((ds[0]["dialog_id"], m, 1), {})
            c.require(src["source"]["physical_execution_id"] == source_row.get("physical_execution_id"), "actual_source_physical_id", rid)
            width = g["num_heads"] * g["head_dim"] * 2; vn = g["v_token_num"]; nl = g["num_layers"]; structural = set(g["newline_idx"])
            if m == "fullload":
                exp_reads = Counter()
                for li in range(nl):
                    for f in ("k.bin", "v.bin"):
                        exp_reads[(li, f, 0, vn * width, vn * width)] += 1
                c.require(r["k"] == n and r["content_kv_fraction"] == 1, "fullload_full_content", rid)
                c.require(Counter(event_key(e) for e in trace) == exp_reads, "independent_os_ranges", rid)
                c.require(r["preadv_calls"] == 0 and r["pread_calls"] == 2 * nl, "fullload_pread_count", rid)
            else:
                vec = vector_of(m)
                counts = layer_keep_counts(config["layer_budgets"][vec]["layer_keep_ratios"], n)
                c.require(result.get("k_per_layer") == counts == r["k_per_layer"], "PK_k_per_layer_recomputed", rid)
                c.require(result.get("attended_content_kv_count_per_layer") == counts and result.get("k_total") == sum(counts) == nl * ((n + 3) // 4), "PK_budget_total", rid)
                c.require(result.get("layer_budget_sha256") == config["layer_budgets"][vec]["layer_budget_sha256"] and result.get("layer_budget_id") == vec, "PK_vector_identity_hit", rid)
                c.require(result.get("selected_stored_ids") == list(range(max(counts))), "PK_prefix_contract", rid)
                sep = 2 * nl * len(structural) * width
                total = sum(2 * kk * width for kk in counts) + sep
                # G7: exactly one preadv of the whole packed file, zero pread, range independent of the question
                c.require(len(trace) == 1 and trace[0]["syscall"] == "preadv" and Path(trace[0]["path"]).name == PACKED_NAME
                          and trace[0]["offset"] == 0 and trace[0]["returned"] == trace[0]["requested"] == total, "PK_single_preadv_whole_file", rid)
                c.require(r["preadv_calls"] == 1 and r["pread_calls"] == 0 and result.get("read_calls") == 1, "PK_read_call_counts", rid)
                c.require(r["ssd_read_bytes"] == total, "PK_bytes_equal_budget", rid)
                original = result.get("selected_original_ids", [])
                c.require(len(original) == max(counts) and len(set(original)) == len(original) and all(0 <= i < vn and i not in structural for i in original), "original_ids_real", rid)
                c.require(result.get("keep_count_per_layer") == [kk + len(structural) for kk in counts], "structural_keep", rid)
                c.require(r["budget_unit"] == "visual_kv" and r["store_format"] == STORE_FORMAT, "visual_kv_unit_packed", rid)
                c.require(result.get("store_format") == STORE_FORMAT and result.get("directcopy") is False, "PK_scatter_path", rid)
                sel_sets.setdefault(m, []).append((original, counts, total))
            c.require(result.get("scoring_calls") == 0 and result.get("query_score_calls", 0) == 0, "online_scoring_zero", rid)
        calls = result.get("projection_calls", {})
        for li in range(g["num_layers"]):
            observed = calls.get(str(li), {})
            c.require(observed.get("prefill") == {"q": 1, "k": 1, "v": 1}, "one_prefill_projection_each_layer", rid)
            c.require(observed.get("decode", {"q": 0, "k": 0, "v": 0}) == {"q": len(tokens) - 1, "k": len(tokens) - 1, "v": len(tokens) - 1}, "decode_projection_counts", rid)
    for m, sets in sel_sets.items():
        c.require(all(s == sets[0] for s in sets), "same_image_selection_invariant", f"{image_id}/{m}")
    if sel_sets:
        totals = {m: sets[0][2] for m, sets in sel_sets.items()}
        c.require(len(set(totals.values())) == 1, "same_bytes_every_vector", image_id)
    if metas:
        for m, sets in sel_sets.items():
            meta = metas[kind_of(m)]
            c.require(sets[0][0] == meta["order"][:len(sets[0][0])], "meta_to_actual_selection", f"{image_id}/{m}")
            c.require(sets[0][2] == meta["packed_total_bytes"], "meta_packed_bytes_to_actual_read", f"{image_id}/{m}")
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
        receipt = audit_image_rows(load_jsonl(raw), manifest, config, phase, cp.parent.name, folder)
        if receipt["status"] != "PASS":
            raise RuntimeError("resume independent audit failed: " + repr(receipt["failures"][:5]))
        done[cp.parent.name] = doc
    return done


def run_image(runner, server, dialogs, ordinals, phase, run, manifest, config):
    methods = config["methods"]; docs = config["layer_budgets"]
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
    kinds = {m: kind_of(m) for m in methods[1:]}
    paths = {k: scratch / k for k in kinds.values()}
    metas, hashes, persistence, contexts, store_info = {}, {}, {}, {}, {}
    rows = []; pending = []; success = False
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
                    capture = kinds.get(method, "none") if di == 0 and turn == 1 else "none"
                    if turn == 1 or method == "recompute":
                        result, diag = normal_request(runner, server, ROOT / dialog["image_path"], factory, "raster" if capture == "raster" else ("image_only" if capture != "none" else "none"))
                        append(artifact / "execution_events.jsonl", dict(source, turn_id=turn, status="EXECUTED", prediction=result["answer"],
                                                                        generated_token_ids=result["generated_token_ids"], ttft_ms=result["end_to_end_ttft_ms"],
                                                                        request_e2e_ms=result["request_e2e_ms"]))
                        if capture != "none":
                            start = time.perf_counter()
                            p = persist(runner, result, diag, paths[capture], image_id, capture, source, docs)
                            outer_ms = (time.perf_counter() - start) * 1000
                            persistence[capture] = dict(p, outer_helper_wall_ms=outer_ms, source=source,
                                                        capture_accounting="capture and exit materialization already within T1 request E2E; not added again")
                            metas[capture] = json.loads((paths[capture] / "meta.json").read_text())
                            hashes[capture] = file_hashes(paths[capture])
                            store_info[capture] = {"path": str(paths[capture]), "meta_sha256": sha(paths[capture] / "meta.json"), "source": source,
                                                   "file_sha256": hashes[capture], "physical_build_event_id": p["event_id"]}
                            atomic_json(artifact / f"{capture}_persistence.json", persistence[capture])
                            atomic_json(artifact / f"{capture}_meta.json", metas[capture])
                            atomic_json(artifact / f"{capture}_hashes.json", hashes[capture])
                        result.pop("captured_past_key_values", None)
                        saved = {a: b for a, b in diag.items() if a not in ("enc_cpu", "hidden", "vision")}
                        if di == 0 and turn == 1:
                            pending.append((method, result, saved, physical))
                            del diag, result
                            continue
                        diag = saved
                    else:
                        kind = kinds[method]
                        if method not in contexts:
                            contexts[method] = activate(runner, method, paths[kind])
                            atomic_json(artifact / f"{method}_activation.json", contexts[method].activation)
                        result, diag = stored_request(runner, server, contexts[method], factory, method, image_id)
                        result["metadata_activation"] = contexts[method].activation
                    kind = kinds.get(method, "raster")
                    meta = metas[kind] if kind in metas else metas[next(iter(metas))]
                    row = make_row(runner, result, diag, meta, methods=methods, method=method, dialog=dialog, turn=turn, phase=phase, run=run,
                                   attempt=attempt, physical_id=physical, ordinal=ordinal, store_info=store_info.get(kind), config=config)
                    row["image_sha256"] = source_image_sha
                    append(artifact / "raw.jsonl", row); rows.append(row); generated[method][turn] = row
                    append(artifact / "attempts.jsonl", dict(source, turn_id=turn, status="COMPLETED", at_unix=time.time()))
                    del result, diag
                if di == 0 and turn == 1:
                    assert set(metas) == set(kinds.values())
                    for method, result, diag, physical in pending:
                        kind = kinds.get(method, "raster")
                        row = make_row(runner, result, diag, metas[kind], methods=methods, method=method, dialog=dialog, turn=turn, phase=phase, run=run,
                                       attempt=attempt, physical_id=physical, ordinal=ordinal, store_info=store_info.get(kind), config=config)
                        row["image_sha256"] = source_image_sha
                        append(artifact / "raw.jsonl", row); rows.append(row); generated[method][1] = row
                        append(artifact / "attempts.jsonl", {"physical_execution_id": physical, "logical_request_id": row["logical_request_id"], "status": "COMPLETED", "at_unix": time.time()})
                    pending.clear(); del result, diag
                    for kind in metas:
                        assert metas[kind]["prefix_input_ids"] == metas["raster"]["prefix_input_ids"]
                        assert metas[kind]["image_input_sha256"] == metas["raster"]["image_input_sha256"]
        for kind in kinds.values():
            assert file_hashes(paths[kind]) == hashes[kind], "store mutated during requests"
        atomic_json(artifact / "store_integrity_after.json", {"status": "PASS", "file_sha256": hashes})
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
    if support.source_hashes(ROOT) != config["source_sha256"]:
        raise RuntimeError("FROZEN_SOURCE_CONFIG_MISMATCH")
    if sha(CONTRACT) != config["contract_sha256"] or sha(B_CONTRACT) != config["b_main_contract_sha256"]:
        raise RuntimeError("FROZEN_CONTRACT_MISMATCH")
    if frozen_dialogues() != manifest["dialogues"]:
        raise RuntimeError("FROZEN_DATASET_MISMATCH")
    for image_id, item in manifest["image_identity"].items():
        if sha(ROOT / item["image_path"]) != item["sha256"]:
            raise RuntimeError("FROZEN_IMAGE_MISMATCH: " + image_id)
    for alloc, doc in config["layer_budgets"].items():
        fresh = load_budget_document(doc["path"])
        assert fresh["layer_budget_sha256"] == doc["layer_budget_sha256"] and fresh["layer_keep_ratios"] == doc["layer_keep_ratios"]
    config.update(_file_sha256=sha(run / "config.json"), _manifest_sha256=sha(run / "manifest.json"))
    return config, manifest


def run_phase(runner, server, run, phase, config, manifest):
    done = completed_images(run, phase, config, manifest)
    ids = manifest["smoke_dialogue_ids"] if phase == "smoke" else [d["dialog_id"] for d in manifest["dialogues"]]
    keep = set(ids); groups = OrderedDict(); ordinals = {}
    for ordinal, d in enumerate(manifest["dialogues"]):
        ordinals[d["dialog_id"]] = ordinal
        if d["dialog_id"] in keep:
            groups.setdefault(d["image_id"], []).append(d)
    per_dialog = 3 * len(config["methods"])
    expected = len(ids) * per_dialog; completed = sum(x["requests"] for x in done.values())
    started = time.time()
    for index, (image, ds) in enumerate(groups.items()):
        if image in done:
            continue
        if (run / "STOP_REQUESTED").exists():
            raise RuntimeError("STOP_REQUESTED")
        t0 = time.time()
        completed += run_image(runner, server, ds, ordinals, phase, run, manifest, config)
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
def freeze(parent_run):
    run = parent_run / "hold"
    run.mkdir(exist_ok=False)
    sel = json.loads((parent_run / "dev_selection.json").read_text())
    methods = sel["hold_arms"]
    assert methods[:2] == ["recompute", "fullload"] and all(m.startswith(PK) and m.endswith(PK_SUFFIX) for m in methods[2:]) and len(methods) == 7
    assert sel["selected"] == "A1-2" and f"{PK}A1-2_packed" in methods and f"{PK}U_packed" in methods
    prior_manifest = json.loads((PRIOR / "manifest.json").read_text())
    ds = frozen_dialogues()
    assert prior_manifest["dialogues"] == ds
    for image, identity in prior_manifest["image_identity"].items():
        assert sha(ROOT / identity["image_path"]) == identity["sha256"]
    manifest = {"dataset": "MT-GQA-reconstructed", "schema_version": SCHEMA, "index_sha256": INDEX_SHA, "workload_sha256": WORKLOAD_SHA,
                "dialogues": ds, "image_identity": prior_manifest["image_identity"], "smoke_dialogue_ids": prior_manifest["smoke_dialogue_ids"],
                "methods": methods, "expected_requests": {"smoke": len(prior_manifest["smoke_dialogue_ids"]) * 3 * len(methods), "main": 12183 * len(methods)},
                "prior_manifest_sha256": sha(PRIOR / "manifest.json"), "smoke_policy": "prior frozen smoke dialogue ids (first four images)"}
    support.publish(run / "manifest.json", manifest)
    plan0 = json.loads((parent_run / "phase0/storage_plan.json").read_text())
    free = shutil.disk_usage(run).free
    plan = {"policy": "FRESH_IMAGE_STREAMING", "minimum_reserve_bytes": plan0["hold"]["minimum_reserve_bytes"],
            "peak_working_set_bytes": plan0["hold"]["peak_working_set_bytes"], "largest_image_bundle_bound_bytes": plan0["hold"]["largest_image_bundle_bound_bytes"],
            "maximum_new_payload_bytes": 20 << 30, "raw_and_evidence_growth_bound_bytes": plan0["hold"]["raw_and_evidence_growth_bound_bytes"],
            "free_bytes_at_freeze": free, "methods": len(methods),
            "required_free_bytes": plan0["hold"]["minimum_reserve_bytes"] + plan0["hold"]["peak_working_set_bytes"] + plan0["hold"]["raw_and_evidence_growth_bound_bytes"],
            "deletion_scope": "runs/<RUN>/hold/scratch only (exact SHA allowlist after audit+commit); existing stores never deleted"}
    plan["status"] = "PASS" if free >= plan["required_free_bytes"] else "BLOCKED_STORAGE"
    support.publish(run / "storage_plan.json", plan)
    if plan["status"] != "PASS":
        raise RuntimeError(f"BLOCKED_STORAGE: free {free} < required {plan['required_free_bytes']}")
    budgets = {}
    for m in methods[2:]:
        vec = vector_of(m)
        p = parent_run / "phase2" / f"layer_budget_llava_{vec}.json"
        doc = load_budget_document(p)
        assert doc["layer_budget_sha256"] == sel["vectors_sha256"][vec]
        budgets[vec] = {"path": str(p), "layer_budget_sha256": doc["layer_budget_sha256"], "layer_keep_ratios": doc["layer_keep_ratios"],
                        "r_min": doc["r_min"], "r_max": doc["r_max"], "alloc_id": vec, "policy": doc["policy"], "schema": doc["schema"],
                        "num_layers": doc["num_layers"], "total_ratio": doc["total_ratio"], "source": doc["source"]}
    config = {"schema_version": SCHEMA, "model": MODEL_ID, "revision": REVISION, "methods": methods, "seed": 1234, "greedy": True, "max_new_tokens": 16,
              "batch_size": 1, "cpu_threads": 2, "history_policy": "method_local_generated", "packed_storage_policy": "prefix25_packed",
              "store_format": STORE_FORMAT, "read_syscall": "os.preadv into pinned host buffer, one call per hit", "layer_budgets": budgets,
              "dev_selection_sha256": sha(parent_run / "dev_selection.json"), "selected": sel["selected"],
              "source_sha256": support.source_hashes(ROOT), "contract_sha256": sha(CONTRACT), "b_main_contract_sha256": sha(B_CONTRACT),
              "manifest_sha256": sha(run / "manifest.json"), "storage_plan_sha256": sha(run / "storage_plan.json"),
              "source_tests_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "tests").glob("test_*.py"))},
              "source_T1_capture": "first frozen dialogue per image per stored method; all dialogues/methods still actual full-pixel T1",
              "prior_B_main_reference_only": str(PRIOR.parent), "v1_run_reference_only": "runs/layer_budget_kv25_20261105T125754Z", "parent_run": str(parent_run)}
    support.publish(run / "config.json", config)
    shutil.copy2(CONTRACT, run / "CONTRACT.md"); shutil.copy2(B_CONTRACT, run / "B_MAIN_CONTRACT.md")
    for n in config["source_sha256"]:
        target = run / "source_frozen" / n; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / n, target)
    env = {"python": sys.version, "torch": torch.__version__, "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
           "pip_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)}
    support.publish(run / "environment_before.json", env)
    print(json.dumps({"freeze": "PASS", "run": str(run), "methods": methods, "expected_main": manifest["expected_requests"]["main"], "storage": plan["status"]}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True, help="parent layer_budget_kv25_v2 run")
    p.add_argument("--stage", choices=("freeze", "smoke", "main"), required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    parent_run = args.run.resolve()
    if args.stage == "freeze":
        return freeze(parent_run)
    run = parent_run / "hold"
    config, manifest = verify_freeze(run)
    assert json.loads((run / "storage_plan.json").read_text())["status"] == "PASS"
    torch.set_num_threads(config["cpu_threads"]); torch.manual_seed(1234); np.random.seed(1234)
    torch.set_grad_enabled(False)
    assert_gpu_exclusive(); runner = V.load_runner()
    from mmimpress.serve import Server
    server = Server(runner, max_new_tokens=16); QA._warmup(runner, server)
    phase = args.stage
    try:
        if phase == "main":
            smoke = json.loads((run / "smoke" / "validation.json").read_text())
            if smoke["status"] != "PASS" or smoke["requests"] != manifest["expected_requests"]["smoke"]:
                raise RuntimeError("SMOKE_NOT_PASS")
            cap = capacity_guard(run)
            atomic_json(run / f"main_start_{time.strftime('%Y%m%dT%H%M%S')}.json", {"started_unix": time.time(), "capacity": cap, "pid": os.getpid()})
        if (run / phase).exists() and not args.resume:
            raise FileExistsError("use --resume; previous artifacts preserved")
        receipt = run_phase(runner, server, run, phase, config, manifest)
        atomic_json(run / f"{phase}_execution_complete.json", {"status": "PASS", **receipt, "finished_unix": time.time()}, replace=True)
    except BaseException as exc:
        error = {"status": "PARTIAL", "error": repr(exc), "traceback": traceback.format_exc(),
                 "final_requests": sum(v["requests"] for v in completed_images(run, phase, config, manifest).values()),
                 "resume_command": f"{sys.executable} scripts/162_eval_layer_budget_v2_hold.py --run {parent_run} --stage {phase} --resume"}
        atomic_json(run / ("interruption_" + time.strftime("%Y%m%dT%H%M%S") + ".json"), error)
        print(json.dumps({k: v for k, v in error.items() if k != "traceback"}), flush=True)
        raise


if __name__ == "__main__":
    main()

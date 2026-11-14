#!/usr/bin/env python3
"""LLaVA-1.5-7B MT-GQA HOLD: recompute / U100 / U50 / U25 / PYR-b20 packed
(contract docs/llava15_mt_gqa_contract.md).

Lifecycle = scripts/164_eval_layer_budget_pyrbeta_hold.py (itself a thin copy of 162):
request generation, dialogue-wise arm rotation (``ordinal % number of arms``), method-local
generated history, strict normalized EM, image-atomic attempts, independent per-image audit,
atomic commit and exact-allowlist cleanup.  Differences, all forced by the model or by the
ratio axis and marked ``# llava15:``:

* LLaVA-1.5 runner / store / hit path (``mmimpress.llava15``), no ``image_sizes``, no separator;
* the arm list and the four frozen vectors come from ``<run>/config.json`` (Phase 0 freeze);
* every k_l recomputation uses the vector's own total ratio and the contract clip range
  [0.0, 1.0] plus an independent integer Hamilton apportionment (never module defaults);
* byte equality is required only between arms of the same total ratio (U25 vs PYR-b20);
  the audit adds the expected file size per arm and the nesting U25 < U50 < U100 of the
  selected original ids.

Stages: --stage integration | smoke | main [--resume]      (freeze is scripts/177)
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import llava15_support as S
from llava15_support import MODEL_ID, PK, REVISION, SCHEMA, SHORT_ANSWER, kind_of, stored_methods, vector_of
from mmimpress.llava15.store import STORE_FORMAT, keep_counts
from mmimpress.packed_store import PACKED_NAME

H1 = S.lifecycle()
QA = S.qa()
import main_b_support as support

sha, canonical_hash, safe, atomic_json, append, load_jsonl = H1.sha, H1.canonical_hash, H1.safe, H1.atomic_json, H1.append, H1.load_jsonl
frozen_dialogues, method_order, strict_score, assert_gpu_exclusive = H1.frozen_dialogues, H1.method_order, H1.strict_score, H1.assert_gpu_exclusive
file_hashes, Checks, capacity_guard, cleanup_committed = H1.file_hashes, H1.Checks, H1.capacity_guard, H1.cleanup_committed
SMOKE_LIKE = ("smoke", "integration")


def doc_counts(doc, n):      # llava15: k_l from the frozen document (explicit num_layers / total_ratio / r_min / r_max)
    return keep_counts(doc, int(n))


def ceil_ratio(vec, n):      # llava15: integer ceil(p/q * N)
    p, q = S.FRACTION[vec]
    return -(-p * int(n) // q)


def make_row(runner, result, diag, meta, *, methods, method, dialog, turn, phase, run, attempt, physical_id, ordinal, store_info, config):
    question = dialog["turns"][turn - 1]
    n = int(meta["n_spatial"]); hit = turn > 1
    stored = method.startswith(PK)
    served = hit and stored
    k = ceil_ratio(vector_of(method), n) if served else n
    logical = S.request_id(run.name, phase, method, dialog["dialog_id"], turn)
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
            "N_content": n, "k": k, "k_per_layer": result.get("k_per_layer") if served else None,
            "k_total": result.get("k_total") if served else None,
            "content_kv_fraction": (result["k_total"] / (32 * n)) if served else 1.0, "geometry": g,
            "budget_unit": "visual_kv" if stored else "full_context",
            "budget_policy": "layerwise_v1" if stored else None, "vector": vector_of(method) if stored else None,
            "total_ratio": S.TOTAL_RATIO[vector_of(method)] if stored else None,
            "store_format": STORE_FORMAT if stored else None,
            "source_store": store_info if hit and method != "recompute" else None,
            "provisioning_mode": "FRESH_IMAGE_STREAMING", "result": safe(value)}


# ------------------------------------------------------------------- audit
def audit_image_rows(rows, manifest, config, phase, image_id, artifact=None):
    methods = config["methods"]
    c = Checks()
    ds = [d for d in manifest["dialogues"] if d["image_id"] == image_id]
    if phase in SMOKE_LIKE:
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
        for m in stored_methods(methods):
            kind = kind_of(m); vec = vector_of(m)
            p = json.loads((artifact / f"{kind}_persistence.json").read_text())
            meta = json.loads((artifact / f"{kind}_meta.json").read_text())
            hashes = json.loads((artifact / f"{kind}_hashes.json").read_text())
            files = p["files"]; builds.append(p["event_id"]); metas[kind] = meta
            c.require(hashes == {n: f["sha256"] for n, f in files.items()}, "complete_store_hash_inventory", kind)
            c.require(p["file_logical_bytes"] == sum(f["size"] for f in files.values()), "file_occupancy_sums", kind)
            c.require(p["service_persistence_ms"] >= p["native_writer_wall_ms"] > 0 and p["os_write_bytes"] >= p["file_logical_bytes"], "measured_store_completion", kind)
            c.require(p["source"]["dialog_id"] == ds[0]["dialog_id"] and p["source"]["method_id"] == m, "first_source_build_own_method", kind)
            n = meta["v_token_num"] - len(meta["newline_idx"]); width = meta["num_heads"] * meta["head_dim"] * 2
            doc = config["layer_budgets"][vec]
            counts = doc_counts(doc, n)
            c.require(n == 576 and meta["v_token_num"] == 576 and meta["newline_idx"] == [], "llava15_geometry_576_no_separator", kind)      # llava15
            c.require(counts == S.independent_counts(doc["layer_keep_ratios"], n, S.FRACTION[vec]) and min(counts) >= 1, "PK_counts_independent_nonzero", kind)
            c.require(meta["layer_budget_r_min"] == doc["r_min"] == 0.0 and meta["layer_budget_r_max"] == doc["r_max"] == 1.0
                      and meta["layer_budget_total_ratio"] == doc["total_ratio"] == S.TOTAL_RATIO[vec]
                      and meta["layer_keep_ratios"] == doc["layer_keep_ratios"], "PK_clip_range_total_ratio_and_ratios_from_document", kind)      # llava15
            c.require(meta["storage_policy"] == "prefix_ratio_packed" and meta["store_format"] == STORE_FORMAT and meta["budget_policy"] == "layerwise_v1", "PK_policy", kind)
            c.require(meta["layer_keep_counts"] == counts == meta["payload_rows_per_layer"], "PK_counts_recomputed", kind)
            c.require(sum(counts) == 32 * ceil_ratio(vec, n), "PK_budget_equals_32_ceil_ratio_N", kind)      # llava15
            c.require(meta["layer_budget_sha256"] == doc["layer_budget_sha256"] == S.RATIO_SHA[vec] and meta["layer_budget_id"] == vec, "PK_vector_identity", kind)
            c.require(meta["full_importance_permutation"] == meta["order"] and sorted(meta["order"]) == list(range(meta["v_token_num"])), "PK_full_mapping", kind)
            total = sum(2 * kk * width for kk in counts)
            c.require(meta["packed_total_bytes"] == total == files[PACKED_NAME]["size"] == S.EXPECTED_BYTES[vec] and meta["packed_sep_bytes"] == 0, "PK_exact_file_length", kind)      # llava15
            c.require(files[PACKED_NAME]["sha256"] == meta["packed_sha256"], "PK_packed_sha256_matches_inventory", kind)
            c.require(not any(f.startswith("layer_") or f == "sep_kv.bin" for f in files), "PK_no_unpacked_payload_files", kind)
            if vec.startswith("U"):
                c.require(counts == [ceil_ratio(vec, n)] * meta["num_layers"], "PK_uniform_rows", kind)
        c.require(len(set(builds)) == len(stored_methods(methods)), "distinct_physical_build_events", image_id)
        for m in stored_methods(methods):
            a = json.loads((artifact / f"{m}_activation.json").read_text())
            c.require(a["event_kind"] == "activation" and a["outer_wall_ms"] > 0, "actual_activation_event", m)
        orders = {k: json.dumps(v["order"]) for k, v in metas.items()}
        c.require(len(set(orders.values())) <= 1, "same_image_same_permutation", image_id)
    sel_sets = {}
    for key, r in bykey.items():
        rid = r["request_id"]; did, m, t = key; result = r["result"]; question = expected.get(key)
        if question is None:
            continue
        c.require(r["schema_version"] == SCHEMA and r["model_id"] == MODEL_ID and r["model_revision"] == REVISION, "schema_model", rid)
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
        g = r["geometry"]; n = g["v_token_num"] - len(set(g["newline_idx"]) | set(g["padding_idx"]))
        c.require(n == r["N_content"] == 576 and g["newline_idx"] == [] and g["prefix_len"] == g["v_token_start"] + 576, "actual_content_geometry", rid)      # llava15
        if t == 1 or m == "recompute":
            c.require(r["k"] == n and r["ssd_read_bytes"] == 0 and r["content_kv_fraction"] == 1.0, "full_pixel_request", rid)
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
            width = g["num_heads"] * g["head_dim"] * 2; vn = g["v_token_num"]; nl = g["num_layers"]
            vec = vector_of(m); doc = config["layer_budgets"][vec]
            counts = doc_counts(doc, n)
            c.require(counts == S.independent_counts(doc["layer_keep_ratios"], n, S.FRACTION[vec]), "PK_k_per_layer_independent", rid)
            c.require(result.get("k_per_layer") == counts == r["k_per_layer"], "PK_k_per_layer_recomputed", rid)
            c.require(result.get("attended_content_kv_count_per_layer") == counts and result.get("k_total") == sum(counts) == nl * ceil_ratio(vec, n), "PK_budget_total", rid)
            c.require(result.get("layer_budget_sha256") == doc["layer_budget_sha256"] and result.get("layer_budget_id") == vec
                      and result.get("ratio") == S.TOTAL_RATIO[vec] == r["total_ratio"], "PK_vector_identity_hit", rid)
            c.require(result.get("selected_stored_ids") == list(range(max(counts))), "PK_prefix_contract", rid)
            total = sum(2 * kk * width for kk in counts)
            # LG7: exactly one preadv of the whole packed file, zero pread, range independent of the question
            c.require(len(trace) == 1 and trace[0]["syscall"] == "preadv" and Path(trace[0]["path"]).name == PACKED_NAME
                      and trace[0]["offset"] == 0 and trace[0]["returned"] == trace[0]["requested"] == total, "PK_single_preadv_whole_file", rid)
            c.require(r["preadv_calls"] == 1 and r["pread_calls"] == 0 and result.get("read_calls") == 1, "PK_read_call_counts", rid)
            c.require(r["ssd_read_bytes"] == total == S.EXPECTED_BYTES[vec], "PK_bytes_equal_budget", rid)
            original = result.get("selected_original_ids", [])
            c.require(len(original) == max(counts) and len(set(original)) == len(original) and all(0 <= i < vn for i in original), "original_ids_real", rid)
            c.require(result.get("keep_count_per_layer") == counts, "keep_equals_k_l_no_structural_rows", rid)      # llava15
            c.require(r["budget_unit"] == "visual_kv" and r["store_format"] == STORE_FORMAT, "visual_kv_unit_packed", rid)
            c.require(result.get("store_format") == STORE_FORMAT and result.get("directcopy") is False, "PK_scatter_path", rid)
            c.require(math.isclose(r["content_kv_fraction"], sum(counts) / (32 * n), abs_tol=1e-12), "content_fraction", rid)
            sel_sets.setdefault(m, []).append((original, counts, total))
            c.require(result.get("scoring_calls") == 0 and result.get("query_score_calls", 0) == 0 and result.get("static_score_calls", 0) == 0, "online_scoring_zero", rid)
        calls = result.get("projection_calls", {})
        for li in range(g["num_layers"]):
            observed = calls.get(str(li), {})
            c.require(observed.get("prefill") == {"q": 1, "k": 1, "v": 1}, "one_prefill_projection_each_layer", rid)
            c.require(observed.get("decode", {"q": 0, "k": 0, "v": 0}) == {"q": len(tokens) - 1, "k": len(tokens) - 1, "v": len(tokens) - 1}, "decode_projection_counts", rid)
    for m, sets in sel_sets.items():
        c.require(all(s == sets[0] for s in sets), "same_image_selection_invariant", f"{image_id}/{m}")
    if sel_sets:      # llava15: equal bytes only within one total ratio; nesting across ratios
        by_ratio = {}
        for m, sets in sel_sets.items():
            by_ratio.setdefault(S.TOTAL_RATIO[vector_of(m)], set()).add(sets[0][2])
        c.require(all(len(v) == 1 for v in by_ratio.values()), "same_bytes_within_equal_total_ratio", image_id)
        chain = [m for m in (S.U25, S.U50, S.U100) if m in sel_sets]
        for small, large in zip(chain, chain[1:]):
            a, b = sel_sets[small][0][0], sel_sets[large][0][0]
            c.require(b[:len(a)] == a and len(a) < len(b), "nested_selection_across_ratios", f"{image_id}/{small}<{large}")
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
    kinds = {m: kind_of(m) for m in stored_methods(methods)}
    ref_kind = next(iter(kinds.values()))
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
                    logical = S.request_id(run.name, phase, method, dialog["dialog_id"], turn)
                    source = {"dialog_id": dialog["dialog_id"], "method_id": method, "physical_execution_id": physical, "logical_request_id": logical}
                    append(artifact / "attempts.jsonl", dict(source, turn_id=turn, status="STARTED", at_unix=time.time()))
                    factory = lambda m=method: S.prompt_factory(methods, dialog, turn, m, generated[m])
                    capture = kinds.get(method, "none") if di == 0 and turn == 1 else "none"
                    if turn == 1 or method == "recompute":
                        result, diag = S.normal_request(runner, server, ROOT / dialog["image_path"], factory, capture != "none")
                        append(artifact / "execution_events.jsonl", dict(source, turn_id=turn, status="EXECUTED", prediction=result["answer"],
                                                                        generated_token_ids=result["generated_token_ids"], ttft_ms=result["end_to_end_ttft_ms"],
                                                                        request_e2e_ms=result["request_e2e_ms"]))
                        if capture != "none":
                            start = time.perf_counter()
                            p = S.persist(runner, result, diag, paths[capture], image_id, capture, source, docs)
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
                        saved = {a: b for a, b in diag.items() if a not in ("enc_cpu", "vision")}
                        if di == 0 and turn == 1:
                            pending.append((method, result, saved, physical))
                            del diag, result
                            continue
                        diag = saved
                    else:
                        kind = kinds[method]
                        if method not in contexts:
                            contexts[method] = S.activate(runner, method, paths[kind], docs[vector_of(method)])
                            atomic_json(artifact / f"{method}_activation.json", contexts[method].activation)
                        result, diag = S.stored_request(runner, server, contexts[method], factory, image_id)
                        result["metadata_activation"] = contexts[method].activation
                    kind = kinds.get(method, ref_kind)
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
                        kind = kinds.get(method, ref_kind)
                        row = make_row(runner, result, diag, metas[kind], methods=methods, method=method, dialog=dialog, turn=turn, phase=phase, run=run,
                                       attempt=attempt, physical_id=physical, ordinal=ordinal, store_info=store_info.get(kind), config=config)
                        row["image_sha256"] = source_image_sha
                        append(artifact / "raw.jsonl", row); rows.append(row); generated[method][1] = row
                        append(artifact / "attempts.jsonl", {"physical_execution_id": physical, "logical_request_id": row["logical_request_id"], "status": "COMPLETED", "at_unix": time.time()})
                    pending.clear(); del result, diag
                    for kind in metas:
                        assert metas[kind]["prefix_input_ids"] == metas[ref_kind]["prefix_input_ids"]
                        assert metas[kind]["image_input_sha256"] == metas[ref_kind]["image_input_sha256"]
                        assert metas[kind]["order"] == metas[ref_kind]["order"]
        for kind in kinds.values():
            assert file_hashes(paths[kind]) == hashes[kind], "store mutated during requests"
        atomic_json(artifact / "store_integrity_after.json", {"status": "PASS", "file_sha256": hashes})
        audit = audit_image_rows(load_jsonl(artifact / "raw.jsonl"), manifest, config, phase, image_id, artifact)
        audit["raw_sha256"] = sha(artifact / "raw.jsonl"); atomic_json(artifact / "independent_audit.json", audit)
        if audit["status"] != "PASS":
            raise RuntimeError("INDEPENDENT_IMAGE_AUDIT_FAIL: " + repr(audit["failures"][:10]))
        allowlist = file_hashes(scratch)
        atomic_json(artifact / "cleanup_allowlist.json", {"scratch": str(scratch), "file_sha256": allowlist, "ownership_run": run.name,
                                                         "bytes": sum((scratch / r).stat().st_size for r in allowlist)})
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
            atomic_json(artifact / "cleanup_receipt.json", {"status": "PASS", "removed_new_files": len(allowlist), "at_unix": time.time()})
    return len(rows)


def verify_freeze(run):
    config = json.loads((run / "config.json").read_text()); manifest = json.loads((run / "manifest.json").read_text())
    if support.source_hashes(ROOT) != config["source_sha256"]:
        raise RuntimeError("FROZEN_SOURCE_CONFIG_MISMATCH")
    if sha(S.CONTRACT) != config["contract_sha256"] or sha(S.B_CONTRACT) != config["b_main_contract_sha256"]:
        raise RuntimeError("FROZEN_CONTRACT_MISMATCH")
    if sha(run / "manifest.json") != config["manifest_sha256"] or sha(run / "storage_plan.json") != config["storage_plan_sha256"]:
        raise RuntimeError("FROZEN_MANIFEST_OR_STORAGE_PLAN_MISMATCH")
    if frozen_dialogues() != manifest["dialogues"]:
        raise RuntimeError("FROZEN_DATASET_MISMATCH")
    for image_id, item in manifest["image_identity"].items():
        if sha(ROOT / item["image_path"]) != item["sha256"]:
            raise RuntimeError("FROZEN_IMAGE_MISMATCH: " + image_id)
    if config["methods"] != S.METHODS or config["model"] != MODEL_ID or config["revision"] != REVISION:
        raise RuntimeError("FROZEN_ARMS_OR_MODEL_MISMATCH")
    fresh = S.load_vectors(run)
    for vec, doc in config["layer_budgets"].items():
        assert fresh[vec]["layer_budget_sha256"] == doc["layer_budget_sha256"] == S.RATIO_SHA[vec] and fresh[vec]["layer_keep_ratios"] == doc["layer_keep_ratios"]
        assert sha(S.vector_path(run, vec)) == config["vectors_file_sha256"][vec]
    config.update(_file_sha256=sha(run / "config.json"), _manifest_sha256=sha(run / "manifest.json"))
    return config, manifest


def run_phase(runner, server, run, phase, config, manifest):
    done = completed_images(run, phase, config, manifest)
    ids = manifest["smoke_dialogue_ids"] if phase in SMOKE_LIKE else [d["dialog_id"] for d in manifest["dialogues"]]
    keep = set(ids); groups = OrderedDict(); ordinals = {}
    for ordinal, d in enumerate(manifest["dialogues"]):
        ordinals[d["dialog_id"]] = ordinal
        if d["dialog_id"] in keep:
            groups.setdefault(d["image_id"], []).append(d)
    per_dialog = 3 * len(config["methods"])
    expected = len(ids) * per_dialog; completed = sum(x["requests"] for x in done.values())
    assert expected == manifest["expected_requests"][phase]
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--stage", choices=("integration", "smoke", "main"), required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    run = args.run.resolve()
    assert run.parent == ROOT / "runs" and run.name.startswith("llava15_mt_gqa_")
    config, manifest = verify_freeze(run)
    assert json.loads((run / "storage_plan.json").read_text())["status"] == "PASS"
    phase = args.stage
    if phase in ("smoke", "main"):      # llava15: every gate (LG1-LG10) must be recorded PASS for the current freeze
        gates = json.loads((run / "gates" / "summary.json").read_text())
        if gates["status"] != "PASS" or gates["config_sha256"] != config["_file_sha256"]:
            raise RuntimeError("GATES_NOT_PASS_FOR_THIS_FREEZE")
    else:      # llava15: the integration run is gate LG10 itself; LG1-LG9 must already be PASS for this freeze
        for name in ("cpu", "gpu"):
            g = json.loads((run / "gates" / f"{name}.json").read_text())
            if g["status"] != "PASS" or g["config_sha256"] != config["_file_sha256"]:
                raise RuntimeError(f"GATE_{name.upper()}_NOT_PASS_FOR_THIS_FREEZE")
    torch.set_num_threads(config["cpu_threads"]); torch.manual_seed(1234); np.random.seed(1234)
    torch.set_grad_enabled(False)
    assert_gpu_exclusive()
    from mmimpress.llava15.runner import load_runner
    from mmimpress.serve import Server
    runner = load_runner(1234)
    server = Server(runner, max_new_tokens=16)
    warm = S.warmup(runner, server)
    try:
        if phase == "main":
            smoke = json.loads((run / "smoke" / "validation.json").read_text())
            if smoke["status"] != "PASS" or smoke["requests"] != manifest["expected_requests"]["smoke"] or smoke["config_sha256"] != config["_file_sha256"]:
                raise RuntimeError("SMOKE_NOT_PASS")
        cap = capacity_guard(run)
        atomic_json(run / f"{phase}_start_{time.strftime('%Y%m%dT%H%M%S')}.json", {"started_unix": time.time(), "capacity": cap, "pid": os.getpid(), "warmup": warm,
                                                                               "model": runner.describe()})
        if (run / phase).exists() and not args.resume:
            raise FileExistsError("use --resume; previous artifacts preserved")
        receipt = run_phase(runner, server, run, phase, config, manifest)
        atomic_json(run / f"{phase}_execution_complete.json", {"status": "PASS", **receipt, "finished_unix": time.time()}, replace=True)
    except BaseException as exc:
        error = {"status": "PARTIAL", "error": repr(exc), "traceback": traceback.format_exc(),
                 "final_requests": sum(v["requests"] for v in completed_images(run, phase, config, manifest).values()),
                 "resume_command": f"{S.ENV} {S.PY} scripts/179_eval_llava15_mt_gqa_hold.py --run {run} --stage {phase} --resume"}
        atomic_json(run / ("interruption_" + phase + "_" + time.strftime("%Y%m%dT%H%M%S") + ".json"), error)
        print(json.dumps({k: v for k, v in error.items() if k != "traceback"}), flush=True)
        raise


if __name__ == "__main__":
    main()

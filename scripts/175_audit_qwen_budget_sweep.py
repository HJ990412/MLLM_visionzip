#!/usr/bin/env python3
"""Independent raw auditor for the Qwen2.5-VL retention-ratio sweep (25 / 30 / 40 / 50 / 100 %).

A thin adaptation of scripts/170 (PyramidKV run), itself a port of the rules of scripts/111, to
seven arms.  No production selector, loader, packed reader, layer-budget module, renderer or
summary code is imported: stable rank, k = ceil(p*N/q) in integer arithmetic, byte tables,
prompts, tokenisation, scores and timers are all recomputed here from raw evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRACTION = {"U25": (1, 4), "U30": (3, 10), "U40": (2, 5), "U50": (1, 2), "U100": (1, 1)}     # the frozen ratios as integer fractions
ALLOCS = tuple(FRACTION)
PACKED = {f"qwen_ours_kv_{a}_packed": a for a in ALLOCS}
METHODS = ("recompute", "fullload") + tuple(PACKED)
STORED = METHODS[1:]
U100 = "qwen_ours_kv_U100_packed"
LABELS = dict(zip(METHODS, ("ReComp", "FullLoad") + tuple(f"{a}_packed" for a in ALLOCS)))
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
SCHEMA = "qwen-mtgqa-budget-sweep-v1"
PACKED_FORMAT = "qwen25_bf16_prefix_ratio_packed_v1"
RATIO_SHA = {"U25": "1343c3ea849a9e70c965cea28530125c1072d841fc23cfd019f4282e11c7fc76",
             "U30": "b726c756cd0680a018200d934536542ea2a31effbe086535577e6f895984a0e7",
             "U40": "d14e8bb77c002045dad35ce55ba2b4b31301af1fbde74383244139a6b949e464",
             "U50": "40302d54a7eba3a7ef3fcc11e150ddef996c4e96a55e8df52af97cb1336d750b"}      # U100 is recomputed here from [1.0]*28
L, HEADS, HEAD_DIM = 28, 4, 128
ROW = HEADS * HEAD_DIM * 2                 # bytes of one K (or V) row of one layer
ROW_ALL = 2 * L * ROW                      # bytes of one prefix row over all layers, K+V
EXPECT = {"main": 85281, "smoke": 84, "integration": 84, "per_method": 12183, "per_method_hit": 8122, "images": 398, "dialogues": 4061}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canonical(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_rows(p):
    with Path(p).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def dump(p, x):
    with Path(p).open("x") as f:
        json.dump(x, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def csv_write(p, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def score(p, g):
    def norm(x):
        return " ".join(w for w in re.sub(r"[^\w\s]", " ", str(x).lower()).split() if w not in ("a", "an", "the"))
    return float(norm(p) == norm(g[0]))


class Checks:
    def __init__(self):
        self.count = 0
        self.failures = []
        self.counts = Counter()

    def require(self, ok, name, context=None):
        self.count += 1
        if not ok:
            self.counts[name] += 1
            if len(self.failures) < 50:
                self.failures.append({"check": name, "context": context})

    def result(self):
        return {"status": "PASS" if not self.counts else "FAIL", "checks": self.count, "failure_counts": dict(self.counts), "failures": self.failures}


# ----------------------------------------------------------- independent arithmetic
def ratio_sha(ratios):
    return hashlib.sha256(json.dumps([float(v) for v in ratios], separators=(",", ":")).encode("ascii")).hexdigest()


def ratio_float(alloc):
    p, q = FRACTION[alloc]
    return p / q


def int_k(alloc, n):
    """ceil(p*N/q): integer arithmetic only."""
    p, q = FRACTION[alloc]
    return -((-p * n) // q)


_VECTORS = {}


def vectors(config):
    key = config["_file_sha256"]
    if key not in _VECTORS:
        out = {}
        for alloc, info in config["vectors"].items():
            path = ROOT / info["file"]
            doc = json.loads(path.read_text())
            r = ratio_float(alloc)
            digest = ratio_sha([r] * L)
            ok = (alloc in FRACTION and sha(path) == info["file_sha256"] and doc["layer_budget_sha256"] == info["layer_budget_sha256"] == digest
                  and RATIO_SHA.get(alloc, digest) == digest and doc["layer_keep_ratios"] == [r] * L and doc["alloc_id"] == alloc
                  and (doc["num_layers"], doc["total_ratio"], doc["r_min"], doc["r_max"]) == (28, r, 0.0, 1.0)
                  and abs(sum(doc["layer_keep_ratios"]) - 28 * Fraction(*FRACTION[alloc])) < 1e-9)
            out[alloc] = {"ratios": doc["layer_keep_ratios"], "sha256": digest, "ok": bool(ok)}
        _VECTORS[key] = out
    return _VECTORS[key]


def prompt_text(q, history):
    text = "<|im_start|>system\nYou are a helpful assistant. Give concise factual answers.<|im_end|>\n"
    turns = list(history) + [(q, None)]
    for i, (question, answer) in enumerate(turns):
        text += "<|im_start|>user\n" + ("<|vision_start|><|image_pad|><|vision_end|>" if i == 0 else "") + question + "\nAnswer using a single word or short phrase.<|im_end|>\n"
        text += "<|im_start|>assistant\n"
        if answer is not None:
            text += answer + "<|im_end|>\n"
    return text


_TOKENIZERS = {}


def tokenizer(config):
    p = config["model"]["tokenizer_json"]
    if p not in _TOKENIZERS:
        from tokenizers import Tokenizer
        _TOKENIZERS[p] = Tokenizer.from_file(p)
    return _TOKENIZERS[p]


def rel_name(path):
    p = Path(path)
    return "/".join(p.parts[-2:]) if p.parent.name.startswith("layer_") else p.name


# ------------------------------------------------------------------ per-image audit
def audit_image_rows(rows, manifest, config, phase, iid, artifact=None):
    c = Checks()
    ds = [d for d in manifest["dialogues"] if d["image_id"] == iid]
    if phase in ("integration", "smoke"):
        ds = [d for d in ds if d["dialog_id"] in manifest[phase + "_dialogue_ids"]]
    expected = {(d["dialog_id"], m, t): d["turns"][t - 1] for d in ds for m in METHODS for t in (1, 2, 3)}
    by = {(r["dialog_id"], r["method_id"], r["turn_id"]): r for r in rows}
    ordinals = {d["dialog_id"]: i for i, d in enumerate(manifest["dialogues"])}
    c.require(bool(ds) and set(by) == set(expected) and len(rows) == len(expected), "exact_image_coverage", iid)
    c.require(len({r["logical_request_id"] for r in rows}) == len(rows), "unique_logical", iid)
    c.require(len({r["physical_execution_id"] for r in rows}) == len(rows), "unique_physical", iid)
    c.require(list(config["methods"]) == list(METHODS) and dict(config["packed_methods"]) == PACKED, "frozen_seven_arms", iid)
    vec = vectors(config)
    c.require(set(vec) == set(ALLOCS) and all(v["ok"] for v in vec.values()), "frozen_vectors_sha_uniform_explicit_clip", iid)
    n_manifest = manifest["image_identity"][iid]["N_content"]
    metas, selections, counts, ranks = {}, {}, {}, {}
    if artifact is not None:
        for m in STORED:
            meta = metas[m] = json.loads((artifact / f"{m}_meta.json").read_text())
            sel = selections[m] = json.loads((artifact / f"{m}_selection.json").read_text())
            hashes = json.loads((artifact / f"{m}_hashes.json").read_text())
            n = meta["visual_count"]
            order = meta.get("full_importance_permutation", meta["stored_to_original"])
            c.require((meta["num_layers"], meta["num_kv_heads"], meta["head_dim"], meta["dtype"]) == (28, 4, 128, "bfloat16"), "native_BF16_store_geometry", m)
            c.require(meta["identity"]["image_sha256"] == manifest["image_identity"][iid]["sha256"], "store_image_identity", m)
            c.require(meta["code_revision"] == config["core_model_store_revision"], "core_store_revision", m)
            c.require(n == n_manifest, "preflight_actual_geometry", m)
            c.require(meta["key_rope_state"] == "post_mrope" and meta["identity"]["position_policy"] == config["position_policy"], "position_policy", m)
            c.require(meta["prefix_input_ids"] == sel["prefix_ids"], "captured_prefix_ids", m)
            s = meta["structural_count"]
            if m == "fullload":
                c.require(order == list(range(n)) and meta["chunk_size"] == 64, "canonical_order", m)
            else:
                scores = sel["scores_float32"]
                c.require(len(scores) == n and all(math.isfinite(x) for x in scores), "scores_finite_complete", m)
                rank = sorted(range(n), key=lambda i: (-scores[i], i))
                c.require(order == rank, "independent_stable_rank", m)
            if m == "fullload":
                c.require(sel["selected_original_ids"] == list(range(n)) and sel["k"] == n, "selection_artifact_full", m)
            if m in PACKED:
                alloc = PACKED[m]
                k = int_k(alloc, n)
                k_l = counts[m] = [k] * L
                ranks[m] = rank
                c.require(meta.get("store_format") == meta.get("format") == PACKED_FORMAT and meta.get("storage_policy") == "prefix_ratio_packed", "packed_store_format", m)
                c.require(meta.get("packed_code_revision") == config["packed_code_revision"], "packed_code_revision", m)
                c.require(meta["layer_budget_id"] == alloc and meta["layer_budget_sha256"] == vec[alloc]["sha256"] and meta["layer_keep_ratios"] == [ratio_float(alloc)] * L
                          and ratio_sha(meta["layer_keep_ratios"]) == vec[alloc]["sha256"], "store_vector_is_frozen_vector", m)
                c.require(meta["layer_budget_r_min"] == 0.0 and meta["layer_budget_r_max"] == 1.0 and meta["layer_budget_total_ratio"] == ratio_float(alloc) == meta["retention_ratio"]
                          and meta["retention_fraction"] == list(FRACTION[alloc]) and len(meta["layer_keep_ratios"]) == 28, "explicit_qwen_ratio_and_clip_range_not_llava_defaults", m)
                c.require(meta["layer_keep_counts"] == k_l == sel["k_per_layer"] and meta["uniform_content_count"] == k == sel["k"], "independent_integer_fraction_row_count", m)
                c.require(1 <= k <= n and meta["stored_content_count_total"] == L * k and (alloc != "U100" or k == n), "row_budget_28_times_ceil_ratio_N", m)
                table, off = [], 2 * L * s * ROW
                for li in range(L):
                    table.append({"layer": li, "k_offset": off, "v_offset": off + k * ROW, "rows": k})
                    off += 2 * k * ROW
                c.require(meta["packed_layout"] == table and meta["packed_total_bytes"] == off == 2 * L * s * ROW + L * 2 * k * ROW == (k + s) * ROW_ALL
                          and meta["packed_structural_bytes"] == 2 * L * s * ROW and meta["row_bytes"] == ROW, "independent_packed_byte_table", m)
                c.require(meta["stored_to_original"] == rank[:k] and sel["full_importance_permutation"] == rank and sel["selected_original_ids"] == sorted(rank[:k]), "stored_rows_are_rank_prefix", m)
                c.require(set(hashes) == {"kv_packed.bin", "meta.json", "completion.json"} and hashes["kv_packed.bin"] == meta["packed_sha256"], "one_payload_file_with_recorded_sha", m)
            p = json.loads((artifact / f"{m}_persistence.json").read_text())
            a = json.loads((artifact / f"{m}_activation.json").read_text())
            c.require(p["source"] == sel["source"] and p["source"]["dialog_id"] == ds[0]["dialog_id"] and p["source"]["turn_id"] == 1 and p["source"]["method_id"] == m, "source_capture_persistence", m)
            c.require(p["bytes_written"] == sum(f["size"] for f in p["files"].values()), "actual_persistence_bytes", m)
            c.require(math.isclose(p["charged_setup_before_activation_ms"], p["outer_writer_wall_ms"], abs_tol=1e-8), "no_setup_double_count", m)
            payload = meta["packed_total_bytes"] if m in PACKED else meta["bytes_visual_kv"] + meta["bytes_structural_kv"]
            c.require(a["activation_io"]["bytes"] == payload + meta["bytes_metadata_file"] and a["metadata_resident_bytes"] > 0, "activation_integrity_IO", m)
            if m in PACKED:
                c.require(p["files"]["kv_packed.bin"]["size"] == meta["packed_total_bytes"] and p["files"]["kv_packed.bin"]["sha256"] == meta["packed_sha256"], "persisted_packed_file_matches_meta", m)
        c.require(all(metas[m]["prefix_input_ids"] == metas["fullload"]["prefix_input_ids"] for m in STORED), "source_prefix_compatibility", iid)
        first = next(iter(PACKED))
        c.require(all(metas[m]["full_importance_permutation"] == metas[first]["full_importance_permutation"] for m in PACKED), "same_image_order_in_every_packed_arm", iid)
        chain = [set(selections[m]["selected_original_ids"]) for m in PACKED]
        c.require(all(a <= b and len(a) <= len(b) for a, b in zip(chain, chain[1:])) and chain[-1] == set(range(n_manifest))
                  and [len(x) for x in chain] == [int_k(a, n_manifest) for a in ALLOCS], "selection_nested_U25_in_U30_in_U40_in_U50_in_U100_all", iid)
        sizes = [metas[m]["packed_total_bytes"] for m in PACKED]
        c.require(all(a <= b for a, b in zip(sizes, sizes[1:])) and sizes[-1] == (n_manifest + metas[U100]["structural_count"]) * ROW_ALL, "payload_bytes_monotone_and_U100_is_full_prefix", iid)
    tok = tokenizer(config)
    sets = defaultdict(list)
    for key, r in by.items():
        did, m, t = key
        q = expected.get(key)
        rid = r["logical_request_id"]
        v = r["result"]
        d = r["request_trace"]
        pixel = t == 1 or m == "recompute"
        if q is None:
            continue
        c.require(rid == f"{SCHEMA}:{r['experiment_id']}:{phase}:{MODEL}:{m}:{did}:t{t}", "logical_identity", rid)
        c.require(r["schema_version"] == SCHEMA and r["model_id"] == MODEL and r["model_revision"] == config["model"]["revision"], "model_schema", rid)
        c.require(r["phase"] == phase and r["image_id"] == iid and r["status"] == "PASS", "phase_status", rid)
        c.require(r["question_id"] == str(q["question_id"]) and r["question"] == q["question"] and r["gold"] == q["answers"], "frozen_question_gold", rid)
        c.require(r["config_sha256"] == config["_file_sha256"] and r["manifest_sha256"] == config["_manifest_sha256"] and r["code_sha256"] == canonical(config["source_sha256"]), "config_source_manifest", rid)
        o = ordinals[did]
        rotation = list(METHODS[o % 7:] + METHODS[:o % 7])
        c.require(r["method_order"] == rotation and r["method_order_position"] == rotation.index(m) and r["global_dialog_ordinal"] == o, "rotation_mod_7", rid)
        history, entries = [], []
        for pt in range(1, t):
            prev = by.get((did, m, pt))
            pq = expected.get((did, m, pt))
            if prev is None:
                c.require(False, "missing_history_source", rid)
                continue
            history.append([pq["question"], prev["prediction"]])
            entries.append({"turn_id": pt, "question_id": str(pq["question_id"]), "question": pq["question"], "answer": prev["prediction"],
                            "source_model": MODEL, "source_method_id": m, "source_dialog_id": did,
                            "source_logical_request_id": prev["logical_request_id"], "source_physical_execution_id": prev["physical_execution_id"]})
        text = prompt_text(q["question"], history)
        c.require(r["history"] == history and r["history_entries"] == entries and d["history"] == history and d["history_entries"] == entries, "actual_own_generated_history", rid)
        c.require(r["history_sha256"] == canonical(history) and r["prompt"] == text == d["prompt"] and r["prompt_sha256"] == hashlib.sha256(text.encode()).hexdigest(), "independent_causal_prompt", rid)
        n, s, k = r["N_content"], r["S_structural"], r["k"]
        target = n if pixel or m == "fullload" else int_k(PACKED[m], n)
        c.require(n == n_manifest and n > 0 and s == 21 and k == target, "actual_budget_geometry", rid)
        c.require(r["retention_ratio"] == (ratio_float(PACKED[m]) if m in PACKED else 1.0), "frozen_retention_ratio", rid)
        c.require(r["content_kv_fraction"] == k / n and r["structural_inclusive_kept_ratio"] == (k + s) / (n + s), "retention", rid)
        unit = "visual_kv_uniform_ratio" if m in PACKED else "full_context"
        c.require(r["budget_unit"] == unit, "explicit_budget", rid)
        rawids = tok.encode(text, add_special_tokens=False).ids
        image_token = config["model"]["image_token_id"]
        expanded = []
        c.require(rawids.count(image_token) == 1, "one_image_in_prompt", rid)
        for token in rawids:
            expanded.extend([token] * n if token == image_token else [token])
        c.require(r["input_ids"] == [expanded] == d["input_ids"] and r["input_ids_sha256"] == canonical([expanded]), "independent_input_tokenization", rid)
        c.require(r["image_sha256"] == manifest["image_identity"][iid]["sha256"], "image_hash", rid)
        tokens = v["generated_token_ids"]
        c.require(1 <= len(tokens) <= 16 and tokens[0] == v["first_token_id"] and len(tokens) == v["generated_token_count"] == v["model_forward_calls"], "actual_generated_sequence", rid)
        c.require(r["prediction"] == v["prediction"] == tok.decode(tokens, skip_special_tokens=True).strip(), "raw_decoded_prediction", rid)
        c.require(r["score"] == score(r["prediction"], q["answers"]), "independent_strict_score", rid)
        start, first, end, ret = [v[x] for x in ("request_started_at_s", "first_token_at_s", "request_finished_at_s", "core_returned_at_s")]
        c.require(start < first <= end <= ret and 0 < r["ttft_ms"] <= r["request_e2e_ms"], "timing_order", rid)
        c.require(math.isclose(r["ttft_ms"], 1000 * (first - start), abs_tol=1e-7) and math.isclose(r["request_e2e_ms"], 1000 * (end - start), abs_tol=1e-7), "absolute_true_timers", rid)
        c.require(r["ttft_ms"] == v["ttft_ms"] and r["request_e2e_ms"] == v["request_e2e_ms"]
                  and math.isclose(v["post_generation_tail_ms"], 1000 * (ret - v["generation_finished_at_s"]), abs_tol=1e-7), "timer_consistency", rid)
        c.require(v["core_ttft_ms"] <= r["ttft_ms"] + 1e-6, "core_timer_inside_true_timer", rid)
        capture = (m in STORED and did == ds[0]["dialog_id"] and t == 1)
        c.require(v["actual_vision_calls"] == v["vision_calls"] == int(pixel) and v["actual_score_calls"] == int(capture and m != "fullload"), "actual_vision_score_calls", rid)
        c.require(v["online_query_score_calls"] == 0 and v["request_state_restored"] is True and v["first_logits_finite"] is True and v["residual_layer_hooks"] == 0, "no_online_score_state_restored_no_residual_hook", rid)
        c.require(v["capture_mode"] == ("kv_only" if m == "fullload" else "with_score") if capture else v["capture_mode"] is False, "source_T1_only_capture", rid)
        c.require(v["image_file_decode_ms"] is not None and v["image_file_decode_ms"] >= 0 if pixel else v["image_file_decode_ms"] is None, "per_request_image_decode", rid)
        trace = v["os_pread_trace"]
        c.require(r["ssd_read_bytes"] == sum(e["returned"] for e in trace) and r["ssd_preads"] == len(trace)
                  and r["ssd_pread_calls"] == sum(e["syscall"] == "pread" for e in trace) and r["ssd_preadv_calls"] == sum(e["syscall"] == "preadv" for e in trace), "actual_OS_bytes_calls", rid)
        c.require(all(e["returned"] == e["requested"] for e in trace), "full_reads", rid)
        fullpos = d["position_ids"]
        cache_before = d["cache_before_decode"]
        prefix = n + s
        c.require(d["cache_after_decode"] == cache_before + len(tokens) - 1 and d["decode_start_position"] == max(max(axis[0]) for axis in fullpos) + 1, "full_logical_decode_position", rid)
        expected_input = expanded if pixel else expanded[prefix:]
        expected_positions = fullpos if pixel else [[axis[0][prefix:]] for axis in fullpos]
        c.require(d["actual_prefill_input_ids"] == [expected_input] and d["actual_prefill_position_ids"] == expected_positions, "suffix_full_logical_positions", rid)
        k_l = counts.get(m) if (m in PACKED and not pixel) else None
        k0 = k_l[0] if k_l is not None else k
        firstslot = 0 if pixel else k0 + s
        masklen = len(expanded) if pixel else k0 + s + len(expected_input)
        c.require(d["actual_prefill_cache_position"] == list(range(firstslot, firstslot + len(expected_input)))
                  and d["actual_prefill_attention_mask"] == [[1] * masklen] and cache_before == masklen, "compact_slots_causal_input", rid)
        if pixel:
            c.require(not trace and r["source_store"] is None and not d["layer_prefix_shapes"] and r["k_per_layer"] is None, "pixel_no_SSD_KV", rid)
            continue
        events = v["conditioning_events"]
        paths = {e["path"] for e in events}
        src = r["source_store"]
        prior = by.get((ds[0]["dialog_id"], m, 1), {})
        c.require(src["source"]["physical_execution_id"] == prior.get("physical_execution_id") and src["source"]["logical_request_id"] == prior.get("logical_request_id")
                  and src["source"]["turn_id"] == 1 and src["source"]["method_id"] == m, "actual_source_capture_lineage", rid)
        if m in metas:
            c.require(all(axis[0][:prefix] == metas[m]["logical_position_ids"][i] for i, axis in enumerate(fullpos)), "prefix_MRoPE_identity", rid)
        if m in PACKED:
            total = 2 * L * s * ROW + sum(2 * x * ROW for x in k_l) if k_l is not None else None
            c.require(k_l is not None and r["k_per_layer"] == k_l == v["layer_keep_counts"] and r["layer_budget_id"] == PACKED[m] == v["layer_budget_id"]
                      and r["layer_budget_sha256"] == vec[PACKED[m]]["sha256"] == v["layer_budget_sha256"], "request_layer_counts_and_vector", rid)
            if k_l is None:
                continue
            c.require(v["budget_unit"] == "visual_kv_uniform_ratio" and v["store_format"] == PACKED_FORMAT and v["read_syscall"] == "preadv"
                      and v["retention_ratio"] == ratio_float(PACKED[m]) and v["retention_fraction"] == list(FRACTION[PACKED[m]]), "actual_load_budget_unit", rid)
            c.require(len(trace) == 1 and trace[0]["syscall"] == "preadv" and Path(trace[0]["path"]).name == "kv_packed.bin" and trace[0]["offset"] == 0
                      and trace[0]["requested"] == trace[0]["returned"] == total, "one_preadv_whole_file_zero_pread", rid)
            c.require(v["pread_calls"] == 1 and v["read_io"]["preads"] == 1 and v["read_io"]["bytes"] == total == r["ssd_read_bytes"] == v["packed_read_bytes"]
                      and v["metadata_read_bytes"] == 0, "production_independent_IO_match", rid)
            c.require(v["visual_read_bytes"] == sum(2 * x * ROW for x in k_l) and v["structural_read_bytes"] == 2 * L * s * ROW, "KV_structural_bytes", rid)
            c.require(total == (int_k(PACKED[m], n) + s) * ROW_ALL, "read_bytes_equal_struct_plus_28_2_k_rows", rid)
            shapes = [[[1, 4, x + s, 128], [1, 4, x + s, 128]] for x in k_l]
            c.require(d["layer_prefix_shapes"] == shapes and v["layer_prefix_lengths"] == [x + s for x in k_l] == v["compact_prefix_tokens_per_layer"]
                      and v["compact_prefix_tokens"] == k_l[0] + s, "ragged_layer_cache_rows_exact", rid)
            c.require(v["h2d_kv_bytes"] == total and v["gpu_cache_kv_bytes"] == sum(2 * (x + s) * ROW for x in k_l) == total, "gpu_cache_bytes_equal_read_bytes", rid)
            c.require(v["k_total"] == sum(k_l) and v["k_max"] == max(k_l) == v["k_uniform"] == k_l[0] and v["k_quarter_reference"] == (n + 3) // 4
                      and v["padding_rows_read"] == 0 and v["extra_valid_visual_rows"] == 0, "no_padding_no_extra_rows", rid)
            if m in metas:
                c.require(v["selected_importance_prefix_original"] == metas[m]["full_importance_permutation"][:max(k_l)], "independent_selected_original", rid)
            c.require(v["mask_hook_calls_per_layer"] == len(tokens) and v["mask_hook_realigned_layers"] == sum(x != k_l[0] for x in k_l)
                      and v["residual_layer_hooks_after"] == 0 and v["ragged_design"] == "A_forward_pre_hook_right_aligned_mask", "layer_mask_hook_every_forward_then_removed", rid)
            c.require(len(events) == 1 and events[0]["status"] == "SUCCESS" and Path(events[0]["path"]).name == "kv_packed.bin" and all(e["path"] in paths for e in trace), "packed_file_conditioned", rid)
            sets[m].append((v["selected_importance_prefix_original"], v["layer_keep_counts"]))
            continue
        c.require(m == "fullload" and v["budget_unit"] == "chunk" and r["k_per_layer"] is None, "actual_load_budget_unit", rid)
        mchunk = (target + 63) // 64
        disk = mchunk * 64
        structbytes = 2 * L * s * ROW
        expectedreads = Counter({("structural_kv.bin", 0, structbytes, structbytes): 1})
        for li in range(28):
            for kind in ("k", "v"):
                expectedreads[(f"layer_{li:03d}/{kind}.bin", 0, disk * ROW, disk * ROW)] = 1
        actual = Counter((rel_name(e["path"]), e["offset"], e["requested"], e["returned"]) for e in trace)
        c.require(actual == expectedreads and len(trace) == v["pread_calls"] == 57 and all(e["syscall"] == "pread" for e in trace), "minimum_whole_chunk_OS_ranges", rid)
        c.require(v["read_io"]["bytes"] == r["ssd_read_bytes"] and v["read_io"]["preads"] == 57 and v["metadata_read_bytes"] == 0, "production_independent_IO_match", rid)
        c.require(v["visual_read_bytes"] == 2 * 28 * disk * ROW and v["structural_read_bytes"] == structbytes, "KV_structural_bytes", rid)
        c.require(v["selected_chunks"] == v["normal_chunks_read"] == mchunk and v["kept_tokens"] == v["target_visual_tokens"] == target
                  and v["loaded_valid_visual_rows"] == min(n, disk) and v["extra_valid_visual_rows"] == min(n, disk) - target
                  and v["padding_rows_read"] == disk - min(n, disk), "whole_chunk_extras_padding", rid)
        c.require(v["compact_prefix_tokens"] == target + s and d["layer_prefix_shapes"] == [[[1, 4, target + s, 128], [1, 4, target + s, 128]]] * 28, "all_layer_native_heads_exact_rows", rid)
        c.require(v["h2d_kv_bytes"] == v["gpu_cache_kv_bytes"] == 2 * 28 * (target + s) * ROW, "trim_before_H2D", rid)
        c.require(v["selected_visual_stored"] == list(range(target)), "first_k_stored_rows", rid)
        if m in metas:
            c.require(v["selected_visual_original"] == sorted(metas[m]["stored_to_original"][:target]), "independent_selected_original", rid)
        c.require(len(events) == 57 and all(e["status"] == "SUCCESS" for e in events) and all(e["path"] in paths for e in trace), "all_files_conditioned", rid)
        sets[m].append((v["selected_visual_original"], None))
    c.require(all(all(x == group[0] for x in group) for group in sets.values()), "same_image_T2_T3_selection_invariant_to_question", iid)
    # the request-level selections of the five ratios are nested prefixes of one order
    chosen = [set(sets[m][0][0][:sets[m][0][1][0]]) for m in PACKED if sets.get(m)]
    c.require(len(chosen) in (0, len(PACKED)) and all(a <= b for a, b in zip(chosen, chosen[1:])) and (not chosen or chosen[-1] == set(range(n_manifest))),
              "request_selection_nested_across_ratios", iid)
    # U100_packed and FullLoad serve the same KV: report (not fail) request-level token agreement for this image.
    pairs = [(by[(did, "fullload", t)], by[(did, U100, t)]) for (did, m, t) in by if m == "fullload" and (did, U100, t) in by]
    exact = sum(a["result"]["generated_token_ids"] == b["result"]["generated_token_ids"] for a, b in pairs)
    return dict(c.result(), image_id=iid, phase=phase, requests=len(rows), expected_requests=len(expected),
                u100_packed_vs_fullload_token_exact=exact, u100_packed_vs_fullload_pairs=len(pairs))


# ------------------------------------------------------------------------- summaries
def mean(x):
    return float(np.mean(x)) if len(x) else None


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else None


TIMING_KEYS = ("prompt_preparation", "store_load_inclusive", "raw_pread", "fixed_first_k_planning", "layerwise_planning", "h2d_and_assembly",
               "mask_hook_install", "suffix_prefill", "multimodal_prefill", "decode")


def compact(r):
    v = r["result"]
    d = {k: r[k] for k in ("method_id", "image_id", "dialog_id", "turn_id", "score", "ttft_ms", "request_e2e_ms", "ssd_read_bytes", "ssd_preads",
                           "ssd_pread_calls", "ssd_preadv_calls", "N_content", "S_structural", "k", "content_kv_fraction", "structural_inclusive_kept_ratio",
                           "question_id", "question", "global_dialog_ordinal", "method_order_position", "retention_ratio")}
    for key in ("visual_read_bytes", "structural_read_bytes", "metadata_read_bytes", "padding_rows_read", "extra_valid_visual_rows", "loaded_valid_visual_rows",
                "h2d_kv_bytes", "gpu_cache_kv_bytes", "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes", "generated_token_count", "capture_clone_ms",
                "score_extra_ms", "post_generation_tail_ms", "core_ttft_ms", "first_token_id"):
        d[key] = v.get(key, 0)
    for key in TIMING_KEYS:
        d["t_" + key] = v["timing_ms"].get(key)
    cond = v.get("conditioning") or {}
    d.update(conditioning_outer_wall_ms=cond.get("outer_wall_ms"), cap_reached=v["truncated"], prediction=r["prediction"], gold=json.dumps(r["gold"], ensure_ascii=False),
             generated_token_ids=json.dumps(v["generated_token_ids"]), k_per_layer=json.dumps(r["k_per_layer"]) if r["k_per_layer"] is not None else "",
             history_sha256=r["history_sha256"], physical_execution_id=r["physical_execution_id"], logical_request_id=r["logical_request_id"])
    return d


def summaries(rows):
    summary, scope = [], []
    for m in METHODS:
        rs = [r for r in rows if r["method_id"] == m]
        if not rs:
            continue
        hit = [r for r in rs if r["turn_id"] > 1]
        one = [r for r in rs if r["turn_id"] == 1]
        x = {"method_id": m, "label": LABELS[m], "requests": len(rs), "followups": len(hit), "all_correct": sum(r["score"] for r in rs), "all_acc": mean([r["score"] for r in rs]),
             "followup_correct": sum(r["score"] for r in hit), "followup_acc": mean([r["score"] for r in hit]), "T1_ttft_mean_ms": mean([r["ttft_ms"] for r in one]),
             "followup_ttft_mean_ms": mean([r["ttft_ms"] for r in hit]), "followup_ttft_p50_ms": pct([r["ttft_ms"] for r in hit], 50), "followup_ttft_p95_ms": pct([r["ttft_ms"] for r in hit], 95),
             "followup_e2e_mean_ms": mean([r["request_e2e_ms"] for r in hit]), "followup_e2e_p95_ms": pct([r["request_e2e_ms"] for r in hit], 95),
             "SSD_MB_per_followup": mean([r["ssd_read_bytes"] / 1e6 for r in hit]), "read_calls_per_followup": mean([r["ssd_preads"] for r in hit]),
             "pread_per_followup": mean([r["ssd_pread_calls"] for r in hit]), "preadv_per_followup": mean([r["ssd_preadv_calls"] for r in hit]),
             "gpu_cache_kv_MB_per_followup": mean([r["gpu_cache_kv_bytes"] / 1e6 for r in hit]), "h2d_kv_MB_per_followup": mean([r["h2d_kv_bytes"] / 1e6 for r in hit]),
             "peak_gpu_allocated_bytes_max": max(r["peak_gpu_allocated_bytes"] for r in rs), "followup_peak_gpu_allocated_mean": mean([r["peak_gpu_allocated_bytes"] for r in hit])}
        for t in (1, 2, 3):
            ts = [r for r in rs if r["turn_id"] == t]
            x.update({f"acc_T{t}": mean([r["score"] for r in ts]), f"correct_T{t}": sum(r["score"] for r in ts), f"count_T{t}": len(ts)})
        for key in TIMING_KEYS:
            vals = [r["t_" + key] for r in hit if r["t_" + key] is not None]
            x[f"followup_{key}_mean_ms"] = mean(vals)
        summary.append(x)
        for name, selected in [("T1", one), ("T2", [r for r in rs if r["turn_id"] == 2]), ("T3", [r for r in rs if r["turn_id"] == 3]), ("all", rs), ("followup", hit)]:
            z = {"method_id": m, "scope": name, "requests": len(selected), "correct": sum(r["score"] for r in selected), "accuracy": mean([r["score"] for r in selected])}
            for metric in ("ttft_ms", "request_e2e_ms"):
                values = [r[metric] for r in selected]
                z[metric + "_mean"] = mean(values)
                z[metric + "_p50"] = pct(values, 50)
                z[metric + "_p95"] = pct(values, 95)
            scope.append(z)
    return summary, scope


def sessions(run, rows):
    setups, details, stages = {}, [], []
    for cp in (run / "main/images").glob("*/COMMITTED.json"):
        com = json.loads(cp.read_text())
        folder = cp.parent / com["attempt_id"]
        iid = cp.parent.name
        for m in METHODS:
            if m == "recompute":
                setups[iid, m] = {"source_dialog_id": None, "setup_ms": 0, "write_bytes": 0, "capture_tail_ms": 0, "writer_ms": 0, "activation_ms": 0}
                continue
            p = json.loads((folder / f"{m}_persistence.json").read_text())
            a = json.loads((folder / f"{m}_activation.json").read_text())
            setups[iid, m] = {"source_dialog_id": p["source"]["dialog_id"], "setup_ms": p["charged_setup_before_activation_ms"] + a["activation_ms"], "write_bytes": p["bytes_written"],
                              "capture_tail_ms": p["post_generation_capture_tail_ms"], "writer_ms": p["outer_writer_wall_ms"], "activation_ms": a["activation_ms"]}
            stages.append({"image_id": iid, "method_id": m, **setups[iid, m], "capture_score_ms": p["capture_score_ms"], "capture_clone_ms": p["capture_clone_ms"],
                           **{k: v for k, v in p["timing_ms"].items()}, "native_writer_wall_ms": p["native_writer_wall_ms"], "metadata_resident_bytes": a["metadata_resident_bytes"],
                           "activation_bytes": a["activation_io"]["bytes"], "allocated_bytes": p["allocated_bytes"], "os_write_bytes": p["os_write_bytes"],
                           "scope": "score and clone inside the measured post-generation tail of the source T1; materialize/repack/write/fsync nested in the writer wall; not additive"})
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["image_id"], r["dialog_id"], r["method_id"]].append(r)
    for (iid, did, m), rs in grouped.items():
        s = setups[iid, m]
        source = did == s["source_dialog_id"]
        request = sum(r["request_e2e_ms"] for r in rs)
        details.append({"image_id": iid, "dialog_id": did, "method_id": m, "request_E2E_sum_ms": request, "source_dialogue": source, "actual_setup_ms": s["setup_ms"] if source else 0,
                        "shared_stream_session_E2E_ms": request + (s["setup_ms"] if source else 0), "derived_standalone_E2E_ms": request + s["setup_ms"],
                        "standalone_scope": "DERIVED: the measured image setup is re-charged to every dialogue; non-source capture-hook increment NOT_MEASURED"})
    summary = []
    for m in METHODS:
        ds = [r for r in details if r["method_id"] == m]
        ss = [s for (i, mm), s in setups.items() if mm == m]
        if ds:
            summary.append({"method_id": m, "label": LABELS[m], "provisioning": "FRESH_IMAGE_STREAMING", "setup_ms_per_image": mean([s["setup_ms"] for s in ss]),
                            "capture_tail_ms_per_image": mean([s["capture_tail_ms"] for s in ss]), "writer_ms_per_image": mean([s["writer_ms"] for s in ss]),
                            "writer_ms_p50": pct([s["writer_ms"] for s in ss], 50), "writer_ms_p95": pct([s["writer_ms"] for s in ss], 95),
                            "activation_ms_per_image": mean([s["activation_ms"] for s in ss]), "write_MB_per_image": mean([s["write_bytes"] / 1e6 for s in ss]),
                            "shared_stream_session_E2E_mean_ms": mean([r["shared_stream_session_E2E_ms"] for r in ds]),
                            "shared_stream_session_E2E_p50_ms": pct([r["shared_stream_session_E2E_ms"] for r in ds], 50),
                            "shared_stream_session_E2E_p95_ms": pct([r["shared_stream_session_E2E_ms"] for r in ds], 95),
                            "derived_standalone_E2E_mean_ms": mean([r["derived_standalone_E2E_ms"] for r in ds]),
                            "derived_standalone_E2E_p50_ms": pct([r["derived_standalone_E2E_ms"] for r in ds], 50),
                            "derived_standalone_E2E_p95_ms": pct([r["derived_standalone_E2E_ms"] for r in ds], 95), "builds": len(ss)})
    return details, summary, stages


def run_audit(run, output, allow_partial=False):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "independent_audit.json").exists():
        raise FileExistsError("new output directory required")
    config = json.loads((run / "config.json").read_text())
    config.update(_file_sha256=sha(run / "config.json"), _manifest_sha256=sha(run / "manifest.json"))
    manifest = json.loads((run / "manifest.json").read_text())
    c = Checks()
    rows, receipts, rawhashes, attempts, counts, geometries = [], [], [], Counter(), {}, []
    ids, physical = set(), set()
    exact = pairs = 0
    for phase in ("integration", "smoke", "main"):
        count = 0
        for cp in sorted((run / phase / "images").glob("*/COMMITTED.json")):
            com = json.loads(cp.read_text())
            folder = cp.parent / com["attempt_id"]
            raw = folder / "raw.jsonl"
            for name, digest in com["evidence_sha256"].items():
                c.require(sha(folder / name) == digest, "committed_evidence_hash", str(folder / name))
            rs = list(read_rows(raw))
            receipt = audit_image_rows(rs, manifest, config, phase, cp.parent.name, folder)
            receipts.append(receipt)
            c.require(receipt["status"] == "PASS", "independent_image_audit", receipt)
            count += len(rs)
            c.require(com["requests"] == len(rs) and sha(raw) == com["raw_sha256"], "commit_rows_and_raw_hash", str(cp))
            c.require(json.loads((folder / "cleanup_receipt.json").read_text())["status"] == "PASS", "cleanup_receipt", str(folder))
            allow = json.loads((folder / "cleanup_allowlist.json").read_text())
            scratch = Path(allow["scratch"])
            creation = json.loads((folder / "scratch_creation.json").read_text())
            c.require(not scratch.exists() and scratch.is_relative_to(run / "scratch") and allow["ownership_run"] == run.name and creation["previously_absent"]
                      and creation["path"] == str(scratch), "only_owned_scratch_cleanup", str(folder))
            events = list(read_rows(folder / "attempts.jsonl"))
            started = [e["physical_execution_id"] for e in events if e["status"] == "STARTED"]
            completed = [e["physical_execution_id"] for e in events if e["status"] == "COMPLETED"]
            c.require(Counter(started) == Counter(completed) == Counter(r["physical_execution_id"] for r in rs), "attempt_event_lineage", str(folder))
            rawhashes.append({"phase": phase, "image_id": cp.parent.name, "path": str(raw), "sha256": sha(raw)})
            if phase == "main":
                exact += receipt["u100_packed_vs_fullload_token_exact"]
                pairs += receipt["u100_packed_vs_fullload_pairs"]
                for r in rs:
                    c.require(r["logical_request_id"] not in ids and r["physical_execution_id"] not in physical, "main_unique", r["logical_request_id"])
                    ids.add(r["logical_request_id"])
                    physical.add(r["physical_execution_id"])
                    rows.append(compact(r))
                for m in STORED:
                    r = next(r for r in rs if r["turn_id"] == 2 and r["method_id"] == m)
                    v = r["result"]
                    geometries.append({"image_id": r["image_id"], "method_id": m, "N": r["N_content"], "k": r["k"], "retention_ratio": r["retention_ratio"], "S": r["S_structural"],
                                       "k_first_layer": r["k_per_layer"][0] if r["k_per_layer"] else r["k"], "k_last_layer": r["k_per_layer"][-1] if r["k_per_layer"] else r["k"],
                                       "k_per_layer": json.dumps(r["k_per_layer"]) if r["k_per_layer"] else "", "read_bytes": r["ssd_read_bytes"], "read_calls": r["ssd_preads"],
                                       "gpu_cache_kv_bytes": v["gpu_cache_kv_bytes"], "full_read_bytes": ((r["N_content"] + 63) // 64 * 64 + r["S_structural"]) * ROW_ALL})
        counts[phase] = count
        for a in (run / phase / "images").glob("*/attempt_*"):
            attempts[phase + "_image_attempts"] += 1
            if (a / "failure.json").exists():
                attempts[phase + "_failed_attempts"] += 1
            if (a / "raw.jsonl").exists():
                attempts[phase + "_completed_physical_rows"] += sum(1 for _ in read_rows(a / "raw.jsonl"))
            if (a / "execution_events.jsonl").exists():
                attempts[phase + "_executed_physical_requests"] += sum(1 for _ in read_rows(a / "execution_events.jsonl"))
    complete = counts["main"] == EXPECT["main"]
    c.require(complete and len({r["image_id"] for r in rows}) == EXPECT["images"] and len({r["dialog_id"] for r in rows}) == EXPECT["dialogues"], "main_full_population")
    for m in METHODS:
        rs = [r for r in rows if r["method_id"] == m]
        c.require(len(rs) == EXPECT["per_method"] and sum(r["turn_id"] > 1 for r in rs) == EXPECT["per_method_hit"], "per_method_population", m)
    c.require(counts["smoke"] == EXPECT["smoke"] and counts["integration"] == EXPECT["integration"], "separate_validation_smoke_population")
    c.require(json.loads((ROOT / "data/mt_gqa/dialogues.json").read_text())["dialogues"] == manifest["dialogues"] and sha(ROOT / "data/mt_gqa/dialogues.json") == manifest["index_sha256"], "frozen_full_dataset")
    for name, digest in config["source_sha256"].items():
        c.require(sha(ROOT / name) == digest, "frozen_runtime_source", name)
    for iid, x in manifest["image_identity"].items():
        c.require(sha(ROOT / x["image_path"]) == x["sha256"], "final_external_image_sha", iid)
    # request by request: read bytes grow with the ratio, and the U100 GPU cache holds exactly the FullLoad cache bytes
    lookup = {(r["dialog_id"], r["turn_id"], r["method_id"]): r for r in rows}
    for r in rows:
        if r["method_id"] == "fullload" and r["turn_id"] > 1:
            chain = [lookup.get((r["dialog_id"], r["turn_id"], m)) for m in PACKED]
            c.require(all(o is not None for o in chain) and all(a["ssd_read_bytes"] <= b["ssd_read_bytes"] and a["k"] <= b["k"] for a, b in zip(chain, chain[1:]))
                      and chain[-1]["gpu_cache_kv_bytes"] == r["gpu_cache_kv_bytes"] and chain[-1]["k"] == r["k"] == r["N_content"], "bytes_monotone_and_U100_cache_equals_fullload", r["logical_request_id"])
    summary, scope = summaries(rows)
    details, sess, stages = sessions(run, rows)
    result = c.result()
    valid = complete and result["status"] == "PASS"
    result.update(QWEN_7_ARM_MAIN="VALID" if valid else "INVALID" if complete else "PARTIAL", main_requests=counts["main"], expected_main_requests=EXPECT["main"],
                  phase_counts=counts, attempt_counts=dict(attempts), failed_or_skipped_requests=EXPECT["main"] - counts["main"],
                  u100_packed_vs_fullload_token_exact=exact, u100_packed_vs_fullload_pairs=pairs, image_receipts=receipts, source_sha256=config["source_sha256"],
                  recorded_freeze_amendments=sorted(p.name for p in run.glob("freeze_amendment_*.json")), llava_gpu="NOT RUN", MT_VQA="NOT RUN", GLM="NOT RUN")
    agreement = {}
    for m in METHODS:
        rs = [r for r in rows if r["method_id"] == m]
        if not rs:
            continue
        hit = [r for r in rs if r["turn_id"] > 1]
        agreement[m] = {"generated_length_min": min(r["generated_token_count"] for r in rs), "generated_length_max": max(r["generated_token_count"] for r in rs),
                        "generated_length_mean": mean([r["generated_token_count"] for r in rs]), "cap_count": sum(r["cap_reached"] for r in rs),
                        "peak_allocated_bytes": max(r["peak_gpu_allocated_bytes"] for r in rs), "peak_reserved_bytes": max(r["peak_gpu_reserved_bytes"] for r in rs),
                        "followup_peak_allocated_mean": mean([r["peak_gpu_allocated_bytes"] for r in hit]), "followup_peak_reserved_mean": mean([r["peak_gpu_reserved_bytes"] for r in hit])}
    diagnostics = {"main_final_rows": len(rows), "duplicate_logical": len(rows) - len(ids), "duplicate_physical": len(rows) - len(physical), "attempts": dict(attempts),
                   "methods": agreement, "image_N_min": min((g["N"] for g in geometries), default=None), "image_N_max": max((g["N"] for g in geometries), default=None)}
    for name, data in [("summary", summary), ("quality_timing_by_scope", scope), ("session_per_dialogue", details), ("session_summary", sess),
                       ("persistence_stage_scope", stages), ("request_diagnostics", rows), ("image_geometry", geometries)]:
        csv_write(output / (name + ".csv"), data)
    dump(output / "independent_audit.json", result)
    dump(output / "raw_hashes.json", rawhashes)
    dump(output / "diagnostics.json", diagnostics)
    terminal = {"execution_state": "FINISHED" if valid else "AUDIT_FAILED" if complete else "PARTIAL", "QWEN_7_ARM_MAIN": result["QWEN_7_ARM_MAIN"], "main_final_requests": counts["main"],
                "expected_main_requests": EXPECT["main"], "independent_audit": result["status"], "at_unix": time.time(), "audit_sha256": sha(output / "independent_audit.json")}
    print(json.dumps(terminal), flush=True)
    if not valid and not allow_partial:
        raise RuntimeError("final independent audit did not validate the full main")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true", help="write the audit of an incomplete run without raising (never a final result)")
    a = p.parse_args()
    run_audit(a.run.resolve(), a.output_dir.resolve(), a.allow_partial)


if __name__ == "__main__":
    main()

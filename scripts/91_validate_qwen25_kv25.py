#!/usr/bin/env python3
"""Qwen Visual-KV25 GPU gate, frozen to the ten correctness-v2 pairs.

The immutable v2 validator is run first for every pair, on fresh run-local
stores. New Visual-KV25 is then compared to a separate gather from captured
canonical BF16 KV. No prior results, stores, or source files are rewritten.
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
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmimpress.qwen25.runner import Qwen25Runner  # noqa: E402
import mmimpress.qwen25.store as qstore  # noqa: E402

# This module supplies the already frozen v2 matched-reference, stock MRoPE,
# mask, and numerical-oracle probes. It never writes to its original run.
v2 = None
SCHEMA = "qwen25-kv25-gpu-correctness-v1"
GATES = tuple(f"G{i}" for i in range(1, 13))
MANIFEST_REL = "runs/qwen25_correctness_v2_20260928T081111Z/validation_manifest.json"
PRIOR_VALIDATION_REL = "runs/qwen25_correctness_v2_20260928T081111Z/gpu_validation/validation.json"
PRIOR_MANIFEST_SHA256 = "6458545772438ef1b41cd35bbe7ec17c319be6f779d1b939fc9a94e0ada928a8"
PRIOR_VALIDATION_SHA256 = "a3b851f6fc291711ef53452b6eab164ae84e6d3801ce717b19b8386a120ad860"
FP32_ATOL = FP32_RTOL = 1e-5


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def verify_freeze(path: Path) -> dict:
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze.get("schema") != "qwen25-kv25-gpu-freeze-v1":
        raise ValueError("GPU freeze schema mismatch")
    files = freeze.get("files")
    required = {"docs/qwen25_kv25_budget_contract.md",
                "docs/qwen25_correctness_contract_v2.md",
                "scripts/91_validate_qwen25_kv25.py",
                "scripts/86_validate_qwen25_v2_rerun.py",
                "scripts/78_validate_qwen25.py",
                "scripts/81_debug_qwen25_correctness.py",
                "mmimpress/qwen25/runner.py", "mmimpress/qwen25/store.py",
                "mmimpress/qwen25/vision.py", MANIFEST_REL, PRIOR_VALIDATION_REL,
                "data/index.json"}
    if not isinstance(files, dict) or not required.issubset(files):
        raise ValueError("GPU freeze does not bind all required inputs")
    for rel, expected in files.items():
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise ValueError("freeze paths must be repository-relative")
        if sha(ROOT / rel) != expected:
            raise ValueError(f"frozen file changed: {rel}")
    if files[MANIFEST_REL] != sha(ROOT / MANIFEST_REL):
        raise ValueError("frozen ten-pair manifest changed")
    if files[PRIOR_VALIDATION_REL] != PRIOR_VALIDATION_SHA256:
        raise ValueError("immutable v2 PASS validation hash differs")
    return {"path": str(path.resolve()), "sha256": sha(path),
            "file_count": len(files), "files": files}


def preflight_manifest(path: Path) -> tuple[dict, dict]:
    if path.resolve() != (ROOT / MANIFEST_REL).resolve():
        raise ValueError("use original frozen v2 ten-pair manifest")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != PRIOR_MANIFEST_SHA256:
        raise ValueError("original ten-pair canonical digest differs")
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if v2.canonical_sha256(body) != PRIOR_MANIFEST_SHA256:
        raise ValueError("original ten-pair manifest content changed")
    rows = manifest["samples"]
    expected = [("n355567", "201751701"), ("n355567", "201751740"),
                ("n355567", "201751873"), ("n9181", "20929611"),
                ("n390187", "201861403"), ("n133585", "202108008"),
                ("n272098", "201535625"), ("n472825", "202101069"),
                ("n450919", "2093976"), ("n37274", "202144724")]
    if [(str(x["image_id"]), str(x["question_id"])) for x in rows] != expected:
        raise ValueError("fixed ten-pair order differs")
    for row in rows:
        if sha(Path(row["image_path"])) != row["image_sha256"]:
            raise ValueError(f"fixed image changed: {row['image_id']}")
    prior = json.loads((ROOT / PRIOR_VALIDATION_REL).read_text(encoding="utf-8"))
    if sha(ROOT / PRIOR_VALIDATION_REL) != PRIOR_VALIDATION_SHA256 or prior["status"] != "PASS":
        raise ValueError("prior v2 PASS evidence changed")
    return manifest, prior


def gate(passed: bool, **evidence) -> dict:
    return {"status": "PASS" if passed else "FAIL", "evidence": evidence}


def _read_stats(result: dict, meta: dict, chunks: int) -> dict:
    io = result["read_io"]
    spans = io["span_details"]
    visual = [x for x in spans if x["kind"] == "visual"]
    structural = [x for x in spans if x["kind"] == "structural"]
    row_bytes = int(meta["row_bytes"])
    wanted = chunks * 64 * row_bytes
    planned_visual = 2 * int(meta["num_layers"]) * wanted
    expected_total = planned_visual + int(meta["bytes_structural_kv"])
    okay = (len(visual) == 2 * meta["num_layers"] and len(structural) == 1
            and all(s["offset"] == 0 and s["requested_bytes"] == wanted for s in visual)
            and structural[0]["offset"] == 0
            and structural[0]["requested_bytes"] == meta["bytes_structural_kv"]
            and io["preads"] == io["spans"] == 2 * meta["num_layers"] + 1
            and result["visual_read_bytes"] == planned_visual
            and result["structural_read_bytes"] == meta["bytes_structural_kv"]
            and result["metadata_read_bytes"] == 0
            and io["bytes"] == expected_total)
    return {"passed": bool(okay), "expected_visual_bytes": planned_visual,
            "actual_visual_bytes": result["visual_read_bytes"],
            "expected_total_bytes": expected_total,
            "actual_total_bytes": io["bytes"], "pread_calls": io["preads"],
            "spans": spans}


@contextmanager
def trace_preads():
    actual = []
    original = qstore.os.pread
    def traced(fd, count, offset):
        payload = original(fd, count, offset)
        actual.append({"fd": int(fd), "requested_bytes": int(count),
                       "offset": int(offset), "returned_bytes": len(payload)})
        return payload
    qstore.os.pread = traced
    try:
        yield actual
    finally:
        qstore.os.pread = original


@contextmanager
def finite_extra_sentinel(k: int, loaded_valid: int, row_bytes: int):
    original = qstore._pread_exact
    changed = [0]
    def with_sentinel(fd, count, offset, io, kind, source):
        payload = original(fd, count, offset, io, kind, source)
        if kind == "visual" and loaded_valid > k:
            blob = bytearray(payload)
            start, end = k * row_bytes, loaded_valid * row_bytes
            if end > len(blob):
                raise AssertionError("sentinel range exceeds returned chunk payload")
            blob[start:end] = b"\x1c\x46" * ((end - start) // 2)  # finite BF16 ~10000
            changed[0] += 1
            return bytes(blob)
        return payload
    qstore._pread_exact = with_sentinel
    try:
        yield changed
    finally:
        qstore._pread_exact = original


@contextmanager
def trace_decode(runner: Qwen25Runner):
    original = runner._decode
    traces = []
    def counted(first_id, cache, *, position_start, attention_mask):
        before = int(cache.get_seq_length())
        result = original(first_id, cache, position_start=position_start,
                          attention_mask=attention_mask)
        traces.append({"cache_before": before,
                       "cache_after": int(cache.get_seq_length()),
                       "position_start": int(position_start),
                       "generated_count": len(result[0])})
        return result
    runner._decode = counted
    try:
        yield traces
    finally:
        runner._decode = original


def memory_prefix(capture, selected_original: list[int]) -> tuple[list, list[int]]:
    n = len(capture.prefix_ids)
    vstart, vcount = capture.visual_start, capture.visual_count
    structural = [i for i in range(n) if i < vstart or i >= vstart + vcount]
    logical = sorted(structural + [vstart + i for i in selected_original])
    indices = torch.tensor(logical, dtype=torch.long,
                           device=capture.layers[0][0].device)
    layers = [(k.index_select(2, indices).detach().clone(),
               v.index_select(2, indices).detach().clone())
              for k, v in capture.layers]
    return layers, logical


def validate_new(runner: Qwen25Runner, sample: dict, sample_dir: Path,
                 capture, old: dict, prior: dict) -> dict:
    meta = json.loads((sample_dir / "repacked_store/meta.json").read_text(encoding="utf-8"))
    canonical_meta = json.loads((sample_dir / "canonical_store/meta.json").read_text(encoding="utf-8"))
    store = runner._activate(sample_dir / "repacked_store", sample["image_sha256"])
    n = int(capture.visual_count)
    k = (n + 3) // 4
    m = (k + 63) // 64
    loaded_valid = min(n, m * 64)
    structural_count = len(capture.prefix_ids) - n
    scores = capture.scores.detach().cpu().float().tolist()
    permutation = sorted(range(n), key=lambda i: (-float(scores[i]), i))
    selected = permutation[:k]
    old_chunks = max(1, min(meta["n_chunks"], int(round(.25 * meta["n_chunks"]))))
    old_k = min(n, old_chunks * 64)
    reference, logical = memory_prefix(capture, selected)
    loaded = store.load_prefix(.25, budget_unit="visual_kv")
    bits = v2.v1._bitwise_layers(reference, loaded.layers)
    metadata_ok = (meta["stored_to_original"] == permutation
                   and list(loaded.logical_indices) == logical
                   and list(loaded.selected_visual_original) == sorted(selected)
                   and list(loaded.selected_visual_stored) == list(range(k))
                   and loaded.selected_chunks == m and loaded.kept_visual_tokens == k
                   and loaded.loaded_valid_visual_rows == loaded_valid
                   and loaded.extra_valid_visual_rows == loaded_valid - k
                   and loaded.structural_rows == structural_count
                   and len(loaded.layers) == meta["num_layers"] == 28)
    g = {}
    old_g = old["gates"]
    g["G1"] = gate(old_g["G1"]["status"] == "PASS" and
                   old_g["G3"]["status"] == "PASS",
                   canonical_bf16_roundtrip=old_g["G1"],
                   inverse_repacked_full100=old_g["G3"])
    g["G2"] = gate(old_g["G2"]["status"] == "PASS",
                   matched_fullload=old_g["G2"])
    full_ids, suffix, positions, deltas = runner._cache_hit_ids(
        meta, sample["question"], ())
    full_prefix_len = len(capture.prefix_ids)
    ref_path = v2.trace_suffix(runner, reference, suffix, positions, full_prefix_len)
    ssd_path = v2.trace_suffix(runner, loaded.layers, suffix, positions, full_prefix_len)
    ref_repeat = v2.trace_suffix(runner, reference, suffix, positions, full_prefix_len)
    ssd_repeat = v2.trace_suffix(runner, loaded.layers, suffix, positions, full_prefix_len)
    with trace_preads() as actual_preads, trace_decode(runner) as decodes:
        production = runner.run_cache(sample_dir / "repacked_store", sample["question"],
                                      budget_ratio=.25, budget_unit="visual_kv",
                                      image_sha256=sample["image_sha256"], return_logits=True)
    production_repeat = runner.run_cache(sample_dir / "repacked_store", sample["question"],
                                         budget_ratio=.25, budget_unit="visual_kv",
                                         image_sha256=sample["image_sha256"], return_logits=True)
    matched = v2._matched_gate(ref_path, ssd_path, ref_repeat, ssd_repeat,
                               production, production_repeat)
    g["G3"] = gate(bits["equal"] and metadata_ok and matched["status"] == "PASS",
                   independent_reference="captured full BF16 cache, score-sorted top-k original IDs, direct gather in original logical order",
                   captured_vs_SSD_BF16_bits=bits, selection_metadata_exact=metadata_ok,
                   matched_serving=matched,
                   first_logits_sha256=v2._logits_hash(production["first_logits"]),
                   first_token_id=production["first_token_id"],
                   generated_token_ids=production["generated_token_ids"])
    shapes_ok = all(kv.shape == (1, meta["num_kv_heads"],
                                    k + structural_count, meta["head_dim"])
                    for pair in loaded.layers for kv in pair)
    structural_positions = [i for i in range(full_prefix_len)
                            if i < capture.visual_start or i >= capture.visual_start + n]
    actual_structural = [i for i in loaded.logical_indices
                         if i < capture.visual_start or i >= capture.visual_start + n]
    g["G4"] = gate(shapes_ok and actual_structural == structural_positions
                   and k == len(selected) == len(loaded.selected_visual_original)
                   and production["kept_tokens"] == k
                   and production["compact_prefix_tokens"] == k + structural_count,
                   all_native_layers_heads_shape_exact=shapes_ok,
                   structural_prefix_indices_exact=actual_structural == structural_positions,
                   content_rows=k, structural_rows=structural_count,
                   compact_prefix_rows=production["compact_prefix_tokens"])
    with finite_extra_sentinel(k, loaded_valid, meta["row_bytes"]) as changes:
        poisoned = store.load_prefix(.25, budget_unit="visual_kv")
        poisoned_result = runner.run_cache(
            sample_dir / "repacked_store", sample["question"], budget_ratio=.25,
            budget_unit="visual_kv", image_sha256=sample["image_sha256"],
            return_logits=True)
    poisoned_bits = v2.v1._bitwise_layers(loaded.layers, poisoned.layers)
    poison_output = v2.path_comparison(v2._result_path(production),
                                       v2._result_path(poisoned_result))
    g["G5"] = gate(changes[0] == 2 * meta["num_layers"] * 2
                   and poisoned_bits["equal"] and poison_output["exact_target"]
                   and production["compact_prefix_tokens"] == k + structural_count,
                   finite_extra_rows_poisoned=loaded_valid-k,
                   visual_spans_modified=changes[0],
                   assembled_BF16_bits_unchanged=poisoned_bits,
                   output_unchanged=poison_output,
                   no_padding_or_extra_rows_in_compact_cache=shapes_ok)
    stock_positions, stock_deltas = v2._stock_positions(
        runner, full_ids, torch.tensor(capture.image_grid_thw, dtype=torch.long))
    pos_ok = (v2._position_identity(stock_positions, positions)
              and torch.equal(stock_deltas, deltas)
              and v2._position_identity(loaded.position_ids,
                                        stock_positions[:, :, torch.tensor(logical)]))
    expected_mask = v2._analytic_mask([True] * len(logical), suffix.shape[1])
    mask_ok = (v2._mask_identity(v2._stock_mask(runner, [True] * len(logical),
                                                suffix.shape[1]), expected_mask)
               and v2._actual_mask_checks(ref_path, expected_mask)["all_layers_exact"]
               and v2._actual_mask_checks(ssd_path, expected_mask)["all_layers_exact"])
    dense = []
    dense_mask = torch.zeros((1, full_prefix_len), dtype=torch.long)
    dense_mask[0, logical] = 1
    for key, value in reference:
        dk = torch.zeros((1, key.shape[1], full_prefix_len, key.shape[3]),
                         dtype=key.dtype, device=key.device)
        dv = torch.zeros_like(dk)
        idx = torch.tensor(logical, dtype=torch.long, device=key.device)
        dk.index_copy_(2, idx, key)
        dv.index_copy_(2, idx, value)
        dense.append((dk, dv))
    dense_path = v2.trace_suffix(runner, dense, suffix, positions,
                                 full_prefix_len, dense_mask)
    expected_dense_mask = v2._analytic_mask(dense_mask[0].bool().tolist(), suffix.shape[1])
    dense_mask_ok = (v2._mask_identity(v2._stock_mask(runner, dense_mask[0].bool().tolist(),
                                                     suffix.shape[1]), expected_dense_mask)
                     and v2._actual_mask_checks(dense_path,
                                                 expected_dense_mask)["all_layers_exact"])
    tc = runner.model.config.text_config
    dim = tc.hidden_size // tc.num_attention_heads
    oracle_compact = v2._fp32_oracle(ref_path, reference, [True] * len(logical),
                                      tc.num_attention_heads, tc.num_key_value_heads, dim)
    oracle_dense = v2._fp32_oracle(dense_path, dense,
                                    dense_mask[0].bool().tolist(),
                                    tc.num_attention_heads, tc.num_key_value_heads, dim)
    oracle_delta = (oracle_compact-oracle_dense).abs()
    oracle_ok = bool(torch.allclose(oracle_compact, oracle_dense,
                                    atol=FP32_ATOL, rtol=FP32_RTOL))
    negative = v2._negative_semantic_controls(stock_positions, expected_mask, logical)
    negative_ok = all(negative[key] for key in
                      ("mrope_perturbation_detected", "mask_perturbation_detected",
                       "mapping_perturbation_detected"))
    g["G6"] = gate(pos_ok and mask_ok and dense_mask_ok and oracle_ok and negative_ok,
                   stock_mrope_exact=pos_ok, stock_and_actual_causal_masks_exact=mask_ok,
                   dense_causal_mask_exact=dense_mask_ok,
                   fp32_oracle_allclose_1e_minus_5=oracle_ok,
                   fp32_oracle_max_abs=float(oracle_delta.max()),
                   negative_controls=negative)
    decoded = decodes[0] if len(decodes) == 1 else None
    expected_before = len(logical) + int(suffix.shape[1])
    expected_after = (expected_before + len(production["generated_token_ids"]) - 1)
    decode_ok = bool(decoded and decoded["cache_before"] == expected_before
                     and decoded["cache_after"] == expected_after
                     and decoded["position_start"] == int(stock_positions.max()) + 1
                     and decoded["generated_count"] == len(production["generated_token_ids"])
                     and ref_path["predecode_cache_len"] == expected_before
                     and ssd_path["predecode_cache_len"] == expected_before
                     and runner.model.model.rope_deltas is None
                     and old_g["G10"]["status"] == "PASS")
    g["G7"] = gate(decode_ok, production_decode=decoded,
                   expected_prefill_cache_length=expected_before,
                   expected_final_cache_length=expected_after,
                   full_logical_first_decode_position=int(stock_positions.max())+1,
                   prior_v2_state_isolation=old_g["G10"])
    from mmimpress.qwen25.vision import VisionScoreCapture
    score_calls = [0]
    original_score = VisionScoreCapture._compute_scores
    def counted_score(self):
        score_calls[0] += 1
        return original_score(self)
    VisionScoreCapture._compute_scores = counted_score
    try:
        no_score = runner.run_cache(sample_dir / "repacked_store", sample["question"],
                                    budget_ratio=.25, budget_unit="visual_kv",
                                    image_sha256=sample["image_sha256"])
    finally:
        VisionScoreCapture._compute_scores = original_score
    g["G8"] = gate(old_g["G5"]["status"] == "PASS" and
                   production["vision_calls"] == no_score["vision_calls"] == 0
                   and production["online_query_score_calls"] == 0
                   and score_calls[0] == 0,
                   source_T1_vision_calls=1,
                   hit_vision_calls=production["vision_calls"],
                   online_query_score_calls=score_calls[0],
                   source_T1_evidence=old_g["G5"])
    io_check = _read_stats(production, meta, m)
    # Real syscall trace, independent of the store's IOCounter bookkeeping.
    actual_bytes = sum(x["returned_bytes"] for x in actual_preads)
    actual_calls = len(actual_preads)
    trace_ok = (actual_calls == production["pread_calls"]
                and actual_bytes == production["read_io"]["bytes"]
                and all(x["offset"] == 0 and x["returned_bytes"] == x["requested_bytes"]
                        for x in actual_preads))
    g["G9"] = gate(io_check["passed"] and trace_ok and m == (k+63)//64
                   and production["loaded_valid_visual_rows"] == loaded_valid
                   and production["extra_valid_visual_rows"] == loaded_valid-k,
                   planned_vs_returned=io_check, actual_syscall_count=actual_calls,
                   actual_syscall_returned_bytes=actual_bytes,
                   os_pread_trace_exact=trace_ok, selected_chunks=m,
                   extra_valid_real_rows=loaded_valid-k,
                   padding_rows=m*64-loaded_valid)
    old_ref = prior["methods"]["ours25"]
    new_old = old["methods"]["ours25"]
    prior_exact = (new_old["first_logits_sha256"] == old_ref["first_logits_sha256"]
                   and new_old["generated_token_ids"] == old_ref["generated_token_ids"]
                   and new_old["prediction"] == old_ref["prediction"])
    legacy_gates = ("G4", "G7", "G10", "G11", "G12", "G13")
    legacy_ok = all(old_g[name]["status"] == "PASS" for name in legacy_gates)
    g["G10"] = gate(prior_exact and legacy_ok,
                    immutable_prior_v2_first_logits_and_tokens_exact=prior_exact,
                    prior_v2_methods=old_ref, current_legacy=new_old,
                    prior_v2_gate_statuses={x:old_g[x]["status"] for x in legacy_gates})
    same_set = selected == permutation[:old_k]
    same_set_identity = None
    if same_set:
        same_set_identity = (production["generated_token_ids"] ==
                             old["methods"]["ours25"]["generated_token_ids"]
                             and v2._logits_hash(production["first_logits"]) ==
                             old["methods"]["ours25"]["first_logits_sha256"])
    g["G11"] = gate((not same_set or bool(same_set_identity))
                    and old_k != k and loaded_valid > k,
                    old_kept_content_rows=old_k, new_kept_content_rows=k,
                    same_selected_set=same_set,
                    conditional_equal_set_output_identity=same_set_identity,
                    changed_set_answer_identity_not_required=True,
                    extra_real_rows=loaded_valid-k)
    g["G12"] = {"status":"NOT RUN", "evidence":{
        "reason":"LLaVA CPU and protected-file check supplied by root as separate receipt"}}
    return {"gates":g, "geometry":{"N_content":n,"k_target":k,"m_chunks":m,
                                   "legacy_kept_count":old_k,
                                   "loaded_valid_real_rows":loaded_valid,
                                   "extra_valid_real_rows":loaded_valid-k,
                                   "structural_count":structural_count,
                                   "selected_original_ids":selected,
                                   "permutation_sha256":meta["permutation_sha256"]},
            "new_method":{"first_token_id":production["first_token_id"],
                          "generated_token_ids":production["generated_token_ids"],
                          "prediction":production["prediction"],
                          "first_logits_sha256":v2._logits_hash(production["first_logits"]),
                          "read_io":production["read_io"],
                          "h2d_kv_bytes":production["h2d_kv_bytes"],
                          "gpu_cache_kv_bytes":production["gpu_cache_kv_bytes"],
                          "compact_prefix_tokens":production["compact_prefix_tokens"],
                          "ttft_ms":production["ttft_ms"]}}


def summarize(report: dict) -> None:
    for name in GATES:
        statuses = [s.get("new",{}).get("gates",{}).get(name,{}).get("status","NOT RUN")
                    for s in report["samples"]]
        if name == "G12":
            report["gates"][name] = report.get("llava_protection", {"status":"NOT RUN"})
            continue
        if "FAIL" in statuses:
            status = "FAIL"
        elif "UNRESOLVED" in statuses:
            status = "UNRESOLVED"
        elif len(statuses) == 10 and all(x == "PASS" for x in statuses):
            status = "PASS"
        else:
            status = "NOT RUN"
        report["gates"][name] = {"status":status,"passed_samples":statuses.count("PASS"),
                                  "failed_samples":statuses.count("FAIL"),
                                  "not_run_samples":statuses.count("NOT RUN")}
    statuses = [report["gates"][name]["status"] for name in GATES]
    report["status"] = ("FAIL" if "FAIL" in statuses else
                         "UNRESOLVED" if "UNRESOLVED" in statuses else
                         "PASS" if all(x=="PASS" for x in statuses) else "NOT RUN")
    report["pilot_eligible"] = report["status"] == "PASS"


def main() -> int:
    global v2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--llava-protection-receipt", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    import importlib.util
    spec = importlib.util.spec_from_file_location("qwen25_v2_readonly_kv25", ROOT / "scripts/86_validate_qwen25_v2_rerun.py")
    v2 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = v2
    spec.loader.exec_module(v2)
    freeze = verify_freeze(args.freeze)
    manifest, prior = preflight_manifest(ROOT / MANIFEST_REL)
    q_shape = v2._verify_q_shape_evidence(ROOT / v2.Q_SHAPE_EVIDENCE_REL,
                                          v2.Q_SHAPE_EVIDENCE_SHA256, manifest)
    receipt = json.loads(args.llava_protection_receipt.read_text(encoding="utf-8"))
    llava_pass = (receipt.get("cpu_passed") is True
                  and receipt.get("protected_files_passed") is True
                  and receipt.get("llava_gpu") in ("NOT RUN", "PASS")
                  and sha(Path(receipt["cpu_log"])) == receipt["cpu_log_sha256"])
    if not llava_pass:
        raise ValueError("LLaVA CPU/protection receipt is incomplete")
    if args.preflight_only:
        print(json.dumps({"status":"PASS","freeze":freeze["sha256"],
                          "manifest":manifest["manifest_sha256"],
                          "sample_count":len(manifest["samples"]),
                          "llava_protection_passed":llava_pass},sort_keys=True))
        return 0
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report_path = out / "validation.json"
    report = {"schema":SCHEMA,"created_utc":datetime.now(timezone.utc).isoformat(),
              "status":"NOT RUN","pilot_eligible":False,
              "source_freeze":freeze,"manifest_sha256":manifest["manifest_sha256"],
              "prior_v2_validation_sha256":PRIOR_VALIDATION_SHA256,
              "tolerances":{"matched_KV_and_logits":"bitwise exact",
                            "fp32_oracle_atol":FP32_ATOL,
                            "fp32_oracle_rtol":FP32_RTOL,
                            "v1_numeric_diagnostic_atol":.125,
                            "v1_numeric_diagnostic_rtol":.02},
              "llava_protection":{"status":"PASS","receipt":str(args.llava_protection_receipt.resolve()),
                                  "receipt_sha256":sha(args.llava_protection_receipt),
                                  "llava_gpu":receipt["llava_gpu"]},
              "gates":{name:{"status":"NOT RUN"} for name in GATES},
              "samples":[]}
    write_json(report_path,report)
    runner = Qwen25Runner(attn="sdpa")
    first_store = None
    try:
        runner.load()
        report["runtime"] = runner.runtime_fingerprint()
        write_json(report_path,report)
        for index,sample in enumerate(manifest["samples"]):
            start=time.perf_counter()
            sample_dir=out/f"sample_{index:02d}_{sample['image_id']}_{sample['question_id']}"
            captured={}
            orig=runner.run_pixels
            def intercept(*a,**kw):
                result=orig(*a,**kw)
                if result.get("capture") is not None:
                    captured["prefix"]=result["capture"]
                return result
            runner.run_pixels=intercept
            try:
                old=v2.validate_sample(runner,sample,out,index,q_shape)
                runner.run_pixels=orig
                if any(old["gates"][name]["status"] != "PASS" for name in v2.PER_SAMPLE_GATES):
                    raise AssertionError("legacy v2 per-sample gate failed")
                if old["diagnostics"]["numerical_branch_status"] != "PASS":
                    raise AssertionError("legacy v2 numerical branch unresolved")
                prior_sample=prior["samples"][index]
                if prior_sample["image_id"] != sample["image_id"] or prior_sample["question_id"] != sample["question_id"]:
                    raise AssertionError("prior v2 sample pairing differs")
                new=validate_new(runner,sample,sample_dir,captured["prefix"],old,prior_sample)
                entry={"sample_index":index,"image_id":sample["image_id"],
                       "question_id":sample["question_id"],"old_v2_status":"PASS",
                       "old_v2_numerical_branch_status":old["diagnostics"]["numerical_branch_status"],
                       "new":new,"duration_seconds":time.perf_counter()-start}
                if index==0:
                    first_store=(sample_dir,sample,new)
                if index==2 and first_store is not None:
                    base_dir,base_sample,base_new=first_store
                    same_image=all(manifest["samples"][i]["image_id"]==base_sample["image_id"] for i in range(3))
                    history=[]
                    histories=[]
                    for turn in range(3):
                        row=manifest["samples"][turn]
                        hit=runner.run_cache(base_dir/"repacked_store",row["question"],
                                             history=tuple(history),budget_ratio=.25,
                                             budget_unit="visual_kv",
                                             image_sha256=base_sample["image_sha256"])
                        histories.append({"turn":turn+1,"selected_ids":hit["selected_visual_original"],
                                          "generated_token_ids":hit["generated_token_ids"],
                                          "prediction":hit["prediction"],
                                          "vision_calls":hit["vision_calls"]})
                        history.append((row["question"],hit["prediction"]))
                    hist_ok=(same_image and all(h["selected_ids"]==histories[0]["selected_ids"]
                                                and h["vision_calls"]==0 for h in histories))
                    report["same_image_method_local_history"]={"status":"PASS" if hist_ok else "FAIL",
                                                                 "turns":histories}
                    other_image=runner.run_cache(base_dir/"repacked_store",base_sample["question"],
                                                 budget_ratio=.25,budget_unit="visual_kv",
                                                 image_sha256=base_sample["image_sha256"],return_logits=True)
                    interleaved=(other_image["generated_token_ids"]==base_new["new_method"]["generated_token_ids"]
                                 and v2._logits_hash(other_image["first_logits"])==base_new["new_method"]["first_logits_sha256"])
                    report["cross_image_interleaved_repeat"]={"status":"PASS" if interleaved else "FAIL"}
                    v2._release_sample_stores(runner,0,base_sample,out)
                    first_store=None
            except Exception as exc:
                runner.run_pixels=orig
                entry={"sample_index":index,"image_id":sample["image_id"],
                       "question_id":sample["question_id"],
                       "status":"UNRESOLVED","error":f"{type(exc).__name__}: {exc}",
                       "traceback":traceback.format_exc(limit=20),
                       "duration_seconds":time.perf_counter()-start}
            report["samples"].append(entry)
            summarize(report)
            write_json(report_path,report)
            if index != 0 or first_store is None:
                v2._release_sample_stores(runner,index,sample,out)
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        runner.close()
    # Cross-request and method-local history controls are required, not advisory.
    if report.get("same_image_method_local_history",{}).get("status") != "PASS" or report.get("cross_image_interleaved_repeat",{}).get("status") != "PASS":
        report["gates"]["G7"]["status"]="FAIL"
        report["gates"]["G8"]["status"]="FAIL"
        report["status"]="FAIL"
        report["pilot_eligible"]=False
    write_json(report_path,report)
    print(report_path)
    return 0 if report["pilot_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

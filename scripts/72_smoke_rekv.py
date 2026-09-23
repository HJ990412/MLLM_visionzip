#!/usr/bin/env python3
"""Three-image actual-model correctness gate for the ReKV SSD adaptation.

Each image uses the frozen GQA index and questions[4:6]. Q1 is exactly one
normal pixel inference with direct pre-RoPE K/V capture. The resulting raw-K
store serves Q2 and Q3; Q2 also has an all-block ReKV and FullLoad diagnostic.
All artifacts are run-local and persisted without replacing existing paths.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mmimpress.config import CHUNK_SIZE, MODEL_ID  # noqa: E402
from mmimpress.model import LlavaRunner, cache_layers  # noqa: E402
from mmimpress.rekv import ReKVServer, official_dot_scores  # noqa: E402
from mmimpress.rekv_store import (  # noqa: E402
    ReKVContext, persist_captured_rekv_prefix,
    validate_pre_rope_capture_against_cache,
)
from mmimpress.serve import ImageContext, Server  # noqa: E402
from mmimpress.store import write_image_store  # noqa: E402

SCHEMA_VERSION = "rekv-real-model-smoke-v1"
N_IMAGES = 3
SEED = 1234
MAX_NEW_TOKENS = 16
LOGIT_RTOL = 2.0e-2  # frozen quantized-model precedent: 65_validate_mpic.py
LOGIT_ATOL = 2.0e-2
REQUIRED_CHECKS = (
    "representative_parity", "similarity_parity", "pre_rope_capture",
    "compact_cache", "rope_mask_position", "multi_layer_dependency",
    "retrieval_answer_handoff", "request_isolation",
    "all_block_diagnostic", "actual_model_smoke", "duplicate_payload_read",
)


def _load_runner_module():
    path = ROOT / "scripts/70_eval_rekv_gqa.py"
    spec = importlib.util.spec_from_file_location("rekv_pilot_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen six-arm GQA runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atomic_json(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(path)
    payload = json.dumps(value, indent=2, ensure_ascii=False,
                         allow_nan=False, sort_keys=True).encode() + b"\n"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.rename(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_unit_gates(run_dir: Path) -> dict:
    results = {}
    for name in ("test_rekv.py", "test_rekv_store.py"):
        command = [sys.executable, "-m", "unittest", "discover",
                   "-s", "tests", "-p", name, "-v"]
        process = subprocess.run(command, cwd=ROOT, text=True,
                                 capture_output=True, check=False)
        log_path = run_dir / f"{name}.log"
        log_path.write_text(process.stdout + process.stderr, encoding="utf-8")
        results[name] = {
            "passed": process.returncode == 0,
            "returncode": process.returncode,
            "log": log_path.name,
            "log_sha256": _sha256(log_path),
        }
    return {
        "passed": all(row["passed"] for row in results.values()),
        "suites": results,
        "source_reference_commit": "1fd9a3dbf5dbff7f27069ae2f4463674c495e830",
    }


@torch.no_grad()
def _representative_and_similarity_checks(runner, capture, context, question):
    """Compare sampled metadata and actual layer-0 question dot products."""
    device = runner.model.device
    meta = context.meta
    start, v_num = int(meta["v_token_start"]), int(meta["v_token_num"])
    separators = set(int(x) for x in meta["newline_idx"])
    candidate = [int(x) for x in meta["normal_candidate_chunk_ids"]]
    sampled = sorted({candidate[0], candidate[len(candidate) // 2],
                      candidate[-1]})
    representative_layers = []
    for li, (raw_k, _raw_v) in enumerate(capture.result_cpu()):
        matching = []
        for ci in sampled:
            lo = ci * int(meta["chunk_size"])
            hi = min(lo + int(meta["chunk_size"]), v_num)
            valid = [pos for pos in range(lo, hi) if pos not in separators]
            expected = raw_k[start + torch.tensor(valid)].to(device).mean(
                dim=0).reshape(-1).to(torch.bfloat16)
            actual = context.k_rep[li, ci]
            matching.append(bool(torch.equal(expected, actual)))
        representative_layers.append(all(matching))

    token_ids = runner.processor.tokenizer(
        question, return_tensors="pt").input_ids.to(device)
    embedding = runner.model.get_input_embeddings()(token_ids)
    attention = runner.layers[0].self_attn
    normalized = runner.layers[0].input_layernorm(embedding)
    h, d = int(meta["num_heads"]), int(meta["head_dim"])
    q_raw = attention.q_proj(normalized).view(
        1, token_ids.shape[1], h, d).transpose(1, 2)
    q_rep = q_raw.mean(dim=2).reshape(-1)
    actual_scores = official_dot_scores(q_rep, context.k_rep[0])
    expected_scores = torch.einsum(
        "cd,d->c", context.k_rep[0].float(), q_rep.float())
    diff = (actual_scores - expected_scores).abs()
    return {
        "representative_parity": all(representative_layers),
        "representative_layers_exact": representative_layers,
        "sampled_chunk_ids": sampled,
        "similarity_parity": bool(torch.allclose(
            actual_scores, expected_scores, rtol=1e-5, atol=1e-4)),
        "similarity_max_abs_error": float(diff.max().item()),
        "mean_dtype": str(q_rep.dtype),
        "metadata_dtype": str(context.k_rep.dtype),
        "similarity_dtype": str(actual_scores.dtype),
    }


def _compact_checks(result: dict, meta: dict) -> dict:
    layers = int(meta["num_layers"])
    mappings = result["source_to_compact_positions_per_layer"]
    lengths = result["compact_prefix_lengths"]
    selected = result["selected_chunk_ids_per_layer"]
    candidate = set(int(x) for x in meta["normal_candidate_chunk_ids"])
    budget = int(result["normal_selected_chunk_count"])
    exact = len(mappings) == len(lengths) == len(selected) == layers
    strict_compact = False
    if exact:
        exact = all(
            len(mapping) == length
            and [int(pair[1]) for pair in mapping] == list(range(length))
            and [int(pair[0]) for pair in mapping] == sorted(
                {int(pair[0]) for pair in mapping})
            and len(chunks) == budget
            and set(int(x) for x in chunks) <= candidate
            for mapping, length, chunks in zip(mappings, lengths, selected))
        strict_compact = any(
            int(length) < int(meta["prefix_len"]) for length in lengths)
    return {
        "passed": bool(exact and strict_compact),
        "all_mappings_compact_and_ordered": bool(exact),
        "at_least_one_layer_shorter_than_full_prefix": strict_compact,
        "compact_prefix_lengths": lengths,
        "full_prefix_length": int(meta["prefix_len"]),
        "selected_chunks_per_layer": selected,
        "attention_key_lengths": result["actual_attention_key_lengths"],
    }


def _generation_checks(result: dict, tokenizer) -> bool:
    tokens = result.get("generated_token_ids")
    if not isinstance(tokens, list) or not 1 <= len(tokens) <= MAX_NEW_TOKENS:
        return False
    if int(tokens[0]) != int(result["first_token_id"]):
        return False
    if int(result["generated_token_count"]) != len(tokens):
        return False
    if not isinstance(result.get("answer"), str):
        return False
    if not (0 <= int(tokens[0]) < len(tokenizer)):
        return False
    if any(int(token) == tokenizer.eos_token_id for token in tokens[:-1]):
        return False
    return True


def _reference_full_logits(runner, legacy_server, cache, ids, size, store_dir,
                           question):
    v_start, v_num = runner.visual_span(ids)
    _, _, _, separators = runner.anyres_layout(size, v_num)
    write_image_store(
        store_dir, cache_layers(cache), v_start, v_num,
        ids[0, :v_start + v_num].detach().cpu().tolist(),
        separators, chunk_size=CHUNK_SIZE, dtype="float16",
        stored_to_original=None, separator_sidecar=True)
    context = ImageContext(store_dir, runner.model.device,
                           require_v_hidden=False)
    captured = {}
    lm_head = runner.model.lm_head

    def logit_hook(_module, _inputs, output):
        if "first_logits" not in captured:
            captured["first_logits"] = output[0, -1].detach().float().cpu()

    handle = lm_head.register_forward_hook(logit_hook)
    try:
        reference = legacy_server.request(
            context, question=question, mode="fullload", cold=True)
    finally:
        handle.remove()
        context.close()
    if "first_logits" not in captured:
        raise AssertionError("FullLoad reference prefill logits were not captured")
    return reference, captured["first_logits"]


def _all_block_comparison(result: dict, reference: dict,
                          reference_logits: torch.Tensor) -> dict:
    actual = result["first_logits"]
    if actual is None or actual.shape != reference_logits.shape:
        raise AssertionError("all-block reference logits missing or wrong shape")
    finite = bool(torch.isfinite(actual).all()
                  and torch.isfinite(reference_logits).all())
    difference = (actual - reference_logits).abs()
    matching = bool(torch.allclose(
        actual, reference_logits, rtol=LOGIT_RTOL, atol=LOGIT_ATOL))
    token_matching = (int(result["first_token_id"])
                      == int(reference["first_token_id"]))
    return {
        "passed": bool(finite and matching and token_matching),
        "finite": finite,
        "logits_allclose": matching,
        "first_token_equal": token_matching,
        "rekv_first_token_id": int(result["first_token_id"]),
        "fullload_first_token_id": int(reference["first_token_id"]),
        "max_abs_logit_error": float(difference.max().item()),
        "mean_abs_logit_error": float(difference.mean().item()),
        "logit_tolerance": {"rtol": LOGIT_RTOL, "atol": LOGIT_ATOL},
        "all_block_compact_lengths": result["compact_prefix_lengths"],
    }


def _small_result(result: dict) -> dict:
    keep = (
        "answer", "first_token_id", "generated_token_ids",
        "generated_token_count", "ttft_ms", "request_e2e_ms",
        "retrieval_forward_wall_ms", "answer_prefill_wall_ms",
        "selected_chunk_ids_per_layer", "compact_prefix_lengths",
        "actual_attention_key_lengths", "stage_a_payload_read_bytes",
        "stage_b_payload_read_bytes", "ssd_total_bytes", "pread_count",
        "duplicate_read_bytes", "pread_trace", "fadvise_statuses",
        "source_to_compact_positions_per_layer", "n_init", "n_local",
    )
    return {key: result[key] for key in keep if key in result}


def _one_image(index: int, entry: dict, run_dir: Path, runner,
               legacy_server, rekv_server, all_block_server, pilot) -> dict:
    image_id = str(entry["image_id"])
    questions = entry["questions"][
        pilot.QUESTION_SKIP:pilot.QUESTION_SKIP + pilot.QUESTIONS_PER_IMAGE]
    if len(questions) != 6:
        raise AssertionError("frozen GQA question slice is incomplete")
    q1, q2, q3 = questions[:3]
    image_path = ROOT / entry["image_path"]
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    pixel, diagnostic = pilot._run_rekv_pixels(
        runner, legacy_server, image, q1["question"])
    if int(pixel["vision_forward_count"]) != 1:
        raise AssertionError("Turn-1 ReKV capture changed vision-forward count")
    capture = diagnostic["rekv_capture"]
    pre_rope = validate_pre_rope_capture_against_cache(
        runner, capture, pixel["captured_past_key_values"])
    if not pre_rope["passed"]:
        raise AssertionError("pre-RoPE capture did not reproduce normal cache")
    encoded = diagnostic["enc_cpu"]
    raw_path = run_dir / "stores" / "rekv" / image_id
    raster_path = run_dir / "stores" / "raster_reference" / image_id
    persisted = persist_captured_rekv_prefix(
        runner, pixel["captured_past_key_values"],
        encoded["input_ids"], encoded["image_sizes"][0],
        capture, raw_path, image_id=image_id,
        chunk_size=CHUNK_SIZE,
        image_input_sha256=diagnostic["image_input_sha256"])
    context = ReKVContext(raw_path, runner.model.device, runner=runner)
    try:
        metadata = _representative_and_similarity_checks(
            runner, capture, context, q2["question"])
        source_before = context.source_payload_hash
        if source_before != context.meta["source_payload_sha256"]:
            raise AssertionError("new ReKV source hash disagrees with manifest")

        q2_result, _ = pilot._run_rekv(
            runner, rekv_server, context, q2["question"],
            int(context.meta["bytes_visual_kv"]))
        q3_result, _ = pilot._run_rekv(
            runner, rekv_server, context, q3["question"],
            int(context.meta["bytes_visual_kv"]))
        all_result, _ = pilot._run_rekv(
            runner, all_block_server, context, q2["question"],
            int(context.meta["bytes_visual_kv"]))
        source_after = context.source_payload_hash
        reference, reference_logits = _reference_full_logits(
            runner, legacy_server, pixel["captured_past_key_values"],
            encoded["input_ids"], encoded["image_sizes"][0],
            raster_path, q2["question"])
        all_block = _all_block_comparison(
            all_result, reference, reference_logits)
        all_block["full_source_order_assembled"] = all(
            int(length) == int(context.meta["prefix_len"])
            for length in all_result["compact_prefix_lengths"])
        all_block["passed"] = bool(
            all_block["passed"] and all_block["full_source_order_assembled"])

        q2_compact = _compact_checks(q2_result, context.meta)
        q3_compact = _compact_checks(q3_result, context.meta)
        generations = [
            _generation_checks(row, runner.processor.tokenizer)
            for row in (q2_result, q3_result, all_result)]
        source_unchanged = source_before == source_after
        no_duplicate = all(
            int(row["duplicate_read_bytes"]) == 0
            and int(row["stage_b_payload_read_bytes"]) == 0
            and int(row["pread_count"]) == len(row["pread_trace"])
            for row in (q2_result, q3_result, all_result))
        handoff = all(
            int(row["stage_a_payload_read_bytes"]) > 0
            and int(row["stage_b_payload_read_bytes"]) == 0
            and len(row["answer_prefill_key_lengths_per_layer"])
            == int(context.meta["num_layers"])
            and all(int(after) > int(before)
                    for before, after in zip(
                        row["compact_prefix_lengths"],
                        row["answer_prefill_key_lengths_per_layer"]))
            for row in (q2_result, q3_result, all_result))
        isolation = bool(
            q2_result["question_token_ids"] != q3_result["question_token_ids"]
            and source_unchanged
            and all(len(row["retrieval_layer_log"])
                    == int(context.meta["num_layers"])
                    for row in (q2_result, q3_result)))
        positions = bool(pre_rope["passed"]
                         and all_block["passed"]
                         and all(row["branch"] == "local_only"
                                 for row in q2_result["retrieval_layer_log"]))
        fadvise = all(
            row["fadvise_statuses"]
            and all(item["status"] == "ok"
                    for item in row["fadvise_statuses"])
            for row in (q2_result, q3_result, all_result))
        checks = {
            "representative_parity": bool(metadata["representative_parity"]),
            "similarity_parity": bool(metadata["similarity_parity"]),
            "pre_rope_capture": bool(pre_rope["passed"]),
            "compact_cache": bool(q2_compact["passed"]
                                  and q3_compact["passed"]),
            "rope_mask_position": positions,
            "multi_layer_dependency": bool(
                all(len(row["retrieval_layer_log"])
                    == int(context.meta["num_layers"])
                    for row in (q2_result, q3_result))),
            "retrieval_answer_handoff": bool(handoff),
            "request_isolation": isolation,
            "all_block_diagnostic": bool(all_block["passed"]),
            "actual_model_smoke": bool(
                all(generations) and fadvise
                and 1 <= int(pixel["generated_tokens"]) <= MAX_NEW_TOKENS
                and isinstance(pixel["answer"], str)
                and int(pixel["vision_forward_count"]) == 1
                and all(float(row["ttft_ms"]) > 0
                        for row in (q2_result, q3_result, all_result))),
            "duplicate_payload_read": bool(no_duplicate
                                           and source_unchanged),
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "image_index": index,
            "image_id": image_id,
            "image_path": str(image_path.relative_to(ROOT)),
            "question_ids": [str(q["question_id"]) for q in (q1, q2, q3)],
            "checks": checks,
            "pre_rope": pre_rope,
            "metadata_parity": metadata,
            "compact_q2": q2_compact,
            "compact_q3": q3_compact,
            "all_block_comparison": all_block,
            "source_payload_sha256_before": source_before,
            "source_payload_sha256_after": source_after,
            "source_payload_unchanged": source_unchanged,
            "metadata_activation_ms": context.metadata_activation_ms,
            "initial_context_activation_ms":
                context.initial_context_activation_ms,
            "activation_total_ms": context.activation_total_ms,
            "initial_context_gpu_bytes": context.initial_context_gpu_bytes,
            "metadata_gpu_bytes_total": context.metadata_gpu_bytes_total,
            "representative_metadata_gpu_bytes":
                context.representative_metadata_gpu_bytes,
            "turn1": {
                "answer": pixel["answer"],
                "first_token_id": pixel["first_token_id"],
                "generated_tokens": pixel["generated_tokens"],
                "vision_forward_count": pixel["vision_forward_count"],
                "capture": capture.stats(),
                "persistence": {
                    "timing_ms": persisted["timing_ms"],
                    "bytes": persisted["bytes"],
                    "hashes": persisted["hashes"],
                },
            },
            "q2": _small_result(q2_result),
            "q3": _small_result(q3_result),
            "all_blocks": _small_result(all_result),
            "fullload_reference": {
                "answer": reference["answer"],
                "first_token_id": reference["first_token_id"],
                "ttft_seconds": reference["ttft"],
            },
        }
    finally:
        context.close()
        image.close()


def _write_derived_artifacts(run_dir: Path, images: list[dict],
                             unit: dict) -> None:
    _atomic_json(run_dir / "parity_tests.json", {
        "schema_version": SCHEMA_VERSION,
        "unit_gates": unit,
        "images": [{
            "image_id": row["image_id"],
            "metadata_parity": row["metadata_parity"],
            "pre_rope_passed": row["pre_rope"]["passed"],
        } for row in images],
        "passed": unit["passed"] and all(
            row["metadata_parity"]["representative_parity"]
            and row["metadata_parity"]["similarity_parity"]
            for row in images),
    })
    _atomic_json(run_dir / "cache_handoff_validation.json", {
        "schema_version": SCHEMA_VERSION,
        "images": [{
            "image_id": row["image_id"],
            "checks": {key: row["checks"][key] for key in (
                "compact_cache", "retrieval_answer_handoff",
                "request_isolation", "duplicate_payload_read")},
            "source_payload_sha256_before":
                row["source_payload_sha256_before"],
            "source_payload_sha256_after":
                row["source_payload_sha256_after"],
            "q2": row["q2"], "q3": row["q3"],
        } for row in images],
        "passed": all(row["checks"]["compact_cache"]
                      and row["checks"]["retrieval_answer_handoff"]
                      and row["checks"]["request_isolation"]
                      and row["checks"]["duplicate_payload_read"]
                      for row in images),
    })
    _atomic_json(run_dir / "position_validation.json", {
        "schema_version": SCHEMA_VERSION,
        "logit_tolerance": {"rtol": LOGIT_RTOL, "atol": LOGIT_ATOL},
        "images": [{
            "image_id": row["image_id"],
            "pre_rope": row["pre_rope"],
            "all_block_comparison": row["all_block_comparison"],
            "q2_compact_lengths": row["compact_q2"]["compact_prefix_lengths"],
        } for row in images],
        "passed": all(row["checks"]["pre_rope_capture"]
                      and row["checks"]["rope_mask_position"]
                      and row["checks"]["all_block_diagnostic"]
                      for row in images),
    })


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--index", type=Path, default=ROOT / "data/index.json")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    root = ROOT / "runs/rekv_baseline"
    root.mkdir(parents=True, exist_ok=True)
    run_dir = (args.run_dir if args.run_dir is not None
               else root / f"smoke_{timestamp}")
    run_dir = run_dir.resolve()
    if not run_dir.is_relative_to(root.resolve()):
        raise ValueError("smoke artifacts must be under runs/rekv_baseline")
    run_dir.mkdir(parents=True, exist_ok=False)
    images: list[dict] = []
    unit = {"passed": False}
    error = None
    try:
        pilot = _load_runner_module()
        entries, workload = pilot._validate_workload(args.index)
        if len(entries) != 40:
            raise AssertionError("frozen GQA index is not 40 images")
        _atomic_json(run_dir / "config.json", {
            "schema_version": SCHEMA_VERSION,
            "model_id": MODEL_ID,
            "n_images": N_IMAGES,
            "seed": SEED,
            "max_new_tokens": MAX_NEW_TOKENS,
            "index": str(args.index.resolve()),
            "index_sha256": workload["index_sha256"],
            "selected_image_ids": [str(entry["image_id"])
                                   for entry in entries[:N_IMAGES]],
            "selected_question_ids": [
                [str(q["question_id"]) for q in entry["questions"][
                    pilot.QUESTION_SKIP:pilot.QUESTION_SKIP + 3]]
                for entry in entries[:N_IMAGES]],
            "all_block_logit_tolerance": {
                "rtol": LOGIT_RTOL, "atol": LOGIT_ATOL},
            "source_file_sha256": {
                name: _sha256(ROOT / name) for name in (
                    "docs/rekv_baseline_contract.md",
                    "mmimpress/rekv.py", "mmimpress/rekv_store.py",
                    "scripts/70_eval_rekv_gqa.py", "scripts/72_smoke_rekv.py",
                    "tests/test_rekv.py", "tests/test_rekv_store.py")},
        })
        unit = _run_unit_gates(run_dir)
        if not unit["passed"]:
            raise AssertionError("ReKV core/store parity tests failed")
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        runner = LlavaRunner().load()
        legacy = Server(runner, max_new_tokens=MAX_NEW_TOKENS)
        rekv = ReKVServer(
            runner, max_new_tokens=MAX_NEW_TOKENS,
            check_finite_logits=True)
        all_blocks = ReKVServer(
            runner, max_new_tokens=MAX_NEW_TOKENS, all_blocks=True,
            check_finite_logits=True)
        for index, entry in enumerate(entries[:N_IMAGES]):
            row = _one_image(index, entry, run_dir, runner,
                             legacy, rekv, all_blocks, pilot)
            _atomic_json(run_dir / f"image_{index:02d}.json", row)
            images.append(row)
            print(f"ReKV smoke image {index+1}/{N_IMAGES}: "
                  f"{row['image_id']} checks={all(row['checks'].values())}",
                  flush=True)
            torch.cuda.empty_cache()
        _write_derived_artifacts(run_dir, images, unit)
    except BaseException as failure:
        error = {
            "type": type(failure).__name__,
            "message": str(failure),
            "traceback": traceback.format_exc(),
        }
        print(error["traceback"], file=sys.stderr, flush=True)
    checks = {
        name: bool(unit.get("passed") and len(images) == N_IMAGES
                   and all(row["checks"].get(name) is True for row in images))
        for name in REQUIRED_CHECKS}
    validation = {
        "schema_version": SCHEMA_VERSION,
        "passed": error is None and all(checks.values()),
        "n_images": len(images),
        "images_expected": N_IMAGES,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items()
                          if not passed],
        "unit_gates_passed": bool(unit.get("passed")),
        "image_artifacts": [f"image_{i:02d}.json"
                            for i in range(len(images))],
        "error": error,
        "finished_at_unix": time.time(),
    }
    _atomic_json(run_dir / "validation.json", validation)
    print(f"ReKV smoke validation: {run_dir / 'validation.json'} "
          f"passed={validation['passed']}", flush=True)
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

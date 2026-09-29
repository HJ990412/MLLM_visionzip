#!/usr/bin/env python3
"""Fail-closed Qwen2.5-VL correctness validation on one frozen GQA image.

Writes a new run directory containing validation.json and two new BF16 stores.
The 25% comparison uses the same selected tokens in dense masked and compact
caches. Floating tolerances are fixed in docs/qwen25_port_contract.md.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mmimpress.qwen25.runner import (  # noqa: E402
    CHECKPOINT_REVISION, MAX_NEW_TOKENS, Qwen25Runner, SEED,
)
from mmimpress.qwen25.store import (  # noqa: E402
    inverse_permutation, stable_visual_order,
)
from mmimpress.qwen25.vision import (  # noqa: E402
    VISIONZIP_COMMIT, VISIONZIP_FILE_SHA256,
    merge_window_scores, received_attention_scores,
)

SCHEMA = "qwen25-gpu-correctness-v1"
LOGIT_ATOL = 0.125
LOGIT_RTOL = 0.02
SCORE_CPU_ATOL = 1e-5
SCORE_CPU_RTOL = 1e-5
SCORE_GPU_ATOL = 1e-3
SCORE_GPU_RTOL = 1e-3
GATES = (
    "cpu_score_reference", "runtime_load", "source_capture", "geometry",
    "gpu_score_reference", "query_independence", "persistence", "roundtrip",
    "fullload", "repacked_full100", "prefix25", "capture", "io",
    "request_isolation", "history",
)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_scores(scores: torch.Tensor) -> str:
    return hashlib.sha256(scores.detach().cpu().float().numpy().tobytes()).hexdigest()


def _record_result(result: dict) -> dict:
    return {k: v for k, v in result.items() if k not in ("capture", "first_logits")}


def _compare_logits_and_output(left: dict, right: dict) -> dict:
    a, b = left["first_logits"], right["first_logits"]
    if a is None or b is None or a.shape != b.shape:
        return {"passed": False, "reason": "first logits absent or shape mismatch",
                "left_shape": list(a.shape) if a is not None else None,
                "right_shape": list(b.shape) if b is not None else None}
    delta = (a.float() - b.float()).abs()
    relative = delta / torch.maximum(a.float().abs(), b.float().abs()).clamp_min(1e-12)
    numerical = bool(torch.allclose(a, b, atol=LOGIT_ATOL, rtol=LOGIT_RTOL))
    first_identity = left["first_token_id"] == right["first_token_id"]
    token_identity = left["generated_token_ids"] == right["generated_token_ids"]
    prediction_identity = left["prediction"] == right["prediction"]
    return {
        "passed": numerical and first_identity and token_identity and prediction_identity,
        "numerically_close": numerical, "first_token_identical": first_identity,
        "generated_tokens_identical": token_identity,
        "prediction_identical": prediction_identity,
        "max_abs_logit_error": float(delta.max()),
        "max_relative_logit_error": float(relative.max()),
        "left_first_token": left["first_token_id"],
        "right_first_token": right["first_token_id"],
        "left_prediction": left["prediction"],
        "right_prediction": right["prediction"],
    }


def _bitwise_layers(left: list, right: list) -> dict:
    mismatches = []
    if len(left) != len(right):
        return {"equal": False, "reason": "layer count mismatch",
                "left_layers": len(left), "right_layers": len(right)}
    for layer_no, (left_pair, right_pair) in enumerate(zip(left, right)):
        for kind, a, b in zip(("k", "v"), left_pair, right_pair):
            if a.shape != b.shape or a.dtype != b.dtype:
                mismatches.append({"layer": layer_no, "kind": kind,
                                   "reason": "shape or dtype mismatch",
                                   "left_shape": list(a.shape), "right_shape": list(b.shape),
                                   "left_dtype": str(a.dtype), "right_dtype": str(b.dtype)})
                continue
            aa = a.detach().to("cpu").contiguous().view(torch.int16)
            bb = b.detach().to("cpu").contiguous().view(torch.int16)
            if not torch.equal(aa, bb):
                mismatches.append({"layer": layer_no, "kind": kind,
                                   "different_bf16_words": int((aa != bb).sum())})
    return {"equal": not mismatches, "mismatches": mismatches[:20],
            "mismatch_entry_count": len(mismatches)}


def _cpu_score_fixture() -> dict:
    torch.manual_seed(SEED)
    q = torch.randn(11, 3, 8, dtype=torch.float32)
    k = torch.randn(11, 3, 8, dtype=torch.float32)
    cu = torch.tensor([0, 4, 7, 11])
    logits = torch.matmul(q.transpose(0, 1), k.transpose(0, 1).transpose(-1, -2)) / math.sqrt(8)
    mask = torch.full((11, 11), float("-inf"))
    for start, end in zip(cu[:-1], cu[1:]):
        mask[start:end, start:end] = 0
    reference = torch.softmax(logits + mask, dim=-1, dtype=torch.float32).mean(0).sum(0)
    got = received_attention_scores(q, k, cu, query_block_size=2)
    index = torch.tensor([2, 0, 3, 1])
    grouped = torch.arange(16, dtype=torch.float32)
    mapped = merge_window_scores(grouped, index, 2)
    mapped_reference = grouped.reshape(4, 4).mean(-1)[torch.argsort(index)]
    return {"_passed": bool(torch.allclose(got, reference, atol=SCORE_CPU_ATOL,
                                          rtol=SCORE_CPU_RTOL)
                            and torch.equal(mapped, mapped_reference)),
            "max_abs_score_error": float((got - reference).abs().max()),
            "blocked_query_rows": 2, "frame_boundaries": cu.tolist(),
            "window_mapping_equal": bool(torch.equal(mapped, mapped_reference))}


@torch.inference_mode()
def _memory_prefix_inference(runner: Qwen25Runner, capture, meta: dict,
                             question: str, history=()) -> dict:
    from transformers import DynamicCache

    runner._clear_request_state()
    _, suffix, positions, _ = runner._cache_hit_ids(meta, question, history)
    prefix_len = len(capture.prefix_ids)
    cache = DynamicCache(config=runner.model.config.text_config)
    for layer_no, (key, value) in enumerate(capture.layers):
        cache.update(key.clone(), value.clone(), layer_no)
    mask = torch.ones((1, prefix_len + suffix.shape[1]), dtype=torch.long,
                      device=runner.device)
    try:
        output = runner.model(
            input_ids=suffix.to(runner.device), attention_mask=mask,
            position_ids=positions[:, :, prefix_len:].to(runner.device),
            cache_position=torch.arange(prefix_len, prefix_len + suffix.shape[1],
                                        device=runner.device),
            past_key_values=cache, use_cache=True, return_dict=True,
            logits_to_keep=1)
        torch.cuda.synchronize(runner.device)
        logits = output.logits[0, -1].float().cpu().clone()
        first_id = int(output.logits[0, -1].argmax())
        generated, _ = runner._decode(
            first_id, cache, position_start=int(positions.max()) + 1,
            attention_mask=mask)
        return {"first_logits": logits, "first_token_id": first_id,
                "generated_token_ids": generated,
                "prediction": runner.processor.tokenizer.decode(
                    generated, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False).strip()}
    finally:
        runner._clear_request_state()


def _gpu_dense_score_reference(raw: dict, visual) -> torch.Tensor:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision

    qkv = raw["qkv"]
    n = qkv.shape[0]
    heads = visual.blocks[-1].attn.num_heads
    q, k, _ = qkv.reshape(n, 3, heads, -1).permute(1, 0, 2, 3).unbind(0)
    q, k = apply_rotary_pos_emb_vision(q, k, *raw["positions"])
    patch_score = torch.zeros(n, dtype=torch.float32, device=q.device)
    bounds = raw["cu_seqlens"].tolist()
    for start, end in zip(bounds, bounds[1:]):
        for head in range(heads):
            # Independent dense N x N reference, one head at a time so the
            # validation does not materialize all vision heads at once.
            logits = torch.matmul(q[start:end, head], k[start:end, head].T) / math.sqrt(q.shape[-1])
            probability = torch.softmax(logits, dim=-1, dtype=torch.float32).to(q.dtype)
            patch_score[start:end] += probability.float().sum(0) / heads
    merge = visual.spatial_merge_size
    index, _ = visual.get_window_index(torch.tensor(raw["grid_thw"], dtype=torch.long))
    result = patch_score.reshape(-1, merge**2).mean(-1)
    return result[torch.argsort(index.to(result.device))].float().cpu()


def _physical_hashes(capture, meta: dict) -> dict:
    order = torch.tensor(meta["stored_to_original"], dtype=torch.long)
    pad = int(meta["padding_rows"])
    start, end = capture.visual_start, capture.visual_start + capture.visual_count
    mismatches = []
    for layer_no, pair in enumerate(capture.layers):
        for kind, tensor in zip(("k", "v"), pair):
            rows = tensor[0, :, start:end, :].permute(1, 0, 2).to("cpu").contiguous()
            rows = rows.index_select(0, order)
            if pad:
                rows = torch.cat([rows, torch.zeros((pad, *rows.shape[1:]),
                                                     dtype=torch.bfloat16)], dim=0)
            rows = rows.contiguous()
            raw = ctypes.string_at(rows.data_ptr(), rows.numel() * 2)
            digest = hashlib.sha256(raw).hexdigest()
            rel = f"layer_{layer_no:03d}/{kind}.bin"
            if digest != meta["files"][rel]["sha256"]:
                mismatches.append(rel)
    return {"equal": not mismatches, "mismatched_files": mismatches,
            "checked_visual_files": 2 * len(capture.layers)}


def _io_check(result: dict, meta: dict, *, selective: bool) -> dict:
    io = result["read_io"]
    selected = int(result["selected_chunks"])
    per_file = selected * meta["chunk_size"] * meta["row_bytes"]
    expected_visual = 2 * meta["num_layers"] * per_file
    visual_spans = [span for span in io["span_details"] if span["kind"] == "visual"]
    structural_spans = [span for span in io["span_details"] if span["kind"] == "structural"]
    correct_spans = (len(visual_spans) == 2 * meta["num_layers"]
                     and len({span["source"] for span in visual_spans}) == len(visual_spans)
                     and all(span["offset"] == 0 and span["requested_bytes"] == per_file
                             for span in visual_spans)
                     and len(structural_spans) == 1
                     and structural_spans[0]["offset"] == 0
                     and structural_spans[0]["requested_bytes"] == meta["bytes_structural_kv"])
    exact_bytes = (result["visual_read_bytes"] == expected_visual
                   and result["structural_read_bytes"] == meta["bytes_structural_kv"]
                   and result["metadata_read_bytes"] == 0
                   and io["bytes"] == expected_visual + meta["bytes_structural_kv"])
    no_unselected = (not selective or selected == meta["n_chunks"]
                     or expected_visual < meta["bytes_visual_kv"])
    calls = io["preads"] >= io["spans"] == 2 * meta["num_layers"] + 1
    return {"passed": correct_spans and exact_bytes and no_unselected and calls,
            "visual_spans": len(visual_spans), "structural_spans": len(structural_spans),
            "expected_visual_bytes": expected_visual,
            "returned_visual_bytes": result["visual_read_bytes"],
            "returned_structural_bytes": result["structural_read_bytes"],
            "returned_metadata_bytes": result["metadata_read_bytes"],
            "pread_calls": io["preads"], "spans": io["spans"],
            "selected_chunks": selected, "total_chunks": meta["n_chunks"],
            "no_unselected_visual_payload": no_unselected,
            "span_details": io["span_details"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument("--attn", choices=("sdpa", "eager"), default="sdpa",
                        help="stock attention backend applied to every validation arm")
    parser.add_argument("--cpu-only", action="store_true",
                        help="write a CPU fixture report and mark GPU gates NOT RUN")
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or ROOT / "runs" / f"qwen25_port_validate_{timestamp}_{os.getpid()}"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    index_path = ROOT / "data/index.json"
    source = json.loads(index_path.read_text(encoding="utf-8"))
    if args.image_index < 0 or args.image_index >= len(source):
        parser.error("image-index outside GQA index")
    row = source[args.image_index]
    questions = row["questions"][4:7]
    if len(questions) != 3 or len({q["question_id"] for q in questions}) != 3:
        raise ValueError("chosen GQA image lacks three distinct frozen questions")
    image_path = (ROOT / row["image_path"]).resolve()
    image_hash = _sha_file(image_path)
    adapter_paths = [ROOT / "mmimpress/qwen25/runner.py",
                     ROOT / "mmimpress/qwen25/vision.py",
                     ROOT / "mmimpress/qwen25/store.py"]
    source_hashes = {str(path.relative_to(ROOT)): _sha_file(path)
                     for path in [*adapter_paths, Path(__file__).resolve(),
                                  ROOT / "docs/qwen25_port_contract.md"]}
    combined_adapter_revision = hashlib.sha256(b"".join(
        hashlib.sha256(path.read_bytes()).digest() for path in adapter_paths
    )).hexdigest()
    report = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "NOT RUN", "pilot_eligible": False,
        "run_dir": str(out_dir), "model_revision": CHECKPOINT_REVISION,
        "visionzip_reference": {"commit": VISIONZIP_COMMIT,
                                "source_sha256": VISIONZIP_FILE_SHA256},
        "source_hashes": source_hashes,
        "combined_adapter_revision": combined_adapter_revision,
        "source": {"gqa_index": str(index_path),
                   "gqa_index_sha256": _sha_file(index_path),
                   "image_id": str(row["image_id"]), "image_path": str(image_path),
                   "image_sha256": image_hash,
                   "questions": [{"question_id": str(q["question_id"]),
                                  "question": q["question"], "gold": q["answer"]}
                                 for q in questions]},
        "configuration": {"seed": SEED, "max_new_tokens": MAX_NEW_TOKENS,
                          "attention_backend": args.attn, "chunk_size": 64,
                          "budget_ratio": 0.25},
        "tolerances": {"float32_score_atol": SCORE_CPU_ATOL,
                       "float32_score_rtol": SCORE_CPU_RTOL,
                       "gpu_bf16_score_atol": SCORE_GPU_ATOL,
                       "gpu_bf16_score_rtol": SCORE_GPU_RTOL,
                       "gpu_logit_atol": LOGIT_ATOL,
                       "gpu_logit_rtol": LOGIT_RTOL},
        "gates": {name: {"status": "NOT RUN", "reason": "not reached"} for name in GATES},
    }
    report_path = out_dir / "validation.json"

    def save():
        statuses = [entry["status"] for entry in report["gates"].values()]
        report["status"] = ("FAIL" if "FAIL" in statuses else
                            "PASS" if all(s == "PASS" for s in statuses) else "NOT RUN")
        report["pilot_eligible"] = all(s == "PASS" for s in statuses)
        tmp = out_dir / "validation.json.tmp"
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=False,
                      allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, report_path)

    def gate(name: str, fn, deps=()):
        missing = [d for d in deps if report["gates"][d]["status"] != "PASS"]
        if missing:
            report["gates"][name] = {"status": "NOT RUN",
                                     "reason": f"prerequisite gates did not pass: {missing}"}
            save()
            return None
        started = time.perf_counter()
        try:
            details = fn()
            if details is None:
                details = {}
            passed = bool(details.pop("_passed", True))
            report["gates"][name] = {"status": "PASS" if passed else "FAIL",
                                     "duration_seconds": time.perf_counter() - started,
                                     "details": details}
            result = details
        except Exception as exc:
            report["gates"][name] = {"status": "FAIL",
                                     "duration_seconds": time.perf_counter() - started,
                                     "error": f"{type(exc).__name__}: {exc}",
                                     "traceback": traceback.format_exc(limit=12)}
            result = None
        save()
        return result

    save()
    gate("cpu_score_reference", _cpu_score_fixture)
    if args.cpu_only:
        for name in GATES[1:]:
            report["gates"][name] = {"status": "NOT RUN", "reason": "--cpu-only"}
        save()
        print(report_path)
        return 0

    runner = Qwen25Runner(attn=args.attn)
    state: dict[str, Any] = {}
    try:
        def load_runtime():
            runner.load()
            fingerprint = runner.runtime_fingerprint()
            passed = (fingerprint["checkpoint_revision"] == CHECKPOINT_REVISION
                      and fingerprint["kv_heads"] == 4
                      and fingerprint["head_dim"] == 128
                      and fingerprint["decoder_layers"] == 28
                      and fingerprint["vision_fullatt_block_indexes"][-1] == 31)
            from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
            return {"_passed": passed, "runtime": fingerprint,
                    "installed_modeling_path": modeling_qwen2_5_vl.__file__,
                    "installed_modeling_sha256": _sha_file(Path(modeling_qwen2_5_vl.__file__)),
                    "cuda_name": torch.cuda.get_device_name(runner.device),
                    "cuda_total_memory_bytes": torch.cuda.get_device_properties(runner.device).total_memory}
        gate("runtime_load", load_runtime, ("cpu_score_reference",))

        def source_capture():
            with Image.open(image_path) as src:
                image = src.convert("RGB")
            state["image"] = image
            results = []
            raw: dict[str, Any] = {"grid_thw": None}
            attn = runner.model.visual.blocks[-1].attn
            def model_pre(module, inputs, kwargs):
                ids = kwargs.get("input_ids")
                raw.setdefault("forward_calls", []).append({
                    "input_length": int(ids.shape[1]) if ids is not None else None,
                    "has_pixels": kwargs.get("pixel_values") is not None,
                })

            def attn_pre(module, inputs, kwargs):
                raw["positions"] = kwargs["position_embeddings"]
                raw["cu_seqlens"] = kwargs["cu_seqlens"].detach().cpu().long()

            def qkv_post(module, inputs, output):
                if "qkv" in raw:
                    raise RuntimeError("last vision QKV ran twice in the source request")
                raw["qkv"] = output.detach().clone()

            handles = [
                runner.model.register_forward_pre_hook(model_pre, with_kwargs=True),
                attn.register_forward_pre_hook(attn_pre, with_kwargs=True),
                attn.qkv.register_forward_hook(qkv_post),
            ]
            try:
                first = runner.run_pixels(image, questions[0]["question"],
                                          capture=True, image_sha256=image_hash,
                                          return_logits=True)
            finally:
                for handle in handles:
                    handle.remove()
            results.append(first)
            raw["grid_thw"] = first["capture"].image_grid_thw
            for question in questions[1:]:
                results.append(runner.run_pixels(
                    image, question["question"], capture=True,
                    image_sha256=image_hash, return_logits=True))
            state["source_results"] = results
            state["captures"] = [r["capture"] for r in results]
            state["raw_vision"] = raw
            correct_calls = all(r["vision_calls"] == 1 for r in results)
            return {"_passed": correct_calls,
                    "responses": [{"question_id": str(q["question_id"]),
                                   "prediction": r["prediction"],
                                   "vision_calls": r["vision_calls"],
                                   "score_sha256": _sha_scores(r["capture"].scores),
                                   "score_extra_ms": r["score_extra_ms"],
                                   "score_peak_extra_gpu_bytes": r["score_peak_extra_gpu_bytes"],
                                   "score_peak_gpu_allocated_bytes": r["score_peak_gpu_allocated_bytes"],
                                   "capture_clone_ms": r["capture_clone_ms"]}
                                  for q, r in zip(questions, results)],
                    "first_request_model_forward_calls": raw["forward_calls"]}
        gate("source_capture", source_capture, ("runtime_load",))

        def geometry():
            captures = state["captures"]
            vc = runner.model.config.vision_config
            tc = runner.model.config.text_config
            evidence = []
            passed = True
            for capture, result in zip(captures, state["source_results"]):
                g = result["geometry"]
                expected = int(math.prod(capture.image_grid_thw[0]))
                merged = expected // (vc.spatial_merge_size**2)
                native = [1, tc.num_key_value_heads, len(capture.prefix_ids),
                          tc.hidden_size // tc.num_attention_heads]
                checks = {
                    "patches": g["pre_merger_patches"] == expected,
                    "merger": g["merger_tokens"] == merged,
                    "expanded_image": g["expanded_image_tokens"] == merged,
                    "visual_kv_rows": capture.visual_count == merged,
                    "scores": capture.scores.numel() == merged,
                    "prefix_boundary": len(capture.prefix_ids) == g["prefix_len"],
                    "native_cache": all(list(k.shape) == native and list(v.shape) == native
                                        and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
                                        for k, v in capture.layers),
                    "decoder_layers": len(capture.layers) == tc.num_hidden_layers,
                }
                passed &= all(checks.values())
                evidence.append({"checks": checks, "geometry": g,
                                 "native_kv_shape": native,
                                 "score_count": int(capture.scores.numel())})
            return {"_passed": bool(passed), "per_question": evidence}
        gate("geometry", geometry, ("source_capture",))

        def gpu_score_reference():
            capture = state["captures"][0]
            raw = state.pop("raw_vision")
            dense = _gpu_dense_score_reference(raw, runner.model.visual)
            actual = capture.scores
            delta = (dense - actual).abs()
            numerically_close = bool(torch.allclose(dense, actual,
                                                     atol=SCORE_GPU_ATOL,
                                                     rtol=SCORE_GPU_RTOL))
            order_match = stable_visual_order(dense) == stable_visual_order(actual)
            del raw
            return {"_passed": numerically_close and order_match,
                    "numerically_close": numerically_close,
                    "stable_permutation_identical": order_match,
                    "max_abs_score_error": float(delta.max()),
                    "mean_abs_score_error": float(delta.mean()),
                    "dense_score_count": int(dense.numel()),
                    "blocked_score_count": int(actual.numel())}
        gate("gpu_score_reference", gpu_score_reference, ("geometry",))

        def query_independence():
            captures = state["captures"]
            first = captures[0]
            score_equal = [torch.equal(first.scores, c.scores) for c in captures[1:]]
            orders_equal = [stable_visual_order(first.scores) == stable_visual_order(c.scores)
                            for c in captures[1:]]
            kv_checks = [_bitwise_layers(first.layers, c.layers) for c in captures[1:]]
            prefix_ids_equal = [first.prefix_ids == c.prefix_ids for c in captures[1:]]
            position_equal = [first.logical_position_ids == c.logical_position_ids
                              for c in captures[1:]]
            passed = (all(score_equal) and all(orders_equal) and
                      all(item["equal"] for item in kv_checks) and
                      all(prefix_ids_equal) and all(position_equal))
            return {"_passed": passed, "score_bitwise_equal": score_equal,
                    "stable_permutation_identical": orders_equal,
                    "prefix_kv_bitwise_equal": kv_checks,
                    "prefix_ids_equal": prefix_ids_equal,
                    "prefix_logical_positions_equal": position_equal,
                    "causal_isolation": "captured prefix KV is bitwise identical across three suffix questions"}
        gate("query_independence", query_independence, ("geometry",))

        def persist():
            capture = state["captures"][0]
            canonical_dir = out_dir / "canonical_store"
            repacked_dir = out_dir / "repacked_store"
            canonical = runner.persist(capture, canonical_dir, layout="canonical")
            repacked = runner.persist(capture, repacked_dir, layout="repacked")
            state["canonical_dir"] = canonical_dir
            state["repacked_dir"] = repacked_dir
            state["canonical_meta"] = canonical["metadata"]
            state["repacked_meta"] = repacked["metadata"]
            revisions_match = (canonical["metadata"]["code_revision"] ==
                               repacked["metadata"]["code_revision"] ==
                               combined_adapter_revision)
            return {"_passed": revisions_match,
                    "source_revision_matches_store": revisions_match,
                    "combined_adapter_revision": combined_adapter_revision,
                    "store_code_revision": canonical["metadata"]["code_revision"],
                    "store_environment_revision": canonical["metadata"]["environment_revision"],
                    "canonical": {"store_dir": str(canonical_dir),
                                  "persistence_ms": canonical["persistence_ms"],
                                  "timing_ms": canonical["timing_ms"]},
                    "repacked": {"store_dir": str(repacked_dir),
                                 "persistence_ms": repacked["persistence_ms"],
                                 "timing_ms": repacked["timing_ms"]}}
        gate("persistence", persist, ("geometry", "gpu_score_reference"))

        def roundtrip():
            capture = state["captures"][0]
            canonical = runner._activate(state["canonical_dir"], image_hash)
            repacked = runner._activate(state["repacked_dir"], image_hash)
            c_full = canonical.load_prefix(budget=1.0)
            r_full = repacked.load_prefix(budget=1.0)
            expected_positions = tuple(range(len(capture.prefix_ids)))
            canonical_bits = _bitwise_layers(capture.layers, c_full.layers)
            repacked_bits = _bitwise_layers(capture.layers, r_full.layers)
            expected_order = stable_visual_order(capture.scores)
            order_ok = repacked.meta["stored_to_original"] == expected_order
            inverse_ok = repacked.meta["original_to_stored"] == inverse_permutation(expected_order)
            canonical_physical = _physical_hashes(capture, canonical.meta)
            repacked_physical = _physical_hashes(capture, repacked.meta)
            passed = (canonical_bits["equal"] and repacked_bits["equal"]
                      and c_full.logical_indices == expected_positions
                      and r_full.logical_indices == expected_positions
                      and order_ok and inverse_ok
                      and canonical_physical["equal"] and repacked_physical["equal"])
            return {"_passed": passed,
                    "canonical_bf16_bitwise": canonical_bits,
                    "repacked_inverse_bf16_bitwise": repacked_bits,
                    "canonical_physical_file_hashes": canonical_physical,
                    "repacked_physical_file_hashes": repacked_physical,
                    "canonical_all_prefix_positions": c_full.logical_indices == expected_positions,
                    "repacked_all_prefix_positions": r_full.logical_indices == expected_positions,
                    "stable_order_identical": order_ok, "inverse_mapping_identical": inverse_ok,
                    "canonical_activation": runner.activation_records[str(state["canonical_dir"].resolve())],
                    "repacked_activation": runner.activation_records[str(state["repacked_dir"].resolve())],
                    "native_kv_heads": canonical.meta["num_kv_heads"],
                    "dtype": canonical.meta["dtype"]}
        gate("roundtrip", roundtrip, ("persistence",))

        def fullload():
            results = []
            full = []
            for question, recompute in zip(questions, state["source_results"]):
                memory = _memory_prefix_inference(runner, state["captures"][0],
                                                  state["canonical_meta"], question["question"])
                ssd = runner.run_cache(state["canonical_dir"], question["question"],
                                       budget_ratio=1.0, image_sha256=image_hash,
                                       return_logits=True)
                full.append(ssd)
                results.append({"question_id": str(question["question_id"]),
                                "ssd_vs_memory": _compare_logits_and_output(ssd, memory),
                                "ssd_vs_recompute": _compare_logits_and_output(ssd, recompute),
                                "ssd_request": _record_result(ssd)})
            state["fullload_results"] = full
            passed = all(r["ssd_vs_memory"]["passed"] and
                         r["ssd_vs_recompute"]["passed"] and
                         r["ssd_request"]["vision_calls"] == 0 for r in results)
            return {"_passed": passed, "per_question": results}
        gate("fullload", fullload, ("roundtrip",))

        def repacked_full100():
            results = []
            for question, canonical in zip(questions, state["fullload_results"]):
                repacked = runner.run_cache(state["repacked_dir"], question["question"],
                                            budget_ratio=1.0, image_sha256=image_hash,
                                            return_logits=True)
                results.append({"question_id": str(question["question_id"]),
                                "vs_canonical": _compare_logits_and_output(repacked, canonical),
                                "vision_calls": repacked["vision_calls"]})
            return {"_passed": all(r["vs_canonical"]["passed"] and r["vision_calls"] == 0
                                   for r in results), "per_question": results}
        gate("repacked_full100", repacked_full100, ("roundtrip",))

        def prefix25():
            results = []
            compact_results = []
            selected = runner._activate(state["repacked_dir"], image_hash).load_prefix(budget=0.25)
            meta = state["repacked_meta"]
            selected_set = list(selected.selected_visual_original)
            for question in questions:
                dense = runner.run_cache(state["repacked_dir"], question["question"],
                                         budget_ratio=0.25, image_sha256=image_hash,
                                         dense_reference=True, return_logits=True)
                compact = runner.run_cache(state["repacked_dir"], question["question"],
                                           budget_ratio=0.25, image_sha256=image_hash,
                                           dense_reference=False, return_logits=True)
                compact_results.append(compact)
                expected_compact = meta["structural_count"] + len(selected_set)
                positions_equal = (dense["logical_suffix_first_position"] ==
                                   compact["logical_suffix_first_position"]
                                   and dense["rope_deltas"] == compact["rope_deltas"])
                slot_check = (dense["compact_suffix_first_slot"] == meta["prefix_len"]
                              and compact["compact_suffix_first_slot"] == expected_compact)
                selection_check = (dense["selected_chunks"] == compact["selected_chunks"]
                                   == selected.selected_chunks
                                   and dense["kept_tokens"] == compact["kept_tokens"]
                                   == len(selected_set))
                results.append({"question_id": str(question["question_id"]),
                                "dense_vs_compact": _compare_logits_and_output(dense, compact),
                                "same_full_logical_suffix_positions": positions_equal,
                                "dense_and_compact_cache_slots_correct": slot_check,
                                "same_selected_visual_set": selection_check,
                                "selected_visual_original": selected_set,
                                "dense_suffix_slot": dense["compact_suffix_first_slot"],
                                "compact_suffix_slot": compact["compact_suffix_first_slot"],
                                "logical_suffix_position": compact["logical_suffix_first_position"],
                                "compact_request": _record_result(compact)})
            state["compact_results"] = compact_results
            passed = all(r["dense_vs_compact"]["passed"] and
                         r["same_full_logical_suffix_positions"] and
                         r["dense_and_compact_cache_slots_correct"] and
                         r["same_selected_visual_set"] and
                         r["compact_request"]["vision_calls"] == 0 for r in results)
            return {"_passed": passed, "per_question": results}
        gate("prefix25", prefix25, ("roundtrip",))

        def capture_gate():
            calls = state["raw_vision"]["forward_calls"] if "raw_vision" in state else \
                report["gates"]["source_capture"]["details"]["first_request_model_forward_calls"]
            source_prefill = [x for x in calls if x["has_pixels"]]
            extra_prefix = [x for x in calls if not x["has_pixels"] and x["input_length"] != 1]
            source_vision = [r["vision_calls"] for r in state["source_results"]]
            hits = [r["vision_calls"] for r in state.get("fullload_results", [])
                    + state.get("compact_results", [])]
            passed = (len(source_prefill) == 1 and not extra_prefix
                      and source_vision == [1, 1, 1]
                      and bool(hits) and all(x == 0 for x in hits))
            return {"_passed": passed, "source_prefill_with_pixels": len(source_prefill),
                    "additional_prefix_forwards": len(extra_prefix),
                    "source_vision_calls": source_vision, "cache_hit_vision_calls": hits,
                    "model_forward_calls": calls}
        gate("capture", capture_gate, ("source_capture", "roundtrip"))

        def io_gate():
            if not state.get("compact_results") or not state.get("fullload_results"):
                raise RuntimeError("cache-hit results unavailable for I/O verification")
            ours = _io_check(state["compact_results"][0], state["repacked_meta"], selective=True)
            full = _io_check(state["fullload_results"][0], state["canonical_meta"], selective=False)
            return {"_passed": ours["passed"] and full["passed"],
                    "ours25": ours, "fullload": full,
                    "activation_reads_separate_from_hit": True}
        gate("io", io_gate, ("roundtrip",))

        def request_isolation():
            wrong_hash = "0" * 64 if image_hash != "0" * 64 else "1" * 64
            rejected = False
            rejection = None
            try:
                runner.run_cache(state["repacked_dir"], questions[1]["question"],
                                 budget_ratio=0.25, image_sha256=wrong_hash)
            except ValueError as exc:
                rejected = "image_sha256" in str(exc)
                rejection = str(exc)
            image = state["image"].copy()
            px = image.getpixel((0, 0))
            image.putpixel((0, 0), ((px[0] + 127) % 256, px[1], px[2]))
            mutated_hash = hashlib.sha256(image.tobytes()).hexdigest()
            changed = runner.run_pixels(image, questions[0]["question"],
                                        capture=False, image_sha256=mutated_hash,
                                        return_logits=False)
            repeat = runner.run_cache(state["repacked_dir"], questions[1]["question"],
                                      budget_ratio=0.25, image_sha256=image_hash,
                                      return_logits=True)
            baseline = state["compact_results"][1]
            same = _compare_logits_and_output(repeat, baseline)
            clean_rope = runner.model.model.rope_deltas is None
            passed = rejected and changed["vision_calls"] == 1 and repeat["vision_calls"] == 0 \
                and same["passed"] and clean_rope
            return {"_passed": passed, "wrong_image_identity_rejected": rejected,
                    "rejection": rejection, "mutated_image_sha256": mutated_hash,
                    "mutated_image_vision_calls": changed["vision_calls"],
                    "repeat_original_vision_calls": repeat["vision_calls"],
                    "repeat_vs_pre_switch": same,
                    "rope_deltas_cleared": clean_rope}
        gate("request_isolation", request_isolation, ("roundtrip",))

        def history():
            q1, q2, q3 = [q["question"] for q in questions]
            first_answer = state["source_results"][0]["prediction"]
            evidence = []
            for method in ("recompute", "fullload", "ours25"):
                hist2 = [(q1, first_answer)]
                if method == "recompute":
                    t2 = runner.run_pixels(state["image"], q2, history=hist2,
                                           image_sha256=image_hash)
                else:
                    store = state["canonical_dir"] if method == "fullload" else state["repacked_dir"]
                    budget = 1.0 if method == "fullload" else 0.25
                    t2 = runner.run_cache(store, q2, history=hist2,
                                          budget_ratio=budget, image_sha256=image_hash)
                hist3 = hist2 + [(q2, t2["prediction"])]
                messages = runner._messages(q3, hist3)
                assistant_answers = [m["content"] for m in messages if m["role"] == "assistant"]
                if method == "recompute":
                    t3 = runner.run_pixels(state["image"], q3, history=hist3,
                                           image_sha256=image_hash)
                else:
                    t3 = runner.run_cache(store, q3, history=hist3,
                                          budget_ratio=budget, image_sha256=image_hash)
                owned = assistant_answers == [first_answer, t2["prediction"]]
                vision_ok = (t2["vision_calls"] == t3["vision_calls"]
                             == (1 if method == "recompute" else 0))
                evidence.append({"method": method, "t1_generated_answer": first_answer,
                                 "t2_generated_answer": t2["prediction"],
                                 "t3_generated_answer": t3["prediction"],
                                 "t3_assistant_history": assistant_answers,
                                 "uses_own_generated_answers": owned,
                                 "expected_vision_calls": vision_ok,
                                 "t2_generated_token_count": t2["generated_token_count"],
                                 "t3_generated_token_count": t3["generated_token_count"]})
            return {"_passed": all(e["uses_own_generated_answers"] and
                                   e["expected_vision_calls"] for e in evidence),
                    "per_method": evidence,
                    "gold_answers_used_as_history": False}
        gate("history", history, ("roundtrip",))
    finally:
        runner.close()
        save()

    print(report_path)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

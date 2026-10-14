#!/usr/bin/env python3
"""Real-model correctness and 3--5 image smoke for MPIC SSD adaptation.

The script is intentionally separate from the GQA pilot.  It provisions each
image from one normal Turn-1 pixel request, checks k=0/k=N/k=32 against a
same-embedding full-prefill reference, exercises dummy and shifted-position
diagnostics, and writes fail-closed JSON artifacts.  It never changes an
existing result or store.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mmimpress.model import LlavaRunner, cache_layers  # noqa: E402
from mmimpress.config import CHUNK_SIZE, PROBE_HEADS  # noqa: E402
from mmimpress.mpic import (  # noqa: E402
    MPICContext, MPICSelectivePrefill, MPICServer,
    build_active_row_plan, build_causal_mask, expand_prompt_without_pixels,
    persist_captured_mpic_prefix, seed_dynamic_cache,
)
from mmimpress.piggyback import (  # noqa: E402
    persist_captured_raster_prefix, stable_json_sha256,
)
from mmimpress.serve import ImageContext, Server  # noqa: E402
from mmimpress.store import IOCounter  # noqa: E402


SCHEMA_VERSION = "mpic-correctness-smoke-v1"
RTOL = 2.0e-2
ATOL = 2.0e-2


def _base_module():
    path = ROOT / "scripts/49_eval_query_aware_baseline.py"
    spec = importlib.util.spec_from_file_location("_mpic_pixel_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load validated pixel runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atomic_json(path: Path, value) -> None:
    payload = json.dumps(value, indent=2, ensure_ascii=False,
                         allow_nan=False).encode("utf-8") + b"\n"
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
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
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    paths = (
        "scripts/65_validate_mpic.py",
        "scripts/49_eval_query_aware_baseline.py",
        "mmimpress/mpic.py",
        "mmimpress/config.py",
        "mmimpress/model.py",
        "mmimpress/piggyback.py",
        "mmimpress/serve.py",
        "mmimpress/store.py",
        "tests/test_mpic.py",
    )
    return {name: _sha256(ROOT / name) for name in paths}


@torch.no_grad()
def _full_reference(runner, context, prompt: str):
    """Full language-model prefill from the identical persisted embeddings."""
    model = runner.model
    device = torch.device(model.device)
    n_image = int(context.meta["v_token_num"])
    ids, visual_start = expand_prompt_without_pixels(runner, prompt, n_image)
    visual = context.reader.read_visual_inputs(n_image).to(
        device=device, dtype=model.get_input_embeddings().weight.dtype)
    ids_device = ids.to(device)
    embeddings = model.get_input_embeddings()(ids_device.unsqueeze(0))
    embeddings[:, visual_start:visual_start + n_image, :] = visual.unsqueeze(0)
    positions = torch.arange(ids.numel(), device=device).unsqueeze(0)
    output = model.model.language_model(
        inputs_embeds=embeddings,
        attention_mask=torch.ones(
            1, ids.numel(), dtype=torch.long, device=device),
        position_ids=positions, cache_position=positions[0], use_cache=True)
    logits = model.lm_head(output.last_hidden_state[:, -1:, :])
    torch.cuda.synchronize()
    return ids, visual_start, logits, cache_layers(output.past_key_values)


def _comparison(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    actual = actual.detach().float().cpu()
    reference = reference.detach().float().cpu()
    difference = (actual - reference).abs()
    return {
        "finite": bool(torch.isfinite(actual).all()),
        "shape_equal": list(actual.shape) == list(reference.shape),
        "allclose": bool(torch.allclose(
            actual, reference, rtol=RTOL, atol=ATOL)),
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "actual_first_token_id": int(actual[0, -1].argmax()),
        "reference_first_token_id": int(reference[0, -1].argmax()),
        "first_token_equal": bool(
            int(actual[0, -1].argmax()) == int(reference[0, -1].argmax())),
        "rtol": RTOL, "atol": ATOL,
    }


def _cache_comparison(actual_layers, reference_layers) -> dict:
    """Chunked real-model cache comparison without a second full-size copy."""
    if len(actual_layers) != len(reference_layers):
        return {"shape_equal": False, "finite": False, "allclose": False}
    shape_equal = True
    finite = True
    allclose = True
    maximum = 0.0
    absolute_sum = 0.0
    elements = 0
    for actual_pair, reference_pair in zip(actual_layers, reference_layers):
        for actual, reference in zip(actual_pair, reference_pair):
            if tuple(actual.shape) != tuple(reference.shape):
                shape_equal = False
                continue
            for start in range(0, int(actual.shape[-2]), 256):
                stop = min(start + 256, int(actual.shape[-2]))
                left = actual[:, :, start:stop].detach().float()
                right = reference[:, :, start:stop].detach().float()
                difference = (left - right).abs()
                finite &= bool(torch.isfinite(left).all())
                maximum = max(maximum, float(difference.max()))
                absolute_sum += float(difference.sum())
                elements += int(difference.numel())
                allclose &= bool(torch.allclose(
                    left, right, rtol=RTOL, atol=ATOL))
                del left, right, difference
    return {
        "shape_equal": shape_equal,
        "finite": finite,
        "allclose": bool(shape_equal and allclose),
        "max_abs": maximum,
        "mean_abs": absolute_sum / elements if elements else None,
        "compared_elements": elements,
        "rtol": RTOL,
        "atol": ATOL,
    }


class _LayerCallAudit:
    """Independent hook counts for the manual selective decoder traversal."""

    def __init__(self, runner):
        self.runner = runner
        self.q_proj_calls = [0] * len(runner.layers)
        self.mlp_calls = [0] * len(runner.layers)
        self.handles = []

    @staticmethod
    def _increment(values, index):
        def hook(*_args, **_kwargs):
            values[index] += 1
        return hook

    def __enter__(self):
        for index, layer in enumerate(self.runner.layers):
            self.handles.append(layer.self_attn.q_proj.register_forward_hook(
                self._increment(self.q_proj_calls, index)))
            self.handles.append(layer.mlp.register_forward_hook(
                self._increment(self.mlp_calls, index)))
        return self

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def document(self, expected_per_layer: int) -> dict:
        expected = [int(expected_per_layer)] * len(self.runner.layers)
        return {
            "q_proj_calls_per_layer": self.q_proj_calls,
            "mlp_calls_per_layer": self.mlp_calls,
            "expected_calls_per_layer": expected,
            "passed": (self.q_proj_calls == expected
                       and self.mlp_calls == expected),
        }


@torch.no_grad()
def _selective_once(runner, context, prompt, k, dummy=0.0):
    ids, start = expand_prompt_without_pixels(
        runner, prompt, int(context.meta["v_token_num"]))
    counter = IOCounter()
    base = _base_module()
    with base._NoVisionForward(runner) as vision_guard, \
            _LayerCallAudit(runner) as layer_audit:
        logits, layer_cache, plan, stats = MPICSelectivePrefill(
            runner, context, k_recompute=k, dummy_value=dummy).run(
                ids, start, counter)
    audit = layer_audit.document(1)
    audit["vision_forward_count"] = int(vision_guard.calls)
    audit["passed"] &= vision_guard.calls == 0
    cache = seed_dynamic_cache(layer_cache, config=runner.model.config)
    lengths = [int(layer.keys.shape[-2]) for layer in cache.layers]
    result = {
        "logits": logits,
        "layer_cache": layer_cache,
        "plan": plan,
        "stats": stats,
        "io": counter.summary(),
        "cache_lengths": lengths,
        "cache_length_ok": all(length == ids.numel() for length in lengths),
        "independent_call_audit": audit,
    }
    del cache
    return result


def _load_entries(index: Path, max_images: int):
    entries = json.loads(index.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or len(entries) < max_images:
        raise ValueError("invalid GQA index")
    return entries[:max_images]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=ROOT / "data/index.json")
    parser.add_argument("--max-images", type=int, choices=(1, 3, 4, 5),
                        default=3)
    parser.add_argument("--skip", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    if os.path.lexists(run_dir):
        raise FileExistsError(f"run directory must be new: {run_dir}")
    run_dir.mkdir(parents=True)
    cache_root = run_dir / "stores"
    cache_root.mkdir()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    base = _base_module()
    runner = LlavaRunner().load()
    ordinary_server = Server(runner, max_new_tokens=args.max_new_tokens)
    warmup = base._warmup(runner, ordinary_server)
    entries = _load_entries(args.index, args.max_images)
    source_hashes = _source_hashes()
    config = {
        "schema_version": SCHEMA_VERSION, "status": "running",
        "model": runner.model_id, "model_revision": (
            "c916e6cdcd760b4cecd1dd4907f84ac649f93b23"),
        "max_images": args.max_images, "skip": args.skip,
        "seed": args.seed, "max_new_tokens": args.max_new_tokens,
        "fp32_unit_tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "real_model_logit_tolerance": {"rtol": RTOL, "atol": ATOL},
        "source_sha256": source_hashes,
        "warmup": warmup, "started_at_unix": time.time(),
    }
    _atomic_json(run_dir / "config.json", config)

    rows = []
    position_rows = []
    persistence_rows = []
    all_passed = True
    for image_index, entry in enumerate(entries):
        image_id = str(entry["image_id"])
        questions = entry["questions"][args.skip:args.skip + 2]
        if len(questions) != 2:
            raise ValueError(f"image {image_id} lacks two smoke questions")
        with Image.open(ROOT / entry["image_path"]) as source:
            image = source.convert("RGB")

        turn1, diagnostic = base._run_pixels(
            runner, ordinary_server, image, questions[0]["question"], "qa")
        captured = turn1.pop("captured_past_key_values")
        hidden_capture = diagnostic["hidden_capture"]
        store_dir = cache_root / "mpic" / image_id
        raster_store_dir = cache_root / "raster" / image_id
        persisted = persist_captured_mpic_prefix(
            runner, captured, diagnostic["enc_cpu"]["input_ids"],
            diagnostic["enc_cpu"]["image_sizes"][0],
            hidden_capture.result_cpu(), store_dir,
            image_id=image_id, hidden_capture_stats=hidden_capture,
            model_id=runner.model_id,
            image_input_sha256=diagnostic["image_input_sha256"],
            extra_metadata={"dataset": "gqa", "source_turn_id": 1,
                            "source_question_id": str(
                                questions[0]["question_id"])})
        raster_persisted = persist_captured_raster_prefix(
            runner, captured, diagnostic["enc_cpu"]["input_ids"],
            diagnostic["enc_cpu"]["image_sizes"][0],
            hidden_capture.result_cpu(), raster_store_dir,
            image_id=image_id, model_id=runner.model_id,
            chunk_size=CHUNK_SIZE, probe_heads=PROBE_HEADS,
            hidden_capture_stats=hidden_capture,
            image_input_sha256=diagnostic["image_input_sha256"],
            extra_metadata={"dataset": "gqa", "source_turn_id": 1,
                            "source_question_id": str(
                                questions[0]["question_id"]),
                            "smoke_reference_role": "existing_fullload"})
        persistence_rows.append({
            "image_id": image_id, "source_question_id": str(
                questions[0]["question_id"]),
            "turn1_vision_forward_count": turn1["vision_forward_count"],
            "mpic": persisted,
            "fullload_raster_reference": raster_persisted,
        })
        del captured, diagnostic["enc_cpu"]
        torch.cuda.empty_cache()

        context = MPICContext(
            store_dir, runner.model.device, runner=runner)
        prompt = runner.prompt(questions[1]["question"])
        reference_ids, reference_start, reference_logits, reference_cache = \
            _full_reference(runner, context, prompt)
        per_k = {}
        for k in (0, int(context.meta["v_token_num"]), 32):
            selective = _selective_once(runner, context, prompt, k)
            compare = _comparison(selective["logits"], reference_logits)
            cache_compare = _cache_comparison(
                selective["layer_cache"], reference_cache)
            expected_image = min(k, int(context.meta["v_token_num"]))
            counter_ok = (
                selective["stats"]["active_rows_per_layer"]
                == [selective["plan"].n_active] * int(
                    context.meta["num_layers"])
                and selective["stats"]["recomputed_image_rows_per_layer"]
                == [expected_image] * int(context.meta["num_layers"])
                and selective["stats"]["valid_image_key_count_per_layer"]
                == [int(context.meta["v_token_num"])] * int(
                    context.meta["num_layers"]))
            k_gate = (compare["finite"] and compare["first_token_equal"]
                      and selective["cache_length_ok"] and counter_ok
                      and cache_compare["shape_equal"]
                      and cache_compare["finite"]
                      and selective["independent_call_audit"]["passed"])
            if k == int(context.meta["v_token_num"]):
                k_gate = (k_gate and compare["allclose"]
                          and cache_compare["allclose"])
            per_k[str(k)] = {
                "comparison": compare,
                "cache_comparison": cache_compare,
                "k0_difference_interpretation": (
                    "Persisted image K/V is FP16 while the full reference "
                    "recomputes BF16; k=0 numerical equality is documented "
                    "but the frozen real-model gate requires finite matching "
                    "layout and first-token semantics, not allclose."
                    if k == 0 else None),
                "cache_length_ok": selective["cache_length_ok"],
                "cache_lengths": selective["cache_lengths"],
                "counter_ok": counter_ok,
                "n_active": selective["plan"].n_active,
                "n_text": int(selective["plan"].text_positions.numel()),
                "n_image": expected_image,
                "io": selective["io"],
                "stats": selective["stats"],
                "independent_call_audit": selective[
                    "independent_call_audit"],
                "passed": k_gate,
            }
            all_passed &= k_gate
            del selective
            torch.cuda.empty_cache()

        # The contract asks for the existing FullLoad semantics, not only a
        # dense language-model reference.  Serve the same Q2 from a canonical
        # raster store built from the exact captured Turn-1 forward and compare
        # it with an MPIC k=0 request.  Logit/cache comparisons above remain
        # the stronger structural evidence; this adds the legacy server path.
        raster_context = ImageContext(
            raster_store_dir, runner.model.device, require_v_hidden=True)
        raster_context.validate_qa_select_layout()
        full_result, full_diagnostic = base._run_stored(
            runner, ordinary_server, raster_context,
            questions[1]["question"], "fullload", image_id,
            int(raster_context.meta["bytes_visual_kv"]))
        with base._NoVisionForward(runner) as k0_vision_guard:
            k0_generation = MPICServer(
                runner, k_recompute=0,
                max_new_tokens=args.max_new_tokens).request(
                    context, question=questions[1]["question"], cold=True)
        k0_fullload = {
            "same_prompt_sha256": (
                full_diagnostic["prompt_sha256"]
                == hashlib.sha256(runner.prompt(
                    questions[1]["question"]).encode("utf-8")).hexdigest()),
            "fullload_prediction": full_result["answer"],
            "mpic_k0_prediction": k0_generation["prediction"],
            "prediction_equal": (
                full_result["answer"] == k0_generation["prediction"]),
            "fullload_first_token_id": int(full_result["first_token_id"]),
            "mpic_k0_first_token_id": int(k0_generation["first_token_id"]),
            "first_token_equal": (
                int(full_result["first_token_id"])
                == int(k0_generation["first_token_id"])),
            "fullload_vision_forward_count": int(
                full_result["vision_forward_count"]),
            "mpic_vision_forward_count_observed": int(
                k0_vision_guard.calls),
            "mpic_decode_cache_append_exact": bool(
                k0_generation["decode_cache_append_exact"]),
        }
        k0_fullload["passed"] = (
            k0_fullload["same_prompt_sha256"]
            and k0_fullload["first_token_equal"]
            and k0_fullload["fullload_vision_forward_count"] == 0
            and k0_fullload["mpic_vision_forward_count_observed"] == 0
            and k0_fullload["mpic_decode_cache_append_exact"])
        per_k["0"]["existing_fullload_comparison"] = k0_fullload
        per_k["0"]["passed"] &= k0_fullload["passed"]
        all_passed &= k0_fullload["passed"]
        raster_context.close()
        del raster_context, k0_generation

        sentinel = None
        shifted = None
        if image_index == 0:
            low = _selective_once(runner, context, prompt, 32, -1234.0)
            high = _selective_once(runner, context, prompt, 32, 2345.0)
            sentinel_compare = _comparison(low["logits"], high["logits"])
            sentinel = {
                "comparison": sentinel_compare,
                "passed": sentinel_compare["allclose"]
                          and sentinel_compare["first_token_equal"],
            }
            all_passed &= sentinel["passed"]
            del low, high
            shifted_prompt = "Context shift diagnostic. " + prompt
            source_hash_before = context.source_payload_hash
            shifted_run = _selective_once(
                runner, context, shifted_prompt, 32)
            shifted_ids, shifted_start = expand_prompt_without_pixels(
                runner, shifted_prompt, int(context.meta["v_token_num"]))
            shifted_plan = build_active_row_plan(
                int(shifted_ids.numel()), shifted_start,
                int(context.meta["v_token_num"]), 32)
            shifted_mask = build_causal_mask(
                shifted_plan.active_positions, int(shifted_ids.numel()),
                torch.float32)
            allowed = shifted_mask[0, 0].eq(0)
            expected_allowed = (
                torch.arange(int(shifted_ids.numel())).unsqueeze(0)
                <= shifted_plan.active_positions.unsqueeze(1))
            source_positions = [int(value) for value in
                                context.meta["source_positions"]]
            target_positions = list(range(
                shifted_start,
                shifted_start + int(context.meta["v_token_num"])))
            mapping_delta = target_positions[0] - source_positions[0]
            source_hash_after = context.source_payload_hash
            shifted = {
                "position_handling_policy": shifted_run["stats"][
                    "position_handling_policy"],
                "same_source_target_positions": shifted_run["stats"][
                    "same_source_target_positions"],
                "finite_logits": bool(torch.isfinite(
                    shifted_run["logits"]).all()),
                "cache_length_ok": shifted_run["cache_length_ok"],
                "source_positions": source_positions,
                "target_positions": target_positions,
                "constant_position_delta": mapping_delta,
                "position_mapping_exact": (
                    len(source_positions) == len(target_positions)
                    and all(target == source + mapping_delta
                            for source, target in zip(
                                source_positions, target_positions))
                    and shifted_run["stats"]["source_position_hash"]
                    == stable_json_sha256(source_positions)
                    and shifted_run["stats"]["target_position_hash"]
                    == stable_json_sha256(target_positions)),
                "causal_mask_exact": bool(torch.equal(
                    allowed, expected_allowed)),
                "independent_call_audit": shifted_run[
                    "independent_call_audit"],
                "source_payload_hash_before": source_hash_before,
                "source_payload_hash_after": source_hash_after,
            }
            shifted["passed"] = (
                not shifted["same_source_target_positions"]
                and shifted["finite_logits"]
                and shifted["cache_length_ok"]
                and shifted["position_mapping_exact"]
                and shifted["causal_mask_exact"]
                and shifted["independent_call_audit"]["passed"]
                and shifted["source_payload_hash_before"]
                == shifted["source_payload_hash_after"])
            all_passed &= shifted["passed"]
            position_rows.append(shifted)
            del shifted_run

        with base._NoVisionForward(runner) as generation_vision_guard, \
                _LayerCallAudit(runner) as generation_layer_audit:
            generated = MPICServer(
                runner, k_recompute=32,
                max_new_tokens=args.max_new_tokens).request(
                    context, question=questions[1]["question"], cold=True)
        generation_audit = generation_layer_audit.document(
            int(generated["generated_token_count"]))
        generation_audit["vision_forward_count"] = int(
            generation_vision_guard.calls)
        generation_audit["passed"] &= generation_vision_guard.calls == 0
        generated["independent_call_audit"] = generation_audit
        # The request path intentionally does not reread sampled payload bytes:
        # doing so immediately before/after a timed request can warm the storage
        # hierarchy.  Take one real sample now, after every measured/diagnostic
        # request for this image and outside all request timing boundaries.
        payload_audit_started = time.perf_counter()
        live_payload_hash = context.source_payload_hash
        payload_audit_ms = (
            time.perf_counter() - payload_audit_started) * 1e3
        image_boundary_integrity = {
            "boundary": "after_all_smoke_requests_before_context_close",
            "validated_at_context_open_sha256": (
                context.validated_payload_hash),
            "live_after_all_requests_sha256": live_payload_hash,
            "unchanged": (
                live_payload_hash == context.validated_payload_hash),
            "audit_ms": float(payload_audit_ms),
            "timing_semantics": (
                "one out-of-band live payload sample after all requests for "
                "this image; excluded from TTFT and request E2E"),
        }
        generation_gate = (
            generated["status"] == "ok"
            and generated["vision_forward_count"] == 0
            and generated["decoder_prefill_pass_count"] == 1
            and generated["n_recomputed_image_tokens"] == min(
                32, int(context.meta["v_token_num"]))
            and generated["retained_image_context_ratio"] == 1.0
            and len(generated["generated_token_ids"]) >= 1
            and generated["decode_cache_append_exact"]
            and generation_audit["passed"]
            and image_boundary_integrity["unchanged"])
        all_passed &= generation_gate
        row = {
            "image_id": image_id,
            "source_question_id": str(questions[0]["question_id"]),
            "target_question_id": str(questions[1]["question_id"]),
            "n_image_tokens": int(context.meta["v_token_num"]),
            "reference_prompt_tokens": int(reference_ids.numel()),
            "reference_visual_start": int(reference_start),
            "reference_cache_lengths": [int(k.shape[-2])
                                        for k, _ in reference_cache],
            "k_tests": per_k, "dummy_sentinel": sentinel,
            "shifted_position": shifted,
            "generation": generated,
            "image_boundary_integrity": image_boundary_integrity,
            "generation_passed": generation_gate,
        }
        rows.append(row)
        _atomic_json(run_dir / f"image_{image_index:02d}.json", row)
        context.close()
        del reference_cache, reference_logits, context, image
        torch.cuda.empty_cache()
        print(f"[{image_index + 1}/{len(entries)}] {image_id} "
              f"passed={generation_gate}", flush=True)

    checks = {
        "all_real_logits_finite": all(
            test["comparison"]["finite"]
            for row in rows for test in row["k_tests"].values()),
        "all_first_tokens_match_reference": all(
            test["comparison"]["first_token_equal"]
            for row in rows for test in row["k_tests"].values()),
        "all_kN_logits_within_predeclared_tolerance": all(
            row["k_tests"][str(row["n_image_tokens"])]["comparison"]
            ["allclose"] for row in rows),
        "all_kN_caches_within_predeclared_tolerance": all(
            row["k_tests"][str(row["n_image_tokens"])]["cache_comparison"]
            ["allclose"] for row in rows),
        "all_k0_existing_fullload_comparisons_passed": all(
            row["k_tests"]["0"]["existing_fullload_comparison"]["passed"]
            for row in rows),
        "all_cache_lengths_exact": all(
            test["cache_length_ok"]
            for row in rows for test in row["k_tests"].values()),
        "all_layer_counters_exact": all(
            test["counter_ok"]
            for row in rows for test in row["k_tests"].values()),
        "all_generations_valid": all(row["generation_passed"]
                                     for row in rows),
        "all_independent_selective_call_audits_passed": all(
            test["independent_call_audit"]["passed"]
            for row in rows for test in row["k_tests"].values()),
        "all_decode_cache_appends_exact": all(
            row["generation"]["decode_cache_append_exact"] for row in rows),
        "all_live_payload_hashes_unchanged": all(
            row["image_boundary_integrity"]["unchanged"]
            and row["image_boundary_integrity"][
                "validated_at_context_open_sha256"]
            == row["image_boundary_integrity"][
                "live_after_all_requests_sha256"]
            for row in rows),
        "dummy_sentinel_passed": bool(rows[0]["dummy_sentinel"]["passed"]),
        "shifted_position_mapping_passed": bool(
            rows[0]["shifted_position"]["passed"]),
    }
    validation = {
        "schema_version": SCHEMA_VERSION, "checks": checks,
        "passed": bool(all(checks.values()) and all_passed),
        "images_completed": len(rows), "images_expected": len(entries),
        "failed_checks": [key for key, value in checks.items() if not value],
        "source_sha256": source_hashes,
    }
    _atomic_json(run_dir / "correctness_tests.json", {
        "schema_version": SCHEMA_VERSION, "rows": rows,
        "validation": validation})
    _atomic_json(run_dir / "position_diagnostics.json", {
        "schema_version": SCHEMA_VERSION, "rows": position_rows,
        "paper_specifies_rope_relocation": False,
        "support_level": "LIMITED"})
    _atomic_json(run_dir / "persistence.json", {
        "schema_version": SCHEMA_VERSION, "rows": persistence_rows})
    _atomic_json(run_dir / "validation.json", validation)
    config["status"] = "complete" if validation["passed"] else "failed"
    config["finished_at_unix"] = time.time()
    _atomic_json(run_dir / "config.final.json", config)
    if validation["passed"]:
        _atomic_json(run_dir / "COMPLETED", {
            "schema_version": SCHEMA_VERSION,
            "validation_sha256": _sha256(run_dir / "validation.json")})
    print(json.dumps(validation, indent=2), flush=True)
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

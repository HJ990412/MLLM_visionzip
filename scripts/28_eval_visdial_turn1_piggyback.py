"""VisDial Turn-1-piggyback evaluation with server-side end-to-end TTFT.

This runner is intentionally separate from ``14_eval_multiturn.py``.  The old
run starts its timer after tokenization/H2D and consumes a store built before
the dialogue.  Here every cache arm performs ordinary multimodal inference on
Turn 1, captures the already-produced prefix KV (and image-only saliency) from
that same forward, durably persists one shared store, and only then uses the
store on Turns 2--10.

The main timestamp starts before prompt construction/tokenization.  OS page
cache conditioning is completed before that timestamp.  Dataset JPEG decode
is benchmark-fixture setup; the real LLaVA-NeXT image processor, tensor
creation, and image H2D are inside every normal multimodal request.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmimpress.config import CHUNK_SIZE
from mmimpress.multiturn import (generative_match, load_canonical,
                                 resolve_image_path, sha256_file,
                                 visdial_prior_history_text, visdial_prompt)
from mmimpress.piggyback import (VisionForwardCapture,
                                 deterministic_method_rotation,
                                 persist_captured_visual_prefix,
                                 stable_json_sha256)
from mmimpress.serve import ImageContext, Server
from mmimpress.model import LlavaRunner


SCHEMA_VERSION = "visdial-turn1-piggyback-e2e-ttft-v1"
EXPECTED_MAIN_INDEX_SHA256 = (
    "8c3dd7e983cb39e61d26362a0353b86ac84845bd7537a6331078ab7707777383"
)
EXPECTED_MAIN_REQUEST_KEYS_SHA256 = (
    "395ba928a15eb45bca905aa91d1ec89617981f18b9b22338341b2a189b3f141a"
)
DEFAULT_INDEX = Path(
    "data/visdial_v1.0/subsets/main_seed1234/index.json"
)
METHODS = {
    "recompute": {"label": "ReComp", "budget": None},
    "prefix25": {
        "label": "ImageOnly-Repack Prefix25", "budget": 0.25,
    },
    "prefix45": {
        "label": "ImageOnly-Repack Prefix45", "budget": 0.45,
    },
    "fullload": {"label": "ImageOnly FullLoad", "budget": 1.0},
}
DEFAULT_METHODS = "recompute,prefix25,prefix45,fullload"
WARMUP_PROMPT = (
    "USER: <image>\n"
    "This is an unmeasured serving-path warm-up. Describe the image briefly. "
    "ASSISTANT:"
)


def _atomic_json(path: Path, value) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w") as handle:
        json.dump(value, handle, indent=1, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _hash_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _hash_tensor_mapping(values) -> str:
    digest = hashlib.sha256()
    for key in sorted(values):
        value = values[key]
        if not torch.is_tensor(value):
            continue
        digest.update(str(key).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_hash_tensor(value)))
    return digest.hexdigest()


def _normalise_image_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 5 and tensor.shape[0] == 1:
        return tensor[0]
    return tensor


def _image_input_hash(enc) -> str:
    digest = hashlib.sha256()
    for key in ("pixel_values", "image_sizes"):
        digest.update(key.encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_hash_tensor(
            _normalise_image_tensor(enc[key]))))
    return digest.hexdigest()


class _TimedCallableProxy:
    """Delegate an AutoProcessor component while timing its real call."""

    def __init__(self, target):
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "elapsed_ms", 0.0)
        object.__setattr__(self, "calls", 0)

    def __call__(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return self.target(*args, **kwargs)
        finally:
            self.elapsed_ms += (time.perf_counter() - started) * 1e3
            self.calls += 1

    def __getattr__(self, name):
        return getattr(self.target, name)

    def __setattr__(self, name, value):
        if name in {"target", "elapsed_ms", "calls"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.target, name, value)


def _exact_processor_call(runner, image, prompt):
    """Call the combined processor once and expose non-overlapping phases.

    LLaVA-NeXT expands ``<image>`` according to AnyRes geometry in the
    combined processor.  Independently merging tokenizer and image-processor
    outputs would silently miss that expansion.  Proxies therefore measure the
    components of the one canonical combined call without doing duplicate
    request work.
    """
    processor = runner.processor
    tokenizer = processor.tokenizer
    image_processor = processor.image_processor
    timed_tokenizer = _TimedCallableProxy(tokenizer)
    timed_image = _TimedCallableProxy(image_processor)
    processor.tokenizer = timed_tokenizer
    processor.image_processor = timed_image
    started = time.perf_counter()
    try:
        enc = processor(images=image, text=prompt, return_tensors="pt")
    finally:
        total_ms = (time.perf_counter() - started) * 1e3
        processor.tokenizer = tokenizer
        processor.image_processor = image_processor
    # Current LlavaNextProcessor calls each component once.  Fail closed if a
    # library upgrade changes the operation whose latency we are reporting.
    assert timed_tokenizer.calls == 1, timed_tokenizer.calls
    assert timed_image.calls == 1, timed_image.calls
    token_ms = float(timed_tokenizer.elapsed_ms)
    image_ms = float(timed_image.elapsed_ms)
    prepare_ms = max(0.0, total_ms - token_ms - image_ms)
    return ({key: value for key, value in enc.items()}, {
        "processor_total_ms": float(total_ms),
        "tokenization_ms": token_ms,
        "image_preprocess_ms": image_ms,
        "input_prepare_ms": prepare_ms,
    })


def _synthetic_warmup_image() -> tuple[Image.Image, str]:
    """Return a deterministic in-memory image outside every dataset subset."""
    # 640x480 is the same 4:3 AnyRes geometry as the first frozen VisDial
    # request (500x375), while the pixels are wholly synthetic.  This warms
    # the shape-specific vision and long multimodal-prefill kernels without
    # consuming an experiment image.
    height, width = 480, 640
    yy, xx = np.indices((height, width), dtype=np.uint16)
    pixels = np.stack((
        (3 * xx + yy) % 256,
        (xx + 5 * yy) % 256,
        (7 * xx + 11 * yy) % 256,
    ), axis=-1).astype(np.uint8)
    digest = hashlib.sha256(pixels.tobytes()).hexdigest()
    return Image.fromarray(pixels, mode="RGB"), digest


def _run_unmeasured_warmup(runner, server) -> dict:
    """Warm the exact processor/vision/generation/capture path once.

    The fixture is generated in memory and is not a VisDial image.  The
    captured cache and saliency are discarded, and no SSD store is created.
    Hook/event setup and the complete call are outside every reported request
    timer.  Keeping allocator reservations warm is intentional serving-system
    conditioning; no request data or result is reused.
    """
    image, fixture_sha256 = _synthetic_warmup_image()
    capture = VisionForwardCapture(runner, capture_saliency=True)
    with capture:
        torch.cuda.synchronize()
        started = time.perf_counter()
        enc_cpu, processor_timing = _exact_processor_call(
            runner, image, WARMUP_PROMPT)
        enc_device = runner.to_device(enc_cpu)
        torch.cuda.synchronize()
        result = server.recompute(
            enc_device, return_past_key_values=True)
        request_returned = time.perf_counter()
    finished = time.perf_counter()

    captured_cache = result.pop("captured_past_key_values", None)
    assert captured_cache is not None, \
        "warm-up did not exercise the Turn-1 cache-return path"
    stats = capture.stats()
    assert capture.call_count == 1 and capture.saliency_call_count == 1
    input_sha256 = _hash_tensor_mapping(enc_cpu)
    first_token_id = int(result["first_token_id"])
    generated_tokens = int(result["generated_tokens"])

    # Drop every request-specific object, but deliberately retain CUDA
    # allocator/kernel warm state for the measured serving workload.
    del captured_cache, result, enc_device, enc_cpu, capture, image
    torch.cuda.synchronize()
    return {
        "enabled": True,
        "count": 1,
        "excluded_from_all_latency_metrics": True,
        "fixture": "deterministic_in_memory_rgb_pattern_640x480",
        "fixture_sha256": fixture_sha256,
        "fixture_is_dataset_image": False,
        "consumed_experiment_request": False,
        "prompt": WARMUP_PROMPT,
        "prompt_sha256": hashlib.sha256(
            WARMUP_PROMPT.encode("utf-8")).hexdigest(),
        "input_tensors_sha256": input_sha256,
        "actual_combined_processor": True,
        "normal_multimodal_generation": True,
        "return_past_key_values_path": True,
        "captured_cache_discarded": True,
        "saliency_capture_enabled": True,
        "vision_forward_count": int(stats["vision_call_count"]),
        "saliency_call_count": int(stats["saliency_call_count"]),
        "separate_vision_forward_count": 0,
        "ssd_store_written": False,
        "vision_ms": float(stats["vision_ms"]),
        "saliency_extra_ms": float(stats["saliency_reduction_ms"]),
        "processor_total_ms": float(processor_timing["processor_total_ms"]),
        "first_token_id": first_token_id,
        "generated_tokens": generated_tokens,
        "request_until_return_ms": float(
            (request_returned - started) * 1e3),
        "total_including_capture_cleanup_ms": float(
            (finished - started) * 1e3),
    }


def _suffix_from_tokenized(runner, tokenized):
    ids = tokenized["input_ids"][0]
    positions = (ids == runner.image_token_id).nonzero(as_tuple=True)[0]
    assert positions.numel() == 1, (
        f"stored single-image request needs one <image>, got "
        f"{positions.numel()}"
    )
    return ids[int(positions[0]) + 1:]


class _NoVisionForward:
    """Fail if a supposed SSD-prefix request invokes the vision tower."""

    def __init__(self, runner):
        self.tower = runner.model.model.vision_tower
        self.calls = 0
        self.handle = None

    def __enter__(self):
        def count(*_args, **_kwargs):
            self.calls += 1
        self.handle = self.tower.register_forward_pre_hook(count)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.handle.remove()
        if exc_type is None:
            assert self.calls == 0, (
                f"SSD-prefix request unexpectedly ran vision {self.calls} times"
            )
        return False


def _machine_metadata() -> dict:
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(Path.cwd())
    prop = torch.cuda.get_device_properties(0)
    return {
        "hostname": platform.node(),
        "gpu_name": prop.name,
        "gpu_total_memory_bytes": int(prop.total_memory),
        "system_ram_total_bytes": int(vm.total),
        "system_ram_available_bytes_at_start": int(vm.available),
        "disk_total_bytes": int(disk.total),
        "disk_free_bytes_at_start": int(disk.free),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
    }


def _base_record(dialog, turn, method_key, order, order_position,
                 history_tokens, prompt, budget):
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "visdial_v1.0_val",
        "dialog_id": dialog["dialog_id"],
        "image_id": dialog["image_ids"][0],
        "turn_id": int(turn["turn_id"]),
        "method_key": method_key,
        "method": METHODS[method_key]["label"],
        "budget": budget,
        "method_order": list(order),
        "method_order_position": int(order_position),
        "active_images": 1,
        "active_image_ids": list(turn["active_image_ids"]),
        "new_images_this_turn": len(turn["new_image_ids"]),
        "new_image_ids": list(turn["new_image_ids"]),
        "history_policy": "gold_teacher_forced",
        "history_tokens": int(history_tokens),
        "history_text_tokens": int(history_tokens),
        "question": turn["question"],
        "gold": turn["gold_answer"],
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "text_history_sha256": hashlib.sha256(
            visdial_prior_history_text(dialog, int(turn["turn_id"])).encode(
                "utf-8")
        ).hexdigest(),
    }


def _timing_fields(result, request_started_at, phase, request_returned_at):
    core_started = float(result["core_started_at_s"])
    first_token = float(result["first_token_at_s"])
    model_finished = float(result["model_finished_at_s"])
    server_postprocess_finished = float(result["postprocess_finished_at_s"])
    # Use the common caller-return boundary for request E2E so all method-owned
    # result assembly is included.  The server's internal timestamp remains a
    # diagnostic of the tiny Python return overhead.
    request_finished = float(request_returned_at)
    pre_core_ms = (core_started - request_started_at) * 1e3
    core_ttft_ms = (first_token - core_started) * 1e3
    end_to_end_ttft_ms = (first_token - request_started_at) * 1e3
    decode_ms = (model_finished - first_token) * 1e3
    model_e2e_ms = (model_finished - request_started_at) * 1e3
    postprocess_ms = (request_finished - model_finished) * 1e3
    request_e2e_ms = (request_finished - request_started_at) * 1e3
    phase_sum = sum(float(phase.get(key, 0.0)) for key in (
        "prompt_build_ms", "tokenization_ms", "image_preprocess_ms",
        "input_prepare_ms", "input_h2d_ms",
    ))
    return {
        **phase,
        "pre_core_ms": float(pre_core_ms),
        "pre_core_phase_sum_ms": float(phase_sum),
        "pre_core_unattributed_ms": float(pre_core_ms - phase_sum),
        "core_ttft_ms": float(core_ttft_ms),
        "end_to_end_ttft_ms": float(end_to_end_ttft_ms),
        "ttft_ms": float(end_to_end_ttft_ms),
        "decode_ms": float(decode_ms),
        "model_e2e_ms": float(model_e2e_ms),
        "postprocess_ms": float(postprocess_ms),
        "request_e2e_ms": float(request_e2e_ms),
        "e2e_ms": float(request_e2e_ms),
        "request_started_at_s": float(request_started_at),
        "core_started_at_s": core_started,
        "first_token_at_s": first_token,
        "model_finished_at_s": model_finished,
        "server_postprocess_finished_at_s": server_postprocess_finished,
        "postprocess_finished_at_s": request_finished,
        "request_finished_at_s": request_finished,
        "caller_returned_at_s": float(request_returned_at),
        "caller_return_overhead_ms": float(
            max(0.0, request_returned_at - server_postprocess_finished) * 1e3
        ),
        "ttft_identity_error_ms": float(
            end_to_end_ttft_ms - (pre_core_ms + core_ttft_ms)
        ),
        "model_e2e_identity_error_ms": float(
            model_e2e_ms - (end_to_end_ttft_ms + decode_ms)
        ),
        "request_e2e_identity_error_ms": float(
            request_e2e_ms - (model_e2e_ms + postprocess_ms)
        ),
    }


def _io_fields(result, method_key, full_visual_bytes=None):
    io = result.get("io") or {
        "bytes": 0, "ms": 0.0, "preads": 0, "chunk_units": 0,
        "per_kind": {},
    }
    per_kind = io.get("per_kind", {})
    normal_bytes = sum(int(per_kind.get(kind, {}).get("bytes", 0))
                       for kind in ("k", "v"))
    sep_bytes = int(per_kind.get("sep", {}).get("bytes", 0))
    if method_key == "recompute":
        selected_bytes = 0
        selected_ratio = None
    elif method_key == "fullload":
        selected_bytes = normal_bytes
        selected_ratio = 1.0
    else:
        selected_bytes = normal_bytes + sep_bytes
        selected_ratio = result.get("logical_kv_ratio")
    return {
        "ssd_read_ms": float(io.get("ms", 0.0)),
        "ssd_read_bytes": int(io.get("bytes", 0)),
        "ssd_read_preads": int(io.get("preads", 0)),
        "ssd_read_chunks": int(io.get("chunk_units", 0)),
        "ssd_read_chunk_units": int(io.get("chunk_units", 0)),
        "total_actual_pread_bytes": int(io.get("bytes", 0)),
        "normal_kv_read_bytes": int(normal_bytes),
        "separator_read_bytes": int(sep_bytes),
        "io_detail": per_kind,
        "selected_visual_kv_bytes": int(selected_bytes),
        "selected_kv_ratio": (None if selected_ratio is None
                              else float(selected_ratio)),
        "ssd_payload_ratio_vs_full_visual_kv": (
            float(selected_bytes) / float(full_visual_bytes)
            if full_visual_bytes and method_key != "recompute" else None
        ),
        "selector_ms": float(result.get("selector_ms", 0.0) or 0.0),
        "scatter_ms": (None if method_key == "fullload" else
                       float(result.get("scatter_ms", 0.0) or 0.0)),
        "prefill_ms": float(result.get("prefill_ms", 0.0) or 0.0),
        "prefill_inclusive_ms": (
            float(result.get("prefill_ms", 0.0) or 0.0)
            if method_key in {"recompute", "fullload"} else None
        ),
        "hook_total_ms": (float(result.get("hook_ms", 0.0) or 0.0)
                          if method_key == "fullload" else None),
        "n_chunks_selected": result.get("n_chunks_selected"),
        "n_chunks_total": result.get("n_chunks_total"),
        "selection_mode": result.get("selection_mode"),
        "selected_chunk_ids_per_layer": result.get(
            "selected_chunk_ids_per_layer"),
        "touched_chunk_fraction": result.get("touched_chunk_fraction"),
        "logical_kv_ratio_per_layer": result.get(
            "logical_kv_ratio_per_layer"),
        "static_score_calls": int(result.get("static_score_calls", 0)),
        "query_score_calls": int(result.get("query_score_calls", 0)),
        "diversity_calls": int(result.get("diversity_calls", 0)),
    }


def _run_normal_request(runner, server, image, dialog, turn,
                        capture_saliency, capture_cache):
    capture = VisionForwardCapture(
        runner, capture_saliency=capture_saliency)
    # Hook registration and lazy CUDA-event allocation are measurement setup,
    # so enter before the request boundary.  Only the actual same-request
    # saliency reduction remains in Turn-1 latency.
    with capture:
        torch.cuda.synchronize()
        request_started = time.perf_counter()
        started = time.perf_counter()
        prompt = visdial_prompt(dialog, int(turn["turn_id"]))
        prompt_build_ms = (time.perf_counter() - started) * 1e3

        enc_cpu, processor_timing = _exact_processor_call(
            runner, image, prompt)
        input_prepare_ms = float(processor_timing["input_prepare_ms"])
        h2d_started = time.perf_counter()
        enc_device = runner.to_device(enc_cpu)
        torch.cuda.synchronize()
        input_h2d_ms = (time.perf_counter() - h2d_started) * 1e3
        phase = {
            "prompt_build_ms": float(prompt_build_ms),
            "tokenization_ms": float(processor_timing["tokenization_ms"]),
            "image_preprocess_ms": float(
                processor_timing["image_preprocess_ms"]),
            "input_prepare_ms": input_prepare_ms,
            "input_h2d_ms": float(input_h2d_ms),
            "processor_total_ms": float(
                processor_timing["processor_total_ms"]),
        }
        result = server.recompute(
            enc_device, return_past_key_values=capture_cache)
        method_returned = time.perf_counter()
    capture_finished = time.perf_counter()
    result.update(_timing_fields(
        result, request_started, phase, method_returned))
    result.update({
        "prompt": prompt,
        "enc_cpu": enc_cpu,
        "capture": capture,
        "capture_stats": capture.stats(),
        "vision_ms": float(capture.stats()["vision_ms"]),
        "vision_forward_count": int(capture.call_count),
        "separate_vision_forward_count": 0,
        "capture_cleanup_finished_at_s": float(capture_finished),
        "capture_post_answer_ms": float(max(
            0.0, capture_finished - result["postprocess_finished_at_s"]
        ) * 1e3),
    })
    return result


def _run_stored_request(runner, server, ctx, dialog, turn, method_key,
                        budget, cold, seed, image_id):
    # This assertion-only hook is benchmark instrumentation, not request
    # processing.  Register it before page-cache conditioning and before the
    # request boundary, just like VisionForwardCapture on the normal path.
    with _NoVisionForward(runner) as guard:
        conditioning_started = time.perf_counter()
        conditioning_method = "none_warm"
        if cold:
            ctx.reader.drop_all()
            conditioning_method = "posix_fadvise_DONTNEED"
        conditioning_finished = time.perf_counter()

        torch.cuda.synchronize()
        request_started = time.perf_counter()
        started = time.perf_counter()
        prompt = visdial_prompt(dialog, int(turn["turn_id"]))
        prompt_build_ms = (time.perf_counter() - started) * 1e3
        started = time.perf_counter()
        tokenized = runner.processor.tokenizer(prompt, return_tensors="pt")
        tokenization_ms = (time.perf_counter() - started) * 1e3
        started = time.perf_counter()
        suffix_cpu = _suffix_from_tokenized(runner, tokenized)
        input_prepare_ms = (time.perf_counter() - started) * 1e3
        h2d_started = time.perf_counter()
        suffix_device = suffix_cpu.to(runner.model.device)
        torch.cuda.synchronize()
        input_h2d_ms = (time.perf_counter() - h2d_started) * 1e3
        phase = {
            "prompt_build_ms": float(prompt_build_ms),
            "tokenization_ms": float(tokenization_ms),
            "image_preprocess_ms": 0.0,
            "input_prepare_ms": float(input_prepare_ms),
            "input_h2d_ms": float(input_h2d_ms),
            "processor_total_ms": None,
        }
        if method_key == "fullload":
            result = server.request(
                ctx, mode="fullload", cold=False,
                suffix_ids=suffix_device)
        else:
            result = server.request_cvpr25(
                ctx, static=None, budget=budget, mode="prefix",
                sep_policy="sidecar", cold=False, seed=seed,
                image_id=image_id, suffix_ids=suffix_device,
                expected_prefix_layout="visionzip_image_only")
        method_returned = time.perf_counter()
    result.update(_timing_fields(
        result, request_started, phase, method_returned))
    result.update({
        "prompt": prompt,
        "suffix_cpu": suffix_cpu,
        "vision_ms": 0.0,
        "vision_forward_count": int(guard.calls),
        "separate_vision_forward_count": 0,
        "page_cache_conditioning_started_at_s": float(
            conditioning_started),
        "page_cache_conditioning_finished_at_s": float(
            conditioning_finished),
        "page_cache_conditioning_ms": float(
            (conditioning_finished - conditioning_started) * 1e3),
        "page_cache_conditioning_method": conditioning_method,
        "page_cache_conditioning_excluded_from_ttft": True,
        "cache_conditioning_started_at_s": float(conditioning_started),
        "cache_conditioning_finished_at_s": float(conditioning_finished),
        "cache_conditioning_excluded_from_ttft": True,
    })
    return result


def _history_token_count(runner, dialog, turn_id):
    history = visdial_prior_history_text(dialog, turn_id)
    return len(runner.processor.tokenizer(
        history, add_special_tokens=False).input_ids)


def _request_keys(dialogs, max_turns):
    return [[dialog["dialog_id"], int(turn["turn_id"])]
            for dialog in dialogs
            for turn in dialog["turns"][:max_turns]]


def _persistence_row(dialog, source_method, source_execution_id,
                     capture, persisted, persist_started_at,
                     actual_context_ready_at, context_open_ms):
    timing = persisted["timing_ms"]
    capture_stats = capture.stats()
    helper_persist_ms = float(timing["persist_ms"])
    saliency_d2h_ms = float(
        capture_stats.get("saliency_materialize_ms", 0.0))
    # The helper hashes bounded samples after durable publication but before
    # returning.  Those diagnostics are explicitly outside store-ready.  The
    # conservative serving cost is post-answer saliency D2H + durable helper
    # work + opening/validating the serving context.
    persist_ms = saliency_d2h_ms + helper_persist_ms + context_open_ms
    logical_persist_started_at = persist_started_at - saliency_d2h_ms / 1e3
    logical_store_ready_at = logical_persist_started_at + persist_ms / 1e3
    saliency_postprocess_ms = float(
        saliency_d2h_ms
        + timing.get("token_mapping_ms", 0.0)
    )
    repack_ms = float(
        timing.get("kv_materialize_ms", 0.0)
        + timing.get("kv_repack_ms", 0.0)
    )
    fsync_ms = float(
        timing.get("file_fsync_ms", 0.0)
        + timing.get("directory_fsync_ms", 0.0)
        + timing.get("parent_fsync_ms", 0.0)
    )
    accounted = (
        saliency_postprocess_ms + float(timing["permutation_ms"])
        + repack_ms + float(timing["ssd_write_ms"])
        + fsync_ms + float(timing.get("atomic_rename_ms", 0.0))
        + context_open_ms
    )
    hashes = persisted.get("hashes", {})
    file_hashes = hashes.get("files_sha256", {})
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "visdial_v1.0_val",
        "dialog_id": dialog["dialog_id"],
        "image_id": dialog["image_ids"][0],
        "persist_id": str(uuid.uuid4()),
        "source_turn_id": 1,
        "source_method": METHODS[source_method]["label"],
        "source_method_key": source_method,
        "source_execution_id": source_execution_id,
        "capture_from_turn1": True,
        "capture_from_same_answer1_forward": True,
        "vision_forward_count": int(capture.call_count),
        "separate_vision_forward_count": 0,
        "separate_prefix_forward_count": 0,
        "saliency_capture_mode": (
            "penultimate_cls_to_patch_attention_same_vision_forward"),
        "saliency_extra_ms": float(
            capture_stats.get("saliency_reduction_ms", 0.0)),
        "saliency_postprocess_ms": saliency_postprocess_ms,
        "saliency_d2h_ms": saliency_d2h_ms,
        "token_mapping_ms": float(timing.get("token_mapping_ms", 0.0)),
        "permutation_ms": float(timing["permutation_ms"]),
        "kv_materialize_ms": float(timing["kv_materialize_ms"]),
        "kv_repack_ms": float(timing["kv_repack_ms"]),
        "repack_ms": repack_ms,
        "buffered_write_ms": float(timing["ssd_write_ms"]),
        "ssd_write_ms": float(timing["ssd_write_ms"]),
        "layout_write_ms": float(timing.get("layout_write_ms", 0.0)),
        "fsync_ms": fsync_ms,
        "atomic_rename_ms": float(timing.get("atomic_rename_ms", 0.0)),
        "context_open_ms": float(context_open_ms),
        "helper_persist_ms": helper_persist_ms,
        "persist_unattributed_ms": float(persist_ms - accounted),
        "persist_ms": float(persist_ms),
        "visual_kv_bytes": int(persisted["bytes"]["visual_kv"]),
        "separator_sidecar_bytes": int(
            persisted["bytes"]["separator_sidecar"]),
        "probe_sidecar_bytes": int(persisted["bytes"]["probe_sidecar"]),
        "sys_kv_bytes": int(persisted["bytes"]["system_prefix"]),
        "layout_metadata_bytes": int(
            persisted["bytes"]["layout_metadata"]),
        "meta_bytes": int(persisted["bytes"]["meta"]),
        "total_ssd_write_bytes": int(persisted["bytes"]["total"]),
        "permutation_sha256": hashes.get("permutation_sha256"),
        "prefix_kv_sample_sha256": hashes.get(
            "prefix_kv_sample_sha256"),
        "meta_sha256": file_hashes.get("meta.json"),
        "layout_sha256": file_hashes.get("visionzip_layout.pt"),
        "persist_started_at_s": float(logical_persist_started_at),
        "helper_started_at_s": float(persist_started_at),
        "write_finished_at_s": float(
            logical_persist_started_at
            + (saliency_postprocess_ms + timing["permutation_ms"]
               + repack_ms + timing["ssd_write_ms"]) / 1e3),
        "fsync_finished_at_s": float(
            persist_started_at + helper_persist_ms / 1e3),
        "store_ready_at_s": float(logical_store_ready_at),
        "actual_context_ready_at_s": float(actual_context_ready_at),
        "durable_fsync_completed": bool(
            persisted["durability"]["parent_fsynced_after_rename"]),
        "atomic_no_clobber": bool(
            persisted["durability"]["atomic_no_clobber"]),
        "full_integrity_hash_enabled": bool(
            hashes.get("full_integrity_hash", False)),
        "integrity_hash_ms_excluded_from_persist": float(
            timing.get("integrity_hash_ms", 0.0)),
        "layout_uses_dataset_question": False,
        "future_turns_used_for_layout": 0,
        "calibration_questions": 0,
    }


def _write_flat_csv(path: Path, rows) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (json.dumps(value, sort_keys=True)
                      if isinstance(value, (dict, list)) else value)
                for key, value in row.items()
            })


def _validate_plan(dialogs, method_keys, max_turns, index_sha, seed):
    assert index_sha == EXPECTED_MAIN_INDEX_SHA256, (
        "this final experiment is pinned to the existing VisDial main index"
    )
    assert method_keys[:3] == ["recompute", "prefix25", "prefix45"], (
        "the three required arms and their canonical base order are fixed"
    )
    assert len(dialogs) in (2, 100), (
        "only the requested 2-dialog smoke or 100-dialog main workload is valid"
    )
    assert max_turns == 10
    assert all(len(dialog["turns"]) == 10 for dialog in dialogs)
    assert len({dialog["dialog_id"] for dialog in dialogs}) == len(dialogs)
    assert len({dialog["image_ids"][0] for dialog in dialogs}) == len(dialogs)
    assert seed == 1234


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--store", type=Path, required=True,
                        help="new, empty root for captured stores")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--max-dialogs", type=int, choices=(2, 100),
                        default=100)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warm", action="store_true")
    parser.add_argument("--min-free-after-gib", type=float, default=20.0)
    parser.add_argument("--correctness-validation", type=Path, default=None,
                        help="passed 2-dialog captured-vs-direct validation; "
                             "required by the main launcher")
    args = parser.parse_args()

    index_path = args.index.resolve()
    store_root = args.store.resolve()
    run_dir = args.run_dir.resolve()
    method_keys = [part.strip() for part in args.methods.split(",")
                   if part.strip()]
    if len(set(method_keys)) != len(method_keys):
        raise ValueError("duplicate method key")
    if any(key not in METHODS for key in method_keys):
        raise ValueError(f"unknown methods: {method_keys}")
    if set(method_keys) not in (
        {"recompute", "prefix25", "prefix45"},
        {"recompute", "prefix25", "prefix45", "fullload"},
    ):
        raise ValueError("only the three main arms plus optional FullLoad are valid")

    dialogs = load_canonical(index_path)[:args.max_dialogs]
    index_sha = sha256_file(index_path)
    _validate_plan(dialogs, method_keys, args.max_turns, index_sha, args.seed)
    expected_keys = _request_keys(dialogs, args.max_turns)
    request_keys_sha = stable_json_sha256(expected_keys)
    if args.max_dialogs == 100:
        assert request_keys_sha == EXPECTED_MAIN_REQUEST_KEYS_SHA256, (
            request_keys_sha, EXPECTED_MAIN_REQUEST_KEYS_SHA256)
    correctness_source = None
    if args.correctness_validation is not None:
        validation_path = args.correctness_validation.resolve()
        with validation_path.open() as handle:
            validation_payload = json.load(handle)
        checks = validation_payload.get("checks")
        assert validation_payload.get("passed") is True
        assert isinstance(checks, dict) and checks
        assert all(value is True for value in checks.values())
        correctness_source = {
            "path": str(validation_path),
            "sha256": sha256_file(validation_path),
        }
    if args.max_dialogs == 100 and correctness_source is None:
        raise ValueError(
            "the 100-dialog main run requires --correctness-validation from "
            "the completed 2-dialog smoke/direct-builder gate")

    # Both roots are experiment-owned and must be new.  Existing results and
    # even an incomplete prior attempt are never overwritten implicitly.
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    store_root.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(exist_ok=False)
    try:
        store_root.mkdir(exist_ok=False)
    except BaseException:
        # The run directory is kept as evidence; no recursive cleanup of an
        # existing or partially written artifact is ever attempted here.
        raise

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    runner = LlavaRunner().load()
    server = Server(runner, max_new_tokens=args.max_new_tokens)
    warmup = _run_unmeasured_warmup(runner, server)
    torch.cuda.reset_peak_memory_stats()
    machine = _machine_metadata()
    request_count = len(expected_keys) * len(method_keys)
    config = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "dataset": "visdial_v1.0_val",
        "index": str(index_path),
        "index_sha256": index_sha,
        "expected_main_index_sha256": EXPECTED_MAIN_INDEX_SHA256,
        "expected_request_keys": expected_keys,
        "expected_request_keys_sha256": request_keys_sha,
        "expected_main_request_keys_sha256": (
            EXPECTED_MAIN_REQUEST_KEYS_SHA256),
        "correctness_validation_source": correctness_source,
        "seed": args.seed,
        "n_dialogs": len(dialogs),
        "n_unique_images": len({d["image_ids"][0] for d in dialogs}),
        "turns_per_dialog": args.max_turns,
        "n_turns": len(expected_keys),
        "n_method_requests": request_count,
        "method_keys": method_keys,
        "methods": [METHODS[key]["label"] for key in method_keys],
        "budgets": {key: METHODS[key]["budget"] for key in method_keys},
        "method_order_policy": (
            "cyclic deterministic rotation by (zero_based_dialog_index + seed) "
            "modulo number of methods; same order at every turn of a dialog"),
        "method_orders": {
            dialog["dialog_id"]: list(deterministic_method_rotation(
                method_keys, index, args.seed))
            for index, dialog in enumerate(dialogs)
        },
        "method_order_by_dialog": {
            dialog["dialog_id"]: list(deterministic_method_rotation(
                method_keys, index, args.seed))
            for index, dialog in enumerate(dialogs)
        },
        "unmeasured_warmup": warmup,
        "warmup_policy": (
            "one deterministic synthetic-image request before the measured "
            "workload; actual combined processor, image H2D, normal multimodal "
            "generation, returned-cache path, and same-forward saliency hook; "
            "cache/saliency discarded and no SSD store written"),
        "warmup_excluded_from_latency": True,
        "warmup_uses_experiment_subset": False,
        "history_policy": "gold_teacher_forced",
        "prompt_template": "visdial-gold-history-v1",
        "quality_mode": "generative_auxiliary_normalized_match_not_official",
        "model": runner.model_id,
        "load_4bit": runner.load_4bit,
        "quantization": "4-bit NF4 double-quant",
        "attention": runner.attn,
        "decoding": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "chunk_size": CHUNK_SIZE,
        "physical_layout": "visionzip_image_only",
        "retrieval": "sequential_first_k_repacked_chunks",
        "separator_policy": "sidecar",
        "selector_algorithms_excluded": [
            "Static+Diverse", "SparseVLM", "query-dependent selector",
            "MaxMin"],
        "turn1_policy": (
            "all methods use normal image+Turn1 multimodal inference; one Ours "
            "execution exposes its existing past_key_values and same-forward "
            "image-only saliency; its cache is persisted immediately after "
            "Answer1 and released before a later counterfactual arm; no "
            "second vision or prefix forward"),
        "persistence_schedule": "immediately_after_source_answer1",
        "source_gpu_cache_released_before_next_method": True,
        "turn2_to_10_policy": (
            "ReComp repeats image processor+H2D+vision; Prefix/FullLoad use "
            "the one shared SSD store and never invoke vision"),
        "main_ttft_field": "end_to_end_ttft_ms",
        "main_ttft_metric": "end_to_end_ttft_ms",
        "input_preprocessing_timed": True,
        "page_cache_conditioning_in_ttft": False,
        "cold_page_cache": not args.warm,
        "ssd_read_api": "buffered_pread",
        "o_direct": False,
        "ssd_controller_cache_flushed": False,
        "future_turn_calibration_count": 0,
        "calibration_questions": 0,
        "layout_uses_dataset_question": False,
        "saliency_source": (
            "turn1_vision_penultimate_cls_to_patch_attention_head_sum"),
        "ttft_definition": (
            "after page-cache conditioning, timer before prompt construction "
            "and tokenization -> input preparation -> initial H2D -> ReComp "
            "vision or SSD KV read/scatter -> prefill -> first output token "
            "decision -> CUDA synchronize"),
        "core_ttft_definition": (
            "after initial request input H2D -> model/storage critical path -> "
            "first output token CUDA synchronize"),
        "e2e_definition": (
            "request timer start -> all generated tokens CUDA synchronize -> "
            "CPU detokenization and method result assembly"),
        "model_e2e_field": "model_e2e_ms",
        "request_e2e_field": "request_e2e_ms",
        "cpu_detokenization_in_request_e2e": True,
        "jpeg_file_read_and_decode_timed": False,
        "image_processor_resize_crop_tensorize_timed": True,
        "page_cache": {
            "condition": "OS page-cache cold" if not args.warm else "warm",
            "method": ("posix_fadvise(DONTNEED)"
                       if not args.warm else "none"),
            "conditioning_inside_ttft": False,
            "io_api": "buffered os.pread",
            "o_direct": False,
            "ssd_controller_cache_forced_flush": False,
            "forbidden_claims": ["true cold SSD", "SSD cache fully flushed"],
        },
        "persistence_scenarios": {
            "A": "back-to-back worst case; add persistence once for N>=2",
            "B": "ideal hidden-persistence reference; exclude persistence",
        },
        "store_root": str(store_root),
        "run_dir": str(run_dir),
        "machine": machine,
        "min_free_after_gib": args.min_free_after_gib,
        "run_started_at_unix": time.time(),
    }
    _atomic_json(run_dir / "config.json", config)

    raw_path = run_dir / "raw.jsonl"
    persistence_path = run_dir / "persistence.jsonl"
    persistence_rows = []
    all_rows = []
    process = psutil.Process()
    visual_sizes = []
    position_counts = defaultdict(Counter)
    run_started = time.perf_counter()

    with raw_path.open("x") as raw_handle, persistence_path.open("x") as persist_handle:
        for dialog_index, dialog in enumerate(dialogs):
            image_id = dialog["image_ids"][0]
            image_path = resolve_image_path(dialog["images"][0]["image_path"])
            # Dataset transport/decode is fixed benchmark setup; every method
            # still executes the actual image processor inside its timer.
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
            turns = dialog["turns"][:args.max_turns]
            history_counts = [
                _history_token_count(runner, dialog, int(turn["turn_id"]))
                for turn in turns]
            assert all(a <= b for a, b in zip(
                history_counts, history_counts[1:])), (
                "history token count is not monotonic", dialog["dialog_id"],
                history_counts)
            order = deterministic_method_rotation(
                method_keys, dialog_index, args.seed)
            for position, method_key in enumerate(order):
                position_counts[method_key][position] += 1

            # Pick the last Prefix arm in the rotated Turn-1 order as the
            # physical source.  Persist immediately after that arm returns,
            # then release its ~GB DynamicCache before another counterfactual
            # arm can run.  Both Prefix arms still use identical saliency
            # instrumentation and the normal multimodal path.
            prefix_in_order = [key for key in order if key.startswith("prefix")]
            source_method = prefix_in_order[-1]
            source_execution_id = None
            source_cache = None
            source_capture = None
            source_enc = None
            persistence = None
            persisted = None
            ctx = None
            full_visual_bytes = None
            turn1_rows = []
            turn1_input_hashes = {}
            turn1_first_tokens = {}
            turn1_predictions = {}
            turn1_saliency_hashes = {}

            turn = turns[0]
            history_tokens = _history_token_count(runner, dialog, 1)
            for order_position, method_key in enumerate(order):
                torch.cuda.reset_peak_memory_stats()
                execution_id = str(uuid.uuid4())
                is_prefix = method_key.startswith("prefix")
                capture_cache = method_key == source_method
                measured = _run_normal_request(
                    runner, server, image, dialog, turn,
                    capture_saliency=is_prefix,
                    capture_cache=capture_cache)
                enc_cpu = measured.pop("enc_cpu")
                capture = measured.pop("capture")
                prompt = measured.pop("prompt")
                cache = measured.pop("captured_past_key_values", None)
                request_gpu_memory_allocated = int(
                    torch.cuda.memory_allocated())
                request_gpu_peak_memory_allocated = int(
                    torch.cuda.max_memory_allocated())
                request_process_rss_bytes = int(process.memory_info().rss)
                input_hash = _hash_tensor_mapping(enc_cpu)
                image_hash = _image_input_hash(enc_cpu)
                suffix_placeholder_ids = runner.processor.tokenizer(
                    prompt, return_tensors="pt")["input_ids"]
                suffix_hash = _hash_tensor(_suffix_from_tokenized(
                    runner, {"input_ids": suffix_placeholder_ids}))
                turn1_input_hashes[method_key] = input_hash
                turn1_first_tokens[method_key] = int(measured["first_token_id"])
                turn1_predictions[method_key] = measured["answer"]
                if is_prefix:
                    turn1_saliency_hashes[method_key] = _hash_tensor(
                        capture.result_cpu())
                if capture_cache:
                    assert cache is not None
                    source_execution_id = execution_id
                    source_cache = cache
                    source_capture = capture
                    source_enc = enc_cpu

                    # This is the real per-system chronology: Answer 1 and
                    # saliency D2H have completed, then the already-produced
                    # cache is repacked/persisted exactly once.  Doing it here
                    # also prevents the source cache from changing the GPU
                    # memory state of a later measured Turn-1 arm.
                    persist_started_at = time.perf_counter()
                    persisted = persist_captured_visual_prefix(
                        runner, source_cache, source_enc["input_ids"],
                        source_enc["image_sizes"][0],
                        source_capture.result_cpu(),
                        store_root / image_id, image_id=image_id,
                        model_id=runner.model_id, chunk_size=CHUNK_SIZE,
                        image_input_sha256=_image_input_hash(source_enc),
                        capture_stats=source_capture,
                        extra_metadata={
                            "dataset": "visdial_v1.0_val",
                            "dialog_id": dialog["dialog_id"],
                            "source_turn_id": 1,
                            "source_execution_id": source_execution_id,
                            "history_policy": "gold_teacher_forced",
                            "future_turns_used_for_layout": 0,
                            "persistence_schedule":
                                "immediately_after_source_answer1",
                        })
                    # Remove both Python references to the same DynamicCache
                    # before any later method starts.  CUDA allocator/kernel
                    # warm state is intentionally retained.
                    cache = None
                    source_cache = None
                    torch.cuda.synchronize()
                    context_started = time.perf_counter()
                    ctx = ImageContext(
                        store_root / image_id, runner.model.device,
                        require_v_hidden=False)
                    ctx.validate_prefix_layout("visionzip_image_only")
                    context_open_ms = (
                        time.perf_counter() - context_started) * 1e3
                    store_ready_at = time.perf_counter()
                    persistence = _persistence_row(
                        dialog, source_method, source_execution_id,
                        source_capture, persisted, persist_started_at,
                        store_ready_at, context_open_ms)
                    persistence[
                        "persistence_started_immediately_after_source_response"
                    ] = True
                    persistence["source_gpu_cache_released_before_next_arm"] = True
                    persistence_rows.append(persistence)
                    persist_handle.write(json.dumps(persistence) + "\n")
                    persist_handle.flush()
                    os.fsync(persist_handle.fileno())
                    full_visual_bytes = int(ctx.meta["bytes_visual_kv"])
                    visual_sizes.append(full_visual_bytes)
                else:
                    assert cache is None

                base = _base_record(
                    dialog, turn, method_key, order, order_position,
                    history_tokens, prompt, METHODS[method_key]["budget"])
                record = {
                    **base,
                    "execution_id": execution_id,
                    "request_path": "normal_multimodal_turn1",
                    "execution_mode": "normal_multimodal_turn1",
                    "turn1_normal_inference": True,
                    "store_available_at_request_start": False,
                    "prediction": measured["answer"],
                    "first_token_id": int(measured["first_token_id"]),
                    "generated_tokens": int(measured["generated_tokens"]),
                    "quality_score": generative_match(
                        measured["answer"], turn["gold_answer"]),
                    "quality_metric": (
                        "auxiliary_normalized_match_not_official_visdial"),
                    "input_tensors_sha256": input_hash,
                    "input_ids_sha256": _hash_tensor(enc_cpu["input_ids"]),
                    "image_input_sha256": image_hash,
                    "pixel_values_sha256": _hash_tensor(
                        _normalise_image_tensor(enc_cpu["pixel_values"])),
                    "suffix_ids_sha256": suffix_hash,
                    "total_context_tokens": int(
                        enc_cpu["input_ids"].shape[1]),
                    "vision_ms": float(measured["vision_ms"]),
                    "vision_forward_count": int(
                        measured["vision_forward_count"]),
                    "separate_vision_forward_count": 0,
                    "saliency_capture_enabled": bool(is_prefix),
                    "saliency_extra_ms": float(
                        measured["capture_stats"].get(
                            "saliency_reduction_ms", 0.0)),
                    "saliency_post_answer_ms": float(
                        measured["capture_post_answer_ms"]),
                    "page_cache_conditioning_method": "not_applicable_turn1",
                    "page_cache_conditioning_excluded_from_ttft": True,
                    "page_cache_conditioning_ms": 0.0,
                    **_io_fields(measured, method_key),
                    **{key: value for key, value in measured.items()
                       if key in {
                           "prompt_build_ms", "tokenization_ms",
                           "image_preprocess_ms", "input_prepare_ms",
                           "input_h2d_ms", "processor_total_ms",
                           "pre_core_ms", "pre_core_phase_sum_ms",
                           "pre_core_unattributed_ms", "core_ttft_ms",
                           "end_to_end_ttft_ms", "ttft_ms", "decode_ms",
                           "model_e2e_ms", "postprocess_ms",
                           "request_e2e_ms", "e2e_ms",
                           "request_started_at_s", "core_started_at_s",
                           "first_token_at_s", "model_finished_at_s",
                           "postprocess_finished_at_s",
                           "request_finished_at_s",
                           "server_postprocess_finished_at_s",
                           "caller_returned_at_s",
                           "caller_return_overhead_ms",
                           "ttft_identity_error_ms",
                           "model_e2e_identity_error_ms",
                           "request_e2e_identity_error_ms"}},
                    "gpu_memory_allocated": request_gpu_memory_allocated,
                    "gpu_peak_memory_allocated":
                        request_gpu_peak_memory_allocated,
                    "process_rss_bytes": request_process_rss_bytes,
                }
                assert (record["total_context_tokens"]
                        + args.max_new_tokens
                        <= int(runner.cfg.text_config.max_position_embeddings))
                turn1_rows.append(record)
                print(
                    f"[{dialog_index + 1}/{len(dialogs)} t1] "
                    f"{METHODS[method_key]['label']}: "
                    f"e2e-TTFT={record['end_to_end_ttft_ms']:.1f}ms "
                    f"vision={record['vision_ms']:.1f}ms "
                    f"pred={record['prediction'][:45]!r}", flush=True)

            assert len(set(turn1_input_hashes.values())) == 1, (
                dialog["dialog_id"], turn1_input_hashes)
            assert len(set(turn1_first_tokens.values())) == 1, (
                "Turn1 first-token mismatch", dialog["dialog_id"],
                turn1_first_tokens)
            assert len(set(turn1_predictions.values())) == 1, (
                "Turn1 prediction mismatch", dialog["dialog_id"],
                turn1_predictions)
            assert len(set(turn1_saliency_hashes.values())) == 1, (
                "Prefix-arm saliency mismatch", dialog["dialog_id"],
                turn1_saliency_hashes)
            assert source_cache is None and source_capture is not None
            assert source_enc is not None and source_execution_id is not None
            assert persistence is not None and persisted is not None
            assert ctx is not None and full_visual_bytes is not None

            # Attach the now-known physical KV geometry to every Turn-1 row.
            for record in turn1_rows:
                record.update({
                    "visual_tokens": int(ctx.meta["v_token_num"]),
                    "active_visual_tokens": int(ctx.meta["v_token_num"]),
                    "full_visual_kv_bytes": full_visual_bytes,
                    "selected_visual_kv_bytes": 0,
                    "selected_kv_ratio": None,
                    "ssd_payload_ratio_vs_full_visual_kv": None,
                    "cumulative_visual_kv_bytes_created": full_visual_bytes,
                    "image_visual_kv_bytes": {image_id: full_visual_bytes},
                    "shared_persist_id": persistence["persist_id"],
                    "physical_store_shared_by_prefix_budgets": True,
                })
                all_rows.append(record)
                raw_handle.write(json.dumps(record) + "\n")
            raw_handle.flush()
            os.fsync(raw_handle.fileno())

            # Capacity check uses observed bytes, never a hard-coded KV size.
            remaining = len(dialogs) - dialog_index - 1
            projected = int(np.mean(visual_sizes) * remaining)
            free_now = shutil.disk_usage(store_root).free
            reserve = int(args.min_free_after_gib * (1024 ** 3))
            if free_now - projected < reserve:
                raise RuntimeError(
                    "insufficient disk for remaining captured stores while "
                    f"preserving {args.min_free_after_gib:.1f} GiB: free="
                    f"{free_now}, projected={projected}, reserve={reserve}"
                )

            for turn in turns[1:]:
                turn_id = int(turn["turn_id"])
                history_tokens = _history_token_count(
                    runner, dialog, turn_id)
                method_prompt_hashes = {}
                method_history_hashes = {}
                for order_position, method_key in enumerate(order):
                    torch.cuda.reset_peak_memory_stats()
                    execution_id = str(uuid.uuid4())
                    budget = METHODS[method_key]["budget"]
                    if method_key == "recompute":
                        measured = _run_normal_request(
                            runner, server, image, dialog, turn,
                            capture_saliency=False, capture_cache=False)
                        enc_cpu = measured.pop("enc_cpu")
                        measured.pop("capture")
                        prompt = measured.pop("prompt")
                        input_hash = _hash_tensor(
                            enc_cpu["input_ids"])
                        suffix_ids = runner.processor.tokenizer(
                            prompt, return_tensors="pt")["input_ids"]
                        suffix_hash = _hash_tensor(_suffix_from_tokenized(
                            runner, {"input_ids": suffix_ids}))
                        total_context_tokens = int(
                            enc_cpu["input_ids"].shape[1])
                        measured.update({
                            "page_cache_conditioning_method":
                                "not_applicable_recompute",
                            "page_cache_conditioning_excluded_from_ttft": True,
                            "page_cache_conditioning_ms": 0.0,
                        })
                    else:
                        measured = _run_stored_request(
                            runner, server, ctx, dialog, turn, method_key,
                            budget, not args.warm, args.seed, image_id)
                        suffix_cpu = measured.pop("suffix_cpu")
                        prompt = measured.pop("prompt")
                        input_hash = _hash_tensor(suffix_cpu)
                        suffix_hash = input_hash
                        total_context_tokens = int(
                            ctx.meta["prefix_len"] + suffix_cpu.numel())

                    method_prompt_hashes[method_key] = hashlib.sha256(
                        prompt.encode("utf-8")).hexdigest()
                    method_history_hashes[method_key] = hashlib.sha256(
                        visdial_prior_history_text(dialog, turn_id).encode(
                            "utf-8")).hexdigest()
                    base = _base_record(
                        dialog, turn, method_key, order, order_position,
                        history_tokens, prompt, budget)
                    record = {
                        **base,
                        "execution_id": execution_id,
                        "request_path": (
                            "normal_multimodal_recompute" if method_key == "recompute"
                            else "ssd_full_load" if method_key == "fullload"
                            else "ssd_sequential_prefix"),
                        "execution_mode": (
                            "normal_multimodal_recompute"
                            if method_key == "recompute"
                            else "ssd_full_load" if method_key == "fullload"
                            else "ssd_sequential_prefix"),
                        "turn1_normal_inference": False,
                        "store_available_at_request_start": (
                            method_key != "recompute"),
                        "prediction": measured["answer"],
                        "first_token_id": int(measured["first_token_id"]),
                        "generated_tokens": int(measured["generated_tokens"]),
                        "quality_score": generative_match(
                            measured["answer"], turn["gold_answer"]),
                        "quality_metric": (
                            "auxiliary_normalized_match_not_official_visdial"),
                        "input_tensors_sha256": input_hash,
                        "input_ids_sha256": (
                            _hash_tensor(enc_cpu["input_ids"])
                            if method_key == "recompute" else input_hash),
                        "suffix_ids_sha256": suffix_hash,
                        "visual_tokens": int(ctx.meta["v_token_num"]),
                        "active_visual_tokens": int(ctx.meta["v_token_num"]),
                        "total_context_tokens": total_context_tokens,
                        "vision_ms": float(measured["vision_ms"]),
                        "vision_forward_count": int(
                            measured["vision_forward_count"]),
                        "separate_vision_forward_count": 0,
                        "saliency_capture_enabled": False,
                        "saliency_extra_ms": 0.0,
                        "full_visual_kv_bytes": full_visual_bytes,
                        "cumulative_visual_kv_bytes_created": full_visual_bytes,
                        "image_visual_kv_bytes": {image_id: full_visual_bytes},
                        "shared_persist_id": persistence["persist_id"],
                        "physical_store_shared_by_prefix_budgets": True,
                        **_io_fields(measured, method_key, full_visual_bytes),
                        **{key: value for key, value in measured.items()
                           if key in {
                               "prompt_build_ms", "tokenization_ms",
                               "image_preprocess_ms", "input_prepare_ms",
                               "input_h2d_ms", "processor_total_ms",
                               "pre_core_ms", "pre_core_phase_sum_ms",
                               "pre_core_unattributed_ms", "core_ttft_ms",
                               "end_to_end_ttft_ms", "ttft_ms", "decode_ms",
                               "model_e2e_ms", "postprocess_ms",
                               "request_e2e_ms", "e2e_ms",
                               "request_started_at_s", "core_started_at_s",
                               "first_token_at_s", "model_finished_at_s",
                               "postprocess_finished_at_s",
                               "request_finished_at_s",
                               "server_postprocess_finished_at_s",
                               "caller_returned_at_s",
                               "caller_return_overhead_ms",
                               "ttft_identity_error_ms",
                               "model_e2e_identity_error_ms",
                               "request_e2e_identity_error_ms",
                               "page_cache_conditioning_started_at_s",
                               "page_cache_conditioning_finished_at_s",
                               "page_cache_conditioning_ms",
                               "page_cache_conditioning_method",
                               "page_cache_conditioning_excluded_from_ttft"}},
                        "gpu_memory_allocated": int(
                            torch.cuda.memory_allocated()),
                        "gpu_peak_memory_allocated": int(
                            torch.cuda.max_memory_allocated()),
                        "process_rss_bytes": int(process.memory_info().rss),
                    }
                    assert (record["total_context_tokens"]
                            + args.max_new_tokens
                            <= int(runner.cfg.text_config.max_position_embeddings))
                    # Prefix mode must be literal first-k loading, with no
                    # scoring or diversity calls.  Check every live request.
                    if method_key.startswith("prefix"):
                        expected_k = max(1, min(
                            int(ctx.meta["n_chunks_per_layer"]),
                            int(round(budget * int(
                                ctx.meta["n_chunks_per_layer"])))) )
                        selected = record["selected_chunk_ids_per_layer"]
                        assert selected and all(
                            ids == list(range(expected_k)) for ids in selected)
                        assert record["static_score_calls"] == 0
                        assert record["query_score_calls"] == 0
                        assert record["diversity_calls"] == 0
                    all_rows.append(record)
                    raw_handle.write(json.dumps(record) + "\n")
                    raw_handle.flush()
                    print(
                        f"[{dialog_index + 1}/{len(dialogs)} t{turn_id}] "
                        f"{METHODS[method_key]['label']}: "
                        f"e2e-TTFT={record['end_to_end_ttft_ms']:.1f}ms "
                        f"SSD={record['ssd_read_bytes']/1e6:.1f}MB "
                        f"pred={record['prediction'][:45]!r}", flush=True)
                assert len(set(method_prompt_hashes.values())) == 1
                assert len(set(method_history_hashes.values())) == 1

            ctx.close()
            del ctx, source_capture, source_enc, persisted, image
            torch.cuda.empty_cache()

    _write_flat_csv(run_dir / "persistence_per_image.csv", persistence_rows)
    # Analyzer reads JSONL for nested diagnostics; this CSV is a convenient
    # flat handoff and one of the requested primary artifacts.
    config.update({
        "status": "complete",
        "completed_at_unix": time.time(),
        "elapsed_seconds": time.perf_counter() - run_started,
        "observed_rows": len(all_rows),
        "observed_persistence_rows": len(persistence_rows),
        "observed_dialog_ids_sha256": stable_json_sha256(
            [dialog["dialog_id"] for dialog in dialogs]),
        "observed_method_position_counts": {
            method: {str(position): count
                     for position, count in sorted(counts.items())}
            for method, counts in position_counts.items()
        },
        "disk_free_bytes_at_end": int(shutil.disk_usage(store_root).free),
        "store_total_bytes": int(sum(
            row["total_ssd_write_bytes"] for row in persistence_rows)),
        "visual_kv_total_bytes": int(sum(visual_sizes)),
    })
    _atomic_json(run_dir / "config.json", config)
    print(json.dumps({
        "status": "complete", "run_dir": str(run_dir),
        "store_root": str(store_root), "rows": len(all_rows),
        "persistence_rows": len(persistence_rows),
        "elapsed_seconds": config["elapsed_seconds"],
    }, indent=1), flush=True)


if __name__ == "__main__":
    main()

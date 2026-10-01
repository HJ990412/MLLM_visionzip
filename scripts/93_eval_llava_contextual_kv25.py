#!/usr/bin/env python3
"""Gated, same-run, nine-arm LLaVA-NeXT GQA Visual-KV25 experiment.

Smoke and pilot are separate phases. Every arm performs its own normal pixel
Turn 1, then stored arms read their own SSD payload for independent T2 onward
questions. Only stores created by this run can be removed after their raw rows,
selection artifacts, persistence receipts, and image validation are durable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import psutil
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mmimpress.config import (  # noqa: E402
    ATTN_IMPL, CHUNK_SIZE, COMPUTE_DTYPE, LOAD_4BIT, MODEL_ID, PROBE_HEADS,
)
from mmimpress.dataset import METRICS, question_answers  # noqa: E402
from mmimpress.model import LlavaRunner  # noqa: E402
from mmimpress.piggyback import (  # noqa: E402
    DecoderVisualHiddenCapture, VisionForwardCapture,
    deterministic_method_rotation, persist_captured_raster_prefix,
    persist_captured_visual_prefix,
)
from mmimpress.serve import ImageContext, Server  # noqa: E402


def _load_old_runner():
    path = ROOT / "scripts/89_eval_llava_kv25.py"
    spec = importlib.util.spec_from_file_location("_validated_llava_kv25_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load validated runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


OLD = _load_old_runner()
QA = OLD.QA
SCHEMA = "llava-contextual-representative-kv25-v1"
LAYOUT = "visionzip_contextual_original_v1"
SEED = 1234
SOURCE_PATHS = (
    "mmimpress/cvpr25.py", "mmimpress/contextual_kv25.py",
    "mmimpress/piggyback.py", "mmimpress/store.py",
    "mmimpress/serve.py", "mmimpress/model.py",
    "scripts/49_eval_query_aware_baseline.py", "scripts/89_eval_llava_kv25.py",
    "scripts/92_validate_llava_contextual_kv25.py",
    "scripts/93_eval_llava_contextual_kv25.py",
)
METHODS = (
    "recompute", "fullload", "d25_c0", "d22_5_c2_5", "d20_c5",
    "d17_5_c7_5", "d15_c10", "d20_random5", "d20_uniform5",
)
METHOD_META = {
    "recompute": {"label": "ReComp", "variant": None, "alpha": None},
    "fullload": {"label": "FullLoad", "variant": None, "alpha": None},
    "d25_c0": {"label": "D25+C0", "variant": "dominant", "alpha": 0.0},
    "d22_5_c2_5": {"label": "D22.5+C2.5", "variant": "contextual", "alpha": 0.1},
    "d20_c5": {"label": "D20+C5", "variant": "contextual", "alpha": 0.2},
    "d17_5_c7_5": {"label": "D17.5+C7.5", "variant": "contextual", "alpha": 0.3},
    "d15_c10": {"label": "D15+C10", "variant": "contextual", "alpha": 0.4},
    "d20_random5": {"label": "D20+Random5", "variant": "random", "alpha": 0.2},
    "d20_uniform5": {"label": "D20+Uniform5", "variant": "uniform", "alpha": 0.2},
}
SELECTIVE = tuple(m for m in METHODS if m not in ("recompute", "fullload"))
GQA_INDEX_SHA = OLD.GQA_SHA
GQA_WORKLOAD_SHA = OLD.GQA_WORKLOAD_SHA
FIXED_SAMPLES = [
    ("n355567", "201751701"), ("n9181", "20929611"),
    ("n390187", "201861403"), ("n133585", "202108008"),
    ("n272098", "201535625"),
]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _hash_tensor(tensor: torch.Tensor) -> str:
    t = tensor.detach().to("cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(t.shape)).encode())
    digest.update(str(t.dtype).encode())
    digest.update(t.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _capture_prefix_cpu(cache, prefix_len: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    # Exact full prefix bits, including system and visual rows, not a sample.
    from mmimpress.piggyback import cache_layers
    result = []
    for key, value in cache_layers(cache):
        if key.shape != value.shape or key.shape[2] < prefix_len:
            raise AssertionError("captured cache shape differs from prefix")
        result.append((key[:, :, :prefix_len].detach().cpu().contiguous(),
                       value[:, :, :prefix_len].detach().cpu().contiguous()))
    if not result:
        raise AssertionError("captured cache has no decoder layers")
    return result


def _compare_prefix_cpu(reference, cache, prefix_len: int) -> None:
    from mmimpress.piggyback import cache_layers
    actual = cache_layers(cache)
    if len(actual) != len(reference):
        raise AssertionError("canonical layer count changed between arms")
    for layer, ((expected_k, expected_v), (key, value)) in enumerate(
            zip(reference, actual)):
        if key.shape[2] < prefix_len or value.shape[2] < prefix_len:
            raise AssertionError(f"canonical layer {layer} shortened")
        for name, expected, tensor in (("K", expected_k, key),
                                       ("V", expected_v, value)):
            got = tensor[:, :, :prefix_len].detach().cpu().contiguous()
            if not torch.equal(expected, got):
                raise AssertionError(
                    f"canonical Turn-1 captured {name} differs at layer {layer}")


def _normal_request(runner, server, image_path: Path, question: str, capture_kind: str,
                    capture_keys: bool = False):
    """The validated pixel request with optional same-forward vision-key hook."""
    if capture_kind not in ("none", "raster", "image_only"):
        raise ValueError(capture_kind)
    capture_cache = capture_kind != "none"
    vision = VisionForwardCapture(
        runner, capture_saliency=(capture_kind == "image_only"),
        capture_keys=capture_keys)
    hidden = None
    with vision:
        torch.cuda.synchronize()
        request_started = time.perf_counter()
        image_started = time.perf_counter()
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image.load()
        image_read_decode_ms = (time.perf_counter()-image_started)*1e3
        started = time.perf_counter()
        prompt = runner.prompt(question)
        prompt_ms = (time.perf_counter() - started) * 1e3
        enc_cpu, processor = QA._combined_processor(runner, image, prompt)
        visual_start, visual_count = runner.visual_span(enc_cpu["input_ids"])
        if capture_kind == "raster":
            hidden = DecoderVisualHiddenCapture(
                runner, visual_start, visual_count)
        stack = ExitStack()
        if hidden is not None:
            stack.enter_context(hidden)
        try:
            started = time.perf_counter()
            enc_device = runner.to_device(enc_cpu)
            torch.cuda.synchronize()
            h2d_ms = (time.perf_counter() - started) * 1e3
            result = server.recompute(
                enc_device, return_past_key_values=capture_cache)
            returned = time.perf_counter()
        finally:
            stack.close()
    phases = {
        "prompt_build_ms": float(prompt_ms),
        "image_read_decode_ms": float(image_read_decode_ms),
        "tokenization_ms": processor["tokenization_ms"],
        "image_preprocess_ms": processor["image_preprocess_ms"],
        "input_prepare_ms": processor["input_prepare_ms"],
        "input_h2d_ms": float(h2d_ms),
        "processor_total_ms": processor["processor_total_ms"],
    }
    result.update(QA._timing_fields(result, request_started, returned, phases))
    result.update({
        "vision_ms": float(vision.stats()["vision_ms"]),
        "vision_forward_count": int(vision.call_count),
        "vision_capture_stats": vision.stats(),
        "visual_hidden_capture_stats": hidden.stats() if hidden else None,
        "page_cache_conditioning_method": "not_applicable_pixels",
        "page_cache_conditioning_ms": 0.0,
        "page_cache_conditioning_excluded_from_ttft": True,
    })
    placeholder = runner.processor.tokenizer(prompt, return_tensors="pt")
    suffix = QA._suffix_from_tokenized(runner, placeholder)
    diagnostic = {
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "input_tensors_sha256": QA._hash_tensor_mapping(enc_cpu),
        "input_ids_sha256": QA._hash_tensor(enc_cpu["input_ids"]),
        "image_input_sha256": QA._image_input_hash(enc_cpu),
        "suffix_ids_sha256": QA._hash_tensor(suffix),
        "enc_cpu": enc_cpu, "vision_capture": vision, "hidden_capture": hidden,
        "prefix_len": int(visual_start + visual_count),
    }
    del enc_device
    return result, diagnostic


def _stored_request(runner, server, ctx, question: str, method: str,
                    image_id: str, full_visual_bytes: int):
    with QA._NoVisionForward(runner) as guard:
        condition_started = time.perf_counter()
        ctx.reader.drop_all()
        condition_done = time.perf_counter()
        torch.cuda.synchronize()
        request_started = time.perf_counter()
        prompt_started = time.perf_counter()
        prompt = runner.prompt(question)
        prompt_ms = (time.perf_counter() - prompt_started) * 1e3
        token_started = time.perf_counter()
        tokenized = runner.processor.tokenizer(prompt, return_tensors="pt")
        token_ms = (time.perf_counter() - token_started) * 1e3
        prepare_started = time.perf_counter()
        suffix_cpu = QA._suffix_from_tokenized(runner, tokenized)
        prepare_ms = (time.perf_counter() - prepare_started) * 1e3
        h2d_started = time.perf_counter()
        suffix_device = suffix_cpu.to(runner.model.device)
        torch.cuda.synchronize()
        h2d_ms = (time.perf_counter() - h2d_started) * 1e3
        if method == "fullload":
            result = server.request(ctx, mode="fullload", cold=False,
                                    suffix_ids=suffix_device)
        else:
            result = server.request_cvpr25(
                ctx, static=None, budget=0.25, mode="prefix",
                budget_unit="visual_kv", sep_policy="sidecar", cold=False,
                seed=SEED, image_id=image_id, suffix_ids=suffix_device,
                expected_prefix_layout=LAYOUT)
        returned = time.perf_counter()
    result.update(QA._timing_fields(result, request_started, returned, {
        "prompt_build_ms": float(prompt_ms),
        "tokenization_ms": float(token_ms),
        "image_preprocess_ms": 0.0,
        "input_prepare_ms": float(prepare_ms),
        "input_h2d_ms": float(h2d_ms), "processor_total_ms": None,
    }))
    result.update({
        "vision_forward_count": int(guard.calls),
        "page_cache_conditioning_started_at_s": condition_started,
        "page_cache_conditioning_finished_at_s": condition_done,
        "page_cache_conditioning_ms": (condition_done-condition_started)*1e3,
        "page_cache_conditioning_method": "posix_fadvise_DONTNEED",
        "page_cache_conditioning_excluded_from_ttft": True,
    })
    result = QA._json_result(result, "fullload" if method == "fullload"
                             else "ours25", full_visual_bytes)
    return result, {
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "suffix_ids_sha256": QA._hash_tensor(suffix_cpu),
    }


def _persist_store(runner, result, diagnostic, image_id: str,
                   destination: Path, method: str, phase: str):
    cache = result.get("captured_past_key_values")
    if cache is None:
        raise AssertionError("normal Turn-1 did not return its captured cache")
    encoded = diagnostic["enc_cpu"]
    common = {
        "image_id": image_id, "model_id": runner.model_id,
        "chunk_size": CHUNK_SIZE,
        "image_input_sha256": diagnostic["image_input_sha256"],
        "extra_metadata": {
            "dataset": "gqa", "experiment_schema": SCHEMA,
            "phase": phase, "source_method_key": method, "source_turn_id": 1,
            "store_lifecycle": "run_local_image_scoped_cleanup_after_receipts",
        },
        "full_integrity_hash": False,
    }
    if method == "fullload":
        hidden = diagnostic["hidden_capture"]
        persisted = persist_captured_raster_prefix(
            runner, cache, encoded["input_ids"], encoded["image_sizes"][0],
            hidden.result_cpu(), destination, probe_heads=PROBE_HEADS,
            hidden_capture_stats=hidden, **common)
    else:
        spec = METHOD_META[method]
        vision = diagnostic["vision_capture"]
        descriptors = (vision.result_keys_cpu()
                       if spec["variant"] == "contextual" else None)
        persisted = persist_captured_visual_prefix(
            runner, cache, encoded["input_ids"], encoded["image_sizes"][0],
            vision.result_cpu(), destination, capture_stats=vision,
            selection_variant=spec["variant"],
            contextual_alpha=spec["alpha"],
            vision_key_descriptors=descriptors,
            selection_seed=SEED, **common)
    if not persisted["integrity"]["ok"]:
        raise AssertionError(f"store integrity failed: {destination}")
    return persisted


def _selection_artifact(context: ImageContext, method: str, destination: Path,
                        phase: str, image_id: str) -> dict[str, Any]:
    meta = context.meta
    n = int(meta["n_spatial"])
    k = (n + 3) // 4
    c = int(math.floor(float(METHOD_META[method]["alpha"]) * k))
    order = [int(x) for x in meta["order"]]
    structural = [int(x) for x in meta["newline_idx"]]
    if len(order) != int(meta["v_token_num"]) or len(set(order)) != len(order):
        raise AssertionError("invalid stored-to-original permutation")
    if order[n:] != structural:
        raise AssertionError("structural separator tail changed")
    selected = order[:k]
    if len(set(selected)) != k or set(selected) & set(structural):
        raise AssertionError("selected content ID count or structural status changed")
    layout_path = context.dir / "visionzip_layout.pt"
    layout = torch.load(layout_path, map_location="cpu", weights_only=True)
    if not isinstance(layout, Mapping):
        raise AssertionError("new layout artifact must be a mapping")
    scores = torch.as_tensor(layout["token_score_original"]).flatten()
    if scores.numel() != len(order):
        raise AssertionError("saliency score count disagrees with visual span")
    if any(not math.isfinite(float(scores[i])) for i in order[:n]):
        raise AssertionError("nonfinite content saliency score")
    layout_json = OLD.safe({key: value for key, value in layout.items()
                            if key != "token_score_original"})
    # The existing structural score sentinel is +Inf; JSON records it as null.
    layout_json["token_score_original"] = [
        float(x) if math.isfinite(float(x)) else None for x in scores]
    inverse = [0] * len(order)
    for stored, original in enumerate(order):
        inverse[original] = stored
    payload = {
        "schema_version": SCHEMA, "phase": phase, "image_id": image_id,
        "method_key": method, "method_label": METHOD_META[method]["label"],
        "selection_variant": METHOD_META[method]["variant"],
        "layout_policy": LAYOUT, "alpha": METHOD_META[method]["alpha"],
        "N_content": n, "k": k, "k_dominant": k-c, "k_context": c,
        "actual_dominant_ratio": (k-c)/n,
        "actual_contextual_ratio": c/n,
        "actual_content_retention": k/n,
        "selected_original_ids": selected,
        "stored_to_original": order, "original_to_stored": inverse,
        "structural_original_ids": structural,
        "store_layout_sha256": OLD.sha256_file(layout_path),
        "store_meta_sha256": OLD.sha256_file(context.dir / "meta.json"),
        "layout_artifact": layout_json,
    }
    OLD.atomic_json(destination, payload)
    return payload


def _make_row(*, result: Mapping[str, Any], diagnostic: Mapping[str, Any],
              meta: Mapping[str, Any] | None, method: str, image_id: str,
              image_index: int, question: Mapping[str, Any], turn_id: int,
              order: tuple[str, ...], order_position: int, phase: str,
              image_sha: str, config_sha: str, manifest_sha: str,
              selection_artifact_path: str | None) -> dict[str, Any]:
    value = {key: item for key, item in result.items()
             if key != "captured_past_key_values"}
    if turn_id > 1 and method != "recompute":
        metrics = OLD.content_metrics(
            value, "fullload" if method == "fullload" else "ours_kv25", meta)
    else:
        metrics = OLD.content_metrics({}, "recompute", None)
    gold = question_answers(question)
    prediction = str(value["answer"])
    score = METRICS["gqa"](prediction, gold)
    row = {
        **value, **metrics,
        "schema_version": SCHEMA, "dataset": "gqa", "phase": phase,
        "request_id": f"{phase}:{image_id}:{question['question_id']}:{method}",
        "image_id": image_id, "image_index": image_index,
        "image_sha256": image_sha,
        "question_id": str(question["question_id"]),
        "question": str(question["question"]), "gold": gold,
        "turn_id": turn_id, "method_key": method,
        "method_id": method, "method_label": METHOD_META[method]["label"],
        "selection_variant": METHOD_META[method]["variant"],
        "alpha": METHOD_META[method]["alpha"],
        "budget_unit": ("none" if method == "recompute" else
                        "full_visual_kv" if method == "fullload" else "visual_kv"),
        "budget_ratio": 0.25 if method in SELECTIVE else None,
        "chunk_size": CHUNK_SIZE, "model": MODEL_ID,
        "attention_backend": ATTN_IMPL, "compute_dtype": str(COMPUTE_DTYPE),
        "load_4bit": LOAD_4BIT, "config_sha256": config_sha,
        "manifest_sha256": manifest_sha,
        "method_order": list(order), "method_order_position": order_position,
        "request_path": "normal_pixel_turn1" if turn_id == 1 else
                        "normal_pixel_recompute" if method == "recompute"
                        else "ssd_visual_kv",
        "cache_hit": bool(turn_id > 1 and method != "recompute"),
        "prompt_sha256": diagnostic["prompt_sha256"],
        "suffix_ids_sha256": diagnostic["suffix_ids_sha256"],
        "selection_artifact": selection_artifact_path,
        "prediction": prediction, "correct": float(score),
        "status": "ok",
    }
    if row["cache_hit"]:
        if int(row["vision_forward_count"]) != 0:
            raise AssertionError("cache hit invoked vision")
        for key in ("query_score_calls", "static_score_calls"):
            if int(row.get(key, 0)) != 0:
                raise AssertionError(f"cache hit invoked {key}")
        if int(row.get("probe_read_bytes", 0)) != 0:
            raise AssertionError("cache hit read descriptor/probe payload")
    if not 0 < float(row["end_to_end_ttft_ms"]):
        raise AssertionError("invalid measured outer TTFT")
    return OLD.safe(row)


def _delete_image_stores(run_dir: Path, phase: str, image_id: str,
                         stores: Mapping[str, Path], cleanup_handle) -> None:
    root = run_dir / "stores" / phase / image_id
    if not root.is_dir() or root.is_symlink():
        raise AssertionError("cleanup root missing or unsafe")
    for method, path in stores.items():
        if path.parent != root or path.is_symlink() or not path.is_dir():
            raise AssertionError(f"cleanup target outside new image store: {path}")
        files = sorted(p.relative_to(path).as_posix()
                       for p in path.rglob("*") if p.is_file())
        bytes_used = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        event = {
            "at_utc": _now(), "phase": phase, "image_id": image_id,
            "method_key": method, "target_path": str(path),
            "files": files, "logical_bytes": bytes_used,
            "scope": "only_run_created_image_store",
            "reproduction": "rebuild_from_normal_T1_and_frozen_input",
        }
        OLD.append_jsonl(cleanup_handle, {**event, "event": "delete_intent"})
        shutil.rmtree(path)
        OLD.append_jsonl(cleanup_handle, {**event, "event": "delete_complete"})
    root.rmdir()


def _image_session(*, phase: str, entry: Mapping[str, Any],
                   image_index: int, runner, server, run_dir: Path,
                   raw_handle, cleanup_handle, config_sha: str,
                   manifest_sha: str, cleanup_stores: bool) -> list[dict[str, Any]]:
    OLD.assert_gpu_exclusive()
    free_before = shutil.disk_usage(run_dir).free
    if free_before < 45 * 1024**3:
        raise RuntimeError("less than 45 GiB before an image; 30 GiB reserve unsafe")
    image_id = str(entry["image_id"])
    image_path = ROOT / str(entry["image_path"])
    image_sha = OLD.sha256_file(image_path)
    questions = entry["questions"][4:(7 if phase == "smoke" else 10)]
    expected_turns = 3 if phase == "smoke" else 6
    if len(questions) != expected_turns:
        raise AssertionError("frozen GQA question slice changed")
    order = deterministic_method_rotation(METHODS, image_index, SEED)
    store_root = run_dir / "stores" / phase / image_id
    store_root.mkdir(parents=True, exist_ok=False)
    stores = {method: store_root / method for method in METHODS
              if method != "recompute"}
    artifact_root = run_dir / "image_artifacts" / phase / image_id
    artifact_root.mkdir(parents=True, exist_ok=False)
    contexts: dict[str, ImageContext] = {}
    receipts: dict[str, Any] = {}
    selections: dict[str, dict[str, Any]] = {}
    selection_paths: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    saliency_hashes: dict[str, str] = {}
    descriptor_hashes: dict[str, str] = {}
    canonical_reference = None
    canonical_match_ms: dict[str, float] = {}
    activation_ms: dict[str, float] = {}
    peak_rss = psutil.Process().memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    successful = False
    try:
        for turn_id, question in enumerate(questions, 1):
            OLD.assert_gpu_exclusive()
            turn_rows = []
            for position, method in enumerate(order):
                text = str(question["question"])
                if turn_id == 1 or method == "recompute":
                    kind = ("none" if turn_id > 1 or method == "recompute"
                            else "raster" if method == "fullload"
                            else "image_only")
                    capture_keys = bool(
                        turn_id == 1
                        and METHOD_META[method]["variant"] == "contextual")
                    result, diagnostic = _normal_request(
                        runner, server, image_path, text, kind,
                        capture_keys=capture_keys)
                    if turn_id == 1:
                        input_hashes[method] = diagnostic["image_input_sha256"]
                        if method in SELECTIVE:
                            vision = diagnostic["vision_capture"]
                            saliency_hashes[method] = _hash_tensor(
                                vision.result_cpu())
                            if capture_keys:
                                descriptor_hashes[method] = _hash_tensor(
                                    vision.result_keys_cpu())
                        if method in stores:
                            compare_started = time.perf_counter()
                            captured = result["captured_past_key_values"]
                            prefix_len = diagnostic["prefix_len"]
                            if canonical_reference is None:
                                canonical_reference = _capture_prefix_cpu(
                                    captured, prefix_len)
                            else:
                                _compare_prefix_cpu(
                                    canonical_reference, captured, prefix_len)
                            canonical_match_ms[method] = (
                                time.perf_counter()-compare_started)*1e3
                            persisted = _persist_store(
                                runner, result, diagnostic, image_id,
                                stores[method], method, phase)
                            # The full captured decoder prefix is only needed
                            # through persistence. Keeping this local reference
                            # alive would retain its GPU tensors for the entire
                            # image session, even after result is serialized.
                            del captured
                            activation_started = time.perf_counter()
                            context = ImageContext(
                                stores[method], runner.model.device,
                                drop_cache=True, require_v_hidden=False)
                            if method in SELECTIVE:
                                context.validate_contextual_visual_kv_layout()
                            contexts[method] = context
                            activation_ms[method] = (
                                time.perf_counter()-activation_started)*1e3
                            receipts[method] = OLD.store_receipt(
                                persisted, context)
                            if method in SELECTIVE:
                                selection_path = (
                                    artifact_root / f"{method}_selection.json")
                                selections[method] = _selection_artifact(
                                    context, method, selection_path,
                                    phase, image_id)
                                selection_paths[method] = str(
                                    selection_path.relative_to(run_dir))
                    result = QA._json_result(
                        {k: v for k, v in result.items()
                         if k != "captured_past_key_values"},
                        "ours25" if method in SELECTIVE else method, 0)
                else:
                    if "fullload" not in contexts:
                        raise AssertionError("FullLoad context missing before hit")
                    full_bytes = int(
                        contexts["fullload"].meta["bytes_visual_kv"])
                    result, diagnostic = _stored_request(
                        runner, server, contexts[method], text,
                        method, image_id, full_bytes)
                meta = (contexts[method].meta if turn_id > 1
                        and method != "recompute" else None)
                row = _make_row(
                    result=result, diagnostic=diagnostic, meta=meta,
                    method=method, image_id=image_id, image_index=image_index,
                    question=question, turn_id=turn_id, order=order,
                    order_position=position, phase=phase, image_sha=image_sha,
                    config_sha=config_sha, manifest_sha=manifest_sha,
                    selection_artifact_path=selection_paths.get(method))
                if turn_id > 1 and method == "recompute":
                    row.update(OLD.full_pixel_geometry(
                        contexts["fullload"].meta))
                if turn_id > 1:
                    OLD.append_jsonl(raw_handle, row)
                    rows.append(row)
                turn_rows.append(row)
                del result, diagnostic
                if method in contexts:
                    # PrefixCache.new_request() replaces these lists on every
                    # hit. An image has eight open stores; retaining each
                    # previous full-prefix GPU buffer can exhaust a 24 GiB
                    # card on larger GQA images. Release only the finished
                    # request's tensors, outside the measured TTFT/E2E path.
                    contexts[method].cache.k = None
                    contexts[method].cache.v = None
                torch.cuda.empty_cache()
                peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
            OLD.assert_gpu_exclusive()
            if turn_id == 1:
                if (len({r["prompt_sha256"] for r in turn_rows}) != 1
                        or len({r["first_token_id"] for r in turn_rows}) != 1
                        or len({r["prediction"] for r in turn_rows}) != 1):
                    raise AssertionError("normal Turn-1 output differs across arms")
                if len(set(input_hashes.values())) != 1:
                    raise AssertionError("Turn-1 image inputs differ across arms")
                if len(set(saliency_hashes.values())) != 1:
                    raise AssertionError("image-only saliency differs across arms")
                if len(set(descriptor_hashes.values())) != 1:
                    raise AssertionError("vision-key descriptors differ across arms")
                if len(receipts) != 8 or len(selections) != 7:
                    raise AssertionError("normal Turn-1 did not persist all stores")
                full_meta = contexts["fullload"].meta
                for row in turn_rows:
                    row.update(OLD.full_pixel_geometry(full_meta))
                    OLD.append_jsonl(raw_handle, row)
                    rows.append(row)
        del canonical_reference
        for method in SELECTIVE:
            selected = [r for r in rows if r["method_key"] == method
                        and r["turn_id"] > 1]
            if len(selected) != expected_turns-1:
                raise AssertionError("selective hit coverage incomplete")
            ids = selections[method]["selected_original_ids"]
            if any(r["selected_original_ids"] != ids for r in selected):
                raise AssertionError("question changed selected original IDs")
            if len({OLD.canonical_hash(r["selected_chunk_ids_per_layer"])
                    for r in selected}) != 1:
                raise AssertionError("question changed fixed prefix chunks")
        for turn_id in range(2, expected_turns+1):
            group = [r for r in rows if r["turn_id"] == turn_id
                     and r["method_key"] in SELECTIVE]
            if len(group) != len(SELECTIVE):
                raise AssertionError("selective turn coverage incomplete")
            for field in ("N_content", "k_target", "chunk_size",
                          "normal_kv_read_bytes", "ssd_read_bytes",
                          "normal_kv_preads", "ssd_preads"):
                if len({r[field] for r in group}) != 1:
                    raise AssertionError(
                        f"selective arms have unequal {field} for {image_id} T{turn_id}")
        n = selections["d25_c0"]["N_content"]
        k = selections["d25_c0"]["k"]
        d25_ids = set(selections["d25_c0"]["selected_original_ids"])
        diagnostics = {}
        score_vector = selections["d25_c0"]["layout_artifact"][
            "token_score_original"]
        for method in SELECTIVE:
            item = selections[method]
            ids = set(item["selected_original_ids"])
            changed = k-len(ids & d25_ids)
            plan = item["layout_artifact"].get("selection_plan", {})
            reps = plan.get("representative_ids", [])
            targets = plan.get("target_ids", [])
            extra = ids-d25_ids
            diagnostics[method] = {
                "changed_selected_ids_vs_d25": changed,
                "jaccard_vs_d25": len(ids & d25_ids)/len(ids | d25_ids),
                "added_token_saliency": [float(score_vector[i])
                                         for i in sorted(extra)],
                "cluster_sizes": plan.get("cluster_sizes", []),
                "uniform_target_representative_changed_fraction": (
                    sum(a != b for a, b in zip(targets, reps))/len(targets)
                    if targets and len(targets) == len(reps) else None),
            }
        free_with_stores = shutil.disk_usage(run_dir).free
        if free_with_stores < 30 * 1024**3:
            raise RuntimeError("less than 30 GiB while image stores exist")
        image_receipt = {
            "schema_version": SCHEMA, "phase": phase, "image_id": image_id,
            "image_sha256": image_sha, "method_order": list(order),
            "request_count": len(rows),
            "request_ids_sha256": OLD.canonical_hash(
                [r["request_id"] for r in rows]),
            "turn1_output_identical": True,
            "canonical_captured_prefix_bitwise_equal": True,
            "canonical_comparison_ms": canonical_match_ms,
            "image_input_sha256_by_method": input_hashes,
            "saliency_sha256_by_method": saliency_hashes,
            "descriptor_sha256_by_method": descriptor_hashes,
            "selection_artifacts": selection_paths,
            "diagnostics": diagnostics,
            "persistence_by_method": receipts,
            "activation_ms_by_method": activation_ms,
            "N_content": n, "k_target": k,
            "disk_free_before_bytes": free_before,
            "disk_free_with_stores_bytes": free_with_stores,
            "peak_process_rss_bytes": peak_rss,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cleanup_stores": cleanup_stores,
            "validation": "PASS",
        }
        receipt_path = artifact_root / "image_receipt.json"
        OLD.atomic_json(receipt_path, image_receipt)
        successful = True
    finally:
        for context in contexts.values():
            context.close()
    if successful and cleanup_stores:
        _delete_image_stores(
            run_dir, phase, image_id, stores, cleanup_handle)
        free_after_cleanup = shutil.disk_usage(run_dir).free
        if free_after_cleanup < 30 * 1024**3:
            raise RuntimeError("less than 30 GiB after scoped store cleanup")
        OLD.atomic_json(artifact_root / "cleanup_receipt.json", {
            "schema_version": SCHEMA, "phase": phase, "image_id": image_id,
            "deleted_store_methods": list(stores),
            "disk_free_after_bytes": free_after_cleanup,
            "raw_rows_fsynced_before_cleanup": True,
            "selection_artifacts_fsynced_before_cleanup": True,
            "image_receipt_fsynced_before_cleanup": True,
            "reproduction_requires_store_rebuild": True,
        })
    return rows


def _paired_bootstrap(rows: list[dict[str, Any]], new: str, old: str,
                      field: str, *, hit_only: bool = True,
                      scale: float = 1.0) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for row in rows:
        if row["method_key"] not in (new, old):
            continue
        if hit_only and row["turn_id"] == 1:
            continue
        grouped[row["image_id"]][row["method_key"]].append(float(row[field]))
    pairs = []
    for image_id in sorted(grouped):
        arms = grouped[image_id]
        if (set(arms) != {new, old}
                or len(arms[new]) != len(arms[old])
                or not arms[new]):
            raise AssertionError(f"incomplete paired cluster: {image_id}")
        pairs.append((float(np.mean(arms[new])),
                      float(np.mean(arms[old]))))
    if not pairs:
        raise AssertionError("paired bootstrap has no images")
    values = np.asarray(pairs, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(values), size=(10_000, len(values)))
    new_draw = values[draws, 0].mean(axis=1)
    old_draw = values[draws, 1].mean(axis=1)
    delta_draw = (new_draw-old_draw)*scale
    if field != "correct" and np.any(old_draw <= 0):
        raise AssertionError("nonpositive baseline timing in paired ratio")
    ratio_draw = (new_draw/old_draw if field != "correct" else None)
    result = {
        "new": new, "old": old, "field": field,
        "scope": "cache_hits_T2_T6" if hit_only else "all_turns",
        "mean_difference": float((values[:, 0].mean()-values[:, 1].mean())*scale),
        "difference_unit": "percentage_points" if field == "correct" else "ms",
        "difference_ci95": [float(np.quantile(delta_draw, .025)),
                            float(np.quantile(delta_draw, .975))],
        "image_clusters": len(values), "resamples": 10_000, "seed": SEED,
    }
    if ratio_draw is not None:
        result["mean_ratio"] = float(values[:, 0].mean()/values[:, 1].mean())
        result["ratio_ci95"] = [float(np.quantile(ratio_draw, .025)),
                                float(np.quantile(ratio_draw, .975))]
    return result


def _audit_phase(run_dir: Path, phase: str,
                 entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]],
                                                          dict[str, Any]]:
    raw_path = run_dir / f"{phase}_raw.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()
            if line.strip()]
    expected_questions = 3 if phase == "smoke" else 6
    expected = {
        (str(entry["image_id"]), str(q["question_id"]), method)
        for entry in entries
        for q in entry["questions"][4:4+expected_questions]
        for method in METHODS
    }
    observed = [
        (str(r["image_id"]), str(r["question_id"]), r["method_key"])
        for r in rows
    ]
    counts = Counter(observed)
    missing = sorted(expected-set(counts))
    extra = sorted(set(counts)-expected)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    checks = {
        "exact_coverage": len(rows) == len(expected) and not missing
                          and not extra and not duplicates,
        "nine_arms": {r["method_key"] for r in rows} == set(METHODS),
        "successful_status": all(r.get("status") == "ok" for r in rows),
        "positive_ttft": all(float(r["end_to_end_ttft_ms"]) > 0
                             for r in rows),
        "correct_request_path": all(
            r["request_path"] == ("normal_pixel_turn1" if r["turn_id"] == 1
                                  else "normal_pixel_recompute"
                                  if r["method_key"] == "recompute"
                                  else "ssd_visual_kv") for r in rows),
        "no_hit_vision_or_probe": all(
            r["vision_forward_count"] == 0
            and r.get("probe_read_bytes", 0) == 0
            and r.get("query_score_calls", 0) == 0
            for r in rows if r["cache_hit"]),
        "selection_artifacts": True,
        "selection_question_invariant": True,
        "exact_visual_kv_budget": True,
        "identical_selective_io": True,
        "t1_identical": True,
        "image_validation_receipts": True,
        "scoped_cleanup_receipts": True,
    }
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_image[row["image_id"]].append(row)
    for entry in entries:
        image_id = str(entry["image_id"])
        image_rows = by_image[image_id]
        artifact_root = run_dir / "image_artifacts" / phase / image_id
        receipt_path = artifact_root / "image_receipt.json"
        cleanup_path = artifact_root / "cleanup_receipt.json"
        if not receipt_path.is_file():
            checks["image_validation_receipts"] = False
            continue
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("validation") != "PASS"
                or receipt.get("request_count") != len(METHODS)*expected_questions
                or not receipt.get("canonical_captured_prefix_bitwise_equal")):
            checks["image_validation_receipts"] = False
        if receipt.get("cleanup_stores") and not cleanup_path.is_file():
            checks["scoped_cleanup_receipts"] = False
        t1 = [r for r in image_rows if r["turn_id"] == 1]
        if (len(t1) != len(METHODS)
                or len({r["prediction"] for r in t1}) != 1
                or len({r["first_token_id"] for r in t1}) != 1):
            checks["t1_identical"] = False
        for method in SELECTIVE:
            path = artifact_root / f"{method}_selection.json"
            if not path.is_file():
                checks["selection_artifacts"] = False
                continue
            artifact = json.loads(path.read_text())
            n, k = artifact["N_content"], artifact["k"]
            selected = artifact["selected_original_ids"]
            order = artifact["stored_to_original"]
            if (k != (n+3)//4 or len(selected) != k
                    or len(set(selected)) != k
                    or selected != order[:k]
                    or len(set(order)) != len(order)
                    or len(order) != n+len(artifact["structural_original_ids"])
                    or artifact["k_dominant"]+artifact["k_context"] != k):
                checks["exact_visual_kv_budget"] = False
            hits = [r for r in image_rows
                    if r["method_key"] == method and r["turn_id"] > 1]
            if (len(hits) != expected_questions-1
                    or any(r["selected_original_ids"] != selected for r in hits)):
                checks["selection_question_invariant"] = False
            if any(r["attended_content_kv_count"] != k
                   or r["k_target"] != k
                   or abs(r["logical_content_retention"]-k/n) > 1e-12
                   for r in hits):
                checks["exact_visual_kv_budget"] = False
        for turn in range(2, expected_questions+1):
            same_turn = [r for r in image_rows
                         if r["turn_id"] == turn and r["method_key"] in SELECTIVE]
            for field in ("N_content", "k_target", "normal_kv_read_bytes",
                          "ssd_read_bytes", "normal_kv_preads", "ssd_preads"):
                if (len(same_turn) != len(SELECTIVE)
                        or len({r[field] for r in same_turn}) != 1):
                    checks["identical_selective_io"] = False
    audit = {
        "schema_version": SCHEMA, "phase": phase,
        "expected_requests": len(expected), "observed_requests": len(rows),
        "expected_images": len(entries), "question_count_per_image": expected_questions,
        "missing": missing, "extra": extra, "duplicates": duplicates,
        "checks": checks, "passed": all(checks.values()),
        "raw_sha256": OLD.sha256_file(raw_path), "audited_at_utc": _now(),
    }
    OLD.atomic_json(run_dir / f"{phase}_independent_audit.json", audit)
    if not audit["passed"]:
        raise AssertionError(f"{phase} independent audit failed: {checks}")
    return rows, audit


def _mean(rows, field):
    return float(np.mean([float(r[field]) for r in rows]))


def _summary_and_comparisons(rows: list[dict[str, Any]],
                             run_dir: Path, results_dir: Path,
                             phase: str) -> tuple[list[dict[str, Any]],
                                                  list[dict[str, Any]]]:
    receipt_paths = sorted(
        (run_dir / "image_artifacts" / phase).glob("*/image_receipt.json"))
    receipts = [json.loads(path.read_text()) for path in receipt_paths]
    summaries = []
    persistence_rows = []
    for receipt in receipts:
        for method, persisted in receipt["persistence_by_method"].items():
            timing = persisted["timing_ms"]
            persistence_rows.append({
                "phase": phase, "image_id": receipt["image_id"],
                "method_key": method, "persist_ms": timing.get("persist_ms"),
                "descriptor_mapping_ms": timing.get("descriptor_mapping_ms", 0),
                "clustering_ms": timing.get("clustering_ms", 0),
                "token_mapping_ms": timing.get("token_mapping_ms", 0),
                "kv_repack_ms": timing.get("kv_repack_ms", 0),
                "store_write_ms": timing.get("store_write_ms", 0),
                "file_fsync_ms": timing.get("file_fsync_ms", 0),
                "directory_fsync_ms": timing.get("directory_fsync_ms", 0),
                "bytes_total": persisted["bytes"].get("total"),
                "activation_ms": receipt["activation_ms_by_method"][method],
                "canonical_comparison_ms":
                    receipt["canonical_comparison_ms"][method],
            })
    for method in METHODS:
        selected = [r for r in rows if r["method_key"] == method]
        hits = [r for r in selected if r["turn_id"] > 1]
        t1 = [r for r in selected if r["turn_id"] == 1]
        own_persistence = [r for r in persistence_rows
                           if r["method_key"] == method]
        normal = np.asarray([r["normal_kv_read_bytes"] for r in hits])
        total = np.asarray([r["ssd_read_bytes"] for r in hits])
        ttft = np.asarray([r["end_to_end_ttft_ms"] for r in hits])
        ret = [r["logical_content_retention"] for r in hits
               if r["logical_content_retention"] is not None]
        ret_struct = [r["visual_retention_including_structural"]
                      for r in hits
                      if r["visual_retention_including_structural"] is not None]
        row = {
            "method_key": method, "method": METHOD_META[method]["label"],
            "selection_variant": METHOD_META[method]["variant"],
            "alpha": METHOD_META[method]["alpha"],
            "requests": len(selected), "hits": len(hits),
            "all_accuracy": _mean(selected, "correct"),
            "t1_accuracy": _mean(t1, "correct"),
            "hit_accuracy": _mean(hits, "correct"),
            "hit_ttft_mean_ms": float(ttft.mean()),
            "hit_ttft_p50_ms": float(np.quantile(ttft, .5)),
            "hit_ttft_p95_ms": float(np.quantile(ttft, .95)),
            "hit_e2e_mean_ms": _mean(hits, "request_e2e_ms"),
            "all_e2e_mean_ms": _mean(selected, "request_e2e_ms"),
            "t1_ttft_mean_ms": _mean(t1, "end_to_end_ttft_ms"),
            "t1_e2e_mean_ms": _mean(t1, "request_e2e_ms"),
            "actual_content_retention": float(np.mean(ret)) if ret else None,
            "structural_inclusive_retention": (
                float(np.mean(ret_struct)) if ret_struct else None),
            "normal_ssd_mb_per_hit": float(normal.mean()/1e6),
            "total_ssd_mb_per_hit": float(total.mean()/1e6),
            "normal_preads_per_hit": _mean(hits, "normal_kv_preads"),
            "total_preads_per_hit": _mean(hits, "ssd_preads"),
            "persistence_mean_ms": (
                float(np.mean([r["persist_ms"] for r in own_persistence]))
                if own_persistence else None),
            "capture_saliency_reduction_mean_ms": (
                float(np.mean([
                    r["vision_capture_stats"].get("saliency_reduction_ms", 0)
                    for r in t1]))),
            "capture_key_reduction_mean_ms": (
                float(np.mean([
                    r["vision_capture_stats"].get("key_reduction_ms", 0)
                    for r in t1]))),
            "capture_key_hook_submit_mean_ms": (
                float(np.mean([
                    r["vision_capture_stats"].get("key_hook_submit_ms", 0)
                    for r in t1]))),
            "capture_key_materialize_mean_ms": (
                float(np.mean([
                    r["vision_capture_stats"].get("key_materialize_ms", 0)
                    for r in t1]))),
            "k_mean": (_mean(hits, "k_target")
                       if method in SELECTIVE else None),
            "k_dominant_mean": (
                float(np.mean([
                    json.loads((run_dir/"image_artifacts"/phase/r["image_id"]/
                                f"{method}_selection.json").read_text())[
                                    "k_dominant"]
                    for r in t1])) if method in SELECTIVE else None),
            "k_context_mean": (
                float(np.mean([
                    json.loads((run_dir/"image_artifacts"/phase/r["image_id"]/
                                f"{method}_selection.json").read_text())[
                                    "k_context"]
                    for r in t1])) if method in SELECTIVE else None),
        }
        for turn in range(1, 4 if phase == "smoke" else 7):
            row[f"turn{turn}_accuracy"] = _mean(
                [r for r in selected if r["turn_id"] == turn], "correct")
        summaries.append(row)
    comparisons = []
    for new, old, role in (
        ("d20_c5", "d25_c0", "primary"),
        ("d20_c5", "d20_random5", "representative_control"),
        ("d20_c5", "d20_uniform5", "representative_control"),
        ("d22_5_c2_5", "d25_c0", "exploratory_alpha_sweep"),
        ("d17_5_c7_5", "d25_c0", "exploratory_alpha_sweep"),
        ("d15_c10", "d25_c0", "exploratory_alpha_sweep"),
    ):
        quality = _paired_bootstrap(
            rows, new, old, "correct", scale=100)
        timing = _paired_bootstrap(
            rows, new, old, "end_to_end_ttft_ms")
        comparisons.append({
            "role": role, "new": new, "old": old,
            "quality_difference_pp": quality["mean_difference"],
            "quality_ci95_low_pp": quality["difference_ci95"][0],
            "quality_ci95_high_pp": quality["difference_ci95"][1],
            "hit_ttft_difference_ms": timing["mean_difference"],
            "hit_ttft_ci95_low_ms": timing["difference_ci95"][0],
            "hit_ttft_ci95_high_ms": timing["difference_ci95"][1],
            "hit_ttft_ratio": timing["mean_ratio"],
            "hit_ttft_ratio_ci95_low": timing["ratio_ci95"][0],
            "hit_ttft_ratio_ci95_high": timing["ratio_ci95"][1],
            "image_clusters": quality["image_clusters"],
            "resamples": 10_000, "seed": SEED,
        })
    def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
        with path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
            handle.flush()
            os.fsync(handle.fileno())
    write_csv(results_dir / f"{phase}_summary.csv", summaries)
    write_csv(results_dir / f"{phase}_paired_comparisons.csv", comparisons)
    write_csv(results_dir / f"{phase}_persistence.csv", persistence_rows)
    OLD.atomic_json(results_dir / f"{phase}_summary.json", {
        "schema_version": SCHEMA, "phase": phase,
        "methods": summaries, "paired_comparisons": comparisons,
    })
    return summaries, comparisons


def _fmt(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _report_pilot(rows: list[dict[str, Any]],
                  summaries: list[dict[str, Any]],
                  comparisons: list[dict[str, Any]],
                  run_dir: Path, results_dir: Path, audit: Mapping[str, Any]
                  ) -> None:
    by_method = {r["method_key"]: r for r in summaries}
    by_pair = {(r["new"], r["old"]): r for r in comparisons}
    primary = by_pair["d20_c5", "d25_c0"]
    random_control = by_pair["d20_c5", "d20_random5"]
    uniform_control = by_pair["d20_c5", "d20_uniform5"]
    quality_signal = (
        "POSITIVE" if primary["quality_ci95_low_pp"] > 0 else
        "NEGATIVE" if primary["quality_ci95_high_pp"] < 0 else "INCONCLUSIVE")
    representative_signal = (
        "SUPPORTED" if (random_control["quality_ci95_low_pp"] > 0
                        and uniform_control["quality_ci95_low_pp"] > 0) else
        "NOT SUPPORTED" if (random_control["quality_ci95_high_pp"] <= 0
                            and uniform_control["quality_ci95_high_pp"] <= 0)
        else "INCONCLUSIVE")
    sweep = [by_method[m] for m in (
        "d25_c0", "d22_5_c2_5", "d20_c5", "d17_5_c7_5", "d15_c10")]
    best = max(sweep, key=lambda r: (r["hit_accuracy"], -METHODS.index(
        r["method_key"])))
    if best["method_key"] == "d25_c0":
        best_ci = "reference arm (zero difference by definition)"
    else:
        found = by_pair.get((best["method_key"], "d25_c0"))
        best_ci = (
            f"{found['quality_difference_pp']:+.2f} pp, 95% paired CI "
            f"[{found['quality_ci95_low_pp']:+.2f}, "
            f"{found['quality_ci95_high_pp']:+.2f}] pp")
    image_receipts = [
        json.loads(path.read_text())
        for path in sorted((run_dir/"image_artifacts/pilot").glob(
            "*/image_receipt.json"))]
    diag = defaultdict(list)
    for receipt in image_receipts:
        for method, item in receipt["diagnostics"].items():
            diag[(method, "changed")].append(
                item["changed_selected_ids_vs_d25"])
            diag[(method, "jaccard")].append(item["jaccard_vs_d25"])
            if item["cluster_sizes"]:
                diag[(method, "cluster_sizes")].extend(item["cluster_sizes"])
            if item["uniform_target_representative_changed_fraction"] is not None:
                diag[(method, "target_changed")].append(
                    item["uniform_target_representative_changed_fraction"])
            diag[(method, "extra_saliency")].extend(
                item["added_token_saliency"])
    paired_predictions = defaultdict(dict)
    for row in rows:
        if row["turn_id"] > 1 and row["method_key"] in ("d25_c0", "d20_c5"):
            paired_predictions[(row["image_id"], row["question_id"])][
                row["method_key"]] = row
    gain, loss = [], []
    for (image_id, question_id), pair in sorted(paired_predictions.items()):
        a, b = pair["d25_c0"], pair["d20_c5"]
        case = {
            "image_id": image_id, "question_id": question_id,
            "question": a["question"], "gold": a["gold"],
            "D25_prediction": a["prediction"],
            "D20_C5_prediction": b["prediction"],
        }
        if a["correct"] < b["correct"]:
            gain.append(case)
        elif a["correct"] > b["correct"]:
            loss.append(case)
    OLD.atomic_json(results_dir/"prediction_flips.json", {
        "d25_wrong_d20_right_count": len(gain),
        "d25_right_d20_wrong_count": len(loss),
        "d25_wrong_d20_right_examples": gain[:10],
        "d25_right_d20_wrong_examples": loss[:10],
    })
    selective = [by_method[m] for m in SELECTIVE]
    equal_io = (len({round(x["total_ssd_mb_per_hit"], 8)
                     for x in selective}) == 1
                and len({round(x["total_preads_per_hit"], 8)
                         for x in selective}) == 1)
    extra_saliency = diag[("d20_c5", "extra_saliency")]
    extra_median = float(np.median(extra_saliency)) if extra_saliency else None
    extra_p95 = (float(np.quantile(extra_saliency, .95))
                 if extra_saliency else None)
    lines = [
        "# LLaVA-NeXT GQA Dominant + Contextual Representative Visual-KV25",
        "",
        "VisionZip-inspired original-token selection variant. Contextual token "
        "merging from the paper was not reproduced: all selected rows are "
        "unchanged original decoder K/V rows.",
        "",
        "| Method | D/C tokens (image mean) | All accuracy | Hit accuracy | "
        "Δ hit vs D25 (pp) | Hit TTFT (ms) | SSD MB/hit | Persistence ms/image |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        method = item["method_key"]
        if method in SELECTIVE:
            dc = f"{item['k_dominant_mean']:.1f}/{item['k_context_mean']:.1f}"
        else:
            dc = "N/A"
        delta = (
            (item["hit_accuracy"]-by_method["d25_c0"]["hit_accuracy"])*100)
        lines.append(
            f"| {item['method']} | {dc} | {item['all_accuracy']:.3f} | "
            f"{item['hit_accuracy']:.3f} | {delta:+.2f} | "
            f"{item['hit_ttft_mean_ms']:.2f} | "
            f"{item['total_ssd_mb_per_hit']:.3f} | "
            f"{_fmt(item['persistence_mean_ms'])} |")
    lines += [
        "",
        "## Preregistered comparison",
        "",
        f"D20+C5 minus D25 hit accuracy: "
        f"{primary['quality_difference_pp']:+.2f} percentage points "
        f"(95% image-paired bootstrap CI "
        f"[{primary['quality_ci95_low_pp']:+.2f}, "
        f"{primary['quality_ci95_high_pp']:+.2f}]; "
        "10,000 resamples, seed 1234). "
        + ("The preregistered comparison supports an accuracy increase."
           if quality_signal == "POSITIVE" else
           "The preregistered comparison supports an accuracy decrease."
           if quality_signal == "NEGATIVE" else
           "The interval includes zero, so an increase or equivalence is "
           "not established."),
        "",
        f"Best observed sweep arm: {best['method']}, hit accuracy "
        f"{best['hit_accuracy']:.3f}; difference from D25: {best_ci}. "
        "This arm was chosen after seeing the sweep and is exploratory.",
        "",
        f"D20+C5 versus Random: "
        f"{random_control['quality_difference_pp']:+.2f} pp "
        f"[{random_control['quality_ci95_low_pp']:+.2f}, "
        f"{random_control['quality_ci95_high_pp']:+.2f}]. "
        f"Versus Uniform: "
        f"{uniform_control['quality_difference_pp']:+.2f} pp "
        f"[{uniform_control['quality_ci95_low_pp']:+.2f}, "
        f"{uniform_control['quality_ci95_high_pp']:+.2f}]. "
        + ("Representative-specific benefit is supported on both controls."
           if representative_signal == "SUPPORTED" else
           "Representative-specific benefit is not supported on either control."
           if representative_signal == "NOT SUPPORTED" else
           "Representative-specific benefit remains inconclusive."),
        "",
        "## Budget, SSD reads, and timing",
        "",
        "Every selective arm attended exactly ceil(0.25 × N) original "
        "visual-content tokens per layer; structural separators followed "
        "the frozen sidecar policy. "
        + ("All selective arms had identical measured returned SSD bytes "
           "and pread counts on each paired hit."
           if equal_io else
           "Selective arms did not have identical measured SSD bytes/preads; "
           "the independent audit requires image-question equality and any "
           "difference must be investigated."),
        "",
        "| Method | T1 accuracy | T2 | T3 | T4 | T5 | T6 | Hit TTFT p50/p95 "
        "(ms) | Hit E2E mean (ms) | Content / structural retention | "
        "Normal / total preads |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        turns = [item[f"turn{t}_accuracy"] for t in range(2, 7)]
        lines.append(
            f"| {item['method']} | {item['t1_accuracy']:.3f} | "
            + " | ".join(f"{v:.3f}" for v in turns)
            + f" | {item['hit_ttft_p50_ms']:.2f} / "
            f"{item['hit_ttft_p95_ms']:.2f} | "
            f"{item['hit_e2e_mean_ms']:.2f} | "
            f"{_fmt(item['actual_content_retention'], 4)} / "
            f"{_fmt(item['structural_inclusive_retention'], 4)} | "
            f"{item['normal_preads_per_hit']:.1f} / "
            f"{item['total_preads_per_hit']:.1f} |")
    t1_delta = (by_method["d20_c5"]["t1_ttft_mean_ms"]
                -by_method["d25_c0"]["t1_ttft_mean_ms"])
    persist_delta = (by_method["d20_c5"]["persistence_mean_ms"]
                     -by_method["d25_c0"]["persistence_mean_ms"])
    lines += [
        "",
        f"D20+C5 Turn-1 TTFT was {t1_delta:+.2f} ms versus D25; "
        f"one-time persistence was {persist_delta:+.2f} ms/image. "
        "These measured costs include the opt-in key hook and synchronous "
        "descriptor mapping, clustering, repacking, writing, and fsync. "
        "Persistence is separate from cache-hit TTFT.",
        "",
        f"Paired D20+C5 minus D25 hit TTFT: "
        f"{primary['hit_ttft_difference_ms']:+.2f} ms "
        f"[{primary['hit_ttft_ci95_low_ms']:+.2f}, "
        f"{primary['hit_ttft_ci95_high_ms']:+.2f}]; ratio "
        f"{primary['hit_ttft_ratio']:.3f} "
        f"[{primary['hit_ttft_ratio_ci95_low']:.3f}, "
        f"{primary['hit_ttft_ratio_ci95_high']:.3f}].",
        "",
        "## Selection diagnostics",
        "",
        f"D20+C5 changed a mean of "
        f"{np.mean(diag[('d20_c5','changed')]):.1f} selected IDs/image "
        f"versus D25 (mean Jaccard "
        f"{np.mean(diag[('d20_c5','jaccard')]):.3f}). "
        f"Mean cluster size was "
        f"{np.mean(diag[('d20_c5','cluster_sizes')]):.2f}; "
        f"{np.mean(diag[('d20_c5','target_changed')])*100:.1f}% of "
        "uniform targets changed to another original representative. "
        f"D25 wrong→D20+C5 right: {len(gain)} hits; "
        f"D25 right→D20+C5 wrong: {len(loss)} hits. "
        "Examples are saved in prediction_flips.json. Descriptor-space "
        "coverage and VQA accuracy are separate outcomes.",
        "",
        "## Seven requested conclusions",
        "",
        f"1. Primary D20+C5 versus D25: {quality_signal}; "
        f"{primary['quality_difference_pp']:+.2f} pp, paired 95% CI "
        f"[{primary['quality_ci95_low_pp']:+.2f}, "
        f"{primary['quality_ci95_high_pp']:+.2f}].",
        f"2. Best observed ratio: {best['method']}; {best_ci}. "
        "This is exploratory.",
        f"3. Representative-specific benefit versus Random/Uniform: "
        f"{representative_signal}; paired intervals are above.",
        "4. All selective arms kept exactly ceil(0.25 × N) original content "
        "rows with the frozen structural policy.",
        f"5. D20+C5 read {by_method['d20_c5']['total_ssd_mb_per_hit']:.3f} "
        f"MB and {by_method['d20_c5']['total_preads_per_hit']:.1f} preads "
        f"per hit; paired TTFT changed {primary['hit_ttft_difference_ms']:+.2f} "
        f"ms (ratio {primary['hit_ttft_ratio']:.3f}), with intervals above.",
        f"6. D20+C5 Turn-1 TTFT changed {t1_delta:+.2f} ms and persistence "
        f"changed {persist_delta:+.2f} ms/image versus D25. Mean captured "
        f"key reduction {by_method['d20_c5']['capture_key_reduction_mean_ms']:.2f} "
        f"ms and key materialization "
        f"{by_method['d20_c5']['capture_key_materialize_mean_ms']:.2f} ms; "
        "component costs are in persistence.csv and summary.csv.",
        "7. Additional validation: "
        + ("replicate on an independent workload before promoting this method."
           if quality_signal == "POSITIVE" else
           "do not promote this uncertain pilot; revisit only with a larger "
           "independent workload and a clear expected effect."
           if quality_signal == "INCONCLUSIVE" else
           "stop this variant for now given the negative primary signal."),
        "",
        "Additional D20+C5 distributions: extra-token saliency median "
        f"{_fmt(extra_median, 6)}, p95 {_fmt(extra_p95, 6)}; "
        f"cluster-size median {np.median(diag[('d20_c5','cluster_sizes')]):.1f}, "
        f"p95 {np.quantile(diag[('d20_c5','cluster_sizes')], .95):.1f}.",
        "",
        "## Scope and interpretation",
        "",
        "Actual runtime packages: "
        + ", ".join(f"{name} {version}" for name, version in
                    _package_versions().items())
        + f". Local checkpoint revision: {_checkpoint_snapshot()['revision']}.",
        "",
        "The 40 images are the previously used GQA pilot, not an unseen "
        "holdout. ReComp uses a different numerical path from FP16 SSD KV "
        "serving. OS posix_fadvise(DONTNEED) conditions page cache outside "
        "request TTFT and does not prove a cold SSD controller or NAND. "
        "Saved read bytes are returned by os.pread. "
        "After receipt-protected cleanup, stores require rebuilding for "
        "independent SSD replay.",
        "",
        f"Independent audit: {'PASS' if audit['passed'] else 'FAIL'}, "
        f"{audit['observed_requests']}/{audit['expected_requests']} requests.",
        "",
        "Further validation: "
        + ("A new independent workload is warranted before promoting this "
           "variant; the pilot has a positive primary signal."
           if quality_signal == "POSITIVE" else
           "The pilot does not justify replacing the existing main method. "
           "A larger independent workload is warranted only if the unresolved "
           "effect matters operationally."
           if quality_signal == "INCONCLUSIVE" else
           "Stop this variant for now; the primary pilot signal is negative."),
        "",
        "IMPLEMENTATION: PASS",
        "CPU VALIDATION: PASS",
        "GPU CORRECTNESS: PASS",
        "PILOT: COMPLETE",
        f"QUALITY SIGNAL: {quality_signal}",
        f"REPRESENTATIVE-SPECIFIC BENEFIT: {representative_signal}",
        "QWEN / FULL MT-GQA: NOT RUN",
        "",
    ]
    OLD.write_text(results_dir/"REPORT.md", "\n".join(lines))


def _require_source_freeze(run_dir: Path) -> dict[str, Any]:
    freeze_path = run_dir / "source_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    documents = {
        "protocol_sha256": "PROTOCOL.md",
        "config_sha256": "config.json",
        "workload_manifest_sha256": "workload_manifest.json",
        "source_diff_sha256": "source.diff",
        "protected_before_sha256": "protected_before.json",
    }
    for key, relative in documents.items():
        if freeze.get(key) != OLD.sha256_file(run_dir/relative):
            raise ValueError(f"frozen document changed: {relative}")
    frozen = freeze.get("source_sha256")
    if not isinstance(frozen, dict):
        raise ValueError("source freeze lacks source hashes")
    for relative in SOURCE_PATHS:
        if frozen.get(relative) != OLD.sha256_file(ROOT/relative):
            raise ValueError(f"source changed after freeze: {relative}")
    return {
        "path": str(freeze_path), "sha256": OLD.sha256_file(freeze_path),
        "source_sha256": frozen,
        **{key: freeze[key] for key in documents},
    }


def _package_versions() -> dict[str, str]:
    names = ("torch", "transformers", "accelerate", "bitsandbytes",
             "Pillow", "numpy", "psutil")
    return {name: importlib.metadata.version(name) for name in names}


def _checkpoint_snapshot() -> dict[str, str]:
    root = (Path.home()/".cache/huggingface/hub"/
            "models--llava-hf--llava-v1.6-vicuna-7b-hf/snapshots")
    expected = root/"c916e6cdcd760b4cecd1dd4907f84ac649f93b23"
    if not expected.is_dir() or not (expected/"model.safetensors.index.json").is_file():
        raise FileNotFoundError("frozen LLaVA-HF checkpoint snapshot missing")
    return {"revision": expected.name, "snapshot_dir": str(expected),
            "config_sha256": OLD.sha256_file(expected/"config.json")}


def _copy_new(source: Path, destination: Path) -> None:
    with source.open("rb") as read, destination.open("xb") as write:
        shutil.copyfileobj(read, write, length=8 << 20)
        write.flush()
        os.fsync(write.fileno())
    if OLD.sha256_file(source) != OLD.sha256_file(destination):
        raise AssertionError(f"copied artifact hash mismatch: {destination}")


def _require_gate(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text())
    if receipt.get("GPU_CORRECTNESS") != "PASS":
        raise ValueError("new nine-arm GPU correctness receipt is not PASS")
    source = receipt.get("source_sha256")
    if not isinstance(source, dict):
        raise ValueError("GPU gate lacks source SHA256")
    required = ("mmimpress/cvpr25.py", "mmimpress/contextual_kv25.py",
                "mmimpress/piggyback.py", "mmimpress/store.py",
                "mmimpress/serve.py", "mmimpress/model.py",
                "scripts/92_validate_llava_contextual_kv25.py")
    if any(source.get(item) != OLD.sha256_file(ROOT/item)
           for item in required):
        raise ValueError("GPU gate source hashes are stale")
    if (receipt.get("model_id") != MODEL_ID
            or receipt.get("chunk_size") != CHUNK_SIZE
            or receipt.get("index_sha256") != GQA_INDEX_SHA):
        raise ValueError("GPU gate model/chunk/index mismatch")
    if [tuple(map(str, x)) for x in receipt.get("fixed_samples", [])] != FIXED_SAMPLES:
        raise ValueError("GPU gate fixed samples changed")
    if receipt.get("first_step_fp32_logits_tolerance") != {
            "atol": 0.0001, "rtol": 0.0001}:
        raise ValueError("GPU gate changed matched-logit tolerance")
    if [arm.get("id") for arm in receipt.get("arms", [])] != list(SELECTIVE):
        raise ValueError("GPU gate selective arm list changed")
    samples = receipt.get("samples", [])
    if len(samples) != len(FIXED_SAMPLES):
        raise ValueError("GPU gate lacks all five fixed samples")
    for sample, pair in zip(samples, FIXED_SAMPLES):
        if (str(sample.get("image_id")), str(sample.get("question_id"))) != pair:
            raise ValueError("GPU gate sample identity changed")
        arms = sample.get("arms", {})
        if set(arms) != set(SELECTIVE) or any(
                arms[method].get("status") != "PASS" for method in SELECTIVE):
            raise ValueError("GPU gate lacks seven passed selective arms")
        if sample.get("D25_legacy_regression") is None:
            raise ValueError("GPU gate lacks direct D25 regression")
    return {
        "path": str(path.resolve()), "sha256": OLD.sha256_file(path),
        "GPU_CORRECTNESS": "PASS", "fixed_samples": FIXED_SAMPLES,
        "source_sha256": source,
    }


def _run_phase(args, runner, server, run_dir: Path, results_dir: Path,
               entries: list[dict[str, Any]], gate: Mapping[str, Any],
               config_sha: str, workload: Mapping[str, Any],
               gpu_preflight: Mapping[str, Any]):
    phase = args.phase
    raw_path = run_dir / f"{phase}_raw.jsonl"
    if raw_path.exists():
        raise FileExistsError(f"phase raw already exists: {raw_path}")
    if (run_dir / f"{phase}_independent_audit.json").exists():
        raise FileExistsError(f"phase audit already exists: {phase}")
    # The idle-memory threshold applies before this process loads the model.
    # Once loaded, use the PID-aware check to reject only other GPU jobs.
    OLD.assert_gpu_exclusive()
    gpu = dict(gpu_preflight)
    runner_manifest = {
        "schema_version": SCHEMA, "phase": phase, "run_id": run_dir.name,
        "started_at_utc": _now(), "model": MODEL_ID,
        "checkpoint": _checkpoint_snapshot(),
        "package_versions": _package_versions(),
        "model_config": {
            "load_4bit": LOAD_4BIT, "compute_dtype": str(COMPUTE_DTYPE),
            "attention_backend": ATTN_IMPL, "chunk_size": CHUNK_SIZE,
            "max_new_tokens": 16, "ssd_kv_dtype": "float16",
            "greedy_decoding": True,
        },
        "methods": METHOD_META, "seed": SEED,
        "questions_per_image": 3 if phase == "smoke" else 6,
        "images": len(entries), "expected_requests": (
            len(entries)*(3 if phase == "smoke" else 6)*len(METHODS)),
        "gpu_preflight": gpu, "gpu_correctness_gate": gate,
        "frozen_config_sha256": config_sha, "frozen_workload": workload,
        "source_freeze_sha256": OLD.sha256_file(run_dir/"source_freeze.json"),
        "source_sha256": {
            path: OLD.sha256_file(ROOT/path) for path in SOURCE_PATHS},
        "host": platform.node(), "python": sys.version,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "started_after_freeze": True,
        "store_cleanup_policy": (
            "only this run's image-scoped stores after raw/selection/hash/"
            "validation fsync; log paths; rebuild required"),
    }
    manifest_sha = OLD.canonical_hash(runner_manifest)
    OLD.atomic_json(run_dir/f"{phase}_runner_manifest.json", runner_manifest)
    OLD.atomic_json(results_dir/f"{phase}_runner_manifest.json", runner_manifest)
    warmup = QA._warmup(runner, server)
    OLD.atomic_json(run_dir/f"{phase}_warmup.json", warmup)
    rows = []
    cleanup_path = run_dir/f"{phase}_cleanup_log.jsonl"
    with (raw_path.open("x", encoding="utf-8") as raw_handle,
          cleanup_path.open("x", encoding="utf-8") as cleanup_handle):
        for image_index, entry in enumerate(entries):
            rows.extend(_image_session(
                phase=phase, entry=entry, image_index=image_index,
                runner=runner, server=server, run_dir=run_dir,
                raw_handle=raw_handle, cleanup_handle=cleanup_handle,
                config_sha=config_sha, manifest_sha=manifest_sha,
                cleanup_stores=not args.keep_stores))
            print(f"{phase}: {image_index+1}/{len(entries)} images complete",
                  flush=True)
    audited_rows, audit = _audit_phase(run_dir, phase, entries)
    if len(audited_rows) != len(rows):
        raise AssertionError("raw reread count changed")
    summary, comparisons = _summary_and_comparisons(
        audited_rows, run_dir, results_dir, phase)
    _copy_new(raw_path, results_dir/f"{phase}_raw.jsonl")
    _copy_new(run_dir/f"{phase}_independent_audit.json",
              results_dir/f"{phase}_independent_audit.json")
    if phase == "pilot":
        for destination_root in (run_dir, results_dir):
            _copy_new(raw_path, destination_root/"raw.jsonl")
            for name in ("summary.csv", "paired_comparisons.csv",
                         "persistence.csv", "independent_audit.json"):
                source = (results_dir/f"pilot_{name}"
                          if name != "independent_audit.json"
                          else results_dir/"pilot_independent_audit.json")
                _copy_new(source, destination_root/name)
        _report_pilot(audited_rows, summary, comparisons,
                      run_dir, results_dir, audit)
        _copy_new(results_dir/"REPORT.md", run_dir/"REPORT.md")
        OLD.write_text(results_dir/"REPRODUCE.md", (
            "# Reproduce this GQA pilot\n\n"
            f"Frozen run: `{run_dir}`. Use the validated Conda interpreter "
            "`/home/dblab/anaconda3/envs/mllm_ft/bin/python` and the "
            "cached LLaVA-HF checkpoint at revision "
            "`c916e6cdcd760b4cecd1dd4907f84ac649f93b23`. "
            "Run the CPU and GPU correctness gates against these exact "
            "source hashes before rerunning the smoke and pilot. "
            "After stores were cleaned, rebuilding requires each arm's "
            "normal image Turn 1 and fresh store persistence. "
            "Keep GQA questions[4:10], seed 1234, 64-token chunks, NF4, "
            "BF16 compute, FP16 SSD payload, eager attention, and greedy "
            "max_new_tokens=16. The raw rows and selection JSON are "
            "retained for analysis without rebuilding.\n"))
        _copy_new(results_dir/"REPRODUCE.md", run_dir/"REPRODUCE.md")
    OLD.write_text(run_dir/f"{phase}_COMPLETED", "validated\n")
    return {"phase": phase, "rows": len(audited_rows), "audit": audit}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("smoke", "pilot"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--gpu-gate", type=Path, default=None)
    parser.add_argument("--keep-stores", action="store_true",
                        help="retain only this new run's per-image stores")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve(strict=True)
    expected_parent = (ROOT/"runs").resolve()
    if (run_dir.parent != expected_parent
            or not run_dir.name.startswith("llava_contextual_kv25_")):
        raise ValueError("run-dir must be an existing contextual freeze under runs/")
    results_dir = ROOT/"results"/run_dir.name
    if not results_dir.is_dir():
        raise FileNotFoundError("frozen results directory missing")
    for name in ("PROTOCOL.md", "config.json", "workload_manifest.json",
                 "source.diff", "protected_before.json", "source_freeze.json"):
        if not (run_dir/name).is_file():
            raise FileNotFoundError(f"required pre-GPU freeze missing: {name}")
    if args.phase == "pilot":
        smoke_audit = run_dir/"smoke_independent_audit.json"
        if (not smoke_audit.is_file()
                or not json.loads(smoke_audit.read_text()).get("passed")):
            raise ValueError("pilot requires passed same-run smoke audit")
    source_freeze = _require_source_freeze(run_dir)
    gate = _require_gate(
        args.gpu_gate or run_dir/"gpu_validation.json")
    if gate["source_sha256"] != {
            key: source_freeze["source_sha256"][key]
            for key in gate["source_sha256"]}:
        raise ValueError("GPU gate sources disagree with frozen sources")
    gqa, workload = OLD.frozen_gqa()
    entries = gqa[:4] if args.phase == "smoke" else gqa
    if OLD.sha256_file(ROOT/"data/index.json") != GQA_INDEX_SHA:
        raise ValueError("frozen GQA index hash changed")
    config_sha = OLD.sha256_file(run_dir/"config.json")
    gpu_preflight = OLD.gpu_inventory()
    print(f"{args.phase}: loading {MODEL_ID}", flush=True)
    try:
        runner = LlavaRunner().load()
        server = Server(runner, ratio=.25, probe=PROBE_HEADS,
                        max_new_tokens=16)
        _run_phase(args, runner, server, run_dir, results_dir,
                   entries, gate, config_sha, workload, gpu_preflight)
    except BaseException as exc:
        OLD.atomic_json(run_dir/f"{args.phase}_failure.json", {
            "schema_version": SCHEMA, "phase": args.phase,
            "failed_at_utc": _now(), "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
            "preserve_partial_raw_and_stores": True,
        })
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

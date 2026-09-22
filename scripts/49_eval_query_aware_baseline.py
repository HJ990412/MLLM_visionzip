#!/usr/bin/env python3
"""GQA pilot for the SparseVLM-based QA-Select25 SSD baseline.

The experiment is intentionally image-session shaped.  For every image, all
four arms answer the first selected question through normal pixel-based MLLM
inference.  The QA arm piggybacks canonical/raster Visual-KV, probe keys, and
decoder visual input states from that same Turn-1 forward; the Ours arm
piggybacks its existing image-only saliency and physically repacked Visual-KV.
Questions 2..N then compare ReComp, FullLoad, QA-Select25, and Ours25.

Paper TTFT starts before prompt construction/tokenization/initial H2D and ends
after the first output-token decision and CUDA synchronization.  Page-cache
conditioning and invariant store validation occur before that boundary.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import uuid
from contextlib import ExitStack
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mmimpress.config import CHUNK_SIZE, PROBE_HEADS  # noqa: E402
from mmimpress.dataset import METRICS, load_index, question_answers  # noqa: E402
from mmimpress.model import LlavaRunner  # noqa: E402
from mmimpress.piggyback import (  # noqa: E402
    DecoderVisualHiddenCapture,
    VisionForwardCapture,
    deterministic_method_rotation,
    persist_captured_raster_prefix,
    persist_captured_visual_prefix,
    sha256_file,
    stable_json_sha256,
)
from mmimpress.serve import ImageContext, Server  # noqa: E402


SCHEMA_VERSION = "qa-select-gqa-pilot-v2"
EXPECTED_INDEX_SHA256 = (
    "514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a"
)
EXPECTED_FULL_WORKLOAD_SHA256 = (
    "97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66"
)
REFERENCE_EQUIVALENCE_RATE = 0.02
METHOD_KEYS = ("recompute", "fullload", "qa_select25", "ours25")
METHODS = {
    "recompute": {
        "method_id": "recompute", "display_label": "ReComp",
        "paper_label": "ReComp", "retention_ratio": None,
        "importance_source": "none", "query_dependent": False,
        "physical_layout": "none", "repacking": False,
        "online_selection": False,
    },
    "fullload": {
        "method_id": "fullload", "display_label": "FullLoad",
        "paper_label": "FullLoad", "retention_ratio": 1.0,
        "importance_source": "none", "query_dependent": False,
        "physical_layout": "raster", "repacking": False,
        "online_selection": False,
    },
    "qa_select25": {
        "method_id": "qa_select25", "display_label": "QA-Select25",
        "paper_label": "Query-Aware", "retention_ratio": 0.25,
        "importance_source": (
            "SparseVLM-style text-guided visual attention"),
        "query_dependent": True, "physical_layout": "raster",
        "repacking": False, "online_selection": True,
    },
    "ours25": {
        "method_id": "imageonly_prefix25", "display_label": "Ours25",
        "paper_label": "Ours", "retention_ratio": 0.25,
        "importance_source": "image-only Vision Encoder saliency",
        "query_dependent": False,
        "physical_layout": "importance-aware repacked",
        "repacking": True, "online_selection": False,
    },
}
WARMUP_PROMPT = (
    "USER: <image>\nThis is an unmeasured serving warm-up. "
    "Describe the image briefly. ASSISTANT:"
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_tensor(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _hash_tensor_mapping(values: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(values):
        if torch.is_tensor(values[key]):
            digest.update(str(key).encode("utf-8"))
            digest.update(b"\0")
            digest.update(bytes.fromhex(_hash_tensor(values[key])))
    return digest.hexdigest()


def _normalise_image_tensor(value: torch.Tensor) -> torch.Tensor:
    return value[0] if value.ndim == 5 and value.shape[0] == 1 else value


def _image_input_hash(enc: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in ("pixel_values", "image_sizes"):
        digest.update(key.encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_hash_tensor(
            _normalise_image_tensor(enc[key]))))
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False,
                      allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _sync_stream(handle) -> None:
    """Make the current image boundary durable without timing the fsync."""
    handle.flush()
    os.fsync(handle.fileno())


def _causal_prompt_fields(runner, diagnostic, question, question_id):
    """Prove that a GQA request contains its current question and no future one."""
    expected = runner.prompt(question)
    if diagnostic["prompt"] != expected:
        raise AssertionError("request prompt differs from current-question prompt")
    expected_sha = _sha_bytes(expected.encode("utf-8"))
    if diagnostic["prompt_sha256"] != expected_sha:
        raise AssertionError("prompt hash does not match current-question prompt")
    return {
        "expected_prompt_sha256": expected_sha,
        "current_question_sha256": _sha_bytes(question.encode("utf-8")),
        "causal_question_ids": [str(question_id)],
        "past_history_question_ids": [],
        "future_question_ids_used": [],
        "future_questions_in_prompt": 0,
        "causal_prompt_policy": "current_gqa_question_only_no_history",
    }


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


class _TimedCallableProxy:
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


def _combined_processor(runner, image, prompt):
    processor = runner.processor
    tokenizer = processor.tokenizer
    image_processor = processor.image_processor
    timed_tokenizer = _TimedCallableProxy(tokenizer)
    timed_image = _TimedCallableProxy(image_processor)
    processor.tokenizer = timed_tokenizer
    processor.image_processor = timed_image
    started = time.perf_counter()
    try:
        encoded = processor(images=image, text=prompt, return_tensors="pt")
    finally:
        total_ms = (time.perf_counter() - started) * 1e3
        processor.tokenizer = tokenizer
        processor.image_processor = image_processor
    assert timed_tokenizer.calls == 1 and timed_image.calls == 1
    component = timed_tokenizer.elapsed_ms + timed_image.elapsed_ms
    return dict(encoded), {
        "processor_total_ms": float(total_ms),
        "tokenization_ms": float(timed_tokenizer.elapsed_ms),
        "image_preprocess_ms": float(timed_image.elapsed_ms),
        "input_prepare_ms": float(max(0.0, total_ms - component)),
    }


def _suffix_from_tokenized(runner, tokenized):
    ids = tokenized["input_ids"][0]
    positions = (ids == runner.image_token_id).nonzero(as_tuple=True)[0]
    assert positions.numel() == 1, positions.numel()
    return ids[int(positions[0]) + 1:]


class _NoVisionForward:
    def __init__(self, runner):
        self.tower = runner.model.model.vision_tower
        self.calls = 0
        self.handle = None

    def __enter__(self):
        self.handle = self.tower.register_forward_pre_hook(self._count)
        return self

    def _count(self, *_args, **_kwargs):
        self.calls += 1

    def __exit__(self, exc_type, exc, traceback):
        self.handle.remove()
        if exc_type is None:
            assert self.calls == 0, \
                f"stored request invoked vision {self.calls} times"
        return False


def _timing_fields(result, request_started, returned_at, phases):
    core_started = float(result["core_started_at_s"])
    first = float(result["first_token_at_s"])
    model_finished = float(result["model_finished_at_s"])
    pre_core_ms = (core_started - request_started) * 1e3
    core_ttft_ms = (first - core_started) * 1e3
    ttft_ms = (first - request_started) * 1e3
    decode_ms = (model_finished - first) * 1e3
    request_e2e_ms = (returned_at - request_started) * 1e3
    if not request_started <= core_started <= first <= model_finished <= returned_at:
        raise AssertionError("request timestamps are not monotonic")
    return {
        **phases,
        "pre_core_ms": float(pre_core_ms),
        "core_ttft_ms": float(core_ttft_ms),
        "end_to_end_ttft_ms": float(ttft_ms),
        "ttft_ms": float(ttft_ms),
        "decode_ms": float(decode_ms),
        "model_e2e_ms": float((model_finished - request_started) * 1e3),
        "request_e2e_ms": float(request_e2e_ms),
        "request_started_at_s": float(request_started),
        "core_started_at_s": core_started,
        "first_token_at_s": first,
        "model_finished_at_s": model_finished,
        "request_returned_at_s": float(returned_at),
        "ttft_identity_error_ms": float(
            ttft_ms - pre_core_ms - core_ttft_ms),
    }


def _io_fields(result, method_key, full_visual_bytes):
    io = result.get("io") or {
        "bytes": 0, "ms": 0.0, "preads": 0, "chunk_units": 0,
        "per_kind": {},
    }
    detail = io.get("per_kind", {})
    normal_bytes = sum(int(detail.get(kind, {}).get("bytes", 0))
                       for kind in ("k", "v"))
    normal_preads = sum(int(detail.get(kind, {}).get("preads", 0))
                        for kind in ("k", "v"))
    separator_bytes = int(detail.get("sep", {}).get("bytes", 0))
    separator_preads = int(detail.get("sep", {}).get("preads", 0))
    probe_bytes = int(detail.get("probe", {}).get("bytes", 0))
    probe_preads = int(detail.get("probe", {}).get("preads", 0))
    total = int(io.get("bytes", 0))
    return {
        "ssd_read_ms": float(io.get("ms", 0.0)),
        "ssd_read_bytes": total,
        "actual_ssd_mb": float(total / 1e6),
        "ssd_preads": int(io.get("preads", 0)),
        "ssd_read_chunk_units": int(io.get("chunk_units", 0)),
        "normal_kv_read_bytes": normal_bytes,
        "separator_read_bytes": separator_bytes,
        "probe_read_bytes": probe_bytes,
        "normal_kv_preads": normal_preads,
        "separator_preads": separator_preads,
        "probe_preads": probe_preads,
        "actual_ssd_ratio_vs_fullload": (
            float(total) / float(full_visual_bytes)
            if full_visual_bytes and method_key != "recompute" else 0.0),
        "io_detail": detail,
    }


def _json_result(result, method_key, full_visual_bytes):
    excluded = {"captured_past_key_values"}
    value = {key: item for key, item in result.items() if key not in excluded}
    value.update(_io_fields(result, method_key, full_visual_bytes))
    zero_stages = (
        "rater_selection_ms", "query_projection_ms", "probe_h2d_ms",
        "probe_io_ms", "probe_read_pipeline_ms", "query_scoring_ms",
        "topk_ms", "selected_id_d2h_ms", "chunk_planning_ms",
        "chunk_io_ms", "scatter_ms",
        "selector_ms", "online_selector_total_ms",
    )
    for field in zero_stages:
        value.setdefault(field, 0.0)
    if method_key == "ours25":
        # The fixed prefix has only a tiny first-k planning stage; it performs
        # no query/rater/probe scoring.  Surface that stage under the common
        # online-selector names instead of emitting ambiguous nulls.
        planning = float(value.get("select_ms", value["selector_ms"]) or 0.0)
        value["chunk_planning_ms"] = planning
        value["selector_ms"] = planning
        value["online_selector_total_ms"] = planning
    elif method_key == "fullload":
        # The historical FullLoad path exposes raw pread latency, not the
        # wider host read/conversion interval used by QA/Ours.  Keep the
        # non-comparable phase N/A and report ssd_read_ms alongside it.
        value["chunk_io_ms"] = None
        value["chunk_io_timing_semantics"] = (
            "not separately instrumented; use ssd_read_ms for raw pread time")
    decision = float(value["online_selector_total_ms"] or 0.0)
    pipeline = float(
        decision + float(value.get("chunk_io_ms", 0.0) or 0.0)
        + float(value.get("scatter_ms", 0.0) or 0.0))
    value["online_loading_pipeline_component_sum_ms"] = pipeline
    value["online_qa_pipeline_total_ms"] = pipeline
    return value


def _run_pixels(runner, server, image, question, capture_kind="none"):
    capture_cache = capture_kind in {"qa", "ours"}
    vision = VisionForwardCapture(
        runner, capture_saliency=(capture_kind == "ours"))
    hidden_capture = None
    with vision:
        torch.cuda.synchronize()
        request_started = time.perf_counter()
        started = time.perf_counter()
        prompt = runner.prompt(question)
        prompt_ms = (time.perf_counter() - started) * 1e3
        enc_cpu, processor = _combined_processor(runner, image, prompt)
        v_start, v_num = runner.visual_span(enc_cpu["input_ids"])
        if capture_kind == "qa":
            hidden_capture = DecoderVisualHiddenCapture(
                runner, v_start, v_num)
        stack = ExitStack()
        if hidden_capture is not None:
            stack.enter_context(hidden_capture)
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
        "tokenization_ms": processor["tokenization_ms"],
        "image_preprocess_ms": processor["image_preprocess_ms"],
        "input_prepare_ms": processor["input_prepare_ms"],
        "input_h2d_ms": float(h2d_ms),
        "processor_total_ms": processor["processor_total_ms"],
    }
    result.update(_timing_fields(result, request_started, returned, phases))
    result.update({
        "vision_ms": float(vision.stats()["vision_ms"]),
        "vision_forward_count": int(vision.call_count),
        "vision_capture_stats": vision.stats(),
        "visual_hidden_capture_stats": (
            hidden_capture.stats() if hidden_capture is not None else None),
        "page_cache_conditioning_method": "not_applicable_pixels",
        "page_cache_conditioning_ms": 0.0,
        "page_cache_conditioning_excluded_from_ttft": True,
    })
    placeholder = runner.processor.tokenizer(prompt, return_tensors="pt")
    suffix = _suffix_from_tokenized(runner, placeholder)
    diagnostics = {
        "prompt": prompt,
        "prompt_sha256": _sha_bytes(prompt.encode("utf-8")),
        "input_tensors_sha256": _hash_tensor_mapping(enc_cpu),
        "input_ids_sha256": _hash_tensor(enc_cpu["input_ids"]),
        "image_input_sha256": _image_input_hash(enc_cpu),
        "suffix_ids_sha256": _hash_tensor(suffix),
        "enc_cpu": enc_cpu,
        "vision_capture": vision,
        "hidden_capture": hidden_capture,
    }
    del enc_device
    return result, diagnostics


def _run_stored(runner, server, ctx, question, method_key, image_id,
                full_visual_bytes):
    with _NoVisionForward(runner) as guard:
        conditioned_at = time.perf_counter()
        ctx.reader.drop_all()
        conditioned_done = time.perf_counter()
        torch.cuda.synchronize()
        request_started = time.perf_counter()
        started = time.perf_counter()
        prompt = runner.prompt(question)
        prompt_ms = (time.perf_counter() - started) * 1e3
        started = time.perf_counter()
        tokenized = runner.processor.tokenizer(prompt, return_tensors="pt")
        token_ms = (time.perf_counter() - started) * 1e3
        started = time.perf_counter()
        suffix_cpu = _suffix_from_tokenized(runner, tokenized)
        prepare_ms = (time.perf_counter() - started) * 1e3
        started = time.perf_counter()
        suffix_device = suffix_cpu.to(runner.model.device)
        torch.cuda.synchronize()
        h2d_ms = (time.perf_counter() - started) * 1e3
        if method_key == "fullload":
            result = server.request(
                ctx, mode="fullload", cold=False, suffix_ids=suffix_device)
        elif method_key == "qa_select25":
            result = server.request_qa_select(
                ctx, cold=False, suffix_ids=suffix_device)
        elif method_key == "ours25":
            result = server.request_cvpr25(
                ctx, static=None, budget=0.25, mode="prefix",
                sep_policy="sidecar", cold=False, image_id=image_id,
                suffix_ids=suffix_device,
                expected_prefix_layout="visionzip_image_only")
        else:
            raise ValueError(method_key)
        returned = time.perf_counter()
    phases = {
        "prompt_build_ms": float(prompt_ms),
        "tokenization_ms": float(token_ms),
        "image_preprocess_ms": 0.0,
        "input_prepare_ms": float(prepare_ms),
        "input_h2d_ms": float(h2d_ms),
        "processor_total_ms": None,
    }
    result.update(_timing_fields(result, request_started, returned, phases))
    result.update({
        "vision_forward_count": int(guard.calls),
        "page_cache_conditioning_started_at_s": float(conditioned_at),
        "page_cache_conditioning_finished_at_s": float(conditioned_done),
        "page_cache_conditioning_ms": float(
            (conditioned_done - conditioned_at) * 1e3),
        "page_cache_conditioning_method": "posix_fadvise_DONTNEED",
        "page_cache_conditioning_excluded_from_ttft": True,
    })
    diagnostics = {
        "prompt": prompt,
        "prompt_sha256": _sha_bytes(prompt.encode("utf-8")),
        "suffix_ids_sha256": _hash_tensor(suffix_cpu),
    }
    return _json_result(result, method_key, full_visual_bytes), diagnostics


def _warmup(runner, server):
    height, width = 480, 640
    yy, xx = np.indices((height, width), dtype=np.uint16)
    pixels = np.stack(((3 * xx + yy) % 256, (xx + 5 * yy) % 256,
                       (7 * xx + 11 * yy) % 256), axis=-1).astype(np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    enc = runner.processor(images=image, text=WARMUP_PROMPT,
                           return_tensors="pt")
    v_start, v_num = runner.visual_span(enc["input_ids"])
    vision = VisionForwardCapture(runner, capture_saliency=True)
    hidden = DecoderVisualHiddenCapture(runner, v_start, v_num)
    with vision, hidden:
        result = server.recompute(
            runner.to_device(enc), return_past_key_values=True)
    cache = result.pop("captured_past_key_values")
    del cache, enc, image
    torch.cuda.synchronize()
    return {
        "enabled": True,
        "fixture": "deterministic_in_memory_rgb_pattern_640x480",
        "fixture_sha256": _sha_bytes(pixels.tobytes()),
        "excluded_from_all_metrics": True,
        "vision_forward_count": vision.call_count,
        "saliency_capture_count": vision.saliency_call_count,
        "visual_hidden_capture_count": hidden.visual_capture_count,
    }


def _workload(index_path, skip, questions, max_images):
    entries = load_index(index_path)
    if len(entries) != 40:
        raise ValueError(f"frozen GQA index has {len(entries)} images, not 40")
    selected = entries[:max_images]
    rows = [(str(entry["image_id"]), str(q["question_id"]))
            for entry in selected
            for q in entry["questions"][skip:skip + questions]]
    if any(len(entry["questions"][skip:skip + questions]) != questions
           for entry in selected):
        raise ValueError("selected image lacks the fixed question slice")
    full_rows = [(str(entry["image_id"]), str(q["question_id"]))
                 for entry in entries
                 for q in entry["questions"][skip:skip + questions]]
    return entries, selected, {
        "index_sha256": sha256_file(index_path),
        "full_workload_sha256": _sha_bytes("\n".join(
            f"{image}\t{question}" for image, question in full_rows).encode()),
        "selected_workload_sha256": _sha_bytes("\n".join(
            f"{image}\t{question}" for image, question in rows).encode()),
        "full_images": len(entries), "full_questions": len(full_rows),
        "selected_images": len(selected), "selected_questions": len(rows),
    }


def _selection_set(row, field):
    values = row.get(field) or []
    return {(layer, int(token)) for layer, items in enumerate(values)
            for token in items}


def _layerwise_jaccard(left, right, field):
    left_layers = left.get(field) or []
    right_layers = right.get(field) or []
    if len(left_layers) != len(right_layers):
        raise AssertionError(
            f"selection layer-count mismatch for {field}: "
            f"{len(left_layers)} != {len(right_layers)}")
    return [
        _jaccard({int(item) for item in left_items},
                 {int(item) for item in right_items})
        for left_items, right_items in zip(left_layers, right_layers)
    ]


def _jaccard(left, right):
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def _selection_analysis(rows):
    qa = [row for row in rows
          if row["method_key"] == "qa_select25" and row["turn_id"] > 1]
    by_image = {}
    for row in qa:
        by_image.setdefault(row["image_id"], []).append(row)
    pairs = []
    for image_id, image_rows in sorted(by_image.items()):
        image_rows.sort(key=lambda row: row["turn_id"])
        for left, right in combinations(image_rows, 2):
            token_layer = _layerwise_jaccard(
                left, right, "selected_token_ids_per_layer")
            chunk_layer = _layerwise_jaccard(
                left, right, "selected_chunk_ids_per_layer")
            token_global = _jaccard(
                _selection_set(left, "selected_token_ids_per_layer"),
                _selection_set(right, "selected_token_ids_per_layer"))
            chunk_global = _jaccard(
                _selection_set(left, "selected_chunk_ids_per_layer"),
                _selection_set(right, "selected_chunk_ids_per_layer"))
            token_j = float(np.mean(token_layer))
            chunk_j = float(np.mean(chunk_layer))
            pairs.append({
                "image_id": image_id,
                "left_turn": left["turn_id"],
                "right_turn": right["turn_id"],
                "left_question_id": left["question_id"],
                "right_question_id": right["question_id"],
                "token_jaccard": token_j,
                "chunk_jaccard": chunk_j,
                "global_layer_token_jaccard": token_global,
                "global_layer_chunk_jaccard": chunk_global,
                "token_jaccard_per_layer": token_layer,
                "chunk_jaccard_per_layer": chunk_layer,
                "min_layer_token_jaccard": float(min(token_layer)),
                "max_layer_token_jaccard": float(max(token_layer)),
                "consecutive": right["turn_id"] == left["turn_id"] + 1,
            })
    token_values = [row["token_jaccard"] for row in pairs]
    chunk_values = [row["chunk_jaccard"] for row in pairs]
    consecutive = [row for row in pairs if row["consecutive"]]
    requests = []
    for row in sorted(qa, key=lambda value: (
            value["image_id"], value["turn_id"], value["question_id"])):
        tokens = row["selected_token_ids_per_layer"]
        chunks = row["selected_chunk_ids_per_layer"]
        requests.append({
            "image_id": row["image_id"],
            "turn_id": int(row["turn_id"]),
            "question_id": row["question_id"],
            "question_sha256": row["current_question_sha256"],
            "selected_token_ids_per_layer": tokens,
            "selected_chunk_ids_per_layer": chunks,
            "selected_tokens_sha256": stable_json_sha256(tokens),
            "selected_chunks_sha256": stable_json_sha256(chunks),
        })
    return {
        "scope": (
            "cache-hit turns 2..N only; Turn 1 is deliberately normal "
            "pixel-based inference for all methods"),
        "n_query_requests": len(qa),
        "n_images": len(by_image),
        "requests": requests,
        "pairs": pairs,
        "n_pairs": len(pairs),
        "mean_pairwise_token_jaccard": (
            float(np.mean(token_values)) if token_values else None),
        "mean_pairwise_chunk_jaccard": (
            float(np.mean(chunk_values)) if chunk_values else None),
        "identical_selection_rate": (
            float(np.mean([value == 1.0 for value in token_values]))
            if token_values else None),
        "different_selection_pairs": int(sum(
            value < 1.0 for value in token_values)),
        "n_consecutive_pairs": len(consecutive),
        "mean_consecutive_token_jaccard": (
            float(np.mean([row["token_jaccard"] for row in consecutive]))
            if consecutive else None),
        "mean_consecutive_chunk_jaccard": (
            float(np.mean([row["chunk_jaccard"] for row in consecutive]))
            if consecutive else None),
    }


def _mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _summaries(rows, full_visual_bytes_mean):
    output = {}
    for method_key in METHOD_KEYS:
        all_rows = [row for row in rows if row["method_key"] == method_key]
        hits = [row for row in all_rows if row["turn_id"] > 1]
        source = hits or all_rows
        ttft = [float(row["end_to_end_ttft_ms"]) for row in source]
        output[method_key] = {
            **METHODS[method_key],
            "n_requests": len(all_rows),
            "n_cache_hit_requests": len(hits),
            "accuracy_all_turns": _mean(all_rows, "correct"),
            "accuracy_cache_hits": _mean(hits, "correct"),
            "ttft_cache_hit_mean_ms": float(np.mean(ttft)),
            "ttft_cache_hit_p50_ms": float(np.percentile(ttft, 50)),
            "ttft_cache_hit_p95_ms": float(np.percentile(ttft, 95)),
            "turn1_ttft_mean_ms": _mean(
                [row for row in all_rows if row["turn_id"] == 1],
                "end_to_end_ttft_ms"),
            "actual_ssd_mb_per_cache_hit": _mean(source, "actual_ssd_mb"),
            "actual_ssd_ratio_vs_fullload": _mean(
                source, "actual_ssd_ratio_vs_fullload"),
            "ssd_preads_per_cache_hit": _mean(source, "ssd_preads"),
            "ssd_read_latency_ms": _mean(source, "ssd_read_ms"),
            "touched_chunk_fraction": _mean(
                source, "touched_chunk_fraction"),
            "contiguous_runs_per_layer": _mean(
                source, "contiguous_runs_per_layer_mean"),
            "mean_contiguous_run_length": _mean(
                source, "mean_contiguous_run_length"),
            "selector_ms": _mean(source, "selector_ms"),
            "online_selector_total_ms": _mean(
                source, "online_selector_total_ms"),
            "online_qa_pipeline_total_ms": _mean(
                source, "online_qa_pipeline_total_ms"),
            "rater_selection_ms": _mean(source, "rater_selection_ms"),
            "query_projection_ms": _mean(source, "query_projection_ms"),
            "probe_io_ms": _mean(source, "probe_io_ms"),
            "query_scoring_ms": _mean(source, "query_scoring_ms"),
            "topk_ms": _mean(source, "topk_ms"),
            "selected_id_d2h_ms": _mean(source, "selected_id_d2h_ms"),
            "chunk_planning_ms": _mean(source, "chunk_planning_ms"),
            "chunk_io_ms": _mean(source, "chunk_io_ms"),
            "scatter_ms": _mean(source, "scatter_ms"),
            "prefill_ms": _mean(source, "prefill_ms"),
            "probe_io_mb": (_mean(source, "probe_read_bytes") or 0.0) / 1e6,
            "logical_selected_token_ratio": (
                _mean(source, "logical_selected_token_ratio")
                if _mean(source, "logical_selected_token_ratio") is not None
                else _mean(source, "logical_kv_ratio")),
            "logical_kv_ratio": _mean(source, "logical_kv_ratio"),
            "full_visual_kv_mb_mean": full_visual_bytes_mean / 1e6,
        }
    return output


def _comparison(summaries):
    qa, ours = summaries["qa_select25"], summaries["ours25"]
    return {
        "qa_minus_ours_accuracy_all_turns_pp": (
            qa["accuracy_all_turns"] - ours["accuracy_all_turns"]) * 100,
        "qa_minus_ours_accuracy_cache_hits_pp": (
            qa["accuracy_cache_hits"] - ours["accuracy_cache_hits"]) * 100,
        "qa_minus_ours_ttft_cache_hit_ms": (
            qa["ttft_cache_hit_mean_ms"] - ours["ttft_cache_hit_mean_ms"]),
        "qa_over_ours_ttft_ratio": (
            qa["ttft_cache_hit_mean_ms"] / ours["ttft_cache_hit_mean_ms"]),
    }


def _reference_consistency(rows):
    path = ROOT / "runs/image_only_repack_budget_sweep_with_recomp/main_20_50/results.json"
    if not path.is_file():
        return {"available": False, "path": str(path)}
    reference = json.loads(path.read_text())["rows"]
    by_key = {(str(row["image_id"]), str(row["question_id"])): row
              for row in reference}
    mapping = {
        "recompute": "recompute",
        "fullload": "fullload",
        "ours25": "visionzip_repack_prefix@25",
    }
    result = {"available": True, "path": str(path),
              "sha256": sha256_file(path), "methods": {}}
    for method, ref_method in mapping.items():
        compared = equal = 0
        current_correct = []
        reference_correct = []
        for row in rows:
            if row["method_key"] != method:
                continue
            # Turn 1 deliberately differs for cache arms: it is normal pixel
            # inference in this experiment.  Compare cache hits only.
            if method != "recompute" and row["turn_id"] == 1:
                continue
            ref = by_key.get((row["image_id"], row["question_id"]))
            if ref is None:
                continue
            compared += 1
            reference_result = ref[ref_method]
            equal += row["prediction"] == reference_result["answer"]
            current_correct.append(float(row["correct"]))
            reference_correct.append(float(reference_result["acc"]))
        current_accuracy = (
            float(np.mean(current_correct)) if current_correct else None)
        reference_accuracy = (
            float(np.mean(reference_correct)) if reference_correct else None)
        result["methods"][method] = {
            "compared": compared, "equal_predictions": equal,
            "agreement": float(equal / compared) if compared else None,
            "current_accuracy": current_accuracy,
            "reference_accuracy": reference_accuracy,
            "accuracy_gap_pp": (
                (current_accuracy - reference_accuracy) * 100.0
                if current_accuracy is not None else None),
        }
    return result


def _reference_consistency_passes(reference):
    """Run-to-run equivalence gate for the three pre-existing methods.

    ReComp is expected to reproduce exactly because it does not depend on a
    rebuilt SSD store.  FullLoad/Ours are separately persisted GPU caches; a
    very small number of greedy boundary ties can change exact strings across
    runs without changing the method or aggregate quality.  For those arms we
    require at least 98% exact agreement and an accuracy gap bounded by the
    same discrete 2% allowance (with one sample allowed for small smokes).
    """
    if not reference.get("available"):
        return True
    methods = reference["methods"]
    recompute = methods["recompute"]
    if int(recompute["equal_predictions"]) != int(recompute["compared"]):
        return False
    for key in ("fullload", "ours25"):
        item = methods[key]
        compared = int(item["compared"])
        if compared <= 0:
            return False
        tolerance = max(1, math.ceil(REFERENCE_EQUIVALENCE_RATE * compared))
        if compared - int(item["equal_predictions"]) > tolerance:
            return False
        max_accuracy_gap_pp = 100.0 * tolerance / compared
        if abs(float(item["accuracy_gap_pp"])) > max_accuracy_gap_pp + 1e-9:
            return False
    return True


def _validate(rows, selection, metas, persistence, workload, reference):
    expected = workload["selected_questions"] * len(METHOD_KEYS)
    checks = {}
    checks["row_count"] = len(rows) == expected
    keys = [(row["image_id"], row["question_id"], row["method_key"])
            for row in rows]
    checks["unique_rows"] = len(keys) == len(set(keys))
    checks["method_set"] = set(row["method_key"] for row in rows) == set(METHOD_KEYS)
    checks["stable_method_metadata"] = all(
        all(row.get(key) == value for key, value in METHODS[row["method_key"]].items())
        for row in rows)
    checks["all_ttft_finite_positive"] = all(
        math.isfinite(float(row["end_to_end_ttft_ms"]))
        and float(row["end_to_end_ttft_ms"]) > 0 for row in rows)
    turn1_ok = True
    for image_id in sorted({row["image_id"] for row in rows}):
        group = [row for row in rows
                 if row["image_id"] == image_id and row["turn_id"] == 1]
        turn1_ok &= len(group) == len(METHOD_KEYS)
        for field in ("prompt_sha256", "input_tensors_sha256",
                      "suffix_ids_sha256", "prediction", "first_token_id"):
            turn1_ok &= len({row.get(field) for row in group}) == 1
        turn1_ok &= all(row.get("vision_forward_count") == 1 for row in group)
        turn1_ok &= all(row.get("request_path") == "normal_pixel_turn1"
                        for row in group)
    checks["turn1_prompt_prediction_first_token_agreement"] = bool(turn1_ok)
    hits = [row for row in rows if row["turn_id"] > 1]
    qa = [row for row in hits if row["method_key"] == "qa_select25"]
    ours = [row for row in hits if row["method_key"] == "ours25"]
    full = [row for row in hits if row["method_key"] == "fullload"]
    recomp = [row for row in hits if row["method_key"] == "recompute"]

    def expected_normal_bytes(row, meta):
        row_bytes = int(meta["num_heads"]) * int(meta["head_dim"]) * 2
        total_rows = 0
        for chunks in row["selected_chunk_ids_per_layer"]:
            for chunk in chunks:
                start = int(chunk) * int(meta["chunk_size"])
                total_rows += max(0, min(
                    int(meta["chunk_size"]), int(meta["v_token_num"]) - start))
        return total_rows * row_bytes * 2  # K and V

    def expected_sparse_preads(row, *, probe):
        runs = sum(int(value) for value in row["contiguous_runs_per_layer"])
        layers = int(row.get("expected_layers", len(
            row["selected_chunk_ids_per_layer"])))
        return 1 + 2 * runs + (layers if probe else 0)
    checks["cache_hits_no_vision"] = all(
        row.get("vision_forward_count") == 0 for row in qa + ours + full)
    checks["qa_query_scoring_called"] = bool(qa) and all(
        int(row.get("query_score_calls", 0))
        == int(row.get("expected_layers", -1)) > 0 for row in qa)
    checks["ours_query_scoring_zero"] = bool(ours) and all(
        int(row.get("query_score_calls", 0)) == 0 for row in ours)
    checks["qa_fixed_budget_no_fallback"] = all(
        row.get("fallback_rate") == 0.0
        and row.get("adaptive_ratio") is False
        and int(row.get("static_score_calls", -1)) == 0
        and int(row.get("diversity_calls", -1)) == 0
        and abs(float(row["logical_selected_token_ratio"]) - 0.25) < 0.002
        and all(len(tokens) == int(row["k_keep"])
                for tokens in row["selected_token_ids_per_layer"])
        for row in qa)
    checks["qa_raster_no_repack"] = all(
        meta["qa"]["physical_layout"] == "raster"
        and meta["qa"].get("reordered") is False
        and "order" not in meta["qa"] for meta in metas.values())
    checks["ours_image_only_repacked"] = all(
        meta["ours"]["physical_layout"] == "visionzip_image_only"
        and meta["ours"].get("reordered") is True
        for meta in metas.values())
    checks["selected_chunks_exact_for_selected_tokens"] = all(
        all(sorted({int(token) // int(row["chunk_size"])
                    for token in tokens}) == chunks
            for tokens, chunks in zip(
                row["selected_token_ids_per_layer"],
                row["selected_chunk_ids_per_layer"])) for row in qa)
    checks["actual_bytes_accounting"] = all(
        int(row["ssd_read_bytes"]) == int(row["normal_kv_read_bytes"])
        + int(row["separator_read_bytes"]) + int(row["probe_read_bytes"])
        for row in qa + ours + full)
    checks["independent_sparse_bytes_and_preads"] = all(
        int(row["normal_kv_read_bytes"])
        == expected_normal_bytes(row, metas[row["image_id"]][side])
        and int(row["separator_read_bytes"])
        == int(metas[row["image_id"]][side]["bytes_separator_sidecar"])
        and int(row["probe_read_bytes"])
        == (int(metas[row["image_id"]][side]["bytes_probe_sidecar"])
            if side == "qa" else 0)
        and int(row["ssd_preads"])
        == expected_sparse_preads(row, probe=(side == "qa"))
        for side, group in (("qa", qa), ("ours", ours)) for row in group)
    checks["independent_full_load_bytes"] = bool(full) and all(
        int(row["ssd_read_bytes"])
        == int(metas[row["image_id"]]["qa"]["bytes_visual_kv"])
        and int(row["ssd_preads"])
        == 2 * int(metas[row["image_id"]]["qa"]["num_layers"])
        for row in full)
    checks["full_load_unchanged_contract"] = bool(full) and all(
        int(row.get("query_score_calls", 0)) == 0
        and math.isclose(float(row["actual_ssd_ratio_vs_fullload"]), 1.0,
                         rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(float(row["logical_kv_ratio"]), 1.0,
                         rel_tol=0.0, abs_tol=1e-12)
        for row in full)
    checks["ours_first_k_sequential_contract"] = bool(ours) and all(
        all(chunks == list(range(len(chunks)))
            for chunks in row["selected_chunk_ids_per_layer"])
        and math.isclose(float(row.get("selector_ms", 0.0)),
                         float(row.get("select_ms", 0.0)),
                         rel_tol=0.0, abs_tol=1e-9)
        for row in ours)
    # QA layer hooks execute inside the measured prefill interval.  Rater
    # selection and separator preparation precede _decode but follow the same
    # core-start timestamp, so absolute boundaries prove inclusion without
    # double-counting hook stages on top of prefill_ms.
    checks["qa_selector_inside_ttft"] = all(
        float(row["core_ttft_ms"]) + 1e-3 >= (
            float(row["predecode_setup_host_wall_ms"])
            + float(row["prefill_ms"]))
        and float(row["prefill_ms"]) + 1e-3
        >= float(row["selector_hook_host_wall_ms"])
        and float(row["core_ttft_ms"]) + 1e-3
        >= float(row["online_selector_host_wall_proxy_ms"])
        and float(row.get("rater_selection_ms", 0.0)) > 0.0
        and float(row.get("query_scoring_ms", 0.0)) > 0.0
        for row in qa)
    checks["query_dependent_selection_observed"] = (
        selection["different_selection_pairs"] > 0)
    checks["ours_same_prefix_per_image"] = all(
        len({stable_json_sha256(row["selected_chunk_ids_per_layer"])
             for row in ours if row["image_id"] == image_id}) == 1
        for image_id in {row["image_id"] for row in ours})
    checks["causal_prompt_current_question_only"] = all(
        row.get("prompt_sha256") == row.get("expected_prompt_sha256")
        and row.get("current_question_sha256")
        == _sha_bytes(row["question"].encode("utf-8"))
        and row.get("causal_question_ids") == [str(row["question_id"])]
        and row.get("past_history_question_ids") == []
        and row.get("future_question_ids_used") == []
        for row in rows)
    checks["future_query_leakage_zero"] = all(
        int(row.get("future_questions_in_prompt", -1)) == 0
        and row.get("future_question_ids_used") == [] for row in rows)
    checks["persistence_integrity"] = all(
        item[side]["integrity"]["ok"]
        for item in persistence.values() for side in ("qa", "ours"))
    checks["recomp_zero_ssd"] = all(
        int(row["ssd_read_bytes"]) == 0
        and int(row.get("query_score_calls", 0)) == 0 for row in recomp)
    checks["reference_comparison_reported"] = (
        not reference.get("available") or all(
            int(item.get("compared", 0)) > 0
            for item in reference.get("methods", {}).values()))
    checks["existing_method_reference_consistency"] = (
        _reference_consistency_passes(reference))
    passed = all(checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "passed": passed,
        "checks": checks,
        "selection": selection,
        "reference_consistency": reference,
        "limitations": [
            "GQA questions are treated as independent cache-hit turns; no "
            "conversation history exists in this pilot.",
            "Turn 1 deliberately has no QA selection set: all four arms use "
            "normal pixel inference. Query-overlap evidence therefore covers "
            "the five measured cache-hit questions (turns 2..6) per image.",
            "QA scoring averages the configured probe heads rather than all "
            "decoder heads; this is an SSD adaptation, not exact SparseVLM.",
            "v_hidden.pt is loaded once into CPU RAM when an image context is "
            "opened, outside per-request TTFT and SSD accounting; its per-"
            "request H2D and rater compute remain inside TTFT.",
            "Pairwise token Jaccard is the macro mean of per-decoder-layer "
            "Jaccards; the global layer-token Jaccard is also retained for "
            "every question pair.",
            "Buffered pread with POSIX_FADV_DONTNEED does not flush an SSD "
            "controller cache.",
        ],
    }


def _write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (json.dumps(value, ensure_ascii=False)
                                    if isinstance(value, (dict, list)) else value)
                             for key, value in row.items()})
        handle.flush()
        os.fsync(handle.fileno())


def _summary_csv(path, summaries):
    fields = [
        "method_key", "method_id", "display_label", "paper_label",
        "accuracy_all_turns",
        "accuracy_cache_hits", "ttft_cache_hit_mean_ms",
        "ttft_cache_hit_p50_ms", "ttft_cache_hit_p95_ms",
        "retention_ratio", "logical_selected_token_ratio",
        "actual_ssd_mb_per_cache_hit", "actual_ssd_ratio_vs_fullload",
        "selector_ms", "online_qa_pipeline_total_ms",
        "touched_chunk_fraction", "ssd_preads_per_cache_hit",
        "ssd_read_latency_ms", "contiguous_runs_per_layer",
    ]
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in METHOD_KEYS:
            writer.writerow({"method_key": key, **{
                field: summaries[key].get(field) for field in fields[1:]}})
        handle.flush()
        os.fsync(handle.fileno())


def _readme(config, summaries, comparison, validation):
    lines = [
        "# QA-Select25 GQA pilot", "",
        ("SparseVLM-based query-aware SSD baseline versus fixed image-only "
         "repacked Prefix25."), "",
        "## Main results", "",
        ("Cache-hit TTFT includes prompt construction, tokenization, initial "
         "H2D, all online selection/I/O/scatter, prefill, and synchronized "
         "first-token availability."), "",
        "| Method | Accuracy (all) | Accuracy (hits) | Cache-hit TTFT | "
        "Nominal KV | Actual SSD MB | SSD ratio | Selector ms | Touched chunks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in METHOD_KEYS:
        row = summaries[key]
        def percent(value):
            return "—" if value is None else f"{100*value:.2f}%"
        def number(value, digits=2):
            return "—" if value is None else f"{value:.{digits}f}"
        lines.append(
            f"| {row['display_label']} | {percent(row['accuracy_all_turns'])} | "
            f"{percent(row['accuracy_cache_hits'])} | "
            f"{number(row['ttft_cache_hit_mean_ms'])} ms | "
            f"{percent(row['retention_ratio'])} | "
            f"{number(row['actual_ssd_mb_per_cache_hit'])} | "
            f"{percent(row['actual_ssd_ratio_vs_fullload'])} | "
            f"{number(row['selector_ms'])} | "
            f"{percent(row['touched_chunk_fraction'])} |")
    lines.extend([
        "", "## I/O locality", "",
        "| Method | Logical kept tokens | Probe MB | Preads | SSD read ms | "
        "Runs/layer | Mean run length |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for key in METHOD_KEYS:
        row = summaries[key]
        lines.append(
            f"| {row['display_label']} | "
            f"{percent(row['logical_selected_token_ratio'])} | "
            f"{number(row['probe_io_mb'])} | "
            f"{number(row['ssd_preads_per_cache_hit'])} | "
            f"{number(row['ssd_read_latency_ms'])} | "
            f"{number(row['contiguous_runs_per_layer'])} | "
            f"{number(row['mean_contiguous_run_length'])} |")
    lines.extend([
        "", "## Online phase costs (cache hits)", "",
        "| Method | Raters | Q projection | Probe I/O | Q scoring | Top-k | "
        "ID D2H | Chunk plan | Chunk I/O | Scatter | Prefill | TTFT |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for key in METHOD_KEYS:
        row = summaries[key]
        lines.append(
            f"| {row['display_label']} | {number(row['rater_selection_ms'])} | "
            f"{number(row['query_projection_ms'])} | "
            f"{number(row['probe_io_ms'])} | "
            f"{number(row['query_scoring_ms'])} | "
            f"{number(row['topk_ms'])} | "
            f"{number(row['selected_id_d2h_ms'])} | "
            f"{number(row['chunk_planning_ms'])} | "
            f"{number(row['chunk_io_ms'])} | "
            f"{number(row['scatter_ms'])} | "
            f"{number(row['prefill_ms'])} | "
            f"{number(row['ttft_cache_hit_mean_ms'])} |")
    comp = comparison
    select = validation["selection"]
    lines.extend([
        "", "## Direct comparison", "",
        (f"- QA-Select25 − Ours25 accuracy (all turns): "
         f"{comp['qa_minus_ours_accuracy_all_turns_pp']:+.2f} pp"),
        (f"- QA-Select25 − Ours25 accuracy (cache hits): "
         f"{comp['qa_minus_ours_accuracy_cache_hits_pp']:+.2f} pp"),
        (f"- QA-Select25 − Ours25 cache-hit TTFT: "
         f"{comp['qa_minus_ours_ttft_cache_hit_ms']:+.2f} ms "
         f"({comp['qa_over_ours_ttft_ratio']:.2f}x)"),
        (f"- QA token-selection mean pairwise Jaccard: "
         f"{select['mean_pairwise_token_jaccard']:.4f}"),
        (f"- QA consecutive-query mean token Jaccard: "
         f"{select['mean_consecutive_token_jaccard']:.4f}"),
        (f"- QA identical-selection rate: "
         f"{100*select['identical_selection_rate']:.2f}%"),
        (f"- QA selection evidence: {select['n_query_requests']} requests, "
         f"{select['n_pairs']} within-image query pairs; exact per-layer "
         "token/chunk IDs are in `selection.json` and `raw.jsonl`."),
        "", "## Existing-method consistency", "",
    ])
    reference = validation["reference_consistency"]
    if reference.get("available"):
        lines.append(
            "- Gate: ReComp exact; rebuilt FullLoad/Ours stores allow at most "
            "max(1, ceil(2% × n)) prediction differences and the same "
            "discrete accuracy-gap bound.")
        for key in ("recompute", "fullload", "ours25"):
            item = reference["methods"][key]
            lines.append(
                f"- {summaries[key]['display_label']}: "
                f"{item['equal_predictions']}/{item['compared']} exact "
                f"predictions; accuracy gap {item['accuracy_gap_pp']:+.2f} pp")
    else:
        lines.append("- Frozen prior reference unavailable.")
    lines.extend(["", "## Validation", ""])
    for key, passed in validation["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{key}`")
    lines.extend([
        "", "## Configuration", "",
        f"- Images/questions: {config['n_images']} / {config['n_questions']}",
        f"- Index SHA256: `{config['index_sha256']}`",
        f"- Workload SHA256: `{config['selected_workload_sha256']}`",
        "- QA layout: canonical/original raster; no repacking",
        "- Ours layout: image-only importance-aware physical repacking",
        "- Both nominal budgets: 25%",
        "- QA fallback/adaptive ratio/recycling/merging/diversity: disabled",
        "", "## Limitations", "",
    ])
    lines.extend(f"- {item}" for item in validation["limitations"])
    lines.extend([
        "", ("QUERY-AWARE BASELINE VALIDATED: "
             + ("YES" if validation["passed"] else "NO")), "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=ROOT / "data/index.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument(
        "--results-dir", type=Path,
        help="optional fresh directory for the compact paper-facing bundle")
    parser.add_argument("--max-images", type=int, choices=(5, 10, 40),
                        default=5)
    parser.add_argument("--skip", type=int, default=4)
    parser.add_argument("--questions", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--expected-index-sha256",
                        default=EXPECTED_INDEX_SHA256)
    parser.add_argument("--expected-workload-sha256",
                        default=EXPECTED_FULL_WORKLOAD_SHA256)
    parser.add_argument("--expected-images", type=int, default=40,
                        help="expected image count in the full frozen index")
    parser.add_argument("--expected-questions", type=int, default=240,
                        help="expected question count in the full frozen slice")
    args = parser.parse_args()

    index_path = args.index.resolve()
    run_dir = args.run_dir.resolve()
    store_dir = args.store_dir.resolve()
    results_dir = (args.results_dir.resolve()
                   if args.results_dir is not None else None)
    if args.skip != 4 or args.questions != 6:
        raise ValueError("the frozen pilot requires --skip 4 --questions 6")
    _, entries, workload = _workload(
        index_path, args.skip, args.questions, args.max_images)
    if workload["index_sha256"] != args.expected_index_sha256:
        raise ValueError("frozen index SHA256 mismatch")
    if workload["full_workload_sha256"] != args.expected_workload_sha256:
        raise ValueError("frozen full workload SHA256 mismatch")
    if workload["full_images"] != args.expected_images:
        raise ValueError("frozen full image count mismatch")
    if workload["full_questions"] != args.expected_questions:
        raise ValueError("frozen full question count mismatch")
    output_dirs = [run_dir, store_dir]
    if results_dir is not None:
        output_dirs.append(results_dir)
    if any(os.path.lexists(path) for path in output_dirs):
        raise FileExistsError("all output directories must be new")
    for left, right in combinations(output_dirs, 2):
        if _paths_overlap(left, right):
            raise ValueError(f"output directories overlap: {left} and {right}")
    if any(path.parent.is_symlink() for path in output_dirs):
        raise ValueError("output parents may not be symlinks")
    run_dir.mkdir(parents=True)
    store_dir.mkdir(parents=True)
    if results_dir is not None:
        results_dir.mkdir(parents=True)
    (store_dir / "raster").mkdir()
    (store_dir / "image_only").mkdir()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    runner = LlavaRunner().load()
    server = Server(runner, ratio=0.25, probe=PROBE_HEADS,
                    max_new_tokens=args.max_new_tokens)
    warmup = _warmup(runner, server)

    config = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "dataset": "gqa",
        "index": str(index_path),
        **workload,
        "n_images": workload["selected_images"],
        "n_questions": workload["selected_questions"],
        "skip": args.skip, "questions_per_image": args.questions,
        "seed": args.seed, "method_keys": list(METHOD_KEYS),
        "methods": METHODS, "model": runner.model_id,
        "chunk_size": CHUNK_SIZE, "probe_heads": PROBE_HEADS,
        "max_new_tokens": args.max_new_tokens,
        "history_policy": "none_independent_gqa_questions",
        "future_question_leakage": 0,
        "existing_method_reference_equivalence": (
            "ReComp exact; FullLoad/Ours <= max(1, ceil(2% * n)) exact-"
            "prediction mismatches and accuracy gap within the same discrete "
            "2% allowance"),
        "turn1_policy": (
            "all arms use normal pixel inference; QA and Ours persist only "
            "metadata/KV piggybacked from their same answer-producing forward"),
        "cache_hit_policy": (
            "ReComp pixels; FullLoad canonical full SSD; QA per-query fixed "
            "Top-25% logical tokens on raster SSD; Ours fixed first-k on "
            "image-only repacked SSD"),
        "ttft_definition": (
            "timer before prompt construction/tokenization -> initial H2D -> "
            "selection/I/O/scatter or vision -> prefill -> first output token "
            "-> CUDA synchronize"),
        "cold_cache": True,
        "page_cache_conditioning_in_ttft": False,
        "qa_probe_head_policy": (
            f"mean SparseVLM-style scores over fixed first {PROBE_HEADS} of "
            "32 heads; SSD adaptation, not full SparseVLM reproduction"),
        "qa_visual_hidden_policy": (
            "v_hidden.pt is loaded once at ImageContext construction outside "
            "per-request TTFT/I/O and retained in CPU RAM; its per-request "
            "GPU transfer and rater compute are inside TTFT"),
        "selector_ms_definition": (
            "rater CUDA-event time plus observed per-layer selection host "
            "wall through selected-ID D2H and chunk planning; selected chunk "
            "I/O/scatter are separate and selector runs inside prefill_ms"),
        "prefill_ms_definition": (
            "model-call interval through first token; for QA it includes all "
            "per-layer selector hooks, selected chunk I/O, and scatter"),
        "warmup": warmup,
        "run_dir": str(run_dir), "store_dir": str(store_dir),
        "results_dir": str(results_dir) if results_dir is not None else None,
        "started_at_unix": time.time(),
    }
    _atomic_json(run_dir / "config.json", config)

    rows = []
    persistence = {}
    metas = {}
    full_visual_sizes = []
    raw_path = run_dir / "raw.jsonl"
    persistence_path = run_dir / "persistence.jsonl"
    run_started = time.perf_counter()
    with raw_path.open("x", encoding="utf-8") as raw_handle, \
            persistence_path.open("x", encoding="utf-8") as persist_handle:
        for image_index, entry in enumerate(entries):
            image_id = str(entry["image_id"])
            image_path = ROOT / entry["image_path"]
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            questions = entry["questions"][
                args.skip:args.skip + args.questions]
            order = deterministic_method_rotation(
                METHOD_KEYS, image_index, args.seed)
            qa_ctx = ours_ctx = None
            image_persistence = {}
            image_metas = {}

            # Turn 1: every method is the same normal multimodal request.
            q = questions[0]
            for position, method_key in enumerate(order):
                capture_kind = ("qa" if method_key == "qa_select25" else
                                "ours" if method_key == "ours25" else "none")
                result, diagnostic = _run_pixels(
                    runner, server, image, q["question"], capture_kind)
                captured_cache = result.pop("captured_past_key_values", None)
                record_result = _json_result(result, method_key, 0)
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "dataset": "gqa", "image_id": image_id,
                    "question_id": str(q["question_id"]),
                    "turn_id": 1, "question": q["question"],
                    "gold": question_answers(q),
                    "method_key": method_key,
                    **METHODS[method_key],
                    "method_order": list(order),
                    "method_order_position": position,
                    "request_path": "normal_pixel_turn1",
                    "prediction": result["answer"],
                    "correct": METRICS["gqa"](
                        result["answer"], question_answers(q)),
                    "first_token_id": int(result["first_token_id"]),
                    "prompt_sha256": diagnostic["prompt_sha256"],
                    "input_tensors_sha256": diagnostic[
                        "input_tensors_sha256"],
                    "input_ids_sha256": diagnostic["input_ids_sha256"],
                    "image_input_sha256": diagnostic[
                        "image_input_sha256"],
                    "suffix_ids_sha256": diagnostic["suffix_ids_sha256"],
                    **_causal_prompt_fields(
                        runner, diagnostic, q["question"], q["question_id"]),
                    **record_result,
                }
                # Method metadata is the schema authority.  Serving internals
                # may expose lower-level aliases, but must never rewrite the
                # paper-facing method identity or layout across turns.
                record.update(METHODS[method_key])
                rows.append(record)
                raw_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                raw_handle.flush()

                if method_key == "qa_select25":
                    assert captured_cache is not None
                    hidden_capture = diagnostic["hidden_capture"]
                    persisted = persist_captured_raster_prefix(
                        runner, captured_cache,
                        diagnostic["enc_cpu"]["input_ids"],
                        diagnostic["enc_cpu"]["image_sizes"][0],
                        hidden_capture.result_cpu(),
                        store_dir / "raster" / image_id,
                        image_id=image_id, model_id=runner.model_id,
                        chunk_size=CHUNK_SIZE, probe_heads=PROBE_HEADS,
                        hidden_capture_stats=hidden_capture,
                        image_input_sha256=diagnostic[
                            "image_input_sha256"],
                        extra_metadata={
                            "dataset": "gqa", "source_turn_id": 1,
                        })
                    del captured_cache
                    torch.cuda.synchronize()
                    qa_ctx = ImageContext(
                        store_dir / "raster" / image_id,
                        runner.model.device, require_v_hidden=True)
                    qa_ctx.validate_qa_select_layout()
                    image_persistence["qa"] = persisted
                    image_metas["qa"] = dict(qa_ctx.meta)
                elif method_key == "ours25":
                    assert captured_cache is not None
                    vision_capture = diagnostic["vision_capture"]
                    persisted = persist_captured_visual_prefix(
                        runner, captured_cache,
                        diagnostic["enc_cpu"]["input_ids"],
                        diagnostic["enc_cpu"]["image_sizes"][0],
                        vision_capture.result_cpu(),
                        store_dir / "image_only" / image_id,
                        image_id=image_id, model_id=runner.model_id,
                        chunk_size=CHUNK_SIZE,
                        image_input_sha256=diagnostic[
                            "image_input_sha256"],
                        capture_stats=vision_capture,
                        extra_metadata={
                            "dataset": "gqa", "source_turn_id": 1,
                            "future_questions_used_for_layout": 0,
                        })
                    del captured_cache
                    torch.cuda.synchronize()
                    ours_ctx = ImageContext(
                        store_dir / "image_only" / image_id,
                        runner.model.device, require_v_hidden=False)
                    ours_ctx.validate_prefix_layout("visionzip_image_only")
                    image_persistence["ours"] = persisted
                    image_metas["ours"] = dict(ours_ctx.meta)
                else:
                    assert captured_cache is None
                del diagnostic["enc_cpu"]

            assert qa_ctx is not None and ours_ctx is not None
            assert set(image_persistence) == {"qa", "ours"}
            assert set(image_metas) == {"qa", "ours"}
            persistence[image_id] = image_persistence
            metas[image_id] = image_metas
            full_visual_bytes = int(qa_ctx.meta["bytes_visual_kv"])
            assert full_visual_bytes == int(ours_ctx.meta["bytes_visual_kv"])
            full_visual_sizes.append(full_visual_bytes)
            persist_handle.write(json.dumps({
                "image_id": image_id, **image_persistence,
            }, ensure_ascii=False) + "\n")
            persist_handle.flush()

            # Cache-hit turns.  GQA has no conversational history, so each
            # prompt causally contains exactly its current question.
            for turn_id, q in enumerate(questions[1:], 2):
                for position, method_key in enumerate(order):
                    if method_key == "recompute":
                        result, diagnostic = _run_pixels(
                            runner, server, image, q["question"], "none")
                        result = _json_result(
                            result, method_key, full_visual_bytes)
                    else:
                        context = (ours_ctx if method_key == "ours25"
                                   else qa_ctx)
                        result, diagnostic = _run_stored(
                            runner, server, context, q["question"],
                            method_key, image_id, full_visual_bytes)
                    record = {
                        "schema_version": SCHEMA_VERSION,
                        "dataset": "gqa", "image_id": image_id,
                        "question_id": str(q["question_id"]),
                        "turn_id": turn_id, "question": q["question"],
                        "gold": question_answers(q),
                        "method_key": method_key,
                        **METHODS[method_key],
                        "method_order": list(order),
                        "method_order_position": position,
                        "request_path": ("normal_pixel_recompute" if
                                         method_key == "recompute" else
                                         "ssd_cache_hit"),
                        "prediction": result["answer"],
                        "correct": METRICS["gqa"](
                            result["answer"], question_answers(q)),
                        "first_token_id": int(result["first_token_id"]),
                        "prompt_sha256": diagnostic["prompt_sha256"],
                        "suffix_ids_sha256": diagnostic[
                            "suffix_ids_sha256"],
                        "input_tensors_sha256": diagnostic.get(
                            "input_tensors_sha256"),
                        **_causal_prompt_fields(
                            runner, diagnostic, q["question"],
                            q["question_id"]),
                        "chunk_size": int(qa_ctx.meta["chunk_size"]),
                        **result,
                    }
                    record.update(METHODS[method_key])
                    rows.append(record)
                    raw_handle.write(
                        json.dumps(record, ensure_ascii=False) + "\n")
                    raw_handle.flush()

            _sync_stream(raw_handle)
            _sync_stream(persist_handle)
            qa_ctx.close()
            ours_ctx.close()
            del qa_ctx, ours_ctx, image
            torch.cuda.empty_cache()
            elapsed = time.perf_counter() - run_started
            print(f"[{image_index + 1}/{len(entries)}] {image_id} "
                  f"rows={len(rows)} elapsed={elapsed:.1f}s", flush=True)

    selection = _selection_analysis(rows)
    reference = _reference_consistency(rows)
    summaries = _summaries(rows, float(np.mean(full_visual_sizes)))
    comparison = _comparison(summaries)
    validation = _validate(
        rows, selection, metas, persistence, workload, reference)
    config["status"] = "complete" if validation["passed"] else "failed_validation"
    config["finished_at_unix"] = time.time()
    config["wall_seconds"] = time.perf_counter() - run_started
    _atomic_json(run_dir / "config.json", config)
    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "config": config, "per_method": summaries,
        "comparison": comparison,
        "selection": selection, "reference_consistency": reference,
    }
    _atomic_json(run_dir / "summary.json", summary_payload)
    _atomic_json(run_dir / "validation.json", validation)
    _atomic_json(run_dir / "selection.json", selection)
    _write_csv(run_dir / "per_request.csv", rows)
    _summary_csv(run_dir / "summary.csv", summaries)
    readme = _readme(config, summaries, comparison, validation)
    _atomic_text(run_dir / "README.md", readme)
    if results_dir is not None:
        _atomic_json(results_dir / "summary.json", summary_payload)
        _atomic_json(results_dir / "validation.json", validation)
        _atomic_json(results_dir / "selection.json", selection)
        _summary_csv(results_dir / "summary.csv", summaries)
        _atomic_text(results_dir / "README.md", readme)
        _atomic_json(results_dir / "run_artifacts.json", {
            "schema_version": SCHEMA_VERSION,
            "run_dir": str(run_dir),
            "store_dir": str(store_dir),
            "files_sha256": {
                name: sha256_file(run_dir / name) for name in (
                    "config.json", "raw.jsonl", "persistence.jsonl",
                    "per_request.csv", "summary.json", "validation.json",
                    "selection.json", "summary.csv", "README.md")
            },
        })
    print(json.dumps({
        "run_dir": str(run_dir), "passed": validation["passed"],
        "results_dir": (str(results_dir) if results_dir is not None else None),
        "comparison": comparison,
        "selection": {key: value for key, value in selection.items()
                      if key not in {"pairs", "requests"}},
    }, indent=2), flush=True)
    if not validation["passed"]:
        failed = [key for key, value in validation["checks"].items()
                  if not value]
        raise RuntimeError("validation failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()

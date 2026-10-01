#!/usr/bin/env python3
"""Gated, same-run, eight-arm LLaVA-NeXT spatial Visual-KV25 experiment.

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
SCHEMA = "llava-spatial-uniform-kv25-v1"
LAYOUT = "visionzip_contextual_original_v1"
SPATIAL_LAYOUT = "visionzip_spatial_original_v1"
SEED = 1234
SOURCE_PATHS = (
    "mmimpress/cvpr25.py", "mmimpress/contextual_kv25.py",
    "mmimpress/spatial_kv25.py",
    "mmimpress/piggyback.py", "mmimpress/store.py",
    "mmimpress/serve.py", "mmimpress/model.py",
    "scripts/49_eval_query_aware_baseline.py", "scripts/89_eval_llava_kv25.py",
    "scripts/92_validate_llava_contextual_kv25.py",
    "scripts/93_eval_llava_contextual_kv25.py",
    "scripts/94_validate_llava_spatial_kv25.py",
    "scripts/95_eval_llava_spatial_kv25.py",
)
METHODS = (
    "recompute", "fullload", "d25_c0", "d20_context5",
    "d20_index_uniform5", "d20_spatial_uniform5", "d20_random5",
    "d21_1_c3_9_reference",
)
METHOD_META = {
    "recompute": {"label": "ReComp", "variant": None, "alpha": None},
    "fullload": {"label": "FullLoad", "variant": None, "alpha": None},
    "d25_c0": {"label": "D25+C0", "variant": "dominant", "alpha": 0.0},
    "d20_context5": {"label": "D20+Context5", "variant": "contextual", "alpha": 0.2},
    "d20_index_uniform5": {"label": "D20+IndexUniform5", "variant": "uniform", "alpha": 0.2},
    "d20_spatial_uniform5": {"label": "D20+SpatialUniform5", "variant": "spatial_uniform", "alpha": 0.2},
    "d20_random5": {"label": "D20+Random5", "variant": "random", "alpha": 0.2},
    "d21_1_c3_9_reference": {"label": "D21.1+C3.9 [54:10 reference]", "variant": "contextual", "alpha": 10/64},
}
SELECTIVE = tuple(m for m in METHODS if m not in ("recompute", "fullload"))
COMPARISONS = (
    ("d20_spatial_uniform5", "d20_index_uniform5", "primary"),
    ("d20_spatial_uniform5", "d25_c0", "practical_baseline"),
    ("d20_context5", "d20_index_uniform5", "secondary"),
    ("d20_spatial_uniform5", "d20_context5", "secondary"),
    ("d20_spatial_uniform5", "d20_random5", "secondary"),
    ("d21_1_c3_9_reference", "d25_c0", "secondary"),
    ("d21_1_c3_9_reference", "d20_context5", "secondary"),
)
DIAGNOSTIC_GRID = 8
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


class _OSReadTrace:
    """Record Python os.pread calls without issuing any extra payload read."""

    def __init__(self, store_dir: Path):
        self.store_dir = store_dir.resolve()
        self.rows: list[dict[str, Any]] = []
        self._original = None

    def __enter__(self):
        if self._original is not None:
            raise AssertionError("pread trace entered twice")
        self._original = os.pread
        os.pread = self._pread
        return self

    def _pread(self, fd: int, length: int, offset: int) -> bytes:
        path = Path(os.readlink(f"/proc/self/fd/{fd}")).resolve()
        try:
            relative = path.relative_to(self.store_dir).as_posix()
        except ValueError as exc:
            raise AssertionError(
                f"unplanned Python pread outside active image store: {path}"
            ) from exc
        data = self._original(fd, length, offset)
        self.rows.append({
            "relative_path": relative, "offset_bytes": int(offset),
            "requested_bytes": int(length),
            "returned_bytes": len(data),
        })
        return data

    def __exit__(self, exc_type, exc, traceback):
        os.pread = self._original
        self._original = None


def _assert_os_read_trace(result: Mapping[str, Any], method: str) -> None:
    trace = result.get("os_pread_trace")
    if not isinstance(trace, list):
        raise AssertionError("SSD hit has no independent os.pread trace")
    actual_bytes = sum(int(item["returned_bytes"]) for item in trace)
    if (len(trace) != int(result["ssd_preads"])
            or actual_bytes != int(result["ssd_read_bytes"])
            or (method in SELECTIVE and actual_bytes !=
                int(result["total_actual_pread_bytes"]))
            or any(item["returned_bytes"] != item["requested_bytes"]
                   for item in trace)):
        raise AssertionError("os.pread trace disagrees with measured SSD reads")
    if method not in SELECTIVE:
        return
    planned = Counter((
        f"layer_{int(span['layer']):02d}/{span['kind']}.bin",
        int(span["offset_bytes"]), int(span["length_bytes"]))
        for span in result["planned_normal_read_spans"])
    actual = Counter((
        item["relative_path"], int(item["offset_bytes"]),
        int(item["returned_bytes"]))
        for item in trace
        if item["relative_path"] != "sep_kv.bin")
    separator = [item for item in trace
                 if item["relative_path"] == "sep_kv.bin"]
    if (planned != actual
            or len(separator) != int(result["separator_preads"])
            or len(separator) != 1
            or separator[0]["offset_bytes"] != 0
            or separator[0]["returned_bytes"] !=
                int(result["separator_read_bytes"])
            or sum(item["returned_bytes"] for item in trace
                   if item["relative_path"] != "sep_kv.bin") !=
                int(result["normal_kv_read_bytes"])
            or int(result.get("probe_read_bytes", 0)) != 0):
        raise AssertionError("planned and OS-returned Visual-KV spans differ")


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
        with _OSReadTrace(ctx.dir) as os_trace:
            if method == "fullload":
                result = server.request(ctx, mode="fullload", cold=False,
                                        suffix_ids=suffix_device)
            else:
                result = server.request_cvpr25(
                    ctx, static=None, budget=0.25, mode="prefix",
                    budget_unit="visual_kv", sep_policy="sidecar", cold=False,
                    seed=SEED, image_id=image_id, suffix_ids=suffix_device,
                    expected_prefix_layout=(
                        SPATIAL_LAYOUT if method == "d20_spatial_uniform5"
                        else LAYOUT))
        returned = time.perf_counter()
    result["os_pread_trace"] = os_trace.rows
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
    # QA._json_result adds ssd_preads and ssd_read_bytes from the server
    # I/O counter. Validate the OS trace only after those fields exist.
    result = QA._json_result(result, "fullload" if method == "fullload"
                             else "ours25", full_visual_bytes)
    _assert_os_read_trace(result, method)
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
    c = (0 if method == "d25_c0" else
         (10 * k) // 64 if method == "d21_1_c3_9_reference"
         else k // 5)
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
        "layout_policy": (SPATIAL_LAYOUT if method == "d20_spatial_uniform5" else LAYOUT),
        "alpha": METHOD_META[method]["alpha"],
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
    plan = layout_json.get("selection_plan")
    if not isinstance(plan, dict):
        raise AssertionError("selection plan missing from stored layout")
    if (int(plan["k"]) != k or int(plan["k_dominant"]) != k-c
            or int(plan["k_context"]) != c
            or [int(x) for x in plan["selected_original_ids"]] != selected
            or [int(x) for x in plan["stored_to_original"]] != order
            or plan["layout_policy"] != payload["layout_policy"]):
        raise AssertionError("stored selection plan differs from physical layout")
    if (phase == "smoke" and image_id == FIXED_SAMPLES[0][0]
            and method == "d20_spatial_uniform5"):
        coordinate_table = destination.parent/"fixed_coordinate_table.csv"
        records = plan["spatial"]["coordinates"]["records"]
        fields = (
            "original_visual_id", "content", "branch", "row", "column",
            "grid_height", "grid_width", "x", "y",
            "source_subimage_index", "source_tile_row",
            "source_tile_column", "source_patch_row",
            "source_patch_column",
        )
        with coordinate_table.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for original, record in enumerate(records):
                writer.writerow({
                    **(record or {}),
                    "original_visual_id": original,
                    "content": record is not None,
                })
            handle.flush()
            os.fsync(handle.fileno())
        payload["fixed_real_image_coordinate_table"] = {
            "path": str(coordinate_table),
            "sha256": OLD.sha256_file(coordinate_table),
            "image_id_prefixed_before_analysis": FIXED_SAMPLES[0][0],
            "coordinate_space": "model_input_patch_grid",
            "structural_ids_have_no_coordinate": True,
        }
    OLD.atomic_json(destination, payload)
    return payload


def _spatial_relationships(selections: Mapping[str, Mapping[str, Any]]
                           ) -> dict[str, Any]:
    plans = {method: artifact["layout_artifact"]["selection_plan"]
             for method, artifact in selections.items()}
    d20 = ("d20_context5", "d20_index_uniform5",
           "d20_spatial_uniform5", "d20_random5")
    if any(method not in plans for method in d20):
        raise AssertionError("four D20 selection plans missing")
    dominant = [int(x) for x in plans[d20[0]]["dominant_ids"]]
    if any([int(x) for x in plans[method]["dominant_ids"]] != dominant
           for method in d20[1:]):
        raise AssertionError("D20 methods changed the dominant original IDs")
    index = plans["d20_index_uniform5"]
    spatial = plans["d20_spatial_uniform5"]
    spatial_meta = spatial.get("spatial")
    if not isinstance(spatial_meta, dict):
        raise AssertionError("Spatial selection plan has no geometry metadata")
    coords = spatial_meta["coordinates"]
    records = coords["records"]
    if len(records) != len(spatial["stored_to_original"]):
        raise AssertionError("coordinate and physical visual span disagree")
    index_aux = [int(x) for x in index["contextual_ids"]]
    spatial_aux = [int(x) for x in spatial["contextual_ids"]]
    if (index_aux != [int(x) for x in spatial_meta["index_uniform_aux_ids"]]
            or set(spatial_aux) != set(spatial_meta["selected_aux_ids"])):
        raise AssertionError("Spatial references the wrong IndexUniform IDs")
    if (len(index_aux) != len(spatial_aux)
            or set(index_aux) & set(dominant)
            or set(spatial_aux) & set(dominant)):
        raise AssertionError("D20 auxiliary count or disjointness changed")
    quota = {branch: sum(records[i]["branch"] == branch for i in index_aux)
             for branch in ("base", "high")}
    if quota != spatial_meta["branch_quotas"]:
        raise AssertionError("IndexUniform and Spatial branch quotas differ")
    for branch in ("base", "high"):
        result = spatial_meta["branch_results"][branch]
        selected = [int(x) for x in result["selected_original_ids"]]
        if (len(selected) != quota[branch]
                or len(set(selected)) != quota[branch]
                or any(records[i]["branch"] != branch for i in selected)
                or int(result["quota"]) != quota[branch]
                or len(result["regions"]) != quota[branch]
                or int(result["fallback_count"]) !=
                    int(result["empty_region_count"])):
            raise AssertionError(f"invalid {branch} spatial region allocation")
    k = int(index["k"])
    if int(index["k_context"]) != k//5:
        raise AssertionError("D20 auxiliary budget is not floor(k/5)")
    ref = plans["d21_1_c3_9_reference"]
    if (int(ref["k_context"]) != 10*k//64
            or int(ref["k_dominant"]) != k-(10*k//64)):
        raise AssertionError("54:10 integer allocation changed")
    if any(int(plan["k"]) != k for plan in plans.values()):
        raise AssertionError("selective arms do not share the KV25 budget")
    set_i, set_s = set(index_aux), set(spatial_aux)
    dominant_set = set(dominant)
    full_i, full_s = dominant_set | set_i, dominant_set | set_s
    if (full_i != set(index["selected_original_ids"])
            or full_s != set(spatial["selected_original_ids"])):
        raise AssertionError("Index/Spatial full selected sets disagree with plans")
    aux_union = set_i | set_s
    full_union = full_i | full_s
    return {
        "dominant_original_ids_sha256": OLD.canonical_hash(dominant),
        "dominant_count": len(dominant),
        "index_aux_count": len(index_aux),
        "spatial_aux_count": len(spatial_aux),
        "branch_quotas": quota,
        "index_spatial_auxiliary_jaccard_global": (
            len(set_i & set_s)/len(aux_union) if aux_union else 1.0),
        "index_spatial_full_selected_jaccard_global": (
            len(full_i & full_s)/len(full_union) if full_union else 1.0),
        "added_original_ids": sorted(set_s-set_i),
        "removed_original_ids": sorted(set_i-set_s),
        "spatial_empty_regions_by_branch": {
            branch: int(spatial_meta["branch_results"][branch][
                "empty_region_count"]) for branch in ("base", "high")},
        "spatial_fallbacks_by_branch": {
            branch: int(spatial_meta["branch_results"][branch][
                "fallback_count"]) for branch in ("base", "high")},
        "coordinate_space": coords["coordinate_space"],
        "coordinate_mapping_sha256": OLD.canonical_hash(coords),
    }


def _grid_count_histogram(ids: list[int],
                          records: list[dict[str, Any] | None],
                          branch: str) -> list[int]:
    """Row-major selected-token counts on the fixed diagnostic grid."""
    bins = [0] * (DIAGNOSTIC_GRID*DIAGNOSTIC_GRID)
    for original in ids:
        record = records[original]
        if record is None or record["branch"] != branch:
            continue
        col = min(DIAGNOSTIC_GRID-1, int(float(record["x"])*DIAGNOSTIC_GRID))
        row = min(DIAGNOSTIC_GRID-1, int(float(record["y"])*DIAGNOSTIC_GRID))
        bins[row*DIAGNOSTIC_GRID + col] += 1
    return bins


def _coverage_rows(run_dir: Path, phase: str,
                   image_ids: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for image_id in image_ids:
        root = run_dir/"image_artifacts"/phase/image_id
        artifacts = {
            method: json.loads((root/f"{method}_selection.json").read_text())
            for method in ("d20_index_uniform5", "d20_spatial_uniform5")}
        plans = {method: artifact["layout_artifact"]["selection_plan"]
                 for method, artifact in artifacts.items()}
        geometry = plans["d20_spatial_uniform5"]["spatial"]
        records = geometry["coordinates"]["records"]
        dominant = [int(i) for i in plans["d20_index_uniform5"]["dominant_ids"]]
        index_aux = [int(i) for i in plans["d20_index_uniform5"]["contextual_ids"]]
        spatial_aux = [int(i) for i in plans["d20_spatial_uniform5"]["contextual_ids"]]
        full_index = set(dominant) | set(index_aux)
        full_spatial = set(dominant) | set(spatial_aux)
        full_union = full_index | full_spatial
        auxiliary_union = set(index_aux) | set(spatial_aux)
        full_global_jaccard = (
            len(full_index & full_spatial)/len(full_union)
            if full_union else 1.0)
        auxiliary_global_jaccard = (
            len(set(index_aux) & set(spatial_aux))/len(auxiliary_union)
            if auxiliary_union else 1.0)
        for branch in ("base", "high"):
            n_branch = sum(record is not None and record["branch"] == branch
                           for record in records)
            dominant_branch = {i for i in dominant
                               if records[i]["branch"] == branch}
            index_aux_branch = {i for i in index_aux
                                if records[i]["branch"] == branch}
            spatial_aux_branch = {i for i in spatial_aux
                                  if records[i]["branch"] == branch}
            index_full_branch = dominant_branch | index_aux_branch
            spatial_full_branch = dominant_branch | spatial_aux_branch
            branch_full_union = index_full_branch | spatial_full_branch
            branch_aux_union = index_aux_branch | spatial_aux_branch
            full_branch_jaccard = (
                len(index_full_branch & spatial_full_branch)
                / len(branch_full_union) if branch_full_union else 1.0)
            auxiliary_branch_jaccard = (
                len(index_aux_branch & spatial_aux_branch)
                / len(branch_aux_union) if branch_aux_union else 1.0)
            d_branch = len(dominant_branch)
            result = geometry["branch_results"][branch]
            for method, auxiliary in (
                    ("d20_index_uniform5", index_aux),
                    ("d20_spatial_uniform5", spatial_aux)):
                branch_aux = [i for i in auxiliary
                              if records[i]["branch"] == branch]
                auxiliary_hist = _grid_count_histogram(
                    branch_aux, records, branch)
                full_hist = _grid_count_histogram(
                    dominant+branch_aux, records, branch)
                if (sum(auxiliary_hist) != len(branch_aux)
                        or sum(full_hist) != d_branch+len(branch_aux)):
                    raise AssertionError("diagnostic histogram dropped a selected ID")
                aux_count = sum(count > 0 for count in auxiliary_hist)
                all_count = sum(count > 0 for count in full_hist)
                aux_fraction = aux_count/(DIAGNOSTIC_GRID*DIAGNOSTIC_GRID)
                all_fraction = all_count/(DIAGNOSTIC_GRID*DIAGNOSTIC_GRID)
                output.append({
                    "phase": phase, "image_id": image_id,
                    "branch": branch, "method_key": method,
                    "diagnostic_grid_height": DIAGNOSTIC_GRID,
                    "diagnostic_grid_width": DIAGNOSTIC_GRID,
                    "N_branch_content": n_branch,
                    "dominant_count": d_branch,
                    "auxiliary_count": len(branch_aux),
                    "selected_count": d_branch+len(branch_aux),
                    "auxiliary_occupied_cells": aux_count,
                    "auxiliary_occupied_fraction": aux_fraction,
                    "auxiliary_count_grid_8x8_row_major_json": json.dumps(
                        auxiliary_hist, separators=(",", ":")),
                    "dominant_plus_aux_occupied_cells": all_count,
                    "dominant_plus_aux_occupied_fraction": all_fraction,
                    "full_selected_count_grid_8x8_row_major_json": json.dumps(
                        full_hist, separators=(",", ":")),
                    "spatial_empty_regions": int(result["empty_region_count"])
                        if method == "d20_spatial_uniform5" else None,
                    "spatial_fallback_count": int(result["fallback_count"])
                        if method == "d20_spatial_uniform5" else None,
                    "spatial_region_count": int(result["quota"])
                        if method == "d20_spatial_uniform5" else None,
                    "spatial_empty_region_fraction": (
                        int(result["empty_region_count"])/int(result["quota"])
                        if int(result["quota"]) else 0.0)
                        if method == "d20_spatial_uniform5" else None,
                    "spatial_fallback_fraction": (
                        int(result["fallback_count"])/int(result["quota"])
                        if int(result["quota"]) else 0.0)
                        if method == "d20_spatial_uniform5" else None,
                    "index_spatial_auxiliary_jaccard_global":
                        auxiliary_global_jaccard,
                    "index_spatial_full_selected_jaccard_global":
                        full_global_jaccard,
                    "index_spatial_auxiliary_jaccard_branch":
                        auxiliary_branch_jaccard,
                    "index_spatial_full_selected_jaccard_branch":
                        full_branch_jaccard,
                    "spatial_added_ids_global": json.dumps(
                        geometry["added_ids"], separators=(",", ":")),
                    "spatial_removed_ids_global": json.dumps(
                        geometry["removed_ids"], separators=(",", ":")),
                    "spatial_added_ids_branch": json.dumps(
                        sorted(spatial_aux_branch-index_aux_branch),
                        separators=(",", ":")),
                    "spatial_removed_ids_branch": json.dumps(
                        sorted(index_aux_branch-spatial_aux_branch),
                        separators=(",", ":")),
                    "coordinate_space": geometry["coordinates"][
                        "coordinate_space"],
                })
    return output


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
                            if method == "d20_spatial_uniform5":
                                context.validate_spatial_visual_kv_layout()
                            elif method in SELECTIVE:
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
                if len(receipts) != 7 or len(selections) != 6:
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
        spatial_relationships = _spatial_relationships(selections)
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
            "spatial_relationships": spatial_relationships,
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
    expected_meta = {
        (str(entry["image_id"]), str(q["question_id"]), method): {
            "turn_id": turn_id,
            "question": str(q["question"]),
            "gold": question_answers(q),
        }
        for entry in entries
        for turn_id, q in enumerate(
            entry["questions"][4:4+expected_questions], 1)
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
        "eight_arms": {r["method_key"] for r in rows} == set(METHODS),
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
        "question_turn_gold_match": True,
        "raw_score_recomputed": True,
        "index_spatial_branch_quota_match": True,
        "spatial_geometry_artifact": True,
        "no_phase_failure_receipt": not (run_dir/f"{phase}_failure.json").exists(),
        "os_read_trace_matches_plan": True,
    }
    for row in rows:
        if row["cache_hit"]:
            try:
                _assert_os_read_trace(row, row["method_key"])
            except (KeyError, TypeError, ValueError, AssertionError):
                checks["os_read_trace_matches_plan"] = False
    for row, key in zip(rows, observed):
        expected_row = expected_meta.get(key)
        if expected_row is None:
            checks["question_turn_gold_match"] = False
            continue
        if (int(row["turn_id"]) != expected_row["turn_id"]
                or row["question"] != expected_row["question"]
                or row["gold"] != expected_row["gold"]
                or row["request_id"] !=
                    f"{phase}:{key[0]}:{key[1]}:{key[2]}"):
            checks["question_turn_gold_match"] = False
        rescored = METRICS["gqa"](
            str(row["prediction"]), expected_row["gold"])
        if abs(float(row["correct"])-float(rescored)) > 1e-12:
            checks["raw_score_recomputed"] = False
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
        try:
            relation = _spatial_relationships({
                method: json.loads((artifact_root/f"{method}_selection.json")
                                   .read_text())
                for method in SELECTIVE})
            if relation != receipt.get("spatial_relationships"):
                checks["index_spatial_branch_quota_match"] = False
            spatial_plan = json.loads(
                (artifact_root/"d20_spatial_uniform5_selection.json")
                .read_text())["layout_artifact"]["selection_plan"]
            coordinates = spatial_plan["spatial"]["coordinates"]
            if (coordinates.get("coordinate_space") !=
                    "model_input_patch_grid"
                    or len(coordinates["records"]) !=
                    len(spatial_plan["stored_to_original"])):
                checks["spatial_geometry_artifact"] = False
        except (FileNotFoundError, KeyError, TypeError, ValueError,
                AssertionError):
            checks["index_spatial_branch_quota_match"] = False
            checks["spatial_geometry_artifact"] = False
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


def _paired_confusion(rows: list[dict[str, Any]], new: str,
                      old: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["turn_id"] > 1 and row["method_key"] in (new, old):
            key = (row["image_id"], row["question_id"])
            if row["method_key"] in grouped[key]:
                raise AssertionError("duplicate paired hit row")
            grouped[key][row["method_key"]] = row
    counts = {
        "both_correct": 0, "both_wrong": 0,
        "method_a_only_correct": 0, "method_b_only_correct": 0,
        "prediction_agreement_count": 0,
    }
    flips = []
    for (image_id, question_id), arms in sorted(grouped.items()):
        if set(arms) != {new, old}:
            raise AssertionError("missing paired hit row")
        a, b = arms[new], arms[old]
        ac, bc = bool(a["correct"]), bool(b["correct"])
        if ac and bc:
            counts["both_correct"] += 1
        elif not ac and not bc:
            counts["both_wrong"] += 1
        elif ac:
            counts["method_a_only_correct"] += 1
        else:
            counts["method_b_only_correct"] += 1
        counts["prediction_agreement_count"] += (
            a["prediction"] == b["prediction"])
        if ac != bc:
            flips.append({
                "image_id": image_id, "question_id": question_id,
                "question": a["question"], "gold": a["gold"],
                "method_a": new, "method_b": old,
                "method_a_prediction": a["prediction"],
                "method_b_prediction": b["prediction"],
                "method_a_correct": ac, "method_b_correct": bc,
            })
    counts["paired_hits"] = len(grouped)
    counts["prediction_agreement_fraction"] = (
        counts["prediction_agreement_count"]/len(grouped)
        if grouped else None)
    return counts, flips


def _summary_and_comparisons(rows: list[dict[str, Any]],
                             run_dir: Path, results_dir: Path,
                             phase: str) -> tuple[list[dict[str, Any]],
                                                  list[dict[str, Any]]]:
    receipt_paths = sorted(
        (run_dir / "image_artifacts" / phase).glob("*/image_receipt.json"))
    receipts = [json.loads(path.read_text()) for path in receipt_paths]
    summaries = []
    persistence_rows = []
    t1_by_image_method = {
        (r["image_id"], r["method_key"]): r
        for r in rows if r["turn_id"] == 1}
    for receipt in receipts:
        for method, persisted in receipt["persistence_by_method"].items():
            timing = persisted["timing_ms"]
            capture = t1_by_image_method[(
                receipt["image_id"], method)]["vision_capture_stats"]
            persistence_rows.append({
                "phase": phase, "image_id": receipt["image_id"],
                "method_key": method, "persist_ms": timing.get("persist_ms"),
                "capture_saliency_reduction_ms":
                    capture.get("saliency_reduction_ms", 0),
                "capture_key_reduction_ms":
                    capture.get("key_reduction_ms", 0),
                "capture_key_hook_submit_ms":
                    capture.get("key_hook_submit_ms", 0),
                "capture_key_materialize_ms":
                    capture.get("key_materialize_ms", 0),
                "kv_materialize_ms": timing.get("kv_materialize_ms", 0),
                "permutation_ms": timing.get("permutation_ms", 0),
                "descriptor_mapping_ms": timing.get("descriptor_mapping_ms", 0),
                "clustering_ms": timing.get("clustering_ms", 0),
                "geometry_mapping_ms": timing.get("geometry_mapping_ms", 0),
                "spatial_selection_ms": timing.get("spatial_selection_ms", 0),
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
    for new, old, role in COMPARISONS:
        quality = _paired_bootstrap(
            rows, new, old, "correct", scale=100)
        timing = _paired_bootstrap(
            rows, new, old, "end_to_end_ttft_ms")
        confusion, _ = _paired_confusion(rows, new, old)
        comparisons.append({
            **confusion,
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
    image_ids = [str(receipt["image_id"]) for receipt in receipts]
    coverage_rows = _coverage_rows(run_dir, phase, image_ids)
    write_csv(results_dir / f"{phase}_coverage.csv", coverage_rows)
    confusion_examples = {}
    for new, old, role in COMPARISONS:
        _, flips = _paired_confusion(rows, new, old)
        confusion_examples[f"{new}_minus_{old}"] = {
            "role": role, "flip_count": len(flips),
            "examples": flips[:20],
        }
    OLD.atomic_json(results_dir/f"{phase}_prediction_flips.json",
                    confusion_examples)
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
    by_method = {item["method_key"]: item for item in summaries}
    by_pair = {(item["new"], item["old"]): item for item in comparisons}
    primary = by_pair["d20_spatial_uniform5", "d20_index_uniform5"]
    practical = by_pair["d20_spatial_uniform5", "d25_c0"]
    allocation = by_pair["d21_1_c3_9_reference", "d20_context5"]

    def signal(item: Mapping[str, Any]) -> str:
        if item["quality_ci95_low_pp"] > 0:
            return "POSITIVE"
        if item["quality_ci95_high_pp"] < 0:
            return "NEGATIVE"
        return "INCONCLUSIVE"

    coverage_rows = list(csv.DictReader(
        (results_dir/"pilot_coverage.csv").open(newline="", encoding="utf-8")))
    coverage = defaultdict(list)
    for item in coverage_rows:
        key = (item["branch"], item["method_key"])
        for field in ("auxiliary_occupied_fraction",
                      "dominant_plus_aux_occupied_fraction",
                      "index_spatial_auxiliary_jaccard_branch",
                      "index_spatial_full_selected_jaccard_branch"):
            coverage[(key, field)].append(float(item[field]))
        if item["method_key"] == "d20_spatial_uniform5":
            for field in ("spatial_empty_region_fraction",
                          "spatial_fallback_fraction"):
                coverage[(key, field)].append(float(item[field]))
    receipts = [json.loads(path.read_text()) for path in sorted(
        (run_dir/"image_artifacts/pilot").glob("*/image_receipt.json"))]
    relation = [receipt["spatial_relationships"] for receipt in receipts]
    mean_auxiliary_jaccard_global = float(np.mean([
        item["index_spatial_auxiliary_jaccard_global"]
        for item in relation]))
    mean_full_selected_jaccard_global = float(np.mean([
        item["index_spatial_full_selected_jaccard_global"]
        for item in relation]))
    mean_empty = {
        branch: float(np.mean([
            item["spatial_empty_regions_by_branch"][branch]
            for item in relation]))
        for branch in ("base", "high")}
    mean_fallback = {
        branch: float(np.mean([
            item["spatial_fallbacks_by_branch"][branch]
            for item in relation]))
        for branch in ("base", "high")}
    reference_allocations = [
        json.loads((run_dir/"image_artifacts/pilot"/receipt["image_id"]/
                    "d21_1_c3_9_reference_selection.json").read_text())
        for receipt in receipts]
    reference_d_ratio = float(np.mean([
        item["actual_dominant_ratio"] for item in reference_allocations]))
    reference_c_ratio = float(np.mean([
        item["actual_contextual_ratio"] for item in reference_allocations]))

    lines = [
        "# LLaVA-NeXT GQA SpatialUniform versus IndexUniform Visual-KV25",
        "",
        "This is the previously used 40-image development workload. "
        "The Spatial method samples original decoder KV rows on the model "
        "input patch grids; it does not merge features or KV values. "
        "The 54:10 arm changes only the dominant/contextual integer allocation "
        "within the existing original-token contextual selector, and is not "
        "a reproduction of VisionZip feature merging.",
        "",
        "| Method | D/aux mean tokens | All accuracy | Hit accuracy | "
        "Δ hit vs D25 (pp) | Δ hit vs Index (pp) | Hit TTFT (ms) | "
        "SSD MB/hit | Persistence ms/image |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        method = item["method_key"]
        dc = (f"{item['k_dominant_mean']:.1f}/{item['k_context_mean']:.1f}"
              if method in SELECTIVE else "N/A")
        d25 = 100*(item["hit_accuracy"]-by_method["d25_c0"]["hit_accuracy"])
        index = 100*(item["hit_accuracy"]-
                     by_method["d20_index_uniform5"]["hit_accuracy"])
        lines.append(
            f"| {item['method']} | {dc} | {item['all_accuracy']:.4f} | "
            f"{item['hit_accuracy']:.4f} | {d25:+.2f} | {index:+.2f} | "
            f"{item['hit_ttft_mean_ms']:.2f} | "
            f"{item['total_ssd_mb_per_hit']:.3f} | "
            f"{_fmt(item['persistence_mean_ms'])} |")
    lines += [
        "",
        f"54:10 integer allocation is k_aux=floor(10k/64), "
        f"k_dominant=k-k_aux. Across 40 images, its actual mean "
        f"dominant/content fraction was {reference_d_ratio:.4f} and "
        f"auxiliary/content fraction {reference_c_ratio:.4f}; "
        f"the displayed 21.1/3.9 percentages are nominal. "
        f"Average counts were "
        f"{by_method['d21_1_c3_9_reference']['k_dominant_mean']:.2f}/"
        f"{by_method['d21_1_c3_9_reference']['k_context_mean']:.2f}.",
        "",
        "## Fixed comparisons and paired disagreement",
        "",
        "Primary metric: T2–T6 accuracy, 200 paired hits across 40 image "
        "clusters. One percentage point equals two answers. Intervals use "
        "10,000 image-cluster bootstrap resamples with seed 1234; every "
        "question from one image stays in its cluster. A confidence interval "
        "containing zero establishes neither improvement nor equivalence.",
        "",
        "| Pair (A − B) | Role | Δ hit accuracy (pp), 95% CI | "
        "Δ hit TTFT (ms), 95% CI | TTFT ratio, 95% CI | "
        "Both right / both wrong / A only / B only | Prediction agreement |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        a, b = item["new"], item["old"]
        lines.append(
            f"| {METHOD_META[a]['label']} − {METHOD_META[b]['label']} | "
            f"{item['role']} | {item['quality_difference_pp']:+.2f} "
            f"[{item['quality_ci95_low_pp']:+.2f}, "
            f"{item['quality_ci95_high_pp']:+.2f}] | "
            f"{item['hit_ttft_difference_ms']:+.2f} "
            f"[{item['hit_ttft_ci95_low_ms']:+.2f}, "
            f"{item['hit_ttft_ci95_high_ms']:+.2f}] | "
            f"{item['hit_ttft_ratio']:.3f} "
            f"[{item['hit_ttft_ratio_ci95_low']:.3f}, "
            f"{item['hit_ttft_ratio_ci95_high']:.3f}] | "
            f"{item['both_correct']} / {item['both_wrong']} / "
            f"{item['method_a_only_correct']} / "
            f"{item['method_b_only_correct']} | "
            f"{item['prediction_agreement_fraction']:.3f} |")
    lines += [
        "",
        "The paired counts compare the same image and question. Equal mean "
        "accuracy can still hide different correct and incorrect answers. "
        "Examples are in pilot_prediction_flips.json.",
        "",
        "## Spatial coverage on a separate fixed 8×8 diagnostic grid",
        "",
        "The grid below measures occupied cells in each model-input patch "
        "branch. coverage.csv also records row-major 64-bin selected-token "
        "count histograms for auxiliary IDs and the full dominant plus "
        "auxiliary set on this fixed grid. It is separate from the Spatial "
        "selector's own regions and is not a semantic-coverage measure.",
        "",
        "| Branch | Method | Auxiliary occupied fraction | "
        "Dominant ∪ auxiliary occupied fraction |",
        "|---|---|---:|---:|",
    ]
    for branch in ("base", "high"):
        for method in ("d20_index_uniform5", "d20_spatial_uniform5"):
            aux = np.mean(coverage[((branch, method),
                                    "auxiliary_occupied_fraction")])
            total = np.mean(coverage[((branch, method),
                                      "dominant_plus_aux_occupied_fraction")])
            lines.append(
                f"| {branch} | {METHOD_META[method]['label']} | "
                f"{aux:.4f} | {total:.4f} |")
    lines += [
        "",
        f"Mean full selected-ID Jaccard (dominant ∪ auxiliary) between "
        f"Index and Spatial was {mean_full_selected_jaccard_global:.4f}; "
        f"auxiliary-only Jaccard was "
        f"{mean_auxiliary_jaccard_global:.4f}. "
        "The full-set measure includes the shared dominant IDs. "
        "Per-branch mean full/auxiliary Jaccard: "
        + "; ".join(
            f"{branch} "
            f"{np.mean(coverage[((branch, 'd20_spatial_uniform5'), 'index_spatial_full_selected_jaccard_branch')]):.4f}/"
            f"{np.mean(coverage[((branch, 'd20_spatial_uniform5'), 'index_spatial_auxiliary_jaccard_branch')]):.4f}"
            for branch in ("base", "high"))
        + ".",
        "",
        f"Mean originally empty regions/fallback selections per image: "
        f"base {mean_empty['base']:.2f}/{mean_fallback['base']:.2f}, "
        f"high {mean_empty['high']:.2f}/{mean_fallback['high']:.2f}. "
        "Mean empty-region/fallback fractions by branch: "
        + "; ".join(
            f"{branch} "
            f"{np.mean(coverage[((branch, 'd20_spatial_uniform5'), 'spatial_empty_region_fraction')]):.4f}/"
            f"{np.mean(coverage[((branch, 'd20_spatial_uniform5'), 'spatial_fallback_fraction')]):.4f}"
            for branch in ("base", "high"))
        + ". Fractions are count divided by that branch's quota "
          "(zero when quota is zero). Per-image and per-branch quotas, "
          "added/removed IDs, and coverage are in pilot_coverage.csv "
          "and selection JSON artifacts.",
        "",
        "## Turn-level quality and measured costs",
        "",
        "| Method | T1 | T2 | T3 | T4 | T5 | T6 | "
        "Hit TTFT p50/p95 (ms) | Hit E2E (ms) | "
        "Content / structural inclusive retention | Preads/hit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        turns = " | ".join(f"{item[f'turn{turn}_accuracy']:.4f}"
                           for turn in range(1, 7))
        lines.append(
            f"| {item['method']} | {turns} | "
            f"{item['hit_ttft_p50_ms']:.2f}/"
            f"{item['hit_ttft_p95_ms']:.2f} | "
            f"{item['hit_e2e_mean_ms']:.2f} | "
            f"{_fmt(item['actual_content_retention'], 4)}/"
            f"{_fmt(item['structural_inclusive_retention'], 4)} | "
            f"{item['total_preads_per_hit']:.1f} |")
    lines += [
        "",
        "Persistence is an image-level one-time cost outside hit TTFT. "
        "The T1 TTFT includes image read/decode, processor, vision, "
        "prefill, first-token materialization, and CUDA synchronization. "
        "Hit TTFT includes prompt preparation, real SSD pread, transfer, "
        "prefill, and first-token materialization. The per-component "
        "capture, geometry, spatial selection, clustering, KV repack, "
        "write, and fsync timings are in pilot_persistence.csv. "
        "Components can overlap and are not added into a fabricated "
        "critical path. Metadata activation and DONTNEED page-cache "
        "conditioning are separately recorded; DONTNEED does not prove "
        "cold SSD controller or NAND state. A Python os.pread wrapper "
        "records each returned range in the timed stored-hit path, with "
        "the same instrumentation on every stored arm.",
        "",
        "## Answers to the eight experiment questions",
        "",
        f"1. Spatial versus Index hit quality: {signal(primary)}. "
        f"Paired difference {primary['quality_difference_pp']:+.2f} pp, "
        f"95% CI [{primary['quality_ci95_low_pp']:+.2f}, "
        f"{primary['quality_ci95_high_pp']:+.2f}].",
        f"2. Spatial versus D25: {signal(practical)}. "
        f"Paired difference {practical['quality_difference_pp']:+.2f} pp, "
        f"95% CI [{practical['quality_ci95_low_pp']:+.2f}, "
        f"{practical['quality_ci95_high_pp']:+.2f}].",
        f"3. 54:10 versus Context5 allocation: {signal(allocation)}. "
        f"Paired difference {allocation['quality_difference_pp']:+.2f} pp, "
        f"95% CI [{allocation['quality_ci95_low_pp']:+.2f}, "
        f"{allocation['quality_ci95_high_pp']:+.2f}].",
        f"4. Spatial and Index both right {primary['both_correct']}, "
        f"both wrong {primary['both_wrong']}, Spatial only right "
        f"{primary['method_a_only_correct']}, Index only right "
        f"{primary['method_b_only_correct']}; exact prediction agreement "
        f"{primary['prediction_agreement_fraction']:.3f}.",
    ]
    for branch in ("base", "high"):
        aux_delta = 100*(
            np.mean(coverage[((branch, "d20_spatial_uniform5"),
                              "auxiliary_occupied_fraction")])-
            np.mean(coverage[((branch, "d20_index_uniform5"),
                              "auxiliary_occupied_fraction")]))
        all_delta = 100*(
            np.mean(coverage[((branch, "d20_spatial_uniform5"),
                              "dominant_plus_aux_occupied_fraction")])-
            np.mean(coverage[((branch, "d20_index_uniform5"),
                              "dominant_plus_aux_occupied_fraction")]))
        lines.append(
            f"5. {branch} 8×8 occupied-cell change: auxiliary "
            f"{aux_delta:+.2f} pp; dominant plus auxiliary "
            f"{all_delta:+.2f} pp. These are geometric diagnostics, "
            "not evidence that spatial coverage caused quality change.")
    lines += [
        "6. Every selective arm used exactly ceil(0.25 × N) original "
        "content rows. The four D20 arms share dominant IDs; Spatial "
        "matches Index base/high auxiliary quotas image by image. "
        "The same structural sidecar policy and selective SSD "
        "bytes/pread counts were audited on paired hits.",
        f"7. Spatial minus Index: T1 TTFT "
        f"{by_method['d20_spatial_uniform5']['t1_ttft_mean_ms']-by_method['d20_index_uniform5']['t1_ttft_mean_ms']:+.2f} "
        f"ms, persistence "
        f"{by_method['d20_spatial_uniform5']['persistence_mean_ms']-by_method['d20_index_uniform5']['persistence_mean_ms']:+.2f} "
        f"ms/image, hit TTFT {primary['hit_ttft_difference_ms']:+.2f} "
        "ms; exact component costs are in persistence.csv.",
        "8. This reused development pilot cannot establish independent "
        "generalization. A favorable observed arm should be tested on "
        "a genuinely independent workload before considering a main "
        "method change; no main method is changed here.",
        "",
        "## Scope and audit",
        "",
        f"Checkpoint revision: {_checkpoint_snapshot()['revision']}. "
        "NF4 model, BF16 compute, eager attention, FP16 SSD KV, "
        "64-token chunks, seed 1234, greedy max_new_tokens=16.",
        f"Runner audit: {'PASS' if audit['passed'] else 'FAIL'}; "
        f"{audit['observed_requests']}/{audit['expected_requests']} "
        "requests. Prior source/results/stores were protected. "
        "After scoped cleanup, SSD replay requires rebuilding stores "
        "from each normal Turn 1.",
        "",
        "IMPLEMENTATION: PASS",
        "GEOMETRY VALIDATION: PASS",
        "GPU CORRECTNESS: PASS",
        "PILOT: COMPLETE",
        f"SPATIAL VS INDEX QUALITY: {signal(primary)}",
        f"54:10 ALLOCATION SIGNAL: {signal(allocation)}",
        "INDEPENDENT HOLDOUT: NOT RUN",
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
        raise ValueError("new eight-arm GPU correctness receipt is not PASS")
    source = receipt.get("source_sha256")
    if not isinstance(source, dict):
        raise ValueError("GPU gate lacks source SHA256")
    required = ("mmimpress/cvpr25.py", "mmimpress/contextual_kv25.py",
                "mmimpress/spatial_kv25.py", "mmimpress/piggyback.py",
                "mmimpress/store.py", "mmimpress/serve.py",
                "mmimpress/model.py",
                "scripts/92_validate_llava_contextual_kv25.py",
                "scripts/94_validate_llava_spatial_kv25.py")
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
            raise ValueError("GPU gate lacks six passed selective arms")
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
                         "coverage.csv", "persistence.csv",
                         "prediction_flips.json", "independent_audit.json"):
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
            "retained for analysis without rebuilding. The primary paired "
            "comparison is SpatialUniform minus IndexUniform; the 54:10 "
            "reference uses floor(10*k/64) auxiliary rows. The fixed "
            "independent coverage diagnostic is an 8 by 8 grid per branch.\n"))
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
            or not run_dir.name.startswith("llava_spatial_kv25_")):
        raise ValueError("run-dir must be an existing spatial freeze under runs/")
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

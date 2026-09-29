#!/usr/bin/env python3
"""Four-arm Qwen KV25 cache-hit reevaluation on frozen v2 stores.

The old and new Ours arms open the same immutable repacked store. Every arm
performs its own normal pixel Turn 1; those diagnostic recaptures are discarded,
and previous persistence time is never attributed to this run.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import uuid
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MIGRATION = ROOT / "runs/qwen25_kv25_migration_20260929T042949Z"
BASELINE = MIGRATION / "protected_before.jsonl"
BASELINE_SHA256 = "13c2ba95cf9047fc8a5f300c6b4e18221cd13d5fd5f0af88767a9d0becbfb0b1"
V2_RUN = ROOT / "runs/qwen25_correctness_v2_20260928T081111Z"
GEOMETRY_REFERENCE = MIGRATION / "geometry_reference.json"
STORAGE_PLAN = MIGRATION / "storage_plan.json"
METHODS = ("recompute", "fullload", "qwen_ours_chunk25_legacy",
           "qwen_ours_kv25")
SCHEMA = "qwen25-kv25-four-arm-pilot-v1"
REQUIRED_SOURCES = ("mmimpress/qwen25/runner.py", "mmimpress/qwen25/store.py",
                    "mmimpress/qwen25/vision.py",
                    "docs/qwen25_kv25_budget_contract.md")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V1 = _load("qwen25_pilot_frozen_v1", ROOT / "scripts/79_eval_qwen25_pilot.py")
V2 = _load("qwen25_pilot_frozen_v2", ROOT / "scripts/84_eval_qwen25_v2_pilot.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return V1.canonical_hash(value)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True,
                  allow_nan=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(handle, value) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                            allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _baseline_hashes(paths: set[str]) -> dict[str, str]:
    if sha256(BASELINE) != BASELINE_SHA256:
        raise RuntimeError("pre-edit protection manifest changed")
    found = {}
    with BASELINE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            path = row["path"]
            if path in paths:
                found[path] = row["sha256"]
    if set(found) != paths:
        raise RuntimeError(f"protected inventory lacks {sorted(paths - set(found))}")
    return found


def _phase(dataset: str) -> str:
    return "gqa" if dataset in ("smoke", "gqa") else "mt"


def _frozen_manifest(dataset: str) -> dict:
    phase = _phase(dataset)
    old, binding = V2.frozen_workload(phase)
    if dataset == "smoke":
        images = copy.deepcopy(old["images"][:4])
        for image in images:
            image["turns"] = image["turns"][:3]
    else:
        images = copy.deepcopy(old["images"])
    if len(images) != (4 if dataset == "smoke" else 40):
        raise RuntimeError("frozen image count changed")
    if any(len(i["turns"]) != (3 if dataset != "gqa" else 6)
           for i in images):
        raise RuntimeError("frozen turn count changed")
    for index, image in enumerate(images):
        offset = (index + 1234) % len(METHODS)
        image["method_order"] = list(METHODS[offset:] + METHODS[:offset])
    manifest = {
        "schema_version": SCHEMA, "phase": dataset,
        "source_phase": phase, "model_id": V1.MODEL_ID,
        "model_revision": V1.CHECKPOINT_REVISION,
        "seed": 1234, "max_new_tokens": 16, "chunk_size": 64,
        "budget_ratio": .25, "methods": list(METHODS),
        "history_policy": ("method_own_generated_answers" if dataset == "mt"
                           else "none_independent_questions"),
        "source_frozen_manifest": binding,
        "source_frozen_content_sha256": old["manifest_sha256"],
        "storage_policy": "read_only_v2_stores",
        "persistence_status": "NOT_REMEASURED",
        "images": images,
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    return manifest


def _store_root(dataset: str) -> Path:
    return V2_RUN / ("mt_pilot" if dataset == "mt" else "gqa_pilot") / "stores"


def _preflight_stores(manifest: dict) -> tuple[dict[str, str], dict]:
    """Bind every store meta to pre-edit SHA and frozen manifest image identity."""
    phase = manifest["source_phase"]
    root = _store_root(manifest["phase"])
    reference = read_json(GEOMETRY_REFERENCE)
    if reference.get("schema_version") != (
            "qwen25-kv25-independent-geometry-reference-v1"):
        raise RuntimeError("unexpected independent geometry reference")
    ref_rows = {str(row["image_id"]): row
                for row in reference["phases"][phase]["entries"]}
    requested = set()
    for image in manifest["images"]:
        iid = str(image["image_id"])
        for arm in ("fullload", "ours25"):
            requested.add(str((root / iid / arm / "meta.json").relative_to(ROOT)))
    trusted_hashes = _baseline_hashes(requested)
    allowlist = {}
    image_records = {}
    for image in manifest["images"]:
        iid = str(image["image_id"])
        metas = {}
        for arm in ("fullload", "ours25"):
            store = root / iid / arm
            path = store / "meta.json"
            key = str(path.relative_to(ROOT))
            if sha256(path) != trusted_hashes[key]:
                raise RuntimeError(f"protected Qwen store meta changed: {path}")
            meta = read_json(path)
            identity = meta["identity"]
            if identity["image_sha256"] != image["image_sha256"]:
                raise RuntimeError(f"image/store identity mismatch: {iid}")
            if identity["checkpoint_revision"] != V1.CHECKPOINT_REVISION:
                raise RuntimeError(f"checkpoint revision mismatch: {iid}")
            if identity["processor_settings"] != {
                    "min_pixels": 200704, "max_pixels": 802816,
                    "use_fast": True,
                    "processor_revision": V1.CHECKPOINT_REVISION}:
                raise RuntimeError(f"processor policy mismatch: {iid}")
            if (meta["dtype"] != "bfloat16" or meta["chunk_size"] != 64
                    or meta["num_kv_heads"] != 4
                    or meta["format"] != "qwen25_bf16_visual_kv_v1"
                    or meta["key_rope_state"] != "post_mrope"):
                raise RuntimeError(f"native Qwen store format mismatch: {iid}")
            layout = ("token_major_canonical_bf16" if arm == "fullload"
                      else "token_major_repacked_bf16")
            if meta["layout"] != layout:
                raise RuntimeError(f"wrong store layout: {path}")
            if arm == "ours25" and (
                    meta.get("score_source") != "last_fullatt_ViT_received_attention"
                    or not meta.get("global_order_all_layers")
                    or meta.get("saliency_sha256") is None):
                raise RuntimeError(f"repacked saliency provenance mismatch: {iid}")
            if meta["n_chunks"] != (meta["visual_count"] + 63) // 64:
                raise RuntimeError(f"store chunk count mismatch: {iid}")
            metas[arm] = meta
            allowlist[str(store.resolve())] = trusted_hashes[key]
        full, ours = metas["fullload"], metas["ours25"]
        for key in ("prefix_sha256", "prefix_input_ids", "geometry",
                    "logical_position_ids", "rope_deltas", "image_grid_thw",
                    "visual_count", "visual_start", "prefix_len",
                    "structural_indices", "structural_count",
                    "num_kv_heads", "head_dim", "num_layers", "dtype"):
            if full[key] != ours[key]:
                raise RuntimeError(f"canonical/repacked {key} mismatch: {iid}")
        if full["stored_to_original"] != list(range(full["visual_count"])):
            raise RuntimeError(f"canonical store is not raster: {iid}")
        if sorted(ours["stored_to_original"]) != list(range(ours["visual_count"])):
            raise RuntimeError(f"invalid importance permutation: {iid}")
        ref = ref_rows[iid]
        n = int(ours["visual_count"])
        k = (n + 3) // 4
        legacy_chunks = round(int(ours["n_chunks"]) * .25)
        legacy_kept = min(n, legacy_chunks * 64)
        m = (k + 63) // 64
        if any((ref["N_content"] != n, ref["S_structural"] != ours["structural_count"],
                ref["kv25_k"] != k, ref["kv25_chunks"] != m,
                ref["legacy_kept"] != legacy_kept,
                ref["legacy_chunks"] != legacy_chunks,
                ref["permutation_sha256"] != ours["permutation_sha256"],
                ref["image_sha256"] != image["image_sha256"])):
            raise RuntimeError(f"frozen independent geometry mismatch: {iid}")
        image_records[iid] = {
            "N_content": n, "S_structural": ours["structural_count"],
            "k_target": k, "kv25_chunks": m,
            "legacy_kept": legacy_kept, "legacy_chunks": legacy_chunks,
            "legacy_store": str((root / iid / "ours25").resolve()),
            "fullload_store": str((root / iid / "fullload").resolve()),
            "permutation_sha256": ours["permutation_sha256"],
            "selected_original_legacy": sorted(
                ours["stored_to_original"][:legacy_kept]),
            "selected_original_kv25": sorted(
                ours["stored_to_original"][:k]),
            "selected_stored_kv25": list(range(k)),
            "store_code_revision": ours["code_revision"],
            "row_bytes": int(ours["row_bytes"]),
            "num_layers": int(ours["num_layers"]),
            "prefix_len": int(ours["prefix_len"]),
            "stored_rows": int(ours["stored_rows"]),
            "meta_sha256": {arm: trusted_hashes[str(
                (root / iid / arm / "meta.json").relative_to(ROOT))]
                            for arm in ("fullload", "ours25")},
        }
    return allowlist, image_records


def _gpu_gate(path: Path) -> dict:
    """Bind exact current files and the ten-pair v2 manifest before model load."""
    expected_path = (MIGRATION / "gpu_validation/validation.json").resolve()
    if path.resolve() != expected_path:
        raise RuntimeError("pilot requires this migration's GPU receipt")
    value = read_json(path)
    if value.get("schema") != "qwen25-kv25-gpu-correctness-v1":
        raise RuntimeError("wrong Qwen KV25 GPU validation schema")
    if value.get("status") != "PASS" or value.get("pilot_eligible") is not True:
        raise RuntimeError("Qwen KV25 GPU correctness has not passed")
    gates = value.get("gates")
    expected_gates = {f"G{i}" for i in range(1, 13)}
    if not isinstance(gates, dict) or set(gates) != expected_gates:
        raise RuntimeError("Qwen GPU receipt lacks required G1–G12 gates")
    if any(gates[name].get("status") != "PASS" for name in expected_gates):
        raise RuntimeError("one or more required Qwen GPU gates failed")
    if value.get("manifest_sha256") != (
            "6458545772438ef1b41cd35bbe7ec17c319be6f779d1b939fc9a94e0ada928a8"):
        raise RuntimeError("Qwen GPU receipt used a different ten-pair manifest")
    freeze = value.get("source_freeze")
    if not isinstance(freeze, dict) or not isinstance(freeze.get("files"), dict):
        raise RuntimeError("Qwen GPU receipt lacks source freeze")
    freeze_path = Path(freeze.get("path", ""))
    if not freeze_path.is_file() or sha256(freeze_path) != freeze.get("sha256"):
        raise RuntimeError("Qwen GPU source freeze file changed")
    required = set(REQUIRED_SOURCES) | {
        "scripts/91_validate_qwen25_kv25.py",
        "runs/qwen25_correctness_v2_20260928T081111Z/validation_manifest.json",
        "data/index.json",
    }
    files = freeze["files"]
    if not required.issubset(files):
        raise RuntimeError("Qwen GPU freeze omits required sources")
    for relative, expected in files.items():
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"invalid source-freeze path: {relative}")
        if sha256(ROOT / rel) != expected:
            raise RuntimeError(f"GPU validation source binding mismatch: {relative}")
    if value.get("llava_protection", {}).get("status") != "PASS":
        raise RuntimeError("LLaVA CPU/protection gate did not pass")
    if not isinstance(value.get("samples"), list) or len(value["samples"]) != 10:
        raise RuntimeError("Qwen GPU validation did not cover ten fixed pairs")
    return {"path": str(path.resolve()), "sha256": sha256(path),
            "schema": value["schema"], "status": value["status"],
            "pilot_eligible": True,
            "manifest_sha256": value["manifest_sha256"],
            "source_freeze_sha256": freeze["sha256"],
            "source_hashes": files}


def _free_bytes() -> int:
    stat = os.statvfs(MIGRATION)
    return int(stat.f_bavail * stat.f_frsize)


def _check_space(threshold: int) -> int:
    free = _free_bytes()
    if free < threshold:
        raise RuntimeError(f"free disk bytes {free} below reserved {threshold}")
    return free


def _score(dataset: str, prediction: str, gold: str) -> float:
    return V1._score("mt_gqa_reconstructed" if dataset == "mt" else "gqa",
                     prediction, gold)


def _capture_logits(result: dict) -> None:
    import torch
    logits = result.pop("first_logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 1:
        raise RuntimeError("request omitted first FP32 logits")
    if logits.dtype != torch.float32 or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("first logits are non-finite or not FP32")
    if int(logits.argmax()) != int(result["first_token_id"]):
        raise RuntimeError("first token differs from first logits")
    result["first_logits_sha256"] = hashlib.sha256(
        logits.contiguous().numpy().tobytes()).hexdigest()
    result["first_logits_finite"] = True
    result["first_logits_shape"] = list(logits.shape)


def _run_request(runner, method: str, image: dict, turn: dict,
                 history: tuple[tuple[str, str], ...], geometry: dict,
                 image_path: Path) -> tuple[dict, dict | None]:
    iid = str(image["image_id"])
    if int(turn["turn_id"]) == 1 or method == "recompute":
        capture_mode = (
            "kv_only" if int(turn["turn_id"]) == 1 and method == "fullload"
            else "with_score" if int(turn["turn_id"]) == 1
            and method in METHODS[2:] else False)
        decode_start = time.perf_counter()
        with Image.open(image_path) as src:
            pixel_image = src.convert("RGB")
        decode_ms = (time.perf_counter() - decode_start) * 1e3
        result = runner.run_pixels(
            pixel_image, str(turn["question"]), history=history,
            capture=capture_mode, image_sha256=image["image_sha256"],
            return_logits=True)
        result["image_file_decode_ms"] = decode_ms
        result["timing_ms"]["image_file_decode"] = decode_ms
        result["ttft_ms"] += decode_ms
        result["request_e2e_ms"] += decode_ms
        captured = result.pop("capture", None)
        if capture_mode and captured is None:
            raise RuntimeError(f"Turn1 {method} lacked diagnostic capture")
        if result.get("geometry") is not None:
            g = result["geometry"]
            if int(g["visual_count"]) != geometry[iid]["N_content"]:
                raise RuntimeError(f"Turn1 image geometry changed: {iid}")
        if int(result["vision_calls"]) != 1:
            raise RuntimeError(f"pixel request vision count !=1: {iid}/{method}")
        result["t1_diagnostic_capture_mode"] = capture_mode or "none"
        result["t1_diagnostic_capture_discarded"] = bool(captured is not None)
        result["t1_store_origin"] = "prior_validated_v2_run" if method != "recompute" else None
        if captured is not None:
            del captured
        return result, None

    from mmimpress.qwen25.runner import Qwen25Runner
    if not isinstance(runner, Qwen25Runner):
        raise TypeError("unexpected Qwen runner")
    store = (geometry[iid]["fullload_store"] if method == "fullload"
             else geometry[iid]["legacy_store"])
    unit = ("visual_kv" if method == "qwen_ours_kv25" else "chunk")
    ratio = 1.0 if method == "fullload" else .25
    conditioning = runner.condition_cache(
        store, ratio, image_sha256=image["image_sha256"], budget_unit=unit)
    result = runner.run_cache(
        store, str(turn["question"]), history=history,
        budget_ratio=ratio, budget_unit=unit,
        image_sha256=image["image_sha256"], return_logits=True)
    if result.get("vision_calls") != 0 or result.get("online_query_score_calls") != 0:
        raise RuntimeError(f"cache hit used vision or online score: {iid}/{method}")
    expected = (geometry[iid]["N_content"] if method == "fullload"
                else geometry[iid]["legacy_kept"]
                if method == "qwen_ours_chunk25_legacy"
                else geometry[iid]["k_target"])
    expected_chunks = (math.ceil(geometry[iid]["N_content"] / 64)
                       if method == "fullload" else
                       geometry[iid]["legacy_chunks"]
                       if method == "qwen_ours_chunk25_legacy"
                       else geometry[iid]["kv25_chunks"])
    if result.get("kept_tokens") != expected or result.get("selected_chunks") != expected_chunks:
        raise RuntimeError(f"cache-hit geometry mismatch: {iid}/{method}")
    if method == "qwen_ours_kv25":
        if (result.get("budget_unit") != "visual_kv"
                or result.get("target_visual_tokens") != expected
                or result.get("normal_chunks_read") != expected_chunks
                or result.get("loaded_valid_visual_rows")
                != min(expected_chunks * 64, geometry[iid]["N_content"])
                or result.get("extra_valid_visual_rows")
                != min(expected_chunks * 64, geometry[iid]["N_content"]) - expected
                or result.get("selected_visual_stored") != list(range(expected))
                or result.get("selected_visual_original")
                != geometry[iid]["selected_original_kv25"]
                or result.get("structural_count") != geometry[iid]["S_structural"]
                or not isinstance(result.get("h2d_kv_bytes"), int)
                or result["h2d_kv_bytes"] <= 0):
            raise RuntimeError(f"KV25 result contract mismatch: {iid}")
    return result, conditioning


def run(dataset: str, output: Path, validation: Path) -> dict:
    if dataset not in ("smoke", "gqa", "mt"):
        raise ValueError(dataset)
    gate = _gpu_gate(validation)
    manifest = _frozen_manifest(dataset)
    allowlist, geometry = _preflight_stores(manifest)
    plan = read_json(STORAGE_PLAN)
    if plan.get("persistence_status") != "NOT_REMEASURED":
        raise RuntimeError("storage plan changed")
    free_before = _check_space(int(plan["minimum_free_before_pilot_bytes"]))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest["gpu_gate"] = gate
    manifest["manifest_sha256"] = canonical_hash(
        {k:v for k,v in manifest.items() if k!="manifest_sha256"})
    write_new(output / "manifest.json", manifest)
    write_new(output / "store_inventory.json", {
        "schema_version": SCHEMA, "source_root": str(_store_root(dataset)),
        "protected_manifest_sha256": BASELINE_SHA256,
        "geometry_reference_sha256": sha256(GEOMETRY_REFERENCE),
        "stores": geometry, "store_mode": "read_only"})
    write_new(output / "config.json", {
        "schema_version": SCHEMA, "dataset": dataset,
        "manifest_sha256": manifest["manifest_sha256"],
        "gpu_gate": gate, "storage_plan_sha256": sha256(STORAGE_PLAN),
        "free_bytes_before": free_before,
        "model_id": V1.MODEL_ID, "model_revision": V1.CHECKPOINT_REVISION,
        "attention_backend": "sdpa", "weight_quantization": "NF4",
        "compute_dtype": "bfloat16", "ssd_kv_dtype": "bfloat16",
        "seed": 1234, "max_new_tokens": 16,
        "min_pixels": 200704, "max_pixels": 802816,
        "chunk_size": 64, "ratio": .25,
        "persistence_status": "NOT_REMEASURED",
        "execution_scope": "CACHE-HIT REEVALUATION",
        "t1_store_origin": "prior validated v2 run",
        "t1_recap_policy": "diagnostic only; capture discarded and no new store write",
        "pixel_timing_policy": "per-request PIL file decode plus Qwen processor/vision/generation",
        "image_integrity_hash_timing_policy": "once per image outside request timer",
        "timing_addendum_sha256": sha256(ROOT / "docs/qwen25_kv25_pilot_timing_addendum.md"),
        "source_hashes": {r:sha256(ROOT/r) for r in (*REQUIRED_SOURCES,
            "scripts/94_eval_qwen_kv25_pilot.py",
            "docs/qwen25_kv25_pilot_timing_addendum.md")},
    })

    import torch
    from mmimpress.qwen25.runner import Qwen25Runner
    random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    runner = Qwen25Runner(
        trusted_legacy_store_meta_sha256=allowlist).load()
    raw_count, failed = 0, False
    status = {"schema_version": SCHEMA, "status": "RUNNING",
              "expected_requests": sum(len(i["turns"]) for i in manifest["images"]) * 4,
              "actual_requests": 0, "started_unix": time.time()}
    write_new(output / "status.json", status)
    try:
        write_new(output / "runtime.json", runner.runtime_fingerprint())
        warm = runner.run_pixels(
            Image.new("RGB", (448,448), (127,127,127)),
            "Describe the image briefly.", capture=False, return_logits=True)
        _capture_logits(warm)
        write_new(output / "warmup.json", warm)
        with (output / "raw.jsonl").open("x", encoding="utf-8") as raw_handle, (
                output / "gpu_inventory.jsonl").open("x", encoding="utf-8") as gpu_handle:
            V1._record_gpu_inventory(gpu_handle, "pilot_start")
            for image in manifest["images"]:
                _check_space(int(plan["stop_if_free_below_bytes"]))
                iid = str(image["image_id"])
                V1._record_gpu_inventory(gpu_handle, "image_start", iid)
                path = Path(image["image_path"])
                if sha256(path) != image["image_sha256"]:
                    raise RuntimeError(f"image bytes changed: {iid}")
                histories = {method: [] for method in METHODS}
                for turn in image["turns"]:
                    tid = int(turn["turn_id"])
                    for method in image["method_order"]:
                        prior = histories[method] if dataset == "mt" else []
                        history = tuple((row["question"], row["prediction"])
                                        for row in prior)
                        execution_id = uuid.uuid4().hex
                        started = time.time()
                        try:
                            result, conditioning = _run_request(
                                runner, method, image, turn, history,
                                geometry, path)
                            _capture_logits(result)
                            if result["generated_token_count"] != len(result["generated_token_ids"]):
                                raise RuntimeError("generated token count mismatch")
                            if not (0 <= float(result["ttft_ms"])
                                    <= float(result["request_e2e_ms"]) + 1e-3):
                                raise RuntimeError("invalid request timing")
                            selected_count = (
                                geometry[iid]["N_content"] if method == "fullload"
                                else geometry[iid]["legacy_kept"]
                                if method == "qwen_ours_chunk25_legacy"
                                else geometry[iid]["k_target"]
                                if method == "qwen_ours_kv25" else None)
                            selected_ids = (
                                list(range(geometry[iid]["N_content"]))
                                if method == "fullload"
                                else geometry[iid]["selected_original_legacy"]
                                if method == "qwen_ours_chunk25_legacy"
                                else geometry[iid]["selected_original_kv25"]
                                if method == "qwen_ours_kv25" else None)
                            row = {
                                "schema_version": SCHEMA,
                                "physical_execution_id": execution_id,
                                "attempt": 1, "status": "PASS",
                                "dataset": dataset, "image_id": iid,
                                "image_sha256": image["image_sha256"],
                                "dialog_id": image["dialog_id"],
                                "turn_id": tid,
                                "question_id": str(turn["question_id"]),
                                "question": turn["question"],
                                "gold": turn["gold"],
                                "method_id": method,
                                "budget_unit": (
                                    "pixels" if method == "recompute"
                                    else "full" if method == "fullload"
                                    else "chunk" if method == "qwen_ours_chunk25_legacy"
                                    else "visual_kv"),
                                "ratio": None if method == "recompute"
                                         else 1.0 if method == "fullload" else .25,
                                "chunk_size": 64,
                                "manifest_sha256": manifest["manifest_sha256"],
                                "config_sha256": sha256(output / "config.json"),
                                "model_id": V1.MODEL_ID,
                                "model_revision": V1.CHECKPOINT_REVISION,
                                "method_order": image["method_order"],
                                "method_order_position": image["method_order"].index(method),
                                "request_path": ("normal_pixels" if tid == 1
                                                 or method == "recompute"
                                                 else "read_only_ssd_cache_hit"),
                                "image_file_decode_ms": result.get("image_file_decode_ms"),
                                "timing_scope": (
                                    "per_request_image_decode_plus_processor_vision_generation"
                                    if tid == 1 or method == "recompute" else
                                    "cache_hit_prompt_store_read_assembly_generation"),
                                "history": prior,
                                "history_sha256": canonical_hash(prior),
                                "source_store": (
                                    None if method == "recompute" else
                                    geometry[iid]["fullload_store"] if method == "fullload"
                                    else geometry[iid]["legacy_store"]),
                                "store_persistence": "NOT_REMEASURED",
                                "N_content": geometry[iid]["N_content"],
                                "S_structural": geometry[iid]["S_structural"],
                                "k_target": geometry[iid]["k_target"],
                                "legacy_kept_count": geometry[iid]["legacy_kept"],
                                "selected_original_ids": (
                                    selected_ids if tid > 1 and method != "recompute"
                                    else None),
                                "selected_stored_ids": (
                                    list(range(selected_count)) if tid > 1
                                    and method != "recompute" and method != "fullload"
                                    else None),
                                "permutation_sha256": geometry[iid]["permutation_sha256"],
                                "conditioning": conditioning,
                                "prediction": result["prediction"],
                                "correct": _score(dataset, result["prediction"], turn["gold"]),
                                "ttft_ms": result["ttft_ms"],
                                "request_e2e_ms": result["request_e2e_ms"],
                                "first_token_id": result["first_token_id"],
                                "generated_token_ids": result["generated_token_ids"],
                                "generated_token_count": result["generated_token_count"],
                                "result": result,
                                "started_unix": started,
                                "finished_unix": time.time(),
                            }
                            io = result.get("read_io")
                            if io is not None and not isinstance(io, dict):
                                raise RuntimeError("malformed actual read counter")
                            row.update({
                                "normal_chunks_read": result.get("normal_chunks_read"),
                                "loaded_valid_visual_rows": result.get("loaded_valid_visual_rows"),
                                "extra_valid_visual_rows": result.get("extra_valid_visual_rows"),
                                "padding_rows_read": result.get("padding_rows_read"),
                                "padding_read_bytes": (
                                    result.get("padding_rows_read", 0)
                                    * geometry[iid]["row_bytes"] * 2
                                    * geometry[iid]["num_layers"]
                                    if io is not None else None),
                                "normal_visual_read_bytes": result.get("visual_read_bytes"),
                                "structural_read_bytes": result.get("structural_read_bytes"),
                                "metadata_read_bytes": result.get("metadata_read_bytes"),
                                "total_actual_pread_bytes": io.get("bytes") if io else None,
                                "actual_pread_calls": io.get("preads") if io else None,
                                "actual_read_spans": io.get("spans") if io else None,
                                "compact_content_rows": result.get("kept_tokens"),
                                "compact_total_rows": result.get("compact_prefix_tokens"),
                                "h2d_kv_bytes": result.get("h2d_kv_bytes"),
                                "gpu_cache_kv_bytes": result.get("gpu_cache_kv_bytes"),
                                "peak_gpu_allocated_bytes": result.get("peak_gpu_allocated_bytes"),
                                "peak_gpu_reserved_bytes": result.get("peak_gpu_reserved_bytes"),
                                "vision_calls": result.get("vision_calls"),
                                "online_query_score_calls": result.get("online_query_score_calls"),
                                "activation_ms_outside_timer": (
                                    conditioning.get("activation_ms")
                                    if conditioning else None),
                                "activation_io_outside_timer": (
                                    conditioning.get("activation_io")
                                    if conditioning else None),
                                "logical_content_retention": (
                                    selected_count / geometry[iid]["N_content"]
                                    if selected_count is not None and tid > 1 else None),
                                "structural_inclusive_retention": (
                                    (selected_count + geometry[iid]["S_structural"])
                                    / geometry[iid]["prefix_len"]
                                    if selected_count is not None and tid > 1 else None),
                            })
                            append_jsonl(raw_handle, row)
                            raw_count += 1
                            if dataset == "mt":
                                histories[method].append({
                                    "question_id": str(turn["question_id"]),
                                    "question": turn["question"],
                                    "prediction": result["prediction"]})
                        except BaseException as exc:
                            append_jsonl(raw_handle, {
                                "schema_version": SCHEMA,
                                "physical_execution_id": execution_id,
                                "attempt": 1, "status": "FAIL",
                                "dataset": dataset, "image_id": iid,
                                "turn_id": tid, "question_id": str(turn["question_id"]),
                                "method_id": method,
                                "error": f"{type(exc).__name__}: {exc}",
                                "started_unix": started,
                                "finished_unix": time.time()})
                            failed = True
                            raise
                runner.close()
                V1._record_gpu_inventory(gpu_handle, "image_end", iid)
            V1._record_gpu_inventory(gpu_handle, "pilot_end")
        if failed or raw_count != status["expected_requests"]:
            raise RuntimeError("pilot request coverage incomplete")
        status.update(status="PASS", actual_requests=raw_count,
                      finished_unix=time.time(), free_bytes_after=_free_bytes())
    except BaseException as exc:
        status.update(status="FAIL", actual_requests=raw_count,
                      error=f"{type(exc).__name__}: {exc}",
                      finished_unix=time.time(), free_bytes_after=_free_bytes())
        raise
    finally:
        runner.close()
        write_new(output / "final_status.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("smoke","gqa","mt"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        manifest = _frozen_manifest(args.dataset)
        _, geometry = _preflight_stores(manifest)
        print(json.dumps({"dataset": args.dataset, "manifest_sha256":
                          manifest["manifest_sha256"], "images": len(geometry),
                          "expected_requests": sum(len(i["turns"])
                          for i in manifest["images"]) * 4,
                          "status": "PREFLIGHT PASS"}))
    else:
        result = run(args.dataset, args.run_dir.resolve(),
                     args.validation.resolve())
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

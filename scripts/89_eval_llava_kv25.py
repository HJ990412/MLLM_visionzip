#!/usr/bin/env python3
"""Gated LLaVA image-only KV25 GQA pilot and generated-history MT smoke.

Every method answers T1 through the normal pixel path. FullLoad persists a
canonical store and the legacy Ours T1 persists one image-only store. The two
Ours hit paths use separate ImageContext objects over that same immutable
store. The new Ours T1 also captures image saliency; equality with the legacy
capture is checked before either hit path runs. Each image session retains
its complete run-local stores after raw rows, metadata and measured
persistence receipt have been fsynced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
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
    deterministic_method_rotation, persist_captured_raster_prefix,
    persist_captured_visual_prefix,
)
from mmimpress.serve import ImageContext, Server  # noqa: E402


SCHEMA = "llava-kv25-migration-pilot-v1"
GQA_INDEX = ROOT / "data/index.json"
MT_INDEX = ROOT / "data/mt_gqa/dialogues.json"
GQA_SHA = "514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a"
GQA_WORKLOAD_SHA = "97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66"
MT_SHA = "2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224"
MT_WORKLOAD_SHA = "0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62"
SEED = 1234
METHODS = ("recompute", "fullload", "ours_chunk25_legacy", "ours_kv25")
METHOD_META = {
    "recompute": {"label": "ReComp", "budget_unit": "none", "ratio": None},
    "fullload": {"label": "FullLoad", "budget_unit": "full_visual_kv", "ratio": 1.0},
    "ours_chunk25_legacy": {
        "label": "Ours-Chunk25-Legacy", "budget_unit": "chunk", "ratio": 0.25},
    "ours_kv25": {
        "label": "Ours-KV25-New", "budget_unit": "visual_kv", "ratio": 0.25},
}
SHORT_ANSWER = "Answer the current question with a single word or short phrase."
SOURCE_PATHS = (
    "mmimpress/config.py", "mmimpress/cvpr25.py", "mmimpress/serve.py",
    "mmimpress/store.py", "mmimpress/piggyback.py", "mmimpress/model.py",
    "mmimpress/mt_gqa.py", "scripts/49_eval_query_aware_baseline.py",
    "scripts/73_eval_mt_gqa_5arm_generated_shard.py", "scripts/89_eval_llava_kv25.py",
)


def load_helper(name: str, filename: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


QA = load_helper("_kv25_validated_qa", "49_eval_query_aware_baseline.py")
MT = load_helper("_kv25_validated_mt", "73_eval_mt_gqa_5arm_generated_shard.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")).hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite raw value")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temp.open("x", encoding="utf-8") as handle:
        json.dump(safe(value), handle, ensure_ascii=False, sort_keys=True,
                  indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(handle, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(safe(value), ensure_ascii=False, allow_nan=False)
                 + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def require_gpu_gate(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    verdict = receipt.get("GPU_CORRECTNESS")
    if verdict != "PASS":
        raise ValueError(f"GPU correctness gate is not PASS: {path}")
    required_sources = (
        "mmimpress/cvpr25.py", "mmimpress/serve.py", "mmimpress/store.py",
        "mmimpress/model.py", "mmimpress/piggyback.py")
    source_hashes = receipt.get("source_sha256")
    if not isinstance(source_hashes, dict) or any(
            source_hashes.get(name) != sha256_file(ROOT/name)
            for name in required_sources):
        raise ValueError("GPU correctness receipt is stale for current source")
    contract_path = ROOT / "docs/llava_kv25_budget_contract.md"
    if receipt.get("contract_sha256") != sha256_file(contract_path):
        raise ValueError("GPU correctness receipt is stale for budget contract")
    if (receipt.get("model_id") != MODEL_ID
            or receipt.get("chunk_size") != CHUNK_SIZE
            or receipt.get("index_sha256") != GQA_SHA):
        raise ValueError("GPU correctness receipt model/chunk/index mismatch")
    samples = receipt.get("fixed_samples")
    results = receipt.get("samples")
    if (not isinstance(samples, list) or len(samples) != 5
            or not isinstance(results, list) or len(results) != 5
            or any(row.get("status") != "PASS" for row in results)):
        raise ValueError("GPU correctness receipt lacks five passed fixed samples")
    frozen = json.loads(GQA_INDEX.read_text(encoding="utf-8"))
    valid_pairs = {(str(entry["image_id"]), str(question["question_id"]))
                   for entry in frozen for question in entry["questions"][4:10]}
    pairs = [tuple(map(str, row)) for row in samples]
    expected_pairs = [
        ("n355567", "201751701"), ("n9181", "20929611"),
        ("n390187", "201861403"), ("n133585", "202108008"),
        ("n272098", "201535625"),
    ]
    result_pairs = [(str(row.get("image_id")), str(row.get("question_id")))
                    for row in results]
    if (pairs != expected_pairs or result_pairs != expected_pairs
            or any(row not in valid_pairs for row in pairs)):
        raise ValueError("GPU correctness receipt fixed sample identities changed")
    tolerance = receipt.get("first_step_fp32_logits_tolerance")
    if tolerance != {"atol": 0.0001, "rtol": 0.0001}:
        raise ValueError("GPU correctness receipt tolerance changed")
    return {"path": str(path.resolve()), "sha256": sha256_file(path),
            "verdict": verdict, "source_sha256": source_hashes,
            "contract_sha256": receipt["contract_sha256"],
            "fixed_samples": samples, "tolerance": tolerance}


def frozen_gqa() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _, entries, workload = QA._workload(GQA_INDEX, 4, 6, 40)
    if (workload["index_sha256"] != GQA_SHA
            or workload["full_workload_sha256"] != GQA_WORKLOAD_SHA
            or workload["selected_images"] != 40
            or workload["selected_questions"] != 240):
        raise ValueError("frozen GQA 40x6 manifest changed")
    for entry in entries:
        if not (ROOT / entry["image_path"]).is_file():
            raise FileNotFoundError(entry["image_path"])
    return entries, workload


def frozen_mt(n_images: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if n_images not in (3, 4, 5):
        raise ValueError("MT smoke requires 3-5 distinct images")
    if sha256_file(MT_INDEX) != MT_SHA:
        raise ValueError("frozen MT-GQA index changed")
    source = json.loads(MT_INDEX.read_text(encoding="utf-8"))
    dialogs = source.get("dialogues", [])
    if len(dialogs) != 4061 or len({d["image_id"] for d in dialogs}) != 398:
        raise ValueError("MT-GQA index counts changed")
    keys = "".join(
        f"{d['dialog_id']}\t{t['turn_id']}\t{t['question_id']}\n"
        for d in dialogs for t in d["turns"])
    if hashlib.sha256(keys.encode("utf-8")).hexdigest() != MT_WORKLOAD_SHA:
        raise ValueError("frozen MT-GQA workload changed")
    first_by_image = {}
    for ordinal, dialog in enumerate(dialogs):
        first_by_image.setdefault(dialog["image_id"], (ordinal, dialog))
    picked_ids = random.Random(SEED).sample(sorted(first_by_image), n_images)
    chosen = []
    for image_id in picked_ids:
        ordinal, dialog = first_by_image[image_id]
        row = dict(dialog)
        row["global_dialog_ordinal"] = ordinal
        if not (ROOT / row["image_path"]).is_file():
            raise FileNotFoundError(row["image_path"])
        chosen.append(row)
    return chosen, {
        "index_sha256": MT_SHA, "full_workload_sha256": MT_WORKLOAD_SHA,
        "source_dialogues": 4061, "source_images": 398,
        "sample_seed": SEED, "selected_dialogues": len(chosen),
        "selected_images": len({d["image_id"] for d in chosen}),
        "dialog_ids": [d["dialog_id"] for d in chosen],
        "image_ids": [d["image_id"] for d in chosen],
    }


def gpu_inventory() -> dict[str, Any]:
    cmd = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,"
           "utilization.gpu", "--format=csv,noheader,nounits"]
    value = subprocess.run(cmd, check=True, capture_output=True, text=True)
    rows = [line.strip() for line in value.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected one GPU, found {len(rows)}")
    fields = [part.strip() for part in rows[0].split(",")]
    if len(fields) != 5:
        raise RuntimeError("unexpected nvidia-smi GPU inventory")
    result = dict(zip(("index", "name", "memory_total_mib", "memory_used_mib",
                       "utilization_pct"), fields))
    if int(result["utilization_pct"]) > 10 or int(result["memory_used_mib"]) > 2500:
        raise RuntimeError(f"concurrent GPU load prevents timing: {result}")
    return result


def assert_gpu_exclusive() -> None:
    command = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
               "--format=csv,noheader,nounits"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    other = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        pid = int(line.split(",", 1)[0].strip())
        if pid != os.getpid():
            other.append(line.strip())
    if other:
        raise RuntimeError(f"concurrent GPU compute process prevents valid latency: {other}")


def mt_prompt(dialog: Mapping[str, Any], turn_id: int,
              predictions: Mapping[int, str]) -> tuple[str, str, list[dict[str, Any]]]:
    turns = dialog["turns"]
    if len(turns) != 3 or turn_id not in (1, 2, 3):
        raise ValueError("MT smoke requires an ordered three-turn dialogue")
    lines = []
    entries = []
    for prior_id in range(1, turn_id):
        if prior_id not in predictions:
            raise ValueError(f"generated T{prior_id} answer missing")
        prior = turns[prior_id - 1]
        answer = str(predictions[prior_id])
        lines.extend((f"Q{prior_id}: {prior['question']}",
                      f"A{prior_id}: {answer}"))
        entries.append({"turn_id": prior_id, "question_id": prior["question_id"],
                        "question": prior["question"], "answer": answer,
                        "answer_source": "method_local_generated"})
    history = "\n".join(lines)
    body = []
    if history:
        body.extend((history, ""))
    body.extend((f"Current question Q{turn_id}: {turns[turn_id-1]['question']}",
                 f"{SHORT_ANSWER} ASSISTANT:"))
    return "USER: <image>\n" + "\n".join(body), history, entries


def normal_request(runner, server, image, prompt_factory: Callable,
                   capture: str, phase: str):
    if phase == "gqa":
        prompt, _, _ = prompt_factory()
        capture_kind = {"none": "none", "raster": "qa",
                        "image_only": "ours"}[capture]
        return QA._run_pixels(runner, server, image, prompt, capture_kind)
    return MT._run_pixel_request(runner, server, image, prompt_factory,
                                 capture_kind=capture)


def stored_request(runner, server, ctx, prompt_factory: Callable,
                   method: str, image_id: str, full_visual_bytes: int,
                   phase: str):
    with QA._NoVisionForward(runner) as guard:
        condition_started = time.perf_counter()
        ctx.reader.drop_all()
        condition_done = time.perf_counter()
        torch.cuda.synchronize()
        request_started = time.perf_counter()
        prompt_started = time.perf_counter()
        prompt, history, history_entries = prompt_factory()
        if phase == "gqa":
            prompt = runner.prompt(prompt)
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
            value = server.request(ctx, mode="fullload", cold=False,
                                   suffix_ids=suffix_device)
        else:
            budget_unit = METHOD_META[method]["budget_unit"]
            value = server.request_cvpr25(
                ctx, static=None, budget=0.25, mode="prefix",
                budget_unit=budget_unit, sep_policy="sidecar", cold=False,
                seed=SEED, image_id=image_id, suffix_ids=suffix_device,
                expected_prefix_layout="visionzip_image_only")
        returned = time.perf_counter()
    phases = {
        "prompt_build_ms": float(prompt_ms), "tokenization_ms": float(token_ms),
        "image_preprocess_ms": 0.0, "input_prepare_ms": float(prepare_ms),
        "input_h2d_ms": float(h2d_ms), "processor_total_ms": None,
    }
    value.update(QA._timing_fields(value, request_started, returned, phases))
    value.update({
        "vision_forward_count": int(guard.calls),
        "page_cache_conditioning_started_at_s": condition_started,
        "page_cache_conditioning_finished_at_s": condition_done,
        "page_cache_conditioning_ms": (condition_done-condition_started)*1e3,
        "page_cache_conditioning_method": "posix_fadvise_DONTNEED",
        "page_cache_conditioning_excluded_from_ttft": True,
    })
    value = QA._json_result(value, "ours25" if method.startswith("ours")
                            else "fullload", full_visual_bytes)
    diagnostic = {
        "prompt": prompt, "prompt_sha256": hashlib.sha256(
            prompt.encode("utf-8")).hexdigest(),
        "history": history, "history_entries": history_entries,
        "suffix_ids_sha256": QA._hash_tensor(suffix_cpu),
    }
    return value, diagnostic


def persist_store(runner, result: Mapping[str, Any], diagnostic: Mapping[str, Any],
                  image_id: str, destination: Path, kind: str,
                  phase: str, method: str):
    cache = result.get("captured_past_key_values")
    if cache is None:
        raise AssertionError("T1 capture missing cache")
    encoded = diagnostic["enc_cpu"]
    common = {
        "image_id": image_id, "model_id": runner.model_id,
        "chunk_size": CHUNK_SIZE,
        "image_input_sha256": diagnostic["image_input_sha256"],
        "extra_metadata": {
            "dataset": phase, "migration_schema": SCHEMA,
            "source_method_key": method, "source_turn_id": 1,
            "store_lifecycle": "run_local_retained_read_only_after_image_commit",
        },
    }
    if kind == "raster":
        hidden = diagnostic["hidden_capture"]
        persisted = persist_captured_raster_prefix(
            runner, cache, encoded["input_ids"], encoded["image_sizes"][0],
            hidden.result_cpu(), destination, probe_heads=PROBE_HEADS,
            hidden_capture_stats=hidden, full_integrity_hash=False, **common)
    elif kind == "image_only":
        vision = diagnostic["vision_capture"]
        persisted = persist_captured_visual_prefix(
            runner, cache, encoded["input_ids"], encoded["image_sizes"][0],
            vision.result_cpu(), destination, capture_stats=vision,
            full_integrity_hash=False, **common)
    else:
        raise ValueError(kind)
    if not persisted["integrity"]["ok"]:
        raise AssertionError(f"persisted store integrity failed: {destination}")
    return persisted


def store_receipt(persisted: Mapping[str, Any], ctx: ImageContext) -> dict[str, Any]:
    meta = ctx.meta
    return {
        "store_dir": persisted["store_dir"],
        "meta_sha256": sha256_file(ctx.dir / "meta.json"),
        "file_sizes": persisted["file_sizes"],
        "bytes": persisted["bytes"],
        "timing_ms": persisted["timing_ms"],
        "hashes": persisted["hashes"],
        "durability": persisted["durability"],
        "integrity": persisted["integrity"],
        "physical_layout": meta.get("physical_layout"),
        "permutation_sha256": meta.get("permutation_sha256"),
        "image_input_sha256": meta.get("image_input_sha256"),
        "v_token_num": int(meta["v_token_num"]),
        "n_spatial": int(meta["n_spatial"]),
        "n_structural": len(meta["newline_idx"]),
        "num_layers": int(meta["num_layers"]),
        "num_heads": int(meta["num_heads"]),
        "head_dim": int(meta["head_dim"]),
        "dtype": str(meta["dtype"]),
        "chunk_size": int(meta["chunk_size"]),
    }


def content_metrics(value: Mapping[str, Any], method: str,
                    meta: Mapping[str, Any] | None) -> dict[str, Any]:
    if meta is None:
        return {"N_content": None, "N_structural": None, "k_target": None,
                "attended_content_kv_count": None, "normal_chunks_read": None,
                "normal_chunks_read_per_layer": None,
                "actual_loaded_real_rows": None, "unused_loaded_real_rows": None,
                "selected_stored_ids": None, "selected_original_ids": None,
                "keep_count_per_layer": None, "planned_normal_read_spans": None,
                "logical_content_retention": None,
                "visual_retention_including_structural": None,
                "logical_required_content_bytes": None,
                "unused_loaded_real_bytes": None,
                "actual_loaded_structural_rows": None,
                "structural_bytes_in_normal_payload": None,
                "structural_duplicate_across_files_bytes": 0,
                "same_file_duplicate_returned_bytes": 0,
                "padding_returned_bytes": 0}
    n = int(meta["n_spatial"])
    s = len(meta["newline_idx"])
    layers = int(meta["num_layers"])
    heads = int(meta["num_heads"])
    hd = int(meta["head_dim"])
    dtype_bytes = torch.tensor([], dtype={"float16": torch.float16,
        "bfloat16": torch.bfloat16, "float32": torch.float32}[meta["dtype"]]).element_size()
    bytes_per_content_row = layers * heads * hd * 2 * dtype_bytes
    if method == "fullload":
        k = n
        actual = n
        unused = 0
        stored = None  # canonical raster positions interleave separators
        original = None  # canonical raster order is not an importance ranking
        chunk_counts = [int(meta["n_chunks_per_layer"])] * layers
        keep = [n+s] * layers
        target = n
        spans = None
    elif method == "ours_chunk25_legacy":
        selected = value["selected_chunk_ids_per_layer"]
        if len(selected) != layers or any(row != selected[0] for row in selected):
            raise AssertionError("legacy prefix differs across decoder layers")
        stored = sorted({p for cid in selected[0]
                         for p in range(cid*CHUNK_SIZE, min((cid+1)*CHUNK_SIZE,n))})
        k = len(stored)
        actual = k
        unused = 0
        original = [int(meta["order"][p]) for p in stored]
        chunk_counts = [len(row) for row in selected]
        keep = [k+s] * layers
        target = None  # legacy target is a chunk count, not a token count
        spans = None
    elif method == "ours_kv25":
        target = (n+3)//4
        k = int(value["attended_content_kv_count"])
        actual = value["actual_loaded_real_rows"]
        unused = value["unused_loaded_real_rows"]
        stored = value["selected_stored_ids"]
        original = value["selected_original_ids"]
        raw_chunks = value["normal_chunks_read"]
        chunk_counts = ([int(raw_chunks)] * layers if isinstance(raw_chunks, int)
                        else [int(x) for x in raw_chunks])
        keep = value["keep_count_per_layer"]
        spans = value["planned_normal_read_spans"]
        if k != target:
            raise AssertionError(f"KV25 attended {k} content rows, expected {target}")
        if stored != list(range(target)):
            raise AssertionError("KV25 selected stored IDs are not first k")
        if original != [int(meta["order"][p]) for p in range(target)]:
            raise AssertionError("KV25 original IDs differ from independent permutation")
        if chunk_counts != [(target+CHUNK_SIZE-1)//CHUNK_SIZE]*layers:
            raise AssertionError("KV25 normal chunk count is not minimal prefix")
        if keep != [target+s]*layers:
            raise AssertionError("KV25 keep counts include extra content rows")
        if isinstance(actual, list):
            if len(actual) != layers or len(set(actual)) != 1:
                raise AssertionError("KV25 loaded real rows vary by layer")
            actual = actual[0]
        if isinstance(unused, list):
            if len(unused) != layers or len(set(unused)) != 1:
                raise AssertionError("KV25 unused rows vary by layer")
            unused = unused[0]
        if int(actual)-target != int(unused):
            raise AssertionError("KV25 unused rows do not match loaded minus attended")
    else:
        raise ValueError(method)
    if method == "fullload":
        chunk_ids = [list(range(int(meta["n_chunks_per_layer"])))
                     for _ in range(layers)]
    else:
        chunk_ids = value["selected_chunk_ids_per_layer"]
    raw_sep = meta.get("newline_stored", meta["newline_idx"])
    sep_per_layer = (raw_sep if raw_sep and isinstance(raw_sep[0], list)
                     else [raw_sep] * layers)
    structural_rows = [sum(int(pos)//CHUNK_SIZE in set(chunk_ids[li])
                           for pos in sep_per_layer[li])
                       for li in range(layers)]
    structural_payload_bytes = (sum(structural_rows) * heads * hd * 2 * dtype_bytes)
    if method == "ours_kv25":
        if (int(value["actual_loaded_structural_rows"]) != structural_rows[0]
                or int(value["structural_bytes_in_normal_payload"])
                != structural_payload_bytes):
            raise AssertionError("KV25 structural payload bytes disagree with independent chunk geometry")
    return {
        "N_content": n, "N_structural": s, "k_target": target,
        "attended_content_kv_count": k,
        "normal_chunks_read": chunk_counts[0],
        "normal_chunks_read_per_layer": chunk_counts,
        "actual_loaded_real_rows": actual, "unused_loaded_real_rows": unused,
        "selected_stored_ids": stored, "selected_original_ids": original,
        "keep_count_per_layer": keep, "planned_normal_read_spans": spans,
        "logical_content_retention": k/n,
        "visual_retention_including_structural": (k+s)/(n+s),
        "logical_required_content_bytes": k*bytes_per_content_row,
        "unused_loaded_real_bytes": int(unused)*bytes_per_content_row,
        "actual_loaded_structural_rows": structural_rows[0],
        "structural_bytes_in_normal_payload": structural_payload_bytes,
        "structural_duplicate_across_files_bytes": (
            structural_payload_bytes if method.startswith("ours") else 0),
        "same_file_duplicate_returned_bytes": 0,
        "padding_returned_bytes": 0,
    }


def full_pixel_geometry(meta: Mapping[str, Any]) -> dict[str, Any]:
    fields = content_metrics({}, "fullload", meta)
    layers = int(meta["num_layers"])
    fields.update({
        "k_target": None, "normal_chunks_read": 0,
        "normal_chunks_read_per_layer": [0]*layers,
        "actual_loaded_real_rows": 0,
        "unused_loaded_real_rows": 0,
        "actual_loaded_structural_rows": 0,
        "structural_bytes_in_normal_payload": 0,
        "structural_duplicate_across_files_bytes": 0,
        "selected_stored_ids": None, "selected_original_ids": None,
        "planned_normal_read_spans": None,
    })
    return fields


def make_row(*, phase: str, image_id: str, image_index: int,
             dialog_id: str | None, turn_id: int, question: Mapping[str, Any],
             method: str, order: tuple[str, ...], order_position: int,
             result: Mapping[str, Any], diagnostic: Mapping[str, Any],
             meta: Mapping[str, Any] | None, config_hash: str,
             manifest_hash: str, image_sha256: str) -> dict[str, Any]:
    gold = question_answers(question)
    prediction = str(result["answer"])
    correct = (METRICS["gqa"](prediction, gold) if phase == "gqa"
               else MT.strict_gqa_score(prediction, gold[0]))
    key = (f"gqa:{image_id}:{question['question_id']}:{method}" if phase == "gqa"
           else f"mt:{dialog_id}:t{turn_id}:{method}")
    value = {k: v for k, v in result.items()
             if k != "captured_past_key_values"}
    metrics = content_metrics(value, method, meta) if turn_id > 1 and method != "recompute" \
        else content_metrics({}, "recompute", None)
    row = {
        **value, **metrics,
        "schema_version": SCHEMA, "dataset": phase, "request_id": key,
        "image_id": image_id, "image_index": image_index,
        "image_sha256": image_sha256, "dialog_id": dialog_id,
        "turn_id": turn_id, "question_id": str(question["question_id"]),
        "question": str(question["question"]), "gold": gold,
        "method_key": method, "method_id": method,
        "method_label": METHOD_META[method]["label"],
        "budget_unit": METHOD_META[method]["budget_unit"],
        "ratio": METHOD_META[method]["ratio"], "chunk_size": CHUNK_SIZE,
        "model": MODEL_ID, "attention_backend": ATTN_IMPL,
        "compute_dtype": str(COMPUTE_DTYPE), "load_4bit": LOAD_4BIT,
        "config_sha256": config_hash, "manifest_sha256": manifest_hash,
        "method_order": list(order), "method_order_position": order_position,
        "request_path": "normal_pixel_turn1" if turn_id == 1
                        else "normal_pixel_recompute" if method == "recompute"
                        else "stored_visual_kv",
        "cache_hit": bool(turn_id > 1 and method != "recompute"),
        "prompt_sha256": diagnostic["prompt_sha256"],
        "suffix_ids_sha256": diagnostic["suffix_ids_sha256"],
        "history": diagnostic.get("history", ""),
        "history_entries": diagnostic.get("history_entries", []),
        "prediction": prediction, "correct": correct,
        "status": "ok",
    }
    if row["cache_hit"]:
        if int(row["vision_forward_count"]) != 0:
            raise AssertionError("cache hit invoked vision tower")
        if int(row.get("query_score_calls", 0)) != 0:
            raise AssertionError("image-only cache hit invoked online scoring")
        if int(row.get("static_score_calls", 0)) != 0:
            raise AssertionError("image-only cache hit loaded scored sidecar")
    if not 0 < float(row["end_to_end_ttft_ms"]):
        raise AssertionError("invalid outer TTFT")
    return safe(row)


def image_session(*, phase: str, entry: Mapping[str, Any], image_index: int,
                  runner, server, run_dir: Path, raw_handle,
                  config_hash: str, manifest_hash: str) -> list[dict[str, Any]]:
    assert_gpu_exclusive()
    free_before = shutil.disk_usage(run_dir).free
    if free_before < 34 * 1024**3:
        raise RuntimeError("less than 34 GiB free before image session")
    image_id = str(entry["image_id"])
    dialog_id = str(entry["dialog_id"]) if phase == "mt_smoke" else None
    image_path = ROOT / str(entry["image_path"])
    image_sha = sha256_file(image_path)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    turns = (entry["questions"][4:10] if phase == "gqa"
             else [{"question_id": t["question_id"], "question": t["question"],
                    "answers": t["answers"]} for t in entry["turns"]])
    if len(turns) != (6 if phase == "gqa" else 3):
        raise AssertionError("image-session turn count changed")
    order = deterministic_method_rotation(METHODS, image_index, SEED)
    image_store = run_dir / "stores" / phase / image_id
    image_store.mkdir(parents=True, exist_ok=False)
    paths = {"fullload": image_store / "raster",
             "ours_chunk25_legacy": image_store / "image_only"}
    contexts: dict[str, ImageContext] = {}
    receipts: dict[str, Any] = {}
    activation: list[dict[str, Any]] = []
    generated: dict[str, dict[int, str]] = {method: {} for method in METHODS}
    rows: list[dict[str, Any]] = []
    captures: dict[str, str] = {}
    image_input_hashes: dict[str, str] = {}

    try:
        for turn_id, question in enumerate(turns, 1):
            assert_gpu_exclusive()
            turn_rows = []
            for position, method in enumerate(order):
                def prompt_factory(method_key=method):
                    if phase == "gqa":
                        return str(question["question"]), "", []
                    return mt_prompt(entry, turn_id, generated[method_key])

                is_pixel = turn_id == 1 or method == "recompute"
                if is_pixel:
                    capture = ("raster" if turn_id == 1 and method == "fullload"
                               else "image_only" if turn_id == 1 and method.startswith("ours")
                               else "none")
                    result, diagnostic = normal_request(
                        runner, server, image, prompt_factory, capture, phase)
                    if phase == "gqa":
                        diagnostic["history"] = ""
                        diagnostic["history_entries"] = []
                    if turn_id == 1:
                        image_input_hashes[method] = diagnostic["image_input_sha256"]
                        if method.startswith("ours"):
                            captures[method] = QA._hash_tensor(
                                diagnostic["vision_capture"].result_cpu())
                        if method in paths:
                            persisted = persist_store(
                                runner, result, diagnostic, image_id,
                                paths[method], "raster" if method == "fullload"
                                else "image_only", phase, method)
                            activation_started = time.perf_counter()
                            context = ImageContext(
                                paths[method], runner.model.device,
                                drop_cache=True, require_v_hidden=False)
                            if method.startswith("ours"):
                                context.validate_prefix_layout("visionzip_image_only")
                                context.validate_visual_kv_layout()
                            contexts[method] = context
                            activation.append({"method": method,
                                               "activation_ms": (
                                                   time.perf_counter()-activation_started)*1e3,
                                               "excluded_from_ttft": True})
                            receipts[method] = store_receipt(persisted, context)
                    result = QA._json_result(
                        {k: v for k, v in result.items()
                         if k != "captured_past_key_values"},
                        "ours25" if method.startswith("ours") else method, 0)
                else:
                    if method == "ours_kv25" and method not in contexts:
                        if "ours_chunk25_legacy" not in contexts:
                            raise AssertionError("shared image-only store unavailable")
                        activation_started = time.perf_counter()
                        context = ImageContext(
                            paths["ours_chunk25_legacy"], runner.model.device,
                            drop_cache=True, require_v_hidden=False)
                        context.validate_prefix_layout("visionzip_image_only")
                        context.validate_visual_kv_layout()
                        contexts[method] = context
                        activation.append({"method": method,
                                           "activation_ms": (
                                               time.perf_counter()-activation_started)*1e3,
                                           "excluded_from_ttft": True,
                                           "shared_physical_store": True})
                    context = contexts[method]
                    full_bytes = int(contexts["fullload"].meta["bytes_visual_kv"])
                    result, diagnostic = stored_request(
                        runner, server, context, prompt_factory, method,
                        image_id, full_bytes, phase)
                meta = contexts[method].meta if (turn_id > 1 and method != "recompute") else None
                row = make_row(
                    phase=phase, image_id=image_id, image_index=image_index,
                    dialog_id=dialog_id, turn_id=turn_id, question=question,
                    method=method, order=order, order_position=position,
                    result=result, diagnostic=diagnostic, meta=meta,
                    config_hash=config_hash, manifest_hash=manifest_hash,
                    image_sha256=image_sha)
                if turn_id == 1 and method == "ours_kv25":
                    row["physical_persistence_source"] = "ours_chunk25_legacy"
                if turn_id > 1:
                    if method == "recompute":
                        row.update(full_pixel_geometry(
                            contexts["ours_chunk25_legacy"].meta))
                    append_jsonl(raw_handle, row)
                    rows.append(row)
                turn_rows.append(row)
                generated[method][turn_id] = row["prediction"]
                del result, diagnostic
            assert_gpu_exclusive()
            if turn_id == 1:
                if (len({row["prompt_sha256"] for row in turn_rows}) != 1
                        or len({row["first_token_id"] for row in turn_rows}) != 1
                        or len({row["prediction"] for row in turn_rows}) != 1):
                    raise AssertionError("four-arm T1 normal-pixel output differs")
                if len(set(image_input_hashes.values())) != 1:
                    raise AssertionError("four-arm T1 image input differs")
                if captures.get("ours_chunk25_legacy") != captures.get("ours_kv25"):
                    raise AssertionError("old/new image-only saliency capture differs")
                image_meta = contexts["ours_chunk25_legacy"].meta
                for row in turn_rows:
                    row.update(full_pixel_geometry(image_meta))
                    append_jsonl(raw_handle, row)
                    rows.append(row)

        old_meta = contexts["ours_chunk25_legacy"].meta
        old_rows = [row for row in rows if row["method_key"] == "ours_chunk25_legacy"
                    and row["turn_id"] > 1]
        new_rows = [row for row in rows if row["method_key"] == "ours_kv25"
                    and row["turn_id"] > 1]
        if len({canonical_hash(row["selected_original_ids"]) for row in new_rows}) != 1:
            raise AssertionError("KV25 selected IDs changed with question/history")
        if len({canonical_hash(row["selected_chunk_ids_per_layer"]) for row in new_rows}) != 1:
            raise AssertionError("KV25 chunk IDs changed with question/history")
        old_set = set(old_rows[0]["selected_original_ids"])
        new_set = set(new_rows[0]["selected_original_ids"])
        old_chunks = len(old_rows[0]["selected_chunk_ids_per_layer"][0])
        new_chunks = len(new_rows[0]["selected_chunk_ids_per_layer"][0])
        free_after = shutil.disk_usage(run_dir).free
        if free_after < 30 * 1024**3:
            raise RuntimeError("less than 30 GiB free after image session")
        artifact = {
            "schema_version": SCHEMA, "dataset": phase, "image_id": image_id,
            "disk_free_before_bytes": free_before,
            "disk_free_after_bytes": free_after,
            "dialog_id": dialog_id, "image_sha256": image_sha,
            "method_order": list(order), "request_count": len(rows),
            "request_ids_sha256": canonical_hash([r["request_id"] for r in rows]),
            "T1_saliency_sha256": captures,
            "shared_image_only_store": True,
            "physical_persistence_once": receipts,
            "independent_deployment_persistence_attribution": {
                "ours_chunk25_legacy": receipts["ours_chunk25_legacy"]["timing_ms"],
                "ours_kv25": receipts["ours_chunk25_legacy"]["timing_ms"],
                "semantics": "same measured image-only write cost attributed to each independent deployment; physical experiment wrote once",
            },
            "activation_events": activation,
            "old_new_same_content_set": old_set == new_set,
            "old_content_count": len(old_set), "new_content_count": len(new_set),
            "old_normal_chunks": old_chunks, "new_normal_chunks": new_chunks,
            "new_chunk_comparison": ("same" if new_chunks == old_chunks
                                     else "increase" if new_chunks > old_chunks
                                     else "decrease"),
            "n_content": int(old_meta["n_spatial"]),
            "n_structural": len(old_meta["newline_idx"]),
        }
        atomic_json(run_dir / "image_artifacts" / phase / f"{image_id}.json", artifact)
        return rows
    finally:
        for context in contexts.values():
            context.close()
        # Keep the complete run-local Visual KV stores on SSD for independent
        # read-only revalidation. Their hashes and measured persistence receipts
        # are saved in the per-image artifact before this point.


def image_cluster_ci(rows: list[dict[str, Any]], key: str,
                     new: str = "ours_kv25", old: str = "ours_chunk25_legacy"):
    image_deltas = []
    by_image = defaultdict(dict)
    for row in rows:
        if row["turn_id"] > 1 and row["method_key"] in (new, old):
            by_image[row["image_id"]].setdefault(row["method_key"], []).append(
                float(row[key]))
    for methods in by_image.values():
        if set(methods) != {new, old} or len(methods[new]) != len(methods[old]):
            raise AssertionError("paired image-cluster data incomplete")
        image_deltas.append(float(np.mean(methods[new])-np.mean(methods[old])))
    if not image_deltas:
        return None
    generator = np.random.default_rng(SEED)
    values = np.asarray(image_deltas, dtype=np.float64)
    indices = generator.integers(0, len(values), size=(10_000, len(values)))
    draws = values[indices].mean(axis=1)
    return {"mean_difference_new_minus_old": float(values.mean()),
            "ci95": [float(np.quantile(draws, 0.025)),
                     float(np.quantile(draws, 0.975))],
            "image_clusters": len(values), "bootstrap_resamples": 10_000,
            "seed": SEED}


def summarize(rows: list[dict[str, Any]], phase: str,
              run_dir: Path, results_dir: Path,
              preflight_one_image: bool = False) -> dict[str, Any]:
    expected_images = (1 if phase == "gqa" and preflight_one_image else
                       40 if phase == "gqa" else len({r["image_id"] for r in rows}))
    expected_turns = 6 if phase == "gqa" else 3
    expected_rows = expected_images * expected_turns * 4
    counts = Counter(row["method_key"] for row in rows)
    request_ids = [row["request_id"] for row in rows]
    checks = {
        "exact_request_count": len(rows) == expected_rows,
        "unique_request_ids": len(set(request_ids)) == len(request_ids),
        "four_methods": set(counts) == set(METHODS),
        "equal_method_counts": all(counts[m] == expected_images*expected_turns
                                   for m in METHODS),
        "successful_status": all(row["status"] == "ok" for row in rows),
        "outer_ttft_positive": all(row["end_to_end_ttft_ms"] > 0 for row in rows),
        "cache_hit_vision_zero": all(
            row["vision_forward_count"] == 0 for row in rows if row["cache_hit"]),
        "new_budget_unit": all(row["budget_unit"] == "visual_kv"
                               for row in rows if row["method_key"] == "ours_kv25"),
        "legacy_budget_unit": all(row["budget_unit"] == "chunk"
                                  for row in rows if row["method_key"] == "ours_chunk25_legacy"),
        "image_artifacts_complete": len(list((run_dir/"image_artifacts"/phase).glob("*.json")))
                                    == expected_images,
    }
    hits = [row for row in rows if row["turn_id"] > 1]
    full_by_request = {(r["image_id"], r["question_id"]): r for r in hits
                       if r["method_key"] == "fullload"}
    summaries = []
    for method in METHODS:
        selected = [r for r in rows if r["method_key"] == method]
        selected_hits = [r for r in selected if r["turn_id"] > 1]
        normal_ratios = []
        total_ratios = []
        for row in selected_hits:
            full = full_by_request[(row["image_id"], row["question_id"])]
            if full["normal_kv_read_bytes"]:
                normal_ratios.append(row["normal_kv_read_bytes"] /
                                     full["normal_kv_read_bytes"])
            if full["ssd_read_bytes"]:
                total_ratios.append(row["ssd_read_bytes"] /
                                    full["ssd_read_bytes"])
        summaries.append({
            "method_id": method, "method": METHOD_META[method]["label"],
            "budget_unit": METHOD_META[method]["budget_unit"],
            "requests": len(selected), "hits": len(selected_hits),
            "quality_all": float(np.mean([r["correct"] for r in selected])),
            "quality_hit": float(np.mean([r["correct"] for r in selected_hits])),
            "hit_ttft_ms": float(np.mean([r["end_to_end_ttft_ms"] for r in selected_hits])),
            "normal_read_mb_per_hit": float(np.mean([
                r["normal_kv_read_bytes"] for r in selected_hits]))/1e6,
            "total_read_mb_per_hit": float(np.mean([
                r["ssd_read_bytes"] for r in selected_hits]))/1e6,
            "hit_content_retention_image_mean": (
                float(np.mean([r["logical_content_retention"]
                               for r in selected_hits]))
                if method in ("fullload", "ours_chunk25_legacy", "ours_kv25")
                else None),
            "hit_content_retention_pooled_tokens": (
                sum(r["attended_content_kv_count"] for r in selected_hits) /
                sum(r["N_content"] for r in selected_hits)
                if method in ("fullload", "ours_chunk25_legacy", "ours_kv25")
                else None),
            "hit_structural_inclusive_retention_image_mean": (
                float(np.mean([r["visual_retention_including_structural"]
                               for r in selected_hits]))
                if method in ("fullload", "ours_chunk25_legacy", "ours_kv25")
                else None),
            "hit_structural_inclusive_retention_pooled_tokens": (
                sum(r["attended_content_kv_count"]+r["N_structural"]
                    for r in selected_hits) /
                sum(r["N_content"]+r["N_structural"] for r in selected_hits)
                if method in ("fullload", "ours_chunk25_legacy", "ours_kv25")
                else None),
            "normal_read_ratio_image_mean": float(np.mean(normal_ratios)),
            "total_read_ratio_image_mean": float(np.mean(total_ratios)),
            "normal_read_ratio_pooled_bytes": (
                sum(r["normal_kv_read_bytes"] for r in selected_hits) /
                sum(full_by_request[(r["image_id"],r["question_id"])][
                    "normal_kv_read_bytes"] for r in selected_hits)),
            "total_read_ratio_pooled_bytes": (
                sum(r["ssd_read_bytes"] for r in selected_hits) /
                sum(full_by_request[(r["image_id"],r["question_id"])][
                    "ssd_read_bytes"] for r in selected_hits)),
            "unused_loaded_real_rows_per_hit": float(np.mean([
                r["unused_loaded_real_rows"] or 0 for r in selected_hits])),
            "unused_loaded_real_mb_per_hit": float(np.mean([
                r["unused_loaded_real_bytes"] or 0 for r in selected_hits]))/1e6,
            "structural_normal_mb_per_hit": float(np.mean([
                r["structural_bytes_in_normal_payload"] or 0
                for r in selected_hits]))/1e6,
            "structural_duplicate_across_files_mb_per_hit": float(np.mean([
                r["structural_duplicate_across_files_bytes"] or 0
                for r in selected_hits]))/1e6,
            "same_file_duplicate_returned_mb_per_hit": float(np.mean([
                r["same_file_duplicate_returned_bytes"] or 0
                for r in selected_hits]))/1e6,
            "padding_returned_mb_per_hit": float(np.mean([
                r["padding_returned_bytes"] or 0
                for r in selected_hits]))/1e6,
            "quality_by_turn": {str(t): float(np.mean([
                r["correct"] for r in selected if r["turn_id"] == t]))
                for t in range(1, expected_turns+1)},
        })
    artifacts = [json.loads(path.read_text(encoding="utf-8"))
                 for path in sorted((run_dir/"image_artifacts"/phase).glob("*.json"))]
    comparisons = Counter(a["new_chunk_comparison"] for a in artifacts)
    overhead = {}
    for method in METHODS:
        t1 = [r for r in rows if r["method_key"] == method and r["turn_id"] == 1]
        activation_ms = [event["activation_ms"] for artifact in artifacts
                         for event in artifact["activation_events"]
                         if event["method"] == method]
        receipt_key = ("ours_chunk25_legacy" if method == "ours_kv25"
                       else method)
        persisted = [artifact["physical_persistence_once"][receipt_key]
                     for artifact in artifacts
                     if receipt_key in artifact["physical_persistence_once"]]
        overhead[method] = {
            "t1_ttft_mean_ms": float(np.mean([
                r["end_to_end_ttft_ms"] for r in t1])),
            "t1_e2e_mean_ms": float(np.mean([
                r["request_e2e_ms"] for r in t1])),
            "persistence_mean_ms_independent_deployment": (
                float(np.mean([p["timing_ms"]["persist_ms"] for p in persisted]))
                if persisted else None),
            "physical_persistence_writes_in_this_run": (
                0 if method == "ours_kv25" else len(persisted)),
            "persistence_attribution": (
                "shared_measured_image_only_write" if method == "ours_kv25"
                else "own_measured_write" if persisted else "none"),
            "activation_mean_ms": (float(np.mean(activation_ms))
                                   if activation_ms else None),
        }
    analysis = {
        "overhead_by_method": overhead,
        "same_selected_content_set_images": sum(a["old_new_same_content_set"]
                                                for a in artifacts),
        "different_selected_content_set_images": sum(not a["old_new_same_content_set"]
                                                     for a in artifacts),
        "new_chunk_comparison_images": dict(comparisons),
        "paired_hit_quality_new_minus_old": image_cluster_ci(rows, "correct"),
        "paired_hit_ttft_ms_new_minus_old": image_cluster_ci(
            rows, "end_to_end_ttft_ms"),
    }
    validation = {"schema_version": SCHEMA, "dataset": phase,
                  "passed": all(checks.values()), "checks": checks,
                  "scope": ("one_image_runner_preflight" if preflight_one_image
                            else "frozen_40_image_pilot" if phase == "gqa"
                            else "three_to_five_image_mt_smoke"),
                  "full_gqa_pilot_valid": (all(checks.values()) and phase == "gqa"
                                           and not preflight_one_image),
                  "expected_rows": expected_rows, "observed_rows": len(rows)}
    atomic_json(results_dir / f"{phase}_validation.json", validation)
    atomic_json(results_dir / f"{phase}_summary.json",
                {"methods": summaries, "analysis": analysis})
    with (results_dir/f"{phase}_summary.csv").open("x", newline="",
                                                    encoding="utf-8") as handle:
        fields = [k for k in summaries[0] if k != "quality_by_turn"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            writer.writerow({k: item[k] for k in fields})
    return {"validation": validation, "methods": summaries, "analysis": analysis}


def report(phases: Mapping[str, Any], run_dir: Path, results_dir: Path,
           manifest: Mapping[str, Any]) -> None:
    lines = ["# LLaVA KV25 migration pilot", "",
             f"Run: `{run_dir}`", "",
             (("GQA runner preflight uses only the first frozen image; it is not the 40-image pilot. "
               if manifest["preflight_one_image"] else
               "The GQA pilot uses 40 fixed images and questions[4:10]. ")
              + f"The MT-GQA smoke selection is {manifest['mt_smoke']['selected_images']} distinct images; any executed MT phase is not the full 4,061-dialogue experiment."),
             "",
             "All T1 requests use normal full-image inference. The old and new Ours hit paths share one immutable image-only physical store but use separate serving contexts. One physical write was measured; the same measured cost is attributed separately to either method under independent deployment.", ""]
    for phase, data in phases.items():
        lines.extend((f"## {phase}", "",
                      f"Validation: **{'VALID' if data['validation']['passed'] else 'INVALID'}**", "",
                      "| Method | Budget unit | Hit content retention | Normal read MB/hit | Total read MB/hit | Hit TTFT ms | Hit quality |", 
                      "|---|---|---:|---:|---:|---:|---:|"))
        for item in data["methods"]:
            retention = item["hit_content_retention_image_mean"]
            lines.append("| {method} | {budget_unit} | {ret} | {normal_read_mb_per_hit:.3f} | {total_read_mb_per_hit:.3f} | {hit_ttft_ms:.2f} | {quality_hit:.3f} |".format(
                **item, ret="N/A" if retention is None else f"{retention:.4f}"))
        a = data["analysis"]
        lines.extend(("", "All-question quality and each turn's quality:", "",
                      "| Method | All questions | T1 | T2 | T3 | T4 | T5 | T6 |",
                      "|---|---:|---:|---:|---:|---:|---:|"))
        for item in data["methods"]:
            turn_values = [item["quality_by_turn"].get(str(t))
                           for t in range(1, 7)]
            rendered = ["N/A" if value is None else f"{value:.3f}"
                        for value in turn_values]
            lines.append("| " + " | ".join([item["method"],
                f"{item['quality_all']:.3f}", *rendered]) + " |")
        lines.extend(("", "Hit retention and physical reads use distinct denominators. Image mean is the mean of request-level ratios; pooled is the ratio of summed tokens or bytes.", "",
                      "| Method | Content mean / pooled | Visual incl. structural mean / pooled | Normal read / FullLoad mean / pooled | Total read / FullLoad mean / pooled | Unused real rows / MB per hit | Structural normal MB/hit | Cross-file structural duplicate MB/hit | Same-file duplicate MB/hit | Padding MB/hit |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"))
        def pair(left, right):
            return ("N/A" if left is None or right is None
                    else f"{left:.4f} / {right:.4f}")
        for item in data["methods"]:
            lines.append("| " + " | ".join([
                item["method"],
                pair(item["hit_content_retention_image_mean"],
                     item["hit_content_retention_pooled_tokens"]),
                pair(item["hit_structural_inclusive_retention_image_mean"],
                     item["hit_structural_inclusive_retention_pooled_tokens"]),
                pair(item["normal_read_ratio_image_mean"],
                     item["normal_read_ratio_pooled_bytes"]),
                pair(item["total_read_ratio_image_mean"],
                     item["total_read_ratio_pooled_bytes"]),
                f"{item['unused_loaded_real_rows_per_hit']:.2f} / {item['unused_loaded_real_mb_per_hit']:.3f}",
                f"{item['structural_normal_mb_per_hit']:.3f}",
                f"{item['structural_duplicate_across_files_mb_per_hit']:.3f}",
                f"{item['same_file_duplicate_returned_mb_per_hit']:.3f}",
                f"{item['padding_returned_mb_per_hit']:.3f}",
            ]) + " |")
        lines.extend(("", "T1 and persistence are outside cache-hit means; context activation is outside request TTFT.", "",
                      "| Method | T1 TTFT ms | T1 E2E ms | Persistence ms attributed independently | Physical writes in run | Activation ms |",
                      "|---|---:|---:|---:|---:|---:|"))
        for method in METHODS:
            value = a["overhead_by_method"][method]
            persist = value["persistence_mean_ms_independent_deployment"]
            activation = value["activation_mean_ms"]
            lines.append(f"| {METHOD_META[method]['label']} | {value['t1_ttft_mean_ms']:.2f} | {value['t1_e2e_mean_ms']:.2f} | "
                         + ("N/A" if persist is None else f"{persist:.2f}")
                         + f" | {value['physical_persistence_writes_in_this_run']} | "
                         + ("N/A" if activation is None else f"{activation:.2f}")
                         + " |")
        lines.extend(("", f"Old/new same selected set: {a['same_selected_content_set_images']} images; different: {a['different_selected_content_set_images']}.",
                      f"New chunk count compared with old: {a['new_chunk_comparison_images']}.",
                      f"Paired hit quality (new − old): {a['paired_hit_quality_new_minus_old']}.",
                      f"Paired hit TTFT ms (new − old): {a['paired_hit_ttft_ms_new_minus_old']}.", ""))
    lines.extend(("## Timing and limits", "",
                  "`end_to_end_ttft_ms` starts before prompt and token preparation and ends after the first token decision and CUDA synchronization. OS page-cache conditioning via `posix_fadvise_DONTNEED` is outside this timer; it does not guarantee a cold SSD controller or NAND. Persistence and context activation are separately recorded in each image artifact. Existing historic results were not relabeled or reused as same-run latency.", "",
                  f"Frozen manifest SHA256: `{canonical_hash(manifest)}`.", ""))
    write_text(results_dir / "REPORT.md", "\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("gqa", "mt_smoke", "both"),
                        default="both")
    parser.add_argument("--mt-images", type=int, choices=(3, 4, 5), default=4)
    parser.add_argument("--preflight-one-image", action="store_true",
                        help="execute only the first frozen GQA image; never a valid 40-image pilot")
    parser.add_argument("--gpu-gate", type=Path, required=True,
                        help="GPU correctness receipt containing GPU_CORRECTNESS: PASS")
    parser.add_argument("--run-id", default=None,
                        help="unique llava_kv25_migration_<UTC> directory name")
    args = parser.parse_args()
    if args.preflight_one_image and args.phase != "gqa":
        parser.error("--preflight-one-image requires --phase gqa")
    gate = require_gpu_gate(args.gpu_gate)
    gqa, gqa_workload = frozen_gqa()
    if args.preflight_one_image:
        gqa = gqa[:1]
    mt, mt_workload = frozen_mt(args.mt_images)
    gpu = gpu_inventory()
    run_id = args.run_id or "llava_kv25_migration_" + time.strftime(
        "%Y%m%dT%H%M%SZ", time.gmtime())
    if not run_id.startswith("llava_kv25_migration_") or "/" in run_id or ".." in run_id:
        raise ValueError("unsafe run ID")
    run_dir = ROOT / "runs" / run_id
    results_dir = ROOT / "results" / run_id
    if run_dir.exists() or results_dir.exists():
        raise FileExistsError("run/results directory already exists")
    run_dir.mkdir(parents=True, exist_ok=False)
    results_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": SCHEMA, "run_id": run_id, "phase": args.phase,
        "preflight_one_image": bool(args.preflight_one_image),
        "executed_gqa_images": len(gqa) if args.phase in ("gqa", "both") else 0,
        "seed": SEED, "methods": METHOD_META, "model": MODEL_ID,
        "model_config": {"load_4bit": LOAD_4BIT, "compute_dtype": str(COMPUTE_DTYPE),
                         "attention_backend": ATTN_IMPL, "chunk_size": CHUNK_SIZE,
                         "max_new_tokens": 16},
        "gqa": gqa_workload, "mt_smoke": mt_workload,
        "gpu_correctness_gate": gate, "gpu_preflight": gpu,
        "source_sha256": {p: sha256_file(ROOT/p) for p in SOURCE_PATHS},
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(), "python": sys.version,
        "torch": torch.__version__, "cuda": torch.version.cuda,
    }
    manifest_hash = canonical_hash(manifest)
    atomic_json(run_dir/"manifest.json", manifest)
    atomic_json(results_dir/"manifest.json", manifest)
    config = {"schema_version": SCHEMA, "run_id": run_id, "phase": args.phase,
              "methods": METHOD_META, "model": MODEL_ID,
              "max_new_tokens": 16, "seed": SEED,
              "timing": "outer request start through synchronized first token",
              "page_cache_conditioning": "posix_fadvise_DONTNEED outside timer",
              "store_policy": "run-local per-image full stores retained; one immutable image-only physical store shared by old/new",
              "gpu_gate": gate}
    config_hash = canonical_hash(config)
    atomic_json(run_dir/"config.json", config)
    atomic_json(results_dir/"config.json", config)
    try:
        runner = LlavaRunner().load()
        server = Server(runner, ratio=0.25, probe=PROBE_HEADS,
                        max_new_tokens=16)
        warmup = QA._warmup(runner, server)
        atomic_json(run_dir/"warmup.json", warmup)
        phases = {}
        for phase in (("gqa", "mt_smoke") if args.phase == "both" else (args.phase,)):
            entries = gqa if phase == "gqa" else mt
            rows = []
            raw_path = run_dir/f"{phase}_raw.jsonl"
            with raw_path.open("x", encoding="utf-8") as raw_handle:
                for image_index, entry in enumerate(entries):
                    rows.extend(image_session(
                        phase=phase, entry=entry, image_index=image_index,
                        runner=runner, server=server, run_dir=run_dir,
                        raw_handle=raw_handle, config_hash=config_hash,
                        manifest_hash=manifest_hash))
                    print(f"{phase}: {image_index+1}/{len(entries)} images complete",
                          flush=True)
            phases[phase] = summarize(
                rows, phase, run_dir, results_dir,
                preflight_one_image=args.preflight_one_image)
            shutil.copyfile(raw_path, results_dir/f"{phase}_raw.jsonl")
            atomic_json(run_dir/f"{phase}_validation.json",
                        phases[phase]["validation"])
            atomic_json(results_dir/f"{phase}_raw_artifact.json", {
                "path": str(raw_path), "sha256": sha256_file(raw_path),
                "rows": len(rows)})
        report(phases, run_dir, results_dir, manifest)
        all_pass = all(value["validation"]["passed"] for value in phases.values())
        atomic_json(run_dir/"validation.json", {
            "schema_version": SCHEMA, "passed": all_pass,
            "phases": {k: v["validation"] for k, v in phases.items()}})
        atomic_json(results_dir/"validation.json", {
            "schema_version": SCHEMA, "passed": all_pass,
            "phases": {k: v["validation"] for k, v in phases.items()}})
        if all_pass:
            write_text(run_dir/"COMPLETED", "validated\n")
        return 0 if all_pass else 1
    except Exception as error:
        atomic_json(run_dir/"failure.json", {
            "status": "INVALID", "exception_type": type(error).__name__,
            "exception": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())

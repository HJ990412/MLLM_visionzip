#!/usr/bin/env python3
"""Evaluate one MT-GQA history protocol with four Visual-KV methods.

This is a protocol-scoped, image-sharded GPU evaluator.  One invocation runs
either ``gold_history`` or ``generated_history`` over the frozen
``MT-GQA-reconstructed`` workload.  Gold history teacher-forces prior answers;
generated history feeds each method only its own exact earlier predictions.

The durable unit is one image artifact.  While an image is active, the runner
builds two temporary stores from ordinary first-dialog Turn-1 requests:

* canonical raster K/V + three probe heads + visual hidden states, captured
  by FullLoad's own first Turn-1 request, for FullLoad and QA-Chunk25; and
* image-only saliency-repacked K/V for Ours25.

Both stores are removed only after the complete, content-hashed image artifact
has been atomically published.  A crash therefore redoes at most one image,
while resume skips only artifacts whose identity, request coverage, histories,
method contracts, and content hash validate exactly.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import re
import shutil
import sys
import time
import uuid
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import psutil
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mmimpress.config import (  # noqa: E402
    CHUNK_SIZE, COMPUTE_DTYPE, MODEL_ID, PROBE_HEADS,
)
from mmimpress.cvpr25 import budget_chunk_count  # noqa: E402
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


SCHEMA_VERSION = "mt-gqa-4arm-history-shard-v2"
DATASET = "gqa_testdev_balanced_mt3"
BENCHMARK_TYPE = "MT-GQA-reconstructed"
DEFAULT_INDEX = ROOT / "data/mt_gqa/dialogues.json"
EXPECTED_INDEX_SHA256 = (
    "2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224"
)
EXPECTED_WORKLOAD_SHA256 = (
    "0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62"
)
EXPECTED_DIALOGUES = 4_061
EXPECTED_TURNS = 12_183
EXPECTED_IMAGES = 398
SEED = 1234
MAX_NEW_TOKENS = 16
MIN_SHARD_SIZE = 40
MAX_SHARD_SIZE = 60
DEFAULT_SHARD_SIZE = 50
MIN_FREE_AFTER_GIB = 30.0
BUILD_HEADROOM_GIB = 5.0
PROTOCOLS = ("gold_history", "generated_history")
QA_RATER_ALGORITHM_ID = "sparsevlm_visual_text_mean_threshold_v1"

METHOD_KEYS = ("recompute", "fullload", "qa_chunk25", "ours25")
METHODS = {
    "recompute": {
        "method_id": "recompute", "label": "ReComp",
        "retention_ratio": None, "physical_layout": "none",
        "importance_source": "none", "query_dependent": False,
        "online_selection": False,
    },
    "fullload": {
        "method_id": "fullload", "label": "FullLoad",
        "retention_ratio": 1.0, "physical_layout": "raster",
        "importance_source": "none", "query_dependent": False,
        "online_selection": False,
    },
    "qa_chunk25": {
        "method_id": "qa_chunk25", "label": "QA-Chunk25",
        "retention_ratio": 0.25, "physical_layout": "raster",
        "importance_source": "SparseVLM-style query-aware importance",
        "query_dependent": True, "online_selection": True,
        "selection_granularity": "physical_ssd_chunk",
        "chunk_score": "mean_valid_spatial_token_importance",
        "head_reduce": "mean",
        "rater_algorithm_id": QA_RATER_ALGORITHM_ID,
    },
    "ours25": {
        "method_id": "imageonly_prefix25", "label": "Ours25",
        "retention_ratio": 0.25,
        "physical_layout": "visionzip_image_only",
        "importance_source": "image-only Vision Encoder saliency",
        "query_dependent": False, "online_selection": False,
        "selection_granularity": "fixed_physical_prefix",
    },
}

SHORT_ANSWER_INSTRUCTION = (
    "Answer the current question with a single word or short phrase."
)

_MT_BASE = None
_QA_BASE = None


def _load_script(name: str, filename: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


def mt_base():
    """Load the validated MT workload/sharding helpers without running main."""
    global _MT_BASE
    if _MT_BASE is None:
        _MT_BASE = _load_script(
            "_mt_gqa_history_reused_shard", "37_eval_mt_gqa_full_shard.py")
    return _MT_BASE


def qa_base():
    """Load validated query-aware timing/I/O helpers without running main."""
    global _QA_BASE
    if _QA_BASE is None:
        _QA_BASE = _load_script(
            "_mt_gqa_history_reused_qa", "49_eval_query_aware_baseline.py")
    return _QA_BASE


def _runtime_classes():
    from mmimpress.serve import ImageContext, Server
    return ImageContext, Server


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_answer(value: Any) -> str:
    """Frozen strict-GQA normalization used by the prior MT analysis."""
    text = re.sub(r"[^\w\s]", " ", str(value).lower())
    return " ".join(word for word in text.split()
                    if word not in {"a", "an", "the"})


def strict_gqa_score(prediction: Any, gold_answer: Any) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold_answer))


def logical_request_id(protocol: str, dialog_id: str, turn_id: int,
                       method_key: str) -> str:
    if protocol not in PROTOCOLS or method_key not in METHOD_KEYS:
        raise ValueError("invalid logical request identity")
    return f"{protocol}:{dialog_id}:t{int(turn_id)}:{method_key}"


def _dialog_id(dialog: Mapping[str, Any]) -> str:
    return str(dialog.get("dialog_id", dialog.get("dialogue_id", "")))


def _turn(dialog: Mapping[str, Any], turn_id: int) -> Mapping[str, Any]:
    if not isinstance(turn_id, int) or isinstance(turn_id, bool):
        raise ValueError("turn_id must be an integer in [1,3]")
    turns = dialog.get("turns")
    if not isinstance(turns, list) or len(turns) != 3:
        raise ValueError("MT-GQA dialogue must contain exactly three turns")
    if turn_id not in (1, 2, 3):
        raise ValueError("turn_id must be in [1,3]")
    for expected, item in enumerate(turns, 1):
        if not isinstance(item, Mapping) or int(item.get("turn_id", -1)) != expected:
            raise ValueError("MT-GQA turns must be ordered and 1-indexed")
    return turns[turn_id - 1]


def _gold(turn: Mapping[str, Any]) -> str:
    answers = turn.get("answers")
    if not isinstance(answers, list) or len(answers) != 1:
        raise ValueError("MT-GQA turn must have exactly one gold answer")
    value = str(answers[0]).strip()
    if not value:
        raise ValueError("gold answer must be nonempty")
    return value


def method_order(global_dialog_ordinal: int, seed: int = SEED) -> tuple[str, ...]:
    """Balanced order; seed validates the frozen run but does not phase-shift."""
    if int(seed) != SEED:
        raise ValueError("the frozen method-order contract uses seed 1234")
    return deterministic_method_rotation(
        METHOD_KEYS, int(global_dialog_ordinal), 0)


def expected_request_counts(n_dialogues: int) -> dict[str, int]:
    n = int(n_dialogues)
    if n <= 0:
        raise ValueError("n_dialogues must be positive")
    turns = 3 * n
    per_protocol = turns * len(METHOD_KEYS)
    return {
        "dialogues": n,
        "turns_per_protocol": turns,
        "requests_per_method_per_protocol": turns,
        "turn1_requests_per_method_per_protocol": n,
        "cache_hit_requests_per_method_per_protocol": 2 * n,
        "requests_per_protocol": per_protocol,
        "requests_both_protocols": 2 * per_protocol,
        "main_t2_t3_requests_both_protocols": (
            2 * n * len(METHOD_KEYS) * len(PROTOCOLS)),
        "stored_visual_kv_hits_both_protocols": (
            2 * n * 3 * len(PROTOCOLS)),
    }


def causal_history(
    dialog: Mapping[str, Any],
    turn_id: int,
    protocol: str,
    *,
    method_key: str,
    generated_predictions: Mapping[int, Any] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Render exactly the causally available history and its provenance."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown history protocol: {protocol}")
    if method_key not in METHOD_KEYS:
        raise ValueError(f"unknown method: {method_key}")
    _turn(dialog, turn_id)
    generated = dict(generated_predictions or {})
    if any(int(key) >= int(turn_id) for key in generated):
        raise ValueError("generated state contains current/future turn")
    lines: list[str] = []
    entries: list[dict[str, Any]] = []
    for prior_id in range(1, int(turn_id)):
        prior = _turn(dialog, prior_id)
        question = str(prior.get("question", "")).strip()
        if not question:
            raise ValueError("history question must be nonempty")
        if protocol == "gold_history":
            answer = _gold(prior)
            source = "gold"
            source_method = None
        else:
            if prior_id not in generated:
                raise ValueError(
                    f"missing generated T{prior_id} for {method_key}")
            source_row = generated[prior_id]
            if isinstance(source_row, Mapping):
                if "prediction" not in source_row:
                    raise ValueError("generated lineage omits prediction")
                answer = str(source_row["prediction"])
                source_logical_id = source_row.get("logical_request_id")
                source_physical_id = source_row.get("physical_execution_id")
                if not source_logical_id or not source_physical_id:
                    raise ValueError("generated lineage omits source request IDs")
            else:
                answer = str(source_row)
                source_logical_id = None
                source_physical_id = None
            # Server._decode already strips predictions.  Reject accidental
            # normalization/substitution while allowing a legitimate empty
            # decoded string to remain exact.
            source = "generated"
            source_method = method_key
        if protocol == "gold_history":
            source_logical_id = f"gold:{prior.get('question_id', '')}"
            source_physical_id = None
        lines.extend((f"Q{prior_id}: {question}",
                      f"A{prior_id}: {answer}"))
        entries.append({
            "turn_id": prior_id,
            "question_id": str(prior.get("question_id", "")),
            "question": question,
            "answer": answer,
            "answer_source": source,
            "source_method_key": source_method,
            "source_logical_request_id": source_logical_id,
            "source_physical_execution_id": source_physical_id,
        })
    return "\n".join(lines), entries


def render_causal_prompt(
    dialog: Mapping[str, Any],
    turn_id: int,
    protocol: str,
    *,
    method_key: str,
    generated_predictions: Mapping[int, Any] | None = None,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Canonical MT-GQA prompt with gold or method-local generated history."""
    current = _turn(dialog, turn_id)
    question = str(current.get("question", "")).strip()
    if not question:
        raise ValueError("current question must be nonempty")
    history, entries = causal_history(
        dialog, turn_id, protocol, method_key=method_key,
        generated_predictions=generated_predictions)
    body: list[str] = []
    if history:
        body.extend((history, ""))
    body.extend((
        f"Current question Q{turn_id}: {question}",
        f"{SHORT_ANSWER_INSTRUCTION} ASSISTANT:",
    ))
    return "USER: <image>\n" + "\n".join(body), history, entries


def _artifact_body_hash(value: Mapping[str, Any]) -> str:
    return stable_json_sha256({key: item for key, item in value.items()
                               if key != "artifact_content_sha256"})


def _image_artifact_path(run_dir: Path, image_id: str) -> Path:
    if not image_id or Path(image_id).name != image_id:
        raise ValueError(f"unsafe image id: {image_id!r}")
    return run_dir / "images" / f"{image_id}.json"


def _expected_row_keys(dialogs: Sequence[Mapping[str, Any]]) -> list[tuple]:
    keys = []
    for dialog in dialogs:
        did = _dialog_id(dialog)
        ordinal = int(dialog["global_dialog_ordinal"])
        for turn_id in (1, 2, 3):
            for method in method_order(ordinal):
                keys.append((did, turn_id, method))
    return keys


def _hash_tensor(value: torch.Tensor) -> str:
    return qa_base()._hash_tensor(value)


def _hash_tensor_mapping(values: Mapping[str, Any]) -> str:
    return qa_base()._hash_tensor_mapping(values)


def _image_input_hash(values: Mapping[str, Any]) -> str:
    return qa_base()._image_input_hash(values)


def _suffix_from_prompt(runner, prompt: str) -> torch.Tensor:
    tokenized = runner.processor.tokenizer(prompt, return_tensors="pt")
    return qa_base()._suffix_from_tokenized(runner, tokenized)


def _json_result(result: Mapping[str, Any], method_key: str,
                 full_visual_bytes: int | None) -> dict[str, Any]:
    """Flatten validated serving counters and expose QA-Chunk stage aliases."""
    value = qa_base()._json_result(
        dict(result), method_key, int(full_visual_bytes or 0))
    for field in (
        "chunk_aggregation_ms", "topk_chunk_ms",
        "selector_decision_host_wall_ms", "selected_chunk_io_ms",
        "normal_kv_read_ms", "separator_read_ms",
    ):
        value.setdefault(field, 0.0)
    if method_key == "qa_chunk25":
        value["topk_chunk_ms"] = float(
            value.get("topk_chunk_ms", value.get("topk_ms", 0.0)) or 0.0)
        value["topk_ms"] = value["topk_chunk_ms"]
    io_detail = value.get("io_detail", {})
    if isinstance(io_detail, Mapping):
        normal_raw_ms = 1e3 * sum(float(
            io_detail.get(kind, {}).get("seconds", 0.0) or 0.0)
            for kind in ("k", "v")
            if isinstance(io_detail.get(kind, {}), Mapping))
        separator_raw_ms = 1e3 * float(
            io_detail.get("sep", {}).get("seconds", 0.0) or 0.0
            if isinstance(io_detail.get("sep", {}), Mapping) else 0.0)
        if (int(value.get("normal_kv_read_bytes", 0) or 0) > 0
                and float(value.get("normal_kv_read_ms", 0.0) or 0.0) == 0.0):
            value["normal_kv_read_ms"] = normal_raw_ms
        if (int(value.get("separator_read_bytes", 0) or 0) > 0
                and float(value.get("separator_read_ms", 0.0) or 0.0) == 0.0):
            value["separator_read_ms"] = separator_raw_ms
        if (method_key == "ours25"
                and int(value.get("normal_kv_read_bytes", 0) or 0) > 0):
            value["selected_chunk_io_ms"] = float(
                value.get("normal_kv_read_ms", normal_raw_ms) or 0.0)
    if method_key == "fullload" and int(value.get("ssd_read_bytes", 0)) > 0:
        # FullLoad performs one contiguous full-file read for K and one for V
        # at every decoder layer.  The generic LayerSelector does not expose
        # run counts, so derive the exact per-layer topology from its measured
        # normal-KV preads instead of letting analysis misreport zero runs.
        normal_preads = int(value.get("normal_kv_preads", 0) or 0)
        if normal_preads <= 0 or normal_preads % 2:
            raise ValueError(
                "FullLoad normal K/V preads must be a positive even count")
        layers = normal_preads // 2
        value["contiguous_runs_per_layer"] = [1] * layers
        value["contiguous_runs_per_layer_mean"] = 1.0
        value["normal_kv_read_ms"] = float(value.get("ssd_read_ms", 0.0))
        value["selected_chunk_io_ms"] = None
    value["selected_chunk_payload_read_bytes"] = int(
        value.get("normal_kv_read_bytes", 0) or 0)
    value["pread_count"] = int(value.get("ssd_preads", 0) or 0)
    return value


def _run_pixel_request(
    runner,
    server,
    image,
    prompt_factory: Callable[[], tuple[str, str, list[dict[str, Any]]]],
    *,
    capture_kind: str = "none",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ordinary multimodal request with optional same-forward store capture."""
    if capture_kind not in {"none", "raster", "ours"}:
        raise ValueError(f"invalid capture kind: {capture_kind}")
    capture_cache = capture_kind in {"raster", "ours"}
    vision = VisionForwardCapture(
        runner, capture_saliency=(capture_kind == "ours"))
    hidden_capture = None
    with vision:
        torch.cuda.synchronize()
        request_started = time.perf_counter()
        prompt_started = time.perf_counter()
        prompt, history, history_entries = prompt_factory()
        prompt_ms = (time.perf_counter() - prompt_started) * 1e3
        enc_cpu, processor_timing = qa_base()._combined_processor(
            runner, image, prompt)
        v_start, v_num = runner.visual_span(enc_cpu["input_ids"])
        if capture_kind == "raster":
            hidden_capture = DecoderVisualHiddenCapture(
                runner, v_start, v_num)
        stack = ExitStack()
        if hidden_capture is not None:
            stack.enter_context(hidden_capture)
        try:
            h2d_started = time.perf_counter()
            enc_device = runner.to_device(enc_cpu)
            torch.cuda.synchronize()
            h2d_ms = (time.perf_counter() - h2d_started) * 1e3
            result = server.recompute(
                enc_device, return_past_key_values=capture_cache)
            returned = time.perf_counter()
        finally:
            stack.close()
    phases = {
        "prompt_build_ms": float(prompt_ms),
        "tokenization_ms": float(processor_timing["tokenization_ms"]),
        "image_preprocess_ms": float(
            processor_timing["image_preprocess_ms"]),
        "input_prepare_ms": float(processor_timing["input_prepare_ms"]),
        "input_h2d_ms": float(h2d_ms),
        "processor_total_ms": float(
            processor_timing["processor_total_ms"]),
    }
    result.update(qa_base()._timing_fields(
        result, request_started, returned, phases))
    result.update({
        "vision_ms": float(vision.stats()["vision_ms"]),
        "vision_forward_count": int(vision.call_count),
        "page_cache_conditioning_method": "not_applicable_pixels",
        "page_cache_conditioning_ms": 0.0,
        "page_cache_conditioning_excluded_from_ttft": True,
    })
    suffix = _suffix_from_prompt(runner, prompt)
    combined_suffix = mt_base()._combined_suffix(
        runner, enc_cpu["input_ids"])
    if not torch.equal(combined_suffix.cpu(), suffix.cpu()):
        raise AssertionError(
            "processor-expanded pixel suffix != tokenizer stored-KV suffix")
    history_ids = ([] if not history else runner.processor.tokenizer(
        history, add_special_tokens=False).input_ids)
    diagnostics = {
        "prompt": prompt,
        "history": history,
        "history_entries": history_entries,
        "history_token_count": len(history_ids),
        "input_token_count": int(suffix.numel()),
        "prompt_sha256": sha256_text(prompt),
        "history_sha256": sha256_text(history),
        "suffix_ids_sha256": _hash_tensor(suffix),
        "combined_suffix_ids_sha256": _hash_tensor(combined_suffix),
        "input_tensors_sha256": _hash_tensor_mapping(enc_cpu),
        "input_ids_sha256": _hash_tensor(enc_cpu["input_ids"]),
        "image_input_sha256": _image_input_hash(enc_cpu),
        "enc_cpu": enc_cpu,
        "vision_capture": vision,
        "hidden_capture": hidden_capture,
    }
    del enc_device
    return result, diagnostics


def _dispatch_stored(server, method_key: str, context, suffix_ids,
                     image_id: str):
    if method_key == "fullload":
        return server.request(
            context, mode="fullload", cold=False, suffix_ids=suffix_ids)
    if method_key == "qa_chunk25":
        return server.request_qa_chunk(
            context, cold=False, suffix_ids=suffix_ids)
    if method_key == "ours25":
        return server.request_cvpr25(
            context, static=None, budget=0.25, mode="prefix",
            sep_policy="sidecar", cold=False, seed=SEED,
            image_id=image_id, suffix_ids=suffix_ids,
            expected_prefix_layout="visionzip_image_only")
    raise ValueError(f"not a stored method: {method_key}")


def _run_stored_request(
    runner,
    server,
    context,
    prompt_factory: Callable[[], tuple[str, str, list[dict[str, Any]]]],
    *,
    method_key: str,
    image_id: str,
    full_visual_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cold buffered-SSD request with full outer TTFT timing."""
    with qa_base()._NoVisionForward(runner) as guard:
        conditioned_at = time.perf_counter()
        context.reader.drop_all()
        conditioned_done = time.perf_counter()
        torch.cuda.synchronize()
        request_started = time.perf_counter()
        prompt_started = time.perf_counter()
        prompt, history, history_entries = prompt_factory()
        prompt_ms = (time.perf_counter() - prompt_started) * 1e3
        token_started = time.perf_counter()
        tokenized = runner.processor.tokenizer(prompt, return_tensors="pt")
        token_ms = (time.perf_counter() - token_started) * 1e3
        prepare_started = time.perf_counter()
        suffix_cpu = qa_base()._suffix_from_tokenized(runner, tokenized)
        prepare_ms = (time.perf_counter() - prepare_started) * 1e3
        h2d_started = time.perf_counter()
        suffix_device = suffix_cpu.to(runner.model.device)
        torch.cuda.synchronize()
        h2d_ms = (time.perf_counter() - h2d_started) * 1e3
        result = _dispatch_stored(
            server, method_key, context, suffix_device, image_id)
        returned = time.perf_counter()
    phases = {
        "prompt_build_ms": float(prompt_ms),
        "tokenization_ms": float(token_ms),
        "image_preprocess_ms": 0.0,
        "input_prepare_ms": float(prepare_ms),
        "input_h2d_ms": float(h2d_ms),
        "processor_total_ms": None,
    }
    result.update(qa_base()._timing_fields(
        result, request_started, returned, phases))
    result.update({
        "vision_forward_count": int(guard.calls),
        "page_cache_conditioning_started_at_s": float(conditioned_at),
        "page_cache_conditioning_finished_at_s": float(conditioned_done),
        "page_cache_conditioning_ms": float(
            (conditioned_done - conditioned_at) * 1e3),
        "page_cache_conditioning_method": "posix_fadvise_DONTNEED",
        "page_cache_conditioning_excluded_from_ttft": True,
    })
    history_ids = ([] if not history else runner.processor.tokenizer(
        history, add_special_tokens=False).input_ids)
    diagnostics = {
        "prompt": prompt,
        "history": history,
        "history_entries": history_entries,
        "history_token_count": len(history_ids),
        "input_token_count": int(suffix_cpu.numel()),
        "prompt_sha256": sha256_text(prompt),
        "history_sha256": sha256_text(history),
        "suffix_ids_sha256": _hash_tensor(suffix_cpu),
    }
    # Keep the serving result raw here.  `_make_row` performs the one and only
    # flattening/accounting pass, matching the pixel path and preventing a
    # derived I/O document from being treated as a fresh server result.
    return result, diagnostics


def _store_manifest(root: Path, persisted: Mapping[str, Any],
                    meta: Mapping[str, Any]) -> dict[str, Any]:
    file_sizes = {str(key): int(value)
                  for key, value in persisted["file_sizes"].items()}
    file_hashes = dict(persisted.get("hashes", {}).get("files_sha256", {}))
    meta_hash = file_hashes.get("meta.json")
    if not meta_hash:
        meta_hash = sha256_file(root / "meta.json")
    manifest = {
        "physical_layout": str(meta["physical_layout"]),
        "image_id": str(meta["image_id"]),
        "meta_sha256": str(meta_hash),
        "tree_sha256": persisted.get("hashes", {}).get("tree_sha256"),
        "prefix_kv_sample_sha256": persisted.get("hashes", {}).get(
            "prefix_kv_sample_sha256"),
        "file_sizes": file_sizes,
        "n_files": len(file_sizes),
        "total_store_bytes": int(sum(file_sizes.values())),
        "visual_kv_bytes": int(meta["bytes_visual_kv"]),
        "probe_sidecar_bytes": int(meta["bytes_probe_sidecar"]),
        "separator_sidecar_bytes": int(meta["bytes_separator_sidecar"]),
        "num_layers": int(meta["num_layers"]),
        "num_heads": int(meta["num_heads"]),
        "probe_heads": int(meta["probe_heads"]),
        "chunk_size": int(meta["chunk_size"]),
        "n_chunks_per_layer": int(meta["n_chunks_per_layer"]),
        "v_token_num": int(meta["v_token_num"]),
        "n_spatial": int(meta["n_spatial"]),
        "permutation_sha256": meta.get("permutation_sha256"),
        "identity_permutation_sha256": meta.get(
            "identity_permutation_sha256"),
        "separator_policy": meta.get("separator_policy"),
        "capture_provenance_validated": bool(
            meta.get("capture_provenance_validated", True)),
    }
    if str(meta["physical_layout"]) == "visionzip_image_only":
        layout = torch.load(
            root / "visionzip_layout.pt", map_location="cpu",
            weights_only=False)
        stored_to_original = [int(value) for value in torch.as_tensor(
            layout["stored_to_original"]).reshape(-1).tolist()]
        if len(stored_to_original) != int(meta["v_token_num"]):
            raise ValueError("Ours layout permutation length mismatch")
        k_chunks = budget_chunk_count(int(meta["n_chunks_per_layer"]), 0.25)
        prefix_rows = min(
            len(stored_to_original), k_chunks * int(meta["chunk_size"]))
        original_ids = stored_to_original[:prefix_rows]
        manifest.update({
            "selected_prefix_chunk_ids": list(range(k_chunks)),
            "selected_prefix_physical_row_count": prefix_rows,
            "selected_prefix_original_token_ids": original_ids,
            "selected_prefix_original_token_ids_sha256":
                stable_json_sha256(original_ids),
            "stored_to_original_sha256": stable_json_sha256(
                stored_to_original),
        })
    return manifest


def _persistence_summary(persisted: Mapping[str, Any], *,
                         source_dialog_id: str, source_method_key: str,
                         source_execution_id: str) -> dict[str, Any]:
    return {
        "source_dialog_id": source_dialog_id,
        "source_turn_id": 1,
        "source_method_key": source_method_key,
        "source_execution_id": source_execution_id,
        "capture_from_same_answer_forward": True,
        "store_build_count": 1,
        "timing_ms": dict(persisted["timing_ms"]),
        "bytes": dict(persisted["bytes"]),
        "durability": dict(persisted["durability"]),
        "hashes": dict(persisted["hashes"]),
    }


def _make_row(
    *,
    protocol: str,
    workload: Mapping[str, Any],
    dialog: Mapping[str, Any],
    turn: Mapping[str, Any],
    method_key: str,
    order: Sequence[str],
    order_position: int,
    result: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    execution_id: str,
    dialogue_session_id: str,
    context_instance_id: str | None,
    full_visual_bytes: int | None,
    raster_manifest: Mapping[str, Any] | None,
    ours_manifest: Mapping[str, Any] | None,
    persistence_source: str | None,
) -> dict[str, Any]:
    turn_id = int(turn["turn_id"])
    gold_answer = _gold(turn)
    prediction = str(result["answer"])
    cache_hit = turn_id >= 2 and method_key != "recompute"
    flattened = _json_result(result, method_key, full_visual_bytes)
    flattened.pop("answer", None)
    selected_ids = flattened.get("selected_chunk_ids_per_layer")
    selected_hash = (stable_json_sha256(selected_ids)
                     if selected_ids is not None else None)
    store_kind = (("image_only" if method_key == "ours25" else "raster")
                  if cache_hit else None)
    store_manifest = (ours_manifest if store_kind == "image_only"
                      else raster_manifest if store_kind == "raster"
                      else None)
    history_source = ("gold" if protocol == "gold_history"
                      else "generated")
    logical_id = logical_request_id(
        protocol, _dialog_id(dialog), turn_id, method_key)
    row = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "dataset": DATASET,
        "benchmark_type": workload["benchmark_type"],
        "protocol": protocol,
        "history_source": history_source,
        "history_policy": (
            "gold_teacher_forced" if protocol == "gold_history"
            else "method_local_generated"),
        "official_benchmark_identity_claimed": False,
        "benchmark_disclaimer": workload["benchmark_disclaimer"],
        "dialogues_file_sha256": workload["dialogues_file_sha256"],
        "source_full_workload_sha256": workload[
            "source_full_workload_sha256"],
        "selected_workload_sha256": workload[
            "selected_workload_sha256"],
        "model_revision": workload.get("model_revision"),
        "dialog_id": _dialog_id(dialog),
        "dialogue_id": _dialog_id(dialog),
        "global_dialog_ordinal": int(dialog["global_dialog_ordinal"]),
        "image_id": str(dialog["image_id"]),
        "turn_id": turn_id,
        "turn": turn_id,
        "question_id": str(turn["question_id"]),
        "question": str(turn["question"]),
        "gold": [gold_answer],
        "gold_answer": gold_answer,
        "method_key": method_key,
        "method": METHODS[method_key]["label"],
        **METHODS[method_key],
        "method_order": list(order),
        "method_order_position": int(order_position),
        "execution_id": execution_id,
        "logical_request_id": logical_id,
        "physical_execution_id": execution_id,
        "dialogue_session_id": dialogue_session_id,
        "context_instance_id": context_instance_id,
        "gpu_request_cache_fresh": True,
        "text_kv_reused_from_prior_turn": False,
        "request_path": ("stored_visual_kv" if cache_hit
                         else "normal_multimodal_pixel"),
        "execution_mode": ("stored_visual_kv" if cache_hit
                           else "normal_multimodal_pixel"),
        "cache_hit": bool(cache_hit),
        "used_by_request": bool(cache_hit),
        "history_text": str(diagnostics["history"]),
        "history_entries": list(diagnostics["history_entries"]),
        "history_answers": [str(item["answer"])
                            for item in diagnostics["history_entries"]],
        "history_source_request_ids": [
            str(item["source_logical_request_id"])
            for item in diagnostics["history_entries"]
        ],
        "history_source_physical_execution_ids": [
            item["source_physical_execution_id"]
            for item in diagnostics["history_entries"]
        ],
        "history_turn_ids": [int(item["turn_id"])
                             for item in diagnostics["history_entries"]],
        "history_token_count": int(diagnostics["history_token_count"]),
        "history_text_tokens": int(diagnostics["history_token_count"]),
        "input_token_count": int(diagnostics["input_token_count"]),
        "suffix_tokens": int(diagnostics["input_token_count"]),
        "prompt": str(diagnostics["prompt"]),
        "prompt_sha256": str(diagnostics["prompt_sha256"]),
        "history_text_sha256": str(diagnostics["history_sha256"]),
        "text_history_sha256": str(diagnostics["history_sha256"]),
        "suffix_ids_sha256": str(diagnostics["suffix_ids_sha256"]),
        "prediction": prediction,
        "correct": strict_gqa_score(prediction, gold_answer),
        "strict_correct": strict_gqa_score(prediction, gold_answer),
        "score": strict_gqa_score(prediction, gold_answer),
        "quality_score": strict_gqa_score(prediction, gold_answer),
        "quality_metric": "strict_normalized_exact_match",
        "first_token_id": int(result["first_token_id"]),
        "generated_token_count": int(result["generated_tokens"]),
        "generated_tokens": int(result["generated_tokens"]),
        "max_new_tokens": MAX_NEW_TOKENS,
        "first_turn_cross_protocol_key": (
            f"{_dialog_id(dialog)}:{method_key}" if turn_id == 1 else None),
        "store_kind": store_kind,
        "store_id": (store_manifest.get("meta_sha256")
                     if store_manifest is not None else None),
        "store_permutation_sha256": (
            store_manifest.get("permutation_sha256")
            if store_manifest is not None else None),
        "selected_prefix_original_token_ids_sha256": (
            store_manifest.get("selected_prefix_original_token_ids_sha256")
            if store_manifest is not None else None),
        "raster_store_id": (raster_manifest.get("meta_sha256")
                            if raster_manifest is not None else None),
        "image_only_store_id": (ours_manifest.get("meta_sha256")
                                if ours_manifest is not None else None),
        "persistence_source": persistence_source,
        "persistence_source_request": persistence_source is not None,
        "selection_fingerprint_sha256": selected_hash,
        "layout_questions_used": 0,
        "layout_answers_used": 0,
        "layout_uses_generated_answer": False,
        "calibration_questions": 0,
        "future_questions_used_for_layout": 0,
        "future_answers_used_for_layout": 0,
        "future_leakage": 0,
        "rater_scope": "entire_available_causal_suffix",
        "rater_count": int(flattened.get("n_raters", 0) or 0),
        "selector_wall_ms": float(
            flattened.get("selector_ms", 0.0) or 0.0),
        **flattened,
    }
    for optional in (
        "input_tensors_sha256", "input_ids_sha256", "image_input_sha256",
        "combined_suffix_ids_sha256",
    ):
        if diagnostics.get(optional) is not None:
            row[optional] = diagnostics[optional]
    row["gpu_memory_allocated"] = int(torch.cuda.memory_allocated())
    row["gpu_peak_memory_allocated"] = int(
        torch.cuda.max_memory_allocated())
    row["process_rss_bytes"] = int(psutil.Process().memory_info().rss)
    return row


def _validate_selected_chunks(row: Mapping[str, Any], *,
                              expected_k: int, total_chunks: int,
                              fixed_prefix: bool) -> None:
    selected = row.get("selected_chunk_ids_per_layer")
    if not isinstance(selected, list) or not selected:
        raise ValueError("stored selective row omits selected chunk IDs")
    expected_prefix = list(range(expected_k))
    for layer in selected:
        ids = [int(value) for value in layer]
        if len(ids) != expected_k or len(set(ids)) != expected_k:
            raise ValueError("selected chunk count/uniqueness mismatch")
        if ids != sorted(ids) or any(value < 0 or value >= total_chunks
                                     for value in ids):
            raise ValueError("selected chunk IDs are not canonical")
        if fixed_prefix and ids != expected_prefix:
            raise ValueError("Ours25 did not read the fixed physical prefix")
    fingerprint = stable_json_sha256(selected)
    if row.get("selection_fingerprint_sha256") != fingerprint:
        raise ValueError("selection fingerprint mismatch")


def _validate_cache_io(
    row: Mapping[str, Any], *, method: str,
    raster: Mapping[str, Any], image_only: Mapping[str, Any],
) -> None:
    store = image_only if method == "ours25" else raster
    layers = int(store["num_layers"])
    total_bytes = int(row.get("ssd_read_bytes", -1))
    normal_bytes = int(row.get("normal_kv_read_bytes", -1))
    probe_bytes = int(row.get("probe_read_bytes", -1))
    separator_bytes = int(row.get("separator_read_bytes", -1))
    total_preads = int(row.get("ssd_preads", row.get("pread_count", -1)))
    normal_preads = int(row.get("normal_kv_preads", -1))
    probe_preads = int(row.get("probe_preads", -1))
    separator_preads = int(row.get("separator_preads", -1))
    if (min(total_bytes, normal_bytes, probe_bytes, separator_bytes,
            total_preads, normal_preads, probe_preads, separator_preads) < 0
            or total_bytes != normal_bytes + probe_bytes + separator_bytes
            or total_preads != normal_preads + probe_preads + separator_preads):
        raise ValueError(f"{method} SSD byte/pread decomposition mismatch")
    visual_bytes = int(store["visual_kv_bytes"])
    v_tokens = int(store["v_token_num"])
    denominator = layers * v_tokens
    if denominator <= 0 or visual_bytes % denominator:
        raise ValueError(f"{method} store has non-integral token-row bytes")
    bytes_per_visual_token_pair = visual_bytes // denominator
    selected = row.get("selected_chunk_ids_per_layer")
    expected_selected_bytes = None
    if isinstance(selected, list) and selected:
        chunk_size = int(store["chunk_size"])
        selected_rows = 0
        for layer_ids in selected:
            for chunk in layer_ids:
                start = int(chunk) * chunk_size
                stop = min(start + chunk_size, v_tokens)
                if start < 0 or start >= v_tokens or stop <= start:
                    raise ValueError(f"{method} selected an invalid byte span")
                selected_rows += stop - start
        expected_selected_bytes = (
            selected_rows * bytes_per_visual_token_pair)
    if method == "fullload":
        runs = row.get("contiguous_runs_per_layer")
        if (total_bytes != int(raster["visual_kv_bytes"])
                or normal_bytes != int(raster["visual_kv_bytes"])
                or probe_bytes != 0 or separator_bytes != 0
                or normal_preads != 2 * layers
                or probe_preads != 0 or separator_preads != 0
                or runs != [1] * layers
                or float(row.get("contiguous_runs_per_layer_mean", -1)) != 1.0
                or int(row.get("n_raters", -1)) != 0
                or int(row.get("query_score_calls", 0) or 0) != 0):
            raise ValueError("FullLoad exact full-raster I/O contract failed")
        return
    if method == "qa_chunk25":
        runs = row.get("contiguous_runs_per_layer")
        if (not isinstance(runs, list) or len(runs) != layers
                or normal_bytes <= 0
                or normal_bytes != expected_selected_bytes
                or probe_bytes != int(raster["probe_sidecar_bytes"])
                or separator_bytes != int(raster["separator_sidecar_bytes"])
                or normal_preads != 2 * sum(int(value) for value in runs)
                or probe_preads != layers or separator_preads != 1):
            raise ValueError("QA-Chunk exact selective I/O contract failed")
        return
    if method == "ours25":
        runs = row.get("contiguous_runs_per_layer")
        if (not isinstance(runs, list) or runs != [1] * layers
                or normal_bytes <= 0
                or normal_bytes != expected_selected_bytes
                or probe_bytes != 0
                or separator_bytes != int(image_only["separator_sidecar_bytes"])
                or normal_preads != 2 * layers
                or probe_preads != 0 or separator_preads != 1):
            raise ValueError("Ours25 exact fixed-prefix I/O contract failed")
        return
    raise ValueError(f"unexpected cache method: {method}")


def validate_image_rows(
    rows: Sequence[Mapping[str, Any]],
    group: Mapping[str, Any],
    protocol: str,
    store_manifests: Mapping[str, Mapping[str, Any]],
    *,
    seed: int = SEED,
) -> dict[str, Any]:
    """Independently reconstruct histories and validate every request row."""
    if protocol not in PROTOCOLS:
        raise ValueError("invalid protocol")
    dialogs = list(group["dialogs"])
    expected_keys = _expected_row_keys(dialogs)
    observed_keys = [
        (str(row.get("dialog_id")), int(row.get("turn_id", -1)),
         str(row.get("method_key"))) for row in rows
    ]
    if observed_keys != expected_keys:
        raise ValueError("row coverage/order differs from canonical schedule")
    if len(set(observed_keys)) != len(observed_keys):
        raise ValueError("duplicate request identity")
    if set(store_manifests) != {"raster", "image_only"}:
        raise ValueError("artifact must describe both physical stores")
    raster = store_manifests["raster"]
    image_only = store_manifests["image_only"]
    if raster.get("physical_layout") != "raster":
        raise ValueError("QA/FullLoad store is not canonical raster")
    if image_only.get("physical_layout") != "visionzip_image_only":
        raise ValueError("Ours store is not image-only repacked")
    if (not image_only.get("permutation_sha256")
            or not image_only.get(
                "selected_prefix_original_token_ids_sha256")):
        raise ValueError("Ours store omits permutation/prefix provenance")
    if int(raster["visual_kv_bytes"]) != int(image_only["visual_kv_bytes"]):
        raise ValueError("dual stores disagree on full visual K/V bytes")
    if int(raster["n_chunks_per_layer"]) != int(
            image_only["n_chunks_per_layer"]):
        raise ValueError("dual stores disagree on physical chunk count")
    n_chunks = int(raster["n_chunks_per_layer"])
    expected_k = budget_chunk_count(n_chunks, 0.25)

    cursor = 0
    ours_fingerprints: set[str] = set()
    qa_request_count = 0
    cache_hit_count = 0
    session_ids: set[str] = set()
    context_ids: set[str] = set()
    for dialog in dialogs:
        did = _dialog_id(dialog)
        ordinal = int(dialog["global_dialog_ordinal"])
        order = list(method_order(ordinal, seed))
        generated: dict[str, dict[int, dict[str, str]]] = {
            method: {} for method in METHOD_KEYS}
        dialogue_sessions: set[str] = set()
        dialogue_contexts: set[str] = set()
        for turn_id in (1, 2, 3):
            block = rows[cursor:cursor + len(METHOD_KEYS)]
            cursor += len(METHOD_KEYS)
            if [row["method_key"] for row in block] != order:
                raise ValueError("balanced method order changed")
            prompt_hashes = set()
            predictions = set()
            first_tokens = set()
            counts_by_method: dict[str, int] = {}
            for row in block:
                method = str(row["method_key"])
                expected_prompt, expected_history, expected_entries = \
                    render_causal_prompt(
                        dialog, turn_id, protocol, method_key=method,
                        generated_predictions=generated[method])
                if row.get("protocol") != protocol:
                    raise ValueError("row protocol mismatch")
                if row.get("history_text") != expected_history:
                    raise ValueError("history text is not exact causal history")
                if row.get("history_entries") != expected_entries:
                    raise ValueError("history provenance mismatch")
                if row.get("history_answers") != [
                        entry["answer"] for entry in expected_entries]:
                    raise ValueError("history answer list mismatch")
                if row.get("history_source_request_ids") != [
                        entry["source_logical_request_id"]
                        for entry in expected_entries]:
                    raise ValueError("history source request lineage mismatch")
                if row.get("history_source_physical_execution_ids") != [
                        entry["source_physical_execution_id"]
                        for entry in expected_entries]:
                    raise ValueError("history physical lineage mismatch")
                if row.get("prompt") != expected_prompt:
                    raise ValueError("prompt differs from canonical renderer")
                if row.get("prompt_sha256") != sha256_text(expected_prompt):
                    raise ValueError("prompt hash mismatch")
                if row.get("history_text_sha256") != sha256_text(
                        expected_history):
                    raise ValueError("history hash mismatch")
                if row.get("history_turn_ids") != list(range(1, turn_id)):
                    raise ValueError("current/future turn leaked into history")
                if int(row.get("future_leakage", -1)) != 0:
                    raise ValueError("row reports future leakage")
                if float(row.get("correct", -1)) != strict_gqa_score(
                        row.get("prediction", ""), row.get("gold_answer", "")):
                    raise ValueError("strict normalized exact score mismatch")
                expected_logical_id = logical_request_id(
                    protocol, did, turn_id, method)
                if (row.get("logical_request_id") != expected_logical_id
                        or row.get("physical_execution_id")
                        != row.get("execution_id")):
                    raise ValueError("request identity aliases mismatch")
                if int(row.get("input_token_count", 0)) <= 0:
                    raise ValueError("request omits text input token count")
                if int(row.get("generated_token_count", 0)) < 1:
                    raise ValueError("request omits generated token count")
                if row.get("status") != "ok":
                    raise ValueError("non-ok request in completed artifact")
                is_cache = turn_id >= 2 and method != "recompute"
                if bool(row.get("cache_hit")) != is_cache:
                    raise ValueError("cache-hit flag mismatch")
                if turn_id == 1 or method == "recompute":
                    if (row.get("request_path") != "normal_multimodal_pixel"
                            or int(row.get("ssd_read_bytes", -1)) != 0
                            or int(row.get("vision_forward_count", -1)) != 1
                            or row.get("combined_suffix_ids_sha256")
                            != row.get("suffix_ids_sha256")):
                        raise ValueError("pixel request contract failed")
                else:
                    cache_hit_count += 1
                    if (row.get("request_path") != "stored_visual_kv"
                            or int(row.get("ssd_read_bytes", 0)) <= 0
                            or int(row.get("vision_forward_count", -1)) != 0
                            or row.get(
                                "page_cache_conditioning_excluded_from_ttft")
                            is not True):
                        raise ValueError("stored request contract failed")
                    expected_store = (image_only if method == "ours25"
                                      else raster)
                    if row.get("store_id") != expected_store["meta_sha256"]:
                        raise ValueError("request used the wrong physical store")
                    _validate_cache_io(
                        row, method=method, raster=raster,
                        image_only=image_only)
                if method == "qa_chunk25" and is_cache:
                    qa_request_count += 1
                    _validate_selected_chunks(
                        row, expected_k=expected_k,
                        total_chunks=n_chunks, fixed_prefix=False)
                    counts_by_method[method] = len(
                        row["selected_chunk_ids_per_layer"][0])
                    if (int(row.get("query_score_calls", 0))
                            != int(raster["num_layers"])
                            or int(row.get("chunk_score_calls", 0))
                            != int(raster["num_layers"])
                            or float(row.get("fallback_rate", -1)) != 0.0
                            or bool(row.get("adaptive_ratio", True))):
                        raise ValueError("QA-Chunk fixed selector contract failed")
                    loaded = row.get("actual_loaded_chunk_ids_per_layer")
                    if loaded is not None and loaded != row.get(
                            "selected_chunk_ids_per_layer"):
                        raise ValueError("QA selected/loaded chunks differ")
                if method == "ours25" and is_cache:
                    _validate_selected_chunks(
                        row, expected_k=expected_k,
                        total_chunks=n_chunks, fixed_prefix=True)
                    counts_by_method[method] = len(
                        row["selected_chunk_ids_per_layer"][0])
                    if any(int(row.get(field, 0) or 0) != 0 for field in (
                            "query_score_calls", "static_score_calls",
                            "diversity_calls")):
                        raise ValueError("Ours25 used online query scoring")
                    if (row.get("store_permutation_sha256")
                            != image_only["permutation_sha256"]
                            or row.get(
                                "selected_prefix_original_token_ids_sha256")
                            != image_only[
                                "selected_prefix_original_token_ids_sha256"]):
                        raise ValueError(
                            "Ours row/store permutation provenance mismatch")
                    ours_fingerprints.add(
                        str(row["selection_fingerprint_sha256"]))
                dialogue_sessions.add(str(row["dialogue_session_id"]))
                if row.get("context_instance_id") is not None:
                    dialogue_contexts.add(str(row["context_instance_id"]))
                prompt_hashes.add(str(row["prompt_sha256"]))
                predictions.add(str(row["prediction"]))
                first_tokens.add(int(row["first_token_id"]))
                generated[method][turn_id] = {
                    "prediction": str(row["prediction"]),
                    "logical_request_id": str(row["logical_request_id"]),
                    "physical_execution_id": str(
                        row["physical_execution_id"]),
                }
            if counts_by_method and counts_by_method != {
                    "qa_chunk25": expected_k, "ours25": expected_k}:
                raise ValueError("QA-Chunk/Ours normal chunk budgets differ")
            if protocol == "gold_history" and len(prompt_hashes) != 1:
                raise ValueError("gold history inputs differ across methods")
            if turn_id == 1:
                if (len(prompt_hashes) != 1 or len(predictions) != 1
                        or len(first_tokens) != 1):
                    raise ValueError("Turn-1 four-arm fairness failed")
        if len(dialogue_sessions) != 1:
            raise ValueError(f"dialogue session changed within {did}")
        if len(dialogue_contexts) != 1:
            raise ValueError(f"dialogue contexts changed within {did}")
        session = next(iter(dialogue_sessions))
        context = next(iter(dialogue_contexts))
        if session in session_ids or context in context_ids:
            raise ValueError("session/context ID reused across dialogues")
        session_ids.add(session)
        context_ids.add(context)
    expected_cache_hits = len(dialogs) * 2 * 3
    if cache_hit_count != expected_cache_hits:
        raise ValueError("cache-hit request count mismatch")
    if qa_request_count != len(dialogs) * 2:
        raise ValueError("QA request count mismatch")
    if len(ours_fingerprints) != 1:
        raise ValueError("Ours fixed prefix changed within image/protocol")
    return {
        "passed": True,
        "protocol": protocol,
        "n_dialogs": len(dialogs),
        "n_rows": len(rows),
        "expected_rows": len(dialogs) * 3 * len(METHOD_KEYS),
        "cache_hit_rows": cache_hit_count,
        "qa_chunk_cache_hit_rows": qa_request_count,
        "failed_rows": 0,
        "duplicate_rows": 0,
        "strict_scores_recomputed": True,
        "future_leakage": 0,
        "method_local_generated_history_validated": (
            protocol == "generated_history"),
        "gold_teacher_forcing_validated": protocol == "gold_history",
        "turn1_four_arm_fairness": True,
        "ours_selection_invariant_within_protocol": True,
        "normal_chunk_budget_per_layer": expected_k,
    }


def _execute_image_group(
    *,
    runner,
    server,
    group: Mapping[str, Any],
    workload: Mapping[str, Any],
    protocol: str,
    temp_root: Path,
    experiment_id: str,
    shard_index: int,
    seed: int,
    capacity_before_build: Mapping[str, Any],
    reserve_bytes: int,
    recovered_incomplete_temp_store: bool,
) -> dict[str, Any]:
    """Run all dialogues for one image and return an immutable artifact."""
    ImageContext, _ = _runtime_classes()
    image_id = str(group["image_id"])
    dialogs = list(group["dialogs"])
    if not dialogs:
        raise ValueError("image group has no dialogues")
    image_store = temp_root / "payload" / image_id
    mt_base().assert_owned_temp_path(
        image_store, temp_root, experiment_id, image_id)
    if os.path.lexists(image_store):
        raise FileExistsError(f"temporary image store already exists: {image_store}")
    raster_dir = image_store / "raster"
    ours_dir = image_store / "image_only"

    image_path = mt_base().resolve_image_path(dialogs[0])
    if any(mt_base().resolve_image_path(dialog) != image_path
           for dialog in dialogs):
        raise ValueError("same image ID maps to multiple image paths")
    image_sha256 = sha256_file(image_path)
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    rows: list[dict[str, Any]] = []
    contexts: dict[str, Any] = {}
    store_manifests: dict[str, dict[str, Any]] = {}
    persistence: dict[str, dict[str, Any]] = {}
    source_execution_ids: dict[str, str] = {}
    full_visual_bytes: int | None = None
    store_build_counts = {"raster": 0, "image_only": 0}

    try:
        for dialog_index, dialog in enumerate(dialogs):
            if dialog_index > 0:
                if set(contexts) != {"raster", "image_only"}:
                    raise AssertionError("next dialogue started before dual stores")
                for context in contexts.values():
                    context.close()
                contexts = {
                    "raster": ImageContext(
                        raster_dir, runner.model.device,
                        drop_cache=True, require_v_hidden=True),
                    "image_only": ImageContext(
                        ours_dir, runner.model.device,
                        drop_cache=True, require_v_hidden=False),
                }
                contexts["raster"].validate_qa_select_layout()
                contexts["image_only"].validate_prefix_layout(
                    "visionzip_image_only")

            dialogue_session_id = str(uuid.uuid4())
            context_instance_id = str(uuid.uuid4())
            generated: dict[str, dict[int, dict[str, str]]] = {
                method: {} for method in METHOD_KEYS}
            order = method_order(int(dialog["global_dialog_ordinal"]), seed)
            for turn_id in (1, 2, 3):
                turn = _turn(dialog, turn_id)
                request_rows: list[dict[str, Any]] = []
                for order_position, method_key in enumerate(order):
                    torch.cuda.reset_peak_memory_stats()
                    execution_id = str(uuid.uuid4())

                    def prompt_factory(method=method_key):
                        return render_causal_prompt(
                            dialog, turn_id, protocol, method_key=method,
                            generated_predictions=generated[method])

                    is_pixel = turn_id == 1 or method_key == "recompute"
                    persistence_source = None
                    if is_pixel:
                        capture_kind = "none"
                        if dialog_index == 0 and turn_id == 1:
                            if method_key == "fullload":
                                capture_kind = "raster"
                                persistence_source = "raster"
                            elif method_key == "ours25":
                                capture_kind = "ours"
                                persistence_source = "image_only"
                        result, diagnostics = _run_pixel_request(
                            runner, server, image, prompt_factory,
                            capture_kind=capture_kind)
                        captured_cache = result.pop(
                            "captured_past_key_values", None)
                        if capture_kind == "raster":
                            if captured_cache is None:
                                raise AssertionError(
                                    "FullLoad raster source omitted captured K/V")
                            hidden_capture = diagnostics["hidden_capture"]
                            persisted = persist_captured_raster_prefix(
                                runner, captured_cache,
                                diagnostics["enc_cpu"]["input_ids"],
                                diagnostics["enc_cpu"]["image_sizes"][0],
                                hidden_capture.result_cpu(), raster_dir,
                                image_id=image_id, model_id=runner.model_id,
                                chunk_size=CHUNK_SIZE,
                                probe_heads=PROBE_HEADS,
                                hidden_capture_stats=hidden_capture,
                                image_input_sha256=diagnostics[
                                    "image_input_sha256"],
                                extra_metadata={
                                    "dataset": DATASET,
                                    "history_experiment_protocol": protocol,
                                    "source_dialog_id": _dialog_id(dialog),
                                    "source_turn_id": 1,
                                    "source_method_key": method_key,
                                    "source_execution_id": execution_id,
                                })
                            del captured_cache
                            store_build_counts["raster"] += 1
                            contexts["raster"] = ImageContext(
                                raster_dir, runner.model.device,
                                drop_cache=True, require_v_hidden=True)
                            contexts["raster"].validate_qa_select_layout()
                            store_manifests["raster"] = _store_manifest(
                                raster_dir, persisted,
                                contexts["raster"].meta)
                            persistence["raster"] = _persistence_summary(
                                persisted,
                                source_dialog_id=_dialog_id(dialog),
                                source_method_key=method_key,
                                source_execution_id=execution_id)
                            source_execution_ids["raster"] = execution_id
                            raster_bytes = int(
                                contexts["raster"].meta["bytes_visual_kv"])
                            if (full_visual_bytes is not None
                                    and raster_bytes != full_visual_bytes):
                                raise AssertionError(
                                    "raster/repacked visual byte mismatch")
                            full_visual_bytes = raster_bytes
                        elif capture_kind == "ours":
                            if captured_cache is None:
                                raise AssertionError("Ours source omitted captured K/V")
                            vision_capture = diagnostics["vision_capture"]
                            persisted = persist_captured_visual_prefix(
                                runner, captured_cache,
                                diagnostics["enc_cpu"]["input_ids"],
                                diagnostics["enc_cpu"]["image_sizes"][0],
                                vision_capture.result_cpu(), ours_dir,
                                image_id=image_id, model_id=runner.model_id,
                                chunk_size=CHUNK_SIZE,
                                image_input_sha256=diagnostics[
                                    "image_input_sha256"],
                                capture_stats=vision_capture,
                                extra_metadata={
                                    "dataset": DATASET,
                                    "history_experiment_protocol": protocol,
                                    "source_dialog_id": _dialog_id(dialog),
                                    "source_turn_id": 1,
                                    "source_method_key": method_key,
                                    "source_execution_id": execution_id,
                                    "future_questions_used_for_layout": 0,
                                })
                            del captured_cache
                            store_build_counts["image_only"] += 1
                            contexts["image_only"] = ImageContext(
                                ours_dir, runner.model.device,
                                drop_cache=True, require_v_hidden=False)
                            contexts["image_only"].validate_prefix_layout(
                                "visionzip_image_only")
                            store_manifests["image_only"] = _store_manifest(
                                ours_dir, persisted,
                                contexts["image_only"].meta)
                            persistence["image_only"] = _persistence_summary(
                                persisted,
                                source_dialog_id=_dialog_id(dialog),
                                source_method_key=method_key,
                                source_execution_id=execution_id)
                            source_execution_ids["image_only"] = execution_id
                            ours_bytes = int(contexts[
                                "image_only"].meta["bytes_visual_kv"])
                            if (full_visual_bytes is not None
                                    and ours_bytes != full_visual_bytes):
                                raise AssertionError(
                                    "raster/repacked visual byte mismatch")
                            full_visual_bytes = ours_bytes
                        elif captured_cache is not None:
                            raise AssertionError(
                                "non-source pixel request retained captured K/V")
                    else:
                        if set(contexts) != {"raster", "image_only"}:
                            raise AssertionError("cache request preceded dual stores")
                        context_key = ("image_only" if method_key == "ours25"
                                       else "raster")
                        result, diagnostics = _run_stored_request(
                            runner, server, contexts[context_key],
                            prompt_factory, method_key=method_key,
                            image_id=image_id,
                            full_visual_bytes=int(full_visual_bytes))

                    row = _make_row(
                        protocol=protocol, workload=workload,
                        dialog=dialog, turn=turn, method_key=method_key,
                        order=order, order_position=order_position,
                        result=result, diagnostics=diagnostics,
                        execution_id=execution_id,
                        dialogue_session_id=dialogue_session_id,
                        context_instance_id=context_instance_id,
                        full_visual_bytes=full_visual_bytes,
                        raster_manifest=store_manifests.get("raster"),
                        ours_manifest=store_manifests.get("image_only"),
                        persistence_source=persistence_source)
                    request_rows.append(row)
                    generated[method_key][turn_id] = {
                        "prediction": str(row["prediction"]),
                        "logical_request_id": str(row["logical_request_id"]),
                        "physical_execution_id": str(
                            row["physical_execution_id"]),
                    }
                    diagnostics.pop("enc_cpu", None)
                    diagnostics.pop("vision_capture", None)
                    diagnostics.pop("hidden_capture", None)
                    gc.collect()

                if turn_id == 1:
                    if (len({row["prompt_sha256"] for row in request_rows}) != 1
                            or len({row["prediction"] for row in request_rows}) != 1
                            or len({row["first_token_id"]
                                    for row in request_rows}) != 1):
                        raise AssertionError("Turn-1 four-arm fairness failed")
                rows.extend(request_rows)

        if (set(store_manifests) != {"raster", "image_only"}
                or set(persistence) != {"raster", "image_only"}
                or store_build_counts != {"raster": 1, "image_only": 1}):
            raise AssertionError("image did not build exactly one dual store")
        capacity_after_build = mt_base()._capacity_guard(
            temp_root, reserve_bytes=reserve_bytes, extra_headroom_bytes=0)
        validation = validate_image_rows(
            rows, group, protocol, store_manifests, seed=seed)
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": str(experiment_id),
            "dataset": DATASET,
            "benchmark_type": workload["benchmark_type"],
            "official_benchmark_identity_claimed": False,
            "benchmark_disclaimer": workload["benchmark_disclaimer"],
            "protocol": protocol,
            "history_policy": (
                "gold_teacher_forced" if protocol == "gold_history"
                else "method_local_generated"),
            "protocol_physical_execution_independent": True,
            "image_id": image_id,
            "image_ordinal": int(group["image_ordinal"]),
            "shard_index": int(shard_index),
            "dialogues_file_sha256": workload["dialogues_file_sha256"],
            "source_full_workload_sha256": workload[
                "source_full_workload_sha256"],
            "selected_workload_sha256": workload[
                "selected_workload_sha256"],
            "model_revision": workload.get("model_revision"),
            "source_image_path": str(image_path),
            "source_image_sha256": image_sha256,
            "dialog_ids": [_dialog_id(dialog) for dialog in dialogs],
            "global_dialog_ordinals": [
                int(dialog["global_dialog_ordinal"]) for dialog in dialogs],
            "n_dialogs": len(dialogs),
            "n_turns": len(dialogs) * 3,
            "n_rows": len(rows),
            "logical_request_count": len(rows),
            "physical_execution_count": len(rows),
            "failed_request_count": 0,
            "duplicate_request_count": 0,
            "store_build_counts": store_build_counts,
            "store_manifests": store_manifests,
            "persistence_overhead": persistence,
            "source_execution_ids": source_execution_ids,
            "capacity_before_build": dict(capacity_before_build),
            "capacity_after_build": capacity_after_build,
            "recovered_incomplete_temp_store_before_build": bool(
                recovered_incomplete_temp_store),
            "rows": rows,
            "validation": validation,
        }
        artifact["artifact_content_sha256"] = _artifact_body_hash(artifact)
        return artifact
    finally:
        for context in contexts.values():
            context.close()
        image.close()


def validate_resume_artifact(
    path: Path,
    *,
    experiment_id: str,
    image_group: Mapping[str, Any],
    shard_index: int,
    workload: Mapping[str, Any],
    protocol: str,
    seed: int,
) -> dict[str, Any]:
    """Validate an immutable image artifact completely before skipping it."""
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"resume artifact is not a regular file: {path}")
    artifact = json.loads(path.read_text())
    dialogs = list(image_group["dialogs"])
    if not dialogs:
        raise ValueError("resume image group has no dialogues")
    image_path = mt_base().resolve_image_path(dialogs[0])
    if any(mt_base().resolve_image_path(dialog) != image_path
           for dialog in dialogs):
        raise ValueError("same resumed image ID maps to multiple image paths")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": str(experiment_id),
        "dataset": DATASET,
        "benchmark_type": workload["benchmark_type"],
        "protocol": protocol,
        "image_id": str(image_group["image_id"]),
        "image_ordinal": int(image_group["image_ordinal"]),
        "shard_index": int(shard_index),
        "dialogues_file_sha256": workload["dialogues_file_sha256"],
        "source_full_workload_sha256": workload[
            "source_full_workload_sha256"],
        "selected_workload_sha256": workload[
            "selected_workload_sha256"],
        "model_revision": workload.get("model_revision"),
        "source_image_path": str(image_path),
        "source_image_sha256": sha256_file(image_path),
    }
    mismatch = {key: (artifact.get(key), value)
                for key, value in expected.items()
                if artifact.get(key) != value}
    if mismatch:
        raise ValueError(f"resume artifact identity mismatch: {mismatch}")
    if artifact.get("artifact_content_sha256") != _artifact_body_hash(artifact):
        raise ValueError("resume artifact content hash mismatch")
    rows = artifact.get("rows")
    manifests = artifact.get("store_manifests")
    if not isinstance(rows, list) or not isinstance(manifests, Mapping):
        raise ValueError("resume artifact omits rows or dual-store manifests")
    validation = validate_image_rows(
        rows, image_group, protocol, manifests, seed=seed)
    if artifact.get("validation") != validation:
        raise ValueError("stored validation differs from independent replay")
    if (int(artifact.get("logical_request_count", -1)) != len(rows)
            or int(artifact.get("physical_execution_count", -1)) != len(rows)
            or int(artifact.get("failed_request_count", -1)) != 0
            or int(artifact.get("duplicate_request_count", -1)) != 0):
        raise ValueError("resume request accounting mismatch")
    physical_ids = [str(row.get("physical_execution_id", "")) for row in rows]
    if (any(not value for value in physical_ids)
            or len(set(physical_ids)) != len(physical_ids)):
        raise ValueError("physical execution IDs are empty or duplicated")
    logical_ids = [str(row.get("logical_request_id", "")) for row in rows]
    if (any(not value for value in logical_ids)
            or len(set(logical_ids)) != len(logical_ids)):
        raise ValueError("logical request IDs are empty or duplicated")
    if artifact.get("store_build_counts") != {
            "raster": 1, "image_only": 1}:
        raise ValueError("resume artifact did not build both stores once")
    persistence = artifact.get("persistence_overhead")
    if not isinstance(persistence, Mapping) or set(persistence) != {
            "raster", "image_only"}:
        raise ValueError("resume artifact omits persistence provenance")
    source_ids = artifact.get("source_execution_ids")
    if not isinstance(source_ids, Mapping) or set(source_ids) != {
            "raster", "image_only"}:
        raise ValueError("resume artifact omits store source executions")
    by_physical = {row["physical_execution_id"]: row for row in rows}
    expected_source_methods = {
        "raster": "fullload", "image_only": "ours25"}
    for store_kind, method in expected_source_methods.items():
        source_id = source_ids[store_kind]
        source_row = by_physical.get(source_id)
        if (source_row is None or source_row["turn_id"] != 1
                or source_row["method_key"] != method
                or source_row["persistence_source"] != store_kind
                or persistence[store_kind]["source_execution_id"]
                != source_id):
            raise ValueError("store source request provenance mismatch")
    return artifact


def _model_revision() -> str | None:
    return mt_base()._model_revision()


def _base_config(args, workload: Mapping[str, Any], groups: Sequence[Mapping],
                 n_shards: int, warmup: Mapping[str, Any], runner) -> dict:
    counts = expected_request_counts(int(workload["n_dialogs"]))
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(args.temp_root)
    gpu = torch.cuda.get_device_properties(0)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": str(args.experiment_id),
        "dataset": DATASET,
        "benchmark_type": workload["benchmark_type"],
        "official_benchmark_identity_claimed": False,
        "benchmark_disclaimer": workload["benchmark_disclaimer"],
        "protocol": args.protocol,
        "history_policy": (
            "gold_teacher_forced" if args.protocol == "gold_history"
            else "method_local_generated"),
        "protocol_physical_execution_independent": True,
        "index": str(workload["path"]),
        "dialogues_file_sha256": workload["dialogues_file_sha256"],
        "source_full_workload_sha256": workload[
            "source_full_workload_sha256"],
        "selected_workload_sha256": workload[
            "selected_workload_sha256"],
        "source_full_n_dialogs": int(workload["source_full_n_dialogs"]),
        "partial_workload": bool(workload["partial_workload"]),
        "n_dialogs": int(workload["n_dialogs"]),
        "n_turns": int(workload["n_turns"]),
        "n_images": len(groups),
        "n_requests": counts["requests_per_protocol"],
        "planned_logical_requests_this_protocol": counts[
            "requests_per_protocol"],
        "planned_physical_executions_this_protocol": counts[
            "requests_per_protocol"],
        "planned_logical_requests_both_protocols": counts[
            "requests_both_protocols"],
        "planned_physical_executions_both_protocols": counts[
            "requests_both_protocols"],
        "request_counts": counts,
        "shard_size": int(args.shard_size),
        "n_shards": int(n_shards),
        "seed": int(args.seed),
        "method_keys": list(METHOD_KEYS),
        "methods": METHODS,
        "method_order_policy": (
            "zero-based cyclic rotation by frozen global dialogue ordinal; "
            "identical across turns and protocols"),
        "model": runner.model_id,
        "model_revision": _model_revision(),
        "load_4bit": bool(runner.load_4bit),
        "quantization": "4-bit NF4 double-quant",
        "compute_dtype": COMPUTE_DTYPE,
        "attention": runner.attn,
        "decoding": "greedy",
        "max_new_tokens": int(args.max_new_tokens),
        "chunk_size": CHUNK_SIZE,
        "probe_heads": PROBE_HEADS,
        "qa_chunk_configuration": {
            "physical_layout": "raster",
            "head_reduce": "mean",
            "chunk_aggregation": "mean_valid_spatial_tokens",
            "normal_chunk_budget": 0.25,
            "budget_helper": "budget_chunk_count_round",
            "rater_algorithm_id": QA_RATER_ALGORITHM_ID,
            "rater_scope": "entire_available_causal_suffix",
            "fallback": False,
            "adaptive_budget": False,
        },
        "ours_configuration": {
            "physical_layout": "visionzip_image_only",
            "normal_chunk_budget": 0.25,
            "selection": "fixed_first_k_prefix",
            "online_query_scoring": False,
        },
        "turn1_policy": (
            "all four methods execute independent normal pixel requests; "
            "FullLoad captures/persists the shared canonical raster store "
            "from its own first-dialog T1 and Ours captures/persists its "
            "image-only store from its own first-dialog T1"),
        "qa_raster_source_policy": (
            "QA-Chunk25 reads the byte-identical canonical raster Visual-KV "
            "captured by FullLoad's own T1; only QA's validated online "
            "selection path differs at cache-hit turns"),
        "later_turn_policy": (
            "ReComp pixels; FullLoad/QA use raster SSD; Ours uses independent "
            "image-only repacked SSD"),
        "input_token_count_definition": (
            "tokenizer suffix after the single image marker; excludes expanded "
            "visual rows and is comparable across pixel/stored requests"),
        "generated_token_count_definition": (
            "greedy output decisions including EOS when produced"),
        "quality_metric": "strict_normalized_exact_match",
        "normalization": (
            "lowercase; punctuation to spaces; remove a/an/the; collapse "
            "whitespace; exact equality to the first/only gold answer"),
        "ssd_read_api": "buffered os.pread",
        "o_direct": False,
        "ssd_controller_cache_flushed": False,
        "cache_condition": "OS-page-cache-cold",
        "page_cache_conditioning": "posix_fadvise(DONTNEED)",
        "page_cache_conditioning_inside_ttft": False,
        "main_ttft_field": "end_to_end_ttft_ms",
        "ttft_definition": (
            "after page-cache conditioning, before prompt rendering -> "
            "tokenization/H2D -> selector or vision -> SSD/scatter -> prefill "
            "-> synchronized first output token"),
        "persistence_in_main_ttft": False,
        "qa_v_hidden_policy": (
            "v_hidden.pt is loaded and layout-validated when ImageContext "
            "opens outside request TTFT; per-request H2D, rater selection, "
            "and query scoring are inside TTFT"),
        "context_instance_id_semantics": (
            "dialogue-scoped dual-store context group used to prove no state "
            "crosses dialogue boundaries; not an individual ImageContext ID"),
        "image_at_a_time_temporary_store": True,
        "two_physical_stores_per_protocol_image": True,
        "temporary_payload_deleted_only_after_immutable_image_artifact": True,
        "unmeasured_warmup": dict(warmup),
        "run_dir": str(args.run_dir),
        "temp_root": str(args.temp_root),
        "capacity_guard": {
            "min_free_after_gib": float(args.min_free_after_gib),
            "build_headroom_gib": BUILD_HEADROOM_GIB,
            "max_used_percent_exclusive": 96.0,
        },
        "machine": {
            "hostname": platform.node(),
            "gpu_name": gpu.name,
            "gpu_total_memory_bytes": int(gpu.total_memory),
            "system_ram_total_bytes": int(vm.total),
            "system_ram_available_bytes_at_start": int(vm.available),
            "disk_total_bytes": int(disk.total),
            "disk_free_bytes_at_start": int(disk.free),
            "cpu_count": os.cpu_count(),
            "torch_version": torch.__version__,
        },
        "run_started_at_unix": time.time(),
    }


def _ensure_run_config(run_dir: Path, config: Mapping[str, Any]) -> None:
    if not run_dir.is_absolute():
        raise ValueError("run-dir must be absolute")
    if run_dir.exists() and run_dir.is_symlink():
        raise ValueError("run-dir may not be a symlink")
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config.json"
    invariant = (
        "schema_version", "experiment_id", "dataset", "benchmark_type",
        "protocol", "dialogues_file_sha256",
        "source_full_workload_sha256", "selected_workload_sha256",
        "n_dialogs", "n_images", "seed", "method_keys", "shard_size",
        "n_shards", "max_new_tokens", "model_revision",
        "model", "load_4bit", "quantization", "compute_dtype",
        "attention", "decoding", "chunk_size", "probe_heads",
        "qa_chunk_configuration", "ours_configuration", "turn1_policy",
        "qa_raster_source_policy", "later_turn_policy",
        "planned_physical_executions_this_protocol",
    )
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise ValueError("existing run config is not a regular file")
        current = json.loads(path.read_text())
        mismatch = {key: (current.get(key), config.get(key))
                    for key in invariant
                    if current.get(key) != config.get(key)}
        if mismatch:
            raise ValueError(f"existing run config mismatch: {mismatch}")
    else:
        mt_base()._write_exclusive_json(path, dict(config))
    for child_name in ("images", "shards"):
        child = run_dir / child_name
        if child.exists() and child.is_symlink():
            raise ValueError(f"artifact directory is a symlink: {child}")
        child.mkdir(exist_ok=True)


def _validate_existing_config(run_dir: Path, args,
                              workload: Mapping[str, Any],
                              groups: Sequence[Mapping], n_shards: int) -> dict:
    path = run_dir / "config.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError("resumed run has no regular config.json")
    config = json.loads(path.read_text())
    counts = expected_request_counts(int(workload["n_dialogs"]))
    expected = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": str(args.experiment_id),
        "dataset": DATASET,
        "benchmark_type": workload["benchmark_type"],
        "protocol": args.protocol,
        "dialogues_file_sha256": workload["dialogues_file_sha256"],
        "source_full_workload_sha256": workload[
            "source_full_workload_sha256"],
        "selected_workload_sha256": workload[
            "selected_workload_sha256"],
        "n_dialogs": int(workload["n_dialogs"]),
        "n_images": len(groups),
        "seed": int(args.seed),
        "method_keys": list(METHOD_KEYS),
        "shard_size": int(args.shard_size),
        "n_shards": int(n_shards),
        "max_new_tokens": MAX_NEW_TOKENS,
        "model_revision": _model_revision(),
        "model": MODEL_ID,
        "load_4bit": True,
        "quantization": "4-bit NF4 double-quant",
        "compute_dtype": COMPUTE_DTYPE,
        "attention": "eager",
        "decoding": "greedy",
        "chunk_size": CHUNK_SIZE,
        "probe_heads": PROBE_HEADS,
        "qa_chunk_configuration": {
            "physical_layout": "raster",
            "head_reduce": "mean",
            "chunk_aggregation": "mean_valid_spatial_tokens",
            "normal_chunk_budget": 0.25,
            "budget_helper": "budget_chunk_count_round",
            "rater_algorithm_id": QA_RATER_ALGORITHM_ID,
            "rater_scope": "entire_available_causal_suffix",
            "fallback": False,
            "adaptive_budget": False,
        },
        "ours_configuration": {
            "physical_layout": "visionzip_image_only",
            "normal_chunk_budget": 0.25,
            "selection": "fixed_first_k_prefix",
            "online_query_scoring": False,
        },
        "turn1_policy": (
            "all four methods execute independent normal pixel requests; "
            "FullLoad captures/persists the shared canonical raster store "
            "from its own first-dialog T1 and Ours captures/persists its "
            "image-only store from its own first-dialog T1"),
        "qa_raster_source_policy": (
            "QA-Chunk25 reads the byte-identical canonical raster Visual-KV "
            "captured by FullLoad's own T1; only QA's validated online "
            "selection path differs at cache-hit turns"),
        "later_turn_policy": (
            "ReComp pixels; FullLoad/QA use raster SSD; Ours uses independent "
            "image-only repacked SSD"),
        "planned_physical_executions_this_protocol": counts[
            "requests_per_protocol"],
    }
    mismatch = {key: (config.get(key), value)
                for key, value in expected.items()
                if config.get(key) != value}
    if mismatch:
        raise ValueError(f"existing run config mismatch: {mismatch}")
    return config


def _validate_shard_marker(path: Path, *, args,
                           workload: Mapping[str, Any],
                           shard: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"shard marker is not regular: {path}")
    marker = json.loads(path.read_text())
    image_ids = [str(group["image_id"]) for group in shard["groups"]]
    expected = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": str(args.experiment_id),
        "dataset": DATASET,
        "benchmark_type": workload["benchmark_type"],
        "protocol": args.protocol,
        "dialogues_file_sha256": workload["dialogues_file_sha256"],
        "source_full_workload_sha256": workload[
            "source_full_workload_sha256"],
        "selected_workload_sha256": workload[
            "selected_workload_sha256"],
        "shard_index": int(args.shard_index),
        "shard_size": int(args.shard_size),
        "image_start": int(shard["start"]),
        "image_stop": int(shard["stop"]),
        "image_ids": image_ids,
        "completed_image_ids": image_ids,
        "complete": True,
    }
    mismatch = {key: (marker.get(key), value)
                for key, value in expected.items()
                if marker.get(key) != value}
    if mismatch:
        raise ValueError(f"completed shard marker mismatch: {mismatch}")
    if marker.get("artifact_content_sha256") != _artifact_body_hash(marker):
        raise ValueError("shard marker content hash mismatch")
    file_hashes = marker.get("image_artifact_file_sha256")
    content_hashes = marker.get("image_artifact_content_sha256")
    if (not isinstance(file_hashes, Mapping)
            or set(file_hashes) != set(image_ids)
            or not isinstance(content_hashes, Mapping)
            or set(content_hashes) != set(image_ids)):
        raise ValueError("shard marker image hash coverage mismatch")
    physical_count = 0
    for group in shard["groups"]:
        image_id = str(group["image_id"])
        artifact_path = _image_artifact_path(args.run_dir, image_id)
        artifact = validate_resume_artifact(
            artifact_path, experiment_id=args.experiment_id,
            image_group=group, shard_index=args.shard_index,
            workload=workload, protocol=args.protocol, seed=args.seed)
        if sha256_file(artifact_path) != file_hashes[image_id]:
            raise ValueError("published image file hash changed")
        if artifact["artifact_content_sha256"] != content_hashes[image_id]:
            raise ValueError("published image content hash changed")
        physical_count += int(artifact["physical_execution_count"])
    if (int(marker.get("physical_execution_count", -1)) != physical_count
            or int(marker.get("logical_request_count", -1)) != physical_count):
        raise ValueError("shard request accounting mismatch")
    return marker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, choices=PROTOCOLS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument(
        "--expected-index-sha256", default=EXPECTED_INDEX_SHA256,
        help="SHA256 of the complete canonical dialogues JSON")
    parser.add_argument(
        "--expected-workload-sha256", default=EXPECTED_WORKLOAD_SHA256,
        help="ordered dialogue/turn/question request-key SHA256")
    parser.add_argument("--expected-dialogs", type=int,
                        default=EXPECTED_DIALOGUES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-new-tokens", type=int,
                        default=MAX_NEW_TOKENS)
    parser.add_argument("--min-free-after-gib", type=float,
                        default=MIN_FREE_AFTER_GIB)
    parser.add_argument("--max-dialogs", type=int, choices=(10, 100))
    parser.add_argument("--allow-partial-workload", action="store_true")
    return parser


def _validate_cli(args) -> None:
    if args.protocol not in PROTOCOLS:
        raise ValueError("invalid history protocol")
    if not args.run_dir.is_absolute() or not args.temp_root.is_absolute():
        raise ValueError("run-dir and temp-root must be absolute")
    if ((args.run_dir.exists() and args.run_dir.is_symlink())
            or (args.temp_root.exists() and args.temp_root.is_symlink())):
        raise ValueError("run-dir and temp-root may not be symlinks")
    run_root, temp_root = args.run_dir.resolve(), args.temp_root.resolve()
    if (run_root == temp_root or run_root in temp_root.parents
            or temp_root in run_root.parents):
        raise ValueError("run-dir and temp-root may not overlap")
    if not MIN_SHARD_SIZE <= int(args.shard_size) <= MAX_SHARD_SIZE:
        raise ValueError("shard-size must be in [40,60]")
    if int(args.shard_index) < 0:
        raise ValueError("shard-index must be nonnegative")
    if int(args.seed) != SEED:
        raise ValueError("the frozen MT-GQA experiment uses seed 1234")
    if int(args.max_new_tokens) != MAX_NEW_TOKENS:
        raise ValueError("the frozen experiment uses 16 output tokens")
    if float(args.min_free_after_gib) < MIN_FREE_AFTER_GIB:
        raise ValueError("min-free-after-gib may not be below 30 GiB")
    if (args.max_dialogs is None) != (not args.allow_partial_workload):
        raise ValueError(
            "--max-dialogs and --allow-partial-workload must be paired")
    expected_selected = (EXPECTED_DIALOGUES if args.max_dialogs is None
                         else int(args.max_dialogs))
    if int(args.expected_dialogs) != expected_selected:
        raise ValueError(
            "expected-dialogs must equal the selected full/smoke workload")


def main() -> None:
    args = build_parser().parse_args()
    _validate_cli(args)
    full = mt_base().resolve_dialogues(
        args.index, expected_dialogs=EXPECTED_DIALOGUES,
        expected_dialogues_sha256=args.expected_index_sha256,
        expected_workload_sha256=args.expected_workload_sha256)
    workload = mt_base().select_dialogues(full, args.max_dialogs)
    if int(workload["n_dialogs"]) != int(args.expected_dialogs):
        raise ValueError("selected dialogue count mismatch")
    model_revision = _model_revision()
    if not isinstance(model_revision, str) or not model_revision:
        raise ValueError("local LLaVA-NeXT checkpoint revision is unavailable")
    workload["model_revision"] = model_revision
    groups = mt_base().group_dialogues_by_image(workload["dialogs"])
    if args.max_dialogs is None and len(groups) != EXPECTED_IMAGES:
        raise ValueError("canonical full workload image count mismatch")
    shard = mt_base().shard_image_groups(
        groups, args.shard_index, args.shard_size)

    config_path = args.run_dir / "config.json"
    if config_path.exists():
        _validate_existing_config(
            args.run_dir, args, workload, groups, shard["n_shards"])
    elif args.run_dir.exists():
        if args.run_dir.is_symlink() or not args.run_dir.is_dir():
            raise ValueError("existing run-dir is not a regular directory")
        if any(args.run_dir.iterdir()):
            raise ValueError(
                "nonempty output root has no config.json; refusing to mix runs")

    mt_base()._claim_temp_root(
        args.temp_root, args.experiment_id, DATASET)
    shard_marker = (args.run_dir / "shards" /
                    f"shard_{args.shard_index:03d}.json")
    if shard_marker.is_file():
        _validate_shard_marker(
            shard_marker, args=args, workload=workload, shard=shard)
        print(f"{args.protocol} shard {args.shard_index} already complete")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    runner = LlavaRunner().load()
    _, Server = _runtime_classes()
    server = Server(
        runner, ratio=0.25, probe=PROBE_HEADS,
        max_new_tokens=MAX_NEW_TOKENS)
    warmup = qa_base()._warmup(runner, server)
    config = _base_config(
        args, workload, groups, shard["n_shards"], warmup, runner)
    _ensure_run_config(args.run_dir, config)

    reserve_bytes = int(float(args.min_free_after_gib) * (1024 ** 3))
    headroom_bytes = int(BUILD_HEADROOM_GIB * (1024 ** 3))
    completed: list[str] = []
    physical_count = 0
    started_at = time.time()
    for local_index, group in enumerate(shard["groups"]):
        image_id = str(group["image_id"])
        artifact_path = _image_artifact_path(args.run_dir, image_id)
        image_store = args.temp_root / "payload" / image_id
        mt_base().assert_owned_temp_path(
            image_store, args.temp_root, args.experiment_id, image_id)
        if artifact_path.is_file():
            artifact = validate_resume_artifact(
                artifact_path, experiment_id=args.experiment_id,
                image_group=group, shard_index=args.shard_index,
                workload=workload, protocol=args.protocol, seed=args.seed)
            if os.path.lexists(image_store):
                mt_base().remove_owned_temp_store(
                    image_store, args.temp_root,
                    args.experiment_id, image_id)
            completed.append(image_id)
            physical_count += int(artifact["physical_execution_count"])
            print(f"[{local_index + 1}/{len(shard['groups'])}] "
                  f"{image_id}: validated immutable artifact", flush=True)
            continue
        if os.path.lexists(image_store):
            mt_base().remove_owned_temp_store(
                image_store, args.temp_root, args.experiment_id, image_id)
            recovered = True
        else:
            recovered = False
        capacity_before = mt_base()._capacity_guard(
            args.temp_root, reserve_bytes=reserve_bytes,
            extra_headroom_bytes=headroom_bytes)
        artifact = _execute_image_group(
            runner=runner, server=server, group=group, workload=workload,
            protocol=args.protocol, temp_root=args.temp_root,
            experiment_id=args.experiment_id,
            shard_index=args.shard_index, seed=args.seed,
            capacity_before_build=capacity_before,
            reserve_bytes=reserve_bytes,
            recovered_incomplete_temp_store=recovered)
        mt_base()._write_exclusive_json(artifact_path, artifact)
        mt_base().remove_owned_temp_store(
            image_store, args.temp_root, args.experiment_id, image_id)
        completed.append(image_id)
        physical_count += int(artifact["physical_execution_count"])
        total_store_bytes = sum(
            int(item["total_store_bytes"])
            for item in artifact["store_manifests"].values())
        print(
            f"[{local_index + 1}/{len(shard['groups'])}] {image_id}: "
            f"{artifact['n_dialogs']} dialogues, {artifact['n_rows']} rows, "
            f"dual stores {total_store_bytes / 1e9:.2f} GB", flush=True)
        torch.cuda.empty_cache()
        gc.collect()

    image_ids = [str(group["image_id"]) for group in shard["groups"]]
    if completed != image_ids:
        raise AssertionError("shard did not complete in canonical image order")
    marker = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": str(args.experiment_id),
        "dataset": DATASET,
        "benchmark_type": workload["benchmark_type"],
        "protocol": args.protocol,
        "protocol_physical_execution_independent": True,
        "dialogues_file_sha256": workload["dialogues_file_sha256"],
        "source_full_workload_sha256": workload[
            "source_full_workload_sha256"],
        "selected_workload_sha256": workload[
            "selected_workload_sha256"],
        "shard_index": int(args.shard_index),
        "shard_size": int(args.shard_size),
        "image_start": int(shard["start"]),
        "image_stop": int(shard["stop"]),
        "image_ids": image_ids,
        "completed_image_ids": completed,
        "logical_request_count": physical_count,
        "physical_execution_count": physical_count,
        "failed_request_count": 0,
        "duplicate_request_count": 0,
        "complete": True,
        "elapsed_seconds": float(time.time() - started_at),
        "image_artifact_file_sha256": {
            image_id: sha256_file(_image_artifact_path(
                args.run_dir, image_id)) for image_id in image_ids
        },
        "image_artifact_content_sha256": {
            image_id: json.loads(_image_artifact_path(
                args.run_dir, image_id).read_text())[
                    "artifact_content_sha256"] for image_id in image_ids
        },
    }
    marker["artifact_content_sha256"] = _artifact_body_hash(marker)
    mt_base()._write_exclusive_json(shard_marker, marker)
    print(f"completed {args.protocol} shard {args.shard_index}: "
          f"{len(completed)} images, {physical_count} physical requests")


if __name__ == "__main__":
    main()

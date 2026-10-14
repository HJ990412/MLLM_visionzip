"""Run the frozen MT-GQA Generated-History five-arm SSD comparison.

Each method executes an independent normal pixel Turn 1. Four first-dialogue
Turn-1 requests additionally capture the data needed for their own temporary
image stores: canonical FullLoad, MPIC-32, pre-RoPE ReKV, and image-only Ours.
Each dialogue propagates only each method's exact generated answer. The
published unit is a hash-checked immutable image artifact; temporary stores
are removed after the artifact commits.
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
from mmimpress.mpic import (  # noqa: E402
    MPICContext, MPICServer, persist_captured_mpic_prefix,
)
from mmimpress.rekv import ReKVServer  # noqa: E402
from mmimpress.rekv_store import (  # noqa: E402
    ReKVCapture, ReKVContext, persist_captured_rekv_prefix,
)
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


SCHEMA_VERSION = "mt-gqa-5arm-generated-history-shard-v1"
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
PROTOCOLS = ("generated_history",)

METHOD_KEYS = ("recompute", "fullload", "mpic32", "rekv_chunk25", "ours25")
STORE_KEYS = ("raster", "mpic", "rekv", "image_only")
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
    "mpic32": {
        "method_id": "mpic32_ssd", "label": "MPIC-32 (adapted)",
        "retention_ratio": None, "physical_layout": "canonical_raster",
        "importance_source": "none", "query_dependent": False,
        "online_selection": False,
        "k_recompute": 32,
    },
    "rekv_chunk25": {
        "method_id": "rekv_chunk25_ssd", "label": "ReKV-Chunk25 (adapted)",
        "retention_ratio": 0.25, "physical_layout": "canonical_raster",
        "importance_source": "official-code FP32 dot product",
        "query_dependent": True, "online_selection": True,
        "selection_granularity": "64-token SSD visual chunk",
        "similarity_mode": "official_code_dot",
        "official_commit": "1fd9a3dbf5dbff7f27069ae2f4463674c495e830",
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
        "requests_total": per_protocol,
        "main_t2_t3_requests": 2 * n * len(METHOD_KEYS),
        "stored_visual_kv_hits": 2 * n * 4,
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
    """Canonical MT-GQA prompt with method-local generated history."""
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
    """Flatten counters while retaining each validated serving implementation."""
    if method_key == "mpic32" and "ssd_total_bytes" in result:
        value = dict(result)
        io = value.get("io", {})
        kinds = io.get("per_kind", {})
        value["ssd_read_bytes"] = int(value["ssd_total_bytes"])
        value["ssd_preads"] = int(value["pread_count"])
        value["normal_kv_read_bytes"] = int(value["ssd_kv_bytes"])
        value["embedding_read_bytes"] = int(value["ssd_embedding_bytes"])
        value["separator_read_bytes"] = int(value["ssd_separator_bytes"])
        value["probe_read_bytes"] = 0
        value["normal_kv_preads"] = sum(
            int(kinds.get(kind, {}).get("preads", 0))
            for kind in ("kv_k", "kv_v"))
        value["embedding_preads"] = int(
            kinds.get("embedding", {}).get("preads", 0))
        value["separator_preads"] = 0
        value["probe_preads"] = 0
        value["ssd_read_ms"] = float(
            value.get("kv_read_ms", 0.0) + value.get("embedding_read_ms", 0.0))
        value["actual_ssd_mb"] = value["ssd_read_bytes"] / 1e6
        value["actual_ssd_ratio_vs_fullload"] = (
            value["ssd_read_bytes"] / int(full_visual_bytes or 1))
        value["io_detail"] = kinds
        value["contiguous_runs_per_layer"] = None
        value["contiguous_runs_per_layer_mean"] = None
        value["selected_chunk_payload_read_bytes"] = value["normal_kv_read_bytes"]
        return value
    value = qa_base()._json_result(
        dict(result), method_key, int(full_visual_bytes or 0))
    for field in (
        "chunk_aggregation_ms", "topk_chunk_ms",
        "selector_decision_host_wall_ms", "selected_chunk_io_ms",
        "normal_kv_read_ms", "separator_read_ms",
    ):
        value.setdefault(field, 0.0)
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
        normal_preads = int(value.get("normal_kv_preads", 0) or 0)
        if normal_preads <= 0 or normal_preads % 2:
            raise ValueError("FullLoad normal K/V preads must be positive and even")
        layers = normal_preads // 2
        value["contiguous_runs_per_layer"] = [1] * layers
        value["contiguous_runs_per_layer_mean"] = 1.0
        value["normal_kv_read_ms"] = float(value.get("ssd_read_ms", 0.0))
        value["selected_chunk_io_ms"] = None
    if method_key == "rekv_chunk25" and "selected_payload_bytes" in result:
        value["normal_kv_read_bytes"] = int(result["selected_payload_bytes"])
        value["separator_read_bytes"] = int(result["separator_bytes"])
        value["probe_read_bytes"] = 0
        value["selected_chunk_payload_read_bytes"] = int(
            result["selected_payload_bytes"])
        value["contiguous_runs_per_layer"] = [
            int(layer["contiguous_runs"])
            for layer in result["retrieval_layer_log"]]
        value["contiguous_runs_per_layer_mean"] = float(
            np.mean(value["contiguous_runs_per_layer"]))
        value["online_selector_total_ms"] = float(
            result.get("q_rep_ms", 0.0) + result.get("similarity_ms", 0.0)
            + result.get("topk_ms", 0.0)
            + result.get("selection_d2h_ms", 0.0))
    else:
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
    """Ordinary answer-producing multimodal request with same-forward capture."""
    if capture_kind not in {"none", "raster", "mpic", "rekv", "image_only"}:
        raise ValueError(f"invalid capture kind: {capture_kind}")
    capture_cache = capture_kind != "none"
    vision = VisionForwardCapture(
        runner, capture_saliency=(capture_kind == "image_only"))
    hidden_capture = raw_capture = None
    with vision:
        torch.cuda.synchronize()
        request_started = time.perf_counter()
        prompt_started = time.perf_counter()
        prompt, history, history_entries = prompt_factory()
        prompt_ms = (time.perf_counter() - prompt_started) * 1e3
        enc_cpu, processor_timing = qa_base()._combined_processor(
            runner, image, prompt)
        v_start, v_num = runner.visual_span(enc_cpu["input_ids"])
        if capture_kind in {"raster", "mpic"}:
            hidden_capture = DecoderVisualHiddenCapture(
                runner, v_start, v_num)
        elif capture_kind == "rekv":
            raw_capture = ReKVCapture(runner, v_start, v_num)
        with ExitStack() as stack:
            if hidden_capture is not None:
                stack.enter_context(hidden_capture)
            if raw_capture is not None:
                stack.enter_context(raw_capture)
            h2d_started = time.perf_counter()
            enc_device = runner.to_device(enc_cpu)
            torch.cuda.synchronize()
            h2d_ms = (time.perf_counter() - h2d_started) * 1e3
            result = server.recompute(
                enc_device, return_past_key_values=capture_cache)
            returned = time.perf_counter()
        del enc_device
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
        "rekv_capture": raw_capture,
    }
    return result, diagnostics


def _dispatch_stored(server, method_key: str, context, suffix_ids,
                     image_id: str):
    if method_key == "fullload":
        return server.request(
            context, mode="fullload", cold=False, suffix_ids=suffix_ids)
    if method_key == "ours25":
        return server.request_cvpr25(
            context, static=None, budget=0.25, mode="prefix",
            sep_policy="sidecar", cold=False, seed=SEED,
            image_id=image_id, suffix_ids=suffix_ids,
            expected_prefix_layout="visionzip_image_only")
    raise ValueError(f"not a legacy stored method: {method_key}")


def render_rekv_retrieval_text(history: str, turn_id: int,
                               current_question: str) -> str:
    """Stage A sees exact causal history and current question, never gold."""
    line = f"Current question Q{turn_id}: {current_question}"
    return f"{history}\n{line}" if history else line


def _run_stored_request(
    runner,
    server,
    context,
    prompt_factory: Callable[[], tuple[str, str, list[dict[str, Any]]]],
    *,
    method_key: str,
    image_id: str,
    full_visual_bytes: int,
    turn_id: int,
    current_question: str,
    mpic_server: MPICServer,
    rekv_server: ReKVServer,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One OS-page-cache-conditioned request with full outer TTFT."""
    with qa_base()._NoVisionForward(runner) as guard:
        conditioned_at = time.perf_counter()
        context.reader.drop_all()
        conditioned_done = time.perf_counter()
        torch.cuda.synchronize()
        request_started = time.perf_counter()
        prompt_started = time.perf_counter()
        prompt, history, history_entries = prompt_factory()
        prompt_ms = (time.perf_counter() - prompt_started) * 1e3
        retrieval_text = None
        if method_key in {"fullload", "ours25"}:
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
            phases = {
                "prompt_build_ms": float(prompt_ms),
                "tokenization_ms": float(token_ms),
                "image_preprocess_ms": 0.0,
                "input_prepare_ms": float(prepare_ms),
                "input_h2d_ms": float(h2d_ms),
                "processor_total_ms": None,
            }
        elif method_key == "mpic32":
            result = mpic_server.request(
                context, prompt_text=prompt, cold=False)
            result["answer"] = str(result["prediction"])
            result["generated_tokens"] = int(result["generated_token_count"])
            phases = {
                "prompt_build_ms": float(prompt_ms),
                "tokenization_ms": None,
                "image_preprocess_ms": 0.0,
                "input_prepare_ms": None,
                "input_h2d_ms": float(result["h2d_ms"]),
                "processor_total_ms": None,
            }
        elif method_key == "rekv_chunk25":
            retrieval_text = render_rekv_retrieval_text(
                history, turn_id, current_question)
            result = rekv_server.request(
                context, prompt_text=prompt,
                retrieval_text=retrieval_text, cold=False)
            if (int(result["stage_b_payload_read_bytes"]) != 0
                    or int(result["duplicate_read_bytes"]) != 0):
                raise AssertionError("ReKV reread selected payload")
            phases = {
                "prompt_build_ms": float(prompt_ms),
                "tokenization_ms": None,
                "image_preprocess_ms": 0.0,
                "input_prepare_ms": None,
                "input_h2d_ms": None,
                "processor_total_ms": None,
            }
        else:
            raise ValueError(f"invalid stored method: {method_key}")
        returned = time.perf_counter()
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
    # Diagnostics happen after the timed response, so MPIC/ReKV do not pay
    # duplicate tokenization that is absent in their validated server paths.
    suffix_cpu = _suffix_from_prompt(runner, prompt)
    if retrieval_text is not None:
        if [int(value) for value in suffix_cpu.tolist()] != [
                int(value) for value in result["answer_suffix_token_ids"]]:
            raise AssertionError("ReKV Stage B suffix differs from canonical prompt")
        q_ids = runner.processor.tokenizer(
            retrieval_text, return_tensors="pt").input_ids[0]
        if [int(value) for value in q_ids.tolist()] != [
                int(value) for value in result["question_token_ids"]]:
            raise AssertionError("ReKV Stage A token ids differ from causal text")
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
        "retrieval_text": retrieval_text,
        "retrieval_text_sha256": (
            sha256_text(retrieval_text) if retrieval_text is not None else None),
    }
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
        "probe_sidecar_bytes": int(meta.get("bytes_probe_sidecar", 0)),
        "separator_sidecar_bytes": int(meta.get("bytes_separator_sidecar", 0)),
        "num_layers": int(meta["num_layers"]),
        "num_heads": int(meta["num_heads"]),
        "probe_heads": int(meta.get("probe_heads", 0)),
        "chunk_size": int(meta["chunk_size"]),
        "n_chunks_per_layer": int(meta["n_chunks_per_layer"]),
        "v_token_num": int(meta["v_token_num"]),
        "n_spatial": int(meta.get("n_spatial",
                             int(meta["v_token_num"]) - len(meta.get("newline_idx", [])))),
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
        "hashes": dict(persisted.get("hashes", {})),
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
    mpic_manifest: Mapping[str, Any] | None,
    rekv_manifest: Mapping[str, Any] | None,
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
    store_by_method = {
        "fullload": "raster", "mpic32": "mpic",
        "rekv_chunk25": "rekv", "ours25": "image_only",
    }
    store_kind = store_by_method.get(method_key) if cache_hit else None
    manifests = {
        "raster": raster_manifest, "mpic": mpic_manifest,
        "rekv": rekv_manifest, "image_only": ours_manifest,
    }
    store_manifest = manifests.get(store_kind)
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
        "mpic_store_id": (mpic_manifest.get("meta_sha256")
                          if mpic_manifest is not None else None),
        "rekv_store_id": (rekv_manifest.get("meta_sha256")
                          if rekv_manifest is not None else None),
        "retrieval_text": diagnostics.get("retrieval_text"),
        "retrieval_text_sha256": diagnostics.get("retrieval_text_sha256"),
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
        "rater_scope": (
            "causal_history_plus_current_question"
            if method_key == "rekv_chunk25" else "none"),
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
    row["method"] = METHODS[method_key]["label"]
    row["method_id"] = METHODS[method_key]["method_id"]
    row["retry_count"] = int(flattened.get("retry_count", 0))
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
    manifests: Mapping[str, Mapping[str, Any]],
) -> None:
    store_key = {
        "fullload": "raster", "mpic32": "mpic",
        "rekv_chunk25": "rekv", "ours25": "image_only",
    }[method]
    store = manifests[store_key]
    layers = int(store["num_layers"])
    total = int(row.get("ssd_read_bytes", -1))
    preads = int(row.get("ssd_preads", row.get("pread_count", -1)))
    normal = int(row.get("normal_kv_read_bytes", -1))
    separator = int(row.get("separator_read_bytes", -1))
    probe = int(row.get("probe_read_bytes", -1))
    normal_preads = int(row.get("normal_kv_preads", -1))
    separator_preads = int(row.get("separator_preads", -1))
    probe_preads = int(row.get("probe_preads", -1))
    if min(total, preads, normal, separator, probe,
           normal_preads, separator_preads, probe_preads) < 0:
        raise ValueError(f"{method} negative SSD counter")
    if method == "mpic32":
        embedding = int(row.get("embedding_read_bytes", -1))
        embedding_preads = int(row.get("embedding_preads", -1))
        n_visual = int(store["v_token_num"])
        if (embedding <= 0 or embedding_preads <= 0
                or total != normal + embedding
                or preads != normal_preads + embedding_preads
                or normal <= 0
                or int(row.get("n_recomputed_image_tokens", -1))
                   != min(32, n_visual)
                or int(row.get("n_reused_image_tokens", -1))
                   != n_visual - min(32, n_visual)
                or float(row.get("retained_image_context_ratio", -1)) != 1.0
                or row.get("same_source_target_context") is not True
                or row.get("decode_cache_append_exact") is not True
                or row.get("source_payload_hash_before")
                   != row.get("source_payload_hash_after")):
            raise ValueError("MPIC-32 exact partial-recompute contract failed")
        return
    if total != normal + separator + probe:
        raise ValueError(f"{method} SSD byte decomposition mismatch")
    if preads != normal_preads + separator_preads + probe_preads:
        raise ValueError(f"{method} pread decomposition mismatch")
    visual = int(store["visual_kv_bytes"])
    n_visual = int(store["v_token_num"])
    denominator = layers * n_visual
    if denominator <= 0 or visual % denominator:
        raise ValueError("store visual token-row size is invalid")
    row_bytes = visual // denominator
    selected = row.get("selected_chunk_ids_per_layer")
    selected_bytes = None
    if isinstance(selected, list) and selected:
        selected_rows = sum(
            min((int(ci) + 1) * int(store["chunk_size"]), n_visual)
            - int(ci) * int(store["chunk_size"])
            for layer in selected for ci in layer)
        selected_bytes = selected_rows * row_bytes
    if method == "fullload":
        if (total != visual or normal != visual
                or separator != 0 or probe != 0
                or normal_preads != 2 * layers
                or separator_preads != 0 or probe_preads != 0
                or row.get("contiguous_runs_per_layer") != [1] * layers
                or int(row.get("n_raters", -1)) != 0
                or int(row.get("query_score_calls", 0) or 0) != 0):
            raise ValueError("FullLoad full-raster I/O contract failed")
    elif method == "ours25":
        if (normal <= 0 or normal != selected_bytes
                or separator != int(store["separator_sidecar_bytes"])
                or probe != 0 or normal_preads != 2 * layers
                or separator_preads != 1 or probe_preads != 0
                or row.get("contiguous_runs_per_layer") != [1] * layers
                or int(row.get("query_score_calls", 0) or 0) != 0):
            raise ValueError("Ours25 fixed-prefix I/O contract failed")
    elif method == "rekv_chunk25":
        runs = row.get("contiguous_runs_per_layer")
        if (not isinstance(runs, list) or len(runs) != layers
                or normal <= 0 or normal != selected_bytes
                or separator != int(store["separator_sidecar_bytes"])
                or probe != 0 or normal_preads != 2 * sum(map(int, runs))
                or separator_preads != 1 or probe_preads != 0
                or int(row.get("stage_a_payload_read_bytes", -1)) != total
                or int(row.get("stage_b_payload_read_bytes", -1)) != 0
                or int(row.get("duplicate_read_bytes", -1)) != 0
                or len(row.get("actual_attention_key_lengths", [])) != layers):
            raise ValueError("ReKV exact selection/handoff I/O contract failed")
    else:
        raise ValueError(f"unexpected cache method: {method}")


def validate_image_rows(
    rows: Sequence[Mapping[str, Any]],
    group: Mapping[str, Any],
    protocol: str,
    store_manifests: Mapping[str, Mapping[str, Any]],
    *,
    seed: int = SEED,
) -> dict[str, Any]:
    """Replay each generated history and exact image-store use."""
    if protocol != "generated_history":
        raise ValueError("this experiment permits Generated-History only")
    dialogs = list(group["dialogs"])
    expected_keys = _expected_row_keys(dialogs)
    observed_keys = [
        (str(row.get("dialog_id")), int(row.get("turn_id", -1)),
         str(row.get("method_key"))) for row in rows]
    if observed_keys != expected_keys or len(set(observed_keys)) != len(rows):
        raise ValueError("request coverage/order or uniqueness mismatch")
    if set(store_manifests) != set(STORE_KEYS):
        raise ValueError("image artifact must contain four method-owned stores")
    raster, mpic, rekv, image_only = (
        store_manifests[key] for key in STORE_KEYS)
    if (raster["physical_layout"] != "raster"
            or mpic["physical_layout"] != "canonical_raster"
            or rekv["physical_layout"] != "canonical_raster"
            or image_only["physical_layout"] != "visionzip_image_only"):
        raise ValueError("one store has the wrong physical layout")
    if (not image_only.get("permutation_sha256")
            or not image_only.get(
                "selected_prefix_original_token_ids_sha256")):
        raise ValueError("Ours store lacks permutation provenance")
    for store in (mpic, rekv, image_only):
        if (int(store["visual_kv_bytes"]) != int(raster["visual_kv_bytes"])
                or int(store["n_chunks_per_layer"])
                   != int(raster["n_chunks_per_layer"])
                or int(store["v_token_num"]) != int(raster["v_token_num"])):
            raise ValueError("method stores disagree on visual geometry")
    n_chunks = int(raster["n_chunks_per_layer"])
    expected_k = budget_chunk_count(n_chunks, 0.25)
    cursor = 0
    cache_hits = {method: 0 for method in METHOD_KEYS}
    ours_fingerprints: set[str] = set()
    sessions: set[str] = set()
    contexts: set[str] = set()
    for dialog in dialogs:
        did = _dialog_id(dialog)
        order = list(method_order(int(dialog["global_dialog_ordinal"]), seed))
        generated: dict[str, dict[int, dict[str, str]]] = {
            method: {} for method in METHOD_KEYS}
        dialogue_sessions: set[str] = set()
        dialogue_contexts: set[str] = set()
        for turn_id in (1, 2, 3):
            block = rows[cursor:cursor + len(METHOD_KEYS)]
            cursor += len(METHOD_KEYS)
            if [row["method_key"] for row in block] != order:
                raise ValueError("balanced method order changed")
            q1_prompts, q1_predictions, q1_tokens = set(), set(), set()
            for row in block:
                method = str(row["method_key"])
                prompt, history, entries = render_causal_prompt(
                    dialog, turn_id, protocol, method_key=method,
                    generated_predictions=generated[method])
                if (row.get("protocol") != protocol
                        or row.get("history_text") != history
                        or row.get("history_entries") != entries
                        or row.get("history_answers")
                           != [item["answer"] for item in entries]
                        or row.get("history_source_request_ids")
                           != [item["source_logical_request_id"]
                               for item in entries]
                        or row.get("history_source_physical_execution_ids")
                           != [item["source_physical_execution_id"]
                               for item in entries]
                        or row.get("prompt") != prompt
                        or row.get("prompt_sha256") != sha256_text(prompt)
                        or row.get("history_text_sha256") != sha256_text(history)
                        or row.get("history_turn_ids") != list(range(1, turn_id))
                        or int(row.get("future_leakage", -1)) != 0):
                    raise ValueError("method-local causal prompt/history mismatch")
                if (row.get("method") != METHODS[method]["label"]
                        or row.get("method_id")
                           != METHODS[method]["method_id"]
                        or row.get("logical_request_id") != logical_request_id(
                            protocol, did, turn_id, method)
                        or row.get("physical_execution_id")
                           != row.get("execution_id")
                        or row.get("status") != "ok"
                        or int(row.get("retry_count", 0)) != 0
                        or int(row.get("generated_token_count", 0)) < 1
                        or len(row.get("generated_token_ids", []))
                           != int(row.get("generated_token_count", -1))
                        or int(row.get("input_token_count", 0)) < 1
                        or float(row.get("correct", -1))
                           != strict_gqa_score(
                               row.get("prediction", ""),
                               row.get("gold_answer", ""))):
                    raise ValueError("request identity/output/score invalid")
                if not 0 < float(row.get("end_to_end_ttft_ms", 0)):
                    raise ValueError("request lacks inclusive TTFT")
                is_cache = turn_id >= 2 and method != "recompute"
                if bool(row.get("cache_hit")) != is_cache:
                    raise ValueError("cache-hit flag mismatch")
                if is_cache:
                    cache_hits[method] += 1
                    store_key = {
                        "fullload": "raster", "mpic32": "mpic",
                        "rekv_chunk25": "rekv", "ours25": "image_only"}[method]
                    if (row.get("request_path") != "stored_visual_kv"
                            or int(row.get("vision_forward_count", -1)) != 0
                            or row.get("store_kind") != store_key
                            or row.get("store_id")
                               != store_manifests[store_key]["meta_sha256"]
                            or row.get(
                                "page_cache_conditioning_excluded_from_ttft")
                               is not True):
                        raise ValueError("stored request provenance mismatch")
                    _validate_cache_io(
                        row, method=method, manifests=store_manifests)
                    if method in {"ours25", "rekv_chunk25"}:
                        _validate_selected_chunks(
                            row, expected_k=expected_k,
                            total_chunks=n_chunks,
                            fixed_prefix=(method == "ours25"))
                    if method == "ours25":
                        ours_fingerprints.add(
                            str(row["selection_fingerprint_sha256"]))
                        if (row.get("store_permutation_sha256")
                                != image_only["permutation_sha256"]):
                            raise ValueError("Ours store permutation changed")
                    if method == "rekv_chunk25":
                        wanted_retrieval = render_rekv_retrieval_text(
                            history, turn_id, str(_turn(dialog, turn_id)["question"]))
                        if (row.get("retrieval_text") != wanted_retrieval
                                or row.get("retrieval_text_sha256")
                                   != sha256_text(wanted_retrieval)
                                or int(row.get(
                                    "normal_selected_chunk_count", -1))
                                   != expected_k
                                or row.get("similarity_mode")
                                   != "official_code_dot"):
                            raise ValueError("ReKV causal Stage-A query mismatch")
                else:
                    if (row.get("request_path") != "normal_multimodal_pixel"
                            or int(row.get("ssd_read_bytes", -1)) != 0
                            or int(row.get("vision_forward_count", -1)) != 1
                            or row.get("combined_suffix_ids_sha256")
                               != row.get("suffix_ids_sha256")):
                        raise ValueError("normal pixel request contract failed")
                dialogue_sessions.add(str(row["dialogue_session_id"]))
                dialogue_contexts.add(str(row["context_instance_id"]))
                q1_prompts.add(str(row["prompt_sha256"]))
                q1_predictions.add(str(row["prediction"]))
                q1_tokens.add(int(row["first_token_id"]))
                generated[method][turn_id] = {
                    "prediction": str(row["prediction"]),
                    "logical_request_id": str(row["logical_request_id"]),
                    "physical_execution_id": str(
                        row["physical_execution_id"]),
                }
            if turn_id == 1 and (len(q1_prompts) != 1
                                 or len(q1_predictions) != 1
                                 or len(q1_tokens) != 1):
                raise ValueError("Turn-1 five-arm normal-pixel fairness failed")
        if len(dialogue_sessions) != 1 or len(dialogue_contexts) != 1:
            raise ValueError("dialogue session/context identity changed")
        session = next(iter(dialogue_sessions))
        context = next(iter(dialogue_contexts))
        if session in sessions or context in contexts:
            raise ValueError("dialogue session/context reused")
        sessions.add(session)
        contexts.add(context)
    if cursor != len(rows):
        raise ValueError("image request cursor mismatch")
    if any(cache_hits[key] != (0 if key == "recompute" else 2 * len(dialogs))
           for key in METHOD_KEYS):
        raise ValueError("per-method image cache-hit count mismatch")
    if len(ours_fingerprints) != 1:
        raise ValueError("Ours fixed physical prefix varied within image")
    return {
        "passed": True,
        "protocol": protocol,
        "n_dialogs": len(dialogs),
        "n_rows": len(rows),
        "expected_rows": len(dialogs) * 3 * len(METHOD_KEYS),
        "cache_hit_rows": sum(cache_hits.values()),
        "cache_hits_per_method": cache_hits,
        "failed_rows": 0,
        "duplicate_rows": 0,
        "strict_scores_recomputed": True,
        "future_leakage": 0,
        "method_local_generated_history_validated": True,
        "turn1_five_arm_fairness": True,
        "ours_selection_fixed_within_image": True,
        "rekv_stage_b_duplicate_payload_bytes": 0,
        "normal_chunk_budget_per_layer": expected_k,
    }


def _capacity_guard(path: Path, *, reserve_bytes: int,
                    extra_headroom_bytes: int = 0) -> dict[str, Any]:
    """Guard actual free bytes; preserve legacy used-percent as a diagnostic."""
    usage = shutil.disk_usage(path)
    required = int(reserve_bytes) + int(extra_headroom_bytes)
    used_pct = 100.0 * float(usage.used) / float(usage.used + usage.free)
    if int(usage.free) < required:
        raise RuntimeError(
            f"free-space guard failed: free={usage.free}, required={required}")
    return {
        "disk_total_bytes": int(usage.total),
        "disk_used_bytes": int(usage.used),
        "disk_free_bytes": int(usage.free),
        "disk_used_percent": float(used_pct),
        "reserve_bytes": int(reserve_bytes),
        "extra_headroom_bytes": int(extra_headroom_bytes),
        "guard_basis": "free_bytes_only_with_30_GiB_reserve_and_build_headroom",
    }


def _open_stores(image_store: Path, runner) -> dict[str, Any]:
    ImageContext, _ = _runtime_classes()
    opened: dict[str, Any] = {}
    try:
        opened["raster"] = ImageContext(
            image_store / "raster", runner.model.device,
            drop_cache=True, require_v_hidden=True)
        opened["raster"].validate_qa_select_layout()
        opened["mpic"] = MPICContext(
            image_store / "mpic", runner.model.device, runner=runner)
        opened["rekv"] = ReKVContext(
            image_store / "rekv", runner.model.device, runner=runner)
        opened["image_only"] = ImageContext(
            image_store / "image_only", runner.model.device,
            drop_cache=True, require_v_hidden=False)
        opened["image_only"].validate_prefix_layout("visionzip_image_only")
        return opened
    except BaseException:
        for context in opened.values():
            context.close()
        raise


def _rekv_activation(context: ReKVContext, dialog_id: str) -> dict[str, Any]:
    return {
        "dialog_id": dialog_id,
        "metadata_activation_ms": float(context.metadata_activation_ms),
        "initial_context_activation_ms": float(
            context.initial_context_activation_ms),
        "activation_total_ms": float(context.activation_total_ms),
        "metadata_gpu_bytes_total": int(context.metadata_gpu_bytes_total),
        "initial_context_gpu_bytes": int(context.initial_context_gpu_bytes),
        "initial_context_cpu_bytes": int(context.initial_context_cpu_bytes),
        "timing_semantics": "image-context activation outside request TTFT",
    }


def _execute_image_group(
    *,
    runner,
    server,
    mpic_server: MPICServer,
    rekv_server: ReKVServer,
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
    """Run all dialogues for one image and publish only after complete checks."""
    if protocol != "generated_history":
        raise ValueError("Gold-History is excluded from this experiment")
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
    store_dirs = {kind: image_store / kind for kind in STORE_KEYS}
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
    activation_events: list[dict[str, Any]] = []
    full_visual_bytes: int | None = None
    store_build_counts = {kind: 0 for kind in STORE_KEYS}
    source_kind = {
        "fullload": "raster", "mpic32": "mpic",
        "rekv_chunk25": "rekv", "ours25": "image_only",
    }

    try:
        for dialog_index, dialog in enumerate(dialogs):
            did = _dialog_id(dialog)
            if dialog_index > 0:
                if set(contexts) != set(STORE_KEYS):
                    raise AssertionError("next dialogue started before four stores")
                for context in contexts.values():
                    context.close()
                contexts = _open_stores(image_store, runner)
                activation_events.append(_rekv_activation(
                    contexts["rekv"], did))
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
                            capture_kind = source_kind.get(method_key, "none")
                            if capture_kind != "none":
                                persistence_source = capture_kind
                        result, diagnostics = _run_pixel_request(
                            runner, server, image, prompt_factory,
                            capture_kind=capture_kind)
                        captured_cache = result.pop(
                            "captured_past_key_values", None)
                        if capture_kind != "none":
                            if captured_cache is None:
                                raise AssertionError(
                                    f"{method_key} T1 omitted captured K/V")
                            encoded = diagnostics["enc_cpu"]
                            extra_metadata = {
                                "dataset": DATASET,
                                "history_experiment_protocol": protocol,
                                "source_dialog_id": did,
                                "source_turn_id": 1,
                                "source_method_key": method_key,
                                "source_execution_id": execution_id,
                                "turn1_question_id": str(turn["question_id"]),
                                "turn1_request_id": logical_request_id(
                                    protocol, did, 1, method_key),
                                "store_lifecycle": (
                                    "image_local_ephemeral_after_immutable_commit"),
                                "future_questions_used_for_layout": 0,
                            }
                            common = {
                                "image_id": image_id,
                                "model_id": runner.model_id,
                                "chunk_size": CHUNK_SIZE,
                                "image_input_sha256": diagnostics[
                                    "image_input_sha256"],
                                "extra_metadata": extra_metadata,
                            }
                            destination = store_dirs[capture_kind]
                            if capture_kind == "raster":
                                hidden = diagnostics["hidden_capture"]
                                persisted = persist_captured_raster_prefix(
                                    runner, captured_cache,
                                    encoded["input_ids"], encoded["image_sizes"][0],
                                    hidden.result_cpu(), destination,
                                    probe_heads=PROBE_HEADS,
                                    hidden_capture_stats=hidden, **common)
                                context = ImageContext(
                                    destination, runner.model.device,
                                    drop_cache=True, require_v_hidden=True)
                                context.validate_qa_select_layout()
                            elif capture_kind == "mpic":
                                hidden = diagnostics["hidden_capture"]
                                persisted = persist_captured_mpic_prefix(
                                    runner, captured_cache,
                                    encoded["input_ids"], encoded["image_sizes"][0],
                                    hidden.result_cpu(), destination,
                                    hidden_capture_stats=hidden, **common)
                                context = MPICContext(
                                    destination, runner.model.device,
                                    runner=runner)
                            elif capture_kind == "rekv":
                                raw = diagnostics["rekv_capture"]
                                persisted = persist_captured_rekv_prefix(
                                    runner, captured_cache,
                                    encoded["input_ids"], encoded["image_sizes"][0],
                                    raw, destination, **common)
                                context = ReKVContext(
                                    destination, runner.model.device,
                                    runner=runner)
                                activation_events.append(_rekv_activation(
                                    context, did))
                            elif capture_kind == "image_only":
                                vision = diagnostics["vision_capture"]
                                persisted = persist_captured_visual_prefix(
                                    runner, captured_cache,
                                    encoded["input_ids"], encoded["image_sizes"][0],
                                    vision.result_cpu(), destination,
                                    capture_stats=vision, **common)
                                context = ImageContext(
                                    destination, runner.model.device,
                                    drop_cache=True, require_v_hidden=False)
                                context.validate_prefix_layout(
                                    "visionzip_image_only")
                            else:
                                raise AssertionError("invalid capture source")
                            del captured_cache
                            contexts[capture_kind] = context
                            store_build_counts[capture_kind] += 1
                            store_manifests[capture_kind] = _store_manifest(
                                destination, persisted, context.meta)
                            persistence[capture_kind] = _persistence_summary(
                                persisted, source_dialog_id=did,
                                source_method_key=method_key,
                                source_execution_id=execution_id)
                            source_execution_ids[capture_kind] = execution_id
                            bytes_here = int(context.meta["bytes_visual_kv"])
                            if (full_visual_bytes is not None
                                    and bytes_here != full_visual_bytes):
                                raise AssertionError(
                                    "method stores disagree on visual KV bytes")
                            full_visual_bytes = bytes_here
                        elif captured_cache is not None:
                            raise AssertionError(
                                "non-source pixel request retained captured K/V")
                    else:
                        if set(contexts) != set(STORE_KEYS):
                            raise AssertionError(
                                "cache request preceded four method stores")
                        context_key = source_kind[method_key]
                        result, diagnostics = _run_stored_request(
                            runner, server, contexts[context_key],
                            prompt_factory, method_key=method_key,
                            image_id=image_id,
                            full_visual_bytes=int(full_visual_bytes),
                            turn_id=turn_id,
                            current_question=str(turn["question"]),
                            mpic_server=mpic_server,
                            rekv_server=rekv_server)

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
                        mpic_manifest=store_manifests.get("mpic"),
                        rekv_manifest=store_manifests.get("rekv"),
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
                    diagnostics.pop("rekv_capture", None)
                    gc.collect()

                if turn_id == 1:
                    if (len({row["prompt_sha256"] for row in request_rows}) != 1
                            or len({row["prediction"]
                                    for row in request_rows}) != 1
                            or len({row["first_token_id"]
                                    for row in request_rows}) != 1):
                        raise AssertionError(
                            "Turn-1 five-arm normal-pixel fairness failed")
                rows.extend(request_rows)

        if (set(store_manifests) != set(STORE_KEYS)
                or set(persistence) != set(STORE_KEYS)
                or store_build_counts != {kind: 1 for kind in STORE_KEYS}):
            raise AssertionError("image did not build all four stores once")
        capacity_after_build = _capacity_guard(
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
            "history_policy": "method_local_generated",
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
            "rekv_metadata_activation": {
                "events": activation_events,
                "total_activation_ms": sum(
                    event["activation_total_ms"] for event in activation_events),
                "timing_semantics": "outside every measured request TTFT",
            },
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
        raise ValueError("resume artifact omits rows or four-store manifests")
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
            kind: 1 for kind in STORE_KEYS}:
        raise ValueError("resume artifact did not build four stores once")
    persistence = artifact.get("persistence_overhead")
    if not isinstance(persistence, Mapping) or set(persistence) != set(STORE_KEYS):
        raise ValueError("resume artifact omits persistence provenance")
    source_ids = artifact.get("source_execution_ids")
    if not isinstance(source_ids, Mapping) or set(source_ids) != set(STORE_KEYS):
        raise ValueError("resume artifact omits store source executions")
    activation = artifact.get("rekv_metadata_activation")
    if (not isinstance(activation, Mapping)
            or len(activation.get("events", [])) != len(dialogs)
            or not all(event.get("dialog_id") == _dialog_id(dialog)
                       for event, dialog in zip(activation["events"], dialogs))):
        raise ValueError("resume artifact omits ReKV activation evidence")
    by_physical = {row["physical_execution_id"]: row for row in rows}
    expected_source_methods = {
        "raster": "fullload", "mpic": "mpic32",
        "rekv": "rekv_chunk25", "image_only": "ours25"}
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
        "history_policy": "method_local_generated",
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
        "planned_logical_requests_total": counts["requests_total"],
        "planned_physical_executions_total": counts["requests_total"],
        "request_counts": counts,
        "shard_size": int(args.shard_size),
        "n_shards": int(n_shards),
        "seed": int(args.seed),
        "method_keys": list(METHOD_KEYS),
        "methods": METHODS,
        "method_order_policy": (
            "zero-based cyclic rotation by frozen global dialogue ordinal; "
            "identical across the three turns of each dialogue"),
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
        "mpic_configuration": {
            "k_recompute": 32,
            "same_source_target_image_position_required": True,
            "method_label": "MPIC-32 (adapted)",
        },
        "rekv_configuration": {
            "chunk_size": CHUNK_SIZE,
            "normal_chunk_budget": 0.25,
            "similarity": "official_code_dot",
            "stage_a": "causal_generated_history_plus_current_question",
            "stage_b": "exact_canonical_MT_GQA_answer_prompt",
            "metadata_activation_inside_ttft": False,
            "official_commit": "1fd9a3dbf5dbff7f27069ae2f4463674c495e830",
        },
        "ours_configuration": {
            "physical_layout": "visionzip_image_only",
            "normal_chunk_budget": 0.25,
            "selection": "fixed_first_k_prefix",
            "online_query_scoring": False,
        },
        "turn1_policy": (
            "all five methods execute independent normal pixel requests; "
            "each of FullLoad, MPIC-32, ReKV-Chunk25, Ours25 captures its "
            "own first-dialogue T1 and persists its own image store"),
        "later_turn_policy": (
            "ReComp reprocesses pixels; FullLoad, MPIC-32, ReKV-Chunk25 "
            "and Ours25 use independent SSD stores with their validated algorithms"),
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
        "context_instance_id_semantics": (
            "dialogue-scoped four-store context group; requests retain "
            "fresh per-request text KV and no prior-turn text KV reuse"),
        "rekv_metadata_activation_policy": (
            "active-image GPU representatives and initial raw KV are loaded "
            "when the image context opens outside every request TTFT"),
        "image_at_a_time_temporary_store": True,
        "four_physical_stores_per_image": True,
        "temporary_payload_deleted_only_after_immutable_image_artifact": True,
        "unmeasured_warmup": dict(warmup),
        "run_dir": str(args.run_dir),
        "temp_root": str(args.temp_root),
        "capacity_guard": {
            "min_free_after_gib": float(args.min_free_after_gib),
            "build_headroom_gib": BUILD_HEADROOM_GIB,
            "guard_basis": "free_bytes_only; used_percent_recorded_descriptively",
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
        "n_dialogs", "n_images", "n_requests", "seed", "method_keys",
        "shard_size", "n_shards", "max_new_tokens", "model_revision",
        "model", "load_4bit", "quantization", "compute_dtype",
        "attention", "decoding", "chunk_size", "probe_heads",
        "mpic_configuration", "rekv_configuration",
        "ours_configuration", "turn1_policy", "later_turn_policy",
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
        "protocol": "generated_history",
        "history_policy": "method_local_generated",
        "dialogues_file_sha256": workload["dialogues_file_sha256"],
        "source_full_workload_sha256": workload[
            "source_full_workload_sha256"],
        "selected_workload_sha256": workload[
            "selected_workload_sha256"],
        "n_dialogs": int(workload["n_dialogs"]),
        "n_turns": int(workload["n_turns"]),
        "n_images": len(groups),
        "n_requests": counts["requests_per_protocol"],
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
        "main_ttft_field": "end_to_end_ttft_ms",
        "persistence_in_main_ttft": False,
        "four_physical_stores_per_image": True,
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
    mpic_server = MPICServer(
        runner, k_recompute=32, max_new_tokens=MAX_NEW_TOKENS)
    rekv_server = ReKVServer(
        runner, max_new_tokens=MAX_NEW_TOKENS)
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
        capacity_before = _capacity_guard(
            args.temp_root, reserve_bytes=reserve_bytes,
            extra_headroom_bytes=headroom_bytes)
        artifact = _execute_image_group(
            runner=runner, server=server,
            mpic_server=mpic_server, rekv_server=rekv_server,
            group=group, workload=workload,
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
            f"four stores {total_store_bytes / 1e9:.2f} GB", flush=True)
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

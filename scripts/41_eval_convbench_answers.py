"""ConvBench generated-history answer and system measurement, one conversation at a time.

Only ``--full-run`` permits all 577 conversations.  A completed conversation
is an immutable, content-hashed JSON artifact.  Temporary Visual KV is
experiment-owned and removed only after that artifact is durable.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mmimpress.config import CHUNK_SIZE, MODEL_ID  # noqa: E402
from mmimpress.convbench import validate_index as validate_official_index  # noqa: E402
from mmimpress.model import LlavaRunner  # noqa: E402
from mmimpress.piggyback import (  # noqa: E402
    VisionForwardCapture, deterministic_method_rotation,
    persist_captured_visual_prefix, sha256_file, stable_json_sha256,
    _sampled_store_sha256,
)

SCHEMA_VERSION = "convbench-generated-history-answer-v2"
REQUEST_CHECKPOINT_SCHEMA = "convbench-request-checkpoint-v1"
CONTEXT_POLICY_ID = "native_overflow_no_truncation_v1"
METHOD_KEYS = ("recompute", "fullload", "prefix25", "prefix45")
BUDGETS = {"recompute": None, "fullload": 1.0,
           "prefix25": 0.25, "prefix45": 0.45}
SEED = 1234
EXPECTED_CONVERSATIONS = 577
PROMPT_FORMAT = "vicuna-user-assistant-generated-history-v1"
VICUNA_SYSTEM = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the "
    "human's questions."
)


def _helper(filename: str, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise ImportError(filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _visdial():
    return _helper("28_eval_visdial_turn1_piggyback.py", "_convbench_visdial_helpers")


def _mt():
    return _helper("37_eval_mt_gqa_full_shard.py", "_convbench_mt_helpers")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_id(value: Any) -> str:
    ident = str(value)
    if not ident or ident in {".", ".."} or "/" in ident or "\\" in ident:
        raise ValueError(f"unsafe conversation id {ident!r}")
    return ident


def load_index(index_path: Path) -> tuple[list[dict], str]:
    if index_path.is_symlink():
        raise ValueError("index must not be a symlink")
    index_path = index_path.resolve(strict=True)
    if not index_path.is_file():
        raise ValueError("index must be a regular file")
    payload = json.loads(index_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("index must be an object")
    adapter_config = json.loads((ROOT / "data/convbench/config.json").read_text())
    adapter_provenance = json.loads(
        (ROOT / "data/convbench/provenance.json").read_text())
    validate_official_index(
        payload, adapter_config, adapter_provenance,
        ROOT / "data/convbench_source")
    rows = payload.get("conversations", payload.get("dialogs"))
    if not isinstance(rows, list) or len(rows) != EXPECTED_CONVERSATIONS:
        raise ValueError("official ConvBench index must have 577 conversations")
    seen = set()
    for ordinal, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"conversation {ordinal} is not an object")
        cid = _safe_id(row.get("conversation_id"))
        if cid in seen:
            raise ValueError(f"duplicate conversation ID {cid}")
        seen.add(cid)
        turns = row.get("turns")
        if not isinstance(turns, list) or len(turns) != 3:
            raise ValueError(f"{cid}: expected exactly three turns")
        if [int(t.get("turn_id", -1)) for t in turns] != [1, 2, 3]:
            raise ValueError(f"{cid}: turn IDs must be 1, 2, 3")
        for turn in turns:
            if not isinstance(turn.get("question"), str) or not turn["question"].strip():
                raise ValueError(f"{cid}: empty question")
            if not isinstance(turn.get("reference_answer"), str):
                raise ValueError(f"{cid}: missing reference answer")
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        image_path = image_path.resolve(strict=True)
        if not image_path.is_file():
            raise ValueError(f"{cid}: image missing")
        row["image_path"] = str(image_path)
        row["global_conversation_ordinal"] = ordinal
    return rows, sha256_file(index_path)


def load_source_provenance() -> dict:
    path = ROOT / "data/convbench/provenance.json"
    provenance = json.loads(path.read_text())
    source_path = ROOT / provenance["source_workbook"]
    expected = str(provenance["source_workbook_sha256"])
    if sha256_file(source_path) != expected:
        raise ValueError("official ConvBench workbook SHA256 mismatch")
    pairwise_path = ROOT / provenance["official_pairwise"]
    if sha256_file(pairwise_path) != provenance["official_pairwise_sha256"]:
        raise ValueError("official ConvBench pairwise.npy SHA256 mismatch")
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT / "data/convbench_source"),
         "rev-parse", "HEAD"], text=True).strip()
    if commit != provenance["official_commit"]:
        raise ValueError("official ConvBench repository commit mismatch")
    return {
        "adapter_provenance_sha256": sha256_file(path),
        "source_workbook_sha256": expected,
        "official_repository_commit": commit,
        "official_pairwise_sha256": provenance["official_pairwise_sha256"],
    }


def source_image_fingerprint(dialogs: list[dict],
                             project_root: Path = ROOT) -> tuple[str, int]:
    """Hash the path and bytes of every unique image in the validated index.

    This freezes even images not yet processed when a run is resumed.  The
    path is relative to the project root so the digest does not depend on the
    absolute checkout location.
    """
    root = project_root.resolve()
    unique: dict[str, Path] = {}
    for dialog in dialogs:
        image = Path(dialog["image_path"]).resolve(strict=True)
        relative = image.relative_to(root).as_posix()
        unique[relative] = image
    entries = [{"image_path": relative, "sha256": sha256_file(unique[relative])}
               for relative in sorted(unique)]
    return stable_json_sha256(entries), len(entries)


def method_order(ordinal: int) -> tuple[str, ...]:
    # Same deterministic, balanced rotation used by the MT-GQA experiment.
    return deterministic_method_rotation(METHOD_KEYS, ordinal, 0)


def select_workload(dialogs: list[dict], limit: int, full_run: bool,
                    smoke_ids: str | None) -> tuple[list[dict], str]:
    if smoke_ids is None:
        return dialogs[:limit], "full" if full_run else "first_n"
    source_ids = [part.strip() for part in smoke_ids.split(",")]
    if (not source_ids or any(not item or not item.isdecimal()
                              for item in source_ids)
            or len(set(source_ids)) != len(source_ids)
            or len(source_ids) > 10):
        raise ValueError("--smoke-ids requires 1..10 unique numeric IDs")
    by_source_id = {str(dialog["source_id"]): dialog for dialog in dialogs}
    missing = [item for item in source_ids if item not in by_source_id]
    if missing:
        raise ValueError(f"source IDs absent from validated index: {missing}")
    return [by_source_id[item] for item in source_ids], "explicit_source_ids"


def render_prompt(questions: list[str], previous_answers: list[str]) -> str:
    """LLaVA Vicuna USER/ASSISTANT turns with image only in the first turn.

    Prior answers are generated by the same method.  ``</s>`` separates
    completed assistant turns as in the official LLaVA ``llava_v1`` template.
    The checkpoint and greedy decoding differ from official LLaVA 1.5 and
    are recorded as protocol differences in the run config.
    """
    if not 1 <= len(questions) <= 3 or len(previous_answers) != len(questions) - 1:
        raise ValueError("causal prompt requires N questions and N-1 answers")
    prompt = f"{VICUNA_SYSTEM} USER: <image>\n{questions[0]} ASSISTANT:"
    for index, answer in enumerate(previous_answers, 1):
        prompt += f" {answer}</s>USER: {questions[index]} ASSISTANT:"
    return prompt


def serialized_prior_history(prompt: str, question: str,
                             turn_id: int) -> str:
    if turn_id == 1:
        return ""
    current_tail = f"USER: {question} ASSISTANT:"
    if not prompt.endswith(current_tail):
        raise AssertionError("prompt does not end with current question")
    return prompt[:-len(current_tail)]


class _ModelForwardCounter:
    """Count model invocations, including all autoregressive decode steps."""

    def __init__(self, model):
        self.model = model
        self.calls = 0
        self.handle = None

    def __enter__(self):
        def count(_module, _args):
            self.calls += 1
        self.handle = self.model.register_forward_pre_hook(count)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.handle.remove()
        if exc_type is None and self.calls < 1:
            raise AssertionError("request did not invoke the model")
        return False


def _run_normal(runner, server, image, questions, answers,
                capture_saliency: bool, capture_cache: bool) -> dict:
    helpers = _visdial()
    capture = VisionForwardCapture(runner, capture_saliency=capture_saliency)
    with capture, _ModelForwardCounter(runner.model) as forwards:
        torch.cuda.synchronize()
        request_started = time.perf_counter()
        prompt_t0 = time.perf_counter()
        prompt = render_prompt(questions, answers)
        prompt_build_ms = (time.perf_counter() - prompt_t0) * 1e3
        enc_cpu, processing = helpers._exact_processor_call(runner, image, prompt)
        cap = _effective_cap(
            runner, server, int(enc_cpu["input_ids"].shape[1]))
        h2d_t0 = time.perf_counter()
        enc_device = runner.to_device(enc_cpu)
        torch.cuda.synchronize()
        input_h2d_ms = (time.perf_counter() - h2d_t0) * 1e3
        phase = {
            "prompt_build_ms": prompt_build_ms,
            "tokenization_ms": float(processing["tokenization_ms"]),
            "image_preprocess_ms": float(processing["image_preprocess_ms"]),
            "input_prepare_ms": float(processing["input_prepare_ms"]),
            "input_h2d_ms": input_h2d_ms,
            "processor_total_ms": float(processing["processor_total_ms"]),
        }
        nominal = server.max_new_tokens
        try:
            server.max_new_tokens = cap["effective_max_new_tokens"]
            result = server.recompute(
                enc_device, return_past_key_values=capture_cache)
        finally:
            server.max_new_tokens = nominal
        returned = time.perf_counter()
    result.update(helpers._timing_fields(result, request_started, phase, returned))
    result.update({"prompt": prompt, "enc_cpu": enc_cpu, "capture": capture,
                   **cap,
                   "model_forward_count": forwards.calls,
                   "vision_forward_count": capture.call_count,
                   "separate_vision_forward_count": 0,
                   "vision_ms": float(capture.stats()["vision_ms"])})
    return result


def _run_stored(runner, server, ctx, questions, answers, method_key: str,
                image_id: str) -> dict:
    helpers = _visdial()
    with helpers._NoVisionForward(runner) as guard, \
            _ModelForwardCounter(runner.model) as forwards:
        conditioning_started = time.perf_counter()
        ctx.reader.drop_all()
        conditioning_finished = time.perf_counter()
        torch.cuda.synchronize()
        request_started = time.perf_counter()
        prompt_t0 = time.perf_counter()
        prompt = render_prompt(questions, answers)
        prompt_build_ms = (time.perf_counter() - prompt_t0) * 1e3
        tokenize_t0 = time.perf_counter()
        tokenized = runner.processor.tokenizer(prompt, return_tensors="pt")
        tokenization_ms = (time.perf_counter() - tokenize_t0) * 1e3
        prepare_t0 = time.perf_counter()
        suffix_cpu = helpers._suffix_from_tokenized(runner, tokenized)
        cap = _effective_cap(
            runner, server,
            int(ctx.meta["prefix_len"]) + int(suffix_cpu.numel()))
        input_prepare_ms = (time.perf_counter() - prepare_t0) * 1e3
        h2d_t0 = time.perf_counter()
        suffix_device = suffix_cpu.to(runner.model.device)
        torch.cuda.synchronize()
        input_h2d_ms = (time.perf_counter() - h2d_t0) * 1e3
        phase = {"prompt_build_ms": prompt_build_ms,
                 "tokenization_ms": tokenization_ms,
                 "image_preprocess_ms": 0.0,
                 "input_prepare_ms": input_prepare_ms,
                 "input_h2d_ms": input_h2d_ms,
                 "processor_total_ms": None}
        nominal = server.max_new_tokens
        try:
            server.max_new_tokens = cap["effective_max_new_tokens"]
            if method_key == "fullload":
                result = server.request(ctx, mode="fullload", cold=False,
                                        suffix_ids=suffix_device)
            else:
                result = server.request_cvpr25(
                    ctx, static=None, budget=BUDGETS[method_key], mode="prefix",
                    sep_policy="sidecar", cold=False, seed=SEED,
                    image_id=image_id, suffix_ids=suffix_device,
                    expected_prefix_layout="visionzip_image_only")
        finally:
            server.max_new_tokens = nominal
        returned = time.perf_counter()
    result.update(helpers._timing_fields(result, request_started, phase, returned))
    result.update({"prompt": prompt, "suffix_cpu": suffix_cpu,
                   **cap,
                   "model_forward_count": forwards.calls,
                   "vision_forward_count": guard.calls,
                   "separate_vision_forward_count": 0, "vision_ms": 0.0,
                   "page_cache_conditioning_started_at_s": conditioning_started,
                   "page_cache_conditioning_finished_at_s": conditioning_finished,
                   "page_cache_conditioning_method": "posix_fadvise_DONTNEED",
                   "page_cache_conditioning_excluded_from_ttft": True})
    if not conditioning_finished < request_started:
        raise AssertionError("conditioning entered the request timer")
    return result


def _token_count(runner, text: str) -> int:
    return len(runner.processor.tokenizer(text,
                                         add_special_tokens=False).input_ids)


def _effective_cap(runner, server, input_tokens: int) -> dict:
    limit = int(runner.model.config.text_config.max_position_embeddings)
    if input_tokens < 1:
        raise ValueError("actual input token count must be positive")
    nominal = int(server.max_new_tokens)
    if nominal < 1:
        raise ValueError("nominal generation cap must be positive")
    # The first output token is selected from the last input position.  An
    # additional model forward happens only when a second token is requested.
    # Native LLaVA-NeXT/Vicuna inference was verified on the original
    # ConvBench ID494/418 generated-history prompts above the checkpoint's
    # nominal 4096 positions.  Keep those prompts intact, and let the nominal
    # output cap apply to overflow requests.  The on-model positional extent
    # is recorded per request; this does not alter RoPE or the context config.
    available = limit - input_tokens + 1
    native_overflow = input_tokens > limit
    effective = nominal if native_overflow else min(nominal, available)
    return {
        "context_policy_id": CONTEXT_POLICY_ID,
        "nominal_max_new_tokens": nominal,
        "effective_max_new_tokens": effective,
        "generation_cap_clamped_by_context": effective < nominal,
        "context_input_tokens": input_tokens,
        "context_remaining_positions_before_request": limit - input_tokens,
        "context_available_output_tokens": available,
        "input_overflow_tokens": max(0, input_tokens - limit),
        "native_overflow_execution": native_overflow,
    }


def _row(runner, dialog: Mapping[str, Any], turn: Mapping[str, Any],
         method_key: str, order: tuple[str, ...], result: dict,
         prior_answers: list[str], index_sha: str, store_manifest: dict | None,
         source_method: str, source_request: bool) -> dict:
    tid = int(turn["turn_id"])
    prompt = result.pop("prompt")
    answer = str(result["answer"])
    history_diagnostic = "\n".join(
        f"Q{i+1}: {dialog['turns'][i]['question']}\nA{i+1}: {a}"
        for i, a in enumerate(prior_answers))
    history_serialized = serialized_prior_history(
        prompt, turn["question"], tid)
    suffix = result.pop("suffix_cpu", None)
    enc_cpu = result.pop("enc_cpu", None)
    capture = result.pop("capture", None)
    result.pop("captured_past_key_values", None)
    io = _mt()._io_fields(result, method_key, store_manifest[
        "visual_kv_bytes"] if store_manifest else None)
    if tid == 1:
        io.update({"selected_visual_kv_bytes": 0,
                   "ssd_payload_ratio_vs_full_visual_kv": None,
                   "selected_kv_ratio": None,
                   "actual_selected_normal_chunk_fraction": None})
    prompt_ids = runner.processor.tokenizer(prompt, return_tensors="pt")[
        "input_ids"][0]
    context_input_tokens = int(result["context_input_tokens"])
    text_context_limit = int(
        runner.model.config.text_config.max_position_embeddings)
    generated_tokens = int(result["generated_tokens"])
    first_token_id = int(result["first_token_id"])
    vocab_size = int(runner.model.config.text_config.vocab_size)
    if not (generated_tokens >= 1 and 0 <= first_token_id < vocab_size):
        raise AssertionError("generation returned no valid first token")
    current_question_tail = (f"USER: {turn['question']} ASSISTANT:"
                             if tid > 1 else
                             f"{turn['question']} ASSISTANT:")
    if "<image>" not in prompt or not prompt.endswith(current_question_tail):
        raise AssertionError("image marker or current question missing")
    row = {
        "schema_version": SCHEMA_VERSION,
        "index_sha256": index_sha,
        "conversation_id": dialog["conversation_id"],
        "source_row_index": dialog.get("source_row_index"),
        "global_conversation_ordinal": dialog["global_conversation_ordinal"],
        "image_id": dialog["image_id"],
        "turn_id": tid,
        "category": turn.get("category"),
        "method_key": method_key,
        "method": {"recompute": "ReComp", "fullload": "FullLoad",
                   "prefix25": "Prefix25", "prefix45": "Prefix45"}[method_key],
        "budget": BUDGETS[method_key],
        "method_order": list(order),
        "method_order_position": order.index(method_key),
        "question": turn["question"],
        "reference_answer": turn["reference_answer"],
        "prediction": answer,
        "prompt": prompt,
        "prompt_sha256": _sha(prompt),
        "prompt_tokens": int(prompt_ids.numel()),
        # These counts include the processor-expanded image block for a
        # normal request, or the physically cached image-prefix length plus
        # the suffix for a stored-KV request.  No text is removed.
        "original_prompt_tokens": context_input_tokens,
        "final_prompt_tokens": context_input_tokens,
        "actual_input_tokens": context_input_tokens,
        "expanded_input_tokens": (int(enc_cpu["input_ids"].shape[1])
                                  if enc_cpu is not None else None),
        "context_tokens_for_cache_path": (
            int(store_manifest["prefix_len"]) + int(suffix.numel())
            if suffix is not None and store_manifest else None),
        "text_context_limit": text_context_limit,
        "nominal_context_limit": text_context_limit,
        "context_policy_id": CONTEXT_POLICY_ID,
        "input_overflow_tokens": int(result["input_overflow_tokens"]),
        "context_overflow_tokens": int(result["input_overflow_tokens"]),
        "native_overflow_execution": bool(result[
            "native_overflow_execution"]),
        "truncation_applied": False,
        "truncated_tokens": 0,
        "truncation_source": None,
        "preserved_current_question": True,
        "preserved_image_marker": True,
        "runtime_success": True,
        "first_token_success": True,
        "history_policy": "same_method_generated_answers",
        "history_serialized_text": history_serialized,
        "history_sha256": _sha(history_serialized),
        "history_tokens": _token_count(runner, history_serialized),
        "history_diagnostic_text": history_diagnostic,
        "history_diagnostic_tokens": _token_count(
            runner, history_diagnostic),
        "previous_answer_tokens": sum(_token_count(runner, a)
                                      for a in prior_answers),
        "previous_answers": list(prior_answers),
        "answer_tokens": _token_count(runner, answer),
        "suffix_tokens": int(suffix.numel()) if suffix is not None else None,
        "first_token_id": first_token_id,
        "generated_tokens": generated_tokens,
        "last_generated_sequence_position": (
            context_input_tokens + generated_tokens - 1),
        "last_expected_executed_position": (
            context_input_tokens + generated_tokens - 2),
        "generation_position_overflow_tokens": max(
            0, context_input_tokens + generated_tokens - text_context_limit),
        "nominal_max_new_tokens": int(result["nominal_max_new_tokens"]),
        "effective_max_new_tokens": int(result["effective_max_new_tokens"]),
        "generation_cap_clamped_by_context": bool(result[
            "generation_cap_clamped_by_context"]),
        "generation_cap_reached": (
            int(result["generated_tokens"]) == int(result[
                "effective_max_new_tokens"])),
        "context_input_tokens": context_input_tokens,
        "context_remaining_positions_before_request": int(result[
            "context_remaining_positions_before_request"]),
        "context_available_output_tokens": int(result[
            "context_available_output_tokens"]),
        "model_forward_count": int(result["model_forward_count"]),
        "max_new_tokens": int(result["max_new_tokens"]),
        "request_path": "normal_multimodal_pixel" if tid == 1 or
                        method_key == "recompute" else "ssd_visual_prefix",
        "persistence_source_request": bool(source_request),
        "same_physical_store_id": (store_manifest["meta_sha256"]
                                   if store_manifest else None),
        "physical_layout": ("visionzip_image_only" if store_manifest
                            else None),
        "permutation_sha256": (store_manifest["permutation_sha256"]
                               if store_manifest else None),
        "layout_questions_used": 0,
        "layout_answers_used": 0,
        "calibration_questions": 0,
        "future_turns_used_for_layout": 0,
        "vision_forward_count": int(result["vision_forward_count"]),
        "separate_vision_forward_count": 0,
        **io,
    }
    row["ssd_read_mb"] = row["ssd_read_bytes"] / 1_000_000
    row["pread_count"] = row["ssd_read_preads"]
    row["planning_ms"] = row["first_k_planning_ms"]
    for key in (
        "prompt_build_ms", "tokenization_ms", "image_preprocess_ms",
        "input_prepare_ms", "input_h2d_ms", "processor_total_ms",
        "pre_core_ms", "core_ttft_ms", "end_to_end_ttft_ms", "ttft_ms",
        "decode_ms", "model_e2e_ms", "request_e2e_ms", "e2e_ms",
        "request_started_at_s", "core_started_at_s", "first_token_at_s",
        "model_finished_at_s", "request_finished_at_s",
        "page_cache_conditioning_started_at_s",
        "page_cache_conditioning_finished_at_s",
        "page_cache_conditioning_method",
        "page_cache_conditioning_excluded_from_ttft", "vision_ms",
        "ttft_identity_error_ms", "model_e2e_identity_error_ms",
        "request_e2e_identity_error_ms",
    ):
        if key in result:
            row[key] = result[key]
    if suffix is not None:
        row["suffix_ids_sha256"] = _mt()._hash_tensor(suffix)
    if enc_cpu is not None:
        row["input_tensors_sha256"] = _mt()._hash_tensor_mapping(enc_cpu)
    if capture is not None:
        row["capture_stats"] = capture.stats()
    return row


def _persist(runner, result: dict, dialog: dict, store: Path,
             image_hash: str) -> tuple[dict, dict, Any]:
    capture = result["capture"]
    cache = result["captured_past_key_values"]
    enc = result["enc_cpu"]
    started = time.perf_counter()
    persisted = persist_captured_visual_prefix(
        runner, cache, enc["input_ids"], enc["image_sizes"][0],
        capture.result_cpu(), store,
        image_id=str(dialog["image_id"]), model_id=runner.model_id,
        chunk_size=CHUNK_SIZE, image_input_sha256=image_hash,
        capture_stats=capture,
        extra_metadata={"dataset": "ConvBench", "source_conversation_id":
                        dialog["conversation_id"], "source_turn_id": 1,
                        "history_policy": "same_method_generated_answers",
                        "future_turns_used_for_layout": 0,
                        "layout_questions_used": 0,
                        "layout_answers_used": 0})
    result.pop("captured_past_key_values")
    from mmimpress.serve import ImageContext
    ctx = ImageContext(store, runner.model.device, drop_cache=True,
                       require_v_hidden=False)
    ctx.validate_prefix_layout("visionzip_image_only")
    ready = time.perf_counter()
    manifest = _mt()._store_manifest(store, ctx.meta, persisted)
    timing = persisted["timing_ms"]
    persistence = {
        "conversation_id": dialog["conversation_id"],
        "source_turn_id": 1,
        "capture_from_same_answer1_forward": True,
        "vision_forward_count": capture.call_count,
        "separate_visual_prefix_forward_count": 0,
        "saliency_call_count": capture.saliency_call_count,
        "saliency_reduction_ms": capture.stats().get("saliency_reduction_ms"),
        "saliency_d2h_ms": capture.stats().get("saliency_materialize_ms"),
        "permutation_ms": timing.get("permutation_ms"),
        "kv_repack_ms": timing.get("kv_repack_ms"),
        "ssd_write_ms": timing.get("ssd_write_ms"),
        "fsync_ms": sum(float(timing.get(k, 0.0)) for k in
                        ("file_fsync_ms", "directory_fsync_ms",
                         "parent_fsync_ms")),
        "persist_ms": float((ready - started) * 1e3),
        "persist_started_at_s": started,
        "store_ready_at_s": ready,
        "total_ssd_write_bytes": int(persisted["bytes"]["total"]),
        "durable_fsync_completed": bool(persisted["durability"]
                                        ["parent_fsynced_after_rename"]),
    }
    return manifest, persistence, ctx


def _artifact_path(run_dir: Path, cid: str) -> Path:
    return run_dir / "conversations" / f"{cid}.json"


def _checkpoint_path(run_dir: Path, cid: str, request_index: int) -> Path:
    return run_dir / "request_checkpoints" / cid / f"{request_index:04d}.json"


def _load_request_checkpoints(run_dir: Path, dialog: dict, index_sha: str,
                              config_sha: str, image_sha: str,
                              order: tuple[str, ...]) -> dict[int, dict]:
    """Read committed requests in execution order; reject gaps and corruption.

    An unlinked ``*.tmp`` file is not a committed request.  It remains on disk
    for inspection, while the corresponding request may be executed again.
    """
    cid = _safe_id(dialog["conversation_id"])
    directory = run_dir / "request_checkpoints" / cid
    if not directory.exists():
        return {}
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"invalid request checkpoint directory: {directory}")
    expected_names = [f"{i:04d}.json" for i in range(12)]
    found = {p.name for p in directory.iterdir() if p.suffix == ".json"}
    if not found.issubset(set(expected_names)):
        raise ValueError(f"unexpected request checkpoint file: {directory}")
    committed: dict[int, dict] = {}
    for request_index, name in enumerate(expected_names):
        path = directory / name
        if not path.exists():
            if any((directory / later).exists()
                   for later in expected_names[request_index + 1:]):
                raise ValueError(f"request checkpoint gap: {path}")
            break
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"invalid request checkpoint: {path}")
        value = json.loads(path.read_text())
        claimed = value.get("artifact_content_sha256")
        body = {k: v for k, v in value.items()
                if k != "artifact_content_sha256"}
        if claimed != stable_json_sha256(body):
            raise ValueError(f"corrupt request checkpoint: {path}")
        expected_turn = request_index // 4 + 1
        expected_method = order[request_index % 4]
        if (value.get("schema_version") != REQUEST_CHECKPOINT_SCHEMA or
                value.get("conversation_id") != cid or
                value.get("request_index") != request_index or
                value.get("turn_id") != expected_turn or
                value.get("method_key") != expected_method or
                value.get("index_sha256") != index_sha or
                value.get("run_config_sha256") != config_sha or
                value.get("source_image_path") != dialog["image_path"] or
                value.get("source_image_sha256") != image_sha or
                value.get("method_order") != list(order)):
            raise ValueError(f"request checkpoint metadata mismatch: {path}")
        committed[request_index] = value
    return committed


def _validate_checkpoint_row(checkpoint: dict, dialog: dict, turn: dict,
                             method_key: str, prior_answers: list[str],
                             index_sha: str, source_request: bool,
                             store_manifest: dict | None) -> dict:
    row = checkpoint.get("row")
    if not isinstance(row, dict):
        raise ValueError("request checkpoint has no row")
    prompt = render_prompt([t["question"] for t in
                            dialog["turns"][:int(turn["turn_id"])]],
                           prior_answers)
    if (row.get("schema_version") != SCHEMA_VERSION or
            row.get("index_sha256") != index_sha or
            row.get("conversation_id") != dialog["conversation_id"] or
            row.get("turn_id") != turn["turn_id"] or
            row.get("method_key") != method_key or
            row.get("method_order") != checkpoint["method_order"] or
            row.get("question") != turn["question"] or
            row.get("reference_answer") != turn["reference_answer"] or
            row.get("previous_answers") != prior_answers or
            row.get("prompt") != prompt or
            row.get("prompt_sha256") != _sha(prompt) or
            row.get("history_policy") != "same_method_generated_answers" or
            row.get("persistence_source_request") is not source_request or
            row.get("runtime_success") is not True or
            row.get("first_token_success") is not True or
            row.get("truncation_applied") is not False or
            row.get("context_policy_id") != CONTEXT_POLICY_ID or
            not isinstance(row.get("prediction"), str) or
            not isinstance(row.get("first_token_id"), int) or
            not isinstance(row.get("generated_tokens"), int) or
            row["generated_tokens"] < 1):
        raise ValueError("request checkpoint row/history mismatch")
    if source_request:
        if (not isinstance(checkpoint.get("store_manifest"), dict) or
                not isinstance(checkpoint.get("persistence"), dict)):
            raise ValueError("source request checkpoint lacks durable store")
    elif checkpoint.get("store_manifest") is not None or checkpoint.get(
            "persistence") is not None:
        raise ValueError("non-source request checkpoint has store metadata")
    if int(turn["turn_id"]) > 1 and method_key != "recompute":
        if store_manifest is None or row.get("same_physical_store_id") != \
                store_manifest["meta_sha256"]:
            raise ValueError("cache request checkpoint store mismatch")
    return row


def _validate_committed_store(store: Path, manifest: dict) -> None:
    """Check a committed source request's ephemeral payload before reuse."""
    if store.is_symlink() or not store.is_dir():
        raise ValueError(f"committed source store missing/corrupt: {store}")
    sizes = manifest.get("file_sizes")
    if not isinstance(sizes, dict) or not sizes:
        raise ValueError("committed source store has no file manifest")
    files = {p.relative_to(store).as_posix(): p for p in store.rglob("*")
             if p.is_file()}
    if set(files) != set(sizes):
        raise ValueError(f"committed source store file set changed: {store}")
    for relative, expected_size in sizes.items():
        path = files[relative]
        if path.is_symlink() or path.stat().st_size != expected_size:
            raise ValueError(f"committed source store file changed: {path}")
    if (sha256_file(store / "meta.json") != manifest.get("meta_sha256") or
            sha256_file(store / "visionzip_layout.pt") != manifest.get(
                "layout_sha256") or
            _sampled_store_sha256(store, sizes) != manifest.get(
                "prefix_kv_sample_sha256")):
        raise ValueError(f"committed source store hash mismatch: {store}")


def _validate_artifact(path: Path, cid: str, index_sha: str,
                       max_new_tokens: int, source_image_path: Path) -> dict:
    value = json.loads(path.read_text())
    claimed = value.get("artifact_content_sha256")
    body = {k: v for k, v in value.items() if k != "artifact_content_sha256"}
    if claimed != stable_json_sha256(body):
        raise ValueError(f"corrupt conversation artifact: {path}")
    if (value.get("schema_version") != SCHEMA_VERSION or
            value.get("conversation_id") != cid or
            value.get("index_sha256") != index_sha or
            value.get("max_new_tokens") != max_new_tokens):
        raise ValueError(f"resume artifact metadata mismatch: {path}")
    if (value.get("image_path") != str(source_image_path) or
            value.get("image_file_sha256") != sha256_file(source_image_path)):
        raise ValueError(f"resume artifact source image SHA256 mismatch: {path}")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ValueError(f"incomplete conversation artifact: {path}")
    keys = {(r["turn_id"], r["method_key"]) for r in rows}
    if keys != {(t, m) for t in (1, 2, 3) for m in METHOD_KEYS}:
        raise ValueError(f"duplicate/missing request rows: {path}")
    return value


def _write_immutable(path: Path, value: dict) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value["artifact_content_sha256"] = stable_json_sha256(value)
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("x") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False,
                      allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        tmp.unlink(missing_ok=True)


def _run_conversation(runner, server, dialog: dict, index_sha: str,
                      temp_dir: Path, run_dir: Path,
                      config_sha: str) -> dict:
    from mmimpress.serve import ImageContext  # noqa: F401
    cid = _safe_id(dialog["conversation_id"])
    store = temp_dir / cid
    image_sha = sha256_file(Path(dialog["image_path"]))
    order = method_order(int(dialog["global_conversation_ordinal"]))
    source_method = [m for m in order if m.startswith("prefix")][-1]
    committed = _load_request_checkpoints(
        run_dir, dialog, index_sha, config_sha, image_sha, order)
    source_index = order.index(source_method)
    if source_index in committed:
        manifest = committed[source_index].get("store_manifest")
        persistence = committed[source_index].get("persistence")
        if not isinstance(manifest, dict) or not isinstance(persistence, dict):
            raise ValueError("committed source request lacks store metadata")
        _validate_committed_store(store, manifest)
    else:
        manifest = None
        persistence = None
        if store.is_symlink():
            raise ValueError(f"orphan temporary store is a symlink: {store}")
        if store.exists():
            # No committed source result depends on this run-private store.
            shutil.rmtree(store)
    with Image.open(dialog["image_path"]) as handle:
        image = handle.convert("RGB")
    history: dict[str, list[str]] = {m: [] for m in METHOD_KEYS}
    rows: list[dict] = []
    ctx = None
    try:
        if manifest is not None:
            from mmimpress.serve import ImageContext
            ctx = ImageContext(store, runner.model.device, drop_cache=True,
                               require_v_hidden=False)
            ctx.validate_prefix_layout("visionzip_image_only")
        for tid in (1, 2, 3):
            turn = dialog["turns"][tid - 1]
            questions = [t["question"] for t in dialog["turns"][:tid]]
            turn_rows = []
            for method_key in order:
                prior_answers = list(history[method_key])
                if len(prior_answers) != tid - 1:
                    raise AssertionError("method history has wrong length")
                source_request = tid == 1 and method_key == source_method
                request_index = (tid - 1) * 4 + order.index(method_key)
                new_checkpoint = None
                if request_index in committed:
                    row = _validate_checkpoint_row(
                        committed[request_index], dialog, turn, method_key,
                        prior_answers, index_sha, source_request, manifest)
                else:
                    torch.cuda.reset_peak_memory_stats()
                    if tid == 1 or method_key == "recompute":
                        result = _run_normal(
                            runner, server, image, questions, prior_answers,
                            capture_saliency=source_request,
                            capture_cache=source_request)
                        if source_request:
                            image_hash = _mt()._image_input_hash(result["enc_cpu"])
                            manifest, persistence, ctx = _persist(
                                runner, result, dialog, store, image_hash)
                    else:
                        if ctx is None:
                            raise AssertionError("cache request preceded persistence")
                        result = _run_stored(
                            runner, server, ctx, questions, prior_answers,
                            method_key, str(dialog["image_id"]))
                    result["max_new_tokens"] = server.max_new_tokens
                    row = _row(runner, dialog, turn, method_key, order,
                               result, prior_answers, index_sha, manifest,
                               source_method, source_request)
                    new_checkpoint = {
                        "schema_version": REQUEST_CHECKPOINT_SCHEMA,
                        "conversation_id": cid,
                        "request_index": request_index,
                        "turn_id": tid,
                        "method_key": method_key,
                        "method_order": list(order),
                        "index_sha256": index_sha,
                        "run_config_sha256": config_sha,
                        "source_image_path": dialog["image_path"],
                        "source_image_sha256": image_sha,
                        "row": row,
                        "store_manifest": manifest if source_request else None,
                        "persistence": persistence if source_request else None,
                    }
                if row["vision_forward_count"] != (0 if tid > 1 and
                                                     method_key != "recompute"
                                                     else 1):
                    raise AssertionError("vision-forward count mismatch")
                if row["request_path"] == "normal_multimodal_pixel" and row[
                        "ssd_read_bytes"] != 0:
                    raise AssertionError("normal request performed SSD read")
                if new_checkpoint is not None:
                    _validate_checkpoint_row(
                        new_checkpoint, dialog, turn, method_key,
                        prior_answers, index_sha, source_request, manifest)
                    _write_immutable(
                        _checkpoint_path(run_dir, cid, request_index),
                        new_checkpoint)
                    committed[request_index] = new_checkpoint
                history[method_key].append(str(row["prediction"]))
                turn_rows.append(row)
                gc.collect()
            if tid == 1:
                turn1_generation_agreement = {
                    "answer_text": len({r["prediction"] for r in turn_rows}) == 1,
                    "first_token_id": len({r["first_token_id"] for r in turn_rows}) == 1,
                    "generated_token_count": len({r["generated_tokens"] for r in turn_rows}) == 1,
                    "prompt": len({r["prompt_sha256"] for r in turn_rows}) == 1,
                    "input_tensors": len({r.get("input_tensors_sha256")
                                          for r in turn_rows}) == 1,
                }
                if not all(turn1_generation_agreement.values()):
                    print(f"{cid}: Turn-1 disagreement recorded: "
                          f"{turn1_generation_agreement}", flush=True)
            rows.extend(turn_rows)
        if manifest is None or persistence is None:
            raise AssertionError("no persisted physical store")
        # All three cache methods must use one physical layout.  Prefixes
        # themselves are checked by the MT-GQA shared validator.
        for tid in (2, 3):
            selected = {r["method_key"]: r for r in rows if r["turn_id"] == tid}
            _mt().validate_nested_prefixes(selected["prefix25"],
                                           selected["prefix45"])
            if any(selected[m]["same_physical_store_id"] !=
                   manifest["meta_sha256"] for m in
                   ("fullload", "prefix25", "prefix45")):
                raise AssertionError("cache methods used different stores")
        body = {
            "schema_version": SCHEMA_VERSION,
            "conversation_id": cid,
            "index_sha256": index_sha,
            "image_id": dialog["image_id"],
            "image_path": dialog["image_path"],
            "image_file_sha256": image_sha,
            "source_row_index": dialog.get("source_row_index"),
            "source_id": dialog.get("source_id"),
            "turn_categories": [t.get("category") for t in dialog["turns"]],
            "instruction_conditioned_caption": dialog.get(
                "instruction_conditioned_caption"),
            "third_turn_demands": dialog.get("third_turn_demands"),
            "global_conversation_ordinal": dialog[
                "global_conversation_ordinal"],
            "method_order": list(order),
            "source_method_key": source_method,
            "turn1_generation_agreement": turn1_generation_agreement,
            "max_new_tokens": server.max_new_tokens,
            "prompt_format": PROMPT_FORMAT,
            "physical_model_forward_count": sum(
                int(row["model_forward_count"]) for row in rows),
            "physical_request_inference_count": 12,
            "logical_request_count": 12,
            "store_build_count": 1,
            "store_manifest": manifest,
            "persistence": persistence,
            "rows": rows,
        }
        return body
    finally:
        if ctx is not None:
            ctx.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path,
                        default=ROOT / "data/convbench/index.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True,
                        help="Chosen from official/reference token lengths")
    parser.add_argument("--limit", type=int, default=10,
                        help="Number of conversations for a smoke run")
    parser.add_argument("--smoke-ids", type=str,
                        help="Comma-separated official source IDs (up to 10)")
    parser.add_argument("--full-run", action="store_true",
                        help="Explicitly permit all 577 conversations")
    parser.add_argument("--expected-index-sha256", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_new_tokens < 1 or args.max_new_tokens > 4096:
        raise ValueError("invalid generation cap")
    if args.max_new_tokens != 1024:
        raise ValueError("frozen official ConvBench LLaVA generation cap is 1024")
    if args.full_run:
        if args.smoke_ids is not None or args.limit != EXPECTED_CONVERSATIONS:
            raise ValueError("full run requires --limit 577 and no --smoke-ids")
    elif not 1 <= args.limit <= 10:
        raise ValueError("without --full-run, limit must be 1..10")
    dialogs, index_sha = load_index(args.index)
    if index_sha != args.expected_index_sha256:
        raise ValueError("source index SHA256 mismatch")
    selected, selection_mode = select_workload(
        dialogs, args.limit, args.full_run, args.smoke_ids)
    model_revision = _mt()._model_revision()
    if not model_revision:
        raise ValueError("local LLaVA-NeXT checkpoint revision unavailable")
    source_provenance = load_source_provenance()
    source_images_sha256, source_unique_images = source_image_fingerprint(dialogs)
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and (run_dir.is_symlink() or not run_dir.is_dir()):
        raise ValueError("run-dir must be a regular directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "index": str(args.index.resolve()),
        "index_sha256": index_sha,
        **source_provenance,
        "model_id": MODEL_ID,
        "model_revision": model_revision,
        "source_images_aggregate_sha256": source_images_sha256,
        "source_unique_images": source_unique_images,
        "load_4bit": True,
        "quantization": "NF4 double quantization",
        "compute_dtype": "bfloat16",
        "attention_implementation": "eager",
        "max_new_tokens": args.max_new_tokens,
        "context_cap_policy": (
            "For input_tokens<=4096: effective_max_new_tokens=min(1024, "
            "4096-input_tokens+1). For input_tokens>4096: "
            "effective_max_new_tokens=1024 with native positional "
            "extrapolation. Prompt/history never truncated; RoPE and model "
            "config unchanged."),
        "context_policy_id": CONTEXT_POLICY_ID,
        "text_context_limit": 4096,
        "do_sample": False,
        "seed": SEED,
        "limit": len(selected),
        "selection_mode": selection_mode,
        "selected_conversation_ids": [
            str(dialog["conversation_id"]) for dialog in selected],
        "selected_source_ids": [str(dialog["source_id"])
                                for dialog in selected],
        "full_run": args.full_run,
        "prompt_format": PROMPT_FORMAT,
        "method_keys": list(METHOD_KEYS),
        "os_page_cache_conditioning": "posix_fadvise_DONTNEED; buffered pread",
        "ssd_controller_cache_flushed": False,
        "ttft_definition": "request start before prompt construction to first selected token",
        "history_policy": "same_method_generated_answers",
        "benchmark_protocol_note": (
            "Official LLaVA llava_v1 turn roles and generated-history semantics; "
            "this run uses the project's LLaVA-NeXT 4-bit checkpoint and "
            "greedy decoding, rather than official LLaVA 1.5 sampling."),
    }
    config_path = run_dir / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        claimed = existing.pop("artifact_content_sha256", None)
        if claimed != stable_json_sha256(existing):
            raise ValueError("corrupt run config")
        if existing != config:
            raise ValueError("config mismatch; refusing resume")
    else:
        if any(run_dir.iterdir()):
            raise ValueError("nonempty run-dir without config; refusing overwrite")
        _write_immutable(config_path, config)
        # _write_immutable adds its own content hash; use the stored config for
        # exact resume comparison from this point onward.
        config = json.loads(config_path.read_text())
    config_sha = json.loads(config_path.read_text())["artifact_content_sha256"]
    temp_dir = run_dir / "_temporary_visual_kv"
    temp_dir.mkdir(exist_ok=True)
    marker = temp_dir / ".owner.json"
    owner = {"schema_version": SCHEMA_VERSION, "index_sha256": index_sha,
             "run_dir": str(run_dir)}
    if marker.exists():
        if json.loads(marker.read_text()) != owner:
            raise ValueError("temporary store owner mismatch")
    else:
        if any(temp_dir.iterdir()):
            raise ValueError("unowned temporary store is nonempty")
        marker.write_text(json.dumps(owner, indent=2) + "\n")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    runner = LlavaRunner().load()
    if int(runner.model.config.text_config.max_position_embeddings) != int(
            config["text_context_limit"]):
        raise ValueError("loaded model text context differs from run config")
    from mmimpress.serve import Server
    server = Server(runner, max_new_tokens=args.max_new_tokens)
    _visdial()._run_unmeasured_warmup(runner, server)
    for ordinal, dialog in enumerate(selected):
        cid = _safe_id(dialog["conversation_id"])
        path = _artifact_path(run_dir, cid)
        store = temp_dir / cid
        if path.exists():
            artifact = _validate_artifact(
                path, cid, index_sha, args.max_new_tokens,
                Path(dialog["image_path"]))
            order = method_order(int(dialog["global_conversation_ordinal"]))
            committed = _load_request_checkpoints(
                run_dir, dialog, index_sha, config_sha,
                artifact["image_file_sha256"], order)
            if committed:
                if len(committed) != 12:
                    raise ValueError(f"final artifact has incomplete checkpoints: {cid}")
                for request_index, checkpoint in committed.items():
                    if checkpoint["row"] != artifact["rows"][request_index]:
                        raise ValueError(
                            f"final artifact/checkpoint row mismatch: {cid} "
                            f"request {request_index}")
            if store.exists():
                shutil.rmtree(store)
            print(f"[{ordinal+1}/{len(selected)}] {cid}: resumed", flush=True)
            continue
        artifact = _run_conversation(runner, server, dialog, index_sha,
                                     temp_dir, run_dir, config_sha)
        _write_immutable(path, artifact)
        _validate_artifact(path, cid, index_sha, args.max_new_tokens,
                           Path(dialog["image_path"]))
        shutil.rmtree(store)
        print(f"[{ordinal+1}/{len(selected)}] {cid}: 12 requests complete",
              flush=True)


if __name__ == "__main__":
    main()

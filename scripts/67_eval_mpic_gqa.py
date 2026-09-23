#!/usr/bin/env python3
"""Same-run five-arm GQA pilot including MPIC-32 (SSD adaptation).

The frozen experiment is 40 images x 6 independent questions x 5 methods.
Turn 1 for every arm is an ordinary pixel request.  FullLoad, Ours25, and
MPIC-32 persist their own reusable state from that same answer-producing
forward; QA-Chunk25 shares FullLoad's canonical/raster store.  Turns 2--6 use
the corresponding SSD path, except ReComp which remains a pixel request.

Every request is appended and fsynced before progress advances.  Run-owned
stores are retained while an image is incomplete (so ``--resume`` is safe),
then removed only after all 30 records for that image are durable.  Existing
stores and result directories are never read as writable inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from mmimpress.mpic import (  # noqa: E402
    MPICContext, MPICServer, persist_captured_mpic_prefix,
)
from mmimpress.piggyback import (  # noqa: E402
    deterministic_method_rotation, persist_captured_raster_prefix,
    persist_captured_visual_prefix,
)
from mmimpress.serve import ImageContext, Server  # noqa: E402


SCHEMA_VERSION = "mpic-gqa-five-arm-v1"
EXPECTED_INDEX_SHA256 = (
    "514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a"
)
EXPECTED_WORKLOAD_SHA256 = (
    "97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66"
)
EXPECTED_PAPER_SHA256 = (
    "7253687b8a076fbea6e49fc8d9bffc856c3be33b1b7a372cba5fd5d00eaa503b"
)
MPIC_K = 32
N_IMAGES = 40
QUESTIONS_PER_IMAGE = 6
QUESTION_SKIP = 4
EXPECTED_QUESTIONS = N_IMAGES * QUESTIONS_PER_IMAGE
VALIDATED_SEED = 1234
MIN_FREE_BYTES_FOR_IMAGE_SESSION = 8 * 1024 ** 3
MAX_TECHNICAL_RETRIES = 1

RUNS_ROOT = ROOT / "runs/mpic_baseline"
RESULTS_ROOT = ROOT / "results/mpic_baseline"
DEFAULT_SMOKE_VALIDATION = (
    ROOT / "runs/mpic_baseline/smoke_final_linked_3/validation.json"
)
REQUIRED_SMOKE_CHECKS = {
    "all_real_logits_finite",
    "all_first_tokens_match_reference",
    "all_kN_logits_within_predeclared_tolerance",
    "all_kN_caches_within_predeclared_tolerance",
    "all_k0_existing_fullload_comparisons_passed",
    "all_cache_lengths_exact",
    "all_layer_counters_exact",
    "all_generations_valid",
    "all_independent_selective_call_audits_passed",
    "all_decode_cache_appends_exact",
    "all_live_payload_hashes_unchanged",
    "dummy_sentinel_passed",
    "shifted_position_mapping_passed",
}

METHOD_KEYS = ("recompute", "fullload", "qa_chunk25", "ours25", "mpic32")
METHODS: dict[str, dict[str, Any]] = {
    "recompute": {
        "method_id": "recompute", "display_label": "ReComp",
        "paper_label": "ReComp", "retention_ratio": None,
        "query_dependent": False, "selection_granularity": "none",
        "physical_layout": "none", "repacking": False,
        "online_selection": False,
    },
    "fullload": {
        "method_id": "fullload", "display_label": "FullLoad",
        "paper_label": "FullLoad", "retention_ratio": 1.0,
        "query_dependent": False,
        "selection_granularity": "full_visual_kv",
        "physical_layout": "canonical_raster", "repacking": False,
        "online_selection": False,
    },
    "qa_chunk25": {
        "method_id": "qa_chunk25", "display_label": "QA-Chunk25",
        "paper_label": "Query-Aware Chunk", "retention_ratio": 0.25,
        "importance_source": "SparseVLM-style text-guided visual importance",
        "query_dependent": True, "selection_granularity": "ssd_chunk",
        "chunk_score": "mean_valid_spatial_token_importance",
        "physical_layout": "canonical_raster", "repacking": False,
        "online_selection": True,
    },
    "ours25": {
        "method_id": "imageonly_prefix25", "display_label": "Ours25",
        "paper_label": "Ours", "retention_ratio": 0.25,
        "importance_source": "image-only Vision Encoder saliency",
        "query_dependent": False,
        "selection_granularity": "ssd_chunk_prefix",
        "physical_layout": "importance-aware repacked", "repacking": True,
        "online_selection": False,
    },
    "mpic32": {
        "method_id": "mpic32_ssd",
        "display_label": "MPIC-32 (SSD adaptation)",
        "paper_label": "MPIC-32 (SSD adaptation)",
        "retention_ratio": 1.0,
        "importance_source": "first_k_canonical_image_rows",
        "query_dependent": False,
        "selection_granularity": "first_32_image_tokens_recomputed",
        "physical_layout": "canonical_raster_plus_visual_input_sidecar",
        "repacking": False, "online_selection": False,
        "k_recompute": MPIC_K,
    },
}
STORE_OWNER = {
    "fullload": "raster", "ours25": "image_only", "mpic32": "mpic",
}


_BASE49 = None
_BASE52 = None
_PROTECT = None


class PartialRunStop(RuntimeError):
    """Internal control flow for a deliberate durable checkpoint."""


def _load_script(filename: str, module_name: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _base49():
    global _BASE49
    if _BASE49 is None:
        _BASE49 = _load_script(
            "49_eval_query_aware_baseline.py", "_mpic_validated_gqa_base")
    return _BASE49


def _base52():
    global _BASE52
    if _BASE52 is None:
        _BASE52 = _load_script(
            "52_eval_query_aware_chunk_baseline.py",
            "_mpic_validated_chunk_base")
    return _BASE52


def _protector():
    global _PROTECT
    if _PROTECT is None:
        _PROTECT = _load_script(
            "66_protect_mpic_artifacts.py", "_mpic_artifact_protector")
    return _PROTECT


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_fingerprint(runner) -> dict[str, Any]:
    """Serving identity frozen before any resumable store is consumed."""
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    tokenizer = runner.processor.tokenizer
    return {
        "model_id": str(runner.model_id),
        "model_revision": getattr(runner.model.config, "_commit_hash", None),
        "model_config_sha256": canonical_hash(runner.model.config.to_dict()),
        "model_class": (f"{type(runner.model).__module__}."
                        f"{type(runner.model).__qualname__}"),
        "processor_class": (f"{type(runner.processor).__module__}."
                            f"{type(runner.processor).__qualname__}"),
        "tokenizer_class": (f"{type(tokenizer).__module__}."
                            f"{type(tokenizer).__qualname__}"),
        "tokenizer_vocab_size": len(tokenizer),
        "load_4bit": bool(LOAD_4BIT),
        "compute_dtype": COMPUTE_DTYPE,
        "attention_implementation": ATTN_IMPL,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": _package_version("transformers"),
        "bitsandbytes": _package_version("bitsandbytes"),
        "accelerate": _package_version("accelerate"),
        "numpy": str(np.__version__),
        "pillow": _package_version("Pillow"),
        "torch_cuda_build": torch.version.cuda,
        "cudnn": (None if not torch.backends.cudnn.is_available()
                  else int(torch.backends.cudnn.version())),
        "cuda_device_index": int(device_index),
        "cuda_device_name": str(properties.name),
        "cuda_compute_capability": [
            int(properties.major), int(properties.minor)],
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _is_descendant(path: Path, parent: Path) -> bool:
    path, parent = path.resolve(), parent.resolve()
    return path != parent and parent in path.parents


def _source_hashes() -> dict[str, str]:
    paths = {
        "paper": ROOT / "papers/mpic.md",
        "mpic_implementation": ROOT / "mmimpress/mpic.py",
        "mpic_contract": ROOT / "docs/mpic_baseline_contract.md",
        "mpic_smoke_validator": ROOT / "scripts/65_validate_mpic.py",
        "artifact_protector": ROOT / "scripts/66_protect_mpic_artifacts.py",
        "pilot_runner": Path(__file__).resolve(),
        "pilot_reporter": ROOT / "scripts/68_report_mpic_gqa.py",
        "pixel_baseline_helper": ROOT / "scripts/49_eval_query_aware_baseline.py",
        "chunk_baseline_helper": ROOT / "scripts/52_eval_query_aware_chunk_baseline.py",
        "runtime_config": ROOT / "mmimpress/config.py",
        "dataset_adapter": ROOT / "mmimpress/dataset.py",
        "model_adapter": ROOT / "mmimpress/model.py",
        "ssd_store": ROOT / "mmimpress/store.py",
        "legacy_server": ROOT / "mmimpress/serve.py",
        "piggyback_capture": ROOT / "mmimpress/piggyback.py",
        "image_selector": ROOT / "mmimpress/cvpr25.py",
        "kv_reorder": ROOT / "mmimpress/reorder.py",
        "query_selector": ROOT / "mmimpress/sparsevlm.py",
        "mpic_unit_tests": ROOT / "tests/test_mpic.py",
        "mpic_report_tests": ROOT / "tests/test_mpic_report.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _validate_smoke(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    value = _read_json(path)
    if value.get("passed") is not True:
        raise ValueError(f"MPIC real-model smoke did not pass: {path}")
    checks = value.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("MPIC smoke has no checks mapping")
    missing = REQUIRED_SMOKE_CHECKS - set(checks)
    failed = [str(key) for key, passed in checks.items() if passed is not True]
    if missing or failed:
        raise ValueError(
            "MPIC smoke is not the hardened contract suite: "
            f"missing={sorted(missing)}, failed={sorted(failed)}")
    if int(value.get("images_completed", -1)) < 3:
        raise ValueError("MPIC smoke must cover at least three real images")
    return {
        "path": str(path), "sha256": sha256_file(path),
        "schema_version": value.get("schema_version"),
        "passed": True, "checks": checks,
    }


def request_id(image_id: str, question_id: str, method_key: str) -> str:
    if method_key not in METHOD_KEYS:
        raise ValueError(f"unknown method: {method_key}")
    return f"gqa:{image_id}:{question_id}:{method_key}"


def expected_schedule(entries: Sequence[Mapping[str, Any]], seed: int) \
        -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    for image_index, entry in enumerate(entries):
        image_id = str(entry["image_id"])
        selected = entry["questions"][
            QUESTION_SKIP:QUESTION_SKIP + QUESTIONS_PER_IMAGE]
        if len(selected) != QUESTIONS_PER_IMAGE:
            raise ValueError(f"image {image_id} lacks the frozen question slice")
        order = deterministic_method_rotation(METHOD_KEYS, image_index, seed)
        for turn_id, question in enumerate(selected, 1):
            for position, method_key in enumerate(order):
                qid = str(question["question_id"])
                schedule.append({
                    "request_id": request_id(image_id, qid, method_key),
                    "image_index": image_index, "image_id": image_id,
                    "question_id": qid, "turn_id": turn_id,
                    "method_key": method_key, "method_order": list(order),
                    "method_order_position": position,
                })
    return schedule


def _validate_workload(index_path: Path):
    all_entries, entries, workload = _base49()._workload(
        index_path, QUESTION_SKIP, QUESTIONS_PER_IMAGE, N_IMAGES)
    if len(all_entries) != N_IMAGES or len(entries) != N_IMAGES:
        raise ValueError("frozen MPIC pilot requires exactly 40 images")
    if workload["index_sha256"] != EXPECTED_INDEX_SHA256:
        raise ValueError("frozen index SHA256 mismatch")
    if workload["full_workload_sha256"] != EXPECTED_WORKLOAD_SHA256:
        raise ValueError("frozen GQA workload SHA256 mismatch")
    if workload["selected_questions"] != EXPECTED_QUESTIONS:
        raise ValueError("frozen GQA workload must contain 240 questions")
    return entries, workload


def _prepare_new(args: argparse.Namespace):
    run_dir = args.run_dir.resolve()
    results_dir = args.results_dir.resolve()
    if not _is_descendant(run_dir, RUNS_ROOT):
        raise ValueError(f"run directory must be below {RUNS_ROOT}")
    if not _is_descendant(results_dir, RESULTS_ROOT):
        raise ValueError(f"results directory must be below {RESULTS_ROOT}")
    if results_dir.name != f"gqa40_240_{run_dir.name}":
        raise ValueError(
            "results directory must be named gqa40_240_<run_id> and match "
            "the run directory name")
    # The protection step intentionally creates the run root and publishes
    # exactly one before-manifest there.  Requiring that hand-off makes it
    # impossible to launch the expensive pilot without freezing old outputs.
    protection_name = _protector().MANIFEST_NAME
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise FileNotFoundError(
            "run directory must first be prepared by "
            "scripts/66_protect_mpic_artifacts.py --before")
    existing = sorted(path.name for path in run_dir.iterdir())
    if existing != [protection_name]:
        raise FileExistsError(
            "protector-prepared run directory must contain only "
            f"{protection_name}; found {existing}")
    if os.path.lexists(results_dir):
        raise FileExistsError("results directory must be new")
    before = _read_json(run_dir / protection_name)
    _protector().validate_manifest(before, run_dir, results_dir, ROOT.resolve())
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir()
    options = {
        "index": str(args.index.resolve()),
        "run_dir": str(run_dir), "results_dir": str(results_dir),
        "smoke_validation": str(args.smoke_validation.resolve()),
        "seed": int(args.seed),
        "max_new_tokens": int(args.max_new_tokens),
        "mpic_k": MPIC_K,
    }
    return run_dir, results_dir, options, False


def _prepare_resume(args: argparse.Namespace):
    run_dir = args.resume.resolve()
    if not _is_descendant(run_dir, RUNS_ROOT):
        raise ValueError(f"resume directory must be below {RUNS_ROOT}")
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError(f"invalid resume directory: {run_dir}")
    if (run_dir / "COMPLETED").exists():
        raise RuntimeError(f"run is already complete: {run_dir}")
    manifest = _read_json(run_dir / "manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("resume schema mismatch")
    options = manifest.get("options")
    if not isinstance(options, dict):
        raise ValueError("resume manifest lacks immutable options")
    results_dir = Path(options["results_dir"]).resolve()
    if not _is_descendant(results_dir, RESULTS_ROOT):
        raise ValueError("resume results path escaped the MPIC results root")
    if results_dir.is_symlink() or not results_dir.is_dir():
        raise ValueError(f"invalid resume results directory: {results_dir}")
    return run_dir, results_dir, dict(options), True


def _atomic_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    _base52().atomic_json(path, value, exclusive=exclusive)


def _atomic_bytes(path: Path, value: bytes, *, exclusive: bool = False) -> None:
    _base52().atomic_bytes(path, value, exclusive=exclusive)


def _append_row(path: Path, value: Mapping[str, Any]) -> None:
    _base52().append_jsonl_durable(path, value)


def _progress(run_id: str, expected: int, completed: set[str],
              failures: Sequence[Mapping[str, Any]], status: str,
              last_request: str | None, repaired_tail: int) -> dict[str, Any]:
    unresolved = {str(row["request_id"]) for row in failures} - completed
    return {
        "schema_version": SCHEMA_VERSION, "run_id": run_id,
        "status": status, "expected": expected, "completed": len(completed),
        "remaining": expected - len(completed),
        "failure_events": len(failures), "unresolved_failures": len(unresolved),
        "duplicates": 0, "last_request_id": last_request,
        "repaired_truncated_tail_bytes": repaired_tail,
        "updated_at_unix": time.time(), "pid": os.getpid(),
    }


def _prior_failure_count(failures: Sequence[Mapping[str, Any]],
                         identity: str) -> int:
    return sum(str(row.get("request_id")) == identity for row in failures)


def _require_retry_budget(failures: Sequence[Mapping[str, Any]],
                          identity: str) -> int:
    count = _prior_failure_count(failures, identity)
    if count > MAX_TECHNICAL_RETRIES:
        raise RuntimeError(
            f"technical retry budget exhausted for {identity}: {count} "
            f"> {MAX_TECHNICAL_RETRIES}")
    return count


def _persistence_call(method_key: str, runner, pixel_result,
                      diagnostic: Mapping[str, Any], destination: Path,
                      image_id: str, question_id: str, run_id: str,
                      identity: str) -> dict[str, Any]:
    cache = pixel_result.get("captured_past_key_values")
    if cache is None:
        raise AssertionError(f"{method_key} Turn 1 did not capture its cache")
    encoded = diagnostic["enc_cpu"]
    common = {
        "image_id": image_id, "model_id": runner.model_id,
        "chunk_size": CHUNK_SIZE,
        "image_input_sha256": diagnostic["image_input_sha256"],
        "extra_metadata": {
            "pilot_schema_version": SCHEMA_VERSION,
            "pilot_run_id": run_id, "pilot_method_key": method_key,
            "turn1_question_id": question_id,
            "turn1_request_id": identity,
            "store_lifecycle": "run_local_ephemeral_after_image_commit",
        },
    }
    if method_key == "fullload":
        hidden = diagnostic["hidden_capture"]
        if hidden is None:
            raise AssertionError("raster persistence lacks hidden capture")
        return persist_captured_raster_prefix(
            runner, cache, encoded["input_ids"], encoded["image_sizes"][0],
            hidden.result_cpu(), destination,
            probe_heads=PROBE_HEADS, hidden_capture_stats=hidden,
            full_integrity_hash=False, **common)
    if method_key == "ours25":
        vision = diagnostic["vision_capture"]
        return persist_captured_visual_prefix(
            runner, cache, encoded["input_ids"], encoded["image_sizes"][0],
            vision.result_cpu(), destination, capture_stats=vision,
            full_integrity_hash=False, **common)
    if method_key == "mpic32":
        hidden = diagnostic["hidden_capture"]
        if hidden is None:
            raise AssertionError("MPIC persistence lacks hidden capture")
        return persist_captured_mpic_prefix(
            runner, cache, encoded["input_ids"], encoded["image_sizes"][0],
            hidden.result_cpu(), destination, hidden_capture_stats=hidden,
            **common)
    raise ValueError(f"method does not own a store: {method_key}")


def _normalize_pixel(result: Mapping[str, Any], method_key: str) \
        -> dict[str, Any]:
    value = _base52()._json_result(dict(result), method_key, 0)
    value.setdefault("retry_count", 0)
    value.setdefault("status", "ok")
    return value


def _run_mpic(runner, server: MPICServer, context: MPICContext,
              question: str, full_visual_bytes: int):
    base = _base49()
    with base._NoVisionForward(runner) as guard:
        value = server.request(context, question=question, cold=True)
    if guard.calls != 0 or int(value.get("vision_forward_count", -1)) != 0:
        raise AssertionError("MPIC cache hit invoked the vision tower")
    prompt = runner.prompt(question)
    tokenized = runner.processor.tokenizer(prompt, return_tensors="pt")
    suffix = base._suffix_from_tokenized(runner, tokenized)
    diagnostic = {
        "prompt": prompt,
        "prompt_sha256": base._sha_bytes(prompt.encode("utf-8")),
        "suffix_ids_sha256": base._hash_tensor(suffix),
    }
    normalized = dict(value)
    # MPICServer's response-ready boundary includes token decoding and result
    # construction while deliberately excluding page-cache conditioning and
    # the out-of-band post-response integrity audit.  This is the comparable
    # request E2E metric; request_return_wall_ms is retained only as a separate
    # diagnostic showing synchronous audit overhead.
    normalized["request_e2e_ms"] = float(value["request_e2e_ms"])
    normalized["answer"] = str(value["prediction"])
    normalized["ssd_read_ms"] = float(
        value.get("kv_read_ms", 0.0) + value.get("embedding_read_ms", 0.0))
    normalized["ssd_read_bytes"] = int(value["ssd_total_bytes"])
    normalized["actual_ssd_mb"] = float(value["ssd_total_bytes"] / 1e6)
    normalized["ssd_preads"] = int(value["pread_count"])
    normalized["ssd_read_chunk_units"] = int(
        value.get("io", {}).get("chunk_units", 0))
    normalized["normal_kv_read_bytes"] = int(value["ssd_kv_bytes"])
    normalized["separator_read_bytes"] = int(value["ssd_separator_bytes"])
    normalized["probe_read_bytes"] = 0
    normalized["normal_kv_preads"] = sum(
        int(value.get("io", {}).get("per_kind", {}).get(kind, {}).get(
            "preads", 0)) for kind in ("kv_k", "kv_v"))
    normalized["separator_preads"] = 0
    normalized["probe_preads"] = 0
    normalized["actual_ssd_ratio_vs_fullload"] = (
        float(value["ssd_total_bytes"]) / full_visual_bytes)
    normalized["io_detail"] = value.get("io", {}).get("per_kind", {})
    normalized["prompt_build_ms"] = None
    normalized["tokenization_ms"] = None
    normalized["processor_total_ms"] = float(
        value["prompt_and_tokenization_ms"])
    normalized["image_preprocess_ms"] = 0.0
    normalized["input_prepare_ms"] = None
    normalized["input_h2d_ms"] = float(value["h2d_ms"])
    normalized["prefill_ms"] = float(value["selective_prefill_interval_ms"])
    normalized["end_to_end_ttft_ms"] = float(value["ttft_ms"])
    normalized.setdefault("online_selector_total_ms", 0.0)
    normalized.setdefault("chunk_io_ms", None)
    normalized.setdefault("scatter_ms", 0.0)
    normalized.setdefault("retry_count", 0)
    normalized.setdefault("status", "ok")
    return normalized, diagnostic


def _record(*, config: Mapping[str, Any], runner, question: Mapping[str, Any],
            image_id: str, turn_id: int, method_key: str,
            method_order: Sequence[str], position: int,
            result: Mapping[str, Any], diagnostic: Mapping[str, Any],
            request_path: str, n_image_tokens: int,
            store_capture: str | None = None) -> dict[str, Any]:
    qid = str(question["question_id"])
    identity = request_id(image_id, qid, method_key)
    prediction = str(result.get("answer", result.get("prediction", "")))
    gold = question_answers(question)
    cache_hit = turn_id > 1 and method_key != "recompute"
    value = {
        "schema_version": SCHEMA_VERSION, "run_id": config["run_id"],
        "request_id": identity, "measurement_source": "same_run",
        "dataset": "gqa", "image_id": image_id,
        "question_id": qid, "request_ordinal": turn_id, "turn_id": turn_id,
        "question": question["question"], "gold": gold,
        "method_key": method_key, **METHODS[method_key],
        "method_order": list(method_order), "method_order_position": position,
        "request_path": request_path, "cache_hit_measurement": cache_hit,
        "turn1_store_capture": store_capture,
        "prediction": prediction,
        "correct": METRICS["gqa"](prediction, gold),
        "first_token_id": int(result["first_token_id"]),
        "prompt_sha256": diagnostic["prompt_sha256"],
        "suffix_ids_sha256": diagnostic["suffix_ids_sha256"],
        "input_tensors_sha256": diagnostic.get("input_tensors_sha256"),
        "image_input_sha256": diagnostic.get("image_input_sha256"),
        "n_image_tokens": int(n_image_tokens),
        "retry_count": int(result.get("retry_count", 0)),
        "status": str(result.get("status", "ok")),
        **_base49()._causal_prompt_fields(
            runner, diagnostic, question["question"], qid),
        **dict(result),
    }
    # The normalized cross-arm fields below describe the measured request,
    # while method metadata above describes the cache-hit algorithm.
    if not cache_hit:
        value.update({
            "n_recomputed_image_tokens": int(n_image_tokens),
            "n_reused_image_tokens": 0,
            "retained_image_context_ratio": 1.0,
        })
    elif method_key == "mpic32":
        # MPICServer supplies the precise values; assert rather than replace.
        if int(value["n_recomputed_image_tokens"]) != min(
                MPIC_K, n_image_tokens):
            raise AssertionError("MPIC recomputed-image count is not k=32")
    else:
        value.update({
            "n_recomputed_image_tokens": 0,
            "n_reused_image_tokens": (
                int(n_image_tokens) if method_key == "fullload" else None),
            "retained_image_context_ratio": float(
                METHODS[method_key]["retention_ratio"]),
            "image_token_count_semantics": (
                "all cached image rows reused" if method_key == "fullload"
                else "N/A here; selected retained rows/chunks are reported by "
                "the legacy method-specific selector instrumentation"),
        })
    value.update(METHODS[method_key])
    return value


def _store_paths(run_dir: Path, image_id: str) -> dict[str, Path]:
    root = run_dir / "stores"
    return {name: root / name / image_id
            for name in ("raster", "image_only", "mpic")}


def _provision_path(run_dir: Path, image_id: str, method_key: str) -> Path:
    return run_dir / "checkpoints" / image_id / f"{method_key}.json"


def _pending_path(run_dir: Path, image_id: str, method_key: str) -> Path:
    return run_dir / "checkpoints" / image_id / f"{method_key}.pending.json"


def _recover_provision(path: Path, identity: str) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"provision schema mismatch: {path}")
    record = value.get("record")
    if not isinstance(record, dict) or record.get("request_id") != identity:
        raise ValueError(f"provision identity mismatch: {path}")
    if not isinstance(value.get("persistence"), dict):
        raise ValueError(f"provision lacks persistence evidence: {path}")
    return value


def _recover_pending(path: Path, identity: str) -> dict[str, Any]:
    value = _read_json(path)
    record = value.get("record")
    if (value.get("schema_version") != SCHEMA_VERSION
            or value.get("request_id") != identity
            or not isinstance(record, dict)
            or record.get("request_id") != identity):
        raise ValueError(f"invalid pending Turn-1 transaction: {path}")
    return value


def _recover_published_store(method_key: str, path: Path, runner) \
        -> dict[str, Any]:
    """Validate an orphan publication without pretending timing survived."""
    if method_key == "fullload":
        context = ImageContext(path, runner.model.device, require_v_hidden=True)
        try:
            context.validate_qa_select_layout()
            meta = dict(context.meta)
        finally:
            context.close()
    elif method_key == "ours25":
        context = ImageContext(path, runner.model.device,
                               require_v_hidden=False)
        try:
            context.validate_prefix_layout("visionzip_image_only")
            meta = dict(context.meta)
        finally:
            context.close()
    elif method_key == "mpic32":
        context = MPICContext(path, runner.model.device, runner=runner)
        try:
            meta = dict(context.meta)
        finally:
            context.close()
    else:
        raise ValueError(method_key)
    sizes = {item.relative_to(path).as_posix(): int(item.stat().st_size)
             for item in sorted(path.rglob("*")) if item.is_file()}
    return {
        "store_dir": str(path.resolve()), "meta": meta,
        "file_sizes": sizes,
        "bytes": {"total": int(sum(sizes.values())),
                  "visual_kv": int(meta["bytes_visual_kv"])},
        "timing_ms": None,
        "measurement_complete": False,
        "recovery": {
            "kind": "validated_atomic_store_after_crash_before_provision",
            "store_payload_validated": True,
            "persistence_timing_unrecoverable": True,
        },
    }


def _open_contexts(paths: Mapping[str, Path], runner):
    raster = ImageContext(paths["raster"], runner.model.device,
                          require_v_hidden=True)
    ours = None
    mpic = None
    try:
        raster.validate_qa_select_layout()
        ours = ImageContext(paths["image_only"], runner.model.device,
                            require_v_hidden=False)
        ours.validate_prefix_layout("visionzip_image_only")
        mpic = MPICContext(
            paths["mpic"], runner.model.device, runner=runner)
        byte_counts = {
            int(raster.meta["bytes_visual_kv"]),
            int(ours.meta["bytes_visual_kv"]),
            int(mpic.meta["bytes_visual_kv"]),
        }
        token_counts = {
            int(raster.meta["v_token_num"]), int(ours.meta["v_token_num"]),
            int(mpic.meta["v_token_num"]),
        }
        if len(byte_counts) != 1 or len(token_counts) != 1:
            raise AssertionError("same-image stores disagree on KV geometry")
        return raster, ours, mpic
    except BaseException:
        raster.close()
        if ours is not None:
            ours.close()
        if mpic is not None:
            mpic.close()
        raise


def _remove_completed_image_stores(run_dir: Path, image_id: str,
                                   paths: Mapping[str, Path]) -> None:
    store_root = (run_dir / "stores").resolve()
    removed: list[str] = []
    for name, path in paths.items():
        resolved = path.resolve()
        if store_root not in resolved.parents:
            raise ValueError(f"refusing cleanup outside run store: {resolved}")
        if os.path.lexists(resolved):
            if resolved.is_symlink() or not resolved.is_dir():
                raise ValueError(f"unexpected run-store object: {resolved}")
            shutil.rmtree(resolved)
            removed.append(name)
    marker = run_dir / "cleanup" / f"{image_id}.json"
    payload = {
        "schema_version": SCHEMA_VERSION, "image_id": image_id,
        "policy": "deleted_only_run_local_reproducible_cache_after_30_rows",
        "removed_layouts": removed, "cleaned_at_unix": time.time(),
    }
    if marker.exists():
        previous = _read_json(marker)
        if previous.get("image_id") != image_id:
            raise ValueError(f"cleanup marker mismatch: {marker}")
    else:
        _atomic_json(marker, payload, exclusive=True)


def _image_metadata(run_dir: Path, image_id: str, raster, ours, mpic,
                    provisions: Mapping[str, Mapping[str, Any]]) -> None:
    meta_path = run_dir / "store_metadata" / f"{image_id}.json"
    persistence_path = run_dir / "persistence" / f"{image_id}.json"
    meta = {
        "schema_version": SCHEMA_VERSION, "image_id": image_id,
        "raster": dict(raster.meta), "image_only": dict(ours.meta),
        "mpic": dict(mpic.meta),
    }
    persistence = {
        "schema_version": SCHEMA_VERSION, "image_id": image_id,
        "store_owner": dict(STORE_OWNER),
        "stores": {method: value["persistence"]
                   for method, value in provisions.items()},
    }
    for path, payload in ((meta_path, meta),
                          (persistence_path, persistence)):
        if path.exists():
            if canonical_hash(_read_json(path)) != canonical_hash(payload):
                raise ValueError(f"resume metadata differs: {path}")
        else:
            _atomic_json(path, payload, exclusive=True)


def _write_mpic_image_integrity(run_dir: Path, image_id: str,
                                context: MPICContext) -> None:
    live = context.source_payload_hash
    validated = context.validated_payload_hash
    payload = {
        "schema_version": SCHEMA_VERSION, "image_id": image_id,
        "boundary": (
            "after_all_5_mpic_hits_and_25_turn2_to_turn6_method_requests_"
            "before_context_close"),
        "validated_at_context_open_sha256": validated,
        "live_after_all_hits_sha256": live,
        "unchanged": live == validated,
        "timing_semantics": (
            "one out-of-band integrity sample after all measured requests "
            "for this image; "
            "not included in TTFT or request E2E and cannot warm a subsequent "
            "request for this image"),
        "checked_at_unix": time.time(),
    }
    if not payload["unchanged"]:
        raise ValueError(f"MPIC payload changed while serving image {image_id}")
    path = run_dir / "mpic_image_integrity" / f"{image_id}.json"
    if path.exists():
        previous = _read_json(path)
        comparable = {key: value for key, value in previous.items()
                      if key != "checked_at_unix"}
        current = {key: value for key, value in payload.items()
                   if key != "checked_at_unix"}
        if comparable != current:
            raise ValueError(f"MPIC integrity evidence differs: {path}")
    else:
        _atomic_json(path, payload, exclusive=True)


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in METHOD_KEYS:
        all_rows = [row for row in rows if row["method_key"] == method]
        hits = [row for row in all_rows if int(row["turn_id"]) > 1]
        ttft = np.asarray([float(row["ttft_ms"]) for row in hits],
                          dtype=np.float64)
        result[method] = {
            **METHODS[method], "requests_all": len(all_rows),
            "requests_cache_hit": len(hits),
            "accuracy_all": (float(np.mean([float(row["correct"])
                                             for row in all_rows]))
                             if all_rows else None),
            "accuracy_cache_hit": (float(np.mean([float(row["correct"])
                                                   for row in hits]))
                                   if hits else None),
            "ttft_cache_hit_mean_ms": float(ttft.mean()) if len(ttft) else None,
            "ttft_cache_hit_p50_ms": float(np.percentile(ttft, 50))
            if len(ttft) else None,
            "ttft_cache_hit_p95_ms": float(np.percentile(ttft, 95))
            if len(ttft) else None,
            "ssd_mb_cache_hit_mean": (float(np.mean([
                float(row.get("ssd_read_bytes", 0)) / 1e6 for row in hits]))
                if hits else None),
            "ssd_preads_cache_hit_mean": (float(np.mean([
                float(row.get("ssd_preads", row.get("pread_count", 0)))
                for row in hits])) if hits else None),
            "recomputed_image_tokens_cache_hit_mean": (float(np.mean([
                float(row["n_recomputed_image_tokens"]) for row in hits]))
                if hits else None),
        }
    return result


def _validate(rows: Sequence[Mapping[str, Any]], schedule,
              failures: Sequence[Mapping[str, Any]], run_dir: Path) \
        -> dict[str, Any]:
    expected_ids = [str(row["request_id"]) for row in schedule]
    observed_ids = [str(row["request_id"]) for row in rows]
    completed = set(observed_ids)
    unresolved = [row for row in failures
                  if str(row["request_id"]) not in completed]
    method_counts = Counter(str(row["method_key"]) for row in rows)
    hit_counts = Counter(str(row["method_key"]) for row in rows
                         if int(row["turn_id"]) > 1)
    turn1 = [row for row in rows if int(row["turn_id"]) == 1]
    stored_hits = [row for row in rows
                   if int(row["turn_id"]) > 1
                   and row["method_key"] != "recompute"]
    mpic_hits = [row for row in stored_hits if row["method_key"] == "mpic32"]
    image_ids = sorted({str(row["image_id"]) for row in rows})
    persistence_docs = [
        _read_json(run_dir / "persistence" / f"{image_id}.json")
        for image_id in image_ids
        if (run_dir / "persistence" / f"{image_id}.json").is_file()
    ]
    integrity_docs = [
        _read_json(run_dir / "mpic_image_integrity" / f"{image_id}.json")
        for image_id in image_ids
        if (run_dir / "mpic_image_integrity" / f"{image_id}.json").is_file()
    ]
    prompt_groups: dict[tuple[str, str], set[str]] = {}
    q1_predictions: dict[tuple[str, str], set[tuple[str, int]]] = {}
    q1_input_hashes: dict[tuple[str, str], set[tuple[str, str, str]]] = {}
    for row in rows:
        key = (str(row["image_id"]), str(row["question_id"]))
        prompt_groups.setdefault(key, set()).add(str(row["prompt_sha256"]))
        if int(row["turn_id"]) == 1:
            q1_predictions.setdefault(key, set()).add((
                str(row["prediction"]), int(row["first_token_id"])))
            q1_input_hashes.setdefault(key, set()).add((
                str(row.get("input_tensors_sha256")),
                str(row.get("image_input_sha256")),
                str(row.get("suffix_ids_sha256"))))
    checks = {
        "expected_1200_requests": len(rows) == 1200,
        "unique_request_ids": len(observed_ids) == len(set(observed_ids)),
        "exact_expected_schedule_coverage": set(expected_ids) == completed,
        "240_requests_per_method": all(
            method_counts[key] == 240 for key in METHOD_KEYS),
        "200_cache_hit_requests_per_method": all(
            hit_counts[key] == 200 for key in METHOD_KEYS),
        "q1_all_normal_pixels": len(turn1) == 200 and all(
            row["request_path"] == "normal_pixel_turn1"
            and int(row.get("vision_forward_count", -1)) == 1
            for row in turn1),
        "stored_hits_never_run_vision": len(stored_hits) == 800 and all(
            int(row.get("vision_forward_count", -1)) == 0
            for row in stored_hits),
        "all_prompts_method_identical": len(prompt_groups) == 240 and all(
            len(values) == 1 for values in prompt_groups.values()),
        "q1_normal_pixel_outputs_identical": len(q1_predictions) == 40 and all(
            len(values) == 1 for values in q1_predictions.values()),
        "q1_normal_pixel_inputs_identical": len(q1_input_hashes) == 40 and all(
            len(values) == 1
            and all(part not in {"None", ""} for part in next(iter(values)))
            for values in q1_input_hashes.values()),
        "no_future_question_leakage": all(
            int(row.get("future_questions_in_prompt", -1)) == 0
            for row in rows),
        "no_unresolved_failures": not unresolved,
        "all_requests_status_ok": all(
            row.get("status") == "ok"
            and 0 <= int(row.get("retry_count", -1)) <= MAX_TECHNICAL_RETRIES
            for row in rows),
        "mpic_200_hits": len(mpic_hits) == 200,
        "mpic_k32_or_n": bool(mpic_hits) and all(
            int(row["n_recomputed_image_tokens"])
            == min(MPIC_K, int(row["n_image_tokens"])) for row in mpic_hits),
        "mpic_full_context_retained": bool(mpic_hits) and all(
            float(row["retained_image_context_ratio"]) == 1.0
            and int(row["n_recomputed_image_tokens"])
            + int(row["n_reused_image_tokens"]) == int(row["n_image_tokens"])
            for row in mpic_hits),
        "mpic_one_selective_prefill": bool(mpic_hits) and all(
            int(row.get("decoder_prefill_pass_count", -1)) == 1
            for row in mpic_hits),
        "mpic_main_pilot_same_position_context": bool(mpic_hits) and all(
            row.get("same_source_target_context") is True
            and row.get("same_source_target_positions") is True
            and row.get("source_position_hash")
            == row.get("target_position_hash")
            and row.get("position_handling_policy")
            == "post_rope_cached_k_reused_at_identical_logical_position"
            for row in mpic_hits),
        "mpic_exact_32_layer_counters": bool(mpic_hits) and all(
            len(row.get("active_rows_per_layer", [])) == 32
            and len(row.get("recomputed_image_rows_per_layer", [])) == 32
            and len(row.get("recomputed_text_rows_per_layer", [])) == 32
            and len(row.get("attention_key_length_per_layer", [])) == 32
            and len(row.get("valid_image_key_count_per_layer", [])) == 32
            and all(int(value) == int(row["n_recomputed_image_tokens"])
                    for value in row["recomputed_image_rows_per_layer"])
            and all(int(value) == int(row["n_recomputed_text_tokens"])
                    for value in row["recomputed_text_rows_per_layer"])
            and all(int(value) == (
                int(row["n_recomputed_image_tokens"])
                + int(row["n_recomputed_text_tokens"]))
                    for value in row["active_rows_per_layer"])
            and len(set(int(value) for value in
                        row["attention_key_length_per_layer"])) == 1
            and [int(value) for value in
                 row.get("prefill_cache_lengths", [])]
            == [int(value) for value in
                row["attention_key_length_per_layer"]]
            and all(int(value) == int(row["n_image_tokens"])
                    for value in row["valid_image_key_count_per_layer"])
            for row in mpic_hits),
        "mpic_selected_exact_leading_rows": bool(mpic_hits) and all(
            row.get("selected_image_local_rows")
            == list(range(int(row["n_recomputed_image_tokens"])))
            and row.get("selected_image_logical_rows")
            == row.get("target_positions", [])[
                :int(row["n_recomputed_image_tokens"])]
            for row in mpic_hits),
        "mpic_source_payload_immutable": bool(mpic_hits) and all(
            row.get("source_payload_hash_before")
            == row.get("source_payload_hash_after") for row in mpic_hits),
        "mpic_exact_chunk_aligned_ssd_accounting": bool(mpic_hits) and all(
            int(row.get("ssd_kv_bytes", 0)) > 0
            and int(row.get("ssd_embedding_bytes", 0)) > 0
            and int(row.get("ssd_total_bytes", -1))
            == int(row.get("ssd_kv_bytes", 0))
            + int(row.get("ssd_embedding_bytes", 0))
            and int(row.get("ssd_separator_bytes", -1)) == 0
            and int(row.get("ssd_metadata_bytes", -1)) == 0
            and int(row.get("pread_count", 0)) == 65
            and int(row.get("ssd_preads", 0)) == 65
            and math.isclose(float(row.get("actual_kv_read_ratio", -1.0)),
                             1.0, rel_tol=0.0, abs_tol=0.0)
            and int(row.get("io", {}).get("per_kind", {}).get(
                "embedding", {}).get("preads", 0)) == 1
            and int(row.get("io", {}).get("per_kind", {}).get(
                "kv_k", {}).get("preads", 0)) == 32
            and int(row.get("io", {}).get("per_kind", {}).get(
                "kv_v", {}).get("preads", 0)) == 32
            for row in mpic_hits),
        "mpic_decode_cache_append_exact": bool(mpic_hits) and all(
            row.get("decode_cache_append_exact") is True
            for row in mpic_hits),
        "all_image_persistence_evidence": all(
            (run_dir / "persistence" / f"{image_id}.json").is_file()
            for image_id in image_ids),
        "all_persistence_measurements_complete": (
            len(persistence_docs) == N_IMAGES and all(
                len(document.get("stores", {})) == 3
                and all(store.get("measurement_complete") is True
                        for store in document["stores"].values())
                for document in persistence_docs)),
        "all_mpic_image_boundary_hashes_unchanged": (
            len(integrity_docs) == N_IMAGES and all(
                document.get("unchanged") is True
                and document.get("validated_at_context_open_sha256")
                == document.get("live_after_all_hits_sha256")
                for document in integrity_docs)),
    }
    return {
        "schema_version": SCHEMA_VERSION, "passed": all(checks.values()),
        "checks": checks, "expected_requests": len(expected_ids),
        "observed_requests": len(rows), "method_counts": dict(method_counts),
        "cache_hit_counts": dict(hit_counts),
        "failure_events": len(failures),
        "unresolved_failures": len(unresolved),
        "limitations": [
            "single-image fixed-prefix GQA; not native multi-image MPIC",
            "cache hits are page-cache conditioned with DONTNEED, which does "
            "not guarantee a cold SSD controller cache",
            "MPIC-32 is the repository SSD adaptation, not official MPIC code",
            "mixed cache-hit/cache-miss overlap is not applicable here",
            "layer-wise prefetch is not implemented",
        ],
    }


def _write_summary_csv(path: Path, summary: Mapping[str, Mapping[str, Any]]):
    import csv
    rows = [{"method_key": key, **summary[key]} for key in METHOD_KEYS]
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys and not isinstance(row[key], (dict, list)):
                keys.append(key)
    payload = []
    from io import StringIO
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    payload.append(stream.getvalue())
    _atomic_bytes(path, "".join(payload).encode("utf-8"))


def _finalize(run_dir: Path, results_dir: Path, config: dict,
              schedule: Sequence[Mapping[str, Any]],
              failures: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    raw_partial = run_dir / "results_partial.jsonl"
    rows, repaired = _base52().read_jsonl_unique(raw_partial)
    if repaired:
        raise AssertionError("finalization unexpectedly repaired raw JSONL")
    validation = _validate(rows, schedule, failures, run_dir)
    validation.update({
        "validation_scope": "measurement_and_mpic_correctness_only",
        "artifact_protection_status": "pending_external_verify_after",
        "final_validated_claim": False,
    })
    summary = _summary(rows)
    persistence = [_read_json(path) for path in sorted(
        (run_dir / "persistence").glob("*.json"))]
    metadata = [_read_json(path) for path in sorted(
        (run_dir / "store_metadata").glob("*.json"))]
    config.update({
        "status": (
            "measurements_complete_pending_artifact_protection"
            if validation["passed"] else "failed_validation"),
        "finished_at_unix": time.time(),
    })
    _atomic_json(run_dir / "config.json", config)
    _atomic_bytes(run_dir / "raw.jsonl", raw_partial.read_bytes())
    _atomic_json(run_dir / "summary.json", {
        "schema_version": SCHEMA_VERSION, "per_method": summary})
    _write_summary_csv(run_dir / "summary.csv", summary)
    _atomic_json(run_dir / "persistence.json", {
        "schema_version": SCHEMA_VERSION, "images": persistence})
    _atomic_json(run_dir / "store_metadata.json", {
        "schema_version": SCHEMA_VERSION, "images": metadata})
    _atomic_json(run_dir / "validation.json", validation)
    exported = (
        "config.json", "manifest.json", "raw.jsonl", "summary.json",
        "summary.csv", "persistence.json", "store_metadata.json",
        "validation.json", "runtime_fingerprint.json",
    )
    for name in exported:
        _atomic_bytes(results_dir / name, (run_dir / name).read_bytes())
    artifacts = {
        "schema_version": SCHEMA_VERSION, "run_dir": str(run_dir),
        "results_dir": str(results_dir),
        "files_sha256": {name: sha256_file(run_dir / name)
                         for name in exported},
    }
    _atomic_json(run_dir / "run_artifacts.json", artifacts)
    _atomic_json(results_dir / "run_artifacts.json", artifacts)
    return validation


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-dir", type=Path,
                      help=("protector-prepared runs/mpic_baseline/<run_id> "
                            "directory"))
    mode.add_argument("--resume", type=Path,
                      help="incomplete runs/mpic_baseline/<run_id> directory")
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--index", type=Path, default=ROOT / "data/index.json")
    parser.add_argument("--smoke-validation", type=Path,
                        default=DEFAULT_SMOKE_VALIDATION)
    parser.add_argument("--seed", type=int, default=VALIDATED_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--max-new-requests", type=int,
        help="stop after this many newly durable rows; resume later")
    args = parser.parse_args(argv)
    if args.run_dir is not None and args.results_dir is None:
        parser.error("--results-dir is required for a new run")
    if args.resume is not None and args.results_dir is not None:
        parser.error("--results-dir is immutable and forbidden with --resume")
    if args.max_new_requests is not None and args.max_new_requests <= 0:
        parser.error("--max-new-requests must be positive")
    if args.max_new_tokens != 16:
        parser.error("the validated pilot fixes --max-new-tokens=16")
    if args.seed != VALIDATED_SEED:
        parser.error(f"the validated pilot fixes --seed={VALIDATED_SEED}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.resume is None:
        run_dir, results_dir, options, is_resume = _prepare_new(args)
    else:
        run_dir, results_dir, options, is_resume = _prepare_resume(args)

    with _base52().run_lock(run_dir):
        disk_usage = shutil.disk_usage(run_dir)
        if disk_usage.free < MIN_FREE_BYTES_FOR_IMAGE_SESSION:
            raise OSError(
                "insufficient free SSD space for three run-local per-image "
                f"stores: {disk_usage.free} < "
                f"{MIN_FREE_BYTES_FOR_IMAGE_SESSION} bytes")
        index_path = Path(options["index"])
        entries, workload = _validate_workload(index_path)
        schedule = expected_schedule(entries, int(options["seed"]))
        if len(schedule) != 1200:
            raise AssertionError(f"expected 1200 requests, got {len(schedule)}")
        expected_ids = [row["request_id"] for row in schedule]
        if len(expected_ids) != len(set(expected_ids)):
            raise AssertionError("expected schedule contains duplicate IDs")
        smoke = _validate_smoke(Path(options["smoke_validation"]))
        source_hashes = _source_hashes()
        if source_hashes["paper"] != EXPECTED_PAPER_SHA256:
            raise ValueError("local MPIC paper changed from reviewed source")

        if is_resume:
            manifest = _read_json(run_dir / "manifest.json")
            if manifest["expected_request_ids_sha256"] != canonical_hash(
                    expected_ids):
                raise ValueError("resume schedule differs from manifest")
            if manifest["source_sha256"] != source_hashes:
                raise ValueError("source changed since the run began")
            if manifest["smoke_validation"]["sha256"] != smoke["sha256"]:
                raise ValueError("smoke evidence changed since the run began")
            config = _read_json(run_dir / "config.json")
            config["resume_count"] = int(config.get("resume_count", 0)) + 1
            config.setdefault("resume_events", []).append({
                "at_unix": time.time(), "pid": os.getpid()})
        else:
            run_id = run_dir.name
            manifest = {
                "schema_version": SCHEMA_VERSION, "run_id": run_id,
                "created_at_unix": time.time(), "options": options,
                "method_keys": list(METHOD_KEYS), "methods": METHODS,
                "expected_request_count": len(schedule),
                "expected_request_ids_sha256": canonical_hash(expected_ids),
                "workload": workload, "source_sha256": source_hashes,
                "smoke_validation": smoke,
                "output_policy": "new_isolated_paths_no_overwrite",
                "store_policy": (
                    "run-local per-image stores; retain while incomplete; "
                    "delete only after all 30 image rows are durable"),
            }
            _atomic_json(run_dir / "manifest.json", manifest, exclusive=True)
            config = {
                "schema_version": SCHEMA_VERSION, "run_id": run_id,
                "status": "initializing", "dataset": "gqa", **workload,
                "n_images": N_IMAGES, "n_questions": EXPECTED_QUESTIONS,
                "questions_per_image": QUESTIONS_PER_IMAGE,
                "question_slice": [QUESTION_SKIP,
                                   QUESTION_SKIP + QUESTIONS_PER_IMAGE],
                "seed": int(options["seed"]),
                "max_new_tokens": int(options["max_new_tokens"]),
                "method_keys": list(METHOD_KEYS), "methods": METHODS,
                "model": MODEL_ID, "model_revision": "checkpoint_default",
                "load_4bit": bool(LOAD_4BIT), "quantization": "NF4",
                "compute_dtype": COMPUTE_DTYPE,
                "attention_implementation": ATTN_IMPL,
                "decoding": "greedy", "chunk_size": CHUNK_SIZE,
                "probe_heads": PROBE_HEADS, "mpic_k": MPIC_K,
                "run_dir": str(run_dir), "results_dir": str(results_dir),
                "history_policy": "none_independent_gqa_questions",
                "turn1_policy": (
                    "all five arms use ordinary Image+Q1 pixel inference; "
                    "stores piggyback their arm's answer-producing forward"),
                "cache_hit_policy": (
                    "Q2-Q6: ReComp pixels; FullLoad/QA-Chunk25 share raster; "
                    "Ours25 uses repacked image-only store; MPIC-32 uses its "
                    "dedicated canonical KV plus visual-input sidecar"),
                "ttft_definition": (
                    "request start before prompt/tokenization/input preparation; "
                    "include required SSD reads, H2D, cache assembly, prefill, "
                    "first-token materialization and CUDA sync"),
                "page_cache_conditioning": (
                    "posix_fadvise_DONTNEED outside TTFT; not a guarantee of "
                    "cold physical SSD/controller cache"),
                "retry_policy": {
                    "maximum_technical_retries_per_request":
                        MAX_TECHNICAL_RETRIES,
                    "quality_based_regeneration": False,
                    "retries_require_explicit_resume": True,
                },
                "disk_space_policy": {
                    "free_bytes_at_start": int(disk_usage.free),
                    "total_bytes_at_start": int(disk_usage.total),
                    "minimum_free_bytes": MIN_FREE_BYTES_FOR_IMAGE_SESSION,
                    "peak_storage_strategy": (
                        "three per-image stores only; delete run-local stores "
                        "after all 30 records for that image are durable"),
                },
                "started_at_unix": time.time(), "resume_count": 0,
                "resume_events": [],
            }
            _atomic_json(run_dir / "config.json", config, exclusive=True)

        raw_path = run_dir / "results_partial.jsonl"
        rows, repaired = _base52().read_jsonl_unique(
            raw_path, repair_tail=is_resume, recovery_dir=run_dir / "recovery")
        completed = {str(row["request_id"]) for row in rows}
        if not completed <= set(expected_ids):
            raise ValueError("partial JSONL contains foreign request IDs")
        failure_path = run_dir / "failures.jsonl"
        failures = ([json.loads(line) for line in failure_path.read_text(
            encoding="utf-8").splitlines()] if failure_path.exists() else [])
        last_request = None
        newly_completed = 0
        _atomic_json(run_dir / "progress.json", _progress(
            config["run_id"], len(schedule), completed, failures,
            "resuming" if is_resume else "starting", None, repaired))

        random.seed(int(options["seed"]))
        np.random.seed(int(options["seed"]))
        torch.manual_seed(int(options["seed"]))
        torch.cuda.manual_seed_all(int(options["seed"]))
        runner = LlavaRunner().load()
        current_runtime = _runtime_fingerprint(runner)
        runtime_path = run_dir / "runtime_fingerprint.json"
        if is_resume:
            frozen_runtime = _read_json(runtime_path)
            if frozen_runtime != current_runtime:
                raise ValueError(
                    "model/runtime differs from the immutable first "
                    "invocation; refusing to serve resumable stores")
        else:
            _atomic_json(runtime_path, current_runtime, exclusive=True)
        config["model_revision"] = current_runtime["model_revision"]
        config["model_class"] = current_runtime["model_class"]
        config["processor_class"] = current_runtime["processor_class"]
        config["runtime"] = current_runtime
        legacy_server = Server(
            runner, ratio=0.25, probe=PROBE_HEADS,
            max_new_tokens=int(options["max_new_tokens"]))
        mpic_server = MPICServer(
            runner, k_recompute=MPIC_K,
            max_new_tokens=int(options["max_new_tokens"]))
        warmup = _base49()._warmup(runner, legacy_server)
        config.setdefault("warmup_events", []).append({
            "at_unix": time.time(), "resume": is_resume, **warmup})
        config.setdefault("invocations", []).append({
            "at_unix": time.time(), "pid": os.getpid(),
            "resume": is_resume,
            "max_new_requests": args.max_new_requests,
        })
        config["status"] = "running"
        _atomic_json(run_dir / "config.json", config)

        run_started = time.perf_counter()
        try:
            for image_index, entry in enumerate(entries):
                image_id = str(entry["image_id"])
                questions = entry["questions"][
                    QUESTION_SKIP:QUESTION_SKIP + QUESTIONS_PER_IMAGE]
                order = deterministic_method_rotation(
                    METHOD_KEYS, image_index, int(options["seed"]))
                image_expected = {
                    request_id(image_id, str(question["question_id"]), method)
                    for question in questions for method in METHOD_KEYS}
                paths = _store_paths(run_dir, image_id)
                if image_expected <= completed:
                    # A process can stop after the final durable request row
                    # but before the post-image payload audit and cleanup.
                    # Recover that narrow boundary from the still-owned MPIC
                    # store instead of deleting the only remaining integrity
                    # evidence.  If the normal path already published the
                    # audit, cleanup remains idempotent.
                    integrity_path = (
                        run_dir / "mpic_image_integrity" / f"{image_id}.json")
                    if not integrity_path.is_file():
                        if not paths["mpic"].is_dir():
                            raise ValueError(
                                "completed image lacks both MPIC integrity "
                                f"evidence and recoverable store: {image_id}")
                        recovered_mpic = MPICContext(
                            paths["mpic"], runner.model.device, runner=runner)
                        try:
                            _write_mpic_image_integrity(
                                run_dir, image_id, recovered_mpic)
                        finally:
                            recovered_mpic.close()
                    _remove_completed_image_stores(run_dir, image_id, paths)
                    continue

                image_path = ROOT / entry["image_path"]
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                provisions: dict[str, dict[str, Any]] = {}

                # Turn 1: all methods use normal pixels.  Store-owning arms
                # publish their same-request capture before the raw row.
                first = questions[0]
                for position, method_key in enumerate(order):
                    identity = request_id(
                        image_id, str(first["question_id"]), method_key)
                    provision_path = _provision_path(
                        run_dir, image_id, method_key)
                    pending_path = _pending_path(
                        run_dir, image_id, method_key)
                    owner = STORE_OWNER.get(method_key)
                    if identity in completed:
                        if owner is not None:
                            if not provision_path.is_file():
                                raise ValueError(
                                    f"completed owner lacks provision: {identity}")
                            provisions[method_key] = _recover_provision(
                                provision_path, identity)
                            if not paths[STORE_OWNER[method_key]].is_dir():
                                raise ValueError(
                                    f"incomplete image lacks store: {identity}")
                        continue
                    if _base52().request_cap_reached(
                            newly_completed, args.max_new_requests):
                        raise PartialRunStop
                    try:
                        if provision_path.exists():
                            provision = _recover_provision(
                                provision_path, identity)
                            _append_row(raw_path, provision["record"])
                            record = provision["record"]
                            provisions[method_key] = provision
                        elif (owner is not None
                              and os.path.lexists(paths[owner])):
                            # The store publication is atomic.  A durable
                            # pre-publication pending row lets resume validate
                            # and adopt an orphan store without rerunning Q1.
                            pending = _recover_pending(pending_path, identity)
                            persistence = _recover_published_store(
                                method_key, paths[owner], runner)
                            provision = {
                                "schema_version": SCHEMA_VERSION,
                                "request_id": identity,
                                "record": pending["record"],
                                "persistence": persistence,
                                "published_at_unix": time.time(),
                                "recovered_from_pending": True,
                            }
                            _atomic_json(
                                provision_path, provision, exclusive=True)
                            record = pending["record"]
                            provisions[method_key] = provision
                            _append_row(raw_path, record)
                        else:
                            retry_count = _require_retry_budget(
                                failures, identity)
                            capture_kind = (
                                "qa" if method_key in {"fullload", "mpic32"}
                                else "ours" if method_key == "ours25"
                                else "none")
                            pixel, diagnostic = _base49()._run_pixels(
                                runner, legacy_server, image,
                                first["question"], capture_kind)
                            _, n_image_tokens = runner.visual_span(
                                diagnostic["enc_cpu"]["input_ids"])
                            normalized = _normalize_pixel(pixel, method_key)
                            normalized["retry_count"] = retry_count
                            record = _record(
                                config=config, runner=runner, question=first,
                                image_id=image_id, turn_id=1,
                                method_key=method_key, method_order=order,
                                position=position, result=normalized,
                                diagnostic=diagnostic,
                                request_path="normal_pixel_turn1",
                                n_image_tokens=n_image_tokens,
                                store_capture=owner)
                            if owner is not None:
                                pending = {
                                    "schema_version": SCHEMA_VERSION,
                                    "request_id": identity, "record": record,
                                    "store_owner": owner,
                                    "prepared_at_unix": time.time(),
                                }
                                if pending_path.exists():
                                    previous = _recover_pending(
                                        pending_path, identity)["record"]
                                    invariant_fields = (
                                        "prompt_sha256", "suffix_ids_sha256",
                                        "input_tensors_sha256",
                                        "image_input_sha256", "prediction",
                                        "first_token_id")
                                    if any(previous.get(field) != record.get(field)
                                           for field in invariant_fields):
                                        raise ValueError(
                                            "retried Turn-1 output/input differs "
                                            f"from pending transaction: {identity}")
                                    _atomic_json(pending_path, pending)
                                else:
                                    _atomic_json(
                                        pending_path, pending, exclusive=True)
                                persistence = _persistence_call(
                                    method_key, runner, pixel, diagnostic,
                                    paths[owner], image_id,
                                    str(first["question_id"]),
                                    config["run_id"], identity)
                                persistence["measurement_complete"] = True
                                provision = {
                                    "schema_version": SCHEMA_VERSION,
                                    "request_id": identity, "record": record,
                                    "persistence": persistence,
                                    "published_at_unix": time.time(),
                                }
                                _atomic_json(
                                    provision_path, provision, exclusive=True)
                                provisions[method_key] = provision
                            _append_row(raw_path, record)
                            del pixel, diagnostic
                        completed.add(identity)
                        rows.append(record)
                        newly_completed += 1
                        last_request = identity
                        _atomic_json(run_dir / "progress.json", _progress(
                            config["run_id"], len(schedule), completed,
                            failures, "running", last_request, repaired))
                    except PartialRunStop:
                        raise
                    except Exception as error:
                        failure = {
                            "schema_version": SCHEMA_VERSION,
                            "run_id": config["run_id"],
                            "request_id": identity, "image_id": image_id,
                            "question_id": str(first["question_id"]),
                            "turn_id": 1, "method_key": method_key,
                            "failed_at_unix": time.time(),
                            "exception_type": type(error).__name__,
                            "exception": str(error),
                            "traceback": traceback.format_exc(),
                        }
                        _append_row(failure_path, failure)
                        failures.append(failure)
                        _atomic_json(run_dir / "progress.json", _progress(
                            config["run_id"], len(schedule), completed,
                            failures, "failed_request", identity, repaired))
                        raise

                missing = set(STORE_OWNER) - set(provisions)
                if missing:
                    raise AssertionError(
                        f"Turn-1 store provisions missing: {sorted(missing)}")
                raster_ctx, ours_ctx, mpic_ctx = _open_contexts(paths, runner)
                full_visual_bytes = int(raster_ctx.meta["bytes_visual_kv"])
                n_image_tokens = int(raster_ctx.meta["v_token_num"])
                _image_metadata(
                    run_dir, image_id, raster_ctx, ours_ctx, mpic_ctx,
                    provisions)
                try:
                    for turn_id, question in enumerate(questions[1:], 2):
                        for position, method_key in enumerate(order):
                            identity = request_id(
                                image_id, str(question["question_id"]),
                                method_key)
                            if identity in completed:
                                continue
                            if _base52().request_cap_reached(
                                    newly_completed, args.max_new_requests):
                                raise PartialRunStop
                            try:
                                retry_count = _require_retry_budget(
                                    failures, identity)
                                if method_key == "recompute":
                                    raw_result, diagnostic = \
                                        _base49()._run_pixels(
                                            runner, legacy_server, image,
                                            question["question"], "none")
                                    result = _normalize_pixel(
                                        raw_result, method_key)
                                    request_path = "normal_pixel_recompute"
                                elif method_key == "mpic32":
                                    result, diagnostic = _run_mpic(
                                        runner, mpic_server, mpic_ctx,
                                        question["question"],
                                        full_visual_bytes)
                                    request_path = "ssd_cache_hit_mpic"
                                else:
                                    context = (ours_ctx if method_key == "ours25"
                                               else raster_ctx)
                                    result, diagnostic = _base52()._run_stored(
                                        runner, legacy_server, context,
                                        question["question"], method_key,
                                        image_id, full_visual_bytes)
                                    request_path = "ssd_cache_hit"
                                result["retry_count"] = retry_count
                                record = _record(
                                    config=config, runner=runner,
                                    question=question, image_id=image_id,
                                    turn_id=turn_id, method_key=method_key,
                                    method_order=order, position=position,
                                    result=result, diagnostic=diagnostic,
                                    request_path=request_path,
                                    n_image_tokens=n_image_tokens)
                                _append_row(raw_path, record)
                                completed.add(identity)
                                rows.append(record)
                                newly_completed += 1
                                last_request = identity
                                _atomic_json(
                                    run_dir / "progress.json", _progress(
                                        config["run_id"], len(schedule),
                                        completed, failures, "running",
                                        last_request, repaired))
                            except PartialRunStop:
                                raise
                            except Exception as error:
                                failure = {
                                    "schema_version": SCHEMA_VERSION,
                                    "run_id": config["run_id"],
                                    "request_id": identity,
                                    "image_id": image_id,
                                    "question_id": str(question["question_id"]),
                                    "turn_id": turn_id,
                                    "method_key": method_key,
                                    "failed_at_unix": time.time(),
                                    "exception_type": type(error).__name__,
                                    "exception": str(error),
                                    "traceback": traceback.format_exc(),
                                }
                                _append_row(failure_path, failure)
                                failures.append(failure)
                                _atomic_json(
                                    run_dir / "progress.json", _progress(
                                        config["run_id"], len(schedule),
                                        completed, failures, "failed_request",
                                        identity, repaired))
                                raise
                    _write_mpic_image_integrity(
                        run_dir, image_id, mpic_ctx)
                finally:
                    raster_ctx.close()
                    ours_ctx.close()
                    mpic_ctx.close()
                    del raster_ctx, ours_ctx, mpic_ctx, image
                    torch.cuda.empty_cache()

                if not image_expected <= completed:
                    raise AssertionError("image loop ended without full coverage")
                _remove_completed_image_stores(run_dir, image_id, paths)
                print(
                    f"[{image_index + 1}/{N_IMAGES}] {image_id} "
                    f"completed={len(completed)}/{len(schedule)} "
                    f"elapsed={time.perf_counter() - run_started:.1f}s",
                    flush=True)
        except PartialRunStop:
            config["status"] = "partial"
            config["partial_at_unix"] = time.time()
            config["last_invocation_new_requests"] = newly_completed
            _atomic_json(run_dir / "config.json", config)
            _atomic_json(run_dir / "progress.json", _progress(
                config["run_id"], len(schedule), completed, failures,
                "partial", last_request, repaired))
            print(json.dumps({
                "run_dir": str(run_dir), "status": "partial",
                "new_requests": newly_completed, "completed": len(completed),
                "expected": len(schedule), "resume": f"--resume {run_dir}",
            }, indent=2), flush=True)
            return 0
        except BaseException:
            config["status"] = "interrupted_or_failed"
            config["last_failure_at_unix"] = time.time()
            _atomic_json(run_dir / "config.json", config)
            raise

        validation = _finalize(
            run_dir, results_dir, config, schedule, failures)
        _atomic_json(run_dir / "progress.json", _progress(
            config["run_id"], len(schedule), completed, failures,
            ("measurements_completed_pending_artifact_protection"
             if validation["passed"] else "failed_validation"),
            last_request, repaired))
        if validation["passed"]:
            _atomic_json(run_dir / "COMPLETED", {
                "schema_version": SCHEMA_VERSION,
                "run_id": config["run_id"], "measurement_checks_passed": True,
                "completion_scope": "measurement_runner_only",
                "artifact_protection_status": "pending_external_verify_after",
                "final_validated_claim": False,
                "completed": len(completed),
                "validation_sha256": sha256_file(
                    run_dir / "validation.json"),
                "finished_at_unix": time.time(),
            }, exclusive=True)
        print(json.dumps({
            "run_dir": str(run_dir), "results_dir": str(results_dir),
            "measurement_checks_passed": validation["passed"],
            "artifact_protection_status": "pending_external_verify_after",
            "completed": len(completed),
            "expected": len(schedule),
        }, indent=2), flush=True)
        if not validation["passed"]:
            failed = [key for key, value in validation["checks"].items()
                      if not value]
            raise RuntimeError("validation failed: " + ", ".join(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

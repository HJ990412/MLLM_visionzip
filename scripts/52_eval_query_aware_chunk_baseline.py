#!/usr/bin/env python3
"""Persistent five-arm GQA pilot for the QA-Chunk25 SSD baseline.

This evaluator extends the validated QA-Token25 experiment without rebuilding
or modifying either canonical store.  It reads the raster and image-only trees
from ``gqa40_240_final_store`` in place and runs, in a balanced per-image order,

    ReComp, FullLoad, QA-Token25, QA-Chunk25, Ours25.

Every completed request is appended and fsynced before ``progress.json`` is
advanced.  ``--resume RUN_DIR`` validates the immutable run manifest, rejects
duplicate request identities, skips durable rows, and continues the same
schedule.  Derived summaries are produced only by re-reading the persisted
JSONL.  A ``COMPLETED`` marker is published only after all validation gates and
all final artifact writes succeed.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import stat
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mmimpress.config import CHUNK_SIZE, PROBE_HEADS  # noqa: E402
from mmimpress.cvpr25 import budget_chunk_count  # noqa: E402
from mmimpress.dataset import METRICS, question_answers  # noqa: E402
from mmimpress.model import LlavaRunner  # noqa: E402
from mmimpress.piggyback import deterministic_method_rotation  # noqa: E402
from mmimpress.serve import ImageContext, Server, contiguous_runs  # noqa: E402


SCHEMA_VERSION = "qa-chunk-gqa-pilot-v1"
EXPECTED_INDEX_SHA256 = (
    "514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a"
)
EXPECTED_FULL_WORKLOAD_SHA256 = (
    "97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66"
)
EXPECTED_REFERENCE_HASHES = {
    "config.json": "dfe1629869f7ba6662556bb359d6c52e796c7b7b2c25893f6a89d0e75de6191c",
    "raw.jsonl": "cb39d32d5490a1356ffa307493f8613cd018f06394338c996302bb08035ce63b",
    "validation.json": "dc9d58e41a1ec954c3c941ef18c07501dfc555457ff78f94edd5129412f74c00",
    "summary.json": "a75abac0152dc6e3b69c3d6d125594d41f540a383faf84e268e483ed7805db7a",
    "selection.json": "df65c39b5b961e2d448e718217e397b5b128139e088c8a1246bbe3992b4e532d",
}
DEFAULT_STORE_DIR = (
    ROOT / "runs/query_aware_baseline/gqa40_240_final_store"
)
DEFAULT_REFERENCE_RUN_DIR = (
    ROOT / "runs/query_aware_baseline/gqa40_240_final"
)
REFERENCE_EQUIVALENCE_RATE = 0.02
TTFT_RELATIVE_TOLERANCE = 0.35
TTFT_ABSOLUTE_TOLERANCE_MS = 125.0
QA_TOKEN_SELECTOR_RELATIVE_TOLERANCE = 0.50
QA_TOKEN_SELECTOR_ABSOLUTE_TOLERANCE_MS = 40.0

METHOD_KEYS = (
    "recompute", "fullload", "qa_token25", "qa_chunk25", "ours25",
)
METHODS = {
    "recompute": {
        "method_id": "recompute", "display_label": "ReComp",
        "paper_label": "ReComp", "retention_ratio": None,
        "importance_source": "none", "query_dependent": False,
        "selection_granularity": "none", "physical_layout": "none",
        "repacking": False, "online_selection": False,
    },
    "fullload": {
        "method_id": "fullload", "display_label": "FullLoad",
        "paper_label": "FullLoad", "retention_ratio": 1.0,
        "importance_source": "none", "query_dependent": False,
        "selection_granularity": "full_visual_kv",
        "physical_layout": "raster", "repacking": False,
        "online_selection": False,
    },
    "qa_token25": {
        "method_id": "qa_token25", "display_label": "QA-Token25",
        "paper_label": "Query-Aware Token", "retention_ratio": 0.25,
        "importance_source": (
            "SparseVLM-style text-guided visual importance"),
        "selection_granularity": "visual_token", "query_dependent": True,
        "physical_layout": "raster", "repacking": False,
        "online_selection": True,
    },
    "qa_chunk25": {
        "method_id": "qa_chunk25", "display_label": "QA-Chunk25",
        "paper_label": "Query-Aware Chunk", "retention_ratio": 0.25,
        "importance_source": (
            "SparseVLM-style text-guided visual importance"),
        "selection_granularity": "ssd_chunk",
        "chunk_score": "mean_valid_spatial_token_importance",
        "query_dependent": True, "physical_layout": "raster",
        "repacking": False, "online_selection": True,
    },
    "ours25": {
        "method_id": "imageonly_prefix25", "display_label": "Ours25",
        "paper_label": "Ours", "retention_ratio": 0.25,
        "importance_source": "image-only Vision Encoder saliency",
        "selection_granularity": "ssd_chunk_prefix",
        "query_dependent": False,
        "physical_layout": "importance-aware repacked",
        "repacking": True, "online_selection": False,
    },
}
REFERENCE_METHOD = {
    "recompute": "recompute",
    "fullload": "fullload",
    "qa_token25": "qa_select25",
    "ours25": "ours25",
}

_BASE = None


class PartialRunStop(RuntimeError):
    """Internal control flow for a deliberate, durable partial checkpoint."""


def _base_module():
    """Load the validated evaluator as a helper module without invoking main."""
    global _BASE
    if _BASE is None:
        path = ROOT / "scripts/49_eval_query_aware_baseline.py"
        spec = importlib.util.spec_from_file_location(
            "_validated_qa_token_evaluator", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load validated evaluator: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _BASE = module
    return _BASE


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")).hexdigest()


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, payload: bytes, *, exclusive: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError(f"output parent is not a real directory: {path.parent}")
    if exclusive and os.path.lexists(path):
        raise FileExistsError(f"refusing to replace existing file: {path}")
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(temporary, path)
            temporary.unlink()
        else:
            os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True,
                          ensure_ascii=False, allow_nan=False) + "\n").encode()
    atomic_bytes(path, payload, exclusive=exclusive)


def atomic_text(path: Path, value: str, *, exclusive: bool = False) -> None:
    atomic_bytes(path, value.encode("utf-8"), exclusive=exclusive)


def append_jsonl_durable(path: Path, value: Mapping[str, Any]) -> None:
    """Append one complete JSON record and make it durable before returning."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short JSONL append")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def repair_truncated_jsonl_tail(path: Path, recovery_dir: Path) -> int:
    """Archive and remove only a non-newline crash tail; return archived bytes."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return 0
    payload = path.read_bytes()
    if payload.endswith(b"\n"):
        return 0
    boundary = payload.rfind(b"\n") + 1
    tail = payload[boundary:]
    if not tail:
        return 0
    recovery_dir.mkdir(parents=True, exist_ok=True)
    archived = recovery_dir / (
        f"truncated-tail-{time.time_ns()}-{uuid.uuid4().hex}.bin")
    atomic_bytes(archived, tail, exclusive=True)
    descriptor = os.open(path, os.O_WRONLY)
    try:
        os.ftruncate(descriptor, boundary)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_dir(path.parent)
    return len(tail)


def read_jsonl_unique(path: Path, *, repair_tail: bool = False,
                      recovery_dir: Path | None = None) -> tuple[list[dict], int]:
    path = Path(path)
    repaired = 0
    if repair_tail and path.exists():
        if recovery_dir is None:
            raise ValueError("recovery_dir is required when repair_tail=True")
        repaired = repair_truncated_jsonl_tail(path, recovery_dir)
    if not path.exists():
        return [], repaired
    rows: list[dict] = []
    identities: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line {line_number}: {path}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"malformed JSONL line {line_number}: {path}") from error
            if not isinstance(row, dict) or not isinstance(
                    row.get("request_id"), str):
                raise ValueError(f"malformed request row {line_number}: {path}")
            identity = row["request_id"]
            if identity in identities:
                raise ValueError(f"duplicate request_id in JSONL: {identity}")
            identities.add(identity)
            rows.append(row)
    return rows, repaired


def request_id(image_id: str, question_id: str, method_key: str) -> str:
    if method_key not in METHOD_KEYS:
        raise ValueError(f"unknown method: {method_key}")
    return f"gqa:{image_id}:{question_id}:{method_key}"


def request_cap_reached(newly_completed: int,
                        max_new_requests: int | None) -> bool:
    return (max_new_requests is not None
            and int(newly_completed) >= int(max_new_requests))


def expected_schedule(entries: Sequence[Mapping[str, Any]], *, skip: int,
                      questions: int, seed: int) -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    for image_index, entry in enumerate(entries):
        image_id = str(entry["image_id"])
        selected = entry["questions"][skip:skip + questions]
        order = deterministic_method_rotation(METHOD_KEYS, image_index, seed)
        for turn_id, question in enumerate(selected, 1):
            for position, method_key in enumerate(order):
                qid = str(question["question_id"])
                schedule.append({
                    "request_id": request_id(image_id, qid, method_key),
                    "image_index": image_index, "image_id": image_id,
                    "question_id": qid, "turn_id": turn_id,
                    "method_key": method_key,
                    "method_order": list(order),
                    "method_order_position": position,
                })
    return schedule


def source_store_inventory(store_dir: Path,
                           image_ids: Iterable[str]) -> dict[str, Any]:
    """Cheap no-write fingerprint for the reused 98 GB store.

    Full hashes of all KV payloads would warm the very SSD data being measured.
    Instead, every path, type, byte size and mtime is committed, while all
    metadata JSON files are content-hashed.  The validated source run's own
    persistence evidence remains the content-integrity authority.
    """
    store = Path(store_dir).resolve()
    store_stat = store.stat(follow_symlinks=False)
    if store.is_symlink() or not stat.S_ISDIR(store_stat.st_mode):
        raise ValueError(f"source store root is not a real directory: {store}")
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for side in ("raster", "image_only"):
        for image_id in sorted(str(value) for value in image_ids):
            root = store / side / image_id
            if root.is_symlink() or not root.is_dir():
                raise ValueError(f"missing real source store directory: {root}")
            for path in sorted(root.rglob("*")):
                info = path.stat(follow_symlinks=False)
                relative = path.relative_to(store).as_posix()
                if stat.S_ISLNK(info.st_mode):
                    raise ValueError(f"source store contains symlink: {path}")
                if stat.S_ISDIR(info.st_mode):
                    entries.append({"path": relative, "type": "directory"})
                elif stat.S_ISREG(info.st_mode):
                    row = {
                        "path": relative, "type": "regular_file",
                        "size": int(info.st_size),
                        "mtime_ns": int(info.st_mtime_ns),
                    }
                    if path.name == "meta.json":
                        row["sha256"] = sha256_file(path)
                    entries.append(row)
                    total_bytes += int(info.st_size)
                else:
                    raise ValueError(f"unsupported source store entry: {path}")
    return {
        "store_dir": str(store), "scope": "selected images, both layouts",
        "store_root_device": int(store_stat.st_dev),
        "store_root_inode": int(store_stat.st_ino),
        "payload_hash_policy": (
            "all path/type/size/mtime; full SHA256 for meta.json; source "
            "persistence evidence supplies validated KV sample hashes"),
        "entry_count": len(entries), "total_bytes": total_bytes,
        "inventory_sha256": canonical_hash(entries),
        "entries": entries,
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_reference(reference_dir: Path) -> dict[str, Any]:
    reference = Path(reference_dir).resolve()
    if reference.is_symlink() or not reference.is_dir():
        raise ValueError(f"reference run is not a real directory: {reference}")
    observed = {}
    for name, expected in EXPECTED_REFERENCE_HASHES.items():
        path = reference / name
        observed[name] = sha256_file(path)
        if observed[name] != expected:
            raise ValueError(
                f"canonical reference hash mismatch for {name}: "
                f"{observed[name]} != {expected}")
    config = _read_json(reference / "config.json")
    validation = _read_json(reference / "validation.json")
    summary = _read_json(reference / "summary.json")
    if config.get("status") != "complete" or validation.get("passed") is not True:
        raise ValueError("canonical reference is not a completed validated run")
    if config.get("selected_workload_sha256") != EXPECTED_FULL_WORKLOAD_SHA256:
        raise ValueError("canonical reference workload mismatch")
    rows = [json.loads(line) for line in (
        reference / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != 960:
        raise ValueError(f"canonical reference row count is {len(rows)}, not 960")
    return {
        "run_dir": str(reference), "hashes": observed,
        "config": config, "validation": validation,
        "summary": summary, "rows": rows,
    }


@contextmanager
def run_lock(run_dir: Path):
    path = Path(run_dir) / "run.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another process owns this run: {run_dir}") from error
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _paths_overlap(left: Path, right: Path) -> bool:
    left, right = left.resolve(), right.resolve()
    return left == right or left in right.parents or right in left.parents


def _prepare_new_run(args: argparse.Namespace) -> tuple[Path, Path, dict]:
    run_dir = args.run_dir.resolve()
    results_dir = args.results_dir.resolve()
    store_dir = args.store_dir.resolve()
    reference_dir = args.reference_run_dir.resolve()
    if any(_paths_overlap(left, right) for left, right in combinations(
            (run_dir, results_dir, store_dir, reference_dir), 2)):
        raise ValueError("run/results/source/reference paths may not overlap")
    if run_dir.is_symlink():
        raise ValueError("run directory may not be a symlink")
    run_dir.mkdir(parents=True, exist_ok=True)
    allowed = {
        "command.sh", "environment.txt", "git_state.txt", "run.log",
        "tmux_session.txt", "run.lock",
    }
    unexpected = sorted(path.name for path in run_dir.iterdir()
                        if path.name not in allowed)
    if unexpected:
        raise FileExistsError(
            f"new run directory contains evaluator state: {unexpected}")
    if os.path.lexists(results_dir):
        raise FileExistsError(f"results directory must be new: {results_dir}")
    results_dir.mkdir(parents=True)
    return run_dir, results_dir, {
        "index": str(args.index.resolve()), "store_dir": str(store_dir),
        "reference_run_dir": str(reference_dir),
        "results_dir": str(results_dir), "max_images": args.max_images,
        "skip": args.skip, "questions": args.questions, "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "expected_index_sha256": args.expected_index_sha256,
        "expected_workload_sha256": args.expected_workload_sha256,
        "expected_images": args.expected_images,
        "expected_questions": args.expected_questions,
    }


def _prepare_resume(args: argparse.Namespace) -> tuple[Path, Path, dict]:
    run_dir = args.resume.resolve()
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError(f"resume run is not a real directory: {run_dir}")
    if (run_dir / "COMPLETED").exists():
        raise RuntimeError(f"run is already complete: {run_dir}")
    manifest = _read_json(run_dir / "manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("resume manifest schema mismatch")
    options = manifest.get("options")
    if not isinstance(options, dict):
        raise ValueError("resume manifest has no immutable options")
    results_dir = Path(options["results_dir"]).resolve()
    if results_dir.is_symlink() or not results_dir.is_dir():
        raise ValueError(f"resume results directory is invalid: {results_dir}")
    return run_dir, results_dir, dict(options)


def _validate_workload(options: Mapping[str, Any]):
    base = _base_module()
    index = Path(options["index"])
    all_entries, entries, workload = base._workload(
        index, int(options["skip"]), int(options["questions"]),
        int(options["max_images"]))
    if int(options["skip"]) != 4 or int(options["questions"]) != 6:
        raise ValueError("frozen pilot requires --skip 4 --questions 6")
    if workload["index_sha256"] != options["expected_index_sha256"]:
        raise ValueError("frozen index SHA256 mismatch")
    if workload["full_workload_sha256"] != options["expected_workload_sha256"]:
        raise ValueError("frozen full workload SHA256 mismatch")
    if workload["full_images"] != int(options["expected_images"]):
        raise ValueError("frozen full image count mismatch")
    if workload["full_questions"] != int(options["expected_questions"]):
        raise ValueError("frozen full question count mismatch")
    return all_entries, entries, workload


def dispatch_stored(server, method_key: str, context: ImageContext,
                    suffix_ids, image_id: str):
    """One auditable dispatch table; also unit-tested for legacy stability."""
    if method_key == "fullload":
        return server.request(
            context, mode="fullload", cold=False, suffix_ids=suffix_ids)
    if method_key == "qa_token25":
        return server.request_qa_select(
            context, cold=False, suffix_ids=suffix_ids)
    if method_key == "qa_chunk25":
        return server.request_qa_chunk(
            context, cold=False, suffix_ids=suffix_ids)
    if method_key == "ours25":
        return server.request_cvpr25(
            context, static=None, budget=0.25, mode="prefix",
            sep_policy="sidecar", cold=False, image_id=image_id,
            suffix_ids=suffix_ids,
            expected_prefix_layout="visionzip_image_only")
    raise ValueError(f"not a stored method: {method_key}")


def _json_result(result: Mapping[str, Any], method_key: str,
                 full_visual_bytes: int) -> dict[str, Any]:
    value = _base_module()._json_result(
        dict(result), method_key, full_visual_bytes)
    for field in (
        "chunk_aggregation_ms", "topk_chunk_ms",
        "selector_decision_host_wall_ms", "max_contiguous_run_length",
    ):
        value.setdefault(field, 0.0)
    if method_key == "qa_chunk25":
        value["topk_chunk_ms"] = float(
            value.get("topk_chunk_ms", value.get("topk_ms", 0.0)) or 0.0)
        value["topk_ms"] = value["topk_chunk_ms"]
    value["selected_chunk_payload_read_bytes"] = int(
        value.get("normal_kv_read_bytes", 0))
    return value


def _run_stored(runner, server, context, question, method_key, image_id,
                full_visual_bytes):
    base = _base_module()
    with base._NoVisionForward(runner) as guard:
        conditioned_at = time.perf_counter()
        context.reader.drop_all()
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
        suffix_cpu = base._suffix_from_tokenized(runner, tokenized)
        prepare_ms = (time.perf_counter() - started) * 1e3
        started = time.perf_counter()
        suffix_device = suffix_cpu.to(runner.model.device)
        torch.cuda.synchronize()
        h2d_ms = (time.perf_counter() - started) * 1e3
        result = dispatch_stored(
            server, method_key, context, suffix_device, image_id)
        returned = time.perf_counter()
    phases = {
        "prompt_build_ms": float(prompt_ms),
        "tokenization_ms": float(token_ms), "image_preprocess_ms": 0.0,
        "input_prepare_ms": float(prepare_ms), "input_h2d_ms": float(h2d_ms),
        "processor_total_ms": None,
    }
    result.update(base._timing_fields(
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
    diagnostics = {
        "prompt": prompt,
        "prompt_sha256": base._sha_bytes(prompt.encode("utf-8")),
        "suffix_ids_sha256": base._hash_tensor(suffix_cpu),
    }
    return _json_result(result, method_key, full_visual_bytes), diagnostics


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _max_run_length(row: Mapping[str, Any]) -> int:
    maximum = 0
    for chunks in row.get("selected_chunk_ids_per_layer") or []:
        _, lengths = contiguous_runs(chunks)
        maximum = max([maximum, *lengths])
    return maximum


def summaries_from_rows(rows: Sequence[Mapping[str, Any]],
                        full_visual_bytes_mean: float) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for method_key in METHOD_KEYS:
        all_rows = [row for row in rows if row["method_key"] == method_key]
        hits = [row for row in all_rows if int(row["turn_id"]) > 1]
        source = hits or all_rows
        ttft = [float(row["end_to_end_ttft_ms"]) for row in source]
        output[method_key] = {
            **METHODS[method_key], "n_requests": len(all_rows),
            "n_cache_hit_requests": len(hits),
            "accuracy_all_turns": _mean(all_rows, "correct"),
            "accuracy_cache_hits": _mean(hits, "correct"),
            "ttft_cache_hit_mean_ms": float(np.mean(ttft)),
            "ttft_cache_hit_p50_ms": float(np.percentile(ttft, 50)),
            "ttft_cache_hit_p95_ms": float(np.percentile(ttft, 95)),
            "turn1_ttft_mean_ms": _mean(
                [row for row in all_rows if int(row["turn_id"]) == 1],
                "end_to_end_ttft_ms"),
            "actual_ssd_mb_per_cache_hit": _mean(source, "actual_ssd_mb"),
            "actual_ssd_ratio_vs_fullload": _mean(
                source, "actual_ssd_ratio_vs_fullload"),
            "normal_selected_chunk_ratio": (
                _mean(source, "normal_selected_chunk_ratio")
                if _mean(source, "normal_selected_chunk_ratio") is not None
                else _mean(source, "touched_chunk_fraction")),
            "total_touched_chunk_ratio": (
                _mean(source, "total_touched_chunk_ratio")
                if _mean(source, "total_touched_chunk_ratio") is not None
                else _mean(source, "touched_chunk_fraction")),
            "touched_chunk_fraction": _mean(
                source, "touched_chunk_fraction"),
            "selected_chunk_payload_mb": (
                (_mean(source, "normal_kv_read_bytes") or 0.0) / 1e6),
            "probe_io_mb": (_mean(source, "probe_read_bytes") or 0.0) / 1e6,
            "separator_io_mb": (
                (_mean(source, "separator_read_bytes") or 0.0) / 1e6),
            "ssd_preads_per_cache_hit": _mean(source, "ssd_preads"),
            "ssd_read_latency_ms": _mean(source, "ssd_read_ms"),
            "contiguous_runs_per_layer": _mean(
                source, "contiguous_runs_per_layer_mean"),
            "mean_contiguous_run_length": _mean(
                source, "mean_contiguous_run_length"),
            "max_contiguous_run_length": (
                float(max((_max_run_length(row) for row in source), default=0))
                if source else None),
            "query_score_calls_total": int(sum(
                int(row.get("query_score_calls", 0)) for row in hits)),
            "chunk_score_calls_total": int(sum(
                int(row.get("chunk_score_calls", 0)) for row in hits)),
            "selector_ms": _mean(source, "selector_ms"),
            "online_selector_total_ms": _mean(
                source, "online_selector_total_ms"),
            "selector_decision_host_wall_ms": _mean(
                source, "selector_decision_host_wall_ms"),
            "rater_selection_ms": _mean(source, "rater_selection_ms"),
            "query_projection_ms": _mean(source, "query_projection_ms"),
            "probe_h2d_ms": _mean(source, "probe_h2d_ms"),
            "probe_io_ms": _mean(source, "probe_io_ms"),
            "normal_kv_read_ms": _mean(source, "normal_kv_read_ms"),
            "selected_chunk_io_ms": _mean(source, "selected_chunk_io_ms"),
            "separator_read_ms": _mean(source, "separator_read_ms"),
            "query_scoring_ms": _mean(source, "query_scoring_ms"),
            "chunk_aggregation_ms": _mean(source, "chunk_aggregation_ms"),
            "topk_chunk_ms": _mean(source, "topk_chunk_ms"),
            "selected_id_d2h_ms": _mean(source, "selected_id_d2h_ms"),
            "chunk_planning_ms": _mean(source, "chunk_planning_ms"),
            "chunk_io_ms": _mean(source, "chunk_io_ms"),
            "scatter_ms": _mean(source, "scatter_ms"),
            "prefill_ms": _mean(source, "prefill_ms"),
            "full_visual_kv_mb_mean": full_visual_bytes_mean / 1e6,
        }
    return output


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def _layer_jaccards(left: Sequence[Sequence[int]],
                     right: Sequence[Sequence[int]]) -> list[float]:
    if len(left) != len(right):
        raise ValueError("selection layer count mismatch")
    return [_jaccard(set(map(int, a)), set(map(int, b)))
            for a, b in zip(left, right)]


def selection_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    chunk_rows = [row for row in rows if row["method_key"] == "qa_chunk25"
                  and int(row["turn_id"]) > 1]
    token_rows = {(row["image_id"], row["question_id"]): row for row in rows
                  if row["method_key"] == "qa_token25"
                  and int(row["turn_id"]) > 1}
    by_image: dict[str, list[Mapping[str, Any]]] = {}
    requests = []
    overlaps = []
    for row in chunk_rows:
        by_image.setdefault(str(row["image_id"]), []).append(row)
        selected = row.get("selected_chunk_ids_per_layer") or []
        requests.append({
            "image_id": row["image_id"], "question_id": row["question_id"],
            "turn_id": int(row["turn_id"]),
            "selected_chunk_ids_per_layer": selected,
            "selected_chunks_sha256": canonical_hash(selected),
        })
        token = token_rows[(row["image_id"], row["question_id"])]
        touched = token.get("selected_chunk_ids_per_layer") or []
        per_layer = _layer_jaccards(touched, selected)
        intersection = [len(set(map(int, a)) & set(map(int, b)))
                        for a, b in zip(touched, selected)]
        union = [len(set(map(int, a)) | set(map(int, b)))
                 for a, b in zip(touched, selected)]
        overlaps.append({
            "image_id": row["image_id"], "question_id": row["question_id"],
            "qa_token_touched_chunks_per_layer": [len(value) for value in touched],
            "qa_chunk_selected_chunks_per_layer": [len(value) for value in selected],
            "intersection_per_layer": intersection, "union_per_layer": union,
            "jaccard_per_layer": per_layer,
            "mean_layer_jaccard": float(np.mean(per_layer)),
        })
    pairs = []
    for image_id, image_rows in sorted(by_image.items()):
        image_rows.sort(key=lambda row: int(row["turn_id"]))
        for left, right in combinations(image_rows, 2):
            left_ids = left.get("selected_chunk_ids_per_layer") or []
            right_ids = right.get("selected_chunk_ids_per_layer") or []
            layer = _layer_jaccards(left_ids, right_ids)
            identical = all(value == 1.0 for value in layer)
            pairs.append({
                "image_id": image_id,
                "left_turn": int(left["turn_id"]),
                "right_turn": int(right["turn_id"]),
                "left_question_id": left["question_id"],
                "right_question_id": right["question_id"],
                "chunk_jaccard": float(np.mean(layer)),
                "chunk_jaccard_per_layer": layer,
                "identical": identical,
                "consecutive": int(right["turn_id"]) == int(left["turn_id"]) + 1,
            })
    pair_values = [row["chunk_jaccard"] for row in pairs]
    consecutive = [row["chunk_jaccard"] for row in pairs if row["consecutive"]]
    return {
        "scope": "cache-hit turns 2..6",
        "n_query_requests": len(chunk_rows), "n_images": len(by_image),
        "requests": sorted(requests, key=lambda row: (
            row["image_id"], row["turn_id"], row["question_id"])),
        "pairs": pairs, "n_pairs": len(pairs),
        "mean_pairwise_chunk_jaccard": (
            float(np.mean(pair_values)) if pair_values else None),
        "mean_consecutive_chunk_jaccard": (
            float(np.mean(consecutive)) if consecutive else None),
        "identical_chunk_selection_rate": (
            float(np.mean([row["identical"] for row in pairs]))
            if pairs else None),
        "different_selection_pairs": int(sum(
            not row["identical"] for row in pairs)),
        "n_consecutive_pairs": len(consecutive),
        "qa_token_overlap": {
            "requests": overlaps, "n_requests": len(overlaps),
            "mean_layer_jaccard": (
                float(np.mean([row["mean_layer_jaccard"] for row in overlaps]))
                if overlaps else None),
            "mean_qa_token_touched_chunks_per_layer": (
                float(np.mean([value for row in overlaps for value in
                               row["qa_token_touched_chunks_per_layer"]]))
                if overlaps else None),
            "mean_qa_chunk_selected_chunks_per_layer": (
                float(np.mean([value for row in overlaps for value in
                               row["qa_chunk_selected_chunks_per_layer"]]))
                if overlaps else None),
            "mean_intersection_chunks_per_layer": (
                float(np.mean([value for row in overlaps for value in
                               row["intersection_per_layer"]]))
                if overlaps else None),
        },
    }


def reference_consistency(rows: Sequence[Mapping[str, Any]],
                          reference: Mapping[str, Any]) -> dict[str, Any]:
    prior = {(str(row["image_id"]), str(row["question_id"]),
              str(row["method_key"])): row for row in reference["rows"]}
    result: dict[str, Any] = {
        "run_dir": reference["run_dir"], "hashes": reference["hashes"],
        "methods": {},
    }
    for method, prior_method in REFERENCE_METHOD.items():
        compared = equal = 0
        current_correct: list[float] = []
        prior_correct: list[float] = []
        deterministic_equal = 0
        cache_compared = 0
        for row in rows:
            if row["method_key"] != method:
                continue
            # Match the validated predecessor's equivalence population:
            # ReComp has 240 comparable pixel requests; stored arms compare
            # cache-hit turns 2..6 because Turn 1 is deliberately a fresh,
            # separately timed normal-pixel request in each experiment.
            if method != "recompute" and int(row["turn_id"]) == 1:
                continue
            old = prior.get((str(row["image_id"]), str(row["question_id"]),
                             prior_method))
            if old is None:
                continue
            compared += 1
            equal += row["prediction"] == old["prediction"]
            current_correct.append(float(row["correct"]))
            prior_correct.append(float(old["correct"]))
            if int(row["turn_id"]) > 1:
                cache_compared += 1
                fields = ("ssd_read_bytes", "ssd_preads",
                          "touched_chunk_fraction",
                          "selected_chunk_ids_per_layer")
                deterministic_equal += all(row.get(key) == old.get(key)
                                           for key in fields)
        current_accuracy = float(np.mean(current_correct))
        prior_accuracy = float(np.mean(prior_correct))
        current_hits = [row for row in rows if row["method_key"] == method
                        and int(row["turn_id"]) > 1]
        prior_summary = reference["summary"]["per_method"][prior_method]
        current_ttft = _mean(current_hits, "end_to_end_ttft_ms")
        current_selector = _mean(current_hits, "selector_ms") or 0.0
        result["methods"][method] = {
            "compared": compared, "equal_predictions": equal,
            "agreement": equal / compared if compared else None,
            "current_accuracy": current_accuracy,
            "reference_accuracy": prior_accuracy,
            "accuracy_gap_pp": (current_accuracy - prior_accuracy) * 100.0,
            "cache_hit_deterministic_io_compared": cache_compared,
            "cache_hit_deterministic_io_equal": deterministic_equal,
            "current_ttft_ms": current_ttft,
            "reference_ttft_ms": prior_summary["ttft_cache_hit_mean_ms"],
            "ttft_relative_delta": (
                current_ttft / prior_summary["ttft_cache_hit_mean_ms"] - 1.0),
            "current_selector_ms": current_selector,
            "reference_selector_ms": prior_summary["selector_ms"],
            "ttft_tolerance_ms": max(
                TTFT_ABSOLUTE_TOLERANCE_MS,
                TTFT_RELATIVE_TOLERANCE
                * float(prior_summary["ttft_cache_hit_mean_ms"])),
            "ttft_within_fixed_tolerance": abs(
                float(current_ttft)
                - float(prior_summary["ttft_cache_hit_mean_ms"])) <= max(
                    TTFT_ABSOLUTE_TOLERANCE_MS,
                    TTFT_RELATIVE_TOLERANCE
                    * float(prior_summary["ttft_cache_hit_mean_ms"])),
        }
        if method == "qa_token25":
            reference_selector = float(prior_summary["selector_ms"])
            selector_tolerance = max(
                QA_TOKEN_SELECTOR_ABSOLUTE_TOLERANCE_MS,
                QA_TOKEN_SELECTOR_RELATIVE_TOLERANCE * reference_selector)
            result["methods"][method].update({
                "selector_tolerance_ms": selector_tolerance,
                "selector_within_fixed_tolerance": abs(
                    current_selector - reference_selector)
                <= selector_tolerance,
            })
    return result


def _reference_prediction_accuracy_passes(reference: Mapping[str, Any]) -> bool:
    methods = reference["methods"]
    for method in REFERENCE_METHOD:
        item = methods[method]
        compared = int(item["compared"])
        if compared <= 0:
            return False
        if method == "recompute":
            if int(item["equal_predictions"]) != compared:
                return False
            if abs(float(item["accuracy_gap_pp"])) > 1e-12:
                return False
            continue
        tolerance = max(1, math.ceil(REFERENCE_EQUIVALENCE_RATE * compared))
        if compared - int(item["equal_predictions"]) > tolerance:
            return False
        if abs(float(item["accuracy_gap_pp"])) > 100.0 * tolerance / compared + 1e-9:
            return False
    return True


def _reference_io_passes(reference: Mapping[str, Any]) -> bool:
    return all(
        int(reference["methods"][method][
            "cache_hit_deterministic_io_compared"]) > 0
        and int(reference["methods"][method]["cache_hit_deterministic_io_equal"])
        == int(reference["methods"][method][
            "cache_hit_deterministic_io_compared"])
        for method in ("fullload", "qa_token25", "ours25"))


def _reference_timing_passes(reference: Mapping[str, Any]) -> bool:
    return all(bool(reference["methods"][method][
        "ttft_within_fixed_tolerance"]) for method in REFERENCE_METHOD)


def _reference_selector_passes(reference: Mapping[str, Any]) -> bool:
    return bool(reference["methods"]["qa_token25"][
        "selector_within_fixed_tolerance"])


def _expected_normal_bytes(row: Mapping[str, Any], meta: Mapping[str, Any]) -> int:
    row_bytes = int(meta["num_heads"]) * int(meta["head_dim"]) * 2
    selected_rows = 0
    for chunks in row.get("selected_chunk_ids_per_layer") or []:
        for chunk in chunks:
            start = int(chunk) * int(meta["chunk_size"])
            selected_rows += max(0, min(
                int(meta["chunk_size"]), int(meta["v_token_num"]) - start))
    return selected_rows * row_bytes * 2


def validate_run(rows: Sequence[Mapping[str, Any]], schedule: Sequence[Mapping],
                 selection: Mapping[str, Any], metas: Mapping[str, Mapping],
                 source_unchanged: bool, reference: Mapping[str, Any],
                 workload: Mapping[str, Any], failure_rows: Sequence[Mapping]) -> dict:
    expected_ids = [row["request_id"] for row in schedule]
    observed_ids = [str(row["request_id"]) for row in rows]
    checks: dict[str, bool] = {}
    checks["expected_completed_count"] = len(rows) == len(schedule)
    checks["request_identity_exact"] = set(observed_ids) == set(expected_ids)
    checks["duplicates_zero"] = len(observed_ids) == len(set(observed_ids))
    checks["unresolved_failures_zero"] = not failure_rows
    checks["five_methods_present"] = set(
        row["method_key"] for row in rows) == set(METHOD_KEYS)
    checks["stable_method_metadata"] = all(all(
        row.get(key) == value for key, value in METHODS[row["method_key"]].items())
        for row in rows)
    expected_per_method = int(workload["selected_questions"])
    checks["per_method_counts"] = all(sum(
        row["method_key"] == method for row in rows) == expected_per_method
        for method in METHOD_KEYS)
    checks["all_ttft_finite_positive"] = all(
        math.isfinite(float(row["end_to_end_ttft_ms"]))
        and float(row["end_to_end_ttft_ms"]) > 0 for row in rows)
    turn1_ok = True
    for image_id in {str(row["image_id"]) for row in rows}:
        group = [row for row in rows if str(row["image_id"]) == image_id
                 and int(row["turn_id"]) == 1]
        turn1_ok &= len(group) == len(METHOD_KEYS)
        for field in ("prompt_sha256", "input_tensors_sha256", "prediction",
                      "first_token_id"):
            turn1_ok &= len({json.dumps(row.get(field), sort_keys=True)
                             for row in group}) == 1
        turn1_ok &= all(row.get("request_path") == "normal_pixel_turn1"
                        and int(row.get("vision_forward_count", -1)) == 1
                        for row in group)
    checks["turn1_five_arm_normal_pixel_agreement"] = bool(turn1_ok)
    hits = [row for row in rows if int(row["turn_id"]) > 1]
    groups = {method: [row for row in hits if row["method_key"] == method]
              for method in METHOD_KEYS}
    chunk = groups["qa_chunk25"]
    token = groups["qa_token25"]
    full = groups["fullload"]
    ours = groups["ours25"]
    recomp = groups["recompute"]
    checks["stored_hits_no_vision"] = all(
        int(row.get("vision_forward_count", -1)) == 0
        for row in chunk + token + full + ours)
    checks["qa_chunk_query_scoring_called"] = bool(chunk) and all(
        int(row.get("query_score_calls", 0))
        == int(row.get("expected_layers", -1)) > 0 for row in chunk)
    checks["qa_chunk_chunk_scores_created"] = bool(chunk) and all(
        int(row.get("chunk_score_calls", 0))
        == int(row.get("expected_layers", -1)) > 0 for row in chunk)
    checks["qa_chunk_fixed_budget"] = bool(chunk) and all(
        all(len(layer) == budget_chunk_count(
            int(metas[str(row["image_id"])]["raster"]["n_chunks_per_layer"]),
            0.25) for layer in row["selected_chunk_ids_per_layer"])
        for row in chunk)
    checks["qa_chunk_actual_loaded_equals_selected"] = all(
        row.get("actual_loaded_chunk_ids_per_layer")
        == row.get("selected_chunk_ids_per_layer") for row in chunk)
    checks["qa_chunk_no_fallback_or_adaptive"] = all(
        float(row.get("fallback_rate", -1)) == 0.0
        and int(row.get("full_load_fallback_count", -1)) == 0
        and int(row.get("static_score_calls", -1)) == 0
        and int(row.get("diversity_calls", -1)) == 0
        and row.get("adaptive_ratio") is False for row in chunk)
    checks["qa_chunk_mean_spatial_aggregation"] = all(
        row.get("chunk_score_aggregation")
        == "mean_valid_spatial_token_importance"
        and row.get("selection_granularity") == "ssd_chunk"
        for row in chunk)
    checks["qa_chunk_raster_no_repack"] = all(
        row.get("physical_layout") == "raster"
        and row.get("repacking") is False for row in chunk)
    checks["qa_chunk_same_normal_budget_as_ours"] = all(
        [len(layer) for layer in chunk_row["selected_chunk_ids_per_layer"]]
        == [len(layer) for layer in next(
            row for row in ours if row["image_id"] == chunk_row["image_id"]
            and row["question_id"] == chunk_row["question_id"]
        )["selected_chunk_ids_per_layer"]]
        for chunk_row in chunk)
    checks["qa_token_unchanged_contract"] = bool(token) and all(
        int(row.get("query_score_calls", 0))
        == int(row.get("expected_layers", -1)) > 0
        and math.isclose(float(row.get("logical_selected_token_ratio", -1)),
                         0.25, rel_tol=0.0, abs_tol=0.002)
        for row in token)
    checks["ours_unchanged_prefix_contract"] = bool(ours) and all(
        int(row.get("query_score_calls", 0)) == 0
        and all(layer == list(range(len(layer)))
                for layer in row["selected_chunk_ids_per_layer"])
        for row in ours)
    checks["fullload_unchanged_contract"] = bool(full) and all(
        int(row.get("query_score_calls", 0)) == 0
        and math.isclose(float(row["actual_ssd_ratio_vs_fullload"]), 1.0,
                         rel_tol=0.0, abs_tol=1e-12)
        for row in full)
    checks["recompute_unchanged_zero_ssd"] = bool(recomp) and all(
        int(row["ssd_read_bytes"]) == 0
        and int(row.get("query_score_calls", 0)) == 0 for row in recomp)
    checks["ssd_bytes_include_all_kinds"] = all(
        int(row["ssd_read_bytes"]) == int(row["normal_kv_read_bytes"])
        + int(row["probe_read_bytes"]) + int(row["separator_read_bytes"])
        for row in chunk + token + full + ours)
    checks["qa_chunk_independent_byte_accounting"] = all(
        int(row["normal_kv_read_bytes"]) == _expected_normal_bytes(
            row, metas[str(row["image_id"])]["raster"])
        and int(row["probe_read_bytes"]) == int(
            metas[str(row["image_id"])]["raster"]["bytes_probe_sidecar"])
        and int(row["separator_read_bytes"]) == int(
            metas[str(row["image_id"])]["raster"]["bytes_separator_sidecar"])
        for row in chunk)
    checks["qa_chunk_independent_pread_accounting"] = all(
        int(row["ssd_preads"]) == 1
        + int(row["expected_layers"])
        + 2 * sum(int(value) for value in
                  row["contiguous_runs_per_layer"])
        for row in chunk)
    checks["qa_chunk_selector_inside_ttft"] = all(
        float(row["core_ttft_ms"]) + 1e-3 >= float(row["prefill_ms"])
        and float(row.get("rater_selection_ms", 0)) > 0
        and float(row.get("query_scoring_ms", 0)) > 0
        and float(row.get("chunk_aggregation_ms", 0)) > 0
        for row in chunk)
    checks["causal_prompt_current_question_only"] = all(
        row.get("prompt_sha256") == row.get("expected_prompt_sha256")
        and row.get("causal_question_ids") == [str(row["question_id"])]
        and row.get("past_history_question_ids") == []
        and row.get("future_question_ids_used") == []
        and int(row.get("future_questions_in_prompt", -1)) == 0
        for row in rows)
    checks["qa_chunk_query_dependence_observed"] = (
        int(selection["different_selection_pairs"]) > 0)
    checks["qa_token_overlap_complete"] = (
        int(selection["qa_token_overlap"]["n_requests"]) == len(chunk))
    checks["canonical_source_store_unchanged"] = bool(source_unchanged)
    checks["canonical_reference_prediction_accuracy_equivalence"] = (
        _reference_prediction_accuracy_passes(reference))
    checks["canonical_reference_exact_ssd_and_selection_equivalence"] = (
        _reference_io_passes(reference))
    checks["canonical_reference_ttft_fixed_tolerance"] = (
        _reference_timing_passes(reference))
    checks["qa_token_selector_fixed_tolerance"] = (
        _reference_selector_passes(reference))
    checks["balanced_five_way_rotation"] = all(
        len({row["method_order_position"] for row in rows
             if row["method_key"] == method}) == len(METHOD_KEYS)
        for method in METHOD_KEYS) if workload["selected_images"] >= 5 else True
    return {
        "schema_version": SCHEMA_VERSION, "passed": all(checks.values()),
        "expected": len(schedule), "completed": len(rows),
        "failed": len(failure_rows),
        "duplicates": len(observed_ids) - len(set(observed_ids)),
        "checks": checks, "selection": {
            key: value for key, value in selection.items()
            if key not in {"requests", "pairs"}},
        "reference_consistency": reference,
        "limitations": [
            "GQA questions are independent cache-hit turns; no conversation history exists.",
            "Turn 1 is normal pixel inference and has no query-aware chunk selection.",
            "The validated canonical raster/image-only stores are reused read-only; store build cost is not rerun.",
            "v_hidden is opened before request TTFT; per-request H2D/rater work remains inside TTFT.",
            "Buffered pread plus POSIX_FADV_DONTNEED cannot flush an SSD controller cache.",
            "Component timings can overlap; TTFT and selector wall fields are authoritative, not their arithmetic sum.",
        ],
    }


def _comparison(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    chunk = summaries["qa_chunk25"]
    token = summaries["qa_token25"]
    ours = summaries["ours25"]
    return {
        "qa_chunk_minus_ours_accuracy_pp": (
            chunk["accuracy_all_turns"] - ours["accuracy_all_turns"]) * 100,
        "qa_chunk_minus_ours_ttft_ms": (
            chunk["ttft_cache_hit_mean_ms"] - ours["ttft_cache_hit_mean_ms"]),
        "qa_chunk_over_ours_ttft_ratio": (
            chunk["ttft_cache_hit_mean_ms"] / ours["ttft_cache_hit_mean_ms"]),
        "qa_token_minus_chunk_accuracy_pp": (
            token["accuracy_all_turns"] - chunk["accuracy_all_turns"]) * 100,
        "qa_chunk_minus_token_ssd_mb": (
            chunk["actual_ssd_mb_per_cache_hit"]
            - token["actual_ssd_mb_per_cache_hit"]),
        "qa_chunk_ssd_reduction_vs_token_percent": (
            100.0 * (1.0 - chunk["actual_ssd_mb_per_cache_hit"]
                     / token["actual_ssd_mb_per_cache_hit"])),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (json.dumps(value, ensure_ascii=False)
                                    if isinstance(value, (dict, list)) else value)
                             for key, value in row.items()})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_dir(path.parent)


def _analysis_markdown(config, summaries, comparison, selection, validation):
    def pct(value):
        return "—" if value is None else f"{100 * float(value):.4f}%"

    def number(value, digits=4):
        return "—" if value is None else f"{float(value):.{digits}f}"

    lines = [
        "# QA-Chunk25 GQA pilot", "",
        ("Accuracy는 전체 질문, TTFT/I/O는 cache-hit turns 2–6의 request "
         "mean이다. MB는 10^6 bytes다."), "",
        "| Method | Accuracy | TTFT | Query-aware | Physical Chunk Budget | SSD MB | SSD Ratio | Selector ms | Touched Chunks | Preads |",
        "|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in METHOD_KEYS:
        row = summaries[key]
        lines.append(
            f"| {row['display_label']} | {pct(row['accuracy_all_turns'])} | "
            f"{number(row['ttft_cache_hit_mean_ms'])} | "
            f"{'Yes' if row['query_dependent'] else 'No'} | "
            f"{pct(row['touched_chunk_fraction'])} | "
            f"{number(row['actual_ssd_mb_per_cache_hit'])} | "
            f"{pct(row['actual_ssd_ratio_vs_fullload'])} | "
            f"{number(row['online_selector_total_ms'])} | "
            f"{pct(row['touched_chunk_fraction'])} | "
            f"{number(row['ssd_preads_per_cache_hit'])} |")
    chunk = summaries["qa_chunk25"]
    overlap = selection["qa_token_overlap"]
    lines.extend([
        "", "## QA-Chunk25 definition", "",
        "SparseVLM raters와 probe Q/K token importance는 QA-Token25와 동일하다. Separator를 제외한 valid spatial token score를 canonical raster SSD chunk별 mean으로 집계하고, `round(0.25 × n_chunks)`와 동일한 Ours budget helper로 상위 chunk를 직접 선택한다. Probe/selected K/V/separator I/O와 모든 online selection 단계는 TTFT critical path에 포함한다.",
        "", "## Query dependence and overlap", "",
        f"- Query scoring calls: {chunk['query_score_calls_total']}",
        f"- Selected normal chunk ratio: {pct(chunk['normal_selected_chunk_ratio'])}",
        f"- Pairwise chunk Jaccard: {number(selection['mean_pairwise_chunk_jaccard'], 6)}",
        f"- Consecutive-query Jaccard: {number(selection['mean_consecutive_chunk_jaccard'], 6)}",
        f"- Identical selection rate: {pct(selection['identical_chunk_selection_rate'])}",
        f"- QA-Token touched vs QA-Chunk selected Jaccard: {number(overlap['mean_layer_jaccard'], 6)}",
        "", "## QA-Chunk25 latency and I/O", "",
        f"- Selector wall/alias: {number(chunk['online_selector_total_ms'])} ms",
        f"- Rater/projection/probe-I/O/scoring/aggregation/top-k: {number(chunk['rater_selection_ms'])} / {number(chunk['query_projection_ms'])} / {number(chunk['probe_io_ms'])} / {number(chunk['query_scoring_ms'])} / {number(chunk['chunk_aggregation_ms'])} / {number(chunk['topk_chunk_ms'])} ms",
        f"- ID-D2H/planning/chunk-I/O/scatter/prefill: {number(chunk['selected_id_d2h_ms'])} / {number(chunk['chunk_planning_ms'])} / {number(chunk['chunk_io_ms'])} / {number(chunk['scatter_ms'])} / {number(chunk['prefill_ms'])} ms",
        f"- Probe/selected payload/total SSD: {number(chunk['probe_io_mb'])} / {number(chunk['selected_chunk_payload_mb'])} / {number(chunk['actual_ssd_mb_per_cache_hit'])} MB",
        f"- Runs/layer, mean/max run, preads: {number(chunk['contiguous_runs_per_layer'])}, {number(chunk['mean_contiguous_run_length'])}, {number(chunk['max_contiguous_run_length'])}, {number(chunk['ssd_preads_per_cache_hit'])}",
        "", "## Direct answers", "",
        f"- QA-Token25 대비 SSD 감소: {comparison['qa_chunk_ssd_reduction_vs_token_percent']:.4f}%",
        f"- QA-Token25 − QA-Chunk25 accuracy: {comparison['qa_token_minus_chunk_accuracy_pp']:+.4f} pp",
        f"- QA-Chunk25 − Ours25 accuracy: {comparison['qa_chunk_minus_ours_accuracy_pp']:+.4f} pp",
        f"- QA-Chunk25 − Ours25 TTFT: {comparison['qa_chunk_minus_ours_ttft_ms']:+.4f} ms ({comparison['qa_chunk_over_ours_ttft_ratio']:.4f}×)",
        "", "## Completion", "",
        f"- Expected/completed/failed/duplicates: {validation['expected']}/{validation['completed']}/{validation['failed']}/{validation['duplicates']}",
        f"- Workload SHA256: `{config['selected_workload_sha256']}`",
        f"- Canonical source store: `{config['store_dir']}` (read-only reuse)",
        "", "## Limitations", "",
    ])
    lines.extend(f"- {item}" for item in validation["limitations"])
    lines.extend([
        "", "QA-CHUNK25 BASELINE VALIDATED: "
        + ("YES" if validation["passed"] else "NO"), "",
    ])
    return "\n".join(lines)


def _progress(run_id: str, expected: int, completed: set[str],
              failure_rows: Sequence[Mapping], status: str,
              last_request_id: str | None = None,
              repaired_tail_bytes: int = 0) -> dict[str, Any]:
    unresolved = {str(row["request_id"]) for row in failure_rows} - completed
    return {
        "schema_version": SCHEMA_VERSION, "run_id": run_id,
        "status": status, "expected": expected, "completed": len(completed),
        "remaining": expected - len(completed), "failed": len(unresolved),
        "failure_events": len(failure_rows), "duplicates": 0,
        "last_request_id": last_request_id,
        "repaired_truncated_tail_bytes": repaired_tail_bytes,
        "updated_at_unix": time.time(), "pid": os.getpid(),
    }


def _finalize(run_dir: Path, results_dir: Path, config: dict,
              schedule: Sequence[Mapping], workload: Mapping[str, Any],
              metas: Mapping[str, Mapping], inventory_before: Mapping[str, Any],
              reference_source: Mapping[str, Any]) -> dict[str, Any]:
    partial = run_dir / "results_partial.jsonl"
    rows, repaired = read_jsonl_unique(partial)
    if repaired:
        raise AssertionError("finalization unexpectedly repaired a JSONL tail")
    failure_path = run_dir / "failures.jsonl"
    failure_rows = ([json.loads(line) for line in failure_path.read_text(
        encoding="utf-8").splitlines()] if failure_path.exists() else [])
    # A failure event that was later successfully retried is retained as
    # evidence but is no longer an unresolved failure.
    completed = {row["request_id"] for row in rows}
    unresolved_failures = [row for row in failure_rows
                           if row["request_id"] not in completed]
    inventory_after = source_store_inventory(
        Path(config["store_dir"]), metas.keys())
    source_unchanged = (
        inventory_before["inventory_sha256"]
        == inventory_after["inventory_sha256"])
    full_sizes = [int(value["raster"]["bytes_visual_kv"])
                  for value in metas.values()]
    summaries = summaries_from_rows(rows, float(np.mean(full_sizes)))
    selection = selection_analysis(rows)
    reference = reference_consistency(rows, reference_source)
    comparison = _comparison(summaries)
    validation = validate_run(
        rows, schedule, selection, metas, source_unchanged, reference,
        workload, unresolved_failures)
    config.update({
        "status": "complete" if validation["passed"] else "failed_validation",
        "finished_at_unix": time.time(),
        "source_store_inventory_after_sha256": inventory_after[
            "inventory_sha256"],
    })
    atomic_json(run_dir / "config.json", config)
    summary = {
        "schema_version": SCHEMA_VERSION, "config": config,
        "per_method": summaries, "comparison": comparison,
        "selection": {key: value for key, value in selection.items()
                      if key not in {"requests", "pairs"}},
        "reference_consistency": reference,
    }
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(run_dir / "selection_analysis.json", selection)
    atomic_json(run_dir / "validation.json", validation)
    atomic_bytes(run_dir / "results_final.jsonl", partial.read_bytes())
    _write_csv(run_dir / "per_request.csv", rows)
    _write_csv(run_dir / "summary.csv", [
        {"method_key": key, **summaries[key]} for key in METHOD_KEYS])
    latency_fields = (
        "rater_selection_ms", "query_projection_ms", "probe_h2d_ms",
        "probe_io_ms", "normal_kv_read_ms", "selected_chunk_io_ms",
        "separator_read_ms",
        "query_scoring_ms", "chunk_aggregation_ms",
        "topk_chunk_ms", "selected_id_d2h_ms", "chunk_planning_ms",
        "chunk_io_ms", "scatter_ms", "prefill_ms",
        "online_selector_total_ms", "ttft_cache_hit_mean_ms",
    )
    _write_csv(run_dir / "latency_breakdown.csv", [{
        "method_key": key, **{field: summaries[key].get(field)
                              for field in latency_fields}}
        for key in METHOD_KEYS])
    io_fields = (
        "normal_selected_chunk_ratio", "total_touched_chunk_ratio",
        "probe_io_mb", "selected_chunk_payload_mb", "separator_io_mb",
        "actual_ssd_mb_per_cache_hit", "actual_ssd_ratio_vs_fullload",
        "ssd_preads_per_cache_hit", "ssd_read_latency_ms",
        "contiguous_runs_per_layer", "mean_contiguous_run_length",
        "max_contiguous_run_length",
    )
    _write_csv(run_dir / "io_breakdown.csv", [{
        "method_key": key, **{field: summaries[key].get(field)
                              for field in io_fields}}
        for key in METHOD_KEYS])
    analysis = _analysis_markdown(
        config, summaries, comparison, selection, validation)
    atomic_text(run_dir / "RUN_ANALYSIS.md", analysis)
    exported = (
        "results_final.jsonl", "summary.json", "summary.csv",
        "selection_analysis.json", "latency_breakdown.csv",
        "io_breakdown.csv", "validation.json", "RUN_ANALYSIS.md",
    )
    for name in exported:
        atomic_bytes(results_dir / name, (run_dir / name).read_bytes())
    artifacts = {
        "schema_version": SCHEMA_VERSION, "run_dir": str(run_dir),
        "results_dir": str(results_dir), "store_dir": config["store_dir"],
        "files_sha256": {name: sha256_file(run_dir / name) for name in (
            "manifest.json", "config.json", "results_partial.jsonl",
            *exported)},
    }
    atomic_json(run_dir / "run_artifacts.json", artifacts)
    atomic_json(results_dir / "run_artifacts.json", artifacts)
    completed_payload = {
        "schema_version": SCHEMA_VERSION, "run_id": config["run_id"],
        "passed": validation["passed"], "completed": len(rows),
        "validation_sha256": sha256_file(run_dir / "validation.json"),
        "finished_at_unix": config["finished_at_unix"],
    }
    return {"summary": summary, "validation": validation,
            "selection": selection, "comparison": comparison,
            "completed_payload": completed_payload}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-dir", type=Path,
                      help="new or launcher-prepared unique run directory")
    mode.add_argument("--resume", type=Path,
                      help="resume an incomplete run directory")
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--index", type=Path, default=ROOT / "data/index.json")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument("--reference-run-dir", type=Path,
                        default=DEFAULT_REFERENCE_RUN_DIR)
    parser.add_argument("--max-images", type=int, choices=(3, 4, 5, 40),
                        default=5)
    parser.add_argument("--skip", type=int, default=4)
    parser.add_argument("--questions", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--max-new-requests", type=int,
        help=("invocation-only graceful checkpoint cap; after this many new "
              "durable rows, exit status 0 with status=partial and no "
              "finalization/COMPLETED marker"))
    parser.add_argument("--expected-index-sha256", default=EXPECTED_INDEX_SHA256)
    parser.add_argument("--expected-workload-sha256",
                        default=EXPECTED_FULL_WORKLOAD_SHA256)
    parser.add_argument("--expected-images", type=int, default=40)
    parser.add_argument("--expected-questions", type=int, default=240)
    args = parser.parse_args(argv)
    if args.run_dir is not None and args.results_dir is None:
        parser.error("--results-dir is required for a new run")
    if args.max_new_requests is not None and args.max_new_requests <= 0:
        parser.error("--max-new-requests must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    max_new_requests = args.max_new_requests
    if args.resume is not None:
        run_dir, results_dir, options = _prepare_resume(args)
        is_resume = True
    else:
        run_dir, results_dir, options = _prepare_new_run(args)
        is_resume = False

    with run_lock(run_dir):
        all_entries, entries, workload = _validate_workload(options)
        del all_entries
        schedule = expected_schedule(
            entries, skip=int(options["skip"]),
            questions=int(options["questions"]), seed=int(options["seed"]))
        expected_ids = [row["request_id"] for row in schedule]
        if len(expected_ids) != len(set(expected_ids)):
            raise AssertionError("expected schedule contains duplicate IDs")
        reference = load_reference(Path(options["reference_run_dir"]))
        image_ids = [str(entry["image_id"]) for entry in entries]
        if is_resume:
            manifest = _read_json(run_dir / "manifest.json")
            if manifest.get("expected_request_ids_sha256") != canonical_hash(
                    expected_ids):
                raise ValueError("resume schedule differs from immutable manifest")
            if manifest.get("source_store_realpath") != str(Path(
                    options["store_dir"]).resolve()):
                raise ValueError("resume source-store path differs from manifest")
            inventory_before = manifest["source_store_inventory_before"]
            config = _read_json(run_dir / "config.json")
            config["resume_count"] = int(config.get("resume_count", 0)) + 1
            config.setdefault("resume_events", []).append({
                "at_unix": time.time(), "pid": os.getpid()})
        else:
            inventory_before = source_store_inventory(
                Path(options["store_dir"]), image_ids)
            run_id = uuid.uuid4().hex
            manifest = {
                "schema_version": SCHEMA_VERSION, "run_id": run_id,
                "created_at_unix": time.time(), "options": options,
                "method_keys": list(METHOD_KEYS), "methods": METHODS,
                "expected_request_count": len(schedule),
                "expected_request_ids_sha256": canonical_hash(expected_ids),
                "workload": workload,
                "reference_hashes": reference["hashes"],
                "source_store_realpath": str(Path(
                    options["store_dir"]).resolve()),
                "source_store_inventory_before": inventory_before,
                "source_store_mutation_policy": "read_only_no_copy_no_rebuild",
            }
            atomic_json(run_dir / "manifest.json", manifest, exclusive=True)
            config = {
                "schema_version": SCHEMA_VERSION, "run_id": run_id,
                "status": "initializing", "dataset": "gqa", **workload,
                "n_images": workload["selected_images"],
                "n_questions": workload["selected_questions"],
                "skip": int(options["skip"]),
                "questions_per_image": int(options["questions"]),
                "seed": int(options["seed"]),
                "max_new_tokens": int(options["max_new_tokens"]),
                "method_keys": list(METHOD_KEYS), "methods": METHODS,
                "model": None, "chunk_size": CHUNK_SIZE,
                "probe_heads": PROBE_HEADS, "run_dir": str(run_dir),
                "results_dir": str(results_dir),
                "store_dir": options["store_dir"],
                "reference_run_dir": options["reference_run_dir"],
                "store_policy": "validated canonical store read-only reuse",
                "history_policy": "none_independent_gqa_questions",
                "future_question_leakage": 0,
                "cache_hit_policy": (
                    "cold page-cache conditioning outside TTFT for every "
                    "stored request; balanced five-way image rotation"),
                "ttft_definition": (
                    "individual request start before prompt construction and "
                    "tokenization -> initial H2D -> online selection/I/O/"
                    "scatter or vision -> prefill -> synchronized first token"),
                "chunk_budget_policy": (
                    "budget_chunk_count(n_chunks, 0.25), shared exactly with Ours"),
                "chunk_score_policy": "mean of valid spatial tokens only",
                "reference_equivalence_policy": {
                    "predictions_accuracy": (
                        "ReComp exact; stored arms cache-hit <=2% discrete "
                        "prediction/accuracy allowance"),
                    "deterministic_ssd_selection": (
                        "FullLoad/QA-Token25/Ours25 cache-hit bytes, preads, "
                        "touched ratio, and selected IDs exact"),
                    "ttft_tolerance": (
                        "absolute delta <= max(125 ms, 35% of reference)"),
                    "qa_token_selector_tolerance": (
                        "absolute delta <= max(40 ms, 50% of reference)"),
                },
                "started_at_unix": time.time(), "resume_count": 0,
                "resume_events": [],
            }
            atomic_json(run_dir / "config.json", config, exclusive=True)

        raw_path = run_dir / "results_partial.jsonl"
        rows, repaired_tail = read_jsonl_unique(
            raw_path, repair_tail=is_resume,
            recovery_dir=run_dir / "recovery")
        completed = {row["request_id"] for row in rows}
        if not completed <= set(expected_ids):
            extra = sorted(completed - set(expected_ids))
            raise ValueError(f"partial results contain foreign request IDs: {extra}")
        failure_path = run_dir / "failures.jsonl"
        failure_rows = ([json.loads(line) for line in failure_path.read_text(
            encoding="utf-8").splitlines()] if failure_path.exists() else [])
        atomic_json(run_dir / "progress.json", _progress(
            config["run_id"], len(schedule), completed, failure_rows,
            "resuming" if is_resume else "starting",
            repaired_tail_bytes=repaired_tail))

        random.seed(int(options["seed"]))
        np.random.seed(int(options["seed"]))
        torch.manual_seed(int(options["seed"]))
        torch.cuda.manual_seed_all(int(options["seed"]))
        runner = LlavaRunner().load()
        server = Server(
            runner, ratio=0.25, probe=PROBE_HEADS,
            max_new_tokens=int(options["max_new_tokens"]))
        warmup = _base_module()._warmup(runner, server)
        config["model"] = runner.model_id
        config.setdefault("warmup_events", []).append({
            "at_unix": time.time(), "resume": is_resume, **warmup})
        config.setdefault("invocations", []).append({
            "at_unix": time.time(), "pid": os.getpid(),
            "resume": is_resume,
            "max_new_requests": max_new_requests,
        })
        config["status"] = "running"
        atomic_json(run_dir / "config.json", config)

        metas: dict[str, dict[str, Any]] = {}
        run_started = time.perf_counter()
        last_request = None
        newly_completed = 0
        try:
            for image_index, entry in enumerate(entries):
                image_id = str(entry["image_id"])
                questions = entry["questions"][
                    int(options["skip"]):int(options["skip"])
                    + int(options["questions"])]
                order = deterministic_method_rotation(
                    METHOD_KEYS, image_index, int(options["seed"]))
                image_expected = {
                    request_id(image_id, str(q["question_id"]), method)
                    for q in questions for method in METHOD_KEYS}
                if image_expected <= completed:
                    # Metadata is still needed for final independent accounting.
                    raster_meta = _read_json(Path(options["store_dir"])
                                             / "raster" / image_id / "meta.json")
                    ours_meta = _read_json(Path(options["store_dir"])
                                           / "image_only" / image_id / "meta.json")
                    metas[image_id] = {"raster": raster_meta,
                                       "image_only": ours_meta}
                    continue
                image_path = ROOT / entry["image_path"]
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                raster_ctx = ImageContext(
                    Path(options["store_dir"]) / "raster" / image_id,
                    runner.model.device, require_v_hidden=True)
                ours_ctx = ImageContext(
                    Path(options["store_dir"]) / "image_only" / image_id,
                    runner.model.device, require_v_hidden=False)
                raster_ctx.validate_qa_select_layout()
                ours_ctx.validate_prefix_layout("visionzip_image_only")
                metas[image_id] = {"raster": dict(raster_ctx.meta),
                                   "image_only": dict(ours_ctx.meta)}
                full_visual_bytes = int(raster_ctx.meta["bytes_visual_kv"])
                if full_visual_bytes != int(ours_ctx.meta["bytes_visual_kv"]):
                    raise AssertionError("raster/image-only full KV bytes differ")
                try:
                    for turn_id, question in enumerate(questions, 1):
                        for position, method_key in enumerate(order):
                            identity = request_id(
                                image_id, str(question["question_id"]), method_key)
                            if identity in completed:
                                continue
                            if request_cap_reached(
                                    newly_completed, max_new_requests):
                                raise PartialRunStop
                            try:
                                if turn_id == 1 or method_key == "recompute":
                                    result, diagnostic = _base_module()._run_pixels(
                                        runner, server, image,
                                        question["question"], "none")
                                    result = _json_result(
                                        result, method_key,
                                        full_visual_bytes if turn_id > 1 else 0)
                                else:
                                    context = (ours_ctx if method_key == "ours25"
                                               else raster_ctx)
                                    result, diagnostic = _run_stored(
                                        runner, server, context,
                                        question["question"], method_key,
                                        image_id, full_visual_bytes)
                                record = {
                                    "schema_version": SCHEMA_VERSION,
                                    "run_id": config["run_id"],
                                    "request_id": identity,
                                    "measurement_source": "same_run",
                                    "dataset": "gqa", "image_id": image_id,
                                    "question_id": str(question["question_id"]),
                                    "turn_id": turn_id,
                                    "question": question["question"],
                                    "gold": question_answers(question),
                                    "method_key": method_key,
                                    **METHODS[method_key],
                                    "method_order": list(order),
                                    "method_order_position": position,
                                    "request_path": (
                                        "normal_pixel_turn1" if turn_id == 1
                                        else "normal_pixel_recompute"
                                        if method_key == "recompute"
                                        else "ssd_cache_hit"),
                                    "prediction": result["answer"],
                                    "correct": METRICS["gqa"](
                                        result["answer"],
                                        question_answers(question)),
                                    "first_token_id": int(result["first_token_id"]),
                                    "prompt_sha256": diagnostic["prompt_sha256"],
                                    "suffix_ids_sha256": diagnostic[
                                        "suffix_ids_sha256"],
                                    "input_tensors_sha256": diagnostic.get(
                                        "input_tensors_sha256"),
                                    **_base_module()._causal_prompt_fields(
                                        runner, diagnostic,
                                        question["question"],
                                        question["question_id"]),
                                    "chunk_size": int(raster_ctx.meta["chunk_size"]),
                                    **result,
                                }
                                record.update(METHODS[method_key])
                                append_jsonl_durable(raw_path, record)
                                completed.add(identity)
                                newly_completed += 1
                                rows.append(record)
                                last_request = identity
                                atomic_json(run_dir / "progress.json", _progress(
                                    config["run_id"], len(schedule), completed,
                                    failure_rows, "running", last_request,
                                    repaired_tail))
                            except PartialRunStop:
                                raise
                            except Exception as error:
                                failure = {
                                    "schema_version": SCHEMA_VERSION,
                                    "run_id": config["run_id"],
                                    "request_id": identity,
                                    "image_id": image_id,
                                    "question_id": str(question["question_id"]),
                                    "turn_id": turn_id, "method_key": method_key,
                                    "failed_at_unix": time.time(),
                                    "exception_type": type(error).__name__,
                                    "exception": str(error),
                                    "traceback": traceback.format_exc(),
                                }
                                append_jsonl_durable(failure_path, failure)
                                failure_rows.append(failure)
                                atomic_json(run_dir / "progress.json", _progress(
                                    config["run_id"], len(schedule), completed,
                                    failure_rows, "failed_request", identity,
                                    repaired_tail))
                                raise
                finally:
                    raster_ctx.close()
                    ours_ctx.close()
                    del raster_ctx, ours_ctx, image
                    torch.cuda.empty_cache()
                print(
                    f"[{image_index + 1}/{len(entries)}] {image_id} "
                    f"completed={len(completed)}/{len(schedule)} "
                    f"elapsed={time.perf_counter() - run_started:.1f}s",
                    flush=True)
        except PartialRunStop:
            config["status"] = "partial"
            config["partial_at_unix"] = time.time()
            config["last_invocation_new_requests"] = newly_completed
            atomic_json(run_dir / "config.json", config)
            atomic_json(run_dir / "progress.json", _progress(
                config["run_id"], len(schedule), completed, failure_rows,
                "partial", last_request, repaired_tail))
            print(json.dumps({
                "run_dir": str(run_dir), "status": "partial",
                "new_requests": newly_completed,
                "completed": len(completed), "expected": len(schedule),
                "resume": f"--resume {run_dir}",
            }, indent=2), flush=True)
            return 0
        except BaseException:
            config["status"] = "interrupted_or_failed"
            config["last_failure_at_unix"] = time.time()
            atomic_json(run_dir / "config.json", config)
            raise

        final = _finalize(
            run_dir, results_dir, config, schedule, workload, metas,
            inventory_before, reference)
        atomic_json(run_dir / "progress.json", _progress(
            config["run_id"], len(schedule), completed, failure_rows,
            "completed" if final["validation"]["passed"]
            else "failed_validation", last_request, repaired_tail))
        # Marker-last contract: progress and every final artifact are durable
        # before COMPLETED is exclusively published.
        if final["validation"]["passed"]:
            atomic_json(run_dir / "COMPLETED",
                        final["completed_payload"], exclusive=True)
        print(json.dumps({
            "run_dir": str(run_dir), "results_dir": str(results_dir),
            "passed": final["validation"]["passed"],
            "expected": len(schedule), "completed": len(completed),
            "comparison": final["comparison"],
        }, indent=2), flush=True)
        if not final["validation"]["passed"]:
            failed = [key for key, value in
                      final["validation"]["checks"].items() if not value]
            raise RuntimeError("validation failed: " + ", ".join(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

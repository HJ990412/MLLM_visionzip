#!/usr/bin/env python3
"""Orchestrate the MT-VQA-v2-reconstructed Generated-History evaluation."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mmimpress import mt_vqa_v2  # noqa: E402

PREFIX = "mt_vqa_v2_generated_4arm_"
SCHEMA_VERSION = "mt-vqa-v2-generated-4arm-orchestrator-v1"
SHARD_SCHEMA = "mt-vqa-v2-generated-4arm-shard-v1"
DEFAULT_INDEX = ROOT / "data/mt_vqa_v2/dialogues.json"
DATASET_DIR = ROOT / "data/mt_vqa_v2"
EVALUATOR = ROOT / "scripts/60_eval_mt_vqa_v2_generated_shard.py"
ANALYZER = ROOT / "scripts/61_analyze_mt_vqa_v2_generated.py"
BASE_PROTECTOR = ROOT / "scripts/57_protect_mt_gqa_history_artifacts.py"
PROTECTOR = ROOT / "scripts/64_protect_mt_vqa_v2_generated_artifacts.py"
LAUNCHER = ROOT / "scripts/63_launch_mt_vqa_v2_generated.sh"
EXPECTED_INDEX_SHA256 = (
    "89719b2a1187c07e3228cc76cf1e473b3c713a0dcb3d65da0c596ae81898d6ea"
)
EXPECTED_WORKLOAD_SHA256 = (
    "384e39bad4e2e8d5865fe20bad7661d3cad8fe0b5ce5effbfc43ea896170cbbc"
)
EXPECTED_SOURCE_INDEX_SHA256 = (
    "b83d5fa288fcb722ca073e261d3fec9086629ed0db2e568a2d5d6a24ef1589d7"
)
EXPECTED_MODEL_REVISION = "c916e6cdcd760b4cecd1dd4907f84ac649f93b23"
EXPECTED_DIALOGUES = 250
EXPECTED_IMAGES = 250
TURNS_PER_DIALOGUE = 3
EXPECTED_TURNS = 750
METHOD_KEYS = ("recompute", "fullload", "qa_chunk25", "ours25")
PROTOCOL = "generated_history"
FULL_REQUESTS = EXPECTED_DIALOGUES * TURNS_PER_DIALOGUE * len(METHOD_KEYS)
SMOKE_DIALOGUES = 10
SMOKE_REQUESTS = SMOKE_DIALOGUES * TURNS_PER_DIALOGUE * len(METHOD_KEYS)
PHYSICAL_REQUESTS_WITH_SMOKE = FULL_REQUESTS + SMOKE_REQUESTS
SEED = 1234
MAX_NEW_TOKENS = 16
DEFAULT_SHARD_SIZE = 50
MIN_SHARD_SIZE = 40
MAX_SHARD_SIZE = 60
MIN_FREE_AFTER_GIB = 30.0
MIN_GPU_FREE_MIB = 20_000
TEMP_OWNER_FILE = ".mt_vqa_v2_temp_store_owner.json"
FROZEN_QA_CONFIGURATION = {
    "physical_layout": "raster",
    "head_reduce": "mean",
    "chunk_aggregation": "mean_valid_spatial_tokens",
    "normal_chunk_budget": 0.25,
    "budget_helper": "budget_chunk_count_round",
    "rater_algorithm_id": "sparsevlm_visual_text_mean_threshold_v1",
    "rater_scope": "entire_available_causal_suffix",
    "fallback": False,
    "adaptive_budget": False,
}
FROZEN_OURS_CONFIGURATION = {
    "physical_layout": "visionzip_image_only",
    "normal_chunk_budget": 0.25,
    "selection": "fixed_first_k_prefix",
    "online_query_scoring": False,
}
# Shared with the MT-GQA orchestrator because both own the only GPU.
GLOBAL_RUN_LOCK = Path("/tmp/mllm_v2_mt_gqa_history_run.lock")
MODEL_REF = Path(
    "/home/dblab/.cache/huggingface/hub/"
    "models--llava-hf--llava-v1.6-vicuna-7b-hf/refs/main")
RUNTIME_PACKAGES = (
    "torch", "transformers", "Pillow", "numpy", "psutil",
    "bitsandbytes", "accelerate",
)
RUNTIME_CODE_PATHS = (
    ROOT / "scripts/28_eval_visdial_turn1_piggyback.py",
    ROOT / "scripts/37_eval_mt_gqa_full_shard.py",
    ROOT / "scripts/49_eval_query_aware_baseline.py",
    ROOT / "scripts/54_eval_mt_gqa_history_shard.py",
    ROOT / "scripts/55_analyze_mt_gqa_history.py",
    ROOT / "scripts/56_run_mt_gqa_history.py",
    BASE_PROTECTOR,
    PROTECTOR,
    ROOT / "scripts/59_build_mt_vqa_v2_index.py",
    EVALUATOR,
    ANALYZER,
    Path(__file__).resolve(),
    LAUNCHER,
    ROOT / "mmimpress/__init__.py",
    ROOT / "mmimpress/config.py",
    ROOT / "mmimpress/cvpr25.py",
    ROOT / "mmimpress/dataset.py",
    ROOT / "mmimpress/model.py",
    ROOT / "mmimpress/multiturn.py",
    ROOT / "mmimpress/mt_gqa.py",
    ROOT / "mmimpress/mt_vqa_v2.py",
    ROOT / "mmimpress/piggyback.py",
    ROOT / "mmimpress/reorder.py",
    ROOT / "mmimpress/serve.py",
    ROOT / "mmimpress/sparsevlm.py",
    ROOT / "mmimpress/store.py",
)
REQUIRED_RESULTS = (
    "raw.jsonl", "raw.jsonl.gz", "summary.csv", "quality_by_turn.csv",
    "ttft_by_turn.csv", "token_lengths.csv", "io_breakdown.csv",
    "selector_breakdown.csv", "persistence_summary.csv",
    "session_latency.csv", "selection_analysis.json",
    "error_propagation.csv", "error_propagation.json",
    "paired_quality.json", "dataset_construction.json", "validation.json",
    "config.json", "ANALYSIS.md", "README.md", "COMPLETED",
)


class OrchestrationError(RuntimeError):
    pass


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and os.path.lexists(path):
        raise FileExistsError(path)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True,
                      ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(temporary, path)
            temporary.unlink()
        else:
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


def read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise OrchestrationError(f"required JSON is not regular: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OrchestrationError(f"JSON root is not an object: {path}")
    return value


def validate_roots(run_root: Path, results_root: Path) -> tuple[Path, Path]:
    run = run_root if run_root.is_absolute() else ROOT / run_root
    results = results_root if results_root.is_absolute() else ROOT / results_root
    run, results = run.resolve(strict=False), results.resolve(strict=False)
    if (run.parent != (ROOT / "runs").resolve()
            or results.parent != (ROOT / "results").resolve()
            or not run.name.startswith(PREFIX)
            or not results.name.startswith(PREFIX)
            or run.name != results.name):
        raise ValueError("run/results must be matching dedicated MT-VQA-v2 roots")
    for path in (run, results):
        if os.path.lexists(path) and path.is_symlink():
            raise ValueError(f"output root may not be a symlink: {path}")
    return run, results


def read_index_contract(index: Path) -> dict[str, Any]:
    path = index.resolve()
    if (path != DEFAULT_INDEX.resolve() or path.is_symlink()
            or not path.is_file()):
        raise ValueError(f"canonical index is fixed to {DEFAULT_INDEX}")
    digest = sha256_file(path)
    if digest != EXPECTED_INDEX_SHA256:
        raise ValueError(f"dialogues SHA256 mismatch: {digest}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    dialogs = payload.get("dialogues") if isinstance(payload, dict) else None
    if (not isinstance(dialogs, list) or len(dialogs) != EXPECTED_DIALOGUES
            or payload.get("benchmark_type") != "MT-VQA-v2-reconstructed"):
        raise ValueError("canonical MT-VQA-v2 dialogue envelope mismatch")
    ids: list[str] = []
    image_ids: list[str] = []
    qids: set[str] = set()
    framed: list[str] = []
    for ordinal, dialog in enumerate(dialogs):
        did = str(dialog.get("dialog_id", ""))
        image_id = str(dialog.get("image_id", ""))
        turns = dialog.get("turns")
        if (did != f"mtvqav2_{ordinal + 1:06d}"
                or int(dialog.get("global_dialog_ordinal", -1)) != ordinal
                or not image_id or not isinstance(turns, list)
                or len(turns) != 3):
            raise ValueError(f"invalid canonical dialogue: {did!r}")
        for turn_id, turn in enumerate(turns, 1):
            qid = str(turn.get("question_id", ""))
            answers = turn.get("answers")
            if (int(turn.get("turn_id", -1)) != turn_id or not qid
                    or qid in qids or not isinstance(answers, list)
                    or len(answers) != 10):
                raise ValueError(f"invalid canonical turn: {did}/T{turn_id}")
            qids.add(qid)
            framed.append(f"{did}\t{turn_id}\t{qid}\n")
        ids.append(did)
        image_ids.append(image_id)
    if len(set(image_ids)) != EXPECTED_IMAGES:
        raise ValueError("workload must contain one dialogue per image")
    workload_sha = hashlib.sha256(
        "".join(framed).encode("utf-8")).hexdigest()
    if workload_sha != EXPECTED_WORKLOAD_SHA256:
        raise ValueError(f"workload SHA256 mismatch: {workload_sha}")
    dataset_validation = mt_vqa_v2.validate_artifact_directory(
        DATASET_DIR,
        source_index=ROOT / "data/vqav2/index.json",
        source_config=ROOT / "data/vqav2/config.json",
        strict_canonical=True,
    )
    if (dataset_validation.get("passed") is not True
            or dataset_validation.get("dialogues_sha256")
            != EXPECTED_INDEX_SHA256
            or dataset_validation.get("workload_sha256")
            != EXPECTED_WORKLOAD_SHA256):
        raise ValueError("deterministic MT-VQA-v2 dataset rebuild failed")
    dataset_hashes = {}
    for name in ("dialogues.json", "config.json", "dataset_stats.json",
                 "dataset_provenance.json"):
        artifact = DATASET_DIR / name
        if artifact.is_symlink() or not artifact.is_file():
            raise FileNotFoundError(artifact)
        dataset_hashes[name] = sha256_file(artifact)
    source_digest = sha256_file(ROOT / "data/vqav2/index.json")
    if source_digest != EXPECTED_SOURCE_INDEX_SHA256:
        raise ValueError("frozen VQAv2 source index changed")
    return {
        "index": path, "index_sha256": digest,
        "workload_sha256": workload_sha, "dialogues": dialogs,
        "dialogue_ids": ids, "image_ids": image_ids,
        "dataset_artifact_sha256": dataset_hashes,
        "dataset_validation": dataset_validation,
        "source_index_sha256": source_digest,
    }


def runtime_code_hashes() -> dict[str, str]:
    output: dict[str, str] = {}
    for path in RUNTIME_CODE_PATHS:
        if path.is_symlink() or not path.is_file():
            raise OrchestrationError(f"runtime dependency is not regular: {path}")
        key = path.resolve().relative_to(ROOT.resolve()).as_posix()
        if key in output:
            raise OrchestrationError(f"duplicate runtime dependency: {key}")
        output[key] = sha256_file(path)
    return output


def runtime_environment() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for package in RUNTIME_PACKAGES:
        versions[package] = importlib.metadata.version(package)
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "packages": versions,
    }


def local_model_revision() -> str:
    if MODEL_REF.is_symlink() or not MODEL_REF.is_file():
        raise FileNotFoundError(MODEL_REF)
    revision = MODEL_REF.read_text(encoding="utf-8").strip()
    if revision != EXPECTED_MODEL_REVISION:
        raise ValueError(f"local model revision changed: {revision}")
    snapshot = MODEL_REF.parent.parent / "snapshots" / revision
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise FileNotFoundError(snapshot)
    return revision


def gpu_preflight() -> dict[str, Any]:
    command = [
        "nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,"
        "utilization.gpu,driver_version", "--format=csv,noheader,nounits"]
    completed = subprocess.run(
        command, check=True, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    rows = [line.strip() for line in completed.stdout.splitlines()
            if line.strip()]
    if len(rows) != 1:
        raise OrchestrationError(f"expected exactly one GPU, got {len(rows)}")
    fields = [item.strip() for item in rows[0].split(",")]
    if len(fields) != 6:
        raise OrchestrationError("malformed nvidia-smi output")
    total, used, free, utilization = map(int, fields[1:5])
    if free < MIN_GPU_FREE_MIB or utilization > 10:
        raise OrchestrationError(
            f"GPU is busy: free={free} MiB utilization={utilization}%")
    return {
        "name": fields[0], "memory_total_mib": total,
        "memory_used_mib": used, "memory_free_mib": free,
        "utilization_gpu_pct": utilization, "driver_version": fields[5],
    }


def run_streaming(command: Sequence[str], log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.parent.is_symlink() or log_path.is_symlink():
        raise ValueError(f"unsafe log path: {log_path}")
    environment = dict(os.environ)
    environment.update(
        HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        log.write("COMMAND " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            list(command), cwd=ROOT, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return_code = process.wait()
        os.fsync(log.fileno())
    if return_code:
        raise subprocess.CalledProcessError(return_code, list(command))
    return time.monotonic() - started


def protection_command(mode: str, run: Path, results: Path) -> list[str]:
    if mode not in {"before", "verify"}:
        raise ValueError(mode)
    return [sys.executable, str(PROTECTOR), f"--{mode}",
            "--run-root", str(run), "--results-root", str(results)]


def protection_before(run: Path, results: Path) -> dict[str, Any]:
    completed = subprocess.run(
        protection_command("before", run, results), cwd=ROOT, check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    value = json.loads(completed.stdout)
    if value.get("status") != "recorded":
        raise OrchestrationError("artifact protector did not record snapshot")
    return value


def protection_verify(run: Path, results: Path) -> dict[str, Any]:
    completed = subprocess.run(
        protection_command("verify", run, results), cwd=ROOT, check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    value = json.loads(completed.stdout)
    if value.get("status") != "unchanged":
        raise OrchestrationError("pre-existing artifacts changed")
    return value


def make_manifest(
    run: Path, results: Path, experiment_id: str,
    contract: Mapping[str, Any], gpu: Mapping[str, Any], shard_size: int,
    reserve: float, protection: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "benchmark_type": "MT-VQA-v2-reconstructed",
        "official_benchmark_identity_claimed": False,
        "exact_metacompress_reproduction_claimed": False,
        "run_root": str(run), "results_root": str(results),
        "index": str(contract["index"]),
        "index_sha256": contract["index_sha256"],
        "workload_sha256": contract["workload_sha256"],
        "source_index_sha256": contract["source_index_sha256"],
        "dataset_artifact_sha256": contract["dataset_artifact_sha256"],
        "dataset_validation": contract["dataset_validation"],
        "images": EXPECTED_IMAGES, "dialogues": EXPECTED_DIALOGUES,
        "turns": EXPECTED_TURNS, "methods": list(METHOD_KEYS),
        "protocols": [PROTOCOL],
        "logical_request_counts": {
            "full": FULL_REQUESTS,
            "main_t2_t3_requests": EXPECTED_DIALOGUES * 2 * len(METHOD_KEYS),
            "stored_visual_kv_cache_hits": EXPECTED_DIALOGUES * 2 * 3,
            "recompute_later_turn_reference": EXPECTED_DIALOGUES * 2,
            "turn1": EXPECTED_DIALOGUES * len(METHOD_KEYS),
            "smoke": SMOKE_REQUESTS,
            "physical_including_smoke": PHYSICAL_REQUESTS_WITH_SMOKE,
        },
        "construction": {
            "source_slice": "questions[1:5]",
            "dialogue_membership": "questions[1:4]",
            "one_dialogue_per_image": True,
            "question_overlap": False,
        },
        "history_policy": "same-method generated A1/A2 only",
        "quality_metric": "repository_vqa_consensus_min_matches_over_3",
        "soft_score_primary": True,
        "binary_diagnostics_only": True,
        "shard_size_images": int(shard_size), "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "min_free_after_gib": float(reserve),
        "model": "llava-hf/llava-v1.6-vicuna-7b-hf",
        "model_revision": EXPECTED_MODEL_REVISION,
        "serving_configuration": {
            "chunk_size": 64,
            "probe_heads": 3,
            "qa_chunk25": FROZEN_QA_CONFIGURATION,
            "ours25": FROZEN_OURS_CONFIGURATION,
        },
        "gpu_preflight": dict(gpu),
        "protection_before": dict(protection),
        "code_sha256": runtime_code_hashes(),
        "runtime_environment": runtime_environment(),
        "created_at_unix": time.time(),
    }
    value["manifest_sha256"] = canonical_hash(value)
    return value


def validate_manifest(
    manifest: Mapping[str, Any], run: Path, results: Path,
    contract: Mapping[str, Any],
) -> None:
    body = dict(manifest)
    recorded = body.pop("manifest_sha256", None)
    if recorded != canonical_hash(body):
        raise OrchestrationError("manifest content hash mismatch")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "run_root": str(run), "results_root": str(results),
        "index_sha256": contract["index_sha256"],
        "workload_sha256": contract["workload_sha256"],
        "source_index_sha256": contract["source_index_sha256"],
        "dataset_artifact_sha256": contract["dataset_artifact_sha256"],
        "dataset_validation": contract["dataset_validation"],
        "images": EXPECTED_IMAGES, "dialogues": EXPECTED_DIALOGUES,
        "turns": EXPECTED_TURNS, "methods": list(METHOD_KEYS),
        "protocols": [PROTOCOL], "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "construction": {
            "source_slice": "questions[1:5]",
            "dialogue_membership": "questions[1:4]",
            "one_dialogue_per_image": True,
            "question_overlap": False,
        },
        "history_policy": "same-method generated A1/A2 only",
        "quality_metric": "repository_vqa_consensus_min_matches_over_3",
        "soft_score_primary": True,
        "binary_diagnostics_only": True,
        "model": "llava-hf/llava-v1.6-vicuna-7b-hf",
        "model_revision": EXPECTED_MODEL_REVISION,
        "serving_configuration": {
            "chunk_size": 64,
            "probe_heads": 3,
            "qa_chunk25": FROZEN_QA_CONFIGURATION,
            "ours25": FROZEN_OURS_CONFIGURATION,
        },
        "code_sha256": runtime_code_hashes(),
        "runtime_environment": runtime_environment(),
    }
    mismatch = {key: (manifest.get(key), value)
                for key, value in expected.items()
                if manifest.get(key) != value}
    counts = manifest.get("logical_request_counts", {})
    expected_counts = {
        "full": FULL_REQUESTS,
        "main_t2_t3_requests": EXPECTED_DIALOGUES * 2 * len(METHOD_KEYS),
        "stored_visual_kv_cache_hits": EXPECTED_DIALOGUES * 2 * 3,
        "recompute_later_turn_reference": EXPECTED_DIALOGUES * 2,
        "turn1": EXPECTED_DIALOGUES * len(METHOD_KEYS),
        "smoke": SMOKE_REQUESTS,
        "physical_including_smoke": PHYSICAL_REQUESTS_WITH_SMOKE,
    }
    if counts != expected_counts:
        mismatch["logical_request_counts"] = (counts, expected_counts)
    if mismatch:
        raise OrchestrationError(f"immutable manifest mismatch: {mismatch}")


def marker_path(stage_dir: Path, shard_index: int) -> Path:
    return stage_dir / "shards" / f"shard_{shard_index:03d}.json"


def marker_is_complete(
    path: Path, *, experiment_id: str, shard_index: int, shard_size: int,
) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
        image_ids = [str(item) for item in marker.get("image_ids", [])]
        completed = [str(item) for item in marker.get(
            "completed_image_ids", [])]
        files = marker.get("image_artifact_file_sha256")
        contents = marker.get("image_artifact_content_sha256")
        if (marker.get("schema_version") != SHARD_SCHEMA
                or marker.get("complete") is not True
                or marker.get("experiment_id") != experiment_id
                or marker.get("protocol") != PROTOCOL
                or int(marker.get("shard_index", -1)) != shard_index
                or int(marker.get("shard_size", -1)) != shard_size
                or not image_ids or completed != image_ids
                or len(set(image_ids)) != len(image_ids)
                or not isinstance(files, dict) or set(files) != set(image_ids)
                or not isinstance(contents, dict)
                or set(contents) != set(image_ids)
                or marker.get("artifact_content_sha256") != canonical_hash({
                    key: value for key, value in marker.items()
                    if key != "artifact_content_sha256"})):
            return False
        image_dir = path.parent.parent / "images"
        for image_id in image_ids:
            artifact_path = image_dir / f"{image_id}.json"
            if (not image_id or Path(image_id).name != image_id
                    or artifact_path.is_symlink() or not artifact_path.is_file()
                    or sha256_file(artifact_path) != files[image_id]):
                return False
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            if (artifact.get("experiment_id") != experiment_id
                    or artifact.get("protocol") != PROTOCOL
                    or artifact.get("image_id") != image_id
                    or int(artifact.get("shard_index", -1)) != shard_index
                    or artifact.get("artifact_content_sha256")
                    != contents[image_id]
                    or contents[image_id] != canonical_hash({
                        key: value for key, value in artifact.items()
                        if key != "artifact_content_sha256"})):
                return False
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def evaluator_command(
    stage_dir: Path, temp_root: Path, experiment_id: str,
    shard_index: int, shard_size: int, dialogue_limit: int,
    reserve: float, index: Path,
) -> list[str]:
    if dialogue_limit not in {SMOKE_DIALOGUES, EXPECTED_DIALOGUES}:
        raise ValueError("dialogue limit must be smoke or full")
    command = [
        sys.executable, str(EVALUATOR), "--protocol", PROTOCOL,
        "--index", str(index), "--run-dir", str(stage_dir),
        "--temp-root", str(temp_root), "--experiment-id", experiment_id,
        "--shard-index", str(shard_index), "--shard-size", str(shard_size),
        "--expected-index-sha256", EXPECTED_INDEX_SHA256,
        "--expected-workload-sha256", EXPECTED_WORKLOAD_SHA256,
        "--expected-dialogs", str(dialogue_limit), "--seed", str(SEED),
        "--max-new-tokens", str(MAX_NEW_TOKENS),
        "--min-free-after-gib", str(float(reserve)),
    ]
    if dialogue_limit != EXPECTED_DIALOGUES:
        command.extend(["--max-dialogs", str(dialogue_limit),
                        "--allow-partial-workload"])
    return command


def selected_image_count(dialogues: Sequence[Mapping[str, Any]], limit: int) -> int:
    return len({str(row["image_id"]) for row in dialogues[:limit]})


def validate_stage(
    stage_dir: Path, experiment_id: str,
    dialogues: Sequence[Mapping[str, Any]], dialogue_limit: int,
    shard_size: int,
) -> dict[str, Any]:
    selected = list(dialogues[:dialogue_limit])
    expected_dialogs = {str(row["dialog_id"]) for row in selected}
    expected_images = {str(row["image_id"]) for row in selected}
    expected_rows = dialogue_limit * 3 * len(METHOD_KEYS)
    expected_shards = math.ceil(len(expected_images) / shard_size)
    config = read_json(stage_dir / "config.json")
    expected_config = {
        "schema_version": SHARD_SCHEMA,
        "experiment_id": experiment_id, "dataset":
            "vqav2_validation_mt3_reconstructed",
        "benchmark_type": "MT-VQA-v2-reconstructed",
        "protocol": PROTOCOL,
        "dialogues_file_sha256": EXPECTED_INDEX_SHA256,
        "source_full_workload_sha256": EXPECTED_WORKLOAD_SHA256,
        "n_dialogs": dialogue_limit, "n_turns": dialogue_limit * 3,
        "n_images": len(expected_images), "n_requests": expected_rows,
        "shard_size": shard_size, "n_shards": expected_shards,
        "seed": SEED, "method_keys": list(METHOD_KEYS),
        "model": "llava-hf/llava-v1.6-vicuna-7b-hf",
        "model_revision": EXPECTED_MODEL_REVISION,
        "load_4bit": True, "quantization": "4-bit NF4 double-quant",
        "compute_dtype": "bfloat16", "attention": "eager",
        "decoding": "greedy", "max_new_tokens": MAX_NEW_TOKENS,
        "chunk_size": 64, "probe_heads": 3,
        "quality_metric": "repository_vqa_consensus_min_matches_over_3",
        "quality_metric_implementation": "mmimpress.dataset.vqa_score",
        "history_policy": "method_local_generated",
    }
    mismatch = {key: (config.get(key), value)
                for key, value in expected_config.items()
                if config.get(key) != value}
    if mismatch:
        raise OrchestrationError(f"stage config mismatch: {mismatch}")
    if (config.get("qa_chunk_configuration") != FROZEN_QA_CONFIGURATION
            or config.get("ours_configuration")
            != FROZEN_OURS_CONFIGURATION):
        raise OrchestrationError("frozen QA/Ours serving configuration changed")
    image_paths = sorted((stage_dir / "images").glob("*.json"))
    if len(image_paths) != len(expected_images):
        raise OrchestrationError("image artifact count mismatch")
    observed_images: set[str] = set()
    observed_dialogs: set[str] = set()
    expected_dialog_for_image = {
        str(dialog["image_id"]): str(dialog["dialog_id"])
        for dialog in selected
    }
    logical_ids: set[str] = set()
    physical_ids: set[str] = set()
    failures = 0
    for path in image_paths:
        artifact = read_json(path)
        image_id = str(artifact.get("image_id", ""))
        rows = artifact.get("rows")
        validation = artifact.get("validation", {})
        if (image_id != path.stem or image_id not in expected_images
                or image_id in observed_images or not isinstance(rows, list)
                or validation.get("passed") is not True
                or validation.get("vqa_consensus_scores_recomputed") is not True
                or validation.get("turn1_pixel_and_input_hash_fairness") is not True
                or validation.get("method_local_generated_history_validated") is not True
                or int(validation.get("future_leakage", -1)) != 0
                or validation.get("ours_selection_invariant_within_protocol") is not True
                or int(validation.get("failed_rows", -1)) != 0
                or int(validation.get("duplicate_rows", -1)) != 0
                or int(validation.get("n_rows", -1)) != 12
                or int(validation.get("expected_rows", -1)) != 12
                or int(artifact.get("n_rows", -1)) != 12
                or int(artifact.get("logical_request_count", -1)) != 12
                or int(artifact.get("physical_execution_count", -1)) != 12
                or int(artifact.get("failed_request_count", -1)) != 0
                or int(artifact.get("duplicate_request_count", -1)) != 0
                or artifact.get("store_build_counts")
                != {"raster": 1, "image_only": 1}):
            raise OrchestrationError(f"invalid image artifact: {path}")
        dialog_ids = {str(row.get("dialog_id", "")) for row in rows}
        if (len(dialog_ids) != 1 or "" in dialog_ids
                or observed_dialogs.intersection(dialog_ids)
                or len(rows) != 12
                or dialog_ids != {expected_dialog_for_image[image_id]}):
            raise OrchestrationError(f"dialogue coverage mismatch: {path}")
        blocks: dict[int, list[dict[str, Any]]] = {}
        local_keys: set[tuple[str, int, str]] = set()
        for row in rows:
            turn = int(row.get("turn_id", -1))
            method = str(row.get("method_key", ""))
            logical = str(row.get("logical_request_id", ""))
            physical = str(row.get("physical_execution_id", ""))
            local_key = (str(row.get("dialog_id", "")), turn, method)
            if (str(row.get("image_id", "")) != image_id
                    or turn not in (1, 2, 3) or method not in METHOD_KEYS
                    or local_key in local_keys
                    or not logical or logical in logical_ids or not physical
                    or physical in physical_ids or row.get("protocol") != PROTOCOL):
                raise OrchestrationError(f"invalid request identity: {path}")
            score = float(row.get("quality_score", -1.0))
            answers = row.get("gold_answers")
            valid_scores = (0.0, 1 / 3, 2 / 3, 1.0)
            if (not any(abs(score - value) <= 1e-12
                        for value in valid_scores)
                    or not isinstance(answers, list) or len(answers) != 10):
                raise OrchestrationError(f"invalid VQA scoring row: {path}")
            logical_ids.add(logical)
            physical_ids.add(physical)
            local_keys.add(local_key)
            failures += int(row.get("status") != "ok")
            blocks.setdefault(turn, []).append(row)
        did = expected_dialog_for_image[image_id]
        expected_local_keys = {(did, turn, method) for turn in (1, 2, 3)
                               for method in METHOD_KEYS}
        if (set(blocks) != {1, 2, 3}
                or any(len(blocks[turn]) != 4 for turn in (1, 2, 3))
                or local_keys != expected_local_keys
                or any({row["method_key"] for row in block}
                       != set(METHOD_KEYS) for block in blocks.values())):
            raise OrchestrationError(f"four-arm coverage mismatch: {path}")
        t1 = blocks[1]
        fairness_fields = (
            "prompt_sha256", "image_input_sha256", "input_tensors_sha256",
            "prediction", "first_token_id")
        if any(len({str(row.get(field)) for row in t1}) != 1
               for field in fairness_fields):
            raise OrchestrationError(f"Turn-1 fairness mismatch: {path}")
        observed_images.add(image_id)
        observed_dialogs.update(dialog_ids)
    if (observed_images != expected_images or observed_dialogs != expected_dialogs
            or len(logical_ids) != expected_rows
            or len(physical_ids) != expected_rows or failures):
        raise OrchestrationError("exact stage request coverage failed")
    marker_paths = sorted((stage_dir / "shards").glob("shard_*.json"))
    if len(marker_paths) != expected_shards:
        raise OrchestrationError("shard marker count mismatch")
    marked: list[str] = []
    for shard_index, path in enumerate(marker_paths):
        if (path != marker_path(stage_dir, shard_index)
                or not marker_is_complete(
                    path, experiment_id=experiment_id,
                    shard_index=shard_index, shard_size=shard_size)):
            raise OrchestrationError(f"invalid shard marker: {path}")
        marked.extend(read_json(path)["image_ids"])
    if len(marked) != len(set(marked)) or set(marked) != expected_images:
        raise OrchestrationError("shard image partition mismatch")
    return {
        "passed": True, "protocol": PROTOCOL,
        "dialogues": len(observed_dialogs), "turns": len(observed_dialogs) * 3,
        "images": len(observed_images), "requests": len(logical_ids),
        "failed": failures, "duplicates": 0, "shards": expected_shards,
        "turn1_four_arm_fairness": True,
    }


def ensure_temp_clean(temp_root: Path, experiment_id: str) -> dict[str, Any]:
    if not os.path.lexists(temp_root):
        return {"passed": True, "exists": False, "files": 0}
    owner_path = temp_root / TEMP_OWNER_FILE
    payload = temp_root / "payload"
    if (temp_root.is_symlink() or not temp_root.is_dir()
            or owner_path.is_symlink() or not owner_path.is_file()
            or payload.is_symlink() or not payload.is_dir()):
        raise OrchestrationError("unsafe temporary store ownership structure")
    owner = read_json(owner_path)
    expected = {
        "schema_version": SHARD_SCHEMA, "experiment_id": experiment_id,
        "dataset": "vqav2_validation_mt3_reconstructed",
        "purpose": "temporary_visual_kv_only",
    }
    if any(owner.get(key) != value for key, value in expected.items()):
        raise OrchestrationError("temporary store ownership mismatch")
    if list(payload.iterdir()):
        raise OrchestrationError("temporary payload is not empty")
    unexpected = [path for path in temp_root.iterdir()
                  if path.name not in {owner_path.name, payload.name}]
    if unexpected:
        raise OrchestrationError("unexpected temporary-store entry")
    payload.rmdir()
    owner_path.unlink()
    temp_root.rmdir()
    return {"passed": True, "exists": True, "files": 0}


def run_stage(
    stage_name: str, dialogue_limit: int, run: Path,
    experiment_id: str, contract: Mapping[str, Any], shard_size: int,
    reserve: float, progress: dict[str, Any],
) -> dict[str, Any]:
    image_count = selected_image_count(contract["dialogues"], dialogue_limit)
    n_shards = math.ceil(image_count / shard_size)
    stage_dir = run / stage_name / PROTOCOL
    temp_root = run / "_temporary_visual_kv" / stage_name / PROTOCOL
    launched = 0
    started = time.monotonic()
    for shard_index in range(n_shards):
        marker = marker_path(stage_dir, shard_index)
        if marker_is_complete(
                marker, experiment_id=experiment_id,
                shard_index=shard_index, shard_size=shard_size):
            continue
        command = evaluator_command(
            stage_dir, temp_root, experiment_id, shard_index, shard_size,
            dialogue_limit, reserve, contract["index"])
        run_streaming(
            command, run / "logs" / stage_name /
            f"{PROTOCOL}_shard_{shard_index:03d}.log")
        launched += 1
        if not marker_is_complete(
                marker, experiment_id=experiment_id,
                shard_index=shard_index, shard_size=shard_size):
            raise OrchestrationError("evaluator returned without valid marker")
        progress.update({
            "status": "running", "stage": stage_name,
            "last_shard_index": shard_index, "updated_at_unix": time.time()})
        atomic_json(run / "progress.json", progress)
    validation = validate_stage(
        stage_dir, experiment_id, contract["dialogues"], dialogue_limit,
        shard_size)
    temp = ensure_temp_clean(temp_root, experiment_id)
    return {
        "passed": True, "stage": stage_name, "protocol": PROTOCOL,
        "dialogues": dialogue_limit,
        "requests": dialogue_limit * 3 * len(METHOD_KEYS),
        "images": image_count, "shards": n_shards,
        "validation": validation, "temporary_cleanup": temp,
        "subprocesses_launched": launched,
        "wall_seconds_this_invocation": time.monotonic() - started,
    }


def stage_complete(
    run: Path, stage_name: str, dialogue_limit: int,
    experiment_id: str, dialogues: Sequence[Mapping[str, Any]],
    shard_size: int,
) -> bool:
    count = math.ceil(
        selected_image_count(dialogues, dialogue_limit) / shard_size)
    return all(marker_is_complete(
        marker_path(run / stage_name / PROTOCOL, index),
        experiment_id=experiment_id, shard_index=index,
        shard_size=shard_size) for index in range(count))


def ensure_protection_verified(run: Path, results: Path) -> dict[str, Any]:
    before = read_json(run / "protected_artifacts_before.json")
    declared = read_json(run / "manifest.json").get("protection_before")
    expected = {
        "status": "recorded",
        "manifest": str(run / "protected_artifacts_before.json"),
        "manifest_sha256": before.get("manifest_sha256"),
        "entry_count": before.get("entry_count"),
        "file_count": before.get("file_count"),
        "total_bytes": before.get("total_bytes"),
    }
    if declared != expected:
        raise OrchestrationError("manifest protection binding mismatch")
    path = run / "protected_artifacts_validation.json"
    if not path.exists():
        protection_verify(run, results)
    report = read_json(path)
    if (report.get("passed") is not True
            or report.get("before_manifest_sha256")
            != before.get("manifest_sha256")
            or report.get("after_manifest_sha256")
            != before.get("manifest_sha256")
            or any(report.get(field) for field in (
                "missing_paths", "changed_paths"))):
        raise OrchestrationError("prior-artifact protection validation failed")
    return report


def analyzer_command(
    run_dir: Path, results: Path, index: Path, protection: Path,
) -> list[str]:
    return [
        sys.executable, str(ANALYZER), "--run-dir", str(run_dir),
        "--results-root", str(results), "--index", str(index),
        "--protection-validation", str(protection),
        "--expected-dialogs", str(EXPECTED_DIALOGUES),
        "--expected-images", str(EXPECTED_IMAGES),
        "--bootstrap-resamples", "10000", "--bootstrap-seed", str(SEED),
    ]


def analysis_is_complete(results: Path) -> bool:
    marker = results / "COMPLETED"
    if results.is_symlink() or not results.is_dir() \
            or marker.is_symlink() or not marker.is_file():
        return False
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
        if (value.get("passed") is not True
                or int(value.get("logical_rows", -1)) != FULL_REQUESTS):
            return False
        hashes = value.get("output_sha256")
        if not isinstance(hashes, dict):
            return False
        observed = {
            path.relative_to(results).as_posix(): sha256_file(path)
            for path in sorted(results.rglob("*"))
            if path != marker and path.is_file() and not path.is_symlink()
        }
        return hashes == observed
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def validate_analysis(
    results: Path, *, experiment_id: str, contract: Mapping[str, Any],
) -> dict[str, Any]:
    if not analysis_is_complete(results):
        raise OrchestrationError("analysis publication is incomplete")
    missing = [name for name in REQUIRED_RESULTS
               if not (results / name).is_file()
               or (results / name).is_symlink()]
    if missing:
        raise OrchestrationError(f"analysis outputs missing: {missing}")
    completed = read_json(results / "COMPLETED")
    validation = read_json(results / "validation.json")
    analysis_config = read_json(results / "config.json")
    if (validation.get("passed") is not True
            or int(validation.get("observed_logical_rows", -1))
            != FULL_REQUESTS
            or completed.get("validation_sha256")
            != sha256_file(results / "validation.json")
            or completed.get("analysis_sha256")
            != sha256_file(results / "ANALYSIS.md")):
        raise OrchestrationError("analysis completion contract failed")
    binding = {
        "experiment_id": str(experiment_id),
        "index_sha256": contract["index_sha256"],
        "workload_sha256": contract["workload_sha256"],
        "model_revision": EXPECTED_MODEL_REVISION,
        "seed": SEED,
        "methods": list(METHOD_KEYS),
        "protocol": PROTOCOL,
        "bootstrap": {
            "resamples": 10_000, "seed": SEED, "cluster_unit": "image"},
    }
    for label, document in (("COMPLETED", completed),
                            ("validation", validation),
                            ("config", analysis_config)):
        mismatch = {key: (document.get(key), value)
                    for key, value in binding.items()
                    if document.get(key) != value}
        if mismatch:
            raise OrchestrationError(
                f"{label} analysis binding mismatch: {mismatch}")
    if (int(analysis_config.get("logical_requests", -1)) != FULL_REQUESTS
            or int(validation.get("expected_dialogues", -1))
            != EXPECTED_DIALOGUES
            or int(validation.get("observed_dialogues", -1))
            != EXPECTED_DIALOGUES
            or int(validation.get("expected_images", -1)) != EXPECTED_IMAGES
            or int(validation.get("observed_images", -1)) != EXPECTED_IMAGES):
        raise OrchestrationError("analysis workload binding mismatch")
    with (results / "ttft_by_turn.csv").open(
            newline="", encoding="utf-8") as handle:
        ttft_rows = list(csv.DictReader(handle))
    expected_populations = {
        "turn2": EXPECTED_DIALOGUES,
        "turn3": EXPECTED_DIALOGUES,
        "pooled_t2_t3": EXPECTED_DIALOGUES * 2,
    }
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in ttft_rows:
        key = (str(row.get("method_key", "")),
               str(row.get("population", "")))
        if key in indexed:
            raise OrchestrationError(f"duplicate TTFT aggregate: {key}")
        indexed[key] = row
    expected_keys = {(method, population) for method in METHOD_KEYS
                     for population in expected_populations}
    if set(indexed) != expected_keys:
        raise OrchestrationError("TTFT T2/T3/pooled aggregate coverage failed")
    for (method, population), row in indexed.items():
        if int(row.get("n", -1)) != expected_populations[population]:
            raise OrchestrationError(
                f"TTFT population count mismatch: {method}/{population}")
        values = []
        for field in ("ttft_mean_ms", "ttft_p50_ms", "ttft_p95_ms"):
            try:
                value = float(row.get(field, "nan"))
            except (TypeError, ValueError):
                value = math.nan
            if not math.isfinite(value) or value < 0:
                raise OrchestrationError(
                    f"invalid TTFT statistic: {method}/{population}/{field}")
            values.append(value)
        if values[2] < values[1]:
            raise OrchestrationError(
                f"TTFT p95 is below p50: {method}/{population}")
    source_config = analysis_config.get("source_config", {})
    if (analysis_config.get("main_ttft_population")
            != "turns 2 and 3; stored methods cache-hit, ReComp pixels"
            or not isinstance(source_config, dict)
            or source_config.get("main_ttft_field")
            != "end_to_end_ttft_ms"):
        raise OrchestrationError("main TTFT definition/population changed")
    ours = float(indexed[("ours25", "pooled_t2_t3")]["ttft_mean_ms"])
    reductions = {}
    for baseline in ("recompute", "fullload", "qa_chunk25"):
        value = float(indexed[(baseline, "pooled_t2_t3")]["ttft_mean_ms"])
        if value <= 0:
            raise OrchestrationError("TTFT reduction baseline is not positive")
        reductions[baseline] = 100.0 * (value - ours) / value
        if not math.isfinite(reductions[baseline]):
            raise OrchestrationError("TTFT reduction is not finite")
    return {"passed": True, "required_outputs": len(REQUIRED_RESULTS),
            "results_root": str(results),
            "ttft_population_rows": len(ttft_rows),
            "ours_ttft_reduction_percent": reductions}


def acquire_lock(path: Path, message: str) -> int:
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise OrchestrationError(message)
    os.ftruncate(descriptor, 0)
    os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
    os.fsync(descriptor)
    return descriptor


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", choices=("smoke", "full"),
                        default="full")
    parser.add_argument("--shard-size", type=int)
    parser.add_argument("--min-free-after-gib", type=float)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    return parser.parse_args(argv)


def main_locked(args: argparse.Namespace) -> int:
    run, results = validate_roots(args.run_root, args.results_root)
    contract = read_index_contract(args.index)
    for path in (EVALUATOR, ANALYZER, PROTECTOR, LAUNCHER):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
    if not args.resume:
        if os.path.lexists(run) or os.path.lexists(results):
            raise FileExistsError("fresh experiment roots must be absent")
        shard_size = DEFAULT_SHARD_SIZE if args.shard_size is None \
            else int(args.shard_size)
        reserve = MIN_FREE_AFTER_GIB if args.min_free_after_gib is None \
            else float(args.min_free_after_gib)
        if not MIN_SHARD_SIZE <= shard_size <= MAX_SHARD_SIZE:
            raise ValueError("shard size must be in [40,60]")
        if reserve < MIN_FREE_AFTER_GIB:
            raise ValueError("minimum free-space reserve is 30 GiB")
        gpu = gpu_preflight()
        local_model_revision()
        protection = protection_before(run, results)
        experiment_id = str(uuid.uuid4())
        manifest = make_manifest(
            run, results, experiment_id, contract, gpu, shard_size,
            reserve, protection)
        atomic_json(run / "manifest.json", manifest, exclusive=True)
        progress: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION, "status": "running",
            "stage": "initializing", "experiment_id": experiment_id,
            "updated_at_unix": time.time(),
        }
        atomic_json(run / "progress.json", progress, exclusive=True)
    else:
        if run.is_symlink() or not run.is_dir():
            raise FileNotFoundError(run)
        manifest = read_json(run / "manifest.json")
        validate_manifest(manifest, run, results, contract)
        experiment_id = str(manifest["experiment_id"])
        shard_size = int(manifest["shard_size_images"])
        reserve = float(manifest["min_free_after_gib"])
        if args.shard_size is not None and int(args.shard_size) != shard_size:
            raise ValueError("resume shard size differs from manifest")
        if (args.min_free_after_gib is not None
                and float(args.min_free_after_gib) != reserve):
            raise ValueError("resume reserve differs from manifest")
        progress = read_json(run / "progress.json")
        if (run / "COMPLETED").is_file():
            validate_analysis(
                results, experiment_id=experiment_id, contract=contract)
            print(json.dumps({"status": "already_complete",
                              "run_root": str(run),
                              "results_root": str(results)}, indent=2))
            return 0
        if not (stage_complete(
                run, "smoke", SMOKE_DIALOGUES, experiment_id,
                contract["dialogues"], shard_size) and stage_complete(
                run, "full", EXPECTED_DIALOGUES, experiment_id,
                contract["dialogues"], shard_size)):
            local_model_revision()
            gpu_preflight()
    run_lock = acquire_lock(
        run / ".run.lock", f"another orchestrator owns {run / '.run.lock'}")
    try:
        validate_manifest(manifest, run, results, contract)
        smoke = run_stage(
            "smoke", SMOKE_DIALOGUES, run, experiment_id, contract,
            shard_size, reserve, progress)
        atomic_json(run / "smoke_validation.json", smoke)
        if args.stop_after == "smoke":
            progress.update({"status": "partial", "stage": "smoke_complete",
                             "updated_at_unix": time.time()})
            atomic_json(run / "progress.json", progress)
            print(json.dumps({"status": "smoke_complete",
                              "run_root": str(run),
                              "resume_required": True}, indent=2))
            return 0
        full = run_stage(
            "full", EXPECTED_DIALOGUES, run, experiment_id, contract,
            shard_size, reserve, progress)
        if int(full["requests"]) != FULL_REQUESTS:
            raise OrchestrationError("full logical request count is not exact")
        atomic_json(run / "full_validation.json", full)
        protection_report = ensure_protection_verified(run, results)
        protection_path = run / "protected_artifacts_validation.json"
        if not analysis_is_complete(results):
            if os.path.lexists(results):
                raise OrchestrationError(
                    "partial results root blocks atomic publication")
            run_streaming(analyzer_command(
                run / "full" / PROTOCOL, results, contract["index"],
                protection_path), run / "logs" / "analysis.log")
        analysis = validate_analysis(
            results, experiment_id=experiment_id, contract=contract)
        completed = {
            "schema_version": SCHEMA_VERSION, "passed": True,
            "experiment_id": experiment_id,
            "generated_history_requests": FULL_REQUESTS,
            "physical_requests_including_smoke": PHYSICAL_REQUESTS_WITH_SMOKE,
            "smoke": smoke, "full": full,
            "protection_manifest_sha256": protection_report[
                "after_manifest_sha256"],
            "analysis": analysis, "completed_at_unix": time.time(),
        }
        atomic_json(run / "COMPLETED", completed, exclusive=True)
        progress.update({"status": "complete", "stage": "complete",
                         "completed_at_unix": completed["completed_at_unix"],
                         "updated_at_unix": time.time()})
        atomic_json(run / "progress.json", progress)
        print(json.dumps({"status": "complete", "run_root": str(run),
                          "results_root": str(results),
                          "requests": FULL_REQUESTS}, indent=2))
        return 0
    finally:
        os.close(run_lock)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    global_lock = acquire_lock(
        GLOBAL_RUN_LOCK, "another Visual-KV GPU experiment owns the global lock")
    try:
        return main_locked(args)
    finally:
        os.close(global_lock)


if __name__ == "__main__":
    raise SystemExit(main())

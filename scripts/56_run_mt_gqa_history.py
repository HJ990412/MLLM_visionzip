#!/usr/bin/env python3
"""Orchestrate the paired MT-GQA gold/generated-history evaluation.

The orchestrator owns provenance, immutable experiment metadata, paired shard
ordering, resume, final analysis, and prior-artifact protection.  GPU work is
delegated one protocol/shard at a time to script 54; analysis is invoked once,
and only after exact coverage for both full protocols has been proven.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parent.parent
PREFIX = "mt_gqa_4arm_history_comparison_"
SCHEMA_VERSION = "mt-gqa-4arm-history-orchestrator-v2"
DEFAULT_INDEX = ROOT / "data/mt_gqa/dialogues.json"
EVALUATOR = ROOT / "scripts/54_eval_mt_gqa_history_shard.py"
ANALYZER = ROOT / "scripts/55_analyze_mt_gqa_history.py"
PROTECTOR = ROOT / "scripts/57_protect_mt_gqa_history_artifacts.py"
LAUNCHER = ROOT / "scripts/58_launch_mt_gqa_history.sh"
RUNTIME_CODE_PATHS = (
    ROOT / "scripts/37_eval_mt_gqa_full_shard.py",
    ROOT / "scripts/49_eval_query_aware_baseline.py",
    EVALUATOR,
    ANALYZER,
    Path(__file__).resolve(),
    PROTECTOR,
    LAUNCHER,
    ROOT / "mmimpress/__init__.py",
    ROOT / "mmimpress/config.py",
    ROOT / "mmimpress/cvpr25.py",
    ROOT / "mmimpress/dataset.py",
    ROOT / "mmimpress/model.py",
    ROOT / "mmimpress/mt_gqa.py",
    ROOT / "mmimpress/piggyback.py",
    ROOT / "mmimpress/reorder.py",
    ROOT / "mmimpress/serve.py",
    ROOT / "mmimpress/sparsevlm.py",
    ROOT / "mmimpress/store.py",
)
RUNTIME_PACKAGES = (
    "torch", "transformers", "Pillow", "numpy", "psutil",
    "bitsandbytes", "accelerate",
)
MODEL_REF = Path(
    "/home/dblab/.cache/huggingface/hub/"
    "models--llava-hf--llava-v1.6-vicuna-7b-hf/refs/main")

EXPECTED_INDEX_SHA256 = (
    "2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224")
EXPECTED_WORKLOAD_SHA256 = (
    "0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62")
EXPECTED_MODEL_REVISION = "c916e6cdcd760b4cecd1dd4907f84ac649f93b23"
QA_RATER_ALGORITHM_ID = "sparsevlm_visual_text_mean_threshold_v1"
EXPECTED_QA_CONFIGURATION = {
    "physical_layout": "raster", "head_reduce": "mean",
    "chunk_aggregation": "mean_valid_spatial_tokens",
    "normal_chunk_budget": 0.25,
    "budget_helper": "budget_chunk_count_round",
    "rater_algorithm_id": QA_RATER_ALGORITHM_ID,
    "rater_scope": "entire_available_causal_suffix",
    "fallback": False, "adaptive_budget": False,
}
EXPECTED_OURS_CONFIGURATION = {
    "physical_layout": "visionzip_image_only",
    "normal_chunk_budget": 0.25,
    "selection": "fixed_first_k_prefix",
    "online_query_scoring": False,
}
EXPECTED_DIALOGUES = 4_061
TURNS_PER_DIALOGUE = 3
EXPECTED_TURNS = EXPECTED_DIALOGUES * TURNS_PER_DIALOGUE
EXPECTED_IMAGES = 398
METHOD_KEYS = ("recompute", "fullload", "qa_chunk25", "ours25")
PROTOCOLS = ("gold_history", "generated_history")
REQUESTS_PER_PROTOCOL = EXPECTED_TURNS * len(METHOD_KEYS)
REQUESTS_BOTH_PROTOCOLS = REQUESTS_PER_PROTOCOL * len(PROTOCOLS)
SMOKE_DIALOGUES = 10
SMOKE_REQUESTS_PER_PROTOCOL = SMOKE_DIALOGUES * TURNS_PER_DIALOGUE * len(METHOD_KEYS)
SEED = 1234
MAX_NEW_TOKENS = 16
DEFAULT_SHARD_SIZE = 50
MIN_SHARD_SIZE = 40
MAX_SHARD_SIZE = 60
MIN_FREE_AFTER_GIB = 30.0
MIN_GPU_FREE_MIB = 20_000
GLOBAL_RUN_LOCK = Path("/tmp/mllm_v2_mt_gqa_history_run.lock")


class OrchestrationError(RuntimeError):
    """The durable experiment contract is incomplete or inconsistent."""


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def runtime_code_hashes() -> dict[str, str]:
    """Hash every local module that can affect evaluation or publication."""
    hashes: dict[str, str] = {}
    for path in RUNTIME_CODE_PATHS:
        resolved = path.resolve()
        if path.is_symlink() or not path.is_file():
            raise OrchestrationError(
                f"runtime dependency is not a regular file: {path}")
        try:
            key = resolved.relative_to(ROOT.resolve()).as_posix()
        except ValueError as error:
            raise OrchestrationError(
                f"runtime dependency escapes project root: {resolved}") from error
        if key in hashes:
            raise OrchestrationError(f"duplicate runtime dependency: {key}")
        hashes[key] = sha256_file(resolved)
    return hashes


def runtime_environment() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for package in RUNTIME_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise OrchestrationError(
                f"required runtime package is unavailable: {package}") from error
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "packages": versions,
    }


def atomic_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and os.path.lexists(path):
        raise FileExistsError(f"refusing to replace {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
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


def validate_output_roots(
    run_root: Path | str, results_root: Path | str,
) -> tuple[Path, Path]:
    run = Path(run_root)
    results = Path(results_root)
    if not run.is_absolute():
        run = ROOT / run
    if not results.is_absolute():
        results = ROOT / results
    run = run.resolve(strict=False)
    results = results.resolve(strict=False)
    expected_run_parent = (ROOT / "runs").resolve()
    expected_results_parent = (ROOT / "results").resolve()
    if run.parent != expected_run_parent or not run.name.startswith(PREFIX):
        raise ValueError(f"invalid dedicated run root: {run}")
    if (results.parent != expected_results_parent
            or not results.name.startswith(PREFIX)):
        raise ValueError(f"invalid dedicated results root: {results}")
    if run.name != results.name:
        raise ValueError("run/results root names must match")
    for path, label in ((run, "run"), (results, "results")):
        if os.path.lexists(path) and path.is_symlink():
            raise ValueError(f"{label} root may not be a symlink: {path}")
    return run, results


def _load_dialogues(index: Path) -> list[dict[str, Any]]:
    value = json.loads(index.read_text(encoding="utf-8"))
    rows = value.get("dialogues") if isinstance(value, dict) else value
    if not isinstance(rows, list):
        raise ValueError("MT-GQA index has no dialogues list")
    return rows


def read_index_contract(index: Path = DEFAULT_INDEX) -> dict[str, Any]:
    index = index.resolve()
    if index != DEFAULT_INDEX.resolve() or index.is_symlink() or not index.is_file():
        raise ValueError(f"canonical index is fixed to {DEFAULT_INDEX}")
    digest = sha256_file(index)
    if digest != EXPECTED_INDEX_SHA256:
        raise ValueError(f"canonical index SHA256 mismatch: {digest}")
    dialogues = _load_dialogues(index)
    if len(dialogues) != EXPECTED_DIALOGUES:
        raise ValueError(f"dialogue count {len(dialogues)} != {EXPECTED_DIALOGUES}")
    if any(len(row.get("turns", [])) != TURNS_PER_DIALOGUE for row in dialogues):
        raise ValueError("canonical index contains a non-three-turn dialogue")
    ids = [str(row.get("dialog_id", row.get("dialogue_id", "")))
           for row in dialogues]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("canonical index dialogue IDs are empty or duplicated")
    payload = "".join(
        f"{ids[i]}\t{int(turn['turn_id'])}\t{turn['question_id']}\n"
        for i, row in enumerate(dialogues) for turn in row["turns"]
    ).encode("utf-8")
    workload = hashlib.sha256(payload).hexdigest()
    if workload != EXPECTED_WORKLOAD_SHA256:
        raise ValueError(f"canonical workload SHA256 mismatch: {workload}")
    image_ids = [str(row["image_id"]) for row in dialogues]
    if len(set(image_ids)) != EXPECTED_IMAGES:
        raise ValueError("canonical workload image count mismatch")
    return {
        "index": index,
        "index_sha256": digest,
        "workload_sha256": workload,
        "dialogues": dialogues,
        "dialogue_ids": ids,
        "unique_images": len(set(image_ids)),
    }


def selected_image_count(dialogues: Sequence[Mapping[str, Any]], limit: int) -> int:
    return len({str(row["image_id"]) for row in dialogues[:limit]})


def local_model_revision() -> str:
    if MODEL_REF.is_symlink() or not MODEL_REF.is_file():
        raise FileNotFoundError(MODEL_REF)
    revision = MODEL_REF.read_text(encoding="utf-8").strip()
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError(f"invalid local model revision: {revision!r}")
    if revision != EXPECTED_MODEL_REVISION:
        raise ValueError(
            f"local model revision {revision} != frozen {EXPECTED_MODEL_REVISION}")
    snapshot = MODEL_REF.parent.parent / "snapshots" / revision
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise FileNotFoundError(snapshot)
    return revision


def gpu_preflight() -> dict[str, Any]:
    command = [
        "nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,"
        "utilization.gpu,driver_version", "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(
            command, check=True, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as error:
        raise OrchestrationError(f"GPU preflight failed: {error}") from error
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise OrchestrationError(f"expected exactly one GPU, observed {len(rows)}")
    fields = [part.strip() for part in rows[0].split(",")]
    if len(fields) != 6:
        raise OrchestrationError(f"malformed nvidia-smi output: {rows[0]!r}")
    try:
        total, used, free, utilization = map(int, fields[1:5])
    except ValueError as error:
        raise OrchestrationError(f"malformed numeric GPU fields: {fields}") from error
    if free < MIN_GPU_FREE_MIB:
        raise OrchestrationError(
            f"GPU has only {free} MiB free; require {MIN_GPU_FREE_MIB} MiB")
    if utilization > 10 or used > total - MIN_GPU_FREE_MIB:
        raise OrchestrationError(
            f"GPU is not idle enough (used={used} MiB, utilization={utilization}%)")
    return {"name": fields[0], "memory_total_mib": total,
            "memory_used_mib": used, "memory_free_mib": free,
            "utilization_gpu_pct": utilization, "driver_version": fields[5]}


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
        returncode = process.wait()
        os.fsync(log.fileno())
    if returncode:
        raise subprocess.CalledProcessError(returncode, list(command))
    return time.monotonic() - started


def build_evaluator_command(
    *, protocol: str, stage_dir: Path, temp_root: Path,
    experiment_id: str, shard_index: int, shard_size: int,
    dialogue_limit: int, min_free_after_gib: float,
    index: Path = DEFAULT_INDEX,
) -> list[str]:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol: {protocol}")
    if dialogue_limit not in (SMOKE_DIALOGUES, EXPECTED_DIALOGUES):
        raise ValueError("dialogue limit must be smoke=10 or full=4,061")
    command = [
        sys.executable, str(EVALUATOR),
        "--protocol", protocol,
        "--index", str(index.resolve()),
        "--run-dir", str(stage_dir.resolve()),
        "--temp-root", str(temp_root.resolve()),
        "--experiment-id", experiment_id,
        "--shard-index", str(int(shard_index)),
        "--shard-size", str(int(shard_size)),
        "--expected-index-sha256", EXPECTED_INDEX_SHA256,
        "--expected-workload-sha256", EXPECTED_WORKLOAD_SHA256,
        "--expected-dialogs", str(int(dialogue_limit)),
        "--seed", str(SEED),
        "--max-new-tokens", str(MAX_NEW_TOKENS),
        "--min-free-after-gib", str(float(min_free_after_gib)),
    ]
    if dialogue_limit != EXPECTED_DIALOGUES:
        command.extend([
            "--max-dialogs", str(int(dialogue_limit)),
            "--allow-partial-workload",
        ])
    return command


def build_analyzer_command(
    *, gold_run_dir: Path, generated_run_dir: Path,
    results_root: Path, protection_validation: Path,
    index: Path = DEFAULT_INDEX,
) -> list[str]:
    return [
        sys.executable, str(ANALYZER),
        "--gold-run-dir", str(gold_run_dir.resolve()),
        "--generated-run-dir", str(generated_run_dir.resolve()),
        "--results-root", str(results_root.resolve()),
        "--index", str(index.resolve()),
        "--protection-validation", str(protection_validation.resolve()),
        "--expected-dialogs", str(EXPECTED_DIALOGUES),
        "--expected-images", str(EXPECTED_IMAGES),
    ]


def _protection_command(mode: str, run: Path, results: Path) -> list[str]:
    if mode not in ("before", "verify"):
        raise ValueError(mode)
    return [sys.executable, str(PROTECTOR), f"--{mode}",
            "--run-root", str(run), "--results-root", str(results)]


def protection_before(run: Path, results: Path) -> dict[str, Any]:
    completed = subprocess.run(
        _protection_command("before", run, results), cwd=ROOT, check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise OrchestrationError("protector emitted malformed before JSON") from error
    if value.get("status") != "recorded":
        raise OrchestrationError(f"protector did not record snapshot: {value}")
    return value


def protection_verify(run: Path, results: Path) -> dict[str, Any]:
    completed = subprocess.run(
        _protection_command("verify", run, results), cwd=ROOT, check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise OrchestrationError("protector emitted malformed verify JSON") from error
    if value.get("status") != "unchanged":
        raise OrchestrationError(f"protector did not verify snapshot: {value}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise OrchestrationError(f"required JSON is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise OrchestrationError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise OrchestrationError(f"JSON root is not an object: {path}")
    return value


def make_manifest(
    *, run: Path, results: Path, experiment_id: str,
    contract: Mapping[str, Any], model_revision: str,
    gpu: Mapping[str, Any], shard_size: int, min_free_after_gib: float,
    protection: Mapping[str, Any],
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "benchmark_type": "MT-GQA-reconstructed",
        "exact_metacompress_reproduction_claimed": False,
        "reporting_style": "MetaCompress-compatible Acc1/Acc2/Acc3/Avg",
        "run_root": str(run),
        "results_root": str(results),
        "index": str(contract["index"]),
        "index_sha256": contract["index_sha256"],
        "workload_sha256": contract["workload_sha256"],
        "dialogues": EXPECTED_DIALOGUES,
        "turns_per_protocol": EXPECTED_TURNS,
        "images": EXPECTED_IMAGES,
        "methods": list(METHOD_KEYS),
        "protocols": list(PROTOCOLS),
        "logical_request_counts": {
            "gold_history": REQUESTS_PER_PROTOCOL,
            "generated_history": REQUESTS_PER_PROTOCOL,
            "both_protocols": REQUESTS_BOTH_PROTOCOLS,
            "cache_hit_per_protocol": EXPECTED_DIALOGUES * 2 * len(METHOD_KEYS),
            "cache_hit_both_protocols": (
                EXPECTED_DIALOGUES * 2 * len(METHOD_KEYS) * len(PROTOCOLS)),
            "turn1_per_protocol": EXPECTED_DIALOGUES * len(METHOD_KEYS),
            "turn1_both_protocols": (
                EXPECTED_DIALOGUES * len(METHOD_KEYS) * len(PROTOCOLS)),
            "reported_full_population": REQUESTS_BOTH_PROTOCOLS,
            "staged_validation_outside_full": (
                SMOKE_REQUESTS_PER_PROTOCOL * len(PROTOCOLS)),
            "physical_gpu_requests_including_smoke": (
                REQUESTS_BOTH_PROTOCOLS
                + SMOKE_REQUESTS_PER_PROTOCOL * len(PROTOCOLS)),
        },
        "smoke": {
            "dialogues_per_protocol": SMOKE_DIALOGUES,
            "requests_per_protocol": SMOKE_REQUESTS_PER_PROTOCOL,
            "both_protocols": SMOKE_REQUESTS_PER_PROTOCOL * len(PROTOCOLS),
        },
        "execution_order": (
            "smoke paired by image shard: gold then generated; full paired "
            "by image shard: gold then generated; protect verify; analysis"),
        "protocol_state_policy": "fully independent evaluator invocations and temp roots",
        "history_policy": {
            "gold_history": "teacher-forced gold A1/A2",
            "generated_history": "same-method prior predictions only",
        },
        "shard_size_images": int(shard_size),
        "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "min_free_after_gib": float(min_free_after_gib),
        "model": "llava-hf/llava-v1.6-vicuna-7b-hf",
        "model_revision": model_revision,
        "gpu_preflight": dict(gpu),
        "protection_before": dict(protection),
        "code_sha256": runtime_code_hashes(),
        "runtime_environment": runtime_environment(),
        "created_at_unix": time.time(),
    }
    value["manifest_sha256"] = canonical_hash(value)
    return value


def validate_manifest(
    manifest: Mapping[str, Any], *, run: Path, results: Path,
    contract: Mapping[str, Any],
) -> None:
    copy = dict(manifest)
    recorded = copy.pop("manifest_sha256", None)
    if recorded != canonical_hash(copy):
        raise OrchestrationError("immutable manifest content hash mismatch")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "run_root": str(run),
        "results_root": str(results),
        "index_sha256": contract["index_sha256"],
        "workload_sha256": contract["workload_sha256"],
        "dialogues": EXPECTED_DIALOGUES,
        "turns_per_protocol": EXPECTED_TURNS,
        "images": EXPECTED_IMAGES,
        "methods": list(METHOD_KEYS),
        "protocols": list(PROTOCOLS),
        "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    mismatch = {key: (manifest.get(key), value) for key, value in expected.items()
                if manifest.get(key) != value}
    counts = manifest.get("logical_request_counts", {})
    if counts.get("gold_history") != REQUESTS_PER_PROTOCOL:
        mismatch["gold_history requests"] = (
            counts.get("gold_history"), REQUESTS_PER_PROTOCOL)
    if counts.get("generated_history") != REQUESTS_PER_PROTOCOL:
        mismatch["generated_history requests"] = (
            counts.get("generated_history"), REQUESTS_PER_PROTOCOL)
    if counts.get("both_protocols") != REQUESTS_BOTH_PROTOCOLS:
        mismatch["both protocol requests"] = (
            counts.get("both_protocols"), REQUESTS_BOTH_PROTOCOLS)
    physical_expected = (REQUESTS_BOTH_PROTOCOLS
                         + SMOKE_REQUESTS_PER_PROTOCOL * len(PROTOCOLS))
    if counts.get("physical_gpu_requests_including_smoke") != physical_expected:
        mismatch["physical requests including smoke"] = (
            counts.get("physical_gpu_requests_including_smoke"),
            physical_expected)
    current_code = runtime_code_hashes()
    if manifest.get("code_sha256") != current_code:
        mismatch["code_sha256"] = (manifest.get("code_sha256"), current_code)
    current_environment = runtime_environment()
    if manifest.get("runtime_environment") != current_environment:
        mismatch["runtime_environment"] = (
            manifest.get("runtime_environment"), current_environment)
    if mismatch:
        raise OrchestrationError(f"immutable manifest mismatch: {mismatch}")


def _marker_path(stage_dir: Path, shard_index: int) -> Path:
    return stage_dir / "shards" / f"shard_{shard_index:03d}.json"


def marker_is_complete(
    path: Path, *, experiment_id: str, protocol: str,
    shard_index: int, shard_size: int,
) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(marker, dict):
            return False
        image_ids = [str(value) for value in marker.get("image_ids", [])]
        completed_ids = [str(value) for value in marker.get(
            "completed_image_ids", [])]
        file_hashes = marker.get("image_artifact_file_sha256")
        content_hashes = marker.get("image_artifact_content_sha256")
        if (marker.get("schema_version") != "mt-gqa-4arm-history-shard-v2"
                or marker.get("complete") is not True
                or marker.get("experiment_id") != experiment_id
                or marker.get("protocol") != protocol
                or int(marker.get("shard_index", -1)) != int(shard_index)
                or int(marker.get("shard_size", -1)) != int(shard_size)
                or not image_ids or completed_ids != image_ids
                or len(set(image_ids)) != len(image_ids)
                or not isinstance(file_hashes, dict)
                or set(file_hashes) != set(image_ids)
                or not isinstance(content_hashes, dict)
                or set(content_hashes) != set(image_ids)
                or marker.get("artifact_content_sha256") != canonical_hash({
                    key: value for key, value in marker.items()
                    if key != "artifact_content_sha256"})):
            return False
        image_dir = path.parent.parent / "images"
        for image_id in image_ids:
            if not image_id or Path(image_id).name != image_id:
                return False
            artifact_path = image_dir / f"{image_id}.json"
            if (not artifact_path.is_file() or artifact_path.is_symlink()
                    or sha256_file(artifact_path) != file_hashes[image_id]):
                return False
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            if (not isinstance(artifact, dict)
                    or artifact.get("experiment_id") != experiment_id
                    or artifact.get("protocol") != protocol
                    or artifact.get("image_id") != image_id
                    or int(artifact.get("shard_index", -1)) != int(shard_index)
                    or artifact.get("artifact_content_sha256")
                    != content_hashes[image_id]
                    or artifact.get("artifact_content_sha256") != canonical_hash({
                        key: value for key, value in artifact.items()
                        if key != "artifact_content_sha256"})):
                return False
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def validate_protocol_stage(
    *, stage_dir: Path, protocol: str, experiment_id: str,
    dialogues: Sequence[Mapping[str, Any]], dialogue_limit: int,
    shard_size: int,
) -> dict[str, Any]:
    if protocol not in PROTOCOLS:
        raise ValueError(protocol)
    selected = list(dialogues[:dialogue_limit])
    expected_dialog_ids = {
        str(row.get("dialog_id", row.get("dialogue_id", "")))
        for row in selected}
    expected_image_ids = {str(row["image_id"]) for row in selected}
    expected_rows = dialogue_limit * TURNS_PER_DIALOGUE * len(METHOD_KEYS)
    expected_shards = math.ceil(len(expected_image_ids) / shard_size)

    config = _read_json(stage_dir / "config.json")
    config_expected = {
        "schema_version": "mt-gqa-4arm-history-shard-v2",
        "experiment_id": experiment_id,
        "protocol": protocol,
        "dialogues_file_sha256": EXPECTED_INDEX_SHA256,
        "source_full_workload_sha256": EXPECTED_WORKLOAD_SHA256,
        "n_dialogs": dialogue_limit,
        "n_turns": dialogue_limit * TURNS_PER_DIALOGUE,
        "n_images": len(expected_image_ids),
        "n_requests": expected_rows,
        "shard_size": shard_size,
        "n_shards": expected_shards,
        "seed": SEED,
        "method_keys": list(METHOD_KEYS),
        "model": "llava-hf/llava-v1.6-vicuna-7b-hf",
        "model_revision": EXPECTED_MODEL_REVISION,
        "load_4bit": True,
        "quantization": "4-bit NF4 double-quant",
        "compute_dtype": "bfloat16",
        "attention": "eager",
        "decoding": "greedy",
        "max_new_tokens": MAX_NEW_TOKENS,
        "chunk_size": 64,
        "probe_heads": 3,
        "qa_chunk_configuration": EXPECTED_QA_CONFIGURATION,
        "ours_configuration": EXPECTED_OURS_CONFIGURATION,
    }
    mismatch = {key: (config.get(key), value)
                for key, value in config_expected.items()
                if config.get(key) != value}
    if mismatch:
        raise OrchestrationError(
            f"{protocol} stage config mismatch at {stage_dir}: {mismatch}")
    if ("FullLoad captures/persists" not in str(config.get("turn1_policy", ""))
            or "captured by FullLoad's own T1" not in str(
                config.get("qa_raster_source_policy", ""))):
        raise OrchestrationError(
            f"{protocol} stage does not prove FullLoad-own-T1 raster capture")

    image_dir = stage_dir / "images"
    image_paths = sorted(image_dir.glob("*.json"))
    if len(image_paths) != len(expected_image_ids):
        raise OrchestrationError(
            f"{protocol} image artifact count {len(image_paths)} != "
            f"{len(expected_image_ids)}")
    observed_images: set[str] = set()
    observed_dialogs: set[str] = set()
    logical_ids: set[str] = set()
    observed_rows = 0
    failed = 0
    for path in image_paths:
        artifact = _read_json(path)
        image_id = str(artifact.get("image_id", ""))
        if (not image_id or image_id != path.stem
                or image_id in observed_images or image_id not in expected_image_ids):
            raise OrchestrationError(f"invalid/duplicate image artifact: {path}")
        if (artifact.get("experiment_id") != experiment_id
                or artifact.get("protocol") != protocol
                or artifact.get("validation", {}).get("passed") is not True):
            raise OrchestrationError(f"unvalidated image artifact: {path}")
        rows = artifact.get("rows")
        if not isinstance(rows, list):
            raise OrchestrationError(f"image artifact omits rows: {path}")
        artifact_dialogs = {
            str(value) for value in artifact.get("dialog_ids", [])}
        if not artifact_dialogs:
            artifact_dialogs = {str(row.get("dialog_id", "")) for row in rows}
        if (not artifact_dialogs or "" in artifact_dialogs
                or observed_dialogs.intersection(artifact_dialogs)):
            raise OrchestrationError(f"invalid/reused dialogues in {path}")
        if len(rows) != len(artifact_dialogs) * TURNS_PER_DIALOGUE * len(METHOD_KEYS):
            raise OrchestrationError(f"row count mismatch in {path}")
        per_turn: dict[tuple[str, int], list[str]] = {}
        for row in rows:
            did = str(row.get("dialog_id", ""))
            turn = int(row.get("turn_id", -1))
            method = str(row.get("method_key", ""))
            logical = str(row.get("logical_request_id", ""))
            if (did not in artifact_dialogs or turn not in (1, 2, 3)
                    or method not in METHOD_KEYS or not logical
                    or row.get("protocol") != protocol):
                raise OrchestrationError(f"invalid request identity in {path}")
            if logical in logical_ids:
                raise OrchestrationError(f"duplicate logical request: {logical}")
            logical_ids.add(logical)
            per_turn.setdefault((did, turn), []).append(method)
            failed += int(row.get("status") != "ok")
        if any(sorted(methods) != sorted(METHOD_KEYS)
               for methods in per_turn.values()):
            raise OrchestrationError(f"four-arm coverage mismatch in {path}")
        observed_rows += len(rows)
        observed_images.add(image_id)
        observed_dialogs.update(artifact_dialogs)
    if observed_images != expected_image_ids:
        raise OrchestrationError(f"{protocol} image membership mismatch")
    if observed_dialogs != expected_dialog_ids:
        raise OrchestrationError(f"{protocol} dialogue membership mismatch")
    if observed_rows != expected_rows or len(logical_ids) != expected_rows or failed:
        raise OrchestrationError(
            f"{protocol} request coverage failed: rows={observed_rows}, "
            f"unique={len(logical_ids)}, failed={failed}, expected={expected_rows}")

    marker_paths = sorted((stage_dir / "shards").glob("shard_*.json"))
    if len(marker_paths) != expected_shards:
        raise OrchestrationError(
            f"{protocol} shard marker count {len(marker_paths)} != {expected_shards}")
    marked_images: list[str] = []
    for index, path in enumerate(marker_paths):
        if path != _marker_path(stage_dir, index) or not marker_is_complete(
                path, experiment_id=experiment_id, protocol=protocol,
                shard_index=index, shard_size=shard_size):
            raise OrchestrationError(f"invalid shard marker: {path}")
        marker = _read_json(path)
        marked_images.extend(str(value) for value in marker.get("image_ids", []))
    if len(marked_images) != len(set(marked_images)) \
            or set(marked_images) != expected_image_ids:
        raise OrchestrationError(f"{protocol} shard partition mismatch")
    return {
        "passed": True, "protocol": protocol,
        "dialogues": len(observed_dialogs),
        "turns": len(observed_dialogs) * TURNS_PER_DIALOGUE,
        "images": len(observed_images),
        "requests": observed_rows, "failed": failed, "duplicates": 0,
        "shards": expected_shards,
    }


def validate_cross_protocol_turn1(
    gold_dir: Path, generated_dir: Path,
) -> dict[str, Any]:
    gold_paths = {path.name: path for path in (gold_dir / "images").glob("*.json")}
    generated_paths = {
        path.name: path for path in (generated_dir / "images").glob("*.json")}
    if set(gold_paths) != set(generated_paths):
        raise OrchestrationError("cross-protocol image artifacts differ")
    compared = 0
    mismatches: list[str] = []
    fields = ("question", "prompt", "prompt_sha256", "prediction",
              "first_token_id", "input_token_count", "generated_token_count")
    for name in sorted(gold_paths):
        gold_rows = _read_json(gold_paths[name]).get("rows", [])
        generated_rows = _read_json(generated_paths[name]).get("rows", [])
        gold_t1 = {(str(row["dialog_id"]), str(row["method_key"])): row
                   for row in gold_rows if int(row["turn_id"]) == 1}
        generated_t1 = {
            (str(row["dialog_id"]), str(row["method_key"])): row
            for row in generated_rows if int(row["turn_id"]) == 1}
        if set(gold_t1) != set(generated_t1):
            raise OrchestrationError(f"Turn-1 request keys differ for {name}")
        for key, gold in gold_t1.items():
            generated = generated_t1[key]
            compared += 1
            if any(gold.get(field) != generated.get(field) for field in fields):
                mismatches.append(f"{key[0]}:{key[1]}")
    if mismatches:
        raise OrchestrationError(
            f"cross-protocol Turn-1 mismatch ({len(mismatches)}): "
            + ", ".join(mismatches[:20]))
    return {"passed": True, "compared": compared, "mismatches": 0,
            "prediction_agreement": 1.0, "first_token_agreement": 1.0,
            "prompt_agreement": 1.0}


def ensure_temp_clean(
    temp_root: Path, *, experiment_id: str, protocol: str,
) -> dict[str, Any]:
    if not os.path.lexists(temp_root):
        return {"passed": True, "exists": False, "files": 0}
    if temp_root.is_symlink() or not temp_root.is_dir():
        raise OrchestrationError(f"temporary root is unsafe: {temp_root}")
    owner_path = temp_root / ".mt_gqa_temp_store_owner.json"
    payload = temp_root / "payload"
    if (owner_path.is_symlink() or not owner_path.is_file()
            or payload.is_symlink() or not payload.is_dir()):
        raise OrchestrationError("temporary ownership structure is invalid")
    owner = _read_json(owner_path)
    expected_owner = {
        "experiment_id": experiment_id,
        "dataset": "gqa_testdev_balanced_mt3",
        "purpose": "temporary_visual_kv_only",
    }
    mismatch = {key: (owner.get(key), value)
                for key, value in expected_owner.items()
                if owner.get(key) != value}
    if mismatch:
        raise OrchestrationError(f"temporary ownership mismatch: {mismatch}")
    payload_entries = list(payload.iterdir())
    if payload_entries:
        raise OrchestrationError(
            "temporary Visual-KV payload leaves remain: "
            + ", ".join(path.name for path in payload_entries[:20]))
    unexpected = [path.name for path in temp_root.iterdir()
                  if path.name not in {owner_path.name, payload.name}]
    if unexpected:
        raise OrchestrationError(
            "unexpected temporary-root entries: " + ", ".join(unexpected))
    payload.rmdir()
    owner_path.unlink()
    temp_root.rmdir()
    return {"passed": True, "exists": True, "files": 0}


def run_paired_stage(
    *, stage_name: str, dialogue_limit: int, run: Path,
    experiment_id: str, contract: Mapping[str, Any], shard_size: int,
    min_free_after_gib: float, progress: dict[str, Any],
) -> dict[str, Any]:
    image_count = selected_image_count(contract["dialogues"], dialogue_limit)
    n_shards = math.ceil(image_count / shard_size)
    started = time.monotonic()
    launched = 0
    for shard_index in range(n_shards):
        # Pair the two protocol invocations by image shard so hardware drift is
        # bounded while their histories and temporary stores remain isolated.
        for protocol in PROTOCOLS:
            stage_dir = run / stage_name / protocol
            temp_root = run / "_temporary_visual_kv" / stage_name / protocol
            marker = _marker_path(stage_dir, shard_index)
            if marker_is_complete(
                    marker, experiment_id=experiment_id, protocol=protocol,
                    shard_index=shard_index, shard_size=shard_size):
                continue
            command = build_evaluator_command(
                protocol=protocol, stage_dir=stage_dir, temp_root=temp_root,
                experiment_id=experiment_id, shard_index=shard_index,
                shard_size=shard_size, dialogue_limit=dialogue_limit,
                min_free_after_gib=min_free_after_gib,
                index=contract["index"])
            run_streaming(
                command, run / "logs" / stage_name /
                f"{protocol}_shard_{shard_index:03d}.log")
            launched += 1
            if not marker_is_complete(
                    marker, experiment_id=experiment_id, protocol=protocol,
                    shard_index=shard_index, shard_size=shard_size):
                raise OrchestrationError(
                    f"evaluator returned without complete marker: {marker}")
            progress.update({
                "status": "running", "stage": stage_name,
                "last_protocol": protocol, "last_shard_index": shard_index,
                "updated_at_unix": time.time()})
            atomic_json(run / "progress.json", progress)
    validations = {
        protocol: validate_protocol_stage(
            stage_dir=run / stage_name / protocol, protocol=protocol,
            experiment_id=experiment_id, dialogues=contract["dialogues"],
            dialogue_limit=dialogue_limit, shard_size=shard_size)
        for protocol in PROTOCOLS
    }
    cross = validate_cross_protocol_turn1(
        run / stage_name / "gold_history",
        run / stage_name / "generated_history")
    temp = {
        protocol: ensure_temp_clean(
            run / "_temporary_visual_kv" / stage_name / protocol,
            experiment_id=experiment_id, protocol=protocol)
        for protocol in PROTOCOLS
    }
    return {
        "passed": True, "stage": stage_name,
        "dialogues_per_protocol": dialogue_limit,
        "requests_per_protocol": (
            dialogue_limit * TURNS_PER_DIALOGUE * len(METHOD_KEYS)),
        "requests_both_protocols": (
            dialogue_limit * TURNS_PER_DIALOGUE * len(METHOD_KEYS)
            * len(PROTOCOLS)),
        "images": image_count, "shards": n_shards,
        "protocols": validations, "cross_protocol_turn1": cross,
        "temporary_cleanup": temp, "subprocesses_launched": launched,
        "wall_seconds_this_invocation": time.monotonic() - started,
    }


def _expected_marker_count(
    dialogues: Sequence[Mapping[str, Any]], dialogue_limit: int,
    shard_size: int,
) -> int:
    return math.ceil(selected_image_count(dialogues, dialogue_limit) / shard_size)


def stage_markers_complete(
    *, run: Path, stage_name: str, dialogue_limit: int,
    experiment_id: str, dialogues: Sequence[Mapping[str, Any]],
    shard_size: int,
) -> bool:
    count = _expected_marker_count(dialogues, dialogue_limit, shard_size)
    return all(marker_is_complete(
        _marker_path(run / stage_name / protocol, index),
        experiment_id=experiment_id, protocol=protocol,
        shard_index=index, shard_size=shard_size)
        for protocol in PROTOCOLS for index in range(count))


def ensure_protection_verified(run: Path, results: Path) -> dict[str, Any]:
    path = run / "protected_artifacts_validation.json"
    manifest = _read_json(run / "manifest.json")
    declared = manifest.get("protection_before")
    if not isinstance(declared, Mapping):
        raise OrchestrationError("run manifest omits protection-before binding")
    before = _read_json(run / "protected_artifacts_before.json")
    expected_before = {
        "status": "recorded",
        "manifest": str(run / "protected_artifacts_before.json"),
        "manifest_sha256": before.get("manifest_sha256"),
        "entry_count": before.get("entry_count"),
        "file_count": before.get("file_count"),
        "total_bytes": before.get("total_bytes"),
    }
    if dict(declared) != expected_before:
        raise OrchestrationError(
            "manifest protection binding differs from before snapshot")
    if path.exists():
        report = _read_json(path)
        if (report.get("passed") is not True
                or report.get("before_manifest_sha256")
                != before.get("manifest_sha256")
                or report.get("after_manifest_sha256")
                != before.get("manifest_sha256")):
            raise OrchestrationError("existing artifact-protection report is invalid")
        return report
    protection_verify(run, results)
    report = _read_json(path)
    if (report.get("passed") is not True
            or report.get("before_manifest_sha256")
            != before.get("manifest_sha256")
            or report.get("after_manifest_sha256")
            != before.get("manifest_sha256")):
        raise OrchestrationError("prior-artifact protection failed")
    return report


def analysis_is_complete(results: Path) -> bool:
    if not results.is_dir() or results.is_symlink():
        return False
    marker = results / "COMPLETED"
    if not marker.is_file() or marker.is_symlink():
        return False
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (not isinstance(value, dict) or value.get("passed") is not True
            or int(value.get("logical_rows", -1)) != REQUESTS_BOTH_PROTOCOLS):
        return False
    hashes = value.get("output_sha256")
    if not isinstance(hashes, dict) or not hashes:
        return False
    observed: dict[str, str] = {}
    try:
        for path in sorted(results.rglob("*")):
            if path == marker:
                continue
            if path.is_symlink():
                return False
            if path.is_file():
                observed[path.relative_to(results).as_posix()] = sha256_file(path)
    except OSError:
        return False
    return hashes == observed


def required_analysis_paths(results: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for protocol in PROTOCOLS:
        directory = results / protocol
        paths.extend(directory / name for name in (
            "raw.jsonl", "raw.jsonl.gz", "summary.csv",
            "quality_by_turn.csv", "ttft_by_turn.csv", "token_lengths.csv",
            "io_breakdown.csv", "selection_analysis.json", "validation.json",
            "config.json",
        ))
    comparison = results / "comparison"
    paths.extend(comparison / name for name in (
        "history_comparison.csv", "error_propagation.csv",
        "error_propagation.json", "paired_quality.json",
        "selection_analysis.json", "manual_history_samples.json",
        "validation.json", "ANALYSIS.md", "README.md",
    ))
    paths.extend(results / name for name in (
        "config.json", "validation.json", "README.md", "COMPLETED"))
    return tuple(paths)


def validate_analysis(results: Path) -> dict[str, Any]:
    if not analysis_is_complete(results):
        raise OrchestrationError(f"analysis publication is incomplete: {results}")
    required = required_analysis_paths(results)
    missing = [str(path) for path in required if not path.is_file() or path.is_symlink()]
    if missing:
        raise OrchestrationError(f"analysis omits required outputs: {missing}")
    completion = _read_json(results / "COMPLETED")
    validation = _read_json(results / "validation.json")
    if (validation.get("passed") is not True
            or int(validation.get("observed_logical_rows", -1))
            != REQUESTS_BOTH_PROTOCOLS
            or completion.get("validation_sha256")
            != sha256_file(results / "validation.json")
            or completion.get("analysis_sha256")
            != sha256_file(results / "comparison" / "ANALYSIS.md")):
        raise OrchestrationError("analysis completion/validation contract failed")
    return {"passed": True, "results_root": str(results),
            "required_outputs": len(required)}


def _write_text_exclusive(path: Path, text: str) -> None:
    if os.path.lexists(path):
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
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


def _acquire_run_lock(run: Path):
    path = run / ".run.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise OrchestrationError(f"another orchestrator owns {path}")
    os.ftruncate(descriptor, 0)
    os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
    os.fsync(descriptor)
    return descriptor


def _acquire_global_lock() -> int:
    descriptor = os.open(GLOBAL_RUN_LOCK, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise OrchestrationError(
            f"another MT-GQA history experiment owns {GLOBAL_RUN_LOCK}")
    os.ftruncate(descriptor, 0)
    os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
    os.fsync(descriptor)
    return descriptor


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", choices=("smoke", "full"), default="full")
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--min-free-after-gib", type=float, default=None)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    return parser.parse_args(argv)


def _main_locked(args: argparse.Namespace) -> int:
    run, results = validate_output_roots(args.run_root, args.results_root)
    contract = read_index_contract(args.index)
    for path in (EVALUATOR, ANALYZER, PROTECTOR):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)

    fresh = not args.resume
    if fresh:
        if os.path.lexists(run) or os.path.lexists(results):
            raise FileExistsError(
                "new experiment roots must be absent; use --resume for an existing run")
        shard_size = DEFAULT_SHARD_SIZE if args.shard_size is None else args.shard_size
        reserve = (MIN_FREE_AFTER_GIB if args.min_free_after_gib is None
                   else args.min_free_after_gib)
        if not MIN_SHARD_SIZE <= shard_size <= MAX_SHARD_SIZE:
            raise ValueError("shard size must be between 40 and 60 images")
        if reserve < MIN_FREE_AFTER_GIB:
            raise ValueError("minimum free-space reserve is 30 GiB")
        gpu = gpu_preflight()
        revision = local_model_revision()
        protection = protection_before(run, results)
        experiment_id = str(uuid.uuid4())
        manifest = make_manifest(
            run=run, results=results, experiment_id=experiment_id,
            contract=contract, model_revision=revision, gpu=gpu,
            shard_size=shard_size, min_free_after_gib=reserve,
            protection=protection)
        atomic_json(run / "manifest.json", manifest, exclusive=True)
        progress: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION, "status": "running",
            "stage": "initializing", "experiment_id": experiment_id,
            "updated_at_unix": time.time()}
        atomic_json(run / "progress.json", progress, exclusive=True)
    else:
        if not run.is_dir() or run.is_symlink():
            raise FileNotFoundError(run)
        manifest = _read_json(run / "manifest.json")
        validate_manifest(manifest, run=run, results=results, contract=contract)
        experiment_id = str(manifest["experiment_id"])
        shard_size = int(manifest["shard_size_images"])
        reserve = float(manifest["min_free_after_gib"])
        if args.shard_size is not None and args.shard_size != shard_size:
            raise ValueError("resume shard size differs from immutable manifest")
        if (args.min_free_after_gib is not None
                and args.min_free_after_gib != reserve):
            raise ValueError("resume free-space reserve differs from immutable manifest")
        progress = _read_json(run / "progress.json")
        if (run / "COMPLETED").is_file():
            validate_analysis(results)
            print(json.dumps({"status": "already_complete", "run_root": str(run),
                              "results_root": str(results)}, indent=2))
            return 0
        work_remains = not (
            stage_markers_complete(
                run=run, stage_name="smoke", dialogue_limit=SMOKE_DIALOGUES,
                experiment_id=experiment_id, dialogues=contract["dialogues"],
                shard_size=shard_size)
            and stage_markers_complete(
                run=run, stage_name="full", dialogue_limit=EXPECTED_DIALOGUES,
                experiment_id=experiment_id, dialogues=contract["dialogues"],
                shard_size=shard_size))
        if work_remains:
            gpu_preflight()

    lock_descriptor = _acquire_run_lock(run)
    try:
        validate_manifest(manifest, run=run, results=results, contract=contract)
        smoke = run_paired_stage(
            stage_name="smoke", dialogue_limit=SMOKE_DIALOGUES, run=run,
            experiment_id=experiment_id, contract=contract,
            shard_size=shard_size, min_free_after_gib=reserve,
            progress=progress)
        atomic_json(run / "smoke_validation.json", smoke)
        if args.stop_after == "smoke":
            progress.update({"status": "partial", "stage": "smoke_complete",
                             "updated_at_unix": time.time()})
            atomic_json(run / "progress.json", progress)
            print(json.dumps({"status": "smoke_complete", "run_root": str(run),
                              "resume_required": True}, indent=2))
            return 0

        full = run_paired_stage(
            stage_name="full", dialogue_limit=EXPECTED_DIALOGUES, run=run,
            experiment_id=experiment_id, contract=contract,
            shard_size=shard_size, min_free_after_gib=reserve,
            progress=progress)
        if (full["protocols"]["gold_history"]["requests"]
                != REQUESTS_PER_PROTOCOL
                or full["protocols"]["generated_history"]["requests"]
                != REQUESTS_PER_PROTOCOL):
            raise OrchestrationError("full logical request count is not exact")
        atomic_json(run / "full_validation.json", full)

        # Verify all pre-existing artifacts before any results directory is
        # published; the analyzer receives this PASS evidence explicitly.
        protection_report = ensure_protection_verified(run, results)
        protection_path = run / "protected_artifacts_validation.json"
        if not analysis_is_complete(results):
            if os.path.lexists(results):
                raise OrchestrationError(
                    "partial/unvalidated results root blocks atomic analyzer publication")
            command = build_analyzer_command(
                gold_run_dir=run / "full" / "gold_history",
                generated_run_dir=run / "full" / "generated_history",
                results_root=results,
                protection_validation=protection_path,
                index=contract["index"])
            run_streaming(command, run / "logs" / "analysis.log")
        analysis = validate_analysis(results)
        completed = {
            "schema_version": SCHEMA_VERSION, "passed": True,
            "experiment_id": experiment_id,
            "gold_history_requests": REQUESTS_PER_PROTOCOL,
            "generated_history_requests": REQUESTS_PER_PROTOCOL,
            "total_requests": REQUESTS_BOTH_PROTOCOLS,
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
                          "requests": REQUESTS_BOTH_PROTOCOLS}, indent=2))
        return 0
    finally:
        os.close(lock_descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    global_descriptor = _acquire_global_lock()
    try:
        return _main_locked(args)
    finally:
        os.close(global_descriptor)


if __name__ == "__main__":
    raise SystemExit(main())

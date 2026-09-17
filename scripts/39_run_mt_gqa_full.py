"""Run the staged MT-GQA-reconstructed experiment without touching old runs.

The driver enforces the requested order:

1. provenance/index construction,
2. a ten-dialogue correctness smoke,
3. a one-hundred-dialogue pilot,
4. a written full-run projection, and
5. all 4,061 three-turn dialogues.

GPU work is restartable at immutable per-image artifacts.  The evaluator owns
and removes only its marked temporary Visual-KV payload.  This file never
deletes an existing result, run, or cache tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = ROOT / "data/mt_gqa"
DEFAULT_RUN_ROOT = ROOT / "runs/mt_gqa_full"
DEFAULT_RESULTS_ROOT = ROOT / "results/mt_gqa_full"
DEFAULT_QUESTIONS = Path(
    "/home/dblab/hj/SparseVLMs/playground/data/eval/gqa/data/"
    "testdev_balanced_questions.json"
)
DEFAULT_IMAGE_DIR = Path(
    "/home/dblab/hj/SparseVLMs/playground/data/eval/gqa/data/images"
)

SCHEMA_VERSION = "mt-gqa-full-orchestrator-v1"
EXPECTED_DIALOGUES = 4_061
EXPECTED_TURNS = EXPECTED_DIALOGUES * 3
EXPECTED_METHODS = 4
EXPECTED_REQUESTS = EXPECTED_TURNS * EXPECTED_METHODS
EXPECTED_SOURCE_SHA256 = (
    "14039069c0b3c797c7aa9bcd5f4c2aa4b5976e02c0b6773e1a584d942a03a318"
)
SEED = 1234
BOOTSTRAP_RESAMPLES = 10_000
MIN_FREE_AFTER_GIB = 30.0
OUTPUT_PREFIX = "mt_gqa_full"
STAGE_NAMES = ("smoke_10", "pilot_100", "full_4061")
METHOD_KEYS = ("recompute", "fullload", "prefix25", "prefix45")
EVALUATOR_OWNER_FILE = ".mt_gqa_temp_store_owner.json"
EVALUATOR_DATASET = "gqa_testdev_balanced_mt3"
ANALYSIS_REQUIRED_OUTPUTS = (
    "dataset_provenance.json", "dialogues.json", "dataset_stats.json",
    "config.json", "raw.jsonl", "per_turn.csv", "per_dialog.csv",
    "quality_by_turn.csv", "latency_by_turn.csv", "io_summary.csv",
    "persistence_overhead.csv", "statistical_analysis.json",
    "validation.json", "README.md",
)


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and os.path.lexists(path):
        raise FileExistsError(f"refusing to replace {path}")
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False,
                      allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            # Hard-link publication is atomic and cannot replace a path that
            # appears after the first existence check.
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


def validate_output_roots(run_root: Path, results_root: Path) -> tuple[Path, Path]:
    for value, label in ((run_root, "run"), (results_root, "result")):
        if value.exists() and value.is_symlink():
            raise ValueError(f"{label} root may not be a symlink: {value}")
    run = run_root.resolve()
    results = results_root.resolve()
    if run.parent != (ROOT / "runs").resolve() or not run.name.startswith(
            OUTPUT_PREFIX):
        raise ValueError(f"invalid dedicated run root: {run}")
    if results.parent != (ROOT / "results").resolve() \
            or not results.name.startswith(OUTPUT_PREFIX):
        raise ValueError(f"invalid dedicated result root: {results}")
    if run == results or run in results.parents or results in run.parents:
        raise ValueError("run and result roots overlap")
    return run, results


def local_model_revision() -> str:
    reference = (
        Path("/home/dblab/.cache/huggingface/hub")
        / "models--llava-hf--llava-v1.6-vicuna-7b-hf/refs/main"
    )
    if not reference.is_file() or reference.is_symlink():
        raise FileNotFoundError(reference)
    revision = reference.read_text(encoding="utf-8").strip()
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError(f"invalid local checkpoint revision: {revision!r}")
    snapshot = reference.parent.parent / "snapshots" / revision
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise FileNotFoundError(snapshot)
    return revision


def _iter_protected_entries(
    run_root: Path, results_root: Path,
) -> Iterable[Path]:
    """Walk prior run/result trees without following directory symlinks."""
    excluded = (run_root.resolve(), results_root.resolve())
    for scope in ((ROOT / "runs").resolve(), (ROOT / "results").resolve()):
        if not scope.is_dir():
            continue
        for directory, dirnames, filenames in os.walk(scope, followlinks=False):
            parent = Path(directory)
            kept_dirs = []
            for name in sorted(dirnames):
                path = parent / name
                absolute = path.absolute()
                if any(absolute == root or root in absolute.parents
                       for root in excluded):
                    continue
                yield path
                if not path.is_symlink():
                    kept_dirs.append(name)
            dirnames[:] = kept_dirs
            for name in sorted(filenames):
                path = parent / name
                absolute = path.absolute()
                if any(absolute == root or root in absolute.parents
                       for root in excluded):
                    continue
                yield path


def protected_snapshot(run_root: Path, results_root: Path) -> dict[str, Any]:
    entries: dict[str, dict[str, Any]] = {}
    for path in _iter_protected_entries(run_root, results_root):
        relative = path.relative_to(ROOT).as_posix()
        if path.is_symlink():
            entries[relative] = {
                "type": "symlink",
                "target": os.readlink(path),
            }
        elif path.is_dir():
            entries[relative] = {"type": "directory"}
        elif path.is_file():
            entries[relative] = {
                "type": "regular_file",
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        else:
            entries[relative] = {
                "type": "other",
                "mode": int(path.lstat().st_mode),
            }
    regular = [row for row in entries.values()
               if row["type"] == "regular_file"]
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": (
            "all pre-existing entries under runs/ and results/, including "
            "regular-file content, empty directories, and symlink targets"
        ),
        "excluded_new_roots": [str(run_root), str(results_root)],
        "entry_count": len(entries),
        "file_count": len(regular),
        "total_bytes": sum(item["size_bytes"] for item in regular),
        "entries": entries,
        "manifest_sha256": canonical_hash(entries),
    }


def verify_protected_snapshot(expected: dict[str, Any], run_root: Path,
                              results_root: Path) -> dict[str, Any]:
    observed = protected_snapshot(run_root, results_root)
    before = expected.get("entries", expected.get("files", {}))
    if observed["manifest_sha256"] != expected.get("manifest_sha256") \
            or observed["entries"] != before:
        after = observed["entries"]
        changed = sorted(
            key for key in set(before) | set(after)
            if before.get(key) != after.get(key)
        )
        raise RuntimeError(
            "protected prior artifacts changed: " + ", ".join(changed[:30]))
    return observed


def run_streaming(command: list[str], log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if (log_path.parent.is_symlink()
            or log_path.parent.resolve() != log_path.parent.absolute()):
        raise ValueError(f"log parent may not traverse symlinks: {log_path.parent}")
    if os.path.lexists(log_path) and (
            log_path.is_symlink() or not log_path.is_file()):
        raise ValueError(f"log path is not a regular file: {log_path}")
    environment = dict(os.environ)
    environment.update(
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    rendered = " ".join(command)
    started = time.perf_counter()
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(log_path, flags, 0o644)
    with os.fdopen(descriptor, "a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n$ {rendered}\n")
        print(f"$ {rendered}", flush=True)
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            print(line, end="", flush=True)
        returncode = process.wait()
        log.flush()
        os.fsync(log.fileno())
    elapsed = time.perf_counter() - started
    if returncode:
        raise RuntimeError(f"stage failed ({returncode}): {rendered}")
    return elapsed


def load_dialogues(index_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    dialogues = payload if isinstance(payload, list) else payload.get("dialogues")
    if not isinstance(dialogues, list):
        raise ValueError("dialogues.json omits the dialogue list")
    return dialogues


def selected_image_count(dialogues: list[dict[str, Any]], limit: int) -> int:
    return len({str(item["image_id"]) for item in dialogues[:limit]})


def automatic_shard_size(free_bytes: int, reserve_gib: float) -> int:
    """Choose 40--60 images; stores are still materialised one at a time."""
    free_gib = free_bytes / float(1024 ** 3)
    usable = free_gib - float(reserve_gib)
    if usable >= 80.0:
        return 60
    if usable >= 50.0:
        return 50
    if usable >= 8.0:
        return 40
    raise RuntimeError(
        f"insufficient storage headroom: free={free_gib:.2f} GiB, "
        f"reserve={reserve_gib:.2f} GiB")


def choose_shard_size(*, requested: int | None, free_bytes: int,
                      reserve_gib: float,
                      frozen_experiment: dict[str, Any] | None) -> int:
    """Choose once for a new run and reuse the frozen value on resume."""
    if frozen_experiment is not None:
        frozen = int(frozen_experiment.get("shard_size", -1))
        if not 40 <= frozen <= 60:
            raise ValueError(f"invalid frozen shard size: {frozen}")
        if requested is not None and int(requested) != frozen:
            raise ValueError(
                f"resume shard_size mismatch: {requested} != {frozen}")
        return frozen
    chosen = (automatic_shard_size(free_bytes, reserve_gib)
              if requested is None else int(requested))
    if not 40 <= chosen <= 60:
        raise ValueError("storage shard size must be in [40, 60]")
    return chosen


def cleanup_stage_temp_root(*, temporary: Path, run_root: Path,
                            stage_name: str, experiment_id: str) -> dict[str, Any]:
    """Remove only an empty, exactly-owned evaluator temporary root.

    Image payload leaves are deleted by evaluator 37 only after their durable
    artifacts exist.  The orchestrator never recursively removes them: it
    merely verifies the payload is empty, then removes the marker and empty
    directories belonging to this exact experiment/stage.
    """
    if stage_name not in STAGE_NAMES:
        raise ValueError(f"unknown stage name: {stage_name}")
    run = run_root.resolve()
    wanted = run / "_temporary_visual_kv" / stage_name
    if temporary.absolute() != wanted:
        raise ValueError(f"temporary root is not the exact stage path: {temporary}")
    if not os.path.lexists(temporary):
        return {"passed": True, "removed": False, "payload_leaves": 0}
    if temporary.is_symlink() or not temporary.is_dir():
        raise ValueError(f"temporary stage root is not a real directory: {temporary}")
    owner_path = temporary / EVALUATOR_OWNER_FILE
    payload = temporary / "payload"
    if (not owner_path.is_file() or owner_path.is_symlink()
            or not payload.is_dir() or payload.is_symlink()):
        raise ValueError("temporary stage ownership structure is invalid")
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    expected_owner = {
        "experiment_id": str(experiment_id),
        "dataset": EVALUATOR_DATASET,
        "purpose": "temporary_visual_kv_only",
    }
    mismatch = {key: (owner.get(key), value)
                for key, value in expected_owner.items()
                if owner.get(key) != value}
    if mismatch:
        raise ValueError(f"temporary stage ownership mismatch: {mismatch}")
    payload_entries = list(payload.iterdir())
    if payload_entries:
        raise RuntimeError(
            "temporary Visual-KV payload leak after completed stage: "
            + ", ".join(path.name for path in payload_entries[:20]))
    allowed = {EVALUATOR_OWNER_FILE, "payload"}
    unexpected = sorted(path.name for path in temporary.iterdir()
                        if path.name not in allowed)
    if unexpected:
        raise RuntimeError(
            "unexpected files in owned temporary stage root: "
            + ", ".join(unexpected))
    payload.rmdir()
    owner_path.unlink()
    temporary.rmdir()
    parent = run / "_temporary_visual_kv"
    if parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()
    return {"passed": True, "removed": True, "payload_leaves": 0}


def cleanup_all_stage_temp_roots(run_root: Path, experiment_id: str) -> dict:
    reports = {}
    for stage_name in STAGE_NAMES:
        temporary = run_root / "_temporary_visual_kv" / stage_name
        reports[stage_name] = cleanup_stage_temp_root(
            temporary=temporary, run_root=run_root, stage_name=stage_name,
            experiment_id=experiment_id)
    parent = run_root / "_temporary_visual_kv"
    if os.path.lexists(parent):
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError("temporary Visual-KV parent is not a real directory")
        leftovers = sorted(path.name for path in parent.iterdir())
        if leftovers:
            raise RuntimeError(
                "unknown temporary Visual-KV stage roots remain: "
                + ", ".join(leftovers))
        parent.rmdir()
    return {"passed": True, "temporary_kv_leak_count": 0,
            "stages": reports}


def build_evaluator_command(*, evaluator: Path, stage: Path, temporary: Path,
                            experiment_id: str, contract: dict[str, Any],
                            dialogue_limit: int, shard_index: int,
                            shard_size: int,
                            min_free_after_gib: float) -> list[str]:
    if dialogue_limit not in (10, 100, EXPECTED_DIALOGUES):
        raise ValueError("stage dialogue limit must be 10, 100, or 4,061")
    command = [
        sys.executable, str(evaluator),
        "--index", str(contract["index_path"]),
        "--run-dir", str(stage),
        "--temp-root", str(temporary),
        "--experiment-id", str(experiment_id),
        "--shard-index", str(int(shard_index)),
        "--shard-size", str(int(shard_size)),
        "--expected-index-sha256", str(contract["index_sha256"]),
        "--expected-workload-sha256", str(contract["workload_sha256"]),
        "--expected-dialogs", str(int(dialogue_limit)),
        "--seed", str(SEED),
        "--max-new-tokens", "16",
        "--min-free-after-gib", str(float(min_free_after_gib)),
    ]
    if dialogue_limit != EXPECTED_DIALOGUES:
        command.extend([
            "--max-dialogs", str(int(dialogue_limit)),
            "--allow-partial-workload",
        ])
    return command


def validate_stage_completion(*, stage: Path, experiment_id: str,
                              contract: dict[str, Any],
                              expected_dialogues: int,
                              expected_images: int, shard_size: int,
                              expected_shards: int) -> dict[str, Any]:
    """Independently prove exact dialogue/turn/method request coverage."""
    config_path = stage / "config.json"
    if not config_path.is_file() or config_path.is_symlink():
        raise ValueError(f"stage config is not a regular file: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_requests = int(expected_dialogues) * 3 * EXPECTED_METHODS
    invariants = {
        "experiment_id": str(experiment_id),
        "dialogues_file_sha256": contract["index_sha256"],
        "source_full_workload_sha256": contract["workload_sha256"],
        "n_dialogs": int(expected_dialogues),
        "n_turns": int(expected_dialogues) * 3,
        "n_images": int(expected_images),
        "n_requests": expected_requests,
        "shard_size": int(shard_size),
        "n_shards": int(expected_shards),
        "seed": SEED,
        "method_keys": list(METHOD_KEYS),
    }
    mismatch = {key: (config.get(key), value)
                for key, value in invariants.items()
                if config.get(key) != value}
    if mismatch:
        raise ValueError(f"completed stage config mismatch: {mismatch}")

    image_paths = sorted((stage / "images").glob("*.json"))
    if len(image_paths) != expected_images:
        raise ValueError(
            f"stage image artifact count {len(image_paths)} != {expected_images}")
    dialog_ids: set[str] = set()
    image_ids: set[str] = set()
    coverage: dict[tuple[str, int], list[str]] = {}
    observed_rows = 0
    for path in image_paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"image artifact is not regular: {path}")
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if artifact.get("experiment_id") != str(experiment_id):
            raise ValueError(f"image artifact experiment mismatch: {path}")
        if artifact.get("validation", {}).get("passed") is not True:
            raise ValueError(f"image artifact validation failed: {path}")
        image_id = str(artifact.get("image_id", ""))
        if not image_id or image_id in image_ids or path.stem != image_id:
            raise ValueError(f"duplicate/mismatched image artifact: {path}")
        image_ids.add(image_id)
        rows = artifact.get("rows")
        ids = [str(value) for value in artifact.get("dialog_ids", [])]
        if (not isinstance(rows, list)
                or artifact.get("n_dialogs") != len(ids)
                or artifact.get("n_turns") != len(ids) * 3
                or artifact.get("n_rows") != len(ids) * 12
                or len(rows) != len(ids) * 12):
            raise ValueError(f"image artifact coverage mismatch: {path}")
        if len(set(ids)) != len(ids) or dialog_ids.intersection(ids):
            raise ValueError(f"dialogue reused across image artifacts: {path}")
        dialog_ids.update(ids)
        observed_rows += len(rows)
        for row in rows:
            did = str(row.get("dialog_id", ""))
            tid = int(row.get("turn_id", 0))
            method = str(row.get("method_key", ""))
            if did not in ids or tid not in (1, 2, 3) or method not in METHOD_KEYS:
                raise ValueError(f"invalid request identity in {path}")
            coverage.setdefault((did, tid), []).append(method)
    if len(dialog_ids) != expected_dialogues:
        raise ValueError(
            f"observed dialogues {len(dialog_ids)} != {expected_dialogues}")
    if observed_rows != expected_requests:
        raise ValueError(
            f"observed requests {observed_rows} != {expected_requests}")
    if len(coverage) != expected_dialogues * 3:
        raise ValueError("dialogue-turn coverage is incomplete")
    wanted_methods = sorted(METHOD_KEYS)
    if any(sorted(methods) != wanted_methods for methods in coverage.values()):
        raise ValueError("dialogue-turn method coverage is not exactly four arms")

    marker_paths = sorted((stage / "shards").glob("shard_*.json"))
    if len(marker_paths) != expected_shards:
        raise ValueError(
            f"stage shard marker count {len(marker_paths)} != {expected_shards}")
    marked_images: list[str] = []
    for index, path in enumerate(marker_paths):
        if path.name != f"shard_{index:03d}.json" \
                or not path.is_file() or path.is_symlink():
            raise ValueError(f"noncanonical shard marker: {path}")
        marker = json.loads(path.read_text(encoding="utf-8"))
        if (marker.get("complete") is not True
                or marker.get("experiment_id") != str(experiment_id)
                or marker.get("shard_index") != index
                or marker.get("shard_size") != int(shard_size)):
            raise ValueError(f"invalid shard marker: {path}")
        marked_images.extend(str(value) for value in marker.get("image_ids", []))
    if len(marked_images) != len(set(marked_images)) \
            or set(marked_images) != image_ids:
        raise ValueError("shard markers do not exactly partition image artifacts")
    return {
        "passed": True,
        "dialogues": len(dialog_ids),
        "turns": len(coverage),
        "images": len(image_ids),
        "observed_method_turn_requests": observed_rows,
        "expected_method_turn_requests": expected_requests,
        "exact_request_coverage": True,
    }


def read_index_contract(data_root: Path) -> dict[str, Any]:
    config_path = data_root / "config.json"
    stats_path = data_root / "dataset_stats.json"
    index_path = data_root / "dialogues.json"
    provenance_path = data_root / "dataset_provenance.json"
    for path in (config_path, stats_path, index_path, provenance_path):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    dialogues = load_dialogues(index_path)
    if len(dialogues) != EXPECTED_DIALOGUES:
        raise ValueError(f"dialogue count {len(dialogues)} != {EXPECTED_DIALOGUES}")
    if any(len(item.get("turns", [])) != 3 for item in dialogues):
        raise ValueError("canonical index contains a non-three-turn dialogue")
    if config.get("seed") != SEED:
        raise ValueError("canonical index seed mismatch")
    if config.get("benchmark_type") != "MT-GQA-reconstructed":
        raise ValueError("canonical index is not explicitly reconstructed")
    if config.get("official_benchmark_identity_claimed") is not False:
        raise ValueError("canonical index improperly claims official identity")
    if (config.get("strict_canonical") is not True
            or config.get("source_questions_sha256") != EXPECTED_SOURCE_SHA256
            or config.get("n_dialogues") != EXPECTED_DIALOGUES
            or config.get("n_turns") != EXPECTED_TURNS):
        raise ValueError("canonical index config contract mismatch")
    index_sha = sha256_file(index_path)
    if (config.get("dialogues_sha256") != index_sha
            or config.get("artifact_sha256", {}).get("dialogues.json")
            != index_sha):
        raise ValueError("canonical dialogues hash mismatch")
    workload_hash = config.get("workload_sha256")
    if not isinstance(workload_hash, str) or len(workload_hash) != 64:
        raise ValueError("canonical config omits workload_sha256")
    workload_bytes = "".join(
        f"{dialogue['dialog_id']}\t{int(turn['turn_id'])}\t"
        f"{turn['question_id']}\n"
        for dialogue in dialogues for turn in dialogue["turns"]
    ).encode("utf-8")
    if hashlib.sha256(workload_bytes).hexdigest() != workload_hash:
        raise ValueError("canonical workload SHA256 mismatch")
    if (stats.get("dialogues") != EXPECTED_DIALOGUES
            or stats.get("selected_questions") != EXPECTED_TURNS
            or stats.get("workload_sha256") != workload_hash
            or stats.get("dialogues_sha256") != index_sha):
        raise ValueError("canonical dataset statistics mismatch")
    if (provenance.get("benchmark_type") != "MT-GQA-reconstructed"
            or provenance.get("official_benchmark_identity_claimed") is not False):
        raise ValueError("canonical dataset provenance mismatch")
    return {
        "index_path": index_path,
        "index_sha256": index_sha,
        "workload_sha256": workload_hash,
        "dialogues": dialogues,
        "config": config,
        "stats": stats,
        "provenance_path": provenance_path,
        "provenance_sha256": sha256_file(provenance_path),
    }


def run_inference_stage(*, name: str, dialogue_limit: int,
                        run_root: Path, experiment_id: str,
                        contract: dict[str, Any], shard_size: int,
                        min_free_after_gib: float, resume: bool) -> dict[str, Any]:
    stage = run_root / name
    temporary = run_root / "_temporary_visual_kv" / name
    unique_images = selected_image_count(contract["dialogues"], dialogue_limit)
    n_shards = math.ceil(unique_images / shard_size)
    evaluator = ROOT / "scripts/37_eval_mt_gqa_full_shard.py"
    existing_markers = sorted((stage / "shards").glob("shard_*.json"))
    if len(existing_markers) > n_shards:
        raise ValueError(
            f"stage has {len(existing_markers)} markers but expects {n_shards}")
    if len(existing_markers) == n_shards and (stage / "config.json").is_file():
        completion = validate_stage_completion(
            stage=stage, experiment_id=experiment_id, contract=contract,
            expected_dialogues=dialogue_limit, expected_images=unique_images,
            shard_size=shard_size, expected_shards=n_shards)
        temp_cleanup = cleanup_stage_temp_root(
            temporary=temporary, run_root=run_root, stage_name=name,
            experiment_id=experiment_id)
        return {
            "name": name,
            "dialogues": dialogue_limit,
            "turns": dialogue_limit * 3,
            "planned_method_turn_requests": (
                dialogue_limit * 3 * EXPECTED_METHODS),
            "unique_images": unique_images,
            "shard_size": shard_size,
            "n_shards": n_shards,
            "wall_seconds_this_invocation": 0.0,
            "resume_enabled": bool(resume),
            "completed_stage_reused_without_subprocess": True,
            "run_dir": str(stage),
            "completion_validation": completion,
            "temporary_cleanup": temp_cleanup,
        }
    wall_seconds = 0.0
    for shard_index in range(n_shards):
        command = build_evaluator_command(
            evaluator=evaluator, stage=stage, temporary=temporary,
            experiment_id=experiment_id, contract=contract,
            dialogue_limit=dialogue_limit, shard_index=shard_index,
            shard_size=shard_size,
            min_free_after_gib=min_free_after_gib)
        # Keep the launcher transcript outside the evaluator-owned stage.
        # On a fresh stage, creating ``stage/run.log`` first would make the
        # directory nonempty before evaluator 37 can atomically publish its
        # config, correctly triggering that evaluator's anti-mixing guard.
        wall_seconds += run_streaming(command, run_root / f"{name}.log")
    completion = validate_stage_completion(
        stage=stage, experiment_id=experiment_id, contract=contract,
        expected_dialogues=dialogue_limit, expected_images=unique_images,
        shard_size=shard_size, expected_shards=n_shards)
    temp_cleanup = cleanup_stage_temp_root(
        temporary=temporary, run_root=run_root, stage_name=name,
        experiment_id=experiment_id)
    return {
        "name": name,
        "dialogues": dialogue_limit,
        "turns": dialogue_limit * 3,
        "planned_method_turn_requests": dialogue_limit * 3 * EXPECTED_METHODS,
        "unique_images": unique_images,
        "shard_size": shard_size,
        "n_shards": n_shards,
        "wall_seconds_this_invocation": wall_seconds,
        "resume_enabled": bool(resume),
        "completed_stage_reused_without_subprocess": False,
        "run_dir": str(stage),
        "completion_validation": completion,
        "temporary_cleanup": temp_cleanup,
    }


def run_analysis(stage: Path, destination: Path, expected_dialogues: int,
                 provenance_path: Path, log_path: Path) -> float:
    analyzer = ROOT / "scripts/38_analyze_mt_gqa_full.py"
    command = [
        sys.executable, str(analyzer),
        "--run-root", str(stage),
        "--results-root", str(destination),
        "--expected-dialogs", str(expected_dialogues),
        "--bootstrap-resamples", str(BOOTSTRAP_RESAMPLES),
        "--bootstrap-seed", str(SEED),
        "--dataset-provenance", str(provenance_path),
    ]
    return run_streaming(command, log_path)


def regular_tree_manifest(root: Path) -> dict[str, Any]:
    """Match analyzer 38's immutable source-tree manifest exactly."""
    root = root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"run stage is not a regular directory: {root}")
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"run stage contains a symlink: {path}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "file_count": len(files),
        "total_bytes": sum(row["size_bytes"] for row in files.values()),
        "files": files,
        "manifest_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def validate_analysis_result(destination: Path, *, stage: Path,
                             expected_dialogues: int,
                             experiment_id: str,
                             contract: dict[str, Any]) -> dict[str, Any]:
    """Validate an analyzer publication before skipping it on resume."""
    if not destination.is_dir() or destination.is_symlink():
        raise ValueError(f"analysis destination is not a real directory: {destination}")
    missing = [name for name in ANALYSIS_REQUIRED_OUTPUTS
               if not (destination / name).is_file()
               or (destination / name).is_symlink()]
    if missing:
        raise ValueError(f"analysis result is incomplete: {missing}")
    validation = json.loads(
        (destination / "validation.json").read_text(encoding="utf-8"))
    config = json.loads(
        (destination / "config.json").read_text(encoding="utf-8"))
    expected_rows = int(expected_dialogues) * 3 * EXPECTED_METHODS
    checks = {
        "validation passed": (validation.get("passed"), True),
        "validation n_dialogs": (
            validation.get("n_dialogs"), int(expected_dialogues)),
        "validation expected_dialogs": (
            validation.get("expected_dialogs"), int(expected_dialogues)),
        "validation n_rows": (validation.get("n_rows"), expected_rows),
        "validation expected_rows": (
            validation.get("expected_rows"), expected_rows),
        "config experiment_id": (
            config.get("experiment_id"), str(experiment_id)),
        "config index hash": (
            config.get("dialogues_file_sha256"), contract["index_sha256"]),
        "config workload hash": (
            config.get("source_full_workload_sha256"),
            contract["workload_sha256"]),
        "config n_requests": (config.get("n_requests"), expected_rows),
    }
    mismatch = {label: values for label, values in checks.items()
                if values[0] != values[1]}
    if mismatch:
        raise ValueError(f"analysis result contract mismatch: {mismatch}")
    with (destination / "raw.jsonl").open(encoding="utf-8") as handle:
        raw_rows = sum(1 for line in handle if line.strip())
    if raw_rows != expected_rows:
        raise ValueError(
            f"analysis raw row count {raw_rows} != {expected_rows}")
    source_manifest = regular_tree_manifest(stage)
    if (validation.get("source_manifest_before") != source_manifest
            or validation.get("source_manifest_after_analysis")
            != source_manifest):
        raise ValueError(
            "analysis result was built from a different source run tree")
    return {
        "passed": True,
        "n_dialogues": int(expected_dialogues),
        "n_rows": expected_rows,
        "exact_request_coverage": True,
    }


def ensure_analysis_result(*, stage: Path, destination: Path,
                           expected_dialogues: int, experiment_id: str,
                           contract: dict[str, Any], provenance_path: Path,
                           log_path: Path) -> dict[str, Any]:
    if not destination.exists():
        run_analysis(stage, destination, expected_dialogues,
                     provenance_path, log_path)
    return validate_analysis_result(
        destination, stage=stage, expected_dialogues=expected_dialogues,
        experiment_id=experiment_id, contract=contract)


def write_identical_or_exclusive_json(path: Path, value: Any) -> None:
    """Resume-safe publication for an orchestrator-owned immutable artifact."""
    if os.path.lexists(path):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"existing artifact is not a regular file: {path}")
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != value:
            raise ValueError(f"existing artifact differs: {path}")
        return
    atomic_json(path, value, exclusive=True)


def validate_protected_result(path: Path, protection: dict[str, Any]) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing preservation validation: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("passed") is not True
            or value.get("before_manifest_sha256")
            != protection.get("manifest_sha256")
            or value.get("after_manifest_sha256")
            != protection.get("manifest_sha256")
            or value.get("temporary_visual_kv_leak_count") != 0):
        raise ValueError("preservation validation contract mismatch")
    return value


def projection_from_pilot(pilot: dict[str, Any], pilot_run: Path,
                          full_unique_images: int) -> dict[str, Any]:
    markers = sorted((pilot_run / "shards").glob("shard_*.json"))
    observed_shard_seconds = 0.0
    for path in markers:
        payload = json.loads(path.read_text(encoding="utf-8"))
        observed_shard_seconds += float(payload.get("elapsed_seconds", 0.0))
    observed = observed_shard_seconds or float(
        pilot.get("wall_seconds_this_invocation", 0.0))
    if observed <= 0:
        raise ValueError("pilot did not expose positive elapsed time")
    seconds_per_dialogue = observed / float(pilot["dialogues"])
    requests_per_second = float(pilot["planned_method_turn_requests"]) / observed
    projected_seconds = seconds_per_dialogue * EXPECTED_DIALOGUES

    persistence_rows = []
    for artifact_path in sorted((pilot_run / "images").glob("*.json")):
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        persistence = payload.get("persistence_overhead")
        if isinstance(persistence, dict):
            persistence_rows.append(persistence)
    persist_ms = [float(row.get("persist_ms", row.get("total_persist_ms", 0.0)))
                  for row in persistence_rows]
    store_bytes = [int(row.get("total_ssd_write_bytes", row.get("bytes_written", 0)))
                   for row in persistence_rows]
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "required_100_dialogue_gpu_pilot",
        "pilot_dialogues": int(pilot["dialogues"]),
        "pilot_unique_images": int(pilot["unique_images"]),
        "pilot_method_turn_requests": int(pilot["planned_method_turn_requests"]),
        "pilot_observed_seconds": observed,
        "pilot_requests_per_second": requests_per_second,
        "pilot_seconds_per_dialogue": seconds_per_dialogue,
        "mean_store_persist_seconds_per_image": (
            sum(persist_ms) / len(persist_ms) / 1e3 if persist_ms else None),
        "peak_temporary_store_bytes": max(store_bytes) if store_bytes else None,
        "full_dialogues": EXPECTED_DIALOGUES,
        "full_unique_images": int(full_unique_images),
        "full_planned_method_turn_requests": EXPECTED_REQUESTS,
        "projected_full_seconds": projected_seconds,
        "projected_full_hours": projected_seconds / 3600.0,
        "projection_does_not_cancel_full": True,
        "full_continues_regardless_of_estimate": True,
        "note": (
            "Linear dialogue projection from the mandatory pilot; the full "
            "stage is launched unconditionally after this artifact is synced."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--results-root", type=Path,
                        default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap-resamples", type=int,
                        default=BOOTSTRAP_RESAMPLES)
    parser.add_argument(
        "--min-free-after-gib", type=float, default=None,
        help=("at least 30 GiB; omitted uses 30 for a new run and the frozen "
              "value when resuming"))
    parser.add_argument("--shard-size", type=int, default=None,
                        help="40--60; omitted chooses from current free space")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--stop-after", choices=(
        "provenance", "smoke", "pilot", "projection", "full"),
        default="full",
        help="debug/operator control; the canonical invocation uses full")
    args = parser.parse_args()

    if args.seed != SEED:
        raise ValueError("the final MT-GQA experiment fixes seed=1234")
    if args.bootstrap_resamples != BOOTSTRAP_RESAMPLES:
        raise ValueError("the final analysis requires exactly 10,000 resamples")
    if (args.min_free_after_gib is not None
            and args.min_free_after_gib < MIN_FREE_AFTER_GIB):
        raise ValueError("the final run requires at least a 30 GiB reserve")
    if args.analysis_only and args.stop_after != "full":
        raise ValueError("--analysis-only cannot be combined with --stop-after")
    run_root, results_root = validate_output_roots(
        args.run_root, args.results_root)
    data_root = args.data_root.resolve()
    if data_root != DEFAULT_DATA_ROOT.resolve():
        raise ValueError(
            f"the canonical experiment fixes --data-root={DEFAULT_DATA_ROOT}")
    questions_path = args.questions.resolve()
    image_dir = args.image_dir.resolve()
    if questions_path != DEFAULT_QUESTIONS.resolve():
        raise ValueError(
            f"the canonical experiment fixes --questions={DEFAULT_QUESTIONS}")
    if image_dir != DEFAULT_IMAGE_DIR.resolve():
        raise ValueError(
            f"the canonical experiment fixes --image-dir={DEFAULT_IMAGE_DIR}")

    # STEP 1: provenance.  The builder validates identical existing output and
    # refuses to replace mismatches, so resume never mutates a frozen index.
    builder = ROOT / "scripts/36_build_mt_gqa_index.py"
    builder_command = [
        sys.executable, str(builder),
        "--questions", str(questions_path),
        "--image-dir", str(image_dir),
        "--out-dir", str(data_root),
        "--seed", str(SEED),
        "--expected-dialogues", str(EXPECTED_DIALOGUES),
        "--expected-source-sha256", EXPECTED_SOURCE_SHA256,
    ]
    # Do not create the run root before its protection manifest is captured.
    # The canonical dataset artifacts themselves retain full provenance; this
    # short command transcript is therefore kept outside runs/results.
    provenance_fd, provenance_name = tempfile.mkstemp(
        prefix="mt_gqa_full_provenance-", suffix=".log")
    os.close(provenance_fd)
    provenance_log = Path(provenance_name)
    run_streaming(builder_command, provenance_log)
    provenance_log.unlink()
    contract = read_index_contract(data_root)
    if args.stop_after == "provenance":
        print(json.dumps({
            "status": "provenance_complete",
            "benchmark_type": "MT-GQA-reconstructed",
            "dialogues": EXPECTED_DIALOGUES,
            "index_sha256": contract["index_sha256"],
        }, indent=2), flush=True)
        return

    experiment_path = run_root / "experiment.json"
    protection_path = run_root / "protected_artifacts_before.json"
    existing_experiment: dict[str, Any] | None = None
    if run_root.exists():
        if not (args.resume or args.analysis_only):
            raise FileExistsError(
                f"run root exists; use --resume explicitly: {run_root}")
        for path in (experiment_path, protection_path):
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"resume metadata is not a regular file: {path}")
        existing_experiment = json.loads(
            experiment_path.read_text(encoding="utf-8"))
    elif args.resume or args.analysis_only:
        raise FileNotFoundError(run_root)

    model_revision = local_model_revision()
    disk_free = shutil.disk_usage(ROOT).free
    min_free_after_gib = (
        float(existing_experiment["min_free_after_gib"])
        if existing_experiment is not None and args.min_free_after_gib is None
        else float(MIN_FREE_AFTER_GIB if args.min_free_after_gib is None
                   else args.min_free_after_gib)
    )
    if min_free_after_gib < MIN_FREE_AFTER_GIB:
        raise ValueError("frozen storage reserve is below 30 GiB")
    shard_size = choose_shard_size(
        requested=args.shard_size, free_bytes=disk_free,
        reserve_gib=min_free_after_gib,
        frozen_experiment=existing_experiment)

    if not run_root.exists():
        if results_root.exists():
            raise FileExistsError(results_root)
        protection = protected_snapshot(run_root, results_root)
        run_root.mkdir(parents=True, exist_ok=False)
        experiment_id = str(uuid.uuid4())
        experiment = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "status": "running",
            "benchmark_type": "MT-GQA-reconstructed",
            "exact_official_identity_claimed": False,
            "seed": SEED,
            "model_revision": model_revision,
            "dialogues": EXPECTED_DIALOGUES,
            "turns": EXPECTED_TURNS,
            "methods": EXPECTED_METHODS,
            "planned_method_turn_requests": EXPECTED_REQUESTS,
            "full_planned_method_turn_requests": EXPECTED_REQUESTS,
            "smoke_planned_method_turn_requests": 10 * 3 * EXPECTED_METHODS,
            "pilot_planned_method_turn_requests": 100 * 3 * EXPECTED_METHODS,
            "staged_validation_requests_outside_full": 110 * 3 * EXPECTED_METHODS,
            "shard_size": shard_size,
            "shard_size_policy": (
                "automatic_40_to_60_from_free_space"
                if args.shard_size is None else "explicit_cli_40_to_60"),
            "disk_free_bytes_at_start": int(disk_free),
            "min_free_after_gib": min_free_after_gib,
            "index_sha256": contract["index_sha256"],
            "workload_sha256": contract["workload_sha256"],
            "dataset_provenance_sha256": contract["provenance_sha256"],
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "execution_order": [
                "provenance", "smoke_10", "pilot_100", "projection",
                "full_4061", "analysis"],
            "started_at_unix": time.time(),
        }
        atomic_json(protection_path, protection, exclusive=True)
        atomic_json(experiment_path, experiment, exclusive=True)
    else:
        assert existing_experiment is not None
        experiment = existing_experiment
        protection = json.loads(protection_path.read_text(encoding="utf-8"))
        experiment_id = str(experiment["experiment_id"])
        frozen = {
            "schema_version": SCHEMA_VERSION,
            "benchmark_type": "MT-GQA-reconstructed",
            "exact_official_identity_claimed": False,
            "seed": SEED,
            "model_revision": model_revision,
            "dialogues": EXPECTED_DIALOGUES,
            "turns": EXPECTED_TURNS,
            "methods": EXPECTED_METHODS,
            "planned_method_turn_requests": EXPECTED_REQUESTS,
            "full_planned_method_turn_requests": EXPECTED_REQUESTS,
            "shard_size": shard_size,
            "min_free_after_gib": min_free_after_gib,
            "index_sha256": contract["index_sha256"],
            "workload_sha256": contract["workload_sha256"],
            "dataset_provenance_sha256": contract["provenance_sha256"],
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        }
        for key, expected in frozen.items():
            if experiment.get(key) != expected:
                raise ValueError(
                    f"resume {key} mismatch: {experiment.get(key)!r} != "
                    f"{expected!r}")
        verify_protected_snapshot(protection, run_root, results_root)
        if experiment.get("status") not in {"running", "complete"}:
            raise ValueError(f"invalid experiment status: {experiment.get('status')}")

    print(json.dumps({
        "status": "preflight_complete",
        "experiment_id": experiment_id,
        "full_dialogues": EXPECTED_DIALOGUES,
        "turns": EXPECTED_TURNS,
        "methods": EXPECTED_METHODS,
        "full_planned_method_turn_requests": EXPECTED_REQUESTS,
        "shard_size": shard_size,
        "min_free_after_gib": min_free_after_gib,
        "resume": existing_experiment is not None,
    }, indent=2), flush=True)

    if experiment.get("status") == "complete":
        cleanup_all_stage_temp_roots(run_root, experiment_id)
        verify_protected_snapshot(protection, run_root, results_root)
        result_validation = validate_analysis_result(
            results_root, stage=run_root / "full_4061",
            expected_dialogues=EXPECTED_DIALOGUES,
            experiment_id=experiment_id, contract=contract)
        if result_validation["n_rows"] != EXPECTED_REQUESTS:
            raise ValueError("completed experiment does not contain 48,732 rows")
        validate_protected_result(
            results_root / "protected_artifacts_validation.json", protection)
        print(json.dumps({
            "status": "already_complete_validated",
            "experiment_id": experiment_id,
            "requests": result_validation["n_rows"],
            "results_root": str(results_root),
        }, indent=2), flush=True)
        return

    if args.analysis_only:
        full_unique_images = selected_image_count(
            contract["dialogues"], EXPECTED_DIALOGUES)
        full_n_shards = math.ceil(full_unique_images / shard_size)
        full_completion = validate_stage_completion(
            stage=run_root / "full_4061", experiment_id=experiment_id,
            contract=contract, expected_dialogues=EXPECTED_DIALOGUES,
            expected_images=full_unique_images, shard_size=shard_size,
            expected_shards=full_n_shards)
        experiment["full"] = {
            **experiment.get("full", {}),
            "completion_validation": full_completion,
        }
        experiment["temporary_cleanup"] = cleanup_all_stage_temp_roots(
            run_root, experiment_id)
        atomic_json(experiment_path, experiment)
        ensure_analysis_result(
            stage=run_root / "full_4061", destination=results_root,
            expected_dialogues=EXPECTED_DIALOGUES,
            experiment_id=experiment_id, contract=contract,
            provenance_path=contract["provenance_path"],
            log_path=run_root / "analysis.log")
    else:
        # STEP 2: ten-dialogue correctness smoke.
        smoke = run_inference_stage(
            name="smoke_10", dialogue_limit=10, run_root=run_root,
            experiment_id=experiment_id, contract=contract,
            shard_size=shard_size,
            min_free_after_gib=min_free_after_gib, resume=args.resume)
        smoke_analysis = run_root / "smoke_10_analysis"
        ensure_analysis_result(
            stage=run_root / "smoke_10", destination=smoke_analysis,
            expected_dialogues=10, experiment_id=experiment_id,
            contract=contract, provenance_path=contract["provenance_path"],
            log_path=run_root / "smoke_analysis.log")
        experiment["smoke"] = smoke
        atomic_json(experiment_path, experiment)
        verify_protected_snapshot(protection, run_root, results_root)
        if args.stop_after == "smoke":
            return

        # STEP 3: required 100-dialogue throughput pilot.
        pilot = run_inference_stage(
            name="pilot_100", dialogue_limit=100, run_root=run_root,
            experiment_id=experiment_id, contract=contract,
            shard_size=shard_size,
            min_free_after_gib=min_free_after_gib, resume=args.resume)
        pilot_analysis = run_root / "pilot_100_analysis"
        ensure_analysis_result(
            stage=run_root / "pilot_100", destination=pilot_analysis,
            expected_dialogues=100, experiment_id=experiment_id,
            contract=contract, provenance_path=contract["provenance_path"],
            log_path=run_root / "pilot_analysis.log")
        experiment["pilot"] = pilot
        atomic_json(experiment_path, experiment)
        verify_protected_snapshot(protection, run_root, results_root)
        if args.stop_after == "pilot":
            return

        # STEP 4: project, sync, and continue unconditionally.
        full_unique_images = selected_image_count(
            contract["dialogues"], EXPECTED_DIALOGUES)
        projection = projection_from_pilot(
            pilot, run_root / "pilot_100", full_unique_images)
        projection_path = run_root / "projection.json"
        if projection_path.exists():
            if json.loads(projection_path.read_text(encoding="utf-8")) != projection:
                raise ValueError("resume projection differs from frozen pilot")
        else:
            atomic_json(projection_path, projection, exclusive=True)
        experiment["projection"] = projection
        atomic_json(experiment_path, experiment)
        if args.stop_after == "projection":
            return

        # STEP 5: full 4,061-dialogue run.  No estimate threshold exists here.
        full = run_inference_stage(
            name="full_4061", dialogue_limit=EXPECTED_DIALOGUES,
            run_root=run_root, experiment_id=experiment_id,
            contract=contract, shard_size=shard_size,
            min_free_after_gib=min_free_after_gib, resume=args.resume)
        experiment["full"] = full
        atomic_json(experiment_path, experiment)
        verify_protected_snapshot(protection, run_root, results_root)

        ensure_analysis_result(
            stage=run_root / "full_4061", destination=results_root,
            expected_dialogues=EXPECTED_DIALOGUES,
            experiment_id=experiment_id, contract=contract,
            provenance_path=contract["provenance_path"],
            log_path=run_root / "analysis.log")

    temporary_cleanup = cleanup_all_stage_temp_roots(run_root, experiment_id)
    result_validation = validate_analysis_result(
        results_root, stage=run_root / "full_4061",
        expected_dialogues=EXPECTED_DIALOGUES,
        experiment_id=experiment_id, contract=contract)
    if result_validation["n_rows"] != EXPECTED_REQUESTS:
        raise ValueError(
            f"full request count is not exact: {result_validation['n_rows']} "
            f"!= {EXPECTED_REQUESTS}")
    after = verify_protected_snapshot(protection, run_root, results_root)
    protected_validation = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "before_manifest_sha256": protection["manifest_sha256"],
        "after_manifest_sha256": after["manifest_sha256"],
        "entry_count": after["entry_count"],
        "file_count": after["file_count"],
        "total_bytes": after["total_bytes"],
        "temporary_visual_kv_leak_count": 0,
    }
    write_identical_or_exclusive_json(
        results_root / "protected_artifacts_validation.json",
        protected_validation)
    experiment.update({
        "status": "complete",
        "completed_at_unix": time.time(),
        "protected_prior_artifacts_unchanged": True,
        "actual_full_method_turn_requests": result_validation["n_rows"],
        "exact_48732_request_coverage": True,
        "temporary_cleanup": temporary_cleanup,
        "disk_free_bytes_at_end": int(shutil.disk_usage(ROOT).free),
    })
    atomic_json(experiment_path, experiment)
    print(json.dumps({
        "status": "complete",
        "experiment_id": experiment_id,
        "requests": EXPECTED_REQUESTS,
        "results_root": str(results_root),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the frozen five-arm MT-GQA Generated-History experiment.

This orchestrator owns the immutable run contract, source/artifact protection,
shard scheduling, exact coverage audit, raw publication, and final reporting.
The model and method implementations remain in script 73 and existing modules.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "mt-gqa-5arm-generated-orchestrator-v1"
PROTOCOL = "generated_history"
INDEX = ROOT / "data/mt_gqa/dialogues.json"
INDEX_SHA256 = "2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224"
WORKLOAD_SHA256 = "0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62"
MODEL_REVISION = "c916e6cdcd760b4cecd1dd4907f84ac649f93b23"
MODEL_REF = Path("/home/dblab/.cache/huggingface/hub/models--llava-hf--llava-v1.6-vicuna-7b-hf/refs/main")
EVALUATOR = ROOT / "scripts/73_eval_mt_gqa_5arm_generated_shard.py"
PROTECTOR = ROOT / "scripts/74_protect_mt_gqa_5arm.py"
REPORTER = ROOT / "scripts/75_report_mt_gqa_5arm.py"
RUNTIME_PYTHON = Path("/home/dblab/anaconda3/envs/mllm_ft/bin/python")
METHODS = ("recompute", "fullload", "mpic32", "rekv_chunk25", "ours25")
IMAGE_COUNT = 398
DIALOGUE_COUNT = 4_061
TURNS = (1, 2, 3)
REQUESTS_PER_METHOD = DIALOGUE_COUNT * len(TURNS)
TOTAL_REQUESTS = REQUESTS_PER_METHOD * len(METHODS)
HITS_PER_METHOD = DIALOGUE_COUNT * 2
SEED = 1234
MAX_NEW_TOKENS = 16
DEFAULT_SHARD_SIZE = 50
MIN_SHARD_SIZE = 40
MAX_SHARD_SIZE = 60
MIN_FREE_AFTER_GIB = 30.0
SMOKE_DIALOGUES = 10
LOCK = Path("/tmp/mllm_v2_mt_gqa_5arm_generated.lock")
SHORT_ANSWER_INSTRUCTION = "Answer the current question with a single word or short phrase."


class OrchestrationError(RuntimeError):
    pass


def _load_module(name: str, path: Path):
    if path.is_symlink() or not path.is_file():
        raise OrchestrationError(f"runtime source is missing/unsafe: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise OrchestrationError(f"cannot load runtime source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _protector():
    return _load_module("_mt5_protection", PROTECTOR)


def canonical_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise OrchestrationError(f"required JSON is missing/unsafe: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OrchestrationError(f"JSON root is not an object: {path}")
    return value


def frozen_workload() -> dict[str, Any]:
    if INDEX.is_symlink() or not INDEX.is_file():
        raise OrchestrationError(f"frozen index is missing/unsafe: {INDEX}")
    digest = file_hash(INDEX)
    if digest != INDEX_SHA256:
        raise OrchestrationError(f"frozen index SHA256 mismatch: {digest}")
    value = read_json(INDEX)
    dialogues = value.get("dialogues")
    if not isinstance(dialogues, list) or len(dialogues) != DIALOGUE_COUNT:
        raise OrchestrationError("frozen index dialogue count mismatch")
    keys = []
    image_ids = set()
    dialog_ids = set()
    question_ids = set()
    for ordinal, dialog in enumerate(dialogues):
        did = str(dialog.get("dialog_id", ""))
        image_id = str(dialog.get("image_id", ""))
        turns = dialog.get("turns")
        if (did != f"mtgqa_{ordinal+1:06d}" or did in dialog_ids
                or not image_id or not isinstance(turns, list) or len(turns) != 3):
            raise OrchestrationError(f"invalid frozen dialogue at ordinal {ordinal}")
        dialog_ids.add(did)
        image_ids.add(image_id)
        for turn_id, turn in enumerate(turns, 1):
            qid = str(turn.get("question_id", ""))
            answers = turn.get("answers")
            if (turn.get("turn_id") != turn_id or not qid or qid in question_ids
                    or not str(turn.get("question", "")).strip()
                    or not isinstance(answers, list) or len(answers) != 1):
                raise OrchestrationError(f"invalid frozen turn: {did}/T{turn_id}")
            question_ids.add(qid)
            keys.append(f"{did}\t{turn_id}\t{qid}\n")
    sequence_hash = hashlib.sha256("".join(keys).encode("utf-8")).hexdigest()
    if sequence_hash != WORKLOAD_SHA256 or len(image_ids) != IMAGE_COUNT:
        raise OrchestrationError("frozen request sequence or image count mismatch")
    groups: dict[str, list[dict[str, Any]]] = {}
    for ordinal, dialog in enumerate(dialogues):
        row = dict(dialog)
        row["global_dialog_ordinal"] = ordinal
        groups.setdefault(str(dialog["image_id"]), []).append(row)
    return {"dialogues": dialogues, "groups": groups,
            "image_ids": list(groups), "index_sha256": digest,
            "workload_sha256": sequence_hash}


def model_revision() -> str:
    if MODEL_REF.is_symlink() or not MODEL_REF.is_file():
        raise OrchestrationError(f"model ref is missing/unsafe: {MODEL_REF}")
    revision = MODEL_REF.read_text(encoding="utf-8").strip()
    if revision != MODEL_REVISION:
        raise OrchestrationError(f"model revision changed: {revision}")
    return revision


def gpu_preflight() -> dict[str, Any]:
    command = ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,driver_version",
               "--format=csv,noheader,nounits"]
    completed = subprocess.run(command, check=True, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise OrchestrationError(f"expected one GPU; got {len(lines)}")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 6:
        raise OrchestrationError("malformed GPU preflight result")
    total, used, free, utilization = map(int, fields[1:5])
    if free < 20_000 or utilization > 10:
        raise OrchestrationError(
            f"GPU is not ready: free={free} MiB, utilization={utilization}%")
    return {"name": fields[0], "total_mib": total, "used_mib": used,
            "free_mib": free, "utilization_pct": utilization,
            "driver": fields[5]}


def free_space_preflight(min_gib: float) -> dict[str, Any]:
    value = os.statvfs(ROOT)
    free = value.f_bavail * value.f_frsize
    if free < int((min_gib + 5.0) * (1024 ** 3)):
        raise OrchestrationError(
            f"insufficient free space: {free/(1024**3):.2f} GiB; "
            f"need at least {min_gib+5.0:.2f} GiB")
    return {"free_gib": free / (1024 ** 3), "minimum_after_gib": min_gib}


def _source_gate(expected: Mapping[str, str]) -> None:
    actual = _protector().source_hashes()
    if actual != dict(expected):
        missing = sorted(set(expected) - set(actual))
        added = sorted(set(actual) - set(expected))
        changed = sorted(key for key in set(expected) & set(actual)
                         if expected[key] != actual[key])
        raise OrchestrationError(
            f"runtime source hash gate failed: missing={missing[:5]}, "
            f"added={added[:5]}, changed={changed[:5]}")


def _manifest_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "manifest_sha256"}


def make_manifest(run: Path, results: Path, workload: Mapping[str, Any],
                  before: Mapping[str, Any], shard_size: int, min_free: float,
                  gpu: Mapping[str, Any], space: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(before["source_hashes"])
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": str(uuid.uuid4()),
        "run_root": str(run), "results_root": str(results),
        "dataset": "MT-GQA-reconstructed",
        "official_benchmark_identity_claimed": False,
        "history_protocol": PROTOCOL,
        "history_policy": "method_local_generated",
        "index": str(INDEX),
        "index_sha256": workload["index_sha256"],
        "workload_sha256": workload["workload_sha256"],
        "n_images": IMAGE_COUNT, "n_dialogues": DIALOGUE_COUNT,
        "n_turns_per_method": REQUESTS_PER_METHOD,
        "n_t2_t3_requests_per_method": HITS_PER_METHOD,
        "n_stored_kv_cache_hits_per_stored_method": HITS_PER_METHOD,
        "n_stored_kv_cache_hits_total": HITS_PER_METHOD * (len(METHODS) - 1),
        "n_logical_requests": TOTAL_REQUESTS,
        "method_keys": list(METHODS),
        "method_order_policy": "zero-phase cyclic rotation by global dialogue ordinal",
        "seed": SEED, "max_new_tokens": MAX_NEW_TOKENS,
        "generation": "greedy / do_sample=False / token-wise argmax",
        "deterministic_cuda_mode_forced": False,
        "quality_metric": "strict_normalized_exact_match",
        "shard_size_images": shard_size,
        "n_shards_full": math.ceil(IMAGE_COUNT / shard_size),
        "smoke_dialogues": SMOKE_DIALOGUES,
        "min_free_after_gib": min_free,
        "model": "llava-hf/llava-v1.6-vicuna-7b-hf",
        "model_revision": MODEL_REVISION,
        "runtime_python": str(RUNTIME_PYTHON),
        "runtime_code_sha256": source,
        "code_sha256": source,
        "protected_before_manifest_sha256": before["manifest_sha256"],
        "protected_prior_entry_count": before["entry_count"],
        "gpu_preflight": dict(gpu),
        "disk_preflight": dict(space),
        "host": platform.node(),
        "created_at_unix": time.time(),
    }
    value["manifest_sha256"] = canonical_hash(_manifest_payload(value))
    return value


def validate_manifest(value: Mapping[str, Any], run: Path, results: Path,
                      workload: Mapping[str, Any]) -> None:
    if value.get("manifest_sha256") != canonical_hash(_manifest_payload(value)):
        raise OrchestrationError("immutable run manifest hash mismatch")
    expected = {
        "schema_version": SCHEMA_VERSION, "run_root": str(run),
        "results_root": str(results), "history_protocol": PROTOCOL,
        "index_sha256": workload["index_sha256"],
        "workload_sha256": workload["workload_sha256"],
        "n_images": IMAGE_COUNT, "n_dialogues": DIALOGUE_COUNT,
        "n_logical_requests": TOTAL_REQUESTS,
        "method_keys": list(METHODS), "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "model_revision": MODEL_REVISION,
    }
    mismatch = {key: (value.get(key), expected_value)
                for key, expected_value in expected.items()
                if value.get(key) != expected_value}
    if mismatch:
        raise OrchestrationError(f"immutable run manifest mismatch: {mismatch}")
    before = read_json(run / "protected_artifacts_before.json")
    if (before.get("manifest_sha256") != value.get("protected_before_manifest_sha256")
            or before.get("source_hashes") != value.get("runtime_code_sha256")):
        raise OrchestrationError("run manifest differs from protection snapshot")
    _source_gate(value["runtime_code_sha256"])


def _environment() -> dict[str, str]:
    env = dict(os.environ)
    env.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                "PYTHONPATH": str(ROOT)})
    return env


def run_streaming(command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.is_symlink() or log_path.parent.is_symlink():
        raise OrchestrationError(f"unsafe log path: {log_path}")
    with log_path.open("a", encoding="utf-8") as log:
        log.write("COMMAND " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(list(command), cwd=ROOT, env=_environment(),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
        os.fsync(log.fileno())
    if returncode:
        raise subprocess.CalledProcessError(returncode, list(command))


def _evaluator_command(stage: str, run: Path, experiment_id: str,
                       shard_index: int, shard_size: int, min_free: float) -> list[str]:
    stage_dir = run / stage / PROTOCOL
    temp_root = run / "_temporary_visual_kv" / stage / PROTOCOL
    command = [str(RUNTIME_PYTHON), str(EVALUATOR),
               "--protocol", PROTOCOL,
               "--index", str(INDEX),
               "--run-dir", str(stage_dir),
               "--temp-root", str(temp_root),
               "--experiment-id", experiment_id,
               "--shard-index", str(shard_index),
               "--shard-size", str(shard_size),
               "--expected-index-sha256", INDEX_SHA256,
               "--expected-workload-sha256", WORKLOAD_SHA256,
               "--expected-dialogs", str(SMOKE_DIALOGUES if stage == "smoke" else DIALOGUE_COUNT),
               "--seed", str(SEED),
               "--max-new-tokens", str(MAX_NEW_TOKENS),
               "--min-free-after-gib", str(min_free)]
    if stage == "smoke":
        command.extend(["--max-dialogs", str(SMOKE_DIALOGUES),
                        "--allow-partial-workload"])
    return command


def _selected_dialogues(workload: Mapping[str, Any], stage: str) -> list[dict[str, Any]]:
    dialogs = list(workload["dialogues"])
    return dialogs[:SMOKE_DIALOGUES] if stage == "smoke" else dialogs


def _groups_for_stage(workload: Mapping[str, Any], stage: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for ordinal, dialog in enumerate(_selected_dialogues(workload, stage)):
        row = dict(dialog)
        row["global_dialog_ordinal"] = ordinal
        groups.setdefault(str(dialog["image_id"]), []).append(row)
    return groups


def _selected_workload_hash(dialogues: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(
        f"{dialog['dialog_id']}\t{turn['turn_id']}\t{turn['question_id']}\n"
        for dialog in dialogues for turn in dialog["turns"])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalise_answer(value: Any) -> str:
    text = re.sub(r"[^\w\s]", " ", str(value).lower())
    return " ".join(word for word in text.split()
                    if word not in {"a", "an", "the"})


def _expected_prompt(dialog: Mapping[str, Any], turn_id: int,
                     answers: Sequence[str]) -> tuple[str, str]:
    lines = []
    for prior_id, answer in enumerate(answers, 1):
        lines.extend((f"Q{prior_id}: {dialog['turns'][prior_id-1]['question']}",
                      f"A{prior_id}: {answer}"))
    history = "\n".join(lines)
    body = []
    if history:
        body.extend((history, ""))
    body.extend((f"Current question Q{turn_id}: {dialog['turns'][turn_id-1]['question']}",
                 f"{SHORT_ANSWER_INSTRUCTION} ASSISTANT:"))
    return "USER: <image>\n" + "\n".join(body), history


def _validate_image_rows(rows: Sequence[Mapping[str, Any]],
                         dialogues: Sequence[Mapping[str, Any]],
                         *, global_logical: set[str],
                         global_physical: set[str]) -> tuple[int, int]:
    cursor = 0
    failed = retries = 0
    for dialog in dialogues:
        did = str(dialog["dialog_id"])
        ordinal = int(dialog["global_dialog_ordinal"])
        order = METHODS[ordinal % len(METHODS):] + METHODS[:ordinal % len(METHODS)]
        own_rows: dict[str, list[Mapping[str, Any]]] = {method: [] for method in METHODS}
        for turn_id in TURNS:
            block = rows[cursor:cursor+len(METHODS)]
            cursor += len(METHODS)
            if len(block) != len(METHODS) or tuple(
                    str(row.get("method_key")) for row in block) != order:
                raise OrchestrationError(f"method order/coverage changed: {did}/T{turn_id}")
            for row in block:
                method = str(row["method_key"])
                source_turn = dialog["turns"][turn_id-1]
                prior = own_rows[method]
                expected_prompt, expected_history = _expected_prompt(
                    dialog, turn_id, [str(item["prediction"]) for item in prior])
                logical = f"{PROTOCOL}:{did}:t{turn_id}:{method}"
                physical = str(row.get("physical_execution_id", ""))
                gold = str(source_turn["answers"][0])
                score = float(_normalise_answer(row.get("prediction", ""))
                              == _normalise_answer(gold))
                if (row.get("logical_request_id") != logical
                        or not physical or logical in global_logical
                        or physical in global_physical
                        or row.get("protocol") != PROTOCOL
                        or row.get("dialog_id") != did
                        or str(row.get("image_id")) != str(dialog["image_id"])
                        or int(row.get("turn_id", -1)) != turn_id
                        or str(row.get("question_id")) != str(source_turn["question_id"])
                        or str(row.get("question")) != str(source_turn["question"])
                        or row.get("prompt") != expected_prompt
                        or row.get("history_text") != expected_history
                        or row.get("history_answers")
                        != [str(item["prediction"]) for item in prior]
                        or row.get("history_source_request_ids")
                        != [str(item["logical_request_id"]) for item in prior]
                        or row.get("history_source_physical_execution_ids")
                        != [str(item["physical_execution_id"]) for item in prior]
                        or float(row.get("correct", -1)) != score):
                    raise OrchestrationError(f"request/history/score mismatch: {logical}")
                failed += int(row.get("status") != "ok")
                retries += int(row.get("retry_count", 0) or 0)
                global_logical.add(logical)
                global_physical.add(physical)
                own_rows[method].append(row)
            if turn_id == 1 and (
                    len({str(row.get("prediction")) for row in block}) != 1
                    or len({int(row.get("first_token_id", -1)) for row in block}) != 1):
                raise OrchestrationError(f"five-arm Turn-1 prediction mismatch: {did}")
    if cursor != len(rows):
        raise OrchestrationError("image artifact has extra request rows")
    return failed, retries


def validate_stage(run: Path, stage: str, workload: Mapping[str, Any],
                   manifest: Mapping[str, Any]) -> dict[str, Any]:
    stage_dir = run / stage / PROTOCOL
    config = read_json(stage_dir / "config.json")
    dialogues = _selected_dialogues(workload, stage)
    groups = _groups_for_stage(workload, stage)
    image_ids = list(groups)
    shard_size = int(manifest["shard_size_images"])
    expected_rows = len(dialogues) * 3 * len(METHODS)
    expected_config = {
        "experiment_id": manifest["experiment_id"],
        "protocol": PROTOCOL,
        "dialogues_file_sha256": INDEX_SHA256,
        "source_full_workload_sha256": WORKLOAD_SHA256,
        "selected_workload_sha256": _selected_workload_hash(dialogues),
        "n_dialogs": len(dialogues),
        "n_turns": len(dialogues) * 3,
        "n_images": len(image_ids),
        "n_requests": expected_rows,
        "shard_size": shard_size,
        "n_shards": math.ceil(len(image_ids)/shard_size),
        "seed": SEED,
        "method_keys": list(METHODS),
        "model_revision": MODEL_REVISION,
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    mismatches = {key: (config.get(key), expected)
                  for key, expected in expected_config.items()
                  if config.get(key) != expected}
    if mismatches:
        raise OrchestrationError(f"{stage} config mismatch: {mismatches}")
    image_dir = stage_dir / "images"
    present = {path.stem for path in image_dir.glob("*.json")}
    if present != set(image_ids):
        raise OrchestrationError(
            f"{stage} image artifact coverage mismatch: missing={len(set(image_ids)-present)}, "
            f"extra={len(present-set(image_ids))}")
    logical: set[str] = set()
    physical: set[str] = set()
    file_hashes: dict[str, str] = {}
    failed = retries = recovered_images = 0
    for shard_index in range(math.ceil(len(image_ids)/shard_size)):
        expected_ids = image_ids[shard_index*shard_size:(shard_index+1)*shard_size]
        marker_path = stage_dir / "shards" / f"shard_{shard_index:03d}.json"
        marker = read_json(marker_path)
        if (marker.get("artifact_content_sha256")
                != canonical_hash({key: item for key, item in marker.items()
                                   if key != "artifact_content_sha256"})
                or marker.get("experiment_id") != manifest["experiment_id"]
                or marker.get("protocol") != PROTOCOL
                or marker.get("complete") is not True
                or marker.get("shard_index") != shard_index
                or marker.get("image_ids") != expected_ids
                or marker.get("completed_image_ids") != expected_ids
                or marker.get("failed_request_count") != 0
                or marker.get("duplicate_request_count") != 0):
            raise OrchestrationError(f"invalid {stage} shard marker: {marker_path}")
        marker_file_hashes = marker.get("image_artifact_file_sha256")
        marker_content_hashes = marker.get("image_artifact_content_sha256")
        if (not isinstance(marker_file_hashes, dict)
                or not isinstance(marker_content_hashes, dict)
                or set(marker_file_hashes) != set(expected_ids)
                or set(marker_content_hashes) != set(expected_ids)):
            raise OrchestrationError(f"{stage} marker image hashes incomplete")
        shard_rows = 0
        for image_id in expected_ids:
            path = image_dir / f"{image_id}.json"
            if path.is_symlink() or not path.is_file():
                raise OrchestrationError(f"unsafe image artifact: {path}")
            digest = file_hash(path)
            if digest != marker_file_hashes[image_id]:
                raise OrchestrationError(f"image artifact file SHA mismatch: {path}")
            artifact = read_json(path)
            body_hash = canonical_hash({key: item for key, item in artifact.items()
                                        if key != "artifact_content_sha256"})
            if (artifact.get("artifact_content_sha256") != body_hash
                    or marker_content_hashes[image_id] != body_hash
                    or artifact.get("experiment_id") != manifest["experiment_id"]
                    or artifact.get("protocol") != PROTOCOL
                    or artifact.get("image_id") != image_id
                    or artifact.get("validation", {}).get("passed") is not True
                    or artifact.get("dialog_ids")
                    != [dialog["dialog_id"] for dialog in groups[image_id]]):
                raise OrchestrationError(f"image artifact identity mismatch: {path}")
            recovered_images += int(bool(artifact.get(
                "recovered_incomplete_temp_store_before_build", False)))
            rows = artifact.get("rows")
            if not isinstance(rows, list) or len(rows) != len(groups[image_id])*3*len(METHODS):
                raise OrchestrationError(f"image request count mismatch: {path}")
            image_failed, image_retries = _validate_image_rows(
                rows, groups[image_id], global_logical=logical,
                global_physical=physical)
            failed += image_failed
            retries += image_retries
            shard_rows += len(rows)
            file_hashes[image_id] = digest
        if (int(marker.get("logical_request_count", -1)) != shard_rows
                or int(marker.get("physical_execution_count", -1)) != shard_rows):
            raise OrchestrationError(f"{stage} shard request count mismatch")
    if len(logical) != expected_rows or len(physical) != expected_rows or failed or retries:
        raise OrchestrationError(
            f"{stage} request validation failed: logical={len(logical)}, "
            f"physical={len(physical)}, failed={failed}, retries={retries}")
    return {"passed": True, "stage": stage,
            "n_images": len(image_ids), "n_dialogues": len(dialogues),
            "n_requests": expected_rows, "n_cache_hits": len(dialogues)*2*(len(METHODS)-1),
            "failed_requests": failed, "completed_row_retry_count": retries,
            "incomplete_image_rebuild_count": recovered_images,
            "duplicate_logical_requests": 0,
            "image_artifact_file_sha256": file_hashes}


def _raw_path(run: Path) -> Path:
    return run / "raw.jsonl"


def publish_raw(run: Path, workload: Mapping[str, Any],
                validation: Mapping[str, Any]) -> dict[str, Any]:
    path = _raw_path(run)
    expected_rows = int(validation["n_requests"])
    if path.is_file() and not path.is_symlink():
        return {"path": str(path), "sha256": file_hash(path),
                "rows": sum(1 for _ in path.open("rb"))}
    if os.path.lexists(path):
        raise OrchestrationError(f"unsafe raw output path: {path}")
    image_ids = list(_groups_for_stage(workload, "full"))
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    digest = hashlib.sha256()
    rows_written = 0
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for image_id in image_ids:
                artifact = read_json(run / "full" / PROTOCOL / "images" / f"{image_id}.json")
                for row in artifact["rows"]:
                    line = json.dumps(row, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False, allow_nan=False) + "\n"
                    handle.write(line)
                    digest.update(line.encode("utf-8"))
                    rows_written += 1
            handle.flush()
            os.fsync(handle.fileno())
        if rows_written != expected_rows:
            raise OrchestrationError(f"raw rows {rows_written} != {expected_rows}")
        os.link(temporary, path)
        temporary.unlink()
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(path), "sha256": digest.hexdigest(), "rows": rows_written}


def _progress(run: Path, **fields: Any) -> None:
    path = run / "progress.json"
    value = read_json(path) if path.exists() else {"schema_version": SCHEMA_VERSION}
    value.update(fields)
    value["updated_at_unix"] = time.time()
    atomic_json(path, value)


def _record_attempt(run: Path, value: Mapping[str, Any]) -> None:
    """Durably record each shard invocation, including failed/restarted work."""
    path = run / "attempts.jsonl"
    line = json.dumps(dict(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def attempt_summary(run: Path) -> dict[str, Any]:
    path = run / "attempts.jsonl"
    if not path.is_file() or path.is_symlink():
        return {"shard_invocations": 0, "successful_invocations": 0,
                "failed_invocations": 0, "interrupted_invocations": 0,
                "invocations_with_existing_marker": 0,
                "retried_shard_invocations": 0,
                "retry_count_scope": "additional evaluator invocations for a stage/shard lacking a complete marker",
                "failure_events": [], "interrupted_attempt_ids": []}
    starts: dict[str, dict[str, Any]] = {}
    finishes: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise OrchestrationError(
                    f"malformed shard-attempt ledger line {number}") from error
            if not isinstance(row, dict) or row.get("event") not in {"start", "finish"}:
                raise OrchestrationError("invalid shard-attempt ledger event")
            attempt_id = str(row.get("attempt_id", ""))
            if not attempt_id:
                raise OrchestrationError("shard-attempt event omits identity")
            target = starts if row["event"] == "start" else finishes
            if attempt_id in target:
                raise OrchestrationError("duplicate shard-attempt event")
            target[attempt_id] = row
    if set(finishes) - set(starts):
        raise OrchestrationError("shard-attempt finish has no start")
    failed = [row for row in finishes.values() if row.get("status") != "ok"]
    interrupted = sorted(set(starts) - set(finishes))
    started_per_shard: Counter[tuple[str, int]] = Counter(
        (str(row.get("stage")), int(row.get("shard_index", -1)))
        for row in starts.values() if not row.get("existing_marker"))
    retried_shard_invocations = sum(max(0, count - 1)
                                    for count in started_per_shard.values())
    return {
        "shard_invocations": len(starts),
        "successful_invocations": len(finishes) - len(failed),
        "failed_invocations": len(failed),
        "interrupted_invocations": len(interrupted),
        "invocations_with_existing_marker": sum(
            bool(row.get("existing_marker")) for row in starts.values()),
        "retried_shard_invocations": retried_shard_invocations,
        "retry_count_scope": (
            "additional evaluator invocations for a stage/shard lacking a "
            "complete marker; completed logical rows are independently deduplicated"),
        "failure_events": failed,
        "interrupted_attempt_ids": interrupted,
    }


def _stage(run: Path, name: str, workload: Mapping[str, Any],
           manifest: Mapping[str, Any]) -> dict[str, Any]:
    validation_path = run / f"{name}_validation.json"
    if validation_path.is_file() and not validation_path.is_symlink():
        expected = read_json(validation_path)
        observed = validate_stage(run, name, workload, manifest)
        if expected != observed:
            raise OrchestrationError(
                f"completed {name} stage changed after publication")
        return observed
    if os.path.lexists(validation_path):
        raise OrchestrationError(f"unsafe stage validation path: {validation_path}")
    image_count = len(_groups_for_stage(workload, name))
    shard_size = int(manifest["shard_size_images"])
    for shard_index in range(math.ceil(image_count/shard_size)):
        _source_gate(manifest["runtime_code_sha256"])
        free_space_preflight(float(manifest["min_free_after_gib"]))
        _progress(run, status="running", stage=name, shard_index=shard_index)
        command = _evaluator_command(
            name, run, str(manifest["experiment_id"]), shard_index,
            shard_size, float(manifest["min_free_after_gib"]))
        attempt_id = str(uuid.uuid4())
        marker = run / name / PROTOCOL / "shards" / f"shard_{shard_index:03d}.json"
        _record_attempt(run, {
            "event": "start", "attempt_id": attempt_id,
            "stage": name, "shard_index": shard_index,
            "existing_marker": marker.is_file(), "started_at_unix": time.time()})
        try:
            run_streaming(command, run / "logs" / name / f"shard_{shard_index:03d}.log")
        except BaseException as error:
            _record_attempt(run, {
                "event": "finish", "attempt_id": attempt_id,
                "stage": name, "shard_index": shard_index,
                "status": "failed", "error_type": type(error).__name__,
                "error": str(error), "finished_at_unix": time.time()})
            raise
        _record_attempt(run, {
            "event": "finish", "attempt_id": attempt_id,
            "stage": name, "shard_index": shard_index,
            "status": "ok", "finished_at_unix": time.time()})
    validated = validate_stage(run, name, workload, manifest)
    atomic_json(validation_path, validated, exclusive=True)
    return validated


def _run_reporter(run: Path, results: Path) -> None:
    run_streaming([str(RUNTIME_PYTHON), str(REPORTER),
                   "--run-dir", str(run), "--results-dir", str(results)],
                  run / "logs" / "report.log")


def _validate_published_report(run: Path, results: Path) -> dict[str, Any]:
    completion = read_json(results / "COMPLETED")
    validation = read_json(results / "report_validation.json")
    receipt = read_json(results / "report_artifacts.json")
    if (completion.get("passed") is not True
            or validation.get("passed") is not True
            or receipt.get("passed") is not True
            or int(completion.get("logical_requests", -1)) != TOTAL_REQUESTS
            or completion.get("report_validation_sha256")
            != file_hash(results / "report_validation.json")
            or completion.get("report_artifacts_sha256")
            != file_hash(results / "report_artifacts.json")
            or receipt.get("source_config_sha256")
            != file_hash(run / "config.json")
            or receipt.get("source_manifest_sha256")
            != file_hash(run / "manifest.json")
            or receipt.get("source_protection_validation_sha256")
            != file_hash(run / "protected_artifacts_validation.json")):
        raise OrchestrationError("published report completion receipt is invalid")
    files = receipt.get("files_sha256")
    if not isinstance(files, dict) or not files:
        raise OrchestrationError("published report omits file hashes")
    for name, digest in files.items():
        if (not isinstance(name, str) or Path(name).name != name
                or file_hash(results / name) != digest):
            raise OrchestrationError(f"published report file changed: {name}")
    return validation


def _lock() -> int:
    descriptor = os.open(LOCK, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(descriptor)
        raise OrchestrationError(f"another MT-GQA five-arm run holds {LOCK}") from error
    return descriptor


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", choices=("smoke", "full"), default="full")
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--min-free-after-gib", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protector = _protector()
    run, results = protector.validate_output_roots(args.run_root, args.results_root)
    workload = frozen_workload()
    for source in (EVALUATOR, REPORTER, PROTECTOR):
        if source.is_symlink() or not source.is_file():
            raise OrchestrationError(f"runtime source is missing: {source}")
    if not RUNTIME_PYTHON.is_file():
        raise OrchestrationError(f"required runtime Python is missing: {RUNTIME_PYTHON}")
    descriptor = _lock()
    try:
        if not args.resume:
            if os.path.lexists(run) or os.path.lexists(results):
                raise FileExistsError("new run/results roots must be absent; use --resume")
            shard_size = DEFAULT_SHARD_SIZE if args.shard_size is None else args.shard_size
            min_free = (MIN_FREE_AFTER_GIB if args.min_free_after_gib is None
                        else args.min_free_after_gib)
            if not MIN_SHARD_SIZE <= shard_size <= MAX_SHARD_SIZE:
                raise ValueError("shard size must be 40..60 images")
            if min_free < MIN_FREE_AFTER_GIB:
                raise ValueError("minimum free space may not be below 30 GiB")
            revision = model_revision()
            gpu = gpu_preflight()
            space = free_space_preflight(min_free)
            _, before = protector.record_before(run, results)
            _source_gate(before["source_hashes"])
            manifest = make_manifest(run, results, workload, before,
                                     shard_size, min_free, gpu, space)
            atomic_json(run / "manifest.json", manifest, exclusive=True)
            config = {"schema_version": SCHEMA_VERSION,
                      "experiment_id": manifest["experiment_id"],
                      "index": str(INDEX), "index_sha256": INDEX_SHA256,
                      "workload_sha256": WORKLOAD_SHA256,
                      "n_images": IMAGE_COUNT, "n_dialogues": DIALOGUE_COUNT,
                      "n_requests": TOTAL_REQUESTS,
                      "n_requests_per_method": REQUESTS_PER_METHOD,
                      "n_t2_t3_requests_per_method": HITS_PER_METHOD,
                      "n_stored_kv_cache_hits_per_stored_method": HITS_PER_METHOD,
                      "n_stored_kv_cache_hits_total": HITS_PER_METHOD * (len(METHODS) - 1),
                      "method_keys": list(METHODS),
                      "history_protocol": PROTOCOL,
                      "history_policy": "method_local_generated",
                      "seed": SEED, "max_new_tokens": MAX_NEW_TOKENS,
                      "manifest_sha256": manifest["manifest_sha256"],
                      "runtime_code_sha256": manifest["runtime_code_sha256"],
                      "model_revision": revision,
                      "run_root": str(run), "results_root": str(results)}
            atomic_json(run / "config.json", config, exclusive=True)
            _progress(run, status="running", stage="initializing")
        else:
            if run.is_symlink() or not run.is_dir():
                raise OrchestrationError(f"resume root is missing/unsafe: {run}")
            manifest = read_json(run / "manifest.json")
            validate_manifest(manifest, run, results, workload)
            if (args.shard_size is not None
                    and args.shard_size != manifest["shard_size_images"]):
                raise OrchestrationError("resume shard size differs from immutable manifest")
            if (args.min_free_after_gib is not None
                    and args.min_free_after_gib != manifest["min_free_after_gib"]):
                raise OrchestrationError("resume free-space reserve differs from immutable manifest")
            if (run / "COMPLETED").is_file():
                completed = read_json(run / "COMPLETED")
                if (completed.get("passed") is not True
                        or completed.get("experiment_id")
                        != manifest["experiment_id"]
                        or int(completed.get("n_requests", -1))
                        != TOTAL_REQUESTS
                        or completed.get("run_validation_sha256")
                        != file_hash(run / "validation.json")
                        or completed.get("report_validation_sha256")
                        != file_hash(results / "report_validation.json")):
                    raise OrchestrationError("run completion receipt is invalid")
                _validate_published_report(run, results)
                protector.verify_after(run, results)
                print(json.dumps({"status": "already_complete", "run_root": str(run),
                                  "results_root": str(results)}))
                return 0
            model_revision()
            gpu_preflight()
        smoke = _stage(run, "smoke", workload, manifest)
        if args.stop_after == "smoke":
            _progress(run, status="partial", stage="smoke_complete")
            print(json.dumps({"status": "smoke_complete", "validation": smoke,
                              "run_root": str(run), "resume_required": True}, indent=2))
            return 0
        full = _stage(run, "full", workload, manifest)
        if (full["n_images"] != IMAGE_COUNT
                or full["n_dialogues"] != DIALOGUE_COUNT
                or full["n_requests"] != TOTAL_REQUESTS):
            raise OrchestrationError("full run coverage is incomplete")
        raw = publish_raw(run, workload, full)
        if raw["rows"] != TOTAL_REQUESTS:
            raise OrchestrationError("raw JSONL coverage is incomplete")
        _source_gate(manifest["runtime_code_sha256"])
        protection_path, protection = protector.verify_after(run, results)
        attempts = attempt_summary(run)
        recorded_retry_count = int(attempts["retried_shard_invocations"])
        validation = {"schema_version": SCHEMA_VERSION, "passed": True,
                      "dataset": "MT-GQA-reconstructed",
                      "protocol": PROTOCOL,
                      "n_images": IMAGE_COUNT, "n_dialogues": DIALOGUE_COUNT,
                      "n_requests": TOTAL_REQUESTS,
                      "requests_per_method": REQUESTS_PER_METHOD,
                      "t2_t3_requests_per_method": HITS_PER_METHOD,
                      "stored_kv_cache_hits_per_stored_method": HITS_PER_METHOD,
                      "total_stored_kv_cache_hits": HITS_PER_METHOD * (len(METHODS)-1),
                      "failed_requests": 0,
                      "retry_count": recorded_retry_count,
                      "retry_count_scope": attempts["retry_count_scope"],
                      "completed_logical_row_retry_count": 0,
                      "incomplete_image_rebuild_count": (
                          smoke["incomplete_image_rebuild_count"]
                          + full["incomplete_image_rebuild_count"]),
                      "execution_attempts": attempts,
                      "duplicate_logical_requests": 0,
                      "missing_logical_requests": 0,
                      "index_sha256": INDEX_SHA256,
                      "workload_sha256": WORKLOAD_SHA256,
                      "source_sha256": canonical_hash(manifest["runtime_code_sha256"]),
                      "raw_jsonl": raw,
                      "smoke_validation": str(run / "smoke_validation.json"),
                      "full_validation": str(run / "full_validation.json"),
                      "protection_validation": str(protection_path),
                      "protected_artifacts_passed": protection["passed"],
                      "protected_artifacts": protection,
                      "checks": {
                          "frozen_workload_hash": True,
                          "full_five_arm_coverage": True,
                          "method_local_generated_history": True,
                          "strict_scores_recomputed": True,
                          "zero_failed_final_rows": True,
                          "zero_duplicate_logical_requests": True,
                          "zero_missing_logical_requests": True,
                          "shard_attempts_durably_recorded": True,
                          "prior_artifacts_unchanged": bool(protection["passed"]),
                          "source_files_unchanged": not any(protection[key]
                              for key in ("source_missing_paths",
                                          "source_changed_paths",
                                          "source_added_paths")),
                      }}
        atomic_json(run / "validation.json", validation)
        if os.path.lexists(results):
            if results.is_symlink() or not results.is_dir():
                raise OrchestrationError(f"unsafe results root: {results}")
        else:
            results.mkdir(parents=False, exist_ok=False)
        if not (results / "COMPLETED").is_file():
            _run_reporter(run, results)
        report_validation = _validate_published_report(run, results)
        _source_gate(manifest["runtime_code_sha256"])
        protector.verify_after(run, results)
        completed = {"schema_version": SCHEMA_VERSION,
                     "passed": True, "experiment_id": manifest["experiment_id"],
                     "n_requests": TOTAL_REQUESTS,
                     "run_validation_sha256": file_hash(run / "validation.json"),
                     "report_validation_sha256": file_hash(results / "report_validation.json"),
                     "completed_at_unix": time.time()}
        atomic_json(run / "COMPLETED", completed, exclusive=True)
        _progress(run, status="complete", stage="complete")
        print(json.dumps({"status": "complete", "run_root": str(run),
                          "results_root": str(results),
                          "n_requests": TOTAL_REQUESTS}, indent=2))
        return 0
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())

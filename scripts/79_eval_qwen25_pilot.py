#!/usr/bin/env python3
"""Run a gated Qwen2.5-VL GQA or MT-GQA three-arm pilot.

The workload and method schedule are fixed before the model is loaded. GQA
questions are independent; MT-GQA histories contain only each arm's own
generated answers. Image file access and RGB decode occur outside request
timers; the Qwen runner owns prompt preparation through synchronized TTFT.
"""
from __future__ import annotations

import argparse
import csv
import io
import gc
import hashlib
import json
import math
import os
import random
import sys
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mmimpress.dataset import exact_score  # noqa: E402


SCHEMA_VERSION = "qwen25-image-only-pilot-v1"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
SEED = 1234
METHODS = ("recompute", "fullload", "ours25")
GQA_INDEX = ROOT / "data/index.json"
GQA_INDEX_SHA256 = "514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a"
GQA_WORKLOAD_SHA256 = "97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66"
MT_INDEX = ROOT / "data/mt_gqa/dialogues.json"
MT_INDEX_SHA256 = "2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224"
MT_WORKLOAD_SHA256 = "0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62"
VALIDATOR_SCHEMA = "qwen25-gpu-correctness-v1"
CHECKPOINT_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
STRUCTURAL_GATES = (
    "cpu_score_reference", "runtime_load", "source_capture", "geometry",
    "gpu_score_reference", "query_independence", "persistence", "roundtrip",
    "repacked_full100", "capture", "io", "request_isolation", "history",
)
NUMERICAL_GATES = ("fullload", "prefix25")
ADAPTER_SOURCES = (
    "mmimpress/qwen25/runner.py", "mmimpress/qwen25/vision.py",
    "mmimpress/qwen25/store.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True,
                  ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _append_jsonl(handle: Any, value: Any) -> None:
    handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False,
                            allow_nan=False) + "\n")
    handle.flush()


def _sync_jsonl(handle: Any) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def _gpu_inventory() -> dict[str, Any]:
    """Inspect GPU 0 compute processes; unavailable telemetry fails closed."""
    command = ["nvidia-smi", "--id=0",
               "--query-compute-apps=pid,process_name,used_gpu_memory",
               "--format=csv,noheader,nounits"]
    completed = subprocess.run(command, capture_output=True, text=True,
                               check=False, timeout=15)
    if completed.returncode:
        raise RuntimeError(f"GPU process inventory failed: {completed.stderr.strip()}")
    processes = []
    for row in csv.reader(io.StringIO(completed.stdout)):
        if not row or not any(cell.strip() for cell in row):
            continue
        if len(row) != 3:
            raise RuntimeError(f"unrecognized GPU process row: {row}")
        processes.append({"pid": int(row[0].strip()),
                          "process_name": row[1].strip(),
                          "used_gpu_memory_mib": float(row[2].strip())})
    foreign = [row for row in processes if row["pid"] != os.getpid()]
    return {"gpu_index": 0, "self_pid": os.getpid(),
            "processes": processes, "foreign_processes": foreign,
            "foreign_process_count": len(foreign),
            "observed_unix": time.time()}


def _record_gpu_inventory(handle: Any, phase: str,
                          image_id: str | None = None) -> None:
    inventory = _gpu_inventory()
    _append_jsonl(handle, {"phase": phase, "image_id": image_id,
                           **inventory})
    _sync_jsonl(handle)
    if inventory["foreign_processes"]:
        raise RuntimeError(
            f"concurrent GPU compute process during {phase}: "
            f"{inventory['foreign_processes']}")


def _image_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _rank(namespace: str, *parts: str) -> str:
    fields = (SCHEMA_VERSION, str(SEED), namespace, *parts)
    payload = bytearray()
    for field in fields:
        raw = field.encode("utf-8")
        payload.extend(len(raw).to_bytes(8, "big"))
        payload.extend(raw)
    return hashlib.sha256(payload).hexdigest()


def _gqa_workload(index_path: Path, max_images: int) -> dict[str, Any]:
    index_hash = sha256_file(index_path)
    source = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise ValueError("GQA index must be a JSON list")
    frozen = index_path.resolve() == GQA_INDEX.resolve() and index_hash == GQA_INDEX_SHA256
    if frozen:
        if len(source) != 40:
            raise ValueError("frozen GQA index must contain 40 images")
        picked = source[:max_images]
        q_start = 4
    else:
        # A changed or alternative index is a newly named deterministic pilot.
        eligible = [row for row in source if len(row.get("questions", [])) >= 6]
        picked = sorted(eligible, key=lambda row: (
            _rank("gqa-image", str(row["image_id"])), str(row["image_id"]),
        ))[:max_images]
        q_start = 0
    if not picked:
        raise ValueError("GQA index has no eligible six-question images")
    full_rows = [
        (str(row["image_id"]), str(q["question_id"]))
        for row in source for q in row["questions"][4:10]
    ]
    full_hash = hashlib.sha256("\n".join(
        f"{image}\t{question}" for image, question in full_rows
    ).encode("utf-8")).hexdigest()
    if frozen and full_hash != GQA_WORKLOAD_SHA256:
        raise ValueError("frozen GQA question order changed")
    images = []
    seen_images: set[str] = set()
    seen_questions: set[str] = set()
    for row in picked:
        image_id = str(row["image_id"])
        if image_id in seen_images:
            raise ValueError(f"duplicate GQA image: {image_id}")
        seen_images.add(image_id)
        questions = []
        for turn_id, q in enumerate(row["questions"][q_start:q_start + 6], 1):
            qid = str(q["question_id"])
            if qid in seen_questions:
                raise ValueError(f"duplicate GQA question: {qid}")
            seen_questions.add(qid)
            answers = q.get("answers", [q.get("answer")])
            if len(answers) != 1 or not str(answers[0]).strip():
                raise ValueError(f"GQA question {qid} lacks one gold answer")
            questions.append({"turn_id": turn_id, "question_id": qid,
                              "question": str(q["question"]),
                              "gold": str(answers[0])})
        if len(questions) != 6:
            raise ValueError(f"image {image_id} lacks six questions")
        path = _image_path(str(row["image_path"]))
        images.append({"image_id": image_id, "image_path": str(path),
                       "image_sha256": sha256_file(path),
                       "dialog_id": None, "turns": questions})
    return {"dataset": "gqa", "workload_name": (
        "frozen-gqa40-q5to10" if frozen else "deterministic-new-gqa-six-question-pilot"),
        "source_index": str(index_path.resolve()), "source_index_sha256": index_hash,
        "source_full_workload_sha256": full_hash, "frozen_source": frozen,
        "images": images}


def _mt_workload(index_path: Path, max_images: int) -> dict[str, Any]:
    index_hash = sha256_file(index_path)
    source = json.loads(index_path.read_text(encoding="utf-8"))
    if source.get("benchmark_type") != "MT-GQA-reconstructed":
        raise ValueError("MT index is not MT-GQA-reconstructed")
    dialogs = source.get("dialogues")
    if not isinstance(dialogs, list):
        raise ValueError("MT index lacks dialogues")
    frozen = index_path.resolve() == MT_INDEX.resolve() and index_hash == MT_INDEX_SHA256
    if frozen:
        workload_hash = hashlib.sha256("".join(
            f"{d['dialog_id']}\t{t['turn_id']}\t{t['question_id']}\n"
            for d in dialogs for t in d["turns"]
        ).encode("utf-8")).hexdigest()
        if workload_hash != MT_WORKLOAD_SHA256:
            raise ValueError("frozen MT-GQA dialogue order changed")
    by_image: dict[str, list[Mapping[str, Any]]] = {}
    for dialog in dialogs:
        if len(dialog.get("turns", [])) != 3:
            continue
        by_image.setdefault(str(dialog["image_id"]), []).append(dialog)
    image_ids = sorted(by_image, key=lambda iid: (_rank("mt-image", iid), iid))[:max_images]
    images = []
    for image_id in image_ids:
        dialog = min(by_image[image_id], key=lambda d: (
            _rank("mt-dialog", image_id, str(d["dialog_id"])), str(d["dialog_id"]),
        ))
        turns = []
        for turn_id, turn in enumerate(dialog["turns"], 1):
            if int(turn["turn_id"]) != turn_id or len(turn.get("answers", [])) != 1:
                raise ValueError(f"malformed MT dialogue: {dialog['dialog_id']}")
            turns.append({"turn_id": turn_id,
                          "question_id": str(turn["question_id"]),
                          "question": str(turn["question"]),
                          "gold": str(turn["answers"][0])})
        path = _image_path(str(dialog["image_path"]))
        images.append({"image_id": image_id, "image_path": str(path),
                       "image_sha256": sha256_file(path),
                       "dialog_id": str(dialog["dialog_id"]), "turns": turns})
    if not images:
        raise ValueError("MT index has no complete three-turn dialogue")
    return {"dataset": "mt_gqa_reconstructed",
            "workload_name": "deterministic-40-image-generated-history-subset",
            "official_benchmark_identity_claimed": False,
            "source_index": str(index_path.resolve()),
            "source_index_sha256": index_hash, "frozen_source": frozen,
            "images": images}


def _method_rotation(image_index: int) -> list[str]:
    offset = (int(image_index) + SEED) % len(METHODS)
    return list(METHODS[offset:] + METHODS[:offset])


def build_manifest(dataset: str, index_path: Path, max_images: int) -> dict[str, Any]:
    if max_images < 1 or max_images > 40:
        raise ValueError("max-images must be between 1 and 40")
    workload = (_gqa_workload(index_path, max_images) if dataset == "gqa"
                else _mt_workload(index_path, max_images))
    for image_index, image in enumerate(workload["images"]):
        image["method_order"] = _method_rotation(image_index)
    manifest = {"schema_version": SCHEMA_VERSION, "model_id": MODEL_ID,
                "seed": SEED, "max_new_tokens": 16,
                "processor": {"min_pixels": 256 * 28 * 28,
                              "max_pixels": 1024 * 28 * 28},
                "chunk_size": 64, "nominal_budget_ratio": 0.25,
                "history_policy": ("none_independent_questions" if dataset == "gqa"
                                   else "method_own_generated_answers"),
                "timing_boundary": (
                    "request starts before prompt/tokenization/input preparation; "
                    "TTFT ends after first output token materialization and CUDA sync"),
                "dataset_file_read_and_image_decode_in_ttft": False,
                "page_cache_conditioning_in_ttft": False,
                "methods": list(METHODS), **workload}
    manifest["manifest_sha256"] = canonical_hash(manifest)
    return manifest


def _numeric_comparison_only(comparison: Mapping[str, Any]) -> tuple[bool, bool]:
    """Return (allowed, numerical failure) for one logit/output comparator."""
    if not isinstance(comparison, Mapping):
        return False, False
    if any(comparison.get(key) is not True for key in (
            "first_token_identical", "generated_tokens_identical",
            "prediction_identical")):
        return False, False
    if "reason" in comparison:  # Missing logits or shape mismatch is structural.
        return False, False
    if comparison.get("passed") is True:
        return comparison.get("numerically_close") is True, False
    if comparison.get("passed") is not False or comparison.get("numerically_close") is not False:
        return False, False
    for key in ("max_abs_logit_error", "max_relative_logit_error"):
        value = comparison.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)) or float(value) < 0:
            return False, False
    return True, True


def _numerical_failure_only(name: str, gate: Mapping[str, Any]) -> bool:
    if not isinstance(gate, Mapping) or gate.get("status") != "FAIL":
        return False
    if "error" in gate or "traceback" in gate:
        return False
    details = gate.get("details")
    if not isinstance(details, Mapping):
        return False
    rows = details.get("per_question")
    if not isinstance(rows, list) or not rows:
        return False
    saw_numerical_failure = False
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        if name == "fullload":
            request = row.get("ssd_request")
            if not isinstance(request, Mapping) or request.get("vision_calls") != 0:
                return False
            comparisons = (row.get("ssd_vs_memory"), row.get("ssd_vs_recompute"))
        elif name == "prefix25":
            request = row.get("compact_request")
            if (row.get("same_full_logical_suffix_positions") is not True or
                    row.get("dense_and_compact_cache_slots_correct") is not True or
                    row.get("same_selected_visual_set") is not True or
                    not isinstance(request, Mapping) or request.get("vision_calls") != 0):
                return False
            comparisons = (row.get("dense_vs_compact"),)
        else:
            return False
        for comparison in comparisons:
            allowed, numeric_fail = _numeric_comparison_only(comparison)
            if not allowed:
                return False
            saw_numerical_failure |= numeric_fail
    return saw_numerical_failure


def _gate(validation_path: Path, *,
          diagnostic_after_numerical_fail: bool = False) -> dict[str, Any]:
    """Bind the pilot to exact validator evidence before loading a GPU model."""
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not isinstance(validation, dict) or validation.get("schema_version") != VALIDATOR_SCHEMA:
        raise RuntimeError("wrong Qwen GPU validation schema")
    gates = validation.get("gates")
    if not isinstance(gates, dict) or set(gates) != set(STRUCTURAL_GATES + NUMERICAL_GATES):
        raise RuntimeError("GPU validation gates are missing or unexpected")
    statuses = {}
    for name, gate in gates.items():
        if not isinstance(gate, dict) or gate.get("status") not in {"PASS", "FAIL", "NOT RUN"}:
            raise RuntimeError(f"invalid GPU validation gate: {name}")
        statuses[name] = gate["status"]
    source_hashes = validation.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise RuntimeError("validator source hashes are missing")
    for relative in ADAPTER_SOURCES:
        current = sha256_file(ROOT / relative)
        if source_hashes.get(relative) != current:
            raise RuntimeError(f"adapter source changed after validation: {relative}")
    config = validation.get("configuration")
    if (not isinstance(config, dict) or config.get("seed") != SEED or
            config.get("max_new_tokens") != 16 or config.get("chunk_size") != 64 or
            config.get("budget_ratio") != 0.25 or
            config.get("attention_backend") != "sdpa" or
            validation.get("model_revision") != CHECKPOINT_REVISION or
            validation.get("source", {}).get("gqa_index_sha256") != GQA_INDEX_SHA256):
        raise RuntimeError("validator model, backend, workload, or pilot config mismatch")
    structural_bad = {name: statuses[name] for name in STRUCTURAL_GATES
                      if statuses[name] != "PASS"}
    if structural_bad:
        raise RuntimeError(f"structural GPU gates must all PASS: {structural_bad}")
    numerical_status = {name: statuses[name] for name in NUMERICAL_GATES}
    if diagnostic_after_numerical_fail:
        if (validation.get("status") != "FAIL" or
                validation.get("pilot_eligible") is not False or
                not any(value == "FAIL" for value in numerical_status.values()) or
                any(value == "NOT RUN" for value in numerical_status.values())):
            raise RuntimeError("diagnostic run requires a completed numerical-only validation FAIL")
        for name in NUMERICAL_GATES:
            if statuses[name] == "FAIL" and not _numerical_failure_only(name, gates[name]):
                raise RuntimeError(f"{name} failure is not proven numerical-only")
        benchmark_validated = False
        run_mode = "diagnostic_numerical_mismatch"
    else:
        if (validation.get("status") != "PASS" or
                validation.get("pilot_eligible") is not True or
                any(value != "PASS" for value in numerical_status.values())):
            raise RuntimeError("GPU correctness validation is not fully PASS")
        benchmark_validated = True
        run_mode = "validated_benchmark"
    return {"path": str(validation_path.resolve()),
            "sha256": sha256_file(validation_path),
            "schema_version": VALIDATOR_SCHEMA,
            "validation_status": validation["status"],
            "benchmark_validated": benchmark_validated,
            "run_mode": run_mode,
            "structural_gate_statuses": {name: statuses[name] for name in STRUCTURAL_GATES},
            "numerical_gate_statuses": numerical_status,
            # Preserve both complete validator gate entries, including mismatches,
            # logits, token identity, exceptions, and all per-question evidence.
            "numerical_gate_evidence": {name: gates[name] for name in NUMERICAL_GATES}}


def _prior_history(dataset: str, histories: Mapping[str, list[dict[str, str]]],
                   method: str) -> list[dict[str, str]]:
    """GQA questions are independent; MT reuses only this method's answers."""
    if dataset == "gqa":
        return []
    if dataset != "mt_gqa_reconstructed":
        raise ValueError(f"unknown history policy for dataset: {dataset}")
    return [dict(row) for row in histories[method]]


def _score(dataset: str, prediction: str, gold: str) -> float:
    if dataset == "gqa":
        return float(exact_score(prediction, [gold]))
    # Reuse the MT history runner's strict normalized exact definition.
    import re
    def normalize(value: str) -> str:
        clean = re.sub(r"[^\w\s]", " ", str(value).lower())
        return " ".join(w for w in clean.split() if w not in {"a", "an", "the"})
    return float(normalize(prediction) == normalize(gold))


def _result_record(result: dict[str, Any], *, dataset: str,
                   image: Mapping[str, Any], turn: Mapping[str, Any],
                   method: str, order: list[str], history: list[dict[str, str]],
                   conditioning: Any = None) -> dict[str, Any]:
    if "capture" in result:
        raise ValueError("capture must be removed before serializing a request")
    required = ("prediction", "ttft_ms", "request_e2e_ms",
                "generated_token_count", "first_token_id")
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError(f"Qwen runner omitted required request fields: {missing}")
    prediction = str(result["prediction"])
    ttft = float(result["ttft_ms"])
    e2e = float(result["request_e2e_ms"])
    if not (math.isfinite(ttft) and math.isfinite(e2e) and
            0 <= ttft <= e2e + 1e-3):
        raise ValueError(f"invalid TTFT/E2E: {ttft}, {e2e}")
    if int(result["generated_token_count"]) < 1:
        raise ValueError("no generated token")
    turn_id = int(turn["turn_id"])
    request_id = f"{dataset}:{image['image_id']}:{image['dialog_id'] or '-'}:t{turn_id}:{method}"
    record = {"schema_version": SCHEMA_VERSION, "request_id": request_id,
              "dataset": dataset, "image_id": image["image_id"],
              "image_sha256": image["image_sha256"],
              "dialog_id": image["dialog_id"],
              "turn_id": turn_id, "question_id": turn["question_id"],
              "question": turn["question"], "gold": turn["gold"],
              "method": method, "method_order": order,
              "method_order_position": order.index(method),
              "request_path": ("normal_pixels" if turn_id == 1 or method == "recompute"
                               else "ssd_cache_hit"),
              "history": history, "history_sha256": canonical_hash(history),
              "prediction": prediction,
              "correct": _score(dataset, prediction, str(turn["gold"])),
              "ttft_ms": ttft, "request_e2e_ms": e2e,
              "generated_token_count": int(result["generated_token_count"]),
              "first_token_id": int(result["first_token_id"]),
              "truncated": bool(result.get("truncated", False)),
              "conditioning": conditioning,
              "result": result}
    # Fail immediately if backend diagnostics include a non-serializable value.
    json.dumps(record, ensure_ascii=False, allow_nan=False)
    return record


def run_pilot(manifest: dict[str, Any], run_dir: Path,
              validation: dict[str, Any], *, warmup: bool = True) -> None:
    import torch
    from mmimpress.qwen25.runner import Qwen25Runner

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    runner = Qwen25Runner().load()
    runtime = runner.runtime_fingerprint()
    _write_json_new(run_dir / "runtime.json", runtime)
    if warmup:
        warmup_image = Image.new("RGB", (448, 448), (127, 127, 127))
        warmed = runner.run_pixels(warmup_image, "Describe the image briefly.",
                                   history=(), capture=False)
        if not isinstance(warmed, dict) or "prediction" not in warmed:
            raise RuntimeError("unmeasured common warmup failed")
        _write_json_new(run_dir / "warmup.json", {
            key: value for key, value in warmed.items() if key != "capture"})
    status = {"schema_version": SCHEMA_VERSION, "status": "RUNNING",
              "execution_status": "RUNNING",
              "validation_status": validation["validation_status"],
              "benchmark_validated": validation["benchmark_validated"],
              "run_mode": validation["run_mode"],
              "validation": validation, "started_unix": time.time()}
    _write_json_new(run_dir / "status.json", status)
    raw_count = 0
    persistence_count = 0
    try:
        with (run_dir / "raw.jsonl").open("x", encoding="utf-8") as raw_handle, \
                (run_dir / "persistence.jsonl").open("x", encoding="utf-8") as persist_handle, \
                (run_dir / "gpu_inventory.jsonl").open("x", encoding="utf-8") as gpu_handle:
            _record_gpu_inventory(gpu_handle, "pilot_start")
            for image in manifest["images"]:
                _record_gpu_inventory(gpu_handle, "image_start", str(image["image_id"]))
                # Image file read and RGB decode are explicitly outside TTFT.
                path = Path(image["image_path"])
                if sha256_file(path) != image["image_sha256"]:
                    raise RuntimeError(f"image bytes changed: {image['image_id']}")
                with Image.open(path) as source:
                    pixels = source.convert("RGB")
                order = image["method_order"]
                histories: dict[str, list[dict[str, str]]] = {key: [] for key in METHODS}
                stores = {key: run_dir / "stores" / str(image["image_id"]) / key
                          for key in ("fullload", "ours25")}
                for turn in image["turns"]:
                    turn_id = int(turn["turn_id"])
                    for method in order:
                        prior = _prior_history(manifest["dataset"], histories, method)
                        history_pairs = tuple((row["question"], row["prediction"])
                                              for row in prior)
                        conditioning = None
                        if turn_id == 1 or method == "recompute":
                            result = runner.run_pixels(
                                pixels, str(turn["question"]), history=history_pairs,
                                capture=("kv_only" if turn_id == 1 and method == "fullload"
                                         else "with_score" if turn_id == 1 and method == "ours25"
                                         else False),
                                image_sha256=image["image_sha256"])
                            capture = result.pop("capture", None)
                            if turn_id == 1 and method != "recompute":
                                if capture is None:
                                    raise RuntimeError(f"{method} Turn 1 had no capture")
                                store = stores[method]
                                store.parent.mkdir(parents=True, exist_ok=True)
                                if store.exists():
                                    raise FileExistsError(store)
                                persisted = runner.persist(
                                    capture, store,
                                    layout=("canonical" if method == "fullload"
                                            else "repacked"))
                                if not isinstance(persisted, dict):
                                    raise TypeError("persist must return a JSON object")
                                _append_jsonl(persist_handle, {
                                    "schema_version": SCHEMA_VERSION,
                                    "image_id": image["image_id"],
                                    "image_sha256": image["image_sha256"],
                                    "method": method, "store_dir": str(store),
                                    "source_turn_id": 1, "persistence": persisted})
                                persistence_count += 1
                                del capture
                        else:
                            store = stores[method]
                            budget = 1.0 if method == "fullload" else 0.25
                            conditioning = runner.condition_cache(
                                store, budget, image_sha256=image["image_sha256"])
                            result = runner.run_cache(
                                store, str(turn["question"]),
                                history=history_pairs, budget_ratio=budget,
                                image_sha256=image["image_sha256"])
                        if not isinstance(result, dict):
                            raise TypeError("Qwen runner must return a JSON object")
                        record = _result_record(
                            result, dataset=manifest["dataset"], image=image,
                            turn=turn, method=method, order=order,
                            history=prior, conditioning=conditioning)
                        _append_jsonl(raw_handle, record)
                        raw_count += 1
                        if manifest["dataset"] == "mt_gqa_reconstructed":
                            histories[method].append({
                                "question_id": str(turn["question_id"]),
                                "question": str(turn["question"]),
                                "prediction": record["prediction"]})
                    _sync_jsonl(raw_handle)
                    _sync_jsonl(persist_handle)
                runner.close()  # Release this image's store FDs and parsed metadata.
                del pixels
                gc.collect()
                _record_gpu_inventory(gpu_handle, "image_end", str(image["image_id"]))
            _record_gpu_inventory(gpu_handle, "pilot_end")
        expected = sum(len(image["turns"]) for image in manifest["images"]) * len(METHODS)
        if raw_count != expected or persistence_count != 2 * len(manifest["images"]):
            raise AssertionError("pilot request/store coverage mismatch")
        status.update({
            "status": ("PASS" if validation["benchmark_validated"]
                       else "DIAGNOSTIC COMPLETE"),
            "execution_status": "PASS", "raw_requests": raw_count,
            "persistence_stores": persistence_count,
            "finished_unix": time.time()})
    except BaseException as exc:
        status.update({"status": "FAIL", "execution_status": "FAIL",
                       "raw_requests": raw_count,
                       "persistence_stores": persistence_count,
                       "error": f"{type(exc).__name__}: {exc}",
                       "finished_unix": time.time()})
        _write_json_new(run_dir / "failure.json", status)
        raise
    finally:
        runner.close()
        # Keep status.json immutable: final state is a separate durable file.
        _write_json_new(run_dir / "final_status.json", status)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("gqa", "mt"), required=True)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--max-images", type=int, default=40)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--validation", type=Path,
                        help="GPU correctness validation.json; required for execution")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--diagnostic-after-numerical-fail", action="store_true",
                        help="allow only a structurally PASS, numerical-only FAIL validator; "
                             "labels all outputs diagnostic and benchmark_validated=false")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()
    if not args.manifest_only and args.validation is None:
        parser.error("--validation is required unless --manifest-only is set")
    if args.manifest_only and args.diagnostic_after_numerical_fail:
        parser.error("diagnostic mode requires an actual gated run, not --manifest-only")
    index = args.index or (GQA_INDEX if args.dataset == "gqa" else MT_INDEX)
    manifest = build_manifest(args.dataset, index, args.max_images)
    validation = (None if args.manifest_only else _gate(
        args.validation,
        diagnostic_after_numerical_fail=args.diagnostic_after_numerical_fail))
    if validation is not None:
        manifest.pop("manifest_sha256")
        manifest.update({
            "validation": validation,
            "validation_status": validation["validation_status"],
            "benchmark_validated": validation["benchmark_validated"],
            "run_mode": validation["run_mode"],
        })
        manifest["manifest_sha256"] = canonical_hash(manifest)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json_new(run_dir / "manifest.json", manifest)
    _write_json_new(run_dir / "config.json", {
        "schema_version": SCHEMA_VERSION, "manifest_sha256": manifest["manifest_sha256"],
        "manifest_only": args.manifest_only,
        "validation": validation, "warmup": not args.no_warmup,
        "diagnostic_after_numerical_fail": args.diagnostic_after_numerical_fail,
        "validation_status": (validation["validation_status"] if validation else "NOT RUN"),
        "benchmark_validated": (validation["benchmark_validated"] if validation else False),
        "run_mode": (validation["run_mode"] if validation else "manifest_only"),
        "model_id": MODEL_ID, "seed": SEED,
        "started_unix": time.time()})
    if not args.manifest_only:
        run_pilot(manifest, run_dir, validation, warmup=not args.no_warmup)
    print(json.dumps({"run_dir": str(run_dir),
                      "manifest_sha256": manifest["manifest_sha256"],
                      "images": len(manifest["images"]),
                      "expected_requests": sum(len(i["turns"]) for i in manifest["images"]) * 3,
                      "status": ("MANIFEST ONLY" if args.manifest_only else
                                 "PASS" if validation["benchmark_validated"] else
                                 "DIAGNOSTIC COMPLETE"),
                      "validation_status": (validation["validation_status"] if validation
                                            else "NOT RUN"),
                      "benchmark_validated": (validation["benchmark_validated"] if validation
                                              else False)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

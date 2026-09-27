#!/usr/bin/env python3
"""Independent report for the frozen MT-GQA Generated-History five-arm run.

The runner supplies immutable per-image records and a flat raw export. This
script recomputes quality and aggregate measurements from those records,
checks method-local causal histories, and publishes only under a new result
root after the prior-artifact protection check has passed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import statistics
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "mt-gqa-generated-five-arm-report-v1"
INDEX = ROOT / "data/mt_gqa/dialogues.json"
INDEX_SHA256 = "2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224"
WORKLOAD_SHA256 = "0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62"
METHOD_KEYS = ("recompute", "fullload", "mpic32", "rekv_chunk25", "ours25")
LABELS = {
    "recompute": "ReComp",
    "fullload": "FullLoad",
    "mpic32": "MPIC-32 (adapted)",
    "rekv_chunk25": "ReKV-Chunk25 (adapted)",
    "ours25": "Ours25",
}
STORE_KIND = {
    "fullload": "raster", "mpic32": "mpic",
    "rekv_chunk25": "rekv", "ours25": "image_only",
}
EXPECTED_IMAGES = 398
EXPECTED_DIALOGUES = 4061
EXPECTED_ROWS_PER_METHOD = EXPECTED_DIALOGUES * 3
EXPECTED_HITS_PER_METHOD = EXPECTED_DIALOGUES * 2
EXPECTED_ROWS = EXPECTED_ROWS_PER_METHOD * len(METHOD_KEYS)
SEED = 1234
BOOTSTRAP_SEED = 1234
BOOTSTRAP_REPLICATES = 10_000
SHORT_ANSWER_INSTRUCTION = "Answer the current question with a single word or short phrase."


class ReportError(RuntimeError):
    """A required measurement, integrity, or provenance gate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportError(message)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(),
            f"required JSON is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"required JSON has no object root: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(),
            f"required raw JSONL is not a regular file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            require(line.endswith("\n"), f"truncated raw JSONL line {number}")
            value = json.loads(line)
            require(isinstance(value, dict), f"non-object raw row {number}")
            rows.append(value)
    return rows


def _regular_number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{label} is not numeric")
    result = float(value)
    require(math.isfinite(result) and result >= minimum,
            f"{label} is nonfinite or below {minimum}")
    return result


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    require(isinstance(value, int) and not isinstance(value, bool)
            and value >= minimum, f"{label} is not an integer >= {minimum}")
    return value


def normalize_answer(value: Any) -> str:
    """Exactly the frozen strict MT-GQA normalization in scripts/55."""
    lowered = re.sub(r"[^\w\s]", " ", str(value).lower())
    return " ".join(word for word in lowered.split()
                    if word not in {"a", "an", "the"})


def strict_score(prediction: Any, gold: Any) -> int:
    if isinstance(gold, (list, tuple)):
        require(len(gold) == 1, "MT-GQA gold must contain one answer")
        gold = gold[0]
    return int(normalize_answer(prediction) == normalize_answer(gold))


def _stat(values: Sequence[float]) -> dict[str, float]:
    require(bool(values), "cannot summarize an empty population")
    ordered = sorted(float(value) for value in values)

    def quantile(p: float) -> float:
        position = (len(ordered) - 1) * p
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {"mean": statistics.fmean(ordered), "p50": quantile(.5),
            "p95": quantile(.95)}


def _dialogues() -> tuple[dict[str, dict[str, Any]], list[str]]:
    require(sha256_file(INDEX) == INDEX_SHA256, "canonical MT-GQA index SHA256 changed")
    index = read_json(INDEX)
    require(index.get("benchmark_type") == "MT-GQA-reconstructed",
            "canonical benchmark type changed")
    source = index.get("dialogues")
    require(isinstance(source, list) and len(source) == EXPECTED_DIALOGUES,
            "canonical dialogue count changed")
    dialogs: dict[str, dict[str, Any]] = {}
    images: set[str] = set()
    for ordinal, value in enumerate(source):
        require(isinstance(value, dict), f"invalid dialogue at ordinal {ordinal}")
        did = str(value.get("dialog_id", ""))
        turns = value.get("turns")
        require(did == f"mtgqa_{ordinal + 1:06d}" and did not in dialogs,
                f"unexpected canonical dialogue ID: {did}")
        require(isinstance(turns, list) and len(turns) == 3,
                f"canonical dialogue has wrong turn count: {did}")
        require(all(int(turn.get("turn_id", -1)) == i for i, turn in
                    enumerate(turns, 1)), f"canonical turn order changed: {did}")
        dialogs[did] = value
        images.add(str(value["image_id"]))
    require(len(images) == EXPECTED_IMAGES, "canonical image count changed")
    return dialogs, sorted(images)


def _render_prompt(dialog: Mapping[str, Any], turn: int,
                   previous_answers: Sequence[str]) -> tuple[str, str]:
    require(len(previous_answers) == turn - 1, "incorrect generated-history length")
    lines: list[str] = []
    for previous_turn, answer in enumerate(previous_answers, 1):
        question = str(dialog["turns"][previous_turn - 1]["question"])
        lines.extend((f"Q{previous_turn}: {question}",
                      f"A{previous_turn}: {answer}"))
    history = "\n".join(lines)
    body = [] if not history else [history, ""]
    body.extend((f"Current question Q{turn}: {dialog['turns'][turn-1]['question']}",
                 f"{SHORT_ANSWER_INSTRUCTION} ASSISTANT:"))
    return "USER: <image>\n" + "\n".join(body), history


def _verify_run_contract(run: Path, result: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    require(run.is_dir() and not run.is_symlink(), "run root is not a real directory")
    require(result.is_dir() and not result.is_symlink(), "result root is not a real directory")
    require(run != result and run not in result.parents and result not in run.parents,
            "run and result roots overlap")
    config = read_json(run / "config.json")
    manifest = read_json(run / "manifest.json")
    stage = read_json(run / "full/generated_history/config.json")
    require(stage.get("protocol") == "generated_history"
            and stage.get("history_policy") == "method_local_generated",
            "full stage is not method-local Generated-History")
    require(stage.get("benchmark_type") == "MT-GQA-reconstructed",
            "full stage has wrong benchmark")
    require(stage.get("dialogues_file_sha256") == INDEX_SHA256
            and stage.get("selected_workload_sha256") == WORKLOAD_SHA256,
            "full stage has wrong frozen workload")
    require(int(stage.get("n_dialogs", -1)) == EXPECTED_DIALOGUES
            and int(stage.get("n_images", -1)) == EXPECTED_IMAGES
            and int(stage.get("n_turns", -1)) == EXPECTED_DIALOGUES * 3
            and int(stage.get("n_requests", -1)) == EXPECTED_ROWS,
            "full stage has wrong population count")
    require(tuple(stage.get("method_keys", ())) == METHOD_KEYS,
            "full stage method set/order changed")
    require(int(stage.get("seed", -1)) == SEED
            and int(stage.get("max_new_tokens", -1)) == 16
            and stage.get("decoding") == "greedy",
            "full stage seed or generation settings changed")
    require(stage.get("main_ttft_field") == "end_to_end_ttft_ms"
            and stage.get("persistence_in_main_ttft") is False,
            "full stage timing boundary changed")
    for payload, label in ((config, "root config"), (manifest, "root manifest")):
        if "seed" in payload:
            require(int(payload["seed"]) == SEED, f"{label} seed changed")
        if "method_keys" in payload:
            require(tuple(payload["method_keys"]) == METHOD_KEYS,
                    f"{label} method keys changed")
        if "dialogues_file_sha256" in payload:
            require(payload["dialogues_file_sha256"] == INDEX_SHA256,
                    f"{label} index hash changed")
        if "selected_workload_sha256" in payload:
            require(payload["selected_workload_sha256"] == WORKLOAD_SHA256,
                    f"{label} workload hash changed")
    return config, manifest, stage


def _verify_protection(run: Path) -> dict[str, Any]:
    before = read_json(run / "protected_artifacts_before.json")
    after = read_json(run / "protected_artifacts_validation.json")
    require(after.get("passed") is True, "prior-artifact/source protection did not pass")
    for key in ("missing_paths", "changed_paths", "source_missing_paths",
                "source_added_paths", "source_changed_paths"):
        require(key in after and after[key] == [],
                f"protection has a nonempty or missing {key}")
    require(int(after.get("entry_count_before", -1)) > 0,
            "protection has no prior artifact entries")
    require(after.get("before_manifest_sha256") == before.get("manifest_sha256")
            and after.get("after_manifest_sha256") == before.get("manifest_sha256"),
            "protection before/after manifest hash mismatch")
    require(int(after.get("source_file_count_before", -1)) > 0
            and after.get("source_sha256_before") ==
                after.get("source_sha256_after"),
            "source-code before/after hashes differ")
    if "after_protected_entries_sha256" in after:
        require(after["after_protected_entries_sha256"] is not None,
                "protection omitted after protected-entry hash")
    return after


def _verify_runner_validation(run: Path) -> dict[str, Any]:
    validation = read_json(run / "validation.json")
    require(validation.get("passed") is True, "runner validation did not pass")
    checks = validation.get("checks")
    require(isinstance(checks, dict) and checks and all(value is True for value in checks.values()),
            "runner validation has a failed or missing check")
    for key in ("failed_requests", "duplicate_requests", "duplicate_cells",
                "unresolved_failures"):
        if key in validation:
            require(int(validation[key]) == 0, f"runner validation reports {key}")
    return validation


def _validate_rows(rows: Sequence[dict[str, Any]],
                   dialogs: Mapping[str, dict[str, Any]]) -> tuple[
                       dict[tuple[str, int, str], dict[str, Any]], dict[str, Any]]:
    require(len(rows) == EXPECTED_ROWS,
            f"raw has {len(rows)} requests, expected {EXPECTED_ROWS}")
    matrix: dict[tuple[str, int, str], dict[str, Any]] = {}
    physical_ids: set[str] = set()
    order_counts: dict[str, Counter[int]] = defaultdict(Counter)
    retries = 0
    for ordinal, row in enumerate(rows):
        did = str(row.get("dialog_id", ""))
        method = str(row.get("method_key", ""))
        turn = int(row.get("turn_id", -1))
        require(did in dialogs and method in METHOD_KEYS and turn in (1, 2, 3),
                f"raw row {ordinal} has foreign logical cell")
        key = (did, turn, method)
        require(key not in matrix, f"duplicate logical request {key}")
        matrix[key] = row
        require(row.get("protocol") == "generated_history"
                and row.get("history_policy") == "method_local_generated"
                and row.get("history_source") == "generated",
                f"row {key} has wrong history protocol")
        require(row.get("status") == "ok", f"row {key} failed")
        expected_id = f"generated_history:{did}:t{turn}:{method}"
        require(row.get("logical_request_id") == expected_id,
                f"row {key} has wrong logical ID")
        physical_id = str(row.get("physical_execution_id", ""))
        require(physical_id and physical_id not in physical_ids,
                f"row {key} has duplicate/empty physical execution ID")
        physical_ids.add(physical_id)
        source = dialogs[did]
        require(int(row.get("global_dialog_ordinal", -1)) ==
                int(did[-6:]) - 1,
                f"row {key} global dialogue ordinal changed")
        source_turn = source["turns"][turn - 1]
        require(str(row.get("image_id")) == str(source["image_id"])
                and str(row.get("question_id")) == str(source_turn["question_id"])
                and str(row.get("question")) == str(source_turn["question"])
                and row.get("gold") == source_turn["answers"],
                f"row {key} differs from frozen workload")
        require(row.get("dialogues_file_sha256") == INDEX_SHA256
                and row.get("selected_workload_sha256") == WORKLOAD_SHA256,
                f"row {key} workload hash differs")
        require(str(row.get("method")) == LABELS[method],
                f"row {key} display label differs")
        score = strict_score(row.get("prediction", ""), source_turn["answers"])
        for score_key in ("correct", "strict_correct", "score", "quality_score"):
            if score_key in row:
                require(float(row[score_key]) == float(score),
                        f"row {key} has wrong strict score {score_key}")
        require(int(row.get("max_new_tokens", -1)) == 16,
                f"row {key} has wrong max_new_tokens")
        _integer(row.get("first_token_id"), f"{key}.first_token_id")
        count = _integer(row.get("generated_token_count"),
                         f"{key}.generated_token_count", minimum=1)
        require(count <= 16, f"row {key} exceeds max_new_tokens")
        if "generated_token_ids" in row:
            ids = row["generated_token_ids"]
            require(isinstance(ids, list) and len(ids) == count
                    and int(ids[0]) == int(row["first_token_id"]),
                    f"row {key} generated token trace mismatch")
        ttft = _regular_number(row.get("end_to_end_ttft_ms"),
                               f"{key}.end_to_end_ttft_ms", minimum=1e-9)
        e2e = _regular_number(row.get("request_e2e_ms"),
                              f"{key}.request_e2e_ms", minimum=ttft)
        _regular_number(row.get("ssd_read_bytes"), f"{key}.ssd_read_bytes")
        preads = row.get("ssd_preads", row.get("pread_count"))
        _integer(preads, f"{key}.pread_count")
        if "ssd_preads" in row and "pread_count" in row:
            require(int(row["ssd_preads"]) == int(row["pread_count"]),
                    f"row {key} pread counter mismatch")
        if turn == 1 or method == "recompute":
            require(int(row["ssd_read_bytes"]) == 0 and int(preads) == 0,
                    f"row {key} unexpectedly read SSD KV")
        else:
            require(int(row["ssd_read_bytes"]) > 0 and int(preads) > 0,
                    f"row {key} has no SSD KV read")
        if "cache_hit" in row:
            require(bool(row["cache_hit"]) == (turn > 1 and method != "recompute"),
                    f"row {key} cache-hit status wrong")
        if "vision_forward_count" in row:
            require(int(row["vision_forward_count"]) == (
                1 if turn == 1 or method == "recompute" else 0),
                f"row {key} vision forward count wrong")
        require(row.get("page_cache_conditioning_excluded_from_ttft") is True,
                f"row {key} page-cache conditioning timing changed")
        if turn > 1 and method != "recompute":
            require(row.get("page_cache_conditioning_method")
                    == "posix_fadvise_DONTNEED",
                    f"row {key} missing page-cache condition")
        require(int(row.get("future_leakage", 0)) == 0,
                f"row {key} has future leakage")
        retries += int(row.get("retry_count", 0) or 0)
        order = row.get("method_order")
        require(isinstance(order, list) and tuple(order) ==
                METHOD_KEYS[int(row.get("global_dialog_ordinal", -1)) % 5:]
                + METHOD_KEYS[:int(row.get("global_dialog_ordinal", -1)) % 5],
                f"row {key} method rotation changed")
        position = _integer(row.get("method_order_position"),
                            f"{key}.method_order_position")
        require(position < 5 and order[position] == method,
                f"row {key} method position changed")
        order_counts[method][position] += 1
        if method == "rekv_chunk25" and turn > 1:
            require(int(row.get("stage_a_payload_read_bytes", 0)) > 0
                    and int(row.get("stage_b_payload_read_bytes", -1)) == 0
                    and int(row.get("duplicate_read_bytes", -1)) == 0,
                    f"row {key} violated ReKV Stage-A/B handoff")
    require(len(matrix) == EXPECTED_ROWS and len(physical_ids) == EXPECTED_ROWS,
            "raw matrix/physical execution coverage incomplete")
    for did, dialog in dialogs.items():
        expected_order = METHOD_KEYS[(int(did[-6:]) - 1) % 5:] + \
            METHOD_KEYS[:(int(did[-6:]) - 1) % 5]
        for turn in (1, 2, 3):
            for method in METHOD_KEYS:
                require((did, turn, method) in matrix,
                        f"missing logical request {(did, turn, method)}")
                row = matrix[(did, turn, method)]
                require(tuple(row["method_order"]) == expected_order,
                        f"method order changed within {did}")
        t1 = [matrix[(did, 1, method)] for method in METHOD_KEYS]
        require(len({row["prompt_sha256"] for row in t1}) == 1
                and len({row["prediction"] for row in t1}) == 1
                and len({row["first_token_id"] for row in t1}) == 1,
                f"Turn-1 fairness failed for {did}")
        for method in METHOD_KEYS:
            prior: list[dict[str, Any]] = []
            for turn in (1, 2, 3):
                row = matrix[(did, turn, method)]
                prompt, history = _render_prompt(
                    dialog, turn, [str(value["prediction"]) for value in prior])
                key = (did, turn, method)
                require(row.get("prompt") == prompt
                        and row.get("prompt_sha256")
                        == hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                        and row.get("history_text") == history
                        and row.get("history_answers")
                        == [str(value["prediction"]) for value in prior],
                        f"row {key} does not use exact own generated history")
                expected_logical = [value["logical_request_id"] for value in prior]
                expected_physical = [value["physical_execution_id"] for value in prior]
                require(row.get("history_source_request_ids") == expected_logical
                        and row.get("history_source_physical_execution_ids")
                        == expected_physical,
                        f"row {key} has wrong history request lineage")
                entries = row.get("history_entries")
                require(isinstance(entries, list) and len(entries) == turn - 1,
                        f"row {key} has wrong history entries")
                for previous_turn, entry in enumerate(entries, 1):
                    source_row = prior[previous_turn - 1]
                    question = dialog["turns"][previous_turn - 1]
                    require(entry.get("answer_source") == "generated"
                            and entry.get("source_method_key") == method
                            and entry.get("source_logical_request_id")
                            == source_row["logical_request_id"]
                            and entry.get("source_physical_execution_id")
                            == source_row["physical_execution_id"]
                            and entry.get("answer") == source_row["prediction"]
                            and entry.get("question") == question["question"]
                            and str(entry.get("question_id"))
                            == str(question["question_id"]),
                            f"row {key} history entry {previous_turn} drifted")
                prior.append(row)
    for method in METHOD_KEYS:
        require(sum(order_counts[method].values()) == EXPECTED_ROWS_PER_METHOD,
                f"{method} request coverage wrong")
        require(max(order_counts[method].values()) -
                min(order_counts[method].values()) <= 3,
                f"{method} rotated method position is unbalanced")
    return matrix, {
        "logical_requests": len(rows),
        "physical_executions": len(physical_ids),
        "dialogues": len(dialogs),
        "images": len({str(d["image_id"]) for d in dialogs.values()}),
        "requests_per_method": EXPECTED_ROWS_PER_METHOD,
        "hits_per_method": EXPECTED_HITS_PER_METHOD,
        "recorded_retries": retries,
        "method_position_counts": {key: dict(value)
                                   for key, value in order_counts.items()},
        "method_local_generated_history_validated": True,
        "strict_scores_recomputed": True,
        "rekv_stage_b_and_duplicate_reads_zero": True,
    }


def _validate_image_artifacts(
    run: Path, matrix: Mapping[tuple[str, int, str], Mapping[str, Any]],
    dialogs: Mapping[str, dict[str, Any]], image_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    directory = run / "full/generated_history/images"
    require(directory.is_dir() and not directory.is_symlink(),
            "full per-image artifact directory is missing")
    files = {path.stem: path for path in directory.glob("*.json")}
    require(set(files) == set(image_ids),
            f"per-image artifact coverage differs: {len(files)} vs {len(image_ids)}")
    by_image_dialogs: dict[str, list[str]] = defaultdict(list)
    for did, dialog in dialogs.items():
        by_image_dialogs[str(dialog["image_id"])].append(did)
    evidence: dict[str, dict[str, Any]] = {}
    for image_id in image_ids:
        path = files[image_id]
        artifact = read_json(path)
        body_hash = artifact.get("artifact_content_sha256")
        require(isinstance(body_hash, str) and body_hash ==
                stable_hash({key: value for key, value in artifact.items()
                             if key != "artifact_content_sha256"}),
                f"per-image content hash mismatch: {image_id}")
        require(artifact.get("image_id") == image_id
                and artifact.get("protocol") == "generated_history"
                and artifact.get("history_policy") == "method_local_generated"
                and artifact.get("dialogues_file_sha256") == INDEX_SHA256
                and artifact.get("selected_workload_sha256") == WORKLOAD_SHA256,
                f"per-image identity/protocol mismatch: {image_id}")
        image_dialogs = by_image_dialogs[image_id]
        require(artifact.get("dialog_ids") == image_dialogs,
                f"per-image dialogue list changed: {image_id}")
        image_rows = artifact.get("rows")
        require(isinstance(image_rows, list)
                and len(image_rows) == len(image_dialogs) * 3 * len(METHOD_KEYS)
                and int(artifact.get("logical_request_count", -1)) == len(image_rows)
                and int(artifact.get("failed_request_count", -1)) == 0
                and int(artifact.get("duplicate_request_count", -1)) == 0,
                f"per-image row count/failure mismatch: {image_id}")
        seen: set[tuple[str, int, str]] = set()
        for row in image_rows:
            key = (str(row["dialog_id"]), int(row["turn_id"]),
                   str(row["method_key"]))
            require(key not in seen and key in matrix and
                    dict(matrix[key]) == row,
                    f"flat raw differs from immutable image artifact: {key}")
            seen.add(key)
        stores = artifact.get("persistence_overhead")
        manifests = artifact.get("store_manifests")
        require(isinstance(stores, dict) and set(stores) == set(STORE_KIND.values()),
                f"per-image persistence coverage wrong: {image_id}")
        require(isinstance(manifests, dict)
                and set(manifests) == set(STORE_KIND.values()),
                f"per-image store manifest coverage wrong: {image_id}")
        builds = artifact.get("store_build_counts")
        require(isinstance(builds, dict)
                and all(int(builds.get(kind, -1)) == 1
                        for kind in STORE_KIND.values()),
                f"per-image store build counts wrong: {image_id}")
        first_dialog = image_dialogs[0]
        for method, kind in STORE_KIND.items():
            store = stores[kind]
            require(isinstance(store, dict)
                    and store.get("source_method_key") == method
                    and store.get("source_dialog_id") == first_dialog
                    and int(store.get("source_turn_id", -1)) == 1
                    and store.get("source_execution_id") ==
                    matrix[(first_dialog, 1, method)]["physical_execution_id"]
                    and store.get("capture_from_same_answer_forward") is True,
                    f"per-image persistence source wrong: {image_id}/{method}")
            timing = store.get("timing_ms")
            amounts = store.get("bytes")
            durability = store.get("durability")
            require(isinstance(timing, dict) and isinstance(amounts, dict)
                    and isinstance(durability, dict),
                    f"per-image persistence evidence incomplete: {image_id}/{method}")
            _regular_number(timing.get("persist_ms"),
                            f"{image_id}/{method}.persist_ms", minimum=1e-9)
            _integer(amounts.get("total"), f"{image_id}/{method}.bytes.total",
                     minimum=1)
            parent_synced = (
                durability.get("parent_fsynced_after_rename") is True
                or (method == "mpic32"
                    and durability.get("parent_fsynced") is True))
            require(durability.get("atomic_no_clobber") is True
                    and parent_synced,
                    f"per-image store durability invalid: {image_id}/{method}")
        activation = artifact.get("rekv_metadata_activation")
        require(isinstance(activation, dict),
                f"ReKV metadata activation evidence missing: {image_id}")
        events = activation.get("events")
        require(isinstance(events, list)
                and [event.get("dialog_id") for event in events] == image_dialogs,
                f"ReKV activation must have one ordered event per dialogue: {image_id}")
        event_total = 0.0
        for event in events:
            event_total += _regular_number(
                event.get("activation_total_ms"),
                f"{image_id}/{event.get('dialog_id')}.activation_total_ms")
            _regular_number(event.get("metadata_activation_ms"),
                            f"{image_id}.metadata_activation_ms")
            _regular_number(event.get("initial_context_activation_ms"),
                            f"{image_id}.initial_context_activation_ms")
        require(math.isclose(
            event_total, _regular_number(activation.get("total_activation_ms"),
                                         f"{image_id}.total_activation_ms"),
            rel_tol=1e-6, abs_tol=1e-3),
            f"ReKV activation event total mismatch: {image_id}")
        evidence[image_id] = artifact
    return evidence


def _method_rows(matrix: Mapping[tuple[str, int, str], Mapping[str, Any]],
                 dialogs: Mapping[str, Any], method: str,
                 turns: Sequence[int] = (1, 2, 3)) -> list[Mapping[str, Any]]:
    return [matrix[(did, turn, method)] for did in dialogs for turn in turns]


def _quality_and_latency(
    matrix: Mapping[tuple[str, int, str], Mapping[str, Any]],
    dialogs: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    quality: dict[str, Any] = {
        "schema_version": SCHEMA,
        "population": "MT-GQA-reconstructed Generated-History",
        "metric": "strict_normalized_exact_match",
        "methods": {},
    }
    summary_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    for method in METHOD_KEYS:
        scores: dict[str, Any] = {}
        for turn in (1, 2, 3):
            rows = _method_rows(matrix, dialogs, method, (turn,))
            correct = sum(strict_score(row["prediction"], row["gold"]) for row in rows)
            scores[f"t{turn}"] = {
                "n": len(rows), "correct": correct,
                "accuracy": correct / len(rows),
            }
        all_rows = _method_rows(matrix, dialogs, method)
        total_correct = sum(value["correct"] for value in scores.values())
        scores["all"] = {
            "n": len(all_rows), "correct": total_correct,
            "accuracy": total_correct / len(all_rows),
        }
        quality["methods"][method] = {"label": LABELS[method], **scores}
        for name, turns in (("T2", (2,)), ("T3", (3,)),
                            ("T2_T3_pooled_individual_requests", (2, 3))):
            hit_rows = _method_rows(matrix, dialogs, method, turns)
            ttft = _stat([float(row["end_to_end_ttft_ms"]) for row in hit_rows])
            e2e = _stat([float(row["request_e2e_ms"]) for row in hit_rows])
            ssd_mb = statistics.fmean(float(row["ssd_read_bytes"]) / 1e6
                                      for row in hit_rows)
            preads = statistics.fmean(
                float(row.get("ssd_preads", row.get("pread_count", 0)))
                for row in hit_rows)
            latency_rows.append({
                "method_key": method, "method": LABELS[method],
                "population": name, "n": len(hit_rows),
                "ttft_mean_ms": ttft["mean"], "ttft_p50_ms": ttft["p50"],
                "ttft_p95_ms": ttft["p95"],
                "request_e2e_mean_ms": e2e["mean"],
                "ssd_mb_per_request": ssd_mb,
                "preads_per_request": preads,
                "ttft_definition": "individual server-side request to synchronized first token",
                "page_cache_condition": (
                    "not applicable: ReComp reads no stored KV"
                    if method == "recompute" else
                    "OS-page-cache-cold posix_fadvise(DONTNEED), buffered pread"),
            })
        pooled = latency_rows[-1]
        summary_rows.append({
            "method_key": method, "Method": LABELS[method],
            "Acc T1": scores["t1"]["accuracy"],
            "Acc T2": scores["t2"]["accuracy"],
            "Acc T3": scores["t3"]["accuracy"],
            "Avg Acc": scores["all"]["accuracy"],
            "TTFT mean": pooled["ttft_mean_ms"],
            "TTFT p50": pooled["ttft_p50_ms"],
            "TTFT p95": pooled["ttft_p95_ms"],
            "SSD MB/hit": pooled["ssd_mb_per_request"],
            "Preads/hit": pooled["preads_per_request"],
            "n_dialogues": EXPECTED_DIALOGUES,
            "n_requests": EXPECTED_ROWS_PER_METHOD,
            "n_t2_t3_individual_requests": EXPECTED_HITS_PER_METHOD,
        })
    return quality, summary_rows, latency_rows


def _bootstrap_image_clusters(
    differences: np.ndarray, image_by_dialogue: Sequence[str],
    *, seed: int = BOOTSTRAP_SEED, replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    require(differences.shape == (EXPECTED_DIALOGUES, 3),
            "paired quality differences are not a full dialogue-by-turn matrix")
    images = sorted(set(image_by_dialogue))
    require(len(images) == EXPECTED_IMAGES,
            "bootstrap image cluster count wrong")
    positions = {image: i for i, image in enumerate(images)}
    sums = np.zeros((len(images), 3), dtype=np.float64)
    counts = np.zeros(len(images), dtype=np.int64)
    for ordinal, image in enumerate(image_by_dialogue):
        index = positions[image]
        sums[index] += differences[ordinal]
        counts[index] += 1
    require(int(counts.sum()) == EXPECTED_DIALOGUES and bool(np.all(counts > 0)),
            "bootstrap cluster membership incomplete")
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, 3), dtype=np.float64)
    batch = 128
    for start in range(0, replicates, batch):
        end = min(start + batch, replicates)
        indices = rng.integers(0, len(images),
                               size=(end - start, len(images)), endpoint=False)
        samples[start:end] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)[:, None]
    point = differences.mean(axis=0)
    return {
        "seed": seed, "replicates": replicates,
        "cluster_unit": "image; all dialogues and all three turns retained within sampled image",
        "clusters": len(images), "dialogues": EXPECTED_DIALOGUES,
        "per_turn": {
            f"t{turn}": {
                "difference": float(point[turn - 1]),
                "ci95_low": float(np.percentile(samples[:, turn - 1], 2.5)),
                "ci95_high": float(np.percentile(samples[:, turn - 1], 97.5)),
            } for turn in (1, 2, 3)
        },
        "avg": {
            "difference": float(point.mean()),
            "ci95_low": float(np.percentile(samples.mean(axis=1), 2.5)),
            "ci95_high": float(np.percentile(samples.mean(axis=1), 97.5)),
        },
    }


def _paired_quality(
    matrix: Mapping[tuple[str, int, str], Mapping[str, Any]],
    dialogs: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    image_ids = [str(dialog["image_id"]) for dialog in dialogs.values()]
    comparisons = []
    for candidate in ("rekv_chunk25", "mpic32", "fullload", "recompute"):
        differences = np.empty((EXPECTED_DIALOGUES, 3), dtype=np.float64)
        contingencies: dict[str, Any] = {}
        aggregate = Counter()
        for turn in (1, 2, 3):
            cells = Counter()
            for ordinal, did in enumerate(dialogs):
                reference_score = strict_score(
                    matrix[(did, turn, "ours25")]["prediction"],
                    matrix[(did, turn, "ours25")]["gold"])
                candidate_score = strict_score(
                    matrix[(did, turn, candidate)]["prediction"],
                    matrix[(did, turn, candidate)]["gold"])
                differences[ordinal, turn - 1] = reference_score - candidate_score
                cells[(reference_score, candidate_score)] += 1
                aggregate[(reference_score, candidate_score)] += 1
            contingencies[f"t{turn}"] = {
                "n": EXPECTED_DIALOGUES,
                "both_correct": cells[(1, 1)],
                "both_wrong": cells[(0, 0)],
                "reference_only_correct": cells[(1, 0)],
                "candidate_only_correct": cells[(0, 1)],
                "ours_minus_candidate": float(differences[:, turn - 1].mean()),
            }
        contingencies["all"] = {
            "n": EXPECTED_DIALOGUES * 3,
            "both_correct": aggregate[(1, 1)],
            "both_wrong": aggregate[(0, 0)],
            "reference_only_correct": aggregate[(1, 0)],
            "candidate_only_correct": aggregate[(0, 1)],
            "ours_minus_candidate": float(differences.mean()),
        }
        ci = _bootstrap_image_clusters(differences, image_ids)
        comparisons.append({
            "reference": LABELS["ours25"], "reference_method_key": "ours25",
            "candidate": LABELS[candidate], "candidate_method_key": candidate,
            "difference_direction": "Ours25 minus candidate",
            "contingency": contingencies,
            "image_cluster_bootstrap_95ci": ci,
            "avg_ci_includes_zero": ci["avg"]["ci95_low"] <= 0 <= ci["avg"]["ci95_high"],
        })
    return {
        "schema_version": SCHEMA,
        "quality_metric": "strict_normalized_exact_match",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "image (all dialogues and turns in image retained)",
        "comparisons": comparisons,
        "interpretation_rule": "CI including zero means no statistically clear difference; it does not establish equivalence",
    }


def _optional_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows
              if row.get(key) is not None and isinstance(row[key], (int, float))]
    return statistics.fmean(values) if values else None


def _persistence_and_sessions(
    matrix: Mapping[tuple[str, int, str], Mapping[str, Any]],
    dialogs: Mapping[str, dict[str, Any]],
    artifacts: Mapping[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
           list[dict[str, Any]], dict[str, Any]]:
    persistence: list[dict[str, Any]] = []
    for method in METHOD_KEYS:
        if method == "recompute":
            persistence.append({
                "method_key": method, "method": LABELS[method],
                "n_stores": 0, "store_kind": "none",
                "mean_persist_ms": 0.0, "p50_persist_ms": 0.0,
                "p95_persist_ms": 0.0, "mean_store_mb": 0.0,
                "mean_capture_materialize_ms": 0.0,
                "mean_saliency_ms": 0.0, "mean_representative_build_ms": 0.0,
                "mean_kv_repack_ms": 0.0,
                "mean_store_write_ms": 0.0, "mean_ssd_write_ms": 0.0,
                "mean_file_fsync_ms": 0.0, "mean_fsync_aggregate_ms": 0.0,
                "mean_directory_fsync_ms": 0.0,
                "mean_parent_fsync_ms": 0.0,
                "mean_rekv_activation_ms": 0.0,
                "capture_timing_note": "not applicable",
            })
            continue
        kind = STORE_KIND[method]
        stores = [artifact["persistence_overhead"][kind]
                  for artifact in artifacts.values()]
        require(len(stores) == EXPECTED_IMAGES,
                f"{method} has incomplete persistence images")
        times = [float(store["timing_ms"]["persist_ms"]) for store in stores]
        stats = _stat(times)
        def timing_mean(*names: str) -> float | None:
            values: list[float] = []
            for store in stores:
                timing = store["timing_ms"]
                found = next((timing[name] for name in names
                              if timing.get(name) is not None), None)
                if found is not None:
                    values.append(_regular_number(
                        found, f"{method}.{names[0]}"))
            return statistics.fmean(values) if len(values) == len(stores) else None
        capture: list[float] = []
        saliency: list[float] = []
        for image_id, artifact in artifacts.items():
            did = artifact["persistence_overhead"][kind]["source_dialog_id"]
            row = matrix[(did, 1, method)]
            raw_capture = row.get("rekv_capture_stats")
            if isinstance(raw_capture, dict) and raw_capture.get("materialize_ms") is not None:
                capture.append(float(raw_capture["materialize_ms"]))
            vision = row.get("vision_capture_stats")
            if isinstance(vision, dict) and vision.get("saliency_extra_ms") is not None:
                saliency.append(float(vision["saliency_extra_ms"]))
        activations = [
            event["activation_total_ms"]
            for artifact in artifacts.values()
            for event in artifact["rekv_metadata_activation"]["events"]
        ] if method == "rekv_chunk25" else []
        activation_complete = len(activations) == EXPECTED_DIALOGUES
        persistence.append({
            "method_key": method, "method": LABELS[method],
            "n_stores": len(stores), "store_kind": kind,
            "mean_persist_ms": stats["mean"],
            "p50_persist_ms": stats["p50"], "p95_persist_ms": stats["p95"],
            "mean_store_mb": statistics.fmean(
                float(store["bytes"]["total"]) / 1e6 for store in stores),
            "mean_capture_materialize_ms": (
                statistics.fmean(capture) if len(capture) == len(stores) else None),
            "mean_saliency_ms": (
                statistics.fmean(saliency) if len(saliency) == len(stores) else None),
            "mean_representative_build_ms": timing_mean(
                "representative_build_ms", "rep_build_ms"),
            "mean_kv_repack_ms": timing_mean("kv_repack_ms", "repack_ms"),
            "mean_store_write_ms": timing_mean("store_write_ms"),
            "mean_ssd_write_ms": timing_mean("ssd_write_ms"),
            "mean_file_fsync_ms": timing_mean("file_fsync_ms"),
            "mean_fsync_aggregate_ms": timing_mean("fsync_ms"),
            "mean_directory_fsync_ms": timing_mean("directory_fsync_ms"),
            "mean_parent_fsync_ms": timing_mean("parent_fsync_ms"),
            "mean_rekv_activation_ms": (
                statistics.fmean(float(value) for value in activations)
                if activation_complete else None),
            "capture_timing_note": (
                "same-forward capture occurs within measured Turn-1 request; "
                "post-response persistence is charged separately"),
        })
    session_rows: list[dict[str, Any]] = []
    per_dialogue: list[dict[str, Any]] = []
    activation_coverage = all(
        isinstance(artifact.get("rekv_metadata_activation", {}).get("events"), list)
        for artifact in artifacts.values())
    activation_by_dialogue = {
        event["dialog_id"]: float(event["activation_total_ms"])
        for artifact in artifacts.values()
        for event in artifact["rekv_metadata_activation"]["events"]
    }
    require(len(activation_by_dialogue) == EXPECTED_DIALOGUES,
            "ReKV activation dialogue coverage incomplete")
    for method in METHOD_KEYS:
        by_scenario: dict[str, list[float]] = {
            "physical_stream_amortized": [],
            "standalone_equivalent": [],
        }
        for did, dialog in dialogs.items():
            image = str(dialog["image_id"])
            request_e2e = sum(float(matrix[(did, turn, method)]["request_e2e_ms"])
                              for turn in (1, 2, 3))
            kind = STORE_KIND.get(method)
            if kind is None:
                persistence_ms = 0.0
                source_dialog = None
                activation_ms = 0.0
            else:
                store = artifacts[image]["persistence_overhead"][kind]
                persistence_ms = float(store["timing_ms"]["persist_ms"])
                source_dialog = str(store["source_dialog_id"])
                activation_ms = (activation_by_dialogue[did]
                                 if method == "rekv_chunk25" else 0.0)
            for scenario in ("physical_stream_amortized", "standalone_equivalent"):
                charged = (persistence_ms if scenario == "standalone_equivalent"
                           or did == source_dialog else 0.0)
                # Context activation is measured once for this dialogue in
                # both scenarios; persistence alone is amortized across images.
                activation_charged = activation_ms
                total = request_e2e + charged + activation_charged
                by_scenario[scenario].append(total)
                per_dialogue.append({
                    "dialog_id": did, "image_id": image,
                    "method_key": method, "method": LABELS[method],
                    "scenario": scenario,
                    "t1_request_e2e_ms": matrix[(did, 1, method)]["request_e2e_ms"],
                    "t2_request_e2e_ms": matrix[(did, 2, method)]["request_e2e_ms"],
                    "t3_request_e2e_ms": matrix[(did, 3, method)]["request_e2e_ms"],
                    "charged_persistence_ms": charged,
                    "charged_rekv_activation_ms": activation_charged,
                    "session_total_ms": total,
                })
        for scenario, values in by_scenario.items():
            stats = _stat(values)
            session_rows.append({
                "method_key": method, "method": LABELS[method],
                "scenario": scenario, "n_dialogues": len(values),
                "session_mean_ms": stats["mean"],
                "session_p50_ms": stats["p50"],
                "session_p95_ms": stats["p95"],
                "formula": "T1+T2+T3 request_e2e + charged one-time persistence"
                           " + charged ReKV metadata activation (if measured)",
                "rekv_activation_included": method != "rekv_chunk25"
                                            or activation_coverage,
                "attribution": (
                    "one store build per image, charged only to its source dialogue; "
                    "ReKV activation charged to each measured dialogue"
                    if scenario == "physical_stream_amortized" else
                    "measured per-image build cost charged to each standalone dialogue; "
                    "ReKV activation charged to each measured dialogue"),
            })
    return persistence, session_rows, per_dialogue, {
        "rekv_activation_coverage": activation_coverage,
        "physical_stream_store_builds": EXPECTED_IMAGES,
        "standalone_equivalent_is_derived": True,
    }


def _selection_set(row: Mapping[str, Any], *, method: str) -> set[tuple[int, int]]:
    layers = row.get("selected_chunk_ids_per_layer")
    require(isinstance(layers, list) and len(layers) == 32,
            f"{method} hit lacks 32 layer selections")
    selected: set[tuple[int, int]] = set()
    for layer_index, ids in enumerate(layers):
        require(isinstance(ids, list) and ids and
                ids == sorted(set(int(value) for value in ids)),
                f"{method} layer {layer_index} selected IDs are not unique sorted")
        if method == "ours25":
            require(ids == list(range(len(ids))),
                    "Ours25 selection is not a physical first-k prefix")
        for value in ids:
            selected.add((layer_index, int(value)))
    return selected


def _jaccard(a: set[Any], b: set[Any]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _selection_analysis(
    matrix: Mapping[tuple[str, int, str], Mapping[str, Any]],
    dialogs: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {"schema_version": SCHEMA}
    for method in ("rekv_chunk25", "ours25"):
        pairs: list[float] = []
        across_image: dict[str, list[set[tuple[int, int]]]] = defaultdict(list)
        hits: list[Mapping[str, Any]] = []
        sets: dict[tuple[str, int], set[tuple[int, int]]] = {}
        for did, dialog in dialogs.items():
            for turn in (2, 3):
                row = matrix[(did, turn, method)]
                value = _selection_set(row, method=method)
                sets[(did, turn)] = value
                across_image[str(dialog["image_id"])].append(value)
                hits.append(row)
            pairs.append(_jaccard(sets[(did, 2)], sets[(did, 3)]))
        image_pairs: list[float] = []
        for image_sets in across_image.values():
            for first in range(len(image_sets)):
                for second in range(first + 1, len(image_sets)):
                    image_pairs.append(_jaccard(
                        image_sets[first], image_sets[second]))
        require(len(pairs) == EXPECTED_DIALOGUES
                and len(hits) == EXPECTED_HITS_PER_METHOD,
                f"{method} selection coverage incomplete")
        if method == "ours25":
            require(all(value == 1.0 for value in pairs)
                    and all(value == 1.0 for value in image_pairs),
                    "Ours25 selection changed across turn/question for one image")
        runs = [float(value)
                for row in hits
                for value in row.get("contiguous_runs_per_layer", [])]
        lengths = [float(value)
                   for row in hits
                   for value in row.get("actual_attention_key_lengths", [])]
        result: dict[str, Any] = {
            "method": LABELS[method],
            "n_hits": len(hits),
            "t2_t3_dialogue_pairs": len(pairs),
            "t2_t3_jaccard_mean": statistics.fmean(pairs),
            "t2_t3_jaccard_p50": _stat(pairs)["p50"],
            "t2_t3_identical_count": sum(value == 1.0 for value in pairs),
            "t2_t3_changed_count": sum(value < 1.0 for value in pairs),
            "same_image_question_pair_count": len(image_pairs),
            "same_image_question_pair_jaccard_mean": (
                statistics.fmean(image_pairs) if image_pairs else None),
            "selected_layout_runs_per_layer_mean": (
                statistics.fmean(runs) if runs else None),
            "actual_attention_key_length_mean": (
                statistics.fmean(lengths) if lengths else None),
        }
        if method == "rekv_chunk25":
            missing = [key for key in ("q_rep_ms", "similarity_ms", "topk_ms")
                       if any(row.get(key) is None for row in hits)]
            require(not missing, f"ReKV online timing fields missing: {missing}")
            result.update({
                "online_decision_instrumented_component_sum_ms_mean":
                    statistics.fmean(
                        float(row["q_rep_ms"]) + float(row["similarity_ms"])
                        + float(row["topk_ms"]) for row in hits),
                "online_decision_timing_note": (
                    "sum of instrumented host intervals; queued GPU work can "
                    "overlap, so this is not an isolated GPU kernel time"),
                "retrieval_forward_wall_ms_mean":
                    _optional_mean(hits, "retrieval_forward_wall_ms"),
                "answer_prefill_wall_ms_mean":
                    _optional_mean(hits, "answer_prefill_wall_ms"),
                "stage_a_ssd_mb_per_hit": statistics.fmean(
                    float(row["stage_a_payload_read_bytes"]) / 1e6
                    for row in hits),
                "stage_b_duplicate_ssd_bytes_total": sum(
                    int(row["stage_b_payload_read_bytes"]) for row in hits),
                "duplicate_selected_range_bytes_total": sum(
                    int(row["duplicate_read_bytes"]) for row in hits),
                "metadata_bytes_per_image_mean": statistics.fmean(
                    float(next(row["metadata_bytes_image"] for row in hits
                               if row["image_id"] == image))
                    for image in across_image),
            })
        else:
            result.update({
                "online_selector_ms_mean": _optional_mean(hits, "selector_ms"),
                "first_k_planning_ms_mean": _optional_mean(
                    hits, "first_k_planning_ms"),
                "fixed_physical_prefix_all_hits": True,
            })
        output[method] = result
    mpic_hits = _method_rows(matrix, dialogs, "mpic32", (2, 3))
    require(all(int(row.get("n_recomputed_image_tokens", -1)) == 32
                for row in mpic_hits), "MPIC-32 did not recompute 32 image tokens")
    output["mpic32"] = {
        "method": LABELS["mpic32"],
        "n_hits": len(mpic_hits),
        "recomputed_image_tokens_per_hit": 32,
        "reused_image_tokens_mean": _optional_mean(
            mpic_hits, "n_reused_image_tokens"),
        "actual_kv_read_ratio_mean": _optional_mean(
            mpic_hits, "actual_kv_read_ratio"),
    }
    return output


def _fmt(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _analysis_text(
    summary: Sequence[Mapping[str, Any]],
    paired: Mapping[str, Any],
    persistence: Sequence[Mapping[str, Any]],
    sessions: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> str:
    by_method = {row["method_key"]: row for row in summary}
    persist_by = {row["method_key"]: row for row in persistence}
    session_by = {(row["method_key"], row["scenario"]): row
                  for row in sessions}
    comparisons = {row["candidate_method_key"]: row
                   for row in paired["comparisons"]}
    ours = by_method["ours25"]
    rekv = by_method["rekv_chunk25"]
    full = by_method["fullload"]
    recomp = by_method["recompute"]
    mpic = by_method["mpic32"]
    latency_reduction = 100 * (
        rekv["TTFT mean"] - ours["TTFT mean"]) / rekv["TTFT mean"]
    full_gain = 100 * (
        recomp["TTFT mean"] - full["TTFT mean"]) / recomp["TTFT mean"]
    mpic_io_reduction = 100 * (
        full["SSD MB/hit"] - mpic["SSD MB/hit"]) / full["SSD MB/hit"]
    quality_gap = comparisons["rekv_chunk25"][
        "image_cluster_bootstrap_95ci"]["avg"]
    ours_session = session_by[("ours25", "standalone_equivalent")]["session_mean_ms"]
    recomp_session = session_by[("recompute", "standalone_equivalent")]["session_mean_ms"]
    ours_stream = session_by[("ours25", "physical_stream_amortized")]["session_mean_ms"]
    recomp_stream = session_by[("recompute", "physical_stream_amortized")]["session_mean_ms"]
    changed = selection["rekv_chunk25"]["t2_t3_changed_count"]
    total = selection["rekv_chunk25"]["t2_t3_dialogue_pairs"]
    attempts = validation.get("execution_attempts", {})
    lines = [
        "# MT-GQA Generated-History: five-method same-run main experiment",
        "",
        "## Main results",
        "",
        "| Method | Acc T1 | Acc T2 | Acc T3 | Avg Acc | TTFT mean | TTFT p50 | TTFT p95 | SSD MB/hit | Preads/hit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['Method']} | {row['Acc T1']*100:.2f}% | "
            f"{row['Acc T2']*100:.2f}% | {row['Acc T3']*100:.2f}% | "
            f"{row['Avg Acc']*100:.2f}% | {row['TTFT mean']:.2f} | "
            f"{row['TTFT p50']:.2f} | {row['TTFT p95']:.2f} | "
            f"{row['SSD MB/hit']:.2f} | {row['Preads/hit']:.2f} |")
    lines.extend([
        "",
        "Accuracy is strict normalized exact match on the reconstructed MT-GQA "
        "workload. TTFT values are milliseconds for the pooled **8,122 "
        "individual T2/T3 requests per method**; they are not cumulative "
        "three-turn session latencies. SSD MB uses decimal 10^6 bytes.",
        "",
        "## Dataset, seed, and measurement boundary",
        "",
        f"- Frozen index SHA-256: {INDEX_SHA256}.",
        f"- Frozen dialogue workload SHA-256: {WORKLOAD_SHA256}.",
        f"- {EXPECTED_IMAGES} images, {EXPECTED_DIALOGUES:,} three-turn dialogues, "
        f"{EXPECTED_ROWS:,} logical and physical requests across five methods.",
        f"- Inference seed {SEED}; 16-token greedy generation; method-local "
        "generated answers feed T2 and T3.",
        "- TTFT starts before prompt construction and ends after first-token "
        "materialization with CUDA synchronization. ReComp includes image "
        "processing, vision, and multimodal prefill.",
        "- Cache hits use OS-page-cache-cold conditioning with "
        "posix_fadvise(DONTNEED) and buffered pread. Page-cache "
        "conditioning is excluded from TTFT. SSD controller cache flush "
        "is not established.",
        "- ReKV metadata activation is outside cache-hit TTFT. ReKV and "
        "MPIC are SSD workload adaptations, not their original papers' "
        "complete serving systems.",
        "",
        "## Ours cache-hit TTFT difference against each baseline",
        "",
        "Positive values mean Ours25 has a shorter mean TTFT.",
        "",
        "| Baseline | Baseline − Ours ms | Reduction vs baseline |",
        "|---|---:|---:|",
    ])
    for key in ("recompute", "fullload", "mpic32", "rekv_chunk25"):
        baseline = by_method[key]["TTFT mean"]
        difference = baseline - ours["TTFT mean"]
        lines.append(
            f"| {LABELS[key]} | {difference:+.2f} | "
            f"{difference/baseline*100:+.2f}% |")
    lines.extend([
        "",
        "## Paired quality",
        "",
        "Difference direction is Ours25 minus the named baseline. The bootstrap "
        f"uses {BOOTSTRAP_REPLICATES:,} image-cluster resamples with analysis seed "
        f"{BOOTSTRAP_SEED}, keeping all dialogues and all three turns together "
        "within a sampled image. Paired contingency cells are in "
        "paired_quality.json.",
        "",
        "| Baseline | T1 Δpp | T2 Δpp | T3 Δpp | Avg Δpp | Avg 95% CI Δpp |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for key in ("rekv_chunk25", "mpic32", "fullload", "recompute"):
        comparison = comparisons[key]
        boot = comparison["image_cluster_bootstrap_95ci"]
        lines.append(
            f"| {LABELS[key]} | "
            + " | ".join(f"{boot['per_turn'][f't{turn}']['difference']*100:+.2f}"
                         for turn in (1, 2, 3))
            + f" | {boot['avg']['difference']*100:+.2f} | "
            + f"[{boot['avg']['ci95_low']*100:+.2f}, "
            + f"{boot['avg']['ci95_high']*100:+.2f}] |")
    lines.extend([
        "",
        "A confidence interval that includes zero supports only 'no statistically "
        "clear difference' under this analysis. It does not establish "
        "equivalence or the same quality.",
        "",
        "## ReKV and Ours selection and I/O",
        "",
        f"- ReKV T2↔T3 selected-chunk Jaccard: "
        f"{selection['rekv_chunk25']['t2_t3_jaccard_mean']:.4f} mean; "
        f"{changed:,}/{total:,} dialogue pairs changed. Same-image "
        f"question-pair mean: "
        f"{_fmt(selection['rekv_chunk25']['same_image_question_pair_jaccard_mean'],4)}.",
        f"- Ours T2↔T3 selected-chunk Jaccard: "
        f"{selection['ours25']['t2_t3_jaccard_mean']:.4f}; "
        "every hit uses the same physical first-k chunk prefix for its image.",
        f"- ReKV selected-layout runs/layer mean: "
        f"{_fmt(selection['rekv_chunk25']['selected_layout_runs_per_layer_mean'])}; "
        f"actual attention key length mean: "
        f"{_fmt(selection['rekv_chunk25']['actual_attention_key_length_mean'])}.",
        f"- ReKV instrumented online decision component sum mean: "
        f"{_fmt(selection['rekv_chunk25']['online_decision_instrumented_component_sum_ms_mean'])} ms; "
        f"retrieval-forward wall mean "
        f"{_fmt(selection['rekv_chunk25']['retrieval_forward_wall_ms_mean'])} ms; "
        f"answer-prefill wall mean "
        f"{_fmt(selection['rekv_chunk25']['answer_prefill_wall_ms_mean'])} ms. "
        "The component sum includes host intervals and is not isolated GPU time.",
        f"- ReKV Stage-A SSD mean: "
        f"{selection['rekv_chunk25']['stage_a_ssd_mb_per_hit']:.2f} MB/hit; "
        f"Stage-B duplicate payload read total: "
        f"{selection['rekv_chunk25']['stage_b_duplicate_ssd_bytes_total']} bytes; "
        f"duplicate selected-range read total: "
        f"{selection['rekv_chunk25']['duplicate_selected_range_bytes_total']} bytes.",
        f"- ReKV metadata mean: "
        f"{selection['rekv_chunk25']['metadata_bytes_per_image_mean']/1e6:.3f} MB/image. "
        f"Ours online selector mean: "
        f"{_fmt(selection['ours25']['online_selector_ms_mean'])} ms.",
        f"- MPIC recomputes "
        f"{selection['mpic32']['recomputed_image_tokens_per_hit']} leading "
        "image tokens per cache hit. Its traffic is measured in the main table.",
        "",
        "## One-time persistence and three-turn session latency",
        "",
        "Persistence follows the source method's normal T1 answer forward and "
        "is excluded from T2/T3 TTFT. T1 capture instrumentation is already "
        "within the measured T1 request. The totals below use T1+T2+T3 "
        "request end-to-end times plus charged persistence; when measured, "
        "ReKV metadata activation is also charged. Phase means in "
        "persistence.csv are descriptive and must not be summed to "
        "reconstruct a critical path.",
        "",
        "| Method | Stores | Persist mean ms/image | Store MB/image | "
        "Physical stream session mean ms | Standalone-equivalent session mean ms |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for method in METHOD_KEYS:
        p = persist_by[method]
        stream = session_by[(method, "physical_stream_amortized")]
        standalone = session_by[(method, "standalone_equivalent")]
        lines.append(
            f"| {LABELS[method]} | {p['n_stores']} | "
            f"{p['mean_persist_ms']:.2f} | {p['mean_store_mb']:.2f} | "
            f"{stream['session_mean_ms']:.2f} | "
            f"{standalone['session_mean_ms']:.2f} |")
    lines.extend([
        "",
        "Physical-stream attribution charges each of the 398 store builds "
        "only to its actual source dialogue; later dialogues with the same "
        "image reuse it. Standalone-equivalent attribution charges the "
        "measured per-image build to every three-turn dialogue as a derived "
        "single-session scenario. These columns answer different deployment "
        "questions. Per-dialogue values are in session_per_dialogue.csv.",
        "",
        "## Six direct questions",
        "",
        "1. **Ours25 vs ReKV quality at 25% Visual-KV budget:** "
        f"Ours minus ReKV observed Avg accuracy is "
        f"{quality_gap['difference']*100:+.2f} percentage points "
        f"(image-cluster 95% CI "
        f"[{quality_gap['ci95_low']*100:+.2f}, "
        f"{quality_gap['ci95_high']*100:+.2f}]). "
        + ("No statistically clear difference is established."
           if quality_gap["ci95_low"] <= 0 <= quality_gap["ci95_high"]
           else "The interval excludes zero."),
        "2. **Ours cache-hit TTFT reduction vs ReKV:** "
        f"{latency_reduction:+.2f}% relative to ReKV, with pooled means "
        f"{ours['TTFT mean']:.2f} vs {rekv['TTFT mean']:.2f} ms.",
        "3. **FullLoad vs ReComp cache-hit TTFT:** "
        f"FullLoad is {'faster' if full_gain > 0 else 'slower'} by "
        f"{abs(full_gain):.2f}% "
        f"({full['TTFT mean']:.2f} vs {recomp['TTFT mean']:.2f} ms).",
        "4. **MPIC-32 SSD traffic reduction vs FullLoad:** "
        f"{mpic_io_reduction:+.2f}% "
        f"({mpic['SSD MB/hit']:.2f} vs {full['SSD MB/hit']:.2f} MB/hit).",
        "5. **Observed quality gain from query-dependent retrieval:** "
        f"ReKV minus Ours Avg accuracy is "
        f"{-quality_gap['difference']*100:+.2f} percentage points. "
        + ("The image-cluster interval includes zero, so this run shows no "
           "statistically clear gain; the method comparison does not isolate "
           "retrieval as a causal factor."
           if quality_gap["ci95_low"] <= 0 <= quality_gap["ci95_high"]
           else "The interval excludes zero; the method comparison still "
           "does not isolate retrieval as a causal factor."),
        "6. **Ours vs ReComp with one-time persistence in a three-turn session:** "
        f"Standalone-equivalent Ours is "
        f"{'faster' if ours_session < recomp_session else 'slower'} by "
        f"{abs(ours_session-recomp_session):.2f} ms "
        f"({ours_session:.2f} vs {recomp_session:.2f} ms). "
        f"In the actual shared-image physical stream the corresponding means "
        f"are {ours_stream:.2f} vs {recomp_stream:.2f} ms.",
        "",
        "## Validation and interpretation limits",
        "",
        f"- All {validation['logical_requests']:,} logical requests have unique "
        "physical execution IDs and exact method-local generated-history "
        "lineage. No failed or duplicate final logical request is present.",
        f"- Recorded completed-row retry count: {validation['recorded_retries']}; "
        f"repeated shard invocations: {validation.get('runner_retry_count', 0)}; "
        f"incomplete-image rebuilds: {validation.get('runner_incomplete_image_rebuild_count', 0)}; "
        f"failed/interrupted shard invocations: "
        f"{attempts.get('failed_invocations', 0)}/"
        f"{attempts.get('interrupted_invocations', 0)}. Attempt events are "
        "reported separately from final request coverage.",
        "- Prior run/results/source-store artifacts and source code pass "
        "before/after protection with zero missing or changed paths.",
        "- The dataset is MT-GQA-reconstructed, not the unavailable official "
        "MetaCompress dialogue artifact. The scorer is the repository's "
        "strict normalized exact-match metric.",
        "- ReKV-Chunk25 and MPIC-32 are adapted SSD baselines. ReKV's 25% "
        "visual-chunk budget is this experiment's comparison setting, not a "
        "claim about the original paper's 25% setting.",
        "- OS page-cache conditioning does not prove a cold SSD controller "
        "cache. The reported timings are on this GPU/software stack.",
        "- Same observed accuracy or a confidence interval spanning zero "
        "is not statistical equivalence.",
        "",
        "MT-GQA GENERATED-HISTORY FIVE-ARM MAIN VALIDATED: YES",
        "",
    ])
    return "\n".join(lines)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    require(bool(rows), "cannot write empty CSV")
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return out.getvalue().encode("utf-8")


def _publish_bytes(path: Path, payload: bytes) -> str:
    if path.exists():
        require(path.is_file() and not path.is_symlink()
                and path.read_bytes() == payload,
                f"refusing to overwrite differing result: {path}")
        return sha256_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def _publish_copy(source: Path, destination: Path) -> str:
    source_hash = sha256_file(source)
    if destination.exists():
        require(destination.is_file() and not destination.is_symlink()
                and sha256_file(destination) == source_hash,
                f"refusing to overwrite differing result: {destination}")
        return source_hash
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with source.open("rb") as src, temporary.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=8 << 20)
            dst.flush()
            os.fsync(dst.fileno())
        require(sha256_file(temporary) == source_hash,
                "raw copy differs from source")
        os.link(temporary, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return source_hash


def report(run: Path, result: Path) -> dict[str, Any]:
    run, result = run.resolve(), result.resolve()
    config, manifest, stage = _verify_run_contract(run, result)
    protection = _verify_protection(run)
    runner_validation = _verify_runner_validation(run)
    dialogs, image_ids = _dialogues()
    rows = read_jsonl(run / "raw.jsonl")
    matrix, row_validation = _validate_rows(rows, dialogs)
    require(int(runner_validation.get("completed_logical_row_retry_count", 0)) ==
            row_validation["recorded_retries"],
            "runner and raw completed-row retry counts differ")
    artifacts = _validate_image_artifacts(run, matrix, dialogs, image_ids)
    quality, summary, latency = _quality_and_latency(matrix, dialogs)
    paired = _paired_quality(matrix, dialogs)
    persistence, sessions, per_dialogue, session_note = \
        _persistence_and_sessions(matrix, dialogs, artifacts)
    selection = _selection_analysis(matrix, dialogs)
    report_validation = {
        "schema_version": SCHEMA, "passed": True,
        "checks": {
            "frozen_index_and_workload_sha256": True,
            "generated_history_only": True,
            "exact_398_image_4061_dialogue_60915_request_matrix": True,
            "five_method_t1_t2_t3_coverage": True,
            "strict_quality_recomputed": True,
            "method_local_generated_history_exact": True,
            "raw_matches_398_immutable_image_artifacts": True,
            "four_store_persistence_per_image": True,
            "rekv_stage_b_and_duplicate_payload_read_zero": True,
            "ours_fixed_prefix_across_all_same_image_hits": True,
            "ten_thousand_image_cluster_bootstrap": True,
            "runner_validation_passed": True,
            "prior_artifacts_and_source_code_unchanged": True,
        },
        **row_validation,
        **session_note,
        "execution_attempts": runner_validation.get("execution_attempts", {}),
        "runner_failed_requests": runner_validation.get("failed_requests", 0),
        "runner_retry_count": runner_validation.get("retry_count", 0),
        "runner_completed_logical_row_retry_count": runner_validation.get(
            "completed_logical_row_retry_count", 0),
        "runner_incomplete_image_rebuild_count": runner_validation.get(
            "incomplete_image_rebuild_count", 0),
        "runner_validation_schema": runner_validation.get("schema_version"),
        "protection_schema": protection.get("schema_version"),
        "index_sha256": INDEX_SHA256,
        "workload_sha256": WORKLOAD_SHA256,
        "analysis_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "n_image_artifacts": len(artifacts),
    }
    analysis = _analysis_text(summary, paired, persistence, sessions,
                              selection, report_validation)
    summary_json = {
        "schema_version": SCHEMA,
        "dataset": "MT-GQA-reconstructed",
        "protocol": "generated_history",
        "seed": SEED,
        "index_sha256": INDEX_SHA256,
        "workload_sha256": WORKLOAD_SHA256,
        "methods": summary,
        "cache_hit_population": "T2 and T3 pooled individual requests",
        "cache_condition": (
            "stored-KV methods: OS-page-cache-cold "
            "posix_fadvise(DONTNEED), buffered pread; "
            "ReComp: no stored-KV read"),
    }
    payloads = {
        "summary.json": _json_bytes(summary_json),
        "summary.csv": _csv_bytes(summary),
        "latency.csv": _csv_bytes(latency),
        "quality.json": _json_bytes(quality),
        "paired_quality.json": _json_bytes(paired),
        "persistence.csv": _csv_bytes(persistence),
        "session_latency.csv": _csv_bytes(sessions),
        "session_per_dialogue.csv": _csv_bytes(per_dialogue),
        "selection_analysis.json": _json_bytes(selection),
        "report_validation.json": _json_bytes(report_validation),
        "ANALYSIS.md": analysis.encode("utf-8"),
    }
    published: dict[str, str] = {}
    for name in ("config.json", "validation.json", "protected_artifacts_before.json",
                 "protected_artifacts_validation.json"):
        published[name] = _publish_copy(run / name, result / name)
    published["raw.jsonl"] = _publish_copy(run / "raw.jsonl", result / "raw.jsonl")
    for name, payload in payloads.items():
        published[name] = _publish_bytes(result / name, payload)
    receipt = {
        "schema_version": SCHEMA, "passed": True,
        "run_dir": str(run), "results_dir": str(result),
        "files_sha256": published,
        "source_config_sha256": sha256_file(run / "config.json"),
        "source_manifest_sha256": sha256_file(run / "manifest.json"),
        "source_stage_config_sha256": sha256_file(
            run / "full/generated_history/config.json"),
        "source_protection_validation_sha256": sha256_file(
            run / "protected_artifacts_validation.json"),
    }
    published["report_artifacts.json"] = _publish_bytes(
        result / "report_artifacts.json", _json_bytes(receipt))
    _publish_bytes(result / "COMPLETED", _json_bytes({
        "schema_version": SCHEMA,
        "passed": True, "report_validation_sha256":
            published["report_validation.json"],
        "report_artifacts_sha256": published["report_artifacts.json"],
        "logical_requests": EXPECTED_ROWS,
    }))
    return {"run_dir": str(run), "results_dir": str(result),
            "passed": True, "logical_requests": EXPECTED_ROWS,
            "files": len(published) + 1}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        print(json.dumps(report(args.run_dir, args.results_dir),
                         indent=2, sort_keys=True))
        return 0
    except (ReportError, OSError, ValueError, KeyError, TypeError,
            json.JSONDecodeError) as error:
        print(f"five-arm MT-GQA report failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

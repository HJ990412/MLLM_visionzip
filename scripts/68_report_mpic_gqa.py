#!/usr/bin/env python3
"""Independently validate and publish the MPIC five-arm GQA report bundle.

The GPU runner owns measurements.  This CPU-only second reader validates the
completed 40-image/240-question run directly from ``raw.jsonl``, recomputes
the paper-facing metrics, checks the real-model correctness smoke and the
before/after artifact guard, and publishes the remaining report files.

Publication is fail closed and idempotent: a missing file is installed with
an atomic no-clobber operation, an existing byte-identical file is retained,
and a conflicting file aborts the report.  Runner-owned exports are checked,
never rewritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import stat
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent.parent
RUN_SCHEMA = "mpic-gqa-five-arm-v1"
REPORT_SCHEMA = "mpic-gqa-final-report-v1"
SMOKE_SCHEMA = "mpic-correctness-smoke-v1"
PROTECTION_SCHEMA = "mpic-protected-artifacts-v1"
EXPECTED_INDEX_SHA256 = (
    "514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a"
)
EXPECTED_WORKLOAD_SHA256 = (
    "97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66"
)
EXPECTED_PAPER_SHA256 = (
    "7253687b8a076fbea6e49fc8d9bffc856c3be33b1b7a372cba5fd5d00eaa503b"
)
METHOD_KEYS = ("recompute", "fullload", "qa_chunk25", "ours25", "mpic32")
METHOD_IDS = {
    "recompute": "recompute",
    "fullload": "fullload",
    "qa_chunk25": "qa_chunk25",
    "ours25": "imageonly_prefix25",
    "mpic32": "mpic32_ssd",
}
DISPLAY = {
    "recompute": "ReComp",
    "fullload": "FullLoad",
    "qa_chunk25": "QA-Chunk25",
    "ours25": "Ours25",
    "mpic32": "MPIC-32 (SSD adaptation)",
}
STORE_METHODS = ("fullload", "ours25", "mpic32")
RUNNER_EXPORTS = (
    "config.json", "manifest.json", "raw.jsonl", "summary.json",
    "summary.csv", "persistence.json", "store_metadata.json",
    "validation.json", "runtime_fingerprint.json", "run_artifacts.json",
)
BOOTSTRAP_SEED = 680032
BOOTSTRAP_REPLICATES = 10_000


class ReportValidationError(RuntimeError):
    """Evidence is incomplete, inconsistent, or unsafe to publish."""


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _regular_file(path: Path) -> os.stat_result:
    try:
        value = path.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise ReportValidationError(f"missing required file: {path}") from error
    if not stat.S_ISREG(value.st_mode):
        raise ReportValidationError(f"not a regular file: {path}")
    return value


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    before = _regular_file(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        identity = lambda item: (
            item.st_dev, item.st_ino, item.st_mode, item.st_size,
            item.st_mtime_ns, item.st_ctime_ns,
        )
        if identity(before) != identity(opened):
            raise ReportValidationError(f"file changed while opening: {path}")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, block_size)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if identity(before) != identity(after):
        raise ReportValidationError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    digest = sha256_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReportValidationError(f"cannot parse JSON: {path}") from error
    if not isinstance(value, dict):
        raise ReportValidationError(f"JSON root is not an object: {path}")
    if sha256_file(path) != digest:
        raise ReportValidationError(f"file changed while reading: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    digest = sha256_file(path)
    rows: list[dict[str, Any]] = []
    line_number = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ReportValidationError(
                        f"blank JSONL line {line_number}: {path}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("row is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise ReportValidationError(
            f"cannot parse JSONL {path} at line {line_number}") from error
    if sha256_file(path) != digest:
        raise ReportValidationError(f"file changed while reading: {path}")
    return rows


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportValidationError(f"{label} is not a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ReportValidationError(f"{label} is not a sequence")
    return value


def _finite(value: Any, label: str, *, nonnegative: bool = False,
            allow_bool: bool = False) -> float:
    if isinstance(value, bool) and not allow_bool:
        raise ReportValidationError(f"{label} is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ReportValidationError(f"{label} is not numeric") from error
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ReportValidationError(f"{label} is invalid: {result!r}")
    return result


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [_finite(row[key], key, allow_bool=(key == "correct"))
              for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ReportValidationError("percentile of an empty sequence")
    ordered = sorted(float(item) for item in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportValidationError(message)


def _close(actual: Any, expected: Any, label: str) -> None:
    left, right = _finite(actual, label), _finite(expected, label)
    if not math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10):
        raise ReportValidationError(f"{label} differs: {left} != {right}")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n").encode("utf-8")


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _publish_bytes(path: Path, payload: bytes) -> str:
    """Publish once; retain an identical file and reject every conflict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = hashlib.sha256(payload).hexdigest()
    if os.path.lexists(path):
        if sha256_file(path) != expected:
            raise ReportValidationError(
                f"refusing to replace conflicting artifact: {path}")
        return "identical"
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if sha256_file(path) != expected:
                raise ReportValidationError(
                    f"concurrent conflicting publication: {path}")
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
    return "published"


def _publish_pair(run_dir: Path, results_dir: Path, name: str,
                  payload: bytes) -> dict[str, str]:
    return {
        "run": _publish_bytes(run_dir / name, payload),
        "results": _publish_bytes(results_dir / name, payload),
    }


def _preflight_publications(run_dir: Path, results_dir: Path,
                            payloads: Mapping[str, bytes]) -> None:
    """Reject every known conflict before publishing the first missing file."""
    for name, payload in payloads.items():
        expected = hashlib.sha256(payload).hexdigest()
        for root in (run_dir, results_dir):
            path = root / name
            if os.path.lexists(path) and sha256_file(path) != expected:
                raise ReportValidationError(
                    f"refusing conflicting report bundle: {path}")


def _validate_roots(run_dir: Path, results_dir: Path) -> tuple[Path, Path]:
    run, result = run_dir.resolve(), results_dir.resolve()
    expected_run_parent = (ROOT / "runs/mpic_baseline").resolve()
    expected_result_parent = (ROOT / "results/mpic_baseline").resolve()
    _require(run.parent == expected_run_parent,
             "run directory is outside runs/mpic_baseline")
    _require(result.parent == expected_result_parent,
             "results directory is outside results/mpic_baseline")
    _require(result.name == f"gqa40_240_{run.name}",
             "run/results IDs do not match")
    for path, label in ((run, "run"), (result, "results")):
        _require(path.is_dir() and not path.is_symlink(),
                 f"{label} path is not a real directory")
    return run, result


def _verify_runner_exports(run_dir: Path, results_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in RUNNER_EXPORTS:
        left, right = run_dir / name, results_dir / name
        left_hash, right_hash = sha256_file(left), sha256_file(right)
        _require(left_hash == right_hash,
                 f"runner export differs between run/results: {name}")
        hashes[name] = left_hash
    _require(sha256_file(run_dir / "results_partial.jsonl") == hashes["raw.jsonl"],
             "partial and finalized raw JSONL differ")
    artifacts = _read_json(run_dir / "run_artifacts.json")
    _require(artifacts.get("schema_version") == RUN_SCHEMA,
             "run_artifacts schema mismatch")
    recorded = _mapping(artifacts.get("files_sha256"),
                        "run_artifacts.files_sha256")
    for name, digest in recorded.items():
        _require(sha256_file(run_dir / str(name)) == digest,
                 f"run_artifacts digest mismatch: {name}")
    completed = _read_json(run_dir / "COMPLETED")
    _require(completed.get("schema_version") == RUN_SCHEMA
             and completed.get("measurement_checks_passed") is True
             and completed.get("completion_scope") == "measurement_runner_only"
             and completed.get("artifact_protection_status")
             == "pending_external_verify_after"
             and completed.get("final_validated_claim") is False
             and int(completed.get("completed", -1)) == 1200,
             "COMPLETED marker is invalid")
    _require(completed.get("validation_sha256") == hashes["validation.json"],
             "COMPLETED validation digest mismatch")
    return hashes


def _validate_config(config: Mapping[str, Any], manifest: Mapping[str, Any],
                     run_dir: Path, results_dir: Path) -> None:
    for value, label in ((config, "config"), (manifest, "manifest")):
        _require(value.get("schema_version") == RUN_SCHEMA,
                 f"{label} schema mismatch")
        _require(value.get("run_id") == run_dir.name,
                 f"{label} run ID mismatch")
    _require(config.get("status")
             == "measurements_complete_pending_artifact_protection",
             "runner measurements are not complete or have an unexpected status")
    _require(config.get("dataset") == "gqa", "dataset is not GQA")
    _require(config.get("index_sha256") == EXPECTED_INDEX_SHA256,
             "frozen index digest mismatch")
    _require(config.get("full_workload_sha256") == EXPECTED_WORKLOAD_SHA256
             and config.get("selected_workload_sha256")
             == EXPECTED_WORKLOAD_SHA256,
             "frozen workload digest mismatch")
    _require(int(config.get("selected_images", -1)) == 40
             and int(config.get("selected_questions", -1)) == 240
             and int(config.get("questions_per_image", -1)) == 6
             and config.get("question_slice") == [4, 10],
             "40-image/240-question contract mismatch")
    _require(tuple(config.get("method_keys", ())) == METHOD_KEYS
             and tuple(manifest.get("method_keys", ())) == METHOD_KEYS,
             "five-arm method contract mismatch")
    _require(int(config.get("mpic_k", -1)) == 32,
             "MPIC k is not fixed at 32")
    _require(Path(str(config.get("run_dir"))).resolve() == run_dir
             and Path(str(config.get("results_dir"))).resolve() == results_dir,
             "config output roots mismatch")
    source = _mapping(manifest.get("source_sha256"), "manifest.source_sha256")
    _require(source.get("paper") == EXPECTED_PAPER_SHA256,
             "reviewed MPIC paper digest mismatch")
    current_sources = {
        "paper": ROOT / "papers/mpic.md",
        "mpic_implementation": ROOT / "mmimpress/mpic.py",
        "mpic_contract": ROOT / "docs/mpic_baseline_contract.md",
        "mpic_smoke_validator": ROOT / "scripts/65_validate_mpic.py",
        "artifact_protector": ROOT / "scripts/66_protect_mpic_artifacts.py",
        "pilot_runner": ROOT / "scripts/67_eval_mpic_gqa.py",
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
    _require(set(source) == set(current_sources),
             "runner source-hash inventory mismatch")
    for key, path in current_sources.items():
        _require(source[key] == sha256_file(path),
                 f"source changed after pilot began: {key}")


def _validate_rows(rows: Sequence[Mapping[str, Any]], run_id: str) \
        -> dict[str, Any]:
    _require(len(rows) == 1200, f"expected 1200 rows, found {len(rows)}")
    identities = [str(row.get("request_id")) for row in rows]
    _require(len(identities) == len(set(identities)),
             "raw rows contain duplicate request IDs")
    by_method: dict[str, list[Mapping[str, Any]]] = {
        key: [] for key in METHOD_KEYS}
    by_image: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_prompt: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        prefix = f"row[{index}]"
        method = str(row.get("method_key"))
        _require(row.get("schema_version") == RUN_SCHEMA, f"{prefix} schema")
        _require(row.get("run_id") == run_id, f"{prefix} run ID")
        _require(method in METHOD_KEYS, f"{prefix} unknown method")
        _require(row.get("method_id") == METHOD_IDS[method]
                 and row.get("display_label") == DISPLAY[method],
                 f"{prefix} method metadata mismatch")
        image, question = str(row.get("image_id")), str(row.get("question_id"))
        _require(row.get("request_id") == f"gqa:{image}:{question}:{method}",
                 f"{prefix} noncanonical request ID")
        turn = int(row.get("turn_id", -1))
        _require(1 <= turn <= 6 and int(row.get("request_ordinal", -1)) == turn,
                 f"{prefix} invalid turn")
        order = list(_sequence(row.get("method_order"), f"{prefix}.order"))
        order_position = int(row.get("method_order_position", -1))
        _require(len(order) == 5 and set(order) == set(METHOD_KEYS)
                 and 0 <= order_position < 5
                 and order[order_position] == method,
                 f"{prefix} method rotation mismatch")
        _require(row.get("measurement_source") == "same_run",
                 f"{prefix} is not same-run evidence")
        _require(row.get("status") == "ok", f"{prefix} status is not ok")
        retry = int(row.get("retry_count", -1))
        _require(0 <= retry <= 1, f"{prefix} exceeded retry policy")
        _require(int(row.get("future_questions_in_prompt", -1)) == 0
                 and row.get("future_question_ids_used") == [],
                 f"{prefix} contains future-question leakage")
        _require(str(row.get("prompt_sha256"))
                 == str(row.get("expected_prompt_sha256")),
                 f"{prefix} prompt hash mismatch")
        _require(int(row.get("n_image_tokens", 0)) > 0,
                 f"{prefix} has no image tokens")
        _require(_finite(row.get("ttft_ms"), f"{prefix}.ttft") > 0,
                 f"{prefix} nonpositive TTFT")
        _close(row.get("ttft_ms"), row.get("end_to_end_ttft_ms"),
               f"{prefix}.TTFT aliases")
        _require(_finite(row.get("request_e2e_ms"), f"{prefix}.e2e")
                 + 1e-6 >= _finite(row.get("ttft_ms"), f"{prefix}.ttft"),
                 f"{prefix} E2E precedes TTFT")
        if row.get("generated_token_ids") is not None:
            generated = list(_sequence(row.get("generated_token_ids"),
                                       f"{prefix}.generated_token_ids"))
            _require(bool(generated)
                     and int(row.get("generated_token_count", -1)) == len(generated)
                     and int(row.get("first_token_id", -1)) == int(generated[0]),
                     f"{prefix} generated-token accounting mismatch")
        else:
            _require(int(row.get("generated_tokens", 0)) > 0,
                     f"{prefix} has no generation-length evidence")
        correct = _finite(row.get("correct"), f"{prefix}.correct",
                          allow_bool=True)
        _require(correct in (0.0, 1.0), f"{prefix} invalid correctness")

        n_image = int(row["n_image_tokens"])
        recomputed = int(row.get("n_recomputed_image_tokens", -1))
        reused_raw = row.get("n_reused_image_tokens")
        if turn == 1 or method == "recompute":
            _require(recomputed == n_image and int(reused_raw) == 0,
                     f"{prefix} recomputation counts mismatch")
        elif method == "mpic32":
            k = min(32, n_image)
            _require(recomputed == k and int(reused_raw) == n_image - k,
                     f"{prefix} MPIC k/reuse mismatch")
        elif method == "fullload":
            _require(recomputed == 0 and int(reused_raw) == n_image,
                     f"{prefix} FullLoad reuse count mismatch")
        else:
            _require(recomputed == 0 and reused_raw is None
                     and str(row.get("image_token_count_semantics", "")).startswith(
                         "N/A here"),
                     f"{prefix} selector-arm token-count semantics mismatch")

        if turn == 1:
            _require(row.get("request_path") == "normal_pixel_turn1"
                     and int(row.get("vision_forward_count", -1)) == 1
                     and row.get("cache_hit_measurement") is False,
                     f"{prefix} Turn-1 path mismatch")
        elif method == "recompute":
            _require(row.get("request_path") == "normal_pixel_recompute"
                     and int(row.get("vision_forward_count", -1)) == 1
                     and row.get("cache_hit_measurement") is False,
                     f"{prefix} ReComp path mismatch")
        else:
            expected_path = ("ssd_cache_hit_mpic" if method == "mpic32"
                             else "ssd_cache_hit")
            _require(row.get("request_path") == expected_path
                     and int(row.get("vision_forward_count", -1)) == 0
                     and row.get("cache_hit_measurement") is True,
                     f"{prefix} stored-hit path mismatch")

        by_method[method].append(row)
        by_image[image].append(row)
        by_prompt[(image, question)].append(row)

    _require(len(by_image) == 40, "raw rows do not cover exactly 40 images")
    for method, method_rows in by_method.items():
        _require(len(method_rows) == 240, f"{method} does not have 240 rows")
        _require(sum(int(row["turn_id"]) > 1 for row in method_rows) == 200,
                 f"{method} does not have 200 Q2-Q6 rows")
    for image, image_rows in by_image.items():
        _require(len(image_rows) == 30, f"image {image} does not have 30 rows")
        _require(len({tuple(row["method_order"]) for row in image_rows}) == 1,
                 f"image {image} changed method order across turns")
        for turn in range(1, 7):
            subset = [row for row in image_rows if int(row["turn_id"]) == turn]
            _require(len(subset) == 5
                     and {row["method_key"] for row in subset} == set(METHOD_KEYS)
                     and len({str(row["question_id"]) for row in subset}) == 1,
                     f"image {image} turn {turn} is not a complete five-arm pair")
            _require(len({tuple(row["method_order"]) for row in subset}) == 1,
                     f"image {image} turn {turn} changed method order")
    _require(len(by_prompt) == 240, "expected 240 paired prompts")
    for pair, prompt_rows in by_prompt.items():
        _require(len(prompt_rows) == 5
                 and len({str(row["prompt_sha256"]) for row in prompt_rows}) == 1
                 and len({str(row["question"]) for row in prompt_rows}) == 1
                 and len({json.dumps(row["gold"], sort_keys=True)
                          for row in prompt_rows}) == 1,
                 f"paired prompt mismatch: {pair}")

    turn1 = [row for row in rows if int(row["turn_id"]) == 1]
    for image, image_rows in by_image.items():
        first = [row for row in image_rows if int(row["turn_id"]) == 1]
        _require(len({(str(row["prediction"]), int(row["first_token_id"]))
                      for row in first}) == 1,
                 f"Turn-1 normal-pixel outputs differ for {image}")
        _require(len({(str(row.get("input_tensors_sha256")),
                       str(row.get("image_input_sha256")),
                       str(row.get("suffix_ids_sha256")))
                      for row in first}) == 1,
                 f"Turn-1 normal-pixel inputs differ for {image}")
    _require(len(turn1) == 200, "expected 200 Turn-1 arm requests")

    mpic = [row for row in by_method["mpic32"] if int(row["turn_id"]) > 1]
    for row in mpic:
        identity = str(row["request_id"])
        _require(bool(row.get("generated_token_ids"))
                 and int(row.get("generated_token_count", -1))
                 == len(row["generated_token_ids"]),
                 f"{identity} lacks MPIC generation instrumentation")
        n_image = int(row["n_image_tokens"])
        k = min(32, n_image)
        n_text = int(row.get("n_recomputed_text_tokens", -1))
        _require(int(row.get("k_recompute", -1)) == k and n_text > 0,
                 f"{identity} active-row count mismatch")
        _require(row.get("selected_image_local_rows") == list(range(k)),
                 f"{identity} did not select leading canonical rows")
        target_positions = list(_sequence(
            row.get("target_positions"), f"{identity}.target_positions"))
        _require(row.get("selected_image_logical_rows") == target_positions[:k],
                 f"{identity} logical selected rows mismatch")
        _require(row.get("same_source_target_context") is True
                 and row.get("same_source_target_positions") is True
                 and row.get("source_position_hash")
                 == row.get("target_position_hash")
                 and row.get("position_handling_policy")
                 == "post_rope_cached_k_reused_at_identical_logical_position",
                 f"{identity} main-pilot position policy mismatch")
        _require(float(row.get("retained_image_context_ratio")) == 1.0
                 and math.isclose(float(row.get("recomputed_image_token_ratio")),
                                  k / n_image, rel_tol=1e-12, abs_tol=1e-12)
                 and math.isclose(float(row.get("reused_image_token_ratio")),
                                  (n_image - k) / n_image,
                                  rel_tol=1e-12, abs_tol=1e-12),
                 f"{identity} context ratios mismatch")
        arrays = {
            name: list(_sequence(row.get(name), f"{identity}.{name}"))
            for name in (
                "active_rows_per_layer", "recomputed_image_rows_per_layer",
                "recomputed_text_rows_per_layer",
                "attention_key_length_per_layer",
                "valid_image_key_count_per_layer",
            )
        }
        _require(all(len(values) == 32 for values in arrays.values())
                 and all(int(value) == k for value in
                         arrays["recomputed_image_rows_per_layer"])
                 and all(int(value) == n_text for value in
                         arrays["recomputed_text_rows_per_layer"])
                 and all(int(value) == n_text + k for value in
                         arrays["active_rows_per_layer"])
                 and all(int(value) == n_image for value in
                         arrays["valid_image_key_count_per_layer"])
                 and len(set(map(int, arrays[
                     "attention_key_length_per_layer"]))) == 1,
                 f"{identity} 32-layer instrumentation mismatch")
        _require(int(row.get("decoder_prefill_pass_count", -1)) == 1,
                 f"{identity} did not use one decoder prefill")
        _require(row.get("source_payload_hash_before")
                 == row.get("source_payload_hash_after"),
                 f"{identity} source payload changed")
        kv, embedding = int(row.get("ssd_kv_bytes", -1)), int(
            row.get("ssd_embedding_bytes", -1))
        _require(kv > 0 and embedding > 0
                 and int(row.get("ssd_separator_bytes", -1)) == 0
                 and int(row.get("ssd_metadata_bytes", -1)) == 0
                 and int(row.get("ssd_total_bytes", -1)) == kv + embedding
                 and int(row.get("ssd_read_bytes", -1)) == kv + embedding
                 and int(row.get("pread_count", -1)) == 65
                 and int(row.get("ssd_preads", -1)) == 65,
                 f"{identity} MPIC SSD accounting mismatch")
        io_detail = _mapping(row.get("io"), f"{identity}.io")
        kinds = _mapping(io_detail.get("per_kind"), f"{identity}.io.per_kind")
        _require(int(_mapping(kinds.get("kv_k"), "kv_k").get("preads", -1)) == 32
                 and int(_mapping(kinds.get("kv_v"), "kv_v").get(
                     "preads", -1)) == 32
                 and int(_mapping(kinds.get("embedding"), "embedding").get(
                     "preads", -1)) == 1,
                 f"{identity} MPIC pread split mismatch")
        _require(row.get("decode_cache_append_exact") is True,
                 f"{identity} decode cache append mismatch")

    failures_path = ROOT / "runs/mpic_baseline" / run_id / "failures.jsonl"
    failures = _read_jsonl(failures_path) if failures_path.exists() else []
    completed = set(identities)
    unresolved = [row for row in failures
                  if str(row.get("request_id")) not in completed]
    _require(not unresolved, "run has unresolved technical failures")
    retry_total = sum(int(row.get("retry_count", 0)) for row in rows)
    _require(retry_total == len(failures),
             "retry counters do not match durable failure events")
    return {
        "by_method": by_method,
        "by_image": by_image,
        "by_prompt": by_prompt,
        "failure_events": len(failures),
        "retry_total": retry_total,
        "duplicates": len(identities) - len(set(identities)),
    }


def _summaries(rows_by_method: Mapping[str, Sequence[Mapping[str, Any]]]) \
        -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for method in METHOD_KEYS:
        all_rows = list(rows_by_method[method])
        hits = [row for row in all_rows if int(row["turn_id"]) > 1]
        ttft = [_finite(row["ttft_ms"], "ttft") for row in hits]
        result[method] = {
            "method_key": method,
            "method_id": METHOD_IDS[method],
            "method": DISPLAY[method],
            "requests_all": len(all_rows),
            "requests_hit": len(hits),
            "accuracy_all": _mean(all_rows, "correct"),
            "accuracy_hit": _mean(hits, "correct"),
            "ttft_mean_ms": sum(ttft) / len(ttft),
            "ttft_p50_ms": _percentile(ttft, 50),
            "ttft_p95_ms": _percentile(ttft, 95),
            "ssd_mb_per_hit": _mean(hits, "ssd_read_bytes") / 1e6,
            "ssd_preads_per_hit": _mean(hits, "ssd_preads"),
            "recomputed_image_tokens_mean": _mean(
                hits, "n_recomputed_image_tokens"),
            "retained_image_context_ratio_mean": _mean(
                hits, "retained_image_context_ratio"),
            "request_e2e_mean_ms": _mean(hits, "request_e2e_ms"),
        }
    return result


def _verify_runner_summary(run_dir: Path, summaries: Mapping[str, Mapping[str, Any]]) \
        -> None:
    value = _read_json(run_dir / "summary.json")
    _require(value.get("schema_version") == RUN_SCHEMA,
             "summary schema mismatch")
    per_method = _mapping(value.get("per_method"), "summary.per_method")
    # The runner's atomic JSON serializer sorts object keys; JSON object order
    # is not the method-order contract.  The CSV below carries and validates
    # the canonical presentation order.
    _require(set(per_method) == set(METHOD_KEYS),
             "summary method key set mismatch")
    mapping = {
        "requests_all": "requests_all",
        "requests_cache_hit": "requests_hit",
        "accuracy_all": "accuracy_all",
        "accuracy_cache_hit": "accuracy_hit",
        "ttft_cache_hit_mean_ms": "ttft_mean_ms",
        "ttft_cache_hit_p50_ms": "ttft_p50_ms",
        "ttft_cache_hit_p95_ms": "ttft_p95_ms",
        "ssd_mb_cache_hit_mean": "ssd_mb_per_hit",
        "ssd_preads_cache_hit_mean": "ssd_preads_per_hit",
        "recomputed_image_tokens_cache_hit_mean":
            "recomputed_image_tokens_mean",
    }
    for method in METHOD_KEYS:
        runner = _mapping(per_method.get(method), f"summary.{method}")
        _require(runner.get("method_id") == METHOD_IDS[method]
                 and runner.get("display_label") == DISPLAY[method],
                 f"summary metadata mismatch for {method}")
        for runner_key, report_key in mapping.items():
            _close(runner.get(runner_key), summaries[method][report_key],
                   f"summary.{method}.{runner_key}")
    with (run_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    _require([row.get("method_key") for row in csv_rows] == list(METHOD_KEYS),
             "summary.csv method rows mismatch")


def _validate_runner_validation(run_dir: Path) -> dict[str, Any]:
    value = _read_json(run_dir / "validation.json")
    _require(value.get("schema_version") == RUN_SCHEMA
             and value.get("passed") is True,
             "runner validation did not pass")
    checks = _mapping(value.get("checks"), "runner validation checks")
    _require(bool(checks) and all(item is True for item in checks.values()),
             "runner validation contains a failed check")
    _require(int(value.get("observed_requests", -1)) == 1200
             and int(value.get("expected_requests", -1)) == 1200,
             "runner validation coverage mismatch")
    _require(int(value.get("unresolved_failures", -1)) == 0,
             "runner reports unresolved failures")
    _require(value.get("validation_scope")
             == "measurement_and_mpic_correctness_only"
             and value.get("artifact_protection_status")
             == "pending_external_verify_after"
             and value.get("final_validated_claim") is False,
             "runner did not preserve the protection/reporting hand-off")
    return value


def _validate_smoke(smoke_dir: Path) -> dict[str, Any]:
    _require(smoke_dir.is_dir() and not smoke_dir.is_symlink(),
             "smoke path is not a real directory")
    validation = _read_json(smoke_dir / "validation.json")
    correctness = _read_json(smoke_dir / "correctness_tests.json")
    positions = _read_json(smoke_dir / "position_diagnostics.json")
    _require(validation.get("schema_version") == SMOKE_SCHEMA
             and correctness.get("schema_version") == SMOKE_SCHEMA
             and positions.get("schema_version") == SMOKE_SCHEMA,
             "smoke schema mismatch")
    _require(validation.get("passed") is True
             and int(validation.get("images_completed", -1)) == 3
             and int(validation.get("images_expected", -1)) == 3
             and validation.get("failed_checks") == [],
             "three-image real-model smoke did not pass")
    recorded_sources = _mapping(
        validation.get("source_sha256"), "smoke source_sha256")
    smoke_source_paths = (
        "scripts/65_validate_mpic.py",
        "scripts/49_eval_query_aware_baseline.py",
        "mmimpress/mpic.py",
        "mmimpress/config.py",
        "mmimpress/model.py",
        "mmimpress/piggyback.py",
        "mmimpress/serve.py",
        "mmimpress/store.py",
        "tests/test_mpic.py",
    )
    _require(set(recorded_sources) == set(smoke_source_paths),
             "smoke source-hash inventory mismatch")
    for name in smoke_source_paths:
        _require(recorded_sources[name] == sha256_file(ROOT / name),
                 f"smoke source changed after validation: {name}")
    smoke_checks = _mapping(validation.get("checks"), "smoke checks")
    _require(bool(smoke_checks) and all(item is True for item in smoke_checks.values()),
             "smoke validation has a failed check")
    smoke_rows = _sequence(correctness.get("rows"), "smoke rows")
    _require(correctness.get("validation") == validation
             and len(smoke_rows) == 3,
             "correctness smoke does not embed the same validation")
    for row_value in smoke_rows:
        row = _mapping(row_value, "smoke row")
        boundary = _mapping(
            row.get("image_boundary_integrity"),
            "smoke image-boundary integrity")
        _require(
            boundary.get("boundary")
            == "after_all_smoke_requests_before_context_close"
            and boundary.get("unchanged") is True
            and boundary.get("validated_at_context_open_sha256")
            == boundary.get("live_after_all_requests_sha256")
            and float(boundary.get("audit_ms", -1.0)) >= 0.0
            and "out-of-band live payload sample" in str(
                boundary.get("timing_semantics", "")),
            f"smoke live payload audit failed: {row.get('image_id')}")
    required = {
        "all_kN_logits_within_predeclared_tolerance",
        "all_kN_caches_within_predeclared_tolerance",
        "all_k0_existing_fullload_comparisons_passed",
        "all_cache_lengths_exact", "all_layer_counters_exact",
        "dummy_sentinel_passed", "shifted_position_mapping_passed",
        "all_live_payload_hashes_unchanged",
    }
    _require(required <= {key for key, passed in smoke_checks.items() if passed},
             "smoke omits required selective-attention checks")
    position_rows = _sequence(positions.get("rows"), "position rows")
    _require(positions.get("support_level") == "LIMITED"
             and positions.get("paper_specifies_rope_relocation") is False
             and bool(position_rows)
             and all(_mapping(row, "position row").get("passed") is True
                     for row in position_rows),
             "position-change diagnostic is invalid")
    return {
        "validation": validation, "correctness": correctness,
        "positions": positions,
        "hashes": {
            name: sha256_file(smoke_dir / name) for name in (
                "validation.json", "correctness_tests.json",
                "position_diagnostics.json", "persistence.json",
            )
        },
    }


def _protection_payload(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": snapshot["schema_version"],
        "scope_roots": snapshot["scope_roots"],
        "excluded_new_roots": snapshot["excluded_new_roots"],
        "fingerprint_policy": snapshot["fingerprint_policy"],
        "entries": snapshot["entries"],
    }


def _validate_protection(run_dir: Path, results_dir: Path) -> dict[str, Any]:
    manifest = _read_json(run_dir / "protected_artifacts_before.json")
    validation = _read_json(run_dir / "protected_artifacts_validation.json")
    _require(manifest.get("schema_version") == PROTECTION_SCHEMA
             and validation.get("schema_version") == PROTECTION_SCHEMA,
             "artifact-protection schema mismatch")
    _require(manifest.get("excluded_new_roots")
             == [str(run_dir), str(results_dir)],
             "artifact-protection exclusions mismatch")
    digest = canonical_hash(_protection_payload(manifest))
    _require(manifest.get("manifest_sha256") == digest,
             "before manifest content hash mismatch")
    _require(validation.get("passed") is True
             and validation.get("before_manifest_sha256") == digest
             and validation.get("missing_paths") == []
             and validation.get("changed_paths") == [],
             "prior-artifact before/after protection failed")
    _require(int(manifest.get("entry_count", -1)) > 0
             and int(manifest.get("file_count", -1)) > 0,
             "artifact-protection manifest is empty")
    return {"manifest": manifest, "validation": validation}


def _validate_persistence(value: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    _require(value.get("schema_version") == RUN_SCHEMA,
             "persistence schema mismatch")
    images = list(_sequence(value.get("images"), "persistence.images"))
    _require(len(images) == 40, "persistence does not cover 40 images")
    grouped: dict[str, list[Mapping[str, Any]]] = {key: [] for key in STORE_METHODS}
    seen: set[str] = set()
    for item in images:
        row = _mapping(item, "persistence image")
        image = str(row.get("image_id"))
        _require(image not in seen, f"duplicate persistence image: {image}")
        seen.add(image)
        stores = _mapping(row.get("stores"), f"persistence.{image}.stores")
        _require(set(stores) == set(STORE_METHODS),
                 f"persistence store set mismatch for {image}")
        for method in STORE_METHODS:
            evidence = _mapping(stores[method], f"persistence.{image}.{method}")
            _require(evidence.get("image_id") == image,
                     f"persistence image identity mismatch: {image}/{method}")
            _require(evidence.get("measurement_complete") is True,
                     f"persistence measurement incomplete: {image}/{method}")
            byte_counts = _mapping(evidence.get("bytes"), "persistence.bytes")
            timing = _mapping(evidence.get("timing_ms"), "persistence.timing")
            durability = _mapping(evidence.get("durability"),
                                  "persistence.durability")
            _require(int(byte_counts.get("total", 0)) > 0
                     and int(byte_counts.get("visual_kv", 0)) > 0,
                     f"empty persisted store: {image}/{method}")
            _require(_finite(timing.get("persist_ms"), "persist_ms",
                             nonnegative=True) > 0,
                     f"invalid persistence time: {image}/{method}")
            _require(durability.get("same_filesystem_staging") is True
                     and durability.get("atomic_no_clobber") is True,
                     f"nondurable store publication: {image}/{method}")
            if method == "mpic32":
                _require(int(byte_counts.get("visual_input", 0)) > 0
                         and int(byte_counts.get("probe_sidecar", 0)) == 0,
                         f"MPIC sidecar accounting mismatch: {image}")
            grouped[method].append(evidence)
    return grouped


def _validate_mpic_image_integrity(run_dir: Path,
                                   expected_images: set[str]) -> list[dict[str, Any]]:
    root = run_dir / "mpic_image_integrity"
    _require(root.is_dir() and not root.is_symlink(),
             "MPIC image-boundary integrity directory is missing")
    paths = sorted(root.glob("*.json"))
    _require(len(paths) == 40, "expected 40 MPIC image integrity documents")
    documents = [_read_json(path) for path in paths]
    observed = {str(item.get("image_id")) for item in documents}
    _require(observed == expected_images and len(observed) == len(documents),
             "MPIC image integrity coverage mismatch")
    for item in documents:
        _require(item.get("schema_version") == RUN_SCHEMA
                 and item.get("unchanged") is True
                 and item.get("validated_at_context_open_sha256")
                 == item.get("live_after_all_hits_sha256")
                 and str(item.get("boundary", "")).startswith(
                     "after_all_5_mpic_hits"),
                 f"MPIC image-boundary integrity failed: {item.get('image_id')}")
    return documents


def _persistence_rows(grouped: Mapping[str, Sequence[Mapping[str, Any]]]) \
        -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    policies = {
        "recompute": "none",
        "fullload": "owned canonical raster store",
        "qa_chunk25": "shares FullLoad canonical raster store",
        "ours25": "owned importance-repacked store",
        "mpic32": "owned canonical KV + visual-input sidecar",
    }
    for method in METHOD_KEYS:
        owner = "fullload" if method == "qa_chunk25" else method
        evidence = list(grouped.get(owner, ()))
        backing_totals = [int(_mapping(item["bytes"], "bytes")["total"])
                          for item in evidence]
        timings = [_mapping(item["timing_ms"], "timing") for item in evidence]
        mean_byte = lambda key: (sum(int(_mapping(item["bytes"], "bytes").get(
            key, 0)) for item in evidence) / len(evidence) if evidence else 0.0)
        mean_time = lambda key: (sum(float(item.get(key, 0.0) or 0.0)
                                     for item in timings) / len(timings)
                                 if timings else 0.0)
        fsync_keys = ("fsync_ms", "file_fsync_ms", "directory_fsync_ms")
        fsync = (sum(sum(float(item.get(key, 0.0) or 0.0)
                         for key in fsync_keys) for item in timings)
                 / len(timings) if timings else 0.0)
        # QA-Chunk consumes FullLoad's store; it owns no additional bytes or
        # persistence operation.  Preserve the backing-store size separately.
        owns_store = method in STORE_METHODS
        incremental_totals = backing_totals if owns_store else []
        rows.append({
            "method_key": method, "method": DISPLAY[method],
            "store_policy": policies[method],
            "store_count": len(evidence) if owns_store else 0,
            "backing_store_count": len(evidence),
            "incremental_store_owner": owns_store,
            "total_bytes": sum(incremental_totals),
            "mean_total_bytes": (sum(incremental_totals)
                                 / len(incremental_totals)
                                 if incremental_totals else 0.0),
            "shared_backing_mean_total_bytes": (
                sum(backing_totals) / len(backing_totals)
                if backing_totals else 0.0),
            "mean_visual_kv_bytes": mean_byte("visual_kv") if owns_store else 0.0,
            "mean_visual_input_bytes": (
                mean_byte("visual_input") if owns_store else 0.0),
            "mean_probe_sidecar_bytes": (
                mean_byte("probe_sidecar") if owns_store else 0.0),
            "mean_separator_sidecar_bytes": (
                mean_byte("separator_sidecar") if owns_store else 0.0),
            "mean_persist_ms": mean_time("persist_ms") if owns_store else 0.0,
            "mean_visual_input_capture_materialize_ms": (
                mean_time("visual_input_capture_materialize_ms")
                if owns_store else 0.0),
            "mean_provisioning_post_response_ms": (
                (sum(float(item.get(
                    "provisioning_post_response_ms",
                    item.get("persist_ms", 0.0)) or 0.0)
                     for item in timings) / len(timings))
                if owns_store and timings else 0.0),
            "mean_ssd_write_ms": (
                mean_time("ssd_write_ms") if owns_store else 0.0),
            "mean_fsync_ms": fsync if owns_store else 0.0,
            "storage_interpretation": (
                "shared; zero additional store beyond FullLoad"
                if method == "qa_chunk25" else
                "not applicable" if method == "recompute" else
                "method-owned one-time store"),
        })
    return rows


def _cluster_bootstrap(
    pairs: Mapping[str, Sequence[float]], *, seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Percentile CI after resampling image clusters with replacement."""
    images = sorted(pairs)
    _require(bool(images), "cluster bootstrap has no images")
    clusters = [tuple(float(value) for value in pairs[image]) for image in images]
    _require(all(cluster for cluster in clusters), "bootstrap has an empty cluster")
    observed_values = [value for cluster in clusters for value in cluster]
    observed = sum(observed_values) / len(observed_values)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(replicates):
        total, count = 0.0, 0
        for _ in images:
            cluster = clusters[rng.randrange(len(clusters))]
            total += sum(cluster)
            count += len(cluster)
        samples.append(total / count)
    return {
        "estimate": observed,
        "ci95_low": _percentile(samples, 2.5),
        "ci95_high": _percentile(samples, 97.5),
        "clusters": len(images), "paired_observations": len(observed_values),
        "bootstrap_replicates": replicates, "seed": seed,
        "interval": "image-cluster percentile bootstrap",
    }


def _paired_comparisons(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    indexed = {(str(row["image_id"]), str(row["question_id"]),
                str(row["method_key"])): row for row in rows}
    output: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "difference_direction": "candidate_minus_fullload",
        "bootstrap_unit": "image (all questions in sampled image retained)",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "seed_base": BOOTSTRAP_SEED,
        "comparisons": [],
    }
    for method_index, method in enumerate(METHOD_KEYS):
        if method == "fullload":
            continue
        all_quality: dict[str, list[float]] = defaultdict(list)
        hit_quality: dict[str, list[float]] = defaultdict(list)
        hit_ttft: list[float] = []
        first_token_equal = prediction_equal = 0
        n = 0
        for (image, question, key), candidate in indexed.items():
            if key != method:
                continue
            reference = indexed[(image, question, "fullload")]
            difference = float(candidate["correct"]) - float(reference["correct"])
            all_quality[image].append(difference)
            if int(candidate["turn_id"]) > 1:
                hit_quality[image].append(difference)
                hit_ttft.append(float(candidate["ttft_ms"])
                                - float(reference["ttft_ms"]))
            first_token_equal += (
                int(candidate["first_token_id"]) == int(reference["first_token_id"]))
            prediction_equal += (
                str(candidate["prediction"]) == str(reference["prediction"]))
            n += 1
        output["comparisons"].append({
            "candidate_method_key": method,
            "candidate": DISPLAY[method], "reference": "FullLoad",
            "quality_all": _cluster_bootstrap(
                all_quality, seed=BOOTSTRAP_SEED + method_index * 2),
            "quality_hit": _cluster_bootstrap(
                hit_quality, seed=BOOTSTRAP_SEED + method_index * 2 + 1),
            "paired_hit_ttft_mean_difference_ms": (
                sum(hit_ttft) / len(hit_ttft)),
            "first_token_agreement_all": first_token_equal / n,
            "prediction_agreement_all": prediction_equal / n,
        })
    return output


def _latency_rows(rows_by_method: Mapping[str, Sequence[Mapping[str, Any]]]) \
        -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    fields = (
        "request_e2e_ms", "decode_ms", "prompt_build_ms", "tokenization_ms",
        "processor_total_ms", "input_prepare_ms", "input_h2d_ms",
        "ssd_read_ms", "kv_read_ms", "embedding_read_ms", "h2d_ms",
        "cache_assembly_ms", "position_processing_ms",
        "selective_prefill_interval_ms", "prefill_ms",
    )
    for method in METHOD_KEYS:
        hits = [row for row in rows_by_method[method]
                if int(row["turn_id"]) > 1]
        ttft = [float(row["ttft_ms"]) for row in hits]
        row: dict[str, Any] = {
            "method_key": method, "method": DISPLAY[method],
            "scope": "GQA Q2-Q6 (200 same-run requests)",
            "requests": len(hits), "ttft_mean_ms": sum(ttft) / len(ttft),
            "ttft_p50_ms": _percentile(ttft, 50),
            "ttft_p95_ms": _percentile(ttft, 95),
        }
        row.update({f"{field}_mean": _mean(hits, field) for field in fields})
        row["timing_note"] = (
            "MPIC selective_prefill_interval is inclusive and overlapping; "
            "components must not be subtracted from TTFT"
            if method == "mpic32" else
            "phase fields follow the existing server instrumentation")
        output.append(row)
    return output


def _io_rows(rows_by_method: Mapping[str, Sequence[Mapping[str, Any]]]) \
        -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in METHOD_KEYS:
        hits = [row for row in rows_by_method[method]
                if int(row["turn_id"]) > 1]
        bytes_mean = _mean(hits, "ssd_read_bytes") or 0.0
        output.append({
            "method_key": method, "method": DISPLAY[method],
            "scope": "GQA Q2-Q6 (200 same-run requests)",
            "requests": len(hits), "ssd_total_bytes_mean": bytes_mean,
            "ssd_mb_per_request": bytes_mean / 1e6,
            "preads_mean": _mean(hits, "ssd_preads"),
            "kv_bytes_mean": (_mean(hits, "ssd_kv_bytes")
                              if method == "mpic32"
                              else _mean(hits, "normal_kv_read_bytes")),
            "embedding_bytes_mean": _mean(hits, "ssd_embedding_bytes"),
            "separator_bytes_mean": (
                _mean(hits, "ssd_separator_bytes")
                if method == "mpic32" else _mean(hits, "separator_read_bytes")),
            "metadata_bytes_mean": _mean(hits, "ssd_metadata_bytes"),
            "probe_bytes_mean": _mean(hits, "probe_read_bytes"),
            "actual_ssd_ratio_vs_fullload_mean": _mean(
                hits, "actual_ssd_ratio_vs_fullload"),
            "recomputed_image_tokens_mean": _mean(
                hits, "n_recomputed_image_tokens"),
            "reused_image_tokens_mean": _mean(hits, "n_reused_image_tokens"),
            "retained_image_context_ratio_mean": _mean(
                hits, "retained_image_context_ratio"),
            "visual_kv_scope_note": (
                "zero Visual-KV SSD reads; ordinary image/model file I/O is not claimed zero"
                if method == "recompute" else "measured SSD cache payload"),
        })
    return output


def _environment_text(config: Mapping[str, Any],
                      runtime: Mapping[str, Any]) -> str:
    """Render the runner-frozen GPU/runtime identity without probing CUDA."""
    lines = [
        f"model={config.get('model')}",
        f"model_revision={config.get('model_revision')}",
        f"load_4bit={config.get('load_4bit')}",
        f"quantization={config.get('quantization')}",
        f"compute_dtype={config.get('compute_dtype')}",
        f"attention_implementation={config.get('attention_implementation')}",
        f"decoding={config.get('decoding')}",
        f"max_new_tokens={config.get('max_new_tokens')}",
        f"seed={config.get('seed')}",
    ]
    for key in sorted(runtime):
        value = runtime[key]
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
        else:
            rendered = str(value)
        lines.append(f"runtime.{key}={rendered}")
    return "\n".join(lines) + "\n"


def _source_review() -> str:
    return """# MPIC source review

Primary reference: the complete local paper `papers/mpic.md` (SHA-256
`7253687b8a076fbea6e49fc8d9bffc856c3be33b1b7a372cba5fd5d00eaa503b`),
cross-checked against arXiv v2 and the authors' MPIC project/publication
pages.  No author-linked public source repository, commit, or software
license was found, so this work does not claim to run an official MPIC
implementation.

The reproduced paper mechanism is selective attention: all current text rows
and the first *k* canonical expanded image rows are recomputed through every
decoder layer, while cached K/V for the other image rows remains in the full
attention context.  MPIC-*k* names an image-token count; here *k*=32, not 32%,
32 chunks, a score-selected Top-32, or Ours25's 25% retention budget.  The
paper describes dummy cache slots that are replaced before attention and a
single selective prefill rather than a second full-prefix pass.

Paper-unspecified details are implementation choices: the exact Turn-1 source
prompt, Transformers indexed cache assembly, AnyRes structural-row handling,
and post-RoPE phase relocation for shifted positions.  The paper's mixed
cache-hit/cache-miss load/compute overlap has no opportunity in this
single-image all-hit pilot; layer-wise prefetch was not added.  The correct
label is **MPIC-style selective recomputation implemented in our SSD-resident
serving harness**, reported as **MPIC-32 (SSD adaptation)**.

The paper used vLLM 0.9.0, LLaVA-1.6 7B checkpoints, H800 hardware, and
MMDU/SparklesEval-style workloads.  This repository uses Transformers,
LLaVA-1.6 Vicuna 7B in 4-bit NF4/BF16 compute, one RTX 4090, and independent
single-image fixed-prefix GQA questions.  Consequently this is a mechanism
adaptation and local same-run comparison, not an end-to-end reproduction of
the paper's system or performance claims.

Reviewed web references:

- https://arxiv.org/abs/2502.01960
- https://arxiv.org/html/2502.01960v2
- https://shijuzhao.github.io/pic
"""


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def _analysis(
    summaries: Mapping[str, Mapping[str, Any]],
    comparisons: Mapping[str, Any], latency: Sequence[Mapping[str, Any]],
    io_rows: Sequence[Mapping[str, Any]], persistence: Sequence[Mapping[str, Any]],
    row_evidence: Mapping[str, Any], protection: Mapping[str, Any],
    smoke: Mapping[str, Any],
) -> str:
    lines = [
        "# MPIC-32 SSD adaptation: same-run GQA pilot",
        "",
        "| Method | Acc all | Acc hit | TTFT mean | p50 | p95 | SSD MB/req | Preads | Recomputed image tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    token_label = {
        "recompute": "all", "fullload": "0", "qa_chunk25": "0",
        "ours25": "0", "mpic32": "min(32,N)",
    }
    for method in METHOD_KEYS:
        row = summaries[method]
        lines.append(
            f"| {DISPLAY[method]} | {_fmt(row['accuracy_all'])} | "
            f"{_fmt(row['accuracy_hit'])} | {_fmt(row['ttft_mean_ms'], 2)} ms | "
            f"{_fmt(row['ttft_p50_ms'], 2)} ms | "
            f"{_fmt(row['ttft_p95_ms'], 2)} ms | "
            f"{_fmt(row['ssd_mb_per_hit'], 3)} | "
            f"{_fmt(row['ssd_preads_per_hit'], 2)} | {token_label[method]} |")

    mpic_io = next(row for row in io_rows if row["method_key"] == "mpic32")
    full_io = next(row for row in io_rows if row["method_key"] == "fullload")
    ours_io = next(row for row in io_rows if row["method_key"] == "ours25")
    mpic_rows = [row for row in row_evidence["by_method"]["mpic32"]
                 if int(row["turn_id"]) > 1]
    n_values = [int(row["n_image_tokens"]) for row in mpic_rows]
    active_text = [int(row["n_recomputed_text_tokens"]) for row in mpic_rows]
    mpic_latency = next(row for row in latency if row["method_key"] == "mpic32")
    p_by_key = {row["method_key"]: row for row in persistence}

    lines += [
        "",
        "## Scope and interpretation",
        "",
        "All five arms were measured in this run on the frozen 40-image, "
        "240-question GQA slice. Accuracy uses all 240 questions per method; "
        "latency and SSD statistics use Q2–Q6 (200 requests per method). "
        "The six questions per image are independent requests, not a native "
        "conversation. ReComp's table entry is zero **Visual-KV** SSD traffic; "
        "it is not a claim that ordinary model/image file I/O is zero.",
        "",
        "MPIC and Ours do not share a 25% budget. MPIC retains the complete "
        f"image context ({_fmt(summaries['mpic32']['retained_image_context_ratio_mean'])}) "
        "while recomputing a small prefix; Ours retains approximately "
        f"{_fmt(summaries['ours25']['retained_image_context_ratio_mean'])}. "
        "Ours is a context-pruning quality–I/O trade-off; MPIC is a "
        "full-context partial-recomputation alternative.",
        "",
        "## MPIC selective-attention evidence",
        "",
        f"Across 200 MPIC hits, image-token counts ranged from {min(n_values)} "
        f"to {max(n_values)}. Every layer recomputed canonical local rows "
        f"0..31 plus all current text rows (mean {_fmt(sum(active_text)/len(active_text), 2)} "
        "text rows/request), retained every cached image key, executed one "
        "decoder selective-prefill pass, and then appended decode K/V at the "
        "exact full-cache length. The source payload hashes were unchanged. "
        "The main pilot kept source and target image positions identical, so "
        "the shifted-position path is evidenced only by the separate smoke "
        "diagnostic.",
        "",
        f"MPIC read {_fmt(mpic_io['ssd_mb_per_request'], 3)} MB/request versus "
        f"FullLoad {_fmt(full_io['ssd_mb_per_request'], 3)} and Ours25 "
        f"{_fmt(ours_io['ssd_mb_per_request'], 3)}. Its mean split was "
        f"{_fmt(mpic_io['kv_bytes_mean']/1e6, 3)} MB cached K/V plus "
        f"{_fmt(mpic_io['embedding_bytes_mean']/1e6, 3)} MB recomputation-input "
        "embeddings. Chunk alignment yielded exactly 65 preads/request "
        "(32 K, 32 V, one embedding); a k=32 request can therefore still read "
        "the complete visual-KV chunks in this layout.",
        "",
        "MPIC mean timing instrumentation (overlapping intervals) was: "
        f"KV pread {_fmt(mpic_latency['kv_read_ms_mean'], 2)} ms, embedding "
        f"pread {_fmt(mpic_latency['embedding_read_ms_mean'], 2)} ms, H2D "
        f"{_fmt(mpic_latency['h2d_ms_mean'], 2)} ms, cache assembly "
        f"{_fmt(mpic_latency['cache_assembly_ms_mean'], 2)} ms, position "
        f"processing {_fmt(mpic_latency['position_processing_ms_mean'], 2)} ms, "
        f"and inclusive selective-prefill interval "
        f"{_fmt(mpic_latency['selective_prefill_interval_ms_mean'], 2)} ms. "
        "That interval includes reads, transfers, assembly, all decoder "
        "layers, final norm, and LM head; it is not pure GPU compute and the "
        "components must not be subtracted from TTFT.",
        "",
        "## Paired quality and latency differences",
        "",
        "Differences below are candidate minus FullLoad. Quality intervals "
        f"use {BOOTSTRAP_REPLICATES:,} deterministic percentile-bootstrap "
        "replicates clustered by image. An interval containing zero does not "
        "prove equivalence.",
        "",
        "| Candidate | Acc diff all [95% CI] | Acc diff hit [95% CI] | Paired hit TTFT diff | First-token agreement |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in comparisons["comparisons"]:
        qa, qh = item["quality_all"], item["quality_hit"]
        lines.append(
            f"| {item['candidate']} | {_fmt(qa['estimate'])} "
            f"[{_fmt(qa['ci95_low'])}, {_fmt(qa['ci95_high'])}] | "
            f"{_fmt(qh['estimate'])} "
            f"[{_fmt(qh['ci95_low'])}, {_fmt(qh['ci95_high'])}] | "
            f"{_fmt(item['paired_hit_ttft_mean_difference_ms'], 2)} ms | "
            f"{_fmt(item['first_token_agreement_all'])} |")

    lines += [
        "",
        "## One-time persistence and storage",
        "",
        "Persistence is measured separately from cache-hit TTFT and includes "
        "the store's durable publication. QA-Chunk25 shares FullLoad's "
        "canonical raster store, so its incremental store is zero. The MPIC "
        "provisioning total also includes visual-input D2H materialization "
        "performed when its Turn-1 capture exits; persist-call time is retained "
        "as a separate column.",
        "",
        "| Method | Store policy | Mean bytes/image | Mean persist call | Mean post-response provisioning | Capture D2H | Added visual input |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_KEYS:
        row = p_by_key[method]
        lines.append(
            f"| {DISPLAY[method]} | {row['store_policy']} | "
            f"{_fmt(row['mean_total_bytes']/1e6, 3)} MB | "
            f"{_fmt(row['mean_persist_ms'], 2)} ms | "
            f"{_fmt(row['mean_provisioning_post_response_ms'], 2)} ms | "
            f"{_fmt(row['mean_visual_input_capture_materialize_ms'], 2)} ms | "
            f"{_fmt(row['mean_visual_input_bytes']/1e6, 3)} MB |")

    lines += [
        "",
        "## Correctness, positions, and protection",
        "",
        "The three-image quantized smoke passed finite-logit/generation "
        "checks, k=0 FullLoad comparisons, k=N same-embedding full-prefill "
        "logit and cache comparisons under the predeclared tolerance, k=32 "
        "layer/counter checks, dummy-sentinel replacement, causal masking, "
        "request isolation, and exact decode-cache appends. Its shifted-prefix "
        "diagnostic passed mapping, causal mask, source immutability, and "
        "post-RoPE phase relocation checks. Position-change support remains "
        "LIMITED because relocation does not reconstruct hidden-state changes "
        "caused by different preceding text, and the paper does not specify "
        "this relocation detail.",
        "",
        f"The run contains 1,200 unique completed requests, "
        f"{row_evidence['failure_events']} durable technical failure event(s), "
        f"{row_evidence['retry_total']} retry count(s), and "
        f"{row_evidence['duplicates']} duplicates. The prior-artifact guard "
        f"verified {protection['manifest']['entry_count']} pre-existing paths "
        "with no missing or changed protected entry. Large files use the "
        "manifest's bounded nine-window fingerprint policy to avoid warming "
        "hundreds of GiB of unrelated SSD payload; this is an accidental-change "
        "guard, not an adversarial cryptographic proof.",
        "",
        "## Parallelism and limitations",
        "",
        "The paper-described overlap between cache-hit image transfer and "
        "cache-miss image computation is **not applicable** to this "
        "single-image all-hit pilot. No layer-wise prefetch was implemented. "
        "Synchronous reads/transfers are reported as measured and are not "
        "claimed to overlap. `posix_fadvise(DONTNEED)` conditioning occurs "
        "outside TTFT but does not guarantee a cold SSD controller cache.",
        "",
        "This fixed-prefix GQA workload requires little position relocation "
        "and cannot represent native multi-image MPIC scheduling or the "
        "paper's full system. No MT-GQA, MT-VQA-v2, ConvBench, or ReKV run is "
        "part of this result.",
        "",
        "## Final status",
        "",
        "```text",
        "IMPLEMENTATION: PASS",
        "SELECTIVE-ATTENTION CORRECTNESS: PASS",
        "POSITION-CHANGE SUPPORT: LIMITED",
        "GQA PILOT: COMPLETE",
        "ARTIFACT PROTECTION: PASS",
        "MPIC-32 SSD ADAPTATION VALIDATED: YES",
        "```",
        "",
        "`YES` validates this paper-guided SSD adaptation and its local "
        "measurements; it does not claim an official or complete reproduction "
        "of the MPIC serving system.",
    ]
    return "\n".join(lines) + "\n"


def _source_revision(
    run_dir: Path, smoke_dir: Path, runner_hashes: Mapping[str, str],
) -> dict[str, Any]:
    sources = (
        "papers/mpic.md", "docs/mpic_baseline_contract.md",
        "mmimpress/mpic.py", "scripts/65_validate_mpic.py",
        "scripts/66_protect_mpic_artifacts.py", "scripts/67_eval_mpic_gqa.py",
        "scripts/68_report_mpic_gqa.py", "tests/test_mpic.py",
        "tests/test_mpic_report.py",
        "scripts/49_eval_query_aware_baseline.py",
        "scripts/52_eval_query_aware_chunk_baseline.py",
        "mmimpress/config.py", "mmimpress/dataset.py",
        "mmimpress/model.py", "mmimpress/store.py", "mmimpress/serve.py",
        "mmimpress/piggyback.py", "mmimpress/cvpr25.py",
        "mmimpress/reorder.py", "mmimpress/sparsevlm.py",
    )
    hashes = {name: sha256_file(ROOT / name) for name in sources}
    _require(hashes["papers/mpic.md"] == EXPECTED_PAPER_SHA256,
             "local paper changed after source review")
    return {
        "schema_version": REPORT_SCHEMA,
        "repository": str(ROOT),
        "git_revision": None,
        "git_status": "unavailable: repository .git metadata is empty",
        "primary_reference": "papers/mpic.md",
        "primary_reference_sha256": EXPECTED_PAPER_SHA256,
        "official_implementation": {
            "found": False, "repository_url": None,
            "commit": None, "license": None,
            "claim": "paper-guided local SSD adaptation, not official code",
        },
        "source_files_sha256": hashes,
        "runner_exports_sha256": dict(runner_hashes),
        "smoke_dir": str(smoke_dir.resolve()),
        "run_dir": str(run_dir.resolve()),
    }


def _report_validation(
    runner_validation: Mapping[str, Any], smoke: Mapping[str, Any],
    protection: Mapping[str, Any], row_evidence: Mapping[str, Any],
    integrity_documents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    checks = {
        "runner_validation_passed": runner_validation.get("passed") is True,
        "independent_1200_unique_rows": sum(
            len(value) for value in row_evidence["by_method"].values()) == 1200,
        "exact_five_arm_240_and_200_coverage": all(
            len(rows) == 240 and sum(int(row["turn_id"]) > 1 for row in rows) == 200
            for rows in row_evidence["by_method"].values()),
        "no_duplicate_requests": row_evidence["duplicates"] == 0,
        "no_unresolved_failures": True,
        "three_image_real_model_smoke_passed":
            smoke["validation"].get("passed") is True,
        "shifted_position_support_tested_limited":
            smoke["positions"].get("support_level") == "LIMITED",
        "prior_artifacts_unchanged":
            protection["validation"].get("passed") is True,
        "mpic_payload_unchanged_at_all_40_image_boundaries":
            len(integrity_documents) == 40
            and all(item.get("unchanged") is True
                    for item in integrity_documents),
        "publication_policy_no_overwrite_or_identical": True,
    }
    return {
        "schema_version": REPORT_SCHEMA,
        "passed": all(checks.values()), "checks": checks,
        "status": {
            "implementation": "PASS",
            "selective_attention_correctness": "PASS",
            "position_change_support": "LIMITED",
            "gqa_pilot": "COMPLETE",
            "artifact_protection": "PASS",
            "mpic32_ssd_adaptation_validated": "YES",
        },
        "failure_events": row_evidence["failure_events"],
        "retry_count": row_evidence["retry_total"],
        "duplicate_count": row_evidence["duplicates"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--smoke-dir", type=Path,
        default=ROOT / "runs/mpic_baseline/smoke_final_linked_3")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir, results_dir = _validate_roots(args.run_dir, args.results_dir)
    smoke_dir = args.smoke_dir.resolve()

    runner_hashes = _verify_runner_exports(run_dir, results_dir)
    config = _read_json(run_dir / "config.json")
    manifest = _read_json(run_dir / "manifest.json")
    _validate_config(config, manifest, run_dir, results_dir)
    runtime = _read_json(run_dir / "runtime_fingerprint.json")
    _require(config.get("runtime") == runtime
             and config.get("model") == runtime.get("model_id")
             and config.get("model_revision") == runtime.get("model_revision")
             and config.get("compute_dtype") == runtime.get("compute_dtype")
             and config.get("attention_implementation")
             == runtime.get("attention_implementation"),
             "config and frozen runtime fingerprint differ")
    runner_validation = _validate_runner_validation(run_dir)
    rows = _read_jsonl(run_dir / "raw.jsonl")
    row_evidence = _validate_rows(rows, run_dir.name)
    summaries = _summaries(row_evidence["by_method"])
    _verify_runner_summary(run_dir, summaries)
    smoke = _validate_smoke(smoke_dir)
    smoke_gate = _mapping(manifest.get("smoke_validation"),
                          "manifest.smoke_validation")
    _require(smoke_gate.get("passed") is True
             and Path(str(smoke_gate.get("path"))).resolve()
             == (smoke_dir / "validation.json").resolve()
             and smoke_gate.get("sha256")
             == smoke["hashes"]["validation.json"],
             "pilot did not gate on the published hardened smoke evidence")
    protection = _validate_protection(run_dir, results_dir)
    persistence_value = _read_json(run_dir / "persistence.json")
    persistence_grouped = _validate_persistence(persistence_value)
    integrity_documents = _validate_mpic_image_integrity(
        run_dir, set(row_evidence["by_image"]))
    persistence_rows = _persistence_rows(persistence_grouped)
    comparisons = _paired_comparisons(rows)
    latency_rows = _latency_rows(row_evidence["by_method"])
    io_rows = _io_rows(row_evidence["by_method"])
    report_validation = _report_validation(
        runner_validation, smoke, protection, row_evidence,
        integrity_documents)
    _require(report_validation["passed"], "independent report validation failed")

    summary_fields = (
        "method_key", "method_id", "method", "requests_all", "requests_hit",
        "accuracy_all", "accuracy_hit", "ttft_mean_ms", "ttft_p50_ms",
        "ttft_p95_ms", "ssd_mb_per_hit", "ssd_preads_per_hit",
        "recomputed_image_tokens_mean", "retained_image_context_ratio_mean",
        "request_e2e_mean_ms",
    )
    latency_fields = tuple(latency_rows[0])
    io_fields = tuple(io_rows[0])
    persistence_fields = tuple(persistence_rows[0])
    generated: dict[str, bytes] = {
        "source_review.md": _source_review().encode("utf-8"),
        "implementation_contract.md": (
            ROOT / "docs/mpic_baseline_contract.md").read_bytes(),
        "environment.txt": _environment_text(config, runtime).encode("utf-8"),
        "source_revision.json": _json_bytes(_source_revision(
            run_dir, smoke_dir, runner_hashes)),
        "correctness_tests.json": (
            smoke_dir / "correctness_tests.json").read_bytes(),
        "position_diagnostics.json": (
            smoke_dir / "position_diagnostics.json").read_bytes(),
        "latency_breakdown.csv": _csv_bytes(latency_rows, latency_fields),
        "io_breakdown.csv": _csv_bytes(io_rows, io_fields),
        "persistence.csv": _csv_bytes(persistence_rows, persistence_fields),
        "quality_comparisons.json": _json_bytes(comparisons),
        "mpic_image_integrity.json": _json_bytes({
            "schema_version": REPORT_SCHEMA,
            "images": integrity_documents,
        }),
        "report_validation.json": _json_bytes(report_validation),
        "ANALYSIS.md": _analysis(
            summaries, comparisons, latency_rows, io_rows, persistence_rows,
            row_evidence, protection, smoke).encode("utf-8"),
    }

    # Runner-owned required files are already byte-identical in both roots.
    # The independently recomputed summary is useful additional evidence,
    # while summary.csv itself remains immutable and runner-owned.
    generated["independent_summary.csv"] = _csv_bytes(
        [summaries[key] for key in METHOD_KEYS], summary_fields)
    generated["protected_artifacts_before.json"] = (
        run_dir / "protected_artifacts_before.json").read_bytes()
    generated["protected_artifacts_validation.json"] = (
        run_dir / "protected_artifacts_validation.json").read_bytes()

    required = (
        "source_review.md", "implementation_contract.md", "config.json",
        "environment.txt", "source_revision.json", "raw.jsonl", "summary.csv",
        "correctness_tests.json", "position_diagnostics.json",
        "latency_breakdown.csv", "io_breakdown.csv", "persistence.csv",
        "validation.json", "ANALYSIS.md",
    )
    artifact_names = (*required, "independent_summary.csv",
                      "quality_comparisons.json", "report_validation.json",
                      "mpic_image_integrity.json",
                      "protected_artifacts_before.json",
                      "protected_artifacts_validation.json")
    artifacts = {
        "schema_version": REPORT_SCHEMA,
        "run_dir": str(run_dir), "results_dir": str(results_dir),
        "required_files": list(required),
        "files_sha256": {
            name: (hashlib.sha256(generated[name]).hexdigest()
                   if name in generated else sha256_file(run_dir / name))
            for name in artifact_names
        },
        "policy": "atomic no-clobber; existing files must be byte-identical",
    }
    generated["report_artifacts.json"] = _json_bytes(artifacts)
    _preflight_publications(run_dir, results_dir, generated)
    for name, payload in generated.items():
        _publish_pair(run_dir, results_dir, name, payload)
    for name in required:
        _require(sha256_file(run_dir / name) == sha256_file(results_dir / name),
                 f"required report artifact differs across roots: {name}")
    print(json.dumps({
        "status": "validated", "run_dir": str(run_dir),
        "results_dir": str(results_dir), "requests": len(rows),
        "required_artifacts": len(required),
        "analysis": str(results_dir / "ANALYSIS.md"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReportValidationError, OSError, ValueError) as error:
        print(f"MPIC report failed: {error}", file=sys.stderr)
        raise SystemExit(1)

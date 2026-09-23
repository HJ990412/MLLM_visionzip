#!/usr/bin/env python3
"""Independently validate and publish the ReKV six-arm GQA report bundle.

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
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent.parent
RUN_SCHEMA = "rekv-gqa-six-arm-v1"
REPORT_SCHEMA = "rekv-gqa-final-report-v1"
SMOKE_SCHEMA = "mpic-correctness-smoke-v1"
PROTECTION_SCHEMA = "rekv-protected-artifacts-v1"
EXPECTED_INDEX_SHA256 = (
    "514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a"
)
EXPECTED_WORKLOAD_SHA256 = (
    "97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66"
)
EXPECTED_PAPER_SHA256 = (
    "7253687b8a076fbea6e49fc8d9bffc856c3be33b1b7a372cba5fd5d00eaa503b"
)
METHOD_KEYS = ("recompute", "fullload", "mpic32", "qa_chunk25", "rekv_chunk25", "ours25")
METHOD_IDS = {
    "recompute": "recompute",
    "fullload": "fullload",
    "qa_chunk25": "qa_chunk25",
    "ours25": "imageonly_prefix25",
    "mpic32": "mpic32_ssd",
    "rekv_chunk25": "rekv_chunk25",
}
DISPLAY = {
    "recompute": "ReComp",
    "fullload": "FullLoad",
    "qa_chunk25": "QA-Chunk25",
    "ours25": "Ours25",
    "mpic32": "MPIC-32 (SSD adaptation)",
    "rekv_chunk25": "ReKV-Chunk25 (adapted)",
}
STORE_METHODS = ("fullload", "ours25", "mpic32", "rekv_chunk25")
RUNNER_EXPORTS = (
    "config.json", "manifest.json", "raw.jsonl", "summary.json",
    "summary.csv", "persistence.json", "store_metadata.json",
    "validation.json", "runtime_fingerprint.json", "run_artifacts.json",
)
BOOTSTRAP_SEED = 680032
BOOTSTRAP_REPLICATES = 10_000
REKV_REQUIRED_SMOKE_CHECKS = {
    "representative_parity", "similarity_parity", "pre_rope_capture",
    "compact_cache", "rope_mask_position", "multi_layer_dependency",
    "retrieval_answer_handoff", "request_isolation",
    "all_block_diagnostic", "actual_model_smoke",
    "duplicate_payload_read",
}


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
    expected_run_parent = (ROOT / "runs/rekv_baseline").resolve()
    expected_result_parent = (ROOT / "results/rekv_baseline").resolve()
    _require(run.parent == expected_run_parent,
             "run directory is outside runs/rekv_baseline")
    _require(result.parent == expected_result_parent,
             "results directory is outside results/rekv_baseline")
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
             and int(completed.get("completed", -1)) == 1440,
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
             "six-arm method contract mismatch")
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
        "artifact_protector": ROOT / "scripts/69_protect_rekv_artifacts.py",
        "pilot_runner": ROOT / "scripts/70_eval_rekv_gqa.py",
        "pilot_reporter": ROOT / "scripts/71_report_rekv_gqa.py",
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
        "rekv_paper_markdown": ROOT / "papers/rekv.md",
        "rekv_paper_pdf": ROOT / "papers/rekv.pdf",
        "rekv_contract": ROOT / "docs/rekv_baseline_contract.md",
        "rekv_source_review": ROOT / "docs/rekv_source_review.md",
        "rekv_implementation": ROOT / "mmimpress/rekv.py",
        "rekv_store": ROOT / "mmimpress/rekv_store.py",
        "rekv_smoke_validator": ROOT / "scripts/72_smoke_rekv.py",
    }
    _require(set(source) == set(current_sources),
             "runner source-hash inventory mismatch")
    for key, path in current_sources.items():
        _require(source[key] == sha256_file(path),
                 f"source changed after pilot began: {key}")


def _validate_rows(rows: Sequence[Mapping[str, Any]], run_id: str) \
        -> dict[str, Any]:
    _require(len(rows) == 1440, f"expected 1440 rows, found {len(rows)}")
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
        _require(len(order) == 6 and set(order) == set(METHOD_KEYS)
                 and 0 <= order_position < 6
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
        generated = list(_sequence(row.get("generated_token_ids"),
                                   f"{prefix}.generated_token_ids"))
        _require(bool(generated)
                 and int(row.get("generated_token_count", -1)) == len(generated)
                 and int(row.get("first_token_id", -1)) == int(generated[0]),
                 f"{prefix} generated-token accounting mismatch")
        # MPIC's first token comes from its manual selective prefill. When
        # it is EOS, decode invokes no top-level model forward at all.
        minimum_forwards = 0 if method == "mpic32" and turn > 1 else 1
        _require(int(row.get("model_forward_count", -1)) >= minimum_forwards,
                 f"{prefix} lacks measured model-forward count")
        if method == "rekv_chunk25" and turn > 1:
            _require(int(row["model_forward_count"]) >= 2,
                     f"{prefix} omitted retrieval or answer forward")
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
            expected_path = (
                "ssd_cache_hit_mpic" if method == "mpic32" else
                "ssd_cache_hit_rekv" if method == "rekv_chunk25" else
                "ssd_cache_hit")
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
        _require(len(image_rows) == 36, f"image {image} does not have 36 rows")
        _require(len({tuple(row["method_order"]) for row in image_rows}) == 1,
                 f"image {image} changed method order across turns")
        for turn in range(1, 7):
            subset = [row for row in image_rows if int(row["turn_id"]) == turn]
            _require(len(subset) == 6
                     and {row["method_key"] for row in subset} == set(METHOD_KEYS)
                     and len({str(row["question_id"]) for row in subset}) == 1,
                     f"image {image} turn {turn} is not a complete six-arm pair")
            _require(len({tuple(row["method_order"]) for row in subset}) == 1,
                     f"image {image} turn {turn} changed method order")
    _require(len(by_prompt) == 240, "expected 240 paired prompts")
    for pair, prompt_rows in by_prompt.items():
        _require(len(prompt_rows) == 6
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
    _require(len(turn1) == 240, "expected 240 Turn-1 arm requests")

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

    rekv = [row for row in by_method["rekv_chunk25"]
            if int(row["turn_id"]) > 1]
    _require(len(rekv) == 200, "ReKV hit coverage mismatch")
    for row in rekv:
        identity = str(row["request_id"])
        selected = list(_sequence(row.get("selected_chunk_ids_per_layer"),
                                  f"{identity}.selected_chunk_ids_per_layer"))
        compact = list(_sequence(row.get("compact_key_lengths_per_layer"),
                                 f"{identity}.compact_key_lengths_per_layer"))
        mapping = list(_sequence(row.get("source_to_compact_positions_per_layer"),
                                 f"{identity}.source_to_compact_positions_per_layer"))
        selected_tokens = list(_sequence(
            row.get("selected_valid_spatial_tokens_per_layer"),
            f"{identity}.selected_valid_spatial_tokens_per_layer"))
        _require(len(selected) == len(compact) == len(mapping)
                 == len(selected_tokens) == 32,
                 f"{identity} lacks 32-layer compact retrieval trace")
        n_physical = int(row.get("normal_physical_chunk_count", 0))
        n_candidates = int(row.get("normal_candidate_chunk_count", 0))
        budget = max(1, min(n_physical, int(round(n_physical * 0.25))))
        _require(n_physical > 0 and 0 < n_candidates <= n_physical
                 and int(row.get("normal_selected_chunk_count", -1)) == budget
                 and all(len(ids) == budget and list(ids) == sorted(set(ids))
                         for ids in selected),
                 f"{identity} ReKV normal-chunk budget/order mismatch")
        _require(all(int(length) > 0 for length in compact)
                 and all(int(count) > 0 for count in selected_tokens),
                 f"{identity} invalid compact lengths or spatial tokens")
        _require(_finite(row.get("peak_cpu_pinned_bytes"),
                         f"{identity}.peak_cpu_pinned_bytes",
                         nonnegative=True) >= 0,
                 f"{identity} invalid pinned staging peak")
        _require(_finite(row.get("process_peak_rss_bytes"),
                         f"{identity}.process_peak_rss_bytes",
                         nonnegative=True) > 0,
                 f"{identity} invalid process RSS highwater")
        _require(row.get("similarity_mode") == "official_code_dot"
                 and int(row.get("stage_b_payload_read_bytes", -1)) == 0
                 and 0 < int(row.get("stage_a_payload_read_bytes", -1))
                    <= int(row.get("ssd_total_bytes", -1))
                 and int(row.get("ssd_read_bytes", -1))
                    == int(row.get("ssd_total_bytes", -2))
                 and int(row.get("ssd_preads", -1))
                    == int(row.get("pread_count", -2)),
                 f"{identity} ReKV SSD or similarity contract mismatch")
        ids = list(_sequence(row.get("generated_token_ids"),
                             f"{identity}.generated_token_ids"))
        _require(bool(ids) and len(ids) == int(row.get("generated_token_count", -1))
                 and int(ids[0]) == int(row["first_token_id"]),
                 f"{identity} generated token trace mismatch")

    failures_path = ROOT / "runs/rekv_baseline" / run_id / "failures.jsonl"
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
            "correct_all": sum(int(row["correct"]) for row in all_rows),
            "correct_hit": sum(int(row["correct"]) for row in hits),
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
    _require(int(value.get("observed_requests", -1)) == 1440
             and int(value.get("expected_requests", -1)) == 1440,
             "runner validation coverage mismatch")
    _require(int(value.get("unresolved_failures", -1)) == 0,
             "runner reports unresolved failures")
    _require(value.get("validation_scope")
             == "measurement_and_rekv_correctness_only"
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
    # The historical MPIC smoke predates the output-only token-ID telemetry
    # added to serve.py for this six-arm run. Keep that original evidence
    # immutable; validate all unchanged sources and disclose the one drift.
    historical_serve_sha256 = (
        "116ed0d21f3bdce65e3de663cd794bdfc628014f0d01d9766dbe7046c2f953d7")
    _require(recorded_sources["mmimpress/serve.py"] == historical_serve_sha256,
             "historical MPIC smoke does not reference the reviewed server")
    for name in smoke_source_paths:
        if name != "mmimpress/serve.py":
            _require(recorded_sources[name] == sha256_file(ROOT / name),
                     f"smoke source changed after validation: {name}")
    current_serve_sha256 = sha256_file(ROOT / "mmimpress/serve.py")
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
        "historical_serve_sha256": historical_serve_sha256,
        "current_serve_sha256": current_serve_sha256,
        "serve_source_changed_for_output_telemetry":
            current_serve_sha256 != historical_serve_sha256,
        "hashes": {
            name: sha256_file(smoke_dir / name) for name in (
                "validation.json", "correctness_tests.json",
                "position_diagnostics.json", "persistence.json",
            )
        },
    }


def _validate_rekv_smoke(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    checks = _mapping(value.get("checks"), "ReKV smoke checks")
    missing = REKV_REQUIRED_SMOKE_CHECKS - set(checks)
    _require(value.get("passed") is True
             and int(value.get("n_images", -1)) >= 3
             and not missing
             and all(passed is True for passed in checks.values()),
             f"ReKV smoke/parity gate failed or incomplete: {sorted(missing)}")
    smoke_dir = path.parent
    names = ("validation.json", "parity_tests.json",
             "cache_handoff_validation.json", "position_validation.json")
    hashes = {name: sha256_file(smoke_dir / name) for name in names}
    for name in names[1:]:
        document = _read_json(smoke_dir / name)
        _require(document.get("schema_version")
                 == "rekv-real-model-smoke-v1"
                 and document.get("passed") is True
                 and len(_sequence(document.get("images"), name)) == 3,
                 f"ReKV smoke detail failed: {name}")
    smoke_config = _read_json(smoke_dir / "config.json")
    source_files = _mapping(
        smoke_config.get("source_file_sha256"), "ReKV smoke source hashes")
    _require(len(source_files) >= 7,
             "ReKV smoke source inventory is incomplete")
    for relative_path, digest in source_files.items():
        _require(digest == sha256_file(ROOT / relative_path),
                 f"ReKV smoke source changed: {relative_path}")
    _require(value.get("schema_version") == "rekv-real-model-smoke-v1"
             and value.get("unit_gates_passed") is True
             and int(value.get("images_expected", -1)) == 3
             and len(value.get("image_artifacts", [])) == 3,
             "ReKV smoke validation scope mismatch")
    for name in value["image_artifacts"]:
        _require(Path(name).name == name,
                 "ReKV smoke image artifact escaped its directory")
        image = _read_json(smoke_dir / name)
        checks = _mapping(image.get("checks"), f"ReKV smoke {name} checks")
        _require(image.get("schema_version") == "rekv-real-model-smoke-v1"
                 and REKV_REQUIRED_SMOKE_CHECKS <= set(checks)
                 and all(checks[key] is True for key in checks),
                 f"ReKV smoke image artifact failed: {name}")
        hashes[name] = sha256_file(smoke_dir / name)
    hashes["config.json"] = sha256_file(smoke_dir / "config.json")
    return {"validation": value, "hashes": hashes, "dir": smoke_dir,
            "config": smoke_config}


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
    _require(set(manifest.get("scope_roots", ())) >= {
        "runs", "results", "kvstore", "kvstore_image_only_visionzip",
        "kvstore_image_only_work", "kvstore_reorder_prefix_calib1"},
        "source stores were omitted from artifact protection")
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
            if method == "rekv_chunk25":
                meta = _mapping(evidence.get("meta"), "ReKV persisted meta")
                _require(meta.get("key_representation") == "pre_rope_k_projection"
                         and meta.get("representative_dtype") == "bfloat16"
                         and int(byte_counts.get("representatives", 0)) > 0
                         and int(byte_counts.get("initial_kv", 0)) > 0
                         and int(byte_counts.get("separator", 0)) > 0,
                         f"ReKV store representation/bytes mismatch: {image}")
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


def _validate_rekv_image_integrity(run_dir: Path,
                                   expected_images: set[str]) -> list[dict[str, Any]]:
    root = run_dir / "rekv_image_integrity"
    _require(root.is_dir() and not root.is_symlink(),
             "ReKV image-boundary integrity directory is missing")
    paths = sorted(root.glob("*.json"))
    _require(len(paths) == 40, "expected 40 ReKV image integrity documents")
    documents = [_read_json(path) for path in paths]
    observed = {str(item.get("image_id")) for item in documents}
    _require(observed == expected_images and len(observed) == len(documents),
             "ReKV image integrity coverage mismatch")
    for item in documents:
        _require(item.get("schema_version") == RUN_SCHEMA
                 and item.get("unchanged") is True
                 and item.get("stored_at_context_open_sha256")
                 == item.get("live_after_all_hits_sha256")
                 and item.get("boundary")
                 == "after_all_5_rekv_hits_before_context_close",
                 f"ReKV image-boundary integrity failed: {item.get('image_id')}")
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
        "rekv_chunk25": "owned canonical pre-RoPE KV + BF16 representatives",
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
            "mean_capture_materialize_ms": (
                mean_time("capture_materialize_ms") if owns_store else 0.0),
            "mean_representative_build_ms": (
                mean_time("representative_build_ms") if owns_store else 0.0),
            "mean_store_write_ms": (
                mean_time("store_write_ms") if owns_store else 0.0),
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
        "difference_direction": "ReKV_minus_reference",
        "bootstrap_unit": "image (all questions in sampled image retained)",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "seed_base": BOOTSTRAP_SEED,
        "comparisons": [],
    }
    rekv_rows = [row for row in rows if row["method_key"] == "rekv_chunk25"]
    for ref_index, reference_key in enumerate(
            ("ours25", "qa_chunk25", "fullload")):
        all_quality: dict[str, list[float]] = defaultdict(list)
        hit_quality: dict[str, list[float]] = defaultdict(list)
        counts_all = {"both_correct": 0, "rekv_only_correct": 0,
                      "reference_only_correct": 0, "both_wrong": 0}
        counts_hit = dict.fromkeys(counts_all, 0)
        hit_rekv_ttft: list[float] = []
        hit_ref_ttft: list[float] = []
        first_token_agreement = prediction_agreement = 0
        for candidate in rekv_rows:
            image, question = (str(candidate["image_id"]),
                               str(candidate["question_id"]))
            reference = indexed[(image, question, reference_key)]
            c, r = bool(candidate["correct"]), bool(reference["correct"])
            label = ("both_correct" if c and r else
                     "rekv_only_correct" if c else
                     "reference_only_correct" if r else "both_wrong")
            counts_all[label] += 1
            delta = float(candidate["correct"]) - float(reference["correct"])
            all_quality[image].append(delta)
            if int(candidate["turn_id"]) > 1:
                counts_hit[label] += 1
                hit_quality[image].append(delta)
                hit_rekv_ttft.append(float(candidate["ttft_ms"]))
                hit_ref_ttft.append(float(reference["ttft_ms"]))
            first_token_agreement += int(
                int(candidate["first_token_id"])
                == int(reference["first_token_id"]))
            prediction_agreement += int(
                str(candidate["prediction"]) == str(reference["prediction"]))
        _require(sum(counts_all.values()) == 240
                 and sum(counts_hit.values()) == 200,
                 "paired ReKV quality coverage mismatch")
        rekv_mean = sum(hit_rekv_ttft) / len(hit_rekv_ttft)
        ref_mean = sum(hit_ref_ttft) / len(hit_ref_ttft)
        output["comparisons"].append({
            "candidate_method_key": "rekv_chunk25",
            "reference_method_key": reference_key,
            "candidate": DISPLAY["rekv_chunk25"],
            "reference": DISPLAY[reference_key],
            "quality_all": _cluster_bootstrap(
                all_quality, seed=BOOTSTRAP_SEED + ref_index * 2),
            "quality_hit": _cluster_bootstrap(
                hit_quality, seed=BOOTSTRAP_SEED + ref_index * 2 + 1),
            "contingency_all": counts_all,
            "contingency_hit": counts_hit,
            "rekv_hit_ttft_mean_ms": rekv_mean,
            "reference_hit_ttft_mean_ms": ref_mean,
            "hit_ttft_mean_difference_ms": rekv_mean - ref_mean,
            "hit_ttft_relative_reduction_vs_reference":
                (ref_mean - rekv_mean) / ref_mean,
            "hit_ttft_speedup_reference_over_rekv": ref_mean / rekv_mean,
            "first_token_agreement_all": first_token_agreement / 240,
            "prediction_agreement_all": prediction_agreement / 240,
        })
    return output


def _selection_analysis(
    rows_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    rekv = [row for row in rows_by_method["rekv_chunk25"]
            if int(row["turn_id"]) > 1]
    qa = {(str(row["image_id"]), str(row["question_id"])): row
          for row in rows_by_method["qa_chunk25"]
          if int(row["turn_id"]) > 1}
    ours = [row for row in rows_by_method["ours25"]
            if int(row["turn_id"]) > 1]
    by_image: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    ours_by_image: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rekv:
        by_image[str(row["image_id"])].append(row)
    for row in ours:
        ours_by_image[str(row["image_id"])].append(row)
    _require(len(by_image) == len(ours_by_image) == 40,
             "selection analysis image coverage mismatch")
    jaccard_pairs: list[float] = []
    jaccard_consecutive: list[float] = []
    qa_overlap: list[float] = []
    qa_question_pairs: list[float] = []
    ours_question_pairs: list[float] = []
    identical_layers = 0
    total_layers = 0
    whole_pair_identical = 0
    whole_pairs = 0
    per_layer_varied_images = [0] * 32
    per_image: list[dict[str, Any]] = []

    def jac(left, right):
        a, b = set(left), set(right)
        return len(a & b) / len(a | b) if a or b else 1.0

    for image in sorted(by_image):
        entries = sorted(by_image[image], key=lambda row: int(row["turn_id"]))
        _require(len(entries) == 5
                 and [int(row["turn_id"]) for row in entries] == list(range(2, 7)),
                 f"ReKV selection lacks Q2-Q6 for {image}")
        selection = [row["selected_chunk_ids_per_layer"] for row in entries]
        varied = [len({tuple(item[layer]) for item in selection}) > 1
                  for layer in range(32)]
        per_layer_varied_images = [
            count + int(flag)
            for count, flag in zip(per_layer_varied_images, varied)]
        for left, right in combinations(range(5), 2):
            whole_pairs += 1
            whole_pair_identical += int(selection[left] == selection[right])
            for layer in range(32):
                a, b = selection[left][layer], selection[right][layer]
                jaccard_pairs.append(jac(a, b))
                identical_layers += int(a == b)
                total_layers += 1
                if right == left + 1:
                    jaccard_consecutive.append(jac(a, b))
        for row in entries:
            other = qa[(image, str(row["question_id"]))]
            qa_ids = other["selected_chunk_ids_per_layer"]
            _require(len(qa_ids) == 32,
                     f"QA selection layer count mismatch for {image}")
            for layer in range(32):
                qa_overlap.append(jac(
                    row["selected_chunk_ids_per_layer"][layer], qa_ids[layer]))
        ours_entries = sorted(ours_by_image[image],
                              key=lambda row: int(row["turn_id"]))
        _require(len(ours_entries) == 5,
                 f"Ours selection lacks Q2-Q6 for {image}")
        qa_entries = [qa[(image, str(row["question_id"]))] for row in entries]
        for left, right in combinations(range(5), 2):
            for layer in range(32):
                qa_question_pairs.append(jac(
                    qa_entries[left]["selected_chunk_ids_per_layer"][layer],
                    qa_entries[right]["selected_chunk_ids_per_layer"][layer]))
                ours_question_pairs.append(jac(
                    ours_entries[left]["selected_chunk_ids_per_layer"][layer],
                    ours_entries[right]["selected_chunk_ids_per_layer"][layer]))
        ours_invariant = all(
            row["selected_chunk_ids_per_layer"]
            == ours_entries[0]["selected_chunk_ids_per_layer"]
            for row in ours_entries)
        _require(ours_invariant, f"Ours prefix selection varied in image {image}")
        per_image.append({
            "image_id": image,
            "rekv_questions": [str(row["question_id"]) for row in entries],
            "rekv_varied_layers": [i for i, flag in enumerate(varied) if flag],
            "ours_prefix_invariant": ours_invariant,
        })
    return {
        "schema_version": REPORT_SCHEMA,
        "scope": "40 images; each image Q2-Q6; 32 layers",
        "question_pair_chunk_jaccard": sum(jaccard_pairs) / len(jaccard_pairs),
        "consecutive_query_chunk_jaccard":
            sum(jaccard_consecutive) / len(jaccard_consecutive),
        "identical_selection_rate_layer_pair": identical_layers / total_layers,
        "identical_selection_rate_all_layers_pair":
            whole_pair_identical / whole_pairs,
        "per_layer_varied_image_counts": per_layer_varied_images,
        "rekv_vs_qa_chunk_jaccard": sum(qa_overlap) / len(qa_overlap),
        "qa_question_pair_chunk_jaccard":
            sum(qa_question_pairs) / len(qa_question_pairs),
        "ours_question_pair_chunk_jaccard":
            sum(ours_question_pairs) / len(ours_question_pairs),
        "ours_prefix_invariance_all_images": True,
        "per_image": per_image,
    }


def _latency_rows(rows_by_method: Mapping[str, Sequence[Mapping[str, Any]]]) \
        -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    fields = (
        "request_e2e_ms", "decode_ms", "prompt_build_ms", "tokenization_ms",
        "processor_total_ms", "input_prepare_ms", "input_h2d_ms",
        "ssd_read_ms", "kv_read_ms", "embedding_read_ms", "h2d_ms",
        "cache_assembly_ms", "position_processing_ms",
        "selective_prefill_interval_ms", "prefill_ms",
        "request_prep_ms", "retrieval_forward_wall_ms", "q_rep_ms",
        "similarity_ms", "topk_ms", "selection_d2h_ms",
        "io_planning_ms", "ssd_read_pipeline_ms",
        "compact_assembly_ms", "rope_ms",
        "answer_prefill_wall_ms", "attention_host_wall_ms",
        "online_selector_total_ms", "selector_decision_host_wall_ms",
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
            "ReKV retrieval_forward_wall_ms encloses selected KV I/O, "
            "attention, and MLP; ssd_read_ms is IOCounter pread time, "
            "ssd_read_pipeline_ms is broader, h2d_ms is host staging/async "
            "submit; subcomponents overlap and are not additive"
            if method == "rekv_chunk25" else
            "phase fields follow the existing server instrumentation")
        output.append(row)
    return output


def _io_rows(rows_by_method: Mapping[str, Sequence[Mapping[str, Any]]]) \
        -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    def run_counts(rows):
        counts = []
        lengths = []
        for row in rows:
            for ids in row.get("selected_chunk_ids_per_layer") or []:
                ordered = sorted(set(int(value) for value in ids))
                runs = sum(i == 0 or value != ordered[i-1] + 1
                           for i, value in enumerate(ordered))
                if runs:
                    counts.append(runs)
                    lengths.append(len(ordered) / runs)
        return (sum(counts)/len(counts), sum(lengths)/len(lengths)) if counts else (None, None)
    for method in METHOD_KEYS:
        hits = [row for row in rows_by_method[method]
                if int(row["turn_id"]) > 1]
        bytes_mean = _mean(hits, "ssd_read_bytes") or 0.0
        layout_runs, layout_run_length = run_counts(hits)
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
            "normal_selected_chunk_ratio_mean": _mean(
                hits, "selected_normal_chunk_ratio"),
            "selected_valid_spatial_token_ratio_mean": (
                sum(float(v) for row in hits
                    for v in row.get("retained_spatial_token_ratios_per_layer", []))
                / sum(len(row.get("retained_spatial_token_ratios_per_layer", []))
                      for row in hits)
                if method == "rekv_chunk25" else None),
            "stage_a_payload_bytes_mean": _mean(hits, "stage_a_payload_read_bytes"),
            "stage_b_payload_bytes_mean": _mean(hits, "stage_b_payload_read_bytes"),
            "request_time_metadata_bytes_mean": _mean(
                hits, "request_time_metadata_bytes"),
            "selected_payload_bytes_mean": _mean(hits, "selected_payload_bytes"),
            "duplicate_read_bytes_mean": _mean(hits, "duplicate_read_bytes"),
            "contiguous_runs_per_layer_mean": (
                sum(int(value) for row in hits
                    for value in row.get("contiguous_runs_per_layer", []))
                / sum(len(row.get("contiguous_runs_per_layer", []))
                      for row in hits)
                if method == "rekv_chunk25" else None),
            "mean_run_length_chunks": (
                _mean(hits, "mean_run_length_chunks")
                if method == "rekv_chunk25" else layout_run_length),
            "layout_contiguous_runs_per_layer_mean": layout_runs,
            "visual_kv_scope_note": (
                "zero Visual-KV SSD reads; ordinary image/model file I/O is not claimed zero"
                if method == "recompute" else "measured SSD cache payload"),
        })
    return output


def _memory_analysis(
    run_dir: Path, rows_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
    persistence_grouped: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    root = run_dir / "metadata_activation"
    _require(root.is_dir() and not root.is_symlink(),
             "ReKV metadata activation evidence missing")
    documents = [_read_json(path) for path in sorted(root.glob("*.json"))]
    _require(len(documents) == 40
             and len({str(row.get("image_id")) for row in documents}) == 40,
             "ReKV metadata activation lacks 40 images")
    rekv_hits = [row for row in rows_by_method["rekv_chunk25"]
                 if int(row["turn_id"]) > 1]
    gpu_bytes = [int(row["metadata_gpu_bytes_total"])
                 for row in documents]
    cpu_bytes = [int(row["representative_metadata_cpu_bytes"])
                 for row in documents]
    activation = [float(row["metadata_activation_ms"])
                  for row in documents]
    _require(all(value > 0 for value in gpu_bytes)
             and all(value >= 0 for value in cpu_bytes + activation),
             "invalid metadata activation measurement")
    persistence = list(persistence_grouped["rekv_chunk25"])
    build = [float(item["timing_ms"]["representative_build_ms"])
             for item in persistence]
    return {
        "schema_version": REPORT_SCHEMA,
        "scope": "40 active-image contexts; ReKV Q2-Q6 GPU peaks",
        "metadata_dtype": "bfloat16",
        "metadata_residency": "active image GPU",
        "metadata_gpu_bytes_per_image_mean": sum(gpu_bytes) / len(gpu_bytes),
        "representative_metadata_gpu_bytes_per_image_mean": sum(
            int(row["representative_metadata_gpu_bytes"]) for row in documents)
            / len(documents),
        "valid_counts_metadata_gpu_bytes_per_image_mean": sum(
            int(row["valid_counts_metadata_gpu_bytes"]) for row in documents)
            / len(documents),
        "metadata_cpu_bytes_per_image_mean": sum(cpu_bytes) / len(cpu_bytes),
        "metadata_activation_ms_per_image_mean":
            sum(activation) / len(activation),
        "initial_context_activation_ms_per_image_mean": sum(
            float(row["initial_context_activation_ms"]) for row in documents)
            / len(documents),
        "activation_total_ms_per_image_mean": sum(
            float(row["activation_total_ms"]) for row in documents)
            / len(documents),
        "initial_context_gpu_bytes_per_image_mean": sum(
            int(row["initial_context_gpu_bytes"]) for row in documents)
            / len(documents),
        "initial_context_cpu_bytes_per_image_mean": sum(
            int(row["initial_context_cpu_bytes"]) for row in documents)
            / len(documents),
        "metadata_build_ms_per_image_mean": sum(build) / len(build),
        "metadata_100_images_gpu_bytes_arithmetic":
            100 * sum(gpu_bytes) / len(gpu_bytes),
        "metadata_100_images_measurement_kind": "arithmetic extrapolation; not measured footprint",
        "compact_retrieved_kv_bytes_per_hit_mean": _mean(
            rekv_hits, "compact_retrieved_kv_bytes"),
        "peak_gpu_allocated_bytes_per_hit_mean": _mean(
            rekv_hits, "peak_gpu_allocated_bytes"),
        "peak_gpu_allocated_bytes_per_hit_max": max(
            int(row["peak_gpu_allocated_bytes"]) for row in rekv_hits),
        "peak_gpu_reserved_bytes_per_hit_mean": _mean(
            rekv_hits, "peak_gpu_reserved_bytes"),
        "incremental_peak_gpu_allocated_bytes_per_hit_mean": _mean(
            rekv_hits, "incremental_peak_gpu_allocated_bytes"),
        "peak_cpu_pinned_bytes_per_hit_mean": _mean(
            rekv_hits, "peak_cpu_pinned_bytes"),
        "peak_cpu_pinned_bytes_per_hit_max": max(
            int(row["peak_cpu_pinned_bytes"]) for row in rekv_hits),
        "peak_cpu_pinned_bytes_note": (
            "measured request-local pinned host staging; excludes other "
            "pinned allocations"),
        "process_peak_rss_bytes_mean": _mean(
            rekv_hits, "process_peak_rss_bytes"),
        "process_peak_rss_bytes_max": max(
            int(row["process_peak_rss_bytes"]) for row in rekv_hits),
        "process_peak_rss_note": (
            "Linux process-wide lifetime highwater sampled per hit; "
            "not attributable to an individual request"),
        "metadata_activation": documents,
    }


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
    return (ROOT / "docs/rekv_source_review.md").read_text(encoding="utf-8")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def _analysis(
    summaries: Mapping[str, Mapping[str, Any]],
    comparisons: Mapping[str, Any], latency: Sequence[Mapping[str, Any]],
    io_rows: Sequence[Mapping[str, Any]], persistence: Sequence[Mapping[str, Any]],
    row_evidence: Mapping[str, Any], protection: Mapping[str, Any],
    mpic_smoke: Mapping[str, Any], rekv_smoke: Mapping[str, Any],
    selection: Mapping[str, Any], memory: Mapping[str, Any],
    validation: Mapping[str, Any], config: Mapping[str, Any],
) -> str:
    lat = {row["method_key"]: row for row in latency}
    io = {row["method_key"]: row for row in io_rows}
    provision = {row["method_key"]: row for row in persistence}
    rekv_hits = [row for row in row_evidence["by_method"]["rekv_chunk25"]
                 if int(row["turn_id"]) > 1]
    compact_mean = sum(
        int(length) for row in rekv_hits
        for length in row["compact_key_lengths_per_layer"]) / (len(rekv_hits) * 32)
    online_decision = sum(
        float(lat["rekv_chunk25"].get(f"{field}_mean") or 0.0)
        for field in ("q_rep_ms", "similarity_ms", "topk_ms",
                      "selection_d2h_ms", "io_planning_ms"))
    qa_decision = lat["qa_chunk25"].get(
        "selector_decision_host_wall_ms_mean")
    lines = [
        "# ReKV-Chunk25 (adapted): same-run GQA pilot",
        "",
        "| Method | Acc all | Acc hit | TTFT mean | p50 | p95 | SSD MB/hit | Preads/hit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_KEYS:
        row = summaries[method]
        lines.append(
            f"| {DISPLAY[method]} | {row['correct_all']}/240 "
            f"({_fmt(row['accuracy_all']*100, 2)}%) | "
            f"{row['correct_hit']}/200 ({_fmt(row['accuracy_hit']*100, 2)}%) | "
            f"{_fmt(row['ttft_mean_ms'], 2)} ms | "
            f"{_fmt(row['ttft_p50_ms'], 2)} ms | "
            f"{_fmt(row['ttft_p95_ms'], 2)} ms | "
            f"{_fmt(row['ssd_mb_per_hit'], 3)} | "
            f"{_fmt(row['ssd_preads_per_hit'], 2)} |")

    detail = [
        ("Acc hit", *(
            f"{summaries[m]['correct_hit']}/200"
            for m in ("rekv_chunk25", "qa_chunk25", "ours25"))),
        ("TTFT mean (ms)", *(
            _fmt(summaries[m]["ttft_mean_ms"], 2)
            for m in ("rekv_chunk25", "qa_chunk25", "ours25"))),
        ("Online decision cost (ms)", _fmt(online_decision, 2),
         _fmt(qa_decision, 2), _fmt(lat["ours25"].get(
             "online_selector_total_ms_mean"), 2)),
        ("Retrieval-forward wall (ms)", _fmt(lat["rekv_chunk25"].get(
            "retrieval_forward_wall_ms_mean"), 2), "N/A", "N/A"),
        ("Answer-prefill wall (ms)", _fmt(lat["rekv_chunk25"].get(
            "answer_prefill_wall_ms_mean"), 2), "N/A", "N/A"),
        ("SSD bytes/hit", *(
            _fmt(io[m]["ssd_total_bytes_mean"], 0)
            for m in ("rekv_chunk25", "qa_chunk25", "ours25"))),
        ("Preads/hit", *(
            _fmt(io[m]["preads_mean"], 2)
            for m in ("rekv_chunk25", "qa_chunk25", "ours25"))),
        ("Selected-layout runs/layer", *(
            _fmt(io[m]["layout_contiguous_runs_per_layer_mean"], 2)
            for m in ("rekv_chunk25", "qa_chunk25", "ours25"))),
        ("Metadata bytes/image", _fmt(
            memory["metadata_gpu_bytes_per_image_mean"], 0), "N/A", "N/A"),
        ("Actual attention key length", _fmt(compact_mean, 2), "N/A", "N/A"),
        ("Question-pair chunk Jaccard",
         _fmt(selection["question_pair_chunk_jaccard"], 3),
         _fmt(selection["qa_question_pair_chunk_jaccard"], 3),
         _fmt(selection["ours_question_pair_chunk_jaccard"], 3)),
    ]
    lines += [
        "", "## ReKV, QA-Chunk25, and Ours25", "",
        "| Metric | ReKV-Chunk25 | QA-Chunk25 | Ours25 |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(f"| {name} | {rekv} | {qa} | {ours} |"
                 for name, rekv, qa, ours in detail)
    lines += [
        "", "The ReKV online decision cost above sums the measured Q "
        "representative, similarity, Top-k, selected-ID transfer, and I/O "
        "planning intervals. QA's entry is its separately instrumented selector "
        "decision host interval; the two implementations instrument different "
        "work. Retrieval-forward wall encloses ReKV's selected KV I/O, "
        "attention, and MLP, so its subcomponents are not added to TTFT. "
        "Selected-layout runs describe the chosen chunk IDs; they are not "
        "necessarily the same as actual pread calls.",
        "", "## Paired quality and latency", "",
        "Differences are ReKV minus the named reference. Accuracy confidence "
        f"intervals are {BOOTSTRAP_REPLICATES:,} image-cluster percentile "
        "bootstrap replicates, retaining each sampled image's six matched "
        "questions. A confidence interval containing zero does not establish "
        "equivalence.", "",
        "| Reference | Acc all difference [95% CI] | Acc hit difference [95% CI] | ReKV TTFT minus reference | Relative reduction vs reference | Reference/ReKV speedup |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in comparisons["comparisons"]:
        qa, qh = item["quality_all"], item["quality_hit"]
        lines.append(
            f"| {item['reference']} | {_fmt(qa['estimate'], 4)} "
            f"[{_fmt(qa['ci95_low'], 4)}, {_fmt(qa['ci95_high'], 4)}] | "
            f"{_fmt(qh['estimate'], 4)} "
            f"[{_fmt(qh['ci95_low'], 4)}, {_fmt(qh['ci95_high'], 4)}] | "
            f"{_fmt(item['hit_ttft_mean_difference_ms'], 2)} ms | "
            f"{_fmt(100*item['hit_ttft_relative_reduction_vs_reference'], 2)}% | "
            f"{_fmt(item['hit_ttft_speedup_reference_over_rekv'], 3)}× |")
    lines += [
        "", "The paired contingency counts are in `paired_quality.json` "
        "for all 240 and cache-hit 200 matched questions per comparison.",
        "", "## Selection, I/O, and handoff", "",
        f"ReKV selected chunk IDs varied across questions with mean pair "
        f"Jaccard {_fmt(selection['question_pair_chunk_jaccard'], 3)} and "
        f"consecutive-query Jaccard "
        f"{_fmt(selection['consecutive_query_chunk_jaccard'], 3)}. "
        f"Its per-layer selection overlap with QA-Chunk25 was "
        f"{_fmt(selection['rekv_vs_qa_chunk_jaccard'], 3)}. Ours25's "
        "same-image prefix selection was invariant across Q2–Q6. Identical "
        "ReKV selections are valid; selection difference alone cannot explain "
        "correctness.",
        "", f"Across ReKV cache hits, Stage A read "
        f"{_fmt(io['rekv_chunk25']['stage_a_payload_bytes_mean']/1e6, 3)} "
        "MB/request and Stage B read "
        f"{_fmt(io['rekv_chunk25']['stage_b_payload_bytes_mean'], 0)} "
        "visual payload bytes/request. Mean duplicate-read bytes were "
        f"{_fmt(io['rekv_chunk25']['duplicate_read_bytes_mean'], 0)}. "
        "The full selected K/V, separators, compact position mapping, and "
        "per-layer key lengths are recorded in raw rows; metadata remains "
        "ready on GPU before each timed request.",
        "", "ReKV `ssd_read_ms` is IOCounter pread time; "
        "`ssd_read_pipeline_ms` covers broader host reading. `h2d_ms` measures "
        "host staging and asynchronous transfer submission, not isolated DMA "
        "execution. No cross-stream SSD/compute overlap is claimed. "
        "`posix_fadvise(DONTNEED)` is performed before each SSD hit and its "
        "per-file statuses are stored in raw rows; it does not guarantee a "
        "cold SSD controller cache.",
        "", "## Provisioning and memory", "",
        f"The ReKV raw-K store used a mean "
        f"{_fmt(provision['rekv_chunk25']['mean_total_bytes']/1e6, 3)} "
        "MB/image and mean one-time persistence "
        f"{_fmt(provision['rekv_chunk25']['mean_persist_ms'], 2)} ms. "
        "Mean capture materialization, representative build, store write, "
        "and fsync are in `persistence.csv`; storage is run-local and "
        "persisted after the normal Turn-1 answer.",
        "", f"Active-image ReKV retrieval metadata occupied "
        f"{_fmt(memory['metadata_gpu_bytes_per_image_mean']/1e6, 3)} "
        "MB GPU/image, with mean activation "
        f"{_fmt(memory['metadata_activation_ms_per_image_mean'], 2)} ms "
        "outside TTFT. Initial/system K/V occupied an additional "
        f"{_fmt(memory['initial_context_gpu_bytes_per_image_mean']/1e6, 3)} "
        "MB GPU/image. The 100-image metadata figure in "
        "`memory_analysis.json` is arithmetic extrapolation, not a measured "
        "100-image footprint. Request-local peak pinned host staging was "
        f"{_fmt(memory['peak_cpu_pinned_bytes_per_hit_mean']/1e6, 3)} "
        "MB/hit on average. Linux process-wide RSS highwater sampled on "
        "those hits reached "
        f"{_fmt(memory['process_peak_rss_bytes_max']/1e9, 3)} GB maximum; "
        "it is a process lifetime highwater, not request-attributable.",
        "", "## Workload and integrity", "",
        "The six methods were measured in the same new run over the frozen "
        "40-image/240-question GQA slice. The six questions per image are "
        "independent requests without answer history. Every method used "
        "normal Image+Q1 pixel inference; each Turn-1 prompt, pixel/input "
        "hash, prediction, and first output token matched across methods. "
        "Accuracy uses 240 questions per method. TTFT and I/O use Q2–Q6 "
        "(200 requests per method). Logical requests and actual root-model "
        "forward calls are reported separately in `validation.json` and raw "
        "rows; generated token IDs/counts are recorded for every request.",
    ]
    for method in METHOD_KEYS:
        rows = row_evidence["by_method"][method]
        lines.append(
            f"- {DISPLAY[method]}: 240 logical requests; "
            f"{sum(int(row['model_forward_count']) for row in rows)} "
            "observed model forwards.")
    lines += [
        "", "MPIC's selective first-token prefill runs through its manual "
        "decoder path outside the top-level model-forward hook. If its first "
        "token is EOS, zero counted top-level forwards on that hit is valid; "
        "the raw row still records its generated token.",
        "", f"The run completed 1,440 unique requests with "
        f"{row_evidence['failure_events']} durable technical failure "
        f"events, {row_evidence['retry_total']} retries, and "
        f"{row_evidence['duplicates']} duplicates. The before/after guard "
        f"checked {protection['manifest']['entry_count']} pre-existing "
        "runs/results/source-store entries with no missing or changed paths. "
        "Large payload files use the recorded bounded fingerprint policy.",
        "", "## Implementation scope and limitations", "",
        "The official source commit is "
        "`1fd9a3dbf5dbff7f27069ae2f4463674c495e830`. The paper describes "
        "cosine similarity, whereas the pinned official vector-cache path "
        "computes an unnormalized FP32 dot product. This main run uses "
        "`official_code_dot` throughout. Its video frames, video backbone, "
        "and GPU/CPU payload caches are adapted here to canonical 64-token "
        "single-image SSD chunks, the frozen LLaVA-NeXT Vicuna 7B "
        "4-bit NF4/BF16 model, and metadata-ready/payload-cold SSD hits. "
        "The results do not reproduce original StreamingVQA table numbers. "
        "No MT-GQA, MT-VQA-v2, ConvBench, or VisDial full run is included.",
        "", "## Final status", "", "```text",
    ]
    status = validation["status"]
    for label, key in (
            ("IMPLEMENTATION", "implementation"),
            ("SOURCE RETRIEVAL PARITY", "source_retrieval_parity"),
            ("PRE-ROPE / POSITION CORRECTNESS", "pre_rope_position_correctness"),
            ("COMPACT KV ASSEMBLY", "compact_kv_assembly"),
            ("RETRIEVAL-TO-ANSWER HANDOFF", "retrieval_to_answer_handoff"),
            ("DUPLICATE PAYLOAD READ CHECK", "duplicate_payload_read_check"),
            ("ACTUAL-MODEL SMOKE", "actual_model_smoke"),
            ("GQA PILOT", "gqa_pilot"),
            ("ARTIFACT PROTECTION", "artifact_protection"),
            ("ReKV-CHUNK25 SSD ADAPTATION VALIDATED",
             "rekv_chunk25_ssd_adaptation_validated")):
        lines.append(f"{label}: {status[key]}")
    lines += ["```", "",
              "`YES` validates this documented SSD image adaptation and its "
              "measured serving behavior; it does not claim a full reproduction "
              "of the original ReKV streaming system."]
    return "\n".join(lines) + "\n"


def _source_revision(
    run_dir: Path, smoke_dir: Path, runner_hashes: Mapping[str, str],
) -> dict[str, Any]:
    manifest = _read_json(run_dir / "manifest.json")
    source = _mapping(manifest.get("source_sha256"), "manifest.source_sha256")
    return {
        "schema_version": REPORT_SCHEMA,
        "repository": str(ROOT),
        "git_revision": None,
        "git_status": "unavailable: repository .git metadata is empty",
        "official_rekv_repository": "https://github.com/Becomebright/ReKV",
        "official_rekv_commit": "1fd9a3dbf5dbff7f27069ae2f4463674c495e830",
        "primary_reference": "papers/rekv.pdf",
        "primary_reference_sha256": source["rekv_paper_pdf"],
        "source_files_sha256": dict(source),
        "runner_exports_sha256": dict(runner_hashes),
        "smoke_dir": str(smoke_dir.resolve()),
        "run_dir": str(run_dir.resolve()),
    }


def _report_validation(
    runner_validation: Mapping[str, Any], mpic_smoke: Mapping[str, Any],
    rekv_smoke: Mapping[str, Any], protection: Mapping[str, Any],
    row_evidence: Mapping[str, Any],
    mpic_integrity: Sequence[Mapping[str, Any]],
    rekv_integrity: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "runner_validation_passed": runner_validation.get("passed") is True,
        "independent_1440_unique_rows": sum(
            len(value) for value in row_evidence["by_method"].values()) == 1440,
        "exact_six_arm_240_and_200_coverage": all(
            len(rows) == 240 and sum(int(row["turn_id"]) > 1
                                     for row in rows) == 200
            for rows in row_evidence["by_method"].values()),
        "no_duplicate_requests": row_evidence["duplicates"] == 0,
        "no_unresolved_failures": True,
        "existing_mpic_smoke_passed":
            mpic_smoke["validation"].get("passed") is True,
        "rekv_three_image_parity_smoke_passed":
            rekv_smoke["validation"].get("passed") is True,
        "prior_artifacts_and_source_stores_unchanged":
            protection["validation"].get("passed") is True,
        "mpic_payload_unchanged_at_all_40_image_boundaries":
            len(mpic_integrity) == 40 and all(
                item.get("unchanged") is True for item in mpic_integrity),
        "rekv_payload_unchanged_at_all_40_image_boundaries":
            len(rekv_integrity) == 40 and all(
                item.get("unchanged") is True for item in rekv_integrity),
        "ours_prefix_invariant":
            selection.get("ours_prefix_invariance_all_images") is True,
        "publication_policy_no_overwrite_or_identical": True,
    }
    passed = all(checks.values())
    return {
        "schema_version": REPORT_SCHEMA,
        "passed": passed, "checks": checks,
        "status": {
            "implementation": "PASS" if passed else "FAIL",
            "source_retrieval_parity": "PASS" if passed else "FAIL",
            "pre_rope_position_correctness": "PASS" if passed else "FAIL",
            "compact_kv_assembly": "PASS" if passed else "FAIL",
            "retrieval_to_answer_handoff": "PASS" if passed else "FAIL",
            "duplicate_payload_read_check": "PASS" if passed else "FAIL",
            "actual_model_smoke": "PASS" if passed else "FAIL",
            "gqa_pilot": "COMPLETE" if passed else "INCOMPLETE",
            "artifact_protection": "PASS" if passed else "FAIL",
            "rekv_chunk25_ssd_adaptation_validated": "YES" if passed else "NO",
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
        default=ROOT / "runs/mpic_baseline/smoke_final_linked_3",
        help="historical validated MPIC smoke directory")
    parser.add_argument(
        "--rekv-smoke-validation", type=Path, required=True,
        help="validation.json from the final three-image ReKV smoke")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir, results_dir = _validate_roots(args.run_dir, args.results_dir)
    smoke_dir = args.smoke_dir.resolve()
    rekv_smoke_path = args.rekv_smoke_validation.resolve()

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

    mpic_smoke = _validate_smoke(smoke_dir)
    rekv_smoke = _validate_rekv_smoke(rekv_smoke_path)
    for key, path, digest in (
        ("smoke_validation", smoke_dir / "validation.json",
         mpic_smoke["hashes"]["validation.json"]),
        ("rekv_smoke_validation", rekv_smoke_path,
         rekv_smoke["hashes"]["validation.json"]),
    ):
        gate = _mapping(manifest.get(key), f"manifest.{key}")
        _require(gate.get("passed") is True
                 and Path(str(gate.get("path"))).resolve() == path.resolve()
                 and gate.get("sha256") == digest,
                 f"pilot did not gate on the recorded {key} evidence")

    protection = _validate_protection(run_dir, results_dir)
    persistence_grouped = _validate_persistence(
        _read_json(run_dir / "persistence.json"))
    expected_images = set(row_evidence["by_image"])
    mpic_integrity = _validate_mpic_image_integrity(run_dir, expected_images)
    rekv_integrity = _validate_rekv_image_integrity(run_dir, expected_images)
    selection = _selection_analysis(row_evidence["by_method"])
    memory = _memory_analysis(run_dir, row_evidence["by_method"],
                              persistence_grouped)
    persistence_rows = _persistence_rows(persistence_grouped)
    comparisons = _paired_comparisons(rows)
    latency_rows = _latency_rows(row_evidence["by_method"])
    io_rows = _io_rows(row_evidence["by_method"])
    report_validation = _report_validation(
        runner_validation, mpic_smoke, rekv_smoke, protection, row_evidence,
        mpic_integrity, rekv_integrity, selection)
    _require(report_validation["passed"],
             "independent report validation failed")

    summary_fields = (
        "method_key", "method_id", "method", "requests_all", "requests_hit",
        "accuracy_all", "accuracy_hit", "ttft_mean_ms", "ttft_p50_ms",
        "ttft_p95_ms", "ssd_mb_per_hit", "ssd_preads_per_hit",
        "recomputed_image_tokens_mean", "retained_image_context_ratio_mean",
        "request_e2e_mean_ms",
    )
    analysis = _analysis(
        summaries, comparisons, latency_rows, io_rows, persistence_rows,
        row_evidence, protection, mpic_smoke, rekv_smoke, selection, memory,
        report_validation, config)
    source_revision = _source_revision(
        run_dir, rekv_smoke["dir"], runner_hashes)
    source_revision["historical_mpic_smoke"] = {
        "directory": str(smoke_dir),
        "serve_sha256_at_smoke": mpic_smoke["historical_serve_sha256"],
        "serve_sha256_in_six_arm_run": mpic_smoke["current_serve_sha256"],
        "difference": "output-only generated-token telemetry",
    }
    readme = (
        "# ReKV-Chunk25 (adapted): validated six-arm GQA pilot\n\n"
        "`raw.jsonl` and `summary.csv` are same-run runner exports. "
        "`validation.json` is the runner measurement gate; "
        "`report_validation.json` is the independent final gate.\n\n"
        "`ANALYSIS.md` contains the comparisons and limitations. "
        "`parity_tests.json`, `cache_handoff_validation.json`, and "
        "`position_validation.json` preserve the three-image smoke evidence. "
        "`source_revision.json` records source hashes and the historical "
        "MPIC smoke server telemetry difference.\n"
    )
    generated: dict[str, bytes] = {
        "README.md": readme.encode("utf-8"),
        "source_review.md": _source_review().encode("utf-8"),
        "implementation_contract.md": (
            ROOT / "docs/rekv_baseline_contract.md").read_bytes(),
        "environment.txt": _environment_text(config, runtime).encode("utf-8"),
        "git_state.txt": (
            "Git revision unavailable: repository .git metadata is empty.\n"
        ).encode("utf-8"),
        "source_revision.json": _json_bytes(source_revision),
        "mpic_correctness_tests.json": (
            smoke_dir / "correctness_tests.json").read_bytes(),
        "mpic_position_diagnostics.json": (
            smoke_dir / "position_diagnostics.json").read_bytes(),
        "parity_tests.json": (
            rekv_smoke["dir"] / "parity_tests.json").read_bytes(),
        "cache_handoff_validation.json": (
            rekv_smoke["dir"] / "cache_handoff_validation.json").read_bytes(),
        "position_validation.json": (
            rekv_smoke["dir"] / "position_validation.json").read_bytes(),
        "latency_breakdown.csv": _csv_bytes(
            latency_rows, tuple(latency_rows[0])),
        "io_breakdown.csv": _csv_bytes(io_rows, tuple(io_rows[0])),
        "persistence.csv": _csv_bytes(
            persistence_rows, tuple(persistence_rows[0])),
        "paired_quality.json": _json_bytes(comparisons),
        "selection_analysis.json": _json_bytes(selection),
        "memory_analysis.json": _json_bytes(memory),
        "mpic_image_integrity.json": _json_bytes({
            "schema_version": REPORT_SCHEMA, "images": mpic_integrity}),
        "rekv_image_integrity.json": _json_bytes({
            "schema_version": REPORT_SCHEMA, "images": rekv_integrity}),
        "report_validation.json": _json_bytes(report_validation),
        "ANALYSIS.md": analysis.encode("utf-8"),
        "independent_summary.csv": _csv_bytes(
            [summaries[key] for key in METHOD_KEYS], summary_fields),
        "protected_artifacts_before.json": (
            run_dir / "protected_artifacts_before.json").read_bytes(),
        "protected_artifacts_validation.json": (
            run_dir / "protected_artifacts_validation.json").read_bytes(),
    }
    required = (
        "README.md", "source_review.md", "implementation_contract.md",
        "config.json", "environment.txt", "git_state.txt",
        "source_revision.json", "raw.jsonl", "summary.csv",
        "parity_tests.json", "cache_handoff_validation.json",
        "position_validation.json", "memory_analysis.json",
        "selection_analysis.json", "latency_breakdown.csv",
        "io_breakdown.csv", "persistence.csv", "paired_quality.json",
        "validation.json", "report_validation.json", "ANALYSIS.md",
    )
    artifact_names = (*required, "independent_summary.csv",
                      "mpic_correctness_tests.json",
                      "mpic_position_diagnostics.json",
                      "mpic_image_integrity.json",
                      "rekv_image_integrity.json",
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
        print(f"ReKV report failed: {error}", file=sys.stderr)
        raise SystemExit(1)

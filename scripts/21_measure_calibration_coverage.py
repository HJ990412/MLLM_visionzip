"""Measure calibration-importance mass covered by recorded chunk selections.

This is an analysis-only pass over an already completed GQA 40/240 run.  It
recomputes the SparseVLM importance used by ``scripts/02_reorder.py`` from the
first four calibration questions, then measures how much of that mass falls
inside the *recorded* Prefix, VisionZip Static, and Static+Diverse chunk sets.

The script deliberately does not load ``static.pt``.  In particular,
VisionZip ``token_score``/``chunk_score`` are never substituted for calibration
importance, and the recomputed scores are never passed to a serving selector.
The existing reordered store is opened read-only by ``ImageContext``.

Primary coverage excludes LLaVA-NeXT row-separator tokens from numerator and
denominator.  Secondary coverage includes the union of the selected normal
chunks and the separator sidecar, matching ``sep_policy=sidecar`` serving.

Example (after the baseline run has produced ``per_request.csv``)::

    python scripts/21_measure_calibration_coverage.py \
      --run-dir runs/reorder_prefix_baseline
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import io
import json
import math
import os
import re
import shlex
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmimpress.config import (ATTN_IMPL, CHUNK_SIZE, COMPUTE_DTYPE, DATA_DIR,
                              LOAD_4BIT, MODEL_ID, PROJECT_ROOT, STORE_DIR,
                              STORE_DTYPE)
from mmimpress.cvpr25 import budget_chunk_count
from mmimpress.model import LlavaRunner
from mmimpress.serve import BIAS, ImageContext, Server, calibrate_image


CANONICAL_INDEX_SHA256 = (
    "514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a"
)
CANONICAL_EVAL_WORKLOAD_SHA256 = (
    "97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66"
)
CANONICAL_CALIBRATION_WORKLOAD_SHA256 = (
    "992cc89a81a6cadf363b65b58f8f89fabfc0a1a17358559069f8c0b6cabc1e71"
)
CANONICAL_STORE_META_AGGREGATE_SHA256 = (
    "60f072ca15ae8ddc382ebe8250bef395bd2ecd33d6383e6814e08c60470f5ec3"
)
CANONICAL_STORE_CONTENT_AGGREGATE_SHA256 = (
    "e570a6847743a203fc1e2892d736ebe8aa946647cbb0e280f212388da2c09d68"
)
EXPECTED_MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"
EXPECTED_LAYERS = 32
EXPECTED_CHUNK_SIZE = 64
EXPECTED_STORE_DTYPE = "float16"
ALLOWED_RUN_ROOT = (PROJECT_ROOT / "runs" / "reorder_prefix_baseline").resolve()
EXPERIMENT_SPEC = ALLOWED_RUN_ROOT / "experiment_spec.json"
EXPERIMENT_SPEC_SHA256 = (
    "20d2c1a8ab5359bbceb500b3dcfc6ffc2ed804c9bb97a9d355f6431274774cde"
)
LEGACY_PATHS = tuple((PROJECT_ROOT / rel).resolve() for rel in (
    "runs/gqa40_240_true_ttft",
    "runs/gqa40_240_true_ttft_budget_10_15_20",
    "results/ablation_25",
    "results/budget_sweep",
    "results/eval_b25.json",
    "results_reorder.log",
))

FAMILY_PREFIXES = {
    "prefix": "reorder_prefix_chunk",
    "static": "visionzip_static_chunk",
    "static_diverse": "static_diverse_chunk",
}
EXPECTED_SELECTION_MODE = {
    "prefix": "prefix",
    "static": "static",
    "static_diverse": "static_diverse",
}
DISPLAY_NAME = {
    "prefix": "Reorder + Prefix",
    "static": "Reorder + Static",
    "static_diverse": "Reorder + Static+Diverse",
}
REQUIRED_25_METHODS = {
    "fullload", "recompute", "sparsevlm",
    "reorder_prefix_chunk@25", "visionzip_static_chunk@25",
    "diverse_chunk@25", "static_diverse_chunk@25",
}
ALLOWED_MAIN_METHODS = REQUIRED_25_METHODS | {
    "reorder_prefix_chunk@50", "visionzip_static_chunk@50",
    "diverse_chunk@50", "static_diverse_chunk@50",
}
EXPECTED_SELECTOR_NAMES = {
    "sparsevlm", "reorder_prefix_chunk", "visionzip_static_chunk",
    "diverse_chunk", "static_diverse_chunk",
}


class CoverageError(RuntimeError):
    """Fail-closed validation error for the analysis-only coverage pass."""


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    data: bytes
    sha256: str
    size: int
    mtime_ns: int
    device: int
    inode: int

    def public(self) -> dict:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "device": self.device,
            "inode": self.inode,
        }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stat_signature(stat_result) -> tuple[int, int, int, int]:
    return (int(stat_result.st_dev), int(stat_result.st_ino),
            int(stat_result.st_size), int(stat_result.st_mtime_ns))


def _snapshot_file(path: Path) -> FileSnapshot:
    """Read one immutable input once and bind parsing and hashing to its bytes."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise CoverageError(f"input must be a regular non-symlink file: {path}")
    path = path.resolve()
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if _stat_signature(before) != _stat_signature(after) \
            or len(data) != after.st_size:
        raise CoverageError(f"input changed while it was being read: {path}")
    return FileSnapshot(
        path=path, data=data, sha256=_sha256_bytes(data),
        size=int(after.st_size), mtime_ns=int(after.st_mtime_ns),
        device=int(after.st_dev), inode=int(after.st_ino))


def _assert_snapshot_unchanged(snapshot: FileSnapshot) -> None:
    try:
        stat_result = snapshot.path.stat()
    except FileNotFoundError as exc:
        raise CoverageError(f"input disappeared during analysis: {snapshot.path}") \
            from exc
    expected = (snapshot.device, snapshot.inode,
                snapshot.size, snapshot.mtime_ns)
    if _stat_signature(stat_result) != expected:
        raise CoverageError(f"input metadata changed during analysis: {snapshot.path}")
    if _sha256(snapshot.path) != snapshot.sha256:
        raise CoverageError(f"input content changed during analysis: {snapshot.path}")


def _paths_overlap(a: Path, b: Path) -> bool:
    a, b = a.resolve(), b.resolve()
    return a == b or a in b.parents or b in a.parents


def _validate_paths(args) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    run_raw, store_raw, index_raw = (Path(args.run_dir), Path(args.store),
                                     Path(args.index))
    for label, path in (("run directory", run_raw), ("store", store_raw),
                        ("index", index_raw)):
        if path.is_symlink():
            raise CoverageError(f"refusing symlink {label}: {path}")
    run_dir = run_raw.resolve()
    store = store_raw.resolve()
    index_path = index_raw.resolve()
    if not run_dir.is_dir():
        raise CoverageError(f"completed run directory does not exist: {run_dir}")
    if run_dir == ALLOWED_RUN_ROOT or ALLOWED_RUN_ROOT not in run_dir.parents:
        raise CoverageError(
            f"--run-dir must be a child of {ALLOWED_RUN_ROOT}, got {run_dir}")
    if store != STORE_DIR.resolve():
        raise CoverageError(
            f"coverage is pinned to the canonical store {STORE_DIR.resolve()}, "
            f"got {store}")
    if _paths_overlap(run_dir, store):
        raise CoverageError("run/output directory must be disjoint from KV store")
    for legacy in LEGACY_PATHS:
        if _paths_overlap(run_dir, legacy):
            raise CoverageError(
                f"run/output directory overlaps preserved legacy path: {legacy}")
    expected_trace_raw = run_dir / "per_request.csv"
    if expected_trace_raw.is_symlink():
        raise CoverageError(f"refusing symlink selection trace: {expected_trace_raw}")
    expected_trace = expected_trace_raw.resolve()
    trace_path = (Path(args.trace).resolve() if args.trace
                  else expected_trace)
    if trace_path != expected_trace:
        raise CoverageError(
            "selection trace must be RUN_DIR/per_request.csv so results and "
            f"trace provenance cannot diverge; got {trace_path}")
    results_raw = run_dir / "results.json"
    if results_raw.is_symlink():
        raise CoverageError(f"refusing symlink results input: {results_raw}")
    results_path = results_raw.resolve()
    out_csv = run_dir / "importance_coverage.csv"
    out_json = run_dir / "importance_coverage.json"
    for path in (out_csv, out_json):
        if path.is_symlink():
            raise CoverageError(f"refusing symlink output: {path}")
    for path in (trace_path, results_path, index_path, EXPERIMENT_SPEC):
        if not path.is_file():
            raise FileNotFoundError(f"required input not found: {path}")
    return (run_dir, trace_path, results_path, index_path, store,
            out_csv, out_json)


def _workload_sha(index, start: int, count: int) -> str:
    blob = "\n".join(
        f"{entry['image_id']}\t{question['question_id']}"
        for entry in index
        for question in entry["questions"][start:start + count]
    ).encode()
    return hashlib.sha256(blob).hexdigest()


def _parse_budgets(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        value = float(item.strip())
        if not 0.0 < value <= 1.0:
            raise ValueError(f"budget must be in (0, 1], got {value}")
        if not any(math.isclose(value, old, abs_tol=1e-12)
                   for old in values):
            values.append(value)
    if not values:
        raise ValueError("at least one budget is required")
    return values


def _method_identity(method_key: str) -> tuple[str | None, float | None]:
    for family, prefix in FAMILY_PREFIXES.items():
        if method_key == prefix:
            return family, None
        match = re.fullmatch(re.escape(prefix) + r"@(\d+(?:\.\d+)?)",
                             method_key)
        if match:
            return family, float(match.group(1)) / 100.0
    return None, None


def _row_budget(row: dict, method_key: str, suffix_budget: float | None) -> float:
    raw = str(row.get("retention", "")).strip()
    if not raw or raw.lower() in ("none", "null", "nan"):
        raise ValueError(f"missing retention for chunk method {method_key}")
    value = float(raw)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(f"invalid retention for {method_key}: {raw!r}")
    if suffix_budget is not None and not math.isclose(
            value, suffix_budget, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"method suffix/retention mismatch for {method_key}: {value}")
    return value


def _budget_key(value: float, requested: list[float]) -> float | None:
    for target in requested:
        if math.isclose(value, target, rel_tol=0.0, abs_tol=1e-9):
            return target
    return None


def _normalise_selected(selected, method_key: str, image_id: str):
    if not isinstance(selected, list) or not selected:
        raise ValueError(
            f"missing selected_chunk_ids_per_layer for {method_key}/{image_id}"
        )
    answer = []
    for layer in selected:
        if not isinstance(layer, list):
            raise ValueError(
                f"invalid selected layer for {method_key}/{image_id}")
        chunks = []
        for chunk in layer:
            if isinstance(chunk, bool) or not isinstance(chunk, int):
                raise ValueError(
                    f"non-integral chunk id {chunk!r} for "
                    f"{method_key}/{image_id}")
            chunks.append(chunk)
        answer.append(tuple(chunks))
    return tuple(answer)


def _parse_selected(raw: str, method_key: str, image_id: str):
    try:
        selected = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"invalid selected_chunk_ids_per_layer for {method_key}/{image_id}"
        ) from exc
    return _normalise_selected(selected, method_key, image_id)


def _load_selection_trace(trace_bytes: bytes, index, eval_skip: int,
                          eval_questions: int, budgets: list[float],
                          require_50: bool):
    """Load and strictly validate query-independent selections from CSV."""
    expected_questions = {
        (str(entry["image_id"]), str(q["question_id"])): q["question"]
        for entry in index
        for q in entry["questions"][eval_skip:eval_skip + eval_questions]
    }
    seen_question_rows = defaultdict(set)
    selection_by_image = {}
    selection_by_request = {}
    method_keys = defaultdict(set)
    separator_policies = defaultdict(set)
    selection_modes = defaultdict(set)
    duplicate_rows = []
    all_rows = []
    rows_by_request = {}
    all_request_keys = set()
    all_method_keys = set()
    chunk_budgets_in_trace = set()

    try:
        text = trace_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("selection trace is not UTF-8") from exc
    with io.StringIO(text, newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "dataset", "method_key", "retention", "retention_kind",
            "image_id", "question_id", "question", "prediction",
            "ground_truth", "correct", "n_chunks_selected",
            "n_chunks_total", "normal_chunk_count_total",
            "normal_kv_read_bytes", "separator_read_bytes",
            "total_actual_pread_bytes", "static_score_calls",
            "query_score_calls", "diversity_calls",
            "selected_chunk_ids_per_layer", "separator_policy", "selection_mode",
            "reordered_prefix_store_validated",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"trace is missing columns: {sorted(missing)}")

        for row in reader:
            method_key = str(row["method_key"])
            request_key = (method_key, str(row["image_id"]),
                           str(row["question_id"]))
            if request_key in all_request_keys:
                raise ValueError(f"duplicate method/request row: {request_key}")
            all_request_keys.add(request_key)
            rows_by_request[request_key] = row
            all_method_keys.add(method_key)
            all_rows.append(row)
            if row["dataset"] != "gqa":
                raise ValueError(f"non-GQA trace row: {request_key}")

            family, suffix_budget = _method_identity(method_key)
            if family is None:
                continue
            if row["retention_kind"] != "chunk":
                raise ValueError(
                    f"chunk method has retention_kind={row['retention_kind']!r}: "
                    f"{method_key}")
            expected_calls = {
                "prefix": (0, 0, 0),
                "static": (EXPECTED_LAYERS, 0, 0),
                "static_diverse": (EXPECTED_LAYERS, 0, EXPECTED_LAYERS),
            }[family]
            observed_calls = tuple(_csv_int(row, field) for field in (
                "static_score_calls", "query_score_calls", "diversity_calls"))
            if observed_calls != expected_calls:
                raise ValueError(
                    f"selector counter mismatch for {method_key}: "
                    f"{observed_calls} != {expected_calls}")
            if family == "prefix" and str(
                    row.get("reordered_prefix_store_validated", "")).lower() \
                    != "true":
                raise ValueError(
                    f"prefix store validation is not true for {request_key}")
            row_budget = _row_budget(row, method_key, suffix_budget)
            chunk_budgets_in_trace.add(row_budget)
            budget = _budget_key(row_budget, budgets)
            if budget is None:
                raise ValueError(
                    f"unrequested budget {row_budget:g} for {method_key}")
            combo = (family, budget)
            image_id = str(row["image_id"])
            question_id = str(row["question_id"])
            question_key = (image_id, question_id)
            if question_key not in expected_questions:
                raise ValueError(
                    f"unexpected evaluation request in trace: {question_key}"
                )
            if row["question"] != expected_questions[question_key]:
                raise ValueError(f"question text differs for {question_key}")
            if question_key in seen_question_rows[combo]:
                duplicate_rows.append((family, budget, *question_key))
            seen_question_rows[combo].add(question_key)
            method_keys[combo].add(method_key)
            separator_policies[combo].add(str(row["separator_policy"]))
            selection_modes[combo].add(str(row["selection_mode"]))

            selected = _parse_selected(
                row["selected_chunk_ids_per_layer"], method_key, image_id)
            selection_by_request[request_key] = selected
            image_key = (family, budget, image_id)
            prior = selection_by_image.get(image_key)
            if prior is not None and prior != selected:
                raise ValueError(
                    f"query-dependent trace for {family}@{budget:g}/{image_id}"
                )
            selection_by_image[image_key] = selected

    if duplicate_rows:
        raise ValueError(f"duplicate method/request rows: {duplicate_rows[:3]}")

    expected_set = set(expected_questions)
    if not any(math.isclose(b, 0.25, abs_tol=1e-9) for b in budgets):
        raise ValueError("the primary 25% budget must be requested")
    if require_50 and not any(
            math.isclose(b, 0.50, abs_tol=1e-9) for b in budgets):
        raise ValueError("--require-50 requires 0.50 in --budgets")

    complete_combos = []
    missing_by_budget = {}
    for budget in budgets:
        present = {
            family for family in FAMILY_PREFIXES
            if (family, budget) in seen_question_rows
        }
        if math.isclose(budget, 0.25, abs_tol=1e-9):
            if present != set(FAMILY_PREFIXES):
                raise ValueError(
                    f"25% trace must contain all three families; got {present}"
                )
        if math.isclose(budget, 0.50, abs_tol=1e-9):
            pair = {"prefix", "static_diverse"}
            pair_present = present & pair
            if pair_present and pair_present != pair:
                raise ValueError(
                    "50% Prefix and Static+Diverse must be both present or "
                    f"both absent; got {present}")
            if "static" in present and pair_present != pair:
                raise ValueError(
                    "Static 50% cannot appear without the complete "
                    "Prefix/Static+Diverse 50% pair")
            if require_50 and pair_present != pair:
                raise ValueError(
                    "--require-50 requested, but the complete 50% "
                    "Prefix/Static+Diverse pair is missing")
        missing_by_budget[f"{budget:.12g}"] = sorted(
            set(FAMILY_PREFIXES) - present)
        complete_combos.extend((family, budget) for family in FAMILY_PREFIXES
                               if family in present)

    if not complete_combos:
        raise ValueError("no complete Prefix/Static/Static+Diverse budget found")

    for combo in complete_combos:
        if seen_question_rows[combo] != expected_set:
            missing = expected_set - seen_question_rows[combo]
            extra = seen_question_rows[combo] - expected_set
            raise ValueError(
                f"workload mismatch for {combo}: missing={len(missing)}, "
                f"extra={len(extra)}"
            )
        if len(method_keys[combo]) != 1:
            raise ValueError(f"ambiguous method keys for {combo}: {method_keys[combo]}")
        if separator_policies[combo] != {"sidecar"}:
            raise ValueError(
                f"{combo} must use separator sidecar, got {separator_policies[combo]}"
            )
        expected_mode = EXPECTED_SELECTION_MODE[combo[0]]
        if selection_modes[combo] != {expected_mode}:
            raise ValueError(
                f"{combo} selection_mode must be {expected_mode}, got "
                f"{selection_modes[combo]}"
            )

    return {
        "selections": selection_by_image,
        "request_selections": selection_by_request,
        "combos": complete_combos,
        "method_keys": {
            combo: next(iter(method_keys[combo])) for combo in complete_combos
        },
        "expected_question_count": len(expected_set),
        "trace_rows_used": sum(len(seen_question_rows[c]) for c in complete_combos),
        "all_rows": all_rows,
        "rows_by_request": rows_by_request,
        "all_method_keys": sorted(all_method_keys),
        "all_request_count": len(all_request_keys),
        "chunk_budgets_in_trace": sorted(chunk_budgets_in_trace),
        "missing_families_by_budget": missing_by_budget,
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CoverageError(message)


def _float_list(values, label: str) -> list[float]:
    if not isinstance(values, list):
        raise CoverageError(f"{label} must be a list")
    answer = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise CoverageError(f"invalid {label} value: {value!r}") from exc
        if not math.isfinite(value):
            raise CoverageError(f"non-finite {label} value: {value!r}")
        answer.append(value)
    return answer


def _same_float_sets(left: list[float], right: list[float]) -> bool:
    return (len(left) == len(right)
            and all(any(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)
                        for b in right) for a in left)
            and all(any(math.isclose(b, a, rel_tol=0.0, abs_tol=1e-9)
                        for a in left) for b in right))


def _command_options(command: str) -> tuple[list[str], dict[str, str | bool]]:
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        raise CoverageError("results summary command cannot be parsed") from exc
    _require(len(parts) >= 2 and parts[1].endswith("scripts/04_eval.py"),
             "results command is not scripts/04_eval.py")
    options = {}
    i = 2
    while i < len(parts):
        item = parts[i]
        _require(item.startswith("--"),
                 f"unexpected positional result-command argument: {item}")
        _require(item not in options,
                 f"duplicate result-command option: {item}")
        if i + 1 < len(parts) and not parts[i + 1].startswith("--"):
            options[item] = parts[i + 1]
            i += 2
        else:
            options[item] = True
            i += 1
    return parts, options


def _require_command_value(options: dict, name: str, expected) -> None:
    observed = options.get(name)
    _require(observed is not None and observed is not True,
             f"results command is missing {name}")
    if isinstance(expected, float):
        try:
            valid = math.isclose(float(observed), expected,
                                 rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            valid = False
    elif isinstance(expected, int):
        try:
            valid = int(observed) == expected
        except (TypeError, ValueError):
            valid = False
    else:
        valid = str(observed) == str(expected)
    _require(valid,
             f"results command {name}={observed!r}, expected {expected!r}")


def _csv_number(row: dict, key: str) -> float:
    raw = row.get(key)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise CoverageError(f"invalid CSV numeric {key}={raw!r}") from exc
    _require(math.isfinite(value), f"non-finite CSV numeric {key}={raw!r}")
    return value


def _csv_int(row: dict, key: str) -> int:
    value = _csv_number(row, key)
    rounded = int(round(value))
    _require(math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-9),
             f"non-integral CSV numeric {key}={value!r}")
    return rounded


def _validate_experiment_spec(spec: dict, canonical_protocol: bool) -> None:
    _require(isinstance(spec, dict), "experiment_spec.json must contain an object")
    _require(spec.get("experiment") == "importance-reorder prefix baseline",
             "unexpected experiment_spec experiment")
    _require(spec.get("status") == "preregistered_before_main_run",
             "experiment_spec is not preregistered")
    _require(str(spec.get("dataset", "")).lower() == "gqa",
             "experiment_spec dataset must be GQA")
    _require(spec.get("model") == EXPECTED_MODEL_ID,
             "experiment_spec model mismatch")

    store_spec = spec.get("store") or {}
    _require(store_spec.get("path") == "kvstore",
             "experiment_spec store path mismatch")
    _require(store_spec.get("reuse_existing_read_only") is True,
             "experiment_spec must declare read-only store reuse")
    _require(store_spec.get("importance_reordered") is True
             and store_spec.get("per_layer_order") is True,
             "experiment_spec must declare per-layer importance reorder")
    _require(int(store_spec.get("chunk_size_tokens", -1)) == EXPECTED_CHUNK_SIZE,
             "experiment_spec chunk size mismatch")
    _require(store_spec.get("content_aggregate_sha256")
             == CANONICAL_STORE_CONTENT_AGGREGATE_SHA256,
             "experiment_spec full-store provenance hash mismatch")

    conditions = spec.get("conditions") or {}
    _require(conditions.get("separator_policy") == "sidecar",
             "experiment_spec separator policy mismatch")
    _require(conditions.get("cold_page_cache") is True,
             "experiment_spec must declare cold page cache")
    _require(conditions.get("greedy_decoding") is True,
             "experiment_spec must declare greedy decoding")
    _require(int(conditions.get("max_new_tokens", -1)) == 16,
             "experiment_spec generation length mismatch")
    spec_budgets = _float_list(conditions.get("budgets"),
                               "experiment_spec conditions.budgets")
    _require(_same_float_sets(spec_budgets, [0.25, 0.50]),
             "experiment_spec budgets must be 25% and 50%")

    if canonical_protocol:
        workload = spec.get("main_workload") or {}
        _require(int(workload.get("images", -1)) == 40,
                 "experiment_spec image count mismatch")
        _require(int(workload.get("evaluation_questions", -1)) == 240,
                 "experiment_spec evaluation count mismatch")
        _require(int(workload.get("calibration_questions_per_image", -1)) == 4,
                 "experiment_spec calibration count mismatch")
        _require(workload.get("index_sha256") == CANONICAL_INDEX_SHA256,
                 "experiment_spec index hash mismatch")
        _require(workload.get("evaluation_workload_sha256")
                 == CANONICAL_EVAL_WORKLOAD_SHA256,
                 "experiment_spec evaluation workload hash mismatch")
        _require(workload.get("calibration_question_id_sha256")
                 == CANONICAL_CALIBRATION_WORKLOAD_SHA256,
                 "experiment_spec calibration workload hash mismatch")


def _validate_results(results: dict, trace: dict, index, args,
                      index_sha: str, eval_workload_sha: str,
                      requested_budgets: list[float]) -> dict:
    """Cross-check results.json, its flattened CSV, and the frozen workload."""
    _require(isinstance(results, dict), "results.json must contain an object")
    _require(results.get("schema_version") == 2,
             "results.json top-level schema_version must be 2")
    summary = results.get("summary")
    _require(isinstance(summary, dict), "results.json summary is missing")
    _require(summary.get("schema_version") == 2,
             "results.json summary schema_version must be 2")

    expected_questions = {}
    expected_gold = {}
    for entry in index:
        for q in entry["questions"][
                args.eval_skip:args.eval_skip + args.eval_questions]:
            key = (str(entry["image_id"]), str(q["question_id"]))
            expected_questions[key] = q["question"]
            expected_gold[key] = (q["answers"] if "answers" in q
                                  else [q["answer"]])
    expected_set = set(expected_questions)
    expected_order = list(expected_questions)
    rows_by_method = defaultdict(list)
    for row in trace["all_rows"]:
        key = (str(row["image_id"]), str(row["question_id"]))
        if key not in expected_questions:
            raise CoverageError(f"CSV contains request outside workload: {key}")
        if row["question"] != expected_questions[key]:
            raise CoverageError(f"CSV question differs from index: {key}")
        try:
            csv_gold = json.loads(row["ground_truth"])
        except json.JSONDecodeError as exc:
            raise CoverageError(f"invalid CSV ground truth: {key}") from exc
        _require(csv_gold == expected_gold[key],
                 f"CSV ground truth differs from index: {key}")
        rows_by_method[str(row["method_key"])].append(key)
    for method, keys in rows_by_method.items():
        _require(keys == expected_order,
                 f"CSV workload incomplete for {method}: "
                 f"{len(keys)} != {len(expected_set)}")
    _require(set(rows_by_method) == set(trace["all_method_keys"]),
             "internal CSV method accounting mismatch")
    actual_methods = set(trace["all_method_keys"])
    _require(REQUIRED_25_METHODS.issubset(actual_methods),
             "CSV omits a preregistered 25% method")
    _require(not (actual_methods - ALLOWED_MAIN_METHODS),
             f"CSV contains undeclared methods: "
             f"{sorted(actual_methods - ALLOWED_MAIN_METHODS)}")

    expected_n = len(expected_set)
    _require(int(summary.get("n", -1)) == expected_n,
             "results summary question count mismatch")
    _require(int(summary.get("n_images", -1)) == len(index),
             "results summary image count mismatch")
    _require(int(summary.get("skip", -1)) == args.eval_skip,
             "results summary skip mismatch")
    _require(int(summary.get("questions_per_image_requested", -1))
             == args.eval_questions,
             "results summary questions-per-image mismatch")
    _require(summary.get("metric") == "gqa",
             "results summary metric must be gqa")
    _require(summary.get("sep_policy") == "sidecar",
             "results summary separator policy mismatch")
    _require(summary.get("cold") is True,
             "results summary must be a cold-page-cache run")
    _require(int(summary.get("max_new_tokens", -1)) == 16,
             "results summary max_new_tokens mismatch")
    _require(math.isclose(float(summary.get("ratio", -1)), 0.25,
                          rel_tol=0.0, abs_tol=1e-12),
             "results summary SparseVLM ratio mismatch")
    _require(summary.get("index_sha256") == index_sha,
             "results summary index hash mismatch")
    _require(summary.get("workload_sha256") == eval_workload_sha,
             "results summary workload hash mismatch")
    _require(math.isclose(float(summary.get("alpha", -1)), 0.6,
                          rel_tol=0.0, abs_tol=1e-12),
             "results summary alpha mismatch")
    _require(int(summary.get("probe_heads", -1)) == 3,
             "results summary probe-head count mismatch")
    for name in ("lam_static", "lam_query"):
        _require(math.isclose(float(summary.get(name, -1)), 1.0,
                              rel_tol=0.0, abs_tol=1e-12),
                 f"results summary {name} mismatch")
    _require(math.isclose(float(summary.get("diverse_frac", -1)), 0.25,
                          rel_tol=0.0, abs_tol=1e-12),
             "results summary diversity fraction mismatch")
    qpi = summary.get("questions_per_image") or {}
    _require(int(qpi.get("min", -1)) == args.eval_questions
             and int(qpi.get("max", -1)) == args.eval_questions
             and math.isclose(float(qpi.get("mean", -1)),
                              float(args.eval_questions), abs_tol=1e-12),
             "results summary questions_per_image distribution mismatch")
    expected_latency = {
        "ttft": "request start through first output token",
        "decode": "after first output token through final output token",
        "e2e": "request start through final output token",
    }
    _require(summary.get("latency_definition") == expected_latency,
             "results summary latency definition mismatch")
    result_index = Path(str(summary.get("index", "")))
    if not result_index.is_absolute():
        result_index = PROJECT_ROOT / result_index
    _require(result_index.resolve() == Path(args.index).resolve(),
             "results summary index path mismatch")

    _parts, command_options = _command_options(str(summary.get("command", "")))
    _require("--warm" not in command_options,
             "results command requested a warm page cache")
    _require_command_value(command_options, "--limit", len(index))
    _require_command_value(command_options, "--questions", args.eval_questions)
    _require_command_value(command_options, "--skip", args.eval_skip)
    _require_command_value(command_options, "--ratio", 0.25)
    _require_command_value(command_options, "--sep-policy", "sidecar")
    _require_command_value(command_options, "--metric", "gqa")
    _require_command_value(command_options, "--dataset", "gqa")
    _require_command_value(command_options, "--expect-images", len(index))
    _require_command_value(command_options, "--expect-questions", expected_n)
    command_index = Path(str(command_options.get("--index", "")))
    command_store = Path(str(command_options.get("--store", "")))
    command_run = Path(str(command_options.get("--run-dir", "")))
    if not command_index.is_absolute():
        command_index = PROJECT_ROOT / command_index
    if not command_store.is_absolute():
        command_store = PROJECT_ROOT / command_store
    if not command_run.is_absolute():
        command_run = PROJECT_ROOT / command_run
    _require(command_index.resolve() == Path(args.index).resolve(),
             "results command index path mismatch")
    _require(command_store.resolve() == Path(args.store).resolve(),
             "results command store path mismatch")
    _require(command_run.resolve() == Path(args.run_dir).resolve(),
             "results command run-dir mismatch")
    command_budgets = _parse_budgets(str(command_options.get("--budgets", "")))
    _require(_same_float_sets(command_budgets,
                              _float_list(summary.get("budgets"),
                                          "results summary budgets")),
             "results command/summary budgets mismatch")
    selectors = {
        item.strip() for item in
        str(command_options.get("--selectors", "")).split(",")
        if item.strip()
    }
    _require(selectors == EXPECTED_SELECTOR_NAMES,
             "results command selector set differs from preregistration")

    runtime = summary.get("runtime")
    if runtime is not None:
        _require(isinstance(runtime, dict),
                 "results runtime provenance must be an object")
        expected_runtime = {
            "model_id": EXPECTED_MODEL_ID,
            "load_4bit": True,
            "quant_type": "nf4",
            "double_quant": True,
            "compute_dtype": "bfloat16",
            "attention_implementation": "eager",
            "greedy": True,
        }
        for key, expected in expected_runtime.items():
            _require(runtime.get(key) == expected,
                     f"results runtime provenance mismatch for {key}")

    result_budgets = _float_list(summary.get("budgets"),
                                 "results summary budgets")
    _require(any(math.isclose(x, 0.25, abs_tol=1e-9)
                 for x in result_budgets),
             "results summary omits primary 25% budget")
    _require(_same_float_sets(result_budgets,
                              trace["chunk_budgets_in_trace"]),
             "results summary budgets differ from traced coverage methods")
    _require(all(any(math.isclose(x, wanted, abs_tol=1e-9)
                     for wanted in requested_budgets)
                 for x in result_budgets),
             "results contain a budget not requested for coverage")

    per_method = summary.get("per_method")
    _require(isinstance(per_method, dict),
             "results summary per_method is missing")
    _require(set(per_method) == set(trace["all_method_keys"]),
             "results summary and CSV method sets differ")

    result_rows = results.get("rows")
    _require(isinstance(result_rows, list), "results.json rows is missing")
    _require(len(result_rows) == expected_n,
             "results.json row count mismatch")
    seen = set()
    result_selection_checks = 0
    result_csv_field_checks = 0
    metadata_keys = {"image_id", "question_id", "question", "gold"}
    for record in result_rows:
        _require(isinstance(record, dict), "non-object results row")
        key = (str(record.get("image_id")), str(record.get("question_id")))
        _require(key in expected_questions,
                 f"results.json contains request outside workload: {key}")
        _require(key not in seen, f"duplicate results.json request: {key}")
        seen.add(key)
        _require(record.get("question") == expected_questions[key],
                 f"results.json question differs from index: {key}")
        _require(record.get("gold") == expected_gold[key],
                 f"results.json gold differs from index: {key}")
        record_methods = set(record) - metadata_keys
        _require(record_methods == set(trace["all_method_keys"]),
                 f"results.json method set differs for {key}")
        for method in record_methods:
            family, _ = _method_identity(method)
            if family is None:
                continue
            method_result = record.get(method)
            _require(isinstance(method_result, dict),
                     f"invalid method result {method}/{key}")
            observed = _normalise_selected(
                method_result.get("selected_chunk_ids_per_layer"),
                method, key[0])
            expected = trace["request_selections"][(method, *key)]
            _require(observed == expected,
                     f"results.json/CSV selection mismatch: {method}/{key}")
            result_selection_checks += 1
            csv_row = trace["rows_by_request"][(method, *key)]
            _require(csv_row.get("prediction") == method_result.get("answer"),
                     f"results.json/CSV prediction mismatch: {method}/{key}")
            for csv_key, result_key in (
                    ("correct", "acc"),
                    ("n_chunks_selected", "n_chunks_selected"),
                    ("n_chunks_total", "n_chunks_total"),
                    ("normal_chunk_count_total", "normal_chunk_count_total"),
                    ("normal_kv_read_bytes", "normal_kv_read_bytes"),
                    ("separator_read_bytes", "separator_read_bytes"),
                    ("total_actual_pread_bytes", "total_actual_pread_bytes"),
                    ("static_score_calls", "static_score_calls"),
                    ("query_score_calls", "query_score_calls"),
                    ("diversity_calls", "diversity_calls")):
                observed_number = method_result.get(result_key)
                _require(observed_number is not None
                         and math.isclose(_csv_number(csv_row, csv_key),
                                          float(observed_number),
                                          rel_tol=0.0, abs_tol=1e-9),
                         f"results.json/CSV {csv_key} mismatch: {method}/{key}")
                result_csv_field_checks += 1
            _require(csv_row.get("selection_mode")
                     == method_result.get("selection_mode"),
                     f"results.json/CSV selection_mode mismatch: {method}/{key}")
            _require(csv_row.get("separator_policy")
                     == method_result.get("separator_policy"),
                     f"results.json/CSV separator_policy mismatch: {method}/{key}")
    _require(seen == expected_set, "results.json workload is incomplete")

    return {
        "schema_v2": True,
        "cold_page_cache": True,
        "gqa_metric": True,
        "index_and_workload_hash_match": True,
        "csv_results_method_sets_match": True,
        "csv_results_selections_match": True,
        "selection_crosschecks": result_selection_checks,
        "critical_field_crosschecks": result_csv_field_checks,
        "results_budgets": result_budgets,
        "source_model_runtime_block_present":
            isinstance(summary.get("runtime"), dict),
    }


def _orders(meta: dict) -> list[list[int]]:
    order = meta.get("order")
    if not meta.get("reordered") or not order:
        raise ValueError("coverage requires an already importance-reordered store")
    if meta.get("order_is_per_layer") is not True:
        raise ValueError(
            "coverage requires the per-layer importance order; a shared "
            "Morton/raster order is invalid")
    orders = order
    if len(orders) != int(meta["num_layers"]):
        raise ValueError("store order does not match num_layers")
    return [[int(x) for x in layer] for layer in orders]


def _separator_positions(meta: dict, layer: int) -> list[int]:
    stored = meta.get("newline_stored", meta["newline_idx"])
    row = stored[layer] if stored and isinstance(stored[0], list) else stored
    return [int(x) for x in row]


def _validate_store_mapping(meta: dict, image_id: str):
    if meta.get("model") != EXPECTED_MODEL_ID:
        raise ValueError(f"store model mismatch for {image_id}")
    if meta.get("dtype") != EXPECTED_STORE_DTYPE:
        raise ValueError(f"store dtype mismatch for {image_id}")
    if int(meta.get("chunk_size", -1)) != EXPECTED_CHUNK_SIZE:
        raise ValueError(f"store chunk size mismatch for {image_id}")
    if int(meta.get("num_layers", -1)) != EXPECTED_LAYERS:
        raise ValueError(f"store layer count mismatch for {image_id}")
    vn = int(meta["v_token_num"])
    expected_chunks = (vn + EXPECTED_CHUNK_SIZE - 1) // EXPECTED_CHUNK_SIZE
    if int(meta.get("n_chunks_per_layer", -1)) != expected_chunks:
        raise ValueError(f"store chunk geometry mismatch for {image_id}")
    expected = list(range(vn))
    orders = _orders(meta)
    original_separators = [int(x) for x in meta["newline_idx"]]
    if len(set(original_separators)) != len(original_separators):
        raise ValueError(f"duplicate original separator in {image_id}")
    for layer, order in enumerate(orders):
        if len(order) != vn or sorted(order) != expected:
            raise ValueError(f"invalid stored->original permutation {image_id}/L{layer}")
        stored_separators = _separator_positions(meta, layer)
        if len(set(stored_separators)) != len(original_separators):
            raise ValueError(f"invalid stored separator count {image_id}/L{layer}")
        if any(position < 0 or position >= vn
               for position in stored_separators):
            raise ValueError(f"stored separator out of range {image_id}/L{layer}")
        mapped = [order[position] for position in stored_separators]
        if mapped != original_separators:
            raise ValueError(f"separator mapping mismatch {image_id}/L{layer}")
    return orders


def _rank_metrics(stored_scores: np.ndarray, stored_to_original: list[int],
                  original_separators: list[int], budgets: list[float]):
    """Compare current metadata order with a fresh stable score ranking."""
    vn = stored_scores.size
    order = np.asarray(stored_to_original, dtype=np.int64)
    original_scores = np.empty(vn, dtype=np.float64)
    original_scores[order] = stored_scores

    recalculated = np.argsort(-original_scores, kind="mergesort")
    current_rank = np.empty(vn, dtype=np.int64)
    recalculated_rank = np.empty(vn, dtype=np.int64)
    current_rank[order] = np.arange(vn)
    recalculated_rank[recalculated] = np.arange(vn)
    rho_all = float(np.corrcoef(current_rank, recalculated_rank)[0, 1])

    sep = set(original_separators)
    normal_original = np.asarray([i for i in range(vn) if i not in sep])
    current_normal = np.asarray([i for i in order if i not in sep])
    recalculated_normal = normal_original[
        np.argsort(-original_scores[normal_original], kind="mergesort")]
    current_normal_rank = np.empty(vn, dtype=np.int64)
    recalculated_normal_rank = np.empty(vn, dtype=np.int64)
    current_normal_rank[current_normal] = np.arange(current_normal.size)
    recalculated_normal_rank[recalculated_normal] = np.arange(
        recalculated_normal.size)
    rho_normal = float(np.corrcoef(
        current_normal_rank[normal_original],
        recalculated_normal_rank[normal_original],
    )[0, 1])

    normal_stored_scores = original_scores[current_normal]
    if normal_stored_scores.size > 1:
        adjacent_fraction = float(np.mean(
            normal_stored_scores[:-1] >= normal_stored_scores[1:]))
        maximum_upward_step = float(np.maximum(
            normal_stored_scores[1:] - normal_stored_scores[:-1], 0.0).max())
    else:
        adjacent_fraction = 1.0
        maximum_upward_step = 0.0

    result = {
        "exact_rank_fraction_all_tokens": float(np.mean(recalculated == order)),
        "exact_rank_fraction_normal_tokens": float(np.mean(
            recalculated_normal == current_normal)),
        "spearman_all_tokens": rho_all,
        "spearman_normal_tokens": rho_normal,
        "adjacent_nonincreasing_fraction_normal_tokens": adjacent_fraction,
        "maximum_upward_score_step_normal_tokens": maximum_upward_step,
    }
    for budget in budgets:
        k = max(1, min(current_normal.size,
                       int(round(current_normal.size * budget))))
        a = set(current_normal[:k].tolist())
        b = set(recalculated_normal[:k].tolist())
        result[f"normal_token_top_{int(round(budget * 100))}_jaccard"] = (
            len(a & b) / max(1, len(a | b))
        )
    return result


def _chunk_rows(chunk_ids, visual_tokens: int, chunk_size: int) -> np.ndarray:
    mask = np.zeros(visual_tokens, dtype=bool)
    for chunk in chunk_ids:
        start = int(chunk) * chunk_size
        stop = min(visual_tokens, start + chunk_size)
        mask[start:stop] = True
    return mask


def _summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["budget"])].append(row)
    answer = []
    for (family, budget), values in sorted(
            grouped.items(), key=lambda x: (x[0][1], x[0][0])):
        primary = np.asarray([v["normal_mass_coverage"] for v in values])
        secondary = np.asarray([v["sidecar_union_mass_coverage"] for v in values])
        answer.append({
            "family": family,
            "method": DISPLAY_NAME[family],
            "budget": budget,
            "n_image_layers": len(values),
            "normal_mass_coverage_macro_mean": float(primary.mean()),
            "normal_mass_coverage_median": float(np.median(primary)),
            "normal_mass_coverage_p05": float(np.percentile(primary, 5)),
            "normal_mass_coverage_p95": float(np.percentile(primary, 95)),
            "normal_mass_coverage_global_weighted": float(
                sum(v["selected_normal_importance_mass"] for v in values)
                / sum(v["total_normal_importance_mass"] for v in values)),
            "sidecar_union_mass_coverage_macro_mean": float(secondary.mean()),
            "sidecar_union_mass_coverage_median": float(np.median(secondary)),
            "sidecar_union_mass_coverage_global_weighted": float(
                sum(v["selected_sidecar_union_importance_mass"] for v in values)
                / sum(v["total_visual_importance_mass"] for v in values)),
            "selected_chunks_mean": float(np.mean([
                v["selected_chunk_count"] for v in values])),
            "selected_normal_tokens_mean": float(np.mean([
                v["selected_normal_token_count"] for v in values])),
        })
    return answer


def _store_state(store: Path, image_ids: list[str]):
    state = {}
    for image_id in image_ids:
        root = store / image_id
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise CoverageError(f"symlink inside canonical store: {path}")
            if not path.is_file():
                continue
            stat = path.stat()
            state[str(path.relative_to(store))] = (stat.st_size, stat.st_mtime_ns)
    return state


def _meta_aggregate(store: Path) -> dict:
    """Hash all canonical meta.json files with the frozen tree algorithm."""
    paths = sorted(
        store.glob("*/meta.json"),
        key=lambda path: os.fsencode(
            path.relative_to(PROJECT_ROOT).as_posix()))
    if len(paths) != 40:
        raise CoverageError(
            f"canonical store must contain 40 meta.json files, got {len(paths)}")
    outer = hashlib.sha256()
    total = 0
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise CoverageError(f"invalid canonical metadata file: {path}")
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        size = path.stat().st_size
        total += size
        outer.update(f"{_sha256(path)}  {rel}\n".encode())
    return {"sha256": outer.hexdigest(), "files": len(paths), "bytes": total}


class _OutputLock:
    """Advisory single-writer lock held across validation and GPU analysis."""

    def __init__(self, run_dir: Path):
        key = hashlib.sha256(str(run_dir.resolve()).encode()).hexdigest()[:24]
        self.path = Path("/tmp") / f"mllm_v2_importance_coverage_{key}.lock"
        self.fd = None

    def __enter__(self):
        if self.path.is_symlink():
            raise CoverageError(f"refusing symlink output lock: {self.path}")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        self.fd = os.open(self.path, flags, 0o664)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(self.fd)
            self.fd = None
            raise CoverageError(
                f"another calibration-coverage writer holds {self.path}") from exc
        return self

    def __exit__(self, *_exc):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _stage_bytes(destination: Path, data: bytes, generation: str) -> Path:
    staged = destination.with_name(
        f".{destination.name}.staged.{generation}")
    flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL
             | getattr(os, "O_CLOEXEC", 0))
    fd = os.open(staged, flags, 0o664)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            staged.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(fd)
    return staged


def _publish_pair(out_csv: Path, csv_bytes: bytes,
                  out_json: Path, json_bytes: bytes,
                  overwrite: bool, generation: str) -> None:
    """Publish CSV first and JSON last as its commit manifest.

    Direct filesystem names cannot be committed in one POSIX rename.  The JSON
    therefore carries the generation id and CSV digest and is the commit
    record.  A held writer lock, no-clobber hard links, and rollback make the
    pair safe against concurrent tool invocations and ordinary exceptions.
    """
    for path in (out_csv, out_json):
        if path.parent != out_csv.parent or path.is_symlink():
            raise CoverageError(f"unsafe output path: {path}")
    staged_csv = staged_json = None
    installed = []
    backups = {}
    committed = False
    try:
        staged_csv = _stage_bytes(out_csv, csv_bytes, generation)
        staged_json = _stage_bytes(out_json, json_bytes, generation)
        _fsync_directory(out_csv.parent)

        existing = [path for path in (out_csv, out_json) if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(
                f"refusing to overwrite coverage artifacts: {existing}")

        if overwrite:
            for path in (out_csv, out_json):
                if path.exists():
                    backup = path.with_name(
                        f".{path.name}.backup.{generation}")
                    os.replace(path, backup)
                    backups[path] = backup
            os.replace(staged_csv, out_csv)
            staged_csv = None
            installed.append(out_csv)
            os.replace(staged_json, out_json)  # JSON is the commit record.
            staged_json = None
            installed.append(out_json)
        else:
            # link() is an atomic no-replace publication primitive.
            os.link(staged_csv, out_csv)
            installed.append(out_csv)
            os.link(staged_json, out_json)
            installed.append(out_json)
        _fsync_directory(out_csv.parent)
        committed = True
    except BaseException:
        for path in reversed(installed):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        for path, backup in backups.items():
            if backup.exists():
                os.replace(backup, path)
        _fsync_directory(out_csv.parent)
        raise
    finally:
        for path in (staged_csv, staged_json):
            if path is not None:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
        if committed:
            for backup in backups.values():
                try:
                    backup.unlink()
                except OSError:
                    # A stale backup is safer than deleting the only recovery
                    # copy after an unexpected filesystem error.
                    pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute calibration SparseVLM mass and measure coverage "
                    "of recorded GQA chunk selections (analysis only).")
    parser.add_argument("--run-dir",
                        default="runs/reorder_prefix_baseline/calib4",
                        help="new baseline run directory containing per_request.csv")
    parser.add_argument("--trace", default=None,
                        help="selection trace CSV (default: RUN_DIR/per_request.csv)")
    parser.add_argument("--index", default=str(DATA_DIR / "index.json"))
    parser.add_argument("--store", default=str(STORE_DIR))
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--calib-questions", type=int, default=4)
    parser.add_argument("--eval-skip", type=int, default=4)
    parser.add_argument("--eval-questions", type=int, default=6)
    parser.add_argument("--budgets", default="0.25,0.50",
                        help="analyze complete method triplets at these budgets")
    parser.add_argument("--require-50", action="store_true",
                        help="require the Prefix/Static+Diverse 50%% pair")
    parser.add_argument("--rank-spearman-threshold", type=float, default=0.99,
                        help="minimum per-layer rank consistency (fail below)")
    parser.add_argument("--overwrite", action="store_true",
                        help="transactionally replace only this coverage pair")
    return parser


def _run_locked(args, paths):
    (run_dir, trace_path, results_path, index_path, store,
     out_csv, out_json) = paths
    if not args.overwrite:
        existing = [str(path) for path in (out_csv, out_json) if path.exists()]
        if existing:
            raise FileExistsError(
                f"refusing to overwrite coverage artifacts: {existing}")
    if args.limit <= 0 or args.calib_questions <= 0 \
            or args.eval_questions <= 0:
        raise ValueError("limit and question counts must be positive")
    if not 0.0 <= args.rank_spearman_threshold <= 1.0:
        raise ValueError("rank Spearman threshold must be in [0, 1]")
    budgets = _parse_budgets(args.budgets)

    trace_snapshot = _snapshot_file(trace_path)
    results_snapshot = _snapshot_file(results_path)
    index_snapshot = _snapshot_file(index_path)
    spec_snapshot = _snapshot_file(EXPERIMENT_SPEC)
    if spec_snapshot.sha256 != EXPERIMENT_SPEC_SHA256:
        raise CoverageError(
            "preregistered experiment_spec raw hash mismatch: "
            f"{spec_snapshot.sha256}")
    try:
        full_index = json.loads(index_snapshot.data)
        results = json.loads(results_snapshot.data)
        experiment_spec = json.loads(spec_snapshot.data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CoverageError(f"invalid JSON input: {exc}") from exc
    if not isinstance(full_index, list):
        raise CoverageError("index must contain a JSON list")
    index = full_index[:args.limit]
    if len(index) != args.limit:
        raise ValueError(f"requested {args.limit} images, index has {len(index)}")
    image_ids = [str(entry["image_id"]) for entry in index]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("duplicate image IDs in index slice")
    for entry in index:
        needed = max(args.calib_questions,
                     args.eval_skip + args.eval_questions)
        if len(entry["questions"]) < needed:
            raise ValueError(f"not enough questions for {entry['image_id']}")
        if not (store / str(entry["image_id"]) / "meta.json").is_file():
            raise FileNotFoundError(f"missing store for {entry['image_id']}")

    index_sha = index_snapshot.sha256
    eval_workload_sha = _workload_sha(
        index, args.eval_skip, args.eval_questions)
    calibration_workload_sha = _workload_sha(
        index, 0, args.calib_questions)
    canonical_protocol = (
        args.limit == 40 and args.calib_questions == 4
        and args.eval_skip == 4 and args.eval_questions == 6
    )
    if canonical_protocol:
        if index_sha != CANONICAL_INDEX_SHA256:
            raise ValueError(f"canonical index SHA mismatch: {index_sha}")
        if eval_workload_sha != CANONICAL_EVAL_WORKLOAD_SHA256:
            raise ValueError(
                f"canonical evaluation workload SHA mismatch: {eval_workload_sha}")
        if calibration_workload_sha != CANONICAL_CALIBRATION_WORKLOAD_SHA256:
            raise ValueError(
                "canonical calibration workload SHA mismatch: "
                f"{calibration_workload_sha}")

    _require(MODEL_ID == EXPECTED_MODEL_ID,
             "current config MODEL_ID differs from the frozen model")
    _require(LOAD_4BIT is True,
             "current config must enable 4-bit model loading")
    _require(ATTN_IMPL == "eager",
             "current config attention implementation must be eager")
    _require(STORE_DTYPE == EXPECTED_STORE_DTYPE,
             "current config store dtype mismatch")
    _require(COMPUTE_DTYPE == "bfloat16",
             "current config compute dtype must be bfloat16")
    _require(CHUNK_SIZE == EXPECTED_CHUNK_SIZE,
             "current config chunk size mismatch")
    _validate_experiment_spec(experiment_spec, canonical_protocol)

    calibration_ids = {
        (str(entry["image_id"]), str(q["question_id"]))
        for entry in index for q in entry["questions"][:args.calib_questions]
    }
    evaluation_ids = {
        (str(entry["image_id"]), str(q["question_id"]))
        for entry in index
        for q in entry["questions"][
            args.eval_skip:args.eval_skip + args.eval_questions]
    }
    if calibration_ids & evaluation_ids:
        raise ValueError("calibration and evaluation questions overlap")

    trace = _load_selection_trace(
        trace_snapshot.data, index, args.eval_skip, args.eval_questions,
        budgets, args.require_50)
    results_validation = _validate_results(
        results, trace, index, args, index_sha, eval_workload_sha, budgets)

    meta_by_image = {}
    for image_id in image_ids:
        meta_path = store / image_id / "meta.json"
        try:
            meta = json.loads(meta_path.read_bytes())
        except json.JSONDecodeError as exc:
            raise CoverageError(f"invalid store metadata: {meta_path}") from exc
        _validate_store_mapping(meta, image_id)
        meta_by_image[image_id] = meta
    meta_aggregate_before = _meta_aggregate(store)
    if meta_aggregate_before["sha256"] \
            != CANONICAL_STORE_META_AGGREGATE_SHA256:
        raise CoverageError(
            "canonical store meta aggregate mismatch: "
            f"{meta_aggregate_before['sha256']}")
    store_before = _store_state(store, image_ids)

    print("Loading model for analysis-only SparseVLM calibration recomputation")
    runner = LlavaRunner().load()
    _require(runner.model_id == EXPECTED_MODEL_ID,
             "loaded runner model ID mismatch")
    _require(runner.load_4bit is True,
             "loaded runner is not configured for 4-bit")
    _require(getattr(runner.model, "is_loaded_in_4bit", False) is True,
             "loaded model does not report 4-bit quantization")
    _require(runner.attn == "eager"
             and runner.model.config._attn_implementation == "eager",
             "loaded model attention implementation is not eager")
    _require(runner.model.training is False,
             "loaded model is not in eval mode")
    server = Server(runner)
    coverage_rows = []
    rank_rows = []
    started = time.time()

    for image_number, entry in enumerate(index, 1):
        image_id = str(entry["image_id"])
        image_store = store / image_id
        meta = meta_by_image[image_id]
        orders = _orders(meta)
        context = ImageContext(image_store, runner.model.device,
                               drop_cache=False)
        if context.meta != meta:
            context.close()
            raise CoverageError(
                f"store metadata changed before context open: {image_id}")
        try:
            questions = [q["question"] for q in
                         entry["questions"][:args.calib_questions]]
            BIAS.clear()
            scores_tensor = calibrate_image(server, context, questions)
            scores = scores_tensor.double().numpy()
            expected_shape = (int(meta["num_layers"]),
                              int(meta["v_token_num"]))
            if scores.shape != expected_shape:
                raise ValueError(
                    f"calibration score shape {scores.shape} != {expected_shape}"
                )
            if not np.isfinite(scores).all() or float(scores.min()) < -1e-12:
                raise ValueError(f"invalid calibration importance for {image_id}")

            vn = int(meta["v_token_num"])
            chunk_size = int(meta["chunk_size"])
            n_chunks = int(meta["n_chunks_per_layer"])
            original_separators = [int(x) for x in meta["newline_idx"]]

            for layer in range(int(meta["num_layers"])):
                layer_scores = scores[layer]
                separators = np.zeros(vn, dtype=bool)
                separators[_separator_positions(meta, layer)] = True
                normal = ~separators
                normal_total_mass = float(layer_scores[normal].sum())
                visual_total_mass = float(layer_scores.sum())
                if normal_total_mass <= 0.0 or visual_total_mass <= 0.0:
                    raise ValueError(f"zero calibration mass {image_id}/L{layer}")

                rank = _rank_metrics(
                    layer_scores, orders[layer], original_separators, budgets)
                rank_rows.append({
                    "image_id": image_id,
                    "layer": layer,
                    **rank,
                })

                for family, budget in trace["combos"]:
                    selected_layers = trace["selections"][
                        (family, budget, image_id)]
                    if len(selected_layers) != int(meta["num_layers"]):
                        raise ValueError(
                            f"layer trace mismatch {family}@{budget:g}/{image_id}"
                        )
                    chunk_ids = list(selected_layers[layer])
                    expected_chunks = budget_chunk_count(n_chunks, budget)
                    if len(chunk_ids) != expected_chunks \
                            or len(set(chunk_ids)) != len(chunk_ids):
                        raise ValueError(
                            f"budget violation {family}@{budget:g}/{image_id}/"
                            f"L{layer}: {chunk_ids}")
                    if any(chunk < 0 or chunk >= n_chunks for chunk in chunk_ids):
                        raise ValueError(
                            f"out-of-range chunk {family}@{budget:g}/{image_id}/"
                            f"L{layer}")
                    if family == "prefix" \
                            and chunk_ids != list(range(expected_chunks)):
                        raise ValueError(
                            f"Prefix is not exact first-k for {image_id}/L{layer}"
                        )

                    selected = _chunk_rows(chunk_ids, vn, chunk_size)
                    selected_normal = selected & normal
                    sidecar_union = selected | separators
                    selected_normal_mass = float(
                        layer_scores[selected_normal].sum())
                    selected_union_mass = float(
                        layer_scores[sidecar_union].sum())
                    normal_coverage = selected_normal_mass / normal_total_mass
                    union_coverage = selected_union_mass / visual_total_mass
                    if not (-1e-9 <= normal_coverage <= 1.0 + 1e-9
                            and -1e-9 <= union_coverage <= 1.0 + 1e-9):
                        raise ValueError(
                            f"coverage outside [0,1] for {family}/{image_id}/"
                            f"L{layer}")

                    coverage_rows.append({
                        "image_id": image_id,
                        "layer": layer,
                        "family": family,
                        "method": DISPLAY_NAME[family],
                        "method_key": trace["method_keys"][(family, budget)],
                        "budget": budget,
                        "selected_chunk_count": len(chunk_ids),
                        "total_chunk_count": n_chunks,
                        "selected_chunk_ids": json.dumps(chunk_ids),
                        "selected_normal_token_count": int(
                            selected_normal.sum()),
                        "total_normal_token_count": int(normal.sum()),
                        "selected_sidecar_union_token_count": int(
                            sidecar_union.sum()),
                        "total_visual_token_count": vn,
                        "selected_normal_importance_mass": selected_normal_mass,
                        "total_normal_importance_mass": normal_total_mass,
                        "normal_mass_coverage": normal_coverage,
                        "selected_sidecar_union_importance_mass":
                            selected_union_mass,
                        "total_visual_importance_mass": visual_total_mass,
                        "sidecar_union_mass_coverage": union_coverage,
                        "separator_token_count": int(separators.sum()),
                        "separator_policy": "sidecar",
                        "score_coordinate": "current_stored_visual_position",
                    })
        finally:
            BIAS.clear()
            context.close()
        del scores_tensor, scores, context
        torch.cuda.empty_cache()
        gc.collect()
        print(f"[{image_number}/{len(index)}] {image_id} calibration coverage "
              f"({time.time() - started:.1f}s)")

    store_after = _store_state(store, image_ids)
    if store_after != store_before:
        changed = sorted(set(store_before) ^ set(store_after) | {
            path for path in set(store_before) & set(store_after)
            if store_before[path] != store_after[path]
        })
        raise RuntimeError(f"store file state changed during analysis: {changed[:5]}")
    meta_aggregate_after = _meta_aggregate(store)
    if meta_aggregate_after != meta_aggregate_before:
        raise CoverageError(
            "canonical store metadata aggregate changed during analysis")
    for snapshot in (trace_snapshot, results_snapshot,
                     index_snapshot, spec_snapshot):
        _assert_snapshot_unchanged(snapshot)

    summaries = _summary(coverage_rows)
    all_rho = np.asarray([r["spearman_all_tokens"] for r in rank_rows])
    normal_rho = np.asarray([r["spearman_normal_tokens"] for r in rank_rows])
    exact_all = np.asarray([
        r["exact_rank_fraction_all_tokens"] for r in rank_rows])
    exact_normal = np.asarray([
        r["exact_rank_fraction_normal_tokens"] for r in rank_rows])
    adjacent = np.asarray([
        r["adjacent_nonincreasing_fraction_normal_tokens"]
        for r in rank_rows])
    rank_near_consistent = bool(
        float(normal_rho.min()) >= args.rank_spearman_threshold)
    if not rank_near_consistent:
        raise CoverageError(
            "fresh calibration rank is inconsistent with the stored "
            f"importance order: min normal-token Spearman="
            f"{float(normal_rho.min()):.9f} < "
            f"{args.rank_spearman_threshold:.9f}")

    generation = uuid.uuid4().hex
    for row in coverage_rows:
        row["artifact_set_id"] = generation
    provenance_limitations = [
        ("The 50.8 GB store content aggregate is bound by the preregistered "
         "experiment_spec hash but is not reread in this pass; the observed "
         "canonical meta-only aggregate and all touched-file size/mtime are "
         "verified before and after analysis.")
    ]
    if not results_validation["source_model_runtime_block_present"]:
        provenance_limitations.append(
            "This completed schema-v2 results.json predates structured model/"
            "quantization provenance. Model conditions are cross-checked "
            "indirectly against the preregistered experiment_spec, current "
            "model.py config, every store meta.json, the recorded eval "
            "command, and the model loaded for this analysis. Future "
            "04_eval runs should persist model_id, NF4/double-quant, compute "
            "dtype, eager attention, eval mode, and greedy decoding fields.")

    payload = {
        "schema_version": 1,
        "artifact_set_id": generation,
        "analysis_only": True,
        "coverage_used_for_serving_or_selection": False,
        "raw_calibration_scores_persisted": False,
        "calibration_importance_source": (
            "fresh SparseVLM calibrate_image mean over first calibration "
            "questions; scores are in the current stored-position coordinate"
        ),
        "visionzip_static_pt_loaded": False,
        "visionzip_token_or_chunk_score_used_as_calibration_importance": False,
        "provenance_limitations": provenance_limitations,
        "coverage_definitions": {
            "primary_normal_mass": (
                "selected-chunk calibration importance divided by all visual "
                "calibration importance after excluding row separators from "
                "both numerator and denominator"
            ),
            "secondary_sidecar_union_mass": (
                "importance in selected chunks union all row-separator "
                "positions divided by importance in all visual tokens"
            ),
        },
        "inputs": {
            "run_dir": str(run_dir.resolve()),
            "selection_trace": str(trace_path.resolve()),
            "selection_trace_snapshot": trace_snapshot.public(),
            "results_snapshot": results_snapshot.public(),
            "experiment_spec_snapshot": spec_snapshot.public(),
            "index": str(index_path.resolve()),
            "index_sha256": index_sha,
            "index_snapshot": index_snapshot.public(),
            "store": str(store.resolve()),
            "canonical_store_meta_aggregate": meta_aggregate_before,
            "preregistered_full_store_content_aggregate_sha256":
                CANONICAL_STORE_CONTENT_AGGREGATE_SHA256,
            "full_store_content_rehashed_this_pass": False,
            "limit": args.limit,
            "calibration_questions_per_image": args.calib_questions,
            "evaluation_slice": [args.eval_skip,
                                 args.eval_skip + args.eval_questions],
            "calibration_workload_sha256": calibration_workload_sha,
            "evaluation_workload_sha256": eval_workload_sha,
            "budgets_requested": budgets,
            "method_combinations_analyzed": [
                {"family": family, "budget": budget,
                 "method_key": trace["method_keys"][(family, budget)]}
                for family, budget in trace["combos"]
            ],
            "missing_families_by_budget":
                trace["missing_families_by_budget"],
            "model": runner.model_id,
            "load_4bit": runner.load_4bit,
            "attention_implementation": runner.attn,
            "current_config": {
                "model_id": MODEL_ID,
                "load_4bit": LOAD_4BIT,
                "attention_implementation": ATTN_IMPL,
                "store_dtype": STORE_DTYPE,
                "compute_dtype": COMPUTE_DTYPE,
                "chunk_size": CHUNK_SIZE,
            },
        },
        "validation": {
            "canonical_40_240_protocol": canonical_protocol,
            "calibration_evaluation_disjoint": True,
            "selection_trace_exact_evaluation_workload": True,
            "selection_trace_query_independent_per_image": True,
            "all_methods_separator_sidecar": True,
            "selected_chunks_obey_budget": True,
            "prefix_is_exact_first_k_per_layer": True,
            "all_store_orders_are_valid_permutations": True,
            "stored_separator_mapping_matches_original": True,
            "store_file_size_and_mtime_unchanged": True,
            "canonical_meta_aggregate_before_after_match": True,
            "trace_results_index_spec_snapshots_unchanged": True,
            "results_crosscheck": results_validation,
            "coverage_score_nonnegative_finite": True,
            "rank_spearman_threshold": args.rank_spearman_threshold,
            "rank_order_near_consistent": rank_near_consistent,
            "rank_spearman_gate_passed": True,
            "rank_order_exact_all_layers": bool(np.all(exact_all == 1.0)),
            "rank_order_exact_normal_layers": bool(
                np.all(exact_normal == 1.0)),
        },
        "rank_order_consistency": {
            "note": (
                "Diagnostic comparison of current metadata order against a "
                "fresh stable ranking. It is not used to select chunks."
            ),
            "n_image_layers": len(rank_rows),
            "spearman_all_mean": float(all_rho.mean()),
            "spearman_all_min": float(all_rho.min()),
            "spearman_normal_mean": float(normal_rho.mean()),
            "spearman_normal_min": float(normal_rho.min()),
            "exact_rank_fraction_all_mean": float(exact_all.mean()),
            "exact_rank_fraction_normal_mean": float(exact_normal.mean()),
            "adjacent_nonincreasing_fraction_normal_mean": float(
                adjacent.mean()),
            "per_image_layer": rank_rows,
        },
        "summary": summaries,
        "counts": {
            "images": len(index),
            "layers_per_image": len(rank_rows) // len(index),
            "coverage_csv_rows": len(coverage_rows),
            "trace_rows_used": trace["trace_rows_used"],
            "evaluation_questions": trace["expected_question_count"],
            "calibration_questions": len(calibration_ids),
        },
        "publication": {
            "commit_record": "importance_coverage.json",
            "csv": "importance_coverage.csv",
            "json_published_last": True,
            "writer_lock": args.output_lock_path,
            "overwrite_explicit": bool(args.overwrite),
        },
    }

    csv_fields = list(coverage_rows[0])
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=csv_fields)
    writer.writeheader()
    writer.writerows(coverage_rows)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")
    payload["publication"]["csv_sha256"] = _sha256_bytes(csv_bytes)
    payload["publication"]["csv_bytes"] = len(csv_bytes)
    json_bytes = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8")

    # Close the serialization/publish TOCTOU window as far as ordinary
    # cooperative writers permit.  These checks happen immediately before
    # staging the committed pair.
    for snapshot in (trace_snapshot, results_snapshot,
                     index_snapshot, spec_snapshot):
        _assert_snapshot_unchanged(snapshot)
    if _meta_aggregate(store) != meta_aggregate_before:
        raise CoverageError("store metadata changed before publication")
    if _store_state(store, image_ids) != store_before:
        raise CoverageError("store file state changed before publication")

    _publish_pair(out_csv, csv_bytes, out_json, json_bytes,
                  args.overwrite, generation)
    print(f"wrote {out_csv}")
    print(f"wrote {out_json}")
    for item in summaries:
        print(
            f"{item['method']} {item['budget'] * 100:g}%: "
            f"normal={item['normal_mass_coverage_macro_mean']:.6f}, "
            f"sidecar_union={item['sidecar_union_mass_coverage_macro_mean']:.6f}"
        )


def main():
    args = _build_parser().parse_args()
    paths = _validate_paths(args)
    run_dir = paths[0]
    lock = _OutputLock(run_dir)
    args.output_lock_path = str(lock.path)
    with lock:
        _run_locked(args, paths)


if __name__ == "__main__":
    main()

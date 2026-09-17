#!/usr/bin/env python3
"""Strict, read-only analysis for the MT-GQA full/pilot/smoke runs.

The evaluator publishes one immutable ``images/<image_id>.json`` artifact per
physical image.  This script treats that tree as evidence: it hashes it before
reading, validates the complete dialogue x turn x method matrix, re-scores all
predictions, and verifies the same hashes again before atomically publishing a
new result directory.  It never writes below ``run_root``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import statistics
import tempfile
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "mt-gqa-analysis-v2"
METHOD_KEYS = ("recompute", "fullload", "prefix25", "prefix45")
METHOD_LABELS = {
    "recompute": "ReComp",
    "fullload": "FullLoad",
    "prefix25": "ImageOnly Prefix25",
    "prefix45": "ImageOnly Prefix45",
}
METHOD_ORDER = {key: i for i, key in enumerate(METHOD_KEYS)}
TURNS = (1, 2, 3)
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 1234
PROVENANCE_FIELDS = (
    "dialogues_file_sha256", "source_full_workload_sha256",
    "selected_workload_sha256", "model_revision",
)
REQUIRED_OUTPUTS = (
    "dataset_provenance.json", "dialogues.json", "dataset_stats.json",
    "config.json",
    "raw.jsonl", "per_turn.csv", "per_dialog.csv",
    "quality_by_turn.csv", "latency_by_turn.csv", "io_summary.csv",
    "persistence_overhead.csv", "statistical_analysis.json",
    "validation.json", "README.md",
)


class AnalysisError(RuntimeError):
    """An input violated the frozen MT-GQA analysis contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise AnalysisError(f"run root must be a regular directory: {root}")
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise AnalysisError(f"source run tree contains a symlink: {path}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = {
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "file_count": len(files),
        "total_bytes": sum(row["size_bytes"] for row in files.values()),
        "files": files,
        "manifest_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def _load_regular_json(path: Path) -> Any:
    if not path.is_file() or path.is_symlink():
        raise AnalysisError(f"expected regular JSON file: {path}")
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read JSON: {path}") from exc


def _as_float(value: Any, context: str, *, nonnegative: bool = True) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"invalid number {context}: {value!r}") from exc
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise AnalysisError(f"invalid number {context}: {value!r}")
    return result


def _as_int(value: Any, context: str, *, nonnegative: bool = True) -> int:
    if isinstance(value, bool):
        raise AnalysisError(f"invalid integer {context}: {value!r}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"invalid integer {context}: {value!r}") from exc
    if str(result) != str(value) and not isinstance(value, (int, np.integer)):
        raise AnalysisError(f"non-integral value {context}: {value!r}")
    if nonnegative and result < 0:
        raise AnalysisError(f"negative integer {context}: {value!r}")
    return result


def normalize_answer(value: Any) -> str:
    import re
    text = re.sub(r"[^\w\s]", " ", str(value).lower())
    return " ".join(word for word in text.split()
                    if word not in {"a", "an", "the"})


def gqa_score(prediction: str, gold: Sequence[str]) -> float:
    """Reproduce the runner's legacy prefix-tolerant stored score."""
    if not isinstance(gold, (list, tuple)) or not gold:
        raise AnalysisError("gold must be a nonempty answer list")
    pred = normalize_answer(prediction)
    answer = normalize_answer(gold[0])
    return float(pred == answer or (
        bool(answer) and pred.split()[:len(answer.split())] == answer.split()
    ))


def strict_gqa_score(prediction: str, gold: Sequence[str]) -> float:
    """Primary analysis metric: equality after the frozen GQA normalization."""
    if not isinstance(gold, (list, tuple)) or not gold:
        raise AnalysisError("gold must be a nonempty answer list")
    return float(normalize_answer(prediction) == normalize_answer(gold[0]))


def exact_mcnemar(scores_a: Sequence[float],
                  scores_b: Sequence[float]) -> dict[str, Any]:
    if len(scores_a) != len(scores_b) or not scores_a:
        raise ValueError("McNemar inputs must be nonempty and aligned")
    a_only = b_only = 0
    for a, b in zip(scores_a, scores_b):
        if a not in (0, 0.0, 1, 1.0) or b not in (0, 0.0, 1, 1.0):
            raise ValueError("exact McNemar requires binary scores")
        a_only += int(a == 1 and b == 0)
        b_only += int(a == 0 and b == 1)
    discordant = a_only + b_only
    if discordant:
        tail = sum(math.comb(discordant, i)
                   for i in range(min(a_only, b_only) + 1))
        p_value = min(1.0, float(2 * Fraction(tail, 1 << discordant)))
    else:
        p_value = 1.0
    return {
        "a_only": a_only,
        "b_only": b_only,
        "discordant": discordant,
        "p_value_two_sided_exact": p_value,
    }


def dialogue_cluster_bootstrap(
        values: np.ndarray, *, n_resamples: int = BOOTSTRAP_RESAMPLES,
        seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Bootstrap a ``dialogue x method x metric`` matrix with shared draws."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] != 4:
        raise ValueError("values must have shape [dialogs, 4 methods, metrics]")
    if not np.isfinite(values).all() or n_resamples < 1:
        raise ValueError("invalid bootstrap values or resample count")
    rng = np.random.default_rng(seed)
    distribution = np.empty((n_resamples, values.shape[1], values.shape[2]))
    for start in range(0, n_resamples, 256):
        stop = min(start + 256, n_resamples)
        draw = rng.integers(0, values.shape[0],
                            size=(stop - start, values.shape[0]))
        distribution[start:stop] = values[draw].mean(axis=1)
    absolute = {}
    deltas = {}
    full_index = METHOD_ORDER["fullload"]
    for method, method_index in METHOD_ORDER.items():
        absolute[method] = {
            "estimate": values[:, method_index].mean(axis=0).tolist(),
            "ci95_low": np.percentile(
                distribution[:, method_index], 2.5, axis=0).tolist(),
            "ci95_high": np.percentile(
                distribution[:, method_index], 97.5, axis=0).tolist(),
        }
        delta_dist = (distribution[:, method_index]
                      - distribution[:, full_index])
        deltas[method] = {
            "estimate": (values[:, method_index]
                         - values[:, full_index]).mean(axis=0).tolist(),
            "ci95_low": np.percentile(delta_dist, 2.5, axis=0).tolist(),
            "ci95_high": np.percentile(delta_dist, 97.5, axis=0).tolist(),
        }
    return {
        "cluster_unit": "dialogue",
        "n_clusters": values.shape[0],
        "n_resamples": n_resamples,
        "seed": seed,
        "absolute": absolute,
        "delta_vs_fullload": deltas,
    }


def _pick(mapping: Mapping[str, Any], names: Sequence[str], context: str,
          *, required: bool = True, default: Any = None) -> Any:
    found = [(name, mapping[name]) for name in names
             if name in mapping and mapping[name] is not None]
    if not found:
        if required:
            raise AnalysisError(f"missing {context}; accepted fields={names}")
        return default
    canonical = {_stable_json_hash(value): value for _, value in found}
    if len(canonical) != 1:
        raise AnalysisError(f"conflicting aliases for {context}: {found}")
    return found[0][1]


def _extract_persistence(artifact: Mapping[str, Any], image_id: str) -> dict:
    candidates = []
    for key in ("persistence_overhead", "persistence", "persist"):
        if isinstance(artifact.get(key), Mapping):
            candidates.append(artifact[key])
    profile = artifact.get("build_profile")
    if isinstance(profile, Mapping):
        for key in ("persistence_overhead", "persistence", "persist"):
            if isinstance(profile.get(key), Mapping):
                candidates.append(profile[key])
        if any(key in profile for key in (
                "permutation_ms", "kv_repack_ms", "repack_ms",
                "ssd_write_ms", "buffered_write_ms", "persist_ms")):
            candidates.append(profile)
    if len(candidates) != 1:
        raise AnalysisError(
            f"{image_id}: expected exactly one persistence mapping, "
            f"found {len(candidates)}")
    source = candidates[0]
    aliases = {
        "permutation_ms": ("permutation_ms",),
        # ``repack_ms`` in the piggyback writer is the broader
        # materialize+repack interval, whereas ``kv_repack_ms`` is the exact
        # component requested by this table.  Prefer the canonical component
        # when both are present; accept the historical aggregate only as a
        # backwards-compatible fallback.
        "kv_repack_ms": (
            "kv_repack_ms" if source.get("kv_repack_ms") is not None
            else "repack_ms",
        ),
        "ssd_write_ms": ("ssd_write_ms", "buffered_write_ms"),
        "fsync_ms": ("fsync_ms",),
        "total_persist_ms": ("total_persist_ms", "persist_ms"),
        "bytes_written": (
            "bytes_written", "ssd_bytes_written", "total_ssd_write_bytes"),
    }
    row: dict[str, Any] = {"image_id": image_id}
    for target, names in aliases.items():
        raw = _pick(source, names, f"persistence/{target}")
        row[target] = (_as_int(raw, target) if target == "bytes_written"
                       else _as_float(raw, target))
    component_sum = (row["permutation_ms"] + row["kv_repack_ms"]
                     + row["ssd_write_ms"] + row["fsync_ms"])
    if row["total_persist_ms"] + 1e-6 < component_sum:
        raise AnalysisError(f"{image_id}: persistence total below components")
    return row


def _validate_row(raw: Mapping[str, Any], artifact: Mapping[str, Any],
                  artifact_image: str, config: Mapping[str, Any]) -> dict:
    identity = f"{raw.get('dialog_id')}/{raw.get('turn_id')}/{raw.get('method_key')}"
    required = (
        "dialog_id", "image_id", "turn_id", "question_id", "method_key",
        "prediction", "gold", "score", "end_to_end_ttft_ms",
        "request_e2e_ms", "decode_ms", "ssd_read_bytes", "ssd_read_ms",
        "normal_kv_preads", "separator_preads", "scatter_ms",
        "first_k_planning_ms", "vision_forward_count",
        "first_token_id",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise AnalysisError(f"{identity}: missing row fields {missing}")
    row = dict(raw)
    row["dialog_id"] = str(row["dialog_id"])
    row["image_id"] = str(row["image_id"])
    row["question_id"] = str(row["question_id"])
    row["method_key"] = str(row["method_key"])
    row["turn_id"] = _as_int(row["turn_id"], f"{identity}/turn")
    if row["image_id"] != artifact_image:
        raise AnalysisError(f"{identity}: row/artifact image mismatch")
    if row["turn_id"] not in TURNS or row["method_key"] not in METHOD_KEYS:
        raise AnalysisError(f"{identity}: invalid turn or method")
    if not isinstance(row["prediction"], str):
        raise AnalysisError(f"{identity}: prediction is not a string")
    if not isinstance(row["gold"], list) or not row["gold"]:
        raise AnalysisError(f"{identity}: gold is not a nonempty list")
    row["gold"] = [str(value) for value in row["gold"]]
    cache_hit = _pick(row, ("cache_hit", "used_by_request"),
                      f"{identity}/cache flag")
    if not isinstance(cache_hit, bool):
        raise AnalysisError(f"{identity}: cache flag must be boolean")
    row["cache_hit"] = cache_hit
    numeric = (
        "score", "end_to_end_ttft_ms", "request_e2e_ms", "decode_ms",
        "ssd_read_ms", "scatter_ms", "first_k_planning_ms",
    )
    for field in numeric:
        row[field] = _as_float(row[field], f"{identity}/{field}")
    for field in ("ssd_read_bytes", "normal_kv_preads", "separator_preads",
                  "vision_forward_count", "first_token_id"):
        row[field] = _as_int(row[field], f"{identity}/{field}")
    expected_legacy_score = gqa_score(row["prediction"], row["gold"])
    if abs(row["score"] - expected_legacy_score) > 1e-12:
        raise AnalysisError(
            f"{identity}: stored score {row['score']} != "
            f"{expected_legacy_score}")
    if row.get("quality_score") is not None:
        legacy_quality_score = _as_float(
            row["quality_score"], f"{identity}/quality_score")
        if abs(legacy_quality_score - expected_legacy_score) > 1e-12:
            raise AnalysisError(
                f"{identity}: stored quality_score {legacy_quality_score} != "
                f"{expected_legacy_score}")
        row["legacy_quality_score"] = legacy_quality_score
    row["stored_legacy_score"] = row["score"]
    row["legacy_recomputed_score"] = expected_legacy_score
    primary_score = strict_gqa_score(row["prediction"], row["gold"])
    row["score"] = primary_score
    row["quality_score"] = primary_score
    row["recomputed_score"] = primary_score
    row["scorer_disagreement"] = bool(primary_score != expected_legacy_score)
    if not row["end_to_end_ttft_ms"] < row["request_e2e_ms"]:
        raise AnalysisError(f"{identity}: TTFT must be below request E2E")
    timing_error = abs(row["request_e2e_ms"]
                       - row["end_to_end_ttft_ms"] - row["decode_ms"])
    row["e2e_identity_error_ms"] = timing_error
    if timing_error > max(5.0, 0.05 * row["request_e2e_ms"]):
        raise AnalysisError(f"{identity}: E2E != TTFT + decode")

    # These fields are related but not aliases.  The first is the exact
    # fraction of normal SSD chunks read (the experiment's budget contract),
    # while ``selected_kv_ratio`` can be larger because it includes always-on
    # separator rows.  Prefer the physical chunk ratio and use historical
    # fields only when that canonical measurement is absent.
    ratio = row.get("actual_selected_normal_chunk_fraction")
    if ratio is None:
        ratio = row.get("selected_visual_kv_ratio")
    if ratio is None:
        ratio = row.get("selected_kv_ratio")
    row["selected_ratio"] = (None if ratio is None else
                             _as_float(ratio, f"{identity}/selected ratio"))
    if row["selected_ratio"] is not None and row["selected_ratio"] > 1.0:
        raise AnalysisError(f"{identity}: selected ratio exceeds one")

    method, turn = row["method_key"], row["turn_id"]
    if turn == 1:
        if row["vision_forward_count"] != 1 or row["ssd_read_bytes"] != 0 \
                or row["cache_hit"]:
            raise AnalysisError(f"{identity}: Turn 1 must be pixel cache miss")
    elif method == "recompute":
        if row["vision_forward_count"] != 1 or row["ssd_read_bytes"] != 0 \
                or row["cache_hit"]:
            raise AnalysisError(f"{identity}: ReComp semantics violated")
    else:
        if row["vision_forward_count"] != 0 or row["ssd_read_bytes"] <= 0 \
                or not row["cache_hit"]:
            raise AnalysisError(f"{identity}: cache-hit semantics violated")

    for key in PROVENANCE_FIELDS:
        expected = config.get(key)
        values = [container.get(key) for container in (artifact, row)
                  if container.get(key) is not None]
        # The immutable workload hashes are deliberately repeated in every
        # artifact/row.  Model revision is a run-level invariant in runner 37;
        # accept it from config while still rejecting any conflicting copy.
        copies_required = key != "model_revision"
        if (expected is None or (copies_required and not values)
                or any(value != expected for value in values)):
            raise AnalysisError(f"{identity}: missing/conflicting provenance {key}")
        row[key] = expected
    return row


def load_and_validate(run_root: Path, expected_dialogs: int) -> tuple[
        list[dict], list[dict], dict, dict]:
    config = _load_regular_json(run_root / "config.json")
    if not isinstance(config, Mapping):
        raise AnalysisError("config.json is not an object")
    for key in PROVENANCE_FIELDS:
        if not isinstance(config.get(key), str) or not config[key]:
            raise AnalysisError(f"config missing {key}")
    image_dir = run_root / "images"
    if not image_dir.is_dir() or image_dir.is_symlink():
        raise AnalysisError(f"missing regular images directory: {image_dir}")
    paths = sorted(image_dir.glob("*.json"))
    if not paths:
        raise AnalysisError("no image artifacts found")
    rows: list[dict] = []
    persistence: list[dict] = []
    artifact_hashes = {}
    for path in paths:
        artifact = _load_regular_json(path)
        if not isinstance(artifact, Mapping) or not isinstance(
                artifact.get("rows"), list):
            raise AnalysisError(f"bad image artifact: {path}")
        if artifact.get("validation", {}).get("passed") is not True:
            raise AnalysisError(f"image artifact validation is not passed: {path}")
        image_id = str(artifact.get("image_id", ""))
        if not image_id or path.stem != image_id:
            raise AnalysisError(f"artifact filename/image ID mismatch: {path}")
        if "artifact_content_sha256" in artifact:
            content = {key: value for key, value in artifact.items()
                       if key != "artifact_content_sha256"}
            if _stable_json_hash(content) != artifact["artifact_content_sha256"]:
                raise AnalysisError(f"artifact content hash mismatch: {path}")
        artifact_hashes[image_id] = _sha256_file(path)
        persistence.append(_extract_persistence(artifact, image_id))
        rows.extend(_validate_row(raw, artifact, image_id, config)
                    for raw in artifact["rows"])

    expected_rows = expected_dialogs * len(TURNS) * len(METHOD_KEYS)
    if len(rows) != expected_rows:
        raise AnalysisError(f"row count {len(rows)} != {expected_rows}")
    matrix: dict[tuple[str, int, str], dict] = {}
    dialogs: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["dialog_id"], row["turn_id"], row["method_key"])
        if key in matrix:
            raise AnalysisError(f"duplicate matrix cell: {key}")
        matrix[key] = row
        dialogs[row["dialog_id"]].append(row)
    if len(dialogs) != expected_dialogs:
        raise AnalysisError(f"dialog count {len(dialogs)} != {expected_dialogs}")
    expected_cells = {(turn, method) for turn in TURNS for method in METHOD_KEYS}
    position_counts = {method: [0] * len(METHOD_KEYS) for method in METHOD_KEYS}
    for dialog_id, selected in dialogs.items():
        cells = {(row["turn_id"], row["method_key"]) for row in selected}
        if cells != expected_cells:
            raise AnalysisError(f"{dialog_id}: incomplete method-turn matrix")
        if len({row["image_id"] for row in selected}) != 1:
            raise AnalysisError(f"{dialog_id}: multiple images")
        dialogue_orders = []
        for turn in TURNS:
            turn_rows = [row for row in selected if row["turn_id"] == turn]
            if len({row["question_id"] for row in turn_rows}) != 1 \
                    or len({_stable_json_hash(row["gold"])
                            for row in turn_rows}) != 1:
                raise AnalysisError(f"{dialog_id}/T{turn}: method inputs differ")
            prompt_hashes = {row.get("prompt_sha256") for row in turn_rows}
            history_hashes = {row.get("text_history_sha256") for row in turn_rows}
            if None in prompt_hashes or len(prompt_hashes) != 1 \
                    or None in history_hashes or len(history_hashes) != 1:
                raise AnalysisError(f"{dialog_id}/T{turn}: prompt/history differs")
            orders = {_stable_json_hash(row.get("method_order"))
                      for row in turn_rows}
            if len(orders) != 1:
                raise AnalysisError(f"{dialog_id}/T{turn}: method order differs")
            order = list(turn_rows[0].get("method_order", []))
            if len(order) != 4 or set(order) != set(METHOD_KEYS):
                raise AnalysisError(f"{dialog_id}/T{turn}: invalid method order")
            for row in turn_rows:
                if row.get("method_order_position") != order.index(
                        row["method_key"]):
                    raise AnalysisError(
                        f"{dialog_id}/T{turn}: method position mismatch")
            dialogue_orders.append(order)
            ratios = {row["method_key"]: row["selected_ratio"]
                      for row in turn_rows}
            if turn > 1:
                if any(ratios[key] is None for key in
                       ("fullload", "prefix25", "prefix45")):
                    raise AnalysisError(f"{dialog_id}/T{turn}: missing ratios")
                if not (0 < ratios["prefix25"] < ratios["prefix45"]
                        <= ratios["fullload"] <= 1.0):
                    raise AnalysisError(f"{dialog_id}/T{turn}: ratio order invalid")
        if any(order != dialogue_orders[0] for order in dialogue_orders[1:]):
            raise AnalysisError(f"{dialog_id}: method order changes by turn")
        for position, method in enumerate(dialogue_orders[0]):
            position_counts[method][position] += 1
        qids = [next(row["question_id"] for row in selected
                     if row["turn_id"] == turn) for turn in TURNS]
        if len(set(qids)) != 3:
            raise AnalysisError(f"{dialog_id}: repeated question across turns")
    if any(max(counts) - min(counts) > 1 for counts in position_counts.values()):
        raise AnalysisError(f"method rotation is not position-balanced: {position_counts}")
    rows.sort(key=lambda row: (
        row["dialog_id"], row["turn_id"], METHOD_ORDER[row["method_key"]]))
    return rows, persistence, dict(config), {
        "image_artifact_hashes": artifact_hashes,
        "n_images": len(paths), "n_dialogs": len(dialogs),
        "n_rows": len(rows),
        "method_position_counts": position_counts,
    }


def quality_metric_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize prefix-only wins without changing immutable source artifacts."""
    different = [row for row in rows if row["scorer_disagreement"]]
    return {
        "primary_metric": "normalized_exact_match",
        "legacy_source_metric": "normalized_gold_prefix_match",
        "normalization": (
            "lowercase; replace punctuation with spaces; remove a/an/the; "
            "collapse whitespace; compare against the first gold answer"),
        "legacy_source_behavior": (
            "The stored runner score also accepts a prediction whose first "
            "normalized words equal the complete normalized gold answer."),
        "source_score_fields_validated": ["score", "quality_score_if_present"],
        "analyzed_score_fields": ["score", "quality_score", "recomputed_score"],
        "legacy_preservation_fields": [
            "stored_legacy_score", "legacy_quality_score (if source present)",
            "legacy_recomputed_score"],
        "n_rows": len(rows),
        "discrepancy_count": len(different),
        "discrepancy_fraction": len(different) / len(rows),
        "discrepancy_count_by_method": {
            method: sum(row["method_key"] == method for row in different)
            for method in METHOD_KEYS},
        "discrepancy_count_by_turn": {
            f"turn{turn}": sum(row["turn_id"] == turn for row in different)
            for turn in TURNS},
        "examples": [{
            "dialog_id": row["dialog_id"],
            "turn_id": row["turn_id"],
            "method_key": row["method_key"],
            "prediction": row["prediction"],
            "gold": row["gold"],
            "stored_legacy_score": row["stored_legacy_score"],
            "strict_score": row["score"],
        } for row in different[:10]],
    }


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    return {"mean": statistics.fmean(values),
            "p50": _percentile(values, 50),
            "p95": _percentile(values, 95)}


def build_tables(rows: list[dict], persistence: list[dict]) -> dict[str, Any]:
    grouped = {(row["dialog_id"], row["turn_id"], row["method_key"]): row
               for row in rows}
    dialog_ids = sorted({row["dialog_id"] for row in rows})
    per_turn = []
    for row in rows:
        per_turn.append({key: row.get(key) for key in (
            "dialog_id", "image_id", "turn_id", "question_id", "method_key",
            "prediction", "score", "stored_legacy_score",
            "scorer_disagreement", "end_to_end_ttft_ms", "request_e2e_ms",
            "decode_ms", "ssd_read_bytes", "ssd_read_ms", "normal_kv_preads",
            "separator_preads", "scatter_ms", "first_k_planning_ms",
            "selected_ratio", "vision_forward_count", "cache_hit")})

    per_dialog = []
    score_values = np.empty((len(dialog_ids), 4, 4), dtype=float)
    for di, dialog_id in enumerate(dialog_ids):
        for method, mi in METHOD_ORDER.items():
            turn_rows = [grouped[(dialog_id, turn, method)] for turn in TURNS]
            scores = [row["score"] for row in turn_rows]
            score_values[di, mi] = scores + [statistics.fmean(scores)]
            hits = turn_rows[1:]
            per_dialog.append({
                "dialog_id": dialog_id,
                "image_id": turn_rows[0]["image_id"],
                "method_key": method,
                "acc1": scores[0], "acc2": scores[1], "acc3": scores[2],
                "avg": statistics.fmean(scores),
                "cache_hit_ttft_ms_mean": statistics.fmean(
                    row["end_to_end_ttft_ms"] for row in hits),
                "cache_hit_e2e_ms_mean": statistics.fmean(
                    row["request_e2e_ms"] for row in hits),
                "cache_hit_ssd_read_bytes_mean": statistics.fmean(
                    row["ssd_read_bytes"] for row in hits),
            })

    quality = []
    full = score_values[:, METHOD_ORDER["fullload"]]
    for method, mi in METHOD_ORDER.items():
        values = score_values[:, mi]
        quality.append({
            "method_key": method, "method": METHOD_LABELS[method],
            "acc1": values[:, 0].mean(), "acc2": values[:, 1].mean(),
            "acc3": values[:, 2].mean(), "avg": values[:, 3].mean(),
            "acc2_minus_acc1": (values[:, 1] - values[:, 0]).mean(),
            "acc3_minus_acc1": (values[:, 2] - values[:, 0]).mean(),
            "gap_vs_fullload_t1": (values[:, 0] - full[:, 0]).mean(),
            "gap_vs_fullload_t2": (values[:, 1] - full[:, 1]).mean(),
            "gap_vs_fullload_t3": (values[:, 2] - full[:, 2]).mean(),
            "gap_vs_fullload_avg": (values[:, 3] - full[:, 3]).mean(),
            "gap_growth_t3_minus_t1": (
                values[:, 2] - full[:, 2] - values[:, 0] + full[:, 0]).mean(),
        })

    latency = []
    io_rows = []
    for method in METHOD_KEYS:
        selected = [row for row in rows if row["method_key"] == method]
        latency_row: dict[str, Any] = {
            "method_key": method, "method": METHOD_LABELS[method]}
        for turn in TURNS:
            turn_rows = [row for row in selected if row["turn_id"] == turn]
            ttft = _stats([row["end_to_end_ttft_ms"] for row in turn_rows])
            e2e = _stats([row["request_e2e_ms"] for row in turn_rows])
            for stat, value in ttft.items():
                latency_row[f"turn{turn}_ttft_ms_{stat}"] = value
            latency_row[f"turn{turn}_e2e_ms_mean"] = e2e["mean"]
        hits = [row for row in selected if row["turn_id"] in (2, 3)]
        hit_ttft = _stats([row["end_to_end_ttft_ms"] for row in hits])
        hit_e2e = _stats([row["request_e2e_ms"] for row in hits])
        latency_row.update({
            "cache_hit_t2_t3_ttft_ms_mean": hit_ttft["mean"],
            "cache_hit_t2_t3_ttft_ms_p50": hit_ttft["p50"],
            "cache_hit_t2_t3_ttft_ms_p95": hit_ttft["p95"],
            "cache_hit_t2_t3_e2e_ms_mean": hit_e2e["mean"],
        })
        latency.append(latency_row)
        io_rows.append({
            "method_key": method, "method": METHOD_LABELS[method],
            "cache_hit_requests": len(hits),
            "ssd_read_mb_per_cache_hit": statistics.fmean(
                row["ssd_read_bytes"] for row in hits) / 1_000_000,
            "ssd_read_ms_mean": statistics.fmean(row["ssd_read_ms"] for row in hits),
            "normal_kv_preads_mean": statistics.fmean(
                row["normal_kv_preads"] for row in hits),
            "separator_preads_mean": statistics.fmean(
                row["separator_preads"] for row in hits),
            "scatter_ms_mean": statistics.fmean(row["scatter_ms"] for row in hits),
            "first_k_planning_ms_mean": statistics.fmean(
                row["first_k_planning_ms"] for row in hits),
            "selected_ratio_mean": (None if method == "recompute" else
                statistics.fmean(row["selected_ratio"] for row in hits)),
        })
    latency_by_method = _by_key(latency)
    io_by_method = _by_key(io_rows)
    recomp_ttft = latency_by_method["recompute"][
        "cache_hit_t2_t3_ttft_ms_mean"]
    full_bytes = io_by_method["fullload"]["ssd_read_mb_per_cache_hit"]
    for row in latency:
        row["cache_hit_ttft_reduction_vs_recomp_pct"] = (
            100.0 * (recomp_ttft - row["cache_hit_t2_t3_ttft_ms_mean"])
            / recomp_ttft)
    for row in io_rows:
        value = row["ssd_read_mb_per_cache_hit"]
        row["ssd_reduction_vs_fullload_pct"] = (
            None if full_bytes == 0 else 100.0 * (full_bytes - value) / full_bytes)
    return {
        "per_turn": per_turn, "per_dialog": per_dialog, "quality": quality,
        "latency": latency, "io": io_rows, "persistence": persistence,
        "score_values": score_values, "dialog_ids": dialog_ids,
    }


def build_statistics(rows: list[dict], tables: Mapping[str, Any],
                     *, n_resamples: int, seed: int) -> dict:
    values = tables["score_values"]
    boot = dialogue_cluster_bootstrap(
        values, n_resamples=n_resamples, seed=seed)
    mcnemar = {}
    dialog_ids = tables["dialog_ids"]
    lookup = {(row["dialog_id"], row["turn_id"], row["method_key"]): row
              for row in rows}
    for method in METHOD_KEYS:
        if method == "fullload":
            continue
        mcnemar[method] = {}
        for turn in TURNS:
            a = [lookup[(dialog, turn, method)]["score"] for dialog in dialog_ids]
            b = [lookup[(dialog, turn, "fullload")]["score"]
                 for dialog in dialog_ids]
            mcnemar[method][f"turn{turn}"] = exact_mcnemar(a, b)
    return {
        "schema_version": SCHEMA_VERSION,
        "metric_order": ["Acc1", "Acc2", "Acc3", "Avg"],
        "bootstrap": boot,
        "mcnemar_vs_fullload": mcnemar,
    }


def build_sanity(rows: Sequence[Mapping[str, Any]],
                 tables: Mapping[str, Any]) -> dict[str, Any]:
    """FullLoad numerical/layout sanity plus four-arm Turn-1 fairness."""
    dialog_ids = list(tables["dialog_ids"])
    lookup = {(row["dialog_id"], row["turn_id"], row["method_key"]): row
              for row in rows}
    by_turn = {}
    full_score_deltas = []
    for turn in TURNS:
        pairs = [(lookup[(dialog, turn, "recompute")],
                  lookup[(dialog, turn, "fullload")])
                 for dialog in dialog_ids]
        accuracy_gap = statistics.fmean(
            full["score"] - recomp["score"] for recomp, full in pairs)
        full_score_deltas.append(accuracy_gap)
        by_turn[f"turn{turn}"] = {
            "n_dialogs": len(pairs),
            "prediction_agreement_fraction": statistics.fmean(
                recomp["prediction"] == full["prediction"]
                for recomp, full in pairs),
            "first_token_agreement_fraction": statistics.fmean(
                recomp["first_token_id"] == full["first_token_id"]
                for recomp, full in pairs),
            "accuracy_gap_fullload_minus_recomp": accuracy_gap,
        }
    turn1 = {}
    for method in METHOD_KEYS:
        pairs = [(lookup[(dialog, 1, "recompute")],
                  lookup[(dialog, 1, method)]) for dialog in dialog_ids]
        turn1[method] = {
            "prediction_agreement_vs_recomp_fraction": statistics.fmean(
                recomp["prediction"] == candidate["prediction"]
                for recomp, candidate in pairs),
            "first_token_agreement_vs_recomp_fraction": statistics.fmean(
                recomp["first_token_id"] == candidate["first_token_id"]
                for recomp, candidate in pairs),
            "accuracy_gap_vs_recomp": statistics.fmean(
                candidate["score"] - recomp["score"]
                for recomp, candidate in pairs),
        }
    return {
        "recomp_vs_fullload_by_turn": by_turn,
        "recomp_vs_fullload_avg_accuracy_gap": statistics.fmean(
            full_score_deltas),
        "turn1_four_arm_fairness": turn1,
    }


def _by_key(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row["method_key"]): row for row in rows}


def build_conclusions(tables: Mapping[str, Any], config: Mapping[str, Any]) -> dict:
    quality = _by_key(tables["quality"])
    latency = _by_key(tables["latency"])
    io_rows = _by_key(tables["io"])
    quality_ok = {}
    efficiency_ok = {}
    for method in ("prefix25", "prefix45"):
        quality_ok[method] = (
            quality[method]["gap_vs_fullload_avg"] >= -0.02
            and quality[method]["gap_growth_t3_minus_t1"] >= -0.02)
        efficiency_ok[method] = (
            latency[method]["cache_hit_t2_t3_ttft_ms_mean"]
            < latency["recompute"]["cache_hit_t2_t3_ttft_ms_mean"]
            and io_rows[method]["ssd_read_mb_per_cache_hit"]
            < io_rows["fullload"]["ssd_read_mb_per_cache_hit"])
    quality_verdict = ("SUPPORTED" if all(quality_ok.values()) else
                       "PARTIALLY SUPPORTED" if any(quality_ok.values()) else
                       "NOT SUPPORTED")
    efficiency_verdict = ("SUPPORTED" if all(efficiency_ok.values()) else
                          "PARTIALLY SUPPORTED" if any(efficiency_ok.values()) else
                          "NOT SUPPORTED")

    def reduction(method: str, baseline: str, field: str) -> float:
        base = latency[baseline][field]
        return 100.0 * (base - latency[method][field]) / base

    q = {
        "Q1": (f"{config.get('benchmark_type', config.get('dataset', 'unknown'))}; "
               "source and hashes are "
               "recorded in the immutable run config."),
        "Q2": ("Three questions from one image; Turn 2 receives Q1/gold A1/Q2 "
               "and Turn 3 receives Q1/gold A1/Q2/gold A2/Q3."),
        "Q3": {method: {key: quality[method][key]
                         for key in ("acc1", "acc2", "acc3", "avg")}
               for method in METHOD_KEYS},
        "Q4": {"gap_growth_t3_minus_t1":
               quality["prefix25"]["gap_growth_t3_minus_t1"],
               "gap_worsened": bool(
                   quality["prefix25"]["gap_growth_t3_minus_t1"] < 0)},
        "Q5": {"gap_growth_t3_minus_t1":
               quality["prefix45"]["gap_growth_t3_minus_t1"],
               "gap_worsened": bool(
                   quality["prefix45"]["gap_growth_t3_minus_t1"] < 0)},
        "Q6": f"Future-query robustness verdict: {quality_verdict}.",
        "Q7": {"prefix25_ttft_reduction_vs_recomp_pct": reduction(
            "prefix25", "recompute", "cache_hit_t2_t3_ttft_ms_mean")},
        "Q8": {"prefix45_ttft_reduction_vs_recomp_pct": reduction(
            "prefix45", "recompute", "cache_hit_t2_t3_ttft_ms_mean")},
        "Q9": {"fullload_is_faster_than_recomp": bool(
               latency["fullload"]["cache_hit_t2_t3_ttft_ms_mean"]
               < latency["recompute"]["cache_hit_t2_t3_ttft_ms_mean"])},
        "Q10": {"partial_loading_needed_for_best_measured_ttft": bool(min(
            ("fullload", "prefix25", "prefix45"),
            key=lambda method: latency[method]["cache_hit_t2_t3_ttft_ms_mean"]
        ) != "fullload")},
    }
    return {
        "quality_verdict": quality_verdict,
        "efficiency_verdict": efficiency_verdict,
        "quality_rule": ("SUPPORTED iff both prefixes have Avg gap >= -2pp and "
                         "T3-vs-T1 gap growth >= -2pp; PARTIAL iff one does."),
        "efficiency_rule": ("SUPPORTED iff both prefixes beat ReComp cache-hit "
                            "TTFT and FullLoad SSD bytes; PARTIAL iff one does."),
        "questions": q,
    }


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise AnalysisError("refusing empty CSV")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + "\n"


def load_dataset_evidence(dataset_provenance: Path,
                          run_config: Mapping[str, Any],
                          expected_dialogs: int) -> tuple[dict[str, str], dict]:
    """Load and cross-check the four immutable canonical dataset artifacts."""
    provenance_path = Path(dataset_provenance).resolve()
    if provenance_path.name != "dataset_provenance.json":
        raise AnalysisError("dataset provenance must name dataset_provenance.json")
    directory = provenance_path.parent
    names = ("dataset_provenance.json", "dialogues.json",
             "dataset_stats.json", "config.json")
    paths = {name: directory / name for name in names}
    values = {name: _load_regular_json(path) for name, path in paths.items()}
    texts = {name: path.read_text() for name, path in paths.items()}
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if hashes["dialogues.json"] != run_config["dialogues_file_sha256"]:
        raise AnalysisError("run/dataset dialogues SHA256 mismatch")
    dataset_config = values["config.json"]
    if dataset_config.get("workload_sha256") != run_config[
            "source_full_workload_sha256"]:
        raise AnalysisError("run/dataset full workload SHA256 mismatch")
    if int(dataset_config.get("n_dialogues", -1)) < expected_dialogs:
        raise AnalysisError("canonical dataset is smaller than selected run")
    artifact_hashes = dataset_config.get("artifact_sha256", {})
    for name in ("dialogues.json", "dataset_stats.json",
                 "dataset_provenance.json"):
        if artifact_hashes.get(name) != hashes[name]:
            raise AnalysisError(f"dataset config hash mismatch: {name}")
    benchmark_types = {
        value.get("benchmark_type") for value in values.values()
        if isinstance(value, Mapping)
    }
    if len(benchmark_types) != 1 or None in benchmark_types:
        raise AnalysisError("dataset benchmark_type differs across artifacts")
    return texts, {
        "directory": str(directory),
        "artifact_sha256": hashes,
        "benchmark_type": next(iter(benchmark_types)),
        "full_dialogues": int(dataset_config["n_dialogues"]),
        "full_workload_sha256": dataset_config["workload_sha256"],
        "canonical_dataset_config": dataset_config,
        "canonical_dataset_provenance": values["dataset_provenance.json"],
    }


def _readme(tables: Mapping[str, Any], conclusions: Mapping[str, Any],
            validation: Mapping[str, Any], run_config: Mapping[str, Any],
            sanity: Mapping[str, Any]) -> str:
    quality = _by_key(tables["quality"])
    latency = _by_key(tables["latency"])
    io_rows = _by_key(tables["io"])
    lines = [
        "# MT-GQA Full Analysis", "",
        f"Validated **{validation['n_dialogs']:,} dialogues**, three turns and "
        "four methods per dialogue. Source GQA scores were independently "
        "validated against the original scorer.",
        "", "## Primary quality metric and legacy-score audit", "",
        "All quality tables, confidence intervals, McNemar tests, and verdicts "
        "use **strict normalized exact match**: the normalized prediction "
        "must equal the normalized first gold answer in full. Normalization "
        "lowercases, replaces punctuation with spaces, removes a/an/the, and "
        "collapses whitespace.",
        "", "The original runner stored a **prefix-tolerant** `score` and "
        "`quality_score`: it also counted a longer prediction as correct if "
        "it began with the complete gold answer. The immutable image artifacts "
        "are unchanged. In this derived analysis, `raw.jsonl` uses strict "
        "`score`/`quality_score`, and preserves the source values as "
        "`stored_legacy_score`/`legacy_quality_score`; `recomputed_score` "
        "is strict and `legacy_recomputed_score` is the validated original.",
        "", f"Scorer disagreements: **{validation['quality_metric_audit']['discrepancy_count']:,} "
        f"of {validation['n_rows']:,} requests**. Counts by method: "
        + ", ".join(
            f"{METHOD_LABELS[method]}="
            f"{validation['quality_metric_audit']['discrepancy_count_by_method'][method]}"
            for method in METHOD_KEYS) + ".",
        "", "Representative legacy-correct / strict-incorrect cases:", "",
    ]
    examples = validation["quality_metric_audit"]["examples"]
    if examples:
        for row in examples[:5]:
            lines.append(
                f"- {row['dialog_id']} T{row['turn_id']} "
                f"{METHOD_LABELS[row['method_key']]}: prediction "
                f"`{row['prediction']}` versus gold "
                f"`{row['gold'][0]}`; legacy {row['stored_legacy_score']:g}, "
                f"strict {row['strict_score']:g}.")
    else:
        lines.append("- None in this run.")
    lines += ["", "## Quality", "",
        "| Method | Acc1 | Acc2 | Acc3 | Avg | Δ Avg vs FullLoad |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_KEYS:
        row = quality[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {row['acc1']:.4f} | "
            f"{row['acc2']:.4f} | {row['acc3']:.4f} | {row['avg']:.4f} | "
            f"{row['gap_vs_fullload_avg']:+.4f} |")
    lines += ["", "## Cache-hit system performance (Turns 2–3)", "",
              "| Method | TTFT ms | E2E ms | SSD MB/request |",
              "|---|---:|---:|---:|"]
    for method in METHOD_KEYS:
        lines.append(
            f"| {METHOD_LABELS[method]} | "
            f"{latency[method]['cache_hit_t2_t3_ttft_ms_mean']:.3f} | "
            f"{latency[method]['cache_hit_t2_t3_e2e_ms_mean']:.3f} | "
            f"{io_rows[method]['ssd_read_mb_per_cache_hit']:.3f} |")
    lines += ["", "## Future-query robustness", "",
              "| Method | Gap T1 | Gap T2 | Gap T3 | Gap growth T3−T1 |",
              "|---|---:|---:|---:|---:|"]
    for method in ("prefix25", "prefix45"):
        row = quality[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | "
            f"{row['gap_vs_fullload_t1']:+.4f} | "
            f"{row['gap_vs_fullload_t2']:+.4f} | "
            f"{row['gap_vs_fullload_t3']:+.4f} | "
            f"{row['gap_growth_t3_minus_t1']:+.4f} |")
    lines += ["", "## FullLoad and Turn-1 sanity", "",
              "| Turn | Prediction agreement | First-token agreement | "
              "FullLoad−ReComp Acc |",
              "|---:|---:|---:|---:|"]
    for turn in TURNS:
        row = sanity["recomp_vs_fullload_by_turn"][f"turn{turn}"]
        lines.append(
            f"| {turn} | {row['prediction_agreement_fraction']:.4f} | "
            f"{row['first_token_agreement_fraction']:.4f} | "
            f"{row['accuracy_gap_fullload_minus_recomp']:+.4f} |")
    lines += ["", "Turn 1 four-arm agreement versus ReComp:", ""]
    for method in METHOD_KEYS:
        row = sanity["turn1_four_arm_fairness"][method]
        lines.append(
            f"- {METHOD_LABELS[method]}: prediction "
            f"{row['prediction_agreement_vs_recomp_fraction']:.4f}, first-token "
            f"{row['first_token_agreement_vs_recomp_fraction']:.4f}, accuracy gap "
            f"{row['accuracy_gap_vs_recomp']:+.4f}")
    lines += ["", "## Direct answers", ""]
    for key, value in conclusions["questions"].items():
        lines += [f"### {key}", "", json.dumps(value, ensure_ascii=False), ""]
    lines += [
        "## Verdict", "",
        f"- MT-GQA Quality: **{conclusions['quality_verdict']}**",
        f"- MT-GQA Efficiency: **{conclusions['efficiency_verdict']}**",
        "", conclusions["quality_rule"], conclusions["efficiency_rule"], "",
        "## Protocol and storage conditions", "",
        str(validation["dataset_evidence"]["canonical_dataset_provenance"].get(
            "disclaimer", "MT-GQA-reconstructed; exact official identity is not claimed.")),
        "",
        "- History: gold teacher-forced; no generated answer is fed to a later turn.",
        "- Turn 1: every arm performs normal pixel inference (vision forward 1, SSD read 0).",
        "- Persistence is a one-time post-Turn-1 cost and is not added to cache-hit TTFT.",
        "- Cache hits use OS-page-cache-cold buffered `pread`; O_DIRECT is false.",
        "- SSD controller-cache flush is false/not performed.",
        f"- Run cache condition: `{run_config.get('cache_condition', 'OS-page-cache-cold')}`.",
        f"- Main TTFT field: `{run_config.get('main_ttft_field', 'end_to_end_ttft_ms')}`.",
        "",
        "## Artifacts", "",
    ]
    lines.extend(f"- `{name}`" for name in REQUIRED_OUTPUTS if name != "README.md")
    return "\n".join(lines) + "\n"


def analyze(run_root: Path, results_root: Path, *, expected_dialogs: int,
            bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
            bootstrap_seed: int = BOOTSTRAP_SEED,
            dataset_provenance: Path | None = None) -> dict:
    run_root, results_root = Path(run_root).resolve(), Path(results_root).resolve()
    if expected_dialogs < 1:
        raise AnalysisError("expected dialogs must be positive")
    if bootstrap_resamples != BOOTSTRAP_RESAMPLES:
        raise AnalysisError("MT-GQA requires exactly 10,000 bootstrap resamples")
    if bootstrap_seed != BOOTSTRAP_SEED:
        raise AnalysisError("MT-GQA requires bootstrap seed 1234")
    if results_root.exists():
        raise FileExistsError(f"result destination already exists: {results_root}")
    if run_root == results_root or run_root in results_root.parents \
            or results_root in run_root.parents:
        raise AnalysisError("run and result roots may not overlap")
    before = source_manifest(run_root)
    rows, persistence, config, evidence = load_and_validate(
        run_root, expected_dialogs)
    dataset_path = (Path(dataset_provenance) if dataset_provenance is not None
                    else ROOT / "data/mt_gqa/dataset_provenance.json")
    dataset_texts, dataset_evidence = load_dataset_evidence(
        dataset_path, config, expected_dialogs)
    metric_audit = quality_metric_audit(rows)
    tables = build_tables(rows, persistence)
    statistics_doc = build_statistics(
        rows, tables, n_resamples=bootstrap_resamples, seed=bootstrap_seed)
    sanity = build_sanity(rows, tables)
    statistics_doc["full_load_and_turn1_sanity"] = sanity
    statistics_doc["quality_metric_audit"] = metric_audit
    conclusions = build_conclusions(tables, config)
    statistics_doc["conclusions"] = conclusions
    validation = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "expected_dialogs": expected_dialogs,
        "n_dialogs": evidence["n_dialogs"],
        "n_images": evidence["n_images"],
        "n_rows": evidence["n_rows"],
        "expected_rows": expected_dialogs * 12,
        "turns_per_dialog": 3,
        "method_keys": list(METHOD_KEYS),
        "gqa_scores_recomputed": True,
        "primary_quality_metric": "normalized_exact_match",
        "quality_metric_audit": metric_audit,
        "complete_dialogue_turn_method_matrix": True,
        "method_position_counts": evidence["method_position_counts"],
        "method_rotation_position_balanced": True,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "source_manifest_before": before,
        "provenance": {key: config.get(key) for key in (
            "benchmark_type", "dataset", *PROVENANCE_FIELDS)},
        "image_artifact_hashes": evidence["image_artifact_hashes"],
        "dataset_evidence": dataset_evidence,
        "conclusions": conclusions,
        "full_load_and_turn1_sanity": sanity,
    }
    result_config = dict(config)
    result_config.update({
        "analysis_schema_version": SCHEMA_VERSION,
        "quality_metric": "normalized_exact_match",
        "source_legacy_quality_metric": "normalized_gold_prefix_match",
        "analysis_expected_dialogs": expected_dialogs,
        "analysis_bootstrap_resamples": bootstrap_resamples,
        "analysis_bootstrap_seed": bootstrap_seed,
        "canonical_dataset_artifact_sha256": dataset_evidence[
            "artifact_sha256"],
        "canonical_dataset_config": dataset_evidence[
            "canonical_dataset_config"],
    })
    payloads = {
        **dataset_texts,
        "config.json": _json_text(result_config),
        "raw.jsonl": "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            for row in rows),
        "per_turn.csv": _csv_text(tables["per_turn"]),
        "per_dialog.csv": _csv_text(tables["per_dialog"]),
        "quality_by_turn.csv": _csv_text(tables["quality"]),
        "latency_by_turn.csv": _csv_text(tables["latency"]),
        "io_summary.csv": _csv_text(tables["io"]),
        "persistence_overhead.csv": _csv_text(tables["persistence"]),
        "statistical_analysis.json": _json_text(statistics_doc),
    }
    validation["source_manifest_after_analysis"] = source_manifest(run_root)
    if validation["source_manifest_after_analysis"] != before:
        raise AnalysisError("source run tree changed during analysis")
    payloads["validation.json"] = _json_text(validation)
    payloads["README.md"] = _readme(
        tables, conclusions, validation, config, sanity)

    results_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(
        prefix=f".{results_root.name}.tmp-", dir=results_root.parent))
    try:
        for name in REQUIRED_OUTPUTS:
            (stage / name).write_text(payloads[name])
        if results_root.exists():
            raise FileExistsError(f"result destination appeared: {results_root}")
        os.replace(stage, results_root)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    after_publish = source_manifest(run_root)
    if after_publish != before:
        raise AnalysisError("source run tree changed while publishing results")
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path,
                        default=ROOT / "runs/mt_gqa_full/full_4061")
    parser.add_argument("--results-root", type=Path,
                        default=ROOT / "results/mt_gqa_full")
    parser.add_argument("--expected-dialogs", type=int, default=4061,
                        help="4061 full, 100 pilot, or 10 smoke")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=1234)
    parser.add_argument("--dataset-provenance", type=Path,
                        default=ROOT / "data/mt_gqa/dataset_provenance.json")
    args = parser.parse_args()
    result = analyze(args.run_root, args.results_root,
                     expected_dialogs=args.expected_dialogs,
                     bootstrap_resamples=args.bootstrap_resamples,
                     bootstrap_seed=args.bootstrap_seed,
                     dataset_provenance=args.dataset_provenance)
    print(json.dumps({key: result[key] for key in (
        "passed", "n_dialogs", "n_images", "n_rows")}, indent=1))


if __name__ == "__main__":
    main()

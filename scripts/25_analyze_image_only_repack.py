#!/usr/bin/env python3
"""Strict CPU-only analysis for the image-only Visual-KV repack study.

The script reads completed schema-v2 runs and independently checks their frozen
GQA-40/questions[4:10] workload, Prefix selection/I/O contracts, image-ingestion
profiles, offline importance coverage, and the separately produced correctness
validation.  Source artifacts are never modified.  Structural/provenance gates
fail closed; a strict behavioral-equivalence failure is instead published as an
explicit warning (never as a pass), because it is itself a result.  Publication
is limited to a new or empty ``results/image_only_repack`` or versioned
``results/image_only_repack_recomp`` directory.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import shlex
import stat
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "data/index.json"
OUTPUT = ROOT / "results/image_only_repack"
RECOMP_OUTPUT = ROOT / "results/image_only_repack_recomp"
DEFAULT_CALIB4_RUN = ROOT / "runs/reorder_prefix_baseline/calib4"

INDEX_SHA256 = "514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a"
WORKLOAD_SHA256 = "97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66"
N_IMAGES = 40
N_QUESTIONS = 240
QUESTIONS_PER_IMAGE = 6
SKIP = 4
N_LAYERS = 32
CHUNK_SIZE = 64
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
FRACTIONS = (0.10, 0.20, 0.25, 0.30, 0.50)
IMPORTANCE_SOURCES = (
    "visionzip_image_saliency",
    "sparsevlm_calib4_analysis_only",
)
BASE_LAYOUTS = (
    "raster",
    "morton",
    "visionzip_image_only",
    "calib4_importance_legacy",
)
MATCHED_LAYOUT = "calib4_importance_sep_tail"

# These are operational, predeclared reporting thresholds, not equivalence
# margins learned from the observed result.  Keeping them here makes the final
# STRONG/PARTIAL/NO-GO decision reproducible and exposes every arbitrary choice.
VERDICT_THRESHOLDS = {
    "strong_max_accuracy_drop_vs_same_layout_fullload_pp": 2.0,
    "strong_max_accuracy_drop_vs_calib4_prefix25_pp": 2.0,
    "strong_min_gain_vs_best_raster_or_morton_pp": 0.0,
    "partial_min_gain_vs_best_raster_or_morton_pp": 2.0,
    "max_ssd_bytes_ratio_vs_repacked_fullload": 0.35,
    "min_ttft_reduction_vs_repacked_fullload_pct": 20.0,
    "max_total_preads_ratio_vs_calib4_prefix25": 1.10,
    "max_selector_mean_ms": 1.0,
    "max_selector_p95_ms": 2.0,
}

ROLE_ORDER = (
    "raster_fullload",
    "recomp",
    "raster_prefix25",
    "morton_prefix25",
    "calib4_prefix25",
    "visionzip_prefix25",
    "visionzip_prefix50",
    "visionzip_fullload_sanity",
    "matched_calib_prefix25",
)

ROLE_META = {
    "raster_fullload": ("Raster", "FullLoad", None, 0, True, False),
    "recomp": (
        "Pixel recomputation (historical)", "ReComp", None, "", True, False),
    "raster_prefix25": ("Raster", "Prefix25", 0.25, 0, True, False),
    "morton_prefix25": ("Morton", "Prefix25", 0.25, 0, True, False),
    "calib4_prefix25": (
        "Calib4 importance legacy", "Prefix25", 0.25, 4, True, False),
    "visionzip_prefix25": (
        "VisionZip image-only", "Prefix25", 0.25, 0, True, False),
    "visionzip_prefix50": (
        "VisionZip image-only", "Prefix50", 0.50, 0, True, False),
    "visionzip_fullload_sanity": (
        "VisionZip image-only", "FullLoad", None, 0, False, True),
    "matched_calib_prefix25": (
        "Calib4 importance separator-tail", "Prefix25", 0.25, 4, False, False),
}

COMMON_RUN_FIELDS = {
    "dataset", "method_key", "method", "retention", "retention_kind", "image_id",
    "question_id", "question", "prediction", "ground_truth", "correct",
    "generated_tokens", "selector_ms", "ssd_read_ms", "scatter_ms",
    "prepare_ms", "prefill_ms", "ttft_ms", "decode_ms", "e2e_latency_ms",
    "ssd_read_bytes", "ssd_read_chunks", "ssd_preads", "n_chunks_selected",
    "n_chunks_total", "touched_chunk_fraction", "normal_chunk_count_total",
    "normal_kv_read_bytes", "separator_read_bytes", "normal_kv_preads",
    "separator_preads", "total_actual_pread_bytes", "static_score_calls",
    "query_score_calls", "diversity_calls", "selection_mode",
    "selected_chunk_ids_per_layer", "separator_policy",
    "reordered_prefix_store_validated",
    "logical_kv_ratio",
}

NEW_RUN_FIELDS = {
    "first_token_id", "physical_layout", "calibration_questions", "retrieval",
    "mean_bytes_per_pread", "normal_mean_bytes_per_pread",
    "separator_mean_bytes_per_pread", "validated_prefix_layout",
}

COVERAGE_FIELDS = [
    "image_id", "layer", "layout", "importance_source",
    "analysis_only_calibration", "physical_fraction_requested",
    "selected_chunk_count", "total_chunk_count", "selected_physical_rows",
    "selected_normal_tokens", "total_normal_tokens",
    "realized_normal_token_fraction", "selected_importance_mass",
    "total_importance_mass", "importance_mass_coverage", "separator_tokens",
    "separator_tail", "prefix_rule", "order_sha256",
]

PROFILE_ROW_FIELDS = {
    "image_id", "layout", "v_token_num", "real_visual_tokens",
    "separator_tokens", "n_chunks", "vision_saliency_ms", "permutation_ms",
    "prefix_forward_ms", "kv_materialize_ms", "kv_repack_ms", "ssd_write_ms",
    "total_ingestion_ms", "mapping_metadata_bytes", "visual_kv_bytes",
    "probe_sidecar_bytes", "separator_sidecar_bytes", "permutation_sha256",
    "inverse_permutation_sha256", "image_input_sha256",
    "image_inputs_match_prompt_processor", "layout_uses_dataset_question",
    "llm_used_for_layout_scoring", "calibration_questions",
    "layout_frozen_before_llm_prefix_forward",
}


class AnalysisError(RuntimeError):
    """A fail-closed input or publication error."""


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _sig(path: Path) -> tuple:
    st = path.lstat()
    return (st.st_dev, st.st_ino, stat.S_IFMT(st.st_mode), st.st_size,
            st.st_mtime_ns)


def _sha256(path: Path, block_size: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def _stable_hash(value) -> str:
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _permutation_sha256(order) -> str:
    payload = ",".join(str(int(i)) for i in order).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _resolve(value: str | Path) -> Path:
    p = Path(value)
    # abspath deliberately does not resolve symlinks.  The component-by-
    # component guard must see them before any canonicalisation occurs.
    return Path(os.path.abspath(os.fspath(p if p.is_absolute() else ROOT / p)))


def _reject_symlinks(path: Path) -> None:
    """Reject an existing symlink component below the repository root."""
    try:
        rel = Path(os.path.abspath(os.fspath(path))).relative_to(ROOT.resolve())
    except ValueError as exc:
        raise AnalysisError(f"path escapes repository: {path}") from exc
    cur = ROOT.resolve()
    for part in rel.parts:
        cur = cur / part
        if _lexists(cur) and cur.is_symlink():
            raise AnalysisError(f"symlink path component is forbidden: {cur}")


def _regular(path: Path) -> None:
    if not _lexists(path) or path.is_symlink() or not path.is_file():
        raise AnalysisError(f"missing/non-regular input: {path}")


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise AnalysisError(f"CSV has no header: {path}")
        fields = list(reader.fieldnames)
        duplicates = sorted(k for k, n in Counter(fields).items() if n > 1)
        if duplicates:
            raise AnalysisError(f"duplicate CSV headers in {path}: {duplicates}")
        return fields, list(reader)


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value) -> None:
    with path.open("w") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def _number(value, name: str, *, blank=False) -> float | None:
    if value in (None, ""):
        if blank:
            return None
        raise AnalysisError(f"missing numeric value: {name}")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"invalid numeric value {name}={value!r}") from exc
    if not math.isfinite(out):
        raise AnalysisError(f"non-finite numeric value {name}={value!r}")
    return out


def _integer(value, name: str, *, blank=False) -> int | None:
    out = _number(value, name, blank=blank)
    if out is None:
        return None
    answer = int(round(out))
    if abs(out - answer) > 1e-9:
        raise AnalysisError(f"non-integral value {name}={value!r}")
    return answer


def _bool(value, name: str) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise AnalysisError(f"invalid boolean {name}={value!r}")


def _json_cell(value, name: str):
    if value in (None, "", "null"):
        return None
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"invalid JSON cell {name}={value!r}") from exc


def _chunk_layers(value, context: str) -> list[list[int]]:
    value = _json_cell(value, context)
    if not isinstance(value, list):
        raise AnalysisError(f"{context}: expected list of layers")
    answer = []
    for li, layer in enumerate(value):
        if not isinstance(layer, list) or any(type(x) is not int for x in layer):
            raise AnalysisError(f"{context}: layer {li} has non-integer IDs")
        if layer != sorted(set(layer)):
            raise AnalysisError(f"{context}: layer {li} is not sorted unique")
        answer.append(layer)
    return answer


def _close(a, b, *, atol=1e-6, rtol=1e-9) -> bool:
    return abs(float(a) - float(b)) <= atol + rtol * abs(float(b))


def _budget_chunks(total: int, budget: float) -> int:
    return max(1, min(total, int(round(total * budget))))


def _percentiles(values) -> tuple[float, float, float]:
    a = np.asarray(values, dtype=float)
    if not len(a) or not np.isfinite(a).all():
        raise AnalysisError("cannot summarize empty/non-finite values")
    return float(a.mean()), float(np.median(a)), float(np.percentile(a, 95))


def _exact_mcnemar(a, b) -> dict:
    aa = np.asarray(a, dtype=float) == 1.0
    bb = np.asarray(b, dtype=float) == 1.0
    a_only = int((aa & ~bb).sum())
    b_only = int((~aa & bb).sum())
    n = a_only + b_only
    if n:
        k = min(a_only, b_only)
        p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    else:
        p = 1.0
    return {
        "a_only": a_only,
        "b_only": b_only,
        "both_correct": int((aa & bb).sum()),
        "neither_correct": int((~aa & ~bb).sum()),
        "discordant": n,
        "p_exact_two_sided": float(p),
    }


def _paired_bootstrap(a, b, images, *, n=BOOTSTRAP_RESAMPLES,
                      seed=BOOTSTRAP_SEED) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) != len(images) or not len(a):
        raise AnalysisError("unaligned paired-bootstrap input")
    by_image = defaultdict(list)
    image_order = []
    for i, image in enumerate(images):
        if image not in by_image:
            image_order.append(image)
        by_image[image].append(i)
    if any(len(by_image[x]) != QUESTIONS_PER_IMAGE for x in image_order):
        raise AnalysisError("image clusters do not each contain six questions")

    rng = np.random.RandomState(seed)
    qi = rng.randint(0, len(a), size=(n, len(a)))
    qdelta = (a[qi] - b[qi]).mean(axis=1)
    am = np.asarray([a[by_image[x]].mean() for x in image_order])
    bm = np.asarray([b[by_image[x]].mean() for x in image_order])
    rng = np.random.RandomState(seed)
    ii = rng.randint(0, len(am), size=(n, len(am)))
    idelta = (am[ii] - bm[ii]).mean(axis=1)

    def ci(x):
        return [float(v) for v in np.percentile(x, (2.5, 97.5))]

    return {
        "n_questions": int(len(a)),
        "n_image_clusters": int(len(image_order)),
        "resamples": int(n),
        "seed": int(seed),
        "a_accuracy": float(a.mean()),
        "b_accuracy": float(b.mean()),
        "delta_pp": float((a - b).mean() * 100),
        "image_cluster_delta_ci95_pp": [x * 100 for x in ci(idelta)],
        "question_delta_ci95_pp": [x * 100 for x in ci(qdelta)],
        "mcnemar": _exact_mcnemar(a, b),
    }


def _frozen_workload() -> tuple[list[tuple[str, str]], dict]:
    _regular(INDEX)
    if _sha256(INDEX) != INDEX_SHA256:
        raise AnalysisError("data/index.json SHA256 differs from frozen GQA-40")
    with INDEX.open() as f:
        index = json.load(f)
    if len(index) != N_IMAGES:
        raise AnalysisError(f"frozen index must have {N_IMAGES} images")
    ordered = []
    records = {}
    counts = {}
    for entry in index:
        image = str(entry["image_id"])
        questions = entry["questions"][SKIP:SKIP + QUESTIONS_PER_IMAGE]
        counts[image] = len(questions)
        for question in questions:
            key = (image, str(question["question_id"]))
            if key in records:
                raise AnalysisError(f"duplicate frozen request key: {key}")
            gold = (question["answers"] if "answers" in question
                    else [question["answer"]])
            records[key] = {
                "question": str(question["question"]),
                "gold": [str(x) for x in gold],
            }
            ordered.append(key)
    if len(records) != N_QUESTIONS or set(counts.values()) != {
            QUESTIONS_PER_IMAGE}:
        raise AnalysisError("frozen workload is not 40 images x 6 questions")
    blob = "\n".join(f"{i}\t{q}" for i, q in ordered).encode()
    if hashlib.sha256(blob).hexdigest() != WORKLOAD_SHA256:
        raise AnalysisError("questions[4:10] workload SHA256 differs")
    return ordered, records


def _canonical_gold(value, context: str) -> list[str]:
    value = _json_cell(value, context)
    if isinstance(value, (str, int, float)):
        value = [value]
    if not isinstance(value, list):
        raise AnalysisError(f"{context}: ground truth is not a JSON list")
    return [str(x) for x in value]


def _gqa_score(prediction: str, gold: list[str]) -> float:
    articles = {"a", "an", "the"}

    def norm(value) -> str:
        value = re.sub(r"[^\w\s]", " ", str(value).lower())
        return " ".join(word for word in value.split()
                        if word not in articles)

    predicted = norm(prediction)
    expected = norm(gold[0])
    return float(predicted == expected or (
        expected and predicted.split()[:len(expected.split())]
        == expected.split()))


def _load_run(path: Path, name: str, ordered_keys, frozen,
              *, new_schema: bool, expected_layout: str | None) -> dict:
    path = _resolve(path)
    _reject_symlinks(path)
    if not path.is_dir() or path.is_symlink():
        raise AnalysisError(f"{name} is not a regular run directory: {path}")
    files = {key: path / filename for key, filename in {
        "results": "results.json",
        "csv": "per_request.csv",
        "sanity": "sanity.json",
        "summary_csv": "summary.csv",
    }.items()}
    for file in files.values():
        _regular(file)
    signatures = {p: _sig(p) for p in files.values()}
    hashes = {p: _sha256(p) for p in files.values()}

    with files["results"].open() as f:
        results = json.load(f)
    with files["sanity"].open() as f:
        sanity = json.load(f)
    fields, rows = _read_csv(files["csv"])
    summary_fields, summary_rows = _read_csv(files["summary_csv"])
    required = COMMON_RUN_FIELDS | (NEW_RUN_FIELDS if new_schema else set())
    missing = sorted(required - set(fields))
    if missing:
        raise AnalysisError(f"{name} per_request.csv missing fields: {missing}")
    if results.get("schema_version") != 2:
        raise AnalysisError(f"{name} results schema is not v2")
    summary = results.get("summary")
    if not isinstance(summary, dict) or summary.get("schema_version") != 2:
        raise AnalysisError(f"{name} summary schema is not v2")

    expected_summary = {
        "n": N_QUESTIONS,
        "n_images": N_IMAGES,
        "skip": SKIP,
        "questions_per_image_requested": QUESTIONS_PER_IMAGE,
        "metric": "gqa",
        "cold": True,
        "max_new_tokens": 16,
        "index_sha256": INDEX_SHA256,
        "workload_sha256": WORKLOAD_SHA256,
        "sep_policy": "sidecar",
    }
    mismatches = {
        key: {"expected": expected, "actual": summary.get(key)}
        for key, expected in expected_summary.items()
        if summary.get(key) != expected
    }
    if mismatches:
        raise AnalysisError(f"{name} fixed-condition mismatch: {mismatches}")
    qpi = summary.get("questions_per_image", {})
    if (qpi.get("min") != QUESTIONS_PER_IMAGE
            or qpi.get("max") != QUESTIONS_PER_IMAGE
            or not _close(qpi.get("mean", -1), QUESTIONS_PER_IMAGE)):
        raise AnalysisError(f"{name} questions-per-image summary mismatch")
    if expected_layout is not None and summary.get("prefix_layout") != expected_layout:
        raise AnalysisError(
            f"{name} expected prefix_layout={expected_layout!r}, "
            f"got {summary.get('prefix_layout')!r}")
    if new_schema and expected_layout is None:
        raise AnalysisError(f"{name}: new run must declare a Prefix layout")

    keys = [(str(r.get("method_key")), str(r.get("image_id")),
             str(r.get("question_id"))) for r in rows]
    if len(keys) != len(set(keys)):
        raise AnalysisError(f"{name} has duplicate method/request CSV rows")
    groups = defaultdict(list)
    expected_key_set = set(ordered_keys)
    for row in rows:
        method = str(row["method_key"])
        groups[method].append(row)
        if row.get("dataset") != "gqa":
            raise AnalysisError(f"{name}/{method}: dataset is not gqa")
        key = (str(row["image_id"]), str(row["question_id"]))
        if key not in frozen:
            raise AnalysisError(f"{name}/{method}: unexpected request {key}")
        expected = frozen[key]
        if row["question"] != expected["question"]:
            raise AnalysisError(f"{name}/{method}/{key}: question text differs")
        if _canonical_gold(row["ground_truth"], f"{name}/{method}/{key}") != \
                expected["gold"]:
            raise AnalysisError(f"{name}/{method}/{key}: gold answer differs")
        correct = _number(row["correct"], f"{name}/{method}/{key}/correct")
        if correct not in (0.0, 1.0):
            raise AnalysisError(f"{name}/{method}/{key}: GQA score is not binary")
        recomputed = _gqa_score(row["prediction"], expected["gold"])
        if correct != recomputed:
            raise AnalysisError(
                f"{name}/{method}/{key}: GQA score differs from prediction/gold")
        for field in ("ttft_ms", "decode_ms", "e2e_latency_ms",
                      "ssd_read_ms", "ssd_read_bytes", "ssd_preads"):
            if _number(row[field], f"{name}/{method}/{key}/{field}") < 0:
                raise AnalysisError(f"{name}/{method}/{key}: negative {field}")
        ttft = _number(row["ttft_ms"], "ttft_ms")
        decode = _number(row["decode_ms"], "decode_ms")
        e2e = _number(row["e2e_latency_ms"], "e2e_latency_ms")
        if e2e + 1e-6 < ttft or abs(e2e - ttft - decode) > 1.0:
            raise AnalysisError(f"{name}/{method}/{key}: latency algebra failed")
    for method, method_rows in groups.items():
        method_keys = {(str(r["image_id"]), str(r["question_id"]))
                       for r in method_rows}
        if len(method_rows) != N_QUESTIONS or method_keys != expected_key_set:
            raise AnalysisError(f"{name}/{method}: workload is not exact 40/240")

    # Cross-check the flat CSV against results.json for every emitted method.
    nested = {}
    result_rows = results.get("rows")
    if not isinstance(result_rows, list) or len(result_rows) != N_QUESTIONS:
        raise AnalysisError(f"{name}: results.json does not contain 240 requests")
    for record in result_rows:
        key = (str(record.get("image_id")), str(record.get("question_id")))
        if key not in frozen or key in nested:
            raise AnalysisError(f"{name}: bad/duplicate results.json key {key}")
        if record.get("question") != frozen[key]["question"]:
            raise AnalysisError(f"{name}: JSON question differs for {key}")
        if [str(x) for x in record.get("gold", [])] != frozen[key]["gold"]:
            raise AnalysisError(f"{name}: JSON gold differs for {key}")
        nested[key] = record
    for method, method_rows in groups.items():
        for row in method_rows:
            key = (str(row["image_id"]), str(row["question_id"]))
            value = nested[key].get(method)
            if not isinstance(value, dict):
                raise AnalysisError(f"{name}: JSON missing {method}/{key}")
            if row["prediction"] != str(value.get("answer", "")):
                raise AnalysisError(f"{name}: prediction JSON/CSV mismatch {method}/{key}")
            pairs = (
                ("correct", "acc"), ("ttft_ms", "ttft_ms"),
                ("decode_ms", "decode_ms"), ("e2e_latency_ms", "e2e_latency_ms"),
                ("selector_ms", "selector_ms"),
                ("ssd_read_ms", "ssd_read_ms"),
                ("scatter_ms", "scatter_ms"),
                ("prepare_ms", "prepare_ms"),
                ("prefill_ms", "prefill_ms"),
                ("ssd_read_bytes", "ssd_read_bytes"),
                ("ssd_preads", "preads"),
                ("ssd_read_chunks", "ssd_read_chunks"),
                ("n_chunks_selected", "n_chunks_selected"),
                ("n_chunks_total", "n_chunks_total"),
                ("touched_chunk_fraction", "touched_chunk_fraction"),
                ("logical_kv_ratio", "logical_kv_ratio"),
                ("normal_chunk_count_total", "normal_chunk_count_total"),
                ("normal_kv_read_bytes", "normal_kv_read_bytes"),
                ("separator_read_bytes", "separator_read_bytes"),
                ("normal_kv_preads", "normal_kv_preads"),
                ("separator_preads", "separator_preads"),
                ("total_actual_pread_bytes", "total_actual_pread_bytes"),
                ("mean_bytes_per_pread", "mean_bytes_per_pread"),
                ("normal_mean_bytes_per_pread", "normal_mean_bytes_per_pread"),
                ("separator_mean_bytes_per_pread",
                 "separator_mean_bytes_per_pread"),
                ("static_score_calls", "static_score_calls"),
                ("query_score_calls", "query_score_calls"),
                ("diversity_calls", "diversity_calls"),
            )
            for csv_key, json_key in pairs:
                if csv_key not in row:
                    if new_schema:
                        raise AnalysisError(
                            f"{name}: missing JSON/CSV field {csv_key}")
                    continue
                csv_value = _number(row[csv_key], csv_key, blank=True)
                json_value = _number(value.get(json_key), json_key, blank=True)
                if ((csv_value is None) != (json_value is None)
                        or (csv_value is not None and not _close(
                            csv_value, json_value, atol=1e-5))):
                    raise AnalysisError(
                        f"{name}: JSON/CSV mismatch {method}/{key}/{csv_key}")
            csv_sel = _json_cell(row.get("selected_chunk_ids_per_layer"), "selection")
            json_sel = value.get("selected_chunk_ids_per_layer")
            if csv_sel != json_sel:
                raise AnalysisError(f"{name}: selection JSON/CSV mismatch {method}/{key}")
            for field in ("selection_mode", "separator_policy",
                          "reordered_prefix_store_validated",
                          "validated_prefix_layout"):
                csv_value = row.get(field)
                json_value = value.get(field)
                if csv_value is None:
                    csv_value = ""
                if json_value is None:
                    json_value = ""
                elif type(json_value) is bool:
                    json_value = str(json_value)
                if str(csv_value).lower() != str(json_value).lower():
                    raise AnalysisError(
                        f"{name}: JSON/CSV mismatch {method}/{key}/{field}")
            if new_schema:
                csv_first = _integer(row.get("first_token_id"), "first_token_id",
                                     blank=True)
                json_first = value.get("first_token_id")
                if csv_first != json_first:
                    raise AnalysisError(f"{name}: first-token JSON/CSV mismatch {method}/{key}")

    summary_methods = summary.get("per_method", {})
    if "method_key" not in summary_fields:
        raise AnalysisError(f"{name}: summary.csv lacks method_key")
    summary_keys = [r["method_key"] for r in summary_rows]
    if len(summary_keys) != len(set(summary_keys)):
        raise AnalysisError(f"{name}: summary.csv has duplicate methods")
    csv_summary_map = {r["method_key"]: r for r in summary_rows}
    if set(groups) != set(summary_methods) or set(groups) != set(csv_summary_map):
        raise AnalysisError(f"{name}: method sets differ across artifacts")
    for method, method_rows in groups.items():
        accuracy = float(np.mean([_number(r["correct"], "correct")
                                  for r in method_rows]))
        ttft = float(np.mean([_number(r["ttft_ms"], "ttft_ms")
                              for r in method_rows]))
        bytes_mean = float(np.mean([_number(r["ssd_read_bytes"], "bytes")
                                    for r in method_rows]))
        sm = summary_methods[method]
        sr = csv_summary_map[method]
        for actual, expected, label in (
            (sm.get("acc"), accuracy, "results accuracy"),
            (sm.get("ttft_mean_ms"), ttft, "results TTFT"),
            (sm.get("ssd_read_bytes_mean"), bytes_mean, "results bytes"),
            (sr.get("accuracy"), accuracy, "summary.csv accuracy"),
            (sr.get("ttft_mean_ms"), ttft, "summary.csv TTFT"),
            (sr.get("ssd_read_bytes_mean"), bytes_mean, "summary.csv bytes"),
        ):
            if not _close(_number(actual, label), expected, atol=1e-4):
                raise AnalysisError(f"{name}/{method}: {label} mismatch")

    if (_integer(sanity.get("unique_images"), "sanity.unique_images") != N_IMAGES
            or _integer(sanity.get("unique_questions"),
                        "sanity.unique_questions") != N_QUESTIONS
            or _integer(sanity.get("duplicate_method_image_question_rows"),
                        "sanity.duplicates") != 0
            or not _bool(sanity.get("all_required_times_finite_nonnegative"),
                         "sanity.times")
            or not _bool(sanity.get("generated_token_cap_respected"),
                         "sanity.cap")):
        raise AnalysisError(f"{name}: sanity.json failed")
    expected_csv_rows = N_QUESTIONS * len(groups)
    if (_integer(sanity.get("expected_csv_rows"), "sanity.expected_csv_rows")
            != expected_csv_rows
            or _integer(sanity.get("actual_csv_rows"), "sanity.actual_csv_rows")
            != expected_csv_rows):
        raise AnalysisError(f"{name}: sanity row-count contract failed")
    requests_per_method = sanity.get("requests_per_method")
    if (not isinstance(requests_per_method, dict)
            or set(requests_per_method) != set(groups)
            or any(_integer(value, f"sanity.requests_per_method.{method}")
                   != N_QUESTIONS
                   for method, value in requests_per_method.items())):
        raise AnalysisError(f"{name}: sanity requests-per-method differs")

    return {
        "name": name,
        "path": path,
        "files": files,
        "signatures": signatures,
        "hashes": hashes,
        "results": results,
        "summary": summary,
        "fields": fields,
        "rows": rows,
        "groups": dict(groups),
    }


def _find_method(run: dict, base: str, budget: float | None) -> str:
    candidates = []
    for method, rows in run["groups"].items():
        if method != base and not method.startswith(base + "@"):
            continue
        retentions = {_number(r.get("retention"), "retention", blank=True)
                      for r in rows}
        kinds = {r.get("retention_kind") for r in rows}
        expected_kind = "full" if budget is None else "chunk"
        expected_retention = 1.0 if budget is None else budget
        if (retentions == {expected_retention} and kinds == {expected_kind}):
            candidates.append(method)
    if len(candidates) != 1:
        raise AnalysisError(
            f"{run['name']}: expected one {base} budget={budget}, got {candidates}")
    return candidates[0]


def _find_recompute_method(run: dict) -> str:
    """Find the unique pixel-recomputation arm without overloading FullLoad.

    ``_find_method(..., budget=None)`` deliberately means 100% FullLoad.  A
    ReComp row instead has no retention value and ``retention_kind=none``.
    Keeping that distinction explicit prevents a no-SSD baseline from being
    silently treated as a cache layout.
    """
    candidates = []
    for method, rows in run["groups"].items():
        if method != "recompute":
            continue
        retentions = {_number(row.get("retention"), "retention", blank=True)
                      for row in rows}
        kinds = {row.get("retention_kind") for row in rows}
        if retentions == {None} and kinds == {"none"}:
            candidates.append(method)
    if len(candidates) != 1:
        raise AnalysisError(
            f"{run['name']}: expected one no-retention recompute arm, "
            f"got {candidates}")
    return candidates[0]


def _rows_by_key(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(str(r["image_id"]), str(r["question_id"])): r for r in rows}


def _load_profile(path: Path, expected: str, image_ids: set[str]) -> dict:
    path = _resolve(path)
    _reject_symlinks(path)
    _regular(path)
    signature, digest = _sig(path), _sha256(path)
    with path.open() as f:
        profile = json.load(f)
    if profile.get("schema_version") != 1:
        raise AnalysisError(f"{expected} build profile schema is not v1")
    top_layout = "visionzip" if expected == "visionzip_image_only" else "raster"
    if profile.get("layout") != top_layout:
        raise AnalysisError(
            f"expected {top_layout} build profile, got {profile.get('layout')}")
    if (profile.get("fresh_no_clobber") is not True
            or profile.get("layout_uses_dataset_question") is not False
            or profile.get("llm_used_for_layout_scoring") is not False
            or profile.get("calibration_questions") != 0):
        raise AnalysisError(f"{expected} build profile provenance failed")
    if (_integer(profile.get("n_images"), f"{expected}.n_images") != N_IMAGES
            or not isinstance(profile.get("command"), str)
            or not isinstance(profile.get("store"), str)
            or not profile["store"]):
        raise AnalysisError(f"{expected} build profile top-level metadata failed")
    rows = profile.get("rows")
    if not isinstance(rows, list) or len(rows) != N_IMAGES:
        raise AnalysisError(f"{expected} build profile must contain 40 rows")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise AnalysisError(f"{expected} build profile row {index} is not an object")
        missing = sorted(PROFILE_ROW_FIELDS - set(row))
        if missing:
            raise AnalysisError(
                f"{expected} build profile row {index} missing: {missing}")
    ids = [str(row.get("image_id")) for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != image_ids:
        raise AnalysisError(f"{expected} build profile image set differs")
    by_image = {}
    timing_fields = (
        "vision_saliency_ms", "permutation_ms", "prefix_forward_ms",
        "kv_materialize_ms", "kv_repack_ms", "ssd_write_ms",
        "total_ingestion_ms",
    )
    integer_fields = (
        "v_token_num", "real_visual_tokens", "separator_tokens", "n_chunks",
        "mapping_metadata_bytes", "visual_kv_bytes", "probe_sidecar_bytes",
        "separator_sidecar_bytes",
    )
    for row in rows:
        image = str(row["image_id"])
        if row.get("layout") != expected:
            raise AnalysisError(f"{expected}/{image}: wrong row layout")
        values = {key: _integer(row.get(key), f"{expected}/{image}/{key}")
                  for key in integer_fields}
        if any(value < 0 for value in values.values()):
            raise AnalysisError(f"{expected}/{image}: negative byte/geometry field")
        if (values["real_visual_tokens"] + values["separator_tokens"]
                != values["v_token_num"]
                or values["n_chunks"] != math.ceil(
                    values["v_token_num"] / CHUNK_SIZE)
                or values["separator_sidecar_bytes"] <= 0
                or values["visual_kv_bytes"] <= 0):
            raise AnalysisError(f"{expected}/{image}: invalid profile geometry")
        times = {key: _number(row.get(key), f"{expected}/{image}/{key}")
                 for key in timing_fields}
        if any(value < 0 for value in times.values()):
            raise AnalysisError(f"{expected}/{image}: negative profile timing")
        component_max = max(value for key, value in times.items()
                            if key != "total_ingestion_ms")
        if times["total_ingestion_ms"] < component_max - 1e-6:
            raise AnalysisError(f"{expected}/{image}: total ingestion is too small")
        if (row.get("layout_uses_dataset_question") is not False
                or row.get("llm_used_for_layout_scoring") is not False
                or row.get("calibration_questions") != 0
                or row.get("layout_frozen_before_llm_prefix_forward") is not True):
            raise AnalysisError(f"{expected}/{image}: row provenance failed")
        if expected == "visionzip_image_only":
            for key in ("permutation_sha256", "inverse_permutation_sha256",
                        "image_input_sha256"):
                value = row.get(key)
                if not isinstance(value, str) or len(value) != 64:
                    raise AnalysisError(f"{expected}/{image}: invalid {key}")
            if row.get("image_inputs_match_prompt_processor") is not True:
                raise AnalysisError(f"{expected}/{image}: image inputs differ")
            if times["vision_saliency_ms"] <= 0 or times["kv_repack_ms"] <= 0:
                raise AnalysisError(f"{expected}/{image}: missing repack work")
        else:
            if (not _close(times["vision_saliency_ms"], 0)
                    or not _close(times["permutation_ms"], 0)
                    or not _close(times["kv_repack_ms"], 0)):
                raise AnalysisError(f"raster/{image}: unexpected repack work")
        by_image[image] = row
    return {
        "path": path,
        "signature": signature,
        "sha256": digest,
        "document": profile,
        "rows": rows,
        "by_image": by_image,
    }


def _profile_stats(raster: dict, visionzip: dict) -> tuple[list[dict], dict]:
    image_ids = sorted(raster["by_image"])
    paired_fields = ("v_token_num", "real_visual_tokens", "separator_tokens",
                     "n_chunks", "visual_kv_bytes", "probe_sidecar_bytes")
    for image in image_ids:
        rr, vr = raster["by_image"][image], visionzip["by_image"][image]
        for field in paired_fields:
            if _integer(rr[field], field) != _integer(vr[field], field):
                raise AnalysisError(f"build profiles differ: {image}/{field}")

    timing_fields = (
        "vision_saliency_ms", "permutation_ms", "prefix_forward_ms",
        "kv_materialize_ms", "kv_repack_ms", "ssd_write_ms",
        "total_ingestion_ms",
    )
    summaries = {}
    for layout, profile in (("raster", raster),
                            ("visionzip_image_only", visionzip)):
        result = {}
        for field in timing_fields:
            values = [_number(row[field], field) for row in profile["rows"]]
            mean, p50, p95 = _percentiles(values)
            result[field] = {"mean": mean, "p50": p50, "p95": p95}
        for field in ("mapping_metadata_bytes", "visual_kv_bytes",
                      "probe_sidecar_bytes", "separator_sidecar_bytes"):
            result[field + "_mean"] = float(np.mean([
                _integer(row[field], field) for row in profile["rows"]]))
        summaries[layout] = result

    incremental = (summaries["visionzip_image_only"]["total_ingestion_ms"]["mean"]
                   - summaries["raster"]["total_ingestion_ms"]["mean"])
    rows = []
    for layout in ("raster", "visionzip_image_only"):
        stats = summaries[layout]
        for reuse in (1, 5, 10, 20):
            total = stats["total_ingestion_ms"]["mean"]
            rows.append({
                "layout": layout,
                "n_images": N_IMAGES,
                "reuse_requests_per_image": reuse,
                "vision_saliency_mean_ms": stats["vision_saliency_ms"]["mean"],
                "permutation_mean_ms": stats["permutation_ms"]["mean"],
                "prefix_forward_mean_ms": stats["prefix_forward_ms"]["mean"],
                "kv_materialize_mean_ms": stats["kv_materialize_ms"]["mean"],
                "kv_repack_mean_ms": stats["kv_repack_ms"]["mean"],
                "ssd_write_mean_ms": stats["ssd_write_ms"]["mean"],
                "total_ingestion_mean_ms": total,
                "total_ingestion_all_images_ms": total * N_IMAGES,
                "total_ingestion_p50_ms": stats["total_ingestion_ms"]["p50"],
                "total_ingestion_p95_ms": stats["total_ingestion_ms"]["p95"],
                "amortized_total_ingestion_ms_per_request": total / reuse,
                "incremental_vs_raster_mean_ms": (0.0 if layout == "raster"
                                                   else incremental),
                "incremental_vs_raster_all_images_ms": (
                    0.0 if layout == "raster" else incremental * N_IMAGES),
                "amortized_incremental_vs_raster_ms_per_request": (
                    0.0 if layout == "raster" else incremental / reuse),
                "mapping_metadata_bytes_mean":
                    stats["mapping_metadata_bytes_mean"],
                "visual_kv_bytes_mean": stats["visual_kv_bytes_mean"],
                "probe_sidecar_bytes_mean": stats["probe_sidecar_bytes_mean"],
                "separator_sidecar_bytes_mean":
                    stats["separator_sidecar_bytes_mean"],
            })
    return rows, {
        "per_layout": summaries,
        "visionzip_incremental_ingestion_mean_ms": incremental,
        "reuse_counts": [1, 5, 10, 20],
    }


def _load_coverage(path: Path, image_ids: set[str], matched: bool) -> dict:
    path = _resolve(path)
    _reject_symlinks(path)
    _regular(path)
    signature, digest = _sig(path), _sha256(path)
    fields, rows = _read_csv(path)
    if fields != COVERAGE_FIELDS:
        raise AnalysisError(
            f"coverage header differs from canonical long form: {fields}")
    observed_layouts = {row.get("layout") for row in rows}
    allowed_layout_sets = (set(BASE_LAYOUTS), set(BASE_LAYOUTS) | {MATCHED_LAYOUT})
    if observed_layouts not in allowed_layout_sets:
        raise AnalysisError(f"unexpected/incomplete coverage layouts: {observed_layouts}")
    if matched and MATCHED_LAYOUT not in observed_layouts:
        raise AnalysisError("matched calibration run lacks matched coverage layout")
    layouts = BASE_LAYOUTS + ((MATCHED_LAYOUT,)
                              if MATCHED_LAYOUT in observed_layouts else ())
    expected_count = (N_IMAGES * N_LAYERS * len(layouts)
                      * len(IMPORTANCE_SOURCES) * len(FRACTIONS))
    if len(rows) != expected_count:
        raise AnalysisError(
            f"coverage row count {len(rows)} != expected {expected_count}")
    seen = set()
    geometry = {}
    order_hashes = defaultdict(set)
    grouped = defaultdict(list)
    physical_geometry = {}
    total_mass_invariants = defaultdict(list)
    for row in rows:
        image = str(row["image_id"])
        layer = _integer(row["layer"], "coverage.layer")
        layout = row["layout"]
        source = row["importance_source"]
        fraction = _number(row["physical_fraction_requested"], "coverage.fraction")
        if (image not in image_ids or layer not in range(N_LAYERS)
                or layout not in layouts or source not in IMPORTANCE_SOURCES
                or fraction not in FRACTIONS):
            raise AnalysisError(f"unexpected coverage coordinate: {row}")
        key = (image, layer, layout, source, fraction)
        if key in seen:
            raise AnalysisError(f"duplicate coverage coordinate: {key}")
        seen.add(key)
        analysis_only = _bool(row["analysis_only_calibration"],
                              "analysis_only_calibration")
        if analysis_only != (source == "sparsevlm_calib4_analysis_only"):
            raise AnalysisError(f"coverage analysis-only flag mismatch: {key}")
        total_chunks = _integer(row["total_chunk_count"], "total_chunk_count")
        selected_chunks = _integer(row["selected_chunk_count"],
                                   "selected_chunk_count")
        normal = _integer(row["total_normal_tokens"], "total_normal_tokens")
        sep = _integer(row["separator_tokens"], "separator_tokens")
        selected_rows = _integer(row["selected_physical_rows"],
                                 "selected_physical_rows")
        selected_normal = _integer(row["selected_normal_tokens"],
                                   "selected_normal_tokens")
        vn = normal + sep
        if (total_chunks != math.ceil(vn / CHUNK_SIZE)
                or selected_chunks != _budget_chunks(total_chunks, fraction)
                or selected_rows != min(selected_chunks * CHUNK_SIZE, vn)
                or not 0 <= selected_normal <= min(selected_rows, normal)):
            raise AnalysisError(f"coverage chunk/geometry contract failed: {key}")
        realized = _number(row["realized_normal_token_fraction"], "realized")
        if not _close(realized, selected_normal / normal, atol=1e-10):
            raise AnalysisError(f"coverage realized fraction mismatch: {key}")
        selected_mass = _number(row["selected_importance_mass"], "selected_mass")
        total_mass = _number(row["total_importance_mass"], "total_mass")
        coverage = _number(row["importance_mass_coverage"], "coverage")
        if total_mass <= 0 or selected_mass < 0 or selected_mass > total_mass + 1e-5:
            raise AnalysisError(f"invalid coverage mass: {key}")
        if not _close(coverage, selected_mass / total_mass, atol=1e-9):
            raise AnalysisError(f"coverage mass ratio mismatch: {key}")
        total_mass_invariants[(image, layer, source)].append(total_mass)
        expected_tail = layout in {
            "morton", "visionzip_image_only", MATCHED_LAYOUT}
        if (_bool(row["separator_tail"], "separator_tail") != expected_tail
                or row["prefix_rule"] != "chunks range(0,k)"):
            raise AnalysisError(f"coverage layout policy mismatch: {key}")
        order_hash = row["order_sha256"]
        if (len(order_hash) != 64
                or any(c not in "0123456789abcdef" for c in order_hash)):
            raise AnalysisError(f"invalid coverage order hash: {key}")
        order_hashes[(image, layer, layout)].add(order_hash)
        physical_key = (image, layer, layout, fraction)
        physical_value = (selected_chunks, selected_rows, selected_normal,
                          total_chunks, normal, sep)
        previous = physical_geometry.setdefault(physical_key, physical_value)
        if previous != physical_value:
            raise AnalysisError(
                f"coverage physical selection changes with importance source: "
                f"{physical_key}")
        gkey = (layout, source, fraction)
        grouped[gkey].append((selected_mass, total_mass, coverage, realized))
        old = geometry.setdefault(image, (vn, normal, sep, total_chunks))
        if old != (vn, normal, sep, total_chunks):
            raise AnalysisError(f"coverage geometry varies for image {image}")
    if len(seen) != expected_count or set(geometry) != image_ids:
        raise AnalysisError("coverage matrix is incomplete")
    if any(len(values) != 1 for values in order_hashes.values()):
        raise AnalysisError("coverage order hash changes across source/fraction")
    for layout in ("raster", "morton", "visionzip_image_only"):
        for image in image_ids:
            hashes = {next(iter(order_hashes[(image, layer, layout)]))
                      for layer in range(N_LAYERS)}
            if len(hashes) != 1:
                raise AnalysisError(f"global layout order varies by layer: {layout}/{image}")
    # A larger physical prefix must contain at least as many normal tokens and
    # non-negative importance mass as every smaller prefix.
    coverage_lookup = {}
    for row in rows:
        key = (str(row["image_id"]), _integer(row["layer"], "layer"),
               row["layout"], row["importance_source"])
        coverage_lookup.setdefault(key, []).append((
            _number(row["physical_fraction_requested"], "fraction"),
            _integer(row["selected_normal_tokens"], "selected_normal_tokens"),
            _number(row["selected_importance_mass"], "selected_importance_mass")))
    for key, values in coverage_lookup.items():
        values.sort()
        if any(values[i][1] > values[i + 1][1]
               or values[i][2] > values[i + 1][2] + 1e-7
               for i in range(len(values) - 1)):
            raise AnalysisError(f"coverage is not prefix-monotone: {key}")
    for key, values in total_mass_invariants.items():
        if any(not _close(value, values[0], atol=1e-9)
               for value in values[1:]):
            raise AnalysisError(
                f"total importance mass changes with layout/fraction: {key}")
    for image in image_ids:
        totals = [total_mass_invariants[
            (image, layer, "visionzip_image_saliency")][0]
            for layer in range(N_LAYERS)]
        if any(not _close(value, totals[0], atol=1e-9)
               for value in totals[1:]):
            raise AnalysisError(
                f"repeated VisionZip image saliency mass varies by layer: {image}")

    summary = []
    for (layout, source, fraction), values in sorted(grouped.items()):
        if len(values) != N_IMAGES * N_LAYERS:
            raise AnalysisError(f"coverage group incomplete: {layout}/{source}/{fraction}")
        selected = sum(x[0] for x in values)
        total = sum(x[1] for x in values)
        summary.append({
            "layout": layout,
            "importance_source": source,
            "physical_fraction_requested": fraction,
            "n_image_layers": len(values),
            "macro_mean_coverage": float(np.mean([x[2] for x in values])),
            "global_mass_weighted_coverage": selected / total,
            "mean_realized_normal_token_fraction":
                float(np.mean([x[3] for x in values])),
        })
    return {
        "path": path,
        "signature": signature,
        "sha256": digest,
        "fields": fields,
        "rows": rows,
        "summary": summary,
        "geometry": geometry,
        "physical_geometry": physical_geometry,
        "_order_hashes": dict(order_hashes),
        "layouts": list(layouts),
    }


def _proof_pass(value, name: str) -> bool:
    """Interpret only explicit pass/identity representations."""
    if type(value) is bool:
        return value
    if isinstance(value, dict) and "passed" in value:
        return _bool(value["passed"], f"{name}.passed")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _close(value, 1.0, atol=1e-12)
    raise AnalysisError(f"{name} is not an explicit pass result: {value!r}")


def _agreement(value, name: str) -> float:
    if type(value) is bool:
        return 1.0 if value else 0.0
    answer = _number(value, name)
    if not 0.0 <= answer <= 1.0:
        raise AnalysisError(f"{name} is outside [0,1]")
    return answer


def _load_external_validation(path: Path) -> dict:
    path = _resolve(path)
    _reject_symlinks(path)
    _regular(path)
    signature, digest = _sig(path), _sha256(path)
    with path.open() as f:
        doc = json.load(f)
    if not isinstance(doc, dict):
        raise AnalysisError("validation JSON must be an object")
    if doc.get("schema_version") != 1:
        raise AnalysisError("external validation schema is not v1")
    strict_all_passed = _bool(doc.get("all_passed"), "all_passed")
    gates = doc.get("gates")
    if not isinstance(gates, dict) or not gates:
        raise AnalysisError("external validation has no explicit gates")
    malformed = [name for name, value in gates.items()
                 if not isinstance(value, dict) or "passed" not in value]
    if malformed:
        raise AnalysisError(f"external validation gates malformed: {malformed}")
    gate_passes = {
        name: _proof_pass(value, f"gates.{name}")
        for name, value in gates.items()
    }
    failed = {name for name, passed in gate_passes.items() if not passed}
    allowed_strict_failure = {"full_load_240_identity"}
    if failed - allowed_strict_failure:
        raise AnalysisError(
            f"non-numerical external validation gates failed: {sorted(failed)}")
    if strict_all_passed != all(gate_passes.values()):
        raise AnalysisError("external all_passed disagrees with gate conjunction")

    full = doc.get("full_load_identity")
    if not isinstance(full, dict):
        raise AnalysisError("validation missing full_load_identity")
    full_claimed_pass = _proof_pass(
        full.get("passed"), "full_load_identity.passed")
    for field in ("n_requests", "prediction_agreement", "first_token_agreement",
                  "accuracy_raster", "accuracy_repacked"):
        if field not in full:
            raise AnalysisError(f"full_load_identity missing {field}")
    if _integer(full["n_requests"], "full_load_identity.n_requests") != N_QUESTIONS:
        raise AnalysisError("external FullLoad identity is not the 240 workload")
    prediction_agreement = _agreement(
        full["prediction_agreement"], "prediction_agreement")
    first_token_agreement = _agreement(
        full["first_token_agreement"], "first_token_agreement")
    accuracy_raster = _number(full["accuracy_raster"], "accuracy_raster")
    accuracy_repacked = _number(full["accuracy_repacked"], "accuracy_repacked")
    derived_strict_pass = (
        _close(prediction_agreement, 1.0, atol=1e-12)
        and _close(first_token_agreement, 1.0, atol=1e-12)
        and _close(accuracy_raster, accuracy_repacked, atol=1e-12))
    if full_claimed_pass != derived_strict_pass:
        raise AnalysisError("external FullLoad passed flag disagrees with its values")
    if gate_passes.get("full_load_240_identity") != derived_strict_pass:
        raise AnalysisError("external FullLoad gate disagrees with identity values")

    query = doc.get("query_independence")
    if not isinstance(query, dict):
        raise AnalysisError("validation missing query_independence")
    if not _proof_pass(query.get("passed"), "query_independence.passed"):
        raise AnalysisError("external query_independence is not passing")
    for field in ("n_images", "questions_per_image", "pixels_identical",
                  "image_sizes_identical", "scores_identical",
                  "permutations_identical", "decoder_layer_forward_calls",
                  "forbidden_call_counts"):
        if field not in query:
            raise AnalysisError(f"query_independence missing {field}")
    if (_integer(query["n_images"], "query_independence.n_images") < 3
            or _integer(query["questions_per_image"],
                        "query_independence.questions_per_image") < 3):
        raise AnalysisError("query-independence sample is too small")
    for field in ("pixels_identical", "image_sizes_identical",
                  "scores_identical", "permutations_identical"):
        if not _proof_pass(query[field], f"query_independence.{field}"):
            raise AnalysisError(f"query independence failed: {field}")
    if _integer(query["decoder_layer_forward_calls"],
                "decoder_layer_forward_calls") != 0:
        raise AnalysisError("image-only layout scoring called decoder layers")
    forbidden = query["forbidden_call_counts"]
    if not isinstance(forbidden, dict) or not forbidden:
        raise AnalysisError("forbidden_call_counts must be a nonempty object")
    if any(_integer(value, f"forbidden_call_counts.{key}") != 0
           for key, value in forbidden.items()):
        raise AnalysisError("a forbidden query-dependent call was observed")

    direct = doc.get("direct_vs_posthoc")
    if not isinstance(direct, dict):
        raise AnalysisError("validation missing direct_vs_posthoc")
    if not _proof_pass(direct.get("passed"), "direct_vs_posthoc.passed"):
        raise AnalysisError("external direct_vs_posthoc is not passing")
    for field in ("selected_prefix_kv_exact", "prediction_agreement"):
        if field not in direct or not _proof_pass(
                direct[field], f"direct_vs_posthoc.{field}"):
            raise AnalysisError(f"direct-vs-posthoc validation failed: {field}")
    if not _proof_pass(doc.get("protected_inputs_unchanged"),
                       "protected_inputs_unchanged"):
        raise AnalysisError("protected inputs changed")
    conditions = doc.get("conditions")
    if (not isinstance(conditions, dict) or not conditions
            or any(not _proof_pass(value, f"conditions.{key}")
                   for key, value in conditions.items())):
        raise AnalysisError("external fixed-condition validation failed")
    return {
        "path": path,
        "signature": signature,
        "sha256": digest,
        "document": doc,
        "strict_all_passed": strict_all_passed,
        "gate_passes": gate_passes,
        "strict_full_load_exact_pass": derived_strict_pass,
    }


def _load_full_load_audit(validation_path: Path) -> dict:
    path = validation_path.parent / "full_load_240_audit.json"
    path = _resolve(path)
    _reject_symlinks(path)
    _regular(path)
    signature, digest = _sig(path), _sha256(path)
    with path.open() as f:
        doc = json.load(f)
    if (not isinstance(doc, dict) or doc.get("schema_version") != 1
            or doc.get("audit") != "raster_vs_direct_visionzip_full_load_240"):
        raise AnalysisError("invalid full_load_240_audit.json identity/schema")
    classification = doc.get("classification")
    behavioral = doc.get("behavioral_equivalence")
    structural = doc.get("mapped_logical_kv_audit")
    alignment = doc.get("request_alignment")
    io = doc.get("io_equivalence")
    smoke = doc.get("smoke_logit_evidence")
    inputs = doc.get("inputs")
    if not all(isinstance(value, dict) for value in (
            classification, behavioral, structural, alignment, io, smoke,
            inputs)):
        raise AnalysisError("full-load audit is missing required sections")
    strict = _bool(classification.get("strict_exact_pass"),
                   "audit.strict_exact_pass")
    structural_pass = _bool(classification.get("structural_integrity_pass"),
                            "audit.structural_integrity_pass")
    warning = _bool(classification.get("bf16_numerical_warning"),
                    "audit.bf16_numerical_warning")
    if (strict != _bool(behavioral.get("strict_exact_pass"),
                        "behavioral.strict_exact_pass")
            or not structural_pass
            or (not strict and not warning)
            or not _bool(structural.get("structural_integrity_pass"),
                         "mapped.structural_integrity_pass")
            or not _bool(structural.get("all_logical_kv_exact"),
                         "mapped.all_logical_kv_exact")
            or not _bool(structural.get("all_sys_kv_hashes_equal"),
                         "mapped.all_sys_kv_hashes_equal")
            or not _bool(structural.get("all_v_hidden_hashes_equal"),
                         "mapped.all_v_hidden_hashes_equal")
            or _integer(structural.get("total_elements_compared"),
                        "mapped.total_elements_compared") <= 0
            or _integer(structural.get("total_differing_elements"),
                        "mapped.total_differing_elements") != 0
            or not _close(_number(structural.get("global_max_absolute_difference"),
                                  "mapped.global_max_absolute_difference"), 0.0,
                          atol=0.0, rtol=0.0)):
        raise AnalysisError("full-load structural integrity audit failed")
    if (_integer(alignment.get("n_requests_each"),
                 "audit.request_alignment.n_requests_each") != N_QUESTIONS
            or not _bool(alignment.get("key_sets_equal"), "key_sets_equal")
            or _integer(alignment.get("question_equal_count"),
                        "question_equal_count") != N_QUESTIONS
            or _integer(alignment.get("ground_truth_equal_count"),
                        "ground_truth_equal_count") != N_QUESTIONS
            or not _bool(io.get("all_240_requests_exact"),
                         "io.all_240_requests_exact")
            or not _bool(smoke.get("within_configured_bf16_gate"),
                         "smoke.within_configured_bf16_gate")
            or inputs.get("index_sha256") != INDEX_SHA256
            or inputs.get("evaluation_workload_sha256") != WORKLOAD_SHA256):
        raise AnalysisError("full-load audit workload/I/O/BF16 evidence failed")
    agreement = _agreement(behavioral.get("prediction_agreement"),
                           "audit.prediction_agreement")
    first_agreement = _agreement(behavioral.get("first_token_agreement"),
                                 "audit.first_token_agreement")
    raster_accuracy = _number(behavioral.get("raster_accuracy"),
                              "audit.raster_accuracy")
    vision_accuracy = _number(behavioral.get("visionzip_direct_accuracy"),
                              "audit.visionzip_direct_accuracy")
    mismatch_count = _integer(behavioral.get("mismatch_count"),
                              "audit.mismatch_count")
    derived_strict = (agreement == 1.0 and first_agreement == 1.0
                      and mismatch_count == 0
                      and _close(raster_accuracy, vision_accuracy,
                                 atol=1e-12))
    if strict != derived_strict:
        raise AnalysisError("full-load audit strict classification is inconsistent")
    mismatches = behavioral.get("mismatches")
    if not isinstance(mismatches, list) or len(mismatches) != mismatch_count:
        raise AnalysisError("full-load audit mismatch list/count differs")
    return {
        "path": path,
        "signature": signature,
        "sha256": digest,
        "document": doc,
        "strict_full_load_exact_pass": strict,
        "structural_integrity_pass": structural_pass,
        "bf16_numerical_warning": warning,
    }


def _role_layout(role: str) -> str:
    return {
        "raster_fullload": "raster",
        "recomp": "not_applicable",
        "raster_prefix25": "raster",
        "morton_prefix25": "morton",
        "calib4_prefix25": "calib4_importance_legacy",
        "visionzip_prefix25": "visionzip_image_only",
        "visionzip_prefix50": "visionzip_image_only",
        "visionzip_fullload_sanity": "visionzip_image_only",
        "matched_calib_prefix25": MATCHED_LAYOUT,
    }[role]


def _runtime_layout(role: str) -> str:
    """Name accepted/emitted by scripts/04_eval.py --prefix-layout."""
    if role == "matched_calib_prefix25":
        return "calib_importance_sep_tail"
    return _role_layout(role)


def _extract_roles(runs: dict) -> tuple[dict, dict]:
    methods = {
        "raster_fullload": _find_method(runs["raster"], "fullload", None),
        "recomp": _find_recompute_method(runs["calib4"]),
        "raster_prefix25": _find_method(
            runs["raster"], "layout_prefix_chunk", 0.25),
        "morton_prefix25": _find_method(
            runs["morton"], "layout_prefix_chunk", 0.25),
        "calib4_prefix25": _find_method(
            runs["calib4"], "reorder_prefix_chunk", 0.25),
        "visionzip_prefix25": _find_method(
            runs["visionzip"], "visionzip_repack_prefix", 0.25),
        "visionzip_prefix50": _find_method(
            runs["visionzip"], "visionzip_repack_prefix", 0.50),
        "visionzip_fullload_sanity": _find_method(
            runs["visionzip"], "fullload", None),
    }
    role_runs = {
        "raster_fullload": "raster",
        "recomp": "calib4",
        "raster_prefix25": "raster",
        "morton_prefix25": "morton",
        "calib4_prefix25": "calib4",
        "visionzip_prefix25": "visionzip",
        "visionzip_prefix50": "visionzip",
        "visionzip_fullload_sanity": "visionzip",
    }
    if "matched" in runs:
        methods["matched_calib_prefix25"] = _find_method(
            runs["matched"], "layout_prefix_chunk", 0.25)
        role_runs["matched_calib_prefix25"] = "matched"
    roles = {
        role: runs[role_runs[role]]["groups"][method]
        for role, method in methods.items()
    }
    provenance = {
        role: {"run": role_runs[role], "method_key": method}
        for role, method in methods.items()
    }
    return roles, provenance


def _coverage_selected_normal(coverage: dict, image: str, layout: str,
                              fraction: float) -> list[int]:
    result = []
    for layer in range(N_LAYERS):
        value = coverage["physical_geometry"].get(
            (image, layer, layout, fraction))
        if value is None:
            raise AnalysisError(
                f"coverage lacks physical geometry {image}/{layer}/{layout}/{fraction}")
        result.append(value[2])
    return result


def _validate_roles(roles: dict, coverage: dict, ordered_keys) -> dict:
    """Validate exact prefix/full I/O and selection contracts request by request."""
    expected_keys = set(ordered_keys)
    maps = {}
    for role, rows in roles.items():
        mapping = _rows_by_key(rows)
        if len(rows) != N_QUESTIONS or set(mapping) != expected_keys:
            raise AnalysisError(f"{role}: role is not an exact paired 240-row arm")
        maps[role] = mapping

    # ReComp is a pixel-to-first-token baseline, not an SSD-cache layout.
    # Validate its no-retention/no-I/O contract separately so the Prefix and
    # separator invariants below are never (incorrectly) applied to it.
    recomp_rows = roles.get("recomp")
    if recomp_rows is None:
        raise AnalysisError("the exact-workload ReComp baseline is missing")
    zero_fields = (
        "ssd_read_ms", "ssd_read_bytes", "ssd_read_chunks", "ssd_preads",
        "normal_kv_read_bytes", "separator_read_bytes", "normal_kv_preads",
        "separator_preads", "total_actual_pread_bytes",
    )
    blank_fields = (
        "selector_ms", "scatter_ms", "prepare_ms", "n_chunks_selected",
        "n_chunks_total", "touched_chunk_fraction", "logical_kv_ratio",
        "normal_chunk_count_total", "static_score_calls", "query_score_calls",
        "diversity_calls", "selection_mode", "separator_policy",
        "reordered_prefix_store_validated",
    )
    prefill_deltas = []
    for row in recomp_rows:
        key = (str(row["image_id"]), str(row["question_id"]))
        if (row.get("method_key") != "recompute"
                or row.get("method") != "ReComp"
                or _number(row.get("retention"), "retention", blank=True)
                is not None
                or row.get("retention_kind") != "none"):
            raise AnalysisError(f"recomp/{key}: method/retention contract failed")
        for field in zero_fields:
            if not _close(_number(row.get(field), field), 0.0, atol=1e-12):
                raise AnalysisError(f"recomp/{key}: nonzero {field}")
        for field in blank_fields:
            if row.get(field) not in (None, ""):
                raise AnalysisError(f"recomp/{key}: unexpected {field}")
        if _json_cell(row.get("selected_chunk_ids_per_layer"),
                      f"recomp/{key}/selection") is not None:
            raise AnalysisError(f"recomp/{key}: unexpected chunk selection")
        delta = abs(_number(row["ttft_ms"], "ttft_ms")
                    - _number(row["prefill_ms"], "prefill_ms"))
        if delta > 1.0:
            raise AnalysisError(f"recomp/{key}: TTFT/prefill boundary differs")
        prefill_deltas.append(delta)
    recomp_validation = {
        "passed": True,
        "n_requests": len(recomp_rows),
        "n_images": len({str(row["image_id"]) for row in recomp_rows}),
        "ssd_read_bytes_total": 0,
        "ssd_preads_total": 0,
        "max_abs_ttft_minus_prefill_ms": max(prefill_deltas),
        "cache_layout_applicable": False,
        "cold_page_cache_applicable": False,
    }

    # FullLoad is the byte-per-token reference; repacking must not change its
    # exact payload accounting.
    row_bytes_by_image = {}
    for role in ("raster_fullload", "visionzip_fullload_sanity"):
        layout = _role_layout(role)
        for key in ordered_keys:
            row = maps[role][key]
            image = key[0]
            vn, _normal, _sep, nc = coverage["geometry"][image]
            normal_bytes = _integer(row["normal_kv_read_bytes"],
                                    f"{role}/{key}/normal_bytes")
            if normal_bytes % vn:
                raise AnalysisError(f"{role}/{key}: nonintegral bytes per visual row")
            row_bytes = normal_bytes // vn
            previous = row_bytes_by_image.setdefault(image, row_bytes)
            if previous != row_bytes:
                raise AnalysisError(f"FullLoad payload size changed after repacking: {image}")
            expected_chunks = 2 * N_LAYERS * nc
            values = {
                "ssd": _integer(row["ssd_read_bytes"], "ssd_read_bytes"),
                "actual": _integer(row["total_actual_pread_bytes"],
                                   "total_actual_pread_bytes"),
                "normal": normal_bytes,
                "sep": _integer(row["separator_read_bytes"],
                                "separator_read_bytes"),
                "chunks": _integer(row["ssd_read_chunks"], "ssd_read_chunks"),
                "normal_preads": _integer(row["normal_kv_preads"],
                                           "normal_kv_preads"),
                "sep_preads": _integer(row["separator_preads"],
                                        "separator_preads"),
                "preads": _integer(row["ssd_preads"], "ssd_preads"),
            }
            if (values["ssd"] != normal_bytes or values["actual"] != normal_bytes
                    or values["sep"] != 0 or values["chunks"] != expected_chunks
                    or values["normal_preads"] != 2 * N_LAYERS
                    or values["sep_preads"] != 0
                    or values["preads"] != 2 * N_LAYERS):
                raise AnalysisError(f"{role}/{key}: FullLoad I/O contract failed: {values}")
            if row.get("separator_policy") not in ("", None, "sidecar"):
                raise AnalysisError(f"{role}/{key}: unexpected separator policy")
            if role == "visionzip_fullload_sanity":
                if (row.get("physical_layout") != layout
                        or _integer(row.get("calibration_questions"),
                                    "calibration_questions") != 0):
                    raise AnalysisError(f"{role}/{key}: layout provenance differs")
            for field, numerator, denominator in (
                ("mean_bytes_per_pread", values["ssd"], values["preads"]),
                ("normal_mean_bytes_per_pread", values["normal"],
                 values["normal_preads"]),
                ("separator_mean_bytes_per_pread", 0, 0),
            ):
                if field in row and row[field] not in (None, ""):
                    expected = numerator / denominator if denominator else 0.0
                    if not _close(_number(row[field], field), expected, atol=1e-5):
                        raise AnalysisError(f"{role}/{key}: {field} mismatch")

    selector_stats = {}
    selection_fingerprints = {}
    for role, rows in roles.items():
        budget = ROLE_META[role][2]
        if budget is None:
            continue
        layout = _role_layout(role)
        by_image = defaultdict(list)
        for row in rows:
            by_image[str(row["image_id"])].append(row)
        for image, image_rows in by_image.items():
            vn, _normal, sep, nc = coverage["geometry"][image]
            k = _budget_chunks(nc, budget)
            parsed = [_chunk_layers(row["selected_chunk_ids_per_layer"],
                                    f"{role}/{image}/{row['question_id']}")
                      for row in image_rows]
            if any(value != parsed[0] for value in parsed[1:]):
                raise AnalysisError(f"{role}/{image}: selection depends on question")
            layers = parsed[0]
            if (len(layers) != N_LAYERS
                    or any(ids != list(range(k)) for ids in layers)):
                raise AnalysisError(f"{role}/{image}: selection is not exact first-k")
            selection_fingerprints[f"{role}/{image}"] = _stable_hash(layers)

            row_bytes = row_bytes_by_image[image]
            selected_rows = min(k * CHUNK_SIZE, vn)
            expected_normal_bytes = selected_rows * row_bytes
            expected_sep_bytes = sep * row_bytes
            expected_normal_preads = 2 * N_LAYERS  # one contiguous run/K,V/layer
            expected_total_preads = expected_normal_preads + 1
            selected_normal = _coverage_selected_normal(
                coverage, image, layout, budget)
            expected_logical_ratio = float(np.mean(
                [(value + sep) / vn for value in selected_normal]))
            for row in image_rows:
                key = (image, str(row["question_id"]))
                if (row.get("selection_mode") != "prefix"
                        or row.get("separator_policy") != "sidecar"
                        or not _bool(row.get("reordered_prefix_store_validated"),
                                     f"{role}/{key}/store_validated")):
                    raise AnalysisError(f"{role}/{key}: Prefix runtime contract failed")
                if any(_integer(row[field], f"{role}/{key}/{field}") != 0
                       for field in ("static_score_calls", "query_score_calls",
                                     "diversity_calls")):
                    raise AnalysisError(f"{role}/{key}: Prefix used scoring/diversity")
                if (_integer(row["normal_chunk_count_total"], "normal_chunks")
                        != N_LAYERS * k
                        or _integer(row["ssd_read_chunks"], "ssd_chunks")
                        != 2 * N_LAYERS * k
                        or not _close(_number(row["n_chunks_selected"], "selected"), k)
                        or _integer(row["n_chunks_total"], "total_chunks") != nc
                        or not _close(_number(row["touched_chunk_fraction"], "fraction"),
                                      k / nc, atol=1e-12)):
                    raise AnalysisError(f"{role}/{key}: chunk budget contract failed")
                normal_bytes = _integer(row["normal_kv_read_bytes"], "normal_bytes")
                sep_bytes = _integer(row["separator_read_bytes"], "sep_bytes")
                preads = _integer(row["ssd_preads"], "preads")
                if (normal_bytes != expected_normal_bytes
                        or sep_bytes != expected_sep_bytes
                        or _integer(row["ssd_read_bytes"], "ssd_bytes")
                        != expected_normal_bytes + expected_sep_bytes
                        or _integer(row["total_actual_pread_bytes"], "actual_bytes")
                        != expected_normal_bytes + expected_sep_bytes
                        or _integer(row["normal_kv_preads"], "normal_preads")
                        != expected_normal_preads
                        or _integer(row["separator_preads"], "sep_preads") != 1
                        or preads != expected_total_preads):
                    raise AnalysisError(f"{role}/{key}: exact byte/pread contract failed")
                for field, numerator, denominator in (
                    ("mean_bytes_per_pread", normal_bytes + sep_bytes, preads),
                    ("normal_mean_bytes_per_pread", normal_bytes,
                     expected_normal_preads),
                    ("separator_mean_bytes_per_pread", sep_bytes, 1),
                ):
                    if field in row and row[field] not in (None, ""):
                        if not _close(_number(row[field], field),
                                      numerator / denominator, atol=1e-5):
                            raise AnalysisError(f"{role}/{key}: {field} mismatch")
                if row.get("logical_kv_ratio") not in (None, "") and not _close(
                        _number(row["logical_kv_ratio"], "logical_kv_ratio"),
                        expected_logical_ratio, atol=1e-9):
                    raise AnalysisError(f"{role}/{key}: logical KV ratio mismatch")
                if not _close(_number(row["retention"], "retention"), budget):
                    raise AnalysisError(f"{role}/{key}: retention differs")
                if role != "calib4_prefix25":
                    expected_calib = ROLE_META[role][3]
                    runtime_layout = _runtime_layout(role)
                    if (row.get("physical_layout") != runtime_layout
                            or _integer(row.get("calibration_questions"),
                                        "calibration_questions") != expected_calib
                            or row.get("retrieval") != "prefix"
                            or row.get("validated_prefix_layout") != runtime_layout):
                        raise AnalysisError(f"{role}/{key}: provenance columns differ")
                prepare = _number(row["prepare_ms"], "prepare_ms")
                prefill = _number(row["prefill_ms"], "prefill_ms")
                ttft = _number(row["ttft_ms"], "ttft_ms")
                if abs(ttft - prepare - prefill) > 1.0:
                    raise AnalysisError(f"{role}/{key}: TTFT boundary algebra failed")

        selector = [_number(row["selector_ms"], f"{role}/selector_ms")
                    for row in rows]
        mean, _p50, p95 = _percentiles(selector)
        selector_stats[role] = {
            "n": len(selector), "min_ms": min(selector),
            "mean_ms": mean, "p95_ms": p95,
            "max_allowed_mean_ms": VERDICT_THRESHOLDS["max_selector_mean_ms"],
            "max_allowed_p95_ms": VERDICT_THRESHOLDS["max_selector_p95_ms"],
            "within_threshold": (
                min(selector) >= 0
                and mean <= VERDICT_THRESHOLDS["max_selector_mean_ms"]
                and p95 <= VERDICT_THRESHOLDS["max_selector_p95_ms"]),
        }
        if min(selector) < 0:
            raise AnalysisError(f"{role}: negative Prefix selector timing")
    return {
        "maps": maps,
        "row_bytes_by_image": row_bytes_by_image,
        "recomp": recomp_validation,
        "selector_stats": selector_stats,
        "selection_fingerprints": selection_fingerprints,
    }


def _command_option(command: str, option: str) -> str:
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise AnalysisError(f"cannot parse recorded command: {exc}") from exc
    values = []
    for index, token in enumerate(tokens):
        if token == option:
            if index + 1 >= len(tokens):
                raise AnalysisError(f"recorded command has dangling {option}")
            values.append(tokens[index + 1])
        elif token.startswith(option + "="):
            values.append(token.split("=", 1)[1])
    if len(values) != 1:
        raise AnalysisError(
            f"recorded command must contain {option} exactly once, got {values}")
    return values[0]


def _declared_path(value: str) -> Path:
    path = Path(value)
    return Path(os.path.abspath(os.fspath(path if path.is_absolute()
                                         else ROOT / path)))


def _crosscheck_profiles_and_coverage(profiles: dict, runs: dict,
                                      coverage: dict) -> dict:
    links = (("raster", "raster"), ("visionzip", "visionzip_image_only"))
    binding = {}
    for run_name, profile_name in links:
        profile = profiles[profile_name]
        run = runs[run_name]
        profile_store = _declared_path(profile["document"]["store"])
        run_store = _declared_path(_command_option(
            run["summary"].get("command", ""), "--store"))
        if profile_store != run_store:
            raise AnalysisError(
                f"{run_name}: build profile store and evaluation store differ: "
                f"{profile_store} != {run_store}")
        binding[run_name] = str(run_store)
    calib_store = _declared_path(_command_option(
        runs["calib4"]["summary"].get("command", ""), "--store"))
    canonical_calib_store = ROOT / "kvstore"
    if calib_store != canonical_calib_store:
        raise AnalysisError(
            f"historical calib4 run does not name canonical store: {calib_store}")
    binding["calib4"] = str(calib_store)

    for layout, profile_name in (("raster", "raster"),
                                 ("visionzip_image_only",
                                  "visionzip_image_only")):
        profile = profiles[profile_name]
        for image, row in profile["by_image"].items():
            vn, normal, sep, chunks = coverage["geometry"][image]
            if (_integer(row["v_token_num"], "v_token_num") != vn
                    or _integer(row["real_visual_tokens"],
                                "real_visual_tokens") != normal
                    or _integer(row["separator_tokens"],
                                "separator_tokens") != sep
                    or _integer(row["n_chunks"], "n_chunks") != chunks):
                raise AnalysisError(
                    f"{layout}/{image}: build profile and coverage geometry differ")
            if layout == "visionzip_image_only":
                expected = row["permutation_sha256"]
                observed = {
                    next(iter(values))
                    for (im, _layer, lay), values in
                    coverage["_order_hashes"].items()
                    if im == image and lay == layout
                }
                if observed != {expected}:
                    raise AnalysisError(
                        f"{layout}/{image}: profile/coverage permutation hash differs")
            else:
                observed = {
                    next(iter(values))
                    for (im, _layer, lay), values in
                    coverage["_order_hashes"].items()
                    if im == image and lay == layout
                }
                if observed != {_permutation_sha256(range(vn))}:
                    raise AnalysisError(
                        f"raster/{image}: coverage order is not identity")
    return {"run_store_bindings": binding,
            "visionzip_profile_order_matches_coverage": True,
            "profile_geometry_matches_coverage": True}


def _crosscheck_audit_sources(audit: dict, runs: dict) -> dict:
    inputs = audit["document"]["inputs"]
    expected = {
        "raster_run": runs["raster"]["path"],
        "visionzip_direct_run": runs["visionzip"]["path"],
        "raster_per_request": runs["raster"]["files"]["csv"],
        "visionzip_per_request": runs["visionzip"]["files"]["csv"],
    }
    for field, path in expected.items():
        if field not in inputs or _declared_path(inputs[field]) != path:
            raise AnalysisError(
                f"full-load audit source {field} does not bind to CLI run")
    mapped = audit["document"]["mapped_logical_kv_audit"]
    expected_raster_store = _declared_path(_command_option(
        runs["raster"]["summary"]["command"], "--store"))
    expected_vision_store = _declared_path(_command_option(
        runs["visionzip"]["summary"]["command"], "--store"))
    if (_declared_path(mapped.get("reference_store", ""))
            != expected_raster_store
            or _declared_path(mapped.get("candidate_store", ""))
            != expected_vision_store):
        raise AnalysisError(
            "mapped logical-K/V audit stores do not bind to evaluation stores")
    result = {field: str(path) for field, path in expected.items()}
    result.update({
        "mapped_reference_store": str(expected_raster_store),
        "mapped_candidate_store": str(expected_vision_store),
        "reference_store_posthoc_state_acknowledged": bool(
            mapped.get("reference_store_layout_at_audit") != "raster"),
    })
    return result


def _full_load_identity(role_validation: dict, external: dict, audit: dict,
                        ordered_keys) -> dict:
    maps = role_validation["maps"]
    raster = maps["raster_fullload"]
    repacked = maps["visionzip_fullload_sanity"]
    predictions = [raster[key]["prediction"] == repacked[key]["prediction"]
                   for key in ordered_keys]
    first_tokens = [
        _integer(raster[key].get("first_token_id"), "raster.first_token_id")
        == _integer(repacked[key].get("first_token_id"),
                    "repacked.first_token_id")
        for key in ordered_keys
    ]
    raster_accuracy = float(np.mean([
        _number(raster[key]["correct"], "correct") for key in ordered_keys]))
    repacked_accuracy = float(np.mean([
        _number(repacked[key]["correct"], "correct") for key in ordered_keys]))
    disagreements = []
    for key in ordered_keys:
        if (raster[key]["prediction"] != repacked[key]["prediction"]
                or _integer(raster[key].get("first_token_id"), "first_token_id")
                != _integer(repacked[key].get("first_token_id"),
                            "first_token_id")):
            disagreements.append({
                "image_id": key[0], "question_id": key[1],
                "question": raster[key]["question"],
                "raster_prediction": raster[key]["prediction"],
                "repacked_prediction": repacked[key]["prediction"],
                "raster_first_token_id": _integer(
                    raster[key].get("first_token_id"), "first_token_id"),
                "repacked_first_token_id": _integer(
                    repacked[key].get("first_token_id"), "first_token_id"),
                "raster_correct": _number(raster[key]["correct"], "correct"),
                "repacked_correct": _number(repacked[key]["correct"], "correct"),
            })
    result = {
        "n_requests": N_QUESTIONS,
        "prediction_agreement": float(np.mean(predictions)),
        "first_token_agreement": float(np.mean(first_tokens)),
        "accuracy_raster": raster_accuracy,
        "accuracy_repacked": repacked_accuracy,
        "accuracy_repacked_minus_raster_pp":
            (repacked_accuracy - raster_accuracy) * 100,
        "n_prediction_or_first_token_disagreements": len(disagreements),
        "disagreements": disagreements,
    }
    result["strict_full_load_exact_pass"] = (
        result["prediction_agreement"] == 1.0
        and result["first_token_agreement"] == 1.0
        and _close(raster_accuracy, repacked_accuracy, atol=1e-12))

    claimed = external["document"]["full_load_identity"]
    for field in ("prediction_agreement", "first_token_agreement"):
        if not _close(_agreement(claimed[field], f"external.{field}"),
                      result[field], atol=1e-12):
            raise AnalysisError(f"external/run FullLoad {field} disagrees")
    for field in ("accuracy_raster", "accuracy_repacked"):
        if not _close(_number(claimed[field], f"external.{field}"),
                      result[field], atol=1e-12):
            raise AnalysisError(f"external/run FullLoad {field} disagrees")
    if (external["strict_full_load_exact_pass"]
            != result["strict_full_load_exact_pass"]):
        raise AnalysisError("external/run strict FullLoad pass flag disagrees")
    behavioral = audit["document"]["behavioral_equivalence"]
    audit_values = {
        "prediction_agreement": _agreement(
            behavioral["prediction_agreement"], "audit.prediction_agreement"),
        "first_token_agreement": _agreement(
            behavioral["first_token_agreement"], "audit.first_token_agreement"),
        "accuracy_raster": _number(
            behavioral["raster_accuracy"], "audit.raster_accuracy"),
        "accuracy_repacked": _number(
            behavioral["visionzip_direct_accuracy"],
            "audit.visionzip_direct_accuracy"),
    }
    for field, value in audit_values.items():
        if not _close(value, result[field], atol=1e-12):
            raise AnalysisError(f"full-load audit/run {field} disagrees")
    if audit["strict_full_load_exact_pass"] != result[
            "strict_full_load_exact_pass"]:
        raise AnalysisError("full-load audit/run strict flag disagrees")
    mapped = audit["document"]["mapped_logical_kv_audit"]
    result.update({
        "structural_integrity_pass": audit["structural_integrity_pass"],
        "bf16_numerical_warning": audit["bf16_numerical_warning"],
        "mapped_logical_kv_elements_compared": _integer(
            mapped["total_elements_compared"], "total_elements_compared"),
        "mapped_logical_kv_elements_differing": _integer(
            mapped["total_differing_elements"], "total_differing_elements"),
        "mapped_logical_kv_max_absolute_difference": _number(
            mapped["global_max_absolute_difference"],
            "global_max_absolute_difference"),
        "all_sys_kv_hashes_equal": _bool(
            mapped["all_sys_kv_hashes_equal"], "all_sys_kv_hashes_equal"),
        "all_v_hidden_hashes_equal": _bool(
            mapped["all_v_hidden_hashes_equal"],
            "all_v_hidden_hashes_equal"),
    })
    audit_keys = {(str(row["image_id"]), str(row["question_id"]))
                  for row in behavioral["mismatches"]}
    result_keys = {(row["image_id"], row["question_id"])
                   for row in disagreements}
    if audit_keys != result_keys:
        raise AnalysisError("full-load audit/run disagreement keys differ")
    mapped_images = {str(row["image_id"]) for row in mapped.get("images", [])
                     if isinstance(row, dict) and "image_id" in row}
    if mapped_images != {key[0] for key in result_keys}:
        raise AnalysisError(
            "mapped logical-K/V audit images do not equal behavioral mismatches")
    io_fields = audit["document"]["io_equivalence"].get("fields_checked")
    expected_io_fields = {
        "ssd_read_bytes", "ssd_read_chunks", "ssd_preads",
        "normal_kv_read_bytes", "separator_read_bytes", "normal_kv_preads",
        "separator_preads", "total_actual_pread_bytes",
    }
    if not isinstance(io_fields, list) or set(io_fields) != expected_io_fields:
        raise AnalysisError("full-load audit I/O field set differs")
    for key in ordered_keys:
        for field in expected_io_fields:
            if not _close(_number(raster[key][field], field),
                          _number(repacked[key][field], field), atol=1e-12):
                raise AnalysisError(f"FullLoad raster/repack I/O differs: {key}/{field}")
    return result


def _mean(rows: list[dict], field: str, *, blank=False) -> float | None:
    values = [_number(row.get(field), field, blank=blank) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(np.mean(values))


def _summarize_one(role: str, rows: list[dict], source: dict) -> dict:
    layout_label, arm_label, budget, calibration, main, sanity = ROLE_META[role]
    answer = {
        "analysis_role": role,
        "layout": layout_label,
        "arm": arm_label,
        "budget": "" if budget is None else budget,
        "calibration_questions": calibration,
        "main_table": main,
        "sanity_only": sanity,
        "source_run": source["run"],
        "source_method_key": source["method_key"],
        "n_requests": len(rows),
        "n_images": len({str(row["image_id"]) for row in rows}),
        "accuracy": _mean(rows, "correct"),
    }
    for field, prefix in (
        ("ttft_ms", "ttft"), ("decode_ms", "decode"),
        ("e2e_latency_ms", "e2e"), ("selector_ms", "selector"),
        ("ssd_read_ms", "ssd_read"), ("scatter_ms", "scatter"),
        ("prepare_ms", "prepare"), ("prefill_ms", "prefill"),
    ):
        values = [_number(row.get(field), field, blank=True) for row in rows]
        values = [value for value in values if value is not None]
        if values:
            mean, p50, p95 = _percentiles(values)
            answer[prefix + "_mean_ms"] = mean
            answer[prefix + "_p50_ms"] = p50
            answer[prefix + "_p95_ms"] = p95
        else:
            answer[prefix + "_mean_ms"] = ""
            answer[prefix + "_p50_ms"] = ""
            answer[prefix + "_p95_ms"] = ""
    for field in (
        "ssd_read_bytes", "total_actual_pread_bytes",
        "normal_kv_read_bytes", "separator_read_bytes",
        "ssd_preads", "normal_kv_preads", "separator_preads",
        "ssd_read_chunks", "normal_chunk_count_total", "n_chunks_selected",
        "n_chunks_total", "touched_chunk_fraction", "logical_kv_ratio",
    ):
        answer[field + "_mean"] = _mean(rows, field, blank=True)
    answer["ssd_read_bytes_total"] = int(sum(
        _integer(row["ssd_read_bytes"], "ssd_read_bytes") for row in rows))
    answer["ssd_read_mb_mean"] = answer["ssd_read_bytes_mean"] / 1_000_000
    total_bytes = sum(_integer(row["ssd_read_bytes"], "ssd_read_bytes")
                      for row in rows)
    total_preads = sum(_integer(row["ssd_preads"], "ssd_preads")
                       for row in rows)
    normal_bytes = sum(_integer(row["normal_kv_read_bytes"], "normal_bytes")
                       for row in rows)
    normal_preads = sum(_integer(row["normal_kv_preads"], "normal_preads")
                        for row in rows)
    sep_bytes = sum(_integer(row["separator_read_bytes"], "sep_bytes")
                    for row in rows)
    sep_preads = sum(_integer(row["separator_preads"], "sep_preads")
                     for row in rows)
    answer.update({
        "mean_bytes_per_pread_pooled": (
            total_bytes / total_preads if total_preads else ""),
        "normal_mean_bytes_per_pread_pooled": (
            normal_bytes / normal_preads if normal_preads else ""),
        "separator_mean_bytes_per_pread_pooled": (
            sep_bytes / sep_preads if sep_preads else ""),
    })
    return answer


def _paired_statistics(roles: dict, ordered_keys) -> dict:
    comparisons = (
        ("recomp", "raster_fullload"),
        ("visionzip_prefix25", "visionzip_fullload_sanity"),
        ("visionzip_prefix25", "raster_fullload"),
        ("visionzip_prefix25", "calib4_prefix25"),
        ("visionzip_prefix25", "raster_prefix25"),
        ("visionzip_prefix25", "morton_prefix25"),
        ("visionzip_prefix50", "raster_fullload"),
        ("visionzip_prefix50", "calib4_prefix25"),
        ("visionzip_fullload_sanity", "raster_fullload"),
    )
    if "matched_calib_prefix25" in roles:
        comparisons += (("visionzip_prefix25", "matched_calib_prefix25"),)
    for role in ROLE_ORDER:
        pair = (role, "recomp")
        if role in roles and role != "recomp" and pair not in comparisons:
            comparisons += (pair,)
    maps = {role: _rows_by_key(rows) for role, rows in roles.items()}
    images = [key[0] for key in ordered_keys]
    result = {}
    for a, b in comparisons:
        av = [_number(maps[a][key]["correct"], "correct")
              for key in ordered_keys]
        bv = [_number(maps[b][key]["correct"], "correct")
              for key in ordered_keys]
        result[f"{a}_minus_{b}"] = {
            "a_role": a,
            "b_role": b,
            **_paired_bootstrap(av, bv, images),
        }
    return result


def _summary_rows(roles: dict, provenance: dict,
                  paired: dict) -> list[dict]:
    rows = [_summarize_one(role, roles[role], provenance[role])
            for role in ROLE_ORDER if role in roles]
    by_role = {row["analysis_role"]: row for row in rows}
    references = {
        "raster_fullload": by_role["raster_fullload"],
        "recomp": by_role["recomp"],
        "calib4_prefix25": by_role["calib4_prefix25"],
        "raster_prefix25": by_role["raster_prefix25"],
        "morton_prefix25": by_role["morton_prefix25"],
        "visionzip_fullload_sanity": by_role["visionzip_fullload_sanity"],
    }
    for row in rows:
        for ref_name, ref in references.items():
            row[f"accuracy_delta_vs_{ref_name}_pp"] = (
                row["accuracy"] - ref["accuracy"]) * 100
            row[f"ttft_reduction_vs_{ref_name}_pct"] = (
                (ref["ttft_mean_ms"] - row["ttft_mean_ms"])
                / ref["ttft_mean_ms"] * 100)
            row[f"ssd_bytes_ratio_vs_{ref_name}"] = (
                row["ssd_read_bytes_mean"] / ref["ssd_read_bytes_mean"]
                if ref["ssd_read_bytes_mean"] else "")
            row[f"total_preads_ratio_vs_{ref_name}"] = (
                row["ssd_preads_mean"] / ref["ssd_preads_mean"]
                if ref["ssd_preads_mean"] else "")
        comparison = f"{row['analysis_role']}_minus_raster_fullload"
        row["paired_image_cluster_ci95_vs_raster_fullload_pp"] = (
            json.dumps(paired[comparison]["image_cluster_delta_ci95_pp"])
            if comparison in paired else "")
        row["mcnemar_p_vs_raster_fullload"] = (
            paired[comparison]["mcnemar"]["p_exact_two_sided"]
            if comparison in paired else "")
        local_comparison = (
            f"{row['analysis_role']}_minus_visionzip_fullload_sanity")
        row["paired_image_cluster_ci95_vs_same_layout_fullload_pp"] = (
            json.dumps(paired[local_comparison]["image_cluster_delta_ci95_pp"])
            if local_comparison in paired else "")
        row["mcnemar_p_vs_same_layout_fullload"] = (
            paired[local_comparison]["mcnemar"]["p_exact_two_sided"]
            if local_comparison in paired else "")
        recomp_comparison = f"{row['analysis_role']}_minus_recomp"
        row["paired_image_cluster_ci95_vs_recomp_pp"] = (
            json.dumps(paired[recomp_comparison][
                "image_cluster_delta_ci95_pp"])
            if recomp_comparison in paired else "")
        row["mcnemar_p_vs_recomp"] = (
            paired[recomp_comparison]["mcnemar"]["p_exact_two_sided"]
            if recomp_comparison in paired else "")
    return rows


def _decide_verdict(summary_rows: list[dict]) -> dict:
    by_role = {row["analysis_role"]: row for row in summary_rows}
    vz = by_role["visionzip_prefix25"]
    best_control_accuracy = max(by_role["raster_prefix25"]["accuracy"],
                                by_role["morton_prefix25"]["accuracy"])
    metrics = {
        "accuracy_drop_vs_same_layout_fullload_pp":
            (by_role["visionzip_fullload_sanity"]["accuracy"]
             - vz["accuracy"]) * 100,
        "accuracy_drop_vs_raster_fullload_pp":
            (by_role["raster_fullload"]["accuracy"] - vz["accuracy"]) * 100,
        "accuracy_drop_vs_calib4_prefix25_pp":
            (by_role["calib4_prefix25"]["accuracy"] - vz["accuracy"]) * 100,
        "accuracy_gain_vs_best_raster_or_morton_pp":
            (vz["accuracy"] - best_control_accuracy) * 100,
        "ssd_bytes_ratio_vs_repacked_fullload":
            vz["ssd_read_bytes_mean"] /
            by_role["visionzip_fullload_sanity"]["ssd_read_bytes_mean"],
        "ttft_reduction_vs_repacked_fullload_pct":
            (by_role["visionzip_fullload_sanity"]["ttft_mean_ms"]
             - vz["ttft_mean_ms"])
            / by_role["visionzip_fullload_sanity"]["ttft_mean_ms"] * 100,
        "total_preads_ratio_vs_calib4_prefix25":
            vz["ssd_preads_mean"] /
            by_role["calib4_prefix25"]["ssd_preads_mean"],
        "selector_mean_ms": vz["selector_mean_ms"],
        "selector_p95_ms": vz["selector_p95_ms"],
    }
    t = VERDICT_THRESHOLDS
    eps = 1e-9
    system_pass = (
        metrics["ssd_bytes_ratio_vs_repacked_fullload"]
        <= t["max_ssd_bytes_ratio_vs_repacked_fullload"] + eps
        and metrics["ttft_reduction_vs_repacked_fullload_pct"]
        >= t["min_ttft_reduction_vs_repacked_fullload_pct"] - eps
        and metrics["total_preads_ratio_vs_calib4_prefix25"]
        <= t["max_total_preads_ratio_vs_calib4_prefix25"] + eps
        and metrics["selector_mean_ms"] <= t["max_selector_mean_ms"] + eps
        and metrics["selector_p95_ms"] <= t["max_selector_p95_ms"] + eps)
    strong_quality = (
        metrics["accuracy_drop_vs_same_layout_fullload_pp"]
        <= t["strong_max_accuracy_drop_vs_same_layout_fullload_pp"] + eps
        and metrics["accuracy_drop_vs_calib4_prefix25_pp"]
        <= t["strong_max_accuracy_drop_vs_calib4_prefix25_pp"] + eps
        and metrics["accuracy_gain_vs_best_raster_or_morton_pp"]
        >= t["strong_min_gain_vs_best_raster_or_morton_pp"] - eps)
    partial_quality = (
        metrics["accuracy_gain_vs_best_raster_or_morton_pp"]
        >= t["partial_min_gain_vs_best_raster_or_morton_pp"] - eps)
    if system_pass and strong_quality:
        label = "STRONG"
    elif system_pass and partial_quality:
        label = "PARTIAL"
    else:
        label = "NO-GO"
    return {
        "verdict": label,
        "system_gate_passed": system_pass,
        "strong_quality_gate_passed": strong_quality,
        "partial_quality_gate_passed": partial_quality,
        "metrics": metrics,
        "thresholds": dict(VERDICT_THRESHOLDS),
        "rule": (
            "STRONG iff system gate, the same-layout FullLoad and calib4 "
            "quality drops are both <=2 pp, and "
            "VisionZip25 is no worse than best Raster/Morton Prefix25; "
            "PARTIAL iff system gate and VisionZip25 gains >=2 pp over the "
            "best Raster/Morton Prefix25 but misses STRONG; otherwise NO-GO."
        ),
    }


def _paths_overlap(a: Path, b: Path) -> bool:
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def _rename_noreplace(src_dir_fd: int, src_name: str,
                      dst_dir_fd: int, dst_name: str) -> None:
    """Atomically publish without an overwrite-capable fallback."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise AnalysisError(
            "renameat2(RENAME_NOREPLACE) unavailable; refusing publication")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p,
                          ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    rc = renameat2(src_dir_fd, os.fsencode(src_name),
                   dst_dir_fd, os.fsencode(dst_name), 1)
    if rc:
        err = ctypes.get_errno()
        if err == errno.EEXIST:
            raise AnalysisError(
                f"publication target appeared concurrently: {dst_name}")
        raise OSError(err, os.strerror(err), f"{src_name} -> {dst_name}")


def _input_records(runs: dict, profiles: dict, coverage: dict,
                   external: dict, audit: dict,
                   index_signature: tuple) -> list[dict]:
    records = [{
        "label": "frozen_index", "path": INDEX,
        "signature": index_signature, "sha256": INDEX_SHA256,
    }]
    for run_name, run in runs.items():
        for kind, path in run["files"].items():
            records.append({
                "label": f"run.{run_name}.{kind}", "path": path,
                "signature": run["signatures"][path],
                "sha256": run["hashes"][path],
            })
    for name, profile in profiles.items():
        records.append({
            "label": f"profile.{name}", "path": profile["path"],
            "signature": profile["signature"], "sha256": profile["sha256"],
        })
    for label, item in (("coverage", coverage),
                        ("external_validation", external),
                        ("full_load_240_audit", audit)):
        records.append({
            "label": label, "path": item["path"],
            "signature": item["signature"], "sha256": item["sha256"],
        })
    return records


def _recheck_inputs(records: list[dict]) -> None:
    changed = []
    for record in records:
        path = record["path"]
        if (not _lexists(path) or path.is_symlink() or not path.is_file()
                or _sig(path) != record["signature"]
                or _sha256(path) != record["sha256"]):
            changed.append(record["label"])
    if changed:
        raise AnalysisError(
            f"source artifacts changed during analysis: {changed}")


def _flatten_per_request(roles: dict, provenance: dict,
                         ordered_keys) -> tuple[list[str], list[dict]]:
    prefix = ["analysis_role", "layout", "arm", "main_table", "sanity_only",
              "source_run", "source_method_key"]
    source_fields = []
    for role in ROLE_ORDER:
        if role not in roles:
            continue
        for key in roles[role][0]:
            if key not in source_fields:
                source_fields.append(key)
    rows = []
    for role in ROLE_ORDER:
        if role not in roles:
            continue
        by_key = _rows_by_key(roles[role])
        layout, arm, _budget, _calib, main, sanity = ROLE_META[role]
        for key in ordered_keys:
            source = by_key[key]
            rows.append({
                "analysis_role": role,
                "layout": layout,
                "arm": arm,
                "main_table": main,
                "sanity_only": sanity,
                "source_run": provenance[role]["run"],
                "source_method_key": provenance[role]["method_key"],
                **source,
            })
    return prefix + source_fields, rows


def _csv_fields(rows: list[dict]) -> list[str]:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _fmt(value, digits=3) -> str:
    if value in (None, ""):
        return "NA"
    return f"{float(value):.{digits}f}"


def _readme(summary_rows: list[dict], coverage: dict, verdict: dict,
            full_identity: dict, profile_summary: dict,
            matched: bool) -> str:
    main_rows = [row for row in summary_rows if row["main_table"]]
    by_role = {row["analysis_role"]: row for row in summary_rows}
    recomp = by_role["recomp"]
    visionzip25 = by_role["visionzip_prefix25"]
    visionzip50 = by_role["visionzip_prefix50"]
    table = [
        "| Layout / arm | Accuracy | TTFT mean (ms) | SSD MB/request | "
        "preads (normal/sep/total) | selector mean/p95 (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        table.append(
            f"| {row['layout']} / {row['arm']} | "
            f"{row['accuracy'] * 100:.2f}% | {row['ttft_mean_ms']:.2f} | "
            f"{row['ssd_read_mb_mean']:.2f} | "
            f"{_fmt(row['normal_kv_preads_mean'], 1)}/"
            f"{_fmt(row['separator_preads_mean'], 1)}/"
            f"{_fmt(row['ssd_preads_mean'], 1)} | "
            f"{_fmt(row['selector_mean_ms'])}/"
            f"{_fmt(row['selector_p95_ms'])} |")
    coverage25 = [row for row in coverage["summary"]
                  if _close(row["physical_fraction_requested"], 0.25)]
    cov_table = [
        "| Layout | Importance source | Normal-token fraction | "
        "Coverage macro | Coverage global-mass |",
        "|---|---|---:|---:|---:|",
    ]
    for row in coverage25:
        cov_table.append(
            f"| {row['layout']} | {row['importance_source']} | "
            f"{row['mean_realized_normal_token_fraction'] * 100:.2f}% | "
            f"{row['macro_mean_coverage'] * 100:.2f}% | "
            f"{row['global_mass_weighted_coverage'] * 100:.2f}% |")
    inc = profile_summary["visionzip_incremental_ingestion_mean_ms"]
    decision_metrics = verdict["metrics"]
    optional = ("The optional matched-calibration separator-tail arm is included."
                if matched else
                "No optional matched-calibration run was supplied.")
    return "\n".join([
        "# Image-only VisionZip repack analysis",
        "",
        "This is a strict, CPU-only analysis of the frozen GQA-40 paired "
        "workload (questions[4:10], 240 requests). Source runs and historical "
        "results are read-only; this directory was published without overwrite.",
        "",
        "## Strict correctness warning",
        "",
        ("**PASS:** Raster and repacked FullLoad are exactly identical over "
         "all 240 requests."
         if full_identity["strict_full_load_exact_pass"] else
         "**WARNING — strict FullLoad exactness FAILS.** Raster and repacked "
         f"FullLoad prediction/first-token agreement is "
         f"{full_identity['prediction_agreement'] * 100:.2f}% / "
         f"{full_identity['first_token_agreement'] * 100:.2f}%, with "
         f"{full_identity['n_prediction_or_first_token_disagreements']} "
         "disagreements and a repacked-minus-raster accuracy delta of "
         f"{full_identity['accuracy_repacked_minus_raster_pp']:.2f} pp. This "
         "is never labeled an exact correctness pass."),
        "",
        f"Mapped FP16 Visual-KV structural integrity is "
        f"**{'PASS' if full_identity['structural_integrity_pass'] else 'FAIL'}**: "
        f"{full_identity['mapped_logical_kv_elements_compared']:,} elements "
        f"were compared and "
        f"{full_identity['mapped_logical_kv_elements_differing']:,} differed; "
        "`sys_kv` and `v_hidden` hashes also match. The audit classifies the "
        "two greedy flips as BF16 eager-attention reduction-order sensitivity, "
        "not a mapped-K/V corruption. See `validation.json`.",
        "",
        "## Research-method decision",
        "",
        f"**{verdict['verdict']}** — {verdict['rule']}",
        "",
        f"Observed VisionZip Prefix25: accuracy drop vs same-layout FullLoad "
        f"{decision_metrics['accuracy_drop_vs_same_layout_fullload_pp']:.2f} pp "
        f"(vs canonical Raster FullLoad "
        f"{decision_metrics['accuracy_drop_vs_raster_fullload_pp']:.2f} pp); "
        f"gain vs best Raster/Morton Prefix25 "
        f"{decision_metrics['accuracy_gain_vs_best_raster_or_morton_pp']:.2f} pp; "
        f"SSD ratio vs repacked FullLoad "
        f"{decision_metrics['ssd_bytes_ratio_vs_repacked_fullload']:.3f}; "
        f"TTFT reduction vs repacked FullLoad "
        f"{decision_metrics['ttft_reduction_vs_repacked_fullload_pct']:.2f}%.",
        "",
        "Thresholds are predeclared in `config.json`; they are operational "
        "decision thresholds, not learned equivalence margins.",
        "",
        "## Main paired results",
        "",
        *table,
        "",
        "ReComp is reused from the preserved calib4 schema-v2 run. Its image, "
        "question, gold-answer, model, decoding, and TTFT contracts match this "
        "frozen workload exactly; it recomputes from pixels and has zero KV-store "
        "reads. The historical artifact predates first-token-ID serialization, "
        "and absolute latency comparisons with the newer image-only runs are "
        "therefore explicitly cross-run.",
        "",
        f"Historical ReComp is {recomp['accuracy'] * 100:.2f}% at "
        f"{recomp['ttft_mean_ms']:.2f} ms TTFT. VisionZip Prefix25 changes "
        f"accuracy by {visionzip25['accuracy_delta_vs_recomp_pp']:.2f} pp and "
        f"reduces TTFT by {visionzip25['ttft_reduction_vs_recomp_pct']:.2f}%; "
        f"Prefix50 changes accuracy by "
        f"{visionzip50['accuracy_delta_vs_recomp_pp']:.2f} pp and reduces TTFT "
        f"by {visionzip50['ttft_reduction_vs_recomp_pct']:.2f}%.",
        "",
        "The VisionZip-layout FullLoad arm is sanity-only. Raster and repacked "
        f"FullLoad prediction/first-token agreement were both "
        f"{full_identity['prediction_agreement'] * 100:.1f}% / "
        f"{full_identity['first_token_agreement'] * 100:.1f}% over 240 requests. "
        f"{optional}",
        "",
        "Paired image-cluster bootstrap (10,000 resamples, seed 0), a "
        "question-level bootstrap supplement, and exact two-sided McNemar "
        "results are embedded in `config.json`. Accuracy is binary GQA "
        "normalized exact match.",
        "",
        "## 25% prefix importance coverage",
        "",
        *cov_table,
        "",
        "`importance_coverage.csv` is a byte-identical copy of the supplied "
        "long-form artifact. SparseVLM calib4 importance is analysis-only and "
        "was not used to build or serve image-only layouts.",
        "",
        "## One-time build cost",
        "",
        f"Mean incremental VisionZip ingestion cost versus raster was "
        f"{inc:.2f} ms/image. `layout_stats.csv` reports the observed one-time "
        "components and amortization over 1, 5, 10, and 20 requests/image.",
        "",
        "## Validation scope",
        "",
        "Every arm has the exact same 40 images and 240 question/gold pairs and "
        "uses schema-v2 true TTFT. SSD-backed arms additionally pass the relevant "
        "cold-cache, chunk-size-64, and measured byte/pread checks; Prefix arms "
        "also pass sidecar, exact first-k, and zero-scoring checks. ReComp is "
        "checked separately for no retention, no selection, zero SSD I/O, and "
        "TTFT/prefill consistency because cache layout and cold page-cache "
        "semantics do not apply to pixel recomputation. Profile provenance, "
        "profile/coverage geometry, and the independent validation artifact were "
        "checked fail-closed. No store tree or model/GPU serving is performed by "
        "this analyzer.",
        "",
    ])


def _prepare_destination(out_dir: Path, input_paths: list[Path]) -> tuple:
    out_dir = _resolve(out_dir)
    if out_dir not in (OUTPUT, RECOMP_OUTPUT):
        raise AnalysisError(
            f"--out-dir must be exactly {OUTPUT} or {RECOMP_OUTPUT}")
    _reject_symlinks(out_dir)
    for path in input_paths:
        if _paths_overlap(out_dir, path):
            raise AnalysisError(f"output overlaps an input artifact: {path}")
    existing_signature = None
    if _lexists(out_dir):
        if out_dir.is_symlink() or not out_dir.is_dir():
            raise AnalysisError(f"output is not a regular directory: {out_dir}")
        if any(out_dir.iterdir()):
            raise AnalysisError(f"refusing to overwrite nonempty output: {out_dir}")
        existing_signature = _sig(out_dir)
    parent = out_dir.parent
    _reject_symlinks(parent)
    if not parent.is_dir() or parent.is_symlink():
        raise AnalysisError(f"output parent is not a regular directory: {parent}")
    return out_dir, existing_signature


def _publish(out_dir: Path, existing_signature: tuple | None,
             records: list[dict], per_fields: list[str], per_rows: list[dict],
             summary_rows: list[dict], layout_rows: list[dict], coverage: dict,
             config: dict, validation: dict, readme: str) -> None:
    parent = out_dir.parent
    stage = Path(tempfile.mkdtemp(prefix=".image_only_repack.stage.", dir=parent))
    try:
        _write_json(stage / "config.json", config)
        _write_csv(stage / "per_request.csv", per_fields, per_rows)
        _write_csv(stage / "summary.csv", _csv_fields(summary_rows), summary_rows)
        _write_csv(stage / "layout_stats.csv", _csv_fields(layout_rows), layout_rows)
        shutil.copyfile(coverage["path"], stage / "importance_coverage.csv")
        if _sha256(stage / "importance_coverage.csv") != coverage["sha256"]:
            raise AnalysisError("coverage copy digest differs")
        _write_json(stage / "validation.json", validation)
        (stage / "README.md").write_text(readme)
        expected = {
            "config.json", "per_request.csv", "summary.csv",
            "layout_stats.csv", "importance_coverage.csv", "validation.json",
            "README.md",
        }
        if {path.name for path in stage.iterdir()} != expected:
            raise AnalysisError("staging output file set differs")
        _recheck_inputs(records)

        if existing_signature is not None:
            if (not _lexists(out_dir) or out_dir.is_symlink()
                    or not out_dir.is_dir() or _sig(out_dir) != existing_signature
                    or any(out_dir.iterdir())):
                raise AnalysisError("empty output target changed during analysis")
            os.rmdir(out_dir)
        elif _lexists(out_dir):
            raise AnalysisError("output target appeared during analysis")
        fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            _rename_noreplace(fd, stage.name, fd, out_dir.name)
        finally:
            os.close(fd)
    finally:
        if _lexists(stage):
            shutil.rmtree(stage)


def analyze(*, raster_run: Path, morton_run: Path, visionzip_run: Path,
            calib4_run: Path, matched_calib_run: Path | None,
            raster_build_profile: Path, visionzip_build_profile: Path,
            coverage_csv: Path, validation_json: Path,
            out_dir: Path) -> dict:
    ordered_keys, frozen = _frozen_workload()
    index_signature = _sig(INDEX)
    image_ids = {key[0] for key in ordered_keys}
    runs = {
        "raster": _load_run(raster_run, "raster", ordered_keys, frozen,
                            new_schema=True, expected_layout="raster"),
        "morton": _load_run(morton_run, "morton", ordered_keys, frozen,
                            new_schema=True, expected_layout="morton"),
        "visionzip": _load_run(
            visionzip_run, "visionzip", ordered_keys, frozen,
            new_schema=True, expected_layout="visionzip_image_only"),
        "calib4": _load_run(calib4_run, "calib4", ordered_keys, frozen,
                            new_schema=False, expected_layout=None),
    }
    if matched_calib_run is not None:
        runs["matched"] = _load_run(
            matched_calib_run, "matched", ordered_keys, frozen,
            new_schema=True, expected_layout="calib_importance_sep_tail")
    profiles = {
        "raster": _load_profile(raster_build_profile, "raster", image_ids),
        "visionzip_image_only": _load_profile(
            visionzip_build_profile, "visionzip_image_only", image_ids),
    }
    coverage = _load_coverage(coverage_csv, image_ids,
                              matched_calib_run is not None)
    external = _load_external_validation(validation_json)
    audit = _load_full_load_audit(external["path"])
    audit_sources = _crosscheck_audit_sources(audit, runs)

    profile_links = _crosscheck_profiles_and_coverage(
        profiles, runs, coverage)
    layout_rows, profile_summary = _profile_stats(
        profiles["raster"], profiles["visionzip_image_only"])
    roles, provenance = _extract_roles(runs)
    role_validation = _validate_roles(roles, coverage, ordered_keys)
    full_identity = _full_load_identity(
        role_validation, external, audit, ordered_keys)
    paired = _paired_statistics(roles, ordered_keys)
    summary_rows = _summary_rows(roles, provenance, paired)
    verdict = _decide_verdict(summary_rows)
    per_fields, per_rows = _flatten_per_request(
        roles, provenance, ordered_keys)

    records = _input_records(
        runs, profiles, coverage, external, audit, index_signature)
    input_paths = [record["path"] for record in records]
    destination, existing_signature = _prepare_destination(out_dir, input_paths)
    input_manifest = {
        record["label"]: {
            "path": str(record["path"]), "sha256": record["sha256"]}
        for record in records
    }
    config = {
        "schema_version": 2,
        "analysis": "image_only_visionzip_repack",
        "generated_unix_time": time.time(),
        "cpu_only_analyzer": True,
        "frozen_workload": {
            "dataset": "gqa", "index": str(INDEX),
            "index_sha256": INDEX_SHA256,
            "workload_sha256": WORKLOAD_SHA256,
            "images": N_IMAGES, "questions": N_QUESTIONS,
            "questions_per_image": QUESTIONS_PER_IMAGE,
            "question_slice": [SKIP, SKIP + QUESTIONS_PER_IMAGE],
            "schema_version": 2, "cold_page_cache": True,
            "chunk_size": CHUNK_SIZE, "separator_policy": "sidecar",
        },
        "model_configuration": {
            "model": "llava-hf/llava-v1.6-vicuna-7b-hf",
            "quantization": "4-bit NF4 double-quant",
            "compute_dtype": "bfloat16",
            "attention": "eager",
            "decoding": "greedy",
            "max_new_tokens": 16,
        },
        "proposed_method": {
            "name": "visionzip_repack_prefix",
            "importance": (
                "vision-encoder penultimate-layer CLS-to-patch attention, "
                "head-wise sum"),
            "layout_uses_dataset_question": False,
            "llm_used_for_layout_scoring": False,
            "calibration_questions": 0,
            "global_permutation_shared_by_all_llm_layers": True,
            "normal_patch_order": "stable descending saliency",
            "separator_policy": "stable physical tail plus mandatory sidecar",
            "online_retrieval": "sequential physical Prefix first-k chunks",
            "online_scoring": False,
        },
        "source_methods": provenance,
        "recomp_baseline": {
            "classification": "historical_exact_workload_schema_v2",
            "source_run": str(runs["calib4"]["path"]),
            "source_method_key": provenance["recomp"]["method_key"],
            "same_frozen_image_question_gold_pairs": True,
            "same_documented_model_decoding_ttft_contract": True,
            "kv_store_io_applicable": False,
            "cold_page_cache_applicable": False,
            "first_token_id_available": False,
            "latency_comparison_to_image_only_runs": "cross_run",
            "validation": role_validation["recomp"],
        },
        "inputs": input_manifest,
        "bootstrap": {"primary_unit": "image cluster",
                      "supplement_unit": "question",
                      "resamples": BOOTSTRAP_RESAMPLES,
                      "seed": BOOTSTRAP_SEED,
                      "paired_statistics": paired},
        "importance_coverage_summary": coverage["summary"],
        "build_profile_summary": profile_summary,
        "correctness_classification": {
            "strict_all_passed": external["strict_all_passed"],
            "strict_full_load_exact_pass":
                full_identity["strict_full_load_exact_pass"],
            "structural_integrity_pass":
                full_identity["structural_integrity_pass"],
            "bf16_numerical_warning":
                full_identity["bf16_numerical_warning"],
            "behavioral": full_identity,
            "full_load_audit_path": str(audit["path"]),
            "full_load_audit_sha256": audit["sha256"],
        },
        "research_method_decision": verdict,
    }
    validation = {
        "schema_version": 2,
        "all_passed": external["strict_all_passed"],
        "strict_all_passed": external["strict_all_passed"],
        "strict_full_load_exact_pass":
            full_identity["strict_full_load_exact_pass"],
        "structural_integrity_pass":
            full_identity["structural_integrity_pass"],
        "bf16_numerical_warning": full_identity["bf16_numerical_warning"],
        "research_analysis_complete": True,
        "checks": {
            "frozen_40x240_workload": {"passed": True},
            "historical_recomp_exact_frozen_workload": {"passed": True},
            "recomp_exact_240_zero_ssd_io": role_validation["recomp"],
            "run_json_csv_summary_sanity_agreement": {"passed": True},
            "schema_v2_cold_chunk64_sidecar": {"passed": True},
            "exact_prefix_first_k_budget": {"passed": True},
            "query_independent_selection_and_zero_counters": {"passed": True},
            "exact_measured_byte_and_pread_splits": {"passed": True},
            "true_ttft_timing_algebra": {"passed": True},
            "profile_run_store_binding": {"passed": True},
            "profile_coverage_geometry_and_order": {"passed": True},
            "full_load_raster_repack_strict_exact_identity": {
                "passed": full_identity["strict_full_load_exact_pass"],
                "classification": "PASS" if full_identity[
                    "strict_full_load_exact_pass"] else "WARNING/STRICT_FAIL",
            },
            "mapped_logical_fp16_kv_structural_integrity": {
                "passed": full_identity["structural_integrity_pass"]},
            "external_nonbehavioral_gates": {"passed": True},
            "external_strict_all_gates": {
                "passed": external["strict_all_passed"]},
            "source_artifacts_unchanged_before_publish": {"passed": True},
        },
        "full_load_identity": full_identity,
        "recomp_baseline": role_validation["recomp"],
        "prefix_selector_stats": role_validation["selector_stats"],
        "selection_fingerprints": role_validation["selection_fingerprints"],
        "profile_and_coverage": profile_links,
        "full_load_audit_source_bindings": audit_sources,
        "external_validation": {
            "path": str(external["path"]),
            "sha256": external["sha256"],
            "document": external["document"],
        },
        "full_load_240_audit": {
            "path": str(audit["path"]),
            "sha256": audit["sha256"],
            "document": audit["document"],
        },
        "artifact_hashes": input_manifest,
        "research_method_decision": verdict,
    }
    readme = _readme(summary_rows, coverage, verdict, full_identity,
                     profile_summary, matched_calib_run is not None)
    _publish(destination, existing_signature, records, per_fields, per_rows,
             summary_rows, layout_rows, coverage, config, validation, readme)
    return {
        "out_dir": str(destination),
        "verdict": verdict["verdict"],
        "n_roles": len(roles),
        "n_per_request_rows": len(per_rows),
        "strict_all_passed": external["strict_all_passed"],
        "strict_full_load_exact_pass":
            full_identity["strict_full_load_exact_pass"],
        "structural_integrity_pass":
            full_identity["structural_integrity_pass"],
        "analysis_published": True,
    }


def _self_test() -> None:
    assert _budget_chunks(34, 0.25) == 8  # Python/production ties-to-even
    assert _budget_chunks(36, 0.25) == 9
    layers = [[0, 1], [0, 1]]
    assert _chunk_layers(json.dumps(layers), "self-test") == layers
    for invalid in ([[0.0]], [[True]], [["0"]], [0]):
        try:
            _chunk_layers(invalid, "self-test-invalid")
        except AnalysisError:
            pass
        else:
            raise AssertionError(f"accepted non-integer chunk IDs: {invalid!r}")
    a = [1, 1, 0, 0, 1, 0] * N_IMAGES
    b = [1, 0, 0, 0, 1, 0] * N_IMAGES
    images = [f"i{i}" for i in range(N_IMAGES)
              for _ in range(QUESTIONS_PER_IMAGE)]
    first = _paired_bootstrap(a, b, images, n=1000, seed=7)
    second = _paired_bootstrap(a, b, images, n=1000, seed=7)
    assert first == second and _close(first["delta_pp"], 100 / 6)
    assert first["mcnemar"]["a_only"] == N_IMAGES

    def fake(role, accuracy, ttft, byte, preads, selector=0.1):
        return {"analysis_role": role, "accuracy": accuracy,
                "ttft_mean_ms": ttft, "ssd_read_bytes_mean": byte,
                "ssd_preads_mean": preads, "selector_mean_ms": selector,
                "selector_p95_ms": selector}
    base = [
        fake("raster_fullload", .60, 100, 1000, 64),
        fake("raster_prefix25", .55, 70, 260, 65),
        fake("morton_prefix25", .56, 70, 260, 65),
        fake("calib4_prefix25", .59, 70, 260, 65),
        fake("visionzip_prefix25", .59, 60, 260, 65),
        fake("visionzip_fullload_sanity", .60, 100, 1000, 64),
    ]
    assert _decide_verdict(base)["verdict"] == "STRONG"
    base[1]["accuracy"] = .54
    base[2]["accuracy"] = .55
    base[4]["accuracy"] = .57
    assert _decide_verdict(base)["verdict"] == "PARTIAL"
    base[4]["ttft_mean_ms"] = 90
    assert _decide_verdict(base)["verdict"] == "NO-GO"

    with tempfile.TemporaryDirectory(prefix="image-only-analysis-selftest-") as td:
        root = Path(td)
        (root / "source").write_text("new")
        (root / "existing").write_text("old")
        fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            try:
                _rename_noreplace(fd, "source", fd, "existing")
            except AnalysisError:
                pass
            else:
                raise AssertionError("RENAME_NOREPLACE overwrote a target")
            _rename_noreplace(fd, "source", fd, "published")
            assert (root / "published").read_text() == "new"
        finally:
            os.close(fd)
    print("self-test PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raster-run")
    parser.add_argument("--morton-run")
    parser.add_argument("--visionzip-run")
    parser.add_argument("--matched-calib-run", default=None)
    parser.add_argument("--calib4-run", default=str(DEFAULT_CALIB4_RUN))
    parser.add_argument("--raster-build-profile")
    parser.add_argument("--visionzip-build-profile")
    parser.add_argument("--coverage-csv")
    parser.add_argument("--validation-json")
    parser.add_argument("--out-dir")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    required = ("raster_run", "morton_run", "visionzip_run",
                "raster_build_profile", "visionzip_build_profile",
                "coverage_csv", "validation_json", "out_dir")
    missing = ["--" + name.replace("_", "-") for name in required
               if not getattr(args, name)]
    if missing:
        parser.error("required arguments missing: " + ", ".join(missing))
    result = analyze(
        raster_run=_resolve(args.raster_run),
        morton_run=_resolve(args.morton_run),
        visionzip_run=_resolve(args.visionzip_run),
        calib4_run=_resolve(args.calib4_run),
        matched_calib_run=(_resolve(args.matched_calib_run)
                           if args.matched_calib_run else None),
        raster_build_profile=_resolve(args.raster_build_profile),
        visionzip_build_profile=_resolve(args.visionzip_build_profile),
        coverage_csv=_resolve(args.coverage_csv),
        validation_json=_resolve(args.validation_json),
        out_dir=_resolve(args.out_dir),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

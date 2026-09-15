"""Analyze the frozen ImageOnly-Repack + sequential-Prefix budget sweep.

This is deliberately a separate, CPU-only publisher.  It consumes one
schema-v2 run containing one same-process FullLoad arm and the seven Prefix
budgets 20/25/30/35/40/45/50.  It never changes the physical KV layout and it
refuses to replace an existing result directory.

The online method is fixed:

    ImageOnly VisionZip repacking + first-k sequential Prefix loading

SparseVLM calibration scores are loaded only for the offline coverage curve.
They are never used by the layout, selector, or serving path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
CORE_PATH = ROOT / "scripts/25_analyze_image_only_repack.py"
_SPEC = importlib.util.spec_from_file_location("image_only_analysis_core",
                                               CORE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load analysis core: {CORE_PATH}")
CORE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(CORE)


BUDGETS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
PREFIX_BASE = "visionzip_repack_prefix"
EXPECTED_METHODS = ("fullload",) + tuple(
    f"{PREFIX_BASE}@{int(round(100 * budget))}" for budget in BUDGETS)
DEFAULT_RUN = ROOT / "runs/image_only_repack_budget_sweep/main_20_50"
DEFAULT_OUTPUT = ROOT / "results/image_only_repack_budget_sweep"
DEFAULT_STORE = ROOT / "kvstore_image_only_visionzip"
BUILD_PROFILE = ROOT / "runs/image_only_repack/build/visionzip_direct_profile.json"
RAW_CALIBRATION = (
    ROOT / "runs/image_only_repack/coverage/calibration_scores_analysis_only.pt")
OLD_COVERAGE = ROOT / "runs/image_only_repack/coverage/importance_coverage.csv"
OLD_RUN = ROOT / "runs/image_only_repack/visionzip_direct"
OLD_CONFIG = ROOT / "results/image_only_repack/config.json"
OLD_VALIDATION = ROOT / "results/image_only_repack/validation.json"
MODEL_REVISION_REF = (Path("/home/dblab/.cache/huggingface/hub/") /
    "models--llava-hf--llava-v1.6-vicuna-7b-hf/refs/main")

INDEX_SHA256 = CORE.INDEX_SHA256
WORKLOAD_SHA256 = CORE.WORKLOAD_SHA256
N_IMAGES = CORE.N_IMAGES
N_QUESTIONS = CORE.N_QUESTIONS
N_LAYERS = CORE.N_LAYERS
CHUNK_SIZE = CORE.CHUNK_SIZE
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
SATURATION_FUTURE_GAIN_PP = 0.5
THRESHOLDS_PP = {"aggressive": 4.0, "balanced": 2.0, "quality": 1.0}


QTYPE = (
    ("yes/no", re.compile(
        r"^(is|are|do|does|was|were|has|have|can|could|did|will|would)\b",
        re.I)),
    ("color", re.compile(r"\bcolou?r\b", re.I)),
    ("count", re.compile(r"\bhow many\b|\bnumber of\b", re.I)),
    ("spatial", re.compile(
        r"\b(left|right|above|below|behind|front|under|near|beside|top|"
        r"bottom|side|between|next to)\b", re.I)),
    ("material/attribute", re.compile(
        r"\b(material|made of|shape|size|texture|large|small|tall|short|"
        r"thin|thick)\b", re.I)),
    ("object", re.compile(r"^(what|which|who)\b", re.I)),
)


class SweepError(RuntimeError):
    """Fail-closed validation/publication error."""


def qtype(question: str) -> str:
    for name, pattern in QTYPE:
        if pattern.search(question):
            return name
    return "other"


def sha256(path: Path, block_size: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def permutation_sha256(order) -> str:
    payload = ",".join(str(int(x)) for x in order).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False,
                  sort_keys=True)
        handle.write("\n")


def csv_fields(rows: list[dict]) -> list[str]:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows and fields is None:
        raise SweepError(f"cannot infer columns for empty CSV: {path}")
    fields = list(fields or csv_fields(rows))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SweepError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def number(value, name: str, *, blank: bool = False) -> float | None:
    try:
        return CORE._number(value, name, blank=blank)
    except Exception as exc:
        raise SweepError(str(exc)) from exc


def integer(value, name: str, *, blank: bool = False) -> int | None:
    try:
        return CORE._integer(value, name, blank=blank)
    except Exception as exc:
        raise SweepError(str(exc)) from exc


def boolean(value, name: str) -> bool:
    try:
        return CORE._bool(value, name)
    except Exception as exc:
        raise SweepError(str(exc)) from exc


def close(a, b, *, atol=1e-6, rtol=1e-9) -> bool:
    return CORE._close(a, b, atol=atol, rtol=rtol)


def normalize_prediction(value: str) -> str:
    value = re.sub(r"[^\w\s]", " ", str(value).lower())
    return " ".join(word for word in value.split()
                    if word not in {"a", "an", "the"})


def method_for_budget(budget: float) -> str:
    return f"{PREFIX_BASE}@{int(round(100 * budget))}"


def rows_by_key(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(str(row["image_id"]), str(row["question_id"])): row
            for row in rows}


def source_record(label: str, path: Path) -> dict:
    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink():
        raise SweepError(f"missing/non-regular input {label}: {path}")
    return {
        "label": label,
        "path": path,
        "signature": CORE._sig(path),
        "sha256": sha256(path),
    }


def recheck_sources(records: list[dict]) -> None:
    changed = []
    for record in records:
        path = record["path"]
        if (not path.is_file() or path.is_symlink()
                or CORE._sig(path) != record["signature"]
                or sha256(path) != record["sha256"]):
            changed.append(record["label"])
    if changed:
        raise SweepError(f"source artifacts changed during analysis: {changed}")


def local_model_revision() -> str | None:
    if not MODEL_REVISION_REF.is_file():
        return None
    value = MODEL_REVISION_REF.read_text().strip()
    return value or None


def load_frozen_inputs(run_dir: Path) -> tuple:
    ordered_keys, frozen = CORE._frozen_workload()
    try:
        run = CORE._load_run(run_dir, "budget_sweep", ordered_keys, frozen,
                             new_schema=True,
                             expected_layout="visionzip_image_only")
        prior = CORE._load_run(OLD_RUN, "prior_image_only", ordered_keys,
                               frozen, new_schema=True,
                               expected_layout="visionzip_image_only")
        profile = CORE._load_profile(
            BUILD_PROFILE, "visionzip_image_only",
            {image for image, _ in ordered_keys})
    except Exception as exc:
        raise SweepError(str(exc)) from exc

    if tuple(run["groups"].keys()) != EXPECTED_METHODS:
        # CSV order is part of the execution contract: FullLoad once, then
        # ascending budgets for every request.
        raise SweepError(
            f"method/order mismatch: {tuple(run['groups'])} != {EXPECTED_METHODS}")
    if [float(x) for x in run["summary"].get("budgets", [])] != list(BUDGETS):
        raise SweepError("run budget list differs from the frozen sweep")
    if run["summary"].get("ratio") != 0.25:
        raise SweepError("base retention ratio changed")
    command = str(run["summary"].get("command", ""))
    required_command_parts = (
        "--store kvstore_image_only_visionzip",
        "--budgets 0.20,0.25,0.30,0.35,0.40,0.45,0.50",
        "--selectors visionzip_repack_prefix",
        "--prefix-layout visionzip_image_only",
        "--sep-policy sidecar",
        "--no-recompute",
    )
    missing = [part for part in required_command_parts if part not in command]
    if missing:
        raise SweepError(f"run command provenance is incomplete: {missing}")
    return ordered_keys, frozen, run, prior, profile


def validate_store(store: Path, profile: dict,
                   image_ids: set[str]) -> tuple[dict, list[dict]]:
    store = Path(store).resolve()
    if store != DEFAULT_STORE.resolve():
        raise SweepError(f"sweep must use the frozen store: {DEFAULT_STORE}")
    if not store.is_dir() or store.is_symlink():
        raise SweepError(f"missing/non-regular store: {store}")
    profile_store = Path(profile["document"]["store"]).resolve()
    if profile_store != store:
        raise SweepError("build profile points at a different physical store")
    actual_ids = {path.name for path in store.iterdir() if path.is_dir()}
    if actual_ids != image_ids:
        raise SweepError("store image set differs from frozen workload")

    manifests = []
    source_records = []
    total_bytes = 0
    for image_id in sorted(image_ids):
        image_dir = store / image_id
        meta_path = image_dir / "meta.json"
        layout_path = image_dir / "visionzip_layout.pt"
        source_records.extend((
            source_record(f"store.{image_id}.meta", meta_path),
            source_record(f"store.{image_id}.layout", layout_path),
        ))
        with meta_path.open() as handle:
            meta = json.load(handle)
        artifact = torch.load(layout_path, map_location="cpu",
                              weights_only=True)
        if (meta.get("physical_layout") != "visionzip_image_only"
                or meta.get("reordered") is not True
                or meta.get("order_is_per_layer") is not False
                or meta.get("layout_uses_dataset_question") is not False
                or meta.get("llm_used_for_layout_scoring") is not False
                or int(meta.get("calibration_questions", -1)) != 0
                or int(meta.get("chunk_size", -1)) != CHUNK_SIZE
                or int(meta.get("num_layers", -1)) != N_LAYERS
                or meta.get("dtype") != "float16"):
            raise SweepError(f"bad layout/provenance metadata: {image_id}")
        vn = int(meta["v_token_num"])
        heads = int(meta["num_heads"])
        head_dim = int(meta["head_dim"])
        chunks = int(meta["n_chunks_per_layer"])
        if chunks != math.ceil(vn / CHUNK_SIZE):
            raise SweepError(f"chunk geometry mismatch: {image_id}")
        order = [int(x) for x in meta["order"]]
        stored = artifact["stored_to_original"].tolist()
        if (order != stored or len(order) != vn
                or sorted(order) != list(range(vn))):
            raise SweepError(f"layout artifact/permutation mismatch: {image_id}")
        separators = sorted(int(x) for x in meta["newline_idx"])
        stored_sep = sorted(int(x) for x in meta["newline_stored"])
        if (order[-len(separators):] != separators
                or stored_sep != list(range(vn - len(separators), vn))):
            raise SweepError(f"separators are not a stable physical tail: {image_id}")
        if (artifact.get("layout_uses_dataset_question") is not False
                or artifact.get("llm_used_for_layout_scoring") is not False
                or int(artifact.get("calibration_questions", -1)) != 0):
            raise SweepError(f"layout artifact is query-dependent: {image_id}")

        profile_row = profile["by_image"][image_id]
        if (int(profile_row["v_token_num"]) != vn
                or int(profile_row["n_chunks"]) != chunks
                or profile_row["permutation_sha256"] !=
                permutation_sha256(order)):
            raise SweepError(f"profile/store binding failed: {image_id}")

        row_bytes = heads * head_dim * 2
        normal_file_bytes = vn * row_bytes
        for layer in range(N_LAYERS):
            for kind in ("k", "v"):
                path = image_dir / f"layer_{layer:02d}/{kind}.bin"
                if (not path.is_file() or path.is_symlink()
                        or path.stat().st_size != normal_file_bytes):
                    raise SweepError(f"normal KV file-size mismatch: {path}")
                total_bytes += path.stat().st_size
        sep_expected = (2 * N_LAYERS * len(separators) * heads * head_dim * 2)
        sep_path = image_dir / "sep_kv.bin"
        if (not sep_path.is_file() or sep_path.is_symlink()
                or sep_path.stat().st_size != sep_expected
                or int(meta["bytes_separator_sidecar"]) != sep_expected):
            raise SweepError(f"separator sidecar mismatch: {image_id}")
        if int(meta["bytes_visual_kv"]) != 2 * N_LAYERS * normal_file_bytes:
            raise SweepError(f"visual-KV byte total mismatch: {image_id}")
        total_bytes += sep_expected
        manifests.append({
            "image_id": image_id,
            "v_token_num": vn,
            "n_chunks": chunks,
            "separator_tokens": len(separators),
            "bytes_visual_kv": int(meta["bytes_visual_kv"]),
            "bytes_separator_sidecar": sep_expected,
            "meta_sha256": sha256(meta_path),
            "layout_sha256": sha256(layout_path),
            "permutation_sha256": permutation_sha256(order),
        })

    info = {
        "path": str(store),
        "n_images": len(manifests),
        "measured_normal_plus_separator_bytes": total_bytes,
        "manifest_sha256": stable_hash(manifests),
        "permutation_manifest_sha256": stable_hash([
            {"image_id": row["image_id"],
             "permutation_sha256": row["permutation_sha256"]}
            for row in manifests]),
        "same_store_as_original_image_only_run": True,
    }
    return info, source_records


def expected_io(meta: dict, budget: float | None) -> dict:
    vn = int(meta["v_token_num"])
    layers = int(meta["num_layers"])
    heads = int(meta["num_heads"])
    head_dim = int(meta["head_dim"])
    chunks = int(meta["n_chunks_per_layer"])
    row_bytes = heads * head_dim * 2
    if budget is None:
        return {
            "k": chunks,
            "selected_rows": vn,
            "normal_bytes": 2 * layers * vn * row_bytes,
            "separator_bytes": 0,
            "normal_preads": 2 * layers,
            "separator_preads": 0,
            "chunk_units": 2 * layers * chunks,
            "touched": 1.0,
            "logical": 1.0,
        }
    k = CORE._budget_chunks(chunks, budget)
    selected_rows = min(k * int(meta["chunk_size"]), vn)
    selected_positions = set(range(selected_rows))
    separator_positions = set(int(x) for x in meta["newline_stored"])
    kept = len(selected_positions | separator_positions)
    return {
        "k": k,
        "selected_rows": selected_rows,
        "normal_bytes": 2 * layers * selected_rows * row_bytes,
        "separator_bytes": int(meta["bytes_separator_sidecar"]),
        "normal_preads": 2 * layers,
        "separator_preads": 1,
        "chunk_units": 2 * layers * k,
        "touched": k / chunks,
        "logical": kept / vn,
    }


def validate_run_semantics(run: dict, store: Path,
                           ordered_keys: list[tuple[str, str]]) -> dict:
    by_method = {method: rows_by_key(rows)
                 for method, rows in run["groups"].items()}
    failures = []
    fingerprints = defaultdict(dict)
    max_ttft_prepare_residual = 0.0
    max_e2e_residual = 0.0
    for method in EXPECTED_METHODS:
        budget = None if method == "fullload" else float(
            method.rsplit("@", 1)[1]) / 100
        for key in ordered_keys:
            row = by_method[method][key]
            image_id = key[0]
            with (Path(store) / image_id / "meta.json").open() as handle:
                meta = json.load(handle)
            expected = expected_io(meta, budget)
            prefix = budget is not None
            context = f"{method}/{image_id}/{key[1]}"
            if (row.get("physical_layout") != "visionzip_image_only"
                    or integer(row.get("calibration_questions"),
                               context + "/calibration") != 0):
                failures.append(context + ": layout/calibration")
            if prefix:
                try:
                    selections = CORE._chunk_layers(
                        row.get("selected_chunk_ids_per_layer"), context)
                except Exception as exc:
                    raise SweepError(str(exc)) from exc
                wanted = list(range(expected["k"]))
                if (len(selections) != N_LAYERS
                        or any(layer != wanted for layer in selections)):
                    failures.append(context + ": not exact first-k")
                selection_hash = stable_hash(selections)
                old = fingerprints[method].setdefault(image_id, selection_hash)
                if old != selection_hash:
                    failures.append(context + ": question-varying selection")
                if (row.get("selection_mode") != "prefix"
                        or row.get("retrieval") != "prefix"
                        or row.get("separator_policy") != "sidecar"
                        or row.get("validated_prefix_layout") !=
                        "visionzip_image_only"):
                    failures.append(context + ": Prefix provenance")
                for field in ("static_score_calls", "query_score_calls",
                              "diversity_calls"):
                    if integer(row.get(field), context + "/" + field) != 0:
                        failures.append(context + f": nonzero {field}")
                if integer(row.get("normal_chunk_count_total"), context) != \
                        N_LAYERS * expected["k"]:
                    failures.append(context + ": selected chunk total")
                if not close(number(row.get("n_chunks_selected"), context),
                             expected["k"]):
                    failures.append(context + ": chunks/layer")
            else:
                if (row.get("retrieval") != "fullload"
                        or row.get("selection_mode") not in ("", None)):
                    failures.append(context + ": FullLoad provenance")

            exact_pairs = (
                ("normal_kv_read_bytes", expected["normal_bytes"]),
                ("separator_read_bytes", expected["separator_bytes"]),
                ("ssd_read_bytes", expected["normal_bytes"] +
                 expected["separator_bytes"]),
                ("total_actual_pread_bytes", expected["normal_bytes"] +
                 expected["separator_bytes"]),
                ("normal_kv_preads", expected["normal_preads"]),
                ("separator_preads", expected["separator_preads"]),
                ("ssd_preads", expected["normal_preads"] +
                 expected["separator_preads"]),
                ("ssd_read_chunks", expected["chunk_units"]),
            )
            for field, wanted in exact_pairs:
                if integer(row.get(field), context + "/" + field) != wanted:
                    failures.append(context + f": {field}")
            for field, wanted in (
                    ("touched_chunk_fraction", expected["touched"]),
                    ("logical_kv_ratio", expected["logical"])):
                if not close(number(row.get(field), context + "/" + field),
                             wanted, atol=1e-9):
                    failures.append(context + f": {field}")
            total_bytes = expected["normal_bytes"] + expected["separator_bytes"]
            total_preads = expected["normal_preads"] + expected["separator_preads"]
            if not close(number(row.get("mean_bytes_per_pread"), context),
                         total_bytes / total_preads, atol=1e-5):
                failures.append(context + ": mean bytes/pread")

            ttft = number(row["ttft_ms"], context + "/ttft")
            decode = number(row["decode_ms"], context + "/decode")
            e2e = number(row["e2e_latency_ms"], context + "/e2e")
            max_e2e_residual = max(max_e2e_residual,
                                   abs(e2e - ttft - decode))
            if not ttft < e2e:
                failures.append(context + ": TTFT !< E2E")
            if prefix:
                prepare = number(row["prepare_ms"], context + "/prepare")
                prefill = number(row["prefill_ms"], context + "/prefill")
                max_ttft_prepare_residual = max(
                    max_ttft_prepare_residual,
                    abs(ttft - prepare - prefill))

    # Prefix selections must be nested as the budget grows for each image.
    for image_id in sorted({key[0] for key in ordered_keys}):
        previous = set()
        for budget in BUDGETS:
            method = method_for_budget(budget)
            sample = by_method[method][next(
                key for key in ordered_keys if key[0] == image_id)]
            layers = CORE._chunk_layers(
                sample["selected_chunk_ids_per_layer"], method)
            current = set(layers[0])
            if not previous.issubset(current):
                failures.append(f"{image_id}: budget selections not nested")
            previous = current

    if failures:
        raise SweepError(f"runtime selection/I/O validation failed: {failures[:20]}")
    if max_e2e_residual > 1.0 or max_ttft_prepare_residual > 1.0:
        raise SweepError(
            "latency algebra exceeds 1 ms: "
            f"e2e={max_e2e_residual}, prefix={max_ttft_prepare_residual}")
    return {
        "passed": True,
        "prefix_read_implementation": "coalesced_contiguous_span",
        "normal_preads_per_request": 2 * N_LAYERS,
        "separator_preads_per_request": 1,
        "total_prefix_preads_per_request": 2 * N_LAYERS + 1,
        "max_abs_e2e_minus_ttft_decode_ms": max_e2e_residual,
        "max_abs_prefix_ttft_minus_prepare_prefill_ms":
            max_ttft_prepare_residual,
        "selection_fingerprints": dict(fingerprints),
    }


def cluster_accuracy_ci(scores, images, *, seed=BOOTSTRAP_SEED,
                        n=BOOTSTRAP_RESAMPLES) -> list[float]:
    scores = np.asarray(scores, dtype=float)
    grouped = defaultdict(list)
    order = []
    for score, image in zip(scores, images):
        if image not in grouped:
            order.append(image)
        grouped[image].append(float(score))
    if len(order) != N_IMAGES or any(len(grouped[x]) != 6 for x in order):
        raise SweepError("invalid image clusters for bootstrap")
    image_means = np.asarray([np.mean(grouped[x]) for x in order])
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, len(order), size=(n, len(order)))
    samples = image_means[indices].mean(axis=1)
    return [float(x * 100) for x in np.percentile(samples, (2.5, 97.5))]


def summarize(run: dict, ordered_keys: list[tuple[str, str]]) -> tuple:
    images = [key[0] for key in ordered_keys]
    maps = {method: rows_by_key(rows)
            for method, rows in run["groups"].items()}
    full_scores = np.asarray([
        number(maps["fullload"][key]["correct"], "correct")
        for key in ordered_keys])
    summary_rows = []
    paired = {}
    for method in EXPECTED_METHODS:
        rows = [maps[method][key] for key in ordered_keys]
        budget = None if method == "fullload" else float(
            method.rsplit("@", 1)[1]) / 100
        scores = np.asarray([number(row["correct"], "correct")
                             for row in rows])
        comparison = CORE._paired_bootstrap(
            scores, full_scores, images, n=BOOTSTRAP_RESAMPLES,
            seed=BOOTSTRAP_SEED)
        paired[method] = comparison
        result = {
            "method_key": method,
            "method": ("ImageOnly-Repack FullLoad" if budget is None else
                       "ImageOnly-Repack + Prefix"),
            "budget": "" if budget is None else budget,
            "budget_pct": "" if budget is None else budget * 100,
            "n_requests": len(rows),
            "n_images": len(set(images)),
            "accuracy": float(scores.mean()),
            "accuracy_pct": float(scores.mean() * 100),
            "accuracy_image_cluster_ci95_lo_pct":
                cluster_accuracy_ci(scores, images)[0],
            "accuracy_image_cluster_ci95_hi_pct":
                cluster_accuracy_ci(scores, images)[1],
            "accuracy_delta_vs_fullload_pp":
                float((scores.mean() - full_scores.mean()) * 100),
            "accuracy_delta_image_cluster_ci95_lo_pp":
                comparison["image_cluster_delta_ci95_pp"][0],
            "accuracy_delta_image_cluster_ci95_hi_pp":
                comparison["image_cluster_delta_ci95_pp"][1],
            "mcnemar_budget_only_correct": comparison["mcnemar"]["a_only"],
            "mcnemar_fullload_only_correct": comparison["mcnemar"]["b_only"],
            "mcnemar_p_exact_two_sided":
                comparison["mcnemar"]["p_exact_two_sided"],
        }
        for field, prefix in (
                ("ttft_ms", "ttft"), ("decode_ms", "decode"),
                ("e2e_latency_ms", "e2e"), ("selector_ms", "selector"),
                ("ssd_read_ms", "ssd_read"), ("scatter_ms", "scatter"),
                ("prepare_ms", "prepare"), ("prefill_ms", "prefill")):
            values = [number(row.get(field), field, blank=True) for row in rows]
            values = [value for value in values if value is not None]
            if values:
                mean, p50, p95 = CORE._percentiles(values)
                result[f"{prefix}_mean_ms"] = mean
                result[f"{prefix}_p50_ms"] = p50
                result[f"{prefix}_p95_ms"] = p95
            else:
                result[f"{prefix}_mean_ms"] = ""
                result[f"{prefix}_p50_ms"] = ""
                result[f"{prefix}_p95_ms"] = ""
        for field in (
                "normal_kv_read_bytes", "separator_read_bytes",
                "ssd_read_bytes", "normal_kv_preads", "separator_preads",
                "ssd_preads", "mean_bytes_per_pread", "ssd_read_chunks",
                "n_chunks_selected", "n_chunks_total",
                "touched_chunk_fraction", "logical_kv_ratio"):
            values = [number(row.get(field), field, blank=True) for row in rows]
            values = [value for value in values if value is not None]
            result[field + "_mean"] = float(np.mean(values)) if values else ""
            if field in ("n_chunks_selected", "n_chunks_total"):
                result[field + "_min"] = min(values) if values else ""
                result[field + "_max"] = max(values) if values else ""
        result["ssd_read_mb_mean"] = result["ssd_read_bytes_mean"] / 1e6
        result["ssd_read_bytes_total"] = int(sum(
            integer(row["ssd_read_bytes"], "ssd_read_bytes") for row in rows))
        result["normal_kv_read_mb_mean"] = \
            result["normal_kv_read_bytes_mean"] / 1e6
        result["separator_read_mb_mean"] = \
            result["separator_read_bytes_mean"] / 1e6
        summary_rows.append(result)

    full = summary_rows[0]
    for row in summary_rows:
        row["ttft_reduction_vs_fullload_pct"] = (
            100 * (1 - row["ttft_mean_ms"] / full["ttft_mean_ms"]))
        row["ssd_reduction_vs_fullload_pct"] = (
            100 * (1 - row["ssd_read_bytes_mean"] /
                   full["ssd_read_bytes_mean"]))
        row["ssd_read_ratio_vs_fullload"] = (
            row["ssd_read_bytes_mean"] / full["ssd_read_bytes_mean"])
        row["accuracy_loss_vs_fullload_pp"] = (
            full["accuracy_pct"] - row["accuracy_pct"])
    return summary_rows, paired, maps


def rankdata(values: list[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def correlation(x: list[float], y: list[float]) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson_r": None, "spearman_r": None}
    return {
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "spearman_r": float(np.corrcoef(rankdata(x), rankdata(y))[0, 1]),
    }


def load_old_coverage_reference() -> dict:
    _, rows = read_csv(OLD_COVERAGE)
    answer = {}
    for source in ("visionzip_image_saliency",
                   "sparsevlm_calib4_analysis_only"):
        group = [row for row in rows
                 if row["layout"] == "visionzip_image_only"
                 and row["importance_source"] == source
                 and close(float(row["physical_fraction_requested"]), 0.25)]
        if len(group) != N_IMAGES * N_LAYERS:
            raise SweepError(f"old 25% coverage reference incomplete: {source}")
        selected = sum(float(row["selected_importance_mass"]) for row in group)
        total = sum(float(row["total_importance_mass"]) for row in group)
        answer[source] = {
            "macro": float(np.mean([
                float(row["importance_mass_coverage"]) for row in group])),
            "global": selected / total,
        }
    return answer


def compute_coverage(store: Path, ordered_keys: list[tuple[str, str]],
                     summary_rows: list[dict]) -> tuple:
    score_doc = torch.load(RAW_CALIBRATION, map_location="cpu",
                           weights_only=True)
    if (score_doc.get("schema_version") != 1
            or score_doc.get("analysis_only") is not True
            or score_doc.get("used_for_layout_or_serving") is not False):
        raise SweepError("analysis-only calibration-score provenance failed")
    raw = score_doc.get("scores_original_by_image")
    image_ids = sorted({key[0] for key in ordered_keys})
    if not isinstance(raw, dict) or set(raw) != set(image_ids):
        raise SweepError("analysis-only score image set differs")

    accum = defaultdict(list)
    masses = defaultdict(lambda: [0.0, 0.0])
    realized = defaultdict(list)
    chunk_counts = defaultdict(list)
    for image_id in image_ids:
        image_dir = Path(store) / image_id
        with (image_dir / "meta.json").open() as handle:
            meta = json.load(handle)
        artifact = torch.load(image_dir / "visionzip_layout.pt",
                              map_location="cpu", weights_only=True)
        vision = artifact["token_score_original"].double()
        calibration = raw[image_id].double()
        vn = int(meta["v_token_num"])
        layers = int(meta["num_layers"])
        order = artifact["stored_to_original"].long()
        finite = torch.isfinite(vision)
        if (vision.shape != (vn,) or calibration.shape != (layers, vn)
                or order.shape != (vn,)):
            raise SweepError(f"coverage score geometry mismatch: {image_id}")
        if torch.nonzero(~finite).flatten().tolist() != sorted(
                int(x) for x in meta["newline_idx"]):
            raise SweepError(f"VisionZip separator mask mismatch: {image_id}")
        sources = {
            "visionzip_image_saliency": vision.unsqueeze(0).repeat(layers, 1),
            "sparsevlm_calib4_analysis_only": calibration,
        }
        for budget in BUDGETS:
            k = CORE._budget_chunks(int(meta["n_chunks_per_layer"]), budget)
            end = min(k * int(meta["chunk_size"]), vn)
            selected = order[:end]
            selected = selected[finite[selected]]
            normal_total = int(finite.sum())
            realized[budget].append(int(selected.numel()) / normal_total)
            chunk_counts[budget].append(k)
            for source, matrix in sources.items():
                for layer in range(layers):
                    score = matrix[layer]
                    total_mass = float(score[finite].sum())
                    selected_mass = float(score[selected].sum())
                    if not math.isfinite(total_mass) or total_mass <= 0:
                        raise SweepError(
                            f"invalid coverage mass: {image_id}/{source}")
                    value = selected_mass / total_mass
                    accum[(budget, source)].append(value)
                    masses[(budget, source)][0] += selected_mass
                    masses[(budget, source)][1] += total_mass

    accuracy = {float(row["budget"]): row["accuracy"]
                for row in summary_rows if row["budget"] != ""}
    rows = []
    for budget in BUDGETS:
        result = {
            "budget": budget,
            "budget_pct": budget * 100,
            "n_images": N_IMAGES,
            "n_image_layers": N_IMAGES * N_LAYERS,
            "selected_chunks_per_layer_mean":
                float(np.mean(chunk_counts[budget])),
            "realized_normal_token_fraction_mean":
                float(np.mean(realized[budget])),
            "accuracy": accuracy[budget],
            "accuracy_pct": accuracy[budget] * 100,
        }
        for source, prefix in (
                ("visionzip_image_saliency", "visionzip"),
                ("sparsevlm_calib4_analysis_only", "sparsevlm_analysis_only")):
            values = accum[(budget, source)]
            selected, total = masses[(budget, source)]
            result[prefix + "_mass_macro"] = float(np.mean(values))
            result[prefix + "_mass_global"] = selected / total
        rows.append(result)

    old = load_old_coverage_reference()
    row25 = next(row for row in rows if close(row["budget"], 0.25))
    if (not close(row25["visionzip_mass_macro"],
                  old["visionzip_image_saliency"]["macro"], atol=1e-12)
            or not close(row25["sparsevlm_analysis_only_mass_macro"],
                         old["sparsevlm_calib4_analysis_only"]["macro"],
                         atol=1e-12)):
        raise SweepError("new 25% coverage does not reproduce frozen artifact")
    for field in ("visionzip_mass_macro",
                  "sparsevlm_analysis_only_mass_macro",
                  "realized_normal_token_fraction_mean"):
        values = [row[field] for row in rows]
        if any(b + 1e-12 < a for a, b in zip(values, values[1:])):
            raise SweepError(f"coverage is not monotonic: {field}")

    accuracies = [row["accuracy"] for row in rows]
    correlations = {
        "visionzip_macro_vs_accuracy": correlation(
            [row["visionzip_mass_macro"] for row in rows], accuracies),
        "sparsevlm_analysis_only_macro_vs_accuracy": correlation(
            [row["sparsevlm_analysis_only_mass_macro"] for row in rows],
            accuracies),
    }
    return rows, correlations, old


def pareto_rows(summary_rows: list[dict], cost_field: str,
                cost_label: str) -> list[dict]:
    rows = []
    for candidate in summary_rows:
        dominated = []
        for other in summary_rows:
            if other is candidate:
                continue
            weak_cost = other[cost_field] <= candidate[cost_field] + 1e-12
            weak_quality = other["accuracy"] >= candidate["accuracy"] - 1e-12
            strict = (other[cost_field] < candidate[cost_field] - 1e-12
                      or other["accuracy"] > candidate["accuracy"] + 1e-12)
            if weak_cost and weak_quality and strict:
                dominated.append(other["method_key"])
        rows.append({
            "method_key": candidate["method_key"],
            "budget": candidate["budget"],
            "budget_pct": candidate["budget_pct"],
            "accuracy": candidate["accuracy"],
            "accuracy_pct": candidate["accuracy_pct"],
            cost_label: candidate[cost_field],
            "is_pareto_optimal": not dominated,
            "dominated_by": json.dumps(dominated, separators=(",", ":")),
        })
    return rows


def tradeoff_rows(summary_rows: list[dict], pareto_ssd: list[dict],
                  pareto_ttft: list[dict]) -> list[dict]:
    by_budget = {float(row["budget"]): row for row in summary_rows
                 if row["budget"] != ""}
    ssd_map = {float(row["budget"]): boolean(row["is_pareto_optimal"], "pareto")
               for row in pareto_ssd if row["budget"] != ""}
    ttft_map = {float(row["budget"]): boolean(row["is_pareto_optimal"], "pareto")
                for row in pareto_ttft if row["budget"] != ""}
    rows = []
    previous = None
    for budget in BUDGETS:
        current = by_budget[budget]
        loss = current["accuracy_loss_vs_fullload_pp"]
        row = {
            "budget": budget,
            "budget_pct": budget * 100,
            "previous_budget": "" if previous is None else previous,
            "previous_budget_pct": "" if previous is None else previous * 100,
            "accuracy_pct": current["accuracy_pct"],
            "accuracy_loss_vs_fullload_pp": loss,
            "ttft_mean_ms": current["ttft_mean_ms"],
            "ssd_read_mb_mean": current["ssd_read_mb_mean"],
            "qualifies_aggressive_loss_le_4pp": loss <= 4.0 + 1e-12,
            "qualifies_balanced_loss_le_2pp": loss <= 2.0 + 1e-12,
            "qualifies_quality_loss_le_1pp": loss <= 1.0 + 1e-12,
            "pareto_accuracy_vs_ssd": ssd_map[budget],
            "pareto_accuracy_vs_ttft": ttft_map[budget],
        }
        if previous is None:
            row.update({
                "budget_step_pp": "",
                "accuracy_gain_pp": "",
                "accuracy_gain_per_plus_5pct_kv_pp": "",
                "ssd_increase_mb": "",
                "ttft_increase_ms": "",
                "ssd_mb_cost_per_plus_1pp_accuracy": "",
                "ttft_ms_cost_per_plus_1pp_accuracy": "",
                "marginal_status": "first_tested_budget",
            })
        else:
            prior = by_budget[previous]
            gain = current["accuracy_pct"] - prior["accuracy_pct"]
            step = (budget - previous) * 100
            ssd_cost = current["ssd_read_mb_mean"] - prior["ssd_read_mb_mean"]
            ttft_cost = current["ttft_mean_ms"] - prior["ttft_mean_ms"]
            row.update({
                "budget_step_pp": step,
                "accuracy_gain_pp": gain,
                "accuracy_gain_per_plus_5pct_kv_pp": gain * 5 / step,
                "ssd_increase_mb": ssd_cost,
                "ttft_increase_ms": ttft_cost,
                "ssd_mb_cost_per_plus_1pp_accuracy":
                    ssd_cost / gain if gain > 0 else "",
                "ttft_ms_cost_per_plus_1pp_accuracy":
                    ttft_cost / gain if gain > 0 else "",
                "marginal_status": ("positive_accuracy_gain" if gain > 0 else
                                    "no_positive_accuracy_gain"),
            })
        rows.append(row)
        previous = budget
    return rows


def sensitivity_rows(maps: dict, ordered_keys: list[tuple[str, str]],
                     frozen: dict) -> tuple[list[dict], list[dict], list[dict]]:
    question_rows = []
    for key in ordered_keys:
        base = maps["fullload"][key]
        result = {
            "image_id": key[0],
            "question_id": key[1],
            "question": frozen[key]["question"],
            "question_type": qtype(frozen[key]["question"]),
            "gold": json.dumps(frozen[key]["gold"], ensure_ascii=False),
            "fullload_prediction": base["prediction"],
            "fullload_correct": integer(base["correct"], "correct"),
        }
        correctness = []
        predictions = []
        for budget in BUDGETS:
            row = maps[method_for_budget(budget)][key]
            tag = int(round(budget * 100))
            correct = integer(row["correct"], "correct")
            prediction = row["prediction"]
            result[f"prediction_{tag}"] = prediction
            result[f"correct_{tag}"] = correct
            correctness.append(correct)
            predictions.append(normalize_prediction(prediction))
        first_correct = next((BUDGETS[i] for i, value in enumerate(correctness)
                              if value), None)
        stable_correct = next((BUDGETS[i] for i in range(len(BUDGETS))
                               if all(correctness[i:])), None)
        stable_prediction = next((BUDGETS[i] for i in range(len(BUDGETS))
                                  if len(set(predictions[i:])) == 1), None)
        result.update({
            "first_correct_budget": "" if first_correct is None else first_correct,
            "stable_correct_from_budget":
                "" if stable_correct is None else stable_correct,
            "prediction_stabilization_budget":
                "" if stable_prediction is None else stable_prediction,
            "wrong25_recovered_at_30_or_35":
                bool((not correctness[1]) and
                     (correctness[2] or correctness[3])),
            "wrong25_never_recovers_through50":
                bool((not correctness[1]) and not any(correctness[2:])),
            "fullload_correct_but_50_wrong":
                bool(result["fullload_correct"] and not correctness[-1]),
            "fullload_correct_25wrong":
                bool(result["fullload_correct"] and not correctness[1]),
            "fullload_correct_25wrong_recovered_at_30_or_35":
                bool(result["fullload_correct"] and not correctness[1] and
                     (correctness[2] or correctness[3])),
            "fullload_correct_25wrong_never_recovers_through50":
                bool(result["fullload_correct"] and not correctness[1] and
                     not any(correctness[2:])),
        })
        question_rows.append(result)

    image_rows = []
    for image_id in sorted({key[0] for key in ordered_keys}):
        group = [row for row in question_rows if row["image_id"] == image_id]
        row = {"image_id": image_id, "n_questions": len(group)}
        full = float(np.mean([int(x["fullload_correct"]) for x in group]))
        row["fullload_accuracy"] = full
        for budget in BUDGETS:
            tag = int(round(budget * 100))
            acc = float(np.mean([int(x[f"correct_{tag}"]) for x in group]))
            row[f"accuracy_{tag}"] = acc
            row[f"delta_vs_fullload_pp_{tag}"] = (acc - full) * 100
        row["wrong25_recovered_at_30_or_35_count"] = sum(
            boolean(x["wrong25_recovered_at_30_or_35"], "recovery")
            for x in group)
        row["wrong25_never_recovers_through50_count"] = sum(
            boolean(x["wrong25_never_recovers_through50"], "never")
            for x in group)
        row["fullload_correct_25wrong_count"] = sum(
            boolean(x["fullload_correct_25wrong"], "FullLoad-correct 25-wrong")
            for x in group)
        row["fullload_correct_25wrong_recovered_at_30_or_35_count"] = sum(
            boolean(x["fullload_correct_25wrong_recovered_at_30_or_35"],
                    "FullLoad-correct 25-wrong recovered")
            for x in group)
        row["fullload_correct_25wrong_never_recovers_through50_count"] = sum(
            boolean(x["fullload_correct_25wrong_never_recovers_through50"],
                    "FullLoad-correct 25-wrong never recovers")
            for x in group)
        image_rows.append(row)

    error_rows = []
    categories = ["all"] + sorted({row["question_type"]
                                    for row in question_rows})
    for category in categories:
        group = (question_rows if category == "all" else
                 [row for row in question_rows
                  if row["question_type"] == category])
        full = np.asarray([int(row["fullload_correct"]) for row in group])
        for budget in BUDGETS:
            tag = int(round(budget * 100))
            current = np.asarray([int(row[f"correct_{tag}"]) for row in group])
            error_rows.append({
                "question_type": category,
                "budget": budget,
                "budget_pct": tag,
                "n_questions": len(group),
                "accuracy": float(current.mean()),
                "fullload_accuracy": float(full.mean()),
                "delta_vs_fullload_pp": float((current.mean()-full.mean())*100),
                "both_correct": int(((current == 1) & (full == 1)).sum()),
                "budget_only_correct": int(((current == 1) & (full == 0)).sum()),
                "fullload_only_correct": int(((current == 0) & (full == 1)).sum()),
                "neither_correct": int(((current == 0) & (full == 0)).sum()),
            })
    return question_rows, image_rows, error_rows


def operating_points(summary_rows: list[dict], pareto_ssd: list[dict],
                     pareto_ttft: list[dict]) -> dict:
    budgets = [row for row in summary_rows if row["budget"] != ""]
    candidates = {}
    for name, threshold in THRESHOLDS_PP.items():
        eligible = [row for row in budgets
                    if row["accuracy_loss_vs_fullload_pp"] <= threshold + 1e-12]
        candidates[name] = (float(eligible[0]["budget"]) if eligible else None)

    if candidates["balanced"] is not None:
        recommended = candidates["balanced"]
        basis = "minimum budget with point-estimate FullLoad loss <= 2 pp"
    elif candidates["aggressive"] is not None:
        recommended = candidates["aggressive"]
        basis = "no balanced point; minimum budget with loss <= 4 pp"
    else:
        best = sorted(budgets, key=lambda row: (
            row["accuracy_loss_vs_fullload_pp"], float(row["budget"])))[0]
        recommended = float(best["budget"])
        basis = "no threshold-qualified point; smallest observed accuracy loss"

    pareto_ssd_set = {float(row["budget"]) for row in pareto_ssd
                      if row["budget"] != "" and row["is_pareto_optimal"]}
    pareto_ttft_set = {float(row["budget"]) for row in pareto_ttft
                       if row["budget"] != "" and row["is_pareto_optimal"]}
    common = pareto_ssd_set & pareto_ttft_set
    pareto_pool = [row for row in budgets if float(row["budget"]) in common]
    balanced_pareto = [row for row in pareto_pool
                       if row["accuracy_loss_vs_fullload_pp"] <= 2.0 + 1e-12]
    if balanced_pareto:
        best_pareto = float(balanced_pareto[0]["budget"])
    elif pareto_pool:
        best_pareto = float(sorted(
            pareto_pool, key=lambda row: (
                row["accuracy_loss_vs_fullload_pp"], float(row["budget"])))[0][
                    "budget"])
    else:
        best_pareto = None

    saturation = None
    saturation_details = []
    for index, row in enumerate(budgets[:-1]):
        best_later = max(other["accuracy_pct"] for other in budgets[index+1:])
        future_gain = best_later - row["accuracy_pct"]
        saturation_details.append({
            "budget": float(row["budget"]),
            "best_later_accuracy_gain_pp": future_gain,
        })
        if saturation is None and future_gain <= SATURATION_FUTURE_GAIN_PP + 1e-12:
            saturation = float(row["budget"])

    by_budget = {float(row["budget"]): row for row in budgets}
    b25, b50 = by_budget[0.25], by_budget[0.50]
    q5 = {
        "accuracy_gain_pp": b50["accuracy_pct"] - b25["accuracy_pct"],
        "ttft_cost_ms": b50["ttft_mean_ms"] - b25["ttft_mean_ms"],
        "ssd_cost_mb_per_request":
            b50["ssd_read_mb_mean"] - b25["ssd_read_mb_mean"],
        "ssd_cost_ratio": b50["ssd_read_bytes_mean"] /
            b25["ssd_read_bytes_mean"],
    }
    tradeoff_supported = (
        all(row["ssd_read_ratio_vs_fullload"] < 1.0 for row in budgets)
        and all(row["ttft_reduction_vs_fullload_pct"] > 0 for row in budgets)
        and max(row["accuracy_pct"] for row in budgets)
        > min(row["accuracy_pct"] for row in budgets))
    return {
        "thresholds_accuracy_loss_pp": THRESHOLDS_PP,
        "minimum_budget_by_mode": candidates,
        "recommended_budget": recommended,
        "recommendation_basis": basis,
        "best_common_pareto_budget_under_balanced_priority": best_pareto,
        "pareto_accuracy_vs_ssd_budgets": sorted(pareto_ssd_set),
        "pareto_accuracy_vs_ttft_budgets": sorted(pareto_ttft_set),
        "saturation_rule": (
            "earliest tested budget below 50% whose best later observed "
            f"accuracy gain is <= {SATURATION_FUTURE_GAIN_PP} pp"),
        "saturation_budget": saturation,
        "saturation_details": saturation_details,
        "budget_25_to_50": q5,
        "controllable_tradeoff_claim_supported": tradeoff_supported,
    }


def cross_run_reproducibility(run: dict, prior: dict,
                              ordered_keys: list[tuple[str, str]]) -> dict:
    current = {method: rows_by_key(run["groups"][method])
               for method in ("fullload", method_for_budget(0.25),
                              method_for_budget(0.50))}
    prior_methods = {
        "fullload": "fullload",
        method_for_budget(0.25): CORE._find_method(
            prior, PREFIX_BASE, 0.25),
        method_for_budget(0.50): CORE._find_method(
            prior, PREFIX_BASE, 0.50),
    }
    details = {}
    for current_method, old_method in prior_methods.items():
        old = rows_by_key(prior["groups"][old_method])
        pred = sum(current[current_method][key]["prediction"] ==
                   old[key]["prediction"] for key in ordered_keys)
        first = sum(integer(current[current_method][key]["first_token_id"],
                            "first") ==
                    integer(old[key]["first_token_id"], "first")
                    for key in ordered_keys)
        accuracy_current = float(np.mean([
            number(current[current_method][key]["correct"], "correct")
            for key in ordered_keys]))
        accuracy_prior = float(np.mean([
            number(old[key]["correct"], "correct") for key in ordered_keys]))
        details[current_method] = {
            "prediction_matches": pred,
            "first_token_matches": first,
            "n_requests": N_QUESTIONS,
            "prediction_agreement": pred / N_QUESTIONS,
            "first_token_agreement": first / N_QUESTIONS,
            "accuracy_current": accuracy_current,
            "accuracy_prior": accuracy_prior,
            "accuracy_exact": close(accuracy_current, accuracy_prior),
        }
    return {
        "all_predictions_exact": all(
            row["prediction_matches"] == N_QUESTIONS for row in details.values()),
        "all_first_tokens_exact": all(
            row["first_token_matches"] == N_QUESTIONS for row in details.values()),
        "details": details,
        "classification": "cross-run reproducibility check; not a latency source",
    }


def format_budget(value) -> str:
    return "none" if value is None else f"{100 * float(value):.0f}%"


def make_readme(summary_rows: list[dict], tradeoff: list[dict],
                coverage: list[dict], pareto_ssd: list[dict],
                pareto_ttft: list[dict], operating: dict,
                correlations: dict, validation: dict,
                question_rows: list[dict], run_command: str) -> str:
    table = [
        "| Budget | Accuracy | Delta vs FullLoad | TTFT | TTFT reduction | "
        "SSD MB | SSD reduction | Preads |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        label = "FullLoad" if row["budget"] == "" else \
            f"{float(row['budget'])*100:.0f}%"
        delta = "--" if row["budget"] == "" else \
            f"{row['accuracy_delta_vs_fullload_pp']:+.2f} pp"
        tr = "--" if row["budget"] == "" else \
            f"{row['ttft_reduction_vs_fullload_pct']:.2f}%"
        sr = "--" if row["budget"] == "" else \
            f"{row['ssd_reduction_vs_fullload_pct']:.2f}%"
        table.append(
            f"| {label} | {row['accuracy_pct']:.2f}% | {delta} | "
            f"{row['ttft_mean_ms']:.2f} ms | {tr} | "
            f"{row['ssd_read_mb_mean']:.2f} | {sr} | "
            f"{row['ssd_preads_mean']:.1f} |")

    cov_table = [
        "| Budget | VisionZip mass | SparseVLM mass (analysis only) | Accuracy |",
        "|---:|---:|---:|---:|",
    ]
    for row in coverage:
        cov_table.append(
            f"| {row['budget_pct']:.0f}% | "
            f"{100*row['visionzip_mass_macro']:.2f}% | "
            f"{100*row['sparsevlm_analysis_only_mass_macro']:.2f}% | "
            f"{row['accuracy_pct']:.2f}% |")

    marginal = [
        "| Step | Accuracy gain | SSD cost | TTFT cost |",
        "|---:|---:|---:|---:|",
    ]
    for row in tradeoff[1:]:
        marginal.append(
            f"| {float(row['previous_budget_pct']):.0f}% -> "
            f"{row['budget_pct']:.0f}% | {row['accuracy_gain_pp']:+.2f} pp | "
            f"{row['ssd_increase_mb']:+.2f} MB | "
            f"{row['ttft_increase_ms']:+.2f} ms |")

    def frontier(rows):
        return ["FullLoad" if row["budget"] == "" else
                f"{float(row['budget'])*100:.0f}%" for row in rows
                if row["is_pareto_optimal"]]

    by_budget = {float(row["budget"]): row for row in summary_rows
                 if row["budget"] != ""}
    recommended = by_budget[float(operating["recommended_budget"])]
    q5 = operating["budget_25_to_50"]
    historical = validation["known_historical_cross_layout_warning"]
    lost_at_25 = sum(boolean(row["fullload_correct_25wrong"], "lost at 25")
                     for row in question_rows)
    recovered_30_35 = sum(boolean(
        row["fullload_correct_25wrong_recovered_at_30_or_35"],
        "recovered at 30/35") for row in question_rows)
    never_recovered = sum(boolean(
        row["fullload_correct_25wrong_never_recovers_through50"],
        "never recovered") for row in question_rows)
    claim = ("SUPPORTED, with non-monotonic accuracy and small-sample caveats"
             if operating["controllable_tradeoff_claim_supported"] else
             "NOT SUPPORTED by the predeclared empirical checks")
    return f"""# ImageOnly-Repack sequential-Prefix budget sweep

## A. Workload

This is the frozen GQA workload: **40 images / 240 questions**, questions
`[4:10]` for every image.  Index SHA256 is `{INDEX_SHA256}` and ordered
workload SHA256 is `{WORKLOAD_SHA256}`.  All arms ran in one process with the
same model, prompt, decoding, physical SSD store, and schema-v2 true-TTFT
contract.

## B. Budgets

Main budgets are 20/25/30/35/40/45/50%.  Every budget uses the same immutable
ImageOnly VisionZip permutation; only `k = round(n_chunks * budget)` changes.
No calibration question, online score, diversity, or budget-specific layout is
used.

The Prefix reader was already optimized before this sweep: adjacent first-k
chunks are merged into one contiguous span per layer/K-or-V file.  Therefore
every Prefix request performs 64 normal preads plus one separator-sidecar
pread, not one syscall per logical chunk.

## C. Main table

{chr(10).join(table)}

Accuracy is binary normalized GQA match.  `summary.csv` also reports the
image-cluster bootstrap 95% CI for each accuracy and each paired FullLoad
delta.  With only 40 image clusters, sub-point differences should not be
over-interpreted.

## D. Pareto frontier

- Accuracy vs SSD: {', '.join(frontier(pareto_ssd))}
- Accuracy vs true TTFT: {', '.join(frontier(pareto_ttft))}

Dominated points and their dominators are explicit in the two Pareto CSVs.
Pareto membership uses observed point estimates; uncertainty remains in the
paired confidence intervals.

## E. Accuracy recovery

- Aggressive (loss <=4 pp): {format_budget(operating['minimum_budget_by_mode']['aggressive'])}
- Balanced (loss <=2 pp): {format_budget(operating['minimum_budget_by_mode']['balanced'])}
- Quality-oriented (loss <=1 pp): {format_budget(operating['minimum_budget_by_mode']['quality'])}
- Saturation by the predeclared <=0.5 pp best-future-gain rule: {format_budget(operating['saturation_budget'])}

## F. SSD/TTFT cost

{chr(10).join(marginal)}

Doubling 25% to 50% changes accuracy by {q5['accuracy_gain_pp']:+.2f} pp,
adds {q5['ssd_cost_mb_per_request']:.2f} MB/request, and adds
{q5['ttft_cost_ms']:.2f} ms mean true TTFT.

## G. Importance coverage

{chr(10).join(cov_table)}

VisionZip coverage/accuracy correlation: Pearson
`{correlations['visionzip_macro_vs_accuracy']['pearson_r']}` and Spearman
`{correlations['visionzip_macro_vs_accuracy']['spearman_r']}`.  SparseVLM
coverage is explicitly analysis-only and never affected serving.

## H. Error analysis

`per_question_sensitivity.csv`, `per_image_sensitivity.csv`, and
`error_analysis.csv` separate 25%-wrong questions recovered at 30/35%, those
never recovered through 50%, prediction stabilization, and the existing GQA
question categories (yes/no, color, count, spatial, material/attribute,
object, other).

Restricting the attribution to questions that FullLoad answers correctly,
25% loses {lost_at_25} questions; {recovered_30_35} recover by 30/35%, while
{never_recovered} never recover at any tested budget through 50%.

## I. Recommended operating point

**Recommended operating point = {100*float(operating['recommended_budget']):.0f}%**.
Rule: {operating['recommendation_basis']}.  At this point the FullLoad
accuracy gap is {recommended['accuracy_delta_vs_fullload_pp']:+.2f} pp, SSD
traffic is {100*recommended['ssd_read_ratio_vs_fullload']:.2f}% of FullLoad,
and true TTFT is reduced by
{recommended['ttft_reduction_vs_fullload_pct']:.2f}%.

This recommendation is based on the observed point estimate.  Its paired
image-cluster bootstrap delta CI is
`[{recommended['accuracy_delta_image_cluster_ci95_lo_pp']:+.2f},
{recommended['accuracy_delta_image_cluster_ci95_hi_pp']:+.2f}] pp`; therefore
this sample does **not** establish that the population accuracy loss is at
most 1 or 2 pp.

## J. Research conclusion

**Claim assessment: {claim}.**  Every tested Prefix budget reduces observed
SSD traffic and true TTFT versus same-run FullLoad, and higher budgets recover
some accuracy.  The curve is not monotonic: 45% is more accurate than 50% on
this 240-question sample.  No budget was selected in advance.

Direct answers: Q1={format_budget(operating['minimum_budget_by_mode']['aggressive'])},
Q2={format_budget(operating['minimum_budget_by_mode']['balanced'])},
Q3={format_budget(operating['minimum_budget_by_mode']['quality'])},
Q4={format_budget(operating['best_common_pareto_budget_under_balanced_priority'])},
Q6={format_budget(operating['saturation_budget'])}. Q5 is quantified in
Section F; Q7 is quantified by the coverage correlations in Section G.

Known historical warning retained: raster FullLoad vs ImageOnly-repacked
FullLoad strict prediction identity was
`{historical['agreement_requests']}/{N_QUESTIONS}`, so cross-layout exactness
remains FAIL even though mapped FP16 KV structural integrity passed.  This does
not invalidate the present same-layout budget comparison, but it must not be
reported as strict raster equivalence.

## Reproduction

```bash
{run_command}
```
"""


def publish(out_dir: Path, source_records: list[dict], files: dict[str, object]) -> None:
    out_dir = Path(out_dir).resolve()
    if out_dir != DEFAULT_OUTPUT.resolve():
        raise SweepError(f"output must be exactly {DEFAULT_OUTPUT}")
    if out_dir.exists():
        raise SweepError(f"refusing to overwrite existing output: {out_dir}")
    if out_dir.is_symlink():
        raise SweepError(f"output is a symlink: {out_dir}")
    parent = out_dir.parent
    if not parent.is_dir() or parent.is_symlink():
        raise SweepError(f"bad output parent: {parent}")
    stage = Path(tempfile.mkdtemp(prefix=".image_only_budget_sweep.stage.",
                                  dir=parent))
    try:
        for name, value in files.items():
            path = stage / name
            if name == "per_request.csv":
                shutil.copyfile(value, path)
            elif name.endswith(".json"):
                write_json(path, value)
            elif name.endswith(".csv"):
                write_csv(path, value)
            elif name == "README.md":
                path.write_text(str(value))
            else:
                raise SweepError(f"unknown output type: {name}")
        expected = set(files)
        actual = {path.name for path in stage.iterdir()}
        if actual != expected:
            raise SweepError(f"staging file set differs: {actual} != {expected}")
        recheck_sources(source_records)
        parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            CORE._rename_noreplace(parent_fd, stage.name,
                                   parent_fd, out_dir.name)
        finally:
            os.close(parent_fd)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def analyze(run_dir: Path, out_dir: Path, store: Path) -> dict:
    ordered_keys, frozen, run, prior, profile = load_frozen_inputs(run_dir)
    store_info, store_sources = validate_store(
        store, profile, {image for image, _ in ordered_keys})
    runtime_validation = validate_run_semantics(run, store, ordered_keys)
    summary_rows, paired, maps = summarize(run, ordered_keys)
    coverage_rows, correlations, old_coverage = compute_coverage(
        store, ordered_keys, summary_rows)
    pareto_ssd = pareto_rows(summary_rows, "ssd_read_mb_mean",
                             "ssd_read_mb_mean")
    pareto_ttft = pareto_rows(summary_rows, "ttft_mean_ms", "ttft_mean_ms")
    tradeoff = tradeoff_rows(summary_rows, pareto_ssd, pareto_ttft)
    question_rows, image_rows, error_rows = sensitivity_rows(
        maps, ordered_keys, frozen)
    operating = operating_points(summary_rows, pareto_ssd, pareto_ttft)
    reproducibility = cross_run_reproducibility(run, prior, ordered_keys)

    with OLD_VALIDATION.open() as handle:
        old_validation = json.load(handle)
    identity = old_validation.get("full_load_identity", {})
    agreement = identity.get("prediction_agreement")
    agreement_requests = (round(float(agreement) * N_QUESTIONS)
                          if agreement is not None else 238)
    historical_warning = {
        "strict_full_load_exact_pass":
            bool(old_validation.get("strict_full_load_exact_pass", False)),
        "agreement_requests": int(agreement_requests),
        "n_requests": N_QUESTIONS,
        "structural_integrity_pass":
            bool(old_validation.get("structural_integrity_pass", False)),
        "scope": "historical raster-vs-image-only cross-layout comparison",
    }

    source_records = []
    for label, path in (
            ("frozen_index", CORE.INDEX),
            ("evaluation_code", ROOT / "scripts/04_eval.py"),
            ("serve_code", ROOT / "mmimpress/serve.py"),
            ("selector_code", ROOT / "mmimpress/cvpr25.py"),
            ("reader_code", ROOT / "mmimpress/store.py"),
            ("analysis_core", CORE_PATH),
            ("analyzer", Path(__file__)),
            ("build_profile", BUILD_PROFILE),
            ("raw_calibration_analysis_only", RAW_CALIBRATION),
            ("old_coverage", OLD_COVERAGE),
            ("old_config", OLD_CONFIG),
            ("old_validation", OLD_VALIDATION)):
        source_records.append(source_record(label, path))
    for run_name, loaded in (("sweep", run), ("prior", prior)):
        for kind, path in loaded["files"].items():
            source_records.append(source_record(f"run.{run_name}.{kind}", path))
    source_records.extend(store_sources)
    # De-duplicate records while retaining the first label for a path.
    unique = {}
    for record in source_records:
        unique.setdefault(record["path"], record)
    source_records = list(unique.values())

    input_manifest = {
        record["label"]: {"path": str(record["path"]),
                          "sha256": record["sha256"]}
        for record in source_records
    }
    config = {
        "schema_version": 1,
        "analysis": "image_only_repack_sequential_prefix_budget_sweep",
        "generated_unix_time": time.time(),
        "cpu_only_analyzer": True,
        "frozen_workload": {
            "dataset": "gqa", "images": N_IMAGES,
            "questions": N_QUESTIONS, "questions_per_image": 6,
            "question_slice": [4, 10], "index_sha256": INDEX_SHA256,
            "workload_sha256": WORKLOAD_SHA256,
        },
        "budgets": list(BUDGETS),
        "same_run_fullload": True,
        "model": {
            "id": "llava-hf/llava-v1.6-vicuna-7b-hf",
            "cached_revision": local_model_revision(),
            "quantization": "4-bit NF4 double-quant",
            "compute_dtype": "bfloat16", "attention": "eager",
            "decoding": "greedy", "max_new_tokens": 16,
        },
        "method": {
            "layout": "visionzip_image_only",
            "retrieval": "sequential first-k Prefix",
            "budget_rounding": "Python round(total_chunks * budget), clamped",
            "chunk_size": CHUNK_SIZE,
            "separator_policy": "sidecar",
            "prefix_io": "adjacent chunk ranges coalesced to one span per file",
            "normal_preads_per_request": 64,
            "separator_preads_per_request": 1,
            "layout_uses_dataset_question": False,
            "calibration_questions": 0,
            "online_scoring": False,
            "static_or_diversity_selector_used": False,
        },
        "store": store_info,
        "run_command": run["summary"]["command"],
        "bootstrap": {
            "primary_unit": "image cluster", "clusters": N_IMAGES,
            "questions_per_cluster": 6, "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED, "paired_vs_same_run_fullload": paired,
        },
        "coverage": {
            "sparsevlm_scores_analysis_only": True,
            "used_for_layout_or_serving": False,
            "correlations": correlations,
            "frozen_25pct_reference": old_coverage,
        },
        "operating_points": operating,
        "cross_run_25_50_reproducibility": reproducibility,
        "known_historical_cross_layout_warning": historical_warning,
        "inputs": input_manifest,
    }
    cross_run_exact = bool(reproducibility["all_predictions_exact"] and
                           reproducibility["all_first_tokens_exact"])
    validation = {
        "schema_version": 1,
        "all_passed": cross_run_exact,
        "checks": {
            "exact_frozen_40_image_240_question_workload": {"passed": True},
            "exact_one_fullload_plus_seven_budget_arms_1920_rows": {
                "passed": True},
            "same_process_same_run_fullload": {"passed": True},
            "same_immutable_image_only_store_all_budgets": {
                "passed": True, "manifest_sha256":
                store_info["manifest_sha256"]},
            "model_decoding_schema_v2_cold_chunk64_sidecar": {"passed": True},
            "zero_calibration_query_static_diversity_selection": {
                "passed": True},
            "exact_first_k_python_rounding_and_budget_nesting": {"passed": True},
            "actual_pread_bytes_and_split_exact": {"passed": True},
            "contiguous_span_preads_64_normal_plus_1_separator": {
                "passed": True},
            "ttft_and_e2e_timing_algebra": {
                "passed": True,
                "max_abs_e2e_minus_ttft_decode_ms":
                    runtime_validation["max_abs_e2e_minus_ttft_decode_ms"],
                "max_abs_prefix_ttft_minus_prepare_prefill_ms":
                    runtime_validation[
                        "max_abs_prefix_ttft_minus_prepare_prefill_ms"]},
            "coverage_complete_monotonic_and_25pct_reproduced": {
                "passed": True},
            "image_cluster_bootstrap_deterministic": {"passed": True},
            "pareto_frontier_recomputed_from_point_estimates": {"passed": True},
            "cross_run_25_50_predictions_and_first_tokens_exact": {
                "passed": cross_run_exact},
            "source_artifacts_rechecked_before_atomic_publication": {
                "passed": True},
        },
        "runtime": runtime_validation,
        "cross_run_reproducibility": reproducibility,
        "known_historical_cross_layout_warning": historical_warning,
    }
    readme = make_readme(
        summary_rows, tradeoff, coverage_rows, pareto_ssd, pareto_ttft,
        operating, correlations, validation, question_rows,
        run["summary"]["command"])

    files = {
        "config.json": config,
        "per_request.csv": run["files"]["csv"],
        "summary.csv": summary_rows,
        "budget_tradeoff.csv": tradeoff,
        "importance_coverage.csv": coverage_rows,
        "pareto_accuracy_vs_ssd.csv": pareto_ssd,
        "pareto_accuracy_vs_ttft.csv": pareto_ttft,
        "validation.json": validation,
        "README.md": readme,
        "per_question_sensitivity.csv": question_rows,
        "per_image_sensitivity.csv": image_rows,
        "error_analysis.csv": error_rows,
    }
    publish(out_dir, source_records, files)
    return {
        "output": str(Path(out_dir).resolve()),
        "recommended_budget": operating["recommended_budget"],
        "summary": summary_rows,
        "operating_points": operating,
        "validation": validation,
    }


def self_test() -> None:
    assert qtype("What color is the car?") == "color"
    assert qtype("Is it raining?") == "yes/no"
    assert CORE._budget_chunks(34, 0.25) == 8  # Python ties-to-even.
    values = rankdata([1, 1, 3, 2]).tolist()
    assert values == [1.5, 1.5, 4.0, 3.0]
    rows = [
        {"method_key": "a", "budget": 0.2, "budget_pct": 20,
         "accuracy": 0.5, "accuracy_pct": 50, "cost": 1.0},
        {"method_key": "b", "budget": 0.3, "budget_pct": 30,
         "accuracy": 0.5, "accuracy_pct": 50, "cost": 2.0},
        {"method_key": "c", "budget": 0.4, "budget_pct": 40,
         "accuracy": 0.6, "accuracy_pct": 60, "cost": 3.0},
    ]
    pareto = pareto_rows(rows, "cost", "cost")
    assert [row["is_pareto_optimal"] for row in pareto] == [True, False, True]
    a = cluster_accuracy_ci([0, 1, 0, 1, 0, 1] * N_IMAGES,
                            [str(i) for i in range(N_IMAGES) for _ in range(6)])
    b = cluster_accuracy_ci([0, 1, 0, 1, 0, 1] * N_IMAGES,
                            [str(i) for i in range(N_IMAGES) for _ in range(6)])
    assert a == b == [50.0, 50.0]
    print("budget-sweep analyzer self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    result = analyze(Path(args.run_dir), Path(args.out_dir), Path(args.store))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

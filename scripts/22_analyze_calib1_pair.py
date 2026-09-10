#!/usr/bin/env python3
"""Strict lightweight analysis of the supplementary calib=1 GQA pair run.

This analyzer reads the already completed three-arm run and store metadata.  It
does not load the model, touch CUDA, or hash/read the large K/V payloads.  The
calib=4 analysis remains wholly separate in ``scripts/20_*``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
CORE_PATH = ROOT / "scripts" / "20_analyze_reorder_prefix.py"
_SPEC = importlib.util.spec_from_file_location("reorder_prefix_analysis_core",
                                               CORE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load analysis helpers: {CORE_PATH}")
CORE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(CORE)

AnalysisError = CORE.AnalysisError
SOURCE = (ROOT / "runs/reorder_prefix_baseline/calib1_pair25").resolve()
OUTPUT = (ROOT / "results/reorder_prefix_baseline/calib1_pair25").resolve()
STORE = (ROOT / "kvstore_reorder_prefix_calib1").resolve()
INDEX = (ROOT / "data/index.json").resolve()

METHODS = ["fullload", "reorder_prefix_chunk", "static_diverse_chunk"]
DISPLAY = {
    "fullload": "FullLoad",
    "reorder_prefix_chunk": "Reorder + Prefix 25%",
    "static_diverse_chunk": "Reorder + Static+Diverse 25%",
}
EXPECTED_INDEX_SHA256 = (
    "514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a"
)
EXPECTED_WORKLOAD_SHA256 = (
    "97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66"
)
EXPECTED_CALIB1_IDS_SHA256 = (
    "ba6b7e06c31853e2ef6009627026674b0e54e1b40d77e0def59c7c60be148438"
)
BUDGET = 0.25
LAYERS = 32


def sha256_file(path: Path, block=8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            data = f.read(block)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise AnalysisError(f"CSV has no header: {path}")
        fields = list(reader.fieldnames)
        dup = sorted(k for k, n in Counter(fields).items() if n > 1)
        if dup:
            raise AnalysisError(f"duplicate CSV headers: {dup}")
        return fields, list(reader)


def number(row: dict, key: str, *, optional=False) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        if optional:
            return None
        raise AnalysisError(f"missing numeric field {key}")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"invalid {key}={value!r}") from exc
    if not math.isfinite(value):
        raise AnalysisError(f"non-finite {key}={value!r}")
    return value


def integer(row: dict, key: str, *, optional=False) -> int | None:
    value = number(row, key, optional=optional)
    if value is None:
        return None
    result = int(round(value))
    if abs(value - result) > 1e-9:
        raise AnalysisError(f"non-integral {key}={value}")
    return result


def json_cell(row: dict, key: str):
    value = row.get(key)
    if value in (None, "", "null"):
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"invalid JSON in {key}") from exc


def strict_layers(value, context: str) -> list[list[int]]:
    if not isinstance(value, list):
        raise AnalysisError(f"{context}: selection is not a list")
    result = []
    for li, layer in enumerate(value):
        if not isinstance(layer, list):
            raise AnalysisError(f"{context}: layer {li} is not a list")
        if any(type(x) is not int for x in layer):
            raise AnalysisError(f"{context}: non-integer chunk ID at layer {li}")
        result.append(list(layer))
    return result


def separator_positions(meta: dict, layer: int) -> list[int]:
    values = meta.get("newline_stored", meta.get("newline_idx", []))
    if values and isinstance(values[0], list):
        values = values[layer]
    return [int(x) for x in values]


def layer_bytes(meta: dict, chunks: list[int]) -> int:
    itemsize = {"float16": 2, "bfloat16": 2, "float32": 4}[meta["dtype"]]
    row_bytes = int(meta["num_heads"]) * int(meta["head_dim"]) * itemsize
    vn, cs = int(meta["v_token_num"]), int(meta["chunk_size"])
    rows = sum(max(0, min(vn, (c + 1) * cs) - c * cs) for c in chunks)
    return 2 * rows * row_bytes


def sep_bytes(meta: dict) -> int:
    itemsize = {"float16": 2, "bfloat16": 2, "float32": 4}[meta["dtype"]]
    row_bytes = int(meta["num_heads"]) * int(meta["head_dim"]) * itemsize
    return sum(2 * len(separator_positions(meta, li)) * row_bytes
               for li in range(int(meta["num_layers"])))


def runs(chunks: list[int]) -> int:
    if not chunks:
        return 0
    ordered = sorted(chunks)
    return 1 + sum(b != a + 1 for a, b in zip(ordered, ordered[1:]))


def write_json(path: Path, value) -> None:
    with path.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def source_crosscheck(results: dict, rows: list[dict]) -> list[str]:
    failures = []
    nested = {}
    for rec in results.get("rows", []):
        qkey = (str(rec.get("image_id")), str(rec.get("question_id")))
        for method in METHODS:
            if method not in rec:
                continue
            key = (method, *qkey)
            if key in nested:
                failures.append(f"duplicate results.json key {key}")
            nested[key] = rec[method]
    flat = {(r["method_key"], str(r["image_id"]), str(r["question_id"])): r
            for r in rows}
    if set(flat) != set(nested):
        return ["results.json and per_request.csv key sets differ"]
    int_pairs = (
        ("ssd_read_bytes", "ssd_read_bytes"),
        ("ssd_read_chunks", "ssd_read_chunks"),
        ("preads", "ssd_preads"),
        ("normal_chunk_count_total", "normal_chunk_count_total"),
        ("normal_kv_read_bytes", "normal_kv_read_bytes"),
        ("separator_read_bytes", "separator_read_bytes"),
        ("normal_kv_preads", "normal_kv_preads"),
        ("separator_preads", "separator_preads"),
        ("total_actual_pread_bytes", "total_actual_pread_bytes"),
        ("static_score_calls", "static_score_calls"),
        ("query_score_calls", "query_score_calls"),
        ("diversity_calls", "diversity_calls"),
    )
    for key, value in nested.items():
        row = flat[key]
        if str(value.get("answer", "")) != row.get("prediction", ""):
            failures.append(f"prediction mismatch {key}")
        if abs(float(value["acc"]) - number(row, "correct")) > 1e-12:
            failures.append(f"accuracy mismatch {key}")
        for json_key, csv_key in int_pairs:
            a, b = value.get(json_key), integer(row, csv_key, optional=True)
            if (a is None) != (b is None) or (
                    a is not None and (type(a) is not int or a != b)):
                failures.append(f"{json_key} mismatch {key}")
        for json_key, csv_key, scale in (
                ("ttft", "ttft_ms", 1000.0),
                ("decode_ms", "decode_ms", 1.0),
                ("e2e_latency_ms", "e2e_latency_ms", 1.0),
                ("selector_ms", "selector_ms", 1.0),
                ("ssd_read_ms", "ssd_read_ms", 1.0),
                ("scatter_ms", "scatter_ms", 1.0),
                ("prefill_ms", "prefill_ms", 1.0)):
            a, b = value.get(json_key), number(row, csv_key, optional=True)
            if (a is None) != (b is None) or (
                    a is not None and abs(float(a) * scale - b) > 1e-6):
                failures.append(f"{json_key} mismatch {key}")
        for name in ("selection_mode", "separator_policy"):
            if value.get(name) != (row.get(name) or None):
                failures.append(f"{name} mismatch {key}")
        a, b = value.get("selected_chunk_ids_per_layer"), json_cell(
            row, "selected_chunk_ids_per_layer")
        if a is not None:
            a = strict_layers(a, f"results.json {key}")
        if b is not None:
            b = strict_layers(b, f"per_request.csv {key}")
        if a != b:
            failures.append(f"selection mismatch {key}")
    return failures


def summarize(rows_by_method: dict[str, list[dict]]) -> list[dict]:
    full_acc = np.mean([number(r, "correct")
                        for r in rows_by_method["fullload"]])
    prefix_acc = np.mean([number(r, "correct")
                          for r in rows_by_method["reorder_prefix_chunk"]])
    output = []
    for method in METHODS:
        rows = rows_by_method[method]

        def values(key, optional=False):
            data = [number(r, key, optional=optional) for r in rows]
            return np.asarray([x for x in data if x is not None])

        def mean_optional(key):
            data = values(key, True)
            return float(data.mean()) if data.size else ""

        acc, ttft = values("correct"), values("ttft_ms")
        disk = values("ssd_read_bytes")
        selector = values("selector_ms", True)
        output.append({
            "method_key": method,
            "method": DISPLAY[method],
            "n_requests": len(rows),
            "n_images": len({r["image_id"] for r in rows}),
            "accuracy": float(acc.mean()),
            "delta_accuracy_vs_fullload_pp":
                float((acc.mean() - full_acc) * 100),
            "delta_accuracy_vs_prefix_pp":
                float((acc.mean() - prefix_acc) * 100),
            "ttft_mean_ms": float(ttft.mean()),
            "ttft_p50_ms": float(np.percentile(ttft, 50)),
            "ttft_p95_ms": float(np.percentile(ttft, 95)),
            "decode_mean_ms": float(values("decode_ms").mean()),
            "e2e_mean_ms": float(values("e2e_latency_ms").mean()),
            "ssd_read_bytes_mean": float(disk.mean()),
            "ssd_read_mb_mean": float(disk.mean() / 1e6),
            "ssd_ratio_vs_fullload": float(
                disk.mean() / np.mean([
                    number(r, "ssd_read_bytes")
                    for r in rows_by_method["fullload"]])),
            "selected_chunks_per_layer_mean":
                mean_optional("n_chunks_selected"),
            "touched_chunk_fraction_mean":
                mean_optional("touched_chunk_fraction"),
            "selector_mean_ms": float(selector.mean()) if selector.size else "",
            "selector_p95_ms": (float(np.percentile(selector, 95))
                                  if selector.size else ""),
            "normal_kv_read_bytes_mean":
                mean_optional("normal_kv_read_bytes"),
            "separator_read_bytes_mean":
                mean_optional("separator_read_bytes"),
            "normal_kv_preads_mean": mean_optional("normal_kv_preads"),
            "separator_preads_mean": mean_optional("separator_preads"),
        })
    return output


SUMMARY_FIELDS = [
    "method_key", "method", "n_requests", "n_images", "accuracy",
    "delta_accuracy_vs_fullload_pp", "delta_accuracy_vs_prefix_pp",
    "ttft_mean_ms", "ttft_p50_ms", "ttft_p95_ms", "decode_mean_ms",
    "e2e_mean_ms", "ssd_read_bytes_mean", "ssd_read_mb_mean",
    "ssd_ratio_vs_fullload", "selected_chunks_per_layer_mean",
    "touched_chunk_fraction_mean", "selector_mean_ms", "selector_p95_ms",
    "normal_kv_read_bytes_mean", "separator_read_bytes_mean",
    "normal_kv_preads_mean", "separator_preads_mean",
]

OVERLAP_FIELDS = [
    "image_id", "layer", "n_chunks_total", "k", "intersection_count",
    "union_count", "sd_only_chunk_count", "prefix_only_chunk_count",
    "jaccard", "prefix_chunk_ids", "static_diverse_chunk_ids",
]


def build_readme(summary: list[dict], paired: dict, overlap: dict,
                 config: dict) -> str:
    by = {r["method_key"]: r for r in summary}
    p, sd = by["reorder_prefix_chunk"], by["static_diverse_chunk"]
    ci = paired["bootstrap"]["image_cluster_primary"]["delta_ci95"]
    mc = paired["mcnemar"]
    lines = [
        "| Method | Accuracy | Δ vs FullLoad | TTFT mean / p50 / p95 | "
        "SSD MB/request | SSD/Full | Selector |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        selector = ("-" if row["selector_mean_ms"] == "" else
                    f"{row['selector_mean_ms']:.3f} ms")
        lines.append(
            f"| {row['method']} | {100*row['accuracy']:.2f}% | "
            f"{row['delta_accuracy_vs_fullload_pp']:+.2f} pp | "
            f"{row['ttft_mean_ms']:.2f} / {row['ttft_p50_ms']:.2f} / "
            f"{row['ttft_p95_ms']:.2f} ms | {row['ssd_read_mb_mean']:.3f} | "
            f"{100*row['ssd_ratio_vs_fullload']:.2f}% | {selector} |")
    return f"""# Supplementary calib=1 Reorder+Prefix control

Validation: **PASS**

This is the frozen GQA 40-image / 240-question sensitivity run at a 25%
normal-chunk budget. It is supplementary and does not replace the preregistered
calib=4 primary comparison.

## Results

{chr(10).join(lines)}

Static+Diverse - Prefix accuracy is **{paired['delta_pp']:+.2f} pp**. The
primary 95% image-cluster paired-bootstrap CI is
**[{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] pp** (40 image clusters, 10,000
resamples, seed 0). The supplementary question bootstrap is also recorded in
`paired_stats.json`.

Exact McNemar discordances are SD-only **{mc['a_only']}** and Prefix-only
**{mc['b_only']}** (two-sided p={mc['p_exact_two_sided']:.6g}). Thus this
calib=1 run provides no observed positive selection gain: Prefix is
{100*p['accuracy']:.2f}% and Static+Diverse is {100*sd['accuracy']:.2f}%.
This statement is descriptive; a confidence interval containing zero is not
evidence of equivalence.

## Chunk overlap

Across {overlap['n_image_layers']} image-layer pairs, mean/median Jaccard is
**{overlap['jaccard_mean']:.4f}/{overlap['jaccard_median']:.4f}**. Mean
intersection, SD-only, and Prefix-only counts are
{overlap['intersection_mean']:.3f}, {overlap['sd_only_mean']:.3f}, and
{overlap['prefix_only_mean']:.3f} chunks.

## Calibration provenance limitation

- Intended calib=1 IDs are the first question of each frozen image; their
  ordered SHA256 is `{EXPECTED_CALIB1_IDS_SHA256}`. Evaluation remains the
  disjoint `[4:10]` slice.
- The store metadata records the resulting per-layer permutations but does not
  self-record `calibration_questions=1` or the calibration-ID hash.
- Construction provenance states that `kvstore_reorder_prefix_calib1` was
  composed from a copy of the calib=4 store and then processed for calib=1.
  It was not independently rebuilt from a freshly generated canonical raster
  store. Consequently, when importance values tie, bitwise equality of token
  tie-order with a fresh raster-to-calib1 build is **not guaranteed**.
- Therefore this is a useful sensitivity check of the observed composed
  layout, not cryptographic proof of a uniquely reproducible fresh calib=1
  layout.

## Measurement and validation

True TTFT keeps the schema-v2 boundary through synchronized first-token
determination. `validation.json` checks the exact arms and paired workload,
frozen question/gold content, store geometry and permutations, first-k Prefix
semantics, selector counters, common separator sidecar, real recorded pread
bytes/counts, budgets, and `E2E = TTFT + decode`.

Source run: `{config['source']['run_dir']}`  
Store: `{config['store']['path']}`  
Source results SHA256: `{config['source']['results_sha256']}`
"""


def analyze(source: Path, output: Path, store: Path) -> dict:
    source, output, store = source.resolve(), output.resolve(), store.resolve()
    if source != SOURCE or output != OUTPUT or store != STORE:
        raise AnalysisError(
            "this bounded analyzer accepts only the declared calib1 source, "
            "output, and store")
    if output.exists() or os.path.lexists(output):
        raise AnalysisError(f"refusing to overwrite existing output: {output}")
    result_path, csv_path = source / "results.json", source / "per_request.csv"
    static_build_path = source / "calib1_static_build.json"
    for path in (result_path, csv_path, static_build_path, INDEX):
        if not path.is_file() or path.is_symlink():
            raise AnalysisError(f"missing/non-regular input: {path}")
    with result_path.open() as f:
        results = json.load(f)
    source_summary = results.get("summary", {})
    fields, rows = read_csv(csv_path)

    required_fields = {
        "dataset", "method_key", "retention", "retention_kind", "image_id",
        "question_id", "question", "prediction", "ground_truth", "correct",
        "selector_ms", "ssd_read_ms", "scatter_ms", "prefill_ms", "ttft_ms",
        "decode_ms", "e2e_latency_ms", "ssd_read_bytes", "ssd_read_chunks",
        "ssd_preads", "n_chunks_selected", "n_chunks_total",
        "touched_chunk_fraction", "normal_chunk_count_total",
        "normal_kv_read_bytes", "separator_read_bytes", "normal_kv_preads",
        "separator_preads", "total_actual_pread_bytes", "static_score_calls",
        "query_score_calls", "diversity_calls", "selection_mode",
        "selected_chunk_ids_per_layer", "separator_policy",
        "reordered_prefix_store_validated",
    }
    missing = sorted(required_fields - set(fields))
    if missing:
        raise AnalysisError(f"source CSV missing fields: {missing}")

    checks, failures = {}, []

    def check(name, passed, detail=None):
        checks[name] = {"passed": bool(passed), "detail": detail}
        if not passed:
            failures.append(name if detail is None else f"{name}: {detail}")

    check("schema_v2", results.get("schema_version") == 2
          and source_summary.get("schema_version") == 2)
    methods = {r["method_key"] for r in rows}
    check("exact_three_arms", methods == set(METHODS), sorted(methods))
    if methods != set(METHODS):
        raise AnalysisError(f"method matrix mismatch: {sorted(methods)}")
    keys = [(r["method_key"], str(r["image_id"]), str(r["question_id"]))
            for r in rows]
    check("exact_720_unique_rows", len(rows) == 720
          and len(keys) == len(set(keys)), len(rows))
    rows_by = {m: [r for r in rows if r["method_key"] == m] for m in METHODS}
    qsets = {
        m: {(str(r["image_id"]), str(r["question_id"])) for r in rows_by[m]}
        for m in METHODS
    }
    check("exact_paired_240_keys", all(len(rows_by[m]) == 240 for m in METHODS)
          and qsets[METHODS[0]] == qsets[METHODS[1]] == qsets[METHODS[2]],
          {m: len(qsets[m]) for m in METHODS})
    if not checks["exact_720_unique_rows"]["passed"] or not checks[
            "exact_paired_240_keys"]["passed"]:
        raise AnalysisError("invalid request matrix; paired analysis aborted")
    retention_failures = []
    expected_retention = {
        "fullload": (1.0, "full"),
        "reorder_prefix_chunk": (0.25, "chunk"),
        "static_diverse_chunk": (0.25, "chunk"),
    }
    for method, method_rows in rows_by.items():
        retention, kind = expected_retention[method]
        for row in method_rows:
            if (abs(number(row, "retention") - retention) > 1e-12
                    or row["retention_kind"] != kind
                    or row["dataset"] != "gqa"):
                retention_failures.append(
                    f"{method}/{row['image_id']}/{row['question_id']}")
    check("dataset_and_retention_semantics", not retention_failures,
          retention_failures[:20])

    cross = source_crosscheck(results, rows)
    check("results_json_csv_consistent", not cross, cross[:20])
    check("source_fixed_conditions",
          source_summary.get("n") == 240
          and source_summary.get("n_images") == 40
          and source_summary.get("skip") == 4
          and source_summary.get("questions_per_image_requested") == 6
          and source_summary.get("budgets") == [0.25]
          and abs(float(source_summary.get("ratio", -1)) - BUDGET) < 1e-12
          and abs(float(source_summary.get("budget", -1)) - BUDGET) < 1e-12
          and source_summary.get("metric") == "gqa"
          and source_summary.get("sep_policy") == "sidecar"
          and source_summary.get("cold") is True
          and source_summary.get("max_new_tokens") == 16,
          {k: source_summary.get(k) for k in (
              "n", "n_images", "skip", "questions_per_image_requested",
              "budgets", "ratio", "budget", "metric", "sep_policy", "cold",
              "max_new_tokens")})
    command = source_summary.get("command", "")
    check("single_run_declares_calib1_store_and_pair",
          "--store kvstore_reorder_prefix_calib1" in command
          and "--selectors reorder_prefix_chunk,static_diverse_chunk" in command
          and "--no-recompute" in command,
          command)

    with INDEX.open() as f:
        index = json.load(f)[:40]
    index_sha = sha256_file(INDEX)
    workload = [(str(e["image_id"]), str(q["question_id"]))
                for e in index for q in e["questions"][4:10]]
    workload_sha = hashlib.sha256("\n".join(
        f"{a}\t{b}" for a, b in workload).encode()).hexdigest()
    calib1_ids = [(str(e["image_id"]), str(e["questions"][0]["question_id"]))
                  for e in index]
    calib1_sha = hashlib.sha256("\n".join(
        f"{a}\t{b}" for a, b in calib1_ids).encode()).hexdigest()
    check("frozen_index_and_workload_hashes",
          index_sha == EXPECTED_INDEX_SHA256
          and source_summary.get("index_sha256") == EXPECTED_INDEX_SHA256
          and workload_sha == EXPECTED_WORKLOAD_SHA256
          and source_summary.get("workload_sha256") == EXPECTED_WORKLOAD_SHA256,
          {"index": index_sha, "workload": workload_sha})
    check("calib1_intended_id_hash", calib1_sha == EXPECTED_CALIB1_IDS_SHA256,
          calib1_sha)
    check("no_calibration_evaluation_id_overlap",
          not ({q for _, q in calib1_ids} & {q for _, q in workload}))

    canonical = {}
    for e in index:
        for q in e["questions"][4:10]:
            answers = q.get("answers", [q.get("answer")])
            canonical[(str(e["image_id"]), str(q["question_id"]))] = (
                q["question"], answers)
    content_failures = []
    for row in rows:
        key = (str(row["image_id"]), str(row["question_id"]))
        try:
            gold = json.loads(row["ground_truth"])
        except json.JSONDecodeError:
            content_failures.append(f"invalid gold JSON {key}")
            continue
        if key not in canonical or (row["question"], gold) != canonical[key]:
            content_failures.append(f"content mismatch {key}")
    counts = Counter(i for i, _ in workload)
    check("exact_frozen_question_and_gold_content",
          not content_failures and set(canonical) == qsets[METHODS[0]]
          and len(counts) == 40 and set(counts.values()) == {6},
          content_failures[:20])

    meta_by, meta_hashes = {}, {}
    store_failures = []
    payload_files = payload_bytes = 0
    image_ids = list(dict.fromkeys(image_id for image_id, _ in workload))
    for image_id in image_ids:
        directory = store / image_id
        meta_path = directory / "meta.json"
        static_path, sep_path = directory / "static.pt", directory / "sep_kv.bin"
        if not all(p.is_file() for p in (meta_path, static_path, sep_path)):
            store_failures.append(f"missing metadata/sidecar {image_id}")
            continue
        with meta_path.open() as f:
            meta = json.load(f)
        meta_by[image_id] = meta
        meta_hashes[image_id] = sha256_file(meta_path)
        if not (meta.get("model") == "llava-hf/llava-v1.6-vicuna-7b-hf"
                and meta.get("reordered") is True
                and meta.get("order_is_per_layer") is True
                and meta.get("dtype") == "float16"
                and meta.get("chunk_size") == 64
                and meta.get("num_layers") == LAYERS):
            store_failures.append(f"geometry/model/layout mismatch {image_id}")
            continue
        vn, orders = int(meta["v_token_num"]), meta.get("order", [])
        expected = set(range(vn))
        if len(orders) != LAYERS:
            store_failures.append(f"order layer count {image_id}")
            continue
        original_sep = sorted(int(x) for x in meta["newline_idx"])
        for li, order in enumerate(orders):
            if (len(order) != vn or set(int(x) for x in order) != expected
                    or sorted(int(order[p]) for p in separator_positions(meta, li))
                    != original_sep):
                store_failures.append(f"invalid permutation {image_id}/L{li}")
                break
        if sep_path.stat().st_size != sep_bytes(meta):
            store_failures.append(f"separator size {image_id}")
        expected_layer_bytes = (vn * int(meta["num_heads"])
                                * int(meta["head_dim"]) * 2)
        for li in range(LAYERS):
            for kind in ("k", "v"):
                path = directory / f"layer_{li:02d}/{kind}.bin"
                if not path.is_file() or path.stat().st_size != expected_layer_bytes:
                    store_failures.append(f"payload stat mismatch {image_id}/{li}/{kind}")
                else:
                    payload_files += 1
                    payload_bytes += path.stat().st_size
        if int(meta["bytes_visual_kv"]) != 2 * LAYERS * expected_layer_bytes:
            store_failures.append(f"bytes_visual_kv mismatch {image_id}")
    check("lightweight_store_geometry_and_permutations",
          not store_failures and len(meta_by) == 40,
          store_failures[:20])

    selection = {}
    selection_failures, byte_failures, counter_failures = [], [], []
    modes = {
        "reorder_prefix_chunk": ("prefix", (0, 0, 0)),
        "static_diverse_chunk": ("static_diverse", (LAYERS, 0, LAYERS)),
    }
    for method, (mode, expected_calls) in modes.items():
        grouped = defaultdict(list)
        for row in rows_by[method]:
            grouped[str(row["image_id"])].append(row)
        for image_id in meta_by:
            image_rows = grouped[image_id]
            parsed = [strict_layers(json_cell(r, "selected_chunk_ids_per_layer"),
                                    f"{method}/{image_id}/{r['question_id']}")
                      for r in image_rows]
            if not parsed or any(x != parsed[0] for x in parsed[1:]):
                selection_failures.append(f"question-dependent/missing {method}/{image_id}")
                continue
            layers = parsed[0]
            selection[(method, image_id)] = layers
            meta = meta_by[image_id]
            nc = int(meta["n_chunks_per_layer"])
            k = max(1, min(nc, int(round(BUDGET * nc))))
            if len(layers) != LAYERS:
                selection_failures.append(f"layer count {method}/{image_id}")
                continue
            normal_bytes = 0
            normal_preads = 0
            for li, ids in enumerate(layers):
                if (ids != sorted(set(ids)) or len(ids) != k
                        or any(c < 0 or c >= nc for c in ids)):
                    selection_failures.append(f"budget/IDs {method}/{image_id}/L{li}")
                if mode == "prefix" and ids != list(range(k)):
                    selection_failures.append(f"not first-k {image_id}/L{li}")
                normal_bytes += layer_bytes(meta, ids)
                normal_preads += 2 * runs(ids)
            separator = sep_bytes(meta)
            for row in image_rows:
                calls = tuple(integer(row, key) for key in (
                    "static_score_calls", "query_score_calls", "diversity_calls"))
                if (row["selection_mode"] != mode or calls != expected_calls):
                    counter_failures.append(f"mode/counters {method}/{image_id}")
                if mode == "prefix" and row[
                        "reordered_prefix_store_validated"].lower() != "true":
                    counter_failures.append(f"prefix runtime validation {image_id}")
                if (row["separator_policy"] != "sidecar"
                        or integer(row, "separator_read_bytes") != separator
                        or integer(row, "separator_preads") != 1):
                    byte_failures.append(f"separator {method}/{image_id}")
                if (integer(row, "normal_chunk_count_total") != LAYERS * k
                        or integer(row, "ssd_read_chunks") != 2 * LAYERS * k
                        or abs(number(row, "n_chunks_selected") - k) > 1e-12
                        or integer(row, "n_chunks_total") != nc
                        or abs(number(row, "touched_chunk_fraction") - k / nc) > 1e-12):
                    selection_failures.append(f"request budget {method}/{image_id}")
                if (integer(row, "normal_kv_read_bytes") != normal_bytes
                        or integer(row, "normal_kv_preads") != normal_preads
                        or integer(row, "ssd_read_bytes") != normal_bytes + separator
                        or integer(row, "total_actual_pread_bytes") != normal_bytes + separator
                        or integer(row, "ssd_preads") != normal_preads + 1):
                    byte_failures.append(f"pread accounting {method}/{image_id}")
    check("fixed_question_independent_selections_and_budgets",
          not selection_failures and len(selection) == 80,
          selection_failures[:20])
    check("selector_modes_and_zero_prefix_counters", not counter_failures,
          counter_failures[:20])
    check("sidecar_and_exact_pread_accounting", not byte_failures,
          byte_failures[:20])

    full_failures = []
    for row in rows_by["fullload"]:
        meta = meta_by[str(row["image_id"])]
        expected_bytes = int(meta["bytes_visual_kv"])
        if (integer(row, "ssd_read_bytes") != expected_bytes
                or integer(row, "normal_kv_read_bytes") != expected_bytes
                or integer(row, "separator_read_bytes") != 0
                or integer(row, "total_actual_pread_bytes") != expected_bytes
                or integer(row, "ssd_read_chunks") !=
                2 * LAYERS * int(meta["n_chunks_per_layer"])
                or integer(row, "ssd_preads") != 2 * LAYERS
                or integer(row, "normal_kv_preads") != 2 * LAYERS
                or integer(row, "separator_preads") != 0):
            full_failures.append(f"{row['image_id']}/{row['question_id']}")
    check("fullload_exact_payload_accounting", not full_failures,
          full_failures[:20])

    timing_failures, max_residual = [], 0.0
    for row in rows:
        numeric = [number(row, key) for key in (
            "ttft_ms", "decode_ms", "e2e_latency_ms", "ssd_read_ms",
            "ssd_read_bytes", "ssd_read_chunks", "ssd_preads")]
        if any(x < 0 for x in numeric):
            timing_failures.append(f"negative {row['method_key']}/{row['question_id']}")
        ttft, decode, e2e = numeric[:3]
        if not ttft < e2e:
            timing_failures.append(f"TTFT !< E2E {row['method_key']}/{row['question_id']}")
        max_residual = max(max_residual, abs(e2e - ttft - decode))
        score = number(row, "correct")
        if score not in (0.0, 1.0):
            timing_failures.append(f"nonbinary score {row['method_key']}/{row['question_id']}")
    check("binary_scores_and_true_ttft_invariants",
          not timing_failures and max_residual <= 0.1,
          {"failures": timing_failures[:20], "max_residual_ms": max_residual})

    overlap_rows = []
    for image_id in meta_by:
        prefix = selection[("reorder_prefix_chunk", image_id)]
        sd = selection[("static_diverse_chunk", image_id)]
        nc = int(meta_by[image_id]["n_chunks_per_layer"])
        k = max(1, min(nc, int(round(BUDGET * nc))))
        for li, (pa, sa) in enumerate(zip(prefix, sd)):
            a, b = set(pa), set(sa)
            inter, union = a & b, a | b
            overlap_rows.append({
                "image_id": image_id, "layer": li, "n_chunks_total": nc,
                "k": k, "intersection_count": len(inter),
                "union_count": len(union), "sd_only_chunk_count": len(b-a),
                "prefix_only_chunk_count": len(a-b),
                "jaccard": len(inter) / len(union),
                "prefix_chunk_ids": json.dumps(pa, separators=(",", ":")),
                "static_diverse_chunk_ids": json.dumps(sa, separators=(",", ":")),
            })
    overlap = {
        "n_image_layers": len(overlap_rows),
        "jaccard_mean": float(np.mean([r["jaccard"] for r in overlap_rows])),
        "jaccard_median": float(np.median([r["jaccard"] for r in overlap_rows])),
        "intersection_mean": float(np.mean([
            r["intersection_count"] for r in overlap_rows])),
        "sd_only_mean": float(np.mean([
            r["sd_only_chunk_count"] for r in overlap_rows])),
        "prefix_only_mean": float(np.mean([
            r["prefix_only_chunk_count"] for r in overlap_rows])),
    }
    check("exact_1280_overlap_rows", len(overlap_rows) == 40 * LAYERS,
          len(overlap_rows))

    ordered_keys = workload
    maps = {m: {(str(r["image_id"]), str(r["question_id"])): r
                for r in rows_by[m]} for m in METHODS}
    sd_correct = np.asarray([
        number(maps["static_diverse_chunk"][k], "correct") for k in ordered_keys])
    prefix_correct = np.asarray([
        number(maps["reorder_prefix_chunk"][k], "correct") for k in ordered_keys])
    bootstrap = CORE._paired_bootstrap(
        sd_correct, prefix_correct, [k[0] for k in ordered_keys], 10000, 0)
    paired = {
        "comparison": "Static+Diverse25 - Prefix25",
        "a_method": "static_diverse_chunk",
        "b_method": "reorder_prefix_chunk",
        "delta_pp": float((sd_correct - prefix_correct).mean() * 100),
        "bootstrap": bootstrap,
        "mcnemar": CORE._exact_mcnemar(sd_correct, prefix_correct),
        "overlap_summary": overlap,
        "scope": "supplementary calib1 composed-layout sensitivity run",
    }
    summary = summarize(rows_by)

    validation = {
        "schema_version": 1,
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "counts": {"images": 40, "questions": 240, "methods": 3,
                   "per_request_rows": len(rows),
                   "chunk_overlap_rows": len(overlap_rows)},
    }
    config = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "analysis": "supplementary_calib1_prefix_vs_static_diverse",
        "source": {"run_dir": str(source),
                   "results_sha256": sha256_file(result_path),
                   "per_request_sha256": sha256_file(csv_path),
                   "static_build_sha256": sha256_file(static_build_path),
                   "command": command},
        "workload": {"index": str(INDEX), "index_sha256": index_sha,
                     "evaluation_workload_sha256": workload_sha,
                     "calibration_questions_per_image": 1,
                     "calibration_question_ids_sha256": calib1_sha,
                     "evaluation_slice": "questions[4:10]"},
        "store": {"path": str(store), "metadata_only_validation": True,
                  "meta_sha256_by_image": meta_hashes,
                  "payload_files_stat_checked": payload_files,
                  "payload_bytes_stat_checked": payload_bytes,
                  "large_payload_content_hash_performed": False},
        "statistics": {"primary": "image-cluster paired bootstrap",
                       "resamples": 10000, "seed": 0,
                       "supplement": ["question paired bootstrap",
                                      "exact McNemar"]},
        "provenance_limitation": {
            "status": "declared_composed_layout_not_bitwise_fresh_verified",
            "statement": ("calib1 store was composed from a calib4 store copy; "
                          "bitwise tie-order equivalence to a fresh raster-to-"
                          "calib1 build is not guaranteed"),
            "meta_records_calibration_count": False,
        },
        "code_sha256": {"analyzer": sha256_file(Path(__file__)),
                        "shared_statistics_helpers": sha256_file(CORE_PATH)},
    }
    if failures:
        raise AnalysisError(f"validation failed; no output published: {failures[:5]}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".calib1_pair25.tmp-",
                                    dir=output.parent))
    parent_fd = None
    try:
        write_json(staging / "config.json", config)
        shutil.copyfile(csv_path, staging / "per_request.csv")
        if sha256_file(staging / "per_request.csv") != config[
                "source"]["per_request_sha256"]:
            raise AnalysisError("per_request copy hash mismatch")
        write_csv(staging / "summary.csv", SUMMARY_FIELDS, summary)
        write_json(staging / "validation.json", validation)
        write_json(staging / "paired_stats.json", paired)
        write_csv(staging / "chunk_overlap.csv", OVERLAP_FIELDS, overlap_rows)
        (staging / "README.md").write_text(
            build_readme(summary, paired, overlap, config))
        required = {"config.json", "per_request.csv", "summary.csv",
                    "validation.json", "paired_stats.json",
                    "chunk_overlap.csv", "README.md"}
        if any(not (staging / name).is_file() for name in required):
            raise AnalysisError("staging output incomplete")
        parent_fd = os.open(output.parent,
                            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        CORE._rename_noreplace(parent_fd, staging.name,
                               parent_fd, output.name)
    except Exception:
        if staging.exists() and staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    finally:
        if parent_fd is not None:
            os.close(parent_fd)
    return {"output": str(output), "validation_passed": True,
            "prefix_accuracy": next(r["accuracy"] for r in summary
                                    if r["method_key"] == "reorder_prefix_chunk"),
            "static_diverse_accuracy": next(r["accuracy"] for r in summary
                    if r["method_key"] == "static_diverse_chunk"),
            "delta_pp": paired["delta_pp"],
            "image_cluster_ci95_pp": [100*x for x in bootstrap[
                "image_cluster_primary"]["delta_ci95"]]}


def self_test() -> None:
    assert runs([0, 1, 4, 5, 9]) == 3
    assert strict_layers([[0, 1], [2]], "test") == [[0, 1], [2]]
    for bad in ([[1.0]], [["1"]], [[True]]):
        try:
            strict_layers(bad, "bad")
        except AnalysisError:
            pass
        else:
            raise AssertionError(f"accepted {bad!r}")
    a = np.asarray([1, 0, 1, 1], dtype=float)
    b = np.asarray([0, 0, 1, 0], dtype=float)
    first = CORE._paired_bootstrap(a, b, ["a", "a", "b", "b"], 100, 7)
    second = CORE._paired_bootstrap(a, b, ["a", "a", "b", "b"], 100, 7)
    assert first == second
    print("self-test PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--store", default=str(STORE))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    result = analyze(Path(args.source), Path(args.output), Path(args.store))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

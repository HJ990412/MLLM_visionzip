#!/usr/bin/env python3
"""Strict analysis for the GQA reorder-prefix control experiment.

This script never mutates the source run or the legacy result directories.  It
reads one schema-v2 run produced by ``scripts/04_eval.py``, validates the exact
40-image / 240-question paired workload, and publishes a self-contained report
under a new ``results/`` directory.

The primary inferential comparison is fixed before looking at the result:

    Static+Diverse25 - Reorder+Prefix25

Its 95% confidence interval is an image-cluster paired bootstrap (40 images,
10,000 draws).  A question-level paired bootstrap is included only as a
supplement, and exact McNemar discordances are reported for the binary GQA
outcomes.

Example:

  python scripts/20_analyze_reorder_prefix.py \
    --run-dir runs/reorder_prefix_baseline/main_calib4 \
    --out-dir results/reorder_prefix_baseline/main_calib4 \
    --store kvstore
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import errno
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_INDEX_SHA256 = (
    "514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a"
)
EXPECTED_WORKLOAD_SHA256 = (
    "97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66"
)
EXPECTED_CALIBRATION_WORKLOAD_SHA256 = (
    "992cc89a81a6cadf363b65b58f8f89fabfc0a1a17358559069f8c0b6cabc1e71"
)
EXPECTED_STORE_SHA256 = (
    "e570a6847743a203fc1e2892d736ebe8aa946647cbb0e280f212388da2c09d68"
)
EXPECTED_STORE_SELECTION_SHA256 = (
    "616e50ae052aee5d9a284e31a24fae463bd793e9b572ceeca859ad87270e8a82"
)

# These were recorded before the new baseline experiment.  Directory digests
# use the same canonical format as:
#   find ROOT -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum
LEGACY_PRESERVATION = {
    "runs/gqa40_240_true_ttft": {
        "kind": "tree",
        "sha256": "cc7928fa1300e8a66fad195e2a0110523972f8bffd36a18ec62d1eb74bf7ba19",
    },
    "runs/gqa40_240_true_ttft_budget_10_15_20": {
        "kind": "tree",
        "sha256": "04a7f1bfdef359adbe772f01899da0f1988b596472711d5a97daf78407da98b4",
    },
    "results/ablation_25": {
        "kind": "tree",
        "sha256": "dfc290746067320ecca71cdb7523dc3ce3b85b026fac3ad619bbc7598fbf3634",
    },
    "results/budget_sweep": {
        "kind": "tree",
        "sha256": "0b83ae8e27b25804ce57024c219d1ef51c3f4e934e708fb9737fb7085906b93c",
    },
    "results/eval_b25.json": {
        "kind": "file",
        "sha256": "c656871497716d5f5e2355b9fcacb69ebdaf163b3ed345dfbc6adb3d91786b10",
    },
    "results_reorder.log": {
        "kind": "file",
        "sha256": "fc8af24e66302d6955986f65056c7857e629a29f89755258c7e42a45d7576385",
    },
}

REQUIRED_METHODS = [
    "recompute",
    "fullload",
    "sparsevlm",
    "reorder_prefix_chunk@25",
    "visionzip_static_chunk@25",
    "diverse_chunk@25",
    "static_diverse_chunk@25",
]

OPTIONAL_METHODS = [
    "reorder_prefix_chunk@50",
    "visionzip_static_chunk@50",
    "diverse_chunk@50",
    "static_diverse_chunk@50",
]

METHODS = REQUIRED_METHODS + OPTIONAL_METHODS

CHUNK_METHODS = {
    "reorder_prefix_chunk@25": ("prefix", 0.25),
    "visionzip_static_chunk@25": ("static", 0.25),
    "diverse_chunk@25": ("diverse_only", 0.25),
    "static_diverse_chunk@25": ("static_diverse", 0.25),
    "reorder_prefix_chunk@50": ("prefix", 0.50),
    "visionzip_static_chunk@50": ("static", 0.50),
    "diverse_chunk@50": ("diverse_only", 0.50),
    "static_diverse_chunk@50": ("static_diverse", 0.50),
}

DISPLAY = {
    "recompute": "ReComp",
    "fullload": "FullLoad",
    "sparsevlm": "SparseVLM 25%",
    "reorder_prefix_chunk@25": "Reorder + Prefix 25%",
    "visionzip_static_chunk@25": "Reorder + Static 25%",
    "diverse_chunk@25": "Reorder + Diverse Only 25%",
    "static_diverse_chunk@25": "Reorder + Static+Diverse 25%",
    "reorder_prefix_chunk@50": "Reorder + Prefix 50%",
    "visionzip_static_chunk@50": "Reorder + Static 50%",
    "diverse_chunk@50": "Reorder + Diverse Only 50%",
    "static_diverse_chunk@50": "Reorder + Static+Diverse 50%",
}

DECISION_RULE = (
    "GO iff observed delta Accuracy(Static+Diverse25 - Prefix25) > 0 and "
    "the 95% image-cluster paired-bootstrap CI lower bound > 0; otherwise "
    "RETHINK"
)


class AnalysisError(RuntimeError):
    pass


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _stat_signature(path: Path, *, follow_symlinks=False) -> tuple:
    st = path.stat() if follow_symlinks else path.lstat()
    return _stat_result_signature(st)


def _stat_result_signature(st: os.stat_result) -> tuple:
    return (st.st_dev, st.st_ino, stat.S_IFMT(st.st_mode), st.st_size,
            st.st_mtime_ns)


def _paths_overlap(a: Path, b: Path) -> bool:
    """True for equality or either ancestor/descendant relationship."""
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def _reject_symlink_components(path: Path, anchor: Path) -> None:
    """Fail closed if an existing component below ``anchor`` is a symlink."""
    try:
        rel = path.relative_to(anchor)
    except ValueError as exc:
        raise AnalysisError(f"path escapes repository: {path}") from exc
    cur = anchor
    for part in rel.parts:
        cur = cur / part
        if _lexists(cur) and cur.is_symlink():
            raise AnalysisError(f"symlink path component is not allowed: {cur}")


def _rename_noreplace(src_dir_fd: int, src_name: str,
                      dst_dir_fd: int, dst_name: str) -> None:
    """Linux renameat2(RENAME_NOREPLACE), with no unsafe fallback.

    Analysis publication must never overwrite a path that appeared between a
    guard check and rename.  If the platform lacks this primitive, fail closed
    instead of silently falling back to overwrite-capable ``os.replace``.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise AnalysisError(
            "atomic no-replace rename is unavailable; refusing publication")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p,
                          ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    rc = renameat2(src_dir_fd, os.fsencode(src_name),
                   dst_dir_fd, os.fsencode(dst_name), 1)
    if rc != 0:
        err = ctypes.get_errno()
        if err == errno.EEXIST:
            raise AnalysisError(
                f"atomic publication target appeared concurrently: {dst_name}")
        raise OSError(err, os.strerror(err), f"{src_name} -> {dst_name}")


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_bytes)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _tree_digest(root: Path, include=None) -> tuple[str, int, int]:
    """GNU-sha256sum-compatible aggregate using repo-relative paths."""
    paths = []
    for p in root.rglob("*"):
        if p.is_file() and not p.is_symlink() and (include is None or include(p)):
            paths.append(p)
    paths.sort(key=lambda p: os.fsencode(p.relative_to(PROJECT_ROOT).as_posix()))
    outer = hashlib.sha256()
    total = 0
    for p in paths:
        rel = p.relative_to(PROJECT_ROOT).as_posix()
        size = p.stat().st_size
        total += size
        outer.update(f"{_sha256_file(p)}  {rel}\n".encode())
    return outer.hexdigest(), len(paths), total


def _stable_json_hash(value) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise AnalysisError(f"CSV has no header: {path}")
        fields = list(reader.fieldnames)
        duplicates = sorted(k for k, n in Counter(fields).items() if n > 1)
        if duplicates:
            raise AnalysisError(
                f"CSV has duplicate header names (ambiguous parse): {duplicates}")
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


def _float(row: dict, key: str, *, allow_blank=False) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        if allow_blank:
            return None
        raise AnalysisError(f"missing numeric field {key}: {row}")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"invalid {key}={value!r}") from exc
    if not math.isfinite(out):
        raise AnalysisError(f"non-finite {key}={value!r}")
    return out


def _int(row: dict, key: str, *, allow_blank=False) -> int | None:
    value = _float(row, key, allow_blank=allow_blank)
    if value is None:
        return None
    rounded = int(round(value))
    if abs(value - rounded) > 1e-9:
        raise AnalysisError(f"non-integral {key}={value}")
    return rounded


def _json_cell(row: dict, key: str):
    value = row.get(key)
    if value in (None, "", "null"):
        return None
    try:
        return json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"invalid JSON cell {key}: {value!r}") from exc


def _strict_chunk_layers(value, *, context: str) -> list[list[int]]:
    """Accept only a JSON array of arrays containing true integer IDs.

    In particular, booleans, numeric strings, and integral-looking floats are
    rejected instead of being silently coerced with ``int(...)``.
    """
    if not isinstance(value, list):
        raise AnalysisError(f"{context}: chunk selection is not a list")
    out = []
    for li, layer in enumerate(value):
        if not isinstance(layer, list):
            raise AnalysisError(f"{context}: layer {li} is not a list")
        if any(type(c) is not int for c in layer):
            bad = [repr(c) for c in layer if type(c) is not int][:5]
            raise AnalysisError(
                f"{context}: layer {li} has non-integer chunk IDs {bad}")
        out.append(list(layer))
    return out


def _contiguous_runs(cids: list[int]) -> int:
    """Number of exact byte ranges after merge_ranges(..., max_gap=0)."""
    if not cids:
        return 0
    ordered = sorted(cids)
    return 1 + sum(b != a + 1 for a, b in zip(ordered, ordered[1:]))


def _budget_chunk_count(n_chunks: int, budget: float) -> int:
    # Keep exactly the repository's Python-round (ties-to-even) convention.
    return max(1, min(n_chunks, int(round(budget * n_chunks))))


def _exact_mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact two-sided McNemar for binary a/b, with named discordances."""
    aa = np.asarray(a, dtype=float) == 1.0
    bb = np.asarray(b, dtype=float) == 1.0
    a_only = int((aa & ~bb).sum())
    b_only = int((~aa & bb).sum())
    both = int((aa & bb).sum())
    neither = int((~aa & ~bb).sum())
    n = a_only + b_only
    if n == 0:
        p = 1.0
    else:
        k = min(a_only, b_only)
        p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {
        "a_only": a_only,
        "b_only": b_only,
        "both_correct": both,
        "neither_correct": neither,
        "discordant": n,
        "p_exact_two_sided": p,
    }


def _paired_bootstrap(a: np.ndarray, b: np.ndarray, images: list[str],
                      n_boot: int, seed: int) -> dict:
    """Simultaneous question and image-cluster paired bootstrap."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) != len(images) or len(a) == 0:
        raise AnalysisError("paired bootstrap inputs are not aligned")

    rng = np.random.RandomState(seed)
    q_idx = rng.randint(0, len(a), size=(n_boot, len(a)))
    qa = a[q_idx].mean(axis=1)
    qb = b[q_idx].mean(axis=1)

    ordered_images = list(dict.fromkeys(images))
    by_image = defaultdict(list)
    for i, image_id in enumerate(images):
        by_image[image_id].append(i)
    # The primary workload has exactly six questions per image.  Use image
    # means so every sampled image is one independent cluster.
    ia = np.asarray([a[by_image[i]].mean() for i in ordered_images])
    ib = np.asarray([b[by_image[i]].mean() for i in ordered_images])
    rng = np.random.RandomState(seed)
    i_idx = rng.randint(0, len(ia), size=(n_boot, len(ia)))
    ca = ia[i_idx].mean(axis=1)
    cb = ib[i_idx].mean(axis=1)

    def ci(x):
        return [float(v) for v in np.percentile(x, [2.5, 97.5])]

    return {
        "n_questions": int(len(a)),
        "n_image_clusters": int(len(ordered_images)),
        "n_bootstrap": int(n_boot),
        "seed": int(seed),
        "a_mean": float(a.mean()),
        "b_mean": float(b.mean()),
        "delta_mean": float((a - b).mean()),
        "image_cluster_primary": {
            "a_ci95": ci(ca),
            "b_ci95": ci(cb),
            "delta_ci95": ci(ca - cb),
            "sampling_unit": "image; all six questions retained as one cluster",
        },
        "question_supplement": {
            "a_ci95": ci(qa),
            "b_ci95": ci(qb),
            "delta_ci95": ci(qa - qb),
            "sampling_unit": "question",
        },
    }


def _method_budget(method: str) -> float | None:
    return CHUNK_METHODS.get(method, (None, None))[1]


def _retention(method: str, source_summary: dict) -> tuple[float | None, str]:
    if method == "recompute":
        return None, "none"
    if method == "fullload":
        return 1.0, "full"
    if method == "sparsevlm":
        return float(source_summary.get("ratio", 0.25)), "token"
    return _method_budget(method), "chunk"


def _separator_positions(meta: dict, layer: int) -> list[int]:
    positions = meta.get("newline_stored", meta.get("newline_idx", []))
    if positions and isinstance(positions[0], list):
        positions = positions[layer]
    return [int(x) for x in positions]


def _normal_layer_bytes(meta: dict, cids: list[int]) -> int:
    itemsize = {"float16": 2, "float32": 4, "bfloat16": 2}[meta["dtype"]]
    row_bytes = int(meta["num_heads"]) * int(meta["head_dim"]) * itemsize
    vn, cs = int(meta["v_token_num"]), int(meta["chunk_size"])
    rows = sum(max(0, min(vn, (c + 1) * cs) - c * cs) for c in cids)
    return 2 * rows * row_bytes  # both K and V


def _separator_layer_bytes(meta: dict, layer: int) -> int:
    itemsize = {"float16": 2, "float32": 4, "bfloat16": 2}[meta["dtype"]]
    row_bytes = int(meta["num_heads"]) * int(meta["head_dim"]) * itemsize
    return 2 * len(_separator_positions(meta, layer)) * row_bytes


def _legacy_hashes(expected: dict) -> tuple[dict, list[str]]:
    observed, failures = {}, []
    for rel, spec in expected.items():
        p = PROJECT_ROOT / rel
        if not p.exists():
            observed[rel] = {"exists": False, **spec}
            failures.append(f"preserved path missing: {rel}")
            continue
        if spec["kind"] == "tree":
            digest, n_files, n_bytes = _tree_digest(p)
            row = {"exists": True, "kind": "tree", "sha256": digest,
                   "n_files": n_files, "n_bytes": n_bytes,
                   "expected_sha256": spec["sha256"]}
        else:
            digest = _sha256_file(p)
            row = {"exists": True, "kind": "file", "sha256": digest,
                   "n_files": 1, "n_bytes": p.stat().st_size,
                   "expected_sha256": spec["sha256"]}
        row["matches_expected"] = digest == spec["sha256"]
        observed[rel] = row
        if not row["matches_expected"]:
            failures.append(f"legacy preservation digest changed: {rel}")
    return observed, failures


def _source_results_crosscheck(results: dict, csv_rows: list[dict],
                               methods: list[str]) -> list[str]:
    failures = []
    nested = {}
    for rec in results.get("rows", []):
        qkey = (str(rec.get("image_id")), str(rec.get("question_id")))
        for method in methods:
            if method in rec:
                key = (method, *qkey)
                if key in nested:
                    failures.append(f"duplicate nested result: {key}")
                nested[key] = rec[method]
    flat = {(r["method_key"], str(r["image_id"]), str(r["question_id"])): r
            for r in csv_rows}
    if set(nested) != set(flat):
        failures.append("results.json and per_request.csv request keys differ")
        return failures
    for key, value in nested.items():
        row = flat[key]
        if str(value.get("answer", "")) != row.get("prediction", ""):
            failures.append(f"prediction mismatch between source artifacts: {key}")
        if abs(float(value.get("acc")) - _float(row, "correct")) > 1e-12:
            failures.append(f"accuracy mismatch between source artifacts: {key}")
        if abs(float(value.get("ttft")) * 1000 - _float(row, "ttft_ms")) > 1e-6:
            failures.append(f"TTFT mismatch between source artifacts: {key}")
        float_pairs = (
            ("selector_ms", "selector_ms", 1.0),
            ("ssd_read_ms", "ssd_read_ms", 1.0),
            ("prepare_ms", "prepare_ms", 1.0),
            ("scatter_ms", "scatter_ms", 1.0),
            ("prefill_ms", "prefill_ms", 1.0),
            ("decode_ms", "decode_ms", 1.0),
            ("e2e_latency_ms", "e2e_latency_ms", 1.0),
        )
        for json_key, csv_key, scale in float_pairs:
            jv = value.get(json_key)
            cv = _float(row, csv_key, allow_blank=True)
            if (jv is None) != (cv is None) or (
                    jv is not None and
                    abs(float(jv) * scale - cv) > 1e-6):
                failures.append(
                    f"{json_key} mismatch between source artifacts: {key}")

        integer_pairs = (
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
        for json_key, csv_key in integer_pairs:
            jv = value.get(json_key)
            cv = _int(row, csv_key, allow_blank=True)
            if (jv is None) != (cv is None) or (
                    jv is not None and
                    (type(jv) is not int or jv != cv)):
                failures.append(
                    f"{json_key} mismatch between source artifacts: {key}")

        for json_key, csv_key in (
                ("selection_mode", "selection_mode"),
                ("separator_policy", "separator_policy")):
            jv = value.get(json_key)
            cv = row.get(csv_key) or None
            if jv != cv:
                failures.append(
                    f"{json_key} mismatch between source artifacts: {key}")
        jv = value.get("reordered_prefix_store_validated")
        cv_raw = row.get("reordered_prefix_store_validated", "")
        if cv_raw == "":
            cv = None
        elif cv_raw.lower() in {"true", "1"}:
            cv = True
        elif cv_raw.lower() in {"false", "0"}:
            cv = False
        else:
            cv = object()
        if jv != cv:
            failures.append(
                "runtime store-validation flag mismatch between source "
                f"artifacts: {key}")

        j_selection = value.get("selected_chunk_ids_per_layer")
        c_selection = _json_cell(row, "selected_chunk_ids_per_layer")
        try:
            if j_selection is not None:
                j_selection = _strict_chunk_layers(
                    j_selection, context=f"results.json {key}")
            if c_selection is not None:
                c_selection = _strict_chunk_layers(
                    c_selection, context=f"per_request.csv {key}")
        except AnalysisError as exc:
            failures.append(str(exc))
        else:
            if j_selection != c_selection:
                failures.append(
                    f"chunk selection mismatch between source artifacts: {key}")
    return failures


def _summaries(rows_by_method: dict[str, list[dict]],
               methods: list[str]) -> list[dict]:
    full_bytes = np.mean([_float(r, "ssd_read_bytes")
                          for r in rows_by_method["fullload"]])
    full_acc = np.mean([_float(r, "correct")
                        for r in rows_by_method["fullload"]])
    prefix25_acc = np.mean([_float(r, "correct")
                            for r in rows_by_method["reorder_prefix_chunk@25"]])
    out = []
    for method in methods:
        rows = rows_by_method[method]
        def optional_values(key):
            values = [_float(r, key, allow_blank=True) for r in rows]
            return np.asarray([x for x in values if x is not None])

        def optional_mean(key):
            values = optional_values(key)
            return float(values.mean()) if values.size else ""

        acc = np.asarray([_float(r, "correct") for r in rows])
        ttft = np.asarray([_float(r, "ttft_ms") for r in rows])
        decode = np.asarray([_float(r, "decode_ms") for r in rows])
        e2e = np.asarray([_float(r, "e2e_latency_ms") for r in rows])
        disk = np.asarray([_float(r, "ssd_read_bytes") for r in rows])
        sel = [_float(r, "selector_ms", allow_blank=True) for r in rows]
        sel = np.asarray([x for x in sel if x is not None])
        chunks = [_float(r, "n_chunks_selected", allow_blank=True) for r in rows]
        chunks = np.asarray([x for x in chunks if x is not None])
        touched = [_float(r, "touched_chunk_fraction", allow_blank=True)
                   for r in rows]
        touched = np.asarray([x for x in touched if x is not None])
        retention, retention_kind = _retention(method, {})
        out.append({
            "method_key": method,
            "method": DISPLAY[method],
            "retention": "" if retention is None else retention,
            "retention_kind": retention_kind,
            "n_requests": len(rows),
            "n_images": len({r["image_id"] for r in rows}),
            "accuracy": float(acc.mean()),
            "delta_accuracy_vs_fullload_pp": float((acc.mean() - full_acc) * 100),
            "delta_accuracy_vs_prefix25_pp": (
                float((acc.mean() - prefix25_acc) * 100)
                if method == "static_diverse_chunk@25" else ""),
            "ttft_mean_ms": float(ttft.mean()),
            "ttft_p50_ms": float(np.percentile(ttft, 50)),
            "ttft_p95_ms": float(np.percentile(ttft, 95)),
            "decode_mean_ms": float(decode.mean()),
            "e2e_mean_ms": float(e2e.mean()),
            "ssd_read_bytes_mean": float(disk.mean()),
            "ssd_read_mb_mean": float(disk.mean() / 1e6),
            "ssd_read_chunks_mean": optional_mean("ssd_read_chunks"),
            "ssd_preads_mean": optional_mean("ssd_preads"),
            "ssd_read_mean_ms": optional_mean("ssd_read_ms"),
            "prepare_mean_ms": optional_mean("prepare_ms"),
            "scatter_mean_ms": optional_mean("scatter_ms"),
            "prefill_mean_ms": optional_mean("prefill_ms"),
            "normal_chunk_count_total_mean":
                optional_mean("normal_chunk_count_total"),
            "normal_kv_read_bytes_mean":
                optional_mean("normal_kv_read_bytes"),
            "separator_read_bytes_mean":
                optional_mean("separator_read_bytes"),
            "total_actual_pread_bytes_mean":
                optional_mean("total_actual_pread_bytes"),
            "normal_kv_preads_mean": optional_mean("normal_kv_preads"),
            "separator_preads_mean": optional_mean("separator_preads"),
            "static_score_calls_mean": optional_mean("static_score_calls"),
            "query_score_calls_mean": optional_mean("query_score_calls"),
            "diversity_calls_mean": optional_mean("diversity_calls"),
            "ssd_ratio_vs_fullload": float(disk.mean() / full_bytes)
            if full_bytes else 0.0,
            "selected_chunks_per_layer_mean": float(chunks.mean())
            if chunks.size else "",
            "touched_chunk_fraction_mean": float(touched.mean())
            if touched.size else "",
            "selector_mean_ms": float(sel.mean()) if sel.size else "",
            "selector_p50_ms": float(np.percentile(sel, 50)) if sel.size else "",
            "selector_p95_ms": float(np.percentile(sel, 95)) if sel.size else "",
        })
    return out


SUMMARY_FIELDS = [
    "method_key", "method", "retention", "retention_kind", "n_requests",
    "n_images", "accuracy", "delta_accuracy_vs_fullload_pp",
    "delta_accuracy_vs_prefix25_pp", "ttft_mean_ms", "ttft_p50_ms",
    "ttft_p95_ms", "decode_mean_ms", "e2e_mean_ms",
    "ssd_read_bytes_mean", "ssd_read_mb_mean", "ssd_read_chunks_mean",
    "ssd_preads_mean", "ssd_read_mean_ms", "prepare_mean_ms",
    "scatter_mean_ms", "prefill_mean_ms",
    "normal_chunk_count_total_mean", "normal_kv_read_bytes_mean",
    "separator_read_bytes_mean", "total_actual_pread_bytes_mean",
    "normal_kv_preads_mean", "separator_preads_mean",
    "static_score_calls_mean", "query_score_calls_mean",
    "diversity_calls_mean", "ssd_ratio_vs_fullload",
    "selected_chunks_per_layer_mean", "touched_chunk_fraction_mean",
    "selector_mean_ms", "selector_p50_ms", "selector_p95_ms",
]


OVERLAP_FIELDS = [
    "image_id", "layer", "budget", "n_chunks_total", "k",
    "intersection_count", "union_count", "sd_only_chunk_count",
    "prefix_only_chunk_count", "jaccard", "prefix_chunk_ids",
    "static_diverse_chunk_ids",
]


def _format_pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def _format_pp(x: float) -> str:
    return f"{x:+.2f} pp"


def _build_readme(summary_rows: list[dict], paired: dict, overlap_summary: dict,
                  validation: dict, config: dict) -> str:
    by = {r["method_key"]: r for r in summary_rows}
    p = by["reorder_prefix_chunk@25"]
    sd = by["static_diverse_chunk@25"]
    main = paired["static_diverse25_minus_prefix25"]
    ci = main["bootstrap"]["image_cluster_primary"]["delta_ci95"]
    mc = main["mcnemar"]
    ov = overlap_summary["25"]
    decision = main["decision"]

    table = [
        "| Method | Accuracy | Δ vs FullLoad | True TTFT mean / p50 / p95 | "
        "SSD MB/request | SSD/Full | Chunks/layer | Touched | Selector |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        chunks = ("-" if r["selected_chunks_per_layer_mean"] == "" else
                  f"{r['selected_chunks_per_layer_mean']:.3f}")
        touched = ("-" if r["touched_chunk_fraction_mean"] == "" else
                   f"{100*r['touched_chunk_fraction_mean']:.2f}%")
        selector = ("-" if r["selector_mean_ms"] == "" else
                    f"{r['selector_mean_ms']:.3f} ms")
        table.append(
            f"| {r['method']} | {_format_pct(r['accuracy'])} | "
            f"{_format_pp(r['delta_accuracy_vs_fullload_pp'])} | "
            f"{r['ttft_mean_ms']:.2f} / {r['ttft_p50_ms']:.2f} / "
            f"{r['ttft_p95_ms']:.2f} ms | {r['ssd_read_mb_mean']:.3f} | "
            f"{100*r['ssd_ratio_vs_fullload']:.2f}% | {chunks} | {touched} | "
            f"{selector} |")

    cause = ("C. importance reorder와 Static+Diverse selection의 결합"
             if decision == "GO" else
             "A를 주된 원인으로 재검토해야 함. 다만 RETHINK는 통계적 "
             "동등성 증명이 아니라 Static+Diverse의 양의 추가 이득이 이 "
             "실험에서 확립되지 않았다는 operational 판정이다.")
    status = "PASS" if validation["passed"] else "FAIL — 결과 해석 금지"
    return f"""# Reorder + Prefix baseline: GQA 40 images / 240 questions

Validation: **{status}**

이 디렉터리는 source run을 수정하지 않고 생성한 분석 사본이다. Primary
comparison과 판정 규칙은 다음과 같이 고정했다.

```text
{DECISION_RULE}
```

## 결과표

{chr(10).join(table)}

## 핵심 질문

### Q1. Importance reorder 뒤 first 25%만 읽은 정확도

**{_format_pct(p['accuracy'])}**

### Q2. Static+Diverse25의 추가 accuracy

Static+Diverse25 - Prefix25 = **{_format_pp(main['delta_pp'])}**.
Primary 95% image-cluster bootstrap CI는
**[{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] pp**다.

McNemar discordance는 SD-only **{mc['a_only']}**, Prefix-only
**{mc['b_only']}**, exact two-sided p={mc['p_exact_two_sided']:.6g}다.

### Q3. 선택 chunk overlap

25%의 (image, layer) {ov['n_image_layers']}쌍에서 평균 Jaccard는
**{ov['jaccard_mean']:.4f}**, median은 **{ov['jaccard_median']:.4f}**다.
평균 intersection/SD-only/Prefix-only는 각각
{ov['intersection_mean']:.3f}/{ov['sd_only_mean']:.3f}/
{ov['prefix_only_mean']:.3f} chunks다. Separator sidecar는 공통이므로
Jaccard에서 제외했다.

### Q4. SSD read와 true TTFT

- Prefix25: {p['ssd_read_mb_mean']:.3f} MB/request,
  {p['ttft_mean_ms']:.2f} ms, normal/separator preads
  {p['normal_kv_preads_mean']:.2f}/{p['separator_preads_mean']:.2f}
- Static+Diverse25: {sd['ssd_read_mb_mean']:.3f} MB/request,
  {sd['ttft_mean_ms']:.2f} ms, normal/separator preads
  {sd['normal_kv_preads_mean']:.2f}/{sd['separator_preads_mean']:.2f}
- SD - Prefix: {(sd['ssd_read_mb_mean']-p['ssd_read_mb_mean']):+.3f} MB,
  {(sd['ttft_mean_ms']-p['ttft_mean_ms']):+.2f} ms

### Q5. 현재 60.8% 성능의 주된 원인

**{cause}**

### Q6. Contribution 판정

**{decision}**

## 통계와 범위

- Primary CI: 이미지 40개를 cluster 단위로 10,000회 paired bootstrap,
  seed={config['bootstrap']['seed']}
- Supplement: 질문 240개 paired bootstrap
- GQA score는 질문별 binary normalized exact match
- McNemar는 질문별 paired binary disagreement의 exact-binomial 결과다.
  이미지당 6개 질문의 상관 때문에 primary uncertainty는 image-cluster CI다.
- Calibration SparseVLM score magnitude는 legacy store에 보존되지 않아
  calibration importance-mass coverage는 이 분석에서 만들지 않았다.
  `static.pt`의 VisionZip saliency를 calibration importance로 대체하지 않았다.

## Provenance

- Source run: `{config['source']['run_dir']}`
- Source results SHA256: `{config['source']['results_sha256']}`
- Store: `{config['store']['path']}`
- Store content SHA256: `{config['store'].get('content_sha256')}`
- Index SHA256: `{config['workload']['index_sha256']}`
- Workload SHA256: `{config['workload']['ordered_workload_sha256']}`
- Separator policy: `sidecar`
- Decision rule: `{DECISION_RULE}`

상세 검증은 `validation.json`, paired 통계는 `paired_stats.json`, 실제 선택
ID는 `selection_trace.jsonl`, overlap 행은 `chunk_overlap.csv`에 있다.
"""


def analyze(run_dir: Path, out_dir: Path, store_dir: Path, *, force=False,
            n_boot=10000, seed=0, expected_images=40,
            expected_questions=240, expected_layers=32,
            expected_store_sha=EXPECTED_STORE_SHA256,
            expected_store_selection_sha=EXPECTED_STORE_SELECTION_SHA256,
            preservation_expected=LEGACY_PRESERVATION,
            prefix_selector_max_mean_ms=1.0,
            prefix_selector_max_p95_ms=2.0) -> dict:
    repo = PROJECT_ROOT.resolve()
    raw_run = Path(os.path.abspath(os.fspath(run_dir)))
    raw_out = Path(os.path.abspath(os.fspath(out_dir)))
    raw_store = Path(os.path.abspath(os.fspath(store_dir)))
    for p in (raw_run, raw_out, raw_store):
        _reject_symlink_components(p, repo)
    run_dir = raw_run.resolve()
    out_dir = raw_out.resolve()
    store_dir = raw_store.resolve()
    run_scope = (repo / "runs" / "reorder_prefix_baseline").resolve()
    out_scope = (repo / "results" / "reorder_prefix_baseline").resolve()
    if not run_dir.is_relative_to(run_scope) or run_dir == run_scope:
        raise AnalysisError(
            f"--run-dir must be a strict child of {run_scope}: {run_dir}")
    if not out_dir.is_relative_to(out_scope) or out_dir == out_scope:
        raise AnalysisError(
            f"--out-dir must be a strict child of {out_scope}: {out_dir}")
    if _paths_overlap(run_dir, out_dir):
        raise AnalysisError(
            "source and output must not be equal or ancestors/descendants")
    # Path isolation is unconditional, even when hash checks are explicitly
    # skipped for a lightweight developer test.
    legacy_paths = [
        (repo / rel).resolve() for rel in LEGACY_PRESERVATION]
    if any(_paths_overlap(store_dir, legacy) for legacy in legacy_paths):
        raise AnalysisError("--store overlaps a protected legacy artifact")
    protected = [store_dir] + legacy_paths
    for candidate_name, candidate in (("source", run_dir), ("output", out_dir)):
        for protected_path in protected:
            if _paths_overlap(candidate, protected_path):
                raise AnalysisError(
                    f"{candidate_name} path overlaps protected store/legacy "
                    f"path: {candidate} vs {protected_path}")

    # Create only the dedicated result namespace, then pin its parent state.
    # We recheck this guard immediately before the atomic rename so --force
    # cannot operate on a target swapped after validation began.
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(out_dir.parent, repo)
    out_parent_identity = _stat_signature(out_dir.parent)[:3]
    out_initial_exists = _lexists(out_dir)
    if out_initial_exists:
        if out_dir.is_symlink() or not out_dir.is_dir():
            raise AnalysisError(f"output target must be a real directory: {out_dir}")
        out_initial_signature = _stat_signature(out_dir)
        if not force:
            raise AnalysisError(
                f"output exists; use --force for atomic refresh: {out_dir}")
    else:
        out_initial_signature = None

    source_results_path = run_dir / "results.json"
    source_csv_path = run_dir / "per_request.csv"
    for p in (source_results_path, source_csv_path):
        if not p.is_file():
            raise AnalysisError(f"missing source artifact: {p}")
        if p.is_symlink():
            raise AnalysisError(f"source artifacts may not be symlinks: {p}")
    source_signatures = {
        source_results_path: _stat_signature(source_results_path),
        source_csv_path: _stat_signature(source_csv_path),
    }
    source_hashes = {
        source_results_path: _sha256_file(source_results_path),
        source_csv_path: _sha256_file(source_csv_path),
    }
    if any(_stat_signature(path) != signature
           for path, signature in source_signatures.items()):
        raise AnalysisError("source artifact changed during initial hashing")

    def verify_source_unchanged() -> None:
        for path in (source_results_path, source_csv_path):
            if (not _lexists(path) or path.is_symlink()
                    or _stat_signature(path) != source_signatures[path]
                    or _sha256_file(path) != source_hashes[path]):
                raise AnalysisError(
                    f"source artifact changed during analysis: {path}")
    with source_results_path.open() as f:
        source_results = json.load(f)
    source_summary = source_results.get("summary", {})
    fields, rows = _read_csv(source_csv_path)

    required_fields = {
        "dataset", "method_key", "retention", "retention_kind", "image_id",
        "question_id", "question",
        "prediction", "ground_truth", "correct", "selector_ms",
        "ssd_read_ms", "prepare_ms", "scatter_ms", "prefill_ms",
        "ttft_ms", "decode_ms", "e2e_latency_ms",
        "ssd_read_bytes", "ssd_read_chunks", "ssd_preads",
        "n_chunks_selected",
        "n_chunks_total", "touched_chunk_fraction",
        "normal_chunk_count_total", "normal_kv_read_bytes",
        "separator_read_bytes", "normal_kv_preads", "separator_preads",
        "total_actual_pread_bytes",
        "static_score_calls", "query_score_calls", "diversity_calls",
        "selection_mode", "selected_chunk_ids_per_layer", "separator_policy",
        "reordered_prefix_store_validated",
    }
    missing_fields = sorted(required_fields - set(fields))
    if missing_fields:
        raise AnalysisError(f"source CSV missing fields: {missing_fields}")

    checks, failures = {}, []

    def check(name: str, passed: bool, detail=None):
        passed = bool(passed)
        checks[name] = {"passed": passed, "detail": detail}
        if not passed:
            failures.append(name if detail is None else f"{name}: {detail}")

    check("source_schema_v2", source_results.get("schema_version") == 2,
          source_results.get("schema_version"))
    request_keys = [(r["method_key"], str(r["image_id"]),
                     str(r["question_id"])) for r in rows]
    duplicate_request_count = len(request_keys) - len(set(request_keys))
    check("no_duplicate_method_requests",
          duplicate_request_count == 0, duplicate_request_count)

    actual_methods = sorted({r["method_key"] for r in rows})
    actual_method_set = set(actual_methods)
    unknown_methods = actual_method_set - set(METHODS)
    missing_required = set(REQUIRED_METHODS) - actual_method_set
    present_methods = [m for m in METHODS if m in actual_method_set]
    check("exact_required_25pct_matrix_with_only_declared_50pct_extras",
          not unknown_methods and not missing_required,
          {"required": REQUIRED_METHODS, "optional": OPTIONAL_METHODS,
           "missing_required": sorted(missing_required),
           "unknown": sorted(unknown_methods), "actual": actual_methods})
    optional_pair = {
        "reorder_prefix_chunk@50", "static_diverse_chunk@50"}
    present_optional_pair = optional_pair & actual_method_set
    any_optional_50 = bool(actual_method_set & set(OPTIONAL_METHODS))
    optional_pair_complete = (not any_optional_50
                              or present_optional_pair == optional_pair)
    check("optional_50pct_prefix_sd_pair_complete", optional_pair_complete,
          {"required_as_pair": sorted(optional_pair),
           "present": sorted(present_optional_pair)})
    if missing_required or unknown_methods or not optional_pair_complete:
        raise AnalysisError(
            "method-matrix validation failed cleanly before paired analysis: "
            f"missing_required={sorted(missing_required)}, "
            f"unknown={sorted(unknown_methods)}, "
            f"optional50_pair_present={sorted(present_optional_pair)}")
    source_cross = _source_results_crosscheck(
        source_results, rows, present_methods)
    check("source_json_csv_consistent", not source_cross,
          source_cross[:20])
    rows_by_method = {m: [r for r in rows if r["method_key"] == m]
                      for m in present_methods}
    check("exact_rows_per_method",
          all(len(rows_by_method[m]) == expected_questions
              for m in present_methods),
          {m: len(rows_by_method[m]) for m in present_methods})
    check("exact_total_rows",
          len(rows) == expected_questions * len(present_methods),
          len(rows))
    method_question_keys = {
        m: {(str(r["image_id"]), str(r["question_id"]))
            for r in rows_by_method[m]} for m in present_methods
    }
    reference_question_keys = method_question_keys[REQUIRED_METHODS[0]]
    aligned_method_keys = all(
        method_question_keys[m] == reference_question_keys
        for m in present_methods)
    check("exact_paired_question_keys_across_methods", aligned_method_keys,
          {m: len(method_question_keys[m]) for m in present_methods})
    if (duplicate_request_count or
            any(len(rows_by_method[m]) != expected_questions
                for m in present_methods)
            or not aligned_method_keys):
        raise AnalysisError(
            "request-matrix validation failed cleanly before paired analysis; "
            "inspect duplicate counts, rows per method, and paired key sets")
    retention_failures = []
    for method in present_methods:
        expected_retention, expected_kind = _retention(method, source_summary)
        for row in rows_by_method[method]:
            observed = _float(row, "retention", allow_blank=True)
            if ((expected_retention is None and observed is not None)
                    or (expected_retention is not None
                        and (observed is None
                             or abs(observed - expected_retention) > 1e-12))
                    or row.get("retention_kind") != expected_kind):
                retention_failures.append(
                    f"{method}/{row['image_id']}/{row['question_id']}")
    check("declared_retention_and_budget_semantics",
          not retention_failures, retention_failures[:20])

    qmeta = defaultdict(set)
    ordered_questions = []
    seen_q = set()
    for r in rows:
        qkey = (str(r["image_id"]), str(r["question_id"]))
        qmeta[qkey].add((r["question"], r["ground_truth"], r["dataset"]))
        if qkey not in seen_q:
            seen_q.add(qkey)
            ordered_questions.append(qkey)
    images = list(dict.fromkeys(i for i, _ in ordered_questions))
    q_count = Counter(i for i, _ in ordered_questions)
    check("exact_40_images", len(images) == expected_images, len(images))
    check("exact_240_questions", len(ordered_questions) == expected_questions,
          len(ordered_questions))
    check("exact_six_questions_per_image",
          all(v == expected_questions // expected_images for v in q_count.values())
          and set(q_count) == set(images), dict(q_count))
    check("identical_question_gold_dataset_across_methods",
          all(len(x) == 1 for x in qmeta.values()),
          sum(len(x) != 1 for x in qmeta.values()))
    check("dataset_is_gqa", {r["dataset"] for r in rows} == {"gqa"},
          sorted({r["dataset"] for r in rows}))

    workload_blob = "\n".join(f"{i}\t{q}" for i, q in ordered_questions).encode()
    workload_sha = hashlib.sha256(workload_blob).hexdigest()
    # The legacy store itself records only the resulting per-layer permutation,
    # not the calibration count.  Preserve the strongest available provenance:
    # the first-four question-ID workload plus the immutable reorder log hash.
    index_value = source_summary.get("index") or "data/index.json"
    index_path = Path(index_value)
    if not index_path.is_absolute():
        index_path = PROJECT_ROOT / index_path
    index_content_sha = (_sha256_file(index_path)
                         if index_path.is_file() else None)
    check("frozen_index_hash",
          source_summary.get("index_sha256") == EXPECTED_INDEX_SHA256
          and index_content_sha == EXPECTED_INDEX_SHA256,
          {"summary": source_summary.get("index_sha256"),
           "recomputed": index_content_sha})
    check("frozen_ordered_workload_hash",
          source_summary.get("workload_sha256") == EXPECTED_WORKLOAD_SHA256
          and workload_sha == EXPECTED_WORKLOAD_SHA256,
          {"summary": source_summary.get("workload_sha256"),
           "recomputed": workload_sha})
    calibration_sha = None
    calibration_failure = None
    index_rows = []
    try:
        with index_path.open() as f:
            index_rows = json.load(f)[:expected_images]
        calibration_blob = "\n".join(
            f"{e['image_id']}\t{q['question_id']}"
            for e in index_rows for q in e["questions"][:4]
        ).encode()
        calibration_sha = hashlib.sha256(calibration_blob).hexdigest()
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        calibration_failure = str(exc)
    check("calibration_first_four_workload_hash",
          calibration_sha == EXPECTED_CALIBRATION_WORKLOAD_SHA256,
          {"expected": EXPECTED_CALIBRATION_WORKLOAD_SHA256,
           "observed": calibration_sha, "error": calibration_failure})

    canonical_eval = {}
    for entry in index_rows:
        image_id = str(entry.get("image_id"))
        for q in entry.get("questions", [])[4:10]:
            answers = q.get("answers")
            if answers is None:
                answers = [q.get("answer")]
            canonical_eval[(image_id, str(q.get("question_id")))] = (
                q.get("question"), answers)
    source_question_failures = []
    for qkey, variants in qmeta.items():
        expected = canonical_eval.get(qkey)
        if expected is None:
            source_question_failures.append(f"missing from index: {qkey}")
            continue
        for question, gold_cell, _dataset in variants:
            try:
                gold = json.loads(gold_cell)
            except (TypeError, json.JSONDecodeError):
                source_question_failures.append(f"invalid gold JSON: {qkey}")
                continue
            if question != expected[0] or gold != expected[1]:
                source_question_failures.append(f"question/gold mismatch: {qkey}")
    check("source_questions_and_gold_match_frozen_index",
          not source_question_failures
          and set(canonical_eval) == set(ordered_questions),
          {"failures": source_question_failures[:20],
           "index_keys": len(canonical_eval),
           "source_keys": len(ordered_questions)})
    expected_budgets = ([0.25, 0.5]
                        if actual_method_set & set(OPTIONAL_METHODS)
                        else [0.25])
    check("source_fixed_conditions",
          source_summary.get("skip") == 4
          and source_summary.get("questions_per_image_requested") == 6
          and [float(x) for x in source_summary.get("budgets", [])]
          == expected_budgets
          and abs(float(source_summary.get("ratio", -1)) - 0.25) < 1e-12
          and source_summary.get("sep_policy") == "sidecar"
          and source_summary.get("cold") is True
          and source_summary.get("metric") == "gqa"
          and int(source_summary.get("max_new_tokens", -1)) == 16,
          {k: source_summary.get(k) for k in
           ("skip", "questions_per_image_requested", "budgets", "ratio",
            "sep_policy", "cold", "metric", "max_new_tokens")})

    meta_by_image, meta_digest_by_image, static_digest_by_image = {}, {}, {}
    store_failures = []
    for image_id in images:
        d = store_dir / image_id
        mp, sp, sep = d / "meta.json", d / "static.pt", d / "sep_kv.bin"
        if not all(p.is_file() for p in (mp, sp, sep)):
            store_failures.append(f"missing store sidecar for {image_id}")
            continue
        with mp.open() as f:
            meta = json.load(f)
        meta_by_image[image_id] = meta
        meta_digest_by_image[image_id] = _sha256_file(mp)
        static_digest_by_image[image_id] = _sha256_file(sp)
        if not (meta.get("reordered") is True and meta.get("order")
                and meta.get("order_is_per_layer") is True):
            store_failures.append(f"not per-layer reordered: {image_id}")
        if (int(meta.get("num_layers", -1)) != expected_layers
                or int(meta.get("chunk_size", -1)) != 64
                or meta.get("dtype") != "float16"
                or meta.get("model") != "llava-hf/llava-v1.6-vicuna-7b-hf"):
            store_failures.append(f"store geometry/model mismatch: {image_id}")
        if len(meta.get("order", [])) != expected_layers:
            store_failures.append(f"per-layer order count mismatch: {image_id}")
        else:
            vn = int(meta.get("v_token_num", -1))
            expected_positions = set(range(vn))
            original_sep = sorted(int(x) for x in meta.get("newline_idx", []))
            for li, order in enumerate(meta["order"]):
                try:
                    order = [int(x) for x in order]
                    stored_sep = _separator_positions(meta, li)
                except (TypeError, ValueError, IndexError):
                    store_failures.append(
                        f"invalid permutation metadata: {image_id}/L{li}")
                    break
                if len(order) != vn or set(order) != expected_positions:
                    store_failures.append(
                        f"invalid permutation: {image_id}/L{li}")
                    break
                if sorted(order[p] for p in stored_sep) != original_sep:
                    store_failures.append(
                        f"separator/permutation mismatch: {image_id}/L{li}")
                    break
        sep_expected = sum(_separator_layer_bytes(meta, li)
                           for li in range(int(meta["num_layers"])))
        if sep.stat().st_size != sep_expected:
            store_failures.append(f"separator size mismatch: {image_id}")
    check("single_valid_reordered_store", not store_failures
          and len(meta_by_image) == expected_images,
          store_failures[:20])

    selection_store_sha, selection_files, selection_bytes = _tree_digest(
        store_dir, include=lambda p: p.name in {"meta.json", "static.pt", "sep_kv.bin"})
    check("store_selection_metadata_hash",
          expected_store_selection_sha is None
          or selection_store_sha == expected_store_selection_sha,
          {"expected": expected_store_selection_sha,
           "observed": selection_store_sha,
           "files": selection_files, "bytes": selection_bytes})
    if expected_store_sha is None:
        content_store_sha = None
        content_files = content_bytes = None
        check("full_store_content_hash", True, "skipped by explicit test option")
    else:
        content_store_sha, content_files, content_bytes = _tree_digest(store_dir)
        check("full_store_content_hash", content_store_sha == expected_store_sha,
              {"expected": expected_store_sha, "observed": content_store_sha,
               "files": content_files, "bytes": content_bytes})

    selection_canonical = {}
    trace_rows = []
    selection_failures, budget_failures, counter_failures = [], [], []
    byte_failures, pread_failures, separator_failures = [], [], []
    expected_modes = {
        "prefix": (0, 0, 0),
        "static": (expected_layers, 0, 0),
        "diverse_only": (0, 0, expected_layers),
        "static_diverse": (expected_layers, 0, expected_layers),
    }

    present_chunk_methods = {
        method: spec for method, spec in CHUNK_METHODS.items()
        if method in actual_method_set
    }
    for method, (mode, budget) in present_chunk_methods.items():
        grouped = defaultdict(list)
        for row in rows_by_method[method]:
            grouped[str(row["image_id"])].append(row)
        for image_id in images:
            meta = meta_by_image.get(image_id)
            if meta is None:
                continue
            image_rows = grouped.get(image_id, [])
            parsed = []
            for row in image_rows:
                cids = _json_cell(row, "selected_chunk_ids_per_layer")
                try:
                    cids = _strict_chunk_layers(
                        cids, context=f"{method}/{image_id}/"
                        f"{row['question_id']}")
                except AnalysisError as exc:
                    selection_failures.append(str(exc))
                    continue
                parsed.append(cids)
            if not parsed:
                continue
            canonical_blob = json.dumps(parsed[0], separators=(",", ":"))
            if any(json.dumps(x, separators=(",", ":")) != canonical_blob
                   for x in parsed[1:]):
                selection_failures.append(
                    f"question-dependent selection {method}/{image_id}")
            cids_per_layer = parsed[0]
            selection_canonical[(method, image_id)] = cids_per_layer
            L = int(meta["num_layers"])
            nc = int(meta["n_chunks_per_layer"])
            k = _budget_chunk_count(nc, budget)
            if len(cids_per_layer) != L:
                budget_failures.append(f"layer count {method}/{image_id}")
                continue

            expected_normal_bytes = 0
            expected_normal_preads = 0
            expected_sep_bytes = sum(_separator_layer_bytes(meta, li)
                                     for li in range(L))
            for li, layer_ids in enumerate(cids_per_layer):
                if (layer_ids != sorted(set(layer_ids))
                        or any(c < 0 or c >= nc for c in layer_ids)
                        or len(layer_ids) != k):
                    budget_failures.append(f"chunk budget/ids {method}/{image_id}/L{li}")
                if mode == "prefix" and layer_ids != list(range(k)):
                    budget_failures.append(f"not first-k {method}/{image_id}/L{li}")
                layer_bytes = _normal_layer_bytes(meta, layer_ids)
                layer_preads = 2 * _contiguous_runs(layer_ids)
                expected_normal_bytes += layer_bytes
                expected_normal_preads += layer_preads
                trace_rows.append({
                    "dataset": "gqa",
                    "image_id": image_id,
                    "layer": li,
                    "method": method,
                    "selection_mode": mode,
                    "budget": budget,
                    "n_chunks_total": nc,
                    "k": k,
                    "selected_normal_chunk_ids": layer_ids,
                    "normal_kv_bytes_layer": layer_bytes,
                    "normal_kv_preads_layer": layer_preads,
                    "separator_kv_bytes_layer": _separator_layer_bytes(meta, li),
                    "separator_kv_bytes_request": expected_sep_bytes,
                    "separator_policy": "sidecar",
                    "meta_sha256": meta_digest_by_image[image_id],
                    "static_sha256": (None if mode == "prefix"
                                      else static_digest_by_image[image_id]),
                })

            exp_calls = expected_modes[mode]
            for row in image_rows:
                if row.get("selection_mode") != mode:
                    counter_failures.append(f"mode {method}/{image_id}")
                if (mode == "prefix"
                        and str(row.get("reordered_prefix_store_validated", ""))
                        .lower() not in {"true", "1"}):
                    counter_failures.append(
                        f"runtime prefix store validation {method}/{image_id}")
                calls = tuple(_int(row, x) for x in
                              ("static_score_calls", "query_score_calls",
                               "diversity_calls"))
                if calls != exp_calls:
                    counter_failures.append(
                        f"selector counters {method}/{image_id}: {calls} != {exp_calls}")
                if row.get("separator_policy") != "sidecar":
                    separator_failures.append(f"sep policy {method}/{image_id}")
                if _int(row, "normal_chunk_count_total") != L * k:
                    counter_failures.append(f"normal chunk total {method}/{image_id}")
                if _int(row, "ssd_read_chunks") != 2 * L * k:
                    counter_failures.append(f"physical K/V chunk units {method}/{image_id}")
                if abs(_float(row, "n_chunks_selected") - k) > 1e-9:
                    budget_failures.append(f"mean chunks {method}/{image_id}")
                if abs(_float(row, "n_chunks_total") - nc) > 1e-9:
                    budget_failures.append(f"total chunks {method}/{image_id}")
                if abs(_float(row, "touched_chunk_fraction") - k / nc) > 1e-9:
                    budget_failures.append(f"touched ratio {method}/{image_id}")
                normal = _int(row, "normal_kv_read_bytes")
                sep_bytes = _int(row, "separator_read_bytes")
                total = _int(row, "total_actual_pread_bytes")
                ssd = _int(row, "ssd_read_bytes")
                normal_preads = _int(row, "normal_kv_preads")
                sep_preads = _int(row, "separator_preads")
                total_preads = _int(row, "ssd_preads")
                if normal != expected_normal_bytes:
                    byte_failures.append(f"normal bytes {method}/{image_id}")
                if sep_bytes != expected_sep_bytes:
                    separator_failures.append(f"sep bytes {method}/{image_id}")
                if total != normal + sep_bytes or ssd != total:
                    byte_failures.append(f"actual pread sum {method}/{image_id}")
                if normal_preads != expected_normal_preads:
                    pread_failures.append(
                        f"contiguous-run preads {method}/{image_id}: "
                        f"{normal_preads} != {expected_normal_preads}")
                if sep_preads != 1:
                    pread_failures.append(
                        f"separator preads {method}/{image_id}: {sep_preads}")
                if total_preads != normal_preads + sep_preads:
                    pread_failures.append(
                        f"total pread split {method}/{image_id}: "
                        f"{total_preads} != {normal_preads}+{sep_preads}")

    check("selection_question_invariant", not selection_failures,
          selection_failures[:20])
    check("exact_chunk_budgets_and_prefix_first_k", not budget_failures,
          budget_failures[:20])
    check("selector_usage_counters", not counter_failures,
          counter_failures[:20])
    check("actual_pread_byte_accounting", not byte_failures,
          byte_failures[:20])
    check("exact_contiguous_run_pread_accounting", not pread_failures,
          pread_failures[:20])
    check("identical_sidecar_policy_and_bytes", not separator_failures,
          separator_failures[:20])
    expected_trace = (expected_images * expected_layers
                      * len(present_chunk_methods))
    check("selection_trace_complete", len(trace_rows) == expected_trace,
          {"expected": expected_trace, "actual": len(trace_rows)})

    # Same budget means the four methods must buy the same normal chunk count.
    same_budget_failures = []
    for image_id in images:
        for budget, suffix in ((0.25, "25"), (0.50, "50")):
            methods = [m for m in (
                f"reorder_prefix_chunk@{suffix}",
                f"visionzip_static_chunk@{suffix}",
                f"diverse_chunk@{suffix}",
                f"static_diverse_chunk@{suffix}")
                if m in actual_method_set]
            if not methods:
                continue
            lengths = []
            for method in methods:
                x = selection_canonical.get((method, image_id))
                lengths.append(None if x is None else [len(v) for v in x])
            if any(x is None for x in lengths) or any(x != lengths[0] for x in lengths[1:]):
                same_budget_failures.append(f"{image_id}@{budget}")
    check("same_normal_chunk_budget_across_selectors", not same_budget_failures,
          same_budget_failures[:20])

    # Prefix has no importance/query/diversity work.  Validate every budget
    # independently so a fast 25% arm cannot hide a broken 50% arm.
    prefix_selector_stats = {}
    for suffix in ("25", "50"):
        method = f"reorder_prefix_chunk@{suffix}"
        if method not in rows_by_method:
            continue
        values = np.asarray([
            _float(r, "selector_ms") for r in rows_by_method[method]])
        stats = {
            "n": int(values.size),
            "min_ms": float(values.min()),
            "mean_ms": float(values.mean()),
            "p95_ms": float(np.percentile(values, 95)),
            "max_allowed_mean_ms": prefix_selector_max_mean_ms,
            "max_allowed_p95_ms": prefix_selector_max_p95_ms,
        }
        prefix_selector_stats[suffix] = stats
        check(f"prefix{suffix}_selector_nonnegative_and_near_zero",
              stats["min_ms"] >= 0
              and stats["mean_ms"] <= prefix_selector_max_mean_ms
              and stats["p95_ms"] <= prefix_selector_max_p95_ms,
              stats)

    # Timing is checked on exact per-request timestamps, not rounded summaries.
    timing_failures = []
    max_residual = 0.0
    for row in rows:
        nums = [_float(row, x) for x in
                ("ttft_ms", "decode_ms", "e2e_latency_ms", "ssd_read_ms",
                 "ssd_read_bytes", "ssd_read_chunks")]
        if any(x < 0 for x in nums):
            timing_failures.append(
                f"negative timing/I/O {row['method_key']}/{row['image_id']}/"
                f"{row['question_id']}")
        ttft, decode, e2e = nums[:3]
        if not ttft < e2e:
            timing_failures.append(
                f"TTFT !< E2E {row['method_key']}/{row['image_id']}/{row['question_id']}")
        max_residual = max(max_residual, abs(e2e - ttft - decode))
    check("true_ttft_decode_e2e_invariants",
          not timing_failures and max_residual <= 0.1,
          {"failures": timing_failures[:20],
           "max_abs_e2e_minus_ttft_minus_decode_ms": max_residual,
           "tolerance_ms": 0.1})
    check("recompute_has_zero_ssd",
          all(_int(r, "ssd_read_bytes") == 0
              and _int(r, "ssd_read_chunks") == 0
              and _int(r, "ssd_preads") == 0
              and _int(r, "normal_kv_read_bytes") == 0
              and _int(r, "separator_read_bytes") == 0
              and _int(r, "normal_kv_preads") == 0
              and _int(r, "separator_preads") == 0
              and _int(r, "total_actual_pread_bytes") == 0
              for r in rows_by_method["recompute"]))
    nonbinary = [
        (r["method_key"], r["image_id"], r["question_id"], r["correct"])
        for r in rows if _float(r, "correct") not in (0.0, 1.0)
    ]
    check("gqa_scores_are_binary", not nonbinary, nonbinary[:20])
    full_byte_failures = []
    for row in rows_by_method["fullload"]:
        meta = meta_by_image.get(str(row["image_id"]))
        if meta is None:
            continue
        expected_bytes = int(meta["bytes_visual_kv"])
        expected_preads = 2 * int(meta["num_layers"])
        if (_int(row, "ssd_read_bytes") != expected_bytes
                or _int(row, "total_actual_pread_bytes") != expected_bytes
                or _int(row, "normal_kv_read_bytes") != expected_bytes
                or _int(row, "separator_read_bytes") != 0
                or _int(row, "ssd_read_chunks") !=
                2 * int(meta["num_layers"]) * int(meta["n_chunks_per_layer"])
                or _int(row, "ssd_preads") != expected_preads
                or _int(row, "normal_kv_preads") != expected_preads
                or _int(row, "separator_preads") != 0):
            full_byte_failures.append(f"{row['image_id']}/{row['question_id']}")
    check("fullload_exact_visual_kv_bytes_and_preads", not full_byte_failures,
          full_byte_failures[:20])

    overlap_rows, overlap_summary = [], {}
    for budget, suffix in ((0.25, "25"), (0.50, "50")):
        p_method = f"reorder_prefix_chunk@{suffix}"
        s_method = f"static_diverse_chunk@{suffix}"
        for image_id in images:
            p_layers = selection_canonical.get((p_method, image_id), [])
            s_layers = selection_canonical.get((s_method, image_id), [])
            meta = meta_by_image.get(image_id)
            if meta is None or len(p_layers) != len(s_layers):
                continue
            nc = int(meta["n_chunks_per_layer"])
            k = _budget_chunk_count(nc, budget)
            for li, (pa, sa) in enumerate(zip(p_layers, s_layers)):
                A, B = set(pa), set(sa)
                inter, union = A & B, A | B
                overlap_rows.append({
                    "image_id": image_id,
                    "layer": li,
                    "budget": budget,
                    "n_chunks_total": nc,
                    "k": k,
                    "intersection_count": len(inter),
                    "union_count": len(union),
                    "sd_only_chunk_count": len(B - A),
                    "prefix_only_chunk_count": len(A - B),
                    "jaccard": len(inter) / len(union) if union else 1.0,
                    "prefix_chunk_ids": json.dumps(pa, separators=(",", ":")),
                    "static_diverse_chunk_ids": json.dumps(sa, separators=(",", ":")),
                })
        br = [r for r in overlap_rows if float(r["budget"]) == budget]
        def mean(field):
            return float(np.mean([float(r[field]) for r in br]))
        overlap_summary[suffix] = {
            "budget": budget,
            "n_image_layers": len(br),
            "jaccard_mean": mean("jaccard") if br else None,
            "jaccard_median": float(np.median([r["jaccard"] for r in br]))
            if br else None,
            "intersection_mean": mean("intersection_count") if br else None,
            "sd_only_mean": mean("sd_only_chunk_count") if br else None,
            "prefix_only_mean": mean("prefix_only_chunk_count") if br else None,
        }
    check("overlap_exact_image_layer_rows",
          overlap_summary.get("25", {}).get("n_image_layers") ==
          expected_images * expected_layers,
          overlap_summary)

    by_key_method = {
        m: {(str(r["image_id"]), str(r["question_id"])): r
            for r in rows_by_method[m]} for m in present_methods
    }
    paired = {}
    for suffix in ("25", "50"):
        a_method = f"static_diverse_chunk@{suffix}"
        b_method = f"reorder_prefix_chunk@{suffix}"
        if a_method not in actual_method_set or b_method not in actual_method_set:
            continue
        keys = ordered_questions
        a = np.asarray([_float(by_key_method[a_method][k], "correct") for k in keys])
        b = np.asarray([_float(by_key_method[b_method][k], "correct") for k in keys])
        boot = _paired_bootstrap(a, b, [k[0] for k in keys], n_boot, seed)
        mc = _exact_mcnemar(a, b)
        lower = boot["image_cluster_primary"]["delta_ci95"][0]
        decision = "GO" if boot["delta_mean"] > 0 and lower > 0 else "RETHINK"
        paired[f"static_diverse{suffix}_minus_prefix{suffix}"] = {
            "a_method": a_method,
            "b_method": b_method,
            "delta_pp": boot["delta_mean"] * 100,
            "bootstrap": boot,
            "mcnemar": mc,
            "decision": decision,
            "decision_rule": DECISION_RULE,
            "primary_decision": suffix == "25",
        }
    paired["overlap"] = overlap_summary

    legacy_observed, legacy_failures = _legacy_hashes(preservation_expected)
    check("legacy_results_preserved", not legacy_failures,
          legacy_failures)

    summary_rows = _summaries(rows_by_method, present_methods)
    summary_cross_failures = []
    summary_key_pairs = (
        ("accuracy", "acc"),
        ("ttft_mean_ms", "ttft_mean_ms"),
        ("ssd_read_bytes_mean", "ssd_read_bytes_mean"),
        ("ssd_read_chunks_mean", "ssd_read_chunks_mean"),
        ("ssd_preads_mean", "preads"),
        ("ssd_read_mean_ms", "ssd_read_ms"),
        ("normal_kv_read_bytes_mean", "normal_kv_read_bytes"),
        ("separator_read_bytes_mean", "separator_read_bytes"),
        ("total_actual_pread_bytes_mean", "total_actual_pread_bytes"),
        ("normal_kv_preads_mean", "normal_kv_preads"),
        ("separator_preads_mean", "separator_preads"),
        ("static_score_calls_mean", "static_score_calls"),
        ("query_score_calls_mean", "query_score_calls"),
        ("diversity_calls_mean", "diversity_calls"),
    )
    source_per_method = source_summary.get("per_method", {})
    for row in summary_rows:
        method = row["method_key"]
        source_method = source_per_method.get(method, {})
        for output_key, source_key in summary_key_pairs:
            observed = row[output_key]
            expected = source_method.get(source_key)
            if observed == "" and expected is None:
                continue
            if (observed == "" or expected is None
                    or not math.isclose(float(observed), float(expected),
                                        rel_tol=1e-12, abs_tol=1e-9)):
                summary_cross_failures.append(
                    f"{method}/{output_key}: {observed!r} != {expected!r}")
    check("recomputed_summary_preserves_byte_pread_counter_splits",
          not summary_cross_failures, summary_cross_failures[:20])

    validation = {
        "schema_version": 1,
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "counts": {
            "images": len(images),
            "questions": len(ordered_questions),
            "methods": len(actual_methods),
            "per_request_rows": len(rows),
            "selection_trace_rows": len(trace_rows),
            "chunk_overlap_rows": len(overlap_rows),
        },
        "decision_reportable": not failures,
    }
    config = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "analysis": "reorder_prefix_baseline",
        "source": {
            "run_dir": str(run_dir),
            "results": str(source_results_path),
            "results_sha256": source_hashes[source_results_path],
            "per_request_sha256": source_hashes[source_csv_path],
            "source_command": source_summary.get("command"),
            "schema_version": source_results.get("schema_version"),
        },
        "workload": {
            "dataset": "gqa",
            "images": expected_images,
            "questions": expected_questions,
            "questions_per_image": expected_questions // expected_images,
            "index": source_summary.get("index"),
            "index_sha256": source_summary.get("index_sha256"),
            "recomputed_index_sha256": index_content_sha,
            "ordered_workload_sha256": workload_sha,
            "calibration_questions_per_image": 4,
            "calibration_workload_sha256": calibration_sha,
            "expected_calibration_workload_sha256":
                EXPECTED_CALIBRATION_WORKLOAD_SHA256,
            "skip": 4,
        },
        "methods": present_methods,
        "required_methods": REQUIRED_METHODS,
        "optional_methods": OPTIONAL_METHODS,
        "bootstrap": {
            "primary": "image-cluster paired bootstrap",
            "supplement": "question-level paired bootstrap",
            "n_resamples": n_boot,
            "seed": seed,
            "ci": 0.95,
        },
        "decision_rule": DECISION_RULE,
        "prefix_selector_max_mean_ms": prefix_selector_max_mean_ms,
        "prefix_selector_max_p95_ms": prefix_selector_max_p95_ms,
        "prefix_selector_by_budget": prefix_selector_stats,
        "path_policy": {
            "source_scope": str(run_scope),
            "output_scope": str(out_scope),
            "strict_child_required": True,
            "symlinks_allowed": False,
            "ancestor_descendant_overlap_allowed": False,
            "publication": "renameat2(RENAME_NOREPLACE)",
        },
        "store": {
            "path": str(store_dir),
            "content_sha256": content_store_sha,
            "content_n_files": content_files,
            "content_n_bytes": content_bytes,
            "selection_metadata_sha256": selection_store_sha,
            "selection_metadata_n_files": selection_files,
            "selection_metadata_n_bytes": selection_bytes,
            "expected_content_sha256": expected_store_sha,
            "expected_selection_metadata_sha256": expected_store_selection_sha,
        },
        "legacy_preservation": legacy_observed,
        "latency_definition": source_summary.get("latency_definition"),
        "source_fixed_conditions": {
            k: source_summary.get(k) for k in
            ("ratio", "budgets", "diverse_frac", "sep_policy", "cold",
             "max_new_tokens", "alpha", "probe_heads", "lam_static",
             "lam_query")
        },
        "code_provenance_sha256": {
            rel: _sha256_file(PROJECT_ROOT / rel)
            for rel in (
                "scripts/04_eval.py", "mmimpress/serve.py",
                "mmimpress/cvpr25.py", "mmimpress/model.py",
                "mmimpress/dataset.py")
        },
    }

    # Fail closed: an invalid analysis never replaces an existing report and
    # never publishes a directory that could be mistaken for reportable data.
    if not validation["passed"]:
        raise AnalysisError(
            "validation failed; no analysis artifacts were published: "
            f"{failures[:5]}")

    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp-",
                                    dir=out_dir.parent))
    backup = None
    backup_moved = False
    parent_fd = None
    try:
        verify_source_unchanged()
        _write_json(staging / "config.json", config)
        # Preserve every raw field exactly; the source and destination hashes
        # are therefore directly comparable.
        shutil.copyfile(source_csv_path, staging / "per_request.csv")
        if _sha256_file(staging / "per_request.csv") != source_hashes[source_csv_path]:
            raise AnalysisError("staged per_request.csv is not an exact source copy")
        _write_csv(staging / "summary.csv", SUMMARY_FIELDS, summary_rows)
        _write_json(staging / "validation.json", validation)
        _write_json(staging / "paired_stats.json", paired)
        with (staging / "selection_trace.jsonl").open("w") as f:
            for row in trace_rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        _write_csv(staging / "chunk_overlap.csv", OVERLAP_FIELDS, overlap_rows)
        (staging / "README.md").write_text(
            _build_readme(summary_rows, paired, overlap_summary,
                          validation, config))

        required = {"config.json", "per_request.csv", "summary.csv",
                    "validation.json", "README.md", "selection_trace.jsonl",
                    "chunk_overlap.csv", "paired_stats.json"}
        missing = sorted(x for x in required if not (staging / x).is_file())
        if missing:
            raise AnalysisError(f"staging artifacts missing: {missing}")

        # Pin both source artifacts and destination parent immediately before
        # publication.  All renames below are relative to the open directory
        # descriptor, so even a later pathname swap cannot redirect writes.
        verify_source_unchanged()
        if (_stat_signature(out_dir.parent)[:3] != out_parent_identity
                or out_dir.parent.is_symlink()):
            raise AnalysisError("output parent changed during analysis")
        open_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        open_flags |= getattr(os, "O_NOFOLLOW", 0)
        parent_fd = os.open(out_dir.parent, open_flags)
        if _stat_result_signature(os.fstat(parent_fd))[:3] != out_parent_identity:
            raise AnalysisError("opened output parent is not the guarded directory")

        def stat_at(name: str):
            try:
                return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return None

        current = stat_at(out_dir.name)
        if out_initial_exists:
            if current is None or _stat_result_signature(current) != out_initial_signature:
                raise AnalysisError("force target changed during analysis")
            backup = out_dir.with_name(
                f"{out_dir.name}.previous-{time.strftime('%Y%m%d-%H%M%S')}-"
                f"{os.getpid()}-{time.time_ns()}")
            if not backup.is_relative_to(out_scope) or backup.parent != out_dir.parent:
                raise AnalysisError("refresh backup escaped the dedicated output scope")
            _rename_noreplace(parent_fd, out_dir.name,
                              parent_fd, backup.name)
            backup_moved = True
            moved = stat_at(backup.name)
            if (moved is None or
                    _stat_result_signature(moved) != out_initial_signature):
                # The original target was raced.  Restore without overwriting
                # anything and abort rather than publishing over uncertain data.
                if stat_at(out_dir.name) is None:
                    _rename_noreplace(parent_fd, backup.name,
                                      parent_fd, out_dir.name)
                    backup_moved = False
                raise AnalysisError("force target identity changed during rename")
        elif current is not None:
            raise AnalysisError("output target appeared during analysis")

        _rename_noreplace(parent_fd, staging.name, parent_fd, out_dir.name)
    except Exception:
        # Restore the prior result only into an empty target; never overwrite a
        # concurrently-created path.  If restoration is impossible, the old
        # result remains intact in the uniquely named `.previous-*` directory.
        if parent_fd is not None and backup_moved:
            try:
                target_now = os.stat(
                    out_dir.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                target_now = None
            if target_now is None:
                try:
                    _rename_noreplace(parent_fd, backup.name,
                                      parent_fd, out_dir.name)
                    backup_moved = False
                except Exception:
                    pass
        # Only remove the private staging directory when its parent still has
        # the identity captured at entry; otherwise leave it untouched.
        parent_still_guarded = False
        try:
            parent_still_guarded = (
                _stat_signature(out_dir.parent)[:3] == out_parent_identity)
        except FileNotFoundError:
            pass
        if (parent_still_guarded and _lexists(staging)
                and staging.is_dir() and not staging.is_symlink()):
            shutil.rmtree(staging)
        raise
    finally:
        if parent_fd is not None:
            os.close(parent_fd)

    return {
        "out_dir": str(out_dir),
        "decision": paired["static_diverse25_minus_prefix25"]["decision"],
        "delta_pp": paired["static_diverse25_minus_prefix25"]["delta_pp"],
        "validation_passed": True,
    }


def _self_test() -> None:
    a = np.asarray([1, 1, 0, 1, 0, 1, 1, 1], dtype=float)
    b = np.asarray([1, 0, 0, 0, 0, 1, 0, 1], dtype=float)
    images = ["i0"] * 2 + ["i1"] * 2 + ["i2"] * 2 + ["i3"] * 2
    first = _paired_bootstrap(a, b, images, 1000, 7)
    second = _paired_bootstrap(a, b, images, 1000, 7)
    assert first == second
    assert abs(first["delta_mean"] - 0.375) < 1e-12
    mc = _exact_mcnemar(a, b)
    assert mc["a_only"] == 3 and mc["b_only"] == 0
    assert _budget_chunk_count(34, 0.25) == 8  # ties-to-even
    assert _budget_chunk_count(36, 0.25) == 9
    A, B = set(range(3)), {0, 2, 4}
    assert len(A & B) / len(A | B) == 0.5
    assert _contiguous_runs([0, 1, 2, 5, 8, 9]) == 3
    assert _strict_chunk_layers([[0, 2], [1]], context="self-test") == [
        [0, 2], [1]]
    for invalid in ([[0.0]], [["0"]], [[True]], [0]):
        try:
            _strict_chunk_layers(invalid, context="self-test-invalid")
        except AnalysisError:
            pass
        else:
            raise AssertionError(f"accepted non-integer chunk IDs: {invalid!r}")
    assert _paths_overlap(Path("/x/y"), Path("/x/y/z"))
    assert not _paths_overlap(Path("/x/y"), Path("/x/z"))

    # Exercise the exact no-overwrite primitive used by --force publication.
    with tempfile.TemporaryDirectory(prefix="reorder-prefix-selftest-") as td:
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
                raise AssertionError("RENAME_NOREPLACE overwrote existing data")
            assert (root / "source").read_text() == "new"
            assert (root / "existing").read_text() == "old"
            _rename_noreplace(fd, "source", fd, "published")
            assert not (root / "source").exists()
            assert (root / "published").read_text() == "new"
        finally:
            os.close(fd)
    print("self-test PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--store", default="kvstore")
    ap.add_argument("--bootstrap-resamples", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    ap.add_argument("--prefix-selector-max-mean-ms", type=float, default=1.0)
    ap.add_argument("--prefix-selector-max-p95-ms", type=float, default=2.0)
    ap.add_argument("--force", action="store_true",
                    help="atomically refresh only the requested output; the old "
                         "output is moved to a timestamped .previous directory")
    ap.add_argument("--skip-full-store-hash", action="store_true",
                    help="test/debug only; production analysis must hash the store")
    ap.add_argument("--skip-preservation-check", action="store_true",
                    help="test/debug only; production analysis must check legacy hashes")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.run_dir or not args.out_dir:
        ap.error("--run-dir and --out-dir are required unless --self-test is used")
    result = analyze(
        PROJECT_ROOT / args.run_dir if not Path(args.run_dir).is_absolute()
        else Path(args.run_dir),
        PROJECT_ROOT / args.out_dir if not Path(args.out_dir).is_absolute()
        else Path(args.out_dir),
        PROJECT_ROOT / args.store if not Path(args.store).is_absolute()
        else Path(args.store),
        force=args.force,
        n_boot=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
        expected_store_sha=(None if args.skip_full_store_hash
                            else EXPECTED_STORE_SHA256),
        preservation_expected=({} if args.skip_preservation_check
                               else LEGACY_PRESERVATION),
        prefix_selector_max_mean_ms=args.prefix_selector_max_mean_ms,
        prefix_selector_max_p95_ms=args.prefix_selector_max_p95_ms,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

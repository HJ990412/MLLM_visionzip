#!/usr/bin/env python3
"""Publish the final Korean QA-Select25 analysis from a validated GQA run.

This is deliberately a reporting-only step.  It reads the immutable run
evidence, verifies the experiment and protected-artifact contracts, and then
atomically creates ``ANALYSIS.md`` in the dedicated results directory.  It
never edits the run directory and refuses to replace an existing report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "qa-select-gqa-pilot-v2"
PROTECTION_SCHEMA_VERSION = "query-aware-protected-artifacts-v1"
EXPECTED_INDEX_SHA256 = (
    "514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a"
)
EXPECTED_WORKLOAD_SHA256 = (
    "97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66"
)
METHOD_KEYS = ("recompute", "fullload", "qa_select25", "ours25")
METHOD_IDS = {
    "recompute": "recompute",
    "fullload": "fullload",
    "qa_select25": "qa_select25",
    "ours25": "imageonly_prefix25",
}
DISPLAY_LABELS = {
    "recompute": "ReComp",
    "fullload": "FullLoad",
    "qa_select25": "QA-Select25",
    "ours25": "Ours25",
}
RUN_FILES = (
    "config.json",
    "raw.jsonl",
    "persistence.jsonl",
    "per_request.csv",
    "summary.json",
    "validation.json",
    "selection.json",
    "summary.csv",
    "README.md",
)
SOURCE_FILES = (
    "mmimpress/sparsevlm.py",
    "mmimpress/serve.py",
    "mmimpress/store.py",
    "mmimpress/piggyback.py",
    "mmimpress/reorder.py",
    "scripts/49_eval_query_aware_baseline.py",
    "scripts/50_protect_query_aware_artifacts.py",
    "scripts/51_report_query_aware_baseline.py",
)


class ReportValidationError(RuntimeError):
    """The supplied run/result evidence is incomplete or inconsistent."""


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _regular_file(path: Path) -> os.stat_result:
    try:
        value = path.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise ReportValidationError(f"missing required file: {path}") from error
    if not stat.S_ISREG(value.st_mode):
        raise ReportValidationError(f"required path is not a regular file: {path}")
    return value


def _sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    """Hash a regular file without following a final-component symlink."""
    before = _regular_file(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if ((opened.st_dev, opened.st_ino, opened.st_mode)
                != (before.st_dev, before.st_ino, before.st_mode)):
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
    stable = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in stable):
        raise ReportValidationError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    digest = _sha256_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReportValidationError(f"cannot parse JSON: {path}") from error
    if not isinstance(value, dict):
        raise ReportValidationError(f"JSON root is not an object: {path}")
    # A second hash closes the read-after-hash race without retaining the
    # potentially large selection payload as raw bytes.
    if _sha256_file(path) != digest:
        raise ReportValidationError(f"file changed while reading: {path}")
    return value, digest


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportValidationError(f"{label} is not a mapping")
    return value


def _finite(value: Any, label: str, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool):
        raise ReportValidationError(f"{label} is not numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ReportValidationError(f"{label} is not numeric") from error
    if not math.isfinite(number):
        raise ReportValidationError(f"{label} is not finite")
    return number


def _same_float(actual: Any, expected: float, label: str) -> None:
    value = _finite(actual, label)
    assert value is not None
    if not math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ReportValidationError(
            f"{label} mismatch: observed {value}, independently computed {expected}")


def _selection_signature(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "scope", "n_query_requests", "n_images", "n_pairs",
        "mean_pairwise_token_jaccard", "mean_pairwise_chunk_jaccard",
        "identical_selection_rate", "different_selection_pairs",
        "n_consecutive_pairs", "mean_consecutive_token_jaccard",
        "mean_consecutive_chunk_jaccard",
    )
    return {key: value.get(key) for key in keys}


def _validate_experiment(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ReportValidationError(f"run directory is not a real directory: {run_dir}")
    for name in RUN_FILES:
        _regular_file(run_dir / name)

    config, _ = _read_json(run_dir / "config.json")
    summary, _ = _read_json(run_dir / "summary.json")
    validation, _ = _read_json(run_dir / "validation.json")

    for label, value in (("config", config), ("summary", summary),
                         ("validation", validation)):
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ReportValidationError(f"{label} schema mismatch")
    if summary.get("config") != config:
        raise ReportValidationError("summary/config.json mismatch")
    if config.get("status") != "complete":
        raise ReportValidationError("experiment status is not complete")
    configured_run = config.get("run_dir")
    if not isinstance(configured_run, str) or Path(configured_run).resolve() != run_dir:
        raise ReportValidationError("config run_dir does not identify this run bundle")
    if config.get("dataset") != "gqa":
        raise ReportValidationError("report only accepts the GQA pilot")
    integer_contract = {
        "n_images": 40,
        "n_questions": 240,
        "selected_images": 40,
        "selected_questions": 240,
        "full_images": 40,
        "full_questions": 240,
        "skip": 4,
        "questions_per_image": 6,
    }
    for key, expected in integer_contract.items():
        if config.get(key) != expected:
            raise ReportValidationError(
                f"full-pilot contract mismatch for {key}: {config.get(key)!r}")
    if config.get("index_sha256") != EXPECTED_INDEX_SHA256:
        raise ReportValidationError("frozen GQA index hash mismatch")
    if config.get("full_workload_sha256") != EXPECTED_WORKLOAD_SHA256:
        raise ReportValidationError("frozen full workload hash mismatch")
    if config.get("selected_workload_sha256") != EXPECTED_WORKLOAD_SHA256:
        raise ReportValidationError("selected workload is not the full frozen 40/240 slice")
    if config.get("future_question_leakage") != 0:
        raise ReportValidationError("future-question leakage contract is nonzero")
    if tuple(config.get("method_keys", ())) != METHOD_KEYS:
        raise ReportValidationError("config method set/order mismatch")

    config_methods = _mapping(config.get("methods"), "config.methods")
    methods = _mapping(summary.get("per_method"), "summary.per_method")
    if set(config_methods) != set(METHOD_KEYS) or set(methods) != set(METHOD_KEYS):
        raise ReportValidationError("exactly four expected methods are required")
    numeric_fields = (
        "accuracy_all_turns", "accuracy_cache_hits",
        "ttft_cache_hit_mean_ms", "ttft_cache_hit_p50_ms",
        "ttft_cache_hit_p95_ms", "actual_ssd_mb_per_cache_hit",
        "actual_ssd_ratio_vs_fullload", "selector_ms",
        "online_selector_total_ms", "ssd_preads_per_cache_hit",
        "ssd_read_latency_ms", "probe_io_mb",
    )
    nullable_fields = (
        "touched_chunk_fraction", "contiguous_runs_per_layer",
        "mean_contiguous_run_length", "logical_selected_token_ratio",
    )
    for key in METHOD_KEYS:
        configured = _mapping(config_methods[key], f"config.methods.{key}")
        measured = _mapping(methods[key], f"summary.per_method.{key}")
        for source_name, source in (("config", configured), ("summary", measured)):
            if source.get("method_id") != METHOD_IDS[key]:
                raise ReportValidationError(
                    f"{source_name} method_id mismatch for {key}")
            if source.get("display_label") != DISPLAY_LABELS[key]:
                raise ReportValidationError(
                    f"{source_name} display label mismatch for {key}")
        for field in numeric_fields:
            _finite(measured.get(field), f"{key}.{field}")
        for field in nullable_fields:
            _finite(measured.get(field), f"{key}.{field}", nullable=True)
    if methods["qa_select25"].get("paper_label") != "Query-Aware":
        raise ReportValidationError("QA paper label mismatch")
    if methods["ours25"].get("paper_label") != "Ours":
        raise ReportValidationError("Ours paper label mismatch")
    _same_float(methods["qa_select25"].get("retention_ratio"), 0.25,
                "QA nominal retention")
    _same_float(methods["ours25"].get("retention_ratio"), 0.25,
                "Ours nominal retention")
    _same_float(methods["fullload"].get("retention_ratio"), 1.0,
                "FullLoad retention")

    if validation.get("passed") is not True:
        raise ReportValidationError("experiment validation did not pass")
    checks = _mapping(validation.get("checks"), "validation.checks")
    if not checks or any(value is not True for value in checks.values()):
        failed = [key for key, value in checks.items() if value is not True]
        raise ReportValidationError(
            "one or more validation checks failed: " + ", ".join(failed))

    selection = _mapping(summary.get("selection"), "summary.selection")
    validation_selection = _mapping(
        validation.get("selection"), "validation.selection")
    if _selection_signature(selection) != _selection_signature(validation_selection):
        raise ReportValidationError("summary/validation selection aggregates mismatch")
    if selection.get("n_query_requests") != 200:
        raise ReportValidationError("expected 200 cache-hit QA selection requests")
    if selection.get("n_images") != 40 or selection.get("n_pairs") != 400:
        raise ReportValidationError("expected 40 images and 400 within-image pairs")
    requests = selection.get("requests")
    pairs = selection.get("pairs")
    if not isinstance(requests, Sequence) or isinstance(requests, (str, bytes)):
        raise ReportValidationError("selection requests are missing")
    if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)):
        raise ReportValidationError("selection pairs are missing")
    if len(requests) != 200 or len(pairs) != 400:
        raise ReportValidationError("selection evidence count mismatch")
    if not isinstance(selection.get("different_selection_pairs"), int):
        raise ReportValidationError("different-selection count is malformed")
    if selection["different_selection_pairs"] <= 0:
        raise ReportValidationError("query-dependent selection was not observed")
    for field in (
        "mean_pairwise_token_jaccard", "mean_pairwise_chunk_jaccard",
        "identical_selection_rate", "mean_consecutive_token_jaccard",
        "mean_consecutive_chunk_jaccard",
    ):
        value = _finite(selection.get(field), f"selection.{field}")
        assert value is not None
        if not 0.0 <= value <= 1.0:
            raise ReportValidationError(f"selection.{field} is outside [0,1]")

    comparison = _mapping(summary.get("comparison"), "summary.comparison")
    qa = methods["qa_select25"]
    ours = methods["ours25"]
    expected_comparison = {
        "qa_minus_ours_accuracy_all_turns_pp": (
            float(qa["accuracy_all_turns"]) - float(ours["accuracy_all_turns"])) * 100,
        "qa_minus_ours_accuracy_cache_hits_pp": (
            float(qa["accuracy_cache_hits"]) - float(ours["accuracy_cache_hits"])) * 100,
        "qa_minus_ours_ttft_cache_hit_ms": (
            float(qa["ttft_cache_hit_mean_ms"])
            - float(ours["ttft_cache_hit_mean_ms"])),
        "qa_over_ours_ttft_ratio": (
            float(qa["ttft_cache_hit_mean_ms"])
            / float(ours["ttft_cache_hit_mean_ms"])),
    }
    for key, expected in expected_comparison.items():
        _same_float(comparison.get(key), expected, f"comparison.{key}")

    # The line-oriented files are cheap independent completeness checks and do
    # not require materialising their large per-layer arrays again.
    with (run_dir / "raw.jsonl").open("r", encoding="utf-8") as handle:
        if sum(1 for line in handle if line.strip()) != 960:
            raise ReportValidationError("raw.jsonl does not contain 960 rows")
    with (run_dir / "persistence.jsonl").open("r", encoding="utf-8") as handle:
        if sum(1 for line in handle if line.strip()) != 40:
            raise ReportValidationError("persistence.jsonl does not contain 40 rows")

    hashes = {name: _sha256_file(run_dir / name) for name in RUN_FILES}
    return dict(config), dict(summary), dict(validation), hashes


def _find_and_validate_protection(
    run_dir: Path, results_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    manifest_path = run_dir.parent / "protected_artifacts_before.json"
    manifest, manifest_file_sha = _read_json(manifest_path)
    exclusions = manifest.get("excluded_new_roots")
    if (not isinstance(exclusions, list) or len(exclusions) != 2
            or not all(isinstance(item, str) for item in exclusions)):
        raise ReportValidationError(
            "protected-artifact exclusions are malformed")
    protected_run_root = Path(exclusions[0]).resolve()
    protected_results_root = Path(exclusions[1]).resolve()
    if protected_run_root != run_dir.parent:
        raise ReportValidationError(
            "protected-artifact run exclusion does not match the run root")
    if (results_dir != protected_results_root
            and protected_results_root not in results_dir.parents):
        raise ReportValidationError(
            "result bundle is outside the protected new-results root")
    validation_path = protected_results_root / "protected_artifacts_validation.json"
    report, report_file_sha = _read_json(validation_path)
    if manifest.get("schema_version") != PROTECTION_SCHEMA_VERSION:
        raise ReportValidationError("protected-artifact manifest schema mismatch")
    if report.get("schema_version") != PROTECTION_SCHEMA_VERSION:
        raise ReportValidationError("protected-artifact validation schema mismatch")
    entries = manifest.get("entries")
    if not isinstance(entries, dict):
        raise ReportValidationError("protected-artifact manifest has no entries")
    if _canonical_hash(entries) != manifest.get("manifest_sha256"):
        raise ReportValidationError("protected-artifact manifest was tampered with")
    if report.get("passed") is not True:
        raise ReportValidationError("protected-artifact verification did not pass")
    if report.get("kvstore_trees_hashed") is not False:
        raise ReportValidationError("unexpected protected-artifact scope")
    for key in ("missing_paths", "added_paths", "changed_paths"):
        if report.get(key) != []:
            raise ReportValidationError(f"protected-artifact report has {key}")
    expected_hash = manifest.get("manifest_sha256")
    if (report.get("before_manifest_sha256") != expected_hash
            or report.get("after_manifest_sha256") != expected_hash):
        raise ReportValidationError("protected-artifact before/after hash mismatch")
    for stem in ("entry_count", "file_count", "directory_count",
                 "symlink_count", "total_bytes"):
        if (report.get(f"{stem}_before") != manifest.get(stem)
                or report.get(f"{stem}_after") != manifest.get(stem)):
            raise ReportValidationError(
                f"protected-artifact {stem} mismatch")
    return manifest, report, manifest_file_sha, report_file_sha


def _verify_compact_bundle(
    run_dir: Path, results_dir: Path, run_hashes: Mapping[str, str],
) -> dict[str, str]:
    """Verify evaluator-published copies when present, without requiring them."""
    copied: dict[str, str] = {}
    for name in ("summary.json", "validation.json", "selection.json",
                 "summary.csv", "README.md"):
        path = results_dir / name
        if os.path.lexists(path):
            digest = _sha256_file(path)
            copied[name] = digest
            if digest != run_hashes[name]:
                raise ReportValidationError(
                    f"compact result differs from run evidence: {name}")
    artifacts_path = results_dir / "run_artifacts.json"
    if os.path.lexists(artifacts_path):
        artifacts, digest = _read_json(artifacts_path)
        copied["run_artifacts.json"] = digest
        if artifacts.get("schema_version") != SCHEMA_VERSION:
            raise ReportValidationError("run_artifacts schema mismatch")
        if Path(str(artifacts.get("run_dir"))).resolve() != run_dir:
            raise ReportValidationError("run_artifacts run_dir mismatch")
        recorded = _mapping(
            artifacts.get("files_sha256"), "run_artifacts.files_sha256")
        if dict(recorded) != dict(run_hashes):
            raise ReportValidationError("run_artifacts file hashes mismatch")
    return copied


def _atomic_text_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ReportValidationError(f"output parent is not a real directory: {path.parent}")
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to replace existing report: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard-link publication is atomic and, unlike replace(), preserves
        # the no-clobber contract even if another process wins the race.
        os.link(temporary, path)
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


def _pct(value: Any, digits: int = 4) -> str:
    number = _finite(value, "percentage", nullable=True)
    return "—" if number is None else f"{100.0 * number:.{digits}f}%"


def _number(value: Any, digits: int = 4) -> str:
    number = _finite(value, "number", nullable=True)
    return "—" if number is None else f"{number:.{digits}f}"


def _signed(value: Any, digits: int = 4) -> str:
    number = _finite(value, "signed number")
    assert number is not None
    return f"{number:+.{digits}f}"


def _reference_lines(reference: Mapping[str, Any]) -> list[str]:
    if not reference.get("available"):
        return ["- 고정된 이전 reference가 없어 수치 일관성 비교는 수행되지 않았다."]
    methods = _mapping(reference.get("methods"), "reference.methods")
    lines = []
    for key in ("recompute", "fullload", "ours25"):
        item = _mapping(methods.get(key), f"reference.methods.{key}")
        lines.append(
            f"- {DISPLAY_LABELS[key]}: prediction agreement "
            f"{item.get('equal_predictions')}/{item.get('compared')}, "
            f"accuracy gap {_signed(item.get('accuracy_gap_pp'))} pp")
    return lines


def _conclusion(summary: Mapping[str, Any]) -> str:
    comparison = _mapping(summary["comparison"], "comparison")
    quality = float(comparison["qa_minus_ours_accuracy_all_turns_pp"])
    ratio = float(comparison["qa_over_ours_ttft_ratio"])
    if abs(quality) < 1e-12:
        quality_text = "동일한 all-turn accuracy"
    elif quality > 0:
        quality_text = f"QA-Select25가 {_signed(quality)} pp 높은 all-turn accuracy"
    else:
        quality_text = f"QA-Select25가 {_signed(quality)} pp 낮은 all-turn accuracy"
    return (
        f"이 40-image/240-question GQA pilot에서는 {quality_text}를 보였고, "
        f"cache-hit TTFT는 Ours25 대비 {ratio:.4f}배였다. 따라서 이 pilot의 "
        "관측 범위에서는 per-query adaptive selection의 품질 효과와 Ours의 "
        "selector/locality 이점을 함께 보고해야 하며, 대화형 multi-turn 전체 "
        "데이터셋으로 일반화해서는 안 된다."
    )


def _build_markdown(
    run_dir: Path,
    results_dir: Path,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    validation: Mapping[str, Any],
    run_hashes: Mapping[str, str],
    copied_hashes: Mapping[str, str],
    protection_manifest: Mapping[str, Any],
    protection_report: Mapping[str, Any],
    manifest_file_sha: str,
    protection_file_sha: str,
    test_result: str | None,
) -> str:
    methods = _mapping(summary["per_method"], "per_method")
    comparison = _mapping(summary["comparison"], "comparison")
    selection = _mapping(summary["selection"], "selection")
    reference = _mapping(summary.get("reference_consistency", {}),
                         "reference_consistency")
    checks = _mapping(validation["checks"], "validation.checks")
    limitations = validation.get("limitations")
    if not isinstance(limitations, list) or not all(
            isinstance(item, str) for item in limitations):
        raise ReportValidationError("validation limitations are malformed")
    qa = methods["qa_select25"]
    ours = methods["ours25"]

    lines = [
        "# QA-Select25 GQA pilot 최종 분석", "",
        ("**판정:** SparseVLM-based query-aware SSD baseline과 image-only "
         "repacked Prefix25를 동일한 nominal 25% budget에서 비교한 고정 "
         "GQA 40-image/240-question pilot가 모든 검증을 통과했다."), "",
        _conclusion(summary), "",
        "## Main result", "",
        ("Accuracy는 전체 240개 질문 기준이며, TTFT와 SSD/I/O 열은 "
         "cache-hit turns 2–6의 request mean이다. MB는 decimal MB(10^6 bytes)다."),
        "", "| Method | Accuracy | TTFT | Nominal KV | Actual SSD MB | SSD Ratio | Selector ms | Touched Chunks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in METHOD_KEYS:
        row = methods[key]
        lines.append(
            f"| {row['display_label']} | {_pct(row['accuracy_all_turns'])} | "
            f"{_number(row['ttft_cache_hit_mean_ms'])} ms | "
            f"{_pct(row['retention_ratio'])} | "
            f"{_number(row['actual_ssd_mb_per_cache_hit'])} | "
            f"{_pct(row['actual_ssd_ratio_vs_fullload'])} | "
            f"{_number(row['selector_ms'])} | "
            f"{_pct(row['touched_chunk_fraction'])} |")

    lines.extend([
        "", "## 요청된 1–20 항목", "",
        "### 1. QA-Select exact algorithm", "",
        ("Turn 1은 네 arm 모두 동일한 normal pixel inference다. QA arm은 그 "
         "answer-producing forward에서 canonical Visual-KV, decoder visual "
         "hidden state, probe K를 piggyback 저장한다. 각 cache-hit request에서는 "
         "현재 질문 suffix와 저장된 visual hidden으로 text raters를 한 번 "
         "고르고, decoder layer별 fixed probe-head Q/K attention을 head-mean하여 "
         "visual score를 만든다. separator를 제외한 Top-25% token을 고른 뒤 "
         "token→64-token chunk→contiguous pread plan으로 변환하고, K/V를 읽어 "
         "GPU cache에 scatter/mask한 뒤 prefill·generation한다."), "",
        "### 2. SparseVLM에서 재사용한 코드", "",
        ("`select_raters()`와 `rater_visual_scores_from_qk(..., "
         "head_reduce=\"mean\")`, `select_topk()`/`topk_budget()`를 직접 "
         "재사용했다. `rater_visual_scores()`의 rater-to-visual attention "
         "정의는 Q/K 전용 함수가 동일하게 계산하므로 full attention matrix나 "
         "full Visual-KV preload는 하지 않는다."), "",
        "### 3. IMPRESS-specific mechanism", "",
        ("기존 historical path는 보존했지만 QA-Select25에서는 probe-head "
         "Jaccard voting/threshold, similarity fallback, full-layer fallback, "
         "adaptive layer ratio를 우회했다. SparseVLM의 recycling, merging, "
         "diversity/Static+Diverse, calibration question/training도 사용하지 "
         "않았다. 측정 validation에서 fallback=0, static/diversity calls=0을 "
         "확인했다."), "",
        "### 4. Physical layout", "",
        ("QA-Select25는 original/canonical raster 순서이며 repacking/order "
         "metadata가 없다. separator KV와 probe K만 sidecar다. Ours25는 Turn-1 "
         "image-only Vision Encoder saliency 순서로 물리 repack한 뒤 같은 이미지의 "
         "후속 질문마다 동일 first-k prefix를 읽는다."), "",
        "### 5. Retention 계산", "",
        ("두 selective method의 nominal budget은 0.25다. QA는 layer마다 "
         "`ceil(0.25 × n_spatial)`개를 골라 logical ratio를 고정하고 separator는 "
         "budget 밖 sidecar로 항상 유지한다. Ours는 repacked physical prefix의 "
         "chunk-aligned rows를 읽으므로 logical/actual ratio가 nominal과 조금 "
         "다를 수 있다. 실제 평균 logical ratio는 QA "
         f"{_pct(qa['logical_selected_token_ratio'])}, Ours "
         f"{_pct(ours['logical_selected_token_ratio'])}다."), "",
        "### 6. Query-dependent selection 검증", "",
        (f"QA cache-hit {selection['n_query_requests']}건에서 매 layer query "
         "scoring call과 current-question-only prompt를 검증했고, "
         f"{selection['n_pairs']}개 within-image pair 중 "
         f"{selection['different_selection_pairs']}개가 다른 selection이었다. "
         "Ours query scoring call은 0이며 image별 prefix hash는 모든 turn에서 "
         "동일했다. future query leakage는 0이다."), "",
        "### 7. 질문별 selected-token Jaccard", "",
        (f"layer-macro pairwise token Jaccard={_number(selection['mean_pairwise_token_jaccard'], 6)}, "
         f"consecutive={_number(selection['mean_consecutive_token_jaccard'], 6)}, "
         f"chunk Jaccard={_number(selection['mean_pairwise_chunk_jaccard'], 6)}, "
         f"identical-selection rate={_pct(selection['identical_selection_rate'], 4)}다. "
         "400개 pair의 question IDs와 layer별 token/chunk Jaccard, 그리고 200개 "
         "request의 정확한 selected IDs는 `selection.json`에 보존했다."), "",
        "### 8. QA selector overhead", "",
        (f"QA selector_ms={_number(qa['selector_ms'])} ms, "
         f"online_selector_total_ms={_number(qa['online_selector_total_ms'])} ms/request다. "
         f"세부 평균은 rater={_number(qa['rater_selection_ms'])}, "
         f"query projection={_number(qa['query_projection_ms'])}, "
         f"query scoring={_number(qa['query_scoring_ms'])}, "
         f"top-k={_number(qa['topk_ms'])}, ID D2H={_number(qa['selected_id_d2h_ms'])}, "
         f"chunk planning={_number(qa['chunk_planning_ms'])} ms다. 이 component들은 "
         "CUDA/host overlap이 있어 합을 TTFT decomposition으로 해석하면 안 되며, "
         "모두 measured TTFT critical path 안에 있다."), "",
        "### 9. Probe I/O", "",
        (f"QA probe read={_number(qa['probe_io_mb'], 6)} MB/request, "
         f"probe I/O latency={_number(qa['probe_io_ms'])} ms/request다. Ours는 "
         f"{_number(ours['probe_io_mb'], 6)} MB와 "
         f"{_number(ours['probe_io_ms'])} ms로 query probe I/O가 없다."), "",
        "### 10. Actual SSD MB", "",
    ])
    for key in METHOD_KEYS:
        row = methods[key]
        lines.append(
            f"- {row['display_label']}: "
            f"{_number(row['actual_ssd_mb_per_cache_hit'], 6)} MB/request "
            f"({_pct(row['actual_ssd_ratio_vs_fullload'], 4)} of FullLoad)")
    lines.extend([
        "", "logical 25%를 actual 25% bytes로 보이게 만들기 위한 selection 왜곡은 하지 않았다.", "",
        "### 11. Touched chunk fraction", "",
        (f"QA={_pct(qa['touched_chunk_fraction'], 4)}, "
         f"Ours={_pct(ours['touched_chunk_fraction'], 4)}다. QA의 sparse token "
         "분산 때문에 logical ratio와 physical chunk footprint가 다르다."), "",
        "### 12. Contiguous runs/locality", "",
        (f"QA는 layer당 {_number(qa['contiguous_runs_per_layer'])} runs, "
         f"평균 run length {_number(qa['mean_contiguous_run_length'])} chunks, "
         f"{_number(qa['ssd_preads_per_cache_hit'])} preads/request다. Ours는 "
         f"layer당 {_number(ours['contiguous_runs_per_layer'])} run, "
         f"평균 {_number(ours['mean_contiguous_run_length'])} chunks, "
         f"{_number(ours['ssd_preads_per_cache_hit'])} preads/request인 first-k "
         "sequential prefix(+separator sidecar)다."), "",
        "### 13. GQA accuracy", "",
    ])
    for key in METHOD_KEYS:
        row = methods[key]
        lines.append(
            f"- {row['display_label']}: all turns {_pct(row['accuracy_all_turns'])}, "
            f"cache hits {_pct(row['accuracy_cache_hits'])}")
    lines.extend(["", "### 14. GQA TTFT", ""])
    for key in METHOD_KEYS:
        row = methods[key]
        lines.append(
            f"- {row['display_label']}: mean {_number(row['ttft_cache_hit_mean_ms'])} ms, "
            f"p50 {_number(row['ttft_cache_hit_p50_ms'])} ms, "
            f"p95 {_number(row['ttft_cache_hit_p95_ms'])} ms")
    lines.extend([
        "", "TTFT 경계는 prompt construction 전에 시작해 tokenization, initial H2D, "
        "online selection/I/O/scatter, prefill을 거쳐 synchronized first-token "
        "availability에서 끝나며 네 method에 동일하다.", "",
        "### 15. Ours25와 quality gap", "",
        (f"QA−Ours accuracy gap은 all turns "
         f"{_signed(comparison['qa_minus_ours_accuracy_all_turns_pp'])} pp, "
         f"cache hits {_signed(comparison['qa_minus_ours_accuracy_cache_hits_pp'])} pp다."), "",
        "### 16. Ours25와 TTFT gap", "",
        (f"QA−Ours cache-hit mean TTFT gap은 "
         f"{_signed(comparison['qa_minus_ours_ttft_cache_hit_ms'])} ms이며, "
         f"QA/Ours ratio는 {_number(comparison['qa_over_ours_ttft_ratio'], 6)}×다."), "",
        "### 17. ReComp/FullLoad 기존 결과 일관성", "",
    ])
    lines.extend(_reference_lines(reference))
    lines.extend([
        "", "### 18. Test result", "",
        (f"- Run-level validation: {len(checks)}/{len(checks)} checks PASS "
         "(`validation.json`)."),
    ])
    if test_result:
        lines.append(f"- CPU test suite: {test_result}")
    else:
        lines.append(
            "- CPU test suite: 별도 unittest 로그가 이 run bundle에 포함되지 않아 "
            "수치를 추정하지 않았다. 재현 시 `python -m unittest discover -s "
            "tests -p 'test_*.py'`를 실행한다.")
    lines.extend([
        "", "### 19. Artifact protection", "",
        (f"기존 `runs/`/`results/` 보호 항목 "
         f"{protection_report['entry_count_after']:,}개 "
         f"(files {protection_report['file_count_after']:,}, directories "
         f"{protection_report['directory_count_after']:,}, symlinks "
         f"{protection_report['symlink_count_after']:,})의 before/after manifest "
         f"SHA256가 `{protection_report['after_manifest_sha256']}`로 동일했다. "
         "missing/added/changed=0/0/0이다. 새 query-aware roots만 제외됐고, "
         "top-level KV-store tree는 의도적으로 hash scope 밖이다."), "",
        "### 20. 발견된 limitation", "",
    ])
    lines.extend(f"- {item}" for item in limitations)
    lines.extend([
        "- 이 단계는 GQA pilot까지만 수행했다. MT-GQA-reconstructed, ConvBench, "
        "VisDial full rerun은 수행하지 않았으므로 conversational history가 있는 "
        "multi-turn 일반화 결론은 아직 내릴 수 없다.", "",
        "## Hashes and provenance", "",
        f"- Run directory: `{run_dir}`",
        f"- Results directory: `{results_dir}`",
        f"- GQA index SHA256: `{config['index_sha256']}`",
        f"- Frozen full workload SHA256: `{config['full_workload_sha256']}`",
        f"- Selected workload SHA256: `{config['selected_workload_sha256']}`",
        f"- Protection manifest file SHA256: `{manifest_file_sha}`",
        f"- Protection validation file SHA256: `{protection_file_sha}`",
        f"- Protected entries canonical SHA256: `{protection_manifest['manifest_sha256']}`",
        "", "### Run evidence SHA256", "",
        "| File | SHA256 |", "|---|---|",
    ])
    lines.extend(f"| `{name}` | `{digest}` |" for name, digest in run_hashes.items())
    if copied_hashes:
        lines.extend(["", "### Compact result bundle SHA256", "",
                      "| File | SHA256 |", "|---|---|"])
        lines.extend(
            f"| `{name}` | `{digest}` |" for name, digest in copied_hashes.items())
    lines.extend(["", "### Report-time source SHA256", "",
                  "| File | SHA256 |", "|---|---|"])
    for relative in SOURCE_FILES:
        lines.append(f"| `{relative}` | `{_sha256_file(ROOT / relative)}` |")

    executable = "/home/dblab/anaconda3/envs/mllm_ft/bin/python"
    lines.extend([
        "", "## Reproduction", "",
        "출력 경로는 반드시 새 경로를 사용한다.", "",
        "```bash",
        (f"{executable} -m unittest discover -s tests -p 'test_*.py'"),
        (f"{executable} scripts/50_protect_query_aware_artifacts.py --before "
         "--run-root runs/query_aware_repro --results-root results/query_aware_repro"),
        ("HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
         "TOKENIZERS_PARALLELISM=false \\") ,
        (f"  {executable} scripts/49_eval_query_aware_baseline.py \\") ,
        "  --index data/index.json \\",
        "  --run-dir runs/query_aware_repro/gqa40_240 \\",
        "  --store-dir runs/query_aware_repro/gqa40_240_store \\",
        "  --results-dir results/query_aware_repro \\",
        "  --max-images 40 --skip 4 --questions 6 --seed 1234 \\",
        f"  --max-new-tokens {config['max_new_tokens']} \\",
        f"  --expected-index-sha256 {config['index_sha256']} \\",
        f"  --expected-workload-sha256 {config['full_workload_sha256']} \\",
        "  --expected-images 40 --expected-questions 240",
        (f"{executable} scripts/50_protect_query_aware_artifacts.py --verify "
         "--run-root runs/query_aware_repro --results-root results/query_aware_repro"),
        (f"{executable} scripts/51_report_query_aware_baseline.py "
         "--run-dir runs/query_aware_repro/gqa40_240 "
         "--results-dir results/query_aware_repro"),
        "```", "",
        "QUERY-AWARE BASELINE VALIDATED: YES",
    ])
    return "\n".join(lines) + "\n"


def generate_report(
    run_dir: Path | str,
    results_dir: Path | str,
    *,
    test_result: str | None = None,
) -> Path:
    run_argument = Path(run_dir)
    results_argument = Path(results_dir)
    if run_argument.is_symlink() or results_argument.is_symlink():
        raise ReportValidationError("run/results arguments may not be symlinks")
    run = run_argument.resolve()
    results = results_argument.resolve()
    if results.is_symlink() or not results.is_dir():
        raise ReportValidationError(
            f"results directory is not a real existing directory: {results}")
    if run == results or run in results.parents or results in run.parents:
        raise ReportValidationError("run and results directories overlap")
    config, summary, validation, run_hashes = _validate_experiment(run)
    configured_results = config.get("results_dir")
    if (configured_results is not None
            and (not isinstance(configured_results, str)
                 or Path(configured_results).resolve() != results)):
        raise ReportValidationError(
            "config results_dir does not identify this result bundle")
    manifest, protection, manifest_sha, protection_sha = (
        _find_and_validate_protection(run, results))
    copied_hashes = _verify_compact_bundle(run, results, run_hashes)
    markdown = _build_markdown(
        run, results, config, summary, validation, run_hashes,
        copied_hashes, manifest, protection, manifest_sha, protection_sha,
        test_result)
    if not markdown.rstrip().endswith("QUERY-AWARE BASELINE VALIDATED: YES"):
        raise AssertionError("report terminal verdict invariant failed")
    output = results / "ANALYSIS.md"
    _atomic_text_exclusive(output, markdown)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--test-result",
        help=("verbatim observed CPU test summary, for example "
              "'195 tests passed in 6.1s'; never inferred when omitted"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output = generate_report(
            args.run_dir, args.results_dir, test_result=args.test_result)
    except (ReportValidationError, FileExistsError) as error:
        print(f"query-aware report refused: {error}", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": "published",
        "analysis": str(output),
        "sha256": _sha256_file(output),
        "terminal_verdict": "QUERY-AWARE BASELINE VALIDATED: YES",
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

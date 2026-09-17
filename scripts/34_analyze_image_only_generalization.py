"""Analyze the image-only repack cross-dataset generalization experiment.

The evaluator writes one immutable JSON artifact per image.  This program is
the only merge/analysis step: it verifies the frozen GQA-large, VQAv2 and
TextVQA workloads, recomputes all summaries from the per-request rows, and
publishes a new result tree with an atomic rename.  Existing runs and previous
generalization results are read-only evidence and are never modified.

The paper-facing latency is ``end_to_end_ttft_ms``.  ``core_ttft_ms`` is kept
as a diagnostic.  Quality confidence intervals use paired IMAGE-cluster
bootstrap resampling: an image is drawn and all of its evaluation questions
are carried together.

Typical invocation::

    python scripts/34_analyze_image_only_generalization.py \
      --run-root runs/image_only_generalization \
      --results-root results/image_only_generalization
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mmimpress.dataset import METRICS  # noqa: E402

DEFAULT_RUN_ROOT = ROOT / "runs/image_only_generalization"
DEFAULT_RESULTS_ROOT = ROOT / "results/image_only_generalization"
SCHEMA_VERSION = "image-only-generalization-analysis-v1"
RUNNER_SCHEMA_VERSION = "image-only-generalization-e2e-ttft-v1"
MB = 1_000_000.0
TIMING_TOLERANCE_MS = 1.5
SCORE_TOLERANCE = 1e-9

# These are the immutable workloads used by the completed Static+Diverse
# cross-dataset study.  The workload hash is SHA256 of the ordered lines
# ``image_id<TAB>question_id`` without a trailing newline, exactly as in
# scripts/04_eval.py.
DATASETS: dict[str, dict[str, Any]] = {
    "gqa_large": {
        "label": "GQA-large",
        "index": ROOT / "data/gqa_large/index.json",
        "index_sha256": (
            "0d50962f0c1bac3bc6e1836978289d5fde7d60b434f55cebbc4607c4c80a797c"
        ),
        "workload_sha256": (
            "cabec1bb1035c836839b98d72ba6a04558529d246cf55f8499ca75ea2d2f290c"
        ),
        "skip": 1,
        "questions": 3,
        "n_images": 395,
        "n_questions": 1185,
        "metric": "gqa",
    },
    "vqav2": {
        "label": "VQAv2",
        "index": ROOT / "data/vqav2/index.json",
        "index_sha256": (
            "b83d5fa288fcb722ca073e261d3fec9086629ed0db2e568a2d5d6a24ef1589d7"
        ),
        "workload_sha256": (
            "e341b499a968c5caba4ddffc58fdf0ccafe2e0e212b6cb280ca1b11fc9f31d18"
        ),
        "skip": 1,
        "questions": 4,
        "n_images": 250,
        "n_questions": 1000,
        "metric": "vqa",
    },
    "textvqa": {
        "label": "TextVQA",
        "index": ROOT / "data/textvqa/index.json",
        "index_sha256": (
            "b1e5ff0eaba2a45c6398e7cb90631cdc7387eff25ed0997a66374f25968a2f4d"
        ),
        "workload_sha256": (
            "49fa0b15f406132162cba1b47245f28a560d28b4f3af76985c855d77a145a2a6"
        ),
        "skip": 1,
        "questions": 1,
        "n_images": 500,
        "n_questions": 500,
        "metric": "vqa",
    },
}

METHOD_ORDER = ("ReComp", "FullLoad", "Prefix25", "Prefix45")
METHOD_KEY_TO_LABEL = {
    "recompute": "ReComp",
    "recomp": "ReComp",
    "fullload": "FullLoad",
    "full_load": "FullLoad",
    "prefix25": "Prefix25",
    "prefix45": "Prefix45",
}
METHOD_BUDGET = {
    "ReComp": None,
    "FullLoad": 1.0,
    "Prefix25": 0.25,
    "Prefix45": 0.45,
}
PREFIX_METHODS = ("Prefix25", "Prefix45")
CACHE_METHODS = ("FullLoad", "Prefix25", "Prefix45")

QTYPE = (
    ("yes/no", re.compile(
        r"^(is|are|do|does|was|were|has|have|can|did)\b", re.I)),
    ("color", re.compile(r"\bcolou?r|what colou?r\b", re.I)),
    ("count", re.compile(r"\bhow many|number of\b", re.I)),
    ("spatial", re.compile(
        r"\b(left|right|above|below|behind|front|under|near|beside|"
        r"top|bottom|side)\b", re.I)),
    ("material/attr", re.compile(
        r"\b(material|made of|shape|size|texture|large|small|tall|"
        r"short|thin|thick)\b", re.I)),
    ("object", re.compile(r"^(what|which|who)\b", re.I)),
)
CATEGORY_ORDER = tuple(name for name, _ in QTYPE) + ("other",)
CATEGORY_RULES = {
    name: pattern.pattern for name, pattern in QTYPE
} | {"other": "fallback after all preceding rules"}
CATEGORY_DATASET_INTERPRETATION = {
    "gqa_large": (
        "Exploratory GQA facets: yes/no, color, count, spatial, "
        "material/attribute, object, and fallback other."
    ),
    "vqav2": (
        "The yes/no and count buckets approximate VQAv2 yes/no and number; "
        "the remaining lexical buckets are exploratory subdivisions of the "
        "official-style other class."
    ),
    "textvqa": (
        "Every row belongs to the OCR-focused TextVQA workload; these lexical "
        "question-form facets are not official OCR/non-OCR annotations."
    ),
}

DATASET_RESULT_FILES = (
    "config.json",
    "raw.jsonl",
    "per_request.csv",
    "summary.csv",
    "statistical_analysis.csv",
    "disagreement_analysis.csv",
    "category_analysis.csv",
    "persistence_summary.csv",
    "importance_coverage.csv",
    "validation.json",
    "README.md",
)
ROOT_RESULT_FILES = (
    "summary_all.csv",
    "cross_dataset_table.csv",
    "statistical_analysis.csv",
    "category_analysis.csv",
    "fig_accuracy_ttft.csv",
    "fig_accuracy_ssd.csv",
    "fig_ttft_reduction.csv",
    "fig_ssd_reduction.csv",
    "validation.json",
    "README.md",
)


class AnalysisError(RuntimeError):
    """Raised when immutable evidence or a derived claim fails validation."""


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str | None:
    """Hash relative filenames and file contents without changing metadata."""
    if not path.exists():
        return None
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(rel).to_bytes(8, "big"))
        digest.update(rel)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def load_protection_manifest(path: Path | None) -> dict[str, Any] | None:
    """Load several simple file-hash manifest shapes.

    Accepted shapes are ``{"files": {path: sha256}}``, a top-level path/hash
    mapping, or ``{"files": [{"path": ..., "sha256": ...}]}``.  Relative
    paths are resolved against ``root``/``base_path`` in the manifest when
    supplied, otherwise against the repository root.
    """
    if path is None:
        return None
    raw = json.loads(path.read_text())
    base = ROOT
    payload: Any = raw
    if isinstance(raw, Mapping):
        base_value = raw.get("root", raw.get("base_path"))
        if base_value:
            base = Path(str(base_value)).resolve()
        for key in ("files", "hashes", "protected_files", "entries"):
            if key in raw:
                payload = raw[key]
                break
    records: list[dict[str, str]] = []
    if isinstance(payload, Mapping):
        for filename, value in payload.items():
            digest = (value.get("sha256") if isinstance(value, Mapping)
                      else value)
            records.append({"path": str(filename), "sha256": str(digest)})
    elif isinstance(payload, list):
        for value in payload:
            if not isinstance(value, Mapping):
                raise AnalysisError("protection manifest list contains non-object")
            filename = field(value, "path", "file", "relative_path")
            digest = field(value, "sha256", "hash")
            if filename is None or digest is None:
                raise AnalysisError("protection manifest entry lacks path/SHA256")
            records.append({"path": str(filename), "sha256": str(digest)})
    else:
        raise AnalysisError("unsupported protection manifest shape")
    if not records:
        raise AnalysisError("empty protection manifest")
    resolved: list[dict[str, str]] = []
    seen: set[Path] = set()
    for record in records:
        digest = record["sha256"].lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise AnalysisError(
                f"invalid protection SHA256 for {record['path']}: {digest!r}")
        candidate = Path(record["path"])
        absolute = candidate.resolve() if candidate.is_absolute() else (
            base / candidate).resolve()
        if absolute in seen:
            raise AnalysisError(f"duplicate protection path: {absolute}")
        seen.add(absolute)
        resolved.append({"path": str(absolute), "sha256": digest})
    return {
        "manifest_path": str(path.resolve()),
        "manifest_sha256": sha256_file(path),
        "base": str(base),
        "files": resolved,
    }


def check_protection_manifest(manifest: dict[str, Any] | None) -> dict[str, Any]:
    if manifest is None:
        return {
            "provided": False,
            "passed": None,
            "n_files": 0,
            "missing": [],
            "changed": [],
        }
    missing: list[str] = []
    changed: list[dict[str, str]] = []
    for record in manifest["files"]:
        path = Path(record["path"])
        if not path.is_file():
            missing.append(str(path))
            continue
        observed = sha256_file(path)
        if observed != record["sha256"]:
            changed.append({
                "path": str(path), "expected": record["sha256"],
                "observed": observed,
            })
    return {
        "provided": True,
        "passed": not missing and not changed,
        "manifest_path": manifest["manifest_path"],
        "manifest_sha256": manifest["manifest_sha256"],
        "n_files": len(manifest["files"]),
        "missing": missing,
        "changed": changed,
    }


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise AnalysisError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"expected a JSON object: {path}")
    return value


def read_index(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise AnalysisError(f"cannot read index {path}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
        raise AnalysisError(f"expected a list of objects: {path}")
    return value


def dotted(value: Mapping[str, Any], name: str) -> Any:
    current: Any = value
    for part in name.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def field(row: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        value = dotted(row, name)
        if value not in (None, ""):
            return value
    return default


def number(value: Any, context: str, *, allow_none: bool = False,
           default: float | None = None) -> float | None:
    if value in (None, ""):
        if default is not None:
            return float(default)
        if allow_none:
            return None
        raise AnalysisError(f"missing numeric field: {context}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"invalid number {context}: {value!r}") from exc
    if not math.isfinite(result):
        raise AnalysisError(f"non-finite number {context}: {value!r}")
    return result


def integer(value: Any, context: str, *, allow_none: bool = False,
            default: int | None = None) -> int | None:
    if value in (None, "") and default is not None:
        return int(default)
    parsed = number(value, context, allow_none=allow_none)
    if parsed is None:
        return None
    if not parsed.is_integer():
        raise AnalysisError(f"not an integer {context}: {value!r}")
    return int(parsed)


def boolean(value: Any, context: str, *, allow_none: bool = False,
            default: bool | None = None) -> bool | None:
    if value in (None, ""):
        if default is not None:
            return default
        if allow_none:
            return None
        raise AnalysisError(f"missing boolean field: {context}")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "1", "pass", "passed"}:
        return True
    if text in {"false", "no", "0", "fail", "failed"}:
        return False
    raise AnalysisError(f"invalid boolean {context}: {value!r}")


def percentile(values: Iterable[float], q: float) -> float | None:
    data = sorted(float(value) for value in values)
    if not data:
        return None
    position = (len(data) - 1) * q / 100.0
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return data[lower]
    weight = position - lower
    return data[lower] * (1.0 - weight) + data[upper] * weight


def mean(values: Iterable[float]) -> float | None:
    data = [float(value) for value in values]
    return sum(data) / len(data) if data else None


def stats(values: Iterable[float]) -> dict[str, float | None]:
    data = [float(value) for value in values]
    return {
        "mean": mean(data),
        "p50": percentile(data, 50),
        "p95": percentile(data, 95),
        "min": min(data) if data else None,
        "max": max(data) if data else None,
    }


def close(left: float, right: float, tolerance: float = SCORE_TOLERANCE) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def qtype(question: str) -> str:
    for name, pattern in QTYPE:
        if pattern.search(question):
            return name
    return "other"


def normalize_gold(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if value is None:
        raise AnalysisError("missing gold answer(s)")
    return [str(value)]


def canonical_method(row: Mapping[str, Any]) -> str:
    key = str(field(row, "method_key", default="")).strip().lower()
    if key in METHOD_KEY_TO_LABEL:
        return METHOD_KEY_TO_LABEL[key]
    label = re.sub(
        r"[^a-z0-9]+", "", str(field(row, "method", default="")).lower())
    if label in {"recomp", "recompute"}:
        return "ReComp"
    if label in {"fullload", "imageonlyfullload"}:
        return "FullLoad"
    if "prefix25" in label:
        return "Prefix25"
    if "prefix45" in label:
        return "Prefix45"
    raise AnalysisError(f"unknown method: key={key!r}, label={label!r}")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]],
              columns: Sequence[str] | None = None) -> None:
    if columns is None:
        columns = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    columns.append(key)
                    seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False,
                                    allow_nan=False) + "\n")


def budget_chunk_count(n_chunks: int, budget: float) -> int:
    """Match mmimpress.cvpr25: Python ties-to-even round plus clamping."""
    return max(1, min(int(n_chunks), int(round(float(budget) * n_chunks))))


def parse_selected_ids(value: Any, context: str) -> list[list[int]] | None:
    """Normalize a single selection, per-layer list, or layer-keyed mapping."""
    if value in (None, ""):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise AnalysisError(f"invalid selected IDs {context}") from exc
    if isinstance(value, Mapping):
        value = [value[key] for key in sorted(value, key=lambda x: str(x))]
    if not isinstance(value, list):
        raise AnalysisError(f"selected IDs are not a list: {context}")
    if not value:
        return []
    if all(isinstance(item, (int, float)) for item in value):
        return [[int(item) for item in value]]
    result: list[list[int]] = []
    for layer in value:
        if not isinstance(layer, list):
            raise AnalysisError(f"bad per-layer selected IDs: {context}")
        result.append([int(item) for item in layer])
    return result


def exact_binomial_two_sided(left: int, right: int) -> float:
    """Two-sided exact McNemar p-value for discordant binary pairs."""
    n = int(left) + int(right)
    if n == 0:
        return 1.0
    k = min(int(left), int(right))
    numerator = 2 * sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, float(numerator / (2 ** n)))


def recursive_values(value: Any, names: set[str]) -> list[Any]:
    """Find exact key names recursively in provenance/config structures."""
    found: list[Any] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in names and child not in (None, ""):
                found.append(child)
            found.extend(recursive_values(child, names))
    elif isinstance(value, list):
        for child in value:
            found.extend(recursive_values(child, names))
    return found


def unique_recursive_value(containers: Sequence[Any], names: Sequence[str],
                           context: str, *, required: bool = True) -> Any:
    found: list[Any] = []
    for container in containers:
        found.extend(recursive_values(container, set(names)))
    canonical: dict[str, Any] = {}
    for value in found:
        canonical[json.dumps(value, sort_keys=True, default=str)] = value
    if not canonical:
        if required:
            raise AnalysisError(
                f"missing provenance field {context}; accepted names={names}")
        return None
    if len(canonical) != 1:
        raise AnalysisError(
            f"conflicting provenance values for {context}: "
            f"{list(canonical.values())!r}")
    return next(iter(canonical.values()))


def expected_workload(dataset: str) -> tuple[
        list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    spec = DATASETS[dataset]
    index_path = Path(spec["index"])
    actual_index_hash = sha256_file(index_path)
    if actual_index_hash != spec["index_sha256"]:
        raise AnalysisError(
            f"{dataset} index SHA mismatch: {actual_index_hash} != "
            f"{spec['index_sha256']}")
    index = read_index(index_path)
    if len(index) != spec["n_images"]:
        raise AnalysisError(
            f"{dataset}: expected {spec['n_images']} index images, got {len(index)}")

    ordered: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    skip, count = int(spec["skip"]), int(spec["questions"])
    for image_ordinal, entry in enumerate(index):
        image_id = str(entry.get("image_id", ""))
        questions = entry.get("questions")
        if not image_id or not isinstance(questions, list):
            raise AnalysisError(f"bad {dataset} index entry {image_ordinal}")
        selected = questions[skip:skip + count]
        if len(selected) != count:
            raise AnalysisError(
                f"{dataset}/{image_id}: expected {count} evaluation questions")
        for question_ordinal, question in enumerate(selected):
            qid = str(question.get("question_id", ""))
            if not qid:
                raise AnalysisError(f"missing question_id for {dataset}/{image_id}")
            record = {
                "dataset": dataset,
                "image_id": image_id,
                "image_ordinal": image_ordinal,
                "question_id": qid,
                "question_ordinal": question_ordinal,
                "question": str(question.get("question", "")),
                "gold": normalize_gold(field(question, "answers", "answer")),
                "question_type": qtype(str(question.get("question", ""))),
            }
            key = (image_id, qid)
            if key in by_key:
                raise AnalysisError(f"duplicate workload key: {dataset}/{key}")
            by_key[key] = record
            ordered.append(record)

    if len(ordered) != spec["n_questions"]:
        raise AnalysisError(
            f"{dataset}: expected {spec['n_questions']} questions, got "
            f"{len(ordered)}")
    blob = "\n".join(
        f"{row['image_id']}\t{row['question_id']}" for row in ordered
    ).encode("utf-8")
    actual_workload_hash = hashlib.sha256(blob).hexdigest()
    if actual_workload_hash != spec["workload_sha256"]:
        raise AnalysisError(
            f"{dataset} workload SHA mismatch: {actual_workload_hash} != "
            f"{spec['workload_sha256']}")
    return ordered, by_key


def artifact_identity(artifact: Mapping[str, Any], path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "schema_version": artifact.get("schema_version"),
    }


def load_dataset_artifacts(dataset: str, run_dir: Path,
                           expected: Sequence[dict[str, Any]]) -> tuple[
                               dict[str, Any], list[dict[str, Any]],
                               list[dict[str, Any]], list[dict[str, Any]]]:
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise AnalysisError(f"missing runner config: {config_path}")
    config = read_json(config_path)
    if config.get("schema_version") != RUNNER_SCHEMA_VERSION:
        raise AnalysisError(
            f"runner config schema mismatch: {config.get('schema_version')!r} "
            f"!= {RUNNER_SCHEMA_VERSION!r}")
    if str(config.get("dataset", "")) != dataset:
        raise AnalysisError(
            f"runner config dataset mismatch: {config.get('dataset')!r} "
            f"!= {dataset!r}")
    images_dir = run_dir / "images"
    paths = sorted(images_dir.glob("*.json"))
    if not paths:
        raise AnalysisError(f"no per-image artifacts under {images_dir}")
    if len(paths) != DATASETS[dataset]["n_images"]:
        raise AnalysisError(
            f"{dataset}: expected {DATASETS[dataset]['n_images']} image "
            f"artifacts, got {len(paths)}")

    expected_image_order = list(dict.fromkeys(
        str(row["image_id"]) for row in expected))
    expected_ordinal = {image_id: i for i, image_id in enumerate(
        expected_image_order)}
    artifacts: list[tuple[int, Path, dict[str, Any]]] = []
    for path in paths:
        artifact = read_json(path)
        if artifact.get("schema_version") != RUNNER_SCHEMA_VERSION:
            raise AnalysisError(
                f"image artifact schema mismatch in {path}: "
                f"{artifact.get('schema_version')!r}")
        claimed_content_hash = artifact.get("artifact_content_sha256")
        if not isinstance(claimed_content_hash, str) or not re.fullmatch(
                r"[0-9a-f]{64}", claimed_content_hash):
            raise AnalysisError(f"missing/invalid artifact content SHA: {path}")
        observed_content_hash = stable_json_sha256({
            key: value for key, value in artifact.items()
            if key != "artifact_content_sha256"})
        if claimed_content_hash != observed_content_hash:
            raise AnalysisError(f"artifact content SHA mismatch: {path}")
        artifact_validation = artifact.get("validation")
        if (not isinstance(artifact_validation, Mapping)
                or artifact_validation.get("passed") is not True):
            raise AnalysisError(f"image artifact did not pass live validation: {path}")
        image_id = str(artifact.get("image_id", ""))
        if image_id not in expected_ordinal:
            raise AnalysisError(f"unexpected image artifact {path}: {image_id}")
        ordinal = integer(
            artifact.get("image_ordinal"), f"{path}/image_ordinal",
            default=expected_ordinal[image_id])
        if ordinal != expected_ordinal[image_id]:
            raise AnalysisError(
                f"wrong image ordinal for {dataset}/{image_id}: {ordinal}")
        artifacts.append((ordinal, path, artifact))
    artifacts.sort(key=lambda item: item[0])
    image_ids = [str(artifact.get("image_id")) for _, _, artifact in artifacts]
    if image_ids != expected_image_order:
        raise AnalysisError(f"{dataset}: image artifacts do not match index order")
    if len(set(image_ids)) != len(image_ids):
        raise AnalysisError(f"{dataset}: duplicate per-image artifacts")

    rows: list[dict[str, Any]] = []
    build_profiles: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for _ordinal, path, artifact in artifacts:
        if str(artifact.get("dataset")) != dataset:
            raise AnalysisError(f"dataset mismatch in {path}")
        if artifact.get("index_sha256") != DATASETS[dataset]["index_sha256"]:
            raise AnalysisError(f"index SHA mismatch in {path}")
        if artifact.get("workload_sha256") != DATASETS[dataset]["workload_sha256"]:
            raise AnalysisError(f"workload SHA mismatch in {path}")
        image_rows = artifact.get("rows")
        if not isinstance(image_rows, list) or not image_rows:
            raise AnalysisError(f"missing rows in {path}")
        for row_ordinal, row in enumerate(image_rows):
            if not isinstance(row, dict):
                raise AnalysisError(f"non-object row in {path}")
            if row.get("schema_version") != RUNNER_SCHEMA_VERSION:
                raise AnalysisError(
                    f"request row schema mismatch in {path} row "
                    f"{row_ordinal}: {row.get('schema_version')!r}")
            enriched = dict(row)
            enriched["_artifact_path"] = str(path.resolve())
            enriched["_artifact_sha256"] = sha256_file(path)
            enriched["_row_ordinal"] = row_ordinal
            enriched["_artifact"] = artifact
            rows.append(enriched)
        profile = artifact.get("build_profile", {})
        if not isinstance(profile, dict):
            raise AnalysisError(f"build_profile is not an object in {path}")
        build_profiles.append({"image_id": artifact["image_id"], **profile})
        identities.append(artifact_identity(artifact, path))
    return config, rows, build_profiles, identities


def _timing(row: Mapping[str, Any], context: str, *names: str,
            allow_none: bool = False, default: float | None = None) -> float | None:
    return number(field(row, *names), context, allow_none=allow_none,
                  default=default)


def canonicalize_rows(dataset: str, source_rows: Sequence[dict[str, Any]],
                      expected_by_key: Mapping[tuple[str, str],
                                               dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        method = canonical_method(source)
        image_id = str(source.get("image_id", ""))
        question_id = str(source.get("question_id", ""))
        key = (image_id, question_id)
        if key not in expected_by_key:
            raise AnalysisError(f"unexpected request row: {dataset}/{key}/{method}")
        expected = expected_by_key[key]
        question = str(source.get("question", ""))
        gold = normalize_gold(field(source, "gold", "answers", "answer"))
        if question != expected["question"]:
            raise AnalysisError(f"question text mismatch: {dataset}/{key}")
        if gold != expected["gold"]:
            raise AnalysisError(f"gold answers mismatch: {dataset}/{key}")

        budget_raw = field(source, "budget", "retention")
        if method == "ReComp":
            if budget_raw not in (None, ""):
                parsed_budget = number(budget_raw, f"{dataset}/{key}/budget")
                if parsed_budget not in (0.0,):
                    raise AnalysisError(f"ReComp has a budget: {dataset}/{key}")
            budget = None
        else:
            budget = number(budget_raw, f"{dataset}/{key}/{method}/budget")
            if not close(budget, METHOD_BUDGET[method]):
                raise AnalysisError(
                    f"wrong budget for {dataset}/{key}/{method}: {budget}")

        prediction = str(field(source, "prediction", "answer", default=""))
        if not prediction:
            raise AnalysisError(f"missing prediction: {dataset}/{key}/{method}")
        score = number(field(source, "score", "acc", "correct"),
                       f"{dataset}/{key}/{method}/score")
        first_token = integer(
            field(source, "first_token_id"),
            f"{dataset}/{key}/{method}/first_token_id")
        generated_tokens = integer(
            field(source, "generated_tokens"),
            f"{dataset}/{key}/{method}/generated_tokens")
        core_ttft = _timing(
            source, f"{dataset}/{key}/{method}/core_ttft_ms",
            "core_ttft_ms", allow_none=True)
        end_to_end_ttft = _timing(
            source, f"{dataset}/{key}/{method}/end_to_end_ttft_ms",
            "end_to_end_ttft_ms", "ttft_ms")
        decode_ms = _timing(
            source, f"{dataset}/{key}/{method}/decode_ms", "decode_ms")
        request_e2e = _timing(
            source, f"{dataset}/{key}/{method}/request_e2e_ms",
            "request_e2e_ms", "e2e_ms", "e2e_latency_ms")
        model_e2e = _timing(
            source, f"{dataset}/{key}/{method}/model_e2e_ms",
            "model_e2e_ms", allow_none=True)

        ssd_bytes = integer(
            field(source, "ssd_read_bytes", "total_actual_pread_bytes",
                  "io.bytes"), f"{dataset}/{key}/{method}/ssd_read_bytes",
            default=0)
        ssd_ms = _timing(
            source, f"{dataset}/{key}/{method}/ssd_read_ms",
            "ssd_read_ms", "disk_ms", "io.ms", default=0.0)
        ssd_preads = integer(
            field(source, "ssd_read_preads", "preads", "io.preads"),
            f"{dataset}/{key}/{method}/ssd_read_preads", default=0)
        chunk_units = integer(
            field(source, "ssd_read_chunk_units", "ssd_read_chunks",
                  "io.chunk_units"),
            f"{dataset}/{key}/{method}/ssd_read_chunk_units", default=0)
        normal_bytes = integer(
            field(source, "normal_kv_read_bytes", "io.normal_bytes",
                  "io.per_kind.kv.bytes"),
            f"{dataset}/{key}/{method}/normal_kv_read_bytes", default=0)
        separator_bytes = integer(
            field(source, "separator_read_bytes", "io.separator_bytes",
                  "io.per_kind.sep.bytes"),
            f"{dataset}/{key}/{method}/separator_read_bytes", default=0)
        normal_preads = integer(
            field(source, "normal_kv_preads", "normal_preads",
                  "io.normal_preads"),
            f"{dataset}/{key}/{method}/normal_kv_preads", default=0)
        separator_preads = integer(
            field(source, "separator_preads", "separator_read_preads",
                  "io.separator_preads"),
            f"{dataset}/{key}/{method}/separator_preads", default=0)

        n_selected = number(
            field(source, "n_chunks_selected"),
            f"{dataset}/{key}/{method}/n_chunks_selected", allow_none=True)
        n_total = integer(
            field(source, "n_chunks_total"),
            f"{dataset}/{key}/{method}/n_chunks_total", allow_none=True)
        selected_ids = parse_selected_ids(field(
            source, "selected_chunk_ids_per_layer", "selected_chunk_ids"),
            f"{dataset}/{key}/{method}/selected_chunk_ids")
        selected_fraction = number(
            field(source, "actual_selected_normal_chunk_fraction",
                  "touched_chunk_fraction"),
            f"{dataset}/{key}/{method}/selected_fraction", allow_none=True)
        payload_ratio = number(
            field(source, "ssd_payload_ratio_vs_full_visual_kv",
                  "actual_total_ssd_byte_ratio", "ssd_byte_ratio"),
            f"{dataset}/{key}/{method}/payload_ratio", allow_none=True)
        logical_ratio = number(
            field(source, "selected_kv_ratio", "logical_kv_ratio"),
            f"{dataset}/{key}/{method}/logical_kv_ratio", allow_none=True)
        full_visual_bytes = integer(
            field(source, "full_visual_kv_bytes"),
            f"{dataset}/{key}/{method}/full_visual_kv_bytes", allow_none=True)

        row = {
            "dataset": dataset,
            "image_id": image_id,
            "image_ordinal": expected["image_ordinal"],
            "question_id": question_id,
            "question_ordinal": expected["question_ordinal"],
            "question_type": expected["question_type"],
            "method": method,
            "method_key": {
                "ReComp": "recompute", "FullLoad": "fullload",
                "Prefix25": "prefix25", "Prefix45": "prefix45",
            }[method],
            "budget": budget,
            "question": question,
            "gold": gold,
            "prediction": prediction,
            "score": score,
            "first_token_id": first_token,
            "generated_tokens": generated_tokens,
            "suffix_ids_sha256": field(source, "suffix_ids_sha256"),
            "physical_layout": field(source, "physical_layout"),
            "core_ttft_ms": core_ttft,
            "end_to_end_ttft_ms": end_to_end_ttft,
            "decode_ms": decode_ms,
            "model_e2e_ms": model_e2e,
            "e2e_ms": request_e2e,
            "ssd_read_ms": ssd_ms,
            "ssd_read_bytes": ssd_bytes,
            "ssd_read_preads": ssd_preads,
            "ssd_read_chunk_units": chunk_units,
            "normal_kv_read_bytes": normal_bytes,
            "separator_read_bytes": separator_bytes,
            "normal_kv_preads": normal_preads,
            "separator_preads": separator_preads,
            "selector_ms": _timing(
                source, f"{dataset}/{key}/{method}/selector_ms",
                "selector_ms", "first_k_planning_ms", default=0.0),
            "scatter_ms": _timing(
                source, f"{dataset}/{key}/{method}/scatter_ms",
                "scatter_ms", allow_none=True),
            "prefill_ms": _timing(
                source, f"{dataset}/{key}/{method}/prefill_ms",
                "prefill_ms", allow_none=True),
            "n_chunks_selected": n_selected,
            "n_chunks_total": n_total,
            "selected_chunk_ids_per_layer": selected_ids,
            "actual_selected_normal_chunk_fraction": selected_fraction,
            "actual_total_ssd_byte_ratio": payload_ratio,
            "selected_kv_ratio": logical_ratio,
            "full_visual_kv_bytes": full_visual_bytes,
            "importance_mass_coverage": number(
                field(source, "importance_mass_coverage"),
                f"{dataset}/{key}/{method}/importance_mass_coverage",
                allow_none=True),
            "static_score_calls": integer(
                field(source, "static_score_calls"),
                f"{dataset}/{key}/{method}/static_score_calls", default=0),
            "query_score_calls": integer(
                field(source, "query_score_calls"),
                f"{dataset}/{key}/{method}/query_score_calls", default=0),
            "diversity_calls": integer(
                field(source, "diversity_calls"),
                f"{dataset}/{key}/{method}/diversity_calls", default=0),
            "vision_forward_count": integer(
                field(source, "vision_forward_count"),
                f"{dataset}/{key}/{method}/vision_forward_count",
                allow_none=True),
            "order_position": integer(
                field(source, "order_position", "method_order_position"),
                f"{dataset}/{key}/{method}/order_position", allow_none=True),
            "order": field(source, "order", "method_order"),
            "cache_conditioning_outside_timer": boolean(
                field(source, "cache_conditioning_outside_timer",
                      "page_cache_conditioning_outside_timer",
                      "page_cache_conditioning_excluded_from_ttft"),
                f"{dataset}/{key}/{method}/conditioning", allow_none=True),
            "cache_conditioning_end": number(
                field(source, "cache_conditioning_end_ns",
                      "page_cache_conditioning_end_ns",
                      "page_cache_conditioning_finished_at_s"),
                f"{dataset}/{key}/{method}/conditioning_end", allow_none=True),
            "request_start": number(
                field(source, "request_start_ns", "request_started_ns",
                      "request_started_at_s"),
                f"{dataset}/{key}/{method}/request_start", allow_none=True),
            "store_id": field(source, "same_physical_store_id", "store_id",
                              "store_manifest_sha256", "store_sha256"),
            "layout_id": field(source, "layout_id", "layout_sha256",
                               "permutation_sha256"),
            "artifact_path": source["_artifact_path"],
            "artifact_sha256": source["_artifact_sha256"],
        }
        rows.append(row)
    rows.sort(key=lambda row: (
        row["image_ordinal"], row["question_ordinal"],
        METHOD_ORDER.index(row["method"])))
    return rows


class ValidationRecorder:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, passed: bool, observed: Any = None,
              expected: Any = None, detail: str | None = None) -> None:
        record: dict[str, Any] = {
            "name": name,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
        }
        if detail is not None:
            record["detail"] = detail
        self.checks.append(record)

    @property
    def passed(self) -> bool:
        return all(check["passed"] for check in self.checks)

    @property
    def failures(self) -> list[str]:
        return [str(check["name"]) for check in self.checks
                if not check["passed"]]


def verify_declared_boolean(containers: Sequence[Any], names: Sequence[str],
                            context: str, expected: bool) -> bool:
    value = unique_recursive_value(containers, names, context, required=True)
    return bool(boolean(value, context)) is expected


def verify_declared_zero(containers: Sequence[Any], names: Sequence[str],
                         context: str) -> bool:
    value = unique_recursive_value(containers, names, context, required=True)
    return close(number(value, context), 0.0)


def validate_dataset(dataset: str, config: dict[str, Any],
                     source_rows: Sequence[dict[str, Any]],
                     rows: Sequence[dict[str, Any]],
                     expected: Sequence[dict[str, Any]],
                     build_profiles: Sequence[dict[str, Any]],
                     identities: Sequence[dict[str, Any]]) -> dict[str, Any]:
    spec = DATASETS[dataset]
    vr = ValidationRecorder()
    expected_keys = [(row["image_id"], row["question_id"]) for row in expected]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["image_id"], row["question_id"])].append(row)

    vr.check("index_sha256", sha256_file(Path(spec["index"])) ==
             spec["index_sha256"], sha256_file(Path(spec["index"])),
             spec["index_sha256"])
    workload_blob = "\n".join(f"{i}\t{q}" for i, q in expected_keys).encode()
    workload_hash = hashlib.sha256(workload_blob).hexdigest()
    vr.check("workload_sha256", workload_hash == spec["workload_sha256"],
             workload_hash, spec["workload_sha256"])
    vr.check("image_artifact_count", len(identities) == spec["n_images"],
             len(identities), spec["n_images"])
    vr.check("evaluation_question_count", len(grouped) == spec["n_questions"],
             len(grouped), spec["n_questions"])
    vr.check("raw_row_count", len(rows) == 4 * spec["n_questions"],
             len(rows), 4 * spec["n_questions"])
    vr.check("ordered_request_keys", list(grouped) == expected_keys,
             stable_json_sha256(list(grouped)),
             stable_json_sha256(expected_keys))

    method_counts = Counter(row["method"] for row in rows)
    vr.check("method_row_counts", method_counts == Counter({
        method: spec["n_questions"] for method in METHOD_ORDER}),
        dict(method_counts),
        {method: spec["n_questions"] for method in METHOD_ORDER})
    complete_pairs = all(
        {row["method"] for row in request_rows} == set(METHOD_ORDER)
        and len(request_rows) == len(METHOD_ORDER)
        for request_rows in grouped.values())
    vr.check("same_requests_all_methods", complete_pairs, complete_pairs, True)
    suffix_match = all(
        len({row["suffix_ids_sha256"] for row in request_rows}) == 1
        and next(iter({row["suffix_ids_sha256"] for row in request_rows}))
        not in (None, "")
        for request_rows in grouped.values())
    vr.check("same_tokenized_suffix_all_methods", suffix_match,
             suffix_match, True)
    layout_labels_ok = all(
        row["physical_layout"] == (
            "not_applicable_recompute" if row["method"] == "ReComp"
            else "visionzip_image_only")
        for row in rows)
    vr.check("image_only_physical_layout", layout_labels_ok,
             layout_labels_ok, True)

    metric = METRICS[spec["metric"]]
    metric_mismatches = []
    for row in rows:
        recomputed = float(metric(row["prediction"], row["gold"]))
        if not close(recomputed, row["score"]):
            metric_mismatches.append({
                "image_id": row["image_id"],
                "question_id": row["question_id"],
                "method": row["method"],
                "recorded": row["score"],
                "recomputed": recomputed,
            })
    vr.check("scores_recompute_exactly", not metric_mismatches,
             len(metric_mismatches), 0)

    budget_ok = all(
        (row["budget"] is None if row["method"] == "ReComp" else
         close(row["budget"], METHOD_BUDGET[row["method"]]))
        for row in rows)
    vr.check("fixed_budgets_25_45", budget_ok, budget_ok, True)

    timing_positive = all(
        row["end_to_end_ttft_ms"] > 0 and row["decode_ms"] >= 0
        and row["e2e_ms"] > 0 for row in rows)
    vr.check("timings_finite_positive", timing_positive, timing_positive, True)
    ttft_lt_e2e = all(
        row["end_to_end_ttft_ms"] < row["e2e_ms"] for row in rows)
    vr.check("ttft_less_than_e2e", ttft_lt_e2e, ttft_lt_e2e, True)
    alias_consistent = all(
        field(source, "ttft_ms") is not None
        and close(number(field(source, "ttft_ms"), "ttft_ms"),
                  number(field(source, "end_to_end_ttft_ms"),
                         "end_to_end_ttft_ms"),
                  TIMING_TOLERANCE_MS)
        for source in source_rows)
    vr.check("ttft_alias_is_end_to_end", alias_consistent,
             alias_consistent, True)

    recomp = [row for row in rows if row["method"] == "ReComp"]
    cache = [row for row in rows if row["method"] in CACHE_METHODS]
    prefix = [row for row in rows if row["method"] in PREFIX_METHODS]
    vr.check("recomp_ssd_zero", all(
        row["ssd_read_bytes"] == 0 and row["ssd_read_preads"] == 0
        and close(row["ssd_read_ms"], 0.0) for row in recomp),
        sum(row["ssd_read_bytes"] for row in recomp), 0)
    vr.check("cache_methods_actual_pread", all(
        row["ssd_read_bytes"] > 0 and row["ssd_read_preads"] > 0
        and row["ssd_read_ms"] > 0 for row in cache), True, True)
    vr.check("fullload_reads_exact_full_visual_kv", all(
        row["full_visual_kv_bytes"] is not None
        and row["ssd_read_bytes"] == row["full_visual_kv_bytes"]
        for row in rows if row["method"] == "FullLoad"), True, True)

    prefix_selection_failures = []
    artifacts_by_image = {}
    for source in source_rows:
        artifacts_by_image.setdefault(str(source["image_id"]),
                                      source["_artifact"])
    for row in prefix:
        total = row["n_chunks_total"]
        selected = row["n_chunks_selected"]
        selections = row["selected_chunk_ids_per_layer"]
        if total is None or selected is None or selections is None:
            prefix_selection_failures.append(
                (row["image_id"], row["question_id"], row["method"], "missing"))
            continue
        budget = METHOD_BUDGET[row["method"]]
        expected_k = budget_chunk_count(total, budget)
        expected_ids = list(range(expected_k))
        manifest_files = field(
            artifacts_by_image[row["image_id"]], "store_manifest.file_sizes",
            default={})
        expected_layers = (len([
            name for name in manifest_files
            if re.fullmatch(r"layer_\d+/k\.bin", str(name))])
            if isinstance(manifest_files, Mapping) else 0)
        if (not close(selected, expected_k)
                or not selections
                or expected_layers <= 0
                or len(selections) != expected_layers
                or any(ids != expected_ids for ids in selections)):
            prefix_selection_failures.append((
                row["image_id"], row["question_id"], row["method"],
                {"total": total, "selected": selected,
                 "layers": len(selections) if selections else 0,
                 "expected_layers": expected_layers,
                 "first_layer": selections[0] if selections else None,
                 "expected": expected_ids}))
        fraction = row["actual_selected_normal_chunk_fraction"]
        if fraction is None or not close(fraction, expected_k / total, 1e-7):
            prefix_selection_failures.append((
                row["image_id"], row["question_id"], row["method"],
                {"fraction": fraction, "expected_fraction": expected_k / total}))
    vr.check("prefix_exact_first_k", not prefix_selection_failures,
             len(prefix_selection_failures), 0)

    nested_prefix_failures = []
    for request_key, request_rows in grouped.items():
        by_method = {row["method"]: row for row in request_rows}
        p25, p45 = by_method.get("Prefix25"), by_method.get("Prefix45")
        if p25 is None or p45 is None:
            nested_prefix_failures.append((request_key, "missing prefix arm"))
            continue
        ids25 = p25["selected_chunk_ids_per_layer"]
        ids45 = p45["selected_chunk_ids_per_layer"]
        nested = (
            p25["n_chunks_total"] == p45["n_chunks_total"]
            and isinstance(ids25, list) and isinstance(ids45, list)
            and len(ids25) == len(ids45) and len(ids25) > 0
            and all(short == long[:len(short)] and len(short) <= len(long)
                    for short, long in zip(ids25, ids45))
        )
        if not nested:
            nested_prefix_failures.append((
                request_key,
                {"p25_total": p25["n_chunks_total"],
                 "p45_total": p45["n_chunks_total"],
                 "p25_layers": ids25, "p45_layers": ids45}))
    vr.check("prefix25_is_nested_prefix_of_prefix45",
             not nested_prefix_failures, len(nested_prefix_failures), 0)

    online_calls_ok = all(
        row["static_score_calls"] == 0 and row["query_score_calls"] == 0
        and row["diversity_calls"] == 0 for row in prefix)
    vr.check("no_online_scoring_or_diversity", online_calls_ok,
             online_calls_ok, True)

    # A cache request must either carry an explicit outside-timer declaration
    # or enough event timestamps to prove it.  Conditioning duration itself is
    # benchmark setup and is intentionally not added to TTFT.
    conditioning_failures = []
    for row in cache:
        declared = row["cache_conditioning_outside_timer"]
        ended, started = row["cache_conditioning_end"], row["request_start"]
        proven_by_time = ended is not None and started is not None and ended < started
        if declared is not True and not proven_by_time:
            conditioning_failures.append(
                (row["image_id"], row["question_id"], row["method"]))
    vr.check("os_page_cache_conditioning_outside_timer",
             not conditioning_failures, len(conditioning_failures), 0)

    vision_values = [row["vision_forward_count"] for row in rows]
    vision_present = all(value is not None for value in vision_values)
    vision_ok = vision_present and all(
        row["vision_forward_count"] == (1 if row["method"] == "ReComp" else 0)
        for row in rows)
    vr.check("recomp_recomputes_cache_arms_do_not", vision_ok,
             {f"{method}:{count}": n for (method, count), n in Counter(
                 (row["method"], row["vision_forward_count"])
                 for row in rows).items()},
             {"ReComp": 1, "FullLoad": 0, "Prefix25": 0, "Prefix45": 0})

    # Each artifact has exactly one store manifest.  Cache arms may also expose
    # IDs per row; when present they must all name that same image-local store
    # and physical permutation.
    shared_store_failures = []
    for image_id in dict.fromkeys(row["image_id"] for row in rows):
        image_rows = [row for row in cache if row["image_id"] == image_id]
        recomp_rows = [row for row in recomp if row["image_id"] == image_id]
        store_ids = {str(row["store_id"]) for row in image_rows
                     if row["store_id"] not in (None, "")}
        layout_ids = {str(row["layout_id"]) for row in image_rows
                      if row["layout_id"] not in (None, "")}
        artifact = next(source["_artifact"] for source in source_rows
                        if str(source["image_id"]) == image_id)
        manifest = artifact.get("store_manifest")
        if not isinstance(manifest, Mapping):
            shared_store_failures.append((image_id, "missing store_manifest"))
        elif (not isinstance(manifest.get("meta_sha256"), str)
              or not re.fullmatch(r"[0-9a-f]{64}",
                                  str(manifest.get("meta_sha256")))):
            shared_store_failures.append((
                image_id, "missing/invalid store_manifest.meta_sha256"))
        elif store_ids != {str(manifest.get("meta_sha256"))}:
            shared_store_failures.append((
                image_id, {"row_store_ids": sorted(store_ids),
                           "manifest_meta_sha256": manifest.get("meta_sha256")}))
        if any(row["store_id"] in (None, "") for row in image_rows):
            shared_store_failures.append((
                image_id, "one or more cache-arm rows lack a store ID"))
        if any(row["store_id"] not in (None, "") for row in recomp_rows):
            shared_store_failures.append((image_id, "ReComp has a store ID"))
        if len(store_ids) != 1 or len(layout_ids) > 1:
            shared_store_failures.append((
                image_id, {"store_ids": sorted(store_ids),
                           "layout_ids": sorted(layout_ids)}))
    vr.check("full_and_prefix_share_physical_store",
             not shared_store_failures, len(shared_store_failures), 0)

    provenance_artifacts: list[dict[str, Any]] = []
    seen_artifacts: set[str] = set()
    for source in source_rows:
        path = str(source["_artifact_path"])
        if path not in seen_artifacts:
            provenance_artifacts.append(source["_artifact"])
            seen_artifacts.add(path)
    provenance_containers = [config] + provenance_artifacts
    provenance_errors: list[str] = []
    declarations = (
        ("calibration_questions_zero",
         ("calibration_questions", "calibration_questions_used",
          "calibration_questions_used_count"), "zero"),
        ("layout_uses_dataset_question_false",
         ("layout_uses_dataset_question", "uses_dataset_question"), False),
        ("layout_uses_llm_qk_false",
         ("layout_uses_llm_qk", "uses_llm_qk"), False),
        ("layout_uses_generated_answer_false",
         ("layout_uses_generated_answer", "uses_generated_answer"), False),
        ("query_dependent_importance_false",
         ("query_dependent_importance",), False),
        ("tokenization_inside_ttft",
         ("tokenization_inside_ttft", "ttft_includes_tokenization"), True),
        ("initial_h2d_inside_ttft",
         ("initial_h2d_inside_ttft", "ttft_includes_initial_h2d"), True),
    )
    for label, names, wanted in declarations:
        try:
            good = (verify_declared_zero(provenance_containers, names, label)
                    if wanted == "zero" else
                    verify_declared_boolean(provenance_containers, names,
                                            label, bool(wanted)))
        except AnalysisError as exc:
            good = False
            provenance_errors.append(str(exc))
        vr.check(label, good, good, True)

    static_diverse_values = recursive_values(
        provenance_containers,
        {"static_diverse_calls", "static_diverse_selector_calls"})
    static_diverse_ok = bool(static_diverse_values) and all(
        close(number(value, "static_diverse_calls"), 0.0)
        for value in static_diverse_values)
    vr.check("static_diverse_calls_zero", static_diverse_ok,
             static_diverse_values, [0])

    profile_requirements = {
        "saliency_ms": ("saliency_ms", "vision_saliency_ms"),
        "permutation_ms": ("permutation_ms", "permute_ms"),
        "repack_ms": ("repack_ms", "kv_repack_ms"),
        "ssd_write_ms": ("ssd_write_ms", "buffered_write_ms", "write_ms"),
        "mapping_metadata_bytes": ("mapping_metadata_bytes",),
        "visual_kv_bytes": ("visual_kv_bytes", "full_visual_kv_bytes"),
    }
    for metric_name, aliases in profile_requirements.items():
        present = sum(
            first_profile_number(profile, aliases) is not None
            for profile in build_profiles)
        vr.check(f"build_profile_{metric_name}_complete",
                 present == spec["n_images"], present, spec["n_images"])

    # Method order positions should be balanced within one request.  Across the
    # dataset each method may differ by at most one appearance per position.
    order_present = all(row["order_position"] is not None for row in rows)
    order_rows_ok = order_present
    position_counts: dict[str, Counter[int]] = {
        method: Counter() for method in METHOD_ORDER}
    if order_present:
        for request_rows in grouped.values():
            positions = [row["order_position"] for row in request_rows]
            if sorted(positions) not in ([0, 1, 2, 3], [1, 2, 3, 4]):
                order_rows_ok = False
            declared_orders = [row["order"] for row in request_rows]
            if (any(not isinstance(order, list) for order in declared_orders)
                    or any(order != declared_orders[0]
                           for order in declared_orders[1:])):
                order_rows_ok = False
            else:
                one_based = min(positions) == 1
                for row in request_rows:
                    index = row["order_position"] - (1 if one_based else 0)
                    try:
                        declared_method = METHOD_KEY_TO_LABEL[
                            str(declared_orders[0][index]).lower()]
                    except (IndexError, KeyError):
                        order_rows_ok = False
                    else:
                        if declared_method != row["method"]:
                            order_rows_ok = False
            for row in request_rows:
                position_counts[row["method"]][row["order_position"]] += 1
        for position in sorted({p for count in position_counts.values()
                                for p in count}):
            values = [position_counts[method][position]
                      for method in METHOD_ORDER]
            if max(values) - min(values) > 1:
                order_rows_ok = False
    vr.check("deterministic_balanced_method_order", order_rows_ok,
             {method: dict(counts) for method, counts in position_counts.items()},
             "each request is a permutation; each position differs by <=1")

    checks = vr.checks
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "passed": vr.passed,
        "n_checks": len(checks),
        "n_passed": sum(check["passed"] for check in checks),
        "failures": vr.failures,
        "checks": checks,
        "metric_mismatch_examples": metric_mismatches[:10],
        "prefix_selection_failure_examples": prefix_selection_failures[:10],
        "prefix_nesting_failure_examples": nested_prefix_failures[:10],
        "conditioning_failure_examples": conditioning_failures[:10],
        "shared_store_failure_examples": shared_store_failures[:10],
        "provenance_errors": provenance_errors,
        "artifact_manifest_sha256": stable_json_sha256(identities),
    }


def rows_by_method(rows: Sequence[dict[str, Any]]) -> dict[
        str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {
        method: [] for method in METHOD_ORDER}
    for row in rows:
        grouped[row["method"]].append(row)
    return grouped


def request_maps(rows: Sequence[dict[str, Any]]) -> dict[
        str, dict[tuple[str, str], dict[str, Any]]]:
    result: dict[str, dict[tuple[str, str], dict[str, Any]]] = {
        method: {} for method in METHOD_ORDER}
    for row in rows:
        key = (row["image_id"], row["question_id"])
        if key in result[row["method"]]:
            raise AnalysisError(f"duplicate method/request row: {row['method']}/{key}")
        result[row["method"]][key] = row
    return result


def summarize_methods(dataset: str, rows: Sequence[dict[str, Any]]) -> list[
        dict[str, Any]]:
    grouped = rows_by_method(rows)
    recomp_ttft = mean(row["end_to_end_ttft_ms"] for row in grouped["ReComp"])
    recomp_e2e = mean(row["e2e_ms"] for row in grouped["ReComp"])
    full_score = mean(row["score"] for row in grouped["FullLoad"])
    full_bytes = mean(row["ssd_read_bytes"] for row in grouped["FullLoad"])
    if recomp_ttft is None or recomp_e2e is None or full_score is None or not full_bytes:
        raise AnalysisError(f"{dataset}: missing baseline measurements")
    maps = request_maps(rows)
    ordered_keys = list(maps["ReComp"])
    result: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        method_rows = grouped[method]
        ttft = stats(row["end_to_end_ttft_ms"] for row in method_rows)
        core = stats(row["core_ttft_ms"] for row in method_rows
                     if row["core_ttft_ms"] is not None)
        e2e = stats(row["e2e_ms"] for row in method_rows)
        decode = stats(row["decode_ms"] for row in method_rows)
        ssd_bytes = stats(row["ssd_read_bytes"] for row in method_rows)
        ssd_ms = stats(row["ssd_read_ms"] for row in method_rows)
        selector = stats(row["selector_ms"] for row in method_rows)
        scatter = stats(row["scatter_ms"] for row in method_rows
                        if row["scatter_ms"] is not None)
        prefill = stats(row["prefill_ms"] for row in method_rows
                        if row["prefill_ms"] is not None)
        score = mean(row["score"] for row in method_rows)
        prediction_agreement = mean(
            float(maps[method][key]["prediction"] ==
                  maps["ReComp"][key]["prediction"])
            for key in ordered_keys)
        first_token_agreement = mean(
            float(maps[method][key]["first_token_id"] ==
                  maps["ReComp"][key]["first_token_id"])
            for key in ordered_keys)
        byte_mean = ssd_bytes["mean"] or 0.0
        result.append({
            "dataset": dataset,
            "method": method,
            "budget": METHOD_BUDGET[method],
            "n_images": DATASETS[dataset]["n_images"],
            "n_questions": len(method_rows),
            "score": score,
            "score_pct": 100.0 * score,
            "quality_delta_vs_fullload": score - full_score,
            "quality_delta_vs_fullload_pp": 100.0 * (score - full_score),
            "prediction_agreement_vs_recomp": prediction_agreement,
            "first_token_agreement_vs_recomp": first_token_agreement,
            "ttft_mean_ms": ttft["mean"],
            "ttft_p50_ms": ttft["p50"],
            "ttft_p95_ms": ttft["p95"],
            "core_ttft_mean_ms": core["mean"],
            "decode_mean_ms": decode["mean"],
            "e2e_mean_ms": e2e["mean"],
            "e2e_p50_ms": e2e["p50"],
            "e2e_p95_ms": e2e["p95"],
            "ttft_delta_vs_recomp_ms": ttft["mean"] - recomp_ttft,
            "ttft_reduction_vs_recomp_pct": (
                100.0 * (recomp_ttft - ttft["mean"]) / recomp_ttft),
            "e2e_delta_vs_recomp_ms": e2e["mean"] - recomp_e2e,
            "e2e_reduction_vs_recomp_pct": (
                100.0 * (recomp_e2e - e2e["mean"]) / recomp_e2e),
            "ssd_read_bytes_mean": byte_mean,
            "ssd_read_mb_mean": byte_mean / MB,
            "ssd_read_ms_mean": ssd_ms["mean"],
            "ssd_read_ms_p50": ssd_ms["p50"],
            "ssd_read_ms_p95": ssd_ms["p95"],
            "ssd_read_ratio_vs_fullload": (
                byte_mean / full_bytes if method != "ReComp" else None),
            "ssd_reduction_vs_fullload_pct": (
                100.0 * (full_bytes - byte_mean) / full_bytes
                if method != "ReComp" else None),
            "normal_preads_mean": mean(
                row["normal_kv_preads"] for row in method_rows),
            "separator_preads_mean": mean(
                row["separator_preads"] for row in method_rows),
            "total_preads_mean": mean(
                row["ssd_read_preads"] for row in method_rows),
            "ssd_chunk_units_mean": mean(
                row["ssd_read_chunk_units"] for row in method_rows),
            "selector_ms_mean": selector["mean"],
            "selector_ms_p50": selector["p50"],
            "selector_ms_p95": selector["p95"],
            "scatter_ms_mean": scatter["mean"],
            "prefill_ms_mean": prefill["mean"],
            "actual_selected_normal_chunk_fraction": mean(
                row["actual_selected_normal_chunk_fraction"]
                for row in method_rows
                if row["actual_selected_normal_chunk_fraction"] is not None),
            "actual_total_ssd_byte_ratio": mean(
                row["actual_total_ssd_byte_ratio"] for row in method_rows
                if row["actual_total_ssd_byte_ratio"] is not None),
            "selected_kv_ratio": mean(
                row["selected_kv_ratio"] for row in method_rows
                if row["selected_kv_ratio"] is not None),
            "importance_mass_coverage": mean(
                row["importance_mass_coverage"] for row in method_rows
                if row["importance_mass_coverage"] is not None),
        })
    return result


def paired_image_cluster_bootstrap(
        image_ids: Sequence[str], scores_a: Sequence[float],
        scores_b: Sequence[float] | None = None, *,
        n_resamples: int = 10_000, seed: int = 1234) -> dict[str, Any]:
    """Pure paired image-cluster bootstrap helper.

    ``image_ids`` and each score vector contain one element per evaluation
    question.  A resample draws image clusters with replacement and includes
    every question belonging to each selected image; duplicated images
    duplicate all their questions.  When ``scores_b`` is supplied, A, B and
    A-minus-B are computed from the exact same cluster draws.
    """
    if len(image_ids) != len(scores_a) or not image_ids:
        raise ValueError("image_ids and scores_a must be nonempty and aligned")
    if scores_b is not None and len(scores_b) != len(image_ids):
        raise ValueError("scores_b must align with image_ids")
    if int(n_resamples) < 1:
        raise ValueError("n_resamples must be positive")
    cluster_order = list(dict.fromkeys(str(image) for image in image_ids))
    cluster_index = {image: index for index, image in enumerate(cluster_order)}
    counts = np.zeros(len(cluster_order), dtype=np.int64)
    sum_a = np.zeros(len(cluster_order), dtype=np.float64)
    sum_b = (np.zeros(len(cluster_order), dtype=np.float64)
             if scores_b is not None else None)
    for row_index, image in enumerate(image_ids):
        index = cluster_index[str(image)]
        a = float(scores_a[row_index])
        if not math.isfinite(a):
            raise ValueError("scores_a contains a non-finite value")
        counts[index] += 1
        sum_a[index] += a
        if sum_b is not None:
            b = float(scores_b[row_index])
            if not math.isfinite(b):
                raise ValueError("scores_b contains a non-finite value")
            sum_b[index] += b
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(
        0, len(cluster_order),
        size=(int(n_resamples), len(cluster_order)), dtype=np.int32)
    denominators = counts[draws].sum(axis=1)
    dist_a = sum_a[draws].sum(axis=1) / denominators
    a_ci = tuple(float(value) for value in np.percentile(dist_a, [2.5, 97.5]))
    output: dict[str, Any] = {
        "a_mean": float(np.mean(np.asarray(scores_a, dtype=np.float64))),
        "a_ci95": a_ci,
        "estimate": float(np.mean(np.asarray(scores_a, dtype=np.float64))),
        "ci95_low": a_ci[0],
        "ci95_high": a_ci[1],
        "n_clusters": len(cluster_order),
        "n_observations": len(image_ids),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "cluster_unit": "image",
    }
    if sum_b is not None and scores_b is not None:
        dist_b = sum_b[draws].sum(axis=1) / denominators
        delta = dist_a - dist_b
        output.update({
            "b_mean": float(np.mean(np.asarray(scores_b, dtype=np.float64))),
            "b_ci95": tuple(float(value) for value in np.percentile(
                dist_b, [2.5, 97.5])),
            "delta_mean": float(np.mean(
                np.asarray(scores_a, dtype=np.float64)
                - np.asarray(scores_b, dtype=np.float64))),
            "delta_ci95": tuple(float(value) for value in np.percentile(
                delta, [2.5, 97.5])),
        })
        output["estimate"] = output["delta_mean"]
        output["ci95_low"] = output["delta_ci95"][0]
        output["ci95_high"] = output["delta_ci95"][1]
    return output


def cluster_bootstrap(dataset: str, rows: Sequence[dict[str, Any]],
                      n_resamples: int, seed: int) -> tuple[
                          list[dict[str, Any]], dict[tuple[str, str],
                                                    tuple[float, float]]]:
    """Return method/delta CIs using paired image-cluster draws."""
    maps = request_maps(rows)
    images = list(dict.fromkeys(row["image_id"] for row in rows))
    image_index = {image: i for i, image in enumerate(images)}
    sums = {method: np.zeros(len(images), dtype=np.float64)
            for method in METHOD_ORDER}
    counts = np.zeros(len(images), dtype=np.int64)
    for key, reference in maps["ReComp"].items():
        index = image_index[key[0]]
        counts[index] += 1
        for method in METHOD_ORDER:
            sums[method][index] += maps[method][key]["score"]
    if np.any(counts <= 0):
        raise AnalysisError(f"{dataset}: image with no evaluation question")

    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0, len(images), size=(int(n_resamples), len(images)), dtype=np.int32)
    sampled_count = counts[draws].sum(axis=1)
    distributions = {
        method: sums[method][draws].sum(axis=1) / sampled_count
        for method in METHOD_ORDER
    }
    ci_map: dict[tuple[str, str], tuple[float, float]] = {}
    output: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        distribution = distributions[method]
        lo, hi = (float(value) for value in np.percentile(
            distribution, [2.5, 97.5]))
        point = mean(row["score"] for row in rows if row["method"] == method)
        output.append({
            "dataset": dataset,
            "statistic": "score",
            "method": method,
            "reference": "",
            "estimate": point,
            "ci95_low": lo,
            "ci95_high": hi,
            "cluster_unit": "image",
            "n_clusters": len(images),
            "n_resamples": n_resamples,
            "seed": seed,
        })
        ci_map[(method, "score")] = (lo, hi)
    for method in METHOD_ORDER:
        for reference in ("FullLoad", "ReComp"):
            if method == reference:
                continue
            distribution = distributions[method] - distributions[reference]
            lo, hi = (float(value) for value in np.percentile(
                distribution, [2.5, 97.5]))
            point = mean(
                maps[method][key]["score"] - maps[reference][key]["score"]
                for key in maps[method])
            output.append({
                "dataset": dataset,
                "statistic": "paired_score_delta",
                "method": method,
                "reference": reference,
                "estimate": point,
                "ci95_low": lo,
                "ci95_high": hi,
                "cluster_unit": "image",
                "n_clusters": len(images),
                "n_resamples": n_resamples,
                "seed": seed,
            })
            ci_map[(method, reference)] = (lo, hi)
    return output, ci_map


def disagreement_analysis(dataset: str, rows: Sequence[dict[str, Any]]) -> list[
        dict[str, Any]]:
    maps = request_maps(rows)
    keys = list(maps["FullLoad"])
    output: list[dict[str, Any]] = []
    for method in PREFIX_METHODS:
        full_better = prefix_better = equal = 0
        full_correct_prefix_wrong = prefix_correct_full_wrong = 0
        for key in keys:
            full = float(maps["FullLoad"][key]["score"])
            prefix = float(maps[method][key]["score"])
            if full > prefix + SCORE_TOLERANCE:
                full_better += 1
            elif prefix > full + SCORE_TOLERANCE:
                prefix_better += 1
            else:
                equal += 1
            if full >= 1.0 - SCORE_TOLERANCE and prefix <= SCORE_TOLERANCE:
                full_correct_prefix_wrong += 1
            if prefix >= 1.0 - SCORE_TOLERANCE and full <= SCORE_TOLERANCE:
                prefix_correct_full_wrong += 1
        is_binary = DATASETS[dataset]["metric"] == "gqa"
        p_value = (exact_binomial_two_sided(
            full_correct_prefix_wrong, prefix_correct_full_wrong)
            if is_binary else None)
        output.append({
            "dataset": dataset,
            "method": method,
            "reference": "FullLoad",
            "n": len(keys),
            "fullload_score_greater": full_better,
            "prefix_score_greater": prefix_better,
            "equal_score": equal,
            "fullload_correct_prefix_wrong": full_correct_prefix_wrong,
            "prefix_correct_fullload_wrong": prefix_correct_full_wrong,
            "mcnemar_exact_p": p_value,
            "mcnemar_applicable": is_binary,
            "note": ("exact binary McNemar" if is_binary else
                     "soft-score dataset; paired cluster bootstrap is primary"),
        })
    return output


def category_analysis(dataset: str, rows: Sequence[dict[str, Any]]) -> list[
        dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["question_type"], row["method"])].append(row)
    result: list[dict[str, Any]] = []
    for category in CATEGORY_ORDER:
        full_rows = grouped.get((category, "FullLoad"), [])
        if not full_rows:
            continue
        full_score = mean(row["score"] for row in full_rows)
        for method in METHOD_ORDER:
            method_rows = grouped[(category, method)]
            score = mean(row["score"] for row in method_rows)
            result.append({
                "dataset": dataset,
                "category_source": "question-text regex heuristic",
                "category_rule": CATEGORY_RULES[category],
                "dataset_interpretation":
                    CATEGORY_DATASET_INTERPRETATION[dataset],
                "question_type": category,
                "method": method,
                "n_questions": len(method_rows),
                "score": score,
                "score_pct": 100.0 * score,
                "quality_delta_vs_fullload": score - full_score,
                "quality_delta_vs_fullload_pp": 100.0 * (score - full_score),
            })
    return result


def first_profile_number(profile: Mapping[str, Any], names: Sequence[str]) -> float | None:
    """Return the first *alias* present, preferring an authoritative flat key.

    Alias order is semantic here.  In particular, the runner publishes a flat
    ``repack_ms`` equal to materialization plus the lower-level
    ``kv_repack_ms`` component.  Searching for all aliases at once follows
    dictionary traversal order and can therefore silently select the nested
    component before the requested aggregate.  Check aliases one at a time,
    and prefer a top-level value before falling back to legacy nested shapes.
    """
    for name in names:
        direct = dotted(profile, name)
        if direct not in (None, ""):
            return number(direct, name)
        values = recursive_values(profile, {name})
        if values:
            return number(values[0], name)
    return None


def persistence_summary(dataset: str,
                        profiles: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    fields: dict[str, tuple[str, ...]] = {
        "saliency_ms": ("saliency_ms", "vision_saliency_ms",
                        "saliency_postprocess_ms"),
        "permutation_ms": ("permutation_ms", "permute_ms"),
        "repack_ms": ("repack_ms", "kv_repack_ms"),
        "ssd_write_ms": ("ssd_write_ms", "buffered_write_ms", "write_ms"),
        "fsync_ms": ("fsync_ms",),
        "total_ingestion_ms": ("total_ingestion_ms",),
        "mapping_metadata_bytes": ("mapping_metadata_bytes",
                                   "metadata_bytes"),
        "visual_kv_bytes": ("visual_kv_bytes", "full_visual_kv_bytes"),
    }
    output: list[dict[str, Any]] = []
    for metric, aliases in fields.items():
        values = [first_profile_number(profile, aliases)
                  for profile in profiles]
        values = [value for value in values if value is not None]
        summary = stats(values)
        output.append({
            "dataset": dataset,
            "metric": metric,
            "unit": "bytes" if metric.endswith("bytes") else "ms/image",
            "n_images_observed": len(values),
            "mean": summary["mean"],
            "p50": summary["p50"],
            "p95": summary["p95"],
            "min": summary["min"],
            "max": summary["max"],
        })
    return output


def importance_coverage(dataset: str, rows: Sequence[dict[str, Any]]) -> list[
        dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for method in PREFIX_METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        # Coverage is image/layout invariant.  One request per image avoids
        # weighting an image by its question count if future workloads differ.
        per_image: dict[str, float] = {}
        for row in method_rows:
            value = row["importance_mass_coverage"]
            if value is not None:
                per_image.setdefault(row["image_id"], value)
                if not close(per_image[row["image_id"]], value, 1e-7):
                    raise AnalysisError(
                        f"coverage changed across requests: {dataset}/"
                        f"{row['image_id']}/{method}")
        summary = stats(per_image.values())
        result.append({
            "dataset": dataset,
            "method": method,
            "nominal_budget": METHOD_BUDGET[method],
            "n_images_observed": len(per_image),
            "importance_mass_coverage_mean": summary["mean"],
            "importance_mass_coverage_p50": summary["p50"],
            "importance_mass_coverage_p95": summary["p95"],
        })
    return result


def summary_map(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["method"]): row for row in rows}


def dataset_verdict(summary_rows: Sequence[dict[str, Any]],
                    ci_map: Mapping[tuple[str, str],
                                    tuple[float, float]]) -> dict[str, Any]:
    table = summary_map(summary_rows)
    p25, p45 = table["Prefix25"], table["Prefix45"]
    p25_efficiency = (
        p25["ttft_reduction_vs_recomp_pct"] > 0
        and p25["ssd_reduction_vs_fullload_pct"] > 0
        and p25["ttft_mean_ms"] <= p45["ttft_mean_ms"]
        and p25["ssd_read_bytes_mean"] <= p45["ssd_read_bytes_mean"])
    p45_significant_loss = ci_map[("Prefix45", "FullLoad")][1] < 0.0
    p45_quality_point = (
        p45["ttft_reduction_vs_recomp_pct"] > 0
        and p45["ssd_reduction_vs_fullload_pct"] > 0
        and p45["score"] >= p25["score"] - SCORE_TOLERANCE
        and not p45_significant_loss)
    if p25_efficiency and p45_quality_point:
        verdict = "SUPPORTED"
    elif p25_efficiency or p45_quality_point:
        verdict = "PARTIALLY SUPPORTED"
    else:
        verdict = "NOT SUPPORTED"
    return {
        "verdict": verdict,
        "prefix25_efficiency_point": p25_efficiency,
        "prefix45_quality_point": p45_quality_point,
        "prefix45_significant_quality_loss_95ci": p45_significant_loss,
        "rule": (
            "SUPPORTED iff Prefix25 reduces E2E-TTFT and SSD bytes, is no "
            "slower/heavier than Prefix45, and Prefix45 also reduces TTFT/SSD, "
            "scores at least as high as Prefix25, and its paired image-cluster "
            "95% CI versus FullLoad does not exclude zero on the negative side"
        ),
    }


def fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}{suffix}"


def markdown_method_table(summary_rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| Method | Score | Δ vs FullLoad | TTFT mean / p50 / p95 | "
        "TTFT↓ vs ReComp | E2E | SSD MB/req | SSD ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {fmt(row['score_pct'], 2, '%')} | "
            f"{fmt(row['quality_delta_vs_fullload_pp'], 2, ' pp')} | "
            f"{fmt(row['ttft_mean_ms'])} / {fmt(row['ttft_p50_ms'])} / "
            f"{fmt(row['ttft_p95_ms'])} ms | "
            f"{fmt(row['ttft_reduction_vs_recomp_pct'], 2, '%')} | "
            f"{fmt(row['e2e_mean_ms'])} ms | "
            f"{fmt(row['ssd_read_mb_mean'])} | "
            f"{fmt(None if row['ssd_read_ratio_vs_fullload'] is None else 100 * row['ssd_read_ratio_vs_fullload'], 2, '%')} |")
    return "\n".join(lines)


def dataset_readme(dataset: str, analysis_config: Mapping[str, Any],
                   summary_rows: Sequence[dict[str, Any]],
                   statistical_rows: Sequence[dict[str, Any]],
                   disagreement_rows: Sequence[dict[str, Any]],
                   category_rows: Sequence[dict[str, Any]],
                   persistence_rows: Sequence[dict[str, Any]],
                   coverage_rows: Sequence[dict[str, Any]],
                   validation: Mapping[str, Any],
                   verdict: Mapping[str, Any]) -> str:
    spec = DATASETS[dataset]
    table = summary_map(summary_rows)
    p25, p45, full = table["Prefix25"], table["Prefix45"], table["FullLoad"]
    ci = {(row["method"], row["reference"]): row
          for row in statistical_rows
          if row["statistic"] == "paired_score_delta"}
    persistence = {row["metric"]: row for row in persistence_rows}
    sensitive = {}
    for method in PREFIX_METHODS:
        candidates = [row for row in category_rows
                      if row["method"] == method]
        sensitive[method] = min(
            candidates, key=lambda row: row["quality_delta_vs_fullload"])
    return f"""# {spec['label']} ImageOnly-Repack generalization

## Scope and immutable workload

- Index: `{spec['index']}`
- Index SHA256: `{spec['index_sha256']}`
- Evaluation workload SHA256: `{spec['workload_sha256']}`
- Workload: {spec['n_images']} images / {spec['n_questions']} questions; fixed
  slice `questions[{spec['skip']}:{spec['skip'] + spec['questions']}]`
- Calibration questions used by the new layout: **0**.  The skipped question is
  retained only to match the old evaluation IDs.
- Main latency: `end_to_end_ttft_ms`; `core_ttft_ms` is diagnostic only.

## Main results

{markdown_method_table(summary_rows)}

Prefix25 changes quality by {fmt(p25['quality_delta_vs_fullload_pp'], 2, ' pp')}
versus same-layout FullLoad (paired image-cluster 95% CI
[{fmt(100 * ci[('Prefix25', 'FullLoad')]['ci95_low'], 2)},
 {fmt(100 * ci[('Prefix25', 'FullLoad')]['ci95_high'], 2)}] pp), reduces
E2E-TTFT by {fmt(p25['ttft_reduction_vs_recomp_pct'], 2, '%')} versus ReComp,
and reduces SSD bytes by {fmt(p25['ssd_reduction_vs_fullload_pct'], 2, '%')}
versus FullLoad.

Prefix45 changes quality by {fmt(p45['quality_delta_vs_fullload_pp'], 2, ' pp')}
(95% CI [{fmt(100 * ci[('Prefix45', 'FullLoad')]['ci95_low'], 2)},
{fmt(100 * ci[('Prefix45', 'FullLoad')]['ci95_high'], 2)}] pp), reduces E2E-TTFT
by {fmt(p45['ttft_reduction_vs_recomp_pct'], 2, '%')}, and reduces SSD bytes by
{fmt(p45['ssd_reduction_vs_fullload_pct'], 2, '%')}.

## FullLoad numerical sanity

- Prediction agreement with ReComp: {fmt(100 * full['prediction_agreement_vs_recomp'], 2, '%')}
- First-token agreement with ReComp: {fmt(100 * full['first_token_agreement_vs_recomp'], 2, '%')}
- Score delta FullLoad−ReComp: {fmt(full['quality_delta_vs_fullload_pp'] - table['ReComp']['quality_delta_vs_fullload_pp'], 2, ' pp')}

This separates the physical-layout numerical effect (ReComp → repacked
FullLoad) from the partial-loading effect (repacked FullLoad → Prefix).

## Persistence and layout coverage

- Saliency: {fmt(persistence['saliency_ms']['mean'])} ms/image
- Permutation: {fmt(persistence['permutation_ms']['mean'])} ms/image
- Repack: {fmt(persistence['repack_ms']['mean'])} ms/image
- Buffered SSD write: {fmt(persistence['ssd_write_ms']['mean'])} ms/image
- `fsync`: {fmt(persistence['fsync_ms']['mean'])} ms/image
- Mapping metadata: {fmt((persistence['mapping_metadata_bytes']['mean'] or 0) / 1e6)} MB/image
- Visual KV: {fmt((persistence['visual_kv_bytes']['mean'] or 0) / 1e6)} MB/image
- Prefix25/45 saliency-mass coverage: {fmt(100 * (coverage_rows[0]['importance_mass_coverage_mean'] or 0), 2, '%')} /
  {fmt(100 * (coverage_rows[1]['importance_mass_coverage_mean'] or 0), 2, '%')}

Persistence is cache construction overhead and is not part of cache-hit TTFT.

## Category analysis

Categories use the existing ordered question-text regex heuristic: yes/no,
color, count, spatial, material/attribute, object, then fallback other. They
are not official dataset metadata. {CATEGORY_DATASET_INTERPRETATION[dataset]}
Prefix25's most negative category delta is
`{sensitive['Prefix25']['question_type']}`
({fmt(sensitive['Prefix25']['quality_delta_vs_fullload_pp'], 2, ' pp')},
n={sensitive['Prefix25']['n_questions']}); Prefix45's is
`{sensitive['Prefix45']['question_type']}`
({fmt(sensitive['Prefix45']['quality_delta_vs_fullload_pp'], 2, ' pp')}).
See `category_analysis.csv` and `disagreement_analysis.csv` for all counts.

## Statistical policy and metric caveat

Quality intervals use {analysis_config['bootstrap_resamples']:,} seeded paired
image-cluster resamples.  GQA also receives exact McNemar testing.  VQAv2 and
TextVQA retain the repository's normalized 10-answer consensus score and use
paired soft-score deltas rather than thresholded McNemar as their primary test.
The repository VQA normalizer/scorer is not byte-for-byte the official
leave-one-annotator-out evaluator; it is retained to preserve question-level
comparability with the previous cross-dataset experiment.  The repository GQA
metric also accepts a prediction beginning with the complete normalized gold
token sequence.

## Verdict

**{verdict['verdict']}** under the explicit rule recorded in `config.json`.
Automated validation: {validation['n_passed']}/{validation['n_checks']} checks
passed.
"""


def csv_ready_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (list, dict, tuple)):
                converted[key] = json.dumps(value, ensure_ascii=False,
                                            separators=(",", ":"))
            else:
                converted[key] = value
        result.append(converted)
    return result


def analyze_dataset(dataset: str, run_root: Path, output_dir: Path,
                    n_resamples: int, seed: int) -> dict[str, Any]:
    run_dir = run_root / dataset
    ordered_workload, expected_by_key = expected_workload(dataset)
    config, source_rows, profiles, identities = load_dataset_artifacts(
        dataset, run_dir, ordered_workload)
    source_tree_before = tree_sha256(run_dir)
    rows = canonicalize_rows(dataset, source_rows, expected_by_key)
    validation = validate_dataset(
        dataset, config, source_rows, rows, ordered_workload, profiles,
        identities)
    if not validation["passed"]:
        raise AnalysisError(
            f"{dataset} validation failed: {validation['failures']}")

    summary_rows = summarize_methods(dataset, rows)
    statistical_rows, ci_map = cluster_bootstrap(
        dataset, rows, n_resamples, seed)
    disagreement_rows = disagreement_analysis(dataset, rows)
    category_rows = category_analysis(dataset, rows)
    persistence_rows = persistence_summary(dataset, profiles)
    coverage_rows = importance_coverage(dataset, rows)
    verdict = dataset_verdict(summary_rows, ci_map)
    source_tree_after = tree_sha256(run_dir)
    if source_tree_before != source_tree_after:
        raise AnalysisError(f"source run changed during analysis: {run_dir}")

    qcounts = Counter(row["image_id"] for row in ordered_workload)
    analysis_config = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "source_run_dir": str(run_dir.resolve()),
        "source_run_config_sha256": sha256_file(run_dir / "config.json"),
        "source_run_tree_sha256": source_tree_before,
        "source_artifact_manifest_sha256": stable_json_sha256(identities),
        "index": str(DATASETS[dataset]["index"]),
        "index_sha256": DATASETS[dataset]["index_sha256"],
        "evaluation_workload_sha256": DATASETS[dataset]["workload_sha256"],
        "n_images": DATASETS[dataset]["n_images"],
        "n_questions": DATASETS[dataset]["n_questions"],
        "questions_per_image": {
            "min": min(qcounts.values()),
            "mean": sum(qcounts.values()) / len(qcounts),
            "max": max(qcounts.values()),
        },
        "evaluation_slice": {
            "skip": DATASETS[dataset]["skip"],
            "questions": DATASETS[dataset]["questions"],
        },
        "metric": DATASETS[dataset]["metric"],
        "metric_implementation": "mmimpress.dataset.METRICS",
        "main_ttft": "end_to_end_ttft_ms",
        "diagnostic_ttft": "core_ttft_ms",
        "bootstrap": "paired image-cluster",
        "bootstrap_resamples": n_resamples,
        "bootstrap_seed": seed,
        "category_source": "scripts/12_analysis.py-compatible regex heuristic",
        "category_rule_order": list(CATEGORY_ORDER),
        "category_rules": CATEGORY_RULES,
        "category_dataset_interpretation":
            CATEGORY_DATASET_INTERPRETATION[dataset],
        "verdict": verdict,
        "runner_config": config,
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "config.json", analysis_config)
    raw_rows = [{key: value for key, value in row.items()
                 if key != "selected_chunk_ids_per_layer"}
                | {"selected_chunk_ids_per_layer":
                   row["selected_chunk_ids_per_layer"]}
                for row in rows]
    write_jsonl(output_dir / "raw.jsonl", raw_rows)
    write_csv(output_dir / "per_request.csv", csv_ready_rows(raw_rows))
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "statistical_analysis.csv", statistical_rows)
    write_csv(output_dir / "disagreement_analysis.csv", disagreement_rows)
    write_csv(output_dir / "category_analysis.csv", category_rows)
    write_csv(output_dir / "persistence_summary.csv", persistence_rows)
    write_csv(output_dir / "importance_coverage.csv", coverage_rows)
    write_json(output_dir / "validation.json", validation)
    (output_dir / "README.md").write_text(dataset_readme(
        dataset, analysis_config, summary_rows, statistical_rows,
        disagreement_rows, category_rows, persistence_rows, coverage_rows,
        validation, verdict))
    return {
        "dataset": dataset,
        "summary": summary_rows,
        "statistics": statistical_rows,
        "disagreement": disagreement_rows,
        "categories": category_rows,
        "persistence": persistence_rows,
        "coverage": coverage_rows,
        "validation": validation,
        "verdict": verdict,
        "config": analysis_config,
        "source_tree_before": source_tree_before,
        "source_tree_after": source_tree_after,
    }


def cross_dataset_rows(results: Sequence[dict[str, Any]]) -> list[
        dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for result in results:
        dataset = result["dataset"]
        table = summary_map(result["summary"])
        output.append({
            "dataset": dataset,
            "full_load_score": table["FullLoad"]["score"],
            "prefix25_score": table["Prefix25"]["score"],
            "prefix25_delta_vs_fullload":
                table["Prefix25"]["quality_delta_vs_fullload"],
            "prefix45_score": table["Prefix45"]["score"],
            "prefix45_delta_vs_fullload":
                table["Prefix45"]["quality_delta_vs_fullload"],
            "prefix25_ttft_reduction_vs_recomp_pct":
                table["Prefix25"]["ttft_reduction_vs_recomp_pct"],
            "prefix45_ttft_reduction_vs_recomp_pct":
                table["Prefix45"]["ttft_reduction_vs_recomp_pct"],
            "prefix25_e2e_reduction_vs_recomp_pct":
                table["Prefix25"]["e2e_reduction_vs_recomp_pct"],
            "prefix45_e2e_reduction_vs_recomp_pct":
                table["Prefix45"]["e2e_reduction_vs_recomp_pct"],
            "prefix25_ssd_reduction_vs_fullload_pct":
                table["Prefix25"]["ssd_reduction_vs_fullload_pct"],
            "prefix45_ssd_reduction_vs_fullload_pct":
                table["Prefix45"]["ssd_reduction_vs_fullload_pct"],
            "fullload_ttft_delta_vs_recomp_ms":
                table["FullLoad"]["ttft_delta_vs_recomp_ms"],
            "verdict": result["verdict"]["verdict"],
        })
    return output


def cross_verdict(results: Sequence[dict[str, Any]]) -> str:
    verdicts = [result["verdict"]["verdict"] for result in results]
    if all(value == "SUPPORTED" for value in verdicts):
        return "SUPPORTED"
    if any(value != "NOT SUPPORTED" for value in verdicts):
        return "PARTIALLY SUPPORTED"
    return "NOT SUPPORTED"


def root_readme(results: Sequence[dict[str, Any]],
                cross_rows: Sequence[dict[str, Any]],
                validation: Mapping[str, Any], verdict: str,
                n_resamples: int, seed: int) -> str:
    by_dataset = {result["dataset"]: result for result in results}
    tables = {dataset: summary_map(result["summary"])
              for dataset, result in by_dataset.items()}
    lines = [
        "# ImageOnly-Repack cross-dataset generalization",
        "",
        "This analysis merges the immutable image-at-a-time serving artifacts. "
        "No inference is run here, and no prior result is modified.",
        "",
        "## Frozen workloads",
        "",
        "| Dataset | Images | Questions | q/image | Index SHA256 | Workload SHA256 |",
        "|---|---:|---:|---:|---|---|",
    ]
    for dataset in DATASETS:
        spec = DATASETS[dataset]
        lines.append(
            f"| {spec['label']} | {spec['n_images']} | {spec['n_questions']} | "
            f"{spec['questions']} | `{spec['index_sha256']}` | "
            f"`{spec['workload_sha256']}` |")
    lines += [
        "",
        "All layouts use zero calibration questions.  The original per-image "
        "question offset remains fixed only so the evaluation IDs exactly match "
        "the previous Static+Diverse generalization workload.",
        "",
        "## Cross-dataset result",
        "",
        "| Dataset | FullLoad score | P25 score | Δ P25 | P45 score | Δ P45 | "
        "P25 TTFT↓ vs ReComp | P45 TTFT↓ vs ReComp | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in cross_rows:
        lines.append(
            f"| {DATASETS[row['dataset']]['label']} | "
            f"{fmt(100 * row['full_load_score'], 2, '%')} | "
            f"{fmt(100 * row['prefix25_score'], 2, '%')} | "
            f"{fmt(100 * row['prefix25_delta_vs_fullload'], 2, ' pp')} | "
            f"{fmt(100 * row['prefix45_score'], 2, '%')} | "
            f"{fmt(100 * row['prefix45_delta_vs_fullload'], 2, ' pp')} | "
            f"{fmt(row['prefix25_ttft_reduction_vs_recomp_pct'], 2, '%')} | "
            f"{fmt(row['prefix45_ttft_reduction_vs_recomp_pct'], 2, '%')} | "
            f"{row['verdict']} |")

    # Largest method/dataset loss, then its most sensitive heuristic category.
    losses = []
    for result in results:
        table = summary_map(result["summary"])
        for method in PREFIX_METHODS:
            losses.append((table[method]["quality_delta_vs_fullload"],
                           result["dataset"], method))
    largest_delta, largest_dataset, largest_method = min(losses)
    category_candidates = [
        row for row in by_dataset[largest_dataset]["categories"]
        if row["method"] == largest_method]
    sensitive = min(category_candidates,
                    key=lambda row: row["quality_delta_vs_fullload"])
    p25_consistent = all(
        row["prefix25_ttft_reduction_vs_recomp_pct"] > 0
        and row["prefix25_ssd_reduction_vs_fullload_pct"] > 0
        for row in cross_rows)
    p45_quality_all = all(
        result["verdict"]["prefix45_quality_point"] for result in results)
    full_faster = [
        row["dataset"] for row in cross_rows
        if row["fullload_ttft_delta_vs_recomp_ms"] < 0]
    if len(full_faster) == len(DATASETS):
        caching_only_answer = (
            "FullLoad가 세 dataset 모두에서 ReComp보다 빨랐으므로, 이 "
            "측정에서는 SSD KV caching 자체도 latency 이득을 냈습니다. "
            "Prefix loading의 추가 이득은 별도로 해석해야 합니다."
        )
        caching_conclusion = (
            "FullLoad beat ReComp on every dataset, so the measured benefit "
            "cannot be attributed exclusively to partial loading"
        )
    elif full_faster:
        labels = ", ".join(DATASETS[name]["label"] for name in full_faster)
        caching_only_answer = (
            f"dataset에 따라 다릅니다. FullLoad는 {labels}에서만 "
            "ReComp보다 빨랐으므로 SSD caching만으로 보편적인 이득을 "
            "설명할 수 없고 Prefix 결과를 함께 봐야 합니다."
        )
        caching_conclusion = (
            f"FullLoad beat ReComp only on {labels}, so SSD caching alone was "
            "not a cross-dataset explanation of the benefit"
        )
    else:
        caching_only_answer = (
            "아닙니다. FullLoad가 어느 dataset에서도 ReComp보다 빠르지 "
            "않았으므로, 이 측정에서 latency 이득에는 partial loading이 "
            "필요했습니다."
        )
        caching_conclusion = (
            "No dataset made FullLoad faster than ReComp, so partial loading "
            "was necessary for the measured latency benefit"
        )

    lines += [
        "",
        "## Direct answers to Q1–Q10",
        "",
    ]
    for number_, dataset in enumerate(DATASETS, 1):
        table = tables[dataset]
        lines += [
            f"### Q{number_}. {DATASETS[dataset]['label']}",
            "",
            f"Prefix25: quality Δ {fmt(table['Prefix25']['quality_delta_vs_fullload_pp'], 2, ' pp')}, "
            f"TTFT reduction {fmt(table['Prefix25']['ttft_reduction_vs_recomp_pct'], 2, '%')}, "
            f"SSD reduction {fmt(table['Prefix25']['ssd_reduction_vs_fullload_pct'], 2, '%')}. "
            f"Prefix45: quality Δ {fmt(table['Prefix45']['quality_delta_vs_fullload_pp'], 2, ' pp')}, "
            f"TTFT reduction {fmt(table['Prefix45']['ttft_reduction_vs_recomp_pct'], 2, '%')}, "
            f"SSD reduction {fmt(table['Prefix45']['ssd_reduction_vs_fullload_pct'], 2, '%')}.",
            "",
        ]
    lines += [
        "### Q4. Prefix25 efficiency trade-off",
        "",
        ("세 dataset 모두에서 유지됩니다." if p25_consistent else
         "세 dataset 모두에서 유지되지는 않았습니다."),
        "",
        "### Q5. Prefix45 quality-oriented point",
        "",
        ("세 dataset 모두에서 동작합니다." if p45_quality_all else
         "세 dataset 모두에서 통계적으로 quality-oriented point라고 보기는 "
         "어렵습니다. dataset별 CI와 verdict를 확인해야 합니다."),
        "",
        "### Q6. 가장 큰 quality degradation",
        "",
        f"{DATASETS[largest_dataset]['label']}의 {largest_method}: "
        f"{fmt(100 * largest_delta, 2, ' pp')} vs FullLoad.",
        "",
        "### Q7. 가장 민감한 category",
        "",
        f"동일 dataset/method에서 `{sensitive['question_type']}` heuristic "
        f"category가 {fmt(sensitive['quality_delta_vs_fullload_pp'], 2, ' pp')}로 "
        "가장 낮았습니다. 이는 official category metadata가 아닌 question-text "
        "regex 분석입니다.",
        "",
        "### Q8. FullLoad가 ReComp보다 빠른 dataset",
        "",
        (", ".join(DATASETS[name]["label"] for name in full_faster)
         if full_faster else "없습니다."),
        "",
        "### Q9. SSD caching만으로 충분한가?",
        "",
        caching_only_answer,
        "",
        "### Q10. Cross-dataset generalization verdict",
        "",
        f"**{verdict}**. Dataset별 판정을 하나의 평균으로 덮지 않았습니다.",
        "",
        "## Statistical and measurement policy",
        "",
        f"Quality uses {n_resamples:,} paired image-cluster bootstrap resamples "
        f"with seed {seed}. Main latency is server-side `end_to_end_ttft_ms`; "
        "page-cache conditioning is outside that timer. The previous "
        "Static+Diverse artifacts lack the same final E2E boundary, so their "
        "latency is not directly compared. Their frozen IDs, quality, and I/O "
        "remain suitable supplementary references.",
        "",
        "VQAv2/TextVQA scores retain the repository consensus implementation for "
        "exact old/new comparability; this implementation is not byte-for-byte "
        "the official leave-one-annotator-out evaluator. Category labels are "
        "heuristic because the frozen indexes contain no official type metadata.",
        "",
        "## Paper-ready conclusion",
        "",
    ]
    direction = ("consistently" if p25_consistent else "not consistently")
    quality = ("served as a quality-oriented operating point across all three "
               "datasets" if p45_quality_all else
               "did not satisfy the quality-oriented criterion on every dataset")
    lines += [
        ("Across GQA, VQAv2, and TextVQA, image-only importance-aware Visual "
         f"KV repacking {direction} enabled selective sequential SSD loading "
         "without query-dependent cache scoring. Prefix25 provided the most "
         f"aggressive efficiency point, while Prefix45 {quality}. "
         f"{caching_conclusion}. The cross-dataset verdict is reported from the measured "
         f"per-dataset evidence as {verdict}.")
    ]
    lines += [
        "",
        f"Automated validation: {validation['n_passed']}/"
        f"{validation['n_checks']} checks passed.",
        "",
    ]
    return "\n".join(lines)


def fsync_tree(path: Path) -> None:
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        fd = os.open(item, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    for directory in sorted(
            (p for p in path.rglob("*") if p.is_dir()),
            key=lambda p: len(p.parts), reverse=True):
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--results-root", type=Path,
                        default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=1234)
    parser.add_argument("--protection-manifest", type=Path, default=None,
                        help="optional pre-experiment file-level SHA256 manifest")
    args = parser.parse_args()

    datasets = tuple(value.strip() for value in args.datasets.split(",")
                     if value.strip())
    if datasets != tuple(DATASETS):
        raise AnalysisError(
            "final analysis requires datasets in canonical order: "
            + ",".join(DATASETS))
    if args.bootstrap_resamples != 10_000:
        raise AnalysisError("final analysis requires exactly 10,000 resamples")
    run_root = args.run_root.resolve()
    results_root = args.results_root.resolve()
    if not run_root.is_dir():
        raise AnalysisError(f"missing run root: {run_root}")
    if results_root.exists():
        raise AnalysisError(
            f"result destination already exists (fail-close): {results_root}")
    if run_root == results_root or run_root in results_root.parents:
        raise AnalysisError("results path may not contain or equal the source run")

    manifest = load_protection_manifest(
        args.protection_manifest.resolve() if args.protection_manifest else None)
    protection_before = check_protection_manifest(manifest)
    if protection_before["provided"] and not protection_before["passed"]:
        raise AnalysisError(
            "protection manifest already differs before analysis: "
            f"missing={len(protection_before['missing'])}, "
            f"changed={len(protection_before['changed'])}")

    protected_paths = tuple(
        ROOT / path for path in (
            "results/generalization_summary", "results/gqa_large",
            "results/vqav2", "results/textvqa", "results/image_only_repack",
            "results/image_only_repack_budget_sweep",
            "results/image_only_repack_budget_sweep_with_recomp",
            "results/visdial_turn1_piggyback_e2e_ttft",
            "results/visdial_cache_hit_analysis",
        ))
    protected_before = {str(path): tree_sha256(path) for path in protected_paths}

    results_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{results_root.name}.staging-", dir=results_root.parent))
    completed = False
    try:
        results = [
            analyze_dataset(dataset, run_root, staging / dataset,
                            args.bootstrap_resamples, args.bootstrap_seed)
            for dataset in datasets
        ]
        all_summary = [row for result in results for row in result["summary"]]
        all_statistics = [row for result in results
                          for row in result["statistics"]]
        all_categories = [row for result in results
                          for row in result["categories"]]
        cross_rows = cross_dataset_rows(results)
        verdict = cross_verdict(results)

        write_csv(staging / "summary_all.csv", all_summary)
        write_csv(staging / "cross_dataset_table.csv", cross_rows)
        write_csv(staging / "statistical_analysis.csv", all_statistics)
        write_csv(staging / "category_analysis.csv", all_categories)
        write_csv(staging / "fig_accuracy_ttft.csv", [{
            "dataset": row["dataset"], "method": row["method"],
            "score": row["score"], "ttft_ms": row["ttft_mean_ms"],
        } for row in all_summary])
        write_csv(staging / "fig_accuracy_ssd.csv", [{
            "dataset": row["dataset"], "method": row["method"],
            "score": row["score"], "ssd_read_mb": row["ssd_read_mb_mean"],
        } for row in all_summary])
        write_csv(staging / "fig_ttft_reduction.csv", [{
            "dataset": row["dataset"], "method": row["method"],
            "ttft_reduction_vs_recomp_pct":
                row["ttft_reduction_vs_recomp_pct"],
        } for row in all_summary if row["method"] != "ReComp"])
        write_csv(staging / "fig_ssd_reduction.csv", [{
            "dataset": row["dataset"], "method": row["method"],
            "ssd_reduction_vs_fullload_pct":
                row["ssd_reduction_vs_fullload_pct"],
        } for row in all_summary if row["method"] in CACHE_METHODS])

        protected_after = {
            str(path): tree_sha256(path) for path in protected_paths}
        protection_after = check_protection_manifest(manifest)
        protected_unchanged = protected_before == protected_after
        manifest_unchanged = (
            not protection_after["provided"] or protection_after["passed"])
        source_unchanged = all(
            result["source_tree_before"] == result["source_tree_after"]
            for result in results)

        # Reflect the optional file-level protection check in every dataset
        # validation, as well as in the cross-dataset validation.
        for result in results:
            checks = result["validation"]["checks"]
            checks.extend((
                {
                    "name": "source_run_tree_unchanged",
                    "passed": result["source_tree_before"] ==
                              result["source_tree_after"],
                    "observed": result["source_tree_after"],
                    "expected": result["source_tree_before"],
                },
                {
                    "name": "built_in_protected_results_unchanged",
                    "passed": protected_unchanged,
                    "observed": protected_after,
                    "expected": protected_before,
                },
                {
                    "name": "optional_protection_manifest_exact",
                    "passed": manifest_unchanged,
                    "observed": protection_after,
                    "expected": ("exact match" if manifest is not None
                                 else "not provided; built-in protection used"),
                },
            ))
            result["validation"]["n_checks"] = len(checks)
            result["validation"]["n_passed"] = sum(
                check["passed"] for check in checks)
            result["validation"]["passed"] = all(
                check["passed"] for check in checks)
            result["validation"]["failures"] = [
                check["name"] for check in checks if not check["passed"]]
            ds_dir = staging / result["dataset"]
            write_json(ds_dir / "validation.json", result["validation"])
            (ds_dir / "README.md").write_text(dataset_readme(
                result["dataset"], result["config"], result["summary"],
                result["statistics"], result["disagreement"],
                result["categories"], result["persistence"],
                result["coverage"], result["validation"], result["verdict"]))

        root_checks = [
            {"name": "all_dataset_validations_pass",
             "passed": all(result["validation"]["passed"] for result in results)},
            {"name": "all_source_runs_unchanged", "passed": source_unchanged},
            {"name": "built_in_protected_results_unchanged",
             "passed": protected_unchanged, "observed": protected_after,
             "expected": protected_before},
            {"name": "optional_protection_manifest_exact",
             "passed": manifest_unchanged, "observed": protection_after},
            {"name": "exact_dataset_set",
             "passed": datasets == tuple(DATASETS), "observed": datasets,
             "expected": tuple(DATASETS)},
            {"name": "paired_image_cluster_bootstrap_10000",
             "passed": args.bootstrap_resamples == 10_000,
             "observed": args.bootstrap_resamples, "expected": 10_000},
        ]
        root_validation = {
            "schema_version": SCHEMA_VERSION,
            "passed": all(check["passed"] for check in root_checks),
            "n_checks": len(root_checks),
            "n_passed": sum(check["passed"] for check in root_checks),
            "failures": [check["name"] for check in root_checks
                         if not check["passed"]],
            "checks": root_checks,
            "protection_before": protection_before,
            "protection_after": protection_after,
            "cross_dataset_verdict": verdict,
        }
        write_json(staging / "validation.json", root_validation)
        (staging / "README.md").write_text(root_readme(
            results, cross_rows, root_validation, verdict,
            args.bootstrap_resamples, args.bootstrap_seed))

        # Required outputs are checked after the final README/validation write.
        missing = [name for name in ROOT_RESULT_FILES
                   if not (staging / name).is_file()]
        for dataset in datasets:
            missing.extend(
                f"{dataset}/{name}" for name in DATASET_RESULT_FILES
                if not (staging / dataset / name).is_file())
        if missing:
            raise AnalysisError(f"missing required result artifacts: {missing}")
        if not root_validation["passed"]:
            raise AnalysisError(
                f"root validation failed: {root_validation['failures']}")
        fsync_tree(staging)
        os.replace(staging, results_root)
        parent_fd = os.open(results_root.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        completed = True
        print(json.dumps({
            "status": "PASS",
            "results_root": str(results_root),
            "cross_dataset_verdict": verdict,
            "datasets": {result["dataset"]: result["verdict"]["verdict"]
                         for result in results},
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
        }, indent=2))
    finally:
        if not completed and staging.exists():
            # Only the private mkdtemp path created above can be removed.
            if staging.parent != results_root.parent or not staging.name.startswith(
                    f".{results_root.name}.staging-"):
                raise AnalysisError(f"refusing unsafe staging cleanup: {staging}")
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()

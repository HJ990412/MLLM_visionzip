#!/usr/bin/env python3
"""Run a frozen Qwen GQA/MT pilot after the v2 system correctness gate.

The v1 pilot's workload construction and measured request loop are reused
unchanged. This wrapper binds them to the new, matched-computation v2 gate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "qwen25-gpu-correctness-v2"
GATES = tuple(f"G{i}" for i in range(1, 16))
ADAPTER_SOURCES = (
    "mmimpress/qwen25/runner.py",
    "mmimpress/qwen25/store.py",
    "mmimpress/qwen25/vision.py",
)
REQUIRED_FROZEN_SOURCES = (
    "docs/qwen25_correctness_contract_v2.md",
    "scripts/82_validate_qwen25_v2.py",
    "scripts/83_eval_qwen25_v2_smoke.py",
    "scripts/84_eval_qwen25_v2_pilot.py",
    "scripts/85_report_qwen25_v2_pilot.py",
    *ADAPTER_SOURCES,
)
FROZEN_MANIFESTS = {
    "gqa": ROOT / "runs/qwen25_port_20260928T054537Z/gqa_manifest/manifest.json",
    "mt": ROOT / "runs/qwen25_port_20260928T054537Z/mt_manifest/manifest.json",
}
FROZEN_FILE_SHA256 = {
    "gqa": "e29d593ef83b106f2088ceceb607a9fb22dff90a542775cf19f68735739c52f2",
    "mt": "b76425302ad7de6000b9ac3079341120b367c8c5e70a1478469383709a9252b6",
}


def _load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V1_PILOT = _load_script("qwen25_frozen_pilot_v1", ROOT / "scripts/79_eval_qwen25_pilot.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _manifest_content_hash(value: dict[str, Any]) -> str:
    unsigned = dict(value)
    claimed = unsigned.pop("manifest_sha256", None)
    calculated = V1_PILOT.canonical_hash(unsigned)
    _require(claimed == calculated, "fixed validation manifest content hash mismatch")
    return calculated


def _find_validation_manifest(validation_path: Path) -> Path:
    for parent in (validation_path.parent, *validation_path.parents):
        candidate = parent / "validation_manifest.json"
        if candidate.is_file():
            return candidate
        if parent == ROOT:
            break
    raise RuntimeError("v2 validation_manifest.json was not found beside the validation run")


def bind_v2_validation(validation_path: Path) -> dict[str, Any]:
    """Fail before GPU load unless every fixed v2 gate and source is intact."""
    validation_path = validation_path.resolve(strict=True)
    value = _read_json(validation_path)
    _require(value.get("schema_version") == SCHEMA, "wrong v2 validation schema")
    _require(value.get("status") == "PASS", "v2 validation did not PASS")
    _require(value.get("gpu_system_correctness") == "PASS",
             "GPU system correctness did not PASS")
    _require(value.get("pilot_eligible") is True, "v2 pilot is not eligible")
    gates = value.get("gates")
    _require(isinstance(gates, dict) and set(gates) == set(GATES),
             "v2 validation must contain exactly G1 through G15")
    _require(all(isinstance(gates[name], dict) and
                 gates[name].get("status") == "PASS" for name in GATES),
             "every v2 G1 through G15 gate must PASS")

    config = value.get("configuration")
    _require(isinstance(config, dict), "v2 validator configuration is missing")
    expected_config = {
        "attention_backend": "sdpa", "seed": 1234, "max_new_tokens": 16,
        "chunk_size": 64, "budget_ratio": 0.25,
    }
    for key, expected in expected_config.items():
        _require(config.get(key) == expected,
                 f"v2 validation configuration differs: {key}")
    _require(value.get("model_revision") == V1_PILOT.CHECKPOINT_REVISION,
             "v2 checkpoint revision differs from the frozen pilot")

    manifest_path = _find_validation_manifest(validation_path)
    fixed_manifest = _read_json(manifest_path)
    content_hash = _manifest_content_hash(fixed_manifest)
    _require(value.get("manifest_sha256") == content_hash,
             "v2 validation is bound to a different fixed sample manifest")
    _require(fixed_manifest.get("source_index_sha256") ==
             V1_PILOT.GQA_INDEX_SHA256,
             "fixed validation sample source index changed")
    _require(config == fixed_manifest.get("configuration"),
             "v2 validator configuration differs from the frozen manifest")
    _require(config.get("model_id") == V1_PILOT.MODEL_ID and
             config.get("checkpoint_revision") == V1_PILOT.CHECKPOINT_REVISION and
             config.get("methods") == list(V1_PILOT.METHODS) and
             config.get("min_pixels") == 256 * 28 * 28 and
             config.get("max_pixels") == 1024 * 28 * 28,
             "frozen model, method, or processor policy changed")

    freeze_path = manifest_path.parent / "frozen_inputs.json"
    _require(freeze_path.is_file() and
             sha256_file(freeze_path) == value.get("frozen_inputs_sha256"),
             "frozen_inputs.json SHA256 differs from v2 validation")
    freeze = _read_json(freeze_path)
    frozen_files = freeze.get("files")
    _require(freeze.get("schema_version") ==
             "qwen25-correctness-v2-frozen-inputs-v1" and
             isinstance(frozen_files, dict) and
             freeze.get("file_count") == len(frozen_files),
             "invalid frozen input inventory")
    _require(set(REQUIRED_FROZEN_SOURCES).issubset(frozen_files),
             "frozen input inventory lacks required v2 sources")
    for relative, record in frozen_files.items():
        _require(isinstance(relative, str) and isinstance(record, dict),
                 "malformed frozen input record")
        source = (ROOT / relative).resolve(strict=True)
        _require(source.is_relative_to(ROOT) and source.is_file() and
                 source.stat().st_size == record.get("bytes") and
                 sha256_file(source) == record.get("sha256"),
                 f"frozen input changed after validation: {relative}")

    source_hashes = value.get("source_hashes")
    _require(isinstance(source_hashes, dict), "v2 source hashes are missing")
    for relative in REQUIRED_FROZEN_SOURCES:
        _require(relative in source_hashes,
                 f"v2 validation lacks source hash: {relative}")
        _require(source_hashes[relative] == frozen_files[relative]["sha256"],
                 f"validator and frozen inventory disagree: {relative}")
    for relative, expected in source_hashes.items():
        _require(isinstance(relative, str) and isinstance(expected, str) and
                 len(expected) == 64, "malformed v2 source hash")
        source = (ROOT / relative).resolve(strict=True)
        _require(source.is_relative_to(ROOT) and source.is_file(),
                 f"v2 source is outside workspace or missing: {relative}")
        _require(sha256_file(source) == expected,
                 f"v2 source changed after validation: {relative}")

    return {
        "path": str(validation_path),
        "sha256": sha256_file(validation_path),
        "schema_version": SCHEMA,
        "validation_status": "PASS",
        "gpu_system_correctness": "PASS",
        "benchmark_validated": True,
        "run_mode": "validated_benchmark_v2",
        "gate_statuses": {name: gates[name]["status"] for name in GATES},
        "fixed_validation_manifest_path": str(manifest_path),
        "fixed_validation_manifest_file_sha256": sha256_file(manifest_path),
        "fixed_validation_manifest_content_sha256": content_hash,
        "frozen_inputs_path": str(freeze_path),
        "frozen_inputs_sha256": value["frozen_inputs_sha256"],
        "frozen_file_count": len(frozen_files),
        "source_hashes": source_hashes,
        "configuration": config,
        "model_revision": value["model_revision"],
    }


def frozen_workload(dataset: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild 40-image workload and compare every field to its old frozen copy."""
    path = FROZEN_MANIFESTS[dataset]
    _require(sha256_file(path) == FROZEN_FILE_SHA256[dataset],
             f"original {dataset} frozen workload manifest changed")
    old = _read_json(path)
    source = V1_PILOT.GQA_INDEX if dataset == "gqa" else V1_PILOT.MT_INDEX
    fresh = V1_PILOT.build_manifest(dataset, source, 40)
    _require(fresh == old, f"{dataset} workload changed from the original frozen manifest")
    _require(len(fresh["images"]) == 40 and
             len(fresh["methods"]) == 3 and
             all(len(image["turns"]) == (6 if dataset == "gqa" else 3)
                 for image in fresh["images"]),
             f"{dataset} workload size or question count changed")
    return fresh, {
        "path": str(path.resolve()), "file_sha256": FROZEN_FILE_SHA256[dataset],
        "content_sha256": old["manifest_sha256"],
        "source_index": fresh["source_index"],
        "source_index_sha256": fresh["source_index_sha256"],
    }


def _run_with_finite_logits(manifest: dict[str, Any], run_dir: Path,
                            binding: dict[str, Any], *, warmup: bool) -> None:
    """Capture first logits from the existing forward without serializing tensors."""
    import torch
    from mmimpress.qwen25.runner import Qwen25Runner

    original_pixels = Qwen25Runner.run_pixels
    original_cache = Qwen25Runner.run_cache

    def summarize(result: dict[str, Any]) -> dict[str, Any]:
        logits = result.pop("first_logits", None)
        _require(isinstance(logits, torch.Tensor) and logits.ndim == 1 and
                 logits.dtype == torch.float32 and logits.numel() > 0,
                 "runner did not return first-token FP32 logits")
        nonfinite_count = int((~torch.isfinite(logits)).sum().item())
        _require(nonfinite_count == 0, "nonfinite first-token logits")
        _require(int(logits.argmax().item()) == int(result["first_token_id"]),
                 "first-token ID differs from logits argmax")
        result["first_logits_finite"] = True
        result["first_logits_nonfinite_count"] = nonfinite_count
        result["first_logits_dtype"] = "float32"
        result["first_logits_shape"] = list(logits.shape)
        result["first_logits_sha256"] = hashlib.sha256(
            logits.contiguous().numpy().tobytes()).hexdigest()
        return result

    def run_pixels(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        _require("return_logits" not in kwargs,
                 "v2 logits instrumentation owns return_logits")
        return summarize(original_pixels(self, *args, return_logits=True, **kwargs))

    def run_cache(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        _require("return_logits" not in kwargs,
                 "v2 logits instrumentation owns return_logits")
        return summarize(original_cache(self, *args, return_logits=True, **kwargs))

    Qwen25Runner.run_pixels = run_pixels
    Qwen25Runner.run_cache = run_cache
    try:
        V1_PILOT.run_pilot(manifest, run_dir, binding, warmup=warmup)
    finally:
        Qwen25Runner.run_pixels = original_pixels
        Qwen25Runner.run_cache = original_cache


def run(dataset: str, run_dir: Path, validation_path: Path, *,
        manifest_only: bool = False, warmup: bool = True) -> dict[str, Any]:
    _require(dataset in FROZEN_MANIFESTS, "dataset must be gqa or mt")
    binding = bind_v2_validation(validation_path)
    manifest, frozen = frozen_workload(dataset)
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256")
    unsigned.update({
        "validation": binding,
        "validation_status": "PASS",
        "benchmark_validated": True,
        "run_mode": "validated_benchmark_v2",
        "frozen_workload": frozen,
    })
    unsigned["manifest_sha256"] = V1_PILOT.canonical_hash(unsigned)
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    V1_PILOT._write_json_new(run_dir / "manifest.json", unsigned)
    V1_PILOT._write_json_new(run_dir / "config.json", {
        "schema_version": V1_PILOT.SCHEMA_VERSION,
        "v2_validation_schema": SCHEMA,
        "manifest_sha256": unsigned["manifest_sha256"],
        "frozen_workload": frozen,
        "validation": binding,
        "validation_status": "PASS",
        "benchmark_validated": True,
        "run_mode": "validated_benchmark_v2",
        "manifest_only": manifest_only,
        "warmup": warmup,
    })
    if not manifest_only:
        _run_with_finite_logits(unsigned, run_dir, binding, warmup=warmup)
    return {
        "run_dir": str(run_dir), "dataset": dataset, "images": 40,
        "expected_requests": 720 if dataset == "gqa" else 360,
        "manifest_sha256": unsigned["manifest_sha256"],
        "frozen_workload_content_sha256": frozen["content_sha256"],
        "v2_validation_sha256": binding["sha256"],
        "status": "MANIFEST ONLY" if manifest_only else "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(FROZEN_MANIFESTS), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()
    result = run(args.dataset, args.run_dir, args.validation,
                 manifest_only=args.manifest_only, warmup=not args.no_warmup)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

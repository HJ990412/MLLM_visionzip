"""Orchestrate the frozen ImageOnly cross-dataset generalization run.

The GPU evaluator is restartable at image granularity, but this driver fixes
the dataset identities, invocation parameters, and experiment-wide protection
manifest in one place.  It never removes an old result or store.  The only
payload deletion authority lives in ``33_eval_image_only_generalization.py``
and is restricted to that run's marked one-image temporary directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_ROOT = ROOT / "runs/image_only_generalization"
DEFAULT_RESULTS_ROOT = ROOT / "results/image_only_generalization"
SCHEMA_VERSION = "image-only-generalization-orchestrator-v1"
CANONICAL_DATASETS = ("gqa_large", "vqav2", "textvqa")
OUTPUT_NAME_PREFIX = "image_only_generalization"
MIN_FREE_AFTER_GIB = 64.0

SPECS: dict[str, dict[str, Any]] = {
    "gqa_large": {
        "metric": "gqa",
        "index": ROOT / "data/gqa_large/index.json",
        "index_sha256": (
            "0d50962f0c1bac3bc6e1836978289d5fde7d60b434f55cebbc4607c4c80a797c"
        ),
        "workload_sha256": (
            "cabec1bb1035c836839b98d72ba6a04558529d246cf55f8499ca75ea2d2f290c"
        ),
        "images": 395,
        "questions": 1185,
        "skip": 1,
        "questions_per_image": 3,
    },
    "vqav2": {
        "metric": "vqa",
        "index": ROOT / "data/vqav2/index.json",
        "index_sha256": (
            "b83d5fa288fcb722ca073e261d3fec9086629ed0db2e568a2d5d6a24ef1589d7"
        ),
        "workload_sha256": (
            "e341b499a968c5caba4ddffc58fdf0ccafe2e0e212b6cb280ca1b11fc9f31d18"
        ),
        "images": 250,
        "questions": 1000,
        "skip": 1,
        "questions_per_image": 4,
    },
    "textvqa": {
        "metric": "vqa",
        "index": ROOT / "data/textvqa/index.json",
        "index_sha256": (
            "b1e5ff0eaba2a45c6398e7cb90631cdc7387eff25ed0997a66374f25968a2f4d"
        ),
        "workload_sha256": (
            "49fa0b15f406132162cba1b47245f28a560d28b4f3af76985c855d77a145a2a6"
        ),
        "images": 500,
        "questions": 500,
        "skip": 1,
        "questions_per_image": 1,
    },
}


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def local_model_revision() -> str:
    """Resolve the immutable local checkpoint revision used by every shard."""
    ref = (Path.home() / ".cache/huggingface/hub/"
           "models--llava-hf--llava-v1.6-vicuna-7b-hf/refs/main")
    if not ref.is_file() or ref.is_symlink():
        raise FileNotFoundError(f"missing regular local model ref: {ref}")
    revision = ref.read_text().strip()
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError(f"invalid local model revision: {revision!r}")
    snapshot = ref.parent.parent / "snapshots" / revision
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise FileNotFoundError(f"missing local model snapshot: {snapshot}")
    return revision


def validate_output_roots(run_path: Path, results_path: Path) -> tuple[Path, Path]:
    """Restrict writes to new, top-level experiment namespaces."""
    for value, label in ((run_path, "run"), (results_path, "result")):
        if value.exists() and value.is_symlink():
            raise ValueError(f"{label} root may not be a symlink: {value}")
    run_root = run_path.resolve()
    results_root = results_path.resolve()
    expected_run_parent = (ROOT / "runs").resolve()
    expected_results_parent = (ROOT / "results").resolve()
    if (run_root.parent != expected_run_parent
            or not run_root.name.startswith(OUTPUT_NAME_PREFIX)):
        raise ValueError(
            f"run root must be a dedicated {OUTPUT_NAME_PREFIX}* child of "
            f"{expected_run_parent}: {run_root}")
    if (results_root.parent != expected_results_parent
            or not results_root.name.startswith(OUTPUT_NAME_PREFIX)):
        raise ValueError(
            f"result root must be a dedicated {OUTPUT_NAME_PREFIX}* child of "
            f"{expected_results_parent}: {results_root}")
    if (run_root == results_root or run_root in results_root.parents
            or results_root in run_root.parents):
        raise ValueError("run and result roots overlap")
    return run_root, results_root


def validate_main_contract(datasets: list[str], *, shard_size: int,
                           bootstrap_resamples: int,
                           min_free_after_gib: float) -> None:
    """Fail before GPU work if the final experiment contract was weakened."""
    if tuple(datasets) != CANONICAL_DATASETS:
        raise ValueError(
            "final experiment requires datasets in canonical order: "
            + ",".join(CANONICAL_DATASETS))
    if shard_size < 1:
        raise ValueError("shard size must be positive")
    if bootstrap_resamples != 10_000:
        raise ValueError("main experiment requires exactly 10k bootstrap resamples")
    if not math.isfinite(min_free_after_gib) \
            or min_free_after_gib < MIN_FREE_AFTER_GIB:
        raise ValueError(
            f"main experiment requires at least {MIN_FREE_AFTER_GIB:g} GiB "
            "of post-build free-space reserve")


def verify_dataset_model_revision(dataset_run: Path,
                                  expected_revision: str) -> None:
    path = dataset_run / "config.json"
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing regular dataset config: {path}")
    observed = json.loads(path.read_text()).get("model_revision")
    if observed != expected_revision:
        raise RuntimeError(
            f"dataset model revision mismatch: {observed!r} != "
            f"{expected_revision!r}")


def protected_snapshot(run_root: Path = DEFAULT_RUN_ROOT,
                       results_root: Path = DEFAULT_RESULTS_ROOT) -> dict:
    """Content manifest for every pre-existing run/result artifact.

    The two new roots are excluded by exact top-level path, so analysis can
    publish its own files without weakening protection of anything older.
    Store trees are intentionally not hashed: this requirement protects old
    results/raw/README evidence, and hashing hundreds of GiB of immutable KV
    would perturb the SSD benchmark itself.
    """
    files: dict[str, dict[str, Any]] = {}
    scopes = ((ROOT / "results", results_root.resolve()),
              (ROOT / "runs", run_root.resolve()))
    for scope, excluded in scopes:
        if not scope.exists():
            continue
        for path in sorted(p for p in scope.rglob("*") if p.is_file()):
            resolved = path.resolve()
            if resolved == excluded or excluded in resolved.parents:
                continue
            rel = path.relative_to(ROOT).as_posix()
            files[rel] = {
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "all files below pre-existing results/ and runs/",
        "excluded_new_roots": [
            str(results_root.resolve()), str(run_root.resolve())],
        "file_count": len(files),
        "total_bytes": sum(row["size_bytes"] for row in files.values()),
        "files": files,
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
    }


def verify_protected_snapshot(expected: dict, run_root: Path,
                              results_root: Path) -> dict:
    observed = protected_snapshot(run_root, results_root)
    same = (expected.get("files") == observed.get("files")
            and expected.get("manifest_sha256")
            == observed.get("manifest_sha256"))
    if not same:
        before = expected.get("files", {})
        after = observed.get("files", {})
        changed = sorted(
            key for key in set(before) | set(after)
            if before.get(key) != after.get(key))
        raise RuntimeError(
            "protected pre-existing artifacts changed: "
            + ", ".join(changed[:20]))
    return observed


def atomic_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with tmp.open("x") as handle:
        json.dump(value, handle, indent=1, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if exclusive and path.exists():
        tmp.unlink()
        raise FileExistsError(f"refusing to overwrite {path}")
    os.replace(tmp, path)
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def run_streaming(command: list[str], log_path: Path) -> None:
    """Run one stage while preserving both a durable log and live progress."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1")
    with log_path.open("a", buffering=1) as log:
        rendered = " ".join(command)
        log.write(f"\n$ {rendered}\n")
        print(f"$ {rendered}", flush=True)
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            print(line, end="", flush=True)
        returncode = process.wait()
        log.flush()
        os.fsync(log.fileno())
    if returncode:
        raise SystemExit(f"stage failed ({returncode}): {rendered}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--results-root", type=Path,
                        default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--datasets", default="gqa_large,vqav2,textvqa")
    parser.add_argument("--shard-size", type=int, default=1000,
                        help="process-level shard; KV itself remains one image")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-free-after-gib", type=float, default=64.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()

    run_root, results_root = validate_output_roots(
        args.run_root, args.results_root)
    datasets = [item.strip() for item in args.datasets.split(",")
                if item.strip()]
    if not datasets or len(datasets) != len(set(datasets)):
        raise ValueError("datasets must be a nonempty unique list")
    unknown = set(datasets) - set(SPECS)
    if unknown:
        raise ValueError(f"unknown datasets: {sorted(unknown)}")
    validate_main_contract(
        datasets, shard_size=args.shard_size,
        bootstrap_resamples=args.bootstrap_resamples,
        min_free_after_gib=args.min_free_after_gib)
    if args.analysis_only and args.skip_analysis:
        raise ValueError("analysis-only conflicts with skip-analysis")

    experiment_path = run_root / "experiment.json"
    protection_path = run_root / "protected_artifacts_before.json"
    if not run_root.exists():
        if args.resume or args.analysis_only:
            raise FileNotFoundError(f"run root does not exist: {run_root}")
        if results_root.exists():
            raise FileExistsError(f"refusing existing result root: {results_root}")
        model_revision = local_model_revision()
        protection = protected_snapshot(run_root, results_root)
        run_root.mkdir(parents=True, exist_ok=False)
        experiment_id = str(uuid.uuid4())
        experiment = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "status": "running",
            "datasets": datasets,
            "specs": {name: {key: (str(value) if isinstance(value, Path)
                                   else value)
                              for key, value in SPECS[name].items()}
                      for name in datasets},
            "seed": args.seed,
            "shard_size": args.shard_size,
            "min_free_after_gib": args.min_free_after_gib,
            "bootstrap_resamples": args.bootstrap_resamples,
            "model_revision": model_revision,
            "run_root": str(run_root),
            "results_root": str(results_root),
            "started_at_unix": time.time(),
        }
        atomic_json(protection_path, protection, exclusive=True)
        atomic_json(experiment_path, experiment, exclusive=True)
    else:
        if not (args.resume or args.analysis_only):
            raise FileExistsError(
                f"run root exists; pass --resume explicitly: {run_root}")
        experiment = json.loads(experiment_path.read_text())
        protection = json.loads(protection_path.read_text())
        if experiment.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("existing experiment schema mismatch")
        if experiment.get("datasets") != datasets:
            raise ValueError("resume dataset list mismatch")
        expected_resume = {
            "seed": args.seed,
            "shard_size": args.shard_size,
            "bootstrap_resamples": args.bootstrap_resamples,
            "min_free_after_gib": args.min_free_after_gib,
            "run_root": str(run_root),
            "results_root": str(results_root),
        }
        for key, expected in expected_resume.items():
            if experiment.get(key) != expected:
                raise ValueError(
                    f"resume {key} mismatch: {experiment.get(key)!r} != "
                    f"{expected!r}")
        experiment_id = str(experiment["experiment_id"])
        verify_protected_snapshot(protection, run_root, results_root)
        if experiment.get("status") == "complete":
            required = (
                results_root / "README.md",
                results_root / "validation.json",
                results_root / "protected_artifacts_validation.json",
            )
            if not all(path.is_file() and not path.is_symlink()
                       for path in required):
                raise RuntimeError(
                    "experiment says complete but final artifacts are missing")
            print(f"experiment {experiment_id}: already complete", flush=True)
            return
        if results_root.exists():
            raise FileExistsError(
                "result root already exists while experiment is incomplete; "
                f"refusing overwrite/resume: {results_root}")
        model_revision = str(experiment.get("model_revision", ""))
        if (len(model_revision) != 40
                or any(c not in "0123456789abcdef" for c in model_revision)):
            raise ValueError(
                f"invalid frozen experiment model revision: {model_revision!r}")
        # Analysis is deliberately CPU-only and consumes the revision recorded
        # in the immutable run artifacts.  Only a resumed GPU stage needs the
        # current local ref to remain identical.
        if (not args.analysis_only
                and local_model_revision() != model_revision):
            raise RuntimeError("local model revision changed since run creation")

    evaluator = ROOT / "scripts/33_eval_image_only_generalization.py"
    if not args.analysis_only:
        for dataset in datasets:
            spec = SPECS[dataset]
            dataset_run = run_root / dataset
            temp_root = run_root / "_temporary_visual_kv" / dataset
            n_shards = math.ceil(spec["images"] / args.shard_size)
            for shard_index in range(n_shards):
                if local_model_revision() != model_revision:
                    raise RuntimeError("local model revision changed during the run")
                command = [
                    sys.executable, str(evaluator),
                    "--dataset", dataset,
                    "--metric", spec["metric"],
                    "--index", str(spec["index"]),
                    "--run-dir", str(dataset_run),
                    "--temp-root", str(temp_root),
                    "--experiment-id", experiment_id,
                    "--skip", str(spec["skip"]),
                    "--questions", str(spec["questions_per_image"]),
                    "--shard-index", str(shard_index),
                    "--shard-size", str(args.shard_size),
                    "--expected-index-sha256", spec["index_sha256"],
                    "--expected-workload-sha256", spec["workload_sha256"],
                    "--expected-images", str(spec["images"]),
                    "--expected-questions", str(spec["questions"]),
                    "--seed", str(args.seed),
                    "--max-new-tokens", "16",
                    "--min-free-after-gib", str(args.min_free_after_gib),
                ]
                run_streaming(command, dataset_run / "run.log")
                verify_dataset_model_revision(dataset_run, model_revision)
            verify_protected_snapshot(protection, run_root, results_root)

    if not args.skip_analysis:
        if results_root.exists():
            raise FileExistsError(
                f"refusing existing analysis destination: {results_root}")
        analyzer = ROOT / "scripts/34_analyze_image_only_generalization.py"
        command = [
            sys.executable, str(analyzer),
            "--run-root", str(run_root),
            "--results-root", str(results_root),
            "--bootstrap-resamples", str(args.bootstrap_resamples),
            "--bootstrap-seed", str(args.seed),
            "--protection-manifest", str(protection_path),
        ]
        run_streaming(command, run_root / "analysis.log")

    after = verify_protected_snapshot(protection, run_root, results_root)
    if results_root.exists():
        atomic_json(results_root / "protected_artifacts_validation.json", {
            "schema_version": SCHEMA_VERSION,
            "passed": True,
            "before": protection["manifest_sha256"],
            "after": after["manifest_sha256"],
            "file_count": after["file_count"],
            "total_bytes": after["total_bytes"],
            "manifest_source": str(protection_path),
        }, exclusive=True)
    experiment.update({
        "status": "complete" if not args.skip_analysis else "inference_complete",
        "completed_at_unix": time.time(),
        "protected_artifacts_unchanged": True,
        "protected_manifest_sha256_before": protection["manifest_sha256"],
        "protected_manifest_sha256_after": after["manifest_sha256"],
    })
    atomic_json(experiment_path, experiment)
    print(f"experiment {experiment_id}: {experiment['status']}", flush=True)


if __name__ == "__main__":
    main()

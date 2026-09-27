#!/usr/bin/env python3
"""Protect prior experiment artifacts and all runtime source files.

The artifact fingerprint policy is identical to script 69: full SHA-256 for
files through 1 MiB, nine framed 4-KiB windows for larger files, plus inode,
size and timestamp metadata.  Source files receive full SHA-256 regardless of
size.  Only the exact new run/results roots are excluded from artifact scope.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "mt-gqa-5arm-protected-artifacts-v1"
MANIFEST_NAME = "protected_artifacts_before.json"
VALIDATION_NAME = "protected_artifacts_validation.json"
RUN_PREFIX = "mt_gqa_5arm_generated_"
RUN_ID_RE = re.compile(r"^mt_gqa_5arm_generated_[0-9]{8}T[0-9]{6}Z(?:_[A-Za-z0-9]+)?$")
SOURCE_SUFFIXES = frozenset({".py", ".sh"})


class ProtectionError(RuntimeError):
    pass


def _base69():
    path = ROOT / "scripts/69_protect_rekv_artifacts.py"
    spec = importlib.util.spec_from_file_location("_mt5_fingerprint_base69", path)
    if spec is None or spec.loader is None:
        raise ProtectionError(f"cannot load fingerprint implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _base69()
POLICY = BASE.DEFAULT_POLICY
SCOPE_ROOTS = BASE.SCOPE_ROOTS


def canonical_hash(value: Any) -> str:
    return BASE.canonical_hash(value)


def validate_output_roots(run_root: Path | str, results_root: Path | str,
                          *, project_root: Path = ROOT) -> tuple[Path, Path]:
    project = Path(project_root).resolve()
    run_parent = project / "runs"
    result_parent = project / "results"
    for parent in (run_parent, result_parent):
        if parent.is_symlink() or not parent.is_dir():
            raise ProtectionError(f"output parent is not a real directory: {parent}")
    paths = []
    for value in (run_root, results_root):
        path = Path(value)
        if not path.is_absolute():
            path = project / path
        if os.path.lexists(path) and path.is_symlink():
            raise ProtectionError(f"output root is a symlink: {path}")
        paths.append(path.resolve(strict=False))
    run, result = paths
    if (run.parent != run_parent or result.parent != result_parent
            or run.name != result.name or not RUN_ID_RE.fullmatch(run.name)):
        raise ProtectionError(
            "run/results roots must be matching dedicated "
            "runs/mt_gqa_5arm_generated_<UTC>/ and results/<same-name>/")
    return run, result


def _source_paths(project: Path) -> list[Path]:
    paths: list[Path] = []
    for name in ("scripts", "mmimpress", "tests"):
        directory = project / name
        if directory.is_symlink() or not directory.is_dir():
            raise ProtectionError(f"source directory is unsafe: {directory}")
        for parent, dirs, files in os.walk(directory, followlinks=False):
            for dirname in dirs:
                if (Path(parent) / dirname).is_symlink():
                    raise ProtectionError(f"symlink in source tree: {parent}/{dirname}")
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            for filename in sorted(files):
                path = Path(parent) / filename
                if path.suffix not in SOURCE_SUFFIXES:
                    continue
                if path.is_symlink() or not path.is_file():
                    raise ProtectionError(f"source file is unsafe: {path}")
                paths.append(path)
    return sorted(paths)


def _hash_source_file(path: Path) -> str:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ProtectionError(f"source is not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size,
                before.st_mtime_ns, before.st_ctime_ns) != (
                opened.st_dev, opened.st_ino, opened.st_size,
                opened.st_mtime_ns, opened.st_ctime_ns):
            raise ProtectionError(f"source changed while opening: {path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = path.lstat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size,
                              value.st_mtime_ns, value.st_ctime_ns)
    if identity(before) != identity(after_open) or identity(before) != identity(after_path):
        raise ProtectionError(f"source changed while hashing: {path}")
    return digest.hexdigest()


def source_hashes(*, project_root: Path = ROOT) -> dict[str, str]:
    project = Path(project_root).resolve()
    paths = _source_paths(project)
    hashes = {path.relative_to(project).as_posix(): _hash_source_file(path)
              for path in paths}
    if not hashes or "scripts/74_protect_mt_gqa_5arm.py" not in hashes:
        raise ProtectionError("source inventory is incomplete")
    if [path.relative_to(project).as_posix() for path in _source_paths(project)] != list(hashes):
        raise ProtectionError("source inventory changed during hashing")
    return hashes


def _artifact_entries(project: Path, run: Path, result: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for path in BASE._iter_entries(project, run, result):
        relative = path.relative_to(project).as_posix()
        value = path.lstat()
        if stat.S_ISLNK(value.st_mode):
            entries[relative] = {
                "type": "symlink", "target": os.readlink(path),
                **BASE._file_metadata(value),
            }
        elif stat.S_ISDIR(value.st_mode):
            entries[relative] = {
                "type": "directory", **BASE._directory_metadata(value),
            }
        elif stat.S_ISREG(value.st_mode):
            integrity, stable = BASE._fingerprint_regular_file(path, POLICY)
            entries[relative] = {
                "type": "regular_file", "size_bytes": int(stable.st_size),
                "integrity": integrity, **BASE._file_metadata(stable),
            }
        else:
            raise ProtectionError(f"unsupported artifact type: {path}")
    observed = {path.relative_to(project).as_posix(): path
                for path in BASE._iter_entries(project, run, result)}
    if set(observed) != set(entries) or any(
            not BASE._entry_still_matches(observed[name], row)
            for name, row in entries.items()):
        raise ProtectionError("artifact tree changed while snapshotting")
    return entries


def _summary(entries: dict[str, dict[str, Any]]) -> dict[str, int]:
    return BASE._summary(entries)


def _snapshot_payload(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "manifest_sha256"}


def snapshot(run_root: Path | str, results_root: Path | str,
             *, project_root: Path = ROOT) -> dict[str, Any]:
    project = Path(project_root).resolve()
    run, result = validate_output_roots(run_root, results_root,
                                        project_root=project)
    source_before = source_hashes(project_root=project)
    entries = _artifact_entries(project, run, result)
    source_after = source_hashes(project_root=project)
    if source_before != source_after:
        raise ProtectionError("source files changed during artifact snapshot")
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scope_roots": list(SCOPE_ROOTS),
        "excluded_new_roots": [str(run), str(result)],
        "fingerprint_policy": POLICY.as_dict(),
        "source_policy": "full_sha256_all_py_sh_in_scripts_mmimpress_tests",
        "source_hashes": source_after,
        "source_file_count": len(source_after),
        **_summary(entries),
        "entries": entries,
    }
    value["manifest_sha256"] = canonical_hash(_snapshot_payload(value))
    return value


def record_before(run_root: Path | str, results_root: Path | str,
                  *, project_root: Path = ROOT) -> tuple[Path, dict[str, Any]]:
    project = Path(project_root).resolve()
    run, result = validate_output_roots(run_root, results_root,
                                        project_root=project)
    if os.path.lexists(run) or os.path.lexists(result):
        raise FileExistsError("new run/results roots must be absent before snapshot")
    value = snapshot(run, result, project_root=project)
    run.mkdir(parents=False, exist_ok=False)
    path = run / MANIFEST_NAME
    BASE._atomic_json(path, value, exclusive=True)
    return path, value


def _read_manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ProtectionError(f"protection manifest is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtectionError("protection manifest must be an object")
    return value


def verify_after(run_root: Path | str, results_root: Path | str,
                 *, project_root: Path = ROOT) -> tuple[Path, dict[str, Any]]:
    project = Path(project_root).resolve()
    run, result = validate_output_roots(run_root, results_root,
                                        project_root=project)
    before = _read_manifest(run / MANIFEST_NAME)
    if (before.get("schema_version") != SCHEMA_VERSION
            or before.get("scope_roots") != list(SCOPE_ROOTS)
            or before.get("excluded_new_roots") != [str(run), str(result)]
            or before.get("fingerprint_policy") != POLICY.as_dict()
            or before.get("source_policy")
            != "full_sha256_all_py_sh_in_scripts_mmimpress_tests"
            or before.get("source_file_count") != len(before.get("source_hashes", {}))
            or not isinstance(before.get("entries"), dict)
            or _summary(before["entries"]) != {key: before.get(key)
                                               for key in _summary(before["entries"])}
            or canonical_hash(_snapshot_payload(before))
            != before.get("manifest_sha256")):
        raise ProtectionError("before manifest is invalid or altered")
    after = snapshot(run, result, project_root=project)
    prior = before["entries"]
    current = after["entries"]
    source_prior = before["source_hashes"]
    source_current = after["source_hashes"]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "passed": False,
        "before_manifest_sha256": before["manifest_sha256"],
        "after_manifest_sha256": after["manifest_sha256"],
        "entry_count_before": before["entry_count"],
        "entry_count_after": after["entry_count"],
        "source_file_count_before": len(source_prior),
        "source_file_count_after": len(source_current),
        "missing_paths": sorted(set(prior) - set(current)),
        "changed_paths": sorted(key for key in set(prior) & set(current)
                                if prior[key] != current[key]),
        "added_paths": sorted(set(current) - set(prior)),
        "source_missing_paths": sorted(set(source_prior) - set(source_current)),
        "source_changed_paths": sorted(
            key for key in set(source_prior) & set(source_current)
            if source_prior[key] != source_current[key]),
        "source_added_paths": sorted(set(source_current) - set(source_prior)),
        "source_sha256_before": canonical_hash(source_prior),
        "source_sha256_after": canonical_hash(source_current),
        "fingerprint_policy": POLICY.as_dict(),
    }
    report["passed"] = not any(report[key] for key in (
        "missing_paths", "changed_paths", "source_missing_paths",
        "source_changed_paths", "source_added_paths"))
    path = run / VALIDATION_NAME
    if os.path.lexists(path):
        recorded = _read_manifest(path)
        if recorded != report:
            raise ProtectionError("existing protection validation differs from current audit")
    else:
        BASE._atomic_json(path, report, exclusive=True)
    if not report["passed"]:
        raise ProtectionError(
            "prior artifacts or source changed: " + ", ".join(
                f"{key}={len(report[key])}" for key in (
                    "missing_paths", "changed_paths", "source_missing_paths",
                    "source_changed_paths", "source_added_paths")))
    return path, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--before", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.before:
            path, value = record_before(args.run_root, args.results_root)
            output = {"status": "recorded", "manifest": str(path),
                      "manifest_sha256": value["manifest_sha256"],
                      "entry_count": value["entry_count"],
                      "source_file_count": value["source_file_count"]}
        else:
            path, value = verify_after(args.run_root, args.results_root)
            output = {"status": "unchanged", "validation": str(path),
                      "entry_count": value["entry_count_after"],
                      "source_file_count": value["source_file_count_after"]}
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    except (ProtectionError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"MT-GQA artifact protection failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

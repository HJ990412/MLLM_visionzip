#!/usr/bin/env python3
"""Fail-closed protection for the MT-GQA gold/generated-history experiment.

The before operation snapshots every pre-existing entry below ``runs/`` and
``results/`` *before* creating the dedicated experiment run directory.  Only
the two exact new experiment roots are excluded.  The after operation hashes
the same scope and refuses success if a pre-existing entry was removed,
replaced, or changed.  Newly created sibling outputs are reported but do not
alter the frozen set of pre-existing paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
PREFIX = "mt_gqa_4arm_history_comparison_"
SCHEMA_VERSION = "mt-gqa-history-protected-artifacts-v2"
MANIFEST_NAME = "protected_artifacts_before.json"
VALIDATION_NAME = "protected_artifacts_validation.json"


class ProtectionError(RuntimeError):
    """Raised when a protection snapshot cannot be trusted or differs."""

    def __init__(self, message: str, report: dict[str, Any] | None = None):
        super().__init__(message)
        self.report = report


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _metadata(value: os.stat_result) -> dict[str, int]:
    return {
        "mode": int(value.st_mode),
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "uid": int(value.st_uid),
        "gid": int(value.st_gid),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _hash_regular_file(
    path: Path, block_size: int = 8 << 20,
) -> tuple[str, os.stat_result]:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ProtectionError(f"protected path is not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_mode) != (
                before.st_dev, before.st_ino, before.st_mode):
            raise ProtectionError(f"protected file changed while opening: {path}")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, block_size)
            if not block:
                break
            digest.update(block)
        after_read = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = path.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise ProtectionError(f"protected file disappeared: {path}") from error
    stable = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after_read, key)
           or getattr(before, key) != getattr(after_path, key) for key in stable):
        raise ProtectionError(f"protected file changed while hashing: {path}")
    return digest.hexdigest(), after_path


def validate_output_roots(
    run_root: Path | str,
    results_root: Path | str,
    *,
    project_root: Path = ROOT,
) -> tuple[Path, Path]:
    project = Path(project_root).resolve()
    runs = project / "runs"
    results = project / "results"
    for parent, label in ((runs, "runs"), (results, "results")):
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError(f"{label} scope is not a real directory: {parent}")
    resolved: list[Path] = []
    for value, parent, label in (
        (run_root, runs, "run"), (results_root, results, "results")
    ):
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = project / candidate
        target = candidate.resolve(strict=False)
        if target.parent != parent.resolve() or not target.name.startswith(PREFIX):
            raise ValueError(
                f"{label} root must be a direct {PREFIX}* child of {parent}: {target}")
        if os.path.lexists(target) and target.is_symlink():
            raise ValueError(f"{label} root may not be a symlink: {target}")
        resolved.append(target)
    run, result = resolved
    if run.name != result.name:
        raise ValueError("run/results experiment root names must match")
    if run == result or run in result.parents or result in run.parents:
        raise ValueError("run and results roots overlap")
    return run, result


def _is_excluded(path: Path, exclusions: tuple[Path, Path]) -> bool:
    absolute = path.absolute()
    return any(absolute == root or root in absolute.parents for root in exclusions)


def _iter_entries(project: Path, run: Path, results: Path) -> Iterable[Path]:
    exclusions = (run.absolute(), results.absolute())
    for scope_name in ("runs", "results"):
        scope = project / scope_name
        if scope.is_symlink() or not scope.is_dir():
            raise ProtectionError(f"protected scope is not a real directory: {scope}")
        for directory, dirnames, filenames in os.walk(scope, followlinks=False):
            parent = Path(directory)
            kept: list[str] = []
            for name in sorted(dirnames):
                path = parent / name
                if _is_excluded(path, exclusions):
                    continue
                yield path
                if not path.is_symlink():
                    kept.append(name)
            dirnames[:] = kept
            for name in sorted(filenames):
                path = parent / name
                if not _is_excluded(path, exclusions):
                    yield path


def protected_snapshot(
    run_root: Path | str,
    results_root: Path | str,
    *,
    project_root: Path = ROOT,
) -> dict[str, Any]:
    project = Path(project_root).resolve()
    run, results = validate_output_roots(
        run_root, results_root, project_root=project)
    entries: dict[str, dict[str, Any]] = {}
    for path in _iter_entries(project, run, results):
        relative = path.relative_to(project).as_posix()
        path_stat = path.lstat()
        mode = path_stat.st_mode
        if stat.S_ISLNK(mode):
            entries[relative] = {
                "type": "symlink", "target": os.readlink(path),
                **_metadata(path_stat),
            }
        elif stat.S_ISDIR(mode):
            entries[relative] = {
                "type": "directory", **_metadata(path_stat)}
        elif stat.S_ISREG(mode):
            digest, stable_stat = _hash_regular_file(path)
            entries[relative] = {
                "type": "regular_file",
                "size_bytes": int(stable_stat.st_size), "sha256": digest,
                **_metadata(stable_stat),
            }
        else:
            raise ProtectionError(f"unsupported protected artifact type: {path}")
    files = [row for row in entries.values() if row["type"] == "regular_file"]
    directories = [row for row in entries.values() if row["type"] == "directory"]
    symlinks = [row for row in entries.values() if row["type"] == "symlink"]
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "all pre-existing entries below runs/ and results/",
        "scope_roots": ["runs", "results"],
        "excluded_new_roots": [str(run), str(results)],
        "entry_count": len(entries),
        "file_count": len(files),
        "directory_count": len(directories),
        "symlink_count": len(symlinks),
        "total_bytes": sum(row["size_bytes"] for row in files),
        "entries": entries,
        "manifest_sha256": canonical_hash(entries),
    }


def _atomic_json(path: Path, value: Any, *, exclusive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and os.path.lexists(path):
        raise FileExistsError(f"refusing to replace {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False,
                      allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(temporary, path)
            temporary.unlink()
        else:
            os.replace(temporary, path)
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


def _require_absent(path: Path, label: str) -> None:
    if not os.path.lexists(path):
        return
    raise FileExistsError(
        f"{label} root must be absent before the protection snapshot: {path}")


def record_before(
    run_root: Path | str,
    results_root: Path | str,
    *,
    project_root: Path = ROOT,
) -> tuple[Path, dict[str, Any]]:
    run, results = validate_output_roots(
        run_root, results_root, project_root=project_root)
    _require_absent(run, "run")
    _require_absent(results, "results")
    # This is deliberately computed before mkdir(run): the new root must not
    # exist until all prior artifact contents have been captured.
    snapshot = protected_snapshot(run, results, project_root=project_root)
    run.mkdir(parents=False, exist_ok=True)
    path = run / MANIFEST_NAME
    _atomic_json(path, snapshot, exclusive=True)
    return path, snapshot


def _read_manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ProtectionError(f"protection manifest is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtectionError(f"cannot read protection manifest: {path}") from error
    if not isinstance(value, dict):
        raise ProtectionError("protection manifest is not a JSON object")
    return value


def validate_manifest(
    manifest: dict[str, Any], run: Path, results: Path, project: Path,
) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ProtectionError("invalid protection manifest schema")
    if manifest.get("scope_roots") != ["runs", "results"]:
        raise ProtectionError("invalid protection manifest scope")
    if manifest.get("excluded_new_roots") != [str(run), str(results)]:
        raise ProtectionError("protection manifest exclusion mismatch")
    entries = manifest.get("entries")
    if not isinstance(entries, dict):
        raise ProtectionError("protection manifest has no entries mapping")
    if canonical_hash(entries) != manifest.get("manifest_sha256"):
        raise ProtectionError("protection manifest content hash mismatch")
    files = [row for row in entries.values()
             if isinstance(row, dict) and row.get("type") == "regular_file"]
    directories = [row for row in entries.values()
                   if isinstance(row, dict) and row.get("type") == "directory"]
    symlinks = [row for row in entries.values()
                if isinstance(row, dict) and row.get("type") == "symlink"]
    derived = {
        "entry_count": len(entries), "file_count": len(files),
        "directory_count": len(directories), "symlink_count": len(symlinks),
        "total_bytes": sum(int(row.get("size_bytes", -1)) for row in files),
    }
    mismatch = {key: (manifest.get(key), value)
                for key, value in derived.items() if manifest.get(key) != value}
    if mismatch:
        raise ProtectionError(f"protection manifest summary mismatch: {mismatch}")
    for relative in entries:
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise ProtectionError(f"invalid protected relative path: {relative!r}")
        absolute = (project / pure).absolute()
        if _is_excluded(absolute, (run.absolute(), results.absolute())):
            raise ProtectionError(f"manifest includes excluded output: {relative}")
        record = entries[relative]
        if not isinstance(record, dict) or record.get("type") not in {
                "regular_file", "directory", "symlink"}:
            raise ProtectionError(f"malformed protection entry: {relative}")
        if record["type"] == "regular_file" and (
                not isinstance(record.get("size_bytes"), int)
                or record["size_bytes"] < 0
                or not isinstance(record.get("sha256"), str)
                or len(record["sha256"]) != 64):
            raise ProtectionError(f"malformed protected file entry: {relative}")
        metadata = ("mode", "device", "inode", "uid", "gid",
                    "mtime_ns", "ctime_ns")
        if any(not isinstance(record.get(name), int) for name in metadata):
            raise ProtectionError(
                f"protected entry omits replacement metadata: {relative}")


def compare_snapshot(expected: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    before = expected["entries"]
    after = observed["entries"]
    missing = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    changed = sorted(key for key in set(before) & set(after)
                     if before[key] != after[key])
    protected_after = {key: after[key] for key in before if key in after}
    passed = not missing and not changed
    return {
        "schema_version": SCHEMA_VERSION,
        "passed": passed,
        "before_manifest_sha256": expected["manifest_sha256"],
        "after_manifest_sha256": canonical_hash(protected_after),
        "observed_scope_manifest_sha256": observed["manifest_sha256"],
        "entry_count_before": expected["entry_count"],
        "entry_count_after": observed["entry_count"],
        "file_count_before": expected["file_count"],
        "file_count_after": observed["file_count"],
        "total_bytes_before": expected["total_bytes"],
        "total_bytes_after": observed["total_bytes"],
        "missing_paths": missing,
        "added_paths": added,
        "changed_paths": changed,
    }


def verify_after(
    run_root: Path | str,
    results_root: Path | str,
    *,
    project_root: Path = ROOT,
) -> tuple[Path, dict[str, Any]]:
    project = Path(project_root).resolve()
    run, results = validate_output_roots(
        run_root, results_root, project_root=project)
    expected = _read_manifest(run / MANIFEST_NAME)
    validate_manifest(expected, run, results, project)
    observed = protected_snapshot(run, results, project_root=project)
    report = compare_snapshot(expected, observed)
    # Keep the results root absent for the analyzer's atomic/no-clobber
    # publication.  The run root is excluded from the protected scope and is
    # already durable, so it is the unambiguous hand-off location.
    path = run / VALIDATION_NAME
    _atomic_json(path, report, exclusive=True)
    if not report["passed"]:
        raise ProtectionError(
            "protected prior artifacts changed "
            f"(missing={len(report['missing_paths'])}, "
            f"changed={len(report['changed_paths'])})", report)
    return path, report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--before", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.before:
            path, value = record_before(args.run_root, args.results_root)
            output = {"status": "recorded", "manifest": str(path),
                      "manifest_sha256": value["manifest_sha256"],
                      "entry_count": value["entry_count"],
                      "file_count": value["file_count"],
                      "total_bytes": value["total_bytes"]}
        else:
            path, value = verify_after(args.run_root, args.results_root)
            output = {"status": "unchanged", "validation": str(path),
                      "manifest_sha256": value["after_manifest_sha256"],
                      "entry_count": value["entry_count_after"],
                      "file_count": value["file_count_after"],
                      "total_bytes": value["total_bytes_after"]}
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    except (ProtectionError, OSError, ValueError) as error:
        print(f"artifact protection failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

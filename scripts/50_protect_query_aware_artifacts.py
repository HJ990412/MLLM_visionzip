#!/usr/bin/env python3
"""Protect prior run/result artifacts around the QA-Select experiment.

The before snapshot covers only the repository's ``runs/`` and ``results/``
trees.  It records regular-file content, directory identities (including empty
directories), and symlink targets without following symlinks.  Top-level KV
stores are deliberately outside that scope: hashing tens of GiB of KV payload
would perturb the storage benchmark and is unnecessary to prove that prior
result evidence was preserved.

The only exclusions are the two explicitly supplied, dedicated query-aware
output roots.  A typical lifecycle is::

    python scripts/50_protect_query_aware_artifacts.py --before \
      --run-root runs/query_aware_baseline \
      --results-root results/query_aware_baseline

    # Run the experiment, writing only below those two roots.

    python scripts/50_protect_query_aware_artifacts.py --verify \
      --run-root runs/query_aware_baseline \
      --results-root results/query_aware_baseline

``--before`` requires absent or empty output roots and publishes an exclusive
manifest.  ``--verify`` writes a deterministic validation JSON in the new
results root and exits nonzero if any protected entry was added, removed, or
changed.
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
SCHEMA_VERSION = "query-aware-protected-artifacts-v1"
OUTPUT_PREFIXES = ("query_aware", "qa_select")
MANIFEST_NAME = "protected_artifacts_before.json"
VALIDATION_NAME = "protected_artifacts_validation.json"


class ArtifactProtectionError(RuntimeError):
    """Raised after a failed verification report has been constructed."""

    def __init__(self, message: str, report: dict[str, Any] | None = None):
        super().__init__(message)
        self.report = report


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_regular_file(path: Path, block_size: int = 8 << 20) -> tuple[str, int]:
    """Hash one regular file while failing closed on replacement or mutation."""
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ArtifactProtectionError(f"protected path is not a regular file: {path}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_mode)
        if (opened.st_dev, opened.st_ino, opened.st_mode) != identity:
            raise ArtifactProtectionError(
                f"protected file changed while opening snapshot: {path}")
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
        raise ArtifactProtectionError(
            f"protected file disappeared during snapshot: {path}") from error
    stable_fields = (
        "st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns"
    )
    if any(getattr(before, field) != getattr(after_read, field)
           or getattr(before, field) != getattr(after_path, field)
           for field in stable_fields):
        raise ArtifactProtectionError(
            f"protected file changed while hashing snapshot: {path}")
    return digest.hexdigest(), int(after_read.st_size)


def _is_query_aware_name(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in OUTPUT_PREFIXES)


def validate_output_roots(
    run_root: Path | str,
    results_root: Path | str,
    *,
    project_root: Path = ROOT,
) -> tuple[Path, Path]:
    """Resolve and constrain exclusions to dedicated direct output children."""
    project = Path(project_root).resolve()
    runs_parent = project / "runs"
    results_parent = project / "results"
    for parent, label in ((runs_parent, "runs"), (results_parent, "results")):
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError(f"{label} scope is not a real directory: {parent}")

    resolved: list[Path] = []
    for value, parent, label in (
        (run_root, runs_parent, "run"),
        (results_root, results_parent, "results"),
    ):
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = project / candidate
        # strict=False is intentional: the new root normally does not exist.
        target = candidate.resolve(strict=False)
        if target.parent != parent.resolve() or not _is_query_aware_name(target.name):
            raise ValueError(
                f"{label} root must be a direct query_aware*/qa_select* child "
                f"of {parent}: {target}")
        if os.path.lexists(target) and target.is_symlink():
            raise ValueError(f"{label} root may not be a symlink: {target}")
        resolved.append(target)

    run, results = resolved
    if run == results or run in results.parents or results in run.parents:
        raise ValueError("run and results roots overlap")
    return run, results


def _excluded(path: Path, exclusions: tuple[Path, Path]) -> bool:
    absolute = path.absolute()
    return any(absolute == root or root in absolute.parents for root in exclusions)


def _iter_protected_entries(
    project_root: Path,
    run_root: Path,
    results_root: Path,
) -> Iterable[Path]:
    """Walk runs/results deterministically without following any symlink."""
    exclusions = (run_root.absolute(), results_root.absolute())
    for scope_name in ("runs", "results"):
        scope = project_root / scope_name
        if scope.is_symlink() or not scope.is_dir():
            raise ArtifactProtectionError(
                f"protected scope is not a real directory: {scope}")
        for directory, dirnames, filenames in os.walk(scope, followlinks=False):
            parent = Path(directory)
            kept_dirs: list[str] = []
            for name in sorted(dirnames):
                path = parent / name
                if _excluded(path, exclusions):
                    continue
                yield path
                if not path.is_symlink():
                    kept_dirs.append(name)
            dirnames[:] = kept_dirs
            for name in sorted(filenames):
                path = parent / name
                if not _excluded(path, exclusions):
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
    for path in _iter_protected_entries(project, run, results):
        relative = path.relative_to(project).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            entries[relative] = {
                "type": "symlink",
                "target": os.readlink(path),
            }
        elif stat.S_ISDIR(mode):
            entries[relative] = {"type": "directory"}
        elif stat.S_ISREG(mode):
            digest, size = sha256_regular_file(path)
            entries[relative] = {
                "type": "regular_file",
                "size_bytes": size,
                "sha256": digest,
            }
        else:
            raise ArtifactProtectionError(
                f"unsupported protected artifact type: {path}")

    files = [row for row in entries.values() if row["type"] == "regular_file"]
    directories = [row for row in entries.values() if row["type"] == "directory"]
    symlinks = [row for row in entries.values() if row["type"] == "symlink"]
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": (
            "all entries below runs/ and results/ only; regular-file content, "
            "directories, and symlink targets are protected; top-level KV "
            "stores are not scanned"
        ),
        "scope_roots": ["runs", "results"],
        "excluded_new_roots": [str(run), str(results)],
        "kvstore_trees_hashed": False,
        "entry_count": len(entries),
        "file_count": len(files),
        "directory_count": len(directories),
        "symlink_count": len(symlinks),
        "total_bytes": sum(row["size_bytes"] for row in files),
        "entries": entries,
        "manifest_sha256": canonical_hash(entries),
    }


def _validate_manifest(
    manifest: dict[str, Any], run_root: Path, results_root: Path,
    project_root: Path,
) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactProtectionError("invalid protection manifest schema")
    if manifest.get("scope_roots") != ["runs", "results"]:
        raise ArtifactProtectionError("invalid protection manifest scope")
    if manifest.get("kvstore_trees_hashed") is not False:
        raise ArtifactProtectionError("invalid KV-store hashing contract")
    if manifest.get("excluded_new_roots") != [str(run_root), str(results_root)]:
        raise ArtifactProtectionError("protection manifest exclusion mismatch")
    entries = manifest.get("entries")
    if not isinstance(entries, dict):
        raise ArtifactProtectionError("protection manifest has no entries mapping")
    if canonical_hash(entries) != manifest.get("manifest_sha256"):
        raise ArtifactProtectionError("protection manifest content hash mismatch")
    excluded_relatives = {
        run_root.relative_to(project_root).as_posix(),
        results_root.relative_to(project_root).as_posix(),
    }
    hexadecimal = set("0123456789abcdef")
    for relative, row in entries.items():
        if not isinstance(relative, str) or not isinstance(row, dict):
            raise ArtifactProtectionError("protection manifest has malformed entries")
        pure = PurePosixPath(relative)
        if (pure.is_absolute() or ".." in pure.parts or len(pure.parts) < 2
                or pure.parts[0] not in {"runs", "results"}):
            raise ArtifactProtectionError(
                f"protection manifest path escapes scope: {relative!r}")
        if any(relative == root or relative.startswith(root + "/")
               for root in excluded_relatives):
            raise ArtifactProtectionError(
                f"protection manifest includes excluded output: {relative}")
        kind = row.get("type")
        if kind == "regular_file":
            size = row.get("size_bytes")
            digest = row.get("sha256")
            if (set(row) != {"type", "size_bytes", "sha256"}
                    or not isinstance(size, int) or isinstance(size, bool)
                    or size < 0 or not isinstance(digest, str)
                    or len(digest) != 64 or any(c not in hexadecimal
                                                for c in digest)):
                raise ArtifactProtectionError(
                    f"malformed regular-file entry: {relative}")
        elif kind == "directory":
            if row != {"type": "directory"}:
                raise ArtifactProtectionError(
                    f"malformed directory entry: {relative}")
        elif kind == "symlink":
            if (set(row) != {"type", "target"}
                    or not isinstance(row.get("target"), str)):
                raise ArtifactProtectionError(
                    f"malformed symlink entry: {relative}")
        else:
            raise ArtifactProtectionError(
                f"unsupported manifest entry: {relative}")
    values = list(entries.values())
    files = [row for row in values if row.get("type") == "regular_file"]
    directories = [row for row in values if row.get("type") == "directory"]
    symlinks = [row for row in values if row.get("type") == "symlink"]
    counts = {
        "entry_count": len(values),
        "file_count": len(files),
        "directory_count": len(directories),
        "symlink_count": len(symlinks),
        "total_bytes": sum(int(row.get("size_bytes", -1)) for row in files),
    }
    for key, expected in counts.items():
        if manifest.get(key) != expected:
            raise ArtifactProtectionError(
                f"protection manifest {key} mismatch")


def compare_snapshot(
    expected: dict[str, Any], observed: dict[str, Any]
) -> dict[str, Any]:
    before = expected["entries"]
    after = observed["entries"]
    missing = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    changed = sorted(
        key for key in set(before) & set(after) if before[key] != after[key]
    )
    passed = not (missing or added or changed)
    return {
        "schema_version": SCHEMA_VERSION,
        "passed": passed,
        "before_manifest_sha256": expected["manifest_sha256"],
        "after_manifest_sha256": observed["manifest_sha256"],
        "entry_count_before": expected["entry_count"],
        "entry_count_after": observed["entry_count"],
        "file_count_before": expected["file_count"],
        "file_count_after": observed["file_count"],
        "directory_count_before": expected["directory_count"],
        "directory_count_after": observed["directory_count"],
        "symlink_count_before": expected["symlink_count"],
        "symlink_count_after": observed["symlink_count"],
        "total_bytes_before": expected["total_bytes"],
        "total_bytes_after": observed["total_bytes"],
        "missing_paths": missing,
        "added_paths": added,
        "changed_paths": changed,
        "kvstore_trees_hashed": False,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: Any, *, exclusive: bool = True) -> None:
    """Durably publish JSON without replacing an existing artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError(f"JSON parent is not a real directory: {path.parent}")
    if exclusive and os.path.lexists(path):
        raise FileExistsError(f"refusing to replace existing artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True,
                      ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(temporary, path)
            temporary.unlink()
        else:
            os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _require_absent_or_empty(path: Path, label: str) -> None:
    if not os.path.lexists(path):
        return
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"new {label} root is not a real directory: {path}")
    if any(path.iterdir()):
        raise FileExistsError(
            f"new {label} root must be absent or empty before snapshot: {path}")


def record_before(
    run_root: Path | str,
    results_root: Path | str,
    *,
    project_root: Path = ROOT,
) -> tuple[Path, dict[str, Any]]:
    run, results = validate_output_roots(
        run_root, results_root, project_root=project_root)
    _require_absent_or_empty(run, "run")
    _require_absent_or_empty(results, "results")
    snapshot = protected_snapshot(
        run, results, project_root=project_root)
    run.mkdir(parents=False, exist_ok=True)
    manifest_path = run / MANIFEST_NAME
    atomic_json(manifest_path, snapshot, exclusive=True)
    return manifest_path, snapshot


def _read_manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ArtifactProtectionError(
            f"protection manifest is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactProtectionError(
            f"cannot read protection manifest: {path}") from error
    if not isinstance(value, dict):
        raise ArtifactProtectionError("protection manifest is not a JSON object")
    return value


def verify_after(
    run_root: Path | str,
    results_root: Path | str,
    *,
    project_root: Path = ROOT,
) -> tuple[Path, dict[str, Any]]:
    run, results = validate_output_roots(
        run_root, results_root, project_root=project_root)
    manifest_path = run / MANIFEST_NAME
    expected = _read_manifest(manifest_path)
    project = Path(project_root).resolve()
    _validate_manifest(expected, run, results, project)
    observed = protected_snapshot(
        run, results, project_root=project)
    report = compare_snapshot(expected, observed)
    results.mkdir(parents=False, exist_ok=True)
    validation_path = results / VALIDATION_NAME
    atomic_json(validation_path, report, exclusive=True)
    if not report["passed"]:
        summary = (
            f"missing={len(report['missing_paths'])}, "
            f"added={len(report['added_paths'])}, "
            f"changed={len(report['changed_paths'])}"
        )
        raise ArtifactProtectionError(
            f"protected prior artifacts changed ({summary})", report)
    return validation_path, report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--before", action="store_true",
                      help="record the exclusive before manifest")
    mode.add_argument("--verify", action="store_true",
                      help="verify and write the after validation")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.before:
            path, snapshot = record_before(args.run_root, args.results_root)
            output = {
                "status": "recorded",
                "manifest": str(path),
                "manifest_sha256": snapshot["manifest_sha256"],
                "entry_count": snapshot["entry_count"],
                "file_count": snapshot["file_count"],
                "total_bytes": snapshot["total_bytes"],
            }
        else:
            path, report = verify_after(args.run_root, args.results_root)
            output = {
                "status": "unchanged",
                "validation": str(path),
                "manifest_sha256": report["after_manifest_sha256"],
                "entry_count": report["entry_count_after"],
                "file_count": report["file_count_after"],
                "total_bytes": report["total_bytes_after"],
            }
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    except ArtifactProtectionError as error:
        print(f"artifact protection failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

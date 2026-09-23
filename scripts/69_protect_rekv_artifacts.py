#!/usr/bin/env python3
"""Fail-closed protection for the ReKV GQA pilot's prior artifacts.

The protected scope is every pre-existing entry below ``runs/``,
``results/``, and the existing top-level ``kvstore*`` source-store roots,
except the two *exact* output subtrees supplied on the command line.  The
supported output layout is deliberately narrow::

    runs/rekv_baseline/<run_id>
    results/rekv_baseline/gqa40_240_<run_id>

``--before`` requires both output roots to be absent.  It completes and
durably publishes the manifest below the new run root before the experiment
may create either output tree.  ``--verify`` re-snapshots the same scope and
fails if a pre-existing entry was removed, replaced, or changed.

Reading hundreds of GiB of old KV payloads would warm the storage benchmark's
page cache.  The fixed integrity policy therefore combines strong filesystem
metadata with bounded content reads:

* files no larger than 1 MiB receive a full SHA-256;
* larger files receive a framed SHA-256 over nine deterministic 4-KiB windows
  spread from the first through the last byte;
* all files additionally record size, device/inode identity, mode, owner,
  nanosecond mtime, and nanosecond ctime;
* symlinks are never followed and record their target plus lstat metadata;
* directories record stable identity/mode/owner metadata.  Directory times
  are intentionally excluded because creating the exact allowed output child
  necessarily changes its existing ancestor's mtime and ctime.

This is an accidental-corruption and no-overwrite guard, not a cryptographic
proof against an adversary able to forge inode timestamps and edit only
unsampled bytes.  The precise policy and every sampled offset are embedded in
the manifest.  A second metadata/name pass makes each snapshot fail closed if
the tree changes while it is being recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "rekv-protected-artifacts-v1"
MANIFEST_NAME = "protected_artifacts_before.json"
VALIDATION_NAME = "protected_artifacts_validation.json"
SCOPE_ROOTS = (
    "runs", "results", "kvstore", "kvstore_image_only_visionzip",
    "kvstore_image_only_work", "kvstore_multiturn",
    "kvstore_reorder_prefix_calib1",
    "kvstore_visdial_turn1_piggyback_e2e_ttft",
)
FULL_HASH_MAX_BYTES = 1 << 20
SAMPLE_WINDOW_BYTES = 4096
SAMPLE_COUNT = 9
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ProtectionError(RuntimeError):
    """Raised when a protection snapshot cannot be trusted or differs."""

    def __init__(self, message: str, report: dict[str, Any] | None = None):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class FingerprintPolicy:
    """Immutable content-fingerprinting policy recorded in the manifest."""

    full_hash_max_bytes: int = FULL_HASH_MAX_BYTES
    sample_window_bytes: int = SAMPLE_WINDOW_BYTES
    sample_count: int = SAMPLE_COUNT

    def __post_init__(self) -> None:
        if self.full_hash_max_bytes < 0:
            raise ValueError("full-hash threshold must be nonnegative")
        if self.sample_window_bytes <= 0:
            raise ValueError("sample window must be positive")
        if self.sample_count < 2:
            raise ValueError("sample count must be at least two")

    def as_dict(self) -> dict[str, Any]:
        return {
            "small_file_algorithm": "sha256-full-v1",
            "large_file_algorithm": "sha256-framed-even-windows-v1",
            "full_hash_max_bytes": self.full_hash_max_bytes,
            "sample_window_bytes": self.sample_window_bytes,
            "sample_count": self.sample_count,
            "file_metadata": [
                "size_bytes", "mode", "device", "inode", "uid", "gid",
                "mtime_ns", "ctime_ns",
            ],
            "directory_metadata": [
                "mode", "device", "inode", "uid", "gid",
            ],
            "directory_times_excluded": True,
            "symlinks_followed": False,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "FingerprintPolicy":
        if not isinstance(value, dict):
            raise ProtectionError("manifest has no fingerprint policy")
        expected_algorithms = {
            "small_file_algorithm": "sha256-full-v1",
            "large_file_algorithm": "sha256-framed-even-windows-v1",
            "directory_times_excluded": True,
            "symlinks_followed": False,
        }
        for key, expected in expected_algorithms.items():
            if value.get(key) != expected:
                raise ProtectionError(f"invalid fingerprint policy field: {key}")
        try:
            policy = cls(
                full_hash_max_bytes=value["full_hash_max_bytes"],
                sample_window_bytes=value["sample_window_bytes"],
                sample_count=value["sample_count"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ProtectionError("invalid numeric fingerprint policy") from error
        if policy.as_dict() != value:
            raise ProtectionError("fingerprint policy has unknown or altered fields")
        return policy


DEFAULT_POLICY = FingerprintPolicy()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_metadata(value: os.stat_result) -> dict[str, int]:
    return {
        "mode": int(value.st_mode),
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "uid": int(value.st_uid),
        "gid": int(value.st_gid),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _directory_metadata(value: os.stat_result) -> dict[str, int]:
    # Adding the one explicitly excluded experiment child changes ancestor
    # directory times, so those two fields cannot be invariants here.
    return {
        "mode": int(value.st_mode),
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "uid": int(value.st_uid),
        "gid": int(value.st_gid),
    }


def _stable_file_tuple(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_mode),
        int(value.st_size), int(value.st_mtime_ns), int(value.st_ctime_ns),
    )


def _sample_offsets(size: int, window: int, count: int) -> list[int]:
    """Return unique, evenly spread offsets including both file endpoints."""
    if size <= 0:
        return [0]
    width = min(size, window)
    last = size - width
    if last == 0:
        return [0]
    offsets = {(index * last) // (count - 1) for index in range(count)}
    return sorted(offsets)


def _fingerprint_regular_file(
    path: Path,
    policy: FingerprintPolicy,
) -> tuple[dict[str, Any], os.stat_result]:
    """Fingerprint a file through a no-follow descriptor and verify stability."""
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ProtectionError(f"protected path is not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if _stable_file_tuple(before) != _stable_file_tuple(opened):
            raise ProtectionError(f"protected file changed while opening: {path}")
        size = int(opened.st_size)
        digest = hashlib.sha256()
        if size <= policy.full_hash_max_bytes:
            kind = "full_sha256"
            offsets: list[int] = []
            bytes_read = 0
            while True:
                block = os.read(descriptor, 8 << 20)
                if not block:
                    break
                digest.update(block)
                bytes_read += len(block)
        else:
            kind = "sampled_sha256"
            offsets = _sample_offsets(
                size, policy.sample_window_bytes, policy.sample_count)
            bytes_read = 0
            # Frame each window with its offset and actual length so the digest
            # cannot be reproduced by rearranging equal-size chunks.
            digest.update(b"rekv-sampled-file-v1\0")
            digest.update(size.to_bytes(16, "big", signed=False))
            for offset in offsets:
                block = os.pread(descriptor, policy.sample_window_bytes, offset)
                digest.update(offset.to_bytes(16, "big", signed=False))
                digest.update(len(block).to_bytes(8, "big", signed=False))
                digest.update(block)
                bytes_read += len(block)
            if bytes_read <= 0:
                raise ProtectionError(f"sampled no bytes from nonempty file: {path}")
        after_read = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = path.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise ProtectionError(f"protected file disappeared: {path}") from error
    if (_stable_file_tuple(before) != _stable_file_tuple(after_read)
            or _stable_file_tuple(before) != _stable_file_tuple(after_path)):
        raise ProtectionError(f"protected file changed while fingerprinting: {path}")
    integrity: dict[str, Any] = {
        "kind": kind,
        "sha256": digest.hexdigest(),
        "bytes_read": bytes_read,
    }
    if kind == "sampled_sha256":
        integrity.update({
            "sample_offsets": offsets,
            "sample_window_bytes": policy.sample_window_bytes,
        })
    return integrity, after_path


def _validate_scope_parent(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} scope is not a real directory: {path}")


def validate_output_roots(
    run_root: Path | str,
    results_root: Path | str,
    *,
    project_root: Path = ROOT,
) -> tuple[Path, Path]:
    """Resolve and constrain exclusions to the dedicated nested ReKV roots."""
    project = Path(project_root).resolve()
    runs = project / "runs"
    results = project / "results"
    _validate_scope_parent(runs, "runs")
    _validate_scope_parent(results, "results")
    run_parent = runs / "rekv_baseline"
    results_parent = results / "rekv_baseline"
    for parent, label in ((run_parent, "run"), (results_parent, "results")):
        if os.path.lexists(parent) and (parent.is_symlink() or not parent.is_dir()):
            raise ValueError(f"{label} ReKV parent is not a real directory: {parent}")

    candidates: list[Path] = []
    for value in (run_root, results_root):
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = project / candidate
        candidates.append(candidate.resolve(strict=False))
    run, result = candidates
    if run.parent != run_parent.resolve(strict=False):
        raise ValueError(
            f"run root must be runs/rekv_baseline/<run_id>: {run}")
    if result.parent != results_parent.resolve(strict=False):
        raise ValueError(
            "results root must be results/rekv_baseline/"
            f"gqa40_240_<run_id>: {result}")
    run_id = run.name
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"invalid ReKV run id: {run_id!r}")
    if result.name != f"gqa40_240_{run_id}":
        raise ValueError(
            "results root name must be gqa40_240_<run_id> matching the run root")
    for target, label in ((run, "run"), (result, "results")):
        if os.path.lexists(target) and target.is_symlink():
            raise ValueError(f"{label} root may not be a symlink: {target}")
    if run == result or run in result.parents or result in run.parents:
        raise ValueError("run and results roots overlap")
    return run, result


def _is_excluded(path: Path, exclusions: tuple[Path, Path]) -> bool:
    absolute = path.absolute()
    return any(absolute == root or root in absolute.parents for root in exclusions)


def _iter_entries(project: Path, run: Path, result: Path) -> Iterable[Path]:
    exclusions = (run.absolute(), result.absolute())
    for scope_name in SCOPE_ROOTS:
        scope = project / scope_name
        _validate_scope_parent(scope, scope_name)
        if scope_name not in {"runs", "results"}:
            yield scope
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


def _entry_still_matches(path: Path, row: dict[str, Any]) -> bool:
    try:
        value = path.lstat()
    except FileNotFoundError:
        return False
    kind = row.get("type")
    if kind == "regular_file":
        return stat.S_ISREG(value.st_mode) and all(
            row.get(key) == expected
            for key, expected in {
                "size_bytes": int(value.st_size), **_file_metadata(value),
            }.items()
        )
    if kind == "directory":
        return stat.S_ISDIR(value.st_mode) and all(
            row.get(key) == expected
            for key, expected in _directory_metadata(value).items()
        )
    if kind == "symlink":
        return stat.S_ISLNK(value.st_mode) and row.get("target") == os.readlink(path) and all(
            row.get(key) == expected
            for key, expected in _file_metadata(value).items()
        )
    return False


def _manifest_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": snapshot["schema_version"],
        "scope_roots": snapshot["scope_roots"],
        "excluded_new_roots": snapshot["excluded_new_roots"],
        "fingerprint_policy": snapshot["fingerprint_policy"],
        "entries": snapshot["entries"],
    }


def protected_snapshot(
    run_root: Path | str,
    results_root: Path | str,
    *,
    project_root: Path = ROOT,
    policy: FingerprintPolicy = DEFAULT_POLICY,
) -> dict[str, Any]:
    project = Path(project_root).resolve()
    run, result = validate_output_roots(
        run_root, results_root, project_root=project)
    entries: dict[str, dict[str, Any]] = {}
    for path in _iter_entries(project, run, result):
        relative = path.relative_to(project).as_posix()
        value = path.lstat()
        if stat.S_ISLNK(value.st_mode):
            entries[relative] = {
                "type": "symlink", "target": os.readlink(path),
                **_file_metadata(value),
            }
        elif stat.S_ISDIR(value.st_mode):
            entries[relative] = {
                "type": "directory", **_directory_metadata(value),
            }
        elif stat.S_ISREG(value.st_mode):
            integrity, stable = _fingerprint_regular_file(path, policy)
            entries[relative] = {
                "type": "regular_file", "size_bytes": int(stable.st_size),
                "integrity": integrity, **_file_metadata(stable),
            }
        else:
            raise ProtectionError(f"unsupported protected artifact type: {path}")

    # A cheap second pass makes concurrent creation/deletion/replacement a
    # snapshot error without rereading any file payload.
    observed_paths = {
        path.relative_to(project).as_posix(): path
        for path in _iter_entries(project, run, result)
    }
    if set(observed_paths) != set(entries):
        raise ProtectionError("protected tree changed while enumerating snapshot")
    unstable = [
        relative for relative, path in observed_paths.items()
        if not _entry_still_matches(path, entries[relative])
    ]
    if unstable:
        raise ProtectionError(
            "protected entries changed while completing snapshot: "
            + ", ".join(sorted(unstable)[:10]))

    files = [row for row in entries.values() if row["type"] == "regular_file"]
    snapshot: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scope": "all pre-existing entries below runs/, results/, and kvstore* roots",
        "scope_roots": list(SCOPE_ROOTS),
        "excluded_new_roots": [str(run), str(result)],
        "fingerprint_policy": policy.as_dict(),
        "entry_count": len(entries),
        "file_count": len(files),
        "directory_count": sum(row["type"] == "directory" for row in entries.values()),
        "symlink_count": sum(row["type"] == "symlink" for row in entries.values()),
        "full_hash_file_count": sum(
            row["integrity"]["kind"] == "full_sha256" for row in files),
        "sampled_file_count": sum(
            row["integrity"]["kind"] == "sampled_sha256" for row in files),
        "total_logical_bytes": sum(row["size_bytes"] for row in files),
        "total_content_bytes_read": sum(
            row["integrity"]["bytes_read"] for row in files),
        "entries": entries,
    }
    snapshot["manifest_sha256"] = canonical_hash(_manifest_payload(snapshot))
    return snapshot


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
    if os.path.lexists(path):
        raise FileExistsError(
            f"{label} root must be absent before the protection snapshot: {path}")


def record_before(
    run_root: Path | str,
    results_root: Path | str,
    *,
    project_root: Path = ROOT,
    policy: FingerprintPolicy = DEFAULT_POLICY,
) -> tuple[Path, dict[str, Any]]:
    project = Path(project_root).resolve()
    run, result = validate_output_roots(
        run_root, results_root, project_root=project)
    _require_absent(run, "run")
    _require_absent(result, "results")
    snapshot = protected_snapshot(
        run, result, project_root=project, policy=policy)
    # The dedicated parent is expected to exist in the repository.  Keeping
    # this non-recursive ensures no unrecorded ancestor gets manufactured.
    run.mkdir(parents=False, exist_ok=False)
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


def _summary(entries: dict[str, dict[str, Any]]) -> dict[str, int]:
    files = [row for row in entries.values() if row.get("type") == "regular_file"]
    return {
        "entry_count": len(entries),
        "file_count": len(files),
        "directory_count": sum(
            row.get("type") == "directory" for row in entries.values()),
        "symlink_count": sum(
            row.get("type") == "symlink" for row in entries.values()),
        "full_hash_file_count": sum(
            row.get("integrity", {}).get("kind") == "full_sha256" for row in files),
        "sampled_file_count": sum(
            row.get("integrity", {}).get("kind") == "sampled_sha256" for row in files),
        "total_logical_bytes": sum(int(row.get("size_bytes", -1)) for row in files),
        "total_content_bytes_read": sum(
            int(row.get("integrity", {}).get("bytes_read", -1)) for row in files),
    }


def validate_manifest(
    manifest: dict[str, Any], run: Path, result: Path, project: Path,
) -> FingerprintPolicy:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ProtectionError("invalid protection manifest schema")
    if manifest.get("scope_roots") != list(SCOPE_ROOTS):
        raise ProtectionError("invalid protection manifest scope")
    if manifest.get("excluded_new_roots") != [str(run), str(result)]:
        raise ProtectionError("protection manifest exclusion mismatch")
    policy = FingerprintPolicy.from_dict(manifest.get("fingerprint_policy"))
    entries = manifest.get("entries")
    if not isinstance(entries, dict):
        raise ProtectionError("protection manifest has no entries mapping")
    if canonical_hash(_manifest_payload(manifest)) != manifest.get("manifest_sha256"):
        raise ProtectionError("protection manifest content hash mismatch")
    derived = _summary(entries)
    mismatch = {
        key: (manifest.get(key), value) for key, value in derived.items()
        if manifest.get(key) != value
    }
    if mismatch:
        raise ProtectionError(f"protection manifest summary mismatch: {mismatch}")
    exclusions = (run.absolute(), result.absolute())
    hexadecimal = set("0123456789abcdef")
    for relative, row in entries.items():
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise ProtectionError(f"invalid protected relative path: {relative!r}")
        absolute = (project / pure).absolute()
        if _is_excluded(absolute, exclusions):
            raise ProtectionError(f"manifest includes excluded output: {relative}")
        if not isinstance(row, dict) or row.get("type") not in {
                "regular_file", "directory", "symlink"}:
            raise ProtectionError(f"malformed protection entry: {relative}")
        metadata = ["mode", "device", "inode", "uid", "gid"]
        if row["type"] != "directory":
            metadata += ["mtime_ns", "ctime_ns"]
        if any(not isinstance(row.get(name), int) for name in metadata):
            raise ProtectionError(f"entry omits replacement metadata: {relative}")
        if row["type"] == "regular_file":
            integrity = row.get("integrity")
            if (not isinstance(row.get("size_bytes"), int)
                    or row["size_bytes"] < 0 or not isinstance(integrity, dict)):
                raise ProtectionError(f"malformed protected file entry: {relative}")
            digest = integrity.get("sha256")
            if (not isinstance(digest, str) or len(digest) != 64
                    or not set(digest) <= hexadecimal):
                raise ProtectionError(f"malformed content digest: {relative}")
            expected_kind = (
                "full_sha256" if row["size_bytes"] <= policy.full_hash_max_bytes
                else "sampled_sha256")
            if integrity.get("kind") != expected_kind:
                raise ProtectionError(f"wrong fingerprint kind: {relative}")
            if not isinstance(integrity.get("bytes_read"), int):
                raise ProtectionError(f"missing fingerprint byte count: {relative}")
            if expected_kind == "full_sha256":
                if integrity["bytes_read"] != row["size_bytes"]:
                    raise ProtectionError(f"wrong full-hash byte count: {relative}")
            else:
                offsets = _sample_offsets(
                    row["size_bytes"], policy.sample_window_bytes,
                    policy.sample_count)
                if (integrity.get("sample_offsets") != offsets
                        or integrity.get("sample_window_bytes")
                        != policy.sample_window_bytes):
                    raise ProtectionError(f"wrong sample layout: {relative}")
    return policy


def compare_snapshot(expected: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    before = expected["entries"]
    after = observed["entries"]
    missing = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    changed = sorted(
        key for key in set(before) & set(after) if before[key] != after[key])
    protected_after = {key: after[key] for key in before if key in after}
    return {
        "schema_version": SCHEMA_VERSION,
        "passed": not missing and not changed,
        "policy": expected["fingerprint_policy"],
        "before_manifest_sha256": expected["manifest_sha256"],
        "after_protected_entries_sha256": canonical_hash(protected_after),
        "observed_scope_manifest_sha256": observed["manifest_sha256"],
        "entry_count_before": expected["entry_count"],
        "entry_count_after": observed["entry_count"],
        "total_logical_bytes_before": expected["total_logical_bytes"],
        "total_logical_bytes_after": observed["total_logical_bytes"],
        "content_bytes_read_before": expected["total_content_bytes_read"],
        "content_bytes_read_after": observed["total_content_bytes_read"],
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
    run, result = validate_output_roots(
        run_root, results_root, project_root=project)
    expected = _read_manifest(run / MANIFEST_NAME)
    policy = validate_manifest(expected, run, result, project)
    observed = protected_snapshot(
        run, result, project_root=project, policy=policy)
    report = compare_snapshot(expected, observed)
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
            output = {
                "status": "recorded", "manifest": str(path),
                "manifest_sha256": value["manifest_sha256"],
                "entry_count": value["entry_count"],
                "file_count": value["file_count"],
                "total_logical_bytes": value["total_logical_bytes"],
                "total_content_bytes_read": value["total_content_bytes_read"],
            }
        else:
            path, value = verify_after(args.run_root, args.results_root)
            output = {
                "status": "unchanged", "validation": str(path),
                "manifest_sha256": value["before_manifest_sha256"],
                "entry_count": value["entry_count_after"],
                "total_logical_bytes": value["total_logical_bytes_after"],
                "content_bytes_read": value["content_bytes_read_after"],
            }
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    except (ProtectionError, OSError, ValueError) as error:
        print(f"artifact protection failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

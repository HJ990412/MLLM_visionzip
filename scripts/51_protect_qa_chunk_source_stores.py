#!/usr/bin/env python3
"""Validate and protect the canonical QA-Token/Ours stores for QA-Chunk25.

The QA-Chunk25 pilot reuses the validated 40-image raster and image-only
stores in place.  Copying roughly 92 GiB would waste disk space, while a
symlink or hard-link clone would not protect the source from accidental
writes.  This helper instead:

* validates the frozen GQA workload and the completed source run;
* checks every persisted store manifest, layout/provenance invariant, file
  name, file size, and the writer's sampled provenance hash;
* records the writer-compatible bounded payload fingerprint plus complete file
  sizes by default (the global artifact manifest supplies the independent full
  payload hash), with an optional ``--full-payload-hash`` mode; and
* compares the same deterministic snapshot after the experiment.

Typical use (after the general old-artifact snapshot, before the pilot)::

    python scripts/51_protect_qa_chunk_source_stores.py --before \
      --manifest runs/query_aware_chunk_baseline/RUN/source_stores_before.json

    python scripts/51_protect_qa_chunk_source_stores.py --verify \
      --manifest runs/query_aware_chunk_baseline/RUN/source_stores_before.json \
      --validation results/query_aware_chunk_baseline/RUN/source_stores_validation.json

No source file is opened writable.  Full mode issues ``POSIX_FADV_DONTNEED``
after hashing when supported so it does not intentionally warm the page cache
used by the cold-cache benchmark.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import stat
import sys
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "qa-chunk-source-store-provenance-v1"
EXPECTED_INDEX_SHA256 = (
    "514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a"
)
EXPECTED_WORKLOAD_SHA256 = (
    "97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66"
)
EXPECTED_MODEL = "llava-hf/llava-v1.6-vicuna-7b-hf"
EXPECTED_METHOD_KEYS = ("recompute", "fullload", "qa_select25", "ours25")
HEX = frozenset("0123456789abcdef")


class SourceStoreProtectionError(RuntimeError):
    """Fail-closed source provenance or before/after verification error."""

    def __init__(self, message: str, report: dict[str, Any] | None = None):
        super().__init__(message)
        self.report = report


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in HEX for char in value))


def _require_sha256(value: Any, label: str) -> str:
    if not _is_sha256(value):
        raise SourceStoreProtectionError(f"invalid SHA256 for {label}")
    return str(value)


def _read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise SourceStoreProtectionError(
            f"required JSON is not a regular file: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise SourceStoreProtectionError(f"cannot parse JSON: {path}") from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise SourceStoreProtectionError(
            f"required JSONL is not a regular file: {path}")
    rows: list[dict[str, Any]] = []
    line_number = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("row is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise SourceStoreProtectionError(
            f"cannot parse JSONL {path} at/near line {line_number}") from error
    return rows


def sha256_regular_file(path: Path, block_size: int = 8 << 20) \
        -> tuple[str, int]:
    """Hash a regular file read-only and detect replacement/mutation races."""
    try:
        before = path.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise SourceStoreProtectionError(f"source file disappeared: {path}") \
            from error
    if not stat.S_ISREG(before.st_mode):
        raise SourceStoreProtectionError(
            f"source path is not a regular file: {path}")
    flags = (os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
             | getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_mode)
        if (opened.st_dev, opened.st_ino, opened.st_mode) != identity:
            raise SourceStoreProtectionError(
                f"source file changed while opening: {path}")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, block_size)
            if not block:
                break
            digest.update(block)
        after_read = os.fstat(descriptor)
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            try:
                os.posix_fadvise(
                    descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError as error:
                if error.errno not in {errno.EINVAL, errno.ENOSYS}:
                    raise
    finally:
        os.close(descriptor)
    try:
        after_path = path.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise SourceStoreProtectionError(
            f"source file disappeared while hashing: {path}") from error
    stable = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns",
              "st_ctime_ns")
    if any(getattr(before, key) != getattr(after_read, key)
           or getattr(before, key) != getattr(after_path, key)
           for key in stable):
        raise SourceStoreProtectionError(
            f"source file changed while hashing: {path}")
    return digest.hexdigest(), int(after_read.st_size)


def _atomic_json_exclusive(path: Path, value: Any) -> None:
    path = path.resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError(f"output parent is not a real directory: {path.parent}")
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to replace existing artifact: {path}")
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True,
                      ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _validate_real_directory(path: Path, label: str) -> Path:
    absolute = path.resolve()
    if path.is_symlink() or not absolute.is_dir():
        raise SourceStoreProtectionError(
            f"{label} is not a real directory: {path}")
    return absolute


def _workload_identity(index_path: Path, skip: int,
                       questions_per_image: int) -> dict[str, Any]:
    index = _read_json(index_path)
    if not isinstance(index, list):
        raise SourceStoreProtectionError("GQA index must be a JSON list")
    image_ids: list[str] = []
    pairs: list[tuple[str, str]] = []
    for entry in index:
        if not isinstance(entry, dict) or "image_id" not in entry:
            raise SourceStoreProtectionError("malformed GQA index entry")
        image_id = str(entry["image_id"])
        questions = entry.get("questions")
        if not isinstance(questions, list):
            raise SourceStoreProtectionError(
                f"malformed questions for image {image_id}")
        selected = questions[skip:skip + questions_per_image]
        if len(selected) != questions_per_image:
            raise SourceStoreProtectionError(
                f"image {image_id} lacks the frozen question slice")
        image_ids.append(image_id)
        for row in selected:
            if not isinstance(row, dict) or "question_id" not in row:
                raise SourceStoreProtectionError(
                    f"malformed question for image {image_id}")
            pairs.append((image_id, str(row["question_id"])))
    if len(set(image_ids)) != len(image_ids):
        raise SourceStoreProtectionError("duplicate image IDs in GQA index")
    if len({question for _, question in pairs}) != len(pairs):
        raise SourceStoreProtectionError("duplicate question IDs in GQA index")
    index_sha, _ = sha256_regular_file(index_path)
    workload_sha = hashlib.sha256("\n".join(
        f"{image}\t{question}" for image, question in pairs
    ).encode("utf-8")).hexdigest()
    return {
        "index_sha256": index_sha,
        "workload_sha256": workload_sha,
        "image_ids": image_ids,
        "question_pairs": pairs,
        "images": len(image_ids),
        "questions": len(pairs),
    }


def _walk_regular_files(root: Path) -> tuple[dict[str, int], set[str]]:
    files: dict[str, int] = {}
    directories: set[str] = set()
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in sorted(dirnames):
            path = parent / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise SourceStoreProtectionError(
                    f"unsafe source-store directory entry: {path}")
            directories.add(path.relative_to(root).as_posix())
        for name in sorted(filenames):
            path = parent / name
            mode = path.lstat().st_mode
            if not stat.S_ISREG(mode):
                raise SourceStoreProtectionError(
                    f"unsafe source-store file entry: {path}")
            files[path.relative_to(root).as_posix()] = int(path.stat().st_size)
    return files, directories


def _sampled_store_sha256(root: Path, sizes: Mapping[str, int],
                          sample_bytes: int = 4096) -> str:
    """Reproduce the writer's persisted head/tail provenance fingerprint."""
    digest = hashlib.sha256()
    for relative in sorted(sizes):
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts:
            raise SourceStoreProtectionError(
                f"unsafe relative path in store manifest: {relative}")
        size = int(sizes[relative])
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(size.to_bytes(8, "big", signed=False))
        path = root / relative
        flags = (os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                 | getattr(os, "O_NOFOLLOW", 0))
        descriptor = os.open(path, flags)
        try:
            take = min(int(sample_bytes), size)
            head = os.pread(descriptor, take, 0)
            tail_offset = max(0, size - take)
            tail = (os.pread(descriptor, take, tail_offset)
                    if size > take else b"")
            if (hasattr(os, "posix_fadvise")
                    and hasattr(os, "POSIX_FADV_DONTNEED")):
                try:
                    os.posix_fadvise(
                        descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
                except OSError as error:
                    if error.errno not in {errno.EINVAL, errno.ENOSYS}:
                        raise
        finally:
            os.close(descriptor)
        digest.update(len(head).to_bytes(4, "big"))
        digest.update(head)
        digest.update(len(tail).to_bytes(4, "big"))
        digest.update(tail)
        digest.update(b"\0")
    return digest.hexdigest()


def _expected_store_inventory(meta: Mapping[str, Any], side: str) \
        -> tuple[set[str], set[str]]:
    layers = int(meta["num_layers"])
    files = {"meta.json", "sys_kv.pt", "sep_kv.bin"}
    files.add("v_hidden.pt" if side == "qa" else "visionzip_layout.pt")
    directories = {f"layer_{layer:02d}" for layer in range(layers)}
    for directory in directories:
        files.update({f"{directory}/k.bin", f"{directory}/v.bin",
                      f"{directory}/probe_k.bin"})
    return files, directories


def _validate_common_meta(meta: Mapping[str, Any], image_id: str,
                          model: str, expected_layers: int,
                          expected_heads: int, expected_head_dim: int,
                          expected_chunk_size: int) -> None:
    expected = {
        "image_id": image_id,
        "model": model,
        "dataset": "gqa",
        "source_turn_id": 1,
        "num_layers": expected_layers,
        "num_heads": expected_heads,
        "head_dim": expected_head_dim,
        "chunk_size": expected_chunk_size,
        "dtype": "float16",
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise SourceStoreProtectionError(
                f"{image_id}: metadata {key}={meta.get(key)!r}, expected {value!r}")
    visual = int(meta.get("v_token_num", 0))
    if visual <= 0:
        raise SourceStoreProtectionError(f"{image_id}: invalid visual span")
    if int(meta.get("n_chunks_per_layer", -1)) != math.ceil(
            visual / expected_chunk_size):
        raise SourceStoreProtectionError(f"{image_id}: invalid chunk count")
    separators = [int(value) for value in meta.get("newline_idx", [])]
    if (len(separators) != len(set(separators))
            or not all(0 <= value < visual for value in separators)):
        raise SourceStoreProtectionError(
            f"{image_id}: invalid structural separators")
    if int(meta.get("n_spatial", -1)) != visual - len(separators):
        raise SourceStoreProtectionError(f"{image_id}: invalid spatial count")
    if meta.get("turn1_normal_inference") is not True:
        raise SourceStoreProtectionError(
            f"{image_id}: store did not come from normal Turn 1")
    if meta.get("layout_source") != "turn1_normal_inference_piggyback":
        raise SourceStoreProtectionError(f"{image_id}: invalid layout source")
    if meta.get("visual_kv_source") != "turn1_captured_past_key_values":
        raise SourceStoreProtectionError(f"{image_id}: invalid KV source")
    if meta.get("capture_provenance_validated") is not True:
        raise SourceStoreProtectionError(
            f"{image_id}: capture provenance is not validated")
    _require_sha256(meta.get("image_input_sha256"),
                    f"{image_id} image input")


def _validate_layout_meta(meta: Mapping[str, Any], side: str,
                          image_id: str, expected_probe_heads: int) -> None:
    visual = int(meta["v_token_num"])
    separators = sorted(int(value) for value in meta["newline_idx"])
    if side == "qa":
        required = {
            "physical_layout": "raster",
            "layout_method": "raster",
            "reordered": False,
            "order_is_per_layer": False,
            "layout_uses_dataset_question": False,
            "layout_uses_generated_answer": False,
            "calibration_questions": 0,
            "future_questions_used": 0,
            "qa_select_compatible": True,
            "probe_heads": expected_probe_heads,
            "probe_heads_required_for_serving": expected_probe_heads,
            "visual_hidden_source": "same_turn1_decoder_layer0_input",
            "separate_vision_forward": False,
            "separate_prefix_forward": False,
            "separate_model_forward_for_visual_hidden": False,
        }
        for key, value in required.items():
            if meta.get(key) != value:
                raise SourceStoreProtectionError(
                    f"{image_id}: raster metadata {key} mismatch")
        order = meta.get("order")
        if order is not None and [int(value) for value in order] != list(
                range(visual)):
            raise SourceStoreProtectionError(
                f"{image_id}: raster store has a physical permutation")
        hidden = meta.get("hidden_capture") or {}
        if (hidden.get("capture_source")
                != "same_turn1_normal_multimodal_prefill"
                or int(hidden.get("visual_hidden_capture_count", -1)) != 1):
            raise SourceStoreProtectionError(
                f"{image_id}: invalid visual-hidden capture provenance")
        if int(meta.get("bytes_probe_sidecar", 0)) <= 0:
            raise SourceStoreProtectionError(
                f"{image_id}: missing raster probe sidecar")
    else:
        required = {
            "physical_layout": "visionzip_image_only",
            "layout_method": "visionzip_image_only",
            "reordered": True,
            "order_is_per_layer": False,
            "global_order_all_layers": True,
            "layout_uses_dataset_question": False,
            "llm_used_for_layout_scoring": False,
            "calibration_questions": 0,
            "future_questions_used_for_layout": 0,
            "probe_heads": 0,
            "probe_heads_required_for_serving": 0,
            "separator_tail": True,
        }
        for key, value in required.items():
            if meta.get(key) != value:
                raise SourceStoreProtectionError(
                    f"{image_id}: image-only metadata {key} mismatch")
        order = [int(value) for value in meta.get("order", [])]
        if len(order) != visual or sorted(order) != list(range(visual)):
            raise SourceStoreProtectionError(
                f"{image_id}: invalid image-only permutation")
        stored = [int(value) for value in meta.get("newline_stored", [])]
        expected_tail = list(range(visual - len(separators), visual))
        if stored != expected_tail:
            raise SourceStoreProtectionError(
                f"{image_id}: separators are not at image-only tail")
        _require_sha256(meta.get("permutation_sha256"),
                        f"{image_id} permutation")
        _require_sha256(meta.get("inverse_permutation_sha256"),
                        f"{image_id} inverse permutation")
    if int(meta.get("bytes_separator_sidecar", 0)) <= 0:
        raise SourceStoreProtectionError(
            f"{image_id}: missing separator sidecar")


def _validate_store_record(record: Mapping[str, Any], store: Path,
                           side: str, image_id: str, model: str,
                           expected_layers: int, expected_heads: int,
                           expected_head_dim: int, expected_chunk_size: int,
                           expected_probe_heads: int) -> dict[str, Any]:
    expected_path = store.resolve()
    if Path(str(record.get("store_dir", ""))).resolve() != expected_path:
        raise SourceStoreProtectionError(
            f"{image_id}: persisted {side} store path mismatch")
    if str(record.get("image_id")) != image_id:
        raise SourceStoreProtectionError(
            f"{image_id}: persisted {side} image ID mismatch")
    if record.get("integrity", {}).get("ok") is not True:
        raise SourceStoreProtectionError(
            f"{image_id}: persisted {side} integrity flag is false")
    meta = _read_json(store / "meta.json")
    if not isinstance(meta, dict) or record.get("meta") != meta:
        raise SourceStoreProtectionError(
            f"{image_id}: persisted and on-disk {side} metadata differ")
    _validate_common_meta(meta, image_id, model, expected_layers,
                          expected_heads, expected_head_dim,
                          expected_chunk_size)
    _validate_layout_meta(meta, side, image_id, expected_probe_heads)

    actual_sizes, directories = _walk_regular_files(store)
    recorded_sizes = record.get("file_sizes")
    if not isinstance(recorded_sizes, dict):
        raise SourceStoreProtectionError(
            f"{image_id}: missing {side} file-size manifest")
    normalised_sizes = {str(key): int(value)
                        for key, value in recorded_sizes.items()}
    if actual_sizes != normalised_sizes:
        raise SourceStoreProtectionError(
            f"{image_id}: {side} file sizes/inventory changed")
    expected_files, expected_directories = _expected_store_inventory(meta, side)
    if set(actual_sizes) != expected_files or directories != expected_directories:
        raise SourceStoreProtectionError(
            f"{image_id}: unexpected {side} file/directory inventory")

    hashes = record.get("hashes") or {}
    sampled = _sampled_store_sha256(store, actual_sizes)
    if sampled != hashes.get("prefix_kv_sample_sha256"):
        raise SourceStoreProtectionError(
            f"{image_id}: {side} sampled store hash mismatch")
    recorded_file_hashes = hashes.get("files_sha256") or {}
    for relative, expected in recorded_file_hashes.items():
        _require_sha256(expected, f"{image_id}/{side}/{relative}")
        actual, _ = sha256_regular_file(store / relative)
        if actual != expected:
            raise SourceStoreProtectionError(
                f"{image_id}: {side} recorded file hash mismatch: {relative}")

    itemsize = 2
    visual_bytes = (int(meta["num_layers"]) * 2 * int(meta["v_token_num"])
                    * int(meta["num_heads"]) * int(meta["head_dim"])
                    * itemsize)
    probe_bytes = (int(meta["num_layers"]) * int(meta["v_token_num"])
                   * int(meta["probe_heads"]) * int(meta["head_dim"])
                   * itemsize)
    if int(meta.get("bytes_visual_kv", -1)) != visual_bytes:
        raise SourceStoreProtectionError(
            f"{image_id}: {side} visual byte accounting mismatch")
    if int(meta.get("bytes_probe_sidecar", -1)) != probe_bytes:
        raise SourceStoreProtectionError(
            f"{image_id}: {side} probe byte accounting mismatch")
    return {
        "physical_layout": meta["physical_layout"],
        "reordered": bool(meta["reordered"]),
        "image_input_sha256": meta["image_input_sha256"],
        "sampled_store_sha256": sampled,
        "file_count": len(actual_sizes),
        "bytes": sum(actual_sizes.values()),
        "v_token_num": int(meta["v_token_num"]),
        "n_spatial": int(meta["n_spatial"]),
        "n_chunks_per_layer": int(meta["n_chunks_per_layer"]),
        "bytes_visual_kv": int(meta["bytes_visual_kv"]),
        "bytes_probe_sidecar": int(meta["bytes_probe_sidecar"]),
        "bytes_separator_sidecar": int(meta["bytes_separator_sidecar"]),
    }


def _validate_source_run(source_run: Path, source_store: Path,
                         source_results: Path, workload: Mapping[str, Any],
                         expected_model: str, expected_probe_heads: int,
                         expected_chunk_size: int) \
        -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    config_path = source_run / "config.json"
    persistence_path = source_run / "persistence.jsonl"
    config = _read_json(config_path)
    if not isinstance(config, dict):
        raise SourceStoreProtectionError("source config is not an object")
    required = {
        "status": "complete",
        "dataset": "gqa",
        "index_sha256": workload["index_sha256"],
        "full_workload_sha256": workload["workload_sha256"],
        "selected_workload_sha256": workload["workload_sha256"],
        "full_images": workload["images"],
        "selected_images": workload["images"],
        "full_questions": workload["questions"],
        "selected_questions": workload["questions"],
        "n_images": workload["images"],
        "n_questions": workload["questions"],
        "model": expected_model,
        "chunk_size": expected_chunk_size,
        "probe_heads": expected_probe_heads,
        "skip": 4,
        "questions_per_image": 6,
        "future_question_leakage": 0,
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise SourceStoreProtectionError(
                f"source config {key}={config.get(key)!r}, expected {value!r}")
    if tuple(config.get("method_keys", ())) != EXPECTED_METHOD_KEYS:
        raise SourceStoreProtectionError("source method set changed")
    if Path(str(config.get("run_dir", ""))).resolve() != source_run:
        raise SourceStoreProtectionError("source config run_dir mismatch")
    if Path(str(config.get("store_dir", ""))).resolve() != source_store:
        raise SourceStoreProtectionError("source config store_dir mismatch")
    if Path(str(config.get("results_dir", ""))).resolve() != source_results:
        raise SourceStoreProtectionError("source config results_dir mismatch")

    run_artifacts_path = source_results / "run_artifacts.json"
    run_artifacts = _read_json(run_artifacts_path)
    if not isinstance(run_artifacts, dict):
        raise SourceStoreProtectionError("source run_artifacts is not an object")
    if (Path(str(run_artifacts.get("run_dir", ""))).resolve() != source_run
            or Path(str(run_artifacts.get("store_dir", ""))).resolve()
            != source_store):
        raise SourceStoreProtectionError("run_artifacts source paths mismatch")
    published = run_artifacts.get("files_sha256")
    if not isinstance(published, dict) or not published:
        raise SourceStoreProtectionError("missing source run artifact hashes")
    for relative, expected in published.items():
        _require_sha256(expected, f"source run artifact {relative}")
        actual, _ = sha256_regular_file(source_run / relative)
        if actual != expected:
            raise SourceStoreProtectionError(
                f"source run artifact hash mismatch: {relative}")
        result_copy = source_results / relative
        if result_copy.is_file():
            copied, _ = sha256_regular_file(result_copy)
            if copied != expected:
                raise SourceStoreProtectionError(
                    f"source results copy hash mismatch: {relative}")

    rows = _read_jsonl(persistence_path)
    control_hashes = {}
    for label, path in {
        "config.json": config_path,
        "persistence.jsonl": persistence_path,
        "run_artifacts.json": run_artifacts_path,
    }.items():
        control_hashes[label] = sha256_regular_file(path)[0]
    return config, rows, control_hashes


def build_snapshot(
    source_store: Path | str,
    source_run: Path | str,
    source_results: Path | str,
    index: Path | str,
    *,
    expected_index_sha256: str = EXPECTED_INDEX_SHA256,
    expected_workload_sha256: str = EXPECTED_WORKLOAD_SHA256,
    expected_images: int = 40,
    expected_questions: int = 240,
    expected_model: str = EXPECTED_MODEL,
    expected_layers: int = 32,
    expected_heads: int = 32,
    expected_head_dim: int = 128,
    expected_probe_heads: int = 3,
    expected_chunk_size: int = 64,
    full_payload_hash: bool = False,
) -> dict[str, Any]:
    """Return a deterministic source-store provenance snapshot.

    The fast default revalidates the persisted 4 KiB head/tail fingerprint of
    every file in every store.  ``full_payload_hash`` additionally rereads and
    hashes all KV bytes.  Production uses the fast mode together with the
    independent global old-artifact manifest, which already full-hashes the
    source tree.
    """
    store = _validate_real_directory(Path(source_store), "source store")
    run = _validate_real_directory(Path(source_run), "source run")
    results = _validate_real_directory(Path(source_results), "source results")
    index_path = Path(index).resolve()
    if store == run or store in run.parents or run in store.parents:
        raise SourceStoreProtectionError(
            "source store and source run must be separate trees")
    workload = _workload_identity(index_path, 4, 6)
    if workload["index_sha256"] != expected_index_sha256:
        raise SourceStoreProtectionError("frozen GQA index SHA256 mismatch")
    if workload["workload_sha256"] != expected_workload_sha256:
        raise SourceStoreProtectionError("frozen GQA workload SHA256 mismatch")
    if (workload["images"] != expected_images
            or workload["questions"] != expected_questions):
        raise SourceStoreProtectionError(
            "frozen GQA image/question count mismatch")
    _, persistence, control_hashes = _validate_source_run(
        run, store, results, workload, expected_model,
        expected_probe_heads, expected_chunk_size)
    image_ids = list(workload["image_ids"])
    if len(persistence) != expected_images:
        raise SourceStoreProtectionError(
            f"source persistence has {len(persistence)} images, expected "
            f"{expected_images}")
    persisted_ids = [str(row.get("image_id")) for row in persistence]
    if persisted_ids != image_ids or len(set(persisted_ids)) != len(image_ids):
        raise SourceStoreProtectionError(
            "source persistence image IDs/order differ from frozen index")
    root_entries = {path.name for path in store.iterdir()}
    if root_entries != {"raster", "image_only"}:
        raise SourceStoreProtectionError(
            "source store root must contain only raster and image_only")
    for layout in ("raster", "image_only"):
        layout_root = _validate_real_directory(store / layout, layout)
        children = {path.name for path in layout_root.iterdir()
                    if path.is_dir() and not path.is_symlink()}
        all_children = {path.name for path in layout_root.iterdir()}
        if children != set(image_ids) or all_children != children:
            raise SourceStoreProtectionError(
                f"{layout} image inventory differs from frozen workload")

    summaries: dict[str, dict[str, Any]] = {}
    for row in persistence:
        image_id = str(row["image_id"])
        if set(row) != {"image_id", "qa", "ours"}:
            raise SourceStoreProtectionError(
                f"{image_id}: unexpected persistence record fields")
        qa = _validate_store_record(
            row["qa"], store / "raster" / image_id, "qa", image_id,
            expected_model, expected_layers, expected_heads,
            expected_head_dim, expected_chunk_size, expected_probe_heads)
        ours = _validate_store_record(
            row["ours"], store / "image_only" / image_id, "ours", image_id,
            expected_model, expected_layers, expected_heads,
            expected_head_dim, expected_chunk_size, expected_probe_heads)
        shared = ("image_input_sha256", "v_token_num", "n_spatial",
                  "n_chunks_per_layer", "bytes_visual_kv")
        if any(qa[key] != ours[key] for key in shared):
            raise SourceStoreProtectionError(
                f"{image_id}: raster/image-only source identity mismatch")
        summaries[image_id] = {"raster": qa, "image_only": ours}

    # The writer-compatible sample above touches every file.  The optional
    # full pass closes its bounded-sampling gap when no independent full
    # artifact manifest is available.
    all_files, _ = _walk_regular_files(store)
    file_rows: dict[str, dict[str, Any]] = {}
    tree = hashlib.sha256() if full_payload_hash else None
    for relative in sorted(all_files):
        size = int(all_files[relative])
        file_rows[relative] = {"size_bytes": size}
        if full_payload_hash:
            digest, checked_size = sha256_regular_file(store / relative)
            if checked_size != size:
                raise SourceStoreProtectionError(
                    f"source file size changed before full hash: {relative}")
            file_rows[relative]["sha256"] = digest
            assert tree is not None
            tree.update(relative.encode("utf-8"))
            tree.update(b"\0")
            tree.update(size.to_bytes(8, "big", signed=False))
            tree.update(bytes.fromhex(digest))
            tree.update(b"\0")

    store_fingerprint = canonical_hash({
        "file_inventory_and_sizes": file_rows,
        "per_store_writer_sample_hashes_and_metadata": summaries,
        "control_file_sha256": control_hashes,
    })

    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "hash_mode": ("full_sha256_every_source_store_file"
                      if full_payload_hash else
                      "writer_head_tail_4096_every_file_plus_full_controls"),
        "full_payload_hash": bool(full_payload_hash),
        "source_store": str(store),
        "source_run": str(run),
        "source_results": str(results),
        "index": str(index_path),
        "index_sha256": workload["index_sha256"],
        "workload_sha256": workload["workload_sha256"],
        "expected_images": expected_images,
        "expected_questions": expected_questions,
        "expected_model": expected_model,
        "expected_layers": expected_layers,
        "expected_heads": expected_heads,
        "expected_head_dim": expected_head_dim,
        "expected_probe_heads": expected_probe_heads,
        "expected_chunk_size": expected_chunk_size,
        "image_ids": image_ids,
        "control_file_sha256": control_hashes,
        "stores": summaries,
        "source_file_count": len(file_rows),
        "source_total_bytes": sum(row["size_bytes"]
                                  for row in file_rows.values()),
        "source_store_fingerprint_sha256": store_fingerprint,
        "source_store_tree_sha256": (
            tree.hexdigest() if tree is not None else None),
        "source_files": file_rows,
        "read_only_contract": (
            "source files opened O_RDONLY/O_NOFOLLOW; no store copy, link, "
            "rename, chmod, metadata rewrite, or writable descriptor"
        ),
    }
    body["manifest_sha256"] = canonical_hash(body)
    return body


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise SourceStoreProtectionError("invalid source manifest schema")
    expected_hash = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if not _is_sha256(expected_hash) or canonical_hash(unsigned) != expected_hash:
        raise SourceStoreProtectionError("source manifest content hash mismatch")
    full = manifest.get("full_payload_hash")
    if not isinstance(full, bool):
        raise SourceStoreProtectionError("source manifest hash scope is missing")
    expected_mode = ("full_sha256_every_source_store_file" if full else
                     "writer_head_tail_4096_every_file_plus_full_controls")
    if manifest.get("hash_mode") != expected_mode:
        raise SourceStoreProtectionError("source manifest hash scope mismatch")
    files = manifest.get("source_files")
    if not isinstance(files, dict) or len(files) != manifest.get(
            "source_file_count"):
        raise SourceStoreProtectionError("malformed source file manifest")
    total = 0
    for relative, row in files.items():
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or not isinstance(row, dict):
            raise SourceStoreProtectionError("unsafe source manifest entry")
        expected_fields = ({"size_bytes", "sha256"} if full
                           else {"size_bytes"})
        if set(row) != expected_fields:
            raise SourceStoreProtectionError("malformed source manifest entry")
        size = row["size_bytes"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise SourceStoreProtectionError("invalid source manifest size")
        if full:
            _require_sha256(row["sha256"], f"source manifest {relative}")
        total += size
    if total != manifest.get("source_total_bytes"):
        raise SourceStoreProtectionError("source manifest byte count mismatch")


def record_before(manifest_path: Path | str, **snapshot_kwargs: Any) \
        -> tuple[Path, dict[str, Any]]:
    path = Path(manifest_path).resolve(strict=False)
    source_paths = [Path(snapshot_kwargs[key]).resolve()
                    for key in ("source_store", "source_run",
                                "source_results")]
    if any(path == source or source in path.parents for source in source_paths):
        raise ValueError("manifest must be outside protected source trees")
    snapshot = build_snapshot(**snapshot_kwargs)
    _atomic_json_exclusive(path, snapshot)
    return path, snapshot


def verify_after(manifest_path: Path | str, validation_path: Path | str) \
        -> tuple[Path, dict[str, Any]]:
    manifest_file = Path(manifest_path).resolve()
    manifest = _read_json(manifest_file)
    if not isinstance(manifest, dict):
        raise SourceStoreProtectionError("source manifest is not an object")
    _validate_manifest(manifest)
    validation = Path(validation_path).resolve(strict=False)
    source_paths = [Path(manifest[key]).resolve()
                    for key in ("source_store", "source_run",
                                "source_results")]
    if any(validation == source or source in validation.parents
           for source in source_paths):
        raise ValueError("validation must be outside protected source trees")
    observed = build_snapshot(
        source_store=manifest["source_store"],
        source_run=manifest["source_run"],
        source_results=manifest["source_results"],
        index=manifest["index"],
        expected_index_sha256=manifest["index_sha256"],
        expected_workload_sha256=manifest["workload_sha256"],
        expected_images=int(manifest["expected_images"]),
        expected_questions=int(manifest["expected_questions"]),
        expected_model=str(manifest["expected_model"]),
        expected_layers=int(manifest["expected_layers"]),
        expected_heads=int(manifest["expected_heads"]),
        expected_head_dim=int(manifest["expected_head_dim"]),
        expected_probe_heads=int(manifest["expected_probe_heads"]),
        expected_chunk_size=int(manifest["expected_chunk_size"]),
        full_payload_hash=bool(manifest["full_payload_hash"]),
    )
    before_files = manifest["source_files"]
    after_files = observed["source_files"]
    missing = sorted(set(before_files) - set(after_files))
    added = sorted(set(after_files) - set(before_files))
    changed = sorted(
        relative for relative in set(before_files) & set(after_files)
        if before_files[relative] != after_files[relative])
    metadata_unchanged = manifest["stores"] == observed["stores"]
    controls_unchanged = (manifest["control_file_sha256"]
                          == observed["control_file_sha256"])
    passed = not (missing or added or changed) and metadata_unchanged \
        and controls_unchanged \
        and manifest["source_store_fingerprint_sha256"] == observed[
            "source_store_fingerprint_sha256"] \
        and manifest["source_store_tree_sha256"] == observed[
            "source_store_tree_sha256"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "passed": passed,
        "before_manifest_sha256": manifest["manifest_sha256"],
        "after_manifest_sha256": observed["manifest_sha256"],
        "before_store_tree_sha256": manifest["source_store_tree_sha256"],
        "after_store_tree_sha256": observed["source_store_tree_sha256"],
        "before_store_fingerprint_sha256": manifest[
            "source_store_fingerprint_sha256"],
        "after_store_fingerprint_sha256": observed[
            "source_store_fingerprint_sha256"],
        "hash_mode": manifest["hash_mode"],
        "full_payload_hash": bool(manifest["full_payload_hash"]),
        "source_file_count_before": manifest["source_file_count"],
        "source_file_count_after": observed["source_file_count"],
        "source_total_bytes_before": manifest["source_total_bytes"],
        "source_total_bytes_after": observed["source_total_bytes"],
        "missing_paths": missing,
        "added_paths": added,
        "changed_paths": changed,
        "metadata_unchanged": metadata_unchanged,
        "control_files_unchanged": controls_unchanged,
        "read_only_source_reuse_validated": passed,
    }
    _atomic_json_exclusive(validation, report)
    if not passed:
        raise SourceStoreProtectionError(
            "canonical source stores or provenance changed", report)
    return validation, report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--before", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--source-store", type=Path, default=(
        ROOT / "runs/query_aware_baseline/gqa40_240_final_store"))
    parser.add_argument("--source-run", type=Path, default=(
        ROOT / "runs/query_aware_baseline/gqa40_240_final"))
    parser.add_argument("--source-results", type=Path, default=(
        ROOT / "results/query_aware_baseline/gqa40_240_final"))
    parser.add_argument("--index", type=Path, default=ROOT / "data/index.json")
    parser.add_argument("--expected-index-sha256",
                        default=EXPECTED_INDEX_SHA256)
    parser.add_argument("--expected-workload-sha256",
                        default=EXPECTED_WORKLOAD_SHA256)
    parser.add_argument("--expected-images", type=int, default=40)
    parser.add_argument("--expected-questions", type=int, default=240)
    parser.add_argument("--expected-model", default=EXPECTED_MODEL)
    parser.add_argument("--expected-layers", type=int, default=32)
    parser.add_argument("--expected-heads", type=int, default=32)
    parser.add_argument("--expected-head-dim", type=int, default=128)
    parser.add_argument("--expected-probe-heads", type=int, default=3)
    parser.add_argument("--expected-chunk-size", type=int, default=64)
    parser.add_argument(
        "--full-payload-hash", action="store_true",
        help=("hash every KV payload byte; normally unnecessary because the "
              "global old-artifact manifest already supplies full hashes"))
    args = parser.parse_args(argv)
    if args.verify and args.validation is None:
        parser.error("--verify requires --validation")
    if args.before and args.validation is not None:
        parser.error("--validation is only valid with --verify")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.before:
            path, snapshot = record_before(
                args.manifest,
                source_store=args.source_store,
                source_run=args.source_run,
                source_results=args.source_results,
                index=args.index,
                expected_index_sha256=args.expected_index_sha256,
                expected_workload_sha256=args.expected_workload_sha256,
                expected_images=args.expected_images,
                expected_questions=args.expected_questions,
                expected_model=args.expected_model,
                expected_layers=args.expected_layers,
                expected_heads=args.expected_heads,
                expected_head_dim=args.expected_head_dim,
                expected_probe_heads=args.expected_probe_heads,
                expected_chunk_size=args.expected_chunk_size,
                full_payload_hash=args.full_payload_hash,
            )
            output = {
                "status": "recorded",
                "manifest": str(path),
                "manifest_sha256": snapshot["manifest_sha256"],
                "hash_mode": snapshot["hash_mode"],
                "source_store_fingerprint_sha256": snapshot[
                    "source_store_fingerprint_sha256"],
                "source_store_tree_sha256": snapshot[
                    "source_store_tree_sha256"],
                "source_file_count": snapshot["source_file_count"],
                "source_total_bytes": snapshot["source_total_bytes"],
            }
        else:
            assert args.validation is not None
            path, report = verify_after(args.manifest, args.validation)
            output = {
                "status": "unchanged",
                "validation": str(path),
                "hash_mode": report["hash_mode"],
                "source_store_fingerprint_sha256": report[
                    "after_store_fingerprint_sha256"],
                "source_store_tree_sha256": report[
                    "after_store_tree_sha256"],
                "source_file_count": report["source_file_count_after"],
                "source_total_bytes": report["source_total_bytes_after"],
            }
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    except (SourceStoreProtectionError, OSError, ValueError,
            FileExistsError) as error:
        print(f"source-store protection failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

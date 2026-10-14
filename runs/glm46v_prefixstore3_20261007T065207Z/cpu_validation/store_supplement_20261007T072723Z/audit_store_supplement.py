"""Run-owned supplemental CPU fixtures and exact diagnostic relocation.

This script does not modify frozen source. Mutation is limited to the twelve
new-run diagnostic files explicitly committed to mutation_allowlist.json.
"""
from __future__ import annotations

import ctypes
import datetime
import errno
import hashlib
import json
import os
from pathlib import Path
import runpy
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
RUN = ROOT / "runs/glm46v_prefixstore3_20261007T065207Z"
OUT = Path(__file__).resolve().parent
from mmimpress.glm46v.store import NativeKVSpec, open_glm_store, write_glm_store

SPEC = NativeKVSpec(40, 2, 128)
FIXTURE = runpy.run_path(str(ROOT / "tests/test_glm46v_store.py"))
libc = ctypes.CDLL(None, use_errno=True)
renameat2 = libc.renameat2
renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
renameat2.restype = ctypes.c_int


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_file(path):
    return digest_bytes(Path(path).read_bytes())


def no_replace(source, target):
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(target))


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(name, value):
    target, pending = OUT / name, OUT / ("." + name + ".pending")
    assert not target.exists() and not pending.exists()
    with pending.open("xb") as handle:
        handle.write(canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    no_replace(pending, target)
    sync_dir(OUT)
    return {"path": str(target), "sha256": digest_file(target)}


def regular_owned(path):
    relative = path.relative_to(RUN)
    current = RUN
    assert not RUN.is_symlink()
    for part in relative.parts:
        current = current / part
        assert not current.is_symlink(), f"symlink refused: {current}"
    metadata = path.lstat()
    assert stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
    assert path.resolve(strict=True).is_relative_to(RUN.resolve(strict=True))
    return metadata


def native_bytes(tensor):
    value = tensor.contiguous().cpu()
    return ctypes.string_at(value.data_ptr(), value.numel() * value.element_size())


def verify_fixture(path, fixture, meta):
    layers, content, prefix, scores, extra, identity = fixture
    selected = (list(range(len(content))) if meta["storage_policy"] == "full"
                else sorted(range(len(content)), key=lambda i: (-scores[i], i))[:(len(content) + 3) // 4])
    physical = [content[i] for i in selected]
    structure = [i for i in range(len(prefix)) if i not in content]
    logical = sorted(structure + physical)
    assert meta["stored_rows"] == len(physical) and meta["padding_rows"] == 0
    assert meta["bytes_content_kv"] == len(physical) * 40960
    for li, pair in enumerate(layers):
        for kind, tensor in zip(("k", "v"), pair):
            actual = (path / f"layer_{li:03d}/{kind}.bin").read_bytes()
            assert actual == FIXTURE["_physical_reference"](tensor, physical)
            assert len(actual) == len(physical) * 512
    expected_structure = b"".join(FIXTURE["_physical_reference"](pair[ki], structure)
                                   for ki in range(2) for pair in layers)
    assert (path / "structural_kv.bin").read_bytes() == expected_structure
    with open_glm_store(path, expected_identity=identity) as reader:
        loaded = reader.load_prefix()
        assert loaded.logical_indices == tuple(logical)
        assert loaded.selected_content_original == tuple(sorted(selected))
        assert loaded.position_ids[:, 0].tolist() == [[axis[i] for i in logical]
                                                     for axis in extra["logical_position_ids"]]
        for original, restored in zip(layers, loaded.layers):
            for tensor, value in zip(original, restored):
                assert native_bytes(value) == FIXTURE["_native_reference"](tensor, logical)
        assert loaded.io.bytes == meta["bytes_content_kv"] + meta["bytes_structural_kv"]
        assert loaded.io.preads == 81 and sum(c["actual_bytes"] for c in loaded.io.call_details) == loaded.io.bytes
        return {"request_io": loaded.io.summary(), "activation_io": reader.activation_io.summary(),
                "restored_bits_all_layers_heads_exact": True}


def cpu_boundaries(manifest):
    rows = []
    for n in manifest["synthetic_chunk_boundaries_N"]:
        started = time.perf_counter()
        fixture = list(FIXTURE["_fixture"](n, spec=SPEC))
        # Synthetic coherent one-row merged geometry; no actual pixels/model
        # inference is claimed. Prefix/content IDs are explicit CPU fixture IDs.
        fixture[4]["geometry"] = {"image_grid_thw": [1, 2, 2 * n], "spatial_merge_size": 2,
                                  "premerge_patch_count": 4 * n, "merged_visual_token_count": n}
        begin, length = fixture[1][0], len(fixture[2])
        positions = [[], [], []]
        for logical in range(length):
            point = ([logical] * 3 if logical < begin or logical >= begin + n
                     else [begin, begin, logical])
            for axis in range(3):
                positions[axis].append(point[axis])
        fixture[4]["logical_position_ids"] = positions
        fixture[4]["rope_offset"] = [0]
        identity = {key: value for key, value in fixture[4].items()
                    if key not in ("score_source", "score_layer", "logical_position_ids")}
        identity.update(prefix_input_ids=fixture[2], content_logical_indices=fixture[1],
                        logical_position_ids_sha256=digest_bytes(canonical(positions)),
                        kv_dtype="bfloat16", num_layers=40, num_kv_heads=2, head_dim=128)
        fixture[-1] = identity
        for policy in ("full", "prefix25"):
            path = OUT / "synthetic_stores" / f"N_{n:04d}" / policy
            timing = {}
            meta = write_glm_store(path, fixture[0], fixture[1], fixture[2],
                                   scores=fixture[3] if policy == "prefix25" else None,
                                   extra=fixture[4], storage_policy=policy,
                                   shape_spec=SPEC, timing_out=timing)
            metrics = verify_fixture(path, fixture, meta)
            rows.append({"N": n, "policy": policy, "k": meta["stored_content_count"],
                         "chunk_size": 64, "n_chunks": meta["n_chunks"],
                         "valid_rows_last_chunk": meta["valid_rows_last_chunk"],
                         "padding_rows": 0, "native_spec": [40, 2, 128],
                         "bytes_per_native_token": 40960, "content_bytes": meta["bytes_content_kv"],
                         "structure_bytes": meta["bytes_structural_kv"],
                         "store_logical_bytes": meta["store_logical_bytes"],
                         "timing": timing, "metadata_sha256": digest_file(path / "meta.json"),
                         "seal_sha256": digest_file(path / "seal.json"),
                         "identity_sha256": meta["identity_sha256"], "path": str(path), **metrics})
        print(f"CPU exact manifest N={n} full/B PASS, elapsed={time.perf_counter()-started:.3f}s", flush=True)
    return atomic_json("cpu_boundaries_receipt.json", {
        "status": "PASS", "scope": "CPU BF16 synthetic storage only; no GPU/model/pilot/main requests",
        "declared_N": manifest["synthetic_chunk_boundaries_N"], "stores": rows,
        "fixture_reference": "frozen independent scalar byte-offset CPU fixture",
        "fixture_source_sha256": digest_file(ROOT / "tests/test_glm46v_store.py"),
        "manifest_sha256": digest_file(RUN / "validation_manifest.json")})


def prepare_repair(receipt):
    entries = []
    for ordinal, sample in enumerate(receipt["samples"]):
        directory = Path(receipt["attempt"]) / f"sample_{ordinal:03d}_{sample['image_id']}"
        assert Path(sample["evidence"]) == directory / "evidence.json"
        evidence = json.loads((directory / "evidence.json").read_text())
        assert digest_file(directory / "evidence.json") == sample["sha256"]
        for policy in ("full", "prefix25"):
            store = directory / policy
            source = store / "attention_layers.json"
            destination = directory / "diagnostics" / policy / "attention_layers.json"
            info = regular_owned(source)
            assert not destination.exists() and not destination.is_symlink()
            for ancestor in destination.parents:
                if ancestor == RUN:
                    break
                assert not ancestor.is_symlink()
            meta = json.loads((store / "meta.json").read_text())
            seal = json.loads((store / "seal.json").read_text())
            for aux in ("meta.json", "seal.json"):
                regular_owned(store / aux)
            assert seal["metadata_sha256"] == digest_file(store / "meta.json")
            assert seal["files_manifest_sha256"] == digest_bytes(canonical(meta["files"]))
            files = {}
            for rel, record in meta["files"].items():
                regular_owned(store / rel)
                assert digest_file(store / rel) == record["sha256"]
                assert (store / rel).stat().st_size == record["size"]
                files[rel] = record
            expected_top = {"meta.json", "seal.json", "structural_kv.bin", "attention_layers.json"}
            expected_top.update(f"layer_{li:03d}" for li in range(40))
            assert {item.name for item in store.iterdir()} == expected_top
            assert json.loads(source.read_text()) == evidence["arms"][policy]["attention_fp64_layers"]
            try:
                reader = open_glm_store(store, expected_identity=meta["identity"])
                reader.close()
                raise AssertionError("expected pre-move reactivation failure was absent")
            except ValueError as error:
                assert str(error) == "unexpected/incomplete store files"
                failure = {"type": type(error).__name__, "message": str(error)}
            entries.append({"image_id": sample["image_id"], "policy": policy,
                            "source": str(source), "destination": str(destination),
                            "source_sha256": digest_file(source), "source_bytes": info.st_size,
                            "source_inode": info.st_ino, "source_device": info.st_dev,
                            "source_nlink": info.st_nlink, "metadata_sha256": digest_file(store / "meta.json"),
                            "seal_sha256": digest_file(store / "seal.json"),
                            "full_payload_hashes_before_move": files,
                            "identity": meta["identity"], "before_failure": failure})
    assert len(entries) == 12 and len({item["source"] for item in entries}) == 12
    before = atomic_json("before_reactivation_failures.json", {
        "status": "EXPECTED_FAIL", "mutations_performed": 0,
        "validation_receipt_sha256": digest_file(RUN / "validation/receipt.json"), "entries": entries})
    allowlist = atomic_json("mutation_allowlist.json", {
        "status": "COMMITTED_BEFORE_ANY_MOVE", "authorization": "root supplemental audit instruction",
        "operation": "renameat2 RENAME_NOREPLACE; source and destination directories fsynced",
        "owned_root": str(RUN), "before_failure_receipt": before, "entries": entries})
    return entries, before, allowlist


def execute_repair(entries, before, allowlist):
    results = []
    for entry in entries:
        source, destination = Path(entry["source"]), Path(entry["destination"])
        current = regular_owned(source)
        assert (current.st_ino, current.st_dev, current.st_size) == (
            entry["source_inode"], entry["source_device"], entry["source_bytes"])
        assert digest_file(source) == entry["source_sha256"]
        destination.parent.parent.mkdir(mode=0o700, exist_ok=True)
        destination.parent.mkdir(mode=0o700, exist_ok=False)
        assert destination.parent.stat().st_dev == current.st_dev
        no_replace(source, destination)
        sync_dir(source.parent)
        sync_dir(destination.parent)
        sync_dir(destination.parent.parent)
        sync_dir(destination.parent.parent.parent)
        moved = regular_owned(destination)
        assert (moved.st_ino, moved.st_dev, moved.st_size) == (
            entry["source_inode"], entry["source_device"], entry["source_bytes"])
        assert digest_file(destination) == entry["source_sha256"] and not source.exists()
        store = source.parent
        assert digest_file(store / "meta.json") == entry["metadata_sha256"]
        assert digest_file(store / "seal.json") == entry["seal_sha256"]
        for rel, record in entry["full_payload_hashes_before_move"].items():
            assert digest_file(store / rel) == record["sha256"]
        with open_glm_store(store, expected_identity=entry["identity"]) as reader:
            loaded = reader.load_prefix()
            meta = reader.meta
            logical = list(loaded.logical_indices)
            # Native bits reconstructed manually from canonical FullLoad bytes,
            # without the port reader's concat/permute/index-select assembly.
            full = store.parent / "full"
            canonical_meta = json.loads((full / "meta.json").read_text())
            structural = (full / "structural_kv.bin").read_bytes()
            structural_map = {index: row for row, index in enumerate(canonical_meta["structural_indices"])}
            content_map = {index: row for row, index in enumerate(canonical_meta["content_logical_indices"])}
            for li, pair in enumerate(loaded.layers):
                for ki, (kind, value) in enumerate(zip(("k", "v"), pair)):
                    content = (full / f"layer_{li:03d}/{kind}.bin").read_bytes()
                    tokens = []
                    for index in logical:
                        if index in structural_map:
                            offset = ((ki * 40 + li) * 19 + structural_map[index]) * 512
                            tokens.append(structural[offset:offset + 512])
                        else:
                            offset = content_map[index] * 512
                            tokens.append(content[offset:offset + 512])
                    expected = b"".join(row[head * 256:(head + 1) * 256]
                                         for head in range(2) for row in tokens)
                    assert native_bytes(value) == expected
            payload_bytes = sum(record["size"] for record in meta["files"].values())
            assert reader.activation_io.bytes == payload_bytes + meta["bytes_metadata_file"] + meta["bytes_seal_file"]
            assert loaded.io.bytes == payload_bytes
            results.append({"image_id": entry["image_id"], "policy": entry["policy"],
                            "destination": str(destination), "diagnostic_sha256": digest_file(destination),
                            "payload_hashes_unchanged": True, "metadata_seal_unchanged": True,
                            "reactivation": "PASS", "restored_bits_against_canonical_raw": "EXACT",
                            "activation_ms_cpu_audit_only": reader.activation_ms,
                            "activation_io": reader.activation_io.summary(), "request_io": loaded.io.summary()})
        print(f"Diagnostic relocation + CPU reactivation {entry['image_id']} {entry['policy']} PASS", flush=True)
    return atomic_json("repair_receipt.json", {
        "status": "PASS", "scope": "diagnostic relocation/CPU hash activation only; no GPU inference changes",
        "before_failure_receipt": before, "atomic_allowlist": allowlist, "results": results,
        "moved_files": len(results), "frozen_source_changed": False})


def main():
    receipt = json.loads((RUN / "validation/receipt.json").read_text())
    assert receipt["status"] == "PASS"
    freeze_path = Path(receipt["source_freeze_path"])
    assert digest_file(freeze_path) == receipt["source_freeze_sha256"]
    freeze = json.loads(freeze_path.read_text())
    assert all(digest_file(ROOT / rel) == expected for rel, expected in freeze["source_sha256"].items())
    manifest = json.loads((RUN / "validation_manifest.json").read_text())
    assert digest_file(RUN / "validation_manifest.json") == receipt["manifest_sha256"]
    original_review = ROOT / "results/glm46v_prefixstore3_20261007T065207Z/STORE_REVIEW.md"
    original_review_sha = digest_file(original_review)
    cpu = cpu_boundaries(manifest)
    entries, before, allowlist = prepare_repair(receipt)
    repair = execute_repair(entries, before, allowlist)
    assert all(digest_file(ROOT / rel) == expected for rel, expected in freeze["source_sha256"].items())
    assert digest_file(original_review) == original_review_sha
    result = atomic_json("receipt.json", {"status": "PASS", "cpu_boundaries": cpu,
                        "repair": repair, "source_freeze_sha256": receipt["source_freeze_sha256"],
                        "original_STORE_REVIEW_unchanged_sha256": original_review_sha,
                        "script_sha256": digest_file(__file__),
                        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        import traceback
        atomic_json("supplement_failure.json", {"status": "FAIL", "error_type": type(error).__name__,
                    "error": str(error), "traceback": traceback.format_exc()})
        raise

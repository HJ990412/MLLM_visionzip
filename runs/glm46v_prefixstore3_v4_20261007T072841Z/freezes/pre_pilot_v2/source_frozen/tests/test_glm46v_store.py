"""Independent CPU references for the native GLM PrefixStore-KV25 contract.

Expected file bits are built by manual BF16 byte offsets, without the port's
permutation, mapping, serialization or reconstruction helpers. These tests do
not certify native GLM GPU logits/position advancement or model fidelity.
"""
from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import math
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from mmimpress.glm46v.store import (
    EXPECTED_NATIVE_KV_SPEC, MODEL_ID, NativeKVSpec, inverse_permutation,
    make_store_identity, open_glm_store, stable_visual_order, write_glm_store,
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _raw(tensor):
    tensor = tensor.contiguous().cpu()
    return ctypes.string_at(tensor.data_ptr(), tensor.numel() * 2)


def _fixture(n=9, *, noncontiguous=False, spec=NativeKVSpec(3, 2, 5)):
    # Native merged raster rows can be expressed explicitly; the store does
    # not infer content from repeated token IDs or Qwen marker names.
    content = ([2 + 2 * index for index in range(n)] if noncontiguous
               else list(range(2, n + 2)))
    length = content[-1] + 3
    layers = []
    for li in range(spec.num_layers):
        pair = []
        for kind in range(2):
            tensor = torch.empty((1, spec.num_kv_heads, length + 2, spec.head_dim),
                                 dtype=torch.bfloat16)
            # Include infinities, signed zeros and noncanonical NaN bits.
            words = [(index * 31 + li * 9001 + kind * 137) % 65536
                     for index in range(tensor.numel())]
            words[:5] = [0x7FC1, 0x8000, 0x7F80, 0xFF80, 0xFFFF]
            ctypes.memmove(tensor.data_ptr(), struct.pack(f"<{len(words)}H", *words),
                           len(words) * 2)
            pair.append(tensor)
        layers.append(tuple(pair))
    prefix = [1000 + index % 7 for index in range(length)]
    scores = [float((index * 17) % 23) / 8 for index in range(n)]
    positions = [[7 + index // 2 for index in range(length)],
                 [11 + (index % 7) for index in range(length)],
                 [23 + (index % 5) for index in range(length)]]
    extra = {
        "model_id": MODEL_ID, "image_sha256": "a" * 64,
        "checkpoint_revision": "411bb4d77144a3f03accbf4b780f5acb8b7cde4e",
        "processor_revision": "411bb4d77144a3f03accbf4b780f5acb8b7cde4e",
        "processor_settings": {"native_test_resolution": [112, 168]},
        "geometry": {"image_grid_thw": [1, 8, 12], "spatial_merge_size": 2,
                     "premerge_patch_count": 96, "merged_visual_token_count": n},
        "position_policy": "native_three_axes_explicit_suffix_positions",
        "logical_position_ids": positions, "key_rope_state": "post_rope",
        "rope_config": {"partial_rotary_factor": 0.5, "mrope_section": [8, 12, 12],
                        "rotate_style": "adjacent_pair"},
        "rope_offset": [-9], "code_revision": "cpu-contract-v1",
        "environment_revision": "isolated-glm-cpu-test",
        "score_source": "vision_last_block_postsoftmax_received_attention",
        "score_layer": 23,
    }
    # Manual expected identity, independent of make_store_identity().
    identity = {key: copy.deepcopy(value) for key, value in extra.items()
                if key not in ("logical_position_ids", "score_source", "score_layer")}
    identity.update(prefix_input_ids=prefix, content_logical_indices=content,
                    logical_position_ids_sha256=_hash(positions), kv_dtype="bfloat16",
                    num_layers=spec.num_layers, num_kv_heads=spec.num_kv_heads,
                    head_dim=spec.head_dim)
    return layers, content, prefix, scores, extra, identity


def _physical_reference(tensor, logical_rows):
    # Convert the native contiguous [1,H,seq,D] byte address into token-major
    # bytes using explicit scalar offsets, with no tensor permutation helper.
    native = _raw(tensor)
    _, heads, length, dimension = tensor.shape
    result = bytearray()
    for row in logical_rows:
        for head in range(heads):
            start = (head * length * dimension + row * dimension) * 2
            result.extend(native[start:start + dimension * 2])
    return bytes(result)


def _native_reference(tensor, logical_rows):
    native = _raw(tensor)
    _, heads, length, dimension = tensor.shape
    result = bytearray()
    for head in range(heads):
        for row in logical_rows:
            start = (head * length * dimension + row * dimension) * 2
            result.extend(native[start:start + dimension * 2])
    return bytes(result)


def _write(path, n=9, *, policy="prefix25", noncontiguous=False, spec=NativeKVSpec(3, 2, 5)):
    fixture = _fixture(n, noncontiguous=noncontiguous, spec=spec)
    layers, content, prefix, scores, extra, identity = fixture
    timing = {}
    meta = write_glm_store(path, layers, content, prefix,
                           scores=scores if policy == "prefix25" else None,
                           extra=extra, storage_policy=policy, shape_spec=spec,
                           timing_out=timing)
    return fixture, meta, timing


def _resign(path, mutate):
    """Re-seal malformed fixtures to test schema, not just hash detection."""
    meta = json.loads((path / "meta.json").read_text())
    seal = json.loads((path / "seal.json").read_text())
    mutate(meta)
    for _ in range(20):
        raw = _canonical(meta) + b"\n"
        seal.update(metadata_sha256=hashlib.sha256(raw).hexdigest(), metadata_bytes=len(raw),
                    files_manifest_sha256=_hash(meta["files"]), identity_sha256=meta["identity_sha256"])
        seal_raw = _canonical(seal) + b"\n"
        logical = meta["bytes_content_kv"] + meta["bytes_structural_kv"] + len(raw) + len(seal_raw)
        if (meta["bytes_metadata_file"], meta["bytes_seal_file"], meta["store_logical_bytes"]) == (
                len(raw), len(seal_raw), logical):
            break
        meta.update(bytes_metadata_file=len(raw), bytes_seal_file=len(seal_raw), store_logical_bytes=logical)
    (path / "meta.json").write_bytes(raw)
    (path / "seal.json").write_bytes(seal_raw)


class GLMStoreTests(unittest.TestCase):
    def test_stable_float32_rank_ties_and_invalid_scores(self):
        self.assertEqual(stable_visual_order([.5, 1., 1., 0., .5]), [1, 2, 0, 4, 3])
        self.assertEqual(inverse_permutation([1, 2, 0]), [2, 0, 1])
        # These become equal in the declared float32 rank policy.
        self.assertEqual(stable_visual_order([1., 1. + 1e-10]), [0, 1])
        for scores in ([0., float("nan")], [float("inf")], [1e99], [True]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                stable_visual_order(scores)
        for order in ([0, 0], [1], [-1, 0], [0., 1], [False]):
            with self.subTest(order=order), self.assertRaises(ValueError):
                inverse_permutation(order)

    def test_expected_native_head_geometry_bytes(self):
        self.assertEqual(EXPECTED_NATIVE_KV_SPEC.bytes_per_token, 40960)
        with tempfile.TemporaryDirectory() as tmp:
            fixture, meta, _ = _write(Path(tmp) / "store", n=1, spec=EXPECTED_NATIVE_KV_SPEC)
            self.assertEqual(meta["bytes_per_native_token"], 40 * 1024)
            self.assertEqual(meta["num_kv_heads"], 2)
            self.assertEqual(meta["bytes_content_kv"], 40960)
            self.assertEqual((Path(tmp) / "store/layer_000/k.bin").stat().st_size, 512)
            with open_glm_store(Path(tmp) / "store", expected_identity=fixture[-1]) as reader:
                self.assertEqual(len(reader.load_prefix().layers), 40)

    def test_full_canonical_native_bits_positions_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "full"
            fixture, meta, timing = _write(path, policy="full", noncontiguous=True)
            layers, content, prefix, _, extra, identity = fixture
            self.assertEqual(meta["full_importance_permutation"], list(range(len(content))))
            self.assertEqual(meta["padding_rows"], 0)
            self.assertIsNone(meta["saliency_sha256"])
            self.assertEqual(make_store_identity(extra, prefix, content, NativeKVSpec(3, 2, 5)), identity)
            with open_glm_store(path, expected_identity=identity) as reader:
                loaded = reader.load_prefix()
                logical = tuple(range(len(prefix)))
                self.assertEqual(loaded.logical_indices, logical)
                self.assertEqual(loaded.prefix_input_ids, tuple(prefix))
                self.assertEqual(loaded.position_ids[:, 0].tolist(), extra["logical_position_ids"])
                self.assertEqual(loaded.rope_offset, [-9])
                self.assertEqual(loaded.native_prefix_length, len(prefix))
                for original, restored in zip(layers, loaded.layers):
                    for source, value in zip(original, restored):
                        self.assertEqual(_raw(value), _native_reference(source, logical))
                self.assertEqual(loaded.io.bytes, meta["bytes_content_kv"] + meta["bytes_structural_kv"])
                self.assertEqual(loaded.h2d_kv_bytes, loaded.io.bytes)
            self.assertEqual(timing["full_cpu_materialized_bytes"], len(prefix) * 120)
            self.assertEqual(timing["full_cpu_content_repacked_bytes"], len(content) * 120)

    def test_independent_b_rank_files_all_native_heads_layers_and_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "B"
            fixture, meta, _ = _write(path, n=25, noncontiguous=True)
            layers, content, prefix, scores, extra, identity = fixture
            # No imported ranking/mapping helper constructs this reference.
            rank = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
            k = math.ceil(len(content) / 4)
            physical = [content[index] for index in rank[:k]]
            structural = [index for index in range(len(prefix)) if index not in content]
            logical = sorted(structural + physical)
            self.assertEqual(meta["full_importance_permutation"], rank)
            self.assertEqual(meta["stored_row_to_original"], rank[:k])
            self.assertEqual(meta["original_to_stored"], [rank.index(index) for index in range(len(content))])
            structural_reference = b"".join(_physical_reference(pair[kind], structural)
                                              for kind in range(2) for pair in layers)
            self.assertEqual((path / "structural_kv.bin").read_bytes(), structural_reference)
            for li, pair in enumerate(layers):
                for kind, tensor in zip(("k", "v"), pair):
                    self.assertEqual((path / f"layer_{li:03d}/{kind}.bin").read_bytes(),
                                     _physical_reference(tensor, physical))
            with open_glm_store(path, expected_identity=identity) as reader:
                loaded = reader.load_prefix()
                self.assertEqual(loaded.logical_indices, tuple(logical))
                self.assertEqual(loaded.selected_content_original, tuple(sorted(rank[:k])))
                self.assertEqual(loaded.selected_content_logical, tuple(sorted(physical)))
                self.assertEqual(loaded.prefix_input_ids, tuple(prefix[index] for index in logical))
                self.assertEqual(loaded.position_ids[:, 0].tolist(),
                                 [[axis[index] for index in logical] for axis in extra["logical_position_ids"]])
                for source, restored in zip(layers, loaded.layers):
                    for tensor, value in zip(source, restored):
                        self.assertEqual(_raw(value), _native_reference(tensor, logical))

    def test_exact_original_N_budget_chunk_boundaries_without_padding(self):
        for n in (1, 2, 3, 4, 5, 63, 64, 65, 251, 252, 253, 255, 256, 257, 511, 512, 513, 1024):
            with self.subTest(n=n), tempfile.TemporaryDirectory() as tmp:
                fixture, meta, timing = _write(Path(tmp) / "store", n)
                k = (n + 3) // 4
                self.assertEqual(meta["stored_rows"], k)
                self.assertEqual(meta["n_chunks"], (k + 63) // 64)
                self.assertEqual(meta["valid_rows_last_chunk"], (k - 1) % 64 + 1)
                self.assertEqual(meta["padding_rows"], 0)
                self.assertEqual(len(meta["full_importance_permutation"]), n)
                self.assertEqual(meta["bytes_content_kv"], k * 120)
                self.assertEqual(timing["full_cpu_content_repacked_bytes"], n * 120)
                with open_glm_store(Path(tmp) / "store", expected_identity=fixture[-1]) as reader:
                    loaded = reader.load_prefix()
                    self.assertEqual(loaded.kept_visual_tokens, k)
                    self.assertEqual(loaded.original_content_count, n)
                    self.assertEqual(loaded.padding_rows_read, 0)
                    self.assertEqual(loaded.io.per_kind["content"]["bytes"], k * 120)

    def test_actual_first_k_sequential_pread_ranges_no_decode_read_or_resident_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store"
            fixture, meta, _ = _write(path, n=257)
            with open_glm_store(path, expected_identity=fixture[-1]) as reader:
                # Full activation integrity bytes are separate from requests.
                self.assertEqual(reader.activation_io.per_kind["activation_content"]["bytes"],
                                 meta["bytes_content_kv"])
                self.assertEqual(reader.activation_io.per_kind["activation_structure"]["bytes"],
                                 meta["bytes_structural_kv"])
                real = os.pread
                calls = []
                def observed(fd, count, offset):
                    calls.append((Path(os.readlink(f"/proc/self/fd/{fd}")).name, count, offset))
                    return real(fd, count, offset)
                with mock.patch("mmimpress.glm46v.store.os.pread", side_effect=observed):
                    loaded = reader.load_prefix()
                    self.assertEqual(len(calls), 7)
                    self.assertEqual(calls[0], ("structural_kv.bin", meta["bytes_structural_kv"], 0))
                    self.assertTrue(all((count, offset) == (65 * meta["row_bytes"], 0)
                                        for _, count, offset in calls[1:]))
                    # The runner owns decoded tensors. Accessing them triggers
                    # no additional store read and the reader retains none.
                    for _ in range(3):
                        _ = loaded.layers[0][0].shape
                    self.assertEqual(len(calls), 7)
                self.assertEqual(loaded.io.preads, 7)
                self.assertEqual(loaded.io.bytes, meta["bytes_content_kv"] + meta["bytes_structural_kv"])
                self.assertTrue(all(item["offset"] == 0 for item in loaded.io.span_details))
                self.assertEqual(sum(item["actual_bytes"] for item in loaded.io.call_details), loaded.io.bytes)
                self.assertFalse(any(isinstance(value, (torch.Tensor, bytes, bytearray))
                                     for value in vars(reader).values()))
                self.assertTrue(reader._fds)

    def test_short_reads_count_actual_offsets_calls_and_eof(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store"
            fixture, meta, _ = _write(path)
            with open_glm_store(path, expected_identity=fixture[-1]) as reader:
                real = os.pread
                seen = set()
                def short(fd, count, offset):
                    if fd not in seen:
                        seen.add(fd)
                        return real(fd, max(1, count // 3), offset)
                    return real(fd, count, offset)
                with mock.patch("mmimpress.glm46v.store.os.pread", side_effect=short):
                    loaded = reader.load_prefix()
                self.assertEqual(loaded.io.preads, 14)
                self.assertEqual(loaded.io.spans, 7)
                self.assertEqual(loaded.io.bytes, meta["bytes_content_kv"] + meta["bytes_structural_kv"])
                for span, a, b in zip(loaded.io.span_details,
                                      loaded.io.call_details[::2], loaded.io.call_details[1::2]):
                    self.assertEqual(a["offset"], span["offset"])
                    self.assertEqual(b["offset"], a["actual_bytes"])
                    self.assertEqual(a["actual_bytes"] + b["actual_bytes"], span["requested_bytes"])
                with mock.patch("mmimpress.glm46v.store.os.pread", return_value=b""):
                    with self.assertRaises(EOFError):
                        reader.load_prefix()

    def test_independent_loads_do_not_leak_dialogue_cache_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store"
            fixture, _, _ = _write(path)
            with open_glm_store(path, expected_identity=fixture[-1]) as reader:
                first = reader.load_prefix()
                expected_bits = _raw(first.layers[0][0])
                first.layers[0][0].zero_()
                first.position_ids.zero_()
                first.rope_offset[0] = 999
                second = reader.load_prefix()
                self.assertEqual(_raw(second.layers[0][0]), expected_bits)
                self.assertNotEqual(second.position_ids.sum().item(), 0)
                self.assertEqual(second.rope_offset, [-9])
                self.assertIsNot(first.layers[0][0], second.layers[0][0])

    def test_actual_os_write_bytes_seal_size_and_outer_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store"
            fixture, meta, timing = _write(path)
            files = [item for item in path.rglob("*") if item.is_file()]
            self.assertEqual(sum(item.stat().st_size for item in files), meta["store_logical_bytes"])
            self.assertEqual(meta["bytes_metadata_file"], (path / "meta.json").stat().st_size)
            self.assertEqual(meta["bytes_seal_file"], (path / "seal.json").stat().st_size)
            self.assertEqual(timing["os_write_bytes"], meta["store_logical_bytes"])
            self.assertEqual(sum(item["actual_bytes"] for item in timing["write_ranges"]), timing["os_write_bytes"])
            self.assertEqual(timing["store_allocated_bytes"], sum(item.stat().st_blocks * 512
                                                                 for item in [path, *path.rglob("*")]))
            components = sum(timing[key] for key in ("kv_materialize_ms", "kv_repack_ms",
                                                     "serialization_ms", "ssd_write_ms", "fsync_ms"))
            self.assertGreaterEqual(timing["total_persistence_ms"], components)
            with open_glm_store(path, expected_identity=fixture[-1]) as reader:
                self.assertGreater(reader.activation_ms, 0)
                self.assertEqual(reader.store_allocated_bytes, timing["store_allocated_bytes"])

    def test_short_writes_record_actual_returned_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            real = os.write
            def short(fd, payload):
                return real(fd, payload[:max(1, len(payload) // 2)])
            with mock.patch("mmimpress.glm46v.store.os.write", side_effect=short):
                fixture, meta, timing = _write(Path(tmp) / "store")
            self.assertGreater(timing["os_write_calls"], len(meta["files"]) + 2)
            self.assertEqual(timing["os_write_bytes"], meta["store_logical_bytes"])
            with open_glm_store(Path(tmp) / "store", expected_identity=fixture[-1]):
                pass

    def test_failure_preserves_unsealed_partial_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store"
            fixture = _fixture()
            timing = {}
            real = os.fsync
            counter = [0]
            def fail(fd):
                counter[0] += 1
                if counter[0] == 2:
                    raise OSError("fixture fsync failure")
                return real(fd)
            with mock.patch("mmimpress.glm46v.store.os.fsync", side_effect=fail):
                with self.assertRaisesRegex(OSError, "fixture fsync failure"):
                    write_glm_store(path, fixture[0], fixture[1], fixture[2], scores=fixture[3],
                                    extra=fixture[4], storage_policy="prefix25", timing_out=timing)
            self.assertTrue(path.is_dir())
            self.assertFalse((path / "seal.json").exists())
            self.assertGreater(timing["os_write_bytes"], 0)
            with self.assertRaises(FileNotFoundError):
                open_glm_store(path, expected_identity=fixture[-1])
            with self.assertRaises(FileExistsError):
                write_glm_store(path, fixture[0], fixture[1], fixture[2], scores=fixture[3],
                                extra=fixture[4], storage_policy="prefix25")

    def test_post_seal_durability_failure_removes_usable_completion_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store"
            fixture = _fixture()
            real = os.fsync
            def fail_after_rename(fd):
                if (path / "seal.json").exists():
                    raise OSError("fixture completion directory fsync failure")
                return real(fd)
            with mock.patch("mmimpress.glm46v.store.os.fsync", side_effect=fail_after_rename):
                with self.assertRaisesRegex(OSError, "completion directory fsync failure"):
                    write_glm_store(path, fixture[0], fixture[1], fixture[2], scores=fixture[3],
                                    extra=fixture[4], storage_policy="prefix25")
            self.assertFalse((path / "seal.json").exists())
            self.assertTrue((path / ".seal.failed").exists())
            with self.assertRaises(FileNotFoundError):
                open_glm_store(path, expected_identity=fixture[-1])

    def test_wrong_image_model_processor_prefix_positions_and_state_identity_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store"
            fixture, _, _ = _write(path)
            for key in ("image_sha256", "checkpoint_revision", "processor_revision", "position_policy",
                        "prefix_input_ids", "content_logical_indices", "logical_position_ids_sha256",
                        "rope_config", "rope_offset", "num_kv_heads", "processor_settings"):
                wrong = copy.deepcopy(fixture[-1])
                wrong[key] = "wrong fixture value"
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "cache identity mismatch"):
                    open_glm_store(path, expected_identity=wrong)
            with self.assertRaisesRegex(ValueError, "complete expected"):
                open_glm_store(path, expected_identity={"image_sha256": "a" * 64})

    def test_corrupted_short_oversized_and_missing_payloads_rejected(self):
        for case in ("corrupt", "short", "oversized", "missing"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "store"
                fixture, _, _ = _write(path)
                payload = path / "layer_000/k.bin"
                original = payload.read_bytes()
                if case == "corrupt":
                    payload.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                elif case == "short":
                    payload.write_bytes(original[:-2])
                elif case == "oversized":
                    payload.write_bytes(original + b"\x00\x00")
                else:
                    payload.unlink()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    open_glm_store(path, expected_identity=fixture[-1])

    def test_seal_and_metadata_hash_required_and_unexpected_full_sidecar_rejected(self):
        for case in ("seal_missing", "seal_incomplete", "meta_corrupt", "extra_full_payload", "seal_pending"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "store"
                fixture, _, _ = _write(path)
                if case == "seal_missing":
                    (path / "seal.json").unlink()
                elif case == "seal_incomplete":
                    seal = json.loads((path / "seal.json").read_text())
                    seal["completed"] = False
                    (path / "seal.json").write_bytes(_canonical(seal))
                elif case == "meta_corrupt":
                    raw = (path / "meta.json").read_bytes()
                    (path / "meta.json").write_bytes(raw[:-1] + b" ")
                elif case == "extra_full_payload":
                    (path / "full_content_sidecar.bin").write_bytes(b"unbudgeted content")
                else:
                    (path / ".seal.pending").write_bytes(b"incomplete")
                with self.assertRaises((ValueError, FileNotFoundError)):
                    open_glm_store(path, expected_identity=fixture[-1])

    def test_resigned_negative_budget_permutation_structure_positions_and_head_expansion(self):
        cases = {
            "double_budget": lambda m: m.update(stored_content_count=(m["stored_content_count"] + 3) // 4),
            "duplicate_content": lambda m: m["content_logical_indices"].__setitem__(1, m["content_logical_indices"][0]),
            "duplicate_structure": lambda m: m["structural_indices"].append(m["structural_indices"][0]),
            "missing_structure": lambda m: m["structural_indices"].pop(),
            "wrong_permutation": lambda m: m["full_importance_permutation"].__setitem__(0, m["full_importance_permutation"][1]),
            "wrong_position": lambda m: m["logical_position_ids"][0].__setitem__(0, 123456),
            "short_position": lambda m: m["extra"]["logical_position_ids"][0].pop(),
            "rerotated_key_state": lambda m: m.update(key_rope_state="pre_rope"),
            "expanded_heads": lambda m: m.update(num_kv_heads=16),
            "padding": lambda m: m.update(padding_rows=1),
            "float_dimension": lambda m: m.update(head_dim=5.0),
        }
        for case, mutate in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "store"
                fixture, _, _ = _write(path, n=17)
                _resign(path, mutate)
                with self.assertRaises(ValueError):
                    open_glm_store(path, expected_identity=fixture[-1])

    def test_writer_rejects_non_native_dtype_shape_ids_and_metadata_before_writes(self):
        for case in ("dtype", "heads", "length", "duplicate", "outside", "unsorted", "position",
                     "negative_position", "model", "image", "offset", "missing_metadata", "scores", "chunk"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                layers, content, prefix, scores, extra, _ = _fixture()
                kwargs = {"storage_policy": "prefix25", "shape_spec": NativeKVSpec(3, 2, 5)}
                if case == "dtype":
                    layers[0] = (layers[0][0].to(torch.float16), layers[0][1])
                elif case == "heads":
                    layers[0] = (layers[0][0][:, :1], layers[0][1][:, :1])
                elif case == "length":
                    layers[0] = (layers[0][0][:, :, :1], layers[0][1])
                elif case == "duplicate":
                    content[1] = content[0]
                elif case == "outside":
                    content[-1] = len(prefix)
                elif case == "unsorted":
                    content.reverse()
                elif case == "position":
                    extra["logical_position_ids"] = [[0] * len(prefix)]
                elif case == "negative_position":
                    extra["logical_position_ids"][0][0] = -1
                elif case == "model":
                    extra["model_id"] = "zai-org/GLM-4.6V"
                elif case == "image":
                    extra["image_sha256"] = "wrong"
                elif case == "offset":
                    extra["rope_offset"] = 3.5
                elif case == "missing_metadata":
                    del extra["processor_revision"]
                elif case == "scores":
                    scores.pop()
                else:
                    kwargs["chunk_size"] = 32
                path = Path(tmp) / "store"
                with self.assertRaises(ValueError):
                    write_glm_store(path, layers, content, prefix, scores=scores, extra=extra, **kwargs)
                self.assertFalse(path.exists())

    def test_full_importance_rejected_fixed_budget_api_and_closed_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _fixture()
            with self.assertRaises(ValueError):
                write_glm_store(Path(tmp) / "bad", fixture[0], fixture[1], fixture[2],
                                scores=fixture[3], extra=fixture[4], storage_policy="full")
            written, _, _ = _write(Path(tmp) / "store")
            reader = open_glm_store(Path(tmp) / "store", expected_identity=written[-1])
            with self.assertRaises(TypeError):
                reader.load_prefix(.25)  # A caller cannot apply a second budget.
            with self.assertRaises(TypeError):
                open_glm_store(Path(tmp) / "store", expected_identity=written[-1], verify=False)
            reader.close()
            self.assertFalse(reader._fds)
            with self.assertRaisesRegex(ValueError, "closed"):
                reader.load_prefix()

    def test_symlink_payload_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store"
            fixture, _, _ = _write(path)
            payload = path / "layer_000/k.bin"
            external = Path(tmp) / "redirected.bin"
            external.write_bytes(payload.read_bytes())
            payload.unlink()
            payload.symlink_to(external)
            with self.assertRaisesRegex(ValueError, "symbolic links"):
                open_glm_store(path, expected_identity=fixture[-1])

    def test_page_cache_hint_reports_unavailable_without_cold_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store"
            fixture, meta, _ = _write(path)
            with open_glm_store(path, expected_identity=fixture[-1]) as reader:
                with mock.patch("mmimpress.glm46v.store.os.posix_fadvise", new=None):
                    result = reader.drop_payload_cache()
                self.assertEqual(result["attempted"], len(meta["files"]))
                self.assertEqual(len(result["failed"]), result["attempted"])


if __name__ == "__main__":
    unittest.main()

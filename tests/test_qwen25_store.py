"""CPU contracts for the Qwen2.5-VL BF16 sequential visual KV store."""
from __future__ import annotations

import ctypes
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from mmimpress.qwen25.store import (
    inverse_permutation, open_qwen_store, stable_visual_order, write_qwen_store,
)


def _raw(tensor):
    tensor = tensor.contiguous().cpu()
    return ctypes.string_at(tensor.data_ptr(), tensor.numel() * 2)


def _fixture(*, dtype=torch.bfloat16):
    visual_start, visual_count, prefix_len = 2, 9, 12
    layers = []
    for li in range(2):
        base = torch.arange(2 * prefix_len * 3, dtype=torch.float32).reshape(
            1, 2, prefix_len, 3)
        layers.append(((base + li * 100).to(dtype),
                       (base + 500 + li * 100).to(dtype)))
    # A noncanonical BF16 NaN bit pattern must survive the raw-bit round trip.
    if dtype == torch.bfloat16:
        ctypes.memmove(layers[0][0].data_ptr(), bytes.fromhex("c17f"), 2)
    scores = [0.1, 0.9, 0.9, 0.2, 0.4, 0.5, 0.4, 0.3, 0.8]
    prefix_ids = list(range(100, 100 + prefix_len))
    extra = {
        "image_sha256": "a" * 64,
        "checkpoint_revision": "checkpoint-test-revision",
        "processor_revision": "checkpoint-test-revision",
        "processor_settings": {"min_pixels": 256, "max_pixels": 1024},
        "image_grid_thw": [1, 6, 6],
        "geometry": {"resized_height": 168, "resized_width": 168},
        "position_policy": "full_prompt_get_rope_index",
        "logical_position_ids": [
            list(range(prefix_len)),
            list(range(10, 10 + prefix_len)),
            list(range(20, 20 + prefix_len)),
        ],
        "key_rope_state": "post_mrope",
        "rope_deltas": [7],
        "score_source": "vision_last_block_received_attention",
        "score_layer": 31,
        "code_revision": "test-code",
        "environment_revision": "test-env",
    }
    return layers, visual_start, visual_count, prefix_ids, scores, extra


def _write(path, *, scores=True, dtype=torch.bfloat16):
    layers, start, count, ids, values, extra = _fixture(dtype=dtype)
    meta = write_qwen_store(path, layers, start, count, ids,
                            scores=values if scores else None, extra=extra,
                            chunk_size=4)
    return layers, meta


class QwenStoreTests(unittest.TestCase):
    def test_stable_global_order_and_inverse(self):
        values = [0.1, 0.9, 0.9, 0.2, 0.4, 0.5, 0.4, 0.3, 0.8]
        order = stable_visual_order(values)
        self.assertEqual(order, [1, 2, 8, 5, 4, 6, 7, 3, 0])
        inverse = inverse_permutation(order)
        self.assertEqual([order[inverse[i]] for i in range(len(order))],
                         list(range(len(order))))
        with self.assertRaises(ValueError):
            stable_visual_order([0.2, float("nan")])
        with self.assertRaises(ValueError):
            inverse_permutation([0, 0])

    def test_native_bf16_direct_repack_full_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layers, repacked = _write(root / "repacked")
            _, canonical = _write(root / "canonical", scores=False)
            self.assertEqual(repacked["stored_to_original"],
                             [1, 2, 8, 5, 4, 6, 7, 3, 0])
            self.assertEqual(canonical["stored_to_original"], list(range(9)))
            self.assertEqual(repacked["dtype"], "bfloat16")
            self.assertEqual(repacked["num_kv_heads"], 2)
            self.assertEqual(repacked["structural_indices"], [0, 1, 11])
            self.assertEqual(repacked["padding_rows"], 3)
            self.assertEqual(repacked["bytes_visual_kv"], 576)
            self.assertEqual(repacked["bytes_structural_kv"], 144)
            self.assertEqual(repacked["bytes_metadata_file"],
                             (root / "repacked" / "meta.json").stat().st_size)
            with open_qwen_store(root / "repacked") as reader:
                self.assertGreater(reader.activation_io.per_kind[
                    "activation_visual"]["bytes"], 0)
                full = reader.load_prefix(1.0)
                self.assertEqual(full.logical_indices, tuple(range(12)))
                self.assertEqual(full.io.summary()["per_kind"]["visual"]["bytes"],
                                 576)
                self.assertEqual(full.padding_rows_read, 3)
                for li, pair in enumerate(full.layers):
                    for ki, value in enumerate(pair):
                        self.assertEqual(_raw(value), _raw(layers[li][ki]))
                self.assertEqual(full.position_ids.shape, (3, 1, 12))
            with open_qwen_store(root / "canonical") as reader:
                full = reader.load_prefix(1.0)
                for li, pair in enumerate(full.layers):
                    for ki, value in enumerate(pair):
                        self.assertEqual(_raw(value), _raw(layers[li][ki]))

    def test_first_k_reads_only_contiguous_spans_and_counts_actual_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store"
            _, meta = _write(path)
            with open_qwen_store(path) as reader:
                # Activation did its full integrity scan. Only request reads
                # are observed below; a full-file scan on hit would fail.
                real_pread = os.pread
                calls = []

                def checked_pread(fd, count, offset):
                    calls.append((os.readlink(f"/proc/self/fd/{fd}"), count, offset))
                    if Path(os.readlink(f"/proc/self/fd/{fd}")).name in ("k.bin", "v.bin"):
                        self.assertEqual((count, offset), (4 * meta["row_bytes"], 0))
                    return real_pread(fd, count, offset)

                with mock.patch("mmimpress.qwen25.store.os.pread",
                                side_effect=checked_pread):
                    loaded = reader.load_prefix(0.25)
                self.assertEqual(loaded.selected_chunks, 1)
                self.assertTrue(math.isfinite(loaded.planning_ms))
                self.assertGreaterEqual(loaded.planning_ms, 0.0)
                self.assertEqual(loaded.kept_visual_tokens, 4)
                self.assertEqual(loaded.selected_visual_original, (1, 2, 5, 8))
                self.assertEqual(loaded.logical_indices, (0, 1, 3, 4, 7, 10, 11))
                self.assertEqual(loaded.io.summary()["spans"], 5)
                self.assertEqual(loaded.io.summary()["preads"], 5)
                self.assertEqual(loaded.io.summary()["bytes"], 336)
                self.assertEqual(loaded.io.summary()["per_kind"]["visual"]["bytes"],
                                 192)
                self.assertEqual(loaded.io.summary()["per_kind"]["structural"]["bytes"],
                                 144)
                spans = loaded.io.summary()["span_details"]
                self.assertEqual(sum(span["requested_bytes"] for span in spans), 336)
                self.assertEqual([span["offset"] for span in spans], [0] * 5)
                self.assertEqual(sum(span["source"].endswith("/k.bin")
                                     for span in spans), 2)
                self.assertEqual(sum(span["source"].endswith("/v.bin")
                                     for span in spans), 2)
                self.assertEqual(loaded.visual_payload_read_ratio, 1 / 3)
                self.assertEqual(loaded.total_payload_read_ratio, 336 / 720)
                self.assertEqual(len(calls), 5)
                two_chunks = reader.load_prefix(0.75)
                self.assertEqual(two_chunks.selected_chunks, 2)
                self.assertEqual(two_chunks.kept_visual_tokens, 8)
                self.assertEqual(two_chunks.io.summary()["spans"], 5)
                self.assertEqual(two_chunks.io.summary()["preads"], 5)
                self.assertEqual(two_chunks.io.summary()["per_kind"]["visual"]["bytes"],
                                 384)
                self.assertEqual([span["requested_bytes"] for span in
                                  two_chunks.io.summary()["span_details"]],
                                 [144, 96, 96, 96, 96])
                # Logical positions come from full-prompt MRoPE coordinates,
                # not from compact slot numbers.
                self.assertEqual(loaded.position_ids[:, 0, :].tolist(), [
                    list(loaded.logical_indices),
                    [10 + i for i in loaded.logical_indices],
                    [20 + i for i in loaded.logical_indices],
                ])

    def test_short_pread_retry_is_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store"
            _write(path)
            with open_qwen_store(path) as reader:
                real_pread = os.pread
                first = set()

                def short_once(fd, count, offset):
                    name = os.readlink(f"/proc/self/fd/{fd}")
                    if Path(name).name in ("k.bin", "v.bin") and fd not in first:
                        first.add(fd)
                        return real_pread(fd, count // 2, offset)
                    return real_pread(fd, count, offset)

                with mock.patch("mmimpress.qwen25.store.os.pread",
                                side_effect=short_once):
                    loaded = reader.load_prefix(0.25)
                self.assertEqual(loaded.io.summary()["spans"], 5)
                self.assertEqual(loaded.io.summary()["preads"], 9)
                self.assertEqual(loaded.io.summary()["bytes"], 336)

    def test_activation_identity_and_integrity_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store"
            layers, _ = _write(path)
            with self.assertRaises(FileExistsError):
                write_qwen_store(path, layers, 2, 9, list(range(100, 112)),
                                 scores=_fixture()[4], extra=_fixture()[5],
                                 chunk_size=4)
            with self.assertRaisesRegex(ValueError, "cache identity mismatch"):
                open_qwen_store(path, expected_identity={
                    "image_sha256": "different-image"})
            payload = path / "layer_000" / "k.bin"
            with payload.open("r+b") as handle:
                handle.seek(0)
                handle.write(b"xx")
            with self.assertRaisesRegex(ValueError, "payload hash mismatch"):
                open_qwen_store(path)

    def test_unavailable_page_cache_hint_has_nonnegative_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store"
            _write(path)
            with open_qwen_store(path) as reader:
                with mock.patch("mmimpress.qwen25.store.os.posix_fadvise",
                                new=None):
                    result = reader.drop_payload_cache()
                self.assertEqual(result["attempted"], len(reader.meta["files"]))
                self.assertEqual(len(result["failed"]), result["attempted"])
                self.assertTrue(all("path" in failure for failure in
                                    result["failed"]))

    def test_fp16_input_cannot_be_mislabeled_bf16(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "expected native BF16"):
                _write(Path(tmp) / "bad", dtype=torch.float16)
            self.assertFalse((Path(tmp) / "bad").exists())


if __name__ == "__main__":
    unittest.main()

"""CPU contracts for Qwen Visual-KV25 on the unchanged BF16 image store."""
from __future__ import annotations

import ctypes
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from mmimpress.cvpr25 import visual_kv_budget_count
from mmimpress.qwen25.runner import (LEGACY_CHUNK25_CODE_REVISION,
                                    POSITION_POLICY, Qwen25Runner)
from mmimpress.qwen25.store import (open_qwen_store, plan_prefix_budget,
                                   write_qwen_store)


CASES = (1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 349, 370, 1024)


def _raw(tensor: torch.Tensor) -> bytes:
    tensor = tensor.contiguous().cpu()
    return ctypes.string_at(tensor.data_ptr(), tensor.numel() * 2)


def _fixture(n: int, *, code_revision: str = "cpu-test-code"):
    start, prefix_len = 3, n + 4
    layers = []
    for li in range(2):
        base = torch.arange(2 * prefix_len * 2, dtype=torch.float32).reshape(
            1, 2, prefix_len, 2)
        layers.append(((base + li * 1024).to(torch.bfloat16),
                       (base + li * 1024 + 8192).to(torch.bfloat16)))
    scores = [float((i * 17) % max(n, 1)) for i in range(n)]
    prefix_ids = [1000 + i for i in range(prefix_len)]
    extra = {
        "image_sha256": "a" * 64,
        "checkpoint_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "processor_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "processor_settings": {"min_pixels": 200704, "max_pixels": 802816,
                               "use_fast": True,
                               "processor_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5"},
        "image_grid_thw": [[1, 2, 2]],
        "geometry": {"visual_start": start, "visual_count": n,
                     "prefix_len": prefix_len},
        "position_policy": POSITION_POLICY,
        "logical_position_ids": [[i + axis * 10 for i in range(prefix_len)]
                                 for axis in range(3)],
        "key_rope_state": "post_mrope",
        "score_source": "last_fullatt_ViT_received_attention",
        "score_layer": 31,
        "code_revision": code_revision,
        "environment_revision": "cpu-test-environment",
    }
    return layers, start, prefix_len, scores, prefix_ids, extra


def _write(path: Path, n: int, *, code_revision="cpu-test-code"):
    layers, start, prefix_len, scores, ids, extra = _fixture(
        n, code_revision=code_revision)
    meta = write_qwen_store(path, layers, start, n, ids, scores=scores,
                            extra=extra, chunk_size=64)
    return layers, scores, meta


class QwenKV25Tests(unittest.TestCase):
    def test_virtual_geometry_parity_and_boundaries(self):
        for n in CASES:
            with self.subTest(n=n):
                meta = {"visual_count": n, "chunk_size": 64,
                        "n_chunks": (n + 63) // 64,
                        "layout": "token_major_repacked_bf16",
                        "global_order_all_layers": True,
                        "score_source": "last_fullatt_ViT_received_attention",
                        "score_layer": 31, "saliency_sha256": "a" * 64}
                plan = plan_prefix_budget(meta, .25, budget_unit="visual_kv")
                k = visual_kv_budget_count(n, .25)
                self.assertEqual(k, (n + 3) // 4)
                self.assertEqual(plan["kept_visual_tokens"], k)
                self.assertEqual(plan["selected_chunks"], (k + 63) // 64)
                self.assertEqual(plan["loaded_valid_visual_rows"],
                                 min(n, 64 * plan["selected_chunks"]))
                self.assertEqual(plan["extra_valid_visual_rows"],
                                 min(n, 64 * plan["selected_chunks"]) - k)
        self.assertEqual((visual_kv_budget_count(349, .25),
                          plan_prefix_budget({"visual_count": 349, "chunk_size": 64,
                                              "n_chunks": 6,
                                              "layout": "token_major_repacked_bf16",
                                              "global_order_all_layers": True,
                                              "score_source": "last_fullatt_ViT_received_attention",
                                              "score_layer": 31,
                                              "saliency_sha256": "a" * 64},
                                             .25, budget_unit="visual_kv")["selected_chunks"]),
                         (88, 2))
        self.assertEqual(visual_kv_budget_count(0, 0), 0)
        self.assertEqual(visual_kv_budget_count(349, 0), 0)
        self.assertEqual(visual_kv_budget_count(349, 1), 349)
        for bad_n, bad_r in ((-1, .25), (1.5, .25), (True, .25),
                             (1, -0.1), (1, 1.01), (1, float("nan")),
                             (1, True)):
            with self.subTest(bad_n=bad_n, bad_r=bad_r):
                with self.assertRaises(ValueError):
                    visual_kv_budget_count(bad_n, bad_r)

    def test_actual_whole_chunk_reads_and_exact_compact_kv(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for n in CASES:
                with self.subTest(n=n):
                    layers, scores, meta = _write(root / f"n{n}", n)
                    k, m = (n + 3) // 4, ((n + 3) // 4 + 63) // 64
                    independent_order = sorted(range(n),
                                               key=lambda i: (-scores[i], i))
                    expected_selected = independent_order[:k]
                    logical = sorted([0, 1, 2, n + 3] +
                                     [3 + i for i in expected_selected])
                    with open_qwen_store(root / f"n{n}") as store:
                        original_pread = os.pread
                        calls = []

                        def traced_pread(fd, size, offset):
                            name = os.readlink(f"/proc/self/fd/{fd}")
                            payload = original_pread(fd, size, offset)
                            calls.append((name, size, offset, len(payload)))
                            return payload

                        with mock.patch("mmimpress.qwen25.store.os.pread",
                                        side_effect=traced_pread):
                            loaded = store.load_prefix(.25,
                                                       budget_unit="visual_kv")
                        self.assertEqual(loaded.selected_chunks, m)
                        self.assertEqual(loaded.kept_visual_tokens, k)
                        self.assertEqual(loaded.selected_visual_stored,
                                         tuple(range(k)))
                        self.assertEqual(loaded.selected_visual_original,
                                         tuple(sorted(expected_selected)))
                        self.assertEqual(loaded.logical_indices, tuple(logical))
                        self.assertEqual(loaded.structural_rows, 4)
                        self.assertEqual(loaded.loaded_valid_visual_rows,
                                         min(n, m * 64))
                        self.assertEqual(loaded.extra_valid_visual_rows,
                                         min(n, m * 64) - k)
                        self.assertEqual(loaded.padding_rows_read,
                                         m * 64 - min(n, m * 64))
                        self.assertEqual(loaded.h2d_kv_bytes,
                                         (k + 4) * meta["row_bytes"] * 2 * 2)
                        self.assertEqual(len(calls), 5)
                        visual_calls = [v for v in calls if v[0].endswith(
                            ("/k.bin", "/v.bin"))]
                        self.assertEqual(len(visual_calls), 4)
                        self.assertTrue(all((size, offset, actual) ==
                                            (m * meta["row_bytes"] * 64, 0,
                                             m * meta["row_bytes"] * 64)
                                            for _, size, offset, actual in visual_calls))
                        self.assertEqual(sum(v[3] for v in visual_calls),
                                         4 * m * 64 * meta["row_bytes"])
                        self.assertEqual(loaded.io.summary()["bytes"],
                                         sum(v[3] for v in calls))
                        self.assertEqual(loaded.io.summary()["preads"],
                                         len(calls))
                        self.assertEqual(loaded.position_ids[:, 0].tolist(),
                                         [[i + axis * 10 for i in logical]
                                          for axis in range(3)])
                        for li, pair in enumerate(loaded.layers):
                            self.assertEqual(pair[0].shape[2], k + 4)
                            for kind, observed in enumerate(pair):
                                expected = layers[li][kind].index_select(
                                    2, torch.tensor(logical))
                                self.assertEqual(_raw(observed), _raw(expected))
                        if n == 256:
                            legacy = store.load_prefix(.25)
                            self.assertEqual(legacy.kept_visual_tokens, 64)
                            self.assertEqual(legacy.logical_indices,
                                             loaded.logical_indices)
                            for legacy_pair, new_pair in zip(legacy.layers,
                                                             loaded.layers):
                                for old, new in zip(legacy_pair, new_pair):
                                    self.assertEqual(_raw(old), _raw(new))

    def test_unused_finite_rows_do_not_enter_compact_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "store"
            _, _, meta = _write(path, 349)
            with open_qwen_store(path) as store:
                baseline = store.load_prefix(.25, budget_unit="visual_kv")
                original_pread = os.pread
                row_bytes = meta["row_bytes"]
                k = baseline.kept_visual_tokens
                valid = baseline.loaded_valid_visual_rows
                sentinel_row = bytes.fromhex("8042") * (row_bytes // 2)

                def changed_pread(fd, size, offset):
                    payload = original_pread(fd, size, offset)
                    name = os.readlink(f"/proc/self/fd/{fd}")
                    if name.endswith(("/k.bin", "/v.bin")):
                        changed = bytearray(payload)
                        for row in range(k, valid):
                            changed[row * row_bytes:(row + 1) * row_bytes] = sentinel_row
                        return bytes(changed)
                    return payload

                with mock.patch("mmimpress.qwen25.store.os.pread",
                                side_effect=changed_pread):
                    altered = store.load_prefix(.25, budget_unit="visual_kv")
                self.assertEqual(altered.extra_valid_visual_rows, 40)
                self.assertEqual(altered.logical_indices, baseline.logical_indices)
                for old, new in zip(baseline.layers, altered.layers):
                    for a, b in zip(old, new):
                        self.assertEqual(_raw(a), _raw(b))
                longer = store.load_prefix(.50, budget_unit="visual_kv")
                shorter = store.load_prefix(.25, budget_unit="visual_kv")
                self.assertGreater(longer.kept_visual_tokens,
                                   shorter.kept_visual_tokens)
                self.assertEqual(shorter.layers[0][0].shape[2], k + 4)

    def test_legacy_default_and_full_visual_regressions(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "store"
            layers, _, meta = _write(path, 349)
            with open_qwen_store(path) as store:
                default = store.load_prefix(.25)
                explicit = store.load_prefix(.25, budget_unit="chunk")
                self.assertEqual(default.kept_visual_tokens, 128)
                self.assertEqual(default.selected_chunks, 2)
                self.assertEqual(default.logical_indices, explicit.logical_indices)
                for old, new in zip(default.layers, explicit.layers):
                    for a, b in zip(old, new):
                        self.assertEqual(_raw(a), _raw(b))
                full = store.load_prefix(1.0, budget_unit="visual_kv")
                self.assertEqual(full.logical_indices,
                                 tuple(range(meta["prefix_len"])))
                self.assertEqual(full.kept_visual_tokens, 349)
                self.assertEqual(full.padding_rows_read, 35)
                for li, pair in enumerate(full.layers):
                    for kind, observed in enumerate(pair):
                        self.assertEqual(_raw(observed), _raw(layers[li][kind]))

    def test_visual_mode_fails_closed_on_wrong_layout_and_bad_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "canonical"
            layers, start, _, _, ids, extra = _fixture(65)
            write_qwen_store(path, layers, start, 65, ids, scores=None,
                             extra=extra, chunk_size=64)
            with open_qwen_store(path) as store:
                with self.assertRaisesRegex(ValueError, "globally repacked"):
                    store.load_prefix(.25, budget_unit="visual_kv")
            fake = {"visual_count": 0, "chunk_size": 64, "n_chunks": 0,
                    "layout": "token_major_repacked_bf16",
                    "global_order_all_layers": True,
                    "score_source": "last_fullatt_ViT_received_attention",
                    "score_layer": 31, "saliency_sha256": "a" * 64}
            with self.assertRaisesRegex(ValueError, "nonempty"):
                plan_prefix_budget(fake, .25, budget_unit="visual_kv")
            fake["visual_count"], fake["n_chunks"] = 65, 2
            with self.assertRaisesRegex(ValueError, "positive retained"):
                plan_prefix_budget(fake, 0, budget_unit="visual_kv")
            fake["chunk_size"] = 32
            with self.assertRaisesRegex(ValueError, "64-row"):
                plan_prefix_budget(fake, .25, budget_unit="visual_kv")
            fake["chunk_size"] = 64
            with self.assertRaisesRegex(ValueError, "unknown budget"):
                plan_prefix_budget(fake, .25, budget_unit="tokens")

    def test_legacy_store_requires_exact_protected_meta_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "legacy"
            _write(path, 65, code_revision=LEGACY_CHUNK25_CODE_REVISION)
            digest = __import__("hashlib").sha256((path / "meta.json").read_bytes()).hexdigest()

            def runner(trusted):
                obj = Qwen25Runner(trusted_legacy_store_meta_sha256=trusted)
                obj._code_revision = lambda: "b" * 64
                obj._environment_revision = lambda: "cpu-test-environment"
                obj.processor_settings = lambda: _fixture(65)[5]["processor_settings"]
                return obj

            for trusted in ({}, {path: "c" * 64}):
                obj = runner(trusted)
                with self.assertRaisesRegex(ValueError, "approved protected store"):
                    obj._activate(path, "a" * 64)
                obj.close()
            obj = runner({path: digest})
            store = obj._activate(path, "a" * 64)
            self.assertEqual(store.metadata_file_sha256, digest)
            self.assertEqual(store.meta["code_revision"],
                             LEGACY_CHUNK25_CODE_REVISION)
            obj.close()


if __name__ == "__main__":
    unittest.main()

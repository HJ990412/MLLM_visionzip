"""CPU-only contracts for the importance-reorder Prefix baseline."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from mmimpress.cvpr25 import budget_chunk_count
from mmimpress.serve import (BIAS, CVPR25ChunkSelector, ImageContext)


ROOT = Path(__file__).resolve().parents[1]


class _FakeReader:
    def __init__(self, meta):
        self.meta = meta

    def read_chunks(self, layer, kind, cids, counter):
        cs, vn = self.meta["chunk_size"], self.meta["v_token_num"]
        rows = []
        for cid in cids:
            rows.extend(range(cid * cs, min((cid + 1) * cs, vn)))
        counter.record(kind, len(rows) * 2, 0.0, preads=1,
                       units=len(cids))
        return (torch.tensor(rows, dtype=torch.long),
                torch.zeros(len(rows), 1, 1, dtype=torch.float16))


class _FakeCache:
    def write(self, layer, kind, rows, values):
        return None


class _FakeContext:
    def __init__(self):
        self.meta = {
            "num_layers": 2,
            "n_chunks_per_layer": 4,
            "chunk_size": 2,
            "v_token_num": 8,
            "v_token_start": 1,
            "prefix_len": 9,
        }
        self.reader = _FakeReader(self.meta)
        self.cache = _FakeCache()

    def read_sep_kv(self, counter):
        # (K/V, layers, separator rows, heads, head dim)
        out = torch.zeros(2, 2, 1, 1, 1, dtype=torch.float16)
        counter.record("sep", out.numel() * out.element_size(), 0.0,
                       preads=1, units=0)
        return out

    def separator_positions(self, layer):
        return [7]


class PrefixSelectionContractTests(unittest.TestCase):
    def test_prefix_prepare_uses_exact_first_k_and_no_scoring(self):
        ctx = _FakeContext()
        runner = SimpleNamespace(
            model=SimpleNamespace(device=torch.device("cpu")))
        selector = CVPR25ChunkSelector(
            runner, ctx, static=None, budget=0.5, mode="prefix",
            sep_policy="sidecar",
        )

        # These are the three forbidden Prefix computations.  Raising sentinels
        # makes the test prove the path is not merely reporting zero counters.
        with mock.patch("torch.cuda.synchronize", return_value=None), \
             mock.patch("mmimpress.cvpr25.zscore",
                        side_effect=AssertionError("static score used")), \
             mock.patch("mmimpress.cvpr25.maxmin_diverse",
                        side_effect=AssertionError("diversity used")), \
             mock.patch.object(
                 CVPR25ChunkSelector, "_query_scores",
                 side_effect=AssertionError("query score used")):
            selector.prepare(text_emb=None)

        expected = list(range(budget_chunk_count(4, 0.5)))
        self.assertEqual(selector.selected_chunk_ids_per_layer,
                         [expected, expected])
        stats = selector.stats()
        self.assertEqual(stats["static_score_calls"], 0)
        self.assertEqual(stats["query_score_calls"], 0)
        self.assertEqual(stats["diversity_calls"], 0)
        self.assertEqual(stats["normal_chunk_count_total"], 4)
        BIAS.clear()

    def test_prefix_rejects_static_metadata_and_non_sidecar_policy(self):
        ctx = _FakeContext()
        runner = SimpleNamespace(
            model=SimpleNamespace(device=torch.device("cpu")))
        with self.assertRaises(AssertionError):
            CVPR25ChunkSelector(
                runner, ctx, static={"unused": True}, budget=0.25,
                mode="prefix", sep_policy="sidecar")
        with self.assertRaises(AssertionError):
            CVPR25ChunkSelector(
                runner, ctx, static=None, budget=0.25,
                mode="prefix", sep_policy="force")


class PrefixStoreValidationTests(unittest.TestCase):
    @staticmethod
    def _valid_context():
        ctx = object.__new__(ImageContext)
        ctx.meta = {
            "reordered": True,
            "order_is_per_layer": True,
            "num_layers": 2,
            "v_token_num": 4,
            "newline_idx": [3],
            # stored position 0 -> original separator 3 in layer 0;
            # stored position 1 -> original separator 3 in layer 1.
            "order": [[3, 0, 1, 2], [0, 3, 2, 1]],
        }
        ctx._separator_positions = [[0], [1]]
        return ctx

    def test_valid_per_layer_reordered_store_passes_once(self):
        ctx = self._valid_context()
        ctx.validate_reordered_prefix_store()
        self.assertTrue(ctx._reordered_prefix_store_validated)
        # Cached validation must remain a no-op on later requests.
        ctx.validate_reordered_prefix_store()

    def test_raster_shared_and_invalid_permutations_fail_closed(self):
        raster = self._valid_context()
        raster.meta["reordered"] = False
        with self.assertRaises(AssertionError):
            raster.validate_reordered_prefix_store()

        shared = self._valid_context()
        shared.meta["order_is_per_layer"] = False
        with self.assertRaises(AssertionError):
            shared.validate_reordered_prefix_store()

        duplicate = self._valid_context()
        duplicate.meta["order"][0] = [3, 0, 1, 1]
        with self.assertRaises(AssertionError):
            duplicate.validate_reordered_prefix_store()

        bad_separator = self._valid_context()
        bad_separator._separator_positions[0] = [1]
        with self.assertRaises(AssertionError):
            bad_separator.validate_reordered_prefix_store()


class EvalPackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "eval04_for_prefix_test", ROOT / "scripts/04_eval.py")
        cls.eval04 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.eval04)

    def test_real_pread_accounting_split_is_preserved(self):
        result = {
            "answer": "yes",
            "ttft": 0.010,
            "decode_latency": 0.020,
            "e2e_latency": 0.030,
            "generated_tokens": 2,
            "io": {
                "bytes": 130,
                "mb": 0.000130,
                "ms": 1.0,
                "preads": 5,
                "chunk_units": 2,
                "per_kind": {
                    "k": {"bytes": 40, "preads": 1, "units": 1},
                    "v": {"bytes": 50, "preads": 2, "units": 1},
                    "sep": {"bytes": 20, "preads": 1, "units": 0},
                    "probe": {"bytes": 20, "preads": 1, "units": 0},
                },
            },
        }
        packed = self.eval04._pack(result, {"answer": "yes"}, "gqa")
        self.assertEqual(packed["normal_kv_read_bytes"], 90)
        self.assertEqual(packed["separator_read_bytes"], 20)
        self.assertEqual(packed["normal_kv_preads"], 3)
        self.assertEqual(packed["separator_preads"], 1)
        # Probe bytes remain honestly included in the actual total.
        self.assertEqual(packed["total_actual_pread_bytes"], 130)


if __name__ == "__main__":
    unittest.main()

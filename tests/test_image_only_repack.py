"""CPU-only contracts for image-only VisionZip physical KV repacking.

The tests in this module deliberately use tiny synthetic prefix caches.  They
exercise the byte layout, mapping, separator sidecar, and Prefix I/O path
without loading a model, touching CUDA, or downloading any artifacts.
"""
from __future__ import annotations

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from mmimpress.cvpr25 import (budget_chunk_count, permutation_sha256,
                              visionzip_repack_order)
from mmimpress.reorder import mapping_from_perm, restore
from mmimpress.serve import BIAS, CVPR25ChunkSelector, ImageContext
from mmimpress.store import (ChunkReader, IOCounter, load_meta,
                             write_image_store,
                             write_separator_sidecar_from_store)


ROOT = Path(__file__).resolve().parents[1]


def _load_reorder_script():
    """Load the numeric-name script without running its CLI."""
    path = ROOT / "scripts/02_reorder.py"
    spec = importlib.util.spec_from_file_location(
        "reorder02_for_image_only_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _layers(num_layers=2, num_heads=2, prefix_len=10, head_dim=2):
    """Deterministic fp16 DynamicCache-shaped K/V tensors."""
    result = []
    size = num_heads * prefix_len * head_dim
    for layer in range(num_layers):
        base = torch.arange(size, dtype=torch.float16).view(
            1, num_heads, prefix_len, head_dim)
        k = base + layer * 1000
        v = base + 4000 + layer * 1000
        result.append((k, v))
    return result


def _visionzip_meta_flags():
    return {
        "physical_layout": "visionzip_image_only",
        "layout_method": "visionzip_image_only",
        "layout_source": "fresh_model_forward",
        "composed_from_store": False,
        "layout_uses_dataset_question": False,
        "llm_used_for_layout_scoring": False,
        "calibration_questions": 0,
        "global_order_all_layers": True,
        "separator_tail": True,
        "separator_policy": "stable_tail_plus_sidecar",
    }


def _write_tiny_store(path: Path, *, v_num=8, v_start=2,
                      newline_idx=(1, 6), chunk_size=2,
                      stored_to_original=None, separator_sidecar=True,
                      extra=None):
    layers = _layers(prefix_len=v_start + v_num)
    meta = write_image_store(
        path, layers, v_start, v_num,
        prefix_input_ids=list(range(v_start + v_num)),
        newline_idx=list(newline_idx), chunk_size=chunk_size,
        dtype="float16", probe_heads=1,
        stored_to_original=stored_to_original,
        separator_sidecar=separator_sidecar, extra=extra,
    )
    return layers, meta


class VisionZipOrderTests(unittest.TestCase):
    def test_stable_ties_separator_tail_bijection_and_inverse(self):
        scores = torch.tensor([
            0.2, float("inf"), 0.9, 0.9, 0.1, float("inf"), 0.2,
        ])

        # The deliberately unsorted separator input also checks that "stable
        # tail" means increasing original token order, not caller order.
        perm = visionzip_repack_order(scores, [5, 1])
        self.assertEqual(perm, [2, 3, 0, 6, 4, 1, 5])
        self.assertEqual(sorted(perm), list(range(scores.numel())))
        self.assertEqual(perm[-2:], [1, 5])

        inverse = mapping_from_perm(perm)
        self.assertEqual([perm[inverse[i]] for i in range(len(perm))],
                         list(range(len(perm))))
        original = torch.arange(len(perm))
        stored = original[torch.tensor(perm)]
        self.assertTrue(torch.equal(restore(stored, inverse), original))
        self.assertEqual(permutation_sha256(perm),
                         permutation_sha256(list(perm)))

    def test_invalid_real_score_and_separator_index_fail_closed(self):
        with self.assertRaises(ValueError):
            visionzip_repack_order([0.1, float("nan"), 0.2], [2])
        with self.assertRaises(ValueError):
            visionzip_repack_order([0.1, 0.2], [2])


class DirectAndPosthocLayoutTests(unittest.TestCase):
    def test_direct_write_matches_fresh_raster_then_posthoc_repack(self):
        scores = torch.tensor([
            0.4, float("inf"), 0.9, 0.9,
            0.2, 0.7, float("inf"), 0.1,
        ])
        newline = [1, 6]
        perm = visionzip_repack_order(scores, newline)
        self.assertEqual(perm, [2, 3, 5, 0, 4, 7, 1, 6])

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            direct = root / "direct"
            posthoc = root / "posthoc"
            layers, direct_meta = _write_tiny_store(
                direct, newline_idx=newline, stored_to_original=perm,
                separator_sidecar=True, extra=_visionzip_meta_flags())
            _, raster_meta = _write_tiny_store(
                posthoc, newline_idx=newline, stored_to_original=None,
                separator_sidecar=False)

            reorder02 = _load_reorder_script()
            perms = reorder02.target_to_current_perms(raster_meta, perm)
            shared = reorder02.commit(posthoc, raster_meta, perms,
                                      metadata_update=_visionzip_meta_flags())
            self.assertTrue(shared)
            write_separator_sidecar_from_store(posthoc)
            posthoc_meta = load_meta(posthoc)

            for layer in range(direct_meta["num_layers"]):
                for filename in ("k.bin", "v.bin", "probe_k.bin"):
                    a = direct / f"layer_{layer:02d}" / filename
                    b = posthoc / f"layer_{layer:02d}" / filename
                    self.assertEqual(a.read_bytes(), b.read_bytes(),
                                     f"physical mismatch: L{layer}/{filename}")
            self.assertEqual((direct / "sep_kv.bin").read_bytes(),
                             (posthoc / "sep_kv.bin").read_bytes())

            for key in ("order", "order_is_per_layer", "newline_stored",
                        "reordered", "bytes_separator_sidecar"):
                self.assertEqual(direct_meta[key], posthoc_meta[key], key)
            self.assertEqual(direct_meta["order"], perm)
            self.assertFalse(direct_meta["order_is_per_layer"])
            self.assertEqual(direct_meta["newline_stored"], [6, 7])

            inverse = mapping_from_perm(perm)
            self.assertEqual([perm[inverse[i]] for i in range(len(perm))],
                             list(range(len(perm))))

            # Independently reconstruct the expected sidecar from the original
            # cache.  This proves equality is not merely two paths sharing the
            # same mistake.
            expected_k, expected_v = [], []
            physical_sep = torch.tensor([6, 7])
            perm_t = torch.tensor(perm)
            for k, v in layers:
                for tensor, output in ((k, expected_k), (v, expected_v)):
                    block = tensor[0, :, 2:10].permute(1, 0, 2).contiguous()
                    output.append(block[perm_t][physical_sep])
            expected = torch.stack([
                torch.stack(expected_k), torch.stack(expected_v)
            ]).numpy().tobytes()
            self.assertEqual((direct / "sep_kv.bin").read_bytes(), expected)


class PrefixLayoutValidatorTests(unittest.TestCase):
    @staticmethod
    def _context(path: Path, meta: dict, separator_positions=None):
        ctx = object.__new__(ImageContext)
        ctx.dir = path
        ctx.meta = copy.deepcopy(meta)
        positions = (separator_positions if separator_positions is not None
                     else ctx.meta.get("newline_stored",
                                       ctx.meta["newline_idx"]))
        if positions and isinstance(positions[0], list):
            ctx._separator_positions = [sorted(map(int, row))
                                        for row in positions]
        else:
            row = sorted(map(int, positions))
            ctx._separator_positions = [list(row)
                                        for _ in range(ctx.meta["num_layers"])]
        return ctx

    @staticmethod
    def _valid_meta():
        meta = {
            "num_layers": 2,
            "v_token_num": 7,
            "newline_idx": [1, 5],
            "order": [2, 3, 0, 4, 6, 1, 5],
            "order_is_per_layer": False,
            "newline_stored": [5, 6],
            "reordered": True,
            **_visionzip_meta_flags(),
        }
        return meta

    def test_accepts_valid_shared_image_only_layout_and_raster_layout(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            (path / "visionzip_layout.pt").write_bytes(b"layout")
            ctx = self._context(path, self._valid_meta())
            ctx.validate_prefix_layout("visionzip_image_only")
            self.assertEqual(ctx._prefix_layout_validated,
                             "visionzip_image_only")
            self.assertTrue(ctx._reordered_prefix_store_validated)

            raster = {
                "physical_layout": "raster",
                "num_layers": 2,
                "v_token_num": 7,
                "newline_idx": [1, 5],
                "layout_uses_dataset_question": False,
                "calibration_questions": 0,
            }
            raster_ctx = self._context(path, raster)
            raster_ctx.validate_prefix_layout("raster")
            self.assertEqual(raster_ctx._prefix_layout_validated, "raster")

    def test_rejects_wrong_provenance_mapping_and_missing_artifact(self):
        cases = {
            "wrong expected layout": lambda m: m.update(
                physical_layout="morton", layout_method="morton"),
            "per-layer image-only order": lambda m: m.update(
                order_is_per_layer=True),
            "not global": lambda m: m.update(global_order_all_layers=False),
            "dataset question": lambda m: m.update(
                layout_uses_dataset_question=True),
            "LLM scoring": lambda m: m.update(
                llm_used_for_layout_scoring=True),
            "calibration": lambda m: m.update(calibration_questions=1),
            "not separator tail": lambda m: m.update(separator_tail=False),
            "duplicate permutation": lambda m: m["order"].__setitem__(0, 3),
            "wrong newline mapping": lambda m: m.update(
                newline_stored=[0, 6]),
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            artifact = root / "visionzip_layout.pt"
            artifact.write_bytes(b"layout")
            for label, mutate in cases.items():
                with self.subTest(label=label):
                    meta = self._valid_meta()
                    mutate(meta)
                    ctx = self._context(root, meta)
                    with self.assertRaises(AssertionError):
                        ctx.validate_prefix_layout("visionzip_image_only")

            artifact.unlink()
            ctx = self._context(root, self._valid_meta())
            with self.assertRaises(AssertionError):
                ctx.validate_prefix_layout("visionzip_image_only")


class RealPrefixIOTests(unittest.TestCase):
    def test_contiguous_first_k_has_exact_actual_bytes_and_preads(self):
        # 31 tokens / 4 = 8 chunks, so Prefix25 buys two touching chunks.
        v_num, v_start, chunk_size = 31, 2, 4
        num_layers, num_heads, head_dim, n_sep = 2, 2, 2, 2
        scores = torch.arange(v_num, dtype=torch.float32)
        newline = [5, 29]
        scores[newline] = float("inf")
        perm = visionzip_repack_order(scores, newline)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            layers = _layers(num_layers=num_layers, num_heads=num_heads,
                             prefix_len=v_start + v_num,
                             head_dim=head_dim)
            meta = write_image_store(
                path, layers, v_start, v_num,
                prefix_input_ids=list(range(v_start + v_num)),
                newline_idx=newline, chunk_size=chunk_size, dtype="float16",
                probe_heads=1, stored_to_original=perm,
                separator_sidecar=True, extra=_visionzip_meta_flags())

            n_chunks = meta["n_chunks_per_layer"]
            k = budget_chunk_count(n_chunks, 0.25)
            self.assertEqual((n_chunks, k), (8, 2))
            cids = list(range(k))
            counter = IOCounter()
            reader = ChunkReader(path, meta, drop_cache=False)
            try:
                for layer in range(num_layers):
                    for kind in ("k", "v"):
                        rows, values = reader.read_chunks(
                            layer, kind, cids, counter)
                        self.assertEqual(rows.tolist(), list(range(8)))
                        self.assertEqual(tuple(values.shape),
                                         (8, num_heads, head_dim))
            finally:
                reader.close()

            sep_ctx = object.__new__(ImageContext)
            sep_ctx.dir = path
            sep_ctx.sep_kv_shape = (
                2, num_layers, n_sep, num_heads, head_dim)
            sep = sep_ctx.read_sep_kv(counter)
            self.assertEqual(tuple(sep.shape), sep_ctx.sep_kv_shape)

            itemsize = 2
            normal_bytes = (2 * num_layers * (k * chunk_size)
                            * num_heads * head_dim * itemsize)
            separator_bytes = (2 * num_layers * n_sep
                               * num_heads * head_dim * itemsize)
            summary = counter.summary()
            self.assertEqual(summary["per_kind"]["k"]["preads"],
                             num_layers)
            self.assertEqual(summary["per_kind"]["v"]["preads"],
                             num_layers)
            self.assertEqual(summary["per_kind"]["sep"]["preads"], 1)
            self.assertEqual(summary["per_kind"]["k"]["bytes"]
                             + summary["per_kind"]["v"]["bytes"],
                             normal_bytes)
            self.assertEqual(summary["per_kind"]["sep"]["bytes"],
                             separator_bytes)
            self.assertEqual(summary["preads"], 2 * num_layers + 1)
            self.assertEqual(summary["bytes"], normal_bytes + separator_bytes)
            self.assertEqual(summary["chunk_units"],
                             2 * num_layers * k)
            self.assertAlmostEqual(
                summary["bytes"] / summary["preads"],
                (normal_bytes + separator_bytes) / (2 * num_layers + 1))
            self.assertNotIn("probe", summary["per_kind"])


class _FakeReader:
    def __init__(self, meta):
        self.meta = meta

    def read_chunks(self, layer, kind, cids, counter):
        cs, vn = self.meta["chunk_size"], self.meta["v_token_num"]
        rows = []
        for cid in cids:
            rows.extend(range(cid * cs, min((cid + 1) * cs, vn)))
        values = torch.zeros(len(rows), self.meta["num_heads"],
                             self.meta["head_dim"], dtype=torch.float16)
        counter.record(kind, values.numel() * values.element_size(), 0.0,
                       preads=1, units=len(cids))
        return torch.tensor(rows, dtype=torch.long), values


class _FakeCache:
    def write(self, layer, kind, rows, values):
        return None


class _FakeContext:
    def __init__(self):
        self.meta = {
            "num_layers": 2,
            "n_chunks_per_layer": 4,
            "chunk_size": 4,
            "v_token_num": 16,
            "v_token_start": 1,
            "prefix_len": 17,
            "num_heads": 1,
            "head_dim": 1,
        }
        self.reader = _FakeReader(self.meta)
        self.cache = _FakeCache()

    def read_sep_kv(self, counter):
        output = torch.zeros(2, 2, 2, 1, 1, dtype=torch.float16)
        counter.record("sep", output.numel() * output.element_size(), 0.0,
                       preads=1, units=0)
        return output

    def separator_positions(self, layer):
        return [14, 15]


class OnlinePrefixIsolationTests(unittest.TestCase):
    def test_prefix_online_path_cannot_call_any_forbidden_scorer(self):
        ctx = _FakeContext()
        runner = SimpleNamespace(
            model=SimpleNamespace(device=torch.device("cpu")))
        selector = CVPR25ChunkSelector(
            runner, ctx, static=None, budget=0.25, mode="prefix",
            sep_policy="sidecar", image_id="synthetic")

        forbidden = AssertionError("forbidden online scoring call")
        with mock.patch("torch.cuda.synchronize", return_value=None), \
             mock.patch("mmimpress.cvpr25.clip_cls_patch_saliency",
                        side_effect=forbidden), \
             mock.patch("mmimpress.cvpr25.anyres_token_scores",
                        side_effect=forbidden), \
             mock.patch("mmimpress.cvpr25.zscore",
                        side_effect=forbidden), \
             mock.patch("mmimpress.cvpr25.maxmin_diverse",
                        side_effect=forbidden), \
             mock.patch("mmimpress.serve.calibrate_image",
                        side_effect=forbidden), \
             mock.patch("mmimpress.serve.Server.raters",
                        side_effect=forbidden), \
             mock.patch("mmimpress.sparsevlm.select_raters",
                        side_effect=forbidden), \
             mock.patch.object(CVPR25ChunkSelector, "_query_scores",
                               side_effect=forbidden):
            selector.prepare(text_emb=None)

        expected = list(range(budget_chunk_count(4, 0.25)))
        self.assertEqual(selector.selected_chunk_ids_per_layer,
                         [expected, expected])
        stats = selector.stats()
        self.assertEqual(stats["static_score_calls"], 0)
        self.assertEqual(stats["query_score_calls"], 0)
        self.assertEqual(stats["diversity_calls"], 0)
        self.assertEqual(stats["normal_chunk_count_total"], 2)
        self.assertEqual(selector.io.summary()["preads"], 5)
        self.assertNotIn("probe", selector.io.summary()["per_kind"])
        BIAS.clear()


if __name__ == "__main__":
    unittest.main()

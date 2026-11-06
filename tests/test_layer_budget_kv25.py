"""CPU contracts for the layer-wise PrefixStore-KV25 budget (layerwise_v1).

Gate G5 of docs/layer_budget_kv25_contract.md: budget arithmetic at the fixed
geometries, clip boundaries, largest-remainder exactness, fail-closed vectors,
and the real writer/reader/selector path on actual FP16 files.
"""
from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mmimpress.layer_budget import (allocate_power, allocate_water_filling,
                                    budget_document, budget_sha256,
                                    clip_redistribute, front_schedule,
                                    keep_bounds, layer_keep_counts,
                                    load_budget_document, total_row_budget,
                                    uniform_ratios, validate_keep_counts,
                                    validate_ratios)
from mmimpress.prefix25 import (ReadTrace, seal_integrity,
                                validate_llava_prefix_layerwise_meta,
                                verify_integrity)
from mmimpress.serve import BIAS, CVPR25ChunkSelector, ImageContext
from mmimpress.store import (ChunkReader, payload_rows_for_layer,
                             write_image_store)
from test_llava_kv25 import _fixture as lfixture

CASES = (1, 63, 64, 65, 127, 128, 129, 256, 349, 2200)
L = 32


def _hamilton_reference(ratios, n, L):
    """Independent apportionment (no shared code with the library)."""
    budget = L * (-(-n // 4))
    from fractions import Fraction
    cap = min(n, math.ceil(Fraction(3, 5) * n))               # ceil(0.6 n), exact
    q = [r * n for r in ratios]
    k = [min(cap, int(math.floor(x))) for x in q]
    frac = [x - math.floor(x) for x in q]
    order = sorted(range(L), key=lambda i: (-frac[i], i))
    need = budget - sum(k)
    while need:
        for i in order:
            if need and k[i] < cap:
                k[i] += 1
                need -= 1
    return k


class BudgetArithmetic(unittest.TestCase):
    def test_uniform_reproduces_ceil_at_all_geometries(self):
        for n in CASES + (2, 3, 5, 6, 7, 10, 1344, 2880):
            with self.subTest(n=n):
                k = layer_keep_counts(uniform_ratios(), n)
                self.assertEqual(k, [(n + 3) // 4] * L)
                self.assertEqual(sum(k), total_row_budget(n))

    def test_largest_remainder_exact_sum_and_bounds(self):
        vectors = {
            "front": front_schedule(), "back": front_schedule(reverse=True),
            "power": allocate_power([(i % 7) + 0.5 for i in range(L)], 1.0),
            "edge": [0.6] * 8 + [0.1] * 8 + [0.15] * 16,
        }
        self.assertAlmostEqual(sum(vectors["edge"]), 8.0, places=12)
        for name, r in vectors.items():
            validate_ratios(r)
            for n in CASES:
                with self.subTest(vector=name, n=n):
                    k = layer_keep_counts(r, n)
                    self.assertEqual(k, _hamilton_reference(r, n, L))
                    self.assertEqual(sum(k), L * math.ceil(n / 4))
                    lo, hi = keep_bounds(n)
                    self.assertTrue(all(lo <= x <= hi for x in k), (name, n, k))
                    self.assertTrue(all(0 <= x <= n for x in k))
                    validate_keep_counts(k, r, n)
                    # the uniform total is preserved exactly: byte budget invariant
                    self.assertEqual(sum(k), sum(layer_keep_counts(uniform_ratios(), n)))

    def test_fail_closed_vectors(self):
        good = front_schedule()
        bad = [
            good[:-1], good + [0.0], [float("nan")] * L, [float("inf")] + good[1:],
            [-0.1] + [8.1 / 31] * 31, [0.7] + [(8.0 - 0.7) / 31] * 31,
            [0.05] + [(8.0 - 0.05) / 31] * 31, [0.26] * L, [0.25] * 31 + [0.2],
            "0.25" * L, None, [True] * L,
        ]
        for vector in bad:
            with self.subTest(vector=str(vector)[:40]), self.assertRaises((ValueError, TypeError)):
                layer_keep_counts(vector, 349)
        for n in (0, -1, 1.5, True, "349"):
            with self.subTest(n=n), self.assertRaises((ValueError, TypeError)):
                layer_keep_counts(good, n)
        k = layer_keep_counts(good, 349)
        wrong = list(k)
        wrong[0] += 1
        wrong[1] -= 1
        with self.assertRaises(ValueError):
            validate_keep_counts(wrong, good, 349)
        with self.assertRaises(ValueError):
            validate_keep_counts(k[:-1], good, 349)
        with self.assertRaises(ValueError):
            validate_keep_counts([float(x) for x in k], good, 349)

    def test_clip_and_redistribute(self):
        raw = [100.0] + [1.0] * 31
        r = clip_redistribute(raw)
        self.assertAlmostEqual(r[0], 0.6, places=12)
        self.assertTrue(all(abs(x - r[1]) < 1e-12 for x in r[1:]))
        self.assertAlmostEqual(sum(r), 8.0, places=9)
        raw = [0.0] * 16 + [1.0] * 16
        r = clip_redistribute(raw)
        self.assertTrue(all(abs(x - 0.1) < 1e-12 for x in r[:16]))
        self.assertTrue(all(abs(x - 0.4) < 1e-12 for x in r[16:]))
        with self.assertRaises(ValueError):
            clip_redistribute([0.0] * L)
        with self.assertRaises(ValueError):
            clip_redistribute([1.0] * 31)
        with self.assertRaises(ValueError):
            allocate_power([0.0] * L, 1.0)
        with self.assertRaises(ValueError):
            allocate_power([1.0] * L, 0.0)
        r = allocate_power([0.0] * 31 + [5.0], 2.0)
        self.assertAlmostEqual(r[-1], 0.6, places=12)
        self.assertAlmostEqual(sum(r), 8.0, places=9)

    def test_water_filling(self):
        fr = [i / 20 for i in range(21)]
        curves = [[min(1.0, f * (1 + li / 8)) for f in fr] for li in range(L)]
        r = allocate_water_filling(fr, curves, 64 / 2112)
        validate_ratios(r)
        self.assertAlmostEqual(sum(r), 8.0, places=9)
        self.assertGreaterEqual(r[-1], r[0])
        with self.assertRaises(ValueError):
            allocate_water_filling(fr[1:], [row[1:] for row in curves], 0.03)
        with self.assertRaises(ValueError):
            allocate_water_filling(fr, curves[:-1], 0.03)
        with self.assertRaises(ValueError):
            allocate_water_filling(fr, curves, 0.0)

    def test_document_roundtrip_and_sha(self):
        r = front_schedule()
        doc = budget_document("F-front", r, model="llava", source={"kind": "fixed"})
        self.assertEqual(doc["layer_budget_sha256"], budget_sha256(r))
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "v.json"
            p.write_text(json.dumps(doc))
            loaded = load_budget_document(p)
            self.assertEqual(loaded["layer_keep_ratios"], r)
            tampered = dict(doc)
            tampered["layer_keep_ratios"] = list(r)
            tampered["layer_keep_ratios"][0] += 1e-6
            tampered["layer_keep_ratios"][1] -= 1e-6
            p.write_text(json.dumps(tampered))
            with self.assertRaises(ValueError):
                load_budget_document(p)


def _layerwise_store(root, n, ratios, alloc_id="T"):
    meta_a, rank, layers = lfixture(root / "a", n)
    extra = {key: meta_a[key] for key in [
        "physical_layout", "layout_method", "layout_source", "visual_kv_source",
        "layout_uses_dataset_question", "llm_used_for_layout_scoring",
        "calibration_questions", "global_order_all_layers", "separator_tail",
        "separator_policy", "permutation_sha256", "inverse_permutation_sha256"]}
    doc = budget_document(alloc_id, ratios, model="fixture", source={},
                          num_layers=2)
    common = dict(extra=extra, stored_to_original=meta_a["order"],
                  separator_sidecar=True, probe_heads=0)
    b = write_image_store(root / "b", layers, meta_a["v_token_start"],
                          meta_a["v_token_num"], meta_a["prefix_input_ids"],
                          meta_a["newline_idx"], storage_policy="prefix25",
                          **common)
    c = write_image_store(root / "c", layers, meta_a["v_token_start"],
                          meta_a["v_token_num"], meta_a["prefix_input_ids"],
                          meta_a["newline_idx"], storage_policy="prefix25_layerwise",
                          layer_budget=doc, **common)
    for name in ("b", "c"):
        (root / name / "visionzip_layout.pt").write_bytes(
            (root / "a" / "visionzip_layout.pt").read_bytes())
        seal_integrity(root / name)
    return meta_a, rank, layers, b, c, doc


class RealLayerwisePath(unittest.TestCase):
    def tearDown(self):
        BIAS.clear()

    def test_uniform_vector_is_byte_identical_to_prefix25(self):
        for n in CASES:
            with self.subTest(n=n), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                _, _, _, b, c, _ = _layerwise_store(root, n, [0.25, 0.25], "U")
                for rel in ["sep_kv.bin", "sys_kv.pt", "visionzip_layout.pt",
                            "layer_00/k.bin", "layer_00/v.bin",
                            "layer_01/k.bin", "layer_01/v.bin"]:
                    self.assertEqual((root / "b" / rel).read_bytes(),
                                     (root / "c" / rel).read_bytes(), rel)
                self.assertEqual(c["payload_rows_per_layer"], [b["payload_rows"]] * 2)
                shared = set(b) & set(c) - {"format", "storage_policy",
                                            "order_semantics", "rounding_policy",
                                            "stored_row_to_original"}
                for key in shared:
                    self.assertEqual(b[key], c[key], key)
                self.assertEqual(c["stored_row_to_original"], b["stored_row_to_original"])

    def test_layerwise_rows_masks_bits_reads_and_invariance(self):
        ratios = [0.35, 0.15]
        for n in CASES:
            with self.subTest(n=n), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                meta_a, rank, layers, b, c, doc = _layerwise_store(root, n, ratios)
                k = layer_keep_counts(ratios, n, num_layers=2)
                self.assertEqual(c["layer_keep_counts"], k)
                self.assertEqual(sum(k), 2 * ((n + 3) // 4))
                validate_llava_prefix_layerwise_meta(c)
                row_bytes = 2 * 2 * 2
                for li in range(2):
                    for kind in ("k", "v"):
                        blob = (root / "c" / f"layer_{li:02d}/{kind}.bin").read_bytes()
                        self.assertEqual(len(blob), k[li] * row_bytes)
                        self.assertEqual(blob, (root / "a" / f"layer_{li:02d}/{kind}.bin").read_bytes()[:k[li] * row_bytes])
                    self.assertEqual(payload_rows_for_layer(c, li), k[li])
                ctx = ImageContext(root / "c", "cpu", drop_cache=False, require_v_hidden=False)
                try:
                    ctx.validate_visual_kv_layout()
                    runner = SimpleNamespace(model=SimpleNamespace(device=torch.device("cpu")))
                    selections = []
                    for text in (None, torch.ones(9, 4)):
                        cache = ctx.cache.new_request()
                        sel = CVPR25ChunkSelector(runner, ctx, None, .25, "prefix",
                                                  sep_policy="sidecar",
                                                  budget_unit="visual_kv")
                        with patch("torch.cuda.synchronize"), ReadTrace(root / "c") as trace:
                            sel.prepare(text)
                        summary = trace.summary()
                        self.assertEqual(sel.io.bytes, summary["bytes"])
                        self.assertEqual(sel.io.preads, summary["preads"])
                        self.assertEqual(summary["bytes"], sum(2 * kk * row_bytes for kk in k)
                                         + (root / "c/sep_kv.bin").stat().st_size)
                        self.assertEqual(summary["preads"], 2 * 2 + 1)
                        for call in summary["calls"]:
                            self.assertEqual(call["offset"], 0)
                            self.assertEqual(call["requested"], call["returned"])
                        st = sel.stats()
                        self.assertEqual(st["k_per_layer"], k)
                        self.assertEqual(st["attended_content_kv_count_per_layer"], k)
                        self.assertEqual(st["keep_count_per_layer"], [kk + 2 for kk in k])
                        self.assertEqual(st["k_uniform_equivalent"], (n + 3) // 4)
                        self.assertEqual(st["unused_loaded_real_rows"], 0)
                        self.assertEqual(st["budget_policy"], "layerwise_v1")
                        self.assertEqual(st["layer_budget_sha256"], doc["layer_budget_sha256"])
                        self.assertEqual(st["selected_original_ids"], rank[:max(k)])
                        self.assertEqual(st["query_score_calls"], 0)
                        self.assertEqual(st["static_score_calls"], 0)
                        for li, (src_k, src_v) in enumerate(layers):
                            expected = [True] * 3 + [True] * k[li] + [False] * (n - k[li]) + [True] * 2
                            self.assertEqual((BIAS[li].flatten() == 0).tolist(), expected)
                            for kind, src in (("keys", src_k), ("values", src_v)):
                                actual = getattr(cache.layers[li], kind)
                                self.assertTrue(torch.equal(
                                    actual[:, :, 3:3 + k[li]],
                                    src[:, :, torch.tensor(rank[:k[li]]) + 3].to(torch.bfloat16)))
                                if k[li] < n:
                                    self.assertTrue(torch.equal(
                                        actual[:, :, 3 + k[li]:3 + n],
                                        torch.zeros_like(actual[:, :, 3 + k[li]:3 + n])))
                        selections.append((st["selected_original_ids"], st["k_per_layer"],
                                           [BIAS[li].clone() for li in range(2)]))
                    self.assertEqual(selections[0][:2], selections[1][:2])
                    for x, y in zip(selections[0][2], selections[1][2]):
                        self.assertTrue(torch.equal(x, y))
                    for ratio, unit in [(1, "visual_kv"), (.26, "visual_kv"), (.2, "visual_kv"), (.25, "chunk")]:
                        with self.assertRaises(ValueError):
                            CVPR25ChunkSelector(runner, ctx, None, ratio, "prefix",
                                                sep_policy="sidecar", budget_unit=unit)
                    with self.assertRaises(ValueError):
                        ctx.reader.read_full(0, "k")
                finally:
                    ctx.close()
                    BIAS.clear()

    def test_fail_closed_store_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, _, _, _, c, doc = _layerwise_store(root, 349, [0.35, 0.15])
            k = c["layer_keep_counts"]
            for field, value in [("layer_keep_counts", [k[0] + 1, k[1] - 1]),
                                 ("layer_keep_ratios", [0.15, 0.35]),
                                 ("layer_budget_sha256", "0" * 64),
                                 ("payload_rows_per_layer", [k[0], k[0]]),
                                 ("stored_content_count_total", sum(k) + 1),
                                 ("budget_policy", "uniform"),
                                 ("layer_budget_r_max", 0.3),
                                 ("payload_rows", k[0])]:
                bad = copy.deepcopy(c)
                bad[field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    validate_llava_prefix_layerwise_meta(bad)
            bad = copy.deepcopy(c)
            bad["layer_keep_ratios"] = [0.15, 0.35]
            bad["layer_keep_counts"] = layer_keep_counts([0.15, 0.35], 349, num_layers=2)
            bad["layer_budget_sha256"] = budget_sha256([0.15, 0.35])
            bad["payload_rows_per_layer"] = bad["layer_keep_counts"]
            bad["stored_row_to_original"] = c["order"][:max(bad["layer_keep_counts"])]
            bad["payload_chunks_per_layer"] = [(x + 63) // 64 for x in bad["layer_keep_counts"]]
            bad["valid_rows_last_chunk_per_layer"] = [(x - 1) % 64 + 1 for x in bad["layer_keep_counts"]]
            bad["stored_content_count_per_layer"] = bad["layer_keep_counts"]
            validate_llava_prefix_layerwise_meta(bad)   # self-consistent metadata...
            (root / "c/meta.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):          # ...but the sealed files disagree
                verify_integrity(root / "c")
            (root / "c/meta.json").write_text(json.dumps(c))
            f = root / "c/layer_00/k.bin"
            blob = f.read_bytes()
            f.write_bytes(blob[:-2])
            with self.assertRaises(ValueError):
                verify_integrity(root / "c")
            ctx = ImageContext.__new__(ImageContext)
            ctx.meta = c
            ctx.dir = root / "c"
            ctx._separator_positions = [sorted(c["newline_stored"])] * 2
            with self.assertRaises(AssertionError):
                ctx.validate_visual_kv_layout()
            f.write_bytes(blob)
            with ChunkReader(root / "c", c, False) as reader:
                with self.assertRaises(ValueError):
                    reader.read_chunks(1, "k", [(k[1] + 63) // 64])
            meta_a, _, layers = lfixture(root / "e", 65)
            args = (layers, meta_a["v_token_start"], meta_a["v_token_num"],
                    meta_a["prefix_input_ids"], meta_a["newline_idx"])
            common = dict(stored_to_original=meta_a["order"],
                          separator_sidecar=True, probe_heads=0)
            with self.assertRaises(ValueError):      # policy without a vector
                write_image_store(root / "d1", *args,
                                  storage_policy="prefix25_layerwise", **common)
            with self.assertRaises(ValueError):      # vector without the policy
                write_image_store(root / "d2", *args, storage_policy="prefix25",
                                  layer_budget=doc, **common)
            wrong = dict(doc)
            wrong["layer_budget_sha256"] = "0" * 64
            with self.assertRaises(ValueError):      # vector/SHA disagreement
                write_image_store(root / "d3", *args,
                                  storage_policy="prefix25_layerwise",
                                  layer_budget=wrong, **common)
            wrong = dict(doc, layer_keep_ratios=[0.5, 0.5],
                         layer_budget_sha256=budget_sha256([0.5, 0.5]))
            with self.assertRaises(ValueError):      # sum != L * 0.25
                write_image_store(root / "d4", *args,
                                  storage_policy="prefix25_layerwise",
                                  layer_budget=wrong, **common)


if __name__ == "__main__":
    unittest.main()

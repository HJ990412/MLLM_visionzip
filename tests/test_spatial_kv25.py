"""CPU contracts for quota-matched LLaVA-NeXT spatial original-token KV."""
from __future__ import annotations

import copy
import math
import unittest
from types import SimpleNamespace

import torch
from transformers.models.llava_next.modeling_llava_next import (
    LlavaNextModel, get_anyres_image_grid_shape)

from mmimpress.contextual_kv25 import (
    build_selection_plan, split_visual_kv_budget)
from mmimpress.spatial_kv25 import (
    LAYOUT_POLICY, anyres_token_coordinates, build_spatial_selection_plan,
    make_spatial_regions, select_spatial_branch)


def _runner(side: int, pinpoint: tuple[int, int]) -> SimpleNamespace:
    cfg = SimpleNamespace(
        vision_config=SimpleNamespace(image_size=side, patch_size=1),
        image_grid_pinpoints=[list(pinpoint)])
    return SimpleNamespace(cfg=cfg)


def _actual_pack(runner: SimpleNamespace, size: tuple[int, int]
                 ) -> torch.Tensor:
    """Call the installed HF packer with source-identifying patch values."""
    side = runner.cfg.vision_config.image_size
    tiles_h, tiles_w = get_anyres_image_grid_shape(
        size, runner.cfg.image_grid_pinpoints, side)
    count = 1 + tiles_h * tiles_w
    feature = torch.arange(count * side * side, dtype=torch.float32)
    feature = feature.view(count, side * side, 1)
    dummy = SimpleNamespace(config=runner.cfg)
    packed, lens = LlavaNextModel.pack_image_features(
        dummy, [feature], torch.tensor([size]), "default",
        image_newline=torch.tensor([-1.]))
    assert lens.tolist() == [len(packed[0])]
    return packed[0].flatten()


def _simple_records(height: int, width: int,
                    ids: list[int] | None = None
                    ) -> list[dict | None]:
    if ids is None:
        ids = list(range(height * width))
    records: list[dict | None] = [None] * (max(ids) + 1 if ids else 0)
    for i, original in enumerate(ids):
        row, col = divmod(i, width)
        records[original] = {
            "original_visual_id": original, "branch": "high",
            "row": row, "column": col,
            "grid_height": height, "grid_width": width,
            "x": (col + .5) / width, "y": (row + .5) / height,
        }
    return records


class AnyResCoordinatesTests(unittest.TestCase):
    def test_matches_actual_hf_packer_patch_by_patch(self):
        # Distinct tile grids and square/wide/tall unpadding. This checks
        # source patch identity, not merely the final array's shape.
        cases = [
            ((4, 4), (2, 2), (4, 4)),
            ((4, 4), (1, 2), (4, 4)),
            ((4, 4), (2, 1), (4, 4)),
            ((4, 8), (1, 2), (4, 8)),
            ((8, 4), (2, 1), (8, 4)),
        ]
        for pin, size, expected_tile_px in cases:
            with self.subTest(pin=pin, size=size):
                runner = _runner(4, pin)
                packed = _actual_pack(runner, size)
                coords = anyres_token_coordinates(runner, size, len(packed))
                self.assertEqual(coords["tile_grid"], {
                    "height": expected_tile_px[0] // 4,
                    "width": expected_tile_px[1] // 4})
                records = coords["records"]
                self.assertEqual(len(records), len(packed))
                for original, value in enumerate(packed.tolist()):
                    record = records[original]
                    if value == -1.:
                        self.assertIsNone(record)
                        self.assertIn(original,
                                      coords["structural_original_ids"])
                        continue
                    self.assertIsNotNone(record)
                    self.assertEqual(record["original_visual_id"], original)
                    source = (record["source_subimage_index"] * 16
                              + record["source_patch_row"] * 4
                              + record["source_patch_column"])
                    self.assertEqual(source, int(value))
                    if record["branch"] == "base":
                        self.assertEqual(record["source_subimage_index"], 0)
                        self.assertLess(original, 16)
                    else:
                        self.assertGreaterEqual(record["source_subimage_index"],
                                                1)
                        self.assertGreaterEqual(original, 16)
                    self.assertEqual(record["x"],
                                     (record["column"] + .5)
                                     / record["grid_width"])
                    self.assertEqual(record["y"],
                                     (record["row"] + .5)
                                     / record["grid_height"])
                self.assertEqual(coords["structural_original_ids"],
                                 [i for i, v in enumerate(packed.tolist())
                                  if v == -1.])

    def test_span_mismatch_fails_closed(self):
        runner = _runner(4, (8, 4))
        packed = _actual_pack(runner, (2, 1))
        with self.assertRaises(ValueError):
            anyres_token_coordinates(runner, (2, 1), len(packed) - 1)


class SpatialRegionsTests(unittest.TestCase):
    def test_equal_area_exact_q_row_rule_and_bounds(self):
        for height, width, q in ((6, 6, 5), (2, 8, 7), (8, 2, 7),
                                 (4, 4, 1), (4, 4, 16)):
            with self.subTest(height=height, width=width, q=q):
                regions = make_spatial_regions(height, width, q)
                self.assertEqual(len(regions), q)
                r_min = (q + width - 1) // width
                r_max = min(q, height)
                r_ideal = math.floor(math.sqrt(q * height / width) + .5)
                rows = min(max(r_ideal, r_min), r_max)
                self.assertEqual(regions[0]["rows"], rows)
                self.assertEqual(sorted({r["row_bin"] for r in regions}),
                                 list(range(rows)))
                for region in regions:
                    self.assertAlmostEqual(
                        (region["x_max"] - region["x_min"])
                        * (region["y_max"] - region["y_min"]), 1 / q)
                    self.assertAlmostEqual(region["normalized_area"], 1 / q)
                    self.assertLess(region["x_min"], region["x_center"])
                    self.assertLess(region["x_center"], region["x_max"])
                    self.assertLess(region["y_min"], region["y_center"])
                    self.assertLess(region["y_center"], region["y_max"])
        self.assertEqual(make_spatial_regions(4, 4, 0), [])
        with self.assertRaises(ValueError):
            make_spatial_regions(2, 2, 5)

    def test_half_open_boundary_goes_to_following_region(self):
        # The middle patch center is x=.5, exactly the x boundary.
        records = _simple_records(1, 3)
        result = select_spatial_branch(records, [0, 1, 2], 2, 1, 3)
        self.assertEqual([r["original_candidate_count"]
                          for r in result["regions"]], [1, 2])
        self.assertEqual(result["empty_region_count"], 0)

    def test_empty_fallback_q_edges_and_id_ties(self):
        records = _simple_records(4, 4)
        self.assertEqual(select_spatial_branch(records, [0, 1, 2, 3],
                                               0, 4, 4)["selected_original_ids"],
                         [])
        one = select_spatial_branch(records, [0, 1, 2, 3], 1, 4, 4)
        self.assertEqual(one["selected_original_ids"], [1])
        all_four = select_spatial_branch(records, [0, 1, 2, 3], 4, 4, 4)
        self.assertEqual(all_four["selected_original_ids"], [0, 2, 1, 3])
        self.assertEqual(all_four["empty_region_count"], 2)
        self.assertEqual(all_four["fallback_count"], 2)
        self.assertEqual([r["fallback"] for r in all_four["regions"]],
                         [False, False, True, True])
        self.assertEqual(set(all_four["selected_original_ids"]),
                         {0, 1, 2, 3})
        with self.assertRaises(ValueError):
            select_spatial_branch(records, [0, 1, 2, 3], 5, 4, 4)

        tied = _simple_records(1, 2, [9, 3])
        tied_result = select_spatial_branch(tied, [3, 9], 1, 1, 2)
        self.assertEqual(tied_result["selected_original_ids"], [3])
        self.assertEqual(tied_result["regions"][0]["distance_to_center"], .5)


class SpatialPlanTests(unittest.TestCase):
    def _plan(self, side=4, size=(4, 4), pin=(8, 8), alpha=.2):
        runner = _runner(side, pin)
        packed = _actual_pack(runner, size)
        coords = anyres_token_coordinates(runner, size, len(packed))
        scores = torch.arange(len(packed), dtype=torch.float32)
        scores[coords["structural_original_ids"]] = float("inf")
        index = build_selection_plan(scores, None,
                                     coords["structural_original_ids"],
                                     alpha, "uniform", "fixed-image")
        return index, coords

    def test_quota_dominant_and_stable_permutation(self):
        index, coords = self._plan()
        before = copy.deepcopy(index)
        plan = build_spatial_selection_plan(index, coords)
        self.assertEqual(index, before)
        self.assertEqual(plan["layout_policy"], LAYOUT_POLICY)
        self.assertEqual(plan["variant"], "spatial_uniform")
        self.assertEqual(plan["dominant_ids"], index["dominant_ids"])
        self.assertEqual(plan["k"], index["k"])
        self.assertEqual(plan["k_context"], index["k_context"])
        self.assertEqual(plan["spatial"]["index_uniform_aux_ids"],
                         index["contextual_ids"])
        quotas = plan["spatial"]["branch_quotas"]
        for branch in ("base", "high"):
            result = plan["spatial"]["branch_results"][branch]
            self.assertEqual(result["quota"], quotas[branch])
            self.assertEqual(len(result["selected_original_ids"]),
                             quotas[branch])
            self.assertEqual(sum(coords["records"][i]["branch"] == branch
                                 for i in index["contextual_ids"]),
                             quotas[branch])
            self.assertTrue(all(coords["records"][i]["branch"] == branch
                                for i in result["selected_original_ids"]))
        self.assertEqual(sum(quotas.values()), plan["k_context"])
        self.assertEqual(len(plan["selected_original_ids"]), plan["k"])
        self.assertFalse(set(plan["dominant_ids"])
                         & set(plan["contextual_ids"]))
        self.assertEqual(len(set(plan["contextual_ids"])), plan["k_context"])
        self.assertFalse(set(plan["contextual_ids"])
                         & set(coords["structural_original_ids"]))
        self.assertEqual(plan["selected_original_ids"],
                         [i for i in index["content_saliency_ranking"]
                          if i in set(plan["dominant_ids"])
                          | set(plan["contextual_ids"])])
        permutation = plan["stored_to_original"]
        inverse = plan["original_to_stored"]
        self.assertEqual(permutation[:plan["k"]],
                         plan["selected_original_ids"])
        self.assertEqual(permutation[-len(coords["structural_original_ids"]):],
                         coords["structural_original_ids"])
        self.assertEqual([permutation[inverse[i]]
                          for i in range(len(permutation))],
                         list(range(len(permutation))))

    def test_zero_auxiliary_budget_is_exact_frozen_uniform_order(self):
        index, coords = self._plan(side=2, size=(1, 2), pin=(4, 4))
        self.assertEqual(index["k_context"], 0)
        plan = build_spatial_selection_plan(index, coords)
        self.assertEqual(plan["stored_to_original"],
                         index["stored_to_original"])
        self.assertEqual(plan["spatial"]["branch_quotas"],
                         {"base": 0, "high": 0})

    def test_54_to_10_integer_allocation_only(self):
        for n in (1, 40, 256, 257, 1024, 4321):
            k, dominant, aux = split_visual_kv_budget(n, 10 / 64)
            self.assertEqual(k, (n + 3) // 4)
            self.assertEqual(aux, (10 * k) // 64)
            self.assertEqual(dominant, k - aux)
        index, coords = self._plan(alpha=10 / 64)
        self.assertEqual(index["k_context"], 10 * index["k"] // 64)
        self.assertEqual(index["variant"], "uniform")


if __name__ == "__main__":
    unittest.main()

"""CPU contracts for original-token contextual Visual-KV25 planning."""
from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch

from mmimpress.contextual_kv25 import (
    LAYOUT_POLICY, anyres_token_descriptors, build_selection_plan,
    contextual_representatives, split_visual_kv_budget, uniform_target_ids,
    vision_key_patch_descriptors)
from mmimpress.cvpr25 import visionzip_repack_order


def _scores(n: int, separators=(0,)) -> torch.Tensor:
    vn = n + len(separators)
    out = torch.full((vn,), float("inf"))
    for index in range(vn):
        if index not in separators:
            out[index] = float((index * 7) % 17)
    return out


def _descriptors(vn: int) -> torch.Tensor:
    return torch.tensor([[float((i * 3) % 7), float((i * 5) % 11)]
                         for i in range(vn)])


class BudgetAndPlanTests(unittest.TestCase):
    def test_fixed_budget_and_chunk_boundaries(self):
        for n in (0, 1, 2, 3, 4, 63, 64, 65, 127, 255, 256, 257):
            with self.subTest(n=n):
                k, dominant, contextual = split_visual_kv_budget(n, .2)
                self.assertEqual(k, (n + 3) // 4)
                self.assertEqual(contextual, math.floor(.2 * k))
                self.assertEqual(dominant + contextual, k)
        self.assertEqual((split_visual_kv_budget(256, 0)[0] + 63) // 64, 1)
        self.assertEqual((split_visual_kv_budget(257, 0)[0] + 63) // 64, 2)
        for invalid in (-.1, 1.1, float("nan"), float("inf"), True):
            with self.subTest(alpha=invalid), self.assertRaises(ValueError):
                split_visual_kv_budget(20, invalid)

    def test_alpha_zero_is_exact_existing_dominant_order(self):
        scores = torch.tensor([float("inf"), .5, .5, .9, .1, .9,
                               float("inf"), .5])
        separators = [6, 0]
        expected = visionzip_repack_order(scores, separators)
        for variant in ("dominant", "contextual", "random", "uniform"):
            with self.subTest(variant=variant):
                plan = build_selection_plan(scores, None, separators, 0,
                                            variant, "img")
                self.assertEqual(plan["stored_to_original"], expected)
                self.assertEqual(plan["selected_original_ids"], expected[:2])
                self.assertEqual(plan["k_context"], 0)
                self.assertEqual(plan["layout_policy"], LAYOUT_POLICY)

    def test_all_variants_are_global_bijective_fixed_budget_plans(self):
        for n in (1, 4, 67, 255, 256, 257):
            scores = _scores(n, (0,))
            descriptors = _descriptors(n + 1)
            plans = {variant: build_selection_plan(
                scores, descriptors, [0], .2, variant, "img")
                for variant in ("contextual", "random", "uniform")}
            for variant, plan in plans.items():
                with self.subTest(n=n, variant=variant):
                    k = (n + 3) // 4
                    order = plan["stored_to_original"]
                    inverse = plan["original_to_stored"]
                    self.assertEqual(plan["k"], k)
                    self.assertEqual(plan["k_dominant"] +
                                     plan["k_context"], k)
                    self.assertEqual(len(plan["selected_original_ids"]), k)
                    self.assertEqual(order[:k], plan["selected_original_ids"])
                    self.assertEqual(order[-1], 0)
                    self.assertEqual(sorted(order), list(range(n + 1)))
                    self.assertEqual([order[inverse[i]]
                                      for i in range(n + 1)],
                                     list(range(n + 1)))
                    self.assertFalse(set(plan["dominant_ids"]) &
                                     set(plan["contextual_ids"]))
                    self.assertEqual(len(set(plan["dominant_ids"]) |
                                         set(plan["contextual_ids"])), k)
                    self.assertEqual(plan["actual_content_retention_fraction"],
                                     k / n)
            self.assertEqual(plans["contextual"]["dominant_ids"],
                             plans["random"]["dominant_ids"])
            self.assertEqual(plans["contextual"]["dominant_ids"],
                             plans["uniform"]["dominant_ids"])

    def test_uniform_target_positions_and_controls(self):
        self.assertEqual(uniform_target_ids(list(range(12)), 3), [2, 6, 10])
        self.assertEqual(uniform_target_ids([1, 3, 5, 7], 4),
                         [1, 3, 5, 7])
        scores = _scores(96)
        d = _descriptors(97)
        contextual = build_selection_plan(scores, d, [0], .2,
                                          "contextual", "image-a")
        uniform = build_selection_plan(scores, None, [0], .2,
                                       "uniform", "image-a")
        self.assertEqual(contextual["target_ids"], uniform["target_ids"])
        self.assertEqual(uniform["contextual_ids"], uniform["target_ids"])
        self.assertEqual(uniform["assignments"], {})
        random_a = build_selection_plan(scores, None, [0], .2,
                                        "random", "image-a")
        random_b = build_selection_plan(scores, None, [0], .2,
                                        "random", "image-a")
        self.assertEqual(random_a, random_b)
        self.assertEqual(random_a["random_seed"], 1234)
        self.assertEqual(len(random_a["random_seed_digest"]), 64)
        self.assertEqual(random_a["target_ids"], [])

    def test_contextual_selection_is_question_independent(self):
        scores, descriptors = _scores(96), _descriptors(97)
        left = build_selection_plan(scores, descriptors, [0], .2,
                                    "contextual", "same-image")
        right = build_selection_plan(scores, descriptors, [0], .2,
                                     "contextual", "another-label")
        self.assertEqual(left["selected_original_ids"],
                         right["selected_original_ids"])
        self.assertEqual(left["assignments"], right["assignments"])
        self.assertEqual(sum(left["cluster_sizes"]),
                         len(left["remaining_ids"]))
        for target in left["target_ids"]:
            self.assertEqual(left["assignments"][str(target)], target)
        self.assertEqual(len(left["representative_ids"]), left["k_context"])

    def test_nonfinite_descriptor_and_bad_variant_fail_closed(self):
        scores, descriptors = _scores(20), _descriptors(21)
        descriptors[3, 0] = float("nan")
        with self.assertRaises(ValueError):
            build_selection_plan(scores, descriptors, [0], .2,
                                 "contextual", "img")
        with self.assertRaises(ValueError):
            build_selection_plan(scores, None, [0], .2,
                                 "contextual", "img")
        with self.assertRaises(ValueError):
            build_selection_plan(scores, None, [0], .2,
                                 "dominant", "img")
        with self.assertRaises(ValueError):
            build_selection_plan(scores, None, [0], 0,
                                 "unknown", "img")


class RepresentativeTests(unittest.TestCase):
    def test_identical_vectors_target_self_cluster_and_id_ties(self):
        result = contextual_representatives(
            [0, 1, 2, 3], [1, 3], torch.ones(4, 2))
        self.assertEqual(result["assignments"],
                         {"0": 1, "1": 1, "2": 1, "3": 3})
        self.assertEqual(result["cluster_sizes"], [3, 1])
        self.assertEqual(result["representative_ids"], [0, 3])

    def test_zero_centroid_uses_target_and_zero_descriptors_stay_finite(self):
        opposite = torch.tensor([[1., 0.], [-1., 0.]])
        result = contextual_representatives([0, 1], [0], opposite)
        self.assertEqual(result["representative_ids"], [0])
        zeros = contextual_representatives([0, 1, 2, 3], [1, 3],
                                           torch.zeros(4, 2))
        self.assertEqual(zeros["representative_ids"], [1, 3])
        self.assertEqual(zeros["assignments"]["3"], 3)


class DescriptorMappingTests(unittest.TestCase):
    @staticmethod
    def _runner():
        return SimpleNamespace(cfg=SimpleNamespace(
            vision_config=SimpleNamespace(image_size=2, patch_size=1),
            image_grid_pinpoints=[[4, 4]]))

    @staticmethod
    def _patches():
        # Base patches 0..3; each crop uses distinct IDs. The second feature
        # channel proves vector channels are neither flattened nor averaged.
        ids = [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23],
               [30, 31, 32, 33], [40, 41, 42, 43]]
        return torch.tensor([[[float(x), float(x + 1000)] for x in row]
                             for row in ids])

    def test_cls_removed_head_mean_float32_l2_and_zero_norm(self):
        raw = torch.tensor([[[99., 99., 99., 99.],
                             [2., 0., 0., 2.],
                             [1., 0., -1., 0.]]], dtype=torch.float16)
        descriptors = vision_key_patch_descriptors(raw, num_heads=2)
        self.assertEqual(descriptors.shape, (1, 2, 2))
        self.assertEqual(descriptors.dtype, torch.float32)
        self.assertTrue(torch.allclose(
            descriptors[0, 0], torch.tensor([2 ** -.5, 2 ** -.5]),
            atol=1e-6))
        self.assertTrue(torch.equal(descriptors[0, 1], torch.zeros(2)))
        raw[0, 1, 0] = float("inf")
        with self.assertRaises(ValueError):
            vision_key_patch_descriptors(raw, 2)

    def test_anyres_vertical_unpad_and_newline_rows(self):
        mapped = anyres_token_descriptors(self._runner(), self._patches(),
                                          [1, 2], 14)
        expected_ids = [0, 1, 2, 3, 12, 13, 22, 23,
                        None, 30, 31, 40, 41, None]
        self.assertEqual(mapped.shape, (14, 2))
        for row, patch_id in enumerate(expected_ids):
            expected = ([0., 0.] if patch_id is None else
                        [float(patch_id), float(patch_id + 1000)])
            self.assertEqual(mapped[row].tolist(), expected)

    def test_anyres_horizontal_unpad_and_newline_rows(self):
        mapped = anyres_token_descriptors(self._runner(), self._patches(),
                                          [2, 1], 16)
        expected_ids = [0, 1, 2, 3, 11, 20, None, 13, 22, None,
                        31, 40, None, 33, 42, None]
        self.assertEqual(mapped.shape, (16, 2))
        for row, patch_id in enumerate(expected_ids):
            expected = ([0., 0.] if patch_id is None else
                        [float(patch_id), float(patch_id + 1000)])
            self.assertEqual(mapped[row].tolist(), expected)
        with self.assertRaises(ValueError):
            anyres_token_descriptors(self._runner(), self._patches(),
                                     [2, 1], 15)


if __name__ == "__main__":
    unittest.main()

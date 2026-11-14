"""CPU tests of the LLaVA-1.5 HOLD harness helpers (scripts/llava15_support.py)."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import llava15_support as S  # noqa: E402
from mmimpress.llava15 import store as ST  # noqa: E402


class HarnessTests(unittest.TestCase):
    def test_arms_and_vectors(self):
        self.assertEqual(S.METHODS, ["recompute", "llava15_kv_U100_packed", "llava15_kv_U50_packed", "llava15_kv_U25_packed", "llava15_kv_PYR-b20_packed"])
        self.assertEqual([S.vector_of(m) for m in S.METHODS[1:]], list(S.VECTORS))
        self.assertEqual(S.kind_of("recompute"), "none")
        self.assertEqual(S.stored_methods(S.METHODS), S.METHODS[1:])
        self.assertEqual(S.EXPECT["main"], 60915)
        self.assertEqual(sum(S.EXPECTED_BYTES.values()), 603_979_776)

    def test_independent_counts_match_library(self):
        import json
        pyr = json.loads(S.PYR_SOURCE.read_text())["layer_keep_ratios"]
        ratios = {"U100": [1.0] * 32, "U50": [0.5] * 32, "U25": [0.25] * 32, "PYR-b20": pyr}
        for vec in S.VECTORS:
            doc = ST.make_budget(vec, ratios[vec], S.TOTAL_RATIO[vec], {})
            counts = S.independent_counts(ratios[vec], 576, S.FRACTION[vec])
            self.assertEqual(counts, ST.keep_counts(doc, 576))
            self.assertEqual(S.expected_bytes(counts), S.EXPECTED_BYTES[vec])
        with self.assertRaises(ValueError):
            S.independent_counts([1.0] * 32, 576, (1, 2))

    def test_prompt_factory_lineage(self):
        dialog = {"dialog_id": "d", "turns": [{"question": " A? ", "question_id": 1}, {"question": "B?", "question_id": 2}, {"question": "C?", "question_id": 3}]}
        p1, h1, e1 = S.prompt_factory(S.METHODS, dialog, 1, S.U25, {})
        self.assertEqual(p1, "USER: <image>\nCurrent question Q1: A?\nAnswer the current question with a single word or short phrase. ASSISTANT:")
        row = {"method_id": S.U25, "dialog_id": "d", "prediction": "x", "logical_request_id": "l", "physical_execution_id": "p"}
        p2, h2, e2 = S.prompt_factory(S.METHODS, dialog, 2, S.U25, {1: row})
        self.assertEqual(p2, "USER: <image>\nQ1: A?\nA1: x\n\nCurrent question Q2: B?\nAnswer the current question with a single word or short phrase. ASSISTANT:")
        self.assertEqual(e2[0]["source_model"], "llava-hf/llava-1.5-7b-hf")
        with self.assertRaises(ValueError):
            S.prompt_factory(S.METHODS, dialog, 2, S.U50, {1: row})          # foreign method history
        with self.assertRaises(ValueError):
            S.prompt_factory(S.METHODS, dialog, 1, S.U25, {1: row})          # current answer in history

    def test_visibility_and_mask_check(self):
        counts = [3, 0, 576] + [1] * 29
        vis = S.independent_visible(581, 5, counts)
        self.assertEqual([int(v.sum()) for v in vis[:3]], [8, 5, 581])
        order = list(range(575, -1, -1))
        orig = S.original_visible(581, 5, order, counts)
        self.assertTrue(bool(orig[0][5 + 575]) and bool(orig[0][5 + 573]) and not bool(orig[0][5 + 572]))
        neg = torch.finfo(torch.bfloat16).min
        table = {}
        for li, v in enumerate(vis):
            b = torch.full((1, 1, 1, 581), neg, dtype=torch.bfloat16)
            b[0, 0, 0, v] = 0
            table[li] = b
        self.assertEqual(S.check_masks(table, 581, 5, counts)["status"], "PASS")
        table[4][0, 0, 0, 5 + 1] = 0
        res = S.check_masks(table, 581, 5, counts)
        self.assertEqual((res["status"], res["bad_layers"]), ("FAIL", [4]))


if __name__ == "__main__":
    unittest.main()

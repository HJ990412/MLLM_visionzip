"""VZ-Pruned-E2E helpers: pruned sequence, assembly, store gates, recorded comparisons."""
import tempfile
import unittest
from pathlib import Path

import torch

from mmimpress.recompute_kv25 import default_inv_freq, reduced_plan, structural_rows
from mmimpress.vz_pruned_e2e import (
    VZ_NEG_UNREALIGNED_MIN_REL_L2, assemble_pruned_embeds, compare_vz, hit_vs_standalone,
    path_equivalence, prefix_embeds_sha256, pruned_sequence)
from test_recompute_kv25 import _geometry, _stores


class Sequence(unittest.TestCase):
    def test_pruned_sequence(self):
        for n in (1, 64, 349):
            with self.subTest(n=n):
                v_start, vn, sep, order = _geometry(n)
                plan = reduced_plan(v_start, vn, order, sep, "compact")
                index, control = pruned_sequence(plan, 7)
                L = len(plan["input_index"])
                self.assertEqual(len(index), L + 7)
                self.assertEqual(index[L:], list(range(v_start + vn, v_start + vn + 7)))
                self.assertEqual(control["position_ids"], list(range(L + 7)))
                rows = structural_rows(plan)
                self.assertEqual(structural_rows(control)[:len(rows)], rows)
                if n > 1:                      # some token dropped: identical structural rows
                    self.assertEqual(structural_rows(control), rows)
                self.assertEqual(index[:L], plan["input_index"])

    def test_rejects_original_policy(self):
        v_start, vn, sep, order = _geometry(64)
        with self.assertRaises(AssertionError):
            pruned_sequence(reduced_plan(v_start, vn, order, sep, "original"), 3)

    def test_assembly_and_sha(self):
        v_start, vn, sep, order = _geometry(129)
        plan = reduced_plan(v_start, vn, order, sep, "compact")
        D = 6
        sys_e = torch.randn(1, v_start, D).bfloat16()
        feats = torch.randn(vn, D).bfloat16()
        text = torch.randn(1, 5, D).bfloat16()
        x = assemble_pruned_embeds(sys_e, feats, text, plan)
        L = len(plan["input_index"])
        self.assertEqual(tuple(x.shape), (1, L + 5, D))
        self.assertTrue(torch.equal(x[0, :v_start], sys_e[0]))
        self.assertTrue(torch.equal(x[0, L:], text[0]))
        for j, i in enumerate(plan["keep_visual_ids"]):
            self.assertTrue(torch.equal(x[0, v_start + j], feats[i]))
        a = prefix_embeds_sha256(sys_e, feats)
        self.assertEqual(a, prefix_embeds_sha256(sys_e.clone(), feats.clone()))
        feats2 = feats.clone(); feats2[0, 0] += 1
        self.assertNotEqual(a, prefix_embeds_sha256(sys_e, feats2))


class Gates(unittest.TestCase):
    def test_pass(self):
        for n in (64, 349):
            with self.subTest(n=n), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                inv, plans, oracle, control = _stores(root, n)
                r = compare_vz(root / "ours", root / "compact", inv, oracle, control)
                self.assertEqual(r["status"], "PASS", r["failed_gates"])
                for name in ("vz_layer0_k_realigned_S", "vz_layer0_k_oracle_ratio_S",
                             "vz_neg_layer0_k_unrealigned_S", "structural_bitident_k", "layer0_v"):
                    self.assertIn(name, r["gates"])
                self.assertNotIn("layer0_k_realigned", r["gates"])
                self.assertIn("rev3_layer0_k_realigned", r["report_only"])
                self.assertLessEqual(r["gates"]["vz_layer0_k_oracle_ratio_S"][0]["oracle_ratio"], 1.5)
                self.assertGreater(r["gates"]["vz_neg_layer0_k_unrealigned_S"][0]["rel_l2"],
                                   VZ_NEG_UNREALIGNED_MIN_REL_L2)
                self.assertEqual(len(r["layers"]), 3)

    def test_failures(self):
        for perturb, failed in ((("compact", "unrotated"), {"vz_layer0_k_realigned_S"}),
                                (("sys_deep", 0.05), {"structural_bitident_k", "structural_bitident_v"})):
            with self.subTest(perturb=perturb), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                inv, plans, oracle, control = _stores(root, 349, perturb=perturb)
                r = compare_vz(root / "ours", root / "compact", inv, oracle, control)
                self.assertEqual(r["status"], "FAIL")
                self.assertTrue(failed <= set(r["failed_gates"]), r["failed_gates"])


class Recorded(unittest.TestCase):
    def test_path_equivalence(self):
        real = {"first_logits": torch.tensor([0.1, 2.0, 0.3]), "generated_token_ids": [1, 2],
                "first_token_id": 1, "pruned_input_len": 9}
        comp = {"first_logits": torch.tensor([0.1, 2.0, 0.3 + 5e-5]), "generated_token_ids": [1, 2],
                "first_token_id": 1, "prefill_len": 9}
        self.assertTrue(path_equivalence(real, comp)["ok"])
        comp["first_logits"] = torch.tensor([0.1, 2.0, 0.31])
        self.assertFalse(path_equivalence(real, comp)["ok"])
        comp["first_logits"] = real["first_logits"]; comp["generated_token_ids"] = [1, 3]
        self.assertFalse(path_equivalence(real, comp)["ok"])

    def test_hit_vs_standalone_never_raises(self):
        hit = {"first_logits": torch.tensor([0.5, 3.0, 2.0]), "first_token_id": 1,
               "generated_token_ids": [1, 2], "answer": "a"}
        sa = {"first_logits": torch.tensor([0.5, 2.0, 3.0]), "first_token_id": 2,
              "generated_token_ids": [2], "answer": "b"}
        out = hit_vs_standalone(hit, sa)
        self.assertFalse(out["first_token_exact"] or out["answer_exact"] or out["gated"])
        self.assertAlmostEqual(out["max_abs_first_logit_delta"], 1.0)
        self.assertEqual(out["hit_top2"]["argmax"], 1)
        hit["first_logits"] = None
        self.assertIsNone(hit_vs_standalone(hit, sa)["max_abs_first_logit_delta"])


if __name__ == "__main__":
    unittest.main()

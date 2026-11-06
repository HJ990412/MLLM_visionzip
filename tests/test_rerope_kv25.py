"""Ours-ReRoPE / Pruned-T1 helpers: re-rotation, store gates, statistics."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mmimpress.recompute_kv25 import (
    StoreRows, bit_equal, default_inv_freq, reduced_plan, rotate)
from mmimpress.rerope_kv25 import (
    NEG_UNROTATED_MIN_REL_L2, LayerPositions, compare_recorded, compare_rerope,
    contrast_bootstrap, rerope_layers, top2_margin)
from test_recompute_kv25 import _geometry, _rope, _write


def _fixture(root, n, L=3, H=2, hd=8, perturb=None):
    """Ours (full BF16), Compact (reduced, p') and ReRoPE stores."""
    v_start, vn, sep, order = _geometry(n)
    P = v_start + vn
    g = torch.Generator().manual_seed(n)
    inv = default_inv_freq(hd)
    raw_k = torch.randn(L, P, H, hd, generator=g, dtype=torch.float64)
    raw_v = torch.randn(L, P, H, hd, generator=g, dtype=torch.float64)
    plan = reduced_plan(v_start, vn, order, sep, "compact")
    pos_full = torch.arange(P, dtype=torch.float64)
    ours, compact = [], []
    for li in range(L):
        k_full = _rope(raw_k[li], pos_full, inv)
        ours.append((k_full.permute(1, 0, 2)[None].bfloat16(),
                     raw_v[li].permute(1, 0, 2)[None].bfloat16()))
        idx = plan["input_index"]
        pos = torch.tensor(plan["position_ids"], dtype=torch.float64)
        # BF16 compute before the FP16 store, as on the GPU (equidistant from the oracle)
        k_red = _rope(raw_k[li, idx], pos, inv).bfloat16().double()
        if perturb == "compact_noise" and li == 0:
            k_red = k_red * 1.2
        full_k = torch.zeros(P, H, hd, dtype=torch.float64)
        full_v = torch.zeros(P, H, hd, dtype=torch.float64)
        full_k[idx] = k_red
        full_v[idx] = raw_v[li, idx]
        compact.append((full_k.permute(1, 0, 2)[None].half(),
                        full_v.permute(1, 0, 2)[None].half()))
    rr, rec = rerope_layers(ours, plan, inv)
    if perturb == "v_changed":
        rr = [(k, v.clone()) for k, v in rr]
        rr[1][1][0, 0, v_start + plan["keep_visual_ids"][0], 0] += 1.0
    if perturb == "unrotated":
        rr = [(k.to(torch.float16), v) for k, v in ours]
    for name, layers in (("ours", ours), ("compact", compact), ("rerope", rr)):
        _write(root / name, layers, v_start, vn, sep, order)
    oracle = {"k": _rope(raw_k[0], pos_full, inv), "v": raw_v[0]}
    return inv, plan, oracle, rec


class ReRotation(unittest.TestCase):
    def test_rows_and_exactness(self):
        v_start, vn, sep, order = _geometry(349)
        plan = reduced_plan(v_start, vn, order, sep, "compact")
        P, H, hd = v_start + vn, 2, 8
        inv = default_inv_freq(hd)
        g = torch.Generator().manual_seed(1)
        layers = [(torch.randn(1, H, P, hd, generator=g).bfloat16(),
                   torch.randn(1, H, P, hd, generator=g).bfloat16()) for _ in range(2)]
        out, rec = rerope_layers(layers, plan, inv)
        self.assertEqual(rec["rotated_rows"], plan["k"] + plan["n_sep"])
        self.assertLessEqual(rec["delta_max"], 0)
        kept = torch.tensor([v_start + i for i in plan["keep_visual_ids"]])
        others = torch.tensor([r for r in range(P) if r not in set(kept.tolist())])
        for (k, v), (k16, v16) in zip(layers, out):
            self.assertIs(v16, v)
            self.assertEqual(k16.dtype, torch.float16)
            self.assertTrue(torch.equal(k16[0, :, others].float(), k[0, :, others].float()))
            delta = torch.tensor([plan["visual_position"][i] - (v_start + i)
                                  for i in plan["keep_visual_ids"]], dtype=torch.float64)
            want = rotate(k[0, :, kept].transpose(0, 1), delta, inv).to(torch.float16)
            self.assertTrue(bit_equal(k16[0, :, kept].transpose(0, 1).contiguous(), want.contiguous()))

    def test_rejects_original_plan(self):
        v_start, vn, sep, order = _geometry(64)
        plan = reduced_plan(v_start, vn, order, sep, "original")
        with self.assertRaises(AssertionError):
            rerope_layers([(torch.zeros(1, 2, v_start + vn, 8).bfloat16(),) * 2], plan, default_inv_freq(8))


class Gates(unittest.TestCase):
    def test_pass_and_records(self):
        for n in (64, 349):
            with self.subTest(n=n), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                inv, plan, oracle, _ = _fixture(root, n)
                r = compare_rerope(root / "ours", root / "rerope", root / "compact", inv, oracle)
                self.assertEqual(r["status"], "PASS", r["failed_gates"])
                g0 = r["gates"]["same_position_layer0_k_S"][0]
                self.assertLess(g0["rel_l2"], 5e-3)
                self.assertGreater(r["gates"]["neg_unrotated_layer0_k_S"][0]["rel_l2"], NEG_UNROTATED_MIN_REL_L2)
                self.assertTrue(all(x["ok"] for x in r["gates"]["v_bitident_S"]))
                self.assertEqual(len(r["layers"]), 3)
                self.assertGreater(r["layers"][0]["k_S_vs_ours_unrotated"]["rel_l2"], 0.1)

    def test_failures(self):
        for perturb, failed in (("unrotated", {"same_position_layer0_k_S", "same_position_layer0_k_sep"}),
                                ("v_changed", {"v_bitident_S"}),
                                ("compact_noise", {"same_position_layer0_k_S", "same_position_layer0_k_sep"})):
            with self.subTest(perturb=perturb), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                inv, plan, oracle, _ = _fixture(root, 349, perturb=perturb)
                r = compare_rerope(root / "ours", root / "rerope", root / "compact", inv, oracle)
                self.assertEqual(r["status"], "FAIL")
                self.assertTrue(failed <= set(r["failed_gates"]), r["failed_gates"])
                if perturb == "unrotated":
                    # round trip of an un-rotated store is not Ours either
                    self.assertIn("roundtrip_k_S", r["failed_gates"])
                if perturb == "compact_noise":
                    self.assertNotIn("roundtrip_k_S", r["failed_gates"])

    def test_rows_are_fp16_exact_copies_of_bf16(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fixture(root, 129)
            ours, rr = StoreRows(root / "ours"), StoreRows(root / "rerope")
            for li in range(ours.L):
                self.assertTrue(bit_equal(rr.system(li, "k"), ours.system(li, "k")))
                self.assertTrue(bit_equal(rr.content(li, "v"), ours.content(li, "v")))
                self.assertFalse(bit_equal(rr.content(li, "k"), ours.content(li, "k")))


class Positions(unittest.TestCase):
    def test_verify(self):
        lp = LayerPositions.__new__(LayerPositions)
        lp.calls = [(torch.arange(7)[None], torch.arange(7)),
                    (torch.tensor([[7]]), torch.tensor([7])), (torch.tensor([[8]]), torch.tensor([8]))]
        self.assertEqual(lp.verify(7)["calls"], 3)
        lp.calls[1] = (torch.tensor([[9]]), torch.tensor([7]))
        with self.assertRaises(AssertionError):
            lp.verify(7)
        lp.calls = [(torch.arange(1, 8)[None], torch.arange(7))]
        with self.assertRaises(AssertionError):
            lp.verify(7)

    def test_top2_margin(self):
        m = top2_margin(torch.tensor([0.5, 3.0, 2.0]))
        self.assertEqual(m["argmax"], 1)
        self.assertAlmostEqual(m["margin"], 1.0)
        self.assertIsNone(top2_margin(None))

    def test_compare_recorded_never_raises(self):
        left = {"logits": [torch.zeros(4)], "first_token_id": 1, "generated_token_ids": [1], "answer": "a"}
        right = {"logits": [torch.full((4,), 0.2)], "first_token_id": 1, "generated_token_ids": [1], "answer": "a"}
        out = compare_recorded(left, right)
        self.assertFalse(out["all_step_logits_within_1e-4"])
        self.assertTrue(out["first_token_exact"] and out["answer_exact"] and not out["gated"])
        self.assertAlmostEqual(out["max_abs_logit_delta"], 0.2, places=6)


class Contrasts(unittest.TestCase):
    def test_joint_bootstrap(self):
        rng = np.random.default_rng(0)
        per = {"ours": rng.random(40), "orig": rng.random(40), "compact": rng.random(40)}
        per["rerope"] = per["compact"].copy()
        out = contrast_bootstrap(per, {
            "same": {"rerope": 1, "compact": -1},
            "context": {"ours": .5, "rerope": .5, "orig": -.5, "compact": -.5},
            "position": {"compact": .5, "rerope": .5, "ours": -.5, "orig": -.5},
            "interaction": {"ours": 1, "rerope": -1, "orig": -1, "compact": 1}}, resamples=500, seed=3)
        self.assertEqual(out["same"]["mean"], 0.0)
        self.assertEqual((out["same"]["ci_low"], out["same"]["ci_high"]), (0.0, 0.0))
        want = ((per["ours"] + per["rerope"]) / 2 - (per["orig"] + per["compact"]) / 2).mean()
        self.assertAlmostEqual(out["context"]["mean"], want)
        self.assertLessEqual(out["context"]["ci_low"], out["context"]["mean"])
        self.assertGreaterEqual(out["context"]["ci_high"], out["context"]["mean"])
        # identical seed -> identical resamples for every contrast
        again = contrast_bootstrap(per, {"context": out["context"]["contrast"]}, resamples=500, seed=3)
        self.assertEqual(again["context"]["ci_low"], out["context"]["ci_low"])
        self.assertTrue(out["interaction"]["joint"])


if __name__ == "__main__":
    unittest.main()

"""PosShift dose-response helpers: schedule, checks, pre-fixed reading."""
import unittest

import torch

from mmimpress.posshift_dose import (
    DELTAS, bitwise_equal_logits, cap_reached, monotone_non_increasing, reading,
    shift_schedule, tokens_equal)


class Schedule(unittest.TestCase):
    def test_full_schedule(self):
        # v0=37, V=2112, k=528, n_sep=... -> prefix 2149, compact 565 (an actual 061243Z image)
        sched, dmax = shift_schedule(2149, 565)
        self.assertEqual(dmax, 1584)
        self.assertEqual([d for d, _, _ in sched], list(DELTAS) + [1584])
        self.assertEqual([s for _, s, _ in sched], [2149 - d for d in DELTAS] + [565])
        self.assertEqual(sched[0][2], "shift_0"); self.assertEqual(sched[-1][2], "shift_max")

    def test_small_dmax_drops_duplicates(self):
        sched, dmax = shift_schedule(1000, 400)
        self.assertEqual(dmax, 600)
        self.assertEqual([d for d, _, _ in sched], [0, 1, 4, 16, 64, 256, 600])
        with self.assertRaises(AssertionError):
            shift_schedule(400, 400)


class Checks(unittest.TestCase):
    def test_logits_and_tokens(self):
        a = [torch.tensor([1., 2.]), torch.tensor([3., 4.])]
        self.assertTrue(bitwise_equal_logits(a, [x.clone() for x in a]))
        b = [a[0], a[1] + 1e-6]
        self.assertFalse(bitwise_equal_logits(a, b))
        self.assertFalse(bitwise_equal_logits(a, a[:1]))
        self.assertFalse(bitwise_equal_logits(None, a))
        r = {"first_token_id": 3869, "generated_token_ids": [3869, 2], "prediction": "Yes"}
        self.assertTrue(tokens_equal(r, dict(r)))
        self.assertFalse(tokens_equal(r, dict(r, prediction="No")))

    def test_cap(self):
        self.assertTrue(cap_reached({"generated_token_ids": [13] * 16}))
        self.assertFalse(cap_reached({"generated_token_ids": [13] * 15 + [2]}))
        self.assertFalse(cap_reached({"generated_token_ids": [3869, 2]}))


class Reading(unittest.TestCase):
    def test_rules(self):
        base = {0: .565, 1: .56, 4: .55, 16: .5, 64: .3, 256: .1, 1024: .0, 1584: .0}
        self.assertEqual(reading(base)["reading"], "graded"); self.assertEqual(reading(base)["delta_star"], 1024)
        self.assertEqual(reading({**base, 256: .04})["delta_star"], 256)   # 10% at 256 is not collapsed; 4% is
        abrupt = {**base, 1: .02, 4: .0}
        self.assertEqual(reading(abrupt)["reading"], "abrupt"); self.assertEqual(reading(abrupt)["delta_star"], 1)
        none = {d: .4 for d in base}
        self.assertEqual(reading(none)["reading"], "not_reproduced"); self.assertIsNone(none and reading(none)["delta_star"])
        self.assertEqual(reading({0: .5, 16: .05})["delta_star"], 16)      # <= 5% counts as collapsed

    def test_monotone(self):
        self.assertTrue(monotone_non_increasing([.5, .505, .4, .4, .0]))
        self.assertFalse(monotone_non_increasing([.5, .52, .4]))


if __name__ == "__main__":
    unittest.main()

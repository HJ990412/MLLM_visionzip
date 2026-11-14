"""CPU tests of the LLaVA-1.5 ratio-general packed store / selector (mmimpress.llava15)."""
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mmimpress import layer_budget as LB
from mmimpress.cvpr25 import visionzip_repack_order
from mmimpress.llava15 import N_VISUAL, NUM_LAYERS, R_MAX, R_MIN
from mmimpress.llava15 import store as ST
from mmimpress.llava15.serve import Llava15PackedContext, Llava15PrefixSelector, PackedReadTrace
from mmimpress.prefix25 import seal_integrity
from mmimpress.serve import BIAS

ROOT = Path(__file__).resolve().parents[1]
PYR = ROOT / "results/layer_budget_kv25_pyrbeta_20261110T073604Z/layer_budget_llava_PYR-beta-raw.json"
SHA = {"U100": "04c01bedee671a22789fcaedf3348f178446db6f914bfdb98bb1699ab64bc114",
       "U50": "673c2550205b17ed2a56548d2c9524811150da3c49cc4b370f80d05dec1201b0",
       "U25": "54ea273e1a38b1e8b02b96237381f982344cd28ca8466162876e161cdb8ff3ba",
       "PYR-b20": "51f9b0db19a93f8ca8b34b8beb8af21285721c5ea460566afbd9a3185b331724"}
ROW = 32 * 128 * 2
STATS = {"capture_saliency": True, "vision_call_count": 1, "saliency_call_count": 1, "extra_vision_forward_calls": 0,
         "extra_saliency_forward_ms": 0.0, "saliency_layer_from_end": 2, "vision_attention_backend": "eager",
         "global_vision_output_attentions": False, "global_model_output_attentions": False, "vision_ms": 1.0,
         "vision_num_layers": 24, "saliency_layer_index": 22}


def budgets():
    pyr = json.loads(PYR.read_text())["layer_keep_ratios"]
    return {"U100": ST.make_budget("U100", [1.0] * 32, 1.0, {"kind": "test"}), "U50": ST.make_budget("U50", [0.5] * 32, 0.5, {"kind": "test"}),
            "U25": ST.make_budget("U25", [0.25] * 32, 0.25, {"kind": "test"}), "PYR-b20": ST.make_budget("PYR-b20", pyr, 0.25, {"kind": "test"})}


class Runner:
    model = SimpleNamespace(device=torch.device("cpu"))
    image_token_id = 32000

    def visual_span(self, ids):
        pos = (ids == self.image_token_id).nonzero(as_tuple=True)[0]
        assert pos.numel() == 576 and int(pos[-1]) - int(pos[0]) == 575
        return int(pos[0]), 576

    def flat_layout(self, v_num):
        assert v_num == 576
        return 24, []


class BudgetTests(unittest.TestCase):
    def test_constants_and_ratio_sha(self):
        self.assertEqual((N_VISUAL, NUM_LAYERS, R_MIN, R_MAX), (576, 32, 0.0, 1.0))
        for name, doc in budgets().items():
            self.assertEqual(doc["layer_budget_sha256"], SHA[name])
            self.assertEqual((doc["r_min"], doc["r_max"], doc["num_layers"]), (0.0, 1.0, 32))

    def test_rows_and_bytes_at_576(self):
        b = budgets()
        self.assertEqual(ST.keep_counts(b["U100"], 576), [576] * 32)
        self.assertEqual(ST.keep_counts(b["U50"], 576), [288] * 32)
        self.assertEqual(ST.keep_counts(b["U25"], 576), [144] * 32)
        pyr = ST.keep_counts(b["PYR-b20"], 576)
        self.assertEqual((pyr[:3], pyr[-4:], sum(pyr)), ([281, 272, 263], [34, 25, 16, 7], 4608))
        want = {"U100": 301_989_888, "U50": 150_994_944, "U25": 75_497_472, "PYR-b20": 75_497_472}
        for name, doc in b.items():
            self.assertEqual(sum(2 * k * ROW for k in ST.keep_counts(doc, 576)), want[name])
            self.assertEqual(ST.uniform_equivalent(doc, 576) * 32, ST.row_budget(doc, 576))

    def test_module_defaults_are_never_sufficient(self):
        b = budgets()
        for name in ("U100", "U50", "PYR-b20"):
            with self.assertRaises(ValueError):
                LB.layer_keep_counts(b[name]["layer_keep_ratios"], 576)
        for key in ("num_layers", "total_ratio", "r_min", "r_max"):
            with self.assertRaises(ValueError):
                ST.keep_counts({k: v for k, v in b["U25"].items() if k != key}, 576)
        for change in (dict(r_min=0.10), dict(r_max=0.60), dict(total_ratio=0.5), dict(num_layers=28), dict(layer_budget_sha256=SHA["U50"])):
            with self.assertRaises(ValueError):
                ST.keep_counts(dict(b["U25"], **change), 576)
        with self.assertRaises(ValueError):
            ST.make_budget("bad", [0.25] * 32, 0.5, {})


class StoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="llava15_store_test_"))
        g = torch.Generator().manual_seed(1234)
        cls.v0 = 5
        cls.layers = [(torch.randn((1, 32, 5 + 576 + 2, 128), generator=g).to(torch.bfloat16),
                       torch.randn((1, 32, 5 + 576 + 2, 128), generator=g).to(torch.bfloat16)) for _ in range(32)]
        cls.scores = torch.rand((1, 576), generator=g)
        cls.scores[0, 3] = cls.scores[0, 500]
        cls.ids = torch.tensor([1, 2, 3, 4, 5] + [32000] * 576 + [9, 9])
        cls.docs = budgets()
        cls.order = visionzip_repack_order(cls.scores[0], [])
        cls.meta = {}
        for name in ("U25", "PYR-b20"):
            r = ST.persist_captured_llava15_prefix(Runner(), cls.layers, cls.ids, cls.scores, cls.tmp / name, budget=cls.docs[name],
                                                   image_id="t", capture_stats=STATS, image_input_sha256="0" * 64)
            seal_integrity(cls.tmp / name)
            cls.meta[name] = r["meta"]

    @classmethod
    def tearDownClass(cls):
        BIAS.clear()
        shutil.rmtree(cls.tmp)

    def test_order_and_file_layout(self):
        self.assertLess(self.order.index(3), self.order.index(500))          # stable tie
        for name, meta in self.meta.items():
            counts = ST.keep_counts(self.docs[name], 576)
            self.assertEqual(meta["order"], self.order)
            self.assertEqual(meta["layer_keep_counts"], counts)
            self.assertEqual((self.tmp / name / "kv_packed.bin").stat().st_size, 75_497_472)
            self.assertEqual((meta["packed_sep_bytes"], meta["newline_idx"], meta["bytes_separator_sidecar"]), (0, [], 0))
            raw = (self.tmp / name / "kv_packed.bin").read_bytes()
            off = 0
            for li in (0, 13, 31):
                off = meta["packed_layout"][li]["k_offset"]
                k = counts[li]
                src = self.layers[li][0]
                ids = torch.tensor(self.order[:k]) + self.v0
                want = src[0][:, ids].to(torch.float16).transpose(0, 1).contiguous().numpy().tobytes()
                self.assertEqual(raw[off:off + k * ROW], want)
            ST.validate_llava15_packed_meta(meta, self.docs[name])

    def test_validator_rejects_tampering(self):
        m = self.meta["PYR-b20"]
        bad = copy.deepcopy(m)
        bad["layer_keep_counts"][0] += 1
        cases = [bad, dict(m, layer_budget_r_min=0.10), dict(m, layer_budget_r_max=0.60), dict(m, layer_budget_total_ratio=0.5),
                 dict(m, layer_budget_sha256=SHA["U25"]), dict(m, newline_idx=[575]), dict(m, v_token_num=575), dict(m, packed_total_bytes=1),
                 dict(m, store_format="prefixstore_kv25_packed_v1"), {k: v for k, v in m.items() if k != "layer_budget_total_ratio"}]
        for case in cases:
            with self.assertRaises(ValueError):
                ST.validate_llava15_packed_meta(case)
        with self.assertRaises(ValueError):
            ST.validate_llava15_packed_meta(m, self.docs["U25"])

    def test_selector_visible_rows_and_bits(self):
        for name in ("U25", "PYR-b20"):
            counts = ST.keep_counts(self.docs[name], 576)
            ctx = Llava15PackedContext(self.tmp / name, "cpu", pin=False, expected_budget=self.docs[name])
            ctx.validate_packed_layout()
            cache = ctx.cache.new_request()
            sel = Llava15PrefixSelector(Runner(), ctx)
            with patch("torch.cuda.synchronize"), PackedReadTrace(self.tmp / name) as trace:
                sel.prepare()
            s = trace.summary()
            self.assertEqual((s["preadv_calls"], s["pread_calls"], s["bytes"]), (1, 0, 75_497_472))
            for li in (0, 15, 31):
                k = counts[li]
                b = BIAS[li][0, 0, 0]
                self.assertEqual(int((b[self.v0:] == 0).sum()), k)
                self.assertTrue(bool((b[:self.v0 + k] == 0).all()))
                ids = torch.tensor(self.order[:k]) + self.v0
                want = self.layers[li][0][0][:, ids].to(torch.float16).to(torch.bfloat16)
                self.assertTrue(torch.equal(cache.layers[li].keys[0, :, self.v0:self.v0 + k], want))
                self.assertTrue(bool((cache.layers[li].keys[0, :, self.v0 + k:] == 0).all()))
            st = sel.stats()
            self.assertEqual(st["k_per_layer"], counts)
            self.assertEqual(st["selected_original_ids"], self.order[:max(counts)])
            BIAS.clear()
            ctx.close()

    def test_activation_fails_closed(self):
        with self.assertRaises(ValueError):
            Llava15PackedContext(self.tmp / "U25", "cpu", pin=False, expected_budget=self.docs["PYR-b20"])
        d = self.tmp / "trunc"
        shutil.copytree(self.tmp / "U25", d)
        with (d / "kv_packed.bin").open("r+b") as f:
            f.truncate(75_497_472 - ROW)
        with self.assertRaises(ValueError):
            Llava15PackedContext(d, "cpu", pin=False)
        shutil.rmtree(d)

    def test_writer_refuses_other_geometry(self):
        with self.assertRaises(ValueError):
            ST.write_llava15_packed_store(self.tmp / "w", self.layers, self.v0, 575, [], self.order[:575], budget=self.docs["U25"])
        with self.assertRaises(ValueError):
            ST.token_scores_from_capture(torch.zeros(2, 576))


if __name__ == "__main__":
    unittest.main()

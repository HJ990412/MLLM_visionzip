"""Reduced-token recompute KV25: plans, positions, store gates, cleanup guard."""
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import torch

from mmimpress.cvpr25 import visionzip_repack_order
from mmimpress.recompute_kv25 import (
    LAYER0_TOL, SENTINEL, PositionRecorder, PositionServer, cache_sentinel_check,
    compare_stores, default_inv_freq, expected_positions, guarded_rmtree,
    paired_bootstrap, payload_sizes, reduced_plan, rotate, selection_digest,
    sentinel_rows, stat_inventory, synthetic_layers, text_position_start,
    bit_equal, structural_identity, structural_rows, within)
from mmimpress.serve import BIAS
from mmimpress.store import write_image_store

CASES = (1, 63, 64, 65, 127, 128, 129, 349, 2200)


def _geometry(n, n_sep=3, v_start=4):
    vn = n + n_sep
    sep = sorted({(vn - 1) * (j + 1) // n_sep for j in range(n_sep)})
    while len(sep) < n_sep:
        sep = sorted(set(sep) | {min(set(range(vn)) - set(sep))})
    scores = torch.full((vn,), float("inf"))
    for i in range(vn):
        if i not in sep:
            scores[i] = float((i * 7) % 23) / 23.0
    order = visionzip_repack_order(scores, sep)
    return v_start, vn, sep, order


def _rope(x, pos, inv):
    """Independent HF-convention RoPE on (rows, H, hd) via apply_rotary_pos_emb."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    freqs = torch.as_tensor(pos, dtype=torch.float64)[:, None] * inv[None]
    emb = torch.cat([freqs, freqs], -1)
    q = x.permute(1, 0, 2)[None].double()
    out, _ = apply_rotary_pos_emb(q, q, emb.cos()[None], emb.sin()[None])
    return out[0].permute(1, 0, 2)


def _write(root, layers, v_start, vn, sep, order):
    extra = {"physical_layout": "visionzip_image_only"}
    return write_image_store(root, layers, v_start, vn,
                             list(range(v_start + vn)), sep, extra=extra,
                             chunk_size=64, probe_heads=0,
                             stored_to_original=order, separator_sidecar=True,
                             storage_policy="prefix25")


def _stores(root, n, L=3, H=2, hd=8, perturb=None):
    """Ours (full context) and Orig/Compact (reduced) stores with known KV."""
    v_start, vn, sep, order = _geometry(n)
    P = v_start + vn
    g = torch.Generator().manual_seed(n)
    inv = default_inv_freq(hd)
    raw_k = torch.randn(L, P, H, hd, generator=g, dtype=torch.float64)
    raw_v = torch.randn(L, P, H, hd, generator=g, dtype=torch.float64)
    ctx = torch.randn(L, P, H, hd, generator=g, dtype=torch.float64)
    plans = {p: reduced_plan(v_start, vn, order, sep, p)
             for p in ("original", "compact")}
    plan = plans["original"]
    first_drop = min([i for i in range(vn) if i not in set(plan["keep_visual_ids"])]
                     or [vn])

    def layers(policy, reduced):
        pl = plans[policy]
        pos = torch.tensor(pl["position_ids"], dtype=torch.float64) \
            if reduced else torch.arange(P, dtype=torch.float64)
        index = pl["input_index"] if reduced else list(range(P))
        out = []
        for li in range(L):
            k, v = raw_k[li, index].clone(), raw_v[li, index].clone()
            if li > 0 and reduced:
                # deeper layers: context differs except sys/common prefix rows
                for r, src in enumerate(index):
                    if src >= v_start + first_drop:
                        k[r] += 0.5 * ctx[li, src]
                        v[r] += 0.5 * ctx[li, src]
            if perturb and perturb[0] == policy and li == 0 and reduced \
                    and not isinstance(perturb[1], str):
                v[v_start:] += perturb[1]
            if perturb and perturb[0] == "sys_deep" and li > 0 and reduced:
                k[:v_start] *= 1 + perturb[1]
                v[:v_start] *= 1 + perturb[1]
            if perturb and perturb[0] == "k0_noise" and li == 0 and reduced:
                k = k + perturb[1] * torch.randn(k.shape, generator=g, dtype=torch.float64)
            k = _rope(k, pos, inv)
            if policy == "compact" and reduced and perturb == ("compact", "unrotated"):
                k = _rope(raw_k[li, index], torch.tensor(pl["input_index"],
                                                        dtype=torch.float64), inv)
            out.append((k.permute(1, 0, 2)[None].half(),
                        v.permute(1, 0, 2)[None].half()))
        if not reduced:
            return out
        return synthetic_layers(out, pl, P)

    for name, policy, reduced in (("ours", "original", False),
                                  ("orig", "original", True),
                                  ("compact", "compact", True)):
        _write(root / name, layers(policy, reduced), v_start, vn, sep, order)
    oracle = {"k": _rope(raw_k[0], torch.arange(P, dtype=torch.float64), inv),
              "v": raw_v[0]}
    # Truncated original sequence: structural rows see the original context.
    rows = structural_rows(plan)
    control = {"rows": rows, "layers": [
        tuple(t[0, :, rows].transpose(0, 1) for t in pair)
        for pair in layers("original", False)]}
    return inv, plans, oracle, control


class Plan(unittest.TestCase):
    def test_boundaries(self):
        for n in CASES:
            with self.subTest(n=n):
                v_start, vn, sep, order = _geometry(n)
                k = (n + 3) // 4
                o = reduced_plan(v_start, vn, order, sep, "original")
                c = reduced_plan(v_start, vn, order, sep, "compact")
                keep = sorted(order[:k] + sep)
                self.assertEqual(o["keep_visual_ids"], keep)
                self.assertEqual(o["input_index"], c["input_index"])
                self.assertEqual(o["input_index"],
                                 list(range(v_start)) + [v_start + i for i in keep])
                self.assertEqual(o["position_ids"], o["input_index"])
                self.assertEqual(c["position_ids"], list(range(v_start + k + len(sep))))
                self.assertEqual(o["text_position_start"], v_start + vn)
                self.assertEqual(c["text_position_start"], v_start + k + len(sep))
                self.assertEqual(o["selected_original_ids"], order[:k])
                meta = {"v_token_start": v_start, "v_token_num": vn,
                        "newline_idx": sep}
                self.assertEqual(text_position_start(meta, "original"), v_start + vn)
                self.assertEqual(text_position_start(meta, "compact"),
                                 v_start + k + len(sep))

    def test_rejects_bad_inputs(self):
        v_start, vn, sep, order = _geometry(10)
        with self.assertRaises(ValueError):
            reduced_plan(v_start, vn, order, sep, "shifted")
        with self.assertRaises(ValueError):
            reduced_plan(v_start, vn, order[:-1] + [order[0]], sep, "original")
        bad = [x for x in order if x not in sep][:1] + sep + \
            [x for x in order if x not in sep][1:]
        with self.assertRaises(ValueError):
            reduced_plan(v_start, vn, bad, sep, "original")

    def test_expected_positions(self):
        p, c = expected_positions(110, 7, 100, None)
        self.assertEqual(p.tolist(), list(range(103, 110)))
        self.assertEqual(c.tolist(), list(range(103, 110)))
        p, c = expected_positions(110, 10, 100, 40)          # prefill
        self.assertEqual(p.tolist(), list(range(40, 50)))
        self.assertEqual(c.tolist(), list(range(100, 110)))
        p, c = expected_positions(113, 1, 100, 40)           # decode step 3
        self.assertEqual(p.tolist(), [52])
        self.assertEqual(c.tolist(), [112])
        with self.assertRaises(ValueError):
            expected_positions(99, 1, 100, 40)


class Rope(unittest.TestCase):
    def test_matches_hf_and_composes(self):
        inv = default_inv_freq(16)
        x = torch.randn(5, 2, 16, dtype=torch.float64)
        pos = torch.tensor([0., 3., 17., 600., 2900.])
        self.assertTrue(torch.allclose(rotate(x, pos, inv), _rope(x, pos, inv),
                                       atol=1e-10))
        q = torch.tensor([1., 2., 3., 4., 5.])
        realigned = rotate(_rope(x, pos, inv), q - pos, inv)
        self.assertTrue(torch.allclose(realigned, _rope(x, q, inv), atol=1e-9))


class StoreGates(unittest.TestCase):
    def test_identity_sentinel_and_pass(self):
        for n in (1, 64, 129, 349):
            with self.subTest(n=n), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                inv, plans, oracle, control = _stores(root, n)
                metas = {x: __import__("json").loads((root / x / "meta.json").read_text())
                         for x in ("ours", "orig", "compact")}
                self.assertEqual(len({selection_digest(m) for m in metas.values()}), 1)
                self.assertEqual(len({tuple(payload_sizes(root / x).items())
                                      for x in metas}), 1)
                for x in metas:
                    self.assertEqual(sentinel_rows(root / x), 0)
                o = compare_stores(root / "ours", root / "orig", "original", inv, oracle,
                                   control)
                self.assertEqual(o["structural_rows"], len(control["rows"]))
                if n == 1:   # nothing dropped: recomputation is indistinguishable
                    self.assertEqual(o["failed_gates"],
                                     ["impl_check_last_layer_context_changed"])
                else:
                    self.assertEqual(o["status"], "PASS", o["failed_gates"])
                c = compare_stores(root / "ours", root / "compact", "compact", inv, oracle,
                                   control)
                failed = set(c["failed_gates"])
                # n=1 has no dropped token, so no row can move position
                if plans["compact"]["k"] + plans["compact"]["n_sep"] == \
                        plans["compact"]["v_num"]:
                    self.assertEqual(failed, {"neg_layer0_k_unaligned_moved_rows"})
                else:
                    self.assertEqual(c["status"], "PASS", c["failed_gates"])

    def test_layer0_mismatch_and_unrotated_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inv, _, oracle, control = _stores(root, 349, perturb=("original", 0.05))
            o = compare_stores(root / "ours", root / "orig", "original", inv, oracle,
                               control)
            self.assertIn("layer0_v", o["failed_gates"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inv, _, oracle, control = _stores(root, 349, perturb=("compact", "unrotated"))
            c = compare_stores(root / "ours", root / "compact", "compact", inv, oracle,
                               control)
            self.assertIn("layer0_k_realigned", c["failed_gates"])
            self.assertIn("neg_layer0_k_unaligned_moved_rows", c["failed_gates"])

    def test_identical_context_fails_implementation_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inv, _, oracle, control = _stores(root, 129)
            o = compare_stores(root / "ours", root / "ours", "original", inv, oracle,
                               control)
            self.assertEqual(o["failed_gates"],
                             ["impl_check_last_layer_context_changed"])

    def test_rev3_structural_gate(self):
        """Any structural-row change fails bit-identity; vs-Ours bounds are report-only."""
        bits = ["structural_bitident_k", "structural_bitident_v"]
        cases = [(("sys_deep", 0.02), "original", bits, True),
                 (("sys_deep", 0.04), "original", bits, False),
                 (("sys_deep", 0.04), "compact", bits, False),
                 (("k0_noise", 0.01), "original", ["layer0_k", "structural_bitident_k"], True)]
        for perturb, policy, failed, within_rev2 in cases:
            with self.subTest(perturb=perturb, policy=policy), \
                    tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                inv, _, oracle, control = _stores(root, 349, perturb=perturb)
                o = compare_stores(root / "ours", root / ("orig" if policy == "original"
                                                          else "compact"),
                                   policy, inv, oracle, control)
                self.assertEqual(o["failed_gates"], failed)
                bad = o["gates"]["structural_bitident_v" if perturb[0] == "sys_deep"
                                 else "structural_bitident_k"]
                self.assertTrue(any(r["mismatched_rows"] for r in bad if not r["ok"]))
                deep = o["report_only"]["sys_deep_v"]
                self.assertEqual([r["layer"] for r in deep], [1, 2])
                if perturb[0] == "sys_deep":
                    self.assertEqual(all(r["within_rev2_bound"] for r in deep), within_rev2)

    def test_control_rows_must_match_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inv, _, oracle, control = _stores(root, 349)
            short = {"rows": control["rows"][:-1],
                     "layers": [(k[:-1], v[:-1]) for k, v in control["layers"]]}
            with self.assertRaises(AssertionError):
                compare_stores(root / "ours", root / "orig", "original", inv, oracle, short)

    def test_structural_rows_and_identity(self):
        v_start, vn, sep, order = _geometry(349)
        for policy in ("original", "compact"):
            plan = reduced_plan(v_start, vn, order, sep, policy)
            rows = structural_rows(plan)
            first_drop = min(i for i in range(vn) if i not in set(plan["keep_visual_ids"]))
            self.assertEqual(rows, list(range(v_start + first_drop)))
            H, hd, n = 2, 4, len(plan["input_index"])
            red = [(torch.randn(1, H, n, hd).bfloat16(), torch.randn(1, H, n, hd).bfloat16())]
            ctl = {"rows": rows, "layers": [tuple(t[0, :, rows].transpose(0, 1).clone()
                                                  for t in red[0])]}
            self.assertTrue(structural_identity(red, ctl)["ok"])
            ctl["layers"][0][1][0, 0, 0] = -ctl["layers"][0][1][0, 0, 0] if \
                ctl["layers"][0][1][0, 0, 0] != 0 else 1.0
            out = structural_identity(red, ctl)
            self.assertFalse(out["ok"])
            self.assertEqual(out["mismatches"], [{"layer": 0, "kind": "v", "mismatched_rows": 1}])
        self.assertFalse(bit_equal(torch.zeros(2), -torch.zeros(2)))
        self.assertTrue(bit_equal(torch.zeros(2).half(), torch.zeros(2).half()))

    def test_sentinel_detected_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _stores(root, 129)
            path = root / "orig" / "layer_01" / "k.bin"
            data = bytearray(path.read_bytes())
            row = torch.full((8,), SENTINEL, dtype=torch.float16).numpy().tobytes()
            data[:len(row)] = row
            path.write_bytes(bytes(data))
            self.assertEqual(sentinel_rows(root / "orig"), 1)

    def test_within_tolerance(self):
        a = torch.ones(10, 2, 4)
        self.assertTrue(within(a * (1 + 2 ** -8), a, LAYER0_TOL)[0])
        self.assertFalse(within(a * 1.05, a, LAYER0_TOL)[0])


class CacheSentinel(unittest.TestCase):
    def tearDown(self):
        BIAS.clear()

    def test_visible_rows_only(self):
        P = 12
        layer = SimpleNamespace(keys=torch.zeros(1, 2, P + 3, 4),
                                values=torch.zeros(1, 2, P + 3, 4))
        cache = SimpleNamespace(layers=[layer])
        bias = torch.zeros(1, 1, 1, P)
        bias[..., 5:8] = torch.finfo(torch.float32).min
        BIAS[0] = bias
        layer.keys[0, :, 6] = SENTINEL           # hidden row: allowed
        self.assertFalse(cache_sentinel_check(cache, P)["sentinel_visible"])
        layer.values[0, 1, 9, 2] = SENTINEL      # visible row: forbidden
        with self.assertRaises(AssertionError):
            cache_sentinel_check(cache, P)


class _Tok:
    eos_token_id = 2

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


class Serving(unittest.TestCase):
    def _server(self):
        from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM
        cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=4, max_position_embeddings=256,
                          attn_implementation="eager")
        torch.manual_seed(0)
        model = LlamaForCausalLM(cfg).eval()
        runner = SimpleNamespace(model=model,
                                 processor=SimpleNamespace(tokenizer=_Tok()))
        server = PositionServer(runner, max_new_tokens=6)

        def cache():
            c = DynamicCache()
            g = torch.Generator().manual_seed(1)
            for li in range(2):
                c.update(torch.randn(1, 4, 20, 8, generator=g),
                         torch.randn(1, 4, 20, 8, generator=g), li)
            return c
        return server, model, cache

    def test_override_equal_start_is_base_and_positions(self):
        server, model, cache = self._server()
        suffix = torch.tensor([5, 9, 11, 3])
        BIAS.clear()
        with PositionRecorder(model) as base_rec:
            base = server._decode(cache(), suffix, 20)
        base_rec.verify(20, None)
        server.text_position_start = 20
        with PositionRecorder(model) as same_rec:
            same = server._decode(cache(), suffix, 20)
        self.assertEqual(base[:2], same[:2])
        self.assertEqual(base[2]["generated_token_ids"],
                         same[2]["generated_token_ids"])
        self.assertEqual(same_rec.verify(20, 20)["prefill_first_position"], 20)
        server.text_position_start = 7
        with PositionRecorder(model) as rec:
            shifted = server._decode(cache(), suffix, 20)
        out = rec.verify(20, 7)
        self.assertEqual(out["prefill_first_position"], 7)
        self.assertEqual(out["calls"], shifted[2]["generated_token_count"])
        with self.assertRaises(AssertionError):
            rec.verify(20, None)


class Cleanup(unittest.TestCase):
    def test_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run" / "stores"
            (root / "img" / "ours").mkdir(parents=True)
            outside = Path(tmp) / "other"
            outside.mkdir()
            (root / "link").symlink_to(outside)
            for bad in (root, outside, root / "link", root / ".." / "..",
                        Path(tmp)):
                with self.assertRaises(RuntimeError):
                    guarded_rmtree(bad, root)
            self.assertTrue(outside.exists())
            guarded_rmtree(root / "img" / "ours", root)
            self.assertFalse((root / "img" / "ours").exists())

    def test_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a"
            p.write_bytes(b"x")
            inv = stat_inventory([tmp], tmp)
            self.assertEqual(inv["a"][0], 1)
            os.utime(p, ns=(0, 0))
            self.assertNotEqual(stat_inventory([tmp], tmp)["a"], inv["a"])


class Stats(unittest.TestCase):
    def test_bootstrap(self):
        same = [(3, 3, 5)] * 40
        r = paired_bootstrap(same, resamples=500)
        self.assertEqual((r["mean"], r["ci_low"], r["ci_high"]), (0.0, 0.0, 0.0))
        up = [(4, 3, 5)] * 40
        r = paired_bootstrap(up, resamples=500)
        self.assertAlmostEqual(r["mean"], 0.2)
        self.assertGreater(r["ci_low"], 0)


if __name__ == "__main__":
    unittest.main()

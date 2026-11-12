"""CPU tests for the Qwen2.5-VL layer-wise packed PrefixStore (no model, no GPU)."""
import hashlib
import json
import math
import os
import shutil
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import torch

from mmimpress import layer_budget as LB
from mmimpress.qwen25 import layerwise as LW
from mmimpress.qwen25 import packed_store as PS
from mmimpress.qwen25.store import open_qwen_store, write_qwen_store

L, H, D = 28, 4, 128
SHA = {"U": "1343c3ea849a9e70c965cea28530125c1072d841fc23cfd019f4282e11c7fc76",
       "PYR-b20": "58dcb86e2f8b1de4ead4fe1672f36516c6472b835f545f45da5e7b4008b7a52b",
       "PYR-b2p5": "5e5c3ef263fbf691e2678ea1deefd81fc6f1516147f679611e01fb0f75d4f6e0"}
KW = dict(num_layers=28, total_ratio=0.25, r_min=0.0, r_max=0.60)


def pyramid(beta):
    lo = 0.25 / beta
    hi = 0.5 - lo
    step = (hi - lo) / 27
    return [hi - i * step for i in range(28)]


def vectors():
    return {"U": [0.25] * 28, "PYR-b20": pyramid(20), "PYR-b2p5": pyramid(2.5)}


def docs():
    return {name: LB.budget_document(name, r, model="Qwen/Qwen2.5-VL-7B-Instruct",
                                     source={"kind": "test"}, **KW)
            for name, r in vectors().items()}


def hamilton(ratios, n, layers=28, r_max="0.6"):
    """Independent largest-remainder apportionment (floor, then +1 by remainder)."""
    budget = layers * ((n + 3) // 4)
    cap = min(n, math.ceil(Fraction(r_max) * n))
    counts = [min(cap, math.floor(float(r) * n)) for r in ratios]
    rema = [float(r) * n - math.floor(float(r) * n) for r in ratios]
    deficit = budget - sum(counts)
    order = sorted(range(layers), key=lambda i: (-rema[i], i))
    while deficit > 0:
        for i in order:
            if deficit == 0:
                break
            if counts[i] < cap:
                counts[i] += 1
                deficit -= 1
    return counts


def synthetic(n=260, pre=3, post=1, seed=7):
    g = torch.Generator().manual_seed(seed)
    prefix = pre + n + post
    layers = [(torch.randn((1, H, prefix + 2, D), generator=g).bfloat16(),
               torch.randn((1, H, prefix + 2, D), generator=g).bfloat16()) for _ in range(L)]
    scores = torch.randint(0, 40, (n,), generator=g).float() / 7.0   # many ties
    extra = {
        "image_sha256": "ab" * 32, "checkpoint_revision": "rev", "processor_revision": "rev",
        "processor_settings": {"min_pixels": 1, "max_pixels": 2},
        "image_grid_thw": [[1, 2, 2]], "geometry": {"visual_count": n},
        "position_policy": "policy", "code_revision": "core", "environment_revision": "env",
        "key_rope_state": "post_mrope",
        "logical_position_ids": [list(range(prefix)) for _ in range(3)],
        "score_source": "last_fullatt_ViT_received_attention", "score_layer": 31,
        "packed_code_revision": LW.packed_code_revision(),
    }
    return SimpleNamespace(layers=layers, scores=scores, extra=extra, n=n, pre=pre,
                           post=post, prefix=prefix, prefix_ids=list(range(1000, 1000 + prefix)))


def write_packed(tmp, cap, doc, name="packed"):
    return PS.write_qwen_packed_store(Path(tmp) / name, cap.layers, cap.pre, cap.n,
                                      cap.prefix_ids, cap.scores, cap.extra, layer_budget=doc)


class VectorTests(unittest.TestCase):
    def test_vector_sha_sum_and_formula(self):
        for name, doc in docs().items():
            self.assertEqual(doc["layer_budget_sha256"], SHA[name])
            self.assertAlmostEqual(sum(doc["layer_keep_ratios"]), 7.0, places=9)
            self.assertEqual((doc["num_layers"], doc["r_min"], doc["r_max"]), (28, 0.0, 0.60))
        v = vectors()
        self.assertAlmostEqual(v["PYR-b20"][0], 0.4875)
        self.assertAlmostEqual(v["PYR-b20"][-1], 0.0125)
        self.assertAlmostEqual(v["PYR-b2p5"][0], 0.40)
        self.assertAlmostEqual(v["PYR-b2p5"][-1], 0.10)

    def test_expected_k_tables(self):
        expected = {260: {"PYR-b20": (127, 3), "PYR-b2p5": (104, 26)},
                    345: {"PYR-b20": (169, 5), "PYR-b2p5": (139, 35)},
                    529: {"PYR-b20": (259, 7), "PYR-b2p5": (212, 54)}}
        for n, table in expected.items():
            for name, (first, last) in table.items():
                counts = LB.layer_keep_counts(vectors()[name], n, **KW)
                self.assertEqual((counts[0], counts[-1]), (first, last))
                self.assertEqual(counts, hamilton(vectors()[name], n))

    def test_row_budget_exact(self):
        for n in (1, 3, 4, 5, 63, 64, 65, 260, 345, 529, 1024):
            for name, r in vectors().items():
                counts = LB.layer_keep_counts(r, n, **KW)
                self.assertEqual(sum(counts), 28 * math.ceil(n / 4), (name, n))
                self.assertEqual(counts, hamilton(r, n), (name, n))

    def test_uniform_is_ceil_quarter_everywhere(self):
        u = vectors()["U"]
        for n in range(1, 2001):
            self.assertEqual(LB.layer_keep_counts(u, n, **KW), [(n + 3) // 4] * 28)

    def test_llava_defaults_reject_qwen_vectors(self):
        with self.assertRaises(ValueError):
            LB.validate_ratios(vectors()["PYR-b20"])                     # 32 layers expected
        with self.assertRaises(ValueError):
            LB.validate_ratios(vectors()["PYR-b20"], num_layers=28, total_ratio=0.25)  # floor 0.10


class PackedStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="qwen_packed_test_")
        self.cap = synthetic()
        self.docs = docs()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_uniform_packed_is_bitwise_the_unpacked_prefix25_store(self):
        cap = self.cap
        meta = write_packed(self.tmp, cap, self.docs["U"], "U")
        ref_dir = Path(self.tmp) / "B"
        ref = write_qwen_store(ref_dir, cap.layers, cap.pre, cap.n, cap.prefix_ids,
                               scores=cap.scores, extra=cap.extra, storage_policy="prefix25")
        blob = (Path(self.tmp) / "U" / PS.PACKED_NAME).read_bytes()
        s = meta["packed_structural_bytes"]
        self.assertEqual(blob[:s], (ref_dir / "structural_kv.bin").read_bytes())
        rb = meta["row_bytes"]
        for row in meta["packed_layout"]:
            li, k = row["layer"], row["rows"]
            self.assertEqual(blob[row["k_offset"]:row["k_offset"] + k * rb],
                             (ref_dir / f"layer_{li:03d}/k.bin").read_bytes())
            self.assertEqual(blob[row["v_offset"]:row["v_offset"] + k * rb],
                             (ref_dir / f"layer_{li:03d}/v.bin").read_bytes())
        self.assertEqual(len(blob), ref["bytes_visual_kv"] + ref["bytes_structural_kv"])
        self.assertEqual(meta["full_importance_permutation"], ref["full_importance_permutation"])
        with PS.open_qwen_packed_store(Path(self.tmp) / "U", expected_budget=self.docs["U"], pin=False) as store, \
                open_qwen_store(ref_dir) as old:
            store.read_all()
            new = store.assemble_cpu()
            loaded = old.load_prefix(0.25, budget_unit="visual_kv")
            self.assertEqual(len(new), 28)
            for (k1, v1), (k2, v2) in zip(new, loaded.layers):
                self.assertTrue(torch.equal(k1, k2) and torch.equal(v1, v2))
                self.assertEqual(k1.dtype, torch.bfloat16)
            self.assertEqual(store.layer_logical_indices(0), list(loaded.logical_indices))

    def test_layerwise_rows_match_independent_canonical_gather(self):
        cap = self.cap
        score = cap.scores.tolist()
        rank = sorted(range(cap.n), key=lambda i: (-score[i], i))
        structural = [i for i in range(cap.prefix) if not cap.pre <= i < cap.pre + cap.n]
        for name in ("PYR-b20", "PYR-b2p5"):
            meta = write_packed(self.tmp, cap, self.docs[name], name)
            counts = hamilton(vectors()[name], cap.n)
            self.assertEqual(meta["layer_keep_counts"], counts)
            self.assertEqual(meta["full_importance_permutation"], rank)
            self.assertEqual(meta["packed_total_bytes"],
                             2 * 28 * len(structural) * H * D * 2 + sum(2 * k * H * D * 2 for k in counts))
            with PS.open_qwen_packed_store(Path(self.tmp) / name, expected_budget=self.docs[name], pin=False) as store:
                self.assertEqual(store.read_all(), meta["packed_total_bytes"])
                plan = store.plan()
                layers = store.assemble_cpu(plan)
                self.assertEqual(plan.lengths, [len(structural) + k for k in counts])
                for li, (k, v) in enumerate(layers):
                    logical = sorted(structural + [cap.pre + i for i in rank[:counts[li]]])
                    ix = torch.tensor(logical)
                    self.assertTrue(torch.equal(k, cap.layers[li][0].index_select(2, ix)))
                    self.assertTrue(torch.equal(v, cap.layers[li][1].index_select(2, ix)))
                    self.assertEqual(store.layer_logical_indices(li), logical)

    def test_one_preadv_and_no_pread_on_the_hit_path(self):
        write_packed(self.tmp, self.cap, self.docs["PYR-b20"], "p")
        store = PS.open_qwen_packed_store(Path(self.tmp) / "p", pin=False)
        calls = {"pread": 0, "preadv": 0, "bytes": 0}
        pread, preadv = os.pread, os.preadv

        def fake_pread(*a):
            calls["pread"] += 1
            return pread(*a)

        def fake_preadv(fd, buffers, offset, *flags):
            calls["preadv"] += 1
            n = preadv(fd, buffers, offset, *flags)
            calls["bytes"] += n
            return n
        os.pread, os.preadv = fake_pread, fake_preadv
        try:
            store.plan()
            store.read_all()
            store.assemble_cpu()
        finally:
            os.pread, os.preadv = pread, preadv
            store.close()
        self.assertEqual(calls, {"pread": 0, "preadv": 1,
                                 "bytes": (Path(self.tmp) / "p" / PS.PACKED_NAME).stat().st_size})

    def _tamper(self, name, mutate):
        src = Path(self.tmp) / name
        dst = Path(self.tmp) / f"{name}_tampered_{len(list(Path(self.tmp).iterdir()))}"
        shutil.copytree(src, dst)
        meta = json.loads((dst / PS.META_NAME).read_text())
        mutate(meta, dst)
        meta["bytes_metadata_file"] = 0
        for _ in range(10):
            enc = (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode()
            if len(enc) == meta["bytes_metadata_file"]:
                break
            meta["bytes_metadata_file"] = len(enc)
        (dst / PS.META_NAME).write_bytes(enc)
        return dst

    def test_validator_rejects_tampered_metadata_and_payload(self):
        write_packed(self.tmp, self.cap, self.docs["PYR-b20"], "p")
        base = Path(self.tmp) / "p"
        PS.open_qwen_packed_store(base, expected_budget=self.docs["PYR-b20"], pin=False).close()

        def count_plus(m, d):
            m["layer_keep_counts"][5] += 1

        def count_minus(m, d):
            m["layer_keep_counts"][5] -= 1

        def rmin(m, d):
            m["layer_budget_r_min"] = 0.10

        def layers32(m, d):
            m["num_layers"] = 32

        def sha(m, d):
            m["layer_budget_sha256"] = SHA["PYR-b2p5"]

        def ratios(m, d):
            m["layer_keep_ratios"] = vectors()["PYR-b2p5"]
            m["layer_budget_sha256"] = SHA["PYR-b2p5"]

        def total(m, d):
            m["packed_total_bytes"] += 2

        def offset(m, d):
            m["packed_layout"][3]["k_offset"] += 1024

        def swap_layers(m, d):
            m["packed_layout"][3], m["packed_layout"][4] = m["packed_layout"][4], m["packed_layout"][3]

        def order(m, d):
            p = m["full_importance_permutation"]
            p[0], p[1] = p[1], p[0]

        def structural(m, d):
            m["structural_indices"] = m["structural_indices"][1:]
            m["structural_count"] -= 1

        def payload_sha(m, d):
            m["packed_sha256"] = "0" * 64

        def fmt(m, d):
            m["store_format"] = "qwen25_bf16_importance_prefix25_v2"

        for mutate in (count_plus, count_minus, rmin, layers32, sha, ratios, total, offset,
                       swap_layers, order, structural, payload_sha, fmt):
            dst = self._tamper("p", mutate)
            with self.assertRaises((ValueError, KeyError, AssertionError), msg=mutate.__name__):
                PS.open_qwen_packed_store(dst, expected_budget=self.docs["PYR-b20"], pin=False)
        # r_min alone is rejected even without an expected budget document.
        with self.assertRaises(ValueError):
            PS.open_qwen_packed_store(self._tamper("p", rmin), pin=False)
        # truncated file, flipped payload bit, two layers' rows swapped on disk
        for kind in ("truncate", "flip", "swap_rows"):
            dst = Path(self.tmp) / f"payload_{kind}"
            shutil.copytree(base, dst)
            path = dst / PS.PACKED_NAME
            meta = json.loads((dst / PS.META_NAME).read_text())
            with path.open("r+b") as f:
                if kind == "truncate":
                    f.truncate(path.stat().st_size - 2)
                elif kind == "flip":
                    f.seek(meta["packed_layout"][9]["k_offset"])
                    b = f.read(1)
                    f.seek(meta["packed_layout"][9]["k_offset"])
                    f.write(bytes([b[0] ^ 1]))
                else:
                    a, b = meta["packed_layout"][26], meta["packed_layout"][27]
                    size = min(a["rows"], b["rows"]) * meta["row_bytes"]
                    f.seek(a["k_offset"]); xa = f.read(size)
                    f.seek(b["k_offset"]); xb = f.read(size)
                    f.seek(a["k_offset"]); f.write(xb)
                    f.seek(b["k_offset"]); f.write(xa)
            with self.assertRaises(ValueError, msg=kind):
                PS.open_qwen_packed_store(dst, expected_budget=self.docs["PYR-b20"], pin=False)
        # a store built with another vector is rejected by the expected budget
        with self.assertRaises(ValueError):
            PS.open_qwen_packed_store(base, expected_budget=self.docs["PYR-b2p5"], pin=False)
        with self.assertRaises(ValueError):
            PS.open_qwen_packed_store(base, expected_identity={"image_sha256": "cd" * 32}, pin=False)

    def test_writer_rejects_bad_budget_documents(self):
        cap = self.cap
        good = self.docs["PYR-b20"]
        for mutate in (lambda d: d.pop("r_min"), lambda d: d.pop("num_layers"),
                       lambda d: d.__setitem__("r_min", 0.10),
                       lambda d: d.__setitem__("num_layers", 32),
                       lambda d: d.__setitem__("layer_budget_sha256", SHA["U"]),
                       lambda d: d.__setitem__("total_ratio", 0.5)):
            doc = dict(good)
            mutate(doc)
            with self.assertRaises(ValueError):
                write_packed(self.tmp, cap, doc, f"bad{len(list(Path(self.tmp).iterdir()))}")
        tiny = synthetic(n=3)
        with self.assertRaises(ValueError):       # beta=20 would leave empty layers at N=3
            write_packed(self.tmp, tiny, good, "tiny")
        write_packed(self.tmp, cap, good, "once")
        with self.assertRaises(FileExistsError):
            write_packed(self.tmp, cap, good, "once")

    def test_no_reliance_on_llava_layer_budget_defaults(self):
        class Poison:
            def _boom(self, *a, **k):
                raise RuntimeError("layer_budget default argument was used")
            __int__ = __float__ = __index__ = __eq__ = __lt__ = __le__ = __gt__ = __ge__ = _boom
            __mul__ = __rmul__ = __add__ = __radd__ = __sub__ = __rsub__ = __hash__ = _boom
        fns = [LB.validate_ratios, LB.total_row_budget, LB.keep_bounds, LB.layer_keep_counts,
               LB.validate_keep_counts, LB.budget_document, LB.uniform_ratios]
        saved = [(f, f.__defaults__, f.__kwdefaults__) for f in fns]
        constants = (LB.NUM_LAYERS, LB.R_MIN)
        try:
            model_specific = {"num_layers", "total_ratio", "r_min", "r_max"}
            for f in fns:
                names = f.__code__.co_varnames[:f.__code__.co_argcount]
                if f.__defaults__:
                    tail = names[len(names) - len(f.__defaults__):]
                    f.__defaults__ = tuple(Poison() if n in model_specific else d
                                           for n, d in zip(tail, f.__defaults__))
                if f.__kwdefaults__:
                    f.__kwdefaults__ = {k: (Poison() if k in model_specific else v)
                                        for k, v in f.__kwdefaults__.items()}
            LB.NUM_LAYERS, LB.R_MIN = Poison(), Poison()
            with self.assertRaises(Exception):
                LB.layer_keep_counts(vectors()["U"], 260)          # the poison is live
            doc = LB.budget_document("PYR-b20", pyramid(20), model="q", source={}, **KW)
            self.assertEqual(doc["layer_budget_sha256"], SHA["PYR-b20"])
            write_packed(self.tmp, self.cap, doc, "explicit")
            with PS.open_qwen_packed_store(Path(self.tmp) / "explicit", expected_budget=doc, pin=False) as store:
                store.read_all()
                self.assertEqual(len(store.assemble_cpu()), 28)
                self.assertEqual(store.meta["layer_keep_counts"], hamilton(pyramid(20), self.cap.n))
        finally:
            for f, d, kd in saved:
                f.__defaults__, f.__kwdefaults__ = d, kd
            LB.NUM_LAYERS, LB.R_MIN = constants
        self.assertEqual((LB.NUM_LAYERS, LB.R_MIN), (32, 0.10))


class MaskHookTests(unittest.TestCase):
    def test_layer_mask_keeps_causal_block_and_fixes_prefix(self):
        q, p0 = 4, 9
        causal = torch.tensor([[j <= i for j in range(q)] for i in range(q)])
        hf = torch.cat((torch.ones(q, p0, dtype=torch.bool), causal), 1)[None, None]
        for need_prefix in (9, 5, 0, 12):
            out = LW.layer_mask(hf, need_prefix + q)
            self.assertEqual(tuple(out.shape), (1, 1, q, need_prefix + q))
            self.assertTrue(bool(out[..., :need_prefix].all()))
            self.assertTrue(torch.equal(out[0, 0, :, need_prefix:], causal))
        self.assertIs(LW.layer_mask(hf, p0 + q), hf)
        with self.assertRaises(AssertionError):
            LW.layer_mask(hf, q - 1)
        additive = torch.zeros((1, 1, q, p0 + q)).masked_fill(~hf, float("-inf"))
        grown = LW.layer_mask(additive, p0 + q + 3)
        self.assertTrue(bool((grown[..., :3] == 0).all()))

    def _fake(self, lengths):
        class Attn(torch.nn.Module):
            def __init__(self, idx):
                super().__init__()
                self.layer_idx = idx
                self.seen = []

            def forward(self, hidden_states=None, attention_mask=None, past_key_values=None):
                self.seen.append(attention_mask)
                past_key_values.lengths[self.layer_idx] += hidden_states.shape[1]
                return hidden_states

        class Cache:
            def __init__(self, lens):
                self.lengths = list(lens)

            def get_seq_length(self, li=0):
                return self.lengths[li]
        layers = [SimpleNamespace(self_attn=Attn(i)) for i in range(len(lengths))]
        model = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)))
        return model, layers, Cache(lengths)

    def test_hooks_realign_per_layer_pass_none_and_are_removed(self):
        lengths = [9, 7, 9, 3]
        model, layers, cache = self._fake(lengths)
        q = 4
        causal = torch.tensor([[j <= i for j in range(q)] for i in range(q)])
        hf = torch.cat((torch.ones(q, lengths[0], dtype=torch.bool), causal), 1)[None, None]
        with LW.LayerMaskHooks(model, lengths) as hooks:
            self.assertEqual(LW.residual_layer_hooks(model), 4)
            for layer in layers:
                layer.self_attn(hidden_states=torch.zeros(1, q, 2), attention_mask=hf, past_key_values=cache)
            for layer in layers:
                layer.self_attn(hidden_states=torch.zeros(1, 1, 2), attention_mask=None, past_key_values=cache)
        self.assertEqual(LW.residual_layer_hooks(model), 0)
        LW.assert_no_layer_hooks(model)
        for li, layer in enumerate(layers):
            seen = layer.self_attn.seen
            self.assertEqual(tuple(seen[0].shape), (1, 1, q, lengths[li] + q))
            self.assertTrue(bool(seen[0][..., :lengths[li]].all()))
            self.assertTrue(torch.equal(seen[0][0, 0, :, lengths[li]:], causal))
            self.assertIsNone(seen[1])
        self.assertIs(layers[0].self_attn.seen[0], hf)
        self.assertEqual(hooks.calls, [2] * 4)
        self.assertEqual(hooks.realigned, [0, 1, 0, 1])
        self.assertEqual(hooks.appended, [q + 1] * 4)

    def test_stale_hook_or_foreign_cache_is_detected(self):
        model, layers, cache = self._fake([9, 7, 9, 3])
        stale = LW.LayerMaskHooks(model, [9, 9, 9, 9]).__enter__()   # lengths of another request
        with self.assertRaises(AssertionError):
            LW.assert_no_layer_hooks(model)
        with self.assertRaises(AssertionError):
            LW.LayerMaskHooks(model, [9, 7, 9, 3]).__enter__()       # refuses to stack
        layers[0].self_attn(hidden_states=torch.zeros(1, 2, 2), attention_mask=None, past_key_values=cache)
        with self.assertRaises(AssertionError):
            layers[1].self_attn(hidden_states=torch.zeros(1, 2, 2), attention_mask=None, past_key_values=cache)
        stale.remove()
        LW.assert_no_layer_hooks(model)


if __name__ == "__main__":
    unittest.main()

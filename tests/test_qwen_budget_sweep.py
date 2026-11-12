"""CPU tests for the Qwen2.5-VL retention-ratio sweep packed store (no model, no GPU)."""
import inspect
import json
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from mmimpress import layer_budget as LB
from mmimpress.qwen25 import layerwise as LW
from mmimpress.qwen25 import packed_store as PS
from mmimpress.qwen25 import ratio_packed as RP
from mmimpress.qwen25.store import open_qwen_store, write_qwen_store

L, H, D = 28, 4, 128
RATIO = {"U25": 0.25, "U30": 0.30, "U40": 0.40, "U50": 0.50, "U100": 1.00}
FRACTION = {"U25": (1, 4), "U30": (3, 10), "U40": (2, 5), "U50": (1, 2), "U100": (1, 1)}
SHA = {"U25": "1343c3ea849a9e70c965cea28530125c1072d841fc23cfd019f4282e11c7fc76",
       "U30": "b726c756cd0680a018200d934536542ea2a31effbe086535577e6f895984a0e7",
       "U40": "d14e8bb77c002045dad35ce55ba2b4b31301af1fbde74383244139a6b949e464",
       "U50": "40302d54a7eba3a7ef3fcc11e150ddef996c4e96a55e8df52af97cb1336d750b"}
EXPECTED_K = {260: (65, 78, 104, 130, 260), 345: (87, 104, 138, 173, 345), 529: (133, 159, 212, 265, 529)}


def kw(name):
    return dict(num_layers=28, total_ratio=RATIO[name], r_min=0.0, r_max=1.0)


def int_k(name, n):
    p, q = FRACTION[name]
    return -((-p * n) // q)


def docs():
    return {name: LB.budget_document(name, [r] * 28, model="Qwen/Qwen2.5-VL-7B-Instruct",
                                     source={"kind": "test"}, **kw(name))
            for name, r in RATIO.items()}


def pyr_run_uniform_doc():
    """The U document exactly as the PyramidKV run froze it (r_max 0.60)."""
    return LB.budget_document("U", [0.25] * 28, model="Qwen/Qwen2.5-VL-7B-Instruct", source={"kind": "test"},
                              num_layers=28, total_ratio=0.25, r_min=0.0, r_max=0.60)


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
        "packed_code_revision": RP.ratio_code_revision(),
    }
    return SimpleNamespace(layers=layers, scores=scores, extra=extra, n=n, pre=pre,
                           post=post, prefix=prefix, prefix_ids=list(range(1000, 1000 + prefix)))


def write_ratio(tmp, cap, doc, name):
    return RP.write_ratio_packed_store(Path(tmp) / name, cap.layers, cap.pre, cap.n,
                                       cap.prefix_ids, cap.scores, cap.extra, layer_budget=doc)


class VectorTests(unittest.TestCase):
    def test_vector_sha_sum_and_explicit_arguments(self):
        for name, doc in docs().items():
            if name in SHA:
                self.assertEqual(doc["layer_budget_sha256"], SHA[name])
            self.assertEqual(doc["layer_budget_sha256"], LB.budget_sha256([RATIO[name]] * 28))
            self.assertAlmostEqual(sum(doc["layer_keep_ratios"]), 28 * RATIO[name], places=9)
            self.assertEqual((doc["num_layers"], doc["total_ratio"], doc["r_min"], doc["r_max"]), (28, RATIO[name], 0.0, 1.0))
            self.assertEqual(RP.budget_fields(doc, 28), ([RATIO[name]] * 28, 0.0, 1.0, RATIO[name]))
        self.assertEqual(docs()["U25"]["layer_budget_sha256"], pyr_run_uniform_doc()["layer_budget_sha256"])

    def test_expected_k_tables(self):
        for n, expected in EXPECTED_K.items():
            for name, k in zip(RATIO, expected):
                self.assertEqual(LB.layer_keep_counts([RATIO[name]] * 28, n, **kw(name)), [k] * 28, (name, n))
                self.assertEqual(RP.uniform_keep_count(RATIO[name], n), k)
                self.assertEqual(int_k(name, n), k)

    def test_k_equals_integer_fraction_for_n_1_to_3000(self):
        for name, r in RATIO.items():
            for n in range(1, 3001):
                k = int_k(name, n)
                counts = LB.layer_keep_counts([r] * 28, n, **kw(name))
                self.assertEqual(counts, [k] * 28, (name, n))
                self.assertEqual(sum(counts), 28 * k)
                self.assertEqual(math.ceil(r * n), k, (name, n))
                self.assertEqual(RP.uniform_keep_count(r, n), k)
                self.assertEqual(RP.ratio_keep_counts([r] * 28, n, 28, 0.0, 1.0, r), [k] * 28)
            self.assertEqual(RP.ratio_fraction(r).as_integer_ratio(), FRACTION[name])

    def test_selection_is_nested_across_ratios(self):
        cap = synthetic(n=345)
        score = cap.scores.tolist()
        rank = sorted(range(cap.n), key=lambda i: (-score[i], i))
        previous = set()
        for name in RATIO:
            current = set(rank[:int_k(name, cap.n)])
            self.assertTrue(previous <= current and len(current) == int_k(name, cap.n))
            previous = current
        self.assertEqual(previous, set(range(cap.n)))

    def test_llava_defaults_reject_qwen_vectors(self):
        with self.assertRaises(ValueError):
            LB.validate_ratios([0.30] * 28)                                    # 32 layers / total 0.25 expected
        with self.assertRaises(ValueError):
            LB.validate_ratios([1.0] * 28, num_layers=28, total_ratio=1.0)     # default r_max 0.60

    def test_bad_ratios_are_rejected(self):
        for bad in (0.0, -0.1, 1.5, True, "0.3", float("nan")):
            with self.assertRaises(ValueError):
                RP.ratio_fraction(bad)
        with self.assertRaises(ValueError):
            RP.uniform_keep_count(0.3, 0)


class RatioStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="qwen_sweep_test_")
        self.cap = synthetic()
        self.docs = docs()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _structural(self, cap):
        return [i for i in range(cap.prefix) if not cap.pre <= i < cap.pre + cap.n]

    def _rank(self, cap):
        score = cap.scores.tolist()
        return sorted(range(cap.n), key=lambda i: (-score[i], i))

    def test_u25_payload_is_bitwise_the_pyramidkv_run_uniform_packed_store(self):
        cap = self.cap
        new = write_ratio(self.tmp, cap, self.docs["U25"], "U25")
        old_extra = dict(cap.extra, packed_code_revision=LW.packed_code_revision())
        old = PS.write_qwen_packed_store(Path(self.tmp) / "U_pyr", cap.layers, cap.pre, cap.n, cap.prefix_ids,
                                         cap.scores, old_extra, layer_budget=pyr_run_uniform_doc())
        a = (Path(self.tmp) / "U25" / RP.PACKED_NAME).read_bytes()
        b = (Path(self.tmp) / "U_pyr" / PS.PACKED_NAME).read_bytes()
        self.assertEqual(a, b)
        self.assertEqual(new["packed_sha256"], old["packed_sha256"])
        self.assertEqual(new["packed_layout"], old["packed_layout"])
        self.assertEqual(new["full_importance_permutation"], old["full_importance_permutation"])
        differing = sorted(k for k in set(new) | set(old) if new.get(k) != old.get(k))
        self.assertEqual(differing, sorted([
            "format", "store_format", "storage_policy", "layout_description", "order_semantics", "rounding_policy",
            "layer_budget_id", "layer_budget_r_max", "layer_budget_binding_sha256", "packed_code_revision",
            "uniform_content_count", "uniform_equivalent_content_count", "retention_fraction", "bytes_metadata_file",
            "extra"]))                                    # "extra" carries packed_code_revision
        with RP.open_ratio_packed_store(Path(self.tmp) / "U25", expected_budget=self.docs["U25"], pin=False) as s1, \
                PS.open_qwen_packed_store(Path(self.tmp) / "U_pyr", expected_budget=pyr_run_uniform_doc(), pin=False) as s2:
            s1.read_all()
            s2.read_all()
            p1, p2 = s1.plan(), s2.plan()
            self.assertTrue(torch.equal(p1.row_index, p2.row_index))
            self.assertEqual((p1.spans, p1.lengths, p1.counts), (p2.spans, p2.lengths, p2.counts))
            for (k1, v1), (k2, v2) in zip(s1.assemble_cpu(p1), s2.assemble_cpu(p2)):
                self.assertTrue(torch.equal(k1, k2) and torch.equal(v1, v2))

    def test_u100_is_the_full_prefix_and_equals_the_fullload_store(self):
        cap = self.cap
        meta = write_ratio(self.tmp, cap, self.docs["U100"], "U100")
        self.assertEqual(meta["layer_keep_counts"], [cap.n] * 28)
        full_extra = dict(cap.extra, score_source=None, score_layer=None)
        ref_dir = Path(self.tmp) / "fullload"
        write_qwen_store(ref_dir, cap.layers, cap.pre, cap.n, cap.prefix_ids, scores=None, extra=full_extra, storage_policy="full")
        with RP.open_ratio_packed_store(Path(self.tmp) / "U100", expected_budget=self.docs["U100"], pin=False) as store, \
                open_qwen_store(ref_dir) as old:
            store.read_all()
            layers = store.assemble_cpu()
            loaded = old.load_prefix(1.0, budget_unit="chunk")
            self.assertEqual(list(loaded.logical_indices), list(range(cap.prefix)))
            self.assertEqual(store.layer_logical_indices(0), list(range(cap.prefix)))
            for li, ((k1, v1), (k2, v2)) in enumerate(zip(layers, loaded.layers)):
                self.assertTrue(torch.equal(k1, k2) and torch.equal(v1, v2), li)
                self.assertTrue(torch.equal(k1, cap.layers[li][0][:, :, :cap.prefix]))
                self.assertTrue(torch.equal(v1, cap.layers[li][1][:, :, :cap.prefix]))

    def test_rows_match_independent_canonical_gather_for_every_ratio(self):
        cap = self.cap
        rank, structural = self._rank(cap), self._structural(cap)
        row = H * D * 2
        for name in RATIO:
            k = int_k(name, cap.n)
            meta = write_ratio(self.tmp, cap, self.docs[name], name)
            self.assertEqual(meta["layer_keep_counts"], [k] * 28)
            self.assertEqual(meta["full_importance_permutation"], rank)
            self.assertEqual(meta["packed_total_bytes"], 2 * 28 * len(structural) * row + 28 * 2 * k * row)
            self.assertEqual((Path(self.tmp) / name / RP.PACKED_NAME).stat().st_size, meta["packed_total_bytes"])
            with RP.open_ratio_packed_store(Path(self.tmp) / name, expected_budget=self.docs[name], pin=False) as store:
                self.assertEqual(store.read_all(), meta["packed_total_bytes"])
                plan = store.plan()
                self.assertEqual(plan.lengths, [len(structural) + k] * 28)
                logical = sorted(structural + [cap.pre + i for i in rank[:k]])
                ix = torch.tensor(logical)
                for li, (kk, vv) in enumerate(store.assemble_cpu(plan)):
                    self.assertTrue(torch.equal(kk, cap.layers[li][0].index_select(2, ix)))
                    self.assertTrue(torch.equal(vv, cap.layers[li][1].index_select(2, ix)))
                    self.assertEqual(store.layer_logical_indices(li), logical)

    def test_one_preadv_and_no_pread_on_the_hit_path(self):
        write_ratio(self.tmp, self.cap, self.docs["U40"], "p")
        store = RP.open_ratio_packed_store(Path(self.tmp) / "p", pin=False)
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
                                 "bytes": (Path(self.tmp) / "p" / RP.PACKED_NAME).stat().st_size})

    def _tamper(self, name, mutate):
        src = Path(self.tmp) / name
        dst = Path(self.tmp) / f"{name}_tampered_{len(list(Path(self.tmp).iterdir()))}"
        shutil.copytree(src, dst)
        meta = json.loads((dst / RP.META_NAME).read_text())
        mutate(meta)
        meta["bytes_metadata_file"] = 0
        for _ in range(10):
            enc = (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode()
            if len(enc) == meta["bytes_metadata_file"]:
                break
            meta["bytes_metadata_file"] = len(enc)
        (dst / RP.META_NAME).write_bytes(enc)
        return dst

    def test_validator_rejects_other_ratio_tampered_metadata_and_payload(self):
        write_ratio(self.tmp, self.cap, self.docs["U30"], "p")
        base = Path(self.tmp) / "p"
        RP.open_ratio_packed_store(base, expected_budget=self.docs["U30"], pin=False).close()
        u40 = write_ratio(self.tmp, self.cap, self.docs["U40"], "q")

        def set_(key, value):
            return lambda m: m.__setitem__(key, value)

        def count_plus(m):
            m["layer_keep_counts"][5] += 1

        def all_counts_plus(m):
            m["layer_keep_counts"] = [c + 1 for c in m["layer_keep_counts"]]
            m["uniform_content_count"] += 1
            m["stored_content_count_total"] += 28

        def other_ratio_meta(m):          # the metadata of the 40% store on top of the 30% payload
            m.clear()
            m.update(json.loads(json.dumps(u40)))

        def non_uniform(m):
            m["layer_keep_ratios"][0], m["layer_keep_ratios"][1] = 0.35, 0.25
            m["layer_budget_sha256"] = LB.budget_sha256(m["layer_keep_ratios"])

        def offset(m):
            m["packed_layout"][3]["k_offset"] += 1024

        def swap_layers(m):
            m["packed_layout"][3], m["packed_layout"][4] = m["packed_layout"][4], m["packed_layout"][3]

        def order(m):
            p = m["full_importance_permutation"]
            p[0], p[1] = p[1], p[0]

        def structural(m):
            m["structural_indices"] = m["structural_indices"][1:]
            m["structural_count"] -= 1

        mutations = {
            "total_ratio_0p25_only": set_("layer_budget_total_ratio", 0.25),
            "retention_ratio_0p25_only": set_("retention_ratio", 0.25),
            "retention_fraction": set_("retention_fraction", [1, 4]),
            "row_count_one_layer": count_plus, "row_count_all_layers": all_counts_plus,
            "uniform_content_count": set_("uniform_content_count", 79),
            "r_min_0p10": set_("layer_budget_r_min", 0.10), "r_max_0p60": set_("layer_budget_r_max", 0.60),
            "num_layers_32": set_("num_layers", 32), "vector_sha_of_U40": set_("layer_budget_sha256", SHA["U40"]),
            "other_ratio_meta": other_ratio_meta, "non_uniform": non_uniform,
            "total_bytes": set_("packed_total_bytes", 1), "offset": offset, "swap_layers": swap_layers,
            "order": order, "structural": structural, "payload_sha": set_("packed_sha256", "0" * 64),
            "old_packed_format": set_("store_format", PS.STORE_FORMAT), "storage_policy": set_("storage_policy", PS.STORAGE_POLICY),
        }
        for name, mutate in mutations.items():
            dst = self._tamper("p", mutate)
            with self.assertRaises((ValueError, KeyError, AssertionError), msg=name):
                RP.open_ratio_packed_store(dst, expected_budget=self.docs["U30"], pin=False)
        # the ratio / clip tampering is rejected even without an expected budget document
        for name in ("total_ratio_0p25_only", "retention_ratio_0p25_only", "r_min_0p10", "r_max_0p60", "row_count_one_layer"):
            with self.assertRaises(ValueError, msg=name):
                RP.open_ratio_packed_store(self._tamper("p", mutations[name]), pin=False)
        # truncated file, flipped payload bit
        for kind in ("truncate", "flip"):
            dst = Path(self.tmp) / f"payload_{kind}"
            shutil.copytree(base, dst)
            path = dst / RP.PACKED_NAME
            meta = json.loads((dst / RP.META_NAME).read_text())
            with path.open("r+b") as f:
                if kind == "truncate":
                    f.truncate(path.stat().st_size - 2)
                else:
                    f.seek(meta["packed_layout"][9]["k_offset"])
                    b = f.read(1)
                    f.seek(meta["packed_layout"][9]["k_offset"])
                    f.write(bytes([b[0] ^ 1]))
            with self.assertRaises(ValueError, msg=kind):
                RP.open_ratio_packed_store(dst, expected_budget=self.docs["U30"], pin=False)
        # a store of another ratio is rejected by the expected budget, and a foreign image by identity
        for other in ("U25", "U40", "U50", "U100"):
            with self.assertRaises(ValueError, msg=other):
                RP.open_ratio_packed_store(base, expected_budget=self.docs[other], pin=False)
        with self.assertRaises(ValueError):
            RP.open_ratio_packed_store(base, expected_identity={"image_sha256": "cd" * 32}, pin=False)
        # the two packed formats do not accept each other's stores
        with self.assertRaises(ValueError):
            PS.open_qwen_packed_store(base, pin=False)
        PS.write_qwen_packed_store(Path(self.tmp) / "old", self.cap.layers, self.cap.pre, self.cap.n, self.cap.prefix_ids,
                                   self.cap.scores, self.cap.extra, layer_budget=pyr_run_uniform_doc())
        with self.assertRaises(ValueError):
            RP.open_ratio_packed_store(Path(self.tmp) / "old", pin=False)

    def test_writer_rejects_bad_budget_documents(self):
        cap = self.cap
        good = self.docs["U30"]
        for mutate in (lambda d: d.pop("r_max"), lambda d: d.pop("total_ratio"), lambda d: d.pop("num_layers"),
                       lambda d: d.__setitem__("num_layers", 32),
                       lambda d: d.__setitem__("total_ratio", 0.25),
                       lambda d: d.__setitem__("r_max", 0.25),
                       lambda d: d.__setitem__("layer_budget_sha256", SHA["U25"]),
                       lambda d: d.__setitem__("layer_keep_ratios", [0.35, 0.25] + [0.30] * 26)):
            doc = json.loads(json.dumps(good))
            mutate(doc)
            with self.assertRaises((ValueError, KeyError)):
                write_ratio(self.tmp, cap, doc, f"bad{len(list(Path(self.tmp).iterdir()))}")
        write_ratio(self.tmp, cap, good, "once")
        with self.assertRaises(FileExistsError):
            write_ratio(self.tmp, cap, good, "once")

    def test_no_reliance_on_llava_layer_budget_defaults(self):
        class Poison:
            def _boom(self, *a, **k):
                raise RuntimeError("layer_budget default argument was used")
            __int__ = __float__ = __index__ = __eq__ = __lt__ = __le__ = __gt__ = __ge__ = _boom
            __mul__ = __rmul__ = __add__ = __radd__ = __sub__ = __rsub__ = __hash__ = _boom
        fns = [LB.validate_ratios, LB.total_row_budget, LB.keep_bounds, LB.layer_keep_counts,
               LB.validate_keep_counts, LB.budget_document, LB.uniform_ratios]
        saved = [(f, f.__defaults__, f.__kwdefaults__) for f in fns]
        constants = (LB.NUM_LAYERS, LB.R_MIN, LB.R_MAX, LB.TOTAL_RATIO)
        try:
            model_specific = {"num_layers", "total_ratio", "r_min", "r_max", "ratio"}
            for f in fns:
                names = f.__code__.co_varnames[:f.__code__.co_argcount]
                if f.__defaults__:
                    tail = names[len(names) - len(f.__defaults__):]
                    f.__defaults__ = tuple(Poison() if n in model_specific else d
                                           for n, d in zip(tail, f.__defaults__))
                if f.__kwdefaults__:
                    f.__kwdefaults__ = {k: (Poison() if k in model_specific else v)
                                        for k, v in f.__kwdefaults__.items()}
            LB.NUM_LAYERS, LB.R_MIN, LB.R_MAX, LB.TOTAL_RATIO = Poison(), Poison(), Poison(), Poison()
            with self.assertRaises(Exception):
                LB.layer_keep_counts([0.30] * 28, 260)                 # the poison is live
            doc = LB.budget_document("U30", [0.30] * 28, model="q", source={}, **kw("U30"))
            self.assertEqual(doc["layer_budget_sha256"], SHA["U30"])
            write_ratio(self.tmp, self.cap, doc, "explicit")
            with RP.open_ratio_packed_store(Path(self.tmp) / "explicit", expected_budget=doc, pin=False) as store:
                store.read_all()
                self.assertEqual(len(store.assemble_cpu()), 28)
                self.assertEqual(store.meta["layer_keep_counts"], [78] * 28)
            # the same store relabelled as the 0.25 total ratio is rejected
            with self.assertRaises(ValueError):
                RP.open_ratio_packed_store(self._tamper("explicit", lambda m: m.__setitem__("layer_budget_total_ratio", 0.25)), pin=False)
        finally:
            for f, d, kd in saved:
                f.__defaults__, f.__kwdefaults__ = d, kd
            LB.NUM_LAYERS, LB.R_MIN, LB.R_MAX, LB.TOTAL_RATIO = constants
        self.assertEqual((LB.NUM_LAYERS, LB.R_MIN, LB.R_MAX, LB.TOTAL_RATIO), (32, 0.10, 0.60, 0.25))

    def test_selection_takes_no_question(self):
        """The stored selection is a function of (scores, N, ratio) only."""
        for fn in (RP.write_ratio_packed_store, RP.ratio_keep_counts, RP.uniform_keep_count, PS.QwenPackedStore.plan):
            names = set(inspect.signature(fn).parameters)
            self.assertFalse(names & {"question", "history", "prompt", "text", "query"}, fn.__name__)
        cap = self.cap
        a = write_ratio(self.tmp, cap, self.docs["U40"], "a")
        b = write_ratio(self.tmp, cap, self.docs["U40"], "b")
        self.assertEqual(a["packed_sha256"], b["packed_sha256"])
        self.assertEqual(a["stored_to_original"], self._rank(cap)[:int_k("U40", cap.n)])


if __name__ == "__main__":
    unittest.main()

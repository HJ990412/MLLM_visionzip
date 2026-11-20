"""CPU contracts of the two runs whose IMPRESS arm is the public-implementation path
(docs/mt_gqa_impress_code_contract.md, docs/llava15_mtvqa_impress_code_contract.md).

``PublicEquivalenceTests`` (need IMPRESS_OFFICIAL_REPO, skipped without it) feed
the same tensors to the public repository's functions and to
``mmimpress.impress_official`` in the LLaVA-1.5 geometry: 576 candidates, no
separator, k = 144.  The LLaVA-NeXT geometry is covered by
``tests/test_impress_official.py``.

The remaining tests pin what the LLaVA-1.5 port relies on (a store without
separators through the same layout writer and the same selector) and the arm
tables of the two harnesses (scripts 196 / 199): one IMPRESS implementation,
the same selection module in both, the gate-only arms on the reordered store.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(Path(__file__).resolve().parent), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_impress_official as base  # noqa: E402
from test_impress_official import OFFICIAL, _Compare, _attention, _device, _quiet, _sets  # noqa: E402

import llava15_support as S  # noqa: E402
from mmimpress import impress_official as io_  # noqa: E402
from mmimpress.serve import BIAS  # noqa: E402
from mmimpress.store import load_meta  # noqa: E402

N15 = 576                    # LLaVA-1.5 visual tokens
K15 = 144                    # ceil(0.25 * 576)


def _world_without_separators(td, seed=0):
    """``test_impress_official._World`` on a store that has no separator rows."""
    original = base._store
    with mock.patch.object(base, "_store", lambda root, seed=0, **kw: original(root, seed=seed, newline=(), **kw)):
        return base._World(td, seed=seed)


@unittest.skipUnless(OFFICIAL, "set IMPRESS_OFFICIAL_REPO to the public checkout")
class PublicEquivalenceTests(_Compare, unittest.TestCase):
    def test_llava15_geometry_selection_equals_the_public_function(self):
        """Per-head top-k, mean Jaccard, fallback decision and the selected set
        on a 576-candidate prefix (k = 144)."""
        self.assertEqual(io_.keep_count(N15), K15)
        kinds = set()
        cases = [dict(seed=s, own=6.0) for s in range(3)] + [dict(seed=3)] + [dict(seed=10 + s, agree=3.0) for s in range(3)]
        cases += [dict(seed=30, same=(1,)), dict(seed=40, same=(1, 2))]
        for case in cases:
            attn = _attention(n_prefix=N15, n_suffix=24, **case)
            dev, seen = _device()
            sel, dele = _quiet(dev._sele_important_tokens, attn, N15, -1, io_.SIM_THRESHOLD, 25)
            self.assertEqual(seen["head_topk"].shape, (3, K15))
            mine = io_.select_tokens(io_.column_importance(attn, N15), K15, io_.SIM_THRESHOLD)
            with self.subTest(case=case):
                self.assert_same_selection(sel, dele, seen, mine, N15, K15, kinds)
        self.assertIn("fallback", kinds)
        self.assertTrue(kinds & {"consensus_no_tie", "consensus_tie"})

    def test_llava15_geometry_stored_prefix_rows_plus_suffix_rows(self):
        """The request-time split (stored prefix-row share + suffix-row share)
        selects like the public function on the whole matrix."""
        kinds = set()
        for case in (dict(seed=1, own=6.0), dict(seed=2), dict(seed=12, agree=3.0)):
            attn = _attention(n_prefix=N15, n_suffix=24, **case)
            dev, seen = _device()
            sel, dele = _quiet(dev._sele_important_tokens, attn, N15, -1, io_.SIM_THRESHOLD, 25)
            split = io_.prefix_row_importance(attn, N15) + io_.suffix_row_importance(attn, N15)
            mine = io_.select_tokens(split, K15, io_.SIM_THRESHOLD)
            with self.subTest(case=case):
                self.assert_same_selection(sel, dele, seen, mine, N15, K15, kinds)
        self.assertIn("fallback", kinds)
        self.assertTrue(kinds & {"consensus_no_tie", "consensus_tie"})

    def test_selector_on_a_store_without_separators_equals_the_public_function(self):
        """The hook (probe sidecar read, stored prefix rows + online suffix
        rows, stored positions) against the public function on the whole
        matrix restricted to [image tokens | suffix]."""
        with tempfile.TemporaryDirectory() as td:
            w = _world_without_separators(td, seed=3)      # a seed with both decisions
            r = w.run()
            m = w.meta
            k = io_.keep_count(w.vn)
            p0 = w.v0 + w.vn
            modes = set()
            for row in r.selector.log:
                li = row["layer"]
                order = torch.tensor(m["order"][li])
                whole = w.whole_matrix(li)
                dev, seen = _device()
                sub = whole[:, :, list(range(w.v0, whole.shape[2]))]
                sel, dele = _quiet(dev._sele_important_tokens, sub, w.vn, -1, io_.SIM_THRESHOLD, 25)
                got = [set(int(order[p]) for p in head) for head in r.selector._diag[li][2]]
                self.assertEqual(_sets(seen["head_topk"]), got)
                self.assertEqual(seen["similarity"], row["sim"])
                self.assertEqual(dele.numel() == 0, row["mode"] == "fallback")
                keep = r.selector.keep_masks[li]
                if row["mode"] == "consensus":
                    chosen = {int(order[p]) for p in keep.nonzero().flatten()}
                    votes = base._votes(_sets(seen["head_topk"]), w.vn)
                    public = set(int(i) for i in sel)
                    self.assertEqual(len(chosen), k)
                    self.assertEqual(sorted(votes[i] for i in public), sorted(votes[i] for i in chosen))
                    cut = min(votes[i] for i in chosen)
                    if sum(v >= cut for v in votes) == k:
                        self.assertEqual(public, chosen)
                else:
                    self.assertEqual(set(int(i) for i in sel), set(range(w.vn)))
                    self.assertTrue(bool(keep.all()))
                modes.add(row["mode"])
                self.assertEqual(p0, m["prefix_len"])
            self.assertEqual(modes, {"consensus", "fallback"})


class NoSeparatorStoreTests(unittest.TestCase):
    """The LLaVA-1.5 case of the shared layout writer and selector."""

    def tearDown(self):
        BIAS.clear()

    def test_layout_is_plain_descending_order_with_an_empty_separator_sidecar(self):
        with tempfile.TemporaryDirectory() as td:
            w = _world_without_separators(td)
            m = load_meta(w.root)
            io_.validate_official_store(m)
            self.assertEqual(m["newline_idx"], [])
            self.assertEqual(m["newline_stored"], [[]] * w.L)
            self.assertEqual((w.root / "sep_kv.bin").stat().st_size, 0)
            key_bytes, trailer_bytes = io_.probe_block_bytes(m)
            self.assertEqual(trailer_bytes, 3 * w.vn * 4)
            for li in range(w.L):
                self.assertEqual(m["order"][li], io_.descending_order(w.order_scores[li]).tolist())
                self.assertEqual(m["order"][li], np.argsort(-w.order_scores[li].numpy(), kind="stable").tolist())
                self.assertEqual((w.root / f"layer_{li:02d}/probe_k.bin").stat().st_size, key_bytes + trailer_bytes)

    def test_every_visual_token_is_a_candidate_and_fallback_masks_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            w = _world_without_separators(td, seed=3)
            r = w.run()
            self.assertEqual(r.selector.n_sep, 0)
            k = io_.keep_count(w.vn)
            self.assertEqual(r.selector.k_keep, k)
            nc = w.meta["n_chunks_per_layer"]
            modes = set()
            for row in r.selector.log:
                li = row["layer"]; keep = r.selector.keep_masks[li]; modes.add(row["mode"])
                if row["mode"] == "fallback":
                    self.assertLess(row["sim"], io_.SIM_THRESHOLD)
                    self.assertTrue(bool(keep.all()))
                    self.assertNotIn(li, r.bias)                       # no mask
                    self.assertEqual(row["k_chunks"], nc)
                    self.assertEqual(row["v_chunks"], nc)
                else:
                    self.assertGreaterEqual(row["sim"], io_.SIM_THRESHOLD)
                    self.assertEqual(int(keep.sum()), k)               # nothing kept on top of the budget
                    self.assertIn(li, r.bias)
            self.assertEqual(modes, {"consensus", "fallback"})

    def test_full_retention_selects_everything_in_every_layer(self):
        with tempfile.TemporaryDirectory() as td:
            w = _world_without_separators(td)
            r = w.run(ratio=1.0)
            nc = w.meta["n_chunks_per_layer"]
            for row in r.selector.log:
                self.assertEqual((row["mode"], row["sim"], row["k_chunks"], row["v_chunks"]), ("consensus", 1.0, nc, nc))
                self.assertTrue(bool(r.selector.keep_masks[row["layer"]].all()))
            for li, bias in r.bias.items():                            # an all-keep mask adds exactly zero
                self.assertFalse(bool(bias.ne(0).any()))


class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A = S.load_script("_mtgqa_impress_code_eval_test", "196_eval_mt_gqa_impress_code.py")
        cls.B = S.load_script("_l15_mtvqa_impress_code_eval_test", "199_eval_llava15_mtvqa_impress_code.py")

    def test_one_impress_implementation_shared_by_both_models(self):
        self.assertIs(self.A.impress_official, self.B.impress_official)
        self.assertIs(self.A.impress_official, io_)
        for E in (self.A, self.B):
            self.assertEqual({a["family"] for a in E.ARMS.values()}, {"recompute", "fullload", "impress_official", "ours"})
            self.assertEqual(E.OFFICIAL["probe_head_ids"], [0, 1, 2])
            self.assertEqual((E.OFFICIAL["sim_threshold"], E.OFFICIAL["ratio"], E.OFFICIAL["chunk_size"]), (0.3, 0.25, 64))
            self.assertEqual(E.OFFICIAL["prefetch_async"], "next_layer")
            self.assertEqual(E.OFFICIAL["reference_commit"], "91df7a2fc12581c6780a6485c5b50e7fd9ef9d0e")
            self.assertFalse(hasattr(E, "impress_mt"))

    def test_main_arms_and_display_names(self):
        for E, small in ((self.A, "dev"), (self.B, "pilot")):
            main = E.PHASE_METHODS["main"]
            self.assertEqual(main, ["recompute", "fullload_async", "impress_official_async", E.OURS_ASYNC])
            self.assertEqual(E.PHASE_METHODS[small], main)
            self.assertEqual([E.ARMS[m]["display"] for m in main], ["ReComp", "FullLoad", "IMPRESS", "Ours"])
            self.assertTrue(all(E.ARMS[m]["async"] for m in main[1:]))
        self.assertEqual(self.A.OURS, "ours_prefixstore_kv25_PYR-beta-raw_packed")
        self.assertEqual(self.B.OURS, "llava15_kv_PYR-b20_packed")

    def test_gate_arms_share_the_reordered_store(self):
        for E in (self.A, self.B):
            gate = E.PHASE_METHODS["gate"]
            self.assertEqual(gate, ["recompute", "fullload", "fullload_async", E.NEW, E.NEW_ASYNC, E.NEW_R100, E.FULL_REORDERED,
                                    E.OURS, E.OURS_ASYNC])
            self.assertEqual(E.store_kinds(gate), ["raster", E.NEW_KIND, E.PACKED_KIND])
            for m in (E.NEW, E.NEW_ASYNC, E.NEW_R100, E.FULL_REORDERED):
                self.assertEqual(E.kind_of(m), E.NEW_KIND)
            self.assertEqual(E.ARMS[E.FULL_REORDERED]["family"], "fullload")        # the FullLoad loader: no selection, no mask
            self.assertEqual(E.ARMS[E.NEW_R100]["ratio"], 1.0)
            self.assertEqual((E.ARMS[E.NEW]["async"], E.ARMS[E.NEW_ASYNC]["async"]), (False, True))
            self.assertEqual(E.READAHEAD[E.NEW_ASYNC], "next_layer")

    def test_run_locations(self):
        self.assertEqual(self.A.RUN_PREFIX, "mt_gqa_impress_code_")
        self.assertEqual(self.B.RUN_PREFIX, "llava15_mtvqa_impress_code_")
        self.assertEqual(self.A.CONTRACT.name, "mt_gqa_impress_code_contract.md")
        self.assertEqual(self.B.CONTRACT.name, "llava15_mtvqa_impress_code_contract.md")
        for E in (self.A, self.B):
            self.assertNotIn("196_eval_mt_gqa_impress_code.py", E.REPORT_ONLY)
            self.assertNotIn("199_eval_llava15_mtvqa_impress_code.py", E.REPORT_ONLY)
            self.assertTrue(all("report" in name for name in E.REPORT_ONLY))

    def test_validators_compare_what_the_contracts_say(self):
        VA = S.load_script("_mtgqa_impress_code_validate_test", "197_validate_mt_gqa_impress_code.py")
        VB = S.load_script("_l15_mtvqa_impress_code_validate_test", "200_validate_llava15_mtvqa_impress_code.py")

        def row(tokens, prompt="p", **extra):
            return dict({"result": {"generated_token_ids": tokens}, "prompt": prompt, "image_id": "i", "dialog_id": "d", "prediction": "x",
                         "ssd_read_bytes": 10, "probe_read_bytes": 1, "k_read_bytes": 4, "v_read_bytes": 5, "packed_read_bytes": 0}, **extra)

        # experiment A: {(method, dialog, turn): row}
        keys = {("d", 1), ("d", 2)}
        rows = {(m, "d", t): row([1, 2]) for m in ("a", "b") for t in (1, 2)}
        self.assertEqual(VA.compare(rows, "a", "b", ["generated_token_ids"] + VA.BYTE_FIELDS, keys)["status"], "PASS")
        rows[("b", "d", 2)] = row([1, 3])
        self.assertEqual(VA.compare(rows, "a", "b", ["generated_token_ids"], keys)["mismatch_count"], 1)
        self.assertEqual(VA.turn1_identity(rows, keys, ["a", "b"])["status"], "PASS")       # turn 1 still equal
        rows[("b", "d", 1)] = row([9])
        self.assertEqual(VA.turn1_identity(rows, keys, ["a", "b"])["status"], "FAIL")
        self.assertIn("step3_keep100_equals_fullload_loader", VA.REQUIRED)
        self.assertIn("step4_async_equals_sync", VA.REQUIRED)
        self.assertNotIn("reference_reordered_vs_raster_fullload", VA.REQUIRED)             # reported only

        # experiment B: {image: {(method, turn): row}}
        by_image = {"i": {(m, t): row([1, 2]) for m in ("a", "b") for t in (1, 2, 3)}}
        out = VB.compare_tokens(by_image, "a", "b")
        self.assertEqual((out["compared"], out["different_tokens"], out["different_input"]), (3, 0, 0))
        by_image["i"][("b", 3)] = row([1, 4], prompt="q")
        out = VB.compare_tokens(by_image, "a", "b")
        self.assertEqual((out["different_tokens"], out["different_tokens_same_input"], out["different_input"]), (1, 0, 1))
        self.assertEqual(VB.compare_sync_async(by_image, "a", "b")["mismatched"], 1)
        by_image["i"][("b", 3)] = row([1, 2], k_read_bytes=3)
        self.assertEqual(VB.compare_sync_async(by_image, "a", "b")["examples"][0]["fields"], ["k_read_bytes"])
        self.assertEqual(VB.t1_identity(by_image, ["a", "b"])["mismatched"], 0)
        self.assertEqual([n for n, _, _ in VB.SYNC_ASYNC], ["FullLoad", "IMPRESS", "Ours"])


if __name__ == "__main__":
    unittest.main()

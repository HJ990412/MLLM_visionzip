"""The IMPRESS selection path written from the public implementation.

Contract: docs/mt_gqa_impress_official_contract.md.

``PublicEquivalenceTests`` feed the same tensors to the functions of the
public repository (https://github.com/ISCS-ZJU/HyperInfer, commit 91df7a2,
``h2o_flexgen/flexgen/pytorch_backend.py``) and to ``mmimpress.impress_official``.
They need a checkout of that repository and are skipped without it:

    IMPRESS_OFFICIAL_REPO=/path/to/HyperInfer python -m unittest discover -s tests

Compared against public functions that are called as they are:

  _sele_important_tokens   per-head top-k, mean Jaccard, fallback decision,
                           final selection
  cal_jaccard_sim          the similarity value
  sele_tokenid             the whole probe-head path from hidden states
  mha_with_filled_selected_prefixkv   the mask on tokens that were not selected
  mha_with_sele_percent_prefixkv(generate_mapping_list=True)   the reorder
                           order (needs CUDA: the function allocates on cuda:0)

Inside one vote count the public code's order is undefined (an unstable
``argsort`` over the counts).  Inputs without a tie at the cut are compared
as sets; inputs with a tie are compared by the vote-count condition: both
selections have k tokens, no token outside has more votes than a token
inside, and the vote counts of the two selections are the same multiset.

Not called (reported as such): ``my_pcache_fast.PrefixKVLayer`` -- its
constructor builds GPU chunk caches and writes ``.npy`` chunk files under a
process-wide directory, and at that commit it applies the mapping as
``key[reorder]`` with ``reorder`` being the token -> rank index, which stores
token ``rank(i)`` at position ``i``.  The storage tests below pin the intended
behaviour instead (position r holds the token of rank r; K, V and probe rows
stay aligned).
"""
from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_impress_fidelity import _FakeCache, _FakeDecoderLayer  # noqa: E402

from mmimpress import impress_official as io_  # noqa: E402
from mmimpress.serve import BIAS, bias_from_keep  # noqa: E402
from mmimpress.store import (  # noqa: E402
    ChunkReader, IOCounter, load_meta, write_image_store)

OFFICIAL = os.environ.get("IMPRESS_OFFICIAL_REPO")
_PB = None


def _public():
    """``pytorch_backend`` of the public checkout, imported once."""
    global _PB
    if _PB is None:
        flexgen = Path(OFFICIAL) / "h2o_flexgen" / "flexgen"
        saved_path = list(sys.path)
        saved = {name: sys.modules.get(name) for name in ("utils", "pytorch_backend")}
        sys.path.insert(0, str(flexgen))
        try:
            import pytorch_backend
            _PB = pytorch_backend
        finally:
            sys.path[:] = saved_path
            for name, module in saved.items():       # leave no generic names behind
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
    return _PB


def _device(name="cpu"):
    """A ``TorchDevice`` without its constructor (which needs the compression
    module); only ``dev`` and ``device_type`` are used by the functions here."""
    pb = _public()
    dev = pb.TorchDevice.__new__(pb.TorchDevice)
    dev.dev = torch.device(name)
    dev.device_type = pb.DeviceType.convert(dev.dev.type)
    dev.layer = 1
    seen = {}
    original = pb.TorchDevice.cal_jaccard_sim

    def spy(indices):
        seen["head_topk"] = indices.detach().cpu().clone()
        seen["similarity"] = original(dev, indices)
        return seen["similarity"]

    dev.cal_jaccard_sim = spy
    return dev, seen


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _wrap(dev, tensor):
    return _public().TorchTensor.create_from_torch(tensor.to(dev.dev), dev)


def _attention(seed, heads=3, n_prefix=48, n_suffix=6, agree=0.0, own=0.0, same=()):
    """Causal attention (heads, s, s).  ``agree`` mixes a column profile that
    all heads share, ``own`` one that each head has for itself (causal column
    sums favour early tokens for every head, so without ``own`` the heads
    agree); heads listed in ``same`` copy head 0 exactly."""
    g = torch.Generator().manual_seed(seed)
    s = n_prefix + n_suffix
    logits = torch.randn(heads, s, s, generator=g, dtype=torch.float64)
    shared = torch.randn(s, generator=g, dtype=torch.float64) * agree
    private = torch.randn(heads, 1, s, generator=g, dtype=torch.float64) * own
    logits = logits + shared.view(1, 1, s) + private
    for h in same:
        logits[h] = logits[0]
    idx = torch.arange(s)
    logits = logits.masked_fill(idx.view(1, 1, s) > idx.view(1, s, 1), float("-inf"))
    return torch.softmax(logits, dim=2)


def _sets(indices):
    return [set(int(i) for i in row) for row in indices]


def _votes(head_sets, n):
    votes = [0] * n
    for chosen in head_sets:
        for i in chosen:
            votes[i] += 1
    return votes


class _Compare:
    """Assertions shared by the selection comparisons."""

    def assert_same_selection(self, public_sel, public_del, seen, mine, n, k, kinds):
        self.assertEqual(_sets(seen["head_topk"]), _sets(mine["head_topk"].cpu()))
        self.assertEqual(seen["similarity"], mine["similarity"])
        public_fallback = public_del.numel() == 0
        self.assertEqual(public_fallback, mine["fallback"])
        chosen = set(int(i) for i in mine["selected"].cpu())
        public = set(int(i) for i in public_sel)
        if mine["fallback"]:
            kinds.add("fallback")
            self.assertEqual(public, set(range(n)))
            self.assertEqual(chosen, set(range(n)))
            return
        self.assertEqual(public | set(int(i) for i in public_del), set(range(n)))
        self.assertFalse(public & set(int(i) for i in public_del))
        votes = _votes(_sets(seen["head_topk"]), n)
        self.assertEqual(votes, mine["votes"].cpu().tolist())
        for name, picked in (("public", public), ("ours", chosen)):
            self.assertEqual(len(picked), k, name)
            inside = min(votes[i] for i in picked)
            outside = max((votes[i] for i in range(n) if i not in picked), default=0)
            self.assertGreaterEqual(inside, outside, name)
        self.assertEqual(sorted(votes[i] for i in public),
                         sorted(votes[i] for i in chosen))
        cut = min(votes[i] for i in chosen)
        if sum(v >= cut for v in votes) == k:          # no tie at the cut
            kinds.add("consensus_no_tie")
            self.assertEqual(public, chosen)
        else:
            kinds.add("consensus_tie")


@unittest.skipUnless(OFFICIAL, "set IMPRESS_OFFICIAL_REPO to the public checkout")
class PublicEquivalenceTests(_Compare, unittest.TestCase):
    def test_reference_checkout_is_the_recorded_commit(self):
        head = subprocess.check_output(
            ["git", "-C", OFFICIAL, "rev-parse", "HEAD"], text=True).strip()
        self.assertEqual(head, io_.OFFICIAL_COMMIT)

    def test_selection_from_the_same_attention_matrix(self):
        kinds = set()
        cases = [dict(seed=s, own=6.0) for s in range(6)]                # heads disagree
        cases += [dict(seed=s) for s in range(4)]
        cases += [dict(seed=10 + s, agree=a) for s in range(6) for a in (1.5, 3.0)]
        cases += [dict(seed=30 + s, same=(1,)) for s in range(3)]        # A, A, C
        cases += [dict(seed=40, same=(1, 2))]                            # A, A, A
        for case in cases:
            for n_prefix in (48, 50, 61):
                attn = _attention(n_prefix=n_prefix, **case)
                dev, seen = _device()
                sel, dele = _quiet(dev._sele_important_tokens, attn, n_prefix,
                                   -1, io_.SIM_THRESHOLD, 25)
                k = io_.keep_count(n_prefix)
                self.assertEqual(seen["head_topk"].shape[1], k)
                mine = io_.select_tokens(
                    io_.column_importance(attn, n_prefix), k, io_.SIM_THRESHOLD)
                with self.subTest(case=case, n_prefix=n_prefix):
                    self.assert_same_selection(sel, dele, seen, mine, n_prefix, k, kinds)
        self.assertEqual(kinds, {"fallback", "consensus_no_tie", "consensus_tie"})

    def test_threshold_is_inclusive_like_the_public_code(self):
        attn = _attention(seed=10, agree=3.0)
        dev, seen = _device()
        _quiet(dev._sele_important_tokens, attn, 48, -1, 0.0, 25)
        sim = seen["similarity"]
        self.assertTrue(0.0 < sim < 1.0)
        imp = io_.column_importance(attn, 48)
        for thr, fallback in ((sim, False), (np.nextafter(sim, 2.0), True)):
            dev, seen = _device()
            sel, dele = _quiet(dev._sele_important_tokens, attn, 48, -1, thr, 25)
            self.assertEqual(dele.numel() == 0, fallback)
            self.assertEqual(io_.select_tokens(imp, 12, thr)["fallback"], fallback)

    def test_jaccard_value(self):
        g = torch.Generator().manual_seed(5)
        pb = _public()
        dev, _ = _device()
        for heads, n, k in ((3, 40, 10), (3, 200, 50), (4, 64, 16), (2, 30, 30)):
            idx = torch.stack([torch.randperm(n, generator=g)[:k] for _ in range(heads)])
            public = pb.TorchDevice.cal_jaccard_sim(dev, idx)
            mine, _ = io_.mean_pairwise_jaccard(idx, n)
            self.assertEqual(public, mine)

    def _opt_layer(self, seed, n_prefix, n_suffix, heads=4, hd=8, mode="random"):
        g = torch.Generator().manual_seed(seed)
        h, s = heads * hd, n_prefix + n_suffix
        t = {name: torch.randn(*shape, generator=g) for name, shape in (
            ("inputs", (1, s, h)), ("w_q", (h, h)), ("b_q", (h,)), ("w_k", (h, h)),
            ("b_k", (h,)), ("w_v", (h, h)), ("b_v", (h,)), ("w_out", (h, h)),
            ("b_out", (h,)), ("w_ln", (h,)), ("b_ln", (h,)))}
        prefix_k = torch.randn(n_prefix, 3, hd, generator=g)
        if mode != "random":                      # make the probe heads agree
            for name in ("w_q", "w_k"):
                t[name][hd:3 * hd] = t[name][:hd].repeat(2, 1)
            for name in ("b_q", "b_k"):
                t[name][hd:3 * hd] = t[name][:hd].repeat(2)
            prefix_k[:, 1:] = prefix_k[:, :1]
            if mode == "noisy":
                t["w_q"][hd:3 * hd] += 0.35 * torch.randn(2 * hd, h, generator=g)
                prefix_k[:, 1:] += 0.35 * torch.randn(n_prefix, 2, hd, generator=g)
        return t, prefix_k, heads, hd

    def _our_probe_attention(self, t, prefix_k, heads, hd, n_prefix, dtype):
        h = heads * hd
        hidden = F.layer_norm(t["inputs"], (h,), weight=t["w_ln"], bias=t["b_ln"])
        s = hidden.shape[1]
        q = (F.linear(hidden, t["w_q"], t["b_q"]) * hd ** -0.5).view(1, s, heads, hd)
        k = F.linear(hidden[:, n_prefix:], t["w_k"], t["b_k"]).view(1, s - n_prefix, heads, hd)
        q = q[0].permute(1, 0, 2)[:3]                               # (3, s, hd)
        keys = torch.cat([prefix_k.permute(1, 0, 2), k[0].permute(1, 0, 2)[:3]], dim=1)
        return io_.causal_attention(q, keys, dtype=dtype)

    def test_sele_tokenid_from_hidden_states(self):
        kinds = set()
        for seed, mode in [(s, "random") for s in range(3)] + \
                [(20 + s, "noisy") for s in range(6)] + [(40, "same")]:
            for n_prefix, n_suffix in ((40, 7), (52, 5)):
                t, prefix_k, heads, hd = self._opt_layer(seed, n_prefix, n_suffix, mode=mode)
                dev, seen = _device()
                w = {name: _wrap(dev, value) for name, value in t.items()}
                mask = _wrap(dev, torch.ones(1, n_prefix + n_suffix, dtype=torch.bool))
                sel, dele = _quiet(
                    dev.sele_tokenid, w["inputs"], mask, w["w_q"], w["b_q"], w["w_k"],
                    w["b_k"], w["w_v"], w["b_v"], w["w_out"], w["b_out"], w["w_ln"],
                    w["b_ln"], heads, (False, False), io_.PROBE_HEADS, prefix_k, -1,
                    io_.SIM_THRESHOLD, 25)
                # the public function sums a float16 softmax; same dtype here
                attn = self._our_probe_attention(t, prefix_k, heads, hd, n_prefix,
                                                 torch.float16)
                k = io_.keep_count(n_prefix)
                mine = io_.select_tokens(io_.column_importance(attn, n_prefix), k,
                                         io_.SIM_THRESHOLD)
                with self.subTest(seed=seed, mode=mode, n_prefix=n_prefix):
                    self.assert_same_selection(sel, dele, seen, mine, n_prefix, k, kinds)
        self.assertIn("fallback", kinds)
        self.assertTrue(kinds & {"consensus_no_tie", "consensus_tie"})

    def test_stored_prefix_rows_plus_suffix_rows_select_like_the_whole_matrix(self):
        """The split used at request time: stored prefix-row share + suffix-row
        share computed from q/k, against the public function on the whole
        matrix."""
        kinds = set()
        for seed, mode in [(s, "random") for s in range(3)] + \
                [(20 + s, "noisy") for s in range(6)]:
            n_prefix, n_suffix = 44, 6
            t, prefix_k, heads, hd = self._opt_layer(seed, n_prefix, n_suffix, mode=mode)
            whole = self._our_probe_attention(t, prefix_k, heads, hd, n_prefix,
                                              torch.float32)
            dev, seen = _device()
            sel, dele = _quiet(dev._sele_important_tokens, whole, n_prefix, -1,
                               io_.SIM_THRESHOLD, 25)
            # prefix rows alone: a request with a DIFFERENT suffix gives the same share
            other, _, _, _ = self._opt_layer(seed, n_prefix, 3, mode=mode)
            other["inputs"][:, :n_prefix] = t["inputs"][:, :n_prefix]
            for name in t:
                if name != "inputs":
                    other[name] = t[name]
            stored = io_.prefix_row_importance(
                self._our_probe_attention(other, prefix_k, heads, hd, n_prefix,
                                          torch.float32), n_prefix)
            torch.testing.assert_close(
                stored, io_.prefix_row_importance(whole, n_prefix), rtol=1e-5, atol=1e-6)
            h = heads * hd
            hidden = F.layer_norm(t["inputs"], (h,), weight=t["w_ln"], bias=t["b_ln"])
            suffix = hidden[:, n_prefix:]
            q = F.linear(suffix, t["w_q"], t["b_q"]).view(n_suffix, heads, hd).permute(1, 0, 2)[:3]
            ks = F.linear(suffix, t["w_k"], t["b_k"]).view(n_suffix, heads, hd).permute(1, 0, 2)[:3]
            keys = torch.cat([prefix_k.permute(1, 0, 2), ks], dim=1)
            online = io_.suffix_row_importance_from_qk(q, keys, n_prefix, hd ** -0.5)
            torch.testing.assert_close(
                online, io_.suffix_row_importance(whole, n_prefix), rtol=1e-4, atol=1e-6)
            k = io_.keep_count(n_prefix)
            mine = io_.select_tokens(stored + online, k, io_.SIM_THRESHOLD)
            with self.subTest(seed=seed, mode=mode):
                self.assert_same_selection(sel, dele, seen, mine, n_prefix, k, kinds)
        self.assertIn("fallback", kinds)
        self.assertTrue(kinds & {"consensus_no_tie", "consensus_tie"})

    @unittest.skipUnless(torch.cuda.is_available(),
                         "the public attention forward mixes a float16 softmax "
                         "with the value tensor: float16 inputs, i.e. CUDA")
    def test_mask_equals_public_filled_selected_attention(self):
        """Tokens that were not selected: the public function writes
        ``finfo.min`` on their columns (their K/V slots hold zeros).  Here the
        same output must come from the full K/V plus the additive mask the
        serving path builds from the keep mask."""
        g = torch.Generator().manual_seed(3)
        heads, hd, n_prefix, n_suffix = 4, 8, 20, 5
        h, s = heads * hd, n_prefix + n_suffix
        t, _, _, _ = self._opt_layer(7, n_prefix, n_suffix)
        t = {name: (value * (1.0 if name in ("inputs", "w_ln", "b_ln") else 0.4))
             .to(torch.float16) for name, value in t.items()}
        prefix_k = torch.randn(n_prefix, heads, hd, generator=g).to(torch.float16)
        prefix_v = torch.randn(n_prefix, heads, hd, generator=g).to(torch.float16)
        keep = torch.zeros(n_prefix, dtype=torch.bool)
        keep[torch.tensor([0, 2, 3, 7, 11, 12, 19])] = True       # token 0 kept
        selected, deleted = keep.nonzero().flatten(), (~keep).nonzero().flatten()
        filled_k, filled_v = prefix_k.clone(), prefix_v.clone()
        filled_k[deleted] = 0
        filled_v[deleted] = 0
        dev, _ = _device("cuda:0")
        w = {name: _wrap(dev, value) for name, value in t.items()}
        mask = _wrap(dev, torch.ones(1, s, dtype=torch.bool))
        value, *_ = _quiet(
            dev.mha_with_filled_selected_prefixkv, w["inputs"], mask, w["w_q"],
            w["b_q"], w["w_k"], w["b_k"], w["w_v"], w["b_v"], w["w_out"], w["b_out"],
            w["w_ln"], w["b_ln"], heads, (False, False), False, None,
            filled_prefix_k=filled_k.cuda(), filled_prefix_v=filled_v.cuda(),
            suffix_len=n_suffix, select_kv_tokenid=selected.cuda(),
            del_tokenids=deleted.cuda())
        public = value.data.float().cpu()

        t = {name: v.float() for name, v in t.items()}       # same values, float32 math
        prefix_k, prefix_v = prefix_k.float(), prefix_v.float()
        hidden = F.layer_norm(t["inputs"], (h,), weight=t["w_ln"], bias=t["b_ln"])
        q = (F.linear(hidden, t["w_q"], t["b_q"]) * hd ** -0.5).view(s, heads, hd).permute(1, 0, 2)
        ks = F.linear(hidden[0, n_prefix:], t["w_k"], t["b_k"]).view(n_suffix, heads, hd)
        vs = F.linear(hidden[0, n_prefix:], t["w_v"], t["b_v"]).view(n_suffix, heads, hd)
        keys = torch.cat([prefix_k, ks]).permute(1, 0, 2)          # full, unmasked
        vals = torch.cat([prefix_v, vs]).permute(1, 0, 2)
        meta = {"prefix_len": n_prefix, "v_token_start": 0, "v_token_num": n_prefix}
        bias = bias_from_keep(keep, meta, torch.device("cpu"), dtype=torch.float32)
        idx = torch.arange(s)
        causal = idx.view(1, 1, s) > idx.view(1, s, 1)

        def forward(with_mask):
            logits = q @ keys.transpose(1, 2)
            if with_mask:
                logits[:, :, :n_prefix] += bias[0, :, 0]
            probs = torch.softmax(logits.masked_fill(causal, float("-inf")), dim=2)
            out = (probs @ vals).permute(1, 0, 2).reshape(1, s, h)
            return F.linear(out, t["w_out"], t["b_out"]) + t["inputs"], probs

        ours, probs = forward(True)
        unmasked, _ = forward(False)
        self.assertTrue(bool((probs[:, :, deleted] == 0).all()))
        error = (public - ours).abs().max()
        effect = (unmasked - ours)[:, n_prefix:].abs().max()
        self.assertLess(float(error), 0.03)
        self.assertGreater(float(effect), 10 * float(error))    # the mask matters

    @unittest.skipUnless(torch.cuda.is_available(),
                         "the public mapping-list path allocates on cuda:0")
    def test_reorder_order_equals_public_mapping_list(self):
        """Order of ``--generate-mapping-list``: column sums over every row,
        summed over every head, descending.  The public path runs in float16
        (softmax and both sums), so two tokens whose float32 scores are closer
        than its rounding may swap; every pair further apart must agree."""
        separated = total = 0
        for seed in range(4):
            n_prefix, n_suffix = 40, 5
            t, _, heads, hd = self._opt_layer(seed, n_prefix, n_suffix)
            t = {name: value.to(torch.float16) for name, value in t.items()}
            g = torch.Generator().manual_seed(100 + seed)
            prefix_k = torch.randn(n_prefix, heads, hd, generator=g).to(torch.float16)
            prefix_v = torch.randn(n_prefix, heads, hd, generator=g).to(torch.float16)
            dev, _ = _device("cuda:0")
            w = {name: _wrap(dev, value) for name, value in t.items()}
            mask = _wrap(dev, torch.ones(1, n_prefix + n_suffix, dtype=torch.bool))
            *_, reverse = _quiet(
                dev.mha_with_sele_percent_prefixkv, w["inputs"], mask, w["w_q"],
                w["b_q"], w["w_k"], w["b_k"], w["w_v"], w["b_v"], w["w_out"],
                w["b_out"], w["w_ln"], w["b_ln"], heads, (False, False), False, None,
                prefix_k=prefix_k.cuda(), prefix_v=prefix_v.cuda(),
                suffix_len=n_suffix, cur_sele_percent=25, generate_mapping_list=True)
            reverse = reverse.cpu()
            self.assertEqual(sorted(reverse.tolist()), list(range(n_prefix)))

            t = {name: v.float() for name, v in t.items()}
            h, s = heads * hd, n_prefix + n_suffix
            hidden = F.layer_norm(t["inputs"], (h,), weight=t["w_ln"], bias=t["b_ln"])
            q = (F.linear(hidden, t["w_q"], t["b_q"]) * hd ** -0.5).view(s, heads, hd).permute(1, 0, 2)
            ks = F.linear(hidden[0, n_prefix:], t["w_k"], t["b_k"]).view(n_suffix, heads, hd)
            keys = torch.cat([prefix_k.float(), ks]).permute(1, 0, 2)
            attn = io_.causal_attention(q, keys)
            # the statistic exactly as the turn-1 capture takes it
            cap = io_.Turn1AttentionStats(0, n_prefix)
            with mock.patch.object(io_.ml, "eager_attention_forward",
                                   lambda module, *a, **k: (None, attn.unsqueeze(0))):
                with cap:
                    io_.ml.eager_attention_forward(SimpleNamespace(layer_idx=0), None, None, None, None)
            summed = cap.result()[1][0]
            torch.testing.assert_close(summed, attn.sum(dim=(0, 1))[:n_prefix])
            order = io_.descending_order(summed)
            mine = io_.reverse_index(order)
            eps = 0.05
            for a in range(n_prefix):
                for b in range(n_prefix):
                    if summed[a] - summed[b] > eps:
                        separated += 1
                        self.assertLess(int(reverse[a]), int(reverse[b]), (seed, a, b))
                        self.assertLess(int(mine[a]), int(mine[b]))
                    total += a < b
            # tokens the public order puts next to each other never rise by more than eps
            public_order = torch.argsort(reverse)
            steps = summed[public_order][1:] - summed[public_order][:-1]
            self.assertLessEqual(float(steps.max()), eps)
        self.assertGreater(separated / total, 0.8)             # the check is not vacuous


# ====================================================== always-run contracts
class SelectionTests(unittest.TestCase):
    def test_keep_count_is_the_public_ceil_formula(self):
        import math
        for n in range(1, 4000):
            self.assertEqual(io_.keep_count(n, 0.25), math.ceil(n * 25 / 100))
        self.assertEqual(io_.keep_count(7, 1.0), 7)

    def test_vote_consensus_prefers_votes_then_lower_index(self):
        masks = torch.zeros(3, 10, dtype=torch.bool)
        masks[0, [1, 2, 3, 4]] = True
        masks[1, [2, 3, 4, 5]] = True
        masks[2, [3, 4, 8, 9]] = True                # votes: 3,4 -> 3; 2 -> 2; 1,5,8,9 -> 1
        chosen, votes = io_.vote_consensus(masks, 4)
        self.assertEqual(votes.tolist(), [0, 1, 2, 3, 3, 1, 0, 0, 1, 1])
        self.assertEqual(chosen.tolist(), [1, 2, 3, 4])          # tie among 1,5,8,9 -> 1

    def test_select_tokens_never_picks_forbidden_candidates(self):
        g = torch.Generator().manual_seed(0)
        imp = torch.rand(3, 30, generator=g)
        imp[:, [0, 1]] = float("-inf")
        for thr in (0.0, 2.0):
            r = io_.select_tokens(imp, io_.keep_count(28), thr)
            self.assertFalse(bool((r["head_topk"] < 2).any()))
            if not r["fallback"]:
                self.assertFalse(set(r["selected"].tolist()) & {0, 1})
            else:
                self.assertEqual(r["selected"].tolist(), list(range(30)))

    def test_full_retention_is_a_consensus_on_every_candidate(self):
        imp = torch.rand(3, 20, generator=torch.Generator().manual_seed(1))
        imp[:, :2] = float("-inf")
        r = io_.select_tokens(imp, io_.keep_count(18, 1.0), io_.SIM_THRESHOLD)
        self.assertFalse(r["fallback"])
        self.assertEqual(r["similarity"], 1.0)
        self.assertEqual(r["selected"].tolist(), list(range(2, 20)))

    def test_layer_order_puts_separators_first_then_descending(self):
        summed = torch.tensor([.5, .9, .1, .7, .7, .3])
        order = io_.layer_order(summed, [4, 2])
        self.assertEqual(order, [2, 4, 1, 3, 0, 5])
        self.assertEqual(io_.reverse_index(order).tolist(), [4, 2, 0, 3, 1, 5])
        self.assertEqual(io_.descending_order(summed).tolist(), [1, 3, 4, 0, 5, 2])


class Turn1CaptureTests(unittest.TestCase):
    def test_capture_reads_the_models_own_prefill_attention(self):
        g = torch.Generator().manual_seed(2)
        v0, vn, n, heads = 2, 9, 4, 5
        s = v0 + vn + n
        weights = {li: torch.rand(1, heads, s, s, generator=g) for li in range(3)}
        calls = []

        def fake_eager(module, query, key, value, attention_mask, **kw):
            calls.append(module.layer_idx)
            if query.shape[-2] == 1:                       # a decoding step
                return "decode", torch.rand(1, heads, 1, s + 1)
            return "prefill", weights[module.layer_idx]

        ml = io_.ml
        with mock.patch.object(ml, "eager_attention_forward", fake_eager):
            cap = io_.Turn1AttentionStats(v0, vn)
            with cap:
                for li in range(3):
                    out = ml.eager_attention_forward(
                        SimpleNamespace(layer_idx=li), torch.zeros(1, heads, s, 2), None, None, None)
                    self.assertEqual(out[0], "prefill")
                for li in range(3):
                    out = ml.eager_attention_forward(
                        SimpleNamespace(layer_idx=li), torch.zeros(1, heads, 1, 2), None, None, None)
                    self.assertEqual(out[0], "decode")
            self.assertIs(ml.eager_attention_forward, fake_eager)   # restored
        prefix, order = cap.result()
        self.assertEqual(cap.calls, 3)
        self.assertEqual((tuple(prefix.shape), tuple(order.shape)), ((3, 3, vn), (3, vn)))
        for li in range(3):
            w = weights[li][0]
            torch.testing.assert_close(prefix[li], w[:3, :v0 + vn, v0:v0 + vn].sum(1))
            torch.testing.assert_close(order[li], w[:, :, v0:v0 + vn].sum((0, 1)))
        with self.assertRaises(RuntimeError):
            cap.__enter__()


def _store(root, seed=0, v_num=40, layers=4, heads=4, hd=8, chunk=4,
           newline=(9, 19, 29, 39)):
    g = torch.Generator().manual_seed(seed)
    v0 = 2
    prefix = v0 + v_num
    kv = [(torch.randn(1, heads, prefix, hd, generator=g) * 2.0,
           torch.randn(1, heads, prefix, hd, generator=g) + 3.0)
          for _ in range(layers)]
    meta = write_image_store(root, kv, v0, v_num, list(range(prefix)), list(newline),
                             chunk_size=chunk, probe_heads=3, separator_sidecar=True)
    return meta, kv


class LayoutTests(unittest.TestCase):
    def test_layout_keeps_k_v_probe_aligned_and_appends_prefix_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "s"
            meta, kv = _store(root)
            L, vn, H, hd = meta["num_layers"], meta["v_token_num"], 4, 8
            raster = {name: [np.fromfile(root / f"layer_{li:02d}/{name}.bin", dtype=np.float16)
                             .reshape(vn, -1, hd).copy() for li in range(L)]
                      for name in ("k", "v", "probe_k")}
            g = torch.Generator().manual_seed(8)
            prefix_rows = torch.rand(L, 3, vn, generator=g)
            scores = torch.rand(L, vn, generator=g)
            timing = io_.write_official_layout(root, prefix_rows, scores,
                                               metadata_update={"note": "x"})
            m = load_meta(root)
            io_.validate_official_store(m)
            self.assertEqual(m["note"], "x")
            seps = sorted(meta["newline_idx"])
            reader = ChunkReader(root, m, drop_cache=False)
            self.addCleanup(reader.close)
            key_bytes, trailer_bytes = io_.probe_block_bytes(m)
            for li in range(L):
                order = m["order"][li]
                self.assertEqual(order[:len(seps)], seps)
                rest = order[len(seps):]
                values = scores[li][torch.tensor(rest)]
                self.assertTrue(bool((values[:-1] >= values[1:]).all()))
                self.assertEqual(order, io_.layer_order(scores[li], seps))
                for name in ("k", "v"):
                    stored = np.fromfile(root / f"layer_{li:02d}/{name}.bin",
                                         dtype=np.float16).reshape(vn, H, hd)
                    np.testing.assert_array_equal(stored, raster[name][li][order])
                self.assertEqual((root / f"layer_{li:02d}/probe_k.bin").stat().st_size,
                                 key_bytes + trailer_bytes)
                counter = IOCounter()
                keys, rows = io_.read_probe_block(reader, li, counter)
                np.testing.assert_array_equal(keys.numpy(), raster["probe_k"][li][order])
                np.testing.assert_array_equal(
                    rows.numpy(), prefix_rows[li].numpy()[:, order].astype("<f4"))
                self.assertEqual(counter.preads, 1)
                self.assertEqual(counter.per_kind["probe"]["bytes"], key_bytes + trailer_bytes)
                # the probe sidecar still is the first three heads of K
                np.testing.assert_array_equal(
                    keys.numpy(), reader.read_full(li, "k").numpy()[:, :3])
            sep = np.fromfile(root / "sep_kv.bin", dtype=np.float16).reshape(2, L, len(seps), H, hd)
            for li in range(L):
                np.testing.assert_array_equal(sep[0, li], raster["k"][li][seps])
                np.testing.assert_array_equal(sep[1, li], raster["v"][li][seps])
            self.assertEqual(timing["trailer_bytes_per_layer"], trailer_bytes)
            with self.assertRaises(ValueError):
                io_.write_official_layout(root, prefix_rows, scores)      # not fresh

    def test_validation_rejects_other_layouts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "s"
            meta, _ = _store(root)
            with self.assertRaises(ValueError):
                io_.validate_official_store(load_meta(root))


class _World:
    """A tiny image prefix with turn-1 prefix-row queries, on a real store.

    Layers 0 and 2: the probe heads share keys and queries (consensus);
    layers 1 and 3: independent noise (whichever the similarity says; with
    the default seed both paths occur).
    """

    def __init__(self, td, seed=0, n_suffix=5):
        self.root = Path(td) / "store"
        self.raster_meta, self.kv = _store(self.root, seed=seed)
        m = self.raster_meta
        self.L, self.H, self.hd = m["num_layers"], m["num_heads"], m["head_dim"]
        self.v0, self.vn = m["v_token_start"], m["v_token_num"]
        g = torch.Generator().manual_seed(seed + 100)
        p0 = self.v0 + self.vn
        # keys as stored (fp16), probe heads only; (P, prefix_len, hd)
        self.keys = []
        for li in range(self.L):
            k = self.kv[li][0][0].clone()
            if li in (0, 2):
                k[1:3] = k[:1]
            self.kv[li] = (k.unsqueeze(0), self.kv[li][1])
        # rewrite the store with the agreeing keys
        import shutil
        shutil.rmtree(self.root)
        self.raster_meta = write_image_store(
            self.root, self.kv, self.v0, self.vn, list(range(p0)),
            list(m["newline_idx"]), chunk_size=m["chunk_size"], probe_heads=3,
            separator_sidecar=True)
        self.sys_k = [self.kv[li][0][0, :, :self.v0].to(torch.float16).float()
                      for li in range(self.L)]
        self.vis_k = [self.kv[li][0][0, :3, self.v0:].to(torch.float16).float()
                      for li in range(self.L)]                       # (3, vn, hd)
        self.q_prefix = [torch.randn(3, p0, self.hd, generator=g) for _ in range(self.L)]
        self.hidden = [torch.randn(1, n_suffix, self.H * self.hd, generator=g) * 2.0
                       for _ in range(self.L)]
        for li in (0, 2):
            self.q_prefix[li][1:] = self.q_prefix[li][:1]
            head0 = self.hidden[li].view(1, n_suffix, self.H, self.hd)[:, :, :1]
            self.hidden[li].view(1, n_suffix, self.H, self.hd)[:, :, :3] = head0
        self.n_suffix = n_suffix
        self.scale = self.hd ** -0.5
        # turn-1 statistics: prefix rows of the probe heads, and an order score
        prefix_keys = [torch.cat([self.sys_k[li][:3], self.vis_k[li]], dim=1)
                       for li in range(self.L)]
        self.prefix_rows = torch.stack([
            io_.prefix_row_importance(
                io_.causal_attention(self.q_prefix[li] * self.scale, prefix_keys[li]),
                p0)[:, self.v0:] for li in range(self.L)])
        self.order_scores = torch.rand(self.L, self.vn, generator=g)
        io_.write_official_layout(self.root, self.prefix_rows, self.order_scores)
        self.meta = load_meta(self.root)
        self.layers = [_FakeDecoderLayer() for _ in range(self.L)]
        self.runner = SimpleNamespace(layers=self.layers,
                                      model=SimpleNamespace(device=torch.device("cpu")))

    def context(self):
        reader = ChunkReader(self.root, self.meta, drop_cache=False)
        return SimpleNamespace(meta=self.meta, reader=reader,
                               cache=_FakeCache(self.meta, self.sys_k))

    def whole_matrix(self, li):
        """Probe-head attention of the whole sequence, ORIGINAL token order."""
        suffix = self.hidden[li].view(self.n_suffix, self.H, self.hd).permute(1, 0, 2)[:3]
        q = torch.cat([self.q_prefix[li], suffix], dim=1) * self.scale
        keys = torch.cat([self.sys_k[li][:3], self.vis_k[li], suffix], dim=1)
        return io_.causal_attention(q, keys)

    def run(self, **kwargs):
        BIAS.clear()
        ctx = self.context()
        counter = IOCounter()
        selector = io_.OfficialImpressSelector(self.runner, ctx, counter=counter, **kwargs)
        with mock.patch("mmimpress.impress_official.ml.apply_rotary_pos_emb",
                        side_effect=lambda query, key, cos, sin: (query, key)):
            with selector:
                for li, layer in enumerate(self.layers):
                    layer(hidden_states=self.hidden[li], position_embeddings=(None, None))
                for li, layer in enumerate(self.layers):      # decoding steps
                    layer(hidden_states=self.hidden[li][:, :1],
                          position_embeddings=(None, None))
        bias = {li: BIAS[li].clone() for li in sorted(BIAS)}
        BIAS.clear()
        ctx.reader.close()
        return SimpleNamespace(selector=selector, ctx=ctx, counter=counter, bias=bias)


class SelectorTests(_Compare, unittest.TestCase):
    def tearDown(self):
        BIAS.clear()

    def test_selector_selects_like_the_whole_matrix_pipeline(self):
        """Stored prefix rows + online suffix rows inside the hook, against the
        whole-matrix column sums; with the public checkout, against the
        public ``_sele_important_tokens`` on that matrix."""
        with tempfile.TemporaryDirectory() as td:
            w = _World(td)
            r = w.run()
            m = w.meta
            k = io_.keep_count(m["n_spatial"])
            p0 = w.v0 + w.vn
            cand = [i for i in range(w.vn) if i not in set(m["newline_idx"])]
            modes = []
            for row in r.selector.log:
                li = row["layer"]
                order = torch.tensor(m["order"][li])
                whole = w.whole_matrix(li)
                imp = io_.column_importance(whole, p0)[:, w.v0:][:, cand]
                ref = io_.select_tokens(imp, k, io_.SIM_THRESHOLD)
                self.assertEqual(row["sim"], ref["similarity"])
                self.assertEqual(row["mode"] == "fallback", ref["fallback"])
                modes.append(row["mode"])
                # selector's head top-k (stored positions) -> candidate indices
                top = r.selector._diag[li][2]
                to_cand = {orig: j for j, orig in enumerate(cand)}
                got = [set(to_cand[int(order[p])] for p in head) for head in top]
                self.assertEqual(got, _sets(ref["head_topk"]))
                keep = r.selector.keep_masks[li]
                if ref["fallback"]:
                    self.assertTrue(bool(keep.all()))
                    self.assertNotIn(li, r.bias)
                else:
                    chosen = {to_cand[int(order[p])] for p in keep.nonzero().flatten()
                              if int(order[p]) in to_cand}
                    votes = ref["votes"].tolist()
                    self.assertEqual(len(chosen), k)
                    self.assertGreaterEqual(min(votes[i] for i in chosen),
                                            max(votes[i] for i in range(len(cand))
                                                if i not in chosen))
                    # separators are kept on top of the budget
                    self.assertEqual(int(keep.sum()), k + len(m["newline_idx"]))
                    self.assertTrue(bool(keep[:len(m["newline_idx"])].all()))
                if OFFICIAL:
                    dev, seen = _device()
                    sub = torch.zeros(3, whole.shape[1], len(cand) + whole.shape[2] - p0)
                    cols = [w.v0 + i for i in cand] + list(range(p0, whole.shape[2]))
                    sub = whole[:, :, cols]
                    sel, dele = _quiet(dev._sele_important_tokens, sub, len(cand), -1,
                                       io_.SIM_THRESHOLD, 25)
                    self.assertEqual(_sets(seen["head_topk"]), got)
                    self.assertEqual(seen["similarity"], row["sim"])
                    self.assertEqual(dele.numel() == 0, row["mode"] == "fallback")
            self.assertEqual(set(modes), {"consensus", "fallback"})

    def test_consensus_reads_chunks_and_masks_fallback_reads_all_and_masks_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            w = _World(td)
            r = w.run()
            m = w.meta
            vn, cs, nc = w.vn, m["chunk_size"], m["n_chunks_per_layer"]
            width = w.H * w.hd * 2
            n_sep = len(m["newline_idx"])
            key_bytes, trailer_bytes = io_.probe_block_bytes(m)
            per = r.counter.per_kind
            self.assertEqual(per["probe"]["bytes"], w.L * (key_bytes + trailer_bytes))
            self.assertEqual(per["probe"]["preads"], w.L)
            expect_bytes = 0
            raster_k = [w.kv[li][0][0, :, w.v0:].permute(1, 0, 2).to(torch.float16).float()
                        for li in range(w.L)]
            raster_v = [w.kv[li][1][0, :, w.v0:].permute(1, 0, 2).to(torch.float16).float()
                        for li in range(w.L)]
            for row in r.selector.log:
                li = row["layer"]
                order = torch.tensor(m["order"][li])
                keep = r.selector.keep_masks[li]
                cache_k = r.ctx.cache.k[li][0, :, w.v0:].permute(1, 0, 2)
                cache_v = r.ctx.cache.v[li][0, :, w.v0:].permute(1, 0, 2)
                if row["mode"] == "fallback":
                    self.assertEqual((row["k_chunks"], row["v_chunks"]), (nc, nc))
                    expect_bytes += vn * width
                    torch.testing.assert_close(cache_k, raster_k[li][order])
                    torch.testing.assert_close(cache_v, raster_v[li][order])
                else:
                    cids = sorted({int(p) // cs for p in keep.nonzero().flatten()})
                    self.assertEqual(row["k_chunks"], len(cids))
                    self.assertLess(len(cids), nc)
                    rows = [p for c in cids for p in range(c * cs, min((c + 1) * cs, vn))]
                    expect_bytes += len(rows) * width
                    loaded = torch.zeros(vn, dtype=torch.bool)
                    loaded[rows] = True
                    torch.testing.assert_close(cache_k[loaded], raster_k[li][order][loaded])
                    torch.testing.assert_close(cache_v[loaded], raster_v[li][order][loaded])
                    self.assertTrue(bool((cache_k[~loaded] == 0).all()))
                    visible = r.bias[li][0, 0, 0, w.v0:w.v0 + vn] == 0
                    self.assertEqual(visible.tolist(), keep.tolist())
                    self.assertTrue(bool(keep[:n_sep].all()))
            self.assertEqual(per["k"]["bytes"], expect_bytes)
            self.assertEqual(per["v"]["bytes"], expect_bytes)
            stats = r.selector.stats()
            modes = [row["mode"] for row in r.selector.log]
            self.assertEqual(set(modes), {"consensus", "fallback"})
            self.assertEqual(stats["fallback_rate"], modes.count("fallback") / w.L)
            self.assertEqual(stats["k_keep"], io_.keep_count(m["n_spatial"]))
            self.assertEqual(stats["threshold"], io_.SIM_THRESHOLD)
            diag = r.selector.diagnostics()
            self.assertEqual([d["layer"] for d in diag], list(range(w.L)))
            for d, row in zip(diag, r.selector.log):
                self.assertTrue(0.0 < d["prefix_share"] < 1.0)
                self.assertTrue(0.0 <= d["order_overlap_head"] <= 1.0)
                self.assertEqual(d["order_overlap_selected"] is None,
                                 row["mode"] == "fallback")
            alt = io_.alternative_chunk_counts(
                r.selector.keep_masks, r.selector.log, m, lambda li: m["order"][li])
            self.assertEqual(len(alt["chunks"]), w.L)
            for li, mode in enumerate(modes):
                if mode == "fallback":
                    self.assertEqual(alt["chunks"][li], nc)
                else:
                    self.assertLessEqual(alt["chunks"][li], nc)

    def test_full_retention_loads_everything_and_hides_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            w = _World(td)
            r = w.run(ratio=1.0)
            m = w.meta
            width = w.H * w.hd * 2
            self.assertEqual([row["mode"] for row in r.selector.log], ["consensus"] * w.L)
            self.assertTrue(all(row["sim"] == 1.0 for row in r.selector.log))
            self.assertEqual(r.counter.per_kind["k"]["bytes"], w.L * w.vn * width)
            self.assertEqual(r.counter.per_kind["v"]["bytes"], w.L * w.vn * width)
            for li in range(w.L):
                self.assertTrue(bool(r.selector.keep_masks[li].all()))
                self.assertTrue(bool((r.bias[li] == 0).all()))
                order = torch.tensor(m["order"][li])
                raster = w.kv[li][0][0, :, w.v0:].permute(1, 0, 2).to(torch.float16).float()
                torch.testing.assert_close(
                    r.ctx.cache.k[li][0, :, w.v0:].permute(1, 0, 2), raster[order])

    def test_next_layer_prefetch_changes_only_who_reads_the_probe_sidecar(self):
        with tempfile.TemporaryDirectory() as td:
            w = _World(td)
            base = w.run()
            pool = ThreadPoolExecutor(max_workers=1)
            self.addCleanup(pool.shutdown, True)
            worker = pool.submit(threading.get_ident).result()
            threads = []
            real = io_.read_probe_block

            def spy(reader, layer, counter=None):
                threads.append((layer, threading.get_ident()))
                return real(reader, layer, counter)

            with mock.patch.object(io_, "read_probe_block", spy):
                pre = w.run(prefetch="next_layer", prefetch_executor=pool)
            self.assertEqual([t[0] for t in threads], list(range(w.L)))
            self.assertNotEqual(threads[0][1], worker)
            self.assertTrue(all(t[1] == worker for t in threads[1:]))
            self.assertEqual(base.selector.log, pre.selector.log)
            for key in ("bytes", "preads"):
                self.assertEqual(getattr(base.counter, key), getattr(pre.counter, key))
            self.assertEqual({k: (v["bytes"], v["preads"]) for k, v in base.counter.per_kind.items()},
                             {k: (v["bytes"], v["preads"]) for k, v in pre.counter.per_kind.items()})
            for li in range(w.L):
                torch.testing.assert_close(base.ctx.cache.k[li], pre.ctx.cache.k[li])
                torch.testing.assert_close(base.ctx.cache.v[li], pre.ctx.cache.v[li])
                self.assertEqual(base.selector.keep_masks[li].tolist(),
                                 pre.selector.keep_masks[li].tolist())
            self.assertEqual(sorted(base.bias), sorted(pre.bias))
            self.assertEqual([p["source"] for p in pre.selector.probe_log],
                             ["hook"] + ["prefetch"] * (w.L - 1))
            self.assertEqual(pre.selector.stats()["prefetched_probe_reads"], w.L - 1)
            self.assertIsNone(pre.selector._pending)
            with self.assertRaises(ValueError):
                io_.OfficialImpressSelector(w.runner, w.context(), prefetch="layer_stream")


if __name__ == "__main__":
    unittest.main()

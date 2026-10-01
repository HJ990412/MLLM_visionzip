"""Independent CPU oracle and actual-os-read checks for SparseVLM SSD KV25.

The oracle never calls the production score, selector, loader or mask builder.
Synthetic score tolerance is frozen at atol=rtol=1e-5.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from mmimpress.sparsevlm_ssd_core import (
    canonical_chunk_plan, exact_topk, score_visual, scoring_head_ids,
    select_raters,
)
from mmimpress.sparsevlm_ssd_store import CanonicalContext


ATOL = RTOL = 1e-5


def independent_score(q, sys_k, visual_k, suffix_k, raters, heads,
                      visual_valid=None, suffix_valid=None, causal=True):
    """Full all-query attention matrix with explicit per-head math."""
    full_k = torch.cat([sys_k, visual_k, suffix_k], dim=1)
    s, p, v, d = q.shape[1], sys_k.shape[1], visual_k.shape[1], q.shape[-1]
    probabilities = []
    for h in range(q.shape[0]):
        logits = (q[h] @ full_k[h].T) / math.sqrt(d)
        mask = torch.ones(s, full_k.shape[1], dtype=torch.bool)
        for i in range(s):
            if causal:
                mask[i, p + v + i + 1:] = False
            if visual_valid is not None:
                mask[i, p:p + v] &= visual_valid
            if suffix_valid is not None:
                mask[i, p + v:] &= suffix_valid
        probabilities.append(logits.float().masked_fill(~mask, -torch.inf).softmax(-1))
    matrix = torch.stack(probabilities)
    return matrix[heads][:, raters, p:p + v].mean(dim=1).mean(dim=0)


def independent_topk(score, structural=(), padding=()):
    ids = [i for i in range(len(score)) if i not in structural and i not in padding]
    ids.sort(key=lambda i: (-float(score[i]), i))
    return sorted(ids[:(len(ids) + 3) // 4])


def fixture(seed=1234, heads=8, visual=17, suffix=5, dim=4):
    rng = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(heads, n, dim, generator=rng) for n in (suffix, 3, visual, suffix))


class RaterTests(unittest.TestCase):
    def test_official_rater_formula_fp32(self):
        g = torch.Generator().manual_seed(1234)
        v, t = torch.randn(19, 12, generator=g), torch.randn(31, 12, generator=g)
        # Literal source formula, with the visual structural row intentionally
        # included. No local legacy/helper implementation is the oracle.
        source_u = (v.float()[None] @ t.float()[None].transpose(1, 2)).softmax(2).mean(1)
        source_ids = torch.where(source_u > source_u.mean())[1]
        output = select_raters(v, t)
        torch.testing.assert_close(output.weights, source_u[0], atol=ATOL, rtol=RTOL)
        self.assertEqual(output.ids.tolist(), source_ids.tolist())
        self.assertFalse(output.fallback)
        self.assertEqual(select_raters(v.half(), t.half()).weights.dtype, torch.float32)

    def test_equal_fallback_single_suffix_and_empty(self):
        result = select_raters(torch.zeros(9, 3), torch.zeros(8, 3))
        self.assertTrue(result.fallback)
        self.assertEqual(result.ids.tolist(), list(range(8)))
        self.assertEqual(select_raters(torch.ones(9, 3), torch.ones(1, 3)).ids.tolist(), [0])
        for n in (0,):
            with self.assertRaises(ValueError):
                select_raters(torch.ones(9, 3), torch.ones(n, 3))
        with self.assertRaises(ValueError):
            select_raters(torch.ones(0, 3), torch.ones(4, 3))

    def test_valid_suffix_and_long_generated_history_scope(self):
        v, t = torch.zeros(7, 8), torch.zeros(2049, 8)
        mask = torch.ones(2049, dtype=torch.bool)
        mask[-2:] = False
        result = select_raters(v, t, mask)
        self.assertEqual(result.ids.tolist(), list(range(2047)))
        self.assertEqual(result.weights[-2:].tolist(), [0, 0])
        for bad in (float('nan'), float('inf')):
            v[0, 0] = bad
            with self.assertRaises(ValueError):
                select_raters(v, t)


class ScoreTests(unittest.TestCase):
    def test_full_matrix_oracle_both_policies_and_query_blocks(self):
        q, system, visual, suffix = fixture()
        raters = [0, 2, 4]
        for policy, heads in [('all', list(range(8))), ('fixed_first_3', [0, 1, 2])]:
            reference = independent_score(q, system, visual, suffix, raters, heads)
            for block in (1, 2, 128):
                observed = score_visual(q, system, visual[heads], suffix, raters, policy, block)
                torch.testing.assert_close(observed, reference, atol=ATOL, rtol=RTOL)
                self.assertEqual(exact_topk(observed).tolist(), independent_topk(reference))

    def test_identical_heads_and_signal_after_head_3(self):
        q, sys_k, vk, sk = fixture(heads=1, visual=8, suffix=1)
        q, sys_k, vk, sk = [x.repeat(8, 1, 1) for x in (q, sys_k, vk, sk)]
        p = score_visual(q, sys_k, vk[:3], sk, [0], 'fixed_first_3')
        a = score_visual(q, sys_k, vk, sk, [0], 'all')
        torch.testing.assert_close(p, a, atol=ATOL, rtol=RTOL)
        self.assertEqual(exact_topk(p).tolist(), exact_topk(a).tolist())
        q.fill_(0); sys_k.fill_(0); sk.fill_(0); vk.fill_(0)
        q[3:, :, 0] = 8
        vk[3:, 7, 0] = 8
        p = score_visual(q, sys_k, vk[:3], sk, [0], 'fixed_first_3')
        a = score_visual(q, sys_k, vk, sk, [0], 'all')
        self.assertEqual(exact_topk(p).tolist(), [0, 1])
        self.assertIn(7, exact_topk(a).tolist())
        self.assertNotIn(7, exact_topk(p).tolist())

    def test_causal_prefix_denominator_padding_and_one_token_suffix(self):
        for suffix_len in (1, 9):
            q, system, visual, suffix = fixture(suffix=suffix_len)
            vv, sv = torch.ones(17, dtype=torch.bool), torch.ones(suffix_len, dtype=torch.bool)
            vv[16] = False
            if suffix_len > 1:
                sv[-1] = False
            raters = [0]
            result = score_visual(q, system, visual, suffix, raters, 'all', visual_valid_mask=vv, suffix_valid_mask=sv)
            reference = independent_score(q, system, visual, suffix, raters, list(range(8)), vv, sv)
            torch.testing.assert_close(result, reference, atol=ATOL, rtol=RTOL)
            self.assertEqual(result[16].item(), 0)
            if suffix_len > 1:
                changed = suffix.clone(); changed[:, 1:] = 123
                torch.testing.assert_close(result, score_visual(q, system, visual, changed, raters, 'all', visual_valid_mask=vv, suffix_valid_mask=sv), atol=0, rtol=0)

    def test_negative_future_mask_and_head_slice_are_detected(self):
        q, system, visual, suffix = fixture()
        q[:, 0] = 1; suffix[:, -1] = 80
        actual = score_visual(q, system, visual[:3], suffix, [0], 'fixed_first_3')
        wrong_future = independent_score(q, system, visual, suffix, [0], [0, 1, 2], causal=False)
        wrong_heads = independent_score(q, system, visual, suffix, [0], [3, 4, 5])
        self.assertFalse(torch.allclose(actual, wrong_future, atol=ATOL, rtol=RTOL))
        self.assertFalse(torch.allclose(actual, wrong_heads, atol=ATOL, rtol=RTOL))

    def test_invalid_heads_ratios_operands_and_empty_raters(self):
        for args in [('all', 8, 2), ('fixed_first_3', 2, 2), (0, 8, 8), ('probe0', 8, 8)]:
            with self.assertRaises(ValueError):
                scoring_head_ids(*args)
        q, sys_k, vk, sk = fixture()
        for raters in ([], [-1], [5], [0, 0]):
            with self.assertRaises(ValueError):
                score_visual(q, sys_k, vk, sk, raters, 'all')
        with self.assertRaises(ValueError):
            score_visual(q, sys_k, vk[:3], sk, [0], 'all')
        with self.assertRaises(ValueError):
            score_visual(q, sys_k.half(), vk, sk, [0], 'all')
        for bad in (float('nan'), float('inf')):
            q[0, 0, 0] = bad
            with self.assertRaises(ValueError):
                score_visual(q, sys_k, vk, sk, [0], 'all')


class PlanTests(unittest.TestCase):
    def test_boundary_budgets_ties_structural_padding_short_chunks(self):
        for n in (1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 349):
            for pad in (0, 3):
                scores = torch.zeros(n + 2 + pad)
                structural = [0, n + 1]
                padding = list(range(n + 2, len(scores)))
                selected = exact_topk(scores, structural, padding)
                self.assertEqual(selected.tolist(), list(range(1, (n + 3) // 4 + 1)))
                plan = canonical_chunk_plan(selected.tolist(), len(scores), structural, padding)
                self.assertEqual(plan['k'], (n + 3) // 4)
                self.assertEqual(plan['N_content'], n)
                self.assertEqual(plan['read_real_rows'], plan['k'] + plan['extra_real_rows'])
                self.assertEqual(set(plan['keep_tokens']), set(selected.tolist() + structural))
                self.assertTrue(set(selected.tolist()).issubset(plan['read_rows']))
        scores = torch.arange(65.)
        plan = canonical_chunk_plan(exact_topk(scores).tolist(), 65)
        self.assertEqual(plan['eof_short_chunk_rows'], 1)
        self.assertEqual(plan['chunk_runs'], [[0, 65]])

    def test_invalid_mapping_nan_inf_empty_budget_and_bad_chunks(self):
        for values in ([float('nan')], [float('inf')], []):
            with self.assertRaises(ValueError):
                exact_topk(torch.tensor(values))
        with self.assertRaises(ValueError):
            exact_topk(torch.ones(2), [0, 1])
        for selected in ([0, 0], [-1], [4], [], [0, 1]):
            with self.assertRaises(ValueError):
                canonical_chunk_plan(selected, 4)
        with self.assertRaises(ValueError):
            canonical_chunk_plan([0], 4, structural_ids=[0])
        with self.assertRaises(ValueError):
            canonical_chunk_plan([0], 4, chunk_size=32)


def write_fixture(path, visual=129, heads=8, dim=4, layers=2, structural=(63, 128), padding=()):
    """Independent byte writer (does not use production persistence)."""
    meta = dict(v_token_start=2, v_token_num=visual, prefix_len=2+visual,
                num_heads=heads, num_layers=layers, head_dim=dim,
                dtype='float16', probe_heads=3, chunk_size=64,
                n_spatial=visual-len(structural)-len(padding),
                newline_idx=list(structural), padding_idx=list(padding),
                prefix_input_ids=[1, 2], physical_layout='raster', reordered=False)
    (path / 'meta.json').write_text(json.dumps(meta))
    keys, values = [], []
    rng = np.random.default_rng(1234)
    for layer in range(layers):
        ld = path / f'layer_{layer:02d}'; ld.mkdir()
        k = rng.standard_normal((visual, heads, dim)).astype(np.float16)
        v = rng.standard_normal((visual, heads, dim)).astype(np.float16)
        (ld / 'k.bin').write_bytes(k.tobytes()); (ld / 'v.bin').write_bytes(v.tobytes())
        (ld / 'probe_k.bin').write_bytes(k[:, :3].copy().tobytes())
        keys.append(k); values.append(v)
    np.stack([np.stack(keys)[:, structural], np.stack(values)[:, structural]]).astype(np.float16).tofile(path / 'sep_kv.bin')
    torch.save({'k': torch.zeros(layers, heads, 2, dim, dtype=torch.float16),
                'v': torch.zeros(layers, heads, 2, dim, dtype=torch.float16)}, path / 'sys_kv.pt')
    torch.save(torch.zeros(visual, 9, dtype=torch.float16), path / 'v_hidden.pt')
    return meta, keys, values


class StorageTests(unittest.TestCase):
    def test_bits_actual_events_and_allhead_no_probe_or_k_reread(self):
        for policy in ('all', 'fixed_first_3'):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                meta, keys, values = write_fixture(path)
                if policy == 'all':
                    for sidecar in path.glob('layer_*/probe_k.bin'):
                        sidecar.unlink()
                context = CanonicalContext(path, policy, validate_provenance=False)
                actual, opened = [], []
                original_pread, original_open = os.pread, os.open
                def observed_pread(fd, count, offset):
                    blob = original_pread(fd, count, offset)
                    actual.append((offset, len(blob)))
                    return blob
                def observed_open(file, *args, **kwargs):
                    opened.append(str(file))
                    return original_open(file, *args, **kwargs)
                with patch('os.pread', side_effect=observed_pread), patch('os.open', side_effect=observed_open):
                    with context.request() as request:
                        for layer in range(2):
                            scoring = request.read_scoring_keys(layer)
                            heads = list(range(8)) if policy == 'all' else [0, 1, 2]
                            self.assertTrue(torch.equal(scoring, torch.from_numpy(keys[layer][:, heads].copy())))
                            selected = exact_topk(torch.arange(129.), [63, 128])
                            plan = canonical_chunk_plan(selected.tolist(), 129, [63, 128])
                            payload = request.read_selected(layer, plan, scoring)
                            self.assertTrue(torch.equal(payload.keys, torch.from_numpy(keys[layer][payload.rows].copy())))
                            self.assertTrue(torch.equal(payload.values, torch.from_numpy(values[layer][payload.rows].copy())))
                            self.assertEqual(payload.stats['k'], 32)
                            with self.assertRaises(RuntimeError):
                                request.read_scoring_keys(layer)
                        report = request.summary()
                self.assertEqual(report['bytes'], sum(n for _, n in actual))
                self.assertEqual(report['preads'], len(actual))
                self.assertEqual(report['bytes'], sum(x['bytes'] for x in report['per_kind'].values()))
                self.assertEqual(report['bytes'], sum(e['returned_bytes'] for e in report['events']))
                if policy == 'all':
                    self.assertTrue(all('probe_k' not in p for p in opened))
                    self.assertEqual(report['per_kind']['selected_k']['bytes'], 0)
                    self.assertEqual(report['per_kind']['scoring_probe_k']['bytes'], 0)
                    self.assertEqual(report['per_kind']['scoring_full_k']['preads'], 2)
                    self.assertEqual(report['per_kind']['selected_v']['bytes'], 2 * 64 * 8 * 4 * 2)
                else:
                    self.assertEqual(report['per_kind']['scoring_full_k']['bytes'], 0)
                    self.assertGreater(report['duplicated_k_bytes'], 0)
                self.assertIsNone(request.structural)
                self.assertIsNone(context._active)
                context.close()

    def test_activation_hash_manifest_and_method_specific_probe_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory); write_fixture(path)
            required = [p for p in path.rglob("*") if p.is_file() and p.name != "probe_k.bin"]
            hashes = {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest() for p in required}
            context = CanonicalContext(path, 'all', validate_provenance=False, expected_hashes=hashes)
            self.assertEqual(context.activation['hash_verification'], 'PASS')
            self.assertEqual(context.activation['provenance_validation'], 'SYNTHETIC_BYPASS')
            self.assertEqual(context.activation['hash_verified_bytes'], sum(p.stat().st_size for p in required))
            for p in path.glob('layer_*/probe_k.bin'):
                hashes[str(p.relative_to(path))] = hashlib.sha256(p.read_bytes()).hexdigest()
            probe = CanonicalContext(path, 'fixed_first_3', validate_provenance=False, expected_hashes=hashes)
            self.assertEqual(probe.activation['hash_verification'], 'PASS')
            hashes['v_hidden.pt'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                CanonicalContext(path, 'fixed_first_3', validate_provenance=False, expected_hashes=hashes)
            with self.assertRaisesRegex(ValueError, 'every required file'):
                CanonicalContext(path, 'all', validate_provenance=False, expected_hashes={})

    def test_nonadjacent_ranges_coalesce_only_adjacent_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            _, keys, values = write_fixture(path, visual=257, structural=(), padding=(256,))
            context = CanonicalContext(path, 'all', validate_provenance=False)
            selected = list(range(32)) + list(range(224, 256))
            plan = canonical_chunk_plan(selected, 257, padding_ids=[256])
            self.assertEqual(plan['chunk_runs'], [[0, 64], [192, 256]])
            with context.request() as request:
                scoring = request.read_scoring_keys(0)
                payload = request.read_selected(0, plan, scoring)
                report = request.summary()
                self.assertEqual(report['per_kind']['selected_v']['preads'], 2)
                self.assertEqual(report['per_kind']['selected_v']['bytes'], 128 * 8 * 4 * 2)
                self.assertEqual(report['per_kind']['structural']['preads'], 0)
                self.assertTrue(torch.equal(payload.keys, torch.from_numpy(keys[0][selected])))
                self.assertTrue(torch.equal(payload.values, torch.from_numpy(values[0][selected])))
                self.assertEqual(payload.stats['read_padding_rows'], 0)

    def test_mapping_tamper_wrong_provenance_and_request_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory); meta, _, _ = write_fixture(path)
            with self.assertRaises(ValueError):
                CanonicalContext(path, 'all')
            context = CanonicalContext(path, 'all', validate_provenance=False)
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                with context.request() as request:
                    request.read_scoring_keys(0)
                    with self.assertRaises(RuntimeError):
                        context.request()
                    with self.assertRaises(RuntimeError):
                        context.drop_cache()
                    raise RuntimeError('injected')
            self.assertIsNone(context._active)
            with context.request() as request:
                scoring = request.read_scoring_keys(0)
                plan = canonical_chunk_plan(exact_topk(torch.arange(129.), [63, 128]).tolist(), 129, [63, 128])
                plan['selected_chunks'] = [0]
                with self.assertRaises(ValueError):
                    request.read_selected(0, plan, scoring)
            meta['order'] = list(reversed(range(129)))
            (path / 'meta.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                CanonicalContext(path, 'all', validate_provenance=False)

    def test_eof_short_read_recorded_before_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory); write_fixture(path)
            context = CanonicalContext(path, 'all', validate_provenance=False)
            # A failure after activation must remain visible in the event log.
            with (path / 'layer_00/k.bin').open('r+b') as handle:
                handle.truncate(1)
            with context.request() as request:
                with self.assertRaises(IOError):
                    request.read_scoring_keys(0)
                report = request.summary()
                self.assertEqual(report['bytes'], 1)
                self.assertEqual(report['preads'], 1)
                self.assertEqual(report['events'][0]['returned_bytes'], 1)

    def test_finite_unselected_sentinel_does_not_change_answer_attention(self):
        # Selection has already been frozen. This checks a real softmax with
        # dense original positions, not just zero payload equality.
        q, sys_k, keys, vals = fixture(visual=65, suffix=65)
        selected = independent_topk(torch.arange(65.))
        keep = torch.tensor([i in selected or i == 64 for i in range(65)])
        logits = q[:, :2] @ keys.transpose(1, 2)
        expected = logits.masked_fill(~keep, -torch.inf).softmax(-1) @ vals
        changed_k, changed_v = keys.clone(), vals.clone()
        changed_k[:, ~keep] = 123
        changed_v[:, ~keep] = -99
        actual = (q[:, :2] @ changed_k.transpose(1, 2)).masked_fill(~keep, -torch.inf).softmax(-1) @ changed_v
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()

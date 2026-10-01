"""Independent review fixtures for scoped adapter state, sentinel and padding."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from transformers.models.llama import modeling_llama as ml

from mmimpress.sparsevlm_ssd_attention import SparseVLMSSDAttention, new_dense_cache
from tests.test_sparsevlm_ssd_attention import Reader, fixture


class AdapterReviewTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_padding_denominator_mask_dense_shape_and_sentinel_are_real(self):
        for policy in ('all', 'fixed_first_3'):
            runner, ctx = fixture(policy)
            ctx.meta['padding_idx'] = [8]
            ctx.meta['n_spatial'] = 7
            # A padding K that would dominate incorrectly unmasked logits.
            ctx.keys[:, 8] = 500
            ctx.values[:, 8] = -700
            cache = new_dense_cache(ctx, 'cpu', torch.float32)
            seen = []
            def observer(row):
                li = row['layer']; keep = row['keep']
                self.assertEqual(row['key'].shape, (1, 4, 12, 8))
                self.assertEqual(row['mask'].shape, (1, 1, 1, 12))
                self.assertFalse(bool(keep[8]))
                self.assertTrue(bool(keep[3]))
                self.assertEqual(row['scores'][8].item(), 0)
                self.assertEqual(len(row['selected']), 2)
                native_k = ctx.keys[li].float().permute(1, 0, 2)
                actual_k = row['key'][0, :, 2:11]
                self.assertTrue(torch.equal(actual_k[:, keep], native_k[:, keep]))
                self.assertTrue(torch.all(actual_k[:, ~keep] == 1234))
                self.assertTrue(torch.all(row['value'][0, :, 2:11][:, ~keep] == 1234))
                self.assertTrue(torch.isneginf(row['mask'][..., 2:11][..., ~keep]).all())
                seen.append(li)
            method = 'sparsevlm_ssd_kv25_' + ('allhead' if policy == 'all' else 'probe3')
            adapter = SparseVLMSSDAttention(runner, ctx, Reader(ctx), cache, [0], method,
                                             observer=observer, sentinel=1234)
            with torch.inference_mode(), adapter:
                runner.model(input_ids=torch.tensor([[4]]), position_ids=torch.tensor([[11]]),
                             cache_position=torch.tensor([11]), past_key_values=cache,
                             attention_mask=torch.ones(1, 12, dtype=torch.long))
            self.assertEqual(seen, [0, 1])

    def test_known_legacy_patch_rejects_active_bias_and_leaves_global_unchanged(self):
        from mmimpress import serve
        runner, ctx = fixture()
        original = ml.eager_attention_forward
        with patch.dict(serve.BIAS, {0: torch.zeros(1)}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'legacy BIAS'):
                SparseVLMSSDAttention(runner, ctx, Reader(ctx), None, [0], 'sparsevlm_ssd_kv25_allhead')
        self.assertIs(ml.eager_attention_forward, original)
        self.assertFalse(serve.BIAS)

    def test_partial_enter_exception_restores_completed_layer_and_existing_owner(self):
        runner, ctx = fixture()
        original = [layer.self_attn.forward for layer in runner.layers]
        owner = object()
        runner.layers[1].self_attn._sparsevlm_ssd_adapter = owner
        adapter = SparseVLMSSDAttention(runner, ctx, Reader(ctx), None, [0], 'sparsevlm_ssd_kv25_allhead')
        with self.assertRaisesRegex(RuntimeError, 'active request'):
            adapter.__enter__()
        self.assertEqual(original, [layer.self_attn.forward for layer in runner.layers])
        self.assertFalse(runner.layers[0].self_attn.q_proj._forward_hooks)
        self.assertFalse(hasattr(runner.layers[0].self_attn, '_sparsevlm_ssd_adapter'))
        self.assertIs(runner.layers[1].self_attn._sparsevlm_ssd_adapter, owner)
        del runner.layers[1].self_attn._sparsevlm_ssd_adapter

    def test_unknown_global_patch_fails_closed(self):
        runner, ctx = fixture()
        with patch.object(ml, 'eager_attention_forward', lambda *a, **k: None):
            with self.assertRaisesRegex(RuntimeError, 'unrecognized global'):
                SparseVLMSSDAttention(runner, ctx, Reader(ctx), None, [0], 'sparsevlm_ssd_kv25_allhead')


if __name__ == '__main__':
    unittest.main()

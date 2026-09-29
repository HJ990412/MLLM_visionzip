"""CPU checks for Qwen2.5-VL image-only received-attention capture."""

from __future__ import annotations

import math
import types
import unittest

import torch

from mmimpress.qwen25.vision import (
    VisionScoreCapture,
    merge_window_scores,
    received_attention_scores,
)


class ReceivedAttentionTests(unittest.TestCase):
    def test_blocked_score_matches_dense_masked_reference(self):
        torch.manual_seed(1234)
        q = torch.randn(11, 3, 8)
        k = torch.randn(11, 3, 8)
        boundaries = torch.tensor([0, 4, 7, 11])
        full = torch.matmul(q.transpose(0, 1), k.transpose(0, 1).transpose(-1, -2)) / math.sqrt(8)
        mask = torch.full((11, 11), float("-inf"))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            mask[start:end, start:end] = 0
        dense = torch.softmax(full + mask, dim=-1, dtype=torch.float32).mean(dim=0).sum(dim=0)
        for block_size in (1, 2, 5, 128):
            actual = received_attention_scores(q, k, boundaries, query_block_size=block_size)
            torch.testing.assert_close(actual, dense, atol=1e-6, rtol=1e-6)
        # Each frame contributes exactly one unit of received attention per query.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            self.assertAlmostEqual(actual[start:end].sum().item(), float(end - start), places=5)

    def test_merge_groups_then_reverse_windows(self):
        patch_scores = torch.arange(32, dtype=torch.float32)
        window_index = torch.tensor([0, 1, 4, 5, 2, 3, 6, 7])
        expected_window = patch_scores.view(8, 4).mean(dim=-1)
        expected = expected_window[torch.argsort(window_index)]
        self.assertTrue(torch.equal(merge_window_scores(patch_scores, window_index, 2), expected))
        with self.assertRaises(ValueError):
            merge_window_scores(patch_scores, torch.tensor([0, 0, 1, 2, 3, 4, 5, 6]), 2)


class StockVisionCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VisionTransformerPretrainedModel,
                apply_rotary_pos_emb_vision,
            )
        except ImportError as exc:
            raise unittest.SkipTest(f"Qwen2.5-VL Transformers unavailable: {exc}")
        cls.config_type = Qwen2_5_VLVisionConfig
        cls.visual_type = Qwen2_5_VisionTransformerPretrainedModel
        cls.apply_rope = staticmethod(apply_rotary_pos_emb_vision)

    def test_single_stock_forward_capture_equals_dense_visionzip_formula(self):
        torch.manual_seed(1234)
        config = self.config_type(
            depth=2,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            patch_size=2,
            spatial_merge_size=2,
            temporal_patch_size=2,
            window_size=8,
            out_hidden_size=32,
            fullatt_block_indexes=[1],
        )
        config._attn_implementation = "sdpa"
        visual = self.visual_type(config).eval()
        model = types.SimpleNamespace(visual=visual, training=False)
        grid = torch.tensor([[1, 4, 8]])
        pixels = torch.randn(32, 3 * 2 * 2 * 2)
        attn = visual.blocks[-1].attn
        observed = {}

        def pre_hook(module, args, kwargs):
            observed["position_embeddings"] = kwargs["position_embeddings"]

        def qkv_hook(module, args, output):
            observed["qkv"] = output.detach()

        pre_handle = attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
        qkv_handle = attn.qkv.register_forward_hook(qkv_hook)
        before = (
            len(visual._forward_pre_hooks),
            len(visual._forward_hooks),
            len(attn._forward_pre_hooks),
            len(attn.qkv._forward_hooks),
        )
        try:
            with torch.inference_mode():
                with VisionScoreCapture(model, query_block_size=3) as capture:
                    merged = visual(pixels, grid)
                normal = visual(pixels, grid)
            self.assertTrue(torch.equal(merged, normal))
            self.assertEqual(capture.call_count, 1)
            self.assertEqual(capture.geometry["premerge_patch_count"], 32)
            self.assertEqual(capture.geometry["merged_visual_token_count"], 8)
            self.assertEqual(capture.geometry["score_count"], 8)
            self.assertEqual(capture.geometry["last_vision_block_index"], 1)
            self.assertEqual(capture.geometry["image_grid_thw"], [[1, 4, 8]])
            self.assertGreaterEqual(capture.score_seconds, 0)
            self.assertEqual(
                before,
                (
                    len(visual._forward_pre_hooks),
                    len(visual._forward_hooks),
                    len(attn._forward_pre_hooks),
                    len(attn.qkv._forward_hooks),
                ),
            )

            q, k, _ = observed["qkv"].reshape(32, 3, 4, 8).permute(1, 0, 2, 3).unbind(0)
            cos, sin = observed["position_embeddings"]
            q, k = self.apply_rope(q, k, cos, sin)
            attention = torch.matmul(q.transpose(0, 1), k.transpose(0, 1).transpose(-1, -2)) / math.sqrt(8)
            probabilities = torch.softmax(attention, dim=-1, dtype=torch.float32).to(q.dtype)
            reference = probabilities.mean(dim=0).sum(dim=0).reshape(8, 4).mean(dim=-1)
            reference = reference[torch.argsort(visual.get_window_index(grid)[0])]
            torch.testing.assert_close(capture.scores, reference, atol=1e-6, rtol=1e-6)
        finally:
            pre_handle.remove()
            qkv_handle.remove()

    def test_rejects_non_full_last_block(self):
        config = self.config_type(
            depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
            patch_size=2, spatial_merge_size=2, temporal_patch_size=2,
            window_size=8, out_hidden_size=32, fullatt_block_indexes=[0],
        )
        visual = self.visual_type(config).eval()
        with self.assertRaisesRegex(ValueError, "full-attention"):
            VisionScoreCapture(types.SimpleNamespace(visual=visual, training=False))


if __name__ == "__main__":
    unittest.main()

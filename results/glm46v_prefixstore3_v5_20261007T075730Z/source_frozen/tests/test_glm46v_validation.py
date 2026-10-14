"""Independent position/attention fixtures used before checkpoint quality."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

_path = Path(__file__).resolve().parents[1] / "scripts/120_validate_glm46v_prefixstore3.py"
_spec = importlib.util.spec_from_file_location("glm_independent_validation", _path)
validation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validation)


class GlmValidationTests(unittest.TestCase):
    class LiteralTokenizer:
        pieces = {1: "<|begin_of_box|>", 2: "Crown", 3: "<|end_of_box|>",
                  4: "<|user|>", 5: "</think>", 6: "\n", 7: " "}

        def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
            assert skip_special_tokens is False
            assert clean_up_tokenization_spaces is False
            return "".join(self.pieces[index] for index in ids)

    def fixture_result(self, generated, body_ids, body):
        return {"generated_token_ids": generated, "history_token_ids": body_ids,
                "history_text": body, "prediction": "Crown",
                "history_native_policy": "actual-generated-body/terminal-EOS-only-v1"}

    def test_native_actual_history_retains_box_controls_and_whitespace(self):
        generated = [6, 1, 2, 3, 7, 4]
        result = self.fixture_result(generated, generated[:-1], "\n<|begin_of_box|>Crown<|end_of_box|> ")
        body = validation.independent_native_history(result, self.LiteralTokenizer(), {4})
        self.assertEqual(body, "\n<|begin_of_box|>Crown<|end_of_box|> ")
        self.assertEqual(generated, [6, 1, 2, 3, 7, 4])

    def test_native_actual_history_preserves_internal_registered_controls(self):
        generated = [1, 5, 4, 2, 3, 4]
        literal = "<|begin_of_box|></think><|user|>Crown<|end_of_box|>"
        result = self.fixture_result(generated, generated[:-1], literal)
        self.assertEqual(validation.independent_native_history(result, self.LiteralTokenizer(), {4}), literal)

    def test_parsed_answer_history_is_rejected_even_when_prediction_is_valid(self):
        result = self.fixture_result([1, 2, 3, 4], [1, 2, 3], "Crown")
        with self.assertRaisesRegex(AssertionError, "literal actual"):
            validation.independent_native_history(result, self.LiteralTokenizer(), {4})

    def test_changed_native_body_ids_and_history_policy_rejected(self):
        result = self.fixture_result([1, 2, 3, 4], [2], "Crown")
        with self.assertRaisesRegex(AssertionError, "non-terminal"):
            validation.independent_native_history(result, self.LiteralTokenizer(), {4})
        result = self.fixture_result([1, 2, 3], [1, 2, 3], "<|begin_of_box|>Crown<|end_of_box|>")
        result["history_native_policy"] = "parsed-final-answer"
        with self.assertRaisesRegex(AssertionError, "frozen v5"):
            validation.independent_native_history(result, self.LiteralTokenizer(), {4})

    def test_geometry_suffix_positions_and_offset(self):
        geometry = {"image_grid_thw": [[1, 4, 6]], "merge_size": 2}
        positions, offset = validation.reference_positions(list(range(12)), list(range(3, 9)), geometry)
        self.assertEqual(positions[:, 0, 3:9].tolist(),
                         [[3, 3, 3, 3, 3, 3], [3, 3, 3, 4, 4, 4], [3, 4, 5, 3, 4, 5]])
        self.assertEqual(positions[:, 0, 9:].tolist(), [[6, 7, 8]] * 3)
        self.assertEqual(offset, [[-3]])
        with self.assertRaises(AssertionError):
            validation.reference_positions(list(range(12)), [3, 5, 6], geometry)

    def test_original_N_budget_rank_and_native_row_bits(self):
        # k=ceil(5/4)=2; it must not be recomputed from 2 selected rows.
        tensor = torch.arange(7 * 2 * 8, dtype=torch.float32).reshape(1, 2, 7, 8).bfloat16()
        capture = SimpleNamespace(content_indices=[1, 2, 3, 4, 5], prefix_ids=list(range(7)),
                                  positions=torch.arange(7).reshape(1, 1, 7).expand(3, -1, -1),
                                  scores=torch.tensor([1., 3., 3., 0., 2.]), layers=[(tensor, tensor + 1)],
                                  rope_offset=[[0]])
        reference = validation.independent_memory(capture, "prefix25")
        self.assertEqual(reference.selected_content_original, (1, 2))
        self.assertEqual(reference.logical_indices, (0, 2, 3, 6))
        self.assertTrue(torch.equal(reference.layers[0][0], tensor[:, :, [0, 2, 3, 6], :]))

    def test_dense_compact_fp64_all_layer_observation(self):
        torch.manual_seed(1234)
        q, k, v = torch.randn(1, 4, 3, 8), torch.randn(1, 2, 6, 8), torch.randn(1, 2, 6, 8)
        visible = torch.ones(1, 1, 3, 6, dtype=torch.bool)
        visible[:, :, 0, 4:] = False
        visible[:, :, 1, 5:] = False
        observer = validation.NativeAttentionObserver((0, 2, 5), 7, expected_layers=1)
        output = observer(q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), attn_mask=visible, scale=8**-.5)
        native = torch.nn.functional.scaled_dot_product_attention(
            q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), attn_mask=visible, scale=8**-.5)
        self.assertTrue(torch.equal(output, native))
        self.assertTrue(observer.rows[0]["pass"])
        self.assertLessEqual(observer.rows[0]["max_abs_error_fp64"], 1e-12)

    def test_future_suffix_visibility_is_rejected(self):
        q, k, v = torch.ones(1, 2, 3, 8), torch.ones(1, 2, 6, 8), torch.ones(1, 2, 6, 8)
        observer = validation.NativeAttentionObserver((0, 2, 5), 7, expected_layers=1)
        with self.assertRaisesRegex(AssertionError, "future visibility"):
            observer(q, k, v, attn_mask=torch.ones(1, 1, 3, 6, dtype=torch.bool))

    def test_upper_left_causal_mask_loses_cache_offset_is_rejected(self):
        q, k, v = torch.ones(1, 2, 3, 8), torch.ones(1, 2, 6, 8), torch.ones(1, 2, 6, 8)
        observer = validation.NativeAttentionObserver((0, 2, 5), 7, expected_layers=1)
        with self.assertRaisesRegex(AssertionError, "lost prefix offset"):
            observer(q, k, v, is_causal=True)

    def test_actual_native_adjacent_pair_partial_rope_keeps_tail_bits(self):
        from transformers.models.glm4v.modeling_glm4v import apply_rotary_pos_emb
        q = torch.linspace(-2, 2, 128).bfloat16().reshape(1, 1, 1, 128)
        k = (q + .125).clone()
        # Native embedding duplicates 32 frequencies to 64 positions; its
        # decoder then repeat_interleaves the first 32 for adjacent-pair RoPE.
        frequency = torch.linspace(.1, 1.5, 32)
        cos = torch.cat([frequency.cos(), frequency.cos()]).bfloat16().reshape(1, 1, 64)
        sin = torch.cat([frequency.sin(), frequency.sin()]).bfloat16().reshape(1, 1, 64)
        actual_q, actual_k = apply_rotary_pos_emb(q, k, cos, sin)
        for native, original in ((actual_q, q), (actual_k, k)):
            expected = original.clone()
            for pair in range(32):
                even, odd = original[..., pair * 2], original[..., pair * 2 + 1]
                c, s = cos[0, 0, pair], sin[0, 0, pair]
                expected[..., pair * 2] = even * c + (-odd) * s
                expected[..., pair * 2 + 1] = odd * c + even * s
            self.assertTrue(torch.equal(native, expected))
            self.assertTrue(torch.equal(native[..., 64:], original[..., 64:]))


if __name__ == "__main__":
    unittest.main()

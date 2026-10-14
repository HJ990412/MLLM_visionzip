"""CPU contracts for MPIC selective recomputation and raw SSD reads.

These tests intentionally use tiny synthetic tensors.  They validate logical
positions, indexed K/V replacement, causal visibility, post-RoPE relocation,
and DynamicCache seeding without loading a model or touching CUDA.
"""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mmimpress.mpic import (
    MPICRawReader,
    MPICSelectivePrefill,
    assemble_linked_kv,
    build_active_row_plan,
    build_causal_mask,
    relocate_post_rope_keys,
    seed_dynamic_cache,
)
from mmimpress.store import IOCounter


def _positions(values) -> list[int]:
    """Normalise a plan field without prescribing its public container type."""
    if isinstance(values, torch.Tensor):
        return [int(value) for value in values.detach().cpu().tolist()]
    return [int(value) for value in values]


def _hf_tensor(rows: int, offset: float, *, heads: int = 2,
               head_dim: int = 3) -> torch.Tensor:
    """Deterministic ``(batch, heads, rows, head_dim)`` FP32 tensor."""
    values = torch.arange(heads * rows * head_dim, dtype=torch.float32)
    return values.view(1, heads, rows, head_dim) + offset


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(values: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor) -> torch.Tensor:
    """Small independent reference for the Llama rotate-half convention."""
    return values * cos.unsqueeze(1) + _rotate_half(values) * sin.unsqueeze(1)


def _rope_tables(angles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Llama repeats the half-dimensional frequencies in the second half.
    phases = torch.cat((angles, angles), dim=-1).unsqueeze(0)
    return phases.cos(), phases.sin()


class ActiveRowPlanTests(unittest.TestCase):
    def assertPlan(self, plan, *, active, image, text, reused):
        self.assertEqual(_positions(plan.active_positions), active)
        self.assertEqual(_positions(plan.image_positions), image)
        self.assertEqual(_positions(plan.text_positions), text)
        self.assertEqual(_positions(plan.reused_image_positions), reused)

    def test_k_zero_reuses_the_complete_image_and_recomputes_all_text(self):
        plan = build_active_row_plan(
            total_tokens=10, visual_start=2, visual_tokens=5,
            k_recompute=0)

        self.assertPlan(
            plan,
            active=[0, 1, 7, 8, 9],
            image=[],
            text=[0, 1, 7, 8, 9],
            reused=[2, 3, 4, 5, 6],
        )

    def test_k_n_recomputes_every_image_row_at_its_logical_position(self):
        plan = build_active_row_plan(
            total_tokens=9, visual_start=2, visual_tokens=5,
            k_recompute=5)

        self.assertPlan(
            plan,
            active=list(range(9)),
            image=[2, 3, 4, 5, 6],
            text=[0, 1, 7, 8],
            reused=[],
        )

    def test_k_32_means_leading_rows_not_percent_or_chunks(self):
        plan = build_active_row_plan(
            total_tokens=47, visual_start=3, visual_tokens=40,
            k_recompute=32)

        image = list(range(3, 35))
        text = [0, 1, 2, 43, 44, 45, 46]
        reused = list(range(35, 43))
        self.assertPlan(
            plan,
            active=sorted(text + image),
            image=image,
            text=text,
            reused=reused,
        )
        # Recomputed and reused rows partition the original image span; no
        # image context is pruned by MPIC-k.
        self.assertEqual(sorted(image + reused), list(range(3, 43)))
        self.assertEqual(len(image), 32)


class LinkedKVAssemblyTests(unittest.TestCase):
    @staticmethod
    def _expected(cached: torch.Tensor, active: torch.Tensor,
                  active_positions: list[int], visual_start: int,
                  total_tokens: int, dummy: float) -> torch.Tensor:
        result = torch.full(
            (cached.shape[0], cached.shape[1], total_tokens, cached.shape[3]),
            dummy, dtype=cached.dtype)
        visual_end = visual_start + cached.shape[2]
        result[:, :, visual_start:visual_end, :] = cached
        indices = torch.tensor(active_positions, dtype=torch.long)
        result.index_copy_(2, indices, active)
        return result

    def _assemble_for_k(self, k_recompute: int):
        total, visual_start, visual_tokens = 9, 2, 5
        plan = build_active_row_plan(
            total, visual_start, visual_tokens, k_recompute)
        positions = _positions(plan.active_positions)
        cached_k = _hf_tensor(visual_tokens, 100.0)
        cached_v = _hf_tensor(visual_tokens, 500.0)
        active_k = _hf_tensor(len(positions), 1000.0)
        active_v = _hf_tensor(len(positions), 5000.0)
        source_k, source_v = cached_k.clone(), cached_v.clone()

        full_k, full_v = assemble_linked_kv(
            cached_k, cached_v, active_k, active_v, positions,
            visual_start, total, dummy_value=-777.0)
        expected_k = self._expected(
            cached_k, active_k, positions, visual_start, total, -777.0)
        expected_v = self._expected(
            cached_v, active_v, positions, visual_start, total, -777.0)
        return (plan, full_k, full_v, expected_k, expected_v,
                cached_k, cached_v, source_k, source_v)

    def test_k_zero_and_k_n_retain_exact_full_logical_context(self):
        for k_recompute in (0, 5):
            with self.subTest(k_recompute=k_recompute):
                (plan, full_k, full_v, expected_k, expected_v,
                 cached_k, cached_v, source_k, source_v) = \
                    self._assemble_for_k(k_recompute)

                self.assertEqual(tuple(full_k.shape), (1, 2, 9, 3))
                self.assertEqual(tuple(full_v.shape), (1, 2, 9, 3))
                self.assertTrue(torch.equal(full_k, expected_k))
                self.assertTrue(torch.equal(full_v, expected_v))
                self.assertFalse((full_k == -777.0).any().item())
                self.assertFalse((full_v == -777.0).any().item())
                self.assertTrue(torch.equal(cached_k, source_k))
                self.assertTrue(torch.equal(cached_v, source_v))

                image = _positions(plan.image_positions)
                reused = _positions(plan.reused_image_positions)
                self.assertEqual(
                    sorted(image + reused), list(range(2, 7)))

    def test_indexed_replacement_preserves_every_nonselected_image_row(self):
        (plan, full_k, full_v, expected_k, expected_v,
         cached_k, cached_v, _, _) = self._assemble_for_k(2)

        self.assertTrue(torch.equal(full_k, expected_k))
        self.assertTrue(torch.equal(full_v, expected_v))
        for logical_position in _positions(plan.reused_image_positions):
            local = logical_position - 2
            self.assertTrue(torch.equal(
                full_k[:, :, logical_position, :],
                cached_k[:, :, local, :]))
            self.assertTrue(torch.equal(
                full_v[:, :, logical_position, :],
                cached_v[:, :, local, :]))

        positions = _positions(plan.active_positions)
        active_k = _hf_tensor(len(positions), 1000.0)
        active_v = _hf_tensor(len(positions), 5000.0)
        for logical_position in _positions(plan.image_positions):
            active_index = positions.index(logical_position)
            self.assertTrue(torch.equal(
                full_k[:, :, logical_position, :],
                active_k[:, :, active_index, :]))
            self.assertTrue(torch.equal(
                full_v[:, :, logical_position, :],
                active_v[:, :, active_index, :]))

    def test_dummy_sentinel_is_fully_overwritten_before_attention(self):
        total, visual_start, visual_tokens = 8, 2, 4
        plan = build_active_row_plan(total, visual_start, visual_tokens, 2)
        positions = _positions(plan.active_positions)
        cached_k = _hf_tensor(visual_tokens, 100.0)
        cached_v = _hf_tensor(visual_tokens, 500.0)
        active_k = _hf_tensor(len(positions), 1000.0)
        active_v = _hf_tensor(len(positions), 5000.0)

        low_k, low_v = assemble_linked_kv(
            cached_k, cached_v, active_k, active_v, positions,
            visual_start, total, dummy_value=-12345.0)
        high_k, high_v = assemble_linked_kv(
            cached_k, cached_v, active_k, active_v, positions,
            visual_start, total, dummy_value=23456.0)

        self.assertTrue(torch.equal(low_k, high_k))
        self.assertTrue(torch.equal(low_v, high_v))
        self.assertFalse((low_k == -12345.0).any().item())
        self.assertFalse((high_k == 23456.0).any().item())
        self.assertEqual(full_length := low_k.shape[-2], total)
        self.assertNotEqual(full_length, total + len(positions))

    def test_k_32_replaces_exactly_32_rows_and_keeps_the_reused_tail(self):
        total, visual_start, visual_tokens = 38, 2, 34
        plan = build_active_row_plan(
            total, visual_start, visual_tokens, k_recompute=32)
        positions = _positions(plan.active_positions)
        cached_k = _hf_tensor(visual_tokens, 1000.0, heads=1, head_dim=1)
        cached_v = _hf_tensor(visual_tokens, 3000.0, heads=1, head_dim=1)
        active_k = torch.tensor(
            positions, dtype=torch.float32).view(1, 1, -1, 1) + 2000.0
        active_v = torch.tensor(
            positions, dtype=torch.float32).view(1, 1, -1, 1) + 4000.0

        full_k, full_v = assemble_linked_kv(
            cached_k, cached_v, active_k, active_v, positions,
            visual_start, total, dummy_value=-1.0)

        self.assertEqual(tuple(full_k.shape), (1, 1, total, 1))
        self.assertEqual(tuple(full_v.shape), (1, 1, total, 1))
        self.assertEqual(_positions(plan.image_positions), list(range(2, 34)))
        self.assertEqual(_positions(plan.reused_image_positions), [34, 35])
        torch.testing.assert_close(
            full_k[0, 0, 2:34, 0],
            torch.arange(2, 34, dtype=torch.float32) + 2000.0,
            rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            full_v[0, 0, 2:34, 0],
            torch.arange(2, 34, dtype=torch.float32) + 4000.0,
            rtol=0.0, atol=0.0)
        self.assertTrue(torch.equal(
            full_k[:, :, 34:36, :], cached_k[:, :, 32:34, :]))
        self.assertTrue(torch.equal(
            full_v[:, :, 34:36, :], cached_v[:, :, 32:34, :]))

    def test_separate_requests_cannot_mutate_source_or_prior_result(self):
        total, visual_start, visual_tokens = 8, 2, 4
        plan = build_active_row_plan(total, visual_start, visual_tokens, 2)
        positions = _positions(plan.active_positions)
        cached_k = _hf_tensor(visual_tokens, 100.0)
        cached_v = _hf_tensor(visual_tokens, 500.0)
        source_k, source_v = cached_k.clone(), cached_v.clone()
        request_a_k = _hf_tensor(len(positions), 1000.0)
        request_a_v = _hf_tensor(len(positions), 5000.0)
        result_a_k, result_a_v = assemble_linked_kv(
            cached_k, cached_v, request_a_k, request_a_v, positions,
            visual_start, total)
        frozen_a_k, frozen_a_v = result_a_k.clone(), result_a_v.clone()

        request_b_k = _hf_tensor(len(positions), 9000.0)
        request_b_v = _hf_tensor(len(positions), 13000.0)
        result_b_k, result_b_v = assemble_linked_kv(
            cached_k, cached_v, request_b_k, request_b_v, positions,
            visual_start, total)

        self.assertTrue(torch.equal(result_a_k, frozen_a_k))
        self.assertTrue(torch.equal(result_a_v, frozen_a_v))
        self.assertTrue(torch.equal(cached_k, source_k))
        self.assertTrue(torch.equal(cached_v, source_v))
        self.assertFalse(torch.equal(result_a_k, result_b_k))
        self.assertFalse(torch.equal(result_a_v, result_b_v))
        for logical_position in _positions(plan.reused_image_positions):
            self.assertTrue(torch.equal(
                result_a_k[:, :, logical_position, :],
                result_b_k[:, :, logical_position, :]))
            self.assertTrue(torch.equal(
                result_a_v[:, :, logical_position, :],
                result_b_v[:, :, logical_position, :]))


class CausalMaskTests(unittest.TestCase):
    def test_original_positions_and_valid_key_mask_define_visibility(self):
        query_positions = torch.tensor([1, 4, 6], dtype=torch.long)
        valid = torch.tensor(
            [True, True, False, True, True, False, True],
            dtype=torch.bool)
        mask = build_causal_mask(
            query_positions, key_length=7, dtype=torch.float32,
            valid_key_mask=valid)

        self.assertEqual(tuple(mask.shape), (1, 1, 3, 7))
        self.assertEqual(mask.dtype, torch.float32)
        matrix = mask[0, 0]
        allowed = ((torch.arange(7).view(1, -1)
                    <= query_positions.view(-1, 1))
                   & valid.view(1, -1))
        self.assertTrue(torch.equal(
            matrix[allowed], torch.zeros_like(matrix[allowed])))
        blocked = matrix[~allowed]
        self.assertTrue(
            (torch.isneginf(blocked) | (blocked < -1.0e20)).all().item())

    def test_changing_future_keys_and_values_cannot_change_past_output(self):
        torch.manual_seed(7)
        query_positions = torch.tensor([1, 4], dtype=torch.long)
        queries = torch.randn(2, 4)
        keys = torch.randn(6, 4)
        values = torch.randn(6, 3)
        mask = build_causal_mask(
            query_positions, key_length=6, dtype=torch.float32)[0, 0]

        def attend(current_keys, current_values):
            scores = queries @ current_keys.transpose(0, 1) / math.sqrt(4)
            return torch.softmax(scores + mask, dim=-1) @ current_values

        reference = attend(keys, values)
        changed_keys = keys.clone()
        changed_values = values.clone()
        changed_keys[2:] = torch.randn_like(changed_keys[2:]) * 100.0
        changed_values[2:] = torch.randn_like(changed_values[2:]) * 100.0
        changed = attend(changed_keys, changed_values)

        torch.testing.assert_close(
            changed[0], reference[0], rtol=1e-5, atol=1e-6)
        self.assertFalse(torch.allclose(changed[1], reference[1]))


class IntegratedReferenceTests(unittest.TestCase):
    @staticmethod
    def _uniform_attention(query_positions, keys, values):
        queries = torch.zeros(
            keys.shape[0], keys.shape[1], len(query_positions),
            keys.shape[-1], dtype=keys.dtype)
        mask = build_causal_mask(
            torch.tensor(query_positions, dtype=torch.long),
            key_length=keys.shape[-2], dtype=keys.dtype)
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        return torch.softmax(scores + mask, dim=-1) @ values

    def test_k_zero_matches_full_context_reference_attention(self):
        plan = build_active_row_plan(
            total_tokens=5, visual_start=1, visual_tokens=2,
            k_recompute=0)
        positions = _positions(plan.active_positions)
        cached_k = torch.zeros(1, 1, 2, 1)
        cached_v = torch.tensor([[[[2.0], [3.0]]]])
        active_k = torch.zeros(1, 1, len(positions), 1)
        active_v = torch.tensor([1.0, 4.0, 5.0]).view(1, 1, -1, 1)
        full_k, full_v = assemble_linked_kv(
            cached_k, cached_v, active_k, active_v, positions,
            visual_start=1, total_tokens=5, dummy_value=-999.0)

        actual = self._uniform_attention(positions, full_k, full_v)
        expected_values = torch.arange(1, 6, dtype=torch.float32).view(
            1, 1, 5, 1)
        reference = self._uniform_attention(
            positions, torch.zeros_like(expected_values), expected_values)
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            actual.flatten(), torch.tensor([1.0, 2.5, 3.0]),
            rtol=1e-5, atol=1e-6)

    def test_k_n_overwrites_stale_image_values_and_matches_full_prefill(self):
        plan = build_active_row_plan(
            total_tokens=5, visual_start=1, visual_tokens=2,
            k_recompute=2)
        positions = _positions(plan.active_positions)
        cached_k = torch.full((1, 1, 2, 1), 99.0)
        cached_v = torch.full((1, 1, 2, 1), 99.0)
        active_k = torch.zeros(1, 1, 5, 1)
        active_v = torch.arange(1, 6, dtype=torch.float32).view(
            1, 1, 5, 1)
        full_k, full_v = assemble_linked_kv(
            cached_k, cached_v, active_k, active_v, positions,
            visual_start=1, total_tokens=5, dummy_value=-999.0)

        self.assertTrue(torch.equal(full_k, active_k))
        self.assertTrue(torch.equal(full_v, active_v))
        actual = self._uniform_attention(positions, full_k, full_v)
        torch.testing.assert_close(
            actual.flatten(),
            torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0]),
            rtol=1e-5, atol=1e-6)


class ProductionSelectivePrefillReferenceTests(unittest.TestCase):
    """Exercise the production selective traversal against a tiny Llama.

    The fixture is created entirely from a random local configuration: it
    downloads no weights and uses no CUDA device.  Four query heads and two KV
    heads also exercise the production ``repeat_kv`` path rather than only the
    equal-head case.
    """

    RTOL = 1e-5
    ATOL = 1e-6

    @staticmethod
    def _capture_layer_hidden(layers, call):
        """Capture each layer's residual-plus-MLP output for one call.

        ``MPICSelectivePrefill`` deliberately invokes the layer components
        directly, so a hook on ``LlamaDecoderLayer.forward`` would not observe
        it.  The input to ``post_attention_layernorm`` is the post-attention
        residual and the MLP hook returns the other term in the final layer
        residual; their sum is therefore exactly the layer output.
        """
        post_attention = [None] * len(layers)
        mlp_outputs = [None] * len(layers)
        handles = []

        def post_hook(index):
            def capture(_module, args):
                post_attention[index] = args[0].detach().clone()
            return capture

        def mlp_hook(index):
            def capture(_module, _args, output):
                mlp_outputs[index] = output.detach().clone()
            return capture

        for index, layer in enumerate(layers):
            handles.append(layer.post_attention_layernorm.
                           register_forward_pre_hook(post_hook(index)))
            handles.append(layer.mlp.register_forward_hook(mlp_hook(index)))
        try:
            result = call()
        finally:
            for handle in handles:
                handle.remove()

        if any(value is None for value in post_attention + mlp_outputs):
            raise AssertionError("a decoder layer did not traverse the fixture")
        hidden = [post + mlp for post, mlp in zip(
            post_attention, mlp_outputs)]
        return result, hidden

    def test_k_zero_partial_and_full_match_fp32_full_prefill(self):
        from transformers import LlamaConfig, LlamaForCausalLM

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1)
            config = LlamaConfig(
                vocab_size=50,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
                attention_dropout=0.0,
                attn_implementation="eager",
            )
            language_model = LlamaForCausalLM(config).eval().float()

            class TinyTopLevelModel:
                device = torch.device("cpu")

                def __init__(self):
                    self.config = config
                    self.model = SimpleNamespace(
                        language_model=language_model.model)
                    self.lm_head = language_model.lm_head

                @staticmethod
                def get_input_embeddings():
                    return language_model.get_input_embeddings()

            runner = SimpleNamespace(
                model=TinyTopLevelModel(),
                layers=language_model.model.layers,
            )
            total, visual_start, visual_tokens = 9, 2, 4
            input_ids = torch.tensor(
                [1, 2, 3, 3, 3, 3, 4, 5, 6], dtype=torch.long)
            visual_inputs = torch.randn(visual_tokens, config.hidden_size)
            embeddings = language_model.get_input_embeddings()(
                input_ids.unsqueeze(0)).detach().clone()
            embeddings[:, visual_start:visual_start + visual_tokens] = \
                visual_inputs.unsqueeze(0)
            positions = torch.arange(total).unsqueeze(0)

            def full_prefill():
                return language_model.model(
                    inputs_embeds=embeddings,
                    attention_mask=torch.ones(1, total, dtype=torch.long),
                    position_ids=positions,
                    cache_position=positions[0],
                    use_cache=True,
                )

            with torch.no_grad():
                reference, reference_hidden = self._capture_layer_hidden(
                    runner.layers, full_prefill)
                reference_logits = language_model.lm_head(
                    reference.last_hidden_state[:, -1:, :])
            reference_cache = [
                (layer.keys.detach().clone(), layer.values.detach().clone())
                for layer in reference.past_key_values.layers
            ]

            with tempfile.TemporaryDirectory() as temporary:
                store = Path(temporary)
                (store / "visual_input.bin").write_bytes(
                    visual_inputs.numpy().astype(
                        np.float32, copy=False).tobytes())
                for layer_index, (key, value) in enumerate(reference_cache):
                    layer_dir = store / f"layer_{layer_index:02d}"
                    layer_dir.mkdir()
                    for kind, tensor in (("k", key), ("v", value)):
                        block = tensor[
                            0, :, visual_start:
                            visual_start + visual_tokens, :
                        ].permute(1, 0, 2).contiguous().numpy()
                        (layer_dir / f"{kind}.bin").write_bytes(
                            block.astype(np.float32, copy=False).tobytes())

                meta = {
                    "dtype": "float32",
                    "v_token_start": visual_start,
                    "v_token_num": visual_tokens,
                    "num_layers": config.num_hidden_layers,
                    "num_heads": config.num_key_value_heads,
                    "head_dim": config.hidden_size
                    // config.num_attention_heads,
                    "hidden_size": config.hidden_size,
                    # Keep k=2 inside the first four-row chunk so the
                    # production reader returns stale selected-row K/V too;
                    # indexed replacement must still recover the reference.
                    "chunk_size": visual_tokens,
                }
                reader = MPICRawReader(store, meta, drop_cache=False)
                context = SimpleNamespace(meta=meta, reader=reader)
                try:
                    for k_recompute in (0, 2, visual_tokens):
                        with self.subTest(k_recompute=k_recompute):
                            counter = IOCounter()

                            def selective_prefill():
                                return MPICSelectivePrefill(
                                    runner, context,
                                    k_recompute=k_recompute).run(
                                        input_ids, visual_start, counter)

                            with torch.no_grad():
                                result, active_hidden = \
                                    self._capture_layer_hidden(
                                        runner.layers, selective_prefill)
                            logits, layer_cache, plan, stats = result

                            torch.testing.assert_close(
                                logits, reference_logits,
                                rtol=self.RTOL, atol=self.ATOL)
                            self.assertEqual(len(layer_cache),
                                             len(reference_cache))
                            for (actual_k, actual_v), (expected_k,
                                                        expected_v) in zip(
                                    layer_cache, reference_cache):
                                torch.testing.assert_close(
                                    actual_k, expected_k,
                                    rtol=self.RTOL, atol=self.ATOL)
                                torch.testing.assert_close(
                                    actual_v, expected_v,
                                    rtol=self.RTOL, atol=self.ATOL)

                            active_positions = plan.active_positions
                            for actual, expected in zip(
                                    active_hidden, reference_hidden):
                                torch.testing.assert_close(
                                    actual,
                                    expected.index_select(
                                        1, active_positions),
                                    rtol=self.RTOL, atol=self.ATOL)

                            expected_image = min(
                                k_recompute, visual_tokens)
                            self.assertEqual(
                                stats["recomputed_image_rows_per_layer"],
                                [expected_image]
                                * config.num_hidden_layers)
                            self.assertEqual(
                                stats["active_rows_per_layer"],
                                [plan.n_active] * config.num_hidden_layers)
                            self.assertEqual(
                                stats["attention_key_length_per_layer"],
                                [total] * config.num_hidden_layers)
                finally:
                    reader.close()


class RoPERelocationTests(unittest.TestCase):
    def setUp(self):
        self.pre_rope = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0],
               [-2.0, 0.5, 1.5, -3.0]],
              [[0.25, -1.0, 2.5, 0.75],
               [3.0, -2.0, 0.5, 1.0]]]],
            dtype=torch.float32)
        self.source_cos, self.source_sin = _rope_tables(torch.tensor(
            [[0.20, -0.35], [0.70, 0.10]], dtype=torch.float32))
        self.target_cos, self.target_sin = _rope_tables(torch.tensor(
            [[0.55, 0.15], [-0.25, 0.90]], dtype=torch.float32))

    def test_same_position_is_identity_not_a_second_rope_application(self):
        source_keys = _apply_rope(
            self.pre_rope, self.source_cos, self.source_sin)
        relocated = relocate_post_rope_keys(
            source_keys, self.source_cos, self.source_sin,
            self.source_cos, self.source_sin)

        torch.testing.assert_close(
            relocated, source_keys, rtol=1e-5, atol=1e-6)
        double_rotated = _apply_rope(
            source_keys, self.source_cos, self.source_sin)
        self.assertFalse(torch.allclose(relocated, double_rotated))

    def test_shift_matches_direct_target_rope_and_keeps_source_immutable(self):
        source_keys = _apply_rope(
            self.pre_rope, self.source_cos, self.source_sin)
        originals = tuple(value.clone() for value in (
            source_keys, self.source_cos, self.source_sin,
            self.target_cos, self.target_sin))
        expected = _apply_rope(
            self.pre_rope, self.target_cos, self.target_sin)

        relocated = relocate_post_rope_keys(
            source_keys, self.source_cos, self.source_sin,
            self.target_cos, self.target_sin)

        torch.testing.assert_close(
            relocated, expected, rtol=1e-5, atol=1e-6)
        for actual, original in zip((
                source_keys, self.source_cos, self.source_sin,
                self.target_cos, self.target_sin), originals):
            self.assertTrue(torch.equal(actual, original))


class DynamicCacheSeedTests(unittest.TestCase):
    def test_seed_has_exact_prompt_length_and_decode_appends_only_one_row(self):
        prompt_length = 6
        layers = [
            (_hf_tensor(prompt_length, 1000.0 * layer),
             _hf_tensor(prompt_length, 10000.0 + 1000.0 * layer))
            for layer in range(2)
        ]
        originals = [(key.clone(), value.clone()) for key, value in layers]

        cache = seed_dynamic_cache(layers)

        self.assertEqual(len(cache.layers), len(layers))
        for layer_index, (expected_k, expected_v) in enumerate(layers):
            self.assertEqual(cache.get_seq_length(layer_index), prompt_length)
            self.assertEqual(cache.layers[layer_index].keys.shape[-2],
                             prompt_length)
            self.assertEqual(cache.layers[layer_index].values.shape[-2],
                             prompt_length)
            self.assertTrue(torch.equal(
                cache.layers[layer_index].keys, expected_k))
            self.assertTrue(torch.equal(
                cache.layers[layer_index].values, expected_v))

        for layer_index in range(len(layers)):
            next_k = _hf_tensor(1, 50000.0 + layer_index)
            next_v = _hf_tensor(1, 60000.0 + layer_index)
            cache.update(next_k, next_v, layer_index)
            self.assertEqual(cache.get_seq_length(layer_index),
                             prompt_length + 1)
            self.assertTrue(torch.equal(
                cache.layers[layer_index].keys[:, :, :prompt_length, :],
                originals[layer_index][0]))
            self.assertTrue(torch.equal(
                cache.layers[layer_index].keys[:, :, -1:, :], next_k))
            self.assertTrue(torch.equal(
                cache.layers[layer_index].values[:, :, -1:, :], next_v))

        for (actual_k, actual_v), (original_k, original_v) in zip(
                layers, originals):
            self.assertTrue(torch.equal(actual_k, original_k))
            self.assertTrue(torch.equal(actual_v, original_v))


class MPICRawReaderTests(unittest.TestCase):
    def test_visual_prefix_and_chunk_aligned_reuse_have_exact_accounting(self):
        visual_tokens = 10
        heads, head_dim, hidden_size, chunk_size = 2, 2, 3, 4
        meta = {
            "num_layers": 1,
            "v_token_num": visual_tokens,
            "num_heads": heads,
            "head_dim": head_dim,
            "hidden_size": hidden_size,
            "chunk_size": chunk_size,
            "dtype": "float16",
        }
        visual = np.arange(
            visual_tokens * hidden_size, dtype=np.float16).reshape(
                visual_tokens, hidden_size)
        key = (np.arange(
            visual_tokens * heads * head_dim, dtype=np.float16).reshape(
                visual_tokens, heads, head_dim) + np.float16(100))
        value = key + np.float16(1000)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layer_dir = root / "layer_00"
            layer_dir.mkdir()
            (root / "visual_input.bin").write_bytes(visual.tobytes())
            (layer_dir / "k.bin").write_bytes(key.tobytes())
            (layer_dir / "v.bin").write_bytes(value.tobytes())

            reader = MPICRawReader(root, meta, drop_cache=False)
            try:
                embedding_counter = IOCounter()
                selected = reader.read_visual_inputs(
                    5, counter=embedding_counter)
                self.assertEqual(tuple(selected.shape), (5, hidden_size))
                self.assertEqual(selected.dtype, torch.float16)
                self.assertTrue(torch.equal(
                    selected, torch.from_numpy(visual[:5].copy())))
                embedding_io = embedding_counter.summary()
                self.assertEqual(
                    embedding_io["bytes"], 5 * hidden_size * 2)
                self.assertEqual(embedding_io["preads"], 1)

                kv_counter = IOCounter()
                rows, reused = reader.read_reused_layer(
                    0, "k", k_recompute=5, counter=kv_counter)
                # Logical reuse starts at row 5, but its physical chunk starts
                # at row 4.  The over-read row must be counted honestly.
                self.assertEqual(_positions(rows), list(range(4, 10)))
                self.assertEqual(tuple(reused.shape), (6, heads, head_dim))
                self.assertTrue(torch.equal(
                    reused, torch.from_numpy(key[4:].copy())))
                value_rows, reused_values = reader.read_reused_layer(
                    0, "v", k_recompute=5, counter=kv_counter)
                self.assertEqual(_positions(value_rows), list(range(4, 10)))
                self.assertTrue(torch.equal(
                    reused_values, torch.from_numpy(value[4:].copy())))
                kv_io = kv_counter.summary()
                self.assertEqual(
                    kv_io["bytes"], 2 * 6 * heads * head_dim * 2)
                self.assertEqual(kv_io["preads"], 2)
            finally:
                close = getattr(reader, "close", None)
                if close is not None:
                    close()


if __name__ == "__main__":
    unittest.main()

"""CPU contracts for Turn-1 piggyback capture and persistence."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from mmimpress.piggyback import (
    VisionForwardCapture,
    VisionSaliencyCapture,
    deterministic_method_rotation,
    persist_captured_visual_prefix,
    stable_json_sha256,
)
from mmimpress.serve import ImageContext
from mmimpress.store import load_meta


class _FakeSelfAttention(nn.Module):
    def __init__(self, layer_index: int, implementation: str = "eager"):
        super().__init__()
        self.layer_index = layer_index
        self.config = SimpleNamespace(
            _attn_implementation=implementation)
        self.output_attention_flags = []
        self.last_weights = None

    def forward(self, hidden_states, attention_mask=None,
                causal_attention_mask=None, output_attentions=False):
        self.output_attention_flags.append(bool(output_attentions))
        batch, tokens, _ = hidden_states.shape
        heads = 2
        weights = torch.arange(
            batch * heads * tokens * tokens, dtype=hidden_states.dtype,
            device=hidden_states.device,
        ).view(batch, heads, tokens, tokens)
        weights = weights + self.layer_index * 1000
        self.last_weights = weights.detach().clone()
        # The attention-output path deliberately does not depend on whether
        # weights are returned, matching the property the hook must preserve.
        output = hidden_states + (self.layer_index + 1)
        return output, (weights if output_attentions else None)


class _FakeEncoderLayer(nn.Module):
    def __init__(self, layer_index: int, implementation: str = "eager",
                 positional: bool = False):
        super().__init__()
        self.self_attn = _FakeSelfAttention(layer_index, implementation)
        self.positional = positional

    def forward(self, hidden_states):
        if self.positional:
            hidden_states, _ = self.self_attn(
                hidden_states, None, None, False)
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=False,
            )
        return hidden_states


class _FakeVisionTower(nn.Module):
    def __init__(self, layers=4, implementation="eager", positional=False):
        super().__init__()
        encoder_layers = nn.ModuleList([
            _FakeEncoderLayer(index, implementation, positional)
            for index in range(layers)
        ])
        self.vision_model = nn.Module()
        self.vision_model.encoder = nn.Module()
        self.vision_model.encoder.layers = encoder_layers

    def forward(self, pixel_values, output_attentions=False):
        hidden = pixel_values
        for layer in self.vision_model.encoder.layers:
            hidden = layer(hidden)
        return hidden


def _fake_runner(implementation="eager", positional=False):
    tower = _FakeVisionTower(implementation=implementation,
                              positional=positional).eval()
    return SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(vision_tower=tower)))


class VisionForwardCaptureTests(unittest.TestCase):
    def test_exact_penultimate_capture_only_and_output_unchanged(self):
        runner = _fake_runner()
        tower = runner.model.model.vision_tower
        pixels = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)

        baseline = tower(pixels)
        for layer in tower.vision_model.encoder.layers:
            layer.self_attn.output_attention_flags.clear()

        with VisionSaliencyCapture(runner) as capture:
            instrumented = tower(pixels)

        self.assertTrue(torch.equal(instrumented, baseline))
        flags = [layer.self_attn.output_attention_flags
                 for layer in tower.vision_model.encoder.layers]
        self.assertEqual(flags, [[False], [False], [True], [False]])
        target = tower.vision_model.encoder.layers[-2].self_attn
        expected = target.last_weights[:, :, 0, 1:].sum(dim=1).float()
        self.assertTrue(torch.equal(capture.per_sub_scores, expected))
        self.assertEqual(capture.call_count, 1)
        self.assertEqual(capture.saliency_call_count, 1)
        self.assertEqual(capture.timing_backend, "cpu_perf_counter")
        self.assertGreaterEqual(capture.vision_elapsed_ms, 0.0)
        self.assertGreaterEqual(capture.saliency_reduction_ms, 0.0)
        self.assertEqual(len(tower._forward_pre_hooks), 0)
        self.assertEqual(len(tower._forward_hooks), 0)
        self.assertEqual(len(target._forward_pre_hooks), 0)
        self.assertEqual(len(target._forward_hooks), 0)

    def test_timing_only_never_requests_any_attention(self):
        runner = _fake_runner()
        tower = runner.model.model.vision_tower
        pixels = torch.ones(1, 4, 2)
        with VisionForwardCapture(runner, capture_saliency=False) as capture:
            tower(pixels)
        flags = [layer.self_attn.output_attention_flags
                 for layer in tower.vision_model.encoder.layers]
        self.assertEqual(flags, [[False], [False], [False], [False]])
        self.assertIsNone(capture.per_sub_scores)
        self.assertEqual(capture.call_count, 1)
        self.assertEqual(capture.saliency_call_count, 0)

    def test_positional_attention_call_is_bound_by_parameter_name(self):
        runner = _fake_runner(positional=True)
        tower = runner.model.model.vision_tower
        with VisionSaliencyCapture(runner) as capture:
            tower(torch.ones(1, 4, 2))
        flags = [layer.self_attn.output_attention_flags
                 for layer in tower.vision_model.encoder.layers]
        self.assertEqual(flags, [[False], [False], [True], [False]])
        self.assertEqual(tuple(capture.result_cpu().shape), (1, 3))

    def test_exception_removes_only_our_hooks_and_active_marker(self):
        runner = _fake_runner()
        tower = runner.model.model.vision_tower
        external_calls = []
        external = tower.register_forward_hook(
            lambda module, args, output: external_calls.append(1))
        try:
            with self.assertRaisesRegex(RuntimeError, "after response"):
                with VisionSaliencyCapture(runner):
                    tower(torch.ones(1, 4, 2))
                    raise RuntimeError("after response")
            self.assertEqual(external_calls, [1])
            self.assertEqual(len(tower._forward_hooks), 1)
            self.assertFalse(hasattr(
                tower, "_mmimpress_active_vision_forward_capture"))
        finally:
            external.remove()

    def test_real_tiny_clip_matches_full_penultimate_attention(self):
        from transformers import CLIPVisionConfig, CLIPVisionModel

        config = CLIPVisionConfig(
            hidden_size=16, intermediate_size=32, num_hidden_layers=3,
            num_attention_heads=4, image_size=8, patch_size=4,
            attention_dropout=0.0,
        )
        config._attn_implementation = "eager"
        tower = CLIPVisionModel(config).eval()
        runner = SimpleNamespace(
            model=SimpleNamespace(
                model=SimpleNamespace(vision_tower=tower),
                config=SimpleNamespace(output_attentions=False),
            ))
        pixels = torch.arange(3 * 8 * 8, dtype=torch.float32).view(
            1, 3, 8, 8) / 255.0
        baseline = tower(pixels, output_attentions=True,
                         return_dict=True)
        with VisionSaliencyCapture(runner) as capture:
            instrumented = tower(pixels, output_attentions=False,
                                 return_dict=True)
        expected = baseline.attentions[-2][:, :, 0, 1:].sum(dim=1).float()
        self.assertTrue(torch.equal(capture.result_cpu(), expected))
        self.assertTrue(torch.equal(instrumented.last_hidden_state,
                                    baseline.last_hidden_state))

    def test_fail_closed_on_backend_no_call_or_multiple_calls(self):
        with self.assertRaisesRegex(AssertionError, "eager"):
            VisionSaliencyCapture(_fake_runner("sdpa"))

        runner = _fake_runner()
        tower = runner.model.model.vision_tower
        capture = VisionSaliencyCapture(runner)
        with self.assertRaisesRegex(AssertionError, "expected one"):
            with capture:
                pass
        self.assertEqual(len(tower._forward_hooks), 0)

        runner = _fake_runner()
        tower = runner.model.model.vision_tower
        capture = VisionForwardCapture(runner, capture_saliency=False)
        with self.assertRaisesRegex(AssertionError, "more than once"):
            with capture:
                tower(torch.ones(1, 4, 2))
                tower(torch.ones(1, 4, 2))
        self.assertEqual(capture.call_count, 2)
        self.assertEqual(len(tower._forward_hooks), 0)


class _GeometryRunner:
    model_id = "synthetic/llava-next"

    def __init__(self):
        self.visual_span_calls = 0
        self.anyres_layout_calls = 0

    def visual_span(self, input_ids):
        self.visual_span_calls += 1
        self.last_ids = input_ids.clone()
        return 2, 7

    def anyres_layout(self, image_size, v_num):
        self.anyres_layout_calls += 1
        self.last_size = list(image_size)
        self.last_v_num = v_num
        # base^2 + hi_h * (hi_w + one separator) = 4 + 1*3 = 7
        return 2, 1, 2, [6]


def _captured_layers(num_layers=2, heads=2, sequence=12, head_dim=3):
    layers = []
    count = heads * sequence * head_dim
    for index in range(num_layers):
        key = torch.arange(count, dtype=torch.float16).view(
            1, heads, sequence, head_dim) + index * 1000
        value = key + 5000
        layers.append((key, value))
    return layers


def _capture_provenance():
    return {
        "capture_saliency": True,
        "vision_call_count": 1,
        "saliency_call_count": 1,
        "extra_vision_forward_calls": 0,
        "extra_saliency_forward_ms": 0.0,
        "saliency_layer_from_end": 2,
        "vision_num_layers": 24,
        "saliency_layer_index": 22,
        "vision_attention_backend": "eager",
        "global_vision_output_attentions": False,
        "global_model_output_attentions": False,
        "vision_ms": 1.0,
    }


class PiggybackPersistenceTests(unittest.TestCase):
    @staticmethod
    def _mapped_scores(*args, **kwargs):
        return torch.tensor([
            0.2, 0.8, 0.8, 0.1, 0.6, 0.3, float("inf")
        ], dtype=torch.float32)

    def test_atomic_store_has_exact_prefix_layout_and_no_v_hidden(self):
        runner = _GeometryRunner()
        ids = torch.arange(12).unsqueeze(0)
        layers = _captured_layers()
        per_sub = torch.ones(2, 4)

        with tempfile.TemporaryDirectory() as td, \
             mock.patch("mmimpress.piggyback.anyres_token_scores",
                        side_effect=self._mapped_scores) as mapper:
            output = Path(td) / "image-7"
            result = persist_captured_visual_prefix(
                runner, layers, ids, torch.tensor([[480, 640]]), per_sub,
                output, image_id="image-7", chunk_size=2,
                capture_stats=_capture_provenance(),
            )

            mapper.assert_called_once()
            self.assertEqual(runner.visual_span_calls, 1)
            self.assertEqual(runner.anyres_layout_calls, 1)
            self.assertTrue(output.is_dir())
            self.assertFalse((output / "v_hidden.pt").exists())
            meta = load_meta(output)
            self.assertEqual(meta["physical_layout"],
                             "visionzip_image_only")
            self.assertEqual(meta["layout_source"],
                             "turn1_normal_inference_piggyback")
            self.assertTrue(meta["turn1_normal_inference"])
            self.assertFalse(meta["separate_vision_forward"])
            self.assertFalse(meta["separate_prefix_forward"])
            self.assertFalse(meta["layout_uses_dataset_question"])
            self.assertFalse(meta["llm_used_for_layout_scoring"])
            self.assertEqual(meta["calibration_questions"], 0)
            self.assertEqual(meta["probe_heads"], 0)
            self.assertEqual(meta["bytes_probe_sidecar"], 0)
            self.assertGreater(meta["bytes_separator_sidecar"], 0)
            self.assertEqual(meta["prefix_len"], 9)
            self.assertEqual(meta["order"], [1, 2, 4, 5, 0, 3, 6])
            self.assertEqual(meta["newline_stored"], [6])

            artifact = torch.load(output / "visionzip_layout.pt",
                                  weights_only=True)
            self.assertEqual(artifact["capture_source"],
                             "same_turn1_normal_inference")
            self.assertFalse(artifact["layout_uses_dataset_question"])
            self.assertEqual(artifact["calibration_questions"], 0)

            self.assertEqual(result["bytes"]["probe_sidecar"], 0)
            self.assertEqual(
                result["bytes"]["total"],
                sum(path.stat().st_size for path in output.rglob("*")
                    if path.is_file()),
            )
            self.assertEqual(set(result["hashes"]["files_sha256"]),
                             {"meta.json", "visionzip_layout.pt"})
            self.assertIsNone(result["hashes"]["tree_sha256"])
            self.assertFalse(result["hashes"]["full_integrity_hash"])
            self.assertEqual(
                len(result["hashes"]["prefix_kv_sample_sha256"]), 64)
            self.assertTrue(result["durability"]["atomic_no_clobber"])
            self.assertTrue(
                result["durability"]["parent_fsynced_after_rename"])
            self.assertGreater(result["durability"]["files_fsynced"], 0)
            self.assertGreaterEqual(result["timing_ms"]["persist_ms"],
                                    result["timing_ms"]["durability_ms"])
            self.assertGreaterEqual(result["timing_ms"]["helper_total_ms"],
                                    result["timing_ms"]["persist_ms"])

            # Prefix-only serving can open the store without an otherwise
            # unused SparseVLM hidden-state artifact.
            context = ImageContext(output, torch.device("cpu"),
                                   drop_cache=False,
                                   require_v_hidden=False)
            try:
                self.assertIsNone(context.v_hidden)
                context.validate_prefix_layout("visionzip_image_only")
            finally:
                context.close()

    def test_existing_target_refused_and_failed_stage_removed(self):
        runner = _GeometryRunner()
        ids = torch.arange(12)
        layers = _captured_layers()
        per_sub = torch.ones(2, 4)
        with tempfile.TemporaryDirectory() as td, \
             mock.patch("mmimpress.piggyback.anyres_token_scores",
                        side_effect=self._mapped_scores):
            parent = Path(td)
            existing = parent / "existing"
            existing.mkdir()
            marker = existing / "keep.txt"
            marker.write_text("do not overwrite")
            with self.assertRaises(FileExistsError):
                persist_captured_visual_prefix(
                    runner, layers, ids, [480, 640], per_sub, existing,
                    image_id="existing")
            self.assertEqual(marker.read_text(), "do not overwrite")

            failed = parent / "failed"
            with mock.patch("mmimpress.piggyback.write_image_store",
                            side_effect=RuntimeError("synthetic write error")):
                with self.assertRaisesRegex(RuntimeError, "synthetic"):
                    persist_captured_visual_prefix(
                        runner, layers, ids, [480, 640], per_sub, failed,
                        image_id="failed",
                        capture_stats=_capture_provenance())
            self.assertFalse(failed.exists())
            self.assertEqual(list(parent.glob(".failed.staging-*")), [])

    def test_unproven_capture_metadata_is_rejected_before_staging(self):
        runner = _GeometryRunner()
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "unproven"
            invalid = _capture_provenance()
            invalid["vision_call_count"] = 2
            with self.assertRaisesRegex(AssertionError, "provenance"):
                persist_captured_visual_prefix(
                    runner, _captured_layers(), torch.arange(12),
                    [480, 640], torch.ones(2, 4), output,
                    image_id="unproven", capture_stats=invalid)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(td).glob(".unproven.staging-*")), [])

    def test_post_publication_integrity_failure_does_not_strand_as_error(self):
        runner = _GeometryRunner()
        with tempfile.TemporaryDirectory() as td, \
             mock.patch("mmimpress.piggyback.anyres_token_scores",
                        side_effect=self._mapped_scores), \
             mock.patch("mmimpress.piggyback._sampled_store_sha256",
                        side_effect=OSError("diagnostic read failed")):
            output = Path(td) / "durable"
            result = persist_captured_visual_prefix(
                runner, _captured_layers(), torch.arange(12), [480, 640],
                torch.ones(2, 4), output, image_id="durable",
                capture_stats=_capture_provenance())
            self.assertTrue(output.is_dir())
            self.assertFalse(result["integrity"]["ok"])
            self.assertIn("diagnostic read failed",
                          result["integrity"]["error"])
            self.assertTrue(
                result["integrity"]["excluded_from_persist_ms"])


class DeterministicHelpersTests(unittest.TestCase):
    def test_rotation_is_balanced_and_stable_hash_is_order_independent(self):
        methods = ("recompute", "prefix_25", "prefix_45")
        self.assertEqual(deterministic_method_rotation(methods, 0), methods)
        self.assertEqual(deterministic_method_rotation(methods, 1),
                         ("prefix_25", "prefix_45", "recompute"))
        self.assertEqual(deterministic_method_rotation(methods, 2),
                         ("prefix_45", "recompute", "prefix_25"))
        self.assertEqual(deterministic_method_rotation(methods, 3), methods)
        self.assertEqual(stable_json_sha256({"a": 1, "b": 2}),
                         stable_json_sha256({"b": 2, "a": 1}))
        with self.assertRaises(ValueError):
            deterministic_method_rotation((), 0)
        with self.assertRaises(ValueError):
            deterministic_method_rotation(("x", "x"), 0)


if __name__ == "__main__":
    unittest.main()

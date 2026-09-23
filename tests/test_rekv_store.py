"""Direct-projection capture, representative, and SSD I/O invariants."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from mmimpress.rekv_store import (
    ReKVCapture, ReKVContext, persist_captured_rekv_prefix,
    representative_keys, validate_pre_rope_capture_against_cache,
)
from mmimpress.store import IOCounter


class _FakeConfig:
    _commit_hash = "frozen-fixture"

    def to_dict(self):
        return {"fixture": "rekv_store", "heads": 2}


class _FakeRunner:
    def __init__(self, n_layers=2, heads=2, head_dim=3, separators=(3, 7, 12)):
        self.n_heads = heads
        self.head_dim = head_dim
        self.model_id = "fake-rekv-model"
        self.cfg = _FakeConfig()
        self.layers = []
        self.separators = list(separators)
        for _ in range(n_layers):
            attention = SimpleNamespace(
                k_proj=nn.Linear(heads * head_dim, heads * head_dim, bias=False),
                v_proj=nn.Linear(heads * head_dim, heads * head_dim, bias=False),
            )
            self.layers.append(SimpleNamespace(self_attn=attention))

    def visual_span(self, input_ids):
        ids = torch.as_tensor(input_ids).reshape(-1)
        positions = (ids == 99).nonzero(as_tuple=True)[0]
        return int(positions[0]), int(positions.numel())

    def anyres_layout(self, image_size, v_num):
        assert v_num == 13
        return 2, 3, 3, self.separators


def _capture_fixture(runner, prefix_len, suffix_len=2,
                     dtype=torch.float32):
    torch.manual_seed(8)
    hidden = torch.randn(1, prefix_len + suffix_len,
                         runner.n_heads * runner.head_dim, dtype=dtype)
    raw_forward = []
    with ReKVCapture(runner, 2, prefix_len - 2) as capture:
        for layer in runner.layers:
            attention = layer.self_attn
            key = attention.k_proj(hidden)
            value = attention.v_proj(hidden)
            raw_forward.append((key.detach(), value.detach()))
        for layer in runner.layers:
            layer.self_attn.k_proj(hidden[:, :1])
            layer.self_attn.v_proj(hidden[:, :1])
    return capture, raw_forward


class ReKVCaptureTests(unittest.TestCase):
    def test_same_forward_projection_capture_and_cleanup(self):
        runner = _FakeRunner()
        capture, raw = _capture_fixture(runner, 15)
        values = capture.result_cpu()
        self.assertEqual(capture.stats()["full_prefill_calls_per_layer"],
                         [[1, 1], [1, 1]])
        self.assertTrue(capture.stats()["hooks_removed"])
        self.assertFalse(hasattr(runner, "_mmimpress_active_rekv_capture"))
        for li, (key, value) in enumerate(values):
            expected_key = raw[li][0][0, :15].reshape(15, 2, 3)
            expected_value = raw[li][1][0, :15].reshape(15, 2, 3)
            self.assertTrue(torch.equal(key, expected_key))
            self.assertTrue(torch.equal(value, expected_value))
            self.assertEqual(tuple(key.shape), (15, 2, 3))

    def test_raw_key_rotates_to_same_turn1_cache_in_bf16(self):
        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import (
            LlamaRotaryEmbedding, apply_rotary_pos_emb,
        )

        runner = _FakeRunner(head_dim=4)
        for layer in runner.layers:
            layer.self_attn.k_proj.to(torch.bfloat16)
            layer.self_attn.v_proj.to(torch.bfloat16)
        config = LlamaConfig(
            hidden_size=8, num_attention_heads=2, num_key_value_heads=2,
            head_dim=4, max_position_embeddings=128)
        rotary = LlamaRotaryEmbedding(config=config)
        runner.model = SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(rotary_emb=rotary)))
        capture, _ = _capture_fixture(runner, 15, dtype=torch.bfloat16)
        positions = torch.arange(15).unsqueeze(0)
        post_cache = []
        for raw_k, raw_v in capture.result_cpu():
            key = raw_k.permute(1, 0, 2).unsqueeze(0)
            cos, sin = rotary(key, positions)
            _, post_k = apply_rotary_pos_emb(
                torch.zeros_like(key), key, cos, sin)
            post_cache.append((post_k, raw_v.permute(1, 0, 2).unsqueeze(0)))
        evidence = validate_pre_rope_capture_against_cache(
            runner, capture, post_cache)
        self.assertTrue(evidence["passed"])
        self.assertTrue(all(row["raw_v_equals_cache_v"]
                            for row in evidence["layers"]))
        post_cache[0][0][0, 0, 0, 0] += 1
        self.assertFalse(validate_pre_rope_capture_against_cache(
            runner, capture, post_cache)["passed"])

    def test_exception_removes_both_projection_hooks(self):
        runner = _FakeRunner()
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with ReKVCapture(runner, 2, 13):
                raise RuntimeError("test failure")
        self.assertFalse(hasattr(runner, "_mmimpress_active_rekv_capture"))
        for layer in runner.layers:
            self.assertEqual(len(layer.self_attn.k_proj._forward_hooks), 0)
            self.assertEqual(len(layer.self_attn.v_proj._forward_hooks), 0)


class ReKVStoreTests(unittest.TestCase):
    def test_valid_spatial_representatives_use_native_mean_and_partial_count(self):
        visual = torch.arange(13 * 2 * 3, dtype=torch.float16).reshape(13, 2, 3)
        reps, counts = representative_keys(visual, [3, 7, 12], 4)
        self.assertEqual(counts, [3, 3, 4, 0])
        self.assertEqual(reps.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(
            reps[0], visual[:3].mean(0).reshape(-1).to(torch.bfloat16)))
        self.assertTrue(torch.equal(
            reps[1], visual[4:7].mean(0).reshape(-1).to(torch.bfloat16)))
        self.assertTrue(torch.equal(reps[3], torch.zeros_like(reps[3])))

    def test_atomic_raw_store_and_contiguous_selected_preads(self):
        runner = _FakeRunner()
        capture, _ = _capture_fixture(runner, 15)
        cache = []
        for key, value in capture.result_cpu():
            cache.append((key.permute(1, 0, 2).unsqueeze(0),
                          value.permute(1, 0, 2).unsqueeze(0)))
        ids = torch.tensor([[1, 2] + [99] * 13 + [3, 4]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rekv"
            persisted = persist_captured_rekv_prefix(
                runner, cache, ids, [100, 100], capture, path,
                image_id="fixture", chunk_size=4,
                image_input_sha256="input-fixture")
            meta = persisted["meta"]
            self.assertEqual(meta["key_representation"], "pre_rope_k_projection")
            self.assertEqual(meta["normal_candidate_chunk_ids"], [0, 1, 2])
            self.assertEqual(meta["valid_spatial_counts"], [3, 3, 4, 0])
            self.assertEqual(meta["representative_metadata_bytes"], 2 * 4 * 6 * 2)
            self.assertEqual(meta["bytes_visual_kv"], 2 * 2 * 13 * 2 * 3 * 2)
            with self.assertRaises(FileExistsError):
                persist_captured_rekv_prefix(
                    runner, cache, ids, [100, 100], capture, path,
                    image_id="fixture", chunk_size=4)

            context = ReKVContext(path, "cpu", runner=runner)
            try:
                self.assertEqual(tuple(context.k_rep.shape), (2, 4, 6))
                self.assertEqual(context.k_rep.dtype, torch.bfloat16)
                self.assertEqual(tuple(context.sys_kv["k"].shape), (2, 2, 2, 3))
                self.assertEqual(context.sys_kv["k"].dtype, torch.bfloat16)
                self.assertEqual(
                    context.initial_context_gpu_bytes,
                    2 * 2 * 2 * 3 * 2 * 2)
                self.assertEqual(
                    context.metadata_gpu_bytes_total,
                    meta["metadata_gpu_bytes_total"])
                statuses = context.reader.drop_all()
                self.assertEqual(len(statuses), 5)
                self.assertTrue(all(row["status"] == "ok" for row in statuses))
                counter = IOCounter()
                rows, selected = context.reader.read_chunks(
                    0, "k", [2, 0, 1], counter)
                self.assertEqual(rows.tolist(), list(range(12)))
                self.assertEqual(counter.summary()["preads"], 1)
                self.assertEqual(tuple(selected.shape), (12, 2, 3))
                self.assertTrue(torch.equal(
                    selected, capture.result_cpu()[0][0][2:14].half()))
                self.assertEqual(tuple(context.read_sep_kv(counter).shape),
                                 (2, 2, 3, 2, 3))
                self.assertEqual(counter.summary()["preads"], 2)
                self.assertEqual(context.reader.read_trace, [
                    (0, "k", 0, 12 * 2 * 3 * 2),
                    (-1, "sep", 0, 2 * 2 * 3 * 2 * 3 * 2),
                ])
                self.assertEqual(context.source_payload_hash,
                                 meta["source_payload_sha256"])
                with self.assertRaises(ValueError):
                    context.reader.read_chunks(0, "k", [3], counter)
                with self.assertRaises(RuntimeError):
                    context.reader.read_full(0, "k", counter)
            finally:
                context.close()


if __name__ == "__main__":
    unittest.main()

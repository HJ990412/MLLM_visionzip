"""CPU SSD, activation, and cache-hit contracts for contextual LLaVA KV25."""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn

from mmimpress.contextual_kv25 import LAYOUT_POLICY, build_selection_plan
from mmimpress.cvpr25 import permutation_sha256, visionzip_repack_order
from mmimpress.piggyback import (
    VisionForwardCapture, persist_captured_visual_prefix,
    stable_json_sha256)
from mmimpress.serve import BIAS, ImageContext, Server
import mmimpress.serve as serve
from mmimpress.store import ChunkReader, IOCounter, write_image_store
from mmimpress.model import LlavaRunner
from tests.test_visdial_turn1_piggyback_core import _fake_runner


VSTART = 3
LAYERS = 2
HEADS = 2
HEAD_DIM = 2
ROW_BYTES = HEADS * HEAD_DIM * 2


def _fixture(root: Path, n: int, variant: str, alpha: float):
    """Write real token-major FP16 files from a full canonical prefix cache."""
    vn = n + 2
    separators = [0, vn - 1]
    scores = torch.full((vn,), float("inf"))
    for original in range(1, vn - 1):
        scores[original] = float((original * 7) % 23)
    descriptors = torch.tensor([
        [float(original % 7), float((original * 5) % 11)]
        for original in range(vn)
    ])
    plan = build_selection_plan(scores, descriptors, separators, alpha,
                                variant, "fixture-image")
    order = plan["stored_to_original"]
    inverse = plan["original_to_stored"]
    layers = []
    for li in range(LAYERS):
        k = torch.empty(1, HEADS, VSTART + vn, HEAD_DIM,
                        dtype=torch.float16)
        v = torch.empty_like(k)
        for head in range(HEADS):
            for position in range(VSTART + vn):
                base = li * 1000 + head * 200 + position
                k[0, head, position] = torch.tensor([base, base + 1])
                v[0, head, position] = torch.tensor([base + 70, base + 71])
        layers.append((k, v))
    meta_flags = {
        "physical_layout": LAYOUT_POLICY,
        "layout_method": LAYOUT_POLICY,
        "layout_policy_version": LAYOUT_POLICY,
        "layout_source": "turn1_normal_inference_piggyback",
        "layout_uses_dataset_question": False,
        "llm_used_for_layout_scoring": False,
        "calibration_questions": 0,
        "global_order_all_layers": True,
        "separator_tail": True,
        "separator_policy": "stable_tail_plus_sidecar",
        "selection_variant": variant,
        "contextual_alpha": float(alpha),
        "selection_seed": 1234,
        "image_id": "fixture-image",
        "k_target": plan["k"],
        "k_dominant": plan["k_dominant"],
        "k_context": plan["k_context"],
        "selection_plan_sha256": stable_json_sha256(plan),
        "key_descriptor_source": (
            "same_turn1_penultimate_vision_k_proj_head_mean_fp32_l2"
            if variant == "contextual" and plan["k_context"] else None),
        "key_descriptor_payload_persisted": False,
        "permutation_sha256": permutation_sha256(order),
        "inverse_permutation_sha256": permutation_sha256(inverse),
    }
    meta = write_image_store(
        root, layers, VSTART, vn, list(range(VSTART + vn)),
        separators, extra=meta_flags, chunk_size=64, dtype="float16",
        probe_heads=0, stored_to_original=order, separator_sidecar=True)
    torch.save({
        "schema_version": 3,
        "layout_policy_version": LAYOUT_POLICY,
        "token_score_original": scores,
        "newline_original": torch.tensor(separators, dtype=torch.int32),
        "stored_to_original": torch.tensor(order, dtype=torch.int32),
        "original_to_stored": torch.tensor(inverse, dtype=torch.int32),
        "selection_plan": plan,
        "selection_plan_sha256": stable_json_sha256(plan),
    }, root / "visionzip_layout.pt")
    return meta, plan, layers


class ContextualStoreHitTests(unittest.TestCase):
    def tearDown(self):
        BIAS.clear()

    def _assert_cache_and_bias(self, ctx, plan, captured):
        meta = ctx.meta
        n, k = plan["n_content"], plan["k"]
        vn = meta["v_token_num"]
        expected_visible = [True] * k + [False] * (n - k) + [True] * 2
        order = plan["stored_to_original"]
        for li in range(LAYERS):
            bias = BIAS[li]
            self.assertEqual(tuple(bias.shape), (1, 1, 1, VSTART + vn))
            self.assertEqual((bias[0, 0, 0, VSTART:] == 0).tolist(),
                             expected_visible)
            for kind, source in (("k", captured[li][0]),
                                 ("v", captured[li][1])):
                cache = (ctx.cache.k if kind == "k" else ctx.cache.v)[li]
                for stored in list(range(k)) + list(range(n, vn)):
                    original = order[stored]
                    actual = cache[0, :, VSTART + stored, :]
                    expected = source[0, :, VSTART + original, :].to(
                        cache.dtype)
                    self.assertTrue(torch.equal(actual, expected),
                                    (kind, li, stored, original))

    def _assert_poisoned_rows_invisible_in_prefill_and_decode(self, ctx, plan):
        k, n = plan["k"], plan["n_content"]
        prefix = int(ctx.meta["prefix_len"])
        base_k = ctx.cache.k[0].float().clone()
        base_v = ctx.cache.v[0].float().clone()
        poison_k, poison_v = base_k.clone(), base_v.clone()
        poison_k[:, :, VSTART + k:VSTART + n] = 10000.0
        poison_v[:, :, VSTART + k:VSTART + n] = -10000.0

        def attention(_module, query, keys, values, mask, **_kw):
            self.assertTrue(torch.all(mask[..., VSTART + k:VSTART + n]
                                      < -1e30))
            return F.scaled_dot_product_attention(
                query, keys, values, attn_mask=mask)

        for qlen, text_len in ((3, 3), (1, 4)):
            query = torch.ones(1, HEADS, qlen, HEAD_DIM)
            text_k = torch.ones(1, HEADS, text_len, HEAD_DIM)
            text_v = torch.full_like(text_k, .25)
            mask = torch.zeros(1, 1, qlen, prefix + text_len)

            def run(image_k, image_v):
                keys = torch.cat((image_k, text_k), dim=2)
                values = torch.cat((image_v, text_v), dim=2)
                with mock.patch.object(serve, "_ORIG_EAGER",
                                       side_effect=attention):
                    return serve._eager_with_bias(
                        SimpleNamespace(layer_idx=0), query, keys, values,
                        mask)

            self.assertTrue(torch.equal(run(base_k, base_v),
                                        run(poison_k, poison_v)))

    def _run_hit(self, ctx, plan, captured, suffix):
        runner = SimpleNamespace(model=SimpleNamespace(device=torch.device("cpu")))
        server = Server(runner)
        seen = []

        def fake_decode(cache, suffix_ids, prefix_len):
            self.assertEqual(prefix_len, int(ctx.meta["prefix_len"]))
            self._assert_cache_and_bias(ctx, plan, captured)
            if plan["n_content"] == 257 and plan["variant"] == "contextual":
                self._assert_poisoned_rows_invisible_in_prefill_and_decode(
                    ctx, plan)
            seen.append([BIAS[li].clone() for li in range(LAYERS)])
            stamp = time.perf_counter()
            return "synthetic", 7, {
                "first_token_at": stamp,
                "finished_at": stamp,
                "prefill_ms": 0.0,
                "decode_ms": 0.0,
                "generated_tokens": 1,
                "generated_token_ids": [7],
                "generated_token_count": 1,
            }

        server._decode = fake_decode
        preads = []
        real_pread = os.pread

        def traced_pread(fd, length, offset):
            payload = real_pread(fd, length, offset)
            preads.append((os.readlink(f"/proc/self/fd/{fd}"),
                           length, offset, len(payload)))
            return payload

        forbidden = AssertionError("online selection or vision work on cache hit")
        with mock.patch("torch.cuda.synchronize", return_value=None), \
             mock.patch("os.pread", side_effect=traced_pread), \
             mock.patch("mmimpress.cvpr25.clip_cls_patch_saliency",
                        side_effect=forbidden), \
             mock.patch("mmimpress.contextual_kv25.contextual_representatives",
                        side_effect=forbidden), \
             mock.patch("mmimpress.serve.CVPR25ChunkSelector._query_scores",
                        side_effect=forbidden):
            result = server.request_cvpr25(
                ctx, static=None, budget=.25, mode="prefix",
                budget_unit="visual_kv", sep_policy="sidecar", cold=False,
                expected_prefix_layout=LAYOUT_POLICY,
                suffix_ids=torch.tensor(suffix))
        self.assertEqual(len(seen), 1)
        self.assertEqual(result["selected_original_ids"],
                         plan["selected_original_ids"])
        self.assertEqual(result["k_target"], plan["k"])
        self.assertEqual(result["query_score_calls"], 0)
        self.assertEqual(result["static_score_calls"], 0)
        self.assertEqual(result["diversity_calls"], 0)
        return result, preads, seen[0]

    def _check_pread_trace(self, ctx, plan, result, calls):
        k = plan["k"]
        n = plan["n_content"]
        vn = ctx.meta["v_token_num"]
        chunks = (k + 63) // 64
        loaded = min(chunks * 64, vn)
        normal = [row for row in calls if row[0].endswith(("/k.bin", "/v.bin"))]
        sep = [row for row in calls if row[0].endswith("/sep_kv.bin")]
        self.assertEqual(len(normal), 2 * LAYERS)
        self.assertEqual(len(sep), 1)
        self.assertEqual(result["selected_chunk_ids_per_layer"],
                         [list(range(chunks))] * LAYERS)
        self.assertEqual(result["actual_loaded_real_rows"], min(loaded, n))
        self.assertEqual(result["actual_loaded_structural_rows"],
                         loaded - min(loaded, n))
        self.assertEqual(result["normal_kv_read_bytes"],
                         2 * LAYERS * loaded * ROW_BYTES)
        self.assertEqual(result["normal_kv_preads"], 2 * LAYERS)
        for path, asked, offset, returned in normal:
            self.assertEqual((offset, asked, returned),
                             (0, loaded * ROW_BYTES, loaded * ROW_BYTES))
            self.assertEqual(os.stat(path).st_size, vn * ROW_BYTES)
        self.assertEqual(sep[0][1:], (os.stat(sep[0][0]).st_size, 0,
                                      os.stat(sep[0][0]).st_size))
        self.assertEqual(result["total_actual_pread_bytes"],
                         sum(row[3] for row in calls))
        self.assertEqual(result["io"]["preads"], len(calls))
        self.assertFalse(any("probe" in row[0] or "descriptor" in row[0]
                             for row in calls))

    def test_policy_activation_hit_io_and_question_invariance(self):
        for n in (63, 255, 256, 257):
            with self.subTest(n=n), tempfile.TemporaryDirectory() as tmp:
                signatures = []
                for variant, alpha in (("dominant", 0.),
                                       ("contextual", .2),
                                       ("random", .2),
                                       ("uniform", .2)):
                    store = Path(tmp) / f"{variant}_{n}"
                    meta, plan, captured = _fixture(store, n, variant, alpha)
                    ctx = ImageContext(store, torch.device("cpu"),
                                       drop_cache=False,
                                       require_v_hidden=False)
                    try:
                        ctx.validate_contextual_visual_kv_layout()
                        self.assertEqual(meta["order"][:plan["k"]],
                                         plan["selected_original_ids"])
                        if variant == "dominant":
                            scores = torch.load(store / "visionzip_layout.pt",
                                                weights_only=True)["token_score_original"]
                            self.assertEqual(meta["order"],
                                             visionzip_repack_order(scores,
                                                                    meta["newline_idx"]))
                        with self.assertRaises((AssertionError, ValueError)):
                            ctx.validate_visual_kv_layout()
                        first, first_calls, first_masks = self._run_hit(
                            ctx, plan, captured, [3, 4])
                        second, second_calls, second_masks = self._run_hit(
                            ctx, plan, captured, [5, 6, 7])
                        self._check_pread_trace(ctx, plan, first, first_calls)
                        self._check_pread_trace(ctx, plan, second, second_calls)
                        self.assertEqual(first["selected_original_ids"],
                                         second["selected_original_ids"])
                        self.assertEqual(first["normal_kv_read_bytes"],
                                         second["normal_kv_read_bytes"])
                        for left, right in zip(first_masks, second_masks):
                            self.assertTrue(torch.equal(left, right))
                        signatures.append((first["normal_kv_read_bytes"],
                                           first["normal_kv_preads"],
                                           first["separator_read_bytes"],
                                           first["separator_preads"]))
                    finally:
                        ctx.close()
                self.assertEqual(len(set(signatures)), 1)

    def test_short_final_chunk_has_no_padding_and_all_rows_are_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "short"
            meta, plan, captured = _fixture(store, 63, "contextual", .2)
            self.assertEqual(meta["v_token_num"], 65)
            with ChunkReader(store, meta, drop_cache=False) as reader:
                counter = IOCounter()
                for kind in ("k", "v"):
                    rows, values = reader.read_chunks(0, kind, [1], counter)
                    self.assertEqual(rows.tolist(), [64])
                    self.assertEqual(tuple(values.shape), (1, HEADS, HEAD_DIM))
                    original = plan["stored_to_original"][64]
                    source = captured[0][0 if kind == "k" else 1]
                    self.assertTrue(torch.equal(values[0],
                        source[0, :, VSTART + original, :]))
            self.assertEqual(counter.summary()["bytes"], 2 * ROW_BYTES)
            self.assertEqual(counter.summary()["preads"], 2)

    def test_active_key_hook_preserves_one_normal_vision_output(self):
        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 2
                self.config = SimpleNamespace(_attn_implementation="eager")
                self.k_proj = nn.Linear(4, 4, bias=False)
                with torch.no_grad():
                    self.k_proj.weight.copy_(torch.eye(4))

            def forward(self, hidden_states, attention_mask=None,
                        output_attentions=False):
                keys = self.k_proj(hidden_states)
                batch, tokens, _ = keys.shape
                weights = torch.arange(batch * 2 * tokens * tokens,
                                       dtype=torch.float32).reshape(
                                           batch, 2, tokens, tokens)
                return hidden_states + keys * 0, (weights if output_attentions
                                                   else None)

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = Attention()

            def forward(self, hidden):
                return self.self_attn(hidden, output_attentions=False)[0]

        class Tower(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(output_attentions=False)
                self.vision_model = nn.Module()
                self.vision_model.encoder = nn.Module()
                self.vision_model.encoder.layers = nn.ModuleList(
                    [Layer() for _ in range(4)])

            def forward(self, pixels):
                for layer in self.vision_model.encoder.layers:
                    pixels = layer(pixels)
                return pixels

        tower = Tower().eval()
        runner = SimpleNamespace(model=SimpleNamespace(
            model=SimpleNamespace(vision_tower=tower),
            config=SimpleNamespace(output_attentions=False),
            device=torch.device("cpu")))
        pixels = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4)
        baseline = tower(pixels)
        capture = VisionForwardCapture(runner, capture_saliency=True,
                                       capture_keys=True)
        with capture:
            instrumented = tower(pixels)
        self.assertTrue(torch.equal(instrumented, baseline))
        self.assertEqual(capture.call_count, 1)
        self.assertEqual(capture.saliency_call_count, 1)
        self.assertEqual(capture.key_call_count, 1)
        self.assertEqual(capture.stats()["key_descriptor_shape"], [2, 4, 2])
        self.assertEqual(capture.stats()["key_descriptor_dtype"],
                         "torch.float32")
        key = tower.vision_model.encoder.layers[-2].self_attn.k_proj(pixels)
        reduced = key[:, 1:, :].float().reshape(2, 4, 2, 2).mean(dim=2)
        norms = torch.linalg.vector_norm(reduced, dim=-1, keepdim=True)
        expected = torch.where(norms > 0,
                               reduced / norms.clamp_min(
                                   torch.finfo(torch.float32).tiny),
                               torch.zeros_like(reduced))
        self.assertTrue(torch.equal(capture.result_keys_cpu(), expected))
        self.assertEqual(len(capture.penultimate_k_proj._forward_hooks), 0)
        self.assertEqual(len(tower._forward_pre_hooks), 0)
        self.assertEqual(len(tower._forward_hooks), 0)

    def test_disabled_key_hook_preserves_legacy_vision_output(self):
        runner = _fake_runner()
        tower = runner.model.model.vision_tower
        pixels = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)
        baseline = tower(pixels)
        capture = VisionForwardCapture(runner, capture_saliency=True,
                                       capture_keys=False)
        with capture:
            instrumented = tower(pixels)
        self.assertTrue(torch.equal(instrumented, baseline))
        self.assertEqual(capture.stats()["capture_keys"], False)
        self.assertEqual(capture.stats()["key_call_count"], 0)
        self.assertEqual(capture.result_cpu().shape, (2, 4))
        self.assertEqual(len(tower._forward_hooks), 0)
        self.assertEqual(len(tower._forward_pre_hooks), 0)
        for layer in tower.vision_model.encoder.layers:
            self.assertEqual(len(layer.self_attn._forward_hooks), 0)
            self.assertEqual(len(layer.self_attn._forward_pre_hooks), 0)


class ContextualPersistenceTests(unittest.TestCase):
    def test_real_persistence_maps_keys_and_preserves_full_original_kv(self):
        class Runner:
            model_id = "synthetic-llava"
            image_token_id = 99
            cfg = SimpleNamespace(
                vision_config=SimpleNamespace(image_size=2, patch_size=1),
                image_grid_pinpoints=[[6, 6]],
            )

            def visual_span(self, ids):
                if ids.ndim == 2:
                    ids = ids[0]
                positions = (ids == self.image_token_id).nonzero(
                    as_tuple=True)[0].tolist()
                assert positions == list(range(VSTART, VSTART + 32))
                return VSTART, len(positions)

            def anyres_layout(self, size, v_num):
                return LlavaRunner.anyres_layout(self, size, v_num)

        runner = Runner()
        vn = 32  # base 2x2, unpadded high-res 4x6, four row separators
        ids = torch.tensor([7, 8, 9] + [99] * vn + [10, 11])
        scores = torch.arange(10 * 4, dtype=torch.float32).view(10, 4) / 40
        keys = torch.arange(10 * 4 * 2, dtype=torch.float32).view(10, 4, 2)
        lengths = torch.linalg.vector_norm(keys, dim=-1, keepdim=True)
        keys = torch.where(lengths > 0, keys / lengths.clamp_min(1e-30),
                           torch.zeros_like(keys))
        layers = []
        for li in range(LAYERS):
            shape = (1, HEADS, VSTART + vn + 2, HEAD_DIM)
            base = torch.arange(torch.tensor(shape).prod().item(),
                                dtype=torch.float16).view(shape)
            layers.append(SimpleNamespace(keys=base + li * 200,
                                          values=base + li * 200 + 1000))
        captured = SimpleNamespace(layers=layers)
        provenance = {
            "capture_saliency": True,
            "vision_call_count": 1,
            "saliency_call_count": 1,
            "extra_vision_forward_calls": 0,
            "extra_saliency_forward_ms": 0.0,
            "saliency_layer_from_end": 2,
            "vision_attention_backend": "eager",
            "global_vision_output_attentions": False,
            "global_model_output_attentions": False,
            "vision_ms": 1.0,
            "vision_num_layers": 4,
            "saliency_layer_index": 2,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stores = {}
            for variant, alpha, desc in ((None, 0., None),
                                         ("dominant", 0., None),
                                         ("contextual", .2, keys)):
                out = root / (variant or "legacy")
                stats = dict(provenance)
                if desc is not None:
                    stats.update({
                        "capture_keys": True, "key_call_count": 1,
                        "key_head_reduction": "fp32_head_mean",
                        "key_normalization": "fp32_l2_zero_stays_zero",
                    })
                persisted = persist_captured_visual_prefix(
                    runner, captured, ids, [1, 2], scores, out,
                    image_id="same-image", chunk_size=64,
                    capture_stats=stats, selection_variant=variant,
                    contextual_alpha=alpha, vision_key_descriptors=desc)
                self.assertTrue(persisted["integrity"]["ok"])
                ctx = ImageContext(out, torch.device("cpu"),
                                   drop_cache=False, require_v_hidden=False)
                try:
                    if variant is None:
                        ctx.validate_visual_kv_layout()
                    else:
                        ctx.validate_contextual_visual_kv_layout()
                    meta = ctx.meta
                    self.assertEqual(meta["n_spatial"], 28)
                    self.assertEqual(meta["newline_stored"], [28, 29, 30, 31])
                    self.assertEqual(meta["bytes_probe_sidecar"], 0)
                    self.assertFalse(meta.get("key_descriptor_payload_persisted",
                                              False))
                    order = meta["order"]
                    for li in range(LAYERS):
                        for kind, source in (("k", layers[li].keys),
                                             ("v", layers[li].values)):
                            stored = ctx.reader.read_full(li, kind)
                            restored = torch.empty_like(stored)
                            restored[torch.tensor(order)] = stored
                            expected = source[0, :, VSTART:VSTART + vn].permute(
                                1, 0, 2)
                            self.assertTrue(torch.equal(restored, expected))
                    stores[variant] = list(order)
                    if variant == "contextual":
                        self.assertEqual(meta["k_target"], 7)
                        self.assertEqual(meta["k_context"], 1)
                        artifact = torch.load(out / "visionzip_layout.pt",
                                              weights_only=True)
                        self.assertEqual(artifact["schema_version"], 3)
                        self.assertEqual(artifact["descriptor_protocol"][
                            "token_mapping"],
                            "base_then_tiled_anyres_unpad_structural_zero")
                        self.assertEqual(sum(artifact["selection_plan"][
                            "cluster_sizes"]), 22)
                finally:
                    ctx.close()
            self.assertEqual(stores[None], stores["dominant"])
            self.assertNotEqual(stores["dominant"], stores["contextual"])


if __name__ == "__main__":
    unittest.main()

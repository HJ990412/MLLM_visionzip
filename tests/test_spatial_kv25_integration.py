"""CPU persistence and SSD-hit contract for the opt-in spatial layout."""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from mmimpress.model import LlavaRunner
from mmimpress.piggyback import persist_captured_visual_prefix
from mmimpress.serve import BIAS, ImageContext, Server

VSTART = 3
VNUM = 32
LAYERS = 2
HEADS = 2
HEAD_DIM = 2


class _Runner:
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
        assert positions == list(range(VSTART, VSTART + VNUM))
        return VSTART, VNUM

    def anyres_layout(self, size, v_num):
        return LlavaRunner.anyres_layout(self, size, v_num)


def _canonical():
    layers = []
    for layer in range(LAYERS):
        shape = (1, HEADS, VSTART + VNUM + 2, HEAD_DIM)
        tensor = torch.arange(torch.tensor(shape).prod().item(),
                              dtype=torch.float16).view(shape)
        layers.append(SimpleNamespace(keys=tensor + layer * 200,
                                      values=tensor + layer * 200 + 1000))
    return SimpleNamespace(layers=layers), layers


class SpatialPersistenceTests(unittest.TestCase):
    def tearDown(self):
        BIAS.clear()

    def test_spatial_opt_in_store_and_hit_preserve_original_kv(self):
        runner = _Runner()
        ids = torch.tensor([7, 8, 9] + [99] * VNUM + [10, 11])
        scores = torch.arange(40, dtype=torch.float32).view(10, 4) / 40
        captured, layers = _canonical()
        capture_stats = {
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
            "capture_keys": False,
            "key_call_count": 0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contexts = {}
            plans = {}
            try:
                for name, variant in (("index", "uniform"),
                                      ("spatial", "spatial_uniform")):
                    path = root / name
                    receipt = persist_captured_visual_prefix(
                        runner, captured, ids, [1, 2], scores, path,
                        image_id="fixed-image", chunk_size=64,
                        capture_stats=capture_stats,
                        selection_variant=variant, contextual_alpha=.2)
                    self.assertTrue(receipt["integrity"]["ok"])
                    self.assertEqual(
                        receipt["timing_ms"]["geometry_mapping_ms"] > 0,
                        variant == "spatial_uniform")
                    ctx = ImageContext(path, torch.device("cpu"),
                                       drop_cache=False,
                                       require_v_hidden=False)
                    contexts[name] = ctx
                    if variant == "spatial_uniform":
                        ctx.validate_spatial_visual_kv_layout()
                    else:
                        ctx.validate_contextual_visual_kv_layout()
                    layout = torch.load(path / "visionzip_layout.pt",
                                        weights_only=True)
                    plan = layout["selection_plan"]
                    plans[name] = plan
                    self.assertEqual(plan["k"], 7)
                    self.assertEqual(plan["k_context"], 1)
                    self.assertEqual(ctx.meta["order"][:7],
                                     plan["selected_original_ids"])
                    order = ctx.meta["order"]
                    for layer in range(LAYERS):
                        for kind, source in (("k", layers[layer].keys),
                                             ("v", layers[layer].values)):
                            stored = ctx.reader.read_full(layer, kind)
                            restored = torch.empty_like(stored)
                            restored[torch.tensor(order)] = stored
                            expected = source[0, :, VSTART:VSTART+VNUM].permute(
                                1, 0, 2)
                            self.assertTrue(torch.equal(restored, expected))
                index = plans["index"]
                spatial = plans["spatial"]
                self.assertEqual(index["dominant_ids"],
                                 spatial["dominant_ids"])
                self.assertEqual(index["contextual_ids"],
                                 spatial["spatial"]["index_uniform_aux_ids"])
                records = spatial["spatial"]["coordinates"]["records"]
                for branch in ("base", "high"):
                    quota = sum(records[i]["branch"] == branch
                                for i in index["contextual_ids"])
                    self.assertEqual(
                        spatial["spatial"]["branch_quotas"][branch], quota)
                    self.assertEqual(sum(records[i]["branch"] == branch
                                         for i in spatial["contextual_ids"]),
                                     quota)
                ctx = contexts["spatial"]
                server = Server(SimpleNamespace(
                    model=SimpleNamespace(device=torch.device("cpu"))))
                def decode(cache, suffix, prefix_len):
                    self.assertEqual(prefix_len, VSTART+VNUM)
                    for layer in range(LAYERS):
                        visible = (BIAS[layer][0, 0, 0, VSTART:] == 0).tolist()
                        self.assertEqual(visible,
                                         [True] * 7 + [False] * 21 + [True] * 4)
                    stamp = time.perf_counter()
                    return "synthetic", 7, {
                        "first_token_at": stamp,
                        "finished_at": stamp,
                        "prefill_ms": 0.,
                        "decode_ms": 0.,
                        "generated_tokens": 1,
                        "generated_token_ids": [7],
                        "generated_token_count": 1,
                    }
                server._decode = decode
                with mock.patch("torch.cuda.synchronize", return_value=None):
                    result = server.request_cvpr25(
                        ctx, static=None, budget=.25, mode="prefix",
                        budget_unit="visual_kv", sep_policy="sidecar",
                        cold=False,
                        expected_prefix_layout="visionzip_spatial_original_v1",
                        suffix_ids=torch.tensor([3, 4]))
                self.assertEqual(result["selected_original_ids"],
                                 spatial["selected_original_ids"])
                self.assertEqual(result["attended_content_kv_count"], 7)
                self.assertEqual(result["query_score_calls"], 0)
                self.assertEqual(result["static_score_calls"], 0)
                self.assertEqual(result["normal_kv_preads"], 2 * LAYERS)
                self.assertEqual(result["separator_preads"], 1)
                self.assertEqual(result.get("probe_read_bytes", 0), 0)
            finally:
                for ctx in contexts.values():
                    ctx.close()


if __name__ == "__main__":
    unittest.main()

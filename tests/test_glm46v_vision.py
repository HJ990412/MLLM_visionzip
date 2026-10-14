"""Independent CPU checks for the frozen GLM vision score/mapping contract."""

import unittest
import torch

from mmimpress.glm46v.vision import (
    VisionScoreCapture,
    merge_native_scores,
    native_patch_coordinates,
    received_attention_scores,
    stable_importance_order,
)


def _dense_received_reference(query, key, boundaries):
    # Independent whole-frame dense attention; no port helper or blocking.
    reference = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        head_scores = []
        for head in range(query.shape[1]):
            logits = query[start:end, head] @ key[start:end, head].T
            logits = logits * (query.shape[-1] ** -0.5)
            probability = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
            head_scores.append(probability.float().sum(0))
        reference.append(torch.stack(head_scores).mean(0))
    return torch.cat(reference)


def _check_blocking_retains_full_key_denominator():
    generator = torch.Generator().manual_seed(1234)
    query = torch.randn(11, 3, 8, generator=generator)
    key = torch.randn(11, 3, 8, generator=generator)
    boundaries = [0, 4, 11]
    reference = _dense_received_reference(query, key, boundaries)
    for block in [1, 2, 3, 128]:
        actual = received_attention_scores(query, key, torch.tensor(boundaries), query_block_size=block)
        torch.testing.assert_close(actual, reference, atol=2e-7, rtol=2e-7)
        assert abs(float(actual[:4].sum()) - 4) < 1e-6
        assert abs(float(actual[4:].sum()) - 7) < 1e-6


def _check_frame_isolation_and_no_key_blocking():
    # A key that attracts every query must receive ~number_of_queries even
    # when query blocks are one row. Softmax over a single key would fail.
    query = torch.ones(7, 2, 1)
    key = torch.zeros_like(query)
    key[2] = 20
    boundaries = torch.tensor([0, 4, 7])
    actual = received_attention_scores(query, key, boundaries, query_block_size=1)
    assert actual[2] > 3.999
    torch.testing.assert_close(actual[4:], torch.ones(3))
    changed = query.clone()
    changed[4:] = 100
    assert torch.equal(actual[:4], received_attention_scores(changed, key, boundaries, query_block_size=1)[:4])


def _check_native_patch_coordinates_and_conv_merge_reference():
    grid = torch.tensor([[1, 4, 6], [2, 2, 4]])
    coordinates = native_patch_coordinates(grid)
    # Raster coordinates created independently from the processor formula:
    # block index, then local 2x2 position. This reference also covers frames.
    expected = []
    for temporal, height, width in grid.tolist():
        for frame in range(temporal):
            for group in range((height // 2) * (width // 2)):
                group_h, group_w = divmod(group, width // 2)
                expected.extend([[frame, group_h * 2 + dh, group_w * 2 + dw]
                                 for dh, dw in [(0, 0), (0, 1), (1, 0), (1, 1)]])
    assert coordinates.tolist() == expected
    scores = torch.arange(len(expected), dtype=torch.float32)
    native_groups = merge_native_scores(scores, grid)
    # Independent spatial Conv2d illustrates its native receptive fields.
    reference = []
    offset = 0
    for temporal, height, width in grid.tolist():
        for _ in range(temporal):
            raster = torch.empty(height, width)
            for index in range(height * width):
                _, row, col = expected[offset + index]
                raster[row, col] = scores[offset + index]
            reference.extend(torch.nn.functional.conv2d(
                raster[None, None], torch.full((1, 1, 2, 2), .25), stride=2,
            ).flatten().tolist())
            offset += height * width
    assert native_groups.tolist() == reference


def _check_stable_descending_rank_with_ties():
    assert stable_importance_order(torch.tensor([2., 3., 3., -1., 2.])).tolist() == [1, 2, 0, 4, 3]


def _check_invalid_frame_boundaries_rejected():
    for boundaries in [[1, 4], [0, 3], [0, 2, 2, 4], [0, 5, 4]]:
        with unittest.TestCase().assertRaises(ValueError):
            received_attention_scores(torch.ones(4, 2, 3), torch.ones(4, 2, 3), torch.tensor(boundaries))


def _check_invalid_native_geometry_rejected():
    for grid in [[[1, 3, 4]], [[0, 2, 2]], [[1, 2]], [[1., 2., 2.]]]:
        with unittest.TestCase().assertRaises(ValueError):
            native_patch_coordinates(torch.tensor(grid))


def _check_invalid_nonfinite_shape_and_block_rejected():
    with unittest.TestCase().assertRaises(ValueError):
        stable_importance_order(torch.tensor([float("nan")]))
    with unittest.TestCase().assertRaises(ValueError):
        received_attention_scores(torch.ones(4, 2, 3), torch.ones(4, 2, 3), torch.tensor([0, 4]), query_block_size=0)
    with unittest.TestCase().assertRaises(ValueError):
        merge_native_scores(torch.ones(3), torch.tensor([[1, 2, 2]]))


def _tiny_native_model():
    from transformers.models.glm4v.configuration_glm4v import Glm4vVisionConfig
    from transformers.models.glm4v.modeling_glm4v import Glm4vVisionModel
    config = Glm4vVisionConfig(
        depth=2, hidden_size=16, num_heads=2, out_hidden_size=32,
        intermediate_size=32, image_size=8, patch_size=2,
        temporal_patch_size=2, spatial_merge_size=2,
        attention_dropout=0., attention_bias=False,
    )
    config._attn_implementation = "sdpa"
    return Glm4vVisionModel(config).eval()


def _check_actual_native_sdpa_hook_preserves_output_and_qk_score():
    torch.manual_seed(1234)
    model = _tiny_native_model()
    grid = torch.tensor([[1, 2, 4]], dtype=torch.long)
    pixels = torch.randn(8, 3 * 2 * 2 * 2)
    independent = {}

    def projected(module, args, output):
        independent["qkv"] = output.detach().clone()

    def positions(module, args, kwargs):
        independent["positions"] = tuple(x.detach().clone() for x in kwargs["position_embeddings"])

    with torch.inference_mode():
        canonical = model(pixels, grid_thw=grid).pooler_output.clone()
        handles = [
            model.blocks[-1].attn.qkv.register_forward_hook(projected),
            model.blocks[-1].attn.register_forward_pre_hook(positions, with_kwargs=True),
        ]
        with VisionScoreCapture(model, query_block_size=3) as capture:
            observed = model(pixels, grid_thw=grid).pooler_output.clone()
        for handle in handles:
            handle.remove()
    assert torch.equal(canonical, observed)
    assert capture.call_count == 1
    assert capture._qk is None and capture._position_embeddings is None and not capture._handles
    assert capture.geometry["merged_visual_token_count"] == 2
    assert capture.qk_capture_bytes == 8 * 2 * 2 * 8 * 4

    qkv = independent["qkv"].reshape(8, 3, 2, 8)
    cos, sin = independent["positions"]
    q, k = qkv[:, 0].float(), qkv[:, 1].float()
    # Independent native vision split-half rotation, no port/native helper.
    q_rot = torch.cat([-q[..., 4:], q[..., :4]], -1)
    k_rot = torch.cat([-k[..., 4:], k[..., :4]], -1)
    q = q * cos[:, None].float() + q_rot * sin[:, None].float()
    k = k * cos[:, None].float() + k_rot * sin[:, None].float()
    patch_reference = _dense_received_reference(q, k, [0, 8])
    group_reference = torch.tensor([patch_reference[:4].mean(), patch_reference[4:].mean()])
    torch.testing.assert_close(capture.scores, group_reference, atol=2e-7, rtol=2e-7)


def _check_hook_failure_and_missing_forward_remove_all_handles():
    model = _tiny_native_model()
    capture = VisionScoreCapture(model)
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "one complete"):
        with capture:
            pass
    assert not capture._handles
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "fixture"):
        with VisionScoreCapture(model) as failed:
            raise RuntimeError("fixture")
    assert not failed._handles and failed._qk is None


class GlmVisionTests(unittest.TestCase):
    """Expose independent reference cases to the repository unittest runner."""


for _name, _case in list(globals().items()):
    if _name.startswith("_check_") and callable(_case):
        setattr(GlmVisionTests, "test_" + _name[len("_check_"):], staticmethod(_case))

del _name, _case


if __name__ == "__main__":
    unittest.main()

"""CPU contracts for LLaVA image-only visual-KV retention and real preads."""

from __future__ import annotations

import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

from mmimpress.cvpr25 import (permutation_sha256, visionzip_repack_order,
                              visual_kv_budget_count)
from mmimpress.serve import BIAS, CVPR25ChunkSelector, ImageContext
from mmimpress.store import ChunkReader, IOCounter, write_image_store


def _fixture(root: Path, n: int, separators: int = 2):
    """Write actual FP16 K/V files with one independent, score-sorted order."""
    assert separators == 2
    vn, vstart, layers, heads, hd = n + separators, 3, 2, 2, 2
    newline = [0, vn - 1]
    real = [i for i in range(vn) if i not in newline]
    scores = torch.full((vn,), float("inf"), dtype=torch.float32)
    for i in real:
        scores[i] = float((i * 7) % 19) / 19.0
    # Independent expected order, with score ties resolved by original ID.
    expected_real = sorted(real, key=lambda i: (-float(scores[i]), i))
    perm = visionzip_repack_order(scores, newline)
    assert perm == expected_real + newline
    inverse = [0] * vn
    for stored, original in enumerate(perm):
        inverse[original] = stored
    tensors = []
    for li in range(layers):
        shape = (1, heads, vstart + vn, hd)
        base = torch.arange(math.prod(shape), dtype=torch.float16).view(shape)
        tensors.append((base + li * 100, base + li * 100 + 1000))
    flags = {
        "physical_layout": "visionzip_image_only",
        "layout_method": "visionzip_image_only",
        "layout_source": "turn1_normal_inference_piggyback",
        "visual_kv_source": "turn1_captured_past_key_values",
        "layout_uses_dataset_question": False,
        "llm_used_for_layout_scoring": False,
        "calibration_questions": 0,
        "global_order_all_layers": True,
        "separator_tail": True,
        "separator_policy": "stable_tail_plus_sidecar",
        "permutation_sha256": permutation_sha256(perm),
        "inverse_permutation_sha256": permutation_sha256(inverse),
    }
    meta = write_image_store(
        root, tensors, vstart, vn, list(range(vstart + vn)), newline,
        extra=flags, chunk_size=64, dtype="float16", probe_heads=0,
        stored_to_original=perm, separator_sidecar=True)
    torch.save({
        "schema_version": 2,
        "token_score_original": scores,
        "newline_original": torch.tensor(newline, dtype=torch.int32),
        "stored_to_original": torch.tensor(perm, dtype=torch.int32),
        "original_to_stored": torch.tensor(inverse, dtype=torch.int32),
        "layout_uses_dataset_question": False,
        "llm_used_for_layout_scoring": False,
        "calibration_questions": 0,
    }, root / "visionzip_layout.pt")
    return meta, expected_real, tensors


class BudgetArithmeticTests(unittest.TestCase):
    def test_edges_and_invalid_inputs(self):
        for n in (0, 1, 63, 64, 65, 127, 128, 129, 256, 349, 2200):
            with self.subTest(n=n):
                self.assertEqual(visual_kv_budget_count(n, 0), 0)
                self.assertEqual(visual_kv_budget_count(n, 1), n)
                self.assertEqual(visual_kv_budget_count(n, 0.25),
                                 (n + 3) // 4)
        for n, ratio in ((-1, 0.25), (True, 0.25), (1.5, 0.25),
                         (1, -0.1), (1, 1.1), (1, float("nan")),
                         (1, float("inf")), (1, "0.25")):
            with self.subTest(n=n, ratio=ratio), self.assertRaises(
                    (TypeError, ValueError)):
                visual_kv_budget_count(n, ratio)


class RealKV25PathTests(unittest.TestCase):
    def tearDown(self):
        BIAS.clear()

    def _run(self, n):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            meta, independent_real_order, tensors = _fixture(root, n)
            ctx = ImageContext(root, torch.device("cpu"), drop_cache=False,
                               require_v_hidden=False)
            try:
                ctx.validate_visual_kv_layout()
                runner = SimpleNamespace(model=SimpleNamespace(
                    device=torch.device("cpu")))
                fresh = ctx.cache.new_request()
                selector = CVPR25ChunkSelector(
                    runner, ctx, static=None, budget=0.25, mode="prefix",
                    sep_policy="sidecar", image_id="fixture",
                    budget_unit="visual_kv")
                calls = []
                original_pread = os.pread

                def traced_pread(fd, length, offset):
                    path = os.readlink(f"/proc/self/fd/{fd}")
                    payload = original_pread(fd, length, offset)
                    calls.append((path, length, offset, len(payload)))
                    return payload

                import builtins
                import io
                original_builtin_open = builtins.open
                original_io_open = io.open

                def guarded_open(original, path, *args, **kwargs):
                    if isinstance(path, (str, os.PathLike)):
                        name = Path(path).name
                        if name in {"visionzip_layout.pt", "static.pt",
                                    "probe_k.bin"} or "score" in name:
                            raise AssertionError(f"forbidden hit open: {path}")
                    return original(path, *args, **kwargs)

                with mock.patch("torch.cuda.synchronize", return_value=None), \
                     mock.patch("os.pread", side_effect=traced_pread), \
                     mock.patch("builtins.open",
                                side_effect=lambda path, *a, **kw:
                                guarded_open(original_builtin_open, path,
                                             *a, **kw)), \
                     mock.patch("io.open",
                                side_effect=lambda path, *a, **kw:
                                guarded_open(original_io_open, path,
                                             *a, **kw)):
                    selector.prepare(text_emb=None)

                k = (n + 3) // 4
                m = (k + 63) // 64
                loaded = min(m * 64, n + 2)
                real_loaded = min(loaded, n)
                expected_mask = [True] * k + [False] * (n - k) + [True] * 2
                expected_original = independent_real_order[:k]
                self.assertEqual(selector.selected_chunk_ids_per_layer,
                                 [list(range(m))] * 2)
                self.assertEqual(selector.kept_tokens_per_layer, [k + 2] * 2)
                self.assertEqual(meta["order"][:k], expected_original)
                stats = selector.stats()
                self.assertEqual(stats["k_target"], k)
                self.assertEqual(stats["attended_content_kv_count"], k)
                self.assertEqual(stats["actual_loaded_real_rows"], real_loaded)
                self.assertEqual(stats["unused_loaded_real_rows"],
                                 real_loaded - k)
                self.assertEqual(stats["actual_loaded_structural_rows"],
                                 loaded - real_loaded)
                self.assertEqual(stats["structural_bytes_in_normal_payload"],
                                 (loaded - real_loaded) * 2 * 2 * 2 * 2 * 2)
                self.assertEqual(stats["normal_chunks_read"], m)
                self.assertEqual(stats["budget_unit"], "visual_kv")
                self.assertEqual(stats["query_score_calls"], 0)
                self.assertEqual(stats["static_score_calls"], 0)
                self.assertEqual(stats["diversity_calls"], 0)
                self.assertEqual(len(calls), 5)
                row_bytes = 2 * 2 * 2  # heads * head_dim * FP16 bytes
                for path, asked, offset, returned in calls:
                    self.assertEqual(offset, 0)
                    self.assertEqual(asked, returned)
                    if path.endswith("/k.bin") or path.endswith("/v.bin"):
                        self.assertEqual(returned, loaded * row_bytes)
                        self.assertEqual(os.stat(path).st_size,
                                         (n + 2) * row_bytes)
                    else:
                        self.assertTrue(path.endswith("/sep_kv.bin"))
                        self.assertEqual(returned, os.stat(path).st_size)
                self.assertFalse(any("probe" in x[0] or "score" in x[0]
                                     or "visionzip_layout" in x[0]
                                     for x in calls))
                self.assertEqual(selector.io.summary()["bytes"],
                                 sum(call[3] for call in calls))
                self.assertEqual(selector.io.summary()["preads"], len(calls))
                # The exact additive bias, not merely its count, governs every
                # KV head in both prefill and decode.
                for li in range(2):
                    bias = BIAS[li][0, 0, 0].cpu()
                    self.assertEqual(bias[:3].tolist(), [0.0] * 3)
                    visible = (bias[3:] == 0).tolist()
                    self.assertEqual(visible, expected_mask)
                    for head in range(2):
                        self.assertEqual(ctx.cache.k[li][0, head, 3:3 + k]
                                         .shape[0], k)
                if n == 349:
                    self._finite_sentinel_prefill_decode(ctx, n, k)
                    # A new request clears previously scattered visual KV.
                    ctx.cache.k[0][0, :, 3 + k:3 + n].fill_(777)
                    ctx.cache.new_request()
                    self.assertTrue(torch.equal(
                        ctx.cache.k[0][0, :, 3:],
                        torch.zeros_like(ctx.cache.k[0][0, :, 3:])))
                    self.assertTrue(torch.equal(
                        ctx.cache.k[0][0, :, :3],
                        tensors[0][0][0, :, :3].to(ctx.cache.dtype)))
                del fresh
            finally:
                ctx.close()

    def test_all_boundary_geometries_and_real_pread_ranges(self):
        for n in (1, 63, 64, 65, 127, 128, 129, 256, 349, 2200):
            with self.subTest(n=n):
                self._run(n)

    def _finite_sentinel_prefill_decode(self, ctx, n, k):
        import mmimpress.serve as serve

        prefix = int(ctx.meta["prefix_len"])
        hd = int(ctx.meta["head_dim"])
        base_k = ctx.cache.k[0].float().clone()
        base_v = ctx.cache.v[0].float().clone()
        altered_k, altered_v = base_k.clone(), base_v.clone()
        altered_k[:, :, 3 + k:3 + n] = 10000.0
        altered_v[:, :, 3 + k:3 + n] = -10000.0

        def run(q_len, extra_text, key, value):
            text_k = torch.ones(1, 2, extra_text, hd)
            text_v = torch.full_like(text_k, 0.25)
            keys = torch.cat((key, text_k), dim=2)
            values = torch.cat((value, text_v), dim=2)
            query = torch.ones(1, 2, q_len, hd)
            causal = torch.zeros(1, 1, q_len, prefix + extra_text)
            for qi in range(q_len):
                causal[:, :, qi, prefix + extra_text - q_len + qi + 1:] = -1e30
            seen = []

            def attention(_module, q, keys_, values_, mask, **_kwargs):
                seen.append(mask)
                return F.scaled_dot_product_attention(q, keys_, values_,
                                                       attn_mask=mask)

            with mock.patch.object(serve, "_ORIG_EAGER", side_effect=attention):
                output = serve._eager_with_bias(SimpleNamespace(layer_idx=0),
                                                query, keys, values, causal)
            self.assertEqual(seen[0].shape[-1], prefix + extra_text)
            self.assertTrue(torch.all(seen[0][..., 3 + k:3 + n] < -1e30))
            return output

        for q_len, extra_text in ((3, 3), (1, 4)):
            reference = run(q_len, extra_text, base_k, base_v)
            poisoned = run(q_len, extra_text, altered_k, altered_v)
            self.assertTrue(torch.equal(reference, poisoned))

    def test_question_independence_and_full_retention_logical_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            meta, _, captured = _fixture(root, 129)
            ctx = ImageContext(root, torch.device("cpu"), drop_cache=False,
                               require_v_hidden=False)
            try:
                ctx.validate_visual_kv_layout()
                runner = SimpleNamespace(model=SimpleNamespace(
                    device=torch.device("cpu")))
                selections = []
                forbidden = AssertionError("online image/question scoring")
                with mock.patch("torch.cuda.synchronize", return_value=None), \
                     mock.patch("mmimpress.cvpr25.clip_cls_patch_saliency",
                                side_effect=forbidden), \
                     mock.patch("mmimpress.serve.CVPR25ChunkSelector._query_scores",
                                side_effect=forbidden):
                    for text in (torch.zeros(2, 4), torch.ones(9, 4)):
                        ctx.cache.new_request()
                        selector = CVPR25ChunkSelector(
                            runner, ctx, static=None, budget=0.25,
                            mode="prefix", sep_policy="sidecar",
                            image_id="same-image", budget_unit="visual_kv")
                        selector.prepare(text_emb=text)
                        selections.append((
                            selector.stats()["selected_original_ids"],
                            [BIAS[li].clone() for li in range(2)]))
                    ctx.cache.new_request()
                    full = CVPR25ChunkSelector(
                        runner, ctx, static=None, budget=1.0,
                        mode="prefix", sep_policy="sidecar",
                        image_id="same-image", budget_unit="visual_kv")
                    full.prepare(text_emb=None)
                self.assertEqual(selections[0][0], selections[1][0])
                for before, after in zip(selections[0][1], selections[1][1]):
                    self.assertTrue(torch.equal(before, after))
                self.assertEqual(full.stats()["attended_content_kv_count"], 129)
                self.assertEqual(full.stats()["keep_count_per_layer"],
                                 [meta["v_token_num"]] * 2)
                order = torch.tensor(meta["order"], dtype=torch.long)
                for li in range(2):
                    for kind, source in (("k", captured[li][0]),
                                         ("v", captured[li][1])):
                        physical = (ctx.cache.k if kind == "k" else
                                    ctx.cache.v)[li][0, :, 3:].permute(1, 0, 2)
                        restored = torch.empty_like(physical)
                        restored[order] = physical
                        expected = source[0, :, 3:].permute(1, 0, 2).to(
                            ctx.cache.dtype)
                        self.assertTrue(torch.equal(restored, expected))
            finally:
                ctx.close()

    def test_short_final_chunk_has_no_padding_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            meta, _, _ = _fixture(Path(td), 63)  # V=65, final chunk has one row
            counter = IOCounter()
            with ChunkReader(Path(td), meta, drop_cache=False) as reader:
                for kind in ("k", "v"):
                    rows, payload = reader.read_chunks(0, kind, [1], counter)
                    self.assertEqual(rows.tolist(), [64])
                    self.assertEqual(tuple(payload.shape), (1, 2, 2))
            self.assertEqual(counter.summary()["bytes"], 2 * 2 * 2 * 2)
            self.assertEqual(counter.summary()["preads"], 2)

    def test_bad_layout_and_zero_content_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            meta, _, _ = _fixture(root, 65)
            ctx = ImageContext(root, torch.device("cpu"), drop_cache=False,
                               require_v_hidden=False)
            try:
                artifact = torch.load(root / "visionzip_layout.pt",
                                      weights_only=True)
                artifact["token_score_original"][meta["order"][0]] = -99.0
                artifact["token_score_original"][meta["order"][64]] = 99.0
                torch.save(artifact, root / "visionzip_layout.pt")
                with self.assertRaises((AssertionError, ValueError)):
                    ctx.validate_visual_kv_layout()
            finally:
                ctx.close()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _fixture(root, 0)
            ctx = ImageContext(root, torch.device("cpu"), drop_cache=False,
                               require_v_hidden=False)
            try:
                with self.assertRaises((AssertionError, ValueError)):
                    ctx.validate_visual_kv_layout()
            finally:
                ctx.close()


if __name__ == "__main__":
    unittest.main()

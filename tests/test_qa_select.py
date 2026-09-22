"""CPU-only contracts for the QA-Select25 serving baseline.

The tests use tiny synthetic tensors and stores.  They exercise the logical
SparseVLM selection, its physical chunk plan and accounting, the fail-closed
raster-layout contract, and same-Turn-1 capture/persistence without loading a
model, touching CUDA, or downloading artifacts.
"""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from mmimpress import sparsevlm as sv
from mmimpress.piggyback import (
    DecoderVisualHiddenCapture,
    persist_captured_raster_prefix,
)
from mmimpress.serve import (
    BIAS,
    ImageContext,
    QASelectLayerSelector,
    contiguous_runs,
    qa_select_plan,
)
from mmimpress.store import ChunkReader, IOCounter, load_meta


class QASelectPlanTests(unittest.TestCase):
    def test_query_dependent_deterministic_fixed_quarter_selection(self):
        # One rater and one probe head are enough to prove that the Q/K score is
        # query-dependent.  Query 1 aligns with visual token 0; query 2 aligns
        # with visual token 1.
        keys = torch.tensor([[
            [4.0, 0.0], [0.0, 4.0], [2.0, 0.0], [0.0, 2.0],
        ]])
        query_x = torch.tensor([[[1.0, 0.0]]])
        query_y = torch.tensor([[[0.0, 1.0]]])

        score_x = sv.rater_visual_scores_from_qk(
            query_x, keys, [0], 0, 4, head_reduce="mean")
        score_y = sv.rater_visual_scores_from_qk(
            query_y, keys, [0], 0, 4, head_reduce="mean")
        first = qa_select_plan(score_x, 0.25, [], 4, 2)
        repeat = qa_select_plan(score_x, 0.25, [], 4, 2)
        changed = qa_select_plan(score_y, 0.25, [], 4, 2)

        self.assertEqual(first, repeat)
        self.assertEqual(first["selected_tokens"], [0])
        self.assertEqual(changed["selected_tokens"], [1])
        self.assertNotEqual(first["selected_tokens"],
                            changed["selected_tokens"])
        self.assertEqual(len(first["selected_tokens"]), 1)

    def test_separator_exclusion_and_exact_token_chunk_run_plan(self):
        v_num, chunk_size = 20, 4
        separators = [3, 6, 11, 15]
        # Sixteen selectable tokens -> ceil(16 * .25) == four.  Give every
        # separator a still-higher score to prove it is excluded from Top-k.
        scores = torch.linspace(0.0, 1.0, v_num)
        scores[torch.tensor(separators)] = 1000.0
        scores[0], scores[7], scores[8], scores[17] = 100, 90, 80, 70
        per_head = torch.stack([scores - 0.25, scores + 0.25])

        plan = qa_select_plan(
            per_head, 0.25, separators, v_num, chunk_size)

        self.assertEqual(plan["selected_tokens"], [0, 7, 8, 17])
        self.assertTrue(set(plan["selected_tokens"]).isdisjoint(separators))
        self.assertEqual(plan["selected_chunks"], [0, 1, 2, 4])
        self.assertEqual(plan["contiguous_runs"], 2)
        self.assertEqual(plan["contiguous_run_lengths"], [3, 1])
        self.assertEqual(contiguous_runs([4, 2, 1, 0, 2]), (2, [3, 1]))
        self.assertEqual(contiguous_runs([]), (0, []))


def _valid_qa_meta():
    return {
        "physical_layout": "raster",
        "layout_method": "raster",
        "layout_source": "turn1_normal_inference_piggyback",
        "reordered": False,
        "order_is_per_layer": False,
        "layout_uses_dataset_question": False,
        "layout_uses_generated_answer": False,
        "calibration_questions": 0,
        "future_questions_used": 0,
        "qa_select_compatible": True,
        "turn1_normal_inference": True,
        "visual_kv_source": "turn1_captured_past_key_values",
        "visual_hidden_source": "same_turn1_decoder_layer0_input",
        "separate_vision_forward": False,
        "separate_prefix_forward": False,
        "separate_model_forward_for_visual_hidden": False,
        "capture_provenance_validated": True,
        "hidden_capture": {
            "capture_source": "same_turn1_normal_multimodal_prefill",
            "visual_hidden_capture_count": 1,
        },
        "num_layers": 2,
        "v_token_num": 12,
        "newline_idx": [2, 11],
        "n_spatial": 10,
        "probe_heads": 1,
        "probe_heads_required_for_serving": 1,
        "bytes_probe_sidecar": 48,
        "bytes_separator_sidecar": 32,
    }


def _validator_context(meta=None, hidden_rows=12):
    context = object.__new__(ImageContext)
    context.dir = Path("/synthetic/qa-raster")
    context.meta = copy.deepcopy(meta or _valid_qa_meta())
    context.v_hidden = (None if hidden_rows is None else
                        torch.zeros(hidden_rows, 4, dtype=torch.float16))
    context._separator_positions = [[2, 11], [2, 11]]
    return context


class QASelectLayoutValidatorTests(unittest.TestCase):
    def test_accepts_only_identity_raster_with_required_sidecars(self):
        context = _validator_context()
        context.validate_qa_select_layout()
        self.assertTrue(context._qa_select_layout_validated)

        explicit_identity = _valid_qa_meta()
        explicit_identity["order"] = list(range(12))
        _validator_context(explicit_identity).validate_qa_select_layout()

    def test_rejects_repacking_leakage_and_missing_qa_metadata(self):
        cases = {
            "wrong layout": lambda meta: meta.update(
                physical_layout="visionzip_image_only"),
            "repacked": lambda meta: meta.update(reordered=True),
            "per-layer order": lambda meta: meta.update(
                order_is_per_layer=True),
            "nonidentity order": lambda meta: meta.update(
                order=[1, 0, *range(2, 12)]),
            "question-built layout": lambda meta: meta.update(
                layout_uses_dataset_question=True),
            "future leakage": lambda meta: meta.update(
                future_questions_used=1),
            "calibration": lambda meta: meta.update(calibration_questions=1),
            "no probe heads": lambda meta: meta.update(probe_heads=0),
            "empty probe sidecar": lambda meta: meta.update(
                bytes_probe_sidecar=0),
            "empty separator sidecar": lambda meta: meta.update(
                bytes_separator_sidecar=0),
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                meta = _valid_qa_meta()
                mutate(meta)
                with self.assertRaises(AssertionError):
                    _validator_context(meta).validate_qa_select_layout()

        with self.assertRaisesRegex(AssertionError, "visual hidden"):
            _validator_context(hidden_rows=None).validate_qa_select_layout()
        with self.assertRaises(AssertionError):
            _validator_context(hidden_rows=11).validate_qa_select_layout()


class _CaptureLayer(nn.Module):
    def forward(self, hidden_states):
        return hidden_states + 1


class DecoderVisualHiddenCaptureTests(unittest.TestCase):
    def test_same_forward_slice_decode_ignored_and_hooks_removed(self):
        layer = _CaptureLayer()
        runner = SimpleNamespace(layers=[layer])
        hidden = torch.arange(1 * 8 * 4, dtype=torch.float32).view(1, 8, 4)
        baseline = layer(hidden)

        with DecoderVisualHiddenCapture(
                runner, visual_start=2, visual_tokens=3) as capture:
            instrumented = layer(hidden)
            layer(torch.zeros(1, 1, 4))
            layer(torch.ones(1, 1, 4))

        self.assertTrue(torch.equal(instrumented, baseline))
        self.assertTrue(torch.equal(
            capture.result_cpu(), hidden[0, 2:5].to(torch.float16)))
        stats = capture.stats()
        self.assertEqual(stats["visual_hidden_capture_count"], 1)
        self.assertEqual(stats["layer0_total_calls"], 3)
        self.assertEqual(stats["autoregressive_layer0_calls"], 2)
        self.assertEqual(stats["separate_model_forward_count"], 0)
        self.assertEqual(stats["separate_vision_forward_count"], 0)
        self.assertEqual(len(layer._forward_pre_hooks), 0)
        self.assertFalse(hasattr(
            layer, "_mmimpress_active_visual_hidden_capture"))

    def test_missing_or_repeated_full_prefill_fails_closed(self):
        layer = _CaptureLayer()
        runner = SimpleNamespace(layers=[layer])
        with self.assertRaisesRegex(AssertionError, "exactly one"):
            with DecoderVisualHiddenCapture(
                    runner, visual_start=2, visual_tokens=3):
                layer(torch.zeros(1, 1, 4))
        self.assertEqual(len(layer._forward_pre_hooks), 0)

        with self.assertRaisesRegex(AssertionError, "multiple"):
            with DecoderVisualHiddenCapture(
                    runner, visual_start=2, visual_tokens=3):
                layer(torch.zeros(1, 8, 4))
                layer(torch.zeros(1, 8, 4))
        self.assertEqual(len(layer._forward_pre_hooks), 0)


class _FakeCache:
    def __init__(self, meta):
        shape = (1, meta["num_heads"], meta["prefix_len"],
                 meta["head_dim"])
        self.k = [torch.zeros(shape) for _ in range(meta["num_layers"])]
        self.v = [torch.zeros(shape) for _ in range(meta["num_layers"])]
        self.writes = []
        self.v_start = int(meta["v_token_start"])

    def write(self, layer, kind, rows, values):
        rows = torch.as_tensor(rows, dtype=torch.long)
        self.writes.append((int(layer), str(kind), rows.tolist()))
        target = (self.k if kind == "k" else self.v)[layer]
        target[0, :, rows + self.v_start] = values.to(
            target.dtype).permute(1, 0, 2)


class _FakeQAReader:
    def __init__(self, meta):
        self.meta = meta

    def read_probe(self, layer, counter):
        m = self.meta
        value = torch.zeros(m["v_token_num"], m["probe_heads"],
                            m["head_dim"], dtype=torch.float16)
        counter.record("probe", value.numel() * value.element_size(), 0.001,
                       preads=1, units=m["n_chunks_per_layer"])
        return value

    def read_chunks(self, layer, kind, cids, counter):
        m = self.meta
        rows = []
        for chunk in sorted(set(int(item) for item in cids)):
            start = chunk * m["chunk_size"]
            rows.extend(range(start, min(start + m["chunk_size"],
                                         m["v_token_num"])))
        value = torch.zeros(len(rows), m["num_heads"], m["head_dim"],
                            dtype=torch.float16)
        runs, _ = contiguous_runs(cids)
        counter.record(kind, value.numel() * value.element_size(), 0.002,
                       preads=runs, units=len(set(cids)))
        return torch.tensor(rows, dtype=torch.long), value


class _FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Identity()
        self.k_proj = nn.Identity()


class _FakeDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.Identity()
        self.self_attn = _FakeAttention()

    def forward(self, hidden_states, position_embeddings=None):
        return hidden_states


class _FakeQAContext:
    def __init__(self):
        self.meta = {
            "num_layers": 1,
            "num_heads": 2,
            "head_dim": 2,
            "v_token_start": 1,
            "v_token_num": 12,
            "prefix_len": 13,
            "n_spatial": 10,
            "chunk_size": 3,
            "n_chunks_per_layer": 4,
            "probe_heads": 1,
        }
        self.reader = _FakeQAReader(self.meta)
        self.cache = _FakeCache(self.meta)

    def separator_positions(self, layer):
        return [2, 11]

    def read_sep_kv(self, counter):
        m = self.meta
        value = torch.zeros(2, m["num_layers"], 2, m["num_heads"],
                            m["head_dim"], dtype=torch.float16)
        counter.record("sep", value.numel() * value.element_size(), 0.0005,
                       preads=1, units=0)
        return value


class QASelectLayerSelectorTests(unittest.TestCase):
    def tearDown(self):
        BIAS.clear()

    def test_sparse_chunk_io_accounting_masking_and_no_fallback(self):
        context = _FakeQAContext()
        layer = _FakeDecoderLayer()
        runner = SimpleNamespace(
            layers=[layer], model=SimpleNamespace(device=torch.device("cpu")))
        counter = IOCounter()
        selector = QASelectLayerSelector(
            runner, context, torch.tensor([0]), ratio=0.25, probe=1,
            counter=counter)
        # Separator 2 is deliberately the maximum.  The three selectable
        # winners live in chunks 0, 1, and 3 (two contiguous runs).
        scores = torch.zeros(12)
        scores[2] = 1000
        scores[0], scores[4], scores[9] = 100, 90, 80

        def fake_scores(query, keys, raters, v_start, v_num, **kwargs):
            self.assertEqual(kwargs["head_reduce"], "mean")
            self.assertEqual(kwargs["causal_from"], 13)
            self.assertEqual(tuple(query.shape), (1, 2, 2))
            self.assertEqual(tuple(keys.shape), (1, 15, 2))
            return scores.clone()

        hidden = torch.arange(8, dtype=torch.float32).view(1, 2, 4)
        forbidden = AssertionError("IMPRESS fallback/Jaccard is forbidden")
        with mock.patch(
                "mmimpress.serve.ml.apply_rotary_pos_emb",
                side_effect=lambda query, key, cos, sin: (query, key)), \
             mock.patch(
                 "mmimpress.sparsevlm.rater_visual_scores_from_qk",
                 side_effect=fake_scores), \
             mock.patch("mmimpress.serve.mean_pairwise_jaccard",
                        side_effect=forbidden):
            selector.prepare_structural()
            with selector:
                layer(hidden_states=hidden,
                      position_embeddings=(None, None))

        stats = selector.stats()
        self.assertEqual(stats["selected_token_ids_per_layer"], [[0, 4, 9]])
        self.assertEqual(stats["selected_chunk_ids_per_layer"], [[0, 1, 3]])
        self.assertEqual(stats["contiguous_runs_per_layer"], [2])
        self.assertAlmostEqual(stats["mean_contiguous_run_length"], 1.5)
        self.assertEqual(stats["logical_selected_tokens"], 3)
        self.assertAlmostEqual(stats["logical_selected_token_ratio"], 0.3)
        self.assertAlmostEqual(
            stats["attended_kv_ratio_including_structural"], 5 / 12)
        self.assertEqual(stats["query_score_calls"], 1)
        self.assertGreaterEqual(stats["selected_id_d2h_ms"], 0.0)
        self.assertGreaterEqual(stats["selector_decision_host_wall_ms"],
                                stats["selected_id_d2h_ms"])
        self.assertEqual(stats["fallback_rate"], 0.0)
        self.assertFalse(stats["adaptive_ratio"])
        self.assertFalse(stats["repacking"])
        self.assertEqual(stats["physical_layout"], "raster")

        # Only the three selected logical tokens and two structural tokens are
        # visible even though nine rows arrived with their containing chunks.
        visual_bias = BIAS[0][0, 0, 0, 1:13]
        kept = {index for index, value in enumerate(visual_bias.tolist())
                if value == 0.0}
        self.assertEqual(kept, {0, 2, 4, 9, 11})

        summary = counter.summary()
        self.assertEqual(summary["per_kind"]["probe"]["bytes"], 48)
        self.assertEqual(summary["per_kind"]["sep"]["bytes"], 32)
        self.assertEqual(summary["per_kind"]["k"]["bytes"], 72)
        self.assertEqual(summary["per_kind"]["v"]["bytes"], 72)
        self.assertEqual(summary["bytes"], 224)
        self.assertEqual(summary["preads"], 6)
        self.assertEqual(stats["probe_read_bytes"], 48)
        self.assertEqual(stats["probe_preads"], 1)
        self.assertAlmostEqual(stats["probe_io_ms"], 1.0)
        self.assertEqual(stats["normal_chunk_count_total"], 3)
        self.assertAlmostEqual(stats["touched_chunk_fraction"], 0.75)
        self.assertEqual(stats["layers_completed"], 1)


class _RasterGeometryRunner:
    model_id = "synthetic/llava-next"

    def __init__(self):
        self.visual_span_calls = 0
        self.anyres_layout_calls = 0
        self.layers = [_CaptureLayer()]

    def visual_span(self, input_ids):
        self.visual_span_calls += 1
        return 2, 7

    def anyres_layout(self, image_size, v_num):
        self.anyres_layout_calls += 1
        self.asserted_size = list(image_size)
        self.asserted_v_num = int(v_num)
        # 2x2 base + one 1x2 high-resolution row + its separator.
        return 2, 1, 2, [6]


def _captured_layers(num_layers=2, heads=2, sequence=12, head_dim=3):
    layers = []
    count = heads * sequence * head_dim
    for layer in range(num_layers):
        key = torch.arange(count, dtype=torch.float16).view(
            1, heads, sequence, head_dim) + layer * 1000
        value = key + 5000
        layers.append((key, value))
    return layers


def _completed_hidden_capture(runner, hidden):
    full = torch.zeros(1, 12, hidden.shape[1], dtype=hidden.dtype)
    full[0, 2:9] = hidden
    capture = DecoderVisualHiddenCapture(runner, 2, 7)
    with capture:
        runner.layers[0](full)
    return capture


class RasterPiggybackPersistenceTests(unittest.TestCase):
    def test_atomic_canonical_store_is_identity_and_qa_servable(self):
        runner = _RasterGeometryRunner()
        layers = _captured_layers()
        ids = torch.arange(12).unsqueeze(0)
        hidden = torch.arange(7 * 5, dtype=torch.float32).view(7, 5)
        capture = _completed_hidden_capture(runner, hidden)

        forbidden = AssertionError("raster persistence must not repack")
        with tempfile.TemporaryDirectory() as td, \
             mock.patch("mmimpress.piggyback.anyres_token_scores",
                        side_effect=forbidden), \
             mock.patch("mmimpress.piggyback.visionzip_repack_order",
                        side_effect=forbidden):
            output = Path(td) / "qa-image"
            result = persist_captured_raster_prefix(
                runner, layers, ids, [480, 640], hidden, output,
                image_id="qa-image", chunk_size=2, probe_heads=1,
                hidden_capture_stats=capture)

            self.assertTrue(output.is_dir())
            self.assertFalse((output / "visionzip_layout.pt").exists())
            meta = load_meta(output)
            self.assertEqual(meta["physical_layout"], "raster")
            self.assertEqual(meta["layout_method"], "raster")
            self.assertFalse(meta["reordered"])
            self.assertFalse(meta["order_is_per_layer"])
            self.assertNotIn("order", meta)
            self.assertFalse(meta["layout_uses_dataset_question"])
            self.assertEqual(meta["calibration_questions"], 0)
            self.assertEqual(meta["newline_idx"], [6])
            self.assertEqual(meta["probe_heads"], 1)
            self.assertGreater(meta["bytes_probe_sidecar"], 0)
            self.assertGreater(meta["bytes_separator_sidecar"], 0)
            self.assertEqual(result["timing_ms"]["kv_repack_ms"], 0.0)
            self.assertEqual(result["timing_ms"]["repack_ms"], 0.0)
            self.assertTrue(result["durability"]["atomic_no_clobber"])
            self.assertTrue(
                result["durability"]["parent_fsynced_after_rename"])
            self.assertEqual(set(result["hashes"]["files_sha256"]),
                             {"meta.json", "v_hidden.pt"})
            self.assertTrue(torch.equal(
                torch.load(output / "v_hidden.pt", weights_only=True),
                hidden.to(torch.float16)))

            reader = ChunkReader(output, meta, drop_cache=False)
            try:
                stored_key = reader.read_full(0, "k")
                probe_key = reader.read_probe(0)
            finally:
                reader.close()
            expected = layers[0][0][0, :, 2:9].permute(1, 0, 2).contiguous()
            self.assertTrue(torch.equal(stored_key, expected))
            self.assertTrue(torch.equal(probe_key, expected[:, :1]))

            context = ImageContext(
                output, torch.device("cpu"), drop_cache=False,
                require_v_hidden=True)
            try:
                context.validate_qa_select_layout()
                self.assertTrue(context._qa_select_layout_validated)
                self.assertEqual(context.separator_positions(0), [6])
            finally:
                context.close()

            # Atomic no-clobber publication must preserve the completed store.
            meta_before = (output / "meta.json").read_bytes()
            with self.assertRaises(FileExistsError):
                repeated_capture = _completed_hidden_capture(runner, hidden)
                persist_captured_raster_prefix(
                    runner, layers, ids, [480, 640], hidden, output,
                    image_id="qa-image", chunk_size=2, probe_heads=1,
                    hidden_capture_stats=repeated_capture)
            self.assertEqual((output / "meta.json").read_bytes(), meta_before)

    def test_failed_atomic_stage_is_removed(self):
        runner = _RasterGeometryRunner()
        hidden = torch.zeros(7, 5)
        capture = _completed_hidden_capture(runner, hidden)
        with tempfile.TemporaryDirectory() as td, \
             mock.patch("mmimpress.piggyback.write_image_store",
                        side_effect=RuntimeError("synthetic raster write")):
            output = Path(td) / "failed"
            with self.assertRaisesRegex(RuntimeError, "synthetic raster"):
                persist_captured_raster_prefix(
                    runner, _captured_layers(), torch.arange(12), [480, 640],
                    hidden, output, image_id="failed",
                    chunk_size=2, probe_heads=1,
                    hidden_capture_stats=capture)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(td).glob(".failed.staging-*")), [])


if __name__ == "__main__":
    unittest.main()

"""CPU contracts for the storage-aware QA-Chunk25 selector."""
from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from mmimpress.serve import (
    BIAS,
    QAChunkLayerSelector,
    contiguous_runs,
    mean_valid_spatial_chunk_scores,
    qa_chunk_plan,
    select_qa_chunks_ours_budget,
)
from mmimpress.store import IOCounter


class QAChunkPlanTests(unittest.TestCase):
    def test_mean_uses_only_real_non_separator_rows(self):
        # Huge separator scores must not affect either the numerator or the
        # denominator.  The final chunk has one real row, not chunk_size rows.
        scores = torch.tensor([
            1.0, 1000.0, 3.0, 1000.0,
            5.0, 7.0, 9.0, 11.0,
            1000.0, 13.0,
        ])
        chunk_scores, counts = mean_valid_spatial_chunk_scores(
            scores, separators=[1, 3, 8], v_num=10, chunk_size=4)
        self.assertEqual(counts.tolist(), [2, 4, 1])
        self.assertTrue(torch.equal(
            chunk_scores, torch.tensor([2.0, 8.0, 13.0])))

    def test_structural_only_chunk_is_never_ranked(self):
        scores = torch.tensor([100.0, 100.0, 100.0, 100.0,
                               1.0, 2.0, 3.0, 4.0])
        plan = qa_chunk_plan(
            scores, 0.25, separators=[0, 1, 2, 3],
            v_num=8, chunk_size=4)
        self.assertEqual(plan["valid_spatial_counts"], [0, 4])
        self.assertTrue(math.isinf(plan["chunk_scores"][0]))
        self.assertLess(plan["chunk_scores"][0], 0)
        self.assertEqual(plan["selected_chunks"], [1])

    def test_budget_exactly_matches_ours_rounding_not_token_ceil(self):
        for n_chunks, expected in ((34, 8), (37, 9), (46, 12)):
            with self.subTest(n_chunks=n_chunks):
                scores = torch.arange(n_chunks, dtype=torch.float32)
                selected = select_qa_chunks_ours_budget(
                    scores, torch.ones(n_chunks, dtype=torch.long), 0.25)
                self.assertEqual(selected.numel(), expected)
        # The important half-way case: token-style ceil would select nine.
        self.assertEqual(select_qa_chunks_ours_budget(
            torch.arange(34), torch.ones(34), 0.25).numel(), 8)

    def test_same_scores_are_deterministic_and_query_change_can_move_chunk(self):
        separators = [3, 7, 11, 15]
        first_scores = torch.zeros(16)
        first_scores[0:3] = 10.0
        second_scores = torch.zeros(16)
        second_scores[8:11] = 10.0
        # Separators are deliberately hottest and still irrelevant.
        first_scores[torch.tensor(separators)] = 1000.0
        second_scores[torch.tensor(separators)] = 1000.0
        first = qa_chunk_plan(first_scores, .25, separators, 16, 4)
        repeat = qa_chunk_plan(first_scores, .25, separators, 16, 4)
        changed = qa_chunk_plan(second_scores, .25, separators, 16, 4)
        self.assertEqual(first, repeat)
        self.assertEqual(first["selected_chunks"], [0])
        self.assertEqual(changed["selected_chunks"], [2])
        self.assertNotEqual(first["selected_chunks"],
                            changed["selected_chunks"])

    def test_equal_chunk_scores_have_stable_physical_id_tie_break(self):
        plan = qa_chunk_plan(torch.ones(16), .25, [], 16, 4)
        self.assertEqual(plan["selected_chunks_ranked"], [0])


class _FakeCache:
    def __init__(self, meta):
        shape = (1, meta["num_heads"], meta["prefix_len"], meta["head_dim"])
        self.k = [torch.zeros(shape) for _ in range(meta["num_layers"])]
        self.v = [torch.zeros(shape) for _ in range(meta["num_layers"])]
        self.v_start = int(meta["v_token_start"])

    def write(self, layer, kind, rows, values):
        rows = torch.as_tensor(rows, dtype=torch.long)
        target = (self.k if kind == "k" else self.v)[layer]
        target[0, :, rows + self.v_start] = values.to(
            target.dtype).permute(1, 0, 2)


class _FakeReader:
    def __init__(self, meta):
        self.meta = meta
        self.requested = []

    def read_probe(self, layer, counter):
        m = self.meta
        value = torch.zeros(m["v_token_num"], m["probe_heads"],
                            m["head_dim"], dtype=torch.float16)
        counter.record("probe", value.numel() * value.element_size(), .001,
                       preads=1, units=m["n_chunks_per_layer"])
        return value

    def read_chunks(self, layer, kind, cids, counter):
        m = self.meta
        chunks = sorted({int(item) for item in cids})
        self.requested.append((int(layer), str(kind), chunks))
        rows = []
        for chunk in chunks:
            start = chunk * m["chunk_size"]
            rows.extend(range(start, min(start + m["chunk_size"],
                                         m["v_token_num"])))
        value = torch.zeros(len(rows), m["num_heads"], m["head_dim"],
                            dtype=torch.float16)
        runs, _ = contiguous_runs(chunks)
        counter.record(kind, value.numel() * value.element_size(), .002,
                       preads=runs, units=len(chunks))
        return torch.tensor(rows, dtype=torch.long), value

    def read_full(self, *args, **kwargs):
        raise AssertionError("QA-Chunk must never use full-load fallback")


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


class _FakeContext:
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
        self.reader = _FakeReader(self.meta)
        self.cache = _FakeCache(self.meta)

    def separator_positions(self, layer):
        return [2, 11]

    def read_sep_kv(self, counter):
        m = self.meta
        value = torch.zeros(2, m["num_layers"], 2, m["num_heads"],
                            m["head_dim"], dtype=torch.float16)
        counter.record("sep", value.numel() * value.element_size(), .0005,
                       preads=1, units=0)
        return value


class QAChunkLayerSelectorTests(unittest.TestCase):
    def tearDown(self):
        BIAS.clear()

    def test_exact_chunk_read_whole_chunk_mask_accounting_and_stats(self):
        context = _FakeContext()
        layer = _FakeDecoderLayer()
        runner = SimpleNamespace(
            layers=[layer], model=SimpleNamespace(device=torch.device("cpu")))
        counter = IOCounter()
        selector = QAChunkLayerSelector(
            runner, context, torch.tensor([0]), ratio=.25, probe=1,
            counter=counter)

        token_scores = torch.zeros(12)
        token_scores[2] = 1000.0       # separator, excluded
        token_scores[9:11] = 10.0      # chunk 3 wins
        token_scores[11] = 1000.0      # separator, excluded

        def fake_scores(query, keys, raters, v_start, v_num, **kwargs):
            self.assertEqual(kwargs["head_reduce"], "mean")
            self.assertEqual(kwargs["causal_from"], 13)
            return token_scores.clone()

        hidden = torch.arange(8, dtype=torch.float32).view(1, 2, 4)
        forbidden = AssertionError("fallback/Jaccard is forbidden")
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
        self.assertEqual(stats["selection_mode"], "qa_chunk25")
        self.assertEqual(stats["selection_granularity"], "ssd_chunk")
        self.assertEqual(stats["chunk_score_aggregation"],
                         "mean_valid_spatial_token_importance")
        self.assertEqual(stats["k_chunks"], 1)
        self.assertEqual(stats["selected_chunk_ids_per_layer"], [[3]])
        self.assertEqual(stats["actual_loaded_chunk_ids_per_layer"], [[3]])
        self.assertTrue(stats["actual_loaded_chunks_match_selected"])
        self.assertEqual(context.reader.requested,
                         [(0, "k", [3]), (0, "v", [3])])
        self.assertEqual(stats["query_score_calls"], 1)
        self.assertEqual(stats["chunk_score_calls"], 1)
        self.assertEqual(stats["chunk_scores_generated_per_layer"], [4])
        self.assertEqual(stats["full_load_fallback_count"], 0)
        self.assertEqual(stats["fallback_rate"], 0.0)
        self.assertFalse(stats["adaptive_ratio"])
        self.assertFalse(stats["repacking"])
        self.assertEqual(stats["physical_layout"], "raster")
        self.assertAlmostEqual(stats["normal_selected_chunk_ratio"], .25)
        self.assertAlmostEqual(stats["touched_chunk_fraction"], .25)
        self.assertEqual(stats["contiguous_runs_per_layer"], [1])
        self.assertEqual(stats["max_contiguous_run_length"], 1)

        # Chunk-level selection attends every row of chunk 3; separator 2 is
        # independently kept by the sidecar.  Every other visual row is masked.
        visual_bias = BIAS[0][0, 0, 0, 1:13]
        kept = {index for index, value in enumerate(visual_bias.tolist())
                if value == 0.0}
        self.assertEqual(kept, {2, 9, 10, 11})

        summary = counter.summary()
        self.assertEqual(summary["per_kind"]["probe"]["bytes"], 48)
        self.assertEqual(summary["per_kind"]["sep"]["bytes"], 32)
        self.assertEqual(summary["per_kind"]["k"]["bytes"], 24)
        self.assertEqual(summary["per_kind"]["v"]["bytes"], 24)
        self.assertEqual(summary["bytes"], 128)
        self.assertEqual(summary["preads"], 4)
        self.assertEqual(stats["probe_read_bytes"], 48)
        self.assertEqual(stats["probe_preads"], 1)
        for field in ("query_projection_ms", "query_scoring_ms",
                      "chunk_aggregation_ms", "topk_chunk_ms",
                      "selected_id_d2h_ms", "chunk_planning_ms",
                      "chunk_io_ms", "scatter_ms"):
            self.assertGreaterEqual(stats[field], 0.0, field)


if __name__ == "__main__":
    unittest.main()

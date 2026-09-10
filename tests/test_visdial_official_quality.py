"""CPU-only regression tests for the VisDial candidate-quality runner."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "visdial_official_quality",
    ROOT / "scripts/18_visdial_official_quality.py",
)
QUALITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALITY)


class _FakeCache:
    def __init__(self, length):
        self.length = length
        self.crop_calls = 0

    def get_seq_length(self):
        return self.length

    def crop(self, length):
        self.length = int(length)
        self.crop_calls += 1


class _FakeModel:
    device = torch.device("cpu")

    def __call__(self, input_ids, past_key_values, **_kwargs):
        n = int(input_ids.shape[1])
        logits = torch.stack([
            torch.arange(8, dtype=torch.float32) * float(position + 1)
            for position in range(n)
        ]).unsqueeze(0)
        past_key_values.length += n
        return SimpleNamespace(logits=logits,
                               past_key_values=past_key_values)


class CandidateScoringTests(unittest.TestCase):
    def test_stable_original_order_tie_ranks(self):
        scores = [1.0, 1.0, 0.0] + [-float(i) for i in range(3, 100)]
        ranks = QUALITY._ranks_from_scores(scores)
        self.assertEqual(ranks[:3], [1, 2, 3])
        self.assertEqual(sorted(ranks), list(range(1, 101)))

    def test_candidate_branches_are_cropped_and_likelihood_is_shifted(self):
        cache = _FakeCache(7)
        runner = SimpleNamespace(model=_FakeModel())
        prompt_logits = torch.arange(8, dtype=torch.float32)
        candidate_ids = [[1, 2, 3] for _ in range(100)]
        scores = QUALITY._score_candidates(
            runner, cache, prompt_logits, candidate_ids, max_context=20)

        base = F.log_softmax(prompt_logits, dim=-1)[1]
        branch_logits = torch.stack([
            torch.arange(8, dtype=torch.float32),
            torch.arange(8, dtype=torch.float32) * 2,
        ])
        expected = base + F.log_softmax(branch_logits, dim=-1)[0, 2]
        expected += F.log_softmax(branch_logits, dim=-1)[1, 3]
        self.assertTrue(all(abs(value - float(expected)) < 1e-6
                            for value in scores))
        self.assertEqual(cache.get_seq_length(), 7)
        self.assertEqual(cache.crop_calls, 100)


class OutputAndResumeTests(unittest.TestCase):
    @staticmethod
    def _dialog_turn():
        candidates = [f"answer-{index}" for index in range(100)]
        turn = {
            "turn_id": 1,
            "question": "what is shown?",
            "gold_answer": candidates[0],
            "gt_index": 0,
            "candidate_answers": candidates,
        }
        dialog = {
            "dialog_id": "visdial:1",
            "image_ids": ["visdial:1"],
            "caption": "a test image",
            "turns": [turn],
        }
        return dialog, turn

    @classmethod
    def _row(cls):
        dialog, turn = cls._dialog_turn()
        scores = [-float(i) for i in range(100)]
        return {
            "schema_version": QUALITY.SCHEMA_VERSION,
            "dataset": "visdial_v1.0_val",
            "dialog_id": "visdial:1",
            "dialog_order": 0,
            "image_id": 1,
            "turn_id": 1,
            "method": "ReComp",
            "method_key": "recompute",
            "budget": None,
            "gt_index": 0,
            "gt_rank": 1,
            "ndcg": None,
            "scores": scores,
            "ranks": QUALITY._ranks_from_scores(scores),
            "candidate_token_lengths_including_eos": [2] * 100,
            "prompt_cache_tokens": 10,
            "prompt_sha256": QUALITY._sha_text(
                QUALITY.visdial_prompt(dialog, 1)),
            "selector_input_ids_sha256": "suffix",
            "candidate_answers_sha256": QUALITY._candidate_sha(
                turn["candidate_answers"]),
            "selection": None,
            "score_definition":
                "unnormalized sum log-probability including EOS",
        }

    def test_required_aggregate_outputs_and_readme_are_written(self):
        row = self._row()
        summary = QUALITY._summarise([row], ["recompute"])
        config = {
            "n_dialogs": 1,
            "n_turns": 1,
            "history_policy": "gold_teacher_forced",
            "calibration_policy": None,
            "candidate_count_per_turn": 100,
            "method_keys": ["recompute"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            dialog, turn = self._dialog_turn()
            validation = QUALITY._write_outputs(
                run_dir, [row], ["recompute"], summary,
                {("visdial:1", 1, "recompute")}, {"passed": True}, config,
                {("visdial:1", 1): (dialog, turn, 0)})
            self.assertTrue(validation["passed"])
            expected = {
                "README.md", "summary.json", "summary.csv", "per_turn.csv",
                "per_dialog.csv", "validation.json", "official_ranks",
            }
            self.assertTrue(expected <= {path.name for path in run_dir.iterdir()})
            self.assertIn("quality-only run",
                          (run_dir / "README.md").read_text())

    def test_validation_rejects_ranks_not_derived_from_scores(self):
        row = self._row()
        row["ranks"] = list(range(1, 101))
        # The chosen descending scores imply the same ranks, so reverse the
        # scores while leaving the superficially valid rank permutation.
        row["scores"] = [float(index) for index in range(100)]
        dialog, turn = self._dialog_turn()
        validation = QUALITY._validate_output_rows(
            [row], {("visdial:1", 1, "recompute")}, {"passed": True},
            {("visdial:1", 1): (dialog, turn, 0)})
        self.assertFalse(validation["passed"])
        self.assertEqual(len(validation["rank_score_alignment_failures"]), 1)

    def test_validation_rejects_nan_ndcg_and_wrong_static_budget(self):
        row = self._row()
        row.update({
            "method": "Static+Diverse 25%",
            "method_key": "static_diverse@25",
            "budget": 0.25,
            "ndcg": float("nan"),
            "selection": {
                "n_chunks_selected": 7.0,
                "n_chunks_total": 37,
                "touched_chunk_fraction": 7.0 / 37.0,
                "logical_kv_ratio": 0.20,
                "fallback_rate": 0.0,
            },
        })
        dialog, turn = self._dialog_turn()
        turn["dense_relevance"] = [1.0] + [0.0] * 99
        validation = QUALITY._validate_output_rows(
            [row], {("visdial:1", 1, "static_diverse@25")},
            {"passed": True}, {("visdial:1", 1): (dialog, turn, 0)})
        self.assertFalse(validation["passed"])
        self.assertEqual(len(validation["ndcg_alignment_failures"]), 1)
        self.assertEqual(len(validation["selection_semantic_failures"]), 1)

    def test_every_semantic_config_change_blocks_resume(self):
        existing = {key: "same" for key in QUALITY._RESUME_SEMANTIC_KEYS}
        QUALITY._config_compatible(existing, dict(existing))
        for key in QUALITY._RESUME_SEMANTIC_KEYS:
            changed = dict(existing)
            changed[key] = "different"
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                QUALITY._config_compatible(existing, changed)


if __name__ == "__main__":
    unittest.main()

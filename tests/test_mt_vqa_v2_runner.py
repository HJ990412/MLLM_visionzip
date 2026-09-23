"""CPU-only contracts for the MT-VQA-v2 Generated-History adapter."""
from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/60_eval_mt_vqa_v2_generated_shard.py"
SPEC = importlib.util.spec_from_file_location("mt_vqa_v2_runner_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _answers(matches: int) -> list[str]:
    return ["cat"] * matches + ["dog"] * (10 - matches)


def _dialog() -> dict:
    return {
        "dialog_id": "mtvqav2_000001",
        "global_dialog_ordinal": 0,
        "image_id": "image_a",
        "turns": [
            {"turn_id": turn, "question_id": f"q{turn}",
             "question": f"Question {turn}?", "answers": _answers(2),
             "source_position": turn}
            for turn in (1, 2, 3)
        ],
    }


def _metric_rows() -> list[dict]:
    score = 2.0 / 3.0
    answers = _answers(2)
    rows = []
    for method in MODULE.BASE.METHOD_KEYS:
        rows.append({
            "dialog_id": "mtvqav2_000001", "turn_id": 1,
            "method_key": method, "prediction": "cat",
            "gold": list(answers), "gold_answers": list(answers),
            "gold_answer": list(answers), "correct": score,
            "score": score, "quality_score": score, "vqa_score": score,
            "quality_metric": MODULE.QUALITY_METRIC,
            "quality_metric_implementation": "mmimpress.dataset.vqa_score",
            "official_vqa_evaluator_claimed": False,
            "binary_correct_threshold": MODULE.VQA_BINARY_CORRECT_THRESHOLD,
            "binary_correct": 1, "full_credit_correct": 0,
            "image_input_sha256": "pixel-hash",
            "input_tensors_sha256": "tensor-hash",
        })
    return rows


class RequestAndScoringTests(unittest.TestCase):
    def test_generated_only_request_accounting_is_exact(self):
        counts = MODULE.expected_request_counts(250)
        self.assertEqual(MODULE.PROTOCOLS, ("generated_history",))
        self.assertEqual(MODULE.BASE.PROTOCOLS, ("generated_history",))
        self.assertEqual(counts["turns_per_protocol"], 750)
        self.assertEqual(counts["requests_per_method_per_protocol"], 750)
        self.assertEqual(counts["requests_per_protocol"], 3000)
        self.assertEqual(counts["requests_both_protocols"], 3000)
        self.assertEqual(counts["main_t2_t3_requests_both_protocols"], 2000)
        self.assertEqual(counts["stored_visual_kv_hits_both_protocols"], 1500)

    def test_repository_vqa_soft_score_and_ten_answer_contract(self):
        self.assertEqual(MODULE._score("cat", _answers(0)), 0.0)
        self.assertEqual(MODULE._score("cat", _answers(1)), 1.0 / 3.0)
        self.assertEqual(MODULE._score("cat", _answers(2)), 2.0 / 3.0)
        self.assertEqual(MODULE._score("cat", _answers(3)), 1.0)
        self.assertEqual(MODULE._score("The cat!", _answers(3)), 1.0)
        for bad in ([], ["cat"] * 9, ["cat"] * 11,
                    ["cat"] * 9 + [""], "cat"):
            with self.subTest(answers=bad):
                with self.assertRaisesRegex(ValueError, "exactly ten"):
                    MODULE._score("cat", bad)

    def test_generated_prompt_uses_only_same_method_prior_predictions(self):
        state = {
            1: {"prediction": "raw A1", "logical_request_id": "qa:t1",
                "physical_execution_id": "exec-1"},
            2: {"prediction": "raw A2", "logical_request_id": "qa:t2",
                "physical_execution_id": "exec-2"},
        }
        prompt, history, entries = MODULE.BASE.render_causal_prompt(
            _dialog(), 3, "generated_history", method_key="qa_chunk25",
            generated_predictions=state)
        self.assertIn("A1: raw A1", history)
        self.assertIn("A2: raw A2", history)
        self.assertNotIn("cat", prompt)
        self.assertEqual([entry["source_method_key"] for entry in entries],
                         ["qa_chunk25", "qa_chunk25"])
        with self.assertRaisesRegex(ValueError, "unknown history protocol"):
            MODULE.BASE.render_causal_prompt(
                _dialog(), 2, "gold_history", method_key="qa_chunk25")
        with self.assertRaisesRegex(ValueError, "missing generated"):
            MODULE.BASE.render_causal_prompt(
                _dialog(), 2, "generated_history", method_key="qa_chunk25",
                generated_predictions={})

    def test_vqa_adapter_rechecks_canonical_gold_aliases_and_scores(self):
        rows = _metric_rows()
        group = {"image_id": "image_a", "dialogs": [_dialog()]}
        base_validation = {"passed": True, "strict_scores_recomputed": True}
        with mock.patch.object(
                MODULE, "_ORIGINAL_VALIDATE_ROWS",
                return_value=base_validation):
            report = MODULE._validate_vqa_rows(
                rows, group, "generated_history", {})
            self.assertTrue(report["vqa_consensus_scores_recomputed"])
            self.assertTrue(report["turn1_pixel_and_input_hash_fairness"])

            wrong_gold = copy.deepcopy(rows)
            wrong_gold[0]["gold_answers"][0] = "wrong"
            with self.assertRaisesRegex(ValueError, "canonical ten answers"):
                MODULE._validate_vqa_rows(
                    wrong_gold, group, "generated_history", {})

            wrong_score = copy.deepcopy(rows)
            wrong_score[0]["quality_score"] = 1.0
            with self.assertRaisesRegex(ValueError, "quality_score mismatch"):
                MODULE._validate_vqa_rows(
                    wrong_score, group, "generated_history", {})

    def test_vqa_specific_temp_owner_and_dataset_hooks_are_bound(self):
        helpers = MODULE.BASE.mt_base()
        self.assertEqual(helpers.OWNER_FILE,
                         ".mt_vqa_v2_temp_store_owner.json")
        self.assertEqual(helpers.DATASET,
                         "vqav2_validation_mt3_reconstructed")
        self.assertIs(helpers._load_mt_helpers(), MODULE.mt_vqa_v2)


if __name__ == "__main__":
    unittest.main()

"""CPU-only semantic regression tests for the canonical multi-turn workload.

These tests intentionally stop before model loading.  They protect the data and
measurement contracts that can be checked from the canonical indices and pure
helpers: causal/teacher-forced prompts, image introduction order, VisDial dense
strata, chunk budgets, retrieval metrics, and schema-v2 timing identities.
"""
from __future__ import annotations

import copy
import json
import math
import unittest
from collections import Counter
from pathlib import Path

import torch

from mmimpress.cvpr25 import (budget_chunk_count, choose_chunks,
                              prefix_chunk_ids)
from mmimpress.multiturn import (
    IMAGE_MARKER,
    load_canonical,
    mmdu_prompt,
    mmdu_question_text,
    ndcg,
    retrieval_metrics,
    visdial_prior_history_text,
    visdial_prompt,
)
from mmimpress.multiturn_results import validate_records


ROOT = Path(__file__).resolve().parents[1]
VISDIAL_INDEX = (
    ROOT / "data/visdial_v1.0/subsets/main_seed1234/index.json"
)
VISDIAL_CONFIG = (
    ROOT / "data/visdial_v1.0/subsets/main_seed1234/config.json"
)
VISDIAL_STORE_INDEX = (
    ROOT / "data/visdial_v1.0/subsets/main_seed1234/store_index.json"
)
MMDU_INDEX = ROOT / "data/mmdu/subsets/smoke_seed1234/index.json"
MMDU_CONFIG = ROOT / "data/mmdu/subsets/smoke_seed1234/config.json"


def _read_json(path: Path):
    with open(path) as f:
        return json.load(f)


class VisDialCanonicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dialogs = load_canonical(VISDIAL_INDEX)
        cls.config = _read_json(VISDIAL_CONFIG)
        cls.store_index = _read_json(VISDIAL_STORE_INDEX)

    def test_ten_turns_share_one_image_store(self):
        self.assertEqual(len(self.dialogs), 100)
        self.assertEqual(self.config["n_dialogs"], 100)
        self.assertEqual(self.config["n_images"], 100)
        self.assertEqual(self.config["n_turns"], 1000)

        store_by_image = {entry["image_id"]: entry
                          for entry in self.store_index}
        self.assertEqual(len(store_by_image), len(self.dialogs))

        for dialog in self.dialogs:
            self.assertEqual(len(dialog["image_ids"]), 1)
            self.assertEqual(len(dialog["turns"]), 10)
            image_id = dialog["image_ids"][0]
            self.assertIn(image_id, store_by_image)
            self.assertEqual(dialog["turns"][0]["new_image_ids"], [image_id])
            for turn in dialog["turns"]:
                self.assertEqual(turn["active_image_ids"], [image_id])
            for turn in dialog["turns"][1:]:
                self.assertEqual(turn["new_image_ids"], [])

    def test_history_is_strictly_growing_gold_history(self):
        for dialog in self.dialogs:
            histories = [visdial_prior_history_text(dialog, turn_id)
                         for turn_id in range(1, 11)]
            for previous, current in zip(histories, histories[1:]):
                self.assertGreater(len(current), len(previous))
                self.assertTrue(current.startswith(previous + "\n"))

            for turn_id, history in enumerate(histories, 1):
                expected = [f"Image caption: {dialog['caption']}"]
                for prior in dialog["turns"][:turn_id - 1]:
                    expected.extend([
                        f"Q{prior['turn_id']}: {prior['question']}",
                        f"A{prior['turn_id']}: {prior['gold_answer']}",
                    ])
                self.assertEqual(history, "\n".join(expected))

    def test_all_methods_receive_identical_teacher_forced_prompt(self):
        methods = (
            "ReComp", "FullLoad", "SparseVLM 25%",
            "Static+Diverse 25%", "Static+Diverse 50%",
        )
        self.assertEqual(self.config["history_policy"],
                         "gold_teacher_forced")
        for dialog in self.dialogs:
            for turn_id in range(1, 11):
                prompts = {method: visdial_prompt(dialog, turn_id)
                           for method in methods}
                self.assertEqual(len(set(prompts.values())), 1)

    def test_prompt_and_store_calibration_cannot_see_future_turns(self):
        self.assertEqual(self.config["calibration_policy"],
                         "caption_only_pre_dialog")
        self.assertEqual(self.config["future_turn_calibration_count"], 0)
        store_by_image = {entry["image_id"]: entry
                          for entry in self.store_index}

        for dialog in self.dialogs:
            calibration = dialog["calibration_contexts"]
            self.assertEqual(calibration,
                             [{"kind": "caption", "text": dialog["caption"]}])
            store_entry = store_by_image[dialog["image_ids"][0]]
            self.assertEqual(len(store_entry["questions"]), 1)
            store_question = store_entry["questions"][0]
            self.assertTrue(store_question["calibration_only"])
            self.assertEqual(store_question["question"],
                             f"Image caption: {dialog['caption']}")

            for turn_id in range(1, 11):
                original = visdial_prompt(dialog, turn_id)
                perturbed = copy.deepcopy(dialog)
                # Current gold and every future field are unavailable when the
                # current prompt is formed; changing them must be unobservable.
                perturbed["turns"][turn_id - 1]["gold_answer"] = \
                    "CURRENT_GOLD_MUST_NOT_APPEAR"
                for future in perturbed["turns"][turn_id:]:
                    future["question"] = "FUTURE_QUESTION_MUST_NOT_APPEAR"
                    future["gold_answer"] = "FUTURE_ANSWER_MUST_NOT_APPEAR"
                self.assertEqual(visdial_prompt(perturbed, turn_id), original)

    def test_main_subset_has_balanced_dense_rounds(self):
        dense_rounds = []
        for dialog in self.dialogs:
            annotated = [turn for turn in dialog["turns"]
                         if "dense_relevance" in turn]
            self.assertEqual(len(annotated), 1)
            self.assertEqual(len(annotated[0]["dense_relevance"]), 100)
            self.assertEqual(len(annotated[0]["candidate_answers"]), 100)
            dense_rounds.append(annotated[0]["turn_id"])
            for turn in dialog["turns"]:
                self.assertEqual(len(turn["candidate_answers"]), 100)
                self.assertTrue(0 <= turn["gt_index"] < 100)
                self.assertEqual(
                    turn["candidate_answers"][turn["gt_index"]],
                    turn["gold_answer"],
                )

        expected = {turn_id: 10 for turn_id in range(1, 11)}
        self.assertEqual(dict(Counter(dense_rounds)), expected)
        self.assertEqual(
            {int(k): int(v) for k, v in
             self.config["dense_round_counts"].items()},
            expected,
        )


class MMDUCanonicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dialogs = load_canonical(MMDU_INDEX)
        cls.config = _read_json(MMDU_CONFIG)

    def test_imagehere_mapping_preserves_image_order(self):
        saw_progressive_introduction = False
        for dialog in self.dialogs:
            expected_ids = [image["image_id"] for image in dialog["images"]]
            self.assertEqual(dialog["image_ids"], expected_ids)
            self.assertEqual([image["source_index"] for image in dialog["images"]],
                             list(range(1, len(dialog["images"]) + 1)))

            introduced = []
            previous_active_count = 0
            for turn in dialog["turns"]:
                marker_count = turn["question"].count(IMAGE_MARKER)
                self.assertEqual(marker_count, turn["image_marker_count"])
                self.assertEqual(marker_count, len(turn["new_image_ids"]))
                converted = mmdu_question_text(turn)
                self.assertNotIn(IMAGE_MARKER, converted)
                self.assertEqual(converted.count("<image>"), marker_count)

                introduced.extend(turn["new_image_ids"])
                self.assertEqual(turn["active_image_ids"], introduced)
                self.assertEqual(introduced,
                                 expected_ids[:len(introduced)])
                self.assertGreaterEqual(len(introduced), previous_active_count)
                previous_active_count = len(introduced)
                if turn["turn_id"] > 1 and turn["new_image_ids"]:
                    saw_progressive_introduction = True

                # The full prompt up to this turn contains exactly the images
                # active at this point, in their source introduction order.
                prompt = mmdu_prompt(dialog, turn["turn_id"])
                self.assertEqual(prompt.count("<image>"), len(introduced))

            self.assertEqual(introduced, expected_ids)
        self.assertTrue(saw_progressive_introduction)

    def test_prompt_is_gold_teacher_forced_and_has_no_future_leakage(self):
        self.assertEqual(self.config["history_policy"],
                         "gold_teacher_forced")
        self.assertEqual(self.config["calibration_policy"],
                         "none_correctness_gate_only")

        for dialog in self.dialogs:
            for turn_id in range(1, len(dialog["turns"]) + 1):
                parts = []
                for turn in dialog["turns"][:turn_id]:
                    question = turn["question"].replace(
                        IMAGE_MARKER, "<image>\n").lstrip()
                    piece = f"USER: {question} ASSISTANT:"
                    if turn["turn_id"] < turn_id:
                        piece += f" {turn['gold_answer']}</s>"
                    parts.append(piece)
                expected = " ".join(parts)
                self.assertEqual(mmdu_prompt(dialog, turn_id), expected)

                perturbed = copy.deepcopy(dialog)
                perturbed["turns"][turn_id - 1]["gold_answer"] = \
                    "CURRENT_GOLD_MUST_NOT_APPEAR"
                for future in perturbed["turns"][turn_id:]:
                    future["question"] = \
                        "<ImageHere> FUTURE_QUESTION_MUST_NOT_APPEAR"
                    future["gold_answer"] = \
                        "FUTURE_ANSWER_MUST_NOT_APPEAR"
                self.assertEqual(mmdu_prompt(perturbed, turn_id), expected)

                active = dialog["turns"][turn_id - 1]["active_image_ids"]
                self.assertEqual(active,
                                 dialog["image_ids"][:len(active)])


class ChunkBudgetTests(unittest.TestCase):
    def test_reordered_prefix_is_exact_first_k_without_static_metadata(self):
        self.assertEqual(prefix_chunk_ids(36, 0.25), list(range(9)))
        self.assertEqual(prefix_chunk_ids(36, 0.50), list(range(18)))
        # Production's shared Python-round convention is intentionally kept.
        self.assertEqual(prefix_chunk_ids(34, 0.25), list(range(8)))
        chosen, score = choose_chunks(
            "prefix", 7, None, 0.25, n_chunks=36,
        )
        self.assertEqual(chosen, list(range(9)))
        self.assertIsNone(score)
        with self.assertRaises(ValueError):
            choose_chunks("prefix", 7, None, 0.25)

    def test_static_diverse_obeys_25_and_50_percent_chunk_budgets(self):
        n_chunks = 8
        static = {
            "chunk_score": torch.arange(n_chunks, dtype=torch.float32)
                .view(1, n_chunks),
            "chunk_keys": torch.eye(n_chunks, dtype=torch.float32)
                .view(1, n_chunks, n_chunks),
        }

        self.assertEqual(budget_chunk_count(n_chunks, 0.25), 2)
        self.assertEqual(budget_chunk_count(n_chunks, 0.50), 4)
        # Keep Python round semantics identical to the production selector;
        # ceil(34 * .25) would incorrectly expect 9 instead of 8.
        self.assertEqual(budget_chunk_count(34, 0.25), 8)
        self.assertEqual(budget_chunk_count(37, 0.25), 9)
        for budget, expected_count in ((0.25, 2), (0.50, 4)):
            chosen, scores = choose_chunks(
                "static_diverse", 0, static, budget,
                diverse_frac=0.25, seed=1234, image_id="synthetic",
            )
            repeated, _ = choose_chunks(
                "static_diverse", 0, static, budget,
                diverse_frac=0.25, seed=1234, image_id="synthetic",
            )
            self.assertEqual(chosen, repeated)
            self.assertEqual(len(chosen), expected_count)
            self.assertEqual(len(set(chosen)), expected_count)
            self.assertTrue(all(0 <= chunk < n_chunks for chunk in chosen))
            self.assertEqual(tuple(scores.shape), (n_chunks,))


class RetrievalMetricTests(unittest.TestCase):
    def test_sparse_retrieval_metrics_match_hand_calculation(self):
        got = retrieval_metrics([1, 2, 5, 10, 20])
        self.assertAlmostEqual(got["mrr"], 0.37)
        self.assertAlmostEqual(got["r@1"], 0.2)
        self.assertAlmostEqual(got["r@5"], 0.6)
        self.assertAlmostEqual(got["r@10"], 0.8)
        self.assertAlmostEqual(got["mean_rank"], 7.6)
        self.assertTrue(all(value is None
                            for value in retrieval_metrics([]).values()))

    def test_ndcg_uses_only_nonzero_relevance_cutoff(self):
        relevance = [0.0, 1.0, 0.5, 0.0]
        self.assertAlmostEqual(ndcg([0.1, 0.9, 0.8, 0.0], relevance), 1.0)

        scores = [0.9, 0.8, 0.7, 0.6]
        expected_dcg = 1.0 / math.log2(3)
        expected_ideal = 1.0 + 0.5 / math.log2(3)
        self.assertAlmostEqual(ndcg(scores, relevance),
                               expected_dcg / expected_ideal)
        self.assertEqual(ndcg(scores, [0.0] * 4), 0.0)


class SyntheticResultValidationTests(unittest.TestCase):
    @staticmethod
    def _rows():
        rows = []
        methods = ("FullLoad", "Static+Diverse 25%")
        timings = {
            1: {"FullLoad": (100.0, 20.0),
                "Static+Diverse 25%": (75.0, 20.0)},
            2: {"FullLoad": (110.0, 30.0),
                "Static+Diverse 25%": (80.0, 25.0)},
        }
        for turn_id in (1, 2):
            for method in methods:
                ttft, decode = timings[turn_id][method]
                rows.append({
                    "dialog_id": "synthetic:1",
                    "turn_id": turn_id,
                    "method": method,
                    "ttft_ms": ttft,
                    "decode_ms": decode,
                    "e2e_ms": ttft + decode,
                    "history_tokens": 10 * turn_id,
                    "text_history_sha256": f"history-{turn_id}",
                    "suffix_ids_sha256": f"suffix-{turn_id}",
                    "active_images": 1,
                    "active_image_ids": ["synthetic:image"],
                    "image_kv_build_count_dialog": 1,
                    "static_metadata_build_count_dialog": 1,
                    "ssd_read_bytes": (1024 if method == "FullLoad" else 256),
                    "full_visual_kv_bytes": 1024,
                    "budget": (1.0 if method == "FullLoad" else 0.25),
                    "n_chunks_total": (None if method == "FullLoad" else 8),
                    "n_chunks_selected": (None if method == "FullLoad" else 2),
                })
        return rows

    @staticmethod
    def _config():
        return {
            "methods": ["FullLoad", "Static+Diverse 25%"],
            "calibration_policy": "caption_only_pre_dialog",
            "future_turn_calibration_count": 0,
            "n_turns": 2,
            "expected_request_keys": [["synthetic:1", 1], ["synthetic:1", 2]],
        }

    def test_valid_true_ttft_records_pass_all_timing_invariants(self):
        result = validate_records(self._rows(), self._config())
        self.assertTrue(result["passed"], result["failures"])
        self.assertTrue(result["checks"]["ttft_lt_e2e"])
        self.assertTrue(
            result["checks"]["e2e_approximately_ttft_plus_decode"])
        self.assertTrue(
            result["checks"]["identical_text_history_across_methods"])
        self.assertTrue(
            result["checks"]["identical_suffix_ids_across_methods"])
        self.assertTrue(result["checks"]["history_tokens_monotonic"])
        self.assertTrue(result["checks"]["no_future_turn_calibration"])

    def test_invalid_timing_or_method_history_is_rejected(self):
        bad_residual = self._rows()
        bad_residual[0]["e2e_ms"] += 10.0
        result = validate_records(bad_residual, self._config())
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["checks"]["e2e_approximately_ttft_plus_decode"])

        bad_order = self._rows()
        bad_order[0]["e2e_ms"] = bad_order[0]["ttft_ms"]
        result = validate_records(bad_order, self._config())
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["ttft_lt_e2e"])

        bad_history = self._rows()
        bad_history[1]["text_history_sha256"] = "different-method-history"
        result = validate_records(bad_history, self._config())
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["checks"]["identical_text_history_across_methods"])

    def test_missing_budget_provenance_and_future_leak_are_rejected(self):
        missing = self._rows()
        static_row = next(r for r in missing
                          if r["method"].startswith("Static+Diverse"))
        static_row.pop("n_chunks_selected")
        result = validate_records(missing, self._config())
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["selected_chunk_counts_present"])

        leaked_config = self._config()
        leaked_config["future_turn_calibration_count"] = 1
        result = validate_records(self._rows(), leaked_config)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["no_future_turn_calibration"])

    def test_duplicate_and_unexpected_request_keys_are_rejected(self):
        duplicate = self._rows() + [copy.deepcopy(self._rows()[0])]
        result = validate_records(duplicate, self._config())
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["unique_dialog_turn_method_keys"])

        unexpected = self._rows()
        unexpected[0]["turn_id"] = 99
        result = validate_records(unexpected, self._config())
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["expected_request_keys_exact"])


if __name__ == "__main__":
    unittest.main()

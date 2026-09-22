"""CPU-only schema and analysis contracts for the QA-Select GQA runner."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/49_eval_query_aware_baseline.py"
SPEC = importlib.util.spec_from_file_location("qa_select_gqa_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class QueryAwareRunnerSchemaTests(unittest.TestCase):
    def test_stable_method_ids_labels_and_layouts(self):
        self.assertEqual(RUNNER.METHODS["qa_select25"]["method_id"],
                         "qa_select25")
        self.assertEqual(RUNNER.METHODS["qa_select25"]["display_label"],
                         "QA-Select25")
        self.assertEqual(RUNNER.METHODS["qa_select25"]["paper_label"],
                         "Query-Aware")
        self.assertEqual(RUNNER.METHODS["qa_select25"]["physical_layout"],
                         "raster")
        self.assertEqual(RUNNER.METHODS["ours25"]["method_id"],
                         "imageonly_prefix25")
        self.assertEqual(RUNNER.METHODS["ours25"]["display_label"],
                         "Ours25")

    def test_causal_prompt_provenance_checks_actual_prompt(self):
        fake = SimpleNamespace(prompt=lambda question: f"PROMPT::{question}")
        question = "current only"
        prompt = fake.prompt(question)
        diagnostic = {
            "prompt": prompt,
            "prompt_sha256": RUNNER._sha_bytes(prompt.encode()),
        }
        fields = RUNNER._causal_prompt_fields(
            fake, diagnostic, question, "q-now")
        self.assertEqual(fields["causal_question_ids"], ["q-now"])
        self.assertEqual(fields["future_question_ids_used"], [])
        self.assertEqual(fields["future_questions_in_prompt"], 0)

        changed = dict(diagnostic, prompt="PROMPT::future")
        with self.assertRaisesRegex(AssertionError, "current-question"):
            RUNNER._causal_prompt_fields(fake, changed, question, "q-now")

    def test_output_roots_must_not_overlap(self):
        root = Path("/tmp/query-aware-contract")
        self.assertTrue(RUNNER._paths_overlap(root, root / "nested"))
        self.assertFalse(RUNNER._paths_overlap(
            root / "run", root / "store"))

    def test_reference_equivalence_is_strict_but_tie_tolerant(self):
        def item(compared, equal, gap):
            return {
                "compared": compared, "equal_predictions": equal,
                "accuracy_gap_pp": gap,
            }

        reference = {
            "available": True,
            "methods": {
                "recompute": item(240, 240, 0.0),
                "fullload": item(200, 197, 1.5),
                "ours25": item(200, 199, 0.0),
            },
        }
        self.assertTrue(RUNNER._reference_consistency_passes(reference))
        reference["methods"]["fullload"] = item(200, 195, 0.0)
        self.assertFalse(RUNNER._reference_consistency_passes(reference))
        reference["methods"]["fullload"] = item(200, 197, 2.5)
        self.assertFalse(RUNNER._reference_consistency_passes(reference))
        reference["methods"]["fullload"] = item(200, 197, 1.5)
        reference["methods"]["recompute"] = item(240, 239, 0.0)
        self.assertFalse(RUNNER._reference_consistency_passes(reference))


class QueryAwareSelectionAnalysisTests(unittest.TestCase):
    @staticmethod
    def _row(turn, question, tokens, chunks):
        return {
            "method_key": "qa_select25",
            "image_id": "image-1",
            "turn_id": turn,
            "question_id": question,
            "current_question_sha256": question * 8,
            "selected_token_ids_per_layer": tokens,
            "selected_chunk_ids_per_layer": chunks,
        }

    def test_pairwise_per_layer_and_consecutive_jaccard(self):
        rows = [
            self._row(2, "q2", [[0, 1], [2, 3]], [[0], [1]]),
            self._row(3, "q3", [[0, 2], [2, 4]], [[0, 1], [1, 2]]),
            self._row(4, "q4", [[5, 6], [7, 8]], [[2], [3]]),
        ]
        result = RUNNER._selection_analysis(rows)
        self.assertEqual(result["n_query_requests"], 3)
        self.assertEqual(result["n_pairs"], 3)
        self.assertEqual(result["n_consecutive_pairs"], 2)
        pair = result["pairs"][0]
        self.assertEqual(pair["token_jaccard_per_layer"], [1 / 3, 1 / 3])
        self.assertAlmostEqual(pair["token_jaccard"], 1 / 3)
        self.assertNotEqual(
            result["requests"][0]["selected_tokens_sha256"],
            result["requests"][1]["selected_tokens_sha256"])

    def test_method_summaries_do_not_contain_comparison(self):
        rows = []
        for method in RUNNER.METHOD_KEYS:
            rows.append({
                "method_key": method, "turn_id": 2, "correct": 1.0,
                "end_to_end_ttft_ms": 10.0, "actual_ssd_mb": 0.0,
                "actual_ssd_ratio_vs_fullload": 0.0, "ssd_preads": 0,
                "ssd_read_ms": 0.0, "probe_read_bytes": 0,
            })
        summaries = RUNNER._summaries(rows, 100.0)
        self.assertEqual(set(summaries), set(RUNNER.METHOD_KEYS))
        comparison = RUNNER._comparison(summaries)
        self.assertEqual(comparison["qa_minus_ours_accuracy_all_turns_pp"],
                         0.0)


if __name__ == "__main__":
    unittest.main()

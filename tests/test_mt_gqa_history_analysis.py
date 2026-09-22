"""CPU-only contracts for the Gold/Generated MT-GQA analyzer."""
from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/55_analyze_mt_gqa_history.py"
SPEC = importlib.util.spec_from_file_location("mt_gqa_history_analysis_test", SCRIPT)
ANALYZER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYZER)


def _dialog(number: int) -> dict:
    return {
        "dialog_id": f"d{number}",
        "image_id": f"image{number}",
        "image_path": f"images/image{number}.jpg",
        "turns": [{
            "turn_id": turn,
            "question_id": f"q{number}{turn}",
            "question": f"question {number} turn {turn}?",
            "answers": [f"gold{number}{turn}"],
        } for turn in (1, 2, 3)],
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class SyntheticRun:
    def __init__(self, root: Path):
        self.root = root
        self.gold = root / "gold_history"
        self.generated = root / "generated_history"
        self.results = root / "results"
        self.index = root / "dialogues.json"
        self.protection = root / "protected_artifacts_validation.json"
        self.dialogs = [_dialog(1), _dialog(2)]
        _write_json(self.index, {
            "schema_version": "mt-gqa-reconstructed-v1",
            "seed": 1234,
            "benchmark_type": "MT-GQA-reconstructed",
            "dialogues": self.dialogs,
        })
        _write_json(self.protection, {
            "passed": True,
            "before_manifest_sha256": "a" * 64,
            "after_manifest_sha256": "a" * 64,
            "missing_paths": [], "added_paths": [], "changed_paths": [],
        })
        for protocol, directory in (("gold_history", self.gold),
                                    ("generated_history", self.generated)):
            _write_json(directory / "config.json", {
                "protocol": protocol, "n_dialogs": 2, "n_images": 2,
                "protocol_physical_execution_independent": True,
                "model": "llava-hf/llava-v1.6-vicuna-7b-hf",
                "model_revision": ANALYZER.FROZEN_MODEL_REVISION,
                "load_4bit": True,
                "quantization": "4-bit NF4 double-quant",
                "compute_dtype": "bfloat16",
                "attention": "eager", "decoding": "greedy",
                "max_new_tokens": 16, "chunk_size": 64,
                "probe_heads": 3,
                "qa_chunk_configuration": ANALYZER.FROZEN_QA_CONFIGURATION,
                "ours_configuration": ANALYZER.FROZEN_OURS_CONFIGURATION,
                "turn1_policy": (
                    "all four methods execute independent normal pixel requests; "
                    "FullLoad captures/persists the shared canonical raster store "
                    "from its own first-dialog T1"),
                "qa_raster_source_policy": (
                    "QA canonical raster Visual-KV captured by FullLoad's own T1"),
                "later_turn_policy": (
                    "ReComp pixels; FullLoad/QA use raster SSD; Ours uses independent "
                    "image-only repacked SSD"),
            })
            for dialog in self.dialogs:
                artifact = self._artifact(protocol, dialog)
                artifact["artifact_content_sha256"] = ANALYZER._stable_json(artifact)
                _write_json(directory / "images" / f"{dialog['image_id']}.json",
                            artifact)

    @staticmethod
    def _prediction(protocol: str, method: str, dialog: dict, turn: int) -> str:
        # T1 must match byte-for-byte across protocols. Later responses may
        # differ by protocol and method, as the real experiment permits.
        if turn == 1:
            return dialog["turns"][0]["answers"][0]
        if protocol == "gold_history":
            return (dialog["turns"][turn - 1]["answers"][0]
                    if method != "ours25" else "wrong")
        return (dialog["turns"][turn - 1]["answers"][0]
                if method in {"recompute", "qa_chunk25"} else "wrong")

    def _artifact(self, protocol: str, dialog: dict) -> dict:
        rows = []
        generated: dict[str, list[dict]] = {
            method: [] for method in ANALYZER.METHOD_KEYS}
        for turn in (1, 2, 3):
            for method_index, method in enumerate(ANALYZER.METHOD_KEYS):
                if protocol == "gold_history":
                    prior_answers = [dialog["turns"][i]["answers"][0]
                                     for i in range(turn - 1)]
                    source_ids = [f"gold:{dialog['turns'][i]['question_id']}"
                                  for i in range(turn - 1)]
                else:
                    prior_answers = [row["prediction"]
                                     for row in generated[method]]
                    source_ids = [row["logical_request_id"]
                                  for row in generated[method]]
                history = ANALYZER.render_history(dialog, turn, prior_answers)
                prompt = ANALYZER.render_prompt(dialog, turn, prior_answers)
                prediction = self._prediction(protocol, method, dialog, turn)
                logical = f"{protocol}:{dialog['dialog_id']}:t{turn}:{method}"
                selected = []
                if turn > 1 and method == "qa_chunk25":
                    selected = ([[0, 1], [0, 1]] if turn == 2
                                else [[1, 2], [1, 2]])
                elif turn > 1 and method == "ours25":
                    selected = [[0, 1], [0, 1]]
                ssd = 0 if method == "recompute" or turn == 1 else (
                    1000 if method == "fullload" else 300)
                row = {
                    "status": "ok", "protocol": protocol,
                    "method_key": method,
                    "dialog_id": dialog["dialog_id"],
                    "image_id": dialog["image_id"], "turn_id": turn,
                    "question_id": dialog["turns"][turn - 1]["question_id"],
                    "question": dialog["turns"][turn - 1]["question"],
                    "gold": dialog["turns"][turn - 1]["answers"],
                    "prediction": prediction,
                    "strict_correct": ANALYZER.strict_exact_score(
                        prediction, dialog["turns"][turn - 1]["answers"]),
                    "logical_request_id": logical,
                    "physical_execution_id": f"physical:{logical}",
                    "execution_id": f"physical:{logical}",
                    "history_source": ("gold" if protocol == "gold_history"
                                       else "generated"),
                    "history_text": history,
                    "history_answers": prior_answers,
                    "history_source_request_ids": source_ids,
                    "prompt": prompt,
                    "prompt_sha256": ANALYZER._sha_bytes(prompt.encode()),
                    "text_history_sha256": ANALYZER._sha_bytes(history.encode()),
                    "first_token_id": 100 + turn,
                    "input_token_count": 20 + turn + len(prior_answers) * 3,
                    "generated_token_count": 2 + turn,
                    "answer_token_count": 1,
                    "end_to_end_ttft_ms": 100 + 10 * method_index + turn,
                    "ssd_read_bytes": ssd,
                    "pread_count": 0 if ssd == 0 else 5 + method_index,
                    "ssd_read_ms": 0 if ssd == 0 else 2 + method_index,
                    "probe_read_bytes": 20 if method == "qa_chunk25" and turn > 1 else 0,
                    "normal_kv_read_bytes": 200 if method in {"qa_chunk25", "ours25"}
                    and turn > 1 else 0,
                    "separator_read_bytes": 10 if ssd else 0,
                    "contiguous_runs_per_layer_mean": 2 if method == "qa_chunk25" else 1,
                    "selector_ms": 8 if method == "qa_chunk25" and turn > 1 else 0,
                    "n_raters": 4 if method == "qa_chunk25" and turn > 1 else 0,
                    "rater_selection_ms": 1 if method == "qa_chunk25" and turn > 1 else 0,
                    "query_projection_ms": 1 if method == "qa_chunk25" and turn > 1 else 0,
                    "probe_io_ms": 1 if method == "qa_chunk25" and turn > 1 else 0,
                    "query_scoring_ms": 1 if method == "qa_chunk25" and turn > 1 else 0,
                    "chunk_aggregation_ms": 1 if method == "qa_chunk25" and turn > 1 else 0,
                    "topk_chunk_ms": 1 if method == "qa_chunk25" and turn > 1 else 0,
                    "selected_chunk_ids_per_layer": selected,
                    "retention_ratio": (0.25 if method in {"qa_chunk25", "ours25"}
                                        else 1.0 if method == "fullload" else None),
                    "n_chunks_total": 8 if method in {"qa_chunk25", "ours25"}
                    and turn > 1 else 0,
                    "physical_layout": (
                        "raster" if method in {"fullload", "qa_chunk25"}
                        else "visionzip_image_only" if method == "ours25" else "none"),
                    "chunk_score": ("mean_valid_spatial_token_importance"
                                    if method == "qa_chunk25" else None),
                    "head_reduce": "mean" if method == "qa_chunk25" else None,
                    "rater_algorithm_id": (
                        ANALYZER.QA_RATER_ALGORITHM_ID
                        if method == "qa_chunk25" else None),
                    "probe_heads_used": 3 if method == "qa_chunk25" and turn > 1 else 0,
                    "fallback_rate": 0.0,
                    "adaptive_ratio": False,
                    "query_score_calls": 2 if method == "qa_chunk25" and turn > 1 else 0,
                    "static_score_calls": 0,
                }
                rows.append(row)
                if protocol == "generated_history":
                    generated[method].append(row)
        return {
            "image_id": dialog["image_id"], "protocol": protocol,
            "store_manifests": {
                "raster": {"permutation_sha256": "identity"},
                "image_only": {
                    "permutation_sha256": f"perm-{dialog['image_id']}"},
            },
            "rows": rows,
        }

    def rehash(self, protocol: str, image: str) -> None:
        path = (self.gold if protocol == "gold_history" else self.generated) \
            / "images" / f"{image}.json"
        value = json.loads(path.read_text())
        value.pop("artifact_content_sha256", None)
        value["artifact_content_sha256"] = ANALYZER._stable_json(value)
        _write_json(path, value)


class MetricTests(unittest.TestCase):
    def test_strict_normalization_does_not_accept_gold_prefix(self):
        self.assertEqual(ANALYZER.strict_exact_score("The Cat!", ["cat"]), 1.0)
        self.assertEqual(ANALYZER.strict_exact_score("cat extra", ["cat"]), 0.0)

    def test_exact_mcnemar_uses_binomial_tail(self):
        self.assertEqual(ANALYZER.exact_mcnemar_pvalue(0, 0), 1.0)
        self.assertAlmostEqual(ANALYZER.exact_mcnemar_pvalue(3, 0), 0.25)
        self.assertAlmostEqual(ANALYZER.exact_mcnemar_pvalue(1, 1), 1.0)

    def test_gold_renderer_matches_authoritative_helper(self):
        dialog = _dialog(1)
        for turn in (1, 2, 3):
            prior = [dialog["turns"][i]["answers"][0] for i in range(turn - 1)]
            self.assertEqual(ANALYZER.render_prompt(dialog, turn, prior),
                             ANALYZER.mt_gqa_prompt(dialog, turn))


class EndToEndTests(unittest.TestCase):
    def test_analyzer_publishes_complete_required_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticRun(Path(temporary))
            result = ANALYZER.analyze(
                fixture.gold, fixture.generated, fixture.results,
                fixture.index, fixture.protection,
                expected_dialogs=2, expected_images=2)
            self.assertTrue(result["validation"]["passed"])
            self.assertEqual(result["validation"]["observed_logical_rows"], 48)
            for protocol in ANALYZER.PROTOCOLS:
                for name in (
                    "raw.jsonl", "raw.jsonl.gz", "summary.csv",
                    "quality_by_turn.csv", "ttft_by_turn.csv",
                    "token_lengths.csv", "io_breakdown.csv",
                    "selection_analysis.json", "validation.json", "config.json",
                ):
                    self.assertTrue((fixture.results / protocol / name).is_file(), name)
            for name in (
                "history_comparison.csv", "error_propagation.csv",
                "paired_quality.json", "ANALYSIS.md", "README.md",
                "manual_history_samples.json",
            ):
                self.assertTrue((fixture.results / "comparison" / name).is_file(), name)
            validation = json.loads((fixture.results / "validation.json").read_text())
            self.assertEqual(len(validation["manual_history_samples"]), 16)
            self.assertTrue(validation["checks"]["prior_artifacts_unchanged"])
            self.assertTrue(validation["checks"]["frozen_source_configuration"])
            completion = json.loads((fixture.results / "COMPLETED").read_text())
            self.assertTrue(completion["passed"])
            self.assertIn("gold_history/raw.jsonl", completion["output_sha256"])
            self.assertNotIn("COMPLETED", completion["output_sha256"])
            analysis = (fixture.results / "comparison/ANALYSIS.md").read_text()
            first = analysis.index("## Gold-History")
            second = analysis.index("## Generated-History")
            third = analysis.index("## Gold vs Generated")
            self.assertLess(first, second)
            self.assertLess(second, third)
            self.assertIn("## Paired QA-Chunk25 vs Ours25", analysis)
            self.assertIn("## Generated-history error propagation", analysis)
            self.assertIn("## Detailed cache-hit I/O and selector costs", analysis)
            self.assertTrue(analysis.rstrip().endswith(
                "MT-GQA GOLD + GENERATED HISTORY 4-ARM EVALUATION VALIDATED: YES"))
            with (fixture.results / "comparison/error_propagation.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            patterns = [row for row in rows
                        if row["metric"] == "t1_t2_joint_correctness_count"]
            self.assertEqual(len(patterns), 16)

    def test_generated_history_contamination_fails_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticRun(Path(temporary))
            path = fixture.generated / "images/image1.json"
            artifact = json.loads(path.read_text())
            row = next(row for row in artifact["rows"]
                       if row["method_key"] == "qa_chunk25" and row["turn_id"] == 2)
            row["history_answers"] = ["other method answer"]
            artifact.pop("artifact_content_sha256")
            artifact["artifact_content_sha256"] = ANALYZER._stable_json(artifact)
            _write_json(path, artifact)
            with self.assertRaisesRegex(ANALYZER.AnalysisError, "history answers"):
                ANALYZER.analyze(
                    fixture.gold, fixture.generated, fixture.results,
                    fixture.index, fixture.protection,
                    expected_dialogs=2, expected_images=2)
            self.assertFalse(fixture.results.exists())

    def test_duplicate_matrix_cell_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticRun(Path(temporary))
            path = fixture.gold / "images/image1.json"
            artifact = json.loads(path.read_text())
            artifact["rows"].append(dict(artifact["rows"][0]))
            artifact.pop("artifact_content_sha256")
            artifact["artifact_content_sha256"] = ANALYZER._stable_json(artifact)
            _write_json(path, artifact)
            with self.assertRaisesRegex(ANALYZER.AnalysisError, "duplicate logical matrix"):
                ANALYZER.analyze(
                    fixture.gold, fixture.generated, fixture.results,
                    fixture.index, fixture.protection,
                    expected_dialogs=2, expected_images=2)


if __name__ == "__main__":
    unittest.main()

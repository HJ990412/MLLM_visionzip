"""Focused CPU contracts for the MT-VQA-v2 Generated-History analyzer."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/61_analyze_mt_vqa_v2_generated.py"
SPEC = importlib.util.spec_from_file_location("mt_vqa_v2_analysis_test", SCRIPT)
ANALYZER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYZER)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _answers(label: str, matches: int = 3) -> list[str]:
    return [label] * matches + [f"other-{index}" for index in range(10 - matches)]


def _dialog(number: int) -> dict:
    return {
        "dialog_id": f"d{number}",
        "global_dialog_ordinal": number - 1,
        "image_id": f"image{number}",
        "image_path": f"images/image{number}.jpg",
        "turns": [{
            "turn_id": turn,
            "question_id": f"q{number}{turn}",
            "question": f"question {number} turn {turn}?",
            "answers": _answers(f"answer{number}{turn}", 3),
            "source_position": turn,
        } for turn in (1, 2, 3)],
    }


class SyntheticRun:
    def __init__(self, root: Path):
        self.root = root
        self.run = root / "run"
        self.results = root / "results"
        self.index = root / "dialogues.json"
        self.protection = root / "protection.json"
        self.dialogs = [_dialog(1), _dialog(2)]
        _write_json(self.index, {
            "schema_version": "mt-vqa-v2-reconstructed-v1",
            "benchmark_type": "MT-VQA-v2-reconstructed",
            "seed": 1234, "dialogues": self.dialogs,
        })
        _write_json(self.protection, {
            "passed": True, "before_manifest_sha256": "a" * 64,
            "after_manifest_sha256": "a" * 64,
            "missing_paths": [], "added_paths": [], "changed_paths": [],
        })
        self.config = self._config()
        index_sha = ANALYZER._sha_file(self.index)
        workload_sha = ANALYZER.mt_vqa_v2.workload_sha256(self.dialogs)
        self.config.update({
            "n_dialogs": 2, "n_turns": 6, "n_images": 2,
            "n_requests": 24,
            "planned_logical_requests_this_protocol": 24,
            "planned_physical_executions_this_protocol": 24,
            "planned_logical_requests_generated_only": 24,
            "planned_physical_executions_generated_only": 24,
            "dialogues_file_sha256": index_sha,
            "source_full_workload_sha256": workload_sha,
            "selected_workload_sha256": workload_sha,
        })
        _write_json(self.run / "config.json", self.config)
        for dialog in self.dialogs:
            artifact = self._artifact(dialog)
            artifact["artifact_content_sha256"] = ANALYZER._stable_json(artifact)
            _write_json(self.run / "images" / f"{dialog['image_id']}.json", artifact)

    @staticmethod
    def _config() -> dict:
        return {
            "schema_version": ANALYZER.SHARD_SCHEMA_VERSION,
            "experiment_id": "synthetic-experiment",
            "seed": 1234,
            "method_keys": list(ANALYZER.METHOD_KEYS),
            "protocol": ANALYZER.PROTOCOL,
            "protocols": [ANALYZER.PROTOCOL],
            "protocol_scope": "generated_history_only",
            "dataset": "vqav2_validation_mt3_reconstructed",
            "benchmark_type": "MT-VQA-v2-reconstructed",
            "history_policy": "method_local_generated",
            "quality_metric": ANALYZER.QUALITY_METRIC,
            "quality_metric_implementation": "mmimpress.dataset.vqa_score",
            "official_vqa_evaluator_claimed": False,
            "binary_correct_threshold": 0.5,
            "model": "llava-hf/llava-v1.6-vicuna-7b-hf",
            "model_revision": ANALYZER.FROZEN_MODEL_REVISION,
            "load_4bit": True, "quantization": "4-bit NF4 double-quant",
            "compute_dtype": "bfloat16", "attention": "eager",
            "decoding": "greedy", "max_new_tokens": 16,
            "chunk_size": 64, "probe_heads": 3,
            "qa_chunk_configuration": ANALYZER.FROZEN_QA_CONFIGURATION,
            "ours_configuration": ANALYZER.FROZEN_OURS_CONFIGURATION,
            "dataset_construction": {
                "source_index_sha256": ANALYZER.mt_vqa_v2.SOURCE_INDEX_SHA256,
                "source_slice": "questions[1:5]",
                "dialogue_membership": "questions[1:4]",
                "dialogues_per_image": 1,
                "dialogue_overlap": False,
                "question_reuse": False,
            },
            "turn1_policy": (
                "all methods pixels; FullLoad captures/persists canonical raster"),
            "qa_raster_source_policy": (
                "canonical raster captured by FullLoad's own T1"),
            "later_turn_policy": (
                "ReComp pixels; FullLoad/QA use raster SSD; Ours uses independent "
                "image-only repacked SSD"),
            "main_ttft_field": "end_to_end_ttft_ms",
        }

    @staticmethod
    def _manifests(image: str) -> dict:
        common = {
            "num_layers": 2, "v_token_num": 4, "chunk_size": 1,
            "n_chunks_per_layer": 4, "visual_kv_bytes": 400,
        }
        return {
            "raster": {**common, "physical_layout": "raster",
                       "probe_sidecar_bytes": 20,
                       "separator_sidecar_bytes": 10,
                       "permutation_sha256": "identity"},
            "image_only": {**common, "physical_layout": "visionzip_image_only",
                           "probe_sidecar_bytes": 0,
                           "separator_sidecar_bytes": 10,
                           "permutation_sha256": f"perm-{image}"},
        }

    @staticmethod
    def _prediction(dialog: dict, method: str, turn: int) -> str:
        # All arms have identical T1 semantics. QA earns 2/3 on later turns,
        # Ours 1/3, ReComp full credit, and FullLoad zero.
        label = f"answer{dialog['dialog_id'][1:]}{turn}"
        if turn == 1 or method == "recompute":
            return label
        if method == "qa_chunk25":
            return "The " + label + "!"
        if method == "ours25":
            return label if dialog["dialog_id"] == "d1" else "wrong"
        return "wrong"

    def _artifact(self, dialog: dict) -> dict:
        manifests = self._manifests(dialog["image_id"])
        rows: list[dict] = []
        prior: dict[str, list[dict]] = {method: [] for method in ANALYZER.METHOD_KEYS}
        for turn in (1, 2, 3):
            for method_index, method in enumerate(ANALYZER.METHOD_KEYS):
                earlier = prior[method]
                history_answers = [row["prediction"] for row in earlier]
                history = ANALYZER.render_history(dialog, turn, history_answers)
                prompt = ANALYZER.render_prompt(dialog, turn, history_answers)
                logical = f"generated_history:{dialog['dialog_id']}:t{turn}:{method}"
                prediction = self._prediction(dialog, method, turn)
                answers = dialog["turns"][turn - 1]["answers"]
                score = ANALYZER.soft_vqa_score(prediction, answers)
                cache = turn > 1 and method != "recompute"
                selected: list[list[int]] = []
                if turn > 1 and method == "qa_chunk25":
                    selected = [[0], [0]] if turn == 2 else [[1], [1]]
                elif turn > 1 and method == "ours25":
                    selected = [[0], [0]]
                normal = probe = separator = normal_preads = probe_preads = sep_preads = 0
                if cache and method == "fullload":
                    normal, normal_preads = 400, 4
                elif cache and method == "qa_chunk25":
                    normal, probe, separator = 100, 20, 10
                    normal_preads, probe_preads, sep_preads = 4, 2, 1
                elif cache and method == "ours25":
                    normal, separator = 100, 10
                    normal_preads, sep_preads = 4, 1
                row = {
                    "schema_version": ANALYZER.SHARD_SCHEMA_VERSION,
                    "dialogues_file_sha256": self.config["dialogues_file_sha256"],
                    "source_full_workload_sha256": self.config["source_full_workload_sha256"],
                    "selected_workload_sha256": self.config["selected_workload_sha256"],
                    "model_revision": ANALYZER.FROZEN_MODEL_REVISION,
                    "status": "ok", "protocol": ANALYZER.PROTOCOL,
                    "history_source": "generated",
                    "history_policy": "method_local_generated",
                    "method_key": method, "method": ANALYZER.METHOD_LABELS[method],
                    "dialog_id": dialog["dialog_id"], "image_id": dialog["image_id"],
                    "turn_id": turn,
                    "question_id": dialog["turns"][turn - 1]["question_id"],
                    "question": dialog["turns"][turn - 1]["question"],
                    "gold": answers, "gold_answers": answers,
                    "prediction": prediction, "vqa_score": score,
                    "quality_score": score, "score": score, "correct": score,
                    "quality_metric": ANALYZER.QUALITY_METRIC,
                    "quality_metric_implementation": "mmimpress.dataset.vqa_score",
                    "official_vqa_evaluator_claimed": False,
                    "binary_correct_threshold": 0.5,
                    "binary_correct": int(score >= 0.5),
                    "full_credit_correct": int(score == 1.0),
                    "logical_request_id": logical,
                    "physical_execution_id": f"physical:{logical}",
                    "execution_id": f"physical:{logical}",
                    "dialogue_session_id": f"session-{dialog['dialog_id']}",
                    "context_instance_id": f"context-{dialog['dialog_id']}",
                    "history_text": history,
                    "history_text_sha256": ANALYZER._sha_bytes(history.encode()),
                    "history_answers": history_answers,
                    "history_source_request_ids": [row["logical_request_id"] for row in earlier],
                    "history_source_physical_execution_ids": [
                        row["physical_execution_id"] for row in earlier],
                    "history_turn_ids": list(range(1, turn)),
                    "history_entries": [{
                        "turn_id": index + 1,
                        "question_id": dialog["turns"][index]["question_id"],
                        "question": dialog["turns"][index]["question"],
                        "answer": earlier[index]["prediction"],
                        "answer_source": "generated",
                        "source_method_key": method,
                        "source_logical_request_id": earlier[index]["logical_request_id"],
                        "source_physical_execution_id": earlier[index]["physical_execution_id"],
                    } for index in range(turn - 1)],
                    "future_leakage": 0,
                    "cache_hit": cache,
                    "request_path": ("stored_visual_kv" if cache
                                     else "normal_multimodal_pixel"),
                    "execution_mode": ("stored_visual_kv" if cache
                                       else "normal_multimodal_pixel"),
                    "vision_forward_count": 0 if cache else 1,
                    "page_cache_conditioning_excluded_from_ttft": True,
                    "prompt": prompt,
                    "prompt_sha256": ANALYZER._sha_bytes(prompt.encode()),
                    "input_tensors_sha256": f"tensors-{dialog['image_id']}-t{turn}",
                    "image_input_sha256": f"pixels-{dialog['image_id']}",
                    "input_ids_sha256": f"ids-{dialog['image_id']}-t{turn}",
                    "first_token_id": 100 + turn,
                    "input_token_count": 20 + turn + len(history_answers) * 3,
                    "generated_token_count": 2 + turn,
                    "end_to_end_ttft_ms": 100 + method_index * 10 + turn,
                    "request_e2e_ms": 130 + method_index * 10 + turn,
                    "request_started_at_s": 1.0,
                    "core_started_at_s": 1.01,
                    "first_token_at_s": 1.0 + (100 + method_index * 10 + turn) / 1000,
                    "model_finished_at_s": 1.0 + (120 + method_index * 10 + turn) / 1000,
                    "request_returned_at_s": 1.0 + (130 + method_index * 10 + turn) / 1000,
                    "ttft_identity_error_ms": 0.0,
                    "ssd_read_bytes": normal + probe + separator,
                    "normal_kv_read_bytes": normal,
                    "probe_read_bytes": probe,
                    "separator_read_bytes": separator,
                    "ssd_preads": normal_preads + probe_preads + sep_preads,
                    "pread_count": normal_preads + probe_preads + sep_preads,
                    "normal_kv_preads": normal_preads,
                    "probe_preads": probe_preads,
                    "separator_preads": sep_preads,
                    "ssd_read_ms": 2.0 if cache else 0.0,
                    "selected_chunk_ids_per_layer": selected,
                    "retention_ratio": (0.25 if method in {"qa_chunk25", "ours25"}
                                        else 1.0 if method == "fullload" else None),
                    "n_chunks_total": 4 if cache and method in {
                        "qa_chunk25", "ours25"} else 0,
                    "physical_layout": (
                        "raster" if method in {"fullload", "qa_chunk25"}
                        else "visionzip_image_only" if method == "ours25" else "none"),
                    "head_reduce": "mean" if method == "qa_chunk25" else None,
                    "chunk_score": ("mean_valid_spatial_token_importance"
                                    if method == "qa_chunk25" else None),
                    "rater_algorithm_id": (ANALYZER.QA_RATER_ALGORITHM_ID
                                           if method == "qa_chunk25" else None),
                    "rater_scope": "entire_available_causal_suffix",
                    "probe_heads_used": 3 if cache and method == "qa_chunk25" else 0,
                    "fallback_rate": 0.0, "adaptive_ratio": False,
                    "query_score_calls": 2 if cache and method == "qa_chunk25" else 0,
                    "chunk_score_calls": 2 if cache and method == "qa_chunk25" else 0,
                    "rater_count": 3 if cache and method == "qa_chunk25" else 0,
                    "static_score_calls": 0, "diversity_calls": 0,
                    "store_permutation_sha256": (
                        manifests["image_only"]["permutation_sha256"]
                        if cache and method == "ours25" else None),
                    "persistence_source": (
                        "raster" if turn == 1 and method == "fullload"
                        else "image_only" if turn == 1 and method == "ours25"
                        else None),
                    "rater_selection_ms": 1.0 if cache and method == "qa_chunk25" else 0.0,
                    "query_projection_ms": 1.0 if cache and method == "qa_chunk25" else 0.0,
                    "probe_io_ms": 1.0 if cache and method == "qa_chunk25" else 0.0,
                    "query_scoring_ms": 1.0 if cache and method == "qa_chunk25" else 0.0,
                    "chunk_aggregation_ms": 1.0 if cache and method == "qa_chunk25" else 0.0,
                    "topk_chunk_ms": 1.0 if cache and method == "qa_chunk25" else 0.0,
                    "selected_id_d2h_ms": 0.2 if cache else 0.0,
                    "chunk_planning_ms": 0.3 if cache else 0.0,
                    "selector_wall_ms": 7.0 if cache and method == "qa_chunk25" else 0.0,
                    "chunk_io_ms": 2.0 if cache else 0.0,
                    "scatter_ms": 1.0 if cache else 0.0,
                    "prefill_ms": 50.0,
                }
                rows.append(row)
                prior[method].append(row)
        return {
            "schema_version": ANALYZER.SHARD_SCHEMA_VERSION,
            "experiment_id": self.config["experiment_id"],
            "dialogues_file_sha256": self.config["dialogues_file_sha256"],
            "source_full_workload_sha256": self.config["source_full_workload_sha256"],
            "selected_workload_sha256": self.config["selected_workload_sha256"],
            "model_revision": ANALYZER.FROZEN_MODEL_REVISION,
            "protocol": ANALYZER.PROTOCOL, "image_id": dialog["image_id"],
            "store_manifests": manifests,
            "store_build_counts": {"raster": 1, "image_only": 1},
            "persistence_overhead": {
                "raster": {
                    "source_dialog_id": dialog["dialog_id"],
                    "source_method_key": "fullload",
                    "source_execution_id": f"physical:generated_history:{dialog['dialog_id']}:t1:fullload",
                    "timing_ms": {"persist_ms": 20.0, "ssd_write_ms": 15.0},
                    "bytes": {"total": 500},
                },
                "image_only": {
                    "source_dialog_id": dialog["dialog_id"],
                    "source_method_key": "ours25",
                    "source_execution_id": f"physical:generated_history:{dialog['dialog_id']}:t1:ours25",
                    "timing_ms": {"persist_ms": 30.0, "ssd_write_ms": 25.0},
                    "bytes": {"total": 450},
                },
            },
            "rows": rows,
        }

    def artifact(self, image: str) -> tuple[Path, dict]:
        path = self.run / "images" / f"{image}.json"
        return path, json.loads(path.read_text())

    def rewrite(self, image: str, value: dict) -> None:
        value.pop("artifact_content_sha256", None)
        value["artifact_content_sha256"] = ANALYZER._stable_json(value)
        _write_json(self.run / "images" / f"{image}.json", value)


class MetricTests(unittest.TestCase):
    def test_repository_soft_scores_and_normalization(self):
        answers = ["cat", "cat", "Cat!", "dog", "dog", "bird",
                   "horse", "cow", "sheep", "goat"]
        self.assertEqual(ANALYZER.soft_vqa_score("zebra", answers), 0.0)
        self.assertAlmostEqual(ANALYZER.soft_vqa_score("bird", answers), 1 / 3)
        self.assertAlmostEqual(ANALYZER.soft_vqa_score("dog", answers), 2 / 3)
        self.assertEqual(ANALYZER.soft_vqa_score("The cat", answers), 1.0)
        with self.assertRaises(ANALYZER.AnalysisError):
            ANALYZER.soft_vqa_score("cat", answers[:1])

    def test_quality_sum_remains_soft(self):
        rows = []
        for method in ANALYZER.METHOD_KEYS:
            for turn, score in zip((1, 2, 3), (1 / 3, 2 / 3, 1.0)):
                rows.append({"method_key": method, "turn_id": turn,
                             "vqa_score": score,
                             "full_credit_correct": int(score == 1.0)})
        summary, per_turn = ANALYZER.quality_tables(rows)
        self.assertAlmostEqual(summary[0]["avg"], 2 / 3)
        self.assertAlmostEqual(per_turn[0]["score_sum"], 1 / 3)

    def test_pooled_percentile_is_recomputed_from_raw_values(self):
        rows = []
        for method in ANALYZER.METHOD_KEYS:
            rows.extend([
                {"method_key": method, "turn_id": 2, "ttft_ms": 0,
                 "input_token_count": 10, "generated_token_count": 1},
                {"method_key": method, "turn_id": 3, "ttft_ms": 100,
                 "input_token_count": 20, "generated_token_count": 2},
            ])
        latency, _ = ANALYZER.latency_and_lengths(rows)
        pooled = ANALYZER._lookup(latency, "recompute", "pooled_t2_t3")
        self.assertEqual(pooled["ttft_p50_ms"], 50.0)

    def test_cluster_bootstrap_is_deterministic(self):
        dialogs = {dialog["dialog_id"]: dialog for dialog in (_dialog(1), _dialog(2))}
        matrix = {}
        for did in dialogs:
            for turn in ANALYZER.TURNS:
                matrix[(did, turn, "qa_chunk25")] = {"vqa_score": 1.0}
                matrix[(did, turn, "ours25")] = {"vqa_score": 0.0 if did == "d1" else 1.0}
        left = ANALYZER.clustered_soft_bootstrap(matrix, dialogs, 100, 7)
        right = ANALYZER.clustered_soft_bootstrap(matrix, dialogs, 100, 7)
        self.assertEqual(left, right)
        self.assertEqual(left["avg"]["difference"], 0.5)


class EndToEndTests(unittest.TestCase):
    def test_complete_root_tree_hashes_and_report_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticRun(Path(temporary))
            result = ANALYZER.analyze(
                fixture.run, fixture.results, fixture.index, fixture.protection,
                expected_dialogs=2, expected_images=2,
                bootstrap_resamples=200, bootstrap_seed=7)
            _, source_artifact = fixture.artifact("image1")
            self.assertTrue(all("experiment_id" not in row
                                for row in source_artifact["rows"]))
            self.assertTrue(result["validation"]["passed"])
            self.assertEqual(result["validation"]["observed_logical_rows"], 24)
            required = {
                "raw.jsonl", "raw.jsonl.gz", "summary.csv",
                "quality_by_turn.csv", "ttft_by_turn.csv", "token_lengths.csv",
                "io_breakdown.csv", "selector_breakdown.csv",
                "persistence_summary.csv", "session_latency.csv",
                "selection_analysis.json", "error_propagation.csv",
                "error_propagation.json", "paired_quality.json",
                "dataset_construction.json", "validation.json", "config.json",
                "ANALYSIS.md", "README.md", "COMPLETED",
            }
            self.assertEqual({path.name for path in fixture.results.iterdir()}, required)
            completed = json.loads((fixture.results / "COMPLETED").read_text())
            self.assertEqual(completed["logical_rows"], 24)
            self.assertEqual(completed["experiment_id"], "synthetic-experiment")
            self.assertEqual(completed["index_sha256"], fixture.config["dialogues_file_sha256"])
            self.assertEqual(completed["workload_sha256"], fixture.config["source_full_workload_sha256"])
            self.assertEqual(completed["model_revision"], ANALYZER.FROZEN_MODEL_REVISION)
            self.assertEqual(completed["methods"], list(ANALYZER.METHOD_KEYS))
            self.assertEqual(completed["protocol"], ANALYZER.PROTOCOL)
            self.assertEqual(completed["bootstrap"], {
                "resamples": 200, "seed": 7, "cluster_unit": "image"})
            self.assertNotIn("COMPLETED", completed["output_sha256"])
            self.assertEqual(set(completed["output_sha256"]), required - {"COMPLETED"})
            for name, digest in completed["output_sha256"].items():
                self.assertEqual(digest, ANALYZER._sha_file(fixture.results / name))
            report = (fixture.results / "ANALYSIS.md").read_text()
            headings = [
                "### MT-VQA-v2 Generated-History Quality",
                "### MT-VQA-v2 Cache-hit TTFT",
                "### Quality–Efficiency Summary",
            ]
            positions = [report.index(heading) for heading in headings]
            self.assertEqual(positions, sorted(positions))
            for number in range(1, 10):
                self.assertIn(f"### Q{number} ", report)
            self.assertIn("VALIDATED: YES", report)
            paired = json.loads((fixture.results / "paired_quality.json").read_text())
            self.assertEqual(paired["primary_metric"], "paired soft VQA score")
            self.assertTrue(paired["full_credit_cells_are_diagnostic_only"])
            with self.assertRaises(FileExistsError):
                ANALYZER.analyze(
                    fixture.run, fixture.results, fixture.index, fixture.protection,
                    expected_dialogs=2, expected_images=2,
                    bootstrap_resamples=10, bootstrap_seed=7)

    def test_cross_method_generated_history_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticRun(Path(temporary))
            _, artifact = fixture.artifact("image1")
            row = next(row for row in artifact["rows"]
                       if row["turn_id"] == 2 and row["method_key"] == "ours25")
            row["history_source_request_ids"] = [
                "generated_history:d1:t1:qa_chunk25"]
            fixture.rewrite("image1", artifact)
            with self.assertRaisesRegex(ANALYZER.AnalysisError, "lineage"):
                ANALYZER.analyze(
                    fixture.run, fixture.results, fixture.index, fixture.protection,
                    expected_dialogs=2, expected_images=2,
                    bootstrap_resamples=20, bootstrap_seed=7)

    def test_all_ten_answers_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticRun(Path(temporary))
            _, artifact = fixture.artifact("image1")
            artifact["rows"][0]["gold"] = artifact["rows"][0]["gold"][:1]
            artifact["rows"][0]["gold_answers"] = artifact["rows"][0]["gold_answers"][:1]
            fixture.rewrite("image1", artifact)
            with self.assertRaisesRegex(ANALYZER.AnalysisError, "ten"):
                ANALYZER.analyze(
                    fixture.run, fixture.results, fixture.index, fixture.protection,
                    expected_dialogs=2, expected_images=2,
                    bootstrap_resamples=20, bootstrap_seed=7)

    def test_ours_selection_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticRun(Path(temporary))
            _, artifact = fixture.artifact("image1")
            row = next(row for row in artifact["rows"]
                       if row["turn_id"] == 3 and row["method_key"] == "ours25")
            row["selected_chunk_ids_per_layer"] = [[1], [1]]
            fixture.rewrite("image1", artifact)
            with self.assertRaisesRegex(ANALYZER.AnalysisError, "fixed-prefix"):
                ANALYZER.analyze(
                    fixture.run, fixture.results, fixture.index, fixture.protection,
                    expected_dialogs=2, expected_images=2,
                    bootstrap_resamples=20, bootstrap_seed=7)


if __name__ == "__main__":
    unittest.main()

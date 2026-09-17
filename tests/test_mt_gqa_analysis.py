"""CPU-only contracts for the strict MT-GQA analyzer."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/38_analyze_mt_gqa_full.py"
SPEC = importlib.util.spec_from_file_location("mt_gqa_analysis_test", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


class StatisticsTests(unittest.TestCase):
    def test_exact_mcnemar_uses_two_sided_binomial(self):
        result = MOD.exact_mcnemar([1, 1, 1, 0], [0, 0, 1, 0])
        self.assertEqual(result["a_only"], 2)
        self.assertEqual(result["b_only"], 0)
        self.assertEqual(result["p_value_two_sided_exact"], 0.5)

    def test_dialogue_bootstrap_is_seeded_and_paired(self):
        values = np.zeros((3, 4, 4), dtype=float)
        values[:, MOD.METHOD_ORDER["fullload"], :] = 0.5
        values[:, MOD.METHOD_ORDER["prefix25"], :] = 0.25
        first = MOD.dialogue_cluster_bootstrap(values, n_resamples=1000, seed=7)
        second = MOD.dialogue_cluster_bootstrap(values, n_resamples=1000, seed=7)
        self.assertEqual(first, second)
        delta = first["delta_vs_fullload"]["prefix25"]
        self.assertEqual(delta["estimate"], [-0.25] * 4)
        self.assertEqual(first["cluster_unit"], "dialogue")


class AnalyzerFixtureTests(unittest.TestCase):
    @staticmethod
    def _make_run(root: Path, dialogs: int = 2) -> tuple[Path, Path]:
        run = root / "run"
        images = run / "images"
        images.mkdir(parents=True)
        dataset = root / "dataset"
        dataset.mkdir()
        dialogues = {
            "schema_version": "synthetic", "benchmark_type": "synthetic",
            "dialogues": [{"dialog_id": f"dialog-{i}"}
                          for i in range(dialogs)],
        }
        dialogues_path = dataset / "dialogues.json"
        dialogues_path.write_text(json.dumps(dialogues))
        dialogues_sha = MOD._sha256_file(dialogues_path)
        workload_sha = "b" * 64
        provenance_doc = {
            "schema_version": "synthetic", "benchmark_type": "synthetic"}
        stats_doc = {
            "schema_version": "synthetic", "benchmark_type": "synthetic"}
        (dataset / "dataset_provenance.json").write_text(
            json.dumps(provenance_doc))
        (dataset / "dataset_stats.json").write_text(json.dumps(stats_doc))
        hashes = {name: MOD._sha256_file(dataset / name) for name in (
            "dialogues.json", "dataset_stats.json", "dataset_provenance.json")}
        (dataset / "config.json").write_text(json.dumps({
            "schema_version": "synthetic", "benchmark_type": "synthetic",
            "n_dialogues": dialogs, "workload_sha256": workload_sha,
            "artifact_sha256": hashes,
        }))
        provenance = {
            "dialogues_file_sha256": dialogues_sha,
            "source_full_workload_sha256": workload_sha,
            "selected_workload_sha256": "d" * 64,
            "model_revision": "c" * 40,
        }
        repeated_hashes = {key: value for key, value in provenance.items()
                           if key != "model_revision"}
        (run / "config.json").write_text(json.dumps({
            "schema_version": "synthetic",
            "benchmark_type": "MT-GQA-reconstructed",
            **provenance,
        }))
        methods = MOD.METHOD_KEYS
        for image_number in range(dialogs):
            image_id = f"image-{image_number}"
            rows = []
            dialog_id = f"dialog-{image_number}"
            order = list(methods[image_number % 4:] + methods[:image_number % 4])
            for turn in MOD.TURNS:
                for position, method in enumerate(order):
                    cached = turn > 1 and method != "recompute"
                    prediction = "The Cat." if method != "prefix25" else "dog"
                    score = 1.0 if method != "prefix25" else 0.0
                    ratio = {"fullload": 1.0, "prefix25": 0.25,
                             "prefix45": 0.5}.get(method)
                    rows.append({
                        "dialog_id": dialog_id,
                        "image_id": image_id,
                        "turn_id": turn,
                        "question_id": f"q-{image_number}-{turn}",
                        "method_key": method,
                        "method_order": order,
                        "method_order_position": position,
                        "prediction": prediction,
                        "gold": ["cat"],
                        "score": score,
                        "quality_score": score,
                        "end_to_end_ttft_ms": 10.0 + turn,
                        "request_e2e_ms": 15.0 + turn,
                        "decode_ms": 5.0,
                        "ssd_read_bytes": (1000 if cached else 0),
                        "ssd_read_ms": (1.0 if cached else 0.0),
                        "normal_kv_preads": (2 if cached else 0),
                        "separator_preads": (1 if cached else 0),
                        "scatter_ms": (0.5 if cached else 0.0),
                        "first_k_planning_ms": (0.1 if cached else 0.0),
                        "selected_visual_kv_ratio": (ratio if cached else None),
                        "actual_selected_normal_chunk_fraction": (
                            ratio if cached else None),
                        # Separator rows make the logical retained-token ratio
                        # larger than the physical normal-chunk budget.
                        "selected_kv_ratio": (
                            ratio + 0.02
                            if cached and method in {"prefix25", "prefix45"}
                            else ratio if cached else None),
                        "vision_forward_count": (0 if cached else 1),
                        "cache_hit": cached,
                        "first_token_id": (17 if method != "prefix25" else 23),
                        "prompt_sha256": f"prompt-{image_number}-{turn}",
                        "text_history_sha256": f"history-{image_number}-{turn}",
                        **repeated_hashes,
                    })
            artifact = {
                "image_id": image_id,
                **repeated_hashes,
                "persistence_overhead": {
                    "permutation_ms": 1.0,
                    "kv_repack_ms": 2.0,
                    # The writer also exposes the broader materialize+repack
                    # aggregate.  It is intentionally not an alias for the
                    # exact KV-repack component analyzed above.
                    "repack_ms": 7.0,
                    "ssd_write_ms": 3.0,
                    "fsync_ms": 4.0,
                    "total_persist_ms": 10.0,
                    "bytes_written": 10000,
                },
                "rows": rows,
                "validation": {"passed": True},
            }
            artifact["artifact_content_sha256"] = MOD._stable_json_hash(artifact)
            (images / f"{image_id}.json").write_text(json.dumps(artifact))
        return run, dataset / "dataset_provenance.json"

    def test_end_to_end_writes_all_outputs_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, provenance = self._make_run(root)
            before = MOD.source_manifest(run)
            result_dir = root / "results"
            validation = MOD.analyze(
                run, result_dir, expected_dialogs=2,
                dataset_provenance=provenance)
            self.assertTrue(validation["passed"])
            self.assertEqual(set(path.name for path in result_dir.iterdir()),
                             set(MOD.REQUIRED_OUTPUTS))
            self.assertEqual(MOD.source_manifest(run), before)
            with (result_dir / "raw.jsonl").open() as handle:
                self.assertEqual(sum(1 for _ in handle), 24)
            stats = json.loads((result_dir / "statistical_analysis.json").read_text())
            self.assertEqual(stats["bootstrap"]["n_resamples"], 10_000)
            self.assertEqual(stats["bootstrap"]["seed"], 1234)
            self.assertIn("conclusions", stats)
            self.assertEqual(
                validation["full_load_and_turn1_sanity"]
                ["recomp_vs_fullload_by_turn"]["turn1"]
                ["first_token_agreement_fraction"], 1.0)
            self.assertIn("### Q10", (result_dir / "README.md").read_text())
            self.assertIn("Future-query robustness",
                          (result_dir / "README.md").read_text())
            result_config = json.loads((result_dir / "config.json").read_text())
            self.assertEqual(result_config["model_revision"], "c" * 40)
            self.assertIn("canonical_dataset_config", result_config)
            self.assertEqual(result_config["quality_metric"],
                             "normalized_exact_match")
            self.assertEqual(validation["quality_metric_audit"]
                             ["discrepancy_count"], 0)

    def test_prefix_tolerant_source_score_is_not_primary_quality(self):
        self.assertEqual(MOD.gqa_score("computer mouse", ["computer"]), 1.0)
        self.assertEqual(MOD.strict_gqa_score(
            "computer mouse", ["computer"]), 0.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, provenance = self._make_run(root)
            path = run / "images/image-0.json"
            artifact = json.loads(path.read_text())
            for row in artifact["rows"]:
                if row["turn_id"] != 2:
                    continue
                row["gold"] = ["computer"]
                row["prediction"] = (
                    "computer mouse" if row["method_key"] == "prefix45"
                    else "computer")
                row["score"] = 1.0
                row["quality_score"] = 1.0
            artifact["artifact_content_sha256"] = MOD._stable_json_hash({
                key: value for key, value in artifact.items()
                if key != "artifact_content_sha256"
            })
            path.write_text(json.dumps(artifact))
            before = MOD.source_manifest(run)
            result = root / "results"
            validation = MOD.analyze(
                run, result, expected_dialogs=2,
                dataset_provenance=provenance)
            self.assertEqual(MOD.source_manifest(run), before)
            audit = validation["quality_metric_audit"]
            self.assertEqual(audit["discrepancy_count"], 1)
            self.assertEqual(audit["discrepancy_count_by_method"]["prefix45"], 1)
            self.assertEqual(audit["examples"][0]["prediction"],
                             "computer mouse")
            raw = [json.loads(line) for line in
                   (result / "raw.jsonl").read_text().splitlines()]
            changed = next(row for row in raw if row["dialog_id"] == "dialog-0"
                           and row["turn_id"] == 2
                           and row["method_key"] == "prefix45")
            self.assertEqual(changed["stored_legacy_score"], 1.0)
            self.assertEqual(changed["legacy_quality_score"], 1.0)
            self.assertEqual(changed["legacy_recomputed_score"], 1.0)
            self.assertEqual(changed["score"], 0.0)
            self.assertEqual(changed["quality_score"], 0.0)
            self.assertEqual(changed["recomputed_score"], 0.0)
            import csv
            with (result / "quality_by_turn.csv").open() as handle:
                quality = {row["method_key"]: row for row in
                           csv.DictReader(handle)}
            self.assertEqual(float(quality["prefix45"]["acc2"]), 0.5)
            self.assertAlmostEqual(float(quality["prefix45"]["avg"]), 5 / 6)
            with (result / "per_turn.csv").open() as handle:
                per_turn = list(csv.DictReader(handle))
            selected = next(row for row in per_turn if
                            row["dialog_id"] == "dialog-0" and
                            row["turn_id"] == "2" and
                            row["method_key"] == "prefix45")
            self.assertEqual(float(selected["score"]), 0.0)
            self.assertEqual(float(selected["stored_legacy_score"]), 1.0)
            stats = json.loads((result / "statistical_analysis.json").read_text())
            self.assertEqual(stats["quality_metric_audit"]["discrepancy_count"], 1)
            self.assertEqual(stats["bootstrap"]["absolute"]["prefix45"]
                             ["estimate"][1], 0.5)
            self.assertEqual(stats["mcnemar_vs_fullload"]["prefix45"]
                             ["turn2"]["b_only"], 1)
            readme = (result / "README.md").read_text()
            self.assertIn("strict normalized exact match", readme)
            self.assertIn("computer mouse", readme)

    def test_bootstrap_protocol_is_frozen_before_any_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, provenance = self._make_run(root)
            result = root / "results"
            with self.assertRaisesRegex(MOD.AnalysisError, "10,000"):
                MOD.analyze(
                    run, result, expected_dialogs=2,
                    bootstrap_resamples=9999,
                    dataset_provenance=provenance)
            self.assertFalse(result.exists())

    def test_existing_destination_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, provenance = self._make_run(root)
            result = root / "results"
            result.mkdir()
            marker = result / "keep"
            marker.write_text("unchanged")
            with self.assertRaises(FileExistsError):
                MOD.analyze(run, result, expected_dialogs=2,
                            dataset_provenance=provenance)
            self.assertEqual(marker.read_text(), "unchanged")

    def test_missing_matrix_cell_fails_without_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, provenance = self._make_run(root)
            path = run / "images/image-0.json"
            artifact = json.loads(path.read_text())
            artifact["rows"].pop()
            artifact["artifact_content_sha256"] = MOD._stable_json_hash({
                key: value for key, value in artifact.items()
                if key != "artifact_content_sha256"
            })
            path.write_text(json.dumps(artifact))
            result = root / "results"
            with self.assertRaises(MOD.AnalysisError):
                MOD.analyze(run, result, expected_dialogs=2,
                            dataset_provenance=provenance)
            self.assertFalse(result.exists())

    def test_score_is_independently_recomputed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, provenance = self._make_run(root)
            path = run / "images/image-0.json"
            artifact = json.loads(path.read_text())
            artifact["rows"][0]["score"] = 0.0
            artifact["artifact_content_sha256"] = MOD._stable_json_hash({
                key: value for key, value in artifact.items()
                if key != "artifact_content_sha256"
            })
            path.write_text(json.dumps(artifact))
            with self.assertRaisesRegex(MOD.AnalysisError, "stored score"):
                MOD.analyze(run, root / "results", expected_dialogs=2,
                            dataset_provenance=provenance)


if __name__ == "__main__":
    unittest.main()

"""CPU-only tests for the QA-Chunk25 final report publisher."""
from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load(
    "qa_chunk_runner_for_report_test",
    ROOT / "scripts/52_eval_query_aware_chunk_baseline.py")
REPORT = _load(
    "qa_chunk_report_tested",
    ROOT / "scripts/53_report_query_aware_chunk_baseline.py")


class QAChunkReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name).resolve()
        self.run_root = root / "runs/query_aware_chunk_baseline"
        self.results_root = root / "results/query_aware_chunk_baseline"
        self.run = self.run_root / "gqa40_240_fixture"
        self.results = self.results_root / "gqa40_240_fixture"
        self.run.mkdir(parents=True)
        self.results.mkdir(parents=True)
        self._write_bundle()

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _json(path: Path, value) -> None:
        path.write_text(json.dumps(
            value, indent=2, sort_keys=True, ensure_ascii=False,
            allow_nan=False) + "\n", encoding="utf-8")

    def _row(self, image: int, turn: int, method: str) -> dict:
        image_id = f"image-{image:02d}"
        question_id = f"question-{image:02d}-{turn}"
        hit = turn > 1
        values = {
            "recompute": (0, 0, 0, 0.0, None, 0.0, 0),
            "fullload": (990, 0, 10, 1.0, 1.0, 4.0, 3),
            "qa_token25": (870, 20, 10, .9, .875, 12.0, 8),
            "qa_chunk25": (200, 20, 10, .23, .25, 6.0, 7),
            "ours25": (240, 0, 10, .25, .25, 1.0, 3),
        }
        normal, probe, separator, ratio, touched, selector, preads = values[method]
        if not hit:
            normal = probe = separator = preads = 0
            ratio, touched, selector = 0.0, None, 0.0
        selected = []
        actual = []
        if hit and method == "qa_chunk25":
            selected = ([[0, 2], [1, 3]] if (image + turn) % 2
                        else [[1, 3], [0, 2]])
            actual = selected
        elif hit and method == "qa_token25":
            selected = [list(range(7)), list(range(7))]
        elif hit and method == "ours25":
            selected = [[0, 1], [0, 1]]
        elif hit and method == "fullload":
            selected = [list(range(8)), list(range(8))]
        total = normal + probe + separator
        row = {
            "schema_version": RUNNER.SCHEMA_VERSION,
            "run_id": "fixture-run",
            "request_id": RUNNER.request_id(image_id, question_id, method),
            "dataset": "gqa", "image_id": image_id,
            "question_id": question_id, "turn_id": turn,
            "method_key": method, **RUNNER.METHODS[method],
            "prediction": "yes", "correct": float(
                method in {"recompute", "fullload", "qa_token25"}),
            "end_to_end_ttft_ms": float({
                "recompute": 500, "fullload": 650, "qa_token25": 900,
                "qa_chunk25": 700, "ours25": 260,
            }[method] + image / 100 + turn / 1000),
            "actual_ssd_mb": total / 1e6,
            "actual_ssd_ratio_vs_fullload": ratio,
            "ssd_read_bytes": total, "normal_kv_read_bytes": normal,
            "probe_read_bytes": probe, "separator_read_bytes": separator,
            "ssd_preads": preads, "ssd_read_ms": float(preads) / 10,
            "normal_selected_chunk_ratio": touched,
            "touched_chunk_fraction": touched,
            "selected_chunk_ids_per_layer": selected,
            "actual_loaded_chunk_ids_per_layer": actual,
            "expected_layers": 2,
            "query_score_calls": 2 if hit and method in {
                "qa_token25", "qa_chunk25"} else 0,
            "chunk_score_calls": 2 if hit and method == "qa_chunk25" else 0,
            "chunk_score_aggregation": (
                "mean_valid_spatial_token_importance"
                if method == "qa_chunk25" else None),
            "full_load_fallback_count": 0,
            "adaptive_ratio": False,
            "contiguous_runs_per_layer_mean": (
                2.0 if hit and method == "qa_chunk25" else
                1.0 if hit and method != "recompute" else None),
            "mean_contiguous_run_length": (
                1.0 if hit and method == "qa_chunk25" else
                2.0 if hit and method == "ours25" else
                7.0 if hit and method == "qa_token25" else
                8.0 if hit and method == "fullload" else None),
            "future_question_ids_used": [],
            "future_questions_in_prompt": 0,
            "selector_ms": selector,
            "online_selector_total_ms": selector,
            "selector_decision_host_wall_ms": selector,
            "rater_selection_ms": selector / 6,
            "query_projection_ms": selector / 6,
            "probe_h2d_ms": selector / 60,
            "probe_io_ms": selector / 6,
            "probe_read_pipeline_ms": selector / 5,
            "normal_kv_read_ms": float(preads) / 25,
            "selected_chunk_io_ms": float(preads) / 25,
            "separator_read_ms": .01 if hit and method != "recompute" else 0,
            "query_scoring_ms": selector / 6,
            "chunk_aggregation_ms": (
                selector / 6 if method == "qa_chunk25" else 0),
            "topk_chunk_ms": (
                selector / 6 if method == "qa_chunk25" else 0),
            "selected_id_d2h_ms": selector / 20,
            "chunk_planning_ms": selector / 20,
            "chunk_io_ms": float(preads) / 20,
            "scatter_ms": selector / 12,
            "prefill_ms": 40.0,
        }
        return row

    def _write_bundle(self) -> None:
        rows = [self._row(image, turn, method)
                for image in range(40) for turn in range(1, 7)
                for method in RUNNER.METHOD_KEYS]
        raw = "".join(json.dumps(
            row, separators=(",", ":"), ensure_ascii=False) + "\n"
            for row in rows)
        (self.run / "results_partial.jsonl").write_text(raw, encoding="utf-8")
        (self.run / "results_final.jsonl").write_text(raw, encoding="utf-8")

        config = {
            "schema_version": RUNNER.SCHEMA_VERSION,
            "run_id": "fixture-run", "status": "complete", "dataset": "gqa",
            "index_sha256": REPORT.EXPECTED_INDEX_SHA256,
            "full_workload_sha256": REPORT.EXPECTED_WORKLOAD_SHA256,
            "selected_workload_sha256": REPORT.EXPECTED_WORKLOAD_SHA256,
            "full_images": 40, "full_questions": 240,
            "selected_images": 40, "selected_questions": 240,
            "n_images": 40, "n_questions": 240,
            "skip": 4, "questions_per_image": 6, "seed": 1234,
            "max_new_tokens": 16, "future_question_leakage": 0,
            "method_keys": list(RUNNER.METHOD_KEYS), "methods": RUNNER.METHODS,
            "run_dir": str(self.run), "results_dir": str(self.results),
            "store_dir": str(self.run_root / "read_only_store"),
        }
        manifest = {
            "schema_version": RUNNER.SCHEMA_VERSION,
            "run_id": "fixture-run", "method_keys": list(RUNNER.METHOD_KEYS),
            "methods": RUNNER.METHODS, "expected_request_count": 1200,
        }
        self._json(self.run / "config.json", config)
        self._json(self.run / "manifest.json", manifest)

        summaries = RUNNER.summaries_from_rows(rows, 1000.0)
        selection = RUNNER.selection_analysis(rows)
        comparison = RUNNER._comparison(summaries)
        summary = {
            "schema_version": RUNNER.SCHEMA_VERSION, "config": config,
            "per_method": summaries, "comparison": comparison,
            "selection": {key: value for key, value in selection.items()
                          if key not in {"requests", "pairs"}},
            "reference_consistency": {"fixture": True},
        }
        validation = {
            "schema_version": RUNNER.SCHEMA_VERSION, "passed": True,
            "expected": 1200, "completed": 1200, "failed": 0,
            "duplicates": 0, "checks": {"fixture_contracts": True},
            "limitations": ["synthetic report unit-test fixture"],
        }
        self._json(self.run / "summary.json", summary)
        self._json(self.run / "selection_analysis.json", selection)
        self._json(self.run / "validation.json", validation)
        RUNNER._write_csv(self.run / "summary.csv", [
            {"method_key": key, **summaries[key]} for key in RUNNER.METHOD_KEYS])
        latency = (
            "rater_selection_ms", "query_projection_ms", "probe_h2d_ms",
            "probe_io_ms", "normal_kv_read_ms", "selected_chunk_io_ms",
            "separator_read_ms", "query_scoring_ms", "chunk_aggregation_ms",
            "topk_chunk_ms", "selected_id_d2h_ms", "chunk_planning_ms",
            "chunk_io_ms", "scatter_ms", "prefill_ms",
            "online_selector_total_ms", "ttft_cache_hit_mean_ms",
        )
        RUNNER._write_csv(self.run / "latency_breakdown.csv", [{
            "method_key": key,
            **{field: summaries[key].get(field) for field in latency},
        } for key in RUNNER.METHOD_KEYS])
        io_fields = (
            "normal_selected_chunk_ratio", "total_touched_chunk_ratio",
            "probe_io_mb", "selected_chunk_payload_mb", "separator_io_mb",
            "actual_ssd_mb_per_cache_hit", "actual_ssd_ratio_vs_fullload",
            "ssd_preads_per_cache_hit", "ssd_read_latency_ms",
            "contiguous_runs_per_layer", "mean_contiguous_run_length",
            "max_contiguous_run_length",
        )
        RUNNER._write_csv(self.run / "io_breakdown.csv", [{
            "method_key": key,
            **{field: summaries[key].get(field) for field in io_fields},
        } for key in RUNNER.METHOD_KEYS])
        (self.run / "RUN_ANALYSIS.md").write_text(
            "runner-owned concise report\n", encoding="utf-8")

        exports = REPORT.RUNNER_EXPORTS
        for name in exports:
            shutil.copyfile(self.run / name, self.results / name)
        files = ("manifest.json", "config.json", "results_partial.jsonl", *exports)
        artifacts = {
            "schema_version": RUNNER.SCHEMA_VERSION,
            "run_dir": str(self.run), "results_dir": str(self.results),
            "store_dir": config["store_dir"],
            "files_sha256": {
                name: REPORT._sha256_file(self.run / name) for name in files},
        }
        self._json(self.run / "run_artifacts.json", artifacts)
        shutil.copyfile(
            self.run / "run_artifacts.json", self.results / "run_artifacts.json")
        self._json(self.run / "COMPLETED", {
            "schema_version": RUNNER.SCHEMA_VERSION, "run_id": "fixture-run",
            "passed": True, "completed": 1200,
            "validation_sha256": REPORT._sha256_file(
                self.run / "validation.json"),
        })

        entries = {}
        canonical = REPORT._canonical_hash(entries)
        protection_manifest = {
            "schema_version": "query-aware-protected-artifacts-v1",
            "excluded_new_roots": [str(self.run_root), str(self.results_root)],
            "entries": entries, "manifest_sha256": canonical,
            "entry_count": 0, "file_count": 0, "directory_count": 0,
            "symlink_count": 0, "total_bytes": 0,
        }
        self._json(
            self.run_root / "protected_artifacts_before.json",
            protection_manifest)
        self._json(self.results_root / "protected_artifacts_validation.json", {
            "passed": True,
            "before_manifest_sha256": canonical,
            "after_manifest_sha256": canonical,
            "entry_count_before": 0, "entry_count_after": 0,
            "file_count_before": 0, "file_count_after": 0,
            "directory_count_before": 0, "directory_count_after": 0,
            "symlink_count_before": 0, "symlink_count_after": 0,
            "total_bytes_before": 0, "total_bytes_after": 0,
            "missing_paths": [], "added_paths": [], "changed_paths": [],
        })

        source_manifest = {
            "schema_version": "qa-chunk-source-store-provenance-v1",
            "source_store_fingerprint_sha256": "a" * 64,
            "source_store_tree_sha256": None,
        }
        source_manifest["manifest_sha256"] = REPORT._canonical_hash(
            source_manifest)
        self._json(self.run_root / "source_stores_before.json", source_manifest)
        self._json(self.results_root / "source_stores_validation.json", {
            "passed": True,
            "read_only_source_reuse_validated": True,
            "before_manifest_sha256": source_manifest["manifest_sha256"],
            "before_store_fingerprint_sha256": "a" * 64,
            "after_store_fingerprint_sha256": "a" * 64,
            "before_store_tree_sha256": None,
            "after_store_tree_sha256": None,
            "missing_paths": [], "added_paths": [], "changed_paths": [],
        })

    def test_publishes_numbered_report_and_preserves_runner_artifacts(self):
        before = {name: REPORT._sha256_file(self.results / name)
                  for name in REPORT.RUNNER_EXPORTS}
        output = REPORT.generate_report(
            self.run, self.results, test_result="217 tests passed in 4.2s")
        self.assertTrue(output["passed"])
        analysis = (self.results / "ANALYSIS.md").read_text(encoding="utf-8")
        for number in range(1, 27):
            self.assertIn(f"### {number}.", analysis)
        for letter in "ABCDEF":
            self.assertIn(f"### {letter}.", analysis)
        self.assertIn("| QA-Chunk25 |", analysis)
        self.assertIn(
            "selected K/V raw pread=0.28 ms/request", analysis)
        self.assertIn(
            "host chunk-I/O interval", analysis)
        self.assertEqual(analysis.rstrip().splitlines()[-1],
                         "QA-CHUNK25 BASELINE VALIDATED: YES")
        after = {name: REPORT._sha256_file(self.results / name)
                 for name in REPORT.RUNNER_EXPORTS}
        self.assertEqual(before, after)
        repeated = REPORT.generate_report(
            self.run, self.results, test_result="217 tests passed in 4.2s")
        self.assertEqual(set(repeated["publish_status"].values()), {"preserved"})

    def test_missing_source_verification_is_reported_as_no(self):
        (self.results_root / "source_stores_validation.json").unlink()
        output = REPORT.generate_report(
            self.run, self.results, test_result="217 tests passed in 4.2s")
        self.assertFalse(output["passed"])
        validation = json.loads((
            self.results / "report_validation.json").read_text())
        self.assertFalse(validation["checks"]["source_stores_unchanged"])
        analysis = (self.results / "ANALYSIS.md").read_text(encoding="utf-8")
        self.assertEqual(analysis.rstrip().splitlines()[-1],
                         "QA-CHUNK25 BASELINE VALIDATED: NO")

    def test_refuses_tampered_runner_export(self):
        with (self.results / "summary.csv").open("a", encoding="utf-8") as handle:
            handle.write("tampered\n")
        with self.assertRaisesRegex(
                REPORT.ReportValidationError, "differs between run/results"):
            REPORT.generate_report(
                self.run, self.results, test_result="217 tests passed")
        self.assertFalse((self.results / "ANALYSIS.md").exists())

    def test_refuses_to_replace_conflicting_analysis(self):
        (self.results / "ANALYSIS.md").write_text(
            "someone else's analysis\n", encoding="utf-8")
        with self.assertRaisesRegex(
                REPORT.ReportValidationError, "refusing to replace"):
            REPORT.generate_report(
                self.run, self.results, test_result="217 tests passed")
        self.assertFalse((self.results / "README.md").exists())
        self.assertFalse((self.results / "report_validation.json").exists())


if __name__ == "__main__":
    unittest.main()

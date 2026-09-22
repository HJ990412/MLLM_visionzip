"""CPU-only tests for the fail-closed QA-Select final reporter."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/51_report_query_aware_baseline.py"
SPEC = importlib.util.spec_from_file_location("query_aware_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def _method(key: str, method_id: str, label: str, base: float) -> dict:
    return {
        "method_id": method_id,
        "display_label": label,
        "paper_label": "Query-Aware" if key == "qa_select25" else (
            "Ours" if key == "ours25" else label),
        "retention_ratio": (
            None if key == "recompute" else 1.0 if key == "fullload" else 0.25),
        "accuracy_all_turns": 0.5 + base,
        "accuracy_cache_hits": 0.48 + base,
        "ttft_cache_hit_mean_ms": 100.0 + 10 * base,
        "ttft_cache_hit_p50_ms": 95.0 + 10 * base,
        "ttft_cache_hit_p95_ms": 130.0 + 10 * base,
        "actual_ssd_mb_per_cache_hit": 0.0 if key == "recompute" else 100.0,
        "actual_ssd_ratio_vs_fullload": 0.0 if key == "recompute" else 1.0,
        "selector_ms": 2.0 if key == "qa_select25" else 0.0,
        "online_selector_total_ms": 2.1 if key == "qa_select25" else 0.0,
        "ssd_preads_per_cache_hit": 0.0 if key == "recompute" else 64.0,
        "ssd_read_latency_ms": 0.0 if key == "recompute" else 20.0,
        "probe_io_mb": 3.0 if key == "qa_select25" else 0.0,
        "probe_io_ms": 1.0 if key == "qa_select25" else 0.0,
        "touched_chunk_fraction": None if key == "recompute" else 0.5,
        "contiguous_runs_per_layer": None if key in {"recompute", "fullload"} else 1.0,
        "mean_contiguous_run_length": None if key in {"recompute", "fullload"} else 4.0,
        "logical_selected_token_ratio": (
            None if key == "recompute" else 1.0 if key == "fullload" else 0.25),
        "rater_selection_ms": 0.5 if key == "qa_select25" else 0.0,
        "query_projection_ms": 0.5 if key == "qa_select25" else 0.0,
        "query_scoring_ms": 0.5 if key == "qa_select25" else 0.0,
        "topk_ms": 0.2 if key == "qa_select25" else 0.0,
        "selected_id_d2h_ms": 0.2 if key == "qa_select25" else 0.0,
        "chunk_planning_ms": 0.1 if key == "qa_select25" else 0.0,
    }


class QueryAwareReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name).resolve()
        self.run_root = root / "runs/query_aware_baseline"
        self.run_dir = self.run_root / "gqa40_240"
        self.results_root = root / "results/query_aware_baseline"
        self.results_dir = self.results_root / "gqa40_240"
        self.run_dir.mkdir(parents=True)
        self.results_dir.mkdir(parents=True)
        self._write_valid_bundle()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_json(self, path: Path, value: object) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8")

    def _write_valid_bundle(self) -> None:
        methods = {
            "recompute": _method("recompute", "recompute", "ReComp", 0.00),
            "fullload": _method("fullload", "fullload", "FullLoad", 0.01),
            "qa_select25": _method(
                "qa_select25", "qa_select25", "QA-Select25", 0.02),
            "ours25": _method(
                "ours25", "imageonly_prefix25", "Ours25", 0.01),
        }
        config_methods = {
            key: {
                "method_id": row["method_id"],
                "display_label": row["display_label"],
                "paper_label": row["paper_label"],
                "retention_ratio": row["retention_ratio"],
            }
            for key, row in methods.items()
        }
        config = {
            "schema_version": REPORT.SCHEMA_VERSION,
            "status": "complete",
            "dataset": "gqa",
            "index_sha256": REPORT.EXPECTED_INDEX_SHA256,
            "full_workload_sha256": REPORT.EXPECTED_WORKLOAD_SHA256,
            "selected_workload_sha256": REPORT.EXPECTED_WORKLOAD_SHA256,
            "n_images": 40,
            "n_questions": 240,
            "selected_images": 40,
            "selected_questions": 240,
            "full_images": 40,
            "full_questions": 240,
            "skip": 4,
            "questions_per_image": 6,
            "seed": 1234,
            "max_new_tokens": 16,
            "future_question_leakage": 0,
            "method_keys": list(REPORT.METHOD_KEYS),
            "methods": config_methods,
            "run_dir": str(self.run_dir),
            "store_dir": str(self.run_root / "store"),
            "results_dir": str(self.results_dir),
        }
        selection = {
            "scope": "cache-hit turns",
            "n_query_requests": 200,
            "n_images": 40,
            "requests": [{} for _ in range(200)],
            "pairs": [{} for _ in range(400)],
            "n_pairs": 400,
            "mean_pairwise_token_jaccard": 0.7,
            "mean_pairwise_chunk_jaccard": 0.8,
            "identical_selection_rate": 0.01,
            "different_selection_pairs": 396,
            "n_consecutive_pairs": 160,
            "mean_consecutive_token_jaccard": 0.69,
            "mean_consecutive_chunk_jaccard": 0.79,
        }
        qa, ours = methods["qa_select25"], methods["ours25"]
        comparison = {
            "qa_minus_ours_accuracy_all_turns_pp": (
                qa["accuracy_all_turns"] - ours["accuracy_all_turns"]) * 100,
            "qa_minus_ours_accuracy_cache_hits_pp": (
                qa["accuracy_cache_hits"] - ours["accuracy_cache_hits"]) * 100,
            "qa_minus_ours_ttft_cache_hit_ms": (
                qa["ttft_cache_hit_mean_ms"] - ours["ttft_cache_hit_mean_ms"]),
            "qa_over_ours_ttft_ratio": (
                qa["ttft_cache_hit_mean_ms"] / ours["ttft_cache_hit_mean_ms"]),
        }
        summary = {
            "schema_version": REPORT.SCHEMA_VERSION,
            "config": config,
            "per_method": methods,
            "comparison": comparison,
            "selection": selection,
            "reference_consistency": {"available": False},
        }
        validation = {
            "schema_version": REPORT.SCHEMA_VERSION,
            "passed": True,
            "checks": {"all_contracts": True},
            "selection": selection,
            "reference_consistency": {"available": False},
            "limitations": ["synthetic unit-test fixture"],
        }
        self._write_json(self.run_dir / "config.json", config)
        self._write_json(self.run_dir / "summary.json", summary)
        self._write_json(self.run_dir / "validation.json", validation)
        self._write_json(self.run_dir / "selection.json", selection)
        (self.run_dir / "raw.jsonl").write_text("{}\n" * 960)
        (self.run_dir / "persistence.jsonl").write_text("{}\n" * 40)
        (self.run_dir / "per_request.csv").write_text("header\n")
        (self.run_dir / "summary.csv").write_text("header\n")
        (self.run_dir / "README.md").write_text("validated fixture\n")

        entries: dict = {}
        canonical = REPORT._canonical_hash(entries)
        manifest = {
            "schema_version": REPORT.PROTECTION_SCHEMA_VERSION,
            "excluded_new_roots": [str(self.run_root), str(self.results_root)],
            "entries": entries,
            "manifest_sha256": canonical,
            "entry_count": 0,
            "file_count": 0,
            "directory_count": 0,
            "symlink_count": 0,
            "total_bytes": 0,
        }
        protection = {
            "schema_version": REPORT.PROTECTION_SCHEMA_VERSION,
            "passed": True,
            "before_manifest_sha256": canonical,
            "after_manifest_sha256": canonical,
            "entry_count_before": 0,
            "entry_count_after": 0,
            "file_count_before": 0,
            "file_count_after": 0,
            "directory_count_before": 0,
            "directory_count_after": 0,
            "symlink_count_before": 0,
            "symlink_count_after": 0,
            "total_bytes_before": 0,
            "total_bytes_after": 0,
            "missing_paths": [],
            "added_paths": [],
            "changed_paths": [],
            "kvstore_trees_hashed": False,
        }
        self._write_json(
            self.run_root / "protected_artifacts_before.json", manifest)
        self._write_json(
            self.results_root / "protected_artifacts_validation.json", protection)

    def test_writes_numbered_korean_analysis_and_terminal_verdict(self):
        output = REPORT.generate_report(
            self.run_dir, self.results_dir,
            test_result="200 passed in 1.23s")
        text = output.read_text(encoding="utf-8")
        for number in range(1, 21):
            self.assertIn(f"### {number}.", text)
        self.assertIn("| QA-Select25 |", text)
        self.assertIn("200 passed in 1.23s", text)
        self.assertEqual(
            text.rstrip().splitlines()[-1],
            "QUERY-AWARE BASELINE VALIDATED: YES")

    def test_refuses_failed_validation_without_publishing(self):
        path = self.run_dir / "validation.json"
        value = json.loads(path.read_text())
        value["passed"] = False
        self._write_json(path, value)
        with self.assertRaisesRegex(
                REPORT.ReportValidationError, "validation did not pass"):
            REPORT.generate_report(self.run_dir, self.results_dir)
        self.assertFalse((self.results_dir / "ANALYSIS.md").exists())

    def test_refuses_wrong_method_id(self):
        path = self.run_dir / "summary.json"
        value = json.loads(path.read_text())
        value["per_method"]["ours25"]["method_id"] = "ours25"
        self._write_json(path, value)
        with self.assertRaisesRegex(
                REPORT.ReportValidationError, "method_id mismatch"):
            REPORT.generate_report(self.run_dir, self.results_dir)
        self.assertFalse((self.results_dir / "ANALYSIS.md").exists())

    def test_refuses_failed_artifact_protection_and_no_clobber(self):
        path = self.results_root / "protected_artifacts_validation.json"
        protection = json.loads(path.read_text())
        protection["passed"] = False
        self._write_json(path, protection)
        with self.assertRaisesRegex(
                REPORT.ReportValidationError, "verification did not pass"):
            REPORT.generate_report(self.run_dir, self.results_dir)

        protection["passed"] = True
        self._write_json(path, protection)
        REPORT.generate_report(self.run_dir, self.results_dir)
        with self.assertRaisesRegex(FileExistsError, "refusing to replace"):
            REPORT.generate_report(self.run_dir, self.results_dir)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the analysis-only VisDial cache-hit paper tables."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/32_analyze_visdial_cache_hits.py"
SPEC = importlib.util.spec_from_file_location("visdial_cache_hit_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def sample_row(method: str, ttft: float, e2e: float, quality: float,
               ssd_bytes: int) -> dict:
    return {
        "method": method,
        "dialog_id": "d1",
        "quality_score": quality,
        "end_to_end_ttft_ms": ttft,
        "request_e2e_ms": e2e,
        "ssd_read_bytes": ssd_bytes,
        "ssd_read_ms": 1.0 if ssd_bytes else 0.0,
        "ssd_preads": 1 if ssd_bytes else 0,
        "scatter_ms": None if method == "FullLoad" else 0.0,
        "prefill_ms": 5.0,
        "selector_ms": 0.1 if method.startswith("Prefix") else 0.0,
        "hook_total_ms": 2.0 if method == "FullLoad" else None,
        "selected_kv_ratio": {
            "ReComp": None, "FullLoad": 1.0,
            "Prefix25": 0.25, "Prefix45": 0.45,
        }[method],
        "static_score_calls": 0,
        "query_score_calls": 0,
        "diversity_calls": 0,
    }


class CacheHitAggregationTests(unittest.TestCase):
    def test_percentile_matches_linear_interpolation(self):
        self.assertEqual(analysis.percentile([1, 2, 3, 4], 50), 2.5)
        self.assertAlmostEqual(analysis.percentile([1, 2, 3, 4], 95), 3.85)

    def test_main_table_uses_actual_bytes_and_recomp_denominator(self):
        rows = [
            sample_row("ReComp", 100, 120, 0.8, 0),
            sample_row("FullLoad", 150, 170, 0.8, 1000),
            sample_row("Prefix25", 50, 70, 0.7, 250),
            sample_row("Prefix45", 75, 95, 0.8, 450),
        ]
        table = {row["method"]: row for row in analysis.build_main_table(rows)}
        self.assertEqual(table["Prefix25"]["ttft_reduction_vs_recomp_pct"], 50)
        self.assertEqual(table["Prefix25"]["ssd_ratio_vs_fullload_pct"], 25)
        self.assertEqual(table["Prefix45"]["ssd_reduction_vs_fullload_pct"], 55)
        self.assertIsNone(table["ReComp"]["ssd_ratio_vs_fullload_pct"])

    def test_full_load_scatter_remains_na_and_prefill_is_labelled(self):
        rows = [
            sample_row("ReComp", 100, 120, 0.8, 0),
            sample_row("FullLoad", 150, 170, 0.8, 1000),
            sample_row("Prefix25", 50, 70, 0.7, 250),
            sample_row("Prefix45", 75, 95, 0.8, 450),
        ]
        io_rows = {row["method"]: row
                   for row in analysis.build_io_breakdown(rows)}
        self.assertIsNone(io_rows["FullLoad"]["scatter_ms_mean"])
        self.assertIn("hook SSD read", io_rows["FullLoad"]["prefill_semantics"])
        self.assertEqual(
            io_rows["FullLoad"][
                "fullload_prefill_minus_hook_ms_mean_diagnostic"], 3.0
        )
        self.assertIn("no online scorer",
                      io_rows["Prefix25"]["planning_semantics"])

    def test_atomic_publication_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            destination.mkdir()
            with self.assertRaises(analysis.AnalysisError):
                analysis.rename_noreplace(source, destination)
            self.assertTrue(source.is_dir())
            self.assertTrue(destination.is_dir())


class QualityMetricTests(unittest.TestCase):
    def test_existing_normalized_generative_match_semantics(self):
        self.assertEqual(analysis.generative_match("The cat.", "cat"), 1.0)
        self.assertEqual(analysis.generative_match("two dogs nearby", "two dogs"), 1.0)
        self.assertEqual(analysis.generative_match("dog", "cat"), 0.0)


if __name__ == "__main__":
    unittest.main()

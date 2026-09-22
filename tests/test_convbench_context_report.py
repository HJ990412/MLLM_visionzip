"""Boundary checks for the standalone ConvBench context-risk summary."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "scripts/47_summarize_convbench_context.py"
SPEC = importlib.util.spec_from_file_location("convbench_context_report", SOURCE)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class ConvBenchContextReportTests(unittest.TestCase):
    def test_boundary_counts_and_linear_percentile(self):
        lengths = [100, 3072, 3073, 4097]
        rows = [
            {"conversation_id": f"convbench:{i + 1}",
             "reference_history_input_tokens": [x, x, x],
             "input_plus_1024_exceeds_4096": [x + 1024 > 4096] * 3}
            for i, x in enumerate(lengths)
        ]
        summary = REPORT.summarize(rows, 4096, 1024)
        for turn in summary["turns"]:
            self.assertEqual(turn["input_exceeds_context_count"], 1)
            self.assertEqual(turn["input_plus_nominal_cap_exceeds_context_count"], 2)
            self.assertEqual(turn["input_exceeds_context_cases"], [
                {"conversation_id": "convbench:4", "input_tokens": 4097}])
            self.assertEqual(turn["input_tokens"]["p50"], 3072.5)

    def test_rejects_inconsistent_preflight_flag(self):
        row = {"conversation_id": "convbench:1",
               "reference_history_input_tokens": [3073] * 3,
               "input_plus_1024_exceeds_4096": [False] * 3}
        with self.assertRaises(ValueError):
            REPORT.summarize([row], 4096, 1024)


if __name__ == "__main__":
    unittest.main()

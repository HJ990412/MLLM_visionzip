"""Safety checks for analysis selection and dynamic context-budget validation."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from mmimpress.convbench_judge import score_records


SOURCE = Path(__file__).resolve().parents[1] / "scripts/45_analyze_convbench_full.py"
SPEC = importlib.util.spec_from_file_location("convbench_analysis", SOURCE)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


class ConvBenchAnalysisTests(unittest.TestCase):
    def test_stage_only_judge_population_and_fixed_denominator(self):
        dialogs = [{"conversation_id": "convbench:1", "source_row_index": 0}]
        turns = ("_first_turn", "_second_turn", "_third_turn")
        records = [{
            "method": method, "judge_turn": turn,
            "conversation_id": "convbench:1", "source_row_index": 0,
            "position": 0,
            "winner": None if method == "ReComp" and turn == "_second_turn" else "A",
            "model_wins": None if method == "ReComp" and turn == "_second_turn" else True,
            "parse_stage": "unresolved" if method == "ReComp" and turn == "_second_turn"
                           else "first_pass",
        } for method in ANALYSIS.LABELS.values() for turn in turns]
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "judge_raw.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8")
            quality = score_records(records, expected_per_method=1, judge_turns=turns)
            rows, _, counts, fixed = ANALYSIS.judge_analysis(
                run, dialogs, quality, bootstrap_resamples=10, judge_turns=turns)
        self.assertEqual(len(rows), 12)
        self.assertEqual(counts["total_expected_judge_decisions"], 12)
        self.assertEqual(counts["main_expected_judge_decisions"], 12)
        self.assertEqual(counts["main_missing"], 0)
        self.assertEqual(counts["unresolved"], 1)
        self.assertTrue(counts["quality_complete"])
        self.assertEqual(fixed["ReComp"]["S2"], 0)
        self.assertEqual(fixed["ReComp"]["Avg"], 200 / 3)
        self.assertIsNone(fixed["ReComp"]["overall_official_secondary"])
        self.assertEqual([row["score_denominator"] for row in rows], [1] * 12)

    def test_judge_analysis_fixed_denominator_keeps_unresolved(self):
        dialogs = [
            {"conversation_id": "convbench:1", "source_row_index": 0},
            {"conversation_id": "convbench:2", "source_row_index": 1},
        ]
        turns = ("_first_turn", "_second_turn", "_third_turn", "_overall_conversation")
        metrics = ("S1", "S2", "S3", "overall_official_secondary")
        records = []
        quality = {}
        for method in ANALYSIS.LABELS.values():
            valid, unresolved = {}, {}
            scores = {}
            for turn, metric in zip(turns, metrics, strict=True):
                for dialog in dialogs:
                    missing = method == "ReComp" and metric == "S1" and dialog == dialogs[1]
                    records.append({
                        "method": method, "judge_turn": turn,
                        "conversation_id": dialog["conversation_id"],
                        "source_row_index": dialog["source_row_index"],
                        "position": 0,
                        "winner": None if missing else "A",
                        "model_wins": None if missing else True,
                        "parse_stage": "unresolved" if missing else "first_pass",
                    })
                valid[metric] = 1 if method == "ReComp" and metric == "S1" else 2
                unresolved[metric] = 2 - valid[metric]
                scores[metric] = 100.0
            quality[method] = {
                **scores, "Avg": 100.0, "n_conversations": 2,
                "valid_decisions_by_turn": valid,
                "unresolved_by_turn": unresolved,
                "total_valid_decisions": sum(valid.values()),
                "total_unresolved": sum(unresolved.values()),
            }
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "judge_raw.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8")
            rows, bootstrap, counts, fixed = ANALYSIS.judge_analysis(
                run, dialogs, quality, bootstrap_resamples=100)
            self.assertEqual(len(bootstrap), 16)
            self.assertEqual(counts["unresolved"], 1)
            self.assertEqual(counts["first_pass_parsed"], 31)
            self.assertEqual(counts["unresolved_rate"], 1 / 32)
            recomp_s1 = next(row for row in rows if
                             row["method"] == "ReComp" and row["metric"] == "S1")
            self.assertEqual(recomp_s1["valid_decisions"], 1)
            self.assertEqual(recomp_s1["score_denominator"], 2)
            self.assertEqual(recomp_s1["score"], 50.0)
            self.assertEqual(fixed["ReComp"]["Avg"], 250 / 3)
            self.assertFalse(counts["bootstrap_suppressed_for_unresolved"])
            bad = {**quality, "ReComp": {**quality["ReComp"],
                   "total_unresolved": 0}}
            with self.assertRaisesRegex(ValueError, "decision counts"):
                ANALYSIS.judge_analysis(run, dialogs, bad, bootstrap_resamples=100)

    def test_every_targeted_stage_has_seven_conversation_denominator(self):
        dialogs = [{"conversation_id": f"convbench:{i}", "source_row_index": i - 1}
                   for i in range(1, 8)]
        turns = ("_first_turn", "_second_turn", "_third_turn",
                 "_overall_conversation")
        metrics = ("S1", "S2", "S3", "overall_official_secondary")
        records = []
        archived_quality = {}
        for method in ANALYSIS.LABELS.values():
            valid = {}
            unresolved = {}
            for turn, metric in zip(turns, metrics, strict=True):
                for dialog in dialogs:
                    no_label = dialog == dialogs[0] and metric == "S1"
                    records.append({
                        "method": method, "judge_turn": turn,
                        "conversation_id": dialog["conversation_id"],
                        "source_row_index": dialog["source_row_index"],
                        "position": 0, "winner": None if no_label else "A",
                        "model_wins": None if no_label else True,
                        "parse_stage": "unresolved" if no_label else "first_pass",
                    })
                valid[metric] = 6 if metric == "S1" else 7
                unresolved[metric] = 1 if metric == "S1" else 0
            archived_quality[method] = {
                "n_conversations": 7, "S1": 100.0, "S2": 100.0,
                "S3": 100.0, "Avg": 100.0,
                "overall_official_secondary": 100.0,
                "valid_decisions_by_turn": valid,
                "unresolved_by_turn": unresolved,
                "total_valid_decisions": sum(valid.values()),
                "total_unresolved": sum(unresolved.values()),
            }
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "judge_raw.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8")
            rows, bootstrap, counts, fixed = ANALYSIS.judge_analysis(
                run, dialogs, archived_quality, bootstrap_resamples=100)
            self.assertEqual(len(rows), 12)
            self.assertEqual(len(bootstrap), 16)
            self.assertTrue(all(r["score_denominator"] == 7 for r in rows))
            self.assertEqual(counts["main_expected_judge_decisions"], 84)
            self.assertEqual(counts["main_unresolved"], 4)
            self.assertEqual(counts["unresolved"], 4)
            self.assertAlmostEqual(fixed["ReComp"]["S1"], 600 / 7)
            self.assertAlmostEqual(fixed["ReComp"]["Avg"], (600 / 7 + 200) / 3)

    def test_missing_judge_record_is_zero_win_with_fixed_denominator(self):
        dialogs = [{"conversation_id": "convbench:1", "source_row_index": 0}]
        turns = ("_first_turn", "_second_turn", "_third_turn",
                 "_overall_conversation")
        records = [{"method": method, "judge_turn": turn,
                    "conversation_id": "convbench:1", "source_row_index": 0,
                    "position": 0, "winner": "A", "model_wins": True,
                    "parse_stage": "first_pass"}
                   for method in ANALYSIS.LABELS.values() for turn in turns
                   if not (method == "ReComp" and turn == "_first_turn")]
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "judge_raw.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8")
            rows, _, counts, fixed = ANALYSIS.judge_analysis(
                run, dialogs, {}, bootstrap_resamples=10)
            self.assertEqual(counts["missing"], 1)
            self.assertEqual(fixed["ReComp"]["S1"], 0)
            self.assertEqual(next(r for r in rows if r["method"] == "ReComp"
                                  and r["metric"] == "S1")["score_denominator"], 1)

    def test_explicit_smoke_selection_preserves_index_identity(self):
        index = {"conversations": [
            {"conversation_id": "convbench:1", "source_id": "1"},
            {"conversation_id": "convbench:55", "source_id": "55"},
            {"conversation_id": "convbench:494", "source_id": "494"},
        ]}
        config = {"selected_conversation_ids": ["convbench:55", "convbench:494"],
                  "selected_source_ids": ["55", "494"]}
        selected = ANALYSIS.selected_dialogs(index, config, 2)
        self.assertEqual([d["conversation_id"] for d in selected],
                         ["convbench:55", "convbench:494"])
        config["selected_source_ids"] = ["55", "1"]
        with self.assertRaises(ValueError):
            ANALYSIS.selected_dialogs(index, config, 2)

    def test_context_cap_boundary_and_corruption(self):
        row = {
            "method_key": "recompute", "turn_id": 1,
            "expanded_input_tokens": 3222, "context_tokens_for_cache_path": None,
            "text_context_limit": 4096, "context_input_tokens": 3222,
            "nominal_max_new_tokens": 1024, "effective_max_new_tokens": 875,
            "context_remaining_positions_before_request": 874,
            "context_available_output_tokens": 875,
            "generation_cap_clamped_by_context": True,
            "generated_tokens": 875, "generation_cap_reached": True,
        }
        config = {"text_context_limit": 4096, "max_new_tokens": 1024}
        self.assertTrue(ANALYSIS.validate_context_caps([row], config))
        bad = {**row, "generated_tokens": 876, "generation_cap_reached": False}
        with self.assertRaises(ValueError):
            ANALYSIS.validate_context_caps([bad], config)
        with self.assertRaises(ValueError):
            ANALYSIS.validate_context_caps([row, {}], config)

    def test_native_overflow_policy_validates_original_final_and_zero_truncation(self):
        base = {
            "method_key": "recompute", "method": "ReComp", "turn_id": 2,
            "conversation_id": "convbench:494", "expanded_input_tokens": 4137,
            "context_tokens_for_cache_path": None,
            "text_context_limit": 4096, "context_input_tokens": 4137,
            "nominal_max_new_tokens": 1024, "effective_max_new_tokens": 1024,
            "context_remaining_positions_before_request": -41,
            "context_available_output_tokens": -40,
            "generation_cap_clamped_by_context": False,
            "generated_tokens": 10, "generation_cap_reached": False,
            "context_policy_id": ANALYSIS.NATIVE_CONTEXT_POLICY,
            "original_prompt_tokens": 4137, "final_prompt_tokens": 4137,
            "input_overflow_tokens": 41, "native_overflow_execution": True,
            "truncation_applied": False, "truncated_tokens": 0,
            "truncation_source": None, "preserved_current_question": True,
            "preserved_image_marker": True, "runtime_success": True,
            "first_token_success": True,
            "last_generated_sequence_position": 4146,
            "last_expected_executed_position": 4145,
            "generation_position_overflow_tokens": 51,
            "end_to_end_ttft_ms": 12.5,
        }
        config = {"text_context_limit": 4096, "max_new_tokens": 1024,
                  "context_policy_id": ANALYSIS.NATIVE_CONTEXT_POLICY}
        self.assertTrue(ANALYSIS.validate_context_caps([base], config))
        exported = ANALYSIS.context_policy_request_rows(
            [base], ANALYSIS.NATIVE_CONTEXT_POLICY)
        self.assertEqual(exported[0]["input_overflow_tokens"], 41)
        self.assertEqual(exported[0]["final_prompt_tokens"], 4137)
        for field, value in (
            ("final_prompt_tokens", 4096), ("truncated_tokens", 41),
            ("truncation_applied", True), ("native_overflow_execution", False),
            ("generation_position_overflow_tokens", 50),
            ("first_token_success", False),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    ANALYSIS.validate_context_caps([{**base, field: value}], config)

    def test_native_policy_preserves_under_limit_dynamic_cap(self):
        row = {
            "method_key": "recompute", "turn_id": 1,
            "expanded_input_tokens": 3222, "context_tokens_for_cache_path": None,
            "text_context_limit": 4096, "context_input_tokens": 3222,
            "nominal_max_new_tokens": 1024, "effective_max_new_tokens": 875,
            "context_remaining_positions_before_request": 874,
            "context_available_output_tokens": 875,
            "generation_cap_clamped_by_context": True,
            "generated_tokens": 1, "generation_cap_reached": False,
            "context_policy_id": ANALYSIS.NATIVE_CONTEXT_POLICY,
            "original_prompt_tokens": 3222, "final_prompt_tokens": 3222,
            "input_overflow_tokens": 0, "native_overflow_execution": False,
            "truncation_applied": False, "truncated_tokens": 0,
            "truncation_source": None, "preserved_current_question": True,
            "preserved_image_marker": True, "runtime_success": True,
            "first_token_success": True,
            "last_generated_sequence_position": 3222,
            "last_expected_executed_position": 3221,
            "generation_position_overflow_tokens": 0,
        }
        config = {"text_context_limit": 4096, "max_new_tokens": 1024,
                  "context_policy_id": ANALYSIS.NATIVE_CONTEXT_POLICY}
        self.assertTrue(ANALYSIS.validate_context_caps([row], config))
        with self.assertRaises(ValueError):
            ANALYSIS.validate_context_caps([{**row, "effective_max_new_tokens": 1024}], config)


if __name__ == "__main__":
    unittest.main()

"""Protocol checks for the local judge and official ConvBench artifacts."""

import json
import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from mmimpress.convbench_judge import (
    TURN_NAMES, build_messages, extract_fallback_winner,
    extract_official_winner, load_official_prompt_builder,
    load_pairwise, normalize_row, score_records,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/convbench_source"
INDEX = ROOT / "data/convbench/index.json"


class JudgeSemanticsTest(unittest.TestCase):
    @staticmethod
    def judge_cli_module():
        spec = importlib.util.spec_from_file_location(
            "convbench_judge_cli", ROOT / "scripts/42_judge_convbench.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_parser_and_unresolved_fixed_denominator(self):
        self.assertEqual(extract_official_winner("Overall, Response B is better."), "B")
        self.assertIsNone(extract_official_winner("Response A is better, but response B is better."))
        self.assertEqual(extract_fallback_winner("A"), "A")
        self.assertEqual(extract_fallback_winner("Final Answer: B"), "B")
        self.assertIsNone(extract_fallback_winner("UNKNOWN"))
        self.assertIsNone(extract_fallback_winner("A or B"))
        records = [
            {"method": "ReComp", "judge_turn": turn, "conversation_id": "c1",
             "source_row_index": 0, "position": 0,
             "winner": None if turn == TURN_NAMES[1] else "A"}
            for turn in TURN_NAMES
        ]
        score = score_records(records, expected_per_method=1)["ReComp"]
        self.assertEqual((score["S1"], score["S2"], score["S3"], score["Avg"]),
                         (100.0, 0.0, 100.0, 200.0 / 3))
        self.assertEqual(score["denominators_by_turn"]["S2"], 1)
        self.assertEqual(score["total_unresolved"], 1)

    def test_fixed_seven_denominator_with_unresolved_and_missing_judgments(self):
        records = []
        for index in range(7):
            for turn in TURN_NAMES:
                if turn == "_third_turn" and index == 6:
                    continue  # Missing judgment still occupies its denominator slot.
                winner = {
                    "_first_turn": "A" if index < 4 else "B",
                    "_second_turn": None if index == 0 else ("A" if index < 4 else "B"),
                    "_third_turn": "C",  # Archived unresolved representation.
                    "_overall_conversation": "A",
                }[turn]
                records.append({
                    "method": "ReComp", "judge_turn": turn,
                    "conversation_id": f"c{index}", "source_row_index": index,
                    "position": 0, "winner": winner,
                })
        score = score_records(records, expected_per_method=7)["ReComp"]
        self.assertEqual([score["denominators_by_turn"][stage] for stage in ("S1", "S2", "S3")],
                         [7, 7, 7])
        self.assertAlmostEqual(score["S1"], 400 / 7)
        self.assertAlmostEqual(score["S2"], 300 / 7)
        self.assertEqual(score["S3"], 0)
        self.assertAlmostEqual(score["Avg"], 100 / 3)
        self.assertEqual(score["overall_official_secondary"], 100)
        self.assertEqual(score["unresolved_by_turn"]["S2"], 1)
        self.assertEqual(score["unresolved_by_turn"]["S3"], 6)
        self.assertEqual(score["missing_by_turn"]["S3"], 1)

    def test_stage_only_scoring_has_no_overall_population(self):
        records = [{
            "method": "ReComp", "judge_turn": turn,
            "conversation_id": f"c{index}", "source_row_index": index,
            "position": 0, "winner": None if index == 0 else "A",
        } for index in range(7) for turn in TURN_NAMES[:3]]
        score = score_records(records, expected_per_method=7,
                              judge_turns=TURN_NAMES[:3])["ReComp"]
        self.assertEqual([score["denominators_by_turn"][stage] for stage in ("S1", "S2", "S3")],
                         [7, 7, 7])
        self.assertEqual((score["S1"], score["S2"], score["S3"]),
                         (600 / 7, 600 / 7, 600 / 7))
        self.assertEqual(score["Avg"], 600 / 7)
        self.assertEqual(score["total_unresolved"], 3)
        self.assertEqual(score["unresolved_rate"], 3 / 21)
        self.assertEqual(score["total_missing"], 0)
        self.assertIsNone(score["overall_official_secondary"])
        self.assertNotIn("overall_official_secondary", score["denominators_by_turn"])

    @unittest.skipUnless(SOURCE.exists() and INDEX.exists(), "official dataset artifacts absent")
    def test_stage_only_cli_three_records_resume_and_mode_mismatch(self):
        module = self.judge_cli_module()
        first = json.loads(INDEX.read_text(encoding="utf-8"))["conversations"][0]

        class FakeJudge:
            calls = 0

            def __init__(self, *_args, **_kwargs):
                pass

            def generate(self, _messages):
                FakeJudge.calls += 1
                return "Overall, Response A is better.", 100, 9

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            answers = root / "answers.jsonl"
            with answers.open("w", encoding="utf-8") as handle:
                for method in ("ReComp", "FullLoad", "Prefix25", "Prefix45"):
                    for turn in (1, 2, 3):
                        handle.write(json.dumps({
                            "conversation_id": first["conversation_id"],
                            "method": method, "turn_id": turn,
                            "prediction": f"generated {method}/{turn}",
                        }) + "\n")
            out = root / "judge"
            argv = ["42_judge_convbench.py", "--source", str(SOURCE), "--index", str(INDEX),
                    "--answers", str(answers), "--out", str(out), "--limit", "1",
                    "--stage-only", "--execute"]
            with patch.object(module, "LocalLlamaJudge", FakeJudge), redirect_stdout(io.StringIO()):
                with patch.object(sys, "argv", argv):
                    module.main()
                    self.assertEqual(FakeJudge.calls, 12)
                    module.main()
                    self.assertEqual(FakeJudge.calls, 12)
                config = json.loads((out / "config.json").read_text())
                self.assertEqual(config["judge_turns"], list(TURN_NAMES[:3]))
                checkpoint = out / "judgments" / "ReComp" / "0000.json"
                payload = json.loads(checkpoint.read_text())
                self.assertEqual(payload["status"], "complete")
                self.assertEqual([row["judge_turn"] for row in payload["records"]],
                                 list(TURN_NAMES[:3]))
                module.atomic_checkpoint(checkpoint, {
                    "records": payload["records"][:2], "status": "partial",
                })
                with patch.object(sys, "argv", argv):
                    module.main()
                self.assertEqual(FakeJudge.calls, 13)
                self.assertEqual(len(json.loads(checkpoint.read_text())["records"]), 3)
                raw = [json.loads(line) for line in (out / "judge_raw.jsonl").read_text().splitlines()]
                self.assertEqual(len(raw), 12)
                self.assertEqual(json.loads((out / "score_provenance.json").read_text())
                                 ["total_expected_judge_decisions"], 12)
                with patch.object(sys, "argv", [arg for arg in argv if arg != "--stage-only"]):
                    with self.assertRaisesRegex(ValueError, "resume refused"):
                        module.main()

    def test_second_pass_recovery_preserves_archived_first_pass(self):
        module = self.judge_cli_module()
        old = {
            "conversation_id": "convbench:3", "source_row_index": 2,
            "method": "ReComp", "judge_turn": "_first_turn", "position": 1,
            "messages": [{"role": "user", "content": "official pairwise prompt"}],
            "raw_response": "Assistant B makes the stronger case.",
            "winner": "C", "parse_stage": "unparseable_C",
            "model_wins": False, "extraction_raw_response": "Unknown",
        }

        class FakeJudge:
            calls = 0

            def generate(self, messages, *, max_new_tokens):
                FakeJudge.calls += 1
                self_messages = messages
                assert "Assistant B makes the stronger case." in self_messages[-1]["content"]
                assert max_new_tokens == 16
                return "B", 120, 1

        recovered = module.recover_record(old, FakeJudge())
        self.assertEqual(FakeJudge.calls, 1)
        self.assertEqual(recovered["raw_response"], old["raw_response"])
        self.assertEqual(recovered["messages"], old["messages"])
        self.assertEqual(recovered["winner"], "B")
        self.assertTrue(recovered["model_wins"])
        self.assertEqual(recovered["parse_stage"], "second_pass")
        self.assertEqual(recovered["archived_extraction_raw_response"], "Unknown")

        class UnknownJudge(FakeJudge):
            def generate(self, messages, *, max_new_tokens):
                return "UNKNOWN", 120, 1

        unresolved = module.recover_record(old, UnknownJudge())
        self.assertIsNone(unresolved["winner"])
        self.assertIsNone(unresolved["model_wins"])
        self.assertEqual(unresolved["parse_stage"], "unresolved")

    @unittest.skipUnless(SOURCE.exists() and INDEX.exists(), "official dataset artifacts absent")
    def test_original_row_pairwise_mapping_and_official_prompt(self):
        rows = json.loads(INDEX.read_text(encoding="utf-8"))["conversations"]
        self.assertEqual(len(rows), 577)
        indices = [int(row["source_row_index"]) for row in rows]
        self.assertEqual(set(range(578)) - set(indices), {130})
        positions = load_pairwise(SOURCE, indices)
        builder, _ = load_official_prompt_builder(SOURCE)
        row = normalize_row(rows[0])
        messages = build_messages(builder, row, ["pred A1", "pred A2", "pred A3"],
                                  positions[row["source_row_index"]], "_third_turn")
        self.assertEqual(len(messages), 4)
        self.assertIn(row["third_turn_demands"], messages[-1]["content"])
        self.assertIn("pred A3", messages[-1]["content"])

    @unittest.skipUnless(SOURCE.exists() and INDEX.exists(), "official dataset artifacts absent")
    def test_smoke_ids_select_exact_official_rows_and_pairwise_positions(self):
        module = self.judge_cli_module()
        rows = module.read_index(INDEX)
        selected = module.select_rows(rows, 5, False, "1,2,3,4,5,494,418")
        self.assertEqual([row["conversation_id"] for row in selected],
                         [f"convbench:{source_id}" for source_id in (1, 2, 3, 4, 5, 494, 418)])
        source_indices = [row["source_row_index"] for row in selected]
        self.assertEqual(source_indices, [0, 1, 2, 3, 4, 493, 417])
        positions = load_pairwise(SOURCE, source_indices)
        self.assertEqual(set(positions), set(source_indices))
        self.assertEqual(module.select_rows(rows, 5, False, None), rows[:5])
        self.assertEqual(module.select_rows(rows, 5, True, None), rows)

        for bad in ("", "1,1", "0", "01", "1,131", "1,,2", "a", ",".join(map(str, range(1, 12)))):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                module.select_rows(rows, 5, False, bad)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            module.select_rows(rows, 5, True, "1")

    @unittest.skipUnless(SOURCE.exists() and INDEX.exists(), "official dataset artifacts absent")
    def test_smoke_ids_cli_preflight_and_resume_selection_guard(self):
        module = self.judge_cli_module()
        selected_ids = (1, 494)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            answers = root / "answers.jsonl"
            with answers.open("w", encoding="utf-8") as handle:
                for source_id in selected_ids:
                    for method in ("ReComp", "FullLoad", "Prefix25", "Prefix45"):
                        for turn in (1, 2, 3):
                            handle.write(json.dumps({
                                "conversation_id": f"convbench:{source_id}",
                                "method": method, "turn_id": turn,
                                "prediction": f"generated {source_id}/{method}/{turn}",
                            }) + "\n")
            out = root / "judge"
            argv = ["42_judge_convbench.py", "--source", str(SOURCE), "--index", str(INDEX),
                    "--answers", str(answers), "--out", str(out), "--smoke-ids", "1,494"]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()) as captured:
                module.main()
            self.assertEqual(json.loads(captured.getvalue())["n_conversations"], 2)
            self.assertFalse(out.exists())

            class FakeJudge:
                def __init__(self, *_args, **_kwargs):
                    pass

                def generate(self, _messages):
                    return "Overall, Response A is better.", 100, 9

            with patch.object(module, "LocalLlamaJudge", FakeJudge), redirect_stdout(io.StringIO()):
                with patch.object(sys, "argv", [*argv, "--execute"]):
                    module.main()
                config = json.loads((out / "config.json").read_text())
                self.assertEqual(config["selected_source_row_indices"], [0, 493])
                with patch.object(sys, "argv", [*argv[:-1], "494,1", "--execute"]):
                    with self.assertRaisesRegex(ValueError, "resume refused"):
                        module.main()

    @unittest.skipUnless(SOURCE.exists() and INDEX.exists(), "official dataset artifacts absent")
    def test_cli_resume_and_corrupt_checkpoint_refusal(self):
        module = self.judge_cli_module()
        first = json.loads(INDEX.read_text(encoding="utf-8"))["conversations"][0]

        class FakeJudge:
            calls = 0

            def __init__(self, *_args, **_kwargs):
                pass

            def generate(self, _messages):
                FakeJudge.calls += 1
                return "Overall, Response A is better.", 100, 9

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            answers = root / "answers.jsonl"
            with answers.open("w", encoding="utf-8") as handle:
                for method in ("ReComp", "FullLoad", "Prefix25", "Prefix45"):
                    for turn in (1, 2, 3):
                        handle.write(json.dumps({
                            "conversation_id": first["conversation_id"],
                            "method": method, "turn_id": turn,
                            "prediction": f"generated {method}/{turn}",
                        }) + "\n")
            out = root / "judge"
            argv = ["42_judge_convbench.py", "--source", str(SOURCE), "--index", str(INDEX),
                    "--answers", str(answers), "--out", str(out), "--limit", "1", "--execute"]
            with patch.object(module, "LocalLlamaJudge", FakeJudge), patch.object(sys, "argv", argv):
                module.main()
                self.assertEqual(FakeJudge.calls, 16)
                module.main()
                self.assertEqual(FakeJudge.calls, 16)
                quality = json.loads((out / "quality_summary.json").read_text())
                self.assertEqual(len(quality), 4)
                checkpoint = out / "judgments" / "ReComp" / "0000.json"
                payload = json.loads(checkpoint.read_text())
                payload["records"][0]["raw_response"] = "tampered"
                checkpoint.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, "checkpoint hash"):
                    module.main()


if __name__ == "__main__":
    unittest.main()

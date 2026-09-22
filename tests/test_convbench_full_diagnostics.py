"""Coverage and paired analysis checks for frozen ConvBench artifacts."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/48_analyze_convbench_full_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("convbench_full_diagnostics", SOURCE)
assert SPEC is not None and SPEC.loader is not None
DIAG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAG)


class FullDiagnosticsTests(unittest.TestCase):
    def test_stage_only_checkpoint_hash_and_raw_equality(self):
        judge_runner = DIAG.load_script("convbench_judge_checkpoint_test", "42_judge_convbench.py")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            index = base / "index.json"
            index.write_text('{}', encoding="utf-8")
            answer = base / "answers"
            answer.mkdir()
            judge = base / "judge"
            judge.mkdir()
            dialog = {"conversation_id": "convbench:1", "source_row_index": 0}
            config = {"index_sha256": DIAG.file_hash(index),
                      "answer_sha256": "synthetic-answer-hash",
                      "selected_source_row_indices": [0],
                      "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                      "second_pass_model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                      "commercial_api_used": False,
                      "decoding": "greedy; temperature=0; do_sample=False",
                      "judge_turns": list(DIAG.STAGES)}
            (judge / "config.json").write_text(json.dumps(config), encoding="utf-8")
            all_records = []
            for method in DIAG.LABELS.values():
                records = [{"method": method, "conversation_id": "convbench:1",
                            "source_row_index": 0, "judge_turn": turn, "position": 0,
                            "winner": "A", "model_wins": True,
                            "parse_stage": "first_pass"}
                           for turn in DIAG.STAGES]
                all_records.extend(records)
                path = judge / "judgments" / method / "0000.json"
                path.parent.mkdir(parents=True)
                payload = {"records": records, "status": "complete"}
                payload["artifact_content_sha256"] = judge_runner.stable_content_hash(payload)
                path.write_text(json.dumps(payload), encoding="utf-8")
            raw = judge / "judge_raw.jsonl"
            raw.write_text("".join(json.dumps(record) + "\n" for record in all_records),
                           encoding="utf-8")
            stub = SimpleNamespace(_judge_answer_input_hash=lambda *_: "synthetic-answer-hash")
            _, coverage, _ = DIAG.validate_judge(judge, answer, index, [dialog], stub)
            self.assertEqual(coverage["judge_checkpoints_verified"], 4)
            changed = [dict(record) for record in all_records]
            changed[0]["raw_response"] = "different but parseable response"
            raw.write_text("".join(json.dumps(record) + "\n" for record in changed),
                           encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checkpoint/raw content mismatch"):
                DIAG.validate_judge(judge, answer, index, [dialog], stub)
            raw.write_text("".join(json.dumps(record) + "\n" for record in all_records),
                           encoding="utf-8")
            checkpoint = judge / "judgments" / "ReComp" / "0000.json"
            payload = json.loads(checkpoint.read_text())
            payload["records"][0]["raw_response"] = "tamper without new hash"
            checkpoint.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Corrupt judge checkpoint hash"):
                DIAG.validate_judge(judge, answer, index, [dialog], stub)

    def test_fixed_denominator_and_unresolved_zero_win(self):
        dialogs = [{"conversation_id": "convbench:1"},
                   {"conversation_id": "convbench:2"}]
        keyed = {}
        for method in DIAG.LABELS.values():
            for cid in ("convbench:1", "convbench:2"):
                for stage in DIAG.STAGES:
                    unresolved = method == "Prefix25" and cid == "convbench:2"
                    keyed[(method, cid, stage)] = {
                        "winner": None if unresolved else "A",
                        "position": 0,
                        "parse_stage": "unresolved" if unresolved else "first_pass",
                    }
        quality, judge, vectors = DIAG.scores_and_judge(keyed, dialogs)
        prefix = next(row for row in quality if row["method"] == "Prefix25")
        self.assertEqual(prefix["fixed_stage_denominator"], 2)
        self.assertEqual(prefix["S1"], 50)
        self.assertEqual(prefix["S2"], 50)
        self.assertEqual(prefix["S3"], 50)
        self.assertEqual(prefix["Avg"], 50)
        self.assertEqual(prefix["unresolved"], 3)
        self.assertEqual(prefix["delta_Avg_vs_ReComp"], -50)
        self.assertTrue(all(record["denominator"] == 2 for record in judge))
        self.assertEqual(vectors["Prefix25"].shape, (2, 3))

    def test_bootstrap_resamples_complete_conversations(self):
        ids = ["convbench:1", "convbench:2"]
        rows = []
        vectors = {}
        for method in DIAG.METHODS:
            label = DIAG.LABELS[method]
            vectors[label] = np.array([[1, 1, 1], [0, 0, 0]], dtype=float)
            for cid in ids:
                for turn in (1, 2, 3):
                    rows.append({"conversation_id": cid, "method_key": method,
                                 "turn_id": turn, "end_to_end_ttft_ms":
                                 10 if method == "recompute" else 5})
        first = DIAG.paired_bootstrap(rows, ids, vectors, resamples=100, seed=1234)
        second = DIAG.paired_bootstrap(rows, ids, vectors, resamples=100, seed=1234)
        self.assertEqual(first, second)
        prefix_latency = next(row for row in first if row["method"] == "Prefix25" and
                              row["metric"] == "cache_hit_TTFT_reduction_vs_ReComp_pct")
        self.assertEqual(prefix_latency["estimate"], 50)
        self.assertEqual((prefix_latency["ci95_low"], prefix_latency["ci95_high"]),
                         (50, 50))
        self.assertTrue(all(row["cluster_unit"] == "conversation" for row in first))

    def test_targeted_artifact_recalculation_and_stage_only_coverage(self):
        answer = ROOT / "runs/convbench_context/targeted_7_native"
        judge = ROOT / "runs/convbench_context/judge_targeted_7_llama_second_pass"
        if not (answer / "config.json").is_file() or not (judge / "judge_raw.jsonl").is_file():
            self.skipTest("archived targeted artifacts unavailable")
        with self.assertRaisesRegex(ValueError, "exactly 577"):
            DIAG.analyze(ROOT / "data/convbench/index.json", answer, judge, 7,
                         full=True, bootstrap_resamples=10)
        result = DIAG.analyze(ROOT / "data/convbench/index.json", answer, judge, 7,
                              bootstrap_resamples=20)
        validation = result["validation"]
        self.assertEqual(validation["completed_generation_requests"], 84)
        self.assertEqual(validation["completed_stage_judgments"], 84)
        self.assertEqual(validation["overall_diagnostic_judgments"], 28)
        self.assertEqual(validation["unresolved"], 10)
        self.assertEqual(validation["turn1_generation_agreement"], 7)
        self.assertEqual(validation["degenerate_repetition_requests"], 8)
        self.assertEqual(validation["above_nominal_4096_requests"], 12)
        self.assertTrue(all(row["denominator"] == 7 for row in result["judge"]))
        self.assertAlmostEqual(next(row for row in result["quality"] if
                                    row["method"] == "ReComp")["Avg"], 500 / 21)
        with tempfile.TemporaryDirectory() as tmp:
            stage_run = Path(tmp)
            stage_config = json.loads((judge / "config.json").read_text())
            stage_config["judge_turns"] = list(DIAG.STAGES)
            (stage_run / "config.json").write_text(json.dumps(stage_config), encoding="utf-8")
            records = [json.loads(line) for line in
                       (judge / "judge_raw.jsonl").read_text(encoding="utf-8").splitlines()
                       if line.strip()]
            stage_records = [record for record in records if record["judge_turn"] in DIAG.STAGES]
            (stage_run / "judge_raw.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in stage_records),
                encoding="utf-8")
            analysis = DIAG.load_script("convbench_analysis_test", "45_analyze_convbench_full.py")
            index = ROOT / "data/convbench/index.json"
            config = json.loads((answer / "config.json").read_text())
            dialogs = analysis.selected_dialogs(json.loads(index.read_text()), config, 7)
            _, coverage, _ = DIAG.validate_judge(
                stage_run, answer, index, dialogs, analysis, verify_checkpoints=False)
            self.assertEqual(coverage["stage_completed"], 84)
            self.assertEqual(coverage["overall_diagnostic_count"], 0)
            (stage_run / "judge_raw.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in stage_records[:-1]),
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing 1 of 84"):
                DIAG.validate_judge(stage_run, answer, index, dialogs, analysis,
                                    verify_checkpoints=False)


if __name__ == "__main__":
    unittest.main()

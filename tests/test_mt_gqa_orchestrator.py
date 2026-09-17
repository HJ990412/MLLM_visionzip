"""CPU-only contract tests for the final MT-GQA orchestrator."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/39_run_mt_gqa_full.py"
SPEC = importlib.util.spec_from_file_location("mt_gqa_orchestrator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ORCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORCH)


class FrozenCliContractTests(unittest.TestCase):
    def test_exact_full_request_arithmetic_and_partial_flags(self):
        self.assertEqual(ORCH.EXPECTED_REQUESTS, 48_732)
        contract = {
            "index_path": Path("/tmp/dialogues.json"),
            "index_sha256": "a" * 64,
            "workload_sha256": "b" * 64,
        }
        common = dict(
            evaluator=Path("/tmp/evaluator.py"), stage=Path("/tmp/stage"),
            temporary=Path("/tmp/temp"), experiment_id="experiment",
            contract=contract, shard_index=0, shard_size=50,
            min_free_after_gib=30.0,
        )
        smoke = ORCH.build_evaluator_command(dialogue_limit=10, **common)
        pilot = ORCH.build_evaluator_command(dialogue_limit=100, **common)
        full = ORCH.build_evaluator_command(
            dialogue_limit=ORCH.EXPECTED_DIALOGUES, **common)
        for command, count in ((smoke, "10"), (pilot, "100")):
            self.assertEqual(command[command.index("--expected-dialogs") + 1],
                             count)
            self.assertIn("--allow-partial-workload", command)
            self.assertEqual(command[command.index("--max-dialogs") + 1], count)
        self.assertNotIn("--allow-partial-workload", full)
        self.assertNotIn("--max-dialogs", full)
        self.assertEqual(full[full.index("--expected-dialogs") + 1], "4061")

    def test_resume_reuses_frozen_shard_size(self):
        frozen = {"shard_size": 50}
        # Current free space would choose 40, but resume must retain 50.
        selected = ORCH.choose_shard_size(
            requested=None, free_bytes=40 * 1024 ** 3,
            reserve_gib=30.0, frozen_experiment=frozen)
        self.assertEqual(selected, 50)
        with self.assertRaisesRegex(ValueError, "resume shard_size mismatch"):
            ORCH.choose_shard_size(
                requested=60, free_bytes=200 * 1024 ** 3,
                reserve_gib=30.0, frozen_experiment=frozen)

    def test_fresh_stage_log_is_outside_evaluator_owned_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory).resolve()
            contract = {
                "dialogues": [{"image_id": "image-1"}] * 10,
                "index_path": Path("/tmp/dialogues.json"),
                "index_sha256": "a" * 64,
                "workload_sha256": "b" * 64,
            }
            with mock.patch.object(
                    ORCH, "run_streaming", return_value=1.0) as streaming, \
                 mock.patch.object(
                    ORCH, "validate_stage_completion",
                    return_value={"passed": True}), \
                 mock.patch.object(
                    ORCH, "cleanup_stage_temp_root",
                    return_value={"passed": True}):
                ORCH.run_inference_stage(
                    name="smoke_10", dialogue_limit=10,
                    run_root=run_root, experiment_id="experiment",
                    contract=contract, shard_size=60,
                    min_free_after_gib=30.0, resume=False)
            log_path = streaming.call_args.args[1]
            self.assertEqual(log_path, run_root / "smoke_10.log")
            self.assertNotEqual(log_path.parent, run_root / "smoke_10")


class ProjectionTests(unittest.TestCase):
    def test_projection_reads_finalized_persistence_overhead_key(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "images").mkdir()
            (run / "shards").mkdir()
            (run / "images/a.json").write_text(json.dumps({
                "persistence_overhead": {
                    "persist_ms": 2_000.0,
                    "total_ssd_write_bytes": 123_456,
                }
            }))
            (run / "shards/shard_000.json").write_text(json.dumps({
                "elapsed_seconds": 20.0,
            }))
            pilot = {
                "dialogues": 100,
                "unique_images": 10,
                "planned_method_turn_requests": 1_200,
                "wall_seconds_this_invocation": 0.0,
            }
            result = ORCH.projection_from_pilot(pilot, run, 398)
            self.assertEqual(result["mean_store_persist_seconds_per_image"], 2.0)
            self.assertEqual(result["peak_temporary_store_bytes"], 123_456)
            self.assertEqual(result["full_planned_method_turn_requests"], 48_732)


class TempOwnershipTests(unittest.TestCase):
    def _make_owned(self, run: Path, stage: str, experiment: str) -> Path:
        temporary = run / "_temporary_visual_kv" / stage
        (temporary / "payload").mkdir(parents=True)
        (temporary / ORCH.EVALUATOR_OWNER_FILE).write_text(json.dumps({
            "schema_version": "runner-schema",
            "experiment_id": experiment,
            "dataset": ORCH.EVALUATOR_DATASET,
            "purpose": "temporary_visual_kv_only",
        }))
        return temporary

    def test_exact_owned_empty_root_is_removed_without_recursive_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory).resolve()
            temporary = self._make_owned(run, "smoke_10", "exp")
            report = ORCH.cleanup_stage_temp_root(
                temporary=temporary, run_root=run,
                stage_name="smoke_10", experiment_id="exp")
            self.assertTrue(report["passed"])
            self.assertFalse(temporary.exists())
            self.assertFalse((run / "_temporary_visual_kv").exists())

    def test_nonempty_or_wrong_owner_fails_and_preserves_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory).resolve()
            temporary = self._make_owned(run, "full_4061", "exp")
            payload_file = temporary / "payload/keep-me"
            payload_file.write_text("evidence")
            with self.assertRaisesRegex(RuntimeError, "payload leak"):
                ORCH.cleanup_stage_temp_root(
                    temporary=temporary, run_root=run,
                    stage_name="full_4061", experiment_id="exp")
            self.assertEqual(payload_file.read_text(), "evidence")

            payload_file.unlink()
            with self.assertRaisesRegex(ValueError, "ownership mismatch"):
                ORCH.cleanup_stage_temp_root(
                    temporary=temporary, run_root=run,
                    stage_name="full_4061", experiment_id="other")
            self.assertTrue(temporary.exists())


class StageCoverageTests(unittest.TestCase):
    def _stage(self, root: Path, *, missing_last_row: bool = False) -> tuple:
        stage = root / "full_4061"
        (stage / "images").mkdir(parents=True)
        (stage / "shards").mkdir()
        experiment = "experiment"
        contract = {
            "index_sha256": "1" * 64,
            "workload_sha256": "2" * 64,
        }
        config = {
            "experiment_id": experiment,
            "dialogues_file_sha256": contract["index_sha256"],
            "source_full_workload_sha256": contract["workload_sha256"],
            "n_dialogs": 1, "n_turns": 3, "n_images": 1,
            "n_requests": 12, "shard_size": 40, "n_shards": 1,
            "seed": 1234, "method_keys": list(ORCH.METHOD_KEYS),
        }
        (stage / "config.json").write_text(json.dumps(config))
        rows = [
            {"dialog_id": "d1", "turn_id": turn, "method_key": method}
            for turn in (1, 2, 3) for method in ORCH.METHOD_KEYS
        ]
        if missing_last_row:
            rows.pop()
        artifact = {
            "experiment_id": experiment,
            "image_id": "image1",
            "dialog_ids": ["d1"],
            "n_dialogs": 1, "n_turns": 3, "n_rows": len(rows),
            "rows": rows,
            "validation": {"passed": True},
        }
        (stage / "images/image1.json").write_text(json.dumps(artifact))
        marker = {
            "complete": True, "experiment_id": experiment,
            "shard_index": 0, "shard_size": 40,
            "image_ids": ["image1"],
        }
        (stage / "shards/shard_000.json").write_text(json.dumps(marker))
        return stage, experiment, contract

    def test_exact_dialogue_turn_method_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            stage, experiment, contract = self._stage(Path(directory))
            report = ORCH.validate_stage_completion(
                stage=stage, experiment_id=experiment, contract=contract,
                expected_dialogues=1, expected_images=1, shard_size=40,
                expected_shards=1)
            self.assertEqual(report["observed_method_turn_requests"], 12)
            self.assertTrue(report["exact_request_coverage"])

    def test_incomplete_image_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            stage, experiment, contract = self._stage(
                Path(directory), missing_last_row=True)
            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                ORCH.validate_stage_completion(
                    stage=stage, experiment_id=experiment, contract=contract,
                    expected_dialogues=1, expected_images=1, shard_size=40,
                    expected_shards=1)


class PreservationTests(unittest.TestCase):
    def test_snapshot_covers_files_empty_dirs_and_symlink_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_root = Path(directory).resolve()
            old = fake_root / "runs/old"
            (old / "empty").mkdir(parents=True)
            (fake_root / "results").mkdir()
            file_path = old / "result.json"
            file_path.write_text("original")
            (old / "link").symlink_to("result.json")
            run_root = fake_root / "runs/mt_gqa_full"
            result_root = fake_root / "results/mt_gqa_full"
            with mock.patch.object(ORCH, "ROOT", fake_root):
                before = ORCH.protected_snapshot(run_root, result_root)
                self.assertEqual(
                    before["entries"]["runs/old/link"]["type"], "symlink")
                self.assertIn("runs/old/empty", before["entries"])
                file_path.write_text("changed")
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    ORCH.verify_protected_snapshot(
                        before, run_root, result_root)


if __name__ == "__main__":
    unittest.main()

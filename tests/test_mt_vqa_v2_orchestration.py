"""Focused CPU-only contracts for MT-VQA-v2 orchestration and launch."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/62_run_mt_vqa_v2_generated.py"
SPEC = importlib.util.spec_from_file_location(
    "mt_vqa_v2_orchestration_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ORCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORCH)


class OrchestratorTests(unittest.TestCase):
    def test_exact_single_protocol_request_accounting(self):
        self.assertEqual(ORCH.METHOD_KEYS,
                         ("recompute", "fullload", "qa_chunk25", "ours25"))
        self.assertEqual(ORCH.PROTOCOL, "generated_history")
        self.assertEqual(ORCH.EXPECTED_DIALOGUES, 250)
        self.assertEqual(ORCH.EXPECTED_TURNS, 750)
        self.assertEqual(ORCH.FULL_REQUESTS, 3000)
        self.assertEqual(ORCH.SMOKE_REQUESTS, 120)
        self.assertEqual(ORCH.PHYSICAL_REQUESTS_WITH_SMOKE, 3120)

    def test_evaluator_commands_are_generated_only_and_smoke_is_explicit(self):
        root = ROOT / "runs/mt_vqa_v2_generated_4arm_fixture"
        smoke = ORCH.evaluator_command(
            root / "smoke/generated_history",
            root / "_temporary_visual_kv/smoke/generated_history",
            "experiment", 0, 50, ORCH.SMOKE_DIALOGUES, 30.0,
            ORCH.DEFAULT_INDEX)
        rendered = " ".join(smoke)
        self.assertIn("60_eval_mt_vqa_v2_generated_shard.py", rendered)
        self.assertIn("--protocol generated_history", rendered)
        self.assertIn("--max-dialogs 10 --allow-partial-workload", rendered)
        self.assertNotIn("gold_history", rendered)
        full = ORCH.evaluator_command(
            root / "full/generated_history",
            root / "_temporary_visual_kv/full/generated_history",
            "experiment", 4, 50, ORCH.EXPECTED_DIALOGUES, 30.0,
            ORCH.DEFAULT_INDEX)
        self.assertNotIn("--max-dialogs", full)
        self.assertEqual(full[full.index("--expected-dialogs") + 1], "250")

    def test_analyzer_command_binds_full_contract_and_protection(self):
        run = ROOT / "runs/mt_vqa_v2_generated_4arm_fixture"
        results = ROOT / "results/mt_vqa_v2_generated_4arm_fixture"
        protection = run / "protected_artifacts_validation.json"
        command = ORCH.analyzer_command(
            run / "full/generated_history", results,
            ORCH.DEFAULT_INDEX, protection)
        rendered = " ".join(command)
        self.assertIn("61_analyze_mt_vqa_v2_generated.py", rendered)
        self.assertIn("--expected-dialogs 250", rendered)
        self.assertIn("--expected-images 250", rendered)
        self.assertIn("--bootstrap-resamples 10000", rendered)
        self.assertIn("--bootstrap-seed 1234", rendered)
        self.assertIn(str(protection), rendered)

    def test_output_roots_must_be_matching_dedicated_siblings(self):
        good_run = ROOT / "runs/mt_vqa_v2_generated_4arm_fixture"
        good_results = ROOT / "results/mt_vqa_v2_generated_4arm_fixture"
        run, results = ORCH.validate_roots(good_run, good_results)
        self.assertEqual(run, good_run.resolve())
        self.assertEqual(results, good_results.resolve())
        with self.assertRaises(ValueError):
            ORCH.validate_roots(
                ROOT / "runs/wrong", ROOT / "results/wrong")
        with self.assertRaises(ValueError):
            ORCH.validate_roots(
                good_run,
                ROOT / "results/mt_vqa_v2_generated_4arm_other")

    def test_completion_marker_deeply_binds_each_image_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            (stage / "images").mkdir()
            (stage / "shards").mkdir()
            artifact = {
                "experiment_id": "exp",
                "protocol": ORCH.PROTOCOL,
                "image_id": "image_a",
                "shard_index": 0,
                "rows": [],
            }
            artifact["artifact_content_sha256"] = ORCH.canonical_hash(artifact)
            artifact_path = stage / "images/image_a.json"
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            marker = {
                "schema_version": ORCH.SHARD_SCHEMA,
                "complete": True,
                "experiment_id": "exp",
                "protocol": ORCH.PROTOCOL,
                "shard_index": 0,
                "shard_size": 50,
                "image_ids": ["image_a"],
                "completed_image_ids": ["image_a"],
                "image_artifact_file_sha256": {
                    "image_a": ORCH.sha256_file(artifact_path)},
                "image_artifact_content_sha256": {
                    "image_a": artifact["artifact_content_sha256"]},
            }
            marker["artifact_content_sha256"] = ORCH.canonical_hash(marker)
            marker_path = stage / "shards/shard_000.json"
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            self.assertTrue(ORCH.marker_is_complete(
                marker_path, experiment_id="exp", shard_index=0,
                shard_size=50))
            artifact["rows"] = [{"tampered": True}]
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            self.assertFalse(ORCH.marker_is_complete(
                marker_path, experiment_id="exp", shard_index=0,
                shard_size=50))

    def test_vqa_owned_temporary_cleanup_is_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "generated_history"
            payload = root / "payload"
            payload.mkdir(parents=True)
            owner = {
                "schema_version": ORCH.SHARD_SCHEMA,
                "experiment_id": "exp",
                "dataset": "vqav2_validation_mt3_reconstructed",
                "purpose": "temporary_visual_kv_only",
            }
            (root / ORCH.TEMP_OWNER_FILE).write_text(
                json.dumps(owner), encoding="utf-8")
            report = ORCH.ensure_temp_clean(root, "exp")
            self.assertTrue(report["passed"])
            self.assertFalse(root.exists())

    def test_launcher_is_detached_offline_and_does_not_precreate_roots(self):
        launcher = ROOT / "scripts/63_launch_mt_vqa_v2_generated.sh"
        subprocess.run(["bash", "-n", str(launcher)], check=True)
        text = launcher.read_text(encoding="utf-8")
        self.assertIn("/home/dblab/anaconda3/envs/mllm_ft/bin/python", text)
        self.assertIn('PREFIX="mt_vqa_v2_generated_4arm_"', text)
        self.assertIn("tmux new-session -d", text)
        self.assertIn("/tmp/mllm_v2_mt_vqa_v2_generated_launcher.lock", text)
        self.assertIn("export HF_HUB_OFFLINE=1", text)
        self.assertIn("export TRANSFORMERS_OFFLINE=1", text)
        self.assertNotIn('mkdir -p "$run_dir"', text)
        self.assertNotIn('mkdir -p "$results_dir"', text)


if __name__ == "__main__":
    unittest.main()

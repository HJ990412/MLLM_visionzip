"""Focused CPU-only tests for MT-GQA history orchestration/protection."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent


def load_script(name: str, filename: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ORCH = load_script("mt_gqa_history_orchestrator_test", "56_run_mt_gqa_history.py")
PROTECT = load_script("mt_gqa_history_protector_test", "57_protect_mt_gqa_history_artifacts.py")


class ProtectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.project = Path(self.temporary.name)
        (self.project / "runs").mkdir()
        (self.project / "results").mkdir()
        (self.project / "runs" / "old").mkdir()
        (self.project / "runs" / "old" / "artifact.bin").write_bytes(b"old-data")
        (self.project / "results" / "old.json").write_text("{}\n")
        name = "mt_gqa_4arm_history_comparison_20260921T000000Z"
        self.run = self.project / "runs" / name
        self.results = self.project / "results" / name

    def tearDown(self):
        self.temporary.cleanup()

    def test_before_is_computed_before_root_creation_and_after_stays_in_run(self):
        path, before = PROTECT.record_before(
            self.run, self.results, project_root=self.project)
        self.assertEqual(path, self.run / PROTECT.MANIFEST_NAME)
        self.assertTrue(path.is_file())
        self.assertFalse(self.results.exists())
        self.assertIn("runs/old/artifact.bin", before["entries"])
        self.assertFalse(any(key.startswith(f"runs/{self.run.name}")
                             for key in before["entries"]))
        validation_path, report = PROTECT.verify_after(
            self.run, self.results, project_root=self.project)
        self.assertTrue(report["passed"])
        self.assertEqual(validation_path.parent, self.run)
        self.assertFalse(self.results.exists())

    def test_mutation_fails_closed_and_publishes_failure_evidence(self):
        PROTECT.record_before(self.run, self.results, project_root=self.project)
        (self.project / "runs" / "old" / "artifact.bin").write_bytes(b"changed")
        with self.assertRaises(PROTECT.ProtectionError):
            PROTECT.verify_after(
                self.run, self.results, project_root=self.project)
        report = json.loads((self.run / PROTECT.VALIDATION_NAME).read_text())
        self.assertFalse(report["passed"])
        self.assertEqual(report["changed_paths"], ["runs/old/artifact.bin"])
        self.assertFalse(self.results.exists())

    def test_only_exact_new_roots_are_excluded(self):
        sibling = self.project / "runs" / (
            "mt_gqa_4arm_history_comparison_existing")
        sibling.mkdir()
        (sibling / "keep.txt").write_text("protected")
        _, before = PROTECT.record_before(
            self.run, self.results, project_root=self.project)
        self.assertIn(
            "runs/mt_gqa_4arm_history_comparison_existing/keep.txt",
            before["entries"])

    def test_new_sibling_is_reported_but_does_not_mutate_frozen_set(self):
        PROTECT.record_before(self.run, self.results, project_root=self.project)
        sibling = self.project / "results" / "later-independent-output"
        sibling.mkdir()
        (sibling / "new.json").write_text("{}\n")
        _, report = PROTECT.verify_after(
            self.run, self.results, project_root=self.project)
        self.assertTrue(report["passed"])
        self.assertIn("results/later-independent-output/new.json",
                      report["added_paths"])
        self.assertEqual(report["before_manifest_sha256"],
                         report["after_manifest_sha256"])

    def test_byte_identical_replacement_is_detected(self):
        PROTECT.record_before(self.run, self.results, project_root=self.project)
        target = self.project / "runs" / "old" / "artifact.bin"
        replacement = self.project.parent / (
            f"replacement-{os.getpid()}-{id(self)}.bin")
        try:
            replacement.write_bytes(b"old-data")
            os.replace(replacement, target)
            with self.assertRaises(PROTECT.ProtectionError):
                PROTECT.verify_after(
                    self.run, self.results, project_root=self.project)
            report = json.loads(
                (self.run / PROTECT.VALIDATION_NAME).read_text())
            self.assertIn("runs/old/artifact.bin", report["changed_paths"])
        finally:
            if replacement.exists():
                replacement.unlink()

    def test_prefix_and_preexisting_empty_root_are_rejected(self):
        with self.assertRaises(ValueError):
            PROTECT.validate_output_roots(
                self.project / "runs" / "wrong_name",
                self.project / "results" / "wrong_name",
                project_root=self.project)
        self.run.mkdir()
        with self.assertRaises(FileExistsError):
            PROTECT.record_before(
                self.run, self.results, project_root=self.project)


class OrchestratorTests(unittest.TestCase):
    def test_frozen_request_accounting(self):
        self.assertEqual(ORCH.REQUESTS_PER_PROTOCOL, 48_732)
        self.assertEqual(ORCH.REQUESTS_BOTH_PROTOCOLS, 97_464)
        self.assertEqual(ORCH.SMOKE_REQUESTS_PER_PROTOCOL, 120)
        physical = (ORCH.REQUESTS_BOTH_PROTOCOLS
                    + 2 * ORCH.SMOKE_REQUESTS_PER_PROTOCOL)
        self.assertEqual(physical, 97_704)

    def test_evaluator_commands_are_protocol_scoped_and_smoke_is_explicit(self):
        root = ROOT / "runs" / "mt_gqa_4arm_history_comparison_fixture"
        command = ORCH.build_evaluator_command(
            protocol="gold_history", stage_dir=root / "smoke/gold_history",
            temp_root=root / "_temporary_visual_kv/smoke/gold_history",
            experiment_id="experiment", shard_index=0, shard_size=50,
            dialogue_limit=10, min_free_after_gib=30.0)
        rendered = " ".join(command)
        self.assertIn("54_eval_mt_gqa_history_shard.py", rendered)
        self.assertIn("--protocol gold_history", rendered)
        self.assertIn("--max-dialogs 10 --allow-partial-workload", rendered)
        self.assertIn(f"--expected-index-sha256 {ORCH.EXPECTED_INDEX_SHA256}", rendered)
        full = ORCH.build_evaluator_command(
            protocol="generated_history",
            stage_dir=root / "full/generated_history",
            temp_root=root / "_temporary_visual_kv/full/generated_history",
            experiment_id="experiment", shard_index=7, shard_size=50,
            dialogue_limit=ORCH.EXPECTED_DIALOGUES,
            min_free_after_gib=30.0)
        self.assertNotIn("--max-dialogs", full)
        self.assertIn("generated_history", full)

    def test_analyzer_receives_both_runs_and_protection_pass(self):
        base = ROOT / "runs" / "mt_gqa_4arm_history_comparison_fixture"
        result = ROOT / "results" / base.name
        command = ORCH.build_analyzer_command(
            gold_run_dir=base / "full/gold_history",
            generated_run_dir=base / "full/generated_history",
            results_root=result,
            protection_validation=base / "protected_artifacts_validation.json")
        rendered = " ".join(command)
        self.assertIn("--gold-run-dir", rendered)
        self.assertIn("--generated-run-dir", rendered)
        self.assertIn("--protection-validation", rendered)
        self.assertIn("--expected-dialogs 4061", rendered)
        self.assertIn("--expected-images 398", rendered)

    def test_manifest_is_content_hashed_and_code_locked(self):
        name = "mt_gqa_4arm_history_comparison_fixture"
        run = ROOT / "runs" / name
        results = ROOT / "results" / name
        contract = ORCH.read_index_contract()
        manifest = ORCH.make_manifest(
            run=run, results=results, experiment_id="exp",
            contract=contract, model_revision="a" * 40,
            gpu={"name": "fixture"}, shard_size=50,
            min_free_after_gib=30.0,
            protection={"manifest_sha256": "b" * 64})
        ORCH.validate_manifest(
            manifest, run=run, results=results, contract=contract)
        self.assertEqual(
            manifest["logical_request_counts"]["physical_gpu_requests_including_smoke"],
            97_704)
        self.assertIn("scripts/37_eval_mt_gqa_full_shard.py",
                      manifest["code_sha256"])
        self.assertIn("mmimpress/serve.py", manifest["code_sha256"])
        self.assertIn("transformers", manifest["runtime_environment"]["packages"])
        tampered = dict(manifest)
        tampered["code_sha256"] = {"fake.py": "0" * 64}
        unhashed = dict(tampered)
        unhashed.pop("manifest_sha256")
        tampered["manifest_sha256"] = ORCH.canonical_hash(unhashed)
        with self.assertRaises(ORCH.OrchestrationError):
            ORCH.validate_manifest(
                tampered, run=run, results=results, contract=contract)

    def test_gpu_preflight_requires_one_idle_gpu_with_headroom(self):
        completed = subprocess.CompletedProcess(
            [], 0, "NVIDIA GeForce RTX 4090, 24564, 30, 24184, 0, 535.0\n", "")
        with mock.patch.object(ORCH.subprocess, "run", return_value=completed):
            value = ORCH.gpu_preflight()
        self.assertEqual(value["memory_free_mib"], 24_184)
        busy = subprocess.CompletedProcess(
            [], 0, "GPU, 24564, 8000, 16564, 95, 535.0\n", "")
        with mock.patch.object(ORCH.subprocess, "run", return_value=busy):
            with self.assertRaises(ORCH.OrchestrationError):
                ORCH.gpu_preflight()

    def test_nonblocking_run_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            first = ORCH._acquire_run_lock(run)
            try:
                with self.assertRaises(ORCH.OrchestrationError):
                    ORCH._acquire_run_lock(run)
            finally:
                os.close(first)

    def test_global_lock_prevents_concurrent_experiments(self):
        first = ORCH._acquire_global_lock()
        try:
            with self.assertRaises(ORCH.OrchestrationError):
                ORCH._acquire_global_lock()
        finally:
            os.close(first)

    def test_analysis_requires_marker_and_full_hashed_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary)
            (results / "comparison").mkdir()
            (results / "validation.json").write_text(
                json.dumps({"passed": True}))
            (results / "comparison" / "ANALYSIS.md").write_text("partial")
            self.assertFalse(ORCH.analysis_is_complete(results))

    def test_owned_temporary_cleanup_is_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "gold_history"
            payload = root / "payload"
            payload.mkdir(parents=True)
            owner = {
                "schema_version": "fixture", "experiment_id": "exp",
                "dataset": "gqa_testdev_balanced_mt3",
                "purpose": "temporary_visual_kv_only"}
            (root / ".mt_gqa_temp_store_owner.json").write_text(
                json.dumps(owner))
            report = ORCH.ensure_temp_clean(
                root, experiment_id="exp", protocol="gold_history")
            self.assertTrue(report["passed"])
            self.assertFalse(root.exists())

    def test_protocol_stage_validation_proves_exact_four_arm_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary) / "gold_history"
            (stage / "images").mkdir(parents=True)
            (stage / "shards").mkdir()
            dialogue = {
                "dialog_id": "d1", "image_id": "i1",
                "turns": [{"turn_id": i, "question_id": f"q{i}"}
                          for i in (1, 2, 3)]}
            config = {
                "schema_version": "mt-gqa-4arm-history-shard-v2",
                "experiment_id": "exp", "protocol": "gold_history",
                "dialogues_file_sha256": ORCH.EXPECTED_INDEX_SHA256,
                "source_full_workload_sha256": ORCH.EXPECTED_WORKLOAD_SHA256,
                "n_dialogs": 1, "n_turns": 3, "n_images": 1,
                "n_requests": 12, "shard_size": 50, "n_shards": 1,
                "seed": ORCH.SEED, "method_keys": list(ORCH.METHOD_KEYS),
                "model": "llava-hf/llava-v1.6-vicuna-7b-hf",
                "model_revision": ORCH.EXPECTED_MODEL_REVISION,
                "load_4bit": True,
                "quantization": "4-bit NF4 double-quant",
                "compute_dtype": "bfloat16", "attention": "eager",
                "decoding": "greedy", "max_new_tokens": 16,
                "chunk_size": 64, "probe_heads": 3,
                "qa_chunk_configuration": ORCH.EXPECTED_QA_CONFIGURATION,
                "ours_configuration": ORCH.EXPECTED_OURS_CONFIGURATION,
                "turn1_policy": "FullLoad captures/persists its T1 raster",
                "qa_raster_source_policy": "captured by FullLoad's own T1",
            }
            (stage / "config.json").write_text(json.dumps(config))
            rows = []
            for turn in (1, 2, 3):
                for method in ORCH.METHOD_KEYS:
                    rows.append({
                        "protocol": "gold_history", "dialog_id": "d1",
                        "turn_id": turn, "method_key": method,
                        "logical_request_id": f"gold:d1:{turn}:{method}",
                        "status": "ok"})
            artifact = {
                "schema_version": "mt-gqa-4arm-history-shard-v2",
                "experiment_id": "exp", "protocol": "gold_history",
                "image_id": "i1", "shard_index": 0,
                "dialog_ids": ["d1"], "rows": rows,
                "validation": {"passed": True}}
            artifact["artifact_content_sha256"] = ORCH.canonical_hash(artifact)
            artifact_path = stage / "images" / "i1.json"
            artifact_path.write_text(json.dumps(artifact))
            marker = {
                "schema_version": "mt-gqa-4arm-history-shard-v2",
                "complete": True, "experiment_id": "exp",
                "protocol": "gold_history", "shard_index": 0,
                "shard_size": 50, "image_ids": ["i1"],
                "completed_image_ids": ["i1"],
                "image_artifact_file_sha256": {
                    "i1": ORCH.sha256_file(artifact_path)},
                "image_artifact_content_sha256": {
                    "i1": artifact["artifact_content_sha256"]},
            }
            marker["artifact_content_sha256"] = ORCH.canonical_hash(marker)
            (stage / "shards" / "shard_000.json").write_text(
                json.dumps(marker))
            report = ORCH.validate_protocol_stage(
                stage_dir=stage, protocol="gold_history",
                experiment_id="exp", dialogues=[dialogue],
                dialogue_limit=1, shard_size=50)
            self.assertEqual(report["requests"], 12)
            rows[0]["logical_request_id"] = rows[1]["logical_request_id"]
            artifact["rows"] = rows
            (stage / "images" / "i1.json").write_text(json.dumps(artifact))
            with self.assertRaises(ORCH.OrchestrationError):
                ORCH.validate_protocol_stage(
                    stage_dir=stage, protocol="gold_history",
                    experiment_id="exp", dialogues=[dialogue],
                    dialogue_limit=1, shard_size=50)

    def test_paired_stage_orders_gold_then_generated_for_each_shard(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            dialogues = [{"dialog_id": f"d{i}", "image_id": f"i{i}"}
                         for i in range(10)]
            calls = []

            def fake_run(command, _log):
                protocol = command[command.index("--protocol") + 1]
                shard = int(command[command.index("--shard-index") + 1])
                stage_dir = Path(command[command.index("--run-dir") + 1])
                marker = ORCH._marker_path(stage_dir, shard)
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(json.dumps({
                    "complete": True, "experiment_id": "exp",
                    "protocol": protocol, "shard_index": shard,
                    "shard_size": 5}))
                calls.append((shard, protocol))
                return 0.0

            with mock.patch.object(ORCH, "run_streaming", side_effect=fake_run), \
                    mock.patch.object(
                        ORCH, "marker_is_complete",
                        side_effect=lambda path, **_: path.is_file()), \
                    mock.patch.object(ORCH, "validate_protocol_stage",
                                      return_value={"passed": True}), \
                    mock.patch.object(ORCH, "validate_cross_protocol_turn1",
                                      return_value={"passed": True}), \
                    mock.patch.object(ORCH, "ensure_temp_clean",
                                      return_value={"passed": True}):
                ORCH.run_paired_stage(
                    stage_name="smoke", dialogue_limit=10, run=run,
                    experiment_id="exp",
                    contract={"dialogues": dialogues, "index": ORCH.DEFAULT_INDEX},
                    shard_size=5, min_free_after_gib=30.0,
                    progress={})
            self.assertEqual(calls, [
                (0, "gold_history"), (0, "generated_history"),
                (1, "gold_history"), (1, "generated_history")])

    def test_launcher_is_detached_and_does_not_precreate_roots(self):
        launcher = ROOT / "scripts/58_launch_mt_gqa_history.sh"
        subprocess.run(["bash", "-n", str(launcher)], check=True)
        text = launcher.read_text(encoding="utf-8")
        self.assertIn("/home/dblab/anaconda3/envs/mllm_ft/bin/python", text)
        self.assertIn("tmux new-session -d", text)
        self.assertIn("/tmp/mllm_v2_mt_gqa_history_launcher.lock", text)
        self.assertNotIn('mkdir -p "$run_dir"', text)


if __name__ == "__main__":
    unittest.main()

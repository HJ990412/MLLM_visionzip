"""CPU-only safety contracts for the generalization orchestrator."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parent.parent


def _load_script(filename: str, module_name: str):
    path = REPO / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ORCH = _load_script(
    "35_run_image_only_generalization.py",
    "mmimpress_image_only_generalization_orchestrator_test",
)
RUNNER = _load_script(
    "33_eval_image_only_generalization.py",
    "mmimpress_image_only_generalization_runner_orchestrator_test",
)


class FrozenSpecificationTests(unittest.TestCase):
    def test_canonical_dataset_order_hashes_and_counts_are_exact(self):
        self.assertEqual(tuple(ORCH.SPECS), ORCH.CANONICAL_DATASETS)
        for dataset in ORCH.CANONICAL_DATASETS:
            spec = ORCH.SPECS[dataset]
            path = Path(spec["index"])
            self.assertEqual(ORCH.sha256_file(path), spec["index_sha256"])
            entries = json.loads(path.read_text())
            request_keys = [
                (str(entry["image_id"]), str(question["question_id"]))
                for entry in entries
                for question in entry["questions"][
                    spec["skip"]:spec["skip"] + spec["questions_per_image"]]
            ]
            payload = "\n".join(
                f"{image_id}\t{question_id}"
                for image_id, question_id in request_keys).encode("utf-8")
            self.assertEqual(len(entries), spec["images"])
            self.assertEqual(len(request_keys), spec["questions"])
            self.assertEqual(
                hashlib.sha256(payload).hexdigest(), spec["workload_sha256"])

    def test_final_contract_rejects_subsets_reordering_and_weak_guards(self):
        valid = list(ORCH.CANONICAL_DATASETS)
        ORCH.validate_main_contract(
            valid, shard_size=1, bootstrap_resamples=10_000,
            min_free_after_gib=64.0)
        invalid = (
            (valid[:2], 1, 10_000, 64.0),
            (list(reversed(valid)), 1, 10_000, 64.0),
            (valid, 0, 10_000, 64.0),
            (valid, 1, 9999, 64.0),
            (valid, 1, 10_000, 63.999),
            (valid, 1, 10_000, float("nan")),
        )
        for datasets, shard_size, resamples, reserve in invalid:
            with self.subTest(datasets=datasets, shard_size=shard_size,
                              resamples=resamples, reserve=reserve):
                with self.assertRaises(ValueError):
                    ORCH.validate_main_contract(
                        datasets, shard_size=shard_size,
                        bootstrap_resamples=resamples,
                        min_free_after_gib=reserve)


class OutputNamespaceTests(unittest.TestCase):
    def test_accepts_only_dedicated_top_level_namespaces(self):
        run = REPO / "runs/image_only_generalization_contract_test"
        result = REPO / "results/image_only_generalization_contract_test"
        self.assertEqual(
            ORCH.validate_output_roots(run, result),
            (run.resolve(), result.resolve()),
        )
        invalid = (
            (REPO / "runs/old_experiment/new", result),
            (REPO / "runs/unrelated", result),
            (run, REPO / "results/old_experiment/new"),
            (run, REPO / "results/unrelated"),
            (REPO / "results/image_only_generalization_bad", result),
        )
        for bad_run, bad_result in invalid:
            with self.subTest(run=bad_run, result=bad_result):
                with self.assertRaises(ValueError):
                    ORCH.validate_output_roots(bad_run, bad_result)

    def test_atomic_json_never_overwrites_in_exclusive_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            ORCH.atomic_json(path, {"version": 1}, exclusive=True)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                ORCH.atomic_json(path, {"version": 2}, exclusive=True)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(path.parent.glob(".*.tmp-*")), [])


class ProtectionManifestTests(unittest.TestCase):
    def test_new_roots_are_excluded_but_every_old_file_is_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            old_result = root / "results/old/raw.jsonl"
            old_run = root / "runs/old/README.md"
            old_result.parent.mkdir(parents=True)
            old_run.parent.mkdir(parents=True)
            old_result.write_text("old-result\n")
            old_run.write_text("old-run\n")
            new_run = root / "runs/image_only_generalization_test"
            new_result = root / "results/image_only_generalization_test"
            with mock.patch.object(ORCH, "ROOT", root):
                snapshot = ORCH.protected_snapshot(new_run, new_result)
                self.assertEqual(set(snapshot["files"]), {
                    "results/old/raw.jsonl", "runs/old/README.md",
                })
                new_run.mkdir()
                new_result.mkdir()
                (new_run / "progress.json").write_text("new\n")
                (new_result / "summary.csv").write_text("new\n")
                ORCH.verify_protected_snapshot(
                    snapshot, new_run, new_result)
                old_result.write_text("mutated\n")
                with self.assertRaisesRegex(
                        RuntimeError, "protected pre-existing artifacts changed"):
                    ORCH.verify_protected_snapshot(
                        snapshot, new_run, new_result)


class ModelRevisionTests(unittest.TestCase):
    def test_local_checkpoint_revision_is_a_present_snapshot(self):
        revision = ORCH.local_model_revision()
        self.assertEqual(
            revision, "c916e6cdcd760b4cecd1dd4907f84ac649f93b23")

    def test_dataset_config_revision_mismatch_is_rejected(self):
        revision = "c916e6cdcd760b4cecd1dd4907f84ac649f93b23"
        with tempfile.TemporaryDirectory() as directory:
            dataset_run = Path(directory)
            config = dataset_run / "config.json"
            config.write_text(json.dumps({"model_revision": revision}))
            ORCH.verify_dataset_model_revision(dataset_run, revision)
            with self.assertRaisesRegex(RuntimeError, "revision mismatch"):
                ORCH.verify_dataset_model_revision(dataset_run, "0" * 40)

    def test_runner_resume_config_treats_revision_as_invariant(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            base = {
                "schema_version": RUNNER.SCHEMA_VERSION,
                "experiment_id": "exp",
                "dataset": "synthetic",
                "index_sha256": "a",
                "workload_sha256": "b",
                "n_images": 1,
                "n_questions": 1,
                "skip": 1,
                "questions_per_image_requested": 1,
                "seed": 1234,
                "method_keys": list(RUNNER.METHOD_KEYS),
                "boundary_text_sha256": "c",
                "model_revision": "1" * 40,
            }
            RUNNER._ensure_run_config(run_dir, base)
            changed = dict(base, model_revision="2" * 40)
            with self.assertRaisesRegex(ValueError, "model_revision"):
                RUNNER._ensure_run_config(run_dir, changed)


class CliCompatibilityTests(unittest.TestCase):
    def test_launcher_forwarded_flags_exist_in_child_clis(self):
        evaluator = subprocess.run(
            [sys.executable, str(REPO / "scripts/33_eval_image_only_generalization.py"),
             "--help"], check=True, text=True, capture_output=True).stdout
        analyzer = subprocess.run(
            [sys.executable, str(REPO / "scripts/34_analyze_image_only_generalization.py"),
             "--help"], check=True, text=True, capture_output=True).stdout
        for flag in (
            "--dataset", "--metric", "--index", "--run-dir", "--temp-root",
            "--experiment-id", "--skip", "--questions", "--shard-index",
            "--shard-size", "--expected-index-sha256",
            "--expected-workload-sha256", "--expected-images",
            "--expected-questions", "--seed", "--max-new-tokens",
            "--min-free-after-gib",
        ):
            self.assertIn(flag, evaluator)
        for flag in (
            "--run-root", "--results-root", "--bootstrap-resamples",
            "--bootstrap-seed", "--protection-manifest",
        ):
            self.assertIn(flag, analyzer)


if __name__ == "__main__":
    unittest.main()

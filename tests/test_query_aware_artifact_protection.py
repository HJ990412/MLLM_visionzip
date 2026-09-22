"""CPU-only tests for query-aware artifact protection."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/50_protect_query_aware_artifacts.py"
SPEC = importlib.util.spec_from_file_location(
    "query_aware_artifact_protection", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PROTECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROTECT)


class ArtifactProtectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.project = Path(self.temporary.name).resolve()
        (self.project / "runs").mkdir()
        (self.project / "results").mkdir()
        self.run_root = self.project / "runs/query_aware_baseline"
        self.results_root = self.project / "results/query_aware_baseline"

    def tearDown(self):
        self.temporary.cleanup()

    def test_snapshot_covers_files_directories_and_symlink_targets_only(self):
        prior_run = self.project / "runs/prior"
        (prior_run / "empty/deep").mkdir(parents=True)
        payload = prior_run / "artifact.json"
        payload.write_text('{"answer": 42}\n', encoding="utf-8")
        prior_results = self.project / "results/prior"
        prior_results.mkdir()
        (prior_results / "artifact-link").symlink_to(
            "../../runs/prior/artifact.json")
        (prior_results / "directory-link").symlink_to("../../runs/prior",
                                                       target_is_directory=True)

        # Explicit output roots are excluded even when they already contain
        # files.  record_before() separately requires these roots to be fresh.
        self.run_root.mkdir()
        (self.run_root / "new-run-file").write_text("excluded")
        self.results_root.mkdir()
        (self.results_root / "new-result-file").write_text("excluded")

        # Top-level KV stores are siblings of runs/results and never scanned.
        kvstore = self.project / "kvstore_query_aware_raster"
        kvstore.mkdir()
        (kvstore / "large-kv.bin").write_bytes(b"not actually large")

        snapshot = PROTECT.protected_snapshot(
            self.run_root, self.results_root, project_root=self.project)
        entries = snapshot["entries"]
        self.assertEqual(
            entries["runs/prior/artifact.json"]["sha256"],
            hashlib.sha256(payload.read_bytes()).hexdigest())
        self.assertEqual(entries["runs/prior/empty/deep"],
                         {"type": "directory"})
        self.assertEqual(entries["results/prior/artifact-link"], {
            "type": "symlink",
            "target": "../../runs/prior/artifact.json",
        })
        self.assertEqual(entries["results/prior/directory-link"], {
            "type": "symlink",
            "target": "../../runs/prior",
        })
        self.assertFalse(any("query_aware_baseline" in key for key in entries))
        self.assertFalse(any("kvstore" in key for key in entries))
        self.assertFalse(snapshot["kvstore_trees_hashed"])
        self.assertEqual(snapshot["file_count"], 1)
        self.assertEqual(snapshot["symlink_count"], 2)

    def test_record_and_verify_allow_changes_only_inside_new_roots(self):
        old = self.project / "results/prior/result.json"
        old.parent.mkdir()
        old.write_text('{"preserved": true}\n', encoding="utf-8")
        manifest_path, before = PROTECT.record_before(
            self.run_root, self.results_root, project_root=self.project)
        self.assertEqual(manifest_path,
                         self.run_root / PROTECT.MANIFEST_NAME)
        self.assertEqual(json.loads(manifest_path.read_text()), before)

        (self.run_root / "raw.jsonl").write_text("new run evidence\n")
        self.results_root.mkdir()
        (self.results_root / "summary.csv").write_text("method,score\nqa,1\n")
        validation_path, report = PROTECT.verify_after(
            self.run_root, self.results_root, project_root=self.project)

        self.assertTrue(report["passed"])
        self.assertEqual(report["missing_paths"], [])
        self.assertEqual(report["added_paths"], [])
        self.assertEqual(report["changed_paths"], [])
        self.assertEqual(report["before_manifest_sha256"],
                         report["after_manifest_sha256"])
        self.assertEqual(json.loads(validation_path.read_text()), report)
        self.assertEqual(old.read_text(), '{"preserved": true}\n')

    def test_verify_reports_added_missing_and_changed_then_fails(self):
        old_run = self.project / "runs/prior"
        old_run.mkdir()
        changed = old_run / "changed.txt"
        changed.write_text("before")
        missing_dir = self.project / "results/empty"
        missing_dir.mkdir()
        symlink = self.project / "results/reference"
        symlink.symlink_to("../runs/prior/changed.txt")
        PROTECT.record_before(
            self.run_root, self.results_root, project_root=self.project)

        changed.write_text("after")
        missing_dir.rmdir()
        symlink.unlink()
        symlink.symlink_to("../runs/prior/other.txt")
        added = self.project / "results/unexpected.json"
        added.write_text("{}\n")

        with self.assertRaises(PROTECT.ArtifactProtectionError) as caught:
            PROTECT.verify_after(
                self.run_root, self.results_root, project_root=self.project)
        report = caught.exception.report
        self.assertIsNotNone(report)
        assert report is not None
        self.assertFalse(report["passed"])
        self.assertEqual(report["missing_paths"], ["results/empty"])
        self.assertEqual(report["added_paths"], ["results/unexpected.json"])
        self.assertEqual(report["changed_paths"], [
            "results/reference", "runs/prior/changed.txt"])
        validation = self.results_root / PROTECT.VALIDATION_NAME
        self.assertTrue(validation.is_file())
        self.assertEqual(json.loads(validation.read_text()), report)

    def test_before_refuses_populated_or_unsafe_exclusion_roots(self):
        self.run_root.mkdir()
        (self.run_root / "existing").write_text("do not hide this")
        with self.assertRaisesRegex(FileExistsError, "absent or empty"):
            PROTECT.record_before(
                self.run_root, self.results_root, project_root=self.project)

        with self.assertRaisesRegex(ValueError, "direct query_aware"):
            PROTECT.validate_output_roots(
                self.project / "runs/not-the-experiment",
                self.results_root,
                project_root=self.project)
        with self.assertRaisesRegex(ValueError, "direct query_aware"):
            PROTECT.validate_output_roots(
                self.project / "runs/nested/query_aware_baseline",
                self.results_root,
                project_root=self.project)

    def test_tampered_manifest_fails_closed(self):
        prior = self.project / "runs/prior.txt"
        prior.write_text("stable")
        manifest_path, _ = PROTECT.record_before(
            self.run_root, self.results_root, project_root=self.project)
        manifest = json.loads(manifest_path.read_text())
        manifest["entries"].pop("runs/prior.txt")
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(
                PROTECT.ArtifactProtectionError, "content hash mismatch"):
            PROTECT.verify_after(
                self.run_root, self.results_root, project_root=self.project)


if __name__ == "__main__":
    unittest.main()

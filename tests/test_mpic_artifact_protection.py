from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "66_protect_mpic_artifacts.py"
SPEC = importlib.util.spec_from_file_location("protect_mpic_artifacts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
protect = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = protect
SPEC.loader.exec_module(protect)


class MPICArtifactProtectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.project = Path(self.temporary.name)
        (self.project / "runs" / "mpic_baseline").mkdir(parents=True)
        (self.project / "results").mkdir()
        self.prior = self.project / "runs" / "old_baseline"
        self.prior.mkdir()
        (self.prior / "small.json").write_text('{"accuracy": 1}\n')
        (self.prior / "large.bin").write_bytes(bytes(range(256)) * 8)
        self.run = self.project / "runs" / "mpic_baseline" / "unit"
        self.result = (
            self.project / "results" / "mpic_baseline" / "gqa40_240_unit")
        self.policy = protect.FingerprintPolicy(
            full_hash_max_bytes=128, sample_window_bytes=16, sample_count=5)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def record(self):
        return protect.record_before(
            self.run, self.result, project_root=self.project, policy=self.policy)

    def test_snapshot_uses_full_and_sampled_hashes(self) -> None:
        snapshot = protect.protected_snapshot(
            self.run, self.result, project_root=self.project, policy=self.policy)
        small = snapshot["entries"]["runs/old_baseline/small.json"]
        large = snapshot["entries"]["runs/old_baseline/large.bin"]
        self.assertEqual(small["integrity"]["kind"], "full_sha256")
        self.assertEqual(small["integrity"]["bytes_read"], small["size_bytes"])
        self.assertEqual(large["integrity"]["kind"], "sampled_sha256")
        self.assertEqual(large["integrity"]["sample_offsets"], [0, 508, 1016, 1524, 2032])
        self.assertEqual(large["integrity"]["bytes_read"], 80)
        self.assertLess(
            snapshot["total_content_bytes_read"], snapshot["total_logical_bytes"])

    def test_nested_output_roots_are_exactly_excluded_and_verify(self) -> None:
        manifest_path, before = self.record()
        self.assertTrue(manifest_path.is_file())
        self.assertNotIn(
            "runs/mpic_baseline/unit", before["entries"])

        (self.run / "raw.jsonl").write_text("new experiment\n")
        self.result.mkdir(parents=True)
        (self.result / "summary.csv").write_text("method,accuracy\n")
        validation_path, report = protect.verify_after(
            self.run, self.result, project_root=self.project)

        self.assertTrue(report["passed"])
        self.assertTrue(validation_path.is_file())
        self.assertIn("results/mpic_baseline", report["added_paths"])
        self.assertNotIn(
            "results/mpic_baseline/gqa40_240_unit", report["added_paths"])
        self.assertEqual(report["missing_paths"], [])
        self.assertEqual(report["changed_paths"], [])

    def test_changed_prior_file_fails_closed_and_writes_report(self) -> None:
        self.record()
        (self.prior / "small.json").write_text('{"accuracy": 0}\n')
        with self.assertRaises(protect.ProtectionError) as caught:
            protect.verify_after(self.run, self.result, project_root=self.project)
        self.assertFalse(caught.exception.report["passed"])
        self.assertIn(
            "runs/old_baseline/small.json",
            caught.exception.report["changed_paths"],
        )
        report = json.loads(
            (self.run / protect.VALIDATION_NAME).read_text())
        self.assertFalse(report["passed"])

    def test_added_sibling_is_reported_but_does_not_change_frozen_set(self) -> None:
        self.record()
        sibling = self.project / "results" / "unrelated_new_output.txt"
        sibling.write_text("new, not a pre-existing artifact\n")
        _, report = protect.verify_after(
            self.run, self.result, project_root=self.project)
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["added_paths"], ["results/unrelated_new_output.txt"])

    def test_tampered_manifest_is_rejected(self) -> None:
        manifest_path, _ = self.record()
        value = json.loads(manifest_path.read_text())
        value["entries"]["runs/old_baseline/small.json"]["size_bytes"] += 1
        manifest_path.write_text(json.dumps(value))
        with self.assertRaisesRegex(
                protect.ProtectionError, "manifest content hash mismatch"):
            protect.verify_after(self.run, self.result, project_root=self.project)

    def test_rejects_wrong_or_mismatched_output_roots(self) -> None:
        with self.assertRaisesRegex(ValueError, "run root must be"):
            protect.validate_output_roots(
                self.project / "runs" / "unit",
                self.result,
                project_root=self.project,
            )
        with self.assertRaisesRegex(ValueError, "matching the run root"):
            protect.validate_output_roots(
                self.run,
                self.project / "results" / "mpic_baseline" / "gqa40_240_other",
                project_root=self.project,
            )

    @unittest.skipUnless(hasattr(os, "symlink"), "requires symlink support")
    def test_rejects_symlinked_mpic_parent(self) -> None:
        other = self.project / "other"
        other.mkdir()
        (self.project / "results" / "mpic_baseline").symlink_to(
            other, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "not a real directory"):
            protect.validate_output_roots(
                self.run, self.result, project_root=self.project)


if __name__ == "__main__":
    unittest.main()

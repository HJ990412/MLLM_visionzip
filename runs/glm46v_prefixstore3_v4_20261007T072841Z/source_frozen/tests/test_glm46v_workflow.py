"""CPU workflow admission and immutable adoption checks; never launches a GPU."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from mmimpress.glm46v.analysis import (
    audit_requests, file_sha256, load_adopted_rows, load_adopted_setup_events,
)
from test_glm46v_analysis import fixture

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("glm_workflow_cpu", ROOT / "scripts/121_eval_glm46v_prefixstore3.py")
WORKFLOW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKFLOW)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def freeze_fixture(root):
    run = root / "run"
    run.mkdir()
    source = root / "source.py"
    source.write_text("immutable source\n")
    baseline = root / "prior_report.md"
    baseline.write_text("prior completed run\n")
    fields = {"config.json": "config_sha256", "environment.json": "environment_sha256",
              "CONTRACT.md": "contract_sha256", "validation_manifest.json": "validation_manifest_sha256"}
    hashes = {}
    for name, field in fields.items():
        (run / name).write_text("{}\n")
        hashes[field] = file_sha256(run / name)
    workload = run / "manifests/main_manifest.json"
    write_json(workload, {"dialogues": []})
    write_json(run / "baseline_protection_cpu_audit.json",
               {"files": {"prior_report.md": {"sha256": file_sha256(baseline)}}})
    write_json(run / "freeze.json", {**hashes, "source_sha256": {"source.py": file_sha256(source)},
                                     "manifests": {"manifests/main_manifest.json": file_sha256(workload)}})
    return run


class GLMWorkflowTest(unittest.TestCase):
    def test_resume_rejects_changed_source_config_workload_and_prior_report(self):
        for relative in ("source.py", "run/config.json", "run/manifests/main_manifest.json", "prior_report.md"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run = freeze_fixture(root)
                with mock.patch.object(WORKFLOW, "ROOT", root):
                    WORKFLOW.verify_freeze(run)
                    (root / relative).write_text("changed\n")
                    with self.assertRaises(ValueError):
                        WORKFLOW.verify_freeze(run)

    def test_cpu_or_failed_gate_cannot_construct_runner(self):
        for status in ("CPU_PASS_GPU_NOT_RUN", "FAIL", "RUNNING"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                run = Path(tmp)
                write_json(run / "validation/receipt.json", {"status": status})
                with mock.patch.object(WORKFLOW, "verify_freeze"), \
                     mock.patch.object(WORKFLOW, "guard_gpu") as guard, \
                     mock.patch.object(WORKFLOW, "GLM46VRunner") as runner:
                    with self.assertRaisesRegex(RuntimeError, "correctness gate"):
                        WORKFLOW.evaluate(run, "gqa_pilot")
                    guard.assert_not_called()
                    runner.assert_not_called()

    def test_main_requires_both_successful_pilots_before_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_json(run / "freeze.json", {})
            write_json(run / "validation/receipt.json", {"status": "PASS", "source_freeze_sha256": file_sha256(run / "freeze.json")})
            write_json(run / "gqa_pilot/COMPLETED.json", {"audit": {"status": "FAIL"}})
            with mock.patch.object(WORKFLOW, "verify_freeze"), \
                 mock.patch.object(WORKFLOW, "guard_gpu") as guard, \
                 mock.patch.object(WORKFLOW, "GLM46VRunner") as runner:
                with self.assertRaisesRegex(RuntimeError, "pilot audit"):
                    WORKFLOW.evaluate(run, "main")
                guard.assert_not_called()
                runner.assert_not_called()

    def test_foreign_validation_freeze_blocked_before_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_json(run / "freeze.json", {})
            write_json(run / "validation/receipt.json", {"status": "PASS", "source_freeze_sha256": "foreign"})
            with mock.patch.object(WORKFLOW, "verify_freeze"), \
                 mock.patch.object(WORKFLOW, "guard_gpu") as guard, \
                 mock.patch.object(WORKFLOW, "GLM46VRunner") as runner:
                with self.assertRaisesRegex(ValueError, "another source freeze"):
                    WORKFLOW.evaluate(run, "gqa_pilot")
                guard.assert_not_called()
                runner.assert_not_called()

    def test_image_slice_keeps_original_rotation_ordinal(self):
        manifest, rows, _ = fixture()
        image_entries = deepcopy(manifest["dialogues"][1:])
        for ordinal, entry in enumerate(image_entries, 1):
            entry["canonical_ordinal"] = ordinal
        image_rows = [r for r in rows if r["image_id"] == "b"]
        audit = audit_requests(image_rows, {"images": image_entries})
        self.assertEqual(audit["status"], "PASS")
        for entry in image_entries:
            del entry["canonical_ordinal"]
        bad = audit_requests(image_rows, {"images": image_entries})
        self.assertIn("frozen_method_rotation", {f["check"] for f in bad["failures"]})

    def test_raw_cannot_forge_canonical_ordinal(self):
        manifest, rows, _ = fixture()
        rows[0]["canonical_ordinal"] = 2
        rows[0]["method_order"] = WORKFLOW.method_order(2)
        bad = audit_requests(rows, manifest)
        self.assertIn("canonical_dialogue_ordinal", {f["check"] for f in bad["failures"]})
        self.assertIn("frozen_method_rotation", {f["check"] for f in bad["failures"]})

    def test_only_adopted_attempt_raw_and_setup_are_read_and_hashed(self):
        _, rows, events = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            image = run / "main/images/a"
            chosen = image / "attempt_002"
            failed = image / "attempt_001"
            chosen.mkdir(parents=True)
            failed.mkdir()
            (failed / "raw.jsonl").write_text('{"this":"must not be adopted"}\n')
            raw = chosen / "raw.jsonl"
            adopted_rows = [r for r in rows if r["image_id"] == "a"]
            raw.write_text("".join(json.dumps(r) + "\n" for r in adopted_rows))
            setup = chosen / "setup_events.jsonl"
            adopted_events = [{**e, "phase": "main"} for e in events if e["image_id"] == "a"]
            setup.write_text("".join(json.dumps(e) + "\n" for e in adopted_events))
            write_json(image / "COMMITTED.json", dict(status="PASS", attempt_id="attempt_002",
                       raw_path=str(raw), raw_sha256=file_sha256(raw), setup_path=str(setup),
                       setup_sha256=file_sha256(setup), requests=len(adopted_rows)))
            actual_rows, _ = load_adopted_rows(run, "main")
            actual_events, _ = load_adopted_setup_events(run, "main")
            self.assertEqual(actual_rows, adopted_rows)
            self.assertEqual(actual_events, adopted_events)
            setup.write_text(setup.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "setup sidecar hash"):
                load_adopted_setup_events(run, "main")
            raw.write_text(raw.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "raw hash"):
                load_adopted_rows(run, "main")


if __name__ == "__main__":
    unittest.main()

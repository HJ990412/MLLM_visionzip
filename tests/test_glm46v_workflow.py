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


class FakeClock:
    """Deterministic caller wall clock; filesystem and test-runner time excluded."""
    def __init__(self):
        self.seconds = 0.
        self.phases = []

    def __call__(self):
        return self.seconds

    def advance(self, milliseconds, phase):
        self.phases.append(phase)
        self.seconds += milliseconds / 1000.


class FakeSetupRunner:
    """Identity preparation and return-summary work surround a timed core.

    No model, native store or GPU is constructed. The real workflow call site
    must measure all phases instead of trusting the narrower returned timer.
    """
    def __init__(self, clock):
        self.clock = clock
        self.persist_calls = []
        self.activation_calls = []
        self.closed = False

    def run_pixels(self, image_path, question, history, *, capture, image_sha256, request_start):
        self.clock.advance(2., "pixels_request")
        result = {"prediction": "blue", "history_text": "blue", "history_token_ids": [5],
                  "generated_token_ids": [5, 151336], "reasoning_leak": False,
                  "truncated": False, "answer_extraction": {"status": "PLAIN_TEXT"},
                  "ssd_read_bytes": 0, "pread_calls": 0,
                  "request_trace": {"geometry": {"N_content": 4, "prefix_len": 5},
                                    "prompt_text": "synthetic native prompt"}}
        if capture:
            result["capture"] = "synthetic-capture"
        return result

    def persist(self, capture, out_dir, *, storage_policy):
        self.persist_calls.append((capture, Path(out_dir), storage_policy))
        self.clock.advance(5., "identity_preparation_before_writer_core")
        core_start = self.clock()
        self.clock.advance(17., "CPU_materialize_repack_write_fsync_seal_core")
        inner_ms = (self.clock() - core_start) * 1000.
        self.clock.advance(3., "expected_identity_and_wrapper_return_tail")
        # Metadata numbers here are inert plumbing fixtures, not native store
        # evidence. Storage correctness is independently tested elsewhere.
        return {"duration_ms": inner_ms,
                "metadata": {"identity": {"test_identity": "frozen"}, "store_logical_bytes": 100,
                             "bytes_content_kv": 90, "bytes_structural_kv": 5,
                             "bytes_metadata_file": 4, "bytes_seal_file": 1},
                "timing_ms": {"store_allocated_bytes": 104, "os_write_bytes": 100, "os_write_calls": 2}}

    def activate(self, out_dir):
        self.activation_calls.append(Path(out_dir))
        self.clock.advance(7., "activation_path_identity_and_FD_preparation")
        core_start = self.clock()
        self.clock.advance(11., "activation_full_integrity_read_core")
        inner_ms = (self.clock() - core_start) * 1000.
        self.clock.advance(2., "activation_IO_summary_and_wrapper_return_tail")
        return object(), {"duration_ms": inner_ms, "activation_io": {"bytes": 100, "preads": 3}}

    def close(self):
        self.closed = True


class GLMWorkflowTest(unittest.TestCase):
    def test_timed_call_build_includes_identity_preparation_and_wrapper_tail(self):
        clock = FakeClock()
        runner = FakeSetupRunner(clock)
        with mock.patch.object(WORKFLOW.time, "perf_counter", side_effect=clock):
            result, outer_ms = WORKFLOW.timed_call(runner.persist, "synthetic-capture", "/tmp/cpu-store",
                                                  storage_policy="prefix25")
        self.assertAlmostEqual(outer_ms, 25.)
        self.assertAlmostEqual(result["duration_ms"], 17.)
        self.assertAlmostEqual(outer_ms - result["duration_ms"], 8.)
        self.assertEqual(runner.persist_calls, [("synthetic-capture", Path("/tmp/cpu-store"), "prefix25")])
        self.assertEqual(clock.phases, ["identity_preparation_before_writer_core",
                         "CPU_materialize_repack_write_fsync_seal_core",
                         "expected_identity_and_wrapper_return_tail"])

    def test_timed_call_activation_includes_metadata_FD_preparation_and_summary_tail(self):
        clock = FakeClock()
        runner = FakeSetupRunner(clock)
        with mock.patch.object(WORKFLOW.time, "perf_counter", side_effect=clock):
            (store, activation), outer_ms = WORKFLOW.timed_call(runner.activate, "/tmp/cpu-store")
        self.assertIsNotNone(store)
        self.assertAlmostEqual(outer_ms, 20.)
        self.assertAlmostEqual(activation["duration_ms"], 11.)
        self.assertAlmostEqual(outer_ms - activation["duration_ms"], 9.)
        self.assertEqual(runner.activation_calls, [Path("/tmp/cpu-store")])
        self.assertEqual(clock.phases, ["activation_path_identity_and_FD_preparation",
                         "activation_full_integrity_read_core", "activation_IO_summary_and_wrapper_return_tail"])

    def test_timed_call_propagates_failure_without_fabricating_successful_setup(self):
        clock = FakeClock()
        def failing_writer():
            clock.advance(5., "identity_preparation")
            raise OSError("fixture durable writer failure")
        with mock.patch.object(WORKFLOW.time, "perf_counter", side_effect=clock):
            with self.assertRaisesRegex(OSError, "durable writer failure"):
                WORKFLOW.timed_call(failing_writer)
        self.assertEqual(clock.phases, ["identity_preparation"])

    def test_actual_image_workflow_records_complete_outer_setup_once_per_method(self):
        clock = FakeClock()
        runner = FakeSetupRunner(clock)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = root / "gqa_pilot/images/a/attempt_001"
            attempt.mkdir(parents=True)
            dialogue = {"execution_dialog_id": "gqa:a", "image_path": str(root / "unused_pixels.jpg"),
                        "image_sha256": "a" * 64,
                        "turns": [{"turn_id": 1, "question_id": "q1", "question": "What color?", "answers": ["blue"]}]}
            with mock.patch.object(WORKFLOW.time, "perf_counter", side_effect=clock), \
                    mock.patch.object(WORKFLOW, "guard_gpu"), \
                    mock.patch.object(WORKFLOW, "audit_requests", return_value={"status": "PASS"}), \
                    mock.patch("torch.cuda.empty_cache"):
                receipt = WORKFLOW.evaluate_image(runner, root, "gqa_pilot", "a", [(0, dialogue)], attempt)
            events = [json.loads(line) for line in (attempt / "setup_events.jsonl").read_text().splitlines()]
            rows = [json.loads(line) for line in (attempt / "raw.jsonl").read_text().splitlines()]
            self.assertEqual(len(events), 4)
            for method in ("fullload", "ours_prefixstore_kv25"):
                by_kind = {event["event_kind"]: event for event in events if event["method_id"] == method}
                self.assertEqual(set(by_kind), {"build", "activation"})
                build, activation = by_kind["build"], by_kind["activation"]
                # These assertions fail if the actual call site regresses to
                # returned inner17/11ms while an unused timed_call still passes.
                self.assertAlmostEqual(build["duration_ms"], 25.)
                self.assertAlmostEqual(build["inner_writer_call_return_ms"], 17.)
                self.assertEqual(build["endpoint"], "outer_writer_return")
                self.assertIn("identity preparation", build["measured_scope"])
                self.assertAlmostEqual(activation["duration_ms"], 20.)
                self.assertAlmostEqual(activation["inner_activation_core_ms"], 11.)
                self.assertEqual(activation["endpoint"], "outer_activation_return")
                self.assertIn("summary and wrapper tail", activation["measured_scope"])
            self.assertEqual(len(runner.persist_calls), 2)
            self.assertEqual([call[-1] for call in runner.persist_calls], ["full", "prefix25"])
            self.assertEqual(len(runner.activation_calls), 2)
            self.assertEqual(receipt["requests"], len(rows))
            self.assertEqual(len(rows), 3)
            self.assertTrue(runner.closed)
            # Request E2E and setup are disjoint: the 2ms pixels calls do not
            # absorb writer/activation cost, and setup is counted exactly once.
            self.assertTrue(all(abs(row["request_e2e_ms"] - 2.) < 1e-9 for row in rows))
            self.assertAlmostEqual(sum(event["duration_ms"] for event in events), 90.)
            self.assertAlmostEqual(sum(row["request_e2e_ms"] for row in rows), 6.)

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

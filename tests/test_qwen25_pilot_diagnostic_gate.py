"""CPU-only checks for Qwen pilot diagnostic gating and GQA history isolation."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, filename: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PILOT = _load("qwen25_pilot_gate_test", "79_eval_qwen25_pilot.py")
REPORT = _load("qwen25_report_gate_test", "80_report_qwen25_pilot.py")


def _comparison(close: bool) -> dict:
    return {
        "passed": close, "numerically_close": close,
        "first_token_identical": True,
        "generated_tokens_identical": True,
        "prediction_identical": True,
        "max_abs_logit_error": 0.2 if not close else 0.0,
        "max_relative_logit_error": 0.1 if not close else 0.0,
    }


def _validation() -> dict:
    gates = {name: {"status": "PASS", "details": {}}
             for name in (*PILOT.STRUCTURAL_GATES, *PILOT.NUMERICAL_GATES)}
    return {
        "schema_version": PILOT.VALIDATOR_SCHEMA,
        "status": "PASS", "pilot_eligible": True,
        "model_revision": PILOT.CHECKPOINT_REVISION,
        "configuration": {"seed": PILOT.SEED, "max_new_tokens": 16,
                          "chunk_size": 64, "budget_ratio": 0.25,
                          "attention_backend": "sdpa"},
        "source": {"gqa_index_sha256": PILOT.GQA_INDEX_SHA256},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in PILOT.ADAPTER_SOURCES},
        "gates": gates,
    }


def _numerical_fail(validation: dict) -> dict:
    value = copy.deepcopy(validation)
    value["status"] = "FAIL"
    value["pilot_eligible"] = False
    value["gates"]["fullload"] = {
        "status": "FAIL", "details": {"per_question": [{
            "ssd_request": {"vision_calls": 0},
            "ssd_vs_memory": _comparison(True),
            "ssd_vs_recompute": _comparison(False),
        }]}}
    value["gates"]["prefix25"] = {
        "status": "FAIL", "details": {"per_question": [{
            "compact_request": {"vision_calls": 0},
            "same_full_logical_suffix_positions": True,
            "dense_and_compact_cache_slots_correct": True,
            "same_selected_visual_set": True,
            "dense_vs_compact": _comparison(False),
        }]}}
    return value


class DiagnosticGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "validation.json"

    def write(self, value: dict) -> Path:
        self.path.write_text(json.dumps(value), encoding="utf-8")
        return self.path

    def test_default_gate_requires_every_gpu_gate_pass(self):
        result = PILOT._gate(self.write(_validation()))
        self.assertTrue(result["benchmark_validated"])
        self.assertEqual(result["validation_status"], "PASS")
        with self.assertRaisesRegex(RuntimeError, "not fully PASS"):
            PILOT._gate(self.write(_numerical_fail(_validation())))

    def test_diagnostic_accepts_only_numerical_failures(self):
        path = self.write(_numerical_fail(_validation()))
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        accepted = PILOT._gate(path, diagnostic_after_numerical_fail=True)
        self.assertEqual(accepted["validation_status"], "FAIL")
        self.assertFalse(accepted["benchmark_validated"])
        self.assertEqual(accepted["run_mode"], "diagnostic_numerical_mismatch")
        self.assertEqual(accepted["numerical_gate_evidence"]["prefix25"]["status"], "FAIL")
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_diagnostic_rejects_every_structural_failure(self):
        for gate, status in (
            ("geometry", "FAIL"), ("gpu_score_reference", "NOT RUN"),
            ("roundtrip", "FAIL"), ("repacked_full100", "NOT RUN"),
            ("io", "FAIL"), ("history", "NOT RUN"),
        ):
            with self.subTest(gate=gate, status=status):
                value = _numerical_fail(_validation())
                value["gates"][gate]["status"] = status
                with self.assertRaisesRegex(RuntimeError, "structural GPU gates"):
                    PILOT._gate(self.write(value), diagnostic_after_numerical_fail=True)

    def test_diagnostic_rejects_output_mismatch_and_shape_failure(self):
        value = _numerical_fail(_validation())
        comparison = value["gates"]["prefix25"]["details"]["per_question"][0]["dense_vs_compact"]
        comparison["prediction_identical"] = False
        with self.assertRaisesRegex(RuntimeError, "not proven numerical-only"):
            PILOT._gate(self.write(value), diagnostic_after_numerical_fail=True)
        comparison["prediction_identical"] = True
        comparison["reason"] = "first logits absent or shape mismatch"
        with self.assertRaisesRegex(RuntimeError, "not proven numerical-only"):
            PILOT._gate(self.write(value), diagnostic_after_numerical_fail=True)

    def test_diagnostic_rejects_stale_adapter_hash(self):
        value = _numerical_fail(_validation())
        value["source_hashes"][PILOT.ADAPTER_SOURCES[0]] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "adapter source changed"):
            PILOT._gate(self.write(value), diagnostic_after_numerical_fail=True)

    def test_report_requires_explicit_diagnostic_option(self):
        bound = PILOT._gate(self.write(_numerical_fail(_validation())),
                            diagnostic_after_numerical_fail=True)
        manifest = {"validation": bound,
                    "validation_status": "FAIL", "benchmark_validated": False,
                    "run_mode": "diagnostic_numerical_mismatch"}
        final = {**manifest, "status": "DIAGNOSTIC COMPLETE",
                 "execution_status": "PASS"}
        with self.assertRaisesRegex(ValueError, "requires explicit diagnostic"):
            REPORT._audit_validation_mode(manifest, final,
                                          diagnostic_after_numerical_fail=False)
        self.assertFalse(REPORT._audit_validation_mode(
            manifest, final,
            diagnostic_after_numerical_fail=True)["benchmark_validated"])

    def test_gqa_history_is_empty_after_prior_answers(self):
        histories = {method: [{"question": "Earlier question",
                               "prediction": f"{method} answer"}]
                     for method in PILOT.METHODS}
        for method in PILOT.METHODS:
            self.assertEqual(PILOT._prior_history("gqa", histories, method), [])
            self.assertEqual(PILOT._prior_history("mt_gqa_reconstructed", histories,
                                                  method), histories[method])


if __name__ == "__main__":
    unittest.main()

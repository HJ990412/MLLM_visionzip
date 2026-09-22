"""CPU checks for the isolated native-overflow diagnostic cap."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/47_probe_convbench_native_overflow.py"
SPEC = importlib.util.spec_from_file_location("convbench_native_probe_test", SCRIPT)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class NativeProbeCapTests(unittest.TestCase):
    def setUp(self):
        self.runner = SimpleNamespace(model=SimpleNamespace(
            config=SimpleNamespace(text_config=SimpleNamespace(
                max_position_embeddings=4096, rope_scaling=None))))
        self.server = SimpleNamespace(max_new_tokens=1024)

    def test_within_context_uses_production_cap(self):
        calls = []

        def production(runner, server, tokens):
            calls.append(tokens)
            return {"effective_max_new_tokens": 4096 - tokens + 1}

        self.assertEqual(PROBE.native_cap(production, self.runner,
                                          self.server, 4096, 32),
                         {"effective_max_new_tokens": 1})
        self.assertEqual(calls, [4096])

    def test_overflow_is_exact_and_does_not_change_config(self):
        def production(*args):
            self.fail("production guard must be bypassed for probe")

        first = PROBE.native_cap(production, self.runner, self.server, 4137, 32)
        second = PROBE.native_cap(production, self.runner, self.server, 4137, 32)
        self.assertEqual(first, second)
        self.assertEqual(first["context_input_tokens"], 4137)
        self.assertEqual(first["context_remaining_positions_before_request"], -41)
        self.assertEqual(first["effective_max_new_tokens"], 32)
        self.assertEqual(self.runner.model.config.text_config.max_position_embeddings,
                         4096)
        self.assertIsNone(self.runner.model.config.text_config.rope_scaling)

    def test_overflow_cap_must_be_bounded(self):
        for cap in (0, 1025):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                PROBE.native_cap(None, self.runner, self.server, 4120, cap)


if __name__ == "__main__":
    unittest.main()

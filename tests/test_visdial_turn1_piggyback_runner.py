"""CPU-only contracts for the final VisDial piggyback runner."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


def _load_runner_module():
    path = (Path(__file__).resolve().parent.parent / "scripts" /
            "28_eval_visdial_turn1_piggyback.py")
    spec = importlib.util.spec_from_file_location(
        "mmimpress_visdial_piggyback_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner_module()


class SyntheticWarmupTests(unittest.TestCase):
    def test_fixture_is_deterministic_in_memory_rgb(self):
        first, first_hash = RUNNER._synthetic_warmup_image()
        second, second_hash = RUNNER._synthetic_warmup_image()
        self.assertEqual(first.mode, "RGB")
        self.assertEqual(first.size, (640, 480))
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(first_hash, second_hash)
        self.assertEqual(len(first_hash), 64)

    def test_warmup_uses_real_call_shape_and_discards_returned_cache(self):
        class FakeCapture:
            call_count = 1
            saliency_call_count = 1

            def __enter__(self):
                return self

            def __exit__(self, *unused):
                return False

            def stats(self):
                return {
                    "vision_call_count": 1,
                    "saliency_call_count": 1,
                    "vision_ms": 2.0,
                    "saliency_reduction_ms": 0.25,
                }

        class FakeServer:
            def __init__(self):
                self.calls = []

            def recompute(self, enc, return_past_key_values=False):
                self.calls.append((enc, return_past_key_values))
                return {
                    "captured_past_key_values": object(),
                    "first_token_id": 7,
                    "generated_tokens": 3,
                }

        enc = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "pixel_values": torch.zeros(1, 1, 3, 2, 2),
            "image_sizes": torch.tensor([[2, 2]]),
        }
        runner = SimpleNamespace(to_device=lambda value: value)
        server = FakeServer()
        timing = {"processor_total_ms": 4.0}
        with mock.patch.object(
                RUNNER, "VisionForwardCapture",
                side_effect=lambda *_args, **_kwargs: FakeCapture()), \
             mock.patch.object(
                 RUNNER, "_exact_processor_call",
                 return_value=(enc, timing)) as processor_call, \
             mock.patch.object(RUNNER.torch.cuda, "synchronize"):
            result = RUNNER._run_unmeasured_warmup(runner, server)

        processor_call.assert_called_once()
        self.assertEqual(len(server.calls), 1)
        self.assertIs(server.calls[0][0], enc)
        self.assertIs(server.calls[0][1], True)
        self.assertTrue(result["excluded_from_all_latency_metrics"])
        self.assertFalse(result["fixture_is_dataset_image"])
        self.assertFalse(result["consumed_experiment_request"])
        self.assertTrue(result["captured_cache_discarded"])
        self.assertFalse(result["ssd_store_written"])
        self.assertEqual(result["vision_forward_count"], 1)
        self.assertEqual(result["saliency_call_count"], 1)


class StoredRequestInstrumentationTests(unittest.TestCase):
    def test_no_vision_guard_is_active_before_cache_conditioning(self):
        events = []

        class FakeGuard:
            def __init__(self, _runner):
                self.calls = 0

            def __enter__(self):
                events.append("guard_enter")
                return self

            def __exit__(self, *unused):
                events.append("guard_exit")
                return False

        class FakeReader:
            def drop_all(self):
                self_guard_active = events and events[-1] == "guard_enter"
                if not self_guard_active:
                    raise AssertionError("conditioning ran before guard setup")
                events.append("drop_all")

        class FakeTokenizer:
            def __call__(self, _prompt, return_tensors=None):
                return {"input_ids": torch.tensor([[1, 32000, 2, 3]])}

        class FakeServer:
            def request_cvpr25(self, *_args, **_kwargs):
                events.append("server")
                return {"answer": "ok"}

        runner = SimpleNamespace(
            image_token_id=32000,
            processor=SimpleNamespace(tokenizer=FakeTokenizer()),
            model=SimpleNamespace(device=torch.device("cpu")),
        )
        ctx = SimpleNamespace(reader=FakeReader())
        dialog = {"dialog_id": "synthetic"}
        turn = {"turn_id": 2}
        with mock.patch.object(RUNNER, "_NoVisionForward", FakeGuard), \
             mock.patch.object(RUNNER, "visdial_prompt", return_value="p"), \
             mock.patch.object(RUNNER, "_timing_fields", return_value={}), \
             mock.patch.object(RUNNER.torch.cuda, "synchronize"):
            result = RUNNER._run_stored_request(
                runner, FakeServer(), ctx, dialog, turn, "prefix25", 0.25,
                True, 1234, "synthetic-image")

        self.assertEqual(events, [
            "guard_enter", "drop_all", "server", "guard_exit"])
        self.assertEqual(result["vision_forward_count"], 0)
        self.assertTrue(result["page_cache_conditioning_excluded_from_ttft"])


if __name__ == "__main__":
    unittest.main()

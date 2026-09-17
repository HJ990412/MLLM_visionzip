"""CPU/synthetic contracts for the full-shard MT-GQA runner."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/37_eval_mt_gqa_full_shard.py"
SPEC = importlib.util.spec_from_file_location("mt_gqa_full_shard_test", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


def _dialog(index: int, image_id: str | None = None) -> dict:
    image_id = image_id or f"image-{index:03d}"
    return {
        "dialog_id": f"dialog-{index:04d}",
        "image_id": image_id,
        "image_path": f"data/gqa_large/images/{image_id}.jpg",
        "global_dialog_ordinal": index,
        "turns": [
            {
                "turn_id": turn,
                "question_id": f"q-{index:04d}-{turn}",
                "question": f"question {index} turn {turn}?",
                "answers": [f"gold-{index}-{turn}"],
            }
            for turn in (1, 2, 3)
        ],
    }


class WorkloadTests(unittest.TestCase):
    def _write(self, directory: str, value) -> Path:
        path = Path(directory) / "dialogues.json"
        path.write_text(json.dumps(value))
        return path

    def test_list_and_payload_forms_have_identical_request_hash(self):
        dialogs = [_dialog(0), _dialog(1)]
        with tempfile.TemporaryDirectory() as directory:
            list_path = self._write(directory, dialogs)
            expected_file = RUNNER.sha256_file(list_path)
            expected_workload = hashlib.sha256("".join(
                f"{dialog['dialog_id']}\t{turn['turn_id']}\t"
                f"{turn['question_id']}\n"
                for dialog in dialogs for turn in dialog["turns"]
            ).encode()).hexdigest()
            loaded = RUNNER.resolve_dialogues(
                list_path, expected_dialogs=2,
                expected_dialogues_sha256=expected_file,
                expected_workload_sha256=expected_workload)
            self.assertEqual(loaded["workload_sha256"], expected_workload)
            self.assertEqual(loaded["n_turns"], 6)

            payload_path = Path(directory) / "payload.json"
            payload_path.write_text(json.dumps({"dialogues": dialogs}))
            payload = RUNNER.resolve_dialogues(
                payload_path, expected_dialogs=2,
                expected_workload_sha256=expected_workload)
            self.assertEqual(payload["workload_sha256"], expected_workload)

    def test_bad_top_level_content_hash_is_rejected(self):
        body = {"schema_version": "synthetic", "dialogs": [_dialog(0)]}
        payload = dict(body)
        payload["artifact_content_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, payload)
            with self.assertRaisesRegex(ValueError, "content hash"):
                RUNNER.resolve_dialogues(path)

    def test_partial_slice_preserves_full_global_ordinals(self):
        dialogs = [_dialog(i) for i in range(100)]
        full = {
            "dialogs": dialogs,
            "n_dialogs": 100,
            "workload_sha256": RUNNER._dialogue_workload_hash(dialogs),
        }
        selected = RUNNER.select_dialogues(full, 10)
        self.assertEqual(
            [d["global_dialog_ordinal"] for d in selected["dialogs"]],
            list(range(10)))
        self.assertEqual(selected["source_full_n_dialogs"], 100)
        self.assertTrue(selected["partial_workload"])

    def test_image_grouping_and_40_to_60_sharding_are_stable(self):
        dialogs = [_dialog(i, image_id=f"image-{i // 2:03d}")
                   for i in range(102)]
        groups = RUNNER.group_dialogues_by_image(dialogs)
        self.assertEqual(len(groups), 51)
        self.assertTrue(all(len(group["dialogs"]) == 2 for group in groups))
        first = RUNNER.shard_image_groups(groups, 0, 50)
        second = RUNNER.shard_image_groups(groups, 1, 50)
        self.assertEqual((first["start"], first["stop"]), (0, 50))
        self.assertEqual((second["start"], second["stop"]), (50, 51))
        with self.assertRaises(ValueError):
            RUNNER.shard_image_groups(groups, 0, 39)
        with self.assertRaises(ValueError):
            RUNNER.shard_image_groups(groups, 0, 61)

    def test_cyclic_order_is_one_order_for_all_dialogue_turns(self):
        orders = [RUNNER.method_order(i, 1234) for i in range(4)]
        self.assertEqual(orders[0], (
            "recompute", "fullload", "prefix25", "prefix45"))
        self.assertEqual(orders[1], (
            "fullload", "prefix25", "prefix45", "recompute"))
        self.assertEqual(len(set(orders)), 4)
        for position in range(4):
            self.assertEqual(
                {order[position] for order in orders},
                set(RUNNER.METHOD_KEYS))
        dialog_order = RUNNER.method_order(17, 1234)
        self.assertTrue(all(
            RUNNER.method_order(17, 1234) == dialog_order
            for _turn in (1, 2, 3)))
        source = RUNNER.designated_source_method(dialog_order)
        self.assertEqual(source, [m for m in dialog_order
                                  if m.startswith("prefix")][-1])


class PromptTests(unittest.TestCase):
    def test_prompt_is_gold_teacher_forced_and_causal(self):
        dialog = _dialog(0)
        prompt = RUNNER.gold_history_prompt(dialog, 2)
        history = RUNNER.prior_history_text(dialog, 2)
        self.assertTrue(prompt.startswith("USER: <image>"))
        self.assertTrue(prompt.endswith("ASSISTANT:"))
        self.assertIn("question 0 turn 1?", prompt)
        self.assertIn("gold-0-1", prompt)
        self.assertIn("question 0 turn 2?", prompt)
        self.assertNotIn("question 0 turn 3?", prompt)
        self.assertNotIn("gold-0-2", history)
        self.assertNotIn("gold-0-3", history)

    def test_stored_prompt_is_built_after_cold_conditioning(self):
        events = []

        class Guard:
            calls = 0
            def __init__(self, _runner):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *_unused):
                return False

        class Tokenizer:
            def __call__(self, _prompt, return_tensors=None):
                events.append("tokenize")
                return {"input_ids": torch.tensor([[1, 32000, 2]])}

        class Reader:
            def drop_all(self):
                events.append("condition")

        class Server:
            def request_cvpr25(self, *_args, **_kwargs):
                events.append("server")
                return {"answer": "ok", "first_token_id": 2,
                        "generated_tokens": 1}

        helper = SimpleNamespace(
            _NoVisionForward=Guard,
            _suffix_from_tokenized=lambda runner, tokenized:
                tokenized["input_ids"][0, 2:],
            _timing_fields=lambda *_args, **_kwargs: {},
        )
        runner = SimpleNamespace(
            processor=SimpleNamespace(tokenizer=Tokenizer()),
            model=SimpleNamespace(device=torch.device("cpu")),
        )
        dialog = _dialog(0)
        with mock.patch.object(RUNNER, "_load_visdial_helpers",
                               return_value=helper), \
             mock.patch.object(RUNNER, "gold_history_prompt",
                               side_effect=lambda *_args: (
                                   events.append("prompt") or "expected")), \
             mock.patch.object(RUNNER.torch.cuda, "synchronize"):
            RUNNER._run_stored_request(
                runner, Server(), SimpleNamespace(reader=Reader()), dialog,
                2, "expected", "prefix25", budget=0.25, cold=True,
                seed=1234, image_id="image-0")
        self.assertEqual(events[:2], ["condition", "prompt"])
        self.assertEqual(events[-1], "server")

    def test_combined_expanded_image_span_yields_exact_cached_suffix(self):
        runner = SimpleNamespace(image_token_id=32000)
        one = torch.tensor([[1, 32000, 7, 8, 9]])
        expanded = torch.tensor([[1, 32000, 32000, 32000, 7, 8, 9]])
        self.assertTrue(torch.equal(
            RUNNER._combined_suffix(runner, one), torch.tensor([7, 8, 9])))
        self.assertTrue(torch.equal(
            RUNNER._combined_suffix(runner, expanded),
            torch.tensor([7, 8, 9])))
        with self.assertRaisesRegex(AssertionError, "contiguous"):
            RUNNER._combined_suffix(
                runner, torch.tensor([[1, 32000, 7, 32000, 8]]))


class PrefixContractTests(unittest.TestCase):
    def test_exact_first_k_and_nested_budgets(self):
        n_chunks = 20
        ids25 = RUNNER.prefix_chunk_ids(n_chunks, 0.25)
        ids45 = RUNNER.prefix_chunk_ids(n_chunks, 0.45)
        self.assertEqual(
            len(ids25), RUNNER.budget_chunk_count(n_chunks, 0.25))
        self.assertEqual(
            len(ids45), RUNNER.budget_chunk_count(n_chunks, 0.45))
        row25 = {
            "n_chunks_total": n_chunks,
            "selected_chunk_ids_per_layer": [ids25, ids25],
        }
        row45 = {
            "n_chunks_total": n_chunks,
            "selected_chunk_ids_per_layer": [ids45, ids45],
        }
        RUNNER.validate_nested_prefixes(row25, row45)
        bad = dict(row25)
        bad["selected_chunk_ids_per_layer"] = [[1, 2, 3, 4, 5]]
        with self.assertRaisesRegex(AssertionError, "first-k"):
            RUNNER.validate_nested_prefixes(bad, row45)


class OwnershipTests(unittest.TestCase):
    def test_cleanup_is_limited_to_one_owned_image_leaf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = (Path(directory) / "temp-store").resolve()
            RUNNER._claim_temp_root(root, "experiment")
            good = root / "payload" / "image-1"
            good.mkdir()
            (good / "payload.bin").write_bytes(b"safe")
            self.assertEqual(
                RUNNER.assert_owned_temp_path(
                    good, root, "experiment", "image-1"), good)
            with self.assertRaises(ValueError):
                RUNNER.assert_owned_temp_path(
                    root / "payload", root, "experiment")
            with self.assertRaises(ValueError):
                RUNNER.assert_owned_temp_path(
                    good / "nested", root, "experiment")
            with self.assertRaises(ValueError):
                RUNNER.assert_owned_temp_path(
                    good, root, "another-experiment", "image-1")
            self.assertTrue(RUNNER.remove_owned_temp_store(
                good, root, "experiment", "image-1"))
            self.assertFalse(good.exists())
            self.assertTrue(root.exists())

    def test_exclusive_artifact_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            RUNNER._write_exclusive_json(path, {"complete": True})
            with self.assertRaises(FileExistsError):
                RUNNER._write_exclusive_json(path, {"complete": False})


class ResumeTests(unittest.TestCase):
    @staticmethod
    def _fixture(directory: Path):
        dialog = _dialog(0, "image-0")
        group = {"image_id": "image-0", "image_ordinal": 0,
                 "dialogs": [dialog]}
        workload = {
            "benchmark_type": "MT-GQA-reconstructed",
            "dialogues_file_sha256": "a" * 64,
            "source_full_workload_sha256": "b" * 64,
            "selected_workload_sha256": "c" * 64,
        }
        store_id = "d" * 64
        rows = []
        order = list(RUNNER.method_order(0, 1234))
        for turn_id in (1, 2, 3):
            for position, method in enumerate(order):
                cached = turn_id >= 2 and method != "recompute"
                budget = RUNNER.METHODS[method]["budget"]
                selected = None
                if method.startswith("prefix") and turn_id >= 2:
                    wanted = RUNNER.prefix_chunk_ids(20, budget)
                    selected = [wanted, wanted]
                rows.append({
                    "dialog_id": dialog["dialog_id"],
                    "image_id": "image-0",
                    "global_dialog_ordinal": 0,
                    "turn_id": turn_id,
                    "method_key": method,
                    "method": RUNNER.METHODS[method]["label"],
                    "budget": budget,
                    "method_order": order,
                    "method_order_position": position,
                    "gpu_request_cache_fresh": True,
                    "text_kv_reused_from_prior_turn": False,
                    "dialogue_session_id": "session-0",
                    "context_instance_id": "context-0",
                    "prompt_sha256": "e" * 64,
                    "text_history_sha256": f"{turn_id}" * 64,
                    "suffix_ids_sha256": f"{turn_id + 3}" * 64,
                    "prediction": f"prediction-{turn_id}",
                    "first_token_id": turn_id,
                    "end_to_end_ttft_ms": 1.0,
                    "request_e2e_ms": 2.0,
                    "used_by_request": cached,
                    "physical_store_exists_at_request_start": turn_id >= 2,
                    "store_id": store_id if cached else None,
                    "ssd_read_bytes": (
                        1000 if method == "fullload" and cached else
                        250 if cached else 0),
                    "vision_forward_count": 0 if cached else 1,
                    "page_cache_conditioning_excluded_from_ttft": (
                        True if cached else None),
                    "n_chunks_total": 20 if selected else None,
                    "selected_chunk_ids_per_layer": selected,
                    "selection_fingerprint_sha256": (
                        RUNNER._json_hash(selected) if selected else None),
                    "permutation_sha256": "f" * 64 if cached else None,
                    "static_score_calls": 0,
                    "query_score_calls": 0,
                    "diversity_calls": 0,
                    "layout_questions_used": 0,
                    "layout_answers_used": 0,
                    "calibration_questions": 0,
                    "future_turns_used_for_layout": 0,
                    "persistence_source_request": (
                        turn_id == 1 and method
                        == RUNNER.designated_source_method(order)),
                })
        artifact = {
            "schema_version": RUNNER.SCHEMA_VERSION,
            "experiment_id": "experiment",
            "dataset": RUNNER.DATASET,
            "benchmark_type": "MT-GQA-reconstructed",
            "image_id": "image-0",
            "image_ordinal": 0,
            "shard_index": 0,
            **workload,
            "store_manifest": {
                "meta_sha256": store_id,
                "visual_kv_bytes": 1000,
                "permutation_sha256": "f" * 64,
            },
            "persistence_overhead": {
                "store_build_count": 1,
                "source_dialog_id": dialog["dialog_id"],
                "source_method_key": RUNNER.designated_source_method(order),
                "source_turn_id": 1,
            },
            "rows": rows,
            "validation": {"passed": True},
        }
        artifact["artifact_content_sha256"] = RUNNER._artifact_body_hash(
            artifact)
        path = directory / "image-0.json"
        path.write_text(json.dumps(artifact))
        return path, artifact, group, workload

    def test_resume_requires_content_hash_and_semantic_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path, artifact, group, workload = self._fixture(Path(directory))
            loaded = RUNNER.validate_resume_artifact(
                path, experiment_id="experiment", image_group=group,
                shard_index=0, workload=workload, seed=1234)
            self.assertEqual(loaded["artifact_content_sha256"],
                             artifact["artifact_content_sha256"])
            artifact["rows"][0]["dialogue_session_id"] = "changed-session"
            artifact["artifact_content_sha256"] = RUNNER._artifact_body_hash(
                artifact)
            path.write_text(json.dumps(artifact))
            with self.assertRaisesRegex(ValueError, "session ID"):
                RUNNER.validate_resume_artifact(
                    path, experiment_id="experiment", image_group=group,
                    shard_index=0, workload=workload, seed=1234)
            artifact["rows"][0]["dialogue_session_id"] = "session-0"
            artifact["rows"][0]["ssd_read_bytes"] = 1
            # Even a self-consistently rehashed semantic corruption must fail.
            artifact["artifact_content_sha256"] = RUNNER._artifact_body_hash(
                artifact)
            path.write_text(json.dumps(artifact))
            with self.assertRaisesRegex(ValueError, "Turn 1"):
                RUNNER.validate_resume_artifact(
                    path, experiment_id="experiment", image_group=group,
                    shard_index=0, workload=workload, seed=1234)


class CliTests(unittest.TestCase):
    def _args(self, **updates):
        values = {
            "run_dir": Path("/tmp/mt-gqa-run"),
            "temp_root": Path("/tmp/mt-gqa-temp"),
            "shard_size": 50,
            "shard_index": 0,
            "seed": 1234,
            "max_new_tokens": 16,
            "min_free_after_gib": 30.0,
            "max_dialogs": None,
            "allow_partial_workload": False,
        }
        values.update(updates)
        return SimpleNamespace(**values)

    def test_frozen_cli_contract(self):
        RUNNER._validate_cli(self._args())
        RUNNER._validate_cli(self._args(
            max_dialogs=10, allow_partial_workload=True))
        for updates in (
            {"shard_size": 39}, {"shard_size": 61}, {"seed": 0},
            {"max_new_tokens": 32}, {"min_free_after_gib": 29.9},
            {"max_dialogs": 10, "allow_partial_workload": False},
        ):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                RUNNER._validate_cli(self._args(**updates))


if __name__ == "__main__":
    unittest.main()

"""CPU contracts for ConvBench prompt, generated history, and resume."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load("_test_convbench_answer_runner",
               ROOT / "scripts/41_eval_convbench_answers.py")
LLAVA_SOURCE = Path("/home/dblab/hj/SparseVLMs/llava/conversation.py")


class ConvBenchAnswerRunnerTests(unittest.TestCase):
    @unittest.skipUnless(LLAVA_SOURCE.is_file(),
                         "local LLaVA conversation template unavailable")
    def test_render_prompt_matches_official_llava_v1(self):
        official = _load("_test_official_llava_conversation", LLAVA_SOURCE)
        for n in (1, 2, 3):
            questions = [f"Question {i}?" for i in range(1, n + 1)]
            answers = [f"Generated answer {i}." for i in range(1, n)]
            conv = official.conv_templates["llava_v1"].copy()
            for i, question in enumerate(questions):
                user_text = f"<image>\n{question}" if i == 0 else question
                conv.append_message(conv.roles[0], user_text)
                conv.append_message(conv.roles[1],
                                    answers[i] if i < len(answers) else None)
            self.assertEqual(RUNNER.render_prompt(questions, answers),
                             conv.get_prompt())

    def test_generated_history_is_method_specific_and_causal(self):
        questions = ["Q1-private", "Q2-private", "Q3-future"]
        generated = {
            "recompute": ["ReComp A1", "ReComp A2"],
            "fullload": ["FullLoad A1", "FullLoad A2"],
            "prefix25": ["Prefix25 A1", "Prefix25 A2"],
            "prefix45": ["Prefix45 A1", "Prefix45 A2"],
        }
        for method, answers in generated.items():
            turn2 = RUNNER.render_prompt(questions[:2], answers[:1])
            turn3 = RUNNER.render_prompt(questions, answers)
            self.assertIn(answers[0], turn2)
            self.assertNotIn(answers[1], turn2)
            self.assertNotIn("Q3-future", turn2)
            self.assertIn(answers[0], turn3)
            self.assertIn(answers[1], turn3)
            serialized = RUNNER.serialized_prior_history(
                turn3, questions[-1], 3)
            self.assertIn(answers[0], serialized)
            self.assertIn(answers[1], serialized)
            self.assertNotIn("Q3-future", serialized)
            for other, other_answers in generated.items():
                if other != method:
                    self.assertNotIn(other_answers[0], turn2)
                    self.assertNotIn(other_answers[1], turn3)
        with self.assertRaises(ValueError):
            RUNNER.render_prompt(questions, generated["recompute"][:1])

    def test_index_and_source_provenance(self):
        rows, index_sha = RUNNER.load_index(
            ROOT / "data/convbench/index.json")
        self.assertEqual(len(rows), 577)
        self.assertEqual(len({row["image_id"] for row in rows}), 573)
        self.assertEqual(len({row["conversation_id"] for row in rows}), 577)
        self.assertEqual(len(index_sha), 64)
        provenance = RUNNER.load_source_provenance()
        self.assertEqual(len(provenance["source_workbook_sha256"]), 64)
        self.assertEqual(len(provenance["official_repository_commit"]), 40)

    def test_mutated_index_rejected_even_with_its_own_hash(self):
        payload = json.loads((ROOT / "data/convbench/index.json").read_text())
        payload["conversations"][0]["turns"][0]["question"] = "tampered"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "index.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "pinned official source"):
                RUNNER.load_index(path)

    def test_model_forward_counter_counts_decode_steps(self):
        model = torch.nn.Linear(1, 1)
        with RUNNER._ModelForwardCounter(model) as counter:
            for _ in range(4):
                model(torch.ones(1, 1))
        self.assertEqual(counter.calls, 4)

    def test_source_image_fingerprint_covers_unprocessed_and_duplicate_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = root / "first.png"
            later = root / "later.png"
            first.write_bytes(b"first")
            later.write_bytes(b"later")
            dialogs = [{"image_path": str(first)},
                       {"image_path": str(later)},
                       {"image_path": str(first)}]
            fingerprint, count = RUNNER.source_image_fingerprint(dialogs, root)
            self.assertEqual(count, 2)
            self.assertEqual(fingerprint,
                             RUNNER.source_image_fingerprint(dialogs[:2], root)[0])
            later.write_bytes(b"mutated before processing")
            self.assertNotEqual(fingerprint,
                                RUNNER.source_image_fingerprint(dialogs, root)[0])

    def test_context_cap_and_server_restoration_contract(self):
        runner = SimpleNamespace(model=SimpleNamespace(
            config=SimpleNamespace(text_config=SimpleNamespace(
                max_position_embeddings=4096))))
        server = SimpleNamespace(max_new_tokens=1024)
        self.assertEqual(RUNNER._effective_cap(
            runner, server, 2203)["effective_max_new_tokens"], 1024)
        clipped = RUNNER._effective_cap(runner, server, 3100)
        self.assertEqual(clipped["effective_max_new_tokens"], 997)
        self.assertTrue(clipped["generation_cap_clamped_by_context"])
        self.assertEqual(RUNNER._effective_cap(
            runner, server, 4096)["effective_max_new_tokens"], 1)
        native = RUNNER._effective_cap(runner, server, 4097)
        self.assertEqual(native["context_policy_id"],
                         "native_overflow_no_truncation_v1")
        self.assertEqual(native["effective_max_new_tokens"], 1024)
        self.assertEqual(native["input_overflow_tokens"], 1)
        self.assertEqual(native["context_remaining_positions_before_request"],
                         -1)
        self.assertEqual(native["context_available_output_tokens"], 0)
        self.assertTrue(native["native_overflow_execution"])
        self.assertFalse(native["generation_cap_clamped_by_context"])
        long_native = RUNNER._effective_cap(runner, server, 5138)
        self.assertEqual(long_native["input_overflow_tokens"], 1042)
        self.assertEqual(long_native["effective_max_new_tokens"], 1024)
        with self.assertRaisesRegex(ValueError, "positive"):
            RUNNER._effective_cap(runner, server, 0)
        self.assertEqual(server.max_new_tokens, 1024)

    def test_native_overflow_policy_keeps_original_generated_history(self):
        questions = ["Q1", "Q2", "Q3"]
        own_answers = ["A1 " * 700, "A2 " * 700]
        prompt = RUNNER.render_prompt(questions, own_answers)
        self.assertIn("<image>\nQ1 ASSISTANT:", prompt)
        self.assertIn(own_answers[0] + "</s>USER: Q2 ASSISTANT:", prompt)
        self.assertIn(own_answers[1] + "</s>USER: Q3 ASSISTANT:", prompt)
        self.assertNotIn("Q4", prompt)
        self.assertEqual(prompt, RUNNER.render_prompt(questions, own_answers))

    def test_explicit_smoke_ids_preserve_global_ordinals(self):
        rows, _ = RUNNER.load_index(ROOT / "data/convbench/index.json")
        chosen, mode = RUNNER.select_workload(rows, 10, False, "55,494")
        self.assertEqual(mode, "explicit_source_ids")
        self.assertEqual([row["source_id"] for row in chosen], ["55", "494"])
        self.assertEqual([row["global_conversation_ordinal"]
                          for row in chosen], [54, 492])
        self.assertEqual(RUNNER.method_order(492),
                         RUNNER.method_order(chosen[1][
                             "global_conversation_ordinal"]))
        with self.assertRaises(ValueError):
            RUNNER.select_workload(rows, 10, False, "55,131")

    def test_immutable_resume_rejects_corruption_and_duplicates(self):
        cid = "convbench:1"
        index_sha = "a" * 64
        rows = [{"turn_id": t, "method_key": method,
                 "prediction": "answer"}
                for t in (1, 2, 3) for method in RUNNER.METHOD_KEYS]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "conversation.json"
            image = Path(folder) / "source.png"
            image.write_bytes(b"original image")
            payload = {"schema_version": RUNNER.SCHEMA_VERSION,
                       "conversation_id": cid, "index_sha256": index_sha,
                       "max_new_tokens": 1024, "rows": rows,
                       "image_path": str(image),
                       "image_file_sha256": RUNNER.sha256_file(image)}
            RUNNER._write_immutable(path, payload)
            self.assertEqual(len(RUNNER._validate_artifact(
                path, cid, index_sha, 1024, image)["rows"]), 12)
            image.write_bytes(b"changed image")
            with self.assertRaisesRegex(ValueError, "source image SHA256 mismatch"):
                RUNNER._validate_artifact(path, cid, index_sha, 1024, image)
            image.write_bytes(b"original image")
            with self.assertRaises(FileExistsError):
                RUNNER._write_immutable(path, payload)
            corrupt = json.loads(path.read_text())
            corrupt["rows"][0]["prediction"] = "tampered"
            path.write_text(json.dumps(corrupt))
            with self.assertRaisesRegex(ValueError, "corrupt"):
                RUNNER._validate_artifact(path, cid, index_sha, 1024, image)
            corrupt["rows"][1]["method_key"] = corrupt["rows"][0][
                "method_key"]
            corrupt["artifact_content_sha256"] = RUNNER.stable_json_sha256(
                {k: v for k, v in corrupt.items()
                 if k != "artifact_content_sha256"})
            path.write_text(json.dumps(corrupt))
            with self.assertRaisesRegex(ValueError, "duplicate/missing"):
                RUNNER._validate_artifact(path, cid, index_sha, 1024, image)

    def test_request_checkpoints_are_ordered_immutable_and_history_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "image.png"
            image.write_bytes(b"image bytes")
            cid = "convbench:7"
            index_sha, config_sha = "a" * 64, "b" * 64
            image_sha = RUNNER.sha256_file(image)
            dialog = {
                "conversation_id": cid, "image_path": str(image),
                "turns": [{"turn_id": i, "question": f"Q{i}?",
                           "reference_answer": f"reference {i}"}
                          for i in (1, 2, 3)],
            }
            order = RUNNER.method_order(0)

            def checkpoint(request_index, prior_answers):
                turn_id = request_index // 4 + 1
                method = order[request_index % 4]
                prompt = RUNNER.render_prompt(
                    [t["question"] for t in dialog["turns"][:turn_id]],
                    prior_answers)
                row = {
                    "schema_version": RUNNER.SCHEMA_VERSION,
                    "index_sha256": index_sha, "conversation_id": cid,
                    "turn_id": turn_id, "method_key": method,
                    "method_order": list(order),
                    "question": dialog["turns"][turn_id - 1]["question"],
                    "reference_answer": dialog["turns"][turn_id - 1][
                        "reference_answer"],
                    "previous_answers": list(prior_answers),
                    "prompt": prompt, "prompt_sha256": RUNNER._sha(prompt),
                    "history_policy": "same_method_generated_answers",
                    "persistence_source_request": False,
                    "runtime_success": True, "first_token_success": True,
                    "truncation_applied": False,
                    "context_policy_id": RUNNER.CONTEXT_POLICY_ID,
                    "prediction": "answer", "first_token_id": 10,
                    "generated_tokens": 1,
                }
                return {
                    "schema_version": RUNNER.REQUEST_CHECKPOINT_SCHEMA,
                    "conversation_id": cid, "request_index": request_index,
                    "turn_id": turn_id, "method_key": method,
                    "method_order": list(order),
                    "index_sha256": index_sha,
                    "run_config_sha256": config_sha,
                    "source_image_path": str(image),
                    "source_image_sha256": image_sha,
                    "row": row, "store_manifest": None,
                    "persistence": None,
                }

            first = checkpoint(0, [])
            RUNNER._write_immutable(RUNNER._checkpoint_path(root, cid, 0), first)
            loaded = RUNNER._load_request_checkpoints(
                root, dialog, index_sha, config_sha, image_sha, order)
            self.assertEqual(list(loaded), [0])
            self.assertEqual(RUNNER._validate_checkpoint_row(
                loaded[0], dialog, dialog["turns"][0], order[0], [],
                index_sha, False, None)["prediction"], "answer")
            with self.assertRaisesRegex(ValueError, "causal prompt|row/history mismatch"):
                RUNNER._validate_checkpoint_row(
                    loaded[0], dialog, dialog["turns"][0], order[0],
                    ["wrong history"], index_sha, False, None)
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                RUNNER._load_request_checkpoints(
                    root, dialog, index_sha, "c" * 64, image_sha, order)
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                RUNNER._load_request_checkpoints(
                    root, dialog, index_sha, config_sha, "d" * 64, order)
            RUNNER._write_immutable(RUNNER._checkpoint_path(root, cid, 2),
                                    checkpoint(2, []))
            with self.assertRaisesRegex(ValueError, "checkpoint gap"):
                RUNNER._load_request_checkpoints(
                    root, dialog, index_sha, config_sha, image_sha, order)
            second_path = RUNNER._checkpoint_path(root, cid, 1)
            RUNNER._write_immutable(second_path, checkpoint(1, []))
            tampered = json.loads(second_path.read_text())
            tampered["row"]["prediction"] = "tampered"
            second_path.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError, "corrupt request checkpoint"):
                RUNNER._load_request_checkpoints(
                    root, dialog, index_sha, config_sha, image_sha, order)

    def test_committed_store_requires_matching_files_and_sampled_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder) / "store"
            store.mkdir()
            for name, data in (("meta.json", b"{}"),
                               ("visionzip_layout.pt", b"layout"),
                               ("kv.bin", b"payload")):
                (store / name).write_bytes(data)
            sizes = {p.name: p.stat().st_size for p in store.iterdir()}
            manifest = {
                "file_sizes": sizes,
                "meta_sha256": RUNNER.sha256_file(store / "meta.json"),
                "layout_sha256": RUNNER.sha256_file(
                    store / "visionzip_layout.pt"),
                "prefix_kv_sample_sha256": RUNNER._sampled_store_sha256(
                    store, sizes),
            }
            RUNNER._validate_committed_store(store, manifest)
            (store / "kv.bin").write_bytes(b"payloAd")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                RUNNER._validate_committed_store(store, manifest)
            (store / "kv.bin").unlink()
            with self.assertRaisesRegex(ValueError, "file set changed"):
                RUNNER._validate_committed_store(store, manifest)


if __name__ == "__main__":
    unittest.main()

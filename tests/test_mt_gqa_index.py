"""CPU-only tests for the deterministic MT-GQA reconstruction."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from mmimpress.mt_gqa import (
    BENCHMARK_TYPE,
    DISCLAIMER,
    EXPECTED_DIALOGUES,
    EXPECTED_DUPLICATE_IMAGES,
    EXPECTED_SELECTED_QUESTIONS,
    EXPECTED_SOURCE_SHA256,
    METACOMPRESS_COMMIT,
    METACOMPRESS_README_SHA256,
    build_artifacts,
    canonical_json_bytes,
    load_gqa_questions,
    mt_gqa_prior_history_text,
    mt_gqa_prompt,
    reconstruct_dialogues,
    resolve_image_path,
    stable_rank,
    validate_artifact_directory,
    validate_dialogues,
    workload_sha256,
    write_artifacts_no_clobber,
)


ROOT = Path(__file__).resolve().parents[1]
FULL_QUESTIONS = (
    ROOT.parent / "SparseVLMs" / "playground" / "data" / "eval" / "gqa"
    / "data" / "testdev_balanced_questions.json"
)
FULL_IMAGES = FULL_QUESTIONS.parent / "images"


def _source_row(image_id: str, question: str, answer: str = "answer") -> dict:
    return {"imageId": image_id, "question": question, "answer": answer}


class SyntheticMTGQATests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        for image_id in ("image_a", "image_b"):
            (self.images / f"{image_id}.jpg").write_bytes(b"fixture")

        # Source rows are deliberately interleaved.  Membership must preserve
        # each image's occurrence order, not global adjacency.
        self.source = {
            "a1": _source_row("image_a", "Alpha one?", "a1"),
            "b1": _source_row("image_b", "Beta one?", "b1"),
            "a2": _source_row("image_a", "Alpha two?", "a2"),
            "a3": _source_row("image_a", "Alpha three?", "a3"),
            "b2": _source_row("image_b", "Beta two?", "b2"),
            # Exact normalized repeat across different image_a dialogues is
            # retained.  It must never occur inside one dialogue.
            "a4": _source_row("image_a", "  ALPHA   ONE? ", "a4"),
            "a5": _source_row("image_a", "Alpha five?", "a5"),
            "b3": _source_row("image_b", "Beta three?", "b3"),
            "a6": _source_row("image_a", "Alpha six?", "a6"),
            "a7": _source_row("image_a", "unused remainder", "a7"),
            "b4": _source_row("image_b", "unused beta remainder", "b4"),
        }
        self.questions = self.root / "questions.json"
        self.questions.write_text(json.dumps(self.source), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def _build(self, seed: int = 1234):
        return build_artifacts(
            self.questions, self.images, seed=seed, strict_canonical=False)

    def test_source_order_contiguous_membership_and_floor_quota(self):
        artifacts, summary = self._build()
        dialogs = artifacts["dialogues.json"]["dialogues"]
        self.assertEqual(summary["n_dialogues"], 3)
        self.assertEqual(summary["n_turns"], 9)

        by_image = defaultdict(list)
        for dialog in dialogs:
            by_image[dialog["image_id"]].append(dialog)
            self.assertEqual([turn["turn_id"] for turn in dialog["turns"]],
                             [1, 2, 3])
            self.assertEqual(len({turn["question_id"]
                                  for turn in dialog["turns"]}), 3)

        a_qids = [[turn["question_id"] for turn in dialog["turns"]]
                  for dialog in by_image["image_a"]]
        b_qids = [[turn["question_id"] for turn in dialog["turns"]]
                  for dialog in by_image["image_b"]]
        self.assertEqual(a_qids, [["a1", "a2", "a3"],
                                  ["a4", "a5", "a6"]])
        self.assertEqual(b_qids, [["b1", "b2", "b3"]])
        self.assertNotIn("a7", {qid for group in a_qids for qid in group})
        self.assertNotIn("b4", {qid for group in b_qids for qid in group})

        stats = artifacts["dataset_stats.json"]
        self.assertEqual(stats["discarded_questions"], 2)
        self.assertEqual(stats["duplicates"]["selected_duplicate_extras"], 1)
        self.assertEqual(
            stats["duplicates"][
                "within_dialogue_normalized_question_duplicates"], 0)

    def test_seed_changes_only_image_order_not_membership(self):
        one, _ = self._build(seed=1234)
        two, _ = self._build(seed=3)

        def memberships(payload):
            return {
                (dialog["image_id"], dialog["image_dialogue_index"]):
                    tuple(turn["question_id"] for turn in dialog["turns"])
                for dialog in payload["dialogues.json"]["dialogues"]
            }

        self.assertEqual(memberships(one), memberships(two))
        self.assertNotEqual(
            [d["image_id"] for d in one["dialogues.json"]["dialogues"]],
            [d["image_id"] for d in two["dialogues.json"]["dialogues"]],
        )

    def test_byte_determinism_and_workload_hash_contract(self):
        first, first_summary = self._build()
        second, second_summary = self._build()
        for name in first:
            self.assertEqual(canonical_json_bytes(first[name]),
                             canonical_json_bytes(second[name]))
        self.assertEqual(first_summary, second_summary)
        dialogs = first["dialogues.json"]["dialogues"]
        manual = "".join(
            f"{dialog['dialog_id']}\t{turn['turn_id']}\t{turn['question_id']}\n"
            for dialog in dialogs for turn in dialog["turns"])
        import hashlib
        self.assertEqual(workload_sha256(dialogs),
                         hashlib.sha256(manual.encode()).hexdigest())
        self.assertEqual(first["config.json"]["workload_sha256"],
                         workload_sha256(dialogs))

    def test_causal_teacher_forced_prompt_has_no_future_leakage(self):
        artifacts, _ = self._build()
        dialog = next(
            row for row in artifacts["dialogues.json"]["dialogues"]
            if row["image_id"] == "image_a"
            and row["image_dialogue_index"] == 1)

        self.assertEqual(mt_gqa_prior_history_text(dialog, 1), "")
        turn1 = mt_gqa_prompt(dialog, 1)
        self.assertTrue(turn1.startswith("USER: <image>\n"))
        self.assertTrue(turn1.endswith("short phrase. ASSISTANT:"))
        self.assertIn("Current question Q1: Alpha one?", turn1)
        self.assertNotIn("Alpha two?", turn1)
        self.assertNotIn("Alpha three?", turn1)
        self.assertNotIn("A1: a1", turn1)

        turn2 = mt_gqa_prompt(dialog, 2)
        self.assertIn("Q1: Alpha one?\nA1: a1", turn2)
        self.assertIn("Current question Q2: Alpha two?", turn2)
        self.assertNotIn("Alpha three?", turn2)
        self.assertNotIn("A2: a2", turn2)

        turn3 = mt_gqa_prompt(dialog, 3)
        self.assertIn("Q1: Alpha one?\nA1: a1", turn3)
        self.assertIn("Q2: Alpha two?\nA2: a2", turn3)
        self.assertIn("Current question Q3: Alpha three?", turn3)
        self.assertNotIn("A3: a3", turn3)
        self.assertIn("single word or short phrase", turn3)
        self.assertTrue(turn3.endswith("ASSISTANT:"))

    def test_prompt_rejects_noncanonical_turn_id(self):
        artifacts, _ = self._build()
        dialog = artifacts["dialogues.json"]["dialogues"][0]
        for bad_turn_id in (0, 4, True, "1"):
            with self.subTest(turn_id=bad_turn_id):
                with self.assertRaises(ValueError):
                    mt_gqa_prompt(dialog, bad_turn_id)

    def test_dialogue_image_path_helper_matches_string_path(self):
        artifacts, _ = self._build()
        dialog = artifacts["dialogues.json"]["dialogues"][0]
        self.assertEqual(resolve_image_path(dialog),
                         resolve_image_path(dialog["image_path"]))

    def test_empty_workload_and_gold_answer_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            validate_dialogues([], check_images=False)
        artifacts, _ = self._build()
        dialog = copy.deepcopy(artifacts["dialogues.json"]["dialogues"][0])
        dialog["turns"][0]["answers"] = ["  "]
        with self.assertRaisesRegex(ValueError, "gold answer"):
            validate_dialogues([dialog], check_images=False)

    def test_stable_rank_has_frozen_portable_value(self):
        self.assertEqual(
            stable_rank(1234, "image-order", "n161313"),
            "ecf3230b766f218c1af8fca2117b2b9558badb79fb9fd47f9c0e345a20e5c287",
        )

    def test_within_dialogue_normalized_duplicate_fails_closed(self):
        bad = {
            "q1": _source_row("image_a", "Same question?"),
            "q2": _source_row("image_a", " SAME   QUESTION? "),
            "q3": _source_row("image_a", "Different question?"),
        }
        with self.assertRaisesRegex(ValueError, "source-order triple"):
            reconstruct_dialogues(bad, self.images)

    def test_missing_image_fails_closed(self):
        missing = {"x1": _source_row("absent", "one"),
                   "x2": _source_row("absent", "two"),
                   "x3": _source_row("absent", "three")}
        with self.assertRaises(FileNotFoundError):
            reconstruct_dialogues(missing, self.images)

    def test_atomic_no_clobber_and_deterministic_validation(self):
        artifacts, _ = self._build()
        output = self.root / "output"
        written = write_artifacts_no_clobber(output, artifacts)
        self.assertEqual(set(written), set(artifacts))
        report = validate_artifact_directory(
            output,
            source_questions=self.questions,
            image_dir=self.images,
            strict_canonical=False,
        )
        self.assertTrue(report["deterministic_rebuild_match"])
        with self.assertRaises(FileExistsError):
            write_artifacts_no_clobber(output, artifacts)

        before = {name: (output / name).read_bytes() for name in artifacts}
        with self.assertRaises(FileExistsError):
            write_artifacts_no_clobber(output, copy.deepcopy(artifacts))
        after = {name: (output / name).read_bytes() for name in artifacts}
        self.assertEqual(before, after)

    def test_provenance_never_claims_official_identity(self):
        artifacts, _ = self._build()
        provenance = artifacts["dataset_provenance.json"]
        self.assertEqual(provenance["benchmark_type"], BENCHMARK_TYPE)
        self.assertIs(provenance["official_benchmark_identity_claimed"], False)
        self.assertEqual(provenance["disclaimer"], DISCLAIMER)
        evidence = provenance["metacompress_repository_evidence"]
        self.assertEqual(evidence["commit"], METACOMPRESS_COMMIT)
        self.assertEqual(evidence["readme_sha256"],
                         METACOMPRESS_README_SHA256)
        self.assertEqual(evidence["evidence_scope"], "README-only")
        self.assertEqual(evidence["tree_files"], ["README.md"])
        self.assertFalse(evidence["official_dialogue_artifact_observed"])

    def test_noncanonical_json_and_symlink_output_fail_closed(self):
        artifacts, _ = self._build()
        output = self.root / "canonical"
        write_artifacts_no_clobber(output, artifacts)
        config_path = output / "config.json"
        config_value = json.loads(config_path.read_text(encoding="utf-8"))
        config_path.write_text(json.dumps(config_value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "noncanonical JSON"):
            validate_artifact_directory(
                output,
                source_questions=self.questions,
                image_dir=self.images,
                strict_canonical=False,
            )

        real_output = self.root / "real-output"
        real_output.mkdir()
        symlink_output = self.root / "linked-output"
        symlink_output.symlink_to(real_output, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            write_artifacts_no_clobber(symlink_output, artifacts)


@unittest.skipUnless(FULL_QUESTIONS.is_file() and FULL_IMAGES.is_dir(),
                     "local frozen GQA testdev-balanced source is unavailable")
class FullLocalMTGQAIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts, cls.summary = build_artifacts(
            FULL_QUESTIONS, FULL_IMAGES, seed=1234, strict_canonical=True)
        cls.dialogs = cls.artifacts["dialogues.json"]["dialogues"]
        cls.stats = cls.artifacts["dataset_stats.json"]

    def test_exact_full_contract(self):
        self.assertEqual(self.summary["source_questions_sha256"],
                         EXPECTED_SOURCE_SHA256)
        self.assertEqual(self.summary["n_dialogues"], EXPECTED_DIALOGUES)
        self.assertEqual(self.summary["n_turns"], EXPECTED_SELECTED_QUESTIONS)
        self.assertEqual(self.summary["n_unique_images"], 398)
        self.assertEqual(self.stats["discarded_questions"], 395)
        self.assertEqual(self.stats["duplicates"]["source_duplicate_extras"], 8)
        self.assertEqual(self.stats["duplicates"]["selected_duplicate_extras"],
                         8)
        self.assertEqual(
            tuple(self.stats["duplicates"]["images_with_selected_duplicates"]),
            EXPECTED_DUPLICATE_IMAGES,
        )
        self.assertEqual(
            self.stats["validation"][
                "within_dialogue_normalized_question_duplicates"], 0)
        self.assertEqual(self.stats["validation"]["duplicate_triplets"], 0)
        self.assertEqual(self.stats["validation"]["unique_question_ids"],
                         EXPECTED_SELECTED_QUESTIONS)

    def test_source_order_membership_is_exact(self):
        rows = load_gqa_questions(FULL_QUESTIONS)
        expected = defaultdict(list)
        for qid, row in rows.items():
            expected[row["image_id"]].append(qid)
        observed = defaultdict(list)
        for dialog in self.dialogs:
            observed[dialog["image_id"]].extend(
                turn["question_id"] for turn in dialog["turns"])
        self.assertEqual(set(observed), set(expected))
        for image_id, qids in expected.items():
            self.assertEqual(observed[image_id], qids[:(len(qids) // 3) * 3])

    def test_all_398_images_are_resolved(self):
        quotas = self.stats["dialogue_quota_by_image"]
        report = validate_dialogues(
            self.dialogs, expected_quotas=quotas, check_images=True)
        self.assertTrue(report["all_images_exist"])
        self.assertEqual(report["unique_images"], 398)

    def test_frozen_canonical_hashes(self):
        self.assertEqual(
            self.summary["dialogues_sha256"],
            "2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224",
        )
        self.assertEqual(
            self.summary["workload_sha256"],
            "0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62",
        )


if __name__ == "__main__":
    unittest.main()

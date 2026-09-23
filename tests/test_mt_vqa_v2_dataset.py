"""CPU-only tests for the reconstructed three-turn MT-VQA-v2 workload."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from mmimpress.mt_vqa_v2 import (
    BENCHMARK_TYPE,
    DEFAULT_SEED,
    EXPECTED_DIALOGUES,
    EXPECTED_IMAGES,
    EXPECTED_TURNS,
    SOURCE_CONFIG_SHA256,
    SOURCE_IMAGE_MANIFEST_SHA256,
    SOURCE_IMAGE_TOTAL_BYTES,
    SOURCE_INDEX_SHA256,
    build_artifacts,
    canonical_json_bytes,
    validate_artifact_directory,
    validate_dialogues,
    workload_sha256,
    write_artifacts_no_clobber,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_INDEX = ROOT / "data/vqav2/index.json"
SOURCE_CONFIG = ROOT / "data/vqav2/config.json"


def _question(image: str, position: int) -> dict:
    return {
        "question_id": f"{image}-q{position}",
        "question": f"Question {position} for {image}?",
        "answers": [f"answer-{position}"] * 10,
    }


class SyntheticReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        rows = []
        for image in ("image_a", "image_b"):
            path = self.images / f"{image}.jpg"
            path.write_bytes((image + "-fixture").encode("ascii"))
            rows.append({
                "image_id": image,
                "image_path": str(path),
                "questions": [_question(image, position)
                              for position in range(5)],
            })
        self.source_index = self.root / "index.json"
        self.source_config = self.root / "config.json"
        self.source_index.write_text(json.dumps(rows), encoding="utf-8")
        self.source_config.write_text(json.dumps({
            "dataset": "vqav2", "seed": DEFAULT_SEED,
            "max_images": 2, "q_per_image": 5,
        }), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def _build(self):
        return build_artifacts(
            self.source_index, self.source_config,
            seed=DEFAULT_SEED, strict_canonical=False)

    def test_source_order_membership_is_questions_one_through_three(self):
        artifacts, summary = self._build()
        dialogues = artifacts["dialogues.json"]["dialogues"]
        self.assertEqual(summary["n_dialogues"], 2)
        self.assertEqual(summary["n_turns"], 6)
        self.assertEqual(
            [[turn["source_position"] for turn in dialog["turns"]]
             for dialog in dialogues],
            [[1, 2, 3], [1, 2, 3]],
        )
        self.assertEqual(
            [[turn["question_id"] for turn in dialog["turns"]]
             for dialog in dialogues],
            [["image_a-q1", "image_a-q2", "image_a-q3"],
             ["image_b-q1", "image_b-q2", "image_b-q3"]],
        )
        self.assertTrue(all(
            len(turn["answers"]) == 10
            for dialog in dialogues for turn in dialog["turns"]))
        stats = artifacts["dataset_stats.json"]
        self.assertEqual(stats["held_out_questions"], 2)
        self.assertEqual(stats["discarded_slice_remainder_questions"], 2)
        self.assertEqual(stats["question_overlap_between_dialogues"], 0)
        self.assertFalse(stats["dialogue_overlap"])

    def test_determinism_workload_hash_and_reconstructed_identity(self):
        first, first_summary = self._build()
        second, second_summary = self._build()
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(
            {name: canonical_json_bytes(value) for name, value in first.items()},
            {name: canonical_json_bytes(value) for name, value in second.items()},
        )
        dialogues = first["dialogues.json"]["dialogues"]
        self.assertEqual(first["config.json"]["workload_sha256"],
                         workload_sha256(dialogues))
        self.assertEqual(first["config.json"]["benchmark_type"],
                         BENCHMARK_TYPE)
        provenance = first["dataset_provenance.json"]
        self.assertIs(provenance["official_benchmark_identity_claimed"], False)
        self.assertIs(provenance[
            "exact_metacompress_reproduction_claimed"], False)

    def test_atomic_no_clobber_rebuild_validation_and_mutation_failure(self):
        artifacts, _ = self._build()
        output = self.root / "published"
        hashes = write_artifacts_no_clobber(output, artifacts)
        self.assertEqual(set(hashes), set(artifacts))
        report = validate_artifact_directory(
            output, source_index=self.source_index,
            source_config=self.source_config, strict_canonical=False)
        self.assertTrue(report["passed"])
        with self.assertRaises(FileExistsError):
            write_artifacts_no_clobber(output, artifacts)

        dialogues_path = output / "dialogues.json"
        changed = copy.deepcopy(artifacts["dialogues.json"])
        changed["dialogues"][0]["turns"][0]["question_id"] = "mutated"
        dialogues_path.write_bytes(canonical_json_bytes(changed))
        with self.assertRaisesRegex(ValueError, "deterministic rebuild"):
            validate_artifact_directory(
                output, source_index=self.source_index,
                source_config=self.source_config, strict_canonical=False)

    def test_validation_rejects_question_reuse_and_wrong_source_position(self):
        artifacts, _ = self._build()
        dialogues = artifacts["dialogues.json"]["dialogues"]
        duplicate = copy.deepcopy(dialogues)
        duplicate[1]["turns"][0]["question_id"] = \
            duplicate[0]["turns"][0]["question_id"]
        with self.assertRaisesRegex(ValueError, "reused question"):
            validate_dialogues(duplicate, check_images=True)
        wrong_position = copy.deepcopy(dialogues)
        wrong_position[0]["turns"][0]["source_position"] = 0
        with self.assertRaisesRegex(ValueError, "source-order membership"):
            validate_dialogues(wrong_position, check_images=True)


@unittest.skipUnless(SOURCE_INDEX.is_file() and SOURCE_CONFIG.is_file(),
                     "frozen local VQAv2 slice is unavailable")
class FrozenLocalIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts, cls.summary = build_artifacts(
            SOURCE_INDEX, SOURCE_CONFIG, seed=DEFAULT_SEED,
            strict_canonical=True)

    def test_exact_frozen_contract(self):
        self.assertEqual(self.summary["source_index_sha256"],
                         SOURCE_INDEX_SHA256)
        self.assertEqual(self.summary["n_dialogues"], EXPECTED_DIALOGUES)
        self.assertEqual(self.summary["n_turns"], EXPECTED_TURNS)
        self.assertEqual(self.summary["n_unique_images"], EXPECTED_IMAGES)
        self.assertEqual(
            self.summary["dialogues_sha256"],
            "89719b2a1187c07e3228cc76cf1e473b3c713a0dcb3d65da0c596ae81898d6ea",
        )
        self.assertEqual(
            self.summary["workload_sha256"],
            "384e39bad4e2e8d5865fe20bad7661d3cad8fe0b5ce5effbfc43ea896170cbbc",
        )

    def test_source_and_image_provenance_are_frozen(self):
        source = self.artifacts["dataset_provenance.json"]["source"]
        self.assertEqual(source["local_index_sha256"], SOURCE_INDEX_SHA256)
        self.assertEqual(source["local_config_sha256"], SOURCE_CONFIG_SHA256)
        self.assertEqual(source["referenced_image_manifest_sha256"],
                         SOURCE_IMAGE_MANIFEST_SHA256)
        self.assertEqual(source["referenced_image_total_bytes"],
                         SOURCE_IMAGE_TOTAL_BYTES)


if __name__ == "__main__":
    unittest.main()

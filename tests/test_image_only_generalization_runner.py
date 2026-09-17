import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/33_eval_image_only_generalization.py"
SPEC = importlib.util.spec_from_file_location(
    "image_only_generalization_runner_test", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class WorkloadTests(unittest.TestCase):
    def test_resolve_frozen_workloads(self):
        cases = [
            (
                "gqa_large", 1, 3, 395, 1185,
                "0d50962f0c1bac3bc6e1836978289d5fde7d60b434f55cebbc4607c4c80a797c",
                "cabec1bb1035c836839b98d72ba6a04558529d246cf55f8499ca75ea2d2f290c",
            ),
            (
                "vqav2", 1, 4, 250, 1000,
                "b83d5fa288fcb722ca073e261d3fec9086629ed0db2e568a2d5d6a24ef1589d7",
                "e341b499a968c5caba4ddffc58fdf0ccafe2e0e212b6cb280ca1b11fc9f31d18",
            ),
            (
                "textvqa", 1, 1, 500, 500,
                "b1e5ff0eaba2a45c6398e7cb90631cdc7387eff25ed0997a66374f25968a2f4d",
                "49fa0b15f406132162cba1b47245f28a560d28b4f3af76985c855d77a145a2a6",
            ),
        ]
        for (dataset, skip, questions, images, requests, index_sha,
             workload_sha) in cases:
            with self.subTest(dataset=dataset):
                result = MOD.resolve_workload(
                    ROOT / "data" / dataset / "index.json",
                    skip=skip,
                    questions=questions,
                    expected_index_sha256=index_sha,
                    expected_workload_sha256=workload_sha,
                    expected_images=images,
                    expected_questions=requests,
                )
                self.assertEqual(result["n_images"], images)
                self.assertEqual(result["n_questions"], requests)
                self.assertEqual(len(result["ordinal_by_key"]), requests)

    def test_balanced_order_uses_global_ordinal(self):
        orders = [MOD.balanced_method_order(i, seed=1234) for i in range(4)]
        self.assertEqual(len({tuple(order) for order in orders}), 4)
        for position in range(4):
            self.assertEqual(
                {order[position] for order in orders}, set(MOD.METHOD_KEYS))
        self.assertEqual(
            MOD.balanced_method_order(7, 1234),
            MOD.balanced_method_order(7, 1234),
        )


class SafetyTests(unittest.TestCase):
    def test_owned_temp_path_is_exact_image_leaf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = (Path(directory) / "owned").resolve()
            MOD._claim_temp_root(root, "test-id", "gqa_large")
            (root / "payload").mkdir()
            good = root / "payload" / "image-1"
            self.assertEqual(
                MOD.assert_owned_temp_path(
                    good, root, "test-id", "image-1"),
                good,
            )
            with self.assertRaises(ValueError):
                MOD.assert_owned_temp_path(
                    root / "payload", root, "test-id")
            with self.assertRaises(ValueError):
                MOD.assert_owned_temp_path(
                    root / "payload" / "image-1" / "nested",
                    root,
                    "test-id",
                )
            with self.assertRaises(ValueError):
                MOD.assert_owned_temp_path(
                    good, root, "test-id", "another-image")

    def test_exclusive_json_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            MOD._write_exclusive_json(path, {"complete": True})
            self.assertEqual(json.loads(path.read_text()), {"complete": True})
            with self.assertRaises(FileExistsError):
                MOD._write_exclusive_json(path, {"complete": False})

    @staticmethod
    def _resume_fixture(directory: Path):
        run = directory / "run"
        images = run / "images"
        shards = run / "shards"
        images.mkdir(parents=True)
        shards.mkdir()
        image_id = "image-1"
        qid = "question-1"
        seed = 1234
        ordinal = 0
        order = list(MOD.balanced_method_order(ordinal, seed))
        store_id = "a" * 64
        visual_bytes = 1000
        rows = []
        for position, method in enumerate(order):
            budget = MOD.METHODS[method]["budget"]
            cached = method != "recompute"
            row = {
                "dataset": "gqa_large",
                "image_id": image_id,
                "image_ordinal": 0,
                "question_id": qid,
                "global_request_ordinal": ordinal,
                "method_key": method,
                "method": MOD.METHODS[method]["label"],
                "budget": budget,
                "method_order": order,
                "method_order_position": position,
                "suffix_ids_sha256": "b" * 64,
                "end_to_end_ttft_ms": 1.0,
                "request_e2e_ms": 2.0,
                "same_physical_store_id": store_id if cached else None,
                "store_id": store_id if cached else None,
                "ssd_read_bytes": (0 if not cached else
                                   visual_bytes if method == "fullload" else 250),
                "ssd_read_preads": 0 if not cached else 1,
                "vision_forward_count": 0 if cached else 1,
                "page_cache_conditioning_excluded_from_ttft": cached,
                "n_chunks_total": 8,
                "selected_chunk_ids_per_layer": (
                    [MOD.prefix_chunk_ids(8, budget)] if method.startswith("prefix")
                    else None),
                "static_score_calls": 0,
                "query_score_calls": 0,
                "diversity_calls": 0,
            }
            rows.append(row)
        artifact = {
            "schema_version": MOD.SCHEMA_VERSION,
            "experiment_id": "experiment-1",
            "dataset": "gqa_large",
            "image_id": image_id,
            "image_ordinal": 0,
            "shard_index": 0,
            "index_sha256": "c" * 64,
            "workload_sha256": "d" * 64,
            "store_manifest": {
                "meta_sha256": store_id,
                "visual_kv_bytes": visual_bytes,
            },
            "rows": rows,
        }
        artifact["artifact_content_sha256"] = MOD._json_hash(artifact)
        artifact_path = images / f"{image_id}.json"
        artifact_path.write_text(json.dumps(artifact))
        workload = {
            "index_sha256": "c" * 64,
            "workload_sha256": "d" * 64,
            "skip": 1,
            "questions": 1,
            "ordinal_by_key": {(image_id, qid): ordinal},
        }
        questions = [{"question_id": qid, "question": "synthetic"}]
        return artifact_path, artifact, workload, questions

    def test_resume_artifact_is_checked_before_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._resume_fixture(Path(directory))
            artifact_path, artifact, workload, questions = fixture
            loaded = MOD._validate_resume_artifact(
                artifact_path, experiment_id="experiment-1",
                dataset="gqa_large", image_id="image-1", image_ordinal=0,
                shard_index=0, seed=1234, workload=workload,
                eval_questions=questions)
            self.assertEqual(loaded["artifact_content_sha256"],
                             artifact["artifact_content_sha256"])

            # A self-consistently re-hashed but semantically invalid ReComp
            # row must not be accepted as a completed image.
            recomp = next(row for row in artifact["rows"]
                          if row["method_key"] == "recompute")
            recomp["ssd_read_bytes"] = 1
            artifact["artifact_content_sha256"] = MOD._json_hash({
                key: value for key, value in artifact.items()
                if key != "artifact_content_sha256"
            })
            artifact_path.write_text(json.dumps(artifact))
            with self.assertRaisesRegex(ValueError, "ReComp"):
                MOD._validate_resume_artifact(
                    artifact_path, experiment_id="experiment-1",
                    dataset="gqa_large", image_id="image-1",
                    image_ordinal=0, shard_index=0, seed=1234,
                    workload=workload, eval_questions=questions)


class CoverageTests(unittest.TestCase):
    def test_prefix_coverage_uses_same_budget_rounding(self):
        # Six real tokens followed by two stable-tail separators, split into four
        # two-row physical chunks. A 25% budget retains only the first chunk.
        layout = {
            "token_score_original": torch.tensor(
                [6., 5., 4., 3., 2., 1., float("inf"), float("inf")]
            ),
            "stored_to_original": torch.arange(8),
        }
        meta = {
            "newline_idx": [6, 7],
            "n_chunks_per_layer": 4,
            "chunk_size": 2,
        }
        result = MOD._importance_coverage(layout, meta, 0.25)
        self.assertEqual(result["selected_importance_mass"], 11.0)
        self.assertEqual(result["total_importance_mass"], 21.0)
        self.assertAlmostEqual(
            result["importance_mass_coverage"], 11 / 21)
        self.assertEqual(
            result["actual_selected_normal_chunk_fraction"], 0.25)


if __name__ == "__main__":
    unittest.main()

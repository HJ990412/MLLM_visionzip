"""CPU-only contracts for the cross-dataset ImageOnly experiment.

These tests intentionally use synthetic indexes and scores.  They exercise the
pieces that must be correct before an expensive model run: immutable workload
slicing, shard-invariant balanced ordering, exact sequential-prefix budgets,
temporary-store deletion ownership, and paired image-cluster bootstrap.
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from mmimpress.cvpr25 import budget_chunk_count, prefix_chunk_ids


ROOT = Path(__file__).resolve().parent.parent


def _load_script(filename: str, module_name: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUNNER = _load_script(
    "33_eval_image_only_generalization.py",
    "mmimpress_image_only_generalization_runner_test",
)
ANALYZER = _load_script(
    "34_analyze_image_only_generalization.py",
    "mmimpress_image_only_generalization_analyzer_test",
)


def _questions(image: int, count: int = 5) -> list[dict]:
    return [
        {
            "question_id": f"q-{image}-{ordinal}",
            "question": f"synthetic question {image}-{ordinal}",
            "answers": [f"a-{image}-{ordinal}"],
        }
        for ordinal in range(count)
    ]


class WorkloadResolutionTests(unittest.TestCase):
    def _resolve(self, *, per_image: int, selected: int):
        entries = [
            {
                "image_id": f"image-{image}",
                "image_path": f"never-read-{image}.png",
                "questions": _questions(image, per_image),
            }
            for image in range(3)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            original = json.dumps(entries)
            path.write_text(original)
            resolved = RUNNER.resolve_workload(
                path,
                skip=1,
                questions=selected,
                expected_images=3,
                expected_questions=3 * selected,
            )
            self.assertEqual(path.read_text(), original)
            return resolved

    def test_gqa_slice_is_exactly_questions_one_through_three(self):
        resolved = self._resolve(per_image=5, selected=3)
        self.assertEqual(
            [row["question_id"] for row in resolved["request_rows"]],
            [f"q-{image}-{question}" for image in range(3)
             for question in (1, 2, 3)],
        )
        self.assertEqual(resolved["questions_per_image"], {
            "min": 3, "mean": 3.0, "max": 3,
        })

    def test_vqav2_and_textvqa_slices_are_exact(self):
        vqa = self._resolve(per_image=5, selected=4)
        text = self._resolve(per_image=2, selected=1)
        self.assertEqual(
            [row["local_question_ordinal"] for row in vqa["request_rows"]],
            [0, 1, 2, 3] * 3,
        )
        self.assertEqual(
            [row["question_id"] for row in text["request_rows"]],
            [f"q-{image}-1" for image in range(3)],
        )

    def test_expected_identity_mismatch_fails_closed(self):
        entries = [{
            "image_id": "image-0",
            "image_path": "never-read.png",
            "questions": _questions(0),
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            path.write_text(json.dumps(entries))
            with self.assertRaisesRegex(ValueError, "index SHA256 mismatch"):
                RUNNER.resolve_workload(
                    path, skip=1, questions=3,
                    expected_index_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "workload SHA256 mismatch"):
                RUNNER.resolve_workload(
                    path, skip=1, questions=3,
                    expected_workload_sha256="0" * 64)

    def test_duplicate_question_id_is_rejected_even_outside_eval_slice(self):
        entries = [
            {"image_id": "a", "image_path": "a.png",
             "questions": _questions(0)},
            {"image_id": "b", "image_path": "b.png",
             "questions": _questions(1)},
        ]
        entries[1]["questions"][4]["question_id"] = "q-0-0"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            path.write_text(json.dumps(entries))
            with self.assertRaisesRegex(ValueError, "duplicate question ID"):
                RUNNER.resolve_workload(path, skip=1, questions=1)


class BalancedOrderTests(unittest.TestCase):
    def test_rotation_is_deterministic_balanced_and_shard_invariant(self):
        methods = RUNNER.METHOD_KEYS
        first = [RUNNER.balanced_method_order(i, seed=1234)
                 for i in range(20)]
        second = [RUNNER.balanced_method_order(i, seed=1234)
                  for i in range(20)]
        self.assertEqual(first, second)
        for order in first:
            self.assertEqual(len(order), len(methods))
            self.assertEqual(set(order), set(methods))
        for position in range(len(methods)):
            counts = {
                method: sum(order[position] == method for order in first)
                for method in methods
            }
            self.assertEqual(set(counts.values()), {5})

        # Re-grouping the same canonical ordinals into different shards must
        # not affect an order derived from the global request ordinal.
        by_four = [order for start in range(0, 20, 4)
                   for order in first[start:start + 4]]
        by_seven = [order for start in range(0, 20, 7)
                    for order in first[start:start + 7]]
        self.assertEqual(by_four, by_seven)

    def test_invalid_method_sets_are_rejected(self):
        for methods in ((), ("recompute", "recompute"), ("unknown",)):
            with self.assertRaises(ValueError):
                RUNNER.balanced_method_order(0, methods=methods)


class SequentialPrefixTests(unittest.TestCase):
    def test_first_k_is_exact_and_nested_for_all_chunk_counts(self):
        for n_chunks in range(1, 101):
            ids25 = prefix_chunk_ids(n_chunks, 0.25)
            ids45 = prefix_chunk_ids(n_chunks, 0.45)
            k25 = budget_chunk_count(n_chunks, 0.25)
            k45 = budget_chunk_count(n_chunks, 0.45)
            self.assertEqual(ids25, list(range(k25)))
            self.assertEqual(ids45, list(range(k45)))
            self.assertTrue(set(ids25).issubset(ids45))
            self.assertLessEqual(k25, k45)
            self.assertLessEqual(k45, n_chunks)

    def test_python_ties_to_even_and_clamping_contract(self):
        self.assertEqual(budget_chunk_count(34, 0.25), 8)
        self.assertEqual(budget_chunk_count(34, 0.45), 15)
        self.assertEqual(prefix_chunk_ids(34, 0.25), list(range(8)))
        self.assertEqual(prefix_chunk_ids(34, 0.45), list(range(15)))
        self.assertEqual(budget_chunk_count(1, 0.0), 1)
        self.assertEqual(budget_chunk_count(3, 2.0), 3)


class OwnedTemporaryPathTests(unittest.TestCase):
    @staticmethod
    def _claim(root: Path, experiment_id: str = "experiment-123") -> None:
        root.mkdir()
        (root / RUNNER.OWNER_FILE).write_text(json.dumps({
            "schema_version": RUNNER.SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "purpose": "temporary_visual_kv_only",
            "dataset": "synthetic",
        }))
        (root / "payload").mkdir()

    def test_accepts_only_owned_image_payload_descendant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "owned"
            self._claim(root)
            candidate = root / "payload" / "image-7"
            self.assertEqual(
                RUNNER.assert_owned_temp_path(
                    candidate, root, "experiment-123"),
                candidate.resolve(),
            )

    def test_rejects_root_outside_relative_and_wrong_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "owned"
            self._claim(root)
            for candidate in (root, base / "outside"):
                with self.assertRaises(ValueError):
                    RUNNER.assert_owned_temp_path(
                        candidate, root, "experiment-123")
            with self.assertRaises(ValueError):
                RUNNER.assert_owned_temp_path(
                    Path("relative/path"), root, "experiment-123")
            with self.assertRaisesRegex(ValueError, "ownership mismatch"):
                RUNNER.assert_owned_temp_path(
                    root / "payload" / "image-7", root, "wrong-id")

    def test_rejects_symlink_candidate_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "owned"
            self._claim(root)
            real = root / "payload" / "real"
            real.mkdir()
            link = root / "payload" / "link"
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                RUNNER.assert_owned_temp_path(link, root, "experiment-123")

            outside = base / "outside"
            outside.mkdir()
            parent_link = root / "payload" / "escape"
            parent_link.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                RUNNER.assert_owned_temp_path(
                    parent_link / "victim", root, "experiment-123")

    def test_rejects_broad_or_unexpected_nested_descendants(self):
        """Deletion authority is exactly one image payload, never a subtree."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "owned"
            self._claim(root)
            candidates = (
                root / "payload",
                root / "unexpected",
                root / "payload" / "nested" / "image-7",
                root / "payload" / ".." / "unexpected",
            )
            for candidate in candidates:
                with self.subTest(candidate=candidate):
                    with self.assertRaises(ValueError):
                        RUNNER.assert_owned_temp_path(
                            candidate, root, "experiment-123")


class CapacityGuardTests(unittest.TestCase):
    def test_free_space_boundary_is_fail_closed(self):
        reserve = 64 * (1024 ** 3)
        headroom = 3 * (1024 ** 3)
        exact = SimpleNamespace(
            total=2_000 * (1024 ** 3),
            used=1_000 * (1024 ** 3),
            free=reserve + headroom,
        )
        with mock.patch.object(RUNNER.shutil, "disk_usage",
                               return_value=exact):
            observed = RUNNER._capacity_guard(
                Path("/not-read"), reserve_bytes=reserve,
                extra_headroom_bytes=headroom)
        self.assertEqual(observed["disk_free_bytes"], reserve + headroom)

        below = SimpleNamespace(
            total=exact.total, used=exact.used,
            free=reserve + headroom - 1,
        )
        with mock.patch.object(RUNNER.shutil, "disk_usage",
                               return_value=below), \
             self.assertRaisesRegex(RuntimeError, "free-space guard"):
            RUNNER._capacity_guard(
                Path("/not-read"), reserve_bytes=reserve,
                extra_headroom_bytes=headroom)

    def test_df_style_96_percent_used_boundary_is_rejected(self):
        # Deliberately make ``total`` include a large reserved-block region.
        # The guard's percentage must use used/(used+available), matching df,
        # rather than used/total, which would incorrectly report only 80%.
        usage = SimpleNamespace(total=1_200, used=960, free=40)
        with mock.patch.object(RUNNER.shutil, "disk_usage",
                               return_value=usage), \
             self.assertRaisesRegex(RuntimeError, "disk-used guard"):
            RUNNER._capacity_guard(
                Path("/not-read"), reserve_bytes=0,
                extra_headroom_bytes=0)


class ImageClusterBootstrapTests(unittest.TestCase):
    def test_identical_paired_scores_have_exact_zero_delta_interval(self):
        images = ["a", "a", "b", "b"]
        scores = [0.0, 1.0, 0.25, 0.75]
        result = ANALYZER.paired_image_cluster_bootstrap(
            images, scores, scores, n_resamples=2_000, seed=77)
        self.assertEqual(result["estimate"], 0.0)
        self.assertEqual(result["delta_mean"], 0.0)
        self.assertEqual(result["ci95_low"], 0.0)
        self.assertEqual(result["ci95_high"], 0.0)
        self.assertEqual(result["cluster_unit"], "image")
        self.assertEqual(result["n_clusters"], 2)

    def test_seeded_result_is_exactly_deterministic(self):
        args = (["a", "a", "b", "c"],
                [0.0, 1.0, 0.5, 0.25],
                [0.25, 0.75, 0.5, 1.0])
        first = ANALYZER.paired_image_cluster_bootstrap(
            *args, n_resamples=1_000, seed=1234)
        second = ANALYZER.paired_image_cluster_bootstrap(
            *args, n_resamples=1_000, seed=1234)
        self.assertEqual(first, second)

    def test_resampling_carries_every_question_in_unequal_clusters(self):
        # Cluster a owns four zero-delta questions; cluster b owns one unit-
        # delta question.  Drawing two image clusters yields possible deltas
        # 0, 0.2, and 1.0.  A naive five-row bootstrap almost never reaches a
        # 97.5th percentile of 1.0, so this is a compact sampling-unit guard.
        result = ANALYZER.paired_image_cluster_bootstrap(
            ["a", "a", "a", "a", "b"],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            n_resamples=10_000,
            seed=1234,
        )
        self.assertAlmostEqual(result["estimate"], 0.2)
        self.assertEqual(result["ci95_low"], 0.0)
        self.assertEqual(result["ci95_high"], 1.0)

    def test_invalid_bootstrap_inputs_fail_closed(self):
        cases = (
            ([], [], None, {"n_resamples": 10}),
            (["a"], [0.0, 1.0], None, {"n_resamples": 10}),
            (["a"], [0.0], [0.0, 1.0], {"n_resamples": 10}),
            (["a"], [float("nan")], None, {"n_resamples": 10}),
            (["a"], [0.0], [float("inf")], {"n_resamples": 10}),
            (["a"], [0.0], None, {"n_resamples": 0}),
        )
        for images, scores_a, scores_b, kwargs in cases:
            with self.subTest(images=images, scores_a=scores_a,
                              scores_b=scores_b, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ANALYZER.paired_image_cluster_bootstrap(
                        images, scores_a, scores_b, **kwargs)


class PersistenceProfileTests(unittest.TestCase):
    def test_alias_priority_prefers_flat_aggregate_over_nested_component(self):
        profile = {
            "writer": {"kv_repack_ms": 7.0},
            "repack_ms": 19.0,
        }
        self.assertEqual(
            ANALYZER.first_profile_number(
                profile, ("repack_ms", "kv_repack_ms")),
            19.0,
        )

    def test_alias_fallback_supports_legacy_nested_profile(self):
        profile = {"writer": {"kv_repack_ms": 7.0}}
        self.assertEqual(
            ANALYZER.first_profile_number(
                profile, ("repack_ms", "kv_repack_ms")),
            7.0,
        )


if __name__ == "__main__":
    unittest.main()

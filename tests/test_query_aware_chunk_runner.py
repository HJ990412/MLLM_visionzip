import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "qa_chunk_runner", ROOT / "scripts/52_eval_query_aware_chunk_baseline.py")
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


class FakeServer:
    def __init__(self):
        self.calls = []

    def request(self, context, **kwargs):
        self.calls.append(("request", context, kwargs))
        return "full"

    def request_qa_select(self, context, **kwargs):
        self.calls.append(("request_qa_select", context, kwargs))
        return "token"

    def request_qa_chunk(self, context, **kwargs):
        self.calls.append(("request_qa_chunk", context, kwargs))
        return "chunk"

    def request_cvpr25(self, context, **kwargs):
        self.calls.append(("request_cvpr25", context, kwargs))
        return "ours"


class QueryAwareChunkRunnerTests(unittest.TestCase):
    def test_five_method_metadata_is_stable_and_distinct(self):
        self.assertEqual(
            RUNNER.METHOD_KEYS,
            ("recompute", "fullload", "qa_token25", "qa_chunk25", "ours25"))
        self.assertEqual(RUNNER.METHODS["qa_token25"]["method_id"],
                         "qa_token25")
        chunk = RUNNER.METHODS["qa_chunk25"]
        self.assertEqual(chunk["method_id"], "qa_chunk25")
        self.assertEqual(chunk["selection_granularity"], "ssd_chunk")
        self.assertEqual(chunk["chunk_score"],
                         "mean_valid_spatial_token_importance")
        self.assertEqual(chunk["physical_layout"], "raster")
        self.assertFalse(chunk["repacking"])
        self.assertEqual(RUNNER.METHODS["ours25"]["method_id"],
                         "imageonly_prefix25")

    def test_stored_dispatch_preserves_all_four_paths(self):
        server = FakeServer()
        context = object()
        suffix = object()
        expected = {
            "fullload": "full", "qa_token25": "token",
            "qa_chunk25": "chunk", "ours25": "ours",
        }
        for method, answer in expected.items():
            self.assertEqual(RUNNER.dispatch_stored(
                server, method, context, suffix, "image"), answer)
        names = [row[0] for row in server.calls]
        self.assertEqual(names, ["request", "request_qa_select",
                                 "request_qa_chunk", "request_cvpr25"])
        self.assertEqual(server.calls[0][2]["mode"], "fullload")
        self.assertEqual(server.calls[1][2]["cold"], False)
        self.assertEqual(server.calls[2][2]["cold"], False)
        ours = server.calls[3][2]
        self.assertEqual(ours["mode"], "prefix")
        self.assertEqual(ours["budget"], 0.25)
        self.assertEqual(ours["expected_prefix_layout"],
                         "visionzip_image_only")

    def test_schedule_has_unique_ids_and_balanced_five_way_rotation(self):
        entries = []
        for image in range(5):
            entries.append({
                "image_id": f"i{image}",
                "questions": [{"question_id": f"q{image}-{q}"}
                              for q in range(10)],
            })
        schedule = RUNNER.expected_schedule(
            entries, skip=4, questions=6, seed=1234)
        self.assertEqual(len(schedule), 5 * 6 * 5)
        ids = [row["request_id"] for row in schedule]
        self.assertEqual(len(ids), len(set(ids)))
        for method in RUNNER.METHOD_KEYS:
            positions = {row["method_order_position"] for row in schedule
                         if row["method_key"] == method}
            self.assertEqual(positions, set(range(5)))

    def test_request_jsonl_is_durable_unique_and_resume_skippable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results_partial.jsonl"
            rows = [
                {"request_id": "gqa:i:q:recompute", "value": 1},
                {"request_id": "gqa:i:q:qa_chunk25", "value": 2},
            ]
            for row in rows:
                RUNNER.append_jsonl_durable(path, row)
            observed, repaired = RUNNER.read_jsonl_unique(path)
            self.assertEqual(observed, rows)
            self.assertEqual(repaired, 0)
            completed = {row["request_id"] for row in observed}
            self.assertIn("gqa:i:q:qa_chunk25", completed)
            self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_duplicate_request_ids_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results_partial.jsonl"
            row = {"request_id": "gqa:i:q:qa_chunk25"}
            RUNNER.append_jsonl_durable(path, row)
            RUNNER.append_jsonl_durable(path, row)
            with self.assertRaisesRegex(ValueError, "duplicate request_id"):
                RUNNER.read_jsonl_unique(path)

    def test_resume_archives_and_repairs_only_truncated_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "results_partial.jsonl"
            good = {"request_id": "gqa:i:q:qa_chunk25"}
            path.write_bytes((json.dumps(good) + "\n{\"request_id\":").encode())
            rows, repaired = RUNNER.read_jsonl_unique(
                path, repair_tail=True, recovery_dir=root / "recovery")
            self.assertEqual(rows, [good])
            self.assertGreater(repaired, 0)
            archives = list((root / "recovery").iterdir())
            self.assertEqual(len(archives), 1)
            self.assertEqual(archives[0].read_bytes(), b'{"request_id":')
            self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_invocation_cap_is_not_part_of_request_identity(self):
        self.assertFalse(RUNNER.request_cap_reached(0, 1))
        self.assertTrue(RUNNER.request_cap_reached(1, 1))
        self.assertFalse(RUNNER.request_cap_reached(10, None))
        identity = RUNNER.request_id("image", "question", "qa_chunk25")
        self.assertEqual(identity, "gqa:image:question:qa_chunk25")

    def test_selection_analysis_records_query_dependence_and_token_overlap(self):
        rows = []
        for turn, chunks in ((2, [[0], [1]]), (3, [[1], [1]]),
                             (4, [[0], [1]])):
            common = {"image_id": "i", "question_id": f"q{turn}",
                      "turn_id": turn}
            rows.append({**common, "method_key": "qa_chunk25",
                         "selected_chunk_ids_per_layer": chunks})
            rows.append({**common, "method_key": "qa_token25",
                         "selected_chunk_ids_per_layer": [[0, 1], [0, 1]],
                         "selected_token_ids_per_layer": [[0, 64], [1, 65]]})
        result = RUNNER.selection_analysis(rows)
        self.assertEqual(result["n_query_requests"], 3)
        self.assertEqual(result["n_pairs"], 3)
        self.assertGreater(result["different_selection_pairs"], 0)
        self.assertEqual(result["qa_token_overlap"]["n_requests"], 3)
        self.assertAlmostEqual(
            result["qa_token_overlap"]["mean_layer_jaccard"], 0.5)

    def test_reference_gates_keep_recompute_exact_and_gate_qa_token_timing(self):
        methods = {}
        for method in RUNNER.REFERENCE_METHOD:
            compared = 240 if method == "recompute" else 200
            methods[method] = {
                "compared": compared, "equal_predictions": compared,
                "accuracy_gap_pp": 0.0,
                "cache_hit_deterministic_io_compared": (
                    0 if method == "recompute" else 200),
                "cache_hit_deterministic_io_equal": (
                    0 if method == "recompute" else 200),
                "ttft_within_fixed_tolerance": True,
            }
        methods["qa_token25"]["selector_within_fixed_tolerance"] = True
        reference = {"methods": methods}
        self.assertTrue(RUNNER._reference_prediction_accuracy_passes(reference))
        self.assertTrue(RUNNER._reference_io_passes(reference))
        self.assertTrue(RUNNER._reference_timing_passes(reference))
        self.assertTrue(RUNNER._reference_selector_passes(reference))
        methods["recompute"]["equal_predictions"] -= 1
        self.assertFalse(RUNNER._reference_prediction_accuracy_passes(reference))
        methods["recompute"]["equal_predictions"] += 1
        methods["qa_token25"]["selector_within_fixed_tolerance"] = False
        self.assertFalse(RUNNER._reference_selector_passes(reference))

    def test_source_inventory_detects_metadata_or_payload_stat_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for side in ("raster", "image_only"):
                image = root / side / "i"
                image.mkdir(parents=True)
                (image / "meta.json").write_text('{"x":1}\n')
                layer = image / "layer_00"
                layer.mkdir()
                (layer / "k.bin").write_bytes(b"1234")
            before = RUNNER.source_store_inventory(root, ["i"])
            (root / "raster/i/layer_00/k.bin").write_bytes(b"12345")
            after = RUNNER.source_store_inventory(root, ["i"])
            self.assertNotEqual(before["inventory_sha256"],
                                after["inventory_sha256"])

    def test_launcher_is_detached_and_exposes_resume_checkpoint(self):
        launcher = (ROOT / "scripts/run_qa_chunk25_gqa_background.sh").read_text()
        self.assertIn("tmux new-session -d", launcher)
        self.assertIn("--resume", launcher)
        self.assertIn("--max-new-requests", launcher)
        self.assertIn("results_partial.jsonl", launcher)
        self.assertNotIn("nohup", launcher)


if __name__ == "__main__":
    unittest.main()

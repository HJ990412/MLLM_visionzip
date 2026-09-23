"""CPU-only unit tests for the MPIC final reporter."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "mpic_report", ROOT / "scripts/68_report_mpic_gqa.py")
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class MPICReportTests(unittest.TestCase):
    def test_runner_summary_accepts_sorted_json_keys_and_checks_csv_order(self):
        report_values = {}
        runner_values = {}
        for method in REPORT.METHOD_KEYS:
            report_values[method] = {
                "requests_all": 240,
                "requests_hit": 200,
                "accuracy_all": 0.5,
                "accuracy_hit": 0.5,
                "ttft_mean_ms": 10.0,
                "ttft_p50_ms": 9.0,
                "ttft_p95_ms": 12.0,
                "ssd_mb_per_hit": 1.0,
                "ssd_preads_per_hit": 2.0,
                "recomputed_image_tokens_mean": 3.0,
            }
            runner_values[method] = {
                "method_id": REPORT.METHOD_IDS[method],
                "display_label": REPORT.DISPLAY[method],
                "requests_all": 240,
                "requests_cache_hit": 200,
                "accuracy_all": 0.5,
                "accuracy_cache_hit": 0.5,
                "ttft_cache_hit_mean_ms": 10.0,
                "ttft_cache_hit_p50_ms": 9.0,
                "ttft_cache_hit_p95_ms": 12.0,
                "ssd_mb_cache_hit_mean": 1.0,
                "ssd_preads_cache_hit_mean": 2.0,
                "recomputed_image_tokens_cache_hit_mean": 3.0,
            }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "summary.json").write_text(json.dumps({
                "schema_version": REPORT.RUN_SCHEMA,
                "per_method": runner_values,
            }, sort_keys=True), encoding="utf-8")
            (root / "summary.csv").write_text(
                "method_key\n" + "".join(
                    f"{method}\n" for method in REPORT.METHOD_KEYS),
                encoding="utf-8")
            REPORT._verify_runner_summary(root, report_values)

    def test_independent_row_validator_accepts_runner_token_semantics(self):
        rows = []
        order = list(REPORT.METHOD_KEYS)
        for image_index in range(40):
            image_id = f"synthetic_image_{image_index:02d}"
            for turn in range(1, 7):
                question_id = f"q{image_index:02d}_{turn}"
                prompt_hash = f"prompt_{image_index}_{turn}"
                for position, method in enumerate(order):
                    cache_hit = turn > 1 and method != "recompute"
                    row = {
                        "schema_version": REPORT.RUN_SCHEMA,
                        "run_id": "synthetic_report_fixture",
                        "request_id": f"gqa:{image_id}:{question_id}:{method}",
                        "measurement_source": "same_run", "dataset": "gqa",
                        "image_id": image_id, "question_id": question_id,
                        "request_ordinal": turn, "turn_id": turn,
                        "question": f"question {question_id}", "gold": ["yes"],
                        "method_key": method,
                        "method_id": REPORT.METHOD_IDS[method],
                        "display_label": REPORT.DISPLAY[method],
                        "method_order": order, "method_order_position": position,
                        "cache_hit_measurement": cache_hit,
                        "prediction": "Yes", "correct": 1.0,
                        "first_token_id": 7, "prompt_sha256": prompt_hash,
                        "expected_prompt_sha256": prompt_hash,
                        "suffix_ids_sha256": "suffix", "input_tensors_sha256": "input",
                        "image_input_sha256": "image", "future_questions_in_prompt": 0,
                        "future_question_ids_used": [], "n_image_tokens": 64,
                        "retry_count": 0, "status": "ok", "ttft_ms": 10.0,
                        "end_to_end_ttft_ms": 10.0, "request_e2e_ms": 12.0,
                        "generated_tokens": 1, "ssd_read_bytes": 0,
                        "ssd_preads": 0,
                    }
                    if turn == 1:
                        row.update({
                            "request_path": "normal_pixel_turn1",
                            "vision_forward_count": 1,
                            "n_recomputed_image_tokens": 64,
                            "n_reused_image_tokens": 0,
                            "retained_image_context_ratio": 1.0,
                        })
                    elif method == "recompute":
                        row.update({
                            "request_path": "normal_pixel_recompute",
                            "vision_forward_count": 1,
                            "n_recomputed_image_tokens": 64,
                            "n_reused_image_tokens": 0,
                            "retained_image_context_ratio": 1.0,
                        })
                    elif method == "fullload":
                        row.update({
                            "request_path": "ssd_cache_hit",
                            "vision_forward_count": 0,
                            "n_recomputed_image_tokens": 0,
                            "n_reused_image_tokens": 64,
                            "retained_image_context_ratio": 1.0,
                        })
                    elif method in {"qa_chunk25", "ours25"}:
                        row.update({
                            "request_path": "ssd_cache_hit",
                            "vision_forward_count": 0,
                            "n_recomputed_image_tokens": 0,
                            "n_reused_image_tokens": None,
                            "retained_image_context_ratio": 0.25,
                            "image_token_count_semantics":
                                "N/A here; selected retained rows/chunks are reported",
                        })
                    else:
                        target = list(range(5, 69))
                        row.update({
                            "request_path": "ssd_cache_hit_mpic",
                            "vision_forward_count": 0, "k_recompute": 32,
                            "n_recomputed_image_tokens": 32,
                            "n_reused_image_tokens": 32,
                            "n_recomputed_text_tokens": 3,
                            "retained_image_context_ratio": 1.0,
                            "recomputed_image_token_ratio": 0.5,
                            "reused_image_token_ratio": 0.5,
                            "selected_image_local_rows": list(range(32)),
                            "selected_image_logical_rows": target[:32],
                            "target_positions": target,
                            "same_source_target_context": True,
                            "same_source_target_positions": True,
                            "source_position_hash": "position",
                            "target_position_hash": "position",
                            "position_handling_policy":
                                "post_rope_cached_k_reused_at_identical_logical_position",
                            "active_rows_per_layer": [35] * 32,
                            "recomputed_image_rows_per_layer": [32] * 32,
                            "recomputed_text_rows_per_layer": [3] * 32,
                            "attention_key_length_per_layer": [72] * 32,
                            "valid_image_key_count_per_layer": [64] * 32,
                            "decoder_prefill_pass_count": 1,
                            "source_payload_hash_before": "payload",
                            "source_payload_hash_after": "payload",
                            "ssd_kv_bytes": 1000, "ssd_embedding_bytes": 100,
                            "ssd_separator_bytes": 0, "ssd_metadata_bytes": 0,
                            "ssd_total_bytes": 1100, "ssd_read_bytes": 1100,
                            "pread_count": 65, "ssd_preads": 65,
                            "io": {"per_kind": {
                                "kv_k": {"preads": 32},
                                "kv_v": {"preads": 32},
                                "embedding": {"preads": 1},
                            }},
                            "decode_cache_append_exact": True,
                            "generated_token_ids": [7],
                            "generated_token_count": 1,
                        })
                    rows.append(row)
        evidence = REPORT._validate_rows(rows, "synthetic_report_fixture")
        self.assertEqual(len(evidence["by_method"]["qa_chunk25"]), 240)
        self.assertEqual(len(evidence["by_method"]["mpic32"]), 240)
        self.assertEqual(evidence["retry_total"], 0)

    def test_cluster_bootstrap_is_deterministic_and_image_clustered(self):
        pairs = {
            "image_a": [1.0] * 6,
            "image_b": [-1.0] * 6,
            "image_c": [0.0] * 6,
        }
        left = REPORT._cluster_bootstrap(pairs, seed=17, replicates=1000)
        right = REPORT._cluster_bootstrap(pairs, seed=17, replicates=1000)
        self.assertEqual(left, right)
        self.assertEqual(left["clusters"], 3)
        self.assertEqual(left["paired_observations"], 18)
        self.assertAlmostEqual(left["estimate"], 0.0)
        self.assertLessEqual(left["ci95_low"], 0.0)
        self.assertGreaterEqual(left["ci95_high"], 0.0)

    def test_publish_is_no_clobber_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.txt"
            self.assertEqual(REPORT._publish_bytes(path, b"stable\n"),
                             "published")
            self.assertEqual(REPORT._publish_bytes(path, b"stable\n"),
                             "identical")
            with self.assertRaises(REPORT.ReportValidationError):
                REPORT._publish_bytes(path, b"different\n")
            self.assertEqual(path.read_bytes(), b"stable\n")

    def test_percentile_matches_linear_interpolation(self):
        self.assertEqual(REPORT._percentile([0.0, 10.0], 50), 5.0)
        self.assertEqual(REPORT._percentile([0.0, 10.0, 20.0], 25), 5.0)

    def test_qa_persistence_reports_zero_incremental_store(self):
        def evidence(method: str):
            byte_counts = {"visual_kv": 100, "total": 120}
            if method == "mpic32":
                byte_counts["visual_input"] = 10
            return {
                "bytes": byte_counts,
                "timing_ms": {"persist_ms": 2.0, "ssd_write_ms": 1.0,
                              "fsync_ms": 0.5,
                              **({
                                  "visual_input_capture_materialize_ms": 0.25,
                                  "provisioning_post_response_ms": 2.25,
                              } if method == "mpic32" else {})},
            }

        grouped = {
            method: [evidence(method), evidence(method)]
            for method in REPORT.STORE_METHODS
        }
        rows = {row["method_key"]: row
                for row in REPORT._persistence_rows(grouped)}
        self.assertEqual(rows["qa_chunk25"]["mean_total_bytes"], 0.0)
        self.assertEqual(rows["qa_chunk25"][
            "shared_backing_mean_total_bytes"], 120.0)
        self.assertFalse(rows["qa_chunk25"]["incremental_store_owner"])
        self.assertEqual(rows["fullload"]["mean_total_bytes"], 120.0)
        self.assertTrue(rows["mpic32"]["incremental_store_owner"])
        self.assertEqual(rows["fullload"][
            "mean_provisioning_post_response_ms"], 2.0)
        self.assertEqual(rows["mpic32"][
            "mean_visual_input_capture_materialize_ms"], 0.25)
        self.assertEqual(rows["mpic32"][
            "mean_provisioning_post_response_ms"], 2.25)


if __name__ == "__main__":
    unittest.main()

"""CPU-only contracts for the Gold/Generated MT-GQA shard evaluator."""
from __future__ import annotations

import copy
import importlib.util
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/54_eval_mt_gqa_history_shard.py"
SPEC = importlib.util.spec_from_file_location("mt_gqa_history_runner", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _dialog():
    return {
        "dialog_id": "mtgqa_000001",
        "global_dialog_ordinal": 0,
        "image_id": "n1",
        "turns": [
            {"turn_id": 1, "question_id": "q1",
             "question": "What is first?", "answers": ["gold one"]},
            {"turn_id": 2, "question_id": "q2",
             "question": "What is second?", "answers": ["gold two"]},
            {"turn_id": 3, "question_id": "q3",
             "question": "What is third?", "answers": ["gold three"]},
        ],
    }


def _manifests():
    common = {
        "visual_kv_bytes": 400,
        "n_chunks_per_layer": 4,
        "num_layers": 2,
        "v_token_num": 4,
        "chunk_size": 1,
        "separator_sidecar_bytes": 10,
    }
    return {
        "raster": {
            **common, "physical_layout": "raster",
            "meta_sha256": "raster-meta",
            "probe_sidecar_bytes": 20,
        },
        "image_only": {
            **common, "physical_layout": "visionzip_image_only",
            "meta_sha256": "ours-meta",
            "permutation_sha256": "ours-permutation",
            "selected_prefix_original_token_ids_sha256": "ours-prefix",
        },
    }


def _synthetic_rows(protocol: str):
    dialog = _dialog()
    generated = {method: {} for method in MODULE.METHOD_KEYS}
    rows = []
    order = MODULE.method_order(0)
    for turn_id in (1, 2, 3):
        for position, method in enumerate(order):
            prompt, history, entries = MODULE.render_causal_prompt(
                dialog, turn_id, protocol, method_key=method,
                generated_predictions=generated[method])
            execution_id = f"physical:{protocol}:{method}:t{turn_id}"
            logical_id = MODULE.logical_request_id(
                protocol, dialog["dialog_id"], turn_id, method)
            if turn_id == 1:
                prediction = "same turn one"
                first_token = 10
            else:
                prediction = f"{method} turn {turn_id}"
                first_token = 10 + turn_id
            cache_hit = turn_id >= 2 and method != "recompute"
            selected = [[0], [0]] if cache_hit and method in {
                "qa_chunk25", "ours25"} else None
            store_id = ("ours-meta" if method == "ours25"
                        else "raster-meta" if cache_hit else None)
            if cache_hit and method == "fullload":
                normal_bytes, probe_bytes, separator_bytes = 400, 0, 0
                normal_preads, probe_preads, separator_preads = 4, 0, 0
            elif cache_hit and method == "qa_chunk25":
                normal_bytes, probe_bytes, separator_bytes = 100, 20, 10
                normal_preads, probe_preads, separator_preads = 4, 2, 1
            elif cache_hit and method == "ours25":
                normal_bytes, probe_bytes, separator_bytes = 100, 0, 10
                normal_preads, probe_preads, separator_preads = 4, 0, 1
            else:
                normal_bytes = probe_bytes = separator_bytes = 0
                normal_preads = probe_preads = separator_preads = 0
            ssd_bytes = normal_bytes + probe_bytes + separator_bytes
            ssd_preads = normal_preads + probe_preads + separator_preads
            row = {
                "status": "ok",
                "protocol": protocol,
                "dialog_id": dialog["dialog_id"],
                "turn_id": turn_id,
                "method_key": method,
                "method_order": list(order),
                "method_order_position": position,
                "history_text": history,
                "history_entries": entries,
                "history_answers": [entry["answer"] for entry in entries],
                "history_source_request_ids": [
                    entry["source_logical_request_id"] for entry in entries],
                "history_source_physical_execution_ids": [
                    entry["source_physical_execution_id"]
                    for entry in entries],
                "history_turn_ids": list(range(1, turn_id)),
                "prompt": prompt,
                "prompt_sha256": MODULE.sha256_text(prompt),
                "history_text_sha256": MODULE.sha256_text(history),
                "suffix_ids_sha256": "same-suffix",
                "combined_suffix_ids_sha256": (
                    None if cache_hit else "same-suffix"),
                "future_leakage": 0,
                "prediction": prediction,
                "gold_answer": f"gold {('one', 'two', 'three')[turn_id - 1]}",
                "correct": 0.0,
                "strict_correct": 0.0,
                "input_token_count": 10,
                "generated_token_count": 2,
                "logical_request_id": logical_id,
                "physical_execution_id": execution_id,
                "execution_id": execution_id,
                "cache_hit": cache_hit,
                "request_path": ("stored_visual_kv" if cache_hit
                                 else "normal_multimodal_pixel"),
                "ssd_read_bytes": ssd_bytes,
                "ssd_preads": ssd_preads,
                "normal_kv_read_bytes": normal_bytes,
                "probe_read_bytes": probe_bytes,
                "separator_read_bytes": separator_bytes,
                "normal_kv_preads": normal_preads,
                "probe_preads": probe_preads,
                "separator_preads": separator_preads,
                "contiguous_runs_per_layer": (
                    [1, 1] if cache_hit else []),
                "contiguous_runs_per_layer_mean": (
                    1.0 if cache_hit else 0.0),
                "n_raters": (
                    2 if cache_hit and method == "qa_chunk25" else 0),
                "vision_forward_count": 0 if cache_hit else 1,
                "page_cache_conditioning_excluded_from_ttft": cache_hit,
                "store_id": store_id,
                "store_permutation_sha256": (
                    "ours-permutation" if cache_hit and method == "ours25"
                    else None),
                "selected_prefix_original_token_ids_sha256": (
                    "ours-prefix" if cache_hit and method == "ours25"
                    else None),
                "dialogue_session_id": f"session:{protocol}",
                "context_instance_id": f"context:{protocol}",
                "selected_chunk_ids_per_layer": selected,
                "selection_fingerprint_sha256": (
                    MODULE.stable_json_sha256(selected)
                    if selected is not None else None),
                "query_score_calls": (
                    2 if cache_hit and method == "qa_chunk25" else 0),
                "chunk_score_calls": (
                    2 if cache_hit and method == "qa_chunk25" else 0),
                "fallback_rate": (
                    0.0 if cache_hit and method == "qa_chunk25" else None),
                "adaptive_ratio": (
                    False if cache_hit and method == "qa_chunk25" else None),
                "static_score_calls": 0,
                "diversity_calls": 0,
                "first_token_id": first_token,
            }
            rows.append(row)
            generated[method][turn_id] = {
                "prediction": prediction,
                "logical_request_id": logical_id,
                "physical_execution_id": execution_id,
            }
    return rows


class HistoryRenderingTests(unittest.TestCase):
    def test_exact_request_accounting(self):
        counts = MODULE.expected_request_counts(4061)
        self.assertEqual(counts["turns_per_protocol"], 12183)
        self.assertEqual(counts["requests_per_method_per_protocol"], 12183)
        self.assertEqual(counts["requests_per_protocol"], 48732)
        self.assertEqual(counts["requests_both_protocols"], 97464)
        self.assertEqual(counts["main_t2_t3_requests_both_protocols"], 64976)
        self.assertEqual(counts["stored_visual_kv_hits_both_protocols"], 48732)

    def test_strict_normalized_exact_is_not_prefix_tolerant(self):
        self.assertEqual(MODULE.strict_gqa_score("The red-car!", "red car"), 1)
        self.assertEqual(MODULE.strict_gqa_score("Computer mouse", "computer"), 0)
        self.assertEqual(MODULE.normalize_answer("A, BLUE bird"), "blue bird")

    def test_gold_prompt_is_causal_and_teacher_forced(self):
        prompt, history, entries = MODULE.render_causal_prompt(
            _dialog(), 3, "gold_history", method_key="recompute")
        self.assertEqual(
            history,
            "Q1: What is first?\nA1: gold one\n"
            "Q2: What is second?\nA2: gold two")
        self.assertIn("Current question Q3: What is third?", prompt)
        self.assertNotIn("gold three", prompt)
        self.assertEqual(
            [entry["source_logical_request_id"] for entry in entries],
            ["gold:q1", "gold:q2"])
        self.assertTrue(all(entry["source_physical_execution_id"] is None
                            for entry in entries))

    def test_generated_prompt_uses_exact_same_method_lineage(self):
        state = {
            1: {"prediction": "raw A1", "logical_request_id": "qa:t1",
                "physical_execution_id": "exec-1"},
            2: {"prediction": "raw A2", "logical_request_id": "qa:t2",
                "physical_execution_id": "exec-2"},
        }
        prompt, history, entries = MODULE.render_causal_prompt(
            _dialog(), 3, "generated_history", method_key="qa_chunk25",
            generated_predictions=state)
        self.assertIn("A1: raw A1", history)
        self.assertIn("A2: raw A2", history)
        self.assertNotIn("gold one", prompt)
        self.assertEqual(
            [entry["source_method_key"] for entry in entries],
            ["qa_chunk25", "qa_chunk25"])
        self.assertEqual(
            [entry["source_physical_execution_id"] for entry in entries],
            ["exec-1", "exec-2"])

    def test_generated_history_fails_closed_on_missing_or_future_state(self):
        with self.assertRaisesRegex(ValueError, "missing generated"):
            MODULE.render_causal_prompt(
                _dialog(), 2, "generated_history", method_key="ours25",
                generated_predictions={})
        with self.assertRaisesRegex(ValueError, "current/future"):
            MODULE.render_causal_prompt(
                _dialog(), 2, "generated_history", method_key="ours25",
                generated_predictions={2: "leak"})

    def test_method_rotation_is_balanced_and_seed_frozen(self):
        orders = [MODULE.method_order(index) for index in range(4)]
        self.assertEqual([order[0] for order in orders],
                         list(MODULE.METHOD_KEYS))
        for position in range(4):
            self.assertEqual(
                {order[position] for order in orders},
                set(MODULE.METHOD_KEYS))
        with self.assertRaises(ValueError):
            MODULE.method_order(0, seed=0)


class ImageValidationTests(unittest.TestCase):
    def _group(self):
        return {"image_id": "n1", "image_ordinal": 0,
                "dialogs": [_dialog()]}

    def test_gold_and_generated_synthetic_rows_validate(self):
        for protocol in MODULE.PROTOCOLS:
            with self.subTest(protocol=protocol):
                result = MODULE.validate_image_rows(
                    _synthetic_rows(protocol), self._group(), protocol,
                    _manifests())
                self.assertTrue(result["passed"])
                self.assertEqual(result["n_rows"], 12)
                self.assertEqual(result["cache_hit_rows"], 6)
                self.assertEqual(result["qa_chunk_cache_hit_rows"], 2)

    def test_generated_cross_method_lineage_is_rejected(self):
        rows = _synthetic_rows("generated_history")
        target = next(row for row in rows
                      if row["turn_id"] == 2
                      and row["method_key"] == "qa_chunk25")
        target["history_entries"][0]["source_method_key"] = "ours25"
        with self.assertRaisesRegex(ValueError, "history provenance"):
            MODULE.validate_image_rows(
                rows, self._group(), "generated_history", _manifests())

    def test_ours_nonprefix_selection_is_rejected(self):
        rows = _synthetic_rows("gold_history")
        target = next(row for row in rows
                      if row["turn_id"] == 2
                      and row["method_key"] == "ours25")
        target["selected_chunk_ids_per_layer"] = [[1], [1]]
        target["selection_fingerprint_sha256"] = MODULE.stable_json_sha256(
            target["selected_chunk_ids_per_layer"])
        with self.assertRaisesRegex(ValueError, "fixed physical prefix"):
            MODULE.validate_image_rows(
                rows, self._group(), "gold_history", _manifests())

    def test_duplicate_request_identity_is_rejected(self):
        rows = _synthetic_rows("gold_history")
        rows[1]["method_key"] = rows[0]["method_key"]
        with self.assertRaisesRegex(ValueError, "coverage/order"):
            MODULE.validate_image_rows(
                rows, self._group(), "gold_history", _manifests())


class CliContractTests(unittest.TestCase):
    def _args(self, root: Path, **updates):
        values = {
            "protocol": "gold_history",
            "run_dir": root / "run",
            "temp_root": root / "temp",
            "shard_size": 50,
            "shard_index": 0,
            "seed": 1234,
            "max_new_tokens": 16,
            "min_free_after_gib": 30.0,
            "max_dialogs": None,
            "allow_partial_workload": False,
            "expected_dialogs": 4061,
        }
        values.update(updates)
        return Namespace(**values)

    def test_cli_requires_protocol_scoped_nonoverlapping_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            MODULE._validate_cli(self._args(root))
            with self.assertRaisesRegex(ValueError, "may not overlap"):
                MODULE._validate_cli(self._args(
                    root, temp_root=root / "run/temp"))
            with self.assertRaisesRegex(ValueError, "expected-dialogs"):
                MODULE._validate_cli(self._args(
                    root, max_dialogs=10, allow_partial_workload=True,
                    expected_dialogs=4061))

    def test_partial_cli_pair_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            MODULE._validate_cli(self._args(
                root, protocol="generated_history", max_dialogs=10,
                allow_partial_workload=True, expected_dialogs=10))
            with self.assertRaisesRegex(ValueError, "must be paired"):
                MODULE._validate_cli(self._args(
                    root, max_dialogs=10, allow_partial_workload=False,
                    expected_dialogs=10))


if __name__ == "__main__":
    unittest.main()

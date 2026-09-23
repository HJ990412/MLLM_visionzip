#!/usr/bin/env python3
"""Run one image shard of the MT-VQA-v2-reconstructed Generated-History eval.

This is a thin, fail-closed dataset adapter over the validated MT-GQA four-arm
serving implementation.  It changes only the frozen workload identity,
Generated-History-only protocol, and VQAv2 consensus scorer; model execution,
cache persistence, QA selection, Ours repacking, TTFT, and SSD accounting stay
on the exact same implementation path.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mmimpress.dataset import vqa_score  # noqa: E402
from mmimpress import mt_vqa_v2  # noqa: E402


SCHEMA_VERSION = "mt-vqa-v2-generated-4arm-shard-v1"
DATASET = "vqav2_validation_mt3_reconstructed"
BENCHMARK_TYPE = mt_vqa_v2.BENCHMARK_TYPE
DEFAULT_INDEX = ROOT / "data/mt_vqa_v2/dialogues.json"
EXPECTED_INDEX_SHA256 = (
    "89719b2a1187c07e3228cc76cf1e473b3c713a0dcb3d65da0c596ae81898d6ea"
)
EXPECTED_WORKLOAD_SHA256 = (
    "384e39bad4e2e8d5865fe20bad7661d3cad8fe0b5ce5effbfc43ea896170cbbc"
)
EXPECTED_DIALOGUES = 250
EXPECTED_TURNS = 750
EXPECTED_IMAGES = 250
PROTOCOLS = ("generated_history",)
VQA_BINARY_CORRECT_THRESHOLD = 0.5
QUALITY_METRIC = "repository_vqa_consensus_min_matches_over_3"
TEMP_OWNER_FILE = ".mt_vqa_v2_temp_store_owner.json"
DATASET_CONSTRUCTION = {
    "source_index_sha256": mt_vqa_v2.SOURCE_INDEX_SHA256,
    "source_slice": "questions[1:5]",
    "dialogue_membership": "questions[1:4]",
    "dialogues_per_image": 1,
    "dialogue_overlap": False,
    "question_reuse": False,
}


def _load_base():
    path = ROOT / "scripts/54_eval_mt_gqa_history_shard.py"
    spec = importlib.util.spec_from_file_location(
        "_mt_vqa_v2_reused_history_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()


def _gold_answers(turn: Mapping[str, Any]) -> list[str]:
    answers = turn.get("answers")
    if (not isinstance(answers, list) or len(answers) != 10
            or any(not str(answer).strip() for answer in answers)):
        raise ValueError("MT-VQA-v2 turn must contain ten nonempty answers")
    return [str(answer) for answer in answers]


def _score(prediction: Any, answers: Any) -> float:
    if (not isinstance(answers, (list, tuple)) or len(answers) != 10
            or any(not str(value).strip() for value in answers)):
        raise ValueError("VQAv2 scorer requires exactly ten nonempty answers")
    return float(vqa_score(str(prediction), [str(value) for value in answers]))


def expected_request_counts(n_dialogues: int) -> dict[str, int]:
    """Exact counts for the single Generated-History protocol.

    The two ``*_both_protocols`` keys are retained only because the reused
    MT-GQA config builder reads them.  Their values deliberately describe the
    only protocol in this run; they are removed from the published config.
    """
    n = int(n_dialogues)
    if n <= 0:
        raise ValueError("n_dialogues must be positive")
    turns = 3 * n
    per_protocol = turns * len(BASE.METHOD_KEYS)
    return {
        "dialogues": n,
        "turns_per_protocol": turns,
        "requests_per_method_per_protocol": turns,
        "turn1_requests_per_method_per_protocol": n,
        "cache_hit_requests_per_method_per_protocol": 2 * n,
        "requests_per_protocol": per_protocol,
        "requests_both_protocols": per_protocol,
        "main_t2_t3_requests_both_protocols": 2 * n * len(BASE.METHOD_KEYS),
        "stored_visual_kv_hits_both_protocols": 2 * n * 3,
    }


_ORIGINAL_MAKE_ROW = BASE._make_row
_ORIGINAL_BASE_CONFIG = BASE._base_config
_ORIGINAL_VALIDATE_CONFIG = BASE._validate_existing_config
_ORIGINAL_VALIDATE_ROWS = BASE.validate_image_rows


def _make_vqa_row(**kwargs):
    row = _ORIGINAL_MAKE_ROW(**kwargs)
    answers = _gold_answers(kwargs["turn"])
    score = _score(row["prediction"], answers)
    row.update({
        "gold": answers,
        "gold_answers": answers,
        "gold_answer": answers,
        "correct": score,
        "score": score,
        "quality_score": score,
        "vqa_score": score,
        "quality_metric": QUALITY_METRIC,
        "quality_metric_implementation": "mmimpress.dataset.vqa_score",
        "binary_correct_threshold": VQA_BINARY_CORRECT_THRESHOLD,
        "binary_correct": int(score >= VQA_BINARY_CORRECT_THRESHOLD),
        "full_credit_correct": int(score == 1.0),
        "official_vqa_evaluator_claimed": False,
        "first_turn_fairness_key": (
            f"{row['dialog_id']}:{row['method_key']}"
            if int(row["turn_id"]) == 1 else None),
    })
    row.pop("strict_correct", None)
    row.pop("first_turn_cross_protocol_key", None)
    return row


def _base_config(args, workload, groups, n_shards, warmup, runner):
    config = _ORIGINAL_BASE_CONFIG(
        args, workload, groups, n_shards, warmup, runner)
    counts = expected_request_counts(int(workload["n_dialogs"]))
    config.pop("protocol_physical_execution_independent", None)
    config.pop("planned_logical_requests_both_protocols", None)
    config.pop("planned_physical_executions_both_protocols", None)
    config.update({
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET,
        "benchmark_type": BENCHMARK_TYPE,
        "protocols": list(PROTOCOLS),
        "history_policy": "method_local_generated",
        "quality_metric": QUALITY_METRIC,
        "quality_metric_implementation": "mmimpress.dataset.vqa_score",
        "normalization": (
            "repository VQA normalization: lowercase; punctuation to spaces; "
            "remove a/an/the; score=min(ten-answer matches/3,1)"
        ),
        "official_vqa_evaluator_claimed": False,
        "binary_correct_threshold": VQA_BINARY_CORRECT_THRESHOLD,
        "binary_diagnostics_only": True,
        "protocol_scope": "generated_history_only",
        "planned_logical_requests_generated_only": counts[
            "requests_per_protocol"],
        "planned_physical_executions_generated_only": counts[
            "requests_per_protocol"],
        "request_counts": {
            "dialogues": counts["dialogues"],
            "turns_generated_only": counts["turns_per_protocol"],
            "requests_per_method_generated_only": counts[
                "requests_per_method_per_protocol"],
            "turn1_requests_per_method_generated_only": counts[
                "turn1_requests_per_method_per_protocol"],
            "cache_hit_requests_per_method_generated_only": counts[
                "cache_hit_requests_per_method_per_protocol"],
            "requests_generated_only": counts["requests_per_protocol"],
            "main_t2_t3_requests_generated_only": counts[
                "main_t2_t3_requests_both_protocols"],
            "stored_visual_kv_hits_generated_only": counts[
                "stored_visual_kv_hits_both_protocols"],
        },
        "method_order_policy": (
            "zero-based cyclic rotation by frozen global dialogue ordinal; "
            "identical across turns in the Generated-History protocol"),
        "dataset_construction": dict(DATASET_CONSTRUCTION),
    })
    return config


def _validate_existing_config(run_dir, args, workload, groups, n_shards):
    config = _ORIGINAL_VALIDATE_CONFIG(
        run_dir, args, workload, groups, n_shards)
    expected = {
        "quality_metric": QUALITY_METRIC,
        "quality_metric_implementation": "mmimpress.dataset.vqa_score",
        "official_vqa_evaluator_claimed": False,
        "binary_correct_threshold": VQA_BINARY_CORRECT_THRESHOLD,
        "history_policy": "method_local_generated",
        "protocol_scope": "generated_history_only",
        "planned_logical_requests_generated_only":
            expected_request_counts(int(workload["n_dialogs"]))[
                "requests_per_protocol"],
        "planned_physical_executions_generated_only":
            expected_request_counts(int(workload["n_dialogs"]))[
                "requests_per_protocol"],
        "dataset_construction": DATASET_CONSTRUCTION,
    }
    mismatch = {key: (config.get(key), value)
                for key, value in expected.items()
                if config.get(key) != value}
    if mismatch:
        raise ValueError(f"existing MT-VQA-v2 config mismatch: {mismatch}")
    return config


def _validate_vqa_rows(*args, **kwargs):
    validation = _ORIGINAL_VALIDATE_ROWS(*args, **kwargs)
    rows = args[0] if args else kwargs["rows"]
    group = args[1] if len(args) > 1 else kwargs["group"]
    canonical: dict[tuple[str, int], list[str]] = {}
    for dialog in group["dialogs"]:
        did = str(dialog["dialog_id"])
        for turn in dialog["turns"]:
            key = (did, int(turn["turn_id"]))
            if key in canonical:
                raise ValueError(f"duplicate canonical VQA turn: {key}")
            canonical[key] = _gold_answers(turn)
    t1_by_dialog: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("dialog_id", "")), int(row.get("turn_id", -1)))
        if key not in canonical:
            raise ValueError(f"row is not in canonical VQA workload: {key}")
        answers = canonical[key]
        for alias in ("gold", "gold_answers", "gold_answer"):
            if row.get(alias) != answers:
                raise ValueError(
                    f"{key}: {alias} differs from canonical ten answers")
        expected_score = _score(row.get("prediction", ""), answers)
        for alias in ("correct", "score", "quality_score", "vqa_score"):
            try:
                observed_score = float(row.get(alias))
            except (TypeError, ValueError):
                raise ValueError(f"{key}: invalid {alias}") from None
            if abs(observed_score - expected_score) > 1e-12:
                raise ValueError(f"{key}: canonical {alias} mismatch")
        if (row.get("quality_metric") != QUALITY_METRIC
                or row.get("quality_metric_implementation")
                != "mmimpress.dataset.vqa_score"
                or row.get("official_vqa_evaluator_claimed") is not False
                or float(row.get("binary_correct_threshold", -1.0))
                != VQA_BINARY_CORRECT_THRESHOLD
                or int(row.get("binary_correct", -1))
                != int(expected_score >= VQA_BINARY_CORRECT_THRESHOLD)
                or int(row.get("full_credit_correct", -1))
                != int(expected_score == 1.0)):
            raise ValueError(f"{key}: VQA metric metadata/diagnostics mismatch")
        if int(row["turn_id"]) == 1:
            t1_by_dialog.setdefault(str(row["dialog_id"]), []).append(row)
    input_fair = all(
        len(group) == 4
        and len({str(row.get("image_input_sha256")) for row in group}) == 1
        and None not in {row.get("image_input_sha256") for row in group}
        and len({str(row.get("input_tensors_sha256")) for row in group}) == 1
        and None not in {row.get("input_tensors_sha256") for row in group}
        for group in t1_by_dialog.values()
    )
    if not input_fair:
        raise ValueError("Turn-1 four-arm pixel/input fairness failed")
    validation.pop("strict_scores_recomputed", None)
    validation.update({
        "vqa_consensus_scores_recomputed": True,
        "vqa_binary_correct_threshold": VQA_BINARY_CORRECT_THRESHOLD,
        "turn1_pixel_and_input_hash_fairness": True,
    })
    return validation


def _configure_base() -> None:
    # Dataset/protocol identities are read dynamically by the reused runner.
    BASE.SCHEMA_VERSION = SCHEMA_VERSION
    BASE.DATASET = DATASET
    BASE.BENCHMARK_TYPE = BENCHMARK_TYPE
    BASE.DEFAULT_INDEX = DEFAULT_INDEX
    BASE.EXPECTED_INDEX_SHA256 = EXPECTED_INDEX_SHA256
    BASE.EXPECTED_WORKLOAD_SHA256 = EXPECTED_WORKLOAD_SHA256
    BASE.EXPECTED_DIALOGUES = EXPECTED_DIALOGUES
    BASE.EXPECTED_TURNS = EXPECTED_TURNS
    BASE.EXPECTED_IMAGES = EXPECTED_IMAGES
    BASE.PROTOCOLS = PROTOCOLS
    BASE.expected_request_counts = expected_request_counts
    BASE._gold = _gold_answers
    BASE.strict_gqa_score = _score
    BASE._make_row = _make_vqa_row
    BASE._base_config = _base_config
    BASE._validate_existing_config = _validate_existing_config
    BASE.validate_image_rows = _validate_vqa_rows

    # Script 37 supplies durable sharding/temp-store helpers.  Bind its
    # dataset-specific resolver hooks to the new immutable VQAv2 adapter.
    shard_helpers = BASE.mt_base()
    shard_helpers.SCHEMA_VERSION = SCHEMA_VERSION
    shard_helpers.DATASET = DATASET
    shard_helpers.BENCHMARK_TYPE = BENCHMARK_TYPE
    shard_helpers.OWNER_FILE = TEMP_OWNER_FILE
    shard_helpers._load_mt_helpers = lambda: mt_vqa_v2


_configure_base()


def main() -> None:
    BASE.main()


if __name__ == "__main__":
    main()

"""Local ConvBench pairwise judge using the official prompt artifact.

Only the official ``generate_request`` function is loaded from
``ConvBenchEval/convbencheval.py``.  Importing that module normally would run
its GPT API evaluation at import time.  The GPT transport is replaced by a
local, greedy Llama-3.1-8B-Instruct call; prompt wording and A/B positions are
read directly from the pinned official repository checkout.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import string
import subprocess
from pathlib import Path
from typing import Any


TURN_NAMES = ("_first_turn", "_second_turn", "_third_turn", "_overall_conversation")
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_revision(source_root: str | Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()


def load_official_prompt_builder(source_root: str | Path):
    """Load the exact upstream prompt function without upstream top-level code."""
    evaluator = Path(source_root) / "ConvBenchEval" / "convbencheval.py"
    prompts_path = Path(source_root) / "ConvBenchEval" / "prompts.py"
    spec = importlib.util.spec_from_file_location("convbench_official_prompts", prompts_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {prompts_path}")
    prompts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompts)

    tree = ast.parse(evaluator.read_text(encoding="utf-8"), filename=str(evaluator))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "generate_request"]
    if len(functions) != 1:
        raise ValueError("Official evaluator must contain one generate_request function")
    namespace = {
        name: getattr(prompts, name)
        for name in dir(prompts)
        if name.startswith("_PROMPT_")
    }
    function_module = ast.Module(body=functions, type_ignores=[])
    exec(compile(function_module, str(evaluator), "exec"), namespace)
    return namespace["generate_request"], prompts


def load_pairwise(source_root: str | Path, source_row_indices: list[int]) -> dict[int, int]:
    import numpy as np

    path = Path(source_root) / "ConvBenchEval" / "pairwise.npy"
    values = np.load(path, allow_pickle=False)
    if values.ndim != 1 or any(index < 0 or index >= len(values) for index in source_row_indices):
        raise ValueError(f"pairwise.npy shape {values.shape} cannot index original source rows")
    positions = {index: int(values[index]) for index in source_row_indices}
    if any(position not in (0, 1) for position in positions.values()):
        raise ValueError("Official pairwise positions must be 0 or 1")
    return positions


def row_value(row: dict[str, Any], *names: str) -> str:
    for name in names:
        if name in row and row[name] is not None:
            value = str(row[name])
            if value.strip():
                return value
    raise ValueError(f"Missing/empty ConvBench field: {names}")


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Accept the local index names and the official workbook column names."""
    return {
        "conversation_id": row_value(row, "conversation_id", "id"),
        "source_row_index": int(row["source_row_index"]),
        "image_path": row_value(row, "image_path"),
        "caption": row_value(row, "instruction_conditioned_caption", "instruction-conditioned-caption"),
        "questions": [
            row_value(row, "Q1", "q1", "The_first_turn_instruction"),
            row_value(row, "Q2", "q2", "The_second_turn_instruction"),
            row_value(row, "Q3", "q3", "The_third_turn_instruction"),
        ],
        "references": [
            row_value(row, "reference_A1", "first_turn_answer"),
            row_value(row, "reference_A2", "second_turn_answer"),
            row_value(row, "reference_A3", "third_turn_answer"),
        ],
        "third_turn_demands": row_value(row, "third_turn_demands"),
    }


def build_messages(builder, row: dict[str, Any], predictions: list[str], position: int,
                   turn_name: str, previous_evaluations: list[str] | None = None) -> list[dict[str, str]]:
    if len(predictions) != 3 or any(not isinstance(answer, str) for answer in predictions):
        raise ValueError("Exactly three generated answers are required")
    if position not in (0, 1) or turn_name not in TURN_NAMES:
        raise ValueError("Invalid position or judge turn")
    questions, refs = row["questions"], row["references"]
    a, b = (predictions, refs) if position == 0 else (refs, predictions)
    kwargs: dict[str, Any] = {}
    if turn_name == "_third_turn":
        kwargs["cur_third_turn_demands"] = row["third_turn_demands"]
    if turn_name == "_overall_conversation":
        if previous_evaluations is None or len(previous_evaluations) != 3:
            raise ValueError("Overall judgment requires three earlier raw evaluations")
        kwargs.update(zip(("perception_prediction", "reasoning_prediction", "composition_prediction"), previous_evaluations))
    return builder(
        turn_name, row["image_path"], row["caption"],
        questions[0], a[0], b[0],
        questions[1], a[1], b[1],
        questions[2], a[2], b[2], **kwargs,
    )


def extract_official_winner(raw: str) -> str | None:
    """First pass of upstream extract_prediction_from_response."""
    selected = {letter: f"response {letter.lower()} is better" in raw.lower() for letter in "AB"}
    winners = [letter for letter, matched in selected.items() if matched]
    return winners[0] if len(winners) == 1 else None


def extract_fallback_winner(raw: str) -> str | None:
    """Accept only a single final A/B label from local second-pass extraction."""
    match = re.fullmatch(r"\s*(?:(?:Final Answer|Response)\s*:\s*|Response\s+)?([AB])\s*[.!]?\s*",
                         raw, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def extract_official_rate(raw: str) -> str:
    """Upstream scoring-loop fallback: Rating:A/B, otherwise C (zero win)."""
    found = re.search("Rating:(.*)", raw)
    value = found.group(1).strip().strip(string.punctuation) if found else "0"
    return value if value in ("A", "B") else "C"


def extraction_messages(raw: str) -> list[dict[str, str]]:
    """Request a label from the first-pass text, without rejudging the answers."""
    return [
        {"role": "system", "content": (
            "Extract the final preference expressed by a pairwise evaluator between "
            "Response A and Response B. Return only A or B when the evaluator's text "
            "supports exactly one preference. If it does not express a recoverable "
            "preference, return UNKNOWN. Do not judge the responses yourself or guess."
        )},
        {"role": "user", "content": (
            "Evaluator text:\n<evaluation>\n" + raw +
            "\n</evaluation>\nFinal preference label (A or B; UNKNOWN if absent):"
        )},
    ]


def score_records(records: list[dict[str, Any]], expected_per_method: int | None = None,
                  judge_turns: tuple[str, ...] = TURN_NAMES) -> dict[str, Any]:
    """Compute turn win rates with a fixed conversation denominator.

    An unresolved (None/C) or missing judgment earns zero wins.  The main
    ConvBench average uses only the first three turns; the overall judgment is
    retained as a secondary diagnostic.
    """
    if expected_per_method is not None and expected_per_method <= 0:
        raise ValueError("expected_per_method must be positive")
    if judge_turns not in (TURN_NAMES[:3], TURN_NAMES):
        raise ValueError("judge_turns must be the three stages or all four official turns")
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record in records:
        method = str(record["method"])
        turn = str(record["judge_turn"])
        if turn not in judge_turns:
            raise ValueError(f"Unknown judge turn {turn}")
        grouped.setdefault(method, {}).setdefault(turn, []).append(record)
    scores = {}
    for method, by_turn in grouped.items():
        populations = [
            {(e["conversation_id"], e["source_row_index"]) for e in by_turn.get(turn, [])}
            for turn in judge_turns
        ]
        population = set().union(*populations)
        denominator = expected_per_method if expected_per_method is not None else len(population)
        if not denominator or len(population) > denominator:
            raise ValueError(f"Judge population exceeds fixed denominator for {method}: "
                             f"{len(population)} > {denominator}")
        turn_scores = []
        valid_by_turn: dict[str, int] = {}
        unresolved_by_turn: dict[str, int] = {}
        missing_by_turn: dict[str, int] = {}
        wins_by_turn: dict[str, int] = {}
        denominators_by_turn: dict[str, int] = {}
        for turn in judge_turns:
            entries = by_turn.get(turn, [])
            keys = [(e["conversation_id"], e["source_row_index"]) for e in entries]
            if len(keys) != len(set(keys)):
                raise ValueError(f"Duplicate judge rows for {method}/{turn}")
            if any(e.get("winner") not in ("A", "B", "C", None) for e in entries):
                raise ValueError(f"Invalid judge decision for {method}/{turn}")
            valid = [entry for entry in entries if entry["winner"] in ("A", "B")]
            metric = {"_first_turn": "S1", "_second_turn": "S2",
                      "_third_turn": "S3", "_overall_conversation": "overall_official_secondary"}[turn]
            valid_by_turn[metric] = len(valid)
            unresolved_by_turn[metric] = len(entries) - len(valid)
            missing_by_turn[metric] = denominator - len(entries)
            denominators_by_turn[metric] = denominator
            wins = sum(e["winner"] == ("A" if e["position"] == 0 else "B") for e in valid)
            wins_by_turn[metric] = wins
            turn_scores.append(100.0 * wins / denominator)
        scores[method] = {
            "S1": turn_scores[0], "S2": turn_scores[1], "S3": turn_scores[2],
            "Avg": sum(turn_scores[:3]) / 3,
            "overall_official_secondary": (turn_scores[3] if len(judge_turns) == 4 else None),
            "n_conversations": denominator,
            "denominators_by_turn": denominators_by_turn,
            "model_wins_by_turn": wins_by_turn,
            "valid_decisions_by_turn": valid_by_turn,
            "unresolved_by_turn": unresolved_by_turn,
            "missing_by_turn": missing_by_turn,
            "total_valid_decisions": sum(valid_by_turn.values()),
            "total_unresolved": sum(unresolved_by_turn.values()),
            "total_missing": sum(missing_by_turn.values()),
            "unresolved_rate": sum(unresolved_by_turn.values()) / (denominator * len(judge_turns)),
        }
    return scores


class LocalLlamaJudge:
    def __init__(self, checkpoint: str | Path, *, max_new_tokens: int = 512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the local 8B judge")
        self.torch = torch
        self.checkpoint = str(checkpoint)
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint, local_files_only=True, dtype=torch.bfloat16,
            device_map={"": 0}, attn_implementation="sdpa",
        )
        self.model.eval()

    def generate(self, messages: list[dict[str, str]], *,
                 max_new_tokens: int | None = None) -> tuple[str, int, int]:
        tokens = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        prompt_tokens = int(tokens.shape[-1])
        max_context = int(self.model.config.max_position_embeddings)
        output_cap = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        if output_cap <= 0 or prompt_tokens + output_cap > max_context:
            raise ValueError(f"Judge prompt exceeds context: {prompt_tokens}+{output_cap}>{max_context}")
        tokens = tokens.to(self.model.device)
        attention_mask = self.torch.ones_like(tokens, dtype=self.torch.long)
        with self.torch.inference_mode():
            output = self.model.generate(
                input_ids=tokens, attention_mask=attention_mask,
                do_sample=False, temperature=None, top_p=None,
                max_new_tokens=output_cap,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, prompt_tokens:]
        return self.tokenizer.decode(generated, skip_special_tokens=True), prompt_tokens, int(generated.numel())

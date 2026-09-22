#!/usr/bin/env python3
"""Run the official ConvBench pairwise protocol with a local Llama judge.

Default mode checks source/index/prompt wiring without loading a model.
``--execute`` judges at most five conversations unless ``--limit``,
``--smoke-ids``, or ``--full`` is explicitly supplied. ``--stage-only`` omits the
optional overall-conversation judgment. One atomic checkpoint per
conversation/method supports safe resume; ``judge_raw.jsonl`` and
``quality_summary.json`` are rebuilt from those checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mmimpress.convbench_judge import (  # noqa: E402
    MODEL_ID, TURN_NAMES, LocalLlamaJudge, build_messages,
    extract_fallback_winner, extract_official_winner,
    extraction_messages,
    load_official_prompt_builder, load_pairwise, normalize_row,
    official_revision, score_records, sha256_file,
)

METHODS = ("ReComp", "FullLoad", "Prefix25", "Prefix45")
DEFAULT_CHECKPOINT = Path(
    "/home/dblab/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct"
    "/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def stable_content_hash(obj: Any) -> str:
    encoded = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_checkpoint(path: Path, obj: dict[str, Any]) -> None:
    obj = dict(obj)
    obj["artifact_content_sha256"] = stable_content_hash(obj)
    atomic_json(path, obj)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_index(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["conversations"] if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise ValueError("ConvBench index must contain a nonempty conversations list")
    normalized = [normalize_row(row) for row in rows]
    for source, row in zip(rows, normalized):
        if "source_id" in source:
            row["source_id"] = str(source["source_id"])
    ids = [row["conversation_id"] for row in normalized]
    indices = [row["source_row_index"] for row in normalized]
    if len(ids) != len(set(ids)) or len(indices) != len(set(indices)):
        raise ValueError("Duplicate conversation ID or source row index")
    return normalized


def select_rows(rows: list[dict[str, Any]], limit: int, full: bool,
                smoke_ids: str | None) -> list[dict[str, Any]]:
    if smoke_ids is None:
        return rows if full else rows[:limit]
    if full:
        raise ValueError("--smoke-ids cannot be combined with --full")
    source_ids = [part.strip() for part in smoke_ids.split(",")]
    if (not 1 <= len(source_ids) <= 10 or
            any(not source_id.isdecimal() or int(source_id) < 1 or
                str(int(source_id)) != source_id for source_id in source_ids) or
            len(source_ids) != len(set(source_ids))):
        raise ValueError("--smoke-ids requires 1..10 unique positive official source IDs")
    by_source_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = row.get("source_id")
        if source_id is None:
            raise ValueError("Official source_id missing from ConvBench index")
        if row["conversation_id"] != f"convbench:{source_id}":
            raise ValueError(f"Source ID/conversation mismatch: {row['conversation_id']}")
        if source_id in by_source_id:
            raise ValueError(f"Duplicate official source ID: {source_id}")
        by_source_id[source_id] = row
    missing = [source_id for source_id in source_ids if source_id not in by_source_id]
    if missing:
        raise ValueError(f"Official source IDs absent from validated index: {missing}")
    return [by_source_id[source_id] for source_id in source_ids]


def _turn_number(value: Any) -> int:
    if isinstance(value, int) and value in (1, 2, 3):
        return value
    stripped = str(value).upper().replace("TURN", "").replace("T", "").strip(" _-")
    if stripped in ("1", "2", "3"):
        return int(stripped)
    raise ValueError(f"Invalid answer turn_id: {value}")


def _answer_value(record: dict[str, Any]) -> str:
    for name in ("prediction", "generated_answer", "answer", "response", "text"):
        value = record.get(name)
        if isinstance(value, str):
            return value
    raise ValueError("Answer row has no generated text")


def answer_input_hash(path: Path, selected: list[dict[str, Any]]) -> str:
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    config = path / "config.json"
    if not config.is_file():
        raise ValueError(f"Answer run directory has no config.json: {path}")
    digest.update(sha256_file(config).encode())
    for row in selected:
        artifact = path / "conversations" / f"{row['conversation_id']}.json"
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        digest.update(row["conversation_id"].encode())
        digest.update(sha256_file(artifact).encode())
    return digest.hexdigest()


def read_answers(path: Path, selected: list[dict[str, Any]],
                 expected_index_sha256: str) -> dict[tuple[str, str], list[str]]:
    staged: dict[tuple[str, str], dict[int, str]] = {}
    if path.is_dir():
        incoming = []
        for row in selected:
            artifact = path / "conversations" / f"{row['conversation_id']}.json"
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            body = {key: value for key, value in payload.items() if key != "artifact_content_sha256"}
            if payload.get("artifact_content_sha256") != stable_content_hash(body):
                raise ValueError(f"Corrupt answer artifact {artifact}")
            if payload.get("conversation_id") != row["conversation_id"] or payload.get(
                    "source_row_index") != row["source_row_index"] or payload.get(
                    "index_sha256") != expected_index_sha256:
                raise ValueError(f"Answer artifact/source mismatch: {artifact}")
            incoming.extend(payload["rows"])
    else:
        with open(path, encoding="utf-8") as handle:
            incoming = [json.loads(line) for line in handle if line.strip()]
    for line_no, record in enumerate(incoming, 1):
        method = str(record["method"])
        if method not in METHODS:
            raise ValueError(f"Unknown method at line {line_no}: {method}")
        conversation_id = str(record.get("conversation_id", record.get("dialogue_id", "")))
        if not conversation_id:
            raise ValueError(f"Missing conversation ID at line {line_no}")
        key = (method, conversation_id)
        group = staged.setdefault(key, {})
        if "answers" in record and isinstance(record["answers"], list):
            if len(record["answers"]) != 3:
                raise ValueError(f"Expected three answers at line {line_no}")
            for turn, answer in enumerate(record["answers"], 1):
                if turn in group:
                    raise ValueError(f"Duplicate answer {key}/{turn}")
                group[turn] = str(answer)
        else:
            turn = _turn_number(record.get("turn_id", record.get("turn")))
            if turn in group:
                raise ValueError(f"Duplicate answer {key}/{turn}")
            group[turn] = _answer_value(record)
    answers = {}
    for key, group in staged.items():
        if set(group) != {1, 2, 3}:
            raise ValueError(f"Incomplete three-turn answers for {key}: {sorted(group)}")
        answers[key] = [group[turn] for turn in (1, 2, 3)]
    return answers


def validate_checkpoint(path: Path, method: str, row: dict[str, Any], position: int,
                        judge_turns: tuple[str, ...] = TURN_NAMES) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "artifact_content_sha256"}
    if payload.get("artifact_content_sha256") != stable_content_hash(body):
        raise ValueError(f"Corrupt judge checkpoint hash {path}")
    records = payload["records"]
    if (not 1 <= len(records) <= len(judge_turns) or
            [record["judge_turn"] for record in records] != list(judge_turns[:len(records)]) or
            payload.get("status") != ("complete" if len(records) == len(judge_turns) else "partial")):
        raise ValueError(f"Corrupt judge checkpoint {path}")
    for record in records:
        if (record["method"], record["conversation_id"], record["source_row_index"], record["position"]) != (
            method, row["conversation_id"], row["source_row_index"], position
        ):
            raise ValueError(f"Mismatched judge checkpoint {path}")
        if record.get("winner") not in ("A", "B", None):
            raise ValueError(f"Invalid judge checkpoint {path}")
        if record.get("model_wins") is not (None if record["winner"] is None else
                                             record["winner"] == ("A" if position == 0 else "B")):
            raise ValueError(f"Inconsistent judge checkpoint decision {path}")
    return records


def make_record(judge: LocalLlamaJudge, builder: Any, row: dict[str, Any],
                predictions: list[str], method: str, position: int,
                turn_name: str, earlier: list[str]) -> dict[str, Any]:
    messages = build_messages(builder, row, predictions, position, turn_name, earlier)
    raw, prompt_tokens, output_tokens = judge.generate(messages)
    winner = extract_official_winner(raw)
    parse_stage = "first_pass"
    extraction_raw = None
    extraction_prompt_tokens = None
    extraction_output_tokens = None
    if winner is None:
        extraction_raw, extraction_prompt_tokens, extraction_output_tokens = judge.generate(
            extraction_messages(raw), max_new_tokens=16
        )
        winner = extract_fallback_winner(extraction_raw)
        parse_stage = "second_pass" if winner is not None else "unresolved"
    return {
        "conversation_id": row["conversation_id"],
        "source_row_index": row["source_row_index"],
        "method": method,
        "judge_turn": turn_name,
        "position": position,
        "pairwise_dict": {"A": "model_answer", "B": "human_answer"} if position == 0 else
                         {"A": "human_answer", "B": "model_answer"},
        "messages": messages,
        "raw_response": raw,
        "winner": winner,
        "model_wins": (winner == ("A" if position == 0 else "B")
                       if winner is not None else None),
        "parse_stage": parse_stage,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "extraction_raw_response": extraction_raw,
        "extraction_prompt_tokens": extraction_prompt_tokens,
        "extraction_output_tokens": extraction_output_tokens,
    }


def recover_record(old: dict[str, Any], judge: LocalLlamaJudge | None) -> dict[str, Any]:
    """Preserve an archived first pass and rerun only its failed extraction."""
    record = dict(old)
    direct = extract_official_winner(record["raw_response"])
    if direct is not None:
        if old.get("winner") != direct:
            raise ValueError("Archived first-pass decision differs from its raw judgment")
        record["parse_stage"] = "first_pass"
    else:
        if old.get("winner") != "C":
            raise ValueError("Archived unresolved judgment is not the expected C fallback")
        if judge is None:
            raise ValueError("Local Llama judge required for second-pass recovery")
        record["extraction_raw_response"], record["extraction_prompt_tokens"], \
            record["extraction_output_tokens"] = judge.generate(
                extraction_messages(record["raw_response"]), max_new_tokens=16)
        record["winner"] = extract_fallback_winner(record["extraction_raw_response"])
        record["parse_stage"] = "second_pass" if record["winner"] else "unresolved"
    record["archived_parse_stage"] = old["parse_stage"]
    record["archived_winner"] = old["winner"]
    record["archived_extraction_raw_response"] = old.get("extraction_raw_response")
    record["model_wins"] = (record["winner"] == ("A" if record["position"] == 0 else "B")
                            if record["winner"] is not None else None)
    return record


def read_recovery_checkpoint(path: Path, method: str, row: dict[str, Any],
                             position: int,
                             judge_turns: tuple[str, ...] = TURN_NAMES) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "artifact_content_sha256"}
    if payload.get("artifact_content_sha256") != stable_content_hash(body):
        raise ValueError(f"Corrupt archived judge checkpoint {path}")
    records = payload.get("records")
    if (payload.get("status") != "complete" or not isinstance(records, list) or
            len(records) != len(judge_turns)):
        raise ValueError(f"Incomplete archived judge checkpoint {path}")
    for turn, record in zip(judge_turns, records, strict=True):
        if (record.get("method"), record.get("conversation_id"),
            record.get("source_row_index"), record.get("position"),
            record.get("judge_turn")) != (
                method, row["conversation_id"], row["source_row_index"], position, turn):
            raise ValueError(f"Archived judge checkpoint identity mismatch {path}")
        if record.get("winner") not in ("A", "B", "C"):
            raise ValueError(f"Invalid archived judge winner {path}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/convbench_source")
    parser.add_argument("--index", type=Path, default=ROOT / "data/convbench/index.json")
    parser.add_argument("--answers", type=Path, help="Generation JSONL with method/conversation/turn/prediction")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/convbench_judge/smoke_5")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--smoke-ids", type=str,
                        help="Comma-separated official source IDs (up to 10); overrides --limit")
    parser.add_argument("--full", action="store_true", help="Allow all index conversations (only after full-run approval)")
    parser.add_argument("--stage-only", action="store_true",
                        help="Run only official first, second, and third stage judgments")
    parser.add_argument("--execute", action="store_true", help="Load local Llama judge and run inference")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--recover-from", type=Path,
                        help="Reuse archived first-pass judgments and rerun only failed extraction")
    args = parser.parse_args()

    if args.limit <= 0 or args.max_new_tokens <= 0:
        parser.error("--limit and --max-new-tokens must be positive")
    judge_turns = TURN_NAMES[:3] if args.stage_only else TURN_NAMES
    rows = read_index(args.index)
    selected = select_rows(rows, args.limit, args.full, args.smoke_ids)
    positions = load_pairwise(args.source, [row["source_row_index"] for row in selected])
    builder, prompts = load_official_prompt_builder(args.source)
    # Validate the exact official function against all selected rows before GPU load.
    for row in selected:
        for turn in judge_turns:
            messages = build_messages(builder, row, ["test answer"] * 3,
                                      positions[row["source_row_index"]], turn,
                                      ["Overall, Response A is better."] * 3)
            if len(messages) != 4 or any(not msg["content"] for msg in messages):
                raise ValueError(f"Invalid official prompt for {row['conversation_id']}/{turn}")

    provenance = {
        "official_repository": "https://github.com/shirlyliu64/ConvBench",
        "official_commit": official_revision(args.source),
        "official_evaluator_sha256": sha256_file(args.source / "ConvBenchEval/convbencheval.py"),
        "official_prompts_sha256": sha256_file(args.source / "ConvBenchEval/prompts.py"),
        "official_pairwise_sha256": sha256_file(args.source / "ConvBenchEval/pairwise.npy"),
        "local_judge_module_sha256": sha256_file(ROOT / "mmimpress/convbench_judge.py"),
        "local_judge_cli_sha256": sha256_file(Path(__file__)),
        "index_sha256": sha256_file(args.index),
        "answer_sha256": answer_input_hash(args.answers, selected) if args.answers else None,
        "model_id": MODEL_ID,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_revision": args.checkpoint.name,
        "dtype": "bfloat16",
        "quantization": "none",
        "attention": "sdpa",
        "attention_mask": "explicit_all_ones_for_unpadded_single_prompt",
        "decoding": "greedy; temperature=0; do_sample=False",
        "max_new_tokens": args.max_new_tokens,
        "selected_source_row_indices": [row["source_row_index"] for row in selected],
        "score_scale": "100 * model wins / fixed selected conversation count per method and turn",
        "score_denominator_per_method_turn": len(selected),
        "score_unresolved_policy": "unresolved/C earns zero wins and remains in fixed denominator",
        "judge_turns": list(judge_turns),
        "second_pass_policy": "strict_raw_preference_extraction_v1",
        "second_pass_model_id": MODEL_ID,
        "second_pass_max_new_tokens": 16,
        "commercial_api_used": False,
    }
    if args.recover_from is not None:
        if args.out.resolve() == args.recover_from.resolve():
            raise ValueError("Recovery output must differ from archived judge run")
        source_config_path = args.recover_from / "config.json"
        source_raw_path = args.recover_from / "judge_raw.jsonl"
        archived = json.loads(source_config_path.read_text(encoding="utf-8"))
        for key in ("official_commit", "official_evaluator_sha256", "official_prompts_sha256",
                    "official_pairwise_sha256", "index_sha256", "answer_sha256", "model_id",
                    "checkpoint", "checkpoint_revision", "max_new_tokens",
                    "selected_source_row_indices", "judge_turns"):
            if archived.get(key) != provenance.get(key):
                raise ValueError(f"Archived judge provenance mismatch: {key}")
        if not source_raw_path.is_file():
            raise FileNotFoundError(source_raw_path)
        provenance["recovery_source"] = str(args.recover_from.resolve())
        provenance["recovery_source_config_sha256"] = sha256_file(source_config_path)
        provenance["recovery_source_judge_raw_sha256"] = sha256_file(source_raw_path)
    if not args.execute:
        print(json.dumps({"status": "prompt_validation_passed", "n_conversations": len(selected),
                          "source_commit": provenance["official_commit"],
                          "pairwise_positions": len(positions)}, indent=2))
        return
    if args.answers is None:
        parser.error("--answers is required with --execute")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Local judge checkpoint absent: {args.checkpoint}")
    answers = read_answers(args.answers, selected, provenance["index_sha256"])
    selected_ids = {row["conversation_id"] for row in selected}
    expected = {(method, cid) for method in METHODS for cid in selected_ids}
    missing = expected - set(answers)
    if missing:
        raise ValueError(f"Missing generated three-turn answers: {sorted(missing)[:8]}")
    if args.out.exists() and (args.out / "config.json").exists():
        previous = json.loads((args.out / "config.json").read_text(encoding="utf-8"))
        if previous != provenance:
            raise ValueError("Judge resume refused: config/source/answers mismatch")
        prior_provenance = args.out / "provenance.json"
        if prior_provenance.exists() and json.loads(prior_provenance.read_text(encoding="utf-8")) != provenance:
            raise ValueError("Judge resume refused: provenance mismatch")
    else:
        if args.out.exists() and any(args.out.iterdir()):
            raise ValueError("Judge output directory exists without config.json")
        atomic_json(args.out / "config.json", provenance)

    judge = None
    all_records: list[dict[str, Any]] = []
    for row in selected:
        cid, source_index = row["conversation_id"], row["source_row_index"]
        position = positions[source_index]
        for method in METHODS:
            checkpoint = args.out / "judgments" / method / f"{source_index:04d}.json"
            records = (validate_checkpoint(checkpoint, method, row, position, judge_turns)
                       if checkpoint.exists() else [])
            if len(records) < len(judge_turns):
                if args.recover_from is not None:
                    archived_path = (args.recover_from / "judgments" / method /
                                     f"{source_index:04d}.json")
                    old_records = read_recovery_checkpoint(archived_path, method, row, position,
                                                           judge_turns)
                    for old in old_records[len(records):]:
                        if old["winner"] == "C" and judge is None:
                            judge = LocalLlamaJudge(args.checkpoint,
                                                    max_new_tokens=args.max_new_tokens)
                        records.append(recover_record(old, judge))
                        atomic_checkpoint(checkpoint, {
                            "records": records,
                            "status": "complete" if len(records) == len(judge_turns) else "partial",
                        })
                else:
                    if judge is None:
                        judge = LocalLlamaJudge(args.checkpoint,
                                                max_new_tokens=args.max_new_tokens)
                    earlier = [record["raw_response"] for record in records[:3]]
                    for turn in judge_turns[len(records):]:
                        record = make_record(judge, builder, row, answers[(method, cid)],
                                             method, position, turn, earlier)
                        records.append(record)
                        if turn != judge_turns[-1]:
                            earlier.append(record["raw_response"])
                        atomic_checkpoint(checkpoint, {
                            "records": records,
                            "status": "complete" if len(records) == len(judge_turns) else "partial",
                        })
            all_records.extend(records)
    expected_decisions = len(selected) * len(METHODS) * len(judge_turns)
    if len(all_records) != expected_decisions:
        raise ValueError(f"Incomplete judge decisions: {len(all_records)} != {expected_decisions}")
    summary = score_records(all_records, expected_per_method=len(selected),
                            judge_turns=judge_turns)
    atomic_jsonl(args.out / "judge_raw.jsonl", all_records)
    atomic_json(args.out / "quality_summary.json", summary)
    parse_stages = {stage: sum(record["parse_stage"] == stage for record in all_records)
                    for stage in ("first_pass", "second_pass", "unresolved")}
    if sum(parse_stages.values()) != len(all_records):
        raise ValueError("Unknown judge parse stage in completed run")
    atomic_json(args.out / "score_provenance.json", {
        "scale": "100 * number of model wins / fixed number of selected conversations per method and turn",
        "S1_S2_S3": "first, second, third pairwise decisions respectively",
        "Avg": "(S1 + S2 + S3) / 3",
        "overall_conversation_semantics": (
            "not_executed" if args.stage_only else "secondary diagnostic, excluded from Avg"),
        "unresolved_semantics": "winner=null/C; zero model wins; included in fixed denominator",
        "n_conversations_per_method": len(selected),
        "fixed_denominator_by_stage": {stage: len(selected) for stage in ("S1", "S2", "S3")},
        "n_judgments": len(all_records),
        "total_expected_judge_decisions": expected_decisions,
        "first_pass_parsed_count": parse_stages["first_pass"],
        "second_pass_recovered_count": parse_stages["second_pass"],
        "unresolved_count": parse_stages["unresolved"],
        "unresolved_rate": parse_stages["unresolved"] / len(all_records),
        "parse_stages": parse_stages,
        "judge_raw_sha256": sha256_file(args.out / "judge_raw.jsonl"),
        "config_sha256": sha256_file(args.out / "config.json"),
    })
    atomic_json(args.out / "provenance.json", provenance)
    print(json.dumps({"status": "complete", "n_judgments": len(all_records), "quality": summary}, indent=2))


if __name__ == "__main__":
    main()

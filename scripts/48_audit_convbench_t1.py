#!/usr/bin/env python3
"""Audit archived seven-case Turn-1 generation and judge inputs without inference.

The answer runner saved the first generated ID and count, but not the full
generated ID stream. Retokenizing decoded text is marked as diagnostic only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mmimpress.convbench_judge import (  # noqa: E402
    build_messages, load_official_prompt_builder, load_pairwise, normalize_row,
)

IDS = ("1", "2", "3", "4", "5", "494", "418")
METHODS = ("ReComp", "FullLoad", "Prefix25", "Prefix45")
ANSWER_RUN = ROOT / "runs/convbench_context/targeted_7_native"
JUDGE_RUN = ROOT / "runs/convbench_context/judge_targeted_7_llama_second_pass"
OUT = ROOT / "runs/convbench_context/turn1_cross_method_audit"
ANSWER_TOKENIZER = (Path("/home/dblab/.cache/huggingface/hub") /
                    "models--llava-hf--llava-v1.6-vicuna-7b-hf/snapshots/"
                    "c916e6cdcd760b4cecd1dd4907f84ac649f93b23")
JUDGE_TOKENIZER = (Path("/home/dblab/.cache/huggingface/hub") /
                   "models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/"
                   "0e9e39f249a16976918f6564b8830bc894c89659")


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha_json(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    from transformers import AutoTokenizer

    answer_config = json.loads((ANSWER_RUN / "config.json").read_text(encoding="utf-8"))
    if answer_config["selected_source_ids"] != list(IDS) or answer_config["full_run"]:
        raise ValueError("Expected the frozen targeted seven-case answer run")
    index = json.loads((ROOT / "data/convbench/index.json").read_text(encoding="utf-8"))
    source_rows = {str(row["source_id"]): normalize_row(row)
                   for row in index["conversations"] if str(row["source_id"]) in IDS}
    positions = load_pairwise(ROOT / "data/convbench_source",
                              [source_rows[source_id]["source_row_index"] for source_id in IDS])
    builder, _ = load_official_prompt_builder(ROOT / "data/convbench_source")
    answer_tokenizer = AutoTokenizer.from_pretrained(
        ANSWER_TOKENIZER, local_files_only=True)
    judge_tokenizer = AutoTokenizer.from_pretrained(
        JUDGE_TOKENIZER, local_files_only=True)

    judge_raw = JUDGE_RUN / "judge_raw.jsonl"
    judge_records = [json.loads(line) for line in judge_raw.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    judge_t1 = {(r["conversation_id"], r["method"]): r for r in judge_records
                if r["judge_turn"] == "_first_turn"}
    if len(judge_t1) != len(IDS) * len(METHODS):
        raise ValueError("Missing or duplicate archived Turn-1 judge rows")

    details = []
    flat = []
    wins = defaultdict(int)
    same_generation = 0
    same_judge_input_pairs = 0
    different_label_cases = []
    for source_id in IDS:
        cid = f"convbench:{source_id}"
        path = ANSWER_RUN / "conversations" / f"{cid}.json"
        artifact = json.loads(path.read_text(encoding="utf-8"))
        body = {key: value for key, value in artifact.items()
                if key != "artifact_content_sha256"}
        if artifact["artifact_content_sha256"] != sha_json(body):
            raise ValueError(f"Corrupt answer artifact {path}")
        by_method_turn = {(r["method"], r["turn_id"]): r for r in artifact["rows"]}
        if set(by_method_turn) != {(method, turn) for method in METHODS
                                  for turn in (1, 2, 3)}:
            raise ValueError(f"Incomplete method-turn answer rows for {cid}")
        base = by_method_turn[("ReComp", 1)]
        prompt_hash = sha_text(base["prompt"])
        answer_hash = sha_text(base["prediction"])
        if prompt_hash != base["prompt_sha256"]:
            raise ValueError(f"Stored Turn-1 prompt hash mismatch for {cid}")
        retokenized_ids = answer_tokenizer.encode(
            base["prediction"], add_special_tokens=False)
        if not retokenized_ids or retokenized_ids[0] != base["first_token_id"]:
            raise ValueError(f"Retokenized first ID differs from recorded first ID for {cid}")
        if len(retokenized_ids) != base["answer_tokens"]:
            raise ValueError(f"Retokenized A1 length differs from recorded answer_tokens for {cid}")
        position = positions[source_rows[source_id]["source_row_index"]]
        per_method = {}
        agreements = {}
        for method in METHODS:
            row = by_method_turn[(method, 1)]
            judge = judge_t1[(cid, method)]
            if row["prompt_sha256"] != sha_text(row["prompt"]):
                raise ValueError(f"Stored prompt hash mismatch for {cid}/{method}")
            if (row["request_path"] != "normal_multimodal_pixel" or
                    row["vision_forward_count"] != 1 or row["ssd_read_bytes"] != 0):
                raise ValueError(f"Turn-1 pixel path mismatch for {cid}/{method}")
            same = all((row[field] == base[field] for field in (
                "prompt", "prompt_sha256", "prediction", "first_token_id",
                "generated_tokens", "input_tensors_sha256")))
            agreements[method] = same
            if not same:
                raise ValueError(f"Method-dependent Turn-1 generation for {cid}/{method}")
            predictions = [by_method_turn[(method, turn)]["prediction"]
                           for turn in (1, 2, 3)]
            official_messages = build_messages(
                builder, source_rows[source_id], predictions, position, "_first_turn")
            if judge["messages"] != official_messages or judge["position"] != position:
                raise ValueError(f"Archived official judge input mismatch for {cid}/{method}")
            model_letter = "A" if position == 0 else "B"
            expected_mapping = {model_letter: "model_answer",
                                "B" if model_letter == "A" else "A": "human_answer"}
            if judge["pairwise_dict"] != expected_mapping:
                raise ValueError(f"Judge pairwise mapping mismatch for {cid}/{method}")
            label = judge["winner"]
            if label not in ("A", "B", None):
                raise ValueError(f"Unexpected final judge label for {cid}/{method}")
            win = int(label == model_letter)
            wins[method] += win
            if judge["model_wins"] is not (bool(win) if label is not None else None):
                raise ValueError(f"Archived judge win flag mismatch for {cid}/{method}")
            rendered = judge_tokenizer.apply_chat_template(
                judge["messages"], tokenize=False, add_generation_prompt=True)
            tokens = judge_tokenizer.apply_chat_template(
                judge["messages"], tokenize=True, add_generation_prompt=True)
            if len(tokens) != judge["prompt_tokens"]:
                raise ValueError(f"Archived judge prompt token count mismatch for {cid}/{method}")
            per_method[method] = {
                "generation_prompt_sha256": row["prompt_sha256"],
                "generation_input_tensors_sha256": row["input_tensors_sha256"],
                "generated_A1_text_sha256": sha_text(row["prediction"]),
                "original_generated_A1_token_ids": None,
                "first_token_id": row["first_token_id"],
                "first_token_text": answer_tokenizer.decode(
                    [row["first_token_id"]], skip_special_tokens=False),
                "generated_token_count_including_specials": row["generated_tokens"],
                "retokenized_A1_text_token_ids": retokenized_ids,
                "retokenized_A1_text_token_ids_sha256": sha_json(retokenized_ids),
                "A2_text_sha256": sha_text(predictions[1]),
                "A3_text_sha256": sha_text(predictions[2]),
                "judge_pairwise_position": position,
                "judge_messages_sha256": sha_json(judge["messages"]),
                "judge_prompt_sha256": sha_text(rendered),
                "judge_prompt_token_count": len(tokens),
                "final_judge_label": label,
                "scored_model_win": win,
            }
        same_generation += 1
        baseline = per_method["ReComp"]
        baseline.update({"A2_same_as_ReComp": True, "A3_same_as_ReComp": True,
                         "judge_prompt_same_as_ReComp": True,
                         "judge_label_same_as_ReComp": True})
        for method in METHODS[1:]:
            current = per_method[method]
            same_judge = current["judge_messages_sha256"] == baseline["judge_messages_sha256"]
            a2_same = current["A2_text_sha256"] == baseline["A2_text_sha256"]
            a3_same = current["A3_text_sha256"] == baseline["A3_text_sha256"]
            later_same = a2_same and a3_same
            current.update({"A2_same_as_ReComp": a2_same,
                            "A3_same_as_ReComp": a3_same,
                            "judge_prompt_same_as_ReComp": same_judge,
                            "judge_label_same_as_ReComp": (
                                current["final_judge_label"] == baseline["final_judge_label"])})
            if same_judge != later_same:
                raise ValueError(f"Judge input difference not explained by A2/A3 for {cid}/{method}")
            if same_judge:
                same_judge_input_pairs += 1
                if current["final_judge_label"] != baseline["final_judge_label"]:
                    raise ValueError(f"Identical judge input has different label for {cid}/{method}")
            if current["final_judge_label"] != baseline["final_judge_label"]:
                if same_judge:
                    raise ValueError(f"Unexplained Turn-1 judge label disagreement for {cid}/{method}")
                different_label_cases.append(f"{cid}:{method}")
        for left, right in combinations(METHODS, 2):
            left_item, right_item = per_method[left], per_method[right]
            same_judge = left_item["judge_prompt_sha256"] == right_item["judge_prompt_sha256"]
            same_later = (left_item["A2_text_sha256"] == right_item["A2_text_sha256"] and
                          left_item["A3_text_sha256"] == right_item["A3_text_sha256"])
            if same_judge != same_later:
                raise ValueError(f"Judge input difference not explained by A2/A3 for {cid}/{left}/{right}")
            if same_judge and left_item["final_judge_label"] != right_item["final_judge_label"]:
                raise ValueError(f"Identical judge input has different label for {cid}/{left}/{right}")
        detail = {
            "conversation_id": cid,
            "generation_prompt_sha256": prompt_hash,
            "generated_A1_text_sha256": answer_hash,
            "original_generated_A1_token_ids_recorded": False,
            "original_generated_A1_token_ids": None,
            "retokenized_A1_text_token_ids": retokenized_ids,
            "retokenized_A1_text_token_ids_sha256": sha_json(retokenized_ids),
            "first_token_id": base["first_token_id"],
            "first_token_text": answer_tokenizer.decode(
                [base["first_token_id"]], skip_special_tokens=False),
            "generated_token_count_including_specials": base["generated_tokens"],
            "ReComp_vs_FullLoad_generation_agreement": agreements["FullLoad"],
            "ReComp_vs_Prefix25_generation_agreement": agreements["Prefix25"],
            "ReComp_vs_Prefix45_generation_agreement": agreements["Prefix45"],
            "judge_pairwise_position": position,
            "methods": per_method,
        }
        details.append(detail)
        flat.append({
            "conversation_id": cid,
            "generation_prompt_sha256": prompt_hash,
            "generated_A1_text_sha256": answer_hash,
            "first_token_id": base["first_token_id"],
            "first_token_text": detail["first_token_text"],
            "generated_token_count_including_specials": base["generated_tokens"],
            "original_generated_A1_token_ids_recorded": False,
            "retokenized_A1_text_token_ids_sha256": detail[
                "retokenized_A1_text_token_ids_sha256"],
            "ReComp_vs_FullLoad_generation_agreement": agreements["FullLoad"],
            "ReComp_vs_Prefix25_generation_agreement": agreements["Prefix25"],
            "ReComp_vs_Prefix45_generation_agreement": agreements["Prefix45"],
            "judge_pairwise_position": position,
            **{f"{method}_judge_prompt_sha256": per_method[method][
                "judge_prompt_sha256"] for method in METHODS},
            **{f"{method}_final_judge_label": per_method[method][
                "final_judge_label"] or "unresolved/C" for method in METHODS},
            **{f"{method}_scored_model_win": per_method[method][
                "scored_model_win"] for method in METHODS},
            **{f"{method}_A2_same_as_ReComp": per_method[method][
                "A2_same_as_ReComp"] for method in METHODS},
            **{f"{method}_A3_same_as_ReComp": per_method[method][
                "A3_same_as_ReComp"] for method in METHODS},
            **{f"{method}_judge_prompt_same_as_ReComp": per_method[method][
                "judge_prompt_same_as_ReComp"] for method in METHODS},
        })
    if wins != {"ReComp": 1, "FullLoad": 1, "Prefix25": 0, "Prefix45": 0}:
        raise ValueError(f"Unexpected targeted Turn-1 win totals: {wins}")
    output = {
        "hash_definitions": {
            "generation_prompt_sha256": "SHA-256 of archived prompt UTF-8 bytes",
            "generated_A1_text_sha256": "SHA-256 of archived prediction UTF-8 bytes",
            "judge_messages_sha256": "SHA-256 of canonical UTF-8 JSON of archived messages",
            "judge_prompt_sha256": "SHA-256 of UTF-8 chat-template text rendered from archived messages with add_generation_prompt=True",
            "retokenized_A1_text_token_ids_sha256": "SHA-256 of canonical JSON array of tokenizer IDs re-encoded from saved A1 text",
        },
        "limitation": ("Original complete generated token-ID streams were not saved. "
                       "The listed retokenized IDs are derived from decoded/stripped A1 text; "
                       "they cannot prove the exact original stream or a removed EOS token."),
        "answer_run_config_sha256": hashlib.sha256((ANSWER_RUN / "config.json").read_bytes()).hexdigest(),
        "judge_raw_sha256": hashlib.sha256(judge_raw.read_bytes()).hexdigest(),
        "conversations": details,
        "summary": {
            "conversations": len(details),
            "turn1_generation_equal_across_all_methods": same_generation,
            "identical_judge_input_pairs_vs_ReComp": same_judge_input_pairs,
            "different_label_cases_vs_ReComp": different_label_cases,
            "turn1_model_wins_by_method": dict(wins),
            "turn1_fixed_denominator": len(IDS),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()

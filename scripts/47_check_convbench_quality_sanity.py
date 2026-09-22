#!/usr/bin/env python3
"""Conservative, reference-free output sanity check for ConvBench smoke runs.

This is a diagnostic gate for obvious generation failures. It does not score
answer correctness. Finite logits are only certified when a native probe's
per-forward diagnostic hook recorded them; a normal answer artifact alone
cannot establish that property.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


TOKEN_RE = re.compile(r"[A-Za-z]+|\d+|[^\w\s]", re.UNICODE)
METHODS = {"recompute", "fullload", "prefix25", "prefix45"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repeated_segments(answer: str) -> list[dict]:
    """Find strong literal repetition signals, with transparent thresholds."""
    found: list[dict] = []
    tokens = TOKEN_RE.findall(answer)
    run_start = 0
    for i in range(1, len(tokens) + 1):
        if i < len(tokens) and tokens[i] == tokens[run_start]:
            continue
        run = i - run_start
        if run >= 16 and tokens[run_start].isalnum():
            found.append({"kind": "identical_lexical_token_run", "copies": run,
                          "unit": tokens[run_start], "threshold": 16})
        run_start = i

    for match in re.finditer(r"(\d)\1{11,}|([A-Za-z])\2{23,}", answer):
        segment = match.group()
        found.append({"kind": "identical_character_run", "copies": len(segment),
                      "unit": segment[0], "threshold": 12 if segment[0].isdigit() else 24})

    # A multi-token phrase repeated >=6 times, spanning >=18 lexical/punctuation
    # tokens, is a strong signal. Keep only the longest repetition found there.
    best = None
    for width in range(2, 17):
        for i in range(len(tokens) - width * 6 + 1):
            unit = tokens[i:i + width]
            copies = 1
            while tokens[i + copies * width:i + (copies + 1) * width] == unit:
                copies += 1
            if copies < 6 or copies * width < 18:
                continue
            candidate = {"kind": "repeated_token_phrase", "copies": copies,
                         "unit": " ".join(unit), "unit_tokens": width,
                         "span_tokens": copies * width, "threshold": 6}
            if best is None or candidate["span_tokens"] > best["span_tokens"]:
                best = candidate
    if best is not None:
        found.append(best)
    return found


def load_run(run_dir: Path) -> tuple[list[dict], list[dict]]:
    journal = run_dir / "request_journal.jsonl"
    sources: list[dict] = []
    if journal.is_file():
        sources.append({"path": str(journal), "sha256": sha256_file(journal)})
        rows = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
        mode = "native_probe"
    else:
        rows = []
        mode = "answer_run"

    conversation_files = sorted((run_dir / "conversations").glob("convbench:*.json"))
    if not conversation_files:
        raise ValueError(f"No conversation artifacts found: {run_dir}")
    indexed = {}
    for path in conversation_files:
        sources.append({"path": str(path), "sha256": sha256_file(path)})
        conversation = json.loads(path.read_text())
        for row in conversation["rows"]:
            key = (row["conversation_id"], int(row["turn_id"]), row["method_key"])
            if key in indexed:
                raise ValueError(f"Duplicate answer artifact row: {key}")
            indexed[key] = row
    if mode == "answer_run":
        rows = [{"conversation_id": row["conversation_id"],
                 "turn_id": row["turn_id"], "method_key": row["method_key"],
                 "answer": row["prediction"], "generated_tokens": row["generated_tokens"],
                 "first_token_id": row["first_token_id"],
                 "first_token_success": row["first_token_id"] is not None,
                 "runtime_success": True, "logit_evidence_available": False,
                 "effective_max_new_tokens": row["effective_max_new_tokens"]}
                for row in indexed.values()]
    else:
        if len(rows) != len(indexed):
            raise ValueError(f"Journal/answer count mismatch: {len(rows)} vs {len(indexed)}")
        for row in rows:
            key = (row["conversation_id"], int(row["turn_id"]), row["method_key"])
            artifact = indexed.get(key)
            if artifact is None or row["answer"] != artifact["prediction"]:
                raise ValueError(f"Journal/answer mismatch: {key}")
    return rows, sources


def inspect_row(row: dict) -> dict:
    answer = row["answer"]
    if not isinstance(answer, str):
        raise ValueError("Answer must be a string")
    key = (row["conversation_id"], int(row["turn_id"]), row["method_key"])
    if key[2] not in METHODS or key[1] not in (1, 2, 3):
        raise ValueError(f"Unexpected method/turn: {key}")
    finite = row.get("all_last_token_logits_finite")
    checked = row.get("logit_decisions_checked")
    forwards = row.get("model_forward_outputs")
    if finite is True and checked == forwards and isinstance(checked, int) and checked > 0:
        logit_status = "all_checked_finite"
    elif finite is False or row.get("nonfinite_or_missing_logits_at_forwards"):
        logit_status = "nonfinite_or_missing"
    else:
        logit_status = "unknown_no_complete_hook_evidence"
    repetition = repeated_segments(answer)
    # Numeric grids can legitimately contain long runs of digits. Retain this
    # evidence for human review, but do not automatically call it degeneration.
    clear_literal_loop = any(
        item["kind"] == "identical_character_run" or
        (item["kind"] == "repeated_token_phrase" and
         any(char.isalpha() for char in item["unit"]))
        for item in repetition
    )
    flags = []
    if not answer.strip():
        flags.append("empty_output")
    if row.get("runtime_success") is not True:
        flags.append("runtime_failure")
    if row.get("first_token_success") is not True:
        flags.append("first_token_failure")
    if logit_status == "nonfinite_or_missing":
        flags.append("nonfinite_or_missing_logits")
    if clear_literal_loop:
        flags.append("clear_literal_loop")
    review_flags = ["numeric_repetition_needs_review"] if repetition and not clear_literal_loop else []
    return {
        "conversation_id": key[0], "turn_id": key[1], "method_key": key[2],
        "answer_chars": len(answer), "generated_tokens": row.get("generated_tokens"),
        "effective_max_new_tokens": row.get("effective_max_new_tokens"),
        "generation_cap_reached": row.get("generated_tokens") == row.get("effective_max_new_tokens"),
        "nonempty": bool(answer.strip()), "runtime_success": row.get("runtime_success"),
        "first_token_success": row.get("first_token_success"),
        "logit_status": logit_status, "repetition": repetition, "flags": flags,
        "review_flags": review_flags,
        "sample_start": answer[:220], "sample_end": answer[-220:],
    }


def make_report(run_dir: Path) -> tuple[dict, str]:
    rows, sources = load_run(run_dir)
    inspected = sorted((inspect_row(row) for row in rows),
                       key=lambda r: (int(r["conversation_id"].split(":")[1]),
                                      r["turn_id"], r["method_key"]))
    keys = [(r["conversation_id"], r["turn_id"], r["method_key"]) for r in inspected]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate request rows")
    counts = Counter(flag for row in inspected for flag in row["flags"])
    review_counts = Counter(flag for row in inspected for flag in row["review_flags"])
    logit_counts = Counter(row["logit_status"] for row in inspected)
    failing = [row for row in inspected if row["flags"]]
    review = [row for row in inspected if row["review_flags"]]
    summary = {
        "scope": "reference-free output sanity; not answer correctness",
        "source_run_dir": str(run_dir), "source_files": sources,
        "conversation_count": len(set(row["conversation_id"] for row in inspected)),
        "request_count": len(inspected), "flagged_request_count": len(failing),
        "flag_counts": dict(sorted(counts.items())),
        "review_request_count": len(review),
        "review_counts": dict(sorted(review_counts.items())),
        "logit_status_counts": dict(sorted(logit_counts.items())),
        "no_clear_generation_failure_gate": not failing,
        "all_logits_proven_finite": logit_counts == {"all_checked_finite": len(inspected)},
        "rows": inspected,
    }
    lines = ["# ConvBench output sanity", "",
             f"Source: `{run_dir}`", "",
             "This flags strong literal repetition without using references. It does not judge factual correctness. ",
             "Numeric repetition is marked for review because a structured grid may legitimately repeat digits.",
             "Finite logits require a complete per-forward native diagnostic hook; ordinary answer artifacts alone cannot prove them.", "",
             f"- Conversations: {summary['conversation_count']}; requests: {len(inspected)}",
             f"- Flagged requests: {len(failing)}",
             f"- Flag counts: `{json.dumps(summary['flag_counts'], sort_keys=True)}`",
             f"- Numeric repetition review requests: {len(review)}",
             f"- Logit evidence: `{json.dumps(summary['logit_status_counts'], sort_keys=True)}`",
             f"- No clear literal generation failure gate: **{'PASS' if not failing else 'FAIL'}**", "",
             "## Flagged samples", ""]
    for row in failing + review:
        label = f"{row['conversation_id']} T{row['turn_id']} {row['method_key']}"
        lines.extend([f"### {label}", "",
                      f"Flags: `{', '.join(row['flags'] + row['review_flags'])}`; generated tokens: {row['generated_tokens']}; "
                      f"cap reached: {row['generation_cap_reached']}; logits: `{row['logit_status']}`", "",
                      f"Repetition evidence: `{json.dumps(row['repetition'], ensure_ascii=False)}`", "",
                      "Start:", "", "```text", row["sample_start"], "```", "",
                      "End:", "", "```text", row["sample_end"], "```", ""])
    return summary, "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Use a new output directory: {args.output_dir}")
    summary, markdown = make_report(args.run_dir)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "sanity.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "README.md").write_text(markdown + "\n")
    print(json.dumps({key: summary[key] for key in (
        "conversation_count", "request_count", "flagged_request_count", "flag_counts",
        "review_request_count", "review_counts", "logit_status_counts", "no_clear_generation_failure_gate",
        "all_logits_proven_finite")}, indent=2))


if __name__ == "__main__":
    main()

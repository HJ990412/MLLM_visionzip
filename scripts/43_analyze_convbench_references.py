#!/usr/bin/env python3
"""Measure ConvBench reference answer lengths with the answerer's tokenizer.

Run with the project's inference environment:
    python scripts/43_analyze_convbench_references.py
    python scripts/43_analyze_convbench_references.py --verify

The output is deterministic and includes every per-conversation length, so the
reported quantiles and generation-cap coverage can be independently recomputed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from huggingface_hub import try_to_load_from_cache
from transformers import AutoTokenizer
import transformers


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"
SOURCE_RUNNER = ROOT / "data/convbench_source/VLMEvalKit/vlmeval/vlm/llava.py"
REFERENCE_FIELDS = ("reference_A1", "reference_A2", "reference_A3")
CAPS = (16, 128, 256, 512, 1024, 1536)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[int], percent: int) -> float:
    """NumPy's default linear percentile, implemented without another dependency."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize empty lengths")
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 4)


def summary(values: list[int]) -> dict:
    return {
        "count": len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
        "above_cap": {str(cap): sum(length > cap for length in values) for cap in CAPS},
    }


def build(index_path: Path, provenance_path: Path) -> dict:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    conversations = index["conversations"]
    if len(conversations) != 577:
        raise ValueError(f"expected 577 image-available conversations, got {len(conversations)}")
    if index["official_commit"] != provenance["official_commit"]:
        raise ValueError("index and provenance official commits differ")
    workbook = ROOT / provenance["source_workbook"]
    if sha256(workbook) != provenance["source_workbook_sha256"]:
        raise ValueError("official workbook hash differs from dataset provenance")
    if not SOURCE_RUNNER.is_file():
        raise FileNotFoundError(SOURCE_RUNNER)
    runner_source = SOURCE_RUNNER.read_text(encoding="utf-8")
    cap_match = re.search(r"kwargs_default\s*=\s*dict\([^\n]*max_new_tokens\s*=\s*(\d+)", runner_source)
    if cap_match is None:
        raise ValueError("official LLaVA generation default not found")

    tokenizer_file = try_to_load_from_cache(MODEL_ID, "tokenizer.json")
    if not isinstance(tokenizer_file, str) or not Path(tokenizer_file).is_file():
        raise FileNotFoundError(f"local tokenizer.json for {MODEL_ID} is unavailable")
    tokenizer_path = Path(tokenizer_file)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    tokenizer.model_max_length = 10**9  # length measurement only; avoid warnings for long references

    ids: set[str] = set()
    rows: list[dict] = []
    per_turn: list[list[int]] = [[], [], []]
    for item in conversations:
        conversation_id = item["conversation_id"]
        if conversation_id in ids:
            raise ValueError(f"duplicate conversation ID {conversation_id}")
        ids.add(conversation_id)
        if not (ROOT / item["image_path"]).is_file():
            raise FileNotFoundError(item["image_path"])
        lengths = []
        for field in REFERENCE_FIELDS:
            answer = item[field]
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"empty reference {conversation_id} {field}")
            lengths.append(len(tokenizer.encode(answer, add_special_tokens=False)))
        for turn, length in enumerate(lengths):
            per_turn[turn].append(length)
        rows.append({
            "conversation_id": conversation_id,
            "source_row_index": item["source_row_index"],
            "reference_A1_tokens": lengths[0],
            "reference_A2_tokens": lengths[1],
            "reference_A3_tokens": lengths[2],
        })

    if len(rows) != 577 or any(len(values) != 577 for values in per_turn):
        raise AssertionError("incomplete reference coverage")
    all_lengths = [length for values in per_turn for length in values]
    return {
        "schema_version": "convbench-reference-lengths-v1",
        "dataset": {
            "index_sha256": sha256(index_path),
            "provenance_sha256": sha256(provenance_path),
            "source_workbook_sha256": provenance["source_workbook_sha256"],
            "official_commit": provenance["official_commit"],
            "selection_policy": provenance["selection_policy"],
            "conversation_count": len(rows),
        },
        "tokenizer": {
            "model_id": MODEL_ID,
            "revision": tokenizer_path.parent.name,
            "tokenizer_json_sha256": sha256(tokenizer_path),
            "transformers_version": transformers.__version__,
            "add_special_tokens": False,
            "length_unit": "token IDs per reference answer",
        },
        "generation_cap_evidence": {
            "official_convbench_llava_v1_5_default_max_new_tokens": int(cap_match.group(1)),
            "official_runner_path": str(SOURCE_RUNNER.relative_to(ROOT)),
            "official_runner_sha256": sha256(SOURCE_RUNNER),
            "official_runner_note": "This default belongs to the official LLaVA-1.5 wrapper, not the current LLaVA-NeXT answerer.",
            "metacompress_cap_status": "Not specified in the paper; the public repository has no implementation as inspected on 2026-09-17.",
            "metacompress_repository": "https://github.com/MArSha1147/MetaCompress",
        },
        "summary": {
            "turn1": summary(per_turn[0]),
            "turn2": summary(per_turn[1]),
            "turn3": summary(per_turn[2]),
            "all_turns": summary(all_lengths),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "data/convbench/index.json")
    parser.add_argument("--provenance", type=Path, default=ROOT / "data/convbench/provenance.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/convbench/reference_token_lengths.json")
    parser.add_argument("--verify", action="store_true", help="recompute and compare with existing output")
    args = parser.parse_args()
    result = build(args.index, args.provenance)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.verify:
        if args.output.read_text(encoding="utf-8") != encoded:
            raise SystemExit("reference length artifact differs from recomputed result")
        print(f"Verified {args.output}: {len(result['rows'])} conversations")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(args.output)
        print(f"Wrote {args.output}: {len(result['rows'])} conversations")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CPU-only LLaVA-NeXT ConvBench context preflight with reference histories."""
from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "convbench_answer_runner", ROOT / "scripts/41_eval_convbench_answers.py")
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    snapshot = (Path("/home/dblab/.cache/huggingface/hub")
                / "models--llava-hf--llava-v1.6-vicuna-7b-hf"
                / "snapshots/c916e6cdcd760b4cecd1dd4907f84ac649f93b23")
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    dialogs = json.loads((ROOT / "data/convbench/index.json").read_text())[
        "conversations"]
    rows = []
    for n, dialog in enumerate(dialogs, 1):
        questions = [dialog[f"Q{i}"] for i in (1, 2, 3)]
        references = [dialog[f"reference_A{i}"] for i in (1, 2, 3)]
        prompts = [runner.render_prompt(questions[:i], references[:i - 1])
                   for i in (1, 2, 3)]
        with Image.open(ROOT / dialog["image_path"]) as im:
            image = im.convert("RGB")
        combined = processor(images=image, text=prompts[0], return_tensors="pt")
        normal_t1 = int(combined["input_ids"].shape[1])
        raw_t1 = len(processor.tokenizer(prompts[0]).input_ids)
        image_expansion = normal_t1 - raw_t1
        estimated = [normal_t1] + [
            len(processor.tokenizer(prompt).input_ids) + image_expansion
            for prompt in prompts[1:]
        ]
        rows.append({"conversation_id": dialog["conversation_id"],
                     "source_row_index": dialog["source_row_index"],
                     "image_expansion_extra_tokens": image_expansion,
                     "reference_history_input_tokens": estimated,
                     "input_plus_1024_exceeds_4096":
                         [value + 1024 > 4096 for value in estimated]})
        if n % 100 == 0:
            print(f"processed {n}/{len(dialogs)}", flush=True)
    summary = {
        "schema_version": "convbench-context-reference-history-preflight-v1",
        "n_conversations": len(rows),
        "model_context_tokens": 4096,
        "nominal_generation_cap": 1024,
        "history_type": "official reference answers for preflight only; main evaluation uses generated answers",
        "method": "T1 exact combined processor; T2/T3 raw text token count plus same-image AnyRes expansion delta",
        "counts_input_plus_cap_exceeds_context_by_turn": [
            sum(row["input_plus_1024_exceeds_4096"][i] for row in rows)
            for i in range(3)],
        "max_reference_history_input_tokens_by_turn": [
            max(row["reference_history_input_tokens"][i] for row in rows)
            for i in range(3)],
        "median_reference_history_input_tokens_by_turn": [
            statistics.median(row["reference_history_input_tokens"][i] for row in rows)
            for i in range(3)],
        "rows": rows,
    }
    out = ROOT / "data/convbench/context_preflight.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                              sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()

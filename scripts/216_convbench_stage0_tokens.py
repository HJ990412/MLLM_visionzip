#!/usr/bin/env python3
"""Stage 0 of the ConvBench main experiment: does the token-sequence defect fixed in the ConvBench
baseline alignment also exist on the MT-GQA / MT-VQA prompt paths?

Read-only.  No model, no GPU, nothing re-run, no code changed.

For each (model, dataset) the prompts are the ones a finished run actually sent: 20 prompts for each of
T1, T2, T3, read from the committed request rows (``prompt``, ``suffix_input_ids``).

  existing path   the HF fast tokenizer of the checkpoint applied to the whole prompt string
                  (``runner.processor.tokenizer(prompt)``; the pixel requests go through the processor,
                  which calls the same tokenizer and only expands ``<image>``).  The part after ``<image>``
                  is checked against the ``suffix_input_ids`` the run recorded, so the comparison is made on
                  the token ids the run really used.
  original LLaVA  ``tokenizer_image_token`` with the slow legacy tokenizer: the text around ``<image>`` is
                  tokenised chunk by chunk, one BOS, the image at -200
                  (``official_token_ids`` of scripts/213_convbench_recomp_generate.py, imported).

Output: <out>/stage0.json and <out>/stage0.md.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NEXT_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"
L15_SNAPSHOT_GLOB = "models--llava-hf--llava-1.5-7b-hf/snapshots/*"
SOURCES = [
    # model, dataset, run with committed rows, phase
    ("LLaVA-NeXT", "MT-GQA", "runs/mt_gqa_ours_order_20261120T052725Z", "main"),
    ("LLaVA-NeXT", "MT-VQA", "runs/llava_next_mtvqa_main_20261121T063602Z", "main"),
    ("LLaVA-1.5", "MT-GQA", "runs/llava15_mt_gqa_main_20261120T165001Z", "main"),
    ("LLaVA-1.5", "MT-VQA", "runs/llava15_mtvqa_impress_code_20261118T053221Z", "main"),
]
PER_TURN = 20
PREFERRED_METHODS = ("recompute", "fullload_async")


def load_213():
    spec = importlib.util.spec_from_file_location("_stage0_213", ROOT / "scripts/213_convbench_recomp_generate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_rows(folder: Path) -> list[dict]:
    for name, opener in (("raw.jsonl", open), ("raw.jsonl.gz", gzip.open)):
        path = folder / name
        if path.exists():
            with opener(path, "rt", encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]
    return []


def sample_prompts(run: Path, phase: str) -> tuple[list[dict], dict]:
    """20 images spread evenly over the committed images of the phase; from each, the first dialogue's
    T1, T2, T3 rows of one method (ReComp when the phase has it, otherwise FullLoad)."""
    images = sorted(p.parent for p in (run / phase / "images").glob("*/COMMITTED.json"))
    stride = max(1, len(images) // PER_TURN)
    picked = []
    index = 0
    while len(picked) < PER_TURN and index < len(images):
        commit = json.loads((images[index] / "COMMITTED.json").read_text())
        rows = read_rows(images[index] / commit["attempt_id"])
        methods = {r["method_id"] for r in rows}
        method = next((m for m in PREFERRED_METHODS if m in methods), None)
        if method is not None:
            mine = [r for r in rows if r["method_id"] == method]
            dialog = mine[0]["dialog_id"]
            turns = {r["turn_id"]: r for r in mine if r["dialog_id"] == dialog}
            if set(turns) == {1, 2, 3}:
                picked.append({"image": images[index].name, "dialog_id": dialog, "method_id": method, "turns": turns})
        index += stride
    return picked, {"committed_images": len(images), "stride": stride, "sampled_images": len(picked)}


def describe(tokenizer, ids: list[int]) -> list[str]:
    return ["<image>" if i == -200 else tokenizer.convert_ids_to_tokens(i) for i in ids]


def first_difference(a: list[int], b: list[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    import transformers

    G = load_213()
    hub = Path.home() / ".cache/huggingface/hub"
    l15 = sorted(hub.glob(L15_SNAPSHOT_GLOB))
    assert len(l15) == 1, l15
    sources = {"LLaVA-NeXT": NEXT_ID, "LLaVA-1.5": str(l15[0])}
    tokenizers = {}
    for name, source in sources.items():
        fast = AutoTokenizer.from_pretrained(source)          # what the processor of the runner holds
        slow = AutoTokenizer.from_pretrained(source, use_fast=False, legacy=True)
        assert fast.is_fast and not slow.is_fast
        tokenizers[name] = (fast, slow, int(fast.convert_tokens_to_ids("<image>")))

    out = {"transformers": transformers.__version__, "per_turn": PER_TURN, "cells": []}
    for model, dataset, run, phase in SOURCES:
        fast, slow, image_id = tokenizers[model]
        picked, sampling = sample_prompts(ROOT / run, phase)
        cell = {"model": model, "dataset": dataset, "run": run, "phase": phase, "sampling": sampling,
                "method_of_rows": sorted({p["method_id"] for p in picked}), "turns": {}, "examples": []}
        patterns = Counter()
        for turn in (1, 2, 3):
            compared = mismatched = recorded_suffix_equal = eos_in_prompt = 0
            for p in picked:
                row = p["turns"][turn]
                prompt = row["prompt"]
                existing = fast(prompt).input_ids
                position = existing.index(image_id)
                assert existing.count(image_id) == 1
                recorded_suffix_equal += existing[position + 1:] == row["suffix_input_ids"]
                existing_c = G.collapse_image(existing, image_id)
                official = G.official_token_ids(slow, prompt)
                eos_in_prompt += "</s>" in prompt
                compared += 1
                if existing_c != official:
                    mismatched += 1
                    at = first_difference(existing_c, official)
                    # every differing position after aligning the two sequences
                    import difflib
                    ops = [op for op in difflib.SequenceMatcher(a=existing_c, b=official, autojunk=False).get_opcodes()
                           if op[0] != "equal"]
                    detail = [{"op": tag, "existing_position": i1, "existing_tokens": describe(fast, existing_c[i1:i2]),
                               "original_position": j1, "original_tokens": describe(fast, official[j1:j2])}
                              for tag, i1, i2, j1, j2 in ops]
                    for d in detail:
                        before = describe(fast, existing_c[max(0, d["existing_position"] - 1):d["existing_position"]])
                        patterns[json.dumps({"after": before, "existing": d["existing_tokens"], "original": d["original_tokens"]},
                                            ensure_ascii=False)] += 1
                    if len(cell["examples"]) < 3:
                        cell["examples"].append({"turn": turn, "image": p["image"], "dialog_id": p["dialog_id"],
                                                 "existing_len": len(existing_c), "original_len": len(official),
                                                 "first_difference_position": at, "image_position": position,
                                                 "differences": detail,
                                                 "existing_head": describe(fast, existing_c[:position + 6]),
                                                 "original_head": describe(fast, official[:position + 7])})
            cell["turns"][f"T{turn}"] = {"compared": compared, "mismatched": mismatched,
                                         "existing_suffix_equals_recorded_suffix": recorded_suffix_equal,
                                         "prompts_containing_eos": eos_in_prompt}
        cell["difference_patterns"] = [dict(json.loads(k), count=v) for k, v in patterns.most_common()]
        out["cells"].append(cell)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stage0.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["| 모델 | 데이터셋 | T1 불일치 | T2 불일치 | T3 불일치 | 다른 위치·토큰 |", "|---|---|---:|---:|---:|---|"]
    for cell in out["cells"]:
        t = cell["turns"]
        pats = "; ".join(f"`{p['after'][0] if p['after'] else ''}` 뒤: 기존 {p['existing']} ↔ 원본 {p['original']} ({p['count']}건)"
                         for p in cell["difference_patterns"]) or "없음"
        lines.append(f"| {cell['model']} | {cell['dataset']} | " + " | ".join(
            f"{t[k]['mismatched']}/{t[k]['compared']}" for k in ("T1", "T2", "T3")) + f" | {pats} |")
    (args.out / "stage0.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    for cell in out["cells"]:
        print(cell["model"], cell["dataset"], cell["sampling"], {k: v["existing_suffix_equals_recorded_suffix"] for k, v in cell["turns"].items()},
              {k: v["prompts_containing_eos"] for k, v in cell["turns"].items()})


if __name__ == "__main__":
    main()

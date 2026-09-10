#!/usr/bin/env python3
"""Validate append-only multi-image caching against full LLaVA-NeXT prefill.

This is deliberately a transformer-cache correctness gate, not an SSD method.
For the two dialogs in the fixed MMDU feasibility subset, it compares the first
three user turns under the same Vicuna/gold-history prompt in two ways:

  * full: recompute the complete prompt and all active images every turn;
  * append: retain one DynamicCache and submit only the exact token suffix and
    newly introduced images at later turns.

Images are first resized to 336 x 336 with LANCZOS, matching the preprocessing
performed by MMDU's official LLaVA-NeXT generation script.  No independently
built image-prefix store is read or concatenated here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmimpress.config import MODEL_ID, PROJECT_ROOT
from mmimpress.model import LlavaRunner
from mmimpress.multiturn import (load_canonical, mmdu_prompt,
                                 resolve_image_path, sha256_file)


SCHEMA_VERSION = "mmdu-dynamic-cache-validation-v1"
N_DIALOGS = 2
N_TURNS = 3
HARD_CONTEXT_LIMIT = 4096
RESIZE_HW = (336, 336)
OFFICIAL_MMDU_REPO = "https://github.com/Liuziyu77/MMDU"
OFFICIAL_LLAVA_SCRIPT = (
    "https://github.com/Liuziyu77/MMDU/blob/main/"
    "model_generation/LLaVa_next_gen_ans.py"
)
OUTPUT_NAMES = (
    "config.json", "raw.jsonl", "per_turn.csv", "per_dialog.csv",
    "summary.csv", "validation.json", "README.md",
)


def _json_dump(path: Path, value) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_ids(ids: list[int]) -> str:
    arr = np.asarray(ids, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _cached_hf_artifact(model_id: str, filename: str) -> dict | None:
    """Resolve and hash the exact cached processor artifact used by HF."""
    from transformers.utils.hub import cached_file

    path = cached_file(
        model_id, filename, local_files_only=True,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
    )
    if path is None:
        return None
    path = Path(path)
    parts = path.parts
    revision = None
    if "snapshots" in parts:
        position = parts.index("snapshots")
        if position + 1 < len(parts):
            revision = parts[position + 1]
    return {
        "filename": filename,
        "cached_path": str(path),
        "sha256": sha256_file(path),
        "snapshot_revision": revision,
    }


def _model_context_limit(runner: LlavaRunner) -> int:
    configured = int(runner.cfg.text_config.max_position_embeddings)
    if configured < HARD_CONTEXT_LIMIT:
        return configured
    return HARD_CONTEXT_LIMIT


def _visual_runs(ids: list[int], image_token_id: int) -> list[tuple[int, int]]:
    """Return (start, length) for each contiguous expanded image-token run."""
    positions = [i for i, token_id in enumerate(ids)
                 if token_id == image_token_id]
    if not positions:
        return []
    out: list[tuple[int, int]] = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position != previous + 1:
            out.append((start, previous - start + 1))
            start = position
        previous = position
    out.append((start, previous - start + 1))
    return out


def _named_spans(ids: list[int], image_ids: list[str], image_token_id: int,
                 offset: int = 0) -> list[dict]:
    runs = _visual_runs(ids, image_token_id)
    if len(runs) != len(image_ids):
        raise RuntimeError(
            "visual span/image mismatch: "
            f"found {len(runs)} token runs for {len(image_ids)} images "
            f"({image_ids})"
        )
    return [{
        "image_id": image_id,
        "start": int(start + offset),
        "length": int(length),
        "end_exclusive": int(start + offset + length),
    } for image_id, (start, length) in zip(image_ids, runs)]


def _cache_length(cache) -> int:
    return int(cache.get_seq_length())


def _visual_kv_geometry(cache, spans: list[dict]) -> dict:
    """Derive visual-KV bytes from the tensors actually produced this turn."""
    layers = list(cache.layers)
    if not layers:
        raise RuntimeError("empty DynamicCache")
    per_token_bytes = 0
    layer_shapes = []
    dtypes = set()
    for layer_index, layer in enumerate(layers):
        key, value = layer.keys, layer.values
        if key is None or value is None or key.ndim != 4 or value.ndim != 4:
            raise RuntimeError(f"invalid cache tensors at layer {layer_index}")
        if key.shape != value.shape or int(key.shape[0]) != 1:
            raise RuntimeError((key.shape, value.shape))
        if int(key.shape[2]) != _cache_length(cache):
            raise RuntimeError((key.shape, _cache_length(cache)))
        dtypes.update((str(key.dtype), str(value.dtype)))
        per_token_bytes += int(
            (key[0, :, 0, :].numel() * key.element_size()) +
            (value[0, :, 0, :].numel() * value.element_size()))
        layer_shapes.append({
            "layer": layer_index,
            "key_shape": list(key.shape),
            "value_shape": list(value.shape),
            "key_dtype": str(key.dtype),
            "value_dtype": str(value.dtype),
            "key_element_size": key.element_size(),
            "value_element_size": value.element_size(),
        })
    visual_tokens = int(sum(int(span["length"]) for span in spans))
    per_image = {
        span["image_id"]: int(span["length"]) * per_token_bytes
        for span in spans
    }
    return {
        "num_layers": len(layers),
        "cache_dtypes": sorted(dtypes),
        "representative_layer": layer_shapes[0],
        "visual_kv_bytes_per_token": per_token_bytes,
        "active_visual_tokens": visual_tokens,
        "full_visual_kv_bytes": visual_tokens * per_token_bytes,
        "image_visual_kv_bytes": per_image,
        "derivation": (
            "sum over actual DynamicCache layers of K/V heads*head_dim*element_size"
        ),
    }


@torch.inference_mode()
def _prefill(runner: LlavaRunner, input_ids: torch.Tensor, cache,
             start_position: int, total_length: int,
             pixel_values: torch.Tensor | None = None,
             image_sizes: torch.Tensor | None = None):
    """Run one absolute-position segment and return cache plus last logits."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise RuntimeError(f"expected one input sequence, got {input_ids.shape}")
    segment_length = int(input_ids.shape[1])
    if segment_length <= 0:
        raise RuntimeError("append segment is empty")
    if start_position + segment_length != total_length:
        raise RuntimeError(
            f"position mismatch: {start_position}+{segment_length}!={total_length}"
        )
    device = runner.model.device
    positions = torch.arange(start_position, total_length, device=device)
    kwargs = {
        "input_ids": input_ids.to(device),
        "attention_mask": torch.ones(
            1, total_length, dtype=torch.long, device=device),
        "position_ids": positions.unsqueeze(0),
        "cache_position": positions,
        "past_key_values": cache,
        "use_cache": True,
        "return_dict": True,
        "logits_to_keep": 1,
    }
    if (pixel_values is None) != (image_sizes is None):
        raise RuntimeError("pixel_values and image_sizes must be supplied together")
    if pixel_values is not None:
        kwargs["pixel_values"] = pixel_values.to(device)
        kwargs["image_sizes"] = image_sizes.to(device)
    output = runner.model(**kwargs)
    logits = output.logits[0, -1].detach().float().cpu()
    return output.past_key_values, logits


def _eos_ids(tokenizer) -> set[int]:
    value = tokenizer.eos_token_id
    if value is None:
        return set()
    if isinstance(value, (tuple, list, set)):
        return {int(x) for x in value}
    return {int(value)}


@torch.inference_mode()
def _greedy_from_first_logits(runner: LlavaRunner, cache,
                              first_logits: torch.Tensor,
                              prompt_length: int,
                              max_new_tokens: int) -> tuple[list[int], str]:
    """Generate exactly as a greedy LM continuation from prefill logits.

    The first output decision is already available from the prompt prefill.
    Subsequent calls feed the previous decision, so the returned cache contains
    prompt_length + generated_count - 1 rows.  The caller may crop it back to
    prompt_length before appending teacher-forced gold history.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    tokenizer = runner.processor.tokenizer
    eos = _eos_ids(tokenizer)
    tokens = [int(first_logits.argmax().item())]
    current_position = int(prompt_length)
    device = runner.model.device
    while tokens[-1] not in eos and len(tokens) < max_new_tokens:
        position = torch.tensor([current_position], device=device)
        output = runner.model(
            input_ids=torch.tensor([[tokens[-1]]], device=device),
            attention_mask=torch.ones(
                1, current_position + 1, dtype=torch.long, device=device),
            position_ids=position.unsqueeze(0),
            cache_position=position,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        cache = output.past_key_values
        tokens.append(int(output.logits[0, -1].argmax().item()))
        current_position += 1
    response = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    return tokens, response


def _resize_dialog_images(dialog: dict) -> tuple[dict[str, Image.Image], dict]:
    images: dict[str, Image.Image] = {}
    metadata: dict[str, dict] = {}
    for item in dialog["images"]:
        image_id = item["image_id"]
        path = resolve_image_path(item["image_path"]).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        with Image.open(path) as source:
            source.load()
            original_size = [int(source.width), int(source.height)]
            resized = source.convert("RGB").resize(
                RESIZE_HW, resample=Image.Resampling.LANCZOS)
        images[image_id] = resized
        metadata[image_id] = {
            "image_id": image_id,
            "path": str(path),
            "sha256": sha256_file(path),
            "source_index": int(item["source_index"]),
            "original_size_wh": original_size,
            "resized_size_wh": [RESIZE_HW[0], RESIZE_HW[1]],
            "resample": "PIL.Image.Resampling.LANCZOS",
        }
    if list(images) != list(dialog["image_ids"]):
        raise RuntimeError("dialog image order differs from canonical image_ids")
    return images, metadata


def _new_image_tensors(encoded, active_ids: list[str],
                       new_ids: list[str]):
    """Take the ordered tail belonging to newly introduced images."""
    if not new_ids:
        return None, None
    if active_ids[-len(new_ids):] != new_ids:
        raise RuntimeError(
            f"new images are not the ordered active-image tail: {new_ids}"
        )
    if "pixel_values" not in encoded or "image_sizes" not in encoded:
        raise RuntimeError("processor omitted multi-image tensors")
    pixel_values, image_sizes = encoded["pixel_values"], encoded["image_sizes"]
    if int(pixel_values.shape[0]) != len(active_ids):
        raise RuntimeError(
            f"pixel batch {pixel_values.shape} does not match {len(active_ids)} images"
        )
    if int(image_sizes.shape[0]) != len(active_ids):
        raise RuntimeError(
            f"image_sizes {image_sizes.shape} does not match active images"
        )
    start = len(active_ids) - len(new_ids)
    return pixel_values[start:], image_sizes[start:]


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_dialogs(turn_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in turn_rows:
        grouped[row["dialog_id"]].append(row)
    output = []
    for dialog_id, rows in grouped.items():
        output.append({
            "dialog_id": dialog_id,
            "turns": len(rows),
            "max_prompt_tokens": max(r["prompt_tokens"] for r in rows),
            "all_prefix_exact": all(r["prefix_exact"] for r in rows),
            "all_token_sequences_exact": all(
                r["token_sequence_exact"] for r in rows),
            "all_visual_spans_exact": all(r["visual_spans_exact"] for r in rows),
            "all_cache_lengths_exact": all(r["cache_length_exact"] for r in rows),
            "all_first_tokens_exact": all(r["first_token_exact"] for r in rows),
            "all_generated_token_ids_exact": all(
                r["generated_token_ids_exact"] for r in rows),
            "all_generated_responses_exact": all(
                r["generated_response_exact"] for r in rows),
            "all_logits_within_tolerance": all(
                r["logits_within_tolerance"] for r in rows),
            "max_first_logit_abs_diff": max(
                r["first_logit_max_abs_diff"] for r in rows),
            "max_first_logit_mean_abs_diff": max(
                r["first_logit_mean_abs_diff"] for r in rows),
        })
    return output


def _build_readme(config: dict, turn_rows: list[dict], dialog_rows: list[dict],
                  validation: dict) -> str:
    verdict = "PASS" if validation["append_only_dynamic_cache_gate_passed"] else "FAIL"
    lines = [
        "# MMDU append-only DynamicCache validation",
        "",
        f"Result: **{verdict}** for the append-only transformer-cache gate.",
        "",
        "## Scope",
        "",
        (f"This run compares full recomputation with one persistent DynamicCache "
         f"for the first {N_TURNS} user turns of each of {N_DIALOGS} fixed MMDU "
         "dialogs. Both paths use the same Vicuna `USER:/ASSISTANT:` canonical "
         "prompt and gold teacher-forced prior answers."),
        "",
        ("Every source image is converted to RGB and resized to 336×336 with "
         "Pillow LANCZOS before the LLaVA-NeXT processor. This resize follows "
         "the official MMDU LLaVA-NeXT script."),
        "",
        ("This is not an official answer-quality reproduction: the official "
         "generation script wraps turns with `[INST]...[/INST]` and carries "
         "generated history, whereas this fixed Vicuna checkpoint uses the "
         "repository's model-native canonical wrapper and gold history. The "
         "same local prompt is used on both sides, so the result isolates "
         "transformer cache equivalence."),
        "",
        f"- MMDU repository: {OFFICIAL_MMDU_REPO}",
        f"- Official LLaVA-NeXT script: {OFFICIAL_LLAVA_SCRIPT}",
        f"- Index: `{config['index']}` (`{config['index_sha256']}`)",
        f"- Model: `{config['model']}`",
        f"- Quantization: `{config['quantization']}`",
        f"- Attention: `{config['attention']}`",
        f"- Decode: greedy, at most {config['max_new_tokens']} new tokens",
        (f"- First-logit tolerance: max absolute <= "
         f"{config['first_logit_tolerance']['max_abs']} and mean absolute <= "
         f"{config['first_logit_tolerance']['mean_abs']}"),
        (f"- Context rule: prompt tokens + reserved decode tokens must be <= "
         f"{config['effective_context_limit']}"),
        "",
        "## Per-turn comparison",
        "",
        "| dialog | turn | prompt tokens | new images | max abs logit diff | mean abs logit diff | first token | generated IDs |",
        "|---|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in turn_rows:
        lines.append(
            f"| {row['dialog_id']} | {row['turn_id']} | {row['prompt_tokens']} | "
            f"{row['new_image_count']} | {row['first_logit_max_abs_diff']:.8g} | "
            f"{row['first_logit_mean_abs_diff']:.8g} | "
            f"{'yes' if row['first_token_exact'] else 'no'} | "
            f"{'yes' if row['generated_token_ids_exact'] else 'no'} |"
        )
    lines.extend([
        "",
        "## Per-dialog gates",
        "",
        "| dialog | turns | prefix | tokens | spans | cache length | first token | response |",
        "|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for row in dialog_rows:
        def yes(field: str) -> str:
            return "yes" if row[field] else "no"
        lines.append(
            f"| {row['dialog_id']} | {row['turns']} | {yes('all_prefix_exact')} | "
            f"{yes('all_token_sequences_exact')} | {yes('all_visual_spans_exact')} | "
            f"{yes('all_cache_lengths_exact')} | {yes('all_first_tokens_exact')} | "
            f"{yes('all_generated_responses_exact')} |"
        )
    lines.extend([
        "",
        "## Important boundary",
        "",
        ("`static_diverse_mmdu_gate_passed` is deliberately **false**. This run "
         "does not open existing SSD stores, does not concatenate independently "
         "computed image-prefix KV tensors, and does not validate Static+Diverse "
         "serving for MMDU. It only gates append-only `DynamicCache` correctness."),
        "",
        "Detailed token IDs, token strings, visual spans, cache lengths, logits, "
        "and responses are in `raw.jsonl`; flattened results are in the CSV files.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "conda activate mllm_ft",
        ("python scripts/17_validate_mmdu_cache.py "
         f"--index {config['index']} --run-dir {config['run_dir']} "
         f"--max-new-tokens {config['max_new_tokens']} "
         f"--logit-max-atol {config['first_logit_tolerance']['max_abs']} "
         f"--logit-mean-atol {config['first_logit_tolerance']['mean_abs']} "
         f"--attention {config['attention']} "
         + ("--load-4bit" if config["load_4bit"] else "--no-load-4bit")),
        "```",
        "",
    ])
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index", type=Path,
        default=PROJECT_ROOT /
        "data/mmdu/subsets/correctness_progressive_seed1234/index.json")
    parser.add_argument(
        "--run-dir", type=Path,
        default=PROJECT_ROOT /
        "runs/mmdu_multiturn/correctness_progressive_seed1234")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--load-4bit", action=argparse.BooleanOptionalAction, default=True,
        help="load with bitsandbytes NF4 double quantization (default: true)")
    parser.add_argument(
        "--attention", choices=("eager", "sdpa"), default="eager",
        help="attention implementation; eager is the validation default")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--logit-max-atol", type=float, default=0.125,
        help="maximum allowed max absolute first-logit difference")
    parser.add_argument(
        "--logit-mean-atol", type=float, default=0.01,
        help="maximum allowed mean absolute first-logit difference")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="overwrite only this script's seven named result artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be >= 1")
    if (not np.isfinite(args.logit_max_atol) or args.logit_max_atol < 0
            or not np.isfinite(args.logit_mean_atol)
            or args.logit_mean_atol < 0):
        raise ValueError("logit tolerances must be finite and >= 0")
    index_path = args.index.resolve()
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    dialogs = load_canonical(index_path)
    if len(dialogs) != N_DIALOGS:
        raise RuntimeError(
            f"correctness index must contain exactly {N_DIALOGS} dialogs; "
            f"found {len(dialogs)}"
        )
    if any(d.get("dataset") != "mmdu" for d in dialogs):
        raise RuntimeError("correctness index contains a non-MMDU dialog")
    if any(len(d["turns"]) < N_TURNS for d in dialogs):
        raise RuntimeError(f"every dialog needs at least {N_TURNS} turns")

    run_dir = args.run_dir.resolve()
    existing = [run_dir / name for name in OUTPUT_NAMES
                if (run_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "result artifacts already exist; choose another --run-dir or pass "
            f"--overwrite: {[str(x) for x in existing]}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("this LLaVA-NeXT validation requires CUDA")

    started = time.time()
    runner = LlavaRunner(
        model_id=args.model_id,
        load_4bit=args.load_4bit,
        attn=args.attention,
    ).load()
    tokenizer = runner.processor.tokenizer
    effective_limit = _model_context_limit(runner)
    vm = psutil.virtual_memory()
    process = psutil.Process()
    config = {
        "schema_version": SCHEMA_VERSION,
        "created_unix": started,
        "index": str(index_path),
        "index_sha256": sha256_file(index_path),
        "run_dir": str(run_dir),
        "dataset": "mmdu",
        "dialog_ids": [d["dialog_id"] for d in dialogs],
        "n_dialogs": N_DIALOGS,
        "turn_policy": "first_3_user_turns_each_dialog",
        "turns_per_dialog": N_TURNS,
        "n_turns": N_DIALOGS * N_TURNS,
        "history_policy": "gold_teacher_forced",
        "prompt_policy": "mmimpress.multiturn.mmdu_prompt_vicuna_append_only",
        "prompt_wrapper": "USER: ... ASSISTANT: ...</s>",
        "image_marker_mapping": "<ImageHere> -> <image>\\n",
        "image_order_policy": (
            "canonical active_image_ids order; new_image_ids must be the ordered "
            "tail and only that tail is sent on append"
        ),
        "model": runner.model_id,
        "model_config_name_or_path": runner.cfg._name_or_path,
        "model_checkpoint_revision": getattr(runner.cfg, "_commit_hash", None),
        "processor_class": type(runner.processor).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_name_or_path": tokenizer.name_or_path,
        "image_processor_class": type(runner.processor.image_processor).__name__,
        "processor_artifacts": {
            filename: _cached_hf_artifact(runner.model_id, filename)
            for filename in (
                "preprocessor_config.json", "processor_config.json",
                "tokenizer_config.json")
        },
        "load_4bit": bool(runner.load_4bit),
        "quantization": (
            "bitsandbytes NF4 double-quant, bfloat16 compute"
            if runner.load_4bit else "none (bfloat16 model load)"),
        "attention": runner.attn,
        "decoding": "greedy_argmax",
        "max_new_tokens": args.max_new_tokens,
        "first_logit_tolerance": {
            "max_abs": args.logit_max_atol,
            "mean_abs": args.logit_mean_atol,
            "declared_before_run": True,
        },
        "seed": args.seed,
        "resize_before_processor": {
            "size_wh": [336, 336],
            "color_mode": "RGB",
            "resample": "PIL.Image.Resampling.LANCZOS",
            "source": OFFICIAL_LLAVA_SCRIPT,
        },
        "official_mmdu_repository": OFFICIAL_MMDU_REPO,
        "official_llava_next_generation_script": OFFICIAL_LLAVA_SCRIPT,
        "official_generation_difference": (
            "Official script uses [INST] wrappers and generated history. This "
            "cache-equivalence test uses the fixed Vicuna checkpoint's local "
            "USER/ASSISTANT wrapper and gold teacher-forced history in both paths."
        ),
        "hard_context_limit": HARD_CONTEXT_LIMIT,
        "model_max_position_embeddings": int(
            runner.cfg.text_config.max_position_embeddings),
        "effective_context_limit": effective_limit,
        "context_gate": "prompt_tokens + max_new_tokens <= effective_context_limit",
        "full_path": "fresh DynamicCache; complete prompt plus all active images",
        "append_path": (
            "one DynamicCache per dialog; exact token suffix plus ordered newly "
            "introduced images; crop generated continuation before next gold turn"
        ),
        "existing_image_prefix_ssd_stores_used": False,
        "independent_image_prefix_kv_concatenation_used": False,
        "static_diverse_mmdu_gate_passed": False,
        "static_diverse_gate_reason": (
            "out of scope: this validates native append-only DynamicCache only"
        ),
        "official_mmdu_generation_protocol_gate_passed": False,
        "official_protocol_gate_reason": (
            "different model-native prompt wrapper and gold rather than generated history"
        ),
        "transformers_cache_class": "transformers.DynamicCache",
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "machine": {
            "gpu_total_memory_bytes": int(
                torch.cuda.get_device_properties(0).total_memory),
            "system_ram_total_bytes": int(vm.total),
            "system_ram_available_bytes_at_start": int(vm.available),
        },
    }
    _json_dump(run_dir / "config.json", config)

    from transformers import DynamicCache

    raw_records: list[dict] = []
    turn_rows: list[dict] = []
    raw_path = run_dir / "raw.jsonl"
    with open(raw_path, "w", encoding="utf-8") as raw_file:
        for dialog in dialogs:
            resized_images, image_metadata = _resize_dialog_images(dialog)
            append_cache = DynamicCache(config=runner.cfg.text_config)
            previous_ids: list[int] = []
            previous_prompt = ""

            for turn in dialog["turns"][:N_TURNS]:
                torch.cuda.reset_peak_memory_stats()
                turn_id = int(turn["turn_id"])
                prompt = mmdu_prompt(dialog, turn_id, generated=None)
                active_ids = list(turn["active_image_ids"])
                new_ids = list(turn["new_image_ids"])
                if not active_ids:
                    raise RuntimeError(
                        f"{dialog['dialog_id']} turn {turn_id}: no active images"
                    )
                expected_markers = len(active_ids)
                if prompt.count("<image>") != expected_markers:
                    raise RuntimeError(
                        f"{dialog['dialog_id']} turn {turn_id}: prompt has "
                        f"{prompt.count('<image>')} image markers, expected "
                        f"{expected_markers}"
                    )
                active_pil = [resized_images[image_id]
                              for image_id in active_ids]
                encoded = runner.processor(
                    text=prompt, images=active_pil, padding=True,
                    return_tensors="pt")
                if int(encoded["input_ids"].shape[0]) != 1:
                    raise RuntimeError("processor returned more than one text batch")
                if not bool(encoded["attention_mask"].bool().all()):
                    raise RuntimeError("unexpected padding in one-sample processor output")
                ids = [int(x) for x in encoded["input_ids"][0].tolist()]
                prompt_length = len(ids)
                if prompt_length + args.max_new_tokens > effective_limit:
                    failure = {
                        "dialog_id": dialog["dialog_id"],
                        "turn_id": turn_id,
                        "prompt_tokens": prompt_length,
                        "reserved_decode_tokens": args.max_new_tokens,
                        "total_reserved_tokens": prompt_length + args.max_new_tokens,
                        "effective_context_limit": effective_limit,
                    }
                    _json_dump(run_dir / "validation.json", {
                        "schema_version": SCHEMA_VERSION,
                        "append_only_dynamic_cache_gate_passed": False,
                        "failure": "context_limit_exceeded",
                        "context": failure,
                        "static_diverse_mmdu_gate_passed": False,
                    })
                    raise RuntimeError(
                        "context limit exceeded: " + json.dumps(failure)
                    )

                prefix_exact = ids[:len(previous_ids)] == previous_ids
                prompt_text_prefix_exact = prompt.startswith(previous_prompt)
                if not prefix_exact:
                    raise RuntimeError(
                        f"{dialog['dialog_id']} turn {turn_id}: full processor "
                        "token IDs are not an exact extension of the prior turn; "
                        "append-only cache is invalid"
                    )
                if not prompt_text_prefix_exact:
                    raise RuntimeError(
                        f"{dialog['dialog_id']} turn {turn_id}: canonical prompt "
                        "is not append-only text"
                    )
                delta_ids = ids[len(previous_ids):]
                reconstructed_ids = previous_ids + delta_ids
                full_spans = _named_spans(
                    ids, active_ids, runner.image_token_id)
                delta_spans = _named_spans(
                    delta_ids, new_ids, runner.image_token_id,
                    offset=len(previous_ids))
                expected_delta_spans = [span for span in full_spans
                                        if span["image_id"] in set(new_ids)]
                visual_spans_exact = delta_spans == expected_delta_spans
                if not visual_spans_exact:
                    raise RuntimeError(
                        f"{dialog['dialog_id']} turn {turn_id}: new-image visual "
                        "spans differ between full and append token segments"
                    )

                full_pixels = encoded["pixel_values"]
                full_sizes = encoded["image_sizes"]
                new_pixels, new_sizes = _new_image_tensors(
                    encoded, active_ids, new_ids)

                fresh_cache = DynamicCache(config=runner.cfg.text_config)
                fresh_cache, full_logits = _prefill(
                    runner,
                    encoded["input_ids"],
                    fresh_cache,
                    start_position=0,
                    total_length=prompt_length,
                    pixel_values=full_pixels,
                    image_sizes=full_sizes,
                )
                full_cache_length = _cache_length(fresh_cache)
                kv_geometry = _visual_kv_geometry(fresh_cache, full_spans)

                append_before = _cache_length(append_cache)
                if append_before != len(previous_ids):
                    raise RuntimeError(
                        f"{dialog['dialog_id']} turn {turn_id}: append cache has "
                        f"{append_before}, expected {len(previous_ids)} rows"
                    )
                delta_tensor = torch.tensor(
                    [delta_ids], dtype=encoded["input_ids"].dtype)
                append_cache, append_logits = _prefill(
                    runner,
                    delta_tensor,
                    append_cache,
                    start_position=append_before,
                    total_length=prompt_length,
                    pixel_values=new_pixels,
                    image_sizes=new_sizes,
                )
                append_after = _cache_length(append_cache)

                logit_diff = (full_logits - append_logits).abs()
                max_abs = float(logit_diff.max().item())
                mean_abs = float(logit_diff.mean().item())
                logits_finite = bool(torch.isfinite(logit_diff).all().item())
                logits_within_tolerance = bool(
                    logits_finite
                    and max_abs <= args.logit_max_atol
                    and mean_abs <= args.logit_mean_atol
                )
                full_first = int(full_logits.argmax().item())
                append_first = int(append_logits.argmax().item())

                full_generated_ids, full_response = _greedy_from_first_logits(
                    runner, fresh_cache, full_logits, prompt_length,
                    args.max_new_tokens)
                del fresh_cache
                append_generated_ids, append_response = _greedy_from_first_logits(
                    runner, append_cache, append_logits, prompt_length,
                    args.max_new_tokens)
                append_cache.crop(prompt_length)
                append_restored = _cache_length(append_cache)
                cache_length_exact = (
                    full_cache_length == prompt_length
                    and append_after == prompt_length
                    and append_restored == prompt_length
                )

                record = {
                    "schema_version": SCHEMA_VERSION,
                    "dialog_id": dialog["dialog_id"],
                    "turn_id": turn_id,
                    "source_question": turn["question"],
                    "source_image_marker": "<ImageHere>",
                    "source_image_marker_count": int(
                        turn.get("image_marker_count",
                                 turn["question"].count("<ImageHere>"))),
                    "processor_image_marker": "<image>\\n",
                    "gold_answer": turn["gold_answer"],
                    "prior_gold_answers": [
                        t["gold_answer"] for t in dialog["turns"][:turn_id - 1]],
                    "history_policy": "gold_teacher_forced",
                    "prompt": prompt,
                    "prompt_sha256": _sha256_text(prompt),
                    "prompt_text_prefix_exact": prompt_text_prefix_exact,
                    "active_image_ids": active_ids,
                    "new_image_ids": new_ids,
                    "active_image_order_exact": (
                        active_ids == list(dialog["image_ids"][:len(active_ids)])),
                    "new_image_order_exact": (
                        not new_ids or active_ids[-len(new_ids):] == new_ids),
                    "active_images": len(active_ids),
                    "new_images_this_turn": len(new_ids),
                    "active_image_metadata": [image_metadata[x] for x in active_ids],
                    "new_image_metadata": [image_metadata[x] for x in new_ids],
                    "full_input_ids": ids,
                    "full_input_tokens": tokenizer.convert_ids_to_tokens(ids),
                    "full_input_ids_sha256": _hash_ids(ids),
                    "full_input_length": prompt_length,
                    "previous_input_length": len(previous_ids),
                    "delta_input_ids": delta_ids,
                    "delta_input_tokens": tokenizer.convert_ids_to_tokens(delta_ids),
                    "delta_input_ids_sha256": _hash_ids(delta_ids),
                    "delta_input_length": len(delta_ids),
                    "append_reconstructed_input_ids": reconstructed_ids,
                    "append_reconstructed_ids_sha256": _hash_ids(
                        reconstructed_ids),
                    "prefix_exact": prefix_exact,
                    "token_sequence_exact": reconstructed_ids == ids,
                    "full_visual_spans": full_spans,
                    "append_delta_visual_spans_absolute": delta_spans,
                    "visual_spans_exact": visual_spans_exact,
                    "cache_lengths": {
                        "full_after_prefill": full_cache_length,
                        "append_before_prefill": append_before,
                        "append_after_prefill": append_after,
                        "append_after_generation_crop": append_restored,
                    },
                    "cache_length_exact": cache_length_exact,
                    "visual_tokens": kv_geometry["active_visual_tokens"],
                    "active_visual_tokens": kv_geometry["active_visual_tokens"],
                    "total_context_tokens": prompt_length,
                    "full_visual_kv_bytes": kv_geometry["full_visual_kv_bytes"],
                    "selected_visual_kv_bytes": None,
                    "cumulative_visual_kv_bytes_created":
                        kv_geometry["full_visual_kv_bytes"],
                    "image_visual_kv_bytes":
                        kv_geometry["image_visual_kv_bytes"],
                    "visual_kv_geometry": kv_geometry,
                    "first_token_logits": {
                        "vocabulary_size": int(full_logits.numel()),
                        "comparison_dtype": "float32_cpu",
                        "max_abs_diff": max_abs,
                        "mean_abs_diff": mean_abs,
                        "all_differences_finite": logits_finite,
                        "max_abs_tolerance": args.logit_max_atol,
                        "mean_abs_tolerance": args.logit_mean_atol,
                        "within_tolerance": logits_within_tolerance,
                    },
                    "full_first_token_id": full_first,
                    "append_first_token_id": append_first,
                    "full_first_token_text": tokenizer.decode(
                        [full_first], skip_special_tokens=False),
                    "append_first_token_text": tokenizer.decode(
                        [append_first], skip_special_tokens=False),
                    "first_token_exact": full_first == append_first,
                    "full_generated_token_ids": full_generated_ids,
                    "append_generated_token_ids": append_generated_ids,
                    "generated_token_ids_exact": (
                        full_generated_ids == append_generated_ids),
                    "full_generated_response": full_response,
                    "append_generated_response": append_response,
                    "generated_response_exact": full_response == append_response,
                    "context": {
                        "prompt_tokens": prompt_length,
                        "reserved_decode_tokens": args.max_new_tokens,
                        "total_reserved_tokens": (
                            prompt_length + args.max_new_tokens),
                        "effective_limit": effective_limit,
                        "within_limit": True,
                    },
                    "existing_image_prefix_ssd_stores_used": False,
                    "independent_image_prefix_kv_concatenation_used": False,
                    "static_diverse_mmdu_gate_passed": False,
                    "gpu_memory_allocated": int(torch.cuda.memory_allocated()),
                    "gpu_peak_memory_allocated": int(
                        torch.cuda.max_memory_allocated()),
                    "process_rss_bytes": int(process.memory_info().rss),
                }
                raw_file.write(json.dumps(
                    record, ensure_ascii=False, separators=(",", ":")) + "\n")
                raw_file.flush()
                raw_records.append(record)
                turn_rows.append({
                    "dialog_id": dialog["dialog_id"],
                    "turn_id": turn_id,
                    "active_image_count": len(active_ids),
                    "new_image_count": len(new_ids),
                    "prompt_tokens": prompt_length,
                    "active_visual_tokens": kv_geometry["active_visual_tokens"],
                    "full_visual_kv_bytes": kv_geometry["full_visual_kv_bytes"],
                    "cumulative_visual_kv_bytes_created":
                        kv_geometry["full_visual_kv_bytes"],
                    "previous_prompt_tokens": len(previous_ids),
                    "delta_tokens": len(delta_ids),
                    "prefix_exact": prefix_exact,
                    "token_sequence_exact": reconstructed_ids == ids,
                    "visual_spans_exact": visual_spans_exact,
                    "cache_length_exact": cache_length_exact,
                    "first_logit_max_abs_diff": max_abs,
                    "first_logit_mean_abs_diff": mean_abs,
                    "logit_differences_finite": logits_finite,
                    "logits_within_tolerance": logits_within_tolerance,
                    "full_first_token_id": full_first,
                    "append_first_token_id": append_first,
                    "first_token_exact": full_first == append_first,
                    "full_generated_tokens": len(full_generated_ids),
                    "append_generated_tokens": len(append_generated_ids),
                    "generated_token_ids_exact": (
                        full_generated_ids == append_generated_ids),
                    "generated_response_exact": full_response == append_response,
                    "full_generated_response": full_response,
                    "append_generated_response": append_response,
                    "context_total_reserved": (
                        prompt_length + args.max_new_tokens),
                    "context_limit": effective_limit,
                })
                previous_ids = ids
                previous_prompt = prompt
                del encoded, full_pixels, full_sizes, new_pixels, new_sizes
                del full_logits, append_logits, logit_diff
                torch.cuda.empty_cache()

            for image in resized_images.values():
                image.close()
            del append_cache
            torch.cuda.empty_cache()

    if len(turn_rows) != N_DIALOGS * N_TURNS:
        raise RuntimeError(
            f"expected {N_DIALOGS * N_TURNS} turn records, got {len(turn_rows)}"
        )

    turn_fields = [
        "dialog_id", "turn_id", "active_image_count", "new_image_count",
        "prompt_tokens", "previous_prompt_tokens", "delta_tokens",
        "active_visual_tokens", "full_visual_kv_bytes",
        "cumulative_visual_kv_bytes_created",
        "prefix_exact", "token_sequence_exact", "visual_spans_exact",
        "cache_length_exact", "first_logit_max_abs_diff",
        "first_logit_mean_abs_diff", "logit_differences_finite",
        "logits_within_tolerance",
        "full_first_token_id", "append_first_token_id", "first_token_exact",
        "full_generated_tokens", "append_generated_tokens",
        "generated_token_ids_exact", "generated_response_exact",
        "full_generated_response", "append_generated_response",
        "context_total_reserved", "context_limit",
    ]
    _write_csv(run_dir / "per_turn.csv", turn_rows, turn_fields)
    dialog_rows = _aggregate_dialogs(turn_rows)
    dialog_fields = [
        "dialog_id", "turns", "max_prompt_tokens", "all_prefix_exact",
        "all_token_sequences_exact", "all_visual_spans_exact",
        "all_cache_lengths_exact", "all_first_tokens_exact",
        "all_generated_token_ids_exact", "all_generated_responses_exact",
        "all_logits_within_tolerance",
        "max_first_logit_abs_diff", "max_first_logit_mean_abs_diff",
    ]
    _write_csv(run_dir / "per_dialog.csv", dialog_rows, dialog_fields)

    all_checks = {
        "dialog_count_exact": len(dialog_rows) == N_DIALOGS,
        "turn_count_exact": len(turn_rows) == N_DIALOGS * N_TURNS,
        "all_prefixes_exact": all(r["prefix_exact"] for r in turn_rows),
        "all_token_sequences_exact": all(
            r["token_sequence_exact"] for r in turn_rows),
        "all_visual_spans_exact": all(
            r["visual_spans_exact"] for r in turn_rows),
        "all_active_image_orders_exact": all(
            r["active_image_order_exact"] for r in raw_records),
        "all_new_image_orders_exact": all(
            r["new_image_order_exact"] for r in raw_records),
        "all_cache_lengths_exact": all(
            r["cache_length_exact"] for r in turn_rows),
        "all_logit_differences_finite": all(
            r["logit_differences_finite"] for r in turn_rows),
        "all_logits_within_predeclared_tolerance": all(
            r["logits_within_tolerance"] for r in turn_rows),
        "all_first_tokens_exact": all(
            r["first_token_exact"] for r in turn_rows),
        "all_generated_token_ids_exact": all(
            r["generated_token_ids_exact"] for r in turn_rows),
        "all_generated_responses_exact": all(
            r["generated_response_exact"] for r in turn_rows),
        "all_contexts_within_4096": all(
            r["context_total_reserved"] <= HARD_CONTEXT_LIMIT
            for r in turn_rows),
    }
    dynamic_gate = all(all_checks.values())
    validation = {
        "schema_version": SCHEMA_VERSION,
        "validation_scope": "native_append_only_transformers_DynamicCache",
        "checks": all_checks,
        "append_only_dynamic_cache_gate_passed": dynamic_gate,
        "max_observed_first_logit_abs_diff": max(
            r["first_logit_max_abs_diff"] for r in turn_rows),
        "max_observed_first_logit_mean_abs_diff": max(
            r["first_logit_mean_abs_diff"] for r in turn_rows),
        "first_logit_tolerance": config["first_logit_tolerance"],
        "note_on_logits": (
            "The predeclared max and mean absolute tolerances are both gated; "
            "the behavioral gate additionally requires exact greedy first-token "
            "and generated-token equality."
        ),
        "existing_image_prefix_ssd_stores_used": False,
        "independent_image_prefix_kv_concatenation_used": False,
        "static_diverse_mmdu_gate_passed": False,
        "static_diverse_gate_reason": (
            "separate gate remains closed; SSD/static-diverse serving was not run"
        ),
        "official_prompt_equivalence_claimed": False,
        "official_mmdu_generation_protocol_gate_passed": False,
        "official_protocol_gate_reason": (
            "official [INST] plus generated-history quality protocol was not run"
        ),
        "official_resize_equivalence_claimed": True,
        "completed_unix": time.time(),
        "elapsed_seconds": time.time() - started,
    }
    _json_dump(run_dir / "validation.json", validation)

    summary_row = {
        "dialogs": len(dialog_rows),
        "turns": len(turn_rows),
        "first_token_match_rate": sum(
            r["first_token_exact"] for r in turn_rows) / len(turn_rows),
        "generated_token_ids_match_rate": sum(
            r["generated_token_ids_exact"] for r in turn_rows) / len(turn_rows),
        "generated_response_match_rate": sum(
            r["generated_response_exact"] for r in turn_rows) / len(turn_rows),
        "max_prompt_tokens": max(r["prompt_tokens"] for r in turn_rows),
        "max_context_reserved_tokens": max(
            r["context_total_reserved"] for r in turn_rows),
        "max_active_visual_tokens": max(
            r["active_visual_tokens"] for r in turn_rows),
        "max_full_visual_kv_bytes": max(
            r["full_visual_kv_bytes"] for r in turn_rows),
        "max_first_logit_abs_diff": validation[
            "max_observed_first_logit_abs_diff"],
        "max_first_logit_mean_abs_diff": validation[
            "max_observed_first_logit_mean_abs_diff"],
        "all_logits_within_tolerance": all(
            r["logits_within_tolerance"] for r in turn_rows),
        "append_only_dynamic_cache_gate_passed": dynamic_gate,
        "static_diverse_mmdu_gate_passed": False,
    }
    _write_csv(run_dir / "summary.csv", [summary_row], list(summary_row))
    observed_geometries = {
        json.dumps(r["visual_kv_geometry"], sort_keys=True)
        for r in raw_records
    }
    config["observed_visual_kv_geometries"] = [
        json.loads(value) for value in sorted(observed_geometries)]
    config["visual_kv_bytes_source"] = "actual generated DynamicCache tensors"
    _json_dump(run_dir / "config.json", config)
    with open(run_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(_build_readme(config, turn_rows, dialog_rows, validation))

    print(json.dumps({
        "run_dir": str(run_dir),
        "append_only_dynamic_cache_gate_passed": dynamic_gate,
        "static_diverse_mmdu_gate_passed": False,
        "turns": len(turn_rows),
        "max_first_logit_abs_diff": validation[
            "max_observed_first_logit_abs_diff"],
        "max_first_logit_mean_abs_diff": validation[
            "max_observed_first_logit_mean_abs_diff"],
    }, indent=2))
    if not dynamic_gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

"""Analyze the full MMDU workload without fabricating a selective-SSD run.

Visual-token and byte constants are read from the actual DynamicCache tensors
recorded by ``17_validate_mmdu_cache.py``.  Text lengths for all 110 official
dialogs are then computed with the checkpoint tokenizer.  This is a capacity
and context-feasibility analysis only; it emits no TTFT, SSD, or quality score.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmimpress.config import MODEL_ID, PROJECT_ROOT
from mmimpress.multiturn import load_canonical, mmdu_prompt, sha256_file


def _write_csv(path, rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _load_geometry(correctness_run):
    rows = [json.loads(line) for line in open(correctness_run / "raw.jsonl")
            if line.strip()]
    if not rows:
        raise RuntimeError("correctness raw.jsonl is empty")
    per_image_tokens = set()
    bytes_per_token = set()
    for row in rows:
        geometry = row["visual_kv_geometry"]
        bytes_per_token.add(int(geometry["visual_kv_bytes_per_token"]))
        per_image_tokens.update(int(span["length"])
                                for span in row["full_visual_spans"])
    if len(per_image_tokens) != 1 or len(bytes_per_token) != 1:
        raise RuntimeError((per_image_tokens, bytes_per_token))
    tokens = per_image_tokens.pop()
    bpt = bytes_per_token.pop()
    return rows, tokens, bpt


def _plots(turns, by_active, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [row["active_images"] for row in by_active]
    y = [row["full_visual_kv_gb"] for row in by_active]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(x, y, marker="o")
    ax.set(xlabel="Active images", ylabel="Required Full visual KV (GB)")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out_dir / "graph_active_images_full_visual_kv.png", dpi=180)
    plt.close(fig)

    grouped = defaultdict(list)
    for row in turns:
        grouped[int(row["turn_id"])].append(row)
    tx = sorted(grouped)
    ty = [statistics.mean(r["full_visual_kv_bytes"] for r in grouped[t]) / 1e9
          for t in tx]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(tx, ty, marker="o", markersize=3)
    ax.set(xlabel="Conversation turn", ylabel="Mean active Full visual KV (GB)")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out_dir / "graph_turn_full_visual_kv.png", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(
        PROJECT_ROOT / "data/mmdu/subsets/full_seed1234/index.json"))
    ap.add_argument("--correctness-run", default=str(
        PROJECT_ROOT /
        "runs/mmdu_multiturn/correctness_progressive_seed1234_v2"))
    ap.add_argument("--out-dir", default=str(
        PROJECT_ROOT / "results/mmdu_multiturn/feasibility_seed1234"))
    ap.add_argument("--max-new-tokens", type=int, default=16)
    args = ap.parse_args()

    index_path = Path(args.index).resolve()
    correctness_run = Path(args.correctness_run).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dialogs = load_canonical(index_path)
    gate = json.load(open(correctness_run / "validation.json"))
    observed, visual_tokens_per_image, kv_bytes_per_token = _load_geometry(
        correctness_run)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    image_token_id = int(tokenizer.convert_tokens_to_ids("<image>"))
    if image_token_id == tokenizer.unk_token_id:
        raise RuntimeError("checkpoint tokenizer has no <image> special token")
    rows = []
    per_dialog = []
    for dialog in dialogs:
        dialog_rows = []
        for turn in dialog["turns"]:
            prompt = mmdu_prompt(dialog, int(turn["turn_id"]))
            raw_ids = tokenizer(prompt, add_special_tokens=True).input_ids
            markers = sum(int(token == image_token_id) for token in raw_ids)
            active = len(turn["active_image_ids"])
            if markers != active:
                raise RuntimeError((dialog["dialog_id"], turn["turn_id"],
                                    markers, active))
            # Each raw marker expands to the observed per-image run.  This
            # arithmetic is verified below against all six processor outputs
            # in the progressive correctness gate.
            total = len(raw_ids) + active * (visual_tokens_per_image - 1)
            full_bytes = active * visual_tokens_per_image * kv_bytes_per_token
            row = {
                "dialog_id": dialog["dialog_id"],
                "turn_id": int(turn["turn_id"]),
                "active_images": active,
                "new_images_this_turn": len(turn["new_image_ids"]),
                "text_and_marker_tokens_before_expansion": len(raw_ids),
                "history_and_prompt_text_tokens": len(raw_ids) - active,
                "active_visual_tokens": active * visual_tokens_per_image,
                "total_context_tokens": total,
                "reserved_context_tokens": total + args.max_new_tokens,
                "within_4096_with_reserved_decode":
                    total + args.max_new_tokens <= 4096,
                "full_visual_kv_bytes": full_bytes,
                "cumulative_visual_kv_bytes_created": full_bytes,
            }
            rows.append(row)
            dialog_rows.append(row)
        per_dialog.append({
            "dialog_id": dialog["dialog_id"],
            "turns": len(dialog_rows),
            "images": len(dialog["images"]),
            "max_active_images": max(r["active_images"] for r in dialog_rows),
            "max_context_tokens": max(r["total_context_tokens"] for r in dialog_rows),
            "max_full_visual_kv_bytes": max(
                r["full_visual_kv_bytes"] for r in dialog_rows),
            "all_turns_within_4096": all(
                r["within_4096_with_reserved_decode"] for r in dialog_rows),
        })

    # Validate the expansion arithmetic against real processor sequences.
    exact_checks = []
    lookup = {(r["dialog_id"], int(r["turn_id"])): r for r in rows}
    for record in observed:
        key = (record["dialog_id"], int(record["turn_id"]))
        exact_checks.append(
            lookup[key]["total_context_tokens"] == record["full_input_length"])
    if not all(exact_checks):
        raise RuntimeError("token-length projection disagrees with real processor")

    groups = defaultdict(list)
    for row in rows:
        groups[row["active_images"]].append(row)
    by_active = [{
        "active_images": active,
        "turns": len(group),
        "full_visual_kv_bytes": group[0]["full_visual_kv_bytes"],
        "full_visual_kv_gb": group[0]["full_visual_kv_bytes"] / 1e9,
        "mean_total_context_tokens": statistics.mean(
            r["total_context_tokens"] for r in group),
        "turns_within_4096": sum(
            r["within_4096_with_reserved_decode"] for r in group),
    } for active, group in sorted(groups.items())]

    vm = psutil.virtual_memory()
    correctness_config = json.load(open(correctness_run / "config.json"))
    gpu_total = int(correctness_config["machine"]["gpu_total_memory_bytes"])
    image_records = [image for dialog in dialogs for image in dialog["images"]]
    unique_image_paths = {str(Path(image["image_path"]).resolve())
                          for image in image_records}
    image_kv_bytes = visual_tokens_per_image * kv_bytes_per_token
    all_dataset_bytes = len(image_records) * image_kv_bytes
    deduplicated_dataset_bytes = len(unique_image_paths) * image_kv_bytes
    summary = {
        "dataset": "MMDU official benchmark",
        "dialogs": len(dialogs),
        "turns": len(rows),
        "images": len(image_records),
        "unique_image_paths": len(unique_image_paths),
        "images_per_dialog_min": min(len(d["images"]) for d in dialogs),
        "images_per_dialog_max": max(len(d["images"]) for d in dialogs),
        "active_images_per_turn_mean": statistics.mean(
            r["active_images"] for r in rows),
        "active_images_per_turn_max": max(r["active_images"] for r in rows),
        "visual_tokens_per_336_image": visual_tokens_per_image,
        "visual_kv_bytes_per_token": kv_bytes_per_token,
        "full_visual_kv_bytes_per_336_image":
            visual_tokens_per_image * kv_bytes_per_token,
        "max_dialog_full_visual_kv_bytes": max(
            r["full_visual_kv_bytes"] for r in rows),
        "all_dataset_images_visual_kv_bytes": all_dataset_bytes,
        "all_unique_image_paths_visual_kv_bytes": deduplicated_dataset_bytes,
        "turns_within_4096_prompt_plus_decode": sum(
            r["within_4096_with_reserved_decode"] for r in rows),
        "turns_over_4096_prompt_plus_decode": sum(
            not r["within_4096_with_reserved_decode"] for r in rows),
        "dialogs_all_turns_within_4096": sum(
            d["all_turns_within_4096"] for d in per_dialog),
        "gpu_total_memory_bytes": gpu_total,
        "system_ram_total_bytes": int(vm.total),
        "system_ram_available_bytes": int(vm.available),
        "all_dataset_visual_kv_over_gpu_capacity": all_dataset_bytes / gpu_total,
        "all_dataset_visual_kv_over_system_ram": all_dataset_bytes / vm.total,
        "all_unique_paths_visual_kv_over_gpu_capacity":
            deduplicated_dataset_bytes / gpu_total,
        "all_unique_paths_visual_kv_over_system_ram":
            deduplicated_dataset_bytes / vm.total,
        "geometry_source": str((correctness_run / "raw.jsonl").resolve()),
        "geometry_source_sha256": sha256_file(correctness_run / "raw.jsonl"),
        "geometry_derivation": (
            "actual bf16 DynamicCache K/V tensor shapes, layer count, head count, "
            "head dimension, and element_size from the GPU correctness run"),
        "token_projection_verified_real_processor_turns": len(exact_checks),
        "token_projection_scope": (
            "all workload rows use fixed-336 tokenizer expansion; exact "
            "processor cross-check is limited to the six progressive gate turns"),
        "append_only_dynamic_cache_gate_passed": gate[
            "append_only_dynamic_cache_gate_passed"],
        "static_diverse_mmdu_gate_passed": False,
        "system_ttft_ssd_quality_run_executed": False,
        "reason_not_executed": (
            "predeclared numeric correctness tolerance failed and the current "
            "single-image SSD prefix abstraction is not a valid interleaved "
            "multi-image cache"),
    }
    _write_csv(out_dir / "per_turn.csv", rows)
    _write_csv(out_dir / "per_dialog.csv", per_dialog)
    _write_csv(out_dir / "by_active_images.csv", by_active)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    with open(out_dir / "config.json", "w") as f:
        json.dump({
            "index": str(index_path), "index_sha256": sha256_file(index_path),
            "correctness_run": str(correctness_run),
            "model": MODEL_ID, "resize": "336x336 LANCZOS",
            "max_new_tokens_reserved": args.max_new_tokens,
            "calculation_mode": "actual-cache-geometry plus tokenizer projection",
        }, f, indent=1)
    _plots(rows, by_active, out_dir)
    with open(out_dir / "README.md", "w") as f:
        f.write(f"""# MMDU workload and capacity feasibility

This is a read-only workload/capacity analysis, not a Static+Diverse result.

- Official dialogs / turns / image records: {summary['dialogs']} / {summary['turns']} / {summary['images']}
- Unique resolved image paths: {summary['unique_image_paths']} (the byte total below follows the current per-ID, non-deduplicated store semantics)
- Active images per turn: mean {summary['active_images_per_turn_mean']:.2f}, max {summary['active_images_per_turn_max']}
- Actual visual KV per resized image: {summary['full_visual_kv_bytes_per_336_image']/1e9:.3f} GB
- Largest dialog visual working set: {summary['max_dialog_full_visual_kv_bytes']/1e9:.3f} GB
- All {summary['images']} per-ID image KVs: {summary['all_dataset_images_visual_kv_bytes']/1e9:.3f} GB; path-deduplicated: {summary['all_unique_image_paths_visual_kv_bytes']/1e9:.3f} GB
- Context-feasible turns (`prompt + {args.max_new_tokens} <= 4096`): {summary['turns_within_4096_prompt_plus_decode']} / {summary['turns']}
- Dialogs with every turn context-feasible: {summary['dialogs_all_turns_within_4096']} / {summary['dialogs']}

Bytes come from actual generated bf16 DynamicCache shapes in the progressive
GPU gate. All 1,645 workload lengths are fixed-336 tokenizer projections;
those projections match the six real processor sequences in the gate, which
is the full extent of the processor-level cross-check.
The strict cache gate failed its predeclared numeric tolerance, and the current
single-image SSD store cannot represent contextual later-image KV. Therefore no
MMDU TTFT/SSD/quality arm was run and no such score is fabricated here.
""")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

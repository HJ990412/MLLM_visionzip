"""True-TTFT system evaluation for canonical multi-turn workloads.

The SSD-backed path is intentionally enabled only for VisDial, whose one image
is a genuine fixed prefix of every turn.  MMDU must first pass the separate
multi-image correctness gate; this script refuses to fake support by tensor-
concatenating independently computed image-prefix KV.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmimpress.model import LlavaRunner
from mmimpress.multiturn import (generative_match, load_canonical,
                                 resolve_image_path, sha256_file,
                                 visdial_prior_history_text, visdial_prompt)
from mmimpress.multiturn_results import build_artifacts
from mmimpress.serve import (ImageContext, Server, load_static,
                             suffix_ids_from_prompt)


METHODS = {
    "recompute": ("ReComp", None),
    "fullload": ("FullLoad", 1.0),
    "sparsevlm": ("SparseVLM 25%", 0.25),
    "static_diverse@25": ("Static+Diverse 25%", 0.25),
    "static_diverse@50": ("Static+Diverse 50%", 0.50),
}
DEFAULT_METHODS = ",".join(METHODS)


def _hash_ids(ids):
    a = ids.detach().cpu().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def _stable_json_hash(value):
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _store_content_snapshot(manifest):
    """Immutable store semantics, excluding mutable launcher timestamps."""
    stages = manifest["stages"]
    value = {
        "index_sha256": manifest["index_sha256"],
        "store_index_sha256": manifest["store_index_sha256"],
        "image_ids": manifest["image_ids"],
        "calibration_policy": manifest["calibration_policy"],
        "future_turn_calibration_count": manifest["future_turn_calibration_count"],
        "calibration_records": manifest["calibration_records"],
        "reorder": manifest["reorder"],
        "static_selector": manifest["static_selector"],
        "build": {
            "image_kv_build_count": stages["build"]["image_kv_build_count"],
            "full_visual_kv_bytes": stages["build"]["full_visual_kv_bytes"],
        },
        "static": {
            "static_metadata_build_count":
                stages["static"]["static_metadata_build_count"],
            "artifacts": stages["static"]["artifacts"],
        },
    }
    return value


def _machine_metadata():
    vm = psutil.virtual_memory()
    prop = torch.cuda.get_device_properties(0)
    return {
        "gpu_name": prop.name,
        "gpu_total_memory_bytes": int(prop.total_memory),
        "system_ram_total_bytes": int(vm.total),
        "system_ram_available_bytes_at_start": int(vm.available),
        "cpu_count": os.cpu_count(),
    }


def _selected_payload(io):
    return int(sum(int(io.get("per_kind", {}).get(k, {}).get("bytes", 0))
                   for k in ("k", "v", "sep")))


def _result_fields(result, method_key, full_bytes):
    io = result.get("io") or {"bytes": 0, "ms": 0.0, "preads": 0,
                              "chunk_units": 0, "per_kind": {}}
    if method_key == "recompute":
        selected_bytes, ratio = 0, None
    elif method_key == "fullload":
        selected_bytes, ratio = int(full_bytes), 1.0
    else:
        selected_bytes = _selected_payload(io)
        ratio = result.get("logical_kv_ratio")
    # Hook-based FullLoad/SparseVLM interleave model prefill with SSD access
    # and scatter at every layer.  Their hook_ms and prefill interval overlap,
    # so presenting either as an exclusive selector/prefill component would be
    # false.  Preserve the inclusive intervals explicitly and leave exclusive
    # components null.  Static+Diverse is hook-free and has real disjoint
    # selector/read/scatter/prefill measurements.
    hook_based = method_key in ("fullload", "sparsevlm")
    selector = (None if method_key == "sparsevlm" else
                result.get("selector_ms", 0.0))
    scatter = (None if hook_based else result.get("scatter_ms", 0.0))
    prefill = (None if hook_based else result.get("prefill_ms", 0.0))
    return {
        "prediction": result["answer"],
        "ttft_ms": float(result["ttft"]) * 1e3,
        "decode_ms": float(result.get("decode_ms",
                                      result.get("decode_latency", 0.0) * 1e3)),
        "e2e_ms": float(result["e2e_latency"]) * 1e3,
        "selector_ms": (float(selector) if selector is not None else None),
        "ssd_read_ms": float(io.get("ms", 0.0)),
        "ssd_read_bytes": int(io.get("bytes", 0)),
        "ssd_read_chunks": int(io.get("chunk_units", 0)),
        "ssd_read_chunk_units": int(io.get("chunk_units", 0)),
        "ssd_preads": int(io.get("preads", 0)),
        "scatter_ms": (float(scatter) if scatter is not None else None),
        "prefill_ms": (float(prefill) if prefill is not None else None),
        "hook_total_ms": (float(result.get("hook_ms", 0.0))
                          if hook_based else None),
        "prefill_inclusive_ms": (float(result.get("prefill_ms", 0.0))
                                 if hook_based else None),
        "breakdown_semantics": (
            "hook_interleaved_components_not_exclusive" if hook_based else
            "exclusive_hook_free_components"),
        "prepare_ms": float(result.get("prepare_ms", 0.0) or 0.0),
        "generated_tokens": int(result.get("generated_tokens", 0)),
        "selected_visual_kv_bytes": selected_bytes,
        "selected_kv_ratio": (float(ratio) if ratio is not None else None),
        "ssd_payload_ratio_vs_full_visual_kv": (
            float(selected_bytes) / float(full_bytes)
            if full_bytes and method_key != "recompute" else None),
        "n_chunks_selected": result.get("n_chunks_selected"),
        "n_chunks_total": result.get("n_chunks_total"),
        "touched_chunk_fraction": result.get("touched_chunk_fraction"),
        "fallback_rate": result.get("fallback_rate"),
        "io_detail": io.get("per_kind", {}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--methods", default=DEFAULT_METHODS)
    ap.add_argument("--max-dialogs", type=int, default=None)
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    index_path, store_dir = Path(args.index), Path(args.store)
    dialogs = load_canonical(index_path)
    if args.max_dialogs is not None:
        dialogs = dialogs[:args.max_dialogs]
    assert dialogs and all(d["dataset"] == "visdial_v1.0_val" for d in dialogs), \
        "SSD multi-turn evaluation currently supports VisDial only; run the MMDU correctness gate"

    method_keys = [x.strip() for x in args.methods.split(",") if x.strip()]
    assert method_keys and all(x in METHODS for x in method_keys), method_keys
    labels = [METHODS[x][0] for x in method_keys]
    manifest_path = store_dir / "pipeline_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"missing causal store provenance manifest: {manifest_path}")
    with open(manifest_path) as f:
        store_manifest = json.load(f)
    assert store_manifest["index_sha256"] == sha256_file(index_path)
    assert store_manifest["calibration_policy"] == "caption_only_pre_dialog"
    assert int(store_manifest["future_turn_calibration_count"]) == 0
    assert store_manifest["stages"]["build"]["done"]
    assert store_manifest["stages"]["reorder"]["done"]
    assert store_manifest["stages"]["static"]["done"]
    built_images = int(store_manifest["stages"]["build"]["image_kv_build_count"])
    static_built = int(store_manifest["stages"]["static"]["static_metadata_build_count"])
    manifest_image_ids = list(store_manifest["image_ids"])
    evaluated_image_ids = [d["image_ids"][0] for d in dialogs]
    assert built_images == static_built == len(manifest_image_ids)
    assert set(evaluated_image_ids).issubset(set(manifest_image_ids))
    static_artifacts = {
        row["image_id"]: row
        for row in store_manifest["stages"]["static"].get("artifacts", [])
    }
    assert set(static_artifacts) == set(manifest_image_ids)
    for image_id in evaluated_image_ids:
        d = store_dir / image_id
        assert sha256_file(d / "meta.json") == static_artifacts[image_id]["meta_sha256"]
        assert sha256_file(d / "static.pt") == static_artifacts[image_id]["static_pt_sha256"]
        assert (d / "sep_kv.bin").stat().st_size == static_artifacts[image_id]["sep_kv_bytes"]
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "raw.jsonl"
    cfg_path = run_dir / "config.json"
    if raw_path.exists() and raw_path.stat().st_size and not args.resume:
        raise FileExistsError(f"{raw_path} exists; choose a new run dir or pass --resume")
    if (raw_path.exists() and raw_path.stat().st_size and args.resume and
            not cfg_path.exists()):
        raise RuntimeError(
            f"unsafe resume: non-empty {raw_path} has no config.json")
    if cfg_path.exists() and not args.resume:
        raise FileExistsError(
            f"{cfg_path} exists; choose a new run dir or pass --resume")

    completed = set()
    if args.resume and raw_path.exists():
        with open(raw_path) as f:
            for line_no, line in enumerate(f, 1):
                if line.strip():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            f"unsafe resume: malformed JSONL at line {line_no}; "
                            "preserve the file and repair it explicitly") from exc
                    completed.add((r["dialog_id"], int(r["turn_id"]), r["method"]))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    runner = LlavaRunner().load()
    server = Server(runner, ratio=0.25, max_new_tokens=args.max_new_tokens)
    machine = _machine_metadata()
    max_context = int(runner.cfg.text_config.max_position_embeddings)
    expected_request_keys = [
        [d["dialog_id"], int(t["turn_id"])]
        for d in dialogs
        for t in d["turns"][:min(len(d["turns"]),
                                 args.max_turns or len(d["turns"]))]
    ]
    config = {
        "schema_version": "multiturn-results-v1-true-ttft",
        "dataset": "visdial_v1.0_val",
        "index": str(index_path.resolve()),
        "index_sha256": sha256_file(index_path),
        "store": str(store_dir.resolve()),
        "store_manifest": str(manifest_path.resolve()),
        # pipeline_manifest.json also tracks later eval/analyze stages and is
        # therefore mutable.  Only this immutable build/reorder/static subset
        # is a semantic run fingerprint.
        "store_manifest_sha256_at_eval_start": sha256_file(manifest_path),
        "store_content_snapshot": _store_content_snapshot(store_manifest),
        "store_content_fingerprint": _stable_json_hash(
            _store_content_snapshot(store_manifest)),
        "seed": args.seed,
        "n_dialogs": len(dialogs),
        "n_turns": sum(min(len(d["turns"]), args.max_turns or 10**9)
                       for d in dialogs),
        "expected_request_keys": expected_request_keys,
        "expected_request_keys_sha256": _stable_json_hash(expected_request_keys),
        "max_dialogs_requested": args.max_dialogs,
        "max_turns_requested": args.max_turns,
        "methods": labels,
        "method_keys": method_keys,
        "history_policy": "gold_teacher_forced",
        "calibration_policy": "caption_only_pre_dialog",
        "future_turn_calibration_count": 0,
        "prompt_template": "visdial-gold-history-v1",
        "model": runner.model_id,
        "load_4bit": runner.load_4bit,
        "quantization": "NF4 double-quant",
        "attention": runner.attn,
        "decoding": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "chunk_size": 64,
        "sparsevlm_budget": 0.25,
        "static_diverse_budgets": [0.25, 0.50],
        "diverse_frac": 0.25,
        "cold_page_cache": not args.warm,
        "ttft_definition": ("request start -> selector -> SSD pread -> cache "
                            "reconstruction/scatter -> prompt prefill -> first "
                            "output token argmax -> CUDA synchronize"),
        "input_preprocessing_timed": False,
        "max_position_embeddings": max_context,
        "quality_mode": "generative_auxiliary_normalized_match",
        "field_semantics": {
            "selected_kv_ratio": "logical kept visual-token ratio",
            "ssd_payload_ratio_vs_full_visual_kv": (
                "actual visual K/V payload (plus mandatory separator sidecar) "
                "divided by Full visual K/V; excludes SparseVLM probe bytes"),
            "ssd_read_bytes": "all bytes returned by actual os.pread calls",
            "ssd_read_chunk_units": (
                "K/V/probe file-level chunk-equivalent units, not unique chunks"),
            "n_chunks_selected": (
                "Static+Diverse mean unique selected chunks per layer, excluding "
                "separator sidecar"),
        },
        "machine": machine,
        "image_kv_build_count": built_images,
        "static_metadata_build_count": static_built,
    }
    prior_elapsed = 0.0
    if cfg_path.exists() and args.resume:
        with open(cfg_path) as f:
            old = json.load(f)
        semantic_keys = (
            "schema_version", "dataset", "index_sha256", "store",
            "store_content_fingerprint", "seed", "n_dialogs", "n_turns",
            "expected_request_keys_sha256",
            "max_dialogs_requested", "max_turns_requested", "methods",
            "method_keys", "history_policy", "calibration_policy",
            "future_turn_calibration_count", "prompt_template", "model",
            "load_4bit", "quantization", "attention", "decoding",
            "max_new_tokens", "chunk_size", "sparsevlm_budget",
            "static_diverse_budgets", "diverse_frac", "cold_page_cache",
            "ttft_definition", "max_position_embeddings", "quality_mode",
        )
        mismatch = [k for k in semantic_keys if old.get(k) != config.get(k)]
        if mismatch:
            details = {k: {"existing": old.get(k), "requested": config.get(k)}
                       for k in mismatch}
            raise RuntimeError("unsafe resume: semantic config mismatch:\n" +
                               json.dumps(details, indent=1))
        prior_elapsed = float(old.get("elapsed_seconds", 0.0))
        config["run_started_at_unix"] = old.get("run_started_at_unix", time.time())
        config["resume_count"] = int(old.get("resume_count", 0)) + 1
        config["machine_at_initial_start"] = old.get("machine_at_initial_start",
                                                         old.get("machine"))
    else:
        config["run_started_at_unix"] = time.time()
        config["resume_count"] = 0
        config["machine_at_initial_start"] = machine
        with open(cfg_path, "w") as f:
            json.dump(config, f, indent=1)

    process = psutil.Process()
    build_count, static_count = {}, {}
    start = time.time()
    with open(raw_path, "a") as raw:
        for di, dialog in enumerate(dialogs, 1):
            image_id = dialog["image_ids"][0]
            image_store = store_dir / image_id
            assert (image_store / "meta.json").exists(), image_store
            assert (image_store / "static.pt").exists(), image_store
            ctx = ImageContext(image_store, runner.model.device)
            build_count[dialog["dialog_id"]] = 1
            static = load_static(ctx)
            static_count[dialog["dialog_id"]] = 1
            image = Image.open(resolve_image_path(
                dialog["images"][0]["image_path"])).convert("RGB")
            full_bytes = int(ctx.meta["bytes_visual_kv"])
            nturn = min(len(dialog["turns"]), args.max_turns or len(dialog["turns"]))
            previous_history_tokens = -1

            for turn in dialog["turns"][:nturn]:
                ti = int(turn["turn_id"])
                prompt = visdial_prompt(dialog, ti)
                history = visdial_prior_history_text(dialog, ti)
                history_ids = runner.processor.tokenizer(
                    history, add_special_tokens=False).input_ids
                assert len(history_ids) >= previous_history_tokens
                previous_history_tokens = len(history_ids)
                suffix_ids = suffix_ids_from_prompt(runner, prompt)
                suffix_hash = _hash_ids(suffix_ids)
                history_hash = hashlib.sha256(history.encode()).hexdigest()

                # One processor call per turn, outside every method's timer.
                # It both supplies ReComp pixels and proves the stored prefix
                # is byte-for-byte the prefix of this accumulated dialogue.
                enc = runner.encode_prompt(image, prompt)
                total_tokens = int(enc["input_ids"].shape[1])
                assert total_tokens == ctx.meta["prefix_len"] + len(suffix_ids)
                expected_prefix = torch.tensor(ctx.meta["prefix_input_ids"],
                                               dtype=enc["input_ids"].dtype)
                assert torch.equal(enc["input_ids"][0, :ctx.meta["prefix_len"]],
                                   expected_prefix)
                assert total_tokens + args.max_new_tokens <= max_context, \
                    (f"context overflow {dialog['dialog_id']} turn {ti}: "
                     f"prompt={total_tokens} + generation={args.max_new_tokens} "
                     f"> {max_context}")

                for method_key in method_keys:
                    label, budget = METHODS[method_key]
                    key = (dialog["dialog_id"], ti, label)
                    if key in completed:
                        continue
                    torch.cuda.reset_peak_memory_stats()
                    if method_key == "recompute":
                        result = server.recompute(enc)
                    elif method_key == "fullload":
                        result = server.request(
                            ctx, mode="fullload", cold=not args.warm,
                            suffix_ids=suffix_ids)
                    elif method_key == "sparsevlm":
                        result = server.request(
                            ctx, mode="impress", cold=not args.warm,
                            suffix_ids=suffix_ids)
                    else:
                        result = server.request_cvpr25(
                            ctx, static=static, budget=budget,
                            mode="static_diverse", sep_policy="sidecar",
                            diverse_frac=0.25, cold=not args.warm,
                            seed=args.seed, image_id=image_id,
                            suffix_ids=suffix_ids)
                    measured = _result_fields(result, method_key, full_bytes)
                    pred = measured["prediction"]
                    rec = {
                        "schema_version": "multiturn-results-v1-true-ttft",
                        "dataset": "visdial_v1.0_val",
                        "dialog_id": dialog["dialog_id"],
                        "turn_id": ti,
                        "method": label,
                        "method_key": method_key,
                        "budget": budget,
                        "active_images": len(turn["active_image_ids"]),
                        "active_image_ids": turn["active_image_ids"],
                        "new_images_this_turn": len(turn["new_image_ids"]),
                        "new_image_ids": turn["new_image_ids"],
                        "history_tokens": len(history_ids),
                        "history_text_tokens": len(history_ids),
                        "visual_tokens": int(ctx.meta["v_token_num"]),
                        "active_visual_tokens": int(ctx.meta["v_token_num"]),
                        "total_context_tokens": total_tokens,
                        "question": turn["question"],
                        "gold": turn["gold_answer"],
                        "quality_score": generative_match(pred, turn["gold_answer"]),
                        "quality_metric": "auxiliary_normalized_match_not_official_visdial",
                        "text_history_sha256": history_hash,
                        "suffix_ids_sha256": suffix_hash,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "full_visual_kv_bytes": full_bytes,
                        "cumulative_visual_kv_bytes_created": full_bytes,
                        "image_visual_kv_bytes": {image_id: full_bytes},
                        "image_kv_build_count_dialog": 1,
                        "static_metadata_build_count_dialog": 1,
                        "image_kv_build_provenance": "verified_store_manifest",
                        "static_metadata_build_provenance": "verified_store_manifest",
                        "image_store_reused_across_turns": True,
                        "gpu_memory_allocated": int(torch.cuda.memory_allocated()),
                        "gpu_peak_memory_allocated": int(torch.cuda.max_memory_allocated()),
                        "process_rss_bytes": int(process.memory_info().rss),
                        **measured,
                    }
                    raw.write(json.dumps(rec) + "\n")
                    raw.flush()
                    print(f"[{di}/{len(dialogs)} t{ti}/{nturn}] {label}: "
                          f"TTFT={rec['ttft_ms']:.1f}ms SSD={rec['ssd_read_bytes']/1e6:.1f}MB "
                          f"pred={pred[:50]!r}")
            ctx.close()
            del ctx, static
            torch.cuda.empty_cache()

    config["image_context_open_count"] = sum(build_count.values())
    config["static_metadata_load_count"] = sum(static_count.values())
    config["elapsed_seconds"] = prior_elapsed + (time.time() - start)
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=1)
    _, _, _, _, validation = build_artifacts(run_dir, config)
    print(json.dumps(validation, indent=1))
    if not validation["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

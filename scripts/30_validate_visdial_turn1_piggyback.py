"""Correctness gate: captured Turn-1 store vs independent direct builder.

This is deliberately run only on the two-dialog smoke workload.  The direct
builder is a validation oracle and may execute its own image-saliency and
prefix forwards; those extra forwards never enter the measured main run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmimpress.multiturn import (load_canonical, sha256_file,
                                 visdial_prompt)
from mmimpress.serve import (ImageContext, Server,
                             suffix_ids_from_prompt)
from mmimpress.model import LlavaRunner
from mmimpress.store import load_meta

# Numeric script modules are loaded explicitly rather than copied.
import importlib.util


SCHEMA_VERSION = "visdial-turn1-piggyback-correctness-v2"
RELATIVE_L2_TOLERANCE = 0.01
COSINE_TOLERANCE = 0.9999


def _load_build_module():
    path = Path(__file__).resolve().parent / "01_build_store.py"
    spec = importlib.util.spec_from_file_location(
        "mmimpress_independent_direct_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _atomic_json(path: Path, value) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("x") as handle:
        json.dump(value, handle, indent=1, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _stream_hash(path: Path, limit=None) -> str:
    digest = hashlib.sha256()
    remaining = limit
    with path.open("rb") as handle:
        while True:
            size = 8 << 20 if remaining is None else min(8 << 20, remaining)
            if size <= 0:
                break
            block = handle.read(size)
            if not block:
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    if limit is not None and remaining != 0:
        raise RuntimeError(f"short validation read: {path}")
    return digest.hexdigest()


def _visual_payload_hash(store_dir: Path, meta: dict, budget=None) -> str:
    """Hash physical K/V payload (all rows, or exact first-k chunks)."""
    digest = hashlib.sha256()
    n_chunks = int(meta["n_chunks_per_layer"])
    if budget is None:
        rows = int(meta["v_token_num"])
    else:
        k = max(1, min(n_chunks, int(round(float(budget) * n_chunks))))
        rows = min(int(meta["v_token_num"]), k * int(meta["chunk_size"]))
    row_bytes = int(meta["num_heads"]) * int(meta["head_dim"]) * 2
    nbytes = rows * row_bytes
    for layer in range(int(meta["num_layers"])):
        for kind in ("k", "v"):
            relative = f"layer_{layer:02d}/{kind}.bin"
            digest.update(relative.encode("ascii"))
            digest.update(b"\0")
            digest.update(bytes.fromhex(_stream_hash(
                store_dir / relative, limit=nbytes)))
    sep = store_dir / "sep_kv.bin"
    digest.update(b"sep_kv.bin\0")
    digest.update(bytes.fromhex(_stream_hash(sep)))
    return digest.hexdigest()


def _payload_error_stats(left_dir: Path, right_dir: Path, meta: dict,
                         budget=None) -> dict:
    """Streaming numerical comparison of the bytes a Prefix arm can load.

    A full-sequence 4-bit GEMM and the independent shorter prefix-forward can
    choose different reduction/kernel shapes, so causal prefix values need not
    be bitwise identical.  Report both the byte hashes and scale-normalised
    numerical error; acceptance additionally requires identical greedy output.
    """
    import numpy as np

    chunks = int(meta["n_chunks_per_layer"])
    if budget is None:
        rows = int(meta["v_token_num"])
    else:
        kept = max(1, min(chunks, int(round(float(budget) * chunks))))
        rows = min(int(meta["v_token_num"]),
                   kept * int(meta["chunk_size"]))
    shape = (int(meta["v_token_num"]), int(meta["num_heads"]),
             int(meta["head_dim"]))
    squared_error = left_squared = right_squared = dot = abs_error = 0.0
    count = 0
    max_abs = 0.0

    def accumulate(left, right):
        nonlocal squared_error, left_squared, right_squared, dot
        nonlocal abs_error, count, max_abs
        a = np.asarray(left, dtype=np.float32)
        b = np.asarray(right, dtype=np.float32)
        delta = a - b
        squared_error += float(np.sum(delta * delta, dtype=np.float64))
        left_squared += float(np.sum(a * a, dtype=np.float64))
        right_squared += float(np.sum(b * b, dtype=np.float64))
        dot += float(np.sum(a * b, dtype=np.float64))
        abs_error += float(np.sum(np.abs(delta), dtype=np.float64))
        count += int(delta.size)
        max_abs = max(max_abs, float(np.max(np.abs(delta))))

    for layer in range(int(meta["num_layers"])):
        for kind in ("k", "v"):
            relative = f"layer_{layer:02d}/{kind}.bin"
            left = np.memmap(left_dir / relative, dtype=np.float16,
                             mode="r", shape=shape)
            right = np.memmap(right_dir / relative, dtype=np.float16,
                              mode="r", shape=shape)
            accumulate(left[:rows], right[:rows])
            del left, right
    left_sep = np.fromfile(left_dir / "sep_kv.bin", dtype=np.float16)
    right_sep = np.fromfile(right_dir / "sep_kv.bin", dtype=np.float16)
    assert left_sep.shape == right_sep.shape
    accumulate(left_sep, right_sep)
    return {
        "n_values": count,
        "mae": abs_error / count,
        "rmse": (squared_error / count) ** 0.5,
        "relative_l2": (squared_error / right_squared) ** 0.5,
        "cosine_similarity": dot / (left_squared * right_squared) ** 0.5,
        "max_abs": max_abs,
        "accepted_relative_l2": RELATIVE_L2_TOLERANCE,
        "accepted_min_cosine_similarity": COSINE_TOLERANCE,
    }


def _numerically_close(stats: dict) -> bool:
    return bool(
        stats["relative_l2"] <= RELATIVE_L2_TOLERANCE
        and stats["cosine_similarity"] >= COSINE_TOLERANCE)


def _system_kv_equal(a: Path, b: Path) -> bool:
    left = torch.load(a / "sys_kv.pt", weights_only=True)
    right = torch.load(b / "sys_kv.pt", weights_only=True)
    return (torch.equal(left["k"], right["k"])
            and torch.equal(left["v"], right["v"]))


def _read_jsonl(path: Path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _run_prefix_pair(server, runner, captured_dir, direct_dir, prompt,
                     budget, image_id):
    suffix = suffix_ids_from_prompt(runner, prompt)
    outputs = []
    for store in (captured_dir, direct_dir):
        ctx = ImageContext(store, runner.model.device,
                           require_v_hidden=False)
        try:
            ctx.validate_prefix_layout("visionzip_image_only")
            result = server.request_cvpr25(
                ctx, static=None, budget=budget, mode="prefix",
                sep_policy="sidecar", cold=False, image_id=image_id,
                suffix_ids=suffix,
                expected_prefix_layout="visionzip_image_only")
            outputs.append({
                "first_token_id": int(result["first_token_id"]),
                "prediction": result["answer"],
                "ssd_read_bytes": int(result["io"]["bytes"]),
                "selected_chunk_ids_per_layer":
                    result["selected_chunk_ids_per_layer"],
                "static_score_calls": int(result["static_score_calls"]),
                "query_score_calls": int(result["query_score_calls"]),
                "diversity_calls": int(result["diversity_calls"]),
            })
        finally:
            ctx.close()
    return outputs


def _update_smoke_config(run_dir: Path, validation_path: Path, passed: bool):
    config_path = run_dir / "config.json"
    with config_path.open() as handle:
        config = json.load(handle)
    source = {"path": str(validation_path.resolve()),
              "sha256": sha256_file(validation_path)}
    config["correctness_validation_source"] = source
    config["captured_kv_correctness_validation"] = {
        **source, "passed": bool(passed)}
    _atomic_json(config_path, config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-run", type=Path, required=True)
    parser.add_argument("--captured-store", type=Path, required=True)
    parser.add_argument("--direct-store", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=Path(
        "data/visdial_v1.0/subsets/main_seed1234/index.json"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reuse-direct-store", action="store_true",
                        help="reuse an already-built failed-gate reference "
                             "without modifying it")
    args = parser.parse_args()

    run_dir = args.smoke_run.resolve()
    captured_root = args.captured_store.resolve()
    direct_root = args.direct_store.resolve()
    index_path = args.index.resolve()
    output = (args.output.resolve() if args.output else
              run_dir / "captured_kv_validation.json")
    if output.exists():
        raise FileExistsError(output)
    direct_root.parent.mkdir(parents=True, exist_ok=True)
    if args.reuse_direct_store:
        if not direct_root.is_dir():
            raise FileNotFoundError(direct_root)
    else:
        direct_root.mkdir(exist_ok=False)

    started = time.time()
    checks = {
        "smoke_config_complete": False,
        "exact_two_dialog_workload": False,
        "source_execution_links_to_turn1": False,
        "captured_provenance_no_second_forward": False,
        "visual_token_span_equal": False,
        "kv_shape_equal": False,
        "system_kv_equal": False,
        "layer0_visual_kv_bitwise_equal": False,
        "full_visual_kv_payload_numerically_close": False,
        "permutation_equal": False,
        "prefix25_payload_numerically_close": False,
        "prefix45_payload_numerically_close": False,
        "prefix25_first_token_equal": False,
        "prefix25_prediction_equal": False,
        "prefix45_first_token_equal": False,
        "prefix45_prediction_equal": False,
        "prefix_reader_has_no_scorer_calls": False,
        "prefix_loaded_chunk_ids_and_bytes_equal": False,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "passed": False,
        "checks": checks,
        "smoke_run": str(run_dir),
        "captured_store": str(captured_root),
        "direct_store": str(direct_root),
        "index": str(index_path),
        "index_sha256": sha256_file(index_path),
        "budgets": [0.25, 0.45],
        "payload_numerical_tolerance": {
            "relative_l2_max": RELATIVE_L2_TOLERANCE,
            "cosine_similarity_min": COSINE_TOLERANCE,
            "bitwise_identity_required": False,
            "reason": (
                "4-bit full-sequence and shorter independent-prefix GEMMs may "
                "use shape-dependent floating reduction kernels; numerical "
                "closeness is paired with exact first-token and prediction "
                "agreement"),
        },
        "direct_builder_note": (
            "independent direct validation path; its extra saliency/prefix "
            "forwards are not part of any reported request or persistence "
            "latency"),
        "per_image": [],
        "started_at_unix": started,
    }

    try:
        with (run_dir / "config.json").open() as handle:
            config = json.load(handle)
        checks["smoke_config_complete"] = config.get("status") == "complete"
        dialogs = load_canonical(index_path)[:2]
        checks["exact_two_dialog_workload"] = bool(
            config.get("n_dialogs") == 2
            and config.get("n_turns") == 20
            and len(dialogs) == 2
            and all(len(dialog["turns"]) == 10 for dialog in dialogs))
        raw = _read_jsonl(run_dir / "raw.jsonl")
        persistence = _read_jsonl(run_dir / "persistence.jsonl")
        by_execution = {row["execution_id"]: row for row in raw}
        checks["source_execution_links_to_turn1"] = bool(
            len(persistence) == 2 and all(
                row["source_execution_id"] in by_execution
                and by_execution[row["source_execution_id"]]["turn_id"] == 1
                and by_execution[row["source_execution_id"]][
                    "saliency_capture_enabled"] is True
                for row in persistence))

        runner = LlavaRunner().load()
        server = Server(runner, max_new_tokens=int(
            config.get("max_new_tokens", 16)))
        builder = _load_build_module()
        per_image = []
        for dialog in dialogs:
            image_id = dialog["image_ids"][0]
            captured_dir = captured_root / image_id
            entry = {
                "image_id": image_id,
                "image_path": dialog["images"][0]["image_path"],
                "questions": [{
                    "question_id": f"{dialog['dialog_id']}:validation",
                    "question": dialog["turns"][0]["question"],
                }],
            }
            direct_dir = direct_root / image_id
            if args.reuse_direct_store:
                if not (direct_dir / "meta.json").is_file():
                    raise FileNotFoundError(direct_dir / "meta.json")
                profile = {
                    "reused_existing_direct_validation_store": True,
                    "store": str(direct_dir),
                }
            else:
                _, profile = builder.build_one(
                    runner, entry, direct_root, layout="visionzip",
                    separator_sidecar=True)
            captured_meta = load_meta(captured_dir)
            direct_meta = load_meta(direct_dir)
            captured_layout = torch.load(
                captured_dir / "visionzip_layout.pt", weights_only=True)
            direct_layout = torch.load(
                direct_dir / "visionzip_layout.pt", weights_only=True)

            span_equal = all(captured_meta[key] == direct_meta[key]
                             for key in (
                                 "v_token_start", "v_token_num",
                                 "prefix_len", "prefix_input_ids"))
            shape_equal = all(captured_meta[key] == direct_meta[key]
                              for key in (
                                  "num_layers", "num_heads", "head_dim",
                                  "dtype", "chunk_size",
                                  "n_chunks_per_layer"))
            permutation_equal = bool(
                captured_meta["order"] == direct_meta["order"]
                and torch.equal(
                    captured_layout["stored_to_original"],
                    direct_layout["stored_to_original"])
                and captured_meta["permutation_sha256"]
                    == direct_meta["permutation_sha256"])
            system_equal = _system_kv_equal(captured_dir, direct_dir)
            layer0_equal = all(
                _stream_hash(captured_dir / "layer_00" / f"{kind}.bin")
                == _stream_hash(direct_dir / "layer_00" / f"{kind}.bin")
                for kind in ("k", "v"))
            full_hashes = {
                "captured": _visual_payload_hash(
                    captured_dir, captured_meta),
                "direct": _visual_payload_hash(direct_dir, direct_meta),
            }
            full_error = _payload_error_stats(
                captured_dir, direct_dir, captured_meta)
            selected_hashes = {}
            generations = {}
            prompt = visdial_prompt(dialog, 2)
            scorer_calls_zero = True
            loaded_geometry_equal = True
            for budget in (0.25, 0.45):
                key = f"prefix{int(round(100 * budget))}"
                selected_hashes[key] = {
                    "captured": _visual_payload_hash(
                        captured_dir, captured_meta, budget),
                    "direct": _visual_payload_hash(
                        direct_dir, direct_meta, budget),
                }
                selected_hashes[key]["numerical_error"] = \
                    _payload_error_stats(
                        captured_dir, direct_dir, captured_meta, budget)
                outputs = _run_prefix_pair(
                    server, runner, captured_dir, direct_dir, prompt,
                    budget, image_id)
                generations[key] = outputs
                loaded_geometry_equal = loaded_geometry_equal and bool(
                    outputs[0]["ssd_read_bytes"] == outputs[1]["ssd_read_bytes"]
                    and outputs[0]["selected_chunk_ids_per_layer"]
                    == outputs[1]["selected_chunk_ids_per_layer"])
                scorer_calls_zero = scorer_calls_zero and all(
                    output[call] == 0 for output in outputs
                    for call in ("static_score_calls", "query_score_calls",
                                 "diversity_calls"))

            source_meta_ok = bool(
                captured_meta.get("layout_source")
                    == "turn1_normal_inference_piggyback"
                and captured_meta.get("visual_kv_source")
                    == "turn1_captured_past_key_values"
                and captured_meta.get("separate_vision_forward") is False
                and captured_meta.get("separate_prefix_forward") is False
                and captured_meta.get("future_turns_used_for_layout") == 0
                and captured_meta.get("calibration_questions") == 0)
            image_result = {
                "dialog_id": dialog["dialog_id"],
                "image_id": image_id,
                "direct_build_profile": profile,
                "visual_token_span_equal": span_equal,
                "kv_shape_equal": shape_equal,
                "system_kv_equal": system_equal,
                "layer0_visual_kv_bitwise_equal": layer0_equal,
                "full_visual_kv_hashes": full_hashes,
                "full_visual_kv_bitwise_equal": (
                    full_hashes["captured"] == full_hashes["direct"]),
                "full_visual_kv_numerical_error": full_error,
                "full_visual_kv_payload_numerically_close":
                    _numerically_close(full_error),
                "permutation_equal": permutation_equal,
                "selected_payload_hashes": selected_hashes,
                "prefix25_payload_bitwise_equal": (
                    selected_hashes["prefix25"]["captured"]
                    == selected_hashes["prefix25"]["direct"]),
                "prefix45_payload_bitwise_equal": (
                    selected_hashes["prefix45"]["captured"]
                    == selected_hashes["prefix45"]["direct"]),
                "prefix25_payload_numerically_close": _numerically_close(
                    selected_hashes["prefix25"]["numerical_error"]),
                "prefix45_payload_numerically_close": _numerically_close(
                    selected_hashes["prefix45"]["numerical_error"]),
                "generations": generations,
                "prefix25_first_token_equal": (
                    generations["prefix25"][0]["first_token_id"]
                    == generations["prefix25"][1]["first_token_id"]),
                "prefix25_prediction_equal": (
                    generations["prefix25"][0]["prediction"]
                    == generations["prefix25"][1]["prediction"]),
                "prefix45_first_token_equal": (
                    generations["prefix45"][0]["first_token_id"]
                    == generations["prefix45"][1]["first_token_id"]),
                "prefix45_prediction_equal": (
                    generations["prefix45"][0]["prediction"]
                    == generations["prefix45"][1]["prediction"]),
                "captured_provenance_no_second_forward": source_meta_ok,
                "prefix_reader_has_no_scorer_calls": scorer_calls_zero,
                "prefix_loaded_chunk_ids_and_bytes_equal":
                    loaded_geometry_equal,
            }
            per_image.append(image_result)
            print(f"validated {image_id}: full KV="
                  f"{image_result['full_visual_kv_payload_numerically_close']} "
                  f"P25={image_result['prefix25_prediction_equal']} "
                  f"P45={image_result['prefix45_prediction_equal']}",
                  flush=True)
            torch.cuda.empty_cache()

        payload["per_image"] = per_image
        aggregate_keys = [
            "captured_provenance_no_second_forward", "visual_token_span_equal",
            "kv_shape_equal", "system_kv_equal",
            "layer0_visual_kv_bitwise_equal",
            "full_visual_kv_payload_numerically_close", "permutation_equal",
            "prefix25_payload_numerically_close",
            "prefix45_payload_numerically_close",
            "prefix25_first_token_equal", "prefix25_prediction_equal",
            "prefix45_first_token_equal", "prefix45_prediction_equal",
            "prefix_reader_has_no_scorer_calls",
            "prefix_loaded_chunk_ids_and_bytes_equal",
        ]
        for key in aggregate_keys:
            checks[key] = bool(per_image and all(row[key] for row in per_image))
        payload["passed"] = all(checks.values())
        payload["completed_at_unix"] = time.time()
        payload["elapsed_seconds"] = time.time() - started
        _atomic_json(output, payload)
        _update_smoke_config(run_dir, output, payload["passed"])
        if not payload["passed"]:
            raise SystemExit(2)
    except BaseException as exc:
        if not output.exists():
            payload["error"] = f"{type(exc).__name__}: {exc}"
            payload["traceback"] = traceback.format_exc()
            payload["completed_at_unix"] = time.time()
            payload["elapsed_seconds"] = time.time() - started
            _atomic_json(output, payload)
            _update_smoke_config(run_dir, output, False)
        raise

    print(json.dumps({
        "passed": payload["passed"], "output": str(output),
        "elapsed_seconds": payload["elapsed_seconds"],
    }, indent=1), flush=True)


if __name__ == "__main__":
    main()

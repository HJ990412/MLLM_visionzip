#!/usr/bin/env python3
"""Gated, frozen 20-image x 3-question Qwen correctness-v2 semantic smoke.

This is a small descriptive smoke, not the 40-image/720-request pilot. It
consumes the separately frozen v2 manifest and starts GPU work only after all
15 v2 system gates pass. Every output directory must be new.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mmimpress.dataset import exact_score  # noqa: E402

RUN_ROOT = ROOT / "runs/qwen25_correctness_v2_20260928T081111Z"
VALIDATION_MANIFEST = RUN_ROOT / "validation_manifest.json"
SMOKE_MANIFEST = RUN_ROOT / "smoke_manifest.json"
METHODS = ("recompute", "fullload", "ours25")
GATES = tuple(f"G{i}" for i in range(1, 16))
SCHEMA = "qwen25-correctness-v2-semantic-smoke-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")
                                     ).encode("utf-8")).hexdigest()


def save_json_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def checkpoint(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(handle, value: object) -> None:
    handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False,
                            allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def load_fixed_manifest(path: Path, *, expected_count: int) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(value)
    digest = unsigned.pop("manifest_sha256", None)
    if digest != canonical_hash(unsigned):
        raise RuntimeError(f"manifest content hash mismatch: {path}")
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.read_text(encoding="ascii") != f"{sha256_file(path)}  {path.name}\n":
        raise RuntimeError(f"manifest sidecar mismatch: {path}")
    if value.get("schema_version") != "qwen25-correctness-v2-fixed-workload-v1":
        raise RuntimeError("wrong frozen workload schema")
    if value.get("source_index") != str((ROOT / "data/index.json").resolve()):
        raise RuntimeError("wrong source index path")
    if value.get("source_index_sha256") != sha256_file(ROOT / "data/index.json"):
        raise RuntimeError("source index changed")
    for relative, expected_hash in value.get("source_hashes", {}).items():
        source = (ROOT / relative).resolve()
        if not source.is_relative_to(ROOT) or not source.is_file():
            raise RuntimeError(f"manifest source missing or escaped workspace: {relative}")
        if sha256_file(source) != expected_hash:
            raise RuntimeError(f"manifest source changed: {relative}")
    rows = value.get("samples")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise RuntimeError(f"expected exactly {expected_count} frozen samples")
    if len({row["question_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate frozen question ID")
    index = json.loads((ROOT / "data/index.json").read_text(encoding="utf-8"))
    coordinates = ([(0, q) for q in (4, 5, 6)] + [(i, 4) for i in range(1, 8)]
                   if expected_count == 10 else
                   [(i, q) for i in range(20, 40) for q in (4, 5, 6)])
    expected_keys = {"image_id", "image_path", "image_sha256",
                     "question_id", "question", "gold"}
    for row, (image_no, question_no) in zip(rows, coordinates):
        source = index[image_no]
        question = source["questions"][question_no]
        image_path = (ROOT / source["image_path"]).resolve()
        expected = {"image_id": str(source["image_id"]),
                    "image_path": str(image_path),
                    "image_sha256": sha256_file(image_path),
                    "question_id": str(question["question_id"]),
                    "question": str(question["question"]),
                    "gold": str(question["answer"])}
        if set(row) != expected_keys or row != expected:
            raise RuntimeError(f"frozen sample differs from source: {image_no}/{question_no}")
    return value


def verify_frozen_inputs() -> str:
    path = RUN_ROOT / "frozen_inputs.json"
    if not path.is_file():
        raise RuntimeError("v2 frozen input manifest is missing")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("schema_version") != "qwen25-correctness-v2-frozen-inputs-v1":
        raise RuntimeError("wrong v2 frozen input schema")
    files = frozen.get("files")
    if not isinstance(files, dict) or frozen.get("file_count") != len(files):
        raise RuntimeError("malformed v2 frozen input manifest")
    required = {"scripts/86_validate_qwen25_v2_rerun.py",
                "scripts/87_eval_qwen25_v2_smoke_rerun.py",
                "runs/qwen25_correctness_v2_20260928T081111Z/validation_manifest.json",
                "runs/qwen25_correctness_v2_20260928T081111Z/smoke_manifest.json"}
    if not required.issubset(files):
        raise RuntimeError("v2 frozen input manifest lacks required files")
    for relative, expected in files.items():
        source = (ROOT / relative).resolve()
        if not source.is_relative_to(ROOT) or not source.is_file():
            raise RuntimeError(f"frozen source missing or escaped workspace: {relative}")
        if not isinstance(expected, dict) or source.stat().st_size != expected.get("bytes") or \
           sha256_file(source) != expected.get("sha256"):
            raise RuntimeError(f"v2 frozen source changed: {relative}")
    return sha256_file(path)


def bind_validation(path: Path, fixed: dict, smoke: dict) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "qwen25-gpu-correctness-v2":
        raise RuntimeError("v2 smoke requires v2 validation schema")
    if (value.get("status") != "PASS" or
            value.get("gpu_system_correctness") != "PASS" or
            value.get("pilot_eligible") is not True):
        raise RuntimeError("v2 GPU system correctness is not PASS")
    gates = value.get("gates")
    if not isinstance(gates, dict) or set(gates) != set(GATES):
        raise RuntimeError("v2 validation does not have exactly G1–G15")
    if any(not isinstance(gates[name], dict) or
           gates[name].get("status") != "PASS" for name in GATES):
        raise RuntimeError("at least one v2 gate is not PASS")
    if value.get("manifest_sha256") != fixed["manifest_sha256"]:
        raise RuntimeError("v2 validator used a different fixed 10-pair manifest")
    frozen_sha256 = verify_frozen_inputs()
    if value.get("frozen_inputs_sha256") != frozen_sha256:
        raise RuntimeError("v2 validation is not bound to current frozen inputs")
    source_hashes = value.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise RuntimeError("v2 validator lacks source hashes")
    for relative in ("data/index.json", "mmimpress/qwen25/runner.py",
                     "mmimpress/qwen25/store.py", "mmimpress/qwen25/vision.py"):
        expected = fixed["source_hashes"][relative]
        if source_hashes.get(relative) != expected or sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"v2 validator source hash mismatch: {relative}")
        if smoke["source_hashes"][relative] != expected:
            raise RuntimeError(f"smoke source hash mismatch: {relative}")
    for relative, expected in source_hashes.items():
        source = (ROOT / relative).resolve()
        if not source.is_relative_to(ROOT) or not source.is_file() or sha256_file(source) != expected:
            raise RuntimeError(f"v2 validation source changed: {relative}")
    configuration = value.get("configuration")
    if not isinstance(configuration, dict):
        raise RuntimeError("v2 validator configuration is missing")
    for name in ("model_id", "checkpoint_revision", "seed", "attention_backend",
                 "min_pixels", "max_pixels", "max_new_tokens", "chunk_size",
                 "budget_ratio"):
        if name not in configuration or configuration[name] != smoke["configuration"][name]:
            raise RuntimeError(f"v2 validator config missing or mismatched: {name}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path),
            "schema_version": value["schema_version"], "status": "PASS",
            "gpu_system_correctness": "PASS", "pilot_eligible": True,
            "manifest_sha256": fixed["manifest_sha256"],
            "frozen_inputs_sha256": frozen_sha256,
            "gate_statuses": {name: "PASS" for name in GATES}}


def gpu_inventory() -> dict:
    command = ["nvidia-smi", "--id=0",
               "--query-compute-apps=pid,process_name,used_gpu_memory",
               "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=15,
                            check=False)
    if result.returncode:
        raise RuntimeError(f"GPU process inventory failed: {result.stderr.strip()}")
    processes = []
    for cells in csv.reader(io.StringIO(result.stdout)):
        if not cells or not any(cell.strip() for cell in cells):
            continue
        if len(cells) != 3:
            raise RuntimeError(f"unrecognized GPU process row: {cells}")
        processes.append({"pid": int(cells[0].strip()),
                          "process_name": cells[1].strip(),
                          "used_gpu_memory_mib": float(cells[2].strip())})
    foreign = [row for row in processes if row["pid"] != os.getpid()]
    if foreign:
        raise RuntimeError(f"concurrent GPU process: {foreign}")
    return {"self_pid": os.getpid(), "processes": processes,
            "foreign_processes": foreign}


def grouped_images(smoke: dict) -> list[dict]:
    samples = smoke["samples"]
    images = []
    for image_no in range(20):
        triple = samples[image_no * 3:(image_no + 1) * 3]
        if len(triple) != 3 or len({row["image_id"] for row in triple}) != 1:
            raise RuntimeError("smoke does not have three questions per image")
        if len({row["image_sha256"] for row in triple}) != 1:
            raise RuntimeError("one image has inconsistent hashes")
        offset = (image_no + 20 + int(smoke["configuration"]["seed"])) % 3
        images.append({"image_id": triple[0]["image_id"],
                       "image_path": triple[0]["image_path"],
                       "image_sha256": triple[0]["image_sha256"],
                       "method_order": list(METHODS[offset:] + METHODS[:offset]),
                       "turns": [{"turn_id": j + 1, **row}
                                 for j, row in enumerate(triple)]})
    if len({row["image_id"] for row in images}) != 20:
        raise RuntimeError("smoke image IDs are not distinct")
    return images


def logit_metadata(result: dict) -> dict:
    import torch
    logits = result.pop("first_logits", None)
    if not isinstance(logits, torch.Tensor) or logits.numel() < 1:
        raise RuntimeError("missing first-token logits")
    logits = logits.detach().to("cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("nonfinite first-token logits")
    return {"shape": list(logits.shape), "dtype": "float32",
            "sha256": hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
            "min": float(logits.min()), "max": float(logits.max()),
            "finite": True}


def selected_metadata(runner, store_path: Path, image_hash: str,
                      result: dict) -> dict:
    store = runner._activate(store_path, image_hash)
    meta = store.meta
    chunks = int(result["selected_chunks"])
    original = [int(i) for i in meta["stored_to_original"][:chunks * int(meta["chunk_size"])]
                if 0 <= int(i) < int(meta["visual_count"])]
    selected = sorted(original)
    if len(selected) != len(set(selected)) or len(selected) != int(result["kept_tokens"]):
        raise RuntimeError("selected visual IDs do not match cache result")
    return {"selected_chunk_ids": list(range(chunks)),
            "selected_original_visual_token_ids": selected,
            "selected_original_prompt_token_ids": [int(meta["visual_start"]) + i
                                                   for i in selected],
            "prefix_len": int(meta["prefix_len"]),
            "visual_count": int(meta["visual_count"]),
            "chunk_size": int(meta["chunk_size"]),
            "store_permutation_sha256": meta["permutation_sha256"]}


def summarize(rows: list[dict]) -> dict:
    duplicate_request_ids = len(rows) - len({row["request_id"] for row in rows})
    nonfinite_logits = sum(row.get("first_logits", {}).get("finite") is not True
                           for row in rows)
    isolation_failures = sum(row.get("rope_deltas_cleared") is not True or
                             row.get("history") != [] for row in rows)
    if len(rows) != 180 or duplicate_request_ids or nonfinite_logits or isolation_failures:
        raise RuntimeError("smoke has incomplete, duplicate, nonfinite, or isolated-state failures")
    by_question: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_question[row["question_id"]][row["method"]] = row
    if len(by_question) != 60 or any(set(group) != set(METHODS)
                                    for group in by_question.values()):
        raise RuntimeError("smoke lacks three-way question coverage")
    scope = {}
    for scope_name, subset in (("all", rows),
                               ("cache_hit_questions", [row for row in rows
                                                        if row["turn_id"] > 1])):
        methods = {}
        for method in METHODS:
            chosen = [row for row in subset if row["method"] == method]
            methods[method] = {
                "requests": len(chosen),
                "accuracy": statistics.fmean(row["correct"] for row in chosen),
                "incorrect_count": sum(row["correct"] == 0 for row in chosen),
                "truncated_count": sum(bool(row["truncated"]) for row in chosen),
                "empty_prediction_count": sum(not row["prediction"].strip() for row in chosen),
                "generated_token_mean": statistics.fmean(row["generated_token_count"]
                                                          for row in chosen),
                "ttft_mean_ms": statistics.fmean(row["ttft_ms"] for row in chosen),
                "visual_read_bytes_mean": statistics.fmean(row["visual_read_bytes"]
                                                           for row in chosen),
                "structural_read_bytes_mean": statistics.fmean(row["structural_read_bytes"]
                                                               for row in chosen),
                "actual_kept_ratio_mean": (statistics.fmean(row["actual_kept_ratio"]
                                                            for row in chosen)
                                           if method != "recompute" and scope_name != "all"
                                           else None),
            }
        questions = {row["question_id"] for row in subset}
        agreements = {}
        for left in ("fullload", "ours25"):
            for right in ("recompute", "fullload"):
                if left == right:
                    continue
                key = f"{left}_vs_{right}"
                agreements[key] = {
                    "first_token": sum(by_question[q][left]["first_token_id"] ==
                                       by_question[q][right]["first_token_id"]
                                       for q in questions),
                    "generated_sequence": sum(by_question[q][left]["generated_token_ids"] ==
                                              by_question[q][right]["generated_token_ids"]
                                              for q in questions),
                    "prediction": sum(by_question[q][left]["prediction"] ==
                                      by_question[q][right]["prediction"]
                                      for q in questions),
                    "denominator": len(questions),
                }
        scope[scope_name] = {"methods": methods, "agreements": agreements}
    return {"schema_version": SCHEMA, "status": "PASS", "images": 20,
            "questions": 60, "requests": 180,
            "cache_hit_questions": 40, "scopes": scope,
            "request_failures": 0, "duplicate_request_ids": duplicate_request_ids,
            "nonfinite_first_logits": nonfinite_logits,
            "request_isolation_failures": isolation_failures,
            "truncated_requests": sum(bool(row["truncated"]) for row in rows),
            "incorrect_requests": sum(row["correct"] == 0 for row in rows),
            "interpretation": "small descriptive semantic smoke, disjoint from v2 validation images"}


def run(smoke: dict, validation_binding: dict, out_dir: Path) -> None:
    import torch
    from PIL import Image
    from mmimpress.qwen25.runner import Qwen25Runner, SEED

    if out_dir.exists():
        raise FileExistsError(out_dir)
    if not out_dir.resolve().is_relative_to(RUN_ROOT.resolve()):
        raise RuntimeError("smoke output must be a new directory under the v2 run")
    images = grouped_images(smoke)
    out_dir.mkdir(parents=True, exist_ok=False)
    save_json_new(out_dir / "binding.json", {
        "schema_version": SCHEMA, "validation": validation_binding,
        "smoke_manifest_path": str(SMOKE_MANIFEST),
        "smoke_manifest_sha256": smoke["manifest_sha256"],
        "smoke_manifest_file_sha256": sha256_file(SMOKE_MANIFEST),
        "script_sha256": sha256_file(Path(__file__)),
        "method_rotation": "(source_index_image_row + seed) % 3",
        "timing_policy": "dataset image file read/RGB decode and SSD conditioning before request timer",
        "history_policy": "none; all GQA questions independent",
        "first_logits_policy": "finite check and SHA256 only; full vector not serialized",
    })
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    status = {"schema_version": SCHEMA, "status": "RUNNING", "raw_requests": 0,
              "persistence_stores": 0, "started_unix": time.time(),
              "validation": validation_binding}
    checkpoint(out_dir / "status.json", status)
    try:
        runner = Qwen25Runner().load()
        save_json_new(out_dir / "runtime.json", runner.runtime_fingerprint())
        warmup = runner.run_pixels(Image.new("RGB", (448, 448), (127, 127, 127)),
                                   "Describe the image briefly.", capture=False)
        if warmup.get("vision_calls") != 1:
            raise RuntimeError("common warmup failed")
    except BaseException as exc:
        status.update({"status": "FAIL", "finished_unix": time.time(),
                       "error": f"{type(exc).__name__}: {exc}"})
        checkpoint(out_dir / "status.json", status)
        raise
    rows = []
    try:
        with (out_dir / "raw.jsonl").open("x", encoding="utf-8") as raw, \
             (out_dir / "persistence.jsonl").open("x", encoding="utf-8") as persistence, \
             (out_dir / "gpu_inventory.jsonl").open("x", encoding="utf-8") as inventory:
            append_jsonl(inventory, {"phase": "smoke_start", **gpu_inventory()})
            for image in images:
                image_id = image["image_id"]
                append_jsonl(inventory, {"phase": "image_start", "image_id": image_id,
                                         **gpu_inventory()})
                image_path = Path(image["image_path"])
                if sha256_file(image_path) != image["image_sha256"]:
                    raise RuntimeError(f"image bytes changed: {image_id}")
                with Image.open(image_path) as src:
                    pixels = src.convert("RGB")
                stores = {method: out_dir / "stores" / image_id / method
                          for method in ("fullload", "ours25")}
                for turn in image["turns"]:
                    turn_id = turn["turn_id"]
                    for method in image["method_order"]:
                        if runner.model.model.rope_deltas is not None:
                            raise RuntimeError("stale rope_deltas before independent request")
                        conditioning = None
                        if turn_id == 1 or method == "recompute":
                            capture_mode = ("kv_only" if turn_id == 1 and method == "fullload"
                                            else "with_score" if turn_id == 1 and method == "ours25"
                                            else False)
                            result = runner.run_pixels(
                                pixels, turn["question"], history=(),
                                capture=capture_mode, image_sha256=image["image_sha256"],
                                return_logits=True)
                            captured = result.pop("capture", None)
                            if turn_id == 1 and method != "recompute":
                                if captured is None:
                                    raise RuntimeError("T1 cache capture missing")
                                store = stores[method]
                                store.parent.mkdir(parents=True, exist_ok=True)
                                if store.exists():
                                    raise FileExistsError(store)
                                persisted = runner.persist(captured, store,
                                    layout=("canonical" if method == "fullload"
                                            else "repacked"))
                                append_jsonl(persistence, {"image_id": image_id,
                                    "image_sha256": image["image_sha256"],
                                    "method": method, "source_turn_id": 1,
                                    "store_dir": str(store), "persistence": persisted})
                                status["persistence_stores"] += 1
                                del captured
                        else:
                            store = stores[method]
                            budget = 1.0 if method == "fullload" else 0.25
                            conditioning = runner.condition_cache(
                                store, budget, image_sha256=image["image_sha256"])
                            result = runner.run_cache(
                                store, turn["question"], history=(),
                                budget_ratio=budget, image_sha256=image["image_sha256"],
                                return_logits=True)
                        logit = logit_metadata(result)
                        if runner.model.model.rope_deltas is not None:
                            raise RuntimeError("stale rope_deltas after independent request")
                        expected_vision = 1 if turn_id == 1 or method == "recompute" else 0
                        if result["vision_calls"] != expected_vision:
                            raise RuntimeError("unexpected vision forward count")
                        if result["online_query_score_calls"] != 0:
                            raise RuntimeError("online query score call on smoke path")
                        generated = result["generated_token_ids"]
                        if (not generated or generated[0] != result["first_token_id"] or
                                len(generated) != result["generated_token_count"]):
                            raise RuntimeError("invalid generated token sequence")
                        selected = (selected_metadata(runner, stores[method],
                                      image["image_sha256"], result)
                                    if turn_id > 1 and method != "recompute" else None)
                        record = {"schema_version": SCHEMA,
                            "request_id": f"gqa:{image_id}:{turn['question_id']}:{method}",
                            "image_id": image_id,
                            "image_sha256": image["image_sha256"],
                            "turn_id": turn_id,
                            "question_id": turn["question_id"],
                            "question": turn["question"], "gold": turn["gold"],
                            "method": method, "method_order": image["method_order"],
                            "request_path": ("normal_pixels" if expected_vision else "ssd_cache_hit"),
                            "history": [], "rope_deltas_cleared": True,
                            "prediction": result["prediction"],
                            "correct": float(exact_score(result["prediction"], [turn["gold"]])),
                            "first_token_id": result["first_token_id"],
                            "generated_token_ids": generated,
                            "generated_token_count": len(generated),
                            "truncated": result["truncated"],
                            "first_logits": logit,
                            "ttft_ms": float(result["ttft_ms"]),
                            "request_e2e_ms": float(result["request_e2e_ms"]),
                            "vision_calls": result["vision_calls"],
                            "online_query_score_calls": result["online_query_score_calls"],
                            "visual_read_bytes": int(result.get("visual_read_bytes", 0)),
                            "structural_read_bytes": int(result.get("structural_read_bytes", 0)),
                            "metadata_read_bytes": int(result.get("metadata_read_bytes", 0)),
                            "actual_kept_ratio": result.get("actual_kept_ratio"),
                            "selected": selected,
                            "conditioning": conditioning, "result": result}
                        if not math.isfinite(record["ttft_ms"]) or record["ttft_ms"] < 0:
                            raise RuntimeError("invalid TTFT")
                        if (not math.isfinite(record["request_e2e_ms"]) or
                                record["request_e2e_ms"] + 1e-3 < record["ttft_ms"]):
                            raise RuntimeError("invalid request duration")
                        append_jsonl(raw, record)
                        rows.append(record)
                        status["raw_requests"] += 1
                        status["last_request_id"] = record["request_id"]
                        checkpoint(out_dir / "status.json", status)
                runner.close()
                append_jsonl(inventory, {"phase": "image_end", "image_id": image_id,
                                         **gpu_inventory()})
            append_jsonl(inventory, {"phase": "smoke_end", **gpu_inventory()})
        if status["persistence_stores"] != 40:
            raise RuntimeError("expected exactly 40 new SSD stores")
        summary = summarize(rows)
        summary["validation"] = validation_binding
        summary["smoke_manifest_sha256"] = smoke["manifest_sha256"]
        save_json_new(out_dir / "summary.json", summary)
        status.update({"status": "PASS", "finished_unix": time.time(),
                       "summary_sha256": sha256_file(out_dir / "summary.json")})
        checkpoint(out_dir / "status.json", status)
        print(json.dumps({"status": "PASS", "out_dir": str(out_dir),
                          "requests": 180, "questions": 60, "images": 20}))
    except BaseException as exc:
        status.update({"status": "FAIL", "finished_unix": time.time(),
                       "error": f"{type(exc).__name__}: {exc}"})
        checkpoint(out_dir / "status.json", status)
        raise
    finally:
        runner.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path,
                        help="new v2 GPU validation.json; required for execution")
    parser.add_argument("--out-dir", type=Path,
                        help="fresh directory under the frozen v2 run")
    parser.add_argument("--check-manifest", action="store_true",
                        help="CPU-only verification of both frozen manifests")
    args = parser.parse_args()
    fixed = load_fixed_manifest(VALIDATION_MANIFEST, expected_count=10)
    smoke = load_fixed_manifest(SMOKE_MANIFEST, expected_count=60)
    if {row["image_id"] for row in fixed["samples"]} & \
       {row["image_id"] for row in smoke["samples"]}:
        raise RuntimeError("smoke and validation images overlap")
    if args.check_manifest:
        print(json.dumps({"validation_samples": 10, "smoke_images": 20,
                          "smoke_questions": 60,
                          "validation_manifest_sha256": fixed["manifest_sha256"],
                          "smoke_manifest_sha256": smoke["manifest_sha256"]}))
        return 0
    if args.validation is None or args.out_dir is None:
        parser.error("--validation and --out-dir are required for GPU smoke")
    binding = bind_validation(args.validation, fixed, smoke)
    run(smoke, binding, args.out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

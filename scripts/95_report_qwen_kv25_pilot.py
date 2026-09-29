#!/usr/bin/env python3
"""Independently audit and report the Qwen KV25 four-arm raw pilot.

This does not import the pilot runner or reuse its arithmetic. It checks each
physical request against the frozen workload and immutable store inventory,
then recomputes quality, retention, actual I/O and image-cluster intervals.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mmimpress.dataset import exact_score

MIGRATION = ROOT / "runs/qwen25_kv25_migration_20260929T042949Z"
METHODS = ("recompute", "fullload", "qwen_ours_chunk25_legacy", "qwen_ours_kv25")
SCHEMA = "qwen25-kv25-four-arm-pilot-v1"
DATASETS = ("smoke", "gqa", "mt")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 1234


def require(ok, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"not a JSON object: {path}")
    return value


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        f.write("\n")


def score(dataset: str, prediction: str, gold: str) -> float:
    if dataset != "mt":
        return float(exact_score(prediction, [gold]))
    import re
    def normalize(x):
        words = re.sub(r"[^\w\s]", " ", str(x).lower()).split()
        return " ".join(w for w in words if w not in {"a", "an", "the"})
    return float(normalize(prediction) == normalize(gold))


def mean(values):
    return statistics.fmean(values) if values else None


def mb(value):
    return value / 1_000_000 if value is not None else None


def _load_raw(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"malformed raw JSONL {path}:{lineno}: {exc}") from exc
    return rows


def audit_phase(dataset: str, directory: Path, gate_sha: str) -> dict:
    manifest_path, config_path = directory / "manifest.json", directory / "config.json"
    inventory_path, raw_path = directory / "store_inventory.json", directory / "raw.jsonl"
    manifest, config, inventory = map(read_json, (manifest_path, config_path, inventory_path))
    final = read_json(directory / "final_status.json")
    require(manifest.get("schema_version") == SCHEMA and
            manifest.get("phase") == dataset and
            config.get("schema_version") == SCHEMA, f"{dataset}: wrong schema")
    unsigned = dict(manifest)
    claimed = unsigned.pop("manifest_sha256")
    require(canonical_hash(unsigned) == claimed == config.get("manifest_sha256"),
            f"{dataset}: manifest hash mismatch")
    require(config.get("gpu_gate", {}).get("sha256") == gate_sha and
            manifest.get("gpu_gate", {}).get("sha256") == gate_sha,
            f"{dataset}: GPU gate binding mismatch")
    require(manifest.get("persistence_status") == "NOT_REMEASURED" and
            config.get("persistence_status") == "NOT_REMEASURED" and
            config.get("execution_scope") == "CACHE-HIT REEVALUATION" and
            inventory.get("store_mode") == "read_only", f"{dataset}: persistence policy mismatch")
    for relative, digest in config.get("source_hashes", {}).items():
        require(sha256(ROOT / relative) == digest, f"{dataset}: source changed: {relative}")
    require("scripts/94_eval_qwen_kv25_pilot.py" in config.get("source_hashes", {}),
            f"{dataset}: pilot script hash absent")
    require(sha256(MIGRATION / "geometry_reference.json") ==
            inventory.get("geometry_reference_sha256"),
            f"{dataset}: geometry reference changed")
    rows = _load_raw(raw_path)
    expected = len(manifest["images"]) * len(manifest["images"][0]["turns"]) * len(METHODS)
    require(final.get("status") == "PASS" and final.get("actual_requests") == expected
            and final.get("expected_requests") == expected,
            f"{dataset}: final pilot status/coverage failed")
    require(len(rows) == expected, f"{dataset}: raw requests {len(rows)} != {expected}")
    require(all(row.get("status") == "PASS" and row.get("attempt") == 1 for row in rows),
            f"{dataset}: failed/retried request exists")
    physical_ids = [row.get("physical_execution_id") for row in rows]
    require(all(isinstance(x, str) and len(x) == 32 for x in physical_ids)
            and len(set(physical_ids)) == len(physical_ids),
            f"{dataset}: physical execution ID missing or duplicated")
    expected_keys = {(str(im["image_id"]), int(t["turn_id"]), method)
                     for im in manifest["images"] for t in im["turns"] for method in METHODS}
    actual_keys = [(str(row["image_id"]), int(row["turn_id"]), row["method_id"])
                   for row in rows]
    require(set(actual_keys) == expected_keys and len(set(actual_keys)) == len(actual_keys),
            f"{dataset}: duplicate or missing logical request")
    image_by_id = {str(im["image_id"]): im for im in manifest["images"]}
    histories = {(iid, method): [] for iid in image_by_id for method in METHODS}
    expected_order = []
    for im in manifest["images"]:
        iid = str(im["image_id"])
        for turn in im["turns"]:
            for method in im["method_order"]:
                expected_order.append((iid, int(turn["turn_id"]), method))
    require(actual_keys == expected_order, f"{dataset}: execution order changed")
    per_method = defaultdict(list)
    by_key = {}
    for row in rows:
        iid, tid, method = str(row["image_id"]), int(row["turn_id"]), row["method_id"]
        im = image_by_id[iid]
        turn = next(t for t in im["turns"] if int(t["turn_id"]) == tid)
        geo = inventory["stores"][iid]
        result = row["result"]
        require(row["schema_version"] == SCHEMA and row["dataset"] == dataset and
                row["manifest_sha256"] == claimed and row["config_sha256"] == sha256(config_path),
                f"{dataset}/{iid}/{tid}/{method}: provenance mismatch")
        require(row["model_id"] == config["model_id"] and
                row["model_revision"] == config["model_revision"] and
                row["image_sha256"] == im["image_sha256"] and
                row["question_id"] == str(turn["question_id"]) and
                row["question"] == turn["question"] and row["gold"] == turn["gold"],
                f"{dataset}/{iid}/{tid}/{method}: workload changed")
        require(row["method_order"] == im["method_order"] and
                row["method_order_position"] == im["method_order"].index(method),
                f"{dataset}/{iid}/{tid}/{method}: method rotation changed")
        expected_history = histories[iid, method] if dataset == "mt" else []
        require(row["history"] == expected_history and
                row["history_sha256"] == canonical_hash(expected_history),
                f"{dataset}/{iid}/{tid}/{method}: method-local history mismatch")
        require(row["N_content"] == geo["N_content"] and
                row["S_structural"] == geo["S_structural"] and
                row["k_target"] == (geo["N_content"] + 3) // 4 and
                row["legacy_kept_count"] == geo["legacy_kept"] and
                row["permutation_sha256"] == geo["permutation_sha256"],
                f"{dataset}/{iid}/{tid}/{method}: geometry mismatch")
        require(row["correct"] == score(dataset, row["prediction"], row["gold"]),
                f"{dataset}/{iid}/{tid}/{method}: scorer mismatch")
        require(row["prediction"] == result["prediction"] and
                row["first_token_id"] == result["first_token_id"] and
                row["generated_token_ids"] == result["generated_token_ids"] and
                row["generated_token_count"] == len(row["generated_token_ids"]) and
                row["first_token_id"] == row["generated_token_ids"][0] and
                row["generated_token_count"] <= 16 and
                row["ttft_ms"] == result["ttft_ms"] and
                row["request_e2e_ms"] == result["request_e2e_ms"] and
                0 <= row["ttft_ms"] <= row["request_e2e_ms"] + 1e-3,
                f"{dataset}/{iid}/{tid}/{method}: generation/timing mismatch")
        require(result.get("first_logits_finite") is True and
                len(result.get("first_logits_sha256", "")) == 64,
                f"{dataset}/{iid}/{tid}/{method}: first logits evidence missing")
        require(row["vision_calls"] == (1 if tid == 1 or method == "recompute" else 0)
                and row["online_query_score_calls"] == 0,
                f"{dataset}/{iid}/{tid}/{method}: vision/online-score counter mismatch")
        if tid == 1:
            require(row["request_path"] == "normal_pixels" and
                    row["selected_original_ids"] is None and
                    row["total_actual_pread_bytes"] is None and
                    row["image_file_decode_ms"] == result["image_file_decode_ms"] ==
                    result["timing_ms"]["image_file_decode"] and
                    row["image_file_decode_ms"] >= 0 and
                    result["t1_diagnostic_capture_discarded"] == (method != "recompute"),
                    f"{dataset}/{iid}/{tid}/{method}: T1 path/capture mismatch")
            require(result["geometry"]["visual_count"] == geo["N_content"] and
                    result["geometry"]["prefix_len"] == geo["prefix_len"],
                    f"{dataset}/{iid}/{tid}/{method}: T1 geometry mismatch")
        elif method == "recompute":
            require(row["request_path"] == "normal_pixels" and
                    row["total_actual_pread_bytes"] is None and
                    row["image_file_decode_ms"] == result["image_file_decode_ms"] ==
                    result["timing_ms"]["image_file_decode"] and
                    row["image_file_decode_ms"] >= 0,
                    f"{dataset}/{iid}/{tid}/{method}: ReComp hit path mismatch")
        else:
            unit = "visual_kv" if method == "qwen_ours_kv25" else "chunk"
            selected = (geo["N_content"] if method == "fullload" else
                        geo["legacy_kept"] if method == "qwen_ours_chunk25_legacy"
                        else geo["k_target"])
            chunks = ((geo["N_content"] + 63) // 64 if method == "fullload" else
                      geo["legacy_chunks"] if method == "qwen_ours_chunk25_legacy"
                      else geo["kv25_chunks"])
            ids = (list(range(selected)) if method == "fullload" else
                   geo["selected_original_legacy"] if unit == "chunk" else
                   geo["selected_original_kv25"])
            source = geo["fullload_store"] if method == "fullload" else geo["legacy_store"]
            require(row["request_path"] == "read_only_ssd_cache_hit" and
                    row["image_file_decode_ms"] is None and
                    row["source_store"] == source and row["store_persistence"] == "NOT_REMEASURED" and
                    row["budget_unit"] == ("full" if method == "fullload" else unit) and
                    result["budget_unit"] == unit and row["selected_original_ids"] == ids and
                    result["selected_visual_original"] == ids and
                    result["selected_visual_stored"] == list(range(selected)) and
                    row["normal_chunks_read"] == chunks and
                    row["compact_content_rows"] == selected and
                    row["compact_total_rows"] == selected + geo["S_structural"] and
                    result["dense_reference"] is False,
                    f"{dataset}/{iid}/{tid}/{method}: selected cache mismatch")
            loaded_valid = min(chunks * 64, geo["N_content"])
            padding_rows = chunks * 64 - loaded_valid
            unit_bytes = geo["row_bytes"] * 2 * geo["num_layers"]
            normal_bytes = chunks * 64 * unit_bytes
            structural_bytes = geo["S_structural"] * unit_bytes
            io = result["read_io"]
            expected_spans = [{"kind": "structural", "source": "structural_kv.bin",
                               "offset": 0, "requested_bytes": structural_bytes}]
            expected_spans.extend(
                {"kind": "visual", "source": f"layer_{layer:03d}/{kv}.bin",
                 "offset": 0, "requested_bytes": chunks * 64 * geo["row_bytes"]}
                for layer in range(geo["num_layers"]) for kv in ("k", "v"))
            require(row["loaded_valid_visual_rows"] == loaded_valid and
                    row["extra_valid_visual_rows"] == loaded_valid - selected and
                    row["padding_rows_read"] == padding_rows and
                    row["padding_read_bytes"] == padding_rows * unit_bytes and
                    row["normal_visual_read_bytes"] == normal_bytes and
                    row["structural_read_bytes"] == structural_bytes and
                    row["metadata_read_bytes"] == 0 and
                    row["total_actual_pread_bytes"] == normal_bytes + structural_bytes and
                    row["actual_read_spans"] == len(expected_spans) == io["spans"] and
                    io["span_details"] == expected_spans and
                    row["actual_pread_calls"] == io["preads"] >= io["spans"] and
                    io["bytes"] == row["total_actual_pread_bytes"] and
                    io["per_kind"]["visual"]["bytes"] == normal_bytes and
                    io["per_kind"]["structural"]["bytes"] == structural_bytes and
                    io["per_kind"]["visual"]["preads"] >= 2 * geo["num_layers"] and
                    io["per_kind"]["structural"]["preads"] >= 1,
                    f"{dataset}/{iid}/{tid}/{method}: actual I/O mismatch")
            require(row["h2d_kv_bytes"] == (selected + geo["S_structural"]) * unit_bytes and
                    row["gpu_cache_kv_bytes"] == row["h2d_kv_bytes"] and
                    row["logical_content_retention"] == selected / geo["N_content"] and
                    row["structural_inclusive_retention"] ==
                    (selected + geo["S_structural"]) / geo["prefix_len"],
                    f"{dataset}/{iid}/{tid}/{method}: compact bytes/retention mismatch")
            require(row["conditioning"] is not None and
                    row["conditioning"]["budget_unit"] == unit and
                    row["conditioning"]["selected_chunks"] == chunks and
                    row["conditioning"]["target_visual_tokens"] == selected and
                    row["activation_ms_outside_timer"] is not None,
                    f"{dataset}/{iid}/{tid}/{method}: cache conditioning absent")
            if method == "qwen_ours_kv25":
                require(row["selected_stored_ids"] == list(range(selected)) and
                        selected == (geo["N_content"] + 3) // 4,
                        f"{dataset}/{iid}/{tid}: KV25 exact target mismatch")
        if dataset == "mt":
            histories[iid, method] = [*expected_history, {"question_id": str(turn["question_id"]),
                "question": turn["question"], "prediction": row["prediction"]}]
        per_method[method].append(row)
        by_key[iid, tid, method] = row
    summary = {}
    for method in METHODS:
        all_rows = per_method[method]
        hit_rows = [r for r in all_rows if int(r["turn_id"]) > 1]
        t1 = [r for r in all_rows if int(r["turn_id"]) == 1]
        unique = {str(r["image_id"]): r for r in hit_rows}
        logical_macro = mean([r["logical_content_retention"] for r in unique.values()]) if method != "recompute" else None
        logical_weighted = (sum(r["compact_content_rows"] for r in unique.values()) /
                            sum(r["N_content"] for r in unique.values())) if method != "recompute" else None
        structural_macro = mean([r["structural_inclusive_retention"] for r in unique.values()]) if method != "recompute" else None
        summary[method] = {
            "requests": len(all_rows), "t1_requests": len(t1), "hit_requests": len(hit_rows),
            "all_quality": mean([r["correct"] for r in all_rows]),
            "hit_quality": mean([r["correct"] for r in hit_rows]),
            "quality_by_turn": {str(t): mean([r["correct"] for r in all_rows if r["turn_id"] == t])
                                for t in sorted({r["turn_id"] for r in all_rows})},
            "t1_ttft_ms": mean([r["ttft_ms"] for r in t1]),
            "t1_e2e_ms": mean([r["request_e2e_ms"] for r in t1]),
            "hit_ttft_ms": mean([r["ttft_ms"] for r in hit_rows]),
            "hit_e2e_ms": mean([r["request_e2e_ms"] for r in hit_rows]),
            "logical_content_retention_macro": logical_macro,
            "logical_content_retention_weighted": logical_weighted,
            "structural_inclusive_retention_macro": structural_macro,
            "normal_read_mb_per_hit": mb(mean([r["normal_visual_read_bytes"] for r in hit_rows
                                                if r["normal_visual_read_bytes"] is not None])),
            "structural_read_mb_per_hit": mb(mean([r["structural_read_bytes"] for r in hit_rows
                                                   if r["structural_read_bytes"] is not None])),
            "metadata_read_mb_per_hit": mb(mean([r["metadata_read_bytes"] for r in hit_rows
                                                 if r["metadata_read_bytes"] is not None])),
            "total_read_mb_per_hit": mb(mean([r["total_actual_pread_bytes"] for r in hit_rows
                                              if r["total_actual_pread_bytes"] is not None])),
            "valid_content_read_mb_per_hit": mb(mean([
                r["loaded_valid_visual_rows"] * inventory["stores"][str(r["image_id"])]["row_bytes"]
                * 2 * inventory["stores"][str(r["image_id"])]["num_layers"]
                for r in hit_rows if r["loaded_valid_visual_rows"] is not None])),
            "retained_content_kv_mb_per_hit": mb(mean([
                r["compact_content_rows"] * inventory["stores"][str(r["image_id"])]["row_bytes"]
                * 2 * inventory["stores"][str(r["image_id"])]["num_layers"]
                for r in hit_rows if r["compact_content_rows"] is not None])),
            "extra_valid_mb_per_hit": mb(mean([
                r["extra_valid_visual_rows"] * inventory["stores"][str(r["image_id"])]["row_bytes"]
                * 2 * inventory["stores"][str(r["image_id"])]["num_layers"]
                for r in hit_rows if r["extra_valid_visual_rows"] is not None])),
            "padding_read_mb_per_hit": mb(mean([r["padding_read_bytes"] for r in hit_rows
                                                if r["padding_read_bytes"] is not None])),
            "h2d_mb_per_hit": mb(mean([r["h2d_kv_bytes"] for r in hit_rows
                                   if r["h2d_kv_bytes"] is not None])),
            "gpu_compact_cache_mb_per_hit": mb(mean([r["gpu_cache_kv_bytes"] for r in hit_rows
                                                     if r["gpu_cache_kv_bytes"] is not None])),
            "peak_gpu_allocated_mb_per_hit": mb(mean([r["peak_gpu_allocated_bytes"] for r in hit_rows])),
            "pread_calls_per_hit": mean([r["actual_pread_calls"] for r in hit_rows
                                          if r["actual_pread_calls"] is not None]),
            "normal_chunks_per_hit": mean([r["normal_chunks_read"] for r in hit_rows
                                            if r["normal_chunks_read"] is not None]),
        }
    full = summary["fullload"]
    for method in METHODS[1:]:
        summary[method]["normal_read_vs_fullload"] = (summary[method]["normal_read_mb_per_hit"] /
                                                       full["normal_read_mb_per_hit"])
        summary[method]["total_read_vs_fullload"] = (summary[method]["total_read_mb_per_hit"] /
                                                      full["total_read_mb_per_hit"])
    geometry = {"images": len(image_by_id), "selected_sets_equal": 0,
                "selected_sets_different": 0, "chunks_new_increase": 0,
                "chunks_new_same": 0, "chunks_new_decrease": 0,
                "unused_valid_rows_new_sum": 0, "unused_valid_bytes_new_sum": 0,
                "padding_rows_new_sum": 0, "padding_bytes_new_sum": 0}
    for iid, geo in inventory["stores"].items():
        geometry["selected_sets_equal" if geo["selected_original_legacy"] ==
                 geo["selected_original_kv25"] else "selected_sets_different"] += 1
        geometry["chunks_new_increase" if geo["kv25_chunks"] > geo["legacy_chunks"] else
                 "chunks_new_same" if geo["kv25_chunks"] == geo["legacy_chunks"] else
                 "chunks_new_decrease"] += 1
        unused = min(geo["kv25_chunks"] * 64, geo["N_content"]) - geo["k_target"]
        pad = geo["kv25_chunks"] * 64 - min(geo["kv25_chunks"] * 64, geo["N_content"])
        unit = geo["row_bytes"] * 2 * geo["num_layers"]
        geometry["unused_valid_rows_new_sum"] += unused
        geometry["unused_valid_bytes_new_sum"] += unused * unit
        geometry["padding_rows_new_sum"] += pad
        geometry["padding_bytes_new_sum"] += pad * unit
    ref = read_json(MIGRATION / "geometry_reference.json")["phases"]["gqa" if dataset == "smoke" else dataset]
    if dataset != "smoke":
        require(geometry["selected_sets_different"] == ref["different_selected_set"] and
                geometry["chunks_new_increase"] == ref["chunk_change_counts"].get("increase", 0) and
                geometry["chunks_new_same"] == ref["chunk_change_counts"].get("same", 0) and
                geometry["chunks_new_decrease"] == ref["chunk_change_counts"].get("decrease", 0),
                f"{dataset}: independent geometry reference disagrees")
    result = {"status": "PASS", "dataset": dataset, "run_dir": str(directory.resolve()),
              "manifest_sha256": sha256(manifest_path), "config_sha256": sha256(config_path),
              "store_inventory_sha256": sha256(inventory_path), "raw_sha256": sha256(raw_path),
              "expected_physical_requests": expected, "actual_physical_requests": len(rows),
              "failures": 0, "retries": 0, "duplicate_physical_ids": 0,
              "duplicate_logical_requests": 0, "summary": summary, "geometry": geometry,
              "rows": rows}
    return result


def percentile(sorted_values, p: float) -> float:
    index = (len(sorted_values) - 1) * p
    lo = int(math.floor(index))
    hi = int(math.ceil(index))
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (index - lo)


def pair_bootstrap(phase: dict) -> dict:
    rows = phase["rows"]
    paired = defaultdict(dict)
    for r in rows:
        if r["turn_id"] > 1 and r["method_id"] in METHODS[2:]:
            paired[str(r["image_id"]), int(r["turn_id"])][r["method_id"]] = r
    image_deltas = defaultdict(list)
    for (iid, _), arms in paired.items():
        require(set(arms) == set(METHODS[2:]), "old/new paired hit missing")
        old, new = arms[METHODS[2]], arms[METHODS[3]]
        image_deltas[iid].append({
            "quality_pp": 100 * (new["correct"] - old["correct"]),
            "ttft_ms": new["ttft_ms"] - old["ttft_ms"],
            "e2e_ms": new["request_e2e_ms"] - old["request_e2e_ms"],
            "ssd_bytes": new["total_actual_pread_bytes"] - old["total_actual_pread_bytes"],
            "normal_bytes": new["normal_visual_read_bytes"] - old["normal_visual_read_bytes"],
        })
    images = sorted(image_deltas)
    require(len(images) == phase["geometry"]["images"], "paired image coverage mismatch")
    keys = ("quality_pp", "ttft_ms", "e2e_ms", "ssd_bytes", "normal_bytes")
    per_image = {iid: {key: mean([x[key] for x in image_deltas[iid]]) for key in keys}
                 for iid in images}
    rng = random.Random(BOOTSTRAP_SEED)
    draws = {key: [] for key in keys}
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [images[rng.randrange(len(images))] for _ in images]
        for key in keys:
            draws[key].append(mean([per_image[iid][key] for iid in sample]))
    effects = {}
    for key in keys:
        values = sorted(draws[key])
        effects[key] = {"mean": mean([per_image[iid][key] for iid in images]),
                        "ci95": [percentile(values, .025), percentile(values, .975)]}
    old = phase["summary"][METHODS[2]]
    effects["ttft_reduction_fraction"] = (old["hit_ttft_ms"] -
        phase["summary"][METHODS[3]]["hit_ttft_ms"]) / old["hit_ttft_ms"]
    effects["ssd_bytes_change_fraction"] = (phase["summary"][METHODS[3]]["total_read_mb_per_hit"] /
        old["total_read_mb_per_hit"] - 1)
    return {"unit": "image cluster", "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED, "images": len(images),
            "sign": "KV25 new minus legacy Chunk25", "effects": effects}


def format_pct(x):
    return "—" if x is None else f"{100*x:.2f}%"


def format_ms(x):
    return "—" if x is None else f"{x:.2f}"


def format_mb(x):
    return "—" if x is None else f"{x:.3f}"


def render_report(phases: dict, gate: dict, audit: dict) -> str:
    lines = ["# Qwen2.5-VL Chunk25 → Visual-KV25 전환 파일럿", "",
        "동일 Qwen/Qwen2.5-VL-7B-Instruct, NF4/BF16/SDPA, 64-row chunk, r=0.25, "
        "seed=1234, 최대 16 token, frozen GQA/MT manifest에서 4-arm을 새로 실행했다. "
        "Old/New Ours hit는 동일한 보호된 v2 repacked BF16 store를 read-only로 사용했다. "
        "각 arm의 T1은 새 full-image inference이며 진단 capture는 저장하지 않았다. "
        "Persistence와 cold-start session은 이번에 재측정하지 않았다.", "",
        f"GPU correctness gate: `{gate['path']}` (`{gate['sha256']}`), PASS. "
        "이 구현 검증은 정확도 유지나 지연 감소의 보장을 뜻하지 않는다.", ""]
    for dataset in ("gqa", "mt"):
        phase = phases[dataset]
        summary = phase["summary"]
        paired = phase["paired"]["effects"]
        lines += [f"## {dataset.upper()} 4-arm", "",
            f"{phase['geometry']['images']} images, {phase['actual_physical_requests']} unique physical requests, "
            "0 실패/중복/retry. Read MB와 TTFT는 hit request당 평균이다.", "",
            "| 방법 | 전체 정답률 | Hit 정답률 | T1 TTFT (ms) | Hit TTFT (ms) | 실제 content retention (macro / weighted) | Normal / total read (MB/hit) |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        labels = {"recompute":"ReComp", "fullload":"FullLoad",
            "qwen_ours_chunk25_legacy":"Ours Chunk25 legacy",
            "qwen_ours_kv25":"Ours Visual-KV25"}
        for method in METHODS:
            s = summary[method]
            ret = (format_pct(s["logical_content_retention_macro"]) + " / " +
                   format_pct(s["logical_content_retention_weighted"]))
            reads = format_mb(s["normal_read_mb_per_hit"]) + " / " + format_mb(s["total_read_mb_per_hit"])
            lines.append(f"| {labels[method]} | {format_pct(s['all_quality'])} | "
                f"{format_pct(s['hit_quality'])} | {format_ms(s['t1_ttft_ms'])} | "
                f"{format_ms(s['hit_ttft_ms'])} | {ret} | {reads} |")
        q,t,b = paired["quality_pp"], paired["ttft_ms"], paired["ssd_bytes"]
        lines += ["", "| 비교 | Hit 정답률 차이 (%p) | Hit TTFT 차이 (ms)·감소율 | SSD bytes 변화 (MB/hit) | Paired 95% CI |",
            "|---|---:|---:|---:|---|",
            "| KV25 − Chunk25 | "
            f"{q['mean']:+.2f} | {t['mean']:+.2f} / {100*paired['ttft_reduction_fraction']:+.2f}% 감소 | "
            f"{b['mean']/1e6:+.3f} | quality [{q['ci95'][0]:+.2f}, {q['ci95'][1]:+.2f}] %p; "
            f"TTFT [{t['ci95'][0]:+.2f}, {t['ci95'][1]:+.2f}] ms; "
            f"SSD [{b['ci95'][0]/1e6:+.3f}, {b['ci95'][1]/1e6:+.3f}] MB |", ""]
        g = phase["geometry"]
        lines += [f"선택 original-ID 집합: 동일 {g['selected_sets_equal']} / 다름 {g['selected_sets_different']} images. "
            f"KV25 normal chunk 수: 증가 {g['chunks_new_increase']}, 동일 {g['chunks_new_same']}, "
            f"감소 {g['chunks_new_decrease']}. 경계 chunk에서 읽고 attention에서 제외한 유효 rows "
            f"{g['unused_valid_rows_new_sum']}개 ({g['unused_valid_bytes_new_sum']/1e6:.3f} MB/image-sum); "
            f"padding rows {g['padding_rows_new_sum']}개 ({g['padding_bytes_new_sum']/1e6:.3f} MB/image-sum).", "",
            "Structural 보존과 실제 메모리/전송량: "
            f"KV25 structural 포함 retention macro {format_pct(summary['qwen_ours_kv25']['structural_inclusive_retention_macro'])}; "
            f"structural {format_mb(summary['qwen_ours_kv25']['structural_read_mb_per_hit'])} MB/hit, "
            f"H2D {format_mb(summary['qwen_ours_kv25']['h2d_mb_per_hit'])} MB/hit, "
            f"compact KV {format_mb(summary['qwen_ours_kv25']['gpu_compact_cache_mb_per_hit'])} MB/hit, "
            f"GPU peak allocated {format_mb(summary['qwen_ours_kv25']['peak_gpu_allocated_mb_per_hit'])} MB/hit. "
            f"물리적으로 읽은 유효 content KV {format_mb(summary['qwen_ours_kv25']['valid_content_read_mb_per_hit'])} MB/hit와 "
            f"실제 attention에 남긴 content KV {format_mb(summary['qwen_ours_kv25']['retained_content_kv_mb_per_hit'])} MB/hit를 구분했다. "
            f"실제 normal read는 FullLoad의 {format_pct(summary['qwen_ours_kv25']['normal_read_vs_fullload'])}, "
            f"total read는 {format_pct(summary['qwen_ours_kv25']['total_read_vs_fullload'])}이다.", "",
            "Turn별 정답률: " + "; ".join(
                f"{labels[m]} " + ", ".join(f"T{t} {format_pct(v)}" for t,v in summary[m]["quality_by_turn"].items())
                for m in METHODS) + ".", "",
            "MT 후속 turn의 차이는 method별로 생성한 history의 차이도 포함한다. "
            "95% CI에 0이 있더라도 동등성 근거로 해석하지 않는다." if dataset == "mt"
            else "95% CI에 0이 있더라도 동등성 근거로 해석하지 않는다.", ""]
    smoke = phases["smoke"]
    lines += ["## 실행 조건과 감사", "",
        f"Smoke: {smoke['geometry']['images']} images × 3 questions × 4 arms = "
        f"{smoke['actual_physical_requests']} requests, report-side 감사 PASS. "
        "첫 smoke 시작은 기본 Python 환경에 transformers가 없어 모델 로드 전에 종료했고 "
        "physical request는 0개였다. 별도 retry1 디렉터리에서 검증된 mllm_ft 환경으로 "
        "전체 48요청을 수행했다. 첫 실패는 startup_failure.json에 보존했다. "
        "GQA/MT 원본 frozen manifest와 protected v2 store 메타데이터 해시를 각 image별로 대조했다.", "",
        "Read-only 활성화 때 보호된 payload 전체 SHA를 검증하고 metadata/FD를 상주한다. "
        "PIL 이미지 파일 hash 검사는 image당 한 번 요청 timer 밖에 수행한다. "
        "RGB decode는 매 normal-pixel request 안에서 수행하여 TTFT/E2E에 더한다. "
        "Qwen processor와 vision도 pixel TTFT 안에 있다. Cache hit에는 decode가 없고 "
        "raw decode field는 null이다 (timing erratum 참조). 활성화 시간/IO 및 "
        "posix_fadvise page-cache 힌트는 hit 요청 timer 밖에 기록한다. "
        "Hit의 실제 pread bytes/calls는 timer 안에 포함된다. Controller/NAND cache를 비웠다고 주장하지 않는다. "
        "T1의 score hook은 정상 full-image forward 안에서 실행되며 capture clone은 request E2E 후 진단으로 수행된다.", "",
        "통계는 image cluster 10,000 bootstrap resamples, seed 1234이다. "
        "품질과 지연의 방향은 관측 결과와 paired CI에 한정하며, 새로운 main rerun의 보장은 아니다.", "",
        "원시 요청, 고정 manifest, config, store inventory, GPU inventory, summary.csv와 "
        "report_audit.json과 별도 independent_audit.json의 해시 및 재실행 명령은 REPRODUCE.md에 있다.", "",
        "## 최종 판정", "",
        "- IMPLEMENTATION: PASS", "- CPU TEST: 42/42 PASS 및 pilot preflight PASS",
        "- GPU CORRECTNESS UNDER v2: PASS",
        "- LEGACY QWEN CHUNK25 REGRESSION: GPU gate G11 PASS; pilot legacy arm VALID UNDER v2",
        "- LLAVA CHUNK25/KV25 PROTECTION: CPU/protected-file PASS; LLaVA GPU NOT RUN in this migration",
        "- GQA 4-ARM PILOT: VALID UNDER v2",
        "- MT 4-ARM PILOT: VALID UNDER v2",
        "- PERSISTENCE/SESSION: NOT_REMEASURED",
        "- READY FOR QWEN KV25 MAIN RERUN: NO pending main-run storage/capacity plan; "
        "correctness passed and this small pilot kept its 30 GiB reserve, but main-run capacity is unverified.", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=MIGRATION)
    ap.add_argument("--results-dir", type=Path, required=True)
    args = ap.parse_args()
    run_root, out = args.run_root.resolve(), args.results_dir.resolve()
    require(not out.exists(), f"results path already exists: {out}")
    gate_path = run_root / "gpu_validation/validation.json"
    gate = read_json(gate_path)
    require(gate.get("status") == "PASS" and gate.get("pilot_eligible") is True,
            "GPU correctness gate is not PASS")
    protection_path = run_root / "llava_protection_receipt.json"
    protection = read_json(protection_path)
    require(protection.get("cpu_passed") is True and protection.get("cpu_tests_run") == 42 and
            protection.get("protected_files_passed") is True and
            protection.get("protected_files_checked") == 30205 and
            protection.get("llava_gpu") == "NOT RUN" and
            sha256(Path(protection["cpu_log"])) == protection["cpu_log_sha256"] and
            sha256(Path(protection["protection_verification"])) ==
            protection["protection_verification_sha256"] and
            gate.get("llava_protection", {}).get("receipt_sha256") == sha256(protection_path),
            "LLaVA CPU/protected-file final receipt invalid")
    gate_binding = {"path": str(gate_path), "sha256": sha256(gate_path)}
    startup = read_json(run_root / "smoke_pilot/startup_failure.json")
    require(startup.get("status") == "FAIL_STARTUP_BEFORE_MODEL_LOAD" and
            startup.get("physical_requests_executed") == 0 and
            startup.get("gpu_gate_sha256") == gate_binding["sha256"] and
            not (run_root / "smoke_pilot/raw.jsonl").exists(),
            "first smoke startup failure was not preserved honestly")
    phase_dirs = {"smoke": "smoke_pilot_retry1", "gqa": "gqa_pilot", "mt": "mt_pilot"}
    phases = {}
    for dataset in DATASETS:
        phases[dataset] = audit_phase(dataset, run_root / phase_dirs[dataset], gate_binding["sha256"])
    for dataset in ("gqa", "mt"):
        phases[dataset]["paired"] = pair_bootstrap(phases[dataset])
    audit = {"schema_version": "qwen25-kv25-independent-audit-v1", "status": "PASS",
             "gpu_gate": gate_binding, "report_source_sha256": sha256(Path(__file__)),
             "llava_protection_receipt": {"path": str(protection_path.resolve()),
                                          "sha256": sha256(protection_path)},
             "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
             "persistence_status": "NOT_REMEASURED", "execution_scope": "CACHE-HIT REEVALUATION",
             "startup_failure": {**startup, "path": str((run_root / "smoke_pilot/startup_failure.json").resolve()),
                                 "sha256": sha256(run_root / "smoke_pilot/startup_failure.json")},
             "phases": {d:{k:v for k,v in phases[d].items() if k != "rows"} for d in DATASETS}}
    report = render_report(phases, gate_binding, audit)
    out.mkdir(parents=True)
    shutil.copyfile(MIGRATION / "storage_plan.json", out / "storage_plan.json")
    shutil.copyfile(ROOT / "docs/qwen25_kv25_pilot_timing_addendum.md",
                    out / "pilot_timing_addendum.md")
    shutil.copyfile(ROOT / "docs/qwen25_kv25_pilot_timing_erratum.md",
                    out / "pilot_timing_erratum.md")
    write_new(out / "report_audit.json", audit)
    with (out / "summary.csv").open("x", newline="", encoding="utf-8") as f:
        fields = ["dataset", "method", "requests", "hit_requests", "all_quality", "hit_quality",
                  "t1_ttft_ms", "hit_ttft_ms", "hit_e2e_ms", "logical_content_retention_macro",
                  "logical_content_retention_weighted", "structural_inclusive_retention_macro",
                  "normal_read_mb_per_hit", "structural_read_mb_per_hit", "metadata_read_mb_per_hit",
                  "total_read_mb_per_hit", "valid_content_read_mb_per_hit", "retained_content_kv_mb_per_hit", "extra_valid_mb_per_hit",
                  "padding_read_mb_per_hit", "h2d_mb_per_hit", "gpu_compact_cache_mb_per_hit",
                  "peak_gpu_allocated_mb_per_hit", "normal_read_vs_fullload", "total_read_vs_fullload"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset in DATASETS:
            for method in METHODS:
                s = phases[dataset]["summary"][method]
                writer.writerow({"dataset":dataset, "method":method,
                    **{key:s.get(key) for key in fields if key not in ("dataset","method")}})
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    reproduce = f"""# Reproduce the Qwen KV25 pilot report

Validated interpreter: `/home/dblab/anaconda3/envs/mllm_ft/bin/python` (Torch 2.5.1+cu121, Transformers 4.57.6, bitsandbytes 0.49.2). The default `python` lacked Transformers and caused the preserved zero-request smoke startup failure.

Run root: `{run_root}`. GPU gate SHA256: `{gate_binding['sha256']}`.
Source pilot SHA256: `{read_json(run_root / 'gqa_pilot/config.json')['source_hashes']['scripts/94_eval_qwen_kv25_pilot.py']}`.
Report source SHA256: `{audit['report_source_sha256']}`.
Storage plan SHA256: `{sha256(MIGRATION / 'storage_plan.json')}`.
Timing addendum SHA256: `{sha256(ROOT / 'docs/qwen25_kv25_pilot_timing_addendum.md')}`.
Timing erratum SHA256: `{sha256(ROOT / 'docs/qwen25_kv25_pilot_timing_erratum.md')}`.
Protection receipt SHA256: `{sha256(protection_path)}`.

To regenerate this report from the **same raw requests**, choose a fresh results directory; the script will not overwrite an existing one:

```bash
/home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/95_report_qwen_kv25_pilot.py \
  --run-root {run_root} \
  --results-dir NEW_EMPTY_RESULTS_DIR
```

The report binds the exact `smoke_pilot_retry1`, `gqa_pilot`, and `mt_pilot` directories plus the preserved `smoke_pilot/startup_failure.json`. It is not a generic reporter for another run root. A fresh benchmark requires a new run-local output directory for each arm/dataset, the same gate and protected-store checks, and a newly bound reporting manifest before comparisons. Do not point this report command at a new raw run without updating and freezing those bindings.

The pilot stores in `runs/qwen25_correctness_v2_20260928T081111Z` are protected read-only. Persistence was NOT_REMEASURED. Exact workload, source and store hashes, GPU gate, and physical request IDs are in each phase's `manifest.json`, `config.json`, `store_inventory.json` and `raw.jsonl`. Report-side recalculations and per-file hashes are in `report_audit.json`; the separately implemented `scripts/96_audit_qwen_kv25_pilot.py` writes `independent_audit.json`.
"""
    (out / "REPRODUCE.md").write_text(reproduce, encoding="utf-8")
    print(json.dumps({"status":"PASS", "results_dir":str(out),
                      "gqa_requests":phases["gqa"]["actual_physical_requests"],
                      "mt_requests":phases["mt"]["actual_physical_requests"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

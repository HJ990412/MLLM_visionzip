#!/usr/bin/env python3
"""Independent, read-only GLM store audit to run AFTER main completion.

Only Python standard-library modules are used. No serving/store helpers,
tensor libraries, GPU calls, mutation, cleanup or non-adopted payload reads.
Native K/V equality is checked directly at token-major BF16 byte offsets.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time
import traceback

EXPECTED_RUN = "glm46v_prefixstore3_v6_20261007T104505Z"
FULL, B = "fullload", "ours_prefixstore_kv25"
METHODS = ("recompute", FULL, B)
MODEL_ID = "zai-org/GLM-4.6V-Flash"
REVISION = "411bb4d77144a3f03accbf4b780f5acb8b7cde4e"
ROW_BYTES, TOKEN_BYTES = 512, 40960
FORMAT = "glm46v_native_bf16_prefixstore_v1"
SEAL_FORMAT = "glm46v_prefixstore_completion_v1"
BUILD_SCOPE = "caller before runner.persist entry through complete return; identity preparation and wrapper tail included"
ACTIVATION_SCOPE = "caller before runner.activate entry through complete return; summary and wrapper tail included"


def require(value, message):
    if not value:
        raise AssertionError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(path.read_bytes(), object_pairs_hook=unique_object)


def safe_file(path, owned_root):
    require(path.is_relative_to(owned_root), f"outside owned root: {path}")
    current = owned_root
    require(not current.is_symlink(), f"symlink owned root: {current}")
    for piece in path.relative_to(owned_root).parts:
        current /= piece
        require(not current.is_symlink(), f"symlink refused: {current}")
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
            f"nonregular or hardlinked file refused: {path}")
    require(path.resolve(strict=True).is_relative_to(owned_root.resolve(strict=True)),
            f"resolved outside owned root: {path}")
    return info


def expected_positions(meta):
    content = meta["content_logical_indices"]
    first, count, length = content[0], len(content), meta["prefix_len"]
    grid = meta["geometry"]["image_grid_thw"]
    require(len(grid) == 1 and len(grid[0]) == 3, "not a single native image grid")
    temporal, height, width = grid[0]
    require(all(type(item) is int for item in grid[0]) and temporal == 1,
            "invalid native image grid integers")
    require(height > 0 and width > 0 and height % 2 == width % 2 == 0,
            "native spatial merge geometry differs")
    hh, ww = height // 2, width // 2
    require(count == hh * ww and content == list(range(first, first + count)),
            "native merged raster content count/order differs")
    result = [[], [], []]
    for logical in range(length):
        if logical < first:
            point = [logical] * 3
        elif logical < first + count:
            row, column = divmod(logical - first, ww)
            point = [first, first + row, first + column]
        else:
            point = [first + max(hh, ww) + logical - first - count] * 3
        for axis in range(3):
            result[axis].append(point[axis])
    return result, [[max(max(axis) for axis in result) + 1 - length]]


def audit_store(store, method, image_id, image_identity, frozen_code_digest, owner):
    for name in ("meta.json", "seal.json"):
        safe_file(store / name, owner)
    metadata_bytes, seal_bytes = (store / "meta.json").read_bytes(), (store / "seal.json").read_bytes()
    m, seal = read_json(store / "meta.json"), read_json(store / "seal.json")
    require(seal.get("format") == SEAL_FORMAT and seal.get("completed") is True,
            "store is not atomically sealed complete")
    require(seal["metadata_sha256"] == hashlib.sha256(metadata_bytes).hexdigest()
            and seal["metadata_bytes"] == len(metadata_bytes), "sealed metadata hash/bytes mismatch")
    require(m["format"] == FORMAT and m["model_family"] == "GLM-4.6V-Flash"
            and m["dtype"] == "bfloat16" and m["byte_order"] == "little", "format/native BF16 label differs")
    require((m["num_layers"], m["num_kv_heads"], m["head_dim"], m["row_bytes"]) == (40, 2, 128, ROW_BYTES)
            and m["bytes_per_native_token"] == TOKEN_BYTES, "native head/layer geometry or 40 KiB/token differs")
    integer_fields = ("num_layers", "num_kv_heads", "head_dim", "row_bytes", "prefix_len",
                      "original_content_count", "stored_content_count", "stored_rows", "chunk_size",
                      "n_chunks", "padding_rows", "valid_rows_last_chunk", "structural_count")
    require(all(type(m[key]) is int and m[key] >= 0 for key in integer_fields), "lossy/bool integer geometry")
    n, p = m["original_content_count"], m["prefix_len"]
    require(n > 0 and p >= n, "invalid original prefix/content lengths")
    full = method == FULL
    k = n if full else (n + 3) // 4
    require(m["storage_policy"] == ("full" if full else "prefix25"), "method/storage policy differs")
    require(m["stored_content_count"] == m["stored_rows"] == k, "original-N budget or exact EOF row count differs")
    require(m["chunk_size"] == 64 and m["n_chunks"] == (k + 63) // 64
            and m["valid_rows_last_chunk"] == (k - 1) % 64 + 1 and m["padding_rows"] == 0,
            "partial 64-row chunk geometry/padding differs")
    require(m["native_kv_shape"] == [1, 2, p, 128] and m["global_order_all_layers"] is True,
            "native KV shape/global rank differs")
    require(m["structural_layout"] == "kv_kind_layer_token_head_dimension", "unknown structure layout")
    content, structure, order = m["content_logical_indices"], m["structural_indices"], m["full_importance_permutation"]
    require(len(content) == n and all(type(i) is int for i in content) and content == sorted(set(content))
            and min(content) >= 0 and max(content) < p, "content logical mapping has duplicates/out-of-range IDs")
    require(structure == sorted(set(range(p)) - set(content)) and m["structural_count"] == len(structure),
            "structure mapping has duplicates/missing/budgeted content")
    require(all(type(i) is int for i in order) and sorted(order) == list(range(n)), "not a full original-N permutation")
    inverse = [0] * n
    for rank, original in enumerate(order):
        inverse[original] = rank
    require(m["original_to_stored"] == inverse and m["permutation_sha256"] == digest(order), "inverse/rank hash differs")
    require(m["stored_to_original"] == m["stored_row_to_original"] == order[:k], "frozen first-k mapping differs")
    require(len(m["prefix_input_ids"]) == p and all(type(i) is int and i >= 0 for i in m["prefix_input_ids"])
            and m["prefix_sha256"] == digest(m["prefix_input_ids"]), "prefix IDs/hash differs")
    require(m["retention_ratio"] == (1.0 if full else .25)
            and m["rounding_policy"] == ("all_original_content" if full else "ceil_original_content"), "budget policy differs")
    if full:
        require(order == list(range(n)) and m["layout"] == "token_major_canonical_bf16"
                and all(m[key] is None for key in ("score_source", "score_layer", "saliency_sha256")),
                "FullLoad canonical policy contains importance sidecar/provenance")
    else:
        require(m["layout"] == "token_major_repacked_bf16" and isinstance(m["score_source"], str)
                and m["score_source"] and type(m["score_layer"]) is int and m["score_layer"] >= 0
                and re.fullmatch(r"[0-9a-f]{64}", m["saliency_sha256"]) is not None, "B saliency provenance missing")
    position_ids, offset = expected_positions(m)
    require(m["logical_position_ids"] == position_ids and m["rope_offset"] == offset,
            "native original positions/offset differ from independent scalar raster reference")
    require(m["key_rope_state"] == "post_rope" and m["rope_config"]["partial_rotary_factor"] == .5
            and m["rope_config"]["mrope_section"] == [8, 12, 12], "native post/partial RoPE state differs")
    extra = m["extra"]
    require(extra["logical_position_ids"] == position_ids and extra["rope_offset"] == offset
            and extra["geometry"] == m["geometry"] and extra["rope_config"] == m["rope_config"]
            and extra["key_rope_state"] == "post_rope", "extra native positions/geometry contradict metadata")
    expected_identity = {key: value for key, value in extra.items()
                         if key not in ("score_source", "score_layer", "logical_position_ids")}
    expected_identity.update(prefix_input_ids=m["prefix_input_ids"], content_logical_indices=content,
                             logical_position_ids_sha256=digest(position_ids), kv_dtype="bfloat16",
                             num_layers=40, num_kv_heads=2, head_dim=128)
    require(m["identity"] == expected_identity and m["identity_sha256"] == digest(expected_identity)
            and seal["identity_sha256"] == m["identity_sha256"], "complete native prefix identity differs")
    identity = m["identity"]
    require(identity["model_id"] == MODEL_ID and identity["checkpoint_revision"] == REVISION
            and identity["processor_revision"] == REVISION and identity["image_sha256"] == image_identity["sha256"]
            and identity["code_revision"] == frozen_code_digest, "image/model/processor/frozen code identity differs")
    require(seal["files_manifest_sha256"] == digest(m["files"]), "seal/payload manifest hash differs")
    payload_names = {"structural_kv.bin"} | {f"layer_{li:03d}/{kind}.bin" for li in range(40) for kind in ("k", "v")}
    require(set(m["files"]) == payload_names, "payload allowlist differs")
    expected_top = {"meta.json", "seal.json", "structural_kv.bin"} | {f"layer_{li:03d}" for li in range(40)}
    require({item.name for item in store.iterdir()} == expected_top, "unsealed extra file/diagnostic inside store")
    for li in range(40):
        layer_dir = store / f"layer_{li:03d}"
        require(not layer_dir.is_symlink() and {item.name for item in layer_dir.iterdir()} == {"k.bin", "v.bin"},
                "layer files differ from strict allowlist")
    payload, evidence = {}, {}
    allocated = 0
    for relative in sorted(payload_names | {"meta.json", "seal.json"}):
        path = store / relative
        info = safe_file(path, owner)
        raw = metadata_bytes if relative == "meta.json" else seal_bytes if relative == "seal.json" else path.read_bytes()
        evidence[str(path.relative_to(owner))] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        allocated += info.st_blocks * 512
        if relative in payload_names:
            expected_size = len(structure) * TOKEN_BYTES if relative == "structural_kv.bin" else k * ROW_BYTES
            record = m["files"][relative]
            require(record["size"] == info.st_size == len(raw) == expected_size
                    and record["sha256"] == hashlib.sha256(raw).hexdigest(), f"payload bits/hash/exact EOF differs:{path}")
            payload[relative] = raw
    content_bytes, structural_bytes = k * TOKEN_BYTES, len(structure) * TOKEN_BYTES
    logical_bytes = content_bytes + structural_bytes + len(metadata_bytes) + len(seal_bytes)
    require(m["bytes_content_kv"] == content_bytes and m["bytes_structural_kv"] == structural_bytes
            and m["bytes_metadata_file"] == len(metadata_bytes) and m["bytes_seal_file"] == len(seal_bytes)
            and m["store_logical_bytes"] == logical_bytes, "payload/metadata/seal logical bytes differ")
    selection = sorted(order[:k])
    row = {"image_id": image_id, "method_id": method, "attempt_id": owner.name,
           "N": n, "stored_content_rows": k, "retention_ratio_actual": k / n,
           "structural_rows": len(structure), "chunk_size": 64, "n_chunks": m["n_chunks"],
           "valid_rows_last_chunk": m["valid_rows_last_chunk"], "padding_rows": 0,
           "native_layers": 40, "native_kv_heads": 2, "head_dim": 128, "dtype": "bfloat16",
           "bytes_per_native_token": TOKEN_BYTES, "content_bytes": content_bytes,
           "structure_bytes": structural_bytes, "sealed_logical_bytes": logical_bytes,
           "sealed_file_allocated_bytes_excluding_directories": allocated,
           "verified_payload_files": len(payload_names), "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
           "seal_sha256": hashlib.sha256(seal_bytes).hexdigest(), "identity_sha256": m["identity_sha256"],
           "selected_set_sha256": digest(selection), "full_hash_status": "PASS"}
    return m, payload, evidence, row


def compare_pair(full, compressed, full_payload, b_payload):
    for key in ("identity", "prefix_input_ids", "prefix_len", "content_logical_indices", "structural_indices",
                "logical_position_ids", "rope_offset", "rope_config", "key_rope_state", "geometry"):
        require(full[key] == compressed[key], f"FullLoad/B source prefix identity differs:{key}")
    require(full_payload["structural_kv.bin"] == b_payload["structural_kv.bin"], "full structural native bits differ")
    comparisons = 0
    for relative, selected_bits in b_payload.items():
        if relative == "structural_kv.bin":
            continue
        original_bits = full_payload[relative]
        for row, original in enumerate(compressed["stored_row_to_original"]):
            require(selected_bits[row * ROW_BYTES:(row + 1) * ROW_BYTES]
                    == original_bits[original * ROW_BYTES:(original + 1) * ROW_BYTES],
                    f"B native head token bits differ from FullLoad original row:{relative}:{original}")
            comparisons += 1
    return comparisons


def audit_setup(setup_path, metas, image_id, owner, source_dialogue):
    events = [json.loads(line, object_pairs_hook=unique_object) for line in setup_path.read_text().splitlines() if line.strip()]
    require(len(events) == 4, "adopted image must have exactly four setup events")
    by_key, event_ids = {}, set()
    for event in events:
        method, kind = event["method_id"], event["event_kind"]
        key = method, kind
        require(method in (FULL, B) and kind in ("build", "activation") and key not in by_key,
                "duplicate/foreign method or setup event; each build/activation must occur once")
        require(event["event_id"] not in event_ids, "duplicate setup event ID")
        event_ids.add(event["event_id"])
        require(event["phase"] == "main" and event["image_id"] == image_id
                and event["source_dialog_id"] == source_dialogue
                and Path(event["store_path"]) == owner / "stores" / method,
                "setup ownership/source dialogue differs")
        inner_key = "inner_writer_call_return_ms" if kind == "build" else "inner_activation_core_ms"
        outer, inner = event["duration_ms"], event[inner_key]
        require(all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value >= 0 for value in (outer, inner)),
                "setup outer/inner durations absent, nonfinite, negative or mislabeled")
        require(outer + 1e-9 >= inner, "caller outer scope excludes part of the recorded inner core")
        require(event["endpoint"] == ("outer_writer_return" if kind == "build" else "outer_activation_return")
                and event["measured_scope"] == (BUILD_SCOPE if kind == "build" else ACTIVATION_SCOPE),
                "setup lacks the required caller outer preparation-through-return scope")
        meta = metas[method]
        if kind == "build":
            require(event["logical_bytes"] == event["os_write_bytes"] == meta["store_logical_bytes"]
                    and event["content_payload_bytes"] == meta["bytes_content_kv"]
                    and event["structure_payload_bytes"] == event["structural_payload_bytes"] == meta["bytes_structural_kv"]
                    and event["metadata_bytes"] == meta["bytes_metadata_file"] + meta["bytes_seal_file"]
                    and event["padding_bytes"] == 0, "build recorded OS write/payload/logical bytes differ")
            require(event["N_content"] == meta["original_content_count"]
                    and event["k"] == event["stored_k"] == meta["stored_content_count"], "build original-N budget differs")
            timing = event["timing_ms"]
            require(timing["os_write_bytes"] == event["os_write_bytes"]
                    and timing["os_write_calls"] == event["os_write_calls"]
                    and timing["total_persistence_ms"] <= inner + 1e-9,
                    "writer component/inner duration contradicts caller outer setup")
        else:
            io = event["activation_io"]
            require(io["bytes"] == meta["store_logical_bytes"]
                    and sum(call["actual_bytes"] for call in io["call_details"]) == io["bytes"]
                    and io["preads"] == len(io["call_details"]), "activation integrity returned-byte accounting differs")
            require(io["per_kind"]["activation_content"]["bytes"] == meta["bytes_content_kv"]
                    and io["per_kind"]["activation_structure"]["bytes"] == meta["bytes_structural_kv"]
                    and io["per_kind"]["activation_metadata"]["bytes"] == meta["bytes_metadata_file"] + meta["bytes_seal_file"],
                    "activation content/structure/metadata integrity scan incomplete")
        by_key[key] = event
    require(set(by_key) == {(method, kind) for method in (FULL, B) for kind in ("build", "activation")},
            "missing per-image build/activation event")
    rows = []
    for method in (FULL, B):
        build, activation = by_key[method, "build"], by_key[method, "activation"]
        rows.append({"image_id": image_id, "method_id": method, "attempt_id": owner.name,
                     "source_dialog_id": source_dialogue, "build_event_id": build["event_id"],
                     "activation_event_id": activation["event_id"],
                     "caller_outer_build_ms": build["duration_ms"],
                     "inner_writer_call_return_ms": build["inner_writer_call_return_ms"],
                     "build_preparation_wrapper_difference_ms": build["duration_ms"] - build["inner_writer_call_return_ms"],
                     "caller_outer_activation_ms": activation["duration_ms"],
                     "inner_activation_core_ms": activation["inner_activation_core_ms"],
                     "activation_preparation_wrapper_difference_ms": activation["duration_ms"] - activation["inner_activation_core_ms"],
                     "physical_setup_outer_ms": build["duration_ms"] + activation["duration_ms"],
                     "build_endpoint": build["endpoint"], "build_measured_scope": build["measured_scope"],
                     "activation_endpoint": activation["endpoint"], "activation_measured_scope": activation["measured_scope"],
                     "os_write_bytes": build["os_write_bytes"], "os_write_calls": build["os_write_calls"],
                     "activation_returned_integrity_bytes": activation["activation_io"]["bytes"],
                     "build_events_count": 1, "activation_events_count": 1, "scope_accounting_status": "PASS"})
    return rows, event_ids


def audit_raw(raw_path, metas, image_id, owner, expected_requests, source_dialogue):
    seen, counts, calls, returned = set(), Counter(), Counter(), Counter()
    selected_sets = {method: set() for method in (FULL, B)}
    source_captures = Counter()
    requests = 0
    with raw_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line, object_pairs_hook=unique_object)
            method, turn = row["method_id"], row["turn_id"]
            require(method in METHODS and turn in (1, 2, 3) and row["phase"] == "main"
                    and row["image_id"] == image_id and row["attempt_id"] == owner.name and row["status"] == "PASS",
                    "adopted raw ownership/method/turn/status differs")
            key = row["dialog_id"], method, turn
            require(key in expected_requests and key not in seen, "missing/duplicate/foreign adopted membership")
            seen.add(key)
            requests += 1
            counts[method] += 1
            if row["source_T1_capture"]:
                require(method in (FULL, B) and turn == 1 and row["dialog_id"] == source_dialogue,
                        "foreign/late source T1 capture")
                source_captures[method] += 1
            trace = row.get("os_pread_trace", [])
            if method == "recompute" or turn == 1:
                require(row["ssd_read_bytes"] == row["pread_calls"] == 0 and not trace,
                        "pixels request claims KV-store reads")
                continue
            meta = metas[method]
            selected = sorted(meta["stored_row_to_original"])
            require(row["selected_content_original"] == selected, "followup selected set changed")
            selected_sets[method].add(tuple(row["selected_content_original"]))
            expected_logical = sorted(meta["structural_indices"] + [meta["content_logical_indices"][i]
                                                                     for i in meta["stored_row_to_original"]])
            request_trace = row["request_trace"]
            require(request_trace["logical_indices"] == expected_logical
                    and request_trace["native_prefix_length"] == meta["prefix_len"]
                    and request_trace["compact_prefix_length"] == len(expected_logical)
                    and request_trace["rope_offset"] == meta["rope_offset"], "followup original IDs/positions/cache length differ")
            native_positions = request_trace["position_ids"]
            require(len(native_positions) == 3 and all(len(axis) == 1 for axis in native_positions)
                    and [axis[0][:meta["prefix_len"]] for axis in native_positions] == meta["logical_position_ids"],
                    "followup native prefix positions differ")
            provenance = row["source_store"]
            require(Path(provenance["path"]) == owner / "stores" / method
                    and provenance["identity"] == meta["identity"]
                    and provenance["source"]["method_id"] == method
                    and provenance["source"]["dialog_id"] == source_dialogue, "foreign store identity/source method")
            require(row["N_content"] == meta["original_content_count"] and row["k"] == meta["stored_content_count"]
                    and row["S_structural"] == meta["structural_count"]
                    and row["kv_geometry"] == {"num_layers": 40, "num_kv_heads": 2, "head_dim": 128, "dtype_bytes": 2},
                    "raw original budget/native geometry differs")
            require(row["decode_store_reads"] == row["online_query_score_calls"] == 0,
                    "decode repeated-read/reselection recorded")
            grouped = defaultdict(list)
            for call in trace:
                grouped[call["source"]].append(call)
            require(set(grouped) == set(meta["files"]), "request read files differ from sealed payload")
            total = 0
            for relative, file_calls in grouped.items():
                cursor, size = 0, meta["files"][relative]["size"]
                expected_kind = "structure" if relative == "structural_kv.bin" else "content"
                for call in file_calls:
                    actual = call["returned"]
                    require(type(actual) is int and actual > 0 and actual <= call["requested"]
                            and call["offset"] == cursor and call["requested"] == size - cursor
                            and call["kind"] == expected_kind, "pread bytes/ranges/kind or repeated load differs")
                    cursor += actual
                    total += actual
                require(cursor == size, "request missing/extra/padded payload bytes")
            require(total == row["ssd_read_bytes"] == meta["bytes_content_kv"] + meta["bytes_structural_kv"]
                    and len(trace) == row["pread_calls"], "actual returned pread summary differs")
            calls[method] += len(trace)
            returned[method] += total
    require(seen == expected_requests, "adopted raw membership incomplete")
    require(source_captures == Counter({FULL: 1, B: 1}), "per-image source capture is not exactly once per store method")
    require(all(len(sets) == 1 for sets in selected_sets.values()), "image followup selection is not invariant")
    return {"requests": requests, "requests_per_method": dict(counts),
            "actual_pread_calls": dict(calls), "actual_returned_pread_bytes": dict(returned),
            "selected_set_invariance": "PASS"}


def save_json(path, value):
    require(not path.exists(), f"refuse output overwrite:{path}")
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_audit(run, output):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "RUNNING", "run": str(run), "output": str(output),
               "started_utc": datetime.now(timezone.utc).isoformat(), "scope": "adopted main stores only; CPU/read-only",
               "gpu_calls": 0, "payload_deleted_or_modified": False, "images": [], "failures": []}
    store_rows, comparison_rows, setup_rows = [], [], []
    seen_setup_event_ids = set()
    try:
        require(run.name == EXPECTED_RUN, "this independent script belongs to the final v6 run")
        complete_path = run / "main/COMPLETED.json"
        require(complete_path.is_file(), "main is not completed; refuse scans during GPU timing")
        completion = read_json(complete_path)
        require(completion["status"] == "PASS" and completion["phase"] == "main", "main completion is not PASS")
        freeze_path = run / "freeze.json"
        if (run / "active_freeze.json").exists():
            active = read_json(run / "active_freeze.json")
            freeze_path = run / active["path"]
            require(freeze_path.resolve(strict=True).is_relative_to(run.resolve(strict=True))
                    and file_hash(freeze_path) == active["sha256"], "active freeze differs")
        freeze = read_json(freeze_path)
        require(file_hash(freeze_path) == completion["freeze_sha256"], "main completed against a different freeze")
        for name, key in (("config.json", "config_sha256"), ("environment.json", "environment_sha256"),
                          ("CONTRACT.md", "contract_sha256"), ("validation_manifest.json", "validation_manifest_sha256")):
            require(file_hash(run / name) == freeze[key], f"frozen config/environment/contract differs:{name}")
        for name, expected in freeze["manifests"].items():
            require(file_hash(run / name) == expected, f"frozen workload manifest differs:{name}")
        frozen_module_hashes = {}
        for relative, expected in freeze["source_sha256"].items():
            require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                    "unsafe frozen source relative path")
            frozen_copy = run / "source_frozen" / relative
            require(file_hash(frozen_copy) == expected, f"frozen source copy differs:{relative}")
            if relative.startswith("mmimpress/glm46v/") and relative.endswith(".py"):
                frozen_module_hashes[Path(relative).name] = expected
        frozen_code_digest = digest(frozen_module_hashes)
        manifest = read_json(run / "manifests/main_manifest.json")
        identities = read_json(run / "manifests/main_image_identity.json")
        source_dialogues = read_json(run / "manifests/main_source_dialogues.json")
        expected = defaultdict(set)
        for dialogue in manifest["dialogues"]:
            for turn in dialogue["turns"]:
                for method in METHODS:
                    key = dialogue["dialog_id"], method, int(turn["turn_id"])
                    require(key not in expected[dialogue["image_id"]], "duplicate frozen workload entry")
                    expected[dialogue["image_id"]].add(key)
        require(len(expected) == 398 and sum(map(len, expected.values())) == 36549,
                "workload not 398 images / 36,549 final requests")
        commits = sorted((run / "main/images").glob("*/COMMITTED.json"))
        require({path.parent.name for path in commits} == set(expected), "committed ownership coverage differs")
        require(completion["images"] == len(commits) and completion["requests"] == sum(map(len, expected.values())),
                "main completion counts differ")
        receipt.update(source_freeze_sha256=file_hash(freeze_path), completion_sha256=file_hash(complete_path),
                       frozen_code_digest=frozen_code_digest, script_sha256=file_hash(Path(__file__)))
        for ordinal, committed_path in enumerate(commits):
            image_dir = committed_path.parent
            safe_file(committed_path, run)
            committed = read_json(committed_path)
            iid, attempt_id = image_dir.name, committed["attempt_id"]
            require(re.fullmatch(r"attempt_[0-9]+", attempt_id) is not None, "unsafe adopted attempt ID")
            owner = image_dir / attempt_id
            require(not owner.is_symlink() and committed["status"] == "PASS"
                    and committed["image_id"] == iid and committed["phase"] == "main", "foreign adopted ownership")
            require(committed.get("payload_cleanup") == "NOT_PERFORMED", "adopted payload was cleaned; regeneration required")
            raw_path, setup_path = owner / "raw.jsonl", owner / "setup_events.jsonl"
            require(Path(committed["raw_path"]) == raw_path and Path(committed["setup_path"]) == setup_path,
                    "committed paths are outside the exact adopted attempt")
            for path in (raw_path, setup_path, owner / "IMAGE_COMPLETE.json", owner / "payload_evidence.json"):
                safe_file(path, owner)
            require(read_json(owner / "IMAGE_COMPLETE.json") == committed, "completion/adoption receipts differ")
            require(file_hash(raw_path) == committed["raw_sha256"] and file_hash(setup_path) == committed["setup_sha256"]
                    and file_hash(owner / "payload_evidence.json") == committed["payload_evidence_sha256"],
                    "committed raw/setup/payload evidence changed")
            stores = owner / "stores"
            require(not stores.is_symlink() and {path.name for path in stores.iterdir()} == {FULL, B},
                    "foreign or unexpected store directories in adopted owner")
            values, observed_evidence = {}, {}
            for method in (FULL, B):
                values[method] = audit_store(stores / method, method, iid, identities[iid], frozen_code_digest, owner)
                _, _, file_evidence, store_row = values[method]
                observed_evidence.update(file_evidence)
                store_rows.append(store_row)
            require(observed_evidence == read_json(owner / "payload_evidence.json"), "adopted payload evidence differs")
            full, b = values[FULL], values[B]
            compared = compare_pair(full[0], b[0], full[1], b[1])
            metas = {method: value[0] for method, value in values.items()}
            source_dialogue = source_dialogues[iid]["dialog_id"]
            image_setup_rows, image_event_ids = audit_setup(setup_path, metas, iid, owner, source_dialogue)
            require(not (seen_setup_event_ids & image_event_ids), "setup event IDs reused across adopted image owners")
            seen_setup_event_ids.update(image_event_ids)
            setup_rows.extend(image_setup_rows)
            raw_metrics = audit_raw(raw_path, metas, iid, owner, expected[iid], source_dialogue)
            require(raw_metrics["requests"] == committed["requests"], "committed request count differs")
            comparison_rows.append({"image_id": iid, "attempt_id": attempt_id,
                "N": full[0]["original_content_count"], "k": b[0]["stored_content_count"],
                "compared_K_or_V_token_rows": compared, "compared_content_bytes": compared * ROW_BYTES,
                "structure_bytes_exact": full[0]["bytes_structural_kv"], "selected_bits_exact": "PASS",
                "structure_bits_exact": "PASS", "identity_positions_exact": "PASS",
                "selected_set_invariance": raw_metrics["selected_set_invariance"],
                "adopted_requests": raw_metrics["requests"],
                "full_followup_returned_bytes": raw_metrics["actual_returned_pread_bytes"][FULL],
                "B_followup_returned_bytes": raw_metrics["actual_returned_pread_bytes"][B]})
            receipt["images"].append({"image_id": iid, "attempt_id": attempt_id,
                                      "committed_sha256": file_hash(committed_path), **raw_metrics})
            print(json.dumps({"audit_image": ordinal + 1, "image_id": iid, "status": "PASS"}), flush=True)
            del values, full, b
        receipt["status"] = "PASS"
    except Exception as error:
        receipt["status"] = "FAIL"
        receipt["failures"].append({"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
    finally:
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - started,
                       audited_images=len(receipt["images"]), audited_stores=len(store_rows),
                       payload_files_fullhash_verified=sum(row["verified_payload_files"] for row in store_rows),
                       K_or_V_selected_token_rows_compared=sum(row["compared_K_or_V_token_rows"] for row in comparison_rows),
                       sealed_logical_bytes=sum(row["sealed_logical_bytes"] for row in store_rows),
                       declared_payload_bytes=sum(row["content_bytes"] + row["structure_bytes"] for row in store_rows),
                       actual_store_allocated_file_bytes_excluding_directories=sum(
                           row["sealed_file_allocated_bytes_excluding_directories"] for row in store_rows),
                       caller_outer_setup_scope_status="PASS" if receipt["status"] == "PASS" else "INCOMPLETE_OR_FAIL",
                       physical_build_events=sum(row["build_events_count"] for row in setup_rows),
                       activation_events=sum(row["activation_events_count"] for row in setup_rows),
                       setup_outer_build_ms=sum(row["caller_outer_build_ms"] for row in setup_rows),
                       setup_inner_writer_ms=sum(row["inner_writer_call_return_ms"] for row in setup_rows),
                       setup_outer_activation_ms=sum(row["caller_outer_activation_ms"] for row in setup_rows),
                       setup_inner_activation_core_ms=sum(row["inner_activation_core_ms"] for row in setup_rows),
                       physical_setup_outer_ms=sum(row["physical_setup_outer_ms"] for row in setup_rows),
                       limitations=["Scores are not persisted: full descending-rank recomputation requires regenerated image-only vision scores.",
                                    "This audit verifies native cached bits/positions/storage and raw recorded OS calls; it does not rerun GPU logits.",
                                    "OS returned bytes and st_blocks are not NAND physical I/O or a cold-NAND guarantee.",
                                    "Only exact adopted COMMITTED owners are scanned; failed/non-adopted payloads are untouched.",
                                    "Caller outer setup durations are authoritative; retained inner durations are nested diagnostics and are not added again.",
                                    "Serving E2E+setup excludes benchmark guards/conditioning, journal/evidence writes, image adoption and audit wall time."])
        for name, rows in (("store_geometry_hashes.csv", store_rows), ("FullLoad_B_bits_selection.csv", comparison_rows),
                           ("caller_outer_setup_scopes.csv", setup_rows)):
            with (output / name).open("x", newline="") as handle:
                if rows:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
        receipt["csv_sha256"] = {name: file_hash(output / name) for name in
                                  ("store_geometry_hashes.csv", "FullLoad_B_bits_selection.csv", "caller_outer_setup_scopes.csv")}
        save_json(output / "receipt.json", receipt)
        print(json.dumps({"status": receipt["status"], "receipt": str(output / "receipt.json")}), flush=True)
    return 0 if receipt["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute-after-main-completion", action="store_true",
                        help="Explicitly allow the full scan; still refuses incomplete main.")
    args = parser.parse_args()
    if not args.execute_after_main_completion:
        parser.error("Scan not authorized by this invocation; add --execute-after-main-completion after timing stops.")
    run = args.run_dir.absolute()
    require(not run.is_symlink(), "symlink run root refused")
    output = (args.output_dir or Path(__file__).resolve().parents[1] / "store_audit" /
              datetime.now(timezone.utc).strftime("attempt_%Y%m%dT%H%M%S%fZ")).absolute()
    results_root = Path(__file__).resolve().parents[1]
    require(output.resolve(strict=False).is_relative_to(results_root.resolve(strict=True)),
            "audit outputs must remain in this v6 result directory")
    return run_audit(run, output)


if __name__ == "__main__":
    raise SystemExit(main())

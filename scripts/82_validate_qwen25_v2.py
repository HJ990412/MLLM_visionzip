#!/usr/bin/env python3
"""Qwen2.5-VL correctness v2: matched serving paths and separate diagnostics.

The workload manifest must already exist, be hashed, and contain ten frozen
image/question pairs. This script creates only fresh artifacts in ``--out-dir``.
It never edits the v1 validator, old stores, or old validation reports.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mmimpress.cvpr25 import budget_chunk_count  # noqa: E402
from mmimpress.dataset import exact_score  # noqa: E402
from mmimpress.qwen25.runner import (  # noqa: E402
    CHECKPOINT_REVISION, MAX_NEW_TOKENS, MAX_PIXELS, MIN_PIXELS, MODEL_ID,
    SEED, Qwen25Runner,
)
from mmimpress.qwen25.store import (  # noqa: E402
    inverse_permutation, stable_visual_order,
)


SCHEMA = "qwen25-gpu-correctness-v2"
GATES = tuple(f"G{i}" for i in range(1, 16))
PER_SAMPLE_GATES = GATES[:13]
V1_ATOL = 0.125
V1_RTOL = 0.02
FP32_ORACLE_ATOL = 1e-5
FP32_ORACLE_RTOL = 1e-5


def _load_script(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper {rel}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v1 = _load_script("qwen25_validation_v1_readonly", "scripts/78_validate_qwen25.py")
debug = _load_script("qwen25_debug_v1_readonly", "scripts/81_debug_qwen25_correctness.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True,
                  ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _resolve_input_path(raw: str) -> Path:
    path = Path(raw)
    return (path if path.is_absolute() else ROOT / path).resolve()


def manifest_preflight(path: Path) -> dict:
    """Fail before model load if workload/content/source identity drifted."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    digest = manifest.get("manifest_sha256")
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if canonical_sha256(body) != digest:
        raise ValueError("frozen validation manifest digest differs")
    rows = manifest.get("samples")
    if not isinstance(rows, list) or len(rows) != 10:
        raise ValueError("v2 validation requires exactly ten frozen pairs")
    old = [("n355567", q) for q in ("201751701", "201751740", "201751873")]
    seen = [(str(x["image_id"]), str(x["question_id"])) for x in rows]
    if seen[:3] != old or len(set(seen)) != len(seen):
        raise ValueError("original three or unique pair order changed")
    for row in rows:
        if set(row) != {"image_id", "image_path", "image_sha256",
                        "question_id", "question", "gold"}:
            raise ValueError("sample keys differ from frozen v2 contract")
        image = _resolve_input_path(row["image_path"])
        if sha256_file(image) != row["image_sha256"]:
            raise ValueError(f"image content changed: {image}")
        if not str(row["question"]).strip() or not str(row["gold"]).strip():
            raise ValueError("empty question or gold answer")
    for rel, expected in manifest.get("source_hashes", {}).items():
        if sha256_file(_resolve_input_path(rel)) != expected:
            raise ValueError(f"frozen source changed: {rel}")
    config = manifest["configuration"]
    expected = {"model_id": MODEL_ID, "checkpoint_revision": CHECKPOINT_REVISION,
                "attention_backend": "sdpa", "budget_ratio": .25,
                "chunk_size": 64, "min_pixels": MIN_PIXELS,
                "max_pixels": MAX_PIXELS, "max_new_tokens": MAX_NEW_TOKENS,
                "seed": SEED}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"frozen configuration changed: {key}")
    return manifest


def logit_stats(left: torch.Tensor, right: torch.Tensor) -> dict:
    if left is None or right is None or left.shape != right.shape:
        return {"comparable": False, "allclose_v1": False}
    a, b = left.float().cpu(), right.float().cpu()
    delta = (a - b).abs()
    allowed = V1_ATOL + V1_RTOL * b.abs()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {"comparable": True, "finite": finite,
            "exact": bool(torch.equal(a, b)),
            "allclose_v1": bool(torch.allclose(a, b, atol=V1_ATOL,
                                               rtol=V1_RTOL)),
            "max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
            "p99_abs": float(torch.quantile(delta, .99)),
            "v1_failed_elements": int((delta > allowed).sum()),
            "element_count": int(delta.numel())}


def path_comparison(left: dict, right: dict) -> dict:
    logits = logit_stats(left["logits"], right["logits"])
    first = left["first_token_id"] == right["first_token_id"]
    generated = left["generated_token_ids"] == right["generated_token_ids"]
    prediction = left["prediction"] == right["prediction"]
    return {"logits": logits, "first_token_agreement": first,
            "generated_sequence_agreement": generated,
            "prediction_agreement": prediction,
            "exact_target": logits.get("exact", False) and first and generated
            and prediction}


def _result_path(result: dict) -> dict:
    return {"logits": result["first_logits"],
            "first_token_id": result["first_token_id"],
            "generated_token_ids": result["generated_token_ids"],
            "prediction": result["prediction"]}


def _logits_hash(logits: torch.Tensor) -> str:
    return hashlib.sha256(logits.float().cpu().contiguous().numpy().tobytes()).hexdigest()


def _public_result(path: dict) -> dict:
    return {k: path[k] for k in ("first_token_id", "generated_token_ids",
                                  "prediction")}


def _memory_selected(capture, logical_indices: list[int]):
    """Independent in-memory reference: select original captured cache rows."""
    indices = torch.tensor(logical_indices, dtype=torch.long,
                           device=capture.layers[0][0].device)
    return [(k.index_select(2, indices).detach().clone(),
             v.index_select(2, indices).detach().clone())
            for k, v in capture.layers]


def _cache_from_layers(runner: Qwen25Runner, layers):
    from transformers import DynamicCache
    cache = DynamicCache(config=runner.model.config.text_config)
    for li, (key, value) in enumerate(layers):
        cache.update(key.to(runner.device).clone(),
                     value.to(runner.device).clone(), li)
    return cache


def trace_suffix(runner: Qwen25Runner, layers, suffix: torch.Tensor,
                 positions: torch.Tensor, full_prefix_len: int,
                 prefix_mask: torch.Tensor | None = None) -> dict:
    cache = _cache_from_layers(runner, layers)
    cache_len = int(cache.get_seq_length())
    if prefix_mask is None:
        prefix_mask = torch.ones((1, cache_len), dtype=torch.long,
                                 device=runner.device)
    else:
        prefix_mask = prefix_mask.to(runner.device)
    mask = torch.cat((prefix_mask,
                      torch.ones((1, suffix.shape[1]), dtype=torch.long,
                                 device=runner.device)), dim=1)
    return debug.execute_path(
        runner, input_ids=suffix.to(runner.device),
        position_ids=positions[:, :, full_prefix_len:].to(runner.device),
        cache_position=torch.arange(cache_len, cache_len + suffix.shape[1],
                                    device=runner.device),
        attention_mask=mask, decode_mask=mask, suffix_from=0, cache=cache)


def _stock_positions(runner: Qwen25Runner, full_ids: torch.Tensor,
                     grid: torch.Tensor):
    """Invoke installed Qwen MRoPE independently of adapter request helper."""
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    return Qwen2_5_VLModel.get_rope_index(
        SimpleNamespace(config=runner.model.config),
        input_ids=full_ids.cpu(), image_grid_thw=grid.cpu(),
        attention_mask=torch.ones_like(full_ids.cpu()))


def _analytic_mask(prefix_enabled: list[bool], suffix_len: int) -> torch.Tensor:
    p = len(prefix_enabled)
    q = torch.arange(suffix_len)[:, None]
    prefix = torch.tensor(prefix_enabled, dtype=torch.bool)[None, :].expand(suffix_len, -1)
    suffix = torch.arange(suffix_len)[None, :] <= q
    return torch.cat((prefix, suffix), dim=1)


def _stock_mask(runner: Qwen25Runner, prefix_enabled: list[bool],
                suffix_len: int) -> torch.Tensor:
    """Independent installed causal-mask construction on CPU dummy cache."""
    from transformers import DynamicCache
    from transformers.masking_utils import create_causal_mask
    tc = runner.model.config.text_config
    p = len(prefix_enabled)
    cache = DynamicCache(config=tc)
    dummy = torch.zeros((1, tc.num_key_value_heads, p,
                         tc.hidden_size // tc.num_attention_heads),
                        dtype=torch.bfloat16)
    cache.update(dummy, dummy, 0)
    mask = create_causal_mask(
        config=tc,
        input_embeds=torch.zeros((1, suffix_len, tc.hidden_size),
                                 dtype=torch.bfloat16),
        attention_mask=torch.tensor([prefix_enabled + [True] * suffix_len],
                                    dtype=torch.long),
        cache_position=torch.arange(p, p + suffix_len),
        past_key_values=cache, position_ids=None)
    if mask is None or mask.shape != (1, 1, suffix_len, p + suffix_len):
        raise AssertionError("stock causal mask has unexpected shape")
    return (mask[0, 0] if mask.dtype == torch.bool
            else mask[0, 0] == 0).cpu()


def _actual_mask_checks(path: dict, expected: torch.Tensor) -> dict:
    rows = []
    for li, mask in enumerate(path["trace"].masks):
        actual = None if mask is None else (
            mask[0, 0] if mask.dtype == torch.bool else mask[0, 0] == 0)
        rows.append({"layer": li, "actual_present": actual is not None,
                     "exact": actual is not None and
                     _mask_identity(actual.cpu(), expected)})
    return {"all_layers_exact": len(rows) == 28 and all(r["exact"] for r in rows),
            "per_layer": rows}


def _position_identity(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    return actual.shape == expected.shape and bool(torch.equal(actual, expected))


def _mapping_identity(actual: list[int], expected: list[int]) -> bool:
    return actual == expected and len(set(actual)) == len(actual)


def _mask_identity(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    return actual.shape == expected.shape and bool(torch.equal(actual, expected))


def _structural_hash(capture, meta: dict) -> dict:
    import ctypes
    indices = torch.tensor(
        [i for i in range(len(capture.prefix_ids))
         if i < capture.visual_start or i >= capture.visual_start + capture.visual_count],
        dtype=torch.long)
    pairs = []
    for kind_index in range(2):
        layers = []
        for key, value in capture.layers:
            tensor = (key, value)[kind_index]
            rows = tensor[0].index_select(1, indices.to(tensor.device))
            layers.append(rows.permute(1, 0, 2).to("cpu").contiguous())
        pairs.append(torch.stack(layers))
    structural = torch.stack(pairs).contiguous()
    raw = ctypes.string_at(structural.data_ptr(), structural.numel() * 2)
    digest = hashlib.sha256(raw).hexdigest()
    actual = meta["files"]["structural_kv.bin"]["sha256"]
    return {"equal": digest == actual,
            "expected_sha256_from_capture": digest,
            "stored_sha256": actual,
            "byte_count": len(raw),
            "structural_indices": indices.tolist()}


def _negative_semantic_controls(positions: torch.Tensor,
                                mask: torch.Tensor,
                                mapping: list[int]) -> dict:
    bad_pos = positions.clone()
    bad_pos[0, 0, -1] += 1
    bad_mask = mask.clone()
    bad_mask[0, -1] = ~bad_mask[0, -1]
    bad_mapping = mapping.copy()
    bad_mapping[-1], bad_mapping[-2] = bad_mapping[-2], bad_mapping[-1]
    return {
        "mrope_perturbation_detected": not _position_identity(bad_pos, positions),
        "mask_perturbation_detected": not _mask_identity(bad_mask, mask),
        "mapping_perturbation_detected": not _mapping_identity(bad_mapping, mapping),
        "changed_position": [0, 0, int(positions.shape[-1] - 1)],
        "changed_mask_cell": [0, int(mask.shape[-1] - 1)],
        "changed_mapping_slot": len(mapping) - 1}


def _fp32_oracle(path: dict, prefix_layers, prefix_mask: list[bool],
                 num_heads: int, num_kv_heads: int, head_dim: int) -> torch.Tensor:
    """Explicit FP32 GQA QK-softmax-V at layer 0, outside SDPA dispatch."""
    record = path["trace"].records[0]
    q = record["q_post_mrope"].float()
    k_suffix = record["k_post_mrope"].float()
    v_flat = record["v"].float()
    suffix_len = q.shape[2]
    v_suffix = v_flat.view(1, suffix_len, num_kv_heads,
                           head_dim).transpose(1, 2)
    k = torch.cat((prefix_layers[0][0].detach().cpu().float(), k_suffix), dim=2)
    v = torch.cat((prefix_layers[0][1].detach().cpu().float(), v_suffix), dim=2)
    k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
    v = v.repeat_interleave(num_heads // num_kv_heads, dim=1)
    logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
    visible = _analytic_mask(prefix_mask, suffix_len)
    logits = logits.masked_fill(~visible[None, None], float("-inf"))
    probabilities = torch.softmax(logits, dim=-1)
    return torch.matmul(probabilities, v).float()


def _layer_output_comparison(left: dict, right: dict) -> dict:
    trace = debug.layer_diffs(left["trace"], right["trace"])
    rows = [x for x in trace["per_stage"] if x["stage"] == "layer_output"]
    return {"first_divergence": trace["first_divergence"],
            "all_exact": trace["first_divergence"] is None,
            "per_layer_output": rows,
            "embedding": trace["embedding"]}


def _io_gate(result: dict, meta: dict, selective: bool) -> dict:
    evidence = v1._io_check(result, meta, selective=selective)
    selected = int(result["selected_chunks"])
    # pread_exact may retry short reads, but every planned span must be exact.
    spans = result["read_io"]["span_details"]
    visual = [s for s in spans if s["kind"] == "visual"]
    expected = selected * meta["chunk_size"] * meta["row_bytes"]
    no_unselected = (all(s["offset"] == 0 and
                         s["requested_bytes"] == expected for s in visual)
                     and result["visual_read_bytes"] == len(visual) * expected
                     and (selected == meta["n_chunks"] or
                          result["visual_read_bytes"] < meta["bytes_visual_kv"]))
    expected_spans = 2 * meta["num_layers"] + 1
    exact_pread_count = (result["read_io"]["preads"] == expected_spans
                         and result["read_io"]["spans"] == expected_spans)
    evidence["strict_no_unselected_visual_payload"] = no_unselected if selective else True
    evidence["pread_count_exact"] = exact_pread_count
    evidence["planned_pread_count"] = expected_spans
    evidence["passed"] = bool(evidence["passed"] and exact_pread_count and
                              evidence["strict_no_unselected_visual_payload"])
    return evidence


def _gate(status: str, **details: Any) -> dict:
    return {"status": status, "details": details}


def _is_pass(gates: dict) -> bool:
    return all(value["status"] == "PASS" for value in gates.values())


def _verify_protected(before_path: Path) -> dict:
    before = json.loads(before_path.read_text(encoding="utf-8"))
    refs = {}
    for key, field in (("legacy_reference", "files"), ("qwen_reference", "qwen")):
        ref = _resolve_input_path(before[key])
        if sha256_file(ref) != before[key + "_sha256"]:
            return {"passed": False, "reason": f"reference digest changed: {key}"}
        refs[key] = json.loads(ref.read_text(encoding="utf-8"))[field]
    groups = {"legacy": refs["legacy_reference"],
              "qwen_v1": refs["qwen_reference"],
              "debug_v1": before["debug_v1"]}
    changed = {}
    count = 0
    for group, records in groups.items():
        mismatch = []
        for rel, expected in records.items():
            target = _resolve_input_path(rel)
            count += 1
            if not target.is_file() or target.stat().st_size != expected["bytes"] or \
                    sha256_file(target) != expected["sha256"]:
                mismatch.append(rel)
        changed[group] = mismatch
    return {"passed": count > 0 and not any(changed.values()),
            "checked_file_count": count, "changed_by_group": changed,
            "reference_sha256_unchanged": True}


def _llava_gate(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    log = _resolve_input_path(record["log_path"])
    passed = (record.get("ok") is True and record.get("exit_code") == 0
              and record.get("tests_run", 0) > 0
              and sha256_file(log) == record["log_sha256"])
    return {"passed": passed, "scope": "cpu",
            "gpu_scope": record.get("actual_llava_gpu_regression", "NOT RUN"),
            "artifact": str(path), "artifact_sha256": sha256_file(path),
            "tests_run": record.get("tests_run"),
            "command": record.get("command"),
            "log_sha256": record.get("log_sha256")}


def _matched_gate(memory, ssd, memory_repeat, ssd_repeat, production, production_repeat):
    pair = path_comparison(memory, ssd)
    mrep = path_comparison(memory, memory_repeat)
    srep = path_comparison(ssd, ssd_repeat)
    prod = path_comparison(ssd, _result_path(production))
    prep = path_comparison(_result_path(production), _result_path(production_repeat))
    trace = _layer_output_comparison(memory, ssd)
    stable = all(x["exact_target"] for x in (mrep, srep, prep))
    exact = pair["exact_target"] and prod["exact_target"] and trace["all_exact"]
    return {"status": "UNRESOLVED" if not stable else "PASS" if exact else "FAIL",
            "details": {"target": "exact BF16 logits, generated tokens and layer trace",
                        "in_memory_vs_ssd": pair, "ssd_trace_vs_production": prod,
                        "in_memory_repeat": mrep, "ssd_repeat": srep,
                        "production_repeat": prep, "per_layer": trace,
                        "memory_state_before": memory["state_before"],
                        "ssd_state_before": ssd["state_before"]}}


def _source_trace(runner, image, question, prefix_len):
    enc_cpu = runner.processor(text=[runner._chat_text(question, ())],
                               images=[image], return_tensors="pt")
    ids = enc_cpu["input_ids"]
    positions, _ = runner._logical_positions(ids, enc_cpu["image_grid_thw"])
    enc = {k: v.to(runner.device) if torch.is_tensor(v) else v
           for k, v in enc_cpu.items()}
    mask = torch.ones_like(enc["input_ids"])
    trace = debug.execute_path(
        runner, input_ids=enc["input_ids"], position_ids=positions.to(runner.device),
        cache_position=torch.arange(ids.shape[1], device=runner.device),
        attention_mask=mask, decode_mask=mask, suffix_from=prefix_len,
        pixel_kwargs={k: enc[k] for k in ("pixel_values", "image_grid_thw")})
    return trace, enc_cpu


def validate_sample(runner, sample, out_dir, sample_index):
    image_id, qid = str(sample["image_id"]), str(sample["question_id"])
    image_sha, question = sample["image_sha256"], sample["question"]
    sample_dir = out_dir / f"sample_{sample_index:02d}_{image_id}_{qid}"
    sample_dir.mkdir(exist_ok=False)
    with Image.open(_resolve_input_path(sample["image_path"])) as source:
        image = source.convert("RGB")
    recompute = runner.run_pixels(image, question, capture=True,
                                  image_sha256=image_sha, return_logits=True)
    capture = recompute["capture"]
    canonical_dir, repacked_dir = sample_dir / "canonical_store", sample_dir / "repacked_store"
    canonical_meta = runner.persist(capture, canonical_dir, layout="canonical")["metadata"]
    repacked_meta = runner.persist(capture, repacked_dir, layout="repacked")["metadata"]
    canonical, repacked = runner._activate(canonical_dir, image_sha), runner._activate(repacked_dir, image_sha)
    full_loaded, repacked_full = canonical.load_prefix(1.0), repacked.load_prefix(1.0)
    compact_loaded = repacked.load_prefix(.25)
    prefix_len = len(capture.prefix_ids)
    selected_chunks = budget_chunk_count(repacked_meta["n_chunks"], .25)
    selected_count = min(capture.visual_count, selected_chunks * 64)
    score_values = capture.scores.detach().cpu().float().tolist()
    permutation = sorted(range(capture.visual_count),
                         key=lambda i: (-float(score_values[i]), i))
    physical_selected = permutation[:selected_count]
    structural_indices = [i for i in range(prefix_len)
                          if i < capture.visual_start or
                          i >= capture.visual_start + capture.visual_count]
    logical_indices = sorted(structural_indices +
                             [capture.visual_start + i for i in physical_selected])
    memory_compact = _memory_selected(capture, logical_indices)
    full_ids, suffix, positions, deltas = runner._cache_hit_ids(canonical_meta, question, ())
    suffix_len = int(suffix.shape[1])
    if suffix_len < 1:
        raise AssertionError("question suffix is empty")
    stock_positions, stock_deltas = _stock_positions(
        runner, full_ids, torch.tensor(capture.image_grid_thw, dtype=torch.long))
    negative = _negative_semantic_controls(
        stock_positions, _analytic_mask([True] * len(logical_indices), suffix_len),
        logical_indices)
    g = {}

    canonical_bits = v1._bitwise_layers(capture.layers, full_loaded.layers)
    repacked_bits = v1._bitwise_layers(capture.layers, repacked_full.layers)
    compact_bits = v1._bitwise_layers(memory_compact, compact_loaded.layers)
    canonical_hashes = v1._physical_hashes(capture, canonical_meta)
    repacked_hashes = v1._physical_hashes(capture, repacked_meta)
    canonical_structural = _structural_hash(capture, canonical_meta)
    repacked_structural = _structural_hash(capture, repacked_meta)
    representative = []
    for layer_index in (0, len(capture.layers) // 2, len(capture.layers) - 1):
        for kind_index, kind in enumerate(("k", "v")):
            tensor = capture.layers[layer_index][kind_index].detach().cpu().reshape(-1)
            representative.append({"layer": layer_index, "kind": kind,
                                   "first_bf16_word": int(tensor[0].view(torch.int16)),
                                   "middle_bf16_word": int(tensor[tensor.numel() // 2].view(torch.int16)),
                                   "last_bf16_word": int(tensor[-1].view(torch.int16))})
    all_positions = tuple(range(prefix_len))
    g["G1"] = _gate(
        "PASS" if canonical_bits["equal"] and canonical_hashes["equal"]
        and canonical_structural["equal"]
        and canonical_meta["dtype"] == "bfloat16"
        and len(capture.layers) == canonical_meta["num_layers"] == 28 else "FAIL",
        native_dtype=canonical_meta["dtype"], layers=len(capture.layers),
        shape=canonical_meta["native_kv_shape"],
        system_and_visual_kv_bits=canonical_bits,
        complete_visual_file_hashes=canonical_hashes,
        structural_file_hash=canonical_structural,
        representative_elements=representative)
    perm_ok = repacked_meta["stored_to_original"] == permutation
    inverse_ok = repacked_meta["original_to_stored"] == inverse_permutation(permutation)
    meta_pos_ok = (repacked_meta["prefix_len"] == canonical_meta["prefix_len"]
                   and repacked_meta["prefix_input_ids"] == capture.prefix_ids
                   and canonical_meta["prefix_input_ids"] == capture.prefix_ids
                   and repacked_meta["logical_position_ids"] == canonical_meta["logical_position_ids"])
    g["G3"] = _gate(
        "PASS" if repacked_bits["equal"] and repacked_hashes["equal"]
        and repacked_structural["equal"]
        and repacked_full.logical_indices == all_positions
        and perm_ok and inverse_ok and meta_pos_ok else "FAIL",
        repacked_full_bits=repacked_bits, physical_file_hashes=repacked_hashes,
        structural_file_hash=repacked_structural,
        full_logical_indices_exact=repacked_full.logical_indices == all_positions,
        permutation_exact=perm_ok, inverse_exact=inverse_ok,
        prefix_and_mrope_metadata_exact=meta_pos_ok)
    selected_tuple = []
    for stored_index, original in enumerate(physical_selected):
        logical = capture.visual_start + original
        selected_tuple.append({"stored_index": stored_index,
                               "original_visual_index": original,
                               "original_prompt_index": logical,
                               "compact_index": logical_indices.index(logical),
                               "token_id": capture.prefix_ids[logical],
                               "mrope_thw": stock_positions[:, 0, logical].tolist()})
    selection_ok = (compact_bits["equal"]
                    and _mapping_identity(list(compact_loaded.logical_indices),
                                          logical_indices)
                    and repacked_meta["structural_indices"] == structural_indices
                    and list(compact_loaded.selected_visual_original) == sorted(physical_selected)
                    and compact_loaded.selected_chunks == selected_chunks
                    and repacked_meta["stored_to_original"][:selected_count] == physical_selected
        and negative["mapping_perturbation_detected"])
    g["G7"] = _gate(
        "PASS" if selection_ok else "FAIL",
        selected_chunk_ids=list(range(selected_chunks)),
        selected_original_visual_ids_physical=physical_selected,
        selected_original_visual_ids_logical=sorted(physical_selected),
        selected_original_prompt_indices=logical_indices,
        selected_tuples=selected_tuple,
        negative_mapping_fixture_detected=negative["mapping_perturbation_detected"],
        selected_kv_bits=compact_bits,
        compact_prefix_length=len(logical_indices))

    full_memory = trace_suffix(runner, capture.layers, suffix, positions, prefix_len)
    full_ssd = trace_suffix(runner, full_loaded.layers, suffix, positions, prefix_len)
    full_memory_repeat = trace_suffix(runner, capture.layers, suffix, positions, prefix_len)
    full_ssd_repeat = trace_suffix(runner, full_loaded.layers, suffix, positions, prefix_len)
    full_production = runner.run_cache(canonical_dir, question, budget_ratio=1.0,
                                       image_sha256=image_sha, return_logits=True)
    full_production_repeat = runner.run_cache(canonical_dir, question, budget_ratio=1.0,
                                              image_sha256=image_sha, return_logits=True)
    g["G2"] = _matched_gate(full_memory, full_ssd, full_memory_repeat,
                            full_ssd_repeat, full_production, full_production_repeat)
    ours_memory = trace_suffix(runner, memory_compact, suffix, positions, prefix_len)
    ours_ssd = trace_suffix(runner, compact_loaded.layers, suffix, positions, prefix_len)
    ours_memory_repeat = trace_suffix(runner, memory_compact, suffix, positions, prefix_len)
    ours_ssd_repeat = trace_suffix(runner, compact_loaded.layers, suffix, positions, prefix_len)
    from mmimpress.qwen25.vision import VisionScoreCapture
    score_calls = [0]
    original_score = VisionScoreCapture._compute_scores
    def counted_score(self):
        score_calls[0] += 1
        return original_score(self)
    VisionScoreCapture._compute_scores = counted_score
    try:
        ours_production = runner.run_cache(repacked_dir, question, budget_ratio=.25,
                                           image_sha256=image_sha, return_logits=True)
        ours_production_repeat = runner.run_cache(repacked_dir, question, budget_ratio=.25,
                                                  image_sha256=image_sha, return_logits=True)
    finally:
        VisionScoreCapture._compute_scores = original_score
    g["G4"] = _matched_gate(ours_memory, ours_ssd, ours_memory_repeat,
                            ours_ssd_repeat, ours_production, ours_production_repeat)
    g["G4"]["details"].update({
        "memory_selected_KV_vs_SSD_bits": compact_bits,
        "logical_indices_exact": list(compact_loaded.logical_indices) == logical_indices,
        "compact_cache_length": len(logical_indices)})
    g["G5"] = _gate(
        "PASS" if recompute["vision_calls"] == 1 and
        full_production["vision_calls"] == 0 and ours_production["vision_calls"] == 0 else "FAIL",
        recompute_vision_calls=recompute["vision_calls"],
        fullload_vision_calls=full_production["vision_calls"],
        ours_vision_calls=ours_production["vision_calls"])
    g["G6"] = _gate(
        "PASS" if score_calls[0] == 0 and ours_production["selector_ms"] == 0 else "FAIL",
        online_query_score_calls=score_calls[0],
        selector_ms=ours_production["selector_ms"])


    # Independent stock MRoPE and independently constructed causal visibility.
    expected_compact_pos = stock_positions[:, :, torch.tensor(logical_indices)]
    mrope_equal = (_position_identity(stock_positions, positions)
                   and _position_identity(compact_loaded.position_ids,
                                          expected_compact_pos)
                   and torch.equal(stock_deltas, deltas)
                   and stock_positions[:, 0, :prefix_len].tolist() ==
                   canonical_meta["logical_position_ids"]
                   and stock_positions[:, 0, :prefix_len].tolist() ==
                   repacked_meta["logical_position_ids"]
                   and canonical_meta["rope_deltas"] == stock_deltas.tolist()
                   and repacked_meta["rope_deltas"] == stock_deltas.tolist()
                   and all(t["mrope_thw"] == stock_positions[:, 0, t["original_prompt_index"]].tolist()
                           for t in selected_tuple))
    g["G8"] = _gate(
        "PASS" if mrope_equal and negative["mrope_perturbation_detected"] else "FAIL",
        stock_vs_runtime_and_store_exact=mrope_equal,
        negative_fixture_detected=negative["mrope_perturbation_detected"],
        full_positions_shape=list(stock_positions.shape),
        suffix_first_thw=stock_positions[:, 0, prefix_len].tolist(),
        suffix_last_thw=stock_positions[:, 0, -1].tolist(),
        rope_deltas=stock_deltas.tolist(),
        selected_token_tuples=selected_tuple)

    expected_full = _analytic_mask([True] * prefix_len, suffix_len)
    expected_ours = _analytic_mask([True] * len(logical_indices), suffix_len)
    stock_full = _stock_mask(runner, [True] * prefix_len, suffix_len)
    stock_ours = _stock_mask(runner, [True] * len(logical_indices), suffix_len)
    stock_matches_math = (_mask_identity(stock_full, expected_full)
                          and _mask_identity(stock_ours, expected_ours))
    mask_checks = {"full_memory": _actual_mask_checks(full_memory, expected_full),
                   "full_ssd": _actual_mask_checks(full_ssd, expected_full),
                   "ours_memory": _actual_mask_checks(ours_memory, expected_ours),
                   "ours_ssd": _actual_mask_checks(ours_ssd, expected_ours)}
    matched_masks = all(x["all_layers_exact"] for x in mask_checks.values())

    # Dense P0 uses identical selected captured KV but keeps original prompt
    # slots and zero-masks omitted visual tokens. This remains a diagnostic.
    dense_prefix = []
    dense_mask = torch.zeros((1, prefix_len), dtype=torch.long)
    dense_mask[0, torch.tensor(logical_indices)] = 1
    for key, value in memory_compact:
        dk = torch.zeros((1, key.shape[1], prefix_len, key.shape[3]),
                         dtype=key.dtype, device=key.device)
        dv = torch.zeros_like(dk)
        indices = torch.tensor(logical_indices, dtype=torch.long, device=key.device)
        dk.index_copy_(2, indices, key)
        dv.index_copy_(2, indices, value)
        dense_prefix.append((dk, dv))
    dense = trace_suffix(runner, dense_prefix, suffix, positions, prefix_len, dense_mask)
    dense_expected = _analytic_mask(dense_mask[0].bool().tolist(), suffix_len)
    dense_stock = _stock_mask(runner, dense_mask[0].bool().tolist(), suffix_len)
    dense_mask_exact = (_mask_identity(dense_expected, dense_stock)
                        and _actual_mask_checks(dense, dense_expected)["all_layers_exact"])
    tc = runner.model.config.text_config
    head_dim = tc.hidden_size // tc.num_attention_heads
    dense_fp32 = _fp32_oracle(dense, dense_prefix, dense_mask[0].bool().tolist(),
                              tc.num_attention_heads, tc.num_key_value_heads, head_dim)
    compact_fp32 = _fp32_oracle(ours_memory, memory_compact,
                                [True] * len(logical_indices),
                                tc.num_attention_heads, tc.num_key_value_heads, head_dim)
    fp32_delta = (dense_fp32 - compact_fp32).abs()
    fp32_close = bool(torch.allclose(dense_fp32, compact_fp32,
                                    atol=FP32_ORACLE_ATOL, rtol=FP32_ORACLE_RTOL))
    g["G9"] = _gate(
        "PASS" if stock_matches_math and matched_masks and dense_mask_exact
        and fp32_close and negative["mask_perturbation_detected"] else "FAIL",
        stock_matches_analytic=stock_matches_math,
        matched_actual_masks_all_layers=mask_checks,
        dense_actual_mask_all_layers=dense_mask_exact,
        negative_fixture_detected=negative["mask_perturbation_detected"],
        fp32_gqa_attention_oracle={
            "atol": FP32_ORACLE_ATOL, "rtol": FP32_ORACLE_RTOL,
            "allclose": fp32_close, "max_abs": float(fp32_delta.max()),
            "mean_abs": float(fp32_delta.mean()),
            "shape": list(dense_fp32.shape)})

    full_io = _io_gate(full_production, canonical_meta, selective=False)
    ours_io = _io_gate(ours_production, repacked_meta, selective=True)
    g["G11"] = _gate(
        "PASS" if ours_io["strict_no_unselected_visual_payload"] else "FAIL",
        selected_chunks=selected_chunks, total_chunks=repacked_meta["n_chunks"],
        visual_read_bytes=ours_production["visual_read_bytes"],
        full_visual_bytes=repacked_meta["bytes_visual_kv"],
        sequential_spans=ours_io["span_details"])
    g["G12"] = _gate(
        "PASS" if full_io["passed"] and ours_io["passed"] else "FAIL",
        fullload=full_io, ours25=ours_io,
        activation_io_excluded_from_request=True)
    g["G13"] = _gate(
        "PASS" if canonical_meta["chunk_size"] == repacked_meta["chunk_size"] == 64
        and compact_loaded.selected_chunks == selected_chunks
        and compact_loaded.kept_visual_tokens == selected_count
        and compact_loaded.padding_rows_read == selected_chunks * 64 - selected_count else "FAIL",
        chunk_size=repacked_meta["chunk_size"], n_chunks=repacked_meta["n_chunks"],
        budget_ratio=.25, selected_chunks=selected_chunks,
        selected_visual_tokens=selected_count,
        padding_rows_read=compact_loaded.padding_rows_read)

    # Wrong image identity must fail. A poisoned model rope_deltas tensor must
    # be cleared by the cache hit, then the original output must repeat exactly.
    wrong_hash = "0" * 64 if image_sha != "0" * 64 else "1" * 64
    wrong_identity_rejected = False
    try:
        runner.run_cache(repacked_dir, question, budget_ratio=.25,
                         image_sha256=wrong_hash, return_logits=False)
    except ValueError as exc:
        wrong_identity_rejected = "image_sha256" in str(exc)
    runner.model.model.rope_deltas = torch.full(
        (1, 1), 123456, dtype=torch.long, device=runner.device)
    try:
        isolation_repeat = runner.run_cache(repacked_dir, question,
                                            budget_ratio=.25, image_sha256=image_sha,
                                            return_logits=True)
        state_clean = runner.model.model.rope_deltas is None
    finally:
        runner._clear_request_state()
    isolation_cmp = path_comparison(_result_path(ours_production),
                                    _result_path(isolation_repeat))
    g["G10"] = _gate(
        "PASS" if wrong_identity_rejected and state_clean
        and isolation_cmp["exact_target"] else "FAIL",
        wrong_image_identity_rejected=wrong_identity_rejected,
        deliberately_poisoned_rope_deltas_cleared=state_clean,
        repeat_original_vs_baseline=isolation_cmp)

    # Both legacy comparisons stay visible under the unchanged v1 tolerance.
    recompute_trace, enc_cpu = _source_trace(runner, image, question, prefix_len)
    if enc_cpu["input_ids"][0].tolist() != full_ids[0].tolist():
        raise AssertionError("recompute prompt IDs differ from cached prompt IDs")
    d1_logits = logit_stats(recompute_trace["logits"], full_memory["logits"])
    d1_k = debug.tensor_compare(
        recompute_trace["trace"].records[0]["k_pre_mrope"],
        full_memory["trace"].records[0]["k_pre_mrope"])
    d1_first = debug.layer_diffs(
        recompute_trace["trace"], full_memory["trace"])["first_divergence"]
    d1_branch_ok = (d1_first is None or
                    (d1_first["layer"] == 0 and
                     d1_first["stage"] in
                     ("k_pre_mrope", "v", "attention_output")))
    d2_logits = logit_stats(dense["logits"], ours_memory["logits"])
    d2_attention = debug.tensor_compare(
        dense["trace"].records[0]["attention_output"],
        ours_memory["trace"].records[0]["attention_output"])
    d2_first = debug.layer_diffs(
        dense["trace"], ours_memory["trace"])["first_divergence"]
    d2_branch_ok = (d2_first is None or
                    (d2_first["layer"] == 0 and
                     d2_first["stage"] == "attention_output"))
    diagnostics = {
        "numerical_branch_status": "PASS" if d1_branch_ok and d2_branch_ok
        else "UNRESOLVED",
        "D1_recompute_full_prefill_vs_split_full_load": {
            "classification": "semantic_regression_numerical_diagnostic",
            "layer0_pre_mrope_k": d1_k, "logits": d1_logits,
            "first_divergence": d1_first,
            "known_numerical_branch": d1_branch_ok,
            "first_token_agreement": recompute_trace["first_token_id"] ==
            full_memory["first_token_id"],
            "generated_sequence_agreement": recompute_trace["generated_token_ids"] ==
            full_memory["generated_token_ids"],
            "prediction_agreement": recompute_trace["prediction"] == full_memory["prediction"],
            "recompute_quality": exact_score(recompute["prediction"], sample["gold"]),
            "fullload_quality": exact_score(full_production["prediction"], sample["gold"])},
        "D2_dense_selected_vs_logical_compact": {
            "classification": "semantic_regression_numerical_diagnostic",
            "layer0_attention_output": d2_attention, "logits": d2_logits,
            "first_divergence": d2_first,
            "known_numerical_branch": d2_branch_ok,
            "first_token_agreement": dense["first_token_id"] == ours_memory["first_token_id"],
            "generated_sequence_agreement": dense["generated_token_ids"] ==
            ours_memory["generated_token_ids"],
            "prediction_agreement": dense["prediction"] == ours_memory["prediction"],
            "fp32_oracle_max_abs": float(fp32_delta.max()),
            "fp32_oracle_mean_abs": float(fp32_delta.mean())}}
    return {
        "sample_index": sample_index, "image_id": image_id, "question_id": qid,
        "question": question, "gold": sample["gold"], "image_sha256": image_sha,
        "store_dirs": {"canonical": str(canonical_dir), "repacked": str(repacked_dir)},
        "gates": g,
        "methods": {
            "recompute": {**_public_result(_result_path(recompute)),
                          "first_logits_sha256": _logits_hash(recompute["first_logits"]),
                          "quality": exact_score(recompute["prediction"], sample["gold"])},
            "fullload": {**_public_result(_result_path(full_production)),
                         "first_logits_sha256": _logits_hash(full_production["first_logits"]),
                         "quality": exact_score(full_production["prediction"], sample["gold"])},
            "ours25": {**_public_result(_result_path(ours_production)),
                       "first_logits_sha256": _logits_hash(ours_production["first_logits"]),
                       "quality": exact_score(ours_production["prediction"], sample["gold"])}},
        "diagnostics": diagnostics}


def _verify_frozen_inputs(path: Path) -> dict:
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("schema_version") != "qwen25-correctness-v2-frozen-inputs-v1":
        raise ValueError("wrong frozen input schema")
    files = frozen.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("empty frozen input set")
    changed = []
    for rel, expected in files.items():
        target = _resolve_input_path(rel)
        if not target.is_file() or target.stat().st_size != expected["bytes"] or \
                sha256_file(target) != expected["sha256"]:
            changed.append(rel)
    if changed:
        raise ValueError(f"frozen input bytes changed: {changed[:20]}")
    required = {"scripts/82_validate_qwen25_v2.py",
                "docs/qwen25_correctness_contract_v2.md",
                "runs/qwen25_correctness_v2_20260928T073413Z/validation_manifest.json"}
    if not required.issubset(files):
        raise ValueError("freeze does not include validator, contract, manifest")
    return {"path": str(path), "sha256": sha256_file(path),
            "file_count": len(files), "all_bytes_unchanged": True,
            "source_hashes": {rel: record["sha256"] for rel, record in files.items()}}


def _d3_reference(path: Path) -> dict:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("schema") != "qwen25_correctness_v2_evidence_reference_v1":
        raise ValueError("wrong evidence manifest schema")
    source = evidence["sources"]["r0_r1_and_p0_p1_p2_trace"]
    source_path = _resolve_input_path(source["relative_path"])
    if sha256_file(source_path) != source["sha256"]:
        raise ValueError("prior P1/P2 trace hash changed")
    facts = evidence["facts"]["p0_p1_p2"]
    return {"classification": "reference_only_numerical_diagnostic",
            "source_path": str(source_path),
            "source_sha256": source["sha256"],
            "evidence_manifest": str(path),
            "evidence_manifest_sha256": sha256_file(path),
            "current_production_path": facts["current_production_path"],
            "p0_vs_p1_logits_max_abs": facts["p0_vs_p1_logits_max_abs_diff"],
            "p0_vs_p2_logits_max_abs": facts["p0_vs_p2_logits_max_abs_diff"],
            "p1_vs_p2_logits_max_abs": facts["p1_vs_p2_logits_max_abs_diff"],
            "generated_sequences": facts["all_paths_generated_tokens"],
            "source_json_pointers": facts["json_pointers"]}


def _preflight_negative_fixtures() -> dict:
    # Uses the exact predicates applied to real MRoPE, token selection, and
    # causal-mask arrays. Each deliberately invalid fixture must be rejected.
    positions = torch.arange(9).view(1, 1, 9).repeat(3, 1, 1)
    candidate = positions.clone()
    candidate[0, 0, -1] += 1
    mrope_rejected = not torch.equal(candidate, positions)
    token_mapping = [0, 2, 4, 6]
    bad_mapping = [0, 4, 2, 6]
    mapping_rejected = (len(set(bad_mapping)) == len(bad_mapping)
                        and not _mapping_identity(bad_mapping, token_mapping))
    mask = _analytic_mask([True, False, True], 4)
    bad_mask = mask.clone()
    bad_mask[0, -1] = ~bad_mask[0, -1]
    mask_rejected = not torch.equal(bad_mask, mask)
    return {"passed": mrope_rejected and mapping_rejected and mask_rejected,
            "mrope_perturbation_rejected": mrope_rejected,
            "selected_token_mapping_perturbation_rejected": mapping_rejected,
            "causal_mask_perturbation_rejected": mask_rejected}


def _summarize(report: dict) -> None:
    for name in GATES:
        if name in ("G14", "G15"):
            continue
        rows = [sample.get("gates", {}).get(name, {"status": "NOT RUN"})
                for sample in report["samples"]]
        statuses = [row["status"] for row in rows]
        status = ("FAIL" if "FAIL" in statuses else
                  "UNRESOLVED" if "UNRESOLVED" in statuses else
                  "PASS" if len(rows) == 10 and all(x == "PASS" for x in statuses)
                  else "NOT RUN")
        if name == "G10":
            cross = report.get("cross_image_isolation")
            if cross is None:
                status = "NOT RUN"
            elif cross["status"] != "PASS":
                status = cross["status"]
        report["gates"][name] = {
            "status": status, "passed_samples": statuses.count("PASS"),
            "failed_samples": statuses.count("FAIL"),
            "unresolved_samples": statuses.count("UNRESOLVED"),
            "not_run_samples": statuses.count("NOT RUN")}
    statuses = [report["gates"][name]["status"] for name in GATES]
    report["gpu_system_correctness"] = (
        "FAIL" if "FAIL" in statuses else
        "UNRESOLVED" if "UNRESOLVED" in statuses else
        "PASS" if all(x == "PASS" for x in statuses) else "NOT RUN")
    branches = [sample.get("diagnostics", {}).get("numerical_branch_status")
                for sample in report["samples"]]
    report["numerical_branch_status"] = (
        "UNRESOLVED" if "UNRESOLVED" in branches else
        "PASS" if len(branches) == 10 and all(x == "PASS" for x in branches)
        else "NOT RUN")
    report["status"] = (
        "UNRESOLVED" if report["gpu_system_correctness"] == "PASS" and
        report["numerical_branch_status"] == "UNRESOLVED"
        else report["gpu_system_correctness"])
    report["pilot_eligible"] = (report["status"] == "PASS" and
                                report["numerical_branch_status"] == "PASS")
    completed = [s for s in report["samples"] if "methods" in s]
    if completed:
        methods = ("recompute", "fullload", "ours25")
        report["dataset_quality"] = {
            method: {"correct": sum(x["methods"][method]["quality"] for x in completed),
                     "evaluated": len(completed),
                     "accuracy": sum(x["methods"][method]["quality"] for x in completed)
                     / len(completed)}
            for method in methods}
        report["dataset_quality"]["recompute_minus_fullload_accuracy"] = (
            report["dataset_quality"]["recompute"]["accuracy"] -
            report["dataset_quality"]["fullload"]["accuracy"])
        report["dataset_quality"]["ours25_minus_fullload_accuracy"] = (
            report["dataset_quality"]["ours25"]["accuracy"] -
            report["dataset_quality"]["fullload"]["accuracy"])


def _release_sample_stores(runner: Qwen25Runner, sample_index: int,
                           sample: dict, out_dir: Path) -> None:
    directory = out_dir / f"sample_{sample_index:02d}_{sample['image_id']}_{sample['question_id']}"
    for name in ("canonical_store", "repacked_store"):
        key = str((directory / name).resolve())
        store = runner._stores.pop(key, None)
        if store is not None:
            store.close()
        runner.activation_records.pop(key, None)


def _cross_image_isolation(runner: Qwen25Runner, first_sample: dict,
                           first_result: dict, out_dir: Path) -> dict:
    directory = out_dir / f"sample_00_{first_sample['image_id']}_{first_sample['question_id']}"
    schedule = ("ours25", "fullload", "ours25", "fullload")
    seen = []
    for method in schedule:
        store_dir = directory / ("repacked_store" if method == "ours25"
                                  else "canonical_store")
        result = runner.run_cache(
            store_dir, first_sample["question"],
            budget_ratio=.25 if method == "ours25" else 1.0,
            image_sha256=first_sample["image_sha256"], return_logits=True)
        baseline = first_result["methods"][method]
        matched = (_logits_hash(result["first_logits"]) ==
                   baseline["first_logits_sha256"]
                   and result["generated_token_ids"] ==
                   baseline["generated_token_ids"]
                   and result["first_token_id"] == baseline["first_token_id"]
                   and result["prediction"] == baseline["prediction"]
                   and result["vision_calls"] == 0)
        seen.append({"method": method, "exact_to_pre_other_image": matched,
                     "first_logits_sha256": _logits_hash(result["first_logits"]),
                     "generated_token_ids": result["generated_token_ids"],
                     "vision_calls": result["vision_calls"]})
    return {"status": "PASS" if all(x["exact_to_pre_other_image"] for x in seen)
            else "FAIL", "method_order": list(schedule),
            "per_request": seen}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = ROOT / "runs/qwen25_correctness_v2_20260928T073413Z"
    parser.add_argument("--manifest", type=Path, default=default_root / "validation_manifest.json")
    parser.add_argument("--out-dir", type=Path, default=default_root / "gpu_validation")
    parser.add_argument("--frozen-inputs", type=Path, default=default_root / "frozen_inputs.json")
    parser.add_argument("--protected-before", type=Path, default=default_root / "protected_before.json")
    parser.add_argument("--llava-regression", type=Path, default=default_root / "llava_cpu_regression.json")
    parser.add_argument("--evidence-manifest", type=Path, default=default_root / "evidence_manifest.json")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    manifest = manifest_preflight(args.manifest.resolve())
    negative = _preflight_negative_fixtures()
    if not negative["passed"]:
        raise RuntimeError("negative semantic fixtures did not fail as expected")
    llava = _llava_gate(args.llava_regression.resolve())
    d3 = _d3_reference(args.evidence_manifest.resolve())
    if not llava["passed"]:
        raise RuntimeError("existing LLaVA CPU regression artifact is not PASS")
    protected_before = json.loads(args.protected_before.read_text(encoding="utf-8"))
    if not protected_before.get("debug_v1"):
        raise RuntimeError("v2 prior-artifact protection baseline is empty")
    freeze = None
    if args.frozen_inputs.is_file():
        freeze = _verify_frozen_inputs(args.frozen_inputs.resolve())
    elif not args.preflight_only:
        raise FileNotFoundError("freeze must exist before GPU validation")
    if args.preflight_only:
        print(json.dumps({
            "status": "PASS" if freeze else "PENDING FREEZE",
            "manifest_sha256": manifest["manifest_sha256"],
            "sample_count": len(manifest["samples"]),
            "negative_fixtures": negative,
            "D3_prior_reference": d3,
            "llava_cpu": {"passed": llava["passed"], "scope": llava["scope"]},
            "protected_before_sha256": sha256_file(args.protected_before),
            "freeze": freeze}, sort_keys=True))
        return 0

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    report_path = out_dir / "validation.json"
    report = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "NOT RUN", "gpu_system_correctness": "NOT RUN",
        "pilot_eligible": False, "run_dir": str(out_dir),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "model_revision": CHECKPOINT_REVISION,
        "frozen_inputs": freeze,
        "frozen_inputs_sha256": freeze["sha256"],
        "source_hashes": {**manifest["source_hashes"],
                          **freeze["source_hashes"]},
        "configuration": manifest["configuration"],
        "tolerances": {"matched_path": "bitwise exact",
                       "legacy_diagnostics_atol": V1_ATOL,
                       "legacy_diagnostics_rtol": V1_RTOL,
                       "fp32_oracle_atol": FP32_ORACLE_ATOL,
                       "fp32_oracle_rtol": FP32_ORACLE_RTOL},
        "negative_fixtures": negative,
        "diagnostic_D3_reference": d3,
        "samples": [],
        "gates": {name: {"status": "NOT RUN"} for name in GATES}}
    atomic_json(report_path, report)
    runner = Qwen25Runner(attn="sdpa")
    import gc
    try:
        runner.load()
        report["runtime"] = runner.runtime_fingerprint()
        atomic_json(report_path, report)
        for index, sample in enumerate(manifest["samples"]):
            start = time.perf_counter()
            try:
                result = validate_sample(runner, sample, out_dir, index)
                result["duration_seconds"] = time.perf_counter() - start
            except Exception as exc:
                result = {
                    "sample_index": index,
                    "image_id": sample["image_id"],
                    "question_id": sample["question_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=20),
                    "duration_seconds": time.perf_counter() - start,
                    "gates": {name: {"status": "UNRESOLVED",
                                     "reason": "sample execution aborted before attribution"}
                              for name in PER_SAMPLE_GATES}}
            report["samples"].append(result)
            if index == 3:
                first = manifest["samples"][0]
                if (sample["image_id"] == first["image_id"] or
                        "methods" not in result or
                        "methods" not in report["samples"][0]):
                    report["cross_image_isolation"] = {
                        "status": "UNRESOLVED",
                        "reason": "different-image predecessor or first baseline unavailable"}
                else:
                    try:
                        report["cross_image_isolation"] = _cross_image_isolation(
                            runner, first, report["samples"][0], out_dir)
                    except Exception as exc:
                        report["cross_image_isolation"] = {
                            "status": "UNRESOLVED",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(limit=12)}
                    finally:
                        _release_sample_stores(runner, 0, first, out_dir)
            _summarize(report)
            atomic_json(report_path, report)
            _release_sample_stores(runner, index, sample, out_dir)
            del result
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        runner.close()

    report["gates"]["G14"] = _gate(
        "PASS" if llava["passed"] else "FAIL", **llava)
    protection = _verify_protected(args.protected_before.resolve())
    report["gates"]["G15"] = _gate(
        "PASS" if protection["passed"] else "FAIL", **protection)
    _summarize(report)
    report["gates"]["G13"]["frozen_inputs_sha256"] = freeze["sha256"]
    report["gates"]["G13"]["frozen_file_count"] = freeze["file_count"]
    atomic_json(report_path, report)
    print(report_path)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


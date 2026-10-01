#!/usr/bin/env python3
"""Fixed-sample GPU gate for original-token contextual Visual-KV25.

The expected selection below is deliberately independent of the production
selector.  The existing KV25 gate supplies only generic cache, attention and
pread observation helpers; its dominant-only gate and layout remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import traceback
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mmimpress.config import CHUNK_SIZE, MODEL_ID  # noqa: E402
from mmimpress.dataset import load_index  # noqa: E402
from mmimpress.model import LlavaRunner, cache_layers  # noqa: E402
from mmimpress.piggyback import (VisionForwardCapture,  # noqa: E402
                                persist_captured_visual_prefix, sha256_file)
from mmimpress.serve import BIAS, ImageContext, Server, suffix_ids_for  # noqa: E402

_old_spec = importlib.util.spec_from_file_location(
    "_frozen_llava_kv25_gpu_gate", ROOT / "scripts/90_validate_llava_kv25.py")
assert _old_spec is not None and _old_spec.loader is not None
OLD = importlib.util.module_from_spec(_old_spec)
_old_spec.loader.exec_module(OLD)

SCHEMA = "llava-contextual-kv25-gpu-correctness-v1"
NEW_LAYOUT = "visionzip_contextual_original_v1"
INDEX_SHA256 = "514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a"
WORKLOAD_SHA256 = "97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66"
FIXED = OLD.FIXED
ATOL = RTOL = 1e-4
SEED = 1234
ARMS = (
    ("d25_c0", "dominant", 0.0),
    ("d22_5_c2_5", "contextual", 0.1),
    ("d20_c5", "contextual", 0.2),
    ("d17_5_c7_5", "contextual", 0.3),
    ("d15_c10", "contextual", 0.4),
    ("d20_random5", "random", 0.2),
    ("d20_uniform5", "uniform", 0.2),
)
SOURCE_PATHS = (
    "mmimpress/cvpr25.py", "mmimpress/contextual_kv25.py",
    "mmimpress/model.py", "mmimpress/piggyback.py", "mmimpress/serve.py",
    "mmimpress/store.py", "scripts/90_validate_llava_kv25.py",
    "scripts/92_validate_llava_contextual_kv25.py",
)


def _anyres_geometry(runner, image_size, v_num: int):
    from transformers.models.llava_next.modeling_llava_next import (
        get_anyres_image_grid_shape, unpad_image)

    vc = runner.cfg.vision_config
    side = vc.image_size // vc.patch_size
    size = [int(v) for v in torch.as_tensor(image_size).reshape(-1).tolist()]
    ph, pw = get_anyres_image_grid_shape(
        size, runner.cfg.image_grid_pinpoints, vc.image_size)
    marker = torch.ones((1, ph * side, pw * side), dtype=torch.float32)
    kept = unpad_image(marker, size)[0]
    height, width = map(int, kept.shape)
    assert side * side + height * (width + 1) == v_num
    separators = [side * side + y * (width + 1) + width
                  for y in range(height)]
    return side, ph, pw, height, width, size, separators, unpad_image


def _map_scores_and_keys(runner, scores: torch.Tensor, keys: torch.Tensor,
                         image_size, v_num: int):
    """Independent AnyRes tiling, CLS removal already done by the capture."""
    side, ph, pw, height, width, size, separators, unpad = _anyres_geometry(
        runner, image_size, v_num)
    score = torch.as_tensor(scores).detach().cpu().float()
    key = torch.as_tensor(keys).detach().cpu().float()
    assert tuple(score.shape) == (1 + ph * pw, side * side)
    assert key.ndim == 3 and tuple(key.shape[:2]) == tuple(score.shape)
    assert torch.isfinite(score).all() and torch.isfinite(key).all()
    norms = torch.linalg.vector_norm(key, dim=-1)
    assert bool(torch.all((norms == 0) | torch.isclose(
        norms, torch.ones_like(norms), atol=2e-5, rtol=2e-5)))

    local_scores = np.full(v_num, np.inf, dtype=np.float32)
    local_keys = np.zeros((v_num, int(key.shape[-1])), dtype=np.float32)
    local_scores[:side * side] = score[0].numpy()
    local_keys[:side * side] = key[0].numpy()
    score_grid = score[1:].reshape(ph, pw, side, side).permute(
        0, 2, 1, 3).reshape(ph * side, pw * side)
    key_grid = key[1:].reshape(ph, pw, side, side, key.shape[-1]).permute(
        0, 2, 1, 3, 4).reshape(ph * side, pw * side, key.shape[-1])
    score_crop = unpad(score_grid.unsqueeze(0), size)[0].numpy()
    key_crop = unpad(key_grid.permute(2, 0, 1), size).permute(
        1, 2, 0).numpy()
    assert score_crop.shape == (height, width)
    assert key_crop.shape == (height, width, key.shape[-1])
    for y in range(height):
        start = side * side + y * (width + 1)
        local_scores[start:start + width] = score_crop[y]
        local_keys[start:start + width] = key_crop[y]
    return local_scores, local_keys, separators


def _independent_plan(scores: np.ndarray, keys: np.ndarray,
                      separators: list[int], alpha: float, variant: str,
                      image_id: str) -> dict:
    """Literal experiment protocol, with no production selection imports."""
    v_num = len(scores)
    sep_set = set(separators)
    content = [i for i in range(v_num) if i not in sep_set]
    assert len(sep_set) == len(separators)
    assert all(0 <= i < v_num for i in separators)
    assert np.isfinite(scores[content]).all()
    assert np.isfinite(keys).all()
    n = len(content)
    assert n > 0
    k = (n + 3) // 4
    c = 0 if variant == "dominant" else math.floor(alpha * k)
    d = k - c
    ranked = sorted(content, key=lambda i: (-float(scores[i]), i))
    dominant = ranked[:d]
    dominant_set = set(dominant)
    remaining = [i for i in content if i not in dominant_set]
    targets: list[int] = []
    assignments: dict[int, int] = {}
    representatives: list[int] = []
    cluster_sizes: list[int] = []
    if c:
        m = len(remaining)
        targets = [remaining[((2 * j + 1) * m) // (2 * c)]
                   for j in range(c)]
        assert len(set(targets)) == c
        if variant == "contextual":
            # Repeat the protocol's FP32 cosine operations locally.  The
            # production plan function and its target/cluster helpers are not
            # called; only the captured, independently mapped vectors enter.
            vectors = torch.as_tensor(keys, dtype=torch.float32).cpu()
            lengths = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
            vectors = vectors / lengths.clamp_min(torch.finfo(torch.float32).tiny)
            similarities = vectors[remaining] @ vectors[targets].T
            clusters = {target: [] for target in targets}
            target_set = set(targets)
            for row, member in enumerate(remaining):
                target = (member if member in target_set else
                          targets[int(torch.argmax(similarities[row]).item())])
                clusters[target].append(member)
                assignments[member] = target
            for target in targets:
                members = clusters[target]
                centroid = vectors[members].mean(dim=0)
                norm = float(torch.linalg.vector_norm(centroid).item())
                representative = (target if norm == 0 else members[int(
                    torch.argmax(vectors[members] @ (centroid / norm)).item())])
                representatives.append(representative)
                cluster_sizes.append(len(members))
        elif variant == "uniform":
            representatives = targets.copy()
        elif variant == "random":
            seed_blob = hashlib.sha256(
                f"{SEED}\0{image_id}".encode("utf-8")).digest()
            derived = int.from_bytes(seed_blob[:8], "big")
            representatives = sorted(random.Random(derived).sample(
                remaining, c))
        else:
            raise ValueError(variant)
    assert len(representatives) == c
    selected_set = dominant_set | set(representatives)
    assert len(selected_set) == k and not dominant_set.intersection(
        representatives)
    prefix = sorted(selected_set, key=lambda i: (-float(scores[i]), i))
    suffix = [i for i in ranked if i not in selected_set]
    order = prefix + suffix + separators
    assert sorted(order) == list(range(v_num))
    inverse = [0] * v_num
    for stored, original in enumerate(order):
        inverse[original] = stored
    return {"k": k, "k_dominant": d, "k_context": c,
            "n_content": n, "dominant_ids": dominant,
            "contextual_ids": representatives,
            "target_ids": targets if variant != "random" else [],
            "assignments": assignments, "representative_ids": representatives,
            "cluster_sizes": cluster_sizes, "selected_original_ids": prefix,
            "stored_to_original": order, "original_to_stored": inverse}


def _assert_plan(meta: dict, artifact: dict, expected: dict, variant: str,
                 alpha: float) -> None:
    assert meta["physical_layout"] == NEW_LAYOUT
    assert meta["layout_policy_version"] == NEW_LAYOUT
    assert meta["selection_variant"] == variant
    assert math.isclose(float(meta["contextual_alpha"]), alpha, abs_tol=1e-12)
    assert int(meta["k_target"]) == expected["k"]
    assert int(meta["k_dominant"]) == expected["k_dominant"]
    assert int(meta["k_context"]) == expected["k_context"]
    assert [int(x) for x in meta["order"]] == expected["stored_to_original"]
    plan = artifact["selection_plan"]
    for field in ("k", "k_dominant", "k_context", "n_content",
                  "dominant_ids", "contextual_ids", "selected_original_ids",
                  "stored_to_original", "original_to_stored"):
        assert plan[field] == expected[field], f"selection plan {field}"
    if variant in ("contextual", "uniform"):
        assert plan["target_ids"] == expected["target_ids"]
    if variant == "contextual":
        assert {int(k): int(v) for k, v in plan["assignments"].items()} == \
            expected["assignments"]
        assert plan["representative_ids"] == expected["representative_ids"]
        assert plan["cluster_sizes"] == expected["cluster_sizes"]


def _verify_inverse_bits(store: Path, meta: dict, source_layers,
                         inverse: list[int]) -> None:
    """Verify all original FP16 KV rows, not just selected prefix rows."""
    v_start, v_num = int(meta["v_token_start"]), int(meta["v_token_num"])
    heads, dim = int(meta["num_heads"]), int(meta["head_dim"])
    inverse_arr = np.asarray(inverse, dtype=np.int64)
    for layer, (src_k, src_v) in enumerate(source_layers):
        for kind, source in (("k", src_k), ("v", src_v)):
            physical = np.memmap(store / f"layer_{layer:02d}" /
                                 f"{kind}.bin", dtype=np.float16, mode="r",
                                 shape=(v_num, heads, dim))
            original = source[0, :, v_start:v_start + v_num].permute(
                1, 0, 2).to(torch.float16).cpu().contiguous().numpy()
            assert np.array_equal(physical[inverse_arr].view(np.uint16),
                                  original.view(np.uint16)), (
                f"inverse permutation changed layer {layer} {kind} bits")
            del physical


def _compare_outputs(production: dict, reference: dict) -> dict:
    assert production["logits"] and reference["logits"]
    lhs, rhs = production["logits"][0], reference["logits"][0]
    logits_match = torch.allclose(lhs, rhs, atol=ATOL, rtol=RTOL)
    ids_match = (production["generated_token_ids"] ==
                 reference["generated_token_ids"])
    first_match = production["first_token_id"] == reference["first_token_id"]
    answer_match = production["answer"] == reference["answer"]
    assert logits_match and ids_match and first_match and answer_match
    return {"first_step_logits_within_frozen_tolerance": True,
            "max_abs_first_step_logit_delta": float((lhs-rhs).abs().max()),
            "first_token_exact": True, "generated_token_ids_exact": True,
            "prediction_exact": True, "atol": ATOL, "rtol": RTOL}


def _run_new_with_trace(server, ctx, question: str, source_layers,
                        content_order: list[int], k: int) -> dict:
    original_decode = server._decode
    box: dict = {"logits": [], "positions": []}

    def output_hook(module, args, output):
        if hasattr(output, "logits"):
            box["logits"].append(output.logits[0, -1].detach().float().cpu())

    def position_hook(module, args, kwargs):
        positions = kwargs["position_ids"].detach().cpu().reshape(-1)
        cache_positions = kwargs["cache_position"].detach().cpu().reshape(-1)
        qlen = int(kwargs["input_ids"].shape[1])
        total = int(kwargs["attention_mask"].shape[-1])
        expected = torch.arange(total - qlen, total)
        assert torch.equal(positions, expected)
        assert torch.equal(cache_positions, expected)
        box["positions"].append((int(expected[0]), int(expected[-1])))

    def traced_decode(cache, suffix, prefix_len):
        box["installed"] = OLD._compare_installed(
            ctx.meta, cache, source_layers, content_order, k)
        handle = server.runner.model.register_forward_hook(output_hook)
        pos_handle = server.runner.model.register_forward_pre_hook(
            position_hook, with_kwargs=True)
        try:
            with OLD._AttentionTrace(ctx.meta, k) as masks:
                decoded = original_decode(cache, suffix, prefix_len)
            box["attention"] = masks.summary()
            return decoded
        finally:
            handle.remove()
            pos_handle.remove()

    server._decode = traced_decode
    vision_calls = []

    def forbidden_vision(module, args, kwargs):
        vision_calls.append(1)
        raise AssertionError("cache hit invoked vision tower")

    vision_handle = server.runner.model.model.vision_tower.register_forward_pre_hook(
        forbidden_vision, with_kwargs=True)
    try:
        forbidden_score = AssertionError("cache hit recomputed image selection")
        with mock.patch("mmimpress.contextual_kv25.build_selection_plan",
                        side_effect=forbidden_score), \
             mock.patch("mmimpress.cvpr25.clip_cls_patch_saliency",
                        side_effect=forbidden_score), \
             OLD._ReadTrace() as read_trace:
            result = server.request_cvpr25(
                ctx, question=question, static=None, mode="prefix",
                budget=.25, budget_unit="visual_kv", sep_policy="sidecar",
                cold=False, expected_prefix_layout=NEW_LAYOUT)
    finally:
        vision_handle.remove()
        server._decode = original_decode
        BIAS.clear()
    assert not vision_calls
    assert result["selected_original_ids"] == content_order[:k]
    assert result["selected_stored_ids"] == list(range(k))
    assert result["attended_content_kv_count"] == k
    assert result["query_score_calls"] == 0
    assert result["static_score_calls"] == 0
    assert result["diversity_calls"] == 0
    assert len(box["logits"]) == len(result["generated_token_ids"])
    assert len(box["positions"]) == len(box["logits"])
    box["reads"] = OLD._check_reads(ctx, read_trace, math.ceil(k / CHUNK_SIZE))
    assert not any("descriptor" in path for path in read_trace.opens)
    assert result["io"]["bytes"] == box["reads"]["actual_returned_bytes"]
    assert result["io"]["preads"] == box["reads"]["actual_pread_calls"]
    box["result"] = result
    return box


def _persist_one(runner, canonical, enc, capture, out_dir: Path,
                 image_id: str, variant: str | None, alpha: float):
    options = {} if variant is None else {
        "selection_variant": variant, "contextual_alpha": alpha,
        "selection_seed": SEED,
    }
    if variant == "contextual" and alpha > 0:
        options["vision_key_descriptors"] = capture.result_keys_cpu()
    return persist_captured_visual_prefix(
        runner, canonical, enc["input_ids"], enc["image_sizes"][0],
        capture.result_cpu(), out_dir, image_id=image_id,
        model_id=runner.model_id, chunk_size=CHUNK_SIZE,
        capture_stats=capture, full_integrity_hash=True,
        extra_metadata={"dataset": "gqa", "source_turn_id": 1,
                        "validation_only": True}, **options)


def _cleanup_gate_store(receipt: dict, receipt_path: Path, store: Path,
                        root: Path, keep: bool) -> None:
    if keep:
        return
    assert store.is_dir() and store.is_relative_to(root)
    receipt["cleanup"].append({"store": str(store), "status": "started",
                               "scope": "new_gpu_gate_store_only"})
    OLD._atomic_json(receipt_path, receipt)
    shutil.rmtree(store)
    receipt["cleanup"][-1]["status"] = "removed"
    OLD._atomic_json(receipt_path, receipt)


def _sample(runner, server, entry, question_id: str, store_root: Path,
            receipt: dict, receipt_path: Path, keep_stores: bool) -> dict:
    image_id = str(entry["image_id"])
    assert str(entry["questions"][4]["question_id"]) == question_id
    question = str(entry["questions"][4]["question"])
    with Image.open(ROOT / entry["image_path"]) as source:
        image = source.convert("RGB")
    enc = runner.encode(image, question)
    v_start, v_num = runner.visual_span(enc["input_ids"])
    plain = server.recompute(runner.to_device(enc),
                             return_past_key_values=False)
    capture = VisionForwardCapture(runner, capture_saliency=True,
                                   capture_keys=True)
    with capture:
        hooked = server.recompute(runner.to_device(enc),
                                  return_past_key_values=True)
    assert plain["first_token_id"] == hooked["first_token_id"]
    assert plain["generated_token_ids"] == hooked["generated_token_ids"]
    assert plain["answer"] == hooked["answer"]
    capture_stats = capture.stats()
    assert capture_stats["vision_call_count"] == 1
    assert capture_stats["saliency_call_count"] == 1
    assert capture_stats["capture_keys"] is True
    assert capture_stats["key_call_count"] == 1
    assert capture_stats["key_head_reduction"] == "fp32_head_mean"
    assert capture_stats["key_normalization"] == "fp32_l2_zero_stays_zero"
    assert capture_stats["extra_vision_forward_calls"] == 0
    canonical = hooked.pop("captured_past_key_values")
    source_layers = cache_layers(canonical)
    scores, keys, separators = _map_scores_and_keys(
        runner, capture.result_cpu(), capture.result_keys_cpu(),
        enc["image_sizes"][0], v_num)
    n = v_num - len(separators)
    assert n > 0 and v_start >= 0
    assert OLD._score_rank(runner, capture.result_cpu(),
                           enc["image_sizes"][0], v_num, separators) == sorted(
        [i for i in range(v_num) if i not in set(separators)],
        key=lambda i: (-float(scores[i]), i))
    sample = {"image_id": image_id, "question_id": question_id,
              "T1_hook_output_preserved": True,
              "T1_capture": capture_stats, "N_content": n,
              "N_structural": len(separators), "arms": {},
              "gate_uses_one_canonical_T1_KV_for_all_arms": True}
    receipt["samples"].append(sample)
    OLD._atomic_json(receipt_path, receipt)
    sample_root = store_root / image_id
    sample_root.mkdir(parents=True, exist_ok=False)
    dominant_production = None
    for arm, variant, alpha in ARMS:
        expected = _independent_plan(scores, keys, separators, alpha,
                                     variant, image_id)
        store = sample_root / arm
        predicted_bytes = (2 * len(source_layers) * v_num *
                           int(source_layers[0][0].shape[1]) *
                           int(source_layers[0][0].shape[3]) * 2)
        assert shutil.disk_usage(store_root).free - predicted_bytes > 30 * 1024**3
        persisted = _persist_one(runner, canonical, enc, capture, store,
                                 image_id, variant, alpha)
        assert persisted["integrity"]["ok"]
        ctx = ImageContext(store, runner.model.device, require_v_hidden=False)
        try:
            artifact = torch.load(store / "visionzip_layout.pt",
                                  map_location="cpu", weights_only=True)
            _assert_plan(ctx.meta, artifact, expected, variant, alpha)
            ctx.validate_contextual_visual_kv_layout()
            _verify_inverse_bits(store, ctx.meta, source_layers,
                                 expected["original_to_stored"])
            content_order = expected["stored_to_original"][:n]
            memory = OLD._memory_reference(
                server, ctx, question, source_layers, content_order,
                expected["k"])
            production = _run_new_with_trace(
                server, ctx, question, source_layers, content_order,
                expected["k"])
            matched = _compare_outputs(
                production["result"] | {"logits": production["logits"]}, memory)
            alternate = _run_new_with_trace(
                server, ctx, str(entry["questions"][5]["question"]),
                source_layers, content_order, expected["k"])
            assert alternate["result"]["selected_original_ids"] == \
                expected["selected_original_ids"]
            assert alternate["result"]["io"]["bytes"] == \
                production["result"]["io"]["bytes"]
            revisited = _run_new_with_trace(
                server, ctx, question, source_layers, content_order,
                expected["k"])
            repeat = _compare_outputs(
                revisited["result"] | {"logits": revisited["logits"]},
                production["result"] | {"logits": production["logits"]})
            if arm == "d25_c0":
                dominant_production = production["result"] | {
                    "logits": production["logits"]}
            arm_result = {
                "status": "PASS", "variant": variant, "alpha": alpha,
                "k": expected["k"], "k_dominant": expected["k_dominant"],
                "k_context": expected["k_context"],
                "selected_original_ids": expected["selected_original_ids"],
                "selection_independent": True, "inverse_full_kv_bits": True,
                "matched_memory_vs_ssd": matched,
                "cache_bits_and_mask": production["installed"],
                "combined_attention": production["attention"],
                "position_checks": production["positions"],
                "physical_reads": production["reads"],
                "alternate_question_same_selection": True,
                "interleaved_repeat_no_state_leak": repeat,
                "hit_vision_forward_count": 0,
                "hit_online_score_calls": 0,
                "actual_read_bytes": production["result"]["io"]["bytes"],
                "actual_preads": production["result"]["io"]["preads"],
                "store_dir": str(store),
                "store_hashes": persisted["hashes"],
            }
            sample["arms"][arm] = arm_result
            OLD._atomic_json(receipt_path, receipt)
        finally:
            ctx.close()
        _cleanup_gate_store(receipt, receipt_path, store, store_root,
                            keep_stores)

    # Every selective arm buys the same number of physical prefix chunks.
    assert len({row["actual_read_bytes"] for row in sample["arms"].values()}) == 1
    assert len({row["actual_preads"] for row in sample["arms"].values()}) == 1
    sample["same_actual_IO_all_selective_arms"] = True
    OLD._atomic_json(receipt_path, receipt)

    # Direct regression against the untouched dominant-only writer and gate.
    assert dominant_production is not None
    old_store = sample_root / "legacy_dominant_only"
    old_persisted = _persist_one(runner, canonical, enc, capture, old_store,
                                 image_id, None, 0.0)
    assert old_persisted["integrity"]["ok"]
    old_ctx = ImageContext(old_store, runner.model.device,
                           require_v_hidden=False)
    try:
        old_ctx.validate_visual_kv_layout()
        expected = _independent_plan(scores, keys, separators, 0.0,
                                     "dominant", image_id)
        assert old_ctx.meta["order"] == expected["stored_to_original"]
        old = OLD._run_with_trace(
            server, old_ctx, question, source_layers,
            expected["stored_to_original"][:n], expected["k"],
            budget=.25, budget_unit="visual_kv")
        regression = _compare_outputs(dominant_production,
                                      old["result"] | {"logits": old["logits"]})
        assert dominant_production["selected_original_ids"] == \
            old["result"]["selected_original_ids"]
        assert dominant_production["io"]["bytes"] == \
            old["result"]["io"]["bytes"]
        sample["D25_legacy_regression"] = regression
    finally:
        old_ctx.close()
    OLD._atomic_json(receipt_path, receipt)
    _cleanup_gate_store(receipt, receipt_path, old_store, store_root,
                        keep_stores)
    return sample


def _verify_source_freeze(run_dir: Path, sources: dict[str, str]) -> dict:
    freeze_path = run_dir / "source_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected_files = {
        "protocol_sha256": "PROTOCOL.md",
        "config_sha256": "config.json",
        "workload_manifest_sha256": "workload_manifest.json",
        "source_diff_sha256": "source.diff",
        "protected_before_sha256": "protected_before.json",
    }
    observed = {key: sha256_file(run_dir / relative)
                for key, relative in expected_files.items()}
    for key, digest in observed.items():
        assert freeze.get(key) == digest, f"frozen {key} disagrees with file"
    frozen_sources = freeze["source_sha256"]
    for relative, digest in sources.items():
        assert frozen_sources.get(relative) == digest, (
            f"source changed after freeze: {relative}")
    return {"source_freeze_sha256": sha256_file(freeze_path), **observed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--keep-stores", action="store_true")
    args = parser.parse_args()
    assert args.max_new_tokens == 16, "frozen gate uses 16 greedy tokens"
    run_dir = args.run_dir.resolve()
    assert run_dir.is_dir(), f"frozen run directory is missing: {run_dir}"
    receipt_path = run_dir / "gpu_validation.json"
    store_root = run_dir / "gpu_validation_stores"
    assert not receipt_path.exists(), f"refusing to replace {receipt_path}"
    source_hashes = {path: sha256_file(ROOT / path) for path in SOURCE_PATHS}
    receipt = {
        "schema_version": SCHEMA, "GPU_CORRECTNESS": "NOT RUN",
        "model_id": MODEL_ID, "chunk_size": CHUNK_SIZE,
        "index_sha256": INDEX_SHA256, "workload_sha256": WORKLOAD_SHA256,
        "fixed_samples": [list(pair) for pair in FIXED],
        "arms": [{"id": arm, "variant": variant, "alpha": alpha}
                 for arm, variant, alpha in ARMS],
        "source_sha256": source_hashes,
        "first_step_fp32_logits_tolerance": {"atol": ATOL, "rtol": RTOL},
        "reference": "independent local AnyRes mapping, clustering and original-token gather",
        "cleanup_rule": "only new stores in this run's gpu_validation_stores, after receipt fsync",
        "keep_stores": bool(args.keep_stores), "cleanup": [], "samples": [],
    }
    OLD._atomic_json(receipt_path, receipt)
    try:
        receipt["freeze_hashes"] = _verify_source_freeze(run_dir, source_hashes)
        index = ROOT / "data/index.json"
        assert sha256_file(index) == INDEX_SHA256
        entries = load_index(index)
        assert len(entries) == 40
        assert all(len(entry["questions"][4:10]) == 6 for entry in entries)
        workload_blob = "\n".join(
            f"{entry['image_id']}\t{question['question_id']}"
            for entry in entries for question in entry["questions"][4:10]
        ).encode("utf-8")
        assert hashlib.sha256(workload_blob).hexdigest() == WORKLOAD_SHA256
        pair_map = {str(entry["image_id"]): entry for entry in entries}
        fixed_entries = [pair_map[image_id] for image_id, _ in FIXED]
        assert all(str(entry["questions"][4]["question_id"]) == question_id
                   for entry, (_, question_id) in zip(fixed_entries, FIXED))
        store_root.mkdir(exist_ok=False)
        OLD._atomic_json(receipt_path, receipt)
        runner = LlavaRunner().load()
        server = Server(runner, max_new_tokens=args.max_new_tokens)
        for (image_id, question_id), entry in zip(FIXED, fixed_entries):
            _sample(runner, server, entry, question_id, store_root,
                    receipt, receipt_path, args.keep_stores)
            print(f"GPU contextual correctness {image_id}: PASS", flush=True)
        assert len(receipt["samples"]) == len(FIXED)
        assert all(len(row["arms"]) == len(ARMS) for row in receipt["samples"])
        receipt["GPU_CORRECTNESS"] = "PASS"
    except BaseException as exc:
        receipt["GPU_CORRECTNESS"] = "FAIL"
        receipt["failure"] = {"type": type(exc).__name__, "message": str(exc),
                              "traceback": traceback.format_exc()}
        OLD._atomic_json(receipt_path, receipt)
        raise
    OLD._atomic_json(receipt_path, receipt)
    print(receipt_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

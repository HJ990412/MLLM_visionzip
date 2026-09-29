#!/usr/bin/env python3
"""Independent, fixed-sample LLaVA visual-KV 25% GPU correctness gate.

The reference ranks scores captured during an ordinary pixel request, gathers
KV from that request's canonical cache, and builds a full-size physical cache
without calling the production selection or mask helpers.  Stores and the
receipt are new run-local artifacts; existing stores are only read if used by
other experiments, never by this script.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mmimpress.config import CHUNK_SIZE  # noqa: E402
from mmimpress.dataset import load_index  # noqa: E402
from mmimpress.model import LlavaRunner, cache_layers  # noqa: E402
from mmimpress.piggyback import (VisionForwardCapture,  # noqa: E402
                                persist_captured_visual_prefix,
                                sha256_file)
from mmimpress.serve import BIAS, ImageContext, Server, suffix_ids_for  # noqa: E402
import mmimpress.serve as serving  # noqa: E402


FIXED = (
    ("n355567", "201751701"),
    ("n9181", "20929611"),
    ("n390187", "201861403"),
    ("n133585", "202108008"),
    ("n272098", "201535625"),
)
INDEX_SHA256 = "514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a"
ATOL = 1e-4
RTOL = 1e-4
SENTINEL = 10000.0  # finite BF16 value; NaN/Inf are invalid mask probes


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _fp16_roundtrip(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Match writer FP16 payload followed by PrefixCache BF16 restoration."""
    return value.detach().to(dtype=torch.float16, device="cpu").to(
        dtype=torch.bfloat16, device=device)


def _score_rank(runner, captured_scores: torch.Tensor, image_size,
                v_num: int, separators: list[int]) -> list[int]:
    """Map raw CLIP scores and stable-rank real original IDs independently.

    The geometry uses the installed LLaVA AnyRes primitives, while tiling,
    score placement and ranking are local and do not use cvpr25 helpers or the
    store's saved permutation.
    """
    from transformers.models.llava_next.modeling_llava_next import (
        get_anyres_image_grid_shape, unpad_image)

    vc = runner.cfg.vision_config
    side = vc.image_size // vc.patch_size
    size = [int(x) for x in torch.as_tensor(image_size).reshape(-1).tolist()]
    ph, pw = get_anyres_image_grid_shape(
        size, runner.cfg.image_grid_pinpoints, vc.image_size)
    raw = captured_scores.detach().cpu().float().numpy()
    assert raw.shape == (1 + ph * pw, side * side), raw.shape
    tiled = np.transpose(raw[1:].reshape(ph, pw, side, side),
                         (0, 2, 1, 3)).reshape(ph * side, pw * side)
    cropped = unpad_image(torch.from_numpy(tiled)[None], size)[0].numpy()
    height, width = cropped.shape
    assert side * side + height * (width + 1) == v_num
    local_scores = np.empty(v_num, dtype=np.float32)
    local_scores[:side * side] = raw[0]
    expected_separators = []
    for row in range(height):
        first = side * side + row * (width + 1)
        local_scores[first:first + width] = cropped[row]
        expected_separators.append(first + width)
    assert expected_separators == separators
    excluded = set(separators)
    real_ids = [i for i in range(v_num) if i not in excluded]
    assert np.isfinite(local_scores[real_ids]).all()
    return sorted(real_ids, key=lambda i: (-float(local_scores[i]), i))


def _expected_visible(meta: dict, k: int) -> torch.Tensor:
    vstart, n, v = (int(meta["v_token_start"]), int(meta["n_spatial"]),
                    int(meta["v_token_num"]))
    expected = torch.zeros(vstart + v, dtype=torch.bool)
    expected[:vstart + k] = True
    expected[vstart + n:vstart + v] = True
    return expected


def _make_memory_cache(runner, meta: dict, capture_layers, ranking: list[int],
                       k: int):
    """Construct installed DynamicCache shape/order from original captured KV."""
    from transformers import DynamicCache

    device = runner.model.device
    p, h, hd = (int(meta["prefix_len"]), int(meta["num_heads"]),
                int(meta["head_dim"]))
    vstart, n = int(meta["v_token_start"]), int(meta["n_spatial"])
    separators = [int(x) for x in meta["newline_idx"]]
    physical = ranking + separators
    assert len(physical) == int(meta["v_token_num"])
    cache = DynamicCache()
    scratch = torch.zeros((1, h, p, hd), dtype=torch.bfloat16, device=device)
    for li, (src_k, src_v) in enumerate(capture_layers):
        cache.update(scratch, scratch, li)
        for name, source in (("keys", src_k), ("values", src_v)):
            target = getattr(cache.layers[li], name)
            target[0, :, :vstart] = _fp16_roundtrip(
                source[0, :, :vstart], device)
            if k:
                original = torch.as_tensor(ranking[:k], device=source.device)
                selected = source[0, :, vstart + original, :]
                target[0, :, vstart:vstart + k, :] = _fp16_roundtrip(
                    selected, device)
            if separators:
                original = torch.as_tensor(separators, device=source.device)
                selected = source[0, :, vstart + original, :]
                target[0, :, vstart + n:vstart + n + len(separators), :] = (
                    _fp16_roundtrip(selected, device))
    return cache


def _install_reference_bias(meta: dict, k: int, layers: int, device) -> None:
    """Independent additive mask with the production cache dimensions."""
    BIAS.clear()
    enabled = _expected_visible(meta, k).to(device)
    base = torch.where(enabled, torch.zeros((), dtype=torch.bfloat16,
                                            device=device),
                       torch.full((), torch.finfo(torch.bfloat16).min,
                                  dtype=torch.bfloat16, device=device))
    for li in range(layers):
        BIAS[li] = base.view(1, 1, 1, -1).clone()


def _compare_installed(meta: dict, cache, capture_layers, ranking: list[int],
                       k: int, *, check_full: bool = False) -> dict:
    """Exact all-layer/head K/V and mask check before model forward."""
    p, n, vstart = (int(meta["prefix_len"]), int(meta["n_spatial"]),
                    int(meta["v_token_start"]))
    separators = [int(x) for x in meta["newline_idx"]]
    expected_enabled = _expected_visible(meta, k)
    assert len(cache.layers) == len(capture_layers) == int(meta["num_layers"])
    assert len(BIAS) == len(capture_layers)
    counts = []
    for li, (source_k, source_v) in enumerate(capture_layers):
        bias = BIAS[li]
        assert tuple(bias.shape) == (1, 1, 1, p)
        enabled = (bias[0, 0, 0].detach().cpu() == 0)
        assert torch.equal(enabled, expected_enabled), f"layer {li} keep mask"
        assert int(enabled[vstart:vstart + n].sum()) == k
        assert int(enabled[vstart + n:p].sum()) == len(separators)
        counts.append(int(enabled.sum()))
        for kind, source in (("keys", source_k), ("values", source_v)):
            actual = getattr(cache.layers[li], kind)
            assert tuple(actual.shape) == (1, int(meta["num_heads"]), p,
                                           int(meta["head_dim"]))
            assert actual.dtype == torch.bfloat16
            system = _fp16_roundtrip(source[0, :, :vstart], actual.device)
            assert torch.equal(actual[0, :, :vstart], system), (
                f"layer {li} {kind} system bits")
            if k:
                ids = torch.as_tensor(ranking[:k], device=source.device)
                ref = _fp16_roundtrip(source[0, :, vstart + ids], actual.device)
                assert torch.equal(actual[0, :, vstart:vstart + k], ref), (
                    f"layer {li} {kind} selected bits")
            if separators:
                ids = torch.as_tensor(separators, device=source.device)
                ref = _fp16_roundtrip(source[0, :, vstart + ids], actual.device)
                assert torch.equal(actual[0, :, vstart + n:p], ref), (
                    f"layer {li} {kind} separator bits")
            if check_full:
                ids = torch.as_tensor(ranking, device=source.device)
                ref = _fp16_roundtrip(source[0, :, vstart + ids], actual.device)
                assert torch.equal(actual[0, :, vstart:vstart + n], ref)
    return {"all_layer_head_kv_bits": True, "all_layer_keep_mask": True,
            "keep_count_per_layer": counts,
            "system_and_separator_bits": True,
            "canonical_full_logical_bits": bool(check_full)}


class _ReadTrace:
    def __init__(self):
        self.calls = []
        self.opens = []

    def __enter__(self):
        self.original_pread, self.original_open = os.pread, os.open

        def open_record(path, flags, *args, **kwargs):
            fd = self.original_open(path, flags, *args, **kwargs)
            self.opens.append(str(path))
            return fd

        def pread_record(fd, length, offset):
            try:
                path = os.readlink(f"/proc/self/fd/{fd}")
            except OSError:
                path = f"fd:{fd}"
            data = self.original_pread(fd, length, offset)
            self.calls.append({"path": path, "offset": int(offset),
                               "requested": int(length), "returned": len(data)})
            return data

        os.open, os.pread = open_record, pread_record
        return self

    def __exit__(self, *exc):
        os.open, os.pread = self.original_open, self.original_pread


class _AttentionTrace:
    """Observe actual eager masks after the production prefix bias is added."""

    def __init__(self, meta: dict, k: int):
        self.expected_prefix = _expected_visible(meta, k)
        self.prefix_len = int(meta["prefix_len"])
        self.layers = int(meta["num_layers"])
        self.calls = 0
        self.prefill_calls = 0
        self.decode_calls = 0
        self.seen_layers = set()

    def __enter__(self):
        self.original = serving._ORIG_EAGER

        def checked(module, query, key, value, attention_mask, **kwargs):
            layer = int(module.layer_idx)
            qlen, klen = int(query.shape[-2]), int(key.shape[-2])
            assert attention_mask is not None, "missing combined eager mask"
            mask = attention_mask.detach().cpu()
            assert mask.ndim == 4 and mask.shape[0] == 1
            assert mask.shape[-2:] == (qlen, klen)
            assert klen >= self.prefix_len + qlen
            allowed = mask == 0
            for head in range(mask.shape[1]):
                assert torch.equal(allowed[0, head, :, :self.prefix_len],
                                   self.expected_prefix.expand(qlen, -1)), (
                    f"layer {layer} prefix mask at qlen {qlen}")
                suffix_len = klen - self.prefix_len
                query_positions = torch.arange(klen - qlen, klen)[:, None]
                suffix_positions = torch.arange(
                    self.prefix_len, self.prefix_len + suffix_len)[None, :]
                expected_suffix = suffix_positions <= query_positions
                assert torch.equal(allowed[0, head, :, self.prefix_len:],
                                   expected_suffix), (
                    f"layer {layer} future text causal mask")
            self.calls += 1
            self.seen_layers.add(layer)
            if qlen > 1:
                self.prefill_calls += 1
            else:
                self.decode_calls += 1
            return self.original(module, query, key, value, attention_mask,
                                 **kwargs)

        serving._ORIG_EAGER = checked
        return self

    def __exit__(self, *exc):
        serving._ORIG_EAGER = self.original

    def summary(self):
        assert self.seen_layers == set(range(self.layers))
        assert self.prefill_calls == self.layers
        return {"combined_prefill_and_decode_masks_exact": True,
                "attention_calls": self.calls,
                "prefill_layer_calls": self.prefill_calls,
                "decode_layer_calls": self.decode_calls,
                "layers_observed": len(self.seen_layers),
                "system_kept": True, "future_text_masked": True,
                "padding_absent": True}


def _check_reads(ctx, trace: _ReadTrace, m: int) -> dict:
    meta = ctx.meta
    rows = min(m * CHUNK_SIZE, int(meta["v_token_num"]))
    row_bytes = int(meta["num_heads"]) * int(meta["head_dim"]) * 2
    expected = {}
    for li in range(int(meta["num_layers"])):
        for kind in ("k", "v"):
            path = str(ctx.dir / f"layer_{li:02d}" / f"{kind}.bin")
            expected[path] = (0, rows * row_bytes)
    sep = str(ctx.dir / "sep_kv.bin")
    expected[sep] = (0, (ctx.dir / "sep_kv.bin").stat().st_size)
    actual = {row["path"]: (row["offset"], row["returned"])
              for row in trace.calls}
    assert len(trace.calls) == len(expected), (
        f"pread count {len(trace.calls)} != {len(expected)}")
    assert len(actual) == len(expected) and actual == expected, (
        "actual os.pread ranges/bytes differ from independent file-size plan")
    assert all(row["requested"] == row["returned"] for row in trace.calls)
    forbidden = ("probe_k.bin", "visionzip_layout.pt", "static.pt", "score")
    assert not any(any(x in path for x in forbidden)
                   for path in trace.opens + [c["path"] for c in trace.calls])
    return {"actual_pread_calls": len(trace.calls),
            "actual_returned_bytes": sum(x["returned"] for x in trace.calls),
            "normal_payload_bytes": 2 * int(meta["num_layers"]) * rows * row_bytes,
            "separator_bytes": expected[sep][1],
            "read_ranges_exact": True,
            "forbidden_sidecar_reads": 0,
            "calls": trace.calls}


def _run_with_trace(server, ctx, question, capture_layers, ranking, k,
                    *, budget: float, budget_unit: str, sentinel: bool = False,
                    inspect: bool = True):
    """Call public serving API and observe actual cache, masks, positions, IO."""
    original_decode = server._decode
    box = {"logits": [], "position_checks": []}
    vstart, n = int(ctx.meta["v_token_start"]), int(ctx.meta["n_spatial"])
    row_end = min(math.ceil(k / CHUNK_SIZE) * CHUNK_SIZE, n)

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
        assert bool(kwargs["attention_mask"].all())
        box["position_checks"].append({
            "query_length": qlen, "first_position": int(expected[0]),
            "last_position": int(expected[-1]), "total_keys": total})

    def traced_decode(cache, suffix, prefix_len):
        if inspect:
            box["cache"] = _compare_installed(
                ctx.meta, cache, capture_layers, ranking, k,
                check_full=(k == n))
        if sentinel:
            assert row_end > k, "sentinel fixture has no unused loaded real row"
            for layer in cache.layers:
                for kind in ("keys", "values"):
                    getattr(layer, kind)[0, :, vstart + k:vstart + row_end] = (
                        SENTINEL)
        handle = server.runner.model.register_forward_hook(output_hook)
        position_handle = server.runner.model.register_forward_pre_hook(
            position_hook, with_kwargs=True)
        try:
            with _AttentionTrace(ctx.meta, k) as masks:
                decoded = original_decode(cache, suffix, prefix_len)
            box["attention_masks"] = masks.summary()
            return decoded
        finally:
            handle.remove()
            position_handle.remove()

    server._decode = traced_decode
    vision_calls = []

    def vision_guard(module, args, kwargs):
        vision_calls.append(1)
        raise AssertionError("cache hit executed the vision tower")

    vision_handle = server.runner.model.model.vision_tower.register_forward_pre_hook(
        vision_guard, with_kwargs=True)
    try:
        with _ReadTrace() as reads:
            result = server.request_cvpr25(
                ctx, question=question, static=None, mode="prefix",
                budget=budget, budget_unit=budget_unit,
                sep_policy="sidecar", cold=False,
                expected_prefix_layout="visionzip_image_only")
    finally:
        vision_handle.remove()
        server._decode = original_decode
        BIAS.clear()
    box["result"] = result
    box["reads"] = _check_reads(
        ctx, reads, math.ceil(k / CHUNK_SIZE)
        if budget_unit == "visual_kv" else len(
            result["selected_chunk_ids_per_layer"][0]))
    assert len(box["logits"]) == len(result["generated_token_ids"])
    assert len(box["position_checks"]) == len(box["logits"])
    assert result["first_token_id"] == result["generated_token_ids"][0]
    box["vision_forward_count"] = len(vision_calls)
    return box


def _memory_reference(server, ctx, question, capture_layers, ranking, k):
    cache = _make_memory_cache(server.runner, ctx.meta, capture_layers, ranking,
                               k)
    _install_reference_bias(ctx.meta, k, len(capture_layers),
                            server.runner.model.device)
    logits = []

    def hook(module, args, output):
        if hasattr(output, "logits"):
            logits.append(output.logits[0, -1].detach().float().cpu())

    handle = server.runner.model.register_forward_hook(hook)
    try:
        answer, first, timing = server._decode(
            cache, suffix_ids_for(server.runner, question).to(
                server.runner.model.device), int(ctx.meta["prefix_len"]))
    finally:
        handle.remove()
        BIAS.clear()
        del cache
    return {"answer": answer, "first_token_id": first,
            "generated_token_ids": timing["generated_token_ids"],
            "logits": logits}


def _compare_outputs(left: dict, right: dict) -> dict:
    a, b = left["logits"], right["logits"]
    assert len(a) == len(b) and len(a) > 0
    diffs = [(x - y).abs() for x, y in zip(a, b)]
    allclose = all(torch.allclose(x, y, atol=ATOL, rtol=RTOL)
                   for x, y in zip(a, b))
    first_exact = bool(torch.equal(a[0], b[0]))
    same_tokens = left["generated_token_ids"] == right["generated_token_ids"]
    same_first = left["first_token_id"] == right["first_token_id"]
    same_answer = left["answer"] == right["answer"]
    verdict = {"all_step_logits_within_frozen_tolerance": allclose,
               "first_step_logits_exact": first_exact,
               "max_abs_logit_delta": max(float(d.max()) for d in diffs),
               "first_token_exact": same_first,
               "generated_token_ids_exact": same_tokens,
               "answer_exact": same_answer,
               "n_steps": len(a), "atol": ATOL, "rtol": RTOL}
    assert allclose and same_first and same_tokens and same_answer, verdict
    return verdict


def _sample(runner, server, entry, question_id, store_root: Path) -> dict:
    image_id = str(entry["image_id"])
    assert str(entry["questions"][4]["question_id"]) == question_id
    question = entry["questions"][4]["question"]
    with Image.open(ROOT / entry["image_path"]) as image_file:
        image = image_file.convert("RGB")
    enc_cpu = runner.encode(image, entry["questions"][4]["question"])
    vstart, v = runner.visual_span(enc_cpu["input_ids"])
    _, _, _, separators = runner.anyres_layout(enc_cpu["image_sizes"][0], v)
    n, k = v - len(separators), (v - len(separators) + 3) // 4
    capture = VisionForwardCapture(runner, capture_saliency=True)
    with capture:
        recompute = server.recompute(runner.to_device(enc_cpu),
                                     return_past_key_values=True)
    canonical = recompute.pop("captured_past_key_values")
    source_layers = cache_layers(canonical)
    ranking = _score_rank(runner, capture.result_cpu(),
                          enc_cpu["image_sizes"][0], v, separators)
    assert len(ranking) == n
    store = store_root / image_id
    persisted = persist_captured_visual_prefix(
        runner, canonical, enc_cpu["input_ids"],
        enc_cpu["image_sizes"][0], capture.result_cpu(), store,
        image_id=image_id, model_id=runner.model_id,
        chunk_size=CHUNK_SIZE, capture_stats=capture,
        extra_metadata={"dataset": "gqa", "source_turn_id": 1,
                        "validation_only": True})
    ctx = ImageContext(store, runner.model.device, require_v_hidden=False)
    try:
        ctx.validate_visual_kv_layout()
        meta = ctx.meta
        assert int(meta["v_token_start"]) == vstart
        assert int(meta["v_token_num"]) == v
        assert int(meta["n_spatial"]) == n
        assert [int(x) for x in meta["order"][:n]] == ranking, (
            "independent captured-score rank differs from saved permutation")
        assert [int(x) for x in meta["order"][n:]] == separators
        assert meta["dtype"] == "float16" and meta["chunk_size"] == 64
        m = (k + 63) // 64
        expected_chunk_ids = list(range(m))
        reference = _memory_reference(server, ctx, question, source_layers,
                                      ranking, k)
        production = _run_with_trace(server, ctx, question, source_layers,
                                     ranking, k, budget=.25,
                                     budget_unit="visual_kv")
        result = production["result"]
        assert result["budget_unit"] == "visual_kv"
        assert result["N_content"] == n and result["N_structural"] == v - n
        assert result["k_target"] == result["attended_content_kv_count"] == k
        assert result["normal_chunks_read"] == m
        assert result["selected_stored_ids"] == list(range(k))
        assert result["selected_original_ids"] == ranking[:k]
        assert result["selected_chunk_ids_per_layer"] == [
            expected_chunk_ids] * len(source_layers)
        assert result["actual_loaded_real_rows"] == min(m * 64, n)
        assert result["unused_loaded_real_rows"] == min(m * 64, n) - k
        assert result["keep_count_per_layer"] == [k + (v - n)] * len(
            source_layers)
        assert result["io"]["bytes"] == production["reads"][
            "actual_returned_bytes"]
        assert result["io"]["preads"] == production["reads"][
            "actual_pread_calls"]
        assert result["query_score_calls"] == 0
        assert result["static_score_calls"] == 0
        assert result["diversity_calls"] == 0
        comparison = _compare_outputs(production["result"] | {
            "logits": production["logits"]}, reference)

        sentinel_result = None
        if min(m * 64, n) > k:
            sentinel = _run_with_trace(
                server, ctx, question, source_layers, ranking, k,
                budget=.25, budget_unit="visual_kv", sentinel=True)
            sentinel_result = _compare_outputs(
                production["result"] | {"logits": production["logits"]},
                sentinel["result"] | {"logits": sentinel["logits"]})
        full = _run_with_trace(server, ctx, question, source_layers, ranking,
                               n, budget=1.0, budget_unit="visual_kv")
        assert full["result"]["attended_content_kv_count"] == n
        assert full["result"]["selected_original_ids"] == ranking

        legacy = None
        legacy_chunks = round(int(meta["n_chunks_per_layer"]) * .25)
        if legacy_chunks * 64 == k:
            old = _run_with_trace(server, ctx, question, source_layers,
                                  ranking, k, budget=.25,
                                  budget_unit="chunk")
            legacy = _compare_outputs(
                production["result"] | {"logits": production["logits"]},
                old["result"] | {"logits": old["logits"]})
            assert old["result"]["selected_chunk_ids_per_layer"] == [
                expected_chunk_ids] * len(source_layers)
        alternate_question = entry["questions"][5]["question"]
        alternate = _run_with_trace(server, ctx, alternate_question,
                                    source_layers, ranking, k,
                                    budget=.25, budget_unit="visual_kv")
        assert alternate["result"]["selected_original_ids"] == ranking[:k]
        assert alternate["result"]["selected_stored_ids"] == list(range(k))
        assert alternate["result"]["query_score_calls"] == 0
        repeat = None
        if image_id == FIXED[0][0]:
            # A different question and Full100 request precede this repeat.
            revisited = _run_with_trace(server, ctx, question, source_layers,
                                        ranking, k, budget=.25,
                                        budget_unit="visual_kv")
            repeat = _compare_outputs(
                production["result"] | {"logits": production["logits"]},
                revisited["result"] | {"logits": revisited["logits"]})
        return {
            "image_id": image_id, "question_id": question_id,
            "N_content": n, "N_structural": v - n, "k": k,
            "m": m, "selected_original_ids_sha256": hashlib.sha256(
                json.dumps(ranking[:k], separators=(",", ":")).encode()).hexdigest(),
            "reference_vs_ssd": comparison,
            "sentinel_prefill_decode": sentinel_result,
            "full100_canonical_logical": full["cache"],
            "legacy_equal_set": legacy,
            "different_question_same_selected_ids": True,
            "interleaved_repeat_no_cache_leak": repeat,
            "cache_exact": production["cache"],
            "combined_attention_masks": production["attention_masks"],
            "original_position_checks": production["position_checks"],
            "physical_reads": production["reads"],
            "persistence_integrity": persisted["integrity"],
            "turn1_vision_forward_count": capture.call_count,
            "hit_vision_forward_count": production["vision_forward_count"],
            "hit_online_score_calls": result["query_score_calls"],
            "first_token_id": result["first_token_id"],
            "generated_token_ids": result["generated_token_ids"],
            "status": "PASS",
        }
    finally:
        ctx.close()
        del canonical, source_layers
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=ROOT / "data/index.json")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--store-name", default="gpu_validation_stores")
    parser.add_argument("--receipt-name", default="gpu_validation.json")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    assert args.receipt_name.startswith("gpu_validation") and args.receipt_name.endswith(".json")
    assert Path(args.receipt_name).name == args.receipt_name
    receipt = run_dir / args.receipt_name
    assert args.store_name.startswith("gpu_validation_stores")
    assert Path(args.store_name).name == args.store_name
    store_root = run_dir / args.store_name
    if receipt.exists() or store_root.exists():
        raise FileExistsError("GPU validation requires new receipt and store root")
    assert sha256_file(args.index) == INDEX_SHA256, "frozen GQA index changed"
    entries = load_index(args.index)
    assert len(entries) == 40
    assert [(str(e["image_id"]), str(e["questions"][4]["question_id"]))
            for e in entries[:5]] == list(FIXED)
    assert torch.cuda.is_available(), "LLaVA GPU correctness needs CUDA"
    store_root.mkdir()
    result = {
        "schema_version": "llava-kv25-gpu-correctness-v1",
        "GPU_CORRECTNESS": "NOT RUN",
        "fixed_samples": [list(x) for x in FIXED],
        "reference": "same-Turn1 canonical captured KV and independently ranked CLIP scores",
        "cache": "same full-size physical order, BF16 compute after FP16 SSD roundtrip",
        "first_step_fp32_logits_tolerance": {"atol": ATOL, "rtol": RTOL},
        "sentinel": SENTINEL, "max_new_tokens": args.max_new_tokens,
        "index_sha256": INDEX_SHA256,
        "source_sha256": {
            relative: sha256_file(ROOT / relative)
            for relative in (
                "mmimpress/cvpr25.py", "mmimpress/serve.py",
                "mmimpress/store.py", "mmimpress/model.py",
                "mmimpress/piggyback.py")},
        "contract_sha256": sha256_file(
            ROOT / "docs/llava_kv25_budget_contract.md"),
        "model_id": "llava-hf/llava-v1.6-vicuna-7b-hf",
        "chunk_size": CHUNK_SIZE,
        "samples": [],
    }
    _atomic_json(receipt, result)
    try:
        runner = LlavaRunner().load()
        server = Server(runner, max_new_tokens=args.max_new_tokens)
        for (image_id, question_id), entry in zip(FIXED, entries):
            assert str(entry["image_id"]) == image_id
            row = _sample(runner, server, entry, question_id, store_root)
            result["samples"].append(row)
            _atomic_json(receipt, result)
            print(f"GPU correctness {image_id}: PASS", flush=True)
        assert len(result["samples"]) == len(FIXED)
        assert any(row["legacy_equal_set"] is not None
                   for row in result["samples"]), (
            "fixed samples have no equal-set legacy fixture")
        assert any(row["sentinel_prefill_decode"] is not None
                   for row in result["samples"]), (
            "fixed samples have no unused real row sentinel fixture")
        result["GPU_CORRECTNESS"] = "PASS"
    except BaseException as exc:
        result["GPU_CORRECTNESS"] = "FAIL"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        _atomic_json(receipt, result)


if __name__ == "__main__":
    main()

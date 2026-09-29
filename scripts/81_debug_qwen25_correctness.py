#!/usr/bin/env python3
"""Isolated Qwen2.5-VL numerical-correctness trace on frozen GQA n355567.

R0 is the normal full multimodal prefill. R1 clones the prefix from that SAME
R0 output and forwards only the suffix, without SSD. Optional P0/P1/P2 read
the frozen repacked SSD prefix once and vary GPU cache layout only. This script
does not modify the adapter, model, old validation, or old stores. Output is
compact JSON statistics; suffix tensors are held only in RAM during tracing.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mmimpress.qwen25.runner import (  # noqa: E402
    CHECKPOINT_REVISION, MAX_NEW_TOKENS, Qwen25Runner, _cache_layers,
)
from mmimpress.qwen25.store import inverse_permutation  # noqa: E402
from mmimpress.cvpr25 import budget_chunk_count  # noqa: E402

VALIDATION = ROOT / "runs/qwen25_port_validate_gpu_20260928_0625_sdpa_frozen/validation.json"
REPACKED = ROOT / "runs/qwen25_port_validate_gpu_20260928_0625_sdpa_frozen/repacked_store"
IMAGE_ID = "n355567"
IMAGE_SHA256 = "6b09d6ab2c13d108951fd6d53522fcb80e6cfd5bb3e1f2925e1001919343a39f"
PRIMARY_QID = "201751740"
LOGIT_ATOL, LOGIT_RTOL = 0.125, 0.02
STAGES = (
    "layer_input", "input_rmsnorm", "attention_input", "q_pre_mrope", "k_pre_mrope", "v",
    "q_post_mrope", "k_post_mrope", "attention_output",
    "post_attention_residual", "post_attention_rmsnorm", "mlp_output",
    "layer_output",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write_new(path: Path, data: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        handle.write("\n")


def tensor_brief(x: torch.Tensor | None) -> dict:
    if x is None:
        return {"present": False}
    y = x.detach().cpu()
    result = {"present": True, "shape": list(y.shape), "dtype": str(y.dtype)}
    if y.numel() and (y.is_floating_point() or y.is_complex()):
        z = y.float()
        result["finite"] = bool(torch.isfinite(z).all())
        result["max_abs"] = (float(z[torch.isfinite(z)].abs().max())
                             if bool(torch.isfinite(z).any()) else None)
    return result


def tensor_compare(a: torch.Tensor | None, b: torch.Tensor | None,
                   *, atol: float | None = None, rtol: float | None = None) -> dict:
    out = {"left": tensor_brief(a), "right": tensor_brief(b)}
    if a is None or b is None or a.shape != b.shape:
        out.update({"comparable": False, "exact_equal": False})
        return out
    x, y = a.detach().cpu(), b.detach().cpu()
    neq = x != y
    out["comparable"] = True
    out["exact_equal"] = bool(not neq.any())
    out["first_mismatch_index"] = (
        [int(z) for z in torch.unravel_index(
            neq.reshape(-1).nonzero()[0, 0], x.shape)]
        if neq.any() else None
    )
    xf, yf = x.float(), y.float()
    diff = (xf - yf).abs()
    finite = torch.isfinite(diff)
    out["max_abs_diff"] = (float(diff[finite].max())
                           if bool(finite.any()) else None)
    out["mean_abs_diff"] = (float(diff[finite].mean())
                            if bool(finite.any()) else None)
    out["finite_both"] = bool(torch.isfinite(xf).all() and torch.isfinite(yf).all())
    if atol is not None and rtol is not None:
        out["allclose"] = bool(torch.allclose(xf, yf, atol=atol, rtol=rtol))
    return out


def compact_diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    c = tensor_compare(a, b, atol=LOGIT_ATOL, rtol=LOGIT_RTOL)
    c["first_token_same"] = bool(int(a.argmax()) == int(b.argmax()))
    c["left_first_token"] = int(a.argmax())
    c["right_first_token"] = int(b.argmax())
    return c


def _field_tensor(kwargs: dict, args: tuple, key: str, arg_idx: int = 0):
    return kwargs.get(key, args[arg_idx] if len(args) > arg_idx else None)


class LayerTrace:
    """Retain only suffix slices of intermediate decoder tensors on CPU."""

    def __init__(self, model, *, suffix_from: int):
        self.model = model
        self.suffix_from = int(suffix_from)
        self.records = [dict() for _ in model.model.language_model.layers]
        self.masks: list[torch.Tensor | None] = [None] * len(self.records)
        self.mask_meta = [dict() for _ in self.records]
        self.embeddings: torch.Tensor | None = None
        self.forward_meta: dict = {}
        self._pending: dict[int, dict] = {}
        self._handles = []

    def _suffix(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:, self.suffix_from:].detach().cpu().clone()

    def _save(self, li: int, stage: str, tensor: torch.Tensor) -> None:
        self.records[li][stage] = self._suffix(tensor)

    def _language_pre(self, _module, args, kwargs):
        embeds = _field_tensor(kwargs, args, "inputs_embeds")
        self.embeddings = self._suffix(embeds)
        self.forward_meta = {
            "input_ids": tensor_brief(kwargs.get("input_ids")),
            "position_ids": tensor_brief(kwargs.get("position_ids")),
            "cache_position": kwargs.get("cache_position").detach().cpu().tolist()
            if torch.is_tensor(kwargs.get("cache_position")) else None,
            "rope_deltas_at_language_entry": tensor_brief(self.model.model.rope_deltas),
        }

    def _layer_pre(self, li, _module, args, kwargs):
        h = _field_tensor(kwargs, args, "hidden_states")
        self._save(li, "layer_input", h)
        cache = kwargs.get("past_key_values")
        past_len = 0
        if cache is not None and hasattr(cache, "layers") and len(cache.layers) > li:
            keys = cache.layers[li].keys
            if keys is not None:
                past_len = int(keys.shape[2])
        cp = kwargs.get("cache_position")
        self.mask_meta[li] = {
            "attention_type": self.model.model.language_model.layers[li].attention_type,
            "query_length": int(h.shape[1]),
            "past_key_length_at_layer_entry": past_len,
            "expected_key_length": past_len + int(h.shape[1]),
            "cache_position": cp.detach().cpu().tolist() if torch.is_tensor(cp) else None,
        }

    def _attn_pre(self, li, _module, args, kwargs):
        emb = kwargs.get("position_embeddings")
        if emb is None:
            raise RuntimeError("Qwen attention did not receive shared MRoPE embeddings")
        self._pending[li] = {"cos": emb[0], "sin": emb[1]}
        h = _field_tensor(kwargs, args, "hidden_states")
        self._save(li, "attention_input", h)
        mask = kwargs.get("attention_mask")
        self.masks[li] = mask.detach().cpu().clone() if torch.is_tensor(mask) else None
        self.mask_meta[li]["mask_argument"] = tensor_brief(mask)
        self.mask_meta[li]["position_ids_argument"] = tensor_brief(kwargs.get("position_ids"))

    def _projection_post(self, li, name, _module, _args, output):
        if name in ("q", "k"):
            self._pending[li][name] = output[:, self.suffix_from:]
        if name == "q":
            self._save(li, "q_pre_mrope", output)
        elif name == "k":
            self._save(li, "k_pre_mrope", output)
        else:
            self._save(li, "v", output)
            pending = self._pending[li]
            q, k = pending["q"], pending["k"]
            attn = self.model.model.language_model.layers[li].self_attn
            q = q.view(q.shape[0], q.shape[1], attn.num_heads, attn.head_dim).transpose(1, 2)
            k = k.view(k.shape[0], k.shape[1], attn.num_key_value_heads,
                       attn.head_dim).transpose(1, 2)
            cos = pending["cos"][:, :, self.suffix_from:, :]
            sin = pending["sin"][:, :, self.suffix_from:, :]
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                apply_multimodal_rotary_pos_emb,
            )
            qr, kr = apply_multimodal_rotary_pos_emb(
                q, k, cos, sin, attn.rope_scaling["mrope_section"])
            self.records[li]["q_post_mrope"] = qr.detach().cpu().clone()
            self.records[li]["k_post_mrope"] = kr.detach().cpu().clone()
            del self._pending[li]

    def _module_pre(self, li, name, _module, args):
        self._save(li, name, args[0])

    def _module_post(self, li, name, _module, _args, output):
        self._save(li, name, output)

    def _attn_post(self, li, _module, _args, output):
        self._save(li, "attention_output", output[0])

    def _layer_post(self, li, _module, _args, output):
        self._save(li, "layer_output", output[0])

    def __enter__(self):
        lm = self.model.model.language_model
        self._handles.append(lm.register_forward_pre_hook(
            self._language_pre, with_kwargs=True))
        for li, layer in enumerate(lm.layers):
            self._handles.append(layer.register_forward_pre_hook(
                lambda m, a, k, i=li: self._layer_pre(i, m, a, k),
                with_kwargs=True))
            self._handles.append(layer.self_attn.register_forward_pre_hook(
                lambda m, a, k, i=li: self._attn_pre(i, m, a, k),
                with_kwargs=True))
            for name, module in (("q", layer.self_attn.q_proj),
                                 ("k", layer.self_attn.k_proj),
                                 ("v", layer.self_attn.v_proj)):
                self._handles.append(module.register_forward_hook(
                    lambda m, a, o, i=li, n=name:
                    self._projection_post(i, n, m, a, o)))
            self._handles.append(layer.input_layernorm.register_forward_hook(
                lambda m, a, o, i=li:
                self._module_post(i, "input_rmsnorm", m, a, o)))
            self._handles.append(layer.self_attn.register_forward_hook(
                lambda m, a, o, i=li: self._attn_post(i, m, a, o)))
            self._handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
                lambda m, a, i=li:
                self._module_pre(i, "post_attention_residual", m, a)))
            self._handles.append(layer.post_attention_layernorm.register_forward_hook(
                lambda m, a, o, i=li:
                self._module_post(i, "post_attention_rmsnorm", m, a, o)))
            self._handles.append(layer.mlp.register_forward_hook(
                lambda m, a, o, i=li:
                self._module_post(i, "mlp_output", m, a, o)))
            self._handles.append(layer.register_forward_hook(
                lambda m, a, o, i=li: self._layer_post(i, m, a, o)))
        return self

    def __exit__(self, *_exc):
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._pending.clear()


class SDPAProbe:
    """Record actual decoder SDPA call shape and optional FP32 math control."""

    def __init__(self, num_heads: int, head_dim: int, fp32: bool = False):
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.fp32 = bool(fp32)
        self.records: list[dict] = []
        self._original = None
        self._old_tf32 = None

    def __enter__(self):
        self._original = F.scaled_dot_product_attention
        self._old_tf32 = torch.backends.cuda.matmul.allow_tf32
        if self.fp32:
            torch.backends.cuda.matmul.allow_tf32 = False

        def wrapped(q, k, v, *args, **kwargs):
            decoder = (q.ndim == 4 and q.shape[1] == self.num_heads
                       and q.shape[-1] == self.head_dim)
            if not decoder:
                return self._original(q, k, v, *args, **kwargs)
            mask = kwargs.get("attn_mask", args[0] if args else None)
            is_causal = kwargs.get("is_causal")
            if is_causal is None:
                is_causal = bool(q.shape[2] > 1 and mask is None)
            self.records.append({
                "q_shape": list(q.shape), "k_shape": list(k.shape),
                "v_shape": list(v.shape), "q_dtype": str(q.dtype),
                "k_dtype": str(k.dtype), "mask": tensor_brief(mask),
                "is_causal": bool(is_causal),
                "enable_gqa": bool(kwargs.get("enable_gqa", False)),
                "fp32_decoder_attention_control": self.fp32,
            })
            if not self.fp32:
                return self._original(q, k, v, *args, **kwargs)
            copied_args = list(args)
            copied_kwargs = dict(kwargs)
            if torch.is_tensor(mask) and mask.is_floating_point():
                mask32 = mask.float()
                if copied_args:
                    copied_args[0] = mask32
                else:
                    copied_kwargs["attn_mask"] = mask32
            with torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_math=True, enable_mem_efficient=False):
                result = self._original(q.float(), k.float(), v.float(),
                                        *copied_args, **copied_kwargs)
            return result.to(q.dtype)

        F.scaled_dot_product_attention = wrapped
        return self

    def __exit__(self, *_exc):
        F.scaled_dot_product_attention = self._original
        torch.backends.cuda.matmul.allow_tf32 = self._old_tf32


def clone_prefix(cache, prefix_len: int, config):
    from transformers import DynamicCache
    result = DynamicCache(config=config)
    for li, (k, v) in enumerate(_cache_layers(cache)):
        result.update(k[:, :, :prefix_len, :].detach().clone(),
                      v[:, :, :prefix_len, :].detach().clone(), li)
    return result


def compare_cache_prefix(a, b, prefix_len: int) -> dict:
    rows = []
    for li, ((ak, av), (bk, bv)) in enumerate(
            zip(_cache_layers(a), _cache_layers(b))):
        for name, x, y in (("k", ak, bk), ("v", av, bv)):
            c = tensor_compare(x[:, :, :prefix_len, :],
                               y[:, :, :prefix_len, :])
            rows.append({"layer": li, "kind": name,
                         "exact_equal": c["exact_equal"],
                         "max_abs_diff": c.get("max_abs_diff"),
                         "first_mismatch_index": c["first_mismatch_index"]})
    return {"all_exact": all(r["exact_equal"] for r in rows), "per_layer": rows}


def explicit_causal_4d(seq_len: int, dtype: torch.dtype, device) -> torch.Tensor:
    """Diagnostic only: same logical causal pattern, explicit additive layout."""
    lower = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).tril()
    mask = torch.full((1, 1, seq_len, seq_len),
                      torch.finfo(dtype).min, dtype=dtype, device=device)
    return mask.masked_fill(lower[None, None], 0)


def _decode_result(runner, output, mask_2d: torch.Tensor,
                   position_start: int) -> dict:
    first = int(output.logits[0, -1].argmax())
    tokens, _ = runner._decode(
        first, output.past_key_values, position_start=position_start,
        attention_mask=mask_2d)
    return {"first_token_id": first, "generated_token_ids": tokens,
            "prediction": runner.processor.tokenizer.decode(
                tokens, skip_special_tokens=True,
                clean_up_tokenization_spaces=False).strip()}


@torch.inference_mode()
def execute_path(runner, *, input_ids: torch.Tensor, position_ids: torch.Tensor,
                 cache_position: torch.Tensor, attention_mask: torch.Tensor,
                 decode_mask: torch.Tensor, suffix_from: int,
                 cache=None, pixel_kwargs: dict | None = None,
                 prefix_clone_len: int | None = None,
                 fp32_attention: bool = False) -> dict:
    runner._clear_request_state()
    state_before = tensor_brief(runner.model.model.rope_deltas)
    trace = LayerTrace(runner.model, suffix_from=suffix_from)
    tc = runner.model.config.text_config
    probe = SDPAProbe(tc.num_attention_heads,
                      tc.hidden_size // tc.num_attention_heads,
                      fp32=fp32_attention)
    with probe:
        with trace:
            output = runner.model(
                input_ids=input_ids, position_ids=position_ids,
                cache_position=cache_position,
                attention_mask=attention_mask, past_key_values=cache,
                use_cache=True, return_dict=True, logits_to_keep=1,
                **(pixel_kwargs or {}))
            torch.cuda.synchronize(runner.device)
        logits = output.logits[0, -1].float().cpu().clone()
        prefix = (clone_prefix(output.past_key_values, prefix_clone_len,
                               runner.model.config.text_config)
                  if prefix_clone_len is not None else None)
        predecode_cache_len = int(output.past_key_values.get_seq_length())
        state_after_prefill = tensor_brief(runner.model.model.rope_deltas)
        decoded = _decode_result(
            runner, output, decode_mask,
            position_start=int(position_ids.max()) + 1)
        torch.cuda.synchronize(runner.device)
    state_after_decode = tensor_brief(runner.model.model.rope_deltas)
    runner._clear_request_state()
    return {"trace": trace, "logits": logits, "prefix_cache": prefix,
            "output_cache": output.past_key_values,
            "sdpa_calls_prefill": probe.records[:len(trace.records)],
            "sdpa_call_count_total": len(probe.records),
            "state_before": state_before,
            "state_after_prefill": state_after_prefill,
            "state_after_decode": state_after_decode,
            "predecode_cache_len": predecode_cache_len,
            **decoded}


def layer_diffs(a: LayerTrace, b: LayerTrace) -> dict:
    if len(a.records) != len(b.records):
        raise AssertionError("decoder layer count mismatch")
    rows, first = [], None
    for li, (left, right) in enumerate(zip(a.records, b.records)):
        for stage in STAGES:
            c = tensor_compare(left.get(stage), right.get(stage))
            entry = {"layer": li, "stage": stage, **c}
            rows.append(entry)
            if first is None and not c["exact_equal"]:
                first = {"layer": li, "stage": stage,
                         "max_abs_diff": c.get("max_abs_diff"),
                         "first_mismatch_index": c.get("first_mismatch_index")}
    return {"first_divergence": first, "per_stage": rows,
            "embedding": tensor_compare(a.embeddings, b.embeddings)}


def _mask_visible(mask: torch.Tensor | None, q_len: int, k_len: int,
                  suffix_from: int, sdpa_record: dict | None) -> torch.Tensor:
    if mask is not None:
        if mask.ndim != 4:
            raise AssertionError(f"expected actual 4D attention mask, got {mask.shape}")
        rows = mask[0, 0, suffix_from:suffix_from + q_len, :k_len]
        if rows.dtype == torch.bool:
            return rows.clone()
        return rows == 0
    causal = (bool(sdpa_record["is_causal"]) if sdpa_record is not None
              else False)  # eager with no mask has no causal filtering
    if causal:
        return torch.arange(k_len)[None, :] <= torch.arange(
            suffix_from, suffix_from + q_len)[:, None]
    return torch.ones((q_len, k_len), dtype=torch.bool)


def mask_report(path: dict, logical_prefix_indices: list[int],
                full_prefix_len: int, full_total_len: int) -> tuple[dict, torch.Tensor]:
    trace = path["trace"]
    q_len = trace.records[0]["layer_input"].shape[1]
    key_mapping = (list(logical_prefix_indices)
                   + list(range(full_prefix_len, full_prefix_len + q_len)))
    k_len = len(key_mapping)
    rec = path["sdpa_calls_prefill"][0] if path["sdpa_calls_prefill"] else None
    visible = _mask_visible(trace.masks[0], q_len, k_len,
                            trace.suffix_from, rec)
    if visible.shape != (q_len, k_len):
        raise AssertionError((visible.shape, q_len, k_len))
    logical = torch.zeros((q_len, full_total_len), dtype=torch.bool)
    logical[:, key_mapping] = visible
    counts = []
    for qi, row in enumerate(logical):
        prefix_count = int(row[:full_prefix_len].sum())
        suffix_count = int(row[full_prefix_len:].sum())
        visible_positions = row.nonzero(as_tuple=True)[0].tolist()
        counts.append({
            "query_original_index": full_prefix_len + qi,
            "visible_total": int(row.sum()),
            "visible_prefix": prefix_count,
            "visible_suffix": suffix_count,
            "future_visible": int(row[full_prefix_len + qi + 1:].sum()),
            "visible_logical_sha256": hashlib.sha256(
                json.dumps(visible_positions, separators=(",", ":")).encode()).hexdigest(),
        })
    return ({
        "actual_layer0_mask": trace.mask_meta[0],
        "sdpa_layer0_call": rec,
        "key_mapping_length": k_len,
        "first_query": counts[0], "last_query": counts[-1],
        "all_future_hidden": all(row["future_visible"] == 0 for row in counts),
        "per_query_counts": counts,
    }, logical)


def mask_compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    if a.shape != b.shape:
        return {"same_pattern": False, "left_shape": list(a.shape),
                "right_shape": list(b.shape)}
    diff = a != b
    first = diff.nonzero()
    return {"same_pattern": bool(not diff.any()),
            "differing_cells": int(diff.sum()),
            "first_mismatch_query_and_logical_key":
            first[0].tolist() if first.numel() else None}


def position_audit(runner, full_ids_cpu: torch.Tensor,
                   grid_cpu: torch.Tensor, positions_cpu: torch.Tensor,
                   deltas_cpu: torch.Tensor, prefix_len: int,
                   prefix_cache) -> dict:
    """Record manual full MRoPE and the stock generation preparation output."""
    full_ids = full_ids_cpu.to(runner.device)
    grid = grid_cpu.to(runner.device)
    n = full_ids.shape[1]
    manual = positions_cpu.to(runner.device)
    report = {
        "manual_rope_deltas": deltas_cpu.tolist(),
        "manual_suffix_first_thw": manual[:, 0, prefix_len].tolist(),
        "manual_suffix_last_thw": manual[:, 0, -1].tolist(),
        "manual_first_generated_thw": [int(manual.max()) + 1] * 3,
        "manual_full_positions_shape": list(manual.shape),
        "prefix_position_last_thw": manual[:, 0, prefix_len - 1].tolist(),
    }
    # prepare_inputs_for_generation is a read-only preparation here; no forward.
    old = runner.model.model.rope_deltas
    try:
        runner.model.model.rope_deltas = None
        stock_full = runner.model.prepare_inputs_for_generation(
            full_ids, attention_mask=torch.ones_like(full_ids),
            cache_position=torch.arange(n, device=runner.device),
            pixel_values=None, image_grid_thw=grid)
        stock_full_pos = stock_full.get("position_ids")
        report["stock_full"] = {
            "position_shape": list(stock_full_pos.shape),
            "manual_thw_exact": bool(torch.equal(stock_full_pos[-3:], manual)),
            "text_axis_first_last":
            ([int(stock_full_pos[0, 0, 0]), int(stock_full_pos[0, 0, -1])]
             if stock_full_pos.shape[0] == 4 else None),
            "prepared_input_ids_equal": bool(torch.equal(
                stock_full.get("input_ids"), full_ids)),
        }
        runner.model.model.rope_deltas = deltas_cpu.to(runner.device)
        stock_suffix = runner.model.prepare_inputs_for_generation(
            full_ids, past_key_values=prefix_cache,
            attention_mask=torch.ones_like(full_ids),
            cache_position=torch.arange(prefix_len, n, device=runner.device),
            image_grid_thw=grid)
        suffix_pos = stock_suffix.get("position_ids")
        report["stock_suffix"] = {
            "position_shape": list(suffix_pos.shape),
            "manual_thw_exact": bool(torch.equal(
                suffix_pos[-3:], manual[:, :, prefix_len:])),
            "prepared_suffix_ids_equal": bool(torch.equal(
                stock_suffix.get("input_ids"), full_ids[:, prefix_len:])),
            "first_thw": suffix_pos[-3:, 0, 0].tolist(),
            "last_thw": suffix_pos[-3:, 0, -1].tolist(),
            "text_axis_first_last":
            ([int(suffix_pos[0, 0, 0]), int(suffix_pos[0, 0, -1])]
             if suffix_pos.shape[0] == 4 else None),
        }
    except Exception as exc:
        report["stock_preparation_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        runner.model.model.rope_deltas = old
    return report


def prepare_fixture(runner, validation: dict, question: dict,
                    repacked_meta: dict | None) -> tuple[dict, dict]:
    source = validation["source"]
    image_path = Path(source["image_path"])
    if source["image_id"] != IMAGE_ID or source["image_sha256"] != IMAGE_SHA256:
        raise AssertionError("frozen validation image identity changed")
    if sha256_file(image_path) != IMAGE_SHA256:
        raise AssertionError("image content differs from frozen validation")
    with Image.open(image_path) as im:
        image = im.convert("RGB")
    text = runner._chat_text(question["question"], ())
    enc = runner.processor(text=[text], images=[image], return_tensors="pt")
    ids = enc["input_ids"]
    geom = runner._image_geometry(ids, enc["image_grid_thw"], enc["pixel_values"])
    positions, deltas = runner._logical_positions(ids, enc["image_grid_thw"])
    prefix_len = geom["prefix_len"]
    selected = []
    tuples = []
    if repacked_meta is not None:
        meta = repacked_meta
        if ids[0, :prefix_len].tolist() != meta["prefix_input_ids"]:
            raise AssertionError("expanded prompt prefix differs from frozen store")
        if geom["visual_count"] != meta["visual_count"]:
            raise AssertionError("visual geometry differs from frozen store")
        if positions[:, 0, :prefix_len].tolist() != meta["logical_position_ids"]:
            raise AssertionError("MRoPE prefix differs from frozen store")
        if meta["chunk_size"] != 64 or meta["image_grid_thw"] != enc["image_grid_thw"].tolist():
            raise AssertionError("frozen store layout changed")
        if inverse_permutation(meta["stored_to_original"]) != meta["original_to_stored"]:
            raise AssertionError("frozen permutation inverse changed")
        chunks = budget_chunk_count(meta["n_chunks"], .25)
        selected_count = min(meta["visual_count"], chunks * meta["chunk_size"])
        selected = [int(x) for x in meta["stored_to_original"][:selected_count]]
        rank = {original: i for i, original in enumerate(sorted(selected))}
        for stored_index, original in enumerate(selected):
            logical = geom["visual_start"] + original
            tuples.append({
                "original_visual_index": original,
                "stored_index": stored_index,
                "compact_p1_index": geom["visual_start"] + stored_index,
                "compact_p2_index": geom["visual_start"] + rank[original],
                "logical_prompt_position": logical,
                "mrope_thw": positions[:, 0, logical].tolist(),
            })
    fixture = {
        "image_id": IMAGE_ID, "image_path": str(image_path),
        "image_sha256": IMAGE_SHA256,
        "question_id": str(question["question_id"]),
        "question": question["question"],
        "prompt_text": text,
        "complete_expanded_input_ids": ids[0].tolist(),
        "input_ids_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        "image_grid_thw": enc["image_grid_thw"].tolist(),
        "visual_token_count": geom["visual_count"],
        "visual_start": geom["visual_start"],
        "prefix_boundary": prefix_len,
        "suffix_input_ids": ids[0, prefix_len:].tolist(),
        "full_logical_position_ids": positions[:, 0].tolist(),
        "rope_deltas": deltas.tolist(),
        "selected_prefix25_original_visual_ids_physical_order": selected,
        "selected_prefix25_original_visual_ids_logical_order": sorted(selected),
        "selected_positional_tuples": tuples,
        "stored_to_original": repacked_meta["stored_to_original"] if repacked_meta else None,
        "original_to_stored": repacked_meta["original_to_stored"] if repacked_meta else None,
        "permutation_sha256": repacked_meta["permutation_sha256"] if repacked_meta else None,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "processor_revision": runner.revision,
        "model_revision": runner.revision,
        "attn": runner.attn,
        "max_new_tokens": MAX_NEW_TOKENS,
        "logit_tolerance": {"atol": LOGIT_ATOL, "rtol": LOGIT_RTOL},
    }
    return fixture, enc


def build_prefix_variants(loaded, meta: dict, device, config):
    """Build P0 dense, P1 physical, P2 logical from ONE sequential SSD read."""
    original_indices = list(loaded.logical_indices)
    start, count, prefix_len = (int(meta["visual_start"]),
                                int(meta["visual_count"]),
                                int(meta["prefix_len"]))
    selected = [int(x) for x in meta["stored_to_original"][
        :min(count, loaded.selected_chunks * meta["chunk_size"])]]
    expected_p2 = list(range(start)) + [
        start + i for i in sorted(selected)] + list(range(start + count, prefix_len))
    if original_indices != expected_p2:
        raise AssertionError("store.load_prefix is not the expected logical P2")
    p1_indices = (list(range(start)) + [start + i for i in selected]
                  + list(range(start + count, prefix_len)))
    if set(p1_indices) != set(original_indices):
        raise AssertionError("P1/P2 selected sets differ")
    p1_reorder = torch.tensor([original_indices.index(i) for i in p1_indices],
                              dtype=torch.long)
    from transformers import DynamicCache

    def build(layers, dense=False, reorder=None):
        c = DynamicCache(config=config)
        for li, (k_cpu, v_cpu) in enumerate(layers):
            pair = []
            for src in (k_cpu, v_cpu):
                if reorder is not None:
                    src = src.index_select(2, reorder)
                if dense:
                    buf = torch.zeros((1, src.shape[1], prefix_len, src.shape[3]),
                                      dtype=src.dtype, device=device)
                    buf.index_copy_(2, torch.tensor(original_indices,
                                                   device=device), src.to(device))
                    src = buf
                else:
                    src = src.to(device)
                pair.append(src)
            c.update(pair[0], pair[1], li)
        return c

    p0 = build(loaded.layers, dense=True)
    p1 = build(loaded.layers, reorder=p1_reorder)
    p2 = build(loaded.layers)
    dense_mask = torch.zeros((1, prefix_len), dtype=torch.long, device=device)
    dense_mask[0, original_indices] = 1
    return {
        "P0_dense": (p0, list(range(prefix_len)), dense_mask),
        "P1_physical": (p1, p1_indices, torch.ones((1, len(p1_indices)),
                                                dtype=torch.long, device=device)),
        "P2_logical": (p2, original_indices, torch.ones((1, len(original_indices)),
                                                      dtype=torch.long, device=device)),
    }


def path_public(result: dict) -> dict:
    return {key: result[key] for key in (
        "first_token_id", "generated_token_ids", "prediction",
        "sdpa_calls_prefill", "sdpa_call_count_total",
        "state_before", "state_after_prefill", "state_after_decode",
        "predecode_cache_len")}


def run_question(runner, validation, question, meta, store, *,
                 forced_mask: bool, fp32_attention: bool, fixture_path: Path):
    fixture, enc_cpu = prepare_fixture(runner, validation, question, meta)
    json_write_new(fixture_path, fixture)
    prefix_len = fixture["prefix_boundary"]
    full_len = len(fixture["complete_expanded_input_ids"])
    positions_cpu, deltas_cpu = runner._logical_positions(
        enc_cpu["input_ids"], enc_cpu["image_grid_thw"])
    enc = {k: v.to(runner.device) if torch.is_tensor(v) else v
           for k, v in enc_cpu.items()}
    ids = enc["input_ids"]
    positions = positions_cpu.to(runner.device)
    suffix = ids[:, prefix_len:]
    if suffix.shape[1] < 1:
        raise AssertionError("frozen question has no suffix")
    full_mask = torch.ones_like(ids)
    r0 = execute_path(
        runner, input_ids=ids, position_ids=positions,
        cache_position=torch.arange(full_len, device=runner.device),
        attention_mask=full_mask, decode_mask=full_mask,
        suffix_from=prefix_len, prefix_clone_len=prefix_len,
        pixel_kwargs={k: enc[k] for k in ("pixel_values", "image_grid_thw")},
        fp32_attention=fp32_attention)
    prefix_copy = clone_prefix(
        r0["prefix_cache"], prefix_len, runner.model.config.text_config)
    prefix_exact_before = compare_cache_prefix(
        r0["prefix_cache"], prefix_copy, prefix_len)
    stock_positions = position_audit(
        runner, enc_cpu["input_ids"], enc_cpu["image_grid_thw"],
        positions_cpu, deltas_cpu, prefix_len, prefix_copy)
    r1 = execute_path(
        runner, input_ids=suffix, position_ids=positions[:, :, prefix_len:],
        cache_position=torch.arange(prefix_len, full_len,
                                    device=runner.device),
        attention_mask=full_mask, decode_mask=full_mask,
        suffix_from=0, cache=prefix_copy,
        fp32_attention=fp32_attention)
    prefix_exact_after = compare_cache_prefix(
        r0["prefix_cache"], prefix_copy, prefix_len)
    r0_mask, r0_logical_mask = mask_report(
        r0, list(range(prefix_len)), prefix_len, full_len)
    r1_mask, r1_logical_mask = mask_report(
        r1, list(range(prefix_len)), prefix_len, full_len)
    result = {
        "question_id": str(question["question_id"]),
        "fixture": str(fixture_path),
        "R0": path_public(r0), "R1": path_public(r1),
        "R0_R1_logits": compact_diff(r0["logits"], r1["logits"]),
        "R0_R1_generated_same":
        r0["generated_token_ids"] == r1["generated_token_ids"],
        "R0_R1_layer_trace": layer_diffs(r0["trace"], r1["trace"]),
        "prefix_cache_R0_to_R1_before": prefix_exact_before,
        "prefix_cache_R0_to_R1_after": prefix_exact_after,
        "stock_vs_manual_mrope": stock_positions,
        "R0_mask": r0_mask, "R1_mask": r1_mask,
        "R0_R1_mask_pattern": mask_compare(r0_logical_mask, r1_logical_mask),
        "forced_explicit_R0_control": {"status": "NOT RUN"},
        "P0_P1_P2": {"status": "NOT RUN"},
    }
    if forced_mask:
        forced = execute_path(
            runner, input_ids=ids, position_ids=positions,
            cache_position=torch.arange(full_len, device=runner.device),
            attention_mask=explicit_causal_4d(
                full_len, torch.bfloat16, runner.device),
            decode_mask=full_mask, suffix_from=prefix_len,
            pixel_kwargs={k: enc[k] for k in ("pixel_values", "image_grid_thw")},
            fp32_attention=fp32_attention)
        forced_mask_report, forced_logical = mask_report(
            forced, list(range(prefix_len)), prefix_len, full_len)
        result["forced_explicit_R0_control"] = {
            "status": "RUN", "path": path_public(forced),
            "R0_vs_forced_logits": compact_diff(r0["logits"], forced["logits"]),
            "forced_vs_R1_logits": compact_diff(forced["logits"], r1["logits"]),
            "R0_vs_forced_layer_trace": layer_diffs(
                r0["trace"], forced["trace"]),
            "forced_vs_R1_layer_trace": layer_diffs(
                forced["trace"], r1["trace"]),
            "mask": forced_mask_report,
            "mask_vs_R0": mask_compare(forced_logical, r0_logical_mask),
            "mask_vs_R1": mask_compare(forced_logical, r1_logical_mask),
        }
    if store is not None:
        loaded = store.load_prefix(budget=.25)
        if loaded.selected_chunks != budget_chunk_count(meta["n_chunks"], .25):
            raise AssertionError("selected 25% chunk count changed")
        variants = build_prefix_variants(
            loaded, meta, runner.device, runner.model.config.text_config)
        paths = {}
        masks = {}
        for label, (cache, logical_indices, prefix_mask) in variants.items():
            suffix_mask = torch.cat([
                prefix_mask, torch.ones((1, suffix.shape[1]),
                                        dtype=torch.long, device=runner.device)], dim=1)
            out = execute_path(
                runner, input_ids=suffix,
                position_ids=positions[:, :, prefix_len:],
                cache_position=torch.arange(
                    len(logical_indices), len(logical_indices) + suffix.shape[1],
                    device=runner.device),
                attention_mask=suffix_mask, decode_mask=suffix_mask,
                suffix_from=0, cache=cache,
                fp32_attention=False)
            paths[label] = out
            mask_summary, masks[label] = mask_report(
                out, logical_indices, prefix_len, full_len)
            paths[label]["mask_summary"] = mask_summary
        p0, p1, p2 = (paths[x] for x in (
            "P0_dense", "P1_physical", "P2_logical"))
        result["P0_P1_P2"] = {
            "status": "RUN",
            "same_one_store_load_io": loaded.io.summary(),
            "selected_chunks": loaded.selected_chunks,
            "selected_visual_original": list(loaded.selected_visual_original),
            "same_selected_set": (
                set(variants["P1_physical"][1])
                == set(variants["P2_logical"][1])
                == {i for i, value in enumerate(
                    variants["P0_dense"][2][0].tolist()) if value}),
            "paths": {label: {**path_public(out),
                              "mask": out["mask_summary"]}
                      for label, out in paths.items()},
            "P0_vs_P1_logits": compact_diff(p0["logits"], p1["logits"]),
            "P0_vs_P2_logits": compact_diff(p0["logits"], p2["logits"]),
            "P1_vs_P2_logits": compact_diff(p1["logits"], p2["logits"]),
            "P0_vs_P1_generated_same":
            p0["generated_token_ids"] == p1["generated_token_ids"],
            "P0_vs_P2_generated_same":
            p0["generated_token_ids"] == p2["generated_token_ids"],
            "P0_vs_P2_layer_trace": layer_diffs(
                p0["trace"], p2["trace"]),
            "P1_vs_P2_layer_trace": layer_diffs(
                p1["trace"], p2["trace"]),
            "P0_vs_P1_mask": mask_compare(
                masks["P0_dense"], masks["P1_physical"]),
            "P0_vs_P2_mask": mask_compare(
                masks["P0_dense"], masks["P2_logical"]),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="new directory only; existing paths are refused")
    parser.add_argument("--attn", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--question-id", default=PRIMARY_QID)
    parser.add_argument("--all-questions", action="store_true")
    parser.add_argument("--forced-mask-control", action="store_true")
    parser.add_argument("--fp32-attention", action="store_true",
                        help="SDPA decoder Q/K/V only: FP32 math accumulation; "
                             "checkpoint and BF16 model remain unchanged")
    parser.add_argument("--store-dir", type=Path, default=REPACKED,
                        help="frozen SDPA repacked store; pass --no-prefix for R0/R1 only")
    parser.add_argument("--no-prefix", action="store_true",
                        help="skip P0/P1/P2 store paths")
    args = parser.parse_args()
    if args.fp32_attention and args.attn != "sdpa":
        parser.error("--fp32-attention requires --attn sdpa")
    if args.fp32_attention and not args.no_prefix:
        parser.error("FP32 decoder-attention control requires --no-prefix "
                     "because the frozen P store has BF16 SDPA provenance")
    if args.attn == "eager" and not args.no_prefix:
        parser.error("eager requires --no-prefix because the frozen P store "
                     "has SDPA prefix provenance")
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    if validation["status"] != "FAIL" or validation["model_revision"] != CHECKPOINT_REVISION:
        raise AssertionError("unexpected frozen validation identity")
    if sha256_file(VALIDATION) == "":
        raise AssertionError("unreachable validation hash")
    source = validation["source"]
    if source["image_id"] != IMAGE_ID or source["image_sha256"] != IMAGE_SHA256:
        raise AssertionError("frozen validation source changed")
    original = {str(q["question_id"]): q for q in source["questions"]}
    if args.all_questions:
        questions = list(source["questions"])
    elif args.question_id in original:
        questions = [original[args.question_id]]
    else:
        parser.error("question ID is absent from frozen three-question validation")
    current_hashes = {}
    for relative in ("mmimpress/qwen25/runner.py", "mmimpress/qwen25/store.py",
                     "mmimpress/qwen25/vision.py"):
        digest = sha256_file(ROOT / relative)
        current_hashes[relative] = digest
        if validation["source_hashes"][relative] != digest:
            raise AssertionError(f"adapter {relative} changed after frozen validation")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "validation_path": str(VALIDATION),
        "validation_sha256": sha256_file(VALIDATION),
        "script_sha256": sha256_file(Path(__file__)),
        "adapter_sha256": current_hashes,
        "attn": args.attn,
        "fp32_decoder_attention_control": args.fp32_attention,
        "forced_explicit_R0_mask_control": args.forced_mask_control,
        "P_store_enabled": not args.no_prefix,
        "P_store_path": str(args.store_dir) if not args.no_prefix else None,
        "question_ids": [str(q["question_id"]) for q in questions],
        "production_backend_unchanged": True,
    }
    json_write_new(args.out_dir / "provenance.json", provenance)
    runner = Qwen25Runner(attn=args.attn).load()
    store = None
    try:
        runtime = runner.runtime_fingerprint()
        if runtime["max_new_tokens"] != MAX_NEW_TOKENS:
            raise AssertionError("decode limit changed")
        if not args.no_prefix:
            store = runner._activate(args.store_dir, image_sha256=IMAGE_SHA256)
            meta = store.meta
            if meta["chunk_size"] != 64:
                raise AssertionError("frozen store chunk size changed")
            if meta["identity"]["image_sha256"] != IMAGE_SHA256:
                raise AssertionError("frozen store image identity changed")
        else:
            # Still freeze selected-token IDs from frozen metadata without reading
            # payload or using it for inference.
            meta = json.loads((REPACKED / "meta.json").read_text(encoding="utf-8"))
        summary = {
            "status": "RUNNING", "runtime": runtime,
            "model_rope_deltas_initial": tensor_brief(runner.model.model.rope_deltas),
            "questions": {},
        }
        for question in questions:
            qid = str(question["question_id"])
            fixture_path = args.out_dir / f"fixture_{qid}.json"
            trace_path = args.out_dir / f"trace_{qid}.json"
            result = run_question(
                runner, validation, question, meta, store,
                forced_mask=args.forced_mask_control,
                fp32_attention=args.fp32_attention,
                fixture_path=fixture_path)
            json_write_new(trace_path, result)
            summary["questions"][qid] = {
                "trace": str(trace_path), "fixture": str(fixture_path),
                "R0_R1_max_abs_logit_diff":
                result["R0_R1_logits"].get("max_abs_diff"),
                "R0_R1_allclose": result["R0_R1_logits"].get("allclose"),
                "R0_R1_first_divergence":
                result["R0_R1_layer_trace"]["first_divergence"],
                "P0_P1_P2_status": result["P0_P1_P2"]["status"],
            }
            print(json.dumps({"question_id": qid,
                              **summary["questions"][qid]}), flush=True)
        summary["status"] = "COMPLETE"
        json_write_new(args.out_dir / "summary.json", summary)
        return 0
    except Exception as exc:
        json_write_new(args.out_dir / "error.json", {
            "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise
    finally:
        runner.close()


if __name__ == "__main__":
    raise SystemExit(main())

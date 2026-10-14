"""Paper-derived multi-image selective attention for LLaVA/Llama.

The manuscript does not unambiguously specify global versus per-image k.
Callers MUST choose ``global`` or ``per_image`` and report that adaptation.
Images retain their original interleaved positions; all text is recomputed.
This module does not alter the legacy single-image MPIC implementation.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time
from typing import Sequence

import torch
import torch.nn.functional as F

from mmimpress.mpic import (MPICContext, _clear_foreign_attention_bias, _sync,
                            _runtime_provenance,
                            build_causal_mask, relocate_post_rope_keys,
                            seed_dynamic_cache)
from mmimpress.store import IOCounter


@dataclass(frozen=True)
class ImageBinding:
    image_id: str
    context: MPICContext
    start: int

    @property
    def count(self):
        return int(self.context.meta["v_token_num"])


@dataclass(frozen=True)
class MultiRowPlan:
    total: int
    active_positions: torch.Tensor
    text_positions: torch.Tensor
    selected_image_positions: torch.Tensor
    selected_counts: tuple[int, ...]
    selection_policy: str


def build_multi_plan(total: int, spans: Sequence[tuple[int, int]],
                     k: int | None, selection_policy: str) -> MultiRowPlan:
    """k=None recomputes all images; finite k is explicitly global/per-image."""
    if selection_policy not in ("global", "per_image"):
        raise ValueError("choose an explicit global or per_image selection policy")
    if total < 1 or not spans or (k is not None and k < 0):
        raise ValueError("invalid total, image spans, or k")
    text = torch.ones(total, dtype=torch.bool)
    active = torch.ones(total, dtype=torch.bool)
    counts, selected, end, remaining = [], [], 0, k
    for start, count in spans:
        if count < 1 or start < end or start + count > total:
            raise ValueError("image spans must be positive, disjoint, and in input order")
        n = count if k is None else min(count, k if selection_policy == "per_image" else remaining)
        counts.append(n)
        selected.extend(range(start, start + n))
        text[start:start + count] = False
        active[start + n:start + count] = False
        end = start + count
        if remaining is not None:
            remaining -= n if selection_policy == "global" else 0
    positions = active.nonzero(as_tuple=True)[0]
    if not len(positions) or int(positions[-1]) != total - 1:
        raise ValueError("the final prompt token must be active (append a text question)")
    return MultiRowPlan(total, positions, text.nonzero(as_tuple=True)[0],
                        torch.tensor(selected, dtype=torch.long), tuple(counts), selection_policy)


def expand_multi_prompt(runner, prompt: str, contexts: Sequence[MPICContext],
                        image_ids: Sequence[str]):
    if not contexts or len(contexts) != len(image_ids):
        raise ValueError("one explicit cache identity per image occurrence is required")
    ids = runner.processor.tokenizer(prompt, return_tensors="pt").input_ids[0].cpu()
    image_at = (ids == runner.image_token_id).nonzero(as_tuple=True)[0].tolist()
    if len(image_at) != len(contexts):
        raise ValueError("image placeholder count does not match cache bindings")
    pieces, bindings, cursor, offset = [], [], 0, 0
    for at, ctx, image_id in zip(image_at, contexts, image_ids):
        if str(ctx.meta.get("image_id")) != str(image_id):
            raise ValueError("image/cache identity mismatch")
        n = int(ctx.meta["v_token_num"])
        pieces += [ids[cursor:at], torch.full((n,), runner.image_token_id, dtype=ids.dtype)]
        bindings.append(ImageBinding(str(image_id), ctx, at + offset))
        offset += n - 1
        cursor = at + 1
    pieces.append(ids[cursor:])
    return torch.cat(pieces), bindings


def assemble_multi_kv(image_pairs, active_k, active_v, positions, spans, total,
                       dummy_value=0.):
    if active_k.shape != active_v.shape or active_k.ndim != 4:
        raise ValueError("active K/V dimensions disagree")
    shape = (1, active_k.shape[1], total, active_k.shape[-1])
    fk = torch.full(shape, dummy_value, dtype=active_k.dtype, device=active_k.device)
    fv = torch.full_like(fk, dummy_value)
    covered = torch.zeros(total, dtype=torch.bool, device=active_k.device)
    for (ik, iv), (start, n) in zip(image_pairs, spans):
        if ik.shape != iv.shape or tuple(ik.shape) != (1, shape[1], n, shape[-1]):
            raise ValueError("image K/V dimensions disagree with active projections")
        fk[:, :, start:start + n] = ik
        fv[:, :, start:start + n] = iv
        covered[start:start + n] = True
    fk.index_copy_(2, positions, active_k)
    fv.index_copy_(2, positions, active_v)
    covered[positions] = True
    if not bool(covered.all()):
        raise AssertionError("uncomputed dummy text would enter attention")
    return fk, fv


class MultiSelectivePrefill:
    def __init__(self, runner, *, k_recompute=32, selection_policy: str,
                 dummy_value=0.):
        self.runner, self.k = runner, k_recompute
        self.policy, self.dummy = selection_policy, dummy_value

    @torch.no_grad()
    def run(self, ids: torch.Tensor, bindings: Sequence[ImageBinding],
            counter: IOCounter, *, trace_tensors=False):
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
        runner, model = self.runner, self.runner.model
        text_config = getattr(model.config, "text_config", model.config)
        if getattr(text_config, "model_type", None) != "llama":
            raise ValueError("this adaptation supports the Llama LLaVA backbone only")
        if any(layer.training for layer in runner.layers):
            raise ValueError("selective prefill requires an eval-mode decoder")
        lm, device = model.model.language_model, torch.device(model.device)
        spans = [(b.start, b.count) for b in bindings]
        plan = build_multi_plan(len(ids), spans, self.k, self.policy)
        if ids.ndim != 1:
            raise ValueError("batch one unpadded expanded IDs required")
        for b in bindings:
            if not bool((ids[b.start:b.start + b.count] == runner.image_token_id).all()):
                raise ValueError("bound span is not the expanded image token interval")
            if int(b.context.meta["num_layers"]) != len(runner.layers):
                raise ValueError("cache layer count differs from active model")
        persisted = [b for b in bindings if "model_config_sha256" in b.context.meta]
        if persisted:
            identity = _runtime_provenance(runner)
            for b in persisted:
                for key, value in identity.items():
                    if b.context.meta.get(key) != value:
                        raise ValueError(f"image store/runtime mismatch for {key}")
        t0 = time.perf_counter()
        pos = plan.active_positions.to(device)
        dtype = model.get_input_embeddings().weight.dtype
        hidden = model.get_input_embeddings()(ids[plan.active_positions].to(device)[None])
        for b, count in zip(bindings, plan.selected_counts):
            inputs = b.context.reader.read_visual_inputs(count, counter).to(device, dtype)
            mask = (pos >= b.start) & (pos < b.start + count)
            hidden[:, mask] = inputs[None]
        mask = build_causal_mask(pos, len(ids), hidden.dtype)
        caches, records, tensors = [], [], []
        for li, layer in enumerate(runner.layers):
            image_pairs = []
            for b, count in zip(bindings, plan.selected_counts):
                ctx = b.context
                rk, ck = ctx.reader.read_reused_layer(li, "k", count, counter)
                rv, cv = ctx.reader.read_reused_layer(li, "v", count, counter)
                if not torch.equal(rk, rv):
                    raise AssertionError("K/V image read ranges differ")
                shape = (1, int(ctx.meta["num_heads"]), b.count, int(ctx.meta["head_dim"]))
                ik = torch.full(shape, self.dummy, dtype=hidden.dtype, device=device)
                iv = torch.full_like(ik, self.dummy)
                if len(rk):
                    ri = rk.to(device)
                    ik.index_copy_(2, ri, ck.to(device, hidden.dtype).permute(1, 0, 2)[None])
                    iv.index_copy_(2, ri, cv.to(device, hidden.dtype).permute(1, 0, 2)[None])
                source = int(ctx.meta["v_token_start"])
                if source != b.start:
                    sp = torch.arange(source, source + b.count, device=device)[None]
                    tp = torch.arange(b.start, b.start + b.count, device=device)[None]
                    sc, ss = lm.rotary_emb(ik, sp)
                    tc, ts = lm.rotary_emb(ik, tp)
                    ik = relocate_post_rope_keys(ik, sc, ss, tc, ts)
                image_pairs.append((ik, iv))
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            attn = layer.self_attn
            shape = (*normalized.shape[:-1], -1, attn.head_dim)
            q = attn.q_proj(normalized).view(shape).transpose(1, 2)
            ak = attn.k_proj(normalized).view(shape).transpose(1, 2)
            av = attn.v_proj(normalized).view(shape).transpose(1, 2)
            cos, sin = lm.rotary_emb(normalized, pos[None])
            q, ak = apply_rotary_pos_emb(q, ak, cos, sin)
            fk, fv = assemble_multi_kv(image_pairs, ak, av, pos, spans, len(ids), self.dummy)
            keys, values = repeat_kv(fk, attn.num_key_value_groups), repeat_kv(fv, attn.num_key_value_groups)
            probabilities = F.softmax(torch.matmul(q, keys.transpose(-2, -1)) * attn.scaling + mask,
                                      dim=-1, dtype=torch.float32).to(q.dtype)
            output = torch.matmul(probabilities, values).transpose(1, 2).contiguous().reshape(*normalized.shape[:-1], -1)
            hidden = residual + attn.o_proj(output)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            caches.append((fk, fv))
            reused_ok = []
            for (ik, iv), b, count in zip(image_pairs, bindings, plan.selected_counts):
                reused_ok.append(bool(torch.equal(fk[:, :, b.start + count:b.start + b.count], ik[:, :, count:])
                                      and torch.equal(fv[:, :, b.start + count:b.start + b.count], iv[:, :, count:])))
            records.append({"layer": li, "Q_shape": list(q.shape), "active_K_shape": list(ak.shape),
                "active_V_shape": list(av.shape), "full_K_shape": list(fk.shape), "full_V_shape": list(fv.shape),
                "hidden_shape": list(hidden.shape), "attention_mask_shape": list(mask.shape),
                "reused_unchanged_per_image": reused_ok,
                "active_K_installed": bool(torch.equal(fk[:, :, pos], ak)),
                "active_V_installed": bool(torch.equal(fv[:, :, pos], av)),
                "future_probability_max": float(probabilities.masked_select(mask.expand_as(probabilities) != 0).abs().max())
                if bool((mask != 0).any()) else 0.})
            if trace_tensors:
                # Explicit diagnostic opt-in: copied CPU tensors never persist
                # on serving contexts or affect the next request.
                tensors.append({name: value.detach().cpu().clone() for name, value in
                    (("Q", q), ("active_K", ak), ("active_V", av), ("K", fk), ("V", fv),
                     ("hidden", hidden), ("mask", mask))})
        logits = model.lm_head(lm.norm(hidden)[:, -1:])
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("non-finite selective-prefill logits")
        _sync(device)
        stats = {"selection_policy": self.policy, "k_recompute_requested": self.k,
            "active_positions": plan.active_positions.tolist(), "text_positions": plan.text_positions.tolist(),
            "selected_image_positions": plan.selected_image_positions.tolist(),
            "selected_image_local_rows": [list(range(c)) for c in plan.selected_counts],
            "bindings": [{"image_id": b.image_id, "source_start": int(b.context.meta["v_token_start"]),
                          "target_start": b.start, "tokens": b.count,
                          "structural_local_rows": b.context.meta.get("newline_idx", [])} for b in bindings],
            "layers": records, "decoder_prefill_pass_count": 1, "vision_forward_count": 0,
            "retained_image_context_ratio": 1., "prefill_inclusive_ms": (time.perf_counter() - t0) * 1000,
            "position_policy": "post-RoPE phase relocation; reused hidden remains source-context dependent"}
        if trace_tensors:
            stats["diagnostic_tensors"] = tensors
        return logits, caches, plan, stats


@torch.no_grad()
def decode_linked(runner, logits, layers, *, max_new_tokens):
    if max_new_tokens < 1:
        raise ValueError("positive max_new_tokens required")
    model, device = runner.model, torch.device(runner.model.device)
    cache = seed_dynamic_cache(layers, config=model.config)
    prefix_len = int(layers[0][0].shape[-2])
    first = int(logits[0, -1].argmax())
    _sync(device)
    first_at = time.perf_counter()
    tokens, current = [first], prefix_len
    while tokens[-1] != runner.processor.tokenizer.eos_token_id and len(tokens) < max_new_tokens:
        cp = torch.tensor([current], device=device)
        output = model(input_ids=torch.tensor([[tokens[-1]]], device=device),
            attention_mask=torch.ones(1, current + 1, dtype=torch.long, device=device),
            position_ids=cp[None], cache_position=cp, past_key_values=cache, use_cache=True)
        tokens.append(int(output.logits[0, -1].argmax()))
        current += 1
    _sync(device)
    lengths = [int(x.keys.shape[-2]) for x in cache.layers]
    if lengths != [prefix_len + len(tokens) - 1] * len(layers):
        raise AssertionError("decode did not append exactly one row per forwarded token")
    return {"first_token_at_s": first_at, "generated_token_ids": tokens, "cache_lengths": lengths,
            "prediction": runner.processor.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
            "cap_reached": len(tokens) == max_new_tokens and tokens[-1] != runner.processor.tokenizer.eos_token_id}


class MPICMultiServer:
    def __init__(self, runner, *, selection_policy: str, k_recompute=32, max_new_tokens=256):
        if selection_policy not in ("global", "per_image"):
            raise ValueError("explicit selection policy required")
        self.runner, self.policy, self.k, self.cap = runner, selection_policy, k_recompute, max_new_tokens

    @torch.no_grad()
    def request(self, contexts, *, prompt_text, image_ids, cold=False, trace_tensors=False):
        _clear_foreign_attention_bias()
        if cold:
            for ctx in dict.fromkeys(contexts):
                ctx.reader.drop_all()
        _sync(torch.device(self.runner.model.device))
        started = time.perf_counter()
        ids, bindings = expand_multi_prompt(self.runner, prompt_text, contexts, image_ids)
        io = IOCounter()
        logits, cache, plan, stats = MultiSelectivePrefill(self.runner, selection_policy=self.policy,
            k_recompute=self.k).run(ids, bindings, io, trace_tensors=trace_tensors)
        result = decode_linked(self.runner, logits, cache, max_new_tokens=self.cap)
        result.update(stats, method="MPIC-32 (adapted)", request_started_at_s=started,
            ttft_ms=(result["first_token_at_s"] - started) * 1000, io=io.summary(),
            token_types=["image" if int(x) == self.runner.image_token_id else "text" for x in ids],
            input_ids=ids.tolist(), fallback=False)
        _clear_foreign_attention_bias()
        return result


class ResidentReader:
    """Request-scoped prepared KV; every actual read is attributed once."""
    def __init__(self, original, counts, io):
        self.original, self.meta, self.counts, self.io = original, original.meta, counts, io
        self.visual = original.read_visual_inputs(counts, io)
        self.payload = {(li, kind): original.read_reused_layer(li, kind, counts, io)
                        for li in range(int(self.meta["num_layers"])) for kind in ("k", "v")}

    def read_visual_inputs(self, k, counter=None):
        if k != self.counts:
            raise ValueError("prepared reader used with a different k")
        return self.visual

    def read_reused_layer(self, layer, kind, k, counter=None):
        if k != self.counts:
            raise ValueError("prepared reader used with a different k")
        return self.payload[layer, kind]


def prepare_mixed(image_ids, existing, miss_builder, *, selected_counts, parallel=True):
    """Overlap actual SSD hit reads on a worker with independent miss compute.

    miss_builder(image_id) must return a validated independent-image context.
    No hidden fallback. Returned readers live only for this request. Host
    timestamps bound real task intervals; GPU kernel overlap needs a profiler.
    """
    from types import SimpleNamespace
    if len(image_ids) != len(selected_counts) or any(k < 0 for k in selected_counts):
        raise ValueError("one nonnegative selected count per image occurrence required")
    timeline, prepared, io = [], {}, IOCounter()
    hit_ids = list(dict.fromkeys(i for i in image_ids if i in existing))
    miss_ids = list(dict.fromkeys(i for i in image_ids if i not in existing))
    counts = {}
    for image_id, k in zip(image_ids, selected_counts):
        if image_id in counts and counts[image_id] != k:
            raise ValueError("repeated image with different selection counts needs occurrence-owned preparation")
        counts[image_id] = k

    def load_hits():
        for image_id in hit_ids:
            start = time.perf_counter()
            ctx = existing[image_id]
            reader = ResidentReader(ctx.reader, counts[image_id], io)
            prepared[image_id] = SimpleNamespace(meta=ctx.meta, reader=reader, dir=ctx.dir)
            timeline.append({"kind": "hit_read", "image_id": image_id, "thread": threading.get_ident(),
                             "start_s": start, "end_s": time.perf_counter()})

    def compute_misses():
        for image_id in miss_ids:
            start = time.perf_counter()
            prepared[image_id] = miss_builder(image_id)
            timeline.append({"kind": "miss_compute", "image_id": image_id, "thread": threading.get_ident(),
                             "start_s": start, "end_s": time.perf_counter()})

    started = time.perf_counter()
    if parallel and hit_ids and miss_ids:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mpic_hit_reader") as pool:
            task = pool.submit(load_hits)
            compute_misses()
            task.result()  # propagate failed loads; no silently substituted context
    else:
        load_hits()
        compute_misses()
    overlap = sum(max(0., min(h["end_s"], m["end_s"]) - max(h["start_s"], m["start_s"]))
                  for h in timeline if h["kind"] == "hit_read" for m in timeline if m["kind"] == "miss_compute")
    return [prepared[i] for i in image_ids], {"timeline": timeline, "io": io.summary(),
        "wall_ms": (time.perf_counter() - started) * 1000, "host_task_overlap_ms": overlap * 1000,
        "parallel_requested": parallel, "mixed": bool(hit_ids and miss_ids), "fallback": False,
        "GPU_kernel_overlap": "NOT VERIFIED by host timestamps alone"}

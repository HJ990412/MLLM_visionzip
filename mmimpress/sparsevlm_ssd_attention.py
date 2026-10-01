"""Scoped, projection-sharing LLaVA SSD KV25 attention (no global patch).

Only attention instances in this request are wrapped. Cached visual keys are
post-RoPE; only newly projected suffix/decode keys receive rotary embeddings.
The dense cache preserves original logical positions and is not a memory
compression claim. This module deliberately does not import legacy serve.py.
"""
from __future__ import annotations

import sys
import time
import types
from collections import defaultdict
from contextlib import AbstractContextManager

import torch
from transformers import DynamicCache
from transformers.models.llama import modeling_llama as ml

from mmimpress.sparsevlm_ssd_core import (
    select_raters, scoring_head_ids, score_visual, exact_topk,
    canonical_chunk_plan,
)

METHOD_POLICIES = {
    "sparsevlm_ssd_kv25_probe3": "fixed_first_3",
    "sparsevlm_ssd_kv25_allhead": "all",
}


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _native_eager():
    # serve.py installs one pre-existing global BIAS function at import time.
    # Call its saved native implementation, never stack another global patch.
    serve = sys.modules.get("mmimpress.serve")
    fn = ml.eager_attention_forward
    if serve is not None and fn is getattr(serve, "_eager_with_bias", None):
        if serve.BIAS:
            raise RuntimeError("legacy BIAS is active; concurrent methods are unsafe")
        fn = serve._ORIG_EAGER
    if fn.__module__ != ml.__name__ or fn.__name__ != "eager_attention_forward":
        raise RuntimeError("unrecognized global LLaMA attention patch")
    return fn


class StageTimes:
    """Non-additive host ranges and CUDA kernel ranges; no stage synchronizes."""
    def __init__(self, device):
        self.device = torch.device(device)
        self.host = defaultdict(float)
        self.events = defaultdict(list)

    def run(self, name, operation):
        start_time = time.perf_counter()
        if self.device.type == "cuda":
            a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
            result = operation()
            b.record()
            self.events[name].append((a, b))
        else:
            result = operation()
        self.host[name] += (time.perf_counter() - start_time) * 1000
        return result

    def result(self):
        return {
            "host_intervals_ms": dict(self.host),
            "cuda_intervals_ms": {k: sum(a.elapsed_time(b) for a, b in v)
                                  for k, v in self.events.items()},
            "additive": False,
            "semantics": "overlapping host/CUDA intervals; never sum as TTFT",
        }


class SparseVLMSSDAttention(AbstractContextManager):
    """One request's instance-local adapter, explicit prefill/decode phases.

    ``observer(dict)`` is a timing-excluded correctness diagnostic callback.
    It receives the same operands and final assembled cache/mask immediately
    before native eager attention; it must not retain tensors in measured runs.
    ``sentinel`` modifies unselected payload only after score/IDs are fixed.
    """
    def __init__(self, runner, ctx, reader, cache, rater_rows, method_id,
                 *, observer=None, sentinel=None, controlled_selected=None):
        if method_id not in METHOD_POLICIES:
            raise ValueError("explicit SparseVLM SSD method_id required")
        self.policy = METHOD_POLICIES[method_id]
        if ctx.head_policy != self.policy:
            raise ValueError("context head policy disagrees with method_id")
        self.runner, self.ctx, self.reader, self.cache = runner, ctx, reader, cache
        self.raters = (rater_rows.detach().cpu().tolist() if torch.is_tensor(rater_rows) else list(rater_rows))
        self.method_id = method_id
        self.observer, self.sentinel = observer, sentinel
        self.controlled_selected = controlled_selected
        self.phase = "prefill"
        self.done, self.keep, self.logs = set(), {}, []
        self.saved, self.handles = [], []
        self.calls = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.times = StageTimes(runner.model.device)
        self.h2d_bytes = 0
        self.native_eager = _native_eager()
        cfg = runner.cfg.text_config
        self.head_ids = scoring_head_ids(self.policy, cfg.num_attention_heads,
                                         cfg.num_key_value_heads)
        m = ctx.meta
        if (cfg.num_attention_heads != m["num_heads"] or
                len(runner.layers) != m["num_layers"] or
                runner.head_dim != m["head_dim"]):
            raise ValueError("model and native payload shape disagree")
        if cfg._attn_implementation != "eager" or runner.model.training:
            raise ValueError("frozen adapter requires eval/eager attention")
        if sentinel is not None and not torch.isfinite(torch.tensor(sentinel)):
            raise ValueError("sentinel must be finite")

    def __enter__(self):
        try:
            for li, layer in enumerate(self.runner.layers):
                attn = layer.self_attn
                if hasattr(attn, "_sparsevlm_ssd_adapter"):
                    raise RuntimeError("attention instance already has an active request")
                own_forward = attn.__dict__.get("forward")
                self.saved.append((attn, own_forward))
                attn._sparsevlm_ssd_adapter = self
                for kind in ("q", "k", "v"):
                    def count(_module, _args, _out, li=li, kind=kind):
                        self.calls[li][self.phase][kind] += 1
                    self.handles.append(getattr(attn, kind + "_proj").register_forward_hook(count))
                def forward(module, hidden_states, position_embeddings,
                            attention_mask=None, past_key_values=None,
                            cache_position=None, _li=li, **kwargs):
                    return self.forward(_li, module, hidden_states, position_embeddings,
                                        attention_mask, past_key_values, cache_position, **kwargs)
                attn.forward = types.MethodType(forward, attn)
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for attn, old in reversed(self.saved):
            if old is None:
                attn.__dict__.pop("forward", None)
            else:
                attn.forward = old
            if getattr(attn, "_sparsevlm_ssd_adapter", None) is self:
                delattr(attn, "_sparsevlm_ssd_adapter")
        self.saved.clear()
        return False

    def begin_decode(self):
        if self.done != set(range(len(self.runner.layers))):
            raise RuntimeError("all layers must score once before decoding")
        self.phase = "decode"

    def _to_compute(self, tensor, device, dtype, stage):
        self.h2d_bytes += tensor.numel() * torch.empty((), dtype=dtype).element_size()
        return self.times.run(stage, lambda: tensor.to(device, dtype))

    def forward(self, li, module, hidden, position_embeddings, attention_mask,
                past_key_values, cache_position, **kwargs):
        if past_key_values is not self.cache:
            raise ValueError("request cache mismatch")
        if hidden.shape[0] != 1:
            raise ValueError("batch_size must be one")
        if self.phase not in ("prefill", "decode"):
            raise RuntimeError("invalid explicit phase")
        if self.phase == "prefill" and li in self.done:
            raise RuntimeError("duplicate prefill for layer")
        m = self.ctx.meta
        length, H, D = hidden.shape[1], m["num_heads"], m["head_dim"]
        start, V, P = m["v_token_start"], m["v_token_num"], m["prefix_len"]
        if cache_position is None:
            raise ValueError("full logical cache positions required")
        expected_start = P if self.phase == "prefill" else self.cache.get_seq_length(li)
        expected_positions = torch.arange(expected_start, expected_start + length,
                                           device=cache_position.device)
        positions_valid = (cache_position == expected_positions).all()
        if cache_position.device.type == "cuda":
            torch._assert_async(positions_valid, "suffix positions were compacted or reused")
        elif not bool(positions_valid):
            raise ValueError("suffix positions were compacted or reused")
        def project():
            q = module.q_proj(hidden).view(1, length, H, D).transpose(1, 2)
            k = module.k_proj(hidden).view(1, length, H, D).transpose(1, 2)
            v = module.v_proj(hidden).view(1, length, H, D).transpose(1, 2)
            q, k = ml.apply_rotary_pos_emb(q, k, *position_embeddings)
            return q, k, v
        q, suffix_k, suffix_v = self.times.run("projection_" + self.phase, project)
        diagnostic = {}
        if self.phase == "prefill":
            t0 = time.perf_counter()
            scoring_cpu = self.reader.read_scoring_keys(li)
            visual_k = self._to_compute(scoring_cpu, q.device, q.dtype, "scoring_k_h2d").permute(1, 0, 2)
            system_k = self.cache.layers[li].keys[0, :, :start]
            scores = self.times.run("scoring", lambda: score_visual(
                q[0], system_k, visual_k, suffix_k[0], self.raters,
                head_policy=self.policy,
                visual_valid_mask=(None if not m.get("padding_idx") else
                    torch.tensor([i not in set(m["padding_idx"]) for i in range(V)],
                                 device=q.device, dtype=torch.bool))))
            selected_tensor = self.times.run("topk", lambda: exact_topk(
                scores, structural_ids=m["newline_idx"], padding_ids=m.get("padding_idx", [])))
            selected = selected_tensor.detach().cpu().tolist()
            if self.controlled_selected is not None:
                selected = list(self.controlled_selected[li])
            plan = canonical_chunk_plan(selected, V, structural_ids=m["newline_idx"],
                                        padding_ids=m.get("padding_idx", []), chunk_size=64)
            payload = self.reader.read_selected(li, plan, scoring_cpu)
            positions = payload.rows.to(q.device)
            kept = torch.zeros(V, device=q.device, dtype=torch.bool)
            kept[positions] = True
            if self.policy == "all":
                # Reuse the already transferred full K; no selected-K read/H2D.
                keys = visual_k[:, positions]
            else:
                keys = self._to_compute(payload.keys, q.device, q.dtype, "selected_k_h2d").permute(1, 0, 2)
            values = self._to_compute(payload.values, q.device, q.dtype, "selected_v_h2d").permute(1, 0, 2)
            def scatter():
                self.cache.layers[li].keys[0, :, positions + start] = keys
                self.cache.layers[li].values[0, :, positions + start] = values
            self.times.run("assembly", scatter)
            self.keep[li] = kept
            self.done.add(li)
            self.logs.append({"layer": li, "n_content": m["n_spatial"],
                              "k": len(selected), "selected_token_ids": selected,
                              "scoring_head_ids": list(self.head_ids), "scoring_calls": 1,
                              "selected_chunk_ids": plan["selected_chunks"],
                              "plan": plan, "io": payload.stats,
                              "cache_shape": list(self.cache.layers[li].keys.shape),
                              "keep_visual_rows": int(positions.numel()),
                              "selector_host_ms": (time.perf_counter() - t0) * 1000})
            if self.observer is not None:
                diagnostic = {"query": q, "system_key": system_k,
                              "visual_key": visual_k, "suffix_key": suffix_k,
                              "scores": scores, "selected": selected,
                              "plan": plan, "rater_rows": self.raters}
            del scoring_cpu, payload, keys, values
        elif li not in self.done:
            raise RuntimeError("decode before layer selection")
        k, v = self.cache.update(suffix_k, suffix_v, li,
                                 {"cos": position_embeddings[0], "sin": position_embeddings[1],
                                  "cache_position": cache_position})
        keep = self.keep[li]
        if self.sentinel is not None:
            # Score and selection are final; sentinel never enters score operands.
            k[0, :, start:P][:, ~keep] = float(self.sentinel)
            v[0, :, start:P][:, ~keep] = float(self.sentinel)
        if attention_mask is None:
            key_pos = torch.arange(k.shape[2], device=q.device)
            allowed = key_pos[None, :] <= cache_position[:, None]
            mask = torch.zeros((1, 1, length, k.shape[2]), device=q.device, dtype=q.dtype)
            mask.masked_fill_(~allowed[None, None], float("-inf"))
        else:
            mask = attention_mask[:, :, :, :k.shape[2]].clone()
        mask[..., start:P].masked_fill_(~keep, float("-inf"))
        if self.observer is not None:
            self.observer({"layer": li, "phase": self.phase, "module": module,
                           "key": k, "value": v, "mask": mask, "keep": keep,
                           "cache_position": cache_position, "query": q, **diagnostic})
        output, weights = self.times.run("actual_attention_" + self.phase, lambda:
            self.native_eager(module, q, k, v, mask, scaling=module.scaling,
                              dropout=0.0, **kwargs))
        output = output.reshape(1, length, -1).contiguous()
        return module.o_proj(output), weights

    def stats(self):
        calls = {str(li): {phase: dict(counts) for phase, counts in phases.items()}
                 for li, phases in self.calls.items()}
        for li in range(len(self.runner.layers)):
            if calls.get(str(li), {}).get("prefill") != {"q": 1, "k": 1, "v": 1}:
                raise AssertionError("Q/K/V projection reuse gate failed")
        return {"layers": self.logs, "projection_calls": calls,
                "scoring_calls": len(self.done), "head_policy": self.policy,
                "scoring_head_ids": list(self.head_ids), "timing": self.times.result(),
                "payload_h2d_compute_bytes": self.h2d_bytes,
                "projection_reuse": True, "dense_logical_positions": True,
                "diagnostic": self.observer is not None or self.sentinel is not None
                              or self.controlled_selected is not None}


def new_dense_cache(ctx, device, dtype):
    m = ctx.meta
    cache = DynamicCache()
    for li in range(m["num_layers"]):
        shape = (1, m["num_heads"], m["prefix_len"], m["head_dim"])
        k, v = (torch.zeros(shape, device=device, dtype=dtype) for _ in range(2))
        k[0, :, :m["v_token_start"]] = ctx.sys_kv["k"][li].to(device, dtype)
        v[0, :, :m["v_token_start"]] = ctx.sys_kv["v"][li].to(device, dtype)
        cache.update(k, v, li)
    return cache


class SparseVLMSSDServer:
    def __init__(self, runner, max_new_tokens=16):
        if max_new_tokens != 16:
            raise ValueError("frozen generation budget is 16")
        self.runner, self.max_new_tokens = runner, max_new_tokens

    @torch.inference_mode()
    def request(self, ctx, *, method_id, question=None, prompt_text=None,
                cold=True, observer=None, sentinel=None,
                controlled_selected=None, return_logits=False, text_spans=None):
        if method_id not in METHOD_POLICIES:
            raise ValueError("explicit method_id required")
        if cold:
            ctx.drop_cache()
        runner, model = self.runner, self.runner.model
        device = model.device
        _sync(device)
        if torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()  # before prompt/tokenization/input preparation
        prompt = runner.prompt(question) if prompt_text is None else prompt_text
        tok = runner.processor.tokenizer
        encoded = tok(prompt, return_tensors="pt", return_offsets_mapping=True)
        ids = encoded.input_ids[0]
        offsets = encoded.offset_mapping[0].tolist()
        loc = (ids == runner.image_token_id).nonzero(as_tuple=True)[0]
        if loc.numel() != 1:
            raise ValueError("exactly one image prefix required")
        boundary = int(loc[0])
        prefix = ctx.meta["prefix_input_ids"]
        if ids[:boundary].tolist() != prefix[:ctx.meta["v_token_start"]]:
            raise ValueError("system/user header is incompatible with captured prefix")
        suffix = ids[boundary + 1:]
        if suffix.numel() == 0:
            raise ValueError("empty suffix")
        suffix_offsets = offsets[boundary + 1:]
        span_records = []
        if text_spans is None and question is not None:
            begin = prompt.rfind(question)
            if begin >= 0:
                text_spans = [{"kind": "current_question", "start_char": begin,
                               "end_char": begin + len(question)}]
        for span in text_spans or []:
            lo, hi = int(span["start_char"]), int(span["end_char"])
            if not 0 <= lo <= hi <= len(prompt):
                raise ValueError("invalid text provenance span")
            span_records.append({**span, "suffix_token_ids": [i for i, (a, b) in
                enumerate(suffix_offsets) if b > lo and a < hi]})
        annotated = set(i for span in span_records for i in span["suffix_token_ids"])
        template_ids = [i for i in range(len(suffix_offsets)) if i not in annotated]
        suffix = suffix.to(device)
        rater_started = time.perf_counter()
        embeddings = model.get_input_embeddings()(suffix)
        raters = select_raters(ctx.v_hidden.to(device), embeddings)
        rater_ids = raters.ids.detach().cpu().tolist()
        rater_ms = (time.perf_counter() - rater_started) * 1000
        cache = new_dense_cache(ctx, device, embeddings.dtype)
        P = ctx.meta["prefix_len"]
        logits = None
        all_logits = []
        with ctx.request() as reader:
            adapter = SparseVLMSSDAttention(runner, ctx, reader, cache, rater_ids,
                method_id, observer=observer, sentinel=sentinel,
                controlled_selected=controlled_selected)
            with adapter:
                pos = torch.arange(P, P + suffix.numel(), device=device)
                output = model(input_ids=suffix[None],
                    attention_mask=torch.ones((1, P + suffix.numel()), device=device, dtype=torch.long),
                    position_ids=pos[None], cache_position=pos,
                    past_key_values=cache, use_cache=True)
                first = int(output.logits[0, -1].argmax())
                _sync(device)
                first_at = time.perf_counter()
                if return_logits:
                    logits = output.logits[0, -1].float().cpu()
                    all_logits.append(logits)
                tokens = [first]
                adapter.begin_decode()
                cur = P + suffix.numel()
                while tokens[-1] != tok.eos_token_id and len(tokens) < self.max_new_tokens:
                    cp = torch.tensor([cur], device=device)
                    output = model(input_ids=torch.tensor([[tokens[-1]]], device=device),
                        attention_mask=torch.ones((1, cur + 1), device=device, dtype=torch.long),
                        position_ids=cp[None], cache_position=cp,
                        past_key_values=cache, use_cache=True)
                    tokens.append(int(output.logits[0, -1].argmax()))
                    if return_logits:
                        all_logits.append(output.logits[0, -1].float().cpu())
                    cur += 1
                _sync(device)
                finished = time.perf_counter()
            stats = adapter.stats()
            io = reader.summary()
        answer = tok.decode(tokens, skip_special_tokens=True).strip()
        result = {"method_id": method_id, "answer": answer, "prediction": answer,
                  "first_token_id": first, "generated_token_ids": tokens,
                  "generated_token_count": len(tokens),
                  "cap_reached": len(tokens) == self.max_new_tokens and tokens[-1] != tok.eos_token_id,
                  "rater_ids": rater_ids, "rater_fallback": bool(raters.fallback),
                  "rater_scope": "entire_actual_suffix_after_expanded_visual_block",
                  "rater_visual_scope_includes_newlines": True,
                  "suffix_input_ids": suffix.cpu().tolist(),
                  "prompt": prompt, "rater_host_ms": rater_ms,
                  "suffix_token_char_offsets": suffix_offsets,
                  "text_spans": span_records, "template_suffix_token_ids": template_ids,
                  "metadata_activation": getattr(ctx, "activation", {}),
                  "rater_visual_h2d_bytes": ctx.v_hidden.numel() * ctx.v_hidden.element_size(),
                  "rater_embedding_compute_dtype": "float32",
                  "compute_dtype": str(embeddings.dtype), "ssd_dtype": ctx.meta["dtype"],
                  "n_content": ctx.meta["n_spatial"],
                  "content_kv_fraction": ((ctx.meta["n_spatial"] + 3) // 4) / ctx.meta["n_spatial"],
                  "request_started_at_s": started, "first_token_at_s": first_at,
                  "model_finished_at_s": finished,
                  "end_to_end_ttft_ms": (first_at - started) * 1000,
                  "request_e2e_ms": (finished - started) * 1000,
                  "ttft": first_at - started, "e2e_latency": finished - started,
                  "io": io, "vision_forward_calls": 0, **stats}
        if torch.device(device).type == "cuda":
            result.update(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device))
        if return_logits:
            result["diagnostic"] = True
            result["first_logits"] = logits
            result["logits"] = all_logits
        return result

"""Method-local ReKV internal retrieval for the SSD image adaptation.

The retrieval metadata and payload format live in ``rekv_store``.  This module
only changes the attention modules while a ReKV request is active; it never
changes the existing QA-Chunk/FullLoad cache or their global attention bias.
The mathematical reference is Becomebright/ReKV commit
1fd9a3dbf5dbff7f27069ae2f4463674c495e830, notably
``rekv_attention_forward`` and ``RotaryEmbeddingESM.forward``.  This is a
fresh implementation for the local LLaVA-NeXT eager-attention interface.
"""
from __future__ import annotations

import hashlib
import resource
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import transformers.models.llama.modeling_llama as llama

from mmimpress.cvpr25 import budget_chunk_count
from mmimpress.serve import (BIAS, contiguous_runs, suffix_ids_for,
                            suffix_ids_from_prompt)
from mmimpress.store import IOCounter


SIMILARITY_MODE = "official_code_dot"
RETRIEVAL_GROUP_SIZE = 1
# The pinned official evaluation uses 15,000.  The active compact image
# context is checked against this limit per request; the larger-context
# init/local branch is also implemented and unit tested.
N_LOCAL = 15_000


def mean_head_vector(raw: torch.Tensor) -> torch.Tensor:
    """Official head-concatenated token mean, input ``(B,H,T,D)``."""
    if raw.ndim != 4 or raw.shape[0] != 1 or raw.shape[2] < 1:
        raise ValueError(f"expected (1,H,T,D), got {tuple(raw.shape)}")
    return raw.mean(dim=2).reshape(1, -1)


def official_dot_scores(q_rep: torch.Tensor,
                        k_rep: torch.Tensor) -> torch.Tensor:
    """The official vector-cache function casts to FP32 then takes a dot."""
    if q_rep.ndim == 2:
        if q_rep.shape[0] != 1:
            raise ValueError("one question at a time")
        q_rep = q_rep[0]
    if q_rep.ndim != 1 or k_rep.ndim != 2 or k_rep.shape[1] != q_rep.numel():
        raise ValueError((tuple(q_rep.shape), tuple(k_rep.shape)))
    return torch.matmul(k_rep.float(), q_rep.float())


def select_chunks(scores: torch.Tensor, valid_counts: torch.Tensor,
                  ratio: float = 0.25) -> tuple[list[int], list[int]]:
    """Official non-tie Top-k; lower chunk ID wins a tie, then source order."""
    if scores.ndim != 1 or valid_counts.shape != scores.shape:
        raise ValueError("scores/counts shape mismatch")
    candidates = torch.nonzero(valid_counts > 0, as_tuple=True)[0]
    if candidates.numel() == 0:
        raise ValueError("image has no spatial-bearing chunk")
    # This is the same Python round/clamp helper used by QA-Chunk and Ours.
    k = budget_chunk_count(int(scores.numel()), float(ratio))
    if not 0 < k <= int(candidates.numel()):
        raise AssertionError(
            f"nominal chunk budget {k} exceeds {candidates.numel()} "
            "spatial-bearing candidates")
    order = torch.argsort(scores.index_select(0, candidates),
                          descending=True, stable=True)
    ranked = [int(x) for x in candidates.index_select(0, order[:k]).cpu()]
    return ranked, sorted(ranked)


def compact_visual_rows(chunks: list[int], separators: list[int],
                        v_num: int, chunk_size: int) -> list[int]:
    """Selected complete physical chunks plus all separators, with no holes."""
    rows = set(int(x) for x in separators)
    for chunk in chunks:
        start = int(chunk) * int(chunk_size)
        if start < 0 or start >= v_num:
            raise ValueError(f"chunk outside visual range: {chunk}")
        rows.update(range(start, min(start + chunk_size, v_num)))
    return sorted(rows)


def _rotate(rotary, tensor: torch.Tensor,
            positions: torch.Tensor,
            timing_ms: dict[str, float] | None = None) -> torch.Tensor:
    started = time.perf_counter()
    cos, sin = rotary(tensor, positions.unsqueeze(0))
    result = llama.apply_rotary_pos_emb(tensor, tensor, cos, sin)[0]
    if timing_ms is not None:
        # Host dispatch interval. CUDA kernels remain asynchronous; this is
        # never described as exclusive GPU compute time.
        timing_ms["rope_ms"] += (time.perf_counter()-started)*1e3
    return result


def rekv_attention_output(attn, rotary, q_raw: torch.Tensor,
                          k_raw: torch.Tensor, v: torch.Tensor, *,
                          n_local: int = N_LOCAL, n_init: int = 0,
                          return_rotated_k: bool = False,
                          timing_ms: dict[str, float] | None = None):
    """ReKV compact RoPE, local-window and initial-prefix attention.

    The ordinary regime (``len_k <= n_local``) is identical to full causal
    attention on a consecutively positioned compact sequence.  Beyond the
    local window, this implements the two disjoint branches of pinned
    ``rekv_attention_forward`` L94–129: right-aligned local RoPE and initial
    raw K with Q at angle ``n_local - 1``.  The split logits share one softmax.
    """
    if q_raw.ndim != 4 or k_raw.ndim != 4 or v.shape != k_raw.shape:
        raise ValueError("expected Q/K/V as (1,H,T,D)")
    if q_raw.shape[0] != 1 or q_raw.shape[1] != k_raw.shape[1]:
        raise ValueError("this adaptation requires single-request MHA")
    nq, nk = q_raw.shape[-2], k_raw.shape[-2]
    if nq < 1 or nk < nq or n_local < 1 or not 0 <= n_init <= nk-nq:
        raise ValueError((nq, nk, n_local, n_init))
    device = q_raw.device
    local_len = min(nk, nq + n_local)
    local_start = nk - local_len
    local_k = k_raw[:, :, local_start:, :]
    local_v = v[:, :, local_start:, :]
    local_positions = torch.arange(local_len, device=device)
    query_positions = local_positions[-nq:]
    local_q_rot = _rotate(rotary, q_raw, query_positions, timing_ms)
    local_k_rot = _rotate(rotary, local_k, local_positions, timing_ms)
    dist = (torch.arange(nq, device=device)[:, None]
            - torch.arange(local_len, device=device)[None, :]
            + local_len - nq)
    local_mask = (dist >= 0) & (dist < n_local)
    local_logits = torch.matmul(
        local_q_rot, local_k_rot.transpose(-1, -2)) * attn.scaling
    local_logits = local_logits.masked_fill(
        ~local_mask[None, None], torch.finfo(local_logits.dtype).min)

    values = [local_v]
    logits = [local_logits]
    branch = "local_only"
    if nk > n_local and n_init:
        # Official code uses raw initial K and fixes Q at n_local - 1.
        init_q = _rotate(rotary, q_raw,
                         torch.full((nq,), n_local - 1, device=device,
                                    dtype=torch.long), timing_ms)
        init_k = k_raw[:, :, :n_init, :]
        init_v = v[:, :, :n_init, :]
        init_dist = (torch.arange(nq, device=device)[:, None]
                     - torch.arange(n_init, device=device)[None, :]
                     + nk - nq)
        init_mask = init_dist >= n_local
        init_logits = torch.matmul(
            init_q, init_k.transpose(-1, -2)) * attn.scaling
        init_logits = init_logits.masked_fill(
            ~init_mask[None, None], torch.finfo(init_logits.dtype).min)
        logits.append(init_logits)
        values.append(init_v)
        branch = "local_plus_init"
    weights = torch.softmax(torch.cat(logits, dim=-1), dim=-1,
                            dtype=torch.float32).to(q_raw.dtype)
    output = torch.matmul(weights, torch.cat(values, dim=-2))
    output = output.transpose(1, 2).contiguous()
    rotated = (local_k_rot if local_start == 0 else None)
    if return_rotated_k:
        return output, rotated, branch
    return output


@dataclass
class LayerHandoff:
    raw_k: torch.Tensor
    value: torch.Tensor
    source_positions: list[int]
    selected_chunks: list[int]
    selected_spatial_tokens: int
    rotated_k: torch.Tensor | None = None
    text_raw_k: torch.Tensor | None = None
    text_v: torch.Tensor | None = None
    text_rotated_k: torch.Tensor | None = None
    branch: str = ""

    @property
    def prefix_len(self) -> int:
        return int(self.raw_k.shape[-2])


@dataclass
class RequestState:
    mode: str = "retrieval"
    layers: dict[int, LayerHandoff] = field(default_factory=dict)
    log: list[dict[str, Any]] = field(default_factory=list)
    timing_ms: dict[str, float] = field(default_factory=lambda: {
        key: 0.0 for key in (
            "q_rep_ms", "similarity_ms", "topk_ms", "selection_d2h_ms",
            "io_planning_ms", "ssd_read_pipeline_ms", "h2d_ms",
            "compact_assembly_ms", "rope_ms",
            "attention_host_wall_ms")})
    stage_a_counter: IOCounter = field(default_factory=IOCounter)
    stage_b_counter: IOCounter = field(default_factory=IOCounter)
    retrieval_calls: int = 0
    answer_calls: int = 0
    separator_kv: Any = None
    pinned_buffers: list[torch.Tensor] = field(default_factory=list)
    peak_pinned_bytes: int = 0


class _ReKVAttentionScope:
    """Patch only this runner's LLaMA attention instances for one request."""

    def __init__(self, runner, ctx, state: RequestState,
                 n_local: int = N_LOCAL, all_blocks: bool = False):
        self.runner, self.ctx, self.state = runner, ctx, state
        self.n_local = int(n_local)
        self.all_blocks = bool(all_blocks)
        self._originals = []
        self.rotary = runner.model.model.language_model.rotary_emb

    def __enter__(self):
        if BIAS:
            raise RuntimeError("foreign attention bias active before ReKV")
        try:
            for li, layer in enumerate(self.runner.layers):
                attn = layer.self_attn
                self._originals.append((attn, attn.forward))
                attn.forward = self._forward_for(li, attn)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        for attn, original in reversed(self._originals):
            attn.forward = original
        self._originals.clear()
        BIAS.clear()

    def _forward_for(self, li, attn):
        def forward(hidden_states, position_embeddings=None,
                    attention_mask=None, past_key_values=None,
                    cache_position=None, **kwargs):
            shape = hidden_states.shape
            if shape[0] != 1:
                raise ValueError("ReKV only supports batch size one")
            cfg = getattr(attn, "config", None)
            h = getattr(cfg, "num_attention_heads",
                        getattr(attn, "num_heads", None))
            kvh = getattr(cfg, "num_key_value_heads",
                          getattr(attn, "num_key_value_heads", None))
            if h is None or kvh is None:
                raise TypeError("attention head counts unavailable")
            h, kvh = int(h), int(kvh)
            if h != kvh:
                raise ValueError("this frozen adaptation requires MHA")
            hd = attn.head_dim
            q = attn.q_proj(hidden_states).view(1, shape[1], h, hd).transpose(1, 2)
            k = attn.k_proj(hidden_states).view(1, shape[1], h, hd).transpose(1, 2)
            v = attn.v_proj(hidden_states).view(1, shape[1], h, hd).transpose(1, 2)
            if self.state.mode == "retrieval":
                out = self._retrieval_layer(li, attn, q, k, v)
            elif self.state.mode == "answer":
                out = self._answer_layer(li, attn, q, k, v)
            else:
                raise RuntimeError(f"invalid ReKV stage: {self.state.mode}")
            out = out.reshape(shape[0], shape[1], -1).contiguous()
            return attn.o_proj(out), None
        return forward

    def _retrieve_payload(self, li: int, chunks: list[int],
                          visual_rows: list[int]):
        ctx, state = self.ctx, self.state
        started = time.perf_counter()
        run_count, run_lengths = contiguous_runs(chunks)
        state.timing_ms["io_planning_ms"] += (time.perf_counter()-started)*1e3
        started = time.perf_counter()
        k_rows, k_vals = ctx.reader.read_chunks(
            li, "k", chunks, state.stage_a_counter)
        v_rows, v_vals = ctx.reader.read_chunks(
            li, "v", chunks, state.stage_a_counter)
        state.timing_ms["ssd_read_pipeline_ms"] += (time.perf_counter()-started)*1e3
        if [int(x) for x in k_rows] != [int(x) for x in v_rows]:
            raise AssertionError("K/V chunk rows differ")
        return k_rows, k_vals, v_vals, run_count, run_lengths

    def _retrieval_layer(self, li, attn, q, k, v):
        ctx, state = self.ctx, self.state
        started = time.perf_counter()
        q_rep = mean_head_vector(q)
        state.timing_ms["q_rep_ms"] += (time.perf_counter()-started)*1e3
        started = time.perf_counter()
        scores = official_dot_scores(q_rep, ctx.k_rep[li])
        state.timing_ms["similarity_ms"] += (time.perf_counter()-started)*1e3
        started = time.perf_counter()
        counts = getattr(ctx, "valid_counts_gpu", None)
        if counts is None:
            # Synthetic parity fixtures may provide only the metadata list;
            # production ReKVContext activates this small mask before TTFT.
            counts = torch.as_tensor(ctx.meta["valid_spatial_counts"],
                                     device=scores.device)
        candidate_ids = torch.nonzero(counts > 0, as_tuple=True)[0]
        if not candidate_ids.numel():
            raise AssertionError("no spatial-bearing chunks")
        k_budget = (int(candidate_ids.numel()) if self.all_blocks else
                    budget_chunk_count(int(scores.numel()), 0.25))
        if k_budget > int(candidate_ids.numel()):
            raise AssertionError("too few spatial-bearing chunks for nominal budget")
        ranked_gpu = candidate_ids.index_select(
            0, torch.argsort(scores.index_select(0, candidate_ids),
                             descending=True, stable=True)[:k_budget])
        state.timing_ms["topk_ms"] += (time.perf_counter()-started)*1e3
        started = time.perf_counter()
        ranked = [int(x) for x in ranked_gpu.cpu()]
        ordered = sorted(ranked)
        state.timing_ms["selection_d2h_ms"] += \
            (time.perf_counter()-started)*1e3
        rows = compact_visual_rows(
            ordered, ctx.meta["newline_idx"], int(ctx.meta["v_token_num"]),
            int(ctx.meta["chunk_size"]))
        disk_rows, disk_k, disk_v, run_count, run_lengths = \
            self._retrieve_payload(li, ordered, rows)
        started = time.perf_counter()
        dev, dtype = q.device, q.dtype
        disk_rows = [int(x) for x in disk_rows]
        row_lookup = {row: idx for idx, row in enumerate(disk_rows)}
        # The all-layer separator sidecar is read once by ReKVServer before
        # Stage A and supplied here; no second SSD access occurs in Stage B.
        sep_positions = [int(x) for x in ctx.meta["newline_idx"]]
        sep_lookup = {row: idx for idx, row in enumerate(sep_positions)}
        sep_k, sep_v = state.separator_kv
        sep_k_li, sep_v_li = sep_k[li], sep_v[li]
        source_k, source_v = [], []
        for row in rows:
            if row in row_lookup:
                idx = row_lookup[row]
                source_k.append(disk_k[idx])
                source_v.append(disk_v[idx])
            else:
                idx = sep_lookup[row]
                source_k.append(sep_k_li[idx])
                source_v.append(sep_v_li[idx])
        visual_k_cpu = torch.stack(source_k).contiguous()
        visual_v_cpu = torch.stack(source_v).contiguous()
        if dev.type == "cuda":
            visual_k_cpu = visual_k_cpu.pin_memory()
            visual_v_cpu = visual_v_cpu.pin_memory()
            state.pinned_buffers.extend((visual_k_cpu, visual_v_cpu))
            state.peak_pinned_bytes = max(
                state.peak_pinned_bytes,
                sum(buf.numel() * buf.element_size()
                    for buf in state.pinned_buffers))
        visual_k = visual_k_cpu.to(dev, dtype, non_blocking=True) \
            .permute(1, 0, 2)[None]
        visual_v = visual_v_cpu.to(dev, dtype, non_blocking=True) \
            .permute(1, 0, 2)[None]
        state.timing_ms["h2d_ms"] += (time.perf_counter()-started)*1e3
        started = time.perf_counter()
        sys_k, sys_v = ctx.sys_kv["k"][li], ctx.sys_kv["v"][li]
        if sys_k.ndim == 3:
            sys_k, sys_v = sys_k[None], sys_v[None]
        sys_k = sys_k.to(dev, dtype)
        sys_v = sys_v.to(dev, dtype)
        raw_k = torch.cat((sys_k, visual_k), dim=-2)
        raw_v = torch.cat((sys_v, visual_v), dim=-2)
        source_positions = list(range(int(ctx.meta["v_token_start"]))) + [
            int(ctx.meta["v_token_start"])+row for row in rows]
        handoff = LayerHandoff(
            raw_k=raw_k, value=raw_v, source_positions=source_positions,
            selected_chunks=ordered,
            selected_spatial_tokens=sum(
                row not in sep_lookup for row in rows))
        state.layers[li] = handoff
        state.timing_ms["compact_assembly_ms"] += \
            (time.perf_counter()-started)*1e3
        started = time.perf_counter()
        full_k = torch.cat((raw_k, k), dim=-2)
        full_v = torch.cat((raw_v, v), dim=-2)
        output, rotated, branch = rekv_attention_output(
            attn, self.rotary, q, full_k, full_v,
            n_local=self.n_local, n_init=int(ctx.meta["v_token_start"]),
            return_rotated_k=True, timing_ms=state.timing_ms)
        # Precompute the exact compact prefix rotation for Stage B; otherwise
        # Stage B would need to recompute it, though it must never re-read SSD.
        if handoff.prefix_len <= self.n_local and rotated is not None:
            handoff.rotated_k = rotated[:, :, :handoff.prefix_len, :]
        handoff.branch = branch
        state.timing_ms["attention_host_wall_ms"] += \
            (time.perf_counter()-started)*1e3
        state.retrieval_calls += 1
        state.log.append({
            "layer": li, "selected_chunk_ids_ranked": ranked,
            "selected_chunk_ids": ordered,
            "source_positions": source_positions,
            "compact_positions": list(range(handoff.prefix_len)),
            "compact_prefix_len": handoff.prefix_len,
            "actual_attention_key_len": int(full_k.shape[-2]),
            "selected_spatial_tokens": handoff.selected_spatial_tokens,
            "candidate_normal_chunks": int(candidate_ids.numel()),
            "nominal_chunk_budget": int(k_budget),
            "contiguous_runs": run_count,
            "contiguous_run_lengths": run_lengths,
            "branch": branch,
        })
        return output

    def _answer_layer(self, li, attn, q, k, v):
        state = self.state
        handoff = state.layers[li]
        prefix = handoff.prefix_len
        prior = 0 if handoff.text_raw_k is None else handoff.text_raw_k.shape[-2]
        # Keep all new prompt/generated K/V as a request-local mutable cache.
        handoff.text_raw_k = (k if prior == 0 else
                              torch.cat((handoff.text_raw_k, k), dim=-2))
        handoff.text_v = (v if prior == 0 else
                          torch.cat((handoff.text_v, v), dim=-2))
        started = time.perf_counter()
        if prefix + prior + q.shape[-2] <= self.n_local:
            if handoff.rotated_k is None:
                handoff.rotated_k = _rotate(
                    self.rotary, handoff.raw_k,
                    torch.arange(prefix, device=q.device), state.timing_ms)
            positions = torch.arange(prefix + prior,
                                     prefix + prior + q.shape[-2],
                                     device=q.device)
            q_rot = _rotate(self.rotary, q, positions, state.timing_ms)
            k_rot = _rotate(self.rotary, k, positions, state.timing_ms)
            handoff.text_rotated_k = (
                k_rot if prior == 0 else
                torch.cat((handoff.text_rotated_k, k_rot), dim=-2))
            whole_k = torch.cat((handoff.rotated_k,
                                 handoff.text_rotated_k), dim=-2)
            whole_v = torch.cat((handoff.value, handoff.text_v), dim=-2)
            key_ids = torch.arange(whole_k.shape[-2], device=q.device)
            mask = key_ids[None, :] <= positions[:, None]
            logits = torch.matmul(q_rot, whole_k.transpose(-1, -2)) \
                * attn.scaling
            logits = logits.masked_fill(
                ~mask[None, None], torch.finfo(logits.dtype).min)
            weights = torch.softmax(logits, dim=-1,
                                    dtype=torch.float32).to(q.dtype)
            output = torch.matmul(weights, whole_v).transpose(1, 2).contiguous()
            handoff.branch = "local_only"
        else:
            # Exact init/local branch for unusually long prompts.  The main
            # pilot records whether this branch is activated.
            full_k = torch.cat((handoff.raw_k, handoff.text_raw_k), dim=-2)
            full_v = torch.cat((handoff.value, handoff.text_v), dim=-2)
            output, _, handoff.branch = rekv_attention_output(
                attn, self.rotary, q, full_k, full_v,
                n_local=self.n_local,
                n_init=int(self.ctx.meta["v_token_start"]),
                return_rotated_k=True, timing_ms=state.timing_ms)
        state.timing_ms["attention_host_wall_ms"] += \
            (time.perf_counter()-started)*1e3
        state.answer_calls += 1
        return output


class ReKVServer:
    """Question-only retrieval followed by separate normal answer prefill."""

    def __init__(self, runner, max_new_tokens: int = 16,
                 n_local: int = N_LOCAL, all_blocks: bool = False,
                 check_finite_logits: bool = False):
        self.runner = runner
        self.max_new_tokens = int(max_new_tokens)
        self.n_local = int(n_local)
        self.all_blocks = bool(all_blocks)
        self.check_finite_logits = bool(check_finite_logits)

    @torch.no_grad()
    def request(self, ctx, *, question: str | None = None,
                prompt_text: str | None = None,
                retrieval_text: str | None = None,
                cold: bool = True) -> dict:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if (question is None) == (prompt_text is None):
            raise ValueError("provide exactly one of question or prompt_text")
        if prompt_text is not None and retrieval_text is None:
            raise ValueError("MT prompt requires explicit causal retrieval_text")
        query_text = question if prompt_text is None else retrieval_text
        if cold:
            ctx.reader.drop_all()  # outside TTFT, matching the other arms
        device = self.runner.model.device
        torch.cuda.synchronize(device)
        gpu_alloc_before = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        prep_at = time.perf_counter()
        tokenizer = self.runner.processor.tokenizer
        question_ids_cpu = tokenizer(query_text, return_tensors="pt").input_ids
        question_hash = hashlib.sha256(
            question_ids_cpu.numpy().tobytes()).hexdigest()
        question_ids = question_ids_cpu.to(device)
        suffix = (suffix_ids_for(self.runner, question)
                  if prompt_text is None else
                  suffix_ids_from_prompt(self.runner, prompt_text)).to(device)
        request_prep_ms = (time.perf_counter()-prep_at)*1e3
        state = RequestState()
        # Sidecar is one request-local SSD read.  It is never resident across
        # requests, and Stage B uses these same tensors through handoff.
        retrieval_event_start = torch.cuda.Event(enable_timing=True)
        retrieval_event_end = torch.cuda.Event(enable_timing=True)
        retrieval_event_start.record(torch.cuda.current_stream(device))
        retrieval_started = time.perf_counter()
        sep_started = time.perf_counter()
        state.separator_kv = ctx.read_sep_kv(state.stage_a_counter)
        state.timing_ms["ssd_read_pipeline_ms"] += (time.perf_counter()-sep_started)*1e3
        first_at = finished_at = None
        try:
            with _ReKVAttentionScope(
                    self.runner, ctx, state, n_local=self.n_local,
                    all_blocks=self.all_blocks):
                self.runner.model(input_ids=question_ids, use_cache=False,
                                  return_dict=True)
                retrieval_event_end.record(torch.cuda.current_stream(device))
                retrieval_done = time.perf_counter()
                if len(state.layers) != len(self.runner.layers):
                    raise AssertionError("retrieval did not traverse every layer")
                if any(h.text_raw_k is not None for h in state.layers.values()):
                    raise AssertionError("Stage-A question K leaked into handoff")
                state.mode = "answer"
                answer_event_start = torch.cuda.Event(enable_timing=True)
                answer_event_end = torch.cuda.Event(enable_timing=True)
                answer_event_start.record(torch.cuda.current_stream(device))
                answer_started = time.perf_counter()
                output = self.runner.model(input_ids=suffix[None],
                                           use_cache=False, return_dict=True)
                if self.check_finite_logits and not torch.isfinite(
                        output.logits).all():
                    raise ArithmeticError("non-finite answer prefill logits")
                first = int(output.logits[0, -1].argmax())
                diagnostic_first_logits = (
                    output.logits[0, -1].detach().float().cpu()
                    if self.all_blocks else None)
                answer_event_end.record(torch.cuda.current_stream(device))
                torch.cuda.synchronize(device)
                first_at = time.perf_counter()
                # Stage A and B share one stream. This first-token sync proves
                # nonblocking H2D has completed, so pinned staging can end.
                state.pinned_buffers.clear()
                answer_prefill_ms = float(
                    answer_event_start.elapsed_time(answer_event_end))
                answer_prefill_host_wall_ms = \
                    (first_at-answer_started)*1e3
                ttft_timing_ms = dict(state.timing_ms)
                tokens = [first]
                while tokens[-1] != tokenizer.eos_token_id \
                        and len(tokens) < self.max_new_tokens:
                    output = self.runner.model(
                        input_ids=torch.tensor([[tokens[-1]]], device=device),
                        use_cache=False, return_dict=True)
                    if self.check_finite_logits and not torch.isfinite(
                            output.logits).all():
                        raise ArithmeticError("non-finite decode logits")
                    tokens.append(int(output.logits[0, -1].argmax()))
                torch.cuda.synchronize(device)
                finished_at = time.perf_counter()
        finally:
            BIAS.clear()
        if state.retrieval_calls != len(self.runner.layers):
            raise AssertionError("wrong Stage-A layer count")
        if state.answer_calls < len(self.runner.layers):
            raise AssertionError("answer prefill missed a layer")
        io_a = state.stage_a_counter.summary()
        io_b = state.stage_b_counter.summary()
        if int(io_b.get("bytes", 0)):
            raise AssertionError("Stage B read visual SSD payload")
        per_kind = io_a.get("per_kind", {})
        normal_bytes = sum(int(per_kind.get(kind, {}).get("bytes", 0))
                           for kind in ("k", "v"))
        structural_bytes = int(per_kind.get("sep", {}).get("bytes", 0))
        logs = sorted(state.log, key=lambda x: x["layer"])
        compact_bytes = sum(
            (h.raw_k.numel() + h.value.numel()) * h.raw_k.element_size()
            for h in state.layers.values())
        read_trace = [
            {"layer": int(li), "file": name, "offset": int(off),
             "length": int(length)}
            for li, name, off, length in ctx.reader.read_trace]
        seen_ranges = {}
        duplicate_read_bytes = 0
        for row in read_trace:
            key = (row["layer"], row["file"])
            start, end = row["offset"], row["offset"] + row["length"]
            for prior_start, prior_end in seen_ranges.get(key, []):
                duplicate_read_bytes += max(
                    0, min(end, prior_end) - max(start, prior_start))
            seen_ranges.setdefault(key, []).append((start, end))
        if duplicate_read_bytes:
            raise AssertionError(
                f"same request reread {duplicate_read_bytes} SSD bytes")
        run_lengths = [length for row in logs
                       for length in row["contiguous_run_lengths"]]
        peak_gpu_alloc = int(torch.cuda.max_memory_allocated(device))
        peak_gpu_reserved = int(torch.cuda.max_memory_reserved(device))
        # Linux ru_maxrss is a process-wide high-water mark in KiB.  It is
        # descriptive context, not memory attributable to this request.
        process_peak_rss_bytes = int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        return {
            "answer": tokenizer.decode(tokens, skip_special_tokens=True).strip(),
            "first_token_id": first,
            "generated_token_ids": tokens,
            "generated_tokens": len(tokens),
            "generated_token_count": len(tokens),
            "question_ids_sha256": question_hash,
            "question_token_ids": question_ids[0].detach().cpu().tolist(),
            "answer_suffix_token_ids": suffix.detach().cpu().tolist(),
            "ttft": first_at-started,
            "ttft_ms": (first_at-started)*1e3,
            "decode_latency": finished_at-first_at,
            "e2e_latency": finished_at-started,
            "request_e2e_ms": (finished_at-started)*1e3,
            "request_prep_ms": request_prep_ms,
            "retrieval_forward_wall_ms": float(
                retrieval_event_start.elapsed_time(retrieval_event_end)),
            "retrieval_forward_host_submit_ms": (
                retrieval_done-retrieval_started)*1e3,
            "answer_prefill_wall_ms": answer_prefill_ms,
            "answer_prefill_host_wall_ms": answer_prefill_host_wall_ms,
            "decode_rope_host_ms": (
                state.timing_ms["rope_ms"] - ttft_timing_ms["rope_ms"]),
            "decode_attention_host_wall_ms": (
                state.timing_ms["attention_host_wall_ms"]
                - ttft_timing_ms["attention_host_wall_ms"]),
            "selected_chunk_ids_per_layer": [
                row["selected_chunk_ids"] for row in logs],
            "source_to_compact_positions_per_layer": [
                list(zip(row["source_positions"],
                         row["compact_positions"])) for row in logs],
            "actual_attention_key_lengths": [
                row["actual_attention_key_len"] for row in logs],
            "compact_prefix_lengths": [
                row["compact_prefix_len"] for row in logs],
            "compact_key_lengths_per_layer": [
                row["actual_attention_key_len"] for row in logs],
            "normal_physical_chunk_count": int(ctx.meta["normal_chunk_count"]),
            "normal_candidate_chunk_count": int(logs[0]["candidate_normal_chunks"]),
            "normal_selected_chunk_count": int(logs[0]["nominal_chunk_budget"]),
            "selected_valid_spatial_tokens_per_layer": [
                row["selected_spatial_tokens"] for row in logs],
            "retrieval_layer_log": logs,
            "io": io_a,
            "stage_a_payload_read_bytes": int(io_a.get("bytes", 0)),
            "stage_b_payload_read_bytes": int(io_b.get("bytes", 0)),
            "ssd_total_bytes": int(io_a.get("bytes", 0)),
            "selected_payload_bytes": normal_bytes,
            "separator_bytes": structural_bytes,
            "request_time_metadata_bytes": 0,
            "selected_normal_chunk_ratio": (
                int(logs[0]["nominal_chunk_budget"])
                / int(ctx.meta["normal_chunk_count"])),
            "selected_candidate_chunk_ratio": (
                int(logs[0]["nominal_chunk_budget"])
                / int(logs[0]["candidate_normal_chunks"])),
            "retained_spatial_token_ratios_per_layer": [
                row["selected_spatial_tokens"] / int(ctx.meta["n_spatial"])
                for row in logs],
            "contiguous_runs_per_layer": [
                row["contiguous_runs"] for row in logs],
            "mean_run_length_chunks": (
                sum(run_lengths) / len(run_lengths) if run_lengths else 0.0),
            "pread_count": int(io_a.get("preads", 0)),
            "pread_trace": read_trace,
            "duplicate_read_bytes": duplicate_read_bytes,
            "fadvise_statuses": list(ctx.reader.last_drop_statuses),
            "fadvise_ms_excluded": float(ctx.reader.last_drop_ms),
            "ssd_read_ms": float(io_a.get("ms", 0.0)),
            "compact_retrieved_kv_bytes": compact_bytes,
            "peak_gpu_allocated_bytes": peak_gpu_alloc,
            "peak_cpu_pinned_bytes": int(state.peak_pinned_bytes),
            "process_peak_rss_bytes": process_peak_rss_bytes,
            "peak_gpu_reserved_bytes": peak_gpu_reserved,
            "incremental_peak_gpu_allocated_bytes": max(
                0, peak_gpu_alloc - gpu_alloc_before),
            "answer_prefill_key_lengths_per_layer": [
                h.prefix_len + int(suffix.numel())
                for _, h in sorted(state.layers.items())],
            "metadata_bytes_image": int(
                ctx.k_rep.numel() * ctx.k_rep.element_size()
                + (0 if getattr(ctx, "valid_counts_gpu", None) is None else
                   ctx.valid_counts_gpu.numel()
                   * ctx.valid_counts_gpu.element_size())),
            "initial_context_gpu_bytes": int(
                getattr(ctx, "initial_context_gpu_bytes", 0)),
            "metadata_activation_ms_excluded": float(
                getattr(ctx, "metadata_activation_ms", 0.0)),
            "initial_context_activation_ms_excluded": float(
                getattr(ctx, "initial_context_activation_ms", 0.0)),
            "n_init": int(ctx.meta["v_token_start"]),
            "n_local": self.n_local,
            "vision_forward_count": 0,
            "similarity_mode": SIMILARITY_MODE,
            "all_block_diagnostic": self.all_blocks,
            "first_logits": diagnostic_first_logits,
            "retrieval_group_size": RETRIEVAL_GROUP_SIZE,
            "core_started_at_s": started,
            "first_token_at_s": first_at,
            "model_finished_at_s": finished_at,
            "postprocess_finished_at_s": time.perf_counter(),
            **ttft_timing_ms,
        }

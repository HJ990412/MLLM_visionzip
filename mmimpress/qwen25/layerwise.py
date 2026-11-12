"""Layer-wise (ragged compact) cache hits for Qwen2.5-VL packed PrefixStore.

Design A of ``docs/qwen_layer_budget_pyr_contract.md``.  Layer ``l`` owns a
compact cache of ``S + k_l`` rows (all structural rows + the first ``k_l``
rows of the image-only importance order, in original-position order).  The
installed text model builds ONE 4D boolean mask per forward from layer 0's
cache length (``[prefix all True | causal suffix]``) and hands the same object
to every decoder layer; ``sdpa_attention_forward`` then only truncates it on
the right.  ``LayerMaskHooks`` therefore re-aligns the mask per layer in a
``forward_pre_hook`` on each ``self_attn``: the last ``q`` columns (HF's own
causal block) are kept untouched and the prefix part is made exactly
``len_l`` all-visible columns.  A ``None`` mask (single-token decode steps) is
left alone.  Hooks live for one request and are always removed.

Normative meaning for a hit, layer ``l``: suffix and generated tokens see
(1) every structural row, (2) the content rows whose original index is in
``full_importance_permutation[:k_l]``, (3) earlier suffix/generated tokens.
K/V are the Turn-1 capture bits (BF16, post-MRoPE keys), never re-rotated;
suffix logical positions are the full-prompt MRoPE positions.

Nothing here modifies ``runner.py`` / ``store.py``; the runner's own helpers
(``_cache_hit_ids``, ``_decode``, ``_result``) are called unchanged.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Sequence

import torch

from .packed_store import (PACKED_NAME, QwenPackedStore, budget_fields,
                           open_qwen_packed_store, write_qwen_packed_store)
from .runner import POSITION_POLICY, _ms, _synchronize
from .store import IOCounter

PACKED_CODE_FILES = ("packed_store.py", "layerwise.py")


def packed_code_revision() -> str:
    here = Path(__file__).resolve().parent
    return hashlib.sha256(b"".join(
        hashlib.sha256((here / name).read_bytes()).digest()
        for name in PACKED_CODE_FILES)).hexdigest()


# ------------------------------------------------------------------- masks
def layer_mask(mask: torch.Tensor, need: int) -> torch.Tensor:
    """Right-align HF's ``[.., q, prefix + q]`` mask to ``need`` key columns.

    The last ``q`` columns are HF's causal block and are never altered; the
    prefix part is all-visible and is cut or extended to ``need - q`` columns.
    """
    have, q = int(mask.shape[-1]), int(mask.shape[-2])
    if need < q:
        raise AssertionError("layer key length shorter than the query length")
    if have == need:
        return mask
    if have > need:
        if have - need > have - q:
            raise AssertionError("mask realignment would cut the causal block")
        return mask[..., have - need:]
    if mask.dtype == torch.bool:
        pad = torch.ones(mask.shape[:-1] + (need - have,), dtype=torch.bool,
                         device=mask.device)
    else:
        pad = torch.zeros(mask.shape[:-1] + (need - have,), dtype=mask.dtype,
                          device=mask.device)
    return torch.cat((pad, mask), dim=-1)


def decoder_layers(model):
    return model.model.language_model.layers


def residual_layer_hooks(model) -> int:
    """Number of forward-pre hooks currently attached to any ``self_attn``."""
    return sum(len(layer.self_attn._forward_pre_hooks) for layer in decoder_layers(model))


def assert_no_layer_hooks(model) -> None:
    count = residual_layer_hooks(model)
    if count:
        raise AssertionError(f"{count} residual self-attention pre-hooks from an earlier request")


class LayerMaskHooks:
    """Per-request, per-layer attention-mask realignment for ragged caches."""

    def __init__(self, model, lengths: Sequence[int]):
        self.layers = decoder_layers(model)
        self.lengths = [int(x) for x in lengths]
        if len(self.lengths) != len(self.layers):
            raise ValueError("one prefix length per decoder layer is required")
        self.handles = []
        self.calls = [0] * len(self.layers)
        self.realigned = [0] * len(self.layers)
        self.appended = [0] * len(self.layers)
        self.install_ms = 0.0

    def _hook(self, module, args, kwargs):
        li = int(module.layer_idx)
        cache = kwargs.get("past_key_values")
        if cache is None:
            raise AssertionError("layer-wise hit forward without a cache")
        current = int(cache.get_seq_length(li))
        if current != self.lengths[li] + self.appended[li]:
            raise AssertionError(
                f"layer {li}: cache length {current} is not the request's "
                f"{self.lengths[li]} + {self.appended[li]} (stale hook or foreign cache)")
        q = int(args[0].shape[1] if args else kwargs["hidden_states"].shape[1])
        self.calls[li] += 1
        self.appended[li] += q
        mask = kwargs.get("attention_mask")
        if mask is None:
            return None
        if mask.ndim != 4 or int(mask.shape[-2]) != q:
            raise AssertionError("unexpected attention mask shape in layer-wise hit")
        new = layer_mask(mask, current + q)
        if new is mask:
            return None
        self.realigned[li] += 1
        kwargs = dict(kwargs)
        kwargs["attention_mask"] = new
        return args, kwargs

    def __enter__(self):
        started = time.perf_counter()
        for layer in self.layers:
            if layer.self_attn._forward_pre_hooks:
                self.remove()
                raise AssertionError("residual self-attention pre-hook before a layer-wise hit")
            self.handles.append(layer.self_attn.register_forward_pre_hook(
                self._hook, with_kwargs=True))
        self.install_ms = (time.perf_counter() - started) * 1e3
        return self

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __exit__(self, *_exc):
        self.remove()


# ------------------------------------------------------------- persistence
def persist_packed(runner, capture, store_dir: str | Path, layer_budget: dict) -> dict:
    """Packed analogue of ``Qwen25Runner.persist(layout='repacked')``."""
    t0 = time.perf_counter()
    timing = {}
    extra = {
        "image_sha256": capture.image_sha256,
        "processor_revision": runner.revision,
        "code_revision": runner._code_revision(),
        "environment_revision": runner._environment_revision(),
        "key_rope_state": "post_mrope",
        "checkpoint_revision": runner.revision,
        "processor_settings": runner.processor_settings(),
        "image_grid_thw": capture.image_grid_thw,
        "geometry": capture.geometry,
        "position_policy": POSITION_POLICY,
        "logical_position_ids": capture.logical_position_ids,
        "rope_deltas": capture.rope_deltas,
        "score_source": "last_fullatt_ViT_received_attention",
        "score_layer": runner.model.config.vision_config.depth - 1,
        "capture_score_ms": capture.score_ms,
        "packed_code_revision": packed_code_revision(),
    }
    meta = write_qwen_packed_store(
        Path(store_dir), capture.layers, capture.visual_start,
        capture.visual_count, capture.prefix_ids, capture.scores, extra,
        layer_budget=layer_budget, timing_out=timing)
    writer_ms = _ms(t0)
    return {"store_dir": str(store_dir), "layout": "packed_repacked",
            "persistence_ms": writer_ms + capture.score_ms + capture.capture_clone_ms,
            "writer_persistence_ms": writer_ms, "timing_ms": timing,
            "capture_score_ms": capture.score_ms,
            "capture_clone_ms": capture.capture_clone_ms,
            "metadata": meta}


# -------------------------------------------------------------- activation
class PackedSession:
    """Registry of activated packed stores for one runner (per-image lifetime)."""

    def __init__(self, runner):
        self.runner = runner
        self.stores: dict[str, QwenPackedStore] = {}
        self.activation_records: dict[str, dict] = {}

    def _validate(self, store: QwenPackedStore, image_sha256: str, layer_budget: dict):
        runner, meta = self.runner, store.meta
        identity = meta["identity"]
        expected = {"checkpoint_revision": runner.revision,
                    "processor_settings": runner.processor_settings(),
                    "position_policy": POSITION_POLICY,
                    "image_sha256": image_sha256}
        for key, value in expected.items():
            if identity.get(key) != value:
                raise ValueError(f"cache identity mismatch for {key}")
        if meta["dtype"] != "bfloat16":
            raise AssertionError("Qwen store must retain native BF16 bits")
        if meta.get("code_revision") != runner._code_revision():
            raise ValueError("packed store core code revision differs from the active runner")
        if meta.get("packed_code_revision") != packed_code_revision():
            raise ValueError("packed store was written by different packed code")
        if meta.get("environment_revision") != runner._environment_revision():
            raise ValueError("cache environment revision differs from active runtime")
        ratios, r_min, r_max = budget_fields(layer_budget, int(meta["num_layers"]))
        if (meta["layer_budget_sha256"] != layer_budget["layer_budget_sha256"]
                or meta["layer_keep_ratios"] != ratios
                or meta["layer_budget_r_min"] != r_min
                or meta["layer_budget_r_max"] != r_max):
            raise ValueError("packed store layer budget differs from the request's")
        if int(meta["num_layers"]) != len(decoder_layers(runner.model)):
            raise ValueError("packed store layer count differs from the decoder")

    def activate(self, store_dir: str | Path, image_sha256: str | None,
                 layer_budget: dict) -> QwenPackedStore:
        if image_sha256 is None:
            raise ValueError("cache activation requires image_sha256")
        key = str(Path(store_dir).resolve())
        if key not in self.stores:
            runner = self.runner
            expected_identity = {
                "image_sha256": image_sha256,
                "checkpoint_revision": runner.revision,
                "processor_settings": runner.processor_settings(),
                "position_policy": POSITION_POLICY,
                "environment_revision": runner._environment_revision(),
            }
            t0 = time.perf_counter()
            store = open_qwen_packed_store(Path(store_dir),
                                           expected_identity=expected_identity,
                                           expected_budget=layer_budget, verify=True)
            try:
                self._validate(store, image_sha256, layer_budget)
            except Exception:
                store.close()
                raise
            self.stores[key] = store
            self.activation_records[key] = {
                "activation_ms": _ms(t0),
                "activation_io": store.activation_io.summary(),
                "metadata_resident_bytes": store.metadata_resident_bytes,
                "pinned": store.pinned, "pinned_host_bytes": store.total,
                "metadata_file_sha256": store.metadata_file_sha256,
            }
        else:
            store = self.stores[key]
            self._validate(store, image_sha256, layer_budget)
        return store

    def condition(self, store_dir: str | Path, image_sha256: str | None,
                  layer_budget: dict) -> dict:
        """Best-effort OS page-cache DONTNEED outside the request timer."""
        key = str(Path(store_dir).resolve())
        fresh = key not in self.stores
        store = self.activate(store_dir, image_sha256, layer_budget)
        conditioned = store.drop_payload_cache()
        activation = self.activation_records[key]
        return {"success": conditioned["attempted"] - len(conditioned["failed"]),
                "failures": len(conditioned["failed"]),
                "failure_details": conditioned["failed"],
                "layer_keep_counts": list(store.meta["layer_keep_counts"]),
                "budget_unit": "visual_kv_layerwise",
                "guarantee": "OS page-cache hint only",
                "activation_ms": activation["activation_ms"] if fresh else 0.0,
                "activation_io": activation["activation_io"] if fresh else None,
                "metadata_resident_bytes": activation["metadata_resident_bytes"]}

    def close(self):
        for store in self.stores.values():
            store.close()
        self.stores.clear()


# --------------------------------------------------------------------- hit
def assemble_cache(runner, store: QwenPackedStore, plan):
    """One H2D of the host buffer, one GPU gather, per-layer views -> DynamicCache."""
    from transformers import DynamicCache
    device = runner.device
    flat = store.rows().to(device, non_blocking=True)
    index = plan.row_index.to(device, non_blocking=True)
    gathered = flat.index_select(0, index)
    cache = DynamicCache(config=runner.model.config.text_config)
    for li, (a, b, c) in enumerate(plan.spans):
        cache.update(gathered[a:b].permute(1, 0, 2).unsqueeze(0),
                     gathered[b:c].permute(1, 0, 2).unsqueeze(0), li)
    return cache


@torch.inference_mode()
def forward_layerwise(runner, cache, lengths: Sequence[int], suffix: torch.Tensor,
                      positions: torch.Tensor, prefix_len: int, *,
                      install_hooks: bool = True, return_logits: bool = False) -> dict:
    """Suffix prefill + greedy decode on a ragged cache.  ``install_hooks=False``
    exists only for negative fixtures."""
    device = runner.device
    len0 = int(cache.get_seq_length())
    if len0 != int(lengths[0]):
        raise AssertionError("layer 0 cache length differs from the plan")
    for li, layer in enumerate(cache.layers):
        if int(layer.keys.shape[0]) != 1 or int(layer.keys.shape[2]) != int(lengths[li]) \
                or layer.values.shape != layer.keys.shape:
            raise AssertionError(f"layer {li} cache shape differs from the plan")
    t = int(suffix.shape[1])
    mask = torch.ones((1, len0 + t), dtype=torch.long, device=device)
    suffix_device = suffix.to(device)
    suffix_pos = positions[:, :, prefix_len:].to(device)
    hooks = LayerMaskHooks(runner.model, lengths) if install_hooks else None
    if hooks is None:
        assert_no_layer_hooks(runner.model)
    try:
        if hooks is not None:
            hooks.__enter__()
        prefill_t0 = time.perf_counter()
        output = runner.model(
            input_ids=suffix_device, attention_mask=mask, position_ids=suffix_pos,
            cache_position=torch.arange(len0, len0 + t, device=device),
            past_key_values=cache, use_cache=True, return_dict=True,
            logits_to_keep=1)
        _synchronize(device)
        prefill_ms = _ms(prefill_t0)
        token = int(output.logits[0, -1].argmax())
        _synchronize(device)
        first_token_at = time.perf_counter()
        logits = output.logits[0, -1].float().cpu().clone() if return_logits else None
        generated, decode_ms = runner._decode(
            token, cache, position_start=int(positions.max()) + 1,
            attention_mask=mask)
        finished_at = time.perf_counter()
    finally:
        if hooks is not None:
            hooks.remove()
    assert_no_layer_hooks(runner.model)
    if hooks is not None:
        steps = len(generated)          # 1 prefill + (len - 1) decode forwards
        if hooks.calls != [steps] * len(hooks.calls):
            raise AssertionError("mask hook did not see every layer of every forward")
        if hooks.appended != [t + steps - 1] * len(hooks.appended):
            raise AssertionError("mask hook token accounting mismatch")
    return {"generated": generated, "first_logits": logits, "prefill_ms": prefill_ms,
            "decode_ms": decode_ms, "first_token_at": first_token_at,
            "finished_at": finished_at, "compact_len_layer0": len0,
            "hook_install_ms": hooks.install_ms if hooks is not None else 0.0,
            "hook_realigned_layers": sum(1 for x in hooks.realigned if x) if hooks is not None else 0,
            "hook_calls_per_layer": hooks.calls[0] if hooks is not None else 0}


@torch.inference_mode()
def run_cache_layerwise(runner, session: PackedSession, store_dir: str | Path,
                        question: str, history: Sequence[tuple[str, str]] = (), *,
                        layer_budget: dict, image_sha256: str | None = None,
                        return_logits: bool = False) -> dict:
    """Packed layer-wise cache hit.  Timing boundaries mirror ``run_cache``:
    TTFT starts before prompt preparation and includes planning, the single
    ``preadv``, H2D, assembly, hook installation and the suffix prefill."""
    store = session.activate(store_dir, image_sha256, layer_budget)
    meta = store.meta
    runner._clear_request_state()
    torch.cuda.reset_peak_memory_stats(runner.device)
    request_t0 = time.perf_counter()
    prompt_t0 = time.perf_counter()
    full_ids, suffix, positions, deltas = runner._cache_hit_ids(meta, question, history)
    prompt_ms = _ms(prompt_t0)
    if suffix.numel() == 0:
        raise AssertionError("cache hit requires a nonempty question suffix")
    read_t0 = time.perf_counter()
    plan = store.plan()
    io = IOCounter()
    returned = store.read_all(io)
    read_ms = _ms(read_t0)
    assembly_t0 = time.perf_counter()
    cache = assemble_cache(runner, store, plan)
    _synchronize(runner.device)
    assembly_ms = _ms(assembly_t0)
    structural = int(meta["structural_count"])
    if plan.lengths != [structural + k for k in plan.counts] \
            or len(cache.layers) != int(meta["num_layers"]):
        raise AssertionError("cache contains wrong number of selected rows")
    out = forward_layerwise(runner, cache, plan.lengths, suffix, positions,
                            int(meta["prefix_len"]), return_logits=return_logits)
    ttft_ms = (out["first_token_at"] - request_t0) * 1e3
    request_e2e_ms = (out["finished_at"] - request_t0) * 1e3
    runner._clear_request_state()
    summary = io.summary()
    row_bytes = int(meta["row_bytes"])
    num_layers = int(meta["num_layers"])
    gpu_kv_bytes = sum(2 * length * row_bytes for length in plan.lengths)
    result = runner._result(
        out["generated"], ttft_ms=ttft_ms, request_e2e_ms=request_e2e_ms,
        timing_ms={"prompt_preparation": prompt_ms,
                   "store_load_inclusive": read_ms,
                   "raw_pread": summary["ms"],
                   "layerwise_planning": plan.planning_ms,
                   "h2d_and_assembly": assembly_ms,
                   "mask_hook_install": out["hook_install_ms"],
                   "suffix_prefill": out["prefill_ms"], "decode": out["decode_ms"]},
        vision_calls=0,
        peak_allocated=torch.cuda.max_memory_allocated(runner.device),
        peak_reserved=torch.cuda.max_memory_reserved(runner.device),
        first_logits=out["first_logits"])
    k_max = max(plan.counts)
    result.update({
        "store_format": meta["store_format"],
        "visual_read_bytes": int(meta["bytes_visual_kv"]),
        "structural_read_bytes": int(meta["bytes_structural_kv"]),
        "metadata_read_bytes": 0,
        "packed_read_bytes": returned,
        "pread_calls": summary["preads"], "read_spans": summary["spans"],
        "read_syscall": "preadv", "read_io": summary,
        "raw_pread_ms": summary["ms"],
        "planning_ms": plan.planning_ms, "selector_ms": 0.0,
        "budget_unit": "visual_kv_layerwise",
        "layer_budget_id": meta["layer_budget_id"],
        "layer_budget_sha256": meta["layer_budget_sha256"],
        "layer_keep_counts": list(plan.counts),
        "layer_prefix_lengths": list(plan.lengths),
        "k_uniform_equivalent": (int(meta["visual_count"]) + 3) // 4,
        "k_total": int(sum(plan.counts)), "k_max": k_max,
        "selected_importance_prefix_original": [int(x) for x in
                                                meta["full_importance_permutation"][:k_max]],
        "selection_rule": "layer l attends original ids selected_importance_prefix_original[:layer_keep_counts[l]]",
        "padding_rows_read": 0, "extra_valid_visual_rows": 0,
        "loaded_valid_visual_rows": int(sum(plan.counts)),
        "structural_count": structural,
        "h2d_kv_bytes": int(meta["packed_total_bytes"]),
        "gpu_cache_kv_bytes": gpu_kv_bytes,
        "total_visual_tokens": int(meta["visual_count"]),
        "mean_kept_ratio": sum(plan.counts) / (num_layers * int(meta["visual_count"])),
        "compact_prefix_tokens": out["compact_len_layer0"],
        "compact_prefix_tokens_per_layer": list(plan.lengths),
        "ragged_design": "A_forward_pre_hook_right_aligned_mask",
        "mask_hook_realigned_layers": out["hook_realigned_layers"],
        "mask_hook_calls_per_layer": out["hook_calls_per_layer"],
        "residual_layer_hooks_after": residual_layer_hooks(runner.model),
        "dense_reference": False,
        "rope_deltas": deltas.tolist(),
        "logical_suffix_first_position": positions[:, 0, meta["prefix_len"]].tolist(),
        "compact_suffix_first_slot": out["compact_len_layer0"],
        "packed_file": PACKED_NAME,
    })
    return result

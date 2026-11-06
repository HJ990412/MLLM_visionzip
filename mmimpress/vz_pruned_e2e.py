"""VZ-Pruned-E2E helpers (contract: docs/vz_pruned_e2e_mtgqa_contract.md).

The first turn runs on ``[system | sorted(S + separators) | question]`` with
contiguous positions through the real serving path: one vision-tower call
(``get_image_features``), the image-only VisionZip order, direct assembly of
the pruned ``inputs_embeds`` and one language-model prefill.  No full-token
``inputs_embeds`` is built and no forward is aborted.  The prefix rows of that
cache are persisted by the unchanged prefix25 writer from a full-size
synthetic tensor pair (rev. 3), so the store has the Compact format and is
served by the rev.-3 Compact hit path.

Gates: the rev.-3 structural bit-identity against a same-length truncation
control (length ``L+q``) and the amended layer-0 rule (realigned K, FP32
oracle ratio, un-realigned negative control; elementwise V).
"""
from __future__ import annotations

import hashlib
import time

import torch

from mmimpress.recompute_kv25 import (
    LAYER0_TOL, SENTINEL, StoreRows, compare_stores, diff_stats, reduced_plan,
    rotate, structural_rows, synthetic_layers, truncation_control, within)

# Frozen in docs/vz_pruned_e2e_mtgqa_contract.md (amendment 1) before GPU execution.
VZ_LAYER0_K_MAX_REL_L2 = 1e-2          # realigned Ours K vs VZ K, layer 0 (S and sep rows)
VZ_LAYER0_K_MAX_ORACLE_RATIO = 1.5     # rel(VZ, oracle_p') <= 1.5 x rel(Ours_realigned, oracle_p')
VZ_NEG_UNREALIGNED_MIN_REL_L2 = 0.1    # un-realigned Ours K vs VZ K on moved rows
PATH_EQUIVALENCE_LOGIT_ATOL = 1e-4     # G4-b: real path vs AbortAfterEmbeds composition


# ------------------------------------------------------------- sequences
def pruned_sequence(plan, n_text):
    """Full-sequence indices of the pruned T1 input and the control plan.

    ``plan`` is ``reduced_plan(..., 'compact')``; the text rows are the
    ``n_text`` tokens after the image span.  Returns ``(input_index,
    control_plan)`` where ``control_plan`` has the same length ``L+q`` with
    original indices/positions, for :func:`truncation_control`.
    """
    assert plan["policy"] == "compact"
    prefix = int(plan["v_start"]) + int(plan["v_num"])
    text = list(range(prefix, prefix + int(n_text)))
    index = list(plan["input_index"]) + text
    control = {"policy": "compact", "v_start": int(plan["v_start"]),
               "input_index": index, "position_ids": list(range(len(index)))}
    rows = structural_rows(plan)
    # identical structural rows; the control may only extend them when no
    # image token was dropped at all (never the case at k = ceil(N/4) < N)
    assert structural_rows(control)[:len(rows)] == rows
    return index, control


@torch.no_grad()
def assemble_pruned_embeds(sys_embeds, image_features, text_embeds, plan):
    """``[sys | image rows at keep | text]`` as (1, L+q, D) in the text dtype."""
    keep = torch.as_tensor(plan["keep_visual_ids"], dtype=torch.long,
                           device=image_features.device)
    assert sys_embeds.shape[1] == plan["v_start"]
    assert image_features.shape[0] == plan["v_num"]
    rows = image_features.index_select(0, keep).to(text_embeds.dtype)
    return torch.cat([sys_embeds, rows.unsqueeze(0), text_embeds], dim=1)


def prefix_embeds_sha256(sys_embeds, image_features):
    """SHA256 of the BF16 ``[sys | all image rows]`` tensor (G2-b)."""
    t = torch.cat([sys_embeds[0], image_features.to(sys_embeds.dtype)], dim=0)
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.int16)
                          .numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------- T1 path
class _VisionCalls:
    def __init__(self, runner):
        self.tower = runner.model.model.vision_tower
        self.calls = 0

    def __enter__(self):
        self._h = self.tower.register_forward_pre_hook(lambda *a, **k: self._count())
        return self

    def _count(self):
        self.calls += 1

    def __exit__(self, *exc):
        self._h.remove()


@torch.no_grad()
def pruned_t1_request(runner, server, image_path, factory, processor_fn,
                      counts_cls=None):
    """Real-path pruned first turn; returns ``(result, diag)`` like 108's T1.

    ``factory()`` returns ``(prompt, history, entries)``; ``processor_fn`` is
    108's ``_combined_processor``; ``counts_cls`` (108's ``PilotCounts``)
    counts language-model projections.  The TTFT boundary is the host
    ``int(argmax)`` of the first token followed by ``cuda.synchronize``.
    """
    from contextlib import ExitStack, nullcontext
    from PIL import Image
    from transformers import DynamicCache
    from mmimpress.cvpr25 import anyres_token_scores, visionzip_repack_order
    from mmimpress.piggyback import VisionForwardCapture
    from mmimpress.rerope_kv25 import LayerPositions

    model, dev = runner.model, runner.model.device
    tok = runner.processor.tokenizer
    embed = model.get_input_embeddings()
    vision = VisionForwardCapture(runner, capture_saliency=True)
    stamps = {}
    with ExitStack() as stack:
        calls = stack.enter_context(_VisionCalls(runner))
        lp = stack.enter_context(LayerPositions(runner))
        counts = stack.enter_context(counts_cls(runner) if counts_cls else nullcontext())
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        prompt, history, entries = factory()
        with Image.open(image_path) as im:
            image = im.convert("RGB"); image.load()
        stamps["image_decoded"] = time.perf_counter()
        enc, processing = processor_fn(runner, image, prompt)
        v0, vn = runner.visual_span(enc["input_ids"])
        enc_dev = runner.to_device(enc)
        torch.cuda.synchronize()
        stamps["encoded"] = time.perf_counter()
        with vision:
            feats = model.model.get_image_features(pixel_values=enc_dev["pixel_values"],
                                                  image_sizes=enc_dev["image_sizes"])
        image_features = torch.cat(list(feats), dim=0)
        assert image_features.shape[0] == vn, (image_features.shape, vn)
        torch.cuda.synchronize()
        stamps["vision"] = time.perf_counter()
        base, _, _, sep = runner.anyres_layout(enc["image_sizes"][0], vn)
        scores = anyres_token_scores(runner, vision.result_cpu(), enc["image_sizes"][0], vn,
                                     base_side=base)
        order = visionzip_repack_order(scores, sep)
        plan = reduced_plan(v0, vn, order, sep, "compact")
        stamps["plan"] = time.perf_counter()
        ids = enc_dev["input_ids"]
        n_text = int(ids.shape[1]) - v0 - vn
        sys_embeds = embed(ids[:, :v0])
        text_embeds = embed(ids[:, v0 + vn:])
        x = assemble_pruned_embeds(sys_embeds, image_features, text_embeds, plan)
        n = int(x.shape[1])
        pos = torch.arange(n, device=dev)
        cache = DynamicCache()
        torch.cuda.synchronize()
        stamps["assembled"] = time.perf_counter()
        prefill_t0 = time.perf_counter()
        out = model(inputs_embeds=x, attention_mask=torch.ones(1, n, dtype=torch.long, device=dev),
                    position_ids=pos.unsqueeze(0), cache_position=pos, past_key_values=cache,
                    use_cache=True)
        first = int(out.logits[0, -1].argmax())
        torch.cuda.synchronize()
        first_token_at = time.perf_counter()
        first_logits = out.logits[0, -1]
        toks, cur = [first], n
        while toks[-1] != tok.eos_token_id and len(toks) < server.max_new_tokens:
            cp = torch.tensor([cur], device=dev)
            out = model(input_ids=torch.tensor([[toks[-1]]], device=dev),
                        attention_mask=torch.ones(1, cur + 1, dtype=torch.long, device=dev),
                        position_ids=cp.unsqueeze(0), cache_position=cp, past_key_values=cache,
                        use_cache=True)
            toks.append(int(out.logits[0, -1].argmax()))
            cur += 1
        torch.cuda.synchronize()
        finished_at = time.perf_counter()
        answer = tok.decode(toks, skip_special_tokens=True).strip()
        returned = time.perf_counter()
    assert vision.call_count == 1 and calls.calls == 1
    assert n == plan["text_position_start"] + n_text
    position_check = lp.verify(n)
    t = {k: (v - started) * 1e3 for k, v in stamps.items()}
    result = {
        "answer": answer, "first_token_id": first, "ttft": first_token_at - started,
        "decode_latency": finished_at - first_token_at, "e2e_latency": finished_at - started,
        "prefill_ms": (first_token_at - prefill_t0) * 1e3,
        "decode_ms": (finished_at - first_token_at) * 1e3,
        "generated_tokens": len(toks), "generated_token_ids": list(toks),
        "generated_token_count": len(toks), "core_started_at_s": started,
        "first_token_at_s": first_token_at, "model_finished_at_s": finished_at,
        "postprocess_finished_at_s": returned,
        "projection_calls": counts.result() if counts_cls else None, "scoring_calls": 0,
        "end_to_end_ttft_ms": (first_token_at - started) * 1e3,
        "request_e2e_ms": (returned - started) * 1e3, "request_started_at_s": started,
        "vision_forward_count": calls.calls,
        "cap_reached": len(toks) == server.max_new_tokens and toks[-1] != tok.eos_token_id,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(), "processor": processing,
        "ttft_breakdown_ms": {
            "image_decode": t["image_decoded"], "processor_and_h2d": t["encoded"] - t["image_decoded"],
            "vision_projector_pack_saliency": t["vision"] - t["encoded"],
            "saliency_order_plan": t["plan"] - t["vision"], "embed_assembly": t["assembled"] - t["plan"],
            "prefill_to_first_token": (first_token_at - prefill_t0) * 1e3},
        "ttft_path": "real pruned serving path; single language-model prefill",
        "pruned_input_len": n, "text_tokens": n_text, "text_position_start": plan["text_position_start"],
        "k_target": plan["k"], "N_content": plan["n"], "n_sep": plan["n_sep"],
        "selected_original_ids": plan["selected_original_ids"],
        "selection_sha256": hashlib.sha256(",".join(map(str, order)).encode()).hexdigest(),
        "position_check": position_check, "online_query_score_calls": 0,
        "language_model_prefill_calls": 1, "cache": "pruned_t1_dynamic_cache",
        "captured_past_key_values": cache, "first_logits": first_logits,
    }
    diag = {"prompt": prompt, "history": history, "history_entries": entries, "enc_cpu": enc,
            "vision": vision, "plan": plan, "order": order, "sep": sep, "v_token_start": v0,
            "v_token_num": vn, "n_text": n_text, "sys_embeds": sys_embeds,
            "image_features": image_features, "text_embeds": text_embeds,
            "prefix_embeds_sha256": prefix_embeds_sha256(sys_embeds, image_features)}
    return result, diag


# ------------------------------------------------------------- persistence
@torch.no_grad()
def vz_prefix_layers(cache, plan):
    """Full-size synthetic layers from the pruned T1 cache rows ``[0, L)``.

    Rows outside ``plan['input_index']`` hold the sentinel; the prefix25
    writer persists only the ``S`` content rows, system rows and separators.
    """
    from mmimpress.model import cache_layers
    L = len(plan["input_index"])
    prefix_len = int(plan["v_start"]) + int(plan["v_num"])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    reduced = [(k[:, :, :L].contiguous(), v[:, :, :L].contiguous()) for k, v in cache_layers(cache)]
    layers = synthetic_layers(reduced, plan, prefix_len)
    del reduced
    torch.cuda.synchronize()
    return layers, {"synthetic_assembly_ms": (time.perf_counter() - t0) * 1e3,
                    "prefix_rows": L, "prefix_len": prefix_len, "sentinel": SENTINEL}


@torch.no_grad()
def vz_control(runner, diag):
    """Same-length (``L+q``) truncation control of the pruned T1 (outside timers)."""
    plan = diag["plan"]
    _, control_plan = pruned_sequence(plan, diag["n_text"])
    full = torch.cat([diag["sys_embeds"], diag["image_features"].to(diag["text_embeds"].dtype).unsqueeze(0),
                      diag["text_embeds"]], dim=1)
    # the control is the ORIGINAL sequence truncated to L+q rows (rev. 3 rule)
    assert full.shape[1] == plan["v_start"] + plan["v_num"] + diag["n_text"] >= len(control_plan["input_index"])
    ctrl = truncation_control(runner, full, control_plan)
    rows = structural_rows(plan)
    if ctrl["rows"] != rows:                      # nothing dropped: keep the prefix rows only
        keep = [i for i, r in enumerate(ctrl["rows"]) if r in set(rows)]
        ctrl["layers"] = [tuple(t[keep] for t in pair) for pair in ctrl["layers"]]
        ctrl["rows"] = rows
    return ctrl


# ------------------------------------------------------------------ gates
def compare_vz(ours_dir, vz_dir, inv_freq, oracle, control):
    """G2 for the VZ store: rev.-3 structural gates plus the amended layer-0 K rule.

    ``compare_stores`` supplies the structural bit-identity gates (G2-c),
    layer-0 V (elementwise) and the recorded per-layer drift; its own
    elementwise realigned-K gates are kept as report-only.  The contract
    gates added here: realigned rel L2 <= 1e-2, FP32-oracle ratio <= 1.5
    and the un-realigned negative control > 0.1 on moved rows.
    """
    base = compare_stores(ours_dir, vz_dir, "compact", inv_freq, oracle, control)
    gates = dict(base["gates"])
    report = dict(base["report_only"])
    for name in ("layer0_k_realigned", "neg_layer0_k_unaligned_moved_rows"):
        if name in gates:
            report[f"rev3_{name}"] = gates.pop(name)
    ours, vz = StoreRows(ours_dir), StoreRows(vz_dir)
    m = ours.meta
    v_start = int(m["v_token_start"])
    order = [int(x) for x in m["full_importance_permutation"]]
    sep = [int(x) for x in m["newline_idx"]]
    plan = reduced_plan(v_start, int(m["v_token_num"]), order, sep, "compact")
    sel = order[:plan["k"]]
    rows = {"S": sel, "sep": sep}
    fail = []
    added = {}
    for name, ids in rows.items():
        orig = torch.tensor([v_start + i for i in ids], dtype=torch.float64)
        new = torch.tensor([plan["visual_position"][i] for i in ids], dtype=torch.float64)
        a = vz.content(0, "k") if name == "S" else vz.separators(0, "k")
        b = ours.content(0, "k") if name == "S" else ours.separators(0, "k")
        realigned = rotate(b, new - orig, inv_freq)
        st = diff_stats(a, realigned)
        ref = rotate(oracle["k"][orig.long()], new - orig, inv_freq)
        r_vz, r_ours = diff_stats(a, ref)["rel_l2"], diff_stats(realigned, ref)["rel_l2"]
        ratio = r_vz / r_ours if r_ours else (0.0 if r_vz == 0 else float("inf"))
        added[f"vz_layer0_k_realigned_{name}"] = (st["rel_l2"] <= VZ_LAYER0_K_MAX_REL_L2, st)
        added[f"vz_layer0_k_oracle_ratio_{name}"] = (
            r_vz <= VZ_LAYER0_K_MAX_ORACLE_RATIO * r_ours + 1e-12,
            {"oracle_rel_l2_vz": r_vz, "oracle_rel_l2_ours_realigned": r_ours, "oracle_ratio": ratio})
        moved = [j for j in range(len(ids)) if float(new[j]) != float(orig[j])]
        neg = diff_stats(a[moved], b[moved]) if moved else {"rel_l2": 0.0, "rows": 0}
        added[f"vz_neg_layer0_k_unrealigned_{name}"] = (
            bool(moved) and neg["rel_l2"] > VZ_NEG_UNREALIGNED_MIN_REL_L2, {"moved_rows": len(moved), **neg})
    for name, (ok, st) in added.items():
        gates[name] = [{"ok": bool(ok), **st}]
    for name, entries in gates.items():
        if not all(e["ok"] for e in entries):
            fail.append(name)
    return {**base, "gates": gates, "report_only": report, "status": "FAIL" if fail else "PASS",
            "failed_gates": sorted(set(fail)), "contract": "vz_pruned_e2e amendment 1"}


# ------------------------------------------------- hit vs standalone (G4-d, G8)
@torch.no_grad()
def standalone_pruned(runner, server, diag, suffix_ids):
    """Cache-less single prefill over ``[sys | keep | suffix]`` at ``arange``.

    ``suffix_ids`` are the hit prompt's tokens after the image span (history +
    current question).  Returns tokens, answer and the first-token logits
    (CPU, FP32) for a recorded comparison with the stored-prefix hit.
    """
    from transformers import DynamicCache
    model, dev = runner.model, runner.model.device
    tok = runner.processor.tokenizer
    embed = model.get_input_embeddings()
    text = embed(suffix_ids.to(dev).unsqueeze(0))
    x = assemble_pruned_embeds(diag["sys_embeds"], diag["image_features"], text, diag["plan"])
    n = int(x.shape[1])
    pos = torch.arange(n, device=dev)
    cache = DynamicCache()
    out = model(inputs_embeds=x, attention_mask=torch.ones(1, n, dtype=torch.long, device=dev),
                position_ids=pos.unsqueeze(0), cache_position=pos, past_key_values=cache, use_cache=True)
    first_logits = out.logits[0, -1].detach().float().cpu()
    toks, cur = [int(first_logits.argmax())], n
    while toks[-1] != tok.eos_token_id and len(toks) < server.max_new_tokens:
        cp = torch.tensor([cur], device=dev)
        out = model(input_ids=torch.tensor([[toks[-1]]], device=dev),
                    attention_mask=torch.ones(1, cur + 1, dtype=torch.long, device=dev),
                    position_ids=cp.unsqueeze(0), cache_position=cp, past_key_values=cache, use_cache=True)
        toks.append(int(out.logits[0, -1].argmax()))
        cur += 1
    del cache
    return {"answer": tok.decode(toks, skip_special_tokens=True).strip(), "first_token_id": toks[0],
            "generated_token_ids": toks, "first_logits": first_logits, "prefill_len": n,
            "text_position_start": diag["plan"]["text_position_start"]}


def hit_vs_standalone(hit, standalone):
    """Recorded (never gated) comparison of a stored-prefix hit with a single prefill."""
    from mmimpress.rerope_kv25 import top2_margin
    a, b = hit.get("first_logits"), standalone["first_logits"]
    delta = float((a.float().cpu() - b).abs().max()) if a is not None else None
    return {"first_token_exact": hit["first_token_id"] == standalone["first_token_id"],
            "generated_token_ids_exact": list(hit["generated_token_ids"]) == list(standalone["generated_token_ids"]),
            "answer_exact": hit["answer"] == standalone["answer"], "max_abs_first_logit_delta": delta,
            "hit_top2": top2_margin(a), "standalone_top2": top2_margin(b),
            "hit_tokens": list(hit["generated_token_ids"]), "standalone_tokens": list(standalone["generated_token_ids"]),
            "gated": False}


def path_equivalence(real, composed, atol=PATH_EQUIVALENCE_LOGIT_ATOL):
    """G4-b: real pruned path vs the Stage-1 AbortAfterEmbeds composition (gated)."""
    a = real["first_logits"].detach().float().cpu()
    b = composed["first_logits"].detach().float().cpu()
    delta = float((a - b).abs().max())
    ok = (real["generated_token_ids"] == composed["generated_token_ids"]
          and real["first_token_id"] == composed["first_token_id"] and delta <= atol)
    return {"ok": bool(ok), "max_abs_first_logit_delta": delta, "atol": atol,
            "tokens_equal": real["generated_token_ids"] == composed["generated_token_ids"],
            "prefill_len": (real["pruned_input_len"], composed["prefill_len"])}

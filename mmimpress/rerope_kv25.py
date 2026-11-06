"""Ours-ReRoPE and Pruned-T1 helpers (contract: docs/rerope_kv25_contract.md).

Ours-ReRoPE keeps the full-context Turn-1 KV of the selected rows and only
re-rotates their post-RoPE Keys from the original position ``p`` to the
contiguous (Compact) position ``p'`` with one RoPE rotation ``R(p' - p)``,
computed in FP64 and cast once to the FP16 store dtype.  Values and system
rows are untouched.  Pruned-T1 runs a single prefill over
``[system | sorted(S + separators) | question]`` with contiguous positions
and no cache at all.

Nothing here edits the existing serving or persistence modules; the stores
are written by the unchanged prefix25 writer from a full-size tensor pair.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from mmimpress.recompute_kv25 import (
    StoreRows, bit_equal, diff_stats, reduced_plan, rotate, within)

# Frozen in docs/rerope_kv25_contract.md before GPU execution.
ROUNDTRIP_TOL = {"rel_l2": 1e-3, "rtol": 1e-3, "atol_scale": 1e-3}   # G-a
SAME_POSITION_MAX_REL_L2 = 1e-2          # G-b layer-0 ReRoPE K vs Compact K
SAME_POSITION_MAX_ORACLE_RATIO = 1.5     # rel(ReRoPE, oracle_p') <= 1.5 x rel(Compact, oracle_p')
NEG_UNROTATED_MIN_REL_L2 = 0.1           # un-rotated Ours K vs Compact K, layer 0


# ------------------------------------------------------------ re-rotation
@torch.no_grad()
def rerope_layers(layers, plan, inv_freq):
    """Full-size (K fp16, V bf16) pairs with the kept rows' K re-rotated.

    ``layers`` are the Turn-1 ``[(k, v)]`` of shape (1, H, P, hd) in BF16;
    ``plan`` is the compact :func:`reduced_plan` of the image.  Every K row is
    cast exactly to FP16; rows of ``keep`` are rotated by ``p' - p`` in FP64
    and cast once.  V is returned by reference.  Timing is CUDA-synchronized.
    """
    assert plan["policy"] == "compact"
    v_start = plan["v_start"]
    rows = torch.tensor([v_start + i for i in plan["keep_visual_ids"]], dtype=torch.long)
    delta = torch.tensor([plan["visual_position"][i] - (v_start + i)
                          for i in plan["keep_visual_ids"]], dtype=torch.float64)
    assert (delta <= 0).all() and plan["position_ids"] == list(range(len(plan["input_index"])))
    inv = torch.as_tensor(inv_freq, dtype=torch.float64)
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    sync()
    t0 = time.perf_counter()
    out = []
    for k, v in layers:
        assert k.dtype == torch.bfloat16 and k.shape[0] == 1
        dev = k.device
        k16 = k.to(torch.float16)                                  # exact
        kept = k[0, :, rows.to(dev)].transpose(0, 1)               # rows, H, hd (BF16)
        rot = rotate(kept, delta.to(dev), inv.to(dev))             # FP64
        k16[0, :, rows.to(dev)] = rot.to(torch.float16).transpose(0, 1)
        out.append((k16, v))
    sync()
    return out, {"rerope_ms": (time.perf_counter() - t0) * 1e3,
                 "rotated_rows": int(rows.numel()), "precision": "float64",
                 "cast": "once to float16", "delta_min": int(delta.min()),
                 "delta_max": int(delta.max())}


# ------------------------------------------------------------ gates
def compare_rerope(ours_dir, rerope_dir, compact_dir, inv_freq, oracle):
    """G-a / G-b / negative control plus per-layer records.

    ``oracle`` is the rev.-3 FP32/FP64 layer-0 oracle at original positions
    (``{'k': (P, H, hd), 'v': ...}``); it is rotated to ``p'`` here.
    """
    ours, rr, cp = StoreRows(ours_dir), StoreRows(rerope_dir), StoreRows(compact_dir)
    m = ours.meta
    v_start = int(m["v_token_start"])
    order = [int(x) for x in m["full_importance_permutation"]]
    sep = [int(x) for x in m["newline_idx"]]
    plan = reduced_plan(v_start, int(m["v_token_num"]), order, sep, "compact")
    k = plan["k"]
    sel = order[:k]
    inv = torch.as_tensor(inv_freq, dtype=torch.float64)
    d_s = torch.tensor([plan["visual_position"][i] - (v_start + i) for i in sel], dtype=torch.float64)
    d_p = torch.tensor([plan["visual_position"][i] - (v_start + i) for i in sep], dtype=torch.float64)
    gates, layers, fail = {}, [], []

    def gate(name, ok, stats):
        gates.setdefault(name, []).append({"ok": bool(ok), **stats})
        if not ok:
            fail.append(name)

    for li in range(ours.L):
        row = {"layer": li}
        for name, delta, rows_abs in (("S", d_s, [v_start + i for i in sel]),
                                      ("sep", d_p, [v_start + i for i in sep])):
            get = (lambda st: st.content(li, "k")) if name == "S" else (lambda st: st.separators(li, "k"))
            getv = (lambda st: st.content(li, "v")) if name == "S" else (lambda st: st.separators(li, "v"))
            ko, kr, kc = get(ours), get(rr), get(cp)
            # G-a round trip (every layer)
            back = rotate(kr, -delta, inv)
            o = ko.double()
            st = diff_stats(back, o)
            ok = st["rel_l2"] <= ROUNDTRIP_TOL["rel_l2"] and bool(torch.allclose(
                back, o, rtol=ROUNDTRIP_TOL["rtol"],
                atol=ROUNDTRIP_TOL["atol_scale"] * float(o.abs().max())))
            gate(f"roundtrip_k_{name}", ok, {"layer": li, **st})
            # V bit-identity with Ours (every layer)
            gate(f"v_bitident_{name}", bit_equal(getv(rr), getv(ours)), {"layer": li})
            # records: context effect at identical positions / realignment only
            row[f"k_{name}_vs_compact"] = diff_stats(kr, kc)
            row[f"v_{name}_vs_compact"] = diff_stats(getv(rr), getv(cp))
            row[f"k_{name}_vs_ours_unrotated"] = diff_stats(kr, ko)
            if li == 0:
                st0 = diff_stats(kr, kc)
                ref = rotate(oracle["k"][torch.as_tensor(rows_abs, dtype=torch.long)], delta, inv)
                r_rr, r_cp = diff_stats(kr, ref)["rel_l2"], diff_stats(kc, ref)["rel_l2"]
                ratio = r_rr / r_cp if r_cp else (0.0 if r_rr == 0 else float("inf"))
                gate(f"same_position_layer0_k_{name}",
                     st0["rel_l2"] <= SAME_POSITION_MAX_REL_L2 and
                     ratio <= SAME_POSITION_MAX_ORACLE_RATIO + 1e-12,
                     {**st0, "oracle_rel_l2_rerope": r_rr, "oracle_rel_l2_compact": r_cp,
                      "oracle_ratio": ratio})
                neg = diff_stats(ko, kc)
                gate(f"neg_unrotated_layer0_k_{name}", neg["rel_l2"] > NEG_UNROTATED_MIN_REL_L2, neg)
        # system rows: K and V bit-identical with Ours
        gate("sys_k_bitident", bit_equal(rr.system(li, "k"), ours.system(li, "k")), {"layer": li})
        gate("sys_v_bitident", bit_equal(rr.system(li, "v"), ours.system(li, "v")), {"layer": li})
        layers.append(row)
    return {"status": "FAIL" if fail else "PASS", "failed_gates": sorted(set(fail)),
            "k": k, "n_sep": len(sep), "gates": gates, "layers": layers}


# ------------------------------------------------------------ Pruned-T1
class _Abort(Exception):
    pass


class AbortAfterEmbeds:
    """Pre-hook on the language model: keep ``inputs_embeds`` and abort.

    Everything before the language model (vision tower, projector, AnyRes
    merge) has executed when the hook fires; the full-token forward itself
    never runs.
    """

    def __init__(self, runner):
        self.lm = runner.model.model.language_model
        self.embeds = None

    def _hook(self, module, args, kwargs):
        emb = kwargs.get("inputs_embeds")
        assert emb is not None and emb.ndim == 3 and emb.shape[0] == 1
        self.embeds = emb.detach()
        raise _Abort()

    def run(self, runner, enc):
        handle = self.lm.register_forward_pre_hook(self._hook, with_kwargs=True)
        try:
            runner.model(**enc, use_cache=False)
        except _Abort:
            pass
        finally:
            handle.remove()
        assert self.embeds is not None, "language model was not reached"
        return self.embeds


class LayerPositions:
    """Record layer-0 ``position_ids``/``cache_position`` of every forward."""

    def __init__(self, runner):
        self.layer = runner.model.model.language_model.layers[0]
        self.calls = []

    def _hook(self, module, args, kwargs):
        self.calls.append((kwargs["position_ids"], kwargs["cache_position"]))

    def __enter__(self):
        self._h = self.layer.register_forward_pre_hook(self._hook, with_kwargs=True)
        return self

    def __exit__(self, *exc):
        self._h.remove()

    def verify(self, prefill_len):
        """Prefill = arange(prefill_len); decode step t = prefill_len + t."""
        assert self.calls, "no forward observed"
        p0, c0 = (t.detach().cpu().reshape(-1) for t in self.calls[0])
        want = torch.arange(prefill_len)
        assert torch.equal(p0, want) and torch.equal(c0, want), (p0[:3], c0[:3])
        for t, (p, c) in enumerate(self.calls[1:]):
            p, c = p.detach().cpu().reshape(-1), c.detach().cpu().reshape(-1)
            assert p.tolist() == [prefill_len + t] and c.tolist() == [prefill_len + t]
        return {"calls": len(self.calls), "prefill_len": int(prefill_len),
                "contiguous": True, "verified": True}


@torch.no_grad()
def pruned_generate(server, embeds, plan, text_index, first_logits_box=None):
    """One prefill over ``[sys | keep | text]`` then greedy decode.

    Mirrors ``Server._decode``: the TTFT boundary is the host ``int(argmax)``
    of the first token followed by ``cuda.synchronize``; decoding stops at EOS
    or ``max_new_tokens`` decisions.  ``text_index`` are the Turn-1 sequence
    indices of the text tokens after the image span.
    """
    from transformers import DynamicCache
    runner = server.runner
    model, dev = runner.model, runner.model.device
    tok = runner.processor.tokenizer
    index = torch.tensor(list(plan["input_index"]) + list(text_index), dtype=torch.long, device=dev)
    n = int(index.numel())
    torch.cuda.synchronize()
    t_gather = time.perf_counter()
    x = embeds.index_select(1, index)
    pos = torch.arange(n, device=dev)
    cache = DynamicCache()
    prefill_t0 = time.perf_counter()
    out = model(inputs_embeds=x, attention_mask=torch.ones(1, n, dtype=torch.long, device=dev),
                position_ids=pos.unsqueeze(0), cache_position=pos, past_key_values=cache, use_cache=True)
    first = int(out.logits[0, -1].argmax())
    torch.cuda.synchronize()
    first_token_at = time.perf_counter()
    if first_logits_box is not None:
        first_logits_box.append(out.logits[0, -1])
    toks, cur = [first], n
    while toks[-1] != tok.eos_token_id and len(toks) < server.max_new_tokens:
        cp = torch.tensor([cur], device=dev)
        out = model(input_ids=torch.tensor([[toks[-1]]], device=dev),
                    attention_mask=torch.ones(1, cur + 1, dtype=torch.long, device=dev),
                    position_ids=cp.unsqueeze(0), cache_position=cp, past_key_values=cache, use_cache=True)
        toks.append(int(out.logits[0, -1].argmax()))
        cur += 1
    torch.cuda.synchronize()
    finished_at = time.perf_counter()
    answer = tok.decode(toks, skip_special_tokens=True).strip()
    del cache
    return {"answer": answer, "first_token_id": first, "generated_token_ids": list(toks),
            "generated_tokens": len(toks), "prefill_len": n,
            "gather_ms": (prefill_t0 - t_gather) * 1e3,
            "prefill_ms": (first_token_at - prefill_t0) * 1e3,
            "decode_ms": (finished_at - first_token_at) * 1e3,
            "first_token_at_s": first_token_at, "finished_at_s": finished_at}


class FirstLogitsRef:
    """Keep a GPU reference to the first forward's last-position logits.

    No copy and no host sync inside the timed request; the top-2 margin is
    computed afterwards by :func:`top2_margin`.
    """

    def __init__(self, model):
        self.model = model
        self.logits = None

    def _hook(self, module, args, output):
        if self.logits is None and hasattr(output, "logits"):
            self.logits = output.logits[0, -1]

    def __enter__(self):
        self._h = self.model.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._h.remove()


def top2_margin(logits):
    """(top-1 id, top-1 − top-2 logit gap) of a (vocab,) tensor, FP32."""
    if logits is None:
        return None
    top = torch.topk(logits.detach().float(), 2)
    return {"argmax": int(top.indices[0]), "margin": float(top.values[0] - top.values[1])}


def compare_recorded(left, right, atol=1e-4, rtol=1e-4):
    """Non-asserting twin of ``90_validate_llava_kv25._compare_outputs``.

    Used where the contract says a comparison is *recorded*, not gated
    (Pruned-T1 two-step diagnostic vs the Compact hit).
    """
    a, b = left["logits"], right["logits"]
    n = min(len(a), len(b))
    diffs = [(x - y).abs() for x, y in zip(a[:n], b[:n])]
    return {"all_step_logits_within_1e-4": bool(n and len(a) == len(b) and all(
                torch.allclose(x, y, atol=atol, rtol=rtol) for x, y in zip(a, b))),
            "first_step_logits_exact": bool(n and torch.equal(a[0], b[0])),
            "max_abs_logit_delta": max((float(d.max()) for d in diffs), default=None),
            "first_token_exact": left["first_token_id"] == right["first_token_id"],
            "generated_token_ids_exact": left["generated_token_ids"] == right["generated_token_ids"],
            "answer_exact": left["answer"] == right["answer"], "n_steps": (len(a), len(b)),
            "gated": False}


# ------------------------------------------------------------ statistics
def contrast_bootstrap(per_image, contrasts, resamples=10000, seed=1234):
    """Joint image-level bootstrap of several linear contrasts.

    ``per_image`` maps arm -> array (images,) of per-image means; every
    contrast ``{arm: coef}`` is evaluated on the **same** resampled image
    sets.  Returns ``{name: {mean, ci_low, ci_high, ...}}``.
    """
    arms = sorted(per_image)
    data = {a: np.asarray(per_image[a], dtype=float) for a in arms}
    n = len(next(iter(data.values())))
    assert all(len(v) == n for v in data.values()) and n > 0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    out = {}
    for name, coef in contrasts.items():
        point = sum(c * data[a].mean() for a, c in coef.items())
        boots = np.zeros(resamples)
        for a, c in coef.items():
            boots += c * data[a][idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[name] = {"mean": float(point), "ci_low": float(lo), "ci_high": float(hi),
                     "contrast": dict(coef), "resamples": resamples, "seed": seed,
                     "unit": "image", "joint": True, "n_images": n}
    return out

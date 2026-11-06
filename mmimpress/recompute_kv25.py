"""Full-context KV25 vs. reduced-token recompute KV25 ablation helpers.

Ours stores the selected rows S of the decoder KV produced by the normal
full-token Turn-1 forward.  The ablation recomputes the KV of exactly the same
S with a separate prefill over ``[system | sorted(S + separators)]`` (original
relative order), using the Turn-1 ``inputs_embeds`` so the vision tower and
projector are not re-run.  Two position rules are supported:

``original``  every kept token keeps its original absolute position; follow-up
              text starts at ``v_start + v_num`` exactly as in Ours.
``compact``   contiguous positions over the reduced sequence (the VisionZip
              LLaVA convention); follow-up text starts at
              ``v_start + k + n_sep``.

Nothing here edits the existing serving path.  Stores are produced by the
unchanged prefix25 writer from a full-size synthetic cache whose unselected
real rows hold a finite sentinel, and ``PositionServer`` only changes the RoPE
``position_ids`` of suffix/decode tokens; ``cache_position`` (physical buffer
and mask) is untouched.  The plan is expressed as (input index, position ids,
text start) so a Qwen/M-RoPE port can supply its own position rule.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from mmimpress.serve import BIAS, Server
from mmimpress.store import load_meta

SENTINEL = 4096.0           # finite and exactly representable in BF16/FP16
POLICIES = ("original", "compact")
# Frozen in docs/recompute_kv25_ablation_contract.md before GPU execution
# (G2 revision 2: set after the dev samples n355567/n9181, before measurement;
# revision 3: structural rows gated by bit-identity with a same-length
# truncation control, the vs-Ours structural bounds are report-only).
LAYER0_TOL = {"rel_l2": 5e-3, "atol": 1e-2, "rtol": 1.6e-2}   # V elementwise
LAYER0_K_MAX_REL_L2 = 5e-3          # post-RoPE K: no elementwise test
LAYER0_K_MAX_ORACLE_RATIO = 1.5     # rel(other, FP32) <= 1.5 x rel(ours, FP32)
DEEP_STRUCTURAL_MAX_REL_L2 = 3e-2   # report-only since rev. 3 (was the rev. 2 gate)
REALIGN_TOL = {"rel_l2": 1e-2, "atol": 5e-2, "rtol": 5e-2}
NEG_REALIGN_MIN_REL_L2 = 0.1
NEG_LAST_LAYER_MIN_REL_L2 = 5e-3


# ------------------------------------------------------------------ plan
def reduced_plan(v_start, v_num, order, newline_idx, policy):
    """Reduced-prefill sequence for one image under one position policy.

    ``order`` is the stored-to-original visual permutation (real rows in
    stable descending saliency, then separators in original order).
    ``input_index`` indexes the Turn-1 ``inputs_embeds`` sequence.
    """
    if policy not in POLICIES:
        raise ValueError(f"unsupported position policy: {policy}")
    order = [int(x) for x in order]
    sep = [int(x) for x in newline_idx]
    v_start, v_num = int(v_start), int(v_num)
    if sorted(order) != list(range(v_num)):
        raise ValueError("order is not a visual-token permutation")
    n = v_num - len(sep)
    if n < 1 or order[n:] != sep:
        raise ValueError("order must end with the stable separator tail")
    k = (n + 3) // 4
    keep = sorted(order[:k] + sep)
    input_index = list(range(v_start)) + [v_start + i for i in keep]
    if policy == "original":
        position_ids = list(input_index)
        text_start = v_start + v_num
    else:
        position_ids = list(range(len(input_index)))
        text_start = v_start + k + len(sep)
    visual_position = {i: position_ids[v_start + j] for j, i in enumerate(keep)}
    return {"policy": policy, "v_start": v_start, "v_num": v_num, "n": n,
            "k": k, "n_sep": len(sep), "selected_original_ids": order[:k],
            "keep_visual_ids": keep, "input_index": input_index,
            "position_ids": position_ids, "text_position_start": text_start,
            "visual_position": visual_position}


def text_position_start(meta, policy):
    """Follow-up text start for a store served under ``policy``."""
    v_start, v_num = int(meta["v_token_start"]), int(meta["v_token_num"])
    n_sep = len(meta["newline_idx"])
    k = (v_num - n_sep + 3) // 4
    if policy == "original":
        return v_start + v_num
    if policy == "compact":
        return v_start + k + n_sep
    raise ValueError(f"unsupported position policy: {policy}")


def expected_positions(total, qlen, prefix_len, text_start):
    """(position_ids, cache_position) expected for one forward call.

    ``total`` is the attention-mask length (physical prefix + text so far +
    this call).  ``text_start=None`` means the unchanged physical positions.
    """
    cache = torch.arange(total - qlen, total)
    if text_start is None:
        return cache.clone(), cache
    offset = total - qlen - int(prefix_len)
    if offset < 0:
        raise ValueError("forward call starts inside the image prefix")
    return torch.arange(qlen) + int(text_start) + offset, cache


# ------------------------------------------------------------- identity
def _sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def selection_digest(meta):
    """Hash of everything that defines S, k and the stored row order."""
    return _sha_json({
        "order": [int(x) for x in meta["full_importance_permutation"]],
        "k": int(meta["stored_content_count"]),
        "newline_idx": [int(x) for x in meta["newline_idx"]],
        "stored_row_to_original": [int(x) for x in meta["stored_row_to_original"]],
        "v_token_start": int(meta["v_token_start"]),
        "v_token_num": int(meta["v_token_num"]),
    })


def payload_sizes(store):
    """Byte sizes of the KV payload files (metadata files excluded)."""
    root = Path(store)
    names = [p for p in root.rglob("*") if p.is_file() and (
        p.name in ("k.bin", "v.bin", "sep_kv.bin", "sys_kv.pt"))]
    return {str(p.relative_to(root)): p.stat().st_size for p in sorted(names)}


def tensor_sha256(t):
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


# ------------------------------------------------------ Turn-1 embeddings
class EmbeddingCapture:
    """Keep the language model's first-call ``inputs_embeds`` (T1 prefill).

    Read-only pre-hook on ``model.model.language_model``; the tensor is kept
    by reference (no copy inside the request).  Later decode calls are only
    counted.
    """

    def __init__(self, runner):
        self.lm = runner.model.model.language_model
        self.embeds = None
        self.position_ids = None
        self.calls = 0
        self._handle = None

    def _hook(self, module, args, kwargs):
        self.calls += 1
        if self.embeds is None:
            emb = kwargs.get("inputs_embeds")
            assert emb is not None and emb.ndim == 3 and emb.shape[0] == 1, \
                "T1 prefill did not pass inputs_embeds"
            self.embeds = emb.detach()
            self.position_ids = kwargs.get("position_ids")

    def __enter__(self):
        self._handle = self.lm.register_forward_pre_hook(self._hook,
                                                         with_kwargs=True)
        return self

    def __exit__(self, *exc):
        self._handle.remove()


@torch.no_grad()
def reduced_prefill(runner, embeds, plan):
    """Run the language model over the reduced sequence; return KV layers.

    The call is causal over ``plan['input_index']`` order with explicit
    ``position_ids``.  Returns ``(layers, record)`` where layers are
    ``[(k, v)]`` of shape (1, H, len, hd) in reduced order.
    """
    from transformers import DynamicCache
    from mmimpress.model import cache_layers
    if BIAS:
        raise AssertionError("additive KV mask must be empty for reduced prefill")
    lm = runner.model.model.language_model
    dev = embeds.device
    seen = []

    def observe(module, args, kwargs):
        seen.append((kwargs["position_ids"].detach().cpu().reshape(-1),
                     kwargs["cache_position"].detach().cpu().reshape(-1)))

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    index = torch.tensor(plan["input_index"], dtype=torch.long, device=dev)
    reduced = embeds.index_select(1, index)
    pos = torch.tensor(plan["position_ids"], dtype=torch.long, device=dev)
    cpos = torch.arange(len(plan["input_index"]), device=dev)
    torch.cuda.synchronize()
    gather_ms = (time.perf_counter() - t0) * 1e3
    handle = lm.layers[0].register_forward_pre_hook(observe, with_kwargs=True)
    try:
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out = lm(inputs_embeds=reduced, position_ids=pos.unsqueeze(0),
                 cache_position=cpos, past_key_values=DynamicCache(),
                 use_cache=True, return_dict=True)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t1) * 1e3
    finally:
        handle.remove()
    assert len(seen) == 1, "reduced prefill must be a single forward"
    assert seen[0][0].tolist() == plan["position_ids"], "reduced position ids"
    assert seen[0][1].tolist() == list(range(len(plan["input_index"])))
    layers = cache_layers(out.past_key_values)
    assert all(k.shape[2] == len(plan["input_index"]) for k, _ in layers)
    return layers, {"embed_gather_ms": gather_ms, "reduced_prefill_ms": prefill_ms,
                    "reduced_tokens": len(plan["input_index"]),
                    "observed_position_ids_sha256": _sha_json(seen[0][0].tolist()),
                    "observed_position_ids_match": True,
                    "position_policy": plan["policy"]}


@torch.no_grad()
def synthetic_layers(reduced, plan, prefix_len, sentinel=SENTINEL):
    """Scatter reduced KV into full-size (1, H, prefix_len, hd) tensors.

    Rows outside ``plan['input_index']`` (the unselected real tokens) hold
    the finite sentinel; the prefix25 writer never persists them.
    """
    index = torch.tensor(plan["input_index"], dtype=torch.long)
    out = []
    for k, v in reduced:
        pair = []
        for t in (k, v):
            full = torch.full((1, t.shape[1], int(prefix_len), t.shape[3]),
                              sentinel, dtype=t.dtype, device=t.device)
            full[:, :, index.to(t.device)] = t
            pair.append(full)
        out.append(tuple(pair))
    return out


def structural_rows(plan):
    """Prefix rows whose reduced index, position and causal context are original.

    System rows plus every image row before the first unselected real token.
    Under both position policies these rows see exactly the original prefix.
    """
    keep, pos = plan["input_index"], plan["position_ids"]
    n = 0
    while n < len(keep) and keep[n] == n:
        n += 1
    assert pos[:n] == list(range(n)) and n >= plan["v_start"]
    return list(range(n))


@torch.no_grad()
def truncation_control(runner, embeds, plan):
    """G2 rev. 3 control: the original sequence truncated to the reduced length.

    Same shape (and therefore the same GEMM kernels) as the reduced prefill,
    with original context and positions; the rows of :func:`structural_rows`
    must be bit-identical to the reduced prefill.  Returns those rows of every
    layer as CPU BF16 ``(rows, H, hd)``.  Callers run it outside every timer.
    """
    n = len(plan["input_index"])
    rows = structural_rows(plan)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    layers, _ = reduced_prefill(runner, embeds, {"policy": "truncation_control",
                                                 "input_index": list(range(n)),
                                                 "position_ids": list(range(n))})
    idx = torch.tensor(rows, dtype=torch.long, device=embeds.device)
    out = [tuple(t[0].index_select(1, idx).transpose(0, 1).contiguous().cpu()
                 for t in pair) for pair in layers]
    del layers
    torch.cuda.synchronize()
    return {"rows": rows, "length": n, "layers": out,
            "control_ms": (time.perf_counter() - t0) * 1e3, "outside_timers": True}


_BITS = {torch.float16: torch.int16, torch.bfloat16: torch.int16,
         torch.float32: torch.int32, torch.float64: torch.int64}


def bit_equal(a, b):
    """Bitwise tensor equality (distinguishes -0/+0; no tolerance)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(a.contiguous().view(_BITS[a.dtype]),
                       b.contiguous().view(_BITS[b.dtype]))


def structural_identity(reduced, control):
    """In-memory bit-identity of structural rows with the control.

    ``reduced`` is ``[(k, v)]`` of shape (1, H, len, hd), either the reduced
    KV or the full-size synthetic cache (structural rows keep their index in
    both).  Returns a per-layer mismatch list.
    """
    rows = control["rows"]
    bad = []
    for li, ((rk, rv), (ck, cv)) in enumerate(zip(reduced, control["layers"])):
        for kind, r, c in (("k", rk, ck), ("v", rv, cv)):
            r = r[0, :, rows].transpose(0, 1).cpu()
            if not bit_equal(r.contiguous(), c):
                diff = (r.double() - c.double()).abs()
                bad.append({"layer": li, "kind": kind,
                            "mismatched_rows": int((diff.flatten(1) != 0).any(1).sum())
                            if r.shape == c.shape else None})
    return {"ok": not bad and len(reduced) == len(control["layers"]),
            "rows": len(rows), "layers": len(control["layers"]), "mismatches": bad}


# --------------------------------------------------------------- RoPE
def default_inv_freq(head_dim, theta=10000.0):
    return 1.0 / (theta ** (torch.arange(0, head_dim, 2,
                                         dtype=torch.float64) / head_dim))


def rotate(x, delta, inv_freq):
    """Apply RoPE rotation by ``delta`` positions; x (..., rows, H, hd).

    Uses the Llama rotate_half convention; R(q) R(-p) = R(q - p).
    """
    x = x.to(torch.float64)
    ang = torch.as_tensor(delta, dtype=torch.float64)[:, None] * \
        torch.as_tensor(inv_freq, dtype=torch.float64)[None, :]
    ang = torch.cat([ang, ang], dim=-1)[:, None, :]       # rows, 1, hd
    half = x.shape[-1] // 2
    rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * ang.cos() + rot * ang.sin()


# ----------------------------------------------------------- KV diffs
def diff_stats(actual, reference):
    a = torch.as_tensor(actual).to(torch.float64)
    b = torch.as_tensor(reference).to(torch.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    if a.numel() == 0:
        return {"rows": 0, "rel_l2": 0.0, "max_abs": 0.0, "cosine": 1.0,
                "bit_exact_fraction": 1.0}
    d = a - b
    denom = float(b.norm())
    flat_a, flat_b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    cos = torch.nn.functional.cosine_similarity(flat_a, flat_b, dim=-1)
    return {"rows": int(a.shape[0]),
            "rel_l2": float(d.norm()) / denom if denom else float(d.norm()),
            "max_abs": float(d.abs().max()), "cosine": float(cos.mean()),
            "bit_exact_fraction": float((a == b).double().mean())}


def within(actual, reference, tol):
    a = torch.as_tensor(actual).to(torch.float64)
    b = torch.as_tensor(reference).to(torch.float64)
    st = diff_stats(a, b)
    ok = st["rel_l2"] <= tol["rel_l2"] and bool(
        torch.allclose(a, b, atol=tol["atol"], rtol=tol["rtol"]))
    return ok, st


class StoreRows:
    """FP16 payload rows of one prefix25 store, as FP32 tensors."""

    def __init__(self, store):
        self.dir = Path(store)
        self.meta = load_meta(self.dir)
        m = self.meta
        self.L, self.H, self.hd = (int(m["num_layers"]), int(m["num_heads"]),
                                   int(m["head_dim"]))
        self.k = int(m["stored_content_count"])
        self.n_sep = len(m["newline_idx"])
        sep = np.fromfile(self.dir / "sep_kv.bin", dtype=np.float16)
        self.sep = torch.from_numpy(sep.reshape(2, self.L, self.n_sep, self.H,
                                                self.hd).copy())
        self.sys = torch.load(self.dir / "sys_kv.pt", weights_only=True)

    def content(self, layer, kind):
        a = np.fromfile(self.dir / f"layer_{layer:02d}" / f"{kind}.bin",
                        dtype=np.float16)
        return torch.from_numpy(a.reshape(self.k, self.H, self.hd).copy())

    def separators(self, layer, kind):
        return self.sep[0 if kind == "k" else 1, layer]

    def system(self, layer, kind):           # (v_start, H, hd)
        return self.sys[kind][layer].permute(1, 0, 2)


def sentinel_rows(store, sentinel=SENTINEL):
    """Count (row, head) vectors equal to the sentinel in every payload file."""
    rows = StoreRows(store)
    hits = 0
    for li in range(rows.L):
        for kind in ("k", "v"):
            for t in (rows.content(li, kind), rows.separators(li, kind)):
                hits += int((t == sentinel).all(dim=-1).sum())
    return hits


def compare_stores(ours_dir, other_dir, policy, inv_freq, oracle, control):
    """Gate G2 (rev. 3) plus per-layer KV differences of ``other`` against Ours.

    ``oracle`` holds FP32 layer-0 ``k``/``v`` for every prefix row at its
    original position, shape (prefix_len, H, hd); see :func:`layer0_oracle`.
    ``control`` is the :func:`truncation_control` of this arm; its structural
    rows must equal the stored rows bit for bit at every layer.  The
    structural vs-Ours bounds of rev. 2 are recorded in ``report_only``.
    """
    ours, other = StoreRows(ours_dir), StoreRows(other_dir)
    m = ours.meta
    if selection_digest(m) != selection_digest(other.meta):
        raise AssertionError("selection digest differs between stores")
    v_start = int(m["v_token_start"])
    order = [int(x) for x in m["full_importance_permutation"]]
    sep = [int(x) for x in m["newline_idx"]]
    plan = reduced_plan(v_start, int(m["v_token_num"]), order, sep, policy)
    k = plan["k"]
    sel = order[:k]
    orig_pos = torch.tensor([v_start + i for i in sel], dtype=torch.float64)
    sep_orig = torch.tensor([v_start + i for i in sep], dtype=torch.float64)
    new_pos = torch.tensor([plan["visual_position"][i] for i in sel],
                           dtype=torch.float64)
    sep_new = torch.tensor([plan["visual_position"][i] for i in sep],
                           dtype=torch.float64)
    selected = set(sel)
    dropped = [i for i in range(len(order)) if i not in selected and i not in sep]
    first_drop = min(dropped) if dropped else len(order)
    common_s = [j for j, i in enumerate(sel) if i < first_drop]
    common_sep = [j for j, i in enumerate(sep) if i < first_drop]
    moved_s = [j for j in range(k) if float(new_pos[j]) != float(orig_pos[j])]

    gates, layers, report = {}, [], {}
    fail = []

    def gate(name, ok, stats):
        gates.setdefault(name, []).append({"ok": bool(ok), **stats})
        if not ok:
            fail.append(name)

    def k0(a, b, rows):
        """Layer-0 post-RoPE K: rel L2 vs Ours and FP32-oracle equidistance."""
        ref = oracle["k"][torch.as_tensor(rows, dtype=torch.long)]
        st = diff_stats(a, b)
        r_other, r_ours = diff_stats(a, ref)["rel_l2"], diff_stats(b, ref)["rel_l2"]
        ok = (st["rel_l2"] <= LAYER0_K_MAX_REL_L2 and
              r_other <= LAYER0_K_MAX_ORACLE_RATIO * r_ours + 1e-12)
        return ok, {**st, "oracle_rel_l2_other": r_other,
                    "oracle_rel_l2_ours": r_ours,
                    "oracle_ratio": r_other / r_ours if r_ours else
                    (0.0 if r_other == 0 else float("inf"))}

    def structural(li, kind, a, b, rows):
        """sys / common-prefix rows vs Ours under the rev. 2 bounds (report-only)."""
        if li > 0:
            st = diff_stats(a, b)
            return st["rel_l2"] <= DEEP_STRUCTURAL_MAX_REL_L2, st
        return k0(a, b, rows) if kind == "k" else within(a, b, LAYER0_TOL)

    def record(name, ok, stats):
        report.setdefault(name, []).append({"within_rev2_bound": bool(ok), **stats})

    # Structural rows as stored, in absolute-row order of control["rows"].
    c_rows = control["rows"]
    struct_s = [(j, v_start + i) for j, i in enumerate(sel) if v_start + i in set(c_rows)]
    struct_p = [(j, v_start + i) for j, i in enumerate(sep) if v_start + i in set(c_rows)]
    if [r for r in c_rows if r >= v_start] != sorted(r for _, r in struct_s + struct_p) or \
            len(c_rows) != v_start + len(common_s) + len(common_sep):
        raise AssertionError("truncation control rows differ from sys + common prefix")

    def stored_structural(store, li, kind):
        by_row = dict(zip(range(v_start), store.system(li, kind)))
        s, p = store.content(li, kind), store.separators(li, kind)
        by_row.update({r: s[j] for j, r in struct_s})
        by_row.update({r: p[j] for j, r in struct_p})
        return torch.stack([by_row[r] for r in c_rows])

    sys_rows = list(range(v_start))
    s_rows = [v_start + i for i in sel]
    sep_rows = [v_start + i for i in sep]
    for li in range(ours.L):
        row = {"layer": li}
        for kind in ("k", "v"):
            a_s, b_s = other.content(li, kind), ours.content(li, kind)
            a_p, b_p = other.separators(li, kind), ours.separators(li, kind)
            row[f"{kind}_S"] = diff_stats(a_s, b_s)
            row[f"{kind}_sep"] = diff_stats(a_p, b_p)
            ok, st = structural(li, kind, other.system(li, kind),
                                ours.system(li, kind), sys_rows)
            row[f"{kind}_sys"] = st
            record(f"sys_{'layer0' if li == 0 else 'deep'}_{kind}", ok,
                   {"layer": li, **st})
            if common_s or common_sep:
                ca = torch.cat([a_s[common_s], a_p[common_sep]])
                cb = torch.cat([b_s[common_s], b_p[common_sep]])
                rows = [s_rows[j] for j in common_s] + [sep_rows[j] for j in common_sep]
                ok, st = structural(li, kind, ca, cb, rows)
                record(f"common_prefix_{'layer0' if li == 0 else 'deep'}_{kind}",
                       ok, {"layer": li, **st})
            # G2 rev. 3 gate: bit-identity with the same-length truncation control
            got = stored_structural(other, li, kind)
            ref = control["layers"][li][0 if kind == "k" else 1].to(got.dtype)
            same = bit_equal(got, ref)
            gate(f"structural_bitident_{kind}", same,
                 {"layer": li, "rows": len(c_rows),
                  "mismatched_rows": int(((got.double() - ref.double()) != 0)
                                         .flatten(1).any(1).sum())})
            # shape-only floor: control vs Ours on the same rows (recorded)
            row[f"{kind}_structural_control_vs_ours"] = diff_stats(
                ref, stored_structural(ours, li, kind))
            row[f"{kind}_common_control_vs_ours"] = diff_stats(
                ref[v_start:], stored_structural(ours, li, kind)[v_start:])
            if kind == "k" and policy == "compact":
                al_s = rotate(b_s, new_pos - orig_pos, inv_freq)
                al_p = rotate(b_p, sep_new - sep_orig, inv_freq)
                row["k_S_realigned"] = diff_stats(a_s, al_s)
                row["k_sep_realigned"] = diff_stats(a_p, al_p)
            if li == 0:
                if kind == "v":
                    ok1, st1 = within(a_s, b_s, LAYER0_TOL)
                    ok2, st2 = within(a_p, b_p, LAYER0_TOL)
                    gate("layer0_v", ok1 and ok2, {"S": st1, "sep": st2})
                elif policy == "original":
                    ok, st = k0(torch.cat([a_s, a_p]), torch.cat([b_s, b_p]),
                                s_rows + sep_rows)
                    gate("layer0_k", ok, st)
                else:
                    ok1, st1 = within(a_s, al_s, REALIGN_TOL)
                    ok2, st2 = within(a_p, al_p, REALIGN_TOL)
                    gate("layer0_k_realigned", ok1 and ok2,
                         {"S": st1, "sep": st2})
                    neg = diff_stats(a_s[moved_s], b_s[moved_s])
                    gate("neg_layer0_k_unaligned_moved_rows",
                         bool(moved_s) and neg["rel_l2"] > NEG_REALIGN_MIN_REL_L2,
                         {"moved_rows": len(moved_s), **neg})
        layers.append(row)
    if policy == "original":
        last = layers[-1]
        gate("impl_check_last_layer_context_changed",
             last["k_S"]["rel_l2"] > NEG_LAST_LAYER_MIN_REL_L2 and
             last["v_S"]["rel_l2"] > NEG_LAST_LAYER_MIN_REL_L2,
             {"k_rel_l2": last["k_S"]["rel_l2"],
              "v_rel_l2": last["v_S"]["rel_l2"],
              "interpretation": "implementation check only"})
    return {"policy": policy, "status": "FAIL" if fail else "PASS",
            "failed_gates": sorted(set(fail)), "k": k, "n_sep": len(sep),
            "common_prefix_rows": len(common_s) + len(common_sep),
            "first_dropped_original_id": first_drop,
            "moved_selected_rows": len(moved_s), "gates": gates,
            "report_only": report, "structural_rows": len(c_rows),
            "layers": layers}


@torch.no_grad()
def layer0_oracle(runner, embeds, prefix_len, inv_freq):
    """FP32 layer-0 K/V of the prefix: RMSNorm -> dequantized NF4 -> RoPE.

    Positions are the original ``arange(prefix_len)``; returned on CPU as
    float64 (prefix_len, H, hd).  Only valid for the bitsandbytes 4-bit model.
    """
    import bitsandbytes.functional as bnbf
    layer = runner.model.model.language_model.layers[0]
    x = embeds[0, :int(prefix_len)].float()
    ln = layer.input_layernorm
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + ln.variance_epsilon) \
        * ln.weight.float()
    out = {}
    for kind, proj in (("k", layer.self_attn.k_proj), ("v", layer.self_attn.v_proj)):
        w = bnbf.dequantize_4bit(proj.weight.data, proj.weight.quant_state).float()
        y = (x @ w.T).view(x.shape[0], -1, runner.head_dim).double().cpu()
        if kind == "k":
            y = rotate(y, torch.arange(x.shape[0], dtype=torch.float64), inv_freq)
        out[kind] = y
    return out


# -------------------------------------------------------------- serving
class PositionServer(Server):
    """Server whose suffix/decode RoPE positions start at a given offset.

    ``text_position_start=None`` is the unchanged base implementation.  When
    set, only ``position_ids`` change; ``cache_position`` and the attention
    mask keep the physical full-size prefix buffer.  ``pre_decode`` is an
    optional diagnostic callback receiving the installed cache.
    """

    text_position_start = None
    pre_decode = None

    @torch.no_grad()
    def _decode(self, cache, suffix_ids, prefix_len):
        if self.pre_decode is not None:
            self.pre_decode(cache, prefix_len)
        if self.text_position_start is None:
            return super()._decode(cache, suffix_ids, prefix_len)
        assert self.max_new_tokens >= 1
        model = self.runner.model
        dev = model.device
        tok = self.runner.processor.tokenizer
        n = suffix_ids.shape[0]
        start = int(self.text_position_start)
        pos = torch.arange(prefix_len, prefix_len + n, device=dev)
        rope = torch.arange(start, start + n, device=dev)

        prefill_t0 = time.perf_counter()
        out = model(input_ids=suffix_ids.to(dev).unsqueeze(0),
                    attention_mask=torch.ones(1, prefix_len + n,
                                              dtype=torch.long, device=dev),
                    position_ids=rope.unsqueeze(0), cache_position=pos,
                    past_key_values=cache, use_cache=True)
        first = int(out.logits[0, -1].argmax())
        torch.cuda.synchronize()
        first_token_at = time.perf_counter()
        prefill_ms = (first_token_at - prefill_t0) * 1e3

        decode_t0 = first_token_at
        toks, cur = [first], prefix_len + n
        while toks[-1] != tok.eos_token_id and len(toks) < self.max_new_tokens:
            prev = toks[-1]
            cp = torch.tensor([cur], device=dev)
            rp = torch.tensor([cur - prefix_len + start], device=dev)
            out = model(input_ids=torch.tensor([[prev]], device=dev),
                        attention_mask=torch.ones(1, cur + 1, dtype=torch.long,
                                                  device=dev),
                        position_ids=rp.unsqueeze(0), cache_position=cp,
                        past_key_values=cache, use_cache=True)
            nxt = int(out.logits[0, -1].argmax())
            cur += 1
            toks.append(nxt)
        torch.cuda.synchronize()
        finished_at = time.perf_counter()
        answer = tok.decode(toks, skip_special_tokens=True).strip()
        return answer, first, {
            "first_token_at": first_token_at,
            "finished_at": finished_at,
            "prefill_ms": prefill_ms,
            "decode_ms": (finished_at - decode_t0) * 1e3,
            "generated_tokens": len(toks),
            "generated_token_ids": list(toks),
            "generated_token_count": len(toks),
        }


class PositionRecorder:
    """Keep references to every forward's position tensors (no host sync).

    Verification happens after the timed request via :meth:`verify`.
    """

    def __init__(self, model):
        self.model = model
        self.calls = []

    def _hook(self, module, args, kwargs):
        self.calls.append((kwargs["position_ids"], kwargs["cache_position"],
                           kwargs["attention_mask"], kwargs["input_ids"].shape[1]))

    def __enter__(self):
        self._handle = self.model.register_forward_pre_hook(self._hook,
                                                            with_kwargs=True)
        return self

    def __exit__(self, *exc):
        self._handle.remove()

    def verify(self, prefix_len, text_start):
        checks = []
        for pos, cpos, mask, qlen in self.calls:
            total = int(mask.shape[-1])
            want_pos, want_cache = expected_positions(total, qlen, prefix_len,
                                                      text_start)
            got_pos = pos.detach().cpu().reshape(-1)
            got_cache = cpos.detach().cpu().reshape(-1)
            assert torch.equal(got_pos, want_pos), (got_pos[:4], want_pos[:4])
            assert torch.equal(got_cache, want_cache)
            assert bool(mask.all())
            checks.append({"qlen": int(qlen), "first_position": int(got_pos[0]),
                           "first_cache_position": int(got_cache[0])})
        assert checks, "no forward call observed"
        return {"calls": len(checks), "prefill_first_position":
                checks[0]["first_position"], "verified": True,
                "text_position_start": (int(text_start) if text_start is not None
                                        else int(prefix_len))}


def cache_sentinel_check(cache, prefix_len, sentinel=SENTINEL):
    """Gate B: no sentinel value in any BIAS-visible prefix row of any layer."""
    visible_rows = 0
    for li, layer in enumerate(cache.layers):
        bias = BIAS.get(li)
        assert bias is not None, f"layer {li} has no installed mask"
        visible = (bias[0, 0, 0, :prefix_len] == 0)
        visible_rows = int(visible.sum())
        for t in (layer.keys, layer.values):
            rows = t[0, :, :prefix_len][:, visible]
            if bool((rows == sentinel).any()):
                raise AssertionError(f"sentinel visible in layer {li}")
    return {"layers": len(cache.layers), "visible_rows_per_layer": visible_rows,
            "sentinel_visible": False}


# ------------------------------------------------------- guarded cleanup
def guarded_rmtree(path, allowed_root):
    """Delete ``path`` only if strictly inside ``allowed_root``; else raise."""
    p = Path(path)
    root = Path(allowed_root).resolve()
    if p.is_symlink():
        raise RuntimeError(f"refusing to delete symlink: {p}")
    rp = p.resolve()
    if rp == root or not rp.is_relative_to(root):
        raise RuntimeError(f"refusing to delete outside {root}: {rp}")
    shutil.rmtree(rp)


# ------------------------------------------------------------- statistics
def paired_bootstrap(per_image, resamples=10000, seed=1234):
    """Image-cluster paired bootstrap of mean(a) - mean(b).

    ``per_image`` is a list of (sum_a, sum_b, count) per image.
    """
    arr = np.asarray(per_image, dtype=np.float64)
    point = (arr[:, 0].sum() - arr[:, 1].sum()) / arr[:, 2].sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(resamples, len(arr)))
    s = arr[idx]
    boot = (s[..., 0].sum(1) - s[..., 1].sum(1)) / s[..., 2].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "resamples": int(resamples), "seed": int(seed), "unit": "image"}


# ------------------------------------------------------------ protection
def _sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_manifest(roots, base, exclude=(), workers=8):
    """Full SHA256 of every regular file under ``roots`` (paths rel. base)."""
    from concurrent.futures import ProcessPoolExecutor
    base = Path(base)
    files = []
    for r in roots:
        r = Path(r)
        it = [r] if r.is_file() else sorted(p for p in r.rglob("*"))
        for p in it:
            rel = str(p.relative_to(base))
            if p.is_file() and not p.is_symlink() and not any(
                    rel == e or rel.startswith(e.rstrip("/") + "/") for e in exclude):
                files.append(p)
    with ProcessPoolExecutor(workers) as pool:
        digests = list(pool.map(_sha_file, files, chunksize=16))
    return {str(p.relative_to(base)): {"size": p.stat().st_size, "sha256": d}
            for p, d in zip(files, digests)}


def stat_inventory(roots, base):
    """(size, mtime_ns, inode) of every file/symlink under ``roots``."""
    base = Path(base)
    out = {}
    for r in roots:
        for dirpath, dirnames, filenames in os.walk(r):
            for name in filenames:
                p = Path(dirpath) / name
                st = p.lstat()
                out[str(p.relative_to(base))] = [st.st_size, st.st_mtime_ns,
                                                 st.st_ino]
    return out

"""CPU unit test for mha_prefix_reuse (Stage 4) against the original mha.

Checks, with random fp32 weights on a tiny layer (no model needed):
  1. ratio=1.0 (keep all): running the original TorchDevice.mha on the FULL
     sequence and slicing the last n rows must equal mha_prefix_reuse fed
     with (a) the prefix KV returned by that full run and (b) only the new
     tokens' hidden — causality makes the two mathematically identical.
     Also the returned k_new/v_new must equal the full run's KV tail.
  2. selection path with k_keep == m (ratio just below 1 on a divisible m)
     must match the ratio=1.0 output (same key set, re-normalized softmax).
  3. ratio=0.5 must (i) run, (ii) differ from full reuse, (iii) match a
     brute-force per-head reference computed with plain loops.

Usage: python -m flexllmgen.impress.test_prefix_reuse_unit
"""

import numpy as np
import torch

from flexllmgen.pytorch_backend import TorchDevice, TorchTensor
from flexllmgen.impress.prefix_reuse import mha_prefix_reuse
from flexllmgen.impress.importance import column_sum_importance

B, N_HEAD, HEAD_DIM = 1, 4, 8
H = N_HEAD * HEAD_DIM
M, N = 12, 5  # prefix / new tokens


def tt(dev, t):
    return TorchTensor.create_from_torch(t, dev)


def make_weights(dev, g):
    def w(*shape):
        return tt(dev, torch.randn(*shape, generator=g) * 0.1)
    return dict(w_q=w(H, H), b_q=w(H), w_k=w(H, H), b_k=w(H),
                w_v=w(H, H), b_v=w(H), w_out=w(H, H), b_out=w(H),
                w_ln=tt(dev, torch.ones(H)), b_ln=w(H))


def brute_force_selected(x_full, wd, ratio):
    """Loop-based reference for the per-head selection path (fp32)."""
    import torch.nn.functional as F
    x = x_full
    hid = F.layer_norm(x, (H,), weight=wd["w_ln"].data, bias=wd["b_ln"].data)
    scaling = HEAD_DIM ** -0.5
    q = (F.linear(hid, wd["w_q"].data, wd["b_q"].data) * scaling)[0]
    kk = F.linear(hid, wd["w_k"].data, wd["b_k"].data)[0]
    vv = F.linear(hid, wd["w_v"].data, wd["b_v"].data)[0]
    q = q.view(M + N, N_HEAD, HEAD_DIM)
    kk = kk.view(M + N, N_HEAD, HEAD_DIM)
    vv = vv.view(M + N, N_HEAD, HEAD_DIM)
    k_keep = max(1, int(round(M * ratio)))
    out = torch.empty(N, N_HEAD, HEAD_DIM)
    for h in range(N_HEAD):
        qh = q[M:, h]                       # (N, hd) new-token queries
        # full attention (prefix + causal new) for importance
        logits = qh @ kk[:, h].T            # (N, M+N)
        for j in range(N):
            logits[j, M + j + 1:] = -1e4
        w_full = torch.softmax(logits, dim=1)
        imp = w_full[:, :M].sum(dim=0)      # H2O column sum, this head
        sel = torch.topk(imp, k_keep).indices
        # re-normalized attention over selected prefix + new tokens
        keys = torch.cat([kk[sel, h], kk[M:, h]])
        vals = torch.cat([vv[sel, h], vv[M:, h]])
        logits2 = qh @ keys.T               # (N, k_keep+N)
        for j in range(N):
            logits2[j, k_keep + j + 1:] = -1e4
        w2 = torch.softmax(logits2, dim=1)
        out[:, h] = w2 @ vals
    val = out.reshape(1, N, H)
    val = F.linear(val, wd["w_out"].data, wd["b_out"].data)
    return val + x[:, M:]


def main():
    g = torch.Generator().manual_seed(0)
    dev = TorchDevice("cpu")
    wd = make_weights(dev, g)
    x_full = torch.randn(B, M + N, H, generator=g)

    # ---- reference: original mha on the full sequence ----
    mask_full = tt(dev, torch.ones(B, M + N, dtype=torch.bool))
    ref_val, ref_k, ref_v = dev.mha(
        tt(dev, x_full.clone()), mask_full,
        wd["w_q"], wd["b_q"], wd["w_k"], wd["b_k"], wd["w_v"], wd["b_v"],
        wd["w_out"], wd["b_out"], wd["w_ln"], wd["b_ln"],
        N_HEAD, [False] * 2, False, None)
    prefix_k = ref_k.data[:M].clone()   # (M, b*n_head, hd) storage layout
    prefix_v = ref_v.data[:M].clone()
    ref_tail = ref_val.data[:, M:]      # full-run output rows of new tokens

    def run(ratio):
        mask = tt(dev, torch.ones(B, M + N, dtype=torch.bool))
        return mha_prefix_reuse(
            dev, tt(dev, x_full[:, M:].clone()), mask, prefix_k, prefix_v,
            wd["w_q"], wd["b_q"], wd["w_k"], wd["b_k"], wd["w_v"], wd["b_v"],
            wd["w_out"], wd["b_out"], wd["w_ln"], wd["b_ln"],
            N_HEAD, retention_ratio=ratio)

    # 1. keep-all reuse == full recompute (causality)
    val1, k_new, v_new = run(1.0)
    assert torch.allclose(val1.data, ref_tail, atol=1e-5), (
        (val1.data - ref_tail).abs().max())
    assert torch.allclose(k_new.data, ref_k.data[M:], atol=1e-6)
    assert torch.allclose(v_new.data, ref_v.data[M:], atol=1e-6)
    print("  1. ratio=1.0 == original mha on full sequence (atol 1e-5)")

    # 2. selection path with k_keep == m must equal keep-all
    ratio_all = 0.9999999  # < 1.0 takes the selection path; round(M*r) == M
    assert int(round(M * ratio_all)) == M
    val2, _, _ = run(ratio_all)
    assert torch.allclose(val2.data, val1.data, atol=1e-5), (
        (val2.data - val1.data).abs().max())
    print("  2. selection path with k_keep=m == keep-all path")

    # 3. ratio=0.5: differs from full, matches brute-force reference
    val3, _, _ = run(0.5)
    assert torch.isfinite(val3.data).all()
    assert not torch.allclose(val3.data, val1.data, atol=1e-4), (
        "50% selection should change the output")
    ref_sel = brute_force_selected(x_full.clone(), wd, 0.5)
    assert torch.allclose(val3.data, ref_sel, atol=1e-4), (
        (val3.data - ref_sel).abs().max())
    print("  3. ratio=0.5 matches brute-force per-head reference (atol 1e-4)")

    print("ALL PASSED")


if __name__ == "__main__":
    main()

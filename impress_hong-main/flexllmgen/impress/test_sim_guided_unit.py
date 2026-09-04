"""CPU unit tests for similarity-guided identification (Stage 5, §4.3).

  1. Figure 9 vector counts: prefix of 4 tokens, 32 heads, keep 1 token
     -> naive 128 + 32 = 160 vectors, probe 12 + 64 = 76 vectors, both via
     the count functions and via an actual mha_prefix_reuse_sim run with
     engineered agreement/disagreement between probe heads.
  2. Threshold formula (§4.3): j = (k/n) / (2 - k/n), t = j ** alpha.
     k/n = 0.25 -> j = 1/7 ~= 0.142857, t(0.6) ~= 0.31105.
  3. Jaccard: the paper's Figure 7(a) example h0 = {0,2}, h1 = {0,1}
     -> J = 1/3.
  4. Probe path with fully agreeing heads must equal the Stage-4 naive
     output (same selected set on every head).

Usage: python -m flexllmgen.impress.test_sim_guided_unit
"""

import math

import torch

from flexllmgen.pytorch_backend import TorchDevice, TorchTensor
from flexllmgen.impress.prefix_reuse import mha_prefix_reuse
from flexllmgen.impress.similarity_guided import (
    SimGuidedConfig, IOCounter, expected_jaccard, similarity_threshold,
    avg_pairwise_jaccard, count_vectors_naive, count_vectors_probe,
    mha_prefix_reuse_sim)

B, N_HEAD, HEAD_DIM = 1, 32, 8
H = N_HEAD * HEAD_DIM
M, N = 4, 2  # Figure 9: 4 prefix tokens; retention -> 1 token


def tt(dev, t):
    return TorchTensor.create_from_torch(t, dev)


def make_weights(dev, g, identity_kq=False):
    def w(*shape):
        return tt(dev, torch.randn(*shape, generator=g) * 0.1)
    wd = dict(w_q=w(H, H), b_q=tt(dev, torch.zeros(H)),
              w_k=w(H, H), b_k=tt(dev, torch.zeros(H)),
              w_v=w(H, H), b_v=tt(dev, torch.zeros(H)),
              w_out=w(H, H), b_out=tt(dev, torch.zeros(H)),
              w_ln=tt(dev, torch.ones(H)), b_ln=tt(dev, torch.zeros(H)))
    if identity_kq:
        wd["w_q"] = tt(dev, torch.eye(H))
        wd["w_k"] = tt(dev, torch.eye(H))
    return wd


def run_sim(dev, wd, x_new, prefix_k, prefix_v, ratio, use_probe,
            config=None, io=None):
    mask = tt(dev, torch.ones(B, M + N, dtype=torch.bool))
    return mha_prefix_reuse_sim(
        dev, tt(dev, x_new.clone()), mask, prefix_k, prefix_v,
        wd["w_q"], wd["b_q"], wd["w_k"], wd["b_k"], wd["w_v"], wd["b_v"],
        wd["w_out"], wd["b_out"], wd["w_ln"], wd["b_ln"],
        N_HEAD, ratio, config or SimGuidedConfig(),
        io_counter=io, use_probe=use_probe, layer_id=0)


def test_thresholds():
    # k/n = 0.25: j = 0.25 / 1.75 = 1/7
    j = expected_jaccard(1, 4)
    assert abs(j - 1.0 / 7.0) < 1e-12, j
    assert abs(expected_jaccard(25, 100) - 1.0 / 7.0) < 1e-12
    t = similarity_threshold(1, 4, alpha=0.6)
    assert abs(t - (1.0 / 7.0) ** 0.6) < 1e-12
    assert abs(t - 0.3111295) < 1e-6, t
    # k/n = 0.4: j = 0.4 / 1.6 = 0.25, t = 0.25 ** 0.6
    assert abs(expected_jaccard(40, 100) - 0.25) < 1e-12
    assert abs(similarity_threshold(40, 100, 0.6) - 0.25 ** 0.6) < 1e-12
    # alpha = 1 -> t = j; alpha = 0 -> t = 1 (never passes)
    assert abs(similarity_threshold(1, 4, 1.0) - 1.0 / 7.0) < 1e-12
    assert similarity_threshold(1, 4, 0.0) == 1.0
    print("  2. threshold formula matches §4.3 "
          f"(k/n=0.25: j={j:.6f}, t={t:.5f})")


def test_jaccard():
    # paper Figure 7(a): h0 = {0, 2}, h1 = {0, 1} -> J = 1/3
    assert abs(avg_pairwise_jaccard([{0, 2}, {0, 1}]) - 1.0 / 3.0) < 1e-12
    assert avg_pairwise_jaccard([{1, 2}, {1, 2}, {1, 2}]) == 1.0
    assert avg_pairwise_jaccard([{0}, {1}, {2}]) == 0.0
    assert abs(avg_pairwise_jaccard(
        [torch.tensor([0, 2]), torch.tensor([0, 1])]) - 1.0 / 3.0) < 1e-12
    print("  3. Jaccard matches Figure 7(a) example (J({0,2},{0,1}) = 1/3)")


def _unit_query():
    """Alternating +-1 vector: zero mean, unit variance per head, so
    layer_norm passes it through unchanged and (with w_q = I) every head's
    query is exactly its +-1 slice — a controllable direction."""
    w = torch.tensor([1.0, -1.0]).repeat(H // 2)
    return w  # (H,), per-head slices w.view(N_HEAD, HEAD_DIM)


def build_agreeing_case(dev, g):
    """Token 0's key is aligned with every head's query direction -> all
    probe heads pick token 0 -> Jaccard 1 -> probe path taken."""
    wd = make_weights(dev, g, identity_kq=True)
    w = _unit_query()
    x_new = w.expand(B, N, H).clone()
    w_heads = w.view(N_HEAD, HEAD_DIM)
    prefix_k = torch.randn(M, B * N_HEAD, HEAD_DIM, generator=g) * 0.01
    prefix_k[0] = 5.0 * w_heads   # aligned with q in every head
    prefix_v = torch.randn(M, B * N_HEAD, HEAD_DIM, generator=g)
    return wd, x_new, prefix_k, prefix_v


def build_disagreeing_case(dev, g):
    """Head h's aligned prefix token is (h % M) -> probe heads 0,1,2 pick
    tokens 0,1,2 -> Jaccard 0 -> fallback."""
    wd = make_weights(dev, g, identity_kq=True)
    w = _unit_query()
    x_new = w.expand(B, N, H).clone()
    w_heads = w.view(N_HEAD, HEAD_DIM)
    prefix_k = torch.randn(M, B * N_HEAD, HEAD_DIM, generator=g) * 0.01
    for head in range(N_HEAD):
        prefix_k[head % M, head] = 5.0 * w_heads[head]
    prefix_v = torch.randn(M, B * N_HEAD, HEAD_DIM, generator=g)
    return wd, x_new, prefix_k, prefix_v


def test_figure9_vector_counts():
    # count functions reproduce the Figure 9 numbers directly
    assert count_vectors_naive(32, 4, 1) == 128 + 32 == 160
    assert count_vectors_probe(3, 32, 4, 1) == 12 + 64 == 76

    dev = TorchDevice("cpu")
    ratio = 1.0 / M  # keep exactly 1 of 4 tokens

    # naive mode (use_probe=False) -> 160 vectors
    g = torch.Generator().manual_seed(0)
    wd, x_new, p_k, p_v = build_agreeing_case(dev, g)
    io = IOCounter()
    run_sim(dev, wd, x_new, p_k, p_v, ratio, use_probe=False, io=io)
    assert io.total_vectors == 160, io.total_vectors
    assert io.records[0]["mode"] == "naive"

    # probe mode, heads agree -> threshold passed -> 76 vectors
    io = IOCounter()
    run_sim(dev, wd, x_new, p_k, p_v, ratio, use_probe=True, io=io)
    r = io.records[0]
    assert r["mode"] == "probe", r
    assert r["sim"] == 1.0 and r["sim"] > r["threshold"], r
    assert io.total_vectors == 76, io.total_vectors

    # probe mode, heads disagree -> fallback -> 160 vectors
    g = torch.Generator().manual_seed(1)
    wd, x_new, p_k, p_v = build_disagreeing_case(dev, g)
    io = IOCounter()
    run_sim(dev, wd, x_new, p_k, p_v, ratio, use_probe=True, io=io)
    r = io.records[0]
    assert r["mode"] == "fallback", r
    assert r["sim"] == 0.0 and r["sim"] <= r["threshold"], r
    assert io.total_vectors == 160, io.total_vectors
    assert io.fallback_rate() == 1.0

    print("  1. Figure 9 example: naive=160, probe=76 vectors; "
          "disagreement -> fallback=160")


def test_probe_equals_naive_when_agreeing():
    dev = TorchDevice("cpu")
    g = torch.Generator().manual_seed(2)
    wd, x_new, p_k, p_v = build_agreeing_case(dev, g)
    ratio = 1.0 / M

    val_probe, _, _ = run_sim(dev, wd, x_new, p_k, p_v, ratio, use_probe=True)
    val_naive, _, _ = run_sim(dev, wd, x_new, p_k, p_v, ratio, use_probe=False)
    assert torch.allclose(val_probe.data, val_naive.data, atol=1e-5), (
        (val_probe.data - val_naive.data).abs().max())

    # and the sim-module naive path == Stage-4 mha_prefix_reuse
    mask = tt(dev, torch.ones(B, M + N, dtype=torch.bool))
    val_s4, _, _ = mha_prefix_reuse(
        dev, tt(dev, x_new.clone()), mask, p_k, p_v,
        wd["w_q"], wd["b_q"], wd["w_k"], wd["b_k"], wd["w_v"], wd["b_v"],
        wd["w_out"], wd["b_out"], wd["w_ln"], wd["b_ln"],
        N_HEAD, retention_ratio=ratio)
    assert torch.allclose(val_naive.data, val_s4.data, atol=1e-5)
    print("  4. probe output == naive output on full agreement; "
          "naive path == Stage-4 function")


if __name__ == "__main__":
    test_figure9_vector_counts()
    test_thresholds()
    test_jaccard()
    test_probe_equals_naive_when_agreeing()
    print("ALL PASSED")

"""Stage 6 verification: KV reordering (paper §4.4.1, Figures 12/13/20).

1. Figure 13 exact reproduction: p0 = [t0..t7], p1 = [t0..t3, t8..t11],
   important tokens {t0, t3, t4, t9}, chunk_size 2. After reordering:
     node s0: stored order [t0,t3,t1,t2], m0 = [0,2,3,1]
     node s1: stored order [t4,t5,t6,t7], m1 = [0,1,2,3] (identity)
     node s2: stored order [t9,t8,t10,t11], m2 = [1,0,2,3]
   and stored'[m] recovers the original sequence and KV bit-exact.
2. Regression: radix-tree matching (R/NR) and KV loads still correct after
   reordering, including a split of a reordered node.
3. Figure 20 direction: on synthetic 50%-important chunks (chunk_size 2),
   the number of chunks read for important tokens drops >= ~1.2x.
4. Asynchrony: the reorder pass runs in a background thread, triggered by
   request count and by time interval, without blocking the caller.

Usage: python -m flexllmgen.impress.test_reordering
"""

import shutil
import tempfile
import time

import numpy as np
import torch

from flexllmgen.impress.radix_tree import PrefixRadixTree
from flexllmgen.impress.reordering import (KVReorderer, compute_reorder,
    reorder_node, reordered_tokens, chunks_needed)
from flexllmgen.impress.test_radix_tree import (NUM_LAYERS, make_kv,
    kv_slice, kv_equal, kv_concat)


def build_figure13_tree(root):
    """Stage-2 Figure 13 scenario: A = [t0..t7], B = [t0..t3, t8..t11]."""
    p0 = list(range(8))
    p1 = [0, 1, 2, 3, 8, 9, 10, 11]
    tree = PrefixRadixTree(root, chunk_size=2)
    kvA = make_kv(8, seed=100)
    tree.insert(p0, kvA)
    kvB_nr = make_kv(4, seed=200)
    tree.insert(p1, kvB_nr)
    return tree, p0, p1, kvA, kvB_nr


def record_fig13_importance(tree, p0, p1):
    """Important tokens: t0, t3, t4, t9 (paper Figure 13)."""
    imp = {0: 1.0, 3: 1.0, 4: 1.0, 9: 1.0}
    for p in (p0, p1):
        m = tree.match(p)
        tree.record_importance(m, [imp.get(t, 0.0) for t in p])


def test_figure13(tmp):
    tree, p0, p1, kvA, kvB_nr = build_figure13_tree(tmp + "/fig13")
    record_fig13_importance(tree, p0, p1)

    reorderer = KVReorderer(tree)
    done = reorderer.reorder_all()

    n0 = tree.root.children[0]          # s0 = [t0,t1,t2,t3]
    n1 = n0.children[4]                 # s1 = [t4..t7]
    n2 = n0.children[8]                 # s2 = [t8..t11]

    # paper Figure 13(b): s0' = [t0,t3,t1,t2], m0 = [0,2,3,1]
    assert reordered_tokens(n0) == [0, 3, 1, 2], reordered_tokens(n0)
    assert n0.mapping == [0, 2, 3, 1], n0.mapping
    # s1 unchanged, m1 = identity (paper shows [0,1,2,3]; we skip the
    # physical rewrite when the order is already correct, so mapping stays
    # None — semantically the same identity mapping)
    assert reordered_tokens(n1) == [4, 5, 6, 7]
    assert n1.mapping in (None, [0, 1, 2, 3]), n1.mapping
    # s2' = [t9,t8,t10,t11], m2 = [1,0,2,3]
    assert reordered_tokens(n2) == [9, 8, 10, 11], reordered_tokens(n2)
    assert n2.mapping == [1, 0, 2, 3], n2.mapping

    # s'[m] recovers the original sequence (tokens and KV, bit-exact)
    s_prime = reordered_tokens(n0)
    assert [s_prime[i] for i in n0.mapping] == n0.tokens == [0, 1, 2, 3]
    mA = tree.match(p0)
    assert kv_equal(tree.load_prefix_kv(mA), kvA), "A KV changed by reorder"
    mB = tree.match(p1)
    assert kv_equal(tree.load_prefix_kv(mB),
                    kv_concat(kv_slice(kvA, 0, 4), kvB_nr))

    # on-disk layer metadata carries the mapping list too
    assert n0.store.layer(0).meta["token_order"] == [0, 2, 3, 1]
    # Figure 12: {t0,t3} now sit in ONE chunk instead of two
    assert chunks_needed(n0, 0, [0, 3]) == 1
    print(f"  1. Figure 13 reproduced: m0=[0,2,3,1], m1=identity, "
          f"m2=[1,0,2,3]; s'[m]==s; KV bit-exact ({len(done)} nodes)")
    return tree, p0, p1, kvA, kvB_nr


def test_regression_after_reorder(tree, p0, p1, kvA, kvB_nr):
    # matching still exact
    mB = tree.match(p1)
    assert mB.R == p1 and mB.NR == []
    mP = tree.match([0, 1, 2, 3, 8, 9, 99])
    assert mP.R == [0, 1, 2, 3, 8, 9] and mP.NR == [99]
    assert kv_equal(
        tree.load_prefix_kv(mP),
        kv_concat(kv_slice(kvA, 0, 4), kv_slice(kvB_nr, 0, 2)))

    # split a REORDERED node: pD diverges at 3 inside reordered s0
    pD = [0, 1, 2, 77, 78]
    mD = tree.match(pD)
    assert mD.R == [0, 1, 2] and mD.NR == [77, 78], mD
    kvD_nr = make_kv(2, seed=300)
    tree.insert(pD, kvD_nr)
    n0 = tree.root.children[0]
    assert n0.tokens == [0, 1, 2] and n0.mapping is None
    assert sorted(n0.children) == [3, 77]
    # all prefixes still load original values
    assert kv_equal(tree.load_prefix_kv(tree.match(p0)), kvA)
    assert kv_equal(tree.load_prefix_kv(tree.match(pD)),
                    kv_concat(kv_slice(kvA, 0, 3), kvD_nr))

    # reopened-from-disk tree behaves identically
    tree2 = PrefixRadixTree(tree.root_dir)
    assert kv_equal(tree2.load_prefix_kv(tree2.match(p0)), kvA)
    n2 = tree2.root.children[0].children[3].children[8]
    assert n2.mapping == [1, 0, 2, 3]
    print("  2. regression: match/load/split-of-reordered-node/persistence OK")


def test_chunk_reduction(tmp):
    """Figure 20 direction: >= ~1.2x fewer chunks for important tokens."""
    n_tokens, chunk_size, trials = 64, 2, 10
    ratios = []
    for seed in range(trials):
        rng = np.random.RandomState(seed)
        important = rng.rand(n_tokens) < 0.5
        important_pos = [i for i in range(n_tokens) if important[i]]
        if not important_pos:
            continue
        tree = PrefixRadixTree(f"{tmp}/chunks_{seed}", chunk_size=chunk_size)
        tokens = list(range(n_tokens))
        tree.insert(tokens, make_kv(n_tokens, seed=seed))
        node = tree.root.children[0]
        m = tree.match(tokens)
        tree.record_importance(m, important.astype(float).tolist())

        before = chunks_needed(node, 0, important_pos)
        assert reorder_node(tree, node)
        after = chunks_needed(node, 0, important_pos)
        # correctness preserved
        assert kv_equal(tree.load_prefix_kv(tree.match(tokens)),
                        make_kv(n_tokens, seed=seed))
        ratios.append(before / after)
    avg = float(np.mean(ratios))
    assert avg >= 1.2, f"avg chunk reduction {avg:.2f}x < 1.2x"
    print(f"  3. chunk reads for important tokens: avg {avg:.2f}x reduction "
          f"over {len(ratios)} trials (paper Figure 20: ~1.2x)")


def test_async(tmp):
    # (a) request-count trigger
    tree, p0, p1, kvA, _ = build_figure13_tree(tmp + "/async_req")
    record_fig13_importance(tree, p0, p1)
    r = KVReorderer(tree, every_n_requests=3)
    r.start()
    n0 = tree.root.children[0]
    assert n0.mapping is None
    for _ in range(3):
        r.notify_request()   # 3rd call kicks the background pass
    deadline = time.time() + 5.0
    while n0.mapping is None and time.time() < deadline:
        time.sleep(0.02)
    assert n0.mapping == [0, 2, 3, 1], "async pass did not run"
    assert r.passes >= 1 and r.nodes_reordered >= 1
    r.stop()

    # (b) interval trigger; serving path (match/load) keeps working
    tree2, p0, p1, kvA2, _ = build_figure13_tree(tmp + "/async_time")
    record_fig13_importance(tree2, p0, p1)
    r2 = KVReorderer(tree2, interval_seconds=0.1)
    r2.start()
    deadline = time.time() + 5.0
    while tree2.root.children[0].mapping is None and time.time() < deadline:
        # concurrent serving-path reads while the reorderer works
        assert kv_equal(tree2.load_prefix_kv(tree2.match(p0)), kvA2)
        time.sleep(0.02)
    assert tree2.root.children[0].mapping == [0, 2, 3, 1]
    assert kv_equal(tree2.load_prefix_kv(tree2.match(p0)), kvA2)
    r2.stop()
    print("  4. async: request-count and interval triggers ran in background;"
          " concurrent loads stayed correct")


if __name__ == "__main__":
    tmp = tempfile.mkdtemp(prefix="impress_reorder_test_")
    try:
        # compute_reorder sanity on the paper example
        perm, mapping = compute_reorder([1.0, 0.0, 0.0, 1.0])
        assert perm == [0, 3, 1, 2] and mapping == [0, 2, 3, 1]
        tree, p0, p1, kvA, kvB_nr = test_figure13(tmp)
        test_regression_after_reorder(tree, p0, p1, kvA, kvB_nr)
        test_chunk_reduction(tmp)
        test_async(tmp)
        print("ALL PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

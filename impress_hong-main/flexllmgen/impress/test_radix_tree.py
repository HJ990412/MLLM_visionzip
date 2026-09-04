"""Stage 2 verification: radix-tree prefix metadata (paper §4.1, §4.4.1).

Scenario follows the paper's Figure 13 example exactly:
  request A: p0 = [t0..t7]                     (8 tokens)
  request B: p1 = [t0,t1,t2,t3, t8,t9,t10,t11] (shares first 4)
with chunk_size = 2, as in the figure ("Assume the chunk size is set to 2").

Checks (all assert-based):
  1. B matches R = [t0..t3], NR = [t8..t11] exactly.
  2. R's KV loaded through the tree (Stage-1 loader per node) is bit-exact
     vs what request A stored.
  3. After inserting B, the tree has the Figure 13(a) shape:
     root -> [t0..t3] -> {[t4..t7], [t8..t11]}, and the common node's
     k_ptr/v_ptr hold 2 chunk pointers per layer (4 tokens / chunk_size 2).
  4. Identical prefix: re-matching p0 gives R = p0, NR = [] and the full KV
     (across the split) still equals A's original.
  5. Fully disjoint prefix: R = [], NR = everything.
  6. Split at a non-chunk-aligned position (re-chunking path).
  7. Persistence: reopening the tree from disk reproduces all of the above.

Usage: python -m flexllmgen.impress.test_radix_tree
"""

import shutil
import tempfile

import torch

from flexllmgen.impress.radix_tree import PrefixRadixTree

NUM_LAYERS = 2
N_HEAD, HEAD_DIM = 8, 16


def make_kv(num_tokens, seed):
    """Synthetic per-layer KV in FlexGen layout (tokens, n_head, head_dim)."""
    g = torch.Generator().manual_seed(seed)
    return {
        j: (torch.randn(num_tokens, N_HEAD, HEAD_DIM, generator=g,
                        dtype=torch.float32).half(),
            torch.randn(num_tokens, N_HEAD, HEAD_DIM, generator=g,
                        dtype=torch.float32).half())
        for j in range(NUM_LAYERS)
    }


def kv_slice(kv, start, end):
    return {j: (k[start:end], v[start:end]) for j, (k, v) in kv.items()}


def kv_equal(a, b):
    assert a.keys() == b.keys()
    return all(torch.equal(a[j][0], b[j][0]) and torch.equal(a[j][1], b[j][1])
               for j in a)


def kv_concat(a, b):
    return {j: (torch.cat([a[j][0], b[j][0]]), torch.cat([a[j][1], b[j][1]]))
            for j in a}


def run_checks(tree, p0, p1, pC, pD, kvA, kvB_nr, kvC, kvD_nr, label):
    """Assertions that must hold on a fully populated tree (fresh or reopened)."""
    # B: fully matched after insertion; KV = A's shared part + B's own NR
    mB = tree.match(p1)
    assert mB.R == p1 and mB.NR == [], f"[{label}] B match: {mB}"
    kvB_full = tree.load_prefix_kv(tree.match(p1))
    assert kv_equal(kvB_full, kv_concat(kv_slice(kvA, 0, 4), kvB_nr)), (
        f"[{label}] B full KV != A[:4] + B_NR")

    # identical prefix: A fully matched, KV intact across the splits
    mA = tree.match(p0)
    assert mA.R == p0 and mA.NR == [], f"[{label}] A match: {mA}"
    assert kv_equal(tree.load_prefix_kv(mA), kvA), (
        f"[{label}] A full KV changed by splits")

    # disjoint prefix C fully matched after insertion
    mC = tree.match(pC)
    assert mC.R == pC and mC.NR == [], f"[{label}] C match: {mC}"
    assert kv_equal(tree.load_prefix_kv(mC), kvC)

    # D (non-aligned split at 3) fully matched
    mD = tree.match(pD)
    assert mD.R == pD and mD.NR == [], f"[{label}] D match: {mD}"
    assert kv_equal(tree.load_prefix_kv(mD),
                    kv_concat(kv_slice(kvA, 0, 3), kvD_nr))
    print(f"  [{label}] all populated-tree checks passed")


def main():
    tmp = tempfile.mkdtemp(prefix="impress_radix_test_")
    try:
        # Paper Figure 13: t0..t7 / t0..t3 + t8..t11, chunk size 2
        p0 = list(range(0, 8))                    # [t0..t7]
        p1 = [0, 1, 2, 3, 8, 9, 10, 11]           # [t0..t3, t8..t11]
        pC = list(range(100, 108))                # fully disjoint
        pD = [0, 1, 2, 99, 100]                   # diverges at 3 (mid-chunk)

        tree = PrefixRadixTree(tmp, chunk_size=2)

        # ---- request A: empty tree -> R=[], NR=p0 ----
        mA0 = tree.match(p0)
        assert mA0.R == [] and mA0.NR == p0, mA0
        kvA = make_kv(8, seed=100)
        tree.insert(p0, kvA)
        print("  A inserted: R=[], NR=p0 (8 tokens, 4 chunks/layer)")

        # ---- request B BEFORE insertion: the paper's core scenario ----
        mB = tree.match(p1)
        assert mB.R == [0, 1, 2, 3], f"R wrong: {mB.R}"
        assert mB.NR == [8, 9, 10, 11], f"NR wrong: {mB.NR}"
        r_kv = tree.load_prefix_kv(mB)  # loads via Stage-1 loader, node path
        assert kv_equal(r_kv, kv_slice(kvA, 0, 4)), (
            "R KV != KV stored at request-A time")
        print("  B match: R=[t0..t3], NR=[t8..t11]; R KV bit-exact vs A")

        kvB_nr = make_kv(4, seed=200)
        tree.insert(p1, kvB_nr)

        # ---- Figure 13(a) tree shape + k_ptr/v_ptr ----
        n_common = tree.root.children[0]
        assert n_common.tokens == [0, 1, 2, 3], n_common
        assert sorted(n_common.children) == [4, 8], n_common
        assert n_common.children[4].tokens == [4, 5, 6, 7]
        assert n_common.children[8].tokens == [8, 9, 10, 11]
        for j in range(NUM_LAYERS):
            # 4 tokens / chunk_size 2 = 2 chunk pointers, as in Figure 13(a)
            assert len(n_common.k_ptr(j)) == 2, n_common.k_ptr(j)
            assert len(n_common.v_ptr(j)) == 2, n_common.v_ptr(j)
        print("  tree shape == Figure 13(a); k_ptr/v_ptr = 2 chunks/layer")

        # ---- identical prefix: no-op insert allowed ----
        mA2 = tree.insert(p0)  # NR empty -> nr_kv not needed
        assert mA2.R == p0 and mA2.NR == []
        assert kv_equal(tree.load_prefix_kv(tree.match(p0)), kvA), (
            "A KV corrupted by split")
        print("  identical prefix: R=p0, NR=[]; A KV intact after split")

        # ---- fully disjoint prefix ----
        mC = tree.match(pC)
        assert mC.R == [] and mC.NR == pC, mC
        kvC = make_kv(8, seed=300)
        tree.insert(pC, kvC)
        print("  disjoint prefix: R=[], NR=all; inserted as new branch")

        # ---- non-chunk-aligned split: diverge at 3 inside [t0..t3] ----
        mD = tree.match(pD)
        assert mD.R == [0, 1, 2] and mD.NR == [99, 100], mD
        assert kv_equal(tree.load_prefix_kv(mD), kv_slice(kvA, 0, 3))
        kvD_nr = make_kv(2, seed=400)
        tree.insert(pD, kvD_nr)
        n0 = tree.root.children[0]
        assert n0.tokens == [0, 1, 2] and sorted(n0.children) == [3, 99], n0
        print("  mid-chunk split at 3: re-chunked, children = {t3, t99}")

        # ---- full checks on the populated tree ----
        run_checks(tree, p0, p1, pC, pD, kvA, kvB_nr, kvC, kvD_nr, "fresh")

        # ---- persistence: reopen from disk only ----
        tree2 = PrefixRadixTree(tmp)
        assert tree2.chunk_size == 2 and tree2.num_layers == NUM_LAYERS
        run_checks(tree2, p0, p1, pC, pD, kvA, kvB_nr, kvC, kvD_nr, "reopened")

        print("ALL PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

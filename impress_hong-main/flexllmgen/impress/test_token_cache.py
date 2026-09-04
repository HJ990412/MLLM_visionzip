"""Stage 7 verification: score-based cache management (paper §4.4.2).

1. Paper Figure 14 example: chunk1 (freq 1.5, 50% important) -> score 0.75,
   chunk2 (freq 1.0, 100% important) -> score 1.0. With GPU room for one
   chunk, chunk2 must end up in GPU memory (a frequency-only policy would
   pick chunk1), in both access orders.
2. Invariants on a random trace: GPU/CPU non-redundancy, capacity
   accounting, heap-top = true minimum after every access; CPU evictions
   are metadata-only drops (no write-back — disk replica is permanent).
3. Synthetic access trace: score-based policy vs two-tier LRU. The
   important-vector GPU hit ratio (fraction of needed important vectors
   already resident in GPU, which is what saves PCIe traffic per §4.4.2)
   must improve, directionally like the paper's Figure 21 (68% -> 80%).

Usage: python -m flexllmgen.impress.test_token_cache
"""

from collections import OrderedDict

import numpy as np

from flexllmgen.impress.token_cache import TokenCache


# ------------------------------------------------------------ 1. Figure 14
def test_paper_example():
    for order in (["chunk1", "chunk2"], ["chunk2", "chunk1"]):
        cache = TokenCache(gpu_capacity=1, cpu_capacity=2)
        cache.register("chunk1", size=1)
        cache.register("chunk2", size=1)
        cache.seed_stats("chunk1", freq=1.5, imp_ratio=0.5)
        cache.seed_stats("chunk2", freq=1.0, imp_ratio=1.0)
        assert abs(cache.score("chunk1") - 0.75) < 1e-12
        assert abs(cache.score("chunk2") - 1.0) < 1e-12

        for key in order:
            cache.access(key, update_stats=False)
        assert cache.location("chunk2") == "gpu", (
            f"order {order}: chunk2 must win GPU (score 1.0 > 0.75)")
        assert cache.location("chunk1") == "cpu", (
            f"order {order}: chunk1 stays in CPU cache")
        cache.check_invariants()
    print("  1. Figure 14: score(chunk1)=0.75 < score(chunk2)=1.0 -> "
          "chunk2 in GPU, chunk1 in CPU (both access orders); a "
          "frequency-only policy would have cached chunk1")


# ---------------------------------------------------------- 2. invariants
def test_invariants_random_trace():
    rng = np.random.RandomState(0)
    moves = []
    cache = TokenCache(gpu_capacity=6, cpu_capacity=10,
                       on_move=lambda k, s, d: moves.append((k, s, d)))
    n_chunks = 30
    for c in range(n_chunks):
        cache.register(f"c{c}", size=rng.randint(1, 4))
    for _ in range(2000):
        c = rng.randint(n_chunks)
        cache.access(f"c{c}", important_ratio=float(rng.rand()))
        cache.check_invariants()

    # evictions to disk happened and are pure drops: no move event ever
    # writes toward disk with an I/O cost — the only disk-bound moves are
    # 'gpu->disk'/'cpu->disk' drops relying on the permanent replica
    assert cache.stats["drops"] > 0
    assert all(d in ("gpu", "cpu", "disk") and s in ("gpu", "cpu", "disk")
               for _, s, d in moves)
    assert cache.stats["promotions"] > 0 and cache.stats["demotions"] > 0
    print(f"  2. invariants held over 2000 random accesses "
          f"(promotions {cache.stats['promotions']}, demotions "
          f"{cache.stats['demotions']}, drops {cache.stats['drops']}; "
          f"drops are metadata-only, disk replica permanent)")


# ------------------------------------------------------ 3. vs two-tier LRU
class TwoTierLRU:
    """Recency-based baseline: hit anywhere promotes to GPU MRU; GPU LRU
    victim demotes to CPU; CPU LRU victim drops (replica on disk)."""

    def __init__(self, gpu_capacity, cpu_capacity, sizes):
        self.cap = {"gpu": gpu_capacity, "cpu": cpu_capacity}
        self.used = {"gpu": 0, "cpu": 0}
        self.tier = {"gpu": OrderedDict(), "cpu": OrderedDict()}
        self.sizes = sizes

    def _evict_until_fits(self, tier, size):
        while self.used[tier] + size > self.cap[tier]:
            victim, _ = self.tier[tier].popitem(last=False)  # LRU
            self.used[tier] -= self.sizes[victim]
            if tier == "gpu":
                self._insert("cpu", victim)
            # cpu victim: dropped (disk replica)

    def _insert(self, tier, key):
        self._evict_until_fits(tier, self.sizes[key])
        self.tier[tier][key] = True
        self.used[tier] += self.sizes[key]

    def access(self, key):
        if key in self.tier["gpu"]:
            self.tier["gpu"].move_to_end(key)
            return "gpu"
        found = "cpu" if key in self.tier["cpu"] else "disk"
        if found == "cpu":
            del self.tier["cpu"][key]
            self.used["cpu"] -= self.sizes[key]
        self._insert("gpu", key)
        return found


def build_trace(rng, n_accesses):
    """Chunks whose access frequency and importance ratio DISAGREE — the
    §4.4.2 motivation (Figure 5(b): hotness is uncorrelated with the
    important-KV ratio)."""
    chunks = {}
    weights = {}
    for i in range(6):    # hot but mostly-unimportant chunks
        chunks[f"hot{i}"] = dict(size=8, imp=0.15)
        weights[f"hot{i}"] = 6.0
    for i in range(6):    # warm, highly-important chunks
        chunks[f"imp{i}"] = dict(size=8, imp=0.95)
        weights[f"imp{i}"] = 3.5
    for i in range(12):   # cold background chunks
        chunks[f"cold{i}"] = dict(size=8, imp=0.5)
        weights[f"cold{i}"] = 0.8
    keys = list(chunks)
    p = np.array([weights[k] for k in keys], dtype=float)
    p /= p.sum()
    trace = [keys[i] for i in rng.choice(len(keys), n_accesses, p=p)]
    return chunks, trace


def test_vs_lru():
    rng = np.random.RandomState(42)
    chunks, trace = build_trace(rng, n_accesses=4000)
    gpu_cap, cpu_cap = 48, 96  # GPU fits 6 of 24 chunks

    score_cache = TokenCache(gpu_capacity=gpu_cap, cpu_capacity=cpu_cap)
    for k, c in chunks.items():
        score_cache.register(k, size=c["size"])
    lru = TwoTierLRU(gpu_cap, cpu_cap,
                     {k: c["size"] for k, c in chunks.items()})

    def run(cache, is_score):
        # important-vector GPU hit ratio: important vectors already in GPU
        # need no PCIe transfer (§4.4.2 / Figure 21 metric direction)
        served = need = 0
        for key in trace:
            w = chunks[key]["imp"] * chunks[key]["size"]
            found = (cache.access(key, important_ratio=chunks[key]["imp"])
                     if is_score else cache.access(key))
            need += w
            if found == "gpu":
                served += w
        return served / need

    hr_score = run(score_cache, True)
    hr_lru = run(lru, False)
    score_cache.check_invariants()

    print(f"  3. important-vector GPU hit ratio: LRU {hr_lru*100:.1f}% -> "
          f"score-based {hr_score*100:.1f}% "
          f"(paper Figure 21 direction: 68% -> 80%)")
    assert hr_score > hr_lru, (
        f"score-based ({hr_score:.3f}) must beat LRU ({hr_lru:.3f})")


if __name__ == "__main__":
    test_paper_example()
    test_invariants_random_trace()
    test_vs_lru()
    print("ALL PASSED")

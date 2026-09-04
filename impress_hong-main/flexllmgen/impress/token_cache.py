"""Score-based cache management (IMPRESS Stage 7, paper §4.4.2).

Paper (re-checked):
- Score-based cache admission: "assigns a score to each chunk based on its
  access frequency and the proportion of important keys or values it holds"
  — score = access_frequency * importance_ratio. Worked example (Figure 14):
  chunk1 (freq 1.5, 50% important) scores 0.75; chunk2 (freq 1.0, 100%)
  scores 1.0, so chunk2 is cached in GPU memory even though chunk1 is
  accessed more often. "The importance ratio is dynamically calculated as a
  moving average, updating online after each chunk access."
- Dual-cache replacement: "two min-heaps in CPU memory to manage chunks in
  both caches and facilitate eviction. The heaps' tops indicate the
  lowest-scored chunks in the GPU and CPU caches."
- Dataflow: GPU hit -> use vectors, update score, retain. CPU hit -> after
  transferring vectors to GPU, update score and compare with the GPU
  cache's lowest; if superior, replace it, else stay in CPU. Disk ->
  load into CPU cache, transfer the needed vectors to GPU, update the
  score, then compare against the lowest scores in BOTH caches to decide:
  promote to GPU / remain in CPU / stay on disk.
- "we ensure non-redundancy between the GPU and CPU caches. Additionally,
  we maintain all chunk replicas on disk, thus eliminating I/O latency when
  chunks are evicted from CPU to disk" — so CPU evictions are metadata-only
  drops (no write-back).

TokenCache (class name per paper §5) is the policy engine: it tracks each
chunk's cache tier ('gpu' / 'cpu' / 'disk' = not cached), scores, and the
two min-heaps. Actual tensor movement is delegated to an optional `on_move`
callback so the Stage-1/2 chunk stores can be wired in later; the disk
replica is permanent by construction (Stage-1 files are never deleted by
the cache).
"""

import heapq
import itertools

TIERS = ("gpu", "cpu")


class ChunkMeta:
    __slots__ = ("key", "size", "freq", "imp_ratio", "imp_seen",
                 "location", "stamp")

    def __init__(self, key, size):
        self.key = key
        self.size = size
        self.freq = 0.0        # decayed access counter (moving average)
        self.imp_ratio = 1.0   # moving average of observed important ratio
        self.imp_seen = False
        self.location = "disk"  # cache tier; the disk replica always exists
        self.stamp = 0          # bumped on every score/location change

    @property
    def score(self):
        """§4.4.2: score = access frequency x importance ratio."""
        return self.freq * self.imp_ratio


class TokenCache:
    """Two-tier (GPU/CPU) score-based chunk cache over a permanent disk
    replica, with one min-heap per tier (lazy deletion via stamps).

    Capacities and chunk sizes are in abstract units (e.g. vectors);
    `on_move(key, src, dst)` is called for every tier transition.
    """

    def __init__(self, gpu_capacity, cpu_capacity,
                 freq_decay=0.9, imp_beta=0.3, on_move=None):
        self.capacity = {"gpu": gpu_capacity, "cpu": cpu_capacity}
        self.used = {"gpu": 0, "cpu": 0}
        self.freq_decay = freq_decay
        self.imp_beta = imp_beta
        self.on_move = on_move
        self.chunks = {}
        self._heaps = {"gpu": [], "cpu": []}
        self._seq = itertools.count()
        self.stats = dict(accesses=0, gpu_hits=0, cpu_hits=0, disk_hits=0,
                          promotions=0, demotions=0, drops=0)

    # ------------------------------------------------------------ chunks
    def register(self, key, size=1):
        assert key not in self.chunks, f"duplicate chunk {key}"
        self.chunks[key] = ChunkMeta(key, size)
        return self.chunks[key]

    def seed_stats(self, key, freq=None, imp_ratio=None):
        """Directly set a chunk's statistics (tests / warm start)."""
        meta = self.chunks[key]
        if freq is not None:
            meta.freq = float(freq)
        if imp_ratio is not None:
            meta.imp_ratio = float(imp_ratio)
            meta.imp_seen = True
        self._touch(meta)

    def location(self, key):
        return self.chunks[key].location

    def score(self, key):
        return self.chunks[key].score

    # ------------------------------------------------------------- heaps
    def _touch(self, meta):
        """Bump stamp and (re-)push onto its tier's heap."""
        meta.stamp += 1
        if meta.location in TIERS:
            heapq.heappush(self._heaps[meta.location],
                           (meta.score, next(self._seq), meta.key, meta.stamp))

    def _peek_min(self, tier):
        heap = self._heaps[tier]
        while heap:
            score, _, key, stamp = heap[0]
            meta = self.chunks[key]
            if meta.location == tier and meta.stamp == stamp:
                return meta
            heapq.heappop(heap)  # stale entry
        return None

    # --------------------------------------------------------- placement
    def _emit(self, key, src, dst):
        if self.on_move is not None:
            self.on_move(key, src, dst)

    def _leave_tier(self, meta):
        assert meta.location in TIERS
        self.used[meta.location] -= meta.size
        meta.location = "disk"
        meta.stamp += 1  # invalidate heap entries

    def _enter_tier(self, meta, tier):
        self.used[tier] += meta.size
        meta.location = tier
        self._touch(meta)

    def _admit(self, meta, tier):
        """Place `meta` (currently in no cache tier) into `tier`, evicting
        strictly-lower-scored chunks as needed. GPU victims demote to CPU;
        CPU victims drop to disk-only (replica already there, §4.4.2).
        Returns True if placed."""
        assert meta.location == "disk"
        if meta.size > self.capacity[tier]:
            return False
        while self.used[tier] + meta.size > self.capacity[tier]:
            victim = self._peek_min(tier)
            if victim is None or victim.score >= meta.score:
                return False
            self._leave_tier(victim)
            if tier == "gpu":
                if self._admit(victim, "cpu"):
                    self.stats["demotions"] += 1
                    self._emit(victim.key, "gpu", "cpu")
                else:
                    self.stats["drops"] += 1
                    self._emit(victim.key, "gpu", "disk")
            else:
                self.stats["drops"] += 1
                self._emit(victim.key, "cpu", "disk")
        self._enter_tier(meta, tier)
        return True

    # ------------------------------------------------------------ access
    def _update_stats(self, meta, observed_important_ratio):
        # decayed access counter — a moving average of access frequency
        meta.freq = meta.freq * self.freq_decay + 1.0
        if observed_important_ratio is not None:
            r = float(observed_important_ratio)
            if meta.imp_seen:  # online moving average (§4.4.2)
                meta.imp_ratio = (self.imp_beta * r
                                  + (1.0 - self.imp_beta) * meta.imp_ratio)
            else:
                meta.imp_ratio = r
                meta.imp_seen = True

    def access(self, key, important_ratio=None, update_stats=True):
        """One request touching this chunk. Returns the tier the chunk was
        found in BEFORE any movement ('gpu' | 'cpu' | 'disk')."""
        meta = self.chunks[key]
        found = meta.location
        self.stats["accesses"] += 1
        self.stats[found + "_hits"] += 1

        if update_stats:
            self._update_stats(meta, important_ratio)

        if found == "gpu":
            # utilize vectors, update score, retain in GPU cache
            self._touch(meta)
        elif found == "cpu":
            # vectors go to GPU for compute; promote if score beats the
            # GPU cache's lowest
            self._leave_tier(meta)
            if self._admit(meta, "gpu"):
                self.stats["promotions"] += 1
                self._emit(key, "cpu", "gpu")
            else:
                ok = self._admit(meta, "cpu")  # slot was just freed
                assert ok
        else:
            # disk: load into CPU cache, send vectors to GPU, then decide
            # among GPU / CPU / disk against both heaps' minimums
            if self._admit(meta, "gpu"):
                self.stats["promotions"] += 1
                self._emit(key, "disk", "gpu")
            elif self._admit(meta, "cpu"):
                self._emit(key, "disk", "cpu")
            # else: stays disk-only
        return found

    def drop(self, key):
        """Force a chunk out of both cache tiers (metadata-only; the disk
        replica remains). Used when a chunk's on-disk content is rewritten
        (KV reordering) and cached copies become stale."""
        meta = self.chunks[key]
        if meta.location in TIERS:
            src = meta.location
            self._leave_tier(meta)
            self._emit(key, src, "disk")

    # ------------------------------------------------------------- misc
    def gpu_hit_ratio(self):
        a = self.stats["accesses"]
        return self.stats["gpu_hits"] / a if a else 0.0

    def check_invariants(self):
        """Non-redundancy, capacity accounting, heap-min correctness."""
        used = {"gpu": 0, "cpu": 0}
        for meta in self.chunks.values():
            assert meta.location in ("gpu", "cpu", "disk"), meta.location
            if meta.location in TIERS:
                used[meta.location] += meta.size
        for tier in TIERS:
            assert used[tier] == self.used[tier], (
                f"{tier} accounting: {used[tier]} != {self.used[tier]}")
            assert self.used[tier] <= self.capacity[tier]
            cached = [m for m in self.chunks.values() if m.location == tier]
            top = self._peek_min(tier)
            if cached:
                true_min = min(m.score for m in cached)
                assert top is not None and abs(top.score - true_min) < 1e-12
            else:
                assert top is None
        # non-redundancy is structural (single `location` per chunk); the
        # disk replica is permanent (the cache never deletes chunk files)
        return True

"""IMPRESS serving mode (Stage 8): orchestration of Stages 1-7.

Wires the verified components into the paper's §4.1 dataflow — no new
algorithms, only orchestration:

  1. request S = [prefix, query] arrives
  2. radix tree search -> longest common prefix R + new part NR   (Stage 2)
  3. similarity-guided ITF identifies R_important during prefill  (Stage 5);
     the chunks backing R are loaded through TokenCache tiers
     GPU / CPU / disk                                             (Stage 7),
     with real tensor movement at chunk granularity               (Stage 1
     chunk files are the permanent disk replica)
  4. prefill runs over R_important + NR + query                   (Stage 4/5)
  5. NR's freshly computed KV is stored on disk and inserted into
     the radix tree                                               (Stage 1/2)
  6. decoding stays on FlexGen's original code path
  7. KV reordering runs periodically in the background            (Stage 6);
     per-request selection results feed both the tree's average token
     importance (reorderer input) and each chunk's importance ratio
     (TokenCache score input)

Known simplification (inherited from Stage 4): the KV written back to
FlexGen's per-request cache covers only the new tokens, so serving uses
gen_len=1 (TTFT-oriented); multi-token decoding with reused prefixes is
future work.
"""

import os
import time
import threading

import numpy as np
import torch

from flexllmgen.impress.radix_tree import PrefixRadixTree
from flexllmgen.impress.reordering import KVReorderer
from flexllmgen.impress.token_cache import TokenCache
from flexllmgen.impress.similarity_guided import (SimGuidedConfig,
    SimGuidedController, install_sim_guided_hook, uninstall_sim_guided_hook)


class ImpressConfig:
    def __init__(self, retention_ratio=0.25, probe_head_count=3, alpha=0.6,
                 gpu_cache_capacity=8192, cpu_cache_capacity=16384,
                 reorder_every_n_requests=10, reorder_interval_seconds=None,
                 chunk_size=None, selective=True):
        self.retention_ratio = retention_ratio
        self.sim = SimGuidedConfig(probe_head_count=probe_head_count,
                                   alpha=alpha)
        self.chunk_size = chunk_size  # None -> Stage-1 default (64)
        # Stage 10: physically load only the chunks that contain selected
        # tokens (probe sidecar first, then selected chunks). False keeps
        # the Stage-8 behavior (assemble all of R, filter on GPU).
        self.selective = selective
        # capacities in vectors (= tokens; one vector is one token's K or V
        # across all heads, matching the paper's Figure 9 counting)
        self.gpu_cache_capacity = gpu_cache_capacity
        self.cpu_cache_capacity = cpu_cache_capacity
        self.reorder_every_n_requests = reorder_every_n_requests
        self.reorder_interval_seconds = reorder_interval_seconds


class ChunkKVManager:
    """Chunk-granular tiered data plane over the Stage-1 chunk files.

    TokenCache (Stage 7) decides each chunk's tier; this class owns the
    actual tensors: a GPU-resident dict, a CPU-resident dict, and the
    permanent disk replica (Stage-1 .npy chunk files). Assembles R's
    per-layer prefix K/V in original token order (applying §4.4.1 mapping
    lists) for the Stage-5 attention. Invalidates cached chunks when the
    background reorderer rewrites a node.
    """

    def __init__(self, tree, gpu_device, cfg, n_head=32):
        from flexllmgen.impress.selective_loading import PhysicalIOCounter
        self.tree = tree
        self.dev = gpu_device
        self.cfg = cfg
        self.n_head = n_head
        self.probe_heads = cfg.sim.probe_head_count
        self.cache = TokenCache(cfg.gpu_cache_capacity,
                                cfg.cpu_cache_capacity,
                                on_move=self._on_move)
        self._gpu = {}
        self._cpu = {}
        self._node_snapshot = {}   # node_id -> mapping tuple (or None)
        self._ratio = {}           # chunk key -> last observed imp ratio
        self.stats = dict(disk_chunk_loads=0, pcie_chunk_transfers=0,
                          gpu_chunk_hits=0)
        self.phys = PhysicalIOCounter()  # physical I/O events (Stage 10)

    @staticmethod
    def _key(node_id, layer_id, kind, idx):
        return (node_id, layer_id, kind, idx)

    # ------------------------------------------------------ registration
    def sync_node(self, node):
        """Register a node's chunks; if the node was reordered since the
        last sync (mapping changed), drop stale cached copies first."""
        snap = node.reorder_version
        if self._node_snapshot.get(node.id, "absent") == snap:
            return
        if node.id in self._node_snapshot:  # reordered: invalidate
            for key in [k for k in self.cache.chunks if k[0] == node.id]:
                self.cache.drop(key)
                self._gpu.pop(key, None)
                self._cpu.pop(key, None)
                self._ratio.pop(key, None)
        for j in range(self.tree.num_layers):
            meta = node.store.layer(j).meta
            for kind in ("k", "v"):
                for c in meta["chunks"]:
                    key = self._key(node.id, j, kind, c["idx"])
                    if key not in self.cache.chunks:
                        self.cache.register(key, size=c["num_tokens"])
            if getattr(self.cfg, "selective", False):
                # probe-key sidecar (paper §6.5: probe heads' keys stored
                # redundantly so identification never reads other heads)
                pkey = self._key(node.id, j, "pk", 0)
                if pkey not in self.cache.chunks:
                    size = max(1, len(node.tokens) * self.probe_heads
                               // self.n_head)
                    self.cache.register(pkey, size=size)
                self._ensure_sidecar(node, j)
        self._node_snapshot[node.id] = snap

    def register_path(self, match):
        with self.tree.lock:
            for node, _ in match.path:
                self.sync_node(node)

    def set_ratio(self, key, ratio):
        self._ratio[key] = ratio

    # -------------------------------------------------------- data plane
    def _on_move(self, key, src, dst):
        if src == "gpu" and dst == "cpu":
            t = self._gpu.pop(key, None)
            if t is not None:
                self._cpu[key] = t.to("cpu")
        elif src == "cpu" and dst == "gpu":
            t = self._cpu.pop(key, None)
            if t is not None:
                self._gpu[key] = t.to(self.dev)
        elif dst == "disk":
            self._gpu.pop(key, None)
            self._cpu.pop(key, None)
        # disk -> gpu/cpu: data is materialized by _chunk_tensor after the
        # policy decision

    def _chunk_tensor(self, node, layer_id, kind, cinfo):
        """One chunk's rows on GPU for compute, routed through the cache."""
        key = self._key(node.id, layer_id, kind, cinfo["idx"])
        found = self.cache.access(key, important_ratio=self._ratio.get(key))
        if found == "gpu":
            self.stats["gpu_chunk_hits"] += 1
            data = self._gpu[key]
        elif found == "cpu" and key in self._cpu:
            self.stats["pcie_chunk_transfers"] += 1
            data = self._cpu[key].to(self.dev)
        else:
            self.stats["disk_chunk_loads"] += 1
            layer = node.store.layer(layer_id)
            path = os.path.join(layer.dir, cinfo[kind + "_file"])
            data = torch.from_numpy(np.load(path)).to(self.dev)
        # settle residency according to the cache's placement decision
        loc = self.cache.location(key)
        if loc == "gpu":
            self._gpu[key] = data
            self._cpu.pop(key, None)
        elif loc == "cpu":
            if key not in self._cpu:
                self._cpu[key] = data.to("cpu")
            self._gpu.pop(key, None)
        return data

    # ------------------------------- selective loading (Stage 10, §4.3+§6.5)
    def _sidecar_path(self, node, layer_id):
        layer = node.store.layer(layer_id)
        return os.path.join(layer.dir, f"probe_{self.probe_heads}.k.npy")

    def _ensure_sidecar(self, node, layer_id):
        """Build the probe-key sidecar (stored token order, first P heads)
        from the node's K chunks if absent. Runs at registration / after a
        reorder rewrite — off the per-request critical path."""
        path = self._sidecar_path(node, layer_id)
        if os.path.exists(path):
            return path
        layer = node.store.layer(layer_id)
        parts = [np.load(os.path.join(layer.dir, c["k_file"]))
                 [:, :self.probe_heads, :] for c in layer.meta["chunks"]]
        np.save(path, np.concatenate(parts, axis=0))
        return path

    def _settle(self, key, data):
        """Place `data` per the cache's post-access location decision."""
        loc = self.cache.location(key)
        if loc == "gpu":
            self._gpu[key] = data
            self._cpu.pop(key, None)
        elif loc == "cpu":
            if key not in self._cpu:
                self._cpu[key] = data.to("cpu")
            self._gpu.pop(key, None)

    def probe_tensor(self, node, layer_id):
        """The node's probe-head keys (stored order), through cache tiers;
        disk path reads ONLY the sidecar file, never the full-head chunks."""
        key = self._key(node.id, layer_id, "pk", 0)
        found = self.cache.access(key)
        if found == "gpu":
            data = self._gpu[key]
            self.phys.record_gpu(data.numel() * data.element_size())
        elif found == "cpu" and key in self._cpu:
            data = self._cpu[key].to(self.dev)
            self.phys.record_pcie(data.numel() * data.element_size())
        else:
            path = self._ensure_sidecar(node, layer_id)
            t0 = time.perf_counter()
            arr = np.load(path)
            self.phys.record_disk(1, arr.nbytes, time.perf_counter() - t0,
                                  path)
            data = torch.from_numpy(arr).to(self.dev)
        self._settle(key, data)
        return data

    def chunk_pair(self, node, layer_id, cinfo):
        """One chunk's (k, v) rows on the compute device, through cache
        tiers. A disk miss reads via the Stage-1 load_chunk API (both kind
        files of that chunk); chunks never requested are never read."""
        keys = {kind: self._key(node.id, layer_id, kind, cinfo["idx"])
                for kind in ("k", "v")}
        found = {kind: self.cache.access(keys[kind],
                                         important_ratio=self._ratio.get(
                                             keys[kind]))
                 for kind in ("k", "v")}
        chunk_np = None
        if any(found[kind] == "disk" or
               (found[kind] == "cpu" and keys[kind] not in self._cpu)
               for kind in ("k", "v")):
            layer = node.store.layer(layer_id)
            t0 = time.perf_counter()
            k_np, v_np = layer.load_chunk(cinfo["idx"])  # Stage-1 API
            dt = time.perf_counter() - t0
            self.phys.record_disk(
                2, k_np.nbytes + v_np.nbytes, dt,
                os.path.join(layer.dir, cinfo["k_file"]),
                os.path.join(layer.dir, cinfo["v_file"]))
            chunk_np = {"k": k_np, "v": v_np}
        out = {}
        for kind in ("k", "v"):
            key = keys[kind]
            if found[kind] == "gpu":
                data = self._gpu[key]
                self.phys.record_gpu(data.numel() * data.element_size())
            elif found[kind] == "cpu" and key in self._cpu:
                data = self._cpu[key].to(self.dev)
                self.phys.record_pcie(data.numel() * data.element_size())
            else:
                data = torch.from_numpy(chunk_np[kind]).to(self.dev)
            self._settle(key, data)
            out[kind] = data
        return out["k"], out["v"]

    def load_prefix_kv(self, match):
        """Assemble R's per-layer (k, v) GPU tensors in ORIGINAL token
        order through the cache tiers. Returns {layer_id: (k, v)}."""
        out = {}
        with self.tree.lock:
            for node, _ in match.path:
                self.sync_node(node)
            for j in range(self.tree.num_layers):
                ks, vs = [], []
                for node, used in match.path:
                    meta = node.store.layer(j).meta
                    k_node = torch.cat(
                        [self._chunk_tensor(node, j, "k", c)
                         for c in meta["chunks"]], dim=0)
                    v_node = torch.cat(
                        [self._chunk_tensor(node, j, "v", c)
                         for c in meta["chunks"]], dim=0)
                    mapping = node.layer_mapping(j)
                    if mapping is not None:
                        idx = torch.tensor(mapping, dtype=torch.long,
                                           device=self.dev)
                        k_node, v_node = k_node[idx], v_node[idx]
                    ks.append(k_node[:used])
                    vs.append(v_node[:used])
                out[j] = (torch.cat(ks, dim=0), torch.cat(vs, dim=0))
        return out


class _ServingReorderer(KVReorderer):
    """KVReorderer that resyncs the chunk manager (cache invalidation +
    probe-sidecar rebuild) for rewritten nodes, still off the serving path."""

    def __init__(self, tree, manager, **kw):
        super().__init__(tree, **kw)
        self.manager = manager

    def reorder_all(self):
        done = super().reorder_all()
        if done:
            with self.tree.lock:
                for nid in done:
                    self.manager.sync_node(self.tree._nodes[nid])
        return done


class ImpressServer:
    """One request = §4.1 dataflow steps 1-5 (+ background step 7)."""

    def __init__(self, model, opt_config, tree_dir, impress_cfg=None,
                 gpu_device=None):
        from flexllmgen.impress.selective_loading import (
            install_selective_hook, uninstall_selective_hook)
        self.model = model
        self.opt_config = opt_config
        self.cfg = impress_cfg or ImpressConfig()
        if self.cfg.chunk_size is not None:
            self.tree = PrefixRadixTree(tree_dir,
                                        chunk_size=self.cfg.chunk_size)
        else:
            self.tree = PrefixRadixTree(tree_dir)
        self.dev = gpu_device or torch.device("cuda:0")
        self.manager = ChunkKVManager(self.tree, self.dev, self.cfg,
                                      n_head=opt_config.n_head)
        if self.cfg.selective:
            self.controller = install_selective_hook(self.cfg.sim)
            self._uninstall = uninstall_selective_hook
        else:
            self.controller = install_sim_guided_hook(self.cfg.sim)
            self._uninstall = uninstall_sim_guided_hook
        self.controller.retention_ratio = self.cfg.retention_ratio
        self.controller.capture_logits = True
        self.reorderer = _ServingReorderer(
            self.tree, self.manager,
            interval_seconds=self.cfg.reorder_interval_seconds,
            every_n_requests=self.cfg.reorder_every_n_requests)
        self.reorderer.start()

    def close(self):
        self.reorderer.stop()
        with self.tree.lock:
            self.tree._save()  # flush deferred importance updates
        self._uninstall()

    # -------------------------------------------------------------- serve
    def request(self, prefix_ids, query_ids):
        """Serve one request; returns (last_token_logits, ttft_seconds,
        num_reused_tokens R, num_new_prefix_tokens NR)."""
        ctrl = self.controller
        t0 = time.perf_counter()

        # (2) longest common prefix R + new part NR
        m = self.tree.match(prefix_ids)
        NR = m.NR
        new_tokens = list(NR) + list(query_ids)

        ctrl.selected_by_layer = {}
        ctrl.token_importance = {}
        ctrl.new_kv = {}
        ctrl.capture_new_kv = bool(NR)

        if m.r_len > 0:
            if self.cfg.selective:
                # (3) Stage 10: hand the attention a chunk-granular provider;
                # ITF runs first per layer and only chunks containing
                # selected tokens are physically loaded
                from flexllmgen.impress.selective_loading import PathKVProvider
                ctrl.provider = PathKVProvider(self.manager, m)
            else:
                # Stage-8 path: assemble all of R up front, filter on GPU
                ctrl.prefix_kv = self.manager.load_prefix_kv(m)
            ctrl.prefix_len = m.r_len
            ctrl.enabled = True
        else:
            ctrl.enabled = False  # nothing to reuse: plain FlexGen prefill

        # (4) prefill over R_important + NR + query; (6) decode unchanged
        self.model.generate((new_tokens,), max_new_tokens=1, do_sample=False)
        logits = ctrl.last_logits[0]
        ttft = time.perf_counter() - t0

        # ---- post-response bookkeeping (off the TTFT path) ----
        # (5) store NR's KV on disk + insert into the radix tree
        if NR:
            nr_len = len(NR)
            nr_kv = {j: (k[:nr_len], v[:nr_len])
                     for j, (k, v) in ctrl.new_kv.items()}
            self.tree.insert(prefix_ids, nr_kv)
            self.manager.register_path(self.tree.match(prefix_ids))
        # feed selection results to the reorderer (avg token importance,
        # §4.4.1) and to chunk importance ratios (§4.4.2)
        if m.r_len > 0 and ctrl.selected_by_layer:
            self._record_selection(m)
        ctrl.capture_new_kv = False
        ctrl.enabled = False
        if getattr(ctrl, "provider", None) is not None:
            ctrl.provider = None
        # (7) background reordering trigger
        self.reorderer.notify_request()
        return logits, ttft, m.r_len, len(NR)

    # ------------------------------------------------------ importance
    def _record_selection(self, match):
        """Convert per-layer ITF selections into (a) per-token importance
        for the tree/reorderer and (b) per-chunk importance ratios for
        TokenCache scores."""
        n_head = self.opt_config.n_head
        r_len = match.r_len
        num_layers = len(self.controller.selected_by_layer)
        per_layer = {}
        for layer_id, (mode, data) in self.controller.selected_by_layer.items():
            s = np.zeros(r_len)
            if mode == "probe":
                s[data.numpy()] = 1.0
            else:  # per-head fallback: fraction of heads selecting the token
                idx, cnt = np.unique(data.numpy().ravel(), return_counts=True)
                s[idx] = cnt / n_head
            per_layer[layer_id] = s

        # §4.4.1 reorders "based on the average token importance": prefer the
        # raw H2O importance values (normalized per layer, then averaged) —
        # far more discriminative than binary selected/not-selected counts,
        # which wash out at high retention ratios.
        ti = getattr(self.controller, "token_importance", None)
        by_layer = None
        if ti:
            token_score = np.zeros(r_len)
            by_layer = {}
            for j, vec in ti.items():
                v = vec.numpy().astype(np.float64)
                v = v / max(v.sum(), 1e-9)
                by_layer[j] = v.tolist()
                token_score += v
            token_score /= len(ti)
        else:
            token_score = sum(per_layer.values()) / max(num_layers, 1)
        self.tree.record_importance(match, token_score.tolist(),
                                    by_layer=by_layer,
                                    persist=False)

        cs = self.tree.chunk_size
        with self.tree.lock:
            for layer_id, s in per_layer.items():
                off = 0
                for node, used in match.path:
                    n_node = len(node.tokens)
                    seg = np.zeros(n_node)
                    seg[:used] = s[off:off + used]
                    off += used
                    mapping = (node.layer_mapping(layer_id)
                               or list(range(n_node)))
                    sums = {}
                    counts = {}
                    for i in range(n_node):
                        c = mapping[i] // cs
                        sums[c] = sums.get(c, 0.0) + seg[i]
                        counts[c] = counts.get(c, 0) + 1
                    for c, total in sums.items():
                        ratio = total / counts[c]
                        for kind in ("k", "v"):
                            self.manager.set_ratio(
                                (node.id, layer_id, kind, c), ratio)

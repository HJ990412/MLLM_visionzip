"""Radix-tree prefix metadata management (IMPRESS Stage 2).

Paper grounding (re-checked against the FAST'25 paper):
- §4.1 (Dataflow): metadata in CPU memory is organized as a radix tree. For a
  new request, search the longest common prefix subsequence R among previous
  requests; the remaining prefix tokens NR have no stored KV. After prefill,
  the newly generated KVs for NR are stored on disk and NR's tokens are
  inserted into the radix tree for future reuse.
- §4.4.1 / Figure 13: each node groups a token subsequence (e.g. the common
  subsequence s0 = [t0,t1,t2,t3] lives in one node) and holds pointer lists
  k_ptr / v_ptr to the key/value chunks of those tokens. Chunks are packed
  per node, and (in a later stage) KV reordering is limited to tokens within
  each node.

Storage: each node's KV lives in a Stage-1 PrefixKVStore rooted at the node's
own directory (root/nodes/node_XXXX/layer_YY/chunk_*.npy), so the chunk file
format is unchanged and loading R's KV is a sequence of Stage-1 loads along
the matched node path. Chunk boundaries are relative to the node's first
token; splitting a node re-packs both halves into fresh chunks.

No importance / reordering yet: PrefixKVLayer's token_order stays None
(identity).
"""

import json
import os
import shutil
import threading

import torch

from flexllmgen.impress.prefix_kv import PrefixKVStore, DEFAULT_CHUNK_SIZE

TREE_META_NAME = "tree.json"


class MatchResult:
    """Result of matching a token sequence against the tree.

    R  = tokens[:r_len]  — longest common prefix subsequence, KV on disk.
    NR = tokens[r_len:]  — must be computed by this request and inserted.
    path — list of (node, used) pairs from the root downward; `used` is the
    number of tokens consumed from that node (== len(node.tokens) except
    possibly for the last entry, which may be a partial in-node match).
    """

    def __init__(self, tokens, r_len, path):
        self.tokens = list(tokens)
        self.r_len = r_len
        self.path = path

    @property
    def R(self):
        return self.tokens[:self.r_len]

    @property
    def NR(self):
        return self.tokens[self.r_len:]

    def __repr__(self):
        return f"MatchResult(R={self.R}, NR={self.NR})"


class RadixNode:
    """One radix-tree node: a token subsequence + its chunked KV on disk."""

    def __init__(self, node_id, tokens, dir_path, chunk_size):
        self.id = node_id
        self.tokens = [int(t) for t in tokens]  # ORIGINAL token order
        self.dir = dir_path  # None only for the root (which stores no KV)
        self.chunk_size = chunk_size
        self.children = {}  # first token of child's segment -> RadixNode
        self._store = None
        # §4.4.1 mapping list m: on-disk KV is stored in reordered form and
        # stored[m] recovers the original order (paper Figure 13's s0'[m0]).
        # None = identity (not reordered). This is the node-level summary
        # mapping; the AUTHORITATIVE per-layer mapping lives in each layer's
        # metadata (token_order) — paper §5: "mapping lists per layer".
        self.mapping = None
        # Running average importance per token (ORIGINAL order), fed by
        # record_importance(); consumed by the KV reorderer. imp_by_layer
        # holds per-layer averages when the caller provides them, enabling
        # per-layer reordering.
        self.importance = None
        self.imp_by_layer = None  # {layer_id: [float] * len(tokens)}
        self.imp_count = 0
        # bumped whenever on-disk chunk content is rewritten (reorder/split)
        self.reorder_version = 0

    def layer_mapping(self, layer_id):
        """The layer's token_order mapping list (stored'[m] = original),
        or None for identity. Falls back to the node-level mapping for
        stores written before per-layer reordering."""
        meta = self.store.layer(layer_id).meta
        if meta is not None and meta.get("token_order") is not None:
            return meta["token_order"]
        return self.mapping

    @property
    def store(self):
        """Stage-1 PrefixKVStore holding this node's KV chunks."""
        assert self.dir is not None, "root node stores no KV"
        if self._store is None:
            self._store = PrefixKVStore(self.dir, self.chunk_size)
        return self._store

    def _ptr(self, layer_id, which):
        layer = self.store.layer(layer_id)
        assert layer.meta is not None, (
            f"node {self.id} layer {layer_id} has no stored KV")
        return [os.path.join(layer.dir, c[which]) for c in layer.meta["chunks"]]

    def k_ptr(self, layer_id):
        """Pointer list to this node's key chunks (paper Figure 13)."""
        return self._ptr(layer_id, "k_file")

    def v_ptr(self, layer_id):
        """Pointer list to this node's value chunks (paper Figure 13)."""
        return self._ptr(layer_id, "v_file")

    def invalidate_store(self):
        self._store = None

    def __repr__(self):
        return (f"RadixNode(id={self.id}, tokens={self.tokens}, "
                f"children={sorted(self.children)})")


class PrefixRadixTree:
    """Radix tree over stored prefixes, persisted under `root_dir`.

    Layout:
        <root_dir>/tree.json                 # tree structure + config
        <root_dir>/nodes/node_0001/...       # per-node Stage-1 KV store
    """

    def __init__(self, root_dir, chunk_size=DEFAULT_CHUNK_SIZE):
        self.root_dir = os.path.abspath(os.path.expanduser(root_dir))
        self.nodes_dir = os.path.join(self.root_dir, "nodes")
        self.chunk_size = chunk_size
        self.num_layers = None
        self.next_id = 1
        self.root = RadixNode(0, [], None, chunk_size)
        self._nodes = {0: self.root}
        # Guards tree structure + node KV rewrites; the async reorderer
        # (reordering.py) takes it per node so serving is barely blocked.
        self.lock = threading.RLock()
        if os.path.exists(os.path.join(self.root_dir, TREE_META_NAME)):
            self._load()

    # ------------------------------------------------------------ search
    def match(self, tokens):
        """Find the longest common prefix subsequence R (paper §4.1)."""
        with self.lock:
            tokens = [int(t) for t in tokens]
            node = self.root
            pos = 0
            path = []
            while pos < len(tokens):
                child = node.children.get(tokens[pos])
                if child is None:
                    break
                seg = child.tokens
                n = 0
                limit = min(len(seg), len(tokens) - pos)
                while n < limit and seg[n] == tokens[pos + n]:
                    n += 1
                path.append((child, n))
                pos += n
                if n < len(seg):  # diverged (or query ended) inside this node
                    break
                node = child
            return MatchResult(tokens, pos, path)

    # ------------------------------------------------------------ insert
    def insert(self, tokens, nr_kv=None):
        """Insert a request's prefix. `nr_kv` holds the freshly computed KV
        for the NR part only: {layer_id: (k, v)} with k/v shaped
        (len(NR), ...). Returns the MatchResult computed before insertion.
        """
        with self.lock:
            m = self.match(tokens)
            if not m.NR:
                return m  # fully covered by stored prefixes; nothing to insert

            assert nr_kv, f"NR={m.NR} is non-empty; nr_kv is required"
            if self.num_layers is None:
                self.num_layers = len(nr_kv)
            assert len(nr_kv) == self.num_layers, (
                f"nr_kv has {len(nr_kv)} layers, tree has {self.num_layers}")

            # If the match ended inside a node, split it at the divergence
            # point so the shared part becomes a common ancestor
            # (paper Figure 13(a)).
            if m.path and m.path[-1][1] < len(m.path[-1][0].tokens):
                node, used = m.path[-1]
                self._split(node, used)
                parent = node
            else:
                parent = m.path[-1][0] if m.path else self.root

            new_node = self._new_node(m.NR)
            for j in range(self.num_layers):
                k_t, v_t = nr_kv[j]
                assert k_t.shape[0] == len(m.NR), (
                    f"layer {j}: nr_kv holds {k_t.shape[0]} tokens, "
                    f"NR has {len(m.NR)}")
                new_node.store.store_layer(j, k_t, v_t)
            parent.children[new_node.tokens[0]] = new_node

            self._save()
            return m

    def _new_node(self, tokens):
        node_id = self.next_id
        self.next_id += 1
        node_dir = os.path.join(self.nodes_dir, f"node_{node_id:04d}")
        node = RadixNode(node_id, tokens, node_dir, self.chunk_size)
        self._nodes[node_id] = node
        return node

    def _split(self, node, split_pos):
        """Split `node` so it keeps tokens[:split_pos]; a new child takes
        tokens[split_pos:] together with the trailing KV. Both halves are
        re-chunked relative to their new first token. If the node was
        reordered (§4.4.1 mapping set), the KV is restored to original
        order first and both halves start unreordered (mapping = None)."""
        assert 0 < split_pos < len(node.tokens)
        rest = node.tokens[split_pos:]
        new_child = self._new_node(rest)

        first_half = {}
        for j in range(self.num_layers):
            k_t, v_t = node.store.load_layer(j)
            assert k_t.shape[0] == len(node.tokens)
            mapping = node.layer_mapping(j)
            if mapping is not None:  # restore original order (s'[m])
                idx = torch.tensor(mapping, dtype=torch.long)
                k_t, v_t = k_t[idx], v_t[idx]
            new_child.store.store_layer(j, k_t[split_pos:], v_t[split_pos:])
            first_half[j] = (k_t[:split_pos], v_t[:split_pos])

        # Rewrite this node's storage with only the first half.
        shutil.rmtree(node.dir)
        node.invalidate_store()
        for j in range(self.num_layers):
            node.store.store_layer(j, *first_half[j])

        new_child.children = node.children
        node.children = {rest[0]: new_child}
        node.tokens = node.tokens[:split_pos]
        node.mapping = None
        node.reorder_version += 1
        if node.importance is not None:
            new_child.importance = node.importance[split_pos:]
            new_child.imp_count = node.imp_count
            node.importance = node.importance[:split_pos]
        if node.imp_by_layer is not None:
            new_child.imp_by_layer = {
                j: v[split_pos:] for j, v in node.imp_by_layer.items()}
            node.imp_by_layer = {
                j: v[:split_pos] for j, v in node.imp_by_layer.items()}

    # -------------------------------------------------------------- load
    def load_prefix_kv(self, match, device=None):
        """Load R's KV along the matched path using the Stage-1 loader.

        Returns {layer_id: (K, V)} with K/V shaped (len(R), ...), or {} if
        nothing matched. Note: a MatchResult becomes stale after inserts
        that split nodes — re-match before loading in that case.
        """
        with self.lock:
            if match.r_len == 0:
                return {}
            assert self.num_layers is not None
            out = {}
            for j in range(self.num_layers):
                ks, vs = [], []
                for node, used in match.path:
                    k_t, v_t = node.store.load_layer(j)
                    mapping = node.layer_mapping(j)
                    if mapping is not None:
                        # recover original order: stored'[m] (Figure 13's
                        # vectorized torch index operation)
                        idx = torch.tensor(mapping, dtype=torch.long)
                        k_t, v_t = k_t[idx], v_t[idx]
                    ks.append(k_t[:used])
                    vs.append(v_t[:used])
                k_full = torch.cat(ks, dim=0)
                v_full = torch.cat(vs, dim=0)
                if device is not None:
                    k_full = k_full.to(device)
                    v_full = v_full.to(device)
                out[j] = (k_full, v_full)
            return out

    # -------------------------------------------------------- importance
    def record_importance(self, match, scores, by_layer=None, persist=True):
        """Update per-node running-average token importance from one request.

        scores: sequence of len(match.R) importance values in ORIGINAL token
        order (e.g. Stage-3/5 per-token scores averaged over heads). Feeds
        the §4.4.1 reorderer ("based on the average token importance").
        by_layer: optional {layer_id: scores} with per-layer importance —
        enables per-layer reordering (paper §5: mapping lists per layer).
        persist=False skips the tree.json write (the serving path calls this
        per request; with many large nodes the JSON dump is expensive, so
        persistence is deferred to reorder passes / server close).
        """
        assert len(scores) == match.r_len, (len(scores), match.r_len)
        with self.lock:
            off = 0
            for node, used in match.path:
                n_node = len(node.tokens)
                if node.importance is None:
                    node.importance = [0.0] * n_node
                    node.imp_count = 0
                c = node.imp_count
                for i in range(used):
                    node.importance[i] = (
                        (node.importance[i] * c + float(scores[off + i]))
                        / (c + 1))
                if by_layer:
                    if node.imp_by_layer is None:
                        node.imp_by_layer = {}
                    for j, ls in by_layer.items():
                        arr = node.imp_by_layer.setdefault(
                            j, [0.0] * n_node)
                        for i in range(used):
                            arr[i] = ((arr[i] * c + float(ls[off + i]))
                                      / (c + 1))
                node.imp_count = c + 1
                off += used
            if persist:
                self._save()

    # ------------------------------------------------------- persistence
    def _save(self):
        os.makedirs(self.root_dir, exist_ok=True)
        nodes = []
        for node in self._nodes.values():
            nodes.append({
                "id": node.id,
                "tokens": node.tokens,
                "dir": (os.path.relpath(node.dir, self.root_dir)
                        if node.dir else None),
                "children": [c.id for c in node.children.values()],
                "mapping": node.mapping,
                "importance": node.importance,
                "imp_by_layer": ({str(j): v for j, v in
                                  node.imp_by_layer.items()}
                                 if node.imp_by_layer is not None else None),
                "imp_count": node.imp_count,
                "reorder_version": node.reorder_version,
            })
        meta = {
            "chunk_size": self.chunk_size,
            "num_layers": self.num_layers,
            "next_id": self.next_id,
            "nodes": nodes,
        }
        with open(os.path.join(self.root_dir, TREE_META_NAME), "w") as f:
            json.dump(meta, f, indent=1)

    def _load(self):
        with open(os.path.join(self.root_dir, TREE_META_NAME)) as f:
            meta = json.load(f)
        self.chunk_size = meta["chunk_size"]
        self.num_layers = meta["num_layers"]
        self.next_id = meta["next_id"]
        self._nodes = {}
        children_ids = {}
        for spec in meta["nodes"]:
            node_dir = (os.path.join(self.root_dir, spec["dir"])
                        if spec["dir"] else None)
            node = RadixNode(spec["id"], spec["tokens"], node_dir,
                             self.chunk_size)
            node.mapping = spec.get("mapping")
            node.importance = spec.get("importance")
            ibl = spec.get("imp_by_layer")
            node.imp_by_layer = ({int(j): v for j, v in ibl.items()}
                                 if ibl is not None else None)
            node.imp_count = spec.get("imp_count", 0)
            node.reorder_version = spec.get("reorder_version", 0)
            self._nodes[node.id] = node
            children_ids[node.id] = spec["children"]
        for node_id, child_ids in children_ids.items():
            node = self._nodes[node_id]
            for cid in child_ids:
                child = self._nodes[cid]
                node.children[child.tokens[0]] = child
        self.root = self._nodes[0]

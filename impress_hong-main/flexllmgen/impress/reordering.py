"""KV Reordering (IMPRESS Stage 6, paper §4.4.1).

Paper (re-checked): "We present the KV reordering method to address the
unnecessary loading of unimportant KVs during important KV retrieval. By
periodically reorganizing and repacking important KVs into denser chunks,
this approach optimizes read efficiency ... Scheduled at regular intervals
(e.g., every 10 minutes), this process is based on the average token
importance and operates asynchronously to avoid disrupting the main I/O
flow."

Rules implemented exactly as in the paper:
- Tokens inside a node are reordered in DESCENDING order of (average)
  importance; ties keep the original order (stable sort). This reproduces
  Figure 13(b): s0 = [t0,t1,t2,t3] with {t0,t3} important becomes
  s0' = [t0,t3,t1,t2] with mapping m0 = [0,2,3,1], and s2 = [t8..t11] with
  {t9} important becomes s2' = [t9,t8,t10,t11] with m2 = [1,0,2,3].
- The mapping list m satisfies stored'[m] = original ("the original s0
  sequence to be recovered using the torch index operation s0'[m0]").
- Reordering NEVER crosses node boundaries ("We explicitly avoid cross-node
  reordering ... it would destroy the radix tree structure ... and would
  result in unnecessary read bandwidth consumption").
- Chunks are repacked after reordering (the node's Stage-1 store is
  rewritten, so chunk boundaries realign to the new order).

KVReorderer runs the pass in a daemon thread, triggered by a time interval
and/or every N requests (both configurable); each node is rewritten under
the tree lock, released between nodes, so the serving path is not blocked
for the whole pass.
"""

import threading

import torch

from flexllmgen.impress.radix_tree import PrefixRadixTree


def compute_reorder(importance):
    """Stable descending-importance order for one node.

    Returns (perm, mapping):
      perm    — stored' = original[perm] (reordered sequence construction)
      mapping — m with stored'[m] = original (paper's mapping list)
    """
    n = len(importance)
    perm = sorted(range(n), key=lambda i: (-float(importance[i]), i))
    mapping = [0] * n
    for new_pos, orig_pos in enumerate(perm):
        mapping[orig_pos] = new_pos
    return perm, mapping


def reorder_node(tree, node, save=True):
    """Reorder one node's tokens by average importance and repack its
    chunks. Within-node only (never crosses node boundaries).

    When per-layer importance is available (node.imp_by_layer, fed by the
    serving path), each layer gets its own order and mapping list — paper
    §5: PrefixKVLayer stores "reordered KVs and mapping lists per layer".
    Otherwise all layers share the node-level order (Stage-6 behavior).
    Returns True if the node was rewritten."""
    with tree.lock:
        n = len(node.tokens)
        if node.importance is None or n < 2:
            return False
        base_perm, base_mapping = compute_reorder(node.importance)
        per_layer = node.imp_by_layer or {}
        if (base_perm == list(range(n)) and node.mapping is None
                and not per_layer):
            return False  # already in the right order; nothing to repack

        assert tree.num_layers is not None
        # compute the target order per layer FIRST and skip the (expensive)
        # rewrite when nothing actually changes — the running-average
        # importance moves a little on every request, but the resulting
        # ORDER is usually stable after the first reorder, and rewriting
        # gigabytes per node under the tree lock would stall serving
        identity = list(range(n))
        new_orders = {}
        changed = False
        for j in range(tree.num_layers):
            if j in per_layer:
                perm_j, mapping_j = compute_reorder(per_layer[j])
            else:
                perm_j, mapping_j = base_perm, base_mapping
            new_orders[j] = (perm_j, mapping_j)
            cur = node.layer_mapping(j)
            if mapping_j != (cur if cur is not None else identity):
                changed = True
        if not changed:
            return False

        staged = {}
        for j in range(tree.num_layers):
            k_t, v_t = node.store.load_layer(j)
            old = node.layer_mapping(j)
            if old is not None:          # restore original order first
                idx = torch.tensor(old, dtype=torch.long)
                k_t, v_t = k_t[idx], v_t[idx]
            perm_j, mapping_j = new_orders[j]
            perm_t = torch.tensor(perm_j, dtype=torch.long)
            staged[j] = (k_t[perm_t], v_t[perm_t], mapping_j)

        import shutil
        shutil.rmtree(node.dir)
        node.invalidate_store()
        for j in range(tree.num_layers):
            k_t, v_t, mapping_j = staged[j]
            node.store.store_layer(j, k_t, v_t, token_order=mapping_j)
        node.mapping = base_mapping
        node.reorder_version += 1
        if save:
            tree._save()
        return True


def reordered_tokens(node):
    """The node's stored (reordered) token sequence s' (Figure 13(b))."""
    if node.mapping is None:
        return list(node.tokens)
    s = [None] * len(node.tokens)
    for orig_pos, new_pos in enumerate(node.mapping):
        s[new_pos] = node.tokens[orig_pos]
    return s


def chunks_needed(node, layer_id, important_orig_positions):
    """How many of this node's chunks must be read to fetch the K (or V)
    vectors of the given important tokens (original-order positions).
    Used to measure the Figure 20-style read reduction."""
    layer = node.store.layer(layer_id)
    assert layer.meta is not None
    chunk_size = layer.meta["chunk_size"]
    mapping = node.layer_mapping(layer_id)
    stored_pos = (lambda i: mapping[i]) if mapping is not None \
        else (lambda i: i)
    return len({stored_pos(i) // chunk_size for i in important_orig_positions})


class KVReorderer:
    """Asynchronous periodic KV reordering over all radix-tree nodes.

    Triggers (§4.4.1 "scheduled at regular intervals"):
      interval_seconds — wall-clock period (paper example: every 10 min);
      every_n_requests — additionally kick a pass after every N calls to
                         notify_request() (request-count trigger).
    Either may be None. The pass runs in a daemon thread; the tree lock is
    taken per node, so serving stays responsive during a pass.
    """

    def __init__(self, tree, interval_seconds=None, every_n_requests=None):
        assert isinstance(tree, PrefixRadixTree)
        self.tree = tree
        self.interval_seconds = interval_seconds
        self.every_n_requests = every_n_requests
        self._request_count = 0
        self._count_lock = threading.Lock()
        self._kick = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self.passes = 0            # completed reorder passes
        self.nodes_reordered = 0   # total nodes rewritten

    # ------------------------------------------------------------ control
    def start(self):
        assert self._thread is None, "already started"
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="impress-kv-reorderer")
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._kick.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def notify_request(self):
        """Called (cheaply) by the serving path once per request; kicks an
        async pass every N requests. Never blocks on reordering."""
        if self.every_n_requests is None:
            return
        with self._count_lock:
            self._request_count += 1
            fire = self._request_count % self.every_n_requests == 0
        if fire:
            self._kick.set()

    # -------------------------------------------------------------- pass
    def reorder_all(self):
        """One synchronous pass over all nodes. Returns rewritten node ids."""
        with self.tree.lock:
            nodes = [n for n in self.tree._nodes.values()
                     if n.id != 0 and n.importance is not None]
        done = []
        for node in nodes:  # lock is taken/released per node inside
            if reorder_node(self.tree, node, save=False):
                done.append(node.id)
        if done:
            with self.tree.lock:
                self.tree._save()  # one JSON dump per pass, not per node
        self.passes += 1
        self.nodes_reordered += len(done)
        return done

    def _loop(self):
        while not self._stop.is_set():
            self._kick.wait(timeout=self.interval_seconds)
            if self._stop.is_set():
                break
            self._kick.clear()
            self.reorder_all()

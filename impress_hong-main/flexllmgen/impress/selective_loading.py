"""Selective chunk loading (Stage 10): read only the chunks that matter.

Closes the gap the paper cares most about (§2.2 Figure 3, §3.2 Challenge 1):
until now ITF correctly *identified* important tokens, but the data plane
still loaded every chunk of R and filtered afterwards, so the measured wins
came only from reduced GPU compute and the §4.4.2 cache. This module
restructures the per-layer flow so identification happens FIRST and only
the needed chunks are physically read:

  1. probe phase — load ONLY the probe heads' keys. Physically this reads
     the probe-key sidecar file (paper §6.5: probe keys are stored
     redundantly, ~1.2% extra space, precisely so identification never
     touches the other heads' data), routed through TokenCache tiers.
  2. threshold passed — the unified selected token indices are translated
     through the §4.4.1 reorder mapping into STORED chunk ids; only those
     chunks' K/V are loaded (GPU hit / CPU -> PCIe / disk via the Stage-1
     PrefixKVLayer.load_chunk API). Chunks with no selected token are
     never opened.
  3. fallback (per-layer, §4.3 step 6) — per-head identification needs all
     heads' keys, so that layer loads all of R's chunks (a naturally wider
     chunk range).

Note (paper Figure 12): chunk skipping only pays off once KV reordering has
packed important tokens densely — with scattered importance nearly every
64-token chunk contains at least one selected token. Experiments therefore
measure after the background reorderer has run.

PhysicalIOCounter is separate from Stage-5's algorithmic IOCounter (which
keeps counting theoretical vectors): it records actual events — chunk files
opened, bytes moved per tier, disk wall-clock time.
"""

import time

import torch
import torch.nn.functional as F

from flexllmgen.pytorch_backend import TorchDevice, TorchTensor, DeviceType
from flexllmgen.flex_opt import SelfAttention, OptLM
from flexllmgen.impress.importance import column_sum_importance
from flexllmgen.impress import prefix_reuse
from flexllmgen.impress.similarity_guided import (
    SimGuidedController, similarity_threshold, count_vectors_naive,
    count_vectors_probe, _capture_new_kv)


class PhysicalIOCounter:
    """Actual I/O events, as opposed to IOCounter's algorithmic vectors."""

    def __init__(self, track_files=True):
        self.track_files = track_files
        self.reset()

    def reset(self):
        self.files_opened = 0
        self.disk_bytes = 0
        self.disk_seconds = 0.0
        self.pcie_bytes = 0
        self.gpu_bytes = 0
        self.opened_files = []

    def record_disk(self, n_files, nbytes, seconds, *paths):
        self.files_opened += n_files
        self.disk_bytes += nbytes
        self.disk_seconds += seconds
        if self.track_files:
            self.opened_files.extend(paths)

    def record_pcie(self, nbytes):
        self.pcie_bytes += nbytes

    def record_gpu(self, nbytes):
        self.gpu_bytes += nbytes

    def summary(self):
        return dict(files=self.files_opened, disk_mb=self.disk_bytes / 1e6,
                    disk_ms=self.disk_seconds * 1e3,
                    pcie_mb=self.pcie_bytes / 1e6,
                    gpu_mb=self.gpu_bytes / 1e6)


class PathKVProvider:
    """Chunk-granular view of R's KV along one matched radix-tree path.

    All loads go through the manager's TokenCache tiers; disk reads use the
    Stage-1 load_chunk API (per-chunk files) and the probe-key sidecar.
    Returned tensors are in ORIGINAL token order (reorder mapping applied).
    """

    def __init__(self, manager, match):
        self.mgr = manager
        self.match = match
        self.num_tokens = match.r_len
        manager.register_path(match)  # sync/invalidate + sidecars
        self.entries = []
        off = 0
        for node, used in match.path:
            self.entries.append((node, used, off))
            off += used

    def probe_keys(self, layer_id):
        """(m, P, head_dim) on the compute device, original order."""
        with self.mgr.tree.lock:
            parts = []
            for node, used, _ in self.entries:
                t = self.mgr.probe_tensor(node, layer_id)  # stored order
                mapping = node.layer_mapping(layer_id)
                if mapping is not None:
                    idx = torch.tensor(mapping, dtype=torch.long,
                                       device=t.device)
                    t = t[idx]
                parts.append(t[:used])
            return torch.cat(parts, dim=0)

    def load_full(self, layer_id):
        """All of R's (k, v): (m, n_head, head_dim) each. Used by fallback
        layers and by retention_ratio >= 1 (FullLoad)."""
        with self.mgr.tree.lock:
            ks, vs = [], []
            for node, used, _ in self.entries:
                meta = node.store.layer(layer_id).meta
                pairs = [self.mgr.chunk_pair(node, layer_id, c)
                         for c in meta["chunks"]]
                k_node = torch.cat([p[0] for p in pairs], dim=0)
                v_node = torch.cat([p[1] for p in pairs], dim=0)
                mapping = node.layer_mapping(layer_id)
                if mapping is not None:
                    idx = torch.tensor(mapping, dtype=torch.long,
                                       device=k_node.device)
                    k_node, v_node = k_node[idx], v_node[idx]
                ks.append(k_node[:used])
                vs.append(v_node[:used])
            return torch.cat(ks, dim=0), torch.cat(vs, dim=0)

    def load_selected(self, layer_id, sel_indices):
        """K/V rows of the selected ORIGINAL-order indices, loading only
        the chunks that contain them: (k_keep, n_head, head_dim) each.
        Row order follows (node, stored-position) order — a set, which is
        exactly what re-normalized attention consumes."""
        sel = sorted(int(i) for i in sel_indices)
        with self.mgr.tree.lock:
            k_parts, v_parts = [], []
            for node, used, off in self.entries:
                local = [g - off for g in sel if off <= g < off + used]
                if not local:
                    continue
                mapping = node.layer_mapping(layer_id)
                stored = sorted(mapping[i] if mapping is not None else i
                                for i in local)
                layer = node.store.layer(layer_id)
                cs = layer.meta["chunk_size"]
                by_chunk = {}
                for p in stored:
                    by_chunk.setdefault(p // cs, []).append(p)
                for cidx in sorted(by_chunk):
                    cinfo = layer.meta["chunks"][cidx]
                    k_c, v_c = self.mgr.chunk_pair(node, layer_id, cinfo)
                    rows = torch.tensor(
                        [p - cinfo["start"] for p in by_chunk[cidx]],
                        dtype=torch.long, device=k_c.device)
                    k_parts.append(k_c[rows])
                    v_parts.append(v_c[rows])
            return torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)


# --------------------------------------------------------------- attention
def mha_selective(dev, inputs, attention_mask, provider, layer_id,
                  w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln,
                  n_head, retention_ratio, config, controller=None,
                  io_counter=None):
    """Stage-5 similarity-guided attention with identify-then-load data
    flow (batch size 1). Numerically it consumes exactly the same selected
    K/V rows as Stage 5/8 — only the path that fetches them changes."""
    if w_q.device.device_type == DeviceType.COMPRESSED:
        w_q = w_q.device.decompress(w_q)
        w_k = w_k.device.decompress(w_k)
        w_v = w_v.device.decompress(w_v)
        w_out = w_out.device.decompress(w_out)

    b, n, h = inputs.shape
    assert b == 1, "selective serving path is batch-1"
    m = provider.num_tokens
    head_dim = h // n_head
    scaling = head_dim ** -0.5

    hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
    q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
    k = F.linear(hidden, w_k.data, bias=b_k.data)
    v = F.linear(hidden, w_v.data, bias=b_v.data)
    q = q.view(n, n_head, head_dim).permute(1, 0, 2)        # (nh, n, hd)
    k_new = k.view(n, n_head, head_dim).permute(1, 2, 0)    # (nh, hd, n)
    v_new = v.view(n, n_head, head_dim).permute(1, 0, 2)    # (nh, n, hd)

    total = m + n
    pos = torch.arange(total, device=dev.dev)
    causal = pos.view(1, total) <= pos[m:].view(n, 1)        # (n, total)
    fmask = (attention_mask.data.view(1, 1, 1, total)
             & causal.view(1, 1, n, total))                  # (1,1,n,total)
    new_mask = (attention_mask.data[:, m:].view(1, 1, 1, n)
                & causal[:, m:].view(1, 1, n, n))

    def attn_over(sel_k, sel_v):
        """Re-normalized attention over [selected prefix | new tokens].
        sel_k/sel_v: (nh, kk, hd)."""
        kk = sel_k.shape[1]
        a = torch.cat([torch.bmm(q, sel_k.transpose(1, 2)),
                       torch.bmm(q, k_new)], dim=2)          # (nh, n, kk+n)
        sel_mask = torch.ones((1, 1, n, kk), dtype=torch.bool,
                              device=dev.dev)
        a = a.view(1, n_head, n, kk + n)
        a = torch.where(torch.cat([sel_mask, new_mask], dim=3), a, -1e4)
        a = F.softmax(a.view(n_head, n, kk + n), dim=2)
        return torch.bmm(a, torch.cat([sel_v, v_new], dim=1))

    sim = threshold = None
    if retention_ratio >= 1.0:
        # FullLoad: every chunk pair, no filtering
        k_full, v_full = provider.load_full(layer_id)        # (m, nh, hd)
        pk = k_full.permute(1, 2, 0)
        pv = v_full.permute(1, 0, 2)
        a = torch.cat([torch.bmm(q, pk), torch.bmm(q, k_new)], dim=2)
        a = a.view(1, n_head, n, total)
        a = torch.where(fmask, a, -1e4)
        a = F.softmax(a.view(n_head, n, total), dim=2)
        value = torch.bmm(a, torch.cat([pv, v_new], dim=1))
        k_keep = m
        if io_counter is not None:
            io_counter.record(layer_id, "full", n_head * m * 2)
    else:
        k_keep = max(1, int(round(m * retention_ratio)))
        P = min(config.probe_head_count, n_head)

        # ---- step 1-2: probe keys ONLY (sidecar), per-head top-k ----
        pkp = provider.probe_keys(layer_id)                  # (m, P, hd)
        ap = torch.cat([torch.bmm(q[:P], pkp.permute(1, 2, 0)),
                        torch.bmm(q[:P], k_new[:P])], dim=2)  # (P, n, total)
        ap = ap.view(1, P, n, total)
        ap = torch.where(fmask, ap, -1e4)
        ap = F.softmax(ap.view(P, n, total), dim=2)
        imp_p = column_sum_importance(ap[:, :, :m])          # (P, m) fp32
        probe_top = imp_p.topk(k_keep, dim=-1).indices       # (P, k)

        masks = torch.zeros(P, m, dtype=torch.bool, device=probe_top.device)
        masks.scatter_(1, probe_top, True)
        if P >= 2:
            sims = []
            for i in range(P):
                for jj in range(i + 1, P):
                    inter = (masks[i] & masks[jj]).sum().float()
                    union = (masks[i] | masks[jj]).sum().float().clamp(min=1)
                    sims.append(inter / union)
            sim = float(torch.stack(sims).mean())
        else:
            sim = 1.0
        threshold = similarity_threshold(k_keep, m, config.alpha)

        if sim > threshold:
            # ---- step 3-4: vote, map to chunks, load ONLY those chunks ----
            votes = masks.sum(dim=0).float()                 # (m,)
            mean_imp = imp_p.mean(dim=0)                     # (m,)
            combined = votes * (10.0 * (n + 1)) + mean_imp
            sel = combined.topk(k_keep).indices              # (k,)
            sel_k, sel_v = provider.load_selected(layer_id, sel.tolist())
            value = attn_over(sel_k.permute(1, 0, 2), sel_v.permute(1, 0, 2))
            if io_counter is not None:
                io_counter.record(layer_id, "probe",
                                  count_vectors_probe(P, n_head, m, k_keep),
                                  sim, threshold)
            if (controller is not None and
                    getattr(controller, "selected_by_layer", None)
                    is not None):
                controller.selected_by_layer[layer_id] = (
                    "probe",
                    torch.sort(sel).values.detach().cpu())
            if (controller is not None and
                    getattr(controller, "token_importance", None)
                    is not None):
                # raw H2O importance (probe-head mean) for the §4.4.1
                # reorderer — "based on the average token importance"
                controller.token_importance[layer_id] = \
                    mean_imp.detach().cpu()
        else:
            # ---- step 5: fallback — all chunks, per-head selection ----
            k_full, v_full = provider.load_full(layer_id)    # (m, nh, hd)
            a = torch.cat([torch.bmm(q, k_full.permute(1, 2, 0)),
                           torch.bmm(q, k_new)], dim=2)
            a = a.view(1, n_head, n, total)
            a = torch.where(fmask, a, -1e4)
            a = F.softmax(a.view(n_head, n, total), dim=2)
            imp = column_sum_importance(a[:, :, :m])         # (nh, m)
            top = imp.topk(k_keep, dim=-1).indices           # (nh, k)
            gidx = top.unsqueeze(-1).expand(n_head, k_keep, head_dim)
            khb = k_full.permute(1, 0, 2)                    # (nh, m, hd)
            vhb = v_full.permute(1, 0, 2)
            sel_k = torch.gather(khb, 1, gidx)
            sel_v = torch.gather(vhb, 1, gidx)
            value = attn_over(sel_k, sel_v)
            if io_counter is not None:
                io_counter.record(layer_id, "fallback",
                                  count_vectors_naive(n_head, m, k_keep),
                                  sim, threshold)
            if (controller is not None and
                    getattr(controller, "selected_by_layer", None)
                    is not None):
                controller.selected_by_layer[layer_id] = (
                    "per_head", top.detach().cpu())
            if (controller is not None and
                    getattr(controller, "token_importance", None)
                    is not None):
                controller.token_importance[layer_id] = \
                    imp.mean(dim=0).detach().cpu()

    value = value.view(n_head, n, head_dim).transpose(0, 1).reshape(1, n, h)
    value = F.linear(value, w_out.data, bias=b_out.data)
    value.add_(inputs.data)

    if controller is not None:
        controller.last_kept = (k_keep, m)

    k_ret = k_new.permute(2, 0, 1)   # (n, nh, hd)
    v_ret = v_new.permute(1, 0, 2)
    return (TorchTensor.create_from_torch(value, dev),
            TorchTensor.create_from_torch(k_ret, dev),
            TorchTensor.create_from_torch(v_ret, dev))


# ------------------------------------------------------------------ hooks
def _make_sa_forward_selective(controller):
    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        if not (controller.enabled and i == 0
                and getattr(controller, "provider", None) is not None):
            ret = prefix_reuse._ORIG_SA_FORWARD(
                self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k)
            if i == 0 and controller.capture_new_kv:
                _capture_new_kv(controller, self.layer_id, cache_write_buf)
            return ret
        if k == self.policy.num_gpu_batches - 1:
            ((w_q, _), (b_q, _), (w_k, _), (b_k, _),
             (w_v, _), (b_v, _), (w_out, _), (b_out, _),
             (w_ln, _), (b_ln, _)) = weight_read_buf.pop()
        else:
            ((w_q, _), (b_q, _), (w_k, _), (b_k, _),
             (w_v, _), (b_v, _), (w_out, _), (b_out, _),
             (w_ln, _), (b_ln, _)) = weight_read_buf.val

        mask, _ = attention_mask.val.smart_copy(self.compute)
        h, k_new, v_new = mha_selective(
            self.compute, hidden.val, mask, controller.provider,
            self.layer_id, w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out,
            w_ln, b_ln, self.config.n_head, controller.retention_ratio,
            controller.config, controller, controller.io)
        cache_write_buf.store((k_new, v_new))
        if controller.capture_new_kv:
            _capture_new_kv(controller, self.layer_id, cache_write_buf)
        hidden.val = h
    return forward


def install_selective_hook(config=None):
    controller = SimGuidedController(config)
    controller.provider = None
    SelfAttention.forward = _make_sa_forward_selective(controller)
    OptLM.update_attention_mask = prefix_reuse._make_update_mask(controller)
    TorchDevice.opt_output_embed = prefix_reuse._make_output_embed(controller)
    return controller


def uninstall_selective_hook():
    SelfAttention.forward = prefix_reuse._ORIG_SA_FORWARD
    OptLM.update_attention_mask = prefix_reuse._ORIG_UPDATE_MASK
    TorchDevice.opt_output_embed = prefix_reuse._ORIG_OUTPUT_EMBED

"""Similarity-guided important token identification (IMPRESS Stage 5, §4.3).

Paper algorithm (re-checked against the FAST'25 paper, Figure 9(b)):
  1. Probe heads: "we simply select the first three heads in each transformer
     layer as the probe heads" (§4.3, Hyperparameter decisions). The count is
     configurable here (SimGuidedConfig.probe_head_count, default 3).
  2. Load only the probe heads' prefix keys; identify each probe head's
     top-k important token index set (Stage-4 method, per head).
  3. "We first compute the similarity of important token sets between each
     pair of probe heads, and then take the average" — average pairwise
     Jaccard J(A,B) = |A∩B| / |A∪B|.
  4. Dynamic per-layer threshold (§4.3): with n prefix tokens and k selected,
     E(Jaccard) of two random selections is j = (k/n) / (2 - k/n); the
     threshold is t = j ** alpha with alpha = 0.6 by default.
  5. If avg similarity > t: the probe heads vote — tokens are ranked by how
     many probe heads marked them important (ties broken by mean probe
     importance) — and the unified top-k set is applied to ALL heads, so
     only the selected tokens' K/V of the remaining heads need loading.
  6. Otherwise this layer falls back to the Stage-4 naive mode (all heads'
     keys, per-head selection).

I/O accounting (IOCounter) follows the paper's Figure 9 example exactly:
  naive/fallback: n_head * m        (all prefix keys)
                + n_head * k_keep   (selected values; keys already loaded)
                  -> 32*4 + 32*1 = 160 for the Figure 9 example
  probe:          P * m             (probe heads' prefix keys)
                + n_head * k_keep*2 (selected tokens' k AND v, all heads)
                  -> 3*4 + 32*1*2 = 76 for the Figure 9 example

Note: at this stage the counter reports the vectors the algorithm WOULD
load — the prefix KV itself stays resident in GPU memory (as in Stage 4);
physical chunk-granular disk I/O arrives with the §4.4 stages (reordering /
score-based cache).
"""

import torch
import torch.nn.functional as F

from flexllmgen.pytorch_backend import TorchDevice, TorchTensor, DeviceType
from flexllmgen.flex_opt import SelfAttention, OptLM
from flexllmgen.impress.importance import column_sum_importance
from flexllmgen.impress import prefix_reuse
from flexllmgen.impress.prefix_reuse import PrefixReuseController


class SimGuidedConfig:
    def __init__(self, probe_head_count=3, alpha=0.6):
        self.probe_head_count = probe_head_count
        self.alpha = alpha


# ------------------------------------------------------------- thresholds
def expected_jaccard(k_keep, n):
    """E(Jaccard) of two independent random k-of-n selections (§4.3):
    j = (k/n) / (2 - k/n)."""
    r = k_keep / n
    return r / (2.0 - r)


def similarity_threshold(k_keep, n, alpha=0.6):
    """t = j ** alpha (§4.3; alpha = 0.6 balances accuracy vs keys loaded)."""
    return expected_jaccard(k_keep, n) ** alpha


def avg_pairwise_jaccard(index_sets):
    """Average pairwise Jaccard over a list of index sets (or 1-D tensors)."""
    sets = [set(s.tolist()) if torch.is_tensor(s) else set(s)
            for s in index_sets]
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = len(sets[i] | sets[j])
            vals.append(len(sets[i] & sets[j]) / union if union else 1.0)
    return sum(vals) / len(vals)


# ------------------------------------------------------------ I/O counter
def count_vectors_naive(n_head, m, k_keep):
    """Figure 9(a): all heads' prefix keys + selected tokens' values."""
    return n_head * m + n_head * k_keep


def count_vectors_probe(probe_head_count, n_head, m, k_keep):
    """Figure 9(b): probe keys + selected tokens' k AND v for all heads."""
    return probe_head_count * m + n_head * k_keep * 2


class IOCounter:
    """Counts prefix-KV vectors loaded per layer, by identification mode."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_vectors = 0
        self.records = []  # dicts: layer, mode, vectors, sim, threshold

    def record(self, layer_id, mode, vectors, sim=None, threshold=None):
        self.total_vectors += vectors
        self.records.append(dict(layer=layer_id, mode=mode, vectors=vectors,
                                 sim=sim, threshold=threshold))

    def mode_counts(self):
        out = {}
        for r in self.records:
            out[r["mode"]] = out.get(r["mode"], 0) + 1
        return out

    def fallback_rate(self):
        """Fraction of probe-attempted layers that fell back (§6.3: <20%
        of layers on average in the paper)."""
        attempted = [r for r in self.records
                     if r["mode"] in ("probe", "fallback")]
        if not attempted:
            return 0.0
        return sum(r["mode"] == "fallback" for r in attempted) / len(attempted)


# ------------------------------------------------------------- attention
def mha_prefix_reuse_sim(dev, inputs, attention_mask, prefix_k, prefix_v,
                         w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out,
                         w_ln, b_ln, n_head, retention_ratio, config,
                         io_counter=None, use_probe=True, layer_id=None,
                         controller=None):
    """Stage-4 mha_prefix_reuse extended with §4.3 probe-head identification.

    use_probe=False reproduces the Stage-4 naive mode exactly (per-head
    selection from all heads' keys) with naive I/O accounting, so one code
    path serves both sides of the comparison.
    """
    if w_q.device.device_type == DeviceType.COMPRESSED:
        w_q = w_q.device.decompress(w_q)
        w_k = w_k.device.decompress(w_k)
        w_v = w_v.device.decompress(w_v)
        w_out = w_out.device.decompress(w_out)

    b, n, h = inputs.shape
    m = prefix_k.shape[0]
    assert m > 0 and prefix_k.shape[1] == b * n_head
    head_dim = h // n_head
    scaling = head_dim ** -0.5

    hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
    q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
    k = F.linear(hidden, w_k.data, bias=b_k.data)
    v = F.linear(hidden, w_v.data, bias=b_v.data)
    q = q.view(b, n, n_head, head_dim).permute(0, 2, 1, 3) \
         .reshape(b * n_head, n, head_dim)
    k_new = k.view(b, n, n_head, head_dim).permute(0, 2, 3, 1) \
             .reshape(b * n_head, head_dim, n)
    v_new = v.view(b, n, n_head, head_dim).permute(0, 2, 1, 3) \
             .reshape(b * n_head, n, head_dim)

    pk = prefix_k.permute(1, 2, 0)   # (b*nh, hd, m)
    pv = prefix_v.permute(1, 0, 2)   # (b*nh, m, hd)

    total = m + n
    pos = torch.arange(total, device=dev.dev)
    causal = pos.view(1, total) <= pos[m:].view(n, 1)              # (n, total)
    bmask = (attention_mask.data.view(b, 1, 1, total)
             & causal.view(1, 1, n, total))                        # (b,1,n,total)

    def full_head_softmax():
        a = torch.cat([torch.bmm(q, pk), torch.bmm(q, k_new)], dim=2)
        a = a.view(b, n_head, n, total)
        a = torch.where(bmask, a, -1e4)
        return F.softmax(a.view(b * n_head, n, total), dim=2)

    sim = threshold = None
    if retention_ratio >= 1.0:
        attn = full_head_softmax()
        value = torch.bmm(attn, torch.cat([pv, v_new], dim=1))
        if io_counter is not None:
            io_counter.record(layer_id, "full", n_head * m * 2)
        k_keep = m
    else:
        k_keep = max(1, int(round(m * retention_ratio)))
        top_idx = None

        if use_probe:
            # --- steps 1-2: probe heads' keys only, per-head top-k sets ---
            P = min(config.probe_head_count, n_head)
            q_bh = q.view(b, n_head, n, head_dim)
            pk_bh = prefix_k.permute(1, 0, 2).view(b, n_head, m, head_dim)
            kn_bh = k_new.view(b, n_head, head_dim, n)
            qp = q_bh[:, :P].reshape(b * P, n, head_dim)
            ap = torch.cat(
                [torch.bmm(qp, pk_bh[:, :P].reshape(b * P, m, head_dim)
                           .transpose(1, 2)),
                 torch.bmm(qp, kn_bh[:, :P].reshape(b * P, head_dim, n))],
                dim=2)
            ap = ap.view(b, P, n, total)
            ap = torch.where(bmask, ap, -1e4)
            ap = F.softmax(ap.view(b * P, n, total), dim=2)
            imp_p = column_sum_importance(ap[:, :, :m], n_head=P)  # (b,P,m)
            probe_top = imp_p.topk(k_keep, dim=-1).indices         # (b,P,k)

            # --- steps 3-4: avg pairwise Jaccard vs dynamic threshold ---
            masks = torch.zeros(b, P, m, dtype=torch.bool,
                                device=probe_top.device)
            masks.scatter_(2, probe_top, True)
            if P >= 2:
                sims = []
                for i in range(P):
                    for jj in range(i + 1, P):
                        inter = (masks[:, i] & masks[:, jj]).sum(-1).float()
                        union = (masks[:, i] | masks[:, jj]).sum(-1).float()
                        sims.append(inter / union.clamp(min=1))
                sim = float(torch.stack(sims).mean())
            else:
                # single probe head: no pairs to compare; §4.3 notes one
                # head "may introduce bias" — it always trusts itself
                sim = 1.0
            threshold = similarity_threshold(k_keep, m, config.alpha)

            if sim > threshold:
                # --- step 5: probe heads vote; unified set for ALL heads ---
                votes = masks.sum(dim=1).float()                   # (b, m)
                mean_imp = imp_p.mean(dim=1)                       # (b, m)
                combined = votes * (10.0 * (n + 1)) + mean_imp
                sel = combined.topk(k_keep, dim=-1).indices        # (b, k)
                top_idx = sel.unsqueeze(1).expand(b, n_head, k_keep)
                if io_counter is not None:
                    io_counter.record(
                        layer_id, "probe",
                        count_vectors_probe(P, n_head, m, k_keep),
                        sim, threshold)
                if (controller is not None and
                        getattr(controller, "selected_by_layer", None)
                        is not None):
                    controller.selected_by_layer[layer_id] = (
                        "probe", sel[0].detach().cpu())
                if (controller is not None and
                        getattr(controller, "token_importance", None)
                        is not None):
                    controller.token_importance[layer_id] = \
                        mean_imp[0].detach().cpu()
            # else: top_idx stays None -> step 6 fallback below

        if top_idx is None:
            # --- Stage-4 naive path (or §4.3 step-6 fallback) ---
            attn = full_head_softmax()
            imp = column_sum_importance(attn[:, :, :m], n_head=n_head)
            top_idx = imp.topk(k_keep, dim=-1).indices             # (b,nh,k)
            if io_counter is not None:
                io_counter.record(
                    layer_id, "fallback" if use_probe else "naive",
                    count_vectors_naive(n_head, m, k_keep), sim, threshold)
            if (controller is not None and
                    getattr(controller, "selected_by_layer", None)
                    is not None):
                controller.selected_by_layer[layer_id] = (
                    "per_head", top_idx[0].detach().cpu())
            if (controller is not None and
                    getattr(controller, "token_importance", None)
                    is not None):
                controller.token_importance[layer_id] = \
                    imp[0].mean(dim=0).detach().cpu()

        # --- shared: gather selected K/V, re-normalized attention ---
        pk_bh = prefix_k.permute(1, 0, 2).view(b, n_head, m, head_dim)
        pv_bh = prefix_v.permute(1, 0, 2).view(b, n_head, m, head_dim)
        gidx = top_idx.unsqueeze(-1).expand(b, n_head, k_keep, head_dim)
        sel_k = torch.gather(pk_bh, 2, gidx).reshape(b * n_head, k_keep,
                                                     head_dim)
        sel_v = torch.gather(pv_bh, 2, gidx).reshape(b * n_head, k_keep,
                                                     head_dim)
        attn2 = torch.cat(
            [torch.bmm(q, sel_k.transpose(1, 2)), torch.bmm(q, k_new)], dim=2)
        attn2 = attn2.view(b, n_head, n, k_keep + n)
        new_mask = (attention_mask.data[:, m:].view(b, 1, 1, n)
                    & causal[:, m:].view(1, 1, n, n))
        sel_mask = torch.ones((b, 1, n, k_keep), dtype=torch.bool,
                              device=dev.dev)
        attn2 = torch.where(torch.cat([sel_mask, new_mask], dim=3),
                            attn2, -1e4)
        attn2 = F.softmax(attn2.view(b * n_head, n, k_keep + n), dim=2)
        value = torch.bmm(attn2, torch.cat([sel_v, v_new], dim=1))

    value = value.view(b, n_head, n, head_dim).transpose(1, 2).reshape(b, n, h)
    value = F.linear(value, w_out.data, bias=b_out.data)
    value.add_(inputs.data)

    if controller is not None:
        controller.last_kept = (k_keep, m)

    k_ret = k_new.permute(2, 0, 1)
    v_ret = v_new.permute(1, 0, 2)
    return (TorchTensor.create_from_torch(value, dev),
            TorchTensor.create_from_torch(k_ret, dev),
            TorchTensor.create_from_torch(v_ret, dev))


# ------------------------------------------------------------------ hooks
class SimGuidedController(PrefixReuseController):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or SimGuidedConfig()
        self.io = IOCounter()
        self.use_probe = True
        # integration (Stage 8): per-request selection record and new-token
        # KV capture for radix-tree insertion of NR
        self.selected_by_layer = None  # dict to enable recording
        self.capture_new_kv = False
        self.new_kv = {}               # layer_id -> (k, v) cpu tensors
        # raw per-token H2O importance per layer (dict to enable); feeds
        # the §4.4.1 reorderer with actual average importance values
        self.token_importance = None


def _capture_new_kv(controller, layer_id, cache_write_buf):
    k_t, v_t = cache_write_buf.val
    controller.new_kv[layer_id] = (
        k_t.data.detach().to("cpu").contiguous().clone(),
        v_t.data.detach().to("cpu").contiguous().clone())


def _make_sa_forward_sim(controller):
    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        if not (controller.enabled and i == 0):
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
        p_k, p_v = controller.prefix_kv[self.layer_id]
        h, k_new, v_new = mha_prefix_reuse_sim(
            self.compute, hidden.val, mask, p_k, p_v,
            w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln,
            self.config.n_head, controller.retention_ratio, controller.config,
            io_counter=controller.io, use_probe=controller.use_probe,
            layer_id=self.layer_id, controller=controller)
        cache_write_buf.store((k_new, v_new))
        if controller.capture_new_kv:
            _capture_new_kv(controller, self.layer_id, cache_write_buf)
        hidden.val = h
    return forward


def install_sim_guided_hook(config=None):
    controller = SimGuidedController(config)
    SelfAttention.forward = _make_sa_forward_sim(controller)
    OptLM.update_attention_mask = prefix_reuse._make_update_mask(controller)
    TorchDevice.opt_output_embed = prefix_reuse._make_output_embed(controller)
    return controller


def uninstall_sim_guided_hook():
    SelfAttention.forward = prefix_reuse._ORIG_SA_FORWARD
    OptLM.update_attention_mask = prefix_reuse._ORIG_UPDATE_MASK
    TorchDevice.opt_output_embed = prefix_reuse._ORIG_OUTPUT_EMBED

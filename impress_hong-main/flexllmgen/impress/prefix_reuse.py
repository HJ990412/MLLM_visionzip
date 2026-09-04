"""Prefix-KV reuse with naive per-head important-token selection (Stage 4).

Implements the paper's Figure 9(a) "w/o similarity-guided token selection"
inside the §4.1 dataflow:
  (1) load the keys (and values) of ALL heads of the stored prefix from the
      Stage-1 store — no probe heads yet;
  (2) compute attention weights between the new tokens' queries and all
      prefix keys, and identify important prefix tokens PER HEAD with the
      Stage-3 H2O column-sum metric (§4.1 "Importance metric");
  (3) keep only the top `retention_ratio` tokens' K/V in each head and
      complete the remaining prefill computation with that subset — i.e.
      R_important (+ the new tokens) becomes the de facto prefix; the
      unimportant prefix KVs "are not reused and do not participate in
      further inference" (§4.1), so attention is re-normalized (softmax)
      over the reduced key set, as in H2O-style KV dropping.

Not implemented yet (later stages): probe-head identification (§4.3),
KV reordering (§4.4.1), score-based caching (§4.4.2).

Integration is monkey-patch only (no FlexGen file modified):
  - SelfAttention.forward  -> at prefill (i == 0), run mha_prefix_reuse with
    the stored prefix KV instead of TorchDevice.mha;
  - OptLM.update_attention_mask -> the prefill mask must cover
    prefix_len + new_len so OPT position ids continue after the prefix
    (opt_input_embed slices positions by mask_len - token_len);
  - TorchDevice.opt_output_embed -> capture last-token logits for
    multiple-choice scoring (copy of pytorch_backend.py:265-285).

Limitation (by design at this stage): the KV written back to the cache
covers only the NEW tokens, so generation beyond the first token does not
attend to the prefix. Use gen_len=1 for accuracy scoring.
"""

import numpy as np
import torch
import torch.nn.functional as F

from flexllmgen.pytorch_backend import TorchDevice, TorchTensor, DeviceType
from flexllmgen.flex_opt import SelfAttention, OptLM
from flexllmgen.impress.importance import column_sum_importance


def mha_prefix_reuse(dev, inputs, attention_mask, prefix_k, prefix_v,
                     w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln,
                     n_head, retention_ratio=1.0, controller=None):
    """Prefill attention for the new tokens only, reusing stored prefix KV,
    with per-head top-k prefix-token selection (paper Figure 9(a)).

    Mirrors TorchDevice.mha (pytorch_backend.py:298-365) for the projections
    and masking; differs in that (i) only the new tokens flow through the
    layer, (ii) prefix K/V come from storage, (iii) per-head selection.

    Args:
      inputs: TorchTensor (b, n_new, h) — hidden states of the NEW tokens.
      attention_mask: TorchTensor bool (b, m + n_new).
      prefix_k / prefix_v: torch.Tensor (m, b * n_head, head_dim), the
        Stage-1 storage layout.
      retention_ratio: keep round(m * ratio) prefix tokens per head (>=1 keeps
        all — identical to full reuse).
    Returns (value, k_new, v_new); k_new/v_new cover the new tokens only,
    in FlexGen cache layout (n_new, b * n_head, head_dim).
    """
    if w_q.device.device_type == DeviceType.COMPRESSED:
        w_q = w_q.device.decompress(w_q)
        w_k = w_k.device.decompress(w_k)
        w_v = w_v.device.decompress(w_v)
        w_out = w_out.device.decompress(w_out)

    b, n, h = inputs.shape
    m = prefix_k.shape[0]
    assert m > 0, "empty prefix; use the original mha"
    assert prefix_k.shape[1] == b * n_head
    head_dim = h // n_head
    scaling = head_dim ** -0.5

    hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

    q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
    k = F.linear(hidden, w_k.data, bias=b_k.data)
    v = F.linear(hidden, w_v.data, bias=b_v.data)
    # (b * n_head, n, head_dim)
    q = q.view(b, n, n_head, head_dim).permute(0, 2, 1, 3) \
         .reshape(b * n_head, n, head_dim)
    # (b * n_head, head_dim, n)
    k_new = k.view(b, n, n_head, head_dim).permute(0, 2, 3, 1) \
             .reshape(b * n_head, head_dim, n)
    # (b * n_head, n, head_dim)
    v_new = v.view(b, n, n_head, head_dim).permute(0, 2, 1, 3) \
             .reshape(b * n_head, n, head_dim)

    # prefix KV: (m, b*n_head, hd) -> bmm layouts
    pk = prefix_k.permute(1, 2, 0)   # (b * n_head, head_dim, m)
    pv = prefix_v.permute(1, 0, 2)   # (b * n_head, m, head_dim)

    total = m + n
    # (b * n_head, n, m + n): new-token queries vs [prefix keys | new keys]
    attn = torch.cat([torch.bmm(q, pk), torch.bmm(q, k_new)], dim=2)

    # every new token (abs pos m+j) sees all prefix tokens + causal among new
    pos = torch.arange(total, device=dev.dev)
    causal = pos.view(1, total) <= pos[m:].view(n, 1)          # (n, total)
    mask = attention_mask.data.view(b, 1, 1, total) & causal.view(1, 1, n, total)
    attn = attn.view(b, n_head, n, total)
    attn = torch.where(mask, attn, -1e4)
    attn = attn.view(b * n_head, n, total)
    attn = F.softmax(attn, dim=2)

    # Stage-3 importance of each prefix token, per head (b, n_head, m)
    importance = column_sum_importance(attn[:, :, :m], n_head=n_head)

    if retention_ratio >= 1.0:
        k_keep = m
        value = torch.bmm(attn, torch.cat([pv, v_new], dim=1))
    else:
        k_keep = max(1, int(round(m * retention_ratio)))
        top_idx = importance.topk(k_keep, dim=-1).indices      # (b, nh, k_keep)
        # gather each head's selected prefix K/V
        pk_bh = prefix_k.permute(1, 0, 2).view(b, n_head, m, head_dim)
        pv_bh = prefix_v.permute(1, 0, 2).view(b, n_head, m, head_dim)
        gidx = top_idx.unsqueeze(-1).expand(b, n_head, k_keep, head_dim)
        sel_k = torch.gather(pk_bh, 2, gidx).reshape(b * n_head, k_keep, head_dim)
        sel_v = torch.gather(pv_bh, 2, gidx).reshape(b * n_head, k_keep, head_dim)

        # re-normalized attention over [selected prefix | new tokens]
        attn2 = torch.cat(
            [torch.bmm(q, sel_k.transpose(1, 2)), torch.bmm(q, k_new)], dim=2)
        attn2 = attn2.view(b, n_head, n, k_keep + n)
        new_mask = (attention_mask.data[:, m:].view(b, 1, 1, n)
                    & causal[:, m:].view(1, 1, n, n))
        sel_mask = torch.ones((b, 1, n, k_keep), dtype=torch.bool,
                              device=dev.dev)
        attn2 = torch.where(torch.cat([sel_mask, new_mask], dim=3), attn2, -1e4)
        attn2 = attn2.view(b * n_head, n, k_keep + n)
        attn2 = F.softmax(attn2, dim=2)
        value = torch.bmm(attn2, torch.cat([sel_v, v_new], dim=1))

    value = value.view(b, n_head, n, head_dim).transpose(1, 2).reshape(b, n, h)
    value = F.linear(value, w_out.data, bias=b_out.data)
    value.add_(inputs.data)

    if controller is not None:
        controller.last_kept = (k_keep, m)

    # new tokens' KV in FlexGen cache layout (n, b * n_head, head_dim)
    k_ret = k_new.permute(2, 0, 1)
    v_ret = v_new.permute(1, 0, 2)
    return (TorchTensor.create_from_torch(value, dev),
            TorchTensor.create_from_torch(k_ret, dev),
            TorchTensor.create_from_torch(v_ret, dev))


# ------------------------------------------------------------------ hooks
class PrefixReuseController:
    def __init__(self):
        self.enabled = False
        self.retention_ratio = 1.0
        self.prefix_kv = {}      # layer_id -> (k, v) GPU torch tensors
        self.prefix_len = 0
        self.capture_logits = False
        self.last_logits = None  # (b, vocab) float32 CPU
        self.capture_full_logits = False
        self.full_logits = None  # (b, s, vocab) float32 CPU, new tokens only
        self.last_kept = None    # (k_keep, m) of the last layer run


_ORIG_SA_FORWARD = SelfAttention.forward
_ORIG_UPDATE_MASK = OptLM.update_attention_mask
_ORIG_OUTPUT_EMBED = TorchDevice.opt_output_embed


def _make_sa_forward(controller):
    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        if not (controller.enabled and i == 0):
            return _ORIG_SA_FORWARD(self, hidden, cache_read_buf,
                                    weight_read_buf, attention_mask,
                                    cache_write_buf, i, k)
        # weight unpacking mirrors flex_opt.py:434-442
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
        h, k_new, v_new = mha_prefix_reuse(
            self.compute, hidden.val, mask, p_k, p_v,
            w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln,
            self.config.n_head, controller.retention_ratio, controller)
        cache_write_buf.store((k_new, v_new))
        hidden.val = h
    return forward


def _make_update_mask(controller):
    def update_attention_mask(self, i, k):
        if not (controller.enabled and i == 0):
            return _ORIG_UPDATE_MASK(self, i, k)
        # prefill mask covers prefix + new tokens so positions continue
        # after the prefix (opt_input_embed, pytorch_backend.py:254-258)
        gpu_batch_size = self.policy.gpu_batch_size
        left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
        input_ids = self.output_ids[left:right, :self.task.prompt_len]
        mask_np = np.concatenate(
            [np.ones((gpu_batch_size, controller.prefix_len), dtype=bool),
             input_ids != self.config.pad_token_id], axis=1)
        attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
                             else self.env.gpu)
        val = attention_compute.allocate(mask_np.shape, bool)
        val.load_from_np(mask_np)
        self.attention_mask[k].store(val)
    return update_attention_mask


def _make_output_embed(controller):
    def opt_output_embed(self, inputs, w_ln, b_ln, w_token, donate,
                         do_sample, temperature):
        """Copy of TorchDevice.opt_output_embed (pytorch_backend.py:265-285)
        with last-token logits capture for multiple-choice scoring."""
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        b, s, h = inputs.shape
        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data,
                              bias=b_ln.data)
        if donate[0]: inputs.delete()

        logits = F.linear(hidden, w_token.data)
        last_token_logits = logits[:, -1, :]

        if controller.capture_logits:
            controller.last_logits = last_token_logits.detach().float().cpu()
        if getattr(controller, "capture_full_logits", False):
            controller.full_logits = logits.detach().float().cpu()

        if do_sample and not temperature < 1e-5:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            ids = torch.multinomial(probs, num_samples=1)
        else:
            ids = last_token_logits.argmax(dim=1, keepdim=True)
        return TorchTensor.create_from_torch(ids, self)
    return opt_output_embed


def install_prefix_reuse_hook():
    controller = PrefixReuseController()
    SelfAttention.forward = _make_sa_forward(controller)
    OptLM.update_attention_mask = _make_update_mask(controller)
    TorchDevice.opt_output_embed = _make_output_embed(controller)
    return controller


def uninstall_prefix_reuse_hook():
    SelfAttention.forward = _ORIG_SA_FORWARD
    OptLM.update_attention_mask = _ORIG_UPDATE_MASK
    TorchDevice.opt_output_embed = _ORIG_OUTPUT_EMBED

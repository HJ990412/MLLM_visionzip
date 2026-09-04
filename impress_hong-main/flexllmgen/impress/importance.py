"""Token importance metric (IMPRESS Stage 3, paper §4.1 "Importance metric").

Paper: "we use the sum of values in each column of the attention weight
matrix as the token's importance, following the same method in H2O [44].
A higher sum indicates greater token importance." Scores are computed per
layer and per head; no filtering happens at this stage.

Access point (Stage-0 survey): FlexGen's prefill attention computes the
post-softmax weights at pytorch_backend.py:342, immediately before the
V-multiply at :344, with shape (b * n_head, s, s). `install_importance_hook`
monkey-patches TorchDevice.mha with a copy of the original function
(pytorch_backend.py:298-365) that adds one importance computation at exactly
that point — no FlexGen file is modified.
"""

import torch
import torch.nn.functional as F

from flexllmgen.pytorch_backend import TorchDevice, TorchTensor, DeviceType
from flexllmgen.flex_opt import SelfAttention


# ----------------------------------------------------------------- metric
def column_sum_importance(attn_weights, n_head=None):
    """Column sums of post-softmax attention weights (H2O metric, §4.1).

    Column k of the (query x key) attention weight matrix corresponds to key
    token k; its sum over the query dim is that token's importance score.
    Computed independently per layer call and per head.

    Accepted shapes (importance is always over the last, key-token dim):
      (q, k)                                  -> (k,)
      (n_head, q, k)                          -> (n_head, k)
      (b * n_head, q, k)  [n_head required]   -> (b, n_head, k)
      (b, n_head, q, k)                       -> (b, n_head, k)

    The (b * n_head, q, k) form matches FlexGen's mha layout, produced by
    `attn_weights.view(b * n_head, s, s)` from (b, n_head, s, s) — the view
    here is its exact inverse. Sums are accumulated in float32.
    """
    w = attn_weights.float()
    if w.dim() == 2:
        return w.sum(dim=0)
    if w.dim() == 3:
        if n_head is None:
            # (n_head, q, k): one head per leading index
            return w.sum(dim=1)
        bh, q, k = w.shape
        assert bh % n_head == 0, f"{bh} not divisible by n_head={n_head}"
        return w.view(bh // n_head, n_head, q, k).sum(dim=2)
    if w.dim() == 4:
        return w.sum(dim=2)
    raise ValueError(f"unsupported attn_weights dim: {w.dim()}")


def to_token_score_pairs(scores_1d):
    """One head's score vector -> [(token_id, score), ...]."""
    assert scores_1d.dim() == 1
    return [(i, float(s)) for i, s in enumerate(scores_1d)]


# ------------------------------------------------------------------- hook
class ImportanceCollector:
    """Per-layer, per-head prefill importance scores.

    scores[layer_id] = float32 CPU tensor of shape (b, n_head, s).
    """

    def __init__(self):
        self.enabled = False
        self.current_layer = None
        self.scores = {}

    def clear(self):
        self.scores = {}

    def stacked(self):
        """(num_layers, b, n_head, s) tensor in layer order."""
        layers = sorted(self.scores)
        return torch.stack([self.scores[j] for j in layers], dim=0)


_ORIG_MHA = TorchDevice.mha
_ORIG_SA_FORWARD = SelfAttention.forward


def _make_hooked_mha(collector):
    def mha_with_importance(self, inputs, attention_mask, w_q, b_q, w_k, b_k,
                            w_v, b_v, w_out, b_out, w_ln, b_ln, n_head,
                            donate, compress_cache, comp_config):
        """Copy of TorchDevice.mha (pytorch_backend.py:298-365) with the
        importance computation inserted between softmax (:342) and the
        V-multiply (:344)."""
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, n_head, head_dim)
        v = v.view(b, s, n_head, head_dim)

        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)

        attn_weights = torch.bmm(q, k)

        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        attn_weights = F.softmax(attn_weights, dim=2)

        # ---- IMPRESS importance metric (paper §4.1), per layer & head ----
        if collector.enabled:
            scores = column_sum_importance(attn_weights, n_head=n_head)
            collector.scores[collector.current_layer] = scores.cpu()
        # ------------------------------------------------------------------

        value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
        value = value.transpose(1, 2).reshape(b, s, h)
        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)

        return TorchTensor.create_from_torch(value, self), k, v

    return mha_with_importance


def _make_hooked_sa_forward(collector):
    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        collector.current_layer = self.layer_id
        return _ORIG_SA_FORWARD(self, hidden, cache_read_buf, weight_read_buf,
                                attention_mask, cache_write_buf, i, k)
    return forward


def install_importance_hook():
    """Patch TorchDevice.mha / SelfAttention.forward; return the collector.
    Call uninstall_importance_hook() to restore the originals."""
    collector = ImportanceCollector()
    TorchDevice.mha = _make_hooked_mha(collector)
    SelfAttention.forward = _make_hooked_sa_forward(collector)
    return collector


def uninstall_importance_hook():
    TorchDevice.mha = _ORIG_MHA
    SelfAttention.forward = _ORIG_SA_FORWARD

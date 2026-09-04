"""Position-embedding extension for OPT via Position Interpolation (PI).

Why this method (design note, see also verify_pos_extend.py output):
- OPT uses LEARNED ABSOLUTE position embeddings (a (max_seq_len + 2, h)
  lookup table; the +2 is OPT's positional offset — FlexGen computes
  positions = cumsum(mask) + 1, so a length-s input uses rows 2..s+1).
- NTK-aware scaling and rotary-style PI operate on RoPE frequencies and do
  not apply to a learned table. An ALiBi-style swap replaces the position
  mechanism entirely and is not zero-shot for a model trained without it.
- The learned-table analogue of Position Interpolation (Chen et al., 2023)
  is LINEAR INTERPOLATION of the trained table: new position i in
  [0, target_len) maps to real coordinate j = i * (L-1)/(target_len-1) on
  the trained axis (L = 2048) and the two neighboring trained vectors are
  lerped. This is the standard zero-shot resize used for ViT/BERT position
  tables and is known to degrade gracefully without fine-tuning.

Short-sequence identity is NOT automatic under pure static PI (compressing
ALL positions would change every sequence), so this module adds an explicit
length switch: whenever the total sequence length fits the trained range
(<= 2048), the ORIGINAL table is used — outputs are bit-exact identical to
an unpatched model. Only longer sequences use the interpolated table.

Integration: monkey-patch of TorchDevice.opt_input_embed only (a copy of
pytorch_backend.py:239-263 with the table choice added). No FlexGen file is
modified; tables are captured lazily from the model's own w_pos weight on
the first embed call. install(target_len=2048) is a strict no-op path.
"""

import torch
import torch.nn.functional as F

from flexllmgen.pytorch_backend import TorchDevice, TorchTensor, DeviceType

_ORIG_INPUT_EMBED = TorchDevice.opt_input_embed

OPT_POS_OFFSET = 2  # OPT's embed_positions offset rows (positions start at 2)


def interpolate_pos_table(orig, target_len):
    """PI-style linear resize of a learned OPT position table.

    orig: (L + 2, h) tensor — rows [0, 2) are the offset rows (kept as-is),
    rows [2, L+2) are the trained position vectors.
    Returns a (target_len + 2, h) tensor of orig's dtype/device. For
    target_len == L this is value-identical to `orig` (torch.equal).
    """
    L = orig.shape[0] - OPT_POS_OFFSET
    if target_len == L:
        return orig.clone()
    head = orig[:OPT_POS_OFFSET]
    body = orig[OPT_POS_OFFSET:].float()          # (L, h), lerp in fp32
    # (1, h, L) -> (1, h, target_len); align_corners keeps both endpoints
    # on trained vectors (i=0 -> row 0, i=target-1 -> row L-1)
    resized = F.interpolate(body.t().unsqueeze(0), size=target_len,
                            mode="linear", align_corners=True)
    new_body = resized[0].t().to(orig.dtype)
    return torch.cat([head, new_body], dim=0)


class PosExtender:
    """State for the patched input embed: original + interpolated tables,
    captured lazily from the model's own w_pos on first use."""

    def __init__(self, target_len):
        self.target_len = target_len
        self.trained_len = None    # L, from the model's table
        self.orig_table = None     # (L+2, h)
        self.ext_table = None      # (target_len+2, h)
        self.uses = dict(original=0, extended=0)

    def tables_from(self, w_pos_data):
        if self.orig_table is None:
            self.orig_table = w_pos_data
            self.trained_len = w_pos_data.shape[0] - OPT_POS_OFFSET
            assert self.target_len >= self.trained_len, (
                "target_len must be >= the trained length")
            self.ext_table = interpolate_pos_table(w_pos_data,
                                                   self.target_len)
        return self.orig_table, self.ext_table

    def pick(self, total_len, w_pos_data):
        """Original table when the sequence fits the trained range
        (bit-exact short-sequence behavior); interpolated table otherwise."""
        orig, ext = self.tables_from(w_pos_data)
        if total_len <= self.trained_len:
            self.uses["original"] += 1
            return orig
        self.uses["extended"] += 1
        return ext


_ACTIVE = None  # the installed PosExtender (None = not installed)


def _make_input_embed(extender):
    def opt_input_embed(self, inputs, attention_mask, w_token, w_pos,
                        pad_token_id, donate):
        """Copy of TorchDevice.opt_input_embed (pytorch_backend.py:239-263)
        with the position table chosen by total sequence length."""
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)
            w_pos = w_pos.device.decompress(w_pos)

        token_ids = inputs.data
        mask = attention_mask.data
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        token_embed = F.embedding(token_ids, w_token.data, pad_token_id)

        positions = torch.cumsum(mask, dim=1).int() * mask + 1
        past_key_values_length = mask.shape[1] - token_ids.shape[1]
        positions = positions[:, past_key_values_length:]

        table = extender.pick(mask.shape[1], w_pos.data)
        pos_embed = F.embedding(positions, table)

        data = token_embed + pos_embed
        return TorchTensor.create_from_torch(data, self)
    return opt_input_embed


def install_pos_extension(target_len):
    """Enable PI-extended positions up to target_len tokens. Returns the
    PosExtender (inspect .uses to see which table served each prefill).
    target_len == 2048 is an exact no-op path (original table always)."""
    global _ACTIVE
    assert _ACTIVE is None, "position extension already installed"
    _ACTIVE = PosExtender(target_len)
    TorchDevice.opt_input_embed = _make_input_embed(_ACTIVE)
    return _ACTIVE


def uninstall_pos_extension():
    global _ACTIVE
    TorchDevice.opt_input_embed = _ORIG_INPUT_EMBED
    _ACTIVE = None

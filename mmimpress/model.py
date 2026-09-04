"""LLaVA-NeXT runner: loading, prompt encoding, visual geometry, prefix forward.

The visual block of LLaVA-NeXT ("anyres") is
    [ base 24x24 low-res patches | hi_h x (hi_w + 1) high-res patches ]
where the last column of each high-res row is a learned row separator
(image_newline).  Those separators are structural, so the selector always keeps
them and they never consume retention budget.  Their positions are derived from
the model's own unpad_image/get_anyres_image_grid_shape helpers rather than
hard-coded, and asserted against the real token count.
"""
from __future__ import annotations

import torch

from mmimpress.config import ATTN_IMPL, COMPUTE_DTYPE, LOAD_4BIT, MODEL_ID

_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16}


class LlavaRunner:
    def __init__(self, model_id=MODEL_ID, load_4bit=LOAD_4BIT, attn=ATTN_IMPL):
        self.model_id = model_id
        self.load_4bit = load_4bit
        self.attn = attn
        self.model = None
        self.processor = None

    def load(self):
        from transformers import (AutoProcessor, BitsAndBytesConfig,
                                  LlavaNextForConditionalGeneration)
        kw = dict(attn_implementation=self.attn,
                  dtype=_DTYPE[COMPUTE_DTYPE], device_map="cuda:0")
        if self.load_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=_DTYPE[COMPUTE_DTYPE],
                bnb_4bit_use_double_quant=True)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id, **kw).eval()
        assert self.model.config._attn_implementation == self.attn, \
            f"attn is {self.model.config._attn_implementation}, need {self.attn}"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        return self

    # ------------------------------------------------------------ config
    @property
    def cfg(self):
        return self.model.config

    @property
    def layers(self):
        lm = self.model.model.language_model
        return lm.layers

    @property
    def image_token_id(self):
        return self.cfg.image_token_index

    @property
    def n_heads(self):
        return self.cfg.text_config.num_attention_heads

    @property
    def head_dim(self):
        tc = self.cfg.text_config
        return getattr(tc, "head_dim", tc.hidden_size // tc.num_attention_heads)

    # ------------------------------------------------------------ encode
    def prompt(self, question: str) -> str:
        return (f"USER: <image>\n{question} "
                "Answer the question using a single word or phrase. ASSISTANT:")

    def encode(self, image, question: str):
        enc = self.processor(images=image, text=self.prompt(question),
                             return_tensors="pt")
        return {k: v for k, v in enc.items()}

    def to_device(self, enc):
        dev = self.model.device
        return {k: (v.to(dev) if torch.is_tensor(v) else v)
                for k, v in enc.items()}

    # ---------------------------------------------------------- geometry
    def visual_span(self, input_ids):
        pos = (input_ids == self.image_token_id).nonzero(as_tuple=True)[0]
        assert pos.numel() > 0, "no image tokens in input_ids"
        v0, v1 = int(pos[0]), int(pos[-1])
        assert pos.numel() == v1 - v0 + 1, "image token span is not contiguous"
        return v0, int(pos.numel())

    def anyres_layout(self, image_size, v_num: int):
        """(base, hi_h, hi_w, newline_local_indices) for one image.

        hi_w excludes the separator column; the separator of row r sits at
        local index base*base + r*(hi_w+1) + hi_w.
        """
        from transformers.models.llava_next.modeling_llava_next import (
            get_anyres_image_grid_shape, unpad_image)
        vc = self.cfg.vision_config
        base = vc.image_size // vc.patch_size
        nph, npw = get_anyres_image_grid_shape(
            image_size, self.cfg.image_grid_pinpoints, vc.image_size)
        probe = torch.zeros(1, base * nph, base * npw)
        unpadded = unpad_image(probe, image_size)
        hi_h, hi_w = int(unpadded.shape[1]), int(unpadded.shape[2])
        assert base * base + hi_h * (hi_w + 1) == v_num, \
            (base, hi_h, hi_w, v_num)
        nl = [base * base + r * (hi_w + 1) + hi_w for r in range(hi_h)]
        return base, hi_h, hi_w, nl

    # ----------------------------------------------------------- forward
    @torch.no_grad()
    def prefix_forward(self, enc, prefix_len: int):
        """Run [system + image block] only and return (outputs, embeddings).

        The cut is right after the image block, so the cached KV is a prefix of
        every request about this image regardless of the question -- the image
        is the reusable prefix, the question is the query.
        """
        enc = self.to_device(enc)
        out = self.model(input_ids=enc["input_ids"][:, :prefix_len],
                         pixel_values=enc["pixel_values"],
                         image_sizes=enc["image_sizes"],
                         attention_mask=torch.ones(
                             1, prefix_len, dtype=torch.long,
                             device=self.model.device),
                         use_cache=True, return_dict=True)
        return out

    @torch.no_grad()
    def input_embeddings(self, enc, prefix_len: int):
        """Pre-layer hidden states of [prefix | suffix], for rater selection.

        Captured with a pre-hook on layer 0 so the multimodal merge (image
        features spliced into the text embeddings) has already happened.
        """
        enc = self.to_device(enc)
        box = {}

        def grab(module, args, kwargs):
            if "h" not in box:
                box["h"] = (args[0] if args else
                            kwargs["hidden_states"]).detach()

        hook = self.layers[0].register_forward_pre_hook(grab, with_kwargs=True)
        try:
            self.model(input_ids=enc["input_ids"],
                       pixel_values=enc["pixel_values"],
                       image_sizes=enc["image_sizes"],
                       attention_mask=torch.ones_like(enc["input_ids"]),
                       use_cache=False, return_dict=True)
        finally:
            hook.remove()
        return box["h"]


def cache_layers(past_key_values):
    """HF cache object -> [(k, v)] per layer, each (1, H, S, hd)."""
    if hasattr(past_key_values, "layers"):
        return [(l.keys, l.values) for l in past_key_values.layers]
    if hasattr(past_key_values, "key_cache"):
        return list(zip(past_key_values.key_cache, past_key_values.value_cache))
    return list(past_key_values)

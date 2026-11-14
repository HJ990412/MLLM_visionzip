"""LLaVA-1.5-7B runner (``llava-hf/llava-1.5-7b-hf``).

Same Vicuna-7B v1.5 decoder (MHA, 32 layers x 32 heads x 128) and the same
NF4 / BF16 / eager loading as ``mmimpress.model.LlavaRunner``; the differences
are the model class (``LlavaForConditionalGeneration``), the absence of
``image_sizes`` and the visual geometry: one CLIP ViT-L/14-336 crop gives
exactly 576 patch tokens (CLS dropped by ``vision_feature_select_strategy =
"default"``), contiguous, with no row separator and no AnyRes grid.

The image preprocessing is whatever ``AutoProcessor`` of the pinned revision
does by default (resize shortest edge 336 + centre crop 336); it is never
overridden here.
"""
from __future__ import annotations

from pathlib import Path

import torch

from mmimpress.config import ATTN_IMPL, COMPUTE_DTYPE, LOAD_4BIT
from mmimpress.llava15 import MODEL_ID, N_VISUAL, REVISION
from mmimpress.model import _DTYPE, LlavaRunner


def snapshot_path(revision: str = REVISION) -> Path:
    return (Path.home() / ".cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots" / revision)


class Llava15Runner(LlavaRunner):
    """``LlavaRunner`` API for LLaVA-1.5; AnyRes entry points fail closed."""

    def __init__(self, model_id=None, load_4bit=LOAD_4BIT, attn=ATTN_IMPL, revision=REVISION):
        source = str(snapshot_path(revision)) if model_id is None else model_id
        super().__init__(model_id=source, load_4bit=load_4bit, attn=attn)
        self.source = source
        self.revision = revision

    def load(self):
        from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
        if not (Path(self.source) / "config.json").is_file():
            raise FileNotFoundError(f"pinned LLaVA-1.5 snapshot missing: {self.source}")
        kw = dict(attn_implementation=self.attn, dtype=_DTYPE[COMPUTE_DTYPE], device_map="cuda:0")
        if self.load_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=_DTYPE[COMPUTE_DTYPE],
                bnb_4bit_use_double_quant=True)
        self.model = LlavaForConditionalGeneration.from_pretrained(self.source, **kw).eval()
        assert type(self.model).__name__ == "LlavaForConditionalGeneration"
        assert self.model.config._attn_implementation == self.attn, \
            f"attn is {self.model.config._attn_implementation}, need {self.attn}"
        self.processor = AutoProcessor.from_pretrained(self.source)
        self.model_id = MODEL_ID
        self.validate_architecture()
        return self

    # ------------------------------------------------------------ contract
    def validate_architecture(self):
        cfg = self.cfg
        tc, vc = cfg.text_config, cfg.vision_config
        assert tc.num_hidden_layers == 32 and tc.num_attention_heads == 32, (tc.num_hidden_layers, tc.num_attention_heads)
        assert tc.num_attention_heads == tc.num_key_value_heads, "MHA required"
        assert self.head_dim == 128 and len(self.layers) == 32
        assert vc.image_size == 336 and vc.patch_size == 14
        assert cfg.vision_feature_layer == -2, cfg.vision_feature_layer
        assert cfg.vision_feature_select_strategy == "default", cfg.vision_feature_select_strategy
        assert (vc.image_size // vc.patch_size) ** 2 == N_VISUAL
        assert not hasattr(cfg, "image_grid_pinpoints") or not getattr(cfg, "image_grid_pinpoints", None)
        assert self.model.dtype == torch.bfloat16 and not self.model.training

    def describe(self) -> dict:
        """Everything Phase 0 records about the pinned model / processor."""
        import bitsandbytes
        import transformers
        cfg = self.cfg
        ip = self.processor.image_processor
        return {
            "model_id": MODEL_ID, "revision": self.revision, "source": self.source,
            "model_class": type(self.model).__name__, "processor_class": type(self.processor).__name__,
            "image_processor_class": type(ip).__name__, "image_processor_config": ip.to_dict(),
            "tokenizer_class": type(self.processor.tokenizer).__name__,
            "processor_attributes": {k: getattr(self.processor, k, None) for k in
                                     ("patch_size", "vision_feature_select_strategy", "num_additional_image_tokens", "image_token")},
            "vision_feature_layer": cfg.vision_feature_layer,
            "vision_feature_select_strategy": cfg.vision_feature_select_strategy,
            "image_token_index": int(cfg.image_token_index),
            "vision": {"image_size": cfg.vision_config.image_size, "patch_size": cfg.vision_config.patch_size,
                       "num_hidden_layers": cfg.vision_config.num_hidden_layers,
                       "patches": (cfg.vision_config.image_size // cfg.vision_config.patch_size) ** 2,
                       "cls_token_dropped_by_default_strategy": True},
            "text": {"num_hidden_layers": cfg.text_config.num_hidden_layers, "num_attention_heads": cfg.text_config.num_attention_heads,
                     "num_key_value_heads": cfg.text_config.num_key_value_heads, "head_dim": self.head_dim,
                     "hidden_size": cfg.text_config.hidden_size},
            "attn_implementation": self.model.config._attn_implementation, "load_4bit": self.load_4bit,
            "quantization": "NF4 double quant, BF16 compute", "compute_dtype": str(self.model.dtype),
            "transformers": transformers.__version__, "bitsandbytes": bitsandbytes.__version__, "torch": torch.__version__,
            "prompt_template": self.prompt("{question}"),
        }

    # -------------------------------------------------------------- encode
    def image_inputs(self, image):
        """Pixels only (no text); LLaVA-1.5 has no ``image_sizes``."""
        enc = self.processor.image_processor(images=image, return_tensors="pt")
        assert "pixel_values" in enc and "image_sizes" not in enc
        return {k: v for k, v in enc.items()}

    # ------------------------------------------------------------ geometry
    def visual_span(self, input_ids):
        start, num = super().visual_span(input_ids)
        if num != N_VISUAL:
            raise AssertionError(f"LLaVA-1.5 visual span must be {N_VISUAL} contiguous tokens, got {num}")
        return start, num

    def flat_layout(self, v_num: int):
        """(grid side, separator indices) of the single 24x24 crop: no separators."""
        vc = self.cfg.vision_config
        base = vc.image_size // vc.patch_size
        if int(v_num) != base * base or int(v_num) != N_VISUAL:
            raise AssertionError((base, v_num))
        return base, []

    def anyres_layout(self, image_size, v_num: int):
        raise NotImplementedError("LLaVA-1.5 has no AnyRes layout; use flat_layout")

    # ------------------------------------------------------------- forward
    @torch.no_grad()
    def prefix_forward(self, enc, prefix_len: int):
        enc = self.to_device(enc)
        return self.model(input_ids=enc["input_ids"][:, :prefix_len], pixel_values=enc["pixel_values"],
                          attention_mask=torch.ones(1, prefix_len, dtype=torch.long, device=self.model.device),
                          use_cache=True, return_dict=True)

    @torch.no_grad()
    def input_embeddings(self, enc, prefix_len: int):
        raise NotImplementedError("rater selection is not part of the LLaVA-1.5 port")


def load_runner(seed: int = 1234) -> Llava15Runner:
    import random

    import numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    runner = Llava15Runner().load()
    assert runner.attn == "eager" and runner.load_4bit
    return runner

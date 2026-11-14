"""LLaVA-1.5-7B port of the packed PrefixStore path (contract docs/llava15_mt_gqa_contract.md).

New modules only; nothing under ``mmimpress/*.py`` or ``mmimpress/qwen25`` is modified.

runner : ``Llava15Runner`` (``LlavaForConditionalGeneration``, 576 visual tokens, no separator)
store  : ratio-general packed writer / validator / Turn-1 piggyback persistence
serve  : packed context, layer-wise prefix selector and hook-free cache hit
"""

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
N_VISUAL = 576
NUM_LAYERS = 32
R_MIN = 0.0          # contract: every layer_budget call of this port passes these explicitly
R_MAX = 1.0

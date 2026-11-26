"""Token sequence of the LLaVA-NeXT MT-GQA prompt.

The MT-GQA prompt string is ``USER: <image>\\n...ASSISTANT:`` (``mmimpress.mt_gqa``,
scripts/156 ``prompt_factory``).  Two procedures turn it into token ids:

original LLaVA
    ``tokenizer_image_token`` of the LLaVA repository with the slow, legacy
    SentencePiece tokenizer: the text before and after ``<image>`` is tokenised
    separately, one BOS is kept, the image stands at -200.  The chunk after the
    image starts with ``\\n``, which the legacy tokenizer encodes as ``▁`` (29871)
    followed by the newline token (13).

HF path of this repository up to 2026-11-22 (``TOKENIZATION_LEGACY``)
    the fast tokenizer of ``llava-hf/llava-v1.6-vicuna-7b-hf`` applied to the whole
    string.  It emits no ``▁`` after ``<image>``: one token fewer than the original
    procedure, in every MT-GQA prompt (ConvBench main report, section 11).

``TOKENIZATION_OFFICIAL`` hands the fast tokenizer the same prompt with one space
inserted right after ``<image>`` (``<image> \\n``) -- the rendering of the ConvBench
baseline alignment (docs/convbench_baseline_alignment.md; ``OFFICIAL_RENDER["next"]``
of scripts/213_convbench_recomp_generate.py).  The fast tokenizer then emits exactly
the original-LLaVA ids.  The prompt string itself (what is recorded, hashed and
rebuilt by the audits) is not changed; only the tokenizer input is.  Whether the ids
are equal is checked per prompt with ``compare`` -- the rendering is never trusted.

The ids before ``<image>`` and the image block are the same in both procedures, so
the stored visual prefix covers the same positions; every token after the image
moves one position to the right.

This module is for the LLaVA-NeXT checkpoint only.  The LLaVA-1.5 path
(``mmimpress/llava15``) does not use it: its fast tokenizer already emits the
original ids (0 / 120 mismatches in the stage-0 check).

Nothing here loads a model or touches a GPU.
"""
from __future__ import annotations

TOKENIZATION_LEGACY = "legacy"
TOKENIZATION_OFFICIAL = "official"
TOKENIZATIONS = (TOKENIZATION_LEGACY, TOKENIZATION_OFFICIAL)
IMAGE_MARKER = "<image>"
OFFICIAL_IMAGE_INDEX = -200          # IMAGE_TOKEN_INDEX of the LLaVA repository
# inserted right after "<image>" in the tokenizer input (OFFICIAL_RENDER["next"][0] of scripts/213 is this + "\n")
OFFICIAL_AFTER_IMAGE = " "
EOS_TEXT = "</s>"


def tokenizer_input(prompt: str, tokenization: str) -> str:
    """The string handed to the HF fast tokenizer for ``prompt``.

    ``legacy``: the prompt itself.  ``official``: the prompt with one space after
    ``<image>``.  The MT-GQA template has exactly one ``<image>``, a newline right
    after it and no ``</s>``; anything else is refused rather than rendered by a
    rule that was never checked for it.
    """
    if tokenization == TOKENIZATION_LEGACY:
        return prompt
    if tokenization != TOKENIZATION_OFFICIAL:
        raise ValueError(f"unknown tokenization: {tokenization!r}")
    if prompt.count(IMAGE_MARKER) != 1:
        raise ValueError("the MT-GQA prompt holds exactly one <image>")
    head, tail = prompt.split(IMAGE_MARKER)
    if not tail.startswith("\n"):
        raise ValueError("the MT-GQA prompt has a newline right after <image>")
    if EOS_TEXT in prompt:
        raise ValueError("the MT-GQA prompt holds no </s>")
    return head + IMAGE_MARKER + OFFICIAL_AFTER_IMAGE + tail


def prompt_of(text: str, tokenization: str) -> str:
    """Inverse of ``tokenizer_input``."""
    if tokenization == TOKENIZATION_LEGACY:
        return text
    if tokenization != TOKENIZATION_OFFICIAL:
        raise ValueError(f"unknown tokenization: {tokenization!r}")
    marker = IMAGE_MARKER + OFFICIAL_AFTER_IMAGE + "\n"
    if text.count(IMAGE_MARKER) != 1 or marker not in text:
        raise ValueError("not an official-rendered MT-GQA tokenizer input")
    return text.replace(marker, IMAGE_MARKER + "\n", 1)


def official_token_ids(slow_tokenizer, prompt: str) -> list[int]:
    """Original LLaVA ``tokenizer_image_token`` (same as scripts/213 ``official_token_ids``)."""
    chunks = [slow_tokenizer(chunk).input_ids for chunk in prompt.split(IMAGE_MARKER)]
    bos = slow_tokenizer.bos_token_id
    ids: list[int] = []
    offset = 0
    if chunks and chunks[0] and chunks[0][0] == bos:
        offset = 1
        ids.append(bos)
    for index, chunk in enumerate(chunks):
        if index > 0:
            ids.append(OFFICIAL_IMAGE_INDEX)
        ids.extend(chunk[offset:])
    return ids


def collapse_image(ids, image_token_id: int) -> list[int]:
    """Every run of the image token (expanded or not) -> one -200."""
    out: list[int] = []
    for value in ids:
        value = int(value)
        if value == image_token_id:
            if not out or out[-1] != OFFICIAL_IMAGE_INDEX:
                out.append(OFFICIAL_IMAGE_INDEX)
        else:
            out.append(value)
    return out


def path_token_ids(fast_tokenizer, image_token_id: int, prompt: str, tokenization: str) -> list[int]:
    """Ids the HF path produces for ``prompt`` under ``tokenization``, the image as one -200."""
    return collapse_image(fast_tokenizer(tokenizer_input(prompt, tokenization)).input_ids, image_token_id)


def compare(fast_tokenizer, slow_tokenizer, image_token_id: int, prompt: str, tokenization: str) -> dict:
    """The HF path's ids for ``prompt`` against the original-LLaVA ids."""
    path = path_token_ids(fast_tokenizer, image_token_id, prompt, tokenization)
    official = official_token_ids(slow_tokenizer, prompt)
    return {"match": path == official, "path_ids": path, "official_ids": official}


def load_tokenizers(source: str, revision: str | None = None):
    """(fast tokenizer the processor of the runner holds, slow legacy tokenizer of the original procedure, image token id)."""
    from transformers import AutoTokenizer
    kwargs = {"revision": revision} if revision else {}
    fast = AutoTokenizer.from_pretrained(source, **kwargs)
    slow = AutoTokenizer.from_pretrained(source, use_fast=False, legacy=True, **kwargs)
    if not fast.is_fast or slow.is_fast:
        raise RuntimeError("expected one fast and one slow tokenizer")
    return fast, slow, int(fast.convert_tokens_to_ids(IMAGE_MARKER))

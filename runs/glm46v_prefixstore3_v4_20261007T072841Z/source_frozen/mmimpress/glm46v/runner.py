"""Native GLM4V Flash inference with GLM46V preprocessing and SSD KV reuse.

Physical cache slots and original three-axis positions are separate. No K is
re-rotated and no vision pass is executed on the SSD path.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

MODEL_ID = "zai-org/GLM-4.6V-Flash"
REVISION = "411bb4d77144a3f03accbf4b780f5acb8b7cde4e"
SYSTEM_PROMPT = "You are a helpful assistant. Give concise factual answers."
ANSWER_INSTRUCTION = "Answer using a single word or short phrase."
IMAGE_SIZE = {"shortest_edge": 112 * 112, "longest_edge": 2 * 28 * 28 * 1024}
POSITION_POLICY = "glm4v-native-3axis-partial-rope-original-logical-compact-physical-v1"
METHODS = ("recompute", "fullload", "ours_prefixstore_kv25")
MAX_NEW_TOKENS = 32
BEGIN_OF_BOX_ID = 151361
END_OF_BOX_ID = 151362


def native_final_answer(tokenizer, generated):
    """Decode GLM's native final-answer span, keeping original IDs elsewhere.

    The pinned tokenizer marks the two answer-boundary control IDs as ordinary
    added tokens, so skip_special_tokens alone does not remove the wrapper.
    This structural parsing uses neither the question nor the gold answer.
    Malformed/multiple boundaries retain the literal output and are audited.
    """
    begins = [i for i, token in enumerate(generated) if token == BEGIN_OF_BOX_ID]
    ends = [i for i, token in enumerate(generated) if token == END_OF_BOX_ID]
    if len(begins) == len(ends) == 1 and begins[0] < ends[0]:
        span = generated[begins[0]+1:ends[0]]
        return tokenizer.decode(span, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False).strip(), {
                    "policy": "native-final-answer-box-v1", "status": "CLOSED_NATIVE_BOX",
                    "answer_start": begins[0]+1, "answer_end": ends[0]}
    literal = tokenizer.decode(generated, skip_special_tokens=True,
                               clean_up_tokenization_spaces=False).strip()
    return literal, {"policy": "native-final-answer-box-v1",
                     "status": "PLAIN_TEXT" if not begins and not ends else "MALFORMED_NATIVE_BOX"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def milliseconds(start):
    return (time.perf_counter() - start) * 1000


def cache_pairs(cache):
    return [(layer.keys, layer.values) for layer in cache.layers]


def prefix_geometry(ids, grid, config):
    """Identify real content and native boundary tokens; reject other layouts."""
    flat = ids.reshape(-1).tolist()
    content = [i for i, token in enumerate(flat) if token == config.image_token_id]
    starts = [i for i, token in enumerate(flat) if token == config.image_start_token_id]
    ends = [i for i, token in enumerate(flat) if token == config.image_end_token_id]
    if len(starts) != 1 or len(ends) != 1 or grid.shape != (1, 3) or not content:
        raise ValueError("exactly one native image is required")
    begin, end = starts[0], ends[0]
    if content != list(range(begin + 1, end)):
        raise ValueError("native content must be contiguous within image boundaries")
    t, h, w = map(int, grid[0].tolist())
    merge = int(config.vision_config.spatial_merge_size)
    if t != 1 or h % merge or w % merge or len(content) != t * h * w // merge**2:
        raise ValueError("native grid/content correspondence changed")
    return {"image_grid_thw": [[t, h, w]], "N_content": len(content),
            "k": (len(content) + 3) // 4, "content_indices": content,
            "prefix_len": end + 1, "image_start_index": begin,
            "image_end_index": end, "merge_size": merge,
            "resized_hw": [h * int(config.vision_config.patch_size),
                           w * int(config.vision_config.patch_size)]}


def independent_positions(ids, content_indices, grid, merge_size=2):
    """Single-image reference derived from geometry, independent of HF helper."""
    p = ids.shape[1]
    a, b = content_indices[0], content_indices[-1] + 1
    t, h, w = map(int, grid[0].tolist())
    if t != 1 or content_indices != list(range(a, b)):
        raise ValueError("reference supports a contiguous single image")
    hh, ww = h // merge_size, w // merge_size
    if b - a != hh * ww:
        raise ValueError("wrong content count")
    positions = torch.empty(3, 1, p, dtype=torch.long)
    positions[:, 0, :a] = torch.arange(a).expand(3, -1)
    positions[0, 0, a:b] = a
    positions[1, 0, a:b] = torch.arange(hh).repeat_interleave(ww) + a
    positions[2, 0, a:b] = torch.arange(ww).repeat(hh) + a
    positions[:, 0, b:] = (torch.arange(p - b) + a + max(hh, ww)).expand(3, -1)
    return positions, torch.tensor([[int(positions.max()) + 1 - p]])


@dataclass
class CapturedPrefix:
    layers: list
    prefix_ids: list[int]
    positions: torch.Tensor
    rope_offset: list
    content_indices: list[int]
    image_sha256: str
    geometry: dict
    scores: torch.Tensor | None
    capture_clone_ms: float
    score_ms: float


class GLM46VRunner:
    def __init__(self, checkpoint, *, max_new_tokens=MAX_NEW_TOKENS):
        self.checkpoint = str(Path(checkpoint).resolve())
        self.max_new_tokens = int(max_new_tokens)
        self.model = self.processor = None
        self.device = torch.device("cuda:0")
        self.stores = {}
        self.expected_identities = {}

    def load(self):
        from transformers import AutoProcessor, BitsAndBytesConfig, Glm4vForConditionalGeneration
        import bitsandbytes as bnb
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; run authorized GPU commands outside sandbox")
        torch.set_num_threads(2)
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.bfloat16,
                                  llm_int8_skip_modules=["model.visual", "lm_head"])
        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint, local_files_only=True, size=IMAGE_SIZE, use_fast=True)
        if (self.processor.tokenizer.convert_tokens_to_ids('<|begin_of_box|>') != BEGIN_OF_BOX_ID
            or self.processor.tokenizer.convert_tokens_to_ids('<|end_of_box|>') != END_OF_BOX_ID):
            raise RuntimeError('pinned native answer-boundary token IDs changed')
        self.model = Glm4vForConditionalGeneration.from_pretrained(
            self.checkpoint, local_files_only=True, dtype=torch.bfloat16,
            quantization_config=quant, device_map={"": 0}, attn_implementation="sdpa").eval()
        if any(p.device.type != "cuda" for p in self.model.parameters()):
            raise RuntimeError("CPU/disk weight offload is forbidden")
        quantized = [n for n, m in self.model.named_modules() if isinstance(m, bnb.nn.Linear4bit)]
        if any("visual" in n or n == "lm_head" for n in quantized):
            raise RuntimeError("vision/merger/lm_head must remain native BF16")
        if not quantized or any(p.dtype != torch.bfloat16 for p in self.model.model.visual.parameters()):
            raise RuntimeError("decoder quantization or BF16 vision policy violated")
        tc = self.model.config.text_config
        actual = (tc.num_hidden_layers, tc.num_key_value_heads, tc.hidden_size // tc.num_attention_heads)
        if actual != (40, 2, 128):
            raise RuntimeError(f"Flash native geometry differs: {actual}")
        self.quantized_modules = quantized
        return self

    def processor_settings(self):
        return {"processor_class": type(self.processor).__name__,
                "image_processor_class": type(self.processor.image_processor).__name__,
                "size": IMAGE_SIZE, "use_fast": True, "enable_thinking": False,
                "system_prompt": SYSTEM_PROMPT, "answer_instruction": ANSWER_INSTRUCTION}

    def fingerprint(self):
        import importlib.metadata as metadata
        tc, vc = self.model.config.text_config, self.model.config.vision_config
        return {"model_id": MODEL_ID, "checkpoint_revision": REVISION,
                "processor_revision": REVISION, "model_class": type(self.model).__name__,
                "versions": {p: metadata.version(p) for p in
                             ["torch", "torchvision", "transformers", "bitsandbytes", "accelerate",
                              "huggingface_hub", "tokenizers", "Pillow"]},
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                "quantized_modules": self.quantized_modules,
                "quantized_vision_modules": 0, "vision_dtype": str(self.model.model.visual.dtype),
                "text_backend": tc._attn_implementation, "vision_backend": vc._attn_implementation,
                "native_layers": tc.num_hidden_layers, "native_kv_heads": tc.num_key_value_heads,
                "native_head_dim": tc.hidden_size // tc.num_attention_heads,
                "native_kv_bytes_per_token": 2 * tc.num_hidden_layers * tc.num_key_value_heads * 128 * 2,
                "rope_parameters": tc.rope_parameters, "processor_settings": self.processor_settings(),
                "seed": 1234, "batch": 1, "max_new_tokens": self.max_new_tokens,
                "decoding": "greedy", "offload": False}

    def messages(self, question, history=()):
        def qtext(q):
            return str(q) + "\n" + ANSWER_INSTRUCTION
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        first = history[0][0] if history else question
        messages.append({"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": qtext(first)}]})
        if history:
            messages.append({"role": "assistant", "content": str(history[0][1])})
            for previous, answer in history[1:]:
                messages.extend([{"role": "user", "content": qtext(previous)},
                                 {"role": "assistant", "content": str(answer)}])
            messages.append({"role": "user", "content": qtext(question)})
        return messages

    def chat(self, question, history):
        text = self.processor.apply_chat_template(self.messages(question, history),
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if not text.endswith("<|assistant|>\n<think></think>\n"):
            # Native templates may omit the newline before the empty think marker.
            if not text.endswith("<think></think>\n"):
                raise ValueError("native non-thinking generation prompt missing")
        return text

    def positions(self, ids, grid):
        types = (ids == self.model.config.image_token_id).to(torch.long)
        return self.model.model.get_rope_index(ids, types, image_grid_thw=grid,
                                             attention_mask=torch.ones_like(ids))

    def _clear(self):
        self.model.model.rope_deltas = None

    def _sync(self):
        torch.cuda.synchronize(self.device)

    @property
    def eos(self):
        eos = self.model.generation_config.eos_token_id
        return set(eos if isinstance(eos, list) else [eos])

    def decode(self, first, cache, mask, native_start):
        ids = [first]
        initial_cache_len = int(cache.get_seq_length())
        while len(ids) < self.max_new_tokens and ids[-1] not in self.eos:
            mask = torch.cat([mask, torch.ones(1, 1, dtype=mask.dtype, device=self.device)], -1)
            output = self.model(input_ids=torch.tensor([[ids[-1]]], device=self.device),
                    past_key_values=cache, attention_mask=mask,
                    position_ids=torch.full((3, 1, 1), native_start + len(ids) - 1,
                                             device=self.device, dtype=torch.long),
                    use_cache=True, logits_to_keep=1, return_dict=True)
            ids.append(int(output.logits[0, -1].argmax()))
        if int(cache.get_seq_length()) != initial_cache_len + len(ids) - 1:
            raise ValueError("native decode cache advancement differs")
        self._sync()
        return ids

    def result(self, generated, start, first_time, trace, timings, logits=None):
        self._sync()
        decoded_raw = self.processor.tokenizer.decode(generated, skip_special_tokens=False,
                                                     clean_up_tokenization_spaces=False)
        answer, extraction = native_final_answer(self.processor.tokenizer, generated)
        result = {"prediction": answer, "answer_extraction": extraction,
                  "generated_text_with_special_tokens": decoded_raw,
                  "generated_token_ids": generated, "first_token_id": generated[0],
                  "truncated": len(generated) == self.max_new_tokens and generated[-1] not in self.eos,
                  "reasoning_leak": "<think>" in decoded_raw or "</think>" in decoded_raw,
                  "ttft_ms": first_time, "request_e2e_ms": milliseconds(start),
                  "timing_ms": timings, "request_trace": trace,
                  "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(0),
                  "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(0),
                  "gpu_peak_scope": "request preparation through required capture/score tail",
                  "input_ids": trace["input_ids"], "online_query_score_calls": 0,
                  "decode_store_reads": 0, "enable_thinking": False}
        if logits is not None:
            result["first_logits"] = logits
        return result

    @torch.inference_mode()
    def run_pixels(self, image_path, question, history=(), *, capture=False,
                   image_sha256, return_logits=False, request_start=None):
        from PIL import Image
        from .vision import VisionScoreCapture
        self._clear()
        torch.cuda.reset_peak_memory_stats(0)
        start = request_start if request_start is not None else time.perf_counter()
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
        text = self.chat(question, history)
        inputs = self.processor(text=[text], images=[rgb], return_tensors="pt")
        inputs.pop("token_type_ids", None)
        ids, grid = inputs["input_ids"], inputs["image_grid_thw"]
        geometry = prefix_geometry(ids, grid, self.model.config)
        positions, offset = self.positions(ids, grid)
        ref_positions, ref_offset = independent_positions(ids, geometry["content_indices"], grid)
        if not torch.equal(positions, ref_positions) or not torch.equal(offset, ref_offset):
            raise ValueError("native position differs from independent spatial reference")
        input_ids = ids.tolist()
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        cap = VisionScoreCapture(self.model) if capture == "with_score" else None
        count = [0]
        hook = self.model.model.visual.register_forward_pre_hook(
            lambda *_: count.__setitem__(0, count[0] + 1))
        timings = {}
        try:
            if cap:
                cap.__enter__()
            prefill = time.perf_counter()
            output = self.model(**inputs, position_ids=positions.to(self.device),
                                use_cache=True, return_dict=True, logits_to_keep=1)
            token = int(output.logits[0, -1].argmax())
            self._sync()
            first_time = milliseconds(start)
            timings["native_full_prefill_ms"] = milliseconds(prefill)
            logits = output.logits[0, -1].float().cpu().clone() if return_logits else None
            decode_start = time.perf_counter()
            generated = self.decode(token, output.past_key_values, inputs["attention_mask"],
                                    int(positions.max()) + 1)
            timings["decode_ms"] = milliseconds(decode_start)
            captured = None
            if capture:
                clone_start = time.perf_counter()
                p = geometry["prefix_len"]
                pairs = [(k[:, :, :p].detach().clone(), v[:, :, :p].detach().clone())
                         for k, v in cache_pairs(output.past_key_values)]
                self._sync()
                clone_ms = milliseconds(clone_start)
                captured = CapturedPrefix(pairs, ids[0, :p].tolist(), positions[:, :, :p].clone(),
                    offset.tolist(), geometry["content_indices"], image_sha256, geometry,
                    None, clone_ms, 0)
        finally:
            try:
                if cap:
                    cap.__exit__(*sys.exc_info())
            finally:
                hook.remove()
                self._clear()
        if count[0] != 1:
            raise ValueError("normal T1 must perform exactly one full vision pass")
        if captured is not None and cap:
            captured.scores = cap.scores
            captured.score_ms = cap.score_seconds * 1000
            timings.update(importance_ms=captured.score_ms,
                           importance_peak_extra_gpu_bytes=cap.score_peak_extra_gpu_bytes,
                           importance_memory_scope="sampled live allocation above score baseline; request CUDA peak separately measured",
                           vision_qk_capture_bytes=cap.qk_capture_bytes,
                           vision_qk_copy_ms=cap.qk_copy_seconds*1000,
                           capture_prefix_clone_ms=captured.capture_clone_ms,
                           capture_hook_cpu_ms=cap.hook_cpu_seconds * 1000)
        trace = {"input_ids": input_ids, "position_ids": positions.tolist(),
                 "rope_offset": offset.tolist(), "prompt_text": text,
                 "geometry": geometry, "native_prompt_length": ids.shape[1]}
        result = self.result(generated, start, first_time, trace, timings, logits)
        result.update(vision_calls=1, ssd_read_bytes=0, pread_calls=0,
                      kv_store_read_bytes=0, image_file_io="MEASURED_IN_TTFT_BYTES_NOT_MEASURED")
        if captured is not None:
            result["capture"] = captured
        return result

    def identity(self, capture):
        source = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in Path(__file__).parent.glob("*.py")}
        return {"model_id": MODEL_ID, "image_sha256": capture.image_sha256,
                "checkpoint_revision": REVISION, "processor_revision": REVISION,
                "processor_settings": self.processor_settings(), "geometry": capture.geometry,
                "position_policy": POSITION_POLICY, "logical_position_ids": capture.positions.tolist(),
                "key_rope_state": "post_rope", "rope_config": self.model.config.text_config.rope_parameters,
                "rope_offset": capture.rope_offset, "code_revision": digest(source),
                "environment_revision": digest(self.fingerprint())}

    def persist(self, capture, out_dir, *, storage_policy):
        from .store import write_glm_store, NativeKVSpec
        extra = self.identity(capture)
        if storage_policy == "prefix25":
            extra.update(score_source="last_vision_block_post_softmax_received_attention",
                         score_layer=self.model.config.vision_config.depth - 1)
        timings = {}
        start = time.perf_counter()
        meta = write_glm_store(out_dir, capture.layers, capture.content_indices, capture.prefix_ids,
                 scores=capture.scores if storage_policy == "prefix25" else None, extra=extra,
                 storage_policy=storage_policy, shape_spec=NativeKVSpec(40, 2, 128), timing_out=timings)
        outer = milliseconds(start)
        self.expected_identities[str(Path(out_dir).resolve())] = meta["identity"]
        return {"duration_ms": outer, "timing_ms": timings, "metadata": meta,
                "capture_clone_ms_in_request": capture.capture_clone_ms,
                "score_ms_in_request": capture.score_ms}

    def activate(self, out_dir, expected_identity=None):
        from .store import open_glm_store
        key = str(Path(out_dir).resolve())
        expected = expected_identity or self.expected_identities[key]
        if key in self.stores:
            if self.stores[key].meta["identity"] != expected:
                raise ValueError("foreign image or prefix identity")
            return self.stores[key], None
        start = time.perf_counter()
        store = open_glm_store(out_dir, expected_identity=expected)
        self.stores[key] = store
        self.expected_identities[key] = expected
        return store, {"duration_ms": milliseconds(start),
                       "activation_io": store.activation_io.summary()}

    def cache_inputs(self, meta, question, history):
        text = self.chat(question, history)
        raw_ids = self.processor.tokenizer(text, return_tensors="pt")["input_ids"]
        ends = (raw_ids[0] == self.model.config.image_end_token_id).nonzero().flatten()
        if ends.numel() != 1:
            raise ValueError("native suffix has wrong image boundary")
        end = int(ends[0])
        prefix = meta["prefix_input_ids"]
        content = meta["content_logical_indices"]
        collapsed = prefix[:content[0]] + [self.model.config.image_token_id] + prefix[content[-1]+1:]
        if raw_ids[0, :end+1].tolist() != collapsed:
            raise ValueError("system/template/prefix identity mismatch")
        suffix = raw_ids[:, end+1:]
        full = torch.cat([torch.tensor(prefix).reshape(1, -1), suffix], -1)
        grid = torch.tensor(meta["identity"]["geometry"]["image_grid_thw"])
        positions, offset = self.positions(full, grid)
        ref_positions, ref_offset = independent_positions(full, content, grid)
        if not torch.equal(positions, ref_positions) or not torch.equal(offset, ref_offset):
            raise ValueError("native history/suffix position differs from independent reference")
        if offset.tolist() != meta["rope_offset"]:
            raise ValueError("native suffix reconstruction changed stored RoPE offset")
        if positions[:, 0, :len(prefix)].tolist() != meta["logical_position_ids"]:
            raise ValueError("stored native prefix position differs")
        return full, suffix, positions, offset, text

    @torch.inference_mode()
    def run_cache(self, out_dir, question, history=(), *, return_logits=False, memory_layers=None,
                  request_start=None):
        try:
            return self._run_cache(out_dir, question, history, return_logits=return_logits,
                                   memory_layers=memory_layers, request_start=request_start)
        finally:
            self._clear()

    def _run_cache(self, out_dir, question, history=(), *, return_logits=False, memory_layers=None,
                   request_start=None):
        from transformers import DynamicCache
        key = str(Path(out_dir).resolve())
        if key not in self.stores:
            raise ValueError("activate and account integrity setup before requests")
        store = self.stores[key]
        self._clear()
        torch.cuda.reset_peak_memory_stats(0)
        start = request_start if request_start is not None else time.perf_counter()
        full, suffix, positions, offset, text = self.cache_inputs(store.meta, question, history)
        load_start = time.perf_counter()
        loaded = store.load_prefix() if memory_layers is None else memory_layers
        read_ms = milliseconds(load_start)
        selected = store.meta["stored_to_original"]
        content = store.meta["content_logical_indices"]
        expected_logical = sorted(store.meta["structural_indices"] + [content[i] for i in selected])
        expected_pos = torch.tensor(store.meta["logical_position_ids"])[:, expected_logical]
        if (list(loaded.logical_indices) != expected_logical
            or not torch.equal(loaded.position_ids.reshape(3, -1), expected_pos)
            or loaded.native_prefix_length != store.meta["prefix_len"]
            or loaded.rope_offset != store.meta["rope_offset"]):
            raise ValueError("loaded native IDs/positions/offset do not match frozen store")
        if len(loaded.layers) != 40 or any(
            tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != (1, 2, len(expected_logical), 128)
            for pair in loaded.layers for tensor in pair):
            raise ValueError("loaded native all-layer/head KV geometry differs")
        cache = DynamicCache(config=self.model.config.text_config)
        h2d_start = time.perf_counter()
        gpu_pairs = [(k.to(self.device), v.to(self.device)) for k, v in loaded.layers]
        self._sync()
        h2d_ms = milliseconds(h2d_start)
        assembly_start = time.perf_counter()
        for layer, (k, v) in enumerate(gpu_pairs):
            cache.update(k, v, layer)
        self._sync()
        assembly_ms = milliseconds(assembly_start)
        compact = int(cache.get_seq_length())
        p = len(store.meta["prefix_input_ids"])
        if compact != len(loaded.logical_indices):
            raise ValueError("compact native cache length differs")
        if not suffix.numel():
            raise ValueError("missing current question suffix")
        mask = torch.ones(1, compact + suffix.shape[1], dtype=torch.long, device=self.device)
        prefill_start = time.perf_counter()
        output = self.model(input_ids=suffix.to(self.device), past_key_values=cache,
                attention_mask=mask, position_ids=positions[:, :, p:].to(self.device),
                use_cache=True, return_dict=True, logits_to_keep=1)
        token = int(output.logits[0, -1].argmax())
        self._sync()
        first_time = milliseconds(start)
        prefill_ms = milliseconds(prefill_start)
        logits = output.logits[0, -1].float().cpu().clone() if return_logits else None
        generated = self.decode(token, cache, mask, int(positions.max()) + 1)
        trace = {"input_ids": full.tolist(), "position_ids": positions.tolist(),
                 "rope_offset": offset.tolist(), "prompt_text": text,
                 "logical_indices": loaded.logical_indices,
                 "compact_prefix_length": compact, "native_prefix_length": p,
                 "native_suffix_first_position": positions[:, 0, p].tolist(),
                 "mask_policy": "all-prefix-visible/physical-suffix-causal",
                 "decode_final_cache_length": int(cache.get_seq_length())}
        io = loaded.io.summary()
        result = self.result(generated, start, first_time, trace,
                    {"store_load_ms": read_ms, "h2d_ms": h2d_ms,
                     "assembly_ms": assembly_ms, "suffix_prefill_ms": prefill_ms}, logits)
        result.update(vision_calls=0, read_io=io, selected_content_original=loaded.selected_content_original,
                      N_content=store.meta["original_content_count"],
                      k=store.meta["stored_content_count"],
                      ssd_read_bytes=io["bytes"], pread_calls=io["preads"],
                      h2d_kv_bytes=sum((k.numel()+v.numel())*k.element_size() for k,v in loaded.layers),
                      image_file_io="NONE_ON_CACHE_REQUEST", padding_read_bytes=0)
        self._clear()
        return result

    def close(self):
        for store in self.stores.values():
            store.close()
        self.stores.clear()
        self.expected_identities.clear()

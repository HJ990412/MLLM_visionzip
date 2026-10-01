"""Single-image Qwen2.5-VL pixel and SSD-prefix inference.

The normal first-turn forward processes the *entire* multimodal prompt once.
Its decoder cache is then sliced to the reusable image prefix. The cache-hit
path never invokes the vision encoder. Qwen's post-MRoPE native KV head tensors
are used directly, while the suffix keeps full-prompt logical MRoPE positions.
"""
from __future__ import annotations

import hashlib
import sys
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
CHECKPOINT_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28
SYSTEM_PROMPT = "You are a helpful assistant. Give concise factual answers."
ANSWER_INSTRUCTION = "Answer using a single word or short phrase."
POSITION_POLICY = "full-prompt-get_rope_index/post-MRoPE-prefix/compact-cache-slots-v1"
MAX_NEW_TOKENS = 16
SEED = 1234
# Digest of the three protected Qwen source files before Visual-KV25 was added.
# A legacy store still requires its exact frozen meta.json SHA256 to be
# explicitly registered at construction and all normal activation checks.
LEGACY_CHUNK25_CODE_REVISION = (
    "9f193bacd446d4a88a662fb77d8789ea22c9e42962ed10a7d2e2986c8127e191")


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cache_layers(cache):
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    return list(cache)


def _sha_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _pil_rgb_sha256(image) -> str:
    """Hash decoded RGB content including its geometry for PIL-only callers."""
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(b"PIL-RGB-v1\0")
    digest.update(int(rgb.width).to_bytes(8, "big"))
    digest.update(int(rgb.height).to_bytes(8, "big"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _one_span(input_ids: torch.Tensor, token_id: int) -> tuple[int, int]:
    ids = input_ids.reshape(-1)
    positions = (ids == token_id).nonzero(as_tuple=True)[0]
    if not positions.numel():
        raise AssertionError("image token is absent")
    first = int(positions[0])
    count = int(positions.numel())
    if not torch.equal(positions, torch.arange(first, first + count,
                                               device=positions.device)):
        raise AssertionError("single image tokens are not contiguous")
    return first, count


@dataclass
class CapturedPrefix:
    layers: list[tuple[torch.Tensor, torch.Tensor]]
    prefix_ids: list[int]
    visual_start: int
    visual_count: int
    image_sha256: str
    image_grid_thw: list[list[int]]
    logical_position_ids: list[list[int]]
    rope_deltas: list[list[int]]
    scores: torch.Tensor | None
    geometry: dict
    score_ms: float
    capture_clone_ms: float


class Qwen25Runner:
    def __init__(self, *, model_id: str = MODEL_ID,
                 revision: str = CHECKPOINT_REVISION,
                 min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS,
                 attn: str = "sdpa", use_fast: bool = True,
                 max_new_tokens: int = MAX_NEW_TOKENS,
                 trusted_legacy_store_meta_sha256: dict[str | Path, str] | None = None):
        if model_id != MODEL_ID:
            raise ValueError(f"this adapter is pinned to {MODEL_ID}")
        self.model_id = model_id
        self.revision = revision
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self.attn = attn
        self.use_fast = bool(use_fast)
        self.max_new_tokens = int(max_new_tokens)
        self.model = None
        self.processor = None
        self.device = None
        self._stores = {}
        self.activation_records = {}
        self._code_revision_cache = None
        self._environment_revision_cache = None
        self.trusted_legacy_store_meta_sha256 = {
            str(Path(path).resolve()): str(digest).lower()
            for path, digest in (trusted_legacy_store_meta_sha256 or {}).items()
        }
        if any(len(digest) != 64 or any(c not in "0123456789abcdef"
                                       for c in digest)
               for digest in self.trusted_legacy_store_meta_sha256.values()):
            raise ValueError("trusted legacy metadata digests must be SHA256 hex")

    def load(self):
        from transformers import (AutoProcessor, BitsAndBytesConfig,
                                  Qwen2_5_VLForConditionalGeneration)

        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, revision=self.revision, local_files_only=True,
            quantization_config=quant, dtype=torch.bfloat16,
            device_map="cuda:0", attn_implementation=self.attn).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, revision=self.revision, local_files_only=True,
            min_pixels=self.min_pixels, max_pixels=self.max_pixels,
            use_fast=self.use_fast)
        self.device = next(self.model.parameters()).device
        if self.device.type != "cuda":
            raise RuntimeError("Qwen weights must reside on CUDA")
        if any(p.device.type != "cuda" for p in self.model.parameters()):
            raise RuntimeError("CPU/disk weight offload is forbidden")
        tc = self.model.config.text_config
        vc = self.model.config.vision_config
        if len(self.model.model.language_model.layers) != tc.num_hidden_layers:
            raise AssertionError("decoder layer count differs from config")
        if vc.depth - 1 not in vc.fullatt_block_indexes:
            raise AssertionError("last vision block must be full-attention")
        if self.processor.image_processor.min_pixels != self.min_pixels \
                or self.processor.image_processor.max_pixels != self.max_pixels:
            raise AssertionError("processor pixel settings changed")
        return self

    def runtime_fingerprint(self) -> dict:
        if self.model is None:
            raise RuntimeError("call load() first")
        import bitsandbytes as bnb
        import transformers
        tc, vc = self.model.config.text_config, self.model.config.vision_config
        quantized = [n for n, m in self.model.named_modules()
                     if isinstance(m, bnb.nn.Linear4bit)]
        return {
            "model_id": self.model_id, "checkpoint_revision": self.revision,
            "processor_revision": self.revision, "tokenizer_revision": self.revision,
            "transformers": transformers.__version__, "torch": torch.__version__,
            "bitsandbytes": bnb.__version__, "attention_backend": self.attn,
            "text_attention_backend": tc._attn_implementation,
            "vision_attention_backend": vc._attn_implementation,
            "hf_device_map": {str(k): str(v) for k, v in
                              (getattr(self.model, "hf_device_map", None) or {}).items()},
            "quantized_module_count": len(quantized),
            "quantized_module_scope_counts": {
                "vision": sum(".visual." in f".{n}." for n in quantized),
                "language_model": sum(".language_model." in f".{n}." for n in quantized),
                "other": sum(not (".visual." in f".{n}." or
                                  ".language_model." in f".{n}.") for n in quantized),
            },
            "quantized_module_prefixes": sorted({n.split(".")[1] if "." in n else n
                                                   for n in quantized}),
            "gpu_name": torch.cuda.get_device_name(self.device),
            "gpu_total_bytes": torch.cuda.get_device_properties(self.device).total_memory,
            "seed": SEED, "decoding": "greedy",
            "max_new_tokens": self.max_new_tokens,
            "eos_token_ids": sorted(self._eos_ids),
            "decoder_layers": tc.num_hidden_layers,
            "query_heads": tc.num_attention_heads,
            "kv_heads": tc.num_key_value_heads,
            "head_dim": tc.hidden_size // tc.num_attention_heads,
            "vision_heads": vc.num_heads,
            "vision_fullatt_block_indexes": vc.fullatt_block_indexes,
            "spatial_merge_size": vc.spatial_merge_size,
            "processor_settings": self.processor_settings(),
            "weight_offload": False,
        }

    def _code_revision(self) -> str:
        if self._code_revision_cache is None:
            files = [Path(__file__), Path(__file__).with_name("vision.py"),
                     Path(__file__).with_name("store.py")]
            self._code_revision_cache = hashlib.sha256(b"".join(
                hashlib.sha256(path.read_bytes()).digest() for path in files
            )).hexdigest()
        return self._code_revision_cache

    def _environment_revision(self) -> str:
        if self._environment_revision_cache is None:
            self._environment_revision_cache = _sha_json(self.runtime_fingerprint())
        return self._environment_revision_cache

    def processor_settings(self) -> dict:
        return {"min_pixels": self.min_pixels, "max_pixels": self.max_pixels,
                "use_fast": self.use_fast, "processor_revision": self.revision}

    def _messages(self, question: str,
                  history: Sequence[tuple[str, str]] = ()) -> list[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if not history:
            messages.append({"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": self._question_text(question)}]})
            return messages
        first_question, first_answer = history[0]
        messages.append({"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": self._question_text(first_question)}]})
        messages.append({"role": "assistant", "content": str(first_answer)})
        for previous_question, previous_answer in history[1:]:
            messages.append({"role": "user", "content": self._question_text(previous_question)})
            messages.append({"role": "assistant", "content": str(previous_answer)})
        messages.append({"role": "user", "content": self._question_text(question)})
        return messages

    @staticmethod
    def _question_text(question: str) -> str:
        return f"{question}\n{ANSWER_INSTRUCTION}"

    def _chat_text(self, question, history):
        return self.processor.apply_chat_template(
            self._messages(question, history), tokenize=False,
            add_generation_prompt=True)

    def _image_geometry(self, ids: torch.Tensor, grid: torch.Tensor,
                        pixel_values: torch.Tensor) -> dict:
        vc = self.model.config.vision_config
        if grid.shape != (1, 3):
            raise AssertionError("this port is single-image only")
        start, count = _one_span(ids, self.model.config.image_token_id)
        t, h, w = [int(x) for x in grid[0]]
        patches = t * h * w
        merged = patches // (vc.spatial_merge_size ** 2)
        if patches % (vc.spatial_merge_size ** 2) or count != merged:
            raise AssertionError((patches, merged, count))
        if pixel_values.shape[0] != patches:
            raise AssertionError((pixel_values.shape, patches))
        end = start + count
        if int(ids[0, start - 1]) != self.model.config.vision_start_token_id:
            raise AssertionError("vision_start is not immediately before expanded image tokens")
        if int(ids[0, end]) != self.model.config.vision_end_token_id:
            raise AssertionError("vision_end is not immediately after expanded image tokens")
        return {"image_grid_thw": [[t, h, w]],
                "pre_merger_patches": patches, "merger_tokens": merged,
                "expanded_image_tokens": count, "visual_kv_rows": count,
                "resized_hw": [h * vc.patch_size, w * vc.patch_size],
                "vision_start_index": start - 1, "visual_start": start,
                "visual_count": count, "vision_end_index": end,
                "prefix_len": end + 1}

    def _logical_positions(self, ids: torch.Tensor, grid: torch.Tensor):
        positions, deltas = self.model.model.get_rope_index(
            input_ids=ids, image_grid_thw=grid,
            attention_mask=torch.ones_like(ids))
        if positions.shape != (3, 1, ids.shape[1]):
            raise AssertionError(positions.shape)
        return positions, deltas

    def _clear_request_state(self):
        self.model.model.rope_deltas = None

    @property
    def _eos_ids(self) -> set[int]:
        eos = self.model.generation_config.eos_token_id
        return {int(v) for v in (eos if isinstance(eos, (tuple, list)) else [eos])}

    def _decode(self, first_id: int, cache, *, position_start: int,
                attention_mask: torch.Tensor) -> tuple[list[int], float]:
        generated = [int(first_id)]
        t0 = time.perf_counter()
        while len(generated) < self.max_new_tokens and generated[-1] not in self._eos_ids:
            current = generated[-1]
            pos = position_start + len(generated) - 1
            compact_position = int(cache.get_seq_length())
            next_mask = torch.cat([attention_mask,
                                   torch.ones((1, 1), dtype=attention_mask.dtype,
                                              device=self.device)], dim=-1)
            output = self.model(
                input_ids=torch.tensor([[current]], device=self.device),
                attention_mask=next_mask,
                position_ids=torch.full((3, 1, 1), pos, dtype=torch.long,
                                        device=self.device),
                cache_position=torch.tensor([compact_position], device=self.device),
                past_key_values=cache, use_cache=True, return_dict=True,
                logits_to_keep=1)
            generated.append(int(output.logits[0, -1].argmax()))
            attention_mask = next_mask
        _synchronize(self.device)
        return generated, _ms(t0)

    def _result(self, generated: list[int], *, ttft_ms: float,
                request_e2e_ms: float, timing_ms: dict, vision_calls: int,
                peak_allocated: int, peak_reserved: int,
                first_logits: torch.Tensor | None = None) -> dict:
        prediction = self.processor.tokenizer.decode(
            generated, skip_special_tokens=True,
            clean_up_tokenization_spaces=False).strip()
        result = {
            "prediction": prediction, "first_token_id": generated[0],
            "generated_token_ids": generated,
            "generated_token_count": len(generated),
            "truncated": len(generated) == self.max_new_tokens
                         and generated[-1] not in self._eos_ids,
            "ttft_ms": ttft_ms, "request_e2e_ms": request_e2e_ms,
            "timing_ms": timing_ms, "vision_calls": vision_calls,
            "online_query_score_calls": 0,
            "peak_gpu_allocated_bytes": peak_allocated,
            "peak_gpu_reserved_bytes": peak_reserved,
        }
        if first_logits is not None:
            result["first_logits"] = first_logits
        return result

    @torch.inference_mode()
    def run_pixels(self, image, question: str,
                   history: Sequence[tuple[str, str]] = (), *,
                   capture: bool | str = False, image_sha256: str | None = None,
                   return_logits: bool = False) -> dict:
        if self.model is None:
            raise RuntimeError("call load() first")
        from PIL import Image
        from .vision import VisionScoreCapture

        if isinstance(image, (str, Path)):
            path = Path(image)
            if image_sha256 is None:
                image_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            with Image.open(path) as src:
                image = src.convert("RGB")
        elif image_sha256 is None:
            image_sha256 = _pil_rgb_sha256(image)
        if not isinstance(image_sha256, str) or len(image_sha256) != 64:
            raise ValueError("image_sha256 must be a SHA256 hex digest")
        if capture not in (False, True, "kv_only", "with_score"):
            raise ValueError("capture must be False, kv_only, or with_score")
        want_capture = bool(capture)
        want_score = capture is True or capture == "with_score"
        self._clear_request_state()
        torch.cuda.reset_peak_memory_stats(self.device)
        request_t0 = time.perf_counter()
        prompt_t0 = time.perf_counter()
        text = self._chat_text(question, history)
        enc = self.processor(text=[text], images=[image], return_tensors="pt")
        ids_cpu = enc["input_ids"]
        geometry = self._image_geometry(ids_cpu, enc["image_grid_thw"],
                                        enc["pixel_values"])
        positions_cpu, deltas_cpu = self._logical_positions(
            ids_cpu, enc["image_grid_thw"])
        prompt_ms = _ms(prompt_t0)
        enc = {k: (v.to(self.device) if torch.is_tensor(v) else v)
               for k, v in enc.items()}
        positions = positions_cpu.to(self.device)
        ids = enc["input_ids"]
        vision_calls = [0]
        hook = self.model.visual.register_forward_pre_hook(
            lambda *_: vision_calls.__setitem__(0, vision_calls[0] + 1))
        cap = VisionScoreCapture(self.model) if want_score else None
        first_logits = None
        try:
            if cap is not None:
                cap.__enter__()
            prefill_t0 = time.perf_counter()
            output = self.model(
                **enc, position_ids=positions,
                cache_position=torch.arange(ids.shape[1], device=self.device),
                use_cache=True, return_dict=True, logits_to_keep=1)
            _synchronize(self.device)
            prefill_ms = _ms(prefill_t0)
            token = int(output.logits[0, -1].argmax())
            _synchronize(self.device)
            ttft_ms = _ms(request_t0)
            if return_logits:
                first_logits = output.logits[0, -1].float().cpu().clone()
            generated, decode_ms = self._decode(
                token, output.past_key_values,
                position_start=int(positions_cpu.max()) + 1,
                attention_mask=enc.get("attention_mask", torch.ones_like(ids)))
            request_e2e_ms = _ms(request_t0)
            request_peak_allocated = torch.cuda.max_memory_allocated(self.device)
            request_peak_reserved = torch.cuda.max_memory_reserved(self.device)
        finally:
            if cap is not None:
                try:
                    cap.__exit__(*sys.exc_info())
                finally:
                    hook.remove()
                    self._clear_request_state()
            else:
                hook.remove()
                self._clear_request_state()
        if vision_calls[0] != 1:
            raise AssertionError(f"expected one vision forward, saw {vision_calls[0]}")
        result = self._result(
            generated, ttft_ms=ttft_ms, request_e2e_ms=request_e2e_ms,
            timing_ms={"prompt_preparation": prompt_ms, "multimodal_prefill": prefill_ms,
                       "decode": decode_ms}, vision_calls=vision_calls[0],
            peak_allocated=max(request_peak_allocated,
                               torch.cuda.max_memory_allocated(self.device)),
            peak_reserved=max(request_peak_reserved,
                              torch.cuda.max_memory_reserved(self.device)),
            first_logits=first_logits)
        result["geometry"] = geometry
        result["rope_deltas"] = deltas_cpu.tolist()
        result["image_sha256"] = image_sha256
        if want_capture:
            if want_score:
                if cap.call_count != 1 or cap.scores is None:
                    raise AssertionError("vision saliency capture was not exactly once")
                if cap.scores.numel() != geometry["visual_count"]:
                    raise AssertionError("saliency count differs from visual token count")
            clone_t0 = time.perf_counter()
            prefix_len = geometry["prefix_len"]
            layers = [(k[:, :, :prefix_len, :].detach().clone(),
                       v[:, :, :prefix_len, :].detach().clone())
                      for k, v in _cache_layers(output.past_key_values)]
            _synchronize(self.device)
            clone_ms = _ms(clone_t0)
            tc = self.model.config.text_config
            for k, v in layers:
                if k.shape != (1, tc.num_key_value_heads, prefix_len,
                               tc.hidden_size // tc.num_attention_heads) \
                        or v.shape != k.shape or k.dtype != torch.bfloat16:
                    raise AssertionError("native BF16 GQA cache shape/dtype mismatch")
            result["capture"] = CapturedPrefix(
                layers=layers,
                prefix_ids=ids_cpu[0, :prefix_len].tolist(),
                visual_start=geometry["visual_start"],
                visual_count=geometry["visual_count"],
                image_sha256=image_sha256,
                image_grid_thw=geometry["image_grid_thw"],
                logical_position_ids=positions_cpu[:, 0, :prefix_len].tolist(),
                rope_deltas=deltas_cpu.tolist(),
                scores=cap.scores if want_score else None,
                geometry=geometry,
                score_ms=cap.score_seconds * 1000.0 if want_score else 0.0,
                capture_clone_ms=clone_ms)
            result["score_extra_ms"] = cap.score_seconds * 1000.0 if want_score else 0.0
            result["score_peak_extra_gpu_bytes"] = (
                cap.score_peak_extra_gpu_bytes if want_score else 0)
            result["score_peak_gpu_allocated_bytes"] = (
                cap.score_peak_gpu_allocated_bytes if want_score else 0)
            result["capture_clone_ms"] = clone_ms
            result["score_source"] = "last_fullatt_ViT_received_attention" if want_score else None
            result["score_layer"] = self.model.config.vision_config.depth - 1 if want_score else None
        return result

    def persist(self, capture: CapturedPrefix, store_dir: str | Path,
                layout: str = "repacked", *, storage_policy: str = "full") -> dict:
        from .store import write_qwen_store
        if layout not in {"canonical", "repacked"}:
            raise ValueError(layout)
        t0 = time.perf_counter()
        timing = {}
        code_revision = self._code_revision()
        environment_revision = self._environment_revision()
        extra = {
            "image_sha256": capture.image_sha256,
            "processor_revision": self.revision,
            "code_revision": code_revision,
            "environment_revision": environment_revision,
            "key_rope_state": "post_mrope",
            "checkpoint_revision": self.revision,
            "processor_settings": self.processor_settings(),
            "image_grid_thw": capture.image_grid_thw,
            "geometry": capture.geometry,
            "position_policy": POSITION_POLICY,
            "logical_position_ids": capture.logical_position_ids,
            "rope_deltas": capture.rope_deltas,
            "score_source": "last_fullatt_ViT_received_attention" if layout == "repacked" else None,
            "score_layer": self.model.config.vision_config.depth - 1 if layout == "repacked" else None,
            "capture_score_ms": capture.score_ms if layout == "repacked" else 0.0,
        }
        meta = write_qwen_store(
            Path(store_dir), capture.layers, capture.visual_start,
            capture.visual_count, capture.prefix_ids,
            scores=capture.scores if layout == "repacked" else None,
            extra=extra, timing_out=timing, storage_policy=storage_policy)
        writer_ms = _ms(t0)
        return {"store_dir": str(store_dir), "layout": layout,
                "persistence_ms": writer_ms + capture.score_ms + capture.capture_clone_ms,
                "writer_persistence_ms": writer_ms, "timing_ms": timing,
                "capture_score_ms": capture.score_ms if layout == "repacked" else 0.0,
                "capture_clone_ms": capture.capture_clone_ms,
                "metadata": meta}

    def _activate(self, store_dir: str | Path, image_sha256: str | None = None):
        from .store import open_qwen_store
        key = str(Path(store_dir).resolve())
        if key not in self._stores:
            if image_sha256 is None:
                raise ValueError("cache activation requires image_sha256")
            expected_identity = {
                "image_sha256": image_sha256,
                "checkpoint_revision": self.revision,
                "processor_settings": self.processor_settings(),
                "position_policy": POSITION_POLICY,
                "environment_revision": self._environment_revision(),
            }
            t0 = time.perf_counter()
            store = open_qwen_store(Path(store_dir),
                                    expected_identity=expected_identity,
                                    verify=True)
            try:
                meta = store.meta
                self._validate_store_meta(meta, image_sha256,
                                          store_path=key,
                                          metadata_sha256=store.metadata_file_sha256)
            except Exception:
                store.close()
                raise
            self._stores[key] = store
            self.activation_records[key] = {
                "activation_ms": _ms(t0),
                "activation_io": store.activation_io.summary(),
                "metadata_resident_bytes": store.metadata_resident_bytes,
            }
        else:
            store = self._stores[key]
            self._validate_store_meta(store.meta, image_sha256,
                                      store_path=key,
                                      metadata_sha256=store.metadata_file_sha256)
        return store

    def _validate_store_meta(self, meta: dict, image_sha256: str | None,
                             *, store_path: str, metadata_sha256: str):
        identity = meta.get("identity", meta)
        expected = {
            "checkpoint_revision": self.revision,
            "processor_settings": self.processor_settings(),
            "position_policy": POSITION_POLICY,
        }
        if image_sha256 is None:
            raise ValueError("cache activation requires image_sha256")
        expected["image_sha256"] = image_sha256
        for key, value in expected.items():
            if identity.get(key, meta.get(key)) != value:
                raise ValueError(f"cache identity mismatch for {key}")
        if meta["dtype"] != "bfloat16":
            raise AssertionError("Qwen store must retain native BF16 bits")
        code_revision = meta.get("code_revision")
        if code_revision != self._code_revision():
            trusted_digest = self.trusted_legacy_store_meta_sha256.get(store_path)
            if code_revision != LEGACY_CHUNK25_CODE_REVISION \
                    or trusted_digest != metadata_sha256:
                raise ValueError("cache code revision is not an approved protected store")
        if meta.get("environment_revision") != self._environment_revision():
            raise ValueError("cache environment revision differs from active runtime")

    def condition_cache(self, store_dir: str | Path, ratio: float = 0.25,
                        *, budget_unit: str = "chunk",
                        image_sha256: str | None = None) -> dict:
        """Best-effort OS page-cache DONTNEED outside the request timer."""
        key = str(Path(store_dir).resolve())
        fresh_activation = key not in self._stores
        store = self._activate(store_dir, image_sha256)
        meta = store.meta
        from .store import plan_prefix_budget
        plan = plan_prefix_budget(meta, ratio, budget_unit=budget_unit)
        conditioned = store.drop_payload_cache()
        activation = self.activation_records[key]
        return {"success": conditioned["attempted"] - len(conditioned["failed"]),
                "failures": len(conditioned["failed"]),
                "failure_details": conditioned["failed"],
                "selected_chunks": plan["selected_chunks"],
                "target_visual_tokens": plan["kept_visual_tokens"],
                "budget_unit": budget_unit,
                "guarantee": "OS page-cache hint only",
                "activation_ms": activation["activation_ms"] if fresh_activation else 0.0,
                "activation_io": activation["activation_io"] if fresh_activation else None,
                "metadata_resident_bytes": activation["metadata_resident_bytes"]}

    def _cache_hit_ids(self, meta: dict, question: str, history):
        raw_text = self._chat_text(question, history)
        raw_ids = self.processor.tokenizer(raw_text, return_tensors="pt")["input_ids"]
        raw_prefix_end = (raw_ids[0] == self.model.config.vision_end_token_id).nonzero(as_tuple=True)[0]
        if raw_prefix_end.numel() != 1:
            raise AssertionError("expected one vision_end in cache-hit chat template")
        raw_prefix_end = int(raw_prefix_end[0])
        prefix_ids = [int(x) for x in meta["prefix_input_ids"]]
        vstart = int(meta["visual_start"])
        vcount = int(meta["visual_count"])
        expected_raw_prefix = (prefix_ids[:vstart]
                               + [self.model.config.image_token_id]
                               + prefix_ids[vstart + vcount:])
        if raw_ids[0, :raw_prefix_end + 1].tolist() != expected_raw_prefix:
            raise ValueError("chat-template prefix IDs differ from stored prefix")
        suffix = raw_ids[:, raw_prefix_end + 1:]
        full_ids = torch.cat([torch.tensor(prefix_ids).view(1, -1), suffix], dim=1)
        grid = torch.tensor(meta["image_grid_thw"], dtype=torch.long)
        positions, deltas = self._logical_positions(full_ids, grid)
        stored_positions = torch.tensor(meta["logical_position_ids"], dtype=torch.long)
        if not torch.equal(positions[:, 0, :len(prefix_ids)], stored_positions):
            raise ValueError("stored prefix MRoPE positions differ")
        return full_ids, suffix, positions, deltas

    @torch.inference_mode()
    def run_cache(self, store_dir: str | Path, question: str,
                  history: Sequence[tuple[str, str]] = (), *,
                  budget_ratio: float = 0.25, budget_unit: str = "chunk",
                  image_sha256: str | None = None,
                  dense_reference: bool = False,
                  return_logits: bool = False) -> dict:
        from transformers import DynamicCache
        store = self._activate(store_dir, image_sha256)
        meta = store.meta
        self._clear_request_state()
        torch.cuda.reset_peak_memory_stats(self.device)
        request_t0 = time.perf_counter()
        prompt_t0 = time.perf_counter()
        full_ids, suffix, positions, deltas = self._cache_hit_ids(
            meta, question, history)
        prompt_ms = _ms(prompt_t0)
        if suffix.numel() == 0:
            raise AssertionError("cache hit requires a nonempty question suffix")
        read_t0 = time.perf_counter()
        loaded = store.load_prefix(budget=budget_ratio,
                                   budget_unit=budget_unit)
        read_ms = _ms(read_t0)
        original_indices = [int(x) for x in loaded.logical_indices]
        if original_indices != sorted(original_indices) or len(original_indices) != len(set(original_indices)):
            raise AssertionError("compact cache rows are not unique/original-position ordered")
        if any(i < 0 or i >= meta["prefix_len"] for i in original_indices):
            raise AssertionError("compact cache logical index out of bounds")
        assembly_t0 = time.perf_counter()
        cache = DynamicCache(config=self.model.config.text_config)
        for li, (k, v) in enumerate(loaded.layers):
            if dense_reference:
                n = int(meta["prefix_len"])
                dk = torch.zeros((1, k.shape[1], n, k.shape[3]),
                                 dtype=k.dtype, device=self.device)
                dv = torch.zeros_like(dk)
                idx = torch.tensor(original_indices, device=self.device)
                dk.index_copy_(2, idx, k.to(self.device))
                dv.index_copy_(2, idx, v.to(self.device))
                cache.update(dk, dv, li)
            else:
                cache.update(k.to(self.device), v.to(self.device), li)
        _synchronize(self.device)
        assembly_ms = _ms(assembly_t0)
        compact_len = int(cache.get_seq_length())
        expected_cache_len = (int(meta["prefix_len"]) if dense_reference else
                              loaded.kept_visual_tokens + loaded.structural_rows)
        if compact_len != expected_cache_len or len(original_indices) != (
                loaded.kept_visual_tokens + loaded.structural_rows):
            raise AssertionError("cache contains wrong number of selected rows")
        if dense_reference:
            prefix_mask = torch.zeros((1, meta["prefix_len"]), dtype=torch.long,
                                      device=self.device)
            prefix_mask[0, original_indices] = 1
        else:
            prefix_mask = torch.ones((1, compact_len), dtype=torch.long,
                                     device=self.device)
        mask = torch.cat([prefix_mask,
                          torch.ones((1, suffix.shape[1]), dtype=torch.long,
                                     device=self.device)], dim=1)
        suffix_device = suffix.to(self.device)
        suffix_pos = positions[:, :, meta["prefix_len"]:].to(self.device)
        prefill_t0 = time.perf_counter()
        output = self.model(
            input_ids=suffix_device, attention_mask=mask,
            position_ids=suffix_pos,
            cache_position=torch.arange(compact_len, compact_len + suffix.shape[1],
                                        device=self.device),
            past_key_values=cache, use_cache=True, return_dict=True,
            logits_to_keep=1)
        _synchronize(self.device)
        prefill_ms = _ms(prefill_t0)
        token = int(output.logits[0, -1].argmax())
        _synchronize(self.device)
        ttft_ms = _ms(request_t0)
        logits = output.logits[0, -1].float().cpu().clone() if return_logits else None
        generated, decode_ms = self._decode(
            token, cache, position_start=int(positions.max()) + 1,
            attention_mask=mask)
        request_e2e_ms = _ms(request_t0)
        self._clear_request_state()
        io = loaded.io.summary()
        result = self._result(
            generated, ttft_ms=ttft_ms, request_e2e_ms=request_e2e_ms,
            timing_ms={"prompt_preparation": prompt_ms,
                       "store_load_inclusive": read_ms,
                       "raw_pread": loaded.io.summary()["ms"],
                       "fixed_first_k_planning": loaded.planning_ms,
                       "h2d_and_assembly": assembly_ms,
                       "suffix_prefill": prefill_ms, "decode": decode_ms},
            vision_calls=0,
            peak_allocated=torch.cuda.max_memory_allocated(self.device),
            peak_reserved=torch.cuda.max_memory_reserved(self.device),
            first_logits=logits)
        result.update({
            "visual_read_bytes": io["per_kind"].get("visual", {}).get("bytes", 0),
            "structural_read_bytes": io["per_kind"].get("structural", {}).get("bytes", 0),
            "metadata_read_bytes": 0,
            "pread_calls": io["preads"], "read_spans": io["spans"],
            "read_io": io,
            "raw_pread_ms": io["ms"],
            "planning_ms": loaded.planning_ms,
            "selector_ms": 0.0,
            "selected_chunks": loaded.selected_chunks,
            "normal_chunks_read": loaded.selected_chunks,
            "padding_rows_read": loaded.padding_rows_read,
            "budget_unit": budget_unit,
            "target_visual_tokens": loaded.kept_visual_tokens,
            "loaded_valid_visual_rows": loaded.loaded_valid_visual_rows,
            "extra_valid_visual_rows": loaded.extra_valid_visual_rows,
            "selected_visual_stored": list(loaded.selected_visual_stored),
            "selected_visual_original": list(loaded.selected_visual_original),
            "structural_count": loaded.structural_rows,
            "h2d_kv_bytes": loaded.h2d_kv_bytes,
            "gpu_cache_kv_bytes": sum((k.numel() + v.numel()) * k.element_size()
                                      for k, v in loaded.layers)
                                  if not dense_reference else
                                  2 * len(loaded.layers) * meta["prefix_len"]
                                  * meta["row_bytes"],
            "visual_payload_read_ratio": loaded.visual_payload_read_ratio,
            "total_payload_read_ratio": loaded.total_payload_read_ratio,
        })
        result.update({
            "kept_tokens": len(loaded.selected_visual_original),
            "total_visual_tokens": meta["visual_count"],
            "actual_kept_ratio": len(loaded.selected_visual_original) / meta["visual_count"],
            "structural_inclusive_kept_ratio": (
                len(loaded.selected_visual_original) + loaded.structural_rows)
                / meta["prefix_len"],
            "compact_prefix_tokens": compact_len,
            "dense_reference": dense_reference,
            "rope_deltas": deltas.tolist(),
            "logical_suffix_first_position": positions[:, 0, meta["prefix_len"]].tolist(),
            "compact_suffix_first_slot": compact_len,
        })
        return result

    def close(self):
        for store in self._stores.values():
            if hasattr(store, "close"):
                store.close()
        self._stores.clear()

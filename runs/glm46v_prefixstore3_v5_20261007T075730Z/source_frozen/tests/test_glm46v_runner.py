"""CPU checks using the pinned native tokenizer and native position methods.

No checkpoint weights are loaded. Decode fixtures check physical mask lengths
and original position advancement; GPU attention/logit correctness is a
separate required gate.
"""
from __future__ import annotations

import copy
import json
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from mmimpress.glm46v.runner import (
    ANSWER_INSTRUCTION, GLM46VRunner, independent_positions, prefix_geometry,
    native_assistant_body, native_final_answer,
)


CHECKPOINT = Path(__file__).resolve().parents[1] / "runs/glm46v_prefixstore3_20261007T065207Z/checkpoint"


def _position_reference(ids, content, grid):
    # Explicit original token coordinates, independent of the port helper and
    # HF get_rope_index/get_vision_position_ids tensor construction.
    length = ids.shape[1]
    begin, after = content[0], content[-1] + 1
    height, width = int(grid[0, 1]) // 2, int(grid[0, 2]) // 2
    axes = [[], [], []]
    for logical in range(length):
        if logical < begin:
            point = [logical, logical, logical]
        elif logical < after:
            original = logical - begin
            point = [begin, begin + original // width, begin + original % width]
        else:
            position = begin + max(height, width) + logical - after
            point = [position, position, position]
        for axis in range(3):
            axes[axis].append(point[axis])
    result = torch.tensor(axes, dtype=torch.long).unsqueeze(1)
    offset = torch.tensor([[max(max(axis) for axis in axes) + 1 - length]])
    return result, offset


class NativeProcessorRunnerTests(unittest.TestCase):
    @property
    def native_eos_ids(self):
        return json.loads((CHECKPOINT / "generation_config.json").read_text())["eos_token_id"]

    def test_native_assistant_body_keeps_box_controls_and_separate_score_answer(self):
        tokenizer = self.processor.tokenizer
        body_ids = tokenizer.encode("<|begin_of_box|>blue<|end_of_box|>", add_special_tokens=False)
        generated = body_ids + [self.native_eos_ids[0]]
        original = list(generated)
        body, replay_ids = native_assistant_body(tokenizer, generated, self.native_eos_ids)
        self.assertEqual(body, "<|begin_of_box|>blue<|end_of_box|>")
        self.assertEqual(replay_ids, body_ids)
        self.assertEqual(generated, original)
        answer, trace = native_final_answer(tokenizer, generated)
        self.assertEqual(answer, "blue")
        self.assertEqual(trace["status"], "CLOSED_NATIVE_BOX")
        self.assertNotEqual(body, answer)

    def test_native_assistant_body_preserves_thinking_delimiters_and_complete_native_body(self):
        tokenizer = self.processor.tokenizer
        # Generated closing tags remain part of the recorded assistant body;
        # the official native template decides how prior reasoning is replayed.
        for content in ("</think>\n<|begin_of_box|>blue<|end_of_box|>",
                        "<think>native generated text</think>\n<|begin_of_box|>red<|end_of_box|>"):
            with self.subTest(content=content):
                expected_ids = tokenizer.encode(content, add_special_tokens=False)
                generated = expected_ids + [self.native_eos_ids[-1]]
                expected = tokenizer.decode(expected_ids, skip_special_tokens=False,
                                            clean_up_tokenization_spaces=False)
                body, replay_ids = native_assistant_body(tokenizer, generated, self.native_eos_ids)
                self.assertEqual(body, expected)
                self.assertEqual(replay_ids, expected_ids)
                self.assertIn("</think>", body)
                self.assertIn("<|begin_of_box|>", body)
                self.assertIn("<|end_of_box|>", body)

    def test_native_assistant_body_strips_only_final_eos_preserving_internal_controls(self):
        tokenizer = self.processor.tokenizer
        text = tokenizer.encode("blue", add_special_tokens=False)
        tail = tokenizer.encode(" retained", add_special_tokens=False)
        for internal in self.native_eos_ids:
            with self.subTest(internal=internal):
                # This negative fixture is not a claimed native generation:
                # greedy generation normally stops at the first EOS. It checks
                # that body construction is not general special-token removal.
                expected_ids = text + [internal] + tail
                generated = expected_ids + [self.native_eos_ids[0]]
                body, replay_ids = native_assistant_body(tokenizer, generated, self.native_eos_ids)
                self.assertEqual(replay_ids, expected_ids)
                self.assertEqual(body, tokenizer.decode(expected_ids, skip_special_tokens=False,
                                                        clean_up_tokenization_spaces=False))
                self.assertIn(internal, replay_ids)
        # Remove only one terminal EOS; do not silently delete an earlier one.
        generated = text + self.native_eos_ids[:2]
        _, replay_ids = native_assistant_body(tokenizer, generated, self.native_eos_ids)
        self.assertEqual(replay_ids, generated[:-1])

    def test_native_assistant_body_empty_eos_cap_and_multiple_boundaries(self):
        tokenizer = self.processor.tokenizer
        for generated in ([], *([eos] for eos in self.native_eos_ids)):
            with self.subTest(generated=generated):
                self.assertEqual(native_assistant_body(tokenizer, generated, self.native_eos_ids), ("", []))
        blue = tokenizer.encode("blue", add_special_tokens=False)
        # A cap-ending incomplete box must remain incomplete in its history
        # body. No EOS, closing control or parsed answer is invented.
        capped = [151361] + (blue * 32)[:31]
        body, replay_ids = native_assistant_body(tokenizer, capped, self.native_eos_ids)
        self.assertEqual(len(replay_ids), 32)
        self.assertEqual(replay_ids, capped)
        self.assertIn("<|begin_of_box|>", body)
        self.assertNotIn("<|end_of_box|>", body)
        _, extraction = native_final_answer(tokenizer, capped)
        self.assertEqual(extraction["status"], "MALFORMED_NATIVE_BOX")
        multiple = [151361, *blue, 151362, 151361, *blue, 151362]
        body, replay_ids = native_assistant_body(tokenizer, multiple + [self.native_eos_ids[0]], self.native_eos_ids)
        self.assertEqual(replay_ids, multiple)
        self.assertEqual(body.count("<|begin_of_box|>"), 2)
        self.assertEqual(body.count("<|end_of_box|>"), 2)

    def test_native_assistant_body_preserves_exact_whitespace_and_returns_independent_ids(self):
        tokenizer = self.processor.tokenizer
        content = " \n<|begin_of_box|>blue<|end_of_box|>\n  "
        expected_ids = tokenizer.encode(content, add_special_tokens=False)
        generated = expected_ids + [self.native_eos_ids[0]]
        snapshot = list(generated)
        expected = tokenizer.decode(expected_ids, skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)
        body, replay_ids = native_assistant_body(tokenizer, generated, self.native_eos_ids)
        self.assertEqual(body, expected)
        self.assertNotEqual(body, body.strip())
        replay_ids.append(1000)
        self.assertEqual(generated, snapshot)

    def test_actual_native_body_history_lineage_and_template_owned_reasoning_handling(self):
        runner = self.runner()
        tokenizer = self.processor.tokenizer
        meta = self.metadata(runner)
        for own_answer, foreign_answer in (("blue", "red"), ("red", "blue")):
            with self.subTest(own_answer=own_answer):
                content = "</think>\n<|begin_of_box|>" + own_answer + "<|end_of_box|>"
                generated = tokenizer.encode(content, add_special_tokens=False) + [self.native_eos_ids[0]]
                body, body_ids = native_assistant_body(tokenizer, generated, self.native_eos_ids)
                history = [("What color is the object?", body)]
                snapshot = copy.deepcopy(history)
                messages = runner.messages("What shape is it?", history)
                assistant = [message for message in messages if message["role"] == "assistant"]
                self.assertEqual(len(assistant), 1)
                self.assertEqual(assistant[0]["content"], body)
                self.assertEqual(body_ids, generated[:-1])
                full, suffix, positions, offset, text = runner.cache_inputs(meta, "What shape is it?", history)
                expected_box = "<|begin_of_box|>" + own_answer + "<|end_of_box|>"
                # The recorded body retained </think>; the pinned official
                # template renders old reasoning as its empty think prefix.
                self.assertIn("<|assistant|>\n<think></think>\n" + expected_box, text)
                decoded_suffix = tokenizer.decode(suffix[0], skip_special_tokens=False)
                self.assertIn(expected_box, decoded_suffix)
                self.assertNotIn("<|begin_of_box|>" + foreign_answer + "<|end_of_box|>", decoded_suffix)
                self.assertEqual(history, snapshot)
                self.assertEqual(full[0, :len(meta["prefix_input_ids"])].tolist(), meta["prefix_input_ids"])
                reference, reference_offset = _position_reference(full, meta["content_logical_indices"],
                    torch.tensor(meta["identity"]["geometry"]["image_grid_thw"]))
                self.assertTrue(torch.equal(positions, reference))
                self.assertTrue(torch.equal(offset, reference_offset))

    def test_result_records_native_body_ids_separately_from_answer_extraction(self):
        runner = self.runner()
        runner._sync = lambda: None
        runner.model.generation_config = types.SimpleNamespace(eos_token_id=self.native_eos_ids)
        generated = self.processor.tokenizer.encode("</think>\n<|begin_of_box|>blue<|end_of_box|>",
                                                     add_special_tokens=False) + [self.native_eos_ids[0]]
        with mock.patch("torch.cuda.max_memory_allocated", return_value=0), \
                mock.patch("torch.cuda.max_memory_reserved", return_value=0):
            result = runner.result(generated, time.perf_counter(), 1., {"input_ids": [[1000]]}, {})
        self.assertEqual(result["generated_token_ids"], generated)
        self.assertEqual(result["history_token_ids"], generated[:-1])
        self.assertEqual(result["history_text"], self.processor.tokenizer.decode(generated[:-1],
            skip_special_tokens=False, clean_up_tokenization_spaces=False))
        self.assertEqual(result["prediction"], "blue")
        self.assertNotEqual(result["history_text"], result["prediction"])
        self.assertEqual(result["history_native_policy"], "actual-generated-body/terminal-EOS-only-v1")

    def test_native_final_answer_boundaries_are_parsed_without_question_or_gold(self):
        tokenizer = self.processor.tokenizer
        blue = tokenizer.encode('blue', add_special_tokens=False)
        result, trace = native_final_answer(tokenizer, [151361, *blue, 151362, 151329])
        self.assertEqual(result, 'blue')
        self.assertEqual(trace['status'], 'CLOSED_NATIVE_BOX')
        self.assertNotIn(151361, tokenizer.all_special_ids)
        self.assertNotIn(151362, tokenizer.all_special_ids)
        plain, trace = native_final_answer(tokenizer, blue)
        self.assertEqual(plain, 'blue')
        self.assertEqual(trace['status'], 'PLAIN_TEXT')
        malformed, trace = native_final_answer(tokenizer, [151361, *blue])
        self.assertIn('<|begin_of_box|>', malformed)
        self.assertEqual(trace['status'], 'MALFORMED_NATIVE_BOX')
        multiple, trace = native_final_answer(tokenizer, [151361,*blue,151362,151361,*blue,151362])
        self.assertEqual(trace['status'], 'MALFORMED_NATIVE_BOX')
        self.assertIn('<|begin_of_box|>', multiple)
    def test_full_native_vision_path_is_excluded_by_installed_quantizer(self):
        from transformers.quantizers.quantizers_utils import should_convert_module
        policy = ["model.visual", "lm_head"]
        for name in ("model.visual.blocks.23.attn.qkv", "model.visual.merger.linear_fc1",
                     "model.visual.downsample", "lm_head"):
            self.assertFalse(should_convert_module(name, policy))
        self.assertTrue(should_convert_module("model.language_model.layers.39.self_attn.k_proj", policy))
    @classmethod
    def setUpClass(cls):
        if not (CHECKPOINT / "tokenizer.json").is_file():
            raise unittest.SkipTest("pinned local GLM checkpoint processor not available")
        try:
            from transformers import AutoConfig, AutoProcessor
            from transformers.models.glm4v.modeling_glm4v import Glm4vModel
        except ImportError as exc:
            raise unittest.SkipTest(f"GLM native Transformers unavailable: {exc}")
        cls.config = AutoConfig.from_pretrained(CHECKPOINT, local_files_only=True)
        cls.processor = AutoProcessor.from_pretrained(CHECKPOINT, local_files_only=True, use_fast=True)
        cls.core = types.SimpleNamespace(config=cls.config, rope_deltas=None)
        cls.core.get_vision_position_ids = types.MethodType(Glm4vModel.get_vision_position_ids, cls.core)
        cls.core.get_rope_index = types.MethodType(Glm4vModel.get_rope_index, cls.core)

    def runner(self):
        runner = GLM46VRunner(CHECKPOINT)
        runner.processor = self.processor
        runner.model = types.SimpleNamespace(config=self.config, model=self.core)
        return runner

    def processed(self, runner, question="What color is the object?", history=()):
        from PIL import Image
        image = Image.new("RGB", (112, 168), (13, 80, 141))
        text = runner.chat(question, history)
        return runner.processor(text=[text], images=[image], return_tensors="pt")

    def metadata(self, runner, question="What color is the object?"):
        inputs = self.processed(runner, question)
        ids, grid = inputs["input_ids"], inputs["image_grid_thw"]
        geometry = prefix_geometry(ids, grid, self.config)
        positions, offset = _position_reference(ids, geometry["content_indices"], grid)
        p = geometry["prefix_len"]
        return {"prefix_input_ids": ids[0, :p].tolist(),
                "content_logical_indices": geometry["content_indices"],
                "logical_position_ids": positions[:, 0, :p].tolist(),
                "rope_offset": offset.tolist(), "identity": {"geometry": geometry}}

    def test_native_non_thinking_template_and_tokenized_prompt(self):
        runner = self.runner()
        text = runner.chat("What color is the object?", ())
        self.assertTrue(text.endswith("<think></think>\n"))
        self.assertIn("/nothink<|assistant|>", text)
        inputs = self.processed(runner)
        tokens = self.processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
        self.assertIn("<think></think>", tokens)
        self.assertIn(ANSWER_INSTRUCTION, tokens)
        self.assertNotIn("<|tool|>", tokens)
        self.assertEqual((inputs["input_ids"] == self.config.image_start_token_id).sum().item(), 1)
        self.assertEqual((inputs["input_ids"] == self.config.image_end_token_id).sum().item(), 1)

    def test_native_positions_geometry_and_suffix_match_independent_scalar_reference(self):
        runner = self.runner()
        for question, history in (("What color?", ()),
                                  ("Where is it?", (("What color?", "blue"),)),
                                  ("What shape?", (("What color?", "blue"), ("Where is it?", "left")))):
            with self.subTest(history=history):
                inputs = self.processed(runner, question, history)
                ids, grid = inputs["input_ids"], inputs["image_grid_thw"]
                geometry = prefix_geometry(ids, grid, self.config)
                reference, reference_offset = _position_reference(ids, geometry["content_indices"], grid)
                native, offset = runner.positions(ids, grid)
                helper, helper_offset = independent_positions(ids, geometry["content_indices"], grid)
                self.assertTrue(torch.equal(native, reference))
                self.assertTrue(torch.equal(helper, reference))
                self.assertTrue(torch.equal(offset, reference_offset))
                self.assertTrue(torch.equal(helper_offset, reference_offset))
                begin = geometry["content_indices"][0]
                p = geometry["prefix_len"]
                suffix_start = begin + max(int(grid[0, 1]) // 2, int(grid[0, 2]) // 2) + 1
                self.assertEqual(native[:, 0, p].tolist(), [suffix_start] * 3)
                self.assertNotEqual(suffix_start, p)  # Native spatial position is not N.

    def test_generated_method_history_reconstructed_and_dialogue_isolated(self):
        runner = self.runner()
        meta = self.metadata(runner)
        histories = [(("What color is the object?", "blue"),),
                     (("What color is the object?", "red"),),
                     (("What color is the object?", "green"), ("Where is it?", "center"))]
        snapshots = copy.deepcopy(histories)
        prior_prefix = None
        for history in histories:
            full, suffix, positions, offset, text = runner.cache_inputs(meta, "What shape is it?", history)
            self.assertEqual(full[0, :len(meta["prefix_input_ids"])].tolist(), meta["prefix_input_ids"])
            self.assertTrue(suffix.numel() > 0)
            decoded_suffix = self.processor.tokenizer.decode(suffix[0], skip_special_tokens=False)
            for _, own_generated_answer in history:
                self.assertIn(own_generated_answer, decoded_suffix)
            for foreign in ("blue", "red", "green"):
                if foreign not in [answer for _, answer in history]:
                    self.assertNotIn(foreign, decoded_suffix)
            reference, reference_offset = _position_reference(full, meta["content_logical_indices"],
                torch.tensor(meta["identity"]["geometry"]["image_grid_thw"]))
            self.assertTrue(torch.equal(positions, reference))
            self.assertTrue(torch.equal(offset, reference_offset))
            self.assertEqual(offset.tolist(), meta["rope_offset"])
            self.assertIn("What shape is it?", text)
            prefix = positions[:, :, :len(meta["prefix_input_ids"])]
            if prior_prefix is not None:
                self.assertTrue(torch.equal(prior_prefix, prefix))
            prior_prefix = prefix.clone()
        self.assertEqual(histories, snapshots)
        self.core.rope_deltas = torch.tensor([[999]])
        runner._clear()
        self.assertIsNone(self.core.rope_deltas)

    def test_same_image_multiple_questions_share_only_native_image_prefix(self):
        runner = self.runner()
        saved = None
        for question in ("What color?", "What object?", "Where is the object? " * 128):
            inputs = self.processed(runner, question)
            geometry = prefix_geometry(inputs["input_ids"], inputs["image_grid_thw"], self.config)
            prefix = inputs["input_ids"][0, :geometry["prefix_len"]].tolist()
            if saved is not None:
                self.assertEqual(prefix, saved)
            saved = prefix
            self.assertGreater(inputs["input_ids"].shape[1], len(prefix))

    def test_wrong_prefix_native_position_template_and_boundary_rejected(self):
        runner = self.runner()
        original = self.metadata(runner)
        for case in ("prefix", "position", "content", "question_image_boundary", "system"):
            with self.subTest(case=case):
                meta = copy.deepcopy(original)
                question = "What shape?"
                if case == "prefix":
                    meta["prefix_input_ids"][0] += 1
                elif case == "position":
                    meta["logical_position_ids"][0][0] += 1
                elif case == "content":
                    meta["content_logical_indices"][1] = meta["content_logical_indices"][0]
                elif case == "question_image_boundary":
                    question += "<|end_of_image|>"
                else:
                    meta["prefix_input_ids"][4] += 1
                with self.assertRaises(ValueError):
                    runner.cache_inputs(meta, question, ())

    def test_geometry_negative_grid_multiple_images_and_noncontent_hole(self):
        runner = self.runner()
        inputs = self.processed(runner)
        for case in ("bad_grid", "two_grids", "hole", "missing_boundary", "second_boundary"):
            with self.subTest(case=case):
                ids, grid = inputs["input_ids"].clone(), inputs["image_grid_thw"].clone()
                geometry = prefix_geometry(ids, grid, self.config)
                if case == "bad_grid":
                    grid[0, 1] += 1
                elif case == "two_grids":
                    grid = grid.repeat(2, 1)
                elif case == "hole":
                    ids[0, geometry["content_indices"][1]] = 1000
                elif case == "missing_boundary":
                    ids[0, geometry["image_end_index"]] = 1000
                else:
                    ids[0, -1] = self.config.image_end_token_id
                with self.assertRaises(ValueError):
                    prefix_geometry(ids, grid, self.config)

    def test_wrong_suffix_positions_and_changed_stored_offset_rejected(self):
        runner = self.runner()
        meta = self.metadata(runner)
        native_method = runner.positions
        def bad_suffix(ids, grid):
            positions, offset = native_method(ids, grid)
            positions[:, 0, -1] += 1
            return positions, offset
        runner.positions = bad_suffix
        with self.assertRaisesRegex(ValueError, "history/suffix position"):
            runner.cache_inputs(meta, "What shape?", (("What color?", "blue"),))
        runner.positions = native_method
        meta["rope_offset"][0][0] += 1
        with self.assertRaisesRegex(ValueError, "changed stored RoPE offset"):
            runner.cache_inputs(meta, "What shape?", ())

    def test_wrong_loaded_ids_positions_offsets_shapes_and_full_fallback_rejected(self):
        runner = self.runner()
        meta = self.metadata(runner)
        content = meta["content_logical_indices"]
        selected = list(range((len(content) + 3) // 4))
        structural = [index for index in range(len(meta["prefix_input_ids"])) if index not in content]
        logical = sorted(structural + [content[index] for index in selected])
        meta.update(prefix_len=len(meta["prefix_input_ids"]), structural_indices=structural,
                    stored_to_original=selected)
        loaded = types.SimpleNamespace(
            logical_indices=tuple(logical),
            position_ids=torch.tensor(meta["logical_position_ids"])[:, logical].unsqueeze(1),
            native_prefix_length=meta["prefix_len"], rope_offset=copy.deepcopy(meta["rope_offset"]),
            layers=[(torch.zeros(1, 2, len(logical), 128, dtype=torch.bfloat16),
                     torch.zeros(1, 2, len(logical), 128, dtype=torch.bfloat16))] * 40)
        key = str(Path("/tmp/glm-cpu-negative-store").resolve())
        runner.stores[key] = types.SimpleNamespace(meta=meta)
        for case in ("logical", "duplicate", "positions", "offset", "original_length",
                     "head_expansion", "dtype", "missing_layer", "full_fallback"):
            with self.subTest(case=case):
                wrong = copy.deepcopy(loaded)
                if case == "logical":
                    wrong.logical_indices = tuple(logical[:-1] + [content[-1]])
                elif case == "duplicate":
                    wrong.logical_indices = tuple([logical[0]] + logical[:-1])
                elif case == "positions":
                    wrong.position_ids[0, 0, 0] += 1
                elif case == "offset":
                    wrong.rope_offset[0][0] += 1
                elif case == "original_length":
                    wrong.native_prefix_length = len(logical)
                elif case == "head_expansion":
                    wrong.layers[0] = tuple(value.repeat(1, 8, 1, 1) for value in wrong.layers[0])
                elif case == "dtype":
                    wrong.layers[0] = tuple(value.to(torch.float16) for value in wrong.layers[0])
                elif case == "missing_layer":
                    wrong.layers.pop()
                else:
                    wrong.logical_indices = tuple(range(meta["prefix_len"]))
                    wrong.position_ids = torch.tensor(meta["logical_position_ids"]).unsqueeze(1)
                self.core.rope_deltas = torch.tensor([[999]])
                with mock.patch("torch.cuda.reset_peak_memory_stats"):
                    with self.assertRaisesRegex(ValueError, "loaded native"):
                        runner.run_cache(key, "What shape?", (), memory_layers=wrong)
                self.assertIsNone(self.core.rope_deltas)

    def test_run_cache_exception_cleans_mutable_native_state(self):
        runner = self.runner()
        def fails(*args, **kwargs):
            self.core.rope_deltas = torch.tensor([[999]])
            raise ValueError("fixture request failure")
        runner._run_cache = fails
        with self.assertRaisesRegex(ValueError, "fixture request failure"):
            runner.run_cache("/tmp/nonexistent-cpu-fixture", "What shape?")
        self.assertIsNone(self.core.rope_deltas)


class _LengthCache:
    def __init__(self, length):
        self.length = length
    def get_seq_length(self):
        return self.length


class _DecodeModel:
    def __init__(self, predictions, *, advance=True):
        self.predictions = list(predictions)
        self.calls = []
        self.advance = advance
        self.generation_config = types.SimpleNamespace(eos_token_id=[2])
    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.advance:
            kwargs["past_key_values"].length += kwargs["input_ids"].shape[1]
        logits = torch.zeros(1, 1, 32)
        logits[0, 0, self.predictions.pop(0)] = 1
        return types.SimpleNamespace(logits=logits)


class DecodePositionTests(unittest.TestCase):
    def runner(self, predictions, *, advance=True):
        runner = GLM46VRunner("cpu-fixture", max_new_tokens=4)
        runner.device = torch.device("cpu")
        runner.model = _DecodeModel(predictions, advance=advance)
        runner._sync = lambda: None
        return runner

    def test_compact_physical_mask_native_decode_advance_eos_and_no_future_prefix(self):
        runner = self.runner([7, 9, 2])
        # First suffix prefill has compact physical length9; original native
        # next position91 is unrelated to that cache length.
        cache = _LengthCache(9)
        initial_mask = torch.ones(1, 9, dtype=torch.long)
        generated = runner.decode(5, cache, initial_mask, native_start=91)
        self.assertEqual(generated, [5, 7, 9, 2])
        self.assertEqual(cache.length, 12)
        self.assertEqual([call["position_ids"].flatten().tolist() for call in runner.model.calls],
                         [[91] * 3, [92] * 3, [93] * 3])
        self.assertEqual([call["attention_mask"].shape[1] for call in runner.model.calls], [10, 11, 12])
        self.assertTrue(all(bool(call["attention_mask"].all()) for call in runner.model.calls))
        self.assertTrue(all(call["input_ids"].shape == (1, 1) for call in runner.model.calls))
        self.assertEqual(initial_mask.shape[1], 9)

    def test_first_eos_has_no_decode_call_and_cap_is_common(self):
        runner = self.runner([])
        self.assertEqual(runner.decode(2, _LengthCache(8), torch.ones(1, 8), 41), [2])
        self.assertFalse(runner.model.calls)
        runner = self.runner([6, 7, 8])
        self.assertEqual(runner.decode(5, _LengthCache(8), torch.ones(1, 8), 41), [5, 6, 7, 8])
        self.assertEqual(len(runner.model.calls), 3)

    def test_broken_decode_cache_advancement_rejected(self):
        runner = self.runner([7, 9, 2], advance=False)
        with self.assertRaisesRegex(ValueError, "cache advancement"):
            runner.decode(5, _LengthCache(9), torch.ones(1, 9), 91)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Compare unchanged v4 manual decoding with native HF generate on a failure.

Derived diagnostic only. No frozen source, settings, question or token filters
are modified. The second history is exactly the actual T1 native decoded body,
with registered EOS removed and ordinary native answer controls retained.
"""
from pathlib import Path
import datetime as dt
import hashlib
import json
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from PIL import Image

from mmimpress.glm46v.runner import GLM46VRunner


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def small_result(result):
    return {key: value for key, value in result.items()
            if key not in ("first_logits", "capture")}


@torch.inference_mode()
def native_request(runner, image_path, question, history):
    runner._clear()
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
    # Native chat template from the same pinned processor; same text/image
    # processing path as v4 run_pixels. Explicit non-thinking is in chat().
    text = runner.chat(question, history)
    inputs = runner.processor(text=[text], images=[rgb], return_tensors="pt")
    inputs.pop("token_type_ids", None)
    input_ids = inputs["input_ids"].tolist()
    native_calls = []

    def observer(module, args, kwargs):
        positions = kwargs.get("position_ids")
        ids = kwargs.get("input_ids")
        cache = kwargs.get("past_key_values")
        native_calls.append({"position_ids": None if positions is None else positions.detach().cpu().tolist(),
                             "input_ids": None if ids is None else ids.detach().cpu().tolist(),
                             "cache_length_before": 0 if cache is None else int(cache.get_seq_length())})
        # No replacement or mutation of the incoming native arguments.

    handle = runner.model.register_forward_pre_hook(observer, with_kwargs=True)
    inputs = {key: value.to(runner.device) if torch.is_tensor(value) else value
              for key, value in inputs.items()}
    try:
        output = runner.model.generate(**inputs, do_sample=False,
                                       max_new_tokens=runner.max_new_tokens,
                                       output_logits=True, return_dict_in_generate=True)
        runner._sync()
        generated = output.sequences[0, len(input_ids[0]):].tolist()
        first_logits = output.logits[0][0].detach().float().cpu().clone()
        literal = runner.processor.tokenizer.decode(generated, skip_special_tokens=False,
                                                    clean_up_tokenization_spaces=False)
        evidence = {"generated_token_ids": generated, "generated_text_with_special_tokens": literal,
                    "native_decoded_body": runner.processor.tokenizer.decode(
                        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                    "first_logits_sha256": digest(first_logits), "prompt_text": text,
                    "input_ids": input_ids, "forward_calls": native_calls,
                    "enable_thinking": False, "do_sample": False,
                    "max_new_tokens": runner.max_new_tokens}
        return evidence, first_logits
    finally:
        handle.remove()
        runner._clear()


def main():
    run = ROOT / "runs/glm46v_prefixstore3_v4_20261007T072841Z"
    source = run / "main/images/n204894/attempt_001/raw.jsonl"
    target_id = "mtgqa_000040"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    t1 = next(row for row in rows if row["dialog_id"] == target_id
              and row["method_id"] == "recompute" and row["turn_id"] == 1)
    manifest = run / "manifests/main_manifest.json"
    dialogue = next(item for item in json.loads(manifest.read_text())["dialogues"]
                    if item["dialog_id"] == target_id)
    # Copy only the immutable question. Gold is never passed to either path.
    question = dialogue["turns"][1]["question"]
    original_history = [(t1["question"], t1["prediction"])]
    image_path = (ROOT / t1["image_path"]).resolve()
    assert file_sha(image_path) == t1["image_sha256"]
    tag = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path(__file__).parent / "non_thinking_diagnostics" / tag
    directory.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "RUNNING", "diagnostic_only": True, "directory": str(directory),
               "source_raw_sha256": file_sha(source), "source_manifest_sha256": file_sha(manifest),
               "source_t1_logical_request_id": t1["logical_request_id"],
               "source_t1_physical_execution_id": t1["physical_execution_id"],
               "source_t1_generated_ids": t1["generated_token_ids"],
               "question": question, "image_sha256": t1["image_sha256"],
               "no_gold_inputs": True, "variants": {}, "errors": []}
    runner = None
    try:
        runner = GLM46VRunner(run / "checkpoint", max_new_tokens=32).load()
        body = runner.processor.tokenizer.decode(t1["generated_token_ids"], skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)
        variants = {"v4_final_answer_history": original_history,
                    "native_generated_t1_body_history": [(t1["question"], body)]}
        receipt["actual_native_t1_body"] = body
        receipt["loaded_fingerprint"] = runner.fingerprint()
        for name, history in variants.items():
            manual = runner.run_pixels(image_path, question, history, image_sha256=t1["image_sha256"],
                                       return_logits=True)
            official, native_logits = native_request(runner, image_path, question, history)
            manual_logits = manual["first_logits"]
            native_positions = official["forward_calls"][0]["position_ids"]
            # Native generate adds one physical text-position axis for packing;
            # compare the remaining three native multimodal RoPE axes exactly.
            multimodal = native_positions[1:] if len(native_positions) == 4 else native_positions
            comparisons = {
                "first_logits_bits_exact": torch.equal(manual_logits, native_logits),
                "first_logits_max_abs": float((manual_logits - native_logits).abs().max()),
                "generated_ids_exact": manual["generated_token_ids"] == official["generated_token_ids"],
                "generated_text_exact": manual["generated_text_with_special_tokens"] == official["generated_text_with_special_tokens"],
                "prompt_text_exact": manual["request_trace"]["prompt_text"] == official["prompt_text"],
                "input_ids_exact": manual["input_ids"] == official["input_ids"],
                "native_three_axis_prefill_positions_exact": manual["request_trace"]["position_ids"] == multimodal,
                "native_control_format": {"begin_box_count": official["generated_token_ids"].count(151361),
                                          "end_box_count": official["generated_token_ids"].count(151362),
                                          "think_open_count": official["generated_text_with_special_tokens"].count("<think>"),
                                          "think_close_count": official["generated_text_with_special_tokens"].count("</think>")},
            }
            evidence = {"variant": name, "history": [list(pair) for pair in history],
                        "manual": small_result(manual), "official_generate": official,
                        "manual_first_logits_sha256": digest(manual_logits), "comparisons": comparisons}
            path = directory / (name + ".json")
            path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
            receipt["variants"][name] = {"evidence": str(path), "sha256": file_sha(path),
                                         "comparisons": comparisons,
                                         "manual_output": manual["generated_text_with_special_tokens"],
                                         "official_output": official["generated_text_with_special_tokens"]}
            (directory / "receipt.partial.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
            print(json.dumps({"variant": name, "comparisons": comparisons,
                              "official_output": official["generated_text_with_special_tokens"]}, ensure_ascii=False), flush=True)
        receipt["status"] = "COMPLETED_DIAGNOSTIC"
    except Exception as error:
        receipt["status"] = "DIAGNOSTIC_FAILED"
        receipt["errors"].append({"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        traceback.print_exc()
    finally:
        if runner is not None:
            runner.close()
        receipt["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        path = directory / "receipt.json"
        path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"status": receipt["status"], "receipt": str(path)}, ensure_ascii=False), flush=True)
    return 0 if receipt["status"] == "COMPLETED_DIAGNOSTIC" else 1


if __name__ == "__main__":
    raise SystemExit(main())

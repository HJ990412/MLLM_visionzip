#!/usr/bin/env python3
"""ConvBench ReComp-only answer generation (no KV store, no SSD method) for the baseline alignment.

One conversation = three plain multimodal requests (image + whole text history each turn), exactly the
ReComp path of scripts/41_eval_convbench_answers.py: same runner loading (NF4 / BF16 / eager unless a
diagnostic says otherwise), same processor call, same greedy ``generate`` arguments, same answer
post-processing.  Timing is not measured here (this script is for answer quality only).

Options (each is one item of the alignment; the defaults reproduce the September run):
  --cap-policy   sept_clamp     min(1024, 4096 - input + 1) for input <= 4096, 1024 above (September)
                 official_1024  always 1024 (ConvBench official llava.py:53; no context adjustment)
  --tokenization sept           "<image>\\n" and "</s>USER:" as rendered in September
                 official       render so that the HF token ids equal the original-LLaVA token ids
                                (tokenizer_image_token with the slow legacy tokenizer); checked per request
  --decoding     greedy | sample_t02 (do_sample, temperature 0.2, top_p None: official llava.py:53; diagnostic)
  --precision    nf4 | fp16 (unquantised torch.float16 weights; diagnostic)
  --image-pad    none | official (LLaVA-1.5 only: pad to square with the processor mean colour; diagnostic)
  --reuse-from-sept   copy the September ReComp answers of conversations that the cap policy cannot change
                 (only valid for sept tokenization + greedy + nf4 on LLaVA-NeXT; see notes.md, stage 3A)

Output: <out>/config.json, <out>/conversations/<conversation_id>.json (atomic, one per conversation),
<out>/progress.json, <out>/DONE.  Re-running the same command continues; finished conversations are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEED = 1234
NOMINAL_CAP = 1024
SEPT_RUN = ROOT / "runs/convbench_context/full_577_native"
VICUNA_SYSTEM = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the "
    "human's questions."
)
# (after "<image>", after "</s>") that make the HF tokenizer of each checkpoint emit the original-LLaVA ids.
# Verified per request against the slow legacy tokenizer; a mismatch aborts the run.
OFFICIAL_RENDER = {"next": (" \n", "</s> "), "llava15": ("\n", "</s>")}
SEPT_RENDER = ("\n", "</s>")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=1)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def render_prompt(questions: list[str], previous_answers: list[str], after_image: str, after_eos: str) -> str:
    """llava_v1 conversation; identical to scripts/41 render_prompt for after_image="\\n", after_eos="</s>"."""
    assert 1 <= len(questions) <= 3 and len(previous_answers) == len(questions) - 1
    prompt = f"{VICUNA_SYSTEM} USER: <image>{after_image}{questions[0]} ASSISTANT:"
    for index, answer in enumerate(previous_answers, 1):
        prompt += f" {answer}{after_eos}USER: {questions[index]} ASSISTANT:"
    return prompt


def canonical_prompt(questions: list[str], previous_answers: list[str]) -> str:
    """The prompt string the official wrapper builds (conv_templates['llava_v1'].get_prompt())."""
    return render_prompt(questions, previous_answers, "\n", "</s>")


def official_token_ids(slow_tokenizer, prompt: str) -> list[int]:
    """Original LLaVA ``tokenizer_image_token``: chunks around <image> tokenised separately, one BOS, -200 = image."""
    chunks = [slow_tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]
    bos = slow_tokenizer.bos_token_id
    ids: list[int] = []
    offset = 0
    if chunks and chunks[0] and chunks[0][0] == bos:
        offset = 1
        ids.append(bos)
    for index, chunk in enumerate(chunks):
        if index > 0:
            ids.append(-200)
        ids.extend(chunk[offset:])
    return ids


def collapse_image(ids: list[int], image_token_id: int) -> list[int]:
    out: list[int] = []
    for value in ids:
        if value == image_token_id:
            if not out or out[-1] != -200:
                out.append(-200)
        else:
            out.append(value)
    return out


def expand2square(image: Image.Image, background: tuple[int, int, int]) -> Image.Image:
    """Original LLaVA mm_utils.expand2square (image_aspect_ratio == 'pad')."""
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    result = Image.new(image.mode, (side, side), background)
    result.paste(image, ((side - width) // 2, (side - height) // 2))
    return result


def effective_cap(policy: str, input_tokens: int, limit: int) -> tuple[int, bool]:
    if policy == "official_1024":
        return NOMINAL_CAP, False
    available = limit - input_tokens + 1
    cap = NOMINAL_CAP if input_tokens > limit else min(NOMINAL_CAP, available)
    return cap, cap < NOMINAL_CAP


def load_index() -> tuple[list[dict], str]:
    path = ROOT / "data/convbench/index.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["conversations"]
    assert len(rows) == 577
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return rows, digest


def sept_recomp_rows(conversation_id: str) -> list[dict]:
    payload = json.loads((SEPT_RUN / "conversations" / f"{conversation_id}.json").read_text(encoding="utf-8"))
    rows = sorted((r for r in payload["rows"] if r["method"] == "ReComp"), key=lambda r: r["turn_id"])
    assert [r["turn_id"] for r in rows] == [1, 2, 3]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=("next", "llava15"), required=True)
    parser.add_argument("--cap-policy", choices=("sept_clamp", "official_1024"), required=True)
    parser.add_argument("--tokenization", choices=("sept", "official"), required=True)
    parser.add_argument("--decoding", choices=("greedy", "sample_t02"), default="greedy")
    parser.add_argument("--precision", choices=("nf4", "fp16"), default="nf4")
    parser.add_argument("--image-pad", choices=("none", "official"), default="none")
    parser.add_argument("--reuse-from-sept", action="store_true")
    parser.add_argument("--source-ids", type=str, help="comma-separated official IDs (subset run, e.g. the replay check)")
    parser.add_argument("--compare-sept", action="store_true",
                        help="after generation compare every answer with the September ReComp answer (replay check)")
    args = parser.parse_args()

    if args.image_pad == "official" and args.model != "llava15":
        parser.error("--image-pad official is the LLaVA-1.5 diagnostic")
    if args.reuse_from_sept and not (args.model == "next" and args.tokenization == "sept" and
                                     args.decoding == "greedy" and args.precision == "nf4"):
        parser.error("--reuse-from-sept needs the September model, tokenization, decoding and precision")

    dialogs, index_sha = load_index()
    if args.source_ids:
        wanted = [s.strip() for s in args.source_ids.split(",")]
        by_id = {str(d["source_id"]): d for d in dialogs}
        dialogs = [by_id[s] for s in wanted]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    import transformers
    config = {
        "schema": "convbench-alignment-recomp-generation-v1",
        "model": args.model, "cap_policy": args.cap_policy, "tokenization": args.tokenization,
        "decoding": args.decoding, "precision": args.precision, "image_pad": args.image_pad,
        "reuse_from_sept": bool(args.reuse_from_sept), "nominal_max_new_tokens": NOMINAL_CAP, "seed": SEED,
        "index_sha256": index_sha, "conversation_ids": [d["conversation_id"] for d in dialogs],
        "attention": "eager", "compute_dtype": "bfloat16" if args.precision == "nf4" else "float16",
        "history_policy": "generated answers of the same run; image in the first turn only",
        "context_policy": "prompt and history are never truncated; inputs above 4096 tokens run as they are",
        "transformers": transformers.__version__, "torch": torch.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    config_path = out / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        for key in config:
            if key not in ("script_sha256",) and previous.get(key) != config[key]:
                raise SystemExit(f"config mismatch on {key}; refusing to resume into {out}")
    else:
        atomic_json(config_path, config)

    conv_dir = out / "conversations"
    todo = [d for d in dialogs if not (conv_dir / f"{d['conversation_id']}.json").exists()]
    started = time.time()

    def progress(done: int, note: str = "") -> None:
        atomic_json(out / "progress.json", {
            "done": done, "total": len(dialogs), "updated_unix": time.time(), "note": note,
            "seconds_per_conversation_this_process": ((time.time() - started) / max(1, done - (len(dialogs) - len(todo)))
                                                      if done > len(dialogs) - len(todo) else None)})

    # ---- conversations that the cap policy cannot change are copied from September (stage 3A only)
    reused = 0
    if args.reuse_from_sept and todo:
        remaining = []
        for dialog in todo:
            rows = sept_recomp_rows(dialog["conversation_id"])
            affected = any(r["generation_cap_reached"] and r["generation_cap_clamped_by_context"] for r in rows)
            if affected:
                remaining.append(dialog)
                continue
            body = {"conversation_id": dialog["conversation_id"], "source_row_index": dialog["source_row_index"],
                    "source_id": dialog["source_id"], "index_sha256": index_sha, "origin": "copied_from_september",
                    "rows": [{"turn_id": r["turn_id"], "method": "ReComp", "question": r["question"],
                              "prediction": r["prediction"], "prompt_sha256": r["prompt_sha256"],
                              "input_tokens": r["context_input_tokens"], "generated_tokens": r["generated_tokens"],
                              "effective_max_new_tokens": NOMINAL_CAP,
                              "september_effective_max_new_tokens": r["effective_max_new_tokens"],
                              "cap_reached": r["generated_tokens"] >= NOMINAL_CAP, "cap_clamped_by_context": False,
                              "input_above_4096": r["context_input_tokens"] > 4096,
                              "official_token_ids_match": None} for r in rows]}
            atomic_json(conv_dir / f"{dialog['conversation_id']}.json", body)
            reused += 1
        todo = remaining
        print(f"copied {reused} conversations from September; {len(todo)} to generate", flush=True)
    progress(len(dialogs) - len(todo), "loading model" if todo else "nothing to generate")

    mismatches: list[dict] = []
    if todo:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        from transformers import AutoTokenizer
        load_4bit = args.precision == "nf4"
        if args.model == "next":
            import mmimpress.model as model_module
            if not load_4bit:
                model_module.COMPUTE_DTYPE = "float16"       # diagnostic only: unquantised FP16 weights
            runner = model_module.LlavaRunner(load_4bit=load_4bit).load()
            tokenizer_source = runner.model_id
        else:
            import mmimpress.llava15.runner as runner_module
            if not load_4bit:
                runner_module.COMPUTE_DTYPE = "float16"
                runner_module.Llava15Runner.validate_architecture = lambda self: None   # asserts BF16
            runner = runner_module.Llava15Runner(load_4bit=load_4bit).load()
            tokenizer_source = runner.source
        tokenizer = runner.processor.tokenizer
        slow_tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False, legacy=True)
        image_token_id = int(runner.cfg.image_token_index)
        limit = int(runner.cfg.text_config.max_position_embeddings)
        after_image, after_eos = OFFICIAL_RENDER[args.model] if args.tokenization == "official" else SEPT_RENDER
        pad_colour = tuple(int(x * 255) for x in runner.processor.image_processor.image_mean)
        runtime = {"model_dtype": str(runner.model.dtype), "load_4bit": load_4bit,
                   "attn_implementation": runner.model.config._attn_implementation,
                   "text_context_limit": limit, "render_after_image": after_image, "render_after_eos": after_eos,
                   "gpu": torch.cuda.get_device_name(0)}
        atomic_json(out / "runtime.json", runtime)

        done = len(dialogs) - len(todo)
        for dialog in todo:
            cid = dialog["conversation_id"]
            with Image.open(ROOT / dialog["image_path"] if not os.path.isabs(dialog["image_path"])
                            else dialog["image_path"]) as handle:
                image = handle.convert("RGB")
            if args.image_pad == "official":
                image = expand2square(image, pad_colour)
            questions = [t["question"] for t in dialog["turns"]]
            answers: list[str] = []
            rows = []
            for turn in (1, 2, 3):
                prompt = render_prompt(questions[:turn], answers, after_image, after_eos)
                enc = runner.encode_prompt(image, prompt)
                input_ids = enc["input_ids"]
                input_tokens = int(input_ids.shape[1])
                official = official_token_ids(slow_tokenizer, canonical_prompt(questions[:turn], answers))
                match = collapse_image(input_ids[0].tolist(), image_token_id) == official
                if args.tokenization == "official" and not match:
                    raise SystemExit(f"{cid} T{turn}: HF token ids differ from the original-LLaVA token ids")
                cap, clamped = effective_cap(args.cap_policy, input_tokens, limit)
                enc = runner.to_device(enc)
                kwargs = dict(max_new_tokens=cap, pad_token_id=tokenizer.eos_token_id, use_cache=True)
                if args.decoding == "greedy":
                    kwargs.update(do_sample=False)
                else:
                    torch.manual_seed(SEED + 10 * int(dialog["source_row_index"]) + turn)
                    torch.cuda.manual_seed_all(SEED + 10 * int(dialog["source_row_index"]) + turn)
                    kwargs.update(do_sample=True, temperature=0.2, top_p=None, num_beams=1)
                with torch.inference_mode():
                    output = runner.model.generate(**enc, **kwargs)
                tokens = output[0, input_tokens:]
                answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                generated = int(tokens.numel())
                rows.append({"turn_id": turn, "method": "ReComp", "question": questions[turn - 1], "prediction": answer,
                             "prompt": prompt, "prompt_sha256": sha256_text(prompt), "input_tokens": input_tokens,
                             "generated_tokens": generated, "effective_max_new_tokens": cap,
                             "cap_reached": generated >= cap, "cap_clamped_by_context": clamped,
                             "input_above_4096": input_tokens > limit,
                             "generation_position_overflow_tokens": max(0, input_tokens + generated - limit),
                             "official_token_ids_match": bool(match),
                             "first_token_id": int(tokens[0]) if generated else None})
                answers.append(answer)
                del output, enc
            body = {"conversation_id": cid, "source_row_index": dialog["source_row_index"],
                    "source_id": dialog["source_id"], "index_sha256": index_sha, "origin": "generated", "rows": rows}
            if args.compare_sept:
                sept = sept_recomp_rows(cid)
                for mine, theirs in zip(rows, sept):
                    same = mine["prediction"] == theirs["prediction"]
                    mine["same_as_september"] = same
                    if not same:
                        mismatches.append({"conversation_id": cid, "turn_id": mine["turn_id"],
                                           "generated_tokens": [mine["generated_tokens"], theirs["generated_tokens"]]})
            atomic_json(conv_dir / f"{cid}.json", body)
            done += 1
            progress(done, cid)
            print(f"[{done}/{len(dialogs)}] {cid}: {[r['generated_tokens'] for r in rows]} tokens", flush=True)
            torch.cuda.empty_cache()

    # ---- summary over the whole run directory
    summary = {"conversations": 0, "requests": 0, "generated": 0, "copied_from_september": 0,
               "cap_reached": {1: 0, 2: 0, 3: 0}, "cap_clamped_by_context": {1: 0, 2: 0, 3: 0},
               "input_above_4096": {1: 0, 2: 0, 3: 0}, "max_input_tokens": 0, "official_token_ids_mismatch": 0,
               "empty_answers": 0}
    same = total = 0
    for dialog in dialogs:
        body = json.loads((conv_dir / f"{dialog['conversation_id']}.json").read_text(encoding="utf-8"))
        summary["conversations"] += 1
        summary["generated" if body["origin"] == "generated" else "copied_from_september"] += 1
        for row in body["rows"]:
            summary["requests"] += 1
            turn = row["turn_id"]
            summary["cap_reached"][turn] += bool(row["cap_reached"])
            summary["cap_clamped_by_context"][turn] += bool(row["cap_clamped_by_context"])
            summary["input_above_4096"][turn] += bool(row["input_above_4096"])
            summary["max_input_tokens"] = max(summary["max_input_tokens"], row["input_tokens"])
            summary["official_token_ids_mismatch"] += row["official_token_ids_match"] is False
            summary["empty_answers"] += not row["prediction"].strip()
            if "same_as_september" in row:
                total += 1
                same += bool(row["same_as_september"])
    if args.compare_sept:
        summary["september_comparison"] = {"requests_compared": total, "identical": same, "mismatches": mismatches}
    atomic_json(out / "summary.json", summary)
    progress(len(dialogs), "complete")
    if args.compare_sept and (total == 0 or same != total):
        print(json.dumps(summary, indent=1))
        raise SystemExit("replay check failed: answers differ from September")
    (out / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Independent pre-quality CPU/GPU gates for GLM PrefixStore-KV25 [B].

GPU execution requires the frozen preflight sample manifest. Observers always
call the original native SDPA function; FP64 is a separate semantic reference,
never the inference precision. Every failed attempt is retained.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from mmimpress.glm46v.runner import GLM46VRunner
from mmimpress.glm46v.store import IOCounter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def bits_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def native_bits_equal(left, right, context):
    require(len(left) == len(right), f"{context}: wrong layer count")
    for layer, (a, b) in enumerate(zip(left, right)):
        for kind, x, y in zip(("K", "V"), a, b):
            require(x.dtype == y.dtype == torch.bfloat16, f"{context}: native BF16 missing")
            require(x.shape == y.shape, f"{context}: layer {layer} {kind} shape mismatch")
            require(torch.equal(x.cpu().view(torch.uint16), y.cpu().view(torch.uint16)),
                    f"{context}: layer {layer} {kind} bit mismatch")


def reference_positions(full_ids, content, geometry):
    """Independent scalar loop, not runner/HF/port positioning helpers."""
    temporal, height, width = geometry["image_grid_thw"][0]
    merge = geometry.get("merge_size", geometry.get("spatial_merge_size", 2))
    require(temporal == 1 and height % merge == width % merge == 0, "bad image geometry")
    hh, ww = height // merge, width // merge
    require(len(content) == hh * ww, "original-N visual budget disagrees with geometry")
    first = content[0]
    require(content == list(range(first, first + len(content))), "noncontiguous native image content")
    columns = []
    for token in range(len(full_ids)):
        if token < first:
            columns.append([token] * 3)
        elif token < first + len(content):
            row, col = divmod(token - first, ww)
            columns.append([first, first + row, first + col])
        else:
            position = first + max(hh, ww) + token - first - len(content)
            columns.append([position] * 3)
    positions = torch.tensor(columns, dtype=torch.long).T[:, None]
    return positions, [[int(positions.max()) + 1 - len(full_ids)]]


def independent_memory(capture, policy):
    """Build rank/retained KV from canonical full capture without store helpers."""
    content = list(capture.content_indices)
    count = len(content)
    budget = (count + 3) // 4
    if policy == "full":
        selected = list(range(count))
    else:
        scores = capture.scores.tolist()
        require(len(scores) == count and all(math.isfinite(x) for x in scores), "invalid image scores")
        selected = sorted(range(count), key=lambda i: (-scores[i], i))[:budget]
    structure = [i for i in range(len(capture.prefix_ids)) if i not in set(content)]
    logical = tuple(sorted(structure + [content[i] for i in selected]))
    layers = []
    for key, value in capture.layers:
        pair = []
        for tensor in (key, value):
            cpu = tensor.detach().cpu()
            # Explicit row stacking avoids the store's repack/assembly code.
            pair.append(torch.stack([cpu[:, :, i, :] for i in logical], dim=2).contiguous())
        layers.append(tuple(pair))
    return SimpleNamespace(
        layers=layers, logical_indices=logical,
        position_ids=capture.positions[:, :, list(logical)].clone(),
        prefix_input_ids=tuple(capture.prefix_ids[i] for i in logical),
        selected_content_original=tuple(sorted(selected)),
        selected_content_logical=tuple(sorted(content[i] for i in selected)),
        native_prefix_length=len(capture.prefix_ids), rope_offset=capture.rope_offset,
        io=IOCounter(),
    )


def independent_payload(path, meta, reference):
    """Parse exact on-disk native bytes without calling a port decode helper."""
    layer_count, heads, dimension = meta["num_layers"], meta["num_kv_heads"], meta["head_dim"]
    rows, structural_rows = meta["stored_rows"], meta["structural_count"]
    structural_raw = (path / "structural_kv.bin").read_bytes()
    require(len(structural_raw) == 2 * layer_count * structural_rows * heads * dimension * 2,
            "structure payload has padding, truncation or unexpected bytes")
    structural = torch.frombuffer(bytearray(structural_raw), dtype=torch.bfloat16).reshape(
        2, layer_count, structural_rows, heads, dimension)
    raw_content_ids = [meta["content_logical_indices"][i] for i in meta["stored_row_to_original"]]
    # Explicit original-row dictionary is independent of store scatter/sort.
    result = []
    for layer in range(layer_count):
        pair = []
        for kind_index, kind in enumerate(("k", "v")):
            raw = (path / f"layer_{layer:03d}/{kind}.bin").read_bytes()
            require(len(raw) == rows * heads * dimension * 2,
                    f"layer {layer} {kind}: file does not end at exact stored rows")
            content_rows = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(rows, heads, dimension)
            mapping = {logical: structural[kind_index, layer, row]
                       for row, logical in enumerate(meta["structural_indices"])}
            mapping.update({logical: content_rows[row] for row, logical in enumerate(raw_content_ids)})
            pair.append(torch.stack([mapping[i] for i in reference.logical_indices], dim=1)[None].contiguous())
        result.append(tuple(pair))
    native_bits_equal(result, reference.layers, "independent raw payload")
    require(meta["padding_rows"] == 0, "padding is forbidden by frozen B contract")
    require(meta["bytes_content_kv"] == rows * 40960, "native head/token payload bytes wrong")


def compare_generation(left, right, label):
    require(torch.equal(left["first_logits"], right["first_logits"]), f"{label}: first logits differ")
    require(left["generated_token_ids"] == right["generated_token_ids"], f"{label}: generated IDs differ")
    require(left["prediction"] == right["prediction"], f"{label}: prediction differs")


def attention_stages(query, key, value, visible, scale, dtype):
    """Convert BEFORE QK, softmax and V aggregation; independent dense math."""
    q, k, v = query.to(dtype), key.to(dtype), value.to(dtype)
    if k.shape[1] != q.shape[1]:
        repetitions = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(repetitions, dim=1), v.repeat_interleave(repetitions, dim=1)
    logits = q @ k.transpose(-1, -2) * scale
    logits.masked_fill_(~visible, float("-inf"))
    probabilities = logits.softmax(-1)
    return logits, probabilities, probabilities @ v


def fp64_attention(query, key, value, visible, scale):
    return attention_stages(query, key, value, visible, scale, torch.float64)[2]


class NativeAttentionObserver:
    """Observe native all-layer suffix SDPA; its output is passed unchanged."""
    def __init__(self, logical, native_prefix_length, expected_layers=40, diagnostic_dir=None):
        self.logical = tuple(logical)
        self.native_prefix_length = native_prefix_length
        self.expected_layers = expected_layers
        self.rows = []
        self.diagnostic_dir = diagnostic_dir
        self.original = torch.nn.functional.scaled_dot_product_attention

    def __call__(self, query, key, value, *args, **kwargs):
        output = self.original(query, key, value, *args, **kwargs)
        # Only the first (suffix prefill) call for each layer is a semantic gate.
        if len(self.rows) >= self.expected_layers:
            return output
        require(query.shape[2] > 1, "suffix prefill was unexpectedly one token")
        q, k, v = query.detach().cpu(), key.detach().cpu(), value.detach().cpu()
        suffix_length, compact_prefix = q.shape[2], len(self.logical)
        require(k.shape[2] == compact_prefix + suffix_length, "native SDPA physical suffix offset wrong")
        expected = torch.zeros(1, 1, suffix_length, k.shape[2], dtype=torch.bool)
        expected[:, :, :, :compact_prefix] = True
        for row in range(suffix_length):
            expected[:, :, row, compact_prefix:compact_prefix + row + 1] = True
        mask = kwargs.get("attn_mask", args[0] if args else None)
        if mask is None:
            require(not kwargs.get("is_causal", False), "native upper-left causal mask lost prefix offset")
            native_visible = torch.ones_like(expected)
        else:
            mask = mask.detach().cpu()
            native_visible = mask if mask.dtype == torch.bool else mask == 0
        require(torch.equal(native_visible.expand_as(expected), expected), "native suffix future visibility differs")
        scale = kwargs.get("scale") or query.shape[-1] ** -0.5
        compact_stages = attention_stages(q, k, v, expected, scale, torch.float64)
        compact_output = compact_stages[2]
        dense_length = self.native_prefix_length + suffix_length
        dense_k = torch.zeros(k.shape[0], k.shape[1], dense_length, k.shape[3], dtype=k.dtype)
        dense_v = torch.zeros_like(dense_k)
        dense_k[:, :, list(self.logical)] = k[:, :, :compact_prefix]
        dense_v[:, :, list(self.logical)] = v[:, :, :compact_prefix]
        dense_k[:, :, self.native_prefix_length:] = k[:, :, compact_prefix:]
        dense_v[:, :, self.native_prefix_length:] = v[:, :, compact_prefix:]
        dense_visible = torch.zeros(1, 1, suffix_length, dense_length, dtype=torch.bool)
        dense_visible[:, :, :, list(self.logical)] = True
        for row in range(suffix_length):
            dense_visible[:, :, row, self.native_prefix_length:self.native_prefix_length + row + 1] = True
        dense_stages = attention_stages(q, dense_k, dense_v, dense_visible, scale, torch.float64)
        dense_output = dense_stages[2]
        error = (dense_output - compact_output).abs()
        # Tolerances are fixed in CONTRACT before any logits or quality reads.
        passing = torch.allclose(dense_output, compact_output, atol=1e-12, rtol=1e-12)
        # Same inputs and separately converted FP32 give a shape-rounding
        # diagnostic; a failure is retained and does not silently relax FP64.
        compact32 = attention_stages(q, k, v, expected, scale, torch.float32)
        dense32 = attention_stages(q, dense_k, dense_v, dense_visible, scale, torch.float32)
        retained = list(self.logical) + list(range(self.native_prefix_length, dense_length))
        valid_qk = expected.expand(compact32[0].shape)
        qk32_difference = dense32[0][..., retained][valid_qk] - compact32[0][valid_qk]
        evidence = {"layer": len(self.rows), "query_shape": list(q.shape),
                    "key_shape": list(k.shape), "dense_key_length": dense_length,
                    "max_abs_error_fp64": float(error.max()), "pass": passing,
                    "fp32_shape_diagnostic": {
                        "max_abs_qk": float(qk32_difference.abs().max()),
                        "max_abs_softmax": float((dense32[1][..., retained] - compact32[1]).abs().max()),
                        "max_abs_v_aggregation": float((dense32[2] - compact32[2]).abs().max()),
                        "atol": 1e-5, "rtol": 1e-5,
                        "pass": torch.allclose(dense32[2], compact32[2], atol=1e-5, rtol=1e-5),
                        "diagnostic_not_native_inference_precision": True},
                    "native_output_bits_sha256": bits_hash(output),
                    "future_suffix_visibility_rejected": True}
        self.rows.append(evidence)
        if self.diagnostic_dir is not None:
            (self.diagnostic_dir / "attention_layers.json").write_text(json.dumps(self.rows, indent=2))
            if not passing:
                torch.save({"query": q, "key": k, "value": v, "mask": native_visible,
                            "logical_indices": self.logical, "native_prefix_length": self.native_prefix_length,
                            "scale": scale}, self.diagnostic_dir / f"failed_attention_layer_{evidence['layer']:03d}.pt")
        require(passing, f"layer {evidence['layer']} dense/compact FP64 semantics failed: max_abs={float(error.max())}")
        return output


def run_sample(runner, sample, directory):
    image = Path(sample["image_path"])
    image_sha = sample["image_sha256"]
    require(sha(image) == image_sha, "frozen validation image hash mismatch")
    question = sample["question"]
    other = sample.get("other_question", question + " What color is visible?")
    plain = runner.run_pixels(image, question, image_sha256=image_sha, return_logits=True)
    require(plain["prediction"] and not plain["reasoning_leak"] and not plain["truncated"],
            "normal non-thinking inference has an empty answer, reasoning leak or unresolved cap truncation")
    canonical = runner.run_pixels(image, question, capture=True, image_sha256=image_sha, return_logits=True)
    scored = runner.run_pixels(image, question, capture="with_score", image_sha256=image_sha, return_logits=True)
    compare_generation(plain, canonical, "stock/capture T1")
    compare_generation(plain, scored, "stock/scored capture T1")
    native_bits_equal(canonical["capture"].layers, scored["capture"].layers, "capture full native prefix")
    alternate = runner.run_pixels(image, other, capture="with_score", image_sha256=image_sha, return_logits=True)
    require(torch.equal(scored["capture"].scores, alternate["capture"].scores), "question changed image-only scores")
    native_bits_equal(scored["capture"].layers, alternate["capture"].layers, "question-independent prefix")
    capture = scored["capture"]
    ref_positions, ref_offset = reference_positions(
        scored["input_ids"][0], capture.content_indices, capture.geometry)
    require(ref_positions.tolist() == scored["request_trace"]["position_ids"], "native position reference mismatch")
    require(ref_offset == scored["request_trace"]["rope_offset"], "native offset reference mismatch")
    full_reference = independent_memory(capture, "full")
    ours_reference = independent_memory(capture, "prefix25")
    evidence = {"image_id": sample["image_id"], "image_sha256": image_sha,
                "question": question, "other_question": other,
                "N": len(capture.content_indices), "k": (len(capture.content_indices) + 3) // 4,
                "geometry": capture.geometry, "native_prefix_length": len(capture.prefix_ids),
                "scores_sha256": bits_hash(capture.scores),
                "selected_original": list(ours_reference.selected_content_original),
                "stock_capture_t1_exact": True, "question_invariance": True,
                "position_geometry_reference": True, "arms": {}}
    try:
        for policy, reference in (("full", full_reference), ("prefix25", ours_reference)):
            path = directory / policy
            persisted = runner.persist(capture, path, storage_policy=policy)
            reader, activation = runner.activate(path)
            require(not any(isinstance(v, (torch.Tensor, bytes, bytearray)) for v in vars(reader).values()),
                    "activated reader retains resident payload")
            loaded = reader.load_prefix()
            native_bits_equal(loaded.layers, reference.layers, policy + " all-layer/native-head SSD bits")
            require(loaded.logical_indices == reference.logical_indices, policy + " original logical IDs mismatch")
            require(loaded.selected_content_original == reference.selected_content_original, policy + " rank mismatch")
            require(torch.equal(loaded.position_ids, reference.position_ids), policy + " stored native positions mismatch")
            independent_payload(path, reader.meta, reference)
            memory = runner.run_cache(path, question, return_logits=True, memory_layers=reference)
            observer = NativeAttentionObserver(reference.logical_indices, len(capture.prefix_ids), diagnostic_dir=path)
            with mock.patch("torch.nn.functional.scaled_dot_product_attention", observer), \
                    mock.patch.object(reader, "load_prefix", wraps=reader.load_prefix) as load_call:
                ssd = runner.run_cache(path, question, return_logits=True)
            require(load_call.call_count == 1, "SSD payload was reloaded during decode")
            compare_generation(memory, ssd, policy + " same-shape memory/SSD")
            require(len(observer.rows) == 40, "did not observe all native decoder layers")
            require(ssd["decode_store_reads"] == ssd["online_query_score_calls"] == 0,
                    "decode re-read/reselection occurred")
            expected_final = len(reference.logical_indices) + len(ssd["input_ids"][0]) - len(capture.prefix_ids) + len(ssd["generated_token_ids"]) - 1
            require(ssd["request_trace"]["decode_final_cache_length"] == expected_final,
                    "physical native decode cache length advancement differs")
            # Own actual generation in history; no gold or ReComp history.
            history = [(question, ssd["prediction"])]
            follow_memory = runner.run_cache(path, other, history, return_logits=True, memory_layers=reference)
            follow_ssd = runner.run_cache(path, other, history, return_logits=True)
            compare_generation(follow_memory, follow_ssd, policy + " generated-history suffix")
            if sample.get("long_question"):
                long_memory = runner.run_cache(path, sample["long_question"], history, return_logits=True, memory_layers=reference)
                long_ssd = runner.run_cache(path, sample["long_question"], history, return_logits=True)
                compare_generation(long_memory, long_ssd, policy + " longest native question suffix")
            # Interleave a foreign dialogue, then repeat exact original state.
            runner.run_cache(path, other, [(other, "different previous answer")], return_logits=True)
            repeat = runner.run_cache(path, question, return_logits=True)
            compare_generation(ssd, repeat, policy + " same-image dialogue state isolation")
            full_pos, off = reference_positions(follow_ssd["input_ids"][0], capture.content_indices, capture.geometry)
            require(full_pos.tolist() == follow_ssd["request_trace"]["position_ids"], "history native positions wrong")
            require(off == follow_ssd["request_trace"]["rope_offset"], "history native offset wrong")
            actual_dialogue_rows = []
            for dialogue in sample.get("actual_shared_dialogues", []):
                history = []
                for turn, item in enumerate(dialogue["turns"]):
                    current = item["question"]
                    if turn == 0:
                        actual = runner.run_pixels(image, current, image_sha256=image_sha, return_logits=True)
                    else:
                        in_memory = runner.run_cache(path, current, history, return_logits=True, memory_layers=reference)
                        actual = runner.run_cache(path, current, history, return_logits=True)
                        compare_generation(in_memory, actual, policy + " shared-image actual-dialogue suffix")
                    actual_dialogue_rows.append({"dialog_id": dialogue["dialog_id"], "turn": turn + 1,
                                                 "question": current, "generated_history": list(history),
                                                 "generated_token_ids": actual["generated_token_ids"],
                                                 "prediction": actual["prediction"]})
                    history.append((current, actual["prediction"]))
            arm = {"payload_bits_exact": True, "same_shape_first_logits_exact": True,
                   "same_shape_generated_ids_exact": True, "history_state_isolation": True,
                   "ssd_read_bytes": ssd["ssd_read_bytes"], "read_io": ssd["read_io"],
                   "store_content_bytes": reader.meta["bytes_content_kv"],
                   "store_structure_bytes": reader.meta["bytes_structural_kv"],
                   "logical_indices": list(reference.logical_indices),
                   "first_logits_sha256": bits_hash(ssd["first_logits"]),
                   "generated_token_ids": ssd["generated_token_ids"],
                   "actual_shared_dialogues": actual_dialogue_rows,
                   "attention_fp64_layers": observer.rows,
                   "source_store_identity": reader.meta["identity_sha256"]}
            if policy == "full":
                arm["recomp_fullload_native_shape_diagnostic"] = {
                    "first_logits_exact": torch.equal(plain["first_logits"], ssd["first_logits"]),
                    "generated_ids_equal": plain["generated_token_ids"] == ssd["generated_token_ids"],
                    "max_abs_first_logits": float((plain["first_logits"] - ssd["first_logits"]).abs().max()),
                    "stock_ids": plain["generated_token_ids"],
                    "separate_from_restore_and_compression_gate": True}
            evidence["arms"][policy] = arm
    finally:
        runner.close()
    # Validation-only clones are released before the next image. Stores stay
    # on disk for the independent audit and reproduction evidence.
    return evidence


def run_stress(runner, specification, directory):
    """Frozen synthetic resource fixture, excluded from dataset quality."""
    from PIL import Image
    import numpy as np
    seed = specification["seed"]
    height, width = specification["height"], specification["width"]
    generator = np.random.default_rng(seed)
    pixels = generator.integers(0, 256, (height, width, 3), dtype=np.uint8)
    path = directory / "resource_stress_image.png"
    Image.fromarray(pixels).save(path)
    image_sha = sha(path)
    result = runner.run_pixels(path, specification["question"], image_sha256=image_sha,
                               capture="with_score", return_logits=True)
    require(result["capture"].geometry["N_content"] == specification["expected_N"],
            "frozen resource stress processor N differs")
    return {"image": str(path), "image_sha256": image_sha, "specification": specification,
            "N": result["capture"].geometry["N_content"],
            "k": result["capture"].geometry["k"],
            "peak_gpu_allocated_bytes": result["peak_gpu_allocated_bytes"],
            "peak_gpu_reserved_bytes": result["peak_gpu_reserved_bytes"],
            "gpu_peak_scope": result["gpu_peak_scope"], "timing_ms": result["timing_ms"],
            "prediction": result["prediction"], "generated_token_ids": result["generated_token_ids"],
            "truncated": result["truncated"], "reasoning_leak": result["reasoning_leak"],
            "not_dataset_quality": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    manifest = (args.manifest or run / "validation_manifest.json").resolve()
    tag = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    attempt = run / "validation" / ("attempt_" + tag)
    attempt.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "RUNNING", "attempt": str(attempt), "started_utc": tag,
               "cpu_tests": {}, "samples": [], "failures": [],
               "fp64_atol": 1e-12, "fp64_rtol": 1e-12,
               "same_shape_native_comparison": "EXACT_BITS_AND_IDS",
               "gpu_quality_requests": 0, "gate_requests_excluded_from_pilot_main": True}
    runner = None
    try:
        for name in ("vision", "store", "runner", "validation", "analysis"):
            log = attempt / f"cpu_{name}.log"
            with log.open("w") as handle:
                proc = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests",
                                       "-p", f"test_glm46v_{name}.py", "-v"],
                                      cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            receipt["cpu_tests"][name] = {"returncode": proc.returncode, "log": str(log), "sha256": sha(log)}
            require(proc.returncode == 0, f"CPU {name} checks failed; log preserved")
            require("Ran 0 tests" not in log.read_text(), f"CPU {name}: no tests found")
        if args.cpu_only:
            receipt["status"] = "CPU_PASS_GPU_NOT_RUN"
        else:
            require(manifest.is_file(), "frozen preflight validation manifest is missing")
            receipt["manifest_sha256"] = sha(manifest)
            freeze_path = run / "freeze.json"
            require(freeze_path.is_file(), "source/config/environment freeze is missing")
            freeze = json.loads(freeze_path.read_text())
            receipt["source_freeze_sha256"] = sha(freeze_path)
            require(receipt["manifest_sha256"] == freeze["validation_manifest_sha256"],
                    "validation manifest differs from immutable freeze")
            for relative, expected in freeze["source_sha256"].items():
                require(sha(ROOT / relative) == expected, f"frozen source changed: {relative}")
            data = json.loads(manifest.read_text())
            samples = [dict(sample) for sample in data["samples"]]
            require(samples and data.get("selected_before_model_outputs", data.get("frozen_before_quality", False)),
                    "manifest was not frozen before quality")
            runner = GLM46VRunner(args.checkpoint).load()
            receipt["loaded_fingerprint"] = runner.fingerprint()
            stress = data.get("synthetic_gpu_stress")
            require(stress is not None, "maximum-N synthetic resource stress was not frozen")
            receipt["resource_stress"] = run_stress(runner, stress, attempt)
            shared_dialogues = data.get("same_image_dialogues", [])
            require(len(shared_dialogues) >= 2, "same-image multiple actual dialogues not frozen")
            shared_identity = data["same_image_identity"]
            shared_id = shared_dialogues[0]["image_id"]
            shared_sample = next((sample for sample in samples if sample["image_id"] == shared_id), None)
            if shared_sample is None:
                shared_sample = dict(shared_identity, image_id=shared_id,
                                     question=shared_dialogues[0]["turns"][0]["question"],
                                     other_question=shared_dialogues[1]["turns"][0]["question"])
                samples.append(shared_sample)
            shared_sample["actual_shared_dialogues"] = shared_dialogues
            for ordinal, sample in enumerate(samples):
                sample_dir = attempt / f"sample_{ordinal:03d}_{sample['image_id']}"
                sample_dir.mkdir()
                evidence = run_sample(runner, sample, sample_dir)
                (sample_dir / "evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False))
                receipt["samples"].append({"image_id": sample["image_id"], "evidence": str(sample_dir / "evidence.json"),
                                           "sha256": sha(sample_dir / "evidence.json")})
                (attempt / "receipt.partial.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False))
            receipt["status"] = "PASS"
    except Exception as error:
        import traceback
        receipt["status"] = "FAIL"
        receipt["failures"].append({"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
    finally:
        if runner is not None:
            runner.close()
        receipt["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        receipt_path = attempt / "receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False))
        if receipt["status"] == "PASS":
            # The main runner consumes only this immutable successful receipt.
            published = run / "validation" / "receipt.json"
            with published.open("x") as handle:
                json.dump(receipt, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
        summary = ["# GLM correctness validation", "", f"실제 상태: **{receipt['status']}**.", "",
                   f"Attempt receipt: `{receipt_path}`", "",
                   f"통과한 checkpoint GPU 샘플 수: {len(receipt['samples'])}.",
                   "Validation 요청은 pilot/main 분모에 포함하지 않는다.", ""]
        for failure in receipt["failures"]:
            summary.append(f"- {failure['type']}: {failure['message']}")
        (run / "VALIDATION.md").write_text("\n".join(summary) + "\n")
        print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path)}, ensure_ascii=False))
    return 0 if receipt["status"] in ("PASS", "CPU_PASS_GPU_NOT_RUN") else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared constants / helpers of the LLaVA-1.5-7B MT-GQA HOLD run (scripts 177-180).

Contract: docs/llava15_mt_gqa_contract.md.  Request generation, rotation, history, scoring,
commit, audit and cleanup rules are the LLaVA-NeXT HOLD lifecycle (scripts 156 / 162 / 164);
the generic helpers are imported from 156 unchanged, the model-bound ones (which hard-code
the LLaVA-NeXT model id, ``image_sizes`` or the 0.25 budget) are re-implemented here.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import uuid
from fractions import Fraction
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mmimpress.llava15 import MODEL_ID, N_VISUAL, NUM_LAYERS, R_MAX, R_MIN, REVISION  # noqa: E402

EXPERIMENT = "llava15_mt_gqa"
SCHEMA = "llava15-mtgqa-hold-v1"
CONTRACT = ROOT / "docs/llava15_mt_gqa_contract.md"
B_CONTRACT = ROOT / "docs/mt_gqa_B_main_contract.md"
INDEX_SHA = "2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224"
WORKLOAD_SHA = "0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62"
PRIOR = ROOT / "runs/mt_gqa_B_llava5_qwen3_20261001T093921Z/llava"           # read-only: frozen manifest (dialogues, images, smoke ids)
NEXT_HOLD = ROOT / "runs/layer_budget_kv25_pyrbeta_20261110T073604Z/hold"    # read-only: LLaVA-NeXT HOLD manifest (same smoke ids)
PYR_SOURCE = ROOT / "results/layer_budget_kv25_pyrbeta_20261110T073604Z/layer_budget_llava_PYR-beta-raw.json"
PYR_SOURCE_FILE_SHA = "8bfc4cbc55eb778a7ef0433ade4bf13f7a24107b76515f23a72ac01e0740ea5d"
SHORT_ANSWER = "Answer the current question with a single word or short phrase."
PK, PK_SUFFIX = "llava15_kv_", "_packed"
VECTORS = ("U100", "U50", "U25", "PYR-b20")
TOTAL_RATIO = {"U100": 1.0, "U50": 0.5, "U25": 0.25, "PYR-b20": 0.25}
FRACTION = {"U100": (1, 1), "U50": (1, 2), "U25": (1, 4), "PYR-b20": (1, 4)}        # independent integer form of the total ratio
RATIO_SHA = {"U100": "04c01bedee671a22789fcaedf3348f178446db6f914bfdb98bb1699ab64bc114",
             "U50": "673c2550205b17ed2a56548d2c9524811150da3c49cc4b370f80d05dec1201b0",
             "U25": "54ea273e1a38b1e8b02b96237381f982344cd28ca8466162876e161cdb8ff3ba",
             "PYR-b20": "51f9b0db19a93f8ca8b34b8beb8af21285721c5ea460566afbd9a3185b331724"}
METHODS = ["recompute"] + [f"{PK}{v}{PK_SUFFIX}" for v in VECTORS]
RECOMP, U100, U50, U25, PYR = METHODS
LABEL = {"recompute": "ReComp", U100: "U100_packed", U50: "U50_packed", U25: "U25_packed", PYR: "PYR-b20_packed"}
ROW_BYTES = 32 * 128 * 2                                   # one token row of one layer, K or V, FP16
EXPECTED_ROWS = {"U100": [576] * 32, "U50": [288] * 32, "U25": [144] * 32}
EXPECTED_PYR_HEAD, EXPECTED_PYR_TAIL, EXPECTED_PYR_SUM = [281, 272, 263], [34, 25, 16, 7], 4608
EXPECTED_BYTES = {"U100": 301_989_888, "U50": 150_994_944, "U25": 75_497_472, "PYR-b20": 75_497_472}
EXPECT = {"per_method": 12183, "main": 12183 * 5, "smoke": 60, "integration": 60, "images": 398, "dialogues": 4061}
FIXTURE_IMAGES = 10
ATOL = RTOL = 1e-4
SEED = 1234
PY = "/home/dblab/anaconda3/envs/mllm_ft/bin/python"
ENV = "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:scripts:tests OMP_NUM_THREADS=2 MKL_NUM_THREADS=2"
GiB = 1 << 30
NEW_PATHS = ["mmimpress/llava15/__init__.py", "mmimpress/llava15/runner.py", "mmimpress/llava15/store.py", "mmimpress/llava15/serve.py",
             "scripts/llava15_support.py", "scripts/177_freeze_llava15_mt_gqa.py", "scripts/178_validate_llava15_mt_gqa.py",
             "scripts/179_eval_llava15_mt_gqa_hold.py", "scripts/180_report_llava15_mt_gqa.py",
             "tests/test_llava15_store.py", "tests/test_llava15_harness.py", "docs/llava15_mt_gqa_contract.md"]


# ------------------------------------------------------------------ small io
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def _safe(x):
    import numpy as np
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_safe(v) for v in x]
    return x


def write(path, obj, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as f:
        json.dump(_safe(obj), f, indent=1, allow_nan=False, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def load_script(name, filename):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def lifecycle():
    """scripts/156 (LLaVA-NeXT HOLD lifecycle helpers), imported unchanged."""
    return load_script("_hold_v1_lifecycle", "156_eval_layer_budget_hold.py")


def qa():
    return load_script("_hold_common_io", "49_eval_query_aware_baseline.py")


# ------------------------------------------------------------------- arms
def vector_of(method):
    assert method.startswith(PK) and method.endswith(PK_SUFFIX), method
    return method[len(PK):-len(PK_SUFFIX)]


def kind_of(method):
    return "packed_" + vector_of(method) if method.startswith(PK) else "none"


def stored_methods(methods):
    return [m for m in methods if kind_of(m) != "none"]


def vector_path(run, vec):
    return Path(run) / "vectors" / f"layer_budget_llava15_{vec}.json"


def load_vectors(run):
    from mmimpress.llava15.store import load_budget
    docs = {}
    for vec in VECTORS:
        doc = load_budget(vector_path(run, vec))
        if doc["layer_budget_sha256"] != RATIO_SHA[vec] or doc["alloc_id"] != vec or float(doc["total_ratio"]) != TOTAL_RATIO[vec]:
            raise RuntimeError(f"BLOCKED: vector {vec} does not match the frozen ratio SHA256 / total ratio")
        docs[vec] = doc
    return docs


def independent_counts(ratios, n, fraction):
    """Hamilton apportionment of 32*ceil(p/q*N): no library code, integer total, no lower bound, cap N."""
    p, q = fraction
    L = len(ratios)
    budget = L * (-(-p * n // q))
    quotas = [Fraction(repr(float(r))) * n for r in ratios]
    k = [min(n, math.floor(x)) for x in quotas]
    frac = [x - math.floor(x) for x in quotas]
    order = sorted(range(L), key=lambda i: (-frac[i], i))
    need = budget - sum(k)
    if need < 0:
        raise ValueError("quota floors exceed the row budget")
    while need:
        moved = False
        for i in order:
            if need and k[i] < n:
                k[i] += 1
                need -= 1
                moved = True
        if not moved:
            raise ValueError("budget cannot be apportioned")
    return k


def expected_bytes(counts):
    return sum(2 * k * ROW_BYTES for k in counts)


# ---------------------------------------------------------------- requests
def request_id(run_id, phase, method, dialog_id, turn):
    return f"{SCHEMA}:{run_id}:{phase}:{MODEL_ID}:{method}:{dialog_id}:t{turn}"


def prompt_factory(methods, dialog, turn, method, generated):
    """The LLaVA-NeXT HOLD multi-turn prompt (156.prompt_factory) with this run's model id in the lineage."""
    if method not in methods or turn not in (1, 2, 3):
        raise ValueError("invalid method/turn")
    if any(t >= turn for t in generated):
        raise ValueError("future/current answer in history")
    lines, entries = [], []
    for t in range(1, turn):
        prior = dialog["turns"][t - 1]
        row = generated[t]
        if row["method_id"] != method or row["dialog_id"] != dialog["dialog_id"]:
            raise ValueError("foreign method/dialogue history")
        answer = row["prediction"]
        lines += [f"Q{t}: {prior['question'].strip()}", f"A{t}: {answer}"]
        entries.append({"turn_id": t, "question_id": str(prior["question_id"]), "question": prior["question"].strip(),
                        "answer": answer, "answer_source": "method_local_generated", "source_model": MODEL_ID,
                        "source_method_id": method, "source_dialog_id": dialog["dialog_id"],
                        "source_logical_request_id": row["logical_request_id"],
                        "source_physical_execution_id": row["physical_execution_id"]})
    history = "\n".join(lines)
    body = ([history, ""] if history else [])
    body += [f"Current question Q{turn}: {dialog['turns'][turn - 1]['question'].strip()}", f"{SHORT_ANSWER} ASSISTANT:"]
    return "USER: <image>\n" + "\n".join(body), history, entries


def image_input_hash(enc):
    """SHA256 of the processor's pixel tensor (LLaVA-1.5 has no ``image_sizes``)."""
    t = enc["pixel_values"].detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(b"pixel_values\0")
    h.update(str(t.dtype).encode("ascii"))
    h.update(json.dumps(list(t.shape)).encode("ascii"))
    h.update(t.numpy().tobytes())
    return h.hexdigest()


def load_image(path):
    from PIL import Image
    with Image.open(path) as im:
        image = im.convert("RGB")
        image.load()
    return image


def normal_request(runner, server, image_path, factory, capture):
    """Full-pixel request (T1 of every arm, every turn of recompute).  Timing boundary of 156.normal_request:
    the timer starts before prompt construction, image decoding and the HF processor."""
    from mmimpress.piggyback import VisionForwardCapture
    H = lifecycle()
    vision = VisionForwardCapture(runner, capture_saliency=bool(capture))
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    prompt, history, entries = factory()
    image = load_image(image_path)
    enc, processing = qa()._combined_processor(runner, image, prompt)
    v0, vn = runner.visual_span(enc["input_ids"])
    with vision, H.PilotCounts(runner) as counts:
        result = server.recompute(runner.to_device(enc), return_past_key_values=bool(capture))
    returned = time.perf_counter()
    assert vision.call_count == 1
    result.update(projection_calls=counts.result(), scoring_calls=0,
                  end_to_end_ttft_ms=(result["first_token_at_s"] - started) * 1000,
                  request_e2e_ms=(returned - started) * 1000, request_started_at_s=started,
                  vision_forward_count=vision.call_count,
                  cap_reached=(len(result["generated_token_ids"]) == 16 and result["generated_token_ids"][-1] != runner.processor.tokenizer.eos_token_id),
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                  processor=processing)
    diag = {"prompt": prompt, "history": history, "history_entries": entries, "enc_cpu": enc, "vision": vision,
            "image_input_sha256": image_input_hash(enc), "v_token_start": v0, "v_token_num": vn}
    return result, diag


def stored_hit(runner, server, ctx, factory, image_id):
    from mmimpress.llava15.serve import PackedReadTrace, request_llava15_packed
    H = lifecycle(); QA = qa()
    with QA._NoVisionForward(runner) as guard:
        c0 = time.perf_counter(); ctx.reader.drop_all(); conditioning = (time.perf_counter() - c0) * 1000
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
        prompt, history, entries = factory()
        tokenized = runner.processor.tokenizer(prompt, return_tensors="pt")
        suffix = QA._suffix_from_tokenized(runner, tokenized).to(runner.model.device)
        with H.PilotCounts(runner) as counts, PackedReadTrace() as reads:
            result = request_llava15_packed(server, ctx, suffix_ids=suffix, cold=False, image_id=image_id)
        returned = time.perf_counter()
    s = reads.summary()
    assert s["bytes"] == result["io"]["bytes"] and s["preads"] == result["io"]["preads"]
    result.update(projection_calls=counts.result(), scoring_calls=0,
                  end_to_end_ttft_ms=(result["first_token_at_s"] - started) * 1000, request_e2e_ms=(returned - started) * 1000,
                  request_started_at_s=started, vision_forward_count=guard.calls, page_cache_conditioning_ms=conditioning,
                  prompt_prepare_ms=(result["core_started_at_s"] - started) * 1000,
                  cap_reached=(len(result["generated_token_ids"]) == 16 and result["generated_token_ids"][-1] != runner.processor.tokenizer.eos_token_id),
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                  os_pread_trace=s["calls"], pread_calls=s["pread_calls"], preadv_calls=s["preadv_calls"])
    ctx.cache.k = None; ctx.cache.v = None
    return result, {"prompt": prompt, "history": history, "history_entries": entries, "suffix_input_ids": suffix.cpu().tolist()}


def stored_request(runner, server, ctx, factory, image_id):
    from mmimpress.serve import BIAS
    H = lifecycle()
    original = [l.self_attn.forward for l in runner.layers]
    with H.conditioning_trace() as conditioning:
        result, diag = stored_hit(runner, server, ctx, factory, image_id)
    assert all(l.self_attn.forward == f for l, f in zip(runner.layers, original)), "adapter not restored"
    assert not BIAS, "stale mask after request"
    assert conditioning and all(e["status"] == "SUCCESS" for e in conditioning), "page cache conditioning not verified"
    result.update(conditioning_events=conditioning, adapter_restored=True,
                  page_cache_conditioning_method="posix_fadvise_DONTNEED", page_cache_conditioning_excluded_from_ttft=True)
    return result, diag


# -------------------------------------------------------------- persistence
def native_persist(runner, captured, enc, vision, path, image_id, doc, image_input_sha256, extra):
    from mmimpress.llava15.store import persist_captured_llava15_prefix
    return persist_captured_llava15_prefix(runner, captured, enc["input_ids"], vision.result_cpu(), path, budget=doc,
                                           image_id=image_id, capture_stats=vision, image_input_sha256=image_input_sha256,
                                           extra_metadata=extra)


def persist(runner, result, diag, path, image_id, kind, source, docs):
    """Turn-1 piggyback persistence + completion seal (scripts/main_b_support.persist_llava, unchanged)."""
    import main_b_support as support
    vec = kind[len("packed_"):]
    extra = {"dataset": "MT-GQA-reconstructed", "source_dialog_id": source["dialog_id"], "source_turn_id": 1,
             "source_method_id": source["method_id"], "source_execution_id": source["physical_execution_id"],
             "source_logical_request_id": source["logical_request_id"], "run_schema": SCHEMA, "checkpoint_revision": REVISION}
    return support.persist_llava(lambda: native_persist(runner, result["captured_past_key_values"], diag["enc_cpu"], diag["vision"], path,
                                                        image_id, docs[vec], diag["image_input_sha256"], extra), path, kind, source)


def activate(runner, method, path, doc):
    from mmimpress.llava15.serve import Llava15PackedContext
    started = time.perf_counter()
    ctx = Llava15PackedContext(path, runner.model.device, expected_budget=doc)
    ctx.validate_packed_layout()
    ctx.activation = {"host_tensor_bytes": sum(t.numel() * t.element_size() for t in ctx.cache.sys_kv.values()),
                      "pinned_host_buffer_bytes": int(ctx.reader.total), "pinned": bool(ctx.reader.pinned), "gpu_tensor_bytes": 0,
                      "visual_payload_resident_bytes": 0,
                      "file_bytes": sum((Path(path) / n).stat().st_size for n in ("meta.json", "sys_kv.pt")),
                      "completion_integrity": ctx.integrity_activation, "event_id": str(uuid.uuid4()), "event_kind": "activation",
                      "method_id": method}
    ctx.activation["outer_wall_ms"] = (time.perf_counter() - started) * 1000
    return ctx


def warmup(runner, server):
    """One synthetic full-pixel request with the capture hooks (excluded from every metric)."""
    import numpy as np
    from PIL import Image
    from mmimpress.piggyback import VisionForwardCapture
    yy, xx = np.indices((480, 640), dtype=np.uint16)
    pixels = np.stack(((3 * xx + yy) % 256, (xx + 5 * yy) % 256, (7 * xx + 11 * yy) % 256), axis=-1).astype(np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    enc = dict(runner.processor(images=image, text="USER: <image>\nCurrent question Q1: What is shown?\n" + SHORT_ANSWER + " ASSISTANT:",
                                return_tensors="pt"))
    vision = VisionForwardCapture(runner, capture_saliency=True)
    with vision:
        result = server.recompute(runner.to_device(enc), return_past_key_values=True)
    result.pop("captured_past_key_values")
    torch.cuda.synchronize()
    return {"enabled": True, "fixture": "deterministic_in_memory_rgb_pattern_640x480", "fixture_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
            "excluded_from_all_metrics": True, "vision_forward_count": vision.call_count}


# ------------------------------------------------------ independent reference
def fp16rt(x, device):
    """Capture tensor -> FP16 on the host -> BF16 on ``device`` (the store round trip, no file involved)."""
    return x.detach().to(dtype=torch.float16, device="cpu").to(dtype=torch.bfloat16, device=device)


def independent_visible(prefix_len, v_start, counts):
    """Stored-layout visibility: system rows + stored rows [0, k_l)."""
    out = []
    for k in counts:
        vis = torch.zeros(prefix_len, dtype=torch.bool)
        vis[:v_start + k] = True
        out.append(vis)
    return out


def original_visible(prefix_len, v_start, order, counts):
    """Original-coordinate visibility: system rows + original tokens order[:k_l]."""
    out = []
    for k in counts:
        vis = torch.zeros(prefix_len, dtype=torch.bool)
        vis[:v_start] = True
        if k:
            vis[v_start + torch.as_tensor(order[:k], dtype=torch.long)] = True
        out.append(vis)
    return out


def dense_reference_cache(runner, layers, v_start, order, counts, fill=0.0):
    """Stored-layout dense cache built from the Turn-1 capture alone: stored row j < k_l holds the FP16
    round trip of original token order[j]; rows >= k_l hold ``fill`` (0 like production, or a sentinel)."""
    from transformers import DynamicCache
    device = runner.model.device
    p = v_start + N_VISUAL
    cache = DynamicCache()
    scratch = torch.zeros((1, 32, p, 128), dtype=torch.bfloat16, device=device)
    for li, (sk, sv) in enumerate(layers):
        cache.update(scratch, scratch, li)
        k = counts[li]
        for name, src in (("keys", sk), ("values", sv)):
            tgt = getattr(cache.layers[li], name)
            tgt[0, :, :v_start] = fp16rt(src[0, :, :v_start], device)
            if k:
                ids = torch.as_tensor(order[:k], device=src.device)
                tgt[0, :, v_start:v_start + k] = fp16rt(src[0, :, v_start + ids], device)
            if k < N_VISUAL and fill:
                tgt[0, :, v_start + k:p] = fill
    return cache


def full_length_reference_cache(runner, layers, v_start, order=None):
    """Full-length cache: every one of the 576 visual rows is real (FP16 round trip of the capture).
    ``order=None`` keeps the model's original token order; otherwise rows are in the stored order."""
    from transformers import DynamicCache
    device = runner.model.device
    p = v_start + N_VISUAL
    cache = DynamicCache()
    for li, (sk, sv) in enumerate(layers):
        k = fp16rt(sk[0:1, :, :p], device)
        v = fp16rt(sv[0:1, :, :p], device)
        if order is not None:
            ids = torch.as_tensor(order, device=device) + v_start
            k = torch.cat([k[:, :, :v_start], k[:, :, ids]], dim=2)
            v = torch.cat([v[:, :, :v_start], v[:, :, ids]], dim=2)
        cache.update(k.contiguous(), v.contiguous(), li)
    return cache


class MaskHooks:
    """Per-layer visibility applied WITHOUT the production BIAS table: a forward pre-hook on every
    decoder self-attention adds an additive mask (0 / finfo.min) to the attention mask it was given."""

    def __init__(self, runner, visible):
        self.runner, self.visible = runner, visible
        self.handles, self.calls = [], 0

    def __enter__(self):
        from mmimpress.serve import BIAS
        assert not BIAS, "reference must run with an empty production BIAS table"
        neg = torch.finfo(torch.bfloat16).min
        dev = self.runner.model.device
        self.bias = []
        for vis in self.visible:
            b = torch.full((1, 1, 1, vis.numel()), neg, dtype=torch.bfloat16, device=dev)
            b[0, 0, 0, vis.to(dev)] = 0
            self.bias.append(b)
        for li, layer in enumerate(self.runner.layers):
            def hook(module, args, kwargs, li=li):
                mask = kwargs.get("attention_mask")
                assert mask is not None and mask.dim() == 4, "eager decoder must receive a 4-D mask"
                pad = mask.shape[-1] - self.bias[li].shape[-1]
                assert pad >= 0
                kwargs["attention_mask"] = mask + torch.nn.functional.pad(self.bias[li], (0, pad)).to(mask.dtype)
                self.calls += 1
                return args, kwargs
            self.handles.append(layer.self_attn.register_forward_pre_hook(hook, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        from mmimpress.serve import BIAS
        for h in self.handles:
            h.remove()
        assert not BIAS


@torch.no_grad()
def reference_decode(runner, cache, suffix_ids, prefix_len, max_new_tokens=16):
    """Own greedy loop (not Server._decode): prefill over the suffix, then token-by-token argmax."""
    model = runner.model
    dev = model.device
    eos = runner.processor.tokenizer.eos_token_id
    ids = suffix_ids.to(dev).view(1, -1)
    n = ids.shape[1]
    positions = torch.arange(prefix_len, prefix_len + n, device=dev)
    out = model(input_ids=ids, attention_mask=torch.ones(1, prefix_len + n, dtype=torch.long, device=dev),
                position_ids=positions.unsqueeze(0), cache_position=positions, past_key_values=cache, use_cache=True)
    logits = [out.logits[0, -1].detach().float().cpu()]
    tokens = [int(out.logits[0, -1].argmax())]
    cur = prefix_len + n
    while tokens[-1] != eos and len(tokens) < max_new_tokens:
        cp = torch.tensor([cur], device=dev)
        out = model(input_ids=torch.tensor([[tokens[-1]]], device=dev), attention_mask=torch.ones(1, cur + 1, dtype=torch.long, device=dev),
                    position_ids=cp.unsqueeze(0), cache_position=cp, past_key_values=cache, use_cache=True)
        logits.append(out.logits[0, -1].detach().float().cpu())
        tokens.append(int(out.logits[0, -1].argmax()))
        cur += 1
    return {"first_token_id": tokens[0], "generated_token_ids": tokens, "logits": logits,
            "answer": runner.processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()}


def reference_run(runner, cache, visible, suffix_ids, prefix_len):
    with MaskHooks(runner, visible) as hooks:
        out = reference_decode(runner, cache, suffix_ids, prefix_len)
    out["mask_hook_calls"] = hooks.calls
    return out


# ------------------------------------------------- independent store / mask checks
def expected_block(src, v_start, order, k):
    """Bytes the store must hold for one layer / kind: capture -> token-major FP16 -> first k rows of the order.
    Written without the writer's code path (gather by original index instead of permute-then-slice)."""
    ids = torch.as_tensor(order[:k], dtype=torch.long, device=src.device) + v_start
    rows = src[0].index_select(1, ids)                     # (H, k, hd), original dtype
    return rows.to(torch.float16).transpose(0, 1).contiguous().cpu().numpy().tobytes()


def check_packed_slices(path, layers, v_start, order, counts):
    """Compare every layer slice of ``kv_packed.bin`` with the capture, bit for bit; offsets are recomputed here."""
    import numpy as np
    raw = np.fromfile(Path(path) / "kv_packed.bin", dtype=np.uint8)
    total = sum(2 * k * ROW_BYTES for k in counts)
    out = {"file_bytes": int(raw.size), "expected_bytes": int(total), "size_equal": int(raw.size) == int(total), "mismatched": []}
    off = 0
    for li, (sk, sv) in enumerate(layers):
        for kind, src in (("k", sk), ("v", sv)):
            n = counts[li] * ROW_BYTES
            want = expected_block(src, v_start, order, counts[li])
            got = raw[off:off + n].tobytes()
            if len(want) != n or got != want:
                out["mismatched"].append({"layer": li, "kind": kind})
            off += n
    out["offset_end_equals_size"] = off == int(raw.size)
    out["file_sha256"] = hashlib.sha256(raw.tobytes()).hexdigest()
    out["status"] = "PASS" if out["size_equal"] and out["offset_end_equals_size"] and not out["mismatched"] else "FAIL"
    return out


def check_masks(bias_table, prefix_len, v_start, counts):
    """Installed production masks vs the independent visibility (system rows + stored rows [0, k_l)), bit for bit."""
    visible = independent_visible(prefix_len, v_start, counts)
    bad = []
    if sorted(bias_table) != list(range(len(counts))):
        return {"status": "FAIL", "reason": "mask table does not cover every layer", "layers": sorted(bias_table)}
    rows = []
    for li, vis in enumerate(visible):
        b = bias_table[li]
        ok = tuple(b.shape) == (1, 1, 1, prefix_len)
        allowed = (b[0, 0, 0].detach().cpu() == 0) if ok else None
        neg = (b[0, 0, 0].detach().cpu() == torch.finfo(b.dtype).min) if ok else None
        if not ok or not torch.equal(allowed, vis) or not torch.equal(neg, ~vis):
            bad.append(li)
        rows.append(int(allowed[v_start:].sum()) if ok else -1)
    return {"status": "PASS" if not bad else "FAIL", "bad_layers": bad, "visible_visual_rows_per_layer": rows,
            "visible_rows_equal_k_l": rows == list(counts)}


def check_cache_bits(cache, layers, v_start, order, counts):
    """Production dense cache vs the capture: system rows and stored rows [0, k_l) are the FP16 round trip of the
    capture rows order[:k_l]; rows beyond k_l are untouched zeros."""
    p = v_start + N_VISUAL
    bad = []
    for li, (sk, sv) in enumerate(layers):
        k = counts[li]
        for name, src in (("keys", sk), ("values", sv)):
            actual = getattr(cache.layers[li], name)
            ok = tuple(actual.shape[:3]) == (1, 32, p) and torch.equal(actual[0, :, :v_start], fp16rt(src[0, :, :v_start], actual.device))
            if ok and k:
                ids = torch.as_tensor(order[:k], device=src.device)
                ok = torch.equal(actual[0, :, v_start:v_start + k], fp16rt(src[0, :, v_start + ids], actual.device))
            if ok and k < N_VISUAL:
                tail = actual[0, :, v_start + k:p]
                ok = bool((tail == 0).all())
            if not ok:
                bad.append({"layer": li, "kind": name})
    return {"status": "PASS" if not bad else "FAIL", "bad": bad}

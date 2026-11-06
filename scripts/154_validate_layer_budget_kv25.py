#!/usr/bin/env python3
"""Gates for the layer-wise PrefixStore-KV25 budget (contract A7).

--stage cpu   : full existing CPU suite + the new G5 suite -> cpu_tests.log
--stage g3    : Sigma k_l == B(N) and bounds for every CAL/DEV/HOLD N and every
                frozen vector (CPU) -> gates/g3_budget_invariance.json
--stage gpu   : G1 (uniform regression vs the untouched prefix25 writer and the
                recorded fresh-[B] outputs), G2 (independent F-front reference:
                visible sets, cache bits, masks, logits 1e-4, reads), G4
                (question invariance) on the five fixed LLaVA cases
                -> gates/gpu_gates.json

Reuses the frozen prefix25-persistence experiment class (scripts/106) for the
Turn-1 capture, the prefix25 persistence, activation and the traced hit, so the
existing arm is literally the existing code path.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

from mmimpress.layer_budget import (budget_sha256, keep_bounds,
                                    layer_keep_counts, load_budget_document,
                                    total_row_budget)
from mmimpress.prefix25 import ReadTrace, process_io, seal_integrity, sha
from mmimpress.serve import BIAS
import mmimpress.serve as serving

REF_RUN = ROOT / "runs/prefix25_persistence_20261001T050113Z"
FIXED_INDEX = [0, 3, 4, 5, 6]
ATOL = RTOL = 1e-4
SENTINEL = 10000.0
SEED = 1234


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=1, allow_nan=False, default=_safe)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def _safe(x):
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x))


# ------------------------------------------------------------------- cpu
def stage_cpu(run):
    """Full existing CPU suite.  The GLM-4.6V test modules were written for a
    transformers 5.13.1 environment (runs/glm46v_prefixstore3_v6_*/environment.json)
    and import none of the modules touched here; in this 4.57.6 environment
    they fail on glm4v API lookups before any assertion.  They are run
    separately and recorded as pre-existing environment errors, never hidden."""
    env = {**os.environ, "PYTHONPATH": ".:scripts:tests", "PYTHONDONTWRITEBYTECODE": "1"}
    modules = sorted(p.stem for p in (ROOT / "tests").glob("test_*.py"))
    glm = [m for m in modules if m.startswith("test_glm46v")]
    main = [m for m in modules if m not in glm]
    log = run / "cpu_tests.log"
    with log.open("w") as f:
        p = subprocess.run([sys.executable, "-m", "unittest"] + main, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
    text = log.read_text()
    ok = p.returncode == 0 and "\nOK" in text
    glm_log = run / "cpu_tests_glm_preexisting.log"
    with glm_log.open("w") as f:
        q = subprocess.run([sys.executable, "-m", "unittest"] + glm, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
    gtext = glm_log.read_text()
    glm_errors = [l for l in gtext.splitlines() if l.startswith("ERROR:") or l.startswith("FAIL:")]
    imports_touched = []
    for m in glm:
        src = (ROOT / "tests" / f"{m}.py").read_text()
        if any(x in src for x in ("mmimpress.serve", "mmimpress.store", "mmimpress.piggyback", "mmimpress.prefix25", "mmimpress.layer_budget")):
            imports_touched.append(m)
    write(run / "gates" / "cpu_tests.json", {
        "status": "PASS" if ok and not imports_touched else "FAIL", "returncode": p.returncode, "modules": len(main),
        "log_sha256": sha(log), "tail": text[-400:],
        "glm_modules_run_separately": glm, "glm_returncode": q.returncode, "glm_errors": glm_errors,
        "glm_log_sha256": sha(glm_log), "glm_imports_touched_modules": imports_touched,
        "glm_note": "pre-existing environment failures (transformers 5.13.1 API) independent of the modified modules; reported, not hidden"})
    print("CPU", "PASS" if ok else "FAIL", "| GLM pre-existing errors:", len(glm_errors), flush=True)
    assert ok and not imports_touched


# -------------------------------------------------------------------- g3
def stage_g3(run):
    dist = json.loads((run / "phase0" / "n_distribution.json").read_text())
    vectors = {p.stem.replace("layer_budget_llava_", ""): load_budget_document(p)
               for p in sorted((run / "phase2").glob("layer_budget_llava_*.json"))}
    assert vectors, "phase2 vectors missing"
    out = {"status": "PASS", "vectors": {a: v["layer_budget_sha256"] for a, v in vectors.items()}, "splits": {}}
    for split in ("cal", "dev", "hold"):
        rows = dist[split]["per_image"]
        checked = 0
        for row in rows:
            n = int(row["N"])
            budget = total_row_budget(n)
            assert budget == 32 * ((n + 3) // 4) == row["B"]
            lo, hi = keep_bounds(n)
            for a, v in vectors.items():
                k = layer_keep_counts(v["layer_keep_ratios"], n)
                assert sum(k) == budget, (split, row["image_id"], a)
                assert all(lo <= x <= hi for x in k), (split, row["image_id"], a, k)
                if a == "U":
                    assert k == [(n + 3) // 4] * 32
                checked += 1
        out["splits"][split] = {"images": len(rows), "vector_image_checks": checked, "sum_equals_B": True, "bounds": True}
    write(run / "gates" / "g3_budget_invariance.json", out)
    print("G3", out["splits"], flush=True)


# ------------------------------------------------------------------- gpu
def independent_counts(ratios, n):
    """Independent Hamilton apportionment (no library code)."""
    from fractions import Fraction
    L = len(ratios)
    budget = L * (-(-n // 4))
    cap = min(n, math.ceil(Fraction(3, 5) * n))
    q = [r * n for r in ratios]
    k = [min(cap, int(math.floor(x))) for x in q]
    frac = [x - math.floor(x) for x in q]
    order = sorted(range(L), key=lambda i: (-frac[i], i))
    need = budget - sum(k)
    while need:
        for i in order:
            if need and k[i] < cap:
                k[i] += 1
                need -= 1
    return k


def fp16rt(x, device):
    return x.detach().to(dtype=torch.float16, device="cpu").to(dtype=torch.bfloat16, device=device)


def independent_cache(runner, meta, layers, ranking, counts):
    from transformers import DynamicCache
    device = runner.model.device
    p, h, hd = int(meta["prefix_len"]), int(meta["num_heads"]), int(meta["head_dim"])
    vstart, n = int(meta["v_token_start"]), int(meta["n_spatial"])
    seps = [int(x) for x in meta["newline_idx"]]
    cache = DynamicCache()
    scratch = torch.zeros((1, h, p, hd), dtype=torch.bfloat16, device=device)
    for li, (sk, sv) in enumerate(layers):
        cache.update(scratch, scratch, li)
        k = counts[li]
        for name, src in (("keys", sk), ("values", sv)):
            tgt = getattr(cache.layers[li], name)
            tgt[0, :, :vstart] = fp16rt(src[0, :, :vstart], device)
            if k:
                ids = torch.as_tensor(ranking[:k], device=src.device)
                tgt[0, :, vstart:vstart + k] = fp16rt(src[0, :, vstart + ids], device)
            ids = torch.as_tensor(seps, device=src.device)
            tgt[0, :, vstart + n:p] = fp16rt(src[0, :, vstart + ids], device)
    return cache


def independent_visible(meta, counts):
    p, vstart, n, v = int(meta["prefix_len"]), int(meta["v_token_start"]), int(meta["n_spatial"]), int(meta["v_token_num"])
    out = []
    for k in counts:
        vis = torch.zeros(p, dtype=torch.bool)
        vis[:vstart + k] = True
        vis[vstart + n:vstart + v] = True
        out.append(vis)
    return out


def install_independent_masks(visible, device):
    BIAS.clear()
    neg = torch.finfo(torch.bfloat16).min
    for li, vis in enumerate(visible):
        b = torch.full((1, 1, 1, vis.numel()), neg, dtype=torch.bfloat16, device=device)
        b[0, 0, 0, vis.to(device)] = 0
        BIAS[li] = b


class LayerMaskTrace:
    """Check the actual eager masks per layer (prefill + every decode step)."""

    def __init__(self, visible, prefix_len):
        self.visible, self.prefix_len = visible, prefix_len
        self.calls = 0
        self.prefill = set()
        self.decode = 0

    def __enter__(self):
        self.original = serving._ORIG_EAGER

        def checked(module, q, k, v, mask, **kw):
            li = int(module.layer_idx)
            qlen, klen = int(q.shape[-2]), int(k.shape[-2])
            assert mask is not None
            m = mask.detach().cpu()
            allowed = m == 0
            for head in range(m.shape[1]):
                assert torch.equal(allowed[0, head, :, :self.prefix_len], self.visible[li].expand(qlen, -1)), f"layer {li} prefix mask"
                qpos = torch.arange(klen - qlen, klen)[:, None]
                spos = torch.arange(self.prefix_len, klen)[None, :]
                assert torch.equal(allowed[0, head, :, self.prefix_len:], spos <= qpos), f"layer {li} causal"
            self.calls += 1
            if qlen > 1:
                self.prefill.add(li)
            else:
                self.decode += 1
            return self.original(module, q, k, v, mask, **kw)

        serving._ORIG_EAGER = checked
        return self

    def __exit__(self, *exc):
        serving._ORIG_EAGER = self.original


def reference_run(server, ctx, question, layers, ranking, counts, sentinel=False):
    from mmimpress.serve import suffix_ids_for
    cache = independent_cache(server.runner, ctx.meta, layers, ranking, counts)
    visible = independent_visible(ctx.meta, counts)
    install_independent_masks(visible, server.runner.model.device)
    if sentinel:
        vstart, n = int(ctx.meta["v_token_start"]), int(ctx.meta["n_spatial"])
        for li, k in enumerate(counts):
            if k < n:
                for name in ("keys", "values"):
                    getattr(cache.layers[li], name)[0, :, vstart + k:vstart + n] = SENTINEL
    logits = []
    h = server.runner.model.register_forward_hook(lambda m, a, o: logits.append(o.logits[0, -1].detach().float().cpu()) if hasattr(o, "logits") else None)
    try:
        answer, first, timing = server._decode(cache, suffix_ids_for(server.runner, question).to(server.runner.model.device), int(ctx.meta["prefix_len"]))
    finally:
        h.remove()
        BIAS.clear()
        del cache
    return {"answer": answer, "first_token_id": first, "generated_token_ids": timing["generated_token_ids"], "logits": logits}


def production_traced(exp, ctx, path, image, question, layers, ranking, counts):
    """Serve through the public API; observe cache bits, masks, reads, BIAS."""
    server = exp.server
    meta = ctx.meta
    visible = independent_visible(meta, counts)
    original = server._decode
    box = {"logits": []}
    vstart, n, p = int(meta["v_token_start"]), int(meta["n_spatial"]), int(meta["prefix_len"])

    def traced(cache, suffix, prefix_len):
        # installed bits before the forward
        assert len(BIAS) == len(layers)
        for li, (sk, sv) in enumerate(layers):
            k = counts[li]
            b = BIAS[li]
            assert tuple(b.shape) == (1, 1, 1, p)
            assert torch.equal((b[0, 0, 0].detach().cpu() == 0), visible[li]), f"layer {li} BIAS"
            for name, src in (("keys", sk), ("values", sv)):
                actual = getattr(cache.layers[li], name)
                assert torch.equal(actual[0, :, :vstart], fp16rt(src[0, :, :vstart], actual.device))
                if k:
                    ids = torch.as_tensor(ranking[:k], device=src.device)
                    assert torch.equal(actual[0, :, vstart:vstart + k], fp16rt(src[0, :, vstart + ids], actual.device)), f"layer {li} {name} bits"
                if k < n:
                    tail = actual[0, :, vstart + k:vstart + n]
                    assert torch.equal(tail, torch.zeros_like(tail)), f"layer {li} rows beyond k_l must be unloaded"
                ids = torch.as_tensor([int(x) for x in meta["newline_idx"]], device=src.device)
                assert torch.equal(actual[0, :, vstart + n:p], fp16rt(src[0, :, vstart + ids], actual.device))
        box["bias"] = {li: BIAS[li].detach().cpu().clone() for li in BIAS}
        h = server.runner.model.register_forward_hook(lambda m, a, o: box["logits"].append(o.logits[0, -1].detach().float().cpu()) if hasattr(o, "logits") else None)
        try:
            with LayerMaskTrace(visible, p) as masks:
                out = original(cache, suffix, prefix_len)
            assert masks.prefill == set(range(len(layers)))
            box["mask_calls"] = masks.calls
            box["decode_mask_calls"] = masks.decode
            return out
        finally:
            h.remove()

    server._decode = traced
    try:
        with ReadTrace(path) as reads:
            value = exp.hit(ctx, path, image, {"question": question}, [], "B", "gqa", False)
    finally:
        server._decode = original
        BIAS.clear()
    row_bytes = int(meta["num_heads"]) * int(meta["head_dim"]) * 2
    want = {str(path / f"layer_{li:02d}/{kind}.bin"): (0, counts[li] * row_bytes) for li in range(len(layers)) for kind in ("k", "v")}
    want[str(path / "sep_kv.bin")] = (0, (path / "sep_kv.bin").stat().st_size)
    actual = {c["path"]: (c["offset"], c["returned"]) for c in reads.calls}
    assert actual == want and len(reads.calls) == len(want) == 65, "independent per-layer read plan differs"
    assert all(c["returned"] == c["requested"] for c in reads.calls)
    assert not any(any(x in c["path"] for x in ("probe_k", "visionzip_layout", "static", "score")) for c in reads.calls)
    assert value["io"]["bytes"] == sum(c["returned"] for c in reads.calls) and value["io"]["preads"] == len(reads.calls)
    reported = value.get("k_per_layer") or [value["k_target"]] * len(layers)   # prefix25 arm reports one k
    assert reported == counts and value["attended_content_kv_count_per_layer"] == counts
    assert value["keep_count_per_layer"] == [k + int(meta["v_token_num"]) - n for k in counts]
    assert value["selected_original_ids"] == ranking[:max(counts)]
    assert value["vision_calls"] == 0 and value["online_query_score_calls"] == 0
    box["result"] = value
    box["reads"] = {"calls": len(reads.calls), "bytes": sum(c["returned"] for c in reads.calls)}
    return box


def compare(ref, prod_logits, prod_first, prod_tokens, prod_answer):
    a, b = ref["logits"], prod_logits
    assert len(a) == len(b) > 0, (len(a), len(b))
    close = all(torch.allclose(x, y, atol=ATOL, rtol=RTOL) for x, y in zip(a, b))
    out = {"all_step_logits_within_1e-4": close, "first_logits_bitwise": bool(torch.equal(a[0], b[0])),
           "all_logits_bitwise": all(torch.equal(x, y) for x, y in zip(a, b)),
           "max_abs_delta": max(float((x - y).abs().max()) for x, y in zip(a, b)),
           "first_token_exact": ref["first_token_id"] == prod_first, "tokens_exact": ref["generated_token_ids"] == prod_tokens,
           "answer_exact": ref["answer"] == prod_answer, "steps": len(a)}
    assert close and out["first_token_exact"] and out["tokens_exact"] and out["answer_exact"], out
    return out


def logits_sha(logits):
    return [hashlib.sha256(x.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest() for x in logits]


def persist_layerwise(exp, capture, path, image, doc):
    from mmimpress.piggyback import persist_captured_visual_prefix
    before = process_io()
    started = time.perf_counter()
    enc = capture["enc"]
    receipt = persist_captured_visual_prefix(
        exp.runner, capture["cache"], enc["input_ids"], enc["image_sizes"][0], capture["vision"].result_cpu(), path,
        capture_stats=capture["vision"], storage_policy="prefix25_layerwise", layer_budget=doc,
        image_id=image["image_id"], model_id=exp.runner.model_id, image_input_sha256=image["image_sha256"],
        extra_metadata={"checkpoint_revision": exp.runner.cfg._commit_hash, "ablation": "layer_budget_kv25"})
    t = time.perf_counter()
    files = seal_integrity(path)
    hash_ms = (time.perf_counter() - t) * 1e3
    total = (time.perf_counter() - started) * 1e3
    after = process_io()
    meta = receipt["meta"]
    return {"physical_write_id": str(uuid.uuid4()), "store_path": str(path), "arm": doc["alloc_id"], "image_id": image["image_id"],
            "metadata": meta, "files": files, "timing_ms": receipt["timing_ms"], "hash_and_seal_ms": hash_ms, "persistence_ms": total,
            "os_write_bytes": after["wchar"] - before["wchar"], "os_write_calls": after["syscw"] - before["syscw"],
            "file_logical_bytes": sum(x["size"] for x in files.values()), "allocated_bytes": sum(x["allocated_bytes"] for x in files.values()),
            "N": meta["original_content_count"], "k_per_layer": meta["layer_keep_counts"], "layer_budget_sha256": meta["layer_budget_sha256"]}


def gpu_case(exp, sample, path, docs, ref_out):
    P = sys.modules["prefix25_persistence_existing"]
    from mmimpress.model import cache_layers
    turn = {"question": sample["question"]}
    out = {"image_id": sample["image_id"], "question_id": sample["question_id"]}
    val, cap = exp.pixel(sample, turn, [], "B", "gqa")
    enc = cap["enc"]
    layers = cache_layers(cap["cache"])
    vs, vn = exp.runner.visual_span(enc["input_ids"])
    _, _, _, sep = exp.runner.anyres_layout(enc["image_sizes"][0], vn)
    n = vn - len(sep)
    k_u = (n + 3) // 4
    ranking = exp.ref._score_rank(exp.runner, cap["vision"].result_cpu(), enc["image_sizes"][0], vn, sep)
    out["N"], out["k_u"] = n, k_u
    # ---- stores: existing prefix25 writer (B) and layerwise uniform (B_U), layerwise F-front (B_F)
    rec_b = exp.persist(cap, path / "B", sample, "B")
    rec_u = persist_layerwise(exp, cap, path / "B_U", sample, docs["U"])
    rec_f = persist_layerwise(exp, cap, path / "B_F", sample, docs["F-front"])
    assert rec_b["metadata"]["order"] == rec_u["metadata"]["order"] == rec_f["metadata"]["order"] == ranking + sep
    # ---- G1: byte identity of payload files and shared meta keys
    files_b, files_u = rec_b["files"], rec_u["files"]
    payload = sorted(k for k in files_b if k != "meta.json" and k != "integrity.json")
    assert payload == sorted(k for k in files_u if k not in ("meta.json", "integrity.json"))
    assert all(files_b[k]["sha256"] == files_u[k]["sha256"] and files_b[k]["size"] == files_u[k]["size"] for k in payload), "G1 payload bytes differ"
    mb, mu = rec_b["metadata"], rec_u["metadata"]
    # 'ablation' is the harness provenance label written by the two persist
    # helpers (prefix25 vs layer_budget_kv25); it is not store content.
    skip = {"format", "storage_policy", "order_semantics", "rounding_policy", "stored_row_to_original", "bytes_visual_kv", "ablation"}
    shared = (set(mb) & set(mu)) - skip
    diff = [k for k in shared if mb[k] != mu[k]]
    assert not diff, f"G1 shared meta keys differ: {diff}"
    assert mb["bytes_visual_kv"] == mu["bytes_visual_kv"] and mu["stored_row_to_original"] == mb["stored_row_to_original"]
    assert mu["layer_keep_counts"] == [k_u] * 32 and mu["payload_rows_per_layer"] == [mb["payload_rows"]] * 32
    out["G1_store"] = {"payload_files": len(payload), "payload_bytes_identical": True, "shared_meta_keys": len(shared), "meta_keys_identical": True,
                       "payload_bytes": sum(files_b[k]["size"] for k in payload)}
    counts_u = [k_u] * 32
    counts_f = independent_counts(docs["F-front"]["layer_keep_ratios"], n)
    assert counts_f == rec_f["k_per_layer"] == layer_keep_counts(docs["F-front"]["layer_keep_ratios"], n)
    assert sum(counts_f) == sum(counts_u) == 32 * k_u
    out["k_per_layer_F"] = counts_f
    # ---- serve: B (existing), B_U, B_F; traced
    served = {}
    for arm, counts in (("B", counts_u), ("B_U", counts_u), ("B_F", counts_f)):
        ctx, act = exp.activate(path / arm, sample, "B")
        served[arm] = production_traced(exp, ctx, path / arm, sample, turn["question"], layers, ranking, counts)
        served[arm]["ctx"] = ctx
    # G1 serving identity B vs B_U
    b, u = served["B"], served["B_U"]
    assert all(torch.equal(b["bias"][li], u["bias"][li]) for li in b["bias"]), "G1 masks differ"
    assert b["result"]["first_token_id"] == u["result"]["first_token_id"]
    assert b["result"]["generated_token_ids"] == u["result"]["generated_token_ids"]
    assert b["result"]["prediction"] == u["result"]["prediction"]
    assert b["reads"] == u["reads"]
    out["G1_serving"] = {"masks_identical": True, "first_token_identical": True, "generated_ids_identical": True, "reads_identical": b["reads"],
                         "logits_bitwise": all(torch.equal(x, y) for x, y in zip(b["logits"], u["logits"])),
                         "logits_max_abs_delta": max(float((x - y).abs().max()) for x, y in zip(b["logits"], u["logits"]))}
    # G1 vs recorded fresh-[B] output of the prior run
    out["G1_prior_reference"] = {"tokens_and_prediction_reproduced": [b["result"][x] for x in ("first_token_id", "generated_token_ids", "prediction")] == [ref_out[x] for x in ("first_token_id", "generated_token_ids", "prediction")],
                                 "logits_sha256_equal": logits_sha(b["logits"]) == ref_out["logits_sha256"],
                                 "B_U_logits_sha256_equal": logits_sha(u["logits"]) == ref_out["logits_sha256"]}
    assert out["G1_prior_reference"]["tokens_and_prediction_reproduced"], "prior fresh-[B] output not reproduced"
    # ---- G2: independent reference for F-front (and, as a control, for uniform)
    for arm, counts in (("B_F", counts_f), ("B_U", counts_u)):
        ref = reference_run(exp.server, served[arm]["ctx"], turn["question"], layers, ranking, counts)
        prod = served[arm]
        cmp = compare(ref, prod["logits"], prod["result"]["first_token_id"], prod["result"]["generated_token_ids"], prod["result"]["prediction"])
        # sentinel: rows beyond k_l filled with a finite value must not change the reference
        sen = reference_run(exp.server, served[arm]["ctx"], turn["question"], layers, ranking, counts, sentinel=True)
        cmp["sentinel_reference_bitwise_identical"] = all(torch.equal(x, y) for x, y in zip(ref["logits"], sen["logits"])) and sen["generated_token_ids"] == ref["generated_token_ids"]
        assert cmp["sentinel_reference_bitwise_identical"]
        cmp["mask_calls"] = prod["mask_calls"]
        cmp["reads"] = prod["reads"]
        out[f"G2_{arm}"] = cmp
    # ---- G4: question invariance on B_F and B_U (fresh contexts after source release)
    for arm in ("B", "B_U", "B_F"):
        served[arm]["ctx"].close()
    del cap, layers, enc, val
    gc.collect()
    torch.cuda.empty_cache()
    g4 = {}
    for arm, counts in (("B_F", counts_f), ("B_U", counts_u)):
        ctx, _ = exp.activate(path / arm, sample, "B")
        a = exp.hit(ctx, path / arm, sample, turn, [], "B", "gqa", True)
        c = exp.hit(ctx, path / arm, sample, {"question": "What is visible?"}, [(turn["question"], a["prediction"])], "B", "mt", True)
        assert a["k_per_layer"] == c["k_per_layer"] == counts
        assert a["selected_original_ids"] == c["selected_original_ids"] == ranking[:max(counts)]
        assert a["vision_calls"] == c["vision_calls"] == 0 and a["online_query_score_calls"] == c["online_query_score_calls"] == 0
        assert (a["actual_io"]["bytes"], a["actual_io"]["preads"]) == (c["actual_io"]["bytes"], c["actual_io"]["preads"])
        # fresh hit reproduces the traced production output of the same store
        prod = served[arm]
        assert a["generated_token_ids"] == prod["result"]["generated_token_ids"] and a["prediction"] == prod["result"]["prediction"]
        g4[arm] = {"k_per_layer_unchanged": True, "selected_ids_unchanged": True, "vision_and_query_score_calls_zero": True,
                   "bytes": a["actual_io"]["bytes"], "preads": a["actual_io"]["preads"], "fresh_hit_reproduces_traced": True,
                   "fresh_logits_sha256_equal_traced": logits_sha(a["all_logits"]) == logits_sha(prod["logits"])}
        ctx.close()
    assert g4["B_F"]["bytes"] == g4["B_U"]["bytes"] and g4["B_F"]["preads"] == g4["B_U"]["preads"] == 65
    out["G4"] = g4
    out["bytes_layerwise_equal_uniform"] = True
    out["persistence"] = {a: {k: r[k] for k in ("persistence_ms", "file_logical_bytes", "os_write_bytes")} for a, r in (("B", rec_b), ("B_U", rec_u), ("B_F", rec_f))}
    out["status"] = "PASS"
    return out


def stage_gpu(run):
    P = load("prefix25_persistence_existing", "106_prefix25_persistence.py")
    docs = {a: load_budget_document(run / "phase2" / f"layer_budget_llava_{a}.json") for a in ("U", "F-front")}
    manifest = json.loads((run / "correctness_manifest.json").read_text())
    samples = [manifest["samples"][i] for i in FIXED_INDEX]
    ref_out = {s["image_id"]: s["fresh_B_output"] for s in json.loads((REF_RUN / "llava/final_gpu_regression.json").read_text())["samples"]}
    P.exclusive()
    torch.set_grad_enabled(False)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)
    exp = P.Experiment(run, "llava")
    gate = {"status": "RUNNING", "atol": ATOL, "rtol": RTOL, "vectors": {a: d["layer_budget_sha256"] for a, d in docs.items()}, "samples": []}
    base = run / "llava" / "gpu_gates"
    try:
        for i, s in enumerate(samples):
            path = base / f"{i:02d}_{s['image_id']}"
            path.mkdir(parents=True, exist_ok=False)
            gate["samples"].append(gpu_case(exp, s, path, docs, ref_out[s["image_id"]]))
            write(run / "gates" / "gpu_gates.json", gate)
            print("GPU gates", i + 1, len(samples), "PASS", flush=True)
            # stores of validated cases are kept (small: three B-sized stores per case)
        gate["status"] = "PASS"
        gate["G1"] = gate["G2"] = gate["G4"] = "PASS"
    except Exception:
        gate["status"] = "FAIL"
        gate["error"] = traceback.format_exc()
        print(gate["error"], flush=True)
        raise
    finally:
        write(run / "gates" / "gpu_gates.json", gate)
        exp.close_contexts()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stage", choices=["cpu", "g3", "gpu"], required=True)
    args = ap.parse_args()
    run = Path(args.run).resolve()
    (run / "gates").mkdir(exist_ok=True)
    if args.stage == "cpu":
        stage_cpu(run)
    elif args.stage == "g3":
        stage_g3(run)
    else:
        if not (run / "correctness_manifest.json").exists():
            import shutil
            shutil.copy2(REF_RUN / "correctness_manifest.json", run / "correctness_manifest.json")
        stage_gpu(run)


if __name__ == "__main__":
    main()

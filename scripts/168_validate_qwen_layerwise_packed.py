#!/usr/bin/env python3
"""Gates QG1-QG6 for the Qwen layer-wise packed PrefixStore (QG7 = 169 --phase integration).

  --stage cpu : QG1 (vectors, row budgets, validator fixtures, no LLaVA-default reliance) and
                the complete existing CPU test suite
  --stage gpu : QG2 U regression, QG3 layer-wise implementation, QG4 FP64 semantics,
                QG5 negative fixtures, QG6 I/O -- on the 10 frozen core pairs

Validation samples are correctness fixtures, not a quality holdout.  A failing gate is recorded
and stops the run; thresholds and samples are fixed by docs/qwen_layer_budget_pyr_contract.md.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import qwen_pyr_support as Q
from qwen_pyr_support import ROOT, load_script, read, sha, write

L = 28
FOLLOWUP_QUESTION = "What is visible?"


# ============================================================== independent pieces
def hamilton(ratios, n, r_max="0.6"):
    budget = L * ((n + 3) // 4)
    cap = min(n, math.ceil(Fraction(r_max) * n))
    counts = [min(cap, math.floor(float(r) * n)) for r in ratios]
    rema = [float(r) * n - math.floor(float(r) * n) for r in ratios]
    deficit = budget - sum(counts)
    order = sorted(range(L), key=lambda i: (-rema[i], i))
    while deficit > 0:
        for i in order:
            if deficit == 0:
                break
            if counts[i] < cap:
                counts[i] += 1
                deficit -= 1
    return counts


def bind(run):
    return {"config_sha256": sha(run / "config.json"), "contract_sha256": read(run / "config.json")["contract_sha256"]}


# ========================================================================= QG1 (CPU)
BUDGET_FUNCS = {"validate_ratios": {"num_layers", "total_ratio", "r_min", "r_max"}, "layer_keep_counts": {"num_layers", "total_ratio", "r_min", "r_max"},
                "validate_keep_counts": {"num_layers", "total_ratio", "r_min", "r_max"}, "total_row_budget": {"num_layers", "total_ratio"},
                "keep_bounds": {"r_min", "r_max"}, "budget_document": {"num_layers", "total_ratio", "r_min", "r_max"},
                "uniform_ratios": {"num_layers", "ratio"}, "clip_redistribute": {"num_layers", "total_ratio", "r_min", "r_max"},
                "front_schedule": {"num_layers", "total_ratio", "r_min", "r_max"}, "load_budget_document": None}


def static_default_scan():
    """Every layer_budget call in the new sources must name the model-specific arguments
    (or expand a frozen kwargs dict); load_budget_document (validates with the document's own
    fields but is LLaVA-era) must not be called at all."""
    calls, offenders = [], []
    for rel in Q.NEW_PATHS:
        if not rel.endswith(".py") or rel == Q.REPORT_SCRIPT or rel.startswith("tests/"):   # serving/experiment code only: the report script is post-hoc, the unit tests call the defaults on purpose inside assertRaises
            continue
        tree = ast.parse((ROOT / rel).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else None
            if name not in BUDGET_FUNCS:
                continue
            need = BUDGET_FUNCS[name]
            named = {k.arg for k in node.keywords if k.arg}
            starred = any(k.arg is None for k in node.keywords)
            ok = need is not None and (need <= named or starred)
            calls.append({"file": rel, "line": node.lineno, "function": name, "named": sorted(named), "kwargs_expansion": starred, "explicit": ok})
            if not ok:
                offenders.append(calls[-1])
    return {"calls": calls, "offenders": offenders, "status": "PASS" if calls and not offenders else "FAIL"}


def stage_cpu(run):
    import unittest
    from mmimpress.layer_budget import budget_sha256
    gates = run / "gates"
    gates.mkdir(exist_ok=True)
    out = {"gate": "QG1", **bind(run)}
    docs = Q.load_vectors(run)
    out["vectors"] = {a: {"sha256": docs[a]["layer_budget_sha256"], "expected": Q.RATIO_SHA[a], "recomputed": budget_sha256(Q.vector(a)),
                          "sum": sum(docs[a]["layer_keep_ratios"]), "num_layers": docs[a]["num_layers"], "r_min": docs[a]["r_min"], "r_max": docs[a]["r_max"]} for a in Q.ALLOCS}
    ok = all(v["sha256"] == v["expected"] == v["recomputed"] and abs(v["sum"] - 7.0) < 1e-9 and (v["num_layers"], v["r_min"], v["r_max"]) == (28, 0.0, 0.6)
             for v in out["vectors"].values())
    hold = sorted(x["N"] for x in read(run / "phase0" / "n_distribution_hold.json")["per_image"])
    special = [1, 3, 4, 5, 63, 64, 65, 260, 345, 529, 1024]
    budget = {}
    for n in sorted(set(hold) | set(special)):
        for a in Q.ALLOCS:
            counts = Q.keep_counts(docs[a], n)
            good = sum(counts) == 28 * math.ceil(n / 4) and counts == hamilton(docs[a]["layer_keep_ratios"], n)
            budget[f"{a}:{n}"] = {"sum": sum(counts), "expected": 28 * math.ceil(n / 4), "min": min(counts), "independent_hamilton_equal": counts == hamilton(docs[a]["layer_keep_ratios"], n)}
            ok = ok and good
    out["row_budget"] = {"hold_images": len(hold), "distinct_hold_N": len(set(hold)), "special_N": special, "cases": len(budget), "all_exact": all(v["sum"] == v["expected"] for v in budget.values()),
                         "zero_row_layers_on_hold": {a: [n for n in sorted(set(hold)) if budget[f"{a}:{n}"]["min"] == 0] for a in Q.ALLOCS}, "detail": budget}
    ok = ok and not any(out["row_budget"]["zero_row_layers_on_hold"].values())
    bad = [n for n in range(1, 2001) if Q.keep_counts(docs["U"], n) != [(n + 3) // 4] * 28]
    out["uniform_1_to_2000"] = {"mismatches": bad, "status": "PASS" if not bad else "FAIL"}
    ok = ok and not bad
    out["static_explicit_argument_scan"] = static_default_scan()
    ok = ok and out["static_explicit_argument_scan"]["status"] == "PASS"
    # validator / writer / hook fixtures: the dedicated unit-test module, test by test
    sys.path.insert(0, str(ROOT / "tests"))
    suite = unittest.defaultTestLoader.loadTestsFromName("test_qwen_layerwise_packed")

    class Recorder(unittest.TextTestResult):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.outcomes = {}

        def addSuccess(self, test):
            super().addSuccess(test)
            self.outcomes[test.id()] = "PASS"

        def addFailure(self, test, err):
            super().addFailure(test, err)
            self.outcomes[test.id()] = "FAIL"

        def addError(self, test, err):
            super().addError(test, err)
            self.outcomes[test.id()] = "ERROR"
    with (gates / "qg1_unittest.log").open("x") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=Recorder).run(suite)
    out["fixture_tests"] = {"outcomes": result.outcomes, "run": result.testsRun, "log_sha256": sha(gates / "qg1_unittest.log"),
                            "covers": {"tampered meta/SHA/row count/length/r_min/num_layers rejected": "PackedStoreTests.test_validator_rejects_tampered_metadata_and_payload",
                                       "no reliance on layer_budget defaults (32, 0.10)": "PackedStoreTests.test_no_reliance_on_llava_layer_budget_defaults + VectorTests.test_llava_defaults_reject_qwen_vectors",
                                       "U packed == unpacked prefix25 bytes/tensors": "PackedStoreTests.test_uniform_packed_is_bitwise_the_unpacked_prefix25_store",
                                       "mask hook realignment / stale hook": "MaskHookTests.*"}}
    ok = ok and result.wasSuccessful() and result.testsRun >= 14
    out["status"] = "PASS" if ok else "FAIL"
    write(gates / "qg1_cpu.json", out)
    # ---- complete existing CPU suite (GLM modules reported separately: pre-existing environment errors)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=".:scripts:tests", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    modules = sorted(p.stem for p in (ROOT / "tests").glob("test_*.py"))
    glm = [m for m in modules if "glm46v" in m]
    rest = [m for m in modules if m not in glm]
    with (gates / "cpu_regression.log").open("x") as log:
        p = subprocess.run([sys.executable, "-m", "unittest"] + rest, cwd=ROOT / "tests", stdout=log, stderr=subprocess.STDOUT, env=dict(env, PYTHONPATH=f"{ROOT}:{ROOT / 'scripts'}:{ROOT / 'tests'}"))
    with (gates / "cpu_regression_glm.log").open("x") as log:
        g = subprocess.run([sys.executable, "-m", "unittest"] + glm, cwd=ROOT / "tests", stdout=log, stderr=subprocess.STDOUT, env=dict(env, PYTHONPATH=f"{ROOT}:{ROOT / 'scripts'}:{ROOT / 'tests'}"))
    text = (gates / "cpu_regression.log").read_text()
    glm_text = (gates / "cpu_regression_glm.log").read_text()
    ran = [line for line in text.splitlines() if line.startswith("Ran ")]
    reg = {"gate": "existing CPU tests", **bind(run), "status": "PASS" if p.returncode == 0 else "FAIL", "returncode": p.returncode, "modules": len(rest),
           "ran": ran[-1] if ran else None, "log_sha256": sha(gates / "cpu_regression.log"), "includes_new_module": "test_qwen_layerwise_packed" in rest,
           "glm_modules_run_separately": glm, "glm_returncode": g.returncode, "glm_errors": [line for line in glm_text.splitlines() if line.startswith(("ERROR:", "FAIL:"))],
           "glm_log_sha256": sha(gates / "cpu_regression_glm.log"),
           "glm_note": "GLM-4.6V tests fail before this change on the installed transformers API (same 3 errors in the baseline taken before any new file existed, "
                       "and in the 2026-11-10 LLaVA run); they import nothing added here; reported, not hidden"}
    write(gates / "cpu_regression.json", reg)
    print(json.dumps({"QG1": out["status"], "cpu_regression": reg["status"], "ran": reg["ran"], "glm_errors": len(reg["glm_errors"])}), flush=True)
    assert out["status"] == "PASS" and reg["status"] == "PASS"


# ========================================================================= GPU gates
def tensor_sha(x):
    import torch
    x = x.detach().cpu().contiguous()
    return hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()


class SdpaTap:
    """Observe (and, for the independent reference only, re-mask) every SDPA call."""

    def __init__(self, override=None, keep=28):
        self.override, self.keep = override, keep
        self.records, self.shapes = [], []

    def __enter__(self):
        import torch.nn.functional as F
        self.F = F
        self.original = F.scaled_dot_product_attention
        tap = self

        def observe(q, k, v, *args, **kwargs):
            index = len(tap.shapes)
            mask = kwargs.get("attn_mask", args[0] if args else None)
            if tap.override is not None:
                new = tap.override(index, q, k, mask)
                if new is not mask:
                    if "attn_mask" in kwargs or not args:
                        kwargs["attn_mask"] = new
                    else:
                        args = (new,) + tuple(args[1:])
                    mask = new
            tap.shapes.append({"q": list(q.shape), "k": list(k.shape), "mask": None if mask is None else list(mask.shape),
                               "is_causal": bool(kwargs.get("is_causal", False)), "enable_gqa": bool(kwargs.get("enable_gqa", False))})
            if index < tap.keep:
                tap.records.append({"q": q.detach().cpu().clone(), "k": k.detach().cpu().clone(), "v": v.detach().cpu().clone(),
                                    "mask": None if mask is None else mask.detach().cpu().clone(),
                                    "meta": {"scale": kwargs.get("scale"), "is_causal": bool(kwargs.get("is_causal", False))}})
            return tap.original(q, k, v, *args, **kwargs)
        F.scaled_dot_product_attention = observe
        return self

    def __exit__(self, *exc):
        self.F.scaled_dot_product_attention = self.original


class IOTap:
    """Independent count of real pread/preadv returns, vision forwards and score calls."""

    def __init__(self, runner):
        self.runner = runner
        self.calls = []
        self.vision = 0
        self.score = 0

    def __enter__(self):
        from mmimpress.qwen25.vision import VisionScoreCapture
        self.V = VisionScoreCapture
        self.pread, self.preadv, self.score_orig = os.pread, os.preadv, VisionScoreCapture._compute_scores
        tap = self

        def pread(fd, count, offset):
            blob = tap.pread(fd, count, offset)
            tap.calls.append({"syscall": "pread", "path": os.readlink(f"/proc/self/fd/{fd}"), "offset": offset, "requested": count, "returned": len(blob)})
            return blob

        def preadv(fd, buffers, offset, *flags):
            n = tap.preadv(fd, buffers, offset, *flags)
            tap.calls.append({"syscall": "preadv", "path": os.readlink(f"/proc/self/fd/{fd}"), "offset": offset,
                              "requested": sum(memoryview(b).nbytes for b in buffers), "returned": n})
            return n

        def score(cap):
            tap.score += 1
            return tap.score_orig(cap)
        os.pread, os.preadv, VisionScoreCapture._compute_scores = pread, preadv, score
        self.handle = self.runner.model.visual.register_forward_pre_hook(lambda *_: setattr(tap, "vision", tap.vision + 1))
        return self

    def __exit__(self, *exc):
        os.pread, os.preadv, self.V._compute_scores = self.pread, self.preadv, self.score_orig
        self.handle.remove()


def stock_positions(runner, full_ids, grid):
    import torch
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    return Qwen2_5_VLModel.get_rope_index(SimpleNamespace(config=runner.model.config), input_ids=full_ids.cpu(), image_grid_thw=grid.cpu(),
                                          attention_mask=torch.ones_like(full_ids.cpu()))


def reference_run(runner, G, cap, logical, suffix, positions, prefix_len, layer_source=None):
    """Independent in-memory hit: canonical capture rows gathered per layer, explicit per-layer
    masks injected at the SDPA call, own greedy loop.  No store, reader, plan or mask hook."""
    import torch
    from transformers import DynamicCache
    device = runner.device
    runner.model.model.rope_deltas = None
    cache = DynamicCache(config=runner.model.config.text_config)
    for li in range(L):
        src = li if layer_source is None else layer_source[li]
        idx = torch.tensor(logical[li], device=device)
        cache.update(cap.layers[src][0].index_select(2, idx), cap.layers[src][1].index_select(2, idx), li)
    t = int(suffix.shape[1])
    masks = [G.visible_mask(logical[li], prefix_len, t)[None, None].to(device) for li in range(L)]
    len0 = len(logical[0])
    hf_layer0 = {}

    def override(index, q, k, mask):
        if index < L:
            if index == 0:
                hf_layer0["mask"] = None if mask is None else mask.detach().cpu().clone()
            return masks[index]
        return mask
    mask2d = torch.ones((1, len0 + t), dtype=torch.long, device=device)
    eos = runner._eos_ids
    with torch.inference_mode(), SdpaTap(override=override) as tap:
        out = runner.model(input_ids=suffix.to(device), attention_mask=mask2d, position_ids=positions[:, :, prefix_len:].to(device),
                           cache_position=torch.arange(len0, len0 + t, device=device), past_key_values=cache, use_cache=True, return_dict=True, logits_to_keep=1)
        logits = out.logits[0, -1].float().cpu().clone()
        generated = [int(out.logits[0, -1].argmax())]
        start = int(positions.max()) + 1
        while len(generated) < runner.max_new_tokens and generated[-1] not in eos:
            mask2d = torch.cat([mask2d, torch.ones((1, 1), dtype=mask2d.dtype, device=device)], dim=-1)
            out = runner.model(input_ids=torch.tensor([[generated[-1]]], device=device), attention_mask=mask2d,
                               position_ids=torch.full((3, 1, 1), start + len(generated) - 1, dtype=torch.long, device=device),
                               cache_position=torch.tensor([int(cache.get_seq_length())], device=device), past_key_values=cache, use_cache=True, return_dict=True, logits_to_keep=1)
            generated.append(int(out.logits[0, -1].argmax()))
    runner.model.model.rope_deltas = None
    return {"first_logits": logits, "generated_token_ids": generated, "tap": tap, "hf_layer0_mask": hf_layer0.get("mask"), "explicit_masks": [m.cpu() for m in masks]}


def records_equal(a, b):
    import torch
    out = []
    for x, y in zip(a, b):
        same = all(torch.equal(x[key], y[key]) for key in ("q", "k", "v")) and ((x["mask"] is None and y["mask"] is None) or
                                                                                (x["mask"] is not None and y["mask"] is not None and torch.equal(x["mask"], y["mask"])))
        out.append(bool(same))
    return out


def masked_probability_zero(G, record, pair, logical, prefix_len):
    """Dense logical reference: every hidden prefix position gets probability exactly 0."""
    import torch
    q = record["q"]
    t = q.shape[2]
    p = len(logical)
    group = q.shape[1] // 4
    ks, vs = record["k"][:, ::group, p:], record["v"][:, ::group, p:]
    ck, cv = (x.cpu() for x in pair)
    dm = G.visible_mask(logical, prefix_len, t, True)
    dense = G.attention(q, torch.cat((ck, ks), 2), torch.cat((cv, vs), 2), dm, torch.float64, record["meta"]["scale"])
    hidden = ~dm[None, None].expand_as(dense["prob"])
    return bool((dense["prob"][hidden] == 0).all()), int(hidden[0, 0].sum())


def guarded_remove(path, run):
    path = Path(path)
    root = (run / "gates" / "gpu").resolve()
    assert path.resolve().is_relative_to(root) and path.name == "stores" and not path.is_symlink()
    files = [p for p in path.rglob("*") if p.is_file() or p.is_symlink()]
    assert all(not p.is_symlink() and p.stat().st_nlink == 1 for p in files)
    removed = [str(p.relative_to(run)) for p in files]
    shutil.rmtree(path)
    return removed


def reject(call):
    try:
        call()
    except (AssertionError, ValueError, RuntimeError, FileNotFoundError, EOFError, KeyError) as exc:
        return {"rejected": True, "reason": repr(exc)[:300]}
    return {"rejected": False}


def tamper_copy(src, dst, mutate):
    from mmimpress.qwen25 import packed_store as PS
    shutil.copytree(src, dst)
    meta = json.loads((dst / PS.META_NAME).read_text())
    mutate(meta)
    meta["bytes_metadata_file"] = 0
    for _ in range(10):
        enc = (json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
        if len(enc) == meta["bytes_metadata_file"]:
            break
        meta["bytes_metadata_file"] = len(enc)
    (dst / PS.META_NAME).write_bytes(enc)
    return dst


def gpu_case(runner, G, sample, case_dir, docs, run):
    import torch
    from mmimpress.qwen25 import layerwise as LW
    from mmimpress.qwen25 import packed_store as PS
    ev = {"image_id": sample["image_id"], "question_id": sample["question_id"], "question": sample["question"]}
    stores = case_dir / "stores"
    stores.mkdir(parents=True)
    image_sha = sample["image_sha256"]
    assert sha(sample["image_path"]) == image_sha
    t1 = runner.run_pixels(sample["image_path"], sample["question"], capture="with_score", image_sha256=image_sha, return_logits=True)
    cap = t1.pop("capture")
    n, prefix_len = cap.visual_count, len(cap.prefix_ids)
    score = cap.scores.float().cpu().tolist()
    rank = sorted(range(n), key=lambda i: (-score[i], i))
    structural = [i for i in range(prefix_len) if not cap.visual_start <= i < cap.visual_start + n]
    s = len(structural)
    ev.update(N=n, S=s, prefix_len=prefix_len, T1_prediction=t1["prediction"])
    runner.persist(cap, stores / "B", "repacked", storage_policy="prefix25")
    for a in Q.ALLOCS:
        LW.persist_packed(runner, cap, stores / a, docs[a])
    session = LW.PackedSession(runner)
    row = 4 * 128 * 2
    k_ind = {a: hamilton(docs[a]["layer_keep_ratios"], n) for a in Q.ALLOCS}
    logical = {a: [sorted(structural + [cap.visual_start + i for i in rank[:k]]) for k in k_ind[a]] for a in Q.ALLOCS}
    forms = [("no_history", sample["question"], ()), ("one_turn_history", FOLLOWUP_QUESTION, ((sample["question"], t1["prediction"]),))]
    grid = torch.tensor(cap.image_grid_thw)
    gate = {g: True for g in ("QG2", "QG3", "QG4", "QG5", "QG6")}

    def check(name, ok, key, detail=None):
        ev.setdefault(name, {})[key] = {"ok": bool(ok), **({"detail": detail} if detail is not None else {})}
        gate[name] = gate[name] and bool(ok)
        return bool(ok)

    try:
        # ---------------------------------------------------------------- QG2: bytes / tensors
        b_dir, u_dir = stores / "B", stores / "U"
        blob = (u_dir / PS.PACKED_NAME).read_bytes()
        u_meta = json.loads((u_dir / PS.META_NAME).read_text())
        b_meta = json.loads((b_dir / "meta.json").read_text())
        struct_bytes = 2 * L * s * row
        check("QG2", blob[:struct_bytes] == (b_dir / "structural_kv.bin").read_bytes(), "structural_block_bitwise")
        off, same = struct_bytes, True
        for li in range(L):
            k = (n + 3) // 4
            same = same and blob[off:off + k * row] == (b_dir / f"layer_{li:03d}/k.bin").read_bytes() and blob[off + k * row:off + 2 * k * row] == (b_dir / f"layer_{li:03d}/v.bin").read_bytes()
            off += 2 * k * row
        check("QG2", same and off == len(blob), "all_28_layer_slices_bitwise")
        b_payload = sum(p.stat().st_size for p in b_dir.rglob("*.bin"))
        check("QG2", len(blob) == b_payload == b_meta["bytes_visual_kv"] + b_meta["bytes_structural_kv"] == u_meta["packed_total_bytes"], "payload_bytes_equal", {"bytes": len(blob)})
        b_store = runner._activate(b_dir, image_sha)
        loaded = b_store.load_prefix(.25, budget_unit="visual_kv")
        u_store = session.activate(u_dir, image_sha, docs["U"])
        u_store.read_all()
        u_layers = u_store.assemble_cpu()
        check("QG2", len(u_layers) == 28 and all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[0].dtype == torch.bfloat16 for a, b in zip(u_layers, loaded.layers)), "assembled_KV_bitwise_vs_load_prefix")
        check("QG2", u_store.layer_logical_indices(0) == list(loaded.logical_indices) == logical["U"][0], "logical_indices_equal")
        del loaded, u_layers

        payload_bytes = {}
        for a in Q.ALLOCS:
            payload_bytes[a] = (stores / a / PS.PACKED_NAME).stat().st_size
        check("QG6", len(set(payload_bytes.values())) == 1 and payload_bytes["U"] == b_payload, "payload_bytes_equal_three_vectors_and_B", payload_bytes)

        # ------------------------------------------------- QG3(a): packed slices vs canonical capture
        for a in ("PYR-b20", "PYR-b2p5"):
            data = (stores / a / PS.PACKED_NAME).read_bytes()
            meta = json.loads((stores / a / PS.META_NAME).read_text())
            off, ok = struct_bytes, meta["layer_keep_counts"] == k_ind[a] and meta["full_importance_permutation"] == rank
            for li, k in enumerate(k_ind[a]):
                idx = torch.tensor([cap.visual_start + i for i in rank[:k]], device=cap.layers[li][0].device)
                for kind in (0, 1):
                    canon = cap.layers[li][kind][0].index_select(1, idx).permute(1, 0, 2).contiguous().cpu()
                    seg = data[off:off + k * row]
                    ok = ok and len(seg) == k * row and hashlib.sha256(seg).digest() == hashlib.sha256(canon.view(torch.uint8).numpy().tobytes()).digest()
                    off += k * row
            check("QG3", ok and off == len(data), f"{a}:a_packed_slices_bitwise_first_k_l_of_independent_rank", {"k_first": k_ind[a][0], "k_last": k_ind[a][-1], "min_k": min(k_ind[a])})

        # ------------------------------------------------- per hit form: production / reference / repeat
        selected = {}
        for form, question, history in forms:
            for a in Q.ALLOCS:
                store_dir = stores / a
                store = session.activate(store_dir, image_sha, docs[a])
                meta = store.meta
                full, suffix, positions, deltas = runner._cache_hit_ids(meta, question, history)
                sp, sd = stock_positions(runner, full, grid)
                t = int(suffix.shape[1])
                tag = f"{a}:{form}"
                session.condition(store_dir, image_sha, docs[a])
                with IOTap(runner) as io, SdpaTap() as prod:
                    hit = LW.run_cache_layerwise(runner, session, store_dir, question, history, layer_budget=docs[a], image_sha256=image_sha, return_logits=True)
                ref = reference_run(runner, G, cap, logical[a], suffix, positions, prefix_len)
                session.condition(store_dir, image_sha, docs[a])
                with SdpaTap() as again_tap:
                    again = LW.run_cache_layerwise(runner, session, store_dir, question, history, layer_budget=docs[a], image_sha256=image_sha, return_logits=True)
                # QG6: I/O
                total = (store_dir / PS.PACKED_NAME).stat().st_size
                check("QG6", len(io.calls) == 1 and io.calls[0]["syscall"] == "preadv" and io.calls[0]["offset"] == 0 and io.calls[0]["returned"] == io.calls[0]["requested"] == total
                      and Path(io.calls[0]["path"]).resolve() == (store_dir / PS.PACKED_NAME).resolve() and hit["pread_calls"] == 1 and hit["read_io"]["bytes"] == total,
                      f"{tag}:one_preadv_zero_pread_whole_file", {"bytes": total})
                check("QG6", io.vision == 0 and io.score == 0 and hit["vision_calls"] == 0 and hit["online_query_score_calls"] == 0, f"{tag}:no_vision_no_score")
                selected.setdefault(a, []).append((hit["selected_importance_prefix_original"], hit["layer_keep_counts"], io.calls[0]["offset"], io.calls[0]["requested"]))
                # QG3(b): SDPA inputs
                eq = records_equal(prod.records, ref["tap"].records)
                eq2 = records_equal(prod.records, again_tap.records)
                lens_ok = all(prod.records[li]["k"].shape[2] == s + k_ind[a][li] + t and list(prod.records[li]["mask"].shape) == [1, 1, t, s + k_ind[a][li] + t] for li in range(L))
                masks_ok = all(torch.equal(prod.records[li]["mask"], ref["explicit_masks"][li]) for li in range(L))
                steps = len(hit["generated_token_ids"])
                decode_ok = len(prod.shapes) == L * steps and all(prod.shapes[f * L + li]["k"][2] == s + k_ind[a][li] + t + f and prod.shapes[f * L + li]["q"][2] == 1
                                                                 and prod.shapes[f * L + li]["mask"] is None for f in range(1, steps) for li in range(L))
                hf0 = ref["hf_layer0_mask"]
                check("QG3", len(prod.records) == 28 and all(eq) and all(eq2), f"{tag}:b_SDPA_QKV_mask_bitwise_28_layers_vs_reference_and_repeat", {"layers_equal": sum(eq)})
                check("QG3", lens_ok and masks_ok, f"{tag}:b_key_count_S_plus_k_l_plus_suffix_and_explicit_mask", {"suffix": t})
                check("QG3", decode_ok, f"{tag}:b_decode_step_plus_one_per_layer", {"forwards": steps})
                check("QG3", hf0 is not None and torch.equal(hf0, ref["explicit_masks"][0]), f"{tag}:b_installed_layer0_mask_is_explicit_mask")
                # QG3(c): logits / ids
                check("QG3", tensor_sha(hit["first_logits"]) == tensor_sha(ref["first_logits"]) == tensor_sha(again["first_logits"])
                      and hit["generated_token_ids"] == ref["generated_token_ids"] == again["generated_token_ids"], f"{tag}:c_logits_bitwise_ids_equal_ssd_memory_repeat",
                      {"logits_sha256": tensor_sha(hit["first_logits"]), "generated": hit["generated_token_ids"]})
                check("QG3", torch.equal(sp, positions) and torch.equal(sd, deltas) and hit["layer_keep_counts"] == k_ind[a] and hit["residual_layer_hooks_after"] == 0
                      and hit["mask_hook_realigned_layers"] == sum(k != k_ind[a][0] for k in k_ind[a]), f"{tag}:stock_positions_counts_hooks_removed")
                # QG4: FP64 semantics, every layer
                fp64_max, legacy_max, legacy_fail, zero_ok, hidden = 0.0, 0.0, 0, True, 0
                try:
                    for li in range(L):
                        result, _ = G.semantic_layer(prod.records[li], cap.layers[li], logical[a][li], prefix_len, repeats=2)
                        fp64_max = max(fp64_max, result["semantic_fp64"]["max_abs"])
                        legacy_max = max(legacy_max, result["legacy_fp32"]["max_abs"])
                        legacy_fail += int(not result["legacy_fp32"]["allclose"])
                        z, h = masked_probability_zero(G, prod.records[li], cap.layers[li], logical[a][li], prefix_len)
                        zero_ok = zero_ok and z and result["semantic_fp64"]["allclose"] and result["semantic_fp64"]["violations"] == 0
                        hidden += h
                    check("QG4", zero_ok, f"{tag}:fp64_dense_vs_compact_1e-12_all_layers_heads_queries",
                          {"fp64_max_abs": fp64_max, "hidden_prefix_positions": hidden, "legacy_fp32_1e-5_diagnostic_max_abs": legacy_max, "legacy_fp32_layers_failing_diagnostic_only": legacy_fail})
                except AssertionError as exc:
                    check("QG4", False, f"{tag}:fp64_dense_vs_compact_1e-12_all_layers_heads_queries", {"error": repr(exc)})
                if a == "U":
                    # QG2: logits / ids / SDPA inputs vs the unpacked B path
                    runner.condition_cache(b_dir, .25, budget_unit="visual_kv", image_sha256=image_sha)
                    with SdpaTap() as b_tap:
                        b_hit = runner.run_cache(b_dir, question, history=history, budget_ratio=.25, budget_unit="visual_kv", image_sha256=image_sha, return_logits=True)
                    check("QG2", tensor_sha(b_hit["first_logits"]) == tensor_sha(hit["first_logits"]) and b_hit["generated_token_ids"] == hit["generated_token_ids"],
                          f"{form}:first_logits_bitwise_and_generated_ids_vs_B", {"generated": hit["generated_token_ids"]})
                    check("QG2", all(records_equal(prod.records, b_tap.records)) and len(b_tap.records) == 28, f"{form}:SDPA_QKV_mask_bitwise_vs_B")
                    check("QG2", hit["gpu_cache_kv_bytes"] == b_hit["gpu_cache_kv_bytes"] and hit["packed_read_bytes"] == b_hit["read_io"]["bytes"] and b_hit["pread_calls"] == 57,
                          f"{form}:same_bytes_57_pread_vs_1_preadv")
                    del b_tap, b_hit
                # QG5: negative fixtures (attention-level) on the first form of the two pyramid vectors
                if a != "U" and form == "no_history":
                    neg = {}
                    j = 13
                    for name, delta in (("layer_k_plus_1", 1), ("layer_k_minus_1", -1)):
                        bad = [list(x) for x in logical[a]]
                        bad[j] = sorted(structural + [cap.visual_start + i for i in rank[:k_ind[a][j] + delta]])
                        r = reference_run(runner, G, cap, bad, suffix, positions, prefix_len)
                        neg[name] = {"layer": j, "next_layer_query_changed": not torch.equal(r["tap"].records[j + 1]["q"], prod.records[j + 1]["q"]),
                                     "first_logits_changed": tensor_sha(r["first_logits"]) != tensor_sha(hit["first_logits"])}
                        neg[name]["detected"] = neg[name]["next_layer_query_changed"] and neg[name]["first_logits_changed"]
                    r = reference_run(runner, G, cap, [logical[a][6] if li == 5 else logical[a][5] if li == 6 else logical[a][li] for li in range(L)], suffix, positions, prefix_len,
                                      layer_source=[6 if li == 5 else 5 if li == 6 else li for li in range(L)])
                    neg["two_layers_rows_swapped"] = {"layers": [5, 6], "first_logits_changed": tensor_sha(r["first_logits"]) != tensor_sha(hit["first_logits"])}
                    neg["two_layers_rows_swapped"]["detected"] = neg["two_layers_rows_swapped"]["first_logits_changed"]
                    # one hidden row exposed in the dense logical reference of layer j (FP64)
                    rec = prod.records[j]
                    p = len(logical[a][j])
                    group = rec["q"].shape[1] // 4
                    ck, cv = (x.cpu() for x in cap.layers[j])
                    dk = torch.cat((ck, rec["k"][:, ::group, p:]), 2)
                    dv = torch.cat((cv, rec["v"][:, ::group, p:]), 2)
                    dm = G.visible_mask(logical[a][j], prefix_len, t, True)
                    exposed = dm.clone()
                    hidden_row = cap.visual_start + rank[k_ind[a][j]]
                    exposed[:, hidden_row] = True
                    good_out = G.attention(rec["q"], dk, dv, dm, torch.float64, rec["meta"]["scale"])["output"]
                    bad_out = G.attention(rec["q"], dk, dv, exposed, torch.float64, rec["meta"]["scale"])["output"]
                    st = G.stats(bad_out, good_out, 1e-12, 1e-12)
                    neg["one_hidden_row_exposed"] = {"layer": j, "original_index": hidden_row, "fp64_max_abs": st["max_abs"], "detected": not st["allclose"]}
                    bad = [[x for x in row_ if x != structural[0]] for row_ in logical[a]]
                    r = reference_run(runner, G, cap, bad, suffix, positions, prefix_len)
                    neg["structural_row_missing"] = {"first_logits_changed": tensor_sha(r["first_logits"]) != tensor_sha(hit["first_logits"])}
                    neg["structural_row_missing"]["detected"] = neg["structural_row_missing"]["first_logits_changed"]
                    # stale hook from an earlier request / missing hook
                    other = "PYR-b2p5" if a == "PYR-b20" else "PYR-b20"
                    stale = LW.LayerMaskHooks(runner.model, [s + k for k in k_ind[other]]).__enter__()
                    guard = reject(lambda: LW.assert_no_layer_hooks(runner.model))
                    stack = reject(lambda: LW.run_cache_layerwise(runner, session, store_dir, question, history, layer_budget=docs[a], image_sha256=image_sha))
                    plan = store.plan()
                    store.read_all()
                    stale_cache = LW.assemble_cache(runner, store, plan)

                    def stale_forward():
                        runner.model.model.rope_deltas = None
                        with torch.inference_mode():
                            runner.model(input_ids=suffix.to(runner.device), attention_mask=torch.ones((1, plan.lengths[0] + t), dtype=torch.long, device=runner.device),
                                         position_ids=positions[:, :, prefix_len:].to(runner.device),
                                         cache_position=torch.arange(plan.lengths[0], plan.lengths[0] + t, device=runner.device),
                                         past_key_values=stale_cache, use_cache=True, return_dict=True, logits_to_keep=1)
                    foreign = reject(stale_forward)
                    stale.remove()
                    runner.model.model.rope_deltas = None
                    LW.assert_no_layer_hooks(runner.model)
                    neg["stale_hook_from_previous_request"] = {"guard_before_next_request": guard, "new_request_refuses_to_stack": stack, "stale_lengths_on_other_cache": foreign,
                                                              "detected": guard["rejected"] and stack["rejected"] and foreign["rejected"]}
                    store.read_all()
                    nohook_cache = LW.assemble_cache(runner, store, plan)
                    try:
                        nh = LW.forward_layerwise(runner, nohook_cache, plan.lengths, suffix, positions, prefix_len, install_hooks=False, return_logits=True)
                        neg["mask_hook_missing"] = {"first_logits_changed": tensor_sha(nh["first_logits"]) != tensor_sha(hit["first_logits"]),
                                                    "max_abs_logit_diff": float((nh["first_logits"] - hit["first_logits"]).abs().max())}
                        neg["mask_hook_missing"]["detected"] = neg["mask_hook_missing"]["first_logits_changed"]
                    except Exception as exc:
                        neg["mask_hook_missing"] = {"raised": repr(exc)[:200], "detected": True}
                    runner._clear_request_state()
                    # the production path is unchanged after the negative fixtures
                    session.condition(store_dir, image_sha, docs[a])
                    after = LW.run_cache_layerwise(runner, session, store_dir, question, history, layer_budget=docs[a], image_sha256=image_sha, return_logits=True)
                    neg["production_unchanged_after_fixtures"] = {"detected": tensor_sha(after["first_logits"]) == tensor_sha(hit["first_logits"]) and after["generated_token_ids"] == hit["generated_token_ids"]}
                    # validator-level fixtures
                    fx = case_dir / "stores" / f"negative_{a}"
                    fx.mkdir()

                    def bump(x):
                        def f(m):
                            m["layer_keep_counts"][j] += x
                        return f
                    files = {
                        "meta_layer_k_plus_1": lambda: PS.open_qwen_packed_store(tamper_copy(store_dir, fx / "kplus", bump(1)), expected_budget=docs[a], pin=False),
                        "meta_layer_k_minus_1": lambda: PS.open_qwen_packed_store(tamper_copy(store_dir, fx / "kminus", bump(-1)), expected_budget=docs[a], pin=False),
                        "meta_two_layers_swapped": lambda: PS.open_qwen_packed_store(tamper_copy(store_dir, fx / "swap", lambda m: m["packed_layout"].__setitem__(slice(5, 7), [m["packed_layout"][6], m["packed_layout"][5]])), expected_budget=docs[a], pin=False),
                        "meta_structural_row_missing": lambda: PS.open_qwen_packed_store(tamper_copy(store_dir, fx / "struct", lambda m: (m.__setitem__("structural_indices", m["structural_indices"][1:]), m.__setitem__("structural_count", m["structural_count"] - 1))), expected_budget=docs[a], pin=False),
                        "meta_r_min_0p10": lambda: PS.open_qwen_packed_store(tamper_copy(store_dir, fx / "rmin", lambda m: m.__setitem__("layer_budget_r_min", 0.10)), pin=False),
                        "meta_num_layers_32": lambda: PS.open_qwen_packed_store(tamper_copy(store_dir, fx / "layers", lambda m: m.__setitem__("num_layers", 32)), pin=False),
                        "meta_other_vector_sha": lambda: PS.open_qwen_packed_store(tamper_copy(store_dir, fx / "sha", lambda m: m.__setitem__("layer_budget_sha256", Q.RATIO_SHA[other])), pin=False),
                        "store_opened_with_other_vector": lambda: PS.open_qwen_packed_store(store_dir, expected_budget=docs[other], pin=False),
                        "session_request_with_other_vector": lambda: session.activate(store_dir, image_sha, docs[other]),
                        "foreign_image_identity": lambda: LW.PackedSession(runner).activate(store_dir, "0" * 64, docs[a]),
                    }
                    for kind in ("truncated_file", "payload_rows_of_two_layers_swapped", "payload_bit_flip"):
                        dst = fx / kind
                        shutil.copytree(store_dir, dst)
                        path = dst / PS.PACKED_NAME
                        lay = meta["packed_layout"]
                        with path.open("r+b") as f:
                            if kind == "truncated_file":
                                f.truncate(path.stat().st_size - 2)
                            elif kind == "payload_bit_flip":
                                f.seek(lay[9]["k_offset"]); b0 = f.read(1); f.seek(lay[9]["k_offset"]); f.write(bytes([b0[0] ^ 1]))
                            else:
                                size = min(lay[26]["rows"], lay[27]["rows"]) * meta["row_bytes"]
                                f.seek(lay[26]["k_offset"]); xa = f.read(size)
                                f.seek(lay[27]["k_offset"]); xb = f.read(size)
                                f.seek(lay[26]["k_offset"]); f.write(xb)
                                f.seek(lay[27]["k_offset"]); f.write(xa)
                        files[kind] = (lambda d: (lambda: PS.open_qwen_packed_store(d, expected_budget=docs[a], pin=False)))(dst)
                    for name, call in files.items():
                        neg[name] = reject(call)
                        neg[name]["detected"] = neg[name]["rejected"]
                    check("QG5", all(x["detected"] for x in neg.values()) and len(neg) == 21, f"{a}:negative_fixtures", neg)
                del prod, ref, again_tap, hit, again
        check("QG6", all(all(x == v[0] for x in v) and len(v) == 2 for v in selected.values()), "question_change_keeps_read_range_and_layer_selection")
        ev["gates"] = gate
        ev["status"] = "PASS" if all(gate.values()) else "FAIL"
    finally:
        for layer in LW.decoder_layers(runner.model):
            layer.self_attn._forward_pre_hooks.clear()
        session.close()
        runner.close()
        del cap
        ev["removed_gate_store_files"] = len(guarded_remove(stores, run))
    return ev


def stage_gpu(run):
    import torch
    config = read(run / "config.json")
    assert Q.source_hashes() == config["source_sha256"], "FROZEN_SOURCE_CONFIG_MISMATCH"
    assert read(run / "gates" / "qg1_cpu.json")["status"] == "PASS" and read(run / "gates" / "cpu_regression.json")["status"] == "PASS"
    assert read(run / "gates" / "qg1_cpu.json")["config_sha256"] == sha(run / "config.json")
    assert not (run / "gates" / "qg_gpu.json").exists(), "GPU gate already recorded; failed attempts are preserved, re-freeze before a new attempt"
    docs = Q.load_vectors(run)
    M = load_script("_qpyr_base110", "110_eval_qwen_mt_gqa_B.py")
    G = load_script("_qpyr_semantic116", "116_qwen_prefixstore_semantic.py")
    E = load_script("_qpyr_eval", "169_eval_qwen_layer_budget_pyr.py")
    samples = read(run / "correctness_manifest.json")["samples"]
    assert len(samples) == 10
    E.seed()
    summary = {"gate": "QG2-QG6", **bind(run), "status": "RUNNING", "samples": [], "fixture": "10 frozen core pairs (parent correctness_manifest.json); two hit prompts each",
               "environment": G.numerical_environment()}
    runner = None
    base = run / "gates" / "gpu"
    base.mkdir()
    try:
        runner = M.load_runner()
        summary["runtime"] = runner.runtime_fingerprint()
        for i, sample in enumerate(samples):
            case = base / f"{i:02d}_{sample['image_id']}"
            case.mkdir()
            t0 = time.time()
            try:
                ev = gpu_case(runner, G, sample, case, docs, run)
            except BaseException as exc:
                write(case / "evidence.json", {"status": "FAIL", "error": repr(exc), "traceback": traceback.format_exc()})
                summary["samples"].append({"index": i, "image_id": sample["image_id"], "status": "FAIL", "error": repr(exc)})
                raise
            ev["elapsed_s"] = time.time() - t0
            write(case / "evidence.json", ev)
            summary["samples"].append({"index": i, "image_id": sample["image_id"], "N": ev["N"], "status": ev["status"], "gates": ev["gates"], "evidence_sha256": sha(case / "evidence.json")})
            print(json.dumps({"core": i + 1, "of": len(samples), "image": sample["image_id"], "N": ev["N"], **ev["gates"]}), flush=True)
            torch.cuda.empty_cache()
        per_gate = {g: all(x["gates"][g] for x in summary["samples"]) for g in ("QG2", "QG3", "QG4", "QG5", "QG6")}
        summary.update(per_gate=per_gate, status="PASS" if all(per_gate.values()) and len(summary["samples"]) == 10 else "FAIL")
    except BaseException as exc:
        summary.update(status="FAIL", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if runner:
            runner.close()
        write(run / "gates" / "qg_gpu.json", summary)
    print(json.dumps({"gpu_gates": summary["status"], **summary.get("per_gate", {})}), flush=True)
    assert summary["status"] == "PASS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--stage", choices=["cpu", "gpu"], required=True)
    a = ap.parse_args()
    run = a.run.resolve()
    if a.stage == "cpu":
        stage_cpu(run)
    else:
        stage_gpu(run)


if __name__ == "__main__":
    main()

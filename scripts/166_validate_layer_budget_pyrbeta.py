#!/usr/bin/env python3
"""Gates of the PyramidKV beta=20 MT-GQA HOLD add-on run
(contract docs/layer_budget_kv25_pyrbeta_contract.md, section 4).

Stages
  --stage g3     : Sigma k_l == 32*ceil(0.25 N) on the 398 HOLD image N list for U, PYR-beta and
                   PYR-beta-raw (document clip range), cross-check against the v2 gate / v2 HOLD stores
  --stage grmin  : G-rmin (CPU) - the document's r_min = 0.0 reaches the store meta, the packed and
                   unpacked writers, every validator and both selectors; no path relies on the module
                   default 0.10; a meta whose r_min is changed to 0.10 fails closed everywhere
  --stage cpu    : full existing CPU suite (v1 stage_cpu)
  --stage gpu    : G6/G7 on the five fixed GPU cases of v2 161 --stage gpu, for PYR-beta and PYR-beta-raw
  --stage gdet --phase smoke|main : U_packed predictions vs the v2 HOLD rows, token for token
"""
from __future__ import annotations

import argparse
import ast
import copy
import gc
import importlib.util
import json
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

from mmimpress.layer_budget import (keep_bounds, layer_keep_counts, load_budget_document,
                                    total_row_budget, validate_keep_counts)
from mmimpress.prefix25 import sha

V2_RUN = ROOT / "runs/layer_budget_kv25_v2_20261106T073502Z"
V2_RES = ROOT / "results/layer_budget_kv25_v2_20261106T073502Z"
VECTORS = ("U", "PYR-beta", "PYR-beta-raw")
GATE_VECTORS = ("PYR-beta", "PYR-beta-raw")
PK = "ours_prefixstore_kv25_"
SEED = 1234
ATOL = RTOL = 1e-4


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _safe(x):
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


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_safe(obj), f, indent=1, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def docs_of(run, names=VECTORS):
    return {a: load_budget_document(run / "vectors" / f"layer_budget_llava_{a}.json") for a in names}


def hamilton(ratios, n, r_max=0.6):
    """Independent Hamilton apportionment: no library code, no lower bound."""
    import math
    from fractions import Fraction
    L = len(ratios)
    budget = L * (-(-n // 4))
    cap = min(n, math.ceil(Fraction(repr(float(r_max))) * n))
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


def dcounts(doc, n):
    return layer_keep_counts(doc["layer_keep_ratios"], n, doc["num_layers"], doc["total_ratio"], doc["r_min"], doc["r_max"])


# -------------------------------------------------------------------- g3
def stage_g3(run):
    docs = docs_of(run)
    v2docs = {a: load_budget_document(V2_RUN / "phase2" / f"layer_budget_llava_{a}.json") for a in ("U", "PYR-beta")}
    v2gate = json.loads((V2_RES / "gate_g3_budget_invariance.json").read_text())
    rows = json.loads((run / "phase0" / "n_distribution_hold.json").read_text())["hold"]["per_image"]
    assert len(rows) == 398
    out = {"status": "PASS", "vectors": {a: d["layer_budget_sha256"] for a, d in docs.items()}, "clip_range": {a: [d["r_min"], d["r_max"]] for a, d in docs.items()},
           "images": len(rows), "checks": 0, "v2_cross_check": {}}
    v2_store_checked = 0
    for row in rows:
        n = int(row["N"])
        budget = total_row_budget(n)
        assert budget == 32 * ((n + 3) // 4) == row["B"]
        for a, d in docs.items():
            k = dcounts(d, n)
            lo, hi = keep_bounds(n, d["r_min"], d["r_max"])
            assert sum(k) == budget, (row["image_id"], a)
            assert all(lo <= x <= hi for x in k) and k == hamilton(d["layer_keep_ratios"], n, d["r_max"]), (row["image_id"], a)
            validate_keep_counts(k, d["layer_keep_ratios"], n, 32, 0.25, d["r_min"], d["r_max"])
            if a == "U":
                assert k == [(n + 3) // 4] * 32
            if a == "PYR-beta-raw":
                assert min(k) >= 1
            if a in v2docs:      # same k_l as the v2 documents with the v2 (default) clip range
                assert k == layer_keep_counts(v2docs[a]["layer_keep_ratios"], n), (row["image_id"], a)
            out["checks"] += 1
        # U against the real v2 HOLD packed store metadata of the same image
        folder = V2_RUN / "hold" / "main" / "images" / row["image_id"]
        commit = json.loads((folder / "COMMITTED.json").read_text())
        meta = json.loads((folder / commit["attempt_id"] / "packed_U_meta.json").read_text())
        assert meta["n_spatial"] == n and meta["layer_keep_counts"] == dcounts(docs["U"], n) and meta["layer_budget_sha256"] == docs["U"]["layer_budget_sha256"]
        assert meta["stored_content_count_total"] == budget
        v2_store_checked += 1
    for a in ("U", "PYR-beta"):
        out["v2_cross_check"][a] = {"v2_gate_status": v2gate["status"], "v2_gate_hold": v2gate["splits"]["hold"],
                                    "ratio_sha_equal_v2_gate": v2gate["vectors"][a] == docs[a]["layer_budget_sha256"],
                                    "k_l_equal_to_v2_document_recomputation_all_images": True}
        assert out["v2_cross_check"][a]["ratio_sha_equal_v2_gate"] and v2gate["status"] == "PASS" and v2gate["splits"]["hold"]["sum_equals_B"]
    out["v2_cross_check"]["U"]["k_l_and_N_equal_to_v2_HOLD_packed_U_store_meta_images"] = v2_store_checked
    out["sum_equals_B"] = True
    out["bounds_document_clip_range"] = True
    out["independent_hamilton_equal"] = True
    write(run / "gates" / "g3_budget_invariance.json", out)
    print("G3 PASS", out["checks"], "vector-image checks;", v2_store_checked, "v2 HOLD U stores cross-checked", flush=True)


# ----------------------------------------------------------------- grmin
def _fixture32(root, n, layers=32, heads=2, hd=2, vstart=3):
    """Real FP16 K/V tensors, 32 layers, image-only VisionZip order, layout artifact (CPU)."""
    from mmimpress.cvpr25 import permutation_sha256, visionzip_repack_order
    vn = n + 2
    newline = [0, vn - 1]
    real = [i for i in range(vn) if i not in newline]
    scores = torch.full((vn,), float("inf"), dtype=torch.float32)
    for i in real:
        scores[i] = float((i * 7) % 19) / 19.0
    perm = visionzip_repack_order(scores, newline)
    assert perm == sorted(real, key=lambda i: (-float(scores[i]), i)) + newline
    inverse = [0] * vn
    for stored, original in enumerate(perm):
        inverse[original] = stored
    g = torch.Generator().manual_seed(SEED)
    tensors = [(torch.randn((1, heads, vstart + vn, hd), generator=g).to(torch.float16),
                torch.randn((1, heads, vstart + vn, hd), generator=g).to(torch.float16)) for _ in range(layers)]
    flags = {"physical_layout": "visionzip_image_only", "layout_method": "visionzip_image_only", "layout_source": "turn1_normal_inference_piggyback",
             "visual_kv_source": "turn1_captured_past_key_values", "layout_uses_dataset_question": False, "llm_used_for_layout_scoring": False,
             "calibration_questions": 0, "global_order_all_layers": True, "separator_tail": True, "separator_policy": "stable_tail_plus_sidecar",
             "permutation_sha256": permutation_sha256(perm), "inverse_permutation_sha256": permutation_sha256(inverse)}
    root.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": 2, "token_score_original": scores, "newline_original": torch.tensor(newline, dtype=torch.int32),
                "stored_to_original": torch.tensor(perm, dtype=torch.int32), "original_to_stored": torch.tensor(inverse, dtype=torch.int32),
                "layout_uses_dataset_question": False, "llm_used_for_layout_scoring": False, "calibration_questions": 0}, root / "visionzip_layout.pt")
    return {"tensors": tensors, "vstart": vstart, "vn": vn, "newline": newline, "perm": perm, "flags": flags, "prefix_input_ids": list(range(vstart + vn))}


def _expect_fail(label, fn, results, exc=(ValueError, AssertionError)):
    try:
        fn()
    except exc as e:
        results[label] = {"fail_closed": True, "error": type(e).__name__ + ": " + str(e)[:160]}
        return
    results[label] = {"fail_closed": False}
    raise AssertionError(f"G-rmin: {label} did not fail closed")


def _default_reliant_calls(path, names=("layer_keep_counts", "validate_keep_counts", "keep_bounds", "validate_ratios", "clip_redistribute")):
    """AST scan: calls of the bound-taking layer_budget functions that leave r_min to the module default."""
    need = {"layer_keep_counts": 5, "validate_keep_counts": 6, "keep_bounds": 2, "validate_ratios": 4, "clip_redistribute": 4}
    found = []
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(node, ast.Call):
            fn = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else None
            if fn in names:
                explicit = len(node.args) >= need[fn] or any(kw.arg == "r_min" for kw in node.keywords) or any(isinstance(a, ast.Starred) for a in node.args) or any(kw.arg is None for kw in node.keywords)
                found.append({"file": str(Path(path).relative_to(ROOT)), "line": node.lineno, "call": fn, "r_min_explicit": bool(explicit)})
    return found


def stage_grmin(run):
    from types import SimpleNamespace
    from unittest.mock import patch
    from mmimpress.packed_serve import PackedImageContext, PackedPrefixSelector, PackedReadTrace
    from mmimpress.packed_store import PACKED_NAME, validate_packed_meta, write_packed_store
    from mmimpress.prefix25 import seal_integrity, validate_llava_prefix_layerwise_meta
    from mmimpress.serve import BIAS, CVPR25ChunkSelector, ImageContext
    from mmimpress.store import write_image_store
    import mmimpress.layer_budget as LB
    H = load("pyrbeta_hold", "164_eval_layer_budget_pyrbeta_hold.py")
    docs = docs_of(run)
    raw = docs["PYR-beta-raw"]
    assert raw["r_min"] == 0.0 and raw["r_max"] == 0.6 and LB.R_MIN == 0.10 and LB.R_MAX == 0.60
    out = {"status": "RUNNING", "global_constants": {"R_MIN": LB.R_MIN, "R_MAX": LB.R_MAX, "unchanged": True},
           "raw_document": {"r_min": raw["r_min"], "r_max": raw["r_max"], "layer_budget_sha256": raw["layer_budget_sha256"]}, "cases": [], "fail_closed": {}}
    base = run / "gates" / "grmin_fixture"
    if base.exists():
        shutil.rmtree(base)
    runner = SimpleNamespace(model=SimpleNamespace(device=torch.device("cpu")))
    try:
        for n in (1344, 2208, 2880):                     # HOLD min / median / max N
            root = base / f"N{n}"
            fx = _fixture32(root / "a", n)
            args = (fx["tensors"], fx["vstart"], fx["vn"], fx["prefix_input_ids"], fx["newline"])
            common = dict(extra=fx["flags"], stored_to_original=fx["perm"], separator_sidecar=True, probe_heads=0)
            lw = write_image_store(root / "lw", *args, storage_policy="prefix25_layerwise", layer_budget=raw, **common)
            pk = write_packed_store(root / "p", *args, storage_policy="prefix25_packed", layer_budget=raw, **common)
            for name in ("lw", "p"):
                (root / name / "visionzip_layout.pt").write_bytes((root / "a" / "visionzip_layout.pt").read_bytes())
                seal_integrity(root / name)
            want = hamilton(raw["layer_keep_ratios"], n, raw["r_max"])
            assert min(want) >= 1 and sum(want) == 32 * ((n + 3) // 4)
            # store meta carries the document's clip range and the independent counts
            for meta in (lw, pk):
                assert meta["layer_budget_r_min"] == 0.0 and meta["layer_budget_r_max"] == 0.6
                assert meta["layer_keep_counts"] == want == meta["payload_rows_per_layer"] and meta["layer_keep_ratios"] == raw["layer_keep_ratios"]
            # validators (all must pass with the meta's r_min = 0.0)
            validate_llava_prefix_layerwise_meta(lw)
            validate_packed_meta(pk)
            ctx_u = ImageContext(root / "lw", "cpu", drop_cache=False, require_v_hidden=False)
            ctx_u.validate_visual_kv_layout()
            ctx_p = PackedImageContext(root / "p", "cpu", pin=False)
            ctx_p.validate_packed_layout()
            assert H.doc_counts(raw, n) == want and H.independent_counts(raw["layer_keep_ratios"], n, raw["r_max"]) == want
            # selectors: visible content rows per layer == k_l exactly (incl. the last layer), packed == unpacked cache and masks
            vstart, vn = fx["vstart"], fx["vn"]
            cache_u = ctx_u.cache.new_request()
            sel_u = CVPR25ChunkSelector(runner, ctx_u, None, .25, "prefix", sep_policy="sidecar", budget_unit="visual_kv")
            with patch("torch.cuda.synchronize"):
                sel_u.prepare(None)
            bias_u = [BIAS[li].clone() for li in range(32)]
            BIAS.clear()
            cache_p = ctx_p.cache.new_request()
            sel_p = PackedPrefixSelector(runner, ctx_p, .25)
            with patch("torch.cuda.synchronize"), PackedReadTrace(root / "p") as trace:
                sel_p.prepare()
            s = trace.summary()
            assert s["preads"] == 1 and s["preadv_calls"] == 1 and s["pread_calls"] == 0 and s["bytes"] == pk["packed_total_bytes"] == (root / "p" / PACKED_NAME).stat().st_size
            visible = []
            for li in range(32):
                b = BIAS[li][0, 0, 0]
                vis = int((b[vstart:vstart + n] == 0).sum())
                assert vis == want[li] and bool((b[vstart:vstart + want[li]] == 0).all()) and bool((b[vstart + n:vstart + vn] == 0).all())
                assert torch.equal(BIAS[li], bias_u[li])
                for kind in ("keys", "values"):
                    assert torch.equal(getattr(cache_p.layers[li], kind), getattr(cache_u.layers[li], kind))
                visible.append(vis)
            assert sel_p.stats()["k_per_layer"] == want == sel_u.stats()["k_per_layer"]
            BIAS.clear()
            out["cases"].append({"N": n, "k_per_layer": want, "k_last_layer": want[-1], "min_k": min(want), "visible_rows_per_layer_equal_k": visible == want,
                                 "validators_pass": ["validate_llava_prefix_layerwise_meta", "validate_packed_meta", "ImageContext.__init__ + validate_visual_kv_layout",
                                                     "PackedImageContext.__init__ + validate_packed_layout", "CVPR25ChunkSelector (layerwise)", "PackedPrefixSelector",
                                                     "164 doc_counts / independent_counts"],
                                 "packed_equals_unpacked_cache_and_masks": True, "single_preadv": True, "packed_total_bytes": pk["packed_total_bytes"]})
            if n == 2208:
                fc = out["fail_closed"]
                # (1) meta dictionaries with r_min changed to 0.10 and nothing else
                _expect_fail("validate_llava_prefix_layerwise_meta(meta r_min=0.10)", lambda: validate_llava_prefix_layerwise_meta(dict(lw, layer_budget_r_min=0.10)), fc)
                _expect_fail("validate_packed_meta(meta r_min=0.10)", lambda: validate_packed_meta(dict(pk, layer_budget_r_min=0.10)), fc)
                # (2) the same stores on disk with meta.json r_min = 0.10 and a re-sealed integrity envelope (so only r_min is wrong)
                for name, opener in (("lw", lambda p: ImageContext(p, "cpu", drop_cache=False, require_v_hidden=False)), ("p", lambda p: PackedImageContext(p, "cpu", pin=False))):
                    bad = root / f"{name}_rmin010"
                    shutil.copytree(root / name, bad)
                    m = json.loads((bad / "meta.json").read_text())
                    m["layer_budget_r_min"] = 0.10
                    (bad / "meta.json").write_text(json.dumps(m, indent=1))
                    (bad / "integrity.json").unlink()
                    seal_integrity(bad)
                    _expect_fail(f"activation of {name} store with meta.json r_min=0.10 (integrity re-sealed)", lambda b=bad, o=opener: o(b), fc)
                # (3) selectors on an already activated context whose in-memory meta is changed
                ctx_p.meta = dict(ctx_p.meta, layer_budget_r_min=0.10)
                _expect_fail("PackedPrefixSelector(meta r_min=0.10)", lambda: PackedPrefixSelector(runner, ctx_p, .25), fc)
                ctx_u.meta = dict(ctx_u.meta, layer_budget_r_min=0.10)
                _expect_fail("CVPR25ChunkSelector(meta r_min=0.10)", lambda: CVPR25ChunkSelector(runner, ctx_u, None, .25, "prefix", sep_policy="sidecar", budget_unit="visual_kv"), fc)
                # (4) writers and document loader with r_min = 0.10
                bad_doc = dict(raw, r_min=0.10)
                _expect_fail("write_packed_store(document r_min=0.10)", lambda: write_packed_store(root / "w1", *args, storage_policy="prefix25_packed", layer_budget=bad_doc, **common), fc)
                _expect_fail("write_image_store(document r_min=0.10)", lambda: write_image_store(root / "w2", *args, storage_policy="prefix25_layerwise", layer_budget=bad_doc, **common), fc)
                tmp = root / "doc_rmin010.json"
                tmp.write_text(json.dumps(bad_doc))
                _expect_fail("load_budget_document(document r_min=0.10)", lambda: load_budget_document(tmp), fc)
                # (5) the default-reliant call pattern of the 156/162 audits rejects the raw vector (so 164 must not use it)
                _expect_fail("layer_keep_counts(raw ratios, N) with module defaults [156/162 audit pattern]", lambda: layer_keep_counts(raw["layer_keep_ratios"], n), fc)
                _expect_fail("164 doc_counts(document r_min=0.10)", lambda: H.doc_counts(bad_doc, n), fc)
            ctx_p.close(); ctx_u.close()
            del fx, cache_u, cache_p
            gc.collect()
        # (6) static scan for calls that leave r_min to the module default
        scan_serving = []
        for f in ("mmimpress/layer_budget.py", "mmimpress/store.py", "mmimpress/packed_store.py", "mmimpress/packed_serve.py", "mmimpress/serve.py", "mmimpress/prefix25.py", "mmimpress/piggyback.py",
                  "scripts/packed_support_v2.py", "scripts/main_b_support.py", "scripts/164_eval_layer_budget_pyrbeta_hold.py", "scripts/165_report_layer_budget_pyrbeta.py",
                  "scripts/166_validate_layer_budget_pyrbeta.py"):
            scan_serving += _default_reliant_calls(ROOT / f)
        implicit = [c for c in scan_serving if not c["r_min_explicit"]]
        # scripts 166 itself: the only default-reliant calls allowed are the deliberate negative test and the v2-document cross-check in G3
        allowed = [c for c in implicit if c["file"] == "scripts/166_validate_layer_budget_pyrbeta.py"]
        offending = [c for c in implicit if c not in allowed]
        reference_only = _default_reliant_calls(ROOT / "scripts/162_eval_layer_budget_v2_hold.py") + _default_reliant_calls(ROOT / "scripts/156_eval_layer_budget_hold.py") + _default_reliant_calls(ROOT / "scripts/154_validate_layer_budget_kv25.py")
        out["static_scan"] = {"files": sorted({c["file"] for c in scan_serving}), "calls": scan_serving, "default_reliant_in_store_serving_or_new_hold_path": offending,
                              "deliberate_default_calls_in_gate_script": allowed,
                              "not_used_by_this_run_but_default_reliant": [c for c in reference_only if not c["r_min_explicit"]],
                              "note": "156/162/154 audits call layer_keep_counts with module defaults; they are not executed for this run (164 is a copy that uses doc_counts). "
                                      "154.persist_layerwise / gpu_case helpers reused by the GPU gate do not call the bound-taking functions with defaults on the raw path."}
        assert not offending, offending
        out["amendment_needed"] = False
        out["status"] = "PASS"
    except Exception:
        out["status"] = "FAIL"
        out["error"] = traceback.format_exc()
        raise
    finally:
        BIAS.clear()
        if base.exists():
            shutil.rmtree(base)                         # run-local CPU fixture only
        out["fixture_deleted"] = not base.exists()
        write(run / "gates" / "g_rmin_floor_propagation.json", out)
    print("G-rmin PASS:", len(out["cases"]), "N cases;", len(out["fail_closed"]), "fail-closed checks; default-reliant calls in the store/serving/new-HOLD path:", len(offending), flush=True)


# ------------------------------------------------------------------- cpu
def stage_cpu(run):
    G = load("layer_budget_gates_v1", "154_validate_layer_budget_kv25.py")
    G.stage_cpu(run)


# ------------------------------------------------------------------- gpu
def packed_traced(exp, G, S, ctx, path, prompt, layers, ranking, counts, directcopy=False):
    """Serve a packed hit through the public path; before the forward, compare the restored cache bits and the
    installed per-layer masks with an independent reconstruction from the Turn-1 capture."""
    server = exp.server
    meta = ctx.meta
    visible = G.independent_visible(meta, counts)
    vstart, n, p = int(meta["v_token_start"]), int(meta["n_spatial"]), int(meta["prefix_len"])
    original = server._decode
    box = {}

    def traced(cache, suffix, prefix_len):
        from mmimpress.serve import BIAS
        assert len(BIAS) == len(layers)
        vis_rows = []
        for li, (sk, sv) in enumerate(layers):
            k = counts[li]
            b = BIAS[li]
            assert tuple(b.shape) == (1, 1, 1, p)
            allowed = b[0, 0, 0].detach().cpu() == 0
            assert torch.equal(allowed, visible[li]), f"layer {li} BIAS"
            vis_rows.append(int(allowed[vstart:vstart + n].sum()))
            for name, src in (("keys", sk), ("values", sv)):
                actual = getattr(cache.layers[li], name)
                assert torch.equal(actual[0, :, :vstart], G.fp16rt(src[0, :, :vstart], actual.device))
                if k:
                    ids = torch.as_tensor(ranking[:k], device=src.device)
                    assert torch.equal(actual[0, :, vstart:vstart + k], G.fp16rt(src[0, :, vstart + ids], actual.device)), f"layer {li} {name} bits"
                if k < n:
                    tail = actual[0, :, vstart + k:vstart + n]
                    assert torch.equal(tail, torch.zeros_like(tail)), f"layer {li} rows beyond k_l must be unloaded"
                ids = torch.as_tensor([int(x) for x in meta["newline_idx"]], device=src.device)
                assert torch.equal(actual[0, :, vstart + n:p], G.fp16rt(src[0, :, vstart + ids], actual.device))
        box["visible_content_rows_per_layer"] = vis_rows
        with G.LayerMaskTrace(visible, p) as masks:
            out = original(cache, suffix, prefix_len)
        assert masks.prefill == set(range(len(layers)))
        box["mask_calls"] = masks.calls
        return out

    server._decode = traced
    try:
        value = S.packed_hit(exp, ctx, path, prompt, logits=True, directcopy=directcopy)
    finally:
        server._decode = original
    assert box["visible_content_rows_per_layer"] == counts
    assert value["k_per_layer"] == counts and value["attended_content_kv_count_per_layer"] == counts
    assert value["keep_count_per_layer"] == [k + int(meta["v_token_num"]) - n for k in counts]
    assert value["selected_original_ids"] == ranking[:max(counts)]
    value["traced"] = box
    return value


def pyr_case(exp, G, S, M, sample, path, docs):
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
    counts = {v: hamilton(docs[v]["layer_keep_ratios"], n, docs[v]["r_max"]) for v in docs}     # independent, no lower bound
    rec = {}
    for v in docs:
        rec[f"LW_{v}"] = G.persist_layerwise(exp, cap, path / f"LW_{v}", sample, docs[v])
        rec[f"P_{v}"] = S.persist_packed(exp, cap, path / f"P_{v}", sample, docs[v])
        assert rec[f"LW_{v}"]["k_per_layer"] == rec[f"P_{v}"]["k_per_layer"] == counts[v] == dcounts(docs[v], n)
        assert sum(counts[v]) == 32 * k_u and min(counts[v]) >= 1
        for key in (f"LW_{v}", f"P_{v}"):
            m = rec[key]["metadata"]
            assert m["layer_budget_r_min"] == docs[v]["r_min"] and m["layer_budget_r_max"] == docs[v]["r_max"] and m["order"] == ranking + sep
    out["k_per_layer"] = counts
    out["k_last_layer"] = {v: counts[v][-1] for v in docs}
    out["meta_r_min"] = {v: rec[f"P_{v}"]["metadata"]["layer_budget_r_min"] for v in docs}
    # ---- G6 bytes: packed slices == unpacked layerwise files, bit for bit; same total bytes for both vectors (== uniform total)
    out["G6_bytes"] = {v: M._check_packed_slices(path / f"P_{v}", path / f"LW_{v}") for v in docs}
    row_bytes = 32 * 128 * 2
    totals = {v: out["G6_bytes"][v]["total_bytes"] for v in docs}
    assert len(set(totals.values())) == 1 and totals[GATE_VECTORS[0]] == 32 * k_u * 2 * row_bytes + 2 * 32 * len(sep) * row_bytes, totals
    out["total_bytes_equal_uniform_formula"] = True
    # ---- serve with the Turn-1 capture alive: unpacked (existing 65-pread path, traced) vs packed (traced) vs independent reference
    prompt = exp.runner.prompt(turn["question"])
    served = {}; cmp = {}; g7 = {}
    for v in docs:
        ctx, _ = exp.activate(path / f"LW_{v}", sample, "B")
        un = G.production_traced(exp, ctx, path / f"LW_{v}", sample, turn["question"], layers, ranking, counts[v])
        ctx.close(); del ctx
        ctxp, act = S.activate_packed(exp, path / f"P_{v}")
        ref = G.reference_run(exp.server, ctxp, turn["question"], layers, ranking, counts[v])
        sen = G.reference_run(exp.server, ctxp, turn["question"], layers, ranking, counts[v], sentinel=True)
        for mode in ("scatter", "directcopy"):
            key = f"P_{v}_{mode}"
            pv = packed_traced(exp, G, S, ctxp, path / f"P_{v}", prompt, layers, ranking, counts[v], directcopy=(mode == "directcopy"))
            a = {"first_token_id": pv["first_token_id"], "generated_token_ids": pv["generated_token_ids"], "prediction": pv["prediction"], "all_logits": pv["all_logits"]}
            b = {"first_token_id": un["result"]["first_token_id"], "generated_token_ids": un["result"]["generated_token_ids"], "prediction": un["result"]["prediction"], "all_logits": un["logits"]}
            c = M._same_output(a, b)
            assert c["first_token_exact"] and c["tokens_exact"] and c["answer_exact"] and c["logits_within_1e-4"], (key, c)
            r = G.compare(ref, pv["all_logits"], pv["first_token_id"], pv["generated_token_ids"], pv["prediction"])
            r["sentinel_reference_bitwise_identical"] = all(torch.equal(x, y) for x, y in zip(ref["logits"], sen["logits"])) and sen["generated_token_ids"] == ref["generated_token_ids"]
            assert r["sentinel_reference_bitwise_identical"]
            cmp[key] = {"vs_unpacked_layerwise": c, "vs_independent_reference": r, "restored_cache_bits_equal_capture": True, "masks_equal_independent": True,
                        "visible_content_rows_per_layer": pv["traced"]["visible_content_rows_per_layer"], "visible_rows_equal_k_l": pv["traced"]["visible_content_rows_per_layer"] == counts[v],
                        "visible_rows_last_layer": pv["traced"]["visible_content_rows_per_layer"][-1], "mask_calls": pv["traced"]["mask_calls"]}
            s = pv["actual_io"]
            g7[key] = {"preadv_calls": s["preadv_calls"], "pread_calls": s["pread_calls"], "returned_bytes": s["bytes"], "file_bytes": ctxp.meta["packed_total_bytes"],
                       "vision_calls": pv["vision_calls"], "query_score_calls": pv["online_query_score_calls"], "activation": act,
                       "packed_read_ms": pv["packed_read_ms"], "scatter_ms": pv["scatter_ms"], "ttft_ms": pv["ttft_ms"]}
            assert s["preadv_calls"] == 1 and s["pread_calls"] == 0 and s["bytes"] == ctxp.meta["packed_total_bytes"] == (path / f"P_{v}" / "kv_packed.bin").stat().st_size
            assert pv["vision_calls"] == 0 and pv["online_query_score_calls"] == 0
            served[key] = {"prediction": pv["prediction"], "generated_token_ids": pv["generated_token_ids"], "k_per_layer": pv["k_per_layer"],
                           "selected_original_ids": pv["selected_original_ids"], "bytes": s["bytes"]}
            del pv
        served[f"LW_{v}"] = {"bytes": un["reads"]["bytes"], "preads": un["reads"]["calls"]}
        assert served[f"LW_{v}"]["bytes"] == served[f"P_{v}_scatter"]["bytes"]
        ctxp.close(); del ctxp, un, ref, sen
        gc.collect(); torch.cuda.empty_cache()
    del cap, layers, enc, val
    gc.collect(); torch.cuda.empty_cache()
    # ---- G7 question invariance on fresh contexts (capture released)
    for v in docs:
        ctxp, _ = S.activate_packed(exp, path / f"P_{v}")
        for mode in ("scatter", "directcopy"):
            key = f"P_{v}_{mode}"
            a = S.packed_hit(exp, ctxp, path / f"P_{v}", prompt, logits=False, directcopy=(mode == "directcopy"))
            alt = S.packed_hit(exp, ctxp, path / f"P_{v}", exp.prompt("What is visible?", [(turn["question"], a["prediction"])], "mt"), logits=False, directcopy=(mode == "directcopy"))
            assert a["generated_token_ids"] == served[key]["generated_token_ids"] and a["prediction"] == served[key]["prediction"]
            assert alt["k_per_layer"] == a["k_per_layer"] == counts[v] and alt["selected_original_ids"] == a["selected_original_ids"] == served[key]["selected_original_ids"]
            ca, cb = a["actual_io"]["calls"][0], alt["actual_io"]["calls"][0]
            assert (ca["offset"], ca["requested"], ca["returned"]) == (cb["offset"], cb["requested"], cb["returned"]) == (0, ctxp.meta["packed_total_bytes"], ctxp.meta["packed_total_bytes"])
            assert alt["vision_calls"] == 0 and alt["online_query_score_calls"] == 0
            g7[key].update(question_invariant_range=True, fresh_hit_reproduces_traced=True)
        ctxp.close()
    out["G6_outputs"] = cmp
    out["G7"] = g7
    out["bytes"] = {k: v["bytes"] for k, v in served.items()}
    assert len(set(out["bytes"].values())) == 1, out["bytes"]
    out["persistence_ms"] = {k: r["persistence_ms"] for k, r in rec.items()}
    out["status"] = "PASS"
    return out


def stage_gpu(run):
    P = load("prefix25_persistence_existing", "106_prefix25_persistence.py")
    G = load("layer_budget_gates_v1", "154_validate_layer_budget_kv25.py")
    S = load("packed_support_v2", "packed_support_v2.py")
    M = load("packed_gates_v2", "161_validate_packed_store_v2.py")
    from mmimpress.recompute_kv25 import guarded_rmtree
    docs = docs_of(run, GATE_VECTORS)
    if not (run / "correctness_manifest.json").exists():
        shutil.copy2(G.REF_RUN / "correctness_manifest.json", run / "correctness_manifest.json")
    assert sha(run / "correctness_manifest.json") == sha(V2_RUN / "correctness_manifest.json")
    manifest = json.loads((run / "correctness_manifest.json").read_text())
    samples = [manifest["samples"][i] for i in G.FIXED_INDEX]      # the five fixed cases of v2 161 --stage gpu
    v2gate = json.loads((V2_RUN / "gates" / "gpu_gates.json").read_text())
    assert [s["image_id"] for s in samples] == [s["image_id"] for s in v2gate["samples"]]
    P.exclusive()
    torch.set_grad_enabled(False)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)
    exp = P.Experiment(run, "llava")
    gate = {"status": "RUNNING", "atol": ATOL, "rtol": RTOL, "vectors": {a: d["layer_budget_sha256"] for a, d in docs.items()},
            "clip_range": {a: [d["r_min"], d["r_max"]] for a, d in docs.items()}, "fixed_cases": [s["image_id"] for s in samples], "samples": [],
            "note": "per case and vector: unpacked layerwise store (existing writer, 65 preads) and packed store built from the same Turn-1 capture; "
                    "G6 = bit-identical slices, equal total bytes, restored cache bits / masks / first token / generated ids / logits; G7 = one preadv, zero pread, "
                    "returned bytes == file size, question-invariant read range, zero vision / score calls; visible rows per layer == independent k_l"}
    base = run / "llava" / "gpu_gates"
    try:
        for i, s in enumerate(samples):
            path = base / f"{i:02d}_{s['image_id']}"
            path.mkdir(parents=True, exist_ok=False)
            case = pyr_case(exp, G, S, M, s, path, docs)
            exp.close_contexts()
            gate["samples"].append(case)
            write(run / "gates" / "gpu_gates.json", gate)
            guarded_rmtree(path, base)                    # run-local gate stores deleted once the receipts are recorded (v2 amendment rule)
            gate["samples"][-1]["stores_deleted_after_record"] = True
            write(run / "gates" / "gpu_gates.json", gate)
            print("GPU gates", i + 1, len(samples), "PASS N", case["N"], "k_last", case["k_last_layer"], flush=True)
        gate["status"] = "PASS"
        gate["G6"] = gate["G7"] = "PASS"
        gate["raw_visible_rows_equal_k_l_all_cases"] = all(c["G6_outputs"][f"P_PYR-beta-raw_{m}"]["visible_rows_equal_k_l"] for c in gate["samples"] for m in ("scatter", "directcopy"))
    except Exception:
        gate["status"] = "FAIL"
        gate["error"] = traceback.format_exc()
        print(gate["error"], flush=True)
        raise
    finally:
        write(run / "gates" / "gpu_gates.json", gate)
        exp.close_contexts()


# ------------------------------------------------------------------ gdet
def _committed_rows(hold, phase, method, images=None):
    rows = {}
    for cp in sorted((hold / phase / "images").glob("*/COMMITTED.json")):
        if images is not None and cp.parent.name not in images:
            continue
        doc = json.loads(cp.read_text())
        raw = cp.parent / doc["attempt_id"] / "raw.jsonl"
        if sha(raw) != doc["raw_sha256"]:
            raise RuntimeError("committed raw hash mismatch: " + str(raw))
        with raw.open() as f:
            for line in f:
                if f'"method_id": "{method}"' not in line:
                    continue
                r = json.loads(line)
                assert r["method_id"] == method
                rows[(r["dialog_id"], r["turn_id"])] = r
    return rows


def stage_gdet(run, phase):
    method = f"{PK}U_packed"
    new = _committed_rows(run / "hold", phase, method)
    images = {r["image_id"] for r in new.values()}
    old = _committed_rows(V2_RUN / "hold", "main", method, images)
    manifest = json.loads((run / "hold" / "manifest.json").read_text())
    expected = len(manifest["smoke_dialogue_ids"]) * 3 if phase == "smoke" else 12183
    mism = []; token_equal = pred_equal = score_equal = prompt_equal = 0
    for key, r in sorted(new.items()):
        o = old.get(key)
        if o is None:
            mism.append({"dialog_id": key[0], "turn_id": key[1], "reason": "missing in v2"}); continue
        t = r["result"]["generated_token_ids"] == o["result"]["generated_token_ids"]
        token_equal += t; pred_equal += r["prediction"] == o["prediction"]; score_equal += r["score"] == o["score"]; prompt_equal += r["prompt"] == o["prompt"]
        if not t:
            mism.append({"dialog_id": key[0], "turn_id": key[1], "image_id": r["image_id"], "new": r["prediction"], "v2": o["prediction"],
                         "new_tokens": r["result"]["generated_token_ids"], "v2_tokens": o["result"]["generated_token_ids"], "prompt_equal": r["prompt"] == o["prompt"]})
    n = len(new)
    full = n == expected and token_equal == n and not mism
    out = {"phase": phase, "method": method, "requests": n, "expected_requests": expected, "v2_rows_found": sum(1 for k in new if k in old),
           "token_exact": int(token_equal), "token_exact_rate": token_equal / n if n else None, "prediction_equal": int(pred_equal), "score_equal": int(score_equal),
           "prompt_equal": int(prompt_equal), "mismatches": mism[:200], "mismatch_count": len(mism),
           "v2_reference": str((V2_RUN / "hold" / "main").relative_to(ROOT)),
           "comparison": "generated_token_ids of the same (dialog_id, turn_id); prompts include the method-local generated history",
           "status": "PASS" if full else ("FAIL" if phase == "smoke" else "NOT_COMPARABLE"),
           "cross_run_accuracy_comparison": "ALLOWED" if full and phase == "main" else ("PENDING_MAIN" if full else "NOT_COMPARABLE")}
    write(run / "gates" / f"g_det_{phase}.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "mismatches"}, indent=1), flush=True)
    if phase == "smoke" and not full:
        raise SystemExit("G-det smoke FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stage", choices=("g3", "grmin", "cpu", "gpu", "gdet"), required=True)
    ap.add_argument("--phase", choices=("smoke", "main"), default="smoke")
    args = ap.parse_args()
    run = Path(args.run).resolve()
    assert run.parent == ROOT / "runs" and run.name.startswith("layer_budget_kv25_pyrbeta_")
    (run / "gates").mkdir(parents=True, exist_ok=True)
    if args.stage == "gdet":
        stage_gdet(run, args.phase)
    else:
        {"g3": stage_g3, "grmin": stage_grmin, "cpu": stage_cpu, "gpu": stage_gpu}[args.stage](run)


if __name__ == "__main__":
    main()

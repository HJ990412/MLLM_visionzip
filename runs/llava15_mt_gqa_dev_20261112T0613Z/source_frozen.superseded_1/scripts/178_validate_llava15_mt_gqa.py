#!/usr/bin/env python3
"""Gates LG1-LG10 of the LLaVA-1.5-7B MT-GQA HOLD (contract docs/llava15_mt_gqa_contract.md, section 5).

  --stage cpu     : LG8 (vectors, row budgets, explicit layer_budget arguments, fail-closed metadata,
                    negative fixtures) and LG9 (complete existing CPU suite, existing sources unmodified)
  --stage gpu     : LG1 runner equivalence, LG2 geometry / order, LG3 capture, LG4 store bytes,
                    LG5 hit-path implementation, LG6 semantics, LG7 I/O -- on the 10 fixture images
  --stage summary : LG10 (integration run of scripts/179 + independent re-audit) and gates/summary.json

Validation samples are correctness fixtures, not a quality holdout.  A failing gate is recorded and
stops the run; thresholds and samples are fixed by the contract.
"""
from __future__ import annotations

import argparse
import ast
import copy
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import llava15_support as S
from llava15_support import ROOT, read, sha, write

import torch

BUDGET_FUNCS = {"validate_ratios": {"num_layers", "total_ratio", "r_min", "r_max"}, "layer_keep_counts": {"num_layers", "total_ratio", "r_min", "r_max"},
                "validate_keep_counts": {"num_layers", "total_ratio", "r_min", "r_max"}, "total_row_budget": {"num_layers", "total_ratio"},
                "keep_bounds": {"r_min", "r_max"}, "clip_redistribute": {"num_layers", "total_ratio", "r_min", "r_max"},
                "uniform_ratios": {"num_layers", "ratio"}, "budget_document": {"num_layers", "total_ratio", "r_min", "r_max"},
                "front_schedule": {"num_layers", "total_ratio", "r_min", "r_max"}, "allocate_water_filling": {"num_layers", "total_ratio", "r_min", "r_max"}}
FORBIDDEN = {"load_budget_document": "validates with the document's own fields but belongs to the LLaVA-NeXT 0.25 path",
             "write_packed_store": "asserts (N + 3) // 4", "PackedPrefixSelector": "rejects budget != 0.25", "validate_packed_meta": "0.25 only",
             "validate_llava_prefix_layerwise_meta": "0.25 only", "request_packed": "0.25 only", "persist_captured_visual_prefix": "AnyRes only",
             "anyres_token_scores": "AnyRes only"}


def bind(run):
    return {"config_sha256": sha(run / "config.json"), "contract_sha256": read(run / "config.json")["contract_sha256"]}


def expect_fail(label, fn, results, exc=(ValueError, AssertionError, RuntimeError)):
    try:
        fn()
    except exc as e:
        results[label] = {"fail_closed": True, "error": type(e).__name__ + ": " + str(e)[:160]}
        return
    results[label] = {"fail_closed": False}
    raise AssertionError(f"{label} did not fail closed")


# =========================================================================== LG8 (CPU)
def scan_budget_calls(path):
    """AST scan: every call of a default-taking mmimpress.layer_budget function must pass the listed
    arguments as explicit keywords; functions bound to the LLaVA-NeXT 0.25 path must not be called."""
    found, forbidden = [], []
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(node, ast.Call):
            fn = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else None
            if fn in BUDGET_FUNCS:
                kws = {kw.arg for kw in node.keywords if kw.arg}
                found.append({"file": str(Path(path).relative_to(ROOT)), "line": node.lineno, "call": fn,
                              "missing_explicit_keywords": sorted(BUDGET_FUNCS[fn] - kws)})
            if fn in FORBIDDEN:
                forbidden.append({"file": str(Path(path).relative_to(ROOT)), "line": node.lineno, "call": fn, "why": FORBIDDEN[fn]})
    return found, forbidden


def synthetic_capture(seed=S.SEED, v_start=5):
    """Real-geometry CPU tensors (32 layers, 32 heads, 5 + 576 rows, 128 dims) and a fixed saliency."""
    g = torch.Generator().manual_seed(seed)
    layers = [(torch.randn((1, 32, v_start + 576 + 3, 128), generator=g).to(torch.bfloat16),
               torch.randn((1, 32, v_start + 576 + 3, 128), generator=g).to(torch.bfloat16)) for _ in range(32)]
    scores = torch.rand((1, 576), generator=g)
    scores[0, 10] = scores[0, 400]                       # one exact tie -> stable order (lower index first)
    ids = torch.tensor([1, 2, 3, 4, 5] + [32000] * 576 + [7, 8, 9])
    return layers, scores, ids, v_start


class CpuRunner:
    """Minimal runner for the CPU fixture (geometry only; never calls a model)."""

    def __init__(self):
        self.model = SimpleNamespace(device=torch.device("cpu"))
        self.image_token_id = 32000

    def visual_span(self, ids):
        pos = (ids == self.image_token_id).nonzero(as_tuple=True)[0]
        assert pos.numel() == 576 and int(pos[-1]) - int(pos[0]) == 575
        return int(pos[0]), 576

    def flat_layout(self, v_num):
        assert v_num == 576
        return 24, []


CAPTURE_STATS = {"capture_saliency": True, "vision_call_count": 1, "saliency_call_count": 1, "extra_vision_forward_calls": 0,
                 "extra_saliency_forward_ms": 0.0, "saliency_layer_from_end": 2, "vision_attention_backend": "eager",
                 "global_vision_output_attentions": False, "global_model_output_attentions": False, "vision_ms": 1.0,
                 "vision_num_layers": 24, "saliency_layer_index": 22}


def build_cpu_store(path, vec, docs, capture):
    from mmimpress.llava15.store import persist_captured_llava15_prefix
    from mmimpress.prefix25 import seal_integrity
    layers, scores, ids, _ = capture
    receipt = persist_captured_llava15_prefix(CpuRunner(), layers, ids, scores, path, budget=docs[vec], image_id="cpu_fixture",
                                              capture_stats=CAPTURE_STATS, image_input_sha256="0" * 64)
    seal_integrity(path)
    return receipt["meta"]


def reseal(path):
    from mmimpress.prefix25 import seal_integrity
    (Path(path) / "integrity.json").unlink()
    seal_integrity(path)


def stage_cpu(run: Path):
    from unittest.mock import patch
    from mmimpress import layer_budget as LB
    from mmimpress.cvpr25 import visionzip_repack_order
    from mmimpress.llava15 import store as ST
    from mmimpress.llava15.serve import Llava15PackedContext, Llava15PrefixSelector, PackedReadTrace
    from mmimpress.serve import BIAS
    out = {"status": "RUNNING", **bind(run), "LG8": {}, "LG9": {}}
    g8 = out["LG8"]
    base = run / "gates" / "cpu_fixture"
    if base.exists():
        shutil.rmtree(base)
    try:
        # ---- vectors: SHA, sums, rows at N = 576
        docs = S.load_vectors(run)
        cfg = read(run / "config.json")
        vec_rec = {}
        for vec, doc in docs.items():
            r = doc["layer_keep_ratios"]
            assert LB.budget_sha256(r) == doc["layer_budget_sha256"] == S.RATIO_SHA[vec] == cfg["layer_budgets"][vec]["layer_budget_sha256"]
            assert abs(sum(r) - 32 * S.TOTAL_RATIO[vec]) <= 1e-9 and doc["r_min"] == 0.0 and doc["r_max"] == 1.0 and doc["num_layers"] == 32
            counts = ST.keep_counts(doc, 576)
            p, q = S.FRACTION[vec]
            assert sum(counts) == 32 * math.ceil(S.TOTAL_RATIO[vec] * 576) == 32 * (-(-p * 576 // q))
            assert counts == S.independent_counts(r, 576, S.FRACTION[vec]) and S.expected_bytes(counts) == S.EXPECTED_BYTES[vec]
            if vec in S.EXPECTED_ROWS:
                assert counts == S.EXPECTED_ROWS[vec]
            else:
                assert counts[:3] == S.EXPECTED_PYR_HEAD and counts[-4:] == S.EXPECTED_PYR_TAIL and sum(counts) == S.EXPECTED_PYR_SUM
            vec_rec[vec] = {"ratio_sha256": doc["layer_budget_sha256"], "ratio_sum": sum(r), "rows_sum_N576": sum(counts), "rows_head": counts[:3], "rows_tail": counts[-4:],
                            "payload_bytes": S.expected_bytes(counts), "file_sha256": sha(S.vector_path(run, vec))}
            assert vec_rec[vec]["file_sha256"] == cfg["vectors_file_sha256"][vec]
        assert docs["U25"]["layer_keep_ratios"] == [0.25] * 32 and docs["U50"]["layer_keep_ratios"] == [0.5] * 32 and docs["U100"]["layer_keep_ratios"] == [1.0] * 32
        src = read(run / "vectors" / "source_layer_budget_llava_PYR-beta-raw.json")
        assert sha(run / "vectors" / "source_layer_budget_llava_PYR-beta-raw.json") == S.PYR_SOURCE_FILE_SHA == sha(S.PYR_SOURCE)
        assert docs["PYR-b20"]["layer_keep_ratios"] == src["layer_keep_ratios"]
        g8["vectors"] = vec_rec
        g8["pyr_b20_copied_not_recomputed"] = {"source_file_sha256": S.PYR_SOURCE_FILE_SHA, "ratios_equal_source": True}

        # ---- explicit arguments: static scan of every new source
        calls, forbidden = [], []
        for rel in S.NEW_PATHS:
            if rel.endswith(".py"):
                c, f = scan_budget_calls(ROOT / rel)
                calls += c
                forbidden += f
        implicit = [c for c in calls if c["missing_explicit_keywords"]]
        allowed_forbidden = [f for f in forbidden if f["file"] in ("scripts/178_validate_llava15_mt_gqa.py", "tests/test_llava15_store.py")]
        offending_forbidden = [f for f in forbidden if f not in allowed_forbidden]
        g8["static_scan"] = {"files": sorted(p for p in S.NEW_PATHS if p.endswith(".py")), "budget_calls": calls, "default_reliant_calls": implicit,
                             "llava_next_bound_calls": offending_forbidden,
                             "note": "names matched inside this gate script / the unit test are string constants or deliberate negative checks, not calls on the serving path"}
        assert calls and not implicit, implicit
        assert not offending_forbidden, offending_forbidden
        assert LB.R_MIN == 0.10 and LB.R_MAX == 0.60 and LB.TOTAL_RATIO == 0.25 and LB.NUM_LAYERS == 32      # module defaults unchanged (and unused)

        # ---- CPU fixture stores at the real geometry
        fc = g8["fail_closed"] = {}
        capture = synthetic_capture()
        layers, scores, ids, v0 = capture
        order = visionzip_repack_order(scores[0], [])
        assert sorted(order) == list(range(576)) and order.index(10) < order.index(400)
        metas = {}
        runner = CpuRunner()
        cases = []
        for vec in S.VECTORS:
            path = base / vec
            meta = build_cpu_store(path, vec, docs, capture)
            metas[vec] = meta
            counts = ST.keep_counts(docs[vec], 576)
            ST.validate_llava15_packed_meta(meta, docs[vec])
            assert meta["order"] == order and meta["layer_keep_counts"] == counts and meta["packed_total_bytes"] == S.EXPECTED_BYTES[vec]
            assert (path / "kv_packed.bin").stat().st_size == S.EXPECTED_BYTES[vec] and meta["packed_sep_bytes"] == 0 and meta["newline_idx"] == []
            assert meta["layer_budget_r_min"] == 0.0 and meta["layer_budget_r_max"] == 1.0 and meta["layer_budget_total_ratio"] == S.TOTAL_RATIO[vec]
            slices = S.check_packed_slices(path, layers, v0, order, counts)
            assert slices["status"] == "PASS" and slices["file_sha256"] == meta["packed_sha256"], slices
            ctx = Llava15PackedContext(path, "cpu", pin=False, expected_budget=docs[vec])
            ctx.validate_packed_layout()
            cache = ctx.cache.new_request()
            sel = Llava15PrefixSelector(runner, ctx)
            with patch("torch.cuda.synchronize"), PackedReadTrace(path) as trace:
                sel.prepare()
            s = trace.summary()
            assert s["preads"] == 1 and s["preadv_calls"] == 1 and s["pread_calls"] == 0 and s["bytes"] == S.EXPECTED_BYTES[vec]
            masks = S.check_masks(BIAS, v0 + 576, v0, counts)
            bits = S.check_cache_bits(cache, layers, v0, order, counts)
            assert masks["status"] == "PASS" and masks["visible_rows_equal_k_l"] and bits["status"] == "PASS", (masks, bits)
            st = sel.stats()
            assert st["k_per_layer"] == counts and st["selected_original_ids"] == order[:max(counts)] and st["ratio"] == S.TOTAL_RATIO[vec]
            if vec == "PYR-b20":      # negative fixture (c): one row that must be hidden is exposed
                li = 20
                tampered = {k: v.clone() for k, v in BIAS.items()}
                tampered[li][0, 0, 0, v0 + counts[li]] = 0
                bad = S.check_masks(tampered, v0 + 576, v0, counts)
                assert bad["status"] == "FAIL" and bad["bad_layers"] == [li]
                fc["negative_one_hidden_row_exposed"] = {"fail_closed": True, "layer": li, "stored_row": counts[li], "detected_by": "check_masks (bitwise mask comparison)"}
                short = {k: v.clone() for k, v in BIAS.items()}
                short[li][0, 0, 0, v0 + counts[li] - 1] = torch.finfo(torch.bfloat16).min
                assert S.check_masks(short, v0 + 576, v0, counts)["bad_layers"] == [li]
                fc["negative_one_visible_row_hidden"] = {"fail_closed": True, "layer": li, "detected_by": "check_masks"}
            BIAS.clear()
            cases.append({"vector": vec, "rows_sum": sum(counts), "bytes": meta["packed_total_bytes"], "slices": "PASS", "masks": "PASS", "cache_bits": "PASS", "single_preadv": True})
            ctx.close()
            del cache, ctx
        g8["cpu_fixture_cases"] = cases
        assert len({json.dumps(m["order"]) for m in metas.values()}) == 1
        # nesting of the visible sets across ratios
        assert metas["U25"]["stored_row_to_original"] == metas["U50"]["stored_row_to_original"][:144] == metas["U100"]["stored_row_to_original"][:144]

        # ---- fail-closed metadata: module defaults in place of the frozen values
        for vec in S.VECTORS:
            m = metas[vec]
            expect_fail(f"{vec}: meta r_min -> 0.10 (module default)", lambda m=m: ST.validate_llava15_packed_meta(dict(m, layer_budget_r_min=0.10)), fc)
            expect_fail(f"{vec}: meta r_max -> 0.60 (module default)", lambda m=m: ST.validate_llava15_packed_meta(dict(m, layer_budget_r_max=0.60)), fc)
            other_total = 0.25 if S.TOTAL_RATIO[vec] != 0.25 else 0.5
            expect_fail(f"{vec}: meta total_ratio -> {other_total}", lambda m=m, t=other_total: ST.validate_llava15_packed_meta(dict(m, layer_budget_total_ratio=t)), fc)
            for key in ("layer_budget_total_ratio", "layer_budget_r_min", "layer_budget_r_max", "layer_budget_num_layers"):
                expect_fail(f"{vec}: meta without {key}", lambda m=m, key=key: ST.validate_llava15_packed_meta({k: v for k, v in m.items() if k != key}), fc)
            expect_fail(f"{vec}: budget document r_min -> 0.10", lambda d=docs[vec]: ST.keep_counts(dict(d, r_min=0.10), 576), fc)
            expect_fail(f"{vec}: budget document without total_ratio", lambda d=docs[vec]: ST.keep_counts({k: v for k, v in d.items() if k != "total_ratio"}, 576), fc)
        # the LLaVA-NeXT default-reliant pattern must not be usable on these vectors
        expect_fail("layer_keep_counts(U100 ratios, 576) with module defaults", lambda: LB.layer_keep_counts(docs["U100"]["layer_keep_ratios"], 576), fc)
        expect_fail("layer_keep_counts(U50 ratios, 576) with module defaults", lambda: LB.layer_keep_counts(docs["U50"]["layer_keep_ratios"], 576), fc)
        expect_fail("layer_keep_counts(PYR-b20 ratios, 576) with module defaults", lambda: LB.layer_keep_counts(docs["PYR-b20"]["layer_keep_ratios"], 576), fc)

        # ---- negative fixture (a): one layer k_l +/- 1 (every dependent field kept self-consistent)
        def shifted(meta, li, delta):
            m = copy.deepcopy(meta)
            counts = list(m["layer_keep_counts"])
            counts[li] += delta
            table, sep_off, _, total = ST.packed_layout(counts, m["packed_row_bytes"], 32, 0)
            m.update(layer_keep_counts=counts, payload_rows_per_layer=list(counts), stored_content_count_per_layer=list(counts),
                     stored_content_count_total=sum(counts), packed_layout=table, packed_sep_offset=sep_off, packed_total_bytes=total,
                     bytes_visual_kv=total, stored_row_to_original=m["order"][:max(counts)])
            return m
        for vec, li in (("U25", 7), ("U50", 0), ("PYR-b20", 31), ("PYR-b20", 0)):
            for delta in (+1, -1):
                expect_fail(f"negative k_l {delta:+d}: {vec} layer {li}", lambda v=vec, li=li, d=delta: ST.validate_llava15_packed_meta(shifted(metas[v], li, d)), fc)
        expect_fail("negative k_l -1: U100 layer 3", lambda: ST.validate_llava15_packed_meta(shifted(metas["U100"], 3, -1)), fc)

        # ---- negative fixture (b): two layers swapped -- counts in the metadata, and payload blocks on disk
        def swapped_counts(meta, a, b):
            m = copy.deepcopy(meta)
            counts = list(m["layer_keep_counts"])
            counts[a], counts[b] = counts[b], counts[a]
            table, sep_off, _, total = ST.packed_layout(counts, m["packed_row_bytes"], 32, 0)
            m.update(layer_keep_counts=counts, payload_rows_per_layer=list(counts), stored_content_count_per_layer=list(counts), packed_layout=table)
            return m
        expect_fail("negative swap: PYR-b20 k_0 <-> k_31 in metadata", lambda: ST.validate_llava15_packed_meta(swapped_counts(metas["PYR-b20"], 0, 31)), fc)
        expect_fail("negative swap: PYR-b20 k_10 <-> k_11 in metadata", lambda: ST.validate_llava15_packed_meta(swapped_counts(metas["PYR-b20"], 10, 11)), fc)
        bad = base / "U25_layers_swapped"
        shutil.copytree(base / "U25", bad)
        blk = 144 * S.ROW_BYTES
        with (bad / "kv_packed.bin").open("r+b") as f:      # swap the K blocks of layers 3 and 7 (same row count -> same file size)
            f.seek(3 * 2 * blk); a = f.read(blk)
            f.seek(7 * 2 * blk); b = f.read(blk)
            f.seek(3 * 2 * blk); f.write(b)
            f.seek(7 * 2 * blk); f.write(a)
        assert (bad / "kv_packed.bin").stat().st_size == S.EXPECTED_BYTES["U25"]
        sl = S.check_packed_slices(bad, layers, v0, order, ST.keep_counts(docs["U25"], 576))
        assert sl["status"] == "FAIL" and sl["mismatched"] == [{"layer": 3, "kind": "k"}, {"layer": 7, "kind": "k"}] and sl["file_sha256"] != metas["U25"]["packed_sha256"]
        fc["negative swap: U25 layer 3 <-> 7 K blocks on disk (independent slice check + packed SHA256)"] = {"fail_closed": True, "mismatched": sl["mismatched"]}
        expect_fail("negative swap: U25 layer 3 <-> 7 K blocks on disk (activation integrity)", lambda: Llava15PackedContext(bad, "cpu", pin=False), fc)

        # ---- negative fixture (d): another vector's SHA / document
        expect_fail("negative vector: U25 store activated as PYR-b20", lambda: Llava15PackedContext(base / "U25", "cpu", pin=False, expected_budget=docs["PYR-b20"]), fc)
        expect_fail("negative vector: PYR-b20 store activated as U25", lambda: Llava15PackedContext(base / "PYR-b20", "cpu", pin=False, expected_budget=docs["U25"]), fc)
        expect_fail("negative vector: U25 meta with the PYR-b20 ratio SHA256", lambda: ST.validate_llava15_packed_meta(dict(metas["U25"], layer_budget_sha256=S.RATIO_SHA["PYR-b20"])), fc)
        expect_fail("negative vector: PYR-b20 meta with the U25 ratios", lambda: ST.validate_llava15_packed_meta(dict(metas["PYR-b20"], layer_keep_ratios=[0.25] * 32)), fc)
        expect_fail("negative vector: U50 store activated as U100", lambda: Llava15PackedContext(base / "U50", "cpu", pin=False, expected_budget=docs["U100"]), fc)

        # ---- on-disk tampering with a re-sealed envelope (only the metadata is wrong) / payload damage
        for label, mutate in (("meta.json r_min 0.10 re-sealed", lambda m: m.update(layer_budget_r_min=0.10)),
                              ("meta.json total_ratio 0.5 re-sealed", lambda m: m.update(layer_budget_total_ratio=0.5)),
                              ("meta.json k_5 + 1 re-sealed", lambda m: m["layer_keep_counts"].__setitem__(5, m["layer_keep_counts"][5] + 1)),
                              ("meta.json order transposition re-sealed", lambda m: m["order"].__setitem__(slice(0, 2), m["order"][1::-1]))):
            d = base / ("U25_" + hashlib.sha256(label.encode()).hexdigest()[:8])
            shutil.copytree(base / "U25", d)
            m = json.loads((d / "meta.json").read_text())
            mutate(m)
            (d / "meta.json").write_text(json.dumps(m, indent=1))
            reseal(d)
            expect_fail(f"activation: {label}", lambda d=d: (lambda c: c.validate_packed_layout())(Llava15PackedContext(d, "cpu", pin=False)), fc)
            shutil.rmtree(d)
        d = base / "U25_truncated"
        shutil.copytree(base / "U25", d)
        with (d / "kv_packed.bin").open("r+b") as f:
            f.truncate(S.EXPECTED_BYTES["U25"] - S.ROW_BYTES)
        expect_fail("activation: truncated kv_packed.bin", lambda: Llava15PackedContext(d, "cpu", pin=False), fc)
        reseal(d)
        expect_fail("activation: truncated kv_packed.bin with a re-sealed envelope", lambda: Llava15PackedContext(d, "cpu", pin=False), fc)
        shutil.rmtree(d)
        d = base / "U25_flipped"
        shutil.copytree(base / "U25", d)
        with (d / "kv_packed.bin").open("r+b") as f:
            f.seek(12345); byte = f.read(1); f.seek(12345); f.write(bytes([byte[0] ^ 0x01]))
        expect_fail("activation: one flipped payload bit (integrity SHA256)", lambda: Llava15PackedContext(d, "cpu", pin=False), fc)
        assert S.check_packed_slices(d, layers, v0, order, ST.keep_counts(docs["U25"], 576))["status"] == "FAIL"
        shutil.rmtree(d)
        d = base / "U25_no_envelope"
        shutil.copytree(base / "U25", d)
        (d / "integrity.json").unlink()
        expect_fail("activation: missing integrity envelope", lambda: Llava15PackedContext(d, "cpu", pin=False), fc)
        shutil.rmtree(d)
        # selector on an activated context whose in-memory metadata is changed
        ctx = Llava15PackedContext(base / "U50", "cpu", pin=False)
        good = ctx.meta
        for label, change in (("selector: meta r_min 0.10", dict(layer_budget_r_min=0.10)), ("selector: meta total_ratio 0.25", dict(layer_budget_total_ratio=0.25)),
                              ("selector: meta ratio SHA256 of U25", dict(layer_budget_sha256=S.RATIO_SHA["U25"]))):
            ctx.meta = dict(good, **change)
            expect_fail(label, lambda: Llava15PrefixSelector(runner, ctx), fc)
        ctx.meta = good
        ctx.close()
        # writer: wrong geometry / documents
        expect_fail("writer: order is not a permutation", lambda: ST.write_llava15_packed_store(base / "w1", layers, v0, 576, ids[:v0 + 576].tolist(), order[:-1] + [order[0]], budget=docs["U25"]), fc)
        expect_fail("writer: 575 visual tokens", lambda: ST.write_llava15_packed_store(base / "w2", layers, v0, 575, ids[:v0 + 575].tolist(), order[:575], budget=docs["U25"]), fc)
        expect_fail("writer: document with r_max 0.60", lambda: ST.write_llava15_packed_store(base / "w3", layers, v0, 576, ids[:v0 + 576].tolist(), order, budget=dict(docs["U25"], r_max=0.60)), fc)
        expect_fail("persist: saliency of shape (1, 575)", lambda: ST.persist_captured_llava15_prefix(runner, layers, ids, scores[:, :575], base / "w4", budget=docs["U25"], image_id="x", capture_stats=CAPTURE_STATS), fc)
        expect_fail("persist: capture provenance with a second vision call", lambda: ST.persist_captured_llava15_prefix(runner, layers, ids, scores, base / "w5", budget=docs["U25"], image_id="x", capture_stats=dict(CAPTURE_STATS, vision_call_count=2)), fc)
        assert all(v["fail_closed"] for v in fc.values())
        g8["fail_closed_checks"] = len(fc)
        g8["status"] = "PASS"
    except Exception:
        g8["status"] = "FAIL"
        g8["error"] = traceback.format_exc()
        out["status"] = "FAIL"
        write(run / "gates" / "cpu.json", out)
        raise
    finally:
        try:
            from mmimpress.serve import BIAS as _B
            _B.clear()
        except Exception:
            pass
        if base.exists():
            shutil.rmtree(base)                         # run-local CPU fixture only
        g8["fixture_deleted"] = not base.exists()

    # ---- LG9: complete existing CPU suite (+ the new tests); GLM modules separately; existing sources unmodified
    g9 = out["LG9"]
    env = {**os.environ, "PYTHONPATH": ".:scripts:tests", "PYTHONDONTWRITEBYTECODE": "1"}
    modules = sorted(p.stem for p in (ROOT / "tests").glob("test_*.py"))
    glm = [m for m in modules if m.startswith("test_glm46v")]
    main = [m for m in modules if m not in glm]
    log = run / "gates" / "cpu_tests.log"
    with log.open("w") as f:
        p = subprocess.run([sys.executable, "-m", "unittest"] + main, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
    text = log.read_text()
    ok = p.returncode == 0 and "\nOK" in text
    ran = [l for l in text.splitlines() if l.startswith("Ran ")]
    glm_log = run / "gates" / "cpu_tests_glm_preexisting.log"
    with glm_log.open("w") as f:
        q = subprocess.run([sys.executable, "-m", "unittest"] + glm, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
    gtext = glm_log.read_text()
    glm_errors = [l for l in gtext.splitlines() if l.startswith("ERROR:") or l.startswith("FAIL:")]
    glm_touch = [m for m in glm if "llava15" in (ROOT / "tests" / f"{m}.py").read_text()]
    before = read(run / "protection_before.json")["sha256"]
    roots = ("mmimpress/", "scripts/", "tests/", "docs/")
    changed = sorted(k for k, v in before.items() if k.startswith(roots) and (not (ROOT / k).is_file() or sha(ROOT / k) != v["sha256"]))
    mods = read(run / "source_modifications.json")
    g9.update({"status": "PASS" if ok and not changed and mods["status"] == "PASS" and not glm_touch else "FAIL", "returncode": p.returncode, "modules": len(main),
               "new_test_modules": [m for m in main if m.startswith("test_llava15")], "ran": ran[-1] if ran else None, "log_sha256": sha(log), "tail": text[-300:],
               "glm_modules_run_separately": glm, "glm_returncode": q.returncode, "glm_errors": glm_errors, "glm_log_sha256": sha(glm_log),
               "glm_note": "pre-existing environment failures of the GLM-4.6V test modules (they need the transformers 5.x API; this environment is 4.57.6); "
                           "they import nothing from mmimpress.llava15; reported separately, not hidden",
               "existing_sources_checked": sum(1 for k in before if k.startswith(roots)), "existing_sources_changed": changed,
               "existing_source_check_vs_previous_run": mods["status"]})
    out["status"] = "PASS" if g8["status"] == "PASS" and g9["status"] == "PASS" else "FAIL"
    write(run / "gates" / "cpu.json", out)
    print(json.dumps({"LG8": g8["status"], "fail_closed_checks": g8.get("fail_closed_checks"), "LG9": g9["status"], "tests": g9["ran"],
                      "glm_preexisting_errors": len(glm_errors), "existing_sources_changed": len(changed)}), flush=True)
    assert out["status"] == "PASS"


# =========================================================================== GPU gates
class LlmCallGuard:
    """Counts decoder-layer forwards (the LLM must not run while the order is computed)."""

    def __init__(self, runner):
        self.runner, self.calls = runner, 0

    def __enter__(self):
        self.h = self.runner.layers[0].register_forward_pre_hook(lambda *a, **k: setattr(self, "calls", self.calls + 1))
        return self

    def __exit__(self, *exc):
        self.h.remove()


def hf_generate(runner, image, prompt):
    """Plain HF path: processor -> model.generate(greedy, max 16).  No Server, no hooks, no stopping criteria."""
    enc = runner.processor(images=image, text=prompt, return_tensors="pt").to(runner.model.device)
    out = runner.model.generate(**enc, do_sample=False, max_new_tokens=16)
    return out[0, enc["input_ids"].shape[1]:].tolist()


def kv_compare(layers, other, upto):
    from mmimpress.model import cache_layers
    other = cache_layers(other)
    unequal, rel = [], []
    for li, ((ck, cv), (pk, pv)) in enumerate(zip(layers, other)):
        if not (torch.equal(ck[:, :, :upto], pk[:, :, :upto]) and torch.equal(cv[:, :, :upto], pv[:, :, :upto])):
            unequal.append(li)
        num = (ck[:, :, :upto].float() - pk[:, :, :upto].float()).norm() + (cv[:, :, :upto].float() - pv[:, :, :upto].float()).norm()
        den = ck[:, :, :upto].float().norm() + cv[:, :, :upto].float().norm()
        rel.append(float(num / den))
    return unequal, rel


def traced_hit(runner, server, ctx, factory, image_id, layers, v0, order, counts):
    """One production hit through the public path.  Before the forward, the restored cache bits and the installed
    masks are compared with the capture / the independent visibility; first-step logits are recorded."""
    from mmimpress.serve import BIAS
    original = server._decode
    box = {"logits": []}

    def traced(cache, suffix, prefix_len):
        box["masks"] = S.check_masks(BIAS, prefix_len, v0, counts)
        box["bits"] = S.check_cache_bits(cache, layers, v0, order, counts) if layers is not None else {"status": "SKIPPED"}
        h = runner.model.register_forward_hook(lambda m, a, o: box["logits"].append(o.logits[0, -1].detach().float().cpu()) if hasattr(o, "logits") else None)
        try:
            return original(cache, suffix, prefix_len)
        finally:
            h.remove()

    server._decode = traced
    try:
        result, diag = S.stored_request(runner, server, ctx, factory, image_id)
    finally:
        server._decode = original
    result["traced"] = box
    return result, diag


def scatter_breakdown(runner, ctx, counts):
    """DIAGNOSTIC (not a gate, not the timed path): the production scatter split into its three operations with
    extra synchronisation -- FP16->BF16 conversion on the host, host-to-device copy, GPU index write + mask.
    Runs on the buffer filled by the preceding hit; returns per-component ms and a bitwise check vs PrefixCache.write."""
    from mmimpress.serve import bias_from_keep
    m = ctx.meta
    dev = runner.model.device
    v0 = int(m["v_token_start"])
    ctx.cache.new_request()
    ref_k = [t.clone() for t in ctx.cache.k]
    ref_v = [t.clone() for t in ctx.cache.v]
    for li in range(32):
        for kind in ("k", "v"):
            if counts[li]:
                ctx.cache.write(li, kind, torch.arange(counts[li], dtype=torch.long), ctx.reader.layer_view(li, kind))
    prod_k = [t.clone() for t in ctx.cache.k]
    prod_v = [t.clone() for t in ctx.cache.v]
    t = {"host_fp16_to_bf16_ms": 0.0, "h2d_ms": 0.0, "gpu_index_write_ms": 0.0, "mask_ms": 0.0}
    equal = True
    for li in range(32):
        k = counts[li]
        for kind, ref, prod in (("k", ref_k, prod_k), ("v", ref_v, prod_v)):
            if not k:
                continue
            vals = ctx.reader.layer_view(li, kind)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            host = vals.to(torch.bfloat16)
            t1 = time.perf_counter()
            gpu = host.to(dev)
            torch.cuda.synchronize(); t2 = time.perf_counter()
            idx = torch.arange(k, dtype=torch.long).to(dev) + v0
            ref[li][0, :, idx, :] = gpu.permute(1, 0, 2)
            torch.cuda.synchronize(); t3 = time.perf_counter()
            t["host_fp16_to_bf16_ms"] += (t1 - t0) * 1e3; t["h2d_ms"] += (t2 - t1) * 1e3; t["gpu_index_write_ms"] += (t3 - t2) * 1e3
            equal = equal and torch.equal(ref[li], prod[li])
        torch.cuda.synchronize(); t0 = time.perf_counter()
        keep = torch.zeros(576, dtype=torch.bool); keep[:k] = True
        _ = bias_from_keep(keep, m, dev)
        torch.cuda.synchronize(); t["mask_ms"] += (time.perf_counter() - t0) * 1e3
    ctx.cache.k = None; ctx.cache.v = None
    t["replica_bitwise_equal_to_production_scatter"] = bool(equal)
    t["h2d_bytes"] = int(sum(2 * k * S.ROW_BYTES for k in counts))
    return t


def gpu_image(runner, server, run, docs, manifest, fixture, methods, first):
    from mmimpress.cvpr25 import clip_cls_patch_saliency, permutation_sha256, visionzip_repack_order
    from mmimpress.llava15 import store as ST
    from mmimpress.llava15.serve import Llava15PackedContext
    from mmimpress.model import cache_layers
    from mmimpress.recompute_kv25 import guarded_rmtree
    dialog = next(d for d in manifest["dialogues"] if d["dialog_id"] == fixture["dialog_id"])
    assert dialog["image_id"] == fixture["image_id"]
    image_id = dialog["image_id"]
    image_path = ROOT / dialog["image_path"]
    image = S.load_image(image_path)
    out = {"image_id": image_id, "dialog_id": dialog["dialog_id"]}
    dev = runner.model.device
    tok = runner.processor.tokenizer

    # ---------------- recompute path, 3 turns, method-local generated history (LG1) ----------------
    generated, rc, prompts = {}, {}, {}
    cap = None
    for turn in (1, 2, 3):
        factory = lambda t=turn: S.prompt_factory(methods, dialog, t, S.RECOMP, generated)
        result, diag = S.normal_request(runner, server, image_path, factory, True)       # capture on every turn (saliency for LG2)
        prompts[turn] = diag["prompt"]
        hf = hf_generate(runner, image, diag["prompt"])
        rc[turn] = {"tokens": result["generated_token_ids"], "first": result["first_token_id"], "answer": result["answer"], "hf_tokens": hf,
                    "saliency": diag["vision"].result_cpu().clone(), "ttft_ms": result["end_to_end_ttft_ms"]}
        generated[turn] = {"method_id": S.RECOMP, "dialog_id": dialog["dialog_id"], "prediction": result["answer"],
                           "logical_request_id": f"gate:{image_id}:t{turn}", "physical_execution_id": f"gate:{image_id}:t{turn}"}
        if turn == 1:
            cap = {"result": result, "diag": diag}
        else:
            result.pop("captured_past_key_values", None)
    out["LG1"] = {"status": "PASS" if all(rc[t]["tokens"] == rc[t]["hf_tokens"] for t in (1, 2, 3)) else "FAIL",
                  "turns": {t: {"runner_tokens": rc[t]["tokens"], "hf_generate_tokens": rc[t]["hf_tokens"], "equal": rc[t]["tokens"] == rc[t]["hf_tokens"],
                                "answer": rc[t]["answer"]} for t in (1, 2, 3)}}
    assert out["LG1"]["status"] == "PASS", out["LG1"]

    # ---------------- geometry and order (LG2) ----------------
    enc = cap["diag"]["enc_cpu"]
    ids = enc["input_ids"][0]
    spans = runner.visual_spans(ids)
    v0, vn = runner.visual_span(ids)
    P = v0 + vn
    base, seps = runner.flat_layout(vn)
    sal = rc[1]["saliency"]
    order = visionzip_repack_order(ST.token_scores_from_capture(sal), [])
    with LlmCallGuard(runner) as guard:
        alone = clip_cls_patch_saliency(runner, runner.image_inputs(image)["pixel_values"])
        alone_again = clip_cls_patch_saliency(runner, runner.image_inputs(image)["pixel_values"])
    order_alone = visionzip_repack_order(ST.token_scores_from_capture(alone), [])
    order_sha = permutation_sha256(order)
    other = {t: permutation_sha256(visionzip_repack_order(ST.token_scores_from_capture(rc[t]["saliency"]), [])) for t in (2, 3)}
    lg2 = {"visual_spans": spans, "v_token_start": v0, "v_token_num": vn, "one_contiguous_span_of_576": spans == [(v0, 576)] and vn == 576,
           "separators": len(seps), "image_token_count": int((ids == runner.image_token_id).sum()), "base_grid": base,
           "pixel_values_shape": list(enc["pixel_values"].shape), "has_image_sizes": "image_sizes" in enc,
           "saliency_shape": list(sal.shape), "saliency_finite": bool(torch.isfinite(sal).all()), "saliency_dtype": str(sal.dtype),
           "order_is_permutation_of_576": sorted(order) == list(range(576)), "order_sha256": order_sha,
           "order_stable_descending": order == sorted(range(576), key=lambda i: (-float(sal[0, i]), i)),
           "vision_only_order_sha256": permutation_sha256(order_alone), "vision_only_llm_layer_calls": guard.calls,
           "vision_only_saliency_bitwise_equal_to_turn1_capture": bool(torch.equal(alone, sal)),
           "vision_only_repeat_bitwise_equal": bool(torch.equal(alone, alone_again)),
           "order_sha256_other_questions": other}
    lg2["status"] = "PASS" if (lg2["one_contiguous_span_of_576"] and lg2["separators"] == 0 and lg2["image_token_count"] == 576 and not lg2["has_image_sizes"]
                               and lg2["saliency_shape"] == [1, 576] and lg2["saliency_finite"] and lg2["order_is_permutation_of_576"] and lg2["order_stable_descending"]
                               and lg2["vision_only_order_sha256"] == order_sha and lg2["vision_only_llm_layer_calls"] == 0 and lg2["vision_only_repeat_bitwise_equal"]
                               and all(v == order_sha for v in other.values())) else "FAIL"
    out["LG2"] = lg2
    assert lg2["status"] == "PASS", lg2

    # ---------------- capture correctness (LG3) ----------------
    layers = cache_layers(cap["result"]["captured_past_key_values"])
    T = int(ids.numel())
    assert len(layers) == 32 and layers[0][0].shape[2] >= T
    px = enc["pixel_values"].to(dev)
    fillers = {"pad_token": torch.full((T - P,), int(tok.pad_token_id)), "token_id_0": torch.zeros(T - P, dtype=torch.long),
               "seeded_random_ids": torch.randint(1000, 30000, (T - P,), generator=torch.Generator().manual_seed(S.SEED))}
    lg3 = {"prefix_len": P, "turn1_sequence_len": T, "variants": {}}
    for name, fill in fillers.items():
        alt = enc["input_ids"].clone()
        alt[0, P:] = fill
        assert not torch.equal(alt, enc["input_ids"]) and torch.equal(alt[0, :P], ids[:P])
        res = runner.model(input_ids=alt.to(dev), pixel_values=px, attention_mask=torch.ones(1, T, dtype=torch.long, device=dev), use_cache=True, return_dict=True)
        unequal, rel = kv_compare(layers, res.past_key_values, P)
        suffix_differs = not torch.equal(cache_layers(res.past_key_values)[8][0][:, :, P:T], layers[8][0][:, :, P:T])
        lg3["variants"][name] = {"layers_not_bitwise_equal": unequal, "max_rel_l2": max(rel), "question_rows_differ_from_turn1": bool(suffix_differs)}
        del res
    literal = runner.prefix_forward(enc, P)
    unequal, rel = kv_compare(layers, literal.past_key_values, P)
    lg3["literal_shorter_prefix_forward_DIAGNOSTIC"] = {"sequence_len": P, "layers_not_bitwise_equal": len(unequal), "layer0_bitwise_equal": 0 not in unequal,
                                                         "max_rel_l2": max(rel), "rel_l2_last_layer": rel[-1],
                                                         "note": "not gated: BF16 GEMM rounding depends on the tensor shape, so a forward of a different length is not bit-identical "
                                                                 "beyond layer 0 (same finding as scripts/30 and docs/recompute_kv25_ablation_contract.md)"}
    del literal
    lg3["status"] = "PASS" if all(not v["layers_not_bitwise_equal"] and v["question_rows_differ_from_turn1"] for v in lg3["variants"].values()) else "FAIL"
    out["LG3"] = lg3
    assert lg3["status"] == "PASS", lg3

    # ---------------- stores (LG4) ----------------
    store_root = run / "gates" / "stores" / image_id
    store_root.mkdir(parents=True, exist_ok=False)
    source = {"dialog_id": dialog["dialog_id"], "method_id": "gate_fixture", "physical_execution_id": f"gate:{image_id}:t1", "logical_request_id": f"gate:{image_id}:t1"}
    lg4, metas = {}, {}
    for vec in S.VECTORS:
        path = store_root / vec
        p = S.persist(runner, cap["result"], cap["diag"], path, image_id, "packed_" + vec, source, docs)
        meta = read(path / "meta.json")
        metas[vec] = meta
        counts = ST.keep_counts(docs[vec], 576)
        ST.validate_llava15_packed_meta(meta, docs[vec])
        sl = S.check_packed_slices(path, layers, v0, order, counts)
        ok = (sl["status"] == "PASS" and sl["file_bytes"] == S.EXPECTED_BYTES[vec] == meta["packed_total_bytes"] and meta["packed_sep_bytes"] == 0
              and sl["file_sha256"] == meta["packed_sha256"] == p["files"]["kv_packed.bin"]["sha256"] and meta["order"] == order
              and meta["layer_keep_counts"] == counts == S.independent_counts(docs[vec]["layer_keep_ratios"], 576, S.FRACTION[vec])
              and meta["v_token_start"] == v0 and meta["prefix_input_ids"] == ids[:P].tolist())
        lg4[vec] = {"status": "PASS" if ok else "FAIL", "file_bytes": sl["file_bytes"], "expected_bytes": S.EXPECTED_BYTES[vec], "separator_bytes": meta["packed_sep_bytes"],
                    "layer_slices_bitwise_equal_to_capture_fp16_rows": sl["status"] == "PASS", "mismatched": sl["mismatched"],
                    "packed_sha256_recomputed_equal": sl["file_sha256"] == meta["packed_sha256"], "persistence_ms": p["service_persistence_ms"],
                    "files": sorted(p["files"])}
        assert ok, lg4[vec]
    if first:      # validator / activation rejections on the real stores of the first fixture image
        neg = {}
        m = metas["PYR-b20"]
        bump = copy.deepcopy(m); bump["layer_keep_counts"][4] += 1
        expect_fail("real store: tampered meta (k_4 + 1)", lambda: ST.validate_llava15_packed_meta(bump), neg)
        expect_fail("real store: tampered meta (r_min 0.10)", lambda: ST.validate_llava15_packed_meta(dict(m, layer_budget_r_min=0.10)), neg)
        expect_fail("real store: tampered meta (total_ratio 0.5)", lambda: ST.validate_llava15_packed_meta(dict(m, layer_budget_total_ratio=0.5)), neg)
        expect_fail("real store: tampered ratio SHA256", lambda: ST.validate_llava15_packed_meta(dict(m, layer_budget_sha256=S.RATIO_SHA["U25"])), neg)
        expect_fail("real store: wrong row count vs another vector", lambda: Llava15PackedContext(store_root / "PYR-b20", dev, expected_budget=docs["U25"]), neg)
        d = store_root / "neg_truncated"
        shutil.copytree(store_root / "U25", d)
        with (d / "kv_packed.bin").open("r+b") as f:
            f.truncate(S.EXPECTED_BYTES["U25"] - 4096)
        expect_fail("real store: truncated file", lambda: Llava15PackedContext(d, dev), neg)
        shutil.rmtree(d)
        d = store_root / "neg_sha"
        shutil.copytree(store_root / "U25", d)
        with (d / "kv_packed.bin").open("r+b") as f:
            f.seek(777777); b = f.read(1); f.seek(777777); f.write(bytes([b[0] ^ 0x80]))
        expect_fail("real store: payload SHA256 mismatch", lambda: Llava15PackedContext(d, dev), neg)
        shutil.rmtree(d)
        lg4["validator_rejections"] = neg
    out["LG4"] = dict(lg4, status="PASS")

    # ---------------- hit path: LG5 / LG6 / LG7 ----------------
    lg5, lg6, lg7, diag_scatter = {}, {}, {}, {}
    u100_first = {}
    for vec in S.VECTORS:
        method = S.PK + vec + S.PK_SUFFIX
        counts = ST.keep_counts(docs[vec], 576)
        ctx = S.activate(runner, method, store_root / vec, docs[vec])
        r5, r6, r7 = {}, {}, {}
        ranges, selected = [], []
        for turn in (1, 2, 3):
            factory = lambda t=turn: S.prompt_factory(methods, dialog, t, S.RECOMP, {k: v for k, v in generated.items() if k < t})
            a, da = traced_hit(runner, server, ctx, factory, image_id, layers, v0, order, counts)
            b, _ = traced_hit(runner, server, ctx, factory, image_id, layers, v0, order, counts)
            assert da["prompt"] == prompts[turn]
            suffix = torch.tensor(da["suffix_input_ids"])
            la, lb = a["traced"]["logits"], b["traced"]["logits"]
            dense = S.reference_run(runner, S.dense_reference_cache(runner, layers, v0, order, counts), S.independent_visible(P, v0, counts), suffix, P)
            r5[turn] = {"masks": a["traced"]["masks"]["status"], "visible_rows_equal_k_l": a["traced"]["masks"]["visible_rows_equal_k_l"],
                        "cache_bits": a["traced"]["bits"]["status"],
                        "first_logits_bitwise_ssd_vs_in_memory_reference": bool(torch.equal(la[0], dense["logits"][0])),
                        "first_logits_bitwise_ssd_repeat": bool(torch.equal(la[0], lb[0])),
                        "all_step_logits_bitwise_vs_reference": len(la) == len(dense["logits"]) and all(torch.equal(x, y) for x, y in zip(la, dense["logits"])),
                        "tokens_equal_reference": a["generated_token_ids"] == dense["generated_token_ids"], "tokens_equal_repeat": a["generated_token_ids"] == b["generated_token_ids"],
                        "tokens": a["generated_token_ids"], "reference_mask_hook_calls": dense["mask_hook_calls"], "k_per_layer_equal": a["k_per_layer"] == counts}
            ok5 = (r5[turn]["masks"] == "PASS" and r5[turn]["visible_rows_equal_k_l"] and r5[turn]["cache_bits"] == "PASS" and r5[turn]["first_logits_bitwise_ssd_vs_in_memory_reference"]
                   and r5[turn]["first_logits_bitwise_ssd_repeat"] and r5[turn]["tokens_equal_reference"] and r5[turn]["tokens_equal_repeat"] and r5[turn]["k_per_layer_equal"]
                   and dense["mask_hook_calls"] == 32 * len(dense["generated_token_ids"]))
            r5[turn]["status"] = "PASS" if ok5 else "FAIL"
            assert ok5, (vec, turn, r5[turn])
            # LG6: full-length reference (all 576 rows real) with the per-layer visible set applied by masks only
            full = S.reference_run(runner, S.full_length_reference_cache(runner, layers, v0, order), S.independent_visible(P, v0, counts), suffix, P)
            orig = S.reference_run(runner, S.full_length_reference_cache(runner, layers, v0, None), S.original_visible(P, v0, order, counts), suffix, P)
            close = bool(torch.allclose(la[0], full["logits"][0], atol=S.ATOL, rtol=S.RTOL))
            r6[turn] = {"first_token_logits_within_1e-4": close, "max_abs_delta": float((la[0] - full["logits"][0]).abs().max()),
                        "first_token_equal": a["first_token_id"] == full["first_token_id"], "tokens_equal": a["generated_token_ids"] == full["generated_token_ids"],
                        "original_coordinate_DIAGNOSTIC": {"max_abs_delta": float((la[0] - orig["logits"][0]).abs().max()),
                                                           "within_1e-4": bool(torch.allclose(la[0], orig["logits"][0], atol=S.ATOL, rtol=S.RTOL)),
                                                           "first_token_equal": a["first_token_id"] == orig["first_token_id"],
                                                           "tokens_equal": a["generated_token_ids"] == orig["generated_token_ids"]}}
            r6[turn]["status"] = "PASS" if close and r6[turn]["first_token_equal"] and r6[turn]["tokens_equal"] else "FAIL"
            assert r6[turn]["status"] == "PASS", (vec, turn, r6[turn])
            if vec == "U100":
                top2 = torch.topk(la[0], 2)
                u100_first[turn] = {"u100_first": a["first_token_id"], "recompute_first": rc[turn]["first"], "equal": a["first_token_id"] == rc[turn]["first"],
                                    "u100_tokens": a["generated_token_ids"], "recompute_tokens": rc[turn]["tokens"],
                                    "u100_top2_logit_margin": float(top2.values[0] - top2.values[1]), "u100_top2_ids": top2.indices.tolist()}
            # LG7
            tr = a["os_pread_trace"]
            r7[turn] = {"preadv_calls": a["preadv_calls"], "pread_calls": a["pread_calls"], "returned_bytes": a["io"]["bytes"],
                        "file_bytes": (store_root / vec / "kv_packed.bin").stat().st_size, "vision_calls": a["vision_forward_count"],
                        "score_calls": a["scoring_calls"] + a["query_score_calls"] + a["static_score_calls"],
                        "range": [tr[0]["offset"], tr[0]["requested"], tr[0]["returned"]] if tr else None,
                        "path_is_packed_file": bool(tr) and Path(tr[0]["path"]).resolve() == (store_root / vec / "kv_packed.bin").resolve()}
            ok7 = (r7[turn]["preadv_calls"] == 1 and r7[turn]["pread_calls"] == 0 and len(tr) == 1 and tr[0]["syscall"] == "preadv"
                   and r7[turn]["returned_bytes"] == r7[turn]["file_bytes"] == S.EXPECTED_BYTES[vec] and r7[turn]["vision_calls"] == 0 and r7[turn]["score_calls"] == 0
                   and r7[turn]["range"] == [0, S.EXPECTED_BYTES[vec], S.EXPECTED_BYTES[vec]] and r7[turn]["path_is_packed_file"])
            r7[turn]["status"] = "PASS" if ok7 else "FAIL"
            assert ok7, (vec, turn, r7[turn])
            ranges.append(r7[turn]["range"]); selected.append(a["selected_original_ids"])
            if turn == 3:
                diag_scatter[vec] = dict(scatter_breakdown(runner, ctx, counts), production_scatter_ms=a["scatter_ms"], production_read_ms=a["packed_read_ms"],
                                         production_prefill_ms=a["prefill_ms"], production_ttft_ms=a["end_to_end_ttft_ms"])
            del a, b, dense, full, orig
        inv = all(x == ranges[0] for x in ranges) and all(x == selected[0] for x in selected) and selected[0] == order[:max(counts)]
        assert inv, (vec, "question invariance")
        lg5[vec] = dict(r5, status="PASS")
        lg6[vec] = dict(r6, status="PASS")
        lg7[vec] = dict(r7, status="PASS", read_range_and_selected_ids_invariant_over_questions=inv, activation_ms=ctx.activation["outer_wall_ms"])
        ctx.close()
        del ctx
        gc.collect()
    out["LG5"] = dict(lg5, status="PASS")
    out["LG6"] = dict(lg6, status="PASS", u100_vs_recompute_first_token=u100_first)
    out["LG7"] = dict(lg7, status="PASS")
    out["scatter_breakdown_DIAGNOSTIC"] = diag_scatter
    out["recompute_ttft_ms_DIAGNOSTIC"] = {t: rc[t]["ttft_ms"] for t in (1, 2, 3)}
    del layers, cap
    gc.collect(); torch.cuda.empty_cache()
    guarded_rmtree(store_root, run / "gates" / "stores")      # run-local gate stores, removed once the receipts are recorded
    out["stores_deleted_after_record"] = not store_root.exists()
    return out


def stage_gpu(run: Path):
    import numpy as np
    import random
    from mmimpress.llava15.runner import load_runner
    from mmimpress.serve import BIAS, Server
    H = S.lifecycle()
    cfg = read(run / "config.json")
    manifest = read(run / "manifest.json")
    docs = S.load_vectors(run)
    assert read(run / "gates" / "cpu.json")["status"] == "PASS" and read(run / "gates" / "cpu.json")["config_sha256"] == sha(run / "config.json"), "CPU gates first"
    H.assert_gpu_exclusive()
    torch.set_grad_enabled(False)
    random.seed(S.SEED); np.random.seed(S.SEED); torch.manual_seed(S.SEED); torch.set_num_threads(cfg["cpu_threads"])
    runner = load_runner(S.SEED)
    server = Server(runner, max_new_tokens=16)
    gate = {"status": "RUNNING", **bind(run), "atol": S.ATOL, "rtol": S.RTOL, "model": runner.describe(), "warmup": S.warmup(runner, server),
            "fixture": manifest["gate_fixture"], "fixture_requests": len(manifest["gate_fixture"]) * 3, "images": []}
    stores = run / "gates" / "stores"
    if stores.exists():
        assert not any(stores.rglob("*.bin")), "stale gate stores"
    path = run / "gates" / "gpu.json"
    try:
        for i, fx in enumerate(manifest["gate_fixture"]):
            case = gpu_image(runner, server, run, docs, manifest, fx, cfg["methods"], first=(i == 0))
            gate["images"].append(case)
            write(path, gate)
            print(json.dumps({"fixture": i + 1, "image": fx["image_id"], **{g: case[g]["status"] for g in ("LG1", "LG2", "LG3", "LG4", "LG5", "LG6", "LG7")}}), flush=True)
        # ---- U100 vs recompute first-token agreement on the 30 fixture requests (LG6)
        pairs = [(c["image_id"], t, v) for c in gate["images"] for t, v in c["LG6"]["u100_vs_recompute_first_token"].items()]
        agree = sum(1 for _, _, v in pairs if v["equal"])
        mism = [{"image_id": i, "turn": t, **v} for i, t, v in pairs if not v["equal"]]
        gate["LG6_u100_vs_recompute"] = {"requests": len(pairs), "first_token_equal": agree, "rate": agree / len(pairs), "threshold": 0.95, "mismatches": mism,
                                         "full_answer_token_equal": sum(1 for _, _, v in pairs if v["u100_tokens"] == v["recompute_tokens"]),
                                         "mismatch_cause_note": "U100 serves the Turn-1 prefix KV (FP16 round trip) and prefills only the suffix, recompute prefills the whole sequence; "
                                                                "BF16 results depend on the tensor shape, so near-tied first tokens can flip (the top-2 logit margin of each mismatch is recorded)",
                                         "status": "PASS" if len(pairs) == 30 and agree / len(pairs) >= 0.95 else "FAIL"}
        for g in ("LG1", "LG2", "LG3", "LG4", "LG5", "LG6", "LG7"):
            gate[g] = "PASS" if all(c[g]["status"] == "PASS" for c in gate["images"]) and len(gate["images"]) == S.FIXTURE_IMAGES else "FAIL"
        if gate["LG6_u100_vs_recompute"]["status"] != "PASS":
            gate["LG6"] = "FAIL"
        # aggregates of the diagnostics
        gate["scatter_breakdown_DIAGNOSTIC_mean_ms"] = {
            vec: {k: float(np.mean([c["scatter_breakdown_DIAGNOSTIC"][vec][k] for c in gate["images"]])) for k in
                  ("host_fp16_to_bf16_ms", "h2d_ms", "gpu_index_write_ms", "mask_ms", "production_scatter_ms", "production_read_ms", "production_prefill_ms", "production_ttft_ms")}
            for vec in S.VECTORS}
        gate["scatter_breakdown_DIAGNOSTIC_note"] = ("instrumented replica of PrefixCache.write on the gate fixture (10 images, turn 3), extra synchronisations, bitwise equal result; "
                                                     "not the timed HOLD path and not a HOLD measurement")
        gate["scatter_replica_bitwise_equal_all"] = all(c["scatter_breakdown_DIAGNOSTIC"][v]["replica_bitwise_equal_to_production_scatter"] for c in gate["images"] for v in S.VECTORS)
        lit = [c["LG3"]["literal_shorter_prefix_forward_DIAGNOSTIC"] for c in gate["images"]]
        gate["LG3_literal_prefix_only_DIAGNOSTIC"] = {"images": len(lit), "layer0_bitwise_equal_all": all(x["layer0_bitwise_equal"] for x in lit),
                                                      "layers_not_bitwise_equal_min_max": [min(x["layers_not_bitwise_equal"] for x in lit), max(x["layers_not_bitwise_equal"] for x in lit)],
                                                      "max_rel_l2": max(x["max_rel_l2"] for x in lit)}
        orig = [c["LG6"][v][t]["original_coordinate_DIAGNOSTIC"] for c in gate["images"] for v in S.VECTORS for t in (1, 2, 3)]
        gate["LG6_original_coordinate_DIAGNOSTIC"] = {"requests": len(orig), "within_1e-4": sum(x["within_1e-4"] for x in orig), "max_abs_delta": max(x["max_abs_delta"] for x in orig),
                                                      "first_token_equal": sum(x["first_token_equal"] for x in orig), "tokens_equal": sum(x["tokens_equal"] for x in orig),
                                                      "note": "same visible sets expressed in the model's original token order (keys permuted relative to production); "
                                                              "differences are BF16 summation-order rounding; recorded, not gated"}
        gate["status"] = "PASS" if all(gate[g] == "PASS" for g in ("LG1", "LG2", "LG3", "LG4", "LG5", "LG6", "LG7")) else "FAIL"
    except Exception:
        gate["status"] = "FAIL"
        gate["error"] = traceback.format_exc()
        print(gate["error"], flush=True)
        raise
    finally:
        BIAS.clear()
        write(path, gate)
    print(json.dumps({"gpu_gates": gate["status"], **{g: gate[g] for g in ("LG1", "LG2", "LG3", "LG4", "LG5", "LG6", "LG7")},
                      "u100_vs_recompute": gate["LG6_u100_vs_recompute"]["rate"]}), flush=True)
    assert gate["status"] == "PASS"


# =========================================================================== summary (LG10)
def stage_summary(run: Path):
    E = S.load_script("_llava15_hold", "179_eval_llava15_mt_gqa_hold.py")
    config, manifest = E.verify_freeze(run)
    cpu, gpu = read(run / "gates" / "cpu.json"), read(run / "gates" / "gpu.json")
    here = sha(run / "config.json")
    assert cpu["config_sha256"] == here == gpu["config_sha256"], "gates were run for another freeze"
    val = read(run / "integration" / "validation.json")
    rows, statuses = 0, {}
    for cp in sorted((run / "integration" / "images").glob("*/COMMITTED.json")):
        doc = read(cp)
        folder = cp.parent / doc["attempt_id"]
        assert sha(folder / "raw.jsonl") == doc["raw_sha256"] and doc["config_sha256"] == here
        receipt = E.audit_image_rows(E.load_jsonl(folder / "raw.jsonl"), manifest, config, "integration", cp.parent.name, folder)
        statuses[cp.parent.name] = receipt["status"]
        rows += receipt["requests"]
    by_arm = {}
    for cp in sorted((run / "integration" / "images").glob("*/COMMITTED.json")):
        for r in E.load_jsonl(cp.parent / read(cp)["attempt_id"] / "raw.jsonl"):
            by_arm.setdefault(r["method_id"], 0)
            by_arm[r["method_id"]] += 1
    leftovers = [str(p) for p in (run / "scratch").rglob("*") if p.is_file()] if (run / "scratch").exists() else []
    lg10 = {"status": "PASS" if val["status"] == "PASS" and val["requests"] == rows == S.EXPECT["integration"] and all(v == "PASS" for v in statuses.values())
                                 and set(by_arm) == set(S.METHODS) and all(v == 12 for v in by_arm.values()) and not leftovers and val["config_sha256"] == here else "FAIL",
            "requests": rows, "expected": S.EXPECT["integration"], "requests_per_arm": by_arm, "independent_reaudit": statuses, "validation": val,
            "scratch_files_left": leftovers, "history": "method-local generated, 3 turns, 5 arms, smoke dialogue ids"}
    gates = {"LG1": gpu["LG1"], "LG2": gpu["LG2"], "LG3": gpu["LG3"], "LG4": gpu["LG4"], "LG5": gpu["LG5"], "LG6": gpu["LG6"], "LG7": gpu["LG7"],
             "LG8": cpu["LG8"]["status"], "LG9": cpu["LG9"]["status"], "LG10": lg10["status"]}
    summary = {"status": "PASS" if all(v == "PASS" for v in gates.values()) else "FAIL", "config_sha256": here, "contract_sha256": config["contract_sha256"],
               "gates": gates, "LG10": lg10, "fixture_images": len(gpu["images"]), "u100_vs_recompute_first_token_rate": gpu["LG6_u100_vs_recompute"]["rate"],
               "cpu_tests": cpu["LG9"]["ran"], "glm_preexisting_errors": cpu["LG9"]["glm_errors"], "at_unix": time.time(),
               "amendments": sorted(p.name for p in run.glob("freeze_amendment_*.json"))}
    write(run / "gates" / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "gates", "config_sha256")}), flush=True)
    assert summary["status"] == "PASS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--stage", choices=("cpu", "gpu", "summary"), required=True)
    a = ap.parse_args()
    run = a.run.resolve()
    assert run.parent == ROOT / "runs" and run.name.startswith("llava15_mt_gqa_")
    (run / "gates").mkdir(parents=True, exist_ok=True)
    {"cpu": stage_cpu, "gpu": stage_gpu, "summary": stage_summary}[a.stage](run)


if __name__ == "__main__":
    main()

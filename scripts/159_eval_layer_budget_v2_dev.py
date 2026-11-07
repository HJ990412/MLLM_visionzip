#!/usr/bin/env python3
"""Phase 3 (v2) — DEV evaluation of packed vs unpacked serving and the
prior-work allocation vectors (contract v2, A5).

Frozen GQA 40 x questions[4:10] workload of scripts/89/106/155 (T1 =
questions[4], five cache hits).  One executed normal full-pixel Turn-1
inference per image is the source of every store; every arm is served from
its own real store (raster for FullLoad, prefix25 for U_unpacked,
prefix25_layerwise for A1-2_unpacked, packed for the seven vectors).  Latency
arms are executed three times per hit (cold each time; outputs must agree);
accuracy uses repeat 0.  Stores are deleted after the image's receipts.

Stages: --freeze ; --gpu smoke[,gqa] ; --analyze
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
import random
import shutil
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
from PIL import Image

from mmimpress.dataset import exact_score
from mmimpress.layer_budget import load_budget_document
from mmimpress.prefix25 import sha
from mmimpress.recompute_kv25 import guarded_rmtree, paired_bootstrap

REF_RUN = ROOT / "runs/prefix25_persistence_20261001T050113Z"
REF_GQA_SHA = "40e23fa7aeb22495717fc7eede1227de3a6b4811fde022d9cda71ef73c5e3895"
INDEX_SHA = "514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a"
CONTRACT = ROOT / "docs/layer_budget_kv25_v2_contract.md"
VECTORS = ["U", "A1-2", "VLC_offline", "SA_offline", "SA-2g_offline", "PYR", "PYR-beta"]
PACKED = [f"{v}_packed" for v in VECTORS]
UNPACKED = ["U_unpacked", "A1-2_unpacked"]
DIAG = ["A1-2_packed_directcopy"]
ARMS = ["recompute", "fullload"] + UNPACKED + PACKED + DIAG
ACCURACY_ARMS = ["recompute", "fullload"] + PACKED
LATENCY_ARMS = ["U_unpacked", "U_packed", "A1-2_unpacked", "A1-2_packed", "VLC_offline_packed", "SA_offline_packed", "PYR_packed", "A1-2_packed_directcopy"]
REPEATS = 3
SELECTED = "A1-2"
SEED = 1234
MIN_FREE = 40 * 2 ** 30
GATES = ("cpu_tests.json", "g3_budget_invariance.json", "gpu_gates.json", "g8_vector_reproducibility.json", "g9_io_microbench.json")


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


P = load("prefix25_persistence_existing", "106_prefix25_persistence.py")
G = load("layer_budget_gates_v1", "154_validate_layer_budget_kv25.py")
S = load("packed_support_v2", "packed_support_v2.py")
write, append, exclusive = P.write, P.append, P.exclusive
import re
QTYPE = (
    ("yes/no", re.compile(r"^(is|are|do|does|was|were|has|have|can|could|did|will|would)\b", re.I)),
    ("color", re.compile(r"\bcolou?r\b", re.I)),
    ("count", re.compile(r"\bhow many\b|\bnumber of\b", re.I)),
    ("spatial", re.compile(r"\b(left|right|above|below|behind|front|under|near|beside|top|bottom|side|between|next to)\b", re.I)),
    ("material/attribute", re.compile(r"\b(material|made of|shape|size|texture|large|small|tall|short|thin|thick)\b", re.I)),
    ("object", re.compile(r"^(what|which|who)\b", re.I)),
)


def qtype(question):          # identical to scripts/155 (v1 DEV question types)
    for name, pattern in QTYPE:
        if pattern.search(question):
            return name
    return "other"


def capacity():
    free = shutil.disk_usage(ROOT).free
    if free < MIN_FREE:
        raise RuntimeError(f"disk floor: {free / 2 ** 30:.2f} GiB free < 40 GiB")
    return free


def store_of(arm):
    """(store key, vector id, kind, directcopy) for an arm."""
    if arm == "fullload":
        return "fullload", None, "raster", False
    if arm in ("recompute", "T1_source"):
        return None, None, "none", False
    if arm == "U_unpacked":
        return "U_unpacked", "U", "prefix25", False
    if arm == "A1-2_unpacked":
        return "A1-2_unpacked", "A1-2", "prefix25_layerwise", False
    if arm.endswith("_packed_directcopy"):
        v = arm[:-len("_packed_directcopy")]
        return f"{v}_packed", v, "packed", True
    assert arm.endswith("_packed"), arm
    v = arm[:-len("_packed")]
    return arm, v, "packed", False


STORE_KEYS = ["fullload", "U_unpacked", "A1-2_unpacked"] + PACKED


# ------------------------------------------------------------------ freeze
def freeze(run):
    assert sha(REF_RUN / "gqa_manifest.json") == REF_GQA_SHA and sha(ROOT / "data/index.json") == INDEX_SHA
    gqa = json.loads((REF_RUN / "gqa_manifest.json").read_text())
    rows = gqa["images"]
    assert len(rows) == 40 and len({r["image_id"] for r in rows}) == 40
    index = json.loads((ROOT / "data/index.json").read_text())
    for r in rows:
        assert sha(r["image_path"]) == r["image_sha256"] and len(r["turns"]) == 6
    assert [[str(q["question_id"]) for q in x["questions"][4:10]] for x in index[:40]] == [[str(t["question_id"]) for t in x["turns"]] for x in rows]
    if not (run / "gqa_manifest.json").exists():
        shutil.copy2(REF_RUN / "gqa_manifest.json", run / "gqa_manifest.json")
    smoke = json.loads(json.dumps(gqa))
    smoke["images"] = smoke["images"][:2]
    for r in smoke["images"]:
        r["turns"] = r["turns"][:3]
    if not (run / "smoke_manifest.json").exists():
        write(run / "smoke_manifest.json", smoke)
    config = json.loads((run / "config.json").read_text())
    vectors = {a: load_budget_document(run / "phase2" / f"layer_budget_llava_{a}.json") for a in VECTORS}
    assert vectors["A1-2"]["layer_budget_sha256"] == config["selected_ratio_sha256"]
    for gate in GATES:
        doc = json.loads((run / "gates" / gate).read_text())
        assert doc["status"] == "PASS", f"gate not PASS: {gate}"
    dev = {"arms": ARMS, "accuracy_arms": ACCURACY_ARMS, "latency_arms": LATENCY_ARMS, "repeats": REPEATS, "selected": SELECTED,
           "vectors_sha256": {a: v["layer_budget_sha256"] for a, v in vectors.items()},
           "store_policy": {"fullload": "raster(full)", "U_unpacked": "prefix25 (existing writer/serving = v1 ours_prefixstore_kv25)",
                            "A1-2_unpacked": "prefix25_layerwise (v1)", **{f"{v}_packed": "prefix25_packed" for v in VECTORS},
                            "A1-2_packed_directcopy": "served from the A1-2_packed store with slice copy"},
           "t1_policy": "one executed normal full-pixel T1 per image (saliency + decoder-hidden capture); T1 row recorded per arm as shared",
           "workload_files": {n: sha(run / f"{n}_manifest.json") for n in ("gqa", "smoke")},
           "contract_sha256": sha(CONTRACT), "min_free_bytes": MIN_FREE, "bootstrap": {"unit": "image", "resamples": 10000, "seed": SEED},
           "gates": {g: sha(run / "gates" / g) for g in GATES}}
    assert dev["contract_sha256"] == config["contract_sha256"], "contract changed since config"
    write(run / "dev_config.json", dev)
    paths = [p for d in ["mmimpress", "tests"] for p in (ROOT / d).rglob("*.py") if "__pycache__" not in p.parts]
    paths += [ROOT / "scripts" / n for n in ("106_prefix25_persistence.py", "89_eval_llava_kv25.py", "90_validate_llava_kv25.py", "49_eval_query_aware_baseline.py",
                                              "154_validate_layer_budget_kv25.py", "155_eval_layer_budget_kv25.py", "packed_support_v2.py",
                                              "158_allocate_prior_work_vectors.py", "159_eval_layer_budget_v2_dev.py", "161_validate_packed_store_v2.py")]
    paths += [CONTRACT, run / "gqa_manifest.json", run / "smoke_manifest.json", run / "dev_config.json", run / "config.json"]
    paths += sorted((run / "phase2").glob("layer_budget_llava_*.json"))
    write(run / "gpu_freeze_dev.json", {"files": {str(p.relative_to(ROOT)): sha(p) for p in paths}})
    print("FROZEN DEV", run, flush=True)


# -------------------------------------------------------------- experiment
class Dev(P.Experiment):
    def __init__(self, run):
        super().__init__(run, "llava")
        self.docs = {a: load_budget_document(run / "phase2" / f"layer_budget_llava_{a}.json") for a in VECTORS}
        self.stores = self.base / "dev_stores"

    def t1(self, image, turn, phase):
        from mmimpress.piggyback import VisionForwardCapture, DecoderVisualHiddenCapture
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
        with Image.open(image["image_path"]) as src:
            pil = src.convert("RGB")
        decode_ms = (time.perf_counter() - t0) * 1e3
        cap = VisionForwardCapture(self.runner, capture_saliency=True)
        with cap:
            enc = self.runner.encode_prompt(pil, self.prompt(turn["question"], [], phase))
            vs, vn = self.runner.visual_span(enc["input_ids"])
            hidden = DecoderVisualHiddenCapture(self.runner, vs, vn)
            with hidden:
                value = self.server.recompute(self.runner.to_device(enc), return_past_key_values=True)
        value["ttft_ms"] = (value["first_token_at_s"] - t0) * 1e3
        value["request_e2e_ms"] = (time.perf_counter() - t0) * 1e3
        value["prediction"] = value["answer"]; value["vision_calls"] = cap.call_count
        value["online_query_score_calls"] = 0; value["capture_stats"] = cap.stats()
        value["image_file_decode_ms"] = decode_ms
        value["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        value["actual_io"] = {"bytes": 0, "preads": 0, "calls": []}
        assert cap.call_count == 1
        diag = {"cache": value.pop("captured_past_key_values"), "enc": enc, "vision": cap, "hidden": hidden}
        return value, diag

    def persist_store(self, capture, image, key, path):
        capacity()
        if key == "fullload":
            return self.persist(capture, path, image, "fullload")
        if key == "U_unpacked":
            return self.persist(capture, path, image, "B")
        if key == "A1-2_unpacked":
            return G.persist_layerwise(self, capture, path, image, self.docs["A1-2"])
        assert key.endswith("_packed")
        return S.persist_packed(self, capture, path, image, self.docs[key[:-len("_packed")]])

    def activate_store(self, key, path, image):
        if key.endswith("_packed"):
            return S.activate_packed(self, path)
        return self.activate(path, image, "fullload" if key == "fullload" else "B")

    def hit_arm(self, ctx, path, image, turn, arm, phase):
        key, vec, kind, directcopy = store_of(arm)
        try:
            if kind == "packed":
                return S.packed_hit(self, ctx, path, self.prompt(turn["question"], [], phase), logits=False, directcopy=directcopy)
            return self.hit(ctx, path, image, turn, [], "fullload" if arm == "fullload" else "B", phase)
        finally:
            # Release the request's dense GPU prefix cache (PrefixCache.k/v keep the last request's
            # tensors alive, ~1.1 GB per context; 10 live contexts per image exhausted the 24 GB GPU).
            ctx.cache.k = None; ctx.cache.v = None

    def row(self, logical, phase, image, arm, ti, turn, value, repeat=0, executed=True, shared_from=None):
        keep = ["prediction", "first_token_id", "generated_token_ids", "ttft_ms", "request_e2e_ms", "prefill_ms", "decode_ms",
                "selected_original_ids", "k_target", "k_per_layer", "k_total", "k_uniform_equivalent", "N_content", "N_structural",
                "keep_count_per_layer", "normal_kv_read_bytes", "separator_read_bytes", "normal_kv_preads", "separator_preads",
                "planned_normal_read_bytes_per_layer", "budget_policy", "layer_budget_id", "layer_budget_sha256",
                "peak_gpu_allocated_bytes", "vision_calls", "online_query_score_calls", "query_score_calls", "static_score_calls",
                "chunk_io_ms", "scatter_ms", "select_ms", "logical_content_retention_ratio", "packed_read_ms", "read_calls", "read_syscall",
                "directcopy", "store_format"]
        key, vec, kind, directcopy = store_of(arm)
        io = value["actual_io"]
        return {"logical_request_id": logical, "execution_id": str(uuid.uuid4()), "status": "OK", "phase": phase, "arm": arm, "repeat": repeat,
                "store_kind": kind, "vector": vec, "image_id": image["image_id"], "turn": ti + 1, "question_id": str(turn["question_id"]),
                "question": turn["question"], "question_type": qtype(turn["question"]), "gold": turn["gold"], "cache_hit": ti > 0 and arm != "recompute",
                "executed": executed, "shared_t1_from": shared_from, "accuracy": exact_score(value["prediction"], [turn["gold"]]),
                "read_bytes": io["bytes"], "preads": io["preads"], "pread_calls": io.get("pread_calls", io["preads"]), "preadv_calls": io.get("preadv_calls", 0),
                **{k: value.get(k) for k in keep}}

    def pilot(self, phase):
        manifest = json.loads((self.run / f"{phase}_manifest.json").read_text())
        out = self.base / phase
        out.mkdir(exist_ok=False)
        raw = out / "raw_requests.jsonl"
        persist_file = out / "persistence.jsonl"
        n_arms = len(ARMS)
        for ix, image in enumerate(manifest["images"]):
            exclusive(); capacity()
            t_img = time.perf_counter()
            contexts, paths, receipts = {}, {}, {}
            turns = image["turns"]
            value, capture = self.t1(image, turns[0], phase)
            source_id = f"llava/{phase}/{image['image_id']}/T1/1"
            append(raw, self.row(source_id, phase, image, "T1_source", 0, turns[0], value))
            for arm in ARMS:
                append(raw, self.row(f"llava/{phase}/{image['image_id']}/{arm}/1", phase, image, arm, 0, turns[0], value, executed=False, shared_from=source_id))
            for key in STORE_KEYS:
                path = self.stores / phase / image["image_id"] / key
                receipt = self.persist_store(capture, image, key, path)
                receipt.update(phase=phase, store_key=key)
                paths[key] = path; receipts[key] = receipt
            del capture
            gc.collect(); torch.cuda.empty_cache()
            for key in STORE_KEYS:
                ctx, act = self.activate_store(key, paths[key], image)
                receipts[key]["activation"] = act
                contexts[key] = ctx
                append(persist_file, receipts[key])
            orders = {k: receipts[k]["metadata"]["order"] for k in STORE_KEYS if k != "fullload"}
            assert len({json.dumps(o) for o in orders.values()}) == 1
            hits = {}
            for ti in range(1, len(turns)):
                turn = turns[ti]
                order = ARMS[(ix + ti) % n_arms:] + ARMS[:(ix + ti) % n_arms]
                for rep in range(REPEATS):
                    for arm in order:
                        if rep > 0 and arm not in LATENCY_ARMS:
                            continue
                        exclusive()
                        logical = f"llava/{phase}/{image['image_id']}/{arm}/{ti + 1}/r{rep}"
                        try:
                            if arm == "recompute":
                                v, cap = self.pixel(image, turn, [], "recompute", phase)
                                del cap
                                v["actual_io"] = {"bytes": 0, "preads": 0, "calls": []}
                            else:
                                key = store_of(arm)[0]
                                v = self.hit_arm(contexts[key], paths[key], image, turn, arm, phase)
                            hits[(arm, ti, rep)] = v
                            append(raw, self.row(logical, phase, image, arm, ti, turn, v, repeat=rep))
                        except Exception:
                            append(raw, {"logical_request_id": logical, "status": "FAIL", "error": traceback.format_exc()})
                            raise
                        if rep > 0:
                            a, b = hits[(arm, ti, 0)], v
                            assert a["generated_token_ids"] == b["generated_token_ids"] and a["prediction"] == b["prediction"], (arm, ti, rep)
            # per-image G3 (bytes identical across all stored arms) and G6 (packed == unpacked outputs)
            g3, g6 = {}, {}
            for ti in range(1, len(turns)):
                ub, up = hits[("U_unpacked", ti, 0)]["actual_io"]["bytes"], hits[("U_unpacked", ti, 0)]["actual_io"]["preads"]
                assert up == 65
                for arm in PACKED + ["A1-2_unpacked"] + DIAG:
                    b, p = hits[(arm, ti, 0)]["actual_io"]["bytes"], hits[(arm, ti, 0)]["actual_io"]["preads"]
                    ratio = b / ub
                    assert abs(ratio - 1.0) <= 0.02, (arm, ti, b, ub)
                    assert p == (65 if arm == "A1-2_unpacked" else 1), (arm, p)
                    g3[f"{arm}/{ti + 1}"] = {"bytes": b, "uniform_unpacked_bytes": ub, "ratio": ratio, "preads": p}
                for packed, unpacked in (("U_packed", "U_unpacked"), ("A1-2_packed", "A1-2_unpacked"), ("A1-2_packed_directcopy", "A1-2_unpacked")):
                    a, b = hits[(packed, ti, 0)], hits[(unpacked, ti, 0)]
                    same = a["first_token_id"] == b["first_token_id"] and a["generated_token_ids"] == b["generated_token_ids"] and a["prediction"] == b["prediction"]
                    kp = a["k_per_layer"] == (b.get("k_per_layer") or [b["k_target"]] * 32)   # prefix25 arm reports one k
                    g6[f"{packed}/{ti + 1}"] = {"outputs_identical": same, "k_per_layer_identical": kp}
                    assert same and kp, (packed, ti)
            append(self.base / "g3_dev_bytes.jsonl", {"phase": phase, "image_id": image["image_id"], "checks": g3, "max_abs_ratio_deviation": max(abs(x["ratio"] - 1) for x in g3.values())})
            append(self.base / "g6_dev_outputs.jsonl", {"phase": phase, "image_id": image["image_id"], "checks": g6, "all_identical": all(v["outputs_identical"] for v in g6.values())})
            for ctx in contexts.values():
                ctx.close()
            contexts.clear(); self.close_contexts()
            for key, path in paths.items():
                append(self.base / "cleanup.jsonl", {"stage": "before", "path": str(path), "store_key": key, "sealed_files": len(receipts[key]["files"])})
                guarded_rmtree(path, self.stores)
                append(self.base / "cleanup.jsonl", {"stage": "after", "path": str(path), "exists": path.exists()})
            print(json.dumps({"phase": phase, "image": ix + 1, "of": len(manifest["images"]), "image_id": image["image_id"],
                              "seconds": round(time.perf_counter() - t_img, 1)}), flush=True)
        return {"status": "PASS", "images": len(manifest["images"])}


# ----------------------------------------------------------------- analysis
def analyze(run):
    rows = [json.loads(l) for l in (run / "llava/gqa/raw_requests.jsonl").read_text().splitlines()]
    assert all(r["status"] == "OK" for r in rows)
    res = ROOT / "results" / run.name / "dev"
    res.mkdir(parents=True, exist_ok=True)
    by = {}
    for r in rows:
        if r["arm"] == "T1_source":
            continue
        by.setdefault(r["arm"], {}).setdefault(r["repeat"], {})[(r["image_id"], r["turn"])] = r
    images = sorted({r["image_id"] for r in rows})
    for arm in ARMS:
        assert len(by[arm][0]) == 240, (arm, len(by[arm][0]))
    hits = {a: {k: v for k, v in by[a][0].items() if k[1] > 1} for a in ARMS}
    assert all(len(h) == 200 for h in hits.values())
    for arm in LATENCY_ARMS:
        assert all(len(by[arm][r]) == 200 for r in range(1, REPEATS))

    def acc(d):
        return float(np.mean([r["accuracy"] for r in d.values()]))

    def pb(a, b):
        per = []
        for img in images:
            ka = [v for k, v in hits[a].items() if k[0] == img]
            kb = [v for k, v in hits[b].items() if k[0] == img]
            per.append((sum(x["accuracy"] for x in ka), sum(x["accuracy"] for x in kb), len(ka)))
        return paired_bootstrap(per, 10000, SEED)

    quality = []
    for arm in ARMS:
        q = {"arm": arm, "store_kind": store_of(arm)[2], "hit_accuracy": acc(hits[arm]), "all_accuracy": acc(by[arm][0]),
             "hit_correct": int(sum(r["accuracy"] for r in hits[arm].values())), "hit_n": 200,
             "first_token_agreement_fullload_hits": float(np.mean([hits[arm][k]["first_token_id"] == hits["fullload"][k]["first_token_id"] for k in hits[arm]])),
             "prediction_agreement_fullload_hits": float(np.mean([hits[arm][k]["prediction"] == hits["fullload"][k]["prediction"] for k in hits[arm]]))}
        for ref in ("A1-2_packed", "U_packed", "fullload", "recompute"):
            if ref != arm:
                d = pb(arm, ref)
                q[f"delta_pp_vs_{ref}"] = 100 * d["mean"]; q[f"ci95_lo_vs_{ref}"] = 100 * d["ci_low"]; q[f"ci95_hi_vs_{ref}"] = 100 * d["ci_high"]
        quality.append(q)
    latency = []
    for arm in ARMS:
        reps = range(REPEATS) if arm in LATENCY_ARMS else [0]
        allh = [v for r in reps for k, v in by[arm][r].items() if k[1] > 1]
        t = np.array([r["ttft_ms"] for r in allh]); e = np.array([r["request_e2e_ms"] for r in allh])
        b = np.array([r["read_bytes"] for r in allh]); p = np.array([r["preads"] for r in allh])
        per_rep = [float(np.mean([v["ttft_ms"] for k, v in by[arm][r].items() if k[1] > 1])) for r in reps]
        entry = {"arm": arm, "store_kind": store_of(arm)[2], "repeats": len(list(reps)), "n_hits": len(allh),
                 "ttft_mean_ms": t.mean(), "ttft_p50_ms": np.median(t), "ttft_p95_ms": np.percentile(t, 95),
                 "ttft_mean_sd_across_repeats_ms": float(np.std(per_rep, ddof=1)) if len(per_rep) > 1 else 0.0, "ttft_mean_per_repeat_ms": per_rep,
                 "e2e_mean_ms": e.mean(), "e2e_p50_ms": np.median(e), "e2e_p95_ms": np.percentile(e, 95),
                 "ssd_mb_per_hit": b.mean() / 1e6, "preads_mean": p.mean(),
                 "pread_calls_mean": float(np.mean([r["pread_calls"] for r in allh])), "preadv_calls_mean": float(np.mean([r["preadv_calls"] for r in allh]))}
        for key in ("chunk_io_ms", "scatter_ms", "packed_read_ms", "prefill_ms"):
            vals = [r[key] for r in allh if r.get(key) is not None]
            entry[f"{key}_mean"] = float(np.mean(vals)) if vals else None
        latency.append(entry)
    lat = {l["arm"]: l for l in latency}
    spread = {a: lat[a]["ttft_mean_ms"] / lat["U_packed"]["ttft_mean_ms"] - 1 for a in PACKED if a != "U_packed"}
    packed_vs_unpacked = {"U": lat["U_packed"]["ttft_mean_ms"] / lat["U_unpacked"]["ttft_mean_ms"], "A1-2": lat["A1-2_packed"]["ttft_mean_ms"] / lat["A1-2_unpacked"]["ttft_mean_ms"],
                          "A1-2_vs_U_unpacked": lat["A1-2_unpacked"]["ttft_mean_ms"] / lat["U_unpacked"]["ttft_mean_ms"],
                          "A1-2_vs_U_packed": lat["A1-2_packed"]["ttft_mean_ms"] / lat["U_packed"]["ttft_mean_ms"],
                          "directcopy_vs_scatter_A1-2": lat["A1-2_packed_directcopy"]["ttft_mean_ms"] / lat["A1-2_packed"]["ttft_mean_ms"]}
    per = [json.loads(l) for l in (run / "llava/gqa/persistence.jsonl").read_text().splitlines()]
    storage = []
    for key in STORE_KEYS:
        rs = [r for r in per if r.get("store_key") == key]
        storage.append({"store_key": key, "images": len(rs), "store_mb_mean": np.mean([r["file_logical_bytes"] for r in rs]) / 1e6,
                        "allocated_mb_mean": np.mean([r["allocated_bytes"] for r in rs]) / 1e6, "persistence_ms_mean": np.mean([r["persistence_ms"] for r in rs]),
                        "persistence_ms_p95": np.percentile([r["persistence_ms"] for r in rs], 95),
                        "native_writer_ssd_write_ms_mean": np.mean([r["timing_ms"]["ssd_write_ms"] for r in rs]) if all("ssd_write_ms" in r["timing_ms"] for r in rs) else None,
                        "packed_sha256_ms_mean": np.mean([r["timing_ms"].get("packed_sha256_ms", 0.0) for r in rs]),
                        "activation_ms_mean": np.mean([r["activation"]["activation_ms"] for r in rs]), "os_write_mb_mean": np.mean([r["os_write_bytes"] for r in rs]) / 1e6})
    err = []
    types = sorted({r["question_type"] for r in rows if r["arm"] == "U_packed"})
    for arm in PACKED + UNPACKED:
        for qt in ["all"] + types:
            keys = [k for k, r in hits[arm].items() if qt == "all" or r["question_type"] == qt]
            if not keys:
                continue
            a = np.array([hits[arm][k]["accuracy"] for k in keys])
            entry = {"arm": arm, "question_type": qt, "n_questions": len(keys), "accuracy": a.mean()}
            for ref in ("fullload", "recompute", "U_packed", "A1-2_packed"):
                b = np.array([hits[ref][k]["accuracy"] for k in keys])
                entry[f"{ref}_accuracy"] = b.mean(); entry[f"delta_vs_{ref}_pp"] = 100 * (a.mean() - b.mean())
                entry[f"both_correct_vs_{ref}"] = int(((a == 1) & (b == 1)).sum()); entry[f"arm_only_vs_{ref}"] = int(((a == 1) & (b == 0)).sum())
                entry[f"{ref}_only"] = int(((a == 0) & (b == 1)).sum()); entry[f"neither_vs_{ref}"] = int(((a == 0) & (b == 0)).sum())
            err.append(entry)
    layers = []
    for arm in PACKED:
        ks = np.array([r["k_per_layer"] for r in hits[arm].values()], dtype=float)
        n = np.array([r["N_content"] for r in hits[arm].values()], dtype=float)
        for li in range(32):
            layers.append({"arm": arm, "layer": li, "k_mean": ks[:, li].mean(), "ratio_mean": (ks[:, li] / n).mean(), "kv_bytes_mean": ks[:, li].mean() * 2 * 32 * 128 * 2})
    g3rows = [json.loads(l) for l in (run / "llava/g3_dev_bytes.jsonl").read_text().splitlines() if '"phase": "gqa"' in l]
    g3 = {"images": len(g3rows), "max_abs_ratio_deviation": max(r["max_abs_ratio_deviation"] for r in g3rows), "tolerance": 0.02, "status": "PASS"}
    g6rows = [json.loads(l) for l in (run / "llava/g6_dev_outputs.jsonl").read_text().splitlines() if '"phase": "gqa"' in l]
    g6 = {"images": len(g6rows), "all_identical": all(r["all_identical"] for r in g6rows), "checks": sum(len(r["checks"]) for r in g6rows), "status": "PASS" if all(r["all_identical"] for r in g6rows) else "FAIL"}
    qmap = {q["arm"]: q for q in quality}
    selection = {"selected": SELECTED, "rule": "fixed before measurement (contract v2 A5); other vectors are comparison columns only",
                 "hit_accuracy": {a: qmap[a]["hit_accuracy"] for a in ACCURACY_ARMS},
                 "best_dev_vector": max(PACKED, key=lambda a: qmap[a]["hit_accuracy"]),
                 "hold_arms": ["recompute", "fullload"] + [f"ours_prefixstore_kv25_{v}_packed" for v in ("U", "A1-2", "VLC_offline", "SA_offline", "PYR")],
                 "vectors_sha256": {a: load_budget_document(run / "phase2" / f"layer_budget_llava_{a}.json")["layer_budget_sha256"] for a in VECTORS}}
    sel_path = run / "dev_selection.json"
    if sel_path.exists():
        assert json.loads(sel_path.read_text()) == selection, "dev_selection.json already frozen with a different content"
    else:
        with sel_path.open("x") as f:
            json.dump(selection, f, indent=1); f.write("\n")

    def csv_write(path, rows_):
        keys = []
        for r in rows_:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows_:
                w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    csv_write(res / "quality.csv", quality); csv_write(res / "latency_io.csv", latency); csv_write(res / "storage.csv", storage)
    csv_write(res / "error_analysis.csv", err); csv_write(res / "layer_counts.csv", layers)
    write(res / "g3_dev_bytes.json", g3); write(res / "g6_dev_outputs.json", g6); write(res / "dev_selection.json", selection)
    write(res / "summary.json", {"quality": quality, "latency_io": latency, "storage": storage, "selection": selection, "g3": g3, "g6": g6,
                                 "packed_ttft_spread_vs_U_packed": spread, "packed_vs_unpacked_ttft_ratio": packed_vs_unpacked,
                                 "n_rows": len(rows), "images": len(images)})
    print(json.dumps({"hit_accuracy": {q["arm"]: round(q["hit_accuracy"], 4) for q in quality},
                      "ttft_mean_ms": {l["arm"]: round(l["ttft_mean_ms"], 1) for l in latency}, "spread": spread, "ratios": packed_vs_unpacked}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--gpu", help="comma list of smoke,gqa")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    run = Path(args.run).resolve()
    assert run.parent == ROOT / "runs" and run.name.startswith("layer_budget_kv25_v2_")
    if args.freeze:
        return freeze(run)
    if args.analyze:
        return analyze(run)
    frozen = json.loads((run / "gpu_freeze_dev.json").read_text())
    for rel, digest in frozen["files"].items():
        assert sha(ROOT / rel) == digest, f"source changed after DEV freeze: {rel}"
    stages = args.gpu.split(",")
    assert set(stages) <= {"smoke", "gqa"}
    exclusive(); torch.set_grad_enabled(False)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)
    status_path = run / "llava" / "dev_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    exp = None
    try:
        exp = Dev(run)
        sample = json.loads((run / "smoke_manifest.json").read_text())["images"][0]
        for _ in range(2):
            v, c = exp.t1(sample, sample["turns"][0], "smoke"); del v, c
        v, c = exp.pixel(sample, sample["turns"][0], [], "recompute", "smoke"); del v, c
        gc.collect(); torch.cuda.empty_cache()
        write(exp.base / f"warmup_dev_{'_'.join(stages)}.json", {"excluded": True, "normal_requests": 3})
        for phase in stages:
            if phase == "gqa":
                assert status.get("SMOKE") == "PASS", "smoke must pass before DEV gqa"
            status[phase.upper()] = "RUNNING"; write(status_path, status)
            exp.pilot(phase)
            status[phase.upper()] = "PASS"; write(status_path, status)
    except Exception:
        err = traceback.format_exc(); print(err, flush=True)
        append(run / "failures.jsonl", {"phase": "dev", "error": err, "utc": time.time(), "stages": stages})
        for k, v in list(status.items()):
            if v == "RUNNING":
                status[k] = "FAIL"
        raise
    finally:
        (run / "llava").mkdir(exist_ok=True); write(status_path, status)
        if exp:
            exp.close_contexts()
        print(status, flush=True)


if __name__ == "__main__":
    main()

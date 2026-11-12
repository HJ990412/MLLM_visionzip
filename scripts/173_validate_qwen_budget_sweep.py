#!/usr/bin/env python3
"""Gates SG1-SG5 for the Qwen2.5-VL retention-ratio sweep (SG6 = 174 --phase integration).

  --stage cpu : SG1 (vectors, k arithmetic N = 1..3000, validator fixtures, no LLaVA-default
                reliance), SG4 (nested selection on all 398 HOLD images) and the complete
                existing CPU test suite
  --stage gpu : SG2 (25% regression vs the PyramidKV-run packed code), SG3 (100% == FullLoad),
                SG5 (I/O), plus an independent in-memory reference / FP64 check of every ratio
                -- on the 10 frozen core pairs, two hit prompts each

Validation samples are correctness fixtures, not a quality holdout.  A failing gate is recorded
and stops the run; thresholds and samples are fixed by docs/qwen_budget_sweep_contract.md.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import qwen_sweep_support as Q
from qwen_sweep_support import ROOT, load_script, read, sha, write

L = 28
ROW = 4 * 128 * 2
FOLLOWUP_QUESTION = "What is visible?"
GPU_GATES = ("SG2", "SG3", "SG5", "SGR")
META_DIFF_ALLOWED = {"format", "store_format", "storage_policy", "layout_description", "order_semantics", "rounding_policy", "layer_budget_id",
                     "layer_budget_r_max", "layer_budget_binding_sha256", "packed_code_revision", "uniform_content_count",
                     "uniform_equivalent_content_count", "retention_fraction", "bytes_metadata_file", "extra"}


def bind(run):
    return {"config_sha256": sha(run / "config.json"), "contract_sha256": read(run / "config.json")["contract_sha256"]}


# ========================================================================= SG1 (CPU)
BUDGET_FUNCS = {"validate_ratios": {"num_layers", "total_ratio", "r_min", "r_max"}, "layer_keep_counts": {"num_layers", "total_ratio", "r_min", "r_max"},
                "validate_keep_counts": {"num_layers", "total_ratio", "r_min", "r_max"}, "total_row_budget": {"num_layers", "total_ratio"},
                "keep_bounds": {"r_min", "r_max"}, "budget_document": {"num_layers", "total_ratio", "r_min", "r_max"},
                "uniform_ratios": {"num_layers", "ratio"}, "clip_redistribute": {"num_layers", "total_ratio", "r_min", "r_max"},
                "front_schedule": {"num_layers", "total_ratio", "r_min", "r_max"}, "load_budget_document": None}


def static_default_scan():
    """Every layer_budget call in the new serving/experiment sources must name the model-specific
    arguments (or expand a frozen kwargs dict); load_budget_document must not be called at all."""
    calls, offenders = [], []
    for rel in Q.NEW_PATHS:
        if not rel.endswith(".py") or rel == Q.REPORT_SCRIPT or rel.startswith("tests/"):   # the report script is post-hoc; the unit tests call the defaults on purpose inside assertRaises
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


def stage_sg1(run, gates):
    import unittest
    from mmimpress.layer_budget import budget_sha256
    from mmimpress.qwen25 import ratio_packed as RP
    out = {"gate": "SG1", **bind(run)}
    docs = Q.load_vectors(run)
    recorded = read(run / "vectors" / "vectors_sha256.json")
    out["vectors"] = {a: {"sha256": docs[a]["layer_budget_sha256"], "expected": Q.RATIO_SHA.get(a, recorded[a]["layer_budget_sha256"]),
                          "expected_source": "prompt" if a in Q.RATIO_SHA else "recorded at Phase 0", "recomputed": budget_sha256(Q.vector(a)),
                          "sum": sum(docs[a]["layer_keep_ratios"]), "expected_sum_28r": 28 * Q.FRACTION[a][0] / Q.FRACTION[a][1],
                          "num_layers": docs[a]["num_layers"], "total_ratio": docs[a]["total_ratio"], "r_min": docs[a]["r_min"], "r_max": docs[a]["r_max"]} for a in Q.ALLOCS}
    ok = all(v["sha256"] == v["expected"] == v["recomputed"] and abs(v["sum"] - v["expected_sum_28r"]) < 1e-9
             and (v["num_layers"], v["r_min"], v["r_max"]) == (28, 0.0, 1.0) and v["total_ratio"] == Q.RATIO[a] for a, v in out["vectors"].items())
    arithmetic = {}
    for a in Q.ALLOCS:
        bad_k, bad_sum, bad_prod = [], [], []
        for n in range(1, 3001):
            counts = Q.keep_counts(docs[a], n)
            k = Q.integer_k(a, n)
            if counts != [k] * 28 or math.ceil(Q.RATIO[a] * n) != k:
                bad_k.append(n)
            if sum(counts) != 28 * k:
                bad_sum.append(n)
            if RP.ratio_keep_counts(docs[a]["layer_keep_ratios"], n, 28, docs[a]["r_min"], docs[a]["r_max"], docs[a]["total_ratio"]) != [k] * 28 \
                    or RP.uniform_keep_count(Q.RATIO[a], n) != k:
                bad_prod.append(n)
        arithmetic[a] = {"integer_fraction": list(Q.FRACTION[a]), "N_range": [1, 3000], "k_mismatches": bad_k, "sum_mismatches": bad_sum, "production_mismatches": bad_prod,
                         "status": "PASS" if not (bad_k or bad_sum or bad_prod) else "FAIL"}
        ok = ok and arithmetic[a]["status"] == "PASS"
    out["k_equals_integer_fraction_N_1_to_3000"] = arithmetic
    hold = sorted(x["N"] for x in read(run / "phase0" / "n_distribution_hold.json")["per_image"])
    out["expected_k_table"] = {str(n): {a: Q.integer_k(a, n) for a in Q.ALLOCS} for n in Q.EXPECTED_K}
    ok = ok and all({a: Q.integer_k(a, n) for a in Q.ALLOCS} == Q.EXPECTED_K[n] for n in Q.EXPECTED_K) and len(hold) == 398
    out["static_explicit_argument_scan"] = static_default_scan()
    ok = ok and out["static_explicit_argument_scan"]["status"] == "PASS"
    sys.path.insert(0, str(ROOT / "tests"))
    suite = unittest.defaultTestLoader.loadTestsFromName("test_qwen_budget_sweep")

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
    with (gates / "sg1_unittest.log").open("x") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=Recorder).run(suite)
    out["fixture_tests"] = {"outcomes": result.outcomes, "run": result.testsRun, "log_sha256": sha(gates / "sg1_unittest.log"),
                            "covers": {"other ratio's meta / truncated file / SHA tamper / row-count tamper rejected": "RatioStoreTests.test_validator_rejects_other_ratio_tampered_metadata_and_payload",
                                       "no reliance on layer_budget defaults (32, 0.10, 0.60, 0.25); 28-layer r_min 0.0 r_max 1.0 total 0.3 passes, total_ratio -> 0.25 rejected":
                                           "RatioStoreTests.test_no_reliance_on_llava_layer_budget_defaults + VectorTests.test_llava_defaults_reject_qwen_vectors",
                                       "U25 payload == PyramidKV-run uniform packed payload (synthetic capture)": "RatioStoreTests.test_u25_payload_is_bitwise_the_pyramidkv_run_uniform_packed_store",
                                       "U100 == FullLoad store rows (synthetic capture)": "RatioStoreTests.test_u100_is_the_full_prefix_and_equals_the_fullload_store",
                                       "k = integer fraction for N = 1..3000": "VectorTests.test_k_equals_integer_fraction_for_n_1_to_3000"}}
    ok = ok and result.wasSuccessful() and result.testsRun >= 14
    out["status"] = "PASS" if ok else "FAIL"
    write(gates / "sg1_cpu.json", out)
    return out


# ========================================================================= SG4 (CPU)
def stage_sg4(run, gates):
    """Nested selection on every HOLD image.  The image-only saliency scores are the ones each
    image's T1 capture produced in the PyramidKV run (U_packed selection artifact); the run-time
    audit of this sweep repeats the check on the scores actually captured here."""
    from mmimpress.qwen25.packed_store import QwenPackedStore
    from mmimpress.qwen25 import ratio_packed as RP
    from mmimpress.qwen25.store import stable_visual_order
    out = {"gate": "SG4", **bind(run), "score_source": str((Q.PYR_RUN / "main/images/<image>/<attempt>/qwen_ours_prefixstore_kv25_U_packed_selection.json").relative_to(ROOT))}
    manifest = read(run / "manifest.json")
    bad, checked, sizes = [], 0, {a: 0 for a in Q.ALLOCS}
    for iid, ident in sorted(manifest["image_identity"].items()):
        commit = read(Q.PYR_RUN / "main/images" / iid / "COMMITTED.json")
        sel = read(Q.PYR_RUN / "main/images" / iid / commit["attempt_id"] / "qwen_ours_prefixstore_kv25_U_packed_selection.json")
        n = int(ident["N_content"])
        scores = sel["scores_float32"]
        order = stable_visual_order(scores, n)                              # the function both packed writers use
        rank = sorted(range(n), key=lambda i: (-scores[i], i))               # independent
        ok = len(scores) == n == sel["N_content"] and order == rank == sel["full_importance_permutation"]
        previous = set()
        for a in Q.ALLOCS:
            k = Q.integer_k(a, n)
            current = set(order[:k])
            ok = ok and len(current) == k and previous <= current and RP.uniform_keep_count(Q.RATIO[a], n) == k
            sizes[a] += k
            previous = current
        ok = ok and previous == set(range(n))
        checked += 1
        if not ok:
            bad.append(iid)
    question_free = {}
    for fn in (RP.write_ratio_packed_store, RP.ratio_keep_counts, RP.uniform_keep_count, QwenPackedStore.plan, stable_visual_order):
        names = set(inspect.signature(fn).parameters)
        question_free[fn.__qualname__] = not (names & {"question", "history", "prompt", "text", "query"})
    out.update(images=checked, failing_images=bad, total_selected_rows_per_ratio=sizes,
               chain="U25 subset U30 subset U40 subset U50 subset U100 == all content ids, sizes == ceil(p*N/q)",
               question_invariance={"selection_functions_take_no_question": question_free,
                                    "mechanism": "the store (and so the selected ids) is written once from the image's T1 capture; every later hit of that image, whatever its question, "
                                                 "reads that file. Verified on GPU with two different hit prompts (SG5) and on every HOLD image by the run-time audit "
                                                 "(same_image_T2_T3_selection_invariant_to_question)"},
               status="PASS" if checked == 398 and not bad and all(question_free.values()) else "FAIL")
    write(gates / "sg4_inclusion.json", out)
    return out


def stage_cpu(run):
    gates = run / "gates"
    gates.mkdir(exist_ok=True)
    sg1 = stage_sg1(run, gates)
    sg4 = stage_sg4(run, gates)
    # ---- complete existing CPU suite (GLM modules reported separately: pre-existing environment errors)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               PYTHONPATH=f"{ROOT}:{ROOT / 'scripts'}:{ROOT / 'tests'}")
    modules = sorted(p.stem for p in (ROOT / "tests").glob("test_*.py"))
    glm = [m for m in modules if "glm46v" in m]
    rest = [m for m in modules if m not in glm]
    with (gates / "cpu_regression.log").open("x") as log:
        p = subprocess.run([sys.executable, "-m", "unittest"] + rest, cwd=ROOT / "tests", stdout=log, stderr=subprocess.STDOUT, env=env)
    with (gates / "cpu_regression_glm.log").open("x") as log:
        g = subprocess.run([sys.executable, "-m", "unittest"] + glm, cwd=ROOT / "tests", stdout=log, stderr=subprocess.STDOUT, env=env)
    text = (gates / "cpu_regression.log").read_text()
    glm_text = (gates / "cpu_regression_glm.log").read_text()
    ran = [line for line in text.splitlines() if line.startswith("Ran ")]
    previous = read(Q.PYR_RUN / "gates" / "cpu_regression.json")
    reg = {"gate": "existing CPU tests", **bind(run), "status": "PASS" if p.returncode == 0 else "FAIL", "returncode": p.returncode, "modules": len(rest),
           "ran": ran[-1] if ran else None, "log_sha256": sha(gates / "cpu_regression.log"), "includes_new_module": "test_qwen_budget_sweep" in rest,
           "glm_modules_run_separately": glm, "glm_returncode": g.returncode, "glm_errors": [line for line in glm_text.splitlines() if line.startswith(("ERROR:", "FAIL:"))],
           "glm_log_sha256": sha(gates / "cpu_regression_glm.log"), "glm_errors_in_pyramidkv_run": previous["glm_errors"],
           "glm_note": "GLM-4.6V tests fail before this change on the installed transformers API (the same errors as in the PyramidKV run and its pre-change baseline); "
                       "they import nothing added here; reported separately as in the PyramidKV run, not hidden"}
    reg["glm_errors_same_as_pyramidkv_run"] = reg["glm_errors"] == previous["glm_errors"]
    write(gates / "cpu_regression.json", reg)
    print(json.dumps({"SG1": sg1["status"], "SG4": sg4["status"], "cpu_regression": reg["status"], "ran": reg["ran"], "glm_errors": len(reg["glm_errors"]),
                      "glm_same_as_before": reg["glm_errors_same_as_pyramidkv_run"]}), flush=True)
    assert sg1["status"] == "PASS" and sg4["status"] == "PASS" and reg["status"] == "PASS"


# ========================================================================= GPU gates
def gpu_case(runner, G, V, sample, case_dir, docs, old_u, run, prior_rank):
    import torch
    from mmimpress.qwen25 import layerwise as LW
    from mmimpress.qwen25 import packed_store as PS
    from mmimpress.qwen25 import ratio_packed as RP
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
    k_int = {a: Q.integer_k(a, n) for a in Q.ALLOCS}
    logical = {a: sorted(structural + [cap.visual_start + i for i in rank[:k_int[a]]]) for a in Q.ALLOCS}
    ev.update(N=n, S=s, prefix_len=prefix_len, T1_prediction=t1["prediction"], k=k_int)
    LW.persist_packed(runner, cap, stores / "U_pyr", old_u)                      # PyramidKV-run writer, its own U document
    runner.persist(cap, stores / "fullload", "canonical")                        # existing FullLoad store
    for a in Q.ALLOCS:
        RP.persist_ratio_packed(runner, cap, stores / a, docs[a])
    session = RP.RatioPackedSession(runner)
    old_session = LW.PackedSession(runner)
    forms = [("no_history", sample["question"], ()), ("one_turn_history", FOLLOWUP_QUESTION, ((sample["question"], t1["prediction"]),))]
    grid = torch.tensor(cap.image_grid_thw)
    gate = {g: True for g in GPU_GATES}
    struct_bytes = 2 * L * s * ROW

    def check(name, ok, key, detail=None):
        ev.setdefault(name, {})[key] = {"ok": bool(ok), **({"detail": detail} if detail is not None else {})}
        gate[name] = gate[name] and bool(ok)
        return bool(ok)

    def canonical_layers(a):
        idx = torch.tensor(logical[a], device=cap.layers[0][0].device)
        return [(cap.layers[li][0].index_select(2, idx).cpu(), cap.layers[li][1].index_select(2, idx).cpu()) for li in range(L)]

    try:
        # ------------------------------------------------------------ SG2: U25 payload / tensors
        new_blob = (stores / "U25" / RP.PACKED_NAME).read_bytes()
        old_blob = (stores / "U_pyr" / PS.PACKED_NAME).read_bytes()
        check("SG2", new_blob == old_blob and len(new_blob) == struct_bytes + L * 2 * k_int["U25"] * ROW, "payload_bitwise_equal_to_pyramidkv_run_writer",
              {"bytes": len(new_blob), "sha256": hashlib.sha256(new_blob).hexdigest()})
        new_meta = json.loads((stores / "U25" / RP.META_NAME).read_text())
        old_meta = json.loads((stores / "U_pyr" / PS.META_NAME).read_text())
        differing = sorted(key for key in set(new_meta) | set(old_meta) if new_meta.get(key) != old_meta.get(key))
        check("SG2", set(differing) <= META_DIFF_ALLOWED and new_meta["packed_sha256"] == old_meta["packed_sha256"] and new_meta["packed_layout"] == old_meta["packed_layout"]
              and new_meta["full_importance_permutation"] == old_meta["full_importance_permutation"] == rank and new_meta["layer_keep_counts"] == old_meta["layer_keep_counts"],
              "meta_differs_only_in_document_fields", {"differing_fields": differing,
                                                       "values": {key: {"sweep": new_meta.get(key), "pyramidkv_run": old_meta.get(key)} for key in differing if key != "extra"}})
        u25 = session.activate(stores / "U25", image_sha, docs["U25"])
        upyr = old_session.activate(stores / "U_pyr", image_sha, old_u)
        u25.read_all()
        upyr.read_all()
        a_layers, b_layers, c_layers = u25.assemble_cpu(), upyr.assemble_cpu(), canonical_layers("U25")
        check("SG2", len(a_layers) == 28 and all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) and torch.equal(x[0], z[0]) and torch.equal(x[1], z[1]) and x[0].dtype == torch.bfloat16
                                                 for x, y, z in zip(a_layers, b_layers, c_layers)), "assembled_KV_bitwise_vs_pyramidkv_run_reader_and_canonical_capture")
        del a_layers, b_layers, c_layers

        # ------------------------------------------------------------ SG3: U100 tensors vs FullLoad
        fl_dir = stores / "fullload"
        fl_store = runner._activate(fl_dir, image_sha)
        loaded = fl_store.load_prefix(1., budget_unit="chunk")
        u100 = session.activate(stores / "U100", image_sha, docs["U100"])
        u100.read_all()
        x_layers = u100.assemble_cpu()
        same_order = list(loaded.logical_indices) == u100.layer_logical_indices(0) == list(range(prefix_len)) == logical["U100"]
        check("SG3", same_order, "row_set_and_order_equal_full_prefix", {"rows": len(loaded.logical_indices)})
        check("SG3", len(x_layers) == 28 and all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) and x[0].dtype == torch.bfloat16 and list(x[0].shape) == [1, 4, prefix_len, 128]
                                                 for x, y in zip(x_layers, loaded.layers)), "layer_KV_bitwise_vs_fullload_load_prefix")
        check("SG3", all(torch.equal(x[0], cap.layers[li][0][:, :, :prefix_len].cpu()) and torch.equal(x[1], cap.layers[li][1][:, :, :prefix_len].cpu()) for li, x in enumerate(x_layers)),
              "layer_KV_bitwise_vs_T1_capture")
        del loaded, x_layers
        if prior_rank is not None:      # the same image in the PyramidKV-run main (its own source dialogue / T1 question) -> same image-only order
            check("SG5", rank == prior_rank["rank"], "image_only_rank_equals_pyramidkv_run_rank",
                  {"pyramidkv_run_T1_question": prior_rank["question"], "this_gate_T1_question": sample["question"], "T1_question_differs": prior_rank["question"] != sample["question"]})

        # ------------------------------------------------- per hit form: production / reference / paths
        selected = {}
        for form, question, history in forms:
            hits = {}
            for a in Q.ALLOCS:
                store_dir = stores / a
                store = session.activate(store_dir, image_sha, docs[a])
                meta = store.meta
                full, suffix, positions, deltas = runner._cache_hit_ids(meta, question, history)
                sp, sd = V.stock_positions(runner, full, grid)
                t = int(suffix.shape[1])
                tag = f"{a}:{form}"
                session.condition(store_dir, image_sha, docs[a])
                with V.IOTap(runner) as io, V.SdpaTap() as prod:
                    hit = RP.run_cache_ratio(runner, session, store_dir, question, history, layer_budget=docs[a], image_sha256=image_sha, return_logits=True)
                if a in ("U25", "U100"):
                    hits[a] = (hit, prod)
                # SG5: I/O
                total = (store_dir / RP.PACKED_NAME).stat().st_size
                check("SG5", len(io.calls) == 1 and io.calls[0]["syscall"] == "preadv" and io.calls[0]["offset"] == 0 and io.calls[0]["returned"] == io.calls[0]["requested"] == total
                      and Path(io.calls[0]["path"]).resolve() == (store_dir / RP.PACKED_NAME).resolve() and hit["pread_calls"] == 1 and hit["read_io"]["bytes"] == total
                      and total == struct_bytes + 28 * 2 * k_int[a] * ROW == meta["packed_total_bytes"], f"{tag}:one_preadv_zero_pread_bytes_equal_file_equal_struct_plus_28_2_k_rows",
                      {"bytes": total, "k": k_int[a]})
                check("SG5", io.vision == 0 and io.score == 0 and hit["vision_calls"] == 0 and hit["online_query_score_calls"] == 0, f"{tag}:no_vision_no_score")
                check("SG5", hit["layer_keep_counts"] == [k_int[a]] * 28 and hit["k_uniform"] == k_int[a] and hit["mask_hook_realigned_layers"] == 0 and hit["residual_layer_hooks_after"] == 0
                      and hit["gpu_cache_kv_bytes"] == total and torch.equal(sp, positions) and torch.equal(sd, deltas), f"{tag}:uniform_k_stock_positions_hooks_removed")
                selected.setdefault(a, []).append((hit["selected_importance_prefix_original"], hit["layer_keep_counts"], io.calls[0]["offset"], io.calls[0]["requested"]))
                # SGR: independent in-memory reference of the same shape, repeat, FP64 semantics
                ref = V.reference_run(runner, G, cap, [logical[a]] * L, suffix, positions, prefix_len)
                session.condition(store_dir, image_sha, docs[a])
                again = RP.run_cache_ratio(runner, session, store_dir, question, history, layer_budget=docs[a], image_sha256=image_sha, return_logits=True)
                eq = V.records_equal(prod.records, ref["tap"].records)
                lens_ok = all(prod.records[li]["k"].shape[2] == s + k_int[a] + t and list(prod.records[li]["mask"].shape) == [1, 1, t, s + k_int[a] + t] for li in range(L))
                check("SGR", len(prod.records) == 28 and all(eq) and lens_ok, f"{tag}:SDPA_QKV_mask_bitwise_28_layers_vs_independent_reference", {"layers_equal": sum(eq), "suffix": t})
                check("SGR", V.tensor_sha(hit["first_logits"]) == V.tensor_sha(ref["first_logits"]) == V.tensor_sha(again["first_logits"])
                      and hit["generated_token_ids"] == ref["generated_token_ids"] == again["generated_token_ids"], f"{tag}:logits_bitwise_ids_equal_ssd_memory_repeat",
                      {"logits_sha256": V.tensor_sha(hit["first_logits"]), "generated": hit["generated_token_ids"]})
                if a in ("U30", "U40", "U50"):
                    fp64_max, legacy_max, legacy_fail, sem_ok = 0.0, 0.0, 0, True
                    try:
                        for li in range(L):
                            result, _ = G.semantic_layer(prod.records[li], cap.layers[li], logical[a], prefix_len, repeats=2)
                            fp64_max = max(fp64_max, result["semantic_fp64"]["max_abs"])
                            legacy_max = max(legacy_max, result["legacy_fp32"]["max_abs"])
                            legacy_fail += int(not result["legacy_fp32"]["allclose"])
                            sem_ok = sem_ok and result["semantic_fp64"]["allclose"] and result["semantic_fp64"]["violations"] == 0
                        check("SGR", sem_ok, f"{tag}:fp64_dense_vs_compact_1e-12_all_layers",
                              {"fp64_max_abs": fp64_max, "legacy_fp32_1e-5_diagnostic_max_abs": legacy_max, "legacy_fp32_layers_failing_diagnostic_only": legacy_fail})
                    except AssertionError as exc:
                        check("SGR", False, f"{tag}:fp64_dense_vs_compact_1e-12_all_layers", {"error": repr(exc)})
                del ref, again
            # ---- SG2: U25 hit vs the PyramidKV-run packed hit path (its own store, session and function)
            hit, prod = hits["U25"]
            old_session.condition(stores / "U_pyr", image_sha, old_u)
            with V.SdpaTap() as old_tap:
                old_hit = LW.run_cache_layerwise(runner, old_session, stores / "U_pyr", question, history, layer_budget=old_u, image_sha256=image_sha, return_logits=True)
            check("SG2", V.tensor_sha(old_hit["first_logits"]) == V.tensor_sha(hit["first_logits"]) and old_hit["generated_token_ids"] == hit["generated_token_ids"],
                  f"{form}:first_logits_bitwise_and_generated_ids_vs_pyramidkv_run_hit", {"generated": hit["generated_token_ids"]})
            check("SG2", all(V.records_equal(prod.records, old_tap.records)) and len(old_tap.records) == 28 and old_hit["packed_read_bytes"] == hit["packed_read_bytes"]
                  and old_hit["layer_keep_counts"] == hit["layer_keep_counts"], f"{form}:SDPA_QKV_mask_bitwise_and_same_bytes_vs_pyramidkv_run_hit")
            del old_tap, old_hit
            # ---- SG3: U100 hit vs the existing FullLoad path
            hit, prod = hits["U100"]
            runner.condition_cache(fl_dir, 1., budget_unit="chunk", image_sha256=image_sha)
            with V.SdpaTap() as fl_tap:
                fl_hit = runner.run_cache(fl_dir, question, history=history, budget_ratio=1., budget_unit="chunk", image_sha256=image_sha, return_logits=True)
            check("SG3", V.tensor_sha(fl_hit["first_logits"]) == V.tensor_sha(hit["first_logits"]), f"{form}:first_logits_bitwise_vs_fullload",
                  {"max_abs_logit_diff": float((fl_hit["first_logits"] - hit["first_logits"]).abs().max())})
            check("SG3", fl_hit["generated_token_ids"] == hit["generated_token_ids"], f"{form}:generated_ids_equal_vs_fullload",
                  {"u100": hit["generated_token_ids"], "fullload": fl_hit["generated_token_ids"]})
            eq = V.records_equal(prod.records, fl_tap.records)
            check("SG3", all(eq) and len(fl_tap.records) == 28 and fl_hit["gpu_cache_kv_bytes"] == hit["gpu_cache_kv_bytes"] and fl_hit["pread_calls"] == 57 and hit["pread_calls"] == 1,
                  f"{form}:SDPA_QKV_mask_bitwise_same_cache_bytes_57_pread_vs_1_preadv", {"layers_equal": sum(eq), "fullload_read_bytes": fl_hit["read_io"]["bytes"], "u100_read_bytes": hit["packed_read_bytes"]})
            del fl_tap, fl_hit, hits
        check("SG5", all(all(x == v[0] for x in v) and len(v) == 2 for v in selected.values()), "question_change_keeps_read_range_and_selection")
        chain = [set(selected[a][0][0][:k_int[a]]) for a in Q.ALLOCS]
        check("SG5", all(x <= y for x, y in zip(chain, chain[1:])) and chain[-1] == set(range(n)) and [len(x) for x in chain] == [k_int[a] for a in Q.ALLOCS], "request_selection_nested_across_ratios")
        # ---- session / validator refuse a store of another ratio or image
        neg = {"U30_store_requested_as_U40": V.reject(lambda: session.activate(stores / "U30", image_sha, docs["U40"])),
               "U100_store_requested_as_U25": V.reject(lambda: session.activate(stores / "U100", image_sha, docs["U25"])),
               "U40_store_opened_with_U50_document": V.reject(lambda: RP.open_ratio_packed_store(stores / "U40", expected_budget=docs["U50"], pin=False)),
               "pyramidkv_run_store_opened_by_ratio_reader": V.reject(lambda: RP.open_ratio_packed_store(stores / "U_pyr", pin=False)),
               "ratio_store_opened_by_pyramidkv_run_reader": V.reject(lambda: PS.open_qwen_packed_store(stores / "U25", pin=False)),
               "foreign_image_identity": V.reject(lambda: RP.RatioPackedSession(runner).activate(stores / "U50", "0" * 64, docs["U50"]))}
        check("SG5", all(x["rejected"] for x in neg.values()), "other_ratio_other_format_foreign_image_rejected", neg)
        ev["gates"] = gate
        ev["status"] = "PASS" if all(gate.values()) else "FAIL"
    finally:
        for layer in LW.decoder_layers(runner.model):
            layer.self_attn._forward_pre_hooks.clear()
        session.close()
        old_session.close()
        runner.close()
        del cap
        ev["removed_gate_store_files"] = len(V.guarded_remove(stores, run))
    return ev


def stage_gpu(run):
    import torch
    config = read(run / "config.json")
    assert Q.source_hashes() == config["source_sha256"], "FROZEN_SOURCE_CONFIG_MISMATCH"
    for name in ("sg1_cpu.json", "sg4_inclusion.json", "cpu_regression.json"):
        doc = read(run / "gates" / name)
        assert doc["status"] == "PASS" and doc["config_sha256"] == sha(run / "config.json"), name
    assert not (run / "gates" / "sg_gpu.json").exists(), "GPU gate already recorded; failed attempts are preserved, re-freeze before a new attempt"
    docs = Q.load_vectors(run)
    old_u = read(Q.PYR_RUN / "vectors" / "layer_budget_qwen_U.json")
    M = load_script("_qsweep_base110", "110_eval_qwen_mt_gqa_B.py")
    G = load_script("_qsweep_semantic116", "116_qwen_prefixstore_semantic.py")
    V = load_script("_qsweep_validate168", "168_validate_qwen_layerwise_packed.py")      # SdpaTap / IOTap / reference_run helpers only
    E = load_script("_qsweep_eval", Q.EVAL_SCRIPT)
    samples = read(run / "correctness_manifest.json")["samples"]
    assert len(samples) == 10
    first_question = {d["dialog_id"]: d["turns"][0]["question"] for d in read(run / "manifest.json")["dialogues"]}
    E.seed()
    summary = {"gate": "SG2-SG3-SG5 (+SGR reference)", **bind(run), "status": "RUNNING", "samples": [],
               "fixture": "10 frozen core pairs (correctness_manifest.json of the earlier Qwen runs); two hit prompts each",
               "gates": {"SG2": "25% regression vs the PyramidKV-run packed writer / reader / hit path", "SG3": "100% equivalence with the existing FullLoad path",
                         "SG5": "I/O: one preadv, zero pread, bytes, no vision/score; selection invariant to the question and nested across ratios",
                         "SGR": "extra: every ratio vs an independent in-memory reference (SDPA inputs, logits, ids), repeat, FP64 dense-vs-compact for 30/40/50%"},
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
            prior_rank = None
            cp = Q.PYR_RUN / "main/images" / sample["image_id"] / "COMMITTED.json"
            if cp.exists():
                sel = read(cp.parent / read(cp)["attempt_id"] / "qwen_ours_prefixstore_kv25_U_packed_selection.json")
                prior_rank = {"rank": sel["full_importance_permutation"], "question": first_question[sel["source"]["dialog_id"]]}
            try:
                ev = gpu_case(runner, G, V, sample, case, docs, old_u, run, prior_rank)
            except BaseException as exc:
                write(case / "evidence.json", {"status": "FAIL", "error": repr(exc), "traceback": traceback.format_exc()})
                summary["samples"].append({"index": i, "image_id": sample["image_id"], "status": "FAIL", "error": repr(exc)})
                raise
            ev["elapsed_s"] = time.time() - t0
            ev["prior_rank_available"] = prior_rank is not None
            write(case / "evidence.json", ev)
            summary["samples"].append({"index": i, "image_id": sample["image_id"], "N": ev["N"], "status": ev["status"], "gates": ev["gates"],
                                       "prior_rank_available": prior_rank is not None, "evidence_sha256": sha(case / "evidence.json")})
            print(json.dumps({"core": i + 1, "of": len(samples), "image": sample["image_id"], "N": ev["N"], **ev["gates"], "s": round(ev["elapsed_s"], 1)}), flush=True)
            torch.cuda.empty_cache()
        per_gate = {g: all(x["gates"][g] for x in summary["samples"]) for g in GPU_GATES}
        summary.update(per_gate=per_gate, status="PASS" if all(per_gate.values()) and len(summary["samples"]) == 10 else "FAIL")
    except BaseException as exc:
        summary.update(status="FAIL", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if runner:
            runner.close()
        write(run / "gates" / "sg_gpu.json", summary)
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

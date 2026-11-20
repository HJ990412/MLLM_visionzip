#!/usr/bin/env python3
"""Validation steps of the MT-GQA run whose IMPRESS arm is the public-implementation
path (contract docs/mt_gqa_impress_code_contract.md, section 8).

  --stage cpu      step 1 (public-equivalence tests against the public checkout,
                   IMPRESS_OFFICIAL_REPO) and step 6 (the whole CPU suite in this
                   interpreter and, when given, in a second one)
  --stage gate     on the committed rows of the ``gate`` phase: step 3 (IMPRESS at
                   100 % retention == the FullLoad loader on the same reordered
                   store), step 4 (asynchronous == synchronous for FullLoad,
                   IMPRESS, Ours: tokens and bytes read), step 5 (ReComp /
                   FullLoad / Ours == the README-38 main run; turn-1 tokens equal
                   across arms)
  --stage prefix   step 2 (GPU): the stored prefix-row importance of every gate
                   image against a second full computation from pixels
  --stage dev      step 7 on the committed rows of the ``dev`` phase, and step 5
                   again over every DEV request
  --stage admit    writes gates/admission.json (PASS only if every gate file is
                   PASS); the main phase refuses to start without it
  --stage posthoc  after the main phase (not a gate): step 5 over all 398 images

Criteria are constants of this file, frozen with the run's source hash.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

NEW_TESTS = ("test_impress_official.py", "test_impress_code_runs.py")
BYTE_FIELDS = ["ssd_read_bytes", "probe_read_bytes", "k_read_bytes", "v_read_bytes", "packed_read_bytes"]


def _load(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


G = _load("188_validate_mt_gqa_async.py", "_mtgqa_async_gates")      # write / committed_rows / compare / run_suite / baselines
write, committed_rows, compare = G.write, G.committed_rows, G.compare


def load_harness():
    return _load("196_eval_mt_gqa_impress_code.py", "_mtgqa_impress_code_harness")


# ---------------------------------------------------------------- steps 1, 6
def public_equivalence(run, python=sys.executable):
    checkout = os.environ.get("IMPRESS_OFFICIAL_REPO")
    if not checkout:
        raise SystemExit("IMPRESS_OFFICIAL_REPO is not set")
    head = subprocess.check_output(["git", "-C", checkout, "rev-parse", "HEAD"], text=True).strip()
    public = []; ran = 0; code = 0; tails = []
    for name in NEW_TESTS:
        p = subprocess.run([python, "-m", "unittest", "discover", "-s", "tests", "-p", name, "-v"], capture_output=True, text=True, cwd=ROOT)
        text = p.stdout + p.stderr
        cases = re.findall(r"^(test\w+) \(([\w.]+)\)(?:\n.*?)? \.\.\. (ok|FAIL|ERROR|skipped.*)$", text, re.M)
        public += [{"file": name, "test": t, "class": c, "outcome": o} for t, c, o in cases if c.endswith("PublicEquivalenceTests")]
        m = re.search(r"^Ran (\d+) tests", text, re.M)
        ran += int(m.group(1)) if m else 0; code = code or p.returncode; tails.append(text[-800:])
    files = {x["file"] for x in public}
    return {"gate": "step1_public_equivalence", "public_checkout": checkout, "public_head": head,
            "criterion": "every PublicEquivalenceTests case of both test files runs (none skipped) and passes against the public checkout: "
                         "per-head top-k, mean Jaccard, fallback decision and the selected set equal the public functions' on the same tensors",
            "ran": ran, "returncode": code, "public_cases": public,
            "status": "PASS" if code == 0 and files == set(NEW_TESTS) and all(x["outcome"] == "ok" for x in public) else "FAIL", "tail": tails}


def stage_cpu(run, extra_python):
    step1 = public_equivalence(run)
    write(run / "gates" / "step1_public_equivalence.json", step1)
    suites = [G.run_suite(sys.executable)]
    if extra_python:
        suites.append(G.run_suite(extra_python))
    step6 = {"gate": "step6_cpu_tests", "suites": suites,
             "criterion": "no failure outside the environment-specific baseline recorded before this work "
                          "(transformers 5.13.1: 2 tests; transformers 4.57.6: the 3 GLM-4.6V tests)",
             "status": "PASS" if all(r["status"] == "PASS" for r in suites) else "FAIL"}
    write(run / "gates" / "step6_cpu_tests.json", step6)
    print(json.dumps({"step1": step1["status"], "public_cases": [(x["test"], x["outcome"]) for x in step1["public_cases"]], "step6": step6["status"],
                      "suites": [{k: r[k] for k in ("transformers", "ran", "failed", "new_failures")} for r in suites]}, indent=1))
    return "PASS" if step1["status"] == step6["status"] == "PASS" else "FAIL"


# ------------------------------------------------------------- shared checks
def phase_keys(run, phase):
    manifest = json.loads((run / "manifest.json").read_text())
    config = json.loads((run / "config.json").read_text())
    keys = {(d, t) for d in manifest["phase_dialogue_ids"][phase] for t in (1, 2, 3)}
    validation = run / phase / "validation.json"
    complete = validation.exists() and json.loads(validation.read_text())["status"] == "PASS"
    return manifest, config, keys, config["phases"][phase]["methods"], complete


def matches_previous_main(H, rows, keys, methods):
    """Generated tokens of this run against the README-38 main run's record of
    the same (dialogue, turn) and the same arm."""
    images = {r["image_id"] for r in rows.values()}
    old_rows = committed_rows(H.PRIOR_ASYNC_RUN, "main", set(methods), images)
    out = []
    for m in methods:
        mism = []; n = 0; prompt_equal = 0
        for dialog, turn in sorted(keys):
            new, old = rows.get((m, dialog, turn)), old_rows.get((m, dialog, turn))
            if new is None or old is None:
                mism.append({"dialog_id": dialog, "turn_id": turn, "reason": "missing in " + ("this run" if new is None else "the README-38 main run")}); continue
            n += 1; prompt_equal += new["prompt"] == old["prompt"]
            if new["result"]["generated_token_ids"] != old["result"]["generated_token_ids"]:
                mism.append({"dialog_id": dialog, "turn_id": turn, "image_id": new["image_id"], "new": new["prediction"], "previous": old["prediction"],
                             "new_tokens": new["result"]["generated_token_ids"], "previous_tokens": old["result"]["generated_token_ids"],
                             "prompt_equal": new["prompt"] == old["prompt"]})
        out.append({"method": m, "requests": n, "expected": len(keys), "prompt_equal": prompt_equal,
                    "mismatch_count": len(mism), "mismatches": mism[:50], "status": "PASS" if n == len(keys) and not mism else "FAIL"})
    return {"criterion": "ReComp, FullLoad, Ours: generated tokens identical to the README-38 main run's record of the same (dialogue, turn)",
            "reference": str((H.PRIOR_ASYNC_RUN / "main").relative_to(ROOT)), "methods": out,
            "status": "PASS" if out and all(x["status"] == "PASS" for x in out) else "FAIL"}


def turn1_identity(rows, keys, methods):
    """Turn-1 generated tokens equal across every arm, per dialogue."""
    dialogs = sorted({d for d, t in keys if t == 1}); bad = []
    for d in dialogs:
        got = {m: rows[(m, d, 1)]["result"]["generated_token_ids"] for m in methods if (m, d, 1) in rows}
        if len(got) != len(methods) or len({tuple(v) for v in got.values()}) != 1:
            bad.append({"dialog_id": d, "tokens": got})
    return {"criterion": "the turn-1 generated tokens are identical across all arms of the phase, for every dialogue",
            "dialogues": len(dialogs), "arms": list(methods), "mismatch_count": len(bad), "mismatches": bad[:20],
            "status": "PASS" if dialogs and not bad else "FAIL"}


def execution(run, phase, rows, manifest, complete):
    return {"gate": f"{phase}_phase_execution", "requests": len(rows), "expected": manifest["expected_requests"][phase],
            "images": len({r["image_id"] for r in rows.values()}), "phase_validation_pass": complete,
            "criterion": "every image committed after its independent audit passed",
            "status": "PASS" if complete and len(rows) == manifest["expected_requests"][phase] else "FAIL"}


# --------------------------------------------------------- steps 3, 4, 5 (gate)
def stage_gate(run):
    H = load_harness()
    manifest, config, keys, methods, complete = phase_keys(run, "gate")
    rows = committed_rows(run, "gate")
    ex = execution(run, "gate", rows, manifest, complete)
    g3 = compare(rows, H.NEW_R100, H.FULL_REORDERED, ["generated_token_ids", "prompt"], keys)
    g3 = {"gate": "step3_keep100_equals_fullload_loader_on_reordered_store",
          "criterion": "generated tokens (and the prompt) identical for every request: the IMPRESS path at 100 % retention vs the FullLoad loader "
                       "(no selection, no mask) reading the same reordered store", **g3}
    ref = compare(rows, H.FULL_REORDERED, "fullload", ["generated_token_ids"], keys)
    ref = {"gate": "reference_reordered_fullload_vs_raster_fullload", "criterion": "none (reported only): requests whose generated tokens differ between "
           "the FullLoad loader on the reordered store and FullLoad on the raster store", **ref, "status": "REPORTED",
           "different_requests": ref["mismatch_count"], "different_requests_same_prompt": sum(1 for x in ref["mismatches"] if x.get("prompt_equal"))}
    pairs = (("FullLoad", "fullload", "fullload_async"), ("IMPRESS", H.NEW, H.NEW_ASYNC), ("Ours", H.OURS, H.OURS_ASYNC))
    per = {name: compare(rows, b, a, ["generated_token_ids"] + BYTE_FIELDS, keys) for name, a, b in pairs}
    g4 = {"gate": "step4_async_equals_sync", "criterion": "for FullLoad, IMPRESS and Ours: generated tokens and bytes read (total and by kind) "
          "identical between the asynchronous and the synchronous arm, every request", "families": per,
          "status": "PASS" if all(v["status"] == "PASS" for v in per.values()) else "FAIL"}
    g5 = {"gate": "step5_matches_previous_main_gate", **matches_previous_main(H, rows, keys, ["recompute", "fullload_async", H.OURS_ASYNC])}
    t1 = {"gate": "step5_turn1_identity_gate", **turn1_identity(rows, keys, methods)}
    for name, obj in (("gate_execution", ex), ("step3_keep100_equals_fullload_loader", g3), ("reference_reordered_vs_raster_fullload", ref),
                      ("step4_async_equals_sync", g4), ("step5_matches_previous_main_gate", g5), ("step5_turn1_identity_gate", t1)):
        write(run / "gates" / f"{name}.json", obj)
    summary = {"execution": ex["status"], "step3": g3["status"], "step4": g4["status"], "step5_previous_main": g5["status"], "step5_turn1": t1["status"]}
    print(json.dumps(summary, indent=1))
    print(json.dumps({"step3": {k: g3[k] for k in ("requests", "equal", "mismatch_count")},
                      "reference_reordered_vs_raster": {k: ref[k] for k in ("requests", "different_requests", "different_requests_same_prompt")},
                      "step4": {n: {k: v[k] for k in ("requests", "mismatch_count")} for n, v in per.items()},
                      "step5": [{k: m[k] for k in ("method", "requests", "prompt_equal", "mismatch_count")} for m in g5["methods"]],
                      "turn1": {k: t1[k] for k in ("dialogues", "mismatch_count")}}, indent=1))
    return "PASS" if all(v == "PASS" for v in summary.values()) else "FAIL"


# ------------------------------------------------------------------- step 2
def stage_prefix(run):
    """Stored prefix-row importance vs a second full computation (GPU).

    For every gate image: the committed evidence (the arrays that were written
    into the store; the image audit already tied the store's bytes to them) is
    compared with the statistics captured from a NEW pixel request that uses
    the same turn-1 prompt.  Criterion: every element equal.  A request with a
    different question is also run and reported (not a criterion): its prefix
    rows are the same quantity computed inside a longer sequence.
    """
    import torch
    H = load_harness()
    manifest, config, keys, methods, complete = phase_keys(run, "gate")
    assert complete, "gate phase is not complete"
    torch.set_num_threads(config["cpu_threads"]); torch.manual_seed(1234); np.random.seed(1234); torch.set_grad_enabled(False)
    H.assert_gpu_exclusive(); runner = H.V.load_runner()
    from mmimpress.serve import Server
    from mmimpress import impress_official as io_
    args = H.SERVER_ARGS
    server = Server(runner, ratio=args["ratio"], probe=args["probe_heads"], alpha=args["alpha"], max_new_tokens=16)
    images = []
    try:
        for cp in sorted((run / "gate" / "images").glob("*/COMMITTED.json")):
            doc = json.loads(cp.read_text()); folder = cp.parent / doc["attempt_id"]
            p = json.loads((folder / f"{H.NEW_KIND}_persistence.json").read_text())
            stats = np.load(folder / p["official_turn1"]["stats_file"])
            rows = H.load_jsonl(folder / "raw.jsonl")
            src = next(r for r in rows if r["physical_execution_id"] == p["source"]["physical_execution_id"])
            other = next(r for r in rows if r["method_id"] == src["method_id"] and r["dialog_id"] == src["dialog_id"] and r["turn_id"] == 2)
            dialog = next(d for d in manifest["dialogues"] if d["dialog_id"] == src["dialog_id"])
            path = ROOT / dialog["image_path"]

            def capture(prompt):
                result, diag = H.turn1_request_with_stats(runner, server, path, lambda: (prompt, "", []))
                prefix, order = diag["turn1_stats"].result()
                result.pop("captured_past_key_values", None)
                tokens = result["generated_token_ids"]
                del result, diag
                return prefix.numpy(), order.numpy(), tokens

            prefix, order, tokens = capture(src["prompt"])
            same = {"prefix_rows_equal": bool(np.array_equal(prefix, stats["prefix_rows"])),
                    "order_scores_equal": bool(np.array_equal(order, stats["order_scores"])),
                    "prefix_rows_max_abs_diff": float(np.abs(prefix - stats["prefix_rows"]).max()),
                    "order_scores_max_abs_diff": float(np.abs(order - stats["order_scores"]).max()),
                    "generated_tokens_equal": tokens == src["result"]["generated_token_ids"]}
            prefix2, _, _ = capture(other["prompt"])      # diagnostic: the same image inside another (longer) prompt
            stored = stats["prefix_rows"]
            cand = np.ones(stored.shape[2], dtype=bool); cand[src["geometry"]["newline_idx"]] = False
            k = io_.keep_count(int(cand.sum()))
            jac = []
            for li in range(stored.shape[0]):
                for h in range(stored.shape[1]):
                    a = set(np.argsort(-stored[li, h][cand], kind="stable")[:k].tolist())
                    b = set(np.argsort(-prefix2[li, h][cand], kind="stable")[:k].tolist())
                    jac.append(len(a & b) / len(a | b))
            other_prompt = {"prefix_rows_equal": bool(np.array_equal(prefix2, stored)), "max_abs_diff": float(np.abs(prefix2 - stored).max()),
                            "mean_abs_diff": float(np.abs(prefix2 - stored).mean()),
                            "prefix_only_topk_jaccard_mean": float(np.mean(jac)), "prefix_only_topk_jaccard_min": float(np.min(jac)),
                            "suffix_tokens_stored_request": len(src["suffix_input_ids"]), "suffix_tokens_other_request": len(other["suffix_input_ids"])}
            images.append({"image_id": cp.parent.name, "source_dialog_id": src["dialog_id"], "source_method_id": src["method_id"],
                           "shape": list(stored.shape), "separators": len(src["geometry"]["newline_idx"]), "same_prompt": same,
                           "other_prompt_diagnostic": other_prompt,
                           "status": "PASS" if same["prefix_rows_equal"] and same["order_scores_equal"] else "FAIL"})
            print(json.dumps({"image": cp.parent.name, **same, "other_prompt_max_abs_diff": other_prompt["max_abs_diff"]}), flush=True)
    finally:
        server.close()
    out = {"gate": "step2_prefix_rows_match_full_recompute",
           "criterion": "for every gate image, the stored prefix-row importance (and the order scores) equal, element for element, "
                        "the values captured from a second full computation from pixels with the same turn-1 prompt",
           "images": images, "status": "PASS" if images and all(x["status"] == "PASS" for x in images) else "FAIL"}
    write(run / "gates" / "step2_prefix_rows.json", out)
    print(json.dumps({"step2": out["status"], "images": len(images)}))
    return out["status"]


# ------------------------------------------------------------------- step 7
def stage_dev(run):
    H = load_harness()
    manifest, config, keys, methods, complete = phase_keys(run, "dev")
    rows = committed_rows(run, "dev")
    ex = execution(run, "dev", rows, manifest, complete)
    g5 = {"gate": "step5_matches_previous_main_dev", **matches_previous_main(H, rows, keys, ["recompute", "fullload_async", H.OURS_ASYNC])}
    t1 = {"gate": "step5_turn1_identity_dev", **turn1_identity(rows, keys, methods)}
    for name, obj in (("dev_execution", ex), ("step5_matches_previous_main_dev", g5), ("step5_turn1_identity_dev", t1)):
        write(run / "gates" / f"{name}.json", obj)
    summary = {"execution": ex["status"], "step5_previous_main": g5["status"], "step5_turn1": t1["status"]}
    print(json.dumps(summary, indent=1))
    print(json.dumps({"step5": [{k: m[k] for k in ("method", "requests", "prompt_equal", "mismatch_count")} for m in g5["methods"]],
                      "turn1": {k: t1[k] for k in ("dialogues", "mismatch_count")}}, indent=1))
    return "PASS" if all(v == "PASS" for v in summary.values()) else "FAIL"


# -------------------------------------------------------------------- admit
REQUIRED = ("step1_public_equivalence", "step2_prefix_rows", "gate_execution", "step3_keep100_equals_fullload_loader",
            "step4_async_equals_sync", "step5_matches_previous_main_gate", "step5_turn1_identity_gate", "step6_cpu_tests",
            "dev_execution", "step5_matches_previous_main_dev", "step5_turn1_identity_dev")


def stage_admit(run):
    H = load_harness()
    statuses = {}
    for name in REQUIRED:
        p = run / "gates" / f"{name}.json"
        statuses[name] = json.loads(p.read_text())["status"] if p.exists() else "MISSING"
    ok = all(v == "PASS" for v in statuses.values())
    out = {"status": "PASS" if ok else "FAIL", "gates": statuses, "config_sha256": H.sha(run / "config.json"),
           "decided_unix": time.time(), "rule": "main runs only if every gate of steps 1-7 is PASS"}
    write(run / "gates" / "admission.json", out)
    print(json.dumps(out, indent=1))
    return out["status"]


def stage_posthoc(run):
    """After the main phase; reported, never a gate."""
    H = load_harness()
    manifest, config, keys, methods, complete = phase_keys(run, "main")
    rows = committed_rows(run, "main")
    g5 = {"gate": "posthoc_matches_previous_main_all_images", **matches_previous_main(H, rows, keys, ["recompute", "fullload_async", H.OURS_ASYNC])}
    t1 = {"gate": "posthoc_turn1_identity_main", **turn1_identity(rows, keys, methods)}
    write(run / "gates" / "posthoc_matches_previous_main.json", g5)
    write(run / "gates" / "posthoc_turn1_identity_main.json", t1)
    print(json.dumps({"complete": complete, "previous_main": [{k: m[k] for k in ("method", "requests", "mismatch_count")} for m in g5["methods"]],
                      "turn1": {k: t1[k] for k in ("dialogues", "mismatch_count")}}, indent=1))
    return "PASS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--stage", choices=("cpu", "gate", "prefix", "dev", "admit", "posthoc"), required=True)
    ap.add_argument("--extra-python", default=None, help="second interpreter for the CPU suite (cpu stage)")
    args = ap.parse_args()
    run = args.run.resolve()
    status = {"cpu": lambda: stage_cpu(run, args.extra_python), "gate": lambda: stage_gate(run), "prefix": lambda: stage_prefix(run),
              "dev": lambda: stage_dev(run), "admit": lambda: stage_admit(run), "posthoc": lambda: stage_posthoc(run)}[args.stage]()
    if status != "PASS":
        raise SystemExit(f"{args.stage}: {status}")


if __name__ == "__main__":
    main()

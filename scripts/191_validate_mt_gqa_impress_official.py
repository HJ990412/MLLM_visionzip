#!/usr/bin/env python3
"""Gates for the MT-GQA run with the IMPRESS arm re-implemented from the public
code (contract docs/mt_gqa_impress_official_contract.md, section 8).

  --stage cpu     steps 1 and 5: the public-equivalence tests against the
                  public checkout (IMPRESS_OFFICIAL_REPO) and the whole CPU
                  suite in this interpreter and, when given, in a second one
  --stage gate    on the committed rows of the ``gate`` phase: step 3 (IMPRESS
                  at 100 % retention == FullLoad, tokens), step 4
                  (ReComp / FullLoad / Ours == the previous main run, tokens),
                  and the new arm's asynchronous == synchronous check
  --stage prefix  step 2 (GPU): the stored prefix-row importance of every gate
                  image against a second full computation from pixels
  --stage dev     step 6 on the committed rows of the ``dev`` phase, and
                  step 4 again over every DEV request
  --stage admit   writes gates/admission.json (PASS only if every gate file is
                  PASS); the main phase refuses to start without it

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

NEW_TESTS = "test_impress_official.py"


def _load(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


G = _load("188_validate_mt_gqa_async.py", "_mtgqa_async_gates")      # write / committed_rows / compare / run_suite / baselines
write, committed_rows, compare = G.write, G.committed_rows, G.compare


def load_harness():
    return _load("190_eval_mt_gqa_impress_official.py", "_mtgqa_impress_official_harness")


# ---------------------------------------------------------------- steps 1, 5
def stage_cpu(run, extra_python):
    checkout = os.environ.get("IMPRESS_OFFICIAL_REPO")
    if not checkout:
        raise SystemExit("IMPRESS_OFFICIAL_REPO is not set")
    head = subprocess.check_output(["git", "-C", checkout, "rev-parse", "HEAD"], text=True).strip()
    p = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", NEW_TESTS, "-v"],
                       capture_output=True, text=True, cwd=ROOT)
    text = p.stdout + p.stderr
    cases = re.findall(r"^(test\w+) \(([\w.]+)\)(?:\n.*?)? \.\.\. (ok|FAIL|ERROR|skipped.*)$", text, re.M)
    public = [{"test": t, "class": c, "outcome": o} for t, c, o in cases if c.endswith("PublicEquivalenceTests")]
    ran = re.search(r"^Ran (\d+) tests", text, re.M)
    step1 = {"gate": "step1_public_equivalence", "public_checkout": checkout, "public_head": head,
             "criterion": "every PublicEquivalenceTests case runs (none skipped) and passes against the public checkout",
             "ran": int(ran.group(1)) if ran else None, "returncode": p.returncode, "public_cases": public,
             "status": "PASS" if p.returncode == 0 and public and all(x["outcome"] == "ok" for x in public) else "FAIL", "tail": text[-1500:]}
    write(run / "gates" / "step1_public_equivalence.json", step1)
    suites = [G.run_suite(sys.executable)]
    if extra_python:
        suites.append(G.run_suite(extra_python))
    step5 = {"gate": "step5_cpu_tests", "suites": suites,
             "criterion": "no failure outside the environment-specific baseline recorded before this work "
                          "(transformers 5.13.1: 2 tests; transformers 4.57.6: the 3 GLM-4.6V tests)",
             "status": "PASS" if all(r["status"] == "PASS" for r in suites) else "FAIL"}
    write(run / "gates" / "step5_cpu_tests.json", step5)
    print(json.dumps({"step1": step1["status"], "public_cases": [(x["test"], x["outcome"]) for x in public], "step5": step5["status"],
                      "suites": [{k: r[k] for k in ("transformers", "ran", "failed", "new_failures")} for r in suites]}, indent=1))
    return "PASS" if step1["status"] == step5["status"] == "PASS" else "FAIL"


# ------------------------------------------------------------- shared checks
def phase_keys(run, phase):
    manifest = json.loads((run / "manifest.json").read_text())
    config = json.loads((run / "config.json").read_text())
    keys = {(d, t) for d in manifest["phase_dialogue_ids"][phase] for t in (1, 2, 3)}
    validation = run / phase / "validation.json"
    complete = validation.exists() and json.loads(validation.read_text())["status"] == "PASS"
    return manifest, config, keys, config["phases"][phase]["methods"], complete


def matches_previous_main(H, rows, keys, methods, blocking):
    """Generated tokens of this run against the previous main run's record of
    the same (dialogue, turn) and the same arm."""
    images = {r["image_id"] for r in rows.values()}
    old_rows = committed_rows(H.PRIOR_ASYNC_RUN, "main", set(methods), images)
    out = []
    for m in methods:
        mism = []; n = 0; prompt_equal = 0
        for dialog, turn in sorted(keys):
            new, old = rows.get((m, dialog, turn)), old_rows.get((m, dialog, turn))
            if new is None or old is None:
                mism.append({"dialog_id": dialog, "turn_id": turn, "reason": "missing in " + ("this run" if new is None else "the previous main run")}); continue
            n += 1; prompt_equal += new["prompt"] == old["prompt"]
            if new["result"]["generated_token_ids"] != old["result"]["generated_token_ids"]:
                mism.append({"dialog_id": dialog, "turn_id": turn, "image_id": new["image_id"], "new": new["prediction"], "previous": old["prediction"],
                             "new_tokens": new["result"]["generated_token_ids"], "previous_tokens": old["result"]["generated_token_ids"],
                             "prompt_equal": new["prompt"] == old["prompt"]})
        out.append({"method": m, "blocking": m in blocking, "requests": n, "expected": len(keys), "prompt_equal": prompt_equal,
                    "mismatch_count": len(mism), "mismatches": mism[:50], "status": "PASS" if n == len(keys) and not mism else "FAIL"})
    return {"criterion": "generated tokens identical to the previous main run's record of the same (dialogue, turn); "
                         "blocking arms: ReComp, FullLoad, Ours",
            "reference": str((H.PRIOR_ASYNC_RUN / "main").relative_to(ROOT)), "methods": out,
            "status": "PASS" if out and all(x["status"] == "PASS" for x in out if x["blocking"]) else "FAIL"}


def execution(run, phase, rows, manifest, complete):
    return {"gate": f"{phase}_phase_execution", "requests": len(rows), "expected": manifest["expected_requests"][phase],
            "images": len({r["image_id"] for r in rows.values()}), "phase_validation_pass": complete,
            "criterion": "every image committed after its independent audit passed",
            "status": "PASS" if complete and len(rows) == manifest["expected_requests"][phase] else "FAIL"}


# ------------------------------------------------------------ steps 3, 4 (gate)
def stage_gate(run):
    H = load_harness()
    manifest, config, keys, methods, complete = phase_keys(run, "gate")
    rows = committed_rows(run, "gate")
    ex = execution(run, "gate", rows, manifest, complete)
    g3 = compare(rows, H.NEW_R100, "fullload_async", ["generated_token_ids"], keys)
    g3 = {"gate": "step3_keep100_equals_fullload",
          "criterion": "generated tokens identical for every request (new IMPRESS path at 100 % retention on the reordered store vs FullLoad)", **g3}
    g4 = {"gate": "step4_matches_previous_main_gate", **matches_previous_main(
        H, rows, keys, ["recompute", "fullload_async", H.OURS_ASYNC], {"recompute", "fullload_async", H.OURS_ASYNC})}
    sync = compare(rows, H.NEW_ASYNC, H.NEW, ["generated_token_ids", "ssd_read_bytes", "probe_read_bytes", "k_read_bytes", "v_read_bytes"], keys)
    sync = {"gate": "new_arm_async_equals_sync", "criterion": "generated tokens and bytes read (total and by kind) identical for every request", **sync}
    for name, obj in (("gate_execution", ex), ("step3_keep100_equals_fullload", g3), ("step4_matches_previous_main_gate", g4),
                      ("new_arm_async_equals_sync", sync)):
        write(run / "gates" / f"{name}.json", obj)
    summary = {"execution": ex["status"], "step3": g3["status"], "step4": g4["status"], "async_equals_sync": sync["status"]}
    print(json.dumps(summary, indent=1))
    print(json.dumps({"step3": {k: g3[k] for k in ("requests", "equal", "mismatch_count")},
                      "step4": [{k: m[k] for k in ("method", "requests", "prompt_equal", "mismatch_count")} for m in g4["methods"]],
                      "async_equals_sync": {k: sync[k] for k in ("requests", "equal", "mismatch_count")}}, indent=1))
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
    server = Server(runner, ratio=0.25, probe=3, alpha=0.6, max_new_tokens=16)
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
            # diagnostic: the same image inside another (longer) prompt
            prefix2, _, _ = capture(other["prompt"])
            stored = stats["prefix_rows"]
            n_sep = len(src["geometry"]["newline_idx"]); cand = np.ones(stored.shape[2], dtype=bool); cand[src["geometry"]["newline_idx"]] = False
            k = io_.keep_count(int(cand.sum()))
            jac = []
            for li in range(stored.shape[0]):
                for h in range(stored.shape[1]):
                    a = set(np.argsort(-stored[li, h][cand], kind="stable")[:k].tolist())
                    b = set(np.argsort(-prefix2[li, h][cand], kind="stable")[:k].tolist())
                    jac.append(len(a & b) / len(a | b))
            denom = np.maximum(np.abs(stored), 1e-12)
            other_prompt = {"prefix_rows_equal": bool(np.array_equal(prefix2, stored)),
                            "max_abs_diff": float(np.abs(prefix2 - stored).max()),
                            "max_rel_diff_where_stored_ge_1e-3": float((np.abs(prefix2 - stored) / denom)[np.abs(stored) >= 1e-3].max()),
                            "mean_abs_diff": float(np.abs(prefix2 - stored).mean()),
                            "prefix_only_topk_jaccard_mean": float(np.mean(jac)), "prefix_only_topk_jaccard_min": float(np.min(jac)),
                            "suffix_tokens_stored_request": len(src["suffix_input_ids"]), "suffix_tokens_other_request": len(other["suffix_input_ids"])}
            images.append({"image_id": cp.parent.name, "source_dialog_id": src["dialog_id"], "source_method_id": src["method_id"],
                           "shape": list(stored.shape), "separators": n_sep, "same_prompt": same, "other_prompt_diagnostic": other_prompt,
                           "status": "PASS" if same["prefix_rows_equal"] and same["order_scores_equal"] else "FAIL"})
            print(json.dumps({"image": cp.parent.name, **same, "other_prompt_max_abs_diff": other_prompt["max_abs_diff"],
                              "other_prompt_topk_jaccard_mean": other_prompt["prefix_only_topk_jaccard_mean"]}), flush=True)
    finally:
        server.close()
    out = {"gate": "step2_prefix_rows_match_full_recompute",
           "criterion": "for every gate image, the stored prefix-row importance (and the order scores) equal, element for element, "
                        "the values captured from a second full computation from pixels with the same turn-1 prompt",
           "images": images, "status": "PASS" if images and all(x["status"] == "PASS" for x in images) else "FAIL"}
    write(run / "gates" / "step2_prefix_rows.json", out)
    print(json.dumps({"step2": out["status"], "images": len(images)}))
    return out["status"]


# ------------------------------------------------------------------- step 6
def stage_dev(run):
    H = load_harness()
    manifest, config, keys, methods, complete = phase_keys(run, "dev")
    rows = committed_rows(run, "dev")
    ex = execution(run, "dev", rows, manifest, complete)
    g4 = {"gate": "step4_matches_previous_main_dev", **matches_previous_main(
        H, rows, keys, ["recompute", "fullload_async", H.OURS_ASYNC, "impress_async"], {"recompute", "fullload_async", H.OURS_ASYNC})}
    for name, obj in (("dev_execution", ex), ("step4_matches_previous_main_dev", g4)):
        write(run / "gates" / f"{name}.json", obj)
    summary = {"execution": ex["status"], "step4_dev": g4["status"]}
    print(json.dumps(summary, indent=1))
    print(json.dumps({"step4": [{k: m[k] for k in ("method", "blocking", "requests", "prompt_equal", "mismatch_count")} for m in g4["methods"]]}, indent=1))
    return "PASS" if all(v == "PASS" for v in summary.values()) else "FAIL"


# -------------------------------------------------------------------- admit
REQUIRED = ("step1_public_equivalence", "step2_prefix_rows", "gate_execution", "step3_keep100_equals_fullload",
            "step4_matches_previous_main_gate", "new_arm_async_equals_sync", "step5_cpu_tests", "dev_execution",
            "step4_matches_previous_main_dev")


def stage_admit(run):
    H = load_harness()
    statuses = {}
    for name in REQUIRED:
        p = run / "gates" / f"{name}.json"
        statuses[name] = json.loads(p.read_text())["status"] if p.exists() else "MISSING"
    ok = all(v == "PASS" for v in statuses.values())
    out = {"status": "PASS" if ok else "FAIL", "gates": statuses, "config_sha256": H.sha(run / "config.json"),
           "decided_unix": time.time(), "rule": "main runs only if every gate of steps 1-6 is PASS"}
    write(run / "gates" / "admission.json", out)
    print(json.dumps(out, indent=1))
    return out["status"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--stage", choices=("cpu", "gate", "prefix", "dev", "admit"), required=True)
    ap.add_argument("--extra-python", default=None, help="second interpreter for the CPU suite (cpu stage)")
    args = ap.parse_args()
    run = args.run.resolve()
    status = {"cpu": lambda: stage_cpu(run, args.extra_python), "gate": lambda: stage_gate(run), "prefix": lambda: stage_prefix(run),
              "dev": lambda: stage_dev(run), "admit": lambda: stage_admit(run)}[args.stage]()
    if status != "PASS":
        raise SystemExit(f"{args.stage}: {status}")


if __name__ == "__main__":
    main()

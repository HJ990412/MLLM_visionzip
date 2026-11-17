#!/usr/bin/env python3
"""Gates for the MT-GQA asynchronous-read run (contract docs/mt_gqa_async_contract.md).

  --stage cpu    step 1: the whole CPU suite in this interpreter (and, when
                 given, in a second one); no failure outside the recorded
                 environment-specific baseline
  --stage gate   steps 2-4 on the committed rows of the ``gate`` phase:
                 G2 async == sync (tokens and bytes), G3 IMPRESS at 100 %
                 retention == FullLoad (tokens), G4 ReComp / FullLoad / Ours
                 == the existing HOLD records (tokens)
  --stage dev    step 5 on the committed rows of the ``dev`` phase: G2 and G4
                 again over every DEV request, plus the pre-fixed
                 sync-vs-async TTFT criterion
  --stage admit  writes gates/admission.json (PASS only if every gate file is
                 PASS); the main phase refuses to start without it

Nothing here changes a criterion after the fact: thresholds are constants of
this file, frozen with the run's source hash.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mmimpress.recompute_kv25 import paired_bootstrap

SEED = 1234
RESAMPLES = 10000
# Environment-specific failures that exist without this work (recorded before
# any edit, see the contract).  transformers 4.57.6 lacks the GLM-4.6V API the
# three GLM tests exercise; 5.13.1 changed the two vision-capture paths.
BASELINE_FAILURES = {
    "4": {"setUpClass (test_glm46v_runner.NativeProcessorRunnerTests)",
          "test_actual_native_adjacent_pair_partial_rope_keeps_tail_bits (test_glm46v_validation.GlmValidationTests)",
          "test_actual_native_sdpa_hook_preserves_output_and_qk_score (test_glm46v_vision.GlmVisionTests)"},
    "5": {"test_single_stock_forward_capture_equals_dense_visionzip_formula (test_qwen25_vision.StockVisionCaptureTests)",
          "test_real_tiny_clip_matches_full_penultimate_attention (test_visdial_turn1_piggyback_core.VisionForwardCaptureTests)"},
}


def load_harness():
    spec = importlib.util.spec_from_file_location("_mtgqa_async_harness", ROOT / "scripts/187_eval_mt_gqa_async.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=False) + "\n")


def committed_rows(root, phase, methods=None, images=None):
    """{(method, dialog_id, turn): row} from committed image attempts only."""
    rows = {}
    for cp in sorted((Path(root) / phase / "images").glob("*/COMMITTED.json")):
        if images is not None and cp.parent.name not in images:
            continue
        doc = json.loads(cp.read_text())
        with (cp.parent / doc["attempt_id"] / "raw.jsonl").open() as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if methods is None or r["method_id"] in methods:
                    rows[(r["method_id"], r["dialog_id"], r["turn_id"])] = r
    return rows


# ---------------------------------------------------------------------- cpu
def run_suite(python):
    env_cmd = [python, "-c", "import sys, transformers, torch; print(sys.version.split()[0], transformers.__version__, torch.__version__)"]
    version = subprocess.run(env_cmd, capture_output=True, text=True, cwd=ROOT).stdout.split()
    p = subprocess.run([python, "-m", "unittest", "discover", "-s", "tests"], capture_output=True, text=True, cwd=ROOT,
                       env={**__import__("os").environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "2",
                            "MKL_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"})
    text = p.stdout + p.stderr
    failed = sorted(set(m.group(2) for m in re.finditer(r"^(ERROR|FAIL): (.+)$", text, re.M)))
    ran = re.search(r"^Ran (\d+) tests", text, re.M)
    baseline = BASELINE_FAILURES.get(version[1].split(".")[0], set()) if len(version) > 1 else set()
    new = [f for f in failed if f not in baseline]
    return {"python": python, "python_version": version[0] if version else None, "transformers": version[1] if len(version) > 1 else None,
            "torch": version[2] if len(version) > 2 else None, "ran": int(ran.group(1)) if ran else None, "failed": failed,
            "baseline_failures_for_this_environment": sorted(baseline), "new_failures": new,
            "status": "PASS" if ran and not new else "FAIL", "tail": text[-1500:]}


def stage_cpu(run, extra_python):
    results = [run_suite(sys.executable)]
    if extra_python:
        results.append(run_suite(extra_python))
    new_tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_mt_gqa_async.py"],
                               capture_output=True, text=True, cwd=ROOT)
    ran = re.search(r"^Ran (\d+) tests", new_tests.stderr, re.M)
    out = {"gate": "step1_cpu_tests", "suites": results,
           "new_unit_tests": {"file": "tests/test_mt_gqa_async.py", "ran": int(ran.group(1)) if ran else None,
                              "ok": new_tests.returncode == 0},
           "criterion": "no failure outside the environment-specific baseline recorded before the change",
           "status": "PASS" if all(r["status"] == "PASS" for r in results) and new_tests.returncode == 0 else "FAIL"}
    write(run / "gates" / "step1_cpu_tests.json", out)
    print(json.dumps({"status": out["status"], "suites": [{k: r[k] for k in ("transformers", "ran", "failed", "new_failures")} for r in results],
                      "new_unit_tests": out["new_unit_tests"]}, indent=1))
    return out["status"]


# ----------------------------------------------------------- shared checks
def compare(rows, a, b, fields, expected_keys):
    """Per-request equality of ``fields`` between arms ``a`` and ``b``."""
    mism = []; n = 0; equal = {f: 0 for f in fields}
    for (dialog, turn) in sorted(expected_keys):
        ra, rb = rows.get((a, dialog, turn)), rows.get((b, dialog, turn))
        if ra is None or rb is None:
            mism.append({"dialog_id": dialog, "turn_id": turn, "reason": "missing"}); continue
        n += 1; bad = []
        for f in fields:
            va = ra["result"]["generated_token_ids"] if f == "generated_token_ids" else ra[f]
            vb = rb["result"]["generated_token_ids"] if f == "generated_token_ids" else rb[f]
            if va == vb:
                equal[f] += 1
            else:
                bad.append({"field": f, a: va, b: vb})
        if bad:
            mism.append({"dialog_id": dialog, "turn_id": turn, "image_id": ra["image_id"], "differences": bad,
                         "prompt_equal": ra["prompt"] == rb["prompt"]})
    return {"a": a, "b": b, "requests": n, "expected": len(expected_keys), "equal": equal, "mismatch_count": len(mism), "mismatches": mism[:50],
            "status": "PASS" if n == len(expected_keys) and not mism else "FAIL"}


def g2_async_equals_sync(H, rows, keys):
    pairs = [("fullload_async", "fullload", ["generated_token_ids", "ssd_read_bytes", "k_read_bytes", "v_read_bytes"]),
             ("impress_async", "impress", ["generated_token_ids", "ssd_read_bytes", "probe_read_bytes", "k_read_bytes", "v_read_bytes"]),
             (H.OURS_ASYNC, H.OURS, ["generated_token_ids", "ssd_read_bytes", "packed_read_bytes"])]
    out = [compare(rows, a, b, fields, keys) for a, b, fields in pairs]
    return {"gate": "step2_async_equals_sync", "criterion": "generated tokens and bytes read identical for every request",
            "pairs": out, "status": "PASS" if all(p["status"] == "PASS" for p in out) else "FAIL"}


def g4_matches_hold(H, rows, keys, methods):
    images = {r["image_id"] for r in rows.values()}
    v2 = committed_rows(H.V2_RUN / "hold", "main", {"recompute", "fullload"}, images)
    pyr = committed_rows(H.PYRBETA_RUN / "hold", "main", {H.OURS}, images)
    refs = {"recompute": (v2, "recompute"), "fullload": (v2, "fullload"), "fullload_async": (v2, "fullload"),
            H.OURS: (pyr, H.OURS), H.OURS_ASYNC: (pyr, H.OURS)}
    out = []
    for m in methods:
        if m not in refs:
            continue
        ref_rows, ref_method = refs[m]
        mism = []; n = 0; prompt_equal = 0
        for dialog, turn in sorted(keys):
            new, old = rows.get((m, dialog, turn)), ref_rows.get((ref_method, dialog, turn))
            if new is None or old is None:
                mism.append({"dialog_id": dialog, "turn_id": turn, "reason": "missing in " + ("this run" if new is None else "HOLD")}); continue
            n += 1; prompt_equal += new["prompt"] == old["prompt"]
            if new["result"]["generated_token_ids"] != old["result"]["generated_token_ids"]:
                mism.append({"dialog_id": dialog, "turn_id": turn, "image_id": new["image_id"], "new": new["prediction"], "hold": old["prediction"],
                             "new_tokens": new["result"]["generated_token_ids"], "hold_tokens": old["result"]["generated_token_ids"],
                             "prompt_equal": new["prompt"] == old["prompt"]})
        out.append({"method": m, "hold_method": ref_method, "requests": n, "expected": len(keys), "prompt_equal": prompt_equal,
                    "mismatch_count": len(mism), "mismatches": mism[:50], "status": "PASS" if n == len(keys) and not mism else "FAIL"})
    return {"gate": "step4_matches_existing_hold", "criterion": "generated tokens identical to the HOLD record of the same (dialogue, turn)",
            "references": {"recompute_fullload": str((H.V2_RUN / "hold" / "main").relative_to(ROOT)), "ours": str((H.PYRBETA_RUN / "hold" / "main").relative_to(ROOT))},
            "methods": out, "status": "PASS" if out and all(x["status"] == "PASS" for x in out) else "FAIL"}


def phase_keys(run, phase):
    manifest = json.loads((run / "manifest.json").read_text())
    config = json.loads((run / "config.json").read_text())
    keys = {(d, t) for d in manifest["phase_dialogue_ids"][phase] for t in (1, 2, 3)}
    methods = config["phases"][phase]["methods"]
    validation = run / phase / "validation.json"
    complete = validation.exists() and json.loads(validation.read_text())["status"] == "PASS"
    return manifest, config, keys, methods, complete


# --------------------------------------------------------------------- gate
def stage_gate(run):
    H = load_harness()
    manifest, config, keys, methods, complete = phase_keys(run, "gate")
    rows = committed_rows(run, "gate")
    execution = {"gate": "gate_phase_execution", "requests": len(rows), "expected": manifest["expected_requests"]["gate"],
                 "phase_validation_pass": complete,
                 "status": "PASS" if complete and len(rows) == manifest["expected_requests"]["gate"] else "FAIL"}
    g2 = g2_async_equals_sync(H, rows, keys)
    g3 = compare(rows, "impress_r100", "fullload", ["generated_token_ids"], keys)
    g3 = {"gate": "step3_impress_keep100_equals_fullload", "criterion": "generated tokens identical for every request (IMPRESS path at 100 % retention on the reordered store vs FullLoad)", **g3}
    g4 = g4_matches_hold(H, rows, keys, methods)
    for name, obj in (("gate_execution", execution), ("step2_async_equals_sync_gate", g2), ("step3_reorder_consistency", g3), ("step4_matches_hold_gate", g4)):
        write(run / "gates" / f"{name}.json", obj)
    summary = {"execution": execution["status"], "step2": g2["status"], "step3": g3["status"], "step4": g4["status"]}
    print(json.dumps(summary, indent=1))
    print(json.dumps({"step2": [{k: p[k] for k in ("a", "b", "requests", "equal", "mismatch_count")} for p in g2["pairs"]],
                      "step3": {k: g3[k] for k in ("requests", "equal", "mismatch_count")},
                      "step4": [{k: m[k] for k in ("method", "requests", "prompt_equal", "mismatch_count")} for m in g4["methods"]]}, indent=1))
    return "PASS" if all(v == "PASS" for v in summary.values()) else "FAIL"


# ---------------------------------------------------------------------- dev
def paired_ttft(rows, a, b, keys, field="ttft_ms"):
    per = defaultdict(lambda: [0.0, 0.0, 0]); diffs = []
    for dialog, turn in sorted(keys):
        if turn == 1:
            continue
        ra, rb = rows[(a, dialog, turn)], rows[(b, dialog, turn)]
        e = per[ra["image_id"]]; e[0] += ra[field]; e[1] += rb[field]; e[2] += 1
        diffs.append(ra[field] - rb[field])
    d = paired_bootstrap([tuple(e) for e in per.values()], RESAMPLES, SEED)
    d.update(n=len(diffs), images=len(per), median=float(np.median(diffs)), a_faster=int(sum(x < 0 for x in diffs)))
    return d


def stage_dev(run):
    H = load_harness()
    manifest, config, keys, methods, complete = phase_keys(run, "dev")
    rows = committed_rows(run, "dev")
    execution = {"gate": "dev_phase_execution", "requests": len(rows), "expected": manifest["expected_requests"]["dev"],
                 "images": len({r["image_id"] for r in rows.values()}), "phase_validation_pass": complete,
                 "status": "PASS" if complete and len(rows) == manifest["expected_requests"]["dev"] else "FAIL"}
    g2 = g2_async_equals_sync(H, rows, keys)
    g4 = g4_matches_hold(H, rows, keys, methods)
    effect = []
    if execution["status"] == "PASS":
        for label, a, b in (("FullLoad", "fullload_async", "fullload"), ("IMPRESS", "impress_async", "impress"), ("Ours", H.OURS_ASYNC, H.OURS)):
            d = paired_ttft(rows, a, b, keys)
            follow = lambda m: [r for (mm, _, t), r in rows.items() if mm == m and t > 1]      # noqa: E731
            sync_mean = float(np.mean([r["ttft_ms"] for r in follow(b)])); async_mean = float(np.mean([r["ttft_ms"] for r in follow(a)]))
            effect.append({"method": label, "async_arm": a, "sync_arm": b, "sync_ttft_mean_ms": sync_mean, "async_ttft_mean_ms": async_mean,
                           "paired_diff_ms": d["mean"], "ci95_low_ms": d["ci_low"], "ci95_high_ms": d["ci_high"], "median_diff_ms": d["median"],
                           "n": d["n"], "images": d["images"], "async_faster_requests": d["a_faster"],
                           "status": "PASS" if d["mean"] < 0 and d["ci_high"] < 0 else "FAIL"})
    g5 = {"gate": "step5_dev_async_effect",
          "criterion": "for each of FullLoad / IMPRESS / Ours: image-level paired bootstrap (10,000 resamples, seed 1234) of followup TTFT, async minus sync, mean < 0 and 95 % CI upper bound < 0",
          "methods": effect, "status": "PASS" if effect and all(e["status"] == "PASS" for e in effect) else "FAIL"}
    for name, obj in (("dev_execution", execution), ("step2_async_equals_sync_dev", g2), ("step4_matches_hold_dev", g4), ("step5_dev_async_effect", g5)):
        write(run / "gates" / f"{name}.json", obj)
    summary = {"execution": execution["status"], "step2_dev": g2["status"], "step4_dev": g4["status"], "step5": g5["status"]}
    print(json.dumps(summary, indent=1))
    print(json.dumps({"step5": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in e.items()} for e in effect]}, indent=1))
    return "PASS" if all(v == "PASS" for v in summary.values()) else "FAIL"


# -------------------------------------------------------------------- admit
REQUIRED = ("step1_cpu_tests", "gate_execution", "step2_async_equals_sync_gate", "step3_reorder_consistency", "step4_matches_hold_gate",
            "dev_execution", "step2_async_equals_sync_dev", "step4_matches_hold_dev", "step5_dev_async_effect")


def stage_admit(run):
    H = load_harness()
    statuses = {}
    for name in REQUIRED:
        p = run / "gates" / f"{name}.json"
        statuses[name] = json.loads(p.read_text())["status"] if p.exists() else "MISSING"
    ok = all(v == "PASS" for v in statuses.values())
    out = {"status": "PASS" if ok else "FAIL", "gates": statuses, "config_sha256": H.sha(run / "config.json"),
           "decided_unix": time.time(), "rule": "main runs only if every gate of steps 1-5 is PASS"}
    write(run / "gates" / "admission.json", out)
    print(json.dumps(out, indent=1))
    return out["status"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--stage", choices=("cpu", "gate", "dev", "admit"), required=True)
    ap.add_argument("--extra-python", default=None, help="second interpreter for the CPU suite (cpu stage)")
    args = ap.parse_args()
    run = args.run.resolve()
    status = {"cpu": lambda: stage_cpu(run, args.extra_python), "gate": lambda: stage_gate(run),
              "dev": lambda: stage_dev(run), "admit": lambda: stage_admit(run)}[args.stage]()
    if status != "PASS":
        raise SystemExit(f"{args.stage}: {status}")


if __name__ == "__main__":
    main()

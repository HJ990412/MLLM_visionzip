#!/usr/bin/env python3
"""Gates of the LLaVA-NeXT x MT-VQA main experiment
(contract docs/llava_next_mtvqa_main_contract.md, section 7).

  --stage tests    gate 1: the whole unit-test suite in the run environment and, as a
                   cross-check, in the transformers 5.13.1 environment; the new test file
  --stage scorer   gate 5: scripts/135_validate_mtvqa25k_scorer.py
  --stage gate     from the committed ``gate`` phase (4 images):
                   gate 2 (Ours: synchronous == asynchronous in tokens and bytes read;
                   the same requests executed twice give the same tokens) and
                   gate 3 (IMPRESS at 100 % == the FullLoad loader on the same reordered
                   store; IMPRESS synchronous == asynchronous in tokens and bytes read)
  --stage pilot    from the committed ``pilot`` phase (40 images): gate 4 (ReComp tokens ==
                   the v3 record of the same request; the three methods' turn-1 tokens ==
                   the v3 turn-1 tokens of the same image) and gate 6 (the pilot ran);
                   writes the pilot-based time estimate
  --stage admit    writes gates/admission.json (PASS only if every gate file is PASS for
                   this freeze); the main phase refuses to start without it

Criteria are constants of this file, frozen with the run's source hash.  A gate file that
already exists for this freeze is never re-judged by the driver.
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
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PY_CROSS = str(ROOT / ".venv-glm46v/bin/python")
NEW_TEST = "test_llava_next_mtvqa_main.py"
BYTE_FIELDS = ("ssd_read_bytes", "probe_read_bytes", "k_read_bytes", "v_read_bytes", "packed_read_bytes")
REQUIRED = ("tests", "scorer", "gate", "pilot")
# test failures that existed before this work, per transformers major version (the list of scripts/188, recorded 2026-11-16)
BASELINE_FAILURES = {
    "4": {"setUpClass (test_glm46v_runner.NativeProcessorRunnerTests)",
          "test_actual_native_adjacent_pair_partial_rope_keeps_tail_bits (test_glm46v_validation.GlmValidationTests)",
          "test_actual_native_sdpa_hook_preserves_output_and_qk_score (test_glm46v_vision.GlmVisionTests)"},
    "5": {"test_single_stock_forward_capture_equals_dense_visionzip_formula (test_qwen25_vision.StockVisionCaptureTests)",
          "test_real_tiny_clip_matches_full_penultimate_attention (test_visdial_turn1_piggyback_core.VisionForwardCaptureTests)"},
}

_H = None


def harness():
    """The evaluation script (imports the model code; loaded only by the stages that need it)."""
    global _H
    if _H is None:
        _H = load_script("_lnext_mtvqa_main_eval", "209_eval_llava_next_mtvqa_main.py")
    return _H


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False, default=str) + "\n")
    os.replace(tmp, path)


def frozen(run):
    config = json.loads((run / "config.json").read_text())
    config["_file_sha256"] = sha(run / "config.json")
    return config


def receipt(run, name, doc, config):
    out = run / "gates" / f"{name}.json"
    out.parent.mkdir(exist_ok=True)
    doc = dict(doc, config_sha256=config["_file_sha256"], written_unix=time.time())
    if out.exists():      # a superseded receipt is kept, never overwritten
        os.replace(out, out.with_name(f"{name}.superseded_{time.strftime('%Y%m%dT%H%M%S')}.json"))
    write_json(out, doc)
    print(json.dumps({"stage": name, "status": doc["status"]}, ensure_ascii=False), flush=True)
    return doc


# ---------------------------------------------------------------- committed io
def read_rows(path):
    import gzip
    path = Path(path)
    if not path.exists() and path.with_suffix(path.suffix + ".gz").exists():
        path = path.with_suffix(path.suffix + ".gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def committed_rows(run, phase):
    """{(method, image_id, turn): row} of every committed image of a phase."""
    rows = {}
    for cp in sorted((run / phase / "images").glob("*/COMMITTED.json")):
        doc = json.loads(cp.read_text())
        got = read_rows(cp.parent / doc["attempt_id"] / "raw.jsonl")
        assert len(got) == doc["requests"]
        for r in got:
            rows[(r["method_id"], r["image_id"], r["turn_id"])] = r
    return rows


def tokens(row):
    return row["result"]["generated_token_ids"]


def phase_state(run, phase, config):
    manifest = json.loads((run / "manifest.json").read_text())
    keys = [(i, t) for i in manifest["phase_images"][phase] for t in (1, 2, 3)]
    path = run / phase / "validation.json"
    v = json.loads(path.read_text()) if path.exists() else {"status": "MISSING"}
    ok = (v.get("status") == "PASS" and v.get("requests") == manifest["expected_requests"][phase] and not v.get("failed_images")
          and v.get("config_sha256") == config["_file_sha256"])
    execution = dict(v, status="PASS" if ok else "FAIL", expected=manifest["expected_requests"][phase],
                     criterion="every image of the phase committed after its independent audit passed, for this freeze")
    return manifest, keys, execution


def compare(rows, a, b, keys, fields=()):
    """Per-request equality of the generated tokens (and ``fields``) between arms ``a`` and ``b``."""
    out = {"arms": [a, b], "requests": 0, "followup_requests": 0, "tokens_differ": 0, "tokens_differ_same_prompt": 0, "prompt_differs": 0,
           "fields_differ": {f: 0 for f in fields}, "missing": 0, "examples": []}
    for image, turn in keys:
        ra, rb = rows.get((a, image, turn)), rows.get((b, image, turn))
        if ra is None or rb is None:
            out["missing"] += 1
            continue
        out["requests"] += 1; out["followup_requests"] += turn > 1
        same_prompt = ra["prompt"] == rb["prompt"]
        out["prompt_differs"] += not same_prompt
        bad = [f for f in fields if ra[f] != rb[f]]
        for f in bad:
            out["fields_differ"][f] += 1
        if tokens(ra) != tokens(rb):
            out["tokens_differ"] += 1; out["tokens_differ_same_prompt"] += same_prompt
            bad.append("generated_token_ids")
        if bad and len(out["examples"]) < 30:
            out["examples"].append({"image_id": image, "turn_id": turn, "fields": bad, "same_prompt": same_prompt,
                                    a: tokens(ra), b: tokens(rb), "predictions": [ra["prediction"], rb["prediction"]]})
    out["status"] = "PASS" if (out["requests"] == len(keys) and not out["missing"] and not out["tokens_differ"]
                               and not any(out["fields_differ"].values())) else "FAIL"
    return out


def load_reference(run, v3_phase):
    from mmimpress.mtvqa25k_packed import reference as REF
    return REF.load_index(run / "reference" / f"v3_{v3_phase}_index.jsonl.gz")


def against_v3(rows, keys, method, reference, v3_method, turns=(1, 2, 3), reference_turn=None):
    """Generated tokens of ``method`` against the v3 record's ``v3_method`` row of the same image and turn
    (``reference_turn`` = 1: against the v3 turn-1 row, used for the turn-1 continuity of the stored methods)."""
    out = {"method": method, "v3_method": v3_method, "requests": 0, "missing": 0, "mismatch_count": 0, "mismatches": []}
    for image, turn in keys:
        if turn not in turns:
            continue
        new = rows.get((method, image, turn)); ref = (reference.get(image) or {}).get((v3_method, reference_turn or turn))
        if new is None or ref is None:
            out["missing"] += 1
            continue
        out["requests"] += 1
        if tokens(new) != ref["generated_token_ids"]:
            out["mismatch_count"] += 1
            if len(out["mismatches"]) < 50:
                out["mismatches"].append({"image_id": image, "turn_id": turn, "new": new["prediction"], "v3": ref["prediction"],
                                          "new_tokens": tokens(new), "v3_tokens": ref["generated_token_ids"],
                                          "prompt_sha_equal": new["prompt_sha256"] == ref.get("prompt_sha256")})
    expected = sum(1 for _, t in keys if t in turns)
    out["status"] = "PASS" if out["requests"] == expected and not out["missing"] and not out["mismatch_count"] else "FAIL"
    return out


def sync_async_timing(rows, keys, pairs):
    """Reference only (not a criterion): synchronous vs asynchronous follow-up TTFT."""
    out = {}
    for name, a, b in pairs:
        fu = [(i, t) for i, t in keys if t > 1 and (a, i, t) in rows and (b, i, t) in rows]
        if not fu:
            continue
        sa = np.array([rows[(a, i, t)]["ttft_ms"] for i, t in fu]); sb = np.array([rows[(b, i, t)]["ttft_ms"] for i, t in fu])
        out[name] = {"sync_mean_ms": float(sa.mean()), "async_mean_ms": float(sb.mean()), "paired_mean_difference_ms": float((sb - sa).mean()),
                     "requests": int(sa.size)}
    return out


# ------------------------------------------------------------------ gate 1
def run_suite(python):
    env_cmd = [python, "-c", "import sys, transformers, torch; print(sys.version.split()[0], transformers.__version__, torch.__version__)"]
    version = subprocess.run(env_cmd, capture_output=True, text=True, cwd=ROOT).stdout.split()
    p = subprocess.run([python, "-m", "unittest", "discover", "-s", "tests"], capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "2",
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


def stage_tests(run, config):
    if "_dev_" in run.name and os.environ.get("LNEXT_MAIN_DEV_SKIP_TESTS") == "1":      # development rehearsals only; never a measurement
        return receipt(run, "tests", {"gate": "gate1_existing_tests", "status": "PASS", "suites": [],
                                      "new_unit_tests": {"file": f"tests/{NEW_TEST}", "ran": 0, "ok": True},
                                      "rehearsal_note": "test suite skipped in a development rehearsal"}, config)
    suites = [run_suite(sys.executable)]
    if Path(PY_CROSS).exists():
        suites.append(run_suite(PY_CROSS))
    p = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", NEW_TEST], capture_output=True, text=True, cwd=ROOT)
    ran = re.search(r"^Ran (\d+) tests", p.stdout + p.stderr, re.M)
    ok = all(r["status"] == "PASS" for r in suites) and p.returncode == 0 and bool(ran)
    return receipt(run, "tests", {
        "gate": "gate1_existing_tests", "status": "PASS" if ok else "FAIL", "suites": suites,
        "new_unit_tests": {"file": f"tests/{NEW_TEST}", "ran": int(ran.group(1)) if ran else None, "ok": p.returncode == 0,
                           "tail": (p.stdout + p.stderr)[-800:] if p.returncode else None},
        "criterion": "no failing test other than the failures that existed before this work (transformers 4.57.6: the 3 GLM-4.6V tests; "
                     "transformers 5.13.1: 2 vision-capture tests); the new test file passes"}, config)


# ------------------------------------------------------------------ gate 5
def stage_scorer(run, config):
    out = run / f"scorer_validation_{time.strftime('%Y%m%dT%H%M%S')}"
    p = subprocess.run([sys.executable, str(ROOT / "scripts/135_validate_mtvqa25k_scorer.py"), "--output", str(out)],
                       capture_output=True, text=True, cwd=ROOT)
    result = json.loads((out / "validation.json").read_text()) if (out / "validation.json").exists() else {}
    ok = p.returncode == 0 and result.get("status") == "PASS" and result.get("per_question_mismatches") == 0 and result.get("normalization_mismatches") == 0
    return receipt(run, "scorer", {
        "gate": "gate5_scorer_validation", "status": "PASS" if ok else "FAIL", "returncode": p.returncode, "output": str(out.relative_to(ROOT)),
        "result": {k: result.get(k) for k in ("status", "revision", "synthetic_fixtures", "actual_existing_raw_requests",
                                              "per_question_mismatches", "normalization_mismatches")},
        "stderr_tail": p.stderr[-1500:] if p.returncode else None,
        "criterion": "scripts/135_validate_mtvqa25k_scorer.py exits 0 with status PASS: the ported official evaluator, the independent "
                     "leave-one-out count and the syntax-translated upstream evaluator agree on every fixture and on the existing raw record"}, config)


# ------------------------------------------------------- gates 2, 3 (4 images)
def stage_gate(run, config):
    H = harness()
    manifest, keys, execution = phase_state(run, "gate", config)
    if execution["status"] != "PASS":
        return receipt(run, "gate", {"status": "FAIL", "execution": execution, "failed": ["execution"]}, config)
    rows = committed_rows(run, "gate")
    g2a = dict(compare(rows, H.ORIG, H.ORIG_ASYNC, keys, ("ssd_read_bytes", "packed_read_bytes")),
               criterion="Ours: 동기 읽기와 비동기 읽기의 생성 토큰과 읽기 바이트가 모든 요청에서 일치")
    g2b = dict(compare(rows, H.ORIG_ASYNC, H.ORIG_REPEAT, keys),
               criterion="Ours: 같은 요청을 두 번 돌린 생성 토큰이 모든 요청에서 일치")
    g2 = {"sync_vs_async": g2a, "executed_twice": g2b, "status": "PASS" if g2a["status"] == g2b["status"] == "PASS" else "FAIL"}
    g3a = dict(compare(rows, H.NEW_R100, H.FULL_REORDERED, keys),
               criterion="IMPRESS: 유지 비율 100%의 생성 토큰이 같은 재배치 store를 읽는 FullLoad loader의 생성 토큰과 모든 요청에서 일치")
    g3b = dict(compare(rows, H.NEW, H.NEW_ASYNC, keys, BYTE_FIELDS),
               criterion="IMPRESS: 비동기 읽기와 동기 읽기의 생성 토큰과 읽기 바이트(전체와 종류별)가 모든 요청에서 일치")
    g3 = {"retention_100_vs_fullload_loader_on_reordered_store": g3a, "sync_vs_async": g3b,
          "status": "PASS" if g3a["status"] == g3b["status"] == "PASS" else "FAIL"}
    parts = {"gate2_ours": g2, "gate3_impress": g3}
    failed = [k for k, v in parts.items() if v["status"] != "PASS"]
    # reported only: continuity with the v3 record on the gate images (gate 4 is judged on the pilot)
    ref = load_reference(run, "main")
    reported = {"recomp_vs_v3_main": against_v3(rows, keys, H.RECOMP, ref, H.V3_RECOMP),
                "turn1_vs_v3": {m: against_v3(rows, keys, m, ref, H.V3_RECOMP, turns=(1,), reference_turn=1) for m in H.PHASE_METHODS["gate"]},
                "fullload_reordered_vs_raster_fullload_tokens_differ": compare(rows, H.FULL_REORDERED, H.FULL_ASYNC, keys)["tokens_differ"]}
    return receipt(run, "gate", {
        "status": "PASS" if not failed else "FAIL", "failed": failed, "images": manifest["phase_images"]["gate"], "execution": execution, **parts,
        "reported_not_judged": reported,
        "reference_sync_vs_async_followup_ttft": sync_async_timing(rows, keys, (("IMPRESS", H.NEW, H.NEW_ASYNC), ("Ours", H.ORIG, H.ORIG_ASYNC)))}, config)


# ------------------------------------------------------ gates 4, 6 (pilot 40)
def stage_pilot(run, config):
    H = harness()
    manifest, keys, execution = phase_state(run, "pilot", config)
    if execution["status"] != "PASS":
        return receipt(run, "pilot", {"status": "FAIL", "execution": execution, "failed": ["execution"]}, config)
    rows = committed_rows(run, "pilot")
    ref_main = load_reference(run, "main"); ref_pilot = load_reference(run, "pilot40")
    recomp = dict(against_v3(rows, keys, H.RECOMP, ref_main, H.V3_RECOMP),
                  criterion="파일럿에서 돌린 ReComp의 생성 토큰이 v3 기록(본실험 단계)의 같은 요청과 세 턴 모두 일치")
    t1 = {m: against_v3(rows, keys, m, ref_main, H.V3_RECOMP, turns=(1,), reference_turn=1) for m in H.MAIN_METHODS}
    t1_status = "PASS" if all(v["status"] == "PASS" for v in t1.values()) else "FAIL"
    g4 = {"recomp_vs_v3": recomp, "turn1_of_the_three_methods_vs_v3": dict(t1, status=t1_status,
                                                                         criterion="FullLoad·IMPRESS·Ours의 T1 생성 토큰이 v3 기록의 같은 이미지 T1 토큰과 일치"),
          "status": "PASS" if recomp["status"] == "PASS" and t1_status == "PASS" else "FAIL"}
    failed = [] if g4["status"] == "PASS" else ["gate4_continuity"]
    # pilot-measured seconds per image -> the time estimate of the main phase (three arms instead of four)
    seconds = []; per_method_s = {}
    for cp in sorted((run / "pilot" / "images").glob("*/COMMITTED.json")):
        seconds.append(json.loads(cp.read_text())["image_seconds"])
    recomp_s = float(np.mean([sum(rows[(H.RECOMP, i, t)]["request_e2e_ms"] for t in (1, 2, 3)) for i in manifest["phase_images"]["pilot"]])) / 1000
    n_main = len(manifest["phase_images"]["main"])
    per_image_main = float(np.mean(seconds)) - recomp_s
    samples = [json.loads(s) for s in (run / "ssd_samples.jsonl").read_text().splitlines() if s.strip()] if (run / "ssd_samples.jsonl").exists() else []
    pilot_writes = [s["store_write_bytes"] for s in samples if s.get("phase") == "pilot" and s.get("label") == "image_committed"]
    estimate = {"basis": "pilot (40 images, four arms): mean seconds per image minus the ReComp requests' end-to-end time",
                "pilot_seconds_per_image_mean": float(np.mean(seconds)), "pilot_seconds_per_image_max": float(np.max(seconds)),
                "pilot_recomp_seconds_per_image": recomp_s, "main_seconds_per_image_estimate": per_image_main,
                "main_images": n_main, "main_hours_estimate": per_image_main * n_main / 3600, "main_days_estimate": per_image_main * n_main / 86400,
                "pilot_store_write_bytes_per_image_mean": float(np.mean(pilot_writes)) if pilot_writes else None,
                "main_store_write_tb_from_pilot_mean": float(np.mean(pilot_writes)) * n_main / 1e12 if pilot_writes else None,
                "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(ROOT / "results" / run.name / "prerun_pilot_estimate.json", estimate)
    H.STATUS.update(run, expected_main_hours=round(estimate["main_hours_estimate"], 1), expected_main_hours_basis="pilot (40 images)")
    return receipt(run, "pilot", {
        "status": "PASS" if not failed else "FAIL", "failed": failed, "image_count": len(manifest["phase_images"]["pilot"]),
        "execution": dict(execution, gate="gate6_pilot_executed"), "gate4_continuity": g4,
        "reported_not_judged": {
            "recomp_vs_v3_pilot40_phase": against_v3(rows, keys, H.RECOMP, ref_pilot, H.V3_RECOMP),
            "turn1_identical_across_arms_images": sum(len({tuple(tokens(rows[(m, i, 1)])) for m in H.PHASE_METHODS["pilot"]}) == 1
                                                      for i in manifest["phase_images"]["pilot"]),
            "fullload_followup_vs_v3_fullload": against_v3(rows, keys, H.FULL_ASYNC, ref_main, H.V3_FULL, turns=(2, 3))},
        "time_estimate": estimate}, config)


# -------------------------------------------------------------------- admit
def stage_admit(run, config):
    steps = {}
    for name in REQUIRED:
        path = run / "gates" / f"{name}.json"
        doc = json.loads(path.read_text()) if path.exists() else {"status": "MISSING"}
        ok = doc.get("status") == "PASS" and doc.get("config_sha256") == config["_file_sha256"]
        steps[name] = {"status": doc.get("status"), "bound_to_this_freeze": doc.get("config_sha256") == config["_file_sha256"], "ok": ok,
                       "failed": doc.get("failed")}
    pre = json.loads((run / "prerun_receipt.json").read_text()) if (run / "prerun_receipt.json").exists() else {"status": "MISSING"}
    steps["prerun"] = {"status": pre.get("status"), "bound_to_this_freeze": pre.get("config_sha256") == config["_file_sha256"],
                       "ok": pre.get("status") == "PASS" and pre.get("config_sha256") == config["_file_sha256"], "failed": pre.get("blocking")}
    status = "PASS" if all(s["ok"] for s in steps.values()) else "FAIL"
    return receipt(run, "admission", {
        "status": status, "steps": steps,
        "rule": "the main phase starts only if the pre-run checks passed, gates 1-5 are PASS and the pilot (gate 6) was executed; "
                "otherwise the driver records the failure and stops"}, config)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--stage", choices=("tests", "scorer", "gate", "pilot", "admit"), required=True)
    args = p.parse_args()
    run = args.run.resolve()
    assert run.parent == ROOT / "runs" and run.name.startswith("llava_next_mtvqa_main_")
    config = frozen(run)
    changed = harness().freeze_violations(config["source_sha256"])
    if changed and "_dev_" not in run.name:
        raise RuntimeError("FROZEN_SOURCE_CONFIG_MISMATCH: " + repr(changed[:10]))
    doc = {"tests": lambda: stage_tests(run, config), "scorer": lambda: stage_scorer(run, config), "gate": lambda: stage_gate(run, config),
           "pilot": lambda: stage_pilot(run, config), "admit": lambda: stage_admit(run, config)}[args.stage]()
    sys.exit(0 if doc["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

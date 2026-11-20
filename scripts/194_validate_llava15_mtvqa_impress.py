#!/usr/bin/env python3
"""Validation steps of the LLaVA-1.5 x MT-VQA IMPRESS run
(contract docs/llava15_mtvqa_impress_v2_contract.md, section 9).

Stages
  cpu       step 1: the whole unit-test suite in the run environment (and, as a
            cross-check, in the transformers 5.13.1 environment); no failure
            other than the recorded environment failures.
  preserve  step 2: the modified code runs the README-34 MT-GQA path
            (scripts/179 ``run_image``; recompute + four synchronous packed
            arms) on two MT-GQA images and reproduces the recorded tokens.
  gate      steps 3, 4, 5 from the committed ``gate`` phase.
  pilot     step 6 from the committed ``pilot`` phase (+ step 5 on the pilot).
  admit     step 7: writes gates/admission.json when 1-6 are PASS.

Every receipt is bound to the SHA256 of the frozen config.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import llava15_support as S  # noqa: E402

E = S.load_script("_l15_mtvqa_impress_eval", "193_eval_llava15_mtvqa_impress.py")
sha, atomic_json, read_rows = E.sha, E.atomic_json, E.read_rows
OURS, OURS_ASYNC = E.OURS, E.OURS_ASYNC
GiB = 1 << 30
PY_RUN = "/home/dblab/anaconda3/envs/mllm_ft/bin/python"
PY_CROSS = str(ROOT / ".venv-glm46v/bin/python")
# environment failures recorded before any change of this task (contract 9.1)
KNOWN_FAILURES = {
    "4.57.6": {"setUpClass (test_glm46v_runner.NativeProcessorRunnerTests)",
               "test_actual_native_adjacent_pair_partial_rope_keeps_tail_bits (test_glm46v_validation.GlmValidationTests)",
               "test_actual_native_sdpa_hook_preserves_output_and_qk_score (test_glm46v_vision.GlmVisionTests)"},
}
# transformers 5.13.1 cross-check: the two known environment failures, matched as substrings of the failing test id
KNOWN_CROSS_SUBSTRINGS = ("test_qwen25_vision", "test_visdial_turn1_piggyback_core")
SYNC_ASYNC = (("FullLoad", "fullload", "fullload_async"), ("IMPRESS", "impress", "impress_async"), ("Ours", OURS, OURS_ASYNC))
BYTE_FIELDS = ("ssd_read_bytes", "probe_read_bytes", "k_read_bytes", "v_read_bytes", "packed_read_bytes")


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
    atomic_json(out, doc)
    print(json.dumps({"stage": name, "status": doc["status"]}, ensure_ascii=False), flush=True)
    return doc


# ------------------------------------------------------------------ step 1
def parse_unittest_log(text):
    ran = re.search(r"^Ran (\d+) tests? in", text, re.M)
    bad = sorted(set(re.findall(r"^(?:ERROR|FAIL): (.+)$", text, re.M)))
    return {"ran": int(ran.group(1)) if ran else None, "failing": bad,
            "ok": bool(re.search(r"^OK", text, re.M)), "failed_line": (re.findall(r"^FAILED .*$", text, re.M) or [None])[-1]}


def new_failures(after, before):
    """Failing test ids of ``after`` that were not failing in ``before``."""
    return sorted(set(after) - set(before))


def stage_cpu(run, config):
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    env.pop("PYTHONPATH", None)
    before_log = run / "source_before" / "cpu_tests_before_mllm_ft.log"
    before = parse_unittest_log(before_log.read_text())
    out = {"before_change_4.57.6": dict(before, log=str(before_log.relative_to(run)))}
    status = "PASS"
    if set(before["failing"]) != KNOWN_FAILURES["4.57.6"]:
        status = "FAIL"; out["before_change_note"] = "the pre-change baseline differs from the failures recorded in the contract"
    for label, py in (("after_4.57.6", PY_RUN), ("after_5.13.1", PY_CROSS)):
        log = run / "gates" / f"cpu_tests_{label}.log"
        log.parent.mkdir(exist_ok=True)
        p = subprocess.run([py, "-m", "unittest", "discover", "-s", "tests"], cwd=ROOT, env=env, capture_output=True, text=True)
        log.write_text(p.stdout + p.stderr + f"\nexit={p.returncode}\n")
        parsed = parse_unittest_log(p.stdout + p.stderr)
        if label == "after_4.57.6":
            parsed["new_failures"] = new_failures(parsed["failing"], before["failing"])
        else:
            parsed["new_failures"] = [t for t in parsed["failing"] if not any(s in t for s in KNOWN_CROSS_SUBSTRINGS)]
        parsed["log"] = str(log.relative_to(run))
        new_tests = subprocess.run([py, "-m", "unittest", "discover", "-s", "tests", "-p", "test_llava15_mtvqa_impress.py"],
                                   cwd=ROOT, env=env, capture_output=True, text=True)
        parsed["new_test_file"] = parse_unittest_log(new_tests.stdout + new_tests.stderr)
        if parsed["ran"] is None or parsed["new_failures"] or not parsed["new_test_file"]["ok"]:
            status = "FAIL"
        out[label] = parsed
    return receipt(run, "cpu", dict(out, status=status, step="1 CPU tests",
                                    rule="no failing test other than the environment failures recorded before the change"), config)


# ------------------------------------------------------------------ step 2
def stage_preserve(run, config):
    import numpy as np
    import torch
    old_run = E.L15_RUN
    H = S.load_script("_l15_hold_179_for_preserve", "179_eval_llava15_mt_gqa_hold.py")
    manifest = json.loads((old_run / "manifest.json").read_text())
    old_config = json.loads((old_run / "config.json").read_text())
    images = []
    for d in manifest["dialogues"]:
        if d["image_id"] not in images:
            images.append(d["image_id"])
    images = images[:2]
    assert images == ["n229548", "n181355"], images
    pres = run / "preserve"
    pres.mkdir(exist_ok=False)
    shutil.copyfile(old_run / "manifest.json", pres / "manifest.json")
    cfg = dict(old_config, source_sha256=E.source_hashes(), preserve_note="README-34 arms and vectors; current source")
    atomic_json(pres / "config.json", cfg)
    atomic_json(pres / "storage_plan.json", {"minimum_reserve_bytes": 30 * GiB, "peak_working_set_bytes": 4 * GiB,
                                             "largest_image_bundle_bound_bytes": GiB, "maximum_new_payload_bytes": 20 * GiB, "status": "PASS"})
    cfg.update(_file_sha256=sha(pres / "config.json"), _manifest_sha256=sha(pres / "manifest.json"))
    assert cfg["methods"] == S.METHODS
    torch.set_num_threads(2); torch.manual_seed(1234); np.random.seed(1234); torch.set_grad_enabled(False)
    H.assert_gpu_exclusive()
    from mmimpress.llava15.runner import load_runner
    from mmimpress.serve import Server
    runner = load_runner(1234)
    server = Server(runner, max_new_tokens=16)
    S.warmup(runner, server)
    ordinals = {d["dialog_id"]: i for i, d in enumerate(manifest["dialogues"])}
    compared = 0; mismatches = []; per_method = {}
    for image in images:
        ds = [d for d in manifest["dialogues"] if d["image_id"] == image]
        H.run_image(runner, server, ds, ordinals, "main", pres, manifest, cfg)
        commit = json.loads((pres / "main" / "images" / image / "COMMITTED.json").read_text())
        new = {(r["dialog_id"], r["method_id"], r["turn_id"]): r for r in read_rows(pres / "main" / "images" / image / commit["attempt_id"] / "raw.jsonl")}
        old_commit = json.loads((old_run / "main" / "images" / image / "COMMITTED.json").read_text())
        old_raw = old_run / "main" / "images" / image / old_commit["attempt_id"] / "raw.jsonl"
        assert sha(old_raw) == old_commit["raw_sha256"]
        old = {(r["dialog_id"], r["method_id"], r["turn_id"]): r for r in read_rows(old_raw)}
        assert set(new) == set(old) and len(new) == len(ds) * 3 * len(S.METHODS)
        for key in sorted(new):
            a, b = new[key]["result"]["generated_token_ids"], old[key]["result"]["generated_token_ids"]
            compared += 1
            c = per_method.setdefault(key[1], {"compared": 0, "mismatched": 0})
            c["compared"] += 1
            if a != b or new[key]["prompt"] != old[key]["prompt"] or new[key]["ssd_read_bytes"] != old[key]["ssd_read_bytes"]:
                c["mismatched"] += 1
                mismatches.append({"dialog_id": key[0], "method_id": key[1], "turn_id": key[2], "new": a, "recorded": b})
    status = "PASS" if compared == 495 and not mismatches else "FAIL"
    return receipt(run, "preserve", {"status": status, "step": "2 path preservation", "images": images, "requests_compared": compared,
                                     "mismatches": len(mismatches), "mismatch_examples": mismatches[:20], "per_method": per_method,
                                     "reference": str((old_run / "main").relative_to(ROOT)),
                                     "rule": "generated token ids (and prompt, read bytes) equal the README-34 main record for every request"}, config)


# -------------------------------------------------------- committed phase io
def phase_rows(run, phase, manifest):
    """{image_id: {(method, turn): row}} of every committed image of a phase."""
    out = {}
    for image in manifest["phase_images"][phase]:
        folder = run / phase / "images" / image
        commit = json.loads((folder / "COMMITTED.json").read_text())
        rows = read_rows(folder / commit["attempt_id"] / "raw.jsonl")
        assert len(rows) == commit["requests"]
        out[image] = {(r["method_id"], r["turn_id"]): r for r in rows}
    return out


def tokens(row):
    return row["result"]["generated_token_ids"]


# ---------------------------------------------------------------- steps 3-5
def compare_sync_async(rows_by_image, sync, async_):
    """Step 3 for one family: tokens and read bytes (total and by kind)."""
    compared = 0; bad = []
    for image, rows in rows_by_image.items():
        for turn in (1, 2, 3):
            a, b = rows[(sync, turn)], rows[(async_, turn)]
            compared += 1
            diff = [f for f in BYTE_FIELDS if a[f] != b[f]]
            if tokens(a) != tokens(b):
                diff.append("generated_token_ids")
            if diff:
                bad.append({"image_id": image, "turn_id": turn, "fields": diff})
    return {"compared": compared, "mismatched": len(bad), "examples": bad[:20]}


def compare_tokens(rows_by_image, a, b, logits_of=None):
    """Generated tokens of arm ``a`` vs arm ``b`` on every request of the phase.

    A request counts as different when the token lists differ; ``same_input``
    says whether the two arms were given the same prompt (a later turn of a
    dialogue that already diverged has a different method-local history).
    ``logits_of``: arm whose recorded per-step top logits are attached at the
    first differing position (evidence only).
    """
    out = {"compared": 0, "different_tokens": 0, "different_tokens_same_input": 0, "different_input": 0, "requests": []}
    for image, rows in rows_by_image.items():
        for turn in (1, 2, 3):
            ra, rb = rows[(a, turn)], rows[(b, turn)]
            out["compared"] += 1
            same_input = ra["prompt"] == rb["prompt"]
            ta, tb = tokens(ra), tokens(rb)
            out["different_input"] += not same_input
            if ta == tb:
                continue
            out["different_tokens"] += 1
            out["different_tokens_same_input"] += same_input
            j = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
            item = {"image_id": image, "dialog_id": ra["dialog_id"], "turn_id": turn, "same_input": same_input, "position": j, a: ta, b: tb}
            if logits_of is not None and same_input:
                steps = rows[(logits_of, turn)]["result"].get("logit_ties") or []
                item["logits_at_position"] = steps[j] if j < len(steps) else None
            out["requests"].append(item)
    return out


def t1_identity(rows_by_image, methods):
    bad = []
    for image, rows in rows_by_image.items():
        if len({tuple(tokens(rows[(m, 1)])) for m in methods}) != 1:
            bad.append({"image_id": image, "tokens": {m: tokens(rows[(m, 1)]) for m in methods}})
    return {"images": len(rows_by_image), "identical": len(rows_by_image) - len(bad), "mismatched": len(bad), "examples": bad[:10]}


def phase_complete(run, phase, manifest, config):
    path = run / phase / "validation.json"
    if not path.exists():
        return {"status": "FAIL", "reason": "phase not complete"}
    v = json.loads(path.read_text())
    ok = (v["status"] == "PASS" and v["requests"] == manifest["expected_requests"][phase] and v["config_sha256"] == config["_file_sha256"]
          and v["independent_image_audit"] == "PASS")
    return dict(v, status="PASS" if ok else "FAIL")


def stage_gate(run, config):
    import numpy as np
    manifest = json.loads((run / "manifest.json").read_text())
    complete = phase_complete(run, "gate", manifest, config)
    if complete["status"] != "PASS":
        return receipt(run, "gate", {"status": "FAIL", "phase": complete}, config)
    rows = phase_rows(run, "gate", manifest)
    step3 = {name: compare_sync_async(rows, a, b) for name, a, b in SYNC_ASYNC}
    # step 4 (contract 9.4): IMPRESS path at 100% vs the FullLoad loader on the SAME reordered store, every request
    step4 = compare_tokens(rows, "impress_r100", "fullload_reordered")
    # reference only: the FullLoad loader on the reordered store vs FullLoad on the raster store
    reference = compare_tokens(rows, "fullload_reordered", "fullload", logits_of="fullload")
    step5 = t1_identity(rows, E.PHASE_METHODS["gate"])
    s3 = all(v["mismatched"] == 0 and v["compared"] == 132 for v in step3.values())
    s4 = step4["compared"] == 132 and step4["different_tokens"] == 0 and step4["different_input"] == 0
    s5 = step5["mismatched"] == 0 and step5["images"] == 44
    # reference only (not a criterion): synchronous vs asynchronous follow-up TTFT on the gate images
    timing = {}
    for name, a, b in SYNC_ASYNC:
        d = np.array([rows[i][(b, t)]["ttft_ms"] - rows[i][(a, t)]["ttft_ms"] for i in rows for t in (2, 3)])
        timing[name] = {"sync_mean_ms": float(np.mean([rows[i][(a, t)]["ttft_ms"] for i in rows for t in (2, 3)])),
                        "async_mean_ms": float(np.mean([rows[i][(b, t)]["ttft_ms"] for i in rows for t in (2, 3)])),
                        "paired_mean_difference_ms": float(d.mean()), "requests": int(d.size)}
    return receipt(run, "gate", {
        "status": "PASS" if s3 and s4 and s5 else "FAIL", "phase": complete,
        "step3_async_consistency": dict(step3, status="PASS" if s3 else "FAIL"),
        "step4_impress_r100_vs_fullload_loader_on_reordered_store": dict(step4, status="PASS" if s4 else "FAIL"),
        "reference_reordered_fullload_vs_raster_fullload": dict(reference, note="not a criterion: key order alone (same K/V bytes, permuted rows)"),
        "step5_turn1_identity": dict(step5, status="PASS" if s5 else "FAIL"),
        "reference_sync_vs_async_followup_ttft": timing}, config)


# ------------------------------------------------------------------ step 6
def stage_pilot(run, config):
    import numpy as np
    manifest = json.loads((run / "manifest.json").read_text())
    complete = phase_complete(run, "pilot", manifest, config)
    if complete["status"] != "PASS":
        return receipt(run, "pilot", {"status": "FAIL", "phase": complete}, config)
    rows = phase_rows(run, "pilot", manifest)
    t1 = t1_identity(rows, E.PHASE_METHODS["pilot"])
    seconds, kept, scratch = [], [], []
    for image in manifest["phase_images"]["pilot"]:
        folder = run / "pilot" / "images" / image
        commit = json.loads((folder / "COMMITTED.json").read_text())
        attempt = folder / commit["attempt_id"]
        seconds.append((attempt / "cleanup_receipt.json").stat().st_mtime - (attempt / "capacity_before.json").stat().st_mtime)
        kept.append(sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()))
        scratch.append(json.loads((attempt / "disk_usage.json").read_text())["peak_scratch_bytes"])
    plan = json.loads((run / "storage_plan.json").read_text())
    free = shutil.disk_usage(run).free
    kept_per_image = float(np.mean(kept))
    need = plan["minimum_reserve_bytes"] + plan["peak_working_set_bytes"] + int(kept_per_image * 25000 * 1.35)
    table = ROOT / "results" / run.name / "pilot" / "arms.csv"
    report = ROOT / "results" / run.name / "pilot" / "report.md"
    table_ok = table.is_file() and report.is_file() and table.stat().st_size > 0
    storage_ok = free >= need
    status = "PASS" if t1["mismatched"] == 0 and t1["images"] == 40 and table_ok and storage_ok else "FAIL"
    return receipt(run, "pilot", {
        "status": status, "phase": complete, "turn1_identity": t1, "pilot_table": str(table.relative_to(ROOT)), "pilot_table_present": table_ok,
        "measured": {"seconds_per_image_mean": float(np.mean(seconds)), "seconds_per_image_p95": float(np.percentile(seconds, 95)),
                     "retained_bytes_per_image_mean": kept_per_image, "peak_scratch_bytes_max": int(max(scratch))},
        "projection_main": {"hours": float(np.mean(seconds)) * 25000 / 3600, "retained_bytes": kept_per_image * 25000,
                            "required_free_bytes": need, "free_bytes": free, "storage": "PASS" if storage_ok else "BLOCKED_STORAGE",
                            "formula": "30 GiB reserve + 4 GiB working set + pilot retained bytes/image x 25,000 x 1.35 (= x 625 x 1.35 of the 40-image total)"}}, config)


# ------------------------------------------------------------------ step 7
def stage_admit(run, config):
    steps = {}
    for name in ("cpu", "preserve", "gate", "pilot"):
        path = run / "gates" / f"{name}.json"
        doc = json.loads(path.read_text()) if path.exists() else {"status": "MISSING"}
        ok = doc.get("status") == "PASS" and doc.get("config_sha256") == config["_file_sha256"]
        steps[name] = {"status": doc.get("status"), "bound_to_this_freeze": doc.get("config_sha256") == config["_file_sha256"], "ok": ok}
    status = "PASS" if all(s["ok"] for s in steps.values()) else "FAIL"
    return receipt(run, "admission", {"status": status, "steps": steps,
                                      "rule": "steps 1-6 all PASS for this freeze; the main stage does not start otherwise"}, config)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--stage", choices=("cpu", "preserve", "gate", "pilot", "admit"), required=True)
    args = p.parse_args()
    run = args.run.resolve()
    assert run.parent == ROOT / "runs" and run.name.startswith(E.RUN_PREFIX)
    if args.stage == "preserve":
        config, _ = E.verify_freeze(run)
    else:
        config = frozen(run)
        if E.source_hashes() != config["source_sha256"] and "_dev_" not in run.name:
            raise RuntimeError("FROZEN_SOURCE_CONFIG_MISMATCH")
    doc = {"cpu": stage_cpu, "preserve": stage_preserve, "gate": stage_gate, "pilot": stage_pilot, "admit": stage_admit}[args.stage](run, config)
    sys.exit(0 if doc["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

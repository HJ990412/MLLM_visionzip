#!/usr/bin/env python3
"""Gates of the MT-GQA HyperInfer v2 run (contract docs/mt_gqa_hyperinfer_v2_contract.md, section 5).

  --stage tests                gate 6: the existing test suites, both environments (no new failure)
  --stage gate  --model M      gates 1-5 on the 4 gate images
  --stage dev   --model M      gates 1-5 on the 40 DEV images; gate 7 (the DEV table) is reported
  --stage admit --model M      PASS iff tests, gate and dev of that model are PASS for this freeze
  --stage posthoc              after the main phases: nothing outside this run changed (reported)

Every stage writes runs/<run>/gates/<name>.json once; a recorded result is not re-judged.
Criteria are the ones frozen in the contract; a failing gate is recorded with its mismatches.
The row readers and comparison functions are those of scripts/226 (the earlier run), unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_script(name, filename):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


E = load_script("_mt_gqa_hyperinfer_v2_eval", "228_eval_mt_gqa_hyperinfer_v2.py")
V1 = load_script("_mt_gqa_hyperinfer_v2_gates_base", "226_validate_mt_gqa_hyperinfer.py")
sha, atomic_json = E.sha, E.atomic_json
FULL, IMPRESS, HYPER, HYPER_OFF, HYPER_REPEAT, HYPER_NAN = E.FULL, E.IMPRESS, E.HYPER, E.HYPER_OFF, E.HYPER_REPEAT, E.HYPER_NAN
PREFETCH_ARMS = [a for a, spec in E.HYPER_ARMS.items() if spec["kv_prefetch"] != "off"]
BYTE_FIELDS, CALL_FIELDS, SHOW = V1.BYTE_FIELDS, V1.CALL_FIELDS, V1.SHOW
frozen, receipt, committed_rows, compare, tokens = V1.frozen, V1.receipt, V1.committed_rows, V1.compare, V1.tokens
assert (V1.IMPRESS, V1.HYPER_NAN, V1.FULL) == (IMPRESS, HYPER_NAN, FULL)


def phase_state(top, model, phase, config):
    manifest = json.loads((top / "manifest.json").read_text())
    keys = sorted((d, t) for d in manifest["phase_dialogue_ids"][phase] for t in (1, 2, 3))
    path = top / E.MODEL_DIR[model] / phase / "validation.json"
    v = json.loads(path.read_text()) if path.exists() else {"status": "MISSING"}
    ok = (v.get("status") == "PASS" and v.get("requests") == manifest["expected_requests"][phase]
          and v.get("config_sha256") == config["_file_sha256"] and v.get("independent_image_audit") == "PASS")
    return manifest, keys, dict(v, status="PASS" if ok else "FAIL", expected=manifest["expected_requests"][phase],
                                criterion="every image of the phase committed after its per-image audit passed, for this freeze")


# -------------------------------------------------------------------- gates
def stop_check(rows, keys):
    """Gate 5: in every layer of every request of every arm that prefetches, no prefetch read was started after
    the GPU had finished the layer (decision time of the read, taken under the job's lock, against the stop time
    taken under the same lock); at most the one read in flight ends after the stop."""
    bad, per_arm = [], {}
    for arm in PREFETCH_ARMS:
        n = jobs = reads = after = late = 0
        begin = []
        for d, t in keys:
            if t == 1:
                continue
            n += 1
            h = rows[(arm, d, t)]["result"]["hyperinfer"]
            cols = h["layers"]
            layers_late = [i for i, x in enumerate(cols["late_starts"]) if x]
            layers_two = [i for i, x in enumerate(cols["reads_done_after_stop"]) if x > 1]
            if layers_late or layers_two or h["late_starts"] != sum(cols["late_starts"]):
                bad.append({"arm": arm, "dialog_id": d, "turn_id": t, "layers_with_late_starts": layers_late[:8],
                            "layers_with_more_than_one_read_after_stop": layers_two[:8]})
            jobs += h["jobs"]
            reads += sum(cols["k_done"]) + sum(cols["v_done"])
            after += sum(cols["reads_done_after_stop"])
            late += sum(cols["late_starts"])
            begin += [x for x in cols["begin_after_stop_ms"] if x is not None]
        per_arm[arm] = {"followup_requests": n, "jobs": jobs, "layers_checked": jobs, "prefetch_reads": reads, "reads_started_after_stop": late,
                        "reads_in_flight_at_stop": after,
                        "latest_read_call_relative_to_stop_ms": float(max(begin)) if begin else None}
    return {"arms": per_arm, "violations": len(bad), "first_violations": bad[:SHOW],
            "criterion": "reads started after the stop = 0 in every layer (and so at most one read, the one in flight, ends after the stop)",
            "note": "latest_read_call_relative_to_stop_ms: the largest (time the read call was issued - stop time) over all prefetch reads; a read is "
                    "allowed under the lock before the stop and issued right after the lock is released, so the value may exceed 0 by that delay",
            "status": "PASS" if not bad else "FAIL"}


def nan_guard(rows, keys):
    out = V1.gate5(rows, keys)
    return out


def recorded_impress(rows, keys, ref):
    """Reported with gate 1 on DEV: HyperInfer's tokens against the tokens recorded for IMPRESS in the earlier main run."""
    bad = [{"dialog_id": d, "turn_id": t} for d, t in keys if tokens(rows[(HYPER, d, t)]) != tokens(ref[(IMPRESS, d, t)])]
    return {"status": "REPORTED", "requests": len(keys), "mismatches": len(bad), "first_mismatches": bad[:SHOW]}


def dev_table(rows, keys, ref):
    """Gate 7 (reported): FullLoad, IMPRESS and HyperInfer of this DEV phase, same requests, rotated arm order."""
    out = {}
    fu = [(d, t) for d, t in keys if t > 1]
    for method in (FULL, IMPRESS, HYPER):
        ttft = np.array([rows[(method, d, t)]["ttft_ms"] for d, t in fu])
        out[method] = {"followup_requests": len(fu), "em": float(100 * np.mean([rows[(method, d, t)]["score"] for d, t in fu])),
                       "ttft_mean_ms": float(ttft.mean()), "ttft_p50_ms": float(np.median(ttft)), "ttft_p95_ms": float(np.percentile(ttft, 95)),
                       "ssd_mb_per_followup": float(np.mean([rows[(method, d, t)]["ssd_read_bytes"] for d, t in fu]) / 1e6)}
    cont = V1.continuity(rows, keys, ref)
    return {"status": "REPORTED", "methods": out, "ttft_ratio_hyperinfer_over_impress": out[HYPER]["ttft_mean_ms"] / out[IMPRESS]["ttft_mean_ms"],
            "continuity_with_earlier_main_run": cont["methods"],
            "note": "not a gate; this DEV phase rotates 6 arms, the earlier main runs rotated 4 (ReComp, FullLoad, IMPRESS, Ours)"}


def judge(top, model, phase, config):
    manifest, keys, execution = phase_state(top, model, phase, config)
    name = f"{model}_{phase}"
    if execution["status"] != "PASS":
        return receipt(top, name, {"status": "FAIL", "model": model, "phase": phase, "execution": execution, "failed": ["execution"]}, config)
    rows = committed_rows(top / E.MODEL_DIR[model], phase)
    off = compare(rows, HYPER_OFF, IMPRESS, keys, fields=BYTE_FIELDS + CALL_FIELDS)
    off["also_reported_per_layer_used_set"] = {k: v for k, v in compare(rows, HYPER_OFF, IMPRESS, keys, selection=True).items()
                                                if k in ("mismatches", "status")}
    gates = {
        "gate1_output_equals_impress": dict(compare(rows, HYPER, IMPRESS, keys, selection=True),
                                            criterion="generated tokens (T1-T3) and per-layer used sets (T2-T3) equal the IMPRESS arm executed in the same "
                                                      "session, every request"),
        "gate2_prefetch_off_equals_impress": dict(off, criterion="prefetch off: generated tokens, read bytes and read call counts equal the IMPRESS arm "
                                                                 "executed in the same session, every request"),
        "gate3_unread_rows_not_used": nan_guard(rows, keys),
        "gate4_repeat": dict(compare(rows, HYPER_REPEAT, HYPER, keys),
                             criterion="the same request executed a second time: generated tokens equal, every request (the prefetched amount may differ)"),
        "gate5_no_read_started_after_stop": stop_check(rows, keys),
    }
    extra = {}
    if phase == "dev":
        ref = V1.reference_rows(model, manifest["phase_images"]["dev"])
        extra["tokens_against_recorded_impress"] = recorded_impress(rows, keys, ref)
        extra["gate7_dev_table"] = dev_table(rows, keys, ref)
    failed = [k for k, g in gates.items() if g["status"] != "PASS"]
    t1 = sum(len({tuple(tokens(rows[(m, d, 1)])) for m in config["phase_methods"][phase]}) == 1 for d, t in keys if t == 1)
    # how much the two executions of the same request differed in what was prefetched (reported with gate 4)
    fu = [(d, t) for d, t in keys if t > 1]
    a = np.array([rows[(HYPER, d, t)]["result"]["hyperinfer"]["prefetched_bytes"] for d, t in fu], dtype=np.float64)
    b = np.array([rows[(HYPER_REPEAT, d, t)]["result"]["hyperinfer"]["prefetched_bytes"] for d, t in fu], dtype=np.float64)
    gates["gate4_repeat"]["prefetched_bytes"] = {"first_mean": float(a.mean()), "second_mean": float(b.mean()),
                                                 "requests_with_a_different_amount": int((a != b).sum()), "requests": len(fu)}
    return receipt(top, name, {"status": "PASS" if not failed else "FAIL", "model": model, "phase": phase, "execution": execution,
                               "requests": len(rows), "gates": gates, "failed": failed, **extra,
                               "turn1_tokens_identical_across_arms": {"dialogues": len(keys) // 3, "identical": int(t1)}}, config)


# -------------------------------------------------------------------- tests
def stage_tests(top, config):
    V = load_script("_mt_gqa_hyperinfer_v2_suite", "223_validate_mt_gqa_main_v2.py")      # run_suite + the recorded pre-existing failures per environment
    suites = [V.run_suite(sys.executable), V.run_suite(V.PY_CROSS)]
    ok = all(r["status"] == "PASS" for r in suites)
    return receipt(top, "tests", {
        "gate": "gate6_existing_tests", "status": "PASS" if ok else "FAIL", "suites": suites,
        "failed": [] if ok else [f for r in suites for f in r["new_failures"]],
        "criterion": "no failing test other than the failures that existed before this work (listed per environment in "
                     "baseline_failures_for_this_environment); tests/test_hyperinfer_v2.py and tests/test_hyperinfer.py are part of both suites"}, config)


def stage_admit(top, model, config):
    steps = {}
    for name in ("tests", f"{model}_gate", f"{model}_dev"):
        path = top / "gates" / f"{name}.json"
        doc = json.loads(path.read_text()) if path.exists() else {"status": "MISSING"}
        ok = doc.get("status") == "PASS" and doc.get("config_sha256") == config["_file_sha256"]
        steps[name] = {"status": doc.get("status"), "bound_to_this_freeze": doc.get("config_sha256") == config["_file_sha256"], "ok": ok,
                       "failed": doc.get("failed")}
    status = "PASS" if all(s["ok"] for s in steps.values()) else "FAIL"
    return receipt(top, f"{model}_admission", {
        "status": status, "model": model, "steps": steps, "failed": [k for k, s in steps.items() if not s["ok"]],
        "rule": "the model's main phase starts only if gates 1-6 are PASS (gates 1-5 on the 4 gate images and on the 40 DEV images); "
                "TTFT against IMPRESS is not a gate"}, config)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--stage", choices=("tests", "gate", "dev", "admit", "posthoc"), required=True)
    p.add_argument("--model", choices=E.MODELS)
    args = p.parse_args()
    top = args.run.resolve()
    assert top.parent == ROOT / "runs" and top.name.startswith(E.RUN_PREFIX)
    config = frozen(top)
    now = E.source_hashes()
    changed = sorted(k for k, v in config["source_sha256"].items() if now.get(k) != v)
    if changed and "_dev_" not in top.name:
        raise RuntimeError("FROZEN_SOURCE_CONFIG_MISMATCH: " + repr(changed[:10]))
    if args.stage == "tests":
        doc = stage_tests(top, config)
    elif args.stage in ("gate", "dev"):
        doc = judge(top, args.model, args.stage, config)
    elif args.stage == "admit":
        doc = stage_admit(top, args.model, config)
    else:
        doc = V1.stage_posthoc(top, config)
    sys.exit(0 if doc["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

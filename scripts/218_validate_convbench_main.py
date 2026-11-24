#!/usr/bin/env python3
"""ConvBench main experiment: the gates (contract docs/convbench_main_contract.md, section "Gate").

Reads committed conversations only; no model is loaded here (gate 6 is scripts/219 --check).

  --stage tests                      gate 1: the existing unit tests, no new failure
  --stage gate  --model M --scale gate    gates 2, 3, 4, 5 on the 4 gate conversations (9 arms)
  --stage gate  --model M --scale pilot   gates 2, 5 on the 40 pilot conversations (4 arms)
  --stage overflow-select --model M  gate 7: does the pilot contain an input above 4096 tokens? if not, pick 2 conversations
  --stage overflow-check  --model M  gate 7: did the two extra conversations finish without a failed request?
  --stage admit --model M            admission of the model's main phase: gates 1-6 all PASS (recorded once)

Every result is written once to <res>/gates/...; a recorded result is never judged again and no criterion
is changed after the fact.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import convbench_main_common as C  # noqa: E402

NEW_TEST = "test_convbench_main.py"
# failures that exist without this work, by major transformers version (the same list as scripts/207 and scripts/210)
BASELINE_FAILURES = {
    "4": {"setUpClass (test_glm46v_runner.NativeProcessorRunnerTests)",
          "test_actual_native_adjacent_pair_partial_rope_keeps_tail_bits (test_glm46v_validation.GlmValidationTests)",
          "test_actual_native_sdpa_hook_preserves_output_and_qk_score (test_glm46v_vision.GlmVisionTests)"},
    "5": {"test_single_stock_forward_capture_equals_dense_visionzip_formula (test_qwen25_vision.StockVisionCaptureTests)",
          "test_real_tiny_clip_matches_full_penultimate_attention (test_visdial_turn1_piggyback_core.VisionForwardCaptureTests)"},
}


def gate_path(res, model, name):
    return Path(res) / "gates" / (model or "") / f"{name}.json"


def write_once(path, doc, config):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already recorded; a gate result is not judged again")
    doc = dict(doc, config_sha256=config["_file_sha256"], at_utc=C.utc())
    C.atomic_json(path, doc)
    print(json.dumps({"gate_file": str(path), "status": doc.get("status")}, ensure_ascii=False), flush=True)
    return doc


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
            "failures_that_existed_before_this_work": sorted(f for f in failed if f in baseline),
            "known_baseline_failures_for_this_environment": sorted(baseline), "new_failures": new,
            "status": "PASS" if ran and not new else "FAIL", "tail": text[-1500:]}


def stage_tests(res, config):
    if "_dev_" in res.name and os.environ.get("CONVBENCH_MAIN_DEV_SKIP_TESTS") == "1":      # development rehearsals only; never a measurement
        return write_once(gate_path(res, None, "tests"), {"gate": "gate1_existing_tests", "status": "PASS", "suite": None,
                                                          "rehearsal_note": "test suite skipped in a development rehearsal"}, config)
    suite = run_suite(sys.executable)
    new = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", NEW_TEST], capture_output=True, text=True, cwd=ROOT)
    ran = re.search(r"^Ran (\d+) tests", new.stdout + new.stderr, re.M)
    ok = new.returncode == 0 and ran is not None and int(ran.group(1)) > 0
    return write_once(gate_path(res, None, "tests"),
                      {"gate": "gate1_existing_tests", "status": "PASS" if suite["status"] == "PASS" and ok else "FAIL", "suite": suite,
                       "new_unit_tests": {"file": f"tests/{NEW_TEST}", "ran": int(ran.group(1)) if ran else 0, "ok": ok},
                       "criterion": "no failing test other than the ones that failed before this work in this environment"}, config)


# ------------------------------------------------------------ gates 2 - 5
def tokens(row):
    return row["generated_token_ids"] if row["status"] == "ok" else None


def compare_pair(docs, a, b, turns, with_bytes):
    """Requests of arm a against the same requests of arm b: generated tokens (and read bytes)."""
    out = {"requests": 0, "token_mismatches": 0, "byte_mismatches": 0, "missing": 0, "examples": []}
    for cid, doc in docs.items():
        by = C.rows_by_key(doc)
        for t in turns:
            ra, rb = by.get((a, t)), by.get((b, t))
            out["requests"] += 1
            if ra is None or rb is None or tokens(ra) is None or tokens(rb) is None:
                out["missing"] += 1
                out["examples"].append({"conversation_id": cid, "turn": t, "problem": "missing or failed request"})
                continue
            bad_t = tokens(ra) != tokens(rb)
            bad_b = with_bytes and t > 1 and (ra["ssd_read_bytes"] != rb["ssd_read_bytes"])
            out["token_mismatches"] += bad_t; out["byte_mismatches"] += bool(bad_b)
            if (bad_t or bad_b) and len(out["examples"]) < 10:
                first = next((i for i, (x, y) in enumerate(zip(tokens(ra), tokens(rb))) if x != y), min(len(tokens(ra)), len(tokens(rb))))
                out["examples"].append({"conversation_id": cid, "turn": t, "generated_tokens": [len(tokens(ra)), len(tokens(rb))],
                                        "first_differing_position": first if bad_t else None,
                                        "read_bytes": [ra["ssd_read_bytes"], rb["ssd_read_bytes"]]})
    out["ok"] = out["requests"] > 0 and not (out["token_mismatches"] or out["byte_mismatches"] or out["missing"])
    return out


def gate2_recomp(model, docs):
    """ReComp of this runner == the alignment run's final ReComp, request by request.

    The alignment rows hold the answer text, the number of generated tokens and the first token id (the token
    id list was not recorded there), so those three are compared; this runner's own rows keep the token ids.
    """
    out = {"requests": 0, "mismatches": 0, "missing": 0, "examples": [], "input_token_mismatches": 0}
    for cid, doc in docs.items():
        by = C.rows_by_key(doc)
        ref = C.alignment_rows(model, cid)
        for t in (1, 2, 3):
            r, a = by.get((C.RECOMP, t)), ref[t - 1]
            out["requests"] += 1
            if r is None or r["status"] != "ok":
                out["missing"] += 1
                out["examples"].append({"conversation_id": cid, "turn": t, "problem": "ReComp request failed in this runner"})
                continue
            same = (r["prediction"] == a["prediction"] and r["generated_tokens"] == a["generated_tokens"]
                    and r["first_token_id"] == a["first_token_id"])
            out["input_token_mismatches"] += r["input_tokens"] != a["input_tokens"]
            if not same:
                out["mismatches"] += 1
                if len(out["examples"]) < 10:
                    out["examples"].append({"conversation_id": cid, "turn": t, "generated_tokens": [r["generated_tokens"], a["generated_tokens"]],
                                            "first_token_id": [r["first_token_id"], a["first_token_id"]],
                                            "text_equal": r["prediction"] == a["prediction"], "input_tokens": [r["input_tokens"], a["input_tokens"]]})
    out["ok"] = out["requests"] > 0 and not (out["mismatches"] or out["missing"] or out["input_token_mismatches"])
    out["compared"] = "answer text, number of generated tokens, first token id, number of input tokens"
    return out


def gate5_t1(docs):
    out = {"conversations": 0, "differing": 0, "examples": []}
    for cid, doc in docs.items():
        by = C.rows_by_key(doc)
        seqs = [tokens(by[(role, 1)]) if (role, 1) in by else None for role in C.MAIN_ROLES]
        out["conversations"] += 1
        if any(s is None for s in seqs) or len({tuple(s) for s in seqs}) != 1:
            out["differing"] += 1
            if len(out["examples"]) < 10:
                out["examples"].append({"conversation_id": cid, "generated_tokens": {role: (len(s) if s is not None else None)
                                                                                    for role, s in zip(C.MAIN_ROLES, seqs)}})
    out["ok"] = out["conversations"] > 0 and out["differing"] == 0
    return out


def stage_gate(res, config, model, scale):
    index, _ = C.load_index()
    if not C.phase_complete(res, model, scale):
        raise SystemExit(f"the {scale} phase of {model} is not complete")
    docs = C.committed(res, model, scale, index)
    failures = [{k: f[k] for k in ("conversation_id", "method", "turn_id", "stage", "error_type", "error")} for d in docs.values() for f in d["failures"]]
    doc = {"gate": f"gates_on_{scale}", "model": model, "scale": scale, "conversations": len(docs),
           "requests": sum(len(d["rows"]) for d in docs.values()), "failed_requests": failures, "gates": {}}
    doc["gates"]["gate2_recomp_equals_alignment"] = gate2_recomp(model, docs)
    doc["gates"]["gate5_turn1_tokens_identical_across_four_methods"] = gate5_t1(docs)
    if scale == "gate":
        g3 = {"sync_vs_async": compare_pair(docs, C.OURS_SYNC, C.OURS, (1, 2, 3), True),
              "same_request_twice": compare_pair(docs, C.OURS, C.OURS_REPEAT, (1, 2, 3), True)}
        g3["ok"] = g3["sync_vs_async"]["ok"] and g3["same_request_twice"]["ok"]
        doc["gates"]["gate3_ours"] = g3
        g4 = {"retention_100_vs_fullload_loader_on_same_store": compare_pair(docs, C.IMP_R100, C.FULL_ON_IMP, (1, 2, 3), False),
              "sync_vs_async": compare_pair(docs, C.IMP_SYNC, C.IMP, (1, 2, 3), True)}
        g4["ok"] = g4["retention_100_vs_fullload_loader_on_same_store"]["ok"] and g4["sync_vs_async"]["ok"]
        doc["gates"]["gate4_impress"] = g4
        doc["no_failed_request"] = not failures
        ok = all(g["ok"] for g in doc["gates"].values()) and not failures
    else:
        ok = all(g["ok"] for g in doc["gates"].values())
    doc["status"] = "PASS" if ok else "FAIL"
    return write_once(gate_path(res, model, f"gate_{scale}"), doc, config)


# ------------------------------------------------------------------ gate 7
def stage_overflow_select(res, config, model):
    index, _ = C.load_index()
    docs = C.committed(res, model, "pilot", index)
    above = [{"conversation_id": cid, "method": r["method"], "turn": r["turn_id"], "input_tokens": r["input_tokens"]}
             for cid, d in docs.items() for r in d["rows"] if r["status"] == "ok" and r["input_above_4096"]]
    path = C.model_dir(res, model) / "overflow_selection.json"
    if path.exists():
        raise FileExistsError(path)
    if above:
        doc = {"needed": False, "reason": "the pilot contains inputs above 4096 tokens", "pilot_requests_above_4096": above, "ordinals": []}
    else:
        ranked = []
        for conv in index:
            rows = C.alignment_rows(model, conv["conversation_id"])
            ranked.append((max(r["input_tokens"] for r in rows), conv["ordinal"], conv["conversation_id"]))
        ranked.sort(reverse=True)
        chosen = ranked[:C.OVERFLOW_CONVERSATIONS]
        doc = {"needed": True, "reason": "the pilot contains no input above 4096 tokens", "pilot_requests_above_4096": [],
               "selection_rule": "the conversations with the largest ReComp input in the alignment run",
               "ordinals": [o for _, o, _ in chosen],
               "selected": [{"conversation_id": c, "ordinal": o, "alignment_recomp_largest_input_tokens": n, "above_4096_in_alignment": n > C.CONTEXT_LIMIT}
                            for n, o, c in chosen],
               "note": None if all(n > C.CONTEXT_LIMIT for n, _, _ in chosen) else
               "no ReComp input of this model exceeded 4096 tokens in the alignment run; the conversations with the longest inputs are run instead"}
    doc.update(model=model, config_sha256=config["_file_sha256"], at_utc=C.utc())
    C.atomic_json(path, doc)
    print(json.dumps({"overflow_selection": {k: doc[k] for k in ("needed", "ordinals")}}), flush=True)


def stage_overflow_check(res, config, model):
    index, _ = C.load_index()
    sel = C.read_json(C.model_dir(res, model) / "overflow_selection.json")
    if not sel["needed"]:
        return write_once(gate_path(res, model, "gate7_overflow"), {"gate": "gate7_inputs_above_4096", "status": "COVERED_BY_PILOT",
                                                                     "pilot_requests_above_4096": sel["pilot_requests_above_4096"]}, config)
    docs = C.committed(res, model, "overflow", index)
    rows = [{"conversation_id": cid, "method": r["method"], "turn": r["turn_id"], "status": r["status"],
             "input_tokens": r.get("input_tokens"), "generated_tokens": r.get("generated_tokens"), "failure": r.get("failure")}
            for cid, d in docs.items() for r in d["rows"]]
    failed = [r for r in rows if r["status"] != "ok"]
    above = [r for r in rows if r["status"] == "ok" and r["input_tokens"] > C.CONTEXT_LIMIT]
    return write_once(gate_path(res, model, "gate7_overflow"),
                      {"gate": "gate7_inputs_above_4096", "status": "PASS" if rows and not failed else "FAIL", "selection": sel,
                       "requests": len(rows), "failed_requests": failed, "requests_above_4096": len(above), "rows": rows,
                       "note": "recorded; not a condition of the admission (gates 1-6 are)"}, config)


# --------------------------------------------------------------- admission
def stage_admit(res, config, model):
    def status(path):
        doc = C.read_json(path)
        return "MISSING" if doc is None else doc.get("status", "MISSING")
    check = C.read_json(Path(res) / "gates" / "judge_check.json") or {}
    parts = {"gate1_tests": status(gate_path(res, None, "tests")),
             "gate6_judge": (check.get("models", {}).get(model, {}).get("status", "MISSING")),
             "gates_2_3_4_5_on_4_conversations": status(gate_path(res, model, "gate_gate")),
             "gates_2_5_on_pilot_40": status(gate_path(res, model, "gate_pilot"))}
    ok = all(v == "PASS" for v in parts.values())
    reasons = []
    for name in ("gate_gate", "gate_pilot"):
        doc = C.read_json(gate_path(res, model, name)) or {}
        for gname, g in doc.get("gates", {}).items():
            if not g.get("ok"):
                reasons.append({"where": name, "gate": gname, "detail": {k: v for k, v in g.items() if k != "examples"}, "examples": g.get("examples", [])[:5]})
        if doc.get("failed_requests") and name == "gate_gate":
            reasons.append({"where": name, "gate": "no_failed_request", "failed_requests": doc["failed_requests"][:10]})
    return write_once(gate_path(res, model, "admission"),
                      {"gate": "admission", "model": model, "status": "PASS" if ok else "FAIL", "parts": parts, "failed_gates": reasons,
                       "gate7": "pilot table and the inputs-above-4096 check are recorded, not a condition"}, config)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", type=Path, required=True)
    p.add_argument("--stage", choices=("tests", "gate", "overflow-select", "overflow-check", "admit"), required=True)
    p.add_argument("--model", choices=C.MODELS)
    p.add_argument("--scale", choices=("gate", "pilot"))
    args = p.parse_args()
    res = args.res.resolve()
    config = C.verify_freeze(res)
    if args.stage == "tests":
        doc = stage_tests(res, config)
    elif args.stage == "gate":
        doc = stage_gate(res, config, args.model, args.scale)
    elif args.stage == "overflow-select":
        return stage_overflow_select(res, config, args.model)
    elif args.stage == "overflow-check":
        doc = stage_overflow_check(res, config, args.model)
    else:
        doc = stage_admit(res, config, args.model)
    sys.exit(0 if doc.get("status") in ("PASS", "COVERED_BY_PILOT") else 1)


if __name__ == "__main__":
    main()

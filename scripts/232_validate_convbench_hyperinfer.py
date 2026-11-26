#!/usr/bin/env python3
"""Gates of the ConvBench HyperInfer run (contract docs/convbench_hyperinfer_contract.md, sections 4 and 5).

Reads committed conversations only; no model is loaded here.

  --stage tests                           gate 7: the existing test suites, both environments (no new failure)
  --stage gate  --model M --scale gate    gates 1-5 on the 4 gate conversations (6 arms)
  --stage gate  --model M --scale pilot   gates 1, 5, 6 on the 40 pilot conversations (FullLoad, IMPRESS, HyperInfer);
                                          the pilot table and the TTFT continuity (gate 8, reported)
  --stage overflow-select --model M       gate 8: does the pilot hold a HyperInfer input above 4096 tokens? if not, pick 2 conversations
  --stage overflow-check  --model M       gate 8: did the two extra conversations finish without a failed request?
  --stage admit --model M                 admission of the model's main phase: gates 1-7 all PASS (recorded once)
  --stage identity --model M              after the main phase: HyperInfer's tokens against the recorded IMPRESS tokens, 1,731 requests
  --stage posthoc                         after everything: files outside this run are as they were before it (reported)

Every result is written once to <res>/gates/...; a recorded result is never judged again and no criterion is changed
after the fact.  The comparison functions follow scripts/226 and scripts/229 (the MT-GQA HyperInfer gates) on the
row layout of scripts/217.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import convbench_hyperinfer_common as HC  # noqa: E402

C = HC.C
V = HC.load_with_common("_convbench_hyperinfer_main_gates", "218_validate_convbench_main.py")      # gate_path, write_once
FULL, IMP, HYPER, HYPER_OFF, HYPER_REPEAT, HYPER_NAN = HC.FULL, HC.IMP, HC.HYPER, HC.HYPER_OFF, HC.HYPER_REPEAT, HC.HYPER_NAN
NEW_TEST = "test_convbench_hyperinfer.py"
BYTE_FIELDS = ("ssd_read_bytes", "probe_read_bytes", "k_read_bytes", "v_read_bytes")
CALL_FIELDS = ("read_calls", "pread_calls", "preadv_calls")
SHOW = HC.SHOW
gate_path, write_once, tokens = V.gate_path, V.write_once, HC.tokens


def first_difference(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


# ----------------------------------------------------------- gates 1 - 5
def compare(docs, a, b, *, selection=False, fields=()):
    """Requests of arm ``a`` against the same requests of arm ``b`` (same session): generated tokens of T1-T3, and on
    T2-T3 the per-layer used sets and the listed row fields.  A missing or failed request is a mismatch."""
    bad, n, followups = [], 0, 0
    for cid, doc in docs.items():
        by = C.rows_by_key(doc)
        for t in (1, 2, 3):
            ra, rb = by.get((a, t)), by.get((b, t))
            n += 1
            followups += t > 1
            if tokens(ra) is None or tokens(rb) is None:
                bad.append({"conversation_id": cid, "turn_id": t, "difference": ["missing or failed request"]})
                continue
            diff = []
            if tokens(ra) != tokens(rb):
                diff.append(f"tokens (lengths {len(tokens(ra))} / {len(tokens(rb))}, first difference at {first_difference(tokens(ra), tokens(rb))})")
            if t > 1:
                if selection:
                    sa, sb = ra["result"]["selection_digest"], rb["result"]["selection_digest"]
                    if sa["all_layers"] != sb["all_layers"]:
                        layers = [i for i, (x, y) in enumerate(zip(sa["per_layer"], sb["per_layer"])) if x != y]
                        diff.append(f"selection (layers {layers[:8]})")
                for f in fields:
                    if ra[f] != rb[f]:
                        diff.append(f"{f}: {ra[f]} != {rb[f]}")
            if diff:
                bad.append({"conversation_id": cid, "turn_id": t, "difference": diff})
    return {"a": a, "b": b, "requests": n, "followup_requests": followups, "mismatches": len(bad), "first_mismatches": bad[:SHOW],
            "compared": ["generated tokens (T1, T2, T3)"] + (["per-layer used set (T2, T3)"] if selection else []) + [f"{f} (T2, T3)" for f in fields],
            "ok": n > 0 and not bad}


def stop_check(docs, arms):
    """Gate 5: in every layer of every follow-up request of every arm that prefetches, no prefetch read was started
    after the GPU had finished the layer (decision time of the read against the stop time, both taken under the job's
    lock); at most the one read in flight ends after the stop."""
    bad, per_arm = [], {}
    for arm in arms:
        n = jobs = reads = after = late = 0
        begin = []
        for cid, doc in docs.items():
            by = C.rows_by_key(doc)
            for t in (2, 3):
                r = by.get((arm, t))
                n += 1
                if tokens(r) is None:
                    bad.append({"arm": arm, "conversation_id": cid, "turn_id": t, "problem": "missing or failed request"})
                    continue
                h = r["result"]["hyperinfer"]
                cols = h["layers"]
                layers_late = [i for i, x in enumerate(cols["late_starts"]) if x]
                layers_two = [i for i, x in enumerate(cols["reads_done_after_stop"]) if x > 1]
                if layers_late or layers_two or h["late_starts"] != sum(cols["late_starts"]) or h.get("truncation") != "reactive":
                    bad.append({"arm": arm, "conversation_id": cid, "turn_id": t, "layers_with_late_starts": layers_late[:8],
                                "layers_with_more_than_one_read_after_stop": layers_two[:8]})
                jobs += h["jobs"]
                reads += sum(cols["k_done"]) + sum(cols["v_done"])
                after += sum(cols["reads_done_after_stop"])
                late += sum(cols["late_starts"])
                begin += [x for x in cols["begin_after_stop_ms"] if x is not None]
        per_arm[arm] = {"followup_requests": n, "layers_checked": jobs, "prefetch_reads": reads, "reads_started_after_stop": late,
                        "reads_in_flight_at_stop": after, "latest_read_call_relative_to_stop_ms": float(max(begin)) if begin else None}
    return {"arms": per_arm, "violations": len(bad), "first_violations": bad[:SHOW],
            "criterion": "reads started after the stop = 0 in every layer (and so at most one read, the one in flight, ends after the stop)",
            "note": "latest_read_call_relative_to_stop_ms: the largest (time the read call was issued - stop time) over all prefetch reads; a read is "
                    "allowed under the lock before the stop and issued right after the lock is released, so the value may exceed 0 by that delay",
            "ok": bool(per_arm) and not bad}


def nan_guard(docs):
    """Gate 3: the arm whose unread visual rows hold NaN."""
    bad, n = [], 0
    totals = {"unread_rows": 0, "used_rows": 0, "attention_calls": 0}
    for cid, doc in docs.items():
        by = C.rows_by_key(doc)
        for t in (2, 3):
            n += 1
            r = by.get((HYPER_NAN, t))
            g = r["result"].get("unread_row_guard") if tokens(r) is not None else None
            if not (g and g["all_outputs_finite"] and g["used_equals_selection"] and g["used_rows_all_read"]):
                bad.append({"conversation_id": cid, "turn_id": t, "guard": g})
            else:
                for k in totals:
                    totals[k] += g[k]
    same = compare(docs, HYPER_NAN, IMP, selection=True)
    return {"requests": n, "guard_failures": len(bad), "first_guard_failures": bad[:SHOW], "totals": totals, "against_impress": same,
            "criterion": "every visual row of the prefix cache holds NaN before the first layer; rows are overwritten only by reads; at masked columns the "
                         "debug wrapper passes 0 (what an unread row holds in a normal request), nowhere else: the attention output of every layer is "
                         "finite, the unmasked visual columns equal the layer's used set, every used row was read, and tokens and used sets equal IMPRESS",
            "ok": n > 0 and not bad and same["ok"]}


def against_recorded(model, docs, role=HYPER, ref_role=None):
    """Tokens of ``role`` in this run against the tokens the main experiment recorded for ``ref_role`` (default IMPRESS)."""
    ref_role = ref_role or C.IMP
    bad, n, equal = [], 0, 0
    per_turn = {t: [0, 0] for t in (1, 2, 3)}
    for cid, doc in docs.items():
        by, ref = C.rows_by_key(doc), HC.recorded_rows(model, cid)
        earlier_equal = True
        for t in (1, 2, 3):
            n += 1; per_turn[t][1] += 1
            a, b = tokens(by.get((role, t))), tokens(ref.get((ref_role, t)) if ref else None)
            if a is not None and a == b:
                equal += 1; per_turn[t][0] += 1
                continue
            entry = {"conversation_id": cid, "turn_id": t, "earlier_turns_of_this_conversation_equal": earlier_equal}
            if a is None or b is None:
                entry["problem"] = "failed request in this run" if a is None else "no recorded IMPRESS request"
            else:
                entry.update(generated_tokens=[len(a), len(b)], first_differing_position=first_difference(a, b))
            bad.append(entry)
            earlier_equal = False
    return {"requests": n, "equal": equal, "mismatches": len(bad), "equal_per_turn": {f"T{t}": v for t, v in per_turn.items()},
            "mismatch_list": bad, "ok": n > 0 and not bad}


def t1_identical(docs, roles):
    same = 0
    for doc in docs.values():
        by = C.rows_by_key(doc)
        seqs = [tokens(by.get((r, 1))) for r in roles]
        same += all(s is not None for s in seqs) and len({tuple(s) for s in seqs}) == 1
    return {"conversations": len(docs), "identical": int(same), "arms": list(roles)}


# ----------------------------------------------------- gate 8: pilot table
def followup_pairs(docs, a_of, b_of):
    """Per conversation (sum over T2, T3) of two TTFT series; conversations with a failed request on either side are left out."""
    sa, sb = [], []
    for cid, doc in docs.items():
        ra, rb = [a_of(cid, doc, t) for t in (2, 3)], [b_of(cid, doc, t) for t in (2, 3)]
        if any(tokens(r) is None for r in ra + rb):
            continue
        sa.append(sum(r["ttft_ms"] for r in ra)); sb.append(sum(r["ttft_ms"] for r in rb))
    return np.asarray(sa), np.asarray(sb)


def ratio_ci(sa, sb):
    if sa.size < 2:
        return {"ratio": float(sa.sum() / sb.sum()) if sa.size else None, "ci_low": None, "ci_high": None, "n": int(sa.size)}
    rng = np.random.default_rng(C.SEED)
    idx = rng.integers(0, sa.size, size=(C.RESAMPLES, sa.size))
    r = sa[idx].sum(axis=1) / sb[idx].sum(axis=1)
    lo, hi = np.percentile(r, [2.5, 97.5])
    d = (sa - sb) / 2
    dm = d[idx].mean(axis=1)
    dlo, dhi = np.percentile(dm, [2.5, 97.5])
    return {"ratio": float(sa.sum() / sb.sum()), "ci_low": float(lo), "ci_high": float(hi), "n": int(sa.size),
            "diff_ms": float(d.mean()), "diff_ci_low": float(dlo), "diff_ci_high": float(dhi)}


def pilot_table(model, docs):
    """FullLoad, IMPRESS and HyperInfer of this pilot (same conversations, rotated arm order) and their TTFT against the
    same requests of the main experiment."""
    here = lambda role: (lambda cid, doc, t: C.rows_by_key(doc).get((role, t)))      # noqa: E731
    there = lambda role: (lambda cid, doc, t: (HC.recorded_rows(model, cid) or {}).get((role, t)))      # noqa: E731
    methods = {}
    for role in HC.PILOT_ROLES:
        fu = [r for d in docs.values() for r in d["rows"] if r["role"] == role and r["turn_id"] > 1 and r["status"] == "ok"]
        ttft = np.array([r["ttft_ms"] for r in fu])
        per_turn = {f"T{t}": float(np.mean([r["ttft_ms"] for r in fu if r["turn_id"] == t])) for t in (2, 3)} if fu else {}
        methods[role] = {"method": HC.DISPLAY[role], "followup_requests": len(fu),
                         "failed_requests": sum(1 for d in docs.values() for r in d["rows"] if r["role"] == role and r["status"] != "ok"),
                         "ttft_mean_ms": float(ttft.mean()) if fu else None, "ttft_p50_ms": float(np.median(ttft)) if fu else None,
                         "ttft_p95_ms": float(np.percentile(ttft, 95)) if fu else None, "ttft_mean_by_turn_ms": per_turn,
                         "ssd_mb_per_followup": float(np.mean([r["ssd_read_bytes"] for r in fu]) / 1e6) if fu else None,
                         "read_calls_per_followup": float(np.mean([r["read_calls"] for r in fu])) if fu else None,
                         "requests_above_4096": sum(1 for d in docs.values() for r in d["rows"]
                                                    if r["role"] == role and r["status"] == "ok" and r["input_above_4096"])}
    continuity = {}
    for role in (FULL, IMP):
        now, old = followup_pairs(docs, here(role), there(role))
        same = sum(tokens(C.rows_by_key(d).get((role, t))) is not None
                   and tokens(C.rows_by_key(d).get((role, t))) == tokens((HC.recorded_rows(model, cid) or {}).get((role, t)))
                   for cid, d in docs.items() for t in (1, 2, 3))
        continuity[role] = {"method": HC.DISPLAY[role], "conversations": int(now.size), "this_run_mean_ms": float(now.sum() / (2 * now.size)) if now.size else None,
                            "main_experiment_mean_ms": float(old.sum() / (2 * old.size)) if old.size else None, **ratio_ci(now, old),
                            "tokens_equal_to_main_experiment": {"requests": 3 * len(docs), "equal": int(same)}}
    hi, imp = followup_pairs(docs, here(HYPER), here(IMP))
    hf, full = followup_pairs(docs, here(HYPER), here(FULL))
    return {"status": "REPORTED", "conversations": len(docs), "methods": methods,
            "hyperinfer_over_impress_same_run": ratio_ci(hi, imp), "hyperinfer_over_fullload_same_run": ratio_ci(hf, full),
            "continuity_with_main_experiment": continuity,
            "note": "not a gate; this pilot rotates 3 arms (FullLoad, IMPRESS, HyperInfer), the main experiment rotated 4 (ReComp, FullLoad, IMPRESS, Ours)"}


# ------------------------------------------------------------- stages
def stage_gate(res, config, model, scale):
    index, _ = C.load_index()
    if not C.phase_complete(res, model, scale):
        raise SystemExit(f"the {scale} phase of {model} is not complete")
    docs = C.committed(res, model, scale, index)
    failures = [{k: f[k] for k in ("conversation_id", "method", "turn_id", "stage", "error_type", "error")} for d in docs.values() for f in d["failures"]]
    doc = {"gate": f"gates_on_{scale}", "model": model, "scale": scale, "conversations": len(docs), "requests": sum(len(d["rows"]) for d in docs.values()),
           "failed_requests": failures, "gates": {}}
    G = doc["gates"]
    G["gate1_output_equals_impress"] = dict(compare(docs, HYPER, IMP, selection=True),
                                            criterion="generated tokens (T1-T3) and per-layer used sets (T2-T3) equal the IMPRESS arm executed in the same "
                                                      "session, every request")
    if scale == "gate":
        off = compare(docs, HYPER_OFF, IMP, fields=BYTE_FIELDS + CALL_FIELDS)
        off["also_reported_per_layer_used_set"] = {k: v for k, v in compare(docs, HYPER_OFF, IMP, selection=True).items() if k in ("mismatches", "ok")}
        G["gate2_prefetch_off_equals_impress"] = dict(off, criterion="prefetch off: generated tokens, read bytes (total, probe, K, V) and read call counts "
                                                                     "equal the IMPRESS arm executed in the same session, every request")
        G["gate3_unread_rows_not_used"] = nan_guard(docs)
        rep = compare(docs, HYPER_REPEAT, HYPER)
        a = [C.rows_by_key(d)[(HYPER, t)]["result"]["hyperinfer"]["prefetched_bytes"] for d in docs.values() for t in (2, 3) if tokens(C.rows_by_key(d).get((HYPER, t))) is not None
             and tokens(C.rows_by_key(d).get((HYPER_REPEAT, t))) is not None]
        b = [C.rows_by_key(d)[(HYPER_REPEAT, t)]["result"]["hyperinfer"]["prefetched_bytes"] for d in docs.values() for t in (2, 3) if tokens(C.rows_by_key(d).get((HYPER, t))) is not None
             and tokens(C.rows_by_key(d).get((HYPER_REPEAT, t))) is not None]
        rep["prefetched_bytes"] = {"first_mean": float(np.mean(a)) if a else None, "second_mean": float(np.mean(b)) if b else None,
                                   "requests_with_a_different_amount": int(sum(x != y for x, y in zip(a, b))), "requests": len(a)}
        G["gate4_repeat"] = dict(rep, criterion="the same request executed a second time: generated tokens equal, every request (the prefetched amount may differ)")
        G["gate5_no_read_started_after_stop"] = stop_check(docs, HC.PREFETCH_ROLES)
        doc["no_failed_request"] = not failures
        ok = all(g["ok"] for g in G.values()) and not failures
    else:
        G["gate5_no_read_started_after_stop"] = stop_check(docs, [HYPER])
        rec = against_recorded(model, docs)
        rec["first_mismatches"] = rec.pop("mismatch_list")[:SHOW]
        G["gate6_tokens_equal_recorded_impress"] = dict(rec, criterion="HyperInfer's generated tokens (T1, T2, T3) equal the tokens the main experiment recorded for "
                                                                       "IMPRESS, same conversation and turn, every request of the pilot")
        doc["gate8_pilot_table"] = pilot_table(model, docs)
        ok = all(g["ok"] for g in G.values())
    doc["turn1_tokens_identical_across_arms"] = t1_identical(docs, HC.PHASE_ROLES[scale])
    doc["failed"] = [k for k, g in G.items() if not g["ok"]] + (["no_failed_request"] if scale == "gate" and failures else [])
    doc["status"] = "PASS" if ok else "FAIL"
    return write_once(gate_path(res, model, f"gate_{scale}"), doc, config)


def stage_overflow_select(res, config, model):
    index, _ = C.load_index()
    docs = C.committed(res, model, "pilot", index)
    above = [{"conversation_id": cid, "method": r["method"], "turn": r["turn_id"], "input_tokens": r["input_tokens"]}
             for cid, d in docs.items() for r in d["rows"] if r["role"] == HYPER and r["status"] == "ok" and r["input_above_4096"]]
    path = C.model_dir(res, model) / "overflow_selection.json"
    if path.exists():
        raise FileExistsError(path)
    if above:
        doc = {"needed": False, "reason": "the pilot contains HyperInfer inputs above 4096 tokens", "pilot_requests_above_4096": above, "ordinals": []}
    else:
        ranked = []
        for conv in index:
            ref = HC.recorded_rows(model, conv["conversation_id"])
            ranked.append((max(ref[(C.IMP, t)]["input_tokens"] for t in (1, 2, 3) if ref[(C.IMP, t)]["status"] == "ok"), conv["ordinal"], conv["conversation_id"]))
        ranked.sort(reverse=True)
        chosen = ranked[:C.OVERFLOW_CONVERSATIONS]
        doc = {"needed": True, "reason": "the pilot contains no HyperInfer input above 4096 tokens", "pilot_requests_above_4096": [],
               "selection_rule": "the conversations with the largest IMPRESS input in the main experiment's record",
               "ordinals": [o for _, o, _ in chosen],
               "selected": [{"conversation_id": c, "ordinal": o, "recorded_impress_largest_input_tokens": n, "above_4096_in_record": n > C.CONTEXT_LIMIT}
                            for n, o, c in chosen],
               "note": None if all(n > C.CONTEXT_LIMIT for n, _, _ in chosen) else
               "no IMPRESS input of this model exceeded 4096 tokens in the main experiment; the conversations with the longest inputs are run instead"}
    doc.update(model=model, config_sha256=config["_file_sha256"], at_utc=C.utc())
    C.atomic_json(path, doc)
    print(json.dumps({"overflow_selection": {k: doc[k] for k in ("needed", "ordinals")}}), flush=True)


def stage_overflow_check(res, config, model):
    index, _ = C.load_index()
    sel = C.read_json(C.model_dir(res, model) / "overflow_selection.json")
    if not sel["needed"]:
        return write_once(gate_path(res, model, "gate8_overflow"), {"gate": "gate8_inputs_above_4096", "status": "COVERED_BY_PILOT",
                                                                     "pilot_requests_above_4096": sel["pilot_requests_above_4096"]}, config)
    docs = C.committed(res, model, "overflow", index)
    rows = [{"conversation_id": cid, "method": r["method"], "turn": r["turn_id"], "status": r["status"], "input_tokens": r.get("input_tokens"),
             "generated_tokens": r.get("generated_tokens"), "ttft_ms": r.get("ttft_ms"), "failure": r.get("failure"),
             "tokens_equal_recorded_impress": tokens(r) is not None and tokens(r) == tokens((HC.recorded_rows(model, cid) or {}).get((C.IMP, r["turn_id"])))}
            for cid, d in docs.items() for r in d["rows"]]
    failed = [r for r in rows if r["status"] != "ok"]
    above = [r for r in rows if r["status"] == "ok" and r["input_tokens"] > C.CONTEXT_LIMIT]
    return write_once(gate_path(res, model, "gate8_overflow"),
                      {"gate": "gate8_inputs_above_4096", "status": "PASS" if rows and not failed else "FAIL", "selection": sel, "requests": len(rows),
                       "failed_requests": failed, "requests_above_4096": len(above), "rows": rows,
                       "note": "recorded; not a condition of the admission (gates 1-7 are)"}, config)


def stage_tests(res, config):
    if "_dev_" in res.name and os.environ.get("CONVBENCH_MAIN_DEV_SKIP_TESTS") == "1":      # development rehearsals only; never a measurement
        return write_once(gate_path(res, None, "tests"), {"gate": "gate7_existing_tests", "status": "PASS", "suites": None,
                                                          "rehearsal_note": "test suites skipped in a development rehearsal"}, config)
    S = HC._load("_convbench_hyperinfer_suite", "223_validate_mt_gqa_main_v2.py")      # run_suite + the recorded pre-existing failures per environment
    suites = [S.run_suite(sys.executable), S.run_suite(S.PY_CROSS)]
    new = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", NEW_TEST], capture_output=True, text=True, cwd=ROOT)
    ran = re.search(r"^Ran (\d+) tests", new.stdout + new.stderr, re.M)
    new_ok = new.returncode == 0 and ran is not None and int(ran.group(1)) > 0
    ok = all(r["status"] == "PASS" for r in suites) and new_ok
    return write_once(gate_path(res, None, "tests"),
                      {"gate": "gate7_existing_tests", "status": "PASS" if ok else "FAIL", "suites": suites,
                       "new_unit_tests": {"file": f"tests/{NEW_TEST}", "ran": int(ran.group(1)) if ran else 0, "ok": new_ok},
                       "failed": [] if ok else [f for r in suites for f in r["new_failures"]],
                       "criterion": "no failing test other than the failures that existed before this work (listed per environment in "
                                    "baseline_failures_for_this_environment)"}, config)


def stage_admit(res, config, model):
    def status(path):
        doc = C.read_json(path)
        return "MISSING" if doc is None else doc.get("status", "MISSING")
    parts = {"gate7_tests": status(gate_path(res, None, "tests")),
             "gates_1_to_5_on_4_conversations": status(gate_path(res, model, "gate_gate")),
             "gates_1_5_6_on_pilot_40": status(gate_path(res, model, "gate_pilot"))}
    ok = all(v == "PASS" for v in parts.values())
    reasons = []
    for name in ("gate_gate", "gate_pilot"):
        doc = C.read_json(gate_path(res, model, name)) or {}
        for gname, g in doc.get("gates", {}).items():
            if not g.get("ok"):
                reasons.append({"where": name, "gate": gname, "detail": {k: v for k, v in g.items() if not k.startswith("first_")},
                                "examples": (g.get("first_mismatches") or g.get("first_violations") or g.get("first_guard_failures") or [])[:5]})
        if doc.get("failed_requests") and name == "gate_gate":
            reasons.append({"where": name, "gate": "no_failed_request", "failed_requests": doc["failed_requests"][:10]})
    tests = C.read_json(gate_path(res, None, "tests")) or {}
    if tests.get("status") not in ("PASS", None):
        reasons.append({"where": "tests", "gate": "gate7_existing_tests", "new_failures": tests.get("failed")})
    return write_once(gate_path(res, model, "admission"),
                      {"gate": "admission", "model": model, "status": "PASS" if ok else "FAIL", "parts": parts, "failed_gates": reasons,
                       "rule": "the model's main phase starts only if gates 1-7 are PASS; the pilot table and the inputs-above-4096 check (gate 8) are "
                               "recorded, not a condition; TTFT against IMPRESS is not a gate"}, config)


def stage_identity(res, config, model):
    """After the main phase: do all 1,731 requests reproduce the recorded IMPRESS tokens?  Decides whether the judge runs."""
    index, _ = C.load_index()
    if not C.phase_complete(res, model, "main"):
        raise SystemExit(f"the main phase of {model} is not complete")
    docs = C.committed(res, model, "main", index)
    out = against_recorded(model, docs)
    expected = len(C.phase_conversations(res, model, "main", index)) * 3
    all_equal = out["ok"] and out["requests"] == expected
    forced = "_dev_" in res.name and os.environ.get("CONVBENCH_HYPERINFER_DEV_FORCE_JUDGE") == "1"      # rehearsals only: exercise the judge path
    by_conv = {}
    for m in out["mismatch_list"]:
        by_conv.setdefault(m["conversation_id"], []).append(m["turn_id"])
    doc = {"gate": "identity_with_recorded_impress", "model": model, "status": "IDENTICAL" if all_equal else "DIFFERENT",
           "requests": out["requests"], "expected_requests": expected, "equal": out["equal"], "mismatches": out["mismatches"],
           "equal_per_turn": out["equal_per_turn"], "conversations_with_a_mismatch": len(by_conv),
           "first_mismatching_turn_per_conversation": {c: min(t) for c, t in by_conv.items()},
           "mismatch_list": out["mismatch_list"],
           "decision": "the recorded IMPRESS judgments are used as HyperInfer's" if all_equal and not forced else
                       "every HyperInfer answer of this model is judged with the frozen judge",
           "judge_needed": bool(not all_equal or forced), "rehearsal_forced_judge": bool(forced),
           "criterion": "generated tokens of all requests of the main phase (T1, T2, T3) equal the tokens the main experiment recorded for IMPRESS; a failed "
                        "request counts as a mismatch"}
    write_once(gate_path(res, model, "identity"), doc, config)
    return dict(doc, status="PASS")


def stage_posthoc(res, config):
    """After the run; reported, not a gate: files outside this run are as they were before the first measurement."""
    E = HC.load_with_common("_convbench_hyperinfer_runner", "217_eval_convbench_main.py")
    inventory = E.Backend("next").B.stat_inventory
    before = json.loads((res / "protection_before.json").read_text())
    after = inventory(res)
    changed = None
    if after["sha256"] != before["sha256"]:      # which files: compare the listings
        import gzip
        with gzip.open(res / "protection_before.tsv.gz", "rt") as f:
            old = dict(line.rstrip("\n").split("\t", 1) for line in f)
        after = inventory(res, res / "protection_after.tsv.gz")
        with gzip.open(res / "protection_after.tsv.gz", "rt") as f:
            now = dict(line.rstrip("\n").split("\t", 1) for line in f)
        changed = {"modified": sorted(k for k in old if k in now and now[k] != old[k])[:200], "removed": sorted(k for k in old if k not in now)[:200],
                   "added": sorted(k for k in now if k not in old)[:200],
                   "counts": {"modified": sum(1 for k in old if k in now and now[k] != old[k]), "removed": sum(1 for k in old if k not in now),
                              "added": sum(1 for k in now if k not in old)}}
    source = C.freeze_violations(config["source_sha256"])
    out = {"status": "REPORTED", "existing_files_unchanged": after["sha256"] == before["sha256"], "before": before, "after": after, "changed": changed,
           "frozen_source_changed": source, "config_sha256": config["_file_sha256"], "at_utc": C.utc(),
           "note": "added files are other work in the repository; modified or removed files of results/, runs/ and kvstore* would be a violation"}
    C.atomic_json(res / "gates" / "posthoc.json", out)
    print(json.dumps({"existing_files_unchanged": out["existing_files_unchanged"], "frozen_source_changed": source}), flush=True)
    return dict(out, status="PASS")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", type=Path, required=True)
    p.add_argument("--stage", choices=("tests", "gate", "overflow-select", "overflow-check", "admit", "identity", "posthoc"), required=True)
    p.add_argument("--model", choices=C.MODELS)
    p.add_argument("--scale", choices=("gate", "pilot"))
    args = p.parse_args()
    res = args.res.resolve()
    assert res.parent == ROOT / "results" and res.name.startswith(C.RUN_PREFIX)
    config = HC.verify_freeze(res)
    if args.stage == "tests":
        doc = stage_tests(res, config)
    elif args.stage == "gate":
        doc = stage_gate(res, config, args.model, args.scale)
    elif args.stage == "overflow-select":
        return stage_overflow_select(res, config, args.model)
    elif args.stage == "overflow-check":
        doc = stage_overflow_check(res, config, args.model)
    elif args.stage == "admit":
        doc = stage_admit(res, config, args.model)
    elif args.stage == "identity":
        doc = stage_identity(res, config, args.model)
    else:
        doc = stage_posthoc(res, config)
    sys.exit(0 if doc.get("status") in ("PASS", "COVERED_BY_PILOT") else 1)


if __name__ == "__main__":
    main()

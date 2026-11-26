#!/usr/bin/env python3
"""Gates of the MT-GQA HyperInfer run (contract docs/mt_gqa_hyperinfer_contract.md, section 6).

  --stage tests                gate 7: the existing test suites, both environments (no new failure)
  --stage logic                gate 1: tests/test_hyperinfer.py with the public checkout (logic comparison)
  --stage gate  --model M      gates 2, 3, 5, 6 on the 4 gate images
  --stage dev   --model M      gates 2, 3, 4, 5, 6 on the 40 DEV images; gate 8 (TTFT continuity) is reported
  --stage admit --model M      PASS iff tests, logic, gate and dev of that model are PASS for this freeze
  --stage posthoc              after the main phases: nothing outside this run changed (reported)

Every stage writes runs/<run>/gates/<name>.json once; a recorded result is not re-judged.
Criteria are the ones frozen in the contract; a failing gate is recorded with its mismatches.
"""
from __future__ import annotations

import argparse
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

from mmimpress.mtvqa25k.adapter import read_rows  # noqa: E402


def load_script(name, filename):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


E = load_script("_mt_gqa_hyperinfer_eval", "225_eval_mt_gqa_hyperinfer.py")
sha, atomic_json = E.sha, E.atomic_json
FULL, IMPRESS, HYPER, HYPER_OFF, HYPER_SYNC, HYPER_REPEAT, HYPER_NAN = (
    E.FULL, E.IMPRESS, E.HYPER, E.HYPER_OFF, E.HYPER_SYNC, E.HYPER_REPEAT, E.HYPER_NAN)
NEW_TEST = "tests.test_hyperinfer"
BYTE_FIELDS = ("ssd_read_bytes", "probe_read_bytes", "k_read_bytes", "v_read_bytes")
CALL_FIELDS = ("ssd_preads", "pread_calls", "preadv_calls")
SHOW = 20


def frozen(top):
    config = json.loads((top / "config.json").read_text())
    config["_file_sha256"] = sha(top / "config.json")
    return config


def receipt(top, name, doc, config):
    out = top / "gates" / f"{name}.json"
    out.parent.mkdir(exist_ok=True)
    doc = dict(doc, config_sha256=config["_file_sha256"], written_unix=time.time())
    if out.exists():      # a superseded receipt is kept, never overwritten
        os.replace(out, out.with_name(f"{name}.superseded_{time.strftime('%Y%m%dT%H%M%S')}.json"))
    atomic_json(out, doc)
    print(json.dumps({"stage": name, "status": doc["status"], "failed": doc.get("failed")}, ensure_ascii=False), flush=True)
    return doc


def committed_rows(run, phase, images=None):
    """{(method, dialog_id, turn): row} of the committed images of a phase."""
    rows = {}
    for cp in sorted((run / phase / "images").glob("*/COMMITTED.json")):
        if images is not None and cp.parent.name not in images:
            continue
        doc = json.loads(cp.read_text())
        got = read_rows(cp.parent / doc["attempt_id"] / "raw.jsonl")
        assert len(got) == doc["requests"]
        for r in got:
            rows[(r["method_id"], r["dialog_id"], r["turn_id"])] = r
    return rows


def tokens(row):
    return row["generated_token_ids"]


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
def compare(rows, a, b, keys, *, selection=False, fields=()):
    """Requests of arm ``a`` against the same requests of arm ``b``."""
    bad, n, followups = [], 0, 0
    for d, t in keys:
        ra, rb = rows[(a, d, t)], rows[(b, d, t)]
        n += 1
        diff = []
        if tokens(ra) != tokens(rb):
            diff.append("tokens")
        if t > 1:
            followups += 1
            if selection:
                sa, sb = ra["result"]["selection_digest"], rb["result"]["selection_digest"]
                if sa["all_layers"] != sb["all_layers"]:
                    layers = [i for i, (x, y) in enumerate(zip(sa["per_layer"], sb["per_layer"])) if x != y]
                    diff.append(f"selection(layers {layers[:8]})")
            for f in fields:
                if ra[f] != rb[f]:
                    diff.append(f"{f}: {ra[f]} != {rb[f]}")
        if diff:
            bad.append({"dialog_id": d, "turn_id": t, "difference": diff, a: tokens(ra), b: tokens(rb)})
    return {"a": a, "b": b, "requests": n, "followup_requests": followups, "mismatches": len(bad), "first_mismatches": bad[:SHOW],
            "compared": ["generated tokens (T1, T2, T3)"] + (["per-layer used set (T2, T3)"] if selection else []) + [f"{f} (T2, T3)" for f in fields],
            "status": "PASS" if not bad else "FAIL"}


def gate5(rows, keys):
    bad, n = [], 0
    totals = {"unread_rows": 0, "used_rows": 0, "attention_calls": 0}
    for d, t in keys:
        if t == 1:
            continue
        n += 1
        g = rows[(HYPER_NAN, d, t)]["result"].get("unread_row_guard")
        ok = bool(g) and g["all_outputs_finite"] and g["used_equals_selection"] and g["used_rows_all_read"]
        if not ok:
            bad.append({"dialog_id": d, "turn_id": t, "guard": g})
        elif g:
            for k in totals:
                totals[k] += g[k]
    same = compare(rows, HYPER_NAN, IMPRESS, keys, selection=True)
    return {"requests": n, "guard_failures": len(bad), "first_guard_failures": bad[:SHOW], "totals": totals, "against_impress": same,
            "criterion": "every visual row of the prefix cache holds NaN before the first layer; rows are overwritten only by reads; at masked columns the "
                         "debug wrapper passes 0 (what an unread row holds in a normal request), nowhere else: the attention output of every layer is "
                         "finite, the unmasked visual columns equal the layer's used set G_l, every used row was read, and tokens and used sets equal IMPRESS",
            "status": "PASS" if not bad and same["status"] == "PASS" else "FAIL"}


def reference_rows(model, images):
    return committed_rows(E.REF_RUN[model], "main", set(images))


def gate4(rows, keys, ref):
    bad = []
    for d, t in keys:
        if tokens(rows[(HYPER, d, t)]) != tokens(ref[(IMPRESS, d, t)]):
            bad.append({"dialog_id": d, "turn_id": t, "hyperinfer": tokens(rows[(HYPER, d, t)]), "impress_recorded": tokens(ref[(IMPRESS, d, t)])})
    return {"requests": len(keys), "mismatches": len(bad), "first_mismatches": bad[:SHOW],
            "criterion": "HyperInfer's generated tokens (T1, T2, T3) equal the tokens recorded for IMPRESS in the earlier main run, same dialogue and turn",
            "status": "PASS" if not bad else "FAIL"}


def continuity(rows, keys, ref):
    """Gate 8 (reported): FullLoad and IMPRESS of this DEV phase against the same requests of the earlier main run."""
    out = {}
    for method in (FULL, IMPRESS):
        entry = {}
        for name, turns in (("followup", (2, 3)), ("turn1", (1,))):
            now = np.array([rows[(method, d, t)]["ttft_ms"] for d, t in keys if t in turns])
            old = np.array([ref[(method, d, t)]["ttft_ms"] for d, t in keys if t in turns])
            entry[name] = {"requests": int(now.size), "this_run_mean_ms": float(now.mean()), "earlier_run_mean_ms": float(old.mean()),
                           "ratio_of_means": float(now.mean() / old.mean()), "this_run_p50_ms": float(np.median(now)),
                           "earlier_run_p50_ms": float(np.median(old)), "mean_of_paired_differences_ms": float((now - old).mean())}
        same = sum(tokens(rows[(method, d, t)]) == tokens(ref[(method, d, t)]) for d, t in keys)
        entry["tokens_equal_to_earlier_run"] = {"requests": len(keys), "equal": int(same)}
        out[method] = entry
    return {"status": "REPORTED", "methods": out,
            "note": "not a gate; this DEV phase rotates 7 arms, the earlier main run rotated 4 (ReComp, FullLoad, IMPRESS, Ours)"}


def judge(top, model, phase, config):
    manifest, keys, execution = phase_state(top, model, phase, config)
    name = f"{model}_{phase}"
    if execution["status"] != "PASS":
        return receipt(top, name, {"status": "FAIL", "model": model, "phase": phase, "execution": execution, "failed": ["execution"]}, config)
    rows = committed_rows(top / E.MODEL_DIR[model], phase)
    gates = {
        "gate2_prefetch_off_equals_impress": dict(compare(rows, HYPER_OFF, IMPRESS, keys, selection=True, fields=BYTE_FIELDS + CALL_FIELDS),
                                                  criterion="prefetch off: generated tokens, per-layer used sets, read bytes and read call counts equal the IMPRESS "
                                                            "arm executed in the same session, every request"),
        "gate3_output_equals_impress": dict(compare(rows, HYPER, IMPRESS, keys, selection=True),
                                            criterion="prefetch on: generated tokens and per-layer used sets equal IMPRESS, every request"),
        "gate5_unread_rows_not_used": gate5(rows, keys),
        "gate6_sync_equals_async": dict(compare(rows, HYPER_SYNC, HYPER, keys),
                                        criterion="prefetch executed on the request thread: generated tokens equal the asynchronous arm, every request"),
        "gate6_repeat": dict(compare(rows, HYPER_REPEAT, HYPER, keys),
                             criterion="the same request executed a second time: generated tokens equal, every request"),
    }
    extra = {}
    if phase == "dev":
        ref = reference_rows(model, manifest["phase_images"]["dev"])
        gates["gate4_tokens_equal_recorded_impress"] = gate4(rows, keys, ref)
        extra["gate8_ttft_continuity"] = continuity(rows, keys, ref)
    failed = [k for k, g in gates.items() if g["status"] != "PASS"]
    t1 = sum(len({tuple(tokens(rows[(m, d, 1)])) for m in config["phase_methods"][phase]}) == 1 for d, t in keys if t == 1)
    return receipt(top, name, {"status": "PASS" if not failed else "FAIL", "model": model, "phase": phase, "execution": execution,
                               "requests": len(rows), "gates": gates, "failed": failed, **extra,
                               "turn1_tokens_identical_across_arms": {"dialogues": len(keys) // 3, "identical": int(t1)}}, config)


# -------------------------------------------------------------- tests, logic
def stage_tests(top, config):
    V = load_script("_mt_gqa_hyperinfer_suite", "223_validate_mt_gqa_main_v2.py")      # run_suite + the recorded pre-existing failures per environment
    suites = [V.run_suite(sys.executable), V.run_suite(V.PY_CROSS)]
    ok = all(r["status"] == "PASS" for r in suites)
    return receipt(top, "tests", {
        "gate": "gate7_existing_tests", "status": "PASS" if ok else "FAIL", "suites": suites,
        "failed": [] if ok else [f for r in suites for f in r["new_failures"]],
        "criterion": "no failing test other than the failures that existed before this work (listed per environment in "
                     "baseline_failures_for_this_environment); tests/test_hyperinfer.py is part of both suites"}, config)


def stage_logic(top, config):
    env = dict(os.environ, PYTHONPATH=".:scripts:tests")
    assert env.get("IMPRESS_OFFICIAL_REPO"), "IMPRESS_OFFICIAL_REPO is not set"
    p = subprocess.run([sys.executable, "-m", "unittest", "-v", NEW_TEST], capture_output=True, text=True, cwd=ROOT, env=env)
    text = p.stdout + p.stderr
    ran = re.search(r"^Ran (\d+) tests", text, re.M)
    public = re.findall(r"^(test_\w+) \((?:tests\.)?test_hyperinfer\.PublicPrefetchTests\S*\) \.\.\. (\w+)", text, re.M)
    skipped = re.findall(r"\.\.\. skipped", text)
    head = subprocess.check_output(["git", "-C", env["IMPRESS_OFFICIAL_REPO"], "rev-parse", "HEAD"], text=True).strip()
    ok = p.returncode == 0 and bool(ran) and len(public) >= 2 and all(s == "ok" for _, s in public) and not skipped and head == E.OFFICIAL_COMMIT
    return receipt(top, "logic", {
        "gate": "gate1_logic_equals_public_code", "status": "PASS" if ok else "FAIL", "failed": [] if ok else ["tests.test_hyperinfer"],
        "ran": int(ran.group(1)) if ran else None, "public_comparison_tests": dict(public), "skipped": len(skipped), "public_checkout_head": head,
        "called_directly": "PrefixKVLayer.prefetch_by_chunk and _prefetch_chunks_direct_pinned of my_pcache_fast.py: order conversion, chunk ids, "
                           "budget cut, half rule, K-then-V order, prefetched token set",
        "not_callable": "the prediction (flex_opt_fast_pcache.py line 1108) and the missing tokens (lines 1125-1126) are single lines of the "
                        "1,300-line layer function, which needs the model and its global state; the same operations are written in the test and compared",
        "tail": text[-3000:]}, config)


def stage_admit(top, model, config):
    steps = {}
    for name in ("tests", "logic", f"{model}_gate", f"{model}_dev"):
        path = top / "gates" / f"{name}.json"
        doc = json.loads(path.read_text()) if path.exists() else {"status": "MISSING"}
        ok = doc.get("status") == "PASS" and doc.get("config_sha256") == config["_file_sha256"]
        steps[name] = {"status": doc.get("status"), "bound_to_this_freeze": doc.get("config_sha256") == config["_file_sha256"], "ok": ok,
                       "failed": doc.get("failed")}
    status = "PASS" if all(s["ok"] for s in steps.values()) else "FAIL"
    return receipt(top, f"{model}_admission", {
        "status": status, "model": model, "steps": steps, "failed": [k for k, s in steps.items() if not s["ok"]],
        "rule": "the model's main phase starts only if gates 1-7 are PASS (gates 2, 3, 5, 6 on the 4 gate images and on the 40 DEV images, gate 4 on "
                "the DEV images); TTFT against IMPRESS is not a gate"}, config)


def stage_posthoc(top, config):
    """After the main phases; reported, not a gate: files outside this run are as they were at the freeze."""
    B = E.Backend("next", top)
    before = json.loads((top / "protection_before.json").read_text())
    after = B.M.stat_inventory(top)
    changed = None
    if after["sha256"] != before["sha256"]:      # which files: compare the listings
        import gzip
        with gzip.open(top / "protection_before.tsv.gz", "rt") as f:
            old = dict(line.rstrip("\n").split("\t", 1) for line in f)
        now = {}
        after = B.M.stat_inventory(top, top / "protection_after.tsv.gz")
        with gzip.open(top / "protection_after.tsv.gz", "rt") as f:
            now = dict(line.rstrip("\n").split("\t", 1) for line in f)
        changed = {"modified": sorted(k for k in old if k in now and now[k] != old[k])[:200], "removed": sorted(k for k in old if k not in now)[:200],
                   "added": sorted(k for k in now if k not in old)[:200]}
    now = E.source_hashes()
    source = sorted(k for k, v in config["source_sha256"].items() if now.get(k) != v)
    out = {"status": "REPORTED", "existing_files_unchanged": after["sha256"] == before["sha256"], "before": before, "after": after, "changed": changed,
           "frozen_source_changed": source, "config_sha256": config["_file_sha256"], "written_unix": time.time(),
           "note": "added files are other work in the repository; modified or removed files of results/, runs/ and kvstore* would be a violation"}
    atomic_json(top / "gates" / "posthoc.json", out)
    print(json.dumps({"existing_files_unchanged": out["existing_files_unchanged"], "frozen_source_changed": source}))
    return dict(out, status="PASS")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--stage", choices=("tests", "logic", "gate", "dev", "admit", "posthoc"), required=True)
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
    elif args.stage == "logic":
        doc = stage_logic(top, config)
    elif args.stage in ("gate", "dev"):
        doc = judge(top, args.model, args.stage, config)
    elif args.stage == "admit":
        doc = stage_admit(top, args.model, config)
    else:
        doc = stage_posthoc(top, config)
    sys.exit(0 if doc["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

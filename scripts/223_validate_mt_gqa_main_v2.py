#!/usr/bin/env python3
"""Gates of the LLaVA-NeXT x MT-GQA main experiment v2 (contract docs/mt_gqa_main_v2_contract.md, section 6).

  --stage tests     gate 7: the whole unit-test suite in the run environment and, as a cross-check, in the
                    transformers 5.13.1 environment; the new test file on its own
  --stage tokens    CPU.  gate 1: for every prompt the earlier main runs sent (12,183 (dialogue, turn) x the
                    histories of the four methods), the fixed path's ids == the original-LLaVA ids;
                    gate 2: the LLaVA-1.5 path's ids are what they were before this work (60 prompts).
                    Also writes the slim copy of the earlier runs' rows the report compares against.
  --stage legacy    gate 3: with tokenization "legacy", ReComp's generated tokens == the 2026-11-20 run (4 images)
  --stage gate      gates 4, 5, 6 on the 4 gate images
  --stage devgate   gates 4, 5 on the 40 DEV images
  --stage dev       gate 6 on the 40 DEV images and the execution of the DEV pilot (gate 8)
  --stage admit     writes gates/admission.json (PASS only if every gate file is PASS for this freeze);
                    the main phase refuses to start without it
  --stage posthoc   after the main phase (reported, not a gate): turn-1 identity and the per-request token
                    check over all 398 images

Criteria are constants of this file, frozen with the run's source hash.  A FAIL is never re-judged: a gate
file that already exists for this freeze is kept (``.superseded_`` copies only appear when a stage is run
again by hand).  Generated tokens that differ from the earlier runs under "official" are not a gate: the
token sequence changed.  They are counted by the report.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mmimpress import mt_gqa_tokens as TOK  # noqa: E402
from mmimpress.mtvqa25k.adapter import read_rows  # noqa: E402


def load_script(name, filename):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


H = load_script("_mtgqa_main_v2_eval", "222_eval_mt_gqa_main_v2.py")
G = load_script("_mtgqa_async_gates", "188_validate_mt_gqa_async.py")          # run_suite, BASELINE_FAILURES (shared helper)
sha, atomic_json = H.sha, H.atomic_json
PY_CROSS = str(ROOT / ".venv-glm46v/bin/python")
NEW_TEST = "test_mt_gqa_main_v2.py"
BYTE_FIELDS = ("ssd_read_bytes", "probe_read_bytes", "k_read_bytes", "v_read_bytes", "packed_read_bytes")
REQUIRED = ("tests", "tokens", "legacy", "gate", "devgate", "dev")
# the earlier main runs: which run holds each method's previous rows (report item 4, gates 1 and 3)
PREVIOUS = {H.RECOMP: H.OURS_REF_RUN, H.FULL_ASYNC: H.OURS_REF_RUN, H.ORIG_ASYNC: H.OURS_REF_RUN, H.NEW_ASYNC: H.IMPL_REF_RUN}
# Tests that failed before this work in the cross-check environment (transformers 5.13.1) and are not in the shared baseline of
# scripts/188: tests/test_convbench_main.py (2026-11-22) asserts the interpreter of the reference runs and the slow legacy tokenizer
# of transformers 4.x.  Verified 2026-11-24 with every file of this work moved out of the tree: both fail the same way.
EXTRA_BASELINE_FAILURES = {
    "5": {"test_official_token_check_on_both_tokenizers (test_convbench_main.FrozenSettings)",
          "test_reference_checks_pass_and_source_is_unchanged (test_convbench_main.ReferenceHarness)"},
}
L15_RUN = ROOT / "runs/llava15_mt_gqa_main_20261120T165001Z"                  # the LLaVA-1.5 x MT-GQA main run (gate 2 reference)
L15_PER_TURN = 20
ORIG_META = f"{H.ORIG_KIND}_meta.json"


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
    print(json.dumps({"stage": name, "status": doc["status"], "failed": doc.get("failed")}, ensure_ascii=False), flush=True)
    return doc


# ---------------------------------------------------------------- committed io
def committed_rows(run, phase):
    """{(method, dialog_id, turn): row} of every committed image of a phase."""
    rows = {}
    for cp in sorted((run / phase / "images").glob("*/COMMITTED.json")):
        doc = json.loads(cp.read_text())
        got = read_rows(cp.parent / doc["attempt_id"] / "raw.jsonl")
        assert len(got) == doc["requests"]
        for r in got:
            rows[(r["method_id"], r["dialog_id"], r["turn_id"])] = r
    return rows


def tokens(row):
    return row["result"]["generated_token_ids"]


def phase_state(run, phase, config):
    manifest = json.loads((run / "manifest.json").read_text())
    keys = sorted((d, t) for d in manifest["phase_dialogue_ids"][phase] for t in (1, 2, 3))
    path = run / phase / "validation.json"
    v = json.loads(path.read_text()) if path.exists() else {"status": "MISSING"}
    ok = (v.get("status") == "PASS" and v.get("requests") == manifest["expected_requests"][phase]
          and v.get("config_sha256") == config["_file_sha256"] and v.get("independent_image_audit") == "PASS")
    execution = dict(v, status="PASS" if ok else "FAIL", expected=manifest["expected_requests"][phase],
                     criterion="every image of the phase committed after its independent audit passed, for this freeze")
    return manifest, keys, execution


def compare(rows, a, b, keys, fields=()):
    """Per-request equality of the generated tokens (and ``fields``) between arms ``a`` and ``b``."""
    out = {"arms": [a, b], "requests": 0, "tokens_differ": 0, "tokens_differ_same_prompt": 0, "prompt_differs": 0,
           "fields_differ": {f: 0 for f in fields}, "missing": 0, "examples": []}
    for dialog, turn in keys:
        ra, rb = rows.get((a, dialog, turn)), rows.get((b, dialog, turn))
        if ra is None or rb is None:
            out["missing"] += 1
            continue
        out["requests"] += 1
        same_prompt = ra["prompt"] == rb["prompt"]
        out["prompt_differs"] += not same_prompt
        bad = [f for f in fields if ra[f] != rb[f]]
        for f in bad:
            out["fields_differ"][f] += 1
        if tokens(ra) != tokens(rb):
            out["tokens_differ"] += 1; out["tokens_differ_same_prompt"] += same_prompt
            bad.append("generated_token_ids")
        if bad and len(out["examples"]) < 30:
            out["examples"].append({"image_id": ra["image_id"], "dialog_id": dialog, "turn_id": turn, "fields": bad, "same_prompt": same_prompt,
                                    a: tokens(ra), b: tokens(rb), "predictions": [ra["prediction"], rb["prediction"]]})
    out["status"] = "PASS" if (out["requests"] == len(keys) and not out["missing"] and not out["tokens_differ"]
                               and not any(out["fields_differ"].values())) else "FAIL"
    return out


def gates_4_5(rows, keys):
    g4a = dict(compare(rows, H.ORIG, H.ORIG_ASYNC, keys, ("ssd_read_bytes", "packed_read_bytes")),
               criterion="Ours: generated tokens and bytes read identical between the synchronous and the asynchronous arm, every request")
    g4b = dict(compare(rows, H.ORIG_ASYNC, H.ORIG_REPEAT, keys),
               criterion="Ours executed twice on the same requests: generated tokens identical, every request")
    g4 = {"sync_vs_async": g4a, "executed_twice": g4b, "status": "PASS" if g4a["status"] == g4b["status"] == "PASS" else "FAIL"}
    g5a = dict(compare(rows, H.NEW_R100, H.FULL_REORDERED, keys),
               criterion="IMPRESS at 100 % retention vs the FullLoad loader reading the same reordered store: generated tokens identical, every request")
    g5b = dict(compare(rows, H.NEW, H.NEW_ASYNC, keys, BYTE_FIELDS),
               criterion="IMPRESS: generated tokens and bytes read (total and by kind) identical between the synchronous and the asynchronous arm, every request")
    g5 = {"retention_100_vs_fullload_loader_on_reordered_store": g5a, "sync_vs_async": g5b,
          "status": "PASS" if g5a["status"] == g5b["status"] == "PASS" else "FAIL"}
    return g4, g5


def turn1_identity(rows, keys, methods, judged):
    """Turn-1 generated tokens identical across ``methods``, per dialogue."""
    dialogs = sorted({d for d, t in keys if t == 1}); bad = []
    for d in dialogs:
        got = {m: tokens(rows[(m, d, 1)]) for m in methods if (m, d, 1) in rows}
        if len(got) != len(methods) or len({tuple(v) for v in got.values()}) != 1:
            bad.append({"dialog_id": d, "tokens": got})
    return {"criterion": ("turn-1 generated tokens identical across the four methods (ReComp, FullLoad, IMPRESS, Ours), every dialogue" if judged
                          else "none (reported only): turn-1 generated tokens identical across every arm of the phase, per dialogue"),
            "dialogues": len(dialogs), "arms": list(methods), "mismatch_count": len(bad), "mismatches": bad[:20],
            "status": ("PASS" if dialogs and not bad else "FAIL") if judged else "REPORTED"}


def token_record(rows, keys, methods, tokenization):
    """The per-request token check the harness recorded (re-derived by every image audit)."""
    n = ok = 0
    for m in methods:
        for d, t in keys:
            r = rows.get((m, d, t))
            if r is None:
                continue
            n += 1; ok += bool(r["official_token_ids_match"]) and r["tokenization"] == tokenization
    return {"requests": n, "requests_whose_ids_equal_original_llava": ok, "tokenization": tokenization}


def sync_async_timing(rows, keys, pairs):
    """Reference only (not a criterion): synchronous vs asynchronous follow-up TTFT."""
    out = {}
    for name, a, b in pairs:
        fu = [(d, t) for d, t in keys if t > 1 and (a, d, t) in rows and (b, d, t) in rows]
        if not fu:
            continue
        sa = np.array([rows[(a, d, t)]["ttft_ms"] for d, t in fu]); sb = np.array([rows[(b, d, t)]["ttft_ms"] for d, t in fu])
        out[name] = {"sync_mean_ms": float(sa.mean()), "async_mean_ms": float(sb.mean()), "paired_mean_difference_ms": float((sb - sa).mean()),
                     "requests": int(sa.size)}
    return out


# ------------------------------------------------------------------ gate 7
def run_suite(python):
    """``run_suite`` of scripts/188 with the failures that pre-date this work in that environment (see EXTRA_BASELINE_FAILURES)."""
    out = G.run_suite(python)
    major = (out.get("transformers") or "").split(".")[0]
    baseline = set(out["baseline_failures_for_this_environment"]) | EXTRA_BASELINE_FAILURES.get(major, set())
    out["baseline_failures_for_this_environment"] = sorted(baseline)
    out["new_failures"] = [f for f in out["failed"] if f not in baseline]
    out["status"] = "PASS" if out["ran"] and not out["new_failures"] else "FAIL"
    return out


def stage_tests(run, config):
    suites = [run_suite(sys.executable), run_suite(PY_CROSS)]
    p = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", NEW_TEST], capture_output=True, text=True, cwd=ROOT)
    ran = re.search(r"^Ran (\d+) tests", p.stdout + p.stderr, re.M)
    ok = all(r["status"] == "PASS" for r in suites) and p.returncode == 0 and bool(ran)
    return receipt(run, "tests", {
        "gate": "gate7_existing_tests", "status": "PASS" if ok else "FAIL", "suites": suites,
        "new_unit_tests": {"file": f"tests/{NEW_TEST}", "ran": int(ran.group(1)) if ran else None, "ok": p.returncode == 0},
        "criterion": "no failing test other than the failures that existed before this work (transformers 4.57.6: the 3 GLM-4.6V tests; "
                     "transformers 5.13.1: 2 vision-capture tests and 2 tests of tests/test_convbench_main.py that assert the "
                     "transformers 4.x run environment); the new test file passes"}, config)


# ------------------------------------------------- previous rows (slim copy)
def _slim_image(args):
    """One committed image of an earlier run -> slim rows of the wanted methods (worker; never raises)."""
    folder, methods = Path(args[0]), set(args[1])
    try:
        commit = json.loads((folder / "COMMITTED.json").read_text())
        attempt = folder / commit["attempt_id"]
        out = []
        for r in read_rows(attempt / "raw.jsonl"):
            if r["method_id"] in methods:
                out.append({"m": r["method_id"], "d": r["dialog_id"], "t": r["turn_id"], "i": r["image_id"], "prompt": r["prompt"],
                            "sfx": r["suffix_input_ids"], "tok": r["result"]["generated_token_ids"], "score": r["score"], "ttft": r["ttft_ms"],
                            "pred": r["prediction"], "bytes": r["ssd_read_bytes"], "reads": r["ssd_preads"]})
        table = None
        meta = attempt / ORIG_META
        if meta.exists() and H.ORIG_ASYNC in methods:
            m = json.loads(meta.read_text())
            table = {"idx": m["stored_original_indices"], "thr": m["stored_min_keep_count"], "counts": m["layer_keep_counts"],
                     "v_token_num": m["v_token_num"], "newline_idx": m["newline_idx"]}
        return {"image_id": folder.name, "status": "OK", "rows": out, "ours_table": table, "raw_sha256": commit["raw_sha256"]}
    except Exception as exc:      # noqa: BLE001  (reported by the caller)
        return {"image_id": folder.name, "status": "ERROR", "error": repr(exc)}


def reference_dir(run):
    return run / "reference"


def build_previous(run, images):
    """Slim copy of the earlier main runs' rows for ``images`` -> <run>/reference/ (read-only on those runs)."""
    by_run = {}
    for method, ref in PREVIOUS.items():
        by_run.setdefault(ref, []).append(method)
    rows, tables, sources = [], {}, {}
    for ref, methods in by_run.items():
        folders = [ref / "main" / "images" / i for i in images]
        with Pool(8) as pool:
            got = pool.map(_slim_image, [(str(f), methods) for f in folders], chunksize=4)
        bad = [g for g in got if g["status"] != "OK"]
        if bad:
            raise RuntimeError("unreadable previous images: " + repr(bad[:3]))
        for g in got:
            rows += g["rows"]
            if g["ours_table"] is not None:
                tables[g["image_id"]] = g["ours_table"]
        sources[str(ref.relative_to(ROOT))] = {"methods": methods, "images": len(got), "rows": sum(len(g["rows"]) for g in got),
                                               "config_sha256": sha(ref / "config.json")}
    out = reference_dir(run); out.mkdir(exist_ok=True)
    for name, value in (("previous_rows.json.gz", rows), ("previous_ours_tables.json.gz", tables)):
        tmp = out / (name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(value, f, separators=(",", ":"))
        os.replace(tmp, out / name)
    atomic_json(out / "previous_sources.json", {"sources": sources, "rows": len(rows), "ours_tables": len(tables),
                                                "files_sha256": {n: sha(out / n) for n in ("previous_rows.json.gz", "previous_ours_tables.json.gz")}}, replace=True)
    return rows, sources


def load_previous(run):
    with gzip.open(reference_dir(run) / "previous_rows.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------------------------------------- gates 1 and 2
def gate1_tokens(previous, manifest):
    """Every prompt the earlier main runs sent, through both paths, against the original-LLaVA ids."""
    fast, slow, image_token = H.tokenizers()
    space = int(fast.convert_tokens_to_ids("▁"))
    cache = {}
    per_method = {}
    patterns = Counter()
    examples = []
    for r in previous:
        prompt = r["prompt"]
        if prompt not in cache:
            official = TOK.official_token_ids(slow, prompt)
            legacy_full = fast(TOK.tokenizer_input(prompt, TOK.TOKENIZATION_LEGACY)).input_ids
            fixed = TOK.path_token_ids(fast, image_token, prompt, TOK.TOKENIZATION_OFFICIAL)
            legacy = TOK.collapse_image(legacy_full, image_token)
            at = legacy.index(TOK.OFFICIAL_IMAGE_INDEX)
            cache[prompt] = {"fixed_match": fixed == official, "legacy_match": legacy == official,
                             "legacy_suffix": legacy_full[legacy_full.index(image_token) + 1:],
                             "one_token_after_image": official == legacy[:at + 1] + [space] + legacy[at + 1:],
                             "lengths": (len(legacy), len(fixed), len(official)), "prefix": tuple(official[:at + 1])}
        c = cache[prompt]
        s = per_method.setdefault(r["m"], {"prompts": 0, "by_turn": {1: 0, 2: 0, 3: 0}, "fixed_path_equal_original": 0, "legacy_path_equal_original": 0,
                                           "legacy_path_suffix_equal_recorded_suffix": 0, "difference_is_one_token_after_image": 0})
        s["prompts"] += 1; s["by_turn"][r["t"]] += 1
        s["fixed_path_equal_original"] += c["fixed_match"]; s["legacy_path_equal_original"] += c["legacy_match"]
        s["legacy_path_suffix_equal_recorded_suffix"] += c["legacy_suffix"] == r["sfx"]
        s["difference_is_one_token_after_image"] += c["one_token_after_image"]
        patterns[c["prefix"]] += 1
        if not c["fixed_match"] and len(examples) < 20:
            examples.append({"method_id": r["m"], "dialog_id": r["d"], "turn_id": r["t"], "prompt": prompt, "lengths_legacy_fixed_original": c["lengths"]})
    expected = len(manifest["phase_dialogue_ids"]["main"]) * 3
    methods_ok = set(per_method) == set(PREVIOUS) and all(s["prompts"] == expected for s in per_method.values())
    ok = (methods_ok and all(s["fixed_path_equal_original"] == s["prompts"] == s["legacy_path_suffix_equal_recorded_suffix"] for s in per_method.values()))
    return {"criterion": "for every prompt the earlier main runs sent -- 12,183 (dialogue, turn) pairs with the generated history of each of the four "
                         "methods -- the ids of the fixed path equal the ids of the original LLaVA procedure (tokenizer_image_token, slow legacy "
                         "tokenizer); and the ids of the legacy path are the ids those runs recorded (so the comparison is made on the path they really used)",
            "prompts_per_method_expected": expected, "prompts_total": len(previous), "distinct_prompts": len(cache), "per_method": per_method,
            "prompts_of_recomp": per_method.get(H.RECOMP, {}).get("prompts"),
            "prefix_ids_before_and_including_image": [{"ids": list(k), "prompts": v} for k, v in patterns.most_common()],
            "mismatch_examples": examples, "status": "PASS" if ok else "FAIL",
            "note": "the main phase's own follow-up prompts depend on what each method generates there; the harness repeats this comparison after "
                    "every request and stops on a mismatch, and the image audit re-derives it (reported after the main phase by --stage posthoc)"}


def gate2_llava15():
    """The LLaVA-1.5 path: its ids for 60 prompts of the LLaVA-1.5 main run are the ids that run recorded,
    and no file that run froze has changed."""
    import llava15_support as S15
    from mmimpress.llava15.runner import Llava15Runner
    source = Llava15Runner().source
    fast, slow, image_token = TOK.load_tokenizers(source)
    config = json.loads((L15_RUN / "config.json").read_text())
    methods = config["phases"]["main"]["methods"]
    manifest = json.loads((L15_RUN / "manifest.json").read_text())
    by_dialog = {d["dialog_id"]: d for d in manifest["dialogues"]}
    changed = sorted(k for k, v in config["source_sha256"].items() if not (ROOT / k).is_file() or sha(ROOT / k) != v)
    images = sorted(p.parent for p in (L15_RUN / "main" / "images").glob("*/COMMITTED.json"))
    stride = max(1, len(images) // L15_PER_TURN)
    out = {"compared": 0, "by_turn": {1: 0, 2: 0, 3: 0}, "prompt_rebuilt_equal": 0, "suffix_ids_equal_recorded": 0, "prefix_length_equal_recorded": 0,
           "equal_original_llava_ids": 0, "examples": []}
    picked = 0; index = 0
    while picked < L15_PER_TURN and index < len(images):
        commit = json.loads((images[index] / "COMMITTED.json").read_text())
        rows = [r for r in read_rows(images[index] / commit["attempt_id"] / "raw.jsonl") if r["method_id"] == "recompute"]
        index += stride
        if not rows:
            continue
        dialog_id = rows[0]["dialog_id"]
        turns = {r["turn_id"]: r for r in rows if r["dialog_id"] == dialog_id}
        if set(turns) != {1, 2, 3}:
            continue
        picked += 1
        for turn in (1, 2, 3):
            row = turns[turn]
            # the path as it is now: prompt construction of the LLaVA-1.5 harness, then the fast tokenizer of the pinned snapshot
            prompt, _, _ = S15.prompt_factory(methods, by_dialog[dialog_id], turn, "recompute", {t: turns[t] for t in range(1, turn)})
            ids = fast(prompt).input_ids
            position = ids.index(image_token)
            checks = {"prompt": prompt == row["prompt"], "suffix": ids.count(image_token) == 1 and ids[position + 1:] == row["suffix_input_ids"],
                      "prefix": position == row["geometry"]["v_token_start"],
                      "official": TOK.collapse_image(ids, image_token) == TOK.official_token_ids(slow, row["prompt"])}
            out["compared"] += 1; out["by_turn"][turn] += 1
            out["prompt_rebuilt_equal"] += checks["prompt"]; out["suffix_ids_equal_recorded"] += checks["suffix"]
            out["prefix_length_equal_recorded"] += checks["prefix"]; out["equal_original_llava_ids"] += checks["official"]
            if not all(checks[k] for k in ("prompt", "suffix", "prefix")) and len(out["examples"]) < 10:
                out["examples"].append({"image_id": images[index - stride].name, "dialog_id": dialog_id, "turn_id": turn, "checks": checks})
    n = out["compared"]
    ok = (n == 3 * L15_PER_TURN and out["prompt_rebuilt_equal"] == n and out["suffix_ids_equal_recorded"] == n and out["prefix_length_equal_recorded"] == n
          and not changed)
    return dict(out, criterion="LLaVA-1.5 path, 60 prompts (20 images spread over the LLaVA-1.5 x MT-GQA main run, the first dialogue's T1, T2, T3 as "
                               "ReComp sent them): the prompt rebuilt by the current code and its token ids equal what that run recorded (ids after the image, "
                               "number of ids before it); and every source file that run froze is byte-identical today",
                reference=str(L15_RUN.relative_to(ROOT)), tokenizer_source=source, frozen_files_of_that_run=len(config["source_sha256"]),
                frozen_files_changed=changed, status="PASS" if ok else "FAIL",
                note="equal_original_llava_ids is reported, not judged (stage 0 of the ConvBench run found 0 mismatches for LLaVA-1.5)")


def stage_tokens(run, config):
    manifest = json.loads((run / "manifest.json").read_text())
    images = sorted({i for imgs in manifest["phase_images"].values() for i in imgs})      # main holds every image (a rehearsal's phases may not nest)
    previous, sources = build_previous(run, images)
    main_images = set(manifest["phase_images"]["main"])
    g1 = gate1_tokens([r for r in previous if r["i"] in main_images], manifest)
    g2 = gate2_llava15()
    parts = {"gate1_fixed_path_equals_original_llava": g1, "gate2_llava15_path_unchanged": g2}
    failed = [k for k, v in parts.items() if v["status"] != "PASS"]
    return receipt(run, "tokens", {"status": "PASS" if not failed else "FAIL", "failed": failed, "previous_rows": sources, **parts}, config)


# ------------------------------------------------------------------ gate 3
def stage_legacy(run, config):
    manifest, keys, execution = phase_state(run, "legacy", config)
    if execution["status"] != "PASS":
        return receipt(run, "legacy", {"status": "FAIL", "execution": execution, "failed": ["execution"]}, config)
    rows = committed_rows(run, "legacy")
    old = {(r["m"], r["d"], r["t"]): r for r in load_previous(run)}
    per_method = {}
    for m in H.PHASE_METHODS["legacy"]:
        s = {"reference": str(PREVIOUS[m].relative_to(ROOT)), "requests": 0, "missing": 0, "tokens_differ": 0, "prompt_equal": 0, "suffix_ids_equal": 0,
             "by_turn_tokens_differ": {1: 0, 2: 0, 3: 0}, "mismatches": []}
        for d, t in keys:
            new, ref = rows.get((m, d, t)), old.get((m, d, t))
            if new is None or ref is None:
                s["missing"] += 1
                continue
            s["requests"] += 1; s["prompt_equal"] += new["prompt"] == ref["prompt"]; s["suffix_ids_equal"] += new["suffix_input_ids"] == ref["sfx"]
            if tokens(new) != ref["tok"]:
                s["tokens_differ"] += 1; s["by_turn_tokens_differ"][t] += 1
                if len(s["mismatches"]) < 20:
                    s["mismatches"].append({"image_id": new["image_id"], "dialog_id": d, "turn_id": t, "prompt_equal": new["prompt"] == ref["prompt"],
                                            "new": new["prediction"], "previous": ref["pred"], "new_tokens": tokens(new), "previous_tokens": ref["tok"]})
        s["status"] = "PASS" if s["requests"] == len(keys) and not s["missing"] and not s["tokens_differ"] else "FAIL"
        per_method[m] = s
    g3 = dict(per_method[H.RECOMP], criterion="tokenization \"legacy\": ReComp's generated tokens identical to the 2026-11-20 run's record of the same "
                                              "request (same dialogue, same turn), every request of the 4 gate images")
    reported = {m: dict(s, status="REPORTED", was="PASS" if s["status"] == "PASS" else "DIFFERS") for m, s in per_method.items() if m != H.RECOMP}
    failed = [] if g3["status"] == "PASS" else ["gate3_legacy_path_reproduces_recomp"]
    return receipt(run, "legacy", {
        "status": "PASS" if not failed else "FAIL", "failed": failed, "images": manifest["phase_images"]["legacy"], "execution": execution,
        "gate3_legacy_path_reproduces_recomp": g3,
        "reported_other_methods_under_legacy": dict(reported, criterion="none (reported only): FullLoad and Ours against the 2026-11-20 run, IMPRESS against the 2026-11-17 run"),
        "reported_token_record": token_record(rows, keys, H.PHASE_METHODS["legacy"], TOK.TOKENIZATION_LEGACY)}, config)


# -------------------------------------------------- gates 4, 5, 6 (4 images)
def stage_gate(run, config):
    manifest, keys, execution = phase_state(run, "gate", config)
    if execution["status"] != "PASS":
        return receipt(run, "gate", {"status": "FAIL", "execution": execution, "failed": ["execution"]}, config)
    rows = committed_rows(run, "gate")
    g4, g5 = gates_4_5(rows, keys)
    g6 = turn1_identity(rows, keys, H.MAIN_METHODS, judged=True)
    parts = {"gate4_ours": g4, "gate5_impress": g5, "gate6_turn1_identical_across_methods": g6}
    failed = [k for k, v in parts.items() if v["status"] != "PASS"]
    return receipt(run, "gate", {
        "status": "PASS" if not failed else "FAIL", "failed": failed, "images": manifest["phase_images"]["gate"], "execution": execution, **parts,
        "reported_turn1_identity_all_arms": turn1_identity(rows, keys, H.PHASE_METHODS["gate"], judged=False),
        "reported_token_record": token_record(rows, keys, H.PHASE_METHODS["gate"], TOK.TOKENIZATION_OFFICIAL),
        "reference_sync_vs_async_followup_ttft": sync_async_timing(rows, keys, (("IMPRESS", H.NEW, H.NEW_ASYNC), ("Ours", H.ORIG, H.ORIG_ASYNC)))}, config)


# ------------------------------------------------------ gates 4, 5 (DEV 40)
def stage_devgate(run, config):
    manifest, keys, execution = phase_state(run, "devgate", config)
    if execution["status"] != "PASS":
        return receipt(run, "devgate", {"status": "FAIL", "execution": execution, "failed": ["execution"]}, config)
    rows = committed_rows(run, "devgate")
    g4, g5 = gates_4_5(rows, keys)
    parts = {"gate4_ours": g4, "gate5_impress": g5}
    failed = [k for k, v in parts.items() if v["status"] != "PASS"]
    return receipt(run, "devgate", {
        "status": "PASS" if not failed else "FAIL", "failed": failed, "image_count": len(manifest["phase_images"]["devgate"]), "execution": execution, **parts,
        "reported_turn1_identity_all_arms": turn1_identity(rows, keys, H.PHASE_METHODS["devgate"], judged=False),
        "reported_token_record": token_record(rows, keys, H.PHASE_METHODS["devgate"], TOK.TOKENIZATION_OFFICIAL),
        "reference_sync_vs_async_followup_ttft": sync_async_timing(rows, keys, (("IMPRESS", H.NEW, H.NEW_ASYNC), ("Ours", H.ORIG, H.ORIG_ASYNC)))}, config)


# ------------------------------------------------------ gates 6, 8 (DEV 40)
def stage_dev(run, config):
    manifest, keys, execution = phase_state(run, "dev", config)
    if execution["status"] != "PASS":
        return receipt(run, "dev", {"status": "FAIL", "execution": execution, "failed": ["execution"]}, config)
    rows = committed_rows(run, "dev")
    g6 = turn1_identity(rows, keys, H.MAIN_METHODS, judged=True)
    parts = {"gate6_turn1_identical_across_methods": g6}
    failed = [k for k, v in parts.items() if v["status"] != "PASS"]
    rates = H.measured_rates(run)
    return receipt(run, "dev", {
        "status": "PASS" if not failed else "FAIL", "failed": failed, "image_count": len(manifest["phase_images"]["dev"]),
        "execution": dict(execution, gate="gate8_dev_pilot_executed"), **parts,
        "reported_token_record": token_record(rows, keys, H.PHASE_METHODS["dev"], TOK.TOKENIZATION_OFFICIAL),
        "pilot_seconds_per_request": rates.get("dev")}, config)


# -------------------------------------------------------------------- admit
def stage_admit(run, config):
    steps = {}
    for name in REQUIRED:
        path = run / "gates" / f"{name}.json"
        doc = json.loads(path.read_text()) if path.exists() else {"status": "MISSING"}
        ok = doc.get("status") == "PASS" and doc.get("config_sha256") == config["_file_sha256"]
        steps[name] = {"status": doc.get("status"), "bound_to_this_freeze": doc.get("config_sha256") == config["_file_sha256"], "ok": ok,
                       "failed": doc.get("failed")}
    status = "PASS" if all(s["ok"] for s in steps.values()) else "FAIL"
    return receipt(run, "admission", {
        "status": status, "steps": steps, "failed": [k for k, s in steps.items() if not s["ok"]],
        "rule": "the main phase starts only if gates 1-7 are PASS (gates 4-6 on the 4 gate images and on the 40 DEV images) and the DEV pilot "
                "(gate 8) was executed; otherwise the driver records the failure and stops"}, config)


def stage_posthoc(run, config):
    """After the main phase; reported, not a gate."""
    manifest, keys, execution = phase_state(run, "main", config)
    rows = committed_rows(run, "main")
    t1 = turn1_identity(rows, keys, H.MAIN_METHODS, judged=False)
    record = token_record(rows, keys, H.MAIN_METHODS, TOK.TOKENIZATION_OFFICIAL)
    audits = Counter()
    for cp in sorted((run / "main" / "images").glob("*/COMMITTED.json")):
        doc = json.loads(cp.read_text())
        audits[json.loads((cp.parent / doc["attempt_id"] / "independent_audit.json").read_text())["status"]] += 1
    out = {"status": "REPORTED", "execution": execution, "turn1_identity_main": t1, "token_record_main": record, "image_audits": dict(audits),
           "config_sha256": config["_file_sha256"], "written_unix": time.time()}
    atomic_json(run / "gates" / "posthoc_main.json", out, replace=True)
    print(json.dumps({"requests": record["requests"], "ids_equal_original": record["requests_whose_ids_equal_original_llava"],
                      "turn1_mismatch_dialogues": t1["mismatch_count"], "audits": dict(audits)}))
    return dict(out, status="PASS")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--stage", choices=("tests", "tokens", "legacy", "gate", "devgate", "dev", "admit", "posthoc"), required=True)
    args = p.parse_args()
    run = args.run.resolve()
    assert run.parent == ROOT / "runs" and run.name.startswith(H.RUN_PREFIX)
    config = frozen(run)
    changed = H.freeze_violations(config["source_sha256"])
    if changed and "_dev_" not in run.name:
        raise RuntimeError("FROZEN_SOURCE_CONFIG_MISMATCH: " + repr(changed[:10]))
    doc = {"tests": stage_tests, "tokens": stage_tokens, "legacy": stage_legacy, "gate": stage_gate, "devgate": stage_devgate,
           "dev": stage_dev, "admit": stage_admit, "posthoc": stage_posthoc}[args.stage](run, config)
    sys.exit(0 if doc["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

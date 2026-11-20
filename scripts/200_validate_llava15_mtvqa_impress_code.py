#!/usr/bin/env python3
"""Validation steps of the LLaVA-1.5 x MT-VQA run whose IMPRESS arm is the
public-implementation path (contract docs/llava15_mtvqa_impress_code_contract.md, section 9).

Stages
  cpu       step 1 (public-equivalence tests against the public checkout,
            IMPRESS_OFFICIAL_REPO) and step 6 (the whole unit-test suite in the
            run environment and, as a cross-check, in the transformers 5.13.1
            environment; no failure other than the recorded environment ones).
  gate      steps 3, 4, 5 from the committed ``gate`` phase: IMPRESS at 100 %
            retention == the FullLoad loader on the same reordered store;
            asynchronous == synchronous (tokens, bytes) for FullLoad, IMPRESS,
            Ours; turn-1 tokens equal across arms.
  prefix    step 2 (GPU): the stored prefix-row importance of every gate image
            against a second full computation from pixels.
  pilot     step 7 from the committed ``pilot`` phase (+ step 5 on the pilot,
            + the storage / time projection of the main run).
  admit     step 8: writes gates/admission.json when 1-7 are PASS.
  posthoc   after the main phase (not a gate): turn-1 identity over 25,000 images.

Every receipt is bound to the SHA256 of the frozen config.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import llava15_support as S  # noqa: E402

E = S.load_script("_l15_mtvqa_impress_code_eval", "199_eval_llava15_mtvqa_impress_code.py")
A = S.load_script("_mtgqa_impress_code_validate", "197_validate_mt_gqa_impress_code.py")      # public_equivalence, run_suite (shared with experiment A)
sha, atomic_json, read_rows = E.sha, E.atomic_json, E.read_rows
OURS, OURS_ASYNC = E.OURS, E.OURS_ASYNC
GiB = 1 << 30
PY_CROSS = str(ROOT / ".venv-glm46v/bin/python")
SYNC_ASYNC = (("FullLoad", "fullload", "fullload_async"), ("IMPRESS", E.NEW, E.NEW_ASYNC), ("Ours", OURS, OURS_ASYNC))
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


# -------------------------------------------------------------- steps 1, 6
def stage_cpu(run, config):
    step1 = A.public_equivalence(run)
    suites = [A.G.run_suite(sys.executable), A.G.run_suite(PY_CROSS)]
    ok6 = all(r["status"] == "PASS" for r in suites)
    return receipt(run, "cpu", {
        "status": "PASS" if step1["status"] == "PASS" and ok6 else "FAIL",
        "step1_public_equivalence": step1,
        "step6_cpu_tests": {"suites": suites, "status": "PASS" if ok6 else "FAIL",
                            "rule": "no failing test other than the environment failures recorded before this work "
                                    "(transformers 4.57.6: the 3 GLM-4.6V tests; transformers 5.13.1: 2 tests)"}}, config)


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


def compare_sync_async(rows_by_image, sync, async_):
    """Step 4 for one family: tokens and read bytes (total and by kind)."""
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

    ``same_input`` says whether the two arms were given the same prompt (a
    later turn of a dialogue that already diverged has a different
    method-local history).  ``logits_of``: arm whose recorded per-step top
    logits are attached at the first differing position (evidence only).
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
    out["requests"] = out["requests"][:50]
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


# ---------------------------------------------------------------- steps 3-5
def stage_gate(run, config):
    import numpy as np
    manifest = json.loads((run / "manifest.json").read_text())
    complete = phase_complete(run, "gate", manifest, config)
    if complete["status"] != "PASS":
        return receipt(run, "gate", {"status": "FAIL", "phase": complete}, config)
    rows = phase_rows(run, "gate", manifest)
    n = 3 * len(rows)
    # step 3: the IMPRESS path at 100 % vs the FullLoad loader on the SAME reordered store, every request
    step3 = compare_tokens(rows, E.NEW_R100, E.FULL_REORDERED)
    # reference only: the FullLoad loader on the reordered store vs FullLoad on the raster store
    reference = compare_tokens(rows, E.FULL_REORDERED, "fullload", logits_of="fullload")
    step4 = {name: compare_sync_async(rows, a, b) for name, a, b in SYNC_ASYNC}
    step5 = t1_identity(rows, E.PHASE_METHODS["gate"])
    s3 = step3["compared"] == n == 132 and step3["different_tokens"] == 0 and step3["different_input"] == 0
    s4 = all(v["mismatched"] == 0 and v["compared"] == n for v in step4.values())
    s5 = step5["mismatched"] == 0 and step5["images"] == 44
    timing = {}      # reference only (not a criterion): synchronous vs asynchronous follow-up TTFT on the gate images
    for name, a, b in SYNC_ASYNC:
        d = np.array([rows[i][(b, t)]["ttft_ms"] - rows[i][(a, t)]["ttft_ms"] for i in rows for t in (2, 3)])
        timing[name] = {"sync_mean_ms": float(np.mean([rows[i][(a, t)]["ttft_ms"] for i in rows for t in (2, 3)])),
                        "async_mean_ms": float(np.mean([rows[i][(b, t)]["ttft_ms"] for i in rows for t in (2, 3)])),
                        "paired_mean_difference_ms": float(d.mean()), "requests": int(d.size)}
    return receipt(run, "gate", {
        "status": "PASS" if s3 and s4 and s5 else "FAIL", "phase": complete,
        "step3_impress_r100_vs_fullload_loader_on_reordered_store": dict(
            step3, status="PASS" if s3 else "FAIL",
            rule="generated tokens (and prompts) identical for every request"),
        "reference_reordered_fullload_vs_raster_fullload": dict(reference, note="not a criterion: key order alone (same K/V bytes, permuted rows)"),
        "step4_async_consistency": dict(step4, status="PASS" if s4 else "FAIL",
                                        rule="per stored-KV method: generated tokens and bytes read (total and by kind) identical, every request"),
        "step5_turn1_identity": dict(step5, status="PASS" if s5 else "FAIL", rule="turn-1 generated tokens identical across all arms, every image"),
        "reference_sync_vs_async_followup_ttft": timing}, config)


# ------------------------------------------------------------------ step 2
def stage_prefix(run, config):
    """Stored prefix-row importance vs a second full computation (GPU).

    For every gate image the committed evidence (the arrays written into the
    store; the image audit already tied the store's bytes to them) is compared
    with the statistics captured from a NEW pixel request with the same turn-1
    prompt.  Criterion: every element equal.  A request with another question
    is also run and reported (not a criterion).
    """
    import numpy as np
    import torch
    manifest = json.loads((run / "manifest.json").read_text())
    complete = phase_complete(run, "gate", manifest, config)
    if complete["status"] != "PASS":
        return receipt(run, "prefix", {"status": "FAIL", "phase": complete}, config)
    torch.set_num_threads(config["cpu_threads"]); torch.manual_seed(1234); np.random.seed(1234); torch.set_grad_enabled(False)
    E.assert_gpu_exclusive()
    runner, servers, _ = E.load_servers()
    by_image = {d["image_id"]: d for d in manifest["dialogues"] if d["image_id"] in set(manifest["phase_images"]["gate"])}
    images = []
    try:
        for image in manifest["phase_images"]["gate"]:
            folder = run / "gate" / "images" / image
            commit = json.loads((folder / "COMMITTED.json").read_text()); attempt = folder / commit["attempt_id"]
            p = json.loads((attempt / f"{E.NEW_KIND}_persistence.json").read_text())
            stats = np.load(attempt / p["official_turn1"]["stats_file"])
            rows = read_rows(attempt / "raw.jsonl")
            src = next(r for r in rows if r["physical_execution_id"] == p["source"]["physical_execution_id"])
            other = next(r for r in rows if r["method_id"] == src["method_id"] and r["turn_id"] == 2)
            path = ROOT / by_image[image]["image_path"]

            def capture(prompt):
                result, diag = E.normal_request(runner, servers["base"], path, lambda: (prompt, "", []), "none", with_stats=True)
                prefix, order = diag["turn1_stats"].result()
                return prefix.numpy(), order.numpy(), result["generated_token_ids"]

            prefix, order, toks = capture(src["prompt"])
            stored = stats["prefix_rows"]
            same = {"prefix_rows_equal": bool(np.array_equal(prefix, stored)), "order_scores_equal": bool(np.array_equal(order, stats["order_scores"])),
                    "prefix_rows_max_abs_diff": float(np.abs(prefix - stored).max()),
                    "order_scores_max_abs_diff": float(np.abs(order - stats["order_scores"]).max()),
                    "generated_tokens_equal": toks == tokens(src)}
            prefix2, _, _ = capture(other["prompt"])
            other_prompt = {"prefix_rows_equal": bool(np.array_equal(prefix2, stored)), "max_abs_diff": float(np.abs(prefix2 - stored).max()),
                            "mean_abs_diff": float(np.abs(prefix2 - stored).mean()),
                            "suffix_tokens_stored_request": len(src["suffix_input_ids"]), "suffix_tokens_other_request": len(other["suffix_input_ids"])}
            images.append({"image_id": image, "source_method_id": src["method_id"], "shape": list(stored.shape), "same_prompt": same,
                           "other_prompt_diagnostic": other_prompt,
                           "status": "PASS" if same["prefix_rows_equal"] and same["order_scores_equal"] else "FAIL"})
    finally:
        for server in servers.values():
            server.close()
    bad = [x["image_id"] for x in images if x["status"] != "PASS"]
    return receipt(run, "prefix", {
        "status": "PASS" if len(images) == 44 and not bad else "FAIL", "step": "2 stored prefix-row importance",
        "rule": "for every gate image, the stored prefix-row importance (and the order scores) equal, element for element, the values "
                "captured from a second full computation from pixels with the same turn-1 prompt",
        "images_checked": len(images), "failed_images": bad, "images": images}, config)


# ------------------------------------------------------------------ step 7
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
    storage_ok = free >= need
    status = "PASS" if t1["mismatched"] == 0 and t1["images"] == 40 and storage_ok else "FAIL"
    return receipt(run, "pilot", {
        "status": status, "phase": complete, "step5_turn1_identity": t1,
        "measured": {"seconds_per_image_mean": float(np.mean(seconds)), "seconds_per_image_p95": float(np.percentile(seconds, 95)),
                     "retained_bytes_per_image_mean": kept_per_image, "peak_scratch_bytes_max": int(max(scratch))},
        "projection_main": {"hours": float(np.mean(seconds)) * 25000 / 3600, "retained_bytes": kept_per_image * 25000,
                            "required_free_bytes": need, "free_bytes": free, "storage": "PASS" if storage_ok else "BLOCKED_STORAGE",
                            "formula": "30 GiB reserve + 4 GiB working set + pilot retained bytes/image x 25,000 x 1.35"}}, config)


# ------------------------------------------------------------------ step 8
def stage_admit(run, config):
    steps = {}
    for name in ("cpu", "prefix", "gate", "pilot"):
        path = run / "gates" / f"{name}.json"
        doc = json.loads(path.read_text()) if path.exists() else {"status": "MISSING"}
        ok = doc.get("status") == "PASS" and doc.get("config_sha256") == config["_file_sha256"]
        steps[name] = {"status": doc.get("status"), "bound_to_this_freeze": doc.get("config_sha256") == config["_file_sha256"], "ok": ok}
    status = "PASS" if all(s["ok"] for s in steps.values()) else "FAIL"
    return receipt(run, "admission", {"status": status, "steps": steps,
                                      "rule": "steps 1-7 all PASS for this freeze; the main stage does not start otherwise"}, config)


def stage_posthoc(run, config):
    """Turn-1 identity over the committed main images (reported, not a gate)."""
    total = same = 0; bad = []
    for cp in sorted((run / "main" / "images").glob("*/COMMITTED.json")):
        doc = json.loads(cp.read_text()); total += 1
        if doc.get("t1_tokens_identical_across_arms"):
            same += 1
        else:
            bad.append(cp.parent.name)
    out = {"status": "REPORTED", "images": total, "turn1_identical_across_arms": same, "mismatched_images": bad[:200]}
    atomic_json(run / "gates" / f"posthoc_turn1_identity_main_{time.strftime('%Y%m%dT%H%M%S')}.json", dict(out, config_sha256=config["_file_sha256"]))
    print(json.dumps({k: out[k] for k in ("images", "turn1_identical_across_arms")}))
    return dict(out, status="PASS")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--stage", choices=("cpu", "gate", "prefix", "pilot", "admit", "posthoc"), required=True)
    args = p.parse_args()
    run = args.run.resolve()
    assert run.parent == ROOT / "runs" and run.name.startswith(E.RUN_PREFIX)
    config = frozen(run)
    if E.source_hashes() != config["source_sha256"] and "_dev_" not in run.name:
        raise RuntimeError("FROZEN_SOURCE_CONFIG_MISMATCH")
    doc = {"cpu": stage_cpu, "gate": stage_gate, "prefix": stage_prefix, "pilot": stage_pilot, "admit": stage_admit,
           "posthoc": stage_posthoc}[args.stage](run, config)
    sys.exit(0 if doc["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

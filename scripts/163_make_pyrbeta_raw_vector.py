#!/usr/bin/env python3
"""Phase 0 of the PyramidKV beta=20 MT-GQA HOLD add-on run
(contract docs/layer_budget_kv25_pyrbeta_contract.md).

Stages
  --stage vectors : create the run root, copy the frozen v2 U and PYR-beta vector
                    documents (SHA-checked), build PYR-beta-raw from the v2 PYR-beta
                    document's ``source.raw_unclipped`` list (values are NOT recomputed),
                    independent re-derivation of the PyramidKV schedule, k_l tables on the
                    398 HOLD image N list (BLOCKED if any k_l == 0)
  --stage freeze  : contract SHA, storage plan (FRESH_IMAGE_STREAMING, arm count),
                    protection "before" inventory, hold_arms.json, config.json

Nothing here fits or selects a vector from data.  ``mmimpress.layer_budget.R_MIN`` /
``R_MAX`` are not touched: the raw vector is expressed only through its document's
``r_min = 0.0`` / ``r_max = 0.60``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mmimpress.layer_budget import (budget_document, layer_keep_counts, load_budget_document,
                                    total_row_budget)
from mmimpress.prefix25 import sha

V2_RUN = ROOT / "runs/layer_budget_kv25_v2_20261106T073502Z"
V2_RES = ROOT / "results/layer_budget_kv25_v2_20261106T073502Z"
CONTRACT = ROOT / "docs/layer_budget_kv25_pyrbeta_contract.md"
RATIO_SHA = {"U": "54ea273e1a38b1e8b02b96237381f982344cd28ca8466162876e161cdb8ff3ba",
             "PYR-beta": "e374616b5a028402f8c479159f458f83690c96c24728307e00113f206a4cd483"}
PK = "ours_prefixstore_kv25_"
REQUIRED_ARMS = [f"{PK}U_packed", f"{PK}PYR-beta_packed", f"{PK}PYR-beta-raw_packed"]
NEW_PATHS = ["docs/layer_budget_kv25_pyrbeta_contract.md", "scripts/163_make_pyrbeta_raw_vector.py",
             "scripts/164_eval_layer_budget_pyrbeta_hold.py", "scripts/165_report_layer_budget_pyrbeta.py",
             "scripts/166_validate_layer_budget_pyrbeta.py", "papers/layer_budget_kv25_pyrbeta_hold_prompt.md"]
BETA = 20
L = 32
GiB = 2 ** 30


def write(path, obj, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as f:
        json.dump(obj, f, indent=1, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def independent_pyramid(beta=BETA, layers=L):
    """PyramidKV schedule re-derived in exact rationals, independent of scripts/158:
    min = 0.25/beta, max = 0.5 - min, ``layers`` equally spaced values from max to min."""
    lo = Fraction(1, 4) / beta
    hi = Fraction(1, 2) - lo
    step = (hi - lo) / (layers - 1)
    exact = [hi - li * step for li in range(layers)]
    assert sum(exact) == Fraction(layers, 4)          # exact: 8 for 32 layers
    return exact


def stage_vectors(run):
    run.mkdir(parents=True, exist_ok=False)
    vec = run / "vectors"
    vec.mkdir()
    (run / "phase0").mkdir()
    (run / "gates").mkdir()
    record = {"v2_run": str(V2_RUN.relative_to(ROOT)), "copied": {}}
    for a in ("U", "PYR-beta"):
        src = V2_RUN / "phase2" / f"layer_budget_llava_{a}.json"
        res = V2_RES / f"layer_budget_llava_{a}.json"
        dst = vec / src.name
        shutil.copyfile(src, dst)
        doc = load_budget_document(dst)
        if not (sha(dst) == sha(src) == sha(res)):
            raise RuntimeError(f"BLOCKED: {a} vector file differs from the v2 run/results copy")
        if doc["layer_budget_sha256"] != RATIO_SHA[a]:
            raise RuntimeError(f"BLOCKED: {a} ratio SHA {doc['layer_budget_sha256']} != v2 REPORT value {RATIO_SHA[a]}")
        record["copied"][a] = {"source": str(src.relative_to(ROOT)), "file_sha256": sha(dst), "layer_budget_sha256": doc["layer_budget_sha256"],
                               "r_min": doc["r_min"], "r_max": doc["r_max"], "matches_v2_report": True}
    v2doc_path = vec / "layer_budget_llava_PYR-beta.json"
    v2doc = json.loads(v2doc_path.read_text())
    raw = v2doc["source"]["raw_unclipped"]                         # the only input; used verbatim
    assert isinstance(raw, list) and len(raw) == L and v2doc["source"]["beta"] == BETA
    exact = independent_pyramid()
    indep = [float(x) for x in exact]
    max_diff = max(abs(a - b) for a, b in zip(raw, indep))
    sum_diff = abs(sum(raw) - 8.0)
    check = {"input": "v2 PYR-beta json source.raw_unclipped (verbatim, not recomputed)",
             "formula": "min = 0.25/beta, max = 0.5 - min, 32 equally spaced values max -> min, beta = 20 (exact rationals -> float)",
             "beta": BETA, "min": float(exact[-1]), "max": float(exact[0]), "exact_sum": str(sum(exact)),
             "max_abs_layer_diff": max_diff, "max_abs_layer_diff_tolerance": 1e-12, "float_sum": sum(raw), "abs_sum_minus_8": sum_diff,
             "sum_tolerance": 1e-9, "strictly_decreasing": all(a > b for a, b in zip(raw, raw[1:])),
             "status": "PASS" if max_diff < 1e-12 and sum_diff < 1e-9 else "FAIL"}
    if check["status"] != "PASS":
        write(vec / "pyrbeta_raw_verification.json", check)
        raise RuntimeError("BLOCKED: raw_unclipped does not match the PyramidKV formula")
    doc = budget_document("PYR-beta-raw", raw, model=v2doc["model"], r_min=0.0, r_max=0.60, source={
        "kind": "PyramidKV beta schedule (window 0, beta=20), unclipped", "beta": BETA, "code": v2doc["source"]["code"],
        "derived_from": {"path": str((V2_RUN / "phase2" / "layer_budget_llava_PYR-beta.json").relative_to(ROOT)),
                         "file_sha256": sha(v2doc_path), "layer_budget_sha256": v2doc["layer_budget_sha256"], "field": "source.raw_unclipped"}})
    assert doc["layer_keep_ratios"] == [float(x) for x in raw] and doc["r_min"] == 0.0 and doc["r_max"] == 0.60
    raw_path = vec / "layer_budget_llava_PYR-beta-raw.json"
    write(raw_path, doc)
    reloaded = load_budget_document(raw_path)
    assert reloaded["layer_budget_sha256"] == doc["layer_budget_sha256"] and reloaded["layer_keep_ratios"] == doc["layer_keep_ratios"]
    check.update(layer_budget_sha256=doc["layer_budget_sha256"], file_sha256=sha(raw_path), r_min=0.0, r_max=0.60,
                 max_ratio=max(raw), min_ratio=min(raw), r_max_binding=bool(max(raw) > 0.60))
    # ---- k_l on the 398 HOLD images
    dist = json.loads((V2_RUN / "phase0" / "n_distribution.json").read_text())["hold"]
    rows = dist["per_image"]
    assert len(rows) == 398 and len({r["image_id"] for r in rows}) == 398
    write(run / "phase0" / "n_distribution_hold.json", {"source": str((V2_RUN / "phase0" / "n_distribution.json").relative_to(ROOT)),
                                                       "source_sha256": sha(V2_RUN / "phase0" / "n_distribution.json"), "hold": dist})
    docs = {"U": load_budget_document(vec / "layer_budget_llava_U.json"), "PYR-beta": load_budget_document(v2doc_path), "PYR-beta-raw": reloaded}
    per_layer_min = {a: [None] * L for a in docs}
    zero_layers = {a: 0 for a in docs}
    for r in rows:
        n = int(r["N"])
        for a, d in docs.items():
            k = layer_keep_counts(d["layer_keep_ratios"], n, L, 0.25, d["r_min"], d["r_max"])
            assert sum(k) == total_row_budget(n) == 32 * ((n + 3) // 4) == r["B"]
            zero_layers[a] += sum(1 for x in k if x == 0)
            per_layer_min[a] = [x if m is None else min(m, x) for m, x in zip(per_layer_min[a], k)]
    n_med = int(statistics.median_low(sorted(int(r["N"]) for r in rows)))
    table = []
    k_med = {a: layer_keep_counts(d["layer_keep_ratios"], n_med, L, 0.25, d["r_min"], d["r_max"]) for a, d in docs.items()}
    row_bytes = 32 * 128 * 2
    for li in range(L):
        table.append({"layer": li, "r_raw": raw[li], "r_floor010": docs["PYR-beta"]["layer_keep_ratios"][li],
                      "r_diff_floor_minus_raw": docs["PYR-beta"]["layer_keep_ratios"][li] - raw[li], "r_U": 0.25,
                      f"k_raw_N{n_med}": k_med["PYR-beta-raw"][li], f"k_floor010_N{n_med}": k_med["PYR-beta"][li], f"k_U_N{n_med}": k_med["U"][li],
                      f"bytes_raw_N{n_med}": 2 * k_med["PYR-beta-raw"][li] * row_bytes, f"bytes_floor010_N{n_med}": 2 * k_med["PYR-beta"][li] * row_bytes,
                      "min_k_raw_over_hold": per_layer_min["PYR-beta-raw"][li], "min_k_floor010_over_hold": per_layer_min["PYR-beta"][li]})
    with (vec / "k_tables.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    ktab = {"hold_images": len(rows), "N_min": dist["N_min"], "N_median": dist["N_median"], "N_max": dist["N_max"], "median_N_used": n_med,
            "per_layer_min_k_over_hold": per_layer_min, "zero_k_layers_over_hold": zero_layers,
            "min_k_overall": {a: min(v) for a, v in per_layer_min.items()}, "k_at_median_N": k_med,
            "sum_k_equals_32ceilN4_all_images": True, "row_bytes": row_bytes, "table": table,
            "status": "PASS" if zero_layers["PYR-beta-raw"] == 0 else "BLOCKED"}
    write(vec / "k_tables.json", ktab)
    write(vec / "pyrbeta_raw_verification.json", check)
    record.update(raw={"path": str(raw_path.relative_to(ROOT)), "file_sha256": sha(raw_path), "layer_budget_sha256": doc["layer_budget_sha256"]},
                  verification_sha256=sha(vec / "pyrbeta_raw_verification.json"), k_tables_sha256=sha(vec / "k_tables.json"), utc=time.time())
    write(run / "phase0" / "vectors_record.json", record)
    if ktab["status"] != "PASS":
        raise RuntimeError("BLOCKED: PYR-beta-raw has a layer with k_l == 0 on a HOLD image")
    print(json.dumps({"vectors": "PASS", "run": str(run), "raw_sha256": doc["layer_budget_sha256"], "max_abs_layer_diff": max_diff, "abs_sum_minus_8": sum_diff,
                      "min_k_raw": ktab["min_k_overall"]["PYR-beta-raw"], "zero_layers_raw": zero_layers["PYR-beta-raw"], "median_N": n_med,
                      "k_last_layer_at_median": {a: k_med[a][-1] for a in k_med}}, indent=1), flush=True)


def stage_freeze(run, arms):
    from mmimpress.recompute_kv25 import sha_manifest, stat_inventory
    assert CONTRACT.exists(), "write the contract before freezing"
    assert arms[:3] == REQUIRED_ARMS and set(arms[3:]) <= {"recompute"}, arms
    vec = run / "vectors"
    docs = {a: load_budget_document(vec / f"layer_budget_llava_{a}.json") for a in ("U", "PYR-beta", "PYR-beta-raw")}
    rec = json.loads((run / "phase0" / "vectors_record.json").read_text())
    assert docs["U"]["layer_budget_sha256"] == RATIO_SHA["U"] and docs["PYR-beta"]["layer_budget_sha256"] == RATIO_SHA["PYR-beta"]
    assert docs["PYR-beta-raw"]["layer_budget_sha256"] == rec["raw"]["layer_budget_sha256"] and sha(vec / "layer_budget_llava_PYR-beta-raw.json") == rec["raw"]["file_sha256"]
    dist = json.loads((run / "phase0" / "n_distribution_hold.json").read_text())["hold"]
    # ---- storage plan (v2 162 FRESH_IMAGE_STREAMING, arm count of this run)
    free = shutil.disk_usage(ROOT).free
    row_bytes = 32 * 128 * 2
    n_max = int(dist["N_max"]); s_max = max(int(r["S"]) for r in dist["per_image"])
    packed_max = int(32 * ((n_max + 3) // 4) * 2 * row_bytes + 2 * 32 * s_max * row_bytes + 8 * 2 ** 20)
    stored = [a for a in arms if a.startswith(PK)]
    hold = {"arms": len(arms), "stored_arms": len(stored), "live_stores_per_image": f"{len(stored)} packed (<= {packed_max / 1e9:.2f} GB each); no raster store",
            "largest_image_bundle_bound_bytes": len(stored) * packed_max, "builder_intermediate_margin_bytes": 4 * GiB,
            "peak_working_set_bytes": len(stored) * packed_max + 4 * GiB, "raw_and_evidence_growth_bound_bytes": 4 * GiB,
            "raw_growth_basis": "v2 7-arm HOLD main directory = 2.1 GB -> 3-4 arms ~0.9-1.2 GB", "minimum_reserve_bytes": 30 * GiB,
            "policy": "FRESH_IMAGE_STREAMING: one live image; stores built -> measured -> audited -> committed -> run-local scratch deleted by exact SHA allowlist"}
    hold["required_free_bytes"] = hold["minimum_reserve_bytes"] + hold["peak_working_set_bytes"] + hold["raw_and_evidence_growth_bound_bytes"]
    gates = {"per_case_bound_bytes": 4 * packed_max + 2 * GiB, "policy": "run-local gate stores deleted after each case (v2 amendment rule); frozen 106 capacity reserve needs >= 46 GiB free"}
    plan = {"measured_at_utc": time.time(), "free_bytes_now": free, "free_gib_now": free / GiB, "packed_store_max_bytes_estimate": packed_max,
            "hold": hold, "gates": gates, "results_and_protection_bound_bytes": 2 * GiB,
            "required_initial_free_bytes": max(hold["required_free_bytes"], 46 * GiB + gates["per_case_bound_bytes"]) + 2 * GiB}
    plan["required_initial_free_gib"] = plan["required_initial_free_bytes"] / GiB
    plan["margin_bytes"] = free - plan["required_initial_free_bytes"]
    plan["status"] = "PASS" if plan["margin_bytes"] >= 0 else "BLOCKED_STORAGE"
    plan["deletion_scope"] = [f"runs/{run.name}/hold/scratch/** (exact SHA allowlist after audit+commit)", f"runs/{run.name}/llava/gpu_gates/** (guarded_rmtree per case)",
                              f"runs/{run.name}/gates/grmin_fixture/** (CPU fixture)"]
    write(run / "phase0" / "storage_plan.json", plan)
    if plan["status"] != "PASS":
        raise RuntimeError(f"BLOCKED_STORAGE: free {free} < required {plan['required_initial_free_bytes']}")
    # ---- protection before: full SHA of results/docs/mmimpress/scripts/tests/README, stat inventory of kvstore* and of every other run
    t0 = time.time()
    excl = [f"results/{run.name}"] + NEW_PATHS
    shas = {k: v for k, v in sha_manifest([ROOT / x for x in ["results", "docs", "mmimpress", "scripts", "tests", "README.md"]], ROOT, exclude=excl).items() if "__pycache__" not in k}
    kv = stat_inventory(sorted(ROOT.glob("kvstore*")), ROOT)
    runs = stat_inventory(sorted(p for p in (ROOT / "runs").iterdir() if p.name != run.name), ROOT)
    write(run / "protection_before.json", {"stage": "before", "utc": time.time(), "elapsed_s": time.time() - t0, "excluded_new_paths": excl,
                                           "sha256_files": len(shas), "kvstore_inventory_files": len(kv), "runs_inventory_files": len(runs),
                                           "sha256": shas, "kvstore_inventory": kv, "runs_inventory": runs})
    v2_hold = json.loads((V2_RUN / "hold" / "config.json").read_text())["source_sha256"]
    changed = sorted(k for k, d in v2_hold.items() if (ROOT / k).exists() and sha(ROOT / k) != d)
    missing = sorted(k for k in v2_hold if not (ROOT / k).exists())
    known = {"scripts/160_report_layer_budget_v2.py": "v2 report script amended after the v2 HOLD freeze (v2 freeze_amendment_report_script*.json); not imported by the serving path"}
    write(run / "source_modifications.json", {"reference": "v2 hold/config.json source_sha256", "changed_existing_files": changed, "missing": missing,
                                              "known_v2_post_hold_edits": {k: v for k, v in known.items() if k in changed}, "new_files_planned": NEW_PATHS,
                                              "note": "this run adds scripts 163-166 and a contract only; store/serving modules must equal the v2 HOLD freeze"})
    unexplained = [k for k in changed if k not in known]
    if unexplained or missing:
        raise RuntimeError(f"BLOCKED: existing sources differ from the v2 HOLD freeze: {unexplained} missing {missing}")
    write(run / "hold_arms.json", {"hold_arms": arms, "control": f"{PK}U_packed", "primary": [f"{PK}PYR-beta_packed", f"{PK}PYR-beta-raw_packed"],
                                   "vectors_sha256": {a: d["layer_budget_sha256"] for a, d in docs.items()},
                                   "vectors_file_sha256": {a: sha(vec / f"layer_budget_llava_{a}.json") for a in docs},
                                   "clip_range": {a: [d["r_min"], d["r_max"]] for a, d in docs.items()},
                                   "rule": "no data-driven vector construction or selection; arms fixed by the prompt before measurement",
                                   "optional_recompute": "included" if "recompute" in arms else "excluded (default; not authorised by the user before execution)"})
    env = {"python": sys.version, "torch": __import__("torch").__version__, "transformers": __import__("transformers").__version__,
           "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
           "read_ahead_kb": Path("/sys/block/nvme0n1/queue/read_ahead_kb").read_text().strip()}
    config = {"run": str(run.relative_to(ROOT)), "contract": str(CONTRACT.relative_to(ROOT)), "contract_sha256": sha(CONTRACT),
              "parent_v2_run": str(V2_RUN.relative_to(ROOT)), "parent_v2_contract_sha256": sha(ROOT / "docs/layer_budget_kv25_v2_contract.md"),
              "parent_v2_decision_sha256": sha(V2_RES / "decision.json"),
              "model": "llava-hf/llava-v1.6-vicuna-7b-hf", "revision": "c916e6cdcd760b4cecd1dd4907f84ac649f93b23", "quantization": "NF4 double, BF16 compute",
              "attention": "eager", "store_dtype": "float16", "chunk_size": 64, "seed": 1234, "greedy": True, "max_new_tokens": 16,
              "total_ratio": 0.25, "num_layers": 32, "budget_policy": "layerwise_v1", "global_R_MIN_R_MAX_unchanged": [0.10, 0.60],
              "vectors": {a: {"layer_budget_sha256": d["layer_budget_sha256"], "r_min": d["r_min"], "r_max": d["r_max"]} for a, d in docs.items()},
              "hold": {"workload": "data/mt_gqa/dialogues.json", "images": 398, "dialogues": 4061, "requests_per_arm": 12183, "arms": arms},
              "decision_rule": {"primary_each_vector": "vec_packed - U_packed, HOLD followup; GO: delta>=+1.0pp & CI excludes 0 & SSD MB/followup <= U*1.02 & TTFT mean <= U*1.05; "
                                                       "PARTIAL: delta>=+0.5pp with CI incl. 0, or GO with exactly one of MB/TTFT violated; else NO-GO",
                                "floor_effect": "D = PYR-beta-raw - PYR-beta (followup, paired bootstrap 95% CI): CI includes 0 -> floor unnecessary (paper default PYR-beta-raw); "
                                                "CI upper < 0 -> floor needed (use PYR-beta); CI lower > 0 -> floor harmful (use PYR-beta-raw)",
                                "secondary": "cross-run paired accuracy vs v2 arms only if G-det == 100%; TTFT never cross-run"},
              "bootstrap": {"unit": "image", "resamples": 10000, "seed": 1234}, "environment": env, "disk_free_bytes_at_config": free,
              "storage_plan_sha256": sha(run / "phase0" / "storage_plan.json"), "hold_arms_sha256": sha(run / "hold_arms.json")}
    write(run / "config.json", config)
    shutil.copy2(CONTRACT, run / "CONTRACT.md")
    print(json.dumps({"freeze": "PASS", "run": str(run), "contract_sha256": config["contract_sha256"], "storage": plan["status"], "free_gib": round(free / GiB, 2),
                      "required_gib": round(plan["required_initial_free_gib"], 2), "protection_files": len(shas), "runs_inventory_files": len(runs), "arms": arms}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stage", choices=("vectors", "freeze"), required=True)
    ap.add_argument("--with-recompute", action="store_true", help="optional recompute arm (only when the user allowed it before execution)")
    args = ap.parse_args()
    run = Path(args.run).resolve()
    assert run.parent == ROOT / "runs" and run.name.startswith("layer_budget_kv25_pyrbeta_")
    if args.stage == "vectors":
        stage_vectors(run)
    else:
        stage_freeze(run, REQUIRED_ARMS + (["recompute"] if args.with_recompute else []))


if __name__ == "__main__":
    main()

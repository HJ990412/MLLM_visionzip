#!/usr/bin/env python3
"""Phase 0 of the Qwen2.5-VL layer-wise budget (PyramidKV beta) + PackedPrefixStore experiment.

  --stage vectors : create the run root, the three frozen ratio vectors (U, PYR-b20, PYR-b2p5),
                    independent verification (rational formula + PyramidKV public code),
                    k_l tables on the 398 HOLD images, installed-transformers source record
  --stage freeze  : (after the contract and design document exist) storage plan, protection
                    "before" inventory, manifest.json, config.json, frozen sources, CONTRACT copy

No data is used to build or select a vector.  Nothing outside the new run/results roots is written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from fractions import Fraction
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import qwen_pyr_support as Q
from qwen_pyr_support import ROOT, read, sha, write

PYRAMIDKV_REPO = "Zefan-Cai/PyramidKV"
PYRAMIDKV_FILE = "pyramidkv/pyramidkv_utils.py"
PYRAMIDKV_COMMITS = ("c97f9d706d63d1b56e980f8b81b2a924c21162b3", "68cd9551")
PRIOR_CITATION = "Zefan-Cai/PyramidKV 68cd9551 pyramidkv/pyramidkv_utils.py L360-370 (scripts/158_allocate_prior_work_vectors.py)"


def independent_pyramid(beta, layers=Q.NUM_LAYERS):
    """Exact-rational PyramidKV schedule expressed as ratios of N (budget = 0.25 N)."""
    b = Fraction(str(beta))
    lo = Fraction(1, 4) / b
    hi = Fraction(1, 2) - lo
    step = (hi - lo) / (layers - 1)
    return [hi - i * step for i in range(layers)]


def pyramidkv_public_code():
    """Fetch the public file at pinned commits and locate the schedule lines; fail soft."""
    record = {"repo": PYRAMIDKV_REPO, "file": PYRAMIDKV_FILE, "prior_citation": PRIOR_CITATION, "fetches": []}
    want = [r"min_num = \(self\.max_capacity_prompt - self\.window_size\) // self\.beta",
            r"max_num = \(self\.max_capacity_prompt - self\.window_size\) \* 2 - min_num",
            r"steps = \(max_num - min_num\) // \(self\.num_hidden_layers - 1\)",
            r"max_capacity_prompt = max_num - self\.layer_idx \* steps"]
    ok = False
    for commit in PYRAMIDKV_COMMITS:
        url = f"https://raw.githubusercontent.com/{PYRAMIDKV_REPO}/{commit}/{PYRAMIDKV_FILE}"
        entry = {"commit_ref": commit, "url": url}
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                blob = resp.read()
            lines = blob.decode("utf-8").splitlines()
            entry.update(file_sha256=hashlib.sha256(blob).hexdigest(), lines=len(lines))
            found = {}
            for pat in want:
                hits = [i + 1 for i, line in enumerate(lines) if re.search(pat, line)]
                found[pat] = hits
            entry["line_numbers"] = found
            first = [hits[0] for hits in found.values() if hits]
            entry["schedule_block_lines"] = [min(first), max(first)] if len(first) == len(want) else None
            entry["schedule_block_text"] = lines[min(first) - 1:max(first)] if len(first) == len(want) else None
            m = re.search(r"class PyramidKVCluster\(\):\s*\n\s*def __init__\(self, ([^)]*)\)", blob.decode("utf-8"))
            entry["cluster_init_defaults"] = m.group(1) if m else None
            entry["beta_default_20"] = bool(m and re.search(r"beta\s*=\s*20\b", m.group(1)))
            entry["status"] = "FETCHED" if len(first) == len(want) else "FETCHED_PATTERN_MISSING"
            ok = ok or entry["status"] == "FETCHED"
        except Exception as exc:  # network unavailable etc.
            entry.update(status="UNAVAILABLE", error=repr(exc))
        record["fetches"].append(entry)
    record["status"] = "REVERIFIED" if ok else "NOT_REVERIFIED"
    record["shape"] = ("min = budget // beta; max = 2*budget - min; steps = (max - min) // (num_hidden_layers - 1); "
                       "layer capacity = max - layer_idx*steps (linear, equal spacing, decreasing with depth); "
                       "window_size = 0 here; ratios drop the integer floor: lo = 0.25/beta, hi = 0.5 - lo, step = (hi - lo)/(L - 1)")
    record["not_adopted"] = ("the q_len clamp branch (max_num >= q_len - window) is not reachable at a 25% budget; PyramidKV's attention-based token "
                             "selection is NOT used -- only the per-layer cut point of the image-only saliency order follows the schedule")
    return record


def stage_vectors(run: Path):
    from mmimpress.layer_budget import budget_document, budget_sha256
    run.mkdir(parents=True, exist_ok=False)
    (run / "vectors").mkdir()
    docs, files, checks = {}, {}, {}
    for alloc in Q.ALLOCS:
        r = Q.vector(alloc)
        if alloc == "U":
            source = {"kind": "uniform 0.25 for every decoder layer (control)"}
        else:
            beta = Q.BETA[alloc]
            exact = independent_pyramid(beta)
            source = {"kind": f"PyramidKV public-code schedule as ratios, beta = {beta:g}, window 0, no floor, no clip",
                      "formula": "lo = 0.25/beta; hi = 0.5 - lo; step = (hi - lo)/27; r_i = hi - i*step", "beta": beta,
                      "same_formula_as": "scripts/158_allocate_prior_work_vectors.py::pyramid_beta (L = 28 instead of 32)",
                      "code": PRIOR_CITATION}
            checks[alloc] = {"max_abs_layer_diff_vs_exact_rational": max(abs(Fraction(a) - b) for a, b in zip(r, exact)).__float__(),
                             "abs_sum_minus_7": abs(sum(r) - 7.0), "exact_first": str(exact[0]), "exact_last": str(exact[-1]),
                             "exact_sum": str(sum(exact)), "strictly_decreasing": all(a > b for a, b in zip(r, r[1:])),
                             "equal_spacing_max_dev": max(abs((r[i] - r[i + 1]) - (r[0] - r[1])) for i in range(27))}
            assert checks[alloc]["max_abs_layer_diff_vs_exact_rational"] < 1e-12 and checks[alloc]["abs_sum_minus_7"] < 1e-9
            assert sum(exact) == 7 and checks[alloc]["strictly_decreasing"]
        doc = budget_document(alloc, r, model=Q.MODEL_ID, source=source, **Q.BUDGET_KW)
        if doc["layer_budget_sha256"] != Q.RATIO_SHA[alloc] or budget_sha256(r) != Q.RATIO_SHA[alloc]:
            raise RuntimeError(f"BLOCKED: ratio SHA256 mismatch for {alloc}: {doc['layer_budget_sha256']}")
        with Q.vector_path(run, alloc).open("x") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")
        docs[alloc] = doc
        files[alloc] = sha(Q.vector_path(run, alloc))
    docs = Q.load_vectors(run)
    write(run / "vectors" / "vectors_sha256.json", {a: {"layer_budget_sha256": docs[a]["layer_budget_sha256"], "file_sha256": files[a],
                                                       "file": str(Q.vector_path(run, a).relative_to(ROOT)), "first": docs[a]["layer_keep_ratios"][0],
                                                       "last": docs[a]["layer_keep_ratios"][-1], "sum": sum(docs[a]["layer_keep_ratios"]),
                                                       "num_layers": docs[a]["num_layers"], "r_min": docs[a]["r_min"], "r_max": docs[a]["r_max"]} for a in Q.ALLOCS})
    public = pyramidkv_public_code()
    write(run / "phase0" / "pyramidkv_verification.json", {"formula_checks": checks, "public_code": public})

    # ---- k_l tables on the 398 HOLD images (N from the frozen parent manifest; re-checked against actual geometry by every audit)
    manifest = read(Q.PARENT_RUN / "manifest.json")
    identity = manifest["image_identity"]
    ns = sorted(int(x["N_content"]) for x in identity.values())
    assert len(ns) == 398
    n_low_median = statistics.median_low(ns)
    per_n = {}
    for n in sorted(set(ns)):
        per_n[n] = {a: Q.keep_counts(docs[a], n) for a in Q.ALLOCS}
        for a in Q.ALLOCS:
            assert sum(per_n[n][a]) == 28 * math.ceil(n / 4), (a, n)
        assert per_n[n]["U"] == [(n + 3) // 4] * 28
    zero = {a: sorted(n for n in per_n if min(per_n[n][a]) == 0) for a in Q.ALLOCS}
    expected = {260: {"PYR-b20": [127, 3], "PYR-b2p5": [104, 26]}, 345: {"PYR-b20": [169, 5], "PYR-b2p5": [139, 35]},
                529: {"PYR-b20": [259, 7], "PYR-b2p5": [212, 54]}}
    observed = {n: {a: [per_n[n][a][0], per_n[n][a][-1]] for a in ("PYR-b20", "PYR-b2p5")} for n in expected}
    table = {"images": 398, "distinct_N": len(per_n), "N_min": ns[0], "N_max": ns[-1], "N_median_arithmetic": statistics.median(ns),
             "N_median_low_actual_image": n_low_median, "N_counts": {str(n): ns.count(n) for n in sorted(set(ns))},
             "k_per_layer_by_N": {str(n): per_n[n] for n in per_n}, "zero_row_layers_N": zero,
             "min_k_overall": {a: min(min(per_n[n][a]) for n in per_n) for a in Q.ALLOCS},
             "expected_first_last": {str(k): v for k, v in expected.items()}, "observed_first_last": {str(k): v for k, v in observed.items()},
             "expected_match": observed == expected,
             "sum_rule": "sum_l k_l == 28*ceil(N/4) for every HOLD N and vector; U gives k_l == (N+3)//4",
             "explicit_arguments": Q.BUDGET_KW,
             "status": "PASS" if observed == expected and not any(zero.values()) else "BLOCKED"}
    write(run / "phase0" / "k_tables.json", table)
    with (run / "phase0" / "k_tables.csv").open("x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "r_U", "r_PYR-b20", "r_PYR-b2p5"] + [f"k_{a}_N{n}" for n in (ns[0], n_low_median, ns[-1]) for a in Q.ALLOCS]
                   + [f"bytes_{a}_N{n_low_median}" for a in Q.ALLOCS])
        for li in range(28):
            w.writerow([li] + [f"{docs[a]['layer_keep_ratios'][li]:.6f}" for a in Q.ALLOCS]
                       + [per_n[n][a][li] for n in (ns[0], n_low_median, ns[-1]) for a in Q.ALLOCS]
                       + [2 * per_n[n_low_median][a][li] * 1024 for a in Q.ALLOCS])
    write(run / "phase0" / "n_distribution_hold.json", {"source": str((Q.PARENT_RUN / "manifest.json").relative_to(ROOT)), "source_sha256": sha(Q.PARENT_RUN / "manifest.json"),
                                                        "per_image": [{"image_id": k, "N": int(v["N_content"]), "S": 21} for k, v in sorted(identity.items())],
                                                        "S_note": "21 structural prefix rows for every image in the parent main (audited s == 21)"})
    # ---- installed transformers sources the ragged-cache design was read from
    import transformers
    base = Path(transformers.__file__).resolve().parent
    srcs = ["models/qwen2_5_vl/modeling_qwen2_5_vl.py", "masking_utils.py", "integrations/sdpa_attention.py", "cache_utils.py"]
    import torch
    write(run / "phase0" / "transformers_sources.json", {"transformers": transformers.__version__, "torch": torch.__version__,
                                                         "parent_run_recorded_transformers": read(Q.PARENT_RUN / "integration_runtime.json")["transformers"],
                                                         "files": {s: sha(base / s) for s in srcs}, "site_packages": str(base)})
    if table["status"] != "PASS":
        raise RuntimeError("BLOCKED: k_l table mismatch or a layer with k_l == 0 on a HOLD image")
    print(json.dumps({"vectors": "PASS", "run": str(run), "sha": {a: docs[a]["layer_budget_sha256"] for a in Q.ALLOCS}, "pyramidkv": public["status"],
                      "N": [ns[0], n_low_median, ns[-1]], "observed_first_last": observed, "min_k": table["min_k_overall"]}, indent=1), flush=True)


def protection_before(run: Path):
    """Full SHA of results/docs/mmimpress/scripts/tests/README (new paths excluded) and a stat
    inventory of kvstore* and of every other run.  Written once; a re-freeze keeps the original."""
    from mmimpress.recompute_kv25 import sha_manifest, stat_inventory
    path = run / "protection_before.json"
    if path.exists():
        doc = read(path)
        return {"sha256_files": doc["sha256_files"], "runs_inventory_files": doc["runs_inventory_files"], "reused": True}
    t0 = time.time()
    excl = [f"results/{run.name}"] + Q.NEW_PATHS
    shas = {k: v for k, v in sha_manifest([ROOT / x for x in ["results", "docs", "mmimpress", "scripts", "tests", "README.md"]], ROOT, exclude=excl).items() if "__pycache__" not in k}
    kv = stat_inventory(sorted(ROOT.glob("kvstore*")), ROOT)
    runs = stat_inventory(sorted(p for p in (ROOT / "runs").iterdir() if p.name != run.name), ROOT)
    write(path, {"stage": "before", "utc": time.time(), "elapsed_s": time.time() - t0, "excluded_new_paths": excl,
                 "sha256_files": len(shas), "kvstore_inventory_files": len(kv), "runs_inventory_files": len(runs),
                 "sha256": shas, "kvstore_inventory": kv, "runs_inventory": runs})
    return {"sha256_files": len(shas), "runs_inventory_files": len(runs), "reused": False}


def stage_refreeze(run: Path):
    """Re-freeze after a recorded amendment.  Nothing is deleted: the superseded config and every
    gate/phase record produced under it move to superseded_N/.  Never allowed once main exists."""
    amendments = sorted(run.glob("freeze_amendment_*.json"))
    assert amendments, "write freeze_amendment_<n>.json (cause, fix, files) before re-freezing"
    assert not (run / "main").exists(), "main has started; a re-freeze is not allowed"
    n = len(list(run.glob("superseded_*"))) + 1
    assert len(amendments) >= n, "one recorded amendment per re-freeze"
    dst = run / f"superseded_{n}"
    dst.mkdir()
    moved = []
    for name in ("config.json", "manifest.json", "storage_plan.json", "CONTRACT.md", "execution_manifest.json", "final_source_freeze.json",
                 "source_modifications.json", "correctness_manifest.json", "source_frozen", "gates", "integration", "integration_validation.json",
                 "integration_runtime.json", "smoke", "scratch", "CURRENT_STATUS.json"):
        if (run / name).exists():
            shutil.move(str(run / name), str(dst / name))
            moved.append(name)
    for p in sorted(run.glob("interruption_*.json")) + sorted(run.glob("main_runtime_*.json")):
        shutil.move(str(p), str(dst / p.name))
        moved.append(p.name)
    write(dst / "superseded.json", {"index": n, "moved": moved, "amendment": amendments[n - 1].name, "at_unix": time.time()})
    stage_freeze(run)


def stage_freeze(run: Path):
    from mmimpress.mt_gqa import workload_sha256
    assert Q.CONTRACT.exists(), "write the contract before freezing"
    assert (run / "design_ragged_cache.md").exists(), "write the ragged-cache design before freezing"
    assert not (run / "config.json").exists(), "already frozen; amendments go to freeze_amendment_*.json"
    docs = Q.load_vectors(run)
    vec = read(run / "vectors" / "vectors_sha256.json")
    for a in Q.ALLOCS:
        assert sha(Q.vector_path(run, a)) == vec[a]["file_sha256"]
    assert read(run / "phase0" / "k_tables.json")["status"] == "PASS"

    # ---- storage plan (measured)
    free = shutil.disk_usage(ROOT).free
    row_all_layers = 28 * 2 * 4 * 128 * 2                      # bytes per prefix row across all layers, K+V
    n_max, s = 529, 21
    fullload = (math.ceil(n_max / 64) * 64 + s) * row_all_layers
    quarter = (math.ceil(n_max / 4) + s) * row_all_layers
    bundle = fullload + 4 * quarter + 8 * (1 << 20)
    plan = dict(minimum_reserve_bytes=30 * Q.GiB, peak_working_set_bytes=4 * Q.GiB, largest_image_bundle_bound_bytes=1 * Q.GiB,
                maximum_new_payload_bytes=20 * Q.GiB, raw_growth_bound_bytes=10 * Q.GiB, validation_growth_bound_bytes=8 * Q.GiB,
                failed_payload_bound_bytes=10 * Q.GiB, required_initial_free_bytes=72 * Q.GiB, free_bytes_at_freeze=free,
                free_gib_at_freeze=free / Q.GiB, measured_at_unix=time.time(),
                estimated_live_bundle_bytes_per_image=bundle,
                estimate_basis="FullLoad (ceil(N/64)*64 + S rows) + B + 3 packed ((ceil(N/4) + S) rows each) at N = 529, S = 21; 57,344 B per row over 28 layers K+V",
                raw_growth_basis="parent 3-arm main = 1.4 GB raw/evidence -> 6 arms about 3 GB",
                policy="FRESH_IMAGE_STREAMING: one live image; stores built -> measured -> audited -> committed -> run-local scratch deleted by exact SHA allowlist",
                mount=subprocess.check_output(["findmnt", "-J", "-T", str(run)], text=True),
                scratch_cleanup="only run-owned exact allowlist after independent image audit, hashes, atomic commit; gate stores under runs/<run>/gates/**/stores are removed per case",
                deletion_scope=[f"runs/{run.name}/scratch/** (exact SHA allowlist after audit + commit)", f"runs/{run.name}/gates/gpu/*/stores/** (per-case allowlist)"])
    plan["status"] = "PASS" if free >= plan["required_initial_free_bytes"] else "BLOCKED_STORAGE"
    write(run / "storage_plan.json", plan)
    if plan["status"] != "PASS":
        raise RuntimeError(f"BLOCKED_STORAGE: free {free} < required {plan['required_initial_free_bytes']}")

    protection = protection_before(run)
    # existing Qwen core sources must equal the parent main freeze (store identity depends on them)
    parent_cfg = read(Q.PARENT_RUN / "config.json")
    core = ["mmimpress/qwen25/runner.py", "mmimpress/qwen25/store.py", "mmimpress/qwen25/vision.py", "mmimpress/layer_budget.py",
            "scripts/110_eval_qwen_mt_gqa_B.py", "scripts/111_audit_qwen_mt_gqa_B.py", "scripts/116_qwen_prefixstore_semantic.py", "scripts/main_b_support.py"]
    changed = sorted(k for k in core if k in parent_cfg["source_sha256"] and sha(ROOT / k) != parent_cfg["source_sha256"][k])
    write(run / "source_modifications.json", {"reference": "parent prefixstore3 config.json source_sha256", "core_checked": core, "changed_core_files": changed,
                                              "not_in_parent_freeze": [k for k in core if k not in parent_cfg["source_sha256"]], "new_files": Q.NEW_PATHS,
                                              "note": "this run only adds files; mmimpress/qwen25/{runner,store,vision}.py, layer_budget.py and every existing script are unmodified"})
    if changed:
        raise RuntimeError(f"BLOCKED: existing Qwen core sources differ from the parent main freeze: {changed}")

    # ---- manifest / config (parent Qwen main manifest; 6 arms)
    manifest = read(Q.PARENT_RUN / "manifest.json")
    cfg = read(Q.PARENT_RUN / "config.json")
    assert sha(ROOT / "data/mt_gqa/dialogues.json") == Q.INDEX_SHA == manifest["index_sha256"]
    assert workload_sha256(manifest["dialogues"]) == Q.WORKLOAD_SHA == manifest["workload_sha256"]
    assert read(ROOT / "data/mt_gqa/dialogues.json")["dialogues"] == manifest["dialogues"]
    assert len(manifest["dialogues"]) == 4061 and len(manifest["image_identity"]) == 398
    for iid, x in manifest["image_identity"].items():
        assert sha(ROOT / x["image_path"]) == x["sha256"], iid
    for name, digest in cfg["model"]["config_file_sha256"].items():
        assert sha(Path(cfg["model"]["snapshot"]) / name) == digest, name
    for name, x in cfg["model"]["weight_files"].items():
        p = Path(cfg["model"]["snapshot"]) / name
        assert str(p.resolve()) == x["resolved"] and p.stat().st_size == x["bytes"], name
    new_manifest = {"schema_version": Q.SCHEMA, "dialogues": manifest["dialogues"], "image_identity": manifest["image_identity"],
                    "index_sha256": manifest["index_sha256"], "workload_sha256": manifest["workload_sha256"],
                    "integration_dialogue_ids": manifest["integration_dialogue_ids"], "smoke_dialogue_ids": manifest["smoke_dialogue_ids"],
                    "history_policy": manifest["history_policy"], "methods": list(Q.METHODS),
                    "parent_manifest": str((Q.PARENT_RUN / "manifest.json").relative_to(ROOT)), "parent_manifest_sha256": sha(Q.PARENT_RUN / "manifest.json"),
                    "fixture_note": "integration/smoke dialogue ids and the core pairs are correctness fixtures reused from the parent run, not a quality holdout"}
    write(run / "manifest.json", new_manifest)
    shutil.copy2(Q.PARENT_RUN / "correctness_manifest.json", run / "correctness_manifest.json")
    from mmimpress.qwen25.runner import POSITION_POLICY, Qwen25Runner
    from mmimpress.qwen25.layerwise import packed_code_revision
    import torch
    import transformers
    env = {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
           "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
           "read_ahead_kb": Path("/sys/block/nvme0n1/queue/read_ahead_kb").read_text().strip()}
    config = {"schema_version": Q.SCHEMA, "experiment": Q.EXPERIMENT, "run": str(run.relative_to(ROOT)),
              "contract": str(Q.CONTRACT.relative_to(ROOT)), "contract_sha256": sha(Q.CONTRACT),
              "design_sha256": sha(run / "design_ragged_cache.md"), "ragged_cache_design": "A_forward_pre_hook_right_aligned_mask",
              "methods": list(Q.METHODS), "packed_methods": dict(Q.PACKED), "control": Q.U_PACKED, "primary": Q.PRIMARY, "sensitivity": Q.SENSITIVITY,
              "model": cfg["model"], "position_policy": POSITION_POLICY, "chunk_size": 64, "ratio": 0.25, "seed": 1234, "cpu_threads": 2,
              "history_policy": "method_local_generated", "provisioning": "FRESH_IMAGE_STREAMING", "B_storage_policy": "prefix25",
              "packed_store_format": "qwen25_bf16_prefix25_packed_v1", "num_layers": 28, "total_ratio": 0.25, "r_min": 0.0, "r_max": 0.60,
              "llava_layer_budget_defaults_not_used": {"NUM_LAYERS": 32, "R_MIN": 0.10},
              "vectors": {a: {"file": str(Q.vector_path(run, a).relative_to(ROOT)), "file_sha256": vec[a]["file_sha256"],
                              "layer_budget_sha256": docs[a]["layer_budget_sha256"], "r_min": docs[a]["r_min"], "r_max": docs[a]["r_max"],
                              "num_layers": docs[a]["num_layers"]} for a in Q.ALLOCS},
              "core_model_store_revision": Qwen25Runner()._code_revision(), "packed_code_revision": packed_code_revision(),
              "transformers_sources": read(run / "phase0" / "transformers_sources.json"),
              "source_sha256": Q.source_hashes(), "storage_plan_sha256": sha(run / "storage_plan.json"),
              "manifest_sha256": sha(run / "manifest.json"), "correctness_manifest_sha256": sha(run / "correctness_manifest.json"),
              "parent_run": str(Q.PARENT_RUN.relative_to(ROOT)), "parent_config_sha256": sha(Q.PARENT_RUN / "config.json"),
              "hold": {"images": 398, "dialogues": 4061, "requests_per_arm": 12183, "arms": 6, "requests": 73098, "smoke_requests": 72, "integration_requests": 72},
              "decision_rule": {"primary": "delta = HOLD followup (T2-T3) accuracy, PYR-b20_packed - U_packed, same run, image-cluster paired bootstrap 95% CI (10,000, seed 1234)",
                                "GO": "delta >= +1.0 pp & CI excludes 0 & SSD MB/followup <= U_packed*1.02 & TTFT mean <= U_packed*1.05",
                                "PARTIAL": "delta >= +0.5 pp with CI including 0, or GO with exactly one of MB/TTFT violated", "NO-GO": "otherwise",
                                "beta": "primary is beta = 20 only; PYR-b2p5 - U_packed and PYR-b20 - PYR-b2p5 are reported, never used for the verdict",
                                "integrity": "U_packed vs B token-exact on all 12,183 requests, else decision = PENDING_REVIEW"},
              "bootstrap": {"unit": "image", "resamples": 10000, "seed": 1234}, "analysis_seed": 1234, "analysis_resamples": 10000,
              "environment": env, "disk_free_bytes_at_freeze": free,
              "expected_duration_after_smoke": {"basis": "parent 3-arm main: 36,549 requests in 4,863 s (0.133 s/request incl. conditioning, audit, cleanup)",
                                                "estimate_hours": [2.7, 3.4], "requests": 73098}}
    write(run / "config.json", config)
    for n in config["source_sha256"]:
        p = run / "source_frozen" / n
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / n, p)
    shutil.copy2(Q.CONTRACT, run / "CONTRACT.md")
    from qwen_pyr_support import load_script
    E = load_script("_qpyr_eval_for_freeze", "169_eval_qwen_layer_budget_pyr.py")
    write(run / "execution_manifest.json", [dict(image_id=d["image_id"], dialog_id=d["dialog_id"], canonical_ordinal=i,
                                                 method_order=list(E.method_order(i)), turn_order=[1, 2, 3]) for i, d in enumerate(new_manifest["dialogues"])])
    write(run / "final_source_freeze.json", dict(files=config["source_sha256"], contract_sha256=config["contract_sha256"], config_sha256=sha(run / "config.json"),
                                                 manifest_sha256=sha(run / "manifest.json"), design_sha256=config["design_sha256"], existing_sources_changed=False,
                                                 entrypoint="scripts/169_eval_qwen_layer_budget_pyr.py (ports the 110 rules to 6 arms; 110/111 are never executed)"))
    res = ROOT / "results" / run.name
    res.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"freeze": "PASS", "run": str(run), "contract_sha256": config["contract_sha256"], "config_sha256": sha(run / "config.json"),
                      "free_gib": round(free / Q.GiB, 2), "required_gib": 72, "protection": protection,
                      "sources": len(config["source_sha256"])}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["vectors", "freeze", "refreeze"], required=True)
    ap.add_argument("--run", type=Path)
    a = ap.parse_args()
    if a.stage == "vectors":
        run = a.run.resolve() if a.run else ROOT / "runs" / f"{Q.EXPERIMENT}_{Q.utc()}"
        stage_vectors(run)
    elif a.stage == "freeze":
        stage_freeze(a.run.resolve())
    else:
        stage_refreeze(a.run.resolve())


if __name__ == "__main__":
    main()

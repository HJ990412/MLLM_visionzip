#!/usr/bin/env python3
"""Phase 0 of the Qwen2.5-VL retention-ratio sweep (U25 / U30 / U40 / U50 / U100 packed).

  --stage vectors : create the run root, the five frozen uniform ratio vectors, the k tables on
                    the 398 HOLD images (float layer_budget result vs integer fraction), the
                    N = 1..3000 arithmetic check, the installed-transformers source record
  --stage freeze  : (after the contract exists) storage plan with measured free space,
                    protection "before" inventory, manifest.json, config.json, frozen sources
  --stage refreeze: after a recorded freeze_amendment_*.json; superseded records are preserved

No data is used to build or select anything.  Nothing outside the new run/results roots is written.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import qwen_sweep_support as Q
from qwen_sweep_support import ROOT, read, sha, write

ROW_ALL = 28 * 2 * 4 * 128 * 2          # bytes of one prefix row over 28 layers, K+V (BF16, 4 KV heads x 128)


def stage_vectors(run: Path):
    from mmimpress.layer_budget import budget_document, budget_sha256
    from mmimpress.qwen25 import ratio_packed as RP
    run.mkdir(parents=True, exist_ok=False)
    (run / "vectors").mkdir()
    files, recorded = {}, {}
    for alloc in Q.ALLOCS:
        r = Q.vector(alloc)
        p, q = Q.FRACTION[alloc]
        source = {"kind": f"uniform {Q.RATIO[alloc]:g} for every decoder layer", "integer_fraction": [p, q],
                  "k_rule": "k = ceil(ratio * N) content rows in every layer; first k of the image-only saliency order"}
        doc = budget_document(alloc, r, model=Q.MODEL_ID, source=source, **Q.budget_kw(alloc))
        digest = budget_sha256(r)
        if doc["layer_budget_sha256"] != digest or (alloc in Q.RATIO_SHA and digest != Q.RATIO_SHA[alloc]):
            raise RuntimeError(f"BLOCKED: ratio SHA256 mismatch for {alloc}: {digest}")
        assert RP.ratio_fraction(Q.RATIO[alloc]).as_integer_ratio() == (p, q)
        with Q.vector_path(run, alloc).open("x") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")
        files[alloc] = sha(Q.vector_path(run, alloc))
        recorded[alloc] = {"layer_budget_sha256": digest, "file_sha256": files[alloc], "file": str(Q.vector_path(run, alloc).relative_to(ROOT)),
                           "ratio": Q.RATIO[alloc], "integer_fraction": [p, q], "sum": sum(r), "expected_sum_28r": 28 * p / q,
                           "num_layers": doc["num_layers"], "total_ratio": doc["total_ratio"], "r_min": doc["r_min"], "r_max": doc["r_max"],
                           "sha_source": "prompt (fixed before the run)" if alloc in Q.RATIO_SHA else "computed at Phase 0 and recorded here"}
    write(run / "vectors" / "vectors_sha256.json", recorded)
    docs = Q.load_vectors(run)

    # ---- pyramidkv-run U vector: same ratios, same SHA (the document differs only in r_max / alloc_id)
    old_u = read(Q.PYR_RUN / "vectors" / "layer_budget_qwen_U.json")
    assert old_u["layer_budget_sha256"] == docs["U25"]["layer_budget_sha256"] and old_u["layer_keep_ratios"] == docs["U25"]["layer_keep_ratios"]

    # ---- k tables on the 398 HOLD images (N from the frozen manifest; re-checked against actual geometry by every audit)
    manifest = read(Q.PYR_RUN / "manifest.json")
    identity = manifest["image_identity"]
    ns = sorted(int(x["N_content"]) for x in identity.values())
    assert len(ns) == 398
    n_low_median = statistics.median_low(ns)
    per_n = {}
    for n in sorted(set(ns)):
        per_n[n] = {}
        for a in Q.ALLOCS:
            counts = Q.keep_counts(docs[a], n)
            k = Q.integer_k(a, n)
            assert counts == [k] * 28 and sum(counts) == 28 * k and RP.uniform_keep_count(Q.RATIO[a], n) == k and math.ceil(Q.RATIO[a] * n) == k, (a, n)
            per_n[n][a] = k
        assert per_n[n]["U100"] == n and per_n[n]["U25"] == (n + 3) // 4
        assert per_n[n]["U25"] <= per_n[n]["U30"] <= per_n[n]["U40"] <= per_n[n]["U50"] <= per_n[n]["U100"]
    observed = {n: per_n[n] for n in Q.EXPECTED_K}
    # ---- N = 1..3000: float layer_budget result vs integer fraction, every ratio
    mismatches = {a: [n for n in range(1, 3001) if Q.keep_counts(docs[a], n) != [Q.integer_k(a, n)] * 28 or math.ceil(Q.RATIO[a] * n) != Q.integer_k(a, n)] for a in Q.ALLOCS}
    s = 21
    table = {"images": 398, "distinct_N": len(per_n), "N_min": ns[0], "N_max": ns[-1], "N_median_arithmetic": statistics.median(ns),
             "N_median_low_actual_image": n_low_median, "N_counts": {str(n): ns.count(n) for n in sorted(set(ns))},
             "k_by_N": {str(n): per_n[n] for n in per_n},
             "packed_bytes_by_N": {str(n): {a: (per_n[n][a] + s) * ROW_ALL for a in Q.ALLOCS} for n in per_n},
             "expected_k": {str(k): v for k, v in Q.EXPECTED_K.items()}, "observed_k": {str(k): v for k, v in observed.items()},
             "expected_match": observed == Q.EXPECTED_K,
             "float_vs_integer_fraction_mismatches_N_1_to_3000": mismatches,
             "rule": "k = ceil(ratio*N) in every layer; sum over layers = 28*k; float layer_budget result == integer fraction ceil(p*N/q)",
             "explicit_arguments": {a: Q.budget_kw(a) for a in Q.ALLOCS},
             "mean_packed_MB_over_HOLD_images": {a: statistics.mean((per_n[n][a] + s) * ROW_ALL for n in ns) / 1e6 for a in Q.ALLOCS},
             "status": "PASS" if observed == Q.EXPECTED_K and not any(mismatches.values()) else "BLOCKED"}
    write(run / "phase0" / "k_tables.json", table)
    with (run / "phase0" / "k_tables.csv").open("x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "images"] + [f"k_{a}" for a in Q.ALLOCS] + [f"packed_MB_{a}" for a in Q.ALLOCS])
        for n in sorted(per_n):
            w.writerow([n, ns.count(n)] + [per_n[n][a] for a in Q.ALLOCS] + [f"{(per_n[n][a] + s) * ROW_ALL / 1e6:.3f}" for a in Q.ALLOCS])
    write(run / "phase0" / "n_distribution_hold.json", {"source": str((Q.PYR_RUN / "manifest.json").relative_to(ROOT)), "source_sha256": sha(Q.PYR_RUN / "manifest.json"),
                                                        "per_image": [{"image_id": k, "N": int(v["N_content"]), "S": s} for k, v in sorted(identity.items())],
                                                        "S_note": "21 structural prefix rows for every image (audited s == 21 in both earlier Qwen mains)"})
    import torch
    import transformers
    base = Path(transformers.__file__).resolve().parent
    srcs = ["models/qwen2_5_vl/modeling_qwen2_5_vl.py", "masking_utils.py", "integrations/sdpa_attention.py", "cache_utils.py"]
    current = {"transformers": transformers.__version__, "torch": torch.__version__, "files": {x: sha(base / x) for x in srcs}, "site_packages": str(base)}
    previous = read(Q.PYR_RUN / "phase0" / "transformers_sources.json")
    current["same_as_pyramidkv_run"] = (previous["transformers"], previous["torch"], previous["files"]) == (current["transformers"], current["torch"], current["files"])
    write(run / "phase0" / "transformers_sources.json", current)
    if table["status"] != "PASS" or not current["same_as_pyramidkv_run"]:
        raise RuntimeError("BLOCKED: k table mismatch, float/integer k mismatch, or the installed transformers differs from the PyramidKV run")
    print(json.dumps({"vectors": "PASS", "run": str(run), "sha": {a: recorded[a]["layer_budget_sha256"] for a in Q.ALLOCS},
                      "N": [ns[0], n_low_median, ns[-1]], "observed_k": observed, "mean_packed_MB": table["mean_packed_MB_over_HOLD_images"]}, indent=1), flush=True)


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
    for p in sorted(run.glob("interruption_*.json")) + sorted(run.glob("main_runtime_*.json")) + sorted(run.glob("*.log")):
        shutil.move(str(p), str(dst / p.name))
        moved.append(p.name)
    write(dst / "superseded.json", {"index": n, "moved": moved, "amendment": amendments[n - 1].name, "at_unix": time.time()})
    stage_freeze(run)


def stage_freeze(run: Path):
    from mmimpress.mt_gqa import workload_sha256
    assert Q.CONTRACT.exists(), "write the contract before freezing"
    assert not (run / "config.json").exists(), "already frozen; amendments go to freeze_amendment_*.json"
    docs = Q.load_vectors(run)
    vec = read(run / "vectors" / "vectors_sha256.json")
    for a in Q.ALLOCS:
        assert sha(Q.vector_path(run, a)) == vec[a]["file_sha256"]
    ktab = read(run / "phase0" / "k_tables.json")
    assert ktab["status"] == "PASS"

    # ---- storage plan (measured)
    free = shutil.disk_usage(ROOT).free
    n_max, s = 529, 21
    fullload = (math.ceil(n_max / 64) * 64 + s) * ROW_ALL
    packed = sum((Q.integer_k(a, n_max) + s) * ROW_ALL for a in Q.ALLOCS)
    bundle = fullload + packed + 8 * (1 << 20)
    pyr_raw = sum(p.stat().st_size for p in (Q.PYR_RUN / "main").rglob("*") if p.is_file())
    plan = dict(minimum_reserve_bytes=30 * Q.GiB, peak_working_set_bytes=4 * Q.GiB, largest_image_bundle_bound_bytes=1 * Q.GiB,
                maximum_new_payload_bytes=20 * Q.GiB, raw_growth_bound_bytes=10 * Q.GiB, validation_growth_bound_bytes=8 * Q.GiB,
                failed_payload_bound_bytes=10 * Q.GiB, required_initial_free_bytes=72 * Q.GiB, free_bytes_at_freeze=free,
                free_gib_at_freeze=free / Q.GiB, measured_at_unix=time.time(),
                estimated_live_bundle_bytes_per_image=bundle,
                estimate_basis="FullLoad (ceil(N/64)*64 + S rows) + five packed stores ((ceil(r*N) + S) rows, r = 0.25/0.30/0.40/0.50/1.00) at N = 529, S = 21; 57,344 B per row over 28 layers K+V",
                raw_growth_basis=f"PyramidKV 6-arm main raw/evidence = {pyr_raw / 1e9:.2f} GB -> 7 arms about {pyr_raw * 7 / 6 / 1e9:.2f} GB",
                policy="FRESH_IMAGE_STREAMING: one live image; stores built -> measured -> audited -> committed -> run-local scratch deleted by exact SHA allowlist",
                mount=subprocess.check_output(["findmnt", "-J", "-T", str(run)], text=True),
                scratch_cleanup="only run-owned exact allowlist after independent image audit, hashes, atomic commit; gate stores under runs/<run>/gates/**/stores are removed per case",
                deletion_scope=[f"runs/{run.name}/scratch/** (exact SHA allowlist after audit + commit)", f"runs/{run.name}/gates/gpu/*/stores/** (per-case allowlist)"])
    plan["status"] = "PASS" if free >= plan["required_initial_free_bytes"] else "BLOCKED_STORAGE"
    write(run / "storage_plan.json", plan)
    if plan["status"] != "PASS":
        raise RuntimeError(f"BLOCKED_STORAGE: free {free} < required {plan['required_initial_free_bytes']}")

    protection = protection_before(run)
    # every source frozen by the PyramidKV run (existing Qwen core + that run's new files) must be unchanged
    pyr_cfg = read(Q.PYR_RUN / "config.json")
    changed = sorted(k for k, d in pyr_cfg["source_sha256"].items() if not (ROOT / k).exists() or sha(ROOT / k) != d)
    write(run / "source_modifications.json", {"reference": "PyramidKV run config.json source_sha256 (every runtime source frozen there)",
                                              "checked_files": len(pyr_cfg["source_sha256"]), "changed_or_missing_files": changed, "new_files": Q.NEW_PATHS,
                                              "note": "this run only adds files; every existing source, including the files added by the PyramidKV run, is unmodified"})
    if changed:
        raise RuntimeError(f"BLOCKED: existing sources differ from the PyramidKV run freeze: {changed}")

    # ---- manifest / config (PyramidKV-run manifest; 7 arms)
    manifest = read(Q.PYR_RUN / "manifest.json")
    assert sha(ROOT / "data/mt_gqa/dialogues.json") == Q.INDEX_SHA == manifest["index_sha256"]
    assert workload_sha256(manifest["dialogues"]) == Q.WORKLOAD_SHA == manifest["workload_sha256"]
    assert read(ROOT / "data/mt_gqa/dialogues.json")["dialogues"] == manifest["dialogues"]
    assert len(manifest["dialogues"]) == 4061 and len(manifest["image_identity"]) == 398
    assert manifest["integration_dialogue_ids"] == read(Q.PARENT_RUN / "manifest.json")["integration_dialogue_ids"]
    assert manifest["smoke_dialogue_ids"] == read(Q.PARENT_RUN / "manifest.json")["smoke_dialogue_ids"]
    for iid, x in manifest["image_identity"].items():
        assert sha(ROOT / x["image_path"]) == x["sha256"], iid
    model = pyr_cfg["model"]
    for name, digest in model["config_file_sha256"].items():
        assert sha(Path(model["snapshot"]) / name) == digest, name
    for name, x in model["weight_files"].items():
        p = Path(model["snapshot"]) / name
        assert str(p.resolve()) == x["resolved"] and p.stat().st_size == x["bytes"], name
    new_manifest = {"schema_version": Q.SCHEMA, "dialogues": manifest["dialogues"], "image_identity": manifest["image_identity"],
                    "index_sha256": manifest["index_sha256"], "workload_sha256": manifest["workload_sha256"],
                    "integration_dialogue_ids": manifest["integration_dialogue_ids"], "smoke_dialogue_ids": manifest["smoke_dialogue_ids"],
                    "history_policy": manifest["history_policy"], "methods": list(Q.METHODS),
                    "parent_manifest": str((Q.PYR_RUN / "manifest.json").relative_to(ROOT)), "parent_manifest_sha256": sha(Q.PYR_RUN / "manifest.json"),
                    "fixture_note": "integration/smoke dialogue ids and the core pairs are correctness fixtures reused from the earlier Qwen runs, not a quality holdout"}
    write(run / "manifest.json", new_manifest)
    shutil.copy2(Q.PYR_RUN / "correctness_manifest.json", run / "correctness_manifest.json")
    from mmimpress.qwen25.layerwise import packed_code_revision
    from mmimpress.qwen25.ratio_packed import STORE_FORMAT, ratio_code_revision
    from mmimpress.qwen25.runner import POSITION_POLICY, Qwen25Runner
    import torch
    import transformers
    assert packed_code_revision() == pyr_cfg["packed_code_revision"], "PyramidKV-run packed code changed"
    env = {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
           "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
           "read_ahead_kb": Path("/sys/block/nvme0n1/queue/read_ahead_kb").read_text().strip()}
    pyr_hours = (read(Q.PYR_RUN / "main_finished_at.json")["at_unix"] - read(Q.PYR_RUN / "main_started_at.json")["at_unix"]) / 3600
    config = {"schema_version": Q.SCHEMA, "experiment": Q.EXPERIMENT, "run": str(run.relative_to(ROOT)),
              "contract": str(Q.CONTRACT.relative_to(ROOT)), "contract_sha256": sha(Q.CONTRACT),
              "methods": list(Q.METHODS), "packed_methods": dict(Q.PACKED), "ratios": dict(Q.RATIO), "integer_fractions": {a: list(f) for a, f in Q.FRACTION.items()},
              "model": model, "position_policy": POSITION_POLICY, "chunk_size": 64, "seed": 1234, "cpu_threads": 2,
              "history_policy": "method_local_generated", "provisioning": "FRESH_IMAGE_STREAMING",
              "packed_store_format": STORE_FORMAT, "num_layers": 28, "r_min": Q.R_MIN, "r_max": Q.R_MAX,
              "hit_path": "mmimpress.qwen25.layerwise.run_cache_layerwise (PyramidKV run, unchanged) via mmimpress.qwen25.ratio_packed",
              "llava_layer_budget_defaults_not_used": {"NUM_LAYERS": 32, "R_MIN": 0.10, "R_MAX": 0.60, "TOTAL_RATIO": 0.25},
              "vectors": {a: {"file": str(Q.vector_path(run, a).relative_to(ROOT)), "file_sha256": vec[a]["file_sha256"],
                              "layer_budget_sha256": docs[a]["layer_budget_sha256"], "total_ratio": docs[a]["total_ratio"], "r_min": docs[a]["r_min"],
                              "r_max": docs[a]["r_max"], "num_layers": docs[a]["num_layers"]} for a in Q.ALLOCS},
              "core_model_store_revision": Qwen25Runner()._code_revision(), "packed_code_revision": ratio_code_revision(),
              "pyr_packed_code_revision": packed_code_revision(),
              "transformers_sources": read(run / "phase0" / "transformers_sources.json"),
              "source_sha256": Q.source_hashes(), "storage_plan_sha256": sha(run / "storage_plan.json"),
              "manifest_sha256": sha(run / "manifest.json"), "correctness_manifest_sha256": sha(run / "correctness_manifest.json"),
              "pyramidkv_run": str(Q.PYR_RUN.relative_to(ROOT)), "pyramidkv_config_sha256": sha(Q.PYR_RUN / "config.json"),
              "hold": {"images": 398, "dialogues": 4061, "requests_per_arm": 12183, "arms": 7, "requests": Q.EXPECT["main"],
                       "smoke_requests": Q.EXPECT["smoke"], "integration_requests": Q.EXPECT["integration"]},
              "report_rule": {"verdict": "none (curve is reported; no GO/NO-GO)",
                              "scope": "HOLD followup (T2-T3), image-cluster paired bootstrap 95% CI (10,000, seed 1234)",
                              "baseline_a_loss_not_detected": "smallest ratio whose (ratio - ReComp) followup accuracy CI includes 0; NONE if there is none",
                              "baseline_b_llava_25_level": f"smallest ratio whose point-estimate loss vs ReComp is <= {Q.LLAVA_U25_LOSS_PP:.2f} pp (LLaVA uniform 25%, v2 HOLD); NONE if there is none",
                              "candidate_ratios_in_order": list(Q.ALLOCS),
                              "wording": "a CI that includes 0 is 'difference not shown', never 'equivalent'; non-monotone segments are shown with their CI",
                              "integrity": "U100_packed vs fullload token-exact on all 12,183 requests, else status = PENDING_REVIEW"},
              "bootstrap": {"unit": "image", "resamples": 10000, "seed": 1234}, "analysis_seed": 1234, "analysis_resamples": 10000,
              "environment": env, "disk_free_bytes_at_freeze": free,
              "expected_duration_after_smoke": {"basis": f"PyramidKV 6-arm main: 73,098 requests in {pyr_hours:.2f} h ({pyr_hours * 3600 / 73098:.3f} s/request incl. conditioning, audit, cleanup); "
                                                         "7 arms with larger stores (U100 reads about 3.4x the bytes of U25)",
                                                "estimate_hours": [2.9, 3.8], "requests": Q.EXPECT["main"]}}
    write(run / "config.json", config)
    for n in config["source_sha256"]:
        p = run / "source_frozen" / n
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / n, p)
    shutil.copy2(Q.CONTRACT, run / "CONTRACT.md")
    E = Q.load_script("_qsweep_eval_for_freeze", Q.EVAL_SCRIPT)
    write(run / "execution_manifest.json", [dict(image_id=d["image_id"], dialog_id=d["dialog_id"], canonical_ordinal=i,
                                                 method_order=list(E.method_order(i)), turn_order=[1, 2, 3]) for i, d in enumerate(new_manifest["dialogues"])])
    write(run / "final_source_freeze.json", dict(files=config["source_sha256"], contract_sha256=config["contract_sha256"], config_sha256=sha(run / "config.json"),
                                                 manifest_sha256=sha(run / "manifest.json"), existing_sources_changed=False,
                                                 entrypoint=f"scripts/{Q.EVAL_SCRIPT} (thin adaptation of 169 to 7 arms; 110/111/169 are never executed)"))
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

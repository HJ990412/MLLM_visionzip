#!/usr/bin/env python3
"""v2 gates and Phase 0 freeze (contract docs/layer_budget_kv25_v2_contract.md).

Stages
  --stage freeze : Phase 0 — contract SHA, v1 reuse SHAs, prior-work source
                   record, storage plan, config, protection "before" snapshot
  --stage cpu    : complete CPU suite (v1 stage_cpu; GLM modules separately)
  --stage g3     : Σ k_l = B(N) for every CAL/DEV/HOLD image and every vector
  --stage g9     : dd-based cold sequential read throughput by file size
  --stage gpu    : G1/G2/G4 (v1 gpu_case) + G6 packed equivalence + G7 read
                   path on the 5 fixed GPU cases
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

from mmimpress.layer_budget import load_budget_document
from mmimpress.prefix25 import process_io, seal_integrity, sha
from mmimpress.recompute_kv25 import sha_manifest, stat_inventory

CONTRACT = ROOT / "docs/layer_budget_kv25_v2_contract.md"
V1_RUN = ROOT / "runs/layer_budget_kv25_20261105T125754Z"
V1_RES = ROOT / "results/layer_budget_kv25_20261105T125754Z"
V1_REUSE = {
    "results/layer_budget_kv25_20261105T125754Z/layer_budget_llava_A1-2.json": "c2bb0d4e9353bc4b461b850f09ef36d25c09a912fc2385786e4a99764ef94b16",
    "results/layer_budget_kv25_20261105T125754Z/layer_budget_llava_U.json": "d6ea0c408baf2a8f5b73181579e7c4b70d090c69c7c1a3e8f010104ac22196e3",
    "results/layer_budget_kv25_20261105T125754Z/layer_budget_llava_F-front.json": "6858154b99fd461fb0a1a06957ff59caa8b338b78af6048c89d38e3e6ac96809",
    "results/layer_budget_kv25_20261105T125754Z/phase1_sensitivity.csv": "9e6100450b916c178669c51cfaade13d8f0bdb9a9ca472696fbed083835986a3",
    "runs/layer_budget_kv25_20261105T125754Z/phase1/coverage_curves.npz": "80f0387c5a3987fa93f34745ab5477d2754af7f2f5810e7c16e4c30a7e26a215",
    "runs/layer_budget_kv25_20261105T125754Z/phase1/cal_manifest.json": "8f50d65f3074356d3d8ab74574b346755bce209f68835f7fd4ecfdf346f8762f",
}
A12_RATIO_SHA = "2c2295cfbe876adc7d9a5e2253f1e1ead13bda3718e7b30e7f34779e03bfa8c7"
NEW_PATHS = ["docs/layer_budget_kv25_v2_contract.md", "mmimpress/packed_store.py", "mmimpress/packed_serve.py", "tests/test_packed_store.py",
             "scripts/158_allocate_prior_work_vectors.py", "scripts/159_eval_layer_budget_v2_dev.py", "scripts/160_report_layer_budget_v2.py",
             "scripts/161_validate_packed_store_v2.py", "scripts/162_eval_layer_budget_v2_hold.py", "papers/layer_budget_kv25_v2_packed_prompt.md"]
PRIOR_WORK_SCRATCH = Path("/tmp/claude-1000/-home-dblab-hj-mllm-v2/4293671a-d46e-4f89-b02f-f0f90d700f98/scratchpad")
FIXED_INDEX = (0, 1, 2, 3, 4)
SEED = 1234
ATOL = RTOL = 1e-4
GiB = 2 ** 30


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_safe(obj), f, indent=1, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def _safe(x):
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_safe(v) for v in x]
    return x


# ----------------------------------------------------------------- freeze
def stage_freeze(run):
    assert CONTRACT.exists()
    for rel, digest in V1_REUSE.items():
        assert sha(ROOT / rel) == digest, f"v1 reuse artifact changed: {rel}"
    a12 = load_budget_document(V1_RES / "layer_budget_llava_A1-2.json")
    assert a12["layer_budget_sha256"] == A12_RATIO_SHA
    (run / "phase0").mkdir(exist_ok=True)
    (run / "phase2").mkdir(exist_ok=True)
    (run / "gates").mkdir(exist_ok=True)
    for a in ("U", "A1-2"):
        src = V1_RES / f"layer_budget_llava_{a}.json"; dst = run / "phase2" / f"layer_budget_llava_{a}.json"
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
        assert sha(dst) == sha(src)
    for rel in ("data_split.json", "phase0/n_distribution.json"):
        src = V1_RUN / rel; dst = run / rel
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
        assert sha(dst) == sha(src)
    v1_decision = json.loads((V1_RES / "decision.json").read_text())
    write(run / "phase0" / "v1_reuse.json", {"v1_run": str(V1_RUN), "v1_results": str(V1_RES), "sha256": V1_REUSE, "A1-2_ratio_sha256": A12_RATIO_SHA,
                                            "v1_decision": v1_decision, "v1_decision_sha256": sha(V1_RES / "decision.json"),
                                            "v1_hold_latency_io_sha256": sha(V1_RES / "hold" / "latency_io.csv")})
    # prior-work sources (read-only clones in the session scratch; key files copied into the run for permanence)
    pw = run / "phase0" / "prior_work"; pw.mkdir(exist_ok=True)
    sources = {"vl_cache": {"official_code": "NOT_FOUND (GitHub search 2026-11-06; VL-Cache/VL-Cache, dezhanTu/VL-Cache, tudezhan/VL-Cache: not found)",
                            "unofficial_repo": "github.com/bhzadjnty7/Efficient-VLM-Inference commit 8705205cf05c556c43bf37aacf4bde730f150926 (notebook; defers to the paper; not used)",
                            "paper": "arXiv 2410.23317 (ICLR 2025)", "definitions": "Eq.1 ThresholdFilter(A,p)_ij = A_ij if A_ij >= p*max_j A_ij else 0; p = 1%; Eq.2 gamma = #zeros/#causal; Alg.1 post-vision rows Q[m-tau:m], beta_l = clip((1-gamma_l)/Z * alpha*L, 0.01, 1)"},
               "squeezeattention": {"repo": "github.com/hetailang/SqueezeAttention", "commit": "a1933d18b09c668312cbabdfd7b1fa73888c0f6f",
                                    "file": "utils_hh/modify_llama_drop.py", "cosine": "L314: cosine_similarity(layer input hidden_states, residual + self_attn output, dim=1); mean over prompt tokens",
                                    "groups": "L527-543: KMeans(n_clusters=3) on per-layer cosine; class3 = highest centre -> budget `percent` (KV_class3); classes 1+2 share (L*ini_size - |c3|*percent)/(L-|c3|)",
                                    "readme_example": "--ini_size 0.21 --KV_class3 0.08"},
               "pyramidkv": {"repo": "github.com/Zefan-Cai/PyramidKV", "commit": "68cd9551a63fedf362ddb0edb008badd04c24996", "file": "pyramidkv/pyramidkv_utils.py",
                             "schedule": "L360-370: min_num=(cap-w)//beta; max_num=(cap-w)*2-min_num; steps=(max_num-min_num)//(L-1); capacity_l=max_num-l*steps; beta default 20"}}
    copies = {"squeezeattention_modify_llama_drop.py": PRIOR_WORK_SCRATCH / "prior_work/squeezeattention/utils_hh/modify_llama_drop.py",
              "pyramidkv_utils.py": PRIOR_WORK_SCRATCH / "prior_work/pyramidkv/pyramidkv/pyramidkv_utils.py",
              "vlcache_2410.23317.txt": PRIOR_WORK_SCRATCH / "vlcache_pdf/vlcache.txt",
              "vlcache_2410.23317.pdf": PRIOR_WORK_SCRATCH / "vlcache_pdf/vlcache.pdf"}
    for name, src in copies.items():
        dst = pw / name
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
        sources.setdefault("copied_files_sha256", {})[name] = sha(dst)
    write(run / "phase0" / "prior_work_sources.json", sources)
    # storage plan
    free = shutil.disk_usage(ROOT).free
    packed_max = int(32 * 720 * 2 * 32 * 128 * 2 + 2 * 32 * 33 * 32 * 128 * 2 + 8 * 2 ** 20)   # N=2880 rows + separator + aux
    raster_max = 1_382_000_000
    hold = {"arms": 7, "stored_arms": 6, "live_stores_per_image": "raster (<=1.38 GB) + 5 packed (<=0.40 GB each)",
            "largest_image_bundle_bound_bytes": raster_max + 5 * packed_max, "builder_intermediate_margin_bytes": 4 * GiB,
            "peak_working_set_bytes": raster_max + 5 * packed_max + 4 * GiB, "raw_and_evidence_growth_bound_bytes": 8 * GiB,
            "raw_growth_basis": "v1 4-arm HOLD main directory = 1.9 GB -> 7 arms ~3.4 GB", "minimum_reserve_bytes": 30 * GiB,
            "policy": "FRESH_IMAGE_STREAMING: one live image; stores built -> measured -> audited -> committed -> run-local scratch deleted by exact SHA allowlist"}
    hold["required_free_bytes"] = hold["minimum_reserve_bytes"] + hold["peak_working_set_bytes"] + hold["raw_and_evidence_growth_bound_bytes"]
    dev = {"live_stores_per_image": "raster + 2 unpacked + 7 packed (<= 5 GB), deleted per image", "raw_bound_bytes": 8 * GiB,
           "peak_working_set_bytes": raster_max + 9 * packed_max + 4 * GiB}
    plan = {"measured_at_utc": time.time(), "free_bytes_now": free, "free_gib_now": free / GiB, "packed_store_max_bytes_estimate": packed_max,
            "hold": hold, "dev": dev, "phase1_prior_bound_bytes": 1 * GiB, "results_and_protection_bound_bytes": 2 * GiB,
            "required_initial_free_bytes": hold["required_free_bytes"] + dev["raw_bound_bytes"] + 3 * GiB}
    plan["required_initial_free_gib"] = plan["required_initial_free_bytes"] / GiB
    plan["margin_bytes"] = free - plan["required_initial_free_bytes"]
    plan["status"] = "PASS" if plan["margin_bytes"] >= 0 else "BLOCKED_STORAGE"
    plan["deletion_scope"] = ["runs/<RUN>/hold/scratch/** (exact allowlist)", "runs/<RUN>/llava/dev_stores/** (guarded_rmtree)"]
    write(run / "phase0" / "storage_plan.json", plan)
    if plan["status"] != "PASS":
        raise RuntimeError(f"BLOCKED_STORAGE: free {free} < required {plan['required_initial_free_bytes']}")
    # protection before (full SHA of results/docs/mmimpress/scripts/tests, kvstore inventory), excluding this run's new paths
    t0 = time.time()
    excl = [f"results/{run.name}"] + NEW_PATHS
    shas = {k: v for k, v in sha_manifest([ROOT / x for x in ["results", "docs", "mmimpress", "scripts", "tests", "README.md"]], ROOT, exclude=excl).items() if "__pycache__" not in k}
    kv = stat_inventory(sorted(ROOT.glob("kvstore*")), ROOT)
    write(run / "protection_before.json", {"stage": "before", "utc": time.time(), "elapsed_s": time.time() - t0, "excluded_new_paths": excl,
                                           "sha256_files": len(shas), "kvstore_inventory_files": len(kv), "sha256": shas, "kvstore_inventory": kv})
    # existing-source state vs the v1 HOLD freeze (v2 must not modify the v1-touched modules)
    v1_hold = json.loads((V1_RUN / "hold" / "config.json").read_text())["source_sha256"]
    changed = sorted(k for k, d in v1_hold.items() if (ROOT / k).exists() and sha(ROOT / k) != d)
    missing = sorted(k for k in v1_hold if not (ROOT / k).exists())
    # scripts/157 was edited after the v1 HOLD freeze (amendment-aware protection check, before the v1 report);
    # v1 recorded it, nothing in v2 imports it, and it is not a store/serving module.
    known = {"scripts/157_report_layer_budget_kv25.py": "v1 report script edited after the v1 HOLD freeze (v1 REPORT protection PASS); not imported by v2"}
    write(run / "source_modifications.json", {"reference": "v1 hold/config.json source_sha256", "changed_existing_files": changed, "missing": missing,
                                              "known_v1_post_hold_edits": {k: v for k, v in known.items() if k in changed},
                                              "new_files_planned": NEW_PATHS, "note": "v2 adds modules; v1-frozen store/serving modules must be unchanged"})
    unexplained = [k for k in changed if k not in known]
    assert not unexplained and not missing, (unexplained, missing)
    env = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
           "transformers": __import__("transformers").__version__,
           "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
           "read_ahead_kb": Path("/sys/block/nvme0n1/queue/read_ahead_kb").read_text().strip(), "fio": shutil.which("fio"), "dd": shutil.which("dd")}
    config = {"run": str(run.relative_to(ROOT)), "contract": str(CONTRACT.relative_to(ROOT)), "contract_sha256": sha(CONTRACT), "contract_version": 2,
              "v1_run": str(V1_RUN.relative_to(ROOT)), "v1_contract_sha256": sha(ROOT / "docs/layer_budget_kv25_contract.md"),
              "model": "llava-hf/llava-v1.6-vicuna-7b-hf", "revision": "c916e6cdcd760b4cecd1dd4907f84ac649f93b23", "quantization": "NF4 double, BF16 compute",
              "attention": "eager", "store_dtype": "float16", "chunk_size": 64, "seed": SEED, "greedy": True, "max_new_tokens": 16,
              "total_ratio": 0.25, "r_min": 0.1, "r_max": 0.6, "num_layers": 32, "budget_policy": "layerwise_v1", "selected": "A1-2", "selected_ratio_sha256": A12_RATIO_SHA,
              "vectors": ["U", "A1-2", "VLC_offline", "SA_offline", "SA-2g_offline", "PYR", "PYR-beta"],
              "prior_work": {"vlc_p_threshold": 0.01, "sa_percent_raw": 0.25 * (0.08 / 0.21), "sa_kmeans": "exact 1-D 3-means", "pyr_beta": 20},
              "packed": {"store_format": "prefixstore_kv25_packed_v1", "read_syscall": "os.preadv into pinned host buffer, 1 call per hit", "directcopy_arm": "DEV only"},
              "dev": {"workload": "runs/prefix25_persistence_20261001T050113Z/gqa_manifest.json", "images": 40, "questions": 240,
                      "accuracy_arms": ["recompute", "fullload", "U_packed", "A1-2_packed", "VLC_offline_packed", "SA_offline_packed", "SA-2g_offline_packed", "PYR_packed", "PYR-beta_packed"],
                      "latency_arms_3x": ["U_unpacked", "U_packed", "A1-2_unpacked", "A1-2_packed", "VLC_offline_packed", "SA_offline_packed", "PYR_packed", "A1-2_packed_directcopy"],
                      "store_policy": "every arm served from its own real store"},
              "hold": {"workload": "data/mt_gqa/dialogues.json", "images": 398, "dialogues": 4061, "requests_per_arm": 12183,
                       "arms": ["recompute", "fullload", "ours_prefixstore_kv25_U_packed", "ours_prefixstore_kv25_A1-2_packed", "ours_prefixstore_kv25_VLC_offline_packed",
                                "ours_prefixstore_kv25_SA_offline_packed", "ours_prefixstore_kv25_PYR_packed"]},
              "decision_rule": {"primary": "A1-2_packed - U_packed, HOLD followup; GO: delta>=+1.0pp & CI excludes 0 & SSD MB/followup <= uniform*1.02 & TTFT mean <= uniform*1.05",
                                "PARTIAL": "delta>=+0.5pp with CI incl. 0, or exactly one of bytes/TTFT violated", "NO-GO": "otherwise",
                                "secondary": ["U_packed vs v1 U_unpacked TTFT (cross-run)", "A1-2 vs VLC/SA/PYR", "packed TTFT spread across vectors within ±2%"]},
              "bootstrap": {"unit": "image", "resamples": 10000, "seed": SEED}, "environment": env, "disk_free_bytes_at_config": free,
              "storage_plan_sha256": sha(run / "phase0" / "storage_plan.json"), "prior_work_sources_sha256": sha(run / "phase0" / "prior_work_sources.json")}
    with (run / "config.json").open("x") as f:
        json.dump(config, f, indent=1); f.write("\n")
    shutil.copy2(CONTRACT, run / "CONTRACT.md")
    print(json.dumps({"freeze": "PASS", "run": str(run), "contract_sha256": config["contract_sha256"], "storage": plan["status"], "free_gib": round(free / GiB, 2),
                      "required_gib": round(plan["required_initial_free_gib"], 2), "protection_files": len(shas)}), flush=True)


# ---------------------------------------------------------------- cpu / g3
def stage_cpu(run):
    G = load("layer_budget_gates_v1", "154_validate_layer_budget_kv25.py")
    G.stage_cpu(run)


def stage_g3(run):
    G = load("layer_budget_gates_v1", "154_validate_layer_budget_kv25.py")
    G.stage_g3(run)


# -------------------------------------------------------------------- g9
def stage_g9(run):
    """Cold sequential read throughput by file size with GNU dd (independent of mmimpress)."""
    import csv
    sizes = [("256KB", 256 << 10), ("1MB", 1 << 20), ("4MB", 4 << 20), ("16MB", 16 << 20), ("64MB", 64 << 20), ("256MB", 256 << 20)]
    base = run / "g9_io_microbench"
    base.mkdir(exist_ok=True)
    rows = []
    dd = shutil.which("dd")
    rng = np.random.default_rng(SEED)
    for label, nbytes in sizes:
        for rep in range(5):
            path = base / f"{label}_{rep}.bin"
            with path.open("wb") as f:
                f.write(rng.bytes(nbytes)); f.flush(); os.fsync(f.fileno())
            for mode in ("buffered_cold", "direct"):
                fd = os.open(path, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
                os.sync()
                cmd = [dd, f"if={path}", "of=/dev/null", f"bs={min(nbytes, 64 << 20)}"] + (["iflag=direct"] if mode == "direct" else [])
                t0 = time.perf_counter()
                p = subprocess.run(cmd, capture_output=True, text=True)
                wall = time.perf_counter() - t0
                assert p.returncode == 0, p.stderr
                # dd reports "<bytes> bytes (...) copied, <seconds> s, <rate>"
                last = p.stderr.strip().splitlines()[-1]
                secs = float(last.split("copied,")[1].split("s,")[0].strip())
                rows.append({"size_label": label, "bytes": nbytes, "rep": rep, "mode": mode, "dd_seconds": secs, "dd_gbps": nbytes / secs / 1e9 if secs > 0 else None,
                             "subprocess_wall_s": wall, "dd_report": last})
            path.unlink()
    base.rmdir()
    with (run / "io_microbench.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = {}
    for label, _ in sizes:
        for mode in ("buffered_cold", "direct"):
            g = [r["dd_gbps"] for r in rows if r["size_label"] == label and r["mode"] == mode and r["dd_gbps"]]
            summary[f"{label}/{mode}"] = {"gbps_mean": float(np.mean(g)), "gbps_min": float(np.min(g)), "gbps_max": float(np.max(g)), "n": len(g)}
    write(run / "gates" / "g9_io_microbench.json", {"status": "PASS", "tool": "GNU coreutils dd (fio not installed)", "read_ahead_kb": Path("/sys/block/nvme0n1/queue/read_ahead_kb").read_text().strip(),
                                                   "sizes": [s for s, _ in sizes], "reps": 5, "summary": summary, "csv_sha256": sha(run / "io_microbench.csv")})
    print(json.dumps(summary, indent=1), flush=True)


# ------------------------------------------------------------------- gpu
def _check_packed_slices(packed_dir, unpacked_dir):
    """G6 bytes: every packed per-layer slice == the unpacked k/v file; trailing block == sep_kv.bin."""
    from mmimpress.packed_store import PACKED_NAME, packed_file_sha256
    meta = json.loads((packed_dir / "meta.json").read_text())
    blob = (packed_dir / PACKED_NAME).read_bytes()
    assert len(blob) == meta["packed_total_bytes"]
    assert packed_file_sha256(packed_dir / PACKED_NAME) == meta["packed_sha256"]
    total = 0
    for row in meta["packed_layout"]:
        li = row["layer"]
        for kind in ("k", "v"):
            ref = (unpacked_dir / f"layer_{li:02d}" / f"{kind}.bin").read_bytes()
            off = row[f"{kind}_offset"]
            assert len(ref) == row["bytes"], (li, kind, len(ref), row["bytes"])
            assert blob[off:off + len(ref)] == ref, f"packed slice differs: layer {li} {kind}"
            total += len(ref)
    sep = (unpacked_dir / "sep_kv.bin").read_bytes()
    assert blob[meta["packed_sep_offset"]:] == sep and total + len(sep) == len(blob)
    assert (unpacked_dir / "sys_kv.pt").read_bytes() == (packed_dir / "sys_kv.pt").read_bytes()
    um = json.loads((unpacked_dir / "meta.json").read_text())
    assert um["order"] == meta["order"] and um["prefix_input_ids"] == meta["prefix_input_ids"]
    return {"layers": len(meta["packed_layout"]), "payload_bytes": total, "separator_bytes": len(sep), "total_bytes": len(blob),
            "unpacked_payload_bytes": sum((unpacked_dir / f"layer_{r['layer']:02d}" / f"{k}.bin").stat().st_size for r in meta["packed_layout"] for k in ("k", "v")),
            "slices_bit_identical": True, "packed_sha256_verified": True}


def _same_output(a, b):
    la, lb = a["all_logits"], b["all_logits"]
    return {"first_token_exact": a["first_token_id"] == b["first_token_id"], "tokens_exact": a["generated_token_ids"] == b["generated_token_ids"],
            "answer_exact": a["prediction"] == b["prediction"], "steps": len(la), "same_steps": len(la) == len(lb),
            "logits_bitwise": len(la) == len(lb) and all(torch.equal(x, y) for x, y in zip(la, lb)),
            "logits_within_1e-4": len(la) == len(lb) and all(torch.allclose(x, y, atol=ATOL, rtol=RTOL) for x, y in zip(la, lb)),
            "max_abs_delta": max(float((x - y).abs().max()) for x, y in zip(la, lb)) if len(la) == len(lb) else None}


def packed_case(exp, G, S, sample, path, docs):
    turn = {"question": sample["question"]}
    out = {"image_id": sample["image_id"], "question_id": sample["question_id"]}
    val, cap = exp.pixel(sample, turn, [], "B", "gqa")
    enc = cap["enc"]
    vs, vn = exp.runner.visual_span(enc["input_ids"])
    _, _, _, sep = exp.runner.anyres_layout(enc["image_sizes"][0], vn)
    n = vn - len(sep)
    out["N"] = n
    rec = {"B": exp.persist(cap, path / "B", sample, "B"),
           "LW_A12": G.persist_layerwise(exp, cap, path / "LW_A12", sample, docs["A1-2"]),
           "P_U": S.persist_packed(exp, cap, path / "P_U", sample, docs["U"]),
           "P_A12": S.persist_packed(exp, cap, path / "P_A12", sample, docs["A1-2"])}
    del cap, val, enc
    gc.collect(); torch.cuda.empty_cache()
    out["G6_bytes"] = {"P_U_vs_B": _check_packed_slices(path / "P_U", path / "B"), "P_A12_vs_LW_A12": _check_packed_slices(path / "P_A12", path / "LW_A12")}
    assert out["G6_bytes"]["P_U_vs_B"]["total_bytes"] == out["G6_bytes"]["P_A12_vs_LW_A12"]["total_bytes"]
    out["persistence_ms"] = {k: r["persistence_ms"] for k, r in rec.items()}
    out["packed_sha256_ms"] = {k: rec[k]["timing_ms"].get("packed_sha256_ms") for k in ("P_U", "P_A12")}
    # serve: unpacked (existing path, 65 preads) with all-step logits
    served = {}
    for arm in ("B", "LW_A12"):
        ctx, _ = exp.activate(path / arm, sample, "B")
        with S.PackedReadTrace(path / arm) as tr:
            served[arm] = exp.hit(ctx, path / arm, sample, turn, [], "B", "gqa", True)
        assert tr.summary()["pread_calls"] == 65 and tr.summary()["preadv_calls"] == 0
        served[arm]["trace"] = tr.summary()
        ctx.close()
    prompt = exp.runner.prompt(turn["question"])
    g7 = {}
    for arm in ("P_U", "P_A12"):
        ctx, act = S.activate_packed(exp, path / arm)
        for mode in ("scatter", "directcopy"):
            key = f"{arm}_{mode}"
            served[key] = S.packed_hit(exp, ctx, path / arm, prompt, logits=True, directcopy=(mode == "directcopy"))
            s = served[key]["actual_io"]
            # G7: one preadv, no pread, returned == file size, no vision/score, and question invariance of the read range
            alt = S.packed_hit(exp, ctx, path / arm, exp.prompt("What is visible?", [(turn["question"], served[key]["prediction"])], "mt"), logits=False, directcopy=(mode == "directcopy"))
            assert alt["k_per_layer"] == served[key]["k_per_layer"] and alt["selected_original_ids"] == served[key]["selected_original_ids"]
            assert alt["actual_io"]["calls"][0]["returned"] == s["calls"][0]["returned"] and alt["actual_io"]["calls"][0]["offset"] == 0
            g7[key] = {"preadv_calls": s["preadv_calls"], "pread_calls": s["pread_calls"], "returned_bytes": s["bytes"], "file_bytes": ctx.meta["packed_total_bytes"],
                       "vision_calls": served[key]["vision_calls"], "query_score_calls": served[key]["online_query_score_calls"],
                       "question_invariant_range": True, "activation": act, "packed_read_ms": served[key]["packed_read_ms"], "chunk_io_ms": served[key]["chunk_io_ms"], "scatter_ms": served[key]["scatter_ms"], "ttft_ms": served[key]["ttft_ms"]}
        ctx.close()
    out["G7"] = g7
    # G6 outputs: packed == unpacked (both copy modes), tokens exact, logits bitwise expected (reported)
    cmp = {"P_U_scatter_vs_B": _same_output(served["P_U_scatter"], served["B"]), "P_U_directcopy_vs_B": _same_output(served["P_U_directcopy"], served["B"]),
           "P_A12_scatter_vs_LW_A12": _same_output(served["P_A12_scatter"], served["LW_A12"]), "P_A12_directcopy_vs_LW_A12": _same_output(served["P_A12_directcopy"], served["LW_A12"])}
    for k, c in cmp.items():
        assert c["first_token_exact"] and c["tokens_exact"] and c["answer_exact"] and c["logits_within_1e-4"], (k, c)
    out["G6_outputs"] = cmp
    out["bytes"] = {"B_unpacked": served["B"]["actual_io"]["bytes"], "LW_A12_unpacked": served["LW_A12"]["actual_io"]["bytes"],
                    "P_U": served["P_U_scatter"]["actual_io"]["bytes"], "P_A12": served["P_A12_scatter"]["actual_io"]["bytes"]}
    assert len(set(out["bytes"].values())) == 1, out["bytes"]
    out["unpacked_vs_packed_k_per_layer_equal"] = served["P_A12_scatter"]["k_per_layer"] == served["LW_A12"]["k_per_layer"] and served["P_U_scatter"]["k_per_layer"] == [served["B"]["k_target"]] * 32
    assert out["unpacked_vs_packed_k_per_layer_equal"]
    out["ttft_ms"] = {k: served[k]["ttft_ms"] for k in served}
    out["status"] = "PASS"
    return out


def stage_gpu(run):
    P = load("prefix25_persistence_existing", "106_prefix25_persistence.py")
    G = load("layer_budget_gates_v1", "154_validate_layer_budget_kv25.py")
    S = load("packed_support_v2", "packed_support_v2.py")
    docs = {a: load_budget_document(run / "phase2" / f"layer_budget_llava_{a}.json") for a in ("U", "A1-2", "PYR")}
    assert docs["A1-2"]["layer_budget_sha256"] == A12_RATIO_SHA
    if not (run / "correctness_manifest.json").exists():
        shutil.copy2(G.REF_RUN / "correctness_manifest.json", run / "correctness_manifest.json")
    manifest = json.loads((run / "correctness_manifest.json").read_text())
    samples = [manifest["samples"][i] for i in G.FIXED_INDEX]
    ref_out = {s["image_id"]: s["fresh_B_output"] for s in json.loads((G.REF_RUN / "llava/final_gpu_regression.json").read_text())["samples"]}
    P.exclusive()
    torch.set_grad_enabled(False)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)
    exp = P.Experiment(run, "llava")
    gate = {"status": "RUNNING", "atol": ATOL, "rtol": RTOL, "vectors": {a: d["layer_budget_sha256"] for a, d in docs.items()}, "samples": [],
            "note": "v1 gpu_case re-run with docs U and PYR(=F-front ratios) for G1/G2/G4; packed_case for G6/G7"}
    base = run / "llava" / "gpu_gates"
    try:
        for i, s in enumerate(samples):
            path = base / f"{i:02d}_{s['image_id']}"
            path.mkdir(parents=True, exist_ok=False)
            v1 = G.gpu_case(exp, s, path / "v1", {"U": docs["U"], "F-front": docs["PYR"]}, ref_out[s["image_id"]])
            exp.close_contexts()
            v2 = packed_case(exp, G, S, s, path / "v2", docs)
            exp.close_contexts()
            gate["samples"].append({"image_id": s["image_id"], "v1_G1_G2_G4": v1, "G6_G7": v2})
            write(run / "gates" / "gpu_gates.json", gate)
            # run-local gate stores (2.1 GB/case) are deleted once the case's receipts are recorded
            # (freeze_amendment_gate_store_cleanup.json); the frozen 106 capacity reserve needs >= 46 GiB free.
            from mmimpress.recompute_kv25 import guarded_rmtree
            guarded_rmtree(path, base)
            gate["samples"][-1]["stores_deleted_after_record"] = True
            write(run / "gates" / "gpu_gates.json", gate)
            print("GPU gates", i + 1, len(samples), "PASS", json.dumps(v2["ttft_ms"]), flush=True)
        gate["status"] = "PASS"
        for g in ("G1", "G2", "G4", "G6", "G7"):
            gate[g] = "PASS"
    except Exception:
        gate["status"] = "FAIL"
        gate["error"] = traceback.format_exc()
        print(gate["error"], flush=True)
        raise
    finally:
        write(run / "gates" / "gpu_gates.json", gate)
        exp.close_contexts()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stage", choices=("freeze", "cpu", "g3", "g9", "gpu"), required=True)
    args = ap.parse_args()
    run = Path(args.run).resolve()
    assert run.parent == ROOT / "runs" and run.name.startswith("layer_budget_kv25_v2_")
    (run / "gates").mkdir(parents=True, exist_ok=True)
    {"freeze": stage_freeze, "cpu": stage_cpu, "g3": stage_g3, "g9": stage_g9, "gpu": stage_gpu}[args.stage](run)


if __name__ == "__main__":
    main()

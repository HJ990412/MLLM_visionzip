#!/usr/bin/env python3
"""Phase 0 of the LLaVA-1.5-7B MT-GQA HOLD (recompute / U100 / U50 / U25 / PYR-b20 packed).

  --stage vectors : create the run root, the four frozen budget vectors (U100 / U50 / U25 are
                    uniform; PYR-b20 copies the PyramidKV run's PYR-beta-raw ratios, nothing is
                    recomputed), the N = 576 row / byte table, the model + processor record
                    (CPU only: config, processor, tokenizer; no weights loaded) and the dataset identity
  --stage freeze  : (after docs/llava15_mt_gqa_contract.md exists) measured storage plan,
                    protection "before" inventory, existing-source check, manifest.json,
                    config.json (contract / source / vector SHA256), frozen source copy
  --stage refreeze: after a recorded freeze_amendment_*.json; superseded records are preserved

No data is used to build or select anything.  Nothing outside the new run / results roots is written.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import llava15_support as S
from llava15_support import ROOT, read, sha, write

QWEN_SWEEP_RUN = ROOT / "runs/qwen_budget_sweep_20261111T152143Z"      # read-only: last protection "after" inventory of the repo


def utc():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def model_record():
    """Pinned snapshot, config, processor and tokenizer facts (no weights, no GPU)."""
    import bitsandbytes
    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor
    from mmimpress.llava15.runner import snapshot_path
    snap = snapshot_path()
    assert snap.name == S.REVISION and (snap / "config.json").is_file(), "pinned LLaVA-1.5 snapshot missing"
    ref = snap.parents[1] / "refs" / "main"
    cfg = AutoConfig.from_pretrained(str(snap))
    proc = AutoProcessor.from_pretrained(str(snap))
    ip = proc.image_processor
    vc, tc = cfg.vision_config, cfg.text_config
    patches = (vc.image_size // vc.patch_size) ** 2
    assert type(cfg).__name__ == "LlavaConfig" and cfg.vision_feature_layer == -2 and cfg.vision_feature_select_strategy == "default"
    assert patches == S.N_VISUAL == 576 and tc.num_hidden_layers == 32 and tc.num_attention_heads == 32 == tc.num_key_value_heads
    assert tc.hidden_size // tc.num_attention_heads == 128
    # the processor expands one <image> into exactly 576 tokens (CLS excluded by the "default" strategy)
    from PIL import Image
    probe = proc(images=Image.new("RGB", (640, 480), (127, 127, 127)), text="USER: <image>\nx ASSISTANT:", return_tensors="pt")
    n_image = int((probe["input_ids"][0] == cfg.image_token_index).sum())
    assert n_image == 576 and tuple(probe["pixel_values"].shape) == (1, 3, 336, 336) and "image_sizes" not in probe
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p) if p.stat().st_size < (64 << 20) else None, "resolved": str(p.resolve())}
             for p in sorted(snap.iterdir()) if p.is_file()}
    return {"model_id": S.MODEL_ID, "revision": S.REVISION, "revision_source": "local HF cache snapshot directory; refs/main = " + (ref.read_text().strip() if ref.exists() else "absent"),
            "snapshot": str(snap), "snapshot_files": files, "model_class": "LlavaForConditionalGeneration",
            "architecture_config": cfg.to_dict().get("architectures"),
            "versions": {"transformers": transformers.__version__, "bitsandbytes": bitsandbytes.__version__, "torch": torch.__version__, "python": sys.version},
            "loading": {"quantization": "bitsandbytes NF4, double quant", "bnb_4bit_compute_dtype": "bfloat16", "dtype": "bfloat16",
                        "attn_implementation": "eager", "device_map": "cuda:0"},
            "processor_class": type(proc).__name__, "image_processor_class": type(ip).__name__, "image_processor_is_fast": bool(getattr(ip, "is_fast", False)),
            "image_processor_config": ip.to_dict(),
            "preprocessing_summary": "HF default, unchanged: convert RGB, resize shortest edge 336 (bicubic, resample=3), centre crop 336x336, rescale 1/255, "
                                     "normalize with the CLIP mean/std; the original LLaVA-1.5 repository pads to a square instead",
            "processor_attributes": {k: getattr(proc, k, None) for k in ("patch_size", "vision_feature_select_strategy", "num_additional_image_tokens", "image_token")},
            "tokenizer_class": type(proc.tokenizer).__name__,
            "vision_feature_layer": cfg.vision_feature_layer, "vision_feature_select_strategy": cfg.vision_feature_select_strategy,
            "image_token_index": int(cfg.image_token_index),
            "visual_tokens": {"per_image": n_image, "patch_grid": [vc.image_size // vc.patch_size] * 2, "patches": patches, "cls_token": "excluded (default strategy drops index 0)",
                              "separators": 0, "contiguous_span": True, "pixel_values_shape": list(probe["pixel_values"].shape)},
            "vision": {"model_type": vc.model_type, "image_size": vc.image_size, "patch_size": vc.patch_size, "num_hidden_layers": vc.num_hidden_layers,
                       "num_attention_heads": vc.num_attention_heads, "saliency_layer": "encoder.layers[-2] (index 22): CLS->patch attention summed over heads"},
            "text": {"backbone": "lmsys/vicuna-7b-v1.5", "num_hidden_layers": tc.num_hidden_layers, "num_attention_heads": tc.num_attention_heads,
                     "num_key_value_heads": tc.num_key_value_heads, "head_dim": tc.hidden_size // tc.num_attention_heads, "hidden_size": tc.hidden_size},
            "prompt_template_single_turn": "USER: <image>\n{question} Answer the question using a single word or phrase. ASSISTANT:",
            "prompt_template_hold": "USER: <image>\n[Q1: ..\nA1: ..\n..\n\n]Current question Q{t}: {question}\n" + S.SHORT_ANSWER + " ASSISTANT:",
            "generation": {"greedy": True, "max_new_tokens": 16, "batch_size": 1, "seed": 1234}, "store_dtype": "float16"}


def stage_vectors(run: Path):
    from mmimpress.layer_budget import budget_sha256
    from mmimpress.llava15.store import keep_counts, make_budget
    from mmimpress.mt_gqa import workload_sha256
    assert run.parent == ROOT / "runs" and run.name.startswith("llava15_mt_gqa_")
    run.mkdir(parents=True, exist_ok=False)
    (run / "vectors").mkdir()
    (run / "phase0").mkdir()
    # ---- PYR-b20: copy the PyramidKV run's PYR-beta-raw document byte for byte, take its ratios, compute nothing
    assert sha(S.PYR_SOURCE) == S.PYR_SOURCE_FILE_SHA, "PyramidKV run PYR-beta-raw vector file changed"
    shutil.copy2(S.PYR_SOURCE, run / "vectors" / "source_layer_budget_llava_PYR-beta-raw.json")
    assert sha(run / "vectors" / "source_layer_budget_llava_PYR-beta-raw.json") == S.PYR_SOURCE_FILE_SHA
    src = read(run / "vectors" / "source_layer_budget_llava_PYR-beta-raw.json")
    assert src["layer_budget_sha256"] == S.RATIO_SHA["PYR-b20"] and src["num_layers"] == 32 and src["total_ratio"] == 0.25 and src["r_min"] == 0.0
    ratios = {"U100": [1.0] * 32, "U50": [0.5] * 32, "U25": [0.25] * 32, "PYR-b20": [float(x) for x in src["layer_keep_ratios"]]}
    sources = {v: {"kind": f"uniform {S.TOTAL_RATIO[v]:g} for every decoder layer", "integer_fraction": list(S.FRACTION[v])} for v in ("U100", "U50", "U25")}
    sources["PYR-b20"] = {"kind": "PyramidKV beta schedule (window 0, beta=20), formula as published, no lower bound; layer allocation only",
                          "copied_from": str(S.PYR_SOURCE.relative_to(ROOT)), "copied_file_sha256": S.PYR_SOURCE_FILE_SHA,
                          "copied_alloc_id": src["alloc_id"], "copied_source": src["source"],
                          "copied_clip_range": [src["r_min"], src["r_max"]],
                          "clip_range_note": "the copied document was validated with r_max 0.6; this port passes r_min=0.0, r_max=1.0 explicitly. "
                                             "max ratio is 0.4875 < 0.6, so k_l is identical under both (checked below at N = 576).",
                          "token_selection": "none from PyramidKV: rows are the first k_l of the same image-only VisionZip order as every other arm"}
    recorded = {}
    for vec in S.VECTORS:
        doc = make_budget(vec, ratios[vec], S.TOTAL_RATIO[vec], sources[vec])
        digest = budget_sha256(ratios[vec])
        if doc["layer_budget_sha256"] != digest or digest != S.RATIO_SHA[vec]:
            raise RuntimeError(f"BLOCKED: ratio SHA256 mismatch for {vec}: {digest}")
        write(S.vector_path(run, vec), doc, exclusive=True)
        recorded[vec] = {"layer_budget_sha256": digest, "file_sha256": sha(S.vector_path(run, vec)), "file": str(S.vector_path(run, vec).relative_to(ROOT)),
                         "total_ratio": doc["total_ratio"], "integer_fraction": list(S.FRACTION[vec]), "sum": sum(ratios[vec]),
                         "num_layers": doc["num_layers"], "r_min": doc["r_min"], "r_max": doc["r_max"], "sha_source": "prompt (fixed before the run)"}
    write(run / "vectors" / "vectors_sha256.json", recorded, exclusive=True)
    docs = S.load_vectors(run)

    # ---- rows and bytes at N = 576 (library with explicit arguments vs independent integer Hamilton vs prompt table)
    from mmimpress.layer_budget import layer_keep_counts
    table = {}
    for vec in S.VECTORS:
        counts = keep_counts(docs[vec], 576)
        p, q = S.FRACTION[vec]
        assert counts == S.independent_counts(ratios[vec], 576, S.FRACTION[vec]), vec
        assert sum(counts) == 32 * (-(-p * 576 // q)), vec
        if vec in S.EXPECTED_ROWS:
            assert counts == S.EXPECTED_ROWS[vec], vec
        else:
            assert counts[:3] == S.EXPECTED_PYR_HEAD and counts[-4:] == S.EXPECTED_PYR_TAIL and sum(counts) == S.EXPECTED_PYR_SUM, counts
            assert counts == layer_keep_counts(ratios[vec], 576, num_layers=32, total_ratio=0.25, r_min=src["r_min"], r_max=src["r_max"]), "copied clip range gives other rows"
            assert counts == sorted(counts, reverse=True) and min(counts) >= 1
        assert S.expected_bytes(counts) == S.EXPECTED_BYTES[vec], vec
        table[vec] = {"k_per_layer": counts, "sum": sum(counts), "payload_bytes": S.expected_bytes(counts), "payload_mb": S.expected_bytes(counts) / 1e6,
                      "separator_bytes": 0, "min": min(counts), "max": max(counts)}
    write(run / "phase0" / "k_table_N576.json", {"status": "PASS", "N": 576, "row_bytes_per_layer_per_kind": S.ROW_BYTES, "vectors": table,
                                                   "checks": "library (explicit num_layers/total_ratio/r_min/r_max) == independent integer Hamilton == prompt table; bytes == prompt bytes"})

    # ---- dataset identity (frozen MT-GQA manifest; same smoke dialogue ids as the LLaVA-NeXT HOLD)
    prior = read(S.PRIOR / "manifest.json")
    nxt = read(S.NEXT_HOLD / "manifest.json")
    assert sha(ROOT / "data/mt_gqa/dialogues.json") == S.INDEX_SHA == nxt["index_sha256"]
    assert workload_sha256(prior["dialogues"]) == S.WORKLOAD_SHA == nxt["workload_sha256"]
    assert read(ROOT / "data/mt_gqa/dialogues.json")["dialogues"] == prior["dialogues"] == nxt["dialogues"]
    assert prior["smoke_dialogue_ids"] == nxt["smoke_dialogue_ids"] and prior["image_identity"] == nxt["image_identity"]
    images = []
    for d in prior["dialogues"]:
        if d["image_id"] not in images:
            images.append(d["image_id"])
    assert len(images) == 398 and len(prior["dialogues"]) == 4061
    first = {}
    for d in prior["dialogues"]:
        first.setdefault(d["image_id"], d["dialog_id"])
    fixture = [{"image_id": i, "dialog_id": first[i]} for i in images[:S.FIXTURE_IMAGES]]
    write(run / "phase0" / "dataset_identity.json", {"dataset": "MT-GQA-reconstructed", "index_sha256": S.INDEX_SHA, "workload_sha256": S.WORKLOAD_SHA,
                                                      "images": 398, "dialogues": 4061, "turns": 12183, "smoke_dialogue_ids": prior["smoke_dialogue_ids"],
                                                      "smoke_ids_equal_llava_next_hold": True, "prior_manifest": str((S.PRIOR / "manifest.json").relative_to(ROOT)),
                                                      "prior_manifest_sha256": sha(S.PRIOR / "manifest.json"),
                                                      "llava_next_hold_manifest_sha256": sha(S.NEXT_HOLD / "manifest.json"),
                                                      "gate_fixture": fixture,
                                                      "gate_fixture_rule": "first 10 images of the manifest in dialogue order, first dialogue of each (correctness fixture, not a quality holdout)"})
    write(run / "phase0" / "model_record.json", model_record())
    free = shutil.disk_usage(ROOT).free
    write(run / "phase0" / "disk_free_at_vectors.json", {"free_bytes": free, "free_gib": free / S.GiB, "measured_at_unix": time.time(),
                                                           "df": subprocess.check_output(["df", "-B1", str(ROOT)], text=True)})
    print(json.dumps({"vectors": "PASS", "run": str(run), "ratio_sha256": {v: recorded[v]["layer_budget_sha256"] for v in S.VECTORS},
                      "rows_N576": {v: [table[v]["k_per_layer"][:3], table[v]["k_per_layer"][-4:], table[v]["sum"]] for v in S.VECTORS},
                      "bytes": {v: table[v]["payload_bytes"] for v in S.VECTORS}, "free_gib": round(free / S.GiB, 1)}), flush=True)


def protection_before(run: Path):
    from mmimpress.recompute_kv25 import sha_manifest, stat_inventory
    t0 = time.time()
    excl = [f"results/{run.name}"] + [p for p in S.NEW_PATHS] + [f"results/{p.name}" for p in (ROOT / "results").glob("llava15_mt_gqa_*")]
    shas = {k: v for k, v in sha_manifest([ROOT / x for x in ["results", "docs", "mmimpress", "scripts", "tests", "README.md"]], ROOT, exclude=excl).items()
            if "__pycache__" not in k}
    kv = stat_inventory(sorted(ROOT.glob("kvstore*")), ROOT)
    runs = stat_inventory(sorted(p for p in (ROOT / "runs").iterdir() if not p.name.startswith("llava15_mt_gqa_")), ROOT)
    write(run / "protection_before.json", {"stage": "before", "utc": time.time(), "elapsed_s": time.time() - t0, "excluded_new_paths": excl,
                                           "sha256_files": len(shas), "kvstore_inventory_files": len(kv), "runs_inventory_files": len(runs),
                                           "sha256": shas, "kvstore_inventory": kv, "runs_inventory": runs})
    return {"sha256_files": len(shas), "kvstore_inventory_files": len(kv), "runs_inventory_files": len(runs), "elapsed_s": round(time.time() - t0, 1)}, shas


def existing_source_check(shas):
    """Existing sources vs the last recorded protection inventory of the repository (Qwen sweep run, 'after')."""
    ref = read(QWEN_SWEEP_RUN / "protection_after.json")["sha256"]
    roots = ("mmimpress/", "scripts/", "tests/", "docs/")
    cur = {k: v["sha256"] for k, v in shas.items() if k.startswith(roots)}
    old = {k: v["sha256"] for k, v in ref.items() if k.startswith(roots) and "__pycache__" not in k}
    # the previous run's inventory excludes that run's own new files; those are pinned by its frozen config instead
    qcfg = read(QWEN_SWEEP_RUN / "config.json")["source_sha256"]
    qdoc = "docs/qwen_budget_sweep_contract.md"
    old.update({k: v for k, v in qcfg.items() if k.startswith(roots) and k not in old})
    if qdoc not in old:
        old[qdoc] = read(QWEN_SWEEP_RUN / "config.json")["contract_sha256"]
    changed = sorted(k for k in old if k in cur and cur[k] != old[k])
    missing = sorted(k for k in old if k not in cur)
    added = sorted(k for k in cur if k not in old)
    # files the previous run never pinned (its own report script): accepted only if last modified before this work created its first file
    boundary = min((ROOT / p).stat().st_mtime for p in S.NEW_PATHS)
    unpinned = [{"file": k, "sha256": cur[k], "mtime_unix": (ROOT / k).stat().st_mtime, "modified_before_this_work": (ROOT / k).stat().st_mtime < boundary} for k in added]
    added = [u["file"] for u in unpinned if not u["modified_before_this_work"]]
    return {"unpinned_by_previous_run_but_older_than_this_work": unpinned, "this_work_first_new_file_mtime_unix": boundary, "reference":str((QWEN_SWEEP_RUN / "protection_after.json").relative_to(ROOT)) + " + " + str((QWEN_SWEEP_RUN / "config.json").relative_to(ROOT)),
            "reference_note": "full SHA256 of every file under mmimpress/ scripts/ tests/ docs/ recorded at the end of the previous run (2026-11-12); "
                              "that run's own new files are taken from its frozen config (source_sha256, contract_sha256)",
            "checked_files": len(old), "changed_files": changed, "missing_files": missing, "unlisted_existing_files": added,
            "new_files_of_this_run": S.NEW_PATHS, "status": "PASS" if not changed and not missing and not added else "FAIL"}


def build_config(run: Path, est_hours):
    import main_b_support as support
    docs = S.load_vectors(run)
    vec = read(run / "vectors" / "vectors_sha256.json")
    for v in S.VECTORS:
        assert sha(S.vector_path(run, v)) == vec[v]["file_sha256"]
    model = read(run / "phase0" / "model_record.json")
    return {"schema_version": S.SCHEMA, "experiment": S.EXPERIMENT, "run": str(run.relative_to(ROOT)),
            "contract": str(S.CONTRACT.relative_to(ROOT)), "contract_sha256": sha(S.CONTRACT), "b_main_contract_sha256": sha(S.B_CONTRACT),
            "model": S.MODEL_ID, "revision": S.REVISION, "model_record_sha256": sha(run / "phase0" / "model_record.json"),
            "versions": model["versions"], "methods": list(S.METHODS), "primary_operating_point": S.U25,
            "labels": dict(S.LABEL), "seed": 1234, "greedy": True, "max_new_tokens": 16, "batch_size": 1, "cpu_threads": 2,
            "history_policy": "method_local_generated", "provisioning": "FRESH_IMAGE_STREAMING",
            "store_format": "llava15_prefixstore_ratio_packed_v1", "store_dtype": "float16",
            "read_syscall": "os.preadv into pinned host buffer, one call per hit",
            "layer_budget_call_arguments": {"num_layers": 32, "r_min": S.R_MIN, "r_max": S.R_MAX, "total_ratio": dict(S.TOTAL_RATIO)},
            "layer_budget_module_defaults_not_used": {"R_MIN": 0.10, "R_MAX": 0.60, "TOTAL_RATIO": 0.25},
            "layer_budgets": {v: dict(docs[v], path=str(S.vector_path(run, v))) for v in S.VECTORS},
            "vectors_file_sha256": {v: vec[v]["file_sha256"] for v in S.VECTORS},
            "expected_rows_N576": read(run / "phase0" / "k_table_N576.json")["vectors"],
            "source_sha256": support.source_hashes(ROOT),
            "source_tests_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "tests").glob("test_*.py"))},
            "manifest_sha256": sha(run / "manifest.json"), "storage_plan_sha256": sha(run / "storage_plan.json"),
            "hold": {"images": 398, "dialogues": 4061, "requests_per_arm": 12183, "arms": 5, "requests": S.EXPECT["main"],
                     "smoke_requests": S.EXPECT["smoke"], "integration_requests": S.EXPECT["integration"]},
            "source_T1_capture": "first frozen dialogue per image per stored method; all dialogues/methods still actual full-pixel T1",
            "timing_boundary": "true TTFT as in the LLaVA-NeXT HOLD (164): request start before prompt construction; pixel requests include image decode + HF processor + vision; "
                               "hits include tokenization, dense cache allocation, one preadv, scatter + masks, suffix prefill; posix_fadvise(DONTNEED) is before the timer",
            "bootstrap": {"unit": "image", "resamples": 10000, "seed": 1234},
            "decision_rule": {"scope": "HOLD followup (T2-T3)", "delta": "PYR-b20_packed - U25_packed (accuracy, pp)",
                              "GO": "delta >= +1.0 pp and CI excludes 0 and SSD MB <= U25 * 1.02 and TTFT mean <= U25 * 1.05",
                              "PARTIAL": "delta >= +0.5 pp with CI including 0, or GO with exactly one of the MB / TTFT conditions violated",
                              "NO-GO": "otherwise"},
            "kv_size_axis": {"R100": "TTFT(U100_packed) / TTFT(recompute)", "S25": "TTFT(U25_packed) / TTFT(U100_packed)", "S50": "TTFT(U50_packed) / TTFT(U100_packed)",
                             "references": {"Qwen2.5-VL": {"R100": 0.462, "S25": 0.892, "path": "same packed path"},
                                            "LLaVA-NeXT-7B": {"R100": 1.39, "S25": 0.34, "path": "v1 run, split-file path (not directly comparable)"}},
                             "prior_hypothesis": {"R100": [0.462, 1.39], "S25": [0.34, 0.892]}},
            "expected_duration_after_smoke": {"estimate_hours": est_hours, "requests": S.EXPECT["main"],
                                              "basis": "dev rehearsal on the gate fixture (not a measurement): about 0.1-0.2 s model time per request plus the per-request "
                                                       "lifecycle overhead of 164 (exclusive-GPU check, fsynced evidence) and per-image store hashing (about 0.6 GB x 6 passes)"},
            "environment": {"python": sys.executable, "env": S.ENV,
                            "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
                            "read_ahead_kb": Path("/sys/block/nvme0n1/queue/read_ahead_kb").read_text().strip()}}


def write_frozen_sources(run: Path, config, tag="source_frozen"):
    for n in config["source_sha256"]:
        p = run / tag / n
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / n, p)
    shutil.copy2(S.CONTRACT, run / tag / "CONTRACT.md")


def stage_freeze(run: Path, est_hours):
    assert S.CONTRACT.exists(), "write the contract before freezing"
    assert not (run / "config.json").exists(), "already frozen; amendments go to freeze_amendment_*.json + --stage refreeze"
    assert read(run / "phase0" / "k_table_N576.json")["status"] == "PASS"
    for p in S.NEW_PATHS:
        assert (ROOT / p).is_file(), f"declared new file missing: {p}"

    # ---- storage plan (measured)
    free = shutil.disk_usage(ROOT).free
    payload = sum(S.EXPECTED_BYTES.values())
    plan = dict(policy="FRESH_IMAGE_STREAMING: one live image; four stores built from each arm's own T1 -> served -> audited -> committed -> "
                       "run-local scratch deleted by exact SHA256 allowlist",
                per_image_payload_bytes=payload, per_image_payload_mb=payload / 1e6,
                per_image_payload_basis="U100 301,989,888 + U50 150,994,944 + U25 75,497,472 + PYR-b20 75,497,472 B (N = 576, no separator) = 603,979,776 B, plus sys_kv / meta / layout / integrity (< 2 MB)",
                largest_image_bundle_bound_bytes=1 * S.GiB, peak_working_set_bytes=4 * S.GiB, minimum_reserve_bytes=30 * S.GiB,
                maximum_new_payload_bytes=20 * S.GiB, raw_and_evidence_growth_bound_bytes=8 * S.GiB, gate_store_bound_bytes=2 * S.GiB,
                required_initial_free_bytes=64 * S.GiB, free_bytes_at_freeze=free, free_gib_at_freeze=free / S.GiB, measured_at_unix=time.time(),
                df=subprocess.check_output(["df", "-B1", str(ROOT)], text=True),
                mount=subprocess.check_output(["findmnt", "-J", "-T", str(run)], text=True),
                deletion_scope=[f"runs/{run.name}/scratch/** (exact SHA256 allowlist after independent image audit + atomic commit)",
                                f"runs/{run.name}/gates/stores/** (run-local gate fixture stores, removed per image after the receipts are written)"],
                existing_artifacts="never deleted or modified; if space runs out the run stops with BLOCKED_STORAGE")
    plan["status"] = "PASS" if free >= plan["required_initial_free_bytes"] else "BLOCKED_STORAGE"
    write(run / "storage_plan.json", plan, exclusive=True)
    if plan["status"] != "PASS":
        raise RuntimeError(f"BLOCKED_STORAGE: free {free} < required {plan['required_initial_free_bytes']}")

    protection, shas = protection_before(run)
    check = existing_source_check(shas)
    write(run / "source_modifications.json", check)
    if check["status"] != "PASS":
        raise RuntimeError("BLOCKED: existing sources differ from the last protection inventory: " + json.dumps({k: check[k] for k in ("changed_files", "missing_files", "unlisted_existing_files")})[:2000])

    prior = read(S.PRIOR / "manifest.json")
    ident = read(run / "phase0" / "dataset_identity.json")
    assert ident["prior_manifest_sha256"] == sha(S.PRIOR / "manifest.json")
    for iid, x in prior["image_identity"].items():
        assert sha(ROOT / x["image_path"]) == x["sha256"], iid
    manifest = {"dataset": "MT-GQA-reconstructed", "schema_version": S.SCHEMA, "index_sha256": S.INDEX_SHA, "workload_sha256": S.WORKLOAD_SHA,
                "dialogues": prior["dialogues"], "image_identity": prior["image_identity"], "smoke_dialogue_ids": prior["smoke_dialogue_ids"],
                "gate_fixture": ident["gate_fixture"], "methods": list(S.METHODS),
                "expected_requests": {"integration": S.EXPECT["integration"], "smoke": S.EXPECT["smoke"], "main": S.EXPECT["main"]},
                "prior_manifest_sha256": ident["prior_manifest_sha256"], "llava_next_hold_manifest_sha256": ident["llava_next_hold_manifest_sha256"],
                "smoke_policy": "prior frozen smoke dialogue ids (first four images); identical to the LLaVA-NeXT HOLD manifests",
                "fixture_note": "gate fixture and smoke dialogues are correctness fixtures, not a quality holdout"}
    assert len(manifest["smoke_dialogue_ids"]) * 3 * 5 == S.EXPECT["smoke"]
    write(run / "manifest.json", manifest, exclusive=True)
    config = build_config(run, est_hours)
    config["disk_free_bytes_at_freeze"] = free
    write(run / "config.json", config, exclusive=True)
    write_frozen_sources(run, config)
    shutil.copy2(S.CONTRACT, run / "CONTRACT.md")
    import torch
    write(run / "environment_before.json", {"python": sys.version, "torch": torch.__version__, "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
                                            "pip_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)})
    write(run / "final_source_freeze.json", dict(contract_sha256=config["contract_sha256"], config_sha256=sha(run / "config.json"),
                                                 manifest_sha256=sha(run / "manifest.json"), files=len(config["source_sha256"]),
                                                 existing_sources_changed=False, amendments=[]))
    (ROOT / "results" / run.name).mkdir(parents=True, exist_ok=True)
    print(json.dumps({"freeze": "PASS", "run": str(run), "contract_sha256": config["contract_sha256"], "config_sha256": sha(run / "config.json"),
                      "free_gib": round(free / S.GiB, 2), "required_gib": 64, "protection": protection, "existing_source_check": check["status"],
                      "sources": len(config["source_sha256"])}), flush=True)


def stage_refreeze(run: Path, est_hours):
    """Re-freeze after a recorded amendment.  The superseded config / frozen sources / gate records are kept."""
    amendments = sorted(run.glob("freeze_amendment_*.json"))
    assert amendments, "record freeze_amendment_<n>.json (cause, fix, affected files) before re-freezing"
    old = read(run / "config.json")
    n = len(list(run.glob("config.superseded_*.json"))) + 1
    assert len(amendments) >= n, "every re-freeze needs its own amendment record"
    assert not (run / "main").exists(), "main already started; a re-freeze would invalidate committed images"
    shutil.move(str(run / "config.json"), str(run / f"config.superseded_{n}.json"))
    if (run / "source_frozen").exists():
        shutil.move(str(run / "source_frozen"), str(run / f"source_frozen.superseded_{n}"))
    for name in ("gates", "integration", "smoke"):
        if (run / name).exists():
            shutil.move(str(run / name), str(run / f"{name}.superseded_{n}"))
    for p in list(run.glob("*_progress.json")) + list(run.glob("*_execution_complete.json")) + list(run.glob("CONTRACT.md")):
        shutil.move(str(p), str(p.with_name(p.name + f".superseded_{n}")))
    if (run / "scratch").exists():
        leftovers = [p for p in (run / "scratch").rglob("*") if p.is_file()]
        assert not leftovers, "scratch must be empty before a re-freeze (failed attempts are cleaned by their own allowlist or inspected first)"
    config = build_config(run, est_hours if est_hours else old["expected_duration_after_smoke"]["estimate_hours"])
    config["disk_free_bytes_at_freeze"] = shutil.disk_usage(ROOT).free
    config["refreeze"] = {"index": n, "amendments": [p.name for p in amendments], "amendment_sha256": {p.name: sha(p) for p in amendments},
                          "superseded_config_sha256": sha(run / f"config.superseded_{n}.json"),
                          "changed_sources": sorted(k for k in set(old["source_sha256"]) | set(config["source_sha256"])
                                                    if old["source_sha256"].get(k) != config["source_sha256"].get(k)),
                          "contract_changed": old["contract_sha256"] != config["contract_sha256"]}
    illegal = [k for k in config["refreeze"]["changed_sources"] if k not in S.NEW_PATHS]
    assert not illegal, f"an existing source changed: {illegal}"
    write(run / "config.json", config, exclusive=True)
    write_frozen_sources(run, config)
    shutil.copy2(S.CONTRACT, run / "CONTRACT.md")
    write(run / "final_source_freeze.json", dict(contract_sha256=config["contract_sha256"], config_sha256=sha(run / "config.json"),
                                                 manifest_sha256=sha(run / "manifest.json"), files=len(config["source_sha256"]),
                                                 existing_sources_changed=False, amendments=[p.name for p in amendments]))
    print(json.dumps({"refreeze": "PASS", "index": n, "config_sha256": sha(run / "config.json"), "changed_sources": config["refreeze"]["changed_sources"],
                      "contract_changed": config["refreeze"]["contract_changed"]}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["vectors", "freeze", "refreeze"], required=True)
    ap.add_argument("--run", type=Path)
    ap.add_argument("--est-hours", type=float, nargs=2, default=None, help="expected main duration range (hours), recorded in config.json")
    a = ap.parse_args()
    if a.stage == "vectors":
        run = a.run.resolve() if a.run else ROOT / "runs" / f"{S.EXPERIMENT}_{utc()}"
        stage_vectors(run)
    elif a.stage == "freeze":
        assert a.est_hours, "--est-hours LOW HIGH is required at freeze"
        stage_freeze(a.run.resolve(), list(a.est_hours))
    else:
        stage_refreeze(a.run.resolve(), list(a.est_hours) if a.est_hours else None)


if __name__ == "__main__":
    main()

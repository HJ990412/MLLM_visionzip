#!/usr/bin/env python3
"""Independent, CPU-only audit of the Qwen KV25 four-arm pilot.

Reads frozen manifests, protected store metadata, per-request raw JSONL and
optional summary.csv. Does not import the pilot runner or report generator.
The output is a fresh JSON receipt; input artifacts are read-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "runs/qwen25_kv25_migration_20260929T042949Z"
V2 = ROOT / "runs/qwen25_correctness_v2_20260928T081111Z"
BASELINE_SHA = "13c2ba95cf9047fc8a5f300c6b4e18221cd13d5fd5f0af88767a9d0becbfb0b1"
SCHEMA = "qwen25-kv25-four-arm-pilot-v1"
METHODS = ("recompute", "fullload", "qwen_ours_chunk25_legacy", "qwen_ours_kv25")
PHASES = ("smoke", "gqa", "mt")
SUMMARY_METRICS = (
    "requests", "hit_requests", "all_quality", "hit_quality", "t1_ttft_ms",
    "hit_ttft_ms", "hit_e2e_ms", "logical_content_retention_macro",
    "logical_content_retention_weighted", "structural_inclusive_retention_macro",
    "normal_read_mb_per_hit", "structural_read_mb_per_hit",
    "metadata_read_mb_per_hit", "total_read_mb_per_hit",
    "valid_content_read_mb_per_hit", "retained_content_kv_mb_per_hit",
    "extra_valid_mb_per_hit",
    "padding_read_mb_per_hit", "h2d_mb_per_hit", "gpu_compact_cache_mb_per_hit",
    "peak_gpu_allocated_mb_per_hit", "normal_read_vs_fullload",
    "total_read_vs_fullload",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def rows_jsonl(path: Path) -> list[dict]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            result.append(row)
    return result


def norm(value) -> str:
    words = re.sub(r"[^\w\s]", " ", str(value).lower()).split()
    return " ".join(w for w in words if w not in {"a", "an", "the"})


def quality(phase: str, prediction: str, gold: str) -> float:
    p, g = norm(prediction), norm(gold)
    if phase == "mt":
        return float(p == g)
    return float(p == g or bool(g) and p.split()[:len(g.split())] == g.split())


def avg(values):
    values = list(values)
    return statistics.fmean(values) if values else None


def as_mb(value):
    return value / 1_000_000 if value is not None else None


def finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def near(a, b, *, atol=1e-7, rtol=1e-7) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol)


class Checks:
    def __init__(self):
        self.count = 0
        self.errors = []

    def want(self, condition, label):
        self.count += 1
        if not condition:
            self.errors.append(label)


def baseline_hashes(required: set[str], checks: Checks) -> dict[str, str]:
    path = MIGRATION / "protected_before.jsonl"
    checks.want(digest(path) == BASELINE_SHA, "protected baseline manifest hash changed")
    found = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("path") in required:
                found[row["path"]] = row.get("sha256")
    checks.want(set(found) == required, "protected baseline lacks requested store metadata")
    return found


def validate_binding(phase: str, manifest: dict, config: dict,
                     inventory: dict, directory: Path, gate: dict,
                     checks: Checks) -> dict[str, dict]:
    ctx = phase + ": "
    checks.want(manifest.get("schema_version") == SCHEMA and
                manifest.get("phase") == phase and config.get("dataset") == phase and
                config.get("schema_version") == SCHEMA and
                inventory.get("schema_version") == SCHEMA, ctx+"schema/phase mismatch")
    unsigned = {k:v for k,v in manifest.items() if k != "manifest_sha256"}
    mhash = canonical(unsigned)
    checks.want(mhash == manifest.get("manifest_sha256") == config.get("manifest_sha256"),
                ctx+"manifest canonical hash mismatch")
    checks.want(config.get("model_id") == manifest.get("model_id") ==
                "Qwen/Qwen2.5-VL-7B-Instruct" and
                config.get("model_revision") == manifest.get("model_revision") ==
                "cc594898137f460bfe9f0759e9844b3ce807cfb5" and
                config.get("attention_backend") == "sdpa" and
                config.get("weight_quantization") == "NF4" and
                config.get("compute_dtype") == config.get("ssd_kv_dtype") == "bfloat16" and
                config.get("seed") == manifest.get("seed") == 1234 and
                config.get("max_new_tokens") == manifest.get("max_new_tokens") == 16 and
                config.get("min_pixels") == 200704 and config.get("max_pixels") == 802816 and
                config.get("chunk_size") == manifest.get("chunk_size") == 64 and
                config.get("ratio") == manifest.get("budget_ratio") == .25,
                ctx+"frozen model/runtime configuration changed")
    checks.want(manifest.get("methods") == list(METHODS) and
                manifest.get("history_policy") ==
                ("method_own_generated_answers" if phase == "mt" else
                 "none_independent_questions") and
                manifest.get("storage_policy") == "read_only_v2_stores" and
                manifest.get("persistence_status") == config.get("persistence_status") ==
                "NOT_REMEASURED" and config.get("execution_scope") ==
                "CACHE-HIT REEVALUATION" and inventory.get("store_mode") == "read_only",
                ctx+"method/history/storage policy changed")
    gate_sha = digest(Path(gate["path"]))
    checks.want(gate.get("status") == "PASS" and gate.get("pilot_eligible") is True and
                manifest.get("gpu_gate",{}).get("sha256") == gate_sha and
                config.get("gpu_gate",{}).get("sha256") == gate_sha,
                ctx+"GPU correctness gate binding mismatch")
    checks.want(inventory.get("protected_manifest_sha256") == BASELINE_SHA and
                inventory.get("geometry_reference_sha256") ==
                digest(MIGRATION / "geometry_reference.json") and
                config.get("storage_plan_sha256") ==
                digest(MIGRATION / "storage_plan.json"),
                ctx+"protected inventory/geometry/storage-plan binding mismatch")
    src_hashes = config.get("source_hashes",{})
    required_sources = {"mmimpress/qwen25/runner.py", "mmimpress/qwen25/store.py",
        "mmimpress/qwen25/vision.py", "docs/qwen25_kv25_budget_contract.md",
        "scripts/94_eval_qwen_kv25_pilot.py"}
    checks.want(required_sources.issubset(src_hashes), ctx+"source hashes missing")
    for rel, expected in src_hashes.items():
        safe = not Path(rel).is_absolute() and ".." not in Path(rel).parts
        checks.want(safe, ctx+f"unsafe source path {rel}")
        if safe:
            checks.want(digest(ROOT / rel) == expected,
                        ctx+f"source drift {rel}")
    frozen = manifest.get("source_frozen_manifest",{})
    original_path = Path(frozen.get("path",""))
    expected_source = ROOT / "runs/qwen25_port_20260928T054537Z" / ("mt_manifest" if phase == "mt" else "gqa_manifest") / "manifest.json"
    checks.want(original_path.is_file() and
                original_path.resolve() == expected_source.resolve(),
                ctx+"original frozen manifest path invalid")
    checks.want(digest(Path(frozen["source_index"])) == frozen.get("source_index_sha256"),
                ctx+"original source index changed")
    original = read_json(original_path)
    checks.want(digest(original_path) == frozen.get("file_sha256") and
                original.get("manifest_sha256") == frozen.get("content_sha256") ==
                manifest.get("source_frozen_content_sha256"),
                ctx+"source frozen manifest changed")
    source_images = original.get("images", [])[:4] if phase == "smoke" else original.get("images", [])
    requested = manifest.get("images", [])
    checks.want(len(requested) == (4 if phase == "smoke" else 40) and
                len(source_images) == len(requested), ctx+"frozen image count mismatch")
    image_map = {}
    for index,(image,old) in enumerate(zip(requested,source_images)):
        iid = str(image.get("image_id"))
        turns = old.get("turns", [])[:3] if phase == "smoke" else old.get("turns", [])
        old_body = {k:v for k,v in old.items() if k not in ("method_order","turns")}
        current_body = {k:v for k,v in image.items() if k not in ("method_order","turns")}
        expected_order = list(METHODS[(index+1234)%4:] + METHODS[:(index+1234)%4])
        checks.want(current_body == old_body and image.get("turns") == turns and
                    image.get("method_order") == expected_order,
                    ctx+f"frozen source workload or deterministic rotation changed {iid}")
        checks.want(len(image.get("turns",[])) == (6 if phase == "gqa" else 3),
                    ctx+f"turn count mismatch {iid}")
        checks.want(iid not in image_map, ctx+f"duplicate manifest image {iid}")
        image_map[iid] = image
        checks.want(digest(Path(image["image_path"])) == image.get("image_sha256"),
                    ctx+f"source image changed {iid}")
    checks.want(set(inventory.get("stores",{})) == set(image_map),
                ctx+"store inventory image set differs")
    store_root = V2 / ("mt_pilot" if phase == "mt" else "gqa_pilot") / "stores"
    needed_meta = set()
    for iid in image_map:
        for arm in ("ours25","fullload"):
            needed_meta.add(str((store_root / iid / arm / "meta.json").relative_to(ROOT)))
    protected = baseline_hashes(needed_meta,checks)
    geometry_ref = read_json(MIGRATION / "geometry_reference.json")
    ref_phase = "gqa" if phase == "smoke" else phase
    ref_by_id = {str(x["image_id"]):x for x in geometry_ref["phases"][ref_phase]["entries"]}
    meta_by_id = {}
    for iid,image in image_map.items():
        record = inventory["stores"].get(iid,{})
        metas = {}
        for arm in ("ours25","fullload"):
            mpath = store_root / iid / arm / "meta.json"
            rel = str(mpath.relative_to(ROOT))
            actual_hash = digest(mpath)
            checks.want(actual_hash == protected.get(rel) ==
                        record.get("meta_sha256",{}).get(arm),
                        ctx+f"protected {arm} meta changed {iid}")
            metas[arm] = read_json(mpath)
        meta,full = metas["ours25"],metas["fullload"]
        n,s = meta["visual_count"],meta["structural_count"]
        nch=(n+63)//64
        old_chunks=max(1,min(nch,int(round(nch*.25))))
        old_kept=min(n,old_chunks*64)
        k=(n+3)//4
        new_chunks=(k+63)//64
        perm=meta["stored_to_original"]
        perm_hash=hashlib.sha256(",".join(map(str,perm)).encode("ascii")).hexdigest()
        checks.want(n>0 and meta["chunk_size"]==64 and meta["n_chunks"]==nch and
                    meta["stored_rows"]==nch*64 and
                    meta["prefix_len"]==n+s and len(perm)==n and
                    sorted(perm)==list(range(n)) and
                    perm_hash==meta["permutation_sha256"] and
                    full["stored_to_original"]==list(range(n)) and
                    meta["dtype"]==full["dtype"]=="bfloat16" and
                    meta["num_kv_heads"]==full["num_kv_heads"]==4 and
                    meta["num_layers"]==full["num_layers"]==28,
                    ctx+f"native BF16 geometry/permutation mismatch {iid}")
        checks.want(meta["identity"]["image_sha256"]==image["image_sha256"] and
                    full["identity"]["image_sha256"]==image["image_sha256"] and
                    meta["prefix_sha256"]==full["prefix_sha256"] and
                    meta["logical_position_ids"]==full["logical_position_ids"] and
                    meta["structural_indices"]==full["structural_indices"] and
                    meta["key_rope_state"]=="post_mrope" and
                    meta["global_order_all_layers"] is True and
                    meta["score_source"]=="last_fullatt_ViT_received_attention",
                    ctx+f"capture/layout identity mismatch {iid}")
        expected = {"N_content":n,"S_structural":s,"k_target":k,
                    "legacy_kept":old_kept,"legacy_chunks":old_chunks,
                    "kv25_chunks":new_chunks,
                    "selected_original_legacy":sorted(perm[:old_kept]),
                    "selected_original_kv25":sorted(perm[:k]),
                    "selected_stored_kv25":list(range(k)),
                    "permutation_sha256":perm_hash,"row_bytes":meta["row_bytes"],
                    "num_layers":meta["num_layers"],"prefix_len":n+s,
                    "stored_rows":nch*64,
                    "legacy_store":str((store_root/iid/"ours25").resolve()),
                    "fullload_store":str((store_root/iid/"fullload").resolve())}
        for key,value in expected.items():
            checks.want(record.get(key)==value, ctx+f"inventory {key} mismatch {iid}")
        reference=ref_by_id.get(iid,{})
        for key,value in (("N_content",n),("S_structural",s),("kv25_k",k),
                          ("legacy_kept",old_kept),("legacy_chunks",old_chunks),
                          ("kv25_chunks",new_chunks),
                          ("permutation_sha256",perm_hash),
                          ("image_sha256",image["image_sha256"])):
            checks.want(reference.get(key)==value,
                        ctx+f"prospective geometry reference {key} mismatch {iid}")
        meta_by_id[iid]={"image":image,"meta":meta,"full":full,"n":n,"s":s,
                          "nchunks":nch,"old_chunks":old_chunks,"old_kept":old_kept,
                          "k":k,"new_chunks":new_chunks,"perm":perm,
                          "row_bytes":meta["row_bytes"],"layers":meta["num_layers"],
                          "store_root":store_root}
    return meta_by_id


def check_request(row: dict, image: dict, turn: dict, geom: dict,
                  phase: str, config_sha: str, manifest_sha: str,
                  checks: Checks) -> None:
    iid,tid,method=str(image["image_id"]),int(turn["turn_id"]),row["method_id"]
    ctx=f"{phase}/{iid}/T{tid}/{method}: "
    res=row.get("result")
    checks.want(isinstance(res,dict),ctx+"missing result")
    if not isinstance(res,dict):
        return
    path="normal_pixels" if tid==1 or method=="recompute" else "read_only_ssd_cache_hit"
    unit={"recompute":"pixels","fullload":"full",
          "qwen_ours_chunk25_legacy":"chunk","qwen_ours_kv25":"visual_kv"}[method]
    ratio=None if method=="recompute" else 1.0 if method=="fullload" else .25
    checks.want(row.get("schema_version")==SCHEMA and row.get("status")=="PASS" and
                row.get("attempt")==1 and row.get("dataset")==phase and
                row.get("image_id")==iid and row.get("image_sha256")==image["image_sha256"] and
                row.get("dialog_id")==image["dialog_id"] and
                row.get("turn_id")==tid and row.get("question_id")==str(turn["question_id"]) and
                row.get("question")==turn["question"] and row.get("gold")==turn["gold"] and
                row.get("budget_unit")==unit and row.get("ratio")==ratio and
                row.get("chunk_size")==64 and row.get("request_path")==path and
                row.get("manifest_sha256")==manifest_sha and
                row.get("config_sha256")==config_sha,
                ctx+"request/workload/config identity mismatch")
    checks.want(row.get("model_id")=="Qwen/Qwen2.5-VL-7B-Instruct" and
                row.get("model_revision")=="cc594898137f460bfe9f0759e9844b3ce807cfb5" and
                row.get("store_persistence")=="NOT_REMEASURED" and
                row.get("N_content")==geom["n"] and row.get("S_structural")==geom["s"] and
                row.get("k_target")==geom["k"] and
                row.get("legacy_kept_count")==geom["old_kept"] and
                row.get("permutation_sha256")==geom["meta"]["permutation_sha256"],
                ctx+"model/persistence/geometry identity mismatch")
    checks.want(row.get("method_order")==image["method_order"] and
                row.get("method_order_position")==image["method_order"].index(method),
                ctx+"method rotation/order mismatch")
    expected_score=quality(phase,row.get("prediction",""),turn["gold"])
    checks.want(near(row.get("correct"),expected_score,atol=0,rtol=0),
                ctx+"quality score differs from independent normalizer")
    generated=row.get("generated_token_ids")
    checks.want(isinstance(generated,list) and 1<=len(generated)<=16 and
                generated==res.get("generated_token_ids") and
                len(generated)==row.get("generated_token_count")==res.get("generated_token_count") and
                generated[0]==row.get("first_token_id")==res.get("first_token_id") and
                row.get("prediction")==res.get("prediction") and
                res.get("first_logits_finite") is True and
                bool(re.fullmatch(r"[0-9a-f]{64}",res.get("first_logits_sha256",""))),
                ctx+"generated IDs/prediction/finite-logit evidence mismatch")
    ttft,e2e=row.get("ttft_ms"),row.get("request_e2e_ms")
    checks.want(finite_number(ttft) and finite_number(e2e) and
                0<=ttft<=e2e+1e-3 and ttft==res.get("ttft_ms") and
                e2e==res.get("request_e2e_ms") and
                finite_number(row.get("started_unix")) and
                finite_number(row.get("finished_unix")) and
                row["started_unix"]<=row["finished_unix"],
                ctx+"TTFT/E2E/request timestamp invalid")
    decode=row.get("image_file_decode_ms")
    if path=="normal_pixels":
        checks.want(finite_number(decode) and decode>=0 and
                    decode==res.get("image_file_decode_ms")==
                    res.get("timing_ms",{}).get("image_file_decode") and
                    ttft>=decode and e2e>=decode,
                    ctx+"pixel-path image decode missing from timing")
        checks.want(row.get("vision_calls")==res.get("vision_calls")==1 and
                    row.get("online_query_score_calls")==0 and
                    row.get("conditioning") is None and
                    row.get("total_actual_pread_bytes") is None and
                    row.get("selected_original_ids") is None and
                    row.get("logical_content_retention") is None and
                    row.get("structural_inclusive_retention") is None,
                    ctx+"pixel request unexpectedly read SSD or retained selected KV")
        if tid==1:
            expected_capture="none" if method=="recompute" else \
                "kv_only" if method=="fullload" else "with_score"
            checks.want(res.get("t1_diagnostic_capture_mode")==expected_capture and
                        res.get("t1_diagnostic_capture_discarded")==
                        (method!="recompute") and
                        res.get("geometry",{}).get("visual_count")==geom["n"] and
                        res.get("geometry",{}).get("prefix_len")==geom["n"]+geom["s"],
                        ctx+"T1 capture/geometry policy mismatch")
        return
    checks.want(decode is None and res.get("image_file_decode_ms") is None and
                row.get("vision_calls")==res.get("vision_calls")==0 and
                row.get("online_query_score_calls")==0 and
                res.get("dense_reference") is False,
                ctx+"cache hit decoded image/ran vision/used dense cache")
    n,s,rb,layers=geom["n"],geom["s"],geom["row_bytes"],geom["layers"]
    if method=="fullload":
        kept,chunks,selected=n,geom["nchunks"],list(range(n))
        store=str((geom["store_root"]/iid/"fullload").resolve())
    elif method=="qwen_ours_chunk25_legacy":
        kept,chunks=geom["old_kept"],geom["old_chunks"]
        selected=sorted(geom["perm"][:kept])
        store=str((geom["store_root"]/iid/"ours25").resolve())
    else:
        kept,chunks=geom["k"],geom["new_chunks"]
        selected=sorted(geom["perm"][:kept])
        store=str((geom["store_root"]/iid/"ours25").resolve())
    valid=min(n,chunks*64)
    extra=valid-kept
    padding=chunks*64-valid
    unit_bytes=rb*2*layers
    normal_bytes=chunks*64*unit_bytes
    structural_bytes=s*unit_bytes
    all_bytes=normal_bytes+structural_bytes
    io=res.get("read_io",{})
    spans=io.get("span_details",[])
    visual=[x for x in spans if x.get("kind")=="visual"]
    structural=[x for x in spans if x.get("kind")=="structural"]
    checks.want(row.get("source_store")==store and
                row.get("selected_original_ids")==selected and
                row.get("selected_stored_ids")==
                (None if method=="fullload" else list(range(kept))) and
                res.get("selected_visual_original")==selected and
                res.get("selected_visual_stored")==list(range(kept)) and
                res.get("budget_unit")==("visual_kv" if method=="qwen_ours_kv25" else "chunk") and
                row.get("normal_chunks_read")==res.get("normal_chunks_read")==chunks and
                row.get("compact_content_rows")==res.get("kept_tokens")==kept and
                row.get("compact_total_rows")==res.get("compact_prefix_tokens")==kept+s,
                ctx+"stored/selected/compact geometry mismatch")
    checks.want(row.get("loaded_valid_visual_rows")==res.get("loaded_valid_visual_rows")==valid and
                row.get("extra_valid_visual_rows")==res.get("extra_valid_visual_rows")==extra and
                row.get("padding_rows_read")==res.get("padding_rows_read")==padding and
                row.get("padding_read_bytes")==padding*unit_bytes and
                row.get("normal_visual_read_bytes")==res.get("visual_read_bytes")==normal_bytes and
                row.get("structural_read_bytes")==res.get("structural_read_bytes")==structural_bytes and
                row.get("metadata_read_bytes")==res.get("metadata_read_bytes")==0 and
                row.get("total_actual_pread_bytes")==io.get("bytes")==all_bytes and
                row.get("actual_pread_calls")==res.get("pread_calls")==io.get("preads") and
                io.get("preads",0)>=57 and
                row.get("actual_read_spans")==res.get("read_spans")==io.get("spans")==57 and
                io.get("per_kind",{}).get("visual",{}).get("bytes")==normal_bytes and
                io.get("per_kind",{}).get("structural",{}).get("bytes")==structural_bytes and
                io.get("per_kind",{}).get("visual",{}).get("spans")==2*layers and
                io.get("per_kind",{}).get("structural",{}).get("spans")==1 and
                io.get("per_kind",{}).get("visual",{}).get("preads",0)>=2*layers and
                io.get("per_kind",{}).get("structural",{}).get("preads",0)>=1 and
                io.get("per_kind",{}).get("visual",{}).get("preads",0)+
                io.get("per_kind",{}).get("structural",{}).get("preads",0)==io.get("preads"),
                ctx+"actual returned I/O, calls, or padding mismatch")
    checks.want(len(visual)==2*layers and len(structural)==1 and
                len({x.get("source") for x in visual})==2*layers and
                all(x.get("offset")==0 and x.get("requested_bytes")==
                    chunks*64*rb for x in visual) and
                structural[0].get("offset")==0 and
                structural[0].get("requested_bytes")==structural_bytes,
                ctx+"pread spans differ from whole-chunk plan")
    checks.want(row.get("h2d_kv_bytes")==res.get("h2d_kv_bytes")==
                (kept+s)*unit_bytes and
                row.get("gpu_cache_kv_bytes")==res.get("gpu_cache_kv_bytes")==
                (kept+s)*unit_bytes and
                near(row.get("logical_content_retention"),kept/n,atol=1e-12,rtol=0) and
                near(row.get("structural_inclusive_retention"),(kept+s)/(n+s),
                     atol=1e-12,rtol=0),
                ctx+"H2D/GPU bytes or logical retention mismatch")
    cond=row.get("conditioning",{})
    checks.want(isinstance(cond,dict) and
                cond.get("budget_unit")==res.get("budget_unit") and
                cond.get("selected_chunks")==chunks and
                cond.get("target_visual_tokens")==kept and
                row.get("activation_ms_outside_timer")==cond.get("activation_ms") and
                row.get("activation_io_outside_timer")==cond.get("activation_io"),
                ctx+"cache conditioning/activation attribution mismatch")


def phase_audit(directory: Path, gate: dict, checks: Checks) -> tuple[str,dict,list[dict]]:
    directory=directory.resolve()
    manifest=read_json(directory/"manifest.json")
    config=read_json(directory/"config.json")
    inventory=read_json(directory/"store_inventory.json")
    final=read_json(directory/"final_status.json")
    phase=manifest.get("phase")
    if phase not in PHASES:
        raise ValueError(f"unknown phase at {directory}: {phase}")
    images=validate_binding(phase,manifest,config,inventory,directory,gate,checks)
    raw=rows_jsonl(directory/"raw.jsonl")
    expected_count=(48 if phase=="smoke" else 960 if phase=="gqa" else 480)
    checks.want(final.get("status")=="PASS" and
                final.get("expected_requests")==expected_count and
                final.get("actual_requests")==expected_count and
                len(raw)==expected_count,
                f"{phase}: final status or raw count differs from {expected_count}")
    ids=[x.get("physical_execution_id") for x in raw]
    checks.want(all(isinstance(x,str) and bool(re.fullmatch(r"[0-9a-f]{32}",x))
                    for x in ids) and len(ids)==len(set(ids)),
                f"{phase}: missing/duplicate physical execution ID")
    actual_keys=[(str(x.get("image_id")),int(x.get("turn_id",-1)),x.get("method_id")) for x in raw]
    expected_order=[(str(im["image_id"]),int(t["turn_id"]),method)
                    for im in manifest["images"] for t in im["turns"]
                    for method in im["method_order"]]
    checks.want(actual_keys==expected_order and len(actual_keys)==len(set(actual_keys)),
                f"{phase}: missing/duplicate/out-of-order logical request")
    checks.want(all(x.get("status")=="PASS" and x.get("attempt")==1 for x in raw),
                f"{phase}: failed/retried request present")
    per_method=defaultdict(list)
    by_image_method=defaultdict(list)
    histories=defaultdict(list)
    config_sha=digest(directory/"config.json")
    manifest_sha=manifest["manifest_sha256"]
    for index,row in enumerate(raw):
        try:
            iid=str(row["image_id"]);method=row["method_id"];tid=int(row["turn_id"])
            image=images[iid]["image"]
            turn=next(x for x in image["turns"] if int(x["turn_id"])==tid)
            if method not in METHODS:
                raise ValueError("unknown method")
            previous=histories[iid,method] if phase=="mt" else []
            checks.want(row.get("history")==previous and
                        row.get("history_sha256")==canonical(previous),
                        f"{phase}/{iid}/T{tid}/{method}: method-local generated history mismatch")
            check_request(row,image,turn,images[iid],phase,config_sha,manifest_sha,checks)
            if phase=="mt":
                histories[iid,method]=previous+[{"question_id":str(turn["question_id"]),
                    "question":turn["question"],"prediction":row["prediction"]}]
            per_method[method].append(row)
            by_image_method[iid,method].append(row)
        except Exception as exc:
            checks.errors.append(f"{phase}: raw row {index} cannot be audited: {type(exc).__name__}: {exc}")
    summary={}
    for method in METHODS:
        records=per_method[method]
        hits=[r for r in records if int(r["turn_id"])>1]
        t1=[r for r in records if int(r["turn_id"])==1]
        one_hit_per_image={str(r["image_id"]):r for r in hits}
        retained=list(one_hit_per_image.values()) if method!="recompute" else []
        val=lambda field: [r[field] for r in hits if r.get(field) is not None]
        summary[method]={
            "requests":len(records),"hit_requests":len(hits),
            "all_quality":avg(r["correct"] for r in records),
            "hit_quality":avg(r["correct"] for r in hits),
            "t1_ttft_ms":avg(r["ttft_ms"] for r in t1),
            "hit_ttft_ms":avg(r["ttft_ms"] for r in hits),
            "hit_e2e_ms":avg(r["request_e2e_ms"] for r in hits),
            "logical_content_retention_macro":avg(r["logical_content_retention"] for r in retained),
            "logical_content_retention_weighted":
                sum(r["compact_content_rows"] for r in retained)/sum(r["N_content"] for r in retained)
                if retained else None,
            "structural_inclusive_retention_macro":
                avg(r["structural_inclusive_retention"] for r in retained),
            "normal_read_mb_per_hit":as_mb(avg(val("normal_visual_read_bytes"))),
            "structural_read_mb_per_hit":as_mb(avg(val("structural_read_bytes"))),
            "metadata_read_mb_per_hit":as_mb(avg(val("metadata_read_bytes"))),
            "total_read_mb_per_hit":as_mb(avg(val("total_actual_pread_bytes"))),
            "valid_content_read_mb_per_hit":as_mb(avg(
                r["loaded_valid_visual_rows"]*images[str(r["image_id"])]["row_bytes"]*2*
                images[str(r["image_id"])]["layers"] for r in hits
                if r.get("loaded_valid_visual_rows") is not None)),
            "retained_content_kv_mb_per_hit":as_mb(avg(
                r["compact_content_rows"]*images[str(r["image_id"])]["row_bytes"]*2*
                images[str(r["image_id"])]["layers"] for r in hits
                if r.get("compact_content_rows") is not None)),
            "extra_valid_mb_per_hit":as_mb(avg(
                r["extra_valid_visual_rows"]*images[str(r["image_id"])]["row_bytes"]*2*
                images[str(r["image_id"])]["layers"] for r in hits
                if r.get("extra_valid_visual_rows") is not None)),
            "padding_read_mb_per_hit":as_mb(avg(val("padding_read_bytes"))),
            "h2d_mb_per_hit":as_mb(avg(val("h2d_kv_bytes"))),
            "gpu_compact_cache_mb_per_hit":as_mb(avg(val("gpu_cache_kv_bytes"))),
            "peak_gpu_allocated_mb_per_hit":as_mb(avg(val("peak_gpu_allocated_bytes"))),
            "quality_by_turn":{str(t):avg(r["correct"] for r in records if int(r["turn_id"])==t)
                               for t in sorted({int(r["turn_id"]) for r in records})},
        }
    full=summary["fullload"]
    for method in METHODS[1:]:
        this=summary[method]
        this["normal_read_vs_fullload"]=this["normal_read_mb_per_hit"]/full["normal_read_mb_per_hit"]
        this["total_read_vs_fullload"]=this["total_read_mb_per_hit"]/full["total_read_mb_per_hit"]
    geom={"images":len(images),"equal_selected_sets":0,"different_selected_sets":0,
          "new_chunks_increase":0,"new_chunks_same":0,"new_chunks_decrease":0,
          "unused_valid_rows_new":0,"unused_valid_bytes_new":0,
          "padding_rows_new":0,"padding_bytes_new":0}
    for iid,info in images.items():
        old=set(info["perm"][:info["old_kept"]]);new=set(info["perm"][:info["k"]])
        geom["equal_selected_sets" if old==new else "different_selected_sets"]+=1
        geom["new_chunks_increase" if info["new_chunks"]>info["old_chunks"] else
             "new_chunks_same" if info["new_chunks"]==info["old_chunks"] else
             "new_chunks_decrease"]+=1
        valid=min(info["n"],info["new_chunks"]*64)
        unused=valid-info["k"]
        pad=info["new_chunks"]*64-valid
        unit=info["row_bytes"]*2*info["layers"]
        geom["unused_valid_rows_new"]+=unused
        geom["unused_valid_bytes_new"]+=unused*unit
        geom["padding_rows_new"]+=pad
        geom["padding_bytes_new"]+=pad*unit
    ref=read_json(MIGRATION/"geometry_reference.json")["phases"]["gqa" if phase=="smoke" else phase]
    if phase!="smoke":
        checks.want(geom["different_selected_sets"]==ref["different_selected_set"] and
                    geom["new_chunks_increase"]==ref["chunk_change_counts"].get("increase",0) and
                    geom["new_chunks_same"]==ref["chunk_change_counts"].get("same",0) and
                    geom["new_chunks_decrease"]==ref["chunk_change_counts"].get("decrease",0),
                    f"{phase}: prospective geometry population counts mismatch")
    phase_result={"phase":phase,"run_dir":str(directory),"raw_sha256":digest(directory/"raw.jsonl"),
        "manifest_file_sha256":digest(directory/"manifest.json"),
        "config_file_sha256":config_sha,
        "store_inventory_sha256":digest(directory/"store_inventory.json"),
        "expected_requests":expected_count,"observed_requests":len(raw),
        "failed_requests":sum(r.get("status")!="PASS" for r in raw),
        "duplicate_physical_execution_ids":len(ids)-len(set(ids)),
        "summary":summary,"geometry":geom}
    return phase,phase_result,raw


def bootstrap_pair(raw: list[dict]) -> dict:
    by_image=defaultdict(list)
    pairs=defaultdict(dict)
    for r in raw:
        if int(r["turn_id"])>1 and r["method_id"] in METHODS[-2:]:
            pairs[str(r["image_id"]),int(r["turn_id"])][r["method_id"]]=r
    for (iid,tid),rows in pairs.items():
        if set(rows)!=set(METHODS[-2:]):
            raise ValueError(f"paired old/new hit missing {iid}/T{tid}")
        old,new=rows[METHODS[-2]],rows[METHODS[-1]]
        by_image[iid].append((100*(new["correct"]-old["correct"]),
                              new["ttft_ms"]-old["ttft_ms"],
                              new["total_actual_pread_bytes"]-old["total_actual_pread_bytes"]))
    image_ids=sorted(by_image)
    values={iid:[avg(x[j] for x in by_image[iid]) for j in range(3)] for iid in image_ids}
    rng=random.Random(1234)
    draws=[[],[],[]]
    for _ in range(10000):
        sample=[image_ids[rng.randrange(len(image_ids))] for _ in image_ids]
        for j in range(3):
            draws[j].append(avg(values[iid][j] for iid in sample))
    def percentile(sorted_values,p):
        x=(len(sorted_values)-1)*p
        lo,hi=math.floor(x),math.ceil(x)
        return sorted_values[lo]+(sorted_values[hi]-sorted_values[lo])*(x-lo)
    names=("quality_pp","ttft_ms","total_pread_bytes")
    return {name:{"mean":avg(values[iid][j] for iid in image_ids),
                  "ci95":[percentile(sorted(draws[j]),.025),percentile(sorted(draws[j]),.975)]}
            for j,name in enumerate(names)}


def compare_csv(path: Path, phases: dict, checks: Checks) -> dict:
    with path.open(newline="",encoding="utf-8") as handle:
        rows=list(csv.DictReader(handle))
    actual={(r.get("dataset"),r.get("method")):r for r in rows}
    expected={(phase,method) for phase in phases for method in METHODS}
    checks.want(len(actual)==len(rows) and set(actual)==expected,
                "summary.csv method/phase coverage, extra rows, or duplicate rows mismatch")
    compared=0
    for key in expected:
        csv_row=actual.get(key)
        if csv_row is None:
            continue
        independent=phases[key[0]]["summary"][key[1]]
        for field in SUMMARY_METRICS:
            if field not in csv_row:
                checks.want(False,f"summary.csv missing field {field}")
                continue
            cell=csv_row[field]
            number=None if cell in ("", "None", "null") else float(cell)
            expected_value=independent.get(field)
            if field in ("requests","hit_requests"):
                okay=number==expected_value
            else:
                okay=near(number,expected_value,atol=1e-6,rtol=1e-7)
            checks.want(okay,f"summary.csv {key[0]}/{key[1]}/{field}: {cell} != {expected_value}")
            compared+=1
    return {"path":str(path.resolve()),"sha256":digest(path),"rows":len(rows),
            "metric_cells_checked":compared}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,action="append",required=True,
                        help="Repeat for smoke_pilot, gqa_pilot, mt_pilot")
    parser.add_argument("--summary-csv",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    checks=Checks()
    receipt={"schema_version":"qwen25-kv25-independent-raw-audit-v1",
             "status":"FAIL","checks":0,"errors":[],"phases":{},
             "source":"scripts/96_audit_qwen_kv25_pilot.py",
             "source_sha256":digest(Path(__file__)),
             "gpu_gate_sha256":None,"summary_csv":None,
             "persistence_status":"NOT_REMEASURED"}
    try:
        gate_path=MIGRATION/"gpu_validation/validation.json"
        gate=read_json(gate_path)
        receipt["gpu_gate_sha256"]=digest(gate_path)
        checks.want(gate.get("status")=="PASS" and gate.get("pilot_eligible") is True and
                    all(gate.get("gates",{}).get(f"G{i}",{}).get("status")=="PASS"
                        for i in range(1,13)),"frozen GPU correctness gate is not PASS")
        gate["path"]=str(gate_path)
        final_protection_path = MIGRATION / "final_protection_receipt.json"
        final_protection = read_json(final_protection_path)
        checks.want(digest(final_protection_path) ==
                    "e096f5631991d744aba92c071f537cfd3e83e18a8bb23e051566310e211b069d" and
                    final_protection.get("schema_version") == "qwen25-kv25-final-protection-v1" and
                    final_protection.get("status") == "PASS" and
                    final_protection.get("protected_file_count") == 30205 and
                    final_protection.get("changed_protected_file_count") == 2 and
                    final_protection.get("all_other_protected_files_unchanged") is True and
                    set(final_protection.get("allowed_changes", [])) ==
                    {"mmimpress/qwen25/runner.py", "mmimpress/qwen25/store.py"} and
                    {x.get("path") for x in final_protection.get("changed_protected_files", [])} ==
                    {"mmimpress/qwen25/runner.py", "mmimpress/qwen25/store.py"} and
                    all(digest(ROOT / x["path"]) == x["after"]
                        for x in final_protection.get("changed_protected_files", [])) and
                    final_protection.get("external_mt_image_count") == 40 and
                    final_protection.get("external_mt_changed") == [] and
                    final_protection.get("gpu_validation_sha256") == receipt["gpu_gate_sha256"] and
                    final_protection.get("protected_baseline_manifest_sha256") == BASELINE_SHA and
                    final_protection.get("llava_cpu_and_file_status") == "PASS" and
                    final_protection.get("llava_gpu_status") == "NOT RUN" and
                    final_protection.get("source_diff_sha256") ==
                    digest(MIGRATION / "source.diff"),
                    "post-pilot 30,205-file/40-external-MT protection receipt mismatch")
        receipt["final_protection"] = {"path":str(final_protection_path),
            "sha256":digest(final_protection_path),
            "protected_files_checked":final_protection.get("protected_file_count"),
            "external_mt_images_checked":final_protection.get("external_mt_image_count")}
        startup = MIGRATION / "smoke_pilot/startup_failure.json"
        if startup.is_file():
            failure = read_json(startup)
            checks.want(failure.get("schema_version") ==
                        "qwen25-kv25-pilot-startup-failure-v1" and
                        failure.get("status") == "FAIL_STARTUP_BEFORE_MODEL_LOAD" and
                        failure.get("physical_requests_executed") == 0 and
                        failure.get("protected_store_mutations") == 0 and
                        failure.get("gpu_gate_sha256") == receipt["gpu_gate_sha256"] and
                        failure.get("source_sha256") ==
                        digest(ROOT / "scripts/94_eval_qwen_kv25_pilot.py") and
                        not (startup.parent / "raw.jsonl").exists(),
                        "preserved original smoke startup failure was not zero-request")
            receipt["startup_failure"] = {"path": str(startup),
                "sha256": digest(startup), "physical_requests_executed":
                failure.get("physical_requests_executed")}
        raw_by_phase={}
        for directory in args.run_dir:
            try:
                phase,detail,raw=phase_audit(directory,gate,checks)
                checks.want(phase not in receipt["phases"],f"duplicate audit phase {phase}")
                receipt["phases"][phase]=detail
                raw_by_phase[phase]=raw
            except Exception as exc:
                checks.errors.append(f"{directory}: {type(exc).__name__}: {exc}")
        checks.want(set(receipt["phases"])==set(PHASES),
                    "all smoke/GQA/MT phases must be supplied for complete pilot audit")
        all_ids=[r.get("physical_execution_id") for raw in raw_by_phase.values() for r in raw]
        checks.want(len(all_ids)==len(set(all_ids)),
                    "physical execution ID reused across phases")
        for phase,raw in raw_by_phase.items():
            if phase in ("gqa","mt"):
                receipt["phases"][phase]["paired_new_minus_legacy"]=bootstrap_pair(raw)
        if args.summary_csv:
            receipt["summary_csv"]=compare_csv(args.summary_csv,receipt["phases"],checks)
        else:
            checks.want(False,"summary.csv required for complete pilot audit")
    except Exception as exc:
        checks.errors.append(f"audit infrastructure error: {type(exc).__name__}: {exc}")
    receipt["checks"]=checks.count
    receipt["errors"]=checks.errors
    receipt["status"]="PASS" if not checks.errors else "FAIL"
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as handle:
        json.dump(receipt,handle,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status":receipt["status"],"checks":checks.count,
                      "errors":len(checks.errors),"out":str(args.out.resolve())},sort_keys=True))
    return 0 if receipt["status"]=="PASS" else 1


if __name__=="__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Qwen2.5-VL MT-GQA six-arm runner: ReComp, FullLoad, B (unpacked), and three packed
layer-wise arms (U, PyramidKV beta=20, beta=2.5).  Image-atomic commits.

This ports the rules of scripts/110 (request construction, method-local generated history,
strict scoring, RequestTrace instrumentation, image-atomic attempt/commit, independent audit,
allowlisted cleanup, resume) to six arms with an ``ordinal % 6`` rotation.  110/111 are never
executed; unchanged helpers are imported from 110, and arms 1-3 call the unchanged runner
methods (run_pixels / run_cache / persist / condition_cache) exactly as 110 does.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import qwen_pyr_support as Q
from qwen_pyr_support import ROOT, load_script
from mmimpress.prefix25 import process_io
from mmimpress.qwen25 import layerwise as LW
from mmimpress.qwen25.packed_store import PACKED_NAME
from mmimpress.qwen25.runner import CHECKPOINT_REVISION, MODEL_ID

M = load_script("_qpyr_base110", "110_eval_qwen_mt_gqa_B.py")      # helpers only
METHODS = Q.METHODS
PACKED = Q.PACKED
B = Q.B
SCHEMA = Q.SCHEMA
EXPECTED_MAIN = 73098
sha, canonical, safe, atomic_json, append, read_rows = M.sha, M.canonical, M.safe, M.atomic_json, M.append, M.read_rows
strict_score, assert_gpu_exclusive, conditioning_trace = M.strict_score, M.assert_gpu_exclusive, M.conditioning_trace
file_hashes, cleanup_committed, capacity_guard = M.file_hashes, M.cleanup_committed, M.capacity_guard


def seed():
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)


def method_order(ordinal):
    i = ordinal % len(METHODS)
    return METHODS[i:] + METHODS[:i]


def logical_id(run, phase, method, did, t):
    return f"{SCHEMA}:{run.name}:{phase}:{MODEL_ID}:{method}:{did}:t{t}"


def history_factory(dialog, turn, method, generated):
    if method not in METHODS or turn not in (1, 2, 3) or set(generated) != set(range(1, turn)):
        raise ValueError("future/missing history")
    entries, history = [], []
    for t in range(1, turn):
        row = generated[t]
        q = dialog["turns"][t - 1]
        if (row["model_id"], row["method_id"], row["dialog_id"], row["turn_id"]) != (MODEL_ID, method, dialog["dialog_id"], t):
            raise ValueError("foreign history")
        history.append((q["question"], row["prediction"]))
        entries.append({"turn_id": t, "question_id": str(q["question_id"]), "question": q["question"], "answer": row["prediction"],
                        "source_model": MODEL_ID, "source_method_id": method, "source_dialog_id": dialog["dialog_id"],
                        "source_logical_request_id": row["logical_request_id"], "source_physical_execution_id": row["physical_execution_id"]})
    return dialog["turns"][turn - 1]["question"], tuple(history), entries


class RequestTrace(M.RequestTrace):
    """110's lightweight observation plus an ``os.preadv`` trace for the packed arms."""

    def __init__(self, runner, store=None, packed=None):
        super().__init__(runner, store)
        self.packed = packed

    def __enter__(self):
        super().__enter__()
        if self.packed is not None:
            self.fd_names[self.packed._fd] = str(self.packed.path / PACKED_NAME)
        self.preadv_orig = os.preadv

        def preadv(fd, buffers, offset, *flags):
            requested = sum(memoryview(b).nbytes for b in buffers)
            n = self.preadv_orig(fd, buffers, offset, *flags)
            self.reads.append({"syscall": "preadv", "path": self.fd_names.get(fd, f"UNKNOWN_FD_{fd}"), "offset": offset, "requested": requested, "returned": n})
            return n
        os.preadv = preadv
        return self

    def __exit__(self, *exc):
        os.preadv = self.preadv_orig
        super().__exit__(*exc)
        for e in self.reads:
            e.setdefault("syscall", "pread")


def execute_request(runner, session, image_path, image_sha, factory, method, turn, store_path=None, doc=None, capture=False, return_logits=False):
    pixel = turn == 1 or method == "recompute"
    conditioning = None
    events = []
    store = packed = None
    LW.assert_no_layer_hooks(runner.model)
    if not pixel:
        if method in PACKED:
            with conditioning_trace() as events:
                c0 = time.perf_counter()
                conditioning = session.condition(store_path, image_sha, doc)
                conditioning["outer_wall_ms"] = (time.perf_counter() - c0) * 1000
            assert conditioning["failures"] == 0 and len(events) == 1 and all(e["status"] == "SUCCESS" for e in events)
            packed = session.stores[str(Path(store_path).resolve())]
        else:
            ratio = 1. if method == "fullload" else .25
            unit = "chunk" if method == "fullload" else "visual_kv"
            with conditioning_trace() as events:
                c0 = time.perf_counter()
                conditioning = runner.condition_cache(store_path, ratio, budget_unit=unit, image_sha256=image_sha)
                conditioning["outer_wall_ms"] = (time.perf_counter() - c0) * 1000
            assert conditioning["failures"] == 0 and len(events) == 57 and all(e["status"] == "SUCCESS" for e in events)
            store = runner._stores[str(Path(store_path).resolve())]
    torch.cuda.synchronize(runner.device)
    with RequestTrace(runner, store, packed) as trace:
        started = time.perf_counter()
        question, history, entries = factory()
        image_decode = None
        if pixel:
            t0 = time.perf_counter()
            with Image.open(image_path) as im:
                image = im.convert("RGB")
                image.load()
            image_decode = (time.perf_counter() - t0) * 1000
            result = runner.run_pixels(image, question, history=history, capture=capture, image_sha256=image_sha, return_logits=return_logits)
        elif method in PACKED:
            result = LW.run_cache_layerwise(runner, session, store_path, question, history, layer_budget=doc, image_sha256=image_sha, return_logits=return_logits)
        else:
            result = runner.run_cache(store_path, question, history=history, budget_ratio=ratio, budget_unit=unit, image_sha256=image_sha, return_logits=return_logits)
        returned = time.perf_counter()
    assert trace.vision_calls == int(pixel) and trace.score_calls == int(capture == "with_score")
    assert runner.model.model.rope_deltas is None
    residual = LW.residual_layer_hooks(runner.model)
    assert residual == 0, "residual layer mask hook"
    assert len(result["generated_token_ids"]) == trace.model_calls
    assert trace.diag["cache_after_decode"] == trace.diag["cache_before_decode"] + len(result["generated_token_ids"]) - 1
    assert bool(torch.isfinite(trace.first_logits_for_finite).all()), "nonfinite first logits"
    core_ttft = result["ttft_ms"]
    core_e2e = result["request_e2e_ms"]
    result.update(core_ttft_ms=core_ttft, core_request_e2e_ms=core_e2e, ttft_ms=1000 * (trace.first_at - started),
                  request_e2e_ms=1000 * ((returned if capture else trace.end_at) - started),
                  request_started_at_s=started, first_token_at_s=trace.first_at, request_finished_at_s=(returned if capture else trace.end_at),
                  generation_finished_at_s=trace.end_at, core_returned_at_s=returned, post_generation_tail_ms=(returned - trace.end_at) * 1000,
                  image_file_decode_ms=image_decode, first_logits_finite=True, actual_vision_calls=trace.vision_calls, actual_score_calls=trace.score_calls,
                  model_forward_calls=trace.model_calls, os_pread_trace=trace.reads, conditioning=conditioning, conditioning_events=events,
                  request_state_restored=True, residual_layer_hooks=residual, capture_mode=capture)
    diag = {k.removesuffix("_tensor"): safe(v) for k, v in trace.diag.items()}
    diag.update(history=safe(history), history_entries=entries)
    if not pixel:
        assert sum(e["returned"] for e in trace.reads) == result["read_io"]["bytes"]
        assert len(trace.reads) == result["pread_calls"]
        if method in PACKED:
            assert len(trace.reads) == 1 and trace.reads[0]["syscall"] == "preadv" and trace.reads[0]["offset"] == 0
            assert trace.reads[0]["returned"] == trace.reads[0]["requested"] == packed.total
            assert trace.reads[0]["path"] == str(packed.path / PACKED_NAME)
        else:
            assert all(e["syscall"] == "pread" for e in trace.reads)
    else:
        assert not trace.reads
    return result, diag


def persist_packed_receipt(runner, cap, path, doc, source):
    """Packed analogue of main_b_support.persist_qwen (same receipt fields and boundaries)."""
    import main_b_support as support
    start = time.perf_counter()
    before = process_io()
    receipt = LW.persist_packed(runner, cap, path, doc)
    writer_ms = (time.perf_counter() - start) * 1000
    t = time.perf_counter()
    meta = receipt["metadata"]
    payload = path / PACKED_NAME
    records = {PACKED_NAME: {"size": payload.stat().st_size, "sha256": meta["packed_sha256"], "allocated_bytes": payload.stat().st_blocks * 512}}
    p = path / "meta.json"
    records["meta.json"] = {"size": p.stat().st_size, "allocated_bytes": p.stat().st_blocks * 512, "sha256": sha(p)}
    files = support.complete(path, records, "completion.json")
    seal_ms = (time.perf_counter() - t) * 1000
    elapsed = (time.perf_counter() - start) * 1000
    after = process_io()
    receipt.update(event_id=str(uuid.uuid4()), event_kind="physical_build", source=source, files=files, service_persistence_ms=elapsed,
                   native_writer_wall_ms=writer_ms, completion_full_hash_ms=0., completion_publication_ms=seal_ms,
                   os_write_bytes=after["wchar"] - before["wchar"], os_write_calls=after["syscw"] - before["syscw"],
                   file_logical_bytes=sum(x["size"] for x in files.values()), allocated_bytes=sum(x["allocated_bytes"] for x in files.values()),
                   provenance="MEASURED", completion_policy="native whole-payload serialized-byte SHA reused; metadata hash and completion+parent fsync")
    return receipt


def make_row(result, diag, method, dialog, turn, phase, run, attempt, physical, ordinal, config, image_sha, store_info=None, meta=None):
    q = dialog["turns"][turn - 1]
    pixel = turn == 1 or method == "recompute"
    n = result["geometry"]["visual_count"] if pixel else meta["visual_count"]
    s = result["geometry"]["prefix_len"] - n if pixel else meta["structural_count"]
    k = n if pixel or method == "fullload" else (n + 3) // 4
    packed_hit = method in PACKED and not pixel
    unit = "visual_kv" if method == B else "visual_kv_layerwise" if method in PACKED else "full_context"
    clean = {key: v for key, v in result.items() if key not in ("capture", "first_logits")}
    trace = result["os_pread_trace"]
    return {"schema_version": SCHEMA, "experiment_id": run.name, "phase": phase, "attempt_id": attempt, "status": "PASS",
            "logical_request_id": logical_id(run, phase, method, dialog["dialog_id"], turn), "physical_execution_id": physical,
            "model_id": MODEL_ID, "model_revision": CHECKPOINT_REVISION, "method_id": method, "image_id": dialog["image_id"], "dialog_id": dialog["dialog_id"], "turn_id": turn,
            "question_id": str(q["question_id"]), "question": q["question"], "gold": q["answers"], "prediction": result["prediction"],
            "score": strict_score(result["prediction"], q["answers"][0]),
            "history": diag["history"], "history_entries": diag["history_entries"], "history_sha256": canonical(diag["history"]),
            "prompt": diag["prompt"], "prompt_sha256": hashlib.sha256(diag["prompt"].encode()).hexdigest(), "input_ids": diag["input_ids"], "input_ids_sha256": canonical(diag["input_ids"]),
            "request_trace": diag, "config_sha256": config["_file_sha256"], "manifest_sha256": config["_manifest_sha256"], "code_sha256": canonical(config["source_sha256"]),
            "image_sha256": image_sha, "global_dialog_ordinal": ordinal, "method_order": list(method_order(ordinal)), "method_order_position": method_order(ordinal).index(method),
            "N_content": n, "S_structural": s, "k": k, "k_per_layer": list(result["layer_keep_counts"]) if packed_hit else None,
            "layer_budget_id": PACKED[method] if packed_hit else None, "layer_budget_sha256": result["layer_budget_sha256"] if packed_hit else None,
            "content_kv_fraction": k / n, "structural_inclusive_kept_ratio": (k + s) / (n + s), "budget_unit": unit,
            "request_path": "normal_pixels" if pixel else "ssd_cache_hit", "provisioning_mode": "FRESH_IMAGE_STREAMING",
            "source_store": store_info if not pixel else None, "ttft_ms": result["ttft_ms"], "request_e2e_ms": result["request_e2e_ms"],
            "ssd_read_bytes": sum(e["returned"] for e in trace), "ssd_preads": len(trace),
            "ssd_pread_calls": sum(e["syscall"] == "pread" for e in trace), "ssd_preadv_calls": sum(e["syscall"] == "preadv" for e in trace),
            "result": safe(clean)}


def run_image(runner, dialogs, ordinals, phase, run, manifest, config, docs, diagnostic=None):
    import main_b_support as support
    iid = dialogs[0]["image_id"]
    parent = run / phase / "images" / iid
    parent.mkdir(parents=True, exist_ok=True)
    if (parent / "COMMITTED.json").exists():
        raise FileExistsError("already committed")
    attempt = f"attempt_{len(list(parent.glob('attempt_*'))) + 1:04d}"
    artifact = parent / attempt
    artifact.mkdir()
    scratch = run / "scratch" / phase / iid / attempt
    atomic_json(artifact / "capacity_before.json", capacity_guard(run))
    assert not scratch.exists()
    atomic_json(artifact / "scratch_creation.json", {"path": str(scratch), "ownership": run.name, "previously_absent": True})
    scratch.mkdir(parents=True)
    paths = {m: scratch / m for m in METHODS[1:]}
    metas, hashes, info, captures, rows = {}, {}, {}, {}, []
    success = False
    session = LW.PackedSession(runner)
    image_path = ROOT / dialogs[0]["image_path"]
    image_sha = sha(image_path)
    assert image_sha == manifest["image_identity"][iid]["sha256"]
    atomic_json(artifact / "image_identity.json", {"image_id": iid, "image_path": str(image_path), "sha256": image_sha})
    try:
        for di, dialog in enumerate(dialogs):
            generated = {m: {} for m in METHODS}
            ordinal = ordinals[dialog["dialog_id"]]
            for turn in (1, 2, 3):
                for method in method_order(ordinal):
                    inv = assert_gpu_exclusive()
                    physical = uuid.uuid4().hex
                    lid = logical_id(run, phase, method, dialog["dialog_id"], turn)
                    source = {"dialog_id": dialog["dialog_id"], "method_id": method, "physical_execution_id": physical, "logical_request_id": lid, "turn_id": turn}
                    append(artifact / "attempts.jsonl", {**source, "status": "STARTED", "at_unix": time.time(), "gpu_inventory": inv})
                    capture = ("kv_only" if method == "fullload" else "with_score") if di == 0 and turn == 1 and method != "recompute" else False
                    factory = lambda: history_factory(dialog, turn, method, generated[method])
                    doc = docs[PACKED[method]] if method in PACKED else None
                    result, diag = execute_request(runner, session, image_path, image_sha, factory, method, turn, paths.get(method), doc,
                                                   capture=capture, return_logits=diagnostic is not None)
                    append(artifact / "execution_events.jsonl", {**source, "status": "EXECUTED", "prediction": result["prediction"], "generated_token_ids": result["generated_token_ids"],
                                                                 "ttft_ms": result["ttft_ms"], "request_e2e_ms": result["request_e2e_ms"]})
                    if capture:
                        cap = result.pop("capture")
                        t0 = time.perf_counter()
                        if method in PACKED:
                            persist = persist_packed_receipt(runner, cap, paths[method], doc, source)
                        else:
                            persist = support.persist_qwen(runner, cap, paths[method], "canonical" if method == "fullload" else "repacked", source)
                        wall = (time.perf_counter() - t0) * 1000
                        meta = persist["metadata"]
                        metas[method] = meta
                        hashes[method] = file_hashes(paths[method])
                        scores = safe(cap.scores) if cap.scores is not None else None
                        n = cap.visual_count
                        selection = {"N_content": n, "k": (n + 3) // 4, "scores_float32": scores, "stored_to_original": meta["stored_to_original"],
                                     "prefix_ids": cap.prefix_ids, "logical_position_ids": cap.logical_position_ids, "geometry": cap.geometry, "source": source}
                        if method in PACKED:
                            selection.update(k_per_layer=meta["layer_keep_counts"], full_importance_permutation=meta["full_importance_permutation"],
                                             layer_budget_id=meta["layer_budget_id"], layer_budget_sha256=meta["layer_budget_sha256"],
                                             selection_rule="layer l keeps full_importance_permutation[:k_per_layer[l]]")
                        else:
                            selection["selected_original_ids"] = sorted(meta["stored_to_original"][:(n + 3) // 4]) if method == B else list(range(n))
                        atomic_json(artifact / f"{method}_selection.json", selection)
                        info[method] = {"path": str(paths[method]), "meta_sha256": sha(paths[method] / "meta.json"), "source": source, "physical_build_event_id": persist["event_id"],
                                        "selection_artifact": str(artifact / f"{method}_selection.json"), "selection_sha256": sha(artifact / f"{method}_selection.json")}
                        p = {key: v for key, v in persist.items() if key != "metadata"}
                        p.update(source=source, outer_writer_wall_ms=wall, post_generation_capture_tail_ms=result["post_generation_tail_ms"],
                                 charged_setup_before_activation_ms=wall, bytes_written=persist["file_logical_bytes"],
                                 scope="MEASURED writer plus completion publication; post-generation score/clone tail already included exactly once in source T1 E2E")
                        atomic_json(artifact / f"{method}_persistence.json", p)
                        atomic_json(artifact / f"{method}_meta.json", meta)
                        atomic_json(artifact / f"{method}_hashes.json", hashes[method])
                        if diagnostic is not None:
                            captures[method] = cap
                        del cap
                    row = make_row(result, diag, method, dialog, turn, phase, run, attempt, physical, ordinal, config, image_sha, info.get(method), metas.get(method))
                    assert row["N_content"] == manifest["image_identity"][iid]["N_content"]
                    append(artifact / "raw.jsonl", row)
                    rows.append(row)
                    generated[method][turn] = row
                    if method in paths and not (artifact / f"{method}_activation.json").exists():
                        key = str(paths[method].resolve())
                        record = session.activation_records.get(key) if method in PACKED else runner.activation_records.get(key)
                        if record is not None:
                            atomic_json(artifact / f"{method}_activation.json", dict(record, event_id=str(uuid.uuid4()), event_kind="activation"))
                    if diagnostic is not None:
                        diagnostic(runner, session, dialog, turn, method, row, result, captures, paths, metas, artifact, docs)
                    append(artifact / "attempts.jsonl", {**source, "status": "COMPLETED", "at_unix": time.time()})
                    del result, diag
        for m in METHODS[1:]:
            assert file_hashes(paths[m]) == hashes[m]
        atomic_json(artifact / "store_integrity_after.json", {"status": "PASS", "file_sha256": hashes})
        aud = load_script("_qpyr_commit_auditor", "170_audit_qwen_layer_budget_pyr.py")
        receipt = aud.audit_image_rows(read_rows(artifact / "raw.jsonl"), manifest, config, phase, iid, artifact)
        receipt["raw_sha256"] = sha(artifact / "raw.jsonl")
        atomic_json(artifact / "independent_audit.json", receipt)
        if receipt["status"] != "PASS":
            raise RuntimeError("INDEPENDENT_IMAGE_AUDIT_FAIL " + json.dumps(receipt))
        allowlist = file_hashes(scratch)
        atomic_json(artifact / "cleanup_allowlist.json", {"scratch": str(scratch), "file_sha256": allowlist, "ownership_run": run.name})
        evidence = {p.name: sha(p) for p in artifact.iterdir() if p.is_file()}
        commit = {"status": "COMMITTED", "image_id": iid, "attempt_id": attempt, "requests": len(rows), "raw_sha256": receipt["raw_sha256"],
                  "config_sha256": config["_file_sha256"], "manifest_sha256": config["_manifest_sha256"], "evidence_sha256": evidence,
                  "first_successful_complete_image_adoption": True}
        atomic_json(parent / "COMMITTED.json", commit)
        success = True
    except BaseException as e:
        atomic_json(artifact / "failure.json", {"status": "PARTIAL", "error": repr(e), "traceback": traceback.format_exc(), "completed_uncommitted_rows": len(rows), "adopted_from_attempt": 0})
        raise
    finally:
        for layer in LW.decoder_layers(runner.model):       # never leak a hook past a failed image
            layer.self_attn._forward_pre_hooks.clear()
        runner.close()
        session.close()
        captures.clear()
        gc.collect()
        if success:
            cleanup_committed(scratch, allowlist, receipt, commit, run)
            atomic_json(artifact / "cleanup_receipt.json", {"status": "PASS", "removed_new_files": len(allowlist),
                                                            "recipe": "rerun this complete image with the exact frozen config/source/manifest in a fresh attempt; regenerate all method-local histories"})
    return len(rows)


def completed_images(run, phase, config, manifest):
    audit = load_script("_qpyr_resume_auditor", "170_audit_qwen_layer_budget_pyr.py")
    done = {}
    for cp in sorted((run / phase / "images").glob("*/COMMITTED.json")):
        c = json.loads(cp.read_text())
        folder = cp.parent / c["attempt_id"]
        if c["config_sha256"] != config["_file_sha256"] or c["manifest_sha256"] != config["_manifest_sha256"]:
            raise RuntimeError("resume identity mismatch")
        for name, digest in c["evidence_sha256"].items():
            if sha(folder / name) != digest:
                raise RuntimeError("committed evidence changed " + str(folder / name))
        receipt = audit.audit_image_rows(read_rows(folder / "raw.jsonl"), manifest, config, phase, cp.parent.name, folder)
        if receipt["status"] != "PASS":
            raise RuntimeError("resume audit failed " + repr(receipt))
        if not (folder / "cleanup_receipt.json").exists():
            a = json.loads((folder / "cleanup_allowlist.json").read_text())
            scratch = Path(a["scratch"])
            if scratch.exists():
                cleanup_committed(scratch, a["file_sha256"], receipt, c, run)
            atomic_json(folder / "cleanup_receipt.json", {"status": "PASS", "resume_cleanup": True, "removed_new_files": len(a["file_sha256"])})
        done[cp.parent.name] = c
    return done


def verify_freeze(run):
    from mmimpress.mt_gqa import workload_sha256
    config = json.loads((run / "config.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    assert list(METHODS) == config["methods"] == manifest["methods"] and dict(PACKED) == config["packed_methods"], "FROZEN_ARMS_MISMATCH"
    assert Q.source_hashes() == config["source_sha256"], "FROZEN_SOURCE_CONFIG_MISMATCH"
    assert sha(run / "CONTRACT.md") == config["contract_sha256"] == sha(Q.CONTRACT) and sha(run / "storage_plan.json") == config["storage_plan_sha256"]
    assert sha(run / "manifest.json") == config["manifest_sha256"] and sha(run / "design_ragged_cache.md") == config["design_sha256"]
    assert sha(ROOT / "data/mt_gqa/dialogues.json") == Q.INDEX_SHA
    ds = json.loads((ROOT / "data/mt_gqa/dialogues.json").read_text())["dialogues"]
    assert ds == manifest["dialogues"] and workload_sha256(ds) == Q.WORKLOAD_SHA
    for iid, x in manifest["image_identity"].items():
        assert sha(ROOT / x["image_path"]) == x["sha256"], iid
    docs = Q.load_vectors(run)
    for a in Q.ALLOCS:
        v = config["vectors"][a]
        assert sha(Q.vector_path(run, a)) == v["file_sha256"] and docs[a]["layer_budget_sha256"] == v["layer_budget_sha256"] == Q.RATIO_SHA[a]
        assert (docs[a]["num_layers"], docs[a]["r_min"], docs[a]["r_max"]) == (28, 0.0, 0.60)
    assert LW.packed_code_revision() == config["packed_code_revision"]
    config.update(_file_sha256=sha(run / "config.json"), _manifest_sha256=sha(run / "manifest.json"))
    return config, manifest, docs


def update_status(run, state, phase, count, images, error=None):
    doc = {"execution_state": state, "phase": phase, "phase_completed_requests": count, "phase_completed_images": images, "expected_main_requests": EXPECTED_MAIN,
           "at_unix": time.time(), "pid": os.getpid(),
           "main_final_requests": sum(json.loads(cp.read_text())["requests"] for cp in (run / "main/images").glob("*/COMMITTED.json")),
           "resume_command": Q.resume_command(run), "error": error}
    atomic_json(run / "CURRENT_STATUS.json", doc, replace=True)
    out = ROOT / "results" / run.name
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "CURRENT_STATUS.json", doc, replace=True)
    print(json.dumps(doc), flush=True)


def run_phase(runner, run, phase, config, manifest, docs, diagnostic=None):
    done = completed_images(run, phase, config, manifest)
    ids = set(manifest[phase + "_dialogue_ids"]) if phase in ("integration", "smoke") else {d["dialog_id"] for d in manifest["dialogues"]}
    groups, ordinals = OrderedDict(), {}
    for i, d in enumerate(manifest["dialogues"]):
        ordinals[d["dialog_id"]] = i
        if d["dialog_id"] in ids:
            groups.setdefault(d["image_id"], []).append(d)
    per_dialogue = 3 * len(METHODS)
    count = sum(c["requests"] for c in done.values())
    for iid, ds in groups.items():
        if iid in done:
            continue
        if (run / "STOP_REQUESTED").exists():
            raise RuntimeError("STOP_REQUESTED")
        count += run_image(runner, ds, ordinals, phase, run, manifest, config, docs, diagnostic)
        done[iid] = {"requests": len(ds) * per_dialogue}
        update_status(run, "RUNNING", phase, count, len(done))
    assert count == len(ids) * per_dialogue
    receipt = {"status": "PASS", "requests": count, "expected_requests": len(ids) * per_dialogue, "images": len(groups),
               "config_sha256": config["_file_sha256"], "source_sha256": config["source_sha256"]}
    p = run / phase / "validation.json"
    if p.exists():
        assert json.loads(p.read_text()) == receipt
    else:
        atomic_json(p, receipt)
    return receipt


# ------------------------------------------------------------ integration (QG7)
class IntegrationDiagnostic:
    """Same-run cross-arm identity on the integration fixture: with identical histories the
    U_packed hit and the unpacked B hit must give bitwise-equal first-token logits and the
    same generated ids (they serve the same KV)."""

    def __init__(self):
        self.records = {}
        self.comparisons = []
        self.capture_digests = {}

    @staticmethod
    def _capture_digest(cap):
        h = hashlib.sha256()
        for k, v in cap.layers:
            for x in (k, v):
                h.update(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        if cap.scores is not None:
            h.update(cap.scores.detach().float().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def __call__(self, runner, session, dialog, turn, method, row, result, captures, paths, metas, artifact, docs):
        key = (dialog["image_id"], method)
        if method in captures and key not in self.capture_digests:
            # every scored arm captures its own T1; U_packed can only equal B if the captured bits are identical
            self.capture_digests[key] = self._capture_digest(captures[method])
            ref = self.capture_digests.get((dialog["image_id"], B))
            if ref is not None and method in PACKED:
                assert self.capture_digests[key] == ref, f"T1 capture bits differ between {method} and B"
        logits = result.get("first_logits")
        digest = hashlib.sha256(logits.contiguous().numpy().tobytes()).hexdigest() if logits is not None else None
        self.records[(dialog["dialog_id"], turn, method)] = {"logits_sha256": digest, "generated_token_ids": row["result"]["generated_token_ids"],
                                                             "history_sha256": row["history_sha256"], "prediction": row["prediction"]}
        other = {B: Q.U_PACKED, Q.U_PACKED: B}.get(method)
        if other and (dialog["dialog_id"], turn, other) in self.records:
            a, b = self.records[(dialog["dialog_id"], turn, method)], self.records[(dialog["dialog_id"], turn, other)]
            cmp = {"dialog_id": dialog["dialog_id"], "turn_id": turn, "same_history": a["history_sha256"] == b["history_sha256"],
                   "logits_bitwise_equal": a["logits_sha256"] == b["logits_sha256"], "generated_ids_equal": a["generated_token_ids"] == b["generated_token_ids"]}
            self.comparisons.append(cmp)
            if cmp["same_history"]:
                assert cmp["logits_bitwise_equal"] and cmp["generated_ids_equal"], cmp
        if method in PACKED and (dialog["dialog_id"], turn, method) in self.records and turn > 1:
            assert row["result"]["layer_keep_counts"] == Q.keep_counts(docs[PACKED[method]], row["N_content"])


def gates_admitted(run, config, upto):
    """Fail-closed admission: earlier gate records must exist, be PASS and bind the frozen sources."""
    need = {"integration": ["gates/qg1_cpu.json", "gates/cpu_regression.json", "gates/qg_gpu.json"],
            "smoke": ["gates/qg1_cpu.json", "gates/cpu_regression.json", "gates/qg_gpu.json", "integration_validation.json"]}
    for rel in need[upto]:
        doc = json.loads((run / rel).read_text())
        assert doc["status"] == "PASS", f"gate not PASS: {rel}"
        assert doc.get("config_sha256") == config["_file_sha256"], f"gate not bound to the frozen config: {rel}"


def stage_integration(run):
    config, manifest, docs = verify_freeze(run)
    gates_admitted(run, config, "integration")
    assert not (run / "integration_validation.json").exists(), "integration already recorded"
    runner = None
    report = {"status": "RUNNING", "config_sha256": config["_file_sha256"], "source_sha256": config["source_sha256"]}
    diag = IntegrationDiagnostic()
    try:
        seed()
        capacity_guard(run)
        runner = M.load_runner()
        atomic_json(run / "integration_runtime.json", runner.runtime_fingerprint())
        report["phase"] = run_phase(runner, run, "integration", config, manifest, docs, diagnostic=diag)
        same = [c for c in diag.comparisons if c["same_history"]]
        assert len(diag.comparisons) == 12 and all(c["logits_bitwise_equal"] and c["generated_ids_equal"] for c in same)
        scored = {}
        for (iid, method), digest in diag.capture_digests.items():
            if method != "fullload":
                scored.setdefault(iid, {})[method] = digest
        assert len(scored) == 4 and all(len(v) == 4 and len(set(v.values())) == 1 for v in scored.values()), "scored-arm T1 captures are not bit-identical"
        report.update(u_packed_vs_B=diag.comparisons, u_packed_vs_B_same_history=len(same), t1_capture_digests=scored,
                      t1_capture_bits_identical_across_scored_arms=True, status="PASS")
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if runner:
            runner.close()
        atomic_json(run / "integration_validation.json", report)


def stage_main(run, phases, resume):
    config, manifest, docs = verify_freeze(run)
    gates_admitted(run, config, "smoke")
    seed()
    runner = None
    try:
        capacity_guard(run)
        runner = M.load_runner()
        fp = runner.runtime_fingerprint()
        assert fp == json.loads((run / "integration_runtime.json").read_text()), "runtime changed"
        atomic_json(run / ("main_runtime_" + time.strftime("%Y%m%dT%H%M%S") + ".json"), fp)
        for phase in phases:
            if phase == "main":
                smoke = json.loads((run / "smoke/validation.json").read_text())
                assert smoke["status"] == "PASS" and smoke["requests"] == 72 and smoke["config_sha256"] == config["_file_sha256"]
                if not (run / "main_started_at.json").exists():
                    atomic_json(run / "main_started_at.json", {"at_unix": time.time()})
            if (run / phase).exists() and not resume:
                raise FileExistsError("use --resume; preserve partial")
            run_phase(runner, run, phase, config, manifest, docs)
        if "main" in phases:
            update_status(run, "AWAITING_INDEPENDENT_FINAL_AUDIT", "main", EXPECTED_MAIN, 398)
            if not (run / "main_finished_at.json").exists():
                atomic_json(run / "main_finished_at.json", {"at_unix": time.time()})
    except BaseException as e:
        update_status(run, "INTERRUPTED", locals().get("phase", "startup"), 0, 0, repr(e))
        atomic_json(run / ("interruption_" + time.strftime("%Y%m%dT%H%M%S") + ".json"),
                    {"status": "PARTIAL", "error": repr(e), "traceback": traceback.format_exc(), "resume_command": Q.resume_command(run)})
        raise
    finally:
        if runner:
            runner.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--phase", choices=("integration", "smoke", "main", "all"), default="all")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    run = a.run.resolve()
    if a.phase == "integration":
        stage_integration(run)
    else:
        stage_main(run, ["smoke", "main"] if a.phase == "all" else [a.phase], a.resume)


if __name__ == "__main__":
    main()

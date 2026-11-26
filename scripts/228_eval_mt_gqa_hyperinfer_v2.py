#!/usr/bin/env python3
"""MT-GQA x {LLaVA-NeXT, LLaVA-1.5}: HyperInfer with the paper's reactive truncation
(contract docs/mt_gqa_hyperinfer_v2_contract.md).

The prefetch of layer l+1 stops when layer l's GPU work is done (mmimpress/hyperinfer_v2.py).
Only HyperInfer is measured in the main phase.  ReComp, FullLoad, IMPRESS and Ours are the rows
of the earlier main runs; the earlier HyperInfer run (runs/mt_gqa_hyperinfer_20261124T171039Z,
public-code time budget) is compared in the report.

Where the code comes from

* scripts/225 (the earlier HyperInfer run, not modified) is loaded as a private module of this
  process: its status writer, manifest builder and ``Backend`` (the imported harness of each
  model -- scripts/222 for LLaVA-NeXT, scripts/206 for LLaVA-1.5 -- with the HyperInfer arms,
  phases, request dispatcher and status attached) are used; its arm table, phase table and
  expected request counts are replaced, in this process only, by the ones below.
* What is written here: the request of the HyperInfer arms (``hyper_hit``: the same skeleton,
  calling ``mmimpress.hyperinfer_v2``), this run's per-image audit, freeze and stages.  There is
  no calibration stage: no time budget and no chunk time constant exist any more.

Arms
  fullload_async            FullLoad (gate 7 table)
  impress_official_async    IMPRESS  (gates 1, 2; gate 7 table; the I/O ratio of the report)
  hyperinfer_async          HyperInfer -- the only arm of the main phase (gates 1, 5)
  hyperinfer_off            HyperInfer with the prefetch switched off      (gate 2)
  hyperinfer_async_repeat   HyperInfer executed a second time              (gate 4)
  hyperinfer_async_nan      HyperInfer, rows that were not read hold NaN   (gate 3)

Phases, per model: gate (4 images, 6 arms) -> dev (DEV 40, 6 arms) -> main (398 images, HyperInfer).

Stages
  --stage freeze                config, manifest, storage plan, protection inventory
  --stage gate|dev|main --model M [--resume]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mmimpress import hyperinfer_v2 as H2  # noqa: E402
from mmimpress import impress_official  # noqa: E402
import llava_next_main_status as RAW_STATUS  # noqa: E402


def load_script(name, filename):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


E1 = load_script("_mt_gqa_hyperinfer_v2_base", "225_eval_mt_gqa_hyperinfer.py")
sha, atomic_json, results_dir, source_hashes, selection_digest = E1.sha, E1.atomic_json, E1.results_dir, E1.source_hashes, E1.selection_digest
MODELS, MODEL_DIR, MODEL_LABEL, BACKEND_SCRIPT, REF_RUN = E1.MODELS, E1.MODEL_DIR, E1.MODEL_LABEL, E1.BACKEND_SCRIPT, E1.REF_RUN
REPORT_SCRIPT, GiB, NO_RETRY_MARKERS = E1.REPORT_SCRIPT, E1.GiB, E1.NO_RETRY_MARKERS

SCHEMA = "mt-gqa-hyperinfer-v2"
RUN_PREFIX = "mt_gqa_hyperinfer_v2_"
CONTRACT = ROOT / "docs/mt_gqa_hyperinfer_v2_contract.md"
PORT_TABLE = ROOT / "docs/hyperinfer_port_contract.md"
DRIVER = "scripts/run_mt_gqa_hyperinfer_v2.sh"
V1_RUN = ROOT / "runs/mt_gqa_hyperinfer_20261124T171039Z"      # the earlier HyperInfer run (public-code time budget)

FULL, IMPRESS = E1.FULL, E1.IMPRESS
HYPER, HYPER_OFF, HYPER_REPEAT, HYPER_NAN = "hyperinfer_async", "hyperinfer_off", "hyperinfer_async_repeat", "hyperinfer_async_nan"
HYPER_ARMS = OrderedDict([
    (HYPER,        {"kv_prefetch": "async", "nan": False, "display": "HyperInfer"}),
    (HYPER_OFF,    {"kv_prefetch": "off",   "nan": False, "display": "HyperInfer, prefetch 끔"}),
    (HYPER_REPEAT, {"kv_prefetch": "async", "nan": False, "display": "HyperInfer, 두 번째 실행"}),
    (HYPER_NAN,    {"kv_prefetch": "async", "nan": True,  "display": "HyperInfer, 읽지 않은 행 NaN"}),
])
GATE_METHODS = [FULL, IMPRESS, HYPER_OFF, HYPER, HYPER_REPEAT, HYPER_NAN]
MAIN_METHODS = [HYPER]
PHASE_METHODS = {"gate": list(GATE_METHODS), "dev": list(GATE_METHODS), "main": list(MAIN_METHODS)}
PHASES = ("gate", "dev", "main")
EXPECTED_REQUESTS = {"gate": 49 * 3 * 6, "dev": 378 * 3 * 6, "main": 12183}
# seconds per request before a phase has measured its own rate (status.json estimate only): the earlier HyperInfer run's rates
SECONDS_PER_REQUEST_PRIOR = {"next": {"gate": 0.85, "dev": 0.78, "main": 0.97}, "l15": {"gate": 0.28, "dev": 0.25, "main": 0.30}}

# this run's tables, in the private copy of scripts/225 (this process only)
E1.SCHEMA, E1.DRIVER = SCHEMA, DRIVER
E1.HYPER_ARMS, E1.GATE_METHODS, E1.MAIN_METHODS, E1.PHASE_METHODS = HYPER_ARMS, GATE_METHODS, MAIN_METHODS, PHASE_METHODS
E1.EXPECTED_REQUESTS, E1.SECONDS_PER_REQUEST_PRIOR = EXPECTED_REQUESTS, SECONDS_PER_REQUEST_PRIOR
assert E1.PHASES == PHASES


# ================================================================== backend
class Backend(E1.Backend):
    """scripts/225's backend with the HyperInfer request and the audit of this run."""

    def hyper_hit(self, runner, servers, ctx, factory, method, image_id, store_path):
        """``stored_hit`` of the imported harness for the HyperInfer arms: the same conditioning, timer
        start, prompt construction and tokenisation inside the timer, counters and read trace."""
        SRC, spec = self.SRC, HYPER_ARMS[method]
        arm = SRC.ARMS[method]
        official = SRC.OFFICIAL
        if spec["kv_prefetch"] != "off":
            H2.position_maps_for(ctx)          # store metadata tables, built once per activated store, outside the timer
        with SRC.QA._NoVisionForward(runner) as guard:
            c0 = time.perf_counter(); ctx.reader.drop_all(); conditioning = (time.perf_counter() - c0) * 1000
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
            prompt, history, entries = factory()
            tokenized = runner.processor.tokenizer(prompt, return_tensors="pt")
            suffix = SRC.QA._suffix_from_tokenized(runner, tokenized).to(runner.model.device)
            selectors, guards = [], []
            with SRC.PilotCounts(runner) as counts, SRC.ThreadedReadTrace() as reads:
                result = H2.request_hyperinfer(
                    servers["base"], ctx, suffix, kv_prefetch=spec["kv_prefetch"], ratio=arm["ratio"],
                    probe=official["probe_heads"], sim_threshold=official["sim_threshold"], cold=False,
                    selector_out=selectors, debug_unread_nan=spec["nan"], guard_out=guards)
            returned = time.perf_counter()
        s = reads.summary()
        assert s["bytes"] == result["io"]["bytes"] and s["preads"] == result["io"]["preads"]
        sel = selectors[0]      # after the request, outside every timer
        result["layer_log"] = [dict(r) for r in sel.log]
        result["probe_sources"] = [[r["layer"], r["source"], r["ready_on_arrival"]] for r in sel.probe_log]
        result["selection_digest"] = selection_digest(sel.keep_masks, int(ctx.meta["num_layers"]))
        result["official_params"] = {"ratio": float(arm["ratio"]), "probe": official["probe_heads"], "sim_threshold": official["sim_threshold"]}
        result["hyperinfer"] = sel.hyperinfer_stats()
        if spec["nan"]:
            result["unread_row_guard"] = guards[0].result(sel.keep_masks)
        del selectors, guards
        result.update(projection_calls=counts.result(), scoring_calls=0,
                      end_to_end_ttft_ms=(result["first_token_at_s"] - started) * 1000, request_e2e_ms=(returned - started) * 1000,
                      request_started_at_s=started, vision_forward_count=guard.calls, page_cache_conditioning_ms=conditioning,
                      cap_reached=(len(result["generated_token_ids"]) == 16 and result["generated_token_ids"][-1] != runner.processor.tokenizer.eos_token_id),
                      peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                      os_read_trace=SRC.compact_trace(s["calls"], store_path, result["core_started_at_s"]),
                      pread_calls=s["pread_calls"], preadv_calls=s["preadv_calls"])
        ctx.cache.k = None; ctx.cache.v = None
        return result, {"prompt": prompt, "history": history, "history_entries": entries, "suffix_input_ids": suffix.cpu().tolist()}

    def audit(self, rows, manifest, config, phase, image_id, artifact=None):
        """Per-image audit of this run, from the journal rows only.  The stop check (gate 5) is judged by
        scripts/229, not here: an audit failure aborts the phase, a gate failure is a recorded result."""
        M = self.M
        methods = config["phases"][phase]["methods"]
        dialogs = [d for d in manifest["dialogues"] if d["image_id"] == image_id and d["dialog_id"] in set(manifest["phase_dialogue_ids"][phase])]
        failures = []

        def bad(check, **where):
            failures.append(dict(where, check=check))

        want = {(d["dialog_id"], m, t) for d in dialogs for m in methods for t in (1, 2, 3)}
        got = [(r["dialog_id"], r["method_id"], r["turn_id"]) for r in rows]
        if len(got) != len(set(got)) or set(got) != want:
            bad("request_set", expected=len(want), found=len(got))
        index = {(r["dialog_id"], r["method_id"], r["turn_id"]): r for r in rows}
        t1_same = True
        for d in dialogs:
            t1 = {tuple(index[(d["dialog_id"], m, 1)]["generated_token_ids"]) for m in methods if (d["dialog_id"], m, 1) in index}
            t1_same = t1_same and len(t1) == 1
            for m in methods:
                generated = {}
                for t in (1, 2, 3):
                    r = index.get((d["dialog_id"], m, t))
                    if r is None:
                        break
                    where = {"dialog_id": d["dialog_id"], "method_id": m, "turn_id": t}
                    q = d["turns"][t - 1]
                    if r["image_id"] != image_id or r["phase"] != phase or r["question_id"] != str(q["question_id"]) or r["gold"] != q["answers"]:
                        bad("identity", **where)
                    if r["score"] != M.strict_score(r["prediction"], q["answers"][0]):
                        bad("score", **where)
                    if r["prompt"] != self.prompt(methods, d, t, m, generated):
                        bad("prompt_history", **where)
                    if self.model == "next" and not (r["tokenization"] == "official" and r["official_token_ids_match"] is True):
                        bad("official_token_ids", **where)
                    if r["config_sha256"] != config["_file_sha256"] or r["manifest_sha256"] != config["_manifest_sha256"]:
                        bad("frozen_config", **where)
                    hit = t > 1
                    on = m in HYPER_ARMS and HYPER_ARMS[m]["kv_prefetch"] != "off"
                    kinds = r["probe_read_bytes"] + r["k_read_bytes"] + r["v_read_bytes"]
                    if hit != (r["ssd_read_bytes"] > 0) or kinds != r["ssd_read_bytes"] or r["packed_read_bytes"] != 0:
                        bad("read_accounting", **where)
                    if hit and r["pread_calls"] + r["preadv_calls"] != r["ssd_preads"]:
                        bad("read_accounting_calls", **where)
                    if hit and not on and r["k_read_bytes"] != r["v_read_bytes"]:
                        bad("read_accounting_kv", **where)
                    res = r["result"]
                    if hit and m != FULL and len(res["selection_digest"]["per_layer"]) != r["geometry"]["num_layers"]:
                        bad("selection_record", **where)
                    if hit and on:
                        h = res["hyperinfer"]
                        cols = h["layers"]
                        if h["prefetched_bytes"] + h["sync_bytes"] != r["k_read_bytes"] + r["v_read_bytes"]:
                            bad("prefetch_byte_accounting", **where)
                        if any(a + b != n or c + e != n for a, b, c, e, n in zip(cols["k_hit"], cols["k_missing"], cols["v_hit"], cols["v_missing"],
                                                                                   cols["needed_chunks"])):
                            bad("prefetch_chunk_accounting", **where)
                        if any(k - v not in (0, 1) for k, v in zip(cols["k_done"], cols["v_done"])):      # K then V, one read at a time
                            bad("prefetch_read_order", **where)
                        if cols["k_done"][0] != 0 or cols["v_done"][0] != 0 or cols["ended"][0] is not None:      # nothing is prefetched for the first layer
                            bad("prefetch_first_layer", **where)
                        if h["jobs"] != r["geometry"]["num_layers"] - 1 or h.get("truncation") != "reactive":
                            bad("prefetch_jobs", **where)
                    if hit and m in HYPER_ARMS and HYPER_ARMS[m]["nan"] and "unread_row_guard" not in res:
                        bad("unread_row_guard_missing", **where)
                    generated[t] = r
        return {"status": "PASS" if not failures else "FAIL", "failures": failures, "rows": len(rows), "methods": methods,
                "t1_tokens_identical_across_arms": bool(t1_same), "audit": "scripts/228 Backend.audit"}


# ------------------------------------------------------------------- freeze
def v1_state():
    """The earlier HyperInfer run: complete for both models, and its frozen source files as they were."""
    config = json.loads((V1_RUN / "config.json").read_text())
    for m in MODELS:
        done = json.loads((V1_RUN / MODEL_DIR[m] / "main_execution_complete.json").read_text())
        assert done["status"] == "PASS", m
    return config


def freeze(top):
    import transformers
    top = Path(top)
    assert top.parent == ROOT / "runs" and top.name.startswith(RUN_PREFIX) and CONTRACT.is_file() and PORT_TABLE.is_file()
    assert not (top / "config.json").exists(), "already frozen"
    refs = {m: json.loads((REF_RUN[m] / "config.json").read_text()) for m in MODELS}
    for m in MODELS:      # the earlier main runs are complete and their frozen files are unchanged
        assert json.loads((REF_RUN[m] / "main_execution_complete.json").read_text())["status"] == "PASS", m
        env = refs[m]["environment"]
        assert (transformers.__version__, torch.__version__, np.__version__, sys.executable) == (
            env["transformers"], env["torch"], env["numpy"], env["executable"]), f"not the environment of the {m} main run"
    v1 = v1_state()
    now = source_hashes()
    changed = {m: sorted(k for k, v in refs[m]["source_sha256"].items() if not REPORT_SCRIPT.match(k) and now.get(k) != v) for m in MODELS}
    changed["hyperinfer_v1"] = sorted(k for k, v in v1["source_sha256"].items() if not REPORT_SCRIPT.match(k) and now.get(k) != v)
    assert not any(changed.values()), ("a source file frozen by an earlier run changed", changed)
    top.mkdir(parents=True, exist_ok=True)
    results_dir(top).mkdir(parents=True, exist_ok=True)
    limit = int(os.environ.get("HYPERINFER_DEV_LIMIT", "0")) if "_dev_" in top.name else 0
    manifest = E1.build_manifest(limit)
    atomic_json(top / "manifest.json", manifest, replace=False)
    # ---- storage plan: store sizes the earlier main runs measured on the same images
    backends = {m: Backend(m, top) for m in MODELS}
    writes, largest = {}, 0
    for m in MODELS:
        B = backends[m]
        images = sorted({i for ph in PHASES for i in manifest["phase_images"][ph]})
        size = {k: E1.reference_store_bytes(m, k, images) for k in ("raster", B.NEW_KIND)}
        write = {"raster": lambda i, s=size: s["raster"][i], B.NEW_KIND: lambda i, s=size, k=B.NEW_KIND: 2 * s[k][i]}
        writes[m] = {ph: int(sum(write[k](i) for i in manifest["phase_images"][ph] for k in B.M.store_kinds(PHASE_METHODS[ph]))) for ph in PHASES}
        writes[m]["total"] = sum(writes[m].values())
        largest = max(largest, max(size["raster"][i] + size[B.NEW_KIND][i] for i in images))
    total_writes = sum(w["total"] for w in writes.values())
    free = shutil.disk_usage(top).free
    plan = {"policy": "FRESH_IMAGE_STREAMING (image-major): one live image; stores built from the arms' own T1 of the image's first dialogue -> "
                      "served -> audited -> committed -> run-local scratch deleted by exact SHA256 allowlist",
            "minimum_reserve_bytes": 30 * GiB, "peak_working_set_bytes": largest + 4 * GiB, "largest_image_bundle_bound_bytes": largest,
            "maximum_new_payload_bytes": 20 * GiB, "raw_and_evidence_growth_bound_bytes": 16 * GiB,
            "expected_ssd_write_bytes": writes, "expected_ssd_write_total_bytes": total_writes,
            "expected_ssd_write_note": "per image: the IMPRESS-layout store (written, then rewritten in the reordered layout) and, in the gate and dev phases, "
                                       "the FullLoad store; sizes from the earlier main run's disk_usage.json of the same image; rows and evidence not included",
            "free_bytes_at_freeze": free,
            "deletion_scope": "runs/<RUN>/<model>/scratch only (exact SHA allowlist after audit+commit; the stores of an interrupted image that cannot be "
                              "verified on resume); existing stores never deleted"}
    plan["required_free_bytes"] = plan["minimum_reserve_bytes"] + plan["peak_working_set_bytes"] + plan["raw_and_evidence_growth_bound_bytes"]
    plan["status"] = "PASS" if free >= plan["required_free_bytes"] else "BLOCKED_STORAGE"
    atomic_json(top / "storage_plan.json", plan, replace=False)
    if plan["status"] != "PASS":
        raise RuntimeError(f"BLOCKED_STORAGE: free {free} < required {plan['required_free_bytes']}")
    models = {}
    for m in MODELS:
        B = backends[m]
        sub = top / MODEL_DIR[m]; sub.mkdir()
        shutil.copyfile(top / "storage_plan.json", sub / "storage_plan.json")
        models[m] = {"label": MODEL_LABEL[m], "model": B.M.MODEL_ID, "revision": B.M.REVISION, "directory": MODEL_DIR[m],
                     "backend": f"scripts/{BACKEND_SCRIPT[m]}", "backend_sha256": sha(ROOT / "scripts" / BACKEND_SCRIPT[m]),
                     "reference_run": str(REF_RUN[m].relative_to(ROOT)), "reference_config_sha256": sha(REF_RUN[m] / "config.json"),
                     "tokenization": "official" if m == "next" else "fast tokenizer (the path of the LLaVA-1.5 main run)",
                     "impress_official": B.SRC.OFFICIAL, "server_args": B.M.SERVER_ARGS,
                     "arms": {a: dict(B.SRC.ARMS[a]) for a in GATE_METHODS},
                     "phases": {ph: {"methods": PHASE_METHODS[ph], "dialogues": len(manifest["phase_dialogue_ids"][ph]),
                                     "images": len(manifest["phase_images"][ph]),
                                     "tokenization": B.M.TOK.TOKENIZATION_OFFICIAL if m == "next" else None} for ph in PHASES}}
    config = {"schema_version": SCHEMA, "dataset": manifest["dataset"], "seed": 1234, "greedy": True, "max_new_tokens": 16, "batch_size": 1,
              "cpu_threads": 2, "history_policy": "method_local_generated", "metric": "strict normalized EM",
              "models": models, "model_order": list(MODELS), "phase_methods": PHASE_METHODS, "main_methods": MAIN_METHODS,
              "hyperinfer_arms": {k: dict(v) for k, v in HYPER_ARMS.items()},
              "hyperinfer": {"module": "mmimpress/hyperinfer_v2.py", "module_sha256": sha(ROOT / "mmimpress/hyperinfer_v2.py"),
                             "prediction": "every token layer l uses (selected image tokens + separators)",
                             "unit": "whole 64-token K and V chunks of layer l+1, ascending chunk id, one pread per read, K chunk then V chunk",
                             "truncation": "reactive (paper IV-C, Fig. 10(c)): no read is started after the GPU has finished layer l; the read in flight ends; "
                                           "no time budget, no chunk time constant, no half rule",
                             "gpu_done": "torch.cuda.synchronize() at the start of layer l+1's hook, then the stop flag under the job's lock",
                             "probe_read_ahead": "kept (the worker reads the next layer's probe block first; the request waits for it only)",
                             "worker": "one background thread; disk reads and host arrays only", "cache_tiers": "none; cold SSD; no warm-up"},
              "earlier_hyperinfer_run": {"run": str(V1_RUN.relative_to(ROOT)), "config_sha256": sha(V1_RUN / "config.json")},
              "arm_order": "cyclic rotation by the dialogue's ordinal in the dataset (ordinal % number of arms of the phase)",
              "source_sha256": now, "frozen_files_of_reference_runs_changed": changed,
              "contract_sha256": sha(CONTRACT), "port_table_sha256": sha(PORT_TABLE),
              "manifest_sha256": sha(top / "manifest.json"), "storage_plan_sha256": sha(top / "storage_plan.json"),
              "bootstrap": {"resamples": 10000, "seed": 1234, "unit": "image", "ci": 0.95},
              "resume_policy": "committed images skipped; inside an interrupted image complete dialogues are kept and the interrupted dialogue restarts "
                               "at turn 1; unverifiable stores -> the image restarts",
              "environment": {"python": sys.version, "executable": sys.executable, "torch": torch.__version__, "transformers": transformers.__version__,
                              "numpy": np.__version__}}
    atomic_json(top / "config.json", config, replace=False)
    shutil.copy2(CONTRACT, top / "CONTRACT.md"); shutil.copy2(PORT_TABLE, top / "PORT_TABLE.md")
    atomic_json(top / "environment_before.json", {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
                                                  "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
                                                  "pip_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)})
    atomic_json(top / "protection_before.json", backends["next"].M.stat_inventory(top, top / "protection_before.tsv.gz"), replace=False)
    seconds = {m: {ph: manifest["expected_requests"][ph] * SECONDS_PER_REQUEST_PRIOR[m][ph] for ph in PHASES} for m in MODELS}
    total_s = sum(sum(v.values()) for v in seconds.values())
    prerun = {"written_before_any_measurement_utc": RAW_STATUS.utc(time.time()), "run": top.name,
              "expected_requests": {m: manifest["expected_requests"] for m in MODELS},
              "expected_requests_total": 2 * sum(manifest["expected_requests"].values()),
              "expected_ssd_write": {"bytes": writes, "total_tb": total_writes / 1e12, "note": plan["expected_ssd_write_note"]},
              "expected_duration": {"seconds_per_request_prior": SECONDS_PER_REQUEST_PRIOR, "phase_hours": {m: {ph: s / 3600 for ph, s in v.items()} for m, v in seconds.items()},
                                    "total_hours": total_s / 3600 + 0.5, "basis": "seconds per request of the earlier HyperInfer run's phases; unit tests, gate checks and the report add about 30 min"},
              "free_space": {"free_bytes": free, "free_gib": free / GiB, "required_free_bytes": plan["required_free_bytes"]},
              "smart": RAW_STATUS.smart_snapshot(), "ssd_temperature_at_freeze": RAW_STATUS.ssd_temperature()}
    atomic_json(results_dir(top) / "prerun.json", prerun)
    RAW_STATUS.update(top, expected_requests=prerun["expected_requests_total"], completed_requests=0,
                      expected_ssd_write_tb=round(total_writes / 1e12, 3), expected_total_hours=round(prerun["expected_duration"]["total_hours"], 1),
                      eta_utc=RAW_STATUS.utc(time.time() + total_s + 1800), eta_note="pre-run estimate (see prerun.json)")
    print(json.dumps({"freeze": "PASS", "run": str(top), "expected": manifest["expected_requests"], "free_gib": free / GiB,
                      "expected_ssd_write_tb": total_writes / 1e12, "expected_hours": prerun["expected_duration"]["total_hours"]}), flush=True)


def verify_freeze(top):
    import transformers
    top = Path(top)
    config = json.loads((top / "config.json").read_text())
    now = source_hashes()
    changed = sorted(k for k, v in config["source_sha256"].items() if now.get(k) != v)
    if changed and not ("_dev_" in top.name and os.environ.get("HYPERINFER_DEV_ALLOW_SOURCE_DRIFT") == "1"):
        raise RuntimeError("FROZEN_SOURCE_CONFIG_MISMATCH: " + repr(changed[:10]))
    if "_dev_" not in top.name and (sha(CONTRACT) != config["contract_sha256"] or sha(PORT_TABLE) != config["port_table_sha256"]):
        raise RuntimeError("FROZEN_CONTRACT_MISMATCH")
    if sha(top / "manifest.json") != config["manifest_sha256"] or sha(top / "storage_plan.json") != config["storage_plan_sha256"]:
        raise RuntimeError("FROZEN_MANIFEST_OR_STORAGE_PLAN_MISMATCH")
    if config["phase_methods"] != PHASE_METHODS or config["hyperinfer_arms"] != {k: dict(v) for k, v in HYPER_ARMS.items()}:
        raise RuntimeError("FROZEN_ARMS_MISMATCH")
    if transformers.__version__ != config["environment"]["transformers"] or torch.__version__ != config["environment"]["torch"]:
        raise RuntimeError("FROZEN_ENVIRONMENT_MISMATCH")
    manifest = json.loads((top / "manifest.json").read_text())
    config["_file_sha256"] = sha(top / "config.json"); config["_manifest_sha256"] = sha(top / "manifest.json")
    manifest["_ordinals"] = {d["dialog_id"]: i for i, d in enumerate(manifest["dialogues"])}
    return config, manifest


def run_stage(top, model, phase, resume):
    top = Path(top)
    config, manifest = verify_freeze(top)
    sub = top / MODEL_DIR[model]
    if phase == "main":      # the main phase is admitted only by the recorded gate decision of this model
        admit = json.loads((top / "gates" / f"{model}_admission.json").read_text())
        if admit["status"] != "PASS" or admit["config_sha256"] != config["_file_sha256"]:
            raise RuntimeError("MAIN_NOT_ADMITTED")
    if (sub / phase).exists() and not resume:
        raise FileExistsError("use --resume; previous artifacts preserved")
    B = Backend(model, top, expected={m: manifest["expected_requests"] for m in MODELS})
    M = B.M
    M.assert_gpu_exclusive()
    cfg = E1.model_config(config, model)
    runner, servers = B.load(cfg)
    try:
        atomic_json(sub / f"{phase}_start_{time.strftime('%Y%m%dT%H%M%S')}.json",
                    {"started_unix": time.time(), "pid": os.getpid(), "resume": bool(resume), "boot_id": RAW_STATUS.boot_id(), "model": model,
                     "idle_synchronize_cost": H2.idle_synchronize_cost(), "ssd_temperature": RAW_STATUS.ssd_temperature()})
        receipt = M.run_phase(runner, servers, sub, phase, cfg, manifest)
        atomic_json(sub / f"{phase}_execution_complete.json", {"status": "PASS", **receipt, "finished_unix": time.time()})
    except BaseException as exc:
        error = {"status": "PARTIAL", "error": repr(exc), "traceback": traceback.format_exc(), "at_unix": time.time(), "model": model,
                 "resume_command": f"{DRIVER} {top.relative_to(ROOT)}"}
        atomic_json(sub / ("interruption_" + phase + "_" + time.strftime("%Y%m%dT%H%M%S") + ".json"), error)
        RAW_STATUS.update(top, last_error={"stage": f"{model}:{phase}_phase", "error": repr(exc)[:800], "at_utc": RAW_STATUS.utc(time.time())})
        if any(marker in repr(exc) for marker in NO_RETRY_MARKERS):      # a restart would stop at the same place
            atomic_json(sub / f"NO_RETRY_{phase}.json", {"error": repr(exc)[:2000], "at_unix": time.time()})
        print(json.dumps({k: v for k, v in error.items() if k != "traceback"}), flush=True)
        raise
    finally:
        for server in servers.values():
            server.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True, help="runs/mt_gqa_hyperinfer_v2_<UTC>")
    p.add_argument("--stage", choices=("freeze",) + PHASES, required=True)
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    top = args.run.resolve()
    assert top.parent == ROOT / "runs" and top.name.startswith(RUN_PREFIX)
    if args.stage == "freeze":
        return freeze(top)
    return run_stage(top, args.model, args.stage, args.resume)


if __name__ == "__main__":
    main()

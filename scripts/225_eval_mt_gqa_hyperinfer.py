#!/usr/bin/env python3
"""MT-GQA x {LLaVA-NeXT, LLaVA-1.5}: HyperInfer (contract docs/mt_gqa_hyperinfer_contract.md).

Only HyperInfer is measured in the main phase.  ReComp, FullLoad, IMPRESS and Ours are the rows of
the earlier main runs (LLaVA-NeXT: runs/mt_gqa_main_v2_20261124T051939Z, LLaVA-1.5:
runs/llava15_mt_gqa_main_20261120T165001Z); the report pairs them by image and turn.

Where the code comes from

* The per-model harness of those two runs is imported and used as it is: scripts/222 (LLaVA-NeXT,
  original-LLaVA token ids) and scripts/206 (LLaVA-1.5).  Their ``run_phase`` / ``run_image``
  (image-major lifecycle, journals, resume, commit, scratch cleanup), pixel requests, store
  writers, activation, ``make_row`` and the FullLoad / IMPRESS request calls run unchanged.
* What this script adds to the imported module, in this process only: the HyperInfer arms in
  its arm table, a dispatcher in front of its stored request (HyperInfer arms -> ``hyper_request``
  below, every other arm -> the module's own function), this run's phases, this run's per-image
  audit and this run's status.json writer.  No file of those harnesses is edited.
* HyperInfer itself: ``mmimpress/hyperinfer.py`` (IMPRESS port + next-layer KV prefetch).

Arms
  fullload_async            FullLoad (gate 8)
  impress_official_async    IMPRESS  (gates 2, 3, 8; the I/O ratio of the report)
  hyperinfer_async          HyperInfer -- the only arm of the main phase
  hyperinfer_off            HyperInfer with the prefetch switched off      (gate 2)
  hyperinfer_sync           prefetch executed on the request thread        (gate 6)
  hyperinfer_async_repeat   HyperInfer executed a second time              (gate 6)
  hyperinfer_async_nan      HyperInfer, rows that were not read hold NaN   (gate 5)

Phases, per model: gate (4 images, 7 arms) -> dev (DEV 40, 7 arms) -> main (398 images, HyperInfer).

Stages
  --stage calibrate --model M   cold-SSD read time of one K chunk and one V chunk (images outside MT-GQA)
  --stage constants             both calibrations -> docs/mt_gqa_hyperinfer_chunk_times.json + the contract's block
  --stage freeze                config, manifest, storage plan, protection inventory
  --stage gate|dev|main --model M [--resume]
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import importlib.util
import json
import os
import re
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

from mmimpress import hyperinfer as HI  # noqa: E402
from mmimpress import impress_official  # noqa: E402
import llava_next_main_status as RAW_STATUS  # noqa: E402  (status.json / SSD counters; keyed by run name)

SCHEMA = "mt-gqa-hyperinfer-v1"
RUN_PREFIX = "mt_gqa_hyperinfer_"
CONTRACT = ROOT / "docs/mt_gqa_hyperinfer_contract.md"
PORT_TABLE = ROOT / "docs/hyperinfer_port_contract.md"
CONSTANTS = ROOT / "docs/mt_gqa_hyperinfer_chunk_times.json"
DRIVER = "scripts/run_mt_gqa_hyperinfer.sh"
BLOCK_BEGIN, BLOCK_END = "<!-- CHUNK_TIME_CONSTANTS:BEGIN -->", "<!-- CHUNK_TIME_CONSTANTS:END -->"
MODELS = ("next", "l15")
MODEL_DIR = {"next": "llava_next", "l15": "llava15"}
MODEL_LABEL = {"next": "LLaVA-NeXT-7B", "l15": "LLaVA-1.5-7B"}
BACKEND_SCRIPT = {"next": "222_eval_mt_gqa_main_v2.py", "l15": "206_eval_llava15_mt_gqa_main.py"}
REF_RUN = {"next": ROOT / "runs/mt_gqa_main_v2_20261124T051939Z", "l15": ROOT / "runs/llava15_mt_gqa_main_20261120T165001Z"}
OFFICIAL_COMMIT = impress_official.OFFICIAL_COMMIT
GiB = 1 << 30

FULL, IMPRESS = "fullload_async", "impress_official_async"
HYPER, HYPER_OFF, HYPER_SYNC = "hyperinfer_async", "hyperinfer_off", "hyperinfer_sync"
HYPER_REPEAT, HYPER_NAN = "hyperinfer_async_repeat", "hyperinfer_async_nan"
HYPER_ARMS = OrderedDict([
    (HYPER,        {"kv_prefetch": "async", "nan": False, "display": "HyperInfer"}),
    (HYPER_OFF,    {"kv_prefetch": "off",   "nan": False, "display": "HyperInfer, prefetch 끔"}),
    (HYPER_SYNC,   {"kv_prefetch": "sync",  "nan": False, "display": "HyperInfer, prefetch 동기 실행"}),
    (HYPER_REPEAT, {"kv_prefetch": "async", "nan": False, "display": "HyperInfer, 두 번째 실행"}),
    (HYPER_NAN,    {"kv_prefetch": "async", "nan": True,  "display": "HyperInfer, 읽지 않은 행 NaN"}),
])
GATE_METHODS = [FULL, IMPRESS, HYPER_OFF, HYPER, HYPER_SYNC, HYPER_REPEAT, HYPER_NAN]
MAIN_METHODS = [HYPER]
PHASE_METHODS = {"gate": list(GATE_METHODS), "dev": list(GATE_METHODS), "main": list(MAIN_METHODS)}
PHASES = ("gate", "dev", "main")
PHASE_IMAGES = {"gate": "gate", "dev": "dev", "main": "main"}
EXPECTED_REQUESTS = {"gate": 49 * 3 * 7, "dev": 378 * 3 * 7, "main": 12183}
# seconds per request before a phase has measured its own rate (status.json estimate only)
SECONDS_PER_REQUEST_PRIOR = {"next": {"gate": 0.80, "dev": 0.80, "main": 1.00}, "l15": {"gate": 0.40, "dev": 0.40, "main": 0.50}}
NO_RETRY_MARKERS = ("OFFICIAL_TOKEN_IDS_MISMATCH", "REQUEST_IDS_DIFFER_FROM_TOKENIZER_INPUT_IDS", "INDEPENDENT_IMAGE_AUDIT_FAIL",
                    "BLOCKED_STORAGE", "FROZEN_")
# chunk read time calibration: images outside MT-GQA (COCO val2014 images of the MT-VQA set), first dialogue order
CALIB_SOURCE = ROOT / "data/mt_vqa_v2_reconstructed_25k/dialogues.json"
CALIB_IMAGES = 4
CALIB_PASSES = 3
CALIB_QUESTION = "What is shown in the image?"
REPORT_SCRIPT = re.compile(r"^scripts/\d+_report_.*\.py$")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def source_hashes():
    paths = (list((ROOT / "mmimpress").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")) + list((ROOT / "tests").glob("test_*.py")))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths) if not REPORT_SCRIPT.match(str(p.relative_to(ROOT)))}


def atomic_json(path, value, replace=True):
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=1, ensure_ascii=False, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def results_dir(top):
    return ROOT / "results" / Path(top).name


def selection_digest(keep_masks, layers):
    """Per layer a short digest of the used set (stored positions), plus one digest of all layers."""
    per = []
    whole = hashlib.sha256()
    for li in range(layers):
        blob = np.packbits(keep_masks[li].numpy().astype(np.uint8)).tobytes()
        per.append(hashlib.sha256(blob).hexdigest()[:12])
        whole.update(blob)
    return {"per_layer": per, "all_layers": whole.hexdigest(), "used_per_layer": [int(keep_masks[li].sum()) for li in range(layers)]}


# =================================================================== status
class StatusAdapter:
    """What the imported harness calls ``STATUS``: everything goes to the one status.json of this run
    (results/<run name>/status.json), the model's own numbers under ``models[<model>]``."""

    utc = staticmethod(RAW_STATUS.utc)
    boot_id = staticmethod(RAW_STATUS.boot_id)
    ssd_temperature = staticmethod(RAW_STATUS.ssd_temperature)
    smart_snapshot = staticmethod(RAW_STATUS.smart_snapshot)
    sectors = staticmethod(RAW_STATUS.sectors)

    def __init__(self, model, top, expected):
        self.model, self.top, self.expected = model, Path(top), expected      # expected: {model: {phase: requests}}

    def _other_seconds(self, doc):
        """Estimated seconds of the models that have not finished and are not the current one."""
        total = 0.0
        for m in MODELS:
            if m == self.model:
                continue
            if doc.get(f"main_{m}") == "complete" or doc.get(f"admission_{m}") == "FAIL":
                continue
            if MODELS.index(m) < MODELS.index(self.model):
                continue
            total += sum(self.expected[m][ph] * SECONDS_PER_REQUEST_PRIOR[m][ph] for ph in PHASES)
        return total

    def update(self, run, **fields):
        doc = RAW_STATUS.read(self.top)
        models = doc.get("models") or {}
        mine = dict(models.get(self.model) or {})
        mine.update({k: v for k, v in fields.items() if k not in ("ssd_written_bytes", "ssd_written_gb", "ssd_written_note", "ssd_temperature_c",
                                                                 "free_gib", "last_error", "python_pid", "eta_note")})
        models[self.model] = mine
        out = {"models": models}
        if "stage" in fields:
            out.update(model=MODEL_LABEL[self.model], stage=f"{self.model}:{fields['stage']}")
        done = sum(int((models.get(m) or {}).get("completed_requests") or 0) for m in MODELS)
        out.update(completed_requests=done, expected_requests=sum(sum(self.expected[m].values()) for m in MODELS))
        if fields.get("eta_utc"):
            sub_eta = calendar.timegm(time.strptime(fields["eta_utc"], "%Y-%m-%dT%H:%M:%SZ"))
            out.update(eta_utc=RAW_STATUS.utc(sub_eta + self._other_seconds(doc)),
                       eta_note="this model at the rate measured in this phase (later phases: prior rates), plus the prior estimate of a model not started yet")
        for k in ("ssd_written_bytes", "ssd_written_gb", "ssd_written_note", "ssd_temperature_c", "free_gib", "last_error", "python_pid"):
            if k in fields:
                out[k] = fields[k]
        if "ssd_written_gb" not in out:
            state = RAW_STATUS.ssd_counter(self.top)
            out.update(ssd_written_bytes=state.get("written_bytes"), ssd_written_gb=round((state.get("written_bytes") or 0) / 1e9, 2),
                       ssd_written_note="device-wide kernel counter since the driver first started (all processes); not NAND writes",
                       ssd_temperature_c=RAW_STATUS.ssd_temperature()["composite_c"], free_gib=round(shutil.disk_usage(self.top).free / GiB, 1))
        return RAW_STATUS.update(self.top, **out)

    def ssd_counter(self, run, label=None):
        return RAW_STATUS.ssd_counter(self.top, label)

    def ssd_sample(self, run, label, **extra):
        return RAW_STATUS.ssd_sample(self.top, label, model=self.model, **extra)


# ================================================================== backend
class Backend:
    """One model's imported harness with this run's arms, phases, audit and status attached."""

    def __init__(self, model, top, constants=None, expected=None):
        self.model, self.top = model, Path(top)
        M = self.M = load_script(f"_mt_gqa_hyperinfer_backend_{model}", BACKEND_SCRIPT[model])
        # the module that defines the stored request (and the names it uses)
        self.SRC = M.N if model == "next" else M
        arms = self.SRC.ARMS
        assert arms is M.ARMS and FULL in arms and arms[IMPRESS]["family"] == "impress_official"
        self.NEW_KIND = M.NEW_KIND
        for name, spec in HYPER_ARMS.items():
            arms[name] = {"family": "hyperinfer", "kind": self.NEW_KIND, "async": spec["kv_prefetch"] != "sync",
                          "display": spec["display"], "ratio": float(impress_official.KEEP_RATIO)}
        M.PHASES, M.PHASE_METHODS, M.MAIN_METHODS, M.EXPECTED_REQUESTS = PHASES, PHASE_METHODS, MAIN_METHODS, EXPECTED_REQUESTS
        M.SCHEMA = SCHEMA
        if model == "next":
            M.N.SCHEMA = SCHEMA
            M.SECONDS_PER_REQUEST_PRIOR = dict(SECONDS_PER_REQUEST_PRIOR[model])
            M.DRIVER = DRIVER
        self.status = StatusAdapter(model, top, expected or {m: dict(EXPECTED_REQUESTS) for m in MODELS})
        M.STATUS = self.status
        M.audit_image_rows = self.audit
        self._stored_request = self.SRC.stored_request
        self.SRC.stored_request = self.dispatch
        self.constants = constants
        self.states = {}

    # ---------------------------------------------------------------- model
    def load(self, config):
        M = self.M
        if self.model == "next":
            return M.load_model(config)
        torch.set_num_threads(config["cpu_threads"]); torch.manual_seed(1234); np.random.seed(1234)
        torch.set_grad_enabled(False)
        runner, servers, self.warmup = M.load_servers()
        return runner, servers

    def factory(self, methods, dialog, turn, method, generated):
        M = self.M
        if self.model == "next":
            return M.PromptFactory(methods, dialog, turn, method, generated, M.TOK.TOKENIZATION_OFFICIAL)
        return lambda: M.S.prompt_factory(methods, dialog, turn, method, generated)

    def prompt(self, methods, dialog, turn, method, generated):
        """The prompt string of a request, recomputed (audit)."""
        M = self.M
        fn = M.H1.prompt_factory if self.model == "next" else M.S.prompt_factory
        return fn(methods, dialog, turn, method, generated)[0]

    def pixel(self, runner, servers, image_path, factory, capture):
        M = self.M
        if self.model == "next":
            return M.pixel_request(runner, servers, image_path, factory, capture)
        return M.normal_request(runner, servers["base"], image_path, factory,
                                "raster" if capture in M.FULL_KINDS else "none", with_stats=capture == self.NEW_KIND)

    def persist(self, runner, servers, result, diag, path, image_id, kind, source, stats_path):
        M = self.M
        if self.model == "next":
            return M.N.persist(runner, result, diag, path, image_id, kind, source, {}, stats_path)
        return M.persist(runner, servers["base"], result, diag, path, image_id, kind, source, {}, stats_path)

    def activate(self, runner, kind, path):
        M = self.M
        return M.N.activate(runner, kind, path) if self.model == "next" else M.activate(runner, kind, path, {})

    # ------------------------------------------------------------- requests
    def dispatch(self, runner, servers, ctx, factory, method, image_id, store_path, *args, **kwargs):
        """Every stored request of the imported ``run_image`` passes here."""
        if method in HYPER_ARMS:
            return self.hyper_request(runner, servers, ctx, factory, method, image_id, store_path)
        if self.SRC.ARMS[method]["family"] != "impress_official":
            return self._stored_request(runner, servers, ctx, factory, method, image_id, store_path, *args, **kwargs)
        # IMPRESS: the module's own request; the selector it creates is kept so that the used sets can be
        # recorded after the request (a pass-through around ``request_official``, restored at once)
        seen = []
        original = impress_official.request_official

        def recorder(*a, **k):
            out = original(*a, **k)
            seen.append(k.get("selector_out"))
            return out

        impress_official.request_official = recorder
        try:
            result, diag = self._stored_request(runner, servers, ctx, factory, method, image_id, store_path, *args, **kwargs)
        finally:
            impress_official.request_official = original
        result["selection_digest"] = selection_digest(seen[0][0].keep_masks, int(ctx.meta["num_layers"]))
        return result, diag

    def hyper_hit(self, runner, servers, ctx, factory, method, image_id, store_path):
        """``stored_hit`` of the imported harness for the HyperInfer arms: the same conditioning, timer
        start, prompt construction and tokenisation inside the timer, counters and read trace."""
        SRC, spec = self.SRC, HYPER_ARMS[method]
        arm = SRC.ARMS[method]
        official = SRC.OFFICIAL
        on = spec["kv_prefetch"] != "off"
        state = None
        if on:
            if method not in self.states:      # one budget state per arm, for the life of this process (the public code's model-level variable)
                self.states[method] = HI.PrefetchState(self.constants["k_chunk_s"], self.constants["v_chunk_s"])
            state = self.states[method]
            HI.position_maps_for(ctx)          # store metadata tables, built once per activated store, outside the timer
        with SRC.QA._NoVisionForward(runner) as guard:
            c0 = time.perf_counter(); ctx.reader.drop_all(); conditioning = (time.perf_counter() - c0) * 1000
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
            prompt, history, entries = factory()
            tokenized = runner.processor.tokenizer(prompt, return_tensors="pt")
            suffix = SRC.QA._suffix_from_tokenized(runner, tokenized).to(runner.model.device)
            selectors, guards = [], []
            with SRC.PilotCounts(runner) as counts, SRC.ThreadedReadTrace() as reads:
                result = HI.request_hyperinfer(
                    servers["base"], ctx, suffix, state=state, kv_prefetch=spec["kv_prefetch"], ratio=arm["ratio"],
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

    def hyper_request(self, runner, servers, ctx, factory, method, image_id, store_path):
        """``stored_request`` of the imported harness (the checks around the request are the same)."""
        from mmimpress.serve import BIAS
        import transformers.models.llama.modeling_llama as ml
        SRC = self.SRC
        original = [l.self_attn.forward for l in runner.layers]
        hooks_before = [len(l._forward_pre_hooks) for l in runner.layers]
        eager = ml.eager_attention_forward
        with SRC.conditioning_trace() as conditioning:
            result, diag = self.hyper_hit(runner, servers, ctx, factory, method, image_id, store_path)
        assert all(l.self_attn.forward == f for l, f in zip(runner.layers, original)), "adapter not restored"
        assert [len(l._forward_pre_hooks) for l in runner.layers] == hooks_before, "layer hook left behind"
        assert ml.eager_attention_forward == eager, "attention function not restored"
        assert not BIAS, "stale mask after request"
        assert conditioning and all(e["status"] == "SUCCESS" for e in conditioning), "page cache conditioning not verified"
        root = Path(store_path).resolve()
        result.update(conditioned_files=sorted(Path(e["path"]).relative_to(root).as_posix() for e in conditioning),
                      adapter_restored=True, page_cache_conditioning_method="posix_fadvise_DONTNEED",
                      page_cache_conditioning_excluded_from_ttft=True)
        return result, diag

    # ---------------------------------------------------------------- audit
    def audit(self, rows, manifest, config, phase, image_id, artifact=None):
        """Per-image audit of this run, from the journal rows only."""
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
                    kinds = r["probe_read_bytes"] + r["k_read_bytes"] + r["v_read_bytes"]
                    if hit != (r["ssd_read_bytes"] > 0) or kinds != r["ssd_read_bytes"] or r["packed_read_bytes"] != 0:
                        bad("read_accounting", **where)
                    if hit and (r["k_read_bytes"] != r["v_read_bytes"] or r["pread_calls"] + r["preadv_calls"] != r["ssd_preads"]):
                        bad("read_accounting_kv", **where)
                    res = r["result"]
                    if hit and m != FULL and len(res["selection_digest"]["per_layer"]) != r["geometry"]["num_layers"]:
                        bad("selection_record", **where)
                    if hit and m in HYPER_ARMS and HYPER_ARMS[m]["kv_prefetch"] != "off":
                        h = res["hyperinfer"]
                        if h["prefetched_bytes"] + h["sync_bytes"] != r["k_read_bytes"] + r["v_read_bytes"]:
                            bad("prefetch_byte_accounting", **where)
                        cols = h["layers"]
                        if any(a + b != c for a, b, c in zip(cols["hit_chunks"], cols["missing_chunks"], cols["needed_chunks"])):
                            bad("prefetch_chunk_accounting", **where)
                        if cols["prefetched_chunks"][0] != 0 or cols["rule"][0] is not None:      # nothing is prefetched for the first layer
                            bad("prefetch_first_layer", **where)
                    if hit and m in HYPER_ARMS and HYPER_ARMS[m]["nan"] and "unread_row_guard" not in res:
                        bad("unread_row_guard_missing", **where)
                    generated[t] = r
        return {"status": "PASS" if not failures else "FAIL", "failures": failures, "rows": len(rows), "methods": methods,
                "t1_tokens_identical_across_arms": bool(t1_same), "audit": "scripts/225 Backend.audit"}


# ============================================================== calibration
def calibration_images():
    doc = json.loads(CALIB_SOURCE.read_text())
    out = []
    for d in doc["dialogues"]:
        if d["image_path"] not in [x["image_path"] for x in out]:
            out.append({"image_id": f"calib_{d['image_id']}", "image_path": d["image_path"], "sha256": sha(ROOT / d["image_path"])})
        if len(out) == CALIB_IMAGES:
            break
    return out


def measure_chunk_times(ctx, passes=CALIB_PASSES):
    """Cold read time of every chunk of every layer, read the way the prefetch worker reads:
    ascending chunk ids, one pread per chunk, K then V; the store's page cache is dropped before each pass."""
    m = ctx.meta
    L, nc, cs, vn = int(m["num_layers"]), int(m["n_chunks_per_layer"]), int(m["chunk_size"]), int(m["v_token_num"])
    full = [c for c in range(nc) if (c + 1) * cs <= vn]
    samples = []
    for p in range(passes):
        ctx.reader.drop_all()
        for li in range(L):
            for cid in range(nc):
                a = time.perf_counter()
                HI.read_one_chunk(ctx.reader, li, "k", cid)
                b = time.perf_counter()
                HI.read_one_chunk(ctx.reader, li, "v", cid)
                c = time.perf_counter()
                if cid in full:
                    samples.append((p, li, cid, b - a, c - b))
    return samples, {"layers": L, "chunks_per_layer": nc, "full_chunks_per_layer": len(full), "chunk_tokens": cs,
                     "chunk_bytes": cs * int(m["num_heads"]) * int(m["head_dim"]) * ctx.reader.itemsize, "v_token_num": vn}


def calibrate(top, model):
    out_path = results_dir(top) / f"calibration_{model}.json"
    if out_path.exists():
        print(json.dumps({"calibrate": "SKIP (recorded)", "model": model})); return
    assert not (Path(top) / "config.json").exists(), "the run is already frozen"
    B = Backend(model, top)
    B.M.assert_gpu_exclusive()
    runner, servers = B.load({"cpu_threads": 2})
    scratch = ROOT / "runs" / (Path(top).name + ".calibration") / MODEL_DIR[model]
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    images = calibration_images()
    gqa = {Path(d["image_path"]).name for d in json.loads((REF_RUN[model] / "manifest.json").read_text())["dialogues"]}
    assert not any(Path(i["image_path"]).name in gqa for i in images), "a calibration image belongs to MT-GQA"
    per_image, k_all, v_all = [], [], []
    try:
        for i, image in enumerate(images):
            dialog = {"dialog_id": f"calib_{i:02d}", "image_id": image["image_id"], "image_path": image["image_path"],
                      "turns": [{"question": CALIB_QUESTION, "answers": ["unknown"], "question_id": f"calib_{i:02d}_{t}", "turn_id": t} for t in (1, 2, 3)]}
            result, diag = B.pixel(runner, servers, ROOT / image["image_path"], B.factory([HYPER], dialog, 1, HYPER, {}), B.NEW_KIND)
            path = scratch / image["image_id"]
            source = {"dialog_id": dialog["dialog_id"], "method_id": HYPER, "physical_execution_id": f"calibration-{i}", "logical_request_id": f"calibration-{i}"}
            B.persist(runner, servers, result, diag, path, image["image_id"], B.NEW_KIND, source, scratch / f"{image['image_id']}_turn1_stats.npz")
            del result, diag
            ctx = B.activate(runner, B.NEW_KIND, path)
            samples, geometry = measure_chunk_times(ctx)
            ctx.close()
            k = np.array([s[3] for s in samples]); v = np.array([s[4] for s in samples])
            first = [s for s in samples if s[2] == 0]
            per_image.append(dict(image, geometry=geometry, samples=len(samples),
                                  k_ms={"mean": 1e3 * k.mean(), "median": 1e3 * float(np.median(k)), "p10": 1e3 * float(np.percentile(k, 10)), "p90": 1e3 * float(np.percentile(k, 90))},
                                  v_ms={"mean": 1e3 * v.mean(), "median": 1e3 * float(np.median(v)), "p10": 1e3 * float(np.percentile(v, 10)), "p90": 1e3 * float(np.percentile(v, 90))},
                                  first_chunk_of_layer_ms={"k_mean": 1e3 * float(np.mean([s[3] for s in first])), "v_mean": 1e3 * float(np.mean([s[4] for s in first]))},
                                  by_chunk_position_ms={"k": [1e3 * float(np.mean([s[3] for s in samples if s[2] == c])) for c in sorted({s[2] for s in samples})][:16],
                                                        "v": [1e3 * float(np.mean([s[4] for s in samples if s[2] == c])) for c in sorted({s[2] for s in samples})][:16]}))
            k_all.extend(k.tolist()); v_all.extend(v.tolist())
            shutil.rmtree(path)
    finally:
        for server in servers.values():
            server.close()
        shutil.rmtree(scratch.parent, ignore_errors=True)
    k_all, v_all = np.array(k_all), np.array(v_all)
    doc = {"model": model, "model_label": MODEL_LABEL[model], "measured_utc": RAW_STATUS.utc(time.time()),
           "definition": "mean wall time of one cold whole-chunk read (64 tokens; one os.pread through the store reader + numpy view), K and V separately; "
                         "every full chunk of every layer, ascending chunk ids, K then V -- the read loop of the prefetch worker; page cache of the "
                         "store dropped (posix_fadvise DONTNEED) before each pass; GPU idle; images outside MT-GQA",
           "passes_per_image": CALIB_PASSES, "images": per_image, "samples": int(k_all.size),
           "k_chunk_s": float(k_all.mean()), "v_chunk_s": float(v_all.mean()),
           "k_chunk_ms": {"mean": 1e3 * float(k_all.mean()), "median": 1e3 * float(np.median(k_all)), "p10": 1e3 * float(np.percentile(k_all, 10)), "p90": 1e3 * float(np.percentile(k_all, 90))},
           "v_chunk_ms": {"mean": 1e3 * float(v_all.mean()), "median": 1e3 * float(np.median(v_all)), "p10": 1e3 * float(np.percentile(v_all, 10)), "p90": 1e3 * float(np.percentile(v_all, 90))},
           "public_code_constant_ms": {"disk": 0.40818, "cpu": 0.08312, "cuda:0": 0.06998},
           "device": RAW_STATUS.block_device()[0], "ssd_temperature": RAW_STATUS.ssd_temperature(), "boot_id": RAW_STATUS.boot_id()}
    results_dir(top).mkdir(parents=True, exist_ok=True)
    atomic_json(out_path, doc, replace=False)
    print(json.dumps({"calibrate": "PASS", "model": model, "k_chunk_ms": doc["k_chunk_ms"], "v_chunk_ms": doc["v_chunk_ms"], "samples": doc["samples"]}), flush=True)


def constants_block(doc):
    lines = [BLOCK_BEGIN,
             "| 모델 | K chunk 읽기 시간 (ms) | V chunk 읽기 시간 (ms) | 표본 수 | 측정 시각 (UTC) |",
             "|---|---|---|---|---|"]
    for m in MODELS:
        c = doc["models"][m]
        lines.append(f"| {MODEL_LABEL[m]} | {1e3 * c['k_chunk_s']:.5f} | {1e3 * c['v_chunk_s']:.5f} | {c['samples']:,} | {c['measured_utc']} |")
    lines += ["", f"상수 파일: `docs/{CONSTANTS.name}` (SHA256은 `config.json`의 `constants_sha256`). 값은 표본 평균이다. 공개 코드의 디스크 상수는 0.40818 ms다.", BLOCK_END]
    return "\n".join(lines)


def write_constants(top):
    """Both calibrations -> the constants file and the contract's block (before the freeze)."""
    assert not (Path(top) / "config.json").exists(), "the run is already frozen"
    models = {}
    for m in MODELS:
        c = json.loads((results_dir(top) / f"calibration_{m}.json").read_text())
        models[m] = {"k_chunk_s": c["k_chunk_s"], "v_chunk_s": c["v_chunk_s"], "samples": c["samples"], "measured_utc": c["measured_utc"],
                     "calibration_sha256": sha(results_dir(top) / f"calibration_{m}.json")}
    doc = {"schema": "mt-gqa-hyperinfer-chunk-times-v1", "run": Path(top).name, "models": models,
           "definition": "seconds; mean cold whole-chunk read time on this machine's SSD (see the calibration files)"}
    atomic_json(CONSTANTS, doc)
    text = CONTRACT.read_text()
    a, b = text.index(BLOCK_BEGIN), text.index(BLOCK_END) + len(BLOCK_END)
    CONTRACT.write_text(text[:a] + constants_block(doc) + text[b:])
    print(json.dumps({"constants": "PASS", "models": {m: {"k_ms": 1e3 * v["k_chunk_s"], "v_ms": 1e3 * v["v_chunk_s"]} for m, v in models.items()}}), flush=True)


# ------------------------------------------------------------------- freeze
def build_manifest(limit=0):
    refs = {m: json.loads((REF_RUN[m] / "manifest.json").read_text()) for m in MODELS}
    ds = refs["next"]["dialogues"]
    assert refs["l15"]["dialogues"] == ds and refs["l15"]["image_identity"] == refs["next"]["image_identity"]
    for name in ("gate", "dev", "main"):
        assert refs["l15"]["phase_images"][name] == refs["next"]["phase_images"][name], name
    for image, identity in refs["next"]["image_identity"].items():
        assert sha(ROOT / identity["image_path"]) == identity["sha256"], image
    image_sets = {ph: list(refs["next"]["phase_images"][PHASE_IMAGES[ph]]) for ph in PHASES}
    if limit:      # development rehearsals only
        image_sets = {"gate": image_sets["gate"][:limit], "dev": image_sets["dev"][:limit], "main": image_sets["main"][:limit + 1]}
    phase_ids = {ph: [d["dialog_id"] for d in ds if d["image_id"] in set(image_sets[ph])] for ph in PHASES}
    manifest = {"dataset": refs["next"]["dataset"], "schema_version": SCHEMA, "index_sha256": refs["next"]["index_sha256"],
                "workload_sha256": refs["next"]["workload_sha256"], "dialogues": ds, "image_identity": refs["next"]["image_identity"],
                "phase_dialogue_ids": phase_ids, "phase_images": image_sets, "phase_methods": PHASE_METHODS,
                "expected_requests": {p: len(phase_ids[p]) * 3 * len(PHASE_METHODS[p]) for p in PHASES},
                "reference_manifest_sha256": {m: sha(REF_RUN[m] / "manifest.json") for m in MODELS},
                "gate_policy": "every dialogue of the four gate images of the earlier main runs",
                "dev_policy": "every MT-GQA dialogue of the 40 DEV images of the earlier main runs",
                "main_policy": "all 398 images / 4,061 dialogues, manifest order"}
    assert limit or manifest["expected_requests"] == EXPECTED_REQUESTS, manifest["expected_requests"]
    if limit:
        manifest["rehearsal_image_limit"] = limit
    return manifest


def reference_store_bytes(model, kind, images):
    out = {}
    for image in images:
        folder = REF_RUN[model] / "main" / "images" / image
        doc = json.loads((folder / "COMMITTED.json").read_text())
        out[image] = json.loads((folder / doc["attempt_id"] / "disk_usage.json").read_text())["store_bytes"][kind]
    return out


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
    now = source_hashes()
    changed = {m: sorted(k for k, v in refs[m]["source_sha256"].items() if not REPORT_SCRIPT.match(k) and now.get(k) != v) for m in MODELS}
    assert not any(changed.values()), ("a file frozen by an earlier main run changed", changed)
    checkout = os.environ.get("IMPRESS_OFFICIAL_REPO")
    head = subprocess.check_output(["git", "-C", checkout, "rev-parse", "HEAD"], text=True).strip() if checkout else None
    assert head == OFFICIAL_COMMIT, ("public checkout", head)
    dirty = subprocess.check_output(["git", "-C", checkout, "status", "--porcelain"], text=True).strip()
    constants = json.loads(CONSTANTS.read_text())
    assert constants["run"] == top.name and constants_block(constants) in CONTRACT.read_text(), "the contract does not hold the measured constants"
    for m in MODELS:
        cal = results_dir(top) / f"calibration_{m}.json"
        assert sha(cal) == constants["models"][m]["calibration_sha256"], m
    top.mkdir(parents=True, exist_ok=True)
    limit = int(os.environ.get("HYPERINFER_DEV_LIMIT", "0")) if "_dev_" in top.name else 0
    manifest = build_manifest(limit)
    atomic_json(top / "manifest.json", manifest, replace=False)
    # ---- storage plan: store sizes the earlier main runs measured on the same images
    backends = {m: Backend(m, top) for m in MODELS}
    writes, largest = {}, 0
    for m in MODELS:
        B = backends[m]
        images = sorted({i for ph in PHASES for i in manifest["phase_images"][ph]})
        size = {k: reference_store_bytes(m, k, images) for k in ("raster", B.NEW_KIND)}
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
        shutil.copyfile(results_dir(top) / f"calibration_{m}.json", top / f"calibration_{m}.json")
        models[m] = {"label": MODEL_LABEL[m], "model": B.M.MODEL_ID, "revision": B.M.REVISION, "directory": MODEL_DIR[m],
                     "backend": f"scripts/{BACKEND_SCRIPT[m]}", "backend_sha256": sha(ROOT / "scripts" / BACKEND_SCRIPT[m]),
                     "reference_run": str(REF_RUN[m].relative_to(ROOT)), "reference_config_sha256": sha(REF_RUN[m] / "config.json"),
                     "tokenization": "official" if m == "next" else "fast tokenizer (the path of the LLaVA-1.5 main run)",
                     "k_chunk_s": constants["models"][m]["k_chunk_s"], "v_chunk_s": constants["models"][m]["v_chunk_s"],
                     "impress_official": B.SRC.OFFICIAL, "server_args": B.M.SERVER_ARGS,
                     "arms": {a: dict(B.SRC.ARMS[a]) for a in GATE_METHODS},
                     "phases": {ph: {"methods": PHASE_METHODS[ph], "dialogues": len(manifest["phase_dialogue_ids"][ph]),
                                     "images": len(manifest["phase_images"][ph]),
                                     "tokenization": B.M.TOK.TOKENIZATION_OFFICIAL if m == "next" else None} for ph in PHASES}}
    config = {"schema_version": SCHEMA, "dataset": manifest["dataset"], "seed": 1234, "greedy": True, "max_new_tokens": 16, "batch_size": 1,
              "cpu_threads": 2, "history_policy": "method_local_generated", "metric": "strict normalized EM",
              "models": models, "model_order": list(MODELS), "phase_methods": PHASE_METHODS, "main_methods": MAIN_METHODS,
              "hyperinfer_arms": {k: dict(v) for k, v in HYPER_ARMS.items()},
              "hyperinfer": {"module": "mmimpress/hyperinfer.py", "module_sha256": sha(ROOT / "mmimpress/hyperinfer.py"),
                             "prediction": "every token layer l uses (selected image tokens + separators)",
                             "unit": "whole 64-token K and V chunks of layer l+1, ascending chunk id, one pread per chunk, K then V",
                             "truncation": "public time budget: (compute + selection time of the previous layer) minus k_chunk_s + v_chunk_s per chunk; budget 0 -> first half",
                             "budget_scope": "per arm, for the life of the process (not reset between requests)",
                             "probe_read_ahead": "kept (the worker reads the next layer's probe block first)",
                             "worker": "one background thread; disk reads and host arrays only", "cache_tiers": "none; cold SSD; no warm-up"},
              "official_checkout": {"path": str(Path(checkout).resolve()), "head": head, "dirty": bool(dirty), "repository": impress_official.OFFICIAL_REPOSITORY},
              "arm_order": "cyclic rotation by the dialogue's ordinal in the dataset (ordinal % number of arms of the phase)",
              "source_sha256": now, "frozen_files_of_reference_runs_changed": changed,
              "contract_sha256": sha(CONTRACT), "port_table_sha256": sha(PORT_TABLE), "constants_sha256": sha(CONSTANTS),
              "manifest_sha256": sha(top / "manifest.json"), "storage_plan_sha256": sha(top / "storage_plan.json"),
              "bootstrap": {"resamples": 10000, "seed": 1234, "unit": "image", "ci": 0.95},
              "resume_policy": "committed images skipped; inside an interrupted image complete dialogues are kept and the interrupted dialogue restarts "
                               "at turn 1; unverifiable stores -> the image restarts",
              "environment": {"python": sys.version, "executable": sys.executable, "torch": torch.__version__, "transformers": transformers.__version__,
                              "numpy": np.__version__}}
    atomic_json(top / "config.json", config, replace=False)
    shutil.copy2(CONTRACT, top / "CONTRACT.md"); shutil.copy2(PORT_TABLE, top / "PORT_TABLE.md"); shutil.copy2(CONSTANTS, top / CONSTANTS.name)
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
                                    "total_hours": total_s / 3600 + 0.75, "basis": "prior seconds per request of the earlier runs' phases; unit tests, gate checks and the report add about 45 min"},
              "free_space": {"free_bytes": free, "free_gib": free / GiB, "required_free_bytes": plan["required_free_bytes"]},
              "smart": RAW_STATUS.smart_snapshot(), "ssd_temperature_at_freeze": RAW_STATUS.ssd_temperature()}
    atomic_json(results_dir(top) / "prerun.json", prerun)
    RAW_STATUS.update(top, expected_requests=prerun["expected_requests_total"], completed_requests=0,
                      expected_ssd_write_tb=round(total_writes / 1e12, 3), expected_total_hours=round(prerun["expected_duration"]["total_hours"], 1),
                      eta_utc=RAW_STATUS.utc(time.time() + total_s + 2700), eta_note="pre-run estimate (see prerun.json)")
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
    if "_dev_" not in top.name and (sha(CONTRACT) != config["contract_sha256"] or sha(PORT_TABLE) != config["port_table_sha256"]
                                    or sha(CONSTANTS) != config["constants_sha256"]):
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


def model_config(config, model):
    """The dictionary the imported harness reads as its run configuration."""
    return {"phases": config["models"][model]["phases"], "layer_budgets": {}, "source_sha256": config["source_sha256"],
            "cpu_threads": config["cpu_threads"], "_file_sha256": config["_file_sha256"], "_manifest_sha256": config["_manifest_sha256"]}


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
    constants = {k: config["models"][model][k] for k in ("k_chunk_s", "v_chunk_s")}
    B = Backend(model, top, constants=constants, expected={m: manifest["expected_requests"] for m in MODELS})
    M = B.M
    M.assert_gpu_exclusive()
    cfg = model_config(config, model)
    runner, servers = B.load(cfg)
    try:
        atomic_json(sub / f"{phase}_start_{time.strftime('%Y%m%dT%H%M%S')}.json",
                    {"started_unix": time.time(), "pid": os.getpid(), "resume": bool(resume), "boot_id": RAW_STATUS.boot_id(), "model": model,
                     "constants": constants, "ssd_temperature": RAW_STATUS.ssd_temperature()})
        receipt = M.run_phase(runner, servers, sub, phase, cfg, manifest)
        states = {k: {"requests": s.requests, "budget_ms_at_end": 1e3 * s.budget_s} for k, s in B.states.items()}
        atomic_json(sub / f"{phase}_execution_complete.json", {"status": "PASS", **receipt, "finished_unix": time.time(), "prefetch_states": states})
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
    p.add_argument("--run", type=Path, required=True, help="runs/mt_gqa_hyperinfer_<UTC>")
    p.add_argument("--stage", choices=("calibrate", "constants", "freeze") + PHASES, required=True)
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    top = args.run.resolve()
    assert top.parent == ROOT / "runs" and top.name.startswith(RUN_PREFIX)
    if args.stage == "calibrate":
        return calibrate(top, args.model)
    if args.stage == "constants":
        return write_constants(top)
    if args.stage == "freeze":
        return freeze(top)
    return run_stage(top, args.model, args.stage, args.resume)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""ConvBench x {LLaVA-NeXT-7b, LLaVA-1.5-7b}: HyperInfer (contract docs/convbench_hyperinfer_contract.md).

Only HyperInfer is measured in the main phase.  ReComp, FullLoad, IMPRESS and Ours are the recorded rows of
the main experiment (results/convbench_main_20261122T103401Z); the report pairs them by conversation and turn.

No method code is changed or added.  Where the code comes from:

* The ConvBench request loop is scripts/217 (the runner of the main experiment), loaded as a private module
  and used as it is: prompt rendering and the official-token-id check before every request, ``max_new_tokens``
  1024, turn 1 from pixels for every arm, the first arm of a store kind builds that store from its own turn 1,
  page cache dropped before every stored request, one committed document per conversation, the failure rule
  and the resume rule.  Its arm and phase tables come from this run's private copy of
  ``convbench_main_common`` (scripts/convbench_hyperinfer_common.py).
* The harness of each model is the one scripts/217 imports (scripts/209 for LLaVA-NeXT, scripts/206 for
  LLaVA-1.5) -- the same modules the MT-GQA HyperInfer run used for its stored requests.
* The HyperInfer request is the request function of the MT-GQA run (scripts/228 ``Backend.hyper_hit``,
  scripts/225 ``Backend.hyper_request`` / ``dispatch``, calling ``mmimpress.hyperinfer_v2``), called through
  an object that only carries the harness module.  The arm ids, the retention ratio, the probe heads and the
  similarity threshold are those of that run (asserted against its config.json at freeze).

What this script adds: the mapping of this run's arms, three extra fields in a row (the per-layer selection
digest, a compact per-layer IMPRESS record of the HyperInfer arms, the number of tokens after the image),
freeze and pre-run checks of this run.

Arms
  fullload            FullLoad                          (pilot table, TTFT continuity)
  impress             IMPRESS                           (gates 1, 2; pilot table, TTFT continuity)
  hyperinfer          HyperInfer -- the only arm of the main phase
  hyperinfer_off      HyperInfer, prefetch off          (gate 2)
  hyperinfer_repeat   HyperInfer, executed a second time (gate 4)
  hyperinfer_nan      HyperInfer, unread rows hold NaN   (gate 3)

Phases, per model: gate (4 conversations, 6 arms) -> pilot (40 conversations, FullLoad / IMPRESS / HyperInfer)
-> overflow (2 conversations, HyperInfer; only if the pilot has no input above 4096 tokens) -> main (577, HyperInfer).

Stages: --stage freeze | prerun                       (no model)
        --stage gate | pilot | overflow | main  --model next | llava15
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import convbench_hyperinfer_common as HC  # noqa: E402
import llava_next_main_status as STATUS  # noqa: E402
from mmimpress import hyperinfer_v2 as H2  # noqa: E402
from mmimpress import impress_official  # noqa: E402

C = HC.C
E = HC.load_with_common("_convbench_hyperinfer_runner", "217_eval_convbench_main.py")            # the ConvBench runner
M228 = E.load_script("_convbench_hyperinfer_mtgqa_eval", "228_eval_mt_gqa_hyperinfer_v2.py")       # the HyperInfer request of the MT-GQA run
GiB = 1 << 30
FREE_SPACE_FACTOR = E.FREE_SPACE_FACTOR


class MtGqaRequests(M228.Backend):
    """``hyper_hit`` (scripts/228), ``hyper_request`` and ``dispatch`` (scripts/225) of the MT-GQA run, bound to
    one model's harness module.  The parent's ``__init__`` (the MT-GQA phase tables, status and audit) is not run:
    those three methods read only ``SRC`` and ``_stored_request``."""

    def __init__(self, src):      # noqa: D401  (deliberately does not call the parent)
        self.SRC = src
        self._stored_request = src.stored_request


class Backend(E.Backend):
    """The backend of scripts/217 with this run's arms; stored requests go through the MT-GQA run's dispatcher."""

    def __init__(self, model):
        super().__init__(model)
        B = self.B
        self.arms = {HC.FULL: self.arms[C.FULL], HC.IMP: self.arms[C.IMP]}
        for role, arm in HC.BACKEND_ARM.items():
            spec = M228.HYPER_ARMS[arm]
            # the arm entry scripts/225 adds to the harness' arm table (same fields, same values)
            B.ARMS[arm] = {"family": "hyperinfer", "kind": B.NEW_KIND, "async": spec["kv_prefetch"] != "sync",
                           "display": spec["display"], "ratio": float(impress_official.KEEP_RATIO)}
            self.arms[role] = arm
        self.requests = MtGqaRequests(B)

    def stored(self, ctx, factory, role, image_id, store_path):
        # FullLoad: the harness' own stored request; IMPRESS: the same, with the per-layer used sets recorded after
        # the request; HyperInfer arms: the MT-GQA run's request
        return self.requests.dispatch(self.runner, self.servers, ctx, factory, self.arms[role], image_id, store_path)

    def describe(self):
        out = super().describe()
        out["hyperinfer_arms"] = {role: dict(self.B.ARMS[arm], kv_prefetch=M228.HYPER_ARMS[arm]["kv_prefetch"], nan=M228.HYPER_ARMS[arm]["nan"])
                                  for role, arm in HC.BACKEND_ARM.items()}
        return out


# ------------------------------------------------------------------- rows
_runner_make_row = E.make_row


def make_row(be, phase, role, turn, order, prompt, result, input_tokens, hit, geometry):
    """The row of scripts/217 plus: tokens after the image, and for the HyperInfer arms the compact per-layer
    IMPRESS record (the parent's layer log is dropped from main-phase rows)."""
    row = _runner_make_row(be, phase, role, turn, order, prompt, result, input_tokens, hit, geometry)
    row["result"]["cap_reached"] = row["cap_reached"]      # the harness compares with its own cap (16)
    if hit:
        row["suffix_tokens"] = int(input_tokens) - int(geometry["prefix_len"])
        if be.family(role) == "hyperinfer":
            log = result["layer_log"]
            row["impress"] = {"fb": [int(x["mode"] == "fallback") for x in log], "sim": [x["sim"] for x in log],
                              "kc": [x["k_chunks"] for x in log], "vc": [x["v_chunks"] for x in log],
                              "nc": (int(geometry["v_token_num"]) + 63) // 64}
    return row


E.make_row = make_row      # the private copy of scripts/217 (this process only)


# ------------------------------------------------------- the recorded main experiment
def recorded_costs(model, index):
    """Per conversation, from the main experiment's record: bytes the store builds wrote and seconds the requests took."""
    out = {}
    for conv in index:
        doc = HC.recorded_doc(model, conv["conversation_id"])
        assert doc is not None and not doc["failures"], conv["conversation_id"]
        kind = HC.impress_kind(doc)
        s_imp, s_full = doc["stores"][kind], doc["stores"]["raster"]
        store_s = lambda s: (float(s["service_persistence_ms"]) + float(s.get("activation_ms") or 0.0)) / 1000      # noqa: E731
        e2e = {r: sum(x["request_e2e_ms"] for x in doc["rows"] if x["role"] == r) / 1000 for r in (C.RECOMP, C.FULL, C.IMP, C.OURS)}
        stores_all = sum(store_s(s) for s in doc["stores"].values())
        out[conv["conversation_id"]] = {
            "write_impress": int(s_imp["os_write_bytes"]), "write_fullload": int(s_full["os_write_bytes"]),
            "disk_impress": int(doc["store_bytes_on_disk"][kind]), "disk_fullload": int(doc["store_bytes_on_disk"]["raster"]),
            "e2e_impress_s": e2e[C.IMP], "e2e_fullload_s": e2e[C.FULL], "store_impress_s": store_s(s_imp), "store_fullload_s": store_s(s_full),
            "other_s_per_request": max(0.0, float(doc["seconds"]) - sum(e2e.values()) - stores_all) / 12,
            "impress_largest_input": max(x["input_tokens"] for x in doc["rows"] if x["role"] == C.IMP)}
    return out


def phase_estimate(costs, cids, roles):
    """(bytes written, seconds) a phase is expected to take: FullLoad arm = recorded FullLoad, every other arm = recorded IMPRESS."""
    writes = seconds = 0.0
    for cid in cids:
        c = costs[cid]
        n_imp = sum(r != HC.FULL for r in roles)
        writes += c["write_impress"] + (c["write_fullload"] if HC.FULL in roles else 0)
        seconds += (n_imp * c["e2e_impress_s"] + c["store_impress_s"] + 3 * len(roles) * c["other_s_per_request"]
                    + ((c["e2e_fullload_s"] + c["store_fullload_s"]) if HC.FULL in roles else 0.0))
    return int(writes), float(seconds)


# ---------------------------------------------------------------- freeze
def freeze_checks():
    """Everything the freeze asserts before a config is written: the earlier runs' frozen files and environment, the
    HyperInfer implementation and settings of the MT-GQA run, the FullLoad / IMPRESS settings of the main experiment.
    Returns (models, changed, main config, MT-GQA config, current source hashes)."""
    import transformers
    G = E.gen_module()
    main_cfg = json.loads((HC.MAIN_RUN / "config.json").read_text())
    mt_cfg = json.loads((HC.MTGQA_RUN / "config.json").read_text())
    now = C.source_hashes()
    # ---- nothing the earlier runs froze has changed; the same environment
    changed = {"convbench_main": sorted(k for k, v in main_cfg["source_sha256"].items() if now.get(k) != v),
               "mt_gqa_hyperinfer_v2": sorted(k for k, v in mt_cfg["source_sha256"].items() if not C.REPORT_SCRIPT.match(k) and now.get(k) != v)}
    assert not any(changed.values()), ("a source file frozen by an earlier run changed", changed)
    for name, cfg in (("convbench_main", main_cfg), ("mt_gqa_hyperinfer_v2", mt_cfg)):
        env = cfg["environment"]
        assert (transformers.__version__, torch.__version__, np.__version__, sys.executable) == (
            env["transformers"], env["torch"], env["numpy"], env["executable"]), f"not the environment of {name}"
    assert C.sha(C.INDEX) == main_cfg["index_sha256"] and C.sha(C.ALIGN_DOC) == main_cfg["alignment_doc_sha256"]
    assert C.sha(HC.MAIN_CONTRACT) == main_cfg["contract_sha256"]
    assert C.sha(HC.MTGQA_CONTRACT) == mt_cfg["contract_sha256"] and C.sha(HC.PORT_TABLE) == mt_cfg["port_table_sha256"]
    # ---- the HyperInfer implementation and settings are those of the MT-GQA run
    assert C.sha(ROOT / "mmimpress/hyperinfer_v2.py") == mt_cfg["hyperinfer"]["module_sha256"]
    assert mt_cfg["hyperinfer_arms"] == {k: dict(v) for k, v in M228.HYPER_ARMS.items()}
    assert set(HC.BACKEND_ARM.values()) == set(M228.HYPER_ARMS)
    # ---- the main experiment is complete (its rows and judgments are what the report reads)
    main_status = json.loads((HC.MAIN_RUN / "status.json").read_text())
    assert main_status["stage"] == "done"
    models = {}
    for model in C.MODELS:
        assert (HC.MAIN_RUN / model / "main" / "DONE.json").exists() and (HC.MAIN_RUN / "judge" / model / "DONE").exists()
        reference = E.reference_checks(E.Backend(model))      # the harness is the code and the settings of its reference runs
        be = Backend(model)
        d = be.describe()
        old = main_cfg["models"][model]
        for key in ("model_id", "revision", "server_args", "impress", "readahead"):
            assert d[key] == old[key], (model, key)
        for role in (HC.FULL, HC.IMP):
            assert d["arms"][role] == old["arms"][role], (model, role)
        assert d["store_kinds"][HC.FULL] == old["store_kinds"][C.FULL] and d["store_kinds"][HC.IMP] == old["store_kinds"][C.IMP]
        assert all(d["store_kinds"][r] == d["store_kinds"][HC.IMP] for r in HC.BACKEND_ARM), "HyperInfer reads the IMPRESS store format"
        mt = mt_cfg["models"][HC.MTGQA_MODEL[model]]
        assert mt["model"] == be.B.MODEL_ID and mt["revision"] == be.B.REVISION
        assert mt["impress_official"] == be.B.OFFICIAL and mt["server_args"] == be.B.SERVER_ARGS
        for arm in HC.BACKEND_ARM.values():
            assert mt["arms"][arm] == dict(be.B.ARMS[arm]), (model, arm)
        # the stored requests of the MT-GQA run went through the same harness module: scripts/206 for LLaVA-1.5, and for
        # LLaVA-NeXT scripts/209, which scripts/222 (that run's backend) loads as its ``N``
        harness = Path(be.reference["harness"]).name
        if model == "next":
            assert harness == "209_eval_llava_next_mtvqa_main.py" and f'"{harness}"' in (ROOT / "scripts" / M228.BACKEND_SCRIPT["next"]).read_text()
        else:
            assert harness == M228.BACKEND_SCRIPT["l15"]
        assert be.requests.SRC is be.B
        models[model] = dict(d, reference_checks=reference, render_after_image=G.OFFICIAL_RENDER[model][0], render_after_eos=G.OFFICIAL_RENDER[model][1])
    assert G.NOMINAL_CAP == C.MAX_NEW
    return models, changed, main_cfg, mt_cfg, now


def freeze(res):
    import transformers
    res = Path(res)
    assert res.parent == ROOT / "results" and res.name.startswith(C.RUN_PREFIX) and C.CONTRACT.is_file()
    if (res / "config.json").exists():
        raise FileExistsError("config.json exists: the run is already frozen")
    index, index_sha = C.load_index()
    models, changed, main_cfg, mt_cfg, now = freeze_checks()
    costs = {m: recorded_costs(m, index) for m in C.MODELS}
    bundle = max(c["disk_impress"] + c["disk_fullload"] for m in C.MODELS for c in costs[m].values())
    config = {
        "schema": C.SCHEMA, "frozen_utc": C.utc(), "contract": str(C.CONTRACT.relative_to(ROOT)), "contract_sha256": C.sha(C.CONTRACT),
        "main_contract_sha256": C.sha(HC.MAIN_CONTRACT), "mt_gqa_contract_sha256": C.sha(HC.MTGQA_CONTRACT), "port_table_sha256": C.sha(HC.PORT_TABLE),
        "alignment_doc_sha256": C.sha(C.ALIGN_DOC), "index_sha256": index_sha,
        "main_run": {"run": str(HC.MAIN_RUN.relative_to(ROOT)), "config_sha256": C.sha(HC.MAIN_RUN / "config.json")},
        "mt_gqa_hyperinfer_run": {"run": str(HC.MTGQA_RUN.relative_to(ROOT)), "config_sha256": C.sha(HC.MTGQA_RUN / "config.json")},
        "frozen_files_of_earlier_runs_changed": changed,
        "models": models, "model_order": list(C.MODELS),
        "hyperinfer": dict(mt_cfg["hyperinfer"], request="scripts/228 Backend.hyper_hit, scripts/225 Backend.hyper_request / dispatch (called as they are)"),
        "backend_arm": HC.BACKEND_ARM, "phase_roles": HC.PHASE_ROLES, "display": HC.DISPLAY,
        "generation": main_cfg["generation"],
        "judge": {"cap": C.JUDGE_CAP, "script": "scripts/235_convbench_hyperinfer_judge.py (the functions of scripts/219 and scripts/214)",
                  "view": "capN", "rule": "tokens of all 1,731 requests equal to the recorded IMPRESS tokens -> the recorded IMPRESS judgments are "
                                          "HyperInfer's; otherwise every HyperInfer answer of the model is judged, with an empty cache"},
        "statistics": {"unit": "conversation", "resamples": C.RESAMPLES, "seed": C.SEED, "ci": 0.95},
        "phases": {"gate": {"ordinals": list(C.GATE_ORDINALS), "roles": HC.GATE_ROLES},
                   "pilot": {"ordinals": list(C.PILOT_ORDINALS), "roles": HC.PILOT_ROLES},
                   "overflow": {"roles": HC.MAIN_ROLES, "selection": "after the pilot: only when the pilot has no HyperInfer input above 4096 tokens"},
                   "main": {"ordinals": "0..576 (index order)", "roles": HC.MAIN_ROLES}},
        "conversation_ids": {"gate": [index[o]["conversation_id"] for o in C.GATE_ORDINALS],
                             "pilot": [index[o]["conversation_id"] for o in C.PILOT_ORDINALS]},
        "arm_rotation": "arm order of a conversation = the phase's arm list rotated by (index ordinal % number of arms)",
        "expected_requests": {"gate_per_model": 4 * 3 * len(HC.GATE_ROLES), "pilot_per_model": 40 * 3 * len(HC.PILOT_ROLES),
                              "main_per_model": C.N_CONV * 3, "main_total": 2 * C.N_CONV * 3},
        "failure_policy": main_cfg["failure_policy"],
        "storage": {"largest_bundle_bytes_bound": int(bundle), "free_space_factor": FREE_SPACE_FACTOR, "required_free_bytes": int(FREE_SPACE_FACTOR * bundle),
                    "bound": "the largest FullLoad store + IMPRESS-layout store of one conversation in the main experiment's record",
                    "policy": "one live conversation; its stores are deleted when it is committed", "deletion_scope": "results/<run>/scratch only"},
        "source_sha256": now,
        "environment": {"python": sys.version, "executable": sys.executable, "torch": torch.__version__, "transformers": transformers.__version__,
                        "numpy": np.__version__},
    }
    C.atomic_json(res / "config.json", config)
    shutil.copy2(C.CONTRACT, res / "CONTRACT.md")
    shutil.copy2(HC.PORT_TABLE, res / "PORT_TABLE.md")
    print(json.dumps({"freeze": "PASS", "run": str(res), "source_files": len(now), "reference_checks": {m: models[m]["reference_checks"] for m in models}}),
          flush=True)


# ---------------------------------------------------------------- prerun
def prerun(res):
    """Checks before any GPU measurement; writes <res>/prerun.json."""
    res = Path(res)
    config = HC.verify_freeze(res)
    index, _ = C.load_index()
    started = time.time()
    limit = C.dev_limit(res)
    writes, seconds, costs = {}, {}, {}
    for model in C.MODELS:
        costs[model] = recorded_costs(model, index)
        ids = {ph: [c["conversation_id"] for c in C.phase_conversations(res, model, ph, index)] for ph in ("gate", "pilot", "main")}
        longest = sorted(costs[model], key=lambda c: -costs[model][c]["impress_largest_input"])[:C.OVERFLOW_CONVERSATIONS]
        est = {ph: phase_estimate(costs[model], ids[ph], HC.PHASE_ROLES[ph]) for ph in ids}
        est["overflow_if_run"] = phase_estimate(costs[model], longest, HC.PHASE_ROLES["overflow"])
        writes[model] = {ph: e[0] for ph, e in est.items()}
        writes[model]["total"] = int(sum(writes[model].values()))
        seconds[model] = {ph: e[1] + 120 for ph, e in est.items()}      # + model loading
        seconds[model]["total"] = float(sum(seconds[model].values()))
    total = sum(writes[m]["total"] for m in C.MODELS)
    fixed = 900      # the test suites of both environments, gate checks, the file inventories and the report
    total_s = sum(seconds[m]["total"] for m in C.MODELS) + fixed
    free = shutil.disk_usage(res).free
    required = int(config["storage"]["required_free_bytes"])
    recorded = {m: {"conversations": len(costs[m]),
                    "impress_requests_above_4096": sum(1 for conv in index for r in HC.recorded_doc(m, conv["conversation_id"])["rows"]
                                                       if r["role"] == C.IMP and r["status"] == "ok" and r["input_above_4096"]),
                    "impress_store_write_gb_mean": float(np.mean([c["write_impress"] for c in costs[m].values()]) / 1e9),
                    "impress_seconds_per_conversation_mean": float(np.mean([c["e2e_impress_s"] + c["store_impress_s"] for c in costs[m].values()]))}
                for m in C.MODELS}
    smart = STATUS.smart_snapshot()
    blocking = []
    if free < required:
        blocking.append(f"free space {free} B is below {FREE_SPACE_FACTOR} x the largest per-conversation store bundle ({required} B)")
    # (size, mtime) of every file of results/, runs/ and kvstore* outside this run: compared again when the run ends
    protection = E.Backend("next").B.stat_inventory(res, res / "protection_before.tsv.gz")
    C.atomic_json(res / "protection_before.json", protection)
    doc = {
        "written_before_any_gpu_measurement_utc": C.utc(), "run": res.name, "status": "PASS" if not blocking else "BLOCKED", "blocking": blocking,
        "rehearsal_limit": limit or None,
        "expected_ssd_write": {
            "definition": "per conversation, the bytes the store builds wrote in the main experiment for the same conversation: the IMPRESS-layout "
                          "store (written, then rewritten in the reordered layout) and, in the gate and pilot phases, the FullLoad store. Request "
                          "rows and status files are not included (well below 1 % of the total).",
            "bytes": writes, "total_bytes": int(total), "total_tb": total / 1e12, "per_model_tb": {m: writes[m]["total"] / 1e12 for m in C.MODELS}},
        "expected_duration": {
            "seconds": seconds, "fixed_seconds": fixed, "total_hours": total_s / 3600,
            "judge_hours_per_model_if_needed": E.ALIGN_JUDGE_SECONDS_PER_ANSWER_SET / 3600,
            "basis": "per conversation: the recorded seconds of the IMPRESS requests (x number of IMPRESS-format arms) and of the FullLoad requests, the "
                     "recorded store writing and activation seconds, and the main experiment's remaining seconds per request; judging is needed only "
                     "if a generated token differs from the recorded IMPRESS tokens (3.6 h per model, not included in the total)"},
        "free_space": {"free_bytes": free, "free_gib": free / GiB, "required_free_bytes": required,
                       "rule": f"do not start when free space < {FREE_SPACE_FACTOR} x the largest per-conversation store bundle; rechecked before every conversation",
                       "ok": free >= required},
        "smart": smart, "ssd_counters_at_prerun": STATUS._safe(STATUS.sectors), "ssd_temperature_at_prerun": STATUS.ssd_temperature(),
        "recorded_main_experiment": recorded,
        "protection_inventory": {"file": "protection_before.json", "files": protection["files"], "sha256": protection["sha256"]},
        "seconds": time.time() - started}
    C.atomic_json(res / "prerun.json", doc)
    C.atomic_json(res / "prerun_receipt.json", {"status": doc["status"], "blocking": blocking, "prerun_sha256": C.sha(res / "prerun.json"),
                                                "config_sha256": config["_file_sha256"], "at_utc": C.utc()})
    STATUS.update(C.res_rel(res), expected_ssd_write_tb=round(total / 1e12, 3), expected_total_hours=round(total_s / 3600, 1),
                  smart=smart["status"], prerun=doc["status"], eta_utc=C.utc(time.time() + total_s), eta_note="pre-run estimate (see prerun.json)")
    print(json.dumps({"prerun": doc["status"], "blocking": blocking, "expected_ssd_write_tb": round(total / 1e12, 3),
                      "hours": round(total_s / 3600, 1), "free_gib": round(free / GiB, 1), "smart": smart["status"]}), flush=True)
    return doc


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", type=Path, required=True, help="results/convbench_hyperinfer_<UTC>")
    p.add_argument("--stage", choices=("freeze", "prerun") + HC.PHASES, required=True)
    p.add_argument("--model", choices=C.MODELS)
    args = p.parse_args()
    res = args.res.resolve()
    assert res.parent == ROOT / "results" and res.name.startswith(C.RUN_PREFIX)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if args.stage == "freeze":
        return freeze(res)
    if args.stage == "prerun":
        doc = prerun(res)
        sys.exit(0 if doc["status"] == "PASS" else 1)
    if args.model is None:
        p.error("--model is required for a request phase")
    config = HC.verify_freeze(res)
    receipt = C.read_json(res / "prerun_receipt.json") or {}
    if receipt.get("status") != "PASS" or receipt.get("config_sha256") != config["_file_sha256"]:
        raise RuntimeError("PRERUN_NOT_PASSED")
    phase, model = args.stage, args.model
    if phase == "main":      # the main phase is admitted only by the recorded gate decision of this model
        admit = C.read_json(res / "gates" / model / "admission.json") or {}
        if admit.get("status") != "PASS" or admit.get("config_sha256") != config["_file_sha256"]:
            raise RuntimeError("MAIN_NOT_ADMITTED")
    if C.phase_complete(res, model, phase):
        print(json.dumps({"phase": phase, "model": model, "status": "already complete"}))
        return
    be = Backend(model)
    E.wait_for_gpu(be, res, phase)
    be.load()
    C.append_jsonl(res / "process_log.jsonl", {"event": "phase_process_start", "model": model, "phase": phase, "pid": os.getpid(),
                                               "boot_id": C.boot_id(), "at_utc": C.utc(), "idle_synchronize_cost": H2.idle_synchronize_cost(),
                                               "ssd_temperature": STATUS.ssd_temperature()})
    try:
        E.run_phase(be, res, phase, config)
    except E.FatalState as exc:
        STATUS.update(C.res_rel(res), last_error={"stage": f"{model}_{phase}", "error": repr(exc)[:500], "at_utc": C.utc(),
                                                  "note": "process state not clean after a failed request; the phase process is restarted"})
        print(json.dumps({"fatal_state": repr(exc)[:500]}), flush=True)
        os._exit(75)
    except BaseException as exc:
        STATUS.update(C.res_rel(res), last_error={"stage": f"{model}_{phase}", "error": repr(exc)[:500], "at_utc": C.utc()})
        print(json.dumps({"error": repr(exc)[:1000], "traceback": traceback.format_exc()[-3000:]}), flush=True)
        raise
    finally:
        be.close()


if __name__ == "__main__":
    main()

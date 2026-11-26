#!/usr/bin/env python3
"""ConvBench HyperInfer run: the driver (started by scripts/run_convbench_hyperinfer.sh, detached and locked).

Runs every step as a subprocess, never waits for input and keeps <res>/status.json current.  A step that
already has its result is skipped, a request phase or a judge run that was interrupted continues, so the
same command continues after an interruption or a reboot.

  freeze -> pre-run checks (prerun.json, SMART) -> gate 7 (tests, both environments)
  for LLaVA-NeXT, then LLaVA-1.5:
      gate phase (4 conversations, 6 arms)                        -> gates 1-5
      pilot (40 conversations, FullLoad / IMPRESS / HyperInfer)   -> gates 1, 5, 6; pilot table, inputs above 4096 (gate 8)
      admission (gates 1-7)                                       -> main (577 conversations, HyperInfer) if admitted
      identity: the main-phase tokens against the recorded IMPRESS tokens (1,731 requests)
  judge (only a model whose tokens are not all identical) -> posthoc -> report.md

A model whose gates fail: its main phase is not started, the failure and its cause are written to status.json
and report.md, and the driver goes on with the other model and the report, then exits.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import convbench_hyperinfer_common as HC  # noqa: E402
import llava_next_main_status as STATUS  # noqa: E402

C = HC.C
PY = C.PY
EVAL, VAL, REPORT, JUDGE = ("scripts/231_eval_convbench_hyperinfer.py", "scripts/232_validate_convbench_hyperinfer.py",
                            "scripts/233_report_convbench_hyperinfer.py", "scripts/235_convbench_hyperinfer_judge.py")
MAX_PHASE_STARTS = int(os.environ.get("CONVBENCH_HYPERINFER_MAX_PHASE_STARTS", "5"))
GPU_WAIT_POLLS = 720      # x 30 s
JUDGE_SECONDS = 12911.0   # one fresh answer set (577 x 3 judgments) in the alignment run
STATE = {"child": None, "stop": False}


def log(msg):
    print(f"=== {C.utc()} {msg}", flush=True)


def on_signal(signum, frame):
    STATE["stop"] = True
    child = STATE["child"]
    if child is not None and child.poll() is None:
        try:
            child.terminate()
        except OSError:
            pass


def gpu_busy():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True, timeout=60).stdout
        return bool(out.strip())
    except Exception:      # noqa: BLE001
        return False


class Driver:
    def __init__(self, res):
        self.res = Path(res).resolve()
        self.rel = C.res_rel(self.res)
        self.env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": ".:scripts:tests", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                    "IMPRESS_OFFICIAL_REPO": "/home/dblab/hj/_external_ref/HyperInfer"}
        self.errors = []

    # ---------------------------------------------------------------- utils
    def status(self, **fields):
        return STATUS.update(self.rel, **fields)

    def stopped(self):
        if STATE["stop"]:
            log("signal received: stopping")
            self.status(stage="stopped_by_signal", driver_pid=None,
                        last_error={"stage": "signal", "error": "driver received TERM/INT; continue with the same command", "at_utc": C.utc()})
            sys.exit(130)

    def run(self, step, cmd, gpu=False):
        """Run one subprocess step; returns its exit code.  Timing goes to driver_steps.jsonl."""
        self.stopped()
        if gpu:
            polls = 0
            while gpu_busy():
                if polls == 0:
                    log(f"{step}: the GPU is used by another process; waiting")
                    self.status(stage=f"waiting_for_gpu_before_{step}")
                if polls >= GPU_WAIT_POLLS:
                    return 97
                time.sleep(30); polls += 1
                self.stopped()
        log(f"START {step}: {' '.join(str(c) for c in cmd)}")
        started = time.time()
        STATE["child"] = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, env=self.env)
        while True:
            try:
                rc = STATE["child"].wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                if STATE["stop"]:
                    try:
                        STATE["child"].wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        STATE["child"].kill()
                    self.stopped()
        STATE["child"] = None
        seconds = time.time() - started
        C.append_jsonl(self.res / "driver_steps.jsonl", {"step": step, "started_utc": C.utc(started), "finished_utc": C.utc(), "seconds": seconds, "rc": rc,
                                                         "boot_id": C.boot_id()})
        STATUS.ssd_sample(self.rel, f"after_{step}")
        log(f"END {step} rc={rc} ({seconds:.0f} s)")
        self.stopped()
        return rc

    def gate_status(self, *parts):
        doc = C.read_json(self.res.joinpath("gates", *parts))
        return "MISSING" if doc is None else doc.get("status", "MISSING")

    def error(self, stage, message, **extra):
        e = {"stage": stage, "error": message, "at_utc": C.utc(), **extra}
        self.errors.append(e)
        log(f"ERROR {stage}: {message}")
        self.status(last_error=e)

    def judge_needed(self, model):
        doc = C.read_json(self.res / "gates" / model / "identity.json")
        return None if doc is None else bool(doc["judge_needed"])

    # ------------------------------------------------------------ estimates
    def estimate(self, step):
        """Seconds a step that has not finished is expected to take."""
        pre = C.read_json(self.res / "prerun.json") or {}
        sec = pre.get("expected_duration", {}).get("seconds", {})
        kind, _, model = step.partition(":")
        if kind in ("gate", "pilot", "main"):
            if C.phase_complete(self.res, model, kind) or self.gate_status(model, "admission.json") == "FAIL":
                return 0.0
            total = float(sec.get(model, {}).get(kind, 900))
            if kind == "main":
                done = len(list(C.phase_dir(self.res, model, "main").glob("conversations/*.json.gz")))
                return total * max(0, C.N_CONV - done) / C.N_CONV
            return total
        if kind == "judge":
            if (self.res / "judge" / model / "DONE").exists() or self.judge_needed(model) is not True:
                return 0.0      # judging happens only when a token differs from the recorded IMPRESS tokens
            return JUDGE_SECONDS
        return {"tests": 150.0, "posthoc": 120.0, "report": 120.0}.get(kind, 60.0)

    def plan(self):
        out = ["tests"]
        for m in C.MODELS:
            out += [f"gate:{m}", f"pilot:{m}", f"main:{m}"]
        return out + [f"judge:{m}" for m in C.MODELS] + ["posthoc", "report"]

    def begin(self, step, title):
        plan = self.plan()
        after = float(sum(self.estimate(s) for s in plan[plan.index(step) + 1:])) if step in plan else 0.0
        now = time.time()
        self.status(stage=step.replace(":", "_"), stage_title=title, estimated_seconds_after_this_stage=after,
                    eta_utc=C.utc(now + self.estimate(step) + after), eta_note="estimate; refined while a request phase or the judge is running")

    # --------------------------------------------------------------- phases
    def phase(self, model, phase):
        """Run or continue a request phase until it is complete.  True = complete."""
        if C.phase_complete(self.res, model, phase):
            log(f"SKIP {model} {phase} phase (complete)")
            return True
        self.begin(f"{phase}:{model}", f"{C.MODEL_NAME[model]} {phase} 단계")
        for n in range(1, MAX_PHASE_STARTS + 1):
            rc = self.run(f"{phase}:{model}", [PY, EVAL, "--res", self.res, "--stage", phase, "--model", model], gpu=True)
            if C.phase_complete(self.res, model, phase):
                return True
            log(f"{model} {phase} phase not complete after start {n} of at most {MAX_PHASE_STARTS} (rc={rc})")
            time.sleep(30)
        return False

    def model_chain(self, model):
        """Gates, pilot, main and the identity check of one model.  Returns "main_complete", "not_admitted" or "error"."""
        name = C.MODEL_NAME[model]
        res = self.res
        if self.gate_status(model, "admission.json") == "MISSING":
            if self.gate_status("tests.json") == "PASS":
                if not self.phase(model, "gate"):
                    self.error(f"gate:{model}", f"{name}: the 4-conversation gate phase did not complete (execution error, not a gate verdict); see driver.log")
                    return "error"
                if self.gate_status(model, "gate_gate.json") == "MISSING":
                    self.run(f"gates_on_4:{model}", [PY, VAL, "--res", res, "--stage", "gate", "--model", model, "--scale", "gate"])
                if self.gate_status(model, "gate_gate.json") == "MISSING":
                    self.error(f"gates_on_4:{model}", "the gate check did not produce a result")
                    return "error"
                if self.gate_status(model, "gate_gate.json") == "PASS":
                    if not self.phase(model, "pilot"):
                        self.error(f"pilot:{model}", f"{name}: the pilot did not complete (execution error, not a gate verdict); see driver.log")
                        return "error"
                    if self.gate_status(model, "gate_pilot.json") == "MISSING":
                        self.run(f"gates_on_pilot:{model}", [PY, VAL, "--res", res, "--stage", "gate", "--model", model, "--scale", "pilot"])
                    if self.gate_status(model, "gate_pilot.json") == "MISSING":
                        self.error(f"gates_on_pilot:{model}", "the gate check did not produce a result")
                        return "error"
                    # gate 8 (recorded, not a condition): inputs above 4096 tokens
                    if not (C.model_dir(res, model) / "overflow_selection.json").exists():
                        self.run(f"overflow_select:{model}", [PY, VAL, "--res", res, "--stage", "overflow-select", "--model", model])
                    sel = C.read_json(C.model_dir(res, model) / "overflow_selection.json")
                    if sel is not None and self.gate_status(model, "gate8_overflow.json") == "MISSING":
                        if sel["needed"] and not self.phase(model, "overflow"):
                            self.error(f"overflow:{model}", f"{name}: the two conversations with long inputs did not complete; recorded, the admission does not depend on it")
                        if not sel["needed"] or C.phase_complete(res, model, "overflow"):
                            self.run(f"overflow_check:{model}", [PY, VAL, "--res", res, "--stage", "overflow-check", "--model", model])
            self.run(f"admission:{model}", [PY, VAL, "--res", res, "--stage", "admit", "--model", model])
        admit = C.read_json(res / "gates" / model / "admission.json")
        if admit is None:
            self.error(f"admission:{model}", "no admission record")
            return "error"
        if admit["status"] != "PASS":
            reason = "; ".join(f"{k}={v}" for k, v in admit["parts"].items() if v != "PASS")
            self.status(**{f"{model}_gates": "FAIL", f"{model}_main": "not started",
                           f"{model}_gate_failure": {"parts": admit["parts"], "failed_gates": [{"where": g.get("where"), "gate": g.get("gate"),
                                                                                              "examples": g.get("examples", [])[:3]} for g in admit["failed_gates"]]}})
            self.error(f"admission:{model}", f"{name}: gates not passed ({reason}); the main phase of this model is not started. "
                                             f"Details: {self.rel}/gates/{model}/ and report.md", gate_failure=True)
            return "not_admitted"
        self.status(**{f"{model}_gates": "PASS", f"{model}_main": "admitted"})
        if not self.phase(model, "main"):
            self.error(f"main:{model}", f"{name}: the main phase stopped before completion; continue with: {HC.DRIVER} {self.rel}")
            self.status(**{f"{model}_main": "incomplete"})
            return "error"
        self.status(**{f"{model}_main": "complete"})
        if self.gate_status(model, "identity.json") == "MISSING":
            self.run(f"identity:{model}", [PY, VAL, "--res", res, "--stage", "identity", "--model", model])
        identity = C.read_json(res / "gates" / model / "identity.json")
        if identity is None:
            self.error(f"identity:{model}", "the comparison with the recorded IMPRESS tokens did not produce a result; the same command retries it")
            return "error"
        self.status(**{f"{model}_tokens_equal_recorded_impress": f"{identity['equal']} / {identity['requests']}",
                       f"{model}_quality_scores": "recorded IMPRESS judgments" if not identity["judge_needed"] else "judged in this run"})
        return "main_complete"

    # ------------------------------------------------------------------ main
    def main(self):
        res = self.res
        kind = "resume" if (res / "driver_invocations.jsonl").exists() else "fresh"
        C.append_jsonl(res / "driver_invocations.jsonl", {"at_utc": C.utc(), "at_unix": time.time(), "boot_id": C.boot_id(), "driver_pid": os.getpid(),
                                                          "kind": kind, "stage_at_start": STATUS.read(self.rel).get("stage")})
        STATUS.ssd_sample(self.rel, f"driver_start_{kind}")
        self.status(stage="starting", driver_pid=os.getpid(), driver_started_utc=C.utc(), driver_kind=kind,
                    driver_invocations=len(C.read_jsonl(res / "driver_invocations.jsonl")), last_error=None,
                    commands={"progress": f"cat {self.rel}/status.json", "stop": f"kill -TERM -- -$(cat {self.rel}/driver.pid)",
                              "continue": f"{HC.DRIVER} {self.rel}"})
        if STATUS.read(self.rel).get("first_started_utc") is None:
            self.status(first_started_utc=C.utc())

        # ---- freeze and pre-run checks
        if not (res / "config.json").exists():
            self.status(stage="freeze")
            if self.run("freeze", [PY, EVAL, "--res", res, "--stage", "freeze"]) != 0:
                self.error("freeze", "freeze failed; nothing was measured (see driver.log)")
                return self.finish(1)
        receipt = C.read_json(res / "prerun_receipt.json") or {}
        if receipt.get("status") != "PASS":
            self.status(stage="prerun")
            rc = self.run("prerun", [PY, EVAL, "--res", res, "--stage", "prerun"])
            if not (res / "smart_before.json").exists():
                C.atomic_json(res / "smart_before.json", STATUS.smart_snapshot())
            if rc != 0:
                self.error("prerun", f"the pre-run checks did not pass; nothing was measured (details: {self.rel}/prerun.json)")
                return self.finish(1)
        if not (res / "smart_before.json").exists():
            C.atomic_json(res / "smart_before.json", STATUS.smart_snapshot())

        # ---- gate 7 (both models)
        if self.gate_status("tests.json") == "MISSING":
            self.begin("tests", "gate 7: 기존 테스트 (두 환경)")
            self.run("tests", [PY, VAL, "--res", res, "--stage", "tests"], gpu=True)
        if self.gate_status("tests.json") == "MISSING":
            self.error("tests.json", "gate 7 (tests) did not produce a result (execution error); the same command retries it")
            return self.finish(1)

        # ---- the two models, one after the other
        outcome = {}
        for model in C.MODELS:
            outcome[model] = self.model_chain(model)
            self.status(**{f"{model}_outcome": outcome[model]})

        # ---- judge: only a model whose tokens are not all equal to the recorded IMPRESS tokens
        judge_missing = False
        for model in C.MODELS:
            if outcome[model] != "main_complete" or not self.judge_needed(model):
                continue
            if (res / "judge" / model / "DONE").exists():
                log(f"SKIP judge {model} (complete)")
                continue
            self.begin(f"judge:{model}", f"{C.MODEL_NAME[model]} 판정 (HyperInfer 답 전체)")
            done = False
            for n in range(1, MAX_PHASE_STARTS + 1):
                self.run(f"judge:{model}", [PY, JUDGE, "--res", res, "--model", model], gpu=True)
                if (res / "judge" / model / "DONE").exists():
                    done = True
                    break
                time.sleep(30)
            if not done:
                judge_missing = True
                self.error(f"judge:{model}", f"the judge run of {C.MODEL_NAME[model]} did not complete; continue with the same command")

        complete = all(o == "main_complete" for o in outcome.values()) and not judge_missing
        gate_failed = any(o == "not_admitted" for o in outcome.values())
        hard_error = any(o == "error" for o in outcome.values()) or judge_missing
        if not hard_error and not (res / "gates" / "posthoc.json").exists():
            self.begin("posthoc", "사후 확인: 기존 파일이 그대로인지")
            self.run("posthoc", [PY, VAL, "--res", res, "--stage", "posthoc"])
        return self.finish(0 if complete else (2 if gate_failed and not hard_error else 1), finished=not hard_error)

    def finish(self, rc, finished=False):
        res = self.res
        STATUS.ssd_sample(self.rel, "driver_finish")
        if finished or rc == 0:
            C.atomic_json(res / "smart_after.json", STATUS.smart_snapshot())
        outcome = {0: "done", 2: "done_with_gate_failure"}.get(rc, "stopped_on_error")
        if (res / "config.json").exists():
            self.status(stage="report", outcome=outcome)
            r = self.run("report", [PY, REPORT, "--res", res])
            if r == 0:
                self.status(report=f"{self.rel}/report.md")
            else:
                self.error("report", f"report script rc={r}")
        if rc == 0:
            self.status(stage="done", stage_title="모든 단계 종료", finished_utc=C.utc(), driver_pid=None, eta_utc=None, last_error=None)
        elif rc == 2:
            self.status(stage="done_with_gate_failure", stage_title="종료: gate에 실패한 모델이 있음 (report.md 참조)", finished_utc=C.utc(), driver_pid=None,
                        eta_utc=None, last_error=self.errors[-1] if self.errors else None)
        else:
            self.status(stage="stopped_on_error", stage_title="오류로 멈춤: 같은 명령으로 이어서 실행할 수 있음", driver_pid=None, eta_utc=None,
                        last_error=self.errors[-1] if self.errors else None, errors=self.errors[-10:])
        log(f"DRIVER END rc={rc}")
        return rc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", type=Path, required=True)
    args = p.parse_args()
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    sys.exit(Driver(args.res).main())


if __name__ == "__main__":
    main()

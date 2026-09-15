"""Fail-closed smoke -> correctness gate -> main -> analysis launcher."""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path("/home/dblab/anaconda3/envs/mllm_ft/bin/python")
INDEX = ROOT / "data/visdial_v1.0/subsets/main_seed1234/index.json"
RUN_BASE = ROOT / "runs/visdial_turn1_piggyback_e2e_ttft"
RESULT_BASE = ROOT / "results/visdial_turn1_piggyback_e2e_ttft"
STORE_BASE = ROOT / "kvstore_visdial_turn1_piggyback_e2e_ttft"


def _run(command, log_handle):
    rendered = shlex.join([str(part) for part in command])
    line = f"\n$ {rendered}\n"
    print(line, end="", flush=True)
    log_handle.write(line)
    log_handle.flush()
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    process = subprocess.Popen(
        [str(part) for part in command], cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1)
    assert process.stdout is not None
    for output in process.stdout:
        print(output, end="", flush=True)
        log_handle.write(output)
        log_handle.flush()
    code = process.wait()
    if code:
        raise SystemExit(f"FAILED ({code}): {rendered}")


def _refuse_existing(paths):
    existing = [str(path) for path in paths if os.path.lexists(path)]
    if existing:
        raise FileExistsError(
            "refusing to overwrite prior/new artifacts:\n" +
            "\n".join(existing))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--without-fullload", action="store_true",
                        help="omit the optional reference; main three arms remain")
    parser.add_argument("--min-free-after-gib", type=float, default=20.0)
    parser.add_argument(
        "--smoke-tag", default="seed1234",
        help=("artifact suffix for the smoke/correctness pair; choose a new "
              "tag to preserve and supersede an earlier smoke run"))
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.smoke_tag):
        raise ValueError("--smoke-tag must be a safe single path component")

    methods = ("recompute,prefix25,prefix45" if args.without_fullload
               else "recompute,prefix25,prefix45,fullload")
    smoke_run = RUN_BASE / f"smoke_{args.smoke_tag}"
    main_run = RUN_BASE / "main_seed1234"
    smoke_store = STORE_BASE / f"smoke_{args.smoke_tag}"
    direct_store = STORE_BASE / f"smoke_direct_{args.smoke_tag}"
    main_store = STORE_BASE / "main_seed1234"
    main_results = RESULT_BASE / "main_seed1234"
    validation = smoke_run / "captured_kv_validation.json"

    targets = [smoke_run, smoke_store, direct_store]
    if not args.smoke_only:
        targets.extend([main_run, main_store, main_results])
    _refuse_existing(targets)
    RUN_BASE.mkdir(parents=True, exist_ok=True)
    log_path = RUN_BASE / f"pipeline_{int(time.time())}.log"
    with log_path.open("x") as log:
        preflight = {
            "index": str(INDEX),
            "methods": methods,
            "smoke_tag": args.smoke_tag,
            "disk_free_bytes": shutil.disk_usage(ROOT).free,
            "smoke_run": str(smoke_run),
            "smoke_store": str(smoke_store),
            "direct_store": str(direct_store),
            "main_run": None if args.smoke_only else str(main_run),
            "main_store": None if args.smoke_only else str(main_store),
            "main_results": None if args.smoke_only else str(main_results),
        }
        log.write(json.dumps(preflight, indent=1) + "\n")
        log.flush()
        if not args.skip_tests:
            _run([PYTHON, "-m", "unittest", "discover", "-s", "tests", "-v"],
                 log)
        _run([
            PYTHON, "scripts/28_eval_visdial_turn1_piggyback.py",
            "--index", INDEX, "--store", smoke_store,
            "--run-dir", smoke_run, "--max-dialogs", "2",
            "--max-turns", "10", "--max-new-tokens", "16",
            "--seed", "1234", "--methods", methods,
            "--min-free-after-gib", str(args.min_free_after_gib),
        ], log)
        _run([
            PYTHON, "scripts/30_validate_visdial_turn1_piggyback.py",
            "--smoke-run", smoke_run, "--captured-store", smoke_store,
            "--direct-store", direct_store, "--index", INDEX,
            "--output", validation,
        ], log)
        if not args.smoke_only:
            _run([
                PYTHON, "scripts/28_eval_visdial_turn1_piggyback.py",
                "--index", INDEX, "--store", main_store,
                "--run-dir", main_run, "--max-dialogs", "100",
                "--max-turns", "10", "--max-new-tokens", "16",
                "--seed", "1234", "--methods", methods,
                "--correctness-validation", validation,
                "--min-free-after-gib", str(args.min_free_after_gib),
            ], log)
            _run([
                PYTHON, "scripts/29_analyze_visdial_turn1_piggyback.py",
                "--run-dir", main_run, "--results-dir", main_results,
            ], log)

    print(json.dumps({
        "status": "complete", "log": str(log_path),
        "smoke_validation": str(validation),
        "main_results": None if args.smoke_only else str(main_results),
    }, indent=1), flush=True)


if __name__ == "__main__":
    main()

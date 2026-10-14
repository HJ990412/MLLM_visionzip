#!/usr/bin/env python3
"""Reconstruct GLM metrics from adopted raw; never starts model inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmimpress.glm46v.analysis import (
    EXPECTED_MAIN_HASH, METHODS, analyze_rows, file_sha256, jsonl,
    load_adopted_rows, load_adopted_setup_events, workload_hash, write_report,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--phase", choices=("main", "gqa_pilot", "mt_pilot"), default="main")
    p.add_argument("--output", type=Path, required=True, help="New output directory; no overwrite")
    p.add_argument("--raw", type=Path, help="Explicit adopted raw JSONL (audited for complete membership)")
    p.add_argument("--setup-events", type=Path, help="Adopted physical build/activation JSONL")
    a = p.parse_args()
    run = a.run.resolve()
    manifest = run / "manifests" / (a.phase + "_manifest.json")
    frozen = json.loads(manifest.read_text())
    if a.phase == "main" and workload_hash(frozen["dialogues"]) != EXPECTED_MAIN_HASH:
        raise ValueError("main ordered workload SHA256 differs from the experiment contract")
    if a.raw:
        rows = jsonl(a.raw)
        evidence = [dict(path=str(a.raw.resolve()), sha256=file_sha256(a.raw), requests=len(rows),
                         adoption="explicit CLI input; independent complete-membership audit required")]
    else:
        rows, evidence = load_adopted_rows(run, a.phase)
    if a.setup_events:
        events = jsonl(a.setup_events)
        evidence.append(dict(path=str(a.setup_events.resolve()), sha256=file_sha256(a.setup_events),
                             kind="adopted setup events"))
    else:
        events, setup_evidence = load_adopted_setup_events(run, a.phase)
        evidence.extend(setup_evidence)
    result = analyze_rows(rows, frozen, events, phase=a.phase, methods=METHODS)
    result["manifest_sha256"] = file_sha256(manifest)
    result["analysis_source_sha256"] = {
        "mmimpress/glm46v/analysis.py": file_sha256(ROOT / "mmimpress/glm46v/analysis.py"),
        "scripts/122_report_glm46v_prefixstore3.py": file_sha256(Path(__file__)),
    }
    write_report(result, a.output, phase=a.phase, evidence=evidence)
    print(json.dumps(dict(status=result["status"], adopted_requests=result["audit"]["requests"],
                          expected_requests=result["audit"]["expected_requests"],
                          report=str(a.output / "REPORT.md")), ensure_ascii=False))
    if result["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

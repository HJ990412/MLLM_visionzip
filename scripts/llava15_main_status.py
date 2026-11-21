#!/usr/bin/env python3
"""status.json of the LLaVA-1.5 x MT-GQA main experiment (docs/llava15_mt_gqa_main_contract.md).

One small JSON file, ``results/<run name>/status.json``, rewritten atomically by
whoever is working: the driver (stage changes, gate results, errors) and the
evaluation process (request counts, estimated end).  Fields are merged, so a
writer only passes what it knows.

  python scripts/llava15_main_status.py <run> key=value [key=value ...]
      values are parsed as JSON when possible, else kept as strings
  python scripts/llava15_main_status.py <run> --show
  python scripts/llava15_main_status.py <run> --ssd-sample <label>          (appends to <run>/ssd_samples.jsonl)
  python scripts/llava15_main_status.py <run> --invocation <pid> <fresh|resume>
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def utc(ts):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def path_of(run):
    return ROOT / "results" / Path(run).name / "status.json"


def read(run):
    p = path_of(run)
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def update(run, **fields):
    """Merge ``fields`` into status.json (atomic replace); returns the document."""
    p = path_of(run)
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = read(run)
    now = time.time()
    doc.setdefault("run", Path(run).name)
    doc.setdefault("started_utc", utc(now))
    doc.update(fields)
    doc["updated_utc"] = utc(now)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False, default=str) + "\n")
    os.replace(tmp, p)
    return doc


def boot_id():
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


def ssd_sample(run, label, device="nvme0n1", **extra):
    """Append the block device's cumulative counters to ``<run>/ssd_samples.jsonl``.

    ``/sys/block/<device>/stat`` restarts at every boot, so each sample carries
    the boot id; the report sums (last - first) sectors written per boot.
    """
    try:
        stat = Path(f"/sys/block/{device}/stat").read_text().split()
        doc = {"at_unix": time.time(), "label": label, "device": device, "boot_id": boot_id(),
               "sectors_read": int(stat[2]), "sectors_written": int(stat[6]), "sector_bytes": 512, **extra}
    except (OSError, ValueError, IndexError) as exc:
        doc = {"at_unix": time.time(), "label": label, "device": device, "boot_id": boot_id(), "error": repr(exc), **extra}
    with (Path(run) / "ssd_samples.jsonl").open("a") as f:
        f.write(json.dumps(doc) + "\n")
    return doc


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    run = argv[0]
    if argv[1] == "--show":
        print(json.dumps(read(run), indent=1, ensure_ascii=False))
        return
    if argv[1] == "--ssd-sample":
        ssd_sample(ROOT / run if not Path(run).is_absolute() else run, argv[2])
        return
    if argv[1] == "--invocation":      # one line per driver start: <run>/driver_invocations.jsonl
        doc = {"at_unix": time.time(), "at_utc": utc(time.time()), "boot_id": boot_id(), "driver_pid": int(argv[2]), "kind": argv[3],
               "stage_at_start": read(run).get("stage")}
        with ((ROOT / run if not Path(run).is_absolute() else Path(run)) / "driver_invocations.jsonl").open("a") as f:
            f.write(json.dumps(doc) + "\n")
        return
    fields = {}
    for item in argv[1:]:
        key, _, value = item.partition("=")
        try:
            fields[key] = json.loads(value)
        except ValueError:
            fields[key] = value
    update(run, **fields)


if __name__ == "__main__":
    main(sys.argv[1:])

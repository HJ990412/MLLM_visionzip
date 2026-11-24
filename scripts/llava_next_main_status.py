#!/usr/bin/env python3
"""status.json and SSD bookkeeping of the LLaVA-NeXT x MT-VQA main experiment
(docs/llava_next_mtvqa_main_contract.md).

``results/<run name>/status.json`` is one small JSON file rewritten atomically by
whoever is working: the driver (stage changes, gate results, errors) and the
evaluation process (image counts, estimated end, SSD writes, SSD temperature).
Fields are merged, so a writer only passes what it knows.

  python scripts/llava_next_main_status.py <run> key=value [key=value ...]
      values are parsed as JSON when possible, else kept as strings
  python scripts/llava_next_main_status.py <run> --show
  python scripts/llava_next_main_status.py <run> --ssd-sample <label>     (appends to <run>/ssd_samples.jsonl)
  python scripts/llava_next_main_status.py <run> --invocation <pid> <fresh|resume>
  python scripts/llava_next_main_status.py <run> --smart <label>          (writes results/<run name>/smart_<label>.json)

SSD numbers
  * written bytes: ``/sys/block/<device>/stat`` (sectors written, whole device, every
    process).  The counter restarts at boot, so the run keeps its own running total in
    ``<run>/ssd_counter.json`` (boot id + last counter value); writes between the last
    sample before a reboot and the reboot itself are not seen.
  * temperature: the NVMe hwmon sensors (``temp1`` = composite), readable without root.
  * SMART (percentage used, data units written): needs the NVMe admin command, i.e.
    root (``nvme smart-log`` / ``smartctl``).  ``smart_snapshot`` tries the tools that
    exist, without and with ``sudo -n``, and records NOT_MEASURED with the reason
    otherwise.  A file ``results/<run name>/smart_manual_<label>.txt`` written by a
    person (the output of ``sudo nvme smart-log /dev/nvme0``) is picked up by the report.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTOR = 512


def utc(ts):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def run_dir(run):
    run = Path(run)
    return run if run.is_absolute() else ROOT / run


def path_of(run):
    return ROOT / "results" / Path(run).name / "status.json"


def read(run):
    try:
        return json.loads(path_of(run).read_text())
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


# ------------------------------------------------------------------- device
def block_device(path=ROOT):
    """(whole-disk name, controller name) of the file system that holds ``path``,
    e.g. ("nvme0n1", "nvme0"); falls back to those names."""
    try:
        st = os.stat(path)
        node = Path(f"/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}").resolve()
        disk = node.parent.name if (node / "partition").exists() else node.name
        m = re.match(r"^(nvme\d+)n\d+$", disk)
        return disk, (m.group(1) if m else None)
    except OSError:
        return "nvme0n1", "nvme0"


def sectors(device=None):
    device = device or block_device()[0]
    stat = Path(f"/sys/block/{device}/stat").read_text().split()
    return {"device": device, "sectors_read": int(stat[2]), "sectors_written": int(stat[6])}


def ssd_temperature(controller=None):
    """{"composite_c": ..., "sensors_c": {...}} from the NVMe hwmon node; None values when unreadable."""
    controller = controller or block_device()[1] or "nvme0"
    out = {}
    for p in sorted(Path(f"/sys/class/nvme/{controller}").glob("hwmon*/temp*_input")):
        try:
            out[p.name.replace("_input", "")] = int(p.read_text()) / 1000.0
        except (OSError, ValueError):
            pass
    return {"composite_c": out.get("temp1"), "sensors_c": out or None, "source": f"/sys/class/nvme/{controller}/hwmon*/temp*_input"}


def ssd_counter(run, label=None):
    """Advance the run's running total of device writes; returns the state.

    Same boot as the previous call: the difference of the kernel counter is
    added.  A new boot: nothing is added for the gap (unknown) and the counter
    restarts from the current value.
    """
    run = run_dir(run)
    path = run / "ssd_counter.json"
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError):
        state = {"written_bytes": 0, "read_bytes": 0, "boots": 0, "boot_id": None, "first_at_unix": time.time()}
    try:
        now = sectors()
    except (OSError, ValueError, IndexError) as exc:
        state["error"] = repr(exc)
        return state
    bid = boot_id()
    if state.get("boot_id") == bid and state.get("device") == now["device"]:
        state["written_bytes"] += max(0, now["sectors_written"] - state["sectors_written"]) * SECTOR
        state["read_bytes"] += max(0, now["sectors_read"] - state["sectors_read"]) * SECTOR
    else:
        state["boots"] = int(state.get("boots", 0)) + 1
    state.update(boot_id=bid, device=now["device"], sectors_written=now["sectors_written"], sectors_read=now["sectors_read"],
                 at_unix=time.time(), label=label)
    state.pop("error", None)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(state) + "\n")
    os.replace(tmp, path)
    return state


def ssd_sample(run, label, **extra):
    """Append the device counters, the running total and the temperature to ``<run>/ssd_samples.jsonl``."""
    run = run_dir(run)
    state = ssd_counter(run, label)
    temp = ssd_temperature()
    doc = {"at_unix": time.time(), "label": label, "device": state.get("device"), "boot_id": state.get("boot_id"),
           "sectors_written": state.get("sectors_written"), "sectors_read": state.get("sectors_read"), "sector_bytes": SECTOR,
           "run_written_bytes": state.get("written_bytes"), "run_read_bytes": state.get("read_bytes"),
           "ssd_temperature_c": temp["composite_c"], "ssd_sensors_c": temp["sensors_c"],
           "free_bytes": shutil.disk_usage(run).free, **extra}
    with (run / "ssd_samples.jsonl").open("a") as f:
        f.write(json.dumps(doc) + "\n")
    return doc


# -------------------------------------------------------------------- SMART
SMART_FIELDS = (("percentage_used", r"percentage[ _]used\s*[:=]\s*(\d+)\s*%?"),
                ("data_units_written", r"data[ _]units[ _]written\s*[:=]\s*([\d,\.]+)"),
                ("temperature", r"^temperature\s*[:=]\s*(\d+)"),
                ("available_spare", r"available[ _]spare\s*[:=]\s*(\d+)\s*%?"),
                ("media_errors", r"media(?:[ _]and[ _]data[ _]integrity)?[ _]errors\s*[:=]\s*(\d+)"),
                ("power_on_hours", r"power[ _]on[ _]hours\s*[:=]\s*([\d,]+)"))


def parse_smart_text(text):
    """Fields of ``nvme smart-log`` / ``smartctl -a`` text output (best effort)."""
    out = {}
    for key, pattern in SMART_FIELDS:
        m = re.search(pattern, text, re.I | re.M)
        if m:
            out[key] = int(m.group(1).replace(",", "").split(".")[0])
    if "data_units_written" in out:      # one data unit = 1000 x 512 bytes
        out["data_units_written_bytes"] = out["data_units_written"] * 512000
    return out


def smart_snapshot(controller=None):
    """SMART health values of the SSD, or NOT_MEASURED with the reason for every attempt."""
    controller = controller or block_device()[1] or "nvme0"
    dev = f"/dev/{controller}"
    attempts = []
    commands = []
    if shutil.which("nvme"):
        commands += [["nvme", "smart-log", dev], ["sudo", "-n", "nvme", "smart-log", dev]]
    if shutil.which("smartctl"):
        commands += [["smartctl", "-a", dev], ["sudo", "-n", "smartctl", "-a", dev]]
    for cmd in commands:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            attempts.append({"command": " ".join(cmd), "error": repr(exc)})
            continue
        fields = parse_smart_text(p.stdout) if p.returncode == 0 else {}
        attempts.append({"command": " ".join(cmd), "returncode": p.returncode, "stderr": p.stderr.strip()[-300:]})
        if "percentage_used" in fields or "data_units_written" in fields:
            return {"status": "MEASURED", "at_unix": time.time(), "at_utc": utc(time.time()), "device": dev, "command": " ".join(cmd),
                    **fields, "raw": p.stdout[-4000:], "hwmon_temperature": ssd_temperature(controller)}
    if not commands:
        attempts.append({"command": None, "error": "neither nvme-cli nor smartmontools is installed"})
    try:
        mode = oct(os.stat(dev).st_mode & 0o777)
    except OSError as exc:
        mode = repr(exc)
    return {"status": "NOT_MEASURED", "at_unix": time.time(), "at_utc": utc(time.time()), "device": dev, "device_node_mode": mode,
            "reason": "the SMART / health log is an NVMe admin command (root only); this account has no passwordless sudo "
                      "and no SMART tool is installed",
            "attempts": attempts, "percentage_used": None, "data_units_written": None,
            "available_without_root": {"hwmon_temperature": ssd_temperature(controller),
                                       "kernel_counter_since_boot": _safe(sectors), "boot_id": boot_id()},
            "how_to_supply": f"sudo nvme smart-log {dev} > results/<run name>/smart_manual_<before|after>.txt   (read by the report)"}


def _safe(fn):
    try:
        return fn()
    except Exception as exc:      # noqa: BLE001  (monitoring only)
        return {"error": repr(exc)}


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    run = argv[0]
    if argv[1] == "--show":
        print(json.dumps(read(run), indent=1, ensure_ascii=False))
        return
    if argv[1] == "--ssd-sample":
        doc = ssd_sample(run, argv[2])
        update(run, ssd_written_bytes=doc.get("run_written_bytes"),
               ssd_written_gb=round((doc.get("run_written_bytes") or 0) / 1e9, 2), ssd_temperature_c=doc.get("ssd_temperature_c"))
        return
    if argv[1] == "--invocation":      # one line per driver start: <run>/driver_invocations.jsonl
        doc = {"at_unix": time.time(), "at_utc": utc(time.time()), "boot_id": boot_id(), "driver_pid": int(argv[2]), "kind": argv[3],
               "stage_at_start": read(run).get("stage")}
        with (run_dir(run) / "driver_invocations.jsonl").open("a") as f:
            f.write(json.dumps(doc) + "\n")
        return
    if argv[1] == "--smart":
        doc = smart_snapshot()
        out = ROOT / "results" / Path(run).name / f"smart_{argv[2]}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
        print(json.dumps({"smart": doc["status"], "file": str(out.relative_to(ROOT))}))
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

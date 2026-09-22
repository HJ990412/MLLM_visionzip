#!/usr/bin/env python3
"""Record and verify prior run/result artifacts around ConvBench work."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "runs/convbench_full/protected_artifacts_before.json"
EXCLUDED = {
    ROOT / "runs/convbench_full",
    ROOT / "runs/convbench_judge",
    ROOT / "runs/convbench_context",
    ROOT / "runs/convbench_native_probe",
    ROOT / "runs/convbench_context_quality",
    ROOT / "results/convbench_full",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def snapshot() -> dict:
    entries = {}
    for base in (ROOT / "runs", ROOT / "results"):
        for parent, dirs, files in os.walk(base, followlinks=False):
            directory = Path(parent)
            kept = []
            for name in sorted(dirs):
                path = directory / name
                if path in EXCLUDED:
                    continue
                rel = path.relative_to(ROOT).as_posix()
                if path.is_symlink():
                    entries[rel] = {"kind": "symlink", "target": os.readlink(path)}
                else:
                    entries[rel] = {"kind": "directory"}
                    kept.append(name)
            dirs[:] = kept
            for name in sorted(files):
                path = directory / name
                rel = path.relative_to(ROOT).as_posix()
                if path.is_symlink():
                    entries[rel] = {"kind": "symlink", "target": os.readlink(path)}
                elif path.is_file():
                    entries[rel] = {"kind": "file", "size": path.stat().st_size,
                                    "sha256": digest(path)}
                else:
                    raise ValueError(f"unexpected prior artifact type: {path}")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--before", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    observed = snapshot()
    payload = {"schema_version": "convbench-protected-artifacts-v1",
               "scope": "preexisting runs/ and results/ excluding new ConvBench output roots",
               "entries": observed}
    if args.before:
        if MANIFEST.exists():
            raise FileExistsError(f"will not replace protection manifest: {MANIFEST}")
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=MANIFEST.parent,
                                         prefix=".protected-", delete=False,
                                         encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
            temporary = Path(f.name)
        try:
            os.link(temporary, MANIFEST)
        finally:
            temporary.unlink()
        print(json.dumps({"status": "recorded", "entries": len(observed),
                          "manifest": str(MANIFEST)}))
    else:
        prior = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if prior["entries"] != observed:
            missing = sorted(set(prior["entries"]) - set(observed))
            added = sorted(set(observed) - set(prior["entries"]))
            changed = sorted(k for k in set(observed) & set(prior["entries"])
                             if observed[k] != prior["entries"][k])
            raise RuntimeError(json.dumps({"status": "changed", "missing": missing,
                                           "added": added, "changed": changed}))
        print(json.dumps({"status": "unchanged", "entries": len(observed)}))


if __name__ == "__main__":
    main()

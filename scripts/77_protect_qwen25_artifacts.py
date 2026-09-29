"""Hash a declared set of pre-existing LLaVA artifacts before/after Qwen work.

Usage:
  python scripts/77_protect_qwen25_artifacts.py capture OUT.json
  python scripts/77_protect_qwen25_artifacts.py verify BEFORE.json AFTER.json

The protected set intentionally excludes new Qwen files and large historical
``runs/`` payloads. It includes the complete original data tree, one complete
canonical LLaVA store tree, and the complete existing results tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TREE_ROOTS = ("data", "kvstore_image_only_visionzip", "results")
SOURCE_ROOTS = ("mmimpress", "scripts", "tests", "docs")
LEGACY_RUNS = (
    "runs/gqa40_240_true_ttft_sanity",
    "runs/image_only_repack_budget_sweep_with_recomp/main_20_50",
)


def protected_paths():
    for name in ("README.md", ".gitignore"):
        path = ROOT / name
        if path.is_file():
            yield path
    for name in SOURCE_ROOTS:
        for path in sorted((ROOT / name).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(ROOT).as_posix()
            if rel.startswith("mmimpress/qwen25/"):
                continue
            if rel.startswith("tests/test_qwen25_"):
                continue
            if rel == "docs/qwen25_port_contract.md":
                continue
            if rel.startswith("scripts/77_") or rel.startswith("scripts/78_") \
                    or rel.startswith("scripts/79_") or rel.startswith("scripts/80_"):
                continue
            yield path
    for name in (*TREE_ROOTS, *LEGACY_RUNS):
        base = ROOT / name
        if base.exists():
            for path in sorted(base.rglob("*")):
                rel = path.relative_to(ROOT).as_posix()
                if rel.startswith("results/qwen25_port_"):
                    continue
                if path.is_file():
                    yield path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def capture(paths):
    entries = {}
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        entries[rel] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {"root": str(ROOT), "scope": {
        "trees": TREE_ROOTS, "source": SOURCE_ROOTS, "legacy_runs": LEGACY_RUNS,
        "excludes": ["new Qwen files", "other historical run payloads"],
    }, "files": entries, "file_count": len(entries),
        "total_bytes": sum(x["bytes"] for x in entries.values())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("capture", "verify"))
    p.add_argument("first", type=Path)
    p.add_argument("second", type=Path, nargs="?")
    args = p.parse_args()
    if args.mode == "capture":
        manifest = capture(protected_paths())
        args.first.parent.mkdir(parents=True, exist_ok=True)
        args.first.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        print(json.dumps({k: manifest[k] for k in ("file_count", "total_bytes")}))
        return
    if args.second is None:
        p.error("verify needs BEFORE.json AFTER.json")
    before = json.loads(args.first.read_text())
    after = capture(ROOT / rel for rel in before["files"])
    after["unexpected_protected_paths"] = sorted(
        {p.relative_to(ROOT).as_posix() for p in protected_paths()}
        - set(before["files"]))
    after["unchanged"] = (before["files"] == after["files"]
                          and not after["unexpected_protected_paths"])
    args.second.parent.mkdir(parents=True, exist_ok=True)
    args.second.write_text(json.dumps(after, indent=2, sort_keys=True))
    print(json.dumps({"unchanged": after["unchanged"],
                      "file_count": after["file_count"],
                      "total_bytes": after["total_bytes"],
                      "unexpected_protected_paths": after["unexpected_protected_paths"]}))
    if not after["unchanged"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

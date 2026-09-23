#!/usr/bin/env python3
"""Build or validate the frozen MT-VQA-v2-reconstructed workload."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mmimpress.mt_vqa_v2 import (  # noqa: E402
    ARTIFACT_FILENAMES,
    DEFAULT_SEED,
    build_artifacts,
    validate_artifact_directory,
    write_artifacts_no_clobber,
)


DEFAULT_SOURCE_INDEX = ROOT / "data/vqav2/index.json"
DEFAULT_SOURCE_CONFIG = ROOT / "data/vqav2/config.json"
DEFAULT_OUTPUT_DIR = ROOT / "data/mt_vqa_v2"


def _presence(output: Path) -> tuple[list[Path], list[Path]]:
    present, missing = [], []
    for name in ARTIFACT_FILENAMES:
        path = output / name
        (present if path.exists() else missing).append(path)
    return present, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path,
                        default=DEFAULT_SOURCE_INDEX)
    parser.add_argument("--source-config", type=Path,
                        default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--validate-only", "--validate",
                      dest="validate_only", action="store_true")
    args = parser.parse_args()
    args.source_index = args.source_index.resolve()
    args.source_config = args.source_config.resolve()
    args.out_dir = args.out_dir.resolve()
    present, missing = _presence(args.out_dir)
    if present and missing:
        raise RuntimeError(
            "partial MT-VQA-v2 artifact set; refusing implicit repair")
    if args.validate_only:
        if missing:
            raise FileNotFoundError("validation requires all four artifacts")
        report = validate_artifact_directory(
            args.out_dir, source_index=args.source_index,
            source_config=args.source_config, strict_canonical=True)
        print(json.dumps({"status": "validated", **report}, indent=2,
                         sort_keys=True))
        return 0
    artifacts, summary = build_artifacts(
        args.source_index, args.source_config, seed=args.seed,
        strict_canonical=True)
    if present:
        report = validate_artifact_directory(
            args.out_dir, source_index=args.source_index,
            source_config=args.source_config, strict_canonical=True)
        print(json.dumps({
            "status": ("dry_run_existing_identical" if args.dry_run
                       else "existing_identical_noop"),
            **summary, "validation": report,
        }, indent=2, sort_keys=True))
        return 0
    if args.dry_run:
        print(json.dumps({"status": "dry_run_no_write", **summary},
                         indent=2, sort_keys=True))
        return 0
    hashes = write_artifacts_no_clobber(args.out_dir, artifacts)
    report = validate_artifact_directory(
        args.out_dir, source_index=args.source_index,
        source_config=args.source_config, strict_canonical=True)
    print(json.dumps({
        "status": "created", "out_dir": str(args.out_dir), **summary,
        "written_sha256": hashes, "validation": report,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build or validate the official-image-available ConvBench dataset index."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mmimpress.convbench import (build_index, validate_index,  # noqa: E402
                                write_artifacts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path,
                        default=ROOT / "data/convbench_source")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "data/convbench")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    names = ("index.json", "config.json", "provenance.json")
    if args.validate_only:
        artifacts = [json.loads((args.out_dir / name).read_text(encoding="utf-8"))
                     for name in names]
        print(json.dumps(validate_index(*artifacts, args.source_dir), indent=2))
        return
    artifacts = build_index(args.source_dir)
    payload = dict(zip(names, artifacts, strict=True))
    if args.out_dir.exists():
        existing = [json.loads((args.out_dir / name).read_text(encoding="utf-8"))
                    for name in names]
        report = validate_index(*existing, args.source_dir)
        print(json.dumps({"status": "existing_identical", **report}, indent=2))
        return
    hashes = write_artifacts(args.out_dir, payload)
    report = validate_index(*artifacts, args.source_dir)
    print(json.dumps({"status": "created", "sha256": hashes, **report}, indent=2))


if __name__ == "__main__":
    main()

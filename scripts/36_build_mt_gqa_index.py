#!/usr/bin/env python3
"""Build or validate the canonical MT-GQA-reconstructed index.

This command is local-only: it reads the frozen GQA testdev-balanced JSON and
JPG directory, performs no Hugging Face or network access, and publishes four
durable JSON artifacts under ``data/mt_gqa``.  Existing identical artifacts
validate as a no-op; partial or different artifacts fail closed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mmimpress.mt_gqa import (  # noqa: E402
    ARTIFACT_FILENAMES,
    DEFAULT_SEED,
    EXPECTED_DIALOGUES,
    EXPECTED_SOURCE_SHA256,
    build_artifacts,
    sha256_file,
    validate_artifact_directory,
    write_artifacts_no_clobber,
)


DEFAULT_GQA_DATA = (
    ROOT.parent / "SparseVLMs" / "playground" / "data" / "eval" / "gqa"
    / "data"
)
DEFAULT_QUESTIONS = DEFAULT_GQA_DATA / "testdev_balanced_questions.json"
DEFAULT_IMAGE_DIR = DEFAULT_GQA_DATA / "images"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "mt_gqa"


def _artifact_presence(output_dir: Path) -> tuple[list[Path], list[Path]]:
    present, missing = [], []
    for name in ARTIFACT_FILENAMES:
        path = output_dir / name
        (present if path.exists() else missing).append(path)
    return present, missing


def _check_expected(summary: dict, *, expected_dialogues: int,
                    expected_source_sha256: str) -> None:
    if summary["source_questions_sha256"] != expected_source_sha256:
        raise ValueError(
            "source SHA256 mismatch: expected "
            f"{expected_source_sha256}, got "
            f"{summary['source_questions_sha256']}")
    if summary["n_dialogues"] != expected_dialogues:
        raise ValueError(
            f"dialogue count mismatch: expected {expected_dialogues}, "
            f"got {summary['n_dialogues']}")


def _validate_existing(args: argparse.Namespace) -> dict:
    validation = validate_artifact_directory(
        args.out_dir,
        source_questions=args.questions,
        image_dir=args.image_dir,
        strict_canonical=True,
    )
    observed = {
        "source_questions_sha256": validation["source_questions_sha256"],
        "n_dialogues": validation["dialogues"],
    }
    _check_expected(
        observed,
        expected_dialogues=args.expected_dialogues,
        expected_source_sha256=args.expected_source_sha256,
    )
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic MT-GQA-reconstructed artifacts")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--expected-dialogues", type=int, default=EXPECTED_DIALOGUES)
    parser.add_argument(
        "--expected-source-sha256", default=EXPECTED_SOURCE_SHA256)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true",
        help="reconstruct and validate in memory without writing")
    mode.add_argument(
        "--validate-only", "--validate", dest="validate_only",
        action="store_true", help="validate the existing four artifacts")
    args = parser.parse_args()

    args.questions = args.questions.resolve()
    args.image_dir = args.image_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.expected_dialogues < 1:
        parser.error("--expected-dialogues must be positive")
    if len(args.expected_source_sha256) != 64:
        parser.error("--expected-source-sha256 must be a 64-character SHA256")
    if not args.questions.is_file():
        raise FileNotFoundError(args.questions)
    if not args.image_dir.is_dir():
        raise FileNotFoundError(args.image_dir)
    observed_source_sha = sha256_file(args.questions)
    if observed_source_sha != args.expected_source_sha256:
        raise ValueError(
            "source SHA256 mismatch before reconstruction: expected "
            f"{args.expected_source_sha256}, got {observed_source_sha}")

    present, missing = _artifact_presence(args.out_dir)
    if present and missing:
        raise RuntimeError(
            "partial MT-GQA artifact set; refusing build/validation. Present: "
            f"{[str(path) for path in present]}; missing: "
            f"{[str(path) for path in missing]}")

    if args.validate_only:
        if missing:
            raise FileNotFoundError(
                "validation requires all artifacts: "
                + ", ".join(str(path) for path in missing))
        report = _validate_existing(args)
        print(json.dumps({"status": "validated", **report}, indent=2,
                         sort_keys=True))
        return

    artifacts, summary = build_artifacts(
        args.questions,
        args.image_dir,
        seed=args.seed,
        strict_canonical=True,
    )
    _check_expected(
        summary,
        expected_dialogues=args.expected_dialogues,
        expected_source_sha256=args.expected_source_sha256,
    )

    if present:
        report = _validate_existing(args)
        status = "dry_run_existing_identical" if args.dry_run else \
            "existing_identical_noop"
        print(json.dumps({"status": status, **summary,
                          "validation": report}, indent=2, sort_keys=True))
        return
    if args.dry_run:
        print(json.dumps({"status": "dry_run_no_write", **summary},
                         indent=2, sort_keys=True))
        return

    written_hashes = write_artifacts_no_clobber(args.out_dir, artifacts)
    report = _validate_existing(args)
    print(json.dumps({
        "status": "created",
        "out_dir": str(args.out_dir),
        **summary,
        "written_sha256": written_hashes,
        "validation": report,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

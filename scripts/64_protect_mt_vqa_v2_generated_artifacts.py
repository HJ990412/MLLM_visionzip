#!/usr/bin/env python3
"""MT-VQA-v2 path adapter for the validated prior-artifact protector."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BASE_PATH = ROOT / "scripts/57_protect_mt_gqa_history_artifacts.py"


def _load_base():
    spec = importlib.util.spec_from_file_location(
        "_mt_vqa_v2_reused_artifact_protector", BASE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
BASE.PREFIX = "mt_vqa_v2_generated_4arm_"
BASE.SCHEMA_VERSION = "mt-vqa-v2-generated-protected-artifacts-v1"


def main(argv: list[str] | None = None) -> int:
    return BASE.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Shared definitions of the ConvBench HyperInfer run (docs/convbench_hyperinfer_contract.md).

No torch import here: the driver, the gate checks and the report use this module without a GPU.

The ConvBench runner, judge and bookkeeping of the main experiment (scripts/217, scripts/219,
scripts/convbench_main_common.py; results/convbench_main_20261122T103401Z) are used as they are.
They read their arm and phase tables from ``convbench_main_common``; this module loads a PRIVATE copy
of that module, puts this run's tables into the copy and hands the copy to the scripts it loads
(``load_with_common``).  The module ``convbench_main_common`` that other code imports is not touched.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name, filename):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("_convbench_hyperinfer_main_common", "convbench_main_common.py")      # private copy

# ---- the earlier runs this one reads (never written)
MAIN_RUN = ROOT / "results/convbench_main_20261122T103401Z"                      # ReComp, FullLoad, IMPRESS, Ours (2026-11-22 .. 24)
MAIN_CONTRACT = ROOT / "docs/convbench_main_contract.md"
MTGQA_RUN = ROOT / "runs/mt_gqa_hyperinfer_v2_20261125T034057Z"                  # the HyperInfer implementation and settings
MTGQA_RES = ROOT / "results/mt_gqa_hyperinfer_v2_20261125T034057Z"
MTGQA_CONTRACT = ROOT / "docs/mt_gqa_hyperinfer_v2_contract.md"
PORT_TABLE = ROOT / "docs/hyperinfer_port_contract.md"
MTGQA_MODEL = {"next": "next", "llava15": "l15"}                                 # model keys of the MT-GQA run
DRIVER = "scripts/run_convbench_hyperinfer.sh"

# ---- arms.  FullLoad and IMPRESS keep the role names of the main experiment (their recorded rows are read by role).
FULL, IMP, RECOMP, OURS = C.FULL, C.IMP, C.RECOMP, C.OURS
HYPER, HYPER_OFF, HYPER_REPEAT, HYPER_NAN = "hyperinfer", "hyperinfer_off", "hyperinfer_repeat", "hyperinfer_nan"
# arm ids of scripts/228 (the request function of the MT-GQA run is called with these)
BACKEND_ARM = {HYPER: "hyperinfer_async", HYPER_OFF: "hyperinfer_off", HYPER_REPEAT: "hyperinfer_async_repeat", HYPER_NAN: "hyperinfer_async_nan"}
PREFETCH_ROLES = [HYPER, HYPER_REPEAT, HYPER_NAN]                                # arms whose prefetch is on
MAIN_ROLES = [HYPER]
PILOT_ROLES = [FULL, IMP, HYPER]
GATE_ROLES = [FULL, IMP, HYPER_OFF, HYPER, HYPER_REPEAT, HYPER_NAN]
FIVE = [RECOMP, FULL, IMP, HYPER, OURS]                                          # the methods of the report, in table order
DISPLAY = {RECOMP: "ReComp", FULL: "FullLoad", IMP: "IMPRESS", HYPER: "HyperInfer", OURS: "Ours",
           HYPER_OFF: "HyperInfer, prefetch 끔", HYPER_REPEAT: "HyperInfer, 두 번째 실행", HYPER_NAN: "HyperInfer, 읽지 않은 행 NaN"}
PHASES = ("gate", "pilot", "overflow", "main")
PHASE_ROLES = {"gate": GATE_ROLES, "pilot": PILOT_ROLES, "overflow": MAIN_ROLES, "main": MAIN_ROLES}

# ---- this run's tables in the private copy (conversation lists, rotation rule, statistics and file layout stay as they are)
C.SCHEMA = "convbench-hyperinfer-v1"
C.RUN_PREFIX = "convbench_hyperinfer_"
C.CONTRACT = ROOT / "docs/convbench_hyperinfer_contract.md"
C.MAIN_ROLES, C.GATE_ROLES, C.DISPLAY, C.PHASES, C.PHASE_ROLES = MAIN_ROLES, GATE_ROLES, DISPLAY, PHASES, PHASE_ROLES

MODELS, MODEL_NAME = C.MODELS, C.MODEL_NAME
N_CONV = C.N_CONV
SHOW = 20


def load_with_common(name, filename):
    """Load scripts/<filename> as a private module whose ``import convbench_main_common`` binds to this run's copy."""
    if name in sys.modules:
        return sys.modules[name]
    saved = sys.modules.get("convbench_main_common")
    sys.modules["convbench_main_common"] = C
    try:
        return _load(name, filename)
    finally:
        if saved is None:
            sys.modules.pop("convbench_main_common", None)
        else:
            sys.modules["convbench_main_common"] = saved


# ---- the recorded rows of the main experiment
def recorded_doc(model, conversation_id, phase="main"):
    """The committed conversation of the main experiment (four methods), or None."""
    path = MAIN_RUN / model / phase / "conversations" / f"{C.safe_name(conversation_id)}.json.gz"
    return C.read_gz_json(path) if path.exists() else None


def recorded_rows(model, conversation_id):
    """{(role, turn): row} of the main experiment for one conversation."""
    doc = recorded_doc(model, conversation_id)
    return None if doc is None else C.rows_by_key(doc)


def impress_kind(doc):
    """Store kind of the IMPRESS-layout store in a committed conversation (the one that is not the FullLoad or the Ours store)."""
    kinds = [k for k in doc["stores"] if k != "raster" and not k.startswith("packed")]
    assert len(kinds) == 1, kinds
    return kinds[0]


def tokens(row):
    return row["generated_token_ids"] if row is not None and row.get("status") == "ok" else None


def verify_freeze(res):
    """``verify_freeze`` of the main experiment plus the documents this run froze in addition."""
    config = C.verify_freeze(res)
    if "_dev_" not in Path(res).name:
        for key, path in (("main_contract_sha256", MAIN_CONTRACT), ("mt_gqa_contract_sha256", MTGQA_CONTRACT), ("port_table_sha256", PORT_TABLE)):
            if C.sha(path) != config[key]:
                raise RuntimeError(f"FROZEN_DOCUMENT_MISMATCH: {path.name}")
    if config["phase_roles"] != PHASE_ROLES or config["backend_arm"] != BACKEND_ARM:
        raise RuntimeError("FROZEN_ARMS_MISMATCH")
    return config

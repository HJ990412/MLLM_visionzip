#!/usr/bin/env python3
"""Shared definitions of the ConvBench main experiment (docs/convbench_main_contract.md).

No torch import here: the driver, the gate checks and the report use this module without a GPU.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = "/home/dblab/anaconda3/envs/mllm_ft/bin/python"
SCHEMA = "convbench-main-v1"
RUN_PREFIX = "convbench_main_"
CONTRACT = ROOT / "docs/convbench_main_contract.md"
ALIGN_DOC = ROOT / "docs/convbench_baseline_alignment.md"
ALIGN = ROOT / "results/convbench_alignment_20261121T083134Z"
ALIGN_GEN = {"next": ALIGN / "gen/next_3B_cap_tok", "llava15": ALIGN / "gen/llava15_final"}
ALIGN_JUDGE = {"next": ALIGN / "judge/next_3B_cap_tok", "llava15": ALIGN / "judge/llava15_final"}
ALIGN_CACHE = ALIGN / "judge_cache"
INDEX = ROOT / "data/convbench/index.json"
MODELS = ("next", "llava15")
MODEL_NAME = {"next": "LLaVA-NeXT-7b", "llava15": "LLaVA-1.5-7b"}
MAX_NEW = 1024
CONTEXT_LIMIT = 4096
SEED = 1234
RESAMPLES = 10_000
JUDGE_CAP = 2048
N_CONV = 577

# ---- arms.  The four methods of the experiment and the five extra arms the gate phase needs.
RECOMP, FULL, IMP, OURS = "recomp", "fullload", "impress", "ours"
IMP_SYNC, IMP_R100, FULL_ON_IMP, OURS_SYNC, OURS_REPEAT = ("impress_sync", "impress_r100", "fullload_loader_on_impress_store",
                                                           "ours_sync", "ours_repeat")
MAIN_ROLES = [RECOMP, FULL, IMP, OURS]
GATE_ROLES = [RECOMP, FULL, IMP_SYNC, IMP, IMP_R100, FULL_ON_IMP, OURS_SYNC, OURS, OURS_REPEAT]
DISPLAY = {RECOMP: "ReComp", FULL: "FullLoad", IMP: "IMPRESS", OURS: "Ours",
           IMP_SYNC: "IMPRESS, 동기 읽기", IMP_R100: "IMPRESS, 유지 비율 100%",
           FULL_ON_IMP: "FullLoad loader (IMPRESS의 재배치 store)", OURS_SYNC: "Ours, 동기 읽기", OURS_REPEAT: "Ours, 두 번째 실행"}
PHASES = ("gate", "pilot", "overflow", "main")
PHASE_ROLES = {"gate": GATE_ROLES, "pilot": MAIN_ROLES, "overflow": MAIN_ROLES, "main": MAIN_ROLES}
GATE_ORDINALS = (72, 216, 360, 504)                       # 4 conversations, evenly spread over the index
PILOT_ORDINALS = tuple(7 + 14 * i for i in range(40))     # 40 conversations, every 14th
OVERFLOW_CONVERSATIONS = 2
METACOMPRESS = {"next": {"S1": 7.95, "S2": 11.46, "S3": 7.58, "Avg": 9.00},
                "llava15": {"S1": 4.33, "S2": 5.72, "S3": 5.55, "Avg": 5.20}}
REPORT_SCRIPT = re.compile(r"^scripts/\d+_report_.*\.py$")
# a request phase stops when this many conversations in a row, or in total, contain a failed request
MAX_CONSECUTIVE_FAILED_CONVERSATIONS = 5
MAX_FAILED_CONVERSATIONS = 40


def utc(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if ts is None else ts))


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def atomic_json(path, obj, indent=1) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, allow_nan=False, default=str)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_gz_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False, default=str).encode("utf-8")
    with open(tmp, "wb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0, compresslevel=6) as z:
        z.write(data)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_gz_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def boot_id():
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


def safe_name(conversation_id: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(conversation_id))
    if not name or name in (".", ".."):
        raise ValueError(f"unsafe conversation id {conversation_id!r}")
    return name


def load_index() -> tuple[list[dict], str]:
    rows = json.loads(INDEX.read_text(encoding="utf-8"))["conversations"]
    assert len(rows) == N_CONV and len({r["conversation_id"] for r in rows}) == N_CONV
    for ordinal, row in enumerate(rows):
        row["ordinal"] = ordinal
        assert [t["turn_id"] for t in row["turns"]] == [1, 2, 3]
    return rows, sha(INDEX)


def rotate(roles: list[str], ordinal: int) -> list[str]:
    """Arm order of a conversation: the phase's arm list rotated by the conversation's index ordinal."""
    i = int(ordinal) % len(roles)
    return list(roles[i:]) + list(roles[:i])


def res_rel(res) -> str:
    """results/<name> relative to the repository (the form the status helpers expect)."""
    return f"results/{Path(res).name}"


def model_dir(res, model) -> Path:
    return Path(res) / model


def phase_dir(res, model, phase) -> Path:
    return Path(res) / model / phase


def conv_path(res, model, phase, conversation_id) -> Path:
    return phase_dir(res, model, phase) / "conversations" / f"{safe_name(conversation_id)}.json.gz"


def phase_ordinals(res, model, phase) -> list[int] | None:
    """Index ordinals of the conversations of a phase (None: the overflow selection does not exist yet)."""
    if phase == "gate":
        return list(GATE_ORDINALS)
    if phase == "pilot":
        return list(PILOT_ORDINALS)
    if phase == "main":
        return list(range(N_CONV))
    doc = read_json(model_dir(res, model) / "overflow_selection.json")
    return None if doc is None else list(doc["ordinals"])


def dev_limit(res) -> int:
    """Development rehearsals only (directory name contains '_dev_'; never a measurement): conversations per phase."""
    return int(os.environ.get("CONVBENCH_MAIN_DEV_LIMIT", "0")) if "_dev_" in Path(res).name else 0


def phase_conversations(res, model, phase, index) -> list[dict] | None:
    ordinals = phase_ordinals(res, model, phase)
    if ordinals is None:
        return None
    limit = dev_limit(res)
    if limit:
        ordinals = ordinals[:limit + (1 if phase == "main" else 0)]
    return [index[o] for o in ordinals]


def committed(res, model, phase, index) -> dict[str, dict]:
    """{conversation id: committed document} of a phase (documents that exist)."""
    out = {}
    convs = phase_conversations(res, model, phase, index) or []
    for conv in convs:
        path = conv_path(res, model, phase, conv["conversation_id"])
        if path.exists():
            out[conv["conversation_id"]] = read_gz_json(path)
    return out


def phase_complete(res, model, phase) -> bool:
    return (phase_dir(res, model, phase) / "DONE.json").exists()


def rows_by_key(doc: dict) -> dict[tuple[str, int], dict]:
    return {(r["role"], r["turn_id"]): r for r in doc["rows"]}


def alignment_rows(model: str, conversation_id: str) -> list[dict]:
    """The three final ReComp rows of the alignment run for one conversation (turn order)."""
    body = json.loads((ALIGN_GEN[model] / "conversations" / f"{conversation_id}.json").read_text(encoding="utf-8"))
    rows = sorted((r for r in body["rows"] if r["method"] == "ReComp"), key=lambda r: r["turn_id"])
    assert [r["turn_id"] for r in rows] == [1, 2, 3]
    return rows


def source_hashes() -> dict[str, str]:
    paths = (list((ROOT / "mmimpress").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")) + list((ROOT / "tests").glob("test_*.py")))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths) if not REPORT_SCRIPT.match(str(p.relative_to(ROOT)))}


def freeze_violations(frozen: dict[str, str]) -> list[str]:
    """Frozen source files that changed or disappeared (files created later and report scripts are not compared)."""
    now = source_hashes()
    return sorted(k for k, v in frozen.items() if now.get(k) != v)


def verify_freeze(res) -> dict:
    res = Path(res)
    config = json.loads((res / "config.json").read_text(encoding="utf-8"))
    changed = freeze_violations(config["source_sha256"])
    if changed and not ("_dev_" in res.name and os.environ.get("CONVBENCH_MAIN_DEV_ALLOW_SOURCE_DRIFT") == "1"):
        raise RuntimeError("FROZEN_SOURCE_CONFIG_MISMATCH: " + repr(changed[:10]))
    if sha(CONTRACT) != config["contract_sha256"] and "_dev_" not in res.name:
        raise RuntimeError("FROZEN_CONTRACT_MISMATCH")
    if sha(INDEX) != config["index_sha256"] or sha(ALIGN_DOC) != config["alignment_doc_sha256"]:
        raise RuntimeError("FROZEN_INDEX_OR_ALIGNMENT_DOC_MISMATCH")
    config["_file_sha256"] = sha(res / "config.json")
    return config

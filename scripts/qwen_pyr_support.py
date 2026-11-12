"""Shared constants/helpers for the Qwen layer-wise budget (PyramidKV) packed experiment
(scripts 167-169, 171).  The independent auditor (170) deliberately does NOT import this."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPERIMENT = "qwen_layer_budget_pyr"
SCHEMA = "qwen-mtgqa-layer-budget-pyr-v1"
CONTRACT = ROOT / "docs/qwen_layer_budget_pyr_contract.md"
PARENT_RUN = ROOT / "runs/qwen_mt_gqa_prefixstore3_20261002T074856Z"
PARENT_RES = ROOT / "results/qwen_mt_gqa_prefixstore3_20261002T074856Z"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
NUM_LAYERS = 28
TOTAL_RATIO = 0.25
R_MIN = 0.0
R_MAX = 0.60
BUDGET_KW = dict(num_layers=NUM_LAYERS, total_ratio=TOTAL_RATIO, r_min=R_MIN, r_max=R_MAX)
INDEX_SHA = "2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224"
WORKLOAD_SHA = "0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62"
B = "qwen_ours_prefixstore_kv25"
ALLOCS = ("U", "PYR-b20", "PYR-b2p5")
BETA = {"PYR-b20": 20.0, "PYR-b2p5": 2.5}
PACKED = {f"{B}_{a}_packed": a for a in ALLOCS}
METHODS = ("recompute", "fullload", B) + tuple(PACKED)
U_PACKED, PRIMARY, SENSITIVITY = (f"{B}_U_packed", f"{B}_PYR-b20_packed", f"{B}_PYR-b2p5_packed")
LABEL = {"recompute": "ReComp", "fullload": "FullLoad", B: "Ours B (unpacked)",
         U_PACKED: "U_packed", PRIMARY: "PYR-b20_packed", SENSITIVITY: "PYR-b2p5_packed"}
RATIO_SHA = {"U": "1343c3ea849a9e70c965cea28530125c1072d841fc23cfd019f4282e11c7fc76",
             "PYR-b20": "58dcb86e2f8b1de4ead4fe1672f36516c6472b835f545f45da5e7b4008b7a52b",
             "PYR-b2p5": "5e5c3ef263fbf691e2678ea1deefd81fc6f1516147f679611e01fb0f75d4f6e0"}
NEW_PATHS = ["mmimpress/qwen25/packed_store.py", "mmimpress/qwen25/layerwise.py",
             "scripts/qwen_pyr_support.py", "scripts/167_freeze_qwen_layer_budget_pyr.py",
             "scripts/168_validate_qwen_layerwise_packed.py", "scripts/169_eval_qwen_layer_budget_pyr.py",
             "scripts/170_audit_qwen_layer_budget_pyr.py", "scripts/171_report_qwen_layer_budget_pyr.py",
             "tests/test_qwen_layerwise_packed.py", "docs/qwen_layer_budget_pyr_contract.md"]
REPORT_SCRIPT = "scripts/171_report_qwen_layer_budget_pyr.py"
PY = "/home/dblab/anaconda3/envs/mllm_ft/bin/python"
ENV = "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:scripts:tests OMP_NUM_THREADS=2 MKL_NUM_THREADS=2"
GiB = 1 << 30


def load_script(name: str, file: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj, exclusive: bool = True):
    """fsynced JSON; exclusive by default so recorded evidence is never overwritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(path)
    tmp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("x") as f:
        json.dump(obj, f, ensure_ascii=False, allow_nan=False, indent=1)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def pyramid_ratios(beta: float, layers: int = NUM_LAYERS) -> list[float]:
    """PyramidKV public-code schedule as ratios (same formula as scripts/158 pyramid_beta, no clip)."""
    lo = 0.25 / beta
    hi = 0.5 - lo
    step = (hi - lo) / (layers - 1)
    return [hi - i * step for i in range(layers)]


def vector(alloc: str) -> list[float]:
    return [0.25] * NUM_LAYERS if alloc == "U" else pyramid_ratios(BETA[alloc])


def vector_path(run: Path, alloc: str) -> Path:
    return Path(run) / "vectors" / f"layer_budget_qwen_{alloc}.json"


def load_vectors(run: Path) -> dict:
    """Frozen budget documents, re-validated with every argument explicit."""
    from mmimpress.layer_budget import budget_sha256, validate_ratios
    docs = {}
    for alloc in ALLOCS:
        doc = read(vector_path(run, alloc))
        if doc.get("schema") != "layer_budget_v1" or doc.get("policy") != "layerwise_v1":
            raise ValueError("not a layer_budget_v1 document")
        if (doc["num_layers"], doc["total_ratio"], doc["r_min"], doc["r_max"]) != (NUM_LAYERS, TOTAL_RATIO, R_MIN, R_MAX):
            raise ValueError(f"vector {alloc} was not frozen with the Qwen arguments")
        r = validate_ratios(doc["layer_keep_ratios"], **BUDGET_KW)
        if doc["layer_budget_sha256"] != budget_sha256(r) or doc["layer_budget_sha256"] != RATIO_SHA[alloc] or doc["alloc_id"] != alloc:
            raise ValueError(f"vector {alloc} SHA256 mismatch")
        docs[alloc] = doc
    return docs


def keep_counts(doc: dict, n: int) -> list[int]:
    from mmimpress.layer_budget import layer_keep_counts
    return layer_keep_counts(doc["layer_keep_ratios"], int(n), num_layers=int(doc["num_layers"]),
                             total_ratio=float(doc["total_ratio"]), r_min=float(doc["r_min"]), r_max=float(doc["r_max"]))


def source_hashes() -> dict:
    """Frozen runtime source set: every mmimpress/scripts/tests python file except the report script."""
    paths = list((ROOT / "mmimpress").rglob("*.py")) + [p for p in (ROOT / "scripts").glob("*.py")] + list((ROOT / "tests").glob("test_*.py"))
    out = {}
    for p in sorted(paths):
        rel = str(p.relative_to(ROOT))
        if "__pycache__" in rel or rel == REPORT_SCRIPT:
            continue
        out[rel] = sha(p)
    return out


def resume_command(run: Path) -> str:
    return f"{ENV} {PY} scripts/169_eval_qwen_layer_budget_pyr.py --run {run} --phase all --resume"


def utc() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

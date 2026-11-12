"""Shared constants/helpers for the Qwen2.5-VL retention-ratio sweep (scripts 172-174, 176).
The independent auditor (175) deliberately does NOT import this."""
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

EXPERIMENT = "qwen_budget_sweep"
SCHEMA = "qwen-mtgqa-budget-sweep-v1"
CONTRACT = ROOT / "docs/qwen_budget_sweep_contract.md"
PYR_RUN = ROOT / "runs/qwen_layer_budget_pyr_20261111T075124Z"          # read-only input (PyramidKV run)
PYR_RES = ROOT / "results/qwen_layer_budget_pyr_20261111T075124Z"
PARENT_RUN = ROOT / "runs/qwen_mt_gqa_prefixstore3_20261002T074856Z"     # read-only input (Qwen 3-arm main)
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
NUM_LAYERS = 28
R_MIN = 0.0
R_MAX = 1.0
INDEX_SHA = "2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224"
WORKLOAD_SHA = "0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62"
RATIO = {"U25": 0.25, "U30": 0.30, "U40": 0.40, "U50": 0.50, "U100": 1.00}
FRACTION = {"U25": (1, 4), "U30": (3, 10), "U40": (2, 5), "U50": (1, 2), "U100": (1, 1)}   # independent integer form of the ratios
ALLOCS = tuple(RATIO)
PACKED = {f"qwen_ours_kv_{a}_packed": a for a in ALLOCS}
METHODS = ("recompute", "fullload") + tuple(PACKED)
U25, U100 = "qwen_ours_kv_U25_packed", "qwen_ours_kv_U100_packed"
LABEL = {"recompute": "ReComp", "fullload": "FullLoad", **{m: f"{a}_packed" for m, a in PACKED.items()}}
RATIO_SHA = {"U25": "1343c3ea849a9e70c965cea28530125c1072d841fc23cfd019f4282e11c7fc76",
             "U30": "b726c756cd0680a018200d934536542ea2a31effbe086535577e6f895984a0e7",
             "U40": "d14e8bb77c002045dad35ce55ba2b4b31301af1fbde74383244139a6b949e464",
             "U50": "40302d54a7eba3a7ef3fcc11e150ddef996c4e96a55e8df52af97cb1336d750b"}     # U100: computed and recorded at Phase 0
EXPECTED_K = {260: {"U25": 65, "U30": 78, "U40": 104, "U50": 130, "U100": 260},
              345: {"U25": 87, "U30": 104, "U40": 138, "U50": 173, "U100": 345},
              529: {"U25": 133, "U30": 159, "U40": 212, "U50": 265, "U100": 529}}
EXPECT = {"main": 85281, "smoke": 84, "integration": 84, "per_method": 12183, "images": 398, "dialogues": 4061}
LLAVA_U25_LOSS_PP = 2.30            # LLaVA uniform 25% loss vs ReComp, v2 HOLD (prompt, fixed before this run)
LLAVA_U25_MEDIAN_ROWS = 552
NEW_PATHS = ["mmimpress/qwen25/ratio_packed.py", "scripts/qwen_sweep_support.py",
             "scripts/172_freeze_qwen_budget_sweep.py", "scripts/173_validate_qwen_budget_sweep.py",
             "scripts/174_eval_qwen_budget_sweep.py", "scripts/175_audit_qwen_budget_sweep.py",
             "scripts/176_report_qwen_budget_sweep.py", "tests/test_qwen_budget_sweep.py",
             "docs/qwen_budget_sweep_contract.md"]
REPORT_SCRIPT = "scripts/176_report_qwen_budget_sweep.py"
EVAL_SCRIPT = "174_eval_qwen_budget_sweep.py"
AUDIT_SCRIPT = "175_audit_qwen_budget_sweep.py"
PY = "/home/dblab/anaconda3/envs/mllm_ft/bin/python"
ENV = "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:scripts:tests OMP_NUM_THREADS=2 MKL_NUM_THREADS=2"
GiB = 1 << 30


def budget_kw(alloc: str) -> dict:
    """Explicit layer_budget arguments for one ratio (never the LLaVA module defaults)."""
    return dict(num_layers=NUM_LAYERS, total_ratio=RATIO[alloc], r_min=R_MIN, r_max=R_MAX)


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


def vector(alloc: str) -> list[float]:
    return [RATIO[alloc]] * NUM_LAYERS


def vector_path(run: Path, alloc: str) -> Path:
    return Path(run) / "vectors" / f"layer_budget_qwen_{alloc}.json"


def integer_k(alloc: str, n: int) -> int:
    """ceil(p*N/q) with the ratio written as an integer fraction; no floating point."""
    p, q = FRACTION[alloc]
    return -((-p * int(n)) // q)


def load_vectors(run: Path) -> dict:
    """Frozen budget documents, re-validated with every argument explicit."""
    from mmimpress.layer_budget import budget_sha256, validate_ratios
    recorded = read(Path(run) / "vectors" / "vectors_sha256.json")
    docs = {}
    for alloc in ALLOCS:
        doc = read(vector_path(run, alloc))
        if doc.get("schema") != "layer_budget_v1" or doc.get("policy") != "layerwise_v1":
            raise ValueError("not a layer_budget_v1 document")
        if (doc["num_layers"], doc["total_ratio"], doc["r_min"], doc["r_max"]) != (NUM_LAYERS, RATIO[alloc], R_MIN, R_MAX):
            raise ValueError(f"vector {alloc} was not frozen with the Qwen sweep arguments")
        r = validate_ratios(doc["layer_keep_ratios"], **budget_kw(alloc))
        expected = RATIO_SHA.get(alloc, recorded[alloc]["layer_budget_sha256"])
        if r != vector(alloc) or doc["layer_budget_sha256"] != budget_sha256(r) or doc["layer_budget_sha256"] != expected or doc["alloc_id"] != alloc:
            raise ValueError(f"vector {alloc} SHA256 mismatch")
        docs[alloc] = doc
    return docs


def keep_counts(doc: dict, n: int) -> list[int]:
    from mmimpress.layer_budget import layer_keep_counts
    return layer_keep_counts(doc["layer_keep_ratios"], int(n), num_layers=int(doc["num_layers"]),
                             total_ratio=float(doc["total_ratio"]), r_min=float(doc["r_min"]), r_max=float(doc["r_max"]))


def source_hashes() -> dict:
    """Frozen runtime source set: every mmimpress/scripts/tests python file except this run's post-hoc report script."""
    paths = list((ROOT / "mmimpress").rglob("*.py")) + [p for p in (ROOT / "scripts").glob("*.py")] + list((ROOT / "tests").glob("test_*.py"))
    out = {}
    for p in sorted(paths):
        rel = str(p.relative_to(ROOT))
        if "__pycache__" in rel or rel == REPORT_SCRIPT:
            continue
        out[rel] = sha(p)
    return out


def resume_command(run: Path) -> str:
    return f"{ENV} {PY} scripts/{EVAL_SCRIPT} --run {run} --phase all --resume"


def utc() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

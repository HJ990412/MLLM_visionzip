"""Read-only provenance capture. Writes exclusively to a new audit directory."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = Path(__file__).resolve().parents[1] / "evidence"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def command(args):
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    return dict(args=args, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    files = [*sorted(ROOT.glob("mmimpress/**/*.py")),
             *sorted(ROOT.glob("scripts/*")), *sorted(ROOT.glob("docs/*.md")),
             *sorted(ROOT.glob("tests/*.py")), ROOT / "papers/mpic.md", ROOT / "README.md"]
    files = [q for q in files if q.is_file()]
    manifest = {q.relative_to(ROOT).as_posix(): sha(q) for q in files}
    write(a.out / "source_sha256_before.json", manifest)
    relevant = ["mmimpress/mpic.py", "mmimpress/model.py", "mmimpress/serve.py",
                "mmimpress/store.py", "mmimpress/piggyback.py", "mmimpress/config.py",
                "scripts/108_eval_llava_mt_gqa_B.py", "scripts/109_audit_llava_mt_gqa_B.py",
                "scripts/65_validate_mpic.py", "scripts/67_eval_mpic_gqa.py",
                "scripts/73_eval_mt_gqa_5arm_generated_shard.py", "tests/test_mpic.py"]
    for name in relevant:
        dest = a.out / "source_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as f:
            f.write((ROOT / name).read_bytes())
    run = ROOT / "runs/mt_gqa_B_llava5_qwen3_20261001T093921Z/llava"
    config = json.loads((run / "config.json").read_text())
    shutil.copyfile(run / "config.json", a.out / "main_config.json")
    comparison = {name: {"main_recorded": config["source_sha256"].get(name),
                         "current": manifest[name],
                         "matches": config["source_sha256"].get(name) == manifest[name]}
                  for name in relevant if name in config["source_sha256"]}
    write(a.out / "main_source_comparison.json", comparison)
    # One deterministically selected image/dialogue, three methods, three turns.
    raw = sorted(run.glob("main/images/*/attempt_0001/raw.jsonl"))[0]
    selected = []
    first_dialog = None
    with raw.open() as f:
        for line in f:
            r = json.loads(line)
            if first_dialog is None:
                first_dialog = r["dialog_id"]
            if r["dialog_id"] != first_dialog:
                break
            if r["method_id"] in ("recompute", "fullload", "mpic32_ssd"):
                selected.append(r)
    with (a.out / "historical_raw_9_requests.jsonl").open("x") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    assert len(selected) == 9
    write(a.out / "historical_provenance.json", {
        "role": "historical evidence only, not a fresh GPU validation",
        "raw_path": str(raw.relative_to(ROOT)), "raw_sha256": sha(raw),
        "dialog_id": first_dialog, "rows": 9,
        "config_sha256": sha(run / "config.json"),
        "current_mpic_sha256": manifest["mmimpress/mpic.py"],
        "previous_reports_used_as_oracle": False})
    packages = {}
    for name in ("torch", "transformers", "numpy", "bitsandbytes", "Pillow", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    import torch
    write(a.out / "environment.json", {
        "client_date": "2026-10-13", "client_timezone": "Asia/Seoul",
        "python": platform.python_version(), "platform": platform.platform(),
        "executable": os.sys.executable, "packages": packages,
        "cuda_available": torch.cuda.is_available(), "cuda_device_count": torch.cuda.device_count(),
        "nvidia_smi": command(["nvidia-smi"]), "memory": command(["free", "-h"]),
        "disk": command(["df", "-h", str(ROOT)]),
        "git_head": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--short"]),
        "cpu_count": os.cpu_count(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "paper_pdf_attached_found": False,
        "paper_local_markdown_sha256": manifest["papers/mpic.md"]})
    print(json.dumps({"source_files": len(manifest), "historical_rows": len(selected),
                      "all_main_sources_match": all(x["matches"] for x in comparison.values()),
                      "cuda_available": torch.cuda.is_available()}))


if __name__ == "__main__":
    main()

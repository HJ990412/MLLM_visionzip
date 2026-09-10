"""Stage-safe launcher for the VisDial multi-turn SSD experiment.

The launcher deliberately refuses MMDU.  MMDU is admitted only through the
separate correctness gate because independently built image-prefix stores are
not composable in an interleaved causal conversation.

Examples:
  python scripts/15_run_multiturn_eval.py --dataset visdial --profile smoke
  python scripts/15_run_multiturn_eval.py --dataset visdial --profile main
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmimpress.config import PROJECT_ROOT
from mmimpress.multiturn import sha256_file


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=1)
    tmp.replace(path)


def _run(argv):
    print("+", " ".join(str(x) for x in argv), flush=True)
    subprocess.run([str(x) for x in argv], cwd=PROJECT_ROOT, check=True)


def _calibration_records(store_index):
    with open(store_index) as f:
        rows = json.load(f)
    records = []
    for row in rows:
        assert len(row["questions"]) == 1
        q = row["questions"][0]
        assert q.get("calibration_only") is True
        records.append({
            "image_id": str(row["image_id"]),
            "context_id": str(q["question_id"]),
            "kind": "caption_only_pre_dialog",
            "sha256": hashlib.sha256(q["question"].encode()).hexdigest(),
            "observed_turn_boundary": 0,
        })
    return records


def _verify_store(store_root, image_ids, require_reorder=False,
                  require_static=False):
    failures = []
    total = 0
    for image_id in image_ids:
        d = store_root / image_id
        meta_path = d / "meta.json"
        if not meta_path.exists():
            failures.append(f"missing {meta_path}")
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        total += int(meta["bytes_visual_kv"])
        dtype_bytes = 2 if meta["dtype"] == "float16" else 4
        layer_bytes = (int(meta["v_token_num"]) * int(meta["num_heads"]) *
                       int(meta["head_dim"]) * dtype_bytes)
        for li in range(int(meta["num_layers"])):
            for kind in ("k", "v"):
                p = d / f"layer_{li:02d}" / f"{kind}.bin"
                if not p.exists() or p.stat().st_size != layer_bytes:
                    failures.append(f"invalid layer payload: {p}")
        if require_reorder and not meta.get("reordered"):
            failures.append(f"not reordered: {image_id}")
        if require_static:
            for name in ("static.pt", "sep_kv.bin"):
                if not (d / name).exists():
                    failures.append(f"missing {d / name}")
    if failures:
        raise RuntimeError("store verification failed:\n" + "\n".join(failures[:20]))
    return total


def _artifact_records(store_root, image_ids, include_static=False):
    records = []
    for image_id in image_ids:
        d = store_root / image_id
        with open(d / "meta.json") as f:
            meta = json.load(f)
        row = {
            "image_id": image_id,
            "meta_sha256": sha256_file(d / "meta.json"),
            "visual_tokens": int(meta["v_token_num"]),
            "prefix_tokens": int(meta["prefix_len"]),
            "full_visual_kv_bytes": int(meta["bytes_visual_kv"]),
            "reordered": bool(meta.get("reordered", False)),
        }
        if include_static:
            row.update({
                "static_pt_sha256": sha256_file(d / "static.pt"),
                "static_pt_bytes": (d / "static.pt").stat().st_size,
                "sep_kv_bytes": (d / "sep_kv.bin").stat().st_size,
            })
        records.append(row)
    return records


def _enrich_static_summary(path, index_path):
    """Report amortisation over evaluated turns, not calibration records."""
    path = Path(path)
    if not path.exists():
        return
    with open(path) as f:
        summary = json.load(f)
    with open(index_path) as f:
        dialogs = json.load(f)["dialogs"]
    n_images = sum(len(d["images"]) for d in dialogs)
    n_turns = sum(len(d["turns"]) for d in dialogs)
    reuse = n_turns / max(n_images, 1)
    legacy_amortised = summary.pop("amortised_ms_per_question", None)
    summary["calibration_contexts_per_image"] = summary.pop(
        "mean_questions_per_image", 1.0)
    if legacy_amortised is not None:
        summary["legacy_amortised_ms_per_calibration_context"] = \
            legacy_amortised
    summary["evaluated_turns_per_image"] = reuse
    summary["amortised_ms_per_evaluated_turn"] = (
        float(summary["total_first_use_ms"]) / max(reuse, 1.0))
    summary["note"] = ("offline image-static metadata is built once and reused; "
                       "calibration records are not evaluation questions")
    _write_json(path, summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("visdial", "mmdu"), default="visdial")
    ap.add_argument("--profile", choices=("smoke", "main"), default="smoke")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--stage", choices=("all", "build", "reorder", "static",
                                        "eval", "analyze"), default="all")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--store", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--max-dialogs", type=int, default=None)
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.dataset != "visdial":
        raise SystemExit(
            "MMDU SSD evaluation is correctness-gated. Run "
            "scripts/17_validate_mmdu_cache.py; independent image-prefix KV "
            "stores must never be concatenated.")

    tag = f"{args.profile}_seed{args.seed}"
    index_dir = Path(args.index_dir or
                     PROJECT_ROOT / "data/visdial_v1.0/subsets" / tag)
    index_path = index_dir / "index.json"
    store_index = index_dir / "store_index.json"
    store_root = Path(args.store or
                      PROJECT_ROOT / "kvstore_multiturn" / f"visdial_{tag}")
    run_dir = Path(args.run_dir or
                   PROJECT_ROOT / "runs/visdial_multiturn" /
                   f"{tag}_true_ttft")
    results_dir = Path(args.results_dir or
                       PROJECT_ROOT / "results/visdial_multiturn" /
                       f"{tag}_true_ttft")

    if not index_path.exists() or not store_index.exists():
        _run([sys.executable, "scripts/13_build_multiturn_index.py",
              "--dataset", "visdial", "--profile", args.profile,
              "--seed", args.seed, "--out-dir", index_dir])
    with open(store_index) as f:
        store_rows = json.load(f)
    image_ids = [str(x["image_id"]) for x in store_rows]
    assert len(image_ids) == len(set(image_ids))

    manifest_path = store_root / "pipeline_manifest.json"
    current = {
        "schema_version": "visdial-store-pipeline-v1",
        "dataset": "visdial_v1.0_val",
        "profile": args.profile,
        "seed": args.seed,
        "index": str(index_path.resolve()),
        "index_sha256": sha256_file(index_path),
        "store_index": str(store_index.resolve()),
        "store_index_sha256": sha256_file(store_index),
        "image_ids": image_ids,
        "n_images": len(image_ids),
        "calibration_policy": "caption_only_pre_dialog",
        "future_turn_calibration_count": 0,
        "calibration_records": _calibration_records(store_index),
        "reorder": {"algorithm": "existing_importance", "calib_questions": 1},
        "static_selector": {
            "algorithm": "existing_static_diverse_chunk",
            "chunk_size": 64,
            "aggregate": "topk_mean",
            "alpha": 0.7,
            "top_frac": 0.25,
            "diverse_frac": 0.25,
        },
        "stages": {},
    }
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        for key in ("index_sha256", "store_index_sha256", "image_ids",
                    "calibration_policy"):
            if manifest.get(key) != current[key]:
                raise RuntimeError(f"existing store manifest mismatch for {key}")
        current["stages"] = manifest.get("stages", {})
        current["created_at_unix"] = manifest.get("created_at_unix", time.time())
    else:
        # A non-empty, unmanifested root might already have been reordered.
        # Refuse to guess because a second physical reorder is not a safe resume.
        if store_root.exists() and any(store_root.iterdir()):
            raise RuntimeError(
                f"non-empty store has no provenance manifest: {store_root}")
        store_root.mkdir(parents=True, exist_ok=True)
        current["created_at_unix"] = time.time()
        _write_json(manifest_path, current)

    stages = current["stages"]
    requested = args.stage

    if requested in ("all", "build") and not stages.get("build", {}).get("done"):
        _run([sys.executable, "scripts/01_build_store.py", "--index", store_index,
              "--out", store_root])
        total = _verify_store(store_root, image_ids)
        stages["build"] = {"done": True, "completed_at_unix": time.time(),
                           "image_kv_build_count": len(image_ids),
                           "full_visual_kv_bytes": total,
                           "artifacts_after_build": _artifact_records(
                               store_root, image_ids)}
        _write_json(manifest_path, current)

    if requested in ("all", "reorder") and not stages.get("reorder", {}).get("done"):
        _verify_store(store_root, image_ids)
        _run([sys.executable, "scripts/02_reorder.py", "--index", store_index,
              "--store", store_root, "--order", "importance",
              "--calib-questions", "1"])
        _verify_store(store_root, image_ids, require_reorder=True)
        stages["reorder"] = {"done": True, "completed_at_unix": time.time(),
                             "calibration_count_per_image": 1,
                             "future_turn_calibration_count": 0,
                             "artifacts_after_reorder": _artifact_records(
                                 store_root, image_ids)}
        _write_json(manifest_path, current)

    if requested in ("all", "static") and not stages.get("static", {}).get("done"):
        _verify_store(store_root, image_ids, require_reorder=True)
        static_summary = results_dir / "static_build.json"
        _run([sys.executable, "scripts/06_build_static.py", "--index", store_index,
              "--store", store_root, "--summary-out", static_summary,
              "--agg", "topk_mean", "--alpha", "0.7", "--top-frac", "0.25"])
        _verify_store(store_root, image_ids, require_reorder=True,
                      require_static=True)
        stages["static"] = {"done": True, "completed_at_unix": time.time(),
                            "static_metadata_build_count": len(image_ids),
                            "summary": str(static_summary.resolve()),
                            "artifacts": _artifact_records(
                                store_root, image_ids, include_static=True)}
        _write_json(manifest_path, current)

    # Older interrupted manifests may predate per-image provenance.  Backfill
    # it from the verified final store without rerunning or rewriting K/V.
    if (stages.get("static", {}).get("done") and
            not stages["static"].get("artifacts")):
        _verify_store(store_root, image_ids, require_reorder=True,
                      require_static=True)
        stages["static"]["artifacts"] = _artifact_records(
            store_root, image_ids, include_static=True)
        _write_json(manifest_path, current)

    static_summary = results_dir / "static_build.json"
    recorded_summary = Path(stages.get("static", {}).get("summary", static_summary))
    if (recorded_summary.exists() and recorded_summary.resolve() !=
            static_summary.resolve() and not static_summary.exists()):
        static_summary.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(recorded_summary, static_summary)
    _enrich_static_summary(static_summary, index_path)

    if requested in ("all", "eval"):
        _verify_store(store_root, image_ids, require_reorder=True,
                      require_static=True)
        cmd = [sys.executable, "scripts/14_eval_multiturn.py",
               "--index", index_path, "--store", store_root,
               "--run-dir", run_dir, "--seed", args.seed,
               "--max-new-tokens", args.max_new_tokens]
        if args.max_dialogs is not None:
            cmd += ["--max-dialogs", args.max_dialogs]
        if args.max_turns is not None:
            cmd += ["--max-turns", args.max_turns]
        if args.warm:
            cmd.append("--warm")
        if args.resume or (run_dir / "raw.jsonl").exists():
            cmd.append("--resume")
        _run(cmd)
        stages["eval"] = {"done": True, "completed_at_unix": time.time(),
                          "run_dir": str(run_dir.resolve())}
        _write_json(manifest_path, current)

    if requested in ("all", "analyze"):
        if not (run_dir / "raw.jsonl").exists():
            raise FileNotFoundError(run_dir / "raw.jsonl")
        _run([sys.executable, "scripts/16_analyze_multiturn.py",
              "--run-dir", run_dir, "--results-dir", results_dir])
        stages["analyze"] = {"done": True, "completed_at_unix": time.time(),
                             "results_dir": str(results_dir.resolve())}
        _write_json(manifest_path, current)

    print(json.dumps({"store": str(store_root), "run": str(run_dir),
                      "results": str(results_dir), "stages": stages}, indent=1))


if __name__ == "__main__":
    main()

"""Sharded large-scale / cross-dataset evaluation.

An image's visual KV is ~1.15 GB, so a 400-image dataset would need ~460 GB of
SSD at once.  This runs the pipeline in IMAGE shards: build the shard's stores,
reorder, build the static sidecars, evaluate every question of those images
against every method, append the raw rows, then delete the shard and move on.

The reuse property the system depends on is preserved exactly -- an image's KV
and its static metadata are built ONCE and used by all of that image's
questions.  Only the residency window is bounded.

Every stage calls the same scripts the small-scale results were produced with
(01 / 02 / 06 / 04) with a --only shard list, so no code path is duplicated for
the large run.

  python scripts/11_run_large_eval.py --dataset gqa_large --order importance \
      --calib-questions 4 --skip 4 --budgets 0.25,0.375,0.5
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mmimpress.config import DATA_DIR, PROJECT_ROOT, RESULTS_DIR

PY = "/home/dblab/anaconda3/envs/mllm_ft/bin/python"
ENV_NOTE = dict(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")


def run(cmd, log):
    log.write(f"\n$ {' '.join(cmd)}\n")
    log.flush()
    import os
    env = {**os.environ, **ENV_NOTE}
    p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                       cwd=str(PROJECT_ROOT))
    if p.returncode != 0:
        raise SystemExit(f"FAILED ({p.returncode}): {' '.join(cmd)}")


def free_gb():
    import shutil as sh
    return sh.disk_usage(str(PROJECT_ROOT)).free / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--metric", default="gqa")
    ap.add_argument("--order", default="importance",
                    choices=("importance", "morton", "none"))
    ap.add_argument("--calib-questions", type=int, default=4)
    ap.add_argument("--skip", type=int, default=4,
                    help="questions per image reserved for calibration and "
                         "excluded from evaluation")
    ap.add_argument("--questions", type=int, default=99,
                    help="evaluated questions per image, after --skip")
    ap.add_argument("--shard-size", type=int, default=80)
    ap.add_argument("--budgets", default="0.25,0.5")
    ap.add_argument("--ratio", type=float, default=0.25,
                    help="SparseVLM token retention (its own budget notion)")
    ap.add_argument("--selectors", default="sparsevlm,static_diverse_chunk")
    ap.add_argument("--no-recompute", action="store_true")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    idx_path = DATA_DIR / args.dataset / "index.json"
    with open(idx_path) as f:
        index = json.load(f)
    if args.max_images:
        index = index[:args.max_images]

    out_dir = RESULTS_DIR / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    store = PROJECT_ROOT / f"kvstore_{args.dataset}"
    log_path = out_dir / "run.log"
    shards = [index[i:i + args.shard_size]
              for i in range(0, len(index), args.shard_size)]

    cfg = {"dataset": args.dataset, "metric": args.metric, "order": args.order,
           "calib_questions": args.calib_questions, "skip": args.skip,
           "questions_per_image": args.questions,
           "budgets": args.budgets, "sparsevlm_ratio": args.ratio,
           "selectors": args.selectors, "shard_size": args.shard_size,
           "n_images": len(index), "n_shards": len(shards),
           "index": str(idx_path)}
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=1)
    print(json.dumps(cfg, indent=1), flush=True)

    t_all = time.time()
    with open(log_path, "a") as log:
        for si, shard in enumerate(shards):
            done_marker = out_dir / f"shard_{si:03d}.json"
            if args.resume and done_marker.exists():
                print(f"[shard {si+1}/{len(shards)}] exists, skip", flush=True)
                continue
            ids = ",".join(str(e["image_id"]) for e in shard)
            t0 = time.time()
            shutil.rmtree(store, ignore_errors=True)
            print(f"[shard {si+1}/{len(shards)}] {len(shard)} images, "
                  f"free {free_gb():.0f} GB", flush=True)

            run([PY, "scripts/01_build_store.py", "--out", str(store),
                 "--index", str(idx_path), "--only", ids], log)
            if args.order != "none":
                cmd = [PY, "scripts/02_reorder.py", "--order", args.order,
                       "--store", str(store), "--index", str(idx_path)]
                if args.order == "importance":
                    cmd += ["--calib-questions", str(args.calib_questions)]
                run(cmd, log)
            run([PY, "scripts/06_build_static.py", "--store", str(store),
                 "--index", str(idx_path), "--only", ids], log)

            cmd = [PY, "scripts/04_eval.py", "--store", str(store),
                   "--index", str(idx_path), "--only", ids,
                   "--metric", args.metric, "--dataset", args.dataset,
                   "--questions", str(args.questions), "--skip", str(args.skip),
                   "--ratio", str(args.ratio), "--budgets", args.budgets,
                   "--selectors", args.selectors,
                   "--out", str(done_marker)]
            if args.no_recompute:
                cmd.append("--no-recompute")
            run(cmd, log)
            print(f"  shard done in {time.time()-t0:.0f}s "
                  f"(total {(time.time()-t_all)/60:.0f} min)", flush=True)
    shutil.rmtree(store, ignore_errors=True)

    # ---- merge shards -------------------------------------------------
    rows = []
    for p in sorted(out_dir.glob("shard_*.json")):
        with open(p) as f:
            rows.extend(json.load(f)["rows"])
    with open(out_dir / "raw.jsonl", "w") as f:
        for r in rows:
            for m, v in r.items():
                if isinstance(v, dict) and "answer" in v:
                    f.write(json.dumps({
                        "dataset": args.dataset, "method": m,
                        "image_id": r["image_id"],
                        "question_id": r["question_id"],
                        "question": r["question"], "gold": r["gold"],
                        "prediction": v["answer"], "score": v["acc"],
                        **{k: v[k] for k in v if k not in
                           ("answer", "acc")}}) + "\n")
    print(f"\nmerged {len(rows)} questions -> {out_dir/'raw.jsonl'} "
          f"({(time.time()-t_all)/60:.0f} min total)", flush=True)


if __name__ == "__main__":
    main()

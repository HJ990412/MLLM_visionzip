"""Three-mode evaluation: IMPRESS vs FullLoad vs ReComp on GQA.

  impress    identify-then-load: probe sidecar -> Jaccard vs threshold ->
             consensus chunks, everything unloaded is masked out
  fullload   reuse the prefix but read every chunk, no selection (AS-like)
  recompute  no store at all, prefill from pixels (ReComp)

All methods answer the same questions with the same model, so accuracy is
directly comparable; disk numbers come from real preads with the page cache
dropped per request.  Latency schema v2 separates true TTFT (through the first
output token), subsequent decode latency, and end-to-end generation latency.

  python scripts/04_eval.py --questions 6 --limit 40 --ratio 0.25
"""
import argparse
import csv
import hashlib
import json
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmimpress.config import (ALPHA, DATA_DIR, PROBE_HEADS, PROJECT_ROOT,
                              RESULTS_DIR, RETENTION_RATIO, STORE_DIR)
from mmimpress.dataset import (METRICS, gqa_accuracy, load_index,
                               question_answers)
from mmimpress.model import LlavaRunner
from mmimpress.serve import ImageContext, Server, load_static

# selector -> (request kind, kwargs).  "impress" is the untouched SparseVLM
# baseline; the cvpr25 ones are hook-free and chunk-first.
SELECTORS = {
    "sparsevlm":              ("impress", {}),
    "visionzip_static_chunk": ("cvpr25", dict(mode="static")),
    "cvpr25_hybrid_chunk":    ("cvpr25", dict(mode="hybrid")),
    "cvpr25_hybrid_diverse":  ("cvpr25", dict(mode="diverse")),
    # ---- ablation: which ingredient actually does the work? ----
    "diverse_chunk":          ("cvpr25", dict(mode="diverse_only")),
    "static_diverse_chunk":   ("cvpr25", dict(mode="static_diverse")),
}
# Random needs several seeds: one draw says nothing about whether the proposed
# selection beats chance.
for _s in range(8):
    SELECTORS[f"random_chunk_s{_s}"] = ("cvpr25", dict(mode="random", seed=_s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=6,
                    help="held-out questions per image")
    ap.add_argument("--skip", type=int, default=4,
                    help="skip the first N questions (used for calibration)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ratio", type=float, default=RETENTION_RATIO)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--probe", type=int, default=PROBE_HEADS)
    ap.add_argument("--warm", action="store_true",
                    help="do not drop the page cache between requests")
    ap.add_argument("--selectors", default="sparsevlm,visionzip_static_chunk,"
                    "cvpr25_hybrid_chunk,cvpr25_hybrid_diverse")
    ap.add_argument("--budget", type=float, default=None,
                    help="chunk/byte budget for the cvpr25 selectors "
                         "(default: same value as --ratio)")
    ap.add_argument("--budgets", default=None,
                    help="comma list of chunk budgets; each cvpr25 selector is "
                         "evaluated at every one of them inside a SINGLE pass, "
                         "so the shared baselines (ReComp / FullLoad / "
                         "SparseVLM) are not re-run per budget")
    ap.add_argument("--sep-policy", default="sidecar",
                    choices=("sidecar", "force", "drop"))
    ap.add_argument("--lam-static", type=float, default=1.0)
    ap.add_argument("--lam-query", type=float, default=1.0)
    ap.add_argument("--diverse-frac", type=float, default=0.25,
                    help="share of the chunk budget filled by DivPrune MaxMin "
                         "instead of by score")
    ap.add_argument("--no-recompute", action="store_true")
    ap.add_argument("--store", default=None)
    ap.add_argument("--index", default=None)
    ap.add_argument("--metric", default="gqa", choices=sorted(METRICS))
    ap.add_argument("--dataset", default="gqa")
    ap.add_argument("--only", default=None,
                    help="comma-separated image_ids (shard)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--run-dir", default=None,
                    help="write per_request.csv, summary.csv and README.md "
                         "alongside the JSON/JSONL result")
    ap.add_argument("--expect-images", type=int, default=None,
                    help="fail before model load unless the runnable workload "
                         "has exactly this many unique images")
    ap.add_argument("--expect-questions", type=int, default=None,
                    help="fail before model load unless the selected workload "
                         "has exactly this many questions")
    args = ap.parse_args()

    store_dir = Path(args.store) if args.store else STORE_DIR
    index = load_index(args.index)
    if args.only:
        keep = set(args.only.split(","))
        index = [e for e in index if str(e["image_id"]) in keep]
    if args.limit:
        index = index[:args.limit]

    # Validate the exact runnable workload before the expensive model load.
    # Only stores that the evaluation loop can actually open count here.
    index = [e for e in index
             if (store_dir / str(e["image_id"]) / "meta.json").exists()]
    q_counts = [len(e["questions"][args.skip:args.skip + args.questions])
                for e in index]
    n_images = len({str(e["image_id"]) for e in index})
    n_questions = sum(q_counts)
    index_path = Path(args.index) if args.index else DATA_DIR / "index.json"
    index_sha256 = hashlib.sha256(index_path.read_bytes()).hexdigest()
    workload_blob = "\n".join(
        f"{e['image_id']}\t{q['question_id']}"
        for e in index
        for q in e["questions"][args.skip:args.skip + args.questions]
    ).encode()
    workload_sha256 = hashlib.sha256(workload_blob).hexdigest()
    if args.expect_images is not None:
        assert n_images == args.expect_images, \
            f"expected {args.expect_images} images, got {n_images}"
    if args.expect_questions is not None:
        assert n_questions == args.expect_questions, \
            f"expected {args.expect_questions} questions, got {n_questions}"
    print(f"workload: images={n_images} questions={n_questions}")
    if q_counts:
        print("questions_per_image: "
              f"min={min(q_counts)} max={max(q_counts)} "
              f"mean={np.mean(q_counts):.3f}")
    print(f"index_sha256={index_sha256}")
    print(f"workload_sha256={workload_sha256}")

    runner = LlavaRunner().load()
    srv = Server(runner, ratio=args.ratio, probe=args.probe, alpha=args.alpha)

    sel_names = [x.strip() for x in args.selectors.split(",") if x.strip()]
    for n in sel_names:
        assert n in SELECTORS, f"unknown selector {n}"
    budget = args.budget if args.budget is not None else args.ratio
    budgets = ([float(x) for x in args.budgets.split(",")]
               if args.budgets else [budget])
    # method key = selector name, plus "@<budget%>" when more than one budget
    # is in play, so every arm is a separate row with its own metrics
    plan = []
    for n in sel_names:
        kind, kw = SELECTORS[n]
        if kind == "cvpr25":
            for b in budgets:
                tag = n if len(budgets) == 1 else f"{n}@{int(round(b*100))}"
                plan.append((tag, kind, kw, b))
        else:
            plan.append((n, kind, kw, budget))
    modes = (["fullload"] + (["recompute"] if not args.no_recompute else [])
             + [t for t, _, _, _ in plan])
    retention = {"fullload": 1.0, "recompute": None}
    retention_kind = {"fullload": "full", "recompute": "none"}
    for tag, kind, _, b in plan:
        retention[tag] = args.ratio if kind == "impress" else b
        retention_kind[tag] = "token" if kind == "impress" else "chunk"

    rows = []
    t_start = time.time()
    for ie, e in enumerate(index):
        d = store_dir / str(e["image_id"])
        if not (d / "meta.json").exists():
            continue
        ctx = ImageContext(d, runner.model.device)
        img = Image.open(PROJECT_ROOT / e["image_path"]).convert("RGB")
        qs = e["questions"][args.skip:args.skip + args.questions]
        static = load_static(ctx) if any(
            k == "cvpr25" for _, k, _, _ in plan) else None
        for q in qs:
            rec = {"image_id": e["image_id"], "question_id": q["question_id"],
                   "question": q["question"],
                   "gold": question_answers(q)}
            if not args.no_recompute:
                rc = srv.recompute(runner.encode(img, q["question"]))
                rec["recompute"] = _pack(rc, q, args.metric)
            r = srv.request(ctx, q["question"], mode="fullload",
                            cold=not args.warm)
            rec["fullload"] = _pack(r, q, args.metric)
            for name, kind, kw, b in plan:
                if kind == "impress":
                    r = srv.request(ctx, q["question"], mode="impress",
                                    cold=not args.warm)
                else:
                    r = srv.request_cvpr25(
                        ctx, q["question"], static, budget=b,
                        sep_policy=args.sep_policy,
                        lam_static=args.lam_static,
                        lam_query=args.lam_query,
                        diverse_frac=args.diverse_frac,
                        cold=not args.warm,
                        image_id=e["image_id"], **kw)
                rec[name] = _pack(r, q, args.metric)
            rows.append(rec)
        ctx.close()
        del ctx
        torch.cuda.empty_cache()
        acc = {m: np.mean([r[m]["acc"] for r in rows]) for m in modes}
        line = "  ".join(f"{m[:9]}={acc[m]:.3f}" for m in modes)
        print(f"[{ie+1}/{len(index)}] {e['image_id']} n={len(rows)} {line}"
              f"  ({time.time()-t_start:.0f}s)")

    # ------------------------------------------------------------ summary
    def agg(m, key, default=0.0):
        vals = [r[m].get(key, default) for r in rows if m in r]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    try:
        with open(RESULTS_DIR / "static_build.json") as f:
            sb = json.load(f)
    except FileNotFoundError:
        sb = {}
    # This global file is overwritten by other dataset/shard builds.  Never
    # attach a mismatched snapshot to a self-contained run.
    if sb.get("n_images") != n_images:
        sb = {}

    s = {"schema_version": 2,
         "n": len(rows), "n_images": n_images,
         "questions_per_image": {
             "min": min(q_counts) if q_counts else 0,
             "max": max(q_counts) if q_counts else 0,
             "mean": float(np.mean(q_counts)) if q_counts else 0.0,
         },
         "ratio": args.ratio, "budget": budget,
         "budgets": budgets, "metric": args.metric,
         "skip": args.skip, "questions_per_image_requested": args.questions,
         "alpha": args.alpha, "probe_heads": args.probe,
         "lam_static": args.lam_static, "lam_query": args.lam_query,
         "diverse_frac": args.diverse_frac,
         "sep_policy": args.sep_policy, "cold": not args.warm,
         "max_new_tokens": srv.max_new_tokens,
         "index": str(index_path),
         "index_sha256": index_sha256,
         "workload_sha256": workload_sha256,
         "latency_definition": {
             "ttft": "request start through first output token",
             "decode": "after first output token through final output token",
             "e2e": "request start through final output token",
         },
         "command": shlex.join([sys.executable, *sys.argv]),
         "static_build": sb, "per_method": {}}
    for m in modes:
        ttft = np.asarray([r[m]["ttft"] for r in rows], dtype=float) * 1e3
        dec = np.asarray([r[m]["decode_latency"] for r in rows],
                         dtype=float) * 1e3
        e2e = np.asarray([r[m]["e2e_latency"] for r in rows],
                         dtype=float) * 1e3
        d = {"acc": agg(m, "acc"),
             "ttft_mean_ms": float(ttft.mean()),
             "ttft_p50_ms": float(np.median(ttft)),
             "ttft_p95_ms": float(np.percentile(ttft, 95)),
             "ttft_std_ms": float(ttft.std()),
             "decode_mean_ms": float(dec.mean()),
             "decode_p50_ms": float(np.median(dec)),
             "decode_p95_ms": float(np.percentile(dec, 95)),
             "decode_std_ms": float(dec.std()),
             "e2e_mean_ms": float(e2e.mean()),
             "e2e_p50_ms": float(np.median(e2e)),
             "e2e_p95_ms": float(np.percentile(e2e, 95)),
             "e2e_std_ms": float(e2e.std()),
             "generated_tokens_mean": agg(m, "generated_tokens"),
             "ssd_read_bytes_mean": agg(m, "ssd_read_bytes"),
             "ssd_read_bytes_total": int(sum(
                 r[m].get("ssd_read_bytes", 0) for r in rows)),
             "ssd_read_chunks_mean": agg(m, "ssd_read_chunks")}
        for k in ("disk_mb", "disk_ms", "preads", "selector_ms", "hook_ms",
                  "touched_chunk_fraction", "logical_kv_ratio",
                  "fallback_rate", "scatter_ms", "model_ms",
                  "n_chunks_selected", "n_chunks_total", "chunk_io_ms",
                  "prepare_ms", "prefill_ms", "ssd_read_ms"):
            v = agg(m, k, None)
            if v is not None:
                d[k] = v
        for k in ("selector_ms", "scatter_ms", "disk_ms"):
            vals = [r[m][k] for r in rows if m in r and r[m].get(k) is not None]
            if vals:
                d[k + "_p50"] = float(np.percentile(vals, 50))
                d[k + "_p95"] = float(np.percentile(vals, 95))
        d["acc_delta_vs_fullload_pp"] = (d["acc"] - agg("fullload", "acc")) * 100
        base = m.split("@")[0]
        if m in ("fullload",) or SELECTORS.get(base, ("", {}))[0] == "cvpr25" \
                or m == "sparsevlm":
            fl = agg("fullload", "disk_mb", None)
            if fl and d.get("disk_mb"):
                d["bytes_vs_fullload"] = d["disk_mb"] / fl
            fms = agg("fullload", "disk_ms", None)
            if fms and d.get("disk_ms"):
                d["io_time_reduction_vs_fullload"] = fms / max(d["disk_ms"], 1e-9)
        d["ttft_speedup_vs_fullload"] = (agg("fullload", "ttft")
                                         / agg(m, "ttft"))
        d["ttft_reduction_vs_fullload_pct"] = (
            (agg("fullload", "ttft") - agg(m, "ttft"))
            / agg("fullload", "ttft") * 100)
        if not args.no_recompute:
            d["ttft_speedup_vs_recompute"] = (agg("recompute", "ttft")
                                              / agg(m, "ttft"))
            d["ttft_reduction_vs_recompute_pct"] = (
                (agg("recompute", "ttft") - agg(m, "ttft"))
                / agg("recompute", "ttft") * 100)
            d["acc_drop_vs_recompute_pp"] = (agg("recompute", "acc")
                                             - d["acc"]) * 100
        if sb and SELECTORS.get(base, ("", {}))[0] == "cvpr25":
            d["first_use_static_ms"] = sb.get("total_first_use_ms")
            d["amortised_static_ms_per_question"] = sb.get(
                "amortised_ms_per_question")
        s["per_method"][m] = d

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else (
        run_dir / "results.json" if run_dir else
        RESULTS_DIR / f"eval_b{int(budget*100)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"schema_version": 2, "summary": s, "rows": rows}, f)
    # per-question raw records, one JSON object per line
    with open(out.with_suffix(".jsonl"), "w") as f:
        for r in rows:
            for m in modes:
                if m in r:
                    f.write(json.dumps({"image_id": r["image_id"],
                                        "question_id": r["question_id"],
                                        "method": m,
                                        "retention": retention[m],
                                        "retention_kind": retention_kind[m],
                                        "dataset": args.dataset,
                                        **r[m]}) + "\n")

    if run_dir:
        _write_run_artifacts(run_dir, rows, modes, s, retention,
                             retention_kind, args)

    hdr = (f"{'method':<24}{'acc':>7}{'TTFT':>9}{'p50':>8}{'p95':>9}"
           f"{'sel ms':>8}{'diskMB':>9}{'diskms':>8}{'chunkf':>8}{'kv':>7}"
           f"{'fb':>7}")
    print(f"\n=== budgets {budgets}  (n={len(rows)}, "
          f"{'cold' if not args.warm else 'warm'}) ===")
    print(hdr)
    for m in modes:
        d = s["per_method"][m]
        def g(k, f="{:>8.1f}", w=8):
            v = d.get(k)
            return f.format(v) if v is not None else " " * w
        print(f"{m:<24}{d['acc']*100:>6.1f}%{d['ttft_mean_ms']:>9.1f}"
              f"{d['ttft_p50_ms']:>8.1f}{d['ttft_p95_ms']:>9.1f}"
              f"{g('selector_ms') if 'selector_ms' in d else g('hook_ms')}"
              f"{g('disk_mb','{:>9.1f}',9)}{g('disk_ms')}"
              f"{g('touched_chunk_fraction','{:>8.3f}')}"
              f"{g('logical_kv_ratio','{:>7.3f}',7)}"
              f"{g('fallback_rate','{:>7.3f}',7)}")
    print(f"\nwrote {out}")


def _pack(r, q, metric="gqa"):
    io = r.get("io") or {}
    d = {"answer": r["answer"],
         # Seconds are retained for compatibility with scripts/07 and /12.
         # In schema v2 this field is TRUE TTFT, not the old E2E value.
         "ttft": r["ttft"],
         "ttft_ms": r["ttft"] * 1e3,
         "decode_latency": r["decode_latency"],
         "decode_ms": r["decode_latency"] * 1e3,
         "e2e_latency": r["e2e_latency"],
         "e2e_latency_ms": r["e2e_latency"] * 1e3,
         "generated_tokens": r["generated_tokens"],
         "acc": METRICS[metric](r["answer"], question_answers(q)),
         "disk_mb": io.get("mb", 0.0),
         "disk_ms": io.get("ms", 0.0),
         "ssd_read_ms": io.get("ms", 0.0),
         "ssd_read_bytes": io.get("bytes", 0),
         # One unit is one K or V chunk span in one layer/file.  The separator
         # sidecar contributes bytes and a pread, but zero chunk units.
         "ssd_read_chunks": io.get("chunk_units", 0),
         "preads": io.get("preads", 0),
         "chunk_units": io.get("chunk_units", 0),
         "fallback_rate": r.get("fallback_rate"),
         "mean_jaccard": r.get("mean_jaccard"),
         "hook_ms": r.get("hook_ms"),
         "prefill_ms": r.get("prefill_ms")}
    for k in ("selector_ms", "select_ms", "query_ms", "chunk_io_ms",
              "scatter_ms", "model_ms", "prepare_ms", "prefill_ms",
              "touched_chunk_fraction", "logical_kv_ratio",
              "n_chunks_selected", "n_chunks_total"):
        if k in r:
            d[k] = r[k]
    return d


def _display_method(key):
    if key == "recompute":
        return "ReComp"
    if key == "fullload":
        return "FullLoad"
    if key == "sparsevlm":
        return "SparseVLM"
    if key.startswith("static_diverse_chunk"):
        return "Static+Diverse"
    return key


def _method_order(keys):
    preferred = ["recompute", "fullload", "sparsevlm",
                 "static_diverse_chunk@25", "static_diverse_chunk@50"]
    rank = {k: i for i, k in enumerate(preferred)}
    return sorted(keys, key=lambda k: (rank.get(k, len(rank)), k))


def _write_run_artifacts(run_dir, rows, modes, summary, retention,
                         retention_kind, args):
    """Write the self-contained schema-v2 CSVs and run README."""
    fields = [
        "dataset", "method_key", "method", "retention", "retention_kind",
        "image_id", "question_id", "question", "prediction",
        "ground_truth", "correct", "generated_tokens", "selector_ms",
        "ssd_read_ms", "scatter_ms", "prepare_ms", "prefill_ms",
        "ttft_ms", "decode_ms", "e2e_latency_ms", "ssd_read_bytes",
        "ssd_read_chunks", "ssd_preads", "chunk_io_ms", "hook_ms",
        "n_chunks_selected", "n_chunks_total", "touched_chunk_fraction",
        "logical_kv_ratio", "fallback_rate",
    ]
    flat = []
    for rec in rows:
        for m in modes:
            v = rec[m]
            flat.append({
                "dataset": args.dataset,
                "method_key": m,
                "method": _display_method(m),
                "retention": retention[m],
                "retention_kind": retention_kind[m],
                "image_id": rec["image_id"],
                "question_id": rec["question_id"],
                "question": rec["question"],
                "prediction": v["answer"],
                "ground_truth": json.dumps(rec["gold"], ensure_ascii=False),
                "correct": v["acc"],
                "generated_tokens": v["generated_tokens"],
                "selector_ms": v.get("selector_ms"),
                "ssd_read_ms": v["ssd_read_ms"],
                "scatter_ms": v.get("scatter_ms"),
                "prepare_ms": v.get("prepare_ms"),
                "prefill_ms": v.get("prefill_ms"),
                "ttft_ms": v["ttft_ms"],
                "decode_ms": v["decode_ms"],
                "e2e_latency_ms": v["e2e_latency_ms"],
                "ssd_read_bytes": v["ssd_read_bytes"],
                "ssd_read_chunks": v["ssd_read_chunks"],
                "ssd_preads": v["preads"],
                "chunk_io_ms": v.get("chunk_io_ms"),
                "hook_ms": v.get("hook_ms"),
                "n_chunks_selected": v.get("n_chunks_selected"),
                "n_chunks_total": v.get("n_chunks_total"),
                "touched_chunk_fraction": v.get("touched_chunk_fraction"),
                "logical_kv_ratio": v.get("logical_kv_ratio"),
                "fallback_rate": v.get("fallback_rate"),
            })

    with open(run_dir / "per_request.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(flat)

    summary_fields = [
        "method_key", "method", "retention", "retention_kind", "n_requests",
        "n_images", "accuracy", "ttft_mean_ms", "ttft_p50_ms",
        "ttft_p95_ms", "ttft_std_ms", "decode_mean_ms", "decode_p50_ms",
        "decode_p95_ms", "e2e_latency_mean_ms", "e2e_latency_p50_ms",
        "e2e_latency_p95_ms", "generated_tokens_mean",
        "ssd_read_bytes_mean", "ssd_read_bytes_total", "ssd_read_mb_mean",
        "ssd_read_chunks_mean", "prepare_mean_ms", "selector_mean_ms",
        "ssd_read_mean_ms", "scatter_mean_ms", "prefill_mean_ms",
        "ttft_reduction_vs_fullload_pct",
        "ttft_reduction_vs_recompute_pct",
    ]
    summary_rows = []
    for m in _method_order(modes):
        d = summary["per_method"][m]
        summary_rows.append({
            "method_key": m,
            "method": _display_method(m),
            "retention": retention[m],
            "retention_kind": retention_kind[m],
            "n_requests": summary["n"],
            "n_images": summary["n_images"],
            "accuracy": d["acc"],
            "ttft_mean_ms": d["ttft_mean_ms"],
            "ttft_p50_ms": d["ttft_p50_ms"],
            "ttft_p95_ms": d["ttft_p95_ms"],
            "ttft_std_ms": d["ttft_std_ms"],
            "decode_mean_ms": d["decode_mean_ms"],
            "decode_p50_ms": d["decode_p50_ms"],
            "decode_p95_ms": d["decode_p95_ms"],
            "e2e_latency_mean_ms": d["e2e_mean_ms"],
            "e2e_latency_p50_ms": d["e2e_p50_ms"],
            "e2e_latency_p95_ms": d["e2e_p95_ms"],
            "generated_tokens_mean": d["generated_tokens_mean"],
            "ssd_read_bytes_mean": d["ssd_read_bytes_mean"],
            "ssd_read_bytes_total": d["ssd_read_bytes_total"],
            "ssd_read_mb_mean": d.get("disk_mb", 0.0),
            "ssd_read_chunks_mean": d["ssd_read_chunks_mean"],
            "prepare_mean_ms": d.get("prepare_ms"),
            "selector_mean_ms": d.get("selector_ms"),
            "ssd_read_mean_ms": d.get("ssd_read_ms", 0.0),
            "scatter_mean_ms": d.get("scatter_ms"),
            "prefill_mean_ms": d.get("prefill_ms"),
            "ttft_reduction_vs_fullload_pct":
                d["ttft_reduction_vs_fullload_pct"],
            "ttft_reduction_vs_recompute_pct":
                d.get("ttft_reduction_vs_recompute_pct"),
        })
    with open(run_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)

    # Algebraic and workload checks use the exact per-request timestamps.
    residuals = [abs(r["e2e_latency_ms"] - r["ttft_ms"] - r["decode_ms"])
                 for r in flat]
    multi = [r for r in flat if int(r["generated_tokens"]) >= 2]
    request_keys = [(r["method_key"], str(r["image_id"]),
                     str(r["question_id"])) for r in flat]
    timed_fields = ("ttft_ms", "decode_ms", "e2e_latency_ms",
                    "ssd_read_ms", "ssd_read_bytes", "ssd_read_chunks")
    sd_rows = [r for r in flat if r["method"] == "Static+Diverse"]
    prep_residuals = [abs(r["ttft_ms"] - r["prepare_ms"] - r["prefill_ms"])
                      for r in sd_rows]
    sanity = {
        "expected_csv_rows": len(rows) * len(modes),
        "actual_csv_rows": len(flat),
        "unique_images": len({r["image_id"] for r in flat}),
        "unique_questions": len({r["question_id"] for r in flat}),
        "duplicate_method_image_question_rows":
            len(request_keys) - len(set(request_keys)),
        "requests_per_method": {
            m: sum(r["method_key"] == m for r in flat) for m in modes},
        "all_required_times_finite_nonnegative": all(
            np.isfinite(float(r[k])) and float(r[k]) >= 0
            for r in flat for k in timed_fields),
        "ttft_lt_e2e_fraction": float(np.mean([
            r["ttft_ms"] < r["e2e_latency_ms"] for r in flat])),
        "multi_token_requests": len(multi),
        "multi_token_decode_positive_fraction": float(np.mean([
            r["decode_ms"] > 0 for r in multi])) if multi else 1.0,
        "max_abs_e2e_minus_ttft_decode_ms": max(residuals, default=0.0),
        "max_abs_static_ttft_minus_prepare_prefill_ms":
            max(prep_residuals, default=0.0),
        "generated_tokens_min": min(
            (int(r["generated_tokens"]) for r in flat), default=0),
        "generated_tokens_max": max(
            (int(r["generated_tokens"]) for r in flat), default=0),
        "generated_token_cap_respected": all(
            1 <= int(r["generated_tokens"]) <= summary["max_new_tokens"]
            for r in flat),
    }
    with open(run_dir / "sanity.json", "w") as f:
        json.dump(sanity, f, indent=1)

    def ret_label(r):
        if r["retention"] in (None, ""):
            return "-"
        suffix = " tok" if r["retention_kind"] == "token" else ""
        return f"{float(r['retention'])*100:.0f}%{suffix}"

    table = [
        "| Method | Retention | Accuracy | True TTFT mean | TTFT p50 | "
        "TTFT p95 | Decode mean | E2E mean | SSD Read |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        table.append(
            f"| {r['method']} | {ret_label(r)} | {r['accuracy']*100:.1f}% | "
            f"{r['ttft_mean_ms']:.1f} ms | {r['ttft_p50_ms']:.1f} ms | "
            f"{r['ttft_p95_ms']:.1f} ms | {r['decode_mean_ms']:.1f} ms | "
            f"{r['e2e_latency_mean_ms']:.1f} ms | "
            f"{r['ssd_read_mb_mean']:.1f} MB |")

    reductions = [
        "| Method | vs FullLoad | vs ReComp |",
        "|---|---:|---:|",
    ]
    for r in summary_rows:
        if r["method"] != "Static+Diverse":
            continue
        reductions.append(
            f"| Static+Diverse {ret_label(r)} | "
            f"{r['ttft_reduction_vs_fullload_pct']:.1f}% | "
            f"{r['ttft_reduction_vs_recompute_pct']:.1f}% |")

    detail = [
        "| Retention | Prepare | Selector | SSD pread | Scatter | Prefill |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        if r["method"] != "Static+Diverse":
            continue
        detail.append(
            f"| {ret_label(r)} | {r['prepare_mean_ms']:.1f} ms | "
            f"{r['selector_mean_ms']:.1f} ms | "
            f"{r['ssd_read_mean_ms']:.1f} ms | "
            f"{r['scatter_mean_ms']:.1f} ms | "
            f"{r['prefill_mean_ms']:.1f} ms |")

    text = f"""# GQA 40/240 true-TTFT rerun

Generated by schema-v2 latency instrumentation on {time.strftime('%Y-%m-%d')}.

## Measurement definitions

- **True TTFT:** request timer start through synchronized availability of the
  first output token. For SSD methods this includes selection, SSD reads,
  cache reconstruction/scatter, prompt prefill, and first-token selection.
- **Decode latency:** immediately after the first-token boundary through the
  synchronized completion of the final generated token.
- **E2E latency:** request timer start through final-token completion. CPU
  detokenization and artifact writing are outside this timer.
- Processor/tokenization, host-to-device input preparation, cache eviction,
  store creation, calibration, reordering, and static metadata construction
  are outside the online timers, matching the historical experiment.

The historical result files called their full-generation measurement `ttft`.
That old field corresponds to the new **E2E latency**, not true TTFT. The old
stored-KV decoder also executed one unused forward whenever generation reached
its 16-token cap; schema v2 removes that forward without changing the cap or
returned tokens. `generated_tokens` counts model token decisions, including
terminal EOS.

## Workload and fixed conditions

- Dataset: GQA frozen `{summary['index']}`; slice
  `[{summary['skip']}:{summary['skip'] + summary['questions_per_image_requested']}]`
- Index SHA256: `{summary['index_sha256']}`
- Ordered evaluation-workload SHA256: `{summary['workload_sha256']}`
- Workload: **{summary['n_images']} unique images / {summary['n']} questions**
- Questions per image: min {summary['questions_per_image']['min']}, max
  {summary['questions_per_image']['max']}, mean
  {summary['questions_per_image']['mean']:.1f}
- Model: `llava-hf/llava-v1.6-vicuna-7b-hf`, 4-bit NF4, eager attention
- Greedy decoding, maximum {summary['max_new_tokens']} output tokens
- 64-token visual-KV chunks; cold page cache; separator sidecar
- SparseVLM retention 25%; Static+Diverse chunk budgets 25% and 50%;
  `diverse_frac=0.25`

## Main results

{chr(10).join(table)}

Positive reductions mean lower true TTFT than the named baseline.

{chr(10).join(reductions)}

## Static+Diverse breakdown

`prepare_ms` spans all preparation work; component timers can leave small
bookkeeping gaps and therefore need not sum exactly to it. `ssd_read_ms` times
only `os.pread`, excluding `posix_fadvise` cache eviction.

For Static+Diverse, `prepare_ms` is the authoritative outer wall-clock span
from request start through text embedding, fresh prefix-cache initialization,
selection, SSD reads/CPU conversion, and cache scatter. Thus it is broader
than `selector_ms + ssd_read_ms + scatter_ms`; `ssd_read_ms` intentionally
excludes CPU buffer-to-tensor conversion. For FullLoad and SparseVLM,
`prefill_ms` is hook-inclusive because SSD reads, selection, and scatter run in
layer pre-hooks during prompt prefill; it is not a pure model-compute timer.

{chr(10).join(detail)}

## Sanity checks

- CSV rows: {sanity['actual_csv_rows']} / expected
  {sanity['expected_csv_rows']}
- Unique images/questions: {sanity['unique_images']} / {sanity['unique_questions']}
- `TTFT < E2E`: {sanity['ttft_lt_e2e_fraction']*100:.1f}% of requests
- Multi-token requests with positive decode: 
  {sanity['multi_token_decode_positive_fraction']*100:.1f}%
- Maximum `abs(E2E - TTFT - decode)`: 
  {sanity['max_abs_e2e_minus_ttft_decode_ms']:.6f} ms
- Duplicate method/image/question rows:
  {sanity['duplicate_method_image_question_rows']}
- All required timing/I/O fields finite and nonnegative:
  {sanity['all_required_times_finite_nonnegative']}
- Maximum Static+Diverse `abs(TTFT - prepare - prefill)`:
  {sanity['max_abs_static_ttft_minus_prepare_prefill_ms']:.6f} ms
- Generated-token range: {sanity['generated_tokens_min']}--{sanity['generated_tokens_max']}
- Generated-token cap respected: {sanity['generated_token_cap_respected']}

## Reproduction

```bash
{summary['command']}
```

Modified files: `mmimpress/serve.py`, `scripts/04_eval.py`, and `README.md`.
The workspace is not a Git worktree, so a Git diff/commit identifier is not
available. Existing result files under `results/` were not overwritten.

`ssd_read_chunks` counts physical K/V chunk spans plus SparseVLM probe-sidecar
chunk-equivalents per layer/file, not `pread` syscalls or unique chunk IDs. The
separator sidecar contributes SSD bytes and one pread but zero chunk units.
Empty CSV cells mean the stage is not separately measurable for that method;
numerical zero means no such work occurred (for example ReComp SSD reads).
"""
    (run_dir / "README.md").write_text(text)


def logical_attn_gflops(meta, n_query, kv_ratio):
    """Prefill attention FLOPs a system would spend if the dropped visual KV
    were physically absent.

    Reported as LOGICAL on purpose: this implementation masks rather than
    shortens the key tensor, so its measured compute is unchanged.  The number
    says what the selection would be worth to an engine that can skip masked
    keys, and keeps that claim separate from what was actually measured.
    """
    L, H, hd = meta["num_layers"], meta["num_heads"], meta["head_dim"]
    kv = meta["v_token_start"] + meta["v_token_num"] * kv_ratio + n_query
    return 2 * 2 * L * H * n_query * kv * hd / 1e9


if __name__ == "__main__":
    main()

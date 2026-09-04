"""Three-mode evaluation: IMPRESS vs FullLoad vs ReComp on GQA.

  impress    identify-then-load: probe sidecar -> Jaccard vs threshold ->
             consensus chunks, everything unloaded is masked out
  fullload   reuse the prefix but read every chunk, no selection (AS-like)
  recompute  no store at all, prefill from pixels (ReComp)

All three answer the same questions with the same model, so accuracy is
directly comparable; disk numbers come from real preads with the page cache
dropped per request.

  python scripts/04_eval.py --questions 6 --limit 40 --ratio 0.25
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmimpress.config import (ALPHA, PROBE_HEADS, PROJECT_ROOT, RESULTS_DIR,
                              RETENTION_RATIO, STORE_DIR)
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
    args = ap.parse_args()

    store_dir = Path(args.store) if args.store else STORE_DIR
    runner = LlavaRunner().load()
    srv = Server(runner, ratio=args.ratio, probe=args.probe, alpha=args.alpha)
    index = load_index(args.index)
    if args.only:
        keep = set(args.only.split(","))
        index = [e for e in index if str(e["image_id"]) in keep]
    if args.limit:
        index = index[:args.limit]

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
                rec["recompute"] = {
                    "answer": rc["answer"], "ttft": rc["ttft"],
                    "acc": METRICS[args.metric](rc["answer"],
                                                question_answers(q))}
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

    s = {"n": len(rows), "ratio": args.ratio, "budget": budget,
         "budgets": budgets, "metric": args.metric,
         "alpha": args.alpha, "probe_heads": args.probe,
         "lam_static": args.lam_static, "lam_query": args.lam_query,
         "diverse_frac": args.diverse_frac,
         "sep_policy": args.sep_policy, "cold": not args.warm,
         "static_build": sb, "per_method": {}}
    for m in modes:
        d = {"acc": agg(m, "acc"),
             "ttft_mean_ms": agg(m, "ttft") * 1e3,
             "ttft_p50_ms": float(np.median([r[m]["ttft"] for r in rows])) * 1e3,
             "ttft_p95_ms": float(np.percentile([r[m]["ttft"] for r in rows],
                                                95)) * 1e3}
        for k in ("disk_mb", "disk_ms", "preads", "selector_ms", "hook_ms",
                  "touched_chunk_fraction", "logical_kv_ratio",
                  "fallback_rate", "scatter_ms", "model_ms",
                  "n_chunks_selected", "n_chunks_total", "chunk_io_ms"):
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
        if not args.no_recompute:
            d["ttft_speedup_vs_recompute"] = (agg("recompute", "ttft")
                                              / agg(m, "ttft"))
            d["acc_drop_vs_recompute_pp"] = (agg("recompute", "acc")
                                             - d["acc"]) * 100
        if sb and SELECTORS.get(base, ("", {}))[0] == "cvpr25":
            d["first_use_static_ms"] = sb.get("total_first_use_ms")
            d["amortised_static_ms_per_question"] = sb.get(
                "amortised_ms_per_question")
        s["per_method"][m] = d

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else (
        RESULTS_DIR / f"eval_b{int(budget*100)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"summary": s, "rows": rows}, f)
    # per-question raw records, one JSON object per line
    with open(out.with_suffix(".jsonl"), "w") as f:
        for r in rows:
            for m in modes:
                if m in r:
                    f.write(json.dumps({"image_id": r["image_id"],
                                        "question_id": r["question_id"],
                                        "method": m, "budget": budget,
                                        "dataset": args.dataset,
                                        **r[m]}) + "\n")

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
    d = {"answer": r["answer"], "ttft": r["ttft"],
         "acc": METRICS[metric](r["answer"], question_answers(q)),
         "disk_mb": r["io"]["mb"], "disk_ms": r["io"]["ms"],
         "preads": r["io"]["preads"], "chunk_units": r["io"]["chunk_units"],
         "fallback_rate": r.get("fallback_rate"),
         "mean_jaccard": r.get("mean_jaccard"),
         "hook_ms": r.get("hook_ms", 0.0)}
    for k in ("selector_ms", "select_ms", "query_ms", "chunk_io_ms",
              "scatter_ms", "model_ms", "prepare_ms",
              "touched_chunk_fraction", "logical_kv_ratio",
              "n_chunks_selected", "n_chunks_total"):
        if k in r:
            d[k] = r[k]
    return d


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

"""Build the reusable per-image static sidecar (VisionZip + PACT metadata).

This is the offline / first-use cost of the new selectors.  Everything written
here is question-independent, so it is paid once per image and amortised over
that image's questions -- GQA gives ~29 questions per image, and the measured
across-question top-k Jaccard of 0.712 is what makes the reuse defensible.

Written to <store>/<image_id>/static.pt:
  token_score   (v_num,)            VisionZip CLS-to-patch saliency, ORIGINAL
                                    token order, +inf on row separators
  chunk_score   (L, n_chunks)       per-layer chunk scores, in THAT layer's
                                    stored order (the store is reordered per
                                    layer, so chunk boundaries differ)
  chunk_keys    (L, n_chunks, hid)  PACT-style image key descriptor per chunk
  sep_chunks    list[list[int]]     chunks holding row separators, per layer

  python scripts/06_build_static.py [--limit N] [--agg topk_mean] [--alpha 0.7]
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

from mmimpress.config import PROJECT_ROOT, RESULTS_DIR, STORE_DIR
from mmimpress.cvpr25 import (aggregate_chunk_scores, anyres_token_scores,
                              clip_cls_patch_saliency, image_chunk_keys)
from mmimpress.dataset import load_index
from mmimpress.model import LlavaRunner
from mmimpress.store import load_meta


def layer_orders(meta):
    """stored -> original permutation for every layer."""
    o = meta.get("order")
    L, vn = meta["num_layers"], meta["v_token_num"]
    if not o:
        return [list(range(vn))] * L
    return o if meta.get("order_is_per_layer") else [o] * L


@torch.no_grad()
def build_one(runner, entry, agg_mode, alpha, top_frac, store=None):
    d = (store or STORE_DIR) / str(entry["image_id"])
    meta = load_meta(d)
    vn, cs = meta["v_token_num"], meta["chunk_size"]
    L, nc = meta["num_layers"], meta["n_chunks_per_layer"]

    img = Image.open(PROJECT_ROOT / entry["image_path"]).convert("RGB")
    enc = runner.encode(img, entry["questions"][0]["question"])

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    per_sub = clip_cls_patch_saliency(runner, enc["pixel_values"])
    tok = anyres_token_scores(runner, per_sub,
                              enc["image_sizes"][0].tolist(), vn)
    torch.cuda.synchronize()
    t_clip = time.perf_counter() - t0

    v_hidden = torch.load(d / "v_hidden.pt", weights_only=True)

    t1 = time.perf_counter()
    orders = layer_orders(meta)
    chunk_score = torch.zeros(L, nc)
    chunk_keys = torch.zeros(L, nc, v_hidden.shape[-1], dtype=torch.float16)
    sep_chunks = []
    for li in range(L):
        perm = torch.tensor(orders[li], dtype=torch.long)   # stored -> original
        tok_stored = tok[perm]
        chunk_score[li] = aggregate_chunk_scores(tok_stored, cs, agg_mode,
                                                 alpha, top_frac)
        chunk_keys[li] = image_chunk_keys(runner, v_hidden[perm], cs)
        sep = torch.nonzero(~torch.isfinite(tok_stored)).flatten()
        sep_chunks.append(sorted({int(p) // cs for p in sep.tolist()}))
    torch.cuda.synchronize()
    t_meta = time.perf_counter() - t1

    # Row separators are structural: dropping them collapses LLaVA-NeXT's
    # spatial layout (measured: the model stops answering and starts
    # captioning).  Forcing the chunks that hold them costs ~1/3 of the chunk
    # budget, because in stored order they are spread over many chunks.  So
    # they get their own tiny always-loaded sidecar instead -- 1.5% of the
    # image's KV, one sequential read, and the chunk budget stays intact.
    t2 = time.perf_counter()
    sep_pos, sep_k, sep_v = [], [], []
    for li in range(L):
        perm = torch.tensor(orders[li], dtype=torch.long)
        pos = torch.nonzero(~torch.isfinite(tok[perm])).flatten()
        sep_pos.append(pos.tolist())
        kb = np.fromfile(d / f"layer_{li:02d}" / "k.bin", dtype=np.float16)
        vb = np.fromfile(d / f"layer_{li:02d}" / "v.bin", dtype=np.float16)
        H, hd = meta["num_heads"], meta["head_dim"]
        kb = torch.from_numpy(kb).view(vn, H, hd)[pos]
        vb = torch.from_numpy(vb).view(vn, H, hd)[pos]
        sep_k.append(kb)
        sep_v.append(vb)
    n_sep = len(sep_pos[0])
    assert all(len(p) == n_sep for p in sep_pos), "separator count varies"
    sep_blob = torch.stack([torch.stack(sep_k), torch.stack(sep_v)])
    sep_blob.numpy().tofile(d / "sep_kv.bin")     # (2, L, n_sep, H, hd) fp16
    t_sep = time.perf_counter() - t2

    torch.save({"token_score": tok, "chunk_score": chunk_score,
                "chunk_keys": chunk_keys, "sep_chunks": sep_chunks,
                "sep_pos": sep_pos, "n_sep": n_sep,
                "sep_kv_shape": list(sep_blob.shape),
                "agg": {"mode": agg_mode, "alpha": alpha,
                        "top_frac": top_frac}},
               d / "static.pt")
    nbytes = ((d / "static.pt").stat().st_size
              + (d / "sep_kv.bin").stat().st_size)
    return {"clip_ms": t_clip * 1e3, "meta_ms": (t_meta + t_sep) * 1e3,
            "bytes": nbytes, "n_questions": len(entry["questions"]),
            "sep_chunks_mean": float(np.mean([len(s) for s in sep_chunks])),
            "n_chunks": nc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--store", default=None)
    ap.add_argument("--agg", default="topk_mean",
                    choices=("topk_mean", "mean", "max", "sum"))
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--top-frac", type=float, default=0.25)
    ap.add_argument("--index", default=None)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    global STORE_DIR
    if args.store:
        import mmimpress.config as _c
        STORE_DIR = Path(args.store)
    runner = LlavaRunner().load()
    index = load_index(args.index) if getattr(args, "index", None) else load_index()
    if args.only:
        keep = set(args.only.split(","))
        index = [e for e in index if str(e["image_id"]) in keep]
    if args.limit:
        index = index[:args.limit]
    rows = []
    for i, e in enumerate(index):
        if not (STORE_DIR / str(e["image_id"]) / "meta.json").exists():
            continue
        r = build_one(runner, e, args.agg, args.alpha, args.top_frac, STORE_DIR)
        rows.append(r)
        print(f"[{i+1}/{len(index)}] {e['image_id']}: CLIP {r['clip_ms']:.0f} ms"
              f" + metadata {r['meta_ms']:.0f} ms, {r['bytes']/1e6:.1f} MB, "
              f"sep chunks {r['sep_chunks_mean']:.1f}/{r['n_chunks']}")
        if (i + 1) % 8 == 0:
            torch.cuda.empty_cache()

    clip = float(np.mean([r["clip_ms"] for r in rows]))
    meta = float(np.mean([r["meta_ms"] for r in rows]))
    nq = float(np.mean([r["n_questions"] for r in rows]))
    summary = {"n_images": len(rows), "clip_ms": clip, "meta_ms": meta,
               "total_first_use_ms": clip + meta,
               "mean_questions_per_image": nq,
               "amortised_ms_per_question": (clip + meta) / max(nq, 1),
               "sidecar_mb": float(np.mean([r["bytes"] for r in rows])) / 1e6,
               "sep_chunks_mean": float(np.mean([r["sep_chunks_mean"]
                                                 for r in rows])),
               "n_chunks": rows[0]["n_chunks"] if rows else 0,
               "agg": {"mode": args.agg, "alpha": args.alpha,
                       "top_frac": args.top_frac}}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "static_build.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nfirst-use per image: CLIP {clip:.0f} ms + metadata {meta:.0f} ms "
          f"= {clip+meta:.0f} ms")
    print(f"amortised over {nq:.1f} questions/image: "
          f"{summary['amortised_ms_per_question']:.1f} ms/question")
    print(f"sidecar {summary['sidecar_mb']:.1f} MB/image")


if __name__ == "__main__":
    main()

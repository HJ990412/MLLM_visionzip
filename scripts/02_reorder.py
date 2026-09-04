"""KV reordering pass (IMPRESS 4.4.1) over the image prefix store.

Without this the store sits in raster order and every 64-token chunk is a thin
horizontal strip of the patch grid, so a scattered selection touches
essentially every chunk and selective loading reads MORE than a full load (the
probe sidecar and the fallback layers are pure overhead).  Repacking is what
turns identification into I/O savings.

Two orders:
  importance  IMPRESS 4.4.1 -- descending AVERAGE SparseVLM importance,
              accumulated over several questions about the same image, one
              order per layer (the consensus path applies one token set to all
              heads, so a per-layer order is what the read pattern wants)
  morton      query-independent Z-order of the patch grid, so a chunk is a
              compact spatial tile instead of a strip; needs no calibration

Rewrites layer_*/k.bin and v.bin in place and records the permutation in
meta.json.  The serving path works in stored positions, so nothing has to be
un-permuted at request time.

  python scripts/02_reorder.py --order importance --calib-questions 4
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmimpress.config import STORE_DIR
from mmimpress.dataset import load_index
from mmimpress.reorder import llava_visual_order, mapping_from_perm
from mmimpress.store import load_meta

_NP = {"float16": np.float16, "float32": np.float32}


def current_order(meta, layer):
    """stored -> original for one layer of the CURRENT store.

    meta["order"] is a single list when every layer shares an order and a list
    of per-layer lists otherwise; a store that was never reordered has none.
    """
    o = meta.get("order")
    if not o:
        return list(range(meta["v_token_num"]))
    return o[layer] if meta.get("order_is_per_layer") else o


def rewrite_layer(ldir: Path, meta, perm_np):
    """Permute the token axis of the token-major (v_num, W, hd) blocks.

    The probe sidecar has to move with them or identification would score the
    wrong rows.
    """
    vn, hd = meta["v_token_num"], meta["head_dim"]
    dt = _NP[meta["dtype"]]
    for name, width in (("k", meta["num_heads"]), ("v", meta["num_heads"]),
                        ("probe_k", meta["probe_heads"])):
        p = ldir / f"{name}.bin"
        a = np.fromfile(p, dtype=dt).reshape(vn, width, hd)
        a[:] = a[perm_np]
        a.tofile(p)


def commit(store_dir: Path, meta, perms):
    """perms: {layer -> permutation over CURRENT stored positions}."""
    per_layer_order, per_layer_nl = [], []
    for li in range(meta["num_layers"]):
        cur = current_order(meta, li)
        perm = perms[li]
        rewrite_layer(store_dir / f"layer_{li:02d}", meta, np.asarray(perm))
        new_order = [cur[p] for p in perm]            # stored -> original
        inv = mapping_from_perm(new_order)            # original -> stored
        per_layer_order.append(new_order)
        per_layer_nl.append([inv[i] for i in meta["newline_idx"]])

    same = all(o == per_layer_order[0] for o in per_layer_order)
    meta["order"] = per_layer_order[0] if same else per_layer_order
    meta["order_is_per_layer"] = not same
    # separators must be locatable per layer; when the order is shared this is
    # a single list, otherwise the selector needs the per-layer variant
    meta["newline_stored"] = (per_layer_nl[0] if same else per_layer_nl)
    meta["reordered"] = True
    with open(store_dir / "meta.json", "w") as f:
        json.dump(meta, f)
    return same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", choices=("importance", "morton"),
                    default="importance")
    ap.add_argument("--calib-questions", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--store", default=str(STORE_DIR))
    ap.add_argument("--index", default=None)
    args = ap.parse_args()

    index = load_index(args.index)
    if args.limit:
        index = index[:args.limit]
    store = Path(args.store)

    runner = server = None
    if args.order == "importance":
        from mmimpress.model import LlavaRunner
        from mmimpress.serve import Server
        runner = LlavaRunner().load()
        server = Server(runner)

    for i, e in enumerate(index):
        d = store / str(e["image_id"])
        if not (d / "meta.json").exists():
            continue
        meta = load_meta(d)
        t0 = time.time()

        if args.order == "morton":
            base = meta["base_grid"]
            hi_h, hi_w = meta["hires_grid"]
            perm, _ = llava_visual_order(base, hi_h, hi_w,
                                         meta["newline_idx"],
                                         meta["v_token_num"])
            # perm is over ORIGINAL positions; convert to current stored space
            perms = {}
            for li in range(meta["num_layers"]):
                pos = {o: s for s, o in enumerate(current_order(meta, li))}
                perms[li] = [pos[o] for o in perm]
        else:
            from mmimpress.serve import ImageContext, calibrate_image
            ctx = ImageContext(d, runner.model.device, drop_cache=False)
            qs = [q["question"] for q in e["questions"][:args.calib_questions]]
            scores = calibrate_image(server, ctx, qs)      # (L, v_num)
            ctx.close()
            del ctx
            torch.cuda.empty_cache()
            perms = {li: torch.argsort(scores[li], descending=True,
                                       stable=True).tolist()
                     for li in range(meta["num_layers"])}

        shared = commit(d, meta, perms)
        print(f"[{i+1}/{len(index)}] {e['image_id']}: {args.order} order "
              f"({'shared' if shared else 'per-layer'}) "
              f"({time.time()-t0:.1f}s)")
    print("reorder done")


if __name__ == "__main__":
    main()

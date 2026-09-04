"""Why selective loading does or does not pay off: measure, do not assume.

IMPRESS's chunk skipping only works if the important-token set is STABLE across
the queries that share a prefix -- reordering commits one layout up front and
every later request has to live with it.  IMPRESS's own Challenge 1 says
importance is query-dependent; this script quantifies how query-dependent it is
for an image prefix, and therefore what fraction of chunks any reordering can
hope to skip.

Per image it scores several questions, then reports per layer:
  jaccard_q       mean pairwise Jaccard of the top-k sets of two questions
  frac_raster     chunk fraction a held-out question touches in stored order
  frac_reordered  ... after reordering by the mean score of the OTHER questions
  frac_oracle     ... if the store were reordered for that question alone
                      (= k / chunk_size / n_chunks, the floor)

  python scripts/03_diagnose.py --questions 8 --limit 4
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmimpress.config import RESULTS_DIR, RETENTION_RATIO, STORE_DIR
from mmimpress.dataset import load_index
from mmimpress.model import LlavaRunner
from mmimpress.serve import (Calibrator, ImageContext, Server, suffix_ids_for)
from mmimpress.sparsevlm import topk_budget


@torch.no_grad()
def score_question(server, ctx, question):
    """(num_layers, v_num) per-layer visual importance for one question."""
    runner = server.runner
    dev = runner.model.device
    suffix_ids = suffix_ids_for(runner, question)
    rr = server.raters(ctx, suffix_ids)
    cache = ctx.cache.new_request()
    for li in range(ctx.meta["num_layers"]):
        for kind in ("k", "v"):
            ctx.cache.write_full(li, kind, ctx.reader.read_full(li, kind))
    cal = Calibrator(runner, ctx, rr)
    with cal:
        P, n = ctx.meta["prefix_len"], suffix_ids.shape[0]
        pos = torch.arange(P, P + n, device=dev)
        runner.model(input_ids=suffix_ids.to(dev).unsqueeze(0),
                     attention_mask=torch.ones(1, P + n, dtype=torch.long,
                                               device=dev),
                     position_ids=pos.unsqueeze(0), cache_position=pos,
                     past_key_values=cache, use_cache=True)
    out = torch.stack([cal.scores[li]
                       for li in range(ctx.meta["num_layers"])])
    del cache
    return out


def chunk_frac(positions, cs, n_chunks):
    return len({int(p) // cs for p in positions}) / n_chunks


def sweep_chunk(tops, order_pos, meta, sizes=(16, 32, 64, 128, 256)):
    """Byte fraction a selection costs at several chunk sizes.

    Smaller chunks waste fewer bytes on the scattered tail of the selection but
    cost more preads; this is IMPRESS's chunk-size sensitivity (Fig 23) with
    the read amplification made explicit.
    """
    vn = order_pos.shape[0]
    out = {}
    for cs in sizes:
        nc = (vn + cs - 1) // cs
        fr, pr = [], []
        for t in tops:
            cids = {int(p) // cs for p in order_pos[t].tolist()}
            fr.append(len(cids) / nc)
            runs = 1 + sum(1 for a, b in zip(sorted(cids), sorted(cids)[1:])
                           if b != a + 1)
            pr.append(runs)
        out[cs] = {"byte_frac": float(np.mean(fr)),
                   "preads_per_layer_kind": float(np.mean(pr)),
                   "n_chunks": nc}
    return out


def analyse(scores, meta, ratio):
    """scores: (Q, L, v_num) in stored order."""
    Q, L, vn = scores.shape
    cs = meta["chunk_size"]
    nc = meta["n_chunks_per_layer"]
    nl = set(meta["newline_idx"] if not meta.get("reordered")
             else (meta["newline_stored"][0]
                   if meta.get("order_is_per_layer")
                   else meta["newline_stored"]))
    k = topk_budget(meta["n_spatial"], ratio)
    masked = scores.clone()
    if nl:
        masked[:, :, torch.tensor(sorted(nl))] = -float("inf")
    tops = masked.topk(k, dim=-1).indices                 # (Q, L, k)

    jac, f_ras, f_reo = [], [], []
    for li in range(L):
        sets = [set(tops[qi, li].tolist()) for qi in range(Q)]
        pj = [len(sets[a] & sets[b]) / max(1, len(sets[a] | sets[b]))
              for a in range(Q) for b in range(a + 1, Q)]
        jac.append(float(np.mean(pj)) if pj else 1.0)
        rr, oo = [], []
        for qi in range(Q):
            rr.append(chunk_frac(tops[qi, li].tolist(), cs, nc))
            # leave-one-out: order by the mean score of the other questions
            others = [j for j in range(Q) if j != qi]
            mean_s = scores[others, li].mean(0)
            if nl:
                mean_s = mean_s.clone()
                mean_s[torch.tensor(sorted(nl))] = -float("inf")
            order = torch.argsort(mean_s, descending=True, stable=True)
            newpos = torch.empty(vn, dtype=torch.long)
            newpos[order] = torch.arange(vn)
            oo.append(chunk_frac(newpos[tops[qi, li]].tolist(), cs, nc))
        f_ras.append(float(np.mean(rr)))
        f_reo.append(float(np.mean(oo)))

    # chunk-size sweep on the middle layer's leave-one-out reordering
    lm = L // 2
    mean_s = scores[:, lm].mean(0).clone()
    if nl:
        mean_s[torch.tensor(sorted(nl))] = -float("inf")
    order = torch.argsort(mean_s, descending=True, stable=True)
    newpos = torch.empty(vn, dtype=torch.long)
    newpos[order] = torch.arange(vn)
    sweep = sweep_chunk([tops[qi, lm] for qi in range(Q)], newpos, meta)

    return {"k": k, "n_chunks": nc, "chunk_size": cs, "sweep": sweep,
            "floor": float(np.ceil(k / cs) / nc),
            "jaccard_q": jac, "frac_raster": f_ras, "frac_reordered": f_reo}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=8)
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--ratio", type=float, default=RETENTION_RATIO)
    args = ap.parse_args()

    runner = LlavaRunner().load()
    server = Server(runner)
    out = {}
    for e in load_index()[:args.limit]:
        d = STORE_DIR / str(e["image_id"])
        if not (d / "meta.json").exists():
            continue
        ctx = ImageContext(d, runner.model.device, drop_cache=False)
        qs = [q["question"] for q in e["questions"][:args.questions]]
        sc = torch.stack([score_question(server, ctx, q) for q in qs])
        res = analyse(sc, ctx.meta, args.ratio)
        out[e["image_id"]] = res
        ctx.close()
        del ctx
        torch.cuda.empty_cache()
        j = np.mean(res["jaccard_q"])
        print(f"{e['image_id']}: k={res['k']}/{ctx_n(res)} "
              f"floor={res['floor']:.3f} | across-question Jaccard "
              f"{j:.3f} | chunk frac raster {np.mean(res['frac_raster']):.3f}"
              f" -> reordered {np.mean(res['frac_reordered']):.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / "diagnose.json"
    with open(p, "w") as f:
        json.dump(out, f)
    if out:
        sizes = sorted(next(iter(out.values()))["sweep"], key=int)
        print("\nchunk-size sweep (mid layer, leave-one-out reorder):")
        print("  size   byte_frac  preads/layer/kind")
        for cs in sizes:
            bf = np.mean([v["sweep"][cs]["byte_frac"] for v in out.values()])
            pr = np.mean([v["sweep"][cs]["preads_per_layer_kind"]
                          for v in out.values()])
            print(f"  {cs:>4}   {bf:8.3f}   {pr:8.1f}")
    allj = [np.mean(v["jaccard_q"]) for v in out.values()]
    allr = [np.mean(v["frac_raster"]) for v in out.values()]
    allo = [np.mean(v["frac_reordered"]) for v in out.values()]
    fl = [v["floor"] for v in out.values()]
    print(f"\nMEAN over {len(out)} images: Jaccard(q,q')={np.mean(allj):.3f} | "
          f"chunk frac raster {np.mean(allr):.3f} -> reordered "
          f"{np.mean(allo):.3f} (floor {np.mean(fl):.3f})")
    print(f"wrote {p}")


def ctx_n(res):
    return res["n_chunks"] * res["chunk_size"]


if __name__ == "__main__":
    main()

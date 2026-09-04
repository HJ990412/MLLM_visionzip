"""Where does the SparseVLM selector's 677 ms actually go?

Instruments LayerSelector by subclassing it (the baseline class is untouched)
and splitting the per-layer hook into the stages that could plausibly dominate:

  qk_proj    re-projecting Q/K for the suffix inside the hook (work the layer
             is about to do again anyway -- pure duplication)
  probe_io   reading the probe-key sidecar off SSD
  score      rater x visual softmax over the probe heads
  jaccard    pairwise agreement + consensus vote
  chunk_io   reading the selected chunks off SSD
  scatter    writing them into the GPU prefix cache
  fallback   the whole-layer path when agreement is below threshold

Every stage is CUDA-synchronised, so GPU time is attributed to the stage that
launched it rather than to whoever synchronises next.

  python scripts/05_profile_sparsevlm.py --questions 3 --limit 4
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import transformers.models.llama.modeling_llama as ml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmimpress.config import RESULTS_DIR, RETENTION_RATIO, STORE_DIR
from mmimpress.dataset import load_index
from mmimpress.model import LlavaRunner
from mmimpress.serve import (BIAS, ImageContext, LayerSelector, Server,
                             bias_from_keep, mean_pairwise_jaccard,
                             suffix_ids_for)
from mmimpress import sparsevlm as sv
from mmimpress.store import chunks_for_tokens


class _T:
    def __init__(self):
        self.t = defaultdict(float)

    def __call__(self, key, fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        self.t[key] += time.perf_counter() - t0
        return out


class ProfiledSelector(LayerSelector):
    """LayerSelector with the same maths, split into timed stages."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.T = _T()

    def _run(self, li, layer, h, kwargs):
        m, ctx = self.ctx.meta, self.ctx
        T = self.T
        dev = h.device
        H, hd = m["num_heads"], m["head_dim"]
        v0, vn = m["v_token_start"], m["v_token_num"]
        n = h.shape[1]

        if self.mode == "fullload":
            T("chunk_io", lambda: self._load_layer_full(li))
            self.log.append({"layer": li, "mode": "full", "sim": None})
            return

        attn = layer.self_attn

        def qk():
            hn = layer.input_layernorm(h)
            q = attn.q_proj(hn).view(1, n, H, hd).transpose(1, 2)
            kn = attn.k_proj(hn).view(1, n, H, hd).transpose(1, 2)
            cos, sin = kwargs["position_embeddings"]
            q, kn = ml.apply_rotary_pos_emb(q, kn, cos, sin)
            return q[0], kn[0]

        q, kn = T("qk_proj", qk)
        P = min(self.probe, H)
        sys_k = ctx.cache.k[li][0, :, :v0]

        pk = T("probe_io", lambda: ctx.reader.read_probe(li, self.io))
        pk = pk.to(dev, q.dtype).permute(1, 0, 2)[:P]
        keys_p = torch.cat([sys_k[:P], pk, kn[:P]], dim=1)
        sp = T("score", lambda: sv.rater_visual_scores_from_qk(
            q[:P], keys_p, self.rater_rows, v0, vn, causal_from=v0 + vn))
        sp[:, self.nl(li).to(dev)] = float("-inf")

        def agree():
            top = sp.topk(self.k_keep, dim=-1).indices
            masks = torch.zeros(P, vn, dtype=torch.bool, device=dev)
            masks.scatter_(1, top, True)
            return masks, (mean_pairwise_jaccard(masks) if P >= 2 else 1.0)

        masks, sim = T("jaccard", agree)

        if sim > self.thr:
            def vote():
                votes = masks.sum(0).float()
                big = 10.0 * (float(sp.max()) + 1.0)
                return (votes * big + sp.mean(0)).topk(self.k_keep).indices
            sel = T("jaccard", vote)
            cids = chunks_for_tokens(sel.tolist() + self.nl(li).tolist(),
                                     vn, m["chunk_size"])
            for kind in ("k", "v"):
                rows, vals = T("chunk_io", lambda k=kind:
                               ctx.reader.read_chunks(li, k, cids, self.io))
                T("scatter", lambda k=kind, r=rows, v=vals:
                  ctx.cache.write(li, k, r, v))
            keep = self._keep_mask(li, sel)
            self.log.append({"layer": li, "mode": "probe", "sim": sim,
                             "chunks": len(cids)})
        else:
            def fb():
                self._load_layer_full(li)
                kv = ctx.cache.k[li][0, :, v0:]
                keys_a = torch.cat([sys_k, kv, kn], dim=1)
                sa = sv.rater_visual_scores_from_qk(
                    q, keys_a, self.rater_rows, v0, vn, causal_from=v0 + vn)
                sa[:, self.nl(li).to(dev)] = float("-inf")
                return sa.topk(self.k_keep, dim=-1).indices
            top_h = T("fallback", fb)
            keep = self._keep_mask(li, top_h, n_heads=H)
            self.log.append({"layer": li, "mode": "fallback", "sim": sim,
                             "chunks": m["n_chunks_per_layer"]})

        T("scatter", lambda: BIAS.__setitem__(li, bias_from_keep(keep, m, dev)))


@torch.no_grad()
def profile_request(srv, ctx, question):
    BIAS.clear()
    ctx.reader.drop_all()
    from mmimpress.store import IOCounter
    counter = IOCounter()
    suffix_ids = suffix_ids_for(srv.runner, question)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    rr = srv.raters(ctx, suffix_ids)
    torch.cuda.synchronize()
    t_rater = time.perf_counter() - t0
    sel = ProfiledSelector(srv.runner, ctx, rr, srv.ratio, srv.probe,
                           srv.alpha, mode="impress", counter=counter)
    t1 = time.perf_counter()
    cache = ctx.cache.new_request()
    torch.cuda.synchronize()
    t_cache = time.perf_counter() - t1
    with sel:
        srv._decode(cache, suffix_ids, ctx.meta["prefix_len"])
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    BIAS.clear()
    del cache
    out = {"total_ms": total * 1e3, "rater_ms": t_rater * 1e3,
           "cache_alloc_ms": t_cache * 1e3,
           "hook_ms": sel.hook_seconds * 1e3,
           "disk_ms": counter.summary()["ms"],
           "fallback_rate": sel.stats()["fallback_rate"]}
    for k, v in sel.T.t.items():
        out[f"stage_{k}_ms"] = v * 1e3
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=3)
    ap.add_argument("--skip", type=int, default=4)
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--ratio", type=float, default=RETENTION_RATIO)
    args = ap.parse_args()

    runner = LlavaRunner().load()
    srv = Server(runner, ratio=args.ratio)
    rows = []
    for e in load_index()[:args.limit]:
        d = STORE_DIR / str(e["image_id"])
        if not (d / "meta.json").exists():
            continue
        ctx = ImageContext(d, runner.model.device)
        for q in e["questions"][args.skip:args.skip + args.questions]:
            rows.append(profile_request(srv, ctx, q["question"]))
        ctx.close()
        del ctx
        torch.cuda.empty_cache()

    keys = sorted({k for r in rows for k in r})
    agg = {k: float(np.mean([r.get(k, 0.0) for r in rows])) for k in keys}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "profile_sparsevlm.json", "w") as f:
        json.dump({"n": len(rows), "mean": agg, "rows": rows}, f, indent=1)

    print(f"\n=== SparseVLM selector profile (n={len(rows)} requests) ===")
    print(f"  total request        {agg['total_ms']:8.1f} ms")
    print(f"    rater selection    {agg['rater_ms']:8.1f} ms")
    print(f"    cache alloc        {agg['cache_alloc_ms']:8.1f} ms")
    print(f"    per-layer hooks    {agg['hook_ms']:8.1f} ms")
    for k in keys:
        if k.startswith("stage_"):
            print(f"      {k[6:-3]:<16} {agg[k]:8.1f} ms")
    print(f"    (disk inside hooks {agg['disk_ms']:8.1f} ms)")
    print(f"  fallback rate        {agg['fallback_rate']*100:8.1f} %")
    comp = agg["hook_ms"] - agg["disk_ms"]
    print(f"\n  ONLINE SELECTOR COMPUTE (hooks minus disk) = {comp:.1f} ms")


if __name__ == "__main__":
    main()

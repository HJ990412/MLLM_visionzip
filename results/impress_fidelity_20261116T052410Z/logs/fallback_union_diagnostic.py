"""Diagnostic: how much of a fallback layer does the union of the per-head
selections cover?  Taps LayerSelector._keep_mask; serving code is unchanged.
Not a timing run (warm cache)."""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path("/home/dblab/hj/mllm_v2"); sys.path.insert(0, str(ROOT))
import mmimpress.serve as serve
from mmimpress.model import LlavaRunner
from mmimpress.dataset import load_index
from mmimpress.store import chunks_for_tokens

store, metric, n_images, n_q, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
records = []
orig = serve.LayerSelector._keep_mask
def tap(self, layer, sel_stored, n_heads=None):
    keep = orig(self, layer, sel_stored, n_heads)
    if n_heads:                                   # fallback layers only
        m = self.ctx.meta
        union = keep.any(dim=0)
        cs, vn = m["chunk_size"], m["v_token_num"]
        cids = chunks_for_tokens(union.nonzero(as_tuple=True)[0].tolist(), vn, cs)
        per_chunk = [int(union[c * cs:(c + 1) * cs].sum()) for c in range(m["n_chunks_per_layer"])]
        records.append({"layer": layer,
                        "union_token_fraction": float(union.float().mean()),
                        "per_head_token_fraction": float(keep.float().mean()),
                        "union_chunk_fraction": len(cids) / m["n_chunks_per_layer"],
                        "min_union_tokens_in_a_chunk": min(per_chunk),
                        "chunks_with_no_union_token": sum(p == 0 for p in per_chunk)})
    return keep
serve.LayerSelector._keep_mask = tap

runner = LlavaRunner().load()
srv = serve.Server(runner, importance_metric=metric, fallback_v="selected")
n_req = 0
for e in load_index(None)[:n_images]:
    ctx = serve.ImageContext(ROOT / store / str(e["image_id"]), runner.model.device,
                             drop_cache=False, require_v_hidden=metric != "h2o")
    for q in e["questions"][4:4 + n_q]:
        srv.request(ctx, q["question"], mode="impress", cold=False); n_req += 1
    ctx.close(); del ctx; torch.cuda.empty_cache()
def stat(key):
    v = np.asarray([r[key] for r in records], dtype=float)
    return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
summary = {"store": store, "importance_metric": metric, "requests": n_req,
           "fallback_layers_observed": len(records),
           **{k: stat(k) for k in ("union_token_fraction", "per_head_token_fraction",
                                   "union_chunk_fraction", "min_union_tokens_in_a_chunk",
                                   "chunks_with_no_union_token")},
           "fallback_layer_histogram": {str(l): c for l, c in sorted(
               zip(*np.unique([r["layer"] for r in records], return_counts=True)))}}
summary["fallback_layer_histogram"] = {k: int(v) for k, v in summary["fallback_layer_histogram"].items()}
Path(out).write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))

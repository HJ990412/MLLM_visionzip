"""Real-request check: new serve.py with default options == original serve.py.

Loads the pre-change serve.py (scratchpad backup) next to the working-tree
module and serves the same cold requests through both.  Read-only on the store.
"""
import importlib.util, json, sys
from pathlib import Path
import torch

ROOT = Path("/home/dblab/hj/mllm_v2")
sys.path.insert(0, str(ROOT))
ORIG = Path(sys.argv[1])
N_IMAGES, Q0, Q1 = int(sys.argv[2]), 4, 4 + int(sys.argv[3])

import mmimpress.serve as new
spec = importlib.util.spec_from_file_location("serve_orig", ORIG)
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
from mmimpress.model import LlavaRunner
from mmimpress.dataset import load_index
from mmimpress.config import STORE_DIR

LEGACY_STATS = ("touched_chunk_fraction", "logical_kv_ratio", "fallback_rate",
                "probe_rate", "mean_jaccard", "threshold", "k_keep")
LEGACY_LOG = ("layer", "mode", "sim", "chunks")

def tap(mod):
    """Capture selector log and the frozen per-layer masks of each request."""
    captured = {}
    orig_decode = mod.Server._decode
    def _decode(self, cache, suffix_ids, prefix_len):
        out = orig_decode(self, cache, suffix_ids, prefix_len)
        captured["bias"] = {li: b.clone() for li, b in mod.BIAS.items()}
        return out
    mod.Server._decode = _decode
    orig_stats = mod.LayerSelector.stats
    def stats(self):
        captured["log"] = [dict(r) for r in self.log]
        return orig_stats(self)
    mod.LayerSelector.stats = stats
    return captured

cap_new, cap_old = tap(new), tap(old)
runner = LlavaRunner().load()
srv_new, srv_old = new.Server(runner), old.Server(runner)

def io_clean(io):
    return {"preads": io["preads"], "bytes": io["bytes"],
            "chunk_units": io["chunk_units"],
            "per_kind": {k: {a: b for a, b in v.items() if a != "seconds"}
                         for k, v in io["per_kind"].items()}}

index = load_index(None)[:N_IMAGES]
mismatch, n = [], 0
for e in index:
    d = STORE_DIR / str(e["image_id"])
    ctx = new.ImageContext(d, runner.model.device)
    for q in e["questions"][Q0:Q1]:
        for mode in ("impress", "fullload"):
            ro = srv_old.request(ctx, q["question"], mode=mode, cold=True)
            lo, bo = cap_old["log"], cap_old["bias"]
            rn = srv_new.request(ctx, q["question"], mode=mode, cold=True)
            ln, bn = cap_new["log"], cap_new["bias"]
            n += 1
            bad = []
            for key in ("answer", "first_token_id", "generated_token_ids",
                        "generated_tokens", "n_raters"):
                if ro[key] != rn[key]: bad.append(key)
            if io_clean(ro["io"]) != io_clean(rn["io"]): bad.append("io")
            for key in LEGACY_STATS:
                if ro[key] != rn[key]: bad.append("stats:" + key)
            if lo != [{k: r[k] for k in LEGACY_LOG} for r in ln]: bad.append("log")
            if sorted(bo) != sorted(bn) or any(
                    not torch.equal(bo[li], bn[li]) for li in bo):
                bad.append("mask")
            if bad:
                mismatch.append((str(e["image_id"]), q["question_id"], mode, bad))
            print(e["image_id"], q["question_id"], mode, "OK" if not bad else bad,
                  f"fb={rn['fallback_rate']:.3f} mb={rn['io']['mb']:.1f} "
                  f"layers_masked={len(bn)}", flush=True)
    ctx.close(); del ctx; torch.cuda.empty_cache()
print(json.dumps({"requests_compared": n, "mismatches": len(mismatch),
                  "detail": mismatch}))

"""Real-request check for the prefetch option.

  old   = serve.py as it was before this change (scratchpad backup)
  none  = working tree, prefetch="none"
  next  = working tree, prefetch="next_layer"

All three serve the same cold requests on the same store with the same
importance metric / fallback V policy.  Everything except wall-clock time must
be equal: tokens, bytes, pread counts, selector log, per-layer masks.
Read-only on the store.
"""
import importlib.util, json, sys
from pathlib import Path
import torch

ROOT = Path("/home/dblab/hj/mllm_v2"); sys.path.insert(0, str(ROOT))
ORIG, STORE, METRIC, FALLBACK_V = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
N_IMAGES, Q0, Q1 = int(sys.argv[5]), 4, 4 + int(sys.argv[6])

import mmimpress.serve as new
spec = importlib.util.spec_from_file_location("serve_before_prefetch", ORIG)
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
assert not hasattr(old, "PREFETCH_POLICIES") and hasattr(new, "PREFETCH_POLICIES")
from mmimpress.model import LlavaRunner
from mmimpress.dataset import load_index

def tap(mod):
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
        captured["probe_log"] = [dict(r) for r in getattr(self, "probe_log", [])]
        return orig_stats(self)
    mod.LayerSelector.stats = stats
    return captured

cap_new, cap_old = tap(new), tap(old)
runner = LlavaRunner().load()
kw = dict(importance_metric=METRIC, fallback_v=FALLBACK_V)
srv_old, srv_new = old.Server(runner, **kw), new.Server(runner, **kw)
OLD_STATS = None

def io_clean(io):
    return {"preads": io["preads"], "bytes": io["bytes"], "chunk_units": io["chunk_units"],
            "per_kind": {k: {a: b for a, b in v.items() if a != "seconds"} for k, v in io["per_kind"].items()},
            "kind_order": list(io["per_kind"])}

def compare(a, la, ba, b, lb, bb, stat_keys):
    bad = []
    for key in ("answer", "first_token_id", "generated_token_ids", "generated_tokens", "n_raters", "n_score_rows"):
        if a[key] != b[key]: bad.append(key)
    if io_clean(a["io"]) != io_clean(b["io"]): bad.append("io")
    for key in stat_keys:
        if a[key] != b[key]: bad.append("stats:" + key)
    if la != lb: bad.append("log")
    if sorted(ba) != sorted(bb) or any(not torch.equal(ba[li], bb[li]) for li in ba): bad.append("mask")
    return bad

TIMING = {"hook_ms", "prefill_ms", "ttft", "decode_latency", "e2e_latency", "io", "core_started_at_s",
          "first_token_at_s", "model_finished_at_s", "postprocess_finished_at_s"}
NEW_KEYS = {"prefetch", "probe_read_ms", "probe_pread_ms", "prefetch_wait_ms", "probe_hook_blocked_ms",
            "prefetched_probe_reads", "prefetch_ready_on_arrival"}
mismatch, n = [], 0
sources = {"none": set(), "next": set()}
for e in load_index(None)[:N_IMAGES]:
    ctx = new.ImageContext(ROOT / STORE / str(e["image_id"]), runner.model.device,
                           require_v_hidden=METRIC != "h2o")
    for q in e["questions"][Q0:Q1]:
        ro = srv_old.request(ctx, q["question"], mode="impress", cold=True); lo, bo = cap_old["log"], cap_old["bias"]
        rn = srv_new.request(ctx, q["question"], mode="impress", cold=True, prefetch="none")
        ln, bn, pn = cap_new["log"], cap_new["bias"], cap_new["probe_log"]
        rp = srv_new.request(ctx, q["question"], mode="impress", cold=True, prefetch="next_layer")
        lp, bp, pp = cap_new["log"], cap_new["bias"], cap_new["probe_log"]
        fo = srv_old.request(ctx, q["question"], mode="fullload", cold=True)
        fn = srv_new.request(ctx, q["question"], mode="fullload", cold=True, prefetch="next_layer")
        n += 1
        old_keys = [k for k in ro if k not in TIMING]
        assert set(rn) - set(ro) == NEW_KEYS, set(rn) ^ set(ro)
        bad = {"old_vs_none": compare(ro, lo, bo, rn, ln, bn, [k for k in old_keys if k not in ("answer",)]),
               "none_vs_next": compare(rn, ln, bn, rp, lp, bp, [k for k in old_keys if k not in ("answer",)]),
               "fullload_old_vs_new": [k for k in ("answer", "generated_token_ids") if fo[k] != fn[k]]
                                      + ([] if io_clean(fo["io"]) == io_clean(fn["io"]) else ["io"])}
        sources["none"].add(tuple(r["source"] for r in pn)); sources["next"].add(tuple(r["source"] for r in pp))
        if any(bad.values()): mismatch.append((str(e["image_id"]), q["question_id"], bad))
        print(e["image_id"], q["question_id"], "OK" if not any(bad.values()) else bad,
              f"fb={rn['fallback_rate']:.3f} mb={rn['io']['mb']:.1f} preads={rn['io']['preads']} "
              f"ttft none={rn['ttft']*1e3:.0f} next={rp['ttft']*1e3:.0f} "
              f"probe_read none={rn['probe_read_ms']:.1f} next={rp['probe_read_ms']:.1f} "
              f"wait={rp['prefetch_wait_ms']:.1f} ready={rp['prefetch_ready_on_arrival']}/{rp['prefetched_probe_reads']} "
              f"fullload_prefetch={fn['prefetch']}", flush=True)
    ctx.close(); del ctx; torch.cuda.empty_cache()
srv_new.close()
L = 32
print(json.dumps({"store": STORE, "importance_metric": METRIC, "fallback_v": FALLBACK_V,
                  "requests_compared": n, "mismatches": len(mismatch), "detail": mismatch,
                  "probe_sources_none_all_hook": sources["none"] == {("hook",) * L},
                  "probe_sources_next_layer0_hook_rest_prefetch": sources["next"] == {("hook",) + ("prefetch",) * (L - 1)}}))

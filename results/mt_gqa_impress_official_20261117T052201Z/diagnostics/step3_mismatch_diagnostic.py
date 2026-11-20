"""Diagnostic for the one step-3 mismatch: same bytes, different key order?"""
import importlib.util, json, sys, glob, shutil, uuid
from pathlib import Path
import numpy as np, torch
ROOT = Path("/home/dblab/hj/mllm_v2"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("_h", ROOT / "scripts/190_eval_mt_gqa_impress_official.py")
H = importlib.util.module_from_spec(spec); sys.modules["_h"] = H; spec.loader.exec_module(H)
run = Path(sys.argv[1]); out = Path(sys.argv[2]); image, dialog_id, turn = "n181355", "mtgqa_000018", 2
config, manifest = H.verify_freeze(run)
rows = {}
for f in glob.glob(str(run / "gate/images" / image / "attempt_*/raw.jsonl")):
    for line in open(f):
        r = json.loads(line); rows[(r["method_id"], r["dialog_id"], r["turn_id"])] = r
torch.set_num_threads(2); torch.manual_seed(1234); torch.set_grad_enabled(False)
runner = H.V.load_runner()
from mmimpress.serve import Server
from mmimpress import impress_official as io_
server = Server(runner, ratio=0.25, probe=3, alpha=0.6, max_new_tokens=16); server._prefetch_pool()
ds = [d for d in manifest["dialogues"] if d["image_id"] == image]
first = ds[0]; path = ROOT / first["image_path"]
t1 = next(r for (m, d, t), r in rows.items() if d == first["dialog_id"] and t == 1 and m == "recompute")
scratch = out / "stores"; shutil.rmtree(scratch, ignore_errors=True); scratch.mkdir(parents=True)
ctxs = {}
for kind in ("raster", H.NEW_KIND):
    factory = lambda: (t1["prompt"], "", [])
    if kind == H.NEW_KIND:
        result, diag = H.turn1_request_with_stats(runner, server, path, factory)
    else:
        result, diag = H.normal_request(runner, server, path, factory, "raster")
    src = {"dialog_id": first["dialog_id"], "method_id": "diag", "physical_execution_id": str(uuid.uuid4()), "logical_request_id": "diag"}
    H.persist(runner, server, result, diag, scratch / kind, image, kind, src, config["layer_budgets"], out / f"{kind}_stats.npz")
    del result, diag
    ctxs[kind] = H.activate(runner, kind, scratch / kind)
req = rows[("fullload_async", dialog_id, turn)]
suffix = torch.tensor(req["suffix_input_ids"])
tok = runner.processor.tokenizer
logits = []
handle = runner.model.register_forward_hook(lambda m, a, o: logits.append(o.logits[0, -1].float().cpu()))
def first_logits(fn):
    logits.clear(); res = fn(); return logits[0].clone(), res
res = {}
lf, r = first_logits(lambda: server.request(ctxs["raster"], mode="fullload", suffix_ids=suffix, cold=True, prompt_text=req["prompt"], cache_id=image, prefetch="layer_stream"))
kf = [k.clone() for k in ctxs["raster"].cache.k]; vf = [v.clone() for v in ctxs["raster"].cache.v]
res["fullload_raster"] = (lf, r["generated_token_ids"])
lr, r = first_logits(lambda: io_.request_official(server, ctxs[H.NEW_KIND], suffix, ratio=1.0, prefetch="next_layer", cold=True))
kr = [k.clone() for k in ctxs[H.NEW_KIND].cache.k]; vr = [v.clone() for v in ctxs[H.NEW_KIND].cache.v]
res["official_r100"] = (lr, r["generated_token_ids"])
# FullLoad's own code on the reordered store (whole K/V in stored order, no selector logic of the new path)
from mmimpress.serve import LayerSelector, BIAS
def fullload_on_reordered():
    ctx = ctxs[H.NEW_KIND]; BIAS.clear(); ctx.reader.drop_all()
    sel = LayerSelector(runner, ctx, torch.empty(0, dtype=torch.long, device=runner.model.device), mode="fullload")
    cache = ctx.cache.new_request()
    with sel:
        ans, first_id, timing = server._decode(cache, suffix.to(runner.model.device), ctx.meta["prefix_len"])
    return {"generated_token_ids": timing["generated_token_ids"]}
lo, r = first_logits(fullload_on_reordered)
res["fullload_code_on_reordered_store"] = (lo, r["generated_token_ids"])
handle.remove()
meta = ctxs[H.NEW_KIND].meta; v0 = meta["v_token_start"]
same_k = same_v = True
for li in range(meta["num_layers"]):
    order = torch.tensor(meta["order"][li], device=kf[li].device)
    same_k &= bool(torch.equal(kr[li][0, :, v0:v0 + len(order)], kf[li][0, :, v0:v0 + len(order)][:, order]))
    same_v &= bool(torch.equal(vr[li][0, :, v0:v0 + len(order)], vf[li][0, :, v0:v0 + len(order)][:, order]))
    same_k &= bool(torch.equal(kr[li][0, :, :v0], kf[li][0, :, :v0])); same_v &= bool(torch.equal(vr[li][0, :, :v0], vf[li][0, :, :v0]))
report = {"image": image, "dialog": dialog_id, "turn": turn, "prefix_kv_equal_up_to_order": {"k": same_k, "v": same_v}, "paths": {}}
ids = {"Man": 2315, "Girl": 18620}
for name, (lg, toks) in res.items():
    top = torch.topk(lg, 3)
    report["paths"][name] = {"tokens": toks, "top3": [(tok.decode([int(i)]), int(i), float(v)) for v, i in zip(top.values, top.indices)],
                             "logit_Man": float(lg[ids["Man"]]), "logit_Girl": float(lg[ids["Girl"]]), "Man_minus_Girl": float(lg[ids["Man"]] - lg[ids["Girl"]]),
                             "max_abs_logit_diff_vs_fullload": float((lg - lf).abs().max())}
print(json.dumps(report, indent=1, ensure_ascii=False))
(out / "diag_r100.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
for c in ctxs.values(): c.close()
server.close(); shutil.rmtree(scratch, ignore_errors=True)

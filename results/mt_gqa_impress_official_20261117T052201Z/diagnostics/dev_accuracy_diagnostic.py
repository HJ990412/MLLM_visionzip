"""Diagnostic for the DEV accuracy of the new IMPRESS arm (not part of the frozen run).

For the first N DEV images (directory order, chosen before looking at results),
every follow-up request of the new arm recorded in the run is replayed twice:

  A. the new arm itself (``request_official``, 25%), keeping its per-layer
     keep masks;
  B. an independent reference on a RASTER store of the same image: FullLoad's
     own loader reads every K/V, and the additive mask of A's token set is
     applied.  While B runs, the model's own queries/keys of the probe heads
     are captured and the public selection logic is applied to the whole
     attention (stored prefix rows + suffix rows computed here from the
     model's q/k) -- without the new arm's probe sidecar / selector code.

Checks: A's tokens == recorded tokens; B's tokens == A's tokens; B's selection
== A's selection per layer.  Also describes what A selects.
"""
import glob, importlib.util, json, shutil, sys, uuid
from pathlib import Path
import numpy as np, torch
ROOT = Path("/home/dblab/hj/mllm_v2"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("_h", ROOT / "scripts/190_eval_mt_gqa_impress_official.py")
H = importlib.util.module_from_spec(spec); sys.modules["_h"] = H; spec.loader.exec_module(H)
run, out, n_images = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
config, manifest = H.verify_freeze(run)
torch.set_num_threads(2); torch.manual_seed(1234); torch.set_grad_enabled(False)
runner = H.V.load_runner()
import transformers.models.llama.modeling_llama as ml
from mmimpress.serve import Server, LayerSelector, BIAS, bias_from_keep
from mmimpress import impress_official as io_
server = Server(runner, ratio=0.25, probe=3, alpha=0.6, max_new_tokens=16); server._prefetch_pool()
images = sorted(p.name for p in (run / "dev/images").iterdir())[:n_images]
NEW, OLD = "impress_official_async", "impress_async"
BASE = 576                                     # LLaVA-NeXT: base-image tokens come first
summary = {"images": images, "requests": 0, "A_equals_record": 0, "B_equals_A": 0, "layers": 0, "layer_mode_equal": 0,
           "consensus_layers": 0, "consensus_set_equal": 0, "consensus_set_min_jaccard": 1.0, "mismatches": [],
           "first_logit_max_abs_diff_B_vs_A": 0.0}
per_layer = {}
selections = {}                                # (image, layer) -> list of raster bool masks (consensus only)
prefix_only = {}
acc = {"A_correct": 0, "old_correct_recorded": 0, "fullload_correct_recorded": 0}
for image in images:
    rows = {}
    for f in glob.glob(str(run / "dev/images" / image / "attempt_*/raw.jsonl")):
        for line in open(f):
            r = json.loads(line); rows[(r["method_id"], r["dialog_id"], r["turn_id"])] = r
    ds = [d for d in manifest["dialogues"] if d["image_id"] == image]
    first = ds[0]; path = ROOT / first["image_path"]
    t1 = rows[("recompute", first["dialog_id"], 1)]
    scratch = out / "stores" / image; shutil.rmtree(scratch, ignore_errors=True); scratch.mkdir(parents=True)
    ctxs = {}
    for kind in ("raster", H.NEW_KIND):
        factory = lambda: (t1["prompt"], "", [])
        if kind == H.NEW_KIND:
            result, diag = H.turn1_request_with_stats(runner, server, path, factory)
        else:
            result, diag = H.normal_request(runner, server, path, factory, "raster")
        src = {"dialog_id": first["dialog_id"], "method_id": "diag", "physical_execution_id": str(uuid.uuid4()), "logical_request_id": "diag"}
        H.persist(runner, server, result, diag, scratch / kind, image, kind, src, config["layer_budgets"], scratch / f"{kind}_stats.npz")
        del result, diag
        ctxs[kind] = H.activate(runner, kind, scratch / kind)
    mr, mn = ctxs["raster"].meta, ctxs[H.NEW_KIND].meta
    v0, vn, L = mr["v_token_start"], mr["v_token_num"], mr["num_layers"]
    seps = torch.tensor(mr["newline_idx"]); assert int(seps.min()) >= BASE
    cand = torch.ones(vn, dtype=torch.bool); cand[seps] = False
    k_keep = io_.keep_count(mr["n_spatial"], 0.25)
    prefix_rows = torch.from_numpy(np.load(scratch / f"{H.NEW_KIND}_stats.npz")["prefix_rows"])    # (L, 3, vn) raster
    orders = [torch.tensor(o) for o in mn["order"]]
    for li in range(L):                        # what the stored prefix rows alone would select
        imp = prefix_rows[li].clone(); imp[:, seps] = float("-inf")
        s = io_.select_tokens(imp, k_keep, 0.3)
        mask = torch.zeros(vn, dtype=torch.bool); mask[s["head_topk"].flatten()] = True
        prefix_only[(image, li)] = (s, mask)
    logits = []
    handle = runner.model.register_forward_hook(lambda m, a, o: logits.append(o.logits[0, -1].float().cpu()))
    for (method, dialog_id, turn), req in sorted(rows.items()):
        if method != NEW or turn == 1:
            continue
        suffix = torch.tensor(req["suffix_input_ids"])
        # ---- A: the new arm
        logits.clear(); holder = []
        ra = io_.request_official(server, ctxs[H.NEW_KIND], suffix, ratio=0.25, prefetch="next_layer", cold=True, selector_out=holder)
        la = logits[0].clone(); sel = holder[0]
        modes = {r["layer"]: r["mode"] for r in sel.log}
        keep_raster = {}
        for li in range(L):
            kr = torch.zeros(vn, dtype=torch.bool); kr[orders[li]] = sel.keep_masks[li]; keep_raster[li] = kr
        # ---- B: FullLoad loader on the raster store + A's token set as a mask + independent selection
        ctx = ctxs["raster"]; BIAS.clear(); ctx.reader.drop_all()
        captured = {}
        prev = ml.eager_attention_forward
        def capture(module, query, key, value, attention_mask, **kw):
            li = getattr(module, "layer_idx", None)
            if li is not None and li not in captured:
                q, k = query[0, :3].float(), key[0, :3].float()
                sc = (q @ k.transpose(1, 2)) * float(kw.get("scaling", mr["head_dim"] ** -0.5))
                if attention_mask is not None:
                    sc = sc + attention_mask[0, 0, :, :sc.shape[-1]].float()
                captured[li] = sc.softmax(dim=-1)[:, :, v0:v0 + vn].sum(dim=1).cpu()      # (3, vn) suffix rows, unmasked
            return prev(module, query, key, value, attention_mask, **kw)
        hooks = []
        def setter(li):
            def hook(module, args, kwargs):
                if modes[li] == "consensus" and li not in BIAS:
                    BIAS[li] = bias_from_keep(keep_raster[li], mr, runner.model.device)
            return hook
        for li, layer in enumerate(runner.layers):
            hooks.append(layer.register_forward_pre_hook(setter(li), with_kwargs=True))
        fl = LayerSelector(runner, ctx, torch.empty(0, dtype=torch.long, device=runner.model.device), mode="fullload")
        cache = ctx.cache.new_request()
        logits.clear(); ml.eager_attention_forward = capture
        try:
            with fl:
                _, _, timing = server._decode(cache, suffix.to(runner.model.device), mr["prefix_len"])
        finally:
            ml.eager_attention_forward = prev
            for h in hooks: h.remove()
        lb = logits[0].clone()
        assert sorted(BIAS) == sorted(li for li in range(L) if modes[li] == "consensus"), "reference mask was not applied"
        BIAS.clear(); del cache
        tb = timing["generated_token_ids"]
        summary["requests"] += 1
        summary["A_equals_record"] += int(ra["generated_token_ids"] == req["result"]["generated_token_ids"])
        same = ra["generated_token_ids"] == tb
        summary["B_equals_A"] += int(same)
        summary["first_logit_max_abs_diff_B_vs_A"] = max(summary["first_logit_max_abs_diff_B_vs_A"], float((la - lb).abs().max()))
        if not same:
            ta, tbv = torch.topk(la, 2), torch.topk(lb, 2)
            summary["mismatches"].append({"image": image, "dialog": dialog_id, "turn": turn, "A": ra["generated_token_ids"], "B": tb,
                                          "A_top2": [(int(i), float(v)) for v, i in zip(ta.values, ta.indices)],
                                          "B_top2": [(int(i), float(v)) for v, i in zip(tbv.values, tbv.indices)]})
        acc["A_correct"] += int(req["score"] == 1.0)
        acc["old_correct_recorded"] += int(rows[(OLD, dialog_id, turn)]["score"] == 1.0)
        acc["fullload_correct_recorded"] += int(rows[("fullload_async", dialog_id, turn)]["score"] == 1.0)
        for li in range(L):
            imp = prefix_rows[li] + captured[li]; imp[:, seps] = float("-inf")
            s = io_.select_tokens(imp, k_keep, 0.3)
            ref_mode = "fallback" if s["fallback"] else "consensus"
            d = per_layer.setdefault(li, {"n": 0, "mode_equal": 0, "consensus": 0, "set_equal": 0, "base_share": [], "prefix_only_overlap": [],
                                          "head_union_share": [], "sim_abs_diff": []})
            d["n"] += 1; summary["layers"] += 1
            eq = ref_mode == modes[li]; d["mode_equal"] += int(eq); summary["layer_mode_equal"] += int(eq)
            d["sim_abs_diff"].append(abs(s["similarity"] - next(r["sim"] for r in sel.log if r["layer"] == li)))
            if modes[li] == "consensus":
                chosen = keep_raster[li] & cand
                d["consensus"] += 1; summary["consensus_layers"] += 1
                d["base_share"].append(float(chosen[:BASE].sum()) / float(chosen.sum()))
                po, po_mask = prefix_only[(image, li)]
                if not po["fallback"]:
                    ref = torch.zeros(vn, dtype=torch.bool); ref[po["selected"]] = True
                    d["prefix_only_overlap"].append(float((ref & chosen).sum()) / k_keep)
                selections.setdefault((image, li), []).append(chosen)
                if ref_mode == "consensus":
                    ref = torch.zeros(vn, dtype=torch.bool); ref[s["selected"]] = True
                    j = float((ref & chosen).sum()) / float((ref | chosen).sum())
                    d["set_equal"] += int(j == 1.0); summary["consensus_set_equal"] += int(j == 1.0)
                    summary["consensus_set_min_jaccard"] = min(summary["consensus_set_min_jaccard"], j)
                    # the public vote order is undefined among equal counts: A's set is valid when it has k tokens
                    # and no unselected candidate has more votes (reference votes) than a selected one
                    votes = s["votes"]
                    valid = int(chosen.sum()) == k_keep and int(votes[chosen].min()) >= int(votes[cand & ~chosen].max())
                    d["vote_valid"] = d.get("vote_valid", 0) + int(valid)
                    summary["consensus_vote_valid"] = summary.get("consensus_vote_valid", 0) + int(valid)
                    d.setdefault("tie_share", []).append(float((chosen & (votes == int(votes[chosen].min()))).sum()) / k_keep)
            # per-head top-k sets: A (stored positions -> raster) against the reference
            top_a = orders[li][sel._diag[li][2].cpu()]
            for hi in range(3):
                a = torch.zeros(vn, dtype=torch.bool); a[top_a[hi]] = True
                b = torch.zeros(vn, dtype=torch.bool); b[s["head_topk"][hi]] = True
                same_head = bool((a == b).all())
                summary["head_sets"] = summary.get("head_sets", 0) + 1
                summary["head_sets_equal"] = summary.get("head_sets_equal", 0) + int(same_head)
                if not same_head:
                    summary["head_set_min_jaccard"] = min(summary.get("head_set_min_jaccard", 1.0), float((a & b).sum()) / float((a | b).sum()))
            # union of the three heads' top-k (what the fallback of the previous implementation would keep per head)
            um = torch.zeros(vn, dtype=torch.bool); um[s["head_topk"].flatten()] = True
            d["head_union_share"].append(float(um.sum()) / float(cand.sum()))
    handle.remove()
    for c in ctxs.values(): c.close()
    shutil.rmtree(scratch, ignore_errors=True)
    print(json.dumps({"image": image, **{k: v for k, v in summary.items() if k not in ("mismatches", "images")}}), flush=True)
# question-dependence: same image, same layer, different requests
cross = {}
for (image, li), masks in selections.items():
    if len(masks) < 2:
        continue
    m = torch.stack(masks).float(); inter = m @ m.T; size = m.sum(1)
    jac = inter / (size[:, None] + size[None, :] - inter)
    iu = torch.triu_indices(len(masks), len(masks), 1)
    cross.setdefault(li, []).extend(jac[iu[0], iu[1]].tolist())
mean = lambda x: (sum(x) / len(x)) if x else None
layers = []
for li in sorted(per_layer):
    d = per_layer[li]
    layers.append({"layer": li, "requests": d["n"], "mode_equal": d["mode_equal"], "consensus": d["consensus"], "consensus_set_equal": d["set_equal"], "consensus_vote_valid": d.get("vote_valid", 0),
                   "share_of_selection_at_lowest_vote_count": mean(d.get("tie_share", [])),
                   "similarity_max_abs_diff": max(d["sim_abs_diff"]), "selected_share_in_base_image": mean(d["base_share"]),
                   "overlap_with_prefix_rows_only_selection": mean(d["prefix_only_overlap"]),
                   "cross_request_jaccard_same_image": mean(cross.get(li, [])), "head_topk_union_share_of_image_tokens": mean(d["head_union_share"])})
allc = [x for v in cross.values() for x in v]
report = {"summary": summary, "recorded_accuracy_on_these_requests": {k: v / summary["requests"] for k, v in acc.items()},
          "base_image_share_of_image_tokens": None, "cross_request_jaccard_same_image_all_consensus_layers": mean(allc), "layers": layers}
(out / "dev_accuracy_diagnostic.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
print(json.dumps({k: v for k, v in report.items() if k != "layers"}, indent=1, ensure_ascii=False))
server.close()

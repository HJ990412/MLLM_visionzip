"""Diagnostic (not a measurement, not part of the repo): how the captured turn-1 visual-prefix K/V depends on the prompt."""
import importlib.util, json, sys, shutil
from pathlib import Path
import numpy as np, torch
ROOT = Path("/home/dblab/hj/mllm_v2"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("_h", ROOT / "scripts/202_eval_mt_gqa_ours_order.py"); H = importlib.util.module_from_spec(spec); sys.modules["_h"] = H; spec.loader.exec_module(H)
from mmimpress.piggyback import cache_layers
from mmimpress.packed_store import PACKED_NAME, packed_file_sha256
run = Path(sys.argv[1]); out = Path(sys.argv[2])
config = json.loads((run / "config.json").read_text()); manifest = json.loads((run / "manifest.json").read_text())
runner, servers = H.load_model(config)
ids = set(manifest["phase_dialogue_ids"]["gate"]); report = []
scratch = run / "scratch" / "diag"; assert not scratch.exists()
for image in manifest["phase_images"]["gate"]:
    ds = [d for d in manifest["dialogues"] if d["image_id"] == image and d["dialog_id"] in ids]
    qs = []
    for d in ds:
        for t in d["turns"]:
            q = t["question"].strip()
            if q not in qs: qs.append(q)
    qs = qs[:3] + [qs[0]]                       # the first question a second time: same prompt, new forward
    caps = []
    for i, q in enumerate(qs):
        prompt = "USER: <image>\n" + "\n".join([f"Current question Q1: {q}", f"{H.SHORT_ANSWER} ASSISTANT:"])
        result, diag = H.normal_request(runner, servers["base"], ROOT / ds[0]["image_path"], lambda: (prompt, "", []), "image_only")
        v0, vn = diag["v_token_start"], diag["v_token_num"]
        layers = cache_layers(result["captured_past_key_values"])
        kv = [(k[0, :, :v0 + vn].to(torch.float16).cpu(), v[0, :, :v0 + vn].to(torch.float16).cpu()) for k, v in layers]
        seq = int(layers[0][0].shape[2])
        shas = {}
        for kind in (H.PACKED_KIND, H.ORIG_KIND):   # both row orders, from the SAME capture
            src = {"dialog_id": ds[0]["dialog_id"], "method_id": H.ORIG_ASYNC, "physical_execution_id": f"diag-{i}", "logical_request_id": f"diag-{image}-{i}"}
            p = scratch / image / f"q{i}_{kind}"
            H.persist(runner, result, diag, p, image, kind, src, config["layer_budgets"])
            shas[kind] = packed_file_sha256(p / PACKED_NAME)[:12]
        shutil.rmtree(scratch / image)
        caps.append({"q": q, "seq": seq, "kv": kv, "sha": shas, "tokens": result["generated_token_ids"]})
        result.pop("captured_past_key_values", None); del result, diag, layers
    ref = caps[0]; rows = []
    for i, c in enumerate(caps[1:], 1):
        per_layer = []
        for li, ((k0, v0_), (k1, v1)) in enumerate(zip(ref["kv"], c["kv"])):
            for name, a, b in (("k", k0, k1), ("v", v0_, v1)):
                d = (a.float() - b.float()).abs()
                per_layer.append((li, name, int((a != b).sum()), a.numel(), float(d.max()), float(a.float().abs().mean())))
        diff = sum(x[2] for x in per_layer); total = sum(x[3] for x in per_layer)
        first = next((x[0] for x in per_layer if x[2]), None)
        rows.append({"vs": i, "question": c["q"], "sequence_len": c["seq"], "ref_sequence_len": ref["seq"], "elements_differ": diff, "elements": total,
                     "fraction": diff / total, "max_abs_diff": max(x[4] for x in per_layer), "mean_abs_value": float(np.mean([x[5] for x in per_layer])),
                     "first_layer_with_difference": first, "layer0_k_differ": per_layer[0][2], "layer0_v_differ": per_layer[1][2],
                     "per_layer_fraction": [round(sum(x[2] for x in per_layer if x[0] == li) / sum(x[3] for x in per_layer if x[0] == li), 5) for li in range(len(ref["kv"]))],
                     "per_layer_max_abs": [round(max(x[4] for x in per_layer if x[0] == li), 5) for li in range(len(ref["kv"]))],
                     "sha_reorder": c["sha"][H.PACKED_KIND], "sha_original": c["sha"][H.ORIG_KIND], "ref_sha_reorder": ref["sha"][H.PACKED_KIND], "ref_sha_original": ref["sha"][H.ORIG_KIND]})
        print(json.dumps({k: v for k, v in rows[-1].items() if k not in ("per_layer_fraction", "per_layer_max_abs")}), flush=True)
    report.append({"image_id": image, "questions": [c["q"] for c in caps], "sequence_lens": [c["seq"] for c in caps], "comparisons": rows})
    del caps
if scratch.exists(): shutil.rmtree(scratch)
out.write_text(json.dumps(report, indent=1))
for s in servers.values(): s.close()

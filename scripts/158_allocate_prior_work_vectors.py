#!/usr/bin/env python3
"""Phase 1 (v2) — prior-work layer-wise allocation vectors on CAL (contract A4).

One fresh full-prompt FullLoad prefill per CAL question (vision tower + the
whole multimodal prompt, no generation, nothing timed) collects, per decoder
layer,

* VL-Cache  : post-vision rows ``Q' = Q[m-tau:m]`` -> all causal keys,
              ThresholdFilter with p = 0.01 relative to the row maximum,
              sparsity gamma_l = mean over heads of #zeros / #causal entries
              (plus a visual-keys-only diagnostic column);
* SqueezeAttention : cos_l = mean over prompt tokens of
              cosine(h_in, h_in + attn_out) (residual-stream input vs the
              post-attention residual, before the MLP).

Vectors (sum = 8.0, clip [0.10, 0.60], v1 clip_redistribute):
VLC_offline  r ∝ 1 - gamma_l ; SA_offline  r ∝ 1 - cos_l ;
SA-2g_offline  exact 1-D 3-means on cos_l, highest-cosine cluster at the
clipped `percent`, the rest equal ; PYR = v1 F-front re-labelled ;
PYR-beta = PyramidKV beta=20 arithmetic schedule ; A1-2, U copied from v1.

Usage: --run RUN [--max-images N] [--aggregate-only]
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mmimpress.layer_budget import (budget_document, budget_sha256,
                                    clip_redistribute, front_schedule,
                                    load_budget_document, validate_ratios)
from mmimpress.model import LlavaRunner
from mmimpress.prefix25 import sha
import mmimpress.serve as serving

SEED = 1234
L = 32
P_THRESHOLD = 0.01            # VL-Cache Eq. 1 default ("p = 1%")
PYR_BETA = 20                 # PyramidKV code default
SA_PERCENT_RAW = 0.25 * (0.08 / 0.21)   # README example ratio KV_class3/ini_size (ASSUMPTION)
V1_RUN = ROOT / "runs/layer_budget_kv25_20261105T125754Z"
V1_RES = ROOT / "results/layer_budget_kv25_20261105T125754Z"
V1_CAL_MANIFEST_SHA = "8f50d65f3074356d3d8ab74574b346755bce209f68835f7fd4ecfdf346f8762f"
MODEL_REVISION = "c916e6cdcd760b4cecd1dd4907f84ac649f93b23"


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=1, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def append(path, obj):
    with Path(path).open("a") as f:
        f.write(json.dumps(obj, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def exclusive():
    p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True)
    other = [x for x in p.stdout.split() if x.strip() and int(x) != os.getpid()]
    if other:
        raise RuntimeError(f"external GPU work: {other}")


# ------------------------------------------------------------ capture hooks
class PriorWorkTrace:
    """Per-layer VL-Cache sparsity and SqueezeAttention cosine for one prefill."""

    def __init__(self, runner, vstart, vn, tau, m):
        self.runner, self.vstart, self.vn, self.tau, self.m = runner, vstart, vn, tau, m
        self.gamma_all = [None] * L
        self.gamma_visual = [None] * L
        self.cos = [None] * L
        self.calls = [0] * L
        self._h_in = {}
        self._handles = []

    def __enter__(self):
        self.original = serving._ORIG_EAGER
        trace = self

        def traced(module, query, key, value, attention_mask, **kw):
            out, weights = trace.original(module, query, key, value, attention_mask, **kw)
            li = int(module.layer_idx)
            trace.calls[li] += 1
            m, tau = trace.m, trace.tau
            assert weights.shape[-2] == m and weights.shape[-1] == m, weights.shape
            w = weights[0, :, m - tau:m, :].float()                     # (H, tau, m)
            rows = torch.arange(m - tau, m, device=w.device)
            valid = (torch.arange(m, device=w.device)[None, :] <= rows[:, None])   # (tau, m) causal
            w = w * valid[None]
            rowmax = w.amax(dim=-1, keepdim=True)
            kept = w >= P_THRESHOLD * rowmax
            zeros = (~kept) & valid[None]
            per_head = zeros.sum(dim=(1, 2)).double() / valid.sum().double()
            trace.gamma_all[li] = float(per_head.mean())
            vis = valid[None].clone()
            vis[:, :, :trace.vstart] = False
            vis[:, :, trace.vstart + trace.vn:] = False
            zv = (~kept) & vis
            trace.gamma_visual[li] = float((zv.sum(dim=(1, 2)).double() / vis[0].sum().double()).mean())
            return out, weights

        serving._ORIG_EAGER = traced
        for li, layer in enumerate(self.runner.layers):
            def pre(module, args, kwargs, li=li):
                h = args[0] if args else kwargs["hidden_states"]
                self._h_in[li] = h.detach()
            def post(module, args, output, li=li):
                attn = output[0] if isinstance(output, tuple) else output
                h = self._h_in.pop(li)
                assert h.shape == attn.shape, (h.shape, attn.shape)
                c = F.cosine_similarity(h.float(), (h + attn).float(), dim=-1)   # (1, m)
                assert c.shape[-1] == self.m
                self.cos[li] = float(c.mean())
            self._handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
            self._handles.append(layer.self_attn.register_forward_hook(post))
        return self

    def __exit__(self, *exc):
        serving._ORIG_EAGER = self.original
        for h in self._handles:
            h.remove()
        if exc[0] is None:
            assert self.calls == [1] * L, self.calls
            assert all(v is not None for v in self.gamma_all + self.cos)


# -------------------------------------------------------------- profiling
def profile(run, manifest, max_images):
    out = run / "phase1_prior"
    jsonl = out / "per_question.jsonl"
    done = set()
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            r = json.loads(line)
            done.add((r["image_id"], r["question_id"]))
    exclusive()
    torch.set_grad_enabled(False)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)
    runner = LlavaRunner().load()
    assert runner.cfg._commit_hash == MODEL_REVISION
    write(out / "environment.json", {"python": sys.version, "torch": torch.__version__, "revision": runner.cfg._commit_hash,
                                     "p_threshold": P_THRESHOLD, "attn": runner.model.config._attn_implementation})
    images = manifest["images"][:max_images] if max_images else manifest["images"]
    for ix, image in enumerate(images):
        pending = [q for q in image["questions"] if (image["image_id"], q["question_id"]) not in done]
        if not pending:
            continue
        t_img = time.perf_counter()
        assert sha(image["image_path"]) == image["image_sha256"]
        with Image.open(image["image_path"]) as im:
            pil = im.convert("RGB")
        for q in pending:
            enc = runner.encode(pil, q["question"])
            ids = enc["input_ids"][0]
            vstart, vn = runner.visual_span(ids)
            m = int(ids.shape[0]); tau = m - (vstart + vn)
            assert tau > 0
            _, _, _, sep = runner.anyres_layout(enc["image_sizes"][0], vn)
            n = vn - len(sep)
            dev_enc = runner.to_device(enc)
            with PriorWorkTrace(runner, vstart, vn, tau, m) as tr:
                runner.model(**dev_enc, use_cache=False)
            torch.cuda.synchronize()
            append(jsonl, {"image_id": image["image_id"], "question_id": q["question_id"], "question": q["question"],
                           "N": n, "S": len(sep), "V": vn, "m": m, "tau": tau, "v_token_start": vstart,
                           "gamma_all": tr.gamma_all, "gamma_visual": tr.gamma_visual, "cos": tr.cos})
            done.add((image["image_id"], q["question_id"]))
            del dev_enc
        torch.cuda.empty_cache()
        print(json.dumps({"image": ix + 1, "of": len(images), "image_id": image["image_id"], "seconds": round(time.perf_counter() - t_img, 2)}), flush=True)
    return jsonl


# ------------------------------------------------------------- allocation
def kmeans_1d_exact(values, k=3):
    """Globally optimal 1-D k-means (min within-cluster SSE) by enumerating
    contiguous partitions of the sorted values.  Deterministic; ties -> first.
    Returns labels in ascending-centre order (0 = lowest centre)."""
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    xs = [values[i] for i in order]
    n = len(xs)
    best = None
    for cuts in itertools.combinations(range(1, n), k - 1):
        bounds = (0,) + cuts + (n,)
        sse = 0.0
        for a, b in zip(bounds, bounds[1:]):
            seg = xs[a:b]; mu = sum(seg) / len(seg)
            sse += sum((v - mu) ** 2 for v in seg)
        if best is None or sse < best[0] - 1e-15:
            best = (sse, bounds)
    bounds = best[1]
    labels = [None] * n
    for c, (a, b) in enumerate(zip(bounds, bounds[1:])):
        for pos in range(a, b):
            labels[order[pos]] = c
    return labels, best[0]


def vectors_from_records(records):
    gamma = np.array([r["gamma_all"] for r in records], dtype=np.float64).mean(0)
    gamma_v = np.array([r["gamma_visual"] for r in records], dtype=np.float64).mean(0)
    cos = np.array([r["cos"] for r in records], dtype=np.float64).mean(0)
    vlc = clip_redistribute([float(1.0 - g) for g in gamma])
    sa = clip_redistribute([float(1.0 - c) for c in cos])
    labels, sse = kmeans_1d_exact([float(c) for c in cos], 3)
    c3 = [li for li in range(L) if labels[li] == 2]
    percent = max(0.10, SA_PERCENT_RAW)            # contract clip r_min
    a = (L * 0.25 - len(c3) * percent) / (L - len(c3))
    sa2g = validate_ratios([percent if li in c3 else a for li in range(L)])
    return {"VLC_offline": vlc, "SA_offline": sa, "SA-2g_offline": sa2g,
            "diag": {"gamma_all_mean": gamma.tolist(), "gamma_visual_mean": gamma_v.tolist(), "cos_mean": cos.tolist(),
                     "sa_labels": labels, "sa_class3_layers": c3, "sa_kmeans_sse": sse, "sa_percent_raw": SA_PERCENT_RAW,
                     "sa_percent_used": percent, "sa_other_ratio": a,
                     "vlc_density_mean": (1 - gamma).tolist(), "vlc_density_visual_mean": (1 - gamma_v).tolist()}}


def pyramid_beta(beta=PYR_BETA):
    lo = 0.25 / beta
    hi = 2 * 0.25 - lo
    step = (hi - lo) / (L - 1)
    raw = [hi - li * step for li in range(L)]
    assert abs(sum(raw) - 8.0) < 1e-9
    return clip_redistribute(raw), raw


def aggregate(run):
    out = run / "phase1_prior"
    p2 = run / "phase2"
    p2.mkdir(exist_ok=True)
    records = [json.loads(l) for l in (out / "per_question.jsonl").read_text().splitlines()]
    assert len(records) == 1250 and len({(r["image_id"], r["question_id"]) for r in records}) == 1250, len(records)
    fwd = vectors_from_records(records)
    rev = vectors_from_records(records[::-1])
    g8 = {a: float(np.max(np.abs(np.array(fwd[a]) - np.array(rev[a])))) for a in ("VLC_offline", "SA_offline", "SA-2g_offline")}
    g8_ok = all(v < 1e-6 for v in g8.values())
    write(run / "gates" / "g8_vector_reproducibility.json", {"status": "PASS" if g8_ok else "FAIL", "max_abs_delta": g8, "orders": ["forward", "reversed"], "n_records": len(records)})
    assert g8_ok, g8
    source_common = {"per_question_jsonl_sha256": sha(out / "per_question.jsonl"), "n_questions": len(records),
                     "n_images": len({r["image_id"] for r in records}), "cal_manifest_sha256": V1_CAL_MANIFEST_SHA}
    model = "llava-hf/llava-v1.6-vicuna-7b-hf"
    docs = {}
    docs["VLC_offline"] = budget_document("VLC_offline", fwd["VLC_offline"], model=model, source={
        **source_common, "kind": "VL-Cache Alg.1 offline (CAL mean)", "p_threshold": P_THRESHOLD, "rows": "post-vision Q[m-tau:m]", "keys": "all causal keys",
        "rule": "r ∝ 1 - gamma_l; v1 clip_redistribute [0.10, 0.60]", "gamma_mean": fwd["diag"]["gamma_all_mean"],
        "paper": "arXiv 2410.23317 Eq.1-2, Algorithm 1 (official code not found; see phase0/prior_work_sources.json)"})
    docs["SA_offline"] = budget_document("SA_offline", fwd["SA_offline"], model=model, source={
        **source_common, "kind": "SqueezeAttention cosine importance offline (CAL mean)", "cos_mean": fwd["diag"]["cos_mean"],
        "rule": "importance = 1 - cos(h_in, h_in + attn_out) over all prompt tokens; r ∝ importance; v1 clip", "code": "hetailang/SqueezeAttention a1933d18 utils_hh/modify_llama_drop.py L314"})
    docs["SA-2g_offline"] = budget_document("SA-2g_offline", fwd["SA-2g_offline"], model=model, source={
        **source_common, "kind": "SqueezeAttention 3-cluster group rule offline", "cos_mean": fwd["diag"]["cos_mean"], "labels": fwd["diag"]["sa_labels"],
        "class3_layers": fwd["diag"]["sa_class3_layers"], "percent_raw": SA_PERCENT_RAW, "percent_used": fwd["diag"]["sa_percent_used"],
        "other_ratio": fwd["diag"]["sa_other_ratio"], "kmeans": "exact 1-D 3-means (enumerated contiguous splits)",
        "assumption": "percent = 0.25 * (0.08/0.21) from README example; below r_min -> clipped to 0.10", "code": "hetailang/SqueezeAttention a1933d18 utils_hh/modify_llama_drop.py L527-543"})
    ff = load_budget_document(V1_RES / "layer_budget_llava_F-front.json")
    assert ff["layer_keep_ratios"] == front_schedule()
    docs["PYR"] = budget_document("PYR", ff["layer_keep_ratios"], model=model, source={
        "kind": "PyramidKV fixed-shape schedule = v1 F-front re-labelled (linear 0.40 -> 0.10, sum 8.0)", "v1_file": str(V1_RES / "layer_budget_llava_F-front.json"),
        "v1_file_sha256": sha(V1_RES / "layer_budget_llava_F-front.json"), "v1_ratio_sha256": ff["layer_budget_sha256"]})
    assert docs["PYR"]["layer_budget_sha256"] == ff["layer_budget_sha256"]
    pb, pb_raw = pyramid_beta()
    docs["PYR-beta"] = budget_document("PYR-beta", pb, model=model, source={
        "kind": "PyramidKV beta schedule (window 0, beta = 20): min = 0.25/beta, max = 0.5 - min, linear; then v1 clip", "beta": PYR_BETA, "raw_unclipped": pb_raw,
        "code": "Zefan-Cai/PyramidKV 68cd9551 pyramidkv/pyramidkv_utils.py L360-370"})
    for a in ("U", "A1-2"):
        src = V1_RES / f"layer_budget_llava_{a}.json"
        dst = p2 / f"layer_budget_llava_{a}.json"
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
        assert sha(dst) == sha(src)
        docs[a] = load_budget_document(dst)
    for a, d in docs.items():
        path = p2 / f"layer_budget_llava_{a}.json"
        if a in ("U", "A1-2"):
            continue
        if path.exists():
            old = load_budget_document(path)
            assert old["layer_keep_ratios"] == d["layer_keep_ratios"], f"frozen vector differs: {a}"
        else:
            with path.open("x") as f:
                json.dump(d, f, indent=1); f.write("\n")
    ids = ["U", "A1-2", "VLC_offline", "SA_offline", "SA-2g_offline", "PYR", "PYR-beta"]
    with (p2 / "vectors.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer"] + ids + ["gamma_all_mean", "gamma_visual_mean", "cos_mean", "sa_label"])
        for li in range(L):
            w.writerow([li] + [f"{docs[a]['layer_keep_ratios'][li]:.8f}" for a in ids] +
                       [f"{fwd['diag']['gamma_all_mean'][li]:.8f}", f"{fwd['diag']['gamma_visual_mean'][li]:.8f}", f"{fwd['diag']['cos_mean'][li]:.8f}", fwd["diag"]["sa_labels"][li]])
    summary = {"alloc_ids": ids, "sha256": {a: docs[a]["layer_budget_sha256"] for a in ids}, "file_sha256": {a: sha(p2 / f"layer_budget_llava_{a}.json") for a in ids},
               "r_min": 0.1, "r_max": 0.6, "p_threshold": P_THRESHOLD, "pyr_beta": PYR_BETA, "g8": g8, "diag": fwd["diag"],
               "clipped_layers": {a: [li for li in range(L) if abs(docs[a]["layer_keep_ratios"][li] - 0.6) < 1e-9 or abs(docs[a]["layer_keep_ratios"][li] - 0.1) < 1e-9] for a in ids},
               "front_mass_first_8_layers": {a: sum(docs[a]["layer_keep_ratios"][:8]) for a in ids},
               "tau_mean": float(np.mean([r["tau"] for r in records])), "m_mean": float(np.mean([r["m"] for r in records]))}
    write(p2 / "summary.json", summary)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.5))
        styles = {"U": ("k", "--"), "A1-2": ("#cc3311", "-"), "VLC_offline": ("#0077bb", "-"), "SA_offline": ("#009988", "-"),
                  "SA-2g_offline": ("#009988", ":"), "PYR": ("#ee7733", "-"), "PYR-beta": ("#ee7733", ":")}
        for a in ids:
            c, ls = styles[a]
            ax.plot(range(L), docs[a]["layer_keep_ratios"], color=c, ls=ls, lw=1.6, marker="o", ms=2.5, label=a)
        ax.set_xlabel("decoder layer"); ax.set_ylabel("r_l (kept fraction of visual rows)"); ax.set_ylim(0, 0.65)
        ax.axhline(0.1, color="grey", lw=0.6); ax.axhline(0.6, color="grey", lw=0.6)
        ax.set_title("Layer-wise allocation vectors (Σ r_l = 8.0, clip [0.10, 0.60])"); ax.legend(ncol=4, fontsize=8)
        fig.tight_layout(); fig.savefig(p2 / "fig_vectors.png", dpi=130); plt.close(fig)
    except Exception as exc:
        summary["figure_error"] = repr(exc); write(p2 / "summary.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    run = Path(args.run).resolve()
    assert run.parent == ROOT / "runs" and run.name.startswith("layer_budget_kv25_v2_")
    (run / "phase1_prior").mkdir(parents=True, exist_ok=True)
    src = V1_RUN / "phase1" / "cal_manifest.json"
    assert sha(src) == V1_CAL_MANIFEST_SHA
    dst = run / "phase1_prior" / "cal_manifest.json"
    if not dst.exists():
        dst.write_bytes(src.read_bytes())
    assert sha(dst) == V1_CAL_MANIFEST_SHA
    manifest = json.loads(dst.read_text())
    if not args.aggregate_only:
        try:
            profile(run, manifest, args.max_images)
        except Exception:
            append(run / "failures.jsonl", {"phase": "phase1_prior", "error": traceback.format_exc(), "utc": time.time()})
            raise
    summary = aggregate(run)
    print(json.dumps({k: summary[k] for k in ("sha256", "g8", "clipped_layers", "front_mass_first_8_layers", "tau_mean")}, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 1 — layer sensitivity profiling on CAL (contract A4).

One normal full-pixel Turn-1 inference per CAL image gives the captured visual
KV and the image-only VisionZip permutation.  The full visual KV stays in GPU
memory in stored (ranked) physical order with the SSD path's FP16 round trip.
Per question every variant is one suffix prefill over that cache with the
production ``bias_from_keep`` masks, then ``DynamicCache.crop``.  Nothing is
read from SSD and nothing is timed for the paper; wall times are only used by
the 10-image probe that fixes the per-image question count.

Usage: --run RUN [--max-images N] [--questions-per-image 5] [--aggregate-only]
"""
from __future__ import annotations

import argparse
import collections
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

from mmimpress.cvpr25 import anyres_token_scores, visionzip_repack_order
from mmimpress.dataset import _norm, exact_score, vqa_score
from mmimpress.layer_budget import total_row_budget
from mmimpress.model import LlavaRunner, cache_layers
from mmimpress.piggyback import VisionForwardCapture
from mmimpress.prefix25 import sha
from mmimpress.serve import BIAS, Server, bias_from_keep, suffix_ids_for
import mmimpress.serve as serving

CAL_INDEX = ROOT / "data/vqav2/index.json"
CAL_INDEX_SHA = "b83d5fa288fcb722ca073e261d3fec9086629ed0db2e568a2d5d6a24ef1589d7"
SEED = 1234
NUM_LAYERS = 32
GRID = 64


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


def gold_answer(answers):
    counts = collections.Counter(_norm(a) for a in answers)
    best = max(counts.values())
    for a in answers:                       # ties: first occurrence
        if counts[_norm(a)] == best:
            return a
    raise ValueError("no answers")


def freeze_manifest(run, questions_per_image):
    path = run / "phase1" / "cal_manifest.json"
    assert sha(CAL_INDEX) == CAL_INDEX_SHA, "CAL index changed"
    index = json.loads(CAL_INDEX.read_text())
    images = []
    for entry in index:
        qs = entry["questions"][:questions_per_image]
        images.append({"image_id": str(entry["image_id"]),
                       "image_path": str(ROOT / entry["image_path"]),
                       "image_sha256": sha(ROOT / entry["image_path"]),
                       "questions": [{"question_id": str(q["question_id"]), "question": q["question"],
                                      "answers": list(q["answers"]), "gold": gold_answer(q["answers"])}
                                     for q in qs]})
    doc = {"source": str(CAL_INDEX), "source_sha256": CAL_INDEX_SHA, "images": images,
           "questions_per_image": questions_per_image,
           "n_images": len(images), "n_questions": sum(len(x["questions"]) for x in images),
           "gold_policy": "most frequent normalized VQAv2 answer; ties -> first occurrence"}
    if path.exists():
        old = json.loads(path.read_text())
        assert old["images"] == images and old["questions_per_image"] == questions_per_image, \
            "frozen CAL manifest differs from the requested configuration"
        return old
    write(path, doc)
    return doc


# ------------------------------------------------------------- GPU helpers
def fp16_roundtrip(x, device):
    return x.detach().to(dtype=torch.float16, device="cpu").to(dtype=torch.bfloat16, device=device)


def build_memory_cache(runner, layers, vstart, vn, order, prefix_len):
    """Full-size prefix cache in stored physical order (ranking + separator tail)."""
    from transformers import DynamicCache
    device = runner.model.device
    h, hd = int(layers[0][0].shape[1]), int(layers[0][0].shape[3])
    perm = torch.as_tensor(order, device=layers[0][0].device)
    cache = DynamicCache()
    scratch = torch.zeros((1, h, prefix_len, hd), dtype=torch.bfloat16, device=device)
    for li, (sk, sv) in enumerate(layers):
        cache.update(scratch, scratch, li)
        for name, src in (("keys", sk), ("values", sv)):
            tgt = getattr(cache.layers[li], name)
            tgt[0, :, :vstart] = fp16_roundtrip(src[0, :, :vstart], device)
            tgt[0, :, vstart:prefix_len] = fp16_roundtrip(src[0, :, vstart + perm], device)
    del scratch
    return cache


def install_masks(meta, k_per_layer, n, device):
    """Production per-layer masks + independent visible-count assertion."""
    vn, vstart, p = meta["v_token_num"], meta["v_token_start"], meta["prefix_len"]
    BIAS.clear()
    for li, k in enumerate(k_per_layer):
        keep = torch.zeros(vn, dtype=torch.bool)
        keep[:k] = True
        keep[n:] = True
        BIAS[li] = bias_from_keep(keep, meta, device)
        zeros = (BIAS[li][0, 0, 0] == 0)
        assert int(zeros[:vstart].sum()) == vstart                       # system rows visible
        assert int(zeros[vstart:vstart + n].sum()) == k                 # exactly k content rows
        assert bool(zeros[vstart:vstart + k].all())                      # the first k
        assert bool(zeros[vstart + n:p].all())                           # structural rows
        assert not bool(zeros[vstart + k:vstart + n].any())              # nothing else
    assert len(BIAS) == len(k_per_layer)


class MassTrace:
    """Accumulate text-query -> visual-row attention mass per layer (FULL only)."""

    def __init__(self, vstart, vn, layers):
        self.vstart, self.vn, self.layers = vstart, vn, layers
        self.mass = None
        self.decode_seen = set()
        self.prefill_seen = set()

    def __enter__(self):
        self.original = serving._ORIG_EAGER
        self.mass = torch.zeros(self.layers, self.vn, dtype=torch.float32)

        def traced(module, query, key, value, attention_mask, **kw):
            out, weights = self.original(module, query, key, value, attention_mask, **kw)
            li = int(module.layer_idx)
            qlen = int(query.shape[-2])
            take = False
            if qlen > 1 and li not in self.prefill_seen:
                self.prefill_seen.add(li)
                take = True
            elif qlen == 1 and li not in self.decode_seen:
                self.decode_seen.add(li)
                take = True
            if take:
                w = weights[0, :, :, self.vstart:self.vstart + self.vn].float().sum(dim=(0, 1))
                self.mass[li] += w.cpu()
            return out, weights

        serving._ORIG_EAGER = traced
        return self

    def __exit__(self, *exc):
        serving._ORIG_EAGER = self.original
        assert self.prefill_seen == set(range(self.layers)) and self.decode_seen == set(range(self.layers))


@torch.no_grad()
def first_logits(model, cache, suffix, prefix_len):
    dev = model.device
    n = int(suffix.shape[0])
    pos = torch.arange(prefix_len, prefix_len + n, device=dev)
    out = model(input_ids=suffix.unsqueeze(0), attention_mask=torch.ones(1, prefix_len + n, dtype=torch.long, device=dev),
                position_ids=pos.unsqueeze(0), cache_position=pos, past_key_values=cache, use_cache=True)
    logits = out.logits[0, -1].detach().float()
    cache.crop(prefix_len)
    assert int(cache.get_seq_length()) == prefix_len
    return logits


def full_decode(server, cache, suffix, prefix_len):
    box = []
    handle = server.runner.model.register_forward_hook(
        lambda m, a, o: box.append(o.logits[0, -1].detach().float()) if hasattr(o, "logits") else None)
    try:
        answer, first, timing = server._decode(cache, suffix, prefix_len)
    finally:
        handle.remove()
    cache.crop(prefix_len)
    assert int(first) == int(box[0].argmax())
    return answer, first, timing["generated_token_ids"], box[0]


def kl_div(p_log, q_log):
    return float((p_log.exp() * (p_log - q_log)).sum())


def coverage_grid(mass_row, n):
    """Cumulative real-row mass share at k = 0, 64, ..., and n."""
    real = mass_row[:n]
    total = float(real.sum())
    cum = torch.cumsum(real, 0)
    ks = list(range(0, n, GRID)) + [n]
    vals = [0.0] + [float(cum[k - 1]) / total if total > 0 else 0.0 for k in ks[1:]]
    return ks, vals


# ---------------------------------------------------------------- main run
def profile(run, manifest, max_images, log):
    out = run / "phase1"
    jsonl = out / "per_question.jsonl"
    done = set()
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            r = json.loads(line)
            done.add((r["image_id"], r["question_id"]))
    exclusive()
    # Every forward here is inference-only; with autograd enabled the NF4
    # matmuls would retain dequantized weights and exhaust the GPU.
    torch.set_grad_enabled(False)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)
    runner = LlavaRunner().load()
    server = Server(runner, max_new_tokens=16)
    tok = runner.processor.tokenizer
    dev = runner.model.device
    env = {"python": sys.version, "torch": torch.__version__, "revision": runner.cfg._commit_hash,
           "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()}
    assert env["revision"] == "c916e6cdcd760b4cecd1dd4907f84ac649f93b23"
    write(out / "environment.json", env)
    images = manifest["images"][:max_images] if max_images else manifest["images"]
    k_u_of = lambda n: (n + 3) // 4
    # warm-up (excluded, 1 normal request on the first image)
    with Image.open(images[0]["image_path"]) as im:
        pil = im.convert("RGB")
    enc = runner.encode(pil, images[0]["questions"][0]["question"])
    server.recompute(runner.to_device(enc))
    torch.cuda.synchronize()
    for ix, image in enumerate(images):
        pending = [q for q in image["questions"] if (image["image_id"], q["question_id"]) not in done]
        if not pending:
            continue
        exclusive()
        t_img = time.perf_counter()
        with Image.open(image["image_path"]) as im:
            pil = im.convert("RGB")
        assert sha(image["image_path"]) == image["image_sha256"]
        cap = VisionForwardCapture(runner, capture_saliency=True)
        with cap:
            enc = runner.encode(pil, pending[0]["question"])
            t1 = server.recompute(runner.to_device(enc), return_past_key_values=True)
        assert cap.call_count == 1
        ids = enc["input_ids"][0]
        vstart, vn = runner.visual_span(ids)
        prefix_len = vstart + vn
        base, hi_h, hi_w, sep = runner.anyres_layout(enc["image_sizes"][0], vn)
        n = vn - len(sep)
        scores = anyres_token_scores(runner, cap.result_cpu(), enc["image_sizes"][0], vn, base_side=base)
        order = visionzip_repack_order(scores, sep)
        assert order[n:] == list(sep) and sorted(order) == list(range(vn))
        layers = cache_layers(t1.pop("captured_past_key_values"))
        cache = build_memory_cache(runner, layers, vstart, vn, order, prefix_len)
        del layers, t1
        torch.cuda.empty_cache()
        meta = {"prefix_len": prefix_len, "v_token_start": vstart, "v_token_num": vn}
        k_u = k_u_of(n)
        t1_ms = (time.perf_counter() - t_img) * 1e3
        for q in pending:
            t_q = time.perf_counter()
            suffix = suffix_ids_for(runner, q["question"]).to(dev)
            gold = q["gold"]
            gold_ids = sorted({int(tok(g, add_special_tokens=False).input_ids[0]) for g in (gold, gold.capitalize(), gold.lower())})
            row = {"image_id": image["image_id"], "question_id": q["question_id"], "question": q["question"],
                   "gold": gold, "answers": q["answers"], "N": n, "S": len(sep), "V": vn, "prefix_len": prefix_len,
                   "k_u": k_u, "B": total_row_budget(n), "gold_first_token_ids": gold_ids,
                   "suffix_len": int(suffix.shape[0])}
            # FULL: full decode with attention-mass trace
            install_masks(meta, [n] * NUM_LAYERS, n, dev)
            with MassTrace(vstart, vn, NUM_LAYERS) as trace:
                ans_full, first_full, toks_full, log_full = full_decode(server, cache, suffix, prefix_len)
            mass = trace.mass
            p_full = F.log_softmax(log_full, dim=-1)
            row["full"] = {"answer": ans_full, "first_token_id": first_full, "generated_token_ids": toks_full,
                           "em": exact_score(ans_full, [gold]), "vqa": vqa_score(ans_full, q["answers"]),
                           "gold_first_nll": min(float(-p_full[g]) for g in gold_ids),
                           "first_matches_gold": int(first_full in gold_ids)}
            cov = {}
            for li in range(NUM_LAYERS):
                ks, vals = coverage_grid(mass[li], n)
                cov[li] = vals
            row["coverage_grid_k"] = ks
            row["coverage"] = [cov[li] for li in range(NUM_LAYERS)]
            sep_share = [float(mass[li, n:].sum() / mass[li].sum()) if float(mass[li].sum()) > 0 else 0.0 for li in range(NUM_LAYERS)]
            row["separator_mass_share"] = sep_share
            # U: full decode
            install_masks(meta, [k_u] * NUM_LAYERS, n, dev)
            ans_u, first_u, toks_u, log_u = full_decode(server, cache, suffix, prefix_len)
            p_u = F.log_softmax(log_u, dim=-1)
            row["U"] = {"answer": ans_u, "first_token_id": first_u, "generated_token_ids": toks_u,
                        "em": exact_score(ans_u, [gold]), "vqa": vqa_score(ans_u, q["answers"]),
                        "kl": kl_div(p_full, p_u), "gold_first_nll": min(float(-p_u[g]) for g in gold_ids),
                        "first_matches_gold": int(first_u in gold_ids), "first_agrees_full": int(first_u == first_full)}
            t_var = time.perf_counter()
            for name, base_k, alt_k in (("R", k_u, n), ("D", n, k_u)):
                kl, nll, agree, match, first_ids = [], [], [], [], []
                for li in range(NUM_LAYERS):
                    ks_ = [base_k] * NUM_LAYERS
                    ks_[li] = alt_k
                    install_masks(meta, ks_, n, dev)
                    lg = first_logits(runner.model, cache, suffix, prefix_len)
                    p_v = F.log_softmax(lg, dim=-1)
                    fid = int(lg.argmax())
                    kl.append(kl_div(p_full, p_v)); nll.append(min(float(-p_v[g]) for g in gold_ids))
                    agree.append(int(fid == first_full)); match.append(int(fid in gold_ids)); first_ids.append(fid)
                row[name] = {"kl": kl, "gold_first_nll": nll, "first_agrees_full": agree,
                             "first_matches_gold": match, "first_token_id": first_ids}
            BIAS.clear()
            torch.cuda.synchronize()
            row["timing_ms"] = {"image_t1_and_cache_ms": t1_ms if q is pending[0] else 0.0,
                                "question_total_ms": (time.perf_counter() - t_q) * 1e3,
                                "variants_64_ms": (time.perf_counter() - t_var) * 1e3,
                                "per_variant_forward_ms": (time.perf_counter() - t_var) * 1e3 / (2 * NUM_LAYERS)}
            append(jsonl, row)
            done.add((image["image_id"], q["question_id"]))
        del cache
        torch.cuda.empty_cache()
        print(json.dumps({"image": ix + 1, "of": len(images), "image_id": image["image_id"], "N": n,
                          "seconds": round(time.perf_counter() - t_img, 2)}), flush=True)
    BIAS.clear()
    return jsonl


def spearman(a, b):
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2 + 1
            i = j + 1
        return r
    ra, rb = np.asarray(rank(a)), np.asarray(rank(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def aggregate(run):
    import csv
    out = run / "phase1"
    rows = [json.loads(l) for l in (out / "per_question.jsonl").read_text().splitlines()]
    assert rows
    L = NUM_LAYERS
    kl_u = np.array([r["U"]["kl"] for r in rows])
    kl_r = np.array([r["R"]["kl"] for r in rows])
    kl_d = np.array([r["D"]["kl"] for r in rows])
    nll_u = np.array([r["U"]["gold_first_nll"] for r in rows])
    nll_r = np.array([r["R"]["gold_first_nll"] for r in rows])
    nll_d = np.array([r["D"]["gold_first_nll"] for r in rows])
    match_u = np.array([r["U"]["first_matches_gold"] for r in rows])
    match_r = np.array([r["R"]["first_matches_gold"] for r in rows])
    match_d = np.array([r["D"]["first_matches_gold"] for r in rows])
    agree_u = np.array([r["U"]["first_agrees_full"] for r in rows])
    agree_r = np.array([r["R"]["first_agrees_full"] for r in rows])
    agree_d = np.array([r["D"]["first_agrees_full"] for r in rows])
    s_raw = (kl_u[:, None] - kl_r).mean(0)
    s = np.clip(s_raw, 0, None)
    d_mean = kl_d.mean(0)
    rho = spearman(list(s_raw), list(d_mean))
    with (out / "sensitivity.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "s_l", "s_l_raw", "kl_U_mean", "kl_R_mean", "kl_D_mean", "kl_R_minus_U_median",
                    "nll_U_mean", "nll_R_mean", "nll_D_mean", "delta_first_match_gold_R_minus_U_pp",
                    "delta_first_match_gold_D_minus_U_pp", "delta_first_agree_full_R_minus_U_pp",
                    "delta_first_agree_full_D_minus_U_pp", "n"])
        for li in range(L):
            w.writerow([li, f"{s[li]:.8g}", f"{s_raw[li]:.8g}", f"{kl_u.mean():.8g}", f"{kl_r[:, li].mean():.8g}",
                        f"{d_mean[li]:.8g}", f"{np.median(kl_r[:, li] - kl_u):.8g}", f"{nll_u.mean():.8g}",
                        f"{nll_r[:, li].mean():.8g}", f"{nll_d[:, li].mean():.8g}",
                        f"{100 * (match_r[:, li].mean() - match_u.mean()):.4f}",
                        f"{100 * (match_d[:, li].mean() - match_u.mean()):.4f}",
                        f"{100 * (agree_r[:, li].mean() - agree_u.mean()):.4f}",
                        f"{100 * (agree_d[:, li].mean() - agree_u.mean()):.4f}", len(rows)])
    # coverage: per-question grids + mean curve on a fixed fraction grid
    fractions = np.linspace(0, 1, 65)
    mean_curve = np.zeros((L, len(fractions)))
    arrays = {"fractions": fractions}
    for i, r in enumerate(rows):
        ks = np.asarray(r["coverage_grid_k"], dtype=np.float64)
        n = r["N"]
        cov = np.asarray(r["coverage"])
        arrays[f"q{i}_k"] = ks
        arrays[f"q{i}_coverage"] = cov
        for li in range(L):
            mean_curve[li] += np.interp(fractions, ks / n, cov[li])
    mean_curve /= len(rows)
    arrays["mean_curve"] = mean_curve
    arrays["n_questions"] = np.array(len(rows))
    arrays["N_per_question"] = np.array([r["N"] for r in rows])
    arrays["separator_mass_share_mean"] = np.array([r["separator_mass_share"] for r in rows]).mean(0)
    np.savez(out / "coverage_curves.npz", **arrays)
    summary = {
        "n_questions": len(rows), "n_images": len({r["image_id"] for r in rows}),
        "spearman_rho_sR_vs_klD": rho, "s_l": s.tolist(), "s_l_raw": s_raw.tolist(), "kl_D_mean": d_mean.tolist(),
        "kl_U_mean": float(kl_u.mean()), "kl_R_mean_min_layer": int(kl_r.mean(0).argmin()),
        "s_positive_layers": int((s_raw > 0).sum()),
        "accuracy": {"FULL_em": float(np.mean([r["full"]["em"] for r in rows])), "U_em": float(np.mean([r["U"]["em"] for r in rows])),
                     "FULL_vqa": float(np.mean([r["full"]["vqa"] for r in rows])), "U_vqa": float(np.mean([r["U"]["vqa"] for r in rows])),
                     "U_first_agrees_full": float(agree_u.mean()), "U_first_matches_gold": float(match_u.mean()),
                     "FULL_first_matches_gold": float(np.mean([r["full"]["first_matches_gold"] for r in rows]))},
        "N": {"min": int(min(r["N"] for r in rows)), "median": float(np.median([r["N"] for r in rows])), "max": int(max(r["N"] for r in rows))},
        "timing_ms": {"per_variant_forward_mean": float(np.mean([r["timing_ms"]["per_variant_forward_ms"] for r in rows])),
                      "question_total_mean": float(np.mean([r["timing_ms"]["question_total_ms"] for r in rows])),
                      "image_t1_and_cache_mean": float(np.mean([r["timing_ms"]["image_t1_and_cache_ms"] for r in rows if r["timing_ms"]["image_t1_and_cache_ms"] > 0]))},
        "separator_mass_share_mean_by_layer": arrays["separator_mass_share_mean"].tolist(),
    }
    write(out / "summary.json", summary)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].bar(range(L), s_raw, color=["#4477aa" if v > 0 else "#cc6677" for v in s_raw])
        ax[0].set_xlabel("layer"); ax[0].set_ylabel("s_l = mean[KL(U) - KL(R_l)]"); ax[0].set_title("Layer-restore sensitivity (CAL)")
        ax[1].bar(range(L), d_mean, color="#999933")
        ax[1].set_xlabel("layer"); ax[1].set_ylabel("mean KL(D_l)"); ax[1].set_title(f"Layer-remove KL (Spearman rho vs s_l = {rho:.3f})")
        fig.tight_layout(); fig.savefig(out / "fig_sensitivity.png", dpi=130); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        cmap = plt.get_cmap("viridis")
        for li in range(L):
            ax.plot(fractions, mean_curve[li], color=cmap(li / (L - 1)), lw=1)
        ax.axvline(0.25, color="k", ls="--", lw=0.8)
        ax.set_xlabel("fraction of stored rows (importance order)"); ax.set_ylabel("cumulative attention mass share")
        ax.set_title("Coverage c_l(k) per layer (CAL mean; dark=layer 0, bright=layer 31)")
        fig.tight_layout(); fig.savefig(out / "fig_coverage.png", dpi=130); plt.close(fig)
    except Exception as exc:                 # figures are not evidence
        summary["figure_error"] = repr(exc)
        write(out / "summary.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--questions-per-image", type=int, default=5)
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--probe-report", help="write timing projection JSON to this path")
    args = ap.parse_args()
    run = Path(args.run).resolve()
    assert run.parent == ROOT / "runs" and run.name.startswith("layer_budget_kv25_")
    (run / "phase1").mkdir(parents=True, exist_ok=True)
    manifest = freeze_manifest(run, args.questions_per_image)
    if not args.aggregate_only:
        try:
            profile(run, manifest, args.max_images, print)
        except Exception:
            append(run / "failures.jsonl", {"phase": "phase1", "error": traceback.format_exc(), "utc": time.time()})
            raise
    summary = aggregate(run)
    if args.probe_report:
        per_q = summary["timing_ms"]["question_total_mean"] / 1e3
        per_img = summary["timing_ms"]["image_t1_and_cache_mean"] / 1e3
        total5 = 250 * (per_img + 5 * per_q)
        total3 = 250 * (per_img + 3 * per_q)
        decision = 5 if total5 <= 4 * 3600 else 3
        write(args.probe_report, {"probe_images": summary["n_images"], "probe_questions": summary["n_questions"],
                                  "per_variant_forward_ms": summary["timing_ms"]["per_variant_forward_mean"],
                                  "per_question_s": per_q, "per_image_t1_and_cache_s": per_img,
                                  "projected_total_s_5_per_image": total5, "projected_total_s_3_per_image": total3,
                                  "limit_s": 4 * 3600, "questions_per_image_decision": decision})
    print(json.dumps({k: v for k, v in summary.items() if k in ("n_questions", "n_images", "spearman_rho_sR_vs_klD", "timing_ms", "accuracy")}, indent=1))


if __name__ == "__main__":
    main()

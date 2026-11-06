#!/usr/bin/env python3
"""Phase 2 — allocation vectors from the CAL sensitivity profile (contract A5).

Reads ``runs/<RUN>/phase1/sensitivity.csv`` and ``coverage_curves.npz`` and
writes one frozen ``layer_budget_llava_<id>.json`` per candidate (U, A1-0.5,
A1-1, A1-2, A2, F-front, F-back) plus ``vectors.csv``, ``fig_ratios.png`` and
``summary.json`` under ``runs/<RUN>/phase2``.  Files are created exclusively;
re-running against an existing phase2 directory verifies rather than rewrites.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from mmimpress.layer_budget import (NUM_LAYERS, R_MAX, R_MIN, allocate_power,
                                    allocate_water_filling, budget_document,
                                    front_schedule, layer_keep_counts,
                                    load_budget_document, uniform_ratios)
from mmimpress.prefix25 import sha

GAMMAS = (0.5, 1.0, 2.0)


def publish(path, doc):
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text())
        assert old == doc, f"frozen vector differs: {path}"
        return
    with path.open("x") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run).resolve()
    p1, p2 = run / "phase1", run / "phase2"
    p2.mkdir(exist_ok=True)
    rows = list(csv.DictReader((p1 / "sensitivity.csv").open()))
    assert [int(r["layer"]) for r in rows] == list(range(NUM_LAYERS))
    s = [float(r["s_l"]) for r in rows]
    npz = np.load(p1 / "coverage_curves.npz")
    fractions = npz["fractions"].tolist()
    curves = npz["mean_curve"].tolist()
    n_med = float(np.median(npz["N_per_question"]))
    step = 64.0 / n_med
    source = {"sensitivity_csv_sha256": sha(p1 / "sensitivity.csv"),
              "coverage_npz_sha256": sha(p1 / "coverage_curves.npz"),
              "cal_manifest_sha256": sha(p1 / "cal_manifest.json"),
              "n_questions": int(npz["n_questions"]), "cal_median_N": n_med}
    vectors = {"U": uniform_ratios()}
    for g in GAMMAS:
        vectors[f"A1-{g:g}"] = allocate_power(s, g)
    vectors["A2"] = allocate_water_filling(fractions, curves, step)
    vectors["F-front"] = front_schedule()
    vectors["F-back"] = front_schedule(reverse=True)
    docs = {}
    for alloc_id, r in vectors.items():
        extra = {"gamma": float(alloc_id.split("-")[1])} if alloc_id.startswith("A1") else {}
        if alloc_id == "A2":
            extra = {"water_filling_step_ratio": step, "grid": "fraction grid 0..1 (65 points), CAL mean curve"}
        if alloc_id.startswith("F-"):
            extra = {"schedule": "linear 0.40 -> 0.10 normalized to 8.0" + (" (reversed)" if alloc_id == "F-back" else "")}
        doc = budget_document(alloc_id, r, model="llava-hf/llava-v1.6-vicuna-7b-hf",
                              source={**source, **extra, "kind": "fixed" if alloc_id in ("U", "F-front", "F-back") else "profiled"})
        publish(p2 / f"layer_budget_llava_{alloc_id}.json", doc)
        docs[alloc_id] = load_budget_document(p2 / f"layer_budget_llava_{alloc_id}.json")
    with (p2 / "vectors.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "s_l"] + list(vectors))
        for li in range(NUM_LAYERS):
            w.writerow([li, f"{s[li]:.8g}"] + [f"{vectors[a][li]:.6f}" for a in vectors])
    # Example per-image counts at the DEV/CAL/HOLD extreme and median N (CPU; G3 sanity)
    examples = {}
    for n in (1344, 1440, 2112, 2208, 2256, 2880):
        examples[str(n)] = {a: layer_keep_counts(vectors[a], n) for a in vectors}
        assert len({sum(v) for v in examples[str(n)].values()}) == 1
    summary = {"alloc_ids": list(vectors), "sha256": {a: docs[a]["layer_budget_sha256"] for a in docs},
               "r_min": R_MIN, "r_max": R_MAX, "water_filling_step_ratio": step, "cal_median_N": n_med,
               "clipped_layers": {a: {"at_r_min": int(sum(abs(x - R_MIN) < 1e-9 for x in vectors[a])),
                                      "at_r_max": int(sum(abs(x - R_MAX) < 1e-9 for x in vectors[a]))} for a in vectors},
               "front_mass_first_8_layers": {a: float(sum(vectors[a][:8]) / 8.0) for a in vectors},
               "example_counts_sum_equal": True, "example_counts": examples}
    publish(p2 / "summary.json", summary)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = np.arange(NUM_LAYERS)
        for i, a in enumerate(vectors):
            ax.plot(x, vectors[a], marker="o", ms=3, lw=1.2, label=a)
        ax.axhline(0.25, color="k", lw=0.6, ls=":")
        ax.set_xlabel("layer"); ax.set_ylabel("r_l (content rows / N)"); ax.set_ylim(0, 0.65)
        ax.set_title("Layer-wise budget candidates (sum r_l = 8.0; clip [0.10, 0.60])")
        ax.legend(ncol=4, fontsize=8)
        fig.tight_layout(); fig.savefig(p2 / "fig_ratios.png", dpi=130); plt.close(fig)
    except Exception as exc:
        print("figure skipped:", repr(exc))
    print(json.dumps({k: summary[k] for k in ("sha256", "clipped_layers", "front_mass_first_8_layers")}, indent=1))


if __name__ == "__main__":
    main()

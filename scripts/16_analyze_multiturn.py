"""Regenerate turn-aware tables, validation, plots, and a concise run report."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmimpress.multiturn_results import build_artifacts, read_jsonl


METHOD_ORDER = ["ReComp", "FullLoad", "SparseVLM 25%",
                "Static+Diverse 25%", "Static+Diverse 50%"]


def _mean(rows, field):
    vals = [float(r[field]) for r in rows if r.get(field) is not None]
    return float(np.mean(vals)) if vals else None


def _fmt(value, digits=2):
    return "—" if value is None else f"{float(value):.{digits}f}"


def _overall_table(overall):
    by_method = {r["method"]: r for r in overall}
    lines = [
        "| Method | Quality† | TTFT mean / p50 / p95 (ms) | SSD read (MB) | Selector (ms) | Logical kept tokens | SSD total / Full KV |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        if method not in by_method:
            continue
        r = by_method[method]
        ttft = "/".join(_fmt(r.get(x), 1) for x in
                         ("ttft_ms_mean", "ttft_ms_p50", "ttft_ms_p95"))
        ratio = r.get("logical_visual_token_ratio_mean")
        ssd_ratio = r.get("ssd_read_ratio_vs_fullload")
        lines.append(
            f"| {method} | {_fmt(r.get('quality_mean'), 3)} | {ttft} | "
            f"{_fmt(r.get('ssd_read_mb_mean'), 1)} | "
            f"{_fmt(r.get('selector_ms_mean'), 2)} | "
            f"{_fmt(100 * ratio if ratio is not None else None, 1)}% | "
            f"{_fmt(100 * ssd_ratio if ssd_ratio is not None else None, 1)}% |")
    lines.append("")
    lines.append("† VisDial system run의 quality는 normalized generative match 보조지표이며 공식 VisDial 점수가 아니다.")
    lines.append("`ssd_read_chunk_units`는 K/V/probe 파일별 chunk-equivalent 합이며 unique selected chunk 수가 아니다.")
    return "\n".join(lines)


def _hypotheses(rows, config):
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    full = by_method.get("FullLoad", [])
    sd25 = by_method.get("Static+Diverse 25%", [])
    sd50 = by_method.get("Static+Diverse 50%", [])

    def read_ratio(method_rows):
        vals = [float(r["ssd_read_bytes"]) / float(r["full_visual_kv_bytes"])
                for r in method_rows if float(r.get("full_visual_kv_bytes") or 0) > 0]
        return float(np.mean(vals)) if vals else None

    def later_adv(method_rows):
        if not method_rows or not full:
            return None
        f = defaultdict(list)
        s = defaultdict(list)
        for r in full:
            f[int(r["turn_id"])].append(float(r["ttft_ms"]))
        for r in method_rows:
            s[int(r["turn_id"])].append(float(r["ttft_ms"]))
        turns = sorted(set(f) & set(s))
        return {str(t): 100.0 * (1.0 - np.mean(s[t]) / np.mean(f[t]))
                for t in turns}

    expected_dialogs = int(config.get("n_dialogs", 0))
    image_builds = int(config.get("image_kv_build_count", -1))
    static_builds = int(config.get("static_metadata_build_count", -1))
    static_loads = int(config.get("static_metadata_load_count", -1))
    return {
        "dataset": config.get("dataset"),
        "h1_single_build_and_reuse": {
            "supported": (image_builds == expected_dialogs and
                          static_builds == expected_dialogs and
                          static_loads == expected_dialogs and
                          all(r.get("image_store_reused_across_turns") for r in rows)),
            "image_kv_build_count": image_builds,
            "static_metadata_build_count": static_builds,
            "static_metadata_load_count": static_loads,
            "dialogs": expected_dialogs,
        },
        "h2_ssd_read_fraction": {
            "fullload": read_ratio(full),
            "static_diverse_25": read_ratio(sd25),
            "static_diverse_50": read_ratio(sd50),
        },
        "h3_selector_overhead_by_turn_ms": {
            str(t): _mean([r for r in sd25 if int(r["turn_id"]) == t],
                          "selector_ms")
            for t in sorted({int(r["turn_id"]) for r in sd25})
        },
        "h4_static_diverse_25_ttft_reduction_by_turn_pct": later_adv(sd25),
        "h5_quality_delta_vs_fullload": {
            "static_diverse_25": ((_mean(sd25, "quality_score") or 0.0) -
                                  (_mean(full, "quality_score") or 0.0))
                                 if sd25 and full else None,
            "static_diverse_50": ((_mean(sd50, "quality_score") or 0.0) -
                                  (_mean(full, "quality_score") or 0.0))
                                 if sd50 and full else None,
        },
    }


def _plot(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {"ReComp": "#555555", "FullLoad": "#d62728",
              "SparseVLM 25%": "#ff7f0e", "Static+Diverse 25%": "#1f77b4",
              "Static+Diverse 50%": "#2ca02c"}

    def line_plot(field, ylabel, filename, scale=1.0):
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for method in METHOD_ORDER:
            mr = [r for r in rows if r["method"] == method]
            if not mr or not any(r.get(field) is not None for r in mr):
                continue
            turns = sorted({int(r["turn_id"]) for r in mr})
            ys = [_mean([r for r in mr if int(r["turn_id"]) == t], field)
                  for t in turns]
            ax.plot(turns, [y / scale for y in ys], marker="o", label=method,
                    color=colors.get(method))
        ax.set_xlabel("Conversation turn")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)

    line_plot("ttft_ms", "True TTFT (ms)", "graph_turn_ttft.png")
    line_plot("ssd_read_bytes", "SSD bytes read / request (MB)",
              "graph_turn_ssd_read.png", 1e6)
    line_plot("selector_ms", "Selector time (ms)", "graph_turn_selector.png")
    line_plot("quality_score", "Auxiliary quality", "graph_turn_quality.png")

    # These are especially informative for MMDU, but emitting them for
    # VisDial makes the fixed one-image working set explicit rather than hidden.
    uniq = {}
    for r in rows:
        uniq[(r["dialog_id"], int(r["turn_id"]))] = r
    turns = sorted({int(r["turn_id"]) for r in uniq.values()})
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    vals = [_mean([r for r in uniq.values() if int(r["turn_id"]) == t],
                  "full_visual_kv_bytes") / 1e9 for t in turns]
    ax.plot(turns, vals, marker="o")
    ax.set(xlabel="Conversation turn", ylabel="Active Full visual KV (GB)")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out_dir / "graph_turn_full_visual_kv.png", dpi=180)
    plt.close(fig)

    active = sorted({int(r["active_images"]) for r in rows})
    unique_requests = {}
    for r in rows:
        unique_requests[(r["dialog_id"], int(r["turn_id"]))] = r
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    full_by_active = [
        _mean([r for r in unique_requests.values()
               if int(r["active_images"]) == n], "full_visual_kv_bytes") / 1e9
        for n in active
    ]
    ax.plot(active, full_by_active, marker="o", color="#9467bd")
    ax.set(xlabel="Active images", ylabel="Required Full visual KV (GB)")
    ax.set_xticks(active)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out_dir / "graph_active_images_full_visual_kv.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for method in ("FullLoad", "Static+Diverse 25%", "Static+Diverse 50%"):
        mr = [r for r in rows if r["method"] == method]
        if not mr:
            continue
        ys = [_mean([r for r in mr if int(r["active_images"]) == n],
                    "ssd_read_bytes") / 1e6 for n in active]
        ax.plot(active, ys, marker="o", label=method, color=colors[method])
    ax.set(xlabel="Active images", ylabel="SSD bytes read / request (MB)")
    ax.set_xticks(active)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "graph_active_images_ssd_read.png", dpi=180)
    plt.close(fig)


def _write_readme(run_dir, config, overall, validation, hypotheses):
    title = ("VisDial v1.0 multi-turn system run" if
             config.get("dataset") == "visdial_v1.0_val" else
             "MMDU multi-turn correctness run")
    limitations = [
        "Static+Diverse는 SSD에서 읽는 payload를 줄이지만 현재 PrefixCache는 full-length GPU KV tensor를 할당한다. selected KV bytes를 실제 GPU allocation 절감으로 해석하면 안 된다.",
        "VisDial generative match는 보조 지표다. candidate conditional-likelihood 기반 MRR/R@K/Mean Rank/NDCG와 구분한다.",
        "caption-only importance calibration만 사용했으며 평가 turn/future answer는 reorder나 selector 입력에 들어가지 않는다.",
    ]
    text = f"""# {title}

- Schema: `{config.get('schema_version')}`
- Dialogs / turns: {config.get('n_dialogs')} / {config.get('n_turns')}
- History: `{config.get('history_policy')}`
- Calibration: `{config.get('calibration_policy')}`
- Page cache: `{'warm' if not config.get('cold_page_cache') else 'cold'}`
- Max new tokens: {config.get('max_new_tokens')}
- Validation: `{'PASS' if validation.get('passed') else 'FAIL'}`

## Overall

{_overall_table(overall)}

## Hypothesis checks

The machine-readable values are in `analysis.json`. The run records every turn separately; `per_turn.csv` and `per_active_images.csv` preserve the two workload axes.

## Timing boundary

`request start → selector → SSD pread → reconstruction/scatter → prompt prefill → first output token argmax → CUDA synchronize` is TTFT. Later autoregressive generation is `decode_ms`; `e2e_ms ≈ ttft_ms + decode_ms` is validated row by row.

## Limitations

""" + "\n".join(f"- {x}" for x in limitations) + "\n"
    with open(run_dir / "README.md", "w") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    overall, _, _, _, validation = build_artifacts(run_dir, config)
    rows = read_jsonl(run_dir / "raw.jsonl")
    if not rows:
        raise RuntimeError("raw.jsonl is empty")
    hypotheses = _hypotheses(rows, config)
    with open(run_dir / "analysis.json", "w") as f:
        json.dump(hypotheses, f, indent=1)
    _plot(rows, run_dir)
    _write_readme(run_dir, config, overall, validation, hypotheses)

    if args.results_dir:
        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        names = ["config.json", "raw.jsonl", "summary.csv", "per_turn.csv",
                 "per_active_images.csv", "per_dialog.csv", "validation.json",
                 "analysis.json", "README.md"]
        names += sorted(p.name for p in run_dir.glob("graph_*.png"))
        for name in names:
            shutil.copy2(run_dir / name, results_dir / name)
        with open(results_dir / "raw_location.json", "w") as f:
            json.dump({"raw_jsonl": str((run_dir / "raw.jsonl").resolve()),
                       "mirrored_raw_jsonl": str(
                           (results_dir / "raw.jsonl").resolve())}, f,
                      indent=1)

    print(json.dumps({"validation": validation,
                      "hypotheses": hypotheses}, indent=1))
    if not validation["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

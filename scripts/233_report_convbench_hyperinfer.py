#!/usr/bin/env python3
"""ConvBench HyperInfer run: report.md and report_tables.json (contract docs/convbench_hyperinfer_contract.md, section 7).

Reads committed conversations, gate files and judge outputs only: HyperInfer from this run; ReComp, FullLoad,
IMPRESS and Ours from the main experiment (results/convbench_main_20261122T103401Z); the MT-GQA HyperInfer
figures from results/mt_gqa_hyperinfer_v2_20261125T034057Z/report.json.  Never touches the GPU or the stores.
Not part of the measurement freeze (file name matches NNN_report_*.py).  Works on a run that is still going or
that stopped at a gate: it reports what exists.

  python scripts/233_report_convbench_hyperinfer.py --res results/convbench_hyperinfer_<UTC>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import convbench_hyperinfer_common as HC  # noqa: E402

C = HC.C
R220 = HC.load_with_common("_convbench_hyperinfer_main_report", "220_report_convbench_main.py")      # table / bootstrap helpers of the main report
P230 = HC._load("_convbench_hyperinfer_mtgqa_report", "230_report_mt_gqa_hyperinfer_v2.py")            # the prefetch summary of the MT-GQA report
fx, table, bootstrap_mean, bootstrap_ratio, dist, ci = R220.fx, R220.table, R220.bootstrap_mean, R220.bootstrap_ratio, R220.dist, R220.ci
N = HC.DISPLAY
MB = 1e6
STAGES = (("_first_turn", "S1"), ("_second_turn", "S2"), ("_third_turn", "S3"))
HYPER, IMP, FULL, RECOMP, OURS = HC.HYPER, HC.IMP, HC.FULL, HC.RECOMP, HC.OURS
TTFT_PAIRS = [(OURS, HYPER), (HYPER, IMP), (HYPER, FULL), (HYPER, RECOMP)]
QUALITY_PAIRS = [(OURS, HYPER), (HYPER, FULL), (HYPER, RECOMP)]
TURNS = (("T2", (2,)), ("T3", (3,)), ("T2–T3", (2, 3)))


def plain(o):
    """JSON-safe copy: numpy values to Python values, non-finite numbers to None."""
    if isinstance(o, dict):
        return {str(k): plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return plain(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def pct(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def nanmean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else None


# ------------------------------------------------------------------ reading
def load_recorded(model, index):
    """{conversation id: document} of the main experiment (four methods)."""
    out = {}
    for conv in index:
        doc = HC.recorded_doc(model, conv["conversation_id"])
        if doc is not None:
            out[conv["conversation_id"]] = doc
    return out


def ok(row):
    return row is not None and row.get("status") == "ok"


def cell(cid, row):
    """One HyperInfer follow-up request in the shape the MT-GQA report's ``prefetch_summary`` reads."""
    h = row["result"]["hyperinfer"]
    L = len(h["layers"]["fallback"])
    return {"cid": cid, "turn": row["turn_id"], "ttft": row["ttft_ms"], "bytes": row["ssd_read_bytes"], "calls": row["read_calls"],
            "kv_bytes": row["k_read_bytes"] + row["v_read_bytes"], "prefill": row["prefill_ms"], "hook": row["hook_ms"],
            "blocked": row["read_blocked_ms"], "suffix": row.get("suffix_tokens"), "input": row["input_tokens"], "imp": row.get("impress"),
            "h": {k: h.get(k) for k in P230.SCALARS}, "hl": {k: P230._arr(h["layers"][k], L) for k in P230.LAYER_COLS},
            "k_ms": np.asarray(h["k_read_ms"], dtype=np.float32), "v_ms": np.asarray(h["v_read_ms"], dtype=np.float32),
            "p_ms": np.asarray(h["probe_read_ms"], dtype=np.float32)}


def series(docs, role):
    """{conversation id: (T2 TTFT, T3 TTFT)} of the conversations whose two follow-up requests of ``role`` are ok."""
    out = {}
    for cid, doc in docs.items():
        by = C.rows_by_key(doc)
        rows = [by.get((role, t)) for t in (2, 3)]
        if all(ok(r) for r in rows):
            out[cid] = tuple(r["ttft_ms"] for r in rows)
    return out


def method_row(docs, role, date):
    out = {"method": N[role], "run_date": date, "conversations": len(docs)}
    for label, turns in TURNS:
        rows = [r for d in docs.values() for r in d["rows"] if r["role"] == role and r["turn_id"] in turns and ok(r)]
        out[label] = dist([r["ttft_ms"] for r in rows])
    fu = [r for d in docs.values() for r in d["rows"] if r["role"] == role and r["turn_id"] in (2, 3) and ok(r)]
    t1 = [r for d in docs.values() for r in d["rows"] if r["role"] == role and r["turn_id"] == 1 and ok(r)]
    out.update(followup_requests=len(fu), ssd_mb_per_followup=float(np.mean([r["ssd_read_bytes"] for r in fu]) / MB) if fu else None,
               read_calls_per_followup=float(np.mean([r["read_calls"] for r in fu])) if fu else None,
               t1_ttft_mean_ms=float(np.mean([r["ttft_ms"] for r in t1])) if t1 else None,
               failed_requests=sum(1 for d in docs.values() for r in d["rows"] if r["role"] == role and not ok(r)),
               generated_tokens_mean=float(np.mean([r["generated_tokens"] for r in fu + t1])) if fu or t1 else None)
    return out


def run_dates(docs):
    stamps = sorted(d["committed_at_utc"] for d in docs.values() if d.get("committed_at_utc"))
    if not stamps:
        return "—"
    a, b = stamps[0][:10], stamps[-1][:10]
    return a if a == b else f"{a} – {b}"


# ------------------------------------------------------------------- stats
def paired_ttft(sa, sb):
    """Paired comparison a vs b over the conversations both have."""
    common = sorted(set(sa) & set(sb))
    if not common:
        return None
    a = np.array([sa[c] for c in common]); b = np.array([sb[c] for c in common])
    d = bootstrap_mean((a.sum(axis=1) - b.sum(axis=1)) / 2)
    r = bootstrap_ratio(a.sum(axis=1), b.sum(axis=1))
    return {"conversations": len(common), "a_mean_ms": float(a.mean()), "b_mean_ms": float(b.mean()),
            "diff": d, "ratio": r,
            "by_turn": {f"T{t}": {"diff_ms": float((a[:, i] - b[:, i]).mean()), "ratio": float(a[:, i].sum() / b[:, i].sum())} for i, t in enumerate((2, 3))}}


def impress_stats(items):
    """Fallback rate, mean Jaccard of the probe heads and the K/V chunk fraction the selection needs, from compact per-layer records."""
    if not items:
        return None
    fb = np.array([x["fb"] for x in items], dtype=float); sim = np.array([x["sim"] for x in items], dtype=float)
    kc = np.array([x["kc"] for x in items], dtype=float); vc = np.array([x["vc"] for x in items], dtype=float)
    nc = np.array([x["nc"] for x in items], dtype=float)[:, None]
    kv = (kc + vc) / (2 * nc)
    cons = fb == 0
    return {"followup_requests": int(fb.shape[0]), "fallback_rate": float(fb.mean()), "mean_jaccard": float(sim.mean()),
            "kv_chunk_fraction": float(kv.mean()), "kv_chunk_fraction_non_fallback_layers": float(kv[cons].mean()) if cons.any() else None,
            "requests_with_any_fallback": float((fb.sum(axis=1) > 0).mean()), "requests_all_layers_fallback": float((fb.sum(axis=1) == fb.shape[1]).mean())}


def kv_full_bytes(geometry):
    g = geometry
    return 2 * g["num_layers"] * g["v_token_num"] * g["num_heads"] * g["head_dim"] * 2      # K and V, FP16


def length_bins(cells, rec_ttft):
    """Follow-up requests in four groups by the number of tokens after the image (quartiles of this model's requests)."""
    cs = [c for c in cells if c["suffix"] is not None]
    if len(cs) < 4:
        return None
    suffix = np.array([c["suffix"] for c in cs], dtype=float)
    edges = np.quantile(suffix, [0.25, 0.5, 0.75])
    which = np.searchsorted(edges, suffix, side="right")
    out = []
    for b in range(4):
        group = [c for c, w in zip(cs, which) if w == b]
        if not group:
            out.append({"bin": b + 1, "requests": 0})
            continue
        win = np.concatenate([c["hl"]["window_ms"][1:] for c in group]).astype(np.float64)
        inw = np.concatenate([c["hl"]["reads_done_in_window"][1:] for c in group]).astype(np.float64)
        cinw = np.concatenate([c["hl"]["chunks_done_in_window"][1:] for c in group]).astype(np.float64)
        pred = np.concatenate([c["hl"]["predicted_chunks"][1:] for c in group]).astype(np.float64)
        hit = sum(c["h"]["hit_tokens"] or 0 for c in group); used = sum(c["h"]["used_tokens_after_layer0"] or 0 for c in group)
        both = [(c["ttft"], rec_ttft[(c["cid"], c["turn"])]) for c in group if (c["cid"], c["turn"]) in rec_ttft]
        sg = np.array([c["suffix"] for c in group])
        out.append({"bin": b + 1, "requests": len(group), "t2": sum(c["turn"] == 2 for c in group), "t3": sum(c["turn"] == 3 for c in group),
                    "suffix_min": int(sg.min()), "suffix_max": int(sg.max()), "suffix_mean": float(sg.mean()),
                    "window_ms_per_layer": nanmean(win), "reads_in_window_per_layer": nanmean(inw), "chunks_in_window_per_layer": nanmean(cinw),
                    "predicted_chunks_per_layer": nanmean(pred), "coverage": hit / used if used else None,
                    "hyperinfer_ttft_ms": float(np.mean([c["ttft"] for c in group])),
                    "paired_requests": len(both), "hyperinfer_ttft_paired_ms": float(np.mean([x for x, _ in both])) if both else None,
                    "impress_ttft_ms": float(np.mean([y for _, y in both])) if both else None,
                    "ratio": float(sum(x for x, _ in both) / sum(y for _, y in both)) if both else None})
    return {"edges": [float(e) for e in edges], "bins": out}


def decomposition(cells):
    if not cells:
        return None
    return {"requests": len(cells), "ttft_ms": float(np.mean([c["ttft"] for c in cells])), "prefill_ms": nanmean([c["prefill"] for c in cells]),
            "hook_ms": nanmean([c["hook"] for c in cells]), "read_wait_ms": nanmean([c["blocked"] for c in cells]),
            "text_compute_prefill_minus_hook_ms": nanmean([c["prefill"] - c["hook"] for c in cells]),
            "outside_prefill_ms": nanmean([c["ttft"] - c["prefill"] for c in cells])}


def recorded_decomposition(rows):
    rows = [r for r in rows if ok(r)]
    if not rows:
        return None
    return {"requests": len(rows), "ttft_ms": float(np.mean([r["ttft_ms"] for r in rows])), "prefill_ms": nanmean([r["prefill_ms"] for r in rows]),
            "hook_ms": nanmean([r["hook_ms"] for r in rows]), "read_wait_ms": nanmean([r["read_blocked_ms"] for r in rows]),
            "text_compute_prefill_minus_hook_ms": nanmean([r["prefill_ms"] - r["hook_ms"] for r in rows]),
            "probe_wait_ms": nanmean([r["result"].get("prefetch_wait_ms") for r in rows if r["result"].get("prefetch_wait_ms") is not None]),
            "outside_prefill_ms": nanmean([r["ttft_ms"] - r["prefill_ms"] for r in rows])}


# ------------------------------------------------------------------ quality
def quality(res, model, identity):
    main_dir = HC.MAIN_RUN / "judge" / model
    summary = C.read_json(main_dir / "summary.json")
    wins = {}
    for r in C.read_jsonl(main_dir / "judge_raw.jsonl"):
        wins.setdefault(r["role"], {}).setdefault(r["conversation_id"], {})[r["judge_turn"]] = 1.0 if r["model_wins"] else 0.0
    out = {"recorded": {role: {k: summary["roles"][role][k] for k in ("S1", "S2", "S3", "Avg", "unresolved", "generation_failed")} for role in summary["roles"]},
           "denominator": summary["denominator"], "hyperinfer": None, "source": None, "pairs": []}
    judged = (Path(res) / "judge" / model / "DONE").exists()
    if judged:
        new = C.read_json(Path(res) / "judge" / model / "summary.json")
        recs = C.read_jsonl(Path(res) / "judge" / model / "judge_raw.jsonl")
        for r in recs:
            wins.setdefault(HYPER, {}).setdefault(r["conversation_id"], {})[r["judge_turn"]] = 1.0 if r["model_wins"] else 0.0
        out["hyperinfer"] = {k: new["roles"][HYPER][k] for k in ("S1", "S2", "S3", "Avg", "unresolved", "generation_failed")}
        out["source"] = "judged"
        out["judge"] = {k: new.get(k) for k in ("conversations", "denominator", "judged_now", "extractions_now", "cache_hits")}
        # consistency: conversations whose three answers equal IMPRESS's, verdicts against the recorded IMPRESS verdicts
        mism = set((identity or {}).get("first_mismatching_turn_per_conversation") or {})
        same = [c for c in wins.get(HYPER, {}) if c not in mism and c in wins.get(IMP, {})]
        agree = sum(all(wins[HYPER][c].get(t) == wins[IMP][c].get(t) for t, _ in STAGES) for c in same)
        out["verdicts_on_identical_answers"] = {"conversations": len(same), "same_verdicts": int(agree)}
    elif identity is not None and not identity["judge_needed"]:
        wins[HYPER] = wins[IMP]
        out["hyperinfer"] = dict(out["recorded"][IMP])
        out["source"] = "recorded_impress"
    else:
        return out
    pairs = list(QUALITY_PAIRS) + ([(HYPER, IMP)] if judged else [])
    for a, b in pairs:
        common = sorted(set(wins.get(a, {})) & set(wins.get(b, {})))
        if not common:
            continue
        pa = np.array([[wins[a][c].get(t, 0.0) for t, _ in STAGES] for c in common]); pb = np.array([[wins[b][c].get(t, 0.0) for t, _ in STAGES] for c in common])
        e = {"pair": f"{N[a]} − {N[b]}", **bootstrap_mean((pa.mean(axis=1) - pb.mean(axis=1)) * 100)}
        e["stage_differences"] = {label: float((pa[:, i].mean() - pb[:, i].mean()) * 100) for i, (_, label) in enumerate(STAGES)}
        e["conversations_with_different_outcome"] = int((np.abs(pa - pb).sum(axis=1) > 0).sum())
        out["pairs"].append(e)
    return out


# ---------------------------------------------------------------- per model
def model_report(res, model, index):
    res = Path(res)
    g = res / "gates"
    out = {"model": model, "label": C.MODEL_NAME[model]}
    out["gates"] = {"tests": C.read_json(g / "tests.json"), "gate": C.read_json(g / model / "gate_gate.json"), "pilot": C.read_json(g / model / "gate_pilot.json"),
                    "overflow": C.read_json(g / model / "gate8_overflow.json"), "overflow_selection": C.read_json(C.model_dir(res, model) / "overflow_selection.json"),
                    "admission": C.read_json(g / model / "admission.json")}
    identity = out["identity"] = C.read_json(g / model / "identity.json")
    recorded = load_recorded(model, index)
    main = C.committed(res, model, "main", index)
    out["main_conversations"] = len(main)
    out["main_expected"] = len(C.phase_conversations(res, model, "main", index))
    out["main_complete"] = C.phase_complete(res, model, "main")
    old_date = run_dates(recorded)
    out["methods"] = [method_row(recorded, r, old_date) for r in (RECOMP, FULL, IMP)]
    if main:
        out["methods"].append(method_row(main, HYPER, run_dates(main)))
    out["methods"].append(method_row(recorded, OURS, old_date))
    if not main:
        out["quality"] = quality(res, model, identity)
        return out

    # ---- 2. paired comparisons (same conversation, same turns)
    S = {r: series(recorded, r) for r in (RECOMP, FULL, IMP, OURS)}
    S[HYPER] = series(main, HYPER)
    out["paired"] = []
    for a, b in TTFT_PAIRS:
        p = paired_ttft(S[a], S[b])
        if p:
            out["paired"].append({"a": N[a], "b": N[b], **p})
    out["pairing"] = {"hyperinfer_conversations_with_both_followups": len(S[HYPER]), "committed": len(main)}

    # ---- HyperInfer follow-up cells
    cells = []
    rec_ttft, rec_rows = {}, []
    for cid, doc in main.items():
        by = C.rows_by_key(doc)
        old = C.rows_by_key(recorded[cid]) if cid in recorded else {}
        for t in (2, 3):
            r = by.get((HYPER, t))
            if ok(r):
                cells.append(cell(cid, r))
                o = old.get((IMP, t))
                if ok(o):
                    rec_ttft[(cid, t)] = o["ttft_ms"]; rec_rows.append((cid, t, o))
    out["followup_requests"] = len(cells)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["prefetch"] = {label: (P230.prefetch_summary([c for c in cells if c["turn"] in turns]) if any(c["turn"] in turns for c in cells) else None)
                           for label, turns in TURNS}
        out["bins"] = length_bins(cells, rec_ttft)
    out["decomposition"] = {label: {"hyperinfer": decomposition([c for c in cells if c["turn"] in turns]),
                                    "impress_recorded": recorded_decomposition([o for _, t, o in rec_rows if t in turns])} for label, turns in TURNS}

    # ---- 7. bytes against IMPRESS (same requests)
    pairs = [(c, next(o for cid, t, o in rec_rows if cid == c["cid"] and t == c["turn"])) for c in cells if (c["cid"], c["turn"]) in rec_ttft]
    if pairs:
        hb = np.array([c["bytes"] for c, _ in pairs], dtype=float); ib = np.array([o["ssd_read_bytes"] for _, o in pairs], dtype=float)
        full = np.array([kv_full_bytes(main[c["cid"]]["geometry"]) for c, _ in pairs], dtype=float)
        hk = np.array([c["kv_bytes"] for c, _ in pairs], dtype=float); ik = np.array([o["k_read_bytes"] + o["v_read_bytes"] for _, o in pairs], dtype=float)
        out["bytes_vs_impress"] = {"requests": len(pairs), "hyperinfer_mb": float(hb.mean() / MB), "impress_mb": float(ib.mean() / MB),
                                   "ratio": float(hb.sum() / ib.sum()), "hyperinfer_kv_read_fraction": float((hk / full).mean()),
                                   "impress_kv_read_fraction": float((ik / full).mean()),
                                   "hyperinfer_read_calls": float(np.mean([c["calls"] for c, _ in pairs])),
                                   "impress_read_calls": float(np.mean([o["read_calls"] for _, o in pairs]))}

    # ---- 10. IMPRESS-style statistics
    out["impress_stats"] = {"hyperinfer": impress_stats([c["imp"] for c in cells if c["imp"]]),
                            "impress_recorded": impress_stats([o["impress"] for _, _, o in rec_rows if "impress" in o]),
                            "impress_recorded_all": impress_stats([r["impress"] for d in recorded.values() for r in d["rows"]
                                                                   if r["role"] == IMP and r["turn_id"] > 1 and ok(r) and "impress" in r])}

    # ---- 11. inputs above 4096 tokens
    above = []
    for cid, doc in main.items():
        old = C.rows_by_key(recorded[cid]) if cid in recorded else {}
        for r in doc["rows"]:
            if r["role"] == HYPER and ((ok(r) and r["input_above_4096"]) or (not ok(r) and ok(old.get((IMP, r["turn_id"]))) and old[(IMP, r["turn_id"])]["input_above_4096"])):
                o = old.get((IMP, r["turn_id"]))
                above.append({"conversation_id": cid, "turn": r["turn_id"], "status": r["status"], "input_tokens": r.get("input_tokens"),
                              "generated_tokens": r.get("generated_tokens"), "ttft_ms": r.get("ttft_ms"), "failure": r.get("failure"),
                              "tokens_equal_recorded_impress": ok(r) and ok(o) and r["generated_token_ids"] == o["generated_token_ids"],
                              "recorded_impress_ttft_ms": o["ttft_ms"] if ok(o) else None})
    hyper_rows = [r for d in main.values() for r in d["rows"] if r["role"] == HYPER]
    out["above_4096"] = {"requests": above, "count": len(above), "failed": sum(a["status"] != "ok" for a in above),
                         "recorded_impress_count": sum(1 for d in recorded.values() for r in d["rows"] if r["role"] == IMP and ok(r) and r["input_above_4096"]),
                         "input_tokens": {f"T{t}": dist([r["input_tokens"] for r in hyper_rows if ok(r) and r["turn_id"] == t]) for t in (1, 2, 3)},
                         "suffix_tokens": {f"T{t}": dist([r["suffix_tokens"] for r in hyper_rows if ok(r) and r["turn_id"] == t]) for t in (2, 3)},
                         "cap_reached": {f"T{t}": sum(bool(r["cap_reached"]) for r in hyper_rows if ok(r) and r["turn_id"] == t) for t in (1, 2, 3)}}
    out["failed_rows"] = [{"conversation_id": cid, "turn_id": r["turn_id"], "cause": r.get("failure"), "executed": r.get("executed")}
                          for cid, d in main.items() for r in d["rows"] if not ok(r)]

    # ---- 12. turn-1 store cost
    stores = [s for d in main.values() for s in d["stores"].values()]
    old_stores = [d["stores"][HC.impress_kind(d)] for cid, d in recorded.items() if cid in main]
    if stores:
        w = np.array([s["native_writer_wall_ms"] for s in stores], dtype=float)
        out["t1_store"] = {"conversations": len(stores), "store_seconds_mean": float(w.mean() / 1000), "store_seconds_p95": float(np.percentile(w, 95) / 1000),
                           "store_mb_mean": float(np.mean([s["file_logical_bytes"] for s in stores]) / MB),
                           "store_mb_max": float(max(s["file_logical_bytes"] for s in stores) / MB),
                           "written_mb_mean": float(np.mean([s["os_write_bytes"] for s in stores]) / MB),
                           "written_gb_total": float(sum(s["os_write_bytes"] for s in stores) / 1e9),
                           "reorder_seconds_mean": nanmean([(s.get("official_turn1") or {}).get("total_after_request_ms") for s in stores if s.get("official_turn1")]),
                           "activation_seconds_mean": nanmean([s["activation_ms"] for s in stores if s.get("activation_ms") is not None]),
                           "recorded_impress_store_seconds_mean": float(np.mean([s["native_writer_wall_ms"] for s in old_stores]) / 1000) if old_stores else None,
                           "recorded_impress_written_mb_mean": float(np.mean([s["os_write_bytes"] for s in old_stores]) / MB) if old_stores else None}
        for k in ("reorder_seconds_mean", "activation_seconds_mean"):
            if out["t1_store"][k] is not None:
                out["t1_store"][k] /= 1000
    out["conversation_seconds_mean"] = float(np.mean([d["seconds"] for d in main.values()]))
    out["written_by_phase_gb"] = {ph: float(sum(s["os_write_bytes"] for d in C.committed(res, model, ph, index).values() for s in d["stores"].values()) / 1e9)
                                  for ph in HC.PHASES if C.phase_dir(res, model, ph).exists()}
    out["quality"] = quality(res, model, identity)
    return out


# ---------------------------------------------------------------- markdown
def ci_ratio(r, d=3):
    return "—" if not r or r.get("ratio") is None else (f"{r['ratio']:.{d}f}" if r.get("ci_low") is None else f"{r['ratio']:.{d}f} [{r['ci_low']:.{d}f}, {r['ci_high']:.{d}f}]")


def ci_diff(e, d=1):
    return "—" if not e or e.get("mean") is None else (fx(e["mean"], d) if e.get("ci_low") is None else f"{e['mean']:+.{d}f} [{e['ci_low']:+.{d}f}, {e['ci_high']:+.{d}f}]")


def gates_md(r):
    G = r["gates"]
    gate, pilot = G["gate"], G["pilot"]

    def st(doc, key):
        if doc is None:
            return "미실행"
        g = doc.get("gates", {}).get(key)
        return "—" if g is None else ("PASS" if g.get("ok") else "FAIL")

    def detail(doc, key, what="mismatches"):
        g = (doc or {}).get("gates", {}).get(key)
        return "—" if g is None else f"{g.get('requests', '—')}요청 중 불일치 {g.get(what, '—')}"
    rows = []
    k = "gate1_output_equals_impress"
    rows.append(["1. 출력 동일성 (생성 토큰 T1–T3, layer별 선택 집합 T2–T3 = 같은 세션의 IMPRESS)", st(gate, k), st(pilot, k), f"4대화: {detail(gate, k)}; 파일럿: {detail(pilot, k)}"])
    k = "gate2_prefetch_off_equals_impress"
    rows.append(["2. prefetch 끔 ≡ IMPRESS (생성 토큰, 읽기 바이트, 읽기 호출 수)", st(gate, k), "(4대화에서만)", detail(gate, k)])
    k = "gate3_unread_rows_not_used"
    g3 = (gate or {}).get("gates", {}).get(k)
    rows.append(["3. 읽지 않은 KV를 쓰지 않음", st(gate, k), "(4대화에서만)",
                 f"{g3['requests']}요청 중 실패 {g3['guard_failures']}; 읽지 않은 행 {g3['totals']['unread_rows']:,}개 중 사용된 것 0, attention 호출 {g3['totals']['attention_calls']:,}회" if g3 else "—"])
    k = "gate4_repeat"
    g4 = (gate or {}).get("gates", {}).get(k)
    rows.append(["4. 같은 요청을 두 번 돌려 생성 토큰이 같음", st(gate, k), "(4대화에서만)",
                 (detail(gate, k) + f"; prefetch된 양이 달랐던 요청 {g4['prefetched_bytes']['requests_with_a_different_amount']} / {g4['prefetched_bytes']['requests']}") if g4 else "—"])
    k = "gate5_no_read_started_after_stop"

    def stop(doc):
        g = (doc or {}).get("gates", {}).get(k)
        if g is None:
            return "—"
        return "; ".join(f"{N[a]}: {v['layers_checked']:,} layer, prefetch 읽기 {v['prefetch_reads']:,}건, 멈춤 뒤 시작 {v['reads_started_after_stop']}건" for a, v in g["arms"].items())
    rows.append(["5. 멈춤 확인 (계산이 끝난 뒤 새로 시작된 prefetch 읽기 0건)", st(gate, k), st(pilot, k), f"4대화 — {stop(gate)} / 파일럿 — {stop(pilot)}"])
    k = "gate6_tokens_equal_recorded_impress"
    g6 = (pilot or {}).get("gates", {}).get(k)
    rows.append(["6. 파일럿의 HyperInfer 생성 토큰 = 기존 본실험의 IMPRESS 기록", "(파일럿에서만)", st(pilot, k),
                 f"{g6['requests']}요청 중 일치 {g6['equal']} (T1/T2/T3: " + "/".join(f"{g6['equal_per_turn'][t][0]}" for t in ("T1", "T2", "T3")) + ")" if g6 else "—"])
    tests = G["tests"]
    suites = (tests or {}).get("suites") or []
    rows.append(["7. 기존 테스트에 새 실패 0 (두 환경)", (tests or {}).get("status", "미실행"), "—",
                 "; ".join(f"transformers {s['transformers']}: {s['ran']}개 실행, 새 실패 {len(s['new_failures'])}, 이전부터 실패 {len(s['failed']) - len(s['new_failures'])}" for s in suites) or "—"])
    over = G["overflow"]
    rows.append(["8. 파일럿 표, 4096 초과 입력 확인 (기록용)", "—", (over or {}).get("status", "미실행"),
                 ("파일럿 안에 4096 초과 입력이 있음" if (over or {}).get("status") == "COVERED_BY_PILOT" else
                  f"따로 돌린 2대화(HyperInfer): {over.get('requests', '—')}요청, 실패 {len(over.get('failed_requests', []))}, 4096 초과 {over.get('requests_above_4096', '—')}" if over else "—")])
    L = [table(["Gate", "4대화", "파일럿 40대화", "내용"], rows, ["---"] * 4), ""]
    for name, doc in (("4대화", gate), ("파일럿", pilot)):
        if doc:
            t1 = doc.get("turn1_tokens_identical_across_arms") or {}
            L.append(f"- {name}: {doc['conversations']}대화, 요청 {doc['requests']}건, 실패한 요청 {len(doc['failed_requests'])}건, T1 생성 토큰이 모든 arm에서 같은 대화 {t1.get('identical', '—')} / {t1.get('conversations', '—')}.")
    for doc in (gate, pilot):
        for key in (doc or {}).get("failed", []):
            gg = doc["gates"].get(key, {})
            ex = gg.get("first_mismatches") or gg.get("first_violations") or gg.get("first_guard_failures") or doc.get("failed_requests") or []
            L.append(f"- **FAIL {doc['scale']} / {key}**: " + json.dumps(ex[:3], ensure_ascii=False)[:900])
    for s in suites:
        if s["failed"]:
            L.append(f"- transformers {s['transformers']} 환경에서 이전부터 실패하던 테스트(새 실패 아님): " + "; ".join(f"`{f}`" for f in s["failed"] if f not in s["new_failures"]))
            if s["new_failures"]:
                L.append(f"- **새 실패** (transformers {s['transformers']}): " + "; ".join(f"`{f}`" for f in s["new_failures"]))
    admit = G["admission"]
    L += ["", f"본실험 진입 판정: **{admit['status']}** ({admit['at_utc']})" if admit else "본실험 진입 판정: 아직 없음", ""]
    if admit and admit["status"] != "PASS":
        L += ["진입하지 못한 이유:", ""] + [f"- {x['where']} / {x['gate']}: " + json.dumps({k: v for k, v in x.items() if k not in ("where", "gate")}, ensure_ascii=False)[:1200]
                                     for x in admit["failed_gates"]] + [f"- 단계별 상태: {admit['parts']}", ""]
    return L


def dist_cell(d, digits=2):
    return "—" if not d else f"{d['mean']:.{digits}f} ({d['p50']:.{digits}f})"


def model_md(r, mt):
    L = [f"# {r['label']}", "", "## Gate", ""] + gates_md(r)
    L += [f"본실험: {r['main_conversations']} / {r['main_expected']} 대화 확정" + ("" if r["main_complete"] else " (**미완료**)") + ".", ""]
    M = r["methods"]
    # ---- 1
    L += ["## 1. TTFT (재사용 턴 T2–T3, ms)", "",
          table(["방법", "실행일", "T2 평균", "T2 p50", "T2 p95", "T3 평균", "T3 p50", "T3 p95", "T2–T3 평균", "T2–T3 p50", "T2–T3 p95", "SSD MB/followup", "읽기 호출 수", "요청 수"],
                [[m["method"], m["run_date"]] + [fx(m[k][s], 1) for k in ("T2", "T3", "T2–T3") for s in ("mean", "p50", "p95")]
                 + [fx(m["ssd_mb_per_followup"], 1), fx(m["read_calls_per_followup"], 1), m["followup_requests"]] for m in M]), "",
          "- HyperInfer는 이번 실행의 본실험(HyperInfer 단독), 나머지 네 방법은 기존 본실험(`results/convbench_main_20261122T103401Z`, 네 방법 회전)의 기록이다.", ""]
    if "paired" not in r:
        L += ["본실험 기록이 없어 이후 항목은 보고하지 않는다.", ""]
        return L
    # ---- 2
    L += ["## 2. 짝지은 비교 (같은 대화·같은 턴, 대화 단위 paired bootstrap 10,000회, seed 1234, 95% CI)", "",
          table(["비교", "TTFT 차이 (ms) [95% CI]", "TTFT 평균의 비 [95% CI]", "T2 차이 / 비", "T3 차이 / 비", "대화 수"],
                [[f"{p['a']} 대 {p['b']} ({p['a']} − {p['b']}, {p['a']} / {p['b']})", ci_diff(p["diff"]), ci_ratio(p["ratio"]),
                  f"{p['by_turn']['T2']['diff_ms']:+.1f} / {p['by_turn']['T2']['ratio']:.3f}", f"{p['by_turn']['T3']['diff_ms']:+.1f} / {p['by_turn']['T3']['ratio']:.3f}",
                  p["conversations"]] for p in r["paired"]]), ""]
    pt = (r["gates"]["pilot"] or {}).get("gate8_pilot_table")
    if pt:
        same = pt["hyperinfer_over_impress_same_run"]
        L += [f"같은 실행에서 잰 값(파일럿 {pt['conversations']}대화, FullLoad·IMPRESS·HyperInfer를 arm 순서 회전으로 함께 실행):", "",
              table(["방법", "TTFT T2–T3 평균", "p50", "p95", "T2 평균", "T3 평균", "SSD MB/followup", "읽기 호출 수", "요청 수", "실패"],
                    [[v["method"], fx(v["ttft_mean_ms"], 1), fx(v["ttft_p50_ms"], 1), fx(v["ttft_p95_ms"], 1), fx(v["ttft_mean_by_turn_ms"].get("T2"), 1),
                      fx(v["ttft_mean_by_turn_ms"].get("T3"), 1), fx(v["ssd_mb_per_followup"], 1), fx(v["read_calls_per_followup"], 1), v["followup_requests"],
                      v["failed_requests"]] for v in pt["methods"].values()]), "",
              f"- **같은 실행의 HyperInfer / IMPRESS = {ci_ratio(same)}** (차이 {fx(same.get('diff_ms'), 1)} ms [{fx(same.get('diff_ci_low'), 1)}, {fx(same.get('diff_ci_high'), 1)}], {same['n']}대화). "
              f"HyperInfer / FullLoad = {ci_ratio(pt['hyperinfer_over_fullload_same_run'])}.", ""]
        # ---- 3
        L += ["## 3. TTFT 연속성 (파일럿에서 다시 잰 값 대 기존 본실험의 같은 요청)", "",
              table(["방법", "이번 파일럿 평균 (ms)", "기존 본실험 평균 (ms)", "비 (이번 / 기존) [95% CI]", "차이 (ms) [95% CI]", "대화 수", "생성 토큰이 기존과 같은 요청"],
                    [[v["method"], fx(v["this_run_mean_ms"], 1), fx(v["main_experiment_mean_ms"], 1), ci_ratio(v),
                      f"{fx(v.get('diff_ms'), 1)} [{fx(v.get('diff_ci_low'), 1)}, {fx(v.get('diff_ci_high'), 1)}]", v["conversations"],
                      f"{v['tokens_equal_to_main_experiment']['equal']} / {v['tokens_equal_to_main_experiment']['requests']}"] for v in pt["continuity_with_main_experiment"].values()]), "",
              "- 이 비가 1에서 벗어난 만큼이 실행 간 변동(날짜, arm 구성)이다. 2절의 \"기존 기록 대비\" 비율은 이 변동을 포함한다.", ""]
    else:
        L += ["파일럿 결과가 없어 같은 실행의 HyperInfer / IMPRESS 비율과 TTFT 연속성(3절)은 보고하지 않는다.", ""]
    # ---- 4
    q = r["quality"]
    L += ["## 4. 품질 (ConvBench, 판정기 Llama-3.1-8B-Instruct; 승률 %, 분모 " + str(q["denominator"]) + ")", ""]
    rows = []
    for role in HC.FIVE:
        v = q["hyperinfer"] if role == HYPER else q["recorded"].get(role)
        if v is None:
            rows.append([N[role], "—", "—", "—", "—", "—"])
            continue
        rows.append([N[role], fx(v["S1"]), fx(v["S2"]), fx(v["S3"]), fx(v["Avg"]), "/".join(str(v["unresolved"][k]) for k in ("S1", "S2", "S3"))])
    L += [table(["방법", "S1", "S2", "S3", "평균", "판정 불가 (S1/S2/S3)"], rows), ""]
    if q["source"] == "recorded_impress":
        L += ["- **HyperInfer의 점수는 IMPRESS의 판정 결과를 그대로 쓴 것이다.** 본실험 1,731요청의 생성 토큰이 기존 IMPRESS 기록과 모두 일치해 판정을 다시 돌리지 않았다(5절).", ""]
    elif q["source"] == "judged":
        v = q.get("verdicts_on_identical_answers") or {}
        L += [f"- **HyperInfer의 점수는 이번 실행에서 새로 판정한 것이다**(동결한 판정기, 빈 cache; 새 판정문 {q['judge'].get('judged_now')}건, 2차 추출 {q['judge'].get('extractions_now')}건). "
              f"세 턴의 답이 기존 IMPRESS와 모두 같은 {v.get('conversations', '—')}대화 중 세 단계의 승패가 기존 IMPRESS 판정과 같은 대화: {v.get('same_verdicts', '—')}.", ""]
    else:
        L += ["- HyperInfer의 점수: 아직 없음(본실험 또는 기존 IMPRESS 기록과의 대조가 끝나지 않았다).", ""]
    if q["pairs"]:
        L += ["짝지은 비교(평균 점수 기준, 대화 단위 paired bootstrap 10,000회, seed 1234):", "",
              table(["비교", "평균 점수 차이 (pp) [95% CI]", "S1 차이", "S2 차이", "S3 차이", "판정 결과가 다른 대화 수", "대화 수"],
                    [[e["pair"], ci(e, 2), fx(e["stage_differences"]["S1"]), fx(e["stage_differences"]["S2"]), fx(e["stage_differences"]["S3"]),
                      e["conversations_with_different_outcome"], e["n"]] for e in q["pairs"]]), ""]
    # ---- 5
    idn = r["identity"]
    L += ["## 5. 출력 동일성", ""]
    if idn:
        L += [f"본실험 {idn['requests']:,}요청(T1–T3) 중 HyperInfer의 생성 토큰이 기존 IMPRESS 기록과 일치한 요청: **{idn['equal']:,} / {idn['requests']:,}** "
              f"(T1 {idn['equal_per_turn']['T1'][0]}/{idn['equal_per_turn']['T1'][1]}, T2 {idn['equal_per_turn']['T2'][0]}/{idn['equal_per_turn']['T2'][1]}, "
              f"T3 {idn['equal_per_turn']['T3'][0]}/{idn['equal_per_turn']['T3'][1]}). 결정: {('기존 IMPRESS의 판정 결과를 사용' if not idn['judge_needed'] else 'HyperInfer 답 전체를 새로 판정')}"
              + (" (리허설에서 판정 경로를 강제로 실행)" if idn.get("rehearsal_forced_judge") else "") + ".", ""]
        if idn["mismatches"]:
            ml = idn["mismatch_list"]
            first = sum(1 for m in ml if m["earlier_turns_of_this_conversation_equal"])
            L += [f"불일치 {idn['mismatches']}요청, {idn['conversations_with_a_mismatch']}대화. 그 대화에서 처음으로 달라진 요청은 {first}건이고(턴별: "
                  + ", ".join(f"T{t} {sum(1 for m in ml if m['earlier_turns_of_this_conversation_equal'] and m['turn_id'] == t)}" for t in (1, 2, 3))
                  + f"), 나머지 {len(ml) - first}건은 앞 턴이 이미 달라 이력(입력)이 달라진 요청이다.", "",
                  table(["대화", "턴", "앞 턴 일치", "생성 토큰 수 (HyperInfer / IMPRESS 기록)", "처음 달라진 위치", "비고"],
                        [[m["conversation_id"], m["turn_id"], "예" if m["earlier_turns_of_this_conversation_equal"] else "아니오",
                          "/".join(map(str, m["generated_tokens"])) if "generated_tokens" in m else "—", m.get("first_differing_position", "—"), m.get("problem", "")]
                         for m in ml[:60]], ["---"] * 6), ""]
            if len(ml) > 60:
                L += ["(처음 60건만 표시; 전체는 `gates/" + r["model"] + "/identity.json`)", ""]
            L += ["원인: 자동으로 확인되지 않았다. 확인된 사실은 다음과 같다. (1) 앞 턴까지 일치하다가 처음 갈린 요청은 입력 토큰열이 기존 IMPRESS 요청과 같다. "
                  "(2) 같은 세션의 IMPRESS와 생성 토큰·layer별 선택 집합이 일치하는지는 4대화와 파일럿에서만 확인했고(gate 1), 본실험에는 같은 세션의 IMPRESS가 없다. "
                  "(3) T1에서 갈린 요청은 저장된 KV를 쓰지 않는 픽셀 경로의 요청이다. 서로 다른 실행 사이의 BF16 계산 차이가 greedy 선택을 가른 것인지, "
                  "선택 집합이 달라진 것인지는 이 기록만으로는 구분되지 않는다.", ""]
    else:
        L += ["기존 IMPRESS 기록과의 대조가 아직 없다(본실험 미완료).", ""]
    # ---- 6
    P = r["prefetch"]
    mp = (mt or {}).get("prefetch") or {}
    md = mp.get("dist") or {}

    def col(s, f):
        return "—" if s is None else f(s)
    L += ["## 6. layer당 실측 (본실험, 재사용 턴; 평균 (p50))", "",
          table(["항목", "ConvBench T2", "ConvBench T3", "ConvBench T2–T3", "MT-GQA (T2–T3)"],
                [["GPU 계산 시간 (ms; prefetch 제출 ~ 다음 layer에서 GPU 동기화 완료)"] + [col(P[k], lambda s: dist_cell(s["dist"]["window_ms"])) for k in ("T2", "T3", "T2–T3")] + [dist_cell(md.get("window_ms"))],
                 ["다음 layer의 probe key 읽기 시간 (ms, worker)"] + [col(P[k], lambda s: dist_cell(s["dist"]["probe_read_ms"])) for k in ("T2", "T3", "T2–T3")] + [dist_cell(md.get("probe_read_ms"))],
                 ["K chunk 읽기 하나 (ms)"] + [col(P[k], lambda s: dist_cell(s["dist"]["k_read_ms"])) for k in ("T2", "T3", "T2–T3")] + [dist_cell(md.get("k_read_ms"))],
                 ["V chunk 읽기 하나 (ms)"] + [col(P[k], lambda s: dist_cell(s["dist"]["v_read_ms"])) for k in ("T2", "T3", "T2–T3")] + [dist_cell(md.get("v_read_ms"))],
                 ["계산 시간 안에 읽기를 마친 chunk 읽기 수 (layer당; K 또는 V 하나가 1건)"] + [col(P[k], lambda s: fx(s["reads_in_window_per_job"], 2)) for k in ("T2", "T3", "T2–T3")] + [fx(mp.get("reads_in_window_per_job"), 2)],
                 ["계산 시간 안에 K·V를 모두 마친 chunk 수 (layer당)"] + [col(P[k], lambda s: fx(s["chunks_in_window_per_job"], 2)) for k in ("T2", "T3", "T2–T3")] + [fx(mp.get("chunks_in_window_per_job"), 2)],
                 ["K·V가 모두 prefetch된 chunk 수 (layer당; 멈춤 때 진행 중이던 읽기 포함)"] + [col(P[k], lambda s: fx(s["chunks_prefetched_per_job"], 2)) for k in ("T2", "T3", "T2–T3")] + [fx(mp.get("chunks_prefetched_per_job"), 2)],
                 ["예측 chunk 수 (layer당)"] + [col(P[k], lambda s: fx(s["predicted_chunks_per_layer"], 1)) for k in ("T2", "T3", "T2–T3")] + [fx(mp.get("predicted_chunks_per_layer"), 1)],
                 ["예측 chunk를 전부 읽고 끝난 layer의 비율"] + [col(P[k], lambda s: pct(s["jobs_exhausted_share"])) for k in ("T2", "T3", "T2–T3")] + [pct(mp.get("jobs_exhausted_share"))],
                 ["요청 thread가 GPU 동기화에서 막힌 시간 (ms, layer당)"] + [col(P[k], lambda s: dist_cell(s["dist"]["sync_wait_ms"])) for k in ("T2", "T3", "T2–T3")] + [dist_cell(md.get("sync_wait_ms"))],
                 ["probe key가 필요한 시점에 이미 읽혀 있던 비율"] + [col(P[k], lambda s: pct(s["probe_ready_share"])) for k in ("T2", "T3", "T2–T3")] + [pct(mp.get("probe_ready_share"))],
                 ["표본 수 (요청 / layer)"] + [col(P[k], lambda s: f"{s['requests']:,} / {s['jobs']:,}") for k in ("T2", "T3", "T2–T3")] + [f"{mp.get('requests', 0):,} / {mp.get('jobs', 0):,}" if mp else "—"]]), "",
          "- MT-GQA 값은 `results/mt_gqa_hyperinfer_v2_20261125T034057Z/report.json`의 같은 항목이다(같은 집계 함수).", ""]
    both = P["T2–T3"]
    if both:
        L += ["layer별 평균 (T2–T3):", "",
              table(["layer", "계산 시간 (ms)", "probe key 읽기 (ms)", "예측 chunk", "창 안에 끝난 읽기", "창 안에 K·V가 다 끝난 chunk", "멈춤 뒤 끝난 읽기", "필요 chunk", "빠진 K chunk", "빠진 V chunk", "채움 비율", "fallback", "다음 layer와 Jaccard"],
                    [[x["layer"], fx(x["window_ms"], 2), fx(x["worker_probe_ms"], 2), fx(x["predicted_chunks"], 1), fx(x["reads_in_window"], 2), fx(x["chunks_in_window"], 2),
                      fx(x["reads_after_stop"], 2), fx(x["needed_chunks"], 1), fx(x["k_missing"], 1), fx(x["v_missing"], 1), pct(x["coverage"]), pct(x["fallback_rate"], 0),
                      fx(x["jaccard_next"], 2)] for x in both["per_layer"]]), ""]
    # ---- 7
    b = r.get("bytes_vs_impress") or {}
    if both:
        L += ["## 7. prefetch 통계 (본실험, 재사용 턴 요청 평균)", "",
              table(["항목", "ConvBench T2", "ConvBench T3", "ConvBench T2–T3", "MT-GQA"],
                    [["사용 집합 중 prefetch로 채워진 비율 (layer 1 이후)"] + [col(P[k], lambda s: pct(s["coverage"])) for k in ("T2", "T3", "T2–T3")] + [pct(mp.get("coverage"))],
                     ["prefetch된 것 중 실제로 쓴 비율 (토큰 기준)"] + [col(P[k], lambda s: pct(s["precision"])) for k in ("T2", "T3", "T2–T3")] + [pct(mp.get("precision"))],
                     ["prefetch로 읽은 양 (MB/followup)"] + [col(P[k], lambda s: fx(s["prefetched_mb_per_followup"], 1)) for k in ("T2", "T3", "T2–T3")] + [fx(mp.get("prefetched_mb_per_followup"), 1)],
                     ["읽었지만 쓰지 않은 양 (MB/followup)"] + [col(P[k], lambda s: fx(s["wasted_mb_per_followup"], 2)) for k in ("T2", "T3", "T2–T3")] + [fx(mp.get("wasted_mb_per_followup"), 2)],
                     ["동기로 읽은 양, K·V (MB/followup)"] + [col(P[k], lambda s: fx(s["sync_mb_per_followup"], 1)) for k in ("T2", "T3", "T2–T3")] + [fx(mp.get("sync_mb_per_followup"), 1)]]), ""]
        if b:
            mb = (mt or {}).get("bytes_vs_impress") or {}
            L += [f"- 총 읽기 바이트, IMPRESS 대비(같은 요청 {b['requests']:,}건, IMPRESS는 기존 본실험 기록): HyperInfer {fx(b['hyperinfer_mb'], 1)} MB 대 IMPRESS {fx(b['impress_mb'], 1)} MB "
                  f"(**×{b['ratio']:.4f}**). 읽기 호출 수 {fx(b['hyperinfer_read_calls'], 1)} 대 {fx(b['impress_read_calls'], 1)}. MT-GQA에서는 ×{fx(mb.get('ratio'), 4)}.",
                  "- 사용 집합은 선택된 이미지 토큰 + separator다. 채워진 토큰은 K와 V가 모두 prefetch된 chunk의 토큰으로 센다.", ""]
    # ---- 8
    D = r["decomposition"]
    mdc = ((mt or {}).get("decomposition") or {}).get("HyperInfer") or {}

    def hrow(label, key, digits=1, source="prefetch"):
        vals = []
        for k in ("T2", "T3", "T2–T3"):
            s = P[k] if source == "prefetch" else (D[k]["hyperinfer"] if D[k] else None)
            vals.append(fx(s.get(key), digits) if s else "—")
        mtv = mp.get(key) if source == "prefetch" else mdc.get(key)
        return [label] + vals + [fx(mtv, digits)]
    L += ["## 8. 임계 경로에 남은 시간 (본실험, 재사용 턴 요청 평균, ms)", "",
          table(["항목 (HyperInfer)", "ConvBench T2", "ConvBench T3", "ConvBench T2–T3", "MT-GQA"],
                [hrow("probe key 대기 (worker의 probe key 읽기가 끝나기를 기다림)", "probe_wait_ms", 2),
                 hrow("진행 중이던 읽기의 마무리 대기", "inflight_wait_ms", 2),
                 hrow("layer 0의 probe key 읽기 (동기)", "probe_sync_ms", 2),
                 hrow("선택 계산", "select_ms"),
                 hrow("빠진 KV 동기 읽기", "sync_read_ms"),
                 hrow("prefetch buffer에서 옮기는 host 복사", "copy_ms", 2),
                 hrow("GPU 적재 (cache 기록 호출)", "gpu_load_ms"),
                 hrow("이미지 뒤 텍스트의 계산 — layer별 계산 시간의 합 (31개 layer)", "window_ms"),
                 hrow("  그중 요청 thread가 GPU 동기화에서 막힌 시간", "sync_wait_ms"),
                 hrow("이미지 뒤 텍스트의 계산 — prefill 구간 − hook 합계 (기존 본실험의 정의)", "text_compute_prefill_minus_hook_ms", 1, "decomp"),
                 hrow("요청 thread의 읽기 대기 합계", "read_wait_ms", 1, "decomp"),
                 hrow("hook 합계", "hook_ms", 1, "decomp"),
                 hrow("prefill 구간", "prefill_ms", 1, "decomp"),
                 hrow("TTFT", "ttft_ms", 1, "decomp")]), ""]
    rows = []
    for k in ("T2", "T3", "T2–T3"):
        o = D[k]["impress_recorded"]
        if o:
            rows.append([k, fx(o["ttft_ms"], 1), fx(o["read_wait_ms"], 1), fx(o["probe_wait_ms"], 2), fx(o["hook_ms"], 1), fx(o["text_compute_prefill_minus_hook_ms"], 1), fx(o["prefill_ms"], 1), o["requests"]])
    if rows:
        L += ["기존 본실험의 IMPRESS(같은 요청):", "",
              table(["턴", "TTFT", "읽기 대기 합계", "그중 probe key 대기", "hook 합계", "prefill − hook", "prefill 구간", "요청 수"], rows), "",
              "- 기존 IMPRESS 행에는 선택 계산, GPU 적재, layer별 계산 시간이 따로 기록돼 있지 않다.",
              "- layer별 계산 시간은 prefetch를 제출한 시각부터 다음 layer hook 첫머리의 GPU 동기화가 끝난 시각까지이며, 마지막 layer와 출력층은 포함하지 않는다. "
              "hook 합계에는 직전 layer의 GPU 계산을 기다린 시간(동기화에서 막힌 시간)이 들어 있으므로 \"prefill − hook\"은 실제 계산 시간보다 작게 나온다. 두 정의를 함께 적은 이유다.", ""]
    if both:
        L += [f"I/O ratio = (probe key 대기 + 진행 중이던 읽기의 마무리 대기 + layer 0의 probe key 읽기 + 빠진 KV 동기 읽기) / (worker 포함 전체 pread 시간): "
              f"**{fx(both['io_ratio_pooled'], 3)}** (요청별 평균 {fx(both['io_ratio_request_mean'], 3)}; MT-GQA {fx(mp.get('io_ratio_pooled'), 3)}).", ""]
    # ---- 9
    bins = r.get("bins")
    L += ["## 9. 이미지 뒤 텍스트 길이별 (본실험, 재사용 턴 요청을 이미지 뒤 토큰 수 사분위로 나눔)", ""]
    if bins:
        L += [table(["구간", "이미지 뒤 토큰 수", "요청 수 (T2 / T3)", "layer당 GPU 계산 시간 (ms)", "창 안에 끝난 읽기 (layer당)", "prefetch로 채워진 비율", "HyperInfer TTFT (ms)", "기존 IMPRESS TTFT (ms)", "HyperInfer / IMPRESS"],
                    [[f"Q{x['bin']}", f"{x['suffix_min']}–{x['suffix_max']} (평균 {x['suffix_mean']:.0f})", f"{x['requests']} ({x['t2']} / {x['t3']})", fx(x["window_ms_per_layer"], 2),
                      fx(x["reads_in_window_per_layer"], 2), pct(x["coverage"]), fx(x["hyperinfer_ttft_paired_ms"], 1), fx(x["impress_ttft_ms"], 1), fx(x["ratio"], 3)]
                     for x in bins["bins"] if x["requests"]]), "",
              f"- 구간 경계(사분위): {', '.join(f'{e:.0f}' for e in bins['edges'])} 토큰. TTFT 두 열과 비율은 기존 IMPRESS 기록이 있는 같은 요청끼리의 평균과 합의 비다.", ""]
    else:
        L += ["요청 수가 부족해 구간을 나누지 않았다.", ""]
    # ---- 10
    st = r.get("impress_stats") or {}
    hs, isr = st.get("hyperinfer"), st.get("impress_recorded")
    if hs and both:
        L += ["## 10. 선택 통계 (본실험, 재사용 턴)", "",
              table(["항목", "HyperInfer (이번 실행)", "IMPRESS (기존 본실험, 같은 요청)", "MT-GQA HyperInfer"],
                    [["fallback 비율 (layer 단위)", pct(hs["fallback_rate"]), pct(isr["fallback_rate"]) if isr else "—", "—"],
                     ["평균 Jaccard (probe head 쌍)", fx(hs["mean_jaccard"], 3), fx(isr["mean_jaccard"], 3) if isr else "—", "—"],
                     ["선택에 필요한 K·V chunk 비율 — 전체", pct(hs["kv_chunk_fraction"]), pct(isr["kv_chunk_fraction"]) if isr else "—", "—"],
                     ["선택에 필요한 K·V chunk 비율 — fallback이 아닌 layer만", pct(hs["kv_chunk_fraction_non_fallback_layers"]), pct(isr["kv_chunk_fraction_non_fallback_layers"]) if isr else "—", "—"],
                     ["실제로 읽은 K·V 바이트 / 전체 K·V 바이트 (읽고 쓰지 않은 것 포함)", pct(b.get("hyperinfer_kv_read_fraction")), pct(b.get("impress_kv_read_fraction")), "—"],
                     ["인접 layer 선택 집합의 Jaccard — 전체 쌍", fx(both["jaccard_all_pairs"], 3), "—", fx(mp.get("jaccard_all_pairs"), 3)],
                     ["인접 layer 선택 집합의 Jaccard — 둘 다 fallback이 아닌 쌍", f"{fx(both['jaccard_consensus_pairs'], 3)} (전체 쌍의 {pct(both['consensus_pair_share'])})", "—",
                      f"{fx(mp.get('jaccard_consensus_pairs'), 3)} (전체 쌍의 {pct(mp.get('consensus_pair_share'))})"]]), "",
              "- 인접 layer Jaccard는 separator를 뺀 이미지 토큰 집합으로 계산했다(기존 IMPRESS 행에는 layer별 선택 집합이 기록돼 있지 않다).", ""]
    # ---- 11
    a = r["above_4096"]
    sel = r["gates"]["overflow_selection"]
    L += ["## 11. 4096 토큰을 넘는 입력", "",
          f"- 본실험의 HyperInfer 요청 중 입력이 4096 토큰을 넘은 것: **{a['count']}건**, 그중 실패 {a['failed']}건 (기존 본실험의 IMPRESS: {a['recorded_impress_count']}건). 입력을 자르지 않고 그대로 실행했다.",
          f"- 입력 토큰 수 T1/T2/T3 평균 {'/'.join(fx(a['input_tokens'][t]['mean'], 0) for t in ('T1', 'T2', 'T3'))}, 최대 {'/'.join(fx(a['input_tokens'][t]['max'], 0) for t in ('T1', 'T2', 'T3'))}; "
          f"이미지 뒤 토큰 수 T2/T3 평균 {'/'.join(fx(a['suffix_tokens'][t]['mean'], 0) for t in ('T2', 'T3'))}, 최대 {'/'.join(fx(a['suffix_tokens'][t]['max'], 0) for t in ('T2', 'T3'))}; "
          f"생성 상한(1024) 도달 T1/T2/T3 {'/'.join(str(a['cap_reached'][t]) for t in ('T1', 'T2', 'T3'))}건."]
    if sel:
        over = r["gates"]["overflow"] or {}
        if sel["needed"]:
            chosen = ", ".join(f"{x['conversation_id']}(기존 IMPRESS 기록의 최대 입력 {x['recorded_impress_largest_input_tokens']:,} 토큰)" for x in sel.get("selected", []))
            L += [f"- 파일럿의 HyperInfer 입력에는 4096 토큰을 넘는 것이 없었다. 기존 IMPRESS 기록에서 입력이 가장 긴 대화를 골라 HyperInfer로 따로 돌렸다: {chosen} — "
                  f"{over.get('requests', '—')}요청, 실패 {len(over.get('failed_requests', []))}건, 4096 초과 {over.get('requests_above_4096', '—')}건"
                  + (", 생성 토큰이 기존 IMPRESS 기록과 같은 요청 " + f"{sum(bool(x['tokens_equal_recorded_impress']) for x in over['rows'])} / {len(over['rows'])}" if over.get("rows") else "") + "."
                  + (" 이 모델은 기존 기록에 4096 토큰을 넘는 입력이 없어, 입력이 가장 긴 대화로 대신했다." if sel.get("note") else "")]
        else:
            L += [f"- 파일럿의 HyperInfer 입력에 4096 토큰을 넘는 것이 {len(sel['pilot_requests_above_4096'])}건 있어, 따로 돌린 대화는 없다."]
    L += [""]
    if a["requests"]:
        L += [table(["대화", "턴", "상태", "입력 토큰 수", "생성 토큰 수", "TTFT (ms)", "기존 IMPRESS TTFT (ms)", "생성 토큰이 기존 IMPRESS와 같음"],
                    [[x["conversation_id"], x["turn"], x["status"], x["input_tokens"], x["generated_tokens"], fx(x["ttft_ms"], 1), fx(x["recorded_impress_ttft_ms"], 1),
                      "예" if x["tokens_equal_recorded_impress"] else "아니오"] for x in a["requests"]]), ""]
    if r["failed_rows"]:
        L += [f"실패한 요청 {len(r['failed_rows'])}건:", "", table(["대화", "턴", "원인", "실행됨"], [[f["conversation_id"], f["turn_id"], str(f["cause"])[:160], "예" if f["executed"] else "아니오(앞 턴 실패)"]
                                                                                        for f in r["failed_rows"][:40]], ["---"] * 4), ""]
    else:
        L += [f"본실험에서 실패한 요청: 0건.", ""]
    # ---- 12 (model part)
    s = r.get("t1_store")
    if s:
        hy = next(m for m in M if m["method"] == N[HYPER]); im = next(m for m in M if m["method"] == N[IMP])
        L += ["## 12. T1 저장 비용 (본실험, 대화당)", "",
              table(["방법", "저장 시간 평균 (s)", "저장 시간 p95 (s)", "저장 크기 평균 (MB)", "저장 크기 최대 (MB)", "쓴 양 평균 (MB)", "T1 TTFT 평균 (ms)", "대화 수"],
                    [[N[HYPER], fx(s["store_seconds_mean"], 2), fx(s["store_seconds_p95"], 2), fx(s["store_mb_mean"], 1), fx(s["store_mb_max"], 1), fx(s["written_mb_mean"], 1),
                      fx(hy["t1_ttft_mean_ms"], 1), s["conversations"]],
                     [N[IMP] + " (기존 본실험, 같은 대화)", fx(s["recorded_impress_store_seconds_mean"], 2), "—", "—", "—", fx(s["recorded_impress_written_mb_mean"], 1),
                      fx(im["t1_ttft_mean_ms"], 1), "—"]]), "",
              f"- 저장 형식이 IMPRESS와 같다: 전체 KV를 쓴 뒤 T1 attention으로 정한 순서로 다시 쓴다(재배치 평균 {fx(s['reorder_seconds_mean'], 2)} s 포함). store 활성화 평균 {fx(s['activation_seconds_mean'], 2)} s. "
              f"이 모델의 본실험에서 store로 쓴 양 합계 {fx(s['written_gb_total'], 1)} GB. 단계별 store 쓰기량(GB): "
              + ", ".join(f"{ph} {v:.1f}" for ph, v in r["written_by_phase_gb"].items()) + f". 대화당 평균 {fx(r['conversation_seconds_mean'], 1)} s.", ""]
    return L


def run_md(res, tables):
    res = Path(res)
    pre = C.read_json(res / "prerun.json") or {}
    counter = C.read_json(res / "ssd_counter.json") or {}
    status = C.read_json(res / "status.json") or {}
    steps = C.read_jsonl(res / "driver_steps.jsonl")
    inv = C.read_jsonl(res / "driver_invocations.jsonl")
    resume = C.read_jsonl(res / "resume_log.jsonl")
    post = C.read_json(res / "gates" / "posthoc.json")
    L = ["# 실행 전체", "", "## 12. SSD 쓰기량과 소요 시간", ""]
    exp = (pre.get("expected_ssd_write") or {}).get("total_tb")
    stores = sum(v for r in tables["models"].values() for v in (r.get("written_by_phase_gb") or {}).values())
    L += [table(["항목", "값"], [
        ["예상 SSD 쓰기량 (실행 전 계산)", f"{fx(exp, 3)} TB" if exp is not None else "—"],
        ["실측: 장치 전체 쓰기 카운터 (드라이버 첫 시작 이후, 모든 프로세스)", f"{(counter.get('written_bytes') or 0) / 1e12:.3f} TB"],
        ["실측: store 쓰기 호출의 합 (확정된 대화)", f"{stores / 1000:.3f} TB"],
        ["예상 소요 시간 (실행 전 계산)", f"{fx((pre.get('expected_duration') or {}).get('total_hours'), 1)} h"],
        ["실측 소요 시간 (단계 합)", f"{sum(s['seconds'] for s in steps) / 3600:.2f} h"],
        ["시작 – 종료", f"{status.get('first_started_utc', '—')} – {status.get('finished_utc') or status.get('updated_utc', '—')}"],
        ["SMART", f"실행 전 {(C.read_json(res / 'smart_before.json') or {}).get('status', '—')}, 실행 후 {(C.read_json(res / 'smart_after.json') or {}).get('status', '—')}"],
    ], ["---", "---:"]), ""]
    by_step = {}
    for s in steps:
        by_step.setdefault(s["step"], [0.0, 0])
        by_step[s["step"]][0] += s["seconds"]; by_step[s["step"]][1] += 1
    if by_step:
        L += [table(["단계", "시간 (분)", "프로세스 시작 횟수"], [[k, fx(v[0] / 60, 1), v[1]] for k, v in by_step.items()]), ""]
    L += [f"- 드라이버 실행 횟수: {len(inv)}회 (이어서 실행 {max(0, len(inv) - 1)}회)" + ("; 시각: " + ", ".join(i["at_utc"] for i in inv) if len(inv) > 1 else "") + "."]
    if resume:
        L += [f"- 다시 실행한 대화 {len(resume)}건:", "", table(["시각", "모델", "단계", "대화", "사유", "버린 요청 기록 수"],
                                                       [[e["at_utc"], e["model"], e["phase"], e["conversation_id"], e["mode"], e["rows_discarded"]] for e in resume], ["---"] * 6), ""]
    else:
        L += ["- 다시 실행한 대화: 없음.", ""]
    if post:
        L += [f"- 기존 파일 보호: 실행 전과 종료 뒤의 `results/`, `runs/`, `kvstore*` 파일 목록(크기, 수정 시각; {post['before']['files']:,}개) 비교 — "
              + ("**변경 없음**." if post["existing_files_unchanged"] else f"**차이 있음**: {json.dumps((post.get('changed') or {}).get('counts'), ensure_ascii=False)} (목록은 `gates/posthoc.json`).")
              + f" 동결한 소스 파일 중 바뀐 것: {len(post['frozen_source_changed'])}개.", ""]
    else:
        L += ["- 기존 파일 보호 확인(사후 비교)은 아직 실행되지 않았다.", ""]
    L += ["## 13. 코드, 지시와 다르게 했거나 지시에 없던 것, 측정하지 않은 것과 한계", "",
          "### 추가한 코드 (기존 파일은 수정하지 않았다)", "",
          "- `scripts/convbench_hyperinfer_common.py` — 이 실행의 arm·단계 표. 기존 본실험의 공통 모듈을 사본으로 불러 이 실행의 표를 넣는다(원래 모듈은 그대로).",
          "- `scripts/231_eval_convbench_hyperinfer.py` — 실행기. 기존 본실험의 ConvBench 실행기(`scripts/217`)를 그대로 불러 쓰고, HyperInfer 요청은 MT-GQA 실행의 요청 함수(`scripts/228`·`225`)를 그대로 부른다. freeze, 실행 전 점검.",
          "- `scripts/232_validate_convbench_hyperinfer.py` — gate 1–8, admission, 기존 IMPRESS 기록과의 대조, 사후 확인.",
          "- `scripts/233_report_convbench_hyperinfer.py` — 이 보고서(동결 대상 아님).",
          "- `scripts/234_convbench_hyperinfer_driver.py`, `scripts/run_convbench_hyperinfer.sh` — 분리 실행 드라이버.",
          "- `scripts/235_convbench_hyperinfer_judge.py` — 판정이 필요할 때 기존 판정 코드(`scripts/219`·`214`)를 HyperInfer 답에 적용.",
          "- `tests/test_convbench_hyperinfer.py` — 새 단위 테스트. `docs/convbench_hyperinfer_contract.md` — 계약.",
          "- 방법 코드(`mmimpress/`)와 기존 스크립트·테스트는 바꾸지 않았다.", "",
          "### 지시와 다르게 했거나 지시에 없던 것 (계약 8절에 실행 전 고지)", "",
          "1. gate 2·3·4는 추가 arm이 필요해 4대화에서만 판정했다. 파일럿은 지시대로 세 arm이고 gate 1·5·6을 거기서 판정했다.",
          "2. gate 7은 두 환경(transformers 4.57.6, 5.13.1)의 전체 테스트로 봤다.",
          "3. 4대화·파일럿에서는 IMPRESS와 HyperInfer arm들이 한 대화의 store 하나를 함께 읽는다(arm 회전에서 먼저 오는 arm이 자기 T1으로 쓴다). 본실험에서는 HyperInfer가 자기 T1으로 쓴다.",
          "4. \"warm-up 없음\"은 미리 만든 KV 없이 T1을 픽셀부터 계산한다는 뜻으로 읽었다. 모델을 올린 직후의 프로세스 warm-up 요청 1건(측정 제외)은 기존 본실험과 같이 뒀다.",
          "5. 4096 초과 대화는 기존 본실험의 IMPRESS 기록에서 입력이 가장 긴 2대화로 골랐다. LLaVA-1.5는 4096을 넘는 입력이 없어 가장 긴 2대화로 대신했다.",
          "6. 판정이 필요한 경우 빈 cache에서 HyperInfer 답 전체를 새로 판정하도록 했다(기존 판정 cache 재사용 없음, 판정기 재현 확인 없음).",
          "7. 텍스트 길이 구간은 모델별 HyperInfer 재사용 턴 요청의 이미지 뒤 토큰 수 사분위다.",
          "8. 어느 한쪽에 실패한 요청이 있는 대화는 짝지은 비교에서 뺐다(건수는 각 표의 대화 수).", "",
          "### 측정하지 않은 것과 한계", "",
          "- HyperInfer는 이번 실행, 나머지 네 방법은 2026-11-22–24의 기존 본실험 기록이다. 서로 다른 날의 TTFT를 짝지었으므로 실행 간 변동이 섞인다(3절). 기존 본실험은 네 방법 회전, 이번 본실험은 HyperInfer 단독이다. 같은 실행 안의 비교는 파일럿 40대화뿐이다.",
          "- ReComp, FullLoad, IMPRESS, Ours는 다시 돌리지 않았다. 본실험에는 같은 세션의 IMPRESS가 없으므로 본실험의 출력 동일성은 기존 IMPRESS 기록과의 대조다(layer별 선택 집합의 대조는 4대화와 파일럿에서만).",
          "- 기존 IMPRESS 행에는 선택 계산, GPU 적재, layer별 계산 시간, layer별 선택 집합이 기록돼 있지 않아 그 항목은 HyperInfer 값만 있다.",
          "- layer별 GPU 계산 시간은 다음 layer의 hook에서 GPU 동기화가 끝난 시각으로 재므로, 요청 thread가 깨어나는 시간이 들어 있다. 마지막 layer의 계산 시간은 재지 않았다.",
          "- 캐시 계층 없음, cold SSD(요청 전 page cache drop), worker 1개, GPU 복사는 요청 thread — MT-GQA 실행과 같은 조건이며 논문의 설정(GPU/CPU 캐시, warm-up)과 다르다(`docs/hyperinfer_port_contract.md`).",
          "- 품질 점수가 IMPRESS의 판정 결과를 쓴 것인 모델에서는 HyperInfer 답을 따로 판정하지 않았다(생성 토큰이 모두 같으므로 판정 입력이 같다).",
          "- SMART 값은 root 권한이 없어 읽지 못했다(커널 쓰기 카운터만 기록).", ""]
    return L


def report(res):
    res = Path(res)
    index, _ = C.load_index()
    status = C.read_json(res / "status.json") or {}
    config = C.read_json(res / "config.json") or {}
    try:
        mt_all = json.loads((HC.MTGQA_RES / "report.json").read_text())["models"]
    except (OSError, ValueError, KeyError):
        mt_all = {}
    tables = {"run": res.name, "written_utc": C.utc(), "models": {}, "errors": []}
    body = []
    for model in C.MODELS:
        try:
            r = model_report(res, model, index)
            tables["models"][model] = r
            body += model_md(r, mt_all.get(HC.MTGQA_MODEL[model]))
        except Exception as exc:      # noqa: BLE001  (the report is written with what could be computed)
            tables["errors"].append({"model": model, "error": repr(exc), "traceback": traceback.format_exc()[-3000:]})
            body += [f"# {C.MODEL_NAME[model]}", "", f"**보고서 생성 중 오류**: `{exc!r}` (report_tables.json의 errors 참조)", ""]
    head = [f"# ConvBench HyperInfer 측정 보고서 — {res.name}", "",
            f"작성 {C.utc()} · 상태 `{status.get('stage', '—')}` · 계약 `docs/convbench_hyperinfer_contract.md` (SHA256 {str(config.get('contract_sha256', ''))[:12]}…)", "",
            "측정 보고이며 GO/NO-GO 판정은 없다. 방법 이름은 ReComp, FullLoad, IMPRESS, HyperInfer, Ours.", "",
            "- HyperInfer: IMPRESS와 같은 선택·같은 저장 형식에, 현재 layer의 계산 중에 다음 layer의 probe key와 예측 chunk를 읽고, 계산이 끝나면 새 읽기를 시작하지 않으며, 선택 뒤 빠진 KV만 동기로 읽는다(MT-GQA 실행 `results/mt_gqa_hyperinfer_v2_20261125T034057Z`의 구현과 설정).",
            "- 본실험에서 돌린 방법은 HyperInfer 하나다(모델당 577대화 × 3턴 = 1,731요청). ReComp, FullLoad, IMPRESS, Ours는 기존 본실험 `results/convbench_main_20261122T103401Z`의 요청별 기록을 같은 대화·같은 턴끼리 짝지었다.",
            "- 공통: NF4/BF16/eager, FP16 SSD KV, greedy, 생성 상한 1024, T1은 픽셀부터 계산, T2–T3는 자기 생성 이력, 요청 전 page cache drop.", "",
            "## 요약", ""]
    rows = []
    for model, r in tables["models"].items():
        p = {(x["a"], x["b"]): x for x in r.get("paired", [])}
        hy = next((m for m in r["methods"] if m["method"] == N[HYPER]), None)
        pt = ((r["gates"]["pilot"] or {}).get("gate8_pilot_table") or {}).get("hyperinfer_over_impress_same_run")
        both = (r.get("prefetch") or {}).get("T2–T3")
        idn = r.get("identity")
        q = r.get("quality") or {}
        rows.append([r["label"], (r["gates"]["admission"] or {}).get("status", "—"), fx(hy["T2–T3"]["mean"], 1) if hy else "—",
                     ci_ratio((p.get((N[HYPER], N[IMP])) or {}).get("ratio")), ci_ratio(pt), ci_ratio((p.get((N[OURS], N[HYPER])) or {}).get("ratio")),
                     fx(both["dist"]["window_ms"]["mean"], 2) if both else "—", pct(both["coverage"]) if both else "—",
                     f"{idn['equal']:,} / {idn['requests']:,}" if idn else "—",
                     {"recorded_impress": "IMPRESS 판정 사용", "judged": "새로 판정"}.get(q.get("source"), "—"),
                     fx(q["hyperinfer"]["Avg"]) if q.get("hyperinfer") else "—"])
    head += [table(["모델", "gate 1–7", "HyperInfer TTFT T2–T3 평균 (ms)", "HyperInfer / IMPRESS (기존 기록 대비)", "HyperInfer / IMPRESS (같은 실행, 파일럿)", "Ours / HyperInfer",
                    "layer당 GPU 계산 시간 (ms)", "prefetch로 채워진 비율", "생성 토큰 = 기존 IMPRESS", "품질 점수의 출처", "품질 평균"], rows), ""]
    if status.get("last_error"):
        head += [f"마지막 오류: `{json.dumps(status['last_error'], ensure_ascii=False)[:600]}`", ""]
    md = head + body + run_md(res, tables)
    if tables["errors"]:
        md += ["# 보고서 생성 오류", ""] + [f"- {e['model']}: `{e['error']}`" for e in tables["errors"]] + [""]
    (res / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    C.atomic_json(res / "report_tables.json", plain(tables))
    print(json.dumps({"report": str(res / "report.md"), "errors": len(tables["errors"])}), flush=True)
    return tables


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", type=Path, required=True)
    args = p.parse_args()
    started = time.time()
    tables = report(args.res.resolve())
    print(json.dumps({"seconds": round(time.time() - started, 1)}), flush=True)
    sys.exit(0 if not tables["errors"] else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Report of the PyramidKV beta=20 MT-GQA HOLD add-on run
(contract docs/layer_budget_kv25_pyrbeta_contract.md, section 6).

Independent HOLD audit (164 audit on every committed image), within-run paired
analysis, mechanical decisions (primary per vector, floor effect), G-det on all
U_packed requests, cross-run paired accuracy vs the v2 HOLD arms (only when
G-det == 100 %), question-type deltas, per-layer k_l / byte table, sessions,
protection after, deletion log, figure, REPORT.md / REPRODUCE.md.

Usage: --run RUN
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from mmimpress.prefix25 import sha
from mmimpress.recompute_kv25 import paired_bootstrap, sha_manifest, stat_inventory

PK = "ours_prefixstore_kv25_"
SEED = 1234
V2_RUN = ROOT / "runs/layer_budget_kv25_v2_20261106T073502Z"
V2_RES = ROOT / "results/layer_budget_kv25_v2_20261106T073502Z"
U = f"{PK}U_packed"
FLOOR = f"{PK}PYR-beta_packed"
RAW = f"{PK}PYR-beta-raw_packed"
V2_ARMS = [f"{PK}PYR_packed", f"{PK}A1-2_packed", f"{PK}VLC_offline_packed", "recompute", "fullload", U]
QTYPE = (      # identical to scripts/159 (v2 REPORT) / scripts/155
    ("yes/no", re.compile(r"^(is|are|do|does|was|were|has|have|can|could|did|will|would)\b", re.I)),
    ("color", re.compile(r"\bcolou?r\b", re.I)),
    ("count", re.compile(r"\bhow many\b|\bnumber of\b", re.I)),
    ("spatial", re.compile(r"\b(left|right|above|below|behind|front|under|near|beside|top|bottom|side|between|next to)\b", re.I)),
    ("material/attribute", re.compile(r"\b(material|made of|shape|size|texture|large|small|tall|short|thin|thick)\b", re.I)),
    ("object", re.compile(r"^(what|which|who)\b", re.I)),
)


def qtype(question):
    for name, pattern in QTYPE:
        if pattern.search(question):
            return name
    return "other"


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def jl(path):
    with Path(path).open() as f:
        return [json.loads(l) for l in f if l.strip()]


def _json_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not serializable: {type(o).__name__}")


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=1, allow_nan=False, default=_json_default, ensure_ascii=False); f.write("\n"); f.flush(); os.fsync(f.fileno())


def csv_write(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def name(method):
    if method == "recompute":
        return "ReComp"
    if method == "fullload":
        return "FullLoad"
    if method.startswith(PK):
        return method[len(PK):]
    return method


def stats(v):
    a = np.asarray(v, dtype=float)
    return float(a.mean()), float(np.median(a)), float(np.percentile(a, 95))


def md_table(head, rows):
    return "| " + " | ".join(head) + " |\n|" + "|".join("---" for _ in head) + "|\n" + "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in rows) + "\n"


def slim(r):
    res = r["result"]
    return {"method_id": r["method_id"], "image_id": r["image_id"], "dialog_id": r["dialog_id"], "turn_id": r["turn_id"], "score": r["score"],
            "prediction": r["prediction"], "question": r["question"], "ttft_ms": r["ttft_ms"], "request_e2e_ms": r["request_e2e_ms"],
            "ssd_read_bytes": r["ssd_read_bytes"], "ssd_preads": r["ssd_preads"], "pread_calls": r.get("pread_calls", 0), "preadv_calls": r.get("preadv_calls", 0),
            "global_dialog_ordinal": r["global_dialog_ordinal"], "k_per_layer": r.get("k_per_layer"), "N_content": r["N_content"],
            "tokens": res["generated_token_ids"], "chunk_io_ms": res.get("chunk_io_ms"), "scatter_ms": res.get("scatter_ms"),
            "packed_read_ms": res.get("packed_read_ms"), "prefill_ms": res.get("prefill_ms")}


# ------------------------------------------------------------------- HOLD
def load_and_audit(hold, config, manifest, H):
    rows = []; attempts = {}
    audit = {"images": 0, "requests": 0, "failures": [], "per_image_status": {}, "checks": 0}
    for cp in sorted((hold / "main" / "images").glob("*/COMMITTED.json")):
        doc = json.loads(cp.read_text()); folder = cp.parent / doc["attempt_id"]
        if sha(folder / "raw.jsonl") != doc["raw_sha256"]:
            raise RuntimeError("committed raw hash mismatch: " + str(folder))
        full = jl(folder / "raw.jsonl")
        receipt = H.audit_image_rows(full, manifest, config, "main", cp.parent.name, folder)
        audit["images"] += 1; audit["requests"] += len(full); audit["checks"] += receipt["checks"]
        audit["per_image_status"][cp.parent.name] = receipt["status"]
        if receipt["status"] != "PASS":
            audit["failures"].append({"image_id": cp.parent.name, "failures": receipt["failures"][:5]})
        rows += [slim(r) for r in full]; attempts[cp.parent.name] = folder
    audit["expected_requests"] = manifest["expected_requests"]["main"]
    audit["expected_images"] = 398
    failed_attempts = sorted(str(p.parent.relative_to(hold)) for p in (hold / "main" / "images").glob("*/attempt_*/failure.json"))
    audit["failed_attempt_records"] = failed_attempts
    audit["status"] = "PASS" if not audit["failures"] and audit["requests"] == audit["expected_requests"] and audit["images"] == 398 else "FAIL"
    return rows, attempts, audit


def v2_rows(methods):
    rows = []
    for cp in sorted((V2_RUN / "hold" / "main" / "images").glob("*/COMMITTED.json")):
        doc = json.loads(cp.read_text()); raw = cp.parent / doc["attempt_id"] / "raw.jsonl"
        if sha(raw) != doc["raw_sha256"]:
            raise RuntimeError("v2 committed raw hash mismatch: " + str(raw))
        with raw.open() as f:
            for line in f:
                r = json.loads(line)
                if r["method_id"] in methods:
                    rows.append(slim(r))
    return rows


def index(rows):
    by = defaultdict(dict)
    for r in rows:
        by[r["method_id"]][(r["dialog_id"], r["turn_id"])] = r
    return by


def paired(by_a, by_b, images, img_of, flt, key="score"):
    acc = {img: [0.0, 0.0, 0] for img in images}
    for k, v in by_a.items():
        if flt(k, v):
            e = acc[img_of[k[0]]]; e[0] += v[key]; e[1] += by_b[k][key]; e[2] += 1
    per = [tuple(e) for e in acc.values() if e[2] > 0]
    d = paired_bootstrap(per, 10000, SEED)
    d["images_with_data"] = len(per); d["n"] = int(sum(e[2] for e in per))
    return d


def cells(by_a, by_b, flt):
    c = {"both": 0, "a_only": 0, "b_only": 0, "neither": 0}; agree = []
    for k, v in by_a.items():
        if not flt(k, v):
            continue
        x, y = v["score"], by_b[k]["score"]
        c["both" if x and y else "a_only" if x else "b_only" if y else "neither"] += 1
        agree.append(v["prediction"] == by_b[k]["prediction"])
    return c, float(np.mean(agree))


FOLLOW = lambda k, v: k[1] > 1
ALL = lambda k, v: True


def hold_analysis(hold, rows, config, attempts, res, H):
    methods = config["methods"]
    by = index(rows)
    images = sorted({r["image_id"] for r in rows})
    img_of = {r["dialog_id"]: r["image_id"] for r in rows}
    quality = []
    for m in methods:
        d = by[m]
        turns = {t: [v["score"] for k, v in d.items() if k[1] == t] for t in (1, 2, 3)}
        follow = turns[2] + turns[3]; allv = turns[1] + follow
        quality.append({"method": m, "name": name(m), "T1_pct": 100 * np.mean(turns[1]), "T2_pct": 100 * np.mean(turns[2]), "T3_pct": 100 * np.mean(turns[3]),
                        "all_pct": 100 * np.mean(allv), "all_correct": int(sum(allv)), "all_n": len(allv),
                        "followup_pct": 100 * np.mean(follow), "followup_correct": int(sum(follow)), "followup_n": len(follow)})
    packed = [m for m in methods if m.startswith(PK)]
    comparisons = [(FLOOR, U), (RAW, U), (RAW, FLOOR)]
    if "recompute" in methods:
        comparisons += [(m, "recompute") for m in packed]
    pairs = []
    for a, b in comparisons:
        for scope, flt in (("followup", FOLLOW), ("all", ALL), ("T1", lambda k, v: k[1] == 1)):
            d = paired(by[a], by[b], images, img_of, flt)
            c, agree = cells(by[a], by[b], flt)
            pairs.append({"a": a, "b": b, "scope": scope, "delta_pp": 100 * d["mean"], "ci95_lo_pp": 100 * d["ci_low"], "ci95_hi_pp": 100 * d["ci_high"],
                          "prediction_agreement": agree, **c, "resamples": d["resamples"], "seed": d["seed"]})
        for metric in ("ttft_ms", "request_e2e_ms", "ssd_read_bytes"):
            d = paired(by[a], by[b], images, img_of, FOLLOW, key=metric)
            pairs.append({"a": a, "b": b, "scope": "followup", "metric": metric, "delta": d["mean"], "ci95_lo": d["ci_low"], "ci95_hi": d["ci_high"]})
    latency = []
    for m in methods:
        f = [v for k, v in by[m].items() if k[1] > 1]
        t = stats([v["ttft_ms"] for v in f]); e = stats([v["request_e2e_ms"] for v in f])
        entry = {"method": m, "name": name(m), "ttft_mean_ms": t[0], "ttft_p50_ms": t[1], "ttft_p95_ms": t[2],
                 "e2e_mean_ms": e[0], "e2e_p50_ms": e[1], "e2e_p95_ms": e[2], "ssd_mb_per_followup": float(np.mean([v["ssd_read_bytes"] for v in f])) / 1e6,
                 "read_calls_mean": float(np.mean([v["ssd_preads"] for v in f])), "pread_calls_mean": float(np.mean([v["pread_calls"] for v in f])),
                 "preadv_calls_mean": float(np.mean([v["preadv_calls"] for v in f])), "followups": len(f)}
        for key in ("chunk_io_ms", "scatter_ms", "packed_read_ms", "prefill_ms"):
            vals = [v[key] for v in f if v.get(key) is not None]
            entry[f"{key}_mean"] = float(np.mean(vals)) if vals else None
        t1 = stats([v["ttft_ms"] for k, v in by[m].items() if k[1] == 1])
        entry["t1_ttft_mean_ms"] = t1[0]
        latency.append(entry)
    storage = []; setup = {}
    for m in H.stored_methods(methods):
        kind = H.kind_of(m); rs = []
        for img in images:
            folder = attempts[img]
            per = json.loads((folder / f"{kind}_persistence.json").read_text()); act = json.loads((folder / f"{m}_activation.json").read_text())
            meta = json.loads((folder / f"{kind}_meta.json").read_text())
            n = meta["v_token_num"] - len(meta["newline_idx"])
            rs.append({"bytes": per["file_logical_bytes"], "allocated": per["allocated_bytes"], "os_write": per["os_write_bytes"], "persist_ms": per["service_persistence_ms"],
                       "writer_ms": per["native_writer_wall_ms"], "act_ms": act["outer_wall_ms"], "N": n, "content_rows_total": meta.get("stored_content_count_total", n * 32),
                       "packed_bytes": meta.get("packed_total_bytes", 0), "sha_ms": per.get("timing_ms", {}).get("packed_sha256_ms", 0.0)})
            setup[(m, img)] = per["service_persistence_ms"] + act["outer_wall_ms"]
        b = stats([x["bytes"] for x in rs]); pm = stats([x["persist_ms"] for x in rs]); am = stats([x["act_ms"] for x in rs])
        storage.append({"method": m, "name": name(m), "images": len(rs), "stored_content_pct": 100 * np.mean([x["content_rows_total"] / (32 * x["N"]) for x in rs]),
                        "store_mb_mean": b[0] / 1e6, "store_mb_p50": b[1] / 1e6, "store_mb_p95": b[2] / 1e6, "packed_file_mb_mean": np.mean([x["packed_bytes"] for x in rs]) / 1e6,
                        "allocated_mb_mean": np.mean([x["allocated"] for x in rs]) / 1e6, "os_write_mb_mean": np.mean([x["os_write"] for x in rs]) / 1e6,
                        "persistence_ms_mean": pm[0], "persistence_ms_p50": pm[1], "persistence_ms_p95": pm[2], "native_writer_ms_mean": np.mean([x["writer_ms"] for x in rs]),
                        "packed_sha256_ms_mean": np.mean([x["sha_ms"] for x in rs]), "activation_ms_mean": am[0], "activation_ms_p50": am[1], "activation_ms_p95": am[2], "provenance": "MEASURED"})
    dialogs = sorted({r["dialog_id"] for r in rows})
    first_dialog = {}
    for r in sorted(rows, key=lambda x: x["global_dialog_ordinal"]):
        first_dialog.setdefault(r["image_id"], r["dialog_id"])
    sessions = []; sess_vals = {}
    for m in methods:
        shared, standalone = [], []
        for d in dialogs:
            e2e = sum(by[m][(d, t)]["request_e2e_ms"] for t in (1, 2, 3))
            su = setup.get((m, img_of[d]), 0.0)
            shared.append(e2e + (su if first_dialog.get(img_of[d]) == d else 0.0)); standalone.append(e2e + su)
        sess_vals[m] = (shared, standalone)
        s1, s2 = stats(shared), stats(standalone)
        sessions.append({"method": m, "name": name(m), "shared_mean_ms": s1[0], "shared_p50_ms": s1[1], "shared_p95_ms": s1[2],
                         "standalone_mean_ms": s2[0], "standalone_p50_ms": s2[1], "standalone_p95_ms": s2[2],
                         "provenance": "MEASURED_NO_KV_SETUP" if m == "recompute" else "MEASURED (shared) / DERIVED (standalone)"})
    for a, b in [(FLOOR, U), (RAW, U), (RAW, FLOOR)]:
        for kind_i, kind in ((0, "shared"), (1, "standalone")):
            per = []
            for img in images:
                idx = [i for i, d in enumerate(dialogs) if img_of[d] == img]
                per.append((sum(sess_vals[a][kind_i][i] for i in idx), sum(sess_vals[b][kind_i][i] for i in idx), len(idx)))
            d = paired_bootstrap(per, 10000, SEED)
            pairs.append({"a": a, "b": b, "scope": f"session_{kind}", "metric": "session_ms", "delta": d["mean"], "ci95_lo": d["ci_low"], "ci95_hi": d["ci_high"]})
    # question types (followup)
    qt_rows = []
    types = ["all"] + [t for t, _ in QTYPE] + ["other"]
    for t in types:
        flt = (lambda k, v, t=t: k[1] > 1 and (t == "all" or qtype(v["question"]) == t))
        n = sum(1 for k, v in by[U].items() if flt(k, v))
        if n == 0:
            continue
        entry = {"question_type": t, "n_followup": n}
        for m in packed:
            entry[f"acc_{name(m)}_pct"] = 100 * float(np.mean([v["score"] for k, v in by[m].items() if flt(k, v)]))
        for a, b, label in ((FLOOR, U, "PYR-beta_minus_U"), (RAW, U, "PYR-beta-raw_minus_U"), (RAW, FLOOR, "raw_minus_floor")):
            d = paired(by[a], by[b], images, img_of, flt)
            entry[f"{label}_pp"] = 100 * d["mean"]; entry[f"{label}_ci_lo"] = 100 * d["ci_low"]; entry[f"{label}_ci_hi"] = 100 * d["ci_high"]
        qt_rows.append(entry)
    # per-layer k_l and bytes actually served (mean over followups)
    layer_rows = []
    row_bytes = 32 * 128 * 2
    kmean = {m: np.mean(np.asarray([v["k_per_layer"] for k, v in by[m].items() if k[1] > 1], dtype=float), axis=0) for m in packed}
    nmean = float(np.mean([v["N_content"] for k, v in by[U].items() if k[1] > 1]))
    for li in range(32):
        e = {"layer": li}
        for m in packed:
            e[f"r_{name(m)[:-7]}"] = config["layer_budgets"][H.vector_of(m)]["layer_keep_ratios"][li]
        for m in packed:
            e[f"k_mean_{name(m)[:-7]}"] = float(kmean[m][li])
        for m in packed:
            e[f"mb_mean_{name(m)[:-7]}"] = float(2 * kmean[m][li] * row_bytes / 1e6)
        layer_rows.append(e)
    for m in packed:
        assert abs(float(kmean[m].sum()) - float(kmean[U].sum())) < 1e-6
    lat = {l["method"]: l for l in latency}
    csv_write(res / "hold" / "quality.csv", quality); csv_write(res / "hold" / "latency_io.csv", latency)
    csv_write(res / "hold" / "storage.csv", storage); csv_write(res / "hold" / "paired_analysis.csv", pairs); csv_write(res / "hold" / "sessions.csv", sessions)
    csv_write(res / "hold" / "question_types.csv", qt_rows); csv_write(res / "hold" / "layer_table.csv", layer_rows)
    write_json(res / "hold" / "paired_analysis.json", pairs)
    return {"quality": quality, "latency": latency, "storage": storage, "pairs": pairs, "sessions": sessions, "images": len(images), "dialogs": len(dialogs),
            "qtypes": qt_rows, "layers": layer_rows, "mean_N_followup": nmean, "by": by, "img_of": img_of, "image_list": images,
            "ttft_ratio_vs_U": {name(m): lat[m]["ttft_mean_ms"] / lat[U]["ttft_mean_ms"] for m in packed}}


def pair_of(h, a, b, scope="followup"):
    return next(p for p in h["pairs"] if p["a"] == a and p["b"] == b and p["scope"] == scope and "metric" not in p)


def decide_vector(h, sel):
    lat = {x["method"]: x for x in h["latency"]}
    d = pair_of(h, sel, U)
    delta, lo, hi = float(d["delta_pp"]), float(d["ci95_lo_pp"]), float(d["ci95_hi_pp"])
    ci_excludes_zero = bool(lo > 0 or hi < 0)
    mb_ok = bool(lat[sel]["ssd_mb_per_followup"] <= lat[U]["ssd_mb_per_followup"] * 1.02)
    ttft_ok = bool(lat[sel]["ttft_mean_ms"] <= lat[U]["ttft_mean_ms"] * 1.05)
    if delta >= 1.0 and ci_excludes_zero and mb_ok and ttft_ok:
        verdict = "GO"
    elif (delta >= 0.5 and not ci_excludes_zero) or (delta >= 1.0 and ci_excludes_zero and (mb_ok != ttft_ok)):
        verdict = "PARTIAL"
    else:
        verdict = "NO-GO"
    note = None
    if verdict == "NO-GO" and ci_excludes_zero and delta > 0:
        note = f"규칙상 NO-GO지만 CI는 0 제외: Δ = {delta:+.3f} pp [{lo:+.3f}, {hi:+.3f}]"
    return {"verdict": verdict, "vector": name(sel), "control": name(U), "delta_followup_pp": delta, "ci95_pp": [lo, hi], "ci_excludes_zero": ci_excludes_zero,
            "ssd_mb_vector": lat[sel]["ssd_mb_per_followup"], "ssd_mb_uniform": lat[U]["ssd_mb_per_followup"], "ssd_ok_1.02": mb_ok,
            "ttft_mean_vector": lat[sel]["ttft_mean_ms"], "ttft_mean_uniform": lat[U]["ttft_mean_ms"], "ttft_ratio": lat[sel]["ttft_mean_ms"] / lat[U]["ttft_mean_ms"],
            "ttft_ok_1.05": ttft_ok, "note": note}


def floor_effect(h):
    d = pair_of(h, RAW, FLOOR)
    delta, lo, hi = float(d["delta_pp"]), float(d["ci95_lo_pp"]), float(d["ci95_hi_pp"])
    if hi < 0:
        cls, text, default = "FLOOR_NEEDED", "하한 필요: PYR-beta-raw가 PYR-beta보다 유의하게 낮다. PYR-beta(하한 0.10)를 쓰고 이 결과를 하한의 근거로 보고한다.", "PYR-beta"
    elif lo > 0:
        cls, text, default = "FLOOR_HARMFUL", ("하한이 손해: PYR-beta-raw가 PYR-beta보다 유의하게 높다. PYR-beta-raw를 쓰고, 하한 0.10에 걸린 기존 벡터(A1-2, VLC_offline 등)도 "
                                               "영향을 받았을 수 있음을 명시한다."), "PYR-beta-raw"
    else:
        cls, text, default = "FLOOR_UNNECESSARY", "하한 불필요: D의 95% CI가 0을 포함한다. 논문 기본안은 PYR-beta-raw(PyramidKV 원식 그대로)이다.", "PYR-beta-raw"
    return {"definition": "D = PYR-beta-raw − PYR-beta, HOLD followup accuracy, image-cluster paired bootstrap 95% CI (10,000 resamples, seed 1234), same run",
            "delta_pp": delta, "ci95_pp": [lo, hi], "classification": cls, "paper_default": default, "sentence": f"{text} (D = {delta:+.3f} pp [{lo:+.3f}, {hi:+.3f}])",
            "a_only_b_only_both_neither": [d["a_only"], d["b_only"], d["both"], d["neither"]], "prediction_agreement": d["prediction_agreement"]}


def cross_run(h, gdet):
    out = {"status": "NOT_COMPARABLE", "reason": "G-det main is not 100%", "label": None, "comparisons": []}
    if gdet.get("cross_run_accuracy_comparison") != "ALLOWED":
        return out, None
    old = index(v2_rows(set(V2_ARMS)))
    by, img_of, images = h["by"], h["img_of"], h["image_list"]
    assert all(set(old[m]) == set(by[U]) for m in V2_ARMS)
    same_u = all(old[U][k]["tokens"] == by[U][k]["tokens"] and old[U][k]["score"] == by[U][k]["score"] for k in by[U])
    assert same_u, "G-det said 100% but the U rows differ"
    comps = []
    for a in (FLOOR, RAW):
        for b in V2_ARMS:
            for scope, flt in (("followup", FOLLOW), ("all", ALL)):
                d = paired(by[a], old[b], images, img_of, flt)
                c, agree = cells(by[a], old[b], flt)
                comps.append({"a": name(a) + " (this run)", "b": name(b) + " (v2 run)", "scope": scope, "delta_pp": 100 * d["mean"], "ci95_lo_pp": 100 * d["ci_low"],
                              "ci95_hi_pp": 100 * d["ci_high"], "prediction_agreement": agree, **c, "label": "cross-run, 결정성 확인됨"})
    v2acc = {name(m): 100 * float(np.mean([v["score"] for k, v in old[m].items() if k[1] > 1])) for m in V2_ARMS}
    v2_u_ttft = float(np.mean([v["ttft_ms"] for k, v in old[U].items() if k[1] > 1]))
    out = {"status": "COMPARED", "label": "cross-run, 결정성 확인됨 (U_packed 12,183 요청 토큰 단위 100% 일치)", "v2_run": str(V2_RUN.relative_to(ROOT)),
           "v2_followup_pct": v2acc, "u_rows_identical_tokens_and_scores": True, "comparisons": comps,
           "ttft_note": "TTFT is never compared across runs for the decision; the U_packed difference below is run-to-run variation only"}
    return out, v2_u_ttft


def figure(res, docs, pyr_linear):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#1f1f1e", "#6b6a63", "#e4e3dc"
    series = [("PYR (linear, v2 HOLD arm)", pyr_linear, "#2a78d6", "-", "o"),
              ("PYR-beta (PyramidKV β=20 + floor 0.10)", docs["PYR-beta"]["layer_keep_ratios"], "#eb6834", "-", "s"),
              ("PYR-beta-raw (PyramidKV β=20, as published)", docs["PYR-beta-raw"]["layer_keep_ratios"], "#1baf7a", "--", "^")]
    fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    x = list(range(32))
    ax.plot(x, docs["U"]["layer_keep_ratios"], color=muted, lw=1.5, ls=":", label="U (uniform 0.25)")
    for label, y, c, ls, mk in series:
        ax.plot(x, y, color=c, lw=2, ls=ls, marker=mk, ms=4, markevery=2, label=label)
    ax.axhline(0.10, color=grid, lw=1)
    ax.text(0.2, 0.105, "floor 0.10 (contract safeguard)", color=muted, fontsize=8, va="bottom")
    ax.annotate("0.0125", (31, docs["PYR-beta-raw"]["layer_keep_ratios"][-1]), textcoords="offset points", xytext=(-4, -12), ha="right", fontsize=8, color=ink)
    ax.annotate("0.4875", (0, docs["PYR-beta-raw"]["layer_keep_ratios"][0]), textcoords="offset points", xytext=(6, 2), fontsize=8, color=ink)
    ax.set_xlabel("decoder layer", color=ink); ax.set_ylabel("keep ratio r_l  (Σ r_l = 8.0)", color=ink)
    ax.set_title("Layer-wise keep ratios: PyramidKV schedules vs uniform", color=ink, fontsize=11, loc="left")
    ax.set_xlim(-0.5, 31.5); ax.set_ylim(0, 0.52)
    ax.grid(color=grid, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8)
    leg = ax.legend(frameon=False, fontsize=8, loc="upper right")
    for t in leg.get_texts():
        t.set_color(ink)
    fig.tight_layout(); fig.savefig(res / "fig_vectors_pyr.png", dpi=150, facecolor=fig.get_facecolor()); plt.close(fig)


def protection_after(run, res):
    before = json.loads((run / "protection_before.json").read_text())
    excl = before["excluded_new_paths"]
    shas = {k: v for k, v in sha_manifest([ROOT / x for x in ["results", "docs", "mmimpress", "scripts", "tests", "README.md"]], ROOT, exclude=excl).items() if "__pycache__" not in k}
    kv = stat_inventory(sorted(ROOT.glob("kvstore*")), ROOT)
    runs = stat_inventory(sorted(p for p in (ROOT / "runs").iterdir() if p.name != run.name), ROOT)
    def inv_diff(b, a):
        return {"changed": sorted(k for k in b if k in a and a[k] != b[k]), "missing": sorted(set(b) - set(a)), "added": sorted(set(a) - set(b))}
    diff = {"sha_changed": sorted(k for k in before["sha256"] if k in shas and shas[k] != before["sha256"][k]),
            "sha_missing": sorted(set(before["sha256"]) - set(shas)), "sha_added": sorted(set(shas) - set(before["sha256"]))}
    kvd = inv_diff(before["kvstore_inventory"], kv); rund = inv_diff(before["runs_inventory"], runs)
    amendments = {}
    for p in sorted(run.glob("freeze_amendment_*.json")):
        a = json.loads(p.read_text())
        if a.get("sha256_now"):
            amendments[a["file"]] = a["sha256_now"]
    unexplained = [k for k in diff["sha_changed"] if shas[k]["sha256"] != amendments.get(k)]
    # other sessions may create new run roots while this run is in progress; additions are reported, changes/missing are failures
    hard = [unexplained, diff["sha_missing"], diff["sha_added"], kvd["changed"], kvd["missing"], kvd["added"], rund["changed"], rund["missing"]]
    protection = {"status": "PASS" if not any(hard) else "FAIL", "files_before": len(before["sha256"]), "files_after": len(shas),
                  "kvstore_files_before": len(before["kvstore_inventory"]), "kvstore_files_after": len(kv), "runs_files_before": len(before["runs_inventory"]), "runs_files_after": len(runs),
                  "differences": {"sha": diff, "kvstore": kvd, "runs": rund}, "sha_changes_unexplained": unexplained,
                  "changes_explained_by_recorded_amendments": [k for k in diff["sha_changed"] if k not in unexplained],
                  "amendment_files": [p.name for p in sorted(run.glob("freeze_amendment_*.json"))], "runs_added_reported_only": rund["added"][:50],
                  "source_modifications": json.loads((run / "source_modifications.json").read_text()),
                  "scope": "full SHA256 of results/docs/mmimpress/scripts/tests/README.md (new paths of this run excluded); (size, mtime_ns, inode) of kvstore* and of every other runs/* root"}
    write_json(res / "protection_result.json", protection)
    write_json(run / "protection_after.json", {"stage": "after", "utc": time.time(), "sha256": shas, "kvstore_inventory": kv, "runs_inventory": runs})
    # results/ gets the full SHA manifest and kvstore inventory; the 1.8M-entry runs inventory stays in the run root and is summarised by a digest
    def digest(inv):
        import hashlib
        return hashlib.sha256(json.dumps(inv, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for stage, sh, kvi, ri, src in (("before", before["sha256"], before["kvstore_inventory"], before["runs_inventory"], "protection_before.json"), ("after", shas, kv, runs, "protection_after.json")):
        write_json(res / f"protection_{stage}.json", {"stage": stage, "full_file": str((run / src).relative_to(ROOT)), "full_file_sha256": sha(run / src), "sha256": sh, "kvstore_inventory": kvi,
                                                      "runs_inventory_files": len(ri), "runs_inventory_digest_sha256": digest(ri)})
    return protection


def deletion_log(run, res, attempts):
    hold = run / "hold"; images = []; files = 0
    for img, folder in attempts.items():
        rc = json.loads((folder / "cleanup_receipt.json").read_text()); allow = json.loads((folder / "cleanup_allowlist.json").read_text())
        assert rc["status"] == "PASS" and rc["removed_new_files"] == len(allow["file_sha256"])
        images.append({"image_id": img, "scratch": allow["scratch"], "removed_files": rc["removed_new_files"], "allowlist_sha256": sha(folder / "cleanup_allowlist.json")}); files += rc["removed_new_files"]
    smoke = []
    for cp in sorted((hold / "smoke" / "images").glob("*/COMMITTED.json")):
        folder = cp.parent / json.loads(cp.read_text())["attempt_id"]; rc = json.loads((folder / "cleanup_receipt.json").read_text())
        smoke.append({"image_id": cp.parent.name, "removed_files": rc["removed_new_files"]})
    leftovers = sorted(str(p.relative_to(run)) for p in (hold / "scratch").rglob("*") if p.is_file()) if (hold / "scratch").exists() else []
    gate = json.loads((run / "gates" / "gpu_gates.json").read_text()); grmin = json.loads((run / "gates" / "g_rmin_floor_propagation.json").read_text())
    gate_left = sorted(str(p.relative_to(run)) for p in (run / "llava" / "gpu_gates").rglob("*") if p.is_file()) if (run / "llava" / "gpu_gates").exists() else []
    log = {"scope": "run-local only: hold/scratch (exact SHA allowlist after audit + commit), llava/gpu_gates (per case, guarded_rmtree), gates/grmin_fixture (CPU fixture)",
           "existing_runs_results_stores_deleted": 0, "hold_main": {"images": len(images), "removed_files": files, "per_image": images}, "hold_smoke": smoke,
           "retained_scratch_files": leftovers, "gpu_gate_cases_deleted": sum(1 for s in gate["samples"] if s.get("stores_deleted_after_record")), "gpu_gate_retained_files": gate_left,
           "grmin_fixture_deleted": grmin.get("fixture_deleted"), "status": "PASS" if not leftovers and not gate_left and len(images) == 398 else "CHECK"}
    write_json(res / "deletion_log.json", log)
    return log


# ------------------------------------------------------------------- main
def fmt(d, lo, hi):
    return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run).resolve(); res = ROOT / "results" / run.name
    res.mkdir(parents=True, exist_ok=True)
    assert run.parent == ROOT / "runs" and run.name.startswith("layer_budget_kv25_pyrbeta_")
    config = json.loads((run / "config.json").read_text())
    hold = run / "hold"
    H = load("pyrbeta_hold", "164_eval_layer_budget_pyrbeta_hold.py")
    Gt = load("pyrbeta_gates", "166_validate_layer_budget_pyrbeta.py")
    hconfig = json.loads((hold / "config.json").read_text()); hmanifest = json.loads((hold / "manifest.json").read_text())
    hconfig.update(_file_sha256=sha(hold / "config.json"), _manifest_sha256=sha(hold / "manifest.json"))
    complete = json.loads((hold / "main_execution_complete.json").read_text())
    rows, attempts, audit = load_and_audit(hold, hconfig, hmanifest, H)
    write_json(res / "independent_audit.json", audit)
    h = hold_analysis(hold, rows, hconfig, attempts, res, H)
    # G-det on every U_packed request
    Gt.stage_gdet(run, "main")
    gdet = json.loads((run / "gates" / "g_det_main.json").read_text()); gsmoke = json.loads((run / "gates" / "g_det_smoke.json").read_text())
    write_json(res / "g_det_equivalence.json", {"smoke": gsmoke, "main": gdet})
    xr, v2_u_ttft_raw = cross_run(h, gdet)
    v2_lat = {r["method"]: r for r in csv.DictReader((V2_RES / "hold" / "latency_io.csv").open())}
    lat = {l["method"]: l for l in h["latency"]}; q = {x["method"]: x for x in h["quality"]}
    v2_u_ttft = float(v2_lat[U]["ttft_mean_ms"])
    xr["U_packed_ttft_cross_run"] = {"this_run_mean_ms": lat[U]["ttft_mean_ms"], "v2_run_mean_ms": v2_u_ttft, "difference_ms": lat[U]["ttft_mean_ms"] - v2_u_ttft,
                                     "ratio": lat[U]["ttft_mean_ms"] / v2_u_ttft, "v2_mean_recomputed_from_raw_ms": v2_u_ttft_raw, "label": "cross-run, 실행 간 변동 참고치 (판정에 쓰지 않음)"}
    write_json(res / "cross_run_v2_comparison.json", xr)
    decisions = {name(m)[:-7]: decide_vector(h, m) for m in (FLOOR, RAW)}
    fe = floor_effect(h)
    main_starts = sorted(hold.glob("main_start_*.json")); interruptions = sorted(p.name for p in hold.glob("interruption_*.json"))
    first_start = json.loads(main_starts[0].read_text())["started_unix"]
    elapsed_h = (complete["finished_unix"] - first_start) / 3600
    failures_file = [str(p.relative_to(run)) for p in hold.rglob("failure.json")]
    decision = {"primary": decisions, "floor_effect": fe, "audit_status": audit["status"], "final": audit["status"] == "PASS" and complete["status"] == "PASS",
                "requests": audit["requests"], "expected_requests": audit["expected_requests"], "images": audit["images"], "failed_or_skipped_requests": 0 if audit["status"] == "PASS" else None,
                "failed_attempt_records": failures_file, "interruptions": interruptions, "main_wall_hours_first_start_to_finish": elapsed_h,
                "followup_pct": {name(m): q[m]["followup_pct"] for m in hconfig["methods"]},
                "g_det": {"smoke": gsmoke["status"], "main": gdet["status"], "token_exact_rate_main": gdet["token_exact_rate"], "cross_run_accuracy_comparison": gdet["cross_run_accuracy_comparison"]},
                "rule": config["decision_rule"], "arms": hconfig["methods"], "optional_recompute": "included" if "recompute" in hconfig["methods"] else "excluded"}
    write_json(res / "decision.json", decision)
    # copies
    shutil.copy2(run / "CONTRACT.md", res / "CONTRACT.md")
    for p in sorted((run / "vectors").glob("*")):
        shutil.copy2(p, res / p.name)
    docs = {a: json.loads((run / "vectors" / f"layer_budget_llava_{a}.json").read_text()) for a in ("U", "PYR-beta", "PYR-beta-raw")}
    write_json(res / "vectors_sha256.json", {a: {"layer_budget_sha256": d["layer_budget_sha256"], "file_sha256": sha(run / "vectors" / f"layer_budget_llava_{a}.json"),
                                                 "r_min": d["r_min"], "r_max": d["r_max"]} for a, d in docs.items()})
    for p in sorted((run / "gates").glob("*.json")):
        shutil.copy2(p, res / ("gate_" + p.name))
    for f in ("source_modifications.json", "hold_arms.json", "config.json"):
        shutil.copy2(run / f, res / ("run_" + f if f == "config.json" else f))
    shutil.copy2(run / "phase0" / "storage_plan.json", res / "phase0_storage_plan.json")
    for p in sorted(run.glob("freeze_amendment_*.json")):
        shutil.copy2(p, res / p.name)
    gates = {p.stem: json.loads(p.read_text())["status"] for p in sorted((run / "gates").glob("*.json"))}
    pyr_linear = json.loads((V2_RUN / "phase2" / "layer_budget_llava_PYR.json").read_text())["layer_keep_ratios"]
    try:
        figure(res, docs, pyr_linear)
    except Exception as exc:
        write_json(res / "fig_vectors_pyr_error.json", {"error": repr(exc)})
    dele = deletion_log(run, res, attempts)
    protection = protection_after(run, res)
    ktab = json.loads((run / "vectors" / "k_tables.json").read_text())
    # ---------------------------------------------------------------- REPORT
    L = []
    L.append("# PyramidKV(β=20) layer 배분 — MT-GQA HOLD 추가 실행 결과 (PYR-beta, PYR-beta-raw vs U_packed)\n")
    L.append(f"Run `{run.relative_to(ROOT)}`. 계약 [CONTRACT.md](CONTRACT.md) (SHA `{config['contract_sha256'][:16]}…`, 측정 전 freeze). 부모 run: v2 `{V2_RES.relative_to(ROOT)}` (판정 GO). "
             "모델 llava-v1.6-vicuna-7b NF4/BF16/eager, FP16 SSD KV, greedy max 16, seed 1234. 데이터로 벡터를 만들거나 고르는 과정은 없다.\n")
    L.append("## 반드시 알아야 할 정의\n")
    L.append("- **PYR-beta-raw** = PyramidKV 공개 코드 식(β=20, window 0)의 layer 비율 그대로(하한 없음, 문서 r_min = 0.0). **PYR-beta** = 같은 식에 우리 계약 하한 0.10을 적용한 것(v2에서 freeze한 벡터). "
             "하한 0.10은 v1 설계에서 정한 안전장치이며 이번 실험 전까지 실험·선행 연구 근거가 없었다.\n"
             "- 원 PyramidKV는 attention 기반 token 선택을 함께 쓰지만, 여기서는 image-only VisionZip 순서를 그대로 두고 layer별 자르는 지점만 PyramidKV를 따랐다.\n"
             "- v2의 `PYR`은 선형 schedule(v1 F-front, 0.40 → 0.10)이며 두 PYR-beta와 다른 벡터다.\n"
             "- 모든 arm은 같은 바이트·같은 packed I/O 경로(hit당 preadv 1회)·같은 token 순서를 쓰고 layer별 자르는 지점만 다르다.\n")
    L.append("## 판정\n")
    rows_d = []
    for key in ("PYR-beta", "PYR-beta-raw"):
        d = decisions[key]
        rows_d.append([key, f"**{d['verdict']}**", fmt(d["delta_followup_pp"], *d["ci95_pp"]), "제외" if d["ci_excludes_zero"] else "포함",
                       f"{d['ssd_mb_vector']:.3f} / {d['ssd_mb_uniform']:.3f} ({'OK' if d['ssd_ok_1.02'] else '위반'})",
                       f"{d['ttft_mean_vector']:.2f} / {d['ttft_mean_uniform']:.2f} = {d['ttft_ratio']:.3f}× ({'OK' if d['ttft_ok_1.05'] else '위반'})"])
    L.append("주 판정(사전 고정, 같은 실행 안, 벡터 − U_packed, HOLD followup, 이미지 단위 paired bootstrap 10,000회, seed 1234):\n")
    L.append(md_table(["벡터", "판정", "Δ pp [95% CI]", "CI의 0", "SSD MB/followup (벡터 / U, ≤1.02×)", "TTFT 평균 ms (벡터 / U, ≤1.05×)"], rows_d))
    for key in ("PYR-beta", "PYR-beta-raw"):
        if decisions[key]["note"]:
            L.append(f"- {key}: {decisions[key]['note']}\n")
    L.append(f"하한 효과 판정(사전 고정): **{fe['classification']}** — {fe['sentence']} 논문 기본안: **{fe['paper_default']}**. "
             f"(raw만 정답 {fe['a_only_b_only_both_neither'][0]} / floor만 정답 {fe['a_only_b_only_both_neither'][1]} / 둘 다 {fe['a_only_b_only_both_neither'][2]} / 둘 다 오답 {fe['a_only_b_only_both_neither'][3]}, "
             f"예측 일치율 {100 * fe['prediction_agreement']:.3f}%)\n")
    L.append(f"독립 감사: {audit['status']} — {audit['requests']:,} / {audit['expected_requests']:,} 요청, {audit['images']} / 398 이미지, 감사 검사 {audit['checks']:,}건, 실패 0. final = {decision['final']}.\n")
    L.append("## Gate\n")
    L.append(md_table(["Gate", "결과"], [[k, v] for k, v in gates.items()] + [["Artifact protection before/after", protection["status"]], ["Deletion log (run-local scratch only)", dele["status"]]]))
    L.append(f"- G-det: smoke {gsmoke['token_exact']}/{gsmoke['requests']} 토큰 일치, main {gdet['token_exact']:,}/{gdet['requests']:,} ({100 * gdet['token_exact_rate']:.4f}%) → cross-run 정확도 비교 {gdet['cross_run_accuracy_comparison']}.\n"
             "- G-rmin: raw 벡터 문서의 r_min = 0.0이 store meta·packed/unpacked writer·모든 validator·두 selector·164 감사 경로까지 전달됨을 확인했고, 같은 store의 meta에서 r_min만 0.10으로 바꾸면 "
             "모두 fail closed된다. store/serving 경로에서 전역 기본값 0.10에 기대는 호출은 발견되지 않아 소스 수정(freeze_amendment)은 없었다 "
             "(156/162의 감사 호출은 기본값을 쓰므로 raw 벡터에 쓸 수 없고, 164는 문서의 r_min을 넘긴다).\n")
    L.append("## HOLD 정확도 (MT-GQA-reconstructed, 398 images, 4,061 dialogues × 3 turns, generated history)\n")
    L.append(md_table(["Method", "T1 %", "T2 %", "T3 %", "All % (correct/N)", "Followup % (correct/N)"],
                      [[x["name"], f"{x['T1_pct']:.3f}", f"{x['T2_pct']:.3f}", f"{x['T3_pct']:.3f}", f"{x['all_pct']:.3f} ({x['all_correct']}/{x['all_n']})",
                        f"{x['followup_pct']:.3f} ({x['followup_correct']}/{x['followup_n']})"] for x in h["quality"]]))
    L.append(md_table(["a − b", "Scope", "Δ pp [95% CI]", "a-only / b-only / both / neither", "예측 일치 %"],
                      [[f"{name(p['a'])} − {name(p['b'])}", p["scope"], fmt(p["delta_pp"], p["ci95_lo_pp"], p["ci95_hi_pp"]),
                        f"{p['a_only']} / {p['b_only']} / {p['both']} / {p['neither']}", f"{100 * p['prediction_agreement']:.3f}"] for p in h["pairs"] if "metric" not in p]))
    L.append("## 지연·I/O (같은 실행 안, followup)\n")
    L.append(md_table(["Method", "TTFT mean / p50 / p95 ms (× U)", "E2E mean / p50 / p95 ms", "read ms", "scatter ms", "prefill ms", "SSD MB/followup", "pread / preadv"],
                      [[l["name"], f"{l['ttft_mean_ms']:.2f} / {l['ttft_p50_ms']:.2f} / {l['ttft_p95_ms']:.2f} ({l['ttft_mean_ms'] / lat[U]['ttft_mean_ms']:.3f}×)",
                        f"{l['e2e_mean_ms']:.2f} / {l['e2e_p50_ms']:.2f} / {l['e2e_p95_ms']:.2f}",
                        f"{l['chunk_io_ms_mean']:.1f}" if l.get("chunk_io_ms_mean") is not None else "—", f"{l['scatter_ms_mean']:.1f}" if l.get("scatter_ms_mean") is not None else "—",
                        f"{l['prefill_ms_mean']:.1f}" if l.get("prefill_ms_mean") is not None else "—",
                        f"{l['ssd_mb_per_followup']:.2f}", f"{l['pread_calls_mean']:.0f} / {l['preadv_calls_mean']:.0f}"] for l in h["latency"]]))
    L.append(md_table(["a − b", "Scope / metric", "Δ [95% CI]"],
                      [[f"{name(p['a'])} − {name(p['b'])}", f"{p['scope']} / {p['metric']}", fmt(p["delta"], p["ci95_lo"], p["ci95_hi"])] for p in h["pairs"] if "metric" in p]))
    xu = xr["U_packed_ttft_cross_run"]
    L.append(f"U_packed TTFT 평균: 이번 실행 {xu['this_run_mean_ms']:.2f} ms vs v2 실행 {xu['v2_run_mean_ms']:.2f} ms (차이 {xu['difference_ms']:+.2f} ms, {xu['ratio']:.3f}×) — cross-run, 실행 간 변동 참고치이며 판정에 쓰지 않는다 "
             "(v2는 7 arm, 이번은 3 arm 회전).\n")
    L.append("## 부 판정 — v2 HOLD arm과의 cross-run 정확도 비교\n")
    if xr["status"] == "COMPARED":
        L.append(f"{xr['label']}. 요청 단위(dialog, turn)로 짝지은 이미지 단위 paired bootstrap. v2 followup 정확도: " + ", ".join(f"{k} {v:.3f}%" for k, v in xr["v2_followup_pct"].items()) + ".\n")
        L.append(md_table(["a (이번 실행) − b (v2 실행)", "Scope", "Δ pp [95% CI]", "a-only / b-only / both / neither", "예측 일치 %"],
                          [[f"{c['a']} − {c['b']}", c["scope"], fmt(c["delta_pp"], c["ci95_lo_pp"], c["ci95_hi_pp"]), f"{c['a_only']} / {c['b_only']} / {c['both']} / {c['neither']}",
                            f"{100 * c['prediction_agreement']:.3f}"] for c in xr["comparisons"]]))
        L.append("TTFT는 cross-run으로 비교하지 않는다(ReComp를 이번에 돌리지 않았으므로 ReComp 대비 TTFT 비교는 NOT_MEASURED).\n")
    else:
        L.append(f"**NOT_COMPARABLE** — {xr['reason']}. 주 판정은 같은 실행 안 비교이므로 영향 없음.\n")
    L.append("## 질문 유형별 Δ (followup, v2 REPORT와 같은 regex 분류; spatial 강조)\n")
    packed = [m for m in hconfig["methods"] if m.startswith(PK)]
    L.append(md_table(["type", "n"] + [f"{name(m)} %" for m in packed] + ["PYR-beta − U pp [CI]", "PYR-beta-raw − U pp [CI]", "raw − floor pp [CI]"],
                      [[("**spatial**" if e["question_type"] == "spatial" else e["question_type"]), e["n_followup"]] + [f"{e[f'acc_{name(m)}_pct']:.2f}" for m in packed] +
                       [fmt(e["PYR-beta_minus_U_pp"], e["PYR-beta_minus_U_ci_lo"], e["PYR-beta_minus_U_ci_hi"]), fmt(e["PYR-beta-raw_minus_U_pp"], e["PYR-beta-raw_minus_U_ci_lo"], e["PYR-beta-raw_minus_U_ci_hi"]),
                        fmt(e["raw_minus_floor_pp"], e["raw_minus_floor_ci_lo"], e["raw_minus_floor_ci_hi"])] for e in h["qtypes"]]))
    L.append("유형별 CI는 다중 비교 보정을 하지 않은 탐색적 수치다.\n")
    L.append(f"## Layer별 비율·k_l·바이트 (두 벡터 나란히; k_l·MB는 HOLD followup 평균, 평균 N = {h['mean_N_followup']:.1f})\n")
    L.append(md_table(["layer", "r 원식(raw)", "r 하한 0.10(PYR-beta)", "차이(하한−원식)", "k_l raw", "k_l PYR-beta", "k_l U", "MB raw", "MB PYR-beta", "MB U"],
                      [[e["layer"], f"{e['r_PYR-beta-raw']:.4f}", f"{e['r_PYR-beta']:.4f}", f"{e['r_PYR-beta'] - e['r_PYR-beta-raw']:+.4f}", f"{e['k_mean_PYR-beta-raw']:.1f}", f"{e['k_mean_PYR-beta']:.1f}",
                        f"{e['k_mean_U']:.1f}", f"{e['mb_mean_PYR-beta-raw']:.2f}", f"{e['mb_mean_PYR-beta']:.2f}", f"{e['mb_mean_U']:.2f}"] for e in h["layers"]]))
    L.append(f"HOLD 398장에서 PYR-beta-raw의 최소 k_l = {ktab['min_k_overall']['PYR-beta-raw']}행(마지막 layer, N = {ktab['N_min']}), k_l = 0인 layer 0개. "
             f"N = {ktab['median_N_used']}(중간값 이미지)에서 마지막 layer: raw {ktab['k_at_median_N']['PYR-beta-raw'][-1]}행, PYR-beta {ktab['k_at_median_N']['PYR-beta'][-1]}행, U {ktab['k_at_median_N']['U'][-1]}행. "
             "Σ k_l = 32·ceil(0.25N)은 세 벡터 모두 동일(G3). ![vectors](fig_vectors_pyr.png)\n")
    L.append("## Persistence·session\n")
    L.append(md_table(["Method", "Stored content %", "Store MB mean / p50 / p95", "OS write MB", "Persistence ms mean / p95 (native writer)", "Activation ms mean / p95"],
                      [[s_["name"], f"{s_['stored_content_pct']:.3f}", f"{s_['store_mb_mean']:.2f} / {s_['store_mb_p50']:.2f} / {s_['store_mb_p95']:.2f}", f"{s_['os_write_mb_mean']:.2f}",
                        f"{s_['persistence_ms_mean']:.1f} / {s_['persistence_ms_p95']:.1f} ({s_['native_writer_ms_mean']:.1f})", f"{s_['activation_ms_mean']:.1f} / {s_['activation_ms_p95']:.1f}"] for s_ in h["storage"]]))
    L.append(md_table(["Method", "Shared-stream session mean / p50 / p95 ms", "Standalone-equivalent mean / p50 / p95 ms", "Provenance"],
                      [[s_["name"], f"{s_['shared_mean_ms']:.1f} / {s_['shared_p50_ms']:.1f} / {s_['shared_p95_ms']:.1f}", f"{s_['standalone_mean_ms']:.1f} / {s_['standalone_p50_ms']:.1f} / {s_['standalone_p95_ms']:.1f}", s_["provenance"]] for s_ in h["sessions"]]))
    L.append("## 실행 기록\n")
    L.append(f"- 실제 요청 수: {audit['requests']:,} / {audit['expected_requests']:,} ({len(hconfig['methods'])} arm × 12,183), 이미지 {audit['images']} / 398. 실패·스킵 요청 0. "
             f"실패한 attempt 기록: {len(failures_file)}건, 중단(interruption) 기록: {len(interruptions)}건.\n"
             f"- HOLD main 소요: {elapsed_h:.2f} h (첫 시작 → 완료, wall). 삭제: run-local scratch store만(이미지 {dele['hold_main']['images']}개, 파일 {dele['hold_main']['removed_files']:,}개, 정확한 SHA allowlist 후), "
             f"gate store {dele['gpu_gate_cases_deleted']} case, 기존 runs/results/stores 삭제 0.\n"
             f"- Artifact protection: {protection['status']} (SHA {protection['files_before']} → {protection['files_after']} files, kvstore inventory {protection['kvstore_files_before']}, 다른 run inventory {protection['runs_files_before']}).\n"
             f"- freeze amendment: {', '.join(protection['amendment_files']) if protection['amendment_files'] else '없음'}.\n")
    L.append("## 한계와 NOT_MEASURED\n")
    L.append("- recompute(ReComp) arm은 이번 실행에서 돌리지 않았다(선택 arm, 실행 전 사용자 허용 없음 → 기본 제외). ReComp·FullLoad 대비 같은 실행 안 TTFT·session 비교는 NOT_MEASURED이며, 정확도만 cross-run(결정성 확인)으로 비교했다.\n"
             "- 원 PyramidKV의 attention 기반 token 선택은 구현·측정하지 않았다(layer별 예산 모양만 적용). 따라서 이 결과는 \"PyramidKV 재현\"이 아니라 \"PyramidKV layer 예산 schedule 적용\"이다.\n"
             "- CI가 0을 포함한다는 것은 동등성 증명이 아니다(이미지 398개 cluster bootstrap). 질문 유형별 수치는 탐색적이다.\n"
             "- H2D bytes, 독립 GPU footprint, NAND 수준 coldness, O_DIRECT serving, Qwen2.5-VL/GLM은 NOT_MEASURED(v2와 동일).\n"
             "- v2 arm과의 TTFT 비교는 하지 않았다(cross-run). U_packed의 실행 간 TTFT 차이는 참고치로만 적었다.\n")
    (res / "REPORT.md").write_text("\n".join(L))
    (res / "REPRODUCE.md").write_text(f"""# Reproduce (PyramidKV beta=20 HOLD add-on)

```bash
cd {ROOT}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.:scripts:tests PYTHONDONTWRITEBYTECODE=1
PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python
RUN=runs/layer_budget_kv25_pyrbeta_<new timestamp>
$PY scripts/163_make_pyrbeta_raw_vector.py --run $RUN --stage vectors      # copy U / PYR-beta (SHA check), build PYR-beta-raw, k_l tables
# write / check docs/layer_budget_kv25_pyrbeta_contract.md, then:
$PY scripts/163_make_pyrbeta_raw_vector.py --run $RUN --stage freeze       # contract SHA, storage plan, protection before, hold_arms.json
$PY scripts/166_validate_layer_budget_pyrbeta.py --run $RUN --stage g3
$PY scripts/166_validate_layer_budget_pyrbeta.py --run $RUN --stage grmin
$PY scripts/166_validate_layer_budget_pyrbeta.py --run $RUN --stage cpu
$PY scripts/166_validate_layer_budget_pyrbeta.py --run $RUN --stage gpu     # G6/G7, five fixed cases, both vectors
$PY scripts/164_eval_layer_budget_pyrbeta_hold.py --run $RUN --stage freeze
$PY scripts/164_eval_layer_budget_pyrbeta_hold.py --run $RUN --stage smoke
$PY scripts/166_validate_layer_budget_pyrbeta.py --run $RUN --stage gdet --phase smoke
$PY scripts/164_eval_layer_budget_pyrbeta_hold.py --run $RUN --stage main   # --resume after an interruption
$PY scripts/165_report_layer_budget_pyrbeta.py --run $RUN                   # audit, G-det main, decisions, cross-run, protection after
```
A new reproduction must use a new run root; existing runs/results/stores are never modified. HOLD payloads are deleted after each
image's audit and commit (run-local scratch only) and must be regenerated to re-serve. The optional recompute arm is added with
`--with-recompute` at the 163 freeze stage (only when allowed before execution).
""")
    print(json.dumps({"report": str(res / "REPORT.md"), "primary": {k: {x: v[x] for x in ("verdict", "delta_followup_pp", "ci95_pp", "ttft_ratio")} for k, v in decisions.items()},
                      "floor_effect": {x: fe[x] for x in ("classification", "delta_pp", "ci95_pp", "paper_default")}, "audit": audit["status"], "g_det_main": gdet["status"],
                      "cross_run": xr["status"], "protection": protection["status"], "deletion": dele["status"], "main_hours": elapsed_h}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

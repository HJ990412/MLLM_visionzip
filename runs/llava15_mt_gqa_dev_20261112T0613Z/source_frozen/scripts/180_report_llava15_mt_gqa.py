#!/usr/bin/env python3
"""Report of the LLaVA-1.5-7B MT-GQA HOLD (contract docs/llava15_mt_gqa_contract.md, section 7).

Independent re-audit of every committed image (179 audit), T1 cross-arm identity check, paired
analysis (image-cluster bootstrap, 10,000 resamples, seed 1234), the mechanical records
(A) main operating point, (B) KV-size axis, (C) layer-allocation verdict, (D) curve + figure,
(E) stage times / persistence / sessions, (F) T1 sanity, protection after, deletion log,
REPORT.md / RESULT_RECORD.md / REPRODUCE.md.  Every threshold comes from the frozen config.

Usage: --run RUN
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import numpy as np

import llava15_support as S
from llava15_support import ROOT, read, sha
from mmimpress.recompute_kv25 import paired_bootstrap, sha_manifest, stat_inventory

SEED, RESAMPLES = 1234, 10000
RECOMP, U100, U50, U25, PYR = S.METHODS
PACKED = [U100, U50, U25, PYR]
name = lambda m: S.LABEL[m]
FOLLOW = lambda k, v: k[1] > 1
ALL = lambda k, v: True
T1 = lambda k, v: k[1] == 1
STATIC_DEVIATIONS = [
    "LG3: 프롬프트의 '독립적인 prefix-only forward'를 문자 그대로(길이 P의 짧은 forward) 구현하면 BF16 GEMM 반올림이 tensor shape에 의존해 layer ≥ 1에서 bitwise가 성립하지 않는다"
    "(개발 probe에서 확인, 저장소 선례 scripts/30·docs/recompute_kv25_ablation_contract.md). freeze 전에 계약에서 LG3 참조를 '질문 자리를 filler 토큰으로 바꾼 같은 길이의 독립 forward(3종)'로 정의했다. "
    "bitwise·전 layer·K/V 기준은 그대로다. 문자 그대로의 짧은 forward는 LG3-literal 진단으로 실행해 수치만 기록했다(gate 아님).",
    "LG6: '전체 길이 참조'는 576행이 모두 실제 값인 cache에 layer별 가시 집합을 mask로만 적용한 독립 구현이며, 행 배치는 production과 같은 저장 순서다(기존 LLaVA G2와 같은 배치·같은 1e-4 기준). "
    "원래 token 순서로 배치한 변형은 key 합산 순서가 달라 BF16 반올림 수준의 차이가 생기므로 진단으로만 기록했다(gate 아님).",
    "(E)의 H2D와 scatter는 HOLD 경로에서 분리 측정하지 않았다. hit 경로는 LLaVA-NeXT와 같은 `PrefixCache.write`(FP16→BF16 변환 + H2D + index write가 한 호출)를 그대로 쓰므로 "
    "HOLD 수치는 'H2D·scatter·mask' 합산이다. 분해 수치는 gate fixture 10장에서 계측용 복제 경로로 잰 진단값이다(HOLD 측정이 아님).",
    "HOLD 프롬프트는 프롬프트 1절의 단일 턴 템플릿이 아니라 '기존 멀티턴 형식'(LLaVA-NeXT HOLD 164와 같은 `Current question Q{t}` 형식)을 모든 턴에 썼다. 단일 턴 템플릿은 runner.prompt()에 기록만 했다.",
    "PYR-b20 벡터 문서: PyramidKV run의 PYR-beta-raw json을 그대로 복사해 보관하고 ratio만 가져왔다(새로 계산하지 않음, ratio SHA 동일). 복사본의 clip 범위는 [0.0, 0.6]이지만 "
    "프롬프트 지시대로 모든 layer_budget 호출은 r_min=0.0, r_max=1.0을 명시한다. 최대 ratio가 0.4875라 k_l은 두 범위에서 같다(Phase 0에서 확인).",
    "실행 환경: conda env mllm_ft의 Python을 절대 경로로 호출했다(셸 PATH의 python은 다른 venv). transformers 4.57.6 / bitsandbytes 0.49.2 / torch 2.5.1+cu121.",
    "개발 rehearsal: 공식 freeze 전에 gate fixture 이미지로 코드 동작을 확인했다(runs/llava15_mt_gqa_dev_*; 측정이 아니며 보고 수치에 쓰지 않음).",
]


def jl(path):
    with Path(path).open() as f:
        return [json.loads(l) for l in f if l.strip()]


def _default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o).__name__)


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=1, allow_nan=False, default=_default, ensure_ascii=False); f.write("\n"); f.flush(); os.fsync(f.fileno())


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


def stats(v):
    a = np.asarray(v, dtype=float)
    return float(a.mean()), float(np.median(a)), float(np.percentile(a, 95))


def md_table(head, rows):
    return "| " + " | ".join(head) + " |\n|" + "|".join("---" for _ in head) + "|\n" + "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in rows) + "\n"


def fmt(d, lo, hi, nd=3):
    return f"{d:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def slim(r):
    res = r["result"]
    proc = res.get("processor") or {}
    return {"method_id": r["method_id"], "image_id": r["image_id"], "dialog_id": r["dialog_id"], "turn_id": r["turn_id"], "score": r["score"],
            "prediction": r["prediction"], "ttft_ms": r["ttft_ms"], "request_e2e_ms": r["request_e2e_ms"], "ssd_read_bytes": r["ssd_read_bytes"],
            "ssd_preads": r["ssd_preads"], "pread_calls": r.get("pread_calls", 0), "preadv_calls": r.get("preadv_calls", 0),
            "global_dialog_ordinal": r["global_dialog_ordinal"], "k_per_layer": r.get("k_per_layer"), "N_content": r["N_content"],
            "tokens": res["generated_token_ids"], "packed_read_ms": res.get("packed_read_ms"), "chunk_io_ms": res.get("chunk_io_ms"),
            "scatter_ms": res.get("scatter_ms"), "prepare_ms": res.get("prepare_ms"), "prompt_prepare_ms": res.get("prompt_prepare_ms"),
            "prefill_ms": res.get("prefill_ms"), "decode_ms": res.get("decode_ms"), "processor_total_ms": proc.get("processor_total_ms"),
            "image_preprocess_ms": proc.get("image_preprocess_ms"), "peak_gpu_allocated_bytes": res.get("peak_gpu_allocated_bytes")}


def load_and_audit(run, phase, config, manifest, E):
    rows, attempts = [], {}
    audit = {"phase": phase, "images": 0, "requests": 0, "failures": [], "per_image_status": {}, "checks": 0}
    for cp in sorted((run / phase / "images").glob("*/COMMITTED.json")):
        doc = read(cp); folder = cp.parent / doc["attempt_id"]
        if sha(folder / "raw.jsonl") != doc["raw_sha256"]:
            raise RuntimeError("committed raw hash mismatch: " + str(folder))
        if doc["config_sha256"] != config["_file_sha256"]:
            raise RuntimeError("committed image belongs to another freeze: " + str(folder))
        full = jl(folder / "raw.jsonl")
        receipt = E.audit_image_rows(full, manifest, config, phase, cp.parent.name, folder)
        audit["images"] += 1; audit["requests"] += len(full); audit["checks"] += receipt["checks"]
        audit["per_image_status"][cp.parent.name] = receipt["status"]
        if receipt["status"] != "PASS":
            audit["failures"].append({"image_id": cp.parent.name, "failures": receipt["failures"][:5]})
        rows += [slim(r) for r in full]; attempts[cp.parent.name] = folder
    audit["expected_requests"] = manifest["expected_requests"][phase]
    audit["expected_images"] = 398 if phase == "main" else len({d["image_id"] for d in manifest["dialogues"] if d["dialog_id"] in manifest["smoke_dialogue_ids"]})
    audit["failed_attempt_records"] = sorted(str(p.parent.relative_to(run)) for p in (run / phase / "images").glob("*/attempt_*/failure.json"))
    audit["requests_per_arm"] = {m: sum(1 for r in rows if r["method_id"] == m) for m in config["methods"]}
    audit["status"] = "PASS" if (not audit["failures"] and audit["requests"] == audit["expected_requests"] and audit["images"] == audit["expected_images"]
                                 and len(set(audit["requests_per_arm"].values())) == 1) else "FAIL"
    return rows, attempts, audit


def index(rows):
    by = defaultdict(dict)
    for r in rows:
        by[r["method_id"]][(r["dialog_id"], r["turn_id"])] = r
    return by


def per_image(by_a, by_b, images, img_of, flt, key="score"):
    acc = {img: [0.0, 0.0, 0] for img in images}
    for k, v in by_a.items():
        if flt(k, v):
            e = acc[img_of[k[0]]]; e[0] += v[key]; e[1] += (by_b[k][key] if by_b is not None else 0.0); e[2] += 1
    return [tuple(e) for e in acc.values() if e[2] > 0]


def paired(by_a, by_b, images, img_of, flt, key="score"):
    per = per_image(by_a, by_b, images, img_of, flt, key)
    d = paired_bootstrap(per, RESAMPLES, SEED)
    d["images_with_data"] = len(per); d["n"] = int(sum(e[2] for e in per))
    return d


def ratio_bootstrap(per):
    """Image-cluster bootstrap of mean(a) / mean(b) (same resampling scheme and seed as paired_bootstrap)."""
    arr = np.asarray(per, dtype=np.float64)
    point = arr[:, 0].sum() / arr[:, 1].sum()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(arr), size=(RESAMPLES, len(arr)))
    s = arr[idx]
    boot = s[..., 0].sum(1) / s[..., 1].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"ratio": float(point), "ci_low": float(lo), "ci_high": float(hi), "resamples": RESAMPLES, "seed": SEED, "unit": "image"}


def cells(by_a, by_b, flt):
    c = {"both": 0, "a_only": 0, "b_only": 0, "neither": 0}; agree = []
    for k, v in by_a.items():
        if not flt(k, v):
            continue
        x, y = v["score"], by_b[k]["score"]
        c["both" if x and y else "a_only" if x else "b_only" if y else "neither"] += 1
        agree.append(v["prediction"] == by_b[k]["prediction"])
    return c, float(np.mean(agree))


def t1_identity(by, methods, expected=4061):
    """Every arm serves T1 from full pixels: the generated ids of the same request must be identical across arms."""
    ref = methods[0]
    keys = [k for k in by[ref] if k[1] == 1]
    mism = []
    for k in keys:
        toks = {m: by[m][k]["tokens"] for m in methods}
        if any(t != toks[ref] for t in toks.values()):
            mism.append({"dialog_id": k[0], "tokens": toks, "predictions": {m: by[m][k]["prediction"] for m in methods}})
    return {"requests": len(keys), "token_identical": len(keys) - len(mism), "mismatch_count": len(mism), "mismatches": mism[:200],
            "status": "PASS" if not mism and len(keys) == expected else "PENDING_REVIEW"}


def analysis(run, rows, config, attempts, res):
    methods = config["methods"]
    by = index(rows)
    images = sorted({r["image_id"] for r in rows})
    img_of = {r["dialog_id"]: r["image_id"] for r in rows}
    quality = []
    for m in methods:
        d = by[m]
        turns = {t: [v["score"] for k, v in d.items() if k[1] == t] for t in (1, 2, 3)}
        follow = turns[2] + turns[3]; allv = turns[1] + follow
        ci = paired(d, None, images, img_of, FOLLOW)
        quality.append({"method": m, "name": name(m), "T1_pct": 100 * np.mean(turns[1]), "T2_pct": 100 * np.mean(turns[2]), "T3_pct": 100 * np.mean(turns[3]),
                        "all_pct": 100 * np.mean(allv), "all_correct": int(sum(allv)), "all_n": len(allv),
                        "followup_pct": 100 * np.mean(follow), "followup_correct": int(sum(follow)), "followup_n": len(follow),
                        "followup_ci95_lo_pct": 100 * ci["ci_low"], "followup_ci95_hi_pct": 100 * ci["ci_high"]})
    comparisons = [(U25, RECOMP), (U50, RECOMP), (U100, RECOMP), (PYR, RECOMP), (PYR, U25), (U50, U25), (U100, U50), (U100, U25), (PYR, U50), (PYR, U100)]
    pairs = []
    for a, b in comparisons:
        for scope, flt in (("followup", FOLLOW), ("all", ALL), ("T1", T1), ("T2", lambda k, v: k[1] == 2), ("T3", lambda k, v: k[1] == 3)):
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
        tci = paired(by[m], None, images, img_of, FOLLOW, key="ttft_ms")
        entry = {"method": m, "name": name(m), "ttft_mean_ms": t[0], "ttft_p50_ms": t[1], "ttft_p95_ms": t[2], "ttft_mean_ci95_lo_ms": tci["ci_low"], "ttft_mean_ci95_hi_ms": tci["ci_high"],
                 "e2e_mean_ms": e[0], "e2e_p50_ms": e[1], "e2e_p95_ms": e[2], "ssd_mb_per_followup": float(np.mean([v["ssd_read_bytes"] for v in f])) / 1e6,
                 "read_calls_mean": float(np.mean([v["ssd_preads"] for v in f])), "pread_calls_mean": float(np.mean([v["pread_calls"] for v in f])),
                 "preadv_calls_mean": float(np.mean([v["preadv_calls"] for v in f])), "followups": len(f),
                 "peak_gpu_alloc_mb_mean": float(np.mean([v["peak_gpu_allocated_bytes"] for v in f])) / 1e6}
        for key in ("prompt_prepare_ms", "packed_read_ms", "chunk_io_ms", "scatter_ms", "prepare_ms", "prefill_ms", "decode_ms", "processor_total_ms", "image_preprocess_ms"):
            vals = [v[key] for v in f if v.get(key) is not None]
            entry[f"{key}_mean"] = float(np.mean(vals)) if vals else None
        if entry["prepare_ms_mean"] is not None:
            entry["cache_alloc_and_other_prepare_ms_mean"] = entry["prepare_ms_mean"] - entry["chunk_io_ms_mean"] - entry["scatter_ms_mean"]
        t1 = stats([v["ttft_ms"] for k, v in by[m].items() if k[1] == 1])
        entry["t1_ttft_mean_ms"], entry["t1_ttft_p50_ms"], entry["t1_ttft_p95_ms"] = t1
        latency.append(entry)
    storage, setup = [], {}
    for m in PACKED:
        kind = S.kind_of(m); rs = []
        for img in images:
            folder = attempts[img]
            per = read(folder / f"{kind}_persistence.json"); act = read(folder / f"{m}_activation.json"); meta = read(folder / f"{kind}_meta.json")
            tm = per.get("timing_ms", {})
            rs.append({"bytes": per["file_logical_bytes"], "allocated": per["allocated_bytes"], "os_write": per["os_write_bytes"], "persist_ms": per["service_persistence_ms"],
                       "writer_ms": per["native_writer_wall_ms"], "hash_ms": per["completion_full_hash_ms"], "act_ms": act["outer_wall_ms"],
                       "rows_total": meta["stored_content_count_total"], "packed_bytes": meta["packed_total_bytes"], "files": len(per["files"]),
                       "fsync_ms": tm.get("durability_ms", 0.0), "write_ms": tm.get("ssd_write_ms", 0.0), "materialize_ms": tm.get("repack_ms", 0.0)})
            setup[(m, img)] = per["service_persistence_ms"] + act["outer_wall_ms"]
        b = stats([x["bytes"] for x in rs]); pm = stats([x["persist_ms"] for x in rs]); am = stats([x["act_ms"] for x in rs])
        storage.append({"method": m, "name": name(m), "images": len(rs), "stored_content_pct": 100 * float(np.mean([x["rows_total"] / (32 * 576) for x in rs])),
                        "files_per_store": int(rs[0]["files"]), "store_mb_mean": b[0] / 1e6, "store_mb_p50": b[1] / 1e6, "store_mb_p95": b[2] / 1e6,
                        "packed_file_mb": float(np.mean([x["packed_bytes"] for x in rs])) / 1e6, "allocated_mb_mean": float(np.mean([x["allocated"] for x in rs])) / 1e6,
                        "os_write_mb_mean": float(np.mean([x["os_write"] for x in rs])) / 1e6, "persistence_ms_mean": pm[0], "persistence_ms_p50": pm[1], "persistence_ms_p95": pm[2],
                        "native_writer_ms_mean": float(np.mean([x["writer_ms"] for x in rs])), "completion_hash_ms_mean": float(np.mean([x["hash_ms"] for x in rs])),
                        "writer_ssd_write_ms_mean": float(np.mean([x["write_ms"] for x in rs])), "writer_fsync_rename_ms_mean": float(np.mean([x["fsync_ms"] for x in rs])),
                        "writer_materialize_repack_ms_mean": float(np.mean([x["materialize_ms"] for x in rs])),
                        "activation_ms_mean": am[0], "activation_ms_p50": am[1], "activation_ms_p95": am[2], "provenance": "MEASURED"})
    dialogs = sorted({r["dialog_id"] for r in rows})
    first_dialog = {}
    for r in sorted(rows, key=lambda x: x["global_dialog_ordinal"]):
        first_dialog.setdefault(r["image_id"], r["dialog_id"])
    sessions, sess_vals = [], {}
    for m in methods:
        shared, standalone = [], []
        for d in dialogs:
            e2e = sum(by[m][(d, t)]["request_e2e_ms"] for t in (1, 2, 3))
            su = setup.get((m, img_of[d]), 0.0)
            shared.append(e2e + (su if first_dialog.get(img_of[d]) == d else 0.0)); standalone.append(e2e + su)
        sess_vals[m] = (shared, standalone)
        s1, s2 = stats(shared), stats(standalone)
        sessions.append({"method": m, "name": name(m), "setup_ms_per_image_mean": float(np.mean([setup[(m, i)] for i in images])) if m in PACKED else 0.0,
                         "shared_mean_ms": s1[0], "shared_p50_ms": s1[1], "shared_p95_ms": s1[2],
                         "standalone_mean_ms": s2[0], "standalone_p50_ms": s2[1], "standalone_p95_ms": s2[2],
                         "provenance": "MEASURED (no KV setup)" if m == RECOMP else "MEASURED (shared-stream) / DERIVED (standalone-equivalent)"})
    for a, b in [(U25, RECOMP), (U50, RECOMP), (U100, RECOMP), (PYR, U25)]:
        for kind_i, kind in ((0, "shared"), (1, "standalone")):
            per = []
            for img in images:
                idx = [i for i, d in enumerate(dialogs) if img_of[d] == img]
                per.append((sum(sess_vals[a][kind_i][i] for i in idx), sum(sess_vals[b][kind_i][i] for i in idx), len(idx)))
            d = paired_bootstrap(per, RESAMPLES, SEED)
            pairs.append({"a": a, "b": b, "scope": f"session_{kind}", "metric": "session_ms", "delta": d["mean"], "ci95_lo": d["ci_low"], "ci95_hi": d["ci_high"]})
    # nested selections (audit already checks the hits of one image; here: every image)
    csv_write(res / "quality.csv", quality); csv_write(res / "latency_io.csv", latency); csv_write(res / "storage.csv", storage)
    csv_write(res / "paired_analysis.csv", pairs); csv_write(res / "sessions.csv", sessions)
    for nm, obj in (("quality", quality), ("latency_io", latency), ("storage", storage), ("paired_analysis", pairs), ("sessions", sessions)):
        write_json(res / f"{nm}.json", obj)
    return {"quality": quality, "latency": latency, "storage": storage, "pairs": pairs, "sessions": sessions, "images": len(images), "dialogs": len(dialogs),
            "by": by, "img_of": img_of, "image_list": images}


def pair_of(h, a, b, scope="followup", metric=None):
    return next(p for p in h["pairs"] if p["a"] == a and p["b"] == b and p["scope"] == scope and p.get("metric") == metric)


def record_a(h):
    lat = {x["method"]: x for x in h["latency"]}; q = {x["method"]: x for x in h["quality"]}
    d = pair_of(h, U25, RECOMP)
    r = ratio_bootstrap(per_image(h["by"][U25], h["by"][RECOMP], h["image_list"], h["img_of"], FOLLOW, "ttft_ms"))
    zero = d["ci95_lo_pp"] <= 0 <= d["ci95_hi_pp"]
    return {"operating_point": name(U25), "baseline": name(RECOMP), "scope": "HOLD followup (T2-T3)",
            "accuracy_pct": q[U25]["followup_pct"], "baseline_accuracy_pct": q[RECOMP]["followup_pct"], "delta_pp": d["delta_pp"], "ci95_pp": [d["ci95_lo_pp"], d["ci95_hi_pp"]],
            "ci_includes_zero": bool(zero), "wording": "차이 미입증" if zero else ("ReComp보다 낮음 (CI 0 제외)" if d["delta_pp"] < 0 else "ReComp보다 높음 (CI 0 제외)"),
            "relative_to_original_pct": 100 * q[U25]["followup_pct"] / q[RECOMP]["followup_pct"],
            "ttft_ms": {"mean": lat[U25]["ttft_mean_ms"], "p50": lat[U25]["ttft_p50_ms"], "p95": lat[U25]["ttft_p95_ms"]},
            "baseline_ttft_ms": {"mean": lat[RECOMP]["ttft_mean_ms"], "p50": lat[RECOMP]["ttft_p50_ms"], "p95": lat[RECOMP]["ttft_p95_ms"]},
            "ttft_ratio_mean": lat[U25]["ttft_mean_ms"] / lat[RECOMP]["ttft_mean_ms"], "ttft_ratio_ci95": [r["ci_low"], r["ci_high"]],
            "ttft_ratio_p50": lat[U25]["ttft_p50_ms"] / lat[RECOMP]["ttft_p50_ms"], "ttft_ratio_p95": lat[U25]["ttft_p95_ms"] / lat[RECOMP]["ttft_p95_ms"],
            "ttft_reduction_pct": 100 * (1 - lat[U25]["ttft_mean_ms"] / lat[RECOMP]["ttft_mean_ms"]), "ssd_mb_per_followup": lat[U25]["ssd_mb_per_followup"]}


def record_b(h, config):
    lat = {x["method"]: x for x in h["latency"]}
    def ratio(a, b):
        r = ratio_bootstrap(per_image(h["by"][a], h["by"][b], h["image_list"], h["img_of"], FOLLOW, "ttft_ms"))
        return {"value": lat[a]["ttft_mean_ms"] / lat[b]["ttft_mean_ms"], "ci95": [r["ci_low"], r["ci_high"]], "numerator_ms": lat[a]["ttft_mean_ms"], "denominator_ms": lat[b]["ttft_mean_ms"]}
    hyp = config["kv_size_axis"]["prior_hypothesis"]
    out = {"R100": ratio(U100, RECOMP), "S25": ratio(U25, U100), "S50": ratio(U50, U100), "definition": {k: config["kv_size_axis"][k] for k in ("R100", "S25", "S50")},
           "kv_mb_per_image": S.EXPECTED_BYTES["U100"] / 1e6, "references": config["kv_size_axis"]["references"],
           "reference_note": "Qwen2.5-VL 값은 같은 packed 경로(22 MB/이미지). LLaVA-NeXT-7B 값은 v1 run의 파일 분리 경로(100% = FullLoad 65 pread, 25% = unpacked)이며 "
                             "LLaVA-NeXT의 100% packed 지점은 측정된 적이 없어 직접 비교가 아니다. 절대 TTFT는 실행 간 비교하지 않는다.",
           "prior_hypothesis": hyp}
    out["hypothesis_check"] = {"R100_between_0.462_and_1.39": bool(hyp["R100"][0] <= out["R100"]["value"] <= hyp["R100"][1]),
                               "S25_between_0.34_and_0.892": bool(hyp["S25"][0] <= out["S25"]["value"] <= hyp["S25"][1])}
    out["status"] = "REPORTED (not a verdict)"
    return out


def record_c(h):
    lat = {x["method"]: x for x in h["latency"]}
    d = pair_of(h, PYR, U25)
    delta, lo, hi = float(d["delta_pp"]), float(d["ci95_lo_pp"]), float(d["ci95_hi_pp"])
    ci_excludes_zero = bool(lo > 0 or hi < 0)
    mb_ok = bool(lat[PYR]["ssd_mb_per_followup"] <= lat[U25]["ssd_mb_per_followup"] * 1.02)
    ttft_ok = bool(lat[PYR]["ttft_mean_ms"] <= lat[U25]["ttft_mean_ms"] * 1.05)
    if delta >= 1.0 and ci_excludes_zero and mb_ok and ttft_ok:
        verdict = "GO"
    elif (delta >= 0.5 and not ci_excludes_zero) or (delta >= 1.0 and ci_excludes_zero and (mb_ok != ttft_ok)):
        verdict = "PARTIAL"
    else:
        verdict = "NO-GO"
    return {"verdict": verdict, "vector": name(PYR), "control": name(U25), "delta_followup_pp": delta, "ci95_pp": [lo, hi], "ci_excludes_zero": ci_excludes_zero,
            "ssd_mb_vector": lat[PYR]["ssd_mb_per_followup"], "ssd_mb_uniform": lat[U25]["ssd_mb_per_followup"], "ssd_ok_1.02": mb_ok,
            "ttft_mean_vector": lat[PYR]["ttft_mean_ms"], "ttft_mean_uniform": lat[U25]["ttft_mean_ms"], "ttft_ratio": lat[PYR]["ttft_mean_ms"] / lat[U25]["ttft_mean_ms"],
            "ttft_ok_1.05": ttft_ok, "a_only_b_only_both_neither": [d["a_only"], d["b_only"], d["both"], d["neither"]], "prediction_agreement": d["prediction_agreement"],
            "same_rule_other_models": {"LLaVA-NeXT-7B": "GO (+1.244 pp, PYR-beta-raw vs U_packed)", "Qwen2.5-VL": "NO-GO (-0.899 pp)"},
            "wording": "CI가 0을 포함: 차이 미입증" if not ci_excludes_zero else "CI 0 제외"}


def figure(res, h):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3dc", "#fcfcfb"
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"            # categorical slots 1-3 (all-pairs safe); shape is the second encoding
    lat = {x["method"]: x for x in h["latency"]}; q = {x["method"]: x for x in h["quality"]}
    def pt(m):
        return (lat[m]["ttft_mean_ms"], q[m]["followup_pct"],
                [[lat[m]["ttft_mean_ms"] - lat[m]["ttft_mean_ci95_lo_ms"]], [lat[m]["ttft_mean_ci95_hi_ms"] - lat[m]["ttft_mean_ms"]]],
                [[q[m]["followup_pct"] - q[m]["followup_ci95_lo_pct"]], [q[m]["followup_ci95_hi_pct"] - q[m]["followup_pct"]]])
    fig, ax = plt.subplots(figsize=(7.6, 4.8), facecolor=surface)
    ax.set_facecolor(surface)
    curve = [U25, U50, U100]
    ax.plot([lat[m]["ttft_mean_ms"] for m in curve], [q[m]["followup_pct"] for m in curve], color=blue, lw=2, zorder=2)
    for m, label in zip(curve, ("25%", "50%", "100%")):
        x, y, xe, ye = pt(m)
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="o", ms=8, color=blue, ecolor=blue, elinewidth=1.2, capsize=3, mec=surface, mew=2, zorder=3)
        ax.annotate(f"Uniform {label}", (x, y), textcoords="offset points", xytext=(8, -14), fontsize=9, color=ink)
    x, y, xe, ye = pt(PYR)
    ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="s", ms=8, color=orange, ecolor=orange, elinewidth=1.2, capsize=3, mec=surface, mew=2, zorder=4)
    ax.annotate("PYR-b20 (25%)", (x, y), textcoords="offset points", xytext=(8, 8), fontsize=9, color=ink)
    x, y, xe, ye = pt(RECOMP)
    ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="D", ms=8, color=aqua, ecolor=aqua, elinewidth=1.2, capsize=3, mec=surface, mew=2, zorder=4)
    ax.annotate("ReComp (no store)", (x, y), textcoords="offset points", xytext=(8, 8), fontsize=9, color=ink)
    ax.plot([], [], color=blue, marker="o", ms=7, lw=2, label="Uniform packed (25 / 50 / 100%)")
    ax.plot([], [], color=orange, marker="s", ms=7, lw=0, label="PYR-b20 packed (25%, layer-wise)")
    ax.plot([], [], color=aqua, marker="D", ms=7, lw=0, label="ReComp")
    ax.set_xlabel("TTFT mean, followup turns (ms)", color=ink)
    ax.set_ylabel("Followup accuracy (%)", color=ink)
    ax.set_title("LLaVA-1.5-7B, MT-GQA HOLD: accuracy vs TTFT", color=ink, fontsize=11, loc="left")
    ax.grid(color=grid, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8)
    ax.margins(x=0.12, y=0.18)
    leg = ax.legend(frameon=False, fontsize=8, loc="lower right")
    for t in leg.get_texts():
        t.set_color(ink)
    fig.text(0.01, 0.01, "Error bars: 95% CI, image-cluster bootstrap (10,000 resamples). 398 images, 8,122 followup requests per arm.", fontsize=7, color=muted)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(res / "fig_accuracy_vs_ttft.png", dpi=150, facecolor=fig.get_facecolor()); plt.close(fig)


def protection_after(run, res):
    before = read(run / "protection_before.json")
    excl = before["excluded_new_paths"]
    shas = {k: v for k, v in sha_manifest([ROOT / x for x in ["results", "docs", "mmimpress", "scripts", "tests", "README.md"]], ROOT, exclude=excl).items() if "__pycache__" not in k}
    kv = stat_inventory(sorted(ROOT.glob("kvstore*")), ROOT)
    runs = stat_inventory(sorted(p for p in (ROOT / "runs").iterdir() if not p.name.startswith("llava15_mt_gqa_")), ROOT)
    def inv_diff(b, a):
        return {"changed": sorted(k for k in b if k in a and a[k] != b[k]), "missing": sorted(set(b) - set(a)), "added": sorted(set(a) - set(b))}
    diff = {"sha_changed": sorted(k for k in before["sha256"] if k in shas and shas[k] != before["sha256"][k]),
            "sha_missing": sorted(set(before["sha256"]) - set(shas)), "sha_added": sorted(set(shas) - set(before["sha256"]))}
    kvd = inv_diff(before["kvstore_inventory"], kv); rund = inv_diff(before["runs_inventory"], runs)
    new_ok = all((ROOT / p).is_file() for p in S.NEW_PATHS)
    hard = [diff["sha_changed"], diff["sha_missing"], diff["sha_added"], kvd["changed"], kvd["missing"], kvd["added"], rund["changed"], rund["missing"]]
    protection = {"status": "PASS" if not any(hard) and new_ok else "FAIL", "files_before": len(before["sha256"]), "files_after": len(shas),
                  "kvstore_files_before": len(before["kvstore_inventory"]), "kvstore_files_after": len(kv), "runs_files_before": len(before["runs_inventory"]), "runs_files_after": len(runs),
                  "differences": {"sha": diff, "kvstore": kvd, "runs": {"changed": rund["changed"], "missing": rund["missing"], "added_count": len(rund["added"])}},
                  "runs_added_reported_only": rund["added"][:50], "new_files_of_this_run": S.NEW_PATHS, "source_modifications": read(run / "source_modifications.json"),
                  "scope": "full SHA256 of results/docs/mmimpress/scripts/tests/README.md (new paths of this run excluded); (size, mtime_ns, inode) of kvstore* and of every other runs/* root"}
    write_json(res / "protection_result.json", protection)
    write_json(run / "protection_after.json", {"stage": "after", "utc": time.time(), "sha256": shas, "kvstore_inventory": kv, "runs_inventory": runs})
    digest = lambda inv: hashlib.sha256(json.dumps(inv, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for stage, sh, kvi, ri, src in (("before", before["sha256"], before["kvstore_inventory"], before["runs_inventory"], "protection_before.json"), ("after", shas, kv, runs, "protection_after.json")):
        write_json(res / f"protection_{stage}.json", {"stage": stage, "full_file": str((run / src).relative_to(ROOT)), "full_file_sha256": sha(run / src), "sha256": sh,
                                                      "kvstore_inventory_files": len(kvi), "kvstore_inventory_digest_sha256": digest(kvi),
                                                      "runs_inventory_files": len(ri), "runs_inventory_digest_sha256": digest(ri)})
    return protection


def deletion_log(run, res, main_phase="main", main_images=398):
    phases = {}
    for phase in ("integration", "smoke", "main"):
        imgs, files, nbytes = [], 0, 0
        for cp in sorted((run / phase / "images").glob("*/COMMITTED.json")):
            folder = cp.parent / read(cp)["attempt_id"]
            rc = read(folder / "cleanup_receipt.json"); allow = read(folder / "cleanup_allowlist.json")
            assert rc["status"] == "PASS" and rc["removed_new_files"] == len(allow["file_sha256"])
            imgs.append({"image_id": cp.parent.name, "scratch": allow["scratch"], "removed_files": rc["removed_new_files"], "removed_bytes": allow.get("bytes"),
                         "allowlist_sha256": sha(folder / "cleanup_allowlist.json")})
            files += rc["removed_new_files"]; nbytes += allow.get("bytes") or 0
        phases[phase] = {"images": len(imgs), "removed_files": files, "removed_bytes": nbytes, "per_image": imgs}
    leftovers = sorted(str(p.relative_to(run)) for p in (run / "scratch").rglob("*") if p.is_file()) if (run / "scratch").exists() else []
    gpu = read(run / "gates" / "gpu.json"); cpu = read(run / "gates" / "cpu.json")
    gate_left = sorted(str(p.relative_to(run)) for p in (run / "gates" / "stores").rglob("*") if p.is_file()) if (run / "gates" / "stores").exists() else []
    superseded = sorted(p.name for p in run.glob("*.superseded_*"))
    log = {"scope": "run-local only: scratch/** (exact SHA256 allowlist after independent audit + atomic commit), gates/stores/** (per fixture image, guarded_rmtree), gates/cpu_fixture (CPU fixture)",
           "existing_runs_results_stores_deleted": 0, "phases": phases, "retained_scratch_files": leftovers,
           "gpu_gate_images_deleted": sum(1 for c in gpu["images"] if c.get("stores_deleted_after_record")), "gpu_gate_retained_files": gate_left,
           "cpu_fixture_deleted": cpu["LG8"].get("fixture_deleted"), "superseded_records_kept": superseded,
           "status": "PASS" if not leftovers and not gate_left and phases[main_phase]["images"] == main_images else "CHECK"}
    write_json(res / "deletion_log.json", log)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--rehearsal-smoke", action="store_true", help="dev only: build the report from the smoke phase into <run>/rehearsal_report (never a result)")
    args = ap.parse_args()
    run = Path(args.run).resolve(); res = ROOT / "results" / run.name
    assert run.parent == ROOT / "runs" and run.name.startswith("llava15_mt_gqa_")
    PHASE = "main"
    if args.rehearsal_smoke:
        assert run.name.startswith("llava15_mt_gqa_dev_"), "rehearsal reports are for dev runs only"
        PHASE, res = "smoke", run / "rehearsal_report"
    res.mkdir(parents=True, exist_ok=True)
    E = S.load_script("_llava15_hold", "179_eval_llava15_mt_gqa_hold.py")
    config, manifest = E.verify_freeze(run)
    complete = read(run / f"{PHASE}_execution_complete.json")
    rows, attempts, audit = load_and_audit(run, PHASE, config, manifest, E)
    _, _, smoke_audit = load_and_audit(run, "smoke", config, manifest, E)
    write_json(res / "independent_audit.json", {"main": audit, "smoke": smoke_audit,
                                                "auditor": "scripts/179 audit_image_rows re-run from the committed raw rows and evidence of every image (prompt / history lineage, EM, rotation, "
                                                           "k_l by library and by an independent integer Hamilton, bytes, single preadv, selection nesting, vision / scoring calls)"})
    h = analysis(run, rows, config, attempts, res)
    t1 = t1_identity(h["by"], config["methods"], 4061 if PHASE == "main" else len(manifest["smoke_dialogue_ids"]))
    write_json(res / "t1_identity.json", t1)
    A, B, C = record_a(h), record_b(h, config), record_c(h)
    lat = {x["method"]: x for x in h["latency"]}; q = {x["method"]: x for x in h["quality"]}
    gates = read(run / "gates" / "summary.json"); gpu = read(run / "gates" / "gpu.json"); cpu = read(run / "gates" / "cpu.json")
    starts = sorted(run.glob(f"{PHASE}_start_*.json")); interruptions = sorted(p.name for p in run.glob("interruption_*.json"))
    elapsed_h = (complete["finished_unix"] - read(starts[0])["started_unix"]) / 3600
    failures_file = sorted(str(p.relative_to(run)) for p in run.rglob("failure.json") if ".superseded_" not in str(p))
    amendments = sorted(p.name for p in run.glob("freeze_amendment_*.json"))
    final = bool(audit["status"] == "PASS" and smoke_audit["status"] == "PASS" and complete["status"] == "PASS" and gates["status"] == "PASS"
                 and gates["config_sha256"] == config["_file_sha256"] and audit["requests"] == S.EXPECT[PHASE])
    status = "FINAL" if final and t1["status"] == "PASS" else ("PENDING_REVIEW" if final else "NOT_FINAL")
    if PHASE != "main":
        status = "REHEARSAL_NOT_A_RESULT" 
    curve = [{"arm": name(m), "ratio_pct": 100 * S.TOTAL_RATIO[S.vector_of(m)] if m != RECOMP else None,
              "rows_per_layer": ("576" if m == U100 else "288" if m == U50 else "144" if m == U25 else "281…7 (합 4,608)" if m == PYR else "–"),
              "followup_pct": q[m]["followup_pct"], "followup_ci95_pct": [q[m]["followup_ci95_lo_pct"], q[m]["followup_ci95_hi_pct"]],
              "delta_vs_recompute_pp": None if m == RECOMP else pair_of(h, m, RECOMP)["delta_pp"],
              "delta_vs_recompute_ci95_pp": None if m == RECOMP else [pair_of(h, m, RECOMP)["ci95_lo_pp"], pair_of(h, m, RECOMP)["ci95_hi_pp"]],
              "ttft_mean_ms": lat[m]["ttft_mean_ms"], "ttft_p50_ms": lat[m]["ttft_p50_ms"], "ttft_p95_ms": lat[m]["ttft_p95_ms"],
              "ttft_over_recompute": lat[m]["ttft_mean_ms"] / lat[RECOMP]["ttft_mean_ms"], "ssd_mb_per_followup": lat[m]["ssd_mb_per_followup"]} for m in config["methods"]]
    decision = {"status": status, "final": final, "A_main_operating_point": A, "B_kv_size_axis": B, "C_layer_allocation": C,
                "rule": config["decision_rule"], "t1_identity": {k: t1[k] for k in ("status", "requests", "token_identical", "mismatch_count")},
                "audit_status": audit["status"], "requests": audit["requests"], "expected_requests": audit["expected_requests"], "images": audit["images"],
                "failed_or_skipped_requests": 0 if audit["status"] == "PASS" else None, "failed_attempt_records": failures_file, "interruptions": interruptions,
                "main_wall_hours_first_start_to_finish": elapsed_h, "amendments": amendments, "gates": gates["gates"],
                "bootstrap": config["bootstrap"], "arms": config["methods"]}
    write_json(res / "decision.json", decision)
    summary = {"status": status, "run": str(run.relative_to(ROOT)), "contract_sha256": config["contract_sha256"], "config_sha256": config["_file_sha256"],
               "model": {"id": config["model"], "revision": config["revision"], "versions": config["versions"], "visual_tokens": 576, "separators": 0,
                         "kv_mb_per_image_100pct": S.EXPECTED_BYTES["U100"] / 1e6},
               "requests": {"main": audit["requests"], "expected": audit["expected_requests"], "per_arm": audit["requests_per_arm"], "smoke": smoke_audit["requests"],
                            "integration": gates["LG10"]["requests"], "failed_or_skipped": 0 if audit["status"] == "PASS" else None},
               "curve": curve, "quality": h["quality"], "latency": h["latency"], "storage": h["storage"], "sessions": h["sessions"],
               "A_main_operating_point": A, "B_kv_size_axis": B, "C_layer_allocation": C, "F_t1_accuracy_pct": {name(m): q[m]["T1_pct"] for m in config["methods"]},
               "t1_identity": {k: t1[k] for k in ("status", "requests", "token_identical", "mismatch_count")},
               "gates": gates["gates"], "u100_vs_recompute_first_token_fixture": gpu["LG6_u100_vs_recompute"],
               "scatter_breakdown_fixture_DIAGNOSTIC": gpu["scatter_breakdown_DIAGNOSTIC_mean_ms"],
               "main_wall_hours": elapsed_h, "interruptions": interruptions, "amendments": amendments,
               "not_measured": ["HOLD 경로에서 H2D 단독 시간(합산 측정; 분해는 fixture 진단)", "NAND/controller 수준 cold read, O_DIRECT (DONTNEED는 page-cache hint)", "독립 GPU footprint",
                                "LLaVA-NeXT 100% packed 지점", "MT-VQA / Qwen / GLM / LLaVA-NeXT (이번에 실행하지 않음)", "FullLoad(파일 분리) arm", "PyramidKV의 attention 기반 token 선택"]}
    write_json(res / "summary.json", summary)
    # ---- copies
    shutil.copy2(run / "CONTRACT.md", res / "CONTRACT.md")
    for p in sorted((run / "vectors").glob("*.json")):
        shutil.copy2(p, res / p.name)
    for p in sorted((run / "gates").glob("*.json")):
        shutil.copy2(p, res / ("gate_" + p.name))
    for f, t in (("source_modifications.json", "source_modifications.json"), ("config.json", "run_config.json"), ("storage_plan.json", "storage_plan.json"),
                 ("phase0/model_record.json", "phase0_model_record.json"), ("phase0/k_table_N576.json", "phase0_k_table_N576.json"),
                 ("phase0/dataset_identity.json", "phase0_dataset_identity.json")):
        shutil.copy2(run / f, res / t)
    for p in amendments:
        shutil.copy2(run / p, res / p)
    try:
        figure(res, h)
        fig_ok = True
    except Exception as exc:
        write_json(res / "fig_accuracy_vs_ttft_error.json", {"error": repr(exc)})
        fig_ok = False
    dele = deletion_log(run, res, PHASE, audit["expected_images"])
    protection = protection_after(run, res)
    model = read(run / "phase0" / "model_record.json")
    sb = gpu["scatter_breakdown_DIAGNOSTIC_mean_ms"]

    # ------------------------------------------------------------------ REPORT
    L = []
    L.append("# LLaVA-1.5-7B 포팅 + MT-GQA HOLD (ReComp / U100 / U50 / U25 / PYR-b20 packed)\n")
    L.append(f"Run `{run.relative_to(ROOT)}`. 계약 [CONTRACT.md](CONTRACT.md) (SHA `{config['contract_sha256'][:16]}…`, gate·smoke·main 전에 freeze). "
             f"단위: MB = 10^6 B, ms, 정확도 차이는 pp. followup = T2–T3. CI = 이미지 단위 paired bootstrap 95% (10,000회, seed 1234).\n")
    L.append(f"상태: **{status}** — main {audit['requests']:,} / {audit['expected_requests']:,} 요청(arm당 {audit['requests_per_arm'][RECOMP]:,}), 이미지 {audit['images']} / 398, 실패·스킵 0, "
             f"독립 감사 {audit['status']}, T1 arm 간 토큰 일치 {t1['token_identical']:,} / {t1['requests']:,} ({t1['status']}).\n")
    L.append("## 이 실험에서 반드시 알아야 할 것\n")
    L.append(f"- **LLaVA-1.5-7B는 이미지당 시각 토큰이 576개로 고정이고 KV는 약 302 MB(301,989,888 B, FP16, 32 layer × K·V × 576행 × 8,192 B)이며 separator가 없다.** "
             f"전처리는 HF 기본 processor 그대로다({model['image_processor_class']}: 짧은 변 336 resize 후 336×336 중앙 자르기). 원 LLaVA-1.5 저장소는 정사각형 패딩이라 공개 GQA 수치와 조건이 다르다.\n"
             "- **모든 packed arm은 같은 중요도 순서(CLIP 끝에서 두 번째 layer의 CLS→patch attention head 합, 내림차순 stable 정렬)와 같은 I/O 경로(이미지당 파일 1개, hit당 preadv 1회, "
             "dense 버퍼 + layer별 mask)를 쓰고, 비율과 layer별 자르는 지점만 다르다.** 질문은 선택에 쓰지 않는다. 데이터로 무엇을 맞추거나 고르지 않았다.\n"
             "- **PYR-b20은 PyramidKV 공개 코드 식(β=20)의 layer 비율 그대로이며 attention 기반 token 선택은 쓰지 않았다.** 총 바이트는 U25와 같다.\n"
             "- **LLaVA-NeXT의 100% packed 지점은 측정된 적이 없다.** (B)의 LLaVA-NeXT 참고값은 다른 I/O 경로(파일 분리)의 값이다.\n"
             "- **MT-VQA·Qwen·GLM·LLaVA-NeXT는 이번에 실행하지 않았다.** 참고값은 선행 run의 기록이다. 절대 TTFT는 실행 간 비교하지 않는다.\n"
             "- 주 운영점은 25%로 실행 전에 고정했다. 50%·100%는 곡선의 점이다.\n")
    L.append("## (D) 곡선 — HOLD followup\n")
    L.append(md_table(["Arm", "비율", "layer당 행", "Followup % [95% CI]", "vs ReComp Δpp [CI]", "TTFT mean / p50 / p95 ms", "TTFT / ReComp", "SSD MB / followup"],
                      [[c["arm"], "–" if c["ratio_pct"] is None else f"{c['ratio_pct']:.0f}%", c["rows_per_layer"],
                        f"{c['followup_pct']:.3f} [{c['followup_ci95_pct'][0]:.3f}, {c['followup_ci95_pct'][1]:.3f}]",
                        "–" if c["delta_vs_recompute_pp"] is None else fmt(c["delta_vs_recompute_pp"], *c["delta_vs_recompute_ci95_pp"]),
                        f"{c['ttft_mean_ms']:.2f} / {c['ttft_p50_ms']:.2f} / {c['ttft_p95_ms']:.2f}", f"{c['ttft_over_recompute']:.3f}×", f"{c['ssd_mb_per_followup']:.2f}"] for c in curve]))
    L.append("Δ는 (행의 arm − ReComp)이다. CI가 0을 포함하면 '차이 미입증'이며 동등하다는 뜻이 아니다. T1은 모든 arm이 full pixel이고, 차이는 generated history가 갈라지는 효과를 포함한다.\n")
    if fig_ok:
        L.append("![accuracy vs TTFT](fig_accuracy_vs_ttft.png)\n")
    L.append("## (A) 주 운영점 — U25_packed vs ReComp\n")
    L.append(md_table(["항목", "값"], [
        ["Followup 정확도", f"U25 {A['accuracy_pct']:.3f}% vs ReComp {A['baseline_accuracy_pct']:.3f}%"],
        ["Δ pp [95% CI]", f"{fmt(A['delta_pp'], *A['ci95_pp'])} — {A['wording']}"],
        ["원래 성능 대비", f"{A['relative_to_original_pct']:.2f}%"],
        ["TTFT mean / p50 / p95 ms", f"U25 {A['ttft_ms']['mean']:.2f} / {A['ttft_ms']['p50']:.2f} / {A['ttft_ms']['p95']:.2f}  vs  ReComp {A['baseline_ttft_ms']['mean']:.2f} / {A['baseline_ttft_ms']['p50']:.2f} / {A['baseline_ttft_ms']['p95']:.2f}"],
        ["TTFT 비율 (mean) [95% CI]", f"{A['ttft_ratio_mean']:.3f}× [{A['ttft_ratio_ci95'][0]:.3f}, {A['ttft_ratio_ci95'][1]:.3f}] (단축 {A['ttft_reduction_pct']:.1f}%); p50 {A['ttft_ratio_p50']:.3f}×, p95 {A['ttft_ratio_p95']:.3f}×"],
        ["SSD MB / followup", f"{A['ssd_mb_per_followup']:.2f}"]]))
    L.append("## (B) KV 크기 축 지표 (보고, 판정 아님)\n")
    refs = B["references"]
    L.append(md_table(["지표", "정의", "LLaVA-1.5-7B (이 실행) [95% CI]", "Qwen2.5-VL (같은 packed 경로)", "LLaVA-NeXT-7B (v1, 파일 분리 경로)"], [
        ["R100", "TTFT(U100) / TTFT(ReComp)", f"**{B['R100']['value']:.3f}** [{B['R100']['ci95'][0]:.3f}, {B['R100']['ci95'][1]:.3f}]", refs["Qwen2.5-VL"]["R100"], refs["LLaVA-NeXT-7B"]["R100"]],
        ["S25", "TTFT(U25) / TTFT(U100)", f"**{B['S25']['value']:.3f}** [{B['S25']['ci95'][0]:.3f}, {B['S25']['ci95'][1]:.3f}]", refs["Qwen2.5-VL"]["S25"], refs["LLaVA-NeXT-7B"]["S25"]],
        ["S50", "TTFT(U50) / TTFT(U100)", f"**{B['S50']['value']:.3f}** [{B['S50']['ci95'][0]:.3f}, {B['S50']['ci95'][1]:.3f}]", "–", "–"],
        ["이미지당 KV (100%)", "", "302 MB", "약 22 MB", "약 1,174 MB"]]))
    hc = B["hypothesis_check"]
    L.append(f"사전 기록 가설: R100은 0.462와 1.39 사이 → **{'맞음' if hc['R100_between_0.462_and_1.39'] else '틀림'}** (관측 {B['R100']['value']:.3f}). "
             f"S25는 0.34와 0.892 사이 → **{'맞음' if hc['S25_between_0.34_and_0.892'] else '틀림'}** (관측 {B['S25']['value']:.3f}). {B['reference_note']}\n")
    L.append("## (C) layer 배분 판정 — PYR-b20_packed − U25_packed\n")
    L.append(md_table(["판정", "Δ pp [95% CI]", "CI의 0", "SSD MB/followup (PYR / U25, ≤1.02×)", "TTFT 평균 ms (PYR / U25, ≤1.05×)", "PYR만 / U25만 / 둘 다 / 둘 다 오답", "예측 일치율"],
                      [[f"**{C['verdict']}**", fmt(C["delta_followup_pp"], *C["ci95_pp"]), "제외" if C["ci_excludes_zero"] else "포함 (차이 미입증)",
                        f"{C['ssd_mb_vector']:.3f} / {C['ssd_mb_uniform']:.3f} ({'OK' if C['ssd_ok_1.02'] else '위반'})",
                        f"{C['ttft_mean_vector']:.2f} / {C['ttft_mean_uniform']:.2f} = {C['ttft_ratio']:.3f}× ({'OK' if C['ttft_ok_1.05'] else '위반'})",
                        " / ".join(str(x) for x in C["a_only_b_only_both_neither"]), f"{100 * C['prediction_agreement']:.2f}%"]]))
    L.append("규칙(실행 전 고정, LLaVA-NeXT·Qwen과 동일): GO = Δ ≥ +1.0 pp & CI 0 제외 & SSD MB ≤ U25 × 1.02 & TTFT 평균 ≤ U25 × 1.05; "
             "PARTIAL = Δ ≥ +0.5 pp이고 CI 0 포함, 또는 GO에서 MB/TTFT 중 정확히 하나 위반; 그 외 NO-GO. "
             f"같은 규칙의 선행 결과: LLaVA-NeXT-7B {C['same_rule_other_models']['LLaVA-NeXT-7B']}, Qwen2.5-VL {C['same_rule_other_models']['Qwen2.5-VL']}.\n")
    L.append("## 정확도 (turn별) — (F) T1 정상성 포함\n")
    L.append(md_table(["Arm", "T1 %", "T2 %", "T3 %", "All % (정답/N)", "Followup % (정답/N)"],
                      [[x["name"], f"{x['T1_pct']:.3f}", f"{x['T2_pct']:.3f}", f"{x['T3_pct']:.3f}", f"{x['all_pct']:.3f} ({x['all_correct']}/{x['all_n']})",
                        f"{x['followup_pct']:.3f} ({x['followup_correct']}/{x['followup_n']})"] for x in h["quality"]]))
    L.append(f"T1은 모든 arm이 full pixel이다. arm 간 T1 생성 토큰 일치: {t1['token_identical']:,} / {t1['requests']:,} 요청 → {t1['status']}. "
             "T1 정확도는 공개 LLaVA-1.5-7B GQA 수치(61.9)와 양자화(NF4)·전처리(중앙 자르기)·멀티턴 프롬프트 구성·398장 재구성 표본이 달라 직접 비교가 아니며 판정에 쓰지 않는다.\n")
    L.append(md_table(["비교 (A − B)", "Followup Δpp [95% CI]", "All Δpp [95% CI]", "T2 Δpp", "T3 Δpp", "A만 / B만 정답 (followup)", "예측 일치율 (followup)"],
                      [[f"{name(a)} − {name(b)}", fmt(pair_of(h, a, b)["delta_pp"], pair_of(h, a, b)["ci95_lo_pp"], pair_of(h, a, b)["ci95_hi_pp"]),
                        fmt(pair_of(h, a, b, "all")["delta_pp"], pair_of(h, a, b, "all")["ci95_lo_pp"], pair_of(h, a, b, "all")["ci95_hi_pp"]),
                        f"{pair_of(h, a, b, 'T2')['delta_pp']:+.3f}", f"{pair_of(h, a, b, 'T3')['delta_pp']:+.3f}",
                        f"{pair_of(h, a, b)['a_only']} / {pair_of(h, a, b)['b_only']}", f"{100 * pair_of(h, a, b)['prediction_agreement']:.2f}%"]
                       for a, b in ((U25, RECOMP), (U50, RECOMP), (U100, RECOMP), (PYR, RECOMP), (PYR, U25), (U50, U25), (U100, U50), (U100, U25))]))
    L.append("## Followup 성능과 I/O\n")
    L.append(md_table(["Arm", "TTFT mean / p50 / p95 ms", "E2E mean / p95 ms", "TTFT / ReComp", "TTFT / U100", "SSD MB", "읽기 호출 (pread / preadv)", "peak GPU alloc MB (평균)", "T1 TTFT mean ms"],
                      [[l["name"], f"{l['ttft_mean_ms']:.2f} / {l['ttft_p50_ms']:.2f} / {l['ttft_p95_ms']:.2f}", f"{l['e2e_mean_ms']:.2f} / {l['e2e_p95_ms']:.2f}",
                        f"{l['ttft_mean_ms'] / lat[RECOMP]['ttft_mean_ms']:.3f}×", f"{l['ttft_mean_ms'] / lat[U100]['ttft_mean_ms']:.3f}×", f"{l['ssd_mb_per_followup']:.2f}",
                        f"{l['pread_calls_mean']:.0f} / {l['preadv_calls_mean']:.0f}", f"{l['peak_gpu_alloc_mb_mean']:.1f}", f"{l['t1_ttft_mean_ms']:.2f}"] for l in h["latency"]]))
    L.append("## (E) 구간별 시간 (followup 평균 ms)\n")
    g = lambda v: "–" if v is None else f"{v:.2f}"
    L.append(md_table(["Arm", "입력 준비 (prompt·tokenize / 이미지 전처리)", "read (preadv 1회)", "H2D·scatter·mask", "cache 할당 등", "prefill", "decode"],
                      [[l["name"], g(l["processor_total_ms_mean"]) + " (processor 전체)" if l["method"] == RECOMP else g(l["prompt_prepare_ms_mean"]),
                        g(l["packed_read_ms_mean"]), g(l["scatter_ms_mean"]), g(l.get("cache_alloc_and_other_prepare_ms_mean")),
                        g(l["prefill_ms_mean"]) + (" (vision + 전체 prefill)" if l["method"] == RECOMP else " (suffix)"), g(l["decode_ms_mean"])] for l in h["latency"]]))
    L.append("구간은 TTFT의 구성 요소지만 계측 경계가 달라 합이 TTFT와 정확히 같지는 않다. H2D와 scatter는 HOLD 경로에서 한 호출(`PrefixCache.write`)이라 분리하지 않았다. "
             "아래는 gate fixture 10장(turn 3)에서 계측용 복제 경로로 잰 분해이며 **HOLD 측정이 아닌 진단값**이다(복제 결과는 production cache와 bitwise 동일).\n")
    L.append(md_table(["Vector", "host FP16→BF16 변환", "H2D 복사", "GPU index write", "mask", "(참고) 같은 fixture의 production scatter", "H2D MB"],
                      [[v, f"{sb[v]['host_fp16_to_bf16_ms']:.2f}", f"{sb[v]['h2d_ms']:.2f}", f"{sb[v]['gpu_index_write_ms']:.2f}", f"{sb[v]['mask_ms']:.2f}",
                        f"{sb[v]['production_scatter_ms']:.2f}", f"{S.EXPECTED_BYTES[v] / 1e6:.2f}"] for v in S.VECTORS]))
    L.append("## 저장과 setup (이미지당 1회)\n")
    L.append(md_table(["Arm", "보존 %", "파일 수", "store MB / 이미지", "persistence mean / p50 / p95 ms", "writer ms (materialize·repack / write / fsync·rename)", "완료 hash ms", "activation mean / p50 / p95 ms"],
                      [[s_["name"], f"{s_['stored_content_pct']:.1f}", s_["files_per_store"], f"{s_['store_mb_mean']:.2f}",
                        f"{s_['persistence_ms_mean']:.1f} / {s_['persistence_ms_p50']:.1f} / {s_['persistence_ms_p95']:.1f}",
                        f"{s_['native_writer_ms_mean']:.1f} ({s_['writer_materialize_repack_ms_mean']:.1f} / {s_['writer_ssd_write_ms_mean']:.1f} / {s_['writer_fsync_rename_ms_mean']:.1f})",
                        f"{s_['completion_hash_ms_mean']:.1f}", f"{s_['activation_ms_mean']:.1f} / {s_['activation_ms_p50']:.1f} / {s_['activation_ms_p95']:.1f}"] for s_ in h["storage"]]))
    L.append("persistence = Turn-1 capture에서 store-ready(fsync + atomic rename)까지 + 완료 봉인(전체 SHA256 1회). activation = 메타 검증 + payload SHA256 전체 검증 + pinned buffer 할당. "
             "네 store는 각 arm 자신의 T1 capture에서 만들었고 순서표는 이미지마다 네 store가 동일하다(감사).\n")
    L.append("## 3-turn session\n")
    L.append(md_table(["Arm", "setup ms / 이미지", "shared-stream mean / p50 / p95 ms", "standalone-equivalent (DERIVED) mean / p50 / p95 ms"],
                      [[s_["name"], f"{s_['setup_ms_per_image_mean']:.1f}", f"{s_['shared_mean_ms']:.1f} / {s_['shared_p50_ms']:.1f} / {s_['shared_p95_ms']:.1f}",
                        f"{s_['standalone_mean_ms']:.1f} / {s_['standalone_p50_ms']:.1f} / {s_['standalone_p95_ms']:.1f}"] for s_ in h["sessions"]]))
    L.append(md_table(["A − B", "accounting", "Δ ms [95% CI]"],
                      [[f"{name(p['a'])} − {name(p['b'])}", p["scope"].replace("session_", ""), fmt(p["delta"], p["ci95_lo"], p["ci95_hi"], 1)] for p in h["pairs"] if p.get("metric") == "session_ms"]))
    L.append("shared-stream은 이미지의 첫 dialogue에만 setup(persistence + activation)을 더한 실측 합이다. standalone-equivalent는 모든 dialogue에 setup을 더한 DERIVED 값이다.\n")
    L.append("## Gate\n")
    L.append(md_table(["Gate", "결과", "내용"], [
        ["LG1 runner 동치", gates["gates"]["LG1"], "새 runner recompute 경로 == HF generate(greedy, max 16), fixture 10장 × 3턴 토큰 전부 동일"],
        ["LG2 geometry·순서", gates["gates"]["LG2"], "시각 span 576개 연속 1구간, separator 0, CLS saliency (1, 576) 유한, 순서 = 0..575 순열, 질문을 바꿔도·다시 실행해도 순열 SHA 동일, LLM 호출 0"],
        ["LG3 capture 정확성", gates["gates"]["LG3"], "T1 capture의 prefix KV == 질문 자리를 filler로 바꾼 같은 길이의 독립 forward(3종), 32 layer K/V bitwise. "
                                                  f"문자 그대로의 짧은 prefix-only forward(진단): layer 0만 bitwise, layer ≥ 1은 최대 rel-L2 {gpu['LG3_literal_prefix_only_DIAGNOSTIC']['max_rel_l2']:.4f}"],
        ["LG4 저장 정확성", gates["gates"]["LG4"], "packed 파일의 layer slice == capture → FP16 → 순서표 앞 k_l행 (bitwise), 파일 크기 = 기대 바이트(separator 0 B), 변조 meta·SHA·행 수·잘린 파일 거부"],
        ["LG5 hit 경로 구현", gates["gates"]["LG5"], "SSD 경로 / 같은 shape의 독립 in-memory 참조 / 반복 실행: 첫 logits bitwise 동일, 생성 ID 동일, layer별 가시 행 == 순서표 앞 k_l개 (mask bit 비교)"],
        ["LG6 의미 검증", gates["gates"]["LG6"], f"전체 길이 참조(576행 실제 값 + layer별 가시 집합 mask) vs production: FP32 첫 토큰 logits atol = rtol = 1e-4. "
                                              f"U100 vs recompute 첫 토큰 일치 {gpu['LG6_u100_vs_recompute']['first_token_equal']} / {gpu['LG6_u100_vs_recompute']['requests']} "
                                              f"({100 * gpu['LG6_u100_vs_recompute']['rate']:.1f}%, 기준 95%)"],
        ["LG7 I/O", gates["gates"]["LG7"], "hit당 preadv 1회·pread 0회, 반환 바이트 = 파일 크기, vision 0·score 0, 질문만 바꿔도 read range·선택 ID 불변"],
        ["LG8 CPU", gates["gates"]["LG8"], f"벡터 SHA·합, N = 576에서 Σ k_l = 32·ceil(r·576), layer_budget 기본값에 기대는 호출 0, fail-closed 검사 {cpu['LG8'].get('fail_closed_checks')}건(negative fixture 포함)"],
        ["LG9 회귀", gates["gates"]["LG9"], f"기존 CPU 테스트 전체 통과({cpu['LG9']['ran']}), GLM 기존 실패 {len(cpu['LG9']['glm_errors'])}건 분리 보고, 기존 소스 무수정"],
        ["LG10 통합", gates["gates"]["LG10"], f"smoke dialogue 5 arm × 3턴 generated history {gates['LG10']['requests']} 요청 + 독립 재감사"],
        ["Artifact protection before/after", protection["status"], f"SHA {protection['files_before']} → {protection['files_after']} files, kvstore inventory {protection['kvstore_files_before']}, 다른 run inventory {protection['runs_files_before']}"],
        ["Deletion log", dele["status"], "run-local scratch / gate store만 삭제, 기존 runs·results·stores 삭제 0"]]))
    um = gpu["LG6_u100_vs_recompute"]
    if um["mismatches"]:
        L.append("U100 vs recompute 첫 토큰 불일치(원인 기록): " + "; ".join(f"{x['image_id']} T{x['turn']} (U100 top-2 logit 차 {x['u100_top2_logit_margin']:.4f})" for x in um["mismatches"])
                 + ". U100은 T1의 prefix KV(FP16 왕복)에 suffix만 prefill하고 recompute는 전체를 prefill하므로 BF16 반올림이 shape에 따라 달라져 동률에 가까운 첫 토큰이 뒤집힐 수 있다.\n")
    od = gpu["LG6_original_coordinate_DIAGNOSTIC"]
    L.append(f"진단(비 gate): 같은 가시 집합을 원래 token 순서로 배치한 참조와 production의 첫 토큰 logits 최대 차 {od['max_abs_delta']:.4f}, "
             f"첫 토큰 일치 {od['first_token_equal']} / {od['requests']}, 생성 ID 일치 {od['tokens_equal']} / {od['requests']} (key 합산 순서에 따른 BF16 반올림).\n")
    L.append("## 실행 기록\n")
    L.append(f"- 실제 요청 수: main {audit['requests']:,} / {audit['expected_requests']:,} (5 arm × 12,183), smoke {smoke_audit['requests']} / {smoke_audit['expected_requests']}, "
             f"integration(LG10) {gates['LG10']['requests']} / {S.EXPECT['integration']}. 실패·스킵 요청 0. 실패 attempt 기록 {len(failures_file)}건, 중단 기록 {len(interruptions)}건.\n"
             f"- main 소요: {elapsed_h:.2f} h (첫 시작 → 완료, wall). freeze 시 예상 {config['expected_duration_after_smoke']['estimate_hours'][0]:g}–{config['expected_duration_after_smoke']['estimate_hours'][1]:g} h.\n"
             f"- 삭제: main scratch store {dele['phases'][PHASE]['images']}개 이미지 · 파일 {dele['phases'][PHASE]['removed_files']:,}개 · {dele['phases'][PHASE]['removed_bytes'] / 1e9:.1f} GB(정확한 SHA allowlist 후), "
             f"gate store {dele['gpu_gate_images_deleted']}개 이미지. 기존 runs/results/stores 삭제 0.\n"
             f"- freeze amendment: {', '.join(amendments) if amendments else '없음'}.\n"
             f"- 환경: {model['versions']['python'].split()[0]}, transformers {model['versions']['transformers']}, bitsandbytes {model['versions']['bitsandbytes']}, torch {model['versions']['torch']}, "
             f"모델 revision `{config['revision']}`.\n")
    L.append("## 프롬프트와 다르게 한 점\n")
    L.append("\n".join(f"{i + 1}. {d}" for i, d in enumerate(STATIC_DEVIATIONS)) + "\n")
    if amendments:
        L.append("freeze amendment 기록: " + ", ".join(f"[{a}]({a})" for a in amendments) + "\n")
    L.append("## 한계와 NOT_MEASURED\n")
    L.append("\n".join(f"- {x}" for x in summary["not_measured"]) + "\n"
             "- DONTNEED는 OS page-cache hint이며 NAND cold나 O_DIRECT가 아니다.\n"
             "- CI가 0을 포함한다는 것은 동등성 증명이 아니다(이미지 398개 cluster bootstrap). 곡선과 (B)·(C)는 HOLD에서 관찰한 값이며 다른 데이터에서 확인되지 않았다.\n"
             "- packed serving은 dense GPU 버퍼를 유지하므로 GPU 메모리 절감은 주장하지 않는다.\n")
    (res / "REPORT.md").write_text("\n".join(L))

    # ------------------------------------------------------------ RESULT_RECORD
    R = []
    R.append(f"실행일 {time.strftime('%Y-%m-%d')}. Run `{run.relative_to(ROOT)}`, 결과 `{res.relative_to(ROOT)}` (REPORT.md, summary.json, decision.json). 상태 **{status}**.\n")
    R.append(f"- 요청: main {audit['requests']:,} / {audit['expected_requests']:,} (5 arm × 12,183), 실패·스킵 0, 독립 감사 {audit['status']}, main {elapsed_h:.2f} h. "
             f"Gate LG1–LG10 모두 PASS. T1 arm 간 토큰 일치 {t1['token_identical']:,} / {t1['requests']:,}.\n"
             f"- 모델: llava-hf/llava-1.5-7b-hf @ `{config['revision'][:12]}`, 576 토큰 고정·separator 없음, KV 301.99 MB/이미지, HF 기본 전처리(중앙 자르기).\n")
    R.append(md_table(["Arm", "Followup %", "vs ReComp Δpp [95% CI]", "TTFT mean ms", "TTFT / ReComp", "SSD MB"],
                      [[c["arm"], f"{c['followup_pct']:.3f}", "–" if c["delta_vs_recompute_pp"] is None else fmt(c["delta_vs_recompute_pp"], *c["delta_vs_recompute_ci95_pp"]),
                        f"{c['ttft_mean_ms']:.2f}", f"{c['ttft_over_recompute']:.3f}×", f"{c['ssd_mb_per_followup']:.2f}"] for c in curve]))
    R.append(f"- (A) 주 운영점 U25 vs ReComp: Δ {fmt(A['delta_pp'], *A['ci95_pp'])} pp ({A['wording']}), 원래 성능 대비 {A['relative_to_original_pct']:.2f}%, "
             f"TTFT {A['ttft_ms']['mean']:.2f} vs {A['baseline_ttft_ms']['mean']:.2f} ms = {A['ttft_ratio_mean']:.3f}×, {A['ssd_mb_per_followup']:.2f} MB/followup.\n"
             f"- (B) R100 = {B['R100']['value']:.3f}, S25 = {B['S25']['value']:.3f}, S50 = {B['S50']['value']:.3f}. 가설(R100 0.462–1.39, S25 0.34–0.892): "
             f"R100 {'맞음' if hc['R100_between_0.462_and_1.39'] else '틀림'}, S25 {'맞음' if hc['S25_between_0.34_and_0.892'] else '틀림'}. "
             "참고: Qwen R100 0.462·S25 0.892(같은 packed 경로), LLaVA-NeXT R100 1.39·S25 0.34(v1, 파일 분리 경로, 직접 비교 아님).\n"
             f"- (C) PYR-b20 − U25: **{C['verdict']}**, Δ {fmt(C['delta_followup_pp'], *C['ci95_pp'])} pp, SSD {C['ssd_mb_vector']:.2f} / {C['ssd_mb_uniform']:.2f} MB, TTFT {C['ttft_ratio']:.3f}×.\n"
             f"- (F) T1 정확도 {q[RECOMP]['T1_pct']:.3f}% (모든 arm full pixel).\n"
             "- 프롬프트와 다르게 한 점: LG3 참조를 같은 길이의 질문 제거 forward로 정의(문자 그대로의 짧은 forward는 BF16 shape 의존 반올림으로 bitwise 불가, 진단 기록), "
             "LG6 참조는 저장 순서 배치, H2D 단독은 fixture 진단. 자세한 내용은 REPORT.md.\n"
             "- NOT_MEASURED: HOLD 경로 H2D 단독, NAND cold / O_DIRECT, GPU footprint, LLaVA-NeXT 100% packed, FullLoad arm, MT-VQA·Qwen·GLM·LLaVA-NeXT 재실행.\n")
    (res / "RESULT_RECORD.md").write_text("\n".join(R))
    (res / "REPRODUCE.md").write_text(f"""# Reproduce (LLaVA-1.5-7B MT-GQA HOLD)

```bash
cd {ROOT}
export {S.ENV.replace(' ', ' ')}
PY={S.PY}
RUN=runs/llava15_mt_gqa_<new timestamp>
$PY scripts/177_freeze_llava15_mt_gqa.py --stage vectors --run $RUN      # 4 vectors (PYR-b20 copied), N = 576 rows / bytes, model + processor record
# write / check docs/llava15_mt_gqa_contract.md, then:
$PY scripts/177_freeze_llava15_mt_gqa.py --stage freeze --run $RUN --est-hours 3 6   # storage plan, protection before, manifest, config
$PY scripts/178_validate_llava15_mt_gqa.py --run $RUN --stage cpu        # LG8, LG9
$PY scripts/178_validate_llava15_mt_gqa.py --run $RUN --stage gpu        # LG1-LG7 on the 10 fixture images
$PY scripts/179_eval_llava15_mt_gqa_hold.py --run $RUN --stage integration   # LG10
$PY scripts/178_validate_llava15_mt_gqa.py --run $RUN --stage summary
$PY scripts/179_eval_llava15_mt_gqa_hold.py --run $RUN --stage smoke
$PY scripts/179_eval_llava15_mt_gqa_hold.py --run $RUN --stage main      # --resume after an interruption
$PY scripts/180_report_llava15_mt_gqa.py --run $RUN
```
The pinned model snapshot must be in the local HF cache (`llava-hf/llava-1.5-7b-hf` @ `{config['revision']}`). A reproduction uses a new
run root; existing runs / results / stores are never modified. HOLD payloads (about 604 MB per image) are deleted after each image's audit
and commit (run-local scratch only). After a recorded `freeze_amendment_<n>.json`, re-freeze with `177 --stage refreeze` and re-run every gate.
""")
    print(json.dumps({"report": str(res / "REPORT.md"), "status": status, "A": {k: A[k] for k in ("delta_pp", "ci95_pp", "ttft_ratio_mean", "relative_to_original_pct")},
                      "B": {k: B[k]["value"] for k in ("R100", "S25", "S50")}, "hypothesis": B["hypothesis_check"], "C": {k: C[k] for k in ("verdict", "delta_followup_pp", "ci95_pp", "ttft_ratio")},
                      "audit": audit["status"], "t1_identity": t1["status"], "protection": protection["status"], "deletion": dele["status"], "main_hours": elapsed_h}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

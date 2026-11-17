#!/usr/bin/env python3
"""Report for the MT-GQA asynchronous-read run (contract docs/mt_gqa_async_contract.md).

Reads committed rows and per-image evidence only; measures nothing.  Writes
``results/<RUN>/report.md`` plus the CSV/JSON tables it is built from.
Values that were not measured are written as "미실행".

  python scripts/189_report_mt_gqa_async.py --run runs/mt_gqa_async_<UTC>
"""
from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mmimpress.recompute_kv25 import paired_bootstrap

SEED = 1234
RESAMPLES = 10000
NOT_RUN = "미실행"


def load_harness():
    spec = importlib.util.spec_from_file_location("_mtgqa_async_harness_report", ROOT / "scripts/187_eval_mt_gqa_async.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=False, default=float) + "\n")


def csv_write(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def md_table(head, rows, align=None):
    align = align or ["---"] + ["---:"] * (len(head) - 1)
    return ("| " + " | ".join(head) + " |\n|" + "|".join(align) + "|\n"
            + "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in rows) + "\n")


def stats(values):
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(np.median(a)), float(np.percentile(a, 95))


def slim(r):
    """Keep only what the report needs from one raw row."""
    res = r["result"]
    keep = {k: r[k] for k in ("method_id", "family", "async_read", "image_id", "dialog_id", "turn_id", "score", "prediction", "question",
                              "ttft_ms", "request_e2e_ms", "ssd_read_bytes", "ssd_preads", "probe_read_bytes", "k_read_bytes",
                              "v_read_bytes", "packed_read_bytes", "read_blocked_ms", "global_dialog_ordinal", "N_content")}
    keep["tokens"] = res["generated_token_ids"]
    keep["prefill_ms"] = res.get("prefill_ms")
    keep["hook_ms"] = res.get("hook_ms")
    keep["io_ms"] = (res.get("io") or {}).get("ms")
    ahead = (res.get("read_wait") or {}).get("readahead")
    if ahead:
        keep["ahead"] = {k: ahead[k] for k in ("wait_ms", "ready_on_arrival", "items_taken", "peak_outstanding", "worker_read_ms")}
    if r["family"] == "impress" and r["turn_id"] > 1:
        keep["layer_log"] = [(x["mode"], x["sim"], x["k_chunks"], x["v_chunks"]) for x in res["layer_log"]]
        keep["unreordered"] = res["unreordered_chunks"]
        for k in ("fallback_rate", "mean_jaccard", "touched_k_chunk_fraction", "touched_v_chunk_fraction", "touched_chunk_fraction",
                  "probe_hook_blocked_ms", "prefetch_wait_ms", "prefetch_ready_on_arrival", "prefetched_probe_reads", "probe_read_ms"):
            keep[k] = res.get(k)
        keep["n_chunks"] = (r["geometry"]["v_token_num"] + 63) // 64
    if r["family"] == "ours" and r["turn_id"] > 1:
        keep["packed_read_ms"] = res.get("packed_read_ms")
        keep["scatter_ms"] = res.get("scatter_ms")
    return keep


def load_phase(run, phase):
    """({(method, dialog, turn): slim row}, {image: attempt folder})."""
    rows, attempts = {}, {}
    for cp in sorted((run / phase / "images").glob("*/COMMITTED.json")):
        doc = json.loads(cp.read_text()); folder = cp.parent / doc["attempt_id"]
        attempts[cp.parent.name] = folder
        with (folder / "raw.jsonl").open() as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[(r["method_id"], r["dialog_id"], r["turn_id"])] = slim(r)
    return rows, attempts


def by_method(rows, method, follow=True):
    return [r for (m, _, t), r in rows.items() if m == method and (t > 1 if follow else True)]


def paired(rows, a, b, field, follow=True):
    per = defaultdict(lambda: [0.0, 0.0, 0])
    for (m, d, t), ra in rows.items():
        if m != a or (follow and t == 1):
            continue
        rb = rows[(b, d, t)]
        e = per[ra["image_id"]]; e[0] += ra[field]; e[1] += rb[field]; e[2] += 1
    d = paired_bootstrap([tuple(e) for e in per.values()], RESAMPLES, SEED)
    d["images"] = len(per); d["n"] = int(sum(e[2] for e in per.values()))
    return d


def fmt_ci(d, scale=1.0, unit="", digits=2):
    return f"{scale * d['mean']:+.{digits}f}{unit} [{scale * d['ci_low']:+.{digits}f}, {scale * d['ci_high']:+.{digits}f}]"


def gate_status(run, name):
    p = run / "gates" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------ pieces
def main_tables(H, rows, methods, res):
    disp = {m: H.ARMS[m]["display"] for m in methods}
    lat = {}
    for m in methods:
        f = by_method(rows, m)
        t = stats([r["ttft_ms"] for r in f]); e = stats([r["request_e2e_ms"] for r in f])
        t1 = by_method(rows, m, follow=False)
        t1 = [r for r in t1 if r["turn_id"] == 1]
        lat[m] = {"method": m, "name": disp[m], "followups": len(f), "followup_acc_pct": 100 * float(np.mean([r["score"] for r in f])),
                  "t1_acc_pct": 100 * float(np.mean([r["score"] for r in t1])),
                  "t2_acc_pct": 100 * float(np.mean([r["score"] for r in f if r["turn_id"] == 2])),
                  "t3_acc_pct": 100 * float(np.mean([r["score"] for r in f if r["turn_id"] == 3])),
                  "ttft_mean_ms": t[0], "ttft_p50_ms": t[1], "ttft_p95_ms": t[2], "e2e_mean_ms": e[0], "e2e_p50_ms": e[1], "e2e_p95_ms": e[2],
                  "t1_ttft_mean_ms": float(np.mean([r["ttft_ms"] for r in t1])),
                  "ssd_mb_per_followup": float(np.mean([r["ssd_read_bytes"] for r in f])) / 1e6,
                  "read_calls_mean": float(np.mean([r["ssd_preads"] for r in f])),
                  "probe_mb": float(np.mean([r["probe_read_bytes"] for r in f])) / 1e6, "k_mb": float(np.mean([r["k_read_bytes"] for r in f])) / 1e6,
                  "v_mb": float(np.mean([r["v_read_bytes"] for r in f])) / 1e6,
                  "read_blocked_ms_mean": (float(np.mean([r["read_blocked_ms"] for r in f])) if f[0]["read_blocked_ms"] is not None else None)}
    full = next((m for m in methods if H.ARMS[m]["family"] == "fullload" and (H.ARMS[m]["async"] or len(methods) > 4)), None)
    full_main = next((m for m in methods if H.ARMS[m]["family"] == "fullload" and H.ARMS[m]["async"]), full)
    for m in methods:
        lat[m]["ttft_vs_fullload"] = lat[m]["ttft_mean_ms"] / lat[full_main]["ttft_mean_ms"] if full_main else None
        lat[m]["ttft_vs_recomp"] = lat[m]["ttft_mean_ms"] / lat["recompute"]["ttft_mean_ms"]
    csv_write(res / "main" / "arms.csv", list(lat.values()))
    return lat


def impress_stats(rows, method):
    f = by_method(rows, method)
    if not f:
        return None
    L = len(f[0]["layer_log"])
    fb = np.asarray([[x[0] == "fallback" for x in r["layer_log"]] for r in f], dtype=float)
    sim = np.asarray([[x[1] for x in r["layer_log"]] for r in f], dtype=float)
    kch = np.asarray([[x[2] for x in r["layer_log"]] for r in f], dtype=float)
    vch = np.asarray([[x[3] for x in r["layer_log"]] for r in f], dtype=float)
    nc = np.asarray([r["n_chunks"] for r in f], dtype=float)[:, None]
    uk = np.asarray([r["unreordered"]["k_chunks"] for r in f], dtype=float)
    uv = np.asarray([r["unreordered"]["v_chunks"] for r in f], dtype=float)
    probe_only = fb == 0
    # counterfactual on the consensus-path layers only, where reordering can matter for K
    out = {"method": method, "followups": len(f), "layers": L,
           "fallback_rate": float(fb.mean()), "mean_jaccard": float(sim.mean()),
           "k_chunk_fraction": float((kch / nc).mean()), "v_chunk_fraction": float((vch / nc).mean()),
           "kv_chunk_fraction": float(((kch + vch) / (2 * nc)).mean()),
           "raster_k_chunk_fraction_computed": float((uk / nc).mean()), "raster_v_chunk_fraction_computed": float((uv / nc).mean()),
           "raster_kv_chunk_fraction_computed": float(((uk + uv) / (2 * nc)).mean()),
           "probe_path_k_chunk_fraction": float(((kch / nc)[probe_only]).mean()) if probe_only.any() else None,
           "probe_path_raster_k_chunk_fraction_computed": float(((uk / nc)[probe_only]).mean()) if probe_only.any() else None,
           "fallback_v_chunk_fraction": float(((vch / nc)[~probe_only]).mean()) if (~probe_only).any() else None,
           "fallback_raster_v_chunk_fraction_computed": float(((uv / nc)[~probe_only]).mean()) if (~probe_only).any() else None,
           "probe_hook_blocked_ms": float(np.mean([r["probe_hook_blocked_ms"] for r in f])),
           "prefetch_wait_ms": float(np.mean([r["prefetch_wait_ms"] for r in f])),
           "prefetched_probe_reads": float(np.mean([r["prefetched_probe_reads"] for r in f])),
           "prefetch_ready_on_arrival": float(np.mean([r["prefetch_ready_on_arrival"] for r in f]))}
    layers = [{"layer": li, "fallback_rate": float(fb[:, li].mean()), "mean_jaccard": float(sim[:, li].mean()),
               "k_chunk_fraction": float((kch[:, li:li + 1] / nc).mean()), "v_chunk_fraction": float((vch[:, li:li + 1] / nc).mean()),
               "raster_k_chunk_fraction_computed": float((uk[:, li:li + 1] / nc).mean()),
               "raster_v_chunk_fraction_computed": float((uv[:, li:li + 1] / nc).mean())} for li in range(L)]
    return out, layers


def storage_and_sessions(H, rows, attempts, methods, res, tag):
    kinds = H.store_kinds(methods)
    storage = []; setup = {}; detail = {}
    for kind in kinds:
        rs = []
        for img, folder in attempts.items():
            per = json.loads((folder / f"{kind}_persistence.json").read_text()); act = json.loads((folder / f"{kind}_activation.json").read_text())
            files = per["files"]
            def size(pred):
                return sum(v["size"] for k, v in files.items() if pred(k))
            row = {"bytes": per["file_logical_bytes"], "persist_ms": per["service_persistence_ms"], "writer_ms": per["native_writer_wall_ms"],
                   "hash_ms": per.get("completion_full_hash_ms", 0.0), "act_ms": act["outer_wall_ms"],
                   "kv_bytes": size(lambda k: k.endswith("/k.bin") or k.endswith("/v.bin") or k == "kv_packed.bin"),
                   "probe_bytes": size(lambda k: k.endswith("/probe_k.bin")), "sep_bytes": size(lambda k: k == "sep_kv.bin"),
                   "hidden_bytes": size(lambda k: k == "v_hidden.pt"),
                   "other_bytes": size(lambda k: not (k.endswith(".bin") or k == "v_hidden.pt"))}
            wb = per.get("write_by_file_name") or {}
            row["probe_write_ms"] = wb.get("probe_k.bin", {}).get("ms")
            row["kv_write_ms"] = (wb.get("k.bin", {}).get("ms", 0.0) + wb.get("v.bin", {}).get("ms", 0.0)) if wb else None
            row["raster_writer_persist_ms"] = (per.get("timing_ms") or {}).get("persist_ms")
            row["fsync_ms"] = (per.get("timing_ms") or {}).get("durability_ms")
            t1 = per.get("impress_turn1")
            if t1:
                ro = t1["reorder"]["per_kind"]
                row.update(importance_ms=t1["importance_ms"], context_open_ms=t1["context_open_ms"], reorder_ms=t1["reorder_ms"],
                           reorder_k_ms=sum(ro["k"][x] for x in ("read_ms", "permute_ms", "write_ms")),
                           reorder_v_ms=sum(ro["v"][x] for x in ("read_ms", "permute_ms", "write_ms")),
                           reorder_probe_ms=sum(ro["probe_k"][x] for x in ("read_ms", "permute_ms", "write_ms")),
                           reorder_write_only_ms=sum(ro[n]["write_ms"] for n in ("k", "v", "probe_k")),
                           reorder_sidecar_ms=t1["reorder"]["separator_sidecar_ms"], impress_turn1_total_ms=t1["total_ms"],
                           suffix_tokens=t1["suffix_tokens"])
            rs.append(row); setup[(kind, img)] = per["service_persistence_ms"] + act["outer_wall_ms"]
        def mean(key):
            vals = [x[key] for x in rs if x.get(key) is not None]
            return float(np.mean(vals)) if vals else None
        entry = {"store_kind": kind, "images": len(rs)}
        for key in ("bytes", "kv_bytes", "probe_bytes", "sep_bytes", "hidden_bytes", "other_bytes"):
            entry[key.replace("bytes", "mb")] = mean(key) / 1e6
        entry["store_mb_max"] = max(x["bytes"] for x in rs) / 1e6
        for key in ("persist_ms", "writer_ms", "hash_ms", "act_ms", "raster_writer_persist_ms", "fsync_ms", "kv_write_ms", "probe_write_ms",
                    "importance_ms", "context_open_ms", "reorder_ms", "reorder_k_ms", "reorder_v_ms", "reorder_probe_ms",
                    "reorder_write_only_ms", "reorder_sidecar_ms", "impress_turn1_total_ms", "suffix_tokens"):
            entry[key + "_mean"] = mean(key)
        storage.append(entry); detail[kind] = rs
    csv_write(res / tag / "storage.csv", storage)
    # sessions, as scripts/165: shared = setup charged to the image's first dialogue, standalone = to every dialogue
    img_of = {}; first = {}
    for (m, d, t), r in sorted(rows.items(), key=lambda kv: kv[1]["global_dialog_ordinal"]):
        img_of[d] = r["image_id"]; first.setdefault(r["image_id"], d)
    dialogs = sorted(img_of)
    sessions = []; vals = {}
    for m in methods:
        kind = H.kind_of(m); shared = []; alone = []
        for d in dialogs:
            e2e = sum(rows[(m, d, t)]["request_e2e_ms"] for t in (1, 2, 3))
            su = setup.get((kind, img_of[d]), 0.0)
            shared.append(e2e + (su if first[img_of[d]] == d else 0.0)); alone.append(e2e + su)
        vals[m] = (shared, alone)
        s1, s2 = stats(shared), stats(alone)
        sessions.append({"method": m, "name": H.ARMS[m]["display"], "shared_mean_ms": s1[0], "shared_p50_ms": s1[1], "shared_p95_ms": s1[2],
                         "standalone_mean_ms": s2[0], "standalone_p50_ms": s2[1], "standalone_p95_ms": s2[2],
                         "request_only_mean_ms": float(np.mean([sum(rows[(m, d, t)]["request_e2e_ms"] for t in (1, 2, 3)) for d in dialogs])),
                         "provenance": "MEASURED, no KV setup" if kind == "none" else "MEASURED (shared) / DERIVED (standalone)"})
    csv_write(res / tag / "sessions.csv", sessions)
    disk = [json.loads((f / "disk_usage.json").read_text()) for f in attempts.values()]
    usage = {"peak_scratch_bytes_max": max(d["peak_scratch_bytes"] for d in disk), "peak_scratch_bytes_mean": float(np.mean([d["peak_scratch_bytes"] for d in disk])),
             "min_free_bytes_after_requests": min(d["free_bytes_after_requests"] for d in disk), "images": len(disk)}
    return storage, sessions, usage


def protection_after(H, run, res):
    before = json.loads((run / "protection_before.json").read_text())
    after = H.stat_inventory(run, res / "protection_after.tsv.gz")
    changed = None
    if after["sha256"] != before["sha256"]:
        def load(p):
            with gzip.open(p, "rt") as f:
                return {line.split("\t")[0]: line.rstrip("\n") for line in f}
        a, b = load(run / "protection_before.tsv.gz"), load(res / "protection_after.tsv.gz")
        changed = {"modified": sorted(k for k in a if k in b and a[k] != b[k])[:200], "missing": sorted(k for k in a if k not in b)[:200],
                   "added_count": sum(1 for k in b if k not in a), "added_sample": sorted(k for k in b if k not in a)[:50]}
    status = "PASS" if changed is None or (not changed["modified"] and not changed["missing"]) else "FAIL"
    out = {"status": status, "before": before, "after": after, "differences": changed,
           "rule": "no pre-existing file under results/, runs/ (other runs), kvstore* may change or disappear; new files elsewhere are listed"}
    write_json(res / "protection_result.json", out)
    return out


# ------------------------------------------------------------------- report
def build(run):
    H = load_harness()
    res = ROOT / "results" / run.name
    res.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / "config.json").read_text()); manifest = json.loads((run / "manifest.json").read_text())
    env = config["environment"]
    L = []
    L.append(f"# MT-GQA 다중 턴 — ReComp / FullLoad / IMPRESS / Ours (저장 KV 비동기 읽기)\n")
    L.append("측정값만 적는다. 측정하지 않은 값은 \"미실행\"이다. 계약: `docs/mt_gqa_async_contract.md` (사본 `CONTRACT.md`).\n")
    L.append("## 설정\n")
    L.append(f"- 실행: `runs/{run.name}` (요청 단위 원본), 이 디렉터리(`results/{run.name}`)는 집계와 보고서.\n"
             f"- 환경: Python {env['python'].split()[0]}, torch {env['torch']}, **transformers {env['transformers']}** (`{env['executable']}`). "
             "기존 HOLD 실행과 같은 버전이다. 이전 IMPRESS 단일 턴 작업은 transformers 5.13.1 환경이었다.\n"
             f"- 모델 `{config['model']}` revision `{config['revision'][:12]}…`, NF4/BF16/eager, FP16 SSD KV, greedy max 16, batch 1, CPU threads 2.\n"
             "- 모집단·절차: MT-GQA-reconstructed 398 images / 4,061 dialogues / 3턴, method-local generated history, strict normalized EM, 대화 단위 arm 순서 회전. "
             "턴 1은 모든 방법이 픽셀부터 계산, 캐시 예열 없음, 저장 KV 읽기는 요청 직전에 page cache를 비운 cold 조건.\n"
             f"- IMPRESS: probe head {H.IMPRESS['probe_heads']}개, α = {H.IMPRESS['alpha']}, layer마다 25% 유지, chunk 64 토큰, 중요도 = suffix 행 전체의 attention 열 합(head 평균), "
             "fallback은 K 전량 + 선택된 V chunk, 재배치는 첫 대화 턴 1 질문 1개의 중요도 내림차순(layer별), 비동기는 다음 layer probe key 선행 읽기.\n"
             "- FullLoad: 전체 Visual KV(raster 순서). 비동기는 워커가 layer 순서로 K·V 전체를 이어 읽음.\n"
             "- Ours: PYR-beta-raw packed store(`layer_budget_kv25_pyrbeta_20261110T073604Z`의 벡터 그대로). 비동기는 워커가 packed 파일을 separator 구간 → layer 순서로 이어 읽음.\n"
             "- 비동기 공통: 백그라운드 워커 1개, 디스크 읽기만 워커가 하고 GPU 적재는 요청 스레드, 선행 읽기는 TTFT 타이머 시작 뒤에만, 앞서 읽는 깊이 상한 없음.\n")

    # ---- gates
    L.append("## 검증 단계\n")
    names = [("1. CPU 테스트", "step1_cpu_tests"), ("2. 비동기 정합성 (gate 49 대화)", "step2_async_equals_sync_gate"),
             ("3. 재배치 정합성 (gate)", "step3_reorder_consistency"), ("4. 환경 일치 (gate)", "step4_matches_hold_gate"),
             ("5a. DEV 40장 실행·감사", "dev_execution"), ("5b. 비동기 정합성 (DEV 전 요청)", "step2_async_equals_sync_dev"),
             ("5c. 환경 일치 (DEV 전 요청)", "step4_matches_hold_dev"), ("5d. 동기 대비 비동기 효과 (DEV)", "step5_dev_async_effect")]
    gate_rows = []
    gates = {}
    for label, name in names:
        g = gate_status(run, name); gates[name] = g
        detail = ""
        if g is None:
            status = NOT_RUN
        else:
            status = g["status"]
            if name == "step1_cpu_tests":
                detail = "; ".join(f"transformers {s['transformers']}: {s['ran']}개 실행, 실패 {len(s['failed'])}개(모두 변경 전부터의 환경 실패), 새 실패 {len(s['new_failures'])}개" for s in g["suites"])
                detail += f"; 신규 테스트 {g['new_unit_tests']['ran']}개"
            elif "pairs" in g:
                detail = "; ".join(f"{H.ARMS[p['a']]['display']}: {p['requests']}건 중 불일치 {p['mismatch_count']}건" for p in g["pairs"])
            elif "methods" in g and name.startswith("step4"):
                detail = "; ".join(f"{m['method']}: {m['requests']}건 중 불일치 {m['mismatch_count']}건" for m in g["methods"])
            elif name == "step3_reorder_consistency":
                detail = f"{g['requests']}건 중 불일치 {g['mismatch_count']}건"
            elif name == "dev_execution":
                detail = f"{g['requests']}/{g['expected']}건, {g.get('images')}장"
            elif name == "step5_dev_async_effect":
                detail = "; ".join(f"{m['method']}: {m['paired_diff_ms']:+.1f} ms [{m['ci95_low_ms']:+.1f}, {m['ci95_high_ms']:+.1f}]" for m in g["methods"])
        gate_rows.append([label, status, detail])
    admission = gate_status(run, "admission")
    gate_rows.append(["6. 본실행 진입", admission["status"] if admission else NOT_RUN, "1–5 모두 PASS일 때만 실행" if admission else ""])
    L.append(md_table(["단계", "결과", "내용"], gate_rows, ["---", "---", "---"]))

    summary = {"run": run.name, "environment": env, "gates": {k: (v["status"] if v else None) for k, v in gates.items()},
               "admission": admission["status"] if admission else None}

    # ---- main
    main_done = (run / "main" / "validation.json").exists()
    if main_done:
        rows, attempts = load_phase(run, "main")
        methods = config["phases"]["main"]["methods"]
        lat = main_tables(H, rows, methods, res)
        L.append("## 본실행 (398장, T2–T3 followup)\n")
        L.append(f"요청 수: {len(rows):,} (arm당 {len(rows) // len(methods):,}, followup {lat[methods[0]]['followups']:,}). 저장 KV 방법은 모두 비동기 읽기다. "
                 "FullLoad·ReComp 대비는 같은 실행 안의 TTFT 평균 비율이다. SSD MB는 read syscall이 실제로 돌려준 바이트의 요청 평균이다.\n")
        L.append(md_table(["방법", "정답률", "TTFT 평균(ms)", "p50", "p95", "FullLoad 대비", "ReComp 대비", "SSD MB/followup"],
                          [[lat[m]["name"], f"{lat[m]['followup_acc_pct']:.2f}%", f"{lat[m]['ttft_mean_ms']:.1f}", f"{lat[m]['ttft_p50_ms']:.1f}", f"{lat[m]['ttft_p95_ms']:.1f}",
                            f"{lat[m]['ttft_vs_fullload']:.3f}×", f"{lat[m]['ttft_vs_recomp']:.3f}×", f"{lat[m]['ssd_mb_per_followup']:.1f}"] for m in methods]))
        imp = next(m for m in methods if H.ARMS[m]["family"] == "impress")
        L.append(f"IMPRESS의 종류별 읽기량(요청 평균): probe {lat[imp]['probe_mb']:.1f} MB / K {lat[imp]['k_mb']:.1f} MB / V {lat[imp]['v_mb']:.1f} MB "
                 f"(합 {lat[imp]['ssd_mb_per_followup']:.1f} MB).\n")
        L.append("참고(턴별 정답률, 턴 1 TTFT):\n")
        L.append(md_table(["방법", "T1 정답률", "T2", "T3", "T1 TTFT 평균(ms)", "followup E2E 평균(ms)", "read 호출 수"],
                          [[lat[m]["name"], f"{lat[m]['t1_acc_pct']:.2f}%", f"{lat[m]['t2_acc_pct']:.2f}%", f"{lat[m]['t3_acc_pct']:.2f}%",
                            f"{lat[m]['t1_ttft_mean_ms']:.1f}", f"{lat[m]['e2e_mean_ms']:.1f}", f"{lat[m]['read_calls_mean']:.1f}"] for m in methods]))
        ours = next(m for m in methods if H.ARMS[m]["family"] == "ours"); full = next(m for m in methods if H.ARMS[m]["family"] == "fullload")
        pairs = []
        for a, b in ((ours, imp), (ours, full), (ours, "recompute"), (imp, full)):
            acc = paired(rows, a, b, "score"); tt = paired(rows, a, b, "ttft_ms")
            pairs.append({"a": H.ARMS[a]["display"], "b": H.ARMS[b]["display"], "acc_delta_pp": 100 * acc["mean"], "acc_ci_low_pp": 100 * acc["ci_low"],
                          "acc_ci_high_pp": 100 * acc["ci_high"], "ttft_delta_ms": tt["mean"], "ttft_ci_low_ms": tt["ci_low"], "ttft_ci_high_ms": tt["ci_high"],
                          "images": acc["images"], "n": acc["n"]})
        csv_write(res / "main" / "paired.csv", pairs)
        L.append("### 이미지 단위 paired bootstrap 95% CI (10,000회, seed 1234, followup)\n")
        L.append(md_table(["비교", "정답률 차이 (pp)", "TTFT 차이 (ms)"],
                          [[f"{p['a']} − {p['b']}", f"{p['acc_delta_pp']:+.2f} [{p['acc_ci_low_pp']:+.2f}, {p['acc_ci_high_pp']:+.2f}]",
                            f"{p['ttft_delta_ms']:+.1f} [{p['ttft_ci_low_ms']:+.1f}, {p['ttft_ci_high_ms']:+.1f}]"] for p in pairs]))
        summary["main"] = {"arms": lat, "paired": pairs, "requests": len(rows)}
    else:
        L.append("## 본실행 (398장)\n\n미실행.\n")
        rows = attempts = None

    # ---- dev
    dev_done = (run / "dev" / "validation.json").exists()
    if dev_done:
        drows, dattempts = load_phase(run, "dev")
        dmethods = config["phases"]["dev"]["methods"]
        L.append("## DEV 40장 파일럿 (7 arm, T2–T3 followup)\n")
        n_follow = len(by_method(drows, "recompute"))
        L.append(f"DEV 40장의 MT-GQA 대화 {len({d for (_, d, _) in drows})}개, arm당 followup {n_follow}건. 동기 arm과 비동기 arm은 같은 store를 읽는다.\n")
        table = []; devsum = []
        rc = by_method(drows, "recompute"); t = stats([r["ttft_ms"] for r in rc])
        table.append(["ReComp", "—", f"{100 * np.mean([r['score'] for r in rc]):.2f}%", f"{t[0]:.1f}", f"{t[1]:.1f}", f"{t[2]:.1f}", "—", "—", "—"])
        for label, s, a in (("FullLoad", "fullload", "fullload_async"), ("IMPRESS", "impress", "impress_async"), ("Ours", H.OURS, H.OURS_ASYNC)):
            d = paired(drows, a, s, "ttft_ms")
            for mode, m in (("동기", s), ("비동기", a)):
                f = by_method(drows, m); t = stats([r["ttft_ms"] for r in f])
                if H.ARMS[m]["family"] == "ours" and not H.ARMS[m]["async"]:
                    wait = float(np.mean([r["packed_read_ms"] for r in f]))
                else:
                    wait = float(np.mean([r["read_blocked_ms"] for r in f]))
                table.append([label, mode, f"{100 * np.mean([r['score'] for r in f]):.2f}%", f"{t[0]:.1f}", f"{t[1]:.1f}", f"{t[2]:.1f}",
                              f"{d['mean']:+.1f} [{d['ci_low']:+.1f}, {d['ci_high']:+.1f}]" if mode == "비동기" else "—",
                              f"{wait:.1f}", f"{np.mean([r['ssd_read_bytes'] for r in f]) / 1e6:.1f}"])
                entry = {"method": label, "mode": mode, "arm": m, "acc_pct": 100 * float(np.mean([r["score"] for r in f])), "ttft_mean_ms": t[0], "ttft_p50_ms": t[1],
                         "ttft_p95_ms": t[2], "read_wait_ms": wait, "ssd_mb": float(np.mean([r["ssd_read_bytes"] for r in f])) / 1e6,
                         "prefill_ms": float(np.mean([r["prefill_ms"] for r in f])), "hook_ms": float(np.mean([r["hook_ms"] for r in f if r["hook_ms"] is not None] or [0.0])),
                         "pread_ms": float(np.mean([r["io_ms"] for r in f]))}
                if mode == "비동기":
                    entry.update(paired_diff_ms=d["mean"], ci_low_ms=d["ci_low"], ci_high_ms=d["ci_high"])
                    ah = [r["ahead"] for r in f if r.get("ahead")]
                    if ah:
                        entry.update(ahead_items=float(np.mean([x["items_taken"] for x in ah])), ahead_ready=float(np.mean([x["ready_on_arrival"] for x in ah])),
                                     ahead_peak_depth_mean=float(np.mean([x["peak_outstanding"] for x in ah])), ahead_peak_depth_max=int(max(x["peak_outstanding"] for x in ah)),
                                     worker_read_ms=float(np.mean([x["worker_read_ms"] for x in ah])))
                devsum.append(entry)
        csv_write(res / "dev" / "sync_async.csv", devsum)
        L.append(md_table(["방법", "읽기", "정답률", "TTFT 평균(ms)", "p50", "p95", "paired 차이 비동기−동기 (ms) [95% CI]", "GPU가 읽기를 기다린 시간(ms)", "SSD MB"], table,
                          ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
        L.append("- paired 차이는 같은 (대화, 턴)의 비동기 − 동기 TTFT, 이미지 단위 bootstrap 10,000회(seed 1234). 음수면 비동기가 빠르다.\n"
                 "- \"GPU가 읽기를 기다린 시간\"은 요청 스레드가 reader 호출 안에 있었거나(동기) 워커의 읽기 도착을 기다린(비동기) 벽시계 시간의 요청 평균이다. "
                 "Ours 동기는 packed 파일 preadv 1회의 시간이다.\n"
                 "- 비동기에서는 읽기와 연산이 겹친다. **hook 시간과 디스크(pread) 시간의 합을 TTFT로 해석하지 않는다.** 두 값은 서로 겹치는 구간을 재며, "
                 "워커의 읽기 시간은 hook 밖에서도 흐른다. TTFT는 TTFT 열만 쓴다.\n")
        ex = [e for e in devsum]
        L.append("참고(요청 평균, ms; 서로 더하지 않는다):\n")
        L.append(md_table(["방법", "읽기", "prefill(첫 토큰까지의 model 구간)", "hook 시간", "pread 시간 합", "워커 읽기 시간 합", "선행 항목 수", "도착 시 완료", "최대 선행 깊이(평균 / 최댓값)"],
                          [[e["method"], e["mode"], f"{e['prefill_ms']:.1f}", f"{e['hook_ms']:.1f}", f"{e['pread_ms']:.1f}",
                            f"{e['worker_read_ms']:.1f}" if "worker_read_ms" in e else "—", f"{e['ahead_items']:.1f}" if "ahead_items" in e else "—",
                            f"{e['ahead_ready']:.2f}" if "ahead_ready" in e else "—",
                            f"{e['ahead_peak_depth_mean']:.2f} / {e['ahead_peak_depth_max']}" if "ahead_items" in e else "—"] for e in ex],
                          ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
        summary["dev"] = {"sync_async": devsum}
    else:
        L.append("## DEV 40장 파일럿\n\n미실행.\n")
        drows = dattempts = None

    # ---- IMPRESS internals
    L.append("## IMPRESS 내부 통계\n")
    blocks = []
    if main_done:
        blocks.append(("본실행 398장", rows, next(m for m in config["phases"]["main"]["methods"] if H.ARMS[m]["family"] == "impress"), "main"))
    if dev_done:
        blocks.append(("DEV 40장 (동기 arm)", drows, "impress", "dev"))
    if not blocks:
        L.append("미실행.\n")
    for title, rr, m, tag in blocks:
        st, layers = impress_stats(rr, m)
        csv_write(res / tag / "impress_layers.csv", layers); write_json(res / tag / "impress_stats.json", st)
        L.append(f"### {title} — followup {st['followups']:,}건\n")
        L.append(md_table(["항목", "값"], [
            ["fallback 비율 (layer·요청 평균)", f"{100 * st['fallback_rate']:.1f}%"],
            ["평균 Jaccard (probe head 3개 쌍 평균)", f"{st['mean_jaccard']:.4f}"],
            ["K chunk 비율 (실측)", f"{100 * st['k_chunk_fraction']:.1f}%"],
            ["V chunk 비율 (실측)", f"{100 * st['v_chunk_fraction']:.1f}%"],
            ["K·V chunk 비율 (실측, 평균)", f"{100 * st['kv_chunk_fraction']:.1f}%"],
            ["재배치 없는 순서였다면 K chunk 비율 (**계산값**, 읽지 않음)", f"{100 * st['raster_k_chunk_fraction_computed']:.1f}%"],
            ["재배치 없는 순서였다면 V chunk 비율 (**계산값**, 읽지 않음)", f"{100 * st['raster_v_chunk_fraction_computed']:.1f}%"],
            ["재배치 없는 순서였다면 K·V chunk 비율 (**계산값**)", f"{100 * st['raster_kv_chunk_fraction_computed']:.1f}%"],
            ["합의(probe) 경로 layer만: K·V chunk 비율 실측 → 계산값", (f"{100 * st['probe_path_k_chunk_fraction']:.1f}% → {100 * st['probe_path_raster_k_chunk_fraction_computed']:.1f}%" if st['probe_path_k_chunk_fraction'] is not None else "—")],
            ["fallback layer만: V chunk 비율 실측 → 계산값", (f"{100 * st['fallback_v_chunk_fraction']:.1f}% → {100 * st['fallback_raster_v_chunk_fraction_computed']:.1f}%" if st['fallback_v_chunk_fraction'] is not None else "—")],
        ], ["---", "---:"]))
        L.append("chunk 비율은 layer당 읽은 chunk 수 / layer chunk 수의 평균이다(probe 사본 제외; fallback layer의 K는 어떤 순서에서도 100%). "
                 "계산값은 요청의 layer별 keep mask(합의 경로: 선택 토큰 + separator, fallback: head별 선택의 합집합 + separator)를 원래 위치로 되돌려 "
                 "raster 64-토큰 chunk로 센 것이며 실제로 읽은 값이 아니다.\n")
        L.append("layer별 (fallback 비율 / 평균 Jaccard / K chunk 실측 / V chunk 실측 / K·V 계산값):\n")
        L.append(md_table(["layer", "fallback", "Jaccard", "K 실측", "V 실측", "K 계산값(raster)", "V 계산값(raster)"],
                          [[x["layer"], f"{100 * x['fallback_rate']:.1f}%", f"{x['mean_jaccard']:.3f}", f"{100 * x['k_chunk_fraction']:.1f}%", f"{100 * x['v_chunk_fraction']:.1f}%",
                            f"{100 * x['raster_k_chunk_fraction_computed']:.1f}%", f"{100 * x['raster_v_chunk_fraction_computed']:.1f}%"] for x in layers]))
        summary.setdefault("impress", {})[tag] = st

    # ---- storage / turn-1 cost / sessions
    L.append("## 저장·턴 1 비용\n")
    src = ("main", rows, attempts, config["phases"]["main"]["methods"]) if main_done else (("dev", drows, dattempts, config["phases"]["dev"]["methods"]) if dev_done else None)
    if src is None:
        L.append("미실행.\n")
    else:
        tag, rr, att, mm = src
        storage, sessions, usage = storage_and_sessions(H, rr, att, mm, res, tag)
        name = {"raster": "FullLoad", "impress": "IMPRESS", H.PACKED_KIND: "Ours"}
        L.append(f"출처: {'본실행 398장' if tag == 'main' else 'DEV 40장'}. store는 이미지당 하나(첫 대화 턴 1에서 생성). 시간은 이미지 평균(ms).\n")
        L.append(md_table(["방법", "store 크기 평균(MB)", "최대(MB)", "K·V", "probe 사본", "separator sidecar", "v_hidden", "기타(meta·system KV 등)"],
                          [[name[s["store_kind"]], f"{s['mb_mean'] if 'mb_mean' in s else s['mb']:.1f}", f"{s['store_mb_max']:.1f}", f"{s['kv_mb']:.1f}", f"{s['probe_mb']:.1f}",
                            f"{s['sep_mb']:.1f}", f"{s['hidden_mb']:.1f}", f"{s['other_mb']:.1f}"] for s in storage]))
        L.append("- FullLoad store에는 기존 HOLD와 같은 writer가 쓰는 probe 사본·separator sidecar·`v_hidden.pt`가 들어 있지만 FullLoad는 이를 읽지 않는다. "
                 "IMPRESS가 읽는 것은 K·V와 probe 사본이다(separator sidecar와 `v_hidden.pt`는 읽지 않음). Ours의 separator 구간은 packed 파일 안에 있어 K·V 열에 포함된다.\n")
        def f(v):
            return "—" if v is None else f"{v:.0f}"
        L.append(md_table(["방법", "저장 전체(쓰기+무결성 hash)", "writer(쓰기+fsync)", "K·V 파일 쓰기", "probe 사본 쓰기", "무결성 hash", "중요도 집계", "재배치(읽기+치환+쓰기)", "그중 probe 재배치", "activation"],
                          [[name[s["store_kind"]], f(s["persist_ms_mean"]), f(s["raster_writer_persist_ms_mean"] if s["raster_writer_persist_ms_mean"] is not None else s["writer_ms_mean"]),
                            f(s["kv_write_ms_mean"]), f(s["probe_write_ms_mean"]), f(s["hash_ms_mean"]), f(s["importance_ms_mean"]), f(s["reorder_ms_mean"]),
                            f(s["reorder_probe_ms_mean"]), f(s["act_ms_mean"])] for s in storage]))
        imp = next((s for s in storage if s["store_kind"] == "impress"), None)
        if imp:
            L.append(f"- IMPRESS의 턴 1 쪽 추가 비용(이미지당 평균): 중요도 집계 {imp['importance_ms_mean']:.0f} ms(store 전량 읽기 + suffix {imp['suffix_tokens_mean']:.1f}토큰 forward 1회; "
                     f"context 열기 {imp['context_open_ms_mean']:.0f} ms 별도), 재배치 {imp['reorder_ms_mean']:.0f} ms"
                     f"(K {imp['reorder_k_ms_mean']:.0f} / V {imp['reorder_v_ms_mean']:.0f} / probe {imp['reorder_probe_ms_mean']:.0f} / separator sidecar 재작성 {imp['reorder_sidecar_ms_mean']:.0f}; "
                     f"그중 순수 쓰기 {imp['reorder_write_only_ms_mean']:.0f} ms), probe 사본 최초 쓰기 {imp['probe_write_ms_mean']:.0f} ms. "
                     "이 비용은 followup TTFT에 들어 있지 않다. 중요도 집계는 턴 1 추론과 별도의 forward로 잰 값이라, 턴 1 추론 중에 집계하는 구현보다 크게 잡혔을 수 있다(그 구현은 미실행).\n")
        L.append("- \"저장 전체\"는 writer + 이미지별 store 전체 SHA256 1회(수명주기 공통) + (IMPRESS는) 중요도 집계·재배치를 포함한 벽시계 시간이다. "
                 "page cache에 쓰는 시간과 fsync 시간이 섞여 있으므로 디스크 순수 쓰기 속도로 해석하지 않는다.\n")
        L.append("### 3턴 세션 시간 (기존 보고서와 같은 방식)\n")
        L.append("세션 = 한 대화의 3턴 요청 E2E 합 + store 준비(저장 전체 + activation). shared: 이미지의 첫 대화에만 준비 시간을 더함(실측 흐름). "
                 "standalone: 모든 대화에 더함(DERIVED).\n")
        L.append(md_table(["방법", "요청만 평균(ms)", "shared 평균 / p50 / p95 (ms)", "standalone 평균 / p50 / p95 (ms)"],
                          [[s["name"] + ("" if tag == "main" else f" ({'비동기' if H.ARMS[s['method']]['async'] else '동기'})" if H.ARMS[s['method']]['family'] != 'recompute' else ""),
                            f"{s['request_only_mean_ms']:.1f}", f"{s['shared_mean_ms']:.1f} / {s['shared_p50_ms']:.1f} / {s['shared_p95_ms']:.1f}",
                            f"{s['standalone_mean_ms']:.1f} / {s['standalone_p50_ms']:.1f} / {s['standalone_p95_ms']:.1f}"] for s in sessions]))
        L.append(f"- 디스크 최대 사용량: 이미지 하나의 scratch store 묶음 최대 {usage['peak_scratch_bytes_max'] / 1e9:.2f} GB (평균 {usage['peak_scratch_bytes_mean'] / 1e9:.2f} GB, {usage['images']}장). "
                 f"실행 중 관측된 최소 여유 공간 {usage['min_free_bytes_after_requests'] / 2**30:.1f} GiB. 이미지 commit 뒤 scratch는 지운다.\n")
        summary["storage"] = storage; summary["sessions"] = sessions; summary["disk_usage"] = usage

    # ---- protection, bookkeeping
    prot = protection_after(H, run, res)
    L.append("## 기존 산출물 보호\n")
    L.append(f"- `results/`, `runs/`(이 run 제외), `kvstore*`의 기존 파일 {prot['before']['files']:,}개 (경로, 크기, mtime) 대조: **{prot['status']}**"
             + ("" if prot["differences"] is None else f" (변경 {len(prot['differences']['modified'])}, 누락 {len(prot['differences']['missing'])}, 새 파일 {prot['differences']['added_count']})") + ".\n")
    summary["protection"] = {"status": prot["status"], "files": prot["before"]["files"]}
    extra = res / "notes.md"
    if extra.exists():
        L.append(extra.read_text())
    (res / "report.md").write_text("\n".join(L))
    write_json(res / "summary.json", summary)
    for name in ("CONTRACT.md", "config.json", "storage_plan.json", "environment_before.json"):
        if (run / name).exists():
            shutil.copy2(run / name, res / name)
    if (run / "gates").exists():
        (res / "gates").mkdir(exist_ok=True)
        for p in (run / "gates").glob("*.json"):
            shutil.copy2(p, res / "gates" / p.name)
    print(f"report: {res / 'report.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    build(args.run.resolve())


if __name__ == "__main__":
    main()

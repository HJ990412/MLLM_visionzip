#!/usr/bin/env python3
"""Report: LLaVA-NeXT x MT-GQA, ReComp / FullLoad / Ours(재배열) / Ours(원래 순서)
(contract docs/mt_gqa_ours_order_contract.md).

Reads committed rows and per-image evidence only; measures nothing.  Writes
``results/<RUN>/report.md``, ``verdict.json`` and the CSV/JSON tables the
report is built from.  Values that were not measured are written as "미실행".
``results/<RUN>/notes.md`` (hand-written) is appended to the report.

  python scripts/204_report_mt_gqa_ours_order.py --run runs/mt_gqa_ours_order_<UTC>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
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
ACC_CI_LOW_MIN_PP = -0.5      # criterion a: CI lower bound must be greater than this
TTFT_RATIO_MAX = 1.02         # criterion b


def _load(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load("189_report_mt_gqa_async.py", "_mtgqa_async_report")      # table / csv / json helpers, protection check
write_json, csv_write, md_table, stats, gate_status = R.write_json, R.csv_write, R.md_table, R.stats, R.gate_status


def slim(r):
    res = r["result"]
    keep = {k: r[k] for k in ("method_id", "image_id", "dialog_id", "turn_id", "score", "prediction", "prompt_sha256", "ttft_ms",
                              "request_e2e_ms", "ssd_read_bytes", "ssd_preads", "read_blocked_ms", "preceding_method_id",
                              "method_order_position", "global_dialog_ordinal")}
    keep["tokens"] = res["generated_token_ids"]
    for k in ("prefill_ms", "hook_ms", "scatter_ms", "packed_read_ms"):
        keep[k] = res.get(k)
    return keep


def load_phase(run, phase):
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


def load_prior_impress(H):
    """The stored per-request IMPRESS rows of the 2026-11-17 main run."""
    out = {}
    for cp in sorted((H.PRIOR_MAIN_RUN / "main" / "images").glob("*/COMMITTED.json")):
        doc = json.loads(cp.read_text())
        with (cp.parent / doc["attempt_id"] / "raw.jsonl").open() as f:
            for line in f:
                if line.strip() and f'"{H.PRIOR_IMPRESS}"' in line:
                    r = json.loads(line)
                    if r["method_id"] == H.PRIOR_IMPRESS:
                        out[(H.PRIOR_IMPRESS, r["dialog_id"], r["turn_id"])] = {
                            "image_id": r["image_id"], "score": r["score"], "ttft_ms": r["ttft_ms"], "turn_id": r["turn_id"]}
    return out


def follow(rows, m):
    return [r for (mm, _, t), r in rows.items() if mm == m and t > 1]


def paired(rows_a, a, rows_b, b, field):
    per = defaultdict(lambda: [0.0, 0.0, 0])
    for (m, d, t), ra in rows_a.items():
        if m != a or t == 1:
            continue
        rb = rows_b[(b, d, t)]
        e = per[ra["image_id"]]; e[0] += ra[field]; e[1] += rb[field]; e[2] += 1
    out = paired_bootstrap([tuple(e) for e in per.values()], RESAMPLES, SEED)
    out["images"] = len(per); out["n"] = int(sum(e[2] for e in per.values()))
    return out, per


def ratio_bootstrap(per):
    """Image-cluster bootstrap of mean(a) / mean(b) (same resampling as the paired difference)."""
    arr = np.asarray([tuple(e) for e in per.values()], dtype=np.float64)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(arr), size=(RESAMPLES, len(arr)))
    s = arr[idx]
    boot = s[..., 0].sum(1) / s[..., 1].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"ratio": float(arr[:, 0].sum() / arr[:, 1].sum()), "ci_low": float(lo), "ci_high": float(hi)}


def arm_table(rows, methods, names):
    out = {}
    for m in methods:
        f = follow(rows, m)
        t = stats([r["ttft_ms"] for r in f])
        t1 = [r for (mm, _, tt), r in rows.items() if mm == m and tt == 1]
        blocked = [r["read_blocked_ms"] for r in f if r["read_blocked_ms"] is not None]
        scatter = [r["scatter_ms"] for r in f if r["scatter_ms"] is not None]
        worker = [r["packed_read_ms"] for r in f if r["packed_read_ms"] is not None]
        hook = [r["hook_ms"] for r in f if r["hook_ms"] is not None]
        out[m] = {"method": m, "name": names[m], "followups": len(f), "followup_acc_pct": 100 * float(np.mean([r["score"] for r in f])),
                  "t1_acc_pct": 100 * float(np.mean([r["score"] for r in t1])),
                  "t2_acc_pct": 100 * float(np.mean([r["score"] for r in f if r["turn_id"] == 2])),
                  "t3_acc_pct": 100 * float(np.mean([r["score"] for r in f if r["turn_id"] == 3])),
                  "ttft_mean_ms": t[0], "ttft_p50_ms": t[1], "ttft_p95_ms": t[2],
                  "e2e_mean_ms": float(np.mean([r["request_e2e_ms"] for r in f])),
                  "t1_ttft_mean_ms": float(np.mean([r["ttft_ms"] for r in t1])),
                  "ssd_mb_per_followup": float(np.mean([r["ssd_read_bytes"] for r in f])) / 1e6,
                  "ssd_bytes_per_followup": float(np.mean([r["ssd_read_bytes"] for r in f])),
                  "read_calls_mean": float(np.mean([r["ssd_preads"] for r in f])),
                  "read_blocked_ms_mean": float(np.mean(blocked)) if blocked else None,
                  "scatter_ms_mean": float(np.mean(scatter)) if scatter else None,
                  "worker_read_ms_mean": float(np.mean(worker)) if worker else None,
                  "hook_ms_mean": float(np.mean(hook)) if hook else None,
                  "prefill_ms_mean": float(np.mean([r["prefill_ms"] for r in f if r["prefill_ms"] is not None] or [np.nan]))}
    return out


def pair_row(names, rows_a, a, rows_b, b, cross_run=False):
    acc, _ = paired(rows_a, a, rows_b, b, "score"); tt, _ = paired(rows_a, a, rows_b, b, "ttft_ms")
    return {"a": names[a], "b": names[b], "acc_delta_pp": 100 * acc["mean"], "acc_ci_low_pp": 100 * acc["ci_low"],
            "acc_ci_high_pp": 100 * acc["ci_high"], "ttft_delta_ms": tt["mean"], "ttft_ci_low_ms": tt["ci_low"], "ttft_ci_high_ms": tt["ci_high"],
            "images": acc["images"], "n": acc["n"], "ttft_from_different_runs": bool(cross_run)}


def pair_table(pairs):
    def verdict(lo, hi):
        return "차이 미입증" if lo <= 0 <= hi else "차이 있음"
    return md_table(["비교", "정답률 차이 (pp) [95% CI]", "", "TTFT 차이 (ms) [95% CI]", ""],
                    [[f"{p['a']} − {p['b']}", f"{p['acc_delta_pp']:+.2f} [{p['acc_ci_low_pp']:+.2f}, {p['acc_ci_high_pp']:+.2f}]",
                      verdict(p["acc_ci_low_pp"], p["acc_ci_high_pp"]),
                      f"{p['ttft_delta_ms']:+.1f} [{p['ttft_ci_low_ms']:+.1f}, {p['ttft_ci_high_ms']:+.1f}]" + (" †" if p["ttft_from_different_runs"] else ""),
                      verdict(p["ttft_ci_low_ms"], p["ttft_ci_high_ms"])] for p in pairs], ["---", "---:", "---", "---:", "---"])


def two_ours(rows, A, B):
    out = []
    for turn in (1, 2, 3):
        n = diff = diff_same = same_prompt = only_a = only_b = only_a_same = only_b_same = 0
        for (m, d, t), ra in rows.items():
            if m != A or t != turn:
                continue
            rb = rows[(B, d, t)]; n += 1
            same = ra["prompt_sha256"] == rb["prompt_sha256"]; same_prompt += same
            if ra["tokens"] != rb["tokens"]:
                diff += 1; diff_same += same
            if ra["score"] > rb["score"]:
                only_a += 1; only_a_same += same
            if rb["score"] > ra["score"]:
                only_b += 1; only_b_same += same
        out.append({"turn": turn, "requests": n, "same_prompt": same_prompt, "tokens_differ": diff, "tokens_differ_same_prompt": diff_same,
                    "only_reorder_correct": only_a, "only_original_correct": only_b,
                    "only_reorder_correct_same_prompt": only_a_same, "only_original_correct_same_prompt": only_b_same})
    return out


def turn1_cost(H, attempts):
    """Turn-1 store cost per image for the two packed kinds, from the persistence receipts and the stored metadata."""
    out = {}; per_image = []
    for kind in (H.PACKED_KIND, H.ORIG_KIND):
        rs = []
        for img, folder in attempts.items():
            per = json.loads((folder / f"{kind}_persistence.json").read_text()); meta = json.loads((folder / f"{kind}_meta.json").read_text())
            act = json.loads((folder / f"{kind}_activation.json").read_text())
            t = per["timing_ms"]; cap = meta["turn1_capture"]; files = per["files"]; w = meta["packed_writer_timing_ms"]
            row = {"image_id": img, "kind": kind,
                   "score_capture_ms": cap["saliency_total_instrumentation_ms"], "token_mapping_ms": t["token_mapping_ms"],
                   "order_ms": t["permutation_ms"], "row_plan_ms": w.get("row_plan_ms", 0.0),
                   "kv_to_host_ms": t["kv_materialize_ms"], "row_select_ms": t["kv_repack_ms"], "serialize_ms": t["prefix_slice_serialization_ms"],
                   "file_write_ms": t["ssd_write_ms"], "fsync_ms": t["file_fsync_ms"], "writer_total_ms": t["persist_ms"],
                   "persist_with_integrity_hash_ms": per["service_persistence_ms"], "activation_ms": act["outer_wall_ms"],
                   "meta_json_bytes": files["meta.json"]["size"], "layout_file_bytes": files.get("visionzip_layout.pt", {}).get("size", 0),
                   "payload_bytes": files["kv_packed.bin"]["size"], "store_bytes": per["file_logical_bytes"]}
            row["importance_ms"] = row["score_capture_ms"] + row["token_mapping_ms"] + row["order_ms"] + row["row_plan_ms"]
            row["metadata_bytes"] = row["meta_json_bytes"] + row["layout_file_bytes"]
            rs.append(row)
        per_image += rs
        out[kind] = {k: float(np.mean([x[k] for x in rs])) for k in rs[0] if k not in ("image_id", "kind")}
        out[kind]["images"] = len(rs)
    return out, per_image


def by_predecessor(rows, methods, names):
    table = []
    for m in methods:
        groups = defaultdict(list)
        for r in follow(rows, m):
            groups[r["preceding_method_id"]].append(r["ttft_ms"])
        for prev, vals in sorted(groups.items(), key=lambda kv: str(kv[0])):
            table.append({"method": names[m], "preceding": names.get(prev, str(prev)), "followups": len(vals), "ttft_mean_ms": float(np.mean(vals)),
                          "ttft_p50_ms": float(np.median(vals))})
    return table


def ssd_written(run):
    p = run / "ssd_stat.txt"
    if not p.exists():
        return None
    marks = {}
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) > 8:
            marks[parts[0]] = int(parts[8]) * 512      # label, timestamp, then the stat fields: field 7 = sectors written
    return marks


CODE_CHANGES = [
    ("`mmimpress/packed_store.py`", "`write_packed_store(..., row_order=\"importance\")`: `\"original\"`이면 layer l의 같은 토큰 집합을 원래 인덱스 오름차순으로 쓰고, "
     "separator 블록은 원래 위치에서 모으며, 메타데이터에서 중요도 순열 관련 필드를 빼고 인덱스 표 두 개를 넣는다. 기본값의 연산·바이트는 그대로다."),
    ("`mmimpress/packed_store.py`", "신규 `original_order_plan`(행 계획), `layer_original_indices`(메타데이터에서 layer별 목록 복원), `validate_packed_original_meta`(fail-closed 검증). "
     "`validate_packed_meta`는 store 형식에 따라 분기한다. `persist_captured_packed_prefix`에 `row_order` 인자."),
    ("`mmimpress/packed_serve.py`", "`PackedImageContext`: 원래 순서 store면 separator 위치를 원래 위치로 두고 layer별 행 인덱스를 activation 때 만든다. 전용 layout 검증 추가. "
     "`PackedPrefixSelector._scatter_layer`: 원래 순서면 행을 원래 토큰 위치에 쓰고 그 위치만 보이게 mask를 만든다. `PackedStreamSelector`는 상속으로 같은 경로를 쓴다. "
     "읽기(`PackedReader`), 읽기 순서, hook, 타이머 경계는 수정하지 않았다. 요청 결과에 `row_order` 필드가 하나 붙는다(기본 경로 포함)."),
    ("`mmimpress/piggyback.py`", "`persist_captured_visual_prefix(..., row_order=\"importance\")`: `\"original\"`이면 순열 hash를 메타데이터에 넣지 않고 `visionzip_layout.pt`를 쓰지 않는다. "
     "staging, fsync, 원자적 publish 단계는 두 방식이 같다."),
    ("`scripts/202_eval_mt_gqa_ours_order.py` (신규)", "실행 harness. `scripts/196`에서 IMPRESS arm을 빼고 Ours(원래 순서) arm, 두 store의 디스크 비교, 자리 교대가 있는 arm 순서를 더했다."),
    ("`scripts/203_validate_mt_gqa_ours_order.py` (신규)", "gate 1–7과 본실험 뒤의 사후 확인."),
    ("`scripts/204_report_mt_gqa_ours_order.py` (신규)", "이 보고서. 측정에 관여하지 않는다."),
    ("`scripts/run_ours_order.sh` (신규)", "freeze → gate → 본실험 → 보고의 실행 순서."),
    ("`tests/test_packed_original_order.py` (신규, 9개)", "행 계획이 `order[:k_l]`의 정렬과 같음 / 입력이 순서와 개수뿐임; 두 store의 토큰별 K·V 바이트·separator·byte table·크기 일치; "
     "메타데이터에 중요도 순열이 없음; 변조된 메타데이터 거부; 적재 위치·mask·읽기 호출이 기대와 같음; 비동기 경로가 동기 경로와 같고 읽기 계획이 재배열 store와 같음; "
     "길이가 다른 세 캡처(질문 3개)로 만든 store가 바이트 단위로 같음; 옵션의 적용 범위 밖 사용 거부."),
]


def fmt(v, digits=1, unit=""):
    return "—" if v is None else f"{v:.{digits}f}{unit}"


def build(run):
    H = _load("202_eval_mt_gqa_ours_order.py", "_mtgqa_ours_order_harness_report")
    res = ROOT / "results" / run.name
    res.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / "config.json").read_text())
    env = config["environment"]
    names = {m: H.ARMS[m]["display"] for m in H.ARMS}
    names[H.PRIOR_IMPRESS] = "IMPRESS"; names[None] = "(이미지의 첫 요청)"
    A, B = H.OURS_ASYNC, H.ORIG_ASYNC
    L = []
    L.append("# LLaVA-NeXT × MT-GQA: Ours에서 중요도 순 재배열을 빼도 되는가\n")
    L.append("측정값만 적는다. 측정하지 않은 값은 \"미실행\"이다. 계약: `docs/mt_gqa_ours_order_contract.md` (사본 `CONTRACT.md`). "
             "두 방식의 이름은 \"Ours(재배열)\"(현재 구현)과 \"Ours(원래 순서)\"(같은 토큰 집합을 원래 이미지 토큰 순서로 저장·적재)다.\n")
    summary = {"run": run.name, "environment": env}

    # ---- main
    main_done = (run / "main" / "validation.json").exists()
    rows = attempts = None
    verdict = {"status": NOT_RUN}
    posthoc = gate_status(run, "posthoc_continuity_main")
    if main_done:
        rows, attempts = load_phase(run, "main")
        methods = config["phases"]["main"]["methods"]
        lat = arm_table(rows, methods, names)
        csv_write(res / "main" / "arms.csv", list(lat.values()))
        acc, _ = paired(rows, B, rows, A, "score"); _, per_t = paired(rows, B, rows, A, "ttft_ms")
        ratio = lat[B]["ttft_mean_ms"] / lat[A]["ttft_mean_ms"]
        rb = ratio_bootstrap(per_t)
        byte_diff = call_diff = pairs_n = 0
        for (m, d, t), ra in rows.items():
            if m == A and t > 1:
                rbw = rows[(B, d, t)]; pairs_n += 1
                byte_diff += ra["ssd_read_bytes"] != rbw["ssd_read_bytes"]; call_diff += ra["ssd_preads"] != rbw["ssd_preads"]
        a_ok = 100 * acc["ci_low"] > ACC_CI_LOW_MIN_PP; b_ok = ratio <= TTFT_RATIO_MAX; c_ok = byte_diff == 0 and call_diff == 0 and pairs_n > 0
        verdict = {"status": "빼도 됨" if (a_ok and b_ok and c_ok) else "판단 보류", "followup_pairs": pairs_n,
                   "a": {"pass": bool(a_ok), "rule": f"95% CI 하한 > {ACC_CI_LOW_MIN_PP} pp", "acc_delta_pp": 100 * acc["mean"],
                         "ci_low_pp": 100 * acc["ci_low"], "ci_high_pp": 100 * acc["ci_high"], "images": acc["images"],
                         "acc_reorder_pct": lat[A]["followup_acc_pct"], "acc_original_pct": lat[B]["followup_acc_pct"]},
                   "b": {"pass": bool(b_ok), "rule": f"TTFT 평균 비율 <= {TTFT_RATIO_MAX}", "ratio": ratio,
                         "ttft_reorder_ms": lat[A]["ttft_mean_ms"], "ttft_original_ms": lat[B]["ttft_mean_ms"],
                         "ratio_bootstrap_ci_reference": [rb["ci_low"], rb["ci_high"]]},
                   "c": {"pass": bool(c_ok), "rule": "모든 followup 요청 쌍에서 읽기 바이트와 읽기 호출 수가 같음",
                         "pairs_with_different_bytes": byte_diff, "pairs_with_different_read_calls": call_diff,
                         "bytes_per_followup_reorder": lat[A]["ssd_bytes_per_followup"], "bytes_per_followup_original": lat[B]["ssd_bytes_per_followup"],
                         "read_calls_reorder": lat[A]["read_calls_mean"], "read_calls_original": lat[B]["read_calls_mean"]}}
        failed = [k for k in "abc" if not verdict[k]["pass"]]
        verdict["failed_criteria"] = failed

        L.append("## 판정\n")
        L.append(f"**{verdict['status']}**" + ("" if not failed else f" — 어긋난 기준: {', '.join(failed)}") + ". "
                 f"본실험 398장의 followup(T2–T3) {pairs_n:,}쌍 기준이며, 기준은 실행 전에 고정한 것이다. 최종 결정은 사용자가 한다.\n")
        L.append(md_table(["기준", "수치", "결과"], [
            ["a. 정답률 차이 (원래 순서 − 재배열)의 95% CI 하한 > −0.5 pp",
             f"{verdict['a']['acc_delta_pp']:+.3f} pp, CI [{verdict['a']['ci_low_pp']:+.3f}, {verdict['a']['ci_high_pp']:+.3f}] "
             f"(재배열 {lat[A]['followup_acc_pct']:.2f}%, 원래 순서 {lat[B]['followup_acc_pct']:.2f}%)", "충족" if a_ok else "**어긋남**"],
            ["b. TTFT 평균 비율 (원래 순서 / 재배열) ≤ 1.02",
             f"{ratio:.4f} ({lat[B]['ttft_mean_ms']:.2f} ms / {lat[A]['ttft_mean_ms']:.2f} ms)", "충족" if b_ok else "**어긋남**"],
            ["c. followup당 SSD 읽기 바이트와 읽기 호출 수가 같음",
             f"바이트가 다른 쌍 {byte_diff}, 호출 수가 다른 쌍 {call_diff} / {pairs_n:,}쌍 "
             f"(평균 {lat[A]['ssd_mb_per_followup']:.2f} MB · {lat[A]['read_calls_mean']:.1f}회 vs {lat[B]['ssd_mb_per_followup']:.2f} MB · {lat[B]['read_calls_mean']:.1f}회)",
             "충족" if c_ok else "**어긋남**"],
        ], ["---", "---", "---"]))
        L.append(f"참고(기준 아님): TTFT 비율의 이미지 단위 bootstrap 95% CI는 [{rb['ci_low']:.4f}, {rb['ci_high']:.4f}]다.\n")

        L.append("## 1. 본실험 표 (398장, T2–T3 followup)\n")
        L.append(f"요청 수: {len(rows):,} (arm당 {len(rows) // len(methods):,}, followup {lat[methods[0]]['followups']:,}). "
                 "SSD MB는 read syscall이 실제로 돌려준 바이트의 요청 평균(1 MB = 1e6 bytes), 읽기 호출 수는 read syscall 수의 요청 평균이다.\n")
        L.append(md_table(["방법", "followup EM", "TTFT 평균(ms)", "p50", "p95", "SSD MB/followup", "읽기 호출 수"],
                          [[lat[m]["name"], f"{lat[m]['followup_acc_pct']:.2f}%", f"{lat[m]['ttft_mean_ms']:.1f}", f"{lat[m]['ttft_p50_ms']:.1f}",
                            f"{lat[m]['ttft_p95_ms']:.1f}", f"{lat[m]['ssd_mb_per_followup']:.1f}", f"{lat[m]['read_calls_mean']:.1f}"] for m in methods]))
        L.append("참고(턴별 EM, 턴 1 TTFT):\n")
        L.append(md_table(["방법", "T1 EM", "T2 EM", "T3 EM", "T1 TTFT 평균(ms)", "followup E2E 평균(ms)"],
                          [[lat[m]["name"], f"{lat[m]['t1_acc_pct']:.2f}%", f"{lat[m]['t2_acc_pct']:.2f}%", f"{lat[m]['t3_acc_pct']:.2f}%",
                            f"{lat[m]['t1_ttft_mean_ms']:.1f}", f"{lat[m]['e2e_mean_ms']:.1f}"] for m in methods]))

        L.append("## 2. 짝지은 비교 (이미지 단위 paired bootstrap 95% CI, 10,000회, seed 1234, followup)\n")
        pairs = [pair_row(names, rows, B, rows, A)]
        for x in (B, A):
            pairs += [pair_row(names, rows, x, rows, "fullload_async"), pair_row(names, rows, x, rows, "recompute")]
        impress_note = ""
        cont_ok = bool(posthoc and posthoc.get("status") == "PASS" and posthoc.get("phase_complete"))
        if cont_ok:
            prior = load_prior_impress(H)
            keys = {(d, t) for (m, d, t) in rows if m == A}
            if all((H.PRIOR_IMPRESS, d, t) in prior for d, t in keys):
                pairs += [pair_row(names, rows, x, prior, H.PRIOR_IMPRESS, cross_run=True) for x in (B, A)]
                f = [r for r in prior.values() if r["turn_id"] > 1]
                impress_note = (f"IMPRESS는 2026-11-17 실행(`runs/{H.PRIOR_MAIN_RUN.name}`)의 저장된 요청별 결과다 "
                                f"(followup EM {100 * float(np.mean([r['score'] for r in f])):.2f}%, TTFT 평균 {float(np.mean([r['ttft_ms'] for r in f])):.1f} ms, {len(f):,}건). "
                                "gate 4의 일치가 398장 전체에서 성립함을 먼저 확인했다(아래 사후 확인). † 표시는 서로 다른 실행에서 잰 TTFT의 차이다.")
                summary["impress_reference"] = {"followups": len(f), "acc_pct": 100 * float(np.mean([r["score"] for r in f])),
                                                "ttft_mean_ms": float(np.mean([r["ttft_ms"] for r in f]))}
            else:
                impress_note = "IMPRESS 비교: 2026-11-17 실행에 대응하는 요청이 빠져 있어 내지 않는다."
        else:
            impress_note = ("IMPRESS 비교는 내지 않는다: gate 4(연속성)의 일치가 398장 전체에서 성립하지 않았거나 확인되지 않았다"
                            + ("" if not posthoc else " (" + "; ".join(f"{m['name']} 불일치 {m['mismatch_count']}건" for m in posthoc["methods"]) + ")") + ".")
        csv_write(res / "main" / "paired.csv", pairs)
        L.append(pair_table(pairs))
        L.append("CI가 0을 포함하면 \"차이 미입증\"이다(동등하다는 뜻이 아니다). " + impress_note + "\n")

        L.append("## 3. TTFT 비율 (followup 평균의 비)\n")
        ratios = [["Ours(원래 순서) / Ours(재배열)", f"{ratio:.4f}"]]
        for x in (B, A):
            ratios += [[f"{names[x]} / FullLoad", f"{lat[x]['ttft_mean_ms'] / lat['fullload_async']['ttft_mean_ms']:.4f}"],
                       [f"{names[x]} / ReComp", f"{lat[x]['ttft_mean_ms'] / lat['recompute']['ttft_mean_ms']:.4f}"]]
        L.append(md_table(["비율", "값"], ratios))

        L.append("## 4. 두 Ours 사이에서 달라진 요청\n")
        diff = two_ours(rows, A, B)
        csv_write(res / "main" / "two_ours_difference.csv", diff)
        L.append(md_table(["턴", "요청 수", "생성 토큰열이 다른 요청", "재배열만 정답", "원래 순서만 정답", "프롬프트가 같은 요청", "그중 토큰열이 다른 요청",
                           "그중 재배열만 정답 / 원래 순서만 정답"],
                          [[f"T{x['turn']}", f"{x['requests']:,}", x["tokens_differ"], x["only_reorder_correct"], x["only_original_correct"],
                            f"{x['same_prompt']:,}", x["tokens_differ_same_prompt"],
                            f"{x['only_reorder_correct_same_prompt']} / {x['only_original_correct_same_prompt']}"] for x in diff]))
        L.append("T1은 두 방식 모두 픽셀부터 계산한다. T3는 자기 arm의 T2 답이 이력에 들어가므로, T2에서 답이 갈린 대화는 T3의 프롬프트부터 다르다. "
                 "\"프롬프트가 같은 요청\"은 그런 경우를 뺀 것이다. 두 방식의 생성 토큰이 다른 것은 gate 실패가 아니다(key 순서에 따른 BF16 합산 순서 차이).\n")

        L.append("## 5. T1 저장 비용 (이미지당 평균, ms)\n")
        cost, per_image = turn1_cost(H, attempts)
        csv_write(res / "main" / "turn1_cost.csv", per_image)
        ca, cb = cost[H.PACKED_KIND], cost[H.ORIG_KIND]
        L.append(md_table(["항목", "Ours(재배열)", "Ours(원래 순서)", "어디서"], [
            ["중요도 계산 합계", f"{ca['importance_ms']:.3f}", f"{cb['importance_ms']:.3f}", ""],
            ["  VisionZip 점수 캡처", f"{ca['score_capture_ms']:.3f}", f"{cb['score_capture_ms']:.3f}", "턴 1 요청 안"],
            ["  토큰 위치 매핑", f"{ca['token_mapping_ms']:.3f}", f"{cb['token_mapping_ms']:.3f}", "요청 밖"],
            ["  순서 계산 (정렬)", f"{ca['order_ms']:.3f}", f"{cb['order_ms']:.3f}", "요청 밖"],
            ["  layer별 인덱스 표 계산", "—", f"{cb['row_plan_ms']:.3f}", "요청 밖"],
            ["행 골라내기 (index_select)", f"{ca['row_select_ms']:.1f}", f"{cb['row_select_ms']:.1f}", "요청 밖"],
            ["SSD 쓰기: 파일 쓰기", f"{ca['file_write_ms']:.1f}", f"{cb['file_write_ms']:.1f}", "요청 밖"],
            ["SSD 쓰기: fsync", f"{ca['fsync_ms']:.1f}", f"{cb['fsync_ms']:.1f}", "요청 밖"],
            ["(참고) KV를 host로 옮기기", f"{ca['kv_to_host_ms']:.1f}", f"{cb['kv_to_host_ms']:.1f}", "요청 밖"],
            ["(참고) writer 전체", f"{ca['writer_total_ms']:.1f}", f"{cb['writer_total_ms']:.1f}", "요청 밖"],
            ["(참고) 저장 전체 (writer + 무결성 hash)", f"{ca['persist_with_integrity_hash_ms']:.1f}", f"{cb['persist_with_integrity_hash_ms']:.1f}", "요청 밖"],
            ["(참고) activation", f"{ca['activation_ms']:.1f}", f"{cb['activation_ms']:.1f}", "요청 밖"],
        ], ["---", "---:", "---:", "---"]))
        L.append(md_table(["메타데이터 크기 (이미지당 평균, bytes)", "Ours(재배열)", "Ours(원래 순서)"], [
            ["`meta.json`", f"{ca['meta_json_bytes']:,.0f}", f"{cb['meta_json_bytes']:,.0f}"],
            ["`visionzip_layout.pt` (점수·순열)", f"{ca['layout_file_bytes']:,.0f}", "없음"],
            ["합", f"{ca['metadata_bytes']:,.0f}", f"{cb['metadata_bytes']:,.0f}"],
            ["(참고) payload `kv_packed.bin`", f"{ca['payload_bytes']:,.0f}", f"{cb['payload_bytes']:,.0f}"],
        ]))
        L.append(f"이미지 {ca['images']:.0f}장 평균. 두 방식은 같은 대화의 서로 다른 턴 1 요청에서 store를 만든다. "
                 "파일 쓰기는 page cache에 쓰는 시간, fsync는 디스크에 내리는 시간이며 디스크 순수 쓰기 속도로 해석하지 않는다. "
                 "`meta.json`에는 두 방식 모두 prefix 토큰 ID 목록 등 공통 필드가 들어 있다.\n")

        L.append("## 6. 재사용 턴 시간 분해 (followup 평균, ms)\n")
        L.append(md_table(["방법", "TTFT", "읽기 대기", "GPU 적재", "워커의 읽기 시간", "hook 시간", "prefill"],
                          [[lat[m]["name"], f"{lat[m]['ttft_mean_ms']:.1f}", fmt(lat[m]["read_blocked_ms_mean"], 2), fmt(lat[m]["scatter_ms_mean"], 2),
                            fmt(lat[m]["worker_read_ms_mean"], 1), fmt(lat[m]["hook_ms_mean"], 1), fmt(lat[m]["prefill_ms_mean"], 1)]
                           for m in (A, B)]))
        L.append("읽기 대기 = 요청 스레드가 워커의 읽기를 기다린 벽시계 시간. GPU 적재 = layer별 적재 호출(행과 separator를 GPU cache에 쓰고 mask를 만드는 구간)의 벽시계 합이며, "
                 "동기 H2D 복사가 앞선 GPU 연산의 완료를 기다리는 시간이 섞여 있다(두 방식에 같은 정의). hook 시간은 읽기 대기와 GPU 적재를 포함한다. "
                 "워커의 읽기는 연산과 겹치므로 이 값들의 합을 TTFT로 해석하지 않는다.\n")
        pred = by_predecessor(rows, (A, B), names)
        csv_write(res / "main" / "ttft_by_preceding_arm.csv", pred)
        L.append("직전에 실행된 arm별 followup TTFT (자리 교대로 두 방식이 같은 직전 arm 구성을 갖는다):\n")
        L.append(md_table(["방법", "직전 arm", "followup 수", "TTFT 평균(ms)", "p50"],
                          [[x["method"], x["preceding"], f"{x['followups']:,}", f"{x['ttft_mean_ms']:.1f}", f"{x['ttft_p50_ms']:.1f}"] for x in pred],
                          ["---", "---", "---:", "---:", "---:"]))
        summary["main"] = {"arms": lat, "paired": pairs, "requests": len(rows), "two_ours_difference": diff, "turn1_cost": cost,
                           "ttft_by_preceding_arm": pred}
    else:
        L.append("## 판정\n\n미실행 (본실험이 끝나지 않았다).\n")
    write_json(res / "verdict.json", verdict)
    summary["verdict"] = verdict

    # ---- gates
    L.append("## Gate (본실험 전)\n")
    spec = [("1. 저장 내용 일치 — gate 4장", "gate1_store_contents_gate"), ("1. 저장 내용 일치 — DEV 40장", "gate1_store_contents_dev"),
            ("2. 질문 무관 — gate 4장", "gate2_question_independent_gate"), ("2. 질문 무관 — DEV 40장", "gate2_question_independent_dev"),
            ("3. Ours(원래 순서) 동기 = 비동기 — gate 4장", "gate3_sync_equals_async_gate"), ("3. Ours(원래 순서) 동기 = 비동기 — DEV 40장", "gate3_sync_equals_async_dev"),
            ("4. 연속성 (2026-11-17 실행과 토큰 일치) — gate 4장", "gate4_continuity_gate"), ("4. 연속성 — DEV 40장", "gate4_continuity_dev"),
            ("5. 재현성 (Ours(원래 순서) 두 번 실행) — gate 4장", "gate5_repeat_gate"), ("5. 재현성 — DEV 40장", "gate5_repeat_dev"),
            ("6. 기존 테스트", "gate6_tests"), ("7. DEV 40장 파일럿 실행·감사 (4 arm)", "dev_execution"),
            ("gate phase 실행·감사 (4장, 6 arm)", "gate_execution"), ("devgate phase 실행·감사 (DEV 40장, 3 arm)", "devgate_execution")]
    gates = {}; gate_rows = []
    for label, name in spec:
        g = gate_status(run, name); gates[name] = g
        detail = ""
        if g is not None:
            if name.startswith("gate1"):
                detail = f"{g['pass_count']}/{g['image_count']}장 일치; 비교한 K·V 행 {g['kv_rows_compared']:,}개 중 다른 행 {g['kv_rows_different']}개; separator 블록·payload 크기 일치"
            elif name.startswith("gate2"):
                detail = (f"{g['pass_count']}/{g['image_count']}장에서 질문 3개로 만든 store의 선택·배치가 같음 (layer별 인덱스 표, 행 수, byte table, 파일 구성, "
                          "요청 기록·payload digest 외 메타데이터), commit된 store의 인덱스 표와도 같음. KV payload 바이트는 판정하지 않고 아래에 보고")
            elif name.startswith("gate3") or name.startswith("gate5"):
                detail = f"{g['requests']:,}건 중 불일치 {g['mismatch_count']}건"
            elif name.startswith("gate4"):
                detail = "; ".join(f"{m['name']}: {m['requests']:,}건 중 불일치 {m['mismatch_count']}건" for m in g["methods"])
            elif name == "gate6_tests":
                detail = "; ".join(f"transformers {s['transformers']}: {s['ran']}개 실행, 실패 {len(s['failed'])}개(모두 이 작업 전부터의 환경 실패), 새 실패 {len(s['new_failures'])}개"
                                   for s in g["suites"]) + f"; 새 테스트 파일 {g['new_unit_tests']['ran']}개 통과"
            elif name.endswith("_execution"):
                detail = f"{g['requests']:,}/{g['expected']:,}건, {g.get('images')}장"
        gate_rows.append([label, g["status"] if g else NOT_RUN, detail])
    admission = gate_status(run, "admission")
    gate_rows.append(["본실험 진입", admission["status"] if admission else NOT_RUN, "gate 1–7이 모두 PASS일 때만 실행" if admission else ""])
    L.append(md_table(["gate", "결과", "내용"], gate_rows, ["---", "---", "---"]))
    summary["gates"] = {k: (v["status"] if v else None) for k, v in gates.items()}; summary["admission"] = admission["status"] if admission else None
    g2 = [(label, gates.get(f"gate2_question_independent_{tag}")) for tag, label in (("gate", "gate 4장"), ("dev", "DEV 40장"))]
    if any(g for _, g in g2):
        L.append("gate 2의 보고 항목(판정 아님) — 질문 3개로 만든 store의 KV payload 바이트:\n")
        L.append(md_table(["대상", "질문 쌍", "프롬프트 길이가 같은 쌍: payload 일치", "길이가 다른 쌍: payload 일치", "길이가 다를 때 다른 FP16 원소 비율(평균)",
                           "최대 절댓값 차", "재배열 순서로 쓴 store도 같은 쌍에서 같고 다른가", "첫 질문의 payload = commit된 store"],
                          [[label, s_["question_pairs"], f"{s_['equal_prompt_length_pairs_with_equal_payload']}/{s_['equal_prompt_length_pairs']}",
                            f"{s_['different_prompt_length_pairs_with_equal_payload']}/{s_['different_prompt_length_pairs']}",
                            "—" if s_["mean_fraction_of_fp16_elements_that_differ_when_lengths_differ"] is None
                            else f"{100 * s_['mean_fraction_of_fp16_elements_that_differ_when_lengths_differ']:.1f}%",
                            "—" if s_["max_abs_diff_when_lengths_differ"] is None else f"{s_['max_abs_diff_when_lengths_differ']:.3f}",
                            f"{s_['pairs_where_importance_order_store_agrees_with_original_order_store_on_equal_or_different']}/{s_['question_pairs']}쌍 같은 양상",
                            f"{s_['images_whose_first_question_payload_equals_committed_store']}/{g['image_count']}장"]
                           for label, g in g2 if g for s_ in [g["payload_summary"]]]))
        L.append("턴 1에서 캡처되는 Visual KV는 그 요청 프롬프트의 토큰 길이에 따라 값이 달라진다(질문 내용이 아니라 길이; 길이가 같으면 바이트까지 같다). "
                 "이 성질은 store 종류와 무관하며, 같은 캡처를 재배열 순서로 쓴 store의 payload도 같은 쌍에서 같고 다르다. "
                 "따라서 gate 2는 사용자 결정(2026-11-20)에 따라 선택·배치의 일치로 판정하고 payload 바이트는 보고만 한다(계약 6절, 12절).\n")
        summary["gate2_payload"] = {label: g["payload_summary"] for label, g in g2 if g}
    for tag, label in (("gate", "gate 4장"), ("dev", "DEV 40장")):
        d = gate_status(run, f"reported_two_ours_difference_{tag}"); t = gate_status(run, f"reported_turn1_identity_{tag}")
        if d and t:
            b = d["by_turn"]
            L.append(f"보고 항목(gate 아님), {label}: 두 Ours의 생성 토큰열이 다른 요청 T2 {b['t2']['tokens_differ']}/{b['t2']['requests']}건, "
                     f"T3 {b['t3']['tokens_differ']}/{b['t3']['requests']}건. 모든 arm의 턴 1 토큰열이 같은 대화 {t['dialogues'] - t['mismatch_count']}/{t['dialogues']}개.\n")
    if posthoc:
        g1p = gate_status(run, "posthoc_store_contents_main"); t1p = gate_status(run, "posthoc_turn1_identity_main")
        L.append("### 사후 확인 (본실험 398장 전체)\n")
        L.append(md_table(["항목", "결과"], [
            ["연속성: 2026-11-17 실행과 생성 토큰열 일치", "; ".join(f"{m['name']} {m['requests']:,}건 중 불일치 {m['mismatch_count']}건" for m in posthoc["methods"])],
            ["저장 내용 일치 (이미지별 감사 항목)", NOT_RUN if not g1p else f"{g1p['pass_count']}/{g1p['image_count']}장, 다른 K·V 행 {g1p['kv_rows_different']}개 / {g1p['kv_rows_compared']:,}개"],
            ["모든 arm의 턴 1 토큰열이 같은 대화", NOT_RUN if not t1p else f"{t1p['dialogues'] - t1p['mismatch_count']:,} / {t1p['dialogues']:,}"],
        ], ["---", "---"]))
        summary["posthoc"] = {"continuity": posthoc["status"], "store_contents": g1p["status"] if g1p else None}

    # ---- dev pilot
    dev_done = (run / "dev" / "validation.json").exists()
    if dev_done:
        drows, dattempts = load_phase(run, "dev")
        dmethods = config["phases"]["dev"]["methods"]
        dlat = arm_table(drows, dmethods, names)
        csv_write(res / "dev" / "arms.csv", list(dlat.values()))
        L.append("## DEV 40장 파일럿 (gate 7; 본조건 4 arm, T2–T3 followup)\n")
        L.append(f"DEV 40장의 대화 {len({d for (_, d, _) in drows})}개, arm당 followup {dlat[dmethods[0]]['followups']}건. 본실험 진입 전 확인용이며 판정에 쓰지 않는다. "
                 "DEV 이미지는 본실험에도 포함된다.\n")
        L.append(md_table(["방법", "followup EM", "TTFT 평균(ms)", "p50", "p95", "SSD MB/followup", "읽기 호출 수"],
                          [[dlat[m]["name"], f"{dlat[m]['followup_acc_pct']:.2f}%", f"{dlat[m]['ttft_mean_ms']:.1f}", f"{dlat[m]['ttft_p50_ms']:.1f}",
                            f"{dlat[m]['ttft_p95_ms']:.1f}", f"{dlat[m]['ssd_mb_per_followup']:.1f}", f"{dlat[m]['read_calls_mean']:.1f}"] for m in dmethods]))
        dp = [pair_row(names, drows, B, drows, A)]
        csv_write(res / "dev" / "paired.csv", dp)
        L.append(pair_table(dp))
        L.append(f"TTFT 평균 비율 (원래 순서 / 재배열): {dlat[B]['ttft_mean_ms'] / dlat[A]['ttft_mean_ms']:.4f}.\n")
        summary["dev"] = {"arms": dlat, "paired": dp}
    else:
        L.append("## DEV 40장 파일럿\n\n미실행.\n")

    # ---- settings, code changes
    L.append("## 설정\n")
    L.append(f"- 실행: `runs/{run.name}` (요청 단위 원본), 이 디렉터리(`results/{run.name}`)는 집계와 보고서.\n"
             f"- 환경: Python {env['python'].split()[0]}, torch {env['torch']}, transformers {env['transformers']} (`{env['executable']}`).\n"
             f"- 모델 `{config['model']}` revision `{config['revision'][:12]}…`, NF4/BF16/eager, FP16 SSD KV, greedy max 16, batch 1, CPU threads 2.\n"
             "- 데이터·절차: MT-GQA-reconstructed 398 images / 4,061 dialogues / 3턴, 2026-11-17 본실험과 같은 프로토콜. 모든 arm이 매 대화 T1을 픽셀부터 계산하고 "
             "T2–T3는 자기 arm의 생성 이력을 쓴다. 저장 KV 읽기는 요청 직전에 page cache를 비운 cold 조건, 비동기 읽기, strict normalized EM.\n"
             "- 두 Ours: PYR-beta-raw 벡터, 총 KV 예산 25%, packed 저장. layer별 토큰 집합과 k_l은 같고, 파일 안의 행 순서와 GPU cache에서의 자리만 다르다.\n"
             "- arm 순서: 대화 단위 순환 회전에 더해, 두 Ours가 한 블록(arm 수만큼의 대화)마다 자리를 맞바꾼다(계약 4절; 이전 프로토콜에 추가한 것).\n")
    L.append("## 8. 기존 코드에서 추가하거나 바꾼 부분과 새 테스트\n")
    L.append(md_table(["파일", "내용"], [list(x) for x in CODE_CHANGES], ["---", "---"]))
    changed = config.get("source_changed_since_20261117_freeze"); added = config.get("source_added_since_20261117_freeze")
    if changed is not None:
        L.append(f"2026-11-17 실행의 소스 freeze와 비교해 SHA256이 달라진 파일: {', '.join('`' + x + '`' for x in changed) or '없음'}. "
                 f"새로 생긴 파일: {', '.join('`' + x + '`' for x in added) or '없음'}. "
                 "기존 재배열 경로는 기본값이며, gate 4(연속성)가 ReComp·FullLoad·Ours(재배열)의 생성 토큰이 그 실행과 같음을 확인한다.\n")

    prot = R.protection_after(H, run, res)
    L.append("## 기존 산출물 보호\n")
    L.append(f"- `results/`, `runs/`(이 run 제외), `kvstore*`의 기존 파일 {prot['before']['files']:,}개 (경로, 크기, mtime) 대조: **{prot['status']}**"
             + ("" if prot["differences"] is None else f" (변경 {len(prot['differences']['modified'])}, 누락 {len(prot['differences']['missing'])}, 새 파일 {prot['differences']['added_count']})") + ".\n")
    summary["protection"] = {"status": prot["status"], "files": prot["before"]["files"]}
    marks = ssd_written(run); plan = json.loads((run / "storage_plan.json").read_text())
    if marks and "before_freeze" in marks and "after_main" in marks:
        measured = marks["after_main"] - marks["before_freeze"]
        L.append(f"- SSD 쓰기량(장치 전체, `/sys/block/nvme0n1/stat`): 실측 {measured / 1e9:.1f} GB, 예상 {plan['expected_ssd_write_bytes']['total'] / 1e9:.1f} GB "
                 "(예상에는 요청 원본·증거 파일이 들어 있지 않다).\n")
        summary["ssd_written_bytes"] = {"measured": measured, "expected": plan["expected_ssd_write_bytes"]["total"]}
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
    print(json.dumps({"verdict": verdict.get("status"), **({k: verdict[k] for k in "abc"} if "a" in verdict else {})}, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    build(args.run.resolve())


if __name__ == "__main__":
    main()

"""Report for the IMPRESS next-layer probe-key prefetch measurement.

Reads one ``scripts/04_eval.py --run-dir`` output that holds four arms
(ReComp, FullLoad, ``impress_sync``, ``impress_prefetch``) measured with
``--alternate-selector-order``, and writes ``report.md`` + ``summary.json``.
Only measured artifacts are read; anything not measured is written "미실행".

  python scripts/186_report_impress_prefetch.py \\
      --run results/impress_prefetch_<UTC>/main \\
      --out results/impress_prefetch_<UTC>

Method names in the report are fixed: "IMPRESS (동기)" and
"IMPRESS (선행 로딩)".  The option values behind them are listed once, in the
settings section.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

NOT_RUN = "미실행"
SYNC, PRE = "impress_sync", "impress_prefetch"
NAMES = {"recompute": "ReComp", "fullload": "FullLoad",
         SYNC: "IMPRESS (동기)", PRE: "IMPRESS (선행 로딩)"}
ARMS = ("recompute", "fullload", SYNC, PRE)
REFERENCE_RUN = "results/impress_fidelity_20261116T052410Z/h2o"
REFERENCE_METHOD = "sparsevlm_fallback_v_selected"
# request-level fields that must match between the two IMPRESS arms / the
# earlier run
IDENTITY_FIELDS = ("generated_token_ids", "ssd_read_bytes", "preads")
REPRO_FIELDS = ("answer", "generated_token_ids", "ssd_read_bytes", "preads",
                "fallback_rate")


def t_critical(df):
    """Two-sided 95% Student-t critical value."""
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except Exception:                                    # noqa: BLE001
        return 1.96 + 2.4 / max(df, 1)       # close for df >= 30


def mean_ci(values):
    """Mean, 95% CI half-width (Student t) and n of a 1-D sample."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return None
    if v.size == 1:
        return {"n": 1, "mean": float(v[0]), "ci_low": None, "ci_high": None}
    se = v.std(ddof=1) / math.sqrt(v.size)
    half = t_critical(v.size - 1) * se
    return {"n": int(v.size), "mean": float(v.mean()),
            "ci_low": float(v.mean() - half), "ci_high": float(v.mean() + half),
            "median": float(np.median(v)), "sd": float(v.std(ddof=1))}


def fmt_ci(c, unit=" ms"):
    if c is None:
        return NOT_RUN
    if c["ci_low"] is None:
        return f"{c['mean']:+.2f}{unit} (n={c['n']})"
    return (f"{c['mean']:+.2f}{unit} (95% CI {c['ci_low']:+.2f} ~ "
            f"{c['ci_high']:+.2f}, n={c['n']})")


def f(value, spec="{:.1f}", scale=1.0):
    return NOT_RUN if value is None else spec.format(value * scale)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows])


def load(path):
    file = Path(path) / "results.json"
    if not file.exists():
        return None
    with open(file) as fh:
        return json.load(fh)


def by_request(doc, method):
    return {(str(r["image_id"]), str(r["question_id"])): r[method]
            for r in doc["rows"] if method in r}


def criterion1(doc, reference):
    """IMPRESS (동기) reproduces the earlier run request by request."""
    if doc is None or reference is None:
        return {"status": NOT_RUN}
    new, ref = by_request(doc, SYNC), by_request(reference, REFERENCE_METHOD)
    keys = sorted(set(new) & set(ref))
    result = {"reference_run": REFERENCE_RUN,
              "reference_method": REFERENCE_METHOD,
              "requests_new": len(new), "requests_reference": len(ref),
              "common": len(keys), "fields": {}}
    differing = set()
    for field in REPRO_FIELDS:
        same = [k for k in keys if new[k].get(field) == ref[k].get(field)]
        result["fields"][field] = len(same)
        differing |= set(keys) - set(same)
    pm, rpm = (doc["summary"]["per_method"][SYNC],
               reference["summary"]["per_method"][REFERENCE_METHOD])
    result["aggregate"] = {
        "accuracy": [rpm["acc"], pm["acc"]],
        "ssd_mb": [rpm["disk_mb"], pm["disk_mb"]],
        "fallback_rate": [rpm["fallback_rate"], pm["fallback_rate"]]}
    result["differing_requests"] = len(differing)
    result["differing_examples"] = [list(k) for k in sorted(differing)[:10]]
    ok = (len(keys) == len(new) == len(ref) and len(keys) > 0
          and not differing)
    result["status"] = "PASS" if ok else "FAIL"
    return result


def criterion2(doc):
    """Both IMPRESS arms: same tokens, bytes and pread counts everywhere."""
    if doc is None:
        return {"status": NOT_RUN}
    a, b = by_request(doc, SYNC), by_request(doc, PRE)
    keys = sorted(set(a) | set(b))
    missing = [k for k in keys if k not in a or k not in b]
    result = {"requests": len(keys), "missing_in_one_arm": len(missing),
              "fields": {}}
    differing = set()
    for field in IDENTITY_FIELDS + ("k_read_bytes", "v_read_bytes",
                                    "probe_read_bytes", "fallback_rate",
                                    "answer"):
        same = [k for k in keys if k not in missing
                and a[k].get(field) is not None
                and a[k].get(field) == b[k].get(field)]
        result["fields"][field] = len(same)
        if field in IDENTITY_FIELDS:
            differing |= set(keys) - set(same)
    result["differing_requests"] = len(differing)
    result["differing_examples"] = [list(k) for k in sorted(differing)[:10]]
    result["status"] = ("PASS" if keys and not missing and not differing
                        else "FAIL")
    return result


def paired(doc):
    """Paired TTFT difference (선행 로딩 - 동기), overall and by order."""
    if doc is None:
        return None
    diffs, first_sync, first_pre, second_minus_first = [], [], [], []
    orders = {"sync_first": 0, "prefetch_first": 0, "unknown": 0}
    for r in doc["rows"]:
        if SYNC not in r or PRE not in r:
            continue
        s, p = r[SYNC], r[PRE]
        d = p["ttft_ms"] - s["ttft_ms"]
        diffs.append(d)
        si, pi = s.get("measure_index"), p.get("measure_index")
        if si is None or pi is None:
            orders["unknown"] += 1
        elif si < pi:
            orders["sync_first"] += 1
            first_sync.append(d)
            second_minus_first.append(d)          # prefetch was second
        else:
            orders["prefetch_first"] += 1
            first_pre.append(d)
            second_minus_first.append(-d)         # sync was second
    diffs = np.asarray(diffs)
    out = {"orders": orders,
           "all": mean_ci(diffs),
           "prefetch_faster_requests": int((diffs < 0).sum()),
           "prefetch_slower_requests": int((diffs > 0).sum()),
           "sync_measured_first": mean_ci(first_sync),
           "prefetch_measured_first": mean_ci(first_pre),
           # position effect: (arm measured second) - (arm measured first)
           "second_minus_first": mean_ci(second_minus_first)}
    if first_sync and first_pre:
        # equal-weight average of the two order groups: the position effect
        # cancels exactly when the groups are the same size
        out["order_balanced_mean"] = float(
            (np.mean(first_sync) + np.mean(first_pre)) / 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference-run", default=REFERENCE_RUN)
    ap.add_argument("--notes", default=None,
                    help="markdown file appended verbatim")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    doc, reference = load(args.run), load(args.reference_run)
    pm = doc["summary"]["per_method"] if doc else {}
    c1, c2, pair = criterion1(doc, reference), criterion2(doc), paired(doc)

    fl, rc = pm.get("fullload"), pm.get("recompute")
    rows = []
    for m in ARMS:
        d = pm.get(m)
        if d is None:
            rows.append([NAMES[m]] + [NOT_RUN] * 8)
            continue
        ttft = d["ttft_mean_ms"]
        rows.append([
            NAMES[m], f(d["acc"], "{:.2f}%", 100.0), f(ttft),
            f(d["ttft_p50_ms"]), f(d["ttft_p95_ms"]),
            f(ttft / fl["ttft_mean_ms"] if fl else None, "{:.3f}×"),
            f(ttft / rc["ttft_mean_ms"] if rc else None, "{:.3f}×"),
            "0.0" if m == "recompute" else f(d.get("disk_mb")),
            ("-" if m in ("recompute", "fullload")
             else f(d.get("fallback_rate"), "{:.1f}%", 100.0)),
        ])

    timing_rows = []
    for m in (SYNC, PRE):
        d = pm.get(m)
        if d is None:
            timing_rows.append([NAMES[m]] + [NOT_RUN] * 9)
            continue
        timing_rows.append([
            NAMES[m], f(d.get("probe_read_ms"), "{:.2f}"),
            f(d.get("probe_pread_ms"), "{:.2f}"),
            f(d.get("probe_hook_blocked_ms"), "{:.2f}"),
            f(d.get("prefetch_wait_ms"), "{:.2f}"),
            f(d.get("prefetched_probe_reads"), "{:.1f}"),
            f(d.get("prefetch_ready_on_arrival"), "{:.2f}"),
            f(d.get("hook_ms")), f(d.get("disk_ms")), f(d.get("prefill_ms")),
        ])

    summary = {
        "schema_version": 1,
        "run": args.run, "reference_run": args.reference_run,
        "settings": ({k: doc["summary"].get(k) for k in (
            "n", "n_images", "ratio", "alpha", "probe_heads", "cold",
            "importance_metric", "fallback_v", "prefetch",
            "alternate_selector_order", "store", "store_importance_metric",
            "skip", "questions_per_image_requested", "workload_sha256",
            "command")} if doc else None),
        "per_method": {m: pm.get(m) for m in ARMS},
        "criterion_1_reproduction": c1,
        "criterion_2_identity": c2,
        "paired_ttft_prefetch_minus_sync_ms": pair,
    }
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)

    L = ["# IMPRESS 다음 layer probe key 선행 로딩 측정", "",
         "측정값만 적는다. 측정하지 않은 값은 \"미실행\"이다.", "",
         "## 설정", ""]
    if doc is None:
        L.append(f"- {NOT_RUN}")
    else:
        s = doc["summary"]
        L += [
            f"- 워크로드: GQA {s['n_images']}장 / {s['n']}문항 "
            f"(`--skip {s['skip']} --questions "
            f"{s['questions_per_image_requested']}`), 비율 {s['ratio']}, "
            f"α {s['alpha']}, probe head {s['probe_heads']}개, "
            f"{'cold' if s['cold'] else 'warm'} page cache",
            f"- store: `{s['store']}` (재배치 지표 "
            f"{s['store_importance_metric']})",
            f"- 두 IMPRESS arm 공통: `importance_metric={s['importance_metric']}`"
            f", `fallback_v={s['fallback_v']}`",
            f"- IMPRESS (동기): `prefetch={pm[SYNC].get('prefetch')}` / "
            f"IMPRESS (선행 로딩): `prefetch={pm[PRE].get('prefetch')}`",
            f"- 측정 순서: 요청마다 ReComp → FullLoad → 두 IMPRESS arm. 두 "
            f"IMPRESS arm의 순서는 요청마다 번갈아 바꿨다"
            f"(`--alternate-selector-order`="
            f"{s.get('alternate_selector_order')}). 요청별 순서는 "
            f"`results.json`의 `selector_order`와 각 arm의 `measure_index`에 "
            f"기록돼 있다."
            + (f" 동기 먼저 {pair['orders']['sync_first']}건, 선행 로딩 먼저 "
               f"{pair['orders']['prefetch_first']}건." if pair else ""),
            f"- 명령: `{s['command']}`"]
    L += ["", "## 결과 표", "",
          "FullLoad·ReComp 대비는 같은 실행 안의 TTFT 평균 비율이다.", "",
          table(["arm", "정확도", "TTFT 평균(ms)", "p50", "p95",
                 "FullLoad 대비", "ReComp 대비", "SSD MB", "fallback 비율"],
                rows), "",
          "## 통과 기준 1 — IMPRESS (동기)가 이전 결과를 요청 단위로 재현하는가",
          ""]
    if c1["status"] == NOT_RUN:
        L.append(f"- {NOT_RUN}")
    else:
        a = c1["aggregate"]
        L += [f"- 판정: **{c1['status']}** (기준: `{c1['reference_run']}`의 "
              f"같은 설정 arm, 공통 요청 {c1['common']}개)",
              "- 요청 단위 일치: " + ", ".join(
                  f"`{k}` {v}/{c1['common']}"
                  for k, v in c1["fields"].items())
              + f". 하나라도 다른 요청 {c1['differing_requests']}개",
              f"- 집계: 정확도 {a['accuracy'][0]*100:.2f}% → "
              f"{a['accuracy'][1]*100:.2f}%, SSD MB {a['ssd_mb'][0]:.1f} → "
              f"{a['ssd_mb'][1]:.1f}, fallback "
              f"{a['fallback_rate'][0]*100:.1f}% → "
              f"{a['fallback_rate'][1]*100:.1f}%"]
    L += ["", "## 통과 기준 2 — 두 IMPRESS arm의 출력과 읽기량이 같은가", ""]
    if c2["status"] == NOT_RUN:
        L.append(f"- {NOT_RUN}")
    else:
        n = c2["requests"]
        L += [f"- 판정: **{c2['status']}** ({n}개 요청, 한쪽 누락 "
              f"{c2['missing_in_one_arm']})",
              "- 일치한 요청 수: " + ", ".join(
                  f"`{k}` {v}/{n}" for k, v in c2["fields"].items())]
    L += ["", "## paired TTFT 차이 (IMPRESS (선행 로딩) − IMPRESS (동기))", "",
          "음수면 선행 로딩이 빠르다. 신뢰구간은 요청 단위 차이의 "
          "Student t 95% 구간이다.", ""]
    if pair is None or pair["all"] is None:
        L.append(f"- {NOT_RUN}")
    else:
        L += [f"- 전체: {fmt_ci(pair['all'])}, 중앙값 "
              f"{pair['all']['median']:+.2f} ms. 선행 로딩이 빠른 요청 "
              f"{pair['prefetch_faster_requests']}개, 느린 요청 "
              f"{pair['prefetch_slower_requests']}개",
              f"- 동기를 먼저 잰 요청: {fmt_ci(pair['sync_measured_first'])}",
              f"- 선행 로딩을 먼저 잰 요청: "
              f"{fmt_ci(pair['prefetch_measured_first'])}",
              f"- 순서 효과 (나중에 잰 arm − 먼저 잰 arm, arm 구분 없이): "
              f"{fmt_ci(pair['second_minus_first'])}"]
        if "order_balanced_mean" in pair:
            L.append(f"- 두 순서 그룹의 평균을 같은 가중치로 합친 차이: "
                     f"{pair['order_balanced_mean']:+.2f} ms")
    L += ["", "## probe key 읽기 시간과 hook 시간 (요청당 평균, ms)", "",
          "선행 로딩에서는 읽기와 연산이 겹친다. 따라서 **hook 시간과 디스크 "
          "시간의 합을 TTFT로 해석하면 안 된다.** 아래 값들은 서로 겹치는 "
          "구간을 재므로 더해서 쓰지 않는다.", "",
          "- probe 읽기: probe key 읽기 한 번의 벽시계 시간(pread + 버퍼 "
          "변환) 합. 어느 스레드에서 읽었든 포함한다.",
          "- probe pread: 그중 `os.pread` 시간만.",
          "- hook이 probe 때문에 멈춘 시간: hook 안의 동기 읽기 시간 + 선행 "
          "읽기를 기다린 시간.",
          "- 선행 읽기 대기: hook이 선행 읽기 결과를 기다린 시간.",
          "- 도착 시 완료: hook이 결과를 요청했을 때 이미 읽기가 끝나 있던 "
          "layer 수(선행 읽기 layer 수 중).", "",
          table(["arm", "probe 읽기", "probe pread", "hook이 probe 때문에 멈춘 시간",
                 "선행 읽기 대기", "선행 읽기 layer 수", "도착 시 완료",
                 "hook 시간", "디스크(pread) 시간", "prefill 시간"],
                timing_rows)]
    if args.notes and Path(args.notes).exists():
        L += ["", Path(args.notes).read_text().rstrip()]
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {out / 'report.md'} and {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

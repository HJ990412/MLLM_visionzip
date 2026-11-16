"""Report for the IMPRESS-fidelity pilot (importance metric x fallback V policy).

Reads the ``scripts/04_eval.py --run-dir`` outputs of the pilot and writes
``report.md`` + ``summary.json``.  It only reads measured artifacts: a value
that was not measured is written as "미실행", never estimated.

  python scripts/185_report_impress_fidelity.py \\
      --out results/impress_fidelity_<UTC> \\
      --gate results/impress_fidelity_<UTC>/gate_sparsevlm_full \\
      --sparsevlm-run results/impress_fidelity_<UTC>/sparsevlm \\
      --h2o-run results/impress_fidelity_<UTC>/h2o

Arms
  sparsevlm+full       selector "sparsevlm"                     (metric run 1)
  sparsevlm+selected   selector "sparsevlm_fallback_v_selected" (metric run 1)
  h2o+full             selector "sparsevlm"                     (metric run 2)
  h2o+selected         selector "sparsevlm_fallback_v_selected" (metric run 2)
Each metric run also carries its own ReComp and FullLoad; ratios against
FullLoad always use the FullLoad of the SAME run (same store, same pass).

``fallback_v_token_exact_bytes`` is a HYPOTHETICAL lower bound (V bytes if the
store could serve exactly the kept (head, token) pairs).  It was never read
from the SSD and is reported in its own table, apart from measured traffic.
"""
import argparse
import json
import sys
from pathlib import Path

NOT_RUN = "미실행"
FULL, SELECTED = "sparsevlm", "sparsevlm_fallback_v_selected"

# criterion 1 tolerances ("roughly reproduces"); chosen for this report and
# printed next to the raw differences so they can be re-judged
TOL_ACC_PP = 2.0          # 240 questions: 2.0 pp == about 5 answers
TOL_MB_REL = 0.01         # 1 % of the reference SSD MB
TOL_FALLBACK_ABS = 0.01   # 1 percentage point of fallback rate
EXPECTED_SELECTED_RATIO = (0.737, 0.772)     # caller's sanity range, not a goal


def load_run(path):
    if path is None:
        return None
    file = Path(path) / "results.json"
    if not file.exists():
        return None
    with open(file) as f:
        doc = json.load(f)
    doc["_dir"] = str(path)
    return doc


def arm(run, method):
    """Per-method summary of a run, or None when it was not measured."""
    if run is None:
        return None
    return run["summary"]["per_method"].get(method)


def fmt(value, spec="{:.1f}", scale=1.0):
    return NOT_RUN if value is None else spec.format(value * scale)


def row_for(label, run, method, fullload):
    d = arm(run, method)
    if d is None:
        return [label] + [NOT_RUN] * 15
    mb = d.get("disk_mb")
    fl_mb = fullload.get("disk_mb") if fullload else None
    selector = method not in ("recompute", "fullload")

    def only_selector(key, spec="{:.3f}", scale=1.0):
        return fmt(d.get(key), spec, scale) if selector else "-"

    def mbytes(key):
        # ReComp reads nothing from the store: the fields are absent, which
        # means "not applicable", not "not measured"
        if method == "recompute":
            return "0.0"
        return fmt(d.get(key), "{:.1f}", 1e-6)

    return [
        label,
        fmt(d.get("acc"), "{:.2f}%", 100.0),
        fmt(d.get("ttft_mean_ms")), fmt(d.get("ttft_p50_ms")),
        fmt(d.get("ttft_p95_ms")),
        "0.0" if method == "recompute" else fmt(mb),
        ("0.0%" if method == "recompute" else
         fmt(mb / fl_mb if mb is not None and fl_mb else None,
             "{:.1f}%", 100.0)),
        mbytes("probe_read_bytes"), mbytes("k_read_bytes"),
        mbytes("v_read_bytes"),
        only_selector("fallback_rate", "{:.1f}%", 100.0),
        only_selector("mean_jaccard", "{:.4f}"),
        ("-" if method == "recompute" else
         fmt(d.get("touched_k_chunk_fraction"), "{:.1f}%", 100.0)),
        ("-" if method == "recompute" else
         fmt(d.get("touched_v_chunk_fraction"), "{:.1f}%", 100.0)),
        "-" if method == "recompute" else fmt(d.get("hook_ms")),
        only_selector("n_score_rows", "{:.1f}"),
    ]


HEADER = ["설정", "정확도", "TTFT 평균(ms)", "p50", "p95", "SSD MB",
          "FullLoad 대비", "probe MB", "K MB", "V MB", "fallback 비율",
          "평균 Jaccard", "K chunk 비율", "V chunk 비율", "hook(ms)",
          "점수 행 수"]


def table(rows, header=HEADER):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def requests(run, method):
    """{(image_id, question_id): packed request} for one method of a run."""
    if run is None:
        return None
    out = {}
    for r in run["rows"]:
        if method in r:
            out[(str(r["image_id"]), str(r["question_id"]))] = r[method]
    return out or None


def criterion1(gate, reference_path):
    """sparsevlm+full vs the stored results/eval_b25.json 'sparsevlm' arm."""
    result = {"status": NOT_RUN}
    got = arm(gate, FULL)
    if got is None or not Path(reference_path).exists():
        return result
    with open(reference_path) as f:
        ref_doc = json.load(f)
    ref = ref_doc["summary"]["per_method"]["sparsevlm"]
    d_acc = (got["acc"] - ref["acc"]) * 100.0
    d_mb = got["disk_mb"] - ref["disk_mb"]
    d_fb = got["fallback_rate"] - ref["fallback_rate"]
    result.update({
        "reference": {"acc": ref["acc"], "disk_mb": ref["disk_mb"],
                      "fallback_rate": ref["fallback_rate"],
                      "bytes_vs_fullload": ref.get("bytes_vs_fullload"),
                      "touched_chunk_fraction":
                          ref.get("touched_chunk_fraction")},
        "measured": {"acc": got["acc"], "disk_mb": got["disk_mb"],
                     "fallback_rate": got["fallback_rate"],
                     "bytes_vs_fullload": got.get("bytes_vs_fullload"),
                     "touched_chunk_fraction":
                         got.get("touched_chunk_fraction")},
        "delta": {"acc_pp": d_acc, "disk_mb": d_mb,
                  "disk_mb_rel": d_mb / ref["disk_mb"],
                  "fallback_rate": d_fb},
        "tolerance": {"acc_pp": TOL_ACC_PP, "disk_mb_rel": TOL_MB_REL,
                      "fallback_rate": TOL_FALLBACK_ABS},
    })
    ok = (abs(d_acc) <= TOL_ACC_PP
          and abs(d_mb) / ref["disk_mb"] <= TOL_MB_REL
          and abs(d_fb) <= TOL_FALLBACK_ABS)
    result["status"] = "PASS" if ok else "FAIL"

    # request-level agreement with the stored run (same image/question ids)
    ref_rows = {(str(r["image_id"]), str(r["question_id"])): r["sparsevlm"]
                for r in ref_doc["rows"] if "sparsevlm" in r}
    new_rows = requests(gate, FULL)
    common = sorted(set(ref_rows) & set(new_rows))
    per_request = {"reference_requests": len(ref_rows),
                   "measured_requests": len(new_rows),
                   "common_requests": len(common)}
    if common:
        per_request.update({
            "same_answer": sum(ref_rows[k]["answer"] == new_rows[k]["answer"]
                               for k in common),
            "same_correctness": sum(ref_rows[k]["acc"] == new_rows[k]["acc"]
                                    for k in common),
            "same_disk_mb": sum(
                abs(ref_rows[k]["disk_mb"] - new_rows[k]["disk_mb"]) < 1e-9
                for k in common),
            "same_fallback_rate": sum(
                ref_rows[k]["fallback_rate"] == new_rows[k]["fallback_rate"]
                for k in common),
            "max_abs_disk_mb_diff": max(
                abs(ref_rows[k]["disk_mb"] - new_rows[k]["disk_mb"])
                for k in common),
        })
    result["per_request"] = per_request
    return result


def criterion2(run):
    """full vs selected: identical generated_token_ids for every request."""
    a, b = requests(run, FULL), requests(run, SELECTED)
    if a is None or b is None:
        return {"status": NOT_RUN}
    keys = sorted(set(a) | set(b))
    missing = [k for k in keys if k not in a or k not in b]
    no_ids = [k for k in keys if k not in missing and (
        a[k].get("generated_token_ids") is None
        or b[k].get("generated_token_ids") is None)]
    differ = [k for k in keys if k not in missing and k not in no_ids
              and a[k]["generated_token_ids"] != b[k]["generated_token_ids"]]
    compared = len(keys) - len(missing) - len(no_ids)
    ok = not missing and not no_ids and not differ and compared > 0
    return {
        "status": "PASS" if ok else "FAIL",
        "requests": len(keys), "compared": compared,
        "identical": compared - len(differ), "different": len(differ),
        "missing_in_one_arm": len(missing),
        "without_token_ids": len(no_ids),
        "different_examples": [
            {"image_id": k[0], "question_id": k[1],
             "full": a[k]["generated_token_ids"],
             "selected": b[k]["generated_token_ids"]} for k in differ[:10]],
        # the keep masks are policy-independent, so these must agree too
        "same_fallback_rate": sum(
            a[k].get("fallback_rate") == b[k].get("fallback_rate")
            for k in keys if k not in missing),
        "same_k_read_bytes": sum(
            a[k].get("k_read_bytes") == b[k].get("k_read_bytes")
            for k in keys if k not in missing),
        "same_probe_read_bytes": sum(
            a[k].get("probe_read_bytes") == b[k].get("probe_read_bytes")
            for k in keys if k not in missing),
        "selected_v_bytes_gt_full": sum(
            (b[k].get("v_read_bytes") or 0) > (a[k].get("v_read_bytes") or 0)
            for k in keys if k not in missing),
    }


def v_volume_rows(label, run):
    """Measured vs hypothetical V volume of the fallback layers."""
    rows = []
    for name, method in (("full", FULL), ("selected", SELECTED)):
        d = arm(run, method)
        if d is None:
            rows.append([f"{label}+{name}"] + [NOT_RUN] * 5)
            continue
        rows.append([
            f"{label}+{name}",
            fmt(d.get("fallback_layers"), "{:.2f}"),
            fmt(d.get("fallback_k_chunks"), "{:.1f}"),
            fmt(d.get("fallback_v_chunks"), "{:.1f}"),
            fmt(d.get("fallback_v_read_bytes"), "{:.1f}", 1e-6),
            fmt(d.get("fallback_v_token_exact_bytes"), "{:.1f}", 1e-6),
        ])
    return rows


def run_conditions(run):
    if run is None:
        return NOT_RUN
    s = run["summary"]
    return (f"n={s['n']} / images={s['n_images']} / ratio={s['ratio']} / "
            f"alpha={s['alpha']} / probe={s['probe_heads']} / "
            f"cold={s['cold']} / importance_metric={s.get('importance_metric')}"
            f" / store_importance_metric={s.get('store_importance_metric')}"
            f" / store={s.get('store')} / workload_sha256="
            f"{s['workload_sha256'][:16]}")


def cross_run(run_a, run_b, method):
    """Agreement of a shared baseline across the two metric runs."""
    a, b = requests(run_a, method), requests(run_b, method)
    if a is None or b is None:
        return None
    keys = sorted(set(a) & set(b))
    return {"common": len(keys),
            "same_answer": sum(a[k]["answer"] == b[k]["answer"] for k in keys),
            "same_token_ids": sum(
                a[k].get("generated_token_ids")
                == b[k].get("generated_token_ids") for k in keys),
            "same_read_bytes": sum(
                a[k].get("ssd_read_bytes") == b[k].get("ssd_read_bytes")
                for k in keys)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate", default=None,
                    help="run dir of the criterion-1 pass (sparsevlm+full)")
    ap.add_argument("--sparsevlm-run", default=None)
    ap.add_argument("--h2o-run", default=None)
    ap.add_argument("--reference", default="results/eval_b25.json")
    ap.add_argument("--notes", default=None,
                    help="markdown file appended verbatim (procedure notes)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gate = load_run(args.gate)
    sv_run = load_run(args.sparsevlm_run)
    h2o_run = load_run(args.h2o_run)

    c1 = criterion1(gate, args.reference)
    c2 = {"sparsevlm": criterion2(sv_run), "h2o": criterion2(h2o_run)}

    rows = []
    for label, run in (("sparsevlm", sv_run), ("h2o", h2o_run)):
        fl = arm(run, "fullload")
        rows.append(row_for(f"{label}+full", run, FULL, fl))
        rows.append(row_for(f"{label}+selected", run, SELECTED, fl))
        rows.append(row_for(f"ReComp ({label} 실행)", run, "recompute", fl))
        rows.append(row_for(f"FullLoad ({label} 실행)", run, "fullload", fl))
    gate_rows = []
    if gate is not None:
        fl = arm(gate, "fullload")
        gate_rows = [row_for("sparsevlm+full (gate)", gate, FULL, fl),
                     row_for("ReComp (gate)", gate, "recompute", fl),
                     row_for("FullLoad (gate)", gate, "fullload", fl)]

    ratios = {}
    for label, run in (("sparsevlm", sv_run), ("h2o", h2o_run)):
        fl = arm(run, "fullload")
        for name, method in (("full", FULL), ("selected", SELECTED)):
            d = arm(run, method)
            ratios[f"{label}+{name}"] = (
                d["disk_mb"] / fl["disk_mb"] if d and fl else None)
    lo, hi = EXPECTED_SELECTED_RATIO
    sel_ratio = ratios.get("sparsevlm+selected")
    expectation = {
        "expected_range": [lo, hi],
        "sparsevlm_full_ratio": ratios.get("sparsevlm+full"),
        "sparsevlm_selected_ratio": sel_ratio,
        "within_expected_range": (None if sel_ratio is None
                                  else bool(lo <= sel_ratio <= hi)),
    }

    summary = {
        "schema_version": 1,
        "inputs": {"gate": args.gate, "sparsevlm_run": args.sparsevlm_run,
                   "h2o_run": args.h2o_run, "reference": args.reference},
        "criterion_1_reproduction": c1,
        "criterion_2_token_identity": c2,
        "bytes_vs_fullload": ratios,
        "expectation_check": expectation,
        "cross_run_baselines": {
            "recompute": cross_run(sv_run, h2o_run, "recompute"),
            "fullload": cross_run(sv_run, h2o_run, "fullload"),
            "gate_vs_sparsevlm_run_sparsevlm_full":
                cross_run(gate, sv_run, FULL)},
        "per_method": {
            "gate": gate["summary"]["per_method"] if gate else None,
            "sparsevlm_run": (sv_run["summary"]["per_method"]
                              if sv_run else None),
            "h2o_run": h2o_run["summary"]["per_method"] if h2o_run else None},
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)

    def c2_line(name):
        c = c2[name]
        if c["status"] == NOT_RUN:
            return f"- {name}: {NOT_RUN}"
        return (f"- {name}: **{c['status']}** — {c['identical']}/"
                f"{c['compared']}개 요청의 `generated_token_ids` 동일 "
                f"(다름 {c['different']}, 한쪽 누락 {c['missing_in_one_arm']}"
                f", token id 없음 {c['without_token_ids']}). "
                f"fallback 비율 동일 {c['same_fallback_rate']}/{c['requests']}"
                f", K 바이트 동일 {c['same_k_read_bytes']}/{c['requests']}, "
                f"probe 바이트 동일 {c['same_probe_read_bytes']}/"
                f"{c['requests']}, selected의 V 바이트가 full보다 큰 요청 "
                f"{c['selected_v_bytes_gt_full']}")

    lines = ["# IMPRESS 이식 경로 충실도 파일럿 (중요도 지표 × fallback V 읽기)",
             "",
             "측정값만 적는다. 측정하지 않은 값은 \"미실행\"이다.",
             "",
             "## 실행 조건",
             "",
             f"- gate: {run_conditions(gate)}",
             f"- sparsevlm 실행: {run_conditions(sv_run)}",
             f"- h2o 실행: {run_conditions(h2o_run)}",
             "",
             "## 결과 표",
             "",
             "FullLoad 대비 비율은 같은 실행(같은 store)의 FullLoad 기준이다. "
             "probe/K/V MB는 `os.pread`가 실제로 돌려준 바이트의 요청 평균이다. "
             "K·V chunk 비율은 layer당 읽은 chunk 수 / layer chunk 수의 평균"
             "(probe sidecar 제외). 점수 행 수는 중요도 점수에 쓰인 suffix "
             "query 행 수(sparsevlm: rater 수, h2o: suffix 토큰 수).",
             "",
             table(rows),
             ""]
    if gate_rows:
        lines += ["### gate 실행 (통과 기준 1 전용, 별도 실행)", "",
                  table(gate_rows), ""]
    lines += [
        "## fallback layer의 V 읽기량: 실측과 가상 하한",
        "",
        "`fallback V 실측 MB`는 fallback layer에서 실제로 pread한 V 바이트다. "
        "`V token-exact 가상 하한 MB`는 head별 토큰 단위로 정확히 읽을 수 "
        "있다고 가정했을 때의 V 바이트로, **실제로 읽지 않은 참고값**이며 "
        "위 표의 SSD MB·V MB에 포함되지 않는다. 값은 요청당 평균이다.",
        "",
        table(v_volume_rows("sparsevlm", sv_run) + v_volume_rows("h2o", h2o_run),
              ["설정", "fallback layer 수", "fallback K chunk 수",
               "fallback V chunk 수", "fallback V 실측 MB",
               "V token-exact 가상 하한 MB (읽지 않음)"]),
        "",
        "## 통과 기준 1 — sparsevlm+full이 `results/eval_b25.json`의 "
        "`sparsevlm`을 재현하는가",
        ""]
    if c1["status"] == NOT_RUN:
        lines.append(f"- {NOT_RUN}")
    else:
        r, m, d, t = (c1["reference"], c1["measured"], c1["delta"],
                      c1["tolerance"])
        lines += [
            f"- 판정: **{c1['status']}** (허용 오차는 이 보고서가 정한 값: "
            f"정확도 ±{t['acc_pp']} pp, SSD MB ±{t['disk_mb_rel']*100:.0f}%, "
            f"fallback ±{t['fallback_rate']*100:.0f} pp)",
            f"- 정확도: 기준 {r['acc']*100:.2f}% → 측정 {m['acc']*100:.2f}% "
            f"(차이 {d['acc_pp']:+.2f} pp)",
            f"- SSD MB: 기준 {r['disk_mb']:.3f} → 측정 {m['disk_mb']:.3f} "
            f"(차이 {d['disk_mb']:+.3f} MB, {d['disk_mb_rel']*100:+.3f}%)",
            f"- fallback 비율: 기준 {r['fallback_rate']*100:.2f}% → 측정 "
            f"{m['fallback_rate']*100:.2f}% (차이 "
            f"{d['fallback_rate']*100:+.2f} pp)",
            f"- FullLoad 대비: 기준 "
            f"{fmt(r.get('bytes_vs_fullload'), '{:.2f}%', 100.0)} → 측정 "
            f"{fmt(m.get('bytes_vs_fullload'), '{:.2f}%', 100.0)}"]
        p = c1["per_request"]
        if p.get("common_requests"):
            lines.append(
                f"- 요청 단위({p['common_requests']}개 공통): 답 동일 "
                f"{p['same_answer']}, 정오 동일 {p['same_correctness']}, "
                f"SSD MB 동일 {p['same_disk_mb']}, fallback 비율 동일 "
                f"{p['same_fallback_rate']}, MB 최대 차이 "
                f"{p['max_abs_disk_mb_diff']:.3f}")
    lines += ["",
              "## 통과 기준 2 — 같은 지표에서 full과 selected의 출력 토큰이 "
              "모두 같은가",
              "", c2_line("sparsevlm"), c2_line("h2o"), "",
              "## 참고 예상치 대조 (목표가 아니라 이상 여부 판단용)",
              ""]
    if sel_ratio is None:
        lines.append(f"- sparsevlm+selected의 FullLoad 대비 읽기량: {NOT_RUN}")
    else:
        lines.append(
            f"- sparsevlm 기준 FullLoad 대비 읽기량: full "
            f"{ratios['sparsevlm+full']*100:.2f}% → selected "
            f"{sel_ratio*100:.2f}% (예상 범위 {lo*100:.1f}~{hi*100:.1f}%: "
            f"{'범위 안' if expectation['within_expected_range'] else '범위 밖'})")
    for name in ("h2o+full", "h2o+selected"):
        lines.append(f"- {name}의 FullLoad 대비 읽기량: "
                     f"{fmt(ratios.get(name), '{:.2f}%', 100.0)}")
    cb = summary["cross_run_baselines"]
    lines += ["", "## 실행 간 기준선 일치", ""]
    for name, label in (("recompute", "ReComp (sparsevlm 실행 vs h2o 실행)"),
                        ("fullload", "FullLoad (sparsevlm 실행 vs h2o 실행)"),
                        ("gate_vs_sparsevlm_run_sparsevlm_full",
                         "sparsevlm+full (gate vs sparsevlm 실행)")):
        c = cb[name]
        lines.append(f"- {label}: " + (NOT_RUN if c is None else (
            f"공통 {c['common']}개 중 답 동일 {c['same_answer']}, "
            f"token id 동일 {c['same_token_ids']}, 읽기 바이트 동일 "
            f"{c['same_read_bytes']}")))
    if args.notes and Path(args.notes).exists():
        lines += ["", Path(args.notes).read_text().rstrip()]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out / 'report.md'} and {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

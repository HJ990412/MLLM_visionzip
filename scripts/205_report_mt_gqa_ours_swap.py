#!/usr/bin/env python3
"""Experiment A table (LLaVA-NeXT x MT-GQA, ReComp / FullLoad / IMPRESS / Ours) with the Ours row
replaced by Ours(원래 순서).

ReComp, FullLoad, IMPRESS: the stored per-request rows of runs/mt_gqa_impress_code_20261117T185832Z.
Ours: the per-request rows of ``ours_origorder_kv25_PYR-beta-raw_packed_async`` from
runs/mt_gqa_ours_order_20261120T052725Z (same dialogues, same turns, same protocol).  That run
reproduced the ReComp / FullLoad / Ours(재배열) tokens of the 11/17 run on every request, so
accuracy pairs across the two runs are exact; TTFT pairs and ratios mix two runs and are marked.

Reads committed rows only; measures nothing; writes results/mt_gqa_impress_code_ours_original_update/.
Neither source run's directories are modified.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
OLD = ROOT / "runs/mt_gqa_impress_code_20261117T185832Z"
NEW = ROOT / "runs/mt_gqa_ours_order_20261120T052725Z"
OUT = ROOT / "results/mt_gqa_impress_code_ours_original_update"
OURS = "ours_prefixstore_kv25_PYR-beta-raw_packed_async"          # the 11/17 method id of the Ours row
ORIG = "ours_origorder_kv25_PYR-beta-raw_packed_async"            # Ours(원래 순서) in the 11/20 run
OLD_KIND, NEW_KIND = "packed_PYR-beta-raw", "packedorig_PYR-beta-raw"


def _load(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


R = _load("198_report_mt_gqa_impress_code.py", "_swap_198")
md_table, csv_write, write_json, stats = R.md_table, R.csv_write, R.write_json, R.stats


def new_rows(phase):
    """Ours(원래 순서) rows of the 11/20 run in the 11/17 report's row format, under the old Ours id."""
    rows, attempts = {}, {}
    for cp in sorted((NEW / phase / "images").glob("*/COMMITTED.json")):
        folder = cp.parent / json.loads(cp.read_text())["attempt_id"]; attempts[cp.parent.name] = folder
        for line in (folder / "raw.jsonl").open():
            if not line.strip() or f'"{ORIG}"' not in line:
                continue
            r = json.loads(line)
            if r["method_id"] != ORIG:
                continue
            res = r["result"]
            rows[(OURS, r["dialog_id"], r["turn_id"])] = {
                "method_id": OURS, "family": "ours", "async_read": True, "image_id": r["image_id"], "dialog_id": r["dialog_id"],
                "turn_id": r["turn_id"], "score": r["score"], "prediction": r["prediction"], "ttft_ms": r["ttft_ms"],
                "request_e2e_ms": r["request_e2e_ms"], "ssd_read_bytes": r["ssd_read_bytes"], "ssd_preads": r["ssd_preads"],
                "probe_read_bytes": 0, "k_read_bytes": 0, "v_read_bytes": 0, "packed_read_bytes": r["packed_read_bytes"],
                "read_blocked_ms": r["read_blocked_ms"], "global_dialog_ordinal": r["global_dialog_ordinal"], "N_content": r["N_content"],
                "tokens": res["generated_token_ids"], "prefill_ms": res.get("prefill_ms"), "hook_ms": res.get("hook_ms"),
                "io_ms": (res.get("io") or {}).get("ms"), "capture_ms": None}
    return rows, attempts


def merged(phase):
    old, old_att = R.load_phase(OLD, phase)
    new, new_att = new_rows(phase)
    before = {k: v for k, v in old.items() if k[0] == OURS}
    assert set(before) == set(new), (phase, len(before), len(new))
    rows = {k: v for k, v in old.items() if k[0] != OURS}; rows.update(new)
    return rows, before, old_att, new_att


def setup_ms(folder, kind):
    per = json.loads((folder / f"{kind}_persistence.json").read_text()); act = json.loads((folder / f"{kind}_activation.json").read_text())
    return per, act, per["service_persistence_ms"] + act["outer_wall_ms"]


def sessions(rows, methods, names, old_att, new_att):
    kinds = {"recompute": None, "fullload_async": ("old", "raster"), "impress_official_async": ("old", "impress_official"), OURS: ("new", NEW_KIND)}
    img_of, first = {}, {}
    for (m, d, t), r in sorted(rows.items(), key=lambda kv: kv[1]["global_dialog_ordinal"]):
        img_of[d] = r["image_id"]; first.setdefault(r["image_id"], d)
    out = []
    for m in methods:
        shared, alone, req = [], [], []
        for d in sorted(img_of):
            e2e = sum(rows[(m, d, t)]["request_e2e_ms"] for t in (1, 2, 3)); req.append(e2e)
            k = kinds[m]
            su = 0.0 if k is None else setup_ms((old_att if k[0] == "old" else new_att)[img_of[d]], k[1])[2]
            shared.append(e2e + (su if first[img_of[d]] == d else 0.0)); alone.append(e2e + su)
        s1, s2 = stats(shared), stats(alone)
        out.append({"method": m, "name": names[m], "request_only_mean_ms": float(np.mean(req)), "shared_mean_ms": s1[0], "shared_p50_ms": s1[1],
                    "shared_p95_ms": s1[2], "standalone_mean_ms": s2[0], "standalone_p50_ms": s2[1], "standalone_p95_ms": s2[2],
                    "source_run": "2026-11-20" if m == OURS else "2026-11-17"})
    return out


def ours_storage(att, kind):
    rs = []
    for folder in att.values():
        per, act, _ = setup_ms(folder, kind); files = per["files"]
        rs.append({"mb": per["file_logical_bytes"] / 1e6, "kv_mb": files["kv_packed.bin"]["size"] / 1e6,
                   "other_mb": (per["file_logical_bytes"] - files["kv_packed.bin"]["size"]) / 1e6,
                   "meta_bytes": files["meta.json"]["size"] + files.get("visionzip_layout.pt", {}).get("size", 0),
                   "persist_ms": per["service_persistence_ms"], "writer_ms": per["timing_ms"]["persist_ms"],
                   "hash_ms": per["completion_full_hash_ms"], "act_ms": act["outer_wall_ms"]})
    out = {k: float(np.mean([x[k] for x in rs])) for k in rs[0]}
    out["mb_max"] = max(x["mb"] for x in rs); out["images"] = len(rs)
    return out


def main():
    names = {"recompute": "ReComp", "fullload_async": "FullLoad", "impress_official_async": "IMPRESS", OURS: "Ours"}
    methods = ["recompute", "fullload_async", "impress_official_async", OURS]
    rows, before, old_att, new_att = merged("main")
    lat = R.arm_table(None, rows, methods, names)
    old_lat = R.arm_table(None, {**{k: v for k, v in rows.items() if k[0] != OURS}, **before}, methods, names)[OURS]
    pairs = []
    for a, b in ((OURS, "impress_official_async"), (OURS, "fullload_async"), (OURS, "recompute"), ("impress_official_async", "fullload_async")):
        acc = R.paired(rows, a, b, "score"); tt = R.paired(rows, a, b, "ttft_ms")
        pairs.append({"a": names[a], "b": names[b], "acc_delta_pp": 100 * acc["mean"], "acc_ci_low_pp": 100 * acc["ci_low"], "acc_ci_high_pp": 100 * acc["ci_high"],
                      "ttft_delta_ms": tt["mean"], "ttft_ci_low_ms": tt["ci_low"], "ttft_ci_high_ms": tt["ci_high"], "ttft_cross_run": a == OURS, "n": acc["n"]})
    drows, dbefore, dold_att, dnew_att = merged("dev")
    dlat = R.arm_table(None, drows, methods, names)
    dpairs = []
    for a, b in ((OURS, "impress_official_async"), (OURS, "fullload_async"), (OURS, "recompute"), ("impress_official_async", "fullload_async")):
        acc = R.paired(drows, a, b, "score"); tt = R.paired(drows, a, b, "ttft_ms")
        dpairs.append({"a": names[a], "b": names[b], "acc_delta_pp": 100 * acc["mean"], "acc_ci_low_pp": 100 * acc["ci_low"], "acc_ci_high_pp": 100 * acc["ci_high"],
                       "ttft_delta_ms": tt["mean"], "ttft_ci_low_ms": tt["ci_low"], "ttft_ci_high_ms": tt["ci_high"], "ttft_cross_run": a == OURS, "n": acc["n"]})
    sess = sessions(rows, methods, names, old_att, new_att)
    st_new, st_old = ours_storage(new_att, NEW_KIND), ours_storage(old_att, OLD_KIND)
    old_storage = {r["store_kind"]: r for r in __import__("csv").DictReader(open(ROOT / "results/mt_gqa_impress_code_20261117T185832Z/main/storage.csv"))}
    same_run = json.loads((ROOT / "results/mt_gqa_ours_order_20261120T052725Z/summary.json").read_text())["main"]["arms"]

    def tbl(L_, lat_):
        return md_table(["방법", "정답률", "TTFT 평균(ms)", "p50", "p95", "FullLoad 대비", "ReComp 대비", "SSD MB/followup"],
                        [[lat_[m]["name"] + (" *" if m == OURS else ""), f"{lat_[m]['followup_acc_pct']:.2f}%", f"{lat_[m]['ttft_mean_ms']:.1f}",
                          f"{lat_[m]['ttft_p50_ms']:.1f}", f"{lat_[m]['ttft_p95_ms']:.1f}", f"{lat_[m]['ttft_vs_fullload']:.3f}×" + (" †" if m == OURS else ""),
                          f"{lat_[m]['ttft_vs_recomp']:.3f}×" + (" †" if m == OURS else ""), f"{lat_[m]['ssd_mb_per_followup']:.1f}"] for m in methods])

    def ptbl(ps):
        return md_table(["비교", "정답률 차이 (pp)", "TTFT 차이 (ms)"],
                        [[f"{p['a']} − {p['b']}", f"{p['acc_delta_pp']:+.2f} [{p['acc_ci_low_pp']:+.2f}, {p['acc_ci_high_pp']:+.2f}]",
                          f"{p['ttft_delta_ms']:+.1f} [{p['ttft_ci_low_ms']:+.1f}, {p['ttft_ci_high_ms']:+.1f}]" + (" †" if p["ttft_cross_run"] else "")] for p in ps])

    L = ["# 실험 A — LLaVA-NeXT × MT-GQA: ReComp / FullLoad / IMPRESS / Ours (Ours = 원래 순서 저장으로 교체)\n",
         "`results/mt_gqa_impress_code_20261117T185832Z/report.md`의 표에서 **Ours 행만** 바꾼 것이다. 원본 보고서와 실행 디렉터리는 수정하지 않았다.\n",
         "- ReComp, FullLoad, IMPRESS: 2026-11-17 실행(`runs/mt_gqa_impress_code_20261117T185832Z`)의 요청별 결과 그대로.\n"
         "- Ours: 2026-11-20 실행(`runs/mt_gqa_ours_order_20261120T052725Z`)의 Ours(원래 순서) 요청별 결과 — 중요도 순 재배열 없이 같은 토큰 집합을 원래 토큰 순서로 저장·적재. "
         "같은 대화·턴·프로토콜이다(PYR-beta-raw, KV 25%, packed, 비동기 읽기, cold SSD).\n"
         "- 두 실행을 섞어도 되는 근거: 11/20 실행의 ReComp·FullLoad·Ours(재배열) 생성 토큰열이 11/17 실행과 398장 전체에서 각 12,183건 모두 같았다. "
         "따라서 **정답률과 정답률 차이는 같은 실행에서 잰 것과 같다.**\n"
         "- **TTFT는 다르다**: Ours의 TTFT는 11/20에, 나머지는 11/17에 잰 값이다. † 표시가 서로 다른 실행의 값을 섞은 비율·차이다. "
         f"참고로 11/20 실행 안에서 잰 같은 실행 비율은 Ours/FullLoad {same_run[ORIG]['ttft_mean_ms'] / same_run['fullload_async']['ttft_mean_ms']:.3f}×, "
         f"Ours/ReComp {same_run[ORIG]['ttft_mean_ms'] / same_run['recompute']['ttft_mean_ms']:.3f}×다(그 실행의 FullLoad {same_run['fullload_async']['ttft_mean_ms']:.1f} ms, ReComp {same_run['recompute']['ttft_mean_ms']:.1f} ms).\n",
         "## 본실행 (398장, T2–T3 followup)\n",
         f"요청 수: arm당 12,183, followup {lat[OURS]['followups']:,}. * = 교체된 행.\n", tbl(None, lat),
         f"교체 전 Ours(재배열) 행: 정답률 {old_lat['followup_acc_pct']:.2f}%, TTFT {old_lat['ttft_mean_ms']:.1f} / {old_lat['ttft_p50_ms']:.1f} / {old_lat['ttft_p95_ms']:.1f} ms, "
         f"FullLoad 대비 {old_lat['ttft_vs_fullload']:.3f}×, ReComp 대비 {old_lat['ttft_vs_recomp']:.3f}×, SSD {old_lat['ssd_mb_per_followup']:.1f} MB/followup.\n",
         "참고(턴별 정답률, 턴 1 TTFT, 읽기 호출 수, GPU가 읽기를 기다린 시간):\n",
         md_table(["방법", "T1 정답률", "T2", "T3", "T1 TTFT 평균(ms)", "followup E2E 평균(ms)", "read 호출 수", "GPU 읽기 대기(ms)"],
                  [[lat[m]["name"] + (" *" if m == OURS else ""), f"{lat[m]['t1_acc_pct']:.2f}%", f"{lat[m]['t2_acc_pct']:.2f}%", f"{lat[m]['t3_acc_pct']:.2f}%",
                    f"{lat[m]['t1_ttft_mean_ms']:.1f}", f"{lat[m]['e2e_mean_ms']:.1f}", f"{lat[m]['read_calls_mean']:.1f}",
                    "—" if lat[m]["read_blocked_ms_mean"] is None else f"{lat[m]['read_blocked_ms_mean']:.1f}"] for m in methods]),
         "### 이미지 단위 paired bootstrap 95% CI (10,000회, seed 1234, followup)\n", ptbl(pairs),
         "IMPRESS − FullLoad 행은 원본과 같다(두 arm 모두 11/17 값).\n",
         "## DEV 40장 (본조건 4 arm, T2–T3 followup)\n",
         "DEV 40장도 같은 방식으로 Ours 행만 11/20 실행의 DEV 파일럿 값으로 바꿨다(11/20 DEV 파일럿의 ReComp·FullLoad·Ours(재배열) 토큰도 11/17과 1,134건씩 모두 같았다).\n",
         tbl(None, dlat), ptbl(dpairs),
         "## 저장·턴 1 비용 (Ours 행)\n",
         md_table(["방법", "store 크기 평균(MB)", "최대(MB)", "KV payload", "기타(meta·system KV 등)", "메타데이터(meta.json + 점수·순열 파일, KB)",
                   "저장 전체(ms)", "writer(ms)", "무결성 hash(ms)", "activation(ms)"],
                  [["Ours *", f"{st_new['mb']:.1f}", f"{st_new['mb_max']:.1f}", f"{st_new['kv_mb']:.1f}", f"{st_new['other_mb']:.1f}", f"{st_new['meta_bytes'] / 1e3:.1f}",
                    f"{st_new['persist_ms']:.0f}", f"{st_new['writer_ms']:.0f}", f"{st_new['hash_ms']:.0f}", f"{st_new['act_ms']:.0f}"],
                   ["(교체 전) Ours(재배열)", f"{st_old['mb']:.1f}", f"{st_old['mb_max']:.1f}", f"{st_old['kv_mb']:.1f}", f"{st_old['other_mb']:.1f}", f"{st_old['meta_bytes'] / 1e3:.1f}",
                    f"{st_old['persist_ms']:.0f}", f"{st_old['writer_ms']:.0f}", f"{st_old['hash_ms']:.0f}", f"{st_old['act_ms']:.0f}"]]),
         "FullLoad·IMPRESS 행과 IMPRESS의 턴 1 추가 비용은 원본 보고서 그대로다. 두 Ours 행은 서로 다른 날 다른 턴 1 요청에서 만든 store의 값이다.\n",
         "### 3턴 세션 시간 (원본과 같은 정의)\n",
         "세션 = 한 대화의 3턴 요청 E2E 합 + store 준비(저장 전체 + activation). shared: 이미지의 첫 대화에만 준비 시간을 더함. standalone: 모든 대화에 더함(DERIVED). "
         "Ours 행은 11/20 실행의 요청 E2E와 store 준비 시간이다.\n",
         md_table(["방법", "요청만 평균(ms)", "shared 평균 / p50 / p95 (ms)", "standalone 평균 / p50 / p95 (ms)"],
                  [[s["name"] + (" *" if s["method"] == OURS else ""), f"{s['request_only_mean_ms']:.1f}",
                    f"{s['shared_mean_ms']:.1f} / {s['shared_p50_ms']:.1f} / {s['shared_p95_ms']:.1f}",
                    f"{s['standalone_mean_ms']:.1f} / {s['standalone_p50_ms']:.1f} / {s['standalone_p95_ms']:.1f}"] for s in sess]),
         "## 바뀌지 않은 부분\n",
         "검증 단계, IMPRESS 내부 통계, 공개 구현과의 대응, 원본과 다른 점, 기존 산출물 보호, 실행 기록은 원본 보고서(`results/mt_gqa_impress_code_20261117T185832Z/report.md`)를 그대로 본다. "
         "Ours(원래 순서)의 검증(저장 내용 일치, 동기 = 비동기, 재현성 등)과 Ours(재배열)과의 비교·판정은 `results/mt_gqa_ours_order_20261120T052725Z/report.md`에 있다.\n"]
    (OUT / "main").mkdir(parents=True, exist_ok=True); (OUT / "dev").mkdir(exist_ok=True)
    (OUT / "report.md").write_text("\n".join(L))
    csv_write(OUT / "main" / "arms.csv", [dict(v, source_run="2026-11-20" if m == OURS else "2026-11-17") for m, v in lat.items()])
    csv_write(OUT / "main" / "paired.csv", pairs); csv_write(OUT / "main" / "sessions.csv", sess)
    csv_write(OUT / "dev" / "arms.csv", list(dlat.values())); csv_write(OUT / "dev" / "paired.csv", dpairs)
    write_json(OUT / "summary.json", {"sources": {"ReComp/FullLoad/IMPRESS": str(OLD.relative_to(ROOT)), "Ours": f"{NEW.relative_to(ROOT)} ({ORIG})"},
                                       "main": lat, "paired": pairs, "dev": dlat, "dev_paired": dpairs, "ours_storage": st_new, "sessions": sess})
    print(OUT / "report.md")


if __name__ == "__main__":
    main()

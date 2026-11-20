#!/usr/bin/env python3
"""Report for the MT-GQA run with the IMPRESS arm re-implemented from the public
code (contract docs/mt_gqa_impress_official_contract.md).

Reads committed rows and per-image evidence only; measures nothing.  Writes
``results/<RUN>/report.md`` plus the CSV/JSON tables it is built from.
Values that were not measured are written as "미실행".

  python scripts/192_report_mt_gqa_impress_official.py --run runs/mt_gqa_impress_official_<UTC>
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


def _load(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load("189_report_mt_gqa_async.py", "_mtgqa_async_report")      # table / csv / json helpers, storage and session tables
write_json, csv_write, md_table, stats, by_method, gate_status = R.write_json, R.csv_write, R.md_table, R.stats, R.by_method, R.gate_status


def slim(r):
    """Keep only what the report needs from one raw row."""
    res = r["result"]
    keep = {k: r[k] for k in ("method_id", "family", "async_read", "image_id", "dialog_id", "turn_id", "score", "prediction",
                              "ttft_ms", "request_e2e_ms", "ssd_read_bytes", "ssd_preads", "probe_read_bytes", "k_read_bytes",
                              "v_read_bytes", "packed_read_bytes", "read_blocked_ms", "global_dialog_ordinal", "N_content")}
    keep["tokens"] = res["generated_token_ids"]
    keep["prefill_ms"] = res.get("prefill_ms")
    keep["hook_ms"] = res.get("hook_ms")
    keep["io_ms"] = (res.get("io") or {}).get("ms")
    keep["capture_ms"] = res.get("turn1_attention_capture_ms")
    vn = r["geometry"]["v_token_num"]
    if r["family"] in ("impress", "impress_official") and r["turn_id"] > 1:
        keep["layer_log"] = [(x["mode"], x["sim"], x["k_chunks"], x["v_chunks"]) for x in res["layer_log"]]
        keep["n_chunks"] = (vn + 63) // 64
        keep["probe_hook_blocked_ms"] = res.get("probe_hook_blocked_ms")
        keep["prefetch_ready_on_arrival"] = res.get("prefetch_ready_on_arrival")
    if r["family"] == "impress" and r["turn_id"] > 1:
        keep["raster"] = res["unreordered_chunks"]["k_chunks"]
    if r["family"] == "impress_official" and r["turn_id"] > 1:
        keep["raster"] = res["raster_chunks"]["chunks"]
        keep["importance_only"] = res["importance_only_chunks"]["chunks"]
        keep["diag"] = [(d["prefix_share"], d["order_overlap_head"], d["order_overlap_selected"]) for d in res["layer_diagnostics"]]
        keep["prefix_row_bytes"] = 12 * vn * r["geometry"]["num_layers"]
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


def arm_table(H, rows, methods, names):
    lat = {}
    for m in methods:
        f = by_method(rows, m)
        t = stats([r["ttft_ms"] for r in f]); e = stats([r["request_e2e_ms"] for r in f])
        t1 = [r for r in by_method(rows, m, follow=False) if r["turn_id"] == 1]
        lat[m] = {"method": m, "name": names[m], "followups": len(f), "followup_acc_pct": 100 * float(np.mean([r["score"] for r in f])),
                  "t1_acc_pct": 100 * float(np.mean([r["score"] for r in t1])),
                  "t2_acc_pct": 100 * float(np.mean([r["score"] for r in f if r["turn_id"] == 2])),
                  "t3_acc_pct": 100 * float(np.mean([r["score"] for r in f if r["turn_id"] == 3])),
                  "ttft_mean_ms": t[0], "ttft_p50_ms": t[1], "ttft_p95_ms": t[2], "e2e_mean_ms": e[0],
                  "t1_ttft_mean_ms": float(np.mean([r["ttft_ms"] for r in t1])),
                  "ssd_mb_per_followup": float(np.mean([r["ssd_read_bytes"] for r in f])) / 1e6,
                  "read_calls_mean": float(np.mean([r["ssd_preads"] for r in f])),
                  "probe_mb": float(np.mean([r["probe_read_bytes"] for r in f])) / 1e6, "k_mb": float(np.mean([r["k_read_bytes"] for r in f])) / 1e6,
                  "v_mb": float(np.mean([r["v_read_bytes"] for r in f])) / 1e6,
                  "prefix_row_mb": (float(np.mean([r["prefix_row_bytes"] for r in f])) / 1e6 if "prefix_row_bytes" in f[0] else None),
                  "read_blocked_ms_mean": (float(np.mean([r["read_blocked_ms"] for r in f])) if f[0]["read_blocked_ms"] is not None else None),
                  "hook_ms_mean": float(np.mean([r["hook_ms"] for r in f if r["hook_ms"] is not None] or [0.0])),
                  "prefill_ms_mean": float(np.mean([r["prefill_ms"] for r in f]))}
    for m in methods:
        lat[m]["ttft_vs_fullload"] = lat[m]["ttft_mean_ms"] / lat["fullload_async"]["ttft_mean_ms"]
        lat[m]["ttft_vs_recomp"] = lat[m]["ttft_mean_ms"] / lat["recompute"]["ttft_mean_ms"]
    return lat


def selector_stats(rows, method):
    """Internal statistics of an IMPRESS arm (new or previous implementation)."""
    f = by_method(rows, method)
    L = len(f[0]["layer_log"])
    fb = np.asarray([[x[0] == "fallback" for x in r["layer_log"]] for r in f], dtype=float)
    sim = np.asarray([[x[1] for x in r["layer_log"]] for r in f], dtype=float)
    kch = np.asarray([[x[2] for x in r["layer_log"]] for r in f], dtype=float)
    vch = np.asarray([[x[3] for x in r["layer_log"]] for r in f], dtype=float)
    nc = np.asarray([r["n_chunks"] for r in f], dtype=float)[:, None]
    raster = np.asarray([r["raster"] for r in f], dtype=float)
    cons = fb == 0
    out = {"method": method, "followups": len(f), "layers": L, "fallback_rate": float(fb.mean()), "mean_jaccard": float(sim.mean()),
           "k_chunk_fraction": float((kch / nc).mean()), "v_chunk_fraction": float((vch / nc).mean()),
           "kv_chunk_fraction": float(((kch + vch) / (2 * nc)).mean()),
           "consensus_kv_chunk_fraction": float((((kch + vch) / (2 * nc))[cons]).mean()) if cons.any() else None,
           "raster_chunk_fraction_computed": float((raster / nc).mean()),
           "consensus_raster_chunk_fraction_computed": float(((raster / nc)[cons]).mean()) if cons.any() else None,
           "read_blocked_ms": float(np.mean([r["read_blocked_ms"] for r in f])),
           "probe_hook_blocked_ms": float(np.mean([r["probe_hook_blocked_ms"] for r in f])),
           "prefetch_ready_on_arrival": float(np.mean([r["prefetch_ready_on_arrival"] for r in f])),
           "requests_with_any_fallback": float((fb.sum(axis=1) > 0).mean())}
    layers = [{"layer": li, "fallback_rate": float(fb[:, li].mean()), "mean_jaccard": float(sim[:, li].mean()),
               "k_chunk_fraction": float((kch[:, li:li + 1] / nc).mean()), "v_chunk_fraction": float((vch[:, li:li + 1] / nc).mean()),
               "raster_chunk_fraction_computed": float((raster[:, li:li + 1] / nc).mean())} for li in range(L)]
    if "diag" in f[0]:
        share = np.asarray([[d[0] for d in r["diag"]] for r in f], dtype=float)
        head = np.asarray([[d[1] for d in r["diag"]] for r in f], dtype=float)
        chosen = np.asarray([[np.nan if d[2] is None else d[2] for d in r["diag"]] for r in f], dtype=float)
        only = np.asarray([r["importance_only"] for r in f], dtype=float)
        out.update(prefix_share=float(share.mean()), order_overlap_head=float(head.mean()),
                   order_overlap_selected=float(np.nanmean(chosen)) if cons.any() else None,
                   importance_only_chunk_fraction_computed=float((only / nc).mean()),
                   consensus_importance_only_chunk_fraction_computed=float(((only / nc)[cons]).mean()) if cons.any() else None)
        for li, x in enumerate(layers):
            col = chosen[:, li]
            x.update(prefix_share=float(share[:, li].mean()), order_overlap_head=float(head[:, li].mean()),
                     order_overlap_selected=(float(np.nanmean(col)) if np.isfinite(col).any() else None),
                     importance_only_chunk_fraction_computed=float((only[:, li:li + 1] / nc).mean()))
    return out, layers


def official_turn1(H, attempts, rows):
    """Turn-1 side of the new IMPRESS arm, per image, from the persistence receipts."""
    per_image = []
    for img, folder in attempts.items():
        per = json.loads((folder / f"{H.NEW_KIND}_persistence.json").read_text())
        t1 = per["official_turn1"]; lay = t1["layout"]; pk = lay["per_kind"]
        src = per["source"]
        row = rows[(src["method_id"], src["dialog_id"], 1)]
        others = [rows[(m, src["dialog_id"], 1)]["ttft_ms"] for (m, d, t) in rows if d == src["dialog_id"] and t == 1 and m == "recompute"]
        per_image.append({"image_id": img, "capture_ms": t1["capture_ms"], "materialize_ms": t1["materialize_ms"], "stats_save_ms": t1["stats_save_ms"],
                          "order_ms": lay["order_ms"], "layout_ms": t1["layout_ms"],
                          "rewrite_k_ms": sum(pk["k"][x] for x in ("read_ms", "permute_ms", "write_ms")),
                          "rewrite_v_ms": sum(pk["v"][x] for x in ("read_ms", "permute_ms", "write_ms")),
                          "rewrite_probe_ms": sum(pk["probe_k"][x] for x in ("read_ms", "permute_ms", "write_ms")),
                          "write_only_ms": sum(pk[n]["write_ms"] for n in ("k", "v", "probe_k")),
                          "sidecar_ms": lay["separator_sidecar_ms"], "trailer_bytes": t1["trailer_bytes_per_layer"] * len(t1["trailer_sha256_per_layer"]),
                          "suffix_tokens": t1["suffix_tokens"], "capture_rows": t1["capture_rows"],
                          "capture_request_t1_ttft_ms": row["ttft_ms"], "recompute_t1_ttft_ms_same_dialogue": others[0] if others else None})
    keys = [k for k in per_image[0] if k != "image_id"]
    return {k: float(np.mean([x[k] for x in per_image if x[k] is not None])) for k in keys}, per_image


def reproduce_main(H, run, rows, methods):
    """Post-hoc: this run's tokens against the previous main run, every request."""
    old = {}
    for cp in sorted((H.PRIOR_ASYNC_RUN / "main" / "images").glob("*/COMMITTED.json")):
        doc = json.loads(cp.read_text())
        with (cp.parent / doc["attempt_id"] / "raw.jsonl").open() as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r["method_id"] in methods:
                        old[(r["method_id"], r["dialog_id"], r["turn_id"])] = (r["result"]["generated_token_ids"], r["score"])
    out = []
    for m in methods:
        keys = [k for k in rows if k[0] == m]
        miss = sum(k not in old for k in keys)
        diff = sum(k in old and rows[k]["tokens"] != old[k][0] for k in keys)
        out.append({"method": m, "name": H.ARMS[m]["display"], "requests": len(keys), "missing_in_previous": miss, "token_mismatch": diff,
                    "previous_acc_followup_pct": 100 * float(np.mean([old[k][1] for k in keys if k in old and k[2] > 1]))})
    return out


MAPPING = [
    ("`sele_tokenid` / `llama_sele_tokenid` (probe head의 attention: 전체 행 × prefix key + suffix key, causal softmax)",
     "`causal_attention`, `column_importance`; 요청 때는 `suffix_row_importance_from_qk` + 저장된 prefix 행 기여분; 턴 1 캡처 `Turn1AttentionStats`"),
    ("`_sele_important_tokens` (열 합, head별 상위 `ceil(n·percent/100)`, 유사도 판정 `sim >= sim_thred`, 득표 합의, fallback = prefix 전체)",
     "`keep_count`, `head_topk`, `vote_consensus`, `select_tokens`"),
    ("`cal_jaccard_sim` (head 쌍의 Jaccard 평균)", "`mean_pairwise_jaccard`"),
    ("`mha_with_filled_selected_prefixkv` (선택 안 된 토큰의 열에 `finfo.min`)",
     "`OfficialImpressSelector._run`의 keep mask → 기존 `bias_from_keep` / `BIAS` (additive mask)"),
    ("`mha_with_sele_percent_prefixkv(generate_mapping_list=True)` (모든 head·모든 행 열 합의 내림차순, `reverse_indices`)",
     "`Turn1AttentionStats`의 재배치 통계, `descending_order`, `reverse_index`, `layer_order`"),
    ("`my_pcache_fast.py` `PrefixKVLayer` (`KV_Division`: K·V 분리, 64 토큰 chunk, `head_token`, `reorder`)",
     "`write_official_layout`(K·V·probe 파일을 같은 순서로), `read_probe_block`, 기존 `ChunkReader.read_chunks` / `read_full`"),
    ("`flex_opt_fast_pcache.py`의 sele-load 분기 (probe key 읽기 → 선택 → 선택된 K·V 읽기 → 계산)", "`OfficialImpressSelector`(layer pre-hook), `request_official`"),
]

DIFFERENCES = [
    ("선택 대상", "prefix 토큰 전부", "이미지 토큰만. 시스템 프롬프트와 separator는 항상 유지, 예산 밖", "정답률에 유리 / 읽기량에 소폭 불리"),
    ("prefix 행 기여분", "정확도 경로: 매 요청 다시 계산(선택된 KV만 보는 상태)", "턴 1(mask 없음)의 값을 저장해 요청 때 더함", "유리 (요청 때 prefix 행 연산 없음)"),
    ("캐시 계층", "GPU·DRAM·SSD 다층, CKLFU", "없음. 모든 방법이 cold SSD", "불리 (모든 방법 공통 조건)"),
    ("다음 layer probe 선행 읽기", "IMPRESS 설정은 `--no-prefetch`", "켬", "유리"),
    ("separator의 저장 위치", "해당 없음", "저장 순서 맨 앞에 모음(항상 읽으므로 chunk 1개에 담김)", "유리"),
    ("득표 동점", "순서 미정의(unstable argsort)", "저장 위치가 앞선 토큰 우선", "유리 또는 중립"),
    ("재배치 통계의 추론 상태", "head별 25% mask 추론(`--full-load --sele-inf`)", "mask 없는 턴 1 추론", "중립"),
    ("재배치 적용", "`key[reorder]` (token→rank 인덱스로 gather; 내용이 어긋남)", "위치 r = rank r 토큰, K·V·probe 정렬 유지", "중립 (접근 위치는 같다)"),
    ("중요도 수치 정밀도", "float16 softmax, float16 합", "BF16 확률 + float32 합 / float32 softmax", "중립"),
    ("Fallback 읽기 단위", "chunk 파일별", "layer의 K 파일 1회, V 파일 1회 pread", "유리 (호출 수 적음, 바이트 같음)"),
    ("선택 chunk 읽기 단위", "chunk 파일별", "인접한 chunk는 pread 하나로 합침", "유리"),
    ("prefix 행 기여분 읽기", "없음", "probe sidecar 끝 12·N bytes/layer를 같은 read로", "불리 (probe 읽기 +1.6%)"),
    ("mask", "정확도 경로만(성능 경로는 0으로 채운 K/V, mask 없음, 답을 내지 않음)", "mask", "중립"),
    ("모델·생성", "OPT(FP16), 1토큰 logits", "LLaVA-NeXT(Vicuna, RoPE, NF4/BF16), greedy 최대 16토큰, 디코딩 중 선택 고정", "중립"),
]


def pct(x, digits=1):
    return "—" if x is None else f"{100 * x:.{digits}f}%"


def internal_block(L, title, st, layers):
    L.append(f"### {title} — followup {st['followups']:,}건\n")
    L.append(md_table(["항목", "값"], [
        ["fallback 비율 (layer·요청 평균)", pct(st["fallback_rate"])],
        ["fallback layer가 하나라도 있는 요청의 비율", pct(st["requests_with_any_fallback"])],
        ["평균 Jaccard (probe head 3개 쌍 평균)", f"{st['mean_jaccard']:.4f}"],
        ["K chunk 비율 (실측)", pct(st["k_chunk_fraction"])],
        ["V chunk 비율 (실측)", pct(st["v_chunk_fraction"])],
        ["합의 layer만: K·V chunk 비율 (실측)", pct(st["consensus_kv_chunk_fraction"])],
        ["중요도 합에서 prefix 행이 차지하는 비중", pct(st.get("prefix_share"))],
        ["재배치 순서 상위 25%와 head별 상위 k의 겹침 (모든 layer)", pct(st.get("order_overlap_head"))],
        ["재배치 순서 상위 25%와 합의 선택의 겹침 (합의 layer만)", pct(st.get("order_overlap_selected"))],
        ["raster 순서였다면 K·V chunk 비율 (**계산값**, 읽지 않음) — 전체 / 합의 layer만",
         f"{pct(st['raster_chunk_fraction_computed'])} / {pct(st['consensus_raster_chunk_fraction_computed'])}"],
        ["separator도 중요도 순서에 놓았다면 K·V chunk 비율 (**계산값**, 읽지 않음) — 전체 / 합의 layer만",
         f"{pct(st.get('importance_only_chunk_fraction_computed'))} / {pct(st.get('consensus_importance_only_chunk_fraction_computed'))}"],
        ["GPU가 읽기를 기다린 시간 (요청 평균)", f"{st['read_blocked_ms']:.1f} ms"],
        ["그중 probe sidecar를 얻느라 hook이 막힌 시간", f"{st['probe_hook_blocked_ms']:.1f} ms"],
        ["선행 읽기한 probe sidecar 31개 중 도착 시 이미 읽혀 있던 수 (요청 평균)", f"{st['prefetch_ready_on_arrival']:.1f}"],
    ], ["---", "---:"]))
    L.append("chunk 비율은 layer당 읽은 chunk 수 / layer chunk 수의 평균이다(probe sidecar 제외; fallback layer는 100%). "
             "계산값은 요청의 layer별 keep mask(합의: 선택된 토큰 + separator)를 다른 저장 순서의 위치로 옮겨 64 토큰 chunk로 센 것이며 실제로 읽은 값이 아니다.\n")
    L.append("layer별:\n")
    L.append(md_table(["layer", "fallback", "Jaccard", "K·V chunk 실측", "prefix 행 비중", "상위 25% 겹침(head별)", "상위 25% 겹침(합의 선택)", "raster 계산값"],
                      [[x["layer"], pct(x["fallback_rate"]), f"{x['mean_jaccard']:.3f}", pct(x["k_chunk_fraction"]), pct(x.get("prefix_share")),
                        pct(x.get("order_overlap_head")), pct(x.get("order_overlap_selected")), pct(x["raster_chunk_fraction_computed"])] for x in layers]))


def build(run):
    H = _load("190_eval_mt_gqa_impress_official.py", "_mtgqa_impress_official_harness_report")
    res = ROOT / "results" / run.name
    res.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / "config.json").read_text())
    env = config["environment"]; off = config["impress_official"]; checkout = config["official_checkout"]
    names = {m: H.ARMS[m]["display"] for m in H.ARMS}
    names["impress_async"] = "IMPRESS (이전 구현)"
    L = []
    L.append("# MT-GQA 다중 턴 — 공개 IMPRESS 구현의 로직을 이식한 IMPRESS arm (ReComp / FullLoad / IMPRESS / Ours)\n")
    L.append("측정값만 적는다. 측정하지 않은 값은 \"미실행\"이다. 계약: `docs/mt_gqa_impress_official_contract.md` (사본 `CONTRACT.md`).\n")
    L.append("## 설정\n")
    L.append(f"- 실행: `runs/{run.name}` (요청 단위 원본), 이 디렉터리(`results/{run.name}`)는 집계와 보고서.\n"
             f"- 환경: Python {env['python'].split()[0]}, torch {env['torch']}, transformers {env['transformers']} (`{env['executable']}`). 직전 본실행과 같은 환경이다.\n"
             f"- 공개 구현: {checkout['repository']}, 커밋 `{checkout['head']}` (체크아웃 `{checkout['path']}`, 레포 밖, 읽기 전용 참조; 코드는 복사하지 않았다).\n"
             f"- 모델 `{config['model']}` revision `{config['revision'][:12]}…`, NF4/BF16/eager, FP16 SSD KV, greedy max 16, batch 1, CPU threads 2.\n"
             "- 모집단·절차(직전 본실행과 동일): MT-GQA-reconstructed 398 images / 4,061 dialogues / 3턴, method-local generated history, strict normalized EM, "
             "대화 단위 arm 순서 회전, 턴 1은 모든 방법이 픽셀부터 계산, 저장 KV 읽기는 요청 직전에 page cache를 비운 cold 조건, 저장 KV 방법은 비동기 읽기.\n"
             f"- IMPRESS(새 구현, `mmimpress/impress_official.py`): probe head {off['probe_head_ids']}, 유지 25%, 유사도 임계값 {off['sim_threshold']}(이상이면 합의), "
             f"chunk {off['chunk_size']} 토큰. 중요도 = 턴 1에서 저장한 prefix 행 기여분 + 요청 때 계산한 suffix 행 기여분. "
             "fallback은 그 layer의 이미지 토큰 K·V 전부를 읽어 mask 없이 사용. 재배치는 첫 대화 턴 1 요청 하나의 통계(모든 head·모든 행 열 합)로 한 번. "
             "다음 layer probe sidecar 선행 읽기.\n"
             "- FullLoad, Ours, ReComp: 직전 본실행과 같은 코드·설정.\n")

    # ---- gates
    L.append("## 검증 단계\n")
    spec = [("1. 공개 구현과의 동등성", "step1_public_equivalence"), ("2. prefix 행 기여분 (gate 4장, GPU)", "step2_prefix_rows"),
            ("3. 재배치 정합성: 유지 100% = FullLoad (gate)", "step3_keep100_equals_fullload"),
            ("4. 재현성: 직전 본실행과 토큰 일치 (gate)", "step4_matches_previous_main_gate"),
            ("4. 재현성 (DEV 전 요청)", "step4_matches_previous_main_dev"), ("5. 기존 테스트", "step5_cpu_tests"),
            ("6. DEV 40장 실행·감사", "dev_execution"), ("추가. 새 arm 비동기 = 동기 (gate)", "new_arm_async_equals_sync"),
            ("gate phase 실행·감사", "gate_execution")]
    gates = {}; gate_rows = []
    for label, name in spec:
        g = gate_status(run, name); gates[name] = g
        detail = ""
        if g is not None:
            if name == "step1_public_equivalence":
                detail = f"공개 함수 호출 테스트 {len(g['public_cases'])}개 모두 실행·통과(skip 0), 신규 테스트 파일 전체 {g['ran']}개"
            elif name == "step2_prefix_rows":
                detail = "; ".join(f"{x['image_id']}: prefix 행 {'일치' if x['same_prompt']['prefix_rows_equal'] else '불일치'}, 재배치 통계 "
                                   f"{'일치' if x['same_prompt']['order_scores_equal'] else '불일치'}" for x in g["images"])
            elif name == "step5_cpu_tests":
                detail = "; ".join(f"transformers {s['transformers']}: {s['ran']}개 실행, 실패 {len(s['failed'])}개(모두 변경 전부터의 환경 실패), 새 실패 {len(s['new_failures'])}개" for s in g["suites"])
            elif "methods" in g:
                detail = "; ".join(f"{names[m['method']]}: {m['requests']}건 중 불일치 {m['mismatch_count']}건" + ("" if m["blocking"] else " (기준 외)") for m in g["methods"])
            elif name in ("step3_keep100_equals_fullload", "new_arm_async_equals_sync"):
                detail = f"{g['requests']}건 중 불일치 {g['mismatch_count']}건"
            elif name.endswith("_execution"):
                detail = f"{g['requests']}/{g['expected']}건, {g.get('images')}장"
        gate_rows.append([label, g["status"] if g else NOT_RUN, detail])
    admission = gate_status(run, "admission")
    gate_rows.append(["7. 본실행 진입", admission["status"] if admission else NOT_RUN, "1–6 모두 PASS일 때만 실행" if admission else ""])
    L.append(md_table(["단계", "결과", "내용"], gate_rows, ["---", "---", "---"]))
    summary = {"run": run.name, "environment": env, "official_checkout": checkout, "gates": {k: (v["status"] if v else None) for k, v in gates.items()},
               "admission": admission["status"] if admission else None}
    g2 = gates.get("step2_prefix_rows")
    if g2:
        d = [x["other_prompt_diagnostic"] for x in g2["images"]]
        L.append("2번의 참고값(기준 아님): 같은 이미지를 다른 질문의 요청 안에서 다시 캡처한 prefix 행 기여분과 저장값의 차이는 "
                 f"최대 절댓값 {max(x['max_abs_diff'] for x in d):.3g}, 평균 절댓값 {np.mean([x['mean_abs_diff'] for x in d]):.3g}, "
                 f"저장값이 1e-3 이상인 원소의 최대 상대 차이 {max(x['max_rel_diff_where_stored_ge_1e-3'] for x in d):.3g}; "
                 f"prefix 행만으로 뽑은 head별 상위 25% 집합의 Jaccard는 평균 {np.mean([x['prefix_only_topk_jaccard_mean'] for x in d]):.4f}, "
                 f"최소 {min(x['prefix_only_topk_jaccard_min'] for x in d):.4f} (이미지 {len(d)}장, layer × probe head 전체). "
                 f"원소 단위로 완전히 같은 이미지는 {sum(x['prefix_rows_equal'] for x in d)}장이다.\n")

    # ---- main
    main_done = (run / "main" / "validation.json").exists()
    rows = attempts = None
    if main_done:
        rows, attempts = load_phase(run, "main")
        methods = config["phases"]["main"]["methods"]
        lat = arm_table(H, rows, methods, names)
        csv_write(res / "main" / "arms.csv", list(lat.values()))
        L.append("## 본실행 (398장, T2–T3 followup)\n")
        L.append(f"요청 수: {len(rows):,} (arm당 {len(rows) // len(methods):,}, followup {lat[methods[0]]['followups']:,}). "
                 "FullLoad·ReComp 대비는 같은 실행 안의 TTFT 평균 비율이다. SSD MB는 read syscall이 실제로 돌려준 바이트의 요청 평균이다.\n")
        L.append(md_table(["방법", "정답률", "TTFT 평균(ms)", "p50", "p95", "FullLoad 대비", "ReComp 대비", "SSD MB/followup"],
                          [[lat[m]["name"], f"{lat[m]['followup_acc_pct']:.2f}%", f"{lat[m]['ttft_mean_ms']:.1f}", f"{lat[m]['ttft_p50_ms']:.1f}", f"{lat[m]['ttft_p95_ms']:.1f}",
                            f"{lat[m]['ttft_vs_fullload']:.3f}×", f"{lat[m]['ttft_vs_recomp']:.3f}×", f"{lat[m]['ssd_mb_per_followup']:.1f}"] for m in methods]))
        imp = H.NEW_ASYNC
        L.append(f"IMPRESS의 종류별 읽기량(요청 평균): probe {lat[imp]['probe_mb']:.1f} MB / K {lat[imp]['k_mb']:.1f} MB / V {lat[imp]['v_mb']:.1f} MB "
                 f"(합 {lat[imp]['ssd_mb_per_followup']:.1f} MB). probe에는 prefix 행 기여분 {lat[imp]['prefix_row_mb']:.2f} MB가 포함된다.\n")
        L.append("참고(턴별 정답률, 턴 1 TTFT, 읽기 호출 수, GPU가 읽기를 기다린 시간):\n")
        L.append(md_table(["방법", "T1 정답률", "T2", "T3", "T1 TTFT 평균(ms)", "followup E2E 평균(ms)", "read 호출 수", "GPU 읽기 대기(ms)"],
                          [[lat[m]["name"], f"{lat[m]['t1_acc_pct']:.2f}%", f"{lat[m]['t2_acc_pct']:.2f}%", f"{lat[m]['t3_acc_pct']:.2f}%",
                            f"{lat[m]['t1_ttft_mean_ms']:.1f}", f"{lat[m]['e2e_mean_ms']:.1f}", f"{lat[m]['read_calls_mean']:.1f}",
                            "—" if lat[m]["read_blocked_ms_mean"] is None else f"{lat[m]['read_blocked_ms_mean']:.1f}"] for m in methods]))
        pairs = []
        for a, b in ((H.OURS_ASYNC, imp), (H.OURS_ASYNC, "fullload_async"), (H.OURS_ASYNC, "recompute"), (imp, "fullload_async")):
            acc = paired(rows, a, b, "score"); tt = paired(rows, a, b, "ttft_ms")
            pairs.append({"a": names[a], "b": names[b], "acc_delta_pp": 100 * acc["mean"], "acc_ci_low_pp": 100 * acc["ci_low"],
                          "acc_ci_high_pp": 100 * acc["ci_high"], "ttft_delta_ms": tt["mean"], "ttft_ci_low_ms": tt["ci_low"], "ttft_ci_high_ms": tt["ci_high"],
                          "images": acc["images"], "n": acc["n"]})
        csv_write(res / "main" / "paired.csv", pairs)
        L.append("### 이미지 단위 paired bootstrap 95% CI (10,000회, seed 1234, followup)\n")
        L.append(md_table(["비교", "정답률 차이 (pp)", "TTFT 차이 (ms)"],
                          [[f"{p['a']} − {p['b']}", f"{p['acc_delta_pp']:+.2f} [{p['acc_ci_low_pp']:+.2f}, {p['acc_ci_high_pp']:+.2f}]",
                            f"{p['ttft_delta_ms']:+.1f} [{p['ttft_ci_low_ms']:+.1f}, {p['ttft_ci_high_ms']:+.1f}]"] for p in pairs]))
        summary["main"] = {"arms": lat, "paired": pairs, "requests": len(rows)}
        rep = reproduce_main(H, run, rows, ["recompute", "fullload_async", H.OURS_ASYNC])
        write_json(res / "main" / "reproduction_vs_previous_main.json", rep)
        L.append("### 사후 확인: 직전 본실행과의 토큰 일치 (398장 전체, 통과 기준 아님)\n")
        L.append(md_table(["방법", "요청 수", "직전 본실행에 없는 요청", "생성 토큰열이 다른 요청"],
                          [[x["name"], f"{x['requests']:,}", x["missing_in_previous"], x["token_mismatch"]] for x in rep]))
        summary["main"]["reproduction"] = rep
    else:
        L.append("## 본실행 (398장)\n\n미실행.\n")

    # ---- dev
    dev_done = (run / "dev" / "validation.json").exists()
    drows = dattempts = None
    if dev_done:
        drows, dattempts = load_phase(run, "dev")
        dmethods = config["phases"]["dev"]["methods"]
        dlat = arm_table(H, drows, dmethods, names)
        csv_write(res / "dev" / "arms.csv", list(dlat.values()))
        new, old = H.NEW_ASYNC, "impress_async"
        sn, _ = selector_stats(drows, new); so, _ = selector_stats(drows, old)
        L.append("## DEV 40장 — 새 구현과 이전 구현 (같은 실행, T2–T3 followup)\n")
        L.append(f"DEV 40장의 MT-GQA 대화 {len({d for (_, d, _) in drows})}개, arm당 followup {dlat[new]['followups']}건. 두 구현은 같은 실행에서 같은 이미지·대화를 받았고 "
                 "각자 자기 store(턴 1에서 생성)를 읽었다. 이전 구현은 직전 본실행의 IMPRESS arm 그대로다.\n")
        L.append(md_table(["구현", "정답률", "TTFT 평균(ms)", "p50", "p95", "SSD MB (probe / K / V)", "read 호출 수", "fallback 비율", "평균 Jaccard", "K chunk 비율", "V chunk 비율", "GPU 읽기 대기(ms)"],
                          [[label, f"{dlat[m]['followup_acc_pct']:.2f}%", f"{dlat[m]['ttft_mean_ms']:.1f}", f"{dlat[m]['ttft_p50_ms']:.1f}", f"{dlat[m]['ttft_p95_ms']:.1f}",
                            f"{dlat[m]['ssd_mb_per_followup']:.1f} ({dlat[m]['probe_mb']:.1f} / {dlat[m]['k_mb']:.1f} / {dlat[m]['v_mb']:.1f})",
                            f"{dlat[m]['read_calls_mean']:.1f}", pct(s["fallback_rate"]), f"{s['mean_jaccard']:.4f}", pct(s["k_chunk_fraction"]), pct(s["v_chunk_fraction"]),
                            f"{dlat[m]['read_blocked_ms_mean']:.1f}"] for label, m, s in (("새 구현", new, sn), ("이전 구현", old, so))]))
        dacc = paired(drows, new, old, "score"); dtt = paired(drows, new, old, "ttft_ms"); dby = paired(drows, new, old, "ssd_read_bytes")
        L.append(f"- 새 구현 − 이전 구현 (이미지 단위 paired bootstrap 95% CI, {dacc['images']}장, {dacc['n']}건): 정답률 {100 * dacc['mean']:+.2f} pp "
                 f"[{100 * dacc['ci_low']:+.2f}, {100 * dacc['ci_high']:+.2f}], TTFT {dtt['mean']:+.1f} ms [{dtt['ci_low']:+.1f}, {dtt['ci_high']:+.1f}], "
                 f"읽기량 {dby['mean'] / 1e6:+.1f} MB [{dby['ci_low'] / 1e6:+.1f}, {dby['ci_high'] / 1e6:+.1f}].\n"
                 "- 두 구현의 fallback은 뜻이 다르다. 새 구현: 그 layer의 K·V 전부를 읽고 mask 없이 사용. 이전 구현: K 전부를 읽고 head별로 25%를 골라 mask, V는 선택된 chunk만.\n"
                 "- 임계값도 다르다. 새 구현: Jaccard ≥ 0.3이면 합의. 이전 구현: Jaccard > (r/(2−r))^0.6 ≈ 0.311이면 합의.\n")
        L.append("같은 DEV 실행의 다른 arm(참고):\n")
        L.append(md_table(["방법", "정답률", "TTFT 평균(ms)", "p50", "p95", "SSD MB/followup"],
                          [[dlat[m]["name"], f"{dlat[m]['followup_acc_pct']:.2f}%", f"{dlat[m]['ttft_mean_ms']:.1f}", f"{dlat[m]['ttft_p50_ms']:.1f}",
                            f"{dlat[m]['ttft_p95_ms']:.1f}", f"{dlat[m]['ssd_mb_per_followup']:.1f}"] for m in dmethods]))
        summary["dev"] = {"arms": dlat, "new_minus_old": {"acc": dacc, "ttft_ms": dtt, "ssd_read_bytes": dby}, "new": sn, "old": so}
    else:
        L.append("## DEV 40장 — 새 구현과 이전 구현\n\n미실행.\n")

    # ---- internals
    L.append("## IMPRESS 내부 통계 (새 구현)\n")
    blocks = []
    if main_done:
        blocks.append(("본실행 398장", rows, "main"))
    if dev_done:
        blocks.append(("DEV 40장", drows, "dev"))
    if not blocks:
        L.append("미실행.\n")
    for title, rr, tag in blocks:
        st, layers = selector_stats(rr, H.NEW_ASYNC)
        csv_write(res / tag / "impress_layers.csv", layers); write_json(res / tag / "impress_stats.json", st)
        internal_block(L, title, st, layers)
        summary.setdefault("impress", {})[tag] = st

    # ---- storage / turn-1 cost
    L.append("## 저장·턴 1 비용\n")
    src = ("main", rows, attempts, config["phases"]["main"]["methods"]) if main_done else (("dev", drows, dattempts, config["phases"]["dev"]["methods"]) if dev_done else None)
    if src is None:
        L.append("미실행.\n")
    else:
        tag, rr, att, mm = src
        storage, sessions, usage = R.storage_and_sessions(H, rr, att, mm, res, tag)
        kind_name = {"raster": "FullLoad", H.NEW_KIND: "IMPRESS", "impress": "IMPRESS (이전 구현)", H.PACKED_KIND: "Ours"}
        t1, per_image = official_turn1(H, att, rr)
        csv_write(res / tag / "impress_turn1.csv", per_image)
        L.append(f"출처: {'본실행 398장' if tag == 'main' else 'DEV 40장'}. store는 이미지당 하나(첫 대화 턴 1에서 생성). 시간은 이미지 평균(ms).\n")
        L.append(md_table(["방법", "store 크기 평균(MB)", "최대(MB)", "K·V", "probe sidecar", "separator sidecar", "v_hidden", "기타(meta·system KV 등)"],
                          [[kind_name[s["store_kind"]], f"{s['mb']:.1f}", f"{s['store_mb_max']:.1f}", f"{s['kv_mb']:.1f}", f"{s['probe_mb']:.1f}",
                            f"{s['sep_mb']:.1f}", f"{s['hidden_mb']:.1f}", f"{s['other_mb']:.1f}"] for s in storage]))
        L.append(f"- IMPRESS의 probe sidecar에는 prefix 행 기여분 {t1['trailer_bytes'] / 1e6:.2f} MB(이미지 평균)가 포함된다. layer별 재배치 순서는 `meta.json`(기타)에 있다. "
                 "FullLoad store의 probe sidecar·separator sidecar·`v_hidden.pt`는 같은 writer가 쓰지만 FullLoad는 읽지 않는다. IMPRESS는 K·V와 probe sidecar만 읽는다.\n")

        def f(v):
            return "—" if v is None else f"{v:.0f}"
        L.append(md_table(["방법", "저장 전체(쓰기+무결성 hash)", "writer(쓰기+fsync)", "무결성 hash", "activation"],
                          [[kind_name[s["store_kind"]], f(s["persist_ms_mean"]), f(s["raster_writer_persist_ms_mean"] if s["raster_writer_persist_ms_mean"] is not None else s["writer_ms_mean"]),
                            f(s["hash_ms_mean"]), f(s["act_ms_mean"])] for s in storage]))
        L.append("IMPRESS의 턴 1 쪽 추가 비용(이미지당 평균, followup TTFT에는 들어 있지 않다):\n")
        L.append(md_table(["항목", "시간(ms)", "어디서"], [
            ["prefix 행 기여분 + 재배치 통계 캡처 (32 layer)", f"{t1['capture_ms']:.1f}", "턴 1 요청 안 (그 요청의 TTFT에 포함)"],
            ["캡처 결과를 host로 옮기기", f"{t1['materialize_ms']:.1f}", "요청 밖"],
            ["증거 파일(.npz) 저장", f"{t1['stats_save_ms']:.1f}", "요청 밖 (실험 기록용)"],
            ["재배치 순서 계산 (32 layer 정렬)", f"{t1['order_ms']:.1f}", "요청 밖"],
            ["재배치 쓰기 전체 (K / V / probe sidecar 읽기+치환+쓰기, meta, separator sidecar)", f"{t1['layout_ms']:.1f}", "요청 밖"],
            ["  그중 K / V / probe sidecar", f"{t1['rewrite_k_ms']:.0f} / {t1['rewrite_v_ms']:.0f} / {t1['rewrite_probe_ms']:.0f}", ""],
            ["  그중 순수 쓰기", f"{t1['write_only_ms']:.0f}", ""],
            ["  그중 separator sidecar 재작성", f"{t1['sidecar_ms']:.0f}", ""],
        ], ["---", "---:", "---"]))
        L.append(f"- 캡처가 걸린 턴 1 요청의 TTFT는 평균 {t1['capture_request_t1_ttft_ms']:.1f} ms, 같은 대화의 ReComp 턴 1은 평균 {t1['recompute_t1_ttft_ms_same_dialogue']:.1f} ms다"
                 "(캡처 요청에는 KV 캡처 등 다른 저장용 캡처도 함께 걸려 있다). 캡처된 행 수는 prefix 길이 + 그 턴 1 suffix 길이"
                 f"(suffix 평균 {t1['suffix_tokens']:.1f} 토큰)다.\n"
                 "- \"저장 전체\"는 writer + 이미지별 store 전체 SHA256 1회(수명주기 공통) + (IMPRESS는) 재배치 쓰기를 포함한 벽시계 시간이다. "
                 "page cache에 쓰는 시간과 fsync 시간이 섞여 있으므로 디스크 순수 쓰기 속도로 해석하지 않는다.\n")
        L.append("### 3턴 세션 시간 (직전 보고서와 같은 방식)\n")
        L.append("세션 = 한 대화의 3턴 요청 E2E 합 + store 준비(저장 전체 + activation). shared: 이미지의 첫 대화에만 준비 시간을 더함(실측 흐름). standalone: 모든 대화에 더함(DERIVED).\n")
        L.append(md_table(["방법", "요청만 평균(ms)", "shared 평균 / p50 / p95 (ms)", "standalone 평균 / p50 / p95 (ms)"],
                          [[names[s["method"]], f"{s['request_only_mean_ms']:.1f}", f"{s['shared_mean_ms']:.1f} / {s['shared_p50_ms']:.1f} / {s['shared_p95_ms']:.1f}",
                            f"{s['standalone_mean_ms']:.1f} / {s['standalone_p50_ms']:.1f} / {s['standalone_p95_ms']:.1f}"] for s in sessions]))
        L.append(f"- 디스크 최대 사용량: 이미지 하나의 scratch store 묶음 최대 {usage['peak_scratch_bytes_max'] / 1e9:.2f} GB (평균 {usage['peak_scratch_bytes_mean'] / 1e9:.2f} GB, {usage['images']}장). "
                 f"실행 중 관측된 최소 여유 공간 {usage['min_free_bytes_after_requests'] / 2**30:.1f} GiB. 이미지 commit 뒤 scratch는 지운다.\n")
        summary["storage"] = storage; summary["sessions"] = sessions; summary["disk_usage"] = usage; summary["impress_turn1"] = t1

    # ---- mapping and differences
    L.append("## 공개 구현의 함수와 새 구현의 대응\n")
    L.append(md_table(["공개 구현 (`h2o_flexgen/flexgen/`)", "새 구현 (`mmimpress/impress_official.py`)"], MAPPING, ["---", "---"]))
    L.append("## 원본과 다른 점\n")
    L.append("\"방향\"은 IMPRESS(baseline)의 TTFT·정답률에 유리한지, 불리한지, 중립인지다. 방향은 설계상의 판단이며 항목별 효과를 따로 재지는 않았다(미실행).\n")
    L.append(md_table(["#", "항목", "공개 구현", "새 구현", "방향"], [[i + 1, *d] for i, d in enumerate(DIFFERENCES)], ["---", "---", "---", "---", "---"]))

    prot = R.protection_after(H, run, res)
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

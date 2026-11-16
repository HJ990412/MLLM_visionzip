#!/usr/bin/env python3
"""Independent audit, request-paired cross-run analysis, mechanical verdict and reports of the
LLaVA-NeXT PYR-beta-raw packed MT-VQA-25k run (contract docs/llava_next_mtvqa_pyr_contract.md, section 8).

  --stage selfcheck   reproduce the v3 paired table from the v3 reference index with this code path (no new-run data)
  --stage phase --phase smoke|pilot40   small non-main analysis (never a main denominator)
  --stage interim     the single pre-registered interim report on the first 2,500 committed main images
  --stage final       full analysis, decision.json, REPORT.md and the result bundle (main must be complete)
"""
import argparse, csv, importlib.util, json, os, shutil, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
from mmimpress.mtvqa25k.common import read, dump, sha, digest
from mmimpress.mtvqa25k_packed import adapter as AD, analysis as AN, checks as CK, reference as REF

INTERIM = 2500
V3_PROMPT_NUMBERS = {'recompute': {'followup_pct': 51.7728, 'ttft_ms': 530.15, 'session_ms': 1679.79}, 'fullload': {'followup_pct': 51.7806, 'ttft_ms': 755.79, 'session_ms': 4541.38},
                     'sparsevlm_ssd_kv25_allhead': {'followup_pct': 51.5712, 'ttft_ms': 844.59, 'session_ms': 4771.00},
                     'ours_prefixstore_kv25': {'followup_pct': 50.1102, 'ttft_ms': 262.46, 'session_ms': 1954.00}}


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / file)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False)      # encode first: a failure never leaves a truncated file
    with path.open('w') as f:
        f.write(text); f.write('\n'); f.flush(); os.fsync(f.fileno())


def v3_sessions():
    out = {}
    with (ROOT / REF.V3_RESULTS / 'llava/main/session.csv').open() as f:
        for r in csv.DictReader(f):
            out[(r['image_id'], r['method'])] = float(r['standalone_equivalent_session_ms'])
    return out


def bootstrap_fn():
    return load('_mtvqa_analysis_142', '142_analyze_mtvqa25k.py').bootstrap


def selfcheck(run, out_dir):
    """The v3 paired rows ([B] minus each baseline, follow-up score) must be reproduced from the reference index by this code path."""
    ref = REF.load_index(run / 'reference/v3_main_index.jsonl.gz')
    manifest = read(run / 'llava/main/manifest.json'); images = sorted(d['image_id'] for d in manifest['dialogues'])
    pseudo = {i: {t: ref[i][(REF.V3_B, t)] for t in (1, 2, 3)} for i in images}
    paired = AN.paired_scores(pseudo, ref, images, bootstrap_fn())
    stored = {}
    with (ROOT / REF.V3_RESULTS / 'llava/main/paired_analysis.csv').open() as f:
        for r in csv.DictReader(f):
            if r['metric'] == 'score_pct':
                stored[(r['baseline'], r['scope'])] = (float(r['ours_minus_baseline']), float(r['ci95_low']), float(r['ci95_high']))
    rows = []; ok = True
    for m in ('recompute', 'fullload', 'sparsevlm_ssd_kv25_allhead'):
        for scope in AN.SCOPES:
            p = AN.find(paired, m, scope); s = stored[(m, scope)]
            same = all(abs(a - b) < 1e-6 for a, b in zip((p['delta_pp'], p['ci95_low'], p['ci95_high']), s))
            ok &= same
            rows.append({'baseline': m, 'scope': scope, 'recomputed': [p['delta_pp'], p['ci95_low'], p['ci95_high']], 'v3_paired_analysis_csv': list(s), 'equal_1e-6': same})
    same_arm = AN.find(paired, REF.V3_B, 'Followup')
    ok &= same_arm['delta_pp'] == 0 and same_arm['ci95_low'] == 0 and same_arm['ci95_high'] == 0
    record = {'status': 'PASS' if ok else 'FAIL', 'images': len(images), 'rows': rows, 'identical_arm_exact_zero_ci': [same_arm['delta_pp'], same_arm['ci95_low'], same_arm['ci95_high']],
              'meaning': 'the reference index + this paired code reproduce the v3 analysis (scripts/142) numbers: B - ReComp follow-up = %.4f [%.4f, %.4f]' % tuple(AN.find(paired, 'recompute', 'Followup')[k] for k in ('delta_pp', 'ci95_low', 'ci95_high')),
              'v3_paired_csv_sha256': sha(ROOT / REF.V3_RESULTS / 'llava/main/paired_analysis.csv'), 'analyzer_sha256': sha(Path(__file__)), 'uses_new_run_data': False}
    write_json(out_dir / 'analysis_selfcheck.json', record)
    print(json.dumps({'selfcheck': record['status'], 'B_minus_ReComp_followup': rows[4]['recomputed']}), flush=True)
    if not ok:
        raise SystemExit('analysis self-check FAIL')
    return record


def analyze(run, phase, dialogues, ref, out_dir, label, workers=8):
    """Audit + tables for one image set.  Returns the summary dict (also written to out_dir)."""
    config = read(run / 'llava' / phase / 'config.json'); method = config['methods'][0]
    assert config['methods'] == [AD.MAIN_METHOD] and method == AD.MAIN_METHOD
    docs = {method: config['layer_budgets'][AD.vector_of(method)]}
    t0 = time.time()
    audits = AN.audit_phase(run, phase, dialogues, [method], docs, workers)
    failed = [a for a in audits if a['status'] != 'PASS']
    images = [d['image_id'] for d in dialogues]
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = {'status': 'PASS' if not failed else 'FAIL', 'label': label, 'phase': phase, 'images': len(images), 'requests': sum(len(a['rows']) for a in audits),
             'expected_requests': 3 * len(images), 'failed_images': len(failed), 'failures': [{'image_id': a['image_id'], 'failures': a['failures']} for a in failed[:100]],
             'checks': 'commit evidence SHA, journal SHA, frozen question/annotation, official VQA score by the unmodified upstream evaluator, native token decode, '
                       'method-local history + prompt reconstruction, actual read bytes, T1 full-pixel without store reads, follow-up = one preadv of the whole packed file with '
                       'k_l from the document clip range (independent Hamilton), zero vision/scoring calls, store meta dataset + clip range, cleanup + per-image audit receipts',
             'images_with_retry_attempts': sum(a.get('attempts', 1) > 1 for a in audits), 'cap_reached_requests': sum(a.get('cap_reached', 0) for a in audits),
             'analyzer_sha256': sha(Path(__file__)), 'official_original_sha256': sha(ROOT / 'mmimpress/mtvqa25k/vendor/vqaEval.py.original'), 'seconds': time.time() - t0}
    write_json(out_dir / 'independent_audit.json', audit)
    if failed:
        raise SystemExit(f'independent audit FAIL on {len(failed)} images: {failed[0]["failures"][:3]}')
    new = {a['image_id']: {r['turn_id']: r for r in a['rows']} for a in audits}
    gdet = AN.g_det_t1(new, ref, images)
    write_json(out_dir / 'g_det_T1.json', dict(gdet, label=label, phase=phase))
    quality = AN.quality_table(new, ref, images)
    paired = AN.paired_scores(new, ref, images, bootstrap_fn())
    agree = AN.agreement(new, ref, images)
    lat = AN.latency_io(new, ref, images)
    sess, per_image_session = AN.session_storage(audits, method, v3_sessions(), images)
    dec = AN.decision(gdet, paired, lat)
    if gdet['status'] != 'PASS':
        for p in paired:
            p['comparison'] = 'NOT_COMPARABLE (G-det-T1 not 100%)'
    answer_types = {at: sum(new[i][t]['answer_type'] == at for i in images for t in (2, 3)) for at in AN.ANSWER_TYPES}
    summary = {'label': label, 'phase': phase, 'run': run.name, 'arm': method, 'images': len(images), 'requests': 3 * len(images), 'followup_requests': 2 * len(images),
               'independent_audit': audit['status'], 'g_det_T1': {k: gdet[k] for k in ('status', 'images', 'T1_token_exact', 'mismatch_count', 'cross_run_accuracy_comparison')},
               'quality': quality, 'paired': paired, 'agreement_with_B': agree, 'latency_io': lat, 'session_storage': sess, 'followup_answer_type_requests': answer_types,
               'v3_reference_numbers_in_prompt': V3_PROMPT_NUMBERS, 'mtgqa_reference': AN.MTGQA_REFERENCE,
               'bootstrap': 'scripts/142 bootstrap: RandomState MT19937 seed 1234, 10,000 image-cluster draws, pooled sums/counts',
               'created_unix': time.time(), 'source_freeze_sha256': sha(run / 'experiment_freeze.json')}
    write_json(out_dir / 'summary.json', summary); write_json(out_dir / 'decision.json', dec)
    write_csv(out_dir / 'quality.csv', quality); write_json(out_dir / 'quality.json', quality)
    write_csv(out_dir / 'paired_analysis.csv', paired); write_json(out_dir / 'paired_analysis.json', paired)
    write_json(out_dir / 'latency.json', {k: v for k, v in lat.items() if k not in ('ssd_condition',)})
    flat = [{'metric': k, **v} for k, v in lat.items() if isinstance(v, dict) and 'p50' in v]
    write_csv(out_dir / 'latency.csv', flat)
    write_json(out_dir / 'actual_io.json', {k: lat[k] for k in ('SSD_MB_per_followup', 'preadv_calls_per_followup', 'pread_calls_per_followup', 'read_calls_per_followup',
                                                               'T1_ssd_read_bytes_total', 'packed_read_MB_per_s', 'followup_packed_read_ms', 'ssd_condition')})
    write_json(out_dir / 'session.json', sess); write_csv(out_dir / 'session.csv', per_image_session)
    write_json(out_dir / 'storage.json', {k: sess[k] for k in ('store_MB_per_image', 'allocated_MB_per_image', 'os_write_MB_per_image', 'persistence_ms', 'activation_ms')})
    write_json(out_dir / 'cross_run_v3_comparison.json', {'rule': 'request-paired (same image, same turn) against the v3 run; allowed because of G-det-T1 and G-det-U',
                                                          'g_det_T1': summary['g_det_T1'], 'paired': paired, 'agreement_with_B': agree, 'decision': dec,
                                                          'ttft_session_note': 'TTFT / session numbers of v3 are cross-run references only; the same-run baseline is the T1 TTFT'})
    write_csv(out_dir / 'per_request.csv', [{'image_id': i, 'turn_id': t, 'score': new[i][t]['score'], 'B_score_v3': ref[i][(REF.V3_B, t)]['score'],
                                            'recompute_score_v3': ref[i][('recompute', t)]['score'], 'prediction': new[i][t]['prediction'], 'B_prediction_v3': ref[i][(REF.V3_B, t)]['prediction'],
                                            'answer_type': new[i][t]['answer_type'], 'ttft_ms': new[i][t]['ttft_ms'], 'request_e2e_ms': new[i][t]['request_e2e_ms'],
                                            'ssd_read_bytes': new[i][t]['ssd_read_bytes'], 'N_content': new[i][t]['N_content'], 'packed_read_ms': new[i][t]['packed_read_ms'],
                                            'scatter_ms': new[i][t]['scatter_ms'], 'prefill_ms': new[i][t]['prefill_ms']} for i in images for t in (1, 2, 3)])
    return summary, dec, audits


# ----------------------------------------------------------------- render
def temp(env):
    t = env.get('ssd_temperature_c_hwmon')
    return f'{max(t.values()):.1f} °C' if isinstance(t, dict) and t else 'NOT_MEASURED'


REPRODUCE = '''# 재현 절차 — LLaVA-NeXT PYR-beta-raw packed MT-VQA 25k 단일 arm

환경: conda env `mllm_ft` (`/home/dblab/anaconda3/envs/mllm_ft/bin/python`), RTX 4090, 저장소 `/home/dblab/hj/mllm_v2`. 다른 GPU compute 프로세스가 없어야 한다.
필수 입력(읽기 전용): v3 run `runs/mt_vqa_v2_reconstructed_25k_v3_20261021T100500Z`(이미지별 raw gzip journal 25,000개)와 그 results, `results/layer_budget_kv25_pyrbeta_20261110T073604Z`의 벡터 json,
`data/mt_vqa_v2_reconstructed_25k`. 계약 `docs/llava_next_mtvqa_pyr_contract.md`.

```
PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python
RUN=runs/llava_next_mtvqa_pyr_<UTC timestamp>          # 새 run root (기존 것을 덮어쓰지 않는다)
# 0) 새 파일을 만들기 전의 소스 SHA256 inventory(source_before.json)를 준비한 뒤
$PY scripts/181_freeze_mtvqa_pyr.py --run $RUN --stage prepare --source-before <source_before.json>   # v3 raw index, 데이터·채점·벡터·k_l·모델/환경 검사
$PY scripts/184_analyze_mtvqa_pyr.py --run $RUN --stage selfcheck                                      # v3 paired 표 재현(분석 코드 자기 검사)
$PY scripts/181_freeze_mtvqa_pyr.py --run $RUN --stage freeze
# 1) gate
for s in vg1 vg2 vg3 vg6 vg4; do $PY scripts/182_validate_mtvqa_pyr.py --run $RUN --stage $s; done
$PY scripts/183_run_mtvqa_pyr.py --run $RUN --phase gate_u_pilot40 && $PY scripts/182_validate_mtvqa_pyr.py --run $RUN --stage vg5
$PY scripts/182_validate_mtvqa_pyr.py --run $RUN --stage summary
# 2) smoke -> pilot40 -> admission -> main
$PY scripts/183_run_mtvqa_pyr.py --run $RUN --phase smoke   && $PY scripts/182_validate_mtvqa_pyr.py --run $RUN --stage t1 --phase smoke
$PY scripts/183_run_mtvqa_pyr.py --run $RUN --phase pilot40 && $PY scripts/182_validate_mtvqa_pyr.py --run $RUN --stage t1 --phase pilot40
$PY scripts/183_run_mtvqa_pyr.py --run $RUN --phase admit
$PY scripts/183_run_mtvqa_pyr.py --run $RUN --phase main            # 중단되면 같은 명령에 --resume (freeze 불변일 때만). 2,500장에서 INTERIM_2500.md를 한 번 쓴다
# 3) 독립 감사·분석·판정·보고
$PY scripts/184_analyze_mtvqa_pyr.py --run $RUN --stage final
```

gate가 실패해 코드를 고치면 `--stage refreeze --amendment <json>`으로 수정 기록을 남기고 모든 gate와 smoke·pilot40을 다시 실행한다.
정확도는 greedy·seed 고정이라 같은 모델 snapshot·패키지 버전에서 같은 token이 나와야 한다(G-det-T1, G-det-U가 그 확인이다). TTFT는 실행 환경에 따라 달라진다.
'''


def ci(p):
    if not p['ci_defined']:
        return f"{p['delta_pp']:+.3f} pp [CI 미정의: 표본이 작음]"
    return f"{p['delta_pp']:+.3f} pp [{p['ci95_low']:+.3f}, {p['ci95_high']:+.3f}]"


def tables(summary, dec):
    """Markdown tables of contract section 8 (shared by INTERIM and REPORT)."""
    P = summary['paired']; lat = summary['latency_io']; sess = summary['session_storage']; A = dec['A_layer_allocation_effect']
    L = []
    g = summary['g_det_T1']
    L += [f"정합성 G-det-T1: PYR arm T1 생성 token = v3 T1 **{g['T1_token_exact']:,}/{g['images']:,}** → {g['status']} (cross-run 정확도 비교: {g['cross_run_accuracy_comparison']}).", '']
    L += ['### (A) PyramidKV 배분 효과 — D = PYR-beta-raw_packed − [B] (followup 공식 VQA score, cross-run, 결정성 확인됨)', '',
          '| 항목 | 값 |', '|---|---|',
          f"| PYR-beta-raw packed followup | {A['PYR_followup_pct']:.4f} % |", f"| [B] v3 followup (같은 이미지) | {A['B_followup_pct']:.4f} % |",
          f"| D, 95% CI | {A['D_pp']:+.4f} pp [{A['ci95'][0]:+.4f}, {A['ci95'][1]:+.4f}] |", f"| 판정 규칙 결과 | **{A['verdict']}** |",
          f"| 참고: D ≥ +1.0 pp (MT-GQA GO 기준) | {'예' if A['reference_not_verdict']['D_at_least_1.0pp_MTGQA_GO_threshold'] else '아니오'} |",
          f"| 참고: 손실 회복률 D / (ReComp − [B]) | {A['reference_not_verdict']['loss_recovery_percent']:.1f} % ({A['reference_not_verdict']['loss_definition']}) |",
          f"| 조건: SSD MB/followup ≤ [B] × 1.02 | {A['condition_SSD_MB_per_followup_le_B_x_1.02']['PYR_MB_per_followup']:.2f} vs {A['condition_SSD_MB_per_followup_le_B_x_1.02']['B_MB_per_followup_v3']:.2f} MB → {'충족' if A['condition_SSD_MB_per_followup_le_B_x_1.02']['within_B_times_1.02'] else '미충족'} |", '']
    L += ['### (B) 원래 성능 대비 (followup, cross-run, 결정성 확인됨)', '', '| 비교 | Δ, 95% CI | baseline % | baseline 대비 % | CI |', '|---|---:|---:|---:|---|']
    for m in ('recompute', 'fullload', 'sparsevlm_ssd_kv25_allhead'):
        b = dec['B_vs_original'][m]
        L.append(f"| PYR − {AN.LABELS[m]} | {b['delta_pp']:+.4f} pp [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] | {b['baseline_followup_pct']:.4f} | {b['percent_of_baseline']:.2f} | {b['ci_statement']} |")
    L += ['', '### (C) turn별 공식 VQA score (%)', '', '| arm | T1 | T2 | T3 | All | Followup |', '|---|---:|---:|---:|---:|---:|']
    for q in summary['quality']:
        L.append(f"| {q['arm']} | " + ' | '.join(f"{q[s]:.4f}" for s in AN.SCOPES) + ' |')
    L += ['', 'turn별 Δ (PYR − baseline, cross-run, 결정성 확인됨):', '', '| baseline | T1 | T2 | T3 | All | Followup |', '|---|---|---|---|---|---|']
    for m in REF.COMPARE_METHODS:
        L.append(f"| {AN.LABELS[m]} | " + ' | '.join(ci(AN.find(P, m, s)) for s in AN.SCOPES) + ' |')
    L += ['', 'answer_type별 followup Δ (탐색적, 다중비교 보정 없음):', '', '| answer_type | 요청 수 | PYR % | [B] % | PYR − [B] | ReComp % | PYR − ReComp |', '|---|---:|---:|---:|---|---:|---|']
    for at in AN.ANSWER_TYPES:
        try:
            b = AN.find(P, REF.V3_B, 'Followup', at); r = AN.find(P, 'recompute', 'Followup', at)
        except StopIteration:
            continue
        L.append(f"| {at} | {b['requests']:,} | {b['new_mean']:.4f} | {b['baseline_mean']:.4f} | {ci(b)} | {r['baseline_mean']:.4f} | {ci(r)} |")
    ag = summary['agreement_with_B']
    L += ['', 'PYR과 [B]의 예측 비교 (정답 = 그 요청의 공식 VQA score > 0):', '', '| 범위 | 요청 수 | 예측 문자열 일치 | 일치율 | PYR만 정답 | [B]만 정답 | PYR score 더 높음 | [B] score 더 높음 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in ('T2', 'T3', 'Followup'):
        a = ag[s]
        L.append(f"| {s} | {a['requests']:,} | {a['prediction_string_equal']:,} | {100 * a['prediction_agreement_rate']:.2f} % | {a['only_PYR_correct']:,} | {a['only_B_correct']:,} | {a['PYR_score_higher']:,} | {a['B_score_higher']:,} |")
    v3 = lat['v3_cross_run_reference']; env = lat['environment_drift']
    L += ['', '### (D) 지연·I/O', '', '| 지표 (이번 실행, PYR-beta-raw packed) | mean | p50 | p95 |', '|---|---:|---:|---:|']
    for key, name in (('followup_ttft_ms', 'followup TTFT ms'), ('followup_e2e_ms', 'followup E2E ms'), ('T2_ttft_ms', 'T2 TTFT ms'), ('T3_ttft_ms', 'T3 TTFT ms'), ('T1_ttft_ms', 'T1 TTFT ms (full pixel)'),
                      ('T1_e2e_ms', 'T1 E2E ms'), ('followup_packed_read_ms', 'packed read ms (preadv)'), ('followup_scatter_ms', 'scatter ms'), ('followup_prefill_ms', 'suffix prefill ms'),
                      ('followup_prepare_ms', 'prepare ms (read + scatter + mask)')):
        s = lat[key]; L.append(f"| {name} | {s['mean']:.2f} | {s['p50']:.2f} | {s['p95']:.2f} |")
    L += ['', f"- SSD {lat['SSD_MB_per_followup']:.2f} MB/followup, preadv {lat['preadv_calls_per_followup']:.2f}회·pread {lat['pread_calls_per_followup']:.2f}회/followup, T1의 store 읽기 {lat['T1_ssd_read_bytes_total']} bytes.",
          f"- **같은 실행 안 비율 R_in = followup TTFT / T1 TTFT = {lat['followup_ttft_ms']['mean']:.2f} / {lat['T1_ttft_ms']['mean']:.2f} = {lat['R_in_followup_over_T1_same_run']:.4f}** (T1은 full pixel 재계산). "
          f"참고로 v3 같은 이미지에서 ReComp followup / T1 = {v3['recompute']['followup_over_T1_in_v3']:.4f}, [B] followup / T1 = {v3['ours_prefixstore_kv25']['followup_over_T1_in_v3']:.4f}.",
          '', '| v3 arm (cross-run 참고치, 절대 TTFT는 실행 간 비교하지 않는다) | followup TTFT ms | T1 TTFT ms | SSD MB/followup | read 호출/followup |', '|---|---:|---:|---:|---:|']
    for m in REF.COMPARE_METHODS:
        L.append(f"| {AN.LABELS[m]} | {v3[m]['followup_ttft_mean_ms']:.2f} | {v3[m]['T1_ttft_mean_ms']:.2f} | {v3[m]['SSD_MB_per_followup']:.2f} | {v3[m]['preads_per_followup']:.2f} |")
    L += ['', f"- 환경 변동 지표(판정 아님): 이번 T1 TTFT {env['T1_ttft_mean_ms_this_run']:.2f} ms / v3 T1 {env['v3_T1_ttft_mean_ms_all_arms_same_images']:.2f} ms(5 arm 평균, 같은 이미지) = {env['ratio_this_over_v3_all_arms']:.4f}"
          f" ([B] arm T1 {env['v3_T1_ttft_mean_ms_B_arm_same_images']:.2f} ms 대비 {env['ratio_this_over_v3_B_arm']:.4f}). packed read 처리율 {env['packed_read_MB_per_s_this_run']:.0f} MB/s"
          f" (MT-GQA PYR run 308.05 MB / 97.5 ms = {env['packed_read_MB_per_s_MTGQA_PYR_run']:.0f} MB/s).", '']
    L += ['### (E) 저장·session', '', '| 지표 (이번 실행) | mean | p50 | p95 |', '|---|---:|---:|---:|']
    for key, name in (('store_MB_per_image', 'store MB/이미지 (logical)'), ('persistence_ms', 'persistence ms'), ('activation_ms', 'activation ms'), ('request_e2e_sum_ms', '3 turn 요청 E2E 합 ms'),
                      ('session_3turn_ms', '3-turn session ms = 3 E2E + persistence + activation')):
        s = sess[key]; L.append(f"| {name} | {s['mean']:.2f} | {s['p50']:.2f} | {s['p95']:.2f} |")
    r = sess['v3_cross_run_reference']
    L += ['', '이미지당 dialogue 1개라 분할 상각이 없다. v3 session(cross-run 참고치, 같은 이미지): ' +
          ', '.join(f"{AN.LABELS[m]} {r[m]['session_mean_ms']:.2f} ms" for m in REF.COMPARE_METHODS if m in r) + '.', '']
    F = dec['F_MTGQA_consistency']
    L += ['### (F) MT-GQA와의 일관성', '',
          f"MT-GQA(EM)에서는 PYR − U = +{F['MTGQA_PYR_minus_U_pp']:.3f} pp [{F['MTGQA_ci95'][0]:+.3f}, {F['MTGQA_ci95'][1]:+.3f}]였다. MT-VQA(공식 VQA score)의 D = {F['MTVQA_D_pp']:+.3f} pp "
          f"[{F['MTVQA_ci95'][0]:+.3f}, {F['MTVQA_ci95'][1]:+.3f}]. 부호 {'같음' if F['same_sign'] else '다름'}. 지표가 EM과 VQA score로 다르므로 크기를 직접 비교하지 않는다.", '']
    return L


def render_interim(run, summary, dec):
    L = ['# INTERIM — LLaVA-NeXT PYR-beta-raw packed MT-VQA 25k, 처음 2,500장 (판정 아님)', '',
         f"Run `runs/{run.name}`. 작성 {time.strftime('%Y-%m-%d %H:%M:%S %Z')}. **이 문서는 사전 고정된 1회 중간 보고이며 판정이 아니다.** "
         '실행은 멈추지 않고 계속되며, 이 결과로 arm·설정·규칙을 바꾸지 않는다. 최종 판정은 25,000장 완주 후 `REPORT.md`와 `decision.json`에만 있다.', '',
         f"- 대상: main manifest 순서의 처음 {summary['images']:,}장(무작위 표본 순서이므로 무작위 부분집합), {summary['requests']:,} 요청(followup {summary['followup_requests']:,}).",
         '- main에서 실행한 arm은 PYR-beta-raw_packed 하나다. ReComp·FullLoad·SparseVLM·[B] 수치는 v3 run(2026-10)의 같은 이미지 기록이며 이번에 다시 실행하지 않았다.',
         '- 아래 "판정 규칙 결과" 칸은 같은 규칙을 2,500장에 기계적으로 적용한 값일 뿐이다(판정 아님).',
         f"- 독립 감사(2,500장): {summary['independent_audit']}.", '']
    L += tables(summary, dec)
    L += ['한계는 최종 보고서와 같다(cross-run TTFT는 참고치, DONTNEED는 cold NAND 보장이 아님, dense GPU 버퍼 유지).']
    return '\n'.join(L) + '\n'


def render_report(run, summary, dec, extra):
    c = extra['counts']; g = extra['gates']
    L = ['# LLaVA-NeXT-7B + PyramidKV 배분(PYR-beta-raw packed) — MT-VQA-v2-reconstructed-25k 단일 arm', '',
         f"Run `runs/{run.name}`, 결과 `results/{run.name}`. 계약 `docs/llava_next_mtvqa_pyr_contract.md` (SHA256 `{extra['contract_sha256']}`). "
         f"작성 {time.strftime('%Y-%m-%d %H:%M:%S %Z')}.", '',
         '## 요약', '',
         f"- main 요청 **{c['main_requests']:,}/75,000** (25,000 images × 3 turns), 실패 {c['failed_attempts']}건·스킵 0건, 독립 감사 **{summary['independent_audit']}**. "
         f"smoke {c['smoke_requests']}·pilot40 {c['pilot40_requests']}·G-det-U {c['gate_u_requests']} 요청은 main 분모에 넣지 않았다.",
         f"- main 소요 시간 {extra['main_hours']:.2f} h (이미지 실행·감사·hash·gzip·삭제 합 {extra['image_wall_hours']:.2f} h; 모델 로딩·중간 보고 포함 경과).",
         f"- **(A) 주 판정: {dec['A_layer_allocation_effect']['verdict']}** — D = PYR-beta-raw_packed − [B] = {dec['A_layer_allocation_effect']['D_pp']:+.4f} pp "
         f"[{dec['A_layer_allocation_effect']['ci95'][0]:+.4f}, {dec['A_layer_allocation_effect']['ci95'][1]:+.4f}] (followup 공식 VQA score, cross-run, 결정성 확인됨).",
         f"- 근거 gate: G-det-T1 {summary['g_det_T1']['T1_token_exact']:,}/{summary['g_det_T1']['images']:,}, G-det-U(pilot40) {g['vg5_equal']}/120.", '']
    L += ['## 반드시 함께 읽을 사실', '',
          '- main에서 실행한 arm은 PYR-beta-raw_packed 하나다. ReComp·FullLoad·SparseVLM·[B] 수치는 v3 run(2026-10)의 기록이며 이번에 다시 실행하지 않았다.',
          f"- 정확도 비교는 요청 단위로 짝지은 cross-run 비교이고, 근거는 G-det-T1({summary['g_det_T1']['T1_token_exact']:,}/{summary['g_det_T1']['images']:,}건)과 G-det-U(pilot40 {g['vg5_equal']}/120건)다.",
          '- TTFT·session의 v3 대비 비교는 cross-run 참고치다. 같은 실행 안 기준은 T1 TTFT뿐이다.',
          '- PYR-beta-raw는 PyramidKV 공개 코드 식(β=20)의 layer 비율 그대로이며 attention 기반 token 선택은 쓰지 않았다. [B]와 같은 token 순서·같은 총 바이트이고, '
          'layer별 자르는 지점과 I/O 경로(packed)만 다르다. 따라서 D에는 layer 배분 효과만 들어가지만(G-det-U로 packed 경로 자체는 예측을 바꾸지 않음을 확인), '
          'TTFT 차이에는 packed 경로 효과가 섞여 있다.',
          '- MPIC는 비교표에서 제외했다(기존 MPIC 결과는 지우지 않았다). Qwen·GLM·LLaVA-1.5·MT-GQA는 이번에 실행하지 않았다.',
          '- v3와 같은 한계: 기존 250장 개발 데이터와 이미지 166개 중복(unseen holdout 아님), DONTNEED는 cold NAND 보장이 아님, dense GPU 버퍼 유지(GPU 메모리 절감 주장 없음).',
          '- CI가 0을 포함하는 비교는 "차이 미입증"이며 동등성의 증거가 아니다.', '']
    L += ['## 판정과 표 (계약 8절, 실행 전 고정)', '']
    L += tables(summary, dec)
    L += ['## Gate (Phase 1)', '', '| gate | 결과 |', '|---|---|']
    for name, state in g['states'].items():
        L.append(f'| {name} | {state} |')
    L += ['', f"freeze 수정(amendment) {extra['amendments']}건. " + (extra['amendment_note'] or ''), '']
    L += ['## 실행 기록', '',
          f"- 실제 요청 수: main {c['main_requests']:,}, smoke {c['smoke_requests']}, pilot40 {c['pilot40_requests']}, G-det-U gate {c['gate_u_requests']} (합 {c['main_requests'] + c['smoke_requests'] + c['pilot40_requests'] + c['gate_u_requests']:,}).",
          f"- 실패 attempt {c['failed_attempts']}건, 재시도로 commit된 이미지 {c['retried_images']}장, 스킵 0. 16 token cap 도달 요청 {c['cap_reached']}건.",
          f"- 중단·재개: main 프로세스 시작 {c['main_starts']}회.",
          f"- scratch store 삭제: {c['cleaned_images']:,}장(정확한 SHA allowlist, 감사·commit 후). 남은 scratch 파일 {c['scratch_left_files']}개.",
          f"- 보호 검사: 기존 소스 변경 {extra['protection']['changed_existing_sources']}건, 기존 runs/results 파일 변경 {extra['protection']['changed_artifacts']}건·삭제 {extra['protection']['missing_artifacts']}건.",
          f"- 환경(판정 아님): 시작 시 여유 {extra['env_before']['disk_free_bytes'] / 1e9:.1f} GB → 종료 시 {extra['env_after']['disk_free_bytes'] / 1e9:.1f} GB, SSD {extra['env_before']['ssd_model']}, "
          f"SSD 온도(hwmon 최고값) 시작 {temp(extra['env_before'])} / 종료 {temp(extra['env_after'])}. 2,500장마다의 T1 TTFT·packed read 처리율은 `environment_log.jsonl`.", '']
    L += ['## 프롬프트와 다르게 한 점', ''] + [f'- {x}' for x in extra['deviations']] + ['']
    L += ['## NOT_MEASURED', ''] + [f'- {x}' for x in extra['not_measured']] + ['']
    return '\n'.join(L) + '\n'


def protection_after(run):
    F = load('_pyr_freeze_for_report', '181_freeze_mtvqa_pyr.py')
    before = json.loads((run / 'protection/artifacts_before.inventory.json').read_text())
    after = F.artifact_inventory()
    missing = [k for k in before if k not in after]
    changed = [k for k in before if k in after and after[k] != before[k]]
    added = [k for k in after if k not in before]
    src = F.check_existing_sources(run)
    return {'status': 'PASS' if not missing and not changed and src['status'] == 'PASS' else 'FAIL', 'files_before': len(before), 'files_after': len(after),
            'missing_artifacts': len(missing), 'changed_artifacts': len(changed), 'added_outside_this_task': len(added), 'examples_missing': missing[:20], 'examples_changed': changed[:20],
            'examples_added': added[:20], 'changed_existing_sources': len(src['changed_existing']), 'existing_source_check': src,
            'policy': 'size + mtime_ns of every file under runs/ and results/ outside this task; full SHA256 of every pre-task source file'}


def stage_phase(run, phase):
    ref = REF.load_index(run / 'reference' / f'v3_{phase}_index.jsonl.gz')
    manifest = read(run / 'llava' / phase / 'manifest.json')
    out = ROOT / 'results' / run.name / phase
    summary, dec, _ = analyze(run, phase, manifest['dialogues'], ref, out, f'{phase} (not main; reference = v3 {phase} rows of the same images)')
    lat = summary['latency_io']
    print(json.dumps({'phase': phase, 'images': summary['images'], 'audit': summary['independent_audit'], 'g_det_T1': summary['g_det_T1']['status'],
                      'followup_ttft_mean_ms': lat['followup_ttft_ms']['mean'], 'T1_ttft_mean_ms': lat['T1_ttft_ms']['mean'], 'R_in': lat['R_in_followup_over_T1_same_run']}), flush=True)


def stage_interim(run):
    target = ROOT / 'results' / run.name / 'INTERIM_2500.md'
    if target.exists():
        raise FileExistsError('the single interim report already exists')
    load('_pyr_runner_for_interim', '183_run_mtvqa_pyr.py').verify_freeze(run)
    manifest = read(run / 'llava/main/manifest.json'); dialogues = manifest['dialogues'][:INTERIM]
    ref = REF.load_index(run / 'reference/v3_main_index.jsonl.gz')
    out = ROOT / 'results' / run.name / 'interim_2500'
    summary, dec, _ = analyze(run, 'main', dialogues, ref, out, 'INTERIM first 2,500 committed main images — not a verdict')
    target.write_text(render_interim(run, summary, dec))
    print(json.dumps({'interim': 'WRITTEN', 'images': summary['images']}), flush=True)


def stage_final(run):
    res = ROOT / 'results' / run.name
    W = load('_pyr_runner_for_report', '183_run_mtvqa_pyr.py')
    freeze = W.verify_freeze(run)
    complete = read(run / 'llava/main/COMPLETE.json')
    assert complete['status'] == 'PASS' and complete['requests'] == 75000 and complete['source_freeze_sha256'] == sha(run / 'experiment_freeze.json')
    manifest = read(run / 'llava/main/manifest.json')
    ref = REF.load_index(run / 'reference/v3_main_index.jsonl.gz')
    check = selfcheck(run, res)
    summary, dec, audits = analyze(run, 'main', manifest['dialogues'], ref, res, 'FINAL main 25,000 images')
    assert summary['requests'] == 75000 and summary['independent_audit'] == 'PASS'
    dec['analysis_selfcheck'] = check['status']; dec['final'] = True
    write_json(res / 'decision.json', dec)
    # ---- bundle
    shutil.copy2(run / 'CONTRACT.md', res / 'CONTRACT.md')
    (res / 'vectors').mkdir(exist_ok=True)
    for p in (run / 'vectors').iterdir():
        shutil.copy2(p, res / 'vectors' / p.name)
    for name in ('k_tables.json', 'k_tables.csv'):
        shutil.copy2(run / 'phase0' / name, res / name)
    (res / 'gates').mkdir(exist_ok=True)
    for p in (run / 'gates').glob('*.json'):
        shutil.copy2(p, res / 'gates' / p.name)
    shutil.copy2(run / 'gates/vg5_g_det_U.json', res / 'g_det_U_pilot40.json')
    for p in list(run.glob('freeze_amendment_*.json')) + [run / 'experiment_freeze.json', run / 'llava/main_resource_admission.json', run / 'reference/reference_check.json']:
        shutil.copy2(p, res / p.name)
    shutil.copy2(run / 'phase0/environment_before.json', res / 'environment_before.json')
    F = load('_pyr_freeze_for_env', '181_freeze_mtvqa_pyr.py')
    env_after = F.environment_record(run); write_json(res / 'environment_after.json', env_after)
    env_before = read(run / 'phase0/environment_before.json')
    for phase in ('main', 'pilot40'):
        for name in ('environment_log.jsonl', 'image_log.jsonl'):
            if (run / 'llava' / phase / name).exists():
                shutil.copy2(run / 'llava' / phase / name, res / f'{phase}_{name}')
    shutil.copy2(run / 'protection/artifacts_before.json', res / 'protection_before.json')
    shutil.copy2(run / 'protection/source_before.json', res / 'protection_source_before.json')
    prot = protection_after(run); write_json(res / 'protection_after.json', prot)
    # ---- counts / deletion log
    def phase_requests(phase):
        p = run / 'llava' / phase / 'COMPLETE.json'
        return read(p)['requests'] if p.exists() else 0
    failed = sorted(str(p.relative_to(run)) for p in run.glob('llava/*/*/images/*/attempt_*/failure.json'))
    scratch_left = [str(p.relative_to(run)) for p in run.glob('llava/*/scratch/**/*') if p.is_file()]
    starts = sorted(p.name for p in (run / 'llava/main').glob('phase_start_*.json'))
    stops = sorted(p.name for p in (run / 'llava/main').glob('phase_stop_*.json'))
    first = read(run / 'llava/main' / starts[0])['started_unix']; last = read(run / 'llava/main' / stops[-1])['stopped_unix']
    log = [json.loads(s) for s in (run / 'llava/main/image_log.jsonl').read_text().splitlines() if s.strip()]
    counts = {'main_requests': summary['requests'], 'smoke_requests': phase_requests('smoke'), 'pilot40_requests': phase_requests('pilot40'), 'gate_u_requests': phase_requests('gate_u_pilot40'),
              'failed_attempts': len(failed), 'retried_images': sum(a.get('attempts', 1) > 1 for a in audits), 'cap_reached': sum(a.get('cap_reached', 0) for a in audits),
              'cleaned_images': sum(a.get('removed_scratch_files', 0) > 0 for a in audits), 'scratch_left_files': len(scratch_left), 'main_starts': len(starts)}
    write_json(res / 'deletion_log.json', {'policy': 'only run-local scratch stores of this run, exact SHA allowlist, after per-image audit + atomic commit; gate fixtures by guarded rmtree',
                                           'main_images_cleaned': counts['cleaned_images'], 'removed_scratch_files_main': sum(a.get('removed_scratch_files', 0) for a in audits),
                                           'scratch_files_left': scratch_left[:200], 'failed_attempts': failed, 'gate_stores': 'llava/gpu_gates/* removed after each receipt; gates/vg3_fixture removed',
                                           'existing_stores_runs_results_deleted': 0, 'mpic_results_deleted': 0})
    gates = read(run / 'gates/COMPLETE.json'); vg5 = read(run / 'gates/vg5_g_det_U.json')
    amendments = sorted(run.glob('freeze_amendment_*.json'))
    extra = {'counts': counts, 'gates': {'states': gates['gates'], 'vg5_equal': min(vg5['equal'].values())}, 'contract_sha256': freeze['contract_sha256'],
             'main_hours': (last - first) / 3600, 'image_wall_hours': sum(r['wall_seconds'] for r in log) / 3600, 'protection': prot, 'env_before': env_before, 'env_after': env_after,
             'amendments': len(amendments), 'amendment_note': ' / '.join(f"{p.name}: {read(p).get('reason', '')}" for p in amendments),
             'deviations': read(run / 'deviations.json')['deviations'] if (run / 'deviations.json').exists() else [],
             'not_measured': read(run / 'deviations.json')['not_measured'] if (run / 'deviations.json').exists() else []}
    summary.update(counts=counts, main_hours=extra['main_hours'], image_wall_hours=extra['image_wall_hours'], gates=gates['gates'], protection=prot['status'], analysis_selfcheck=check['status'])
    write_json(res / 'summary.json', summary)
    (res / 'REPORT.md').write_text(render_report(run, summary, dec, extra))
    (res / 'REPRODUCE.md').write_text(REPRODUCE)
    A = dec['A_layer_allocation_effect']; lat = summary['latency_io']
    record = ['# 결과 기록 — LLaVA-NeXT PYR-beta-raw packed MT-VQA 25k 단일 arm', '',
              f"- Run `runs/{run.name}`, 결과 `results/{run.name}` (REPORT.md, decision.json, summary.json). 작성 {time.strftime('%Y-%m-%d')}.",
              f"- 요청: main {counts['main_requests']:,}/75,000, 실패 {counts['failed_attempts']}, 감사 {summary['independent_audit']}, main {extra['main_hours']:.2f} h. "
              f"G-det-T1 {summary['g_det_T1']['T1_token_exact']:,}/{summary['g_det_T1']['images']:,}, G-det-U {extra['gates']['vg5_equal']}/120.",
              f"- (A) **{A['verdict']}**: D = PYR − [B] = {A['D_pp']:+.4f} pp [{A['ci95'][0]:+.4f}, {A['ci95'][1]:+.4f}] (followup {A['PYR_followup_pct']:.4f} % vs {A['B_followup_pct']:.4f} %, cross-run, 결정성 확인됨). "
              f"D ≥ +1.0 pp: {'예' if A['reference_not_verdict']['D_at_least_1.0pp_MTGQA_GO_threshold'] else '아니오'}, 손실 회복률 {A['reference_not_verdict']['loss_recovery_percent']:.1f} %.",
              '- (B) ' + '; '.join(f"PYR − {AN.LABELS[m]} = {dec['B_vs_original'][m]['delta_pp']:+.4f} pp [{dec['B_vs_original'][m]['ci95'][0]:+.4f}, {dec['B_vs_original'][m]['ci95'][1]:+.4f}]" for m in dec['B_vs_original']) + '.',
              f"- (D) followup TTFT {lat['followup_ttft_ms']['mean']:.2f} ms (p50 {lat['followup_ttft_ms']['p50']:.2f}, p95 {lat['followup_ttft_ms']['p95']:.2f}), T1 TTFT {lat['T1_ttft_ms']['mean']:.2f} ms, "
              f"R_in = {lat['R_in_followup_over_T1_same_run']:.4f}; SSD {lat['SSD_MB_per_followup']:.2f} MB/followup, preadv 1회. v3 TTFT는 cross-run 참고치.",
              f"- (E) 3-turn session {summary['session_storage']['session_3turn_ms']['mean']:.2f} ms, store {summary['session_storage']['store_MB_per_image']['mean']:.2f} MB/이미지.",
              f"- (F) MT-GQA PYR − U = +1.244 pp(EM) vs MT-VQA D = {A['D_pp']:+.3f} pp(VQA score).",
              '- main arm은 PYR-beta-raw_packed 하나이며 ReComp·FullLoad·SparseVLM·[B]는 v3(2026-10) 기록이다. MPIC 제외.']
    (res / 'RESULT_RECORD.md').write_text('\n'.join(record) + '\n')
    print(json.dumps({'final': 'WRITTEN', 'verdict': A['verdict'], 'D_pp': A['D_pp'], 'ci95': A['ci95'], 'requests': summary['requests'], 'protection': prot['status']}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--stage', choices=('selfcheck', 'phase', 'interim', 'final'), required=True); ap.add_argument('--phase', choices=('smoke', 'pilot40'))
    a = ap.parse_args(); run = a.run.resolve()
    assert run.parent == ROOT / 'runs' and run.name.startswith(AD.RUN_PREFIX)
    if a.stage == 'selfcheck':
        selfcheck(run, ROOT / 'results' / run.name)
    elif a.stage == 'phase':
        stage_phase(run, a.phase)
    elif a.stage == 'interim':
        stage_interim(run)
    else:
        stage_final(run)

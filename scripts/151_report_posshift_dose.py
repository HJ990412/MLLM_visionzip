#!/usr/bin/env python3
"""Report for the PosShift dose-response diagnostic: REPORT.md, summary.csv, curve.csv, REPRODUCE.md, analysis.json.

Reads only the run's raw rows; the reading rule is the frozen contract's.
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmimpress.recompute_kv25 import paired_bootstrap
from mmimpress.posshift_dose import cap_reached, monotone_non_increasing, reading


def jl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def table(head, rows):
    out = ['| ' + ' | '.join(head) + ' |', '|' + '|'.join(['---'] + ['---:'] * (len(head) - 1)) + '|']
    return '\n'.join(out + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


def ci_str(c):
    return f'{100 * c["mean"]:+.2f} [{100 * c["ci_low"]:+.2f}, {100 * c["ci_high"]:+.2f}]'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); args = ap.parse_args()
    run = Path(args.run).resolve(); res = ROOT / 'results' / run.name; res.mkdir(exist_ok=True)
    base = run / 'llava'
    raw = jl(base / 'gqa/raw_requests.jsonl')
    fails = [r for r in raw if r.get('status') != 'OK']
    assert not fails, f'{len(fails)} failed raw rows'
    images = sorted({r['image_id'] for r in raw}); assert len(images) == 40
    gates = jl(base / 'gqa/gates.jsonl'); assert len(gates) == 40 and all(g['status'] == 'PASS' for g in gates)
    status = json.loads((base / 'status.json').read_text())
    protection = json.loads((run / 'protection_result.json').read_text()) if (run / 'protection_result.json').exists() else {'status': 'NOT RUN'}
    config = json.loads((run / 'config.json').read_text())
    hits = [r for r in raw if r['cache_hit']]
    labels = ['ours'] + [f'shift_{d}' for d in (0, 1, 4, 16, 64, 256, 1024)] + ['shift_max']
    by = defaultdict(dict)
    for r in hits:
        by[r['arm']][(r['image_id'], r['turn'])] = r
    for a in labels:
        assert len(by[a]) == 200, (a, len(by[a]))
    keys = [(i, t) for i in images for t in range(2, 7)]
    delta_of = {a: (None if a == 'ours' else by[a][keys[0]]['delta']) for a in labels}
    dmax = sorted({by['shift_max'][k]['delta'] for k in keys})

    # gates / reproduction
    g_c = sum(1 for k in keys if by['shift_0'][k]['delta0_bitwise_equal_base'])
    g_d = sum(1 for k in keys if by['ours'][k]['reproduces_prior'])
    rep_max = sum(1 for k in keys if by['shift_max'][k]['reproduces_prior'])
    verified = sum(1 for r in hits if r['position_check']['verified'])
    io_one = all(len({tuple(by[a][k]['io_key']) for a in labels}) == 1 for k in keys)

    # curve
    rows, curve = [], {}
    base_em = {i: [by['ours'][(i, t)]['accuracy'] for t in range(2, 7)] for i in images}
    for a in labels:
        em = float(np.mean([by[a][k]['accuracy'] for k in keys]))
        per = {i: [by[a][(i, t)]['accuracy'] for t in range(2, 7)] for i in images}
        ci = paired_bootstrap([(sum(per[i]), sum(base_em[i]), 5) for i in images])
        ft = float(np.mean([by[a][k]['first_token_id'] == by['ours'][k]['first_token_id'] for k in keys]))
        tok = float(np.mean([by[a][k]['generated_token_ids'] == by['ours'][k]['generated_token_ids'] for k in keys]))
        cap = float(np.mean([cap_reached(by[a][k]) for k in keys]))
        distinct = len({by[a][k]['prediction'] for k in keys})
        blank = float(np.mean([by[a][k]['prediction'].strip() == '' for k in keys]))
        curve[a] = {'delta': delta_of[a], 'hit_em': em, 'em_minus_base': ci, 'first_token_agree_base': ft,
                    'tokens_agree_base': tok, 'cap_reached': cap, 'blank': blank, 'distinct_predictions': distinct,
                    'ttft_ms_mean': float(np.mean([by[a][k]['ttft_ms'] for k in keys]))}
        rows.append([a, 'base' if a == 'ours' else ('max' if a == 'shift_max' else delta_of[a]), f'{100 * em:.2f}',
                     '—' if a == 'ours' else ci_str(ci), f'{100 * ft:.1f}', f'{100 * tok:.1f}', f'{100 * cap:.1f}', f'{100 * blank:.1f}', distinct])
    em_by_delta = {curve[a]['delta']: curve[a]['hit_em'] for a in labels if a != 'ours'}
    em_by_delta = {(max(dmax) if a is None else a): v for a, v in em_by_delta.items()}
    read = reading(em_by_delta)
    points = [curve[a]['hit_em'] for a in labels[1:]]
    mono = monotone_non_increasing(points)
    collapse_text = {'abrupt': '급격', 'graded': '점진', 'not_reproduced': '재현 안 됨', 'undefined': '정의 밖'}[read['reading']]

    analysis = {'run': run.name, 'status': status, 'protection': protection.get('status'), 'images': len(images), 'hits_per_arm': 200,
                'gates': {'G_B_positions_verified': [verified, len(hits)], 'G_C_delta0_bitwise': [g_c, 200],
                          'G_D_base_reproduces_prior': [g_d, 200], 'io_identical_across_arms': io_one,
                          'delta_max_gt_1024_all': all(g['delta_max_gt_1024'] for g in gates)},
                'shift_max_reproduces_prior_posshift': [rep_max, 200], 'delta_max_range': [dmax[0], dmax[-1]],
                'curve': curve, 'reading': read, 'monotone_non_increasing_1pp': mono, 'config': config}
    (res / 'analysis.json').write_text(json.dumps(analysis, indent=1, ensure_ascii=False) + '\n')
    head = ['arm', 'delta', 'hit_em_pct', 'em_minus_base_pp', 'em_minus_base_ci_low_pp', 'em_minus_base_ci_high_pp',
            'first_token_agree_base_pct', 'tokens_agree_base_pct', 'cap_reached_pct', 'blank_pct', 'distinct_predictions', 'ttft_ms_mean']
    with (res / 'summary.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(head)
        for a in labels:
            c = curve[a]; ci = c['em_minus_base']
            w.writerow([a, '' if c['delta'] is None else c['delta'], 100 * c['hit_em'], 100 * ci['mean'], 100 * ci['ci_low'], 100 * ci['ci_high'],
                        100 * c['first_token_agree_base'], 100 * c['tokens_agree_base'], 100 * c['cap_reached'], 100 * c['blank'],
                        c['distinct_predictions'], c['ttft_ms_mean']])
    with (res / 'curve.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['image_id', 'turn', 'delta_max'] + labels)
        for i, t in keys:
            w.writerow([i, t, by['shift_max'][(i, t)]['delta']] + [by[a][(i, t)]['accuracy'] for a in labels])

    rep = [f'# PosShift 용량-반응 진단 — run `{run.name}`', '',
           f'계약 `docs/posshift_dose_contract.md`. 40 GQA images × T2–T6 = 200 hit / 조건, Ours store(전체 T1 문맥, 원래 위치)를 '
           f'이미지마다 다시 만들고 text 시작 위치만 `(v0+V) − δ`로 옮겼다. 기준(base)은 `text_position_start=None`의 Ours hit. '
           f'δmax = V − k − n_sep (이 workload에서 {dmax[0]}–{dmax[-1]}) 는 061243Z PosShift와 같은 시작 위치다. '
           '이 실험은 진단이며 arm 우열을 판정하지 않고 061243Z의 사전 판정도 바꾸지 않는다.', '',
           '## 1. Gate', '',
           table(['Gate', '결과'], [
               ['Stages', json.dumps(status)],
               ['G-B 위치 검증 (prefill = t(δ)+arange, decode 연속)', f'{verified} / {len(hits)}'],
               ['G-C δ=0 logits bit-identical to base', f'{g_c} / 200'],
               ['G-D base hit tokens == 061243Z ours rows', f'{g_d} / 200'],
               ['모든 조건 같은 bytes/preads', io_one],
               ['δmax > 1024 (모든 이미지)', analysis['gates']['delta_max_gt_1024_all']],
               ['G-E protection before/after', protection.get('status')],
               ['Report-only: shift_max tokens == 061243Z posshift rows', f'{rep_max} / 200']]), '',
           '## 2. δ별 hit EM 곡선', '',
           table(['조건', 'δ', 'Hit EM %', 'EM − base pp [95% CI]', '첫 토큰 = base %', '토큰열 = base %', '16-token cap %', '빈 출력 %', '서로 다른 출력'], rows), '',
           'CI는 image-cluster paired bootstrap 10,000회(seed 1234). 1 pp 미만 차이는 해석하지 않는다. 지연은 endpoint가 아니다.', '',
           '## 3. 사전 고정 판독', '',
           f'- δ\\* (EM ≤ 5%가 되는 가장 작은 δ): **{read["delta_star"]}** → **{collapse_text}** ({read["reading"]}).',
           f'- 계약 문구: {read["text"]}',
           f'- 보조: 점추정 단조 비증가(1 pp 허용) = {mono}.', '',
           '## 4. 한계', '',
           '- 같은 40-image 개발 세트, 200 hit/조건. 진단 목적의 탐색적 곡선이며 새로운 arm 비교가 아니다.',
           '- base와 δ=0은 같은 store·같은 bytes를 읽고 position_ids만 다르게 들어간다(G-C). store는 다시 만들었고 G-D로 이전 run과의 재현을 확인했다.',
           '- attention 분포는 측정하지 않았다. 붕괴 기전은 출력과 EM 곡선으로만 기술한다.', '']
    (res / 'REPORT.md').write_text('\n'.join(rep))
    (res / 'REPRODUCE.md').write_text('\n'.join([
        '# Reproduce', '', '```bash', 'cd /home/dblab/hj/mllm_v2',
        'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.:scripts:tests PYTHONDONTWRITEBYTECODE=1',
        'PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python', 'RUN=runs/posshift_dose_$(date -u +%Y%m%dT%H%M%SZ)',
        '$PY scripts/150_eval_posshift_dose.py --run $RUN --freeze',
        '$PY -m unittest tests.test_recompute_kv25 tests.test_posshift_dose > $RUN/cpu_tests.log 2>&1',
        '$PY scripts/150_eval_posshift_dose.py --run $RUN --protect before',
        '$PY scripts/150_eval_posshift_dose.py --run $RUN --gpu smoke,gqa',
        '$PY scripts/150_eval_posshift_dose.py --run $RUN --protect after',
        '$PY scripts/151_report_posshift_dose.py --run $RUN', '```', '',
        f'This run: `{run}`. Raw rows `llava/gqa/raw_requests.jsonl` (one row per hit, with `delta`, `text_position_start`, '
        '`position_check`, `delta0_bitwise_equal_base`, `reproduces_prior`); per-image schedule and gate receipts `llava/gqa/gates.jsonl`; '
        'persistence receipts `llava/gqa/persistence.jsonl`. Stores are deleted after each image (`llava/cleanup.jsonl`). '
        'Prior rows come from `runs/recompute_kv25_ablation_20261102T061243Z/llava/gqa/raw_requests.jsonl` '
        f'(sha256 `{config["prior_raw_sha256"]}`, read only). The report script (151) is outside the freeze.', '']))
    print(json.dumps({'reading': read, 'G_C': g_c, 'G_D': g_d, 'rep_max': rep_max}, ensure_ascii=False))
    print((res / 'REPORT.md').read_text())


if __name__ == '__main__':
    main()

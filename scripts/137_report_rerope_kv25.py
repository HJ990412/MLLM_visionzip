#!/usr/bin/env python3
"""Report for the Ours-ReRoPE / Pruned-T1 run: REPORT.md, summary.csv, REPRODUCE.md.

Reads only this run's raw records (plus the rev.-3 and 2026-10-01 raw rows for
the reproduction gates); decision rules come from the frozen contract.
"""
from __future__ import annotations
import argparse, csv, importlib.util, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmimpress.recompute_kv25 import paired_bootstrap
from mmimpress.rerope_kv25 import contrast_bootstrap

spec = importlib.util.spec_from_file_location('r135', ROOT / 'scripts/135_report_recompute_kv25_ablation.py')
r135 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r135)
jl, pct, pp, stats, table = r135.jl, r135.pct, r135.pp, r135.stats, r135.table

ARMS = ['recompute', 'fullload', 'ours', 'orig', 'compact', 'rerope', 'pruned']
NAME = {'recompute': 'ReComp', 'fullload': 'FullLoad', 'ours': 'Ours-KV25', 'orig': 'Recomp-KV25-Orig',
        'compact': 'Recomp-KV25-Compact', 'rerope': 'Ours-ReRoPE', 'pruned': 'Pruned-T1'}
STORED = ['fullload', 'ours', 'orig', 'compact', 'rerope']
KV25 = ['ours', 'orig', 'compact', 'rerope']
PAIRS = [('rerope', 'ours'), ('rerope', 'compact'), ('ours', 'orig'), ('ours', 'compact'), ('orig', 'compact')]
CONTRASTS = {'context': {'ours': .5, 'rerope': .5, 'orig': -.5, 'compact': -.5},
             'position': {'compact': .5, 'rerope': .5, 'ours': -.5, 'orig': -.5},
             'interaction': {'ours': 1, 'rerope': -1, 'orig': -1, 'compact': 1}}
SIGN = {'context': 'full context higher than reduced (averaged over positions)',
        'position': 'contiguous positions higher than original (averaged over contexts)',
        'interaction': 'original-minus-contiguous gap larger in full context than in reduced context'}
REV3 = ROOT / 'runs/recompute_kv25_ablation_20261102T061243Z'
REF = ROOT / 'runs/prefix25_persistence_20261001T050113Z'
MB = 1e6


def verdict(ci, a, b):
    lo, hi, m = ci['ci_low'], ci['ci_high'], ci['mean']
    if lo > 0:
        base = f'{NAME[a]} higher (CI lower bound > 0)'
    elif hi < 0:
        base = f'{NAME[b]} higher (CI upper bound < 0)'
    else:
        return 'difference not shown (CI includes 0; not equivalence)'
    return base + ('; |difference| < 1 pp, not interpreted as established' if abs(m) < 0.01 else '')


def shown(ci):
    if ci['ci_low'] > 0 or ci['ci_high'] < 0:
        return 'shown' + ('; |effect| < 1 pp, not interpreted' if abs(ci['mean']) < 0.01 else '')
    return 'not shown (CI includes 0)'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); args = ap.parse_args()
    run = Path(args.run).resolve(); res = ROOT / 'results' / run.name; res.mkdir(exist_ok=True)
    base = run / 'llava'
    raw = jl(base / 'gqa/raw_requests.jsonl')
    assert not [r for r in raw if r.get('status') != 'OK'], 'failed raw rows'
    images = sorted({r['image_id'] for r in raw}); assert len(images) == 40
    by = {(r['arm'], r['image_id'], r['turn']): r for r in raw}
    assert len(by) == len(raw) == 40 * 6 * 7, (len(by), len(raw))
    hits = {a: [by[(a, i, t)] for i in images for t in range(2, 7)] for a in ARMS}
    allrows = {a: [by[(a, i, t)] for i in images for t in range(1, 7)] for a in ARMS}
    persist = jl(base / 'gqa/persistence.jsonl')
    gates = [g for g in jl(base / 'gates.jsonl') if g['phase'] == 'gqa']
    rgates = [g for g in jl(base / 'gates_rerope.jsonl') if g['phase'] == 'gqa']
    gpu = json.loads((base / 'gpu_correctness.json').read_text())
    gpu_ext = json.loads((base / 'gpu_correctness_extension.json').read_text())
    status = json.loads((base / 'status.json').read_text())
    protection = json.loads((run / 'protection_result.json').read_text()) if (run / 'protection_result.json').exists() else {'status': 'NOT RUN'}

    # ------------------------------------------------------------ quality
    quality = {a: {'hit_accuracy': float(np.mean([r['accuracy'] for r in hits[a]])),
                   'all_accuracy': float(np.mean([r['accuracy'] for r in allrows[a]])),
                   't1_accuracy': float(np.mean([by[(a, i, 1)]['accuracy'] for i in images]))} for a in ARMS}
    per_hit = {a: np.array([np.mean([by[(a, i, t)]['accuracy'] for t in range(2, 7)]) for i in images]) for a in ARMS}
    paired = {}
    for a, b in PAIRS:
        ci = paired_bootstrap([(sum(by[(a, i, t)]['accuracy'] for t in range(2, 7)), sum(by[(b, i, t)]['accuracy'] for t in range(2, 7)), 5) for i in images])
        disc = [(x['accuracy'], y['accuracy']) for x, y in zip(hits[a], hits[b])]
        ci.update({'a_only_correct': sum(1 for x, y in disc if x > y), 'b_only_correct': sum(1 for x, y in disc if y > x), 'verdict': verdict(ci, a, b)})
        paired[f'{a}-{b}'] = ci
    effects = contrast_bootstrap({a: per_hit[a] for a in KV25}, CONTRASTS)
    for k, v in effects.items():
        v['sign'] = SIGN[k]; v['reading'] = shown(v)
    agree = {a: {'first_token': float(np.mean([x['first_token_id'] == y['first_token_id'] for x, y in zip(hits[a], hits['fullload'])])),
                 'prediction': float(np.mean([x['prediction'] == y['prediction'] for x, y in zip(hits[a], hits['fullload'])]))} for a in ARMS}
    # Pruned-T1 vs full-token standalone (ReComp) on (a) T1, (b) T2–T6, (c) all
    pruned = {}
    for label, turns in (('a_T1', [1]), ('b_T2_T6', list(range(2, 7))), ('c_all', list(range(1, 7)))):
        ci = paired_bootstrap([(sum(by[('pruned', i, t)]['accuracy'] for t in turns), sum(by[('recompute', i, t)]['accuracy'] for t in turns), len(turns)) for i in images])
        ci.update({'n': 40 * len(turns), 'pruned_accuracy': float(np.mean([by[('pruned', i, t)]['accuracy'] for i in images for t in turns])),
                   'recompute_accuracy': float(np.mean([by[('recompute', i, t)]['accuracy'] for i in images for t in turns])), 'reading': shown(ci)})
        pruned[label] = ci
    # dialogue-level designs (1 T1 + 5 hits per image)
    design = {'full_t1_ours_hits': {i: (by[('recompute', i, 1)]['accuracy'] + sum(by[('ours', i, t)]['accuracy'] for t in range(2, 7))) / 6 for i in images},
              'full_t1_rerope_hits': {i: (by[('recompute', i, 1)]['accuracy'] + sum(by[('rerope', i, t)]['accuracy'] for t in range(2, 7))) / 6 for i in images},
              'all_pruned': {i: sum(by[('pruned', i, t)]['accuracy'] for t in range(1, 7)) / 6 for i in images}}
    dnames = list(design)
    dvals = {d: np.array([design[d][i] for i in images]) for d in dnames}
    dcontr = {d: {d: 1} for d in dnames}
    dcontr.update({f'{a}-{b}': {a: 1, b: -1} for j, a in enumerate(dnames) for b in dnames[j + 1:]})
    dialogue = contrast_bootstrap(dvals, dcontr)
    for k, v in dialogue.items():
        v['reading'] = shown(v) if '-' in k else ''
    t1_full_identical = all(len({json.dumps((by[(a, i, 1)]['first_token_id'], by[(a, i, 1)]['generated_token_ids'])) for a in ['recompute', 'fullload', 'ours', 'orig', 'compact']}) == 1 for i in images)

    # ------------------------------------------------------- latency / IO
    latency = {a: {'ttft': stats([r['ttft_ms'] for r in hits[a]]), 'e2e_mean': float(np.mean([r['request_e2e_ms'] for r in hits[a]])),
                   'read_mb': float(np.mean([r['read_bytes'] for r in hits[a]])) / MB, 'preads': float(np.mean([r['preads'] for r in hits[a]]))} for a in ARMS}
    io_equal = all(len({(by[(a, i, t)]['read_bytes'], by[(a, i, t)]['preads']) for a in KV25}) == 1 for i in images for t in range(2, 7))
    ttft_paired = {f'{a}-ours': paired_bootstrap([(sum(by[(a, i, t)]['ttft_ms'] for t in range(2, 7)), sum(by[('ours', i, t)]['ttft_ms'] for t in range(2, 7)), 5) for i in images])
                   for a in ['orig', 'compact', 'rerope']}
    pruned_breakdown = {k: float(np.mean([r['ttft_breakdown_ms'][k] for r in allrows['pruned']])) for k in allrows['pruned'][0]['ttft_breakdown_ms']}
    pruned_breakdown['ttft_total'] = float(np.mean([r['ttft_ms'] for r in allrows['pruned']]))

    # ------------------------------------------------------- persistence
    P = defaultdict(dict)
    for p in persist:
        P[p['arm']][p['image_id']] = p

    def comp(p):
        t = p['timing_ms']; rc = p.get('recompute') or {}; rr = p.get('rerope') or {}
        return {'saliency_ms': p.get('saliency_ms') or 0.0, 'permutation_ms': t.get('permutation_ms', 0.0), 'plan_ms': rc.get('plan_ms', rr.get('plan_ms', 0.0)),
                'embed_gather_ms': rc.get('embed_gather_ms', 0.0), 'reduced_prefill_ms': rc.get('reduced_prefill_ms', 0.0),
                'synthetic_assembly_ms': rc.get('synthetic_assembly_ms', 0.0), 'rerope_ms': rr.get('rerope_ms', 0.0),
                'repack_ms': t.get('repack_ms', 0.0), 'write_ms': t.get('ssd_write_ms', t.get('store_write_ms', 0.0)), 'fsync_ms': t.get('durability_ms', 0.0),
                'seal_ms': p['hash_and_seal_ms'], 'total_ms': p['persistence_ms'], 'file_mb': p['file_logical_bytes'] / MB}
    persistence = {a: {k: float(np.mean([comp(P[a][i])[k] for i in images])) for k in comp(P[a][images[0]])} for a in STORED}
    delta = {}
    for a, key in (('orig', 'prefill_only_ms'), ('compact', 'prefill_only_ms'), ('rerope', 'rerope_ms')):
        prim = [(P[a][i]['recompute'] or P[a][i]['rerope'])[key] for i in images]
        full = paired_bootstrap([(P[a][i]['persistence_ms'], P['ours'][i]['persistence_ms'], 1) for i in images])
        delta[a] = {'primary': key, 'primary_mean_ms': float(np.mean(prim)), 'primary_p95_ms': float(np.percentile(prim, 95)), 'full_delta': full,
                    'amortized_primary': {n: float(np.mean(prim)) / n for n in (1, 3, 5, 10)}, 'amortized_full': {n: full['mean'] / n for n in (1, 3, 5, 10)}}

    # ----------------------------------------------------------- KV diffs
    L = len(gates[0]['G2']['orig']['layers'])
    layer_rows = []
    for li in range(L):
        row = {'layer': li}
        for key in ['k_S', 'v_S']:
            row[f'orig_{key}_vs_ours_rel_l2'] = float(np.mean([g['G2']['orig']['layers'][li][key]['rel_l2'] for g in gates]))
            row[f'compact_{key}_vs_ours_rel_l2'] = float(np.mean([g['G2']['compact']['layers'][li][key]['rel_l2'] for g in gates]))
        row['compact_k_S_realigned_vs_ours_rel_l2'] = float(np.mean([g['G2']['compact']['layers'][li]['k_S_realigned']['rel_l2'] for g in gates]))
        for key in ['k_S_vs_compact', 'v_S_vs_compact', 'k_S_vs_ours_unrotated', 'k_sep_vs_compact']:
            row[f'rerope_{key}_rel_l2'] = float(np.mean([g['Gab']['layers'][li][key]['rel_l2'] for g in rgates]))
        layer_rows.append(row)
    gab = {'images': len(rgates), 'pass': sum(g['status'] == 'PASS' for g in rgates),
           'roundtrip_max_rel_l2': max(x['rel_l2'] for g in rgates for n in ('S', 'sep') for x in g['Gab']['gates'][f'roundtrip_k_{n}']),
           'layer0_vs_compact_max_rel_l2': max(g['Gab']['gates'][f'same_position_layer0_k_{n}'][0]['rel_l2'] for g in rgates for n in ('S', 'sep')),
           'oracle_ratio_max': max(g['Gab']['gates'][f'same_position_layer0_k_{n}'][0]['oracle_ratio'] for g in rgates for n in ('S', 'sep')),
           'neg_unrotated_min_rel_l2': min(g['Gab']['gates'][f'neg_unrotated_layer0_k_{n}'][0]['rel_l2'] for g in rgates for n in ('S', 'sep')),
           'v_sys_bitident_all': all(x['ok'] for g in rgates for n in ('v_bitident_S', 'v_bitident_sep', 'sys_k_bitident', 'sys_v_bitident') for x in g['Gab']['gates'][n]),
           'rerope_ms_mean': persistence['rerope']['rerope_ms']}

    # ------------------------------------------------------- reproduction
    def repro_against(ref_rows, arms_map, label):
        out = {}
        for new, old in arms_map:
            rows = [(r, ref_rows[(old, r['image_id'], r['question_id'])]) for r in allrows[new] if (old, r['image_id'], r['question_id']) in ref_rows]
            mism = [{'image_id': r['image_id'], 'question_id': r['question_id'], 'turn': r['turn'], 'new_prediction': r['prediction'], 'ref_prediction': o['prediction'],
                     'new_first': r['first_token_id'], 'ref_first': o['first_token_id'], 'new_top2_margin': (r.get('first_token_top2') or {}).get('margin'), 'gold': r['gold']}
                    for r, o in rows if r['generated_token_ids'] != o['generated_token_ids'] or r['prediction'] != o['prediction']]
            out[new] = {'reference': label, 'reference_arm': old, 'n': len(rows),
                        'first_token_agreement': float(np.mean([r['first_token_id'] == o['first_token_id'] for r, o in rows])),
                        'prediction_agreement': float(np.mean([r['prediction'] == o['prediction'] for r, o in rows])),
                        'sequence_agreement': float(np.mean([r['generated_token_ids'] == o['generated_token_ids'] for r, o in rows])), 'mismatches': mism}
        return out
    rev3 = {(r['arm'], r['image_id'], str(r['question_id'])): r for r in jl(REV3 / 'llava/gqa/raw_requests.jsonl') if r.get('status') == 'OK'}
    ref = {(r['arm'], r['image_id'], str(r['question_id'])): r for r in jl(REF / 'llava/gqa/raw_requests.jsonl') if r.get('status') == 'OK'}
    repro_rev3 = repro_against(rev3, [(a, a) for a in ['ours', 'orig', 'compact', 'fullload', 'recompute']], 'rev3')
    repro_ref = repro_against(ref, [('ours', 'B')], 'prefix25_20261001')
    # G-f Pruned-T1 vs Compact hits
    gf_rows = [(by[('pruned', i, t)], by[('compact', i, t)]) for i in images for t in range(2, 7)]
    gf = {'n': len(gf_rows), 'first_token_agreement': float(np.mean([p['first_token_id'] == c['first_token_id'] for p, c in gf_rows])),
          'prediction_agreement': float(np.mean([p['prediction'] == c['prediction'] for p, c in gf_rows])),
          'mismatches': [{'image_id': p['image_id'], 'question_id': p['question_id'], 'turn': p['turn'], 'pruned': p['prediction'], 'compact': c['prediction'],
                          'pruned_first': p['first_token_id'], 'compact_first': c['first_token_id'], 'gold': p['gold'],
                          'pruned_top2_margin': (p.get('first_token_top2') or {}).get('margin'), 'compact_top2_margin': (c.get('first_token_top2') or {}).get('margin')}
                         for p, c in gf_rows if p['first_token_id'] != c['first_token_id'] or p['prediction'] != c['prediction']]}

    # ----------------------------------------------------------- gates
    g1 = all(g['G1'] == 'PASS' for g in gates); g2 = all(g['G2_status'] == 'PASS' for g in gates)
    pos = all(r['position_check']['verified'] for a in KV25 + ['pruned'] for r in allrows[a] if r.get('position_check'))
    ts = lambda a, i, t: by[(a, i, t)]['position_check']['text_position_start']
    ts_ok = all(ts('compact', i, t) == ts('rerope', i, t) < ts('ours', i, t) == ts('orig', i, t) for i in images for t in range(2, 7))
    pruned_sel = all(r['selected_original_ids'] == by[('ours', r['image_id'], 2)]['selected_original_ids'] for r in allrows['pruned'])
    pruned_zero = all(r['read_bytes'] == 0 and r['vision_calls'] == 1 for r in allrows['pruned'])
    gate_rows = [('G1 selection/bytes/embeds/T1 identity (rev. 3)', 'PASS' if g1 else 'FAIL'),
                 ('G2 rev. 3 structural bit-identity + layer-0 S/sep (Ours/Orig/Compact)', 'PASS' if g2 else 'FAIL'),
                 ('G-a round trip ReRoPE K -> Ours K (all layers)', f"PASS ({gab['pass']}/{gab['images']}; max rel L2 {gab['roundtrip_max_rel_l2']:.2e})" if gab['pass'] == 40 else 'FAIL'),
                 ('G-b layer-0 ReRoPE K vs Compact K; V/sys bit-identity; negative control',
                  f"PASS (max rel L2 {gab['layer0_vs_compact_max_rel_l2']:.4f} ≤ 0.01; oracle ratio ≤ {gab['oracle_ratio_max']:.3f}; un-rotated ≥ {gab['neg_unrotated_min_rel_l2']:.2f}; V/sys bit-identical {gab['v_sys_bitident_all']})" if gab['pass'] == 40 else 'FAIL'),
                 ('G-c text start / positions (all requests)', 'PASS' if pos and ts_ok else 'FAIL'),
                 ('G-d identity, equal hit reads (4 KV25 arms), Pruned S identity, zero reads', 'PASS' if io_equal and pruned_sel and pruned_zero else 'FAIL'),
                 ('G-e reproduction vs rev. 3 (report-only)', '; '.join(f"{NAME[a]} {pct(v['sequence_agreement'])}" for a, v in repro_rev3.items())),
                 ('G-f Pruned-T1 vs Compact hits (report-only)', f"first token {pct(gf['first_token_agreement'])}, prediction {pct(gf['prediction_agreement'])}, {len(gf['mismatches'])} mismatches"),
                 ('G4 GPU correctness (rev. 3 (a)–(e) + extension (f)–(h))', f"{gpu['status']} / {gpu_ext['status']}"),
                 ('G5 protection before/after', protection.get('status', 'NOT RUN')),
                 ('G6 Ours vs 2026-10-01 reference', f"{pct(repro_ref['ours']['sequence_agreement'])} sequence agreement"),
                 ('G7 sentinel non-leakage (smoke)', status.get('SMOKE', 'NOT RUN'))]

    # ---------------------------------------------------------- outputs
    summary = []
    for a in ARMS:
        row = {'arm': NAME[a], 'hit_accuracy': quality[a]['hit_accuracy'], 't1_accuracy': quality[a]['t1_accuracy'], 'all_accuracy': quality[a]['all_accuracy'],
               'fullload_first_agreement': agree[a]['first_token'], 'fullload_prediction_agreement': agree[a]['prediction'],
               'hit_ttft_mean_ms': latency[a]['ttft']['mean'], 'hit_ttft_p50_ms': latency[a]['ttft']['p50'], 'hit_ttft_p95_ms': latency[a]['ttft']['p95'],
               'hit_e2e_mean_ms': latency[a]['e2e_mean'], 'hit_read_mb': latency[a]['read_mb'], 'hit_preads': latency[a]['preads'],
               'persistence_total_ms': persistence.get(a, {}).get('total_ms'), 'extra_cost_primary_ms': delta.get(a, {}).get('primary_mean_ms'),
               'ttft_is_composed_not_for_efficiency': a == 'pruned'}
        for key, (x, y) in {'rerope_minus_ours': ('rerope', 'ours'), 'rerope_minus_compact': ('rerope', 'compact')}.items():
            if a in (x, y) and f'{x}-{y}' in paired:
                d = paired[f'{x}-{y}']; row.update({f'{key}_pp': 100 * d['mean'], f'{key}_ci_low_pp': 100 * d['ci_low'], f'{key}_ci_high_pp': 100 * d['ci_high']})
        summary.append(row)
    keys = []
    for r in summary:
        keys += [k for k in r if k not in keys]
    with (res / 'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(summary)
    with (res / 'kv_layer_diff.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(layer_rows[0])); w.writeheader(); w.writerows(layer_rows)
    with (res / 'persistence.csv').open('w', newline='') as f:
        cols = list(next(iter(persistence.values()))); w = csv.writer(f); w.writerow(['arm'] + cols)
        for a, v in persistence.items():
            w.writerow([NAME[a]] + [v[c] for c in cols])
    analysis = {'quality': quality, 'paired_accuracy': paired, 'factor_effects': effects, 'pruned_vs_recompute': pruned, 'dialogue_designs': dialogue,
                't1_full_token_identical_across_arms': t1_full_identical, 'fullload_agreement': agree, 'latency': latency, 'ttft_paired_vs_ours': ttft_paired,
                'pruned_ttft_breakdown_ms': pruned_breakdown, 'kv25_io_equal': io_equal, 'persistence': persistence, 'extra_cost': delta, 'gab': gab,
                'reproduction_rev3': repro_rev3, 'reproduction_prefix25': repro_ref, 'gf_pruned_vs_compact': gf, 'gates': dict(gate_rows), 'status': status,
                'protection': {k: v for k, v in protection.items() if k != 'differences'}}
    (res / 'analysis.json').write_text(json.dumps(analysis, indent=1) + '\n')

    q = lambda a: pct(quality[a]['hit_accuracy'])
    cell = lambda a: f"{q(a)} ({NAME[a]})"
    rep = ['# Ours-ReRoPE and Pruned-T1 (LLaVA, GQA Stage 1)', '',
           f'Run: `runs/{run.name}`. Contract: [CONTRACT.md](CONTRACT.md) (frozen before GPU execution; G-a/G-b tolerances fixed from the '
           'rev.-3 G4 stores, evidence [dev/g_ab_evidence.json](dev/g_ab_evidence.json)). Rev.-3 arms Orig and Compact were re-run in this run; '
           'the rev.-3 run is only used for the reproduction gate G-e.', '',
           '**All conclusions are exploratory**: the 40-image GQA development set has been used repeatedly (rev. 3 and earlier runs) and the ReRoPE arm '
           'was motivated by the rev.-3 result on the same images. Differences below 1 pp are not interpreted as established (BF16 shape rounding can '
           'flip boundary first tokens; rev.-3 noise floor).', '',
           'Workload: 40 GQA images × `questions[4:10]` (manifest `40e23fa7…`), T1 = `questions[4]`, 200 cache hits (T2–T6) per stored arm; '
           'Pruned-T1 answers all 240 questions as standalone pruned first turns. LLaVA-NeXT-Vicuna-7B NF4/BF16/eager, FP16 KV, 64-token chunks, '
           'separator sidecar, greedy max 16, cold page cache (DONTNEED) outside timers. ReRoPE shares the Ours T1 capture (its T1 row is the Ours T1).', '',
           '## Gates', '', table(['Gate', 'Result'], gate_rows), '',
           '## 2×2: KV context × position (cache-hit accuracy, T2–T6)', '',
           table(['Context \\ Position', 'Original positions', 'Contiguous positions'],
                 [['Full T1 context', cell('ours'), cell('rerope')], ['Reduced prefill context', cell('orig'), cell('compact')]]), '',
           f"Baselines: FullLoad {q('fullload')}, ReComp {q('recompute')}. Prediction agreement with FullLoad: "
           + ', '.join(f"{NAME[a]} {pct(agree[a]['prediction'])}" for a in KV25) + '.', '',
           '### Paired differences (image-cluster bootstrap, 10,000 resamples, seed 1234)', '',
           table(['Difference', 'Mean', '95% CI', 'a-only / b-only correct', 'Pre-registered reading'],
                 [[f'{NAME[a]} − {NAME[b]}', pp(paired[f'{a}-{b}']['mean']), f"[{pp(paired[f'{a}-{b}']['ci_low'])}, {pp(paired[f'{a}-{b}']['ci_high'])}]",
                   f"{paired[f'{a}-{b}']['a_only_correct']} / {paired[f'{a}-{b}']['b_only_correct']}", paired[f'{a}-{b}']['verdict']] for a, b in PAIRS]), '',
           '### Factor effects (joint image-level bootstrap, same resamples for all three)', '',
           table(['Effect', 'Contrast', 'Positive means', 'Estimate', '95% CI', 'Reading'],
                 [[k, {'context': '((Ours + ReRoPE) − (Orig + Compact)) / 2', 'position': '((Compact + ReRoPE) − (Ours + Orig)) / 2',
                       'interaction': '(Ours − ReRoPE) − (Orig − Compact)'}[k], SIGN[k], pp(v['mean']), f"[{pp(v['ci_low'])}, {pp(v['ci_high'])}]", v['reading']]
                  for k, v in effects.items()]), '',
           '## Decisions (rules fixed in the contract before results)', '',
           f"- **ReRoPE − Ours**: {paired['rerope-ours']['verdict']} (mean {pp(paired['rerope-ours']['mean'])}, CI [{pp(paired['rerope-ours']['ci_low'])}, {pp(paired['rerope-ours']['ci_high'])}]).",
           f"- **ReRoPE − Compact**: {paired['rerope-compact']['verdict']} (mean {pp(paired['rerope-compact']['mean'])}, CI [{pp(paired['rerope-compact']['ci_low'])}, {pp(paired['rerope-compact']['ci_high'])}]).",
           f"- **Context effect** {effects['context']['reading']}; **position effect** {effects['position']['reading']}; **interaction** {effects['interaction']['reading']}.",
           f"- ReRoPE needs no extra prefill: its cost is {persistence['rerope']['rerope_ms']:.2f} ms re-rotation inside persistence (vs {persistence['orig']['reduced_prefill_ms']:.2f} ms reduced prefill for Recomp).", '',
           '## Pruned-T1 (VisionZip-style first turn, no cache) vs full-token standalone inference (ReComp)', '',
           table(['Questions', 'n', 'Pruned-T1', 'ReComp', 'Pruned − ReComp', '95% CI', 'Reading'],
                 [[{'a_T1': '(a) T1', 'b_T2_T6': '(b) T2–T6', 'c_all': '(c) all'}[k], v['n'], pct(v['pruned_accuracy']), pct(v['recompute_accuracy']),
                   pp(v['mean']), f"[{pp(v['ci_low'])}, {pp(v['ci_high'])}]", v['reading']] for k, v in pruned.items()]), '',
           f"G-f: Pruned-T1 T2–T6 vs the Compact hit of the same question: first-token agreement {pct(gf['first_token_agreement'])}, prediction agreement "
           f"{pct(gf['prediction_agreement'])} ({len(gf['mismatches'])} of {gf['n']} differ; the pruned prefill contains the question tokens in the same GEMM and the "
           'Compact hit goes through the FP16 store and the masked full-size buffer).', '',
           '### Dialogue-level expected accuracy (1 T1 + 5 hits per image)', '',
           table(['Design', 'Mean accuracy', '95% CI'],
                 [[{'full_t1_ours_hits': 'T1 full-token + Ours-KV25 hits', 'full_t1_rerope_hits': 'T1 full-token + Ours-ReRoPE hits', 'all_pruned': 'All Pruned-T1'}[d],
                   pct(dialogue[d]['mean']), f"[{pct(dialogue[d]['ci_low'])}, {pct(dialogue[d]['ci_high'])}]"] for d in dnames]), '',
           table(['Difference', 'Mean', '95% CI', 'Reading'],
                 [[k.replace('full_t1_ours_hits', 'T1 full + Ours').replace('full_t1_rerope_hits', 'T1 full + ReRoPE').replace('all_pruned', 'All Pruned').replace('-', ' − '),
                   pp(v['mean']), f"[{pp(v['ci_low'])}, {pp(v['ci_high'])}]", v['reading']] for k, v in dialogue.items() if '-' in k]), '',
           f"\"T1 full-token\" is the ReComp T1 row; every full-token T1 (ReComp, FullLoad, Ours, Orig, Compact) produced identical first tokens and sequences: {t1_full_identical}.", '',
           '## Cache-hit latency and SSD reads', '',
           table(['Arm', 'TTFT mean / p50 / p95 (ms)', 'E2E mean (ms)', 'Read MB/hit', 'preads/hit'],
                 [[NAME[a], f"{latency[a]['ttft']['mean']:.2f} / {latency[a]['ttft']['p50']:.2f} / {latency[a]['ttft']['p95']:.2f}", f"{latency[a]['e2e_mean']:.2f}",
                   f"{latency[a]['read_mb']:.3f}", f"{latency[a]['preads']:.1f}"] for a in ARMS if a != 'pruned']), '',
           f"The four KV25 arms read identical bytes and preads on every hit: {'yes' if io_equal else 'NO'}. "
           + '; '.join(f"{NAME[a]} − Ours TTFT {ttft_paired[f'{a}-ours']['mean']:+.2f} ms [{ttft_paired[f'{a}-ours']['ci_low']:+.2f}, {ttft_paired[f'{a}-ours']['ci_high']:+.2f}]" for a in ['orig', 'compact', 'rerope']) + '.', '',
           f"Pruned-T1 request time (mean over 240 requests): TTFT {pruned_breakdown['ttft_total']:.1f} ms = image decode {pruned_breakdown['image_decode']:.1f} + processor/H2D "
           f"{pruned_breakdown['processor_and_h2d']:.1f} + vision/projector/merge {pruned_breakdown['vision_projector_merge']:.1f} + saliency/order/plan "
           f"{pruned_breakdown['saliency_order_plan']:.1f} + gather {pruned_breakdown['gather']:.1f} + pruned prefill {pruned_breakdown['prefill_to_first_token']:.1f}. "
           '**This is a composed measurement (the full-token forward is aborted after the multimodal merge, not the schema-v2 server path) and is not used for any '
           'efficiency comparison.**', '',
           '## Persistence cost per image (mean ms; not part of cache-hit TTFT)', '',
           table(['Arm', 'Saliency', 'Permutation', 'Plan', 'Embed gather', 'Reduced prefill', 'Synthetic assembly', '**Re-rotation**', 'Repack', 'Write', 'fsync', 'Seal', 'Total', 'File MB'],
                 [[NAME[a]] + [f"{persistence[a][k]:.2f}" for k in ['saliency_ms', 'permutation_ms', 'plan_ms', 'embed_gather_ms', 'reduced_prefill_ms', 'synthetic_assembly_ms',
                                                                   'rerope_ms', 'repack_ms', 'write_ms', 'fsync_ms', 'seal_ms', 'total_ms', 'file_mb']] for a in STORED]), '',
           table(['Arm', 'Primary extra cost', 'mean / p95 (ms)', '÷1', '÷3', '÷5', '÷10', 'Full Δ vs Ours [95% CI]'],
                 [[NAME[a], delta[a]['primary'], f"{delta[a]['primary_mean_ms']:.2f} / {delta[a]['primary_p95_ms']:.2f}"] + [f"{delta[a]['amortized_primary'][n]:.2f}" for n in (1, 3, 5, 10)]
                  + [f"{delta[a]['full_delta']['mean']:+.1f} [{delta[a]['full_delta']['ci_low']:+.1f}, {delta[a]['full_delta']['ci_high']:+.1f}]"] for a in ['orig', 'compact', 'rerope']]), '',
           'ReRoPE shares the Ours T1 capture; its persistence re-runs saliency/permutation on that capture, so the full Δ also contains writer wall-time noise. '
           'The re-rotation (`rerope_ms`, FP64, 32 layers) is the only compute the ReRoPE store needs beyond Ours.', '',
           '## Per-layer KV difference (mean over 40 images, S rows, rel L2)', '',
           table(['Layer', 'Orig vs Ours K / V (context, original pos.)', 'ReRoPE vs Compact K / V (context, contiguous pos.)', 'ReRoPE vs Ours K (realignment only)', 'Compact realigned vs Ours K'],
                 [[r['layer'], f"{r['orig_k_S_vs_ours_rel_l2']:.4f} / {r['orig_v_S_vs_ours_rel_l2']:.4f}", f"{r['rerope_k_S_vs_compact_rel_l2']:.4f} / {r['rerope_v_S_vs_compact_rel_l2']:.4f}",
                   f"{r['rerope_k_S_vs_ours_unrotated_rel_l2']:.4f}", f"{r['compact_k_S_realigned_vs_ours_rel_l2']:.4f}"] for r in layer_rows if r['layer'] in (0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 31)]), '',
           'Full table: [kv_layer_diff.csv](kv_layer_diff.csv).', '',
           '## Reproduction (G-e vs rev. 3; G6 vs 2026-10-01)', '',
           table(['New arm', 'Reference', 'n', 'First-token agr.', 'Prediction agr.', 'Sequence agr.'],
                 [[NAME[a], f"{v['reference']} / {v['reference_arm']}", v['n'], pct(v['first_token_agreement']), pct(v['prediction_agreement']), pct(v['sequence_agreement'])]
                  for a, v in list(repro_rev3.items()) + list(repro_ref.items())]), '']
    for a, v in list(repro_rev3.items()) + list(repro_ref.items()):
        if v['mismatches']:
            rep.append(f"{NAME[a]} vs {v['reference']} mismatches ({len(v['mismatches'])}):")
            rep += [f"- {m['image_id']}/{m['question_id']} T{m['turn']}: new `{m['new_prediction']}` ({m['new_first']}, margin {m['new_top2_margin']}) vs reference `{m['ref_prediction']}` ({m['ref_first']}), gold `{m['gold']}`" for m in v['mismatches']]
            rep.append('')
    if gf['mismatches']:
        rep.append(f"G-f Pruned-T1 vs Compact mismatches ({len(gf['mismatches'])}), with first-token top-2 logit margins:")
        rep += [f"- {m['image_id']}/{m['question_id']} T{m['turn']}: pruned `{m['pruned']}` ({m['pruned_first']}, margin {m['pruned_top2_margin']:.3f}) vs compact `{m['compact']}` ({m['compact_first']}, margin {m['compact_top2_margin']:.3f}), gold `{m['gold']}`" for m in gf['mismatches']]
        rep.append('')
    rep += ['## Limits', '',
            '- Exploratory: 40 image clusters of a development set used several times; the ReRoPE hypothesis came from the rev.-3 result on the same images; 200 hits per arm; Stage 2 not run.',
            '- LLaVA only. Recomp arms and Pruned-T1 use image-only saliency without contextual merge, so they isolate KV context and position rule, not the full VisionZip system.',
            '- Pruned-T1 TTFT is a composed measurement and is not an efficiency result.',
            '- Latency is single-GPU, batch 1, page-cache DONTNEED; no NAND/controller coldness claim.',
            '- G-a/G-b tolerances were fixed from rev.-3 stores before this run; G-e and G-f are report-only.', '']
    (res / 'REPORT.md').write_text('\n'.join(rep) + '\n')
    (res / 'REPRODUCE.md').write_text(f'''# Reproduction

Environment: `conda activate mllm_ft` (torch 2.5.1, transformers 4.57.6), RTX 4090, no other GPU process.
Use a fresh timestamped run name; nothing is overwritten (no-clobber creation).

```bash
RUN=runs/rerope_kv25_$(date -u +%Y%m%dT%H%M%SZ)
python scripts/136_eval_rerope_kv25.py --run $RUN --freeze
PYTHONPATH=.:tests python -m unittest -v test_rerope_kv25 test_recompute_kv25 test_prefix25_persistence test_llava_kv25 \\
    test_image_only_repack test_visdial_turn1_piggyback_core > $RUN/cpu_tests.log 2>&1
python scripts/136_eval_rerope_kv25.py --run $RUN --protect before
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u scripts/136_eval_rerope_kv25.py --run $RUN --gpu correctness,smoke
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u scripts/136_eval_rerope_kv25.py --run $RUN --gpu gqa
python scripts/136_eval_rerope_kv25.py --run $RUN --protect after
python scripts/137_report_rerope_kv25.py --run $RUN
```

Inputs are copied from `runs/recompute_kv25_ablation_20261102T061243Z` (GQA/smoke/correctness manifests, GQA manifest
`40e23fa7…`) and verified by hash at freeze; the G-a/G-b evidence file is copied to `dev/` and its SHA256 is in the contract.
`gpu_freeze.json` binds every source file except the report scripts. Raw rows: `runs/{run.name}/llava/gqa/raw_requests.jsonl`
(7 arms × 6 turns × 40 images); persistence receipts `persistence.jsonl`; rev.-3 G1/G2 receipts `llava/gates.jsonl`; ReRoPE
G-a/G-b receipts and per-layer differences `llava/gates_rerope.jsonl`; G4 evidence `llava/gpu_correctness.json` and
`llava/gpu_correctness_extension.json`. Stage-1 and smoke payload stores are deleted per image after their receipts
(guarded to `llava/stores/`; `llava/cleanup.jsonl` keeps each store's sealed per-file SHA256 list); G4 stores are retained.
''')
    print((res / 'REPORT.md').read_text()[:2500])


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Report for the recompute-KV25 ablation: REPORT.md, summary.csv, REPRODUCE.md.

Reads only this run's raw records (and the reference run's raw rows for the
reproduction gate); decision rules come from the frozen contract.
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmimpress.recompute_kv25 import paired_bootstrap

ARMS = ['recompute', 'fullload', 'ours', 'orig', 'compact', 'posshift']
NAME = {'recompute': 'ReComp', 'fullload': 'FullLoad', 'ours': 'Ours-KV25', 'orig': 'Recomp-KV25-Orig',
        'compact': 'Recomp-KV25-Compact', 'posshift': 'Ours-KV25-PosShift'}
PAIRS = [('ours', 'orig'), ('ours', 'compact'), ('ours', 'posshift'), ('orig', 'compact')]
REF_RUN = ROOT / 'runs/prefix25_persistence_20261001T050113Z'
MB = 1e6


def jl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def pct(x):
    return f'{100 * x:.2f}%'


def pp(x):
    return f'{100 * x:+.2f} pp'


def stats(v):
    v = np.asarray(v, dtype=float)
    return {'mean': float(v.mean()), 'p50': float(np.percentile(v, 50)), 'p95': float(np.percentile(v, 95))}


def table(head, rows):
    out = ['| ' + ' | '.join(head) + ' |', '|' + '|'.join(['---'] + ['---:'] * (len(head) - 1)) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def verdict(ci, a, b):
    lo, hi, m = ci['ci_low'], ci['ci_high'], ci['mean']
    small = abs(m) < 0.01
    if lo > 0:
        base = f'{NAME[a]} higher (CI lower bound > 0)'
    elif hi < 0:
        base = f'{NAME[b]} higher (CI upper bound < 0)'
    else:
        return 'difference not shown (CI includes 0; not equivalence)'
    return base + ('; |difference| < 1 pp, not interpreted as established' if small else '')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); args = ap.parse_args()
    run = Path(args.run).resolve(); res = ROOT / 'results' / run.name; res.mkdir(exist_ok=True)
    base = run / 'llava'
    raw = [r for r in jl(base / 'gqa/raw_requests.jsonl')]
    fails = [r for r in raw if r.get('status') != 'OK']
    assert not fails, f'{len(fails)} failed raw rows'
    raw = [r for r in raw if r['status'] == 'OK']
    images = sorted({r['image_id'] for r in raw})
    assert len(images) == 40, len(images)
    by = {(r['arm'], r['image_id'], r['turn']): r for r in raw}
    assert len(by) == len(raw) == 40 * 6 * 6, (len(by), len(raw))
    hits = {a: [by[(a, i, t)] for i in images for t in range(2, 7)] for a in ARMS}
    persist = jl(base / 'gqa/persistence.jsonl')
    gates = [g for g in jl(base / 'gates.jsonl') if g['phase'] == 'gqa']
    gpu = json.loads((base / 'gpu_correctness.json').read_text())
    status = json.loads((base / 'status.json').read_text())
    protection = json.loads((run / 'protection_result.json').read_text()) if (run / 'protection_result.json').exists() else {'status': 'NOT RUN'}

    # -------------------------------------------------------- quality
    quality = {}
    for a in ARMS:
        allrows = [by[(a, i, t)] for i in images for t in range(1, 7)]
        quality[a] = {'hit_accuracy': float(np.mean([r['accuracy'] for r in hits[a]])),
                      'all_accuracy': float(np.mean([r['accuracy'] for r in allrows])),
                      'n_hits': len(hits[a])}
    per_image = {a: {i: [by[(a, i, t)]['accuracy'] for t in range(2, 7)] for i in images} for a in ARMS}
    paired = {}
    for a, b in PAIRS:
        ci = paired_bootstrap([(sum(per_image[a][i]), sum(per_image[b][i]), 5) for i in images])
        disc = [(x['accuracy'], y['accuracy']) for x, y in zip(hits[a], hits[b])]
        ci.update({'a_only_correct': sum(1 for x, y in disc if x > y), 'b_only_correct': sum(1 for x, y in disc if y > x),
                   'verdict': verdict(ci, a, b)})
        paired[f'{a}-{b}'] = ci
    agree = {a: {'first_token': float(np.mean([x['first_token_id'] == y['first_token_id'] for x, y in zip(hits[a], hits['fullload'])])),
                 'prediction': float(np.mean([x['prediction'] == y['prediction'] for x, y in zip(hits[a], hits['fullload'])]))}
             for a in ARMS}

    # ------------------------------------------------- latency and I/O
    latency = {a: {'ttft': stats([r['ttft_ms'] for r in hits[a]]), 'e2e_mean': float(np.mean([r['request_e2e_ms'] for r in hits[a]])),
                   'read_mb': float(np.mean([r['read_bytes'] for r in hits[a]])) / MB,
                   'preads': float(np.mean([r['preads'] for r in hits[a]]))} for a in ARMS}
    io_equal = all(len({(by[(a, i, t)]['read_bytes'], by[(a, i, t)]['preads']) for a in ['ours', 'orig', 'compact', 'posshift']}) == 1
                   for i in images for t in range(2, 7))
    ttft_paired = {f'{a}-ours': paired_bootstrap([(sum(by[(a, i, t)]['ttft_ms'] for t in range(2, 7)),
                                                   sum(by[('ours', i, t)]['ttft_ms'] for t in range(2, 7)), 5) for i in images])
                   for a in ['orig', 'compact', 'posshift']}

    # ------------------------------------------------------ persistence
    P = defaultdict(dict)
    for p in persist:
        P[p['arm']][p['image_id']] = p
    def comp(p):
        t = p['timing_ms']; rc = p.get('recompute') or {}
        return {'saliency_ms': p.get('saliency_ms') or 0.0, 'permutation_ms': t.get('permutation_ms', 0.0),
                'plan_ms': rc.get('plan_ms', 0.0), 'embed_gather_ms': rc.get('embed_gather_ms', 0.0),
                'reduced_prefill_ms': rc.get('reduced_prefill_ms', 0.0), 'synthetic_assembly_ms': rc.get('synthetic_assembly_ms', 0.0),
                'repack_ms': t.get('repack_ms', 0.0), 'write_ms': t.get('ssd_write_ms', t.get('store_write_ms', 0.0)),
                'fsync_ms': t.get('durability_ms', 0.0), 'seal_ms': p['hash_and_seal_ms'], 'total_ms': p['persistence_ms'],
                'file_mb': p['file_logical_bytes'] / MB, 'reduced_tokens': rc.get('reduced_tokens')}
    persistence = {a: {k: float(np.mean([comp(P[a][i])[k] or 0 for i in images])) for k in comp(P[a][images[0]])}
                   for a in ['fullload', 'ours', 'orig', 'compact']}
    delta = {}
    for a in ['orig', 'compact']:
        prefill = [P[a][i]['recompute']['prefill_only_ms'] for i in images]
        full = paired_bootstrap([(P[a][i]['persistence_ms'], P['ours'][i]['persistence_ms'], 1) for i in images])
        delta[a] = {'prefill_only_mean_ms': float(np.mean(prefill)), 'prefill_only_p95_ms': float(np.percentile(prefill, 95)),
                    'full_delta': full,
                    'amortized_prefill_only': {n: float(np.mean(prefill)) / n for n in (1, 3, 5, 10)},
                    'amortized_full_delta': {n: full['mean'] / n for n in (1, 3, 5, 10)}}

    # ------------------------------------------------------ KV diffs
    layer_rows = []
    L = len(gates[0]['G2']['orig']['layers'])
    for li in range(L):
        row = {'layer': li}
        for a in ['orig', 'compact']:
            for key in ['k_S', 'v_S', 'k_sep', 'v_sep'] + (['k_S_realigned'] if a == 'compact' else []):
                row[f'{a}_{key}_rel_l2'] = float(np.mean([g['G2'][a]['layers'][li][key]['rel_l2'] for g in gates]))
                row[f'{a}_{key}_cosine'] = float(np.mean([g['G2'][a]['layers'][li][key]['cosine'] for g in gates]))
        layer_rows.append(row)
    # rev. 2 structural vs-Ours bounds are report-only since rev. 3
    g2_max = {a: {name: max(max(x.get('rel_l2', 0) for x in g['G2'][a]['report_only'][name]) for g in gates
                            if name in g['G2'][a]['report_only'])
                  for name in ['sys_deep_k', 'sys_deep_v', 'common_prefix_deep_k', 'common_prefix_deep_v']}
              for a in ['orig', 'compact']}
    rev2_exceed = {a: sorted({g['image_id'] for g in gates for name, xs in g['G2'][a]['report_only'].items()
                              if not all(x['within_rev2_bound'] for x in xs)}) for a in ['orig', 'compact']}
    bitident = {a: {'images': sum(all(x['ok'] for n in ('structural_bitident_k', 'structural_bitident_v')
                                      for x in g['G2'][a]['gates'][n]) for g in gates),
                    'rows_min': min(g['G2'][a]['structural_rows'] for g in gates),
                    'rows_max': max(g['G2'][a]['structural_rows'] for g in gates)} for a in ['orig', 'compact']}
    # Stage-1 shape-only drift of common-prefix rows (control vs Ours) against the context effect
    single = []
    for g in gates:
        o = g['G2']['orig']
        if o['common_prefix_rows']:
            ls = o['layers']
            single.append({'image_id': g['image_id'], 'rows': o['common_prefix_rows'],
                           'k': max(ls[li]['k_common_control_vs_ours']['rel_l2'] for li in range(1, len(ls))),
                           'v': max(ls[li]['v_common_control_vs_ours']['rel_l2'] for li in range(1, len(ls))),
                           'context_k': ls[-1]['k_S']['rel_l2'], 'context_v': ls[-1]['v_S']['rel_l2']})
    control_ms = {a: float(np.mean([P[a][i]['truncation_control']['control_ms'] for i in images])) for a in ['orig', 'compact']}
    assert all(P[a][i]['truncation_control']['outside_timers'] for a in ['orig', 'compact'] for i in images)
    oracle_ratio = max(g['G2']['orig']['gates']['layer0_k'][0]['oracle_ratio'] for g in gates)

    # -------------------------------------------------- reference gate A
    ref = {(r['arm'], r['image_id'], str(r['question_id'])): r for r in jl(REF_RUN / 'llava/gqa/raw_requests.jsonl') if r.get('status') == 'OK'}
    repro = {}
    for new, old in [('ours', 'B'), ('fullload', 'fullload'), ('recompute', 'recompute')]:
        rows = [(r, ref[(old, r['image_id'], r['question_id'])]) for r in hits[new]]
        mism = [{'image_id': r['image_id'], 'question_id': r['question_id'], 'turn': r['turn'],
                 'new_prediction': r['prediction'], 'ref_prediction': o['prediction'],
                 'new_first': r['first_token_id'], 'ref_first': o['first_token_id'], 'gold': r['gold']}
                for r, o in rows if r['generated_token_ids'] != o['generated_token_ids'] or r['prediction'] != o['prediction']]
        repro[new] = {'reference_arm': old, 'n': len(rows),
                      'first_token_agreement': float(np.mean([r['first_token_id'] == o['first_token_id'] for r, o in rows])),
                      'prediction_agreement': float(np.mean([r['prediction'] == o['prediction'] for r, o in rows])),
                      'sequence_agreement': float(np.mean([r['generated_token_ids'] == o['generated_token_ids'] for r, o in rows])),
                      'new_hit_accuracy': quality[new]['hit_accuracy'],
                      'reference_hit_accuracy': float(np.mean([o['accuracy'] for _, o in rows])), 'mismatches': mism}

    # ---------------------------------------------------- noise floor
    def nf(s, c, m):
        return (s['noise_floor'].get(c) or {}).get(m)
    floor = [{'image_id': s['image_id'], 'common_prefix_rows': s['noise_floor']['common_prefix_rows'],
              **{f'{c}_{m}': nf(s, c, m) for c in ('prefix_only', 'orig_reduced')
                 for m in ('last_S_k', 'last_S_v', 'sys_max_k', 'sys_max_v')},
              **{f'{c}_{m}': nf(s, c, m) for c in ('orig_reduced', 'truncated_same_len', 'truncated_half_len')
                 for m in ('common_deep_max_k', 'common_deep_max_v')},
              'ratio_k': s['noise_floor']['context_over_floor']['k'], 'ratio_v': s['noise_floor']['context_over_floor']['v'],
              'single_ratio_k': s['noise_floor']['context_over_single_row']['k'],
              'single_ratio_v': s['noise_floor']['context_over_single_row']['v']}
             for s in gpu['samples']]

    # ------------------------------------------------------- gates
    g1 = all(g['G1'] == 'PASS' for g in gates); g2 = all(g['G2_status'] == 'PASS' for g in gates)
    pos = all(r['position_check']['verified'] for a in ['ours', 'orig', 'compact', 'posshift'] for r in hits[a])
    shift_ok = all(r['position_check']['text_position_start'] < by[('ours', r['image_id'], r['turn'])]['position_check']['text_position_start']
                   for r in hits['posshift'] + hits['compact'])
    t1_eq = all(len({json.dumps(g['t1_outputs'][a]) for a in ['ours', 'orig', 'compact']}) == 1 for g in gates)
    gate_rows = [('G1 selection/bytes/embeds/T1 identity', 'PASS' if g1 and t1_eq else 'FAIL'),
                 ('G1 equal hit read bytes/preads (4 KV25 arms)', 'PASS' if io_equal else 'FAIL'),
                 ('G2 rev. 3: structural bit-identity vs truncation control + layer-0 S/sep (all 40 images)', 'PASS' if g2 else 'FAIL'),
                 ('G3 positions (all KV25 hits; persistence)', 'PASS' if pos and shift_ok else 'FAIL'),
                 ('G4 GPU correctness (5 pairs)', gpu['status']),
                 ('G5 protection before/after', protection.get('status', 'NOT RUN')),
                 ('G6 reference reproduction (report gate A)', f"Ours {pct(repro['ours']['sequence_agreement'])} sequence agreement"),
                 ('G7 sentinel non-leakage (smoke)', status.get('SMOKE', 'NOT RUN'))]

    # ------------------------------------------------------- outputs
    summary = []
    for a in ARMS:
        row = {'arm': NAME[a], 'hit_accuracy': quality[a]['hit_accuracy'], 'all_accuracy': quality[a]['all_accuracy'],
               'fullload_first_agreement': agree[a]['first_token'], 'fullload_prediction_agreement': agree[a]['prediction'],
               'hit_ttft_mean_ms': latency[a]['ttft']['mean'], 'hit_ttft_p50_ms': latency[a]['ttft']['p50'],
               'hit_ttft_p95_ms': latency[a]['ttft']['p95'], 'hit_e2e_mean_ms': latency[a]['e2e_mean'],
               'hit_read_mb': latency[a]['read_mb'], 'hit_preads': latency[a]['preads'],
               'persistence_total_ms': persistence.get(STORE(a), {}).get('total_ms') if STORE(a) else None,
               'extra_prefill_only_ms': delta.get(a, {}).get('prefill_only_mean_ms')}
        if a in ('orig', 'compact', 'posshift'):
            d = paired[f'ours-{a}']
            row.update({'ours_minus_arm_pp': 100 * d['mean'], 'ci_low_pp': 100 * d['ci_low'], 'ci_high_pp': 100 * d['ci_high']})
        summary.append(row)
    keys = sorted({k for r in summary for k in r}, key=lambda k: list(summary[0]).index(k) if k in summary[0] else 99)
    with (res / 'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(summary)
    with (res / 'kv_layer_diff.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(layer_rows[0])); w.writeheader(); w.writerows(layer_rows)
    with (res / 'persistence.csv').open('w', newline='') as f:
        cols = list(next(iter(persistence.values())))
        w = csv.writer(f); w.writerow(['arm'] + cols)
        for a, v in persistence.items():
            w.writerow([NAME[a]] + [v[c] for c in cols])
    analysis = {'quality': quality, 'paired_accuracy': paired, 'fullload_agreement': agree, 'latency': latency,
                'ttft_paired_vs_ours': ttft_paired, 'kv25_io_equal': io_equal, 'persistence': persistence,
                'recompute_delta': delta, 'g2_structural_vs_ours_max_report_only': g2_max, 'rev2_bound_exceeded_report_only': rev2_exceed,
                'structural_bitident': bitident, 'single_row_drift_stage1': single, 'truncation_control_ms': control_ms, 'layer0_k_oracle_ratio_max': oracle_ratio,
                'reference_reproduction': repro, 'noise_floor': floor, 'gates': dict(gate_rows), 'status': status,
                'protection': {k: v for k, v in protection.items() if k != 'differences'}}
    (res / 'analysis.json').write_text(json.dumps(analysis, indent=1) + '\n')

    q = lambda a: pct(quality[a]['hit_accuracy'])
    rep = [f'# Full-context KV25 vs. reduced-token Recompute KV25 (LLaVA, GQA Stage 1)', '',
           f'Run: `runs/{run.name}`. Contract: [CONTRACT.md](CONTRACT.md) (frozen; G2 revision 2 was set after '
           'the dev samples n355567/n9181 and before measurement; revision 3 after the rev.-2 Stage-1 run stopped at '
           'n51002 and before any of its quality results were opened — see the amendment history, '
           '[dev_g2_revision1_failure/](dev_g2_revision1_failure/) and [dev_g2_revision2_failure/](dev_g2_revision2_failure/)). '
           'The stopped rev.-2 run is kept as a record only and is not analysed here.', '',
           'Workload: 40 GQA images × `questions[4:10]` (manifest `40e23fa7…`), T1 = `questions[4]` with full-token '
           'inference for every arm, 200 cache-hit requests (T2–T6) per arm. LLaVA-NeXT-Vicuna-7B NF4/BF16/eager, '
           'FP16 KV, 64-token chunks, separator sidecar, greedy max 16, cold page cache (DONTNEED) outside timers. '
           'All three KV25 stores share S, k, order and payload bytes (G1); PosShift reuses the Ours store.', '',
           '## Gates', '', table(['Gate', 'Result'], gate_rows), '',
           '## Cache-hit quality (T2–T6)', '',
           table(['Arm', 'Hit acc.', 'All acc.', 'First-token agr. vs FullLoad', 'Prediction agr. vs FullLoad'],
                 [[NAME[a], q(a), pct(quality[a]['all_accuracy']), pct(agree[a]['first_token']), pct(agree[a]['prediction'])] for a in ARMS]),
           '', 'PosShift T1 is the Ours T1 (shared store, not re-executed).', '',
           '### Paired differences (image-cluster bootstrap, 10,000 resamples, seed 1234)', '',
           table(['Difference', 'Mean', '95% CI', 'a-only / b-only correct', 'Pre-registered reading'],
                 [[f'{NAME[a]} − {NAME[b]}', pp(paired[f'{a}-{b}']['mean']),
                   f"[{pp(paired[f'{a}-{b}']['ci_low'])}, {pp(paired[f'{a}-{b}']['ci_high'])}]",
                   f"{paired[f'{a}-{b}']['a_only_correct']} / {paired[f'{a}-{b}']['b_only_correct']}",
                   paired[f'{a}-{b}']['verdict']] for a, b in PAIRS]), '',
           '## Decisions (rules fixed in the contract before results)', '']
    oo = paired['ours-orig']
    if oo['ci_low'] > 0 and abs(oo['mean']) >= 0.01:
        rep.append(f"- **Ours − Recomp-Orig**: CI lower bound > 0 ({pp(oo['ci_low'])}). Supports \"reusing full-context KV is better\" under equal S, k, order and bytes.")
    elif oo['ci_low'] > 0:
        rep.append(f"- **Ours − Recomp-Orig**: CI lower bound > 0 but |mean| = {pp(oo['mean'])} < 1 pp; not interpreted as established.")
    elif oo['ci_high'] < 0:
        rep.append(f"- **Ours − Recomp-Orig**: Recomp-Orig is significantly higher (CI upper bound {pp(oo['ci_high'])}). Reported as is.")
    else:
        rep.append(f"- **Ours − Recomp-Orig**: CI includes 0 → no quality difference shown. The paper uses only the cost argument (no extra prefill).")
    ps = paired['ours-posshift']
    rep.append(f"- **Ours − PosShift**: {ps['verdict']}." + (" This is evidence that original-input text positions are needed." if ps['ci_low'] > 0 and abs(ps['mean']) >= 0.01 else ""))
    oc = paired['ours-compact']
    rep.append(f"- **Ours − Recomp-Compact** (VisionZip-style alternative): {oc['verdict']}."
               + (f" The CI upper bound is exactly {pp(oc['ci_high'])} (mean {pp(oc['mean'])}, Compact-only correct "
                  f"{oc['b_only_correct']} vs Ours-only {oc['a_only_correct']}); by the fixed rule (upper bound < 0) this is "
                  "not a shown difference, and it is reported as a boundary case." if oc['ci_high'] == 0 else ""))
    cc = paired['orig-compact']
    rep.append(f"- **Recomp-Orig − Recomp-Compact** (no pre-registered decision; reported as is): {cc['verdict']} "
               f"(mean {pp(cc['mean'])}, CI [{pp(cc['ci_low'])}, {pp(cc['ci_high'])}]).")
    rep += ['',
            '## Quality interpretation and numerical noise floor', '',
            'Rows whose context is identical still change under a different prefill shape because BF16 GEMM '
            'kernels and RoPE rounding depend on sequence length. Controls (G4 pairs, recorded, not gated), each compared '
            'with the T1 cache of the same request: prefix-only drops only the (causally irrelevant) question; Orig is the '
            'actual reduced prefill; same-length / half-length truncations cut the original sequence to the reduced length '
            '`L` / `L/2`, so the structural rows keep context and position and only the shape changes.', '',
            table(['Image', 'Prefix-only last-layer S K / V', 'Orig last-layer S K / V', 'Context / prefix-only floor (K, V)',
                   'Orig sys max-layer K / V'],
                  [[f['image_id'], f"{f['prefix_only_last_S_k']:.3f} / {f['prefix_only_last_S_v']:.3f}",
                    f"{f['orig_reduced_last_S_k']:.3f} / {f['orig_reduced_last_S_v']:.3f}",
                    f"{f['ratio_k']:.1f}× / {f['ratio_v']:.1f}×", f"{f['orig_reduced_sys_max_k']:.4f} / {f['orig_reduced_sys_max_v']:.4f}"]
                   for f in floor]), '',
            'Single-row (common-prefix) drift, max over layers ≥ 1, K / V:', '',
            table(['Image', 'Common rows', 'Orig reduced', 'Same-length trunc.', 'Half-length trunc.', 'Context / single-row (K, V)'],
                  [[f['image_id'], f['common_prefix_rows']]
                   + [('—' if f[f'{c}_common_deep_max_k'] is None else
                       f"{f[f'{c}_common_deep_max_k']:.4f} / {f[f'{c}_common_deep_max_v']:.4f}")
                      for c in ('orig_reduced', 'truncated_same_len', 'truncated_half_len')]
                   + ['—' if f['single_ratio_k'] is None else f"{f['single_ratio_k']:.1f}× / {f['single_ratio_v']:.1f}×"]
                   for f in floor]), '',
            'Dev observations behind the G2 revisions: (rev. 2, n355567/n9181) prefix-only last-layer S difference '
            '0.010 / 0.021 (K / V) versus the Recomp-Orig context effect 0.21 / 0.42, about 21× / 20×; (rev. 3, '
            'n51002/n272098/n331357) single-row V drift 0.0308 / 0.0270 / 0.0190 for both the Orig reduced prefill and '
            'the same-length truncation (identical), 0.0329 / 0.0297 / 0.0202 for the half-length truncation, 0 for '
            'prefix-only; K 0.0105–0.0165. Context effect 0.21 / 0.40 (K / V) is 13–20× (K) and 12–21× (V) the single-row drift.', '',
            (f"Stage 1 shape-only drift of common-prefix rows (truncation control vs Ours, max over layers ≥ 1) on "
             f"{len(single)} images with a common prefix: K {min(x['k'] for x in single):.4f}–{max(x['k'] for x in single):.4f}, "
             f"V {min(x['v'] for x in single):.4f}–{max(x['v'] for x in single):.4f}; the last-layer S context effect on the "
             f"same images is K {min(x['context_k'] for x in single):.3f}–{max(x['context_k'] for x in single):.3f}, "
             f"V {min(x['context_v'] for x in single):.3f}–{max(x['context_v'] for x in single):.3f}." if single else
             'No Stage-1 image had a common prefix.'), '',
            '**Because BF16 shape-dependent rounding alone measurably changes deep-layer KV of context-identical '
            'rows and can flip a boundary first token, differences below 1 pp are not interpreted as established.** '
            'With 40 image clusters a CI that includes 0 is also not evidence of equivalence.', '',
            f"G2 rev. 3 over all 40 images: structural rows bit-identical to the same-length truncation control at every layer "
            f"(K and V) for Orig {bitident['orig']['images']}/40 and Compact {bitident['compact']['images']}/40 images "
            f"({bitident['orig']['rows_min']}–{bitident['orig']['rows_max']} rows). Maximum layer-0 K oracle ratio "
            f"{oracle_ratio:.3f} (bound 1.5). Report-only rev.-2 values: maximum deep-layer structural rel L2 vs Ours "
            f"Orig sys {g2_max['orig']['sys_deep_v']:.4f} / common {g2_max['orig']['common_prefix_deep_v']:.4f} (V), "
            f"Compact sys {g2_max['compact']['sys_deep_v']:.4f} / common {g2_max['compact']['common_prefix_deep_v']:.4f} (V); "
            f"images outside the old bound: Orig {len(rev2_exceed['orig'])}, Compact {len(rev2_exceed['compact'])}. "
            f"Truncation control wall time (outside every timer, not in persistence or TTFT): Orig {control_ms['orig']:.1f} ms, "
            f"Compact {control_ms['compact']:.1f} ms per image.", '',
            '## Per-layer KV difference vs Ours (mean over 40 images, S rows)', '',
            table(['Layer', 'Orig K', 'Orig V', 'Compact K raw', 'Compact K realigned', 'Compact V'],
                  [[r['layer'], f"{r['orig_k_S_rel_l2']:.4f}", f"{r['orig_v_S_rel_l2']:.4f}", f"{r['compact_k_S_rel_l2']:.4f}",
                    f"{r['compact_k_S_realigned_rel_l2']:.4f}", f"{r['compact_v_S_rel_l2']:.4f}"]
                   for r in layer_rows if r['layer'] in (0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 31)]), '',
            'Full per-layer table (S and separator rows, rel L2 and cosine): [kv_layer_diff.csv](kv_layer_diff.csv).', '',
            '## Cache-hit latency and SSD reads', '',
            table(['Arm', 'TTFT mean / p50 / p95 (ms)', 'E2E mean (ms)', 'Read MB/hit', 'preads/hit'],
                  [[NAME[a], f"{latency[a]['ttft']['mean']:.2f} / {latency[a]['ttft']['p50']:.2f} / {latency[a]['ttft']['p95']:.2f}",
                    f"{latency[a]['e2e_mean']:.2f}", f"{latency[a]['read_mb']:.3f}", f"{latency[a]['preads']:.1f}"] for a in ARMS]), '',
            f"The four KV25 arms read identical bytes and preads on every hit: {'yes' if io_equal else 'NO'}. "
            + '; '.join(f"{NAME[a]} − Ours TTFT {ttft_paired[f'{a}-ours']['mean']:+.2f} ms [{ttft_paired[f'{a}-ours']['ci_low']:+.2f}, {ttft_paired[f'{a}-ours']['ci_high']:+.2f}]"
                        for a in ['orig', 'compact', 'posshift']) + '.', '',
            '## Persistence cost per image (mean ms; not part of cache-hit TTFT)', '',
            table(['Arm', 'Saliency', 'Permutation', 'Plan', 'Embed gather', 'Reduced prefill', 'Synthetic assembly', 'Repack',
                   'Write', 'fsync', 'Seal', 'Total', 'File MB'],
                  [[NAME[a]] + [f"{persistence[a][k]:.2f}" for k in ['saliency_ms', 'permutation_ms', 'plan_ms', 'embed_gather_ms',
                   'reduced_prefill_ms', 'synthetic_assembly_ms', 'repack_ms', 'write_ms', 'fsync_ms', 'seal_ms', 'total_ms', 'file_mb']]
                   for a in ['fullload', 'ours', 'orig', 'compact']]), '',
            '**Primary metric — prefill-only Δ** (embed gather + CUDA-synchronized reduced prefill), amortized over requests per image:', '',
            table(['Arm', 'Prefill-only Δ mean / p95', '÷1', '÷3', '÷5', '÷10', 'Full Δ vs Ours [95% CI]', 'Full Δ ÷1/3/5/10'],
                  [[NAME[a], f"{delta[a]['prefill_only_mean_ms']:.2f} / {delta[a]['prefill_only_p95_ms']:.2f}"]
                   + [f"{delta[a]['amortized_prefill_only'][n]:.2f}" for n in (1, 3, 5, 10)]
                   + [f"{delta[a]['full_delta']['mean']:+.1f} [{delta[a]['full_delta']['ci_low']:+.1f}, {delta[a]['full_delta']['ci_high']:+.1f}]",
                      ' / '.join(f"{delta[a]['amortized_full_delta'][n]:+.1f}" for n in (1, 3, 5, 10))] for a in ['orig', 'compact']]), '',
            'The full Δ includes the full-size synthetic-cache assembly and plan recomputation that come from reusing the '
            'unchanged prefix25 writer, and wall-time noise of write/fsync; the prefill-only Δ is the cost a native implementation '
            'cannot avoid. The reduced prefill also requires keeping the T1 `inputs_embeds` (~24 MB BF16) until persistence.', '',
            '## Reference reproduction (report gate A)', '',
            table(['New arm', 'Reference arm', 'First-token agr.', 'Prediction agr.', 'Sequence agr.', 'New hit acc.', 'Reference hit acc.'],
                  [[NAME[a], v['reference_arm'], pct(v['first_token_agreement']), pct(v['prediction_agreement']),
                    pct(v['sequence_agreement']), pct(v['new_hit_accuracy']), pct(v['reference_hit_accuracy'])] for a, v in repro.items()]), '']
    for a, v in repro.items():
        if v['mismatches']:
            rep.append(f"{NAME[a]} mismatches ({len(v['mismatches'])}):")
            rep += [f"- {m['image_id']}/{m['question_id']} T{m['turn']}: new `{m['new_prediction']}` ({m['new_first']}) vs reference `{m['ref_prediction']}` ({m['ref_first']}), gold `{m['gold']}`" for m in v['mismatches']]
            rep.append('')
    rep += ['## Limits', '',
            '- 40 image clusters from the existing development set (not a holdout); 200 hits per arm. Stage 2 (MT-GQA) not run.',
            '- LLaVA only; Qwen2.5-VL (M-RoPE, compact cache) not ported — see the contract extension note.',
            '- Recomp arms reuse the arm\'s own T1 `inputs_embeds`; no VisionZip contextual-token merge (to keep S identical). '
            'Results therefore isolate KV context and position rule, not the full VisionZip system.',
            '- G2 was revised twice: rev. 2 after two dev samples before measurement; rev. 3 after the rev.-2 Stage-1 stop at n51002, before any of that run\'s quality results were opened (Stage 1 restarted from G4 in this run). All versions and failure logs are kept.',
            '- Latency is single-GPU, batch 1, page-cache DONTNEED; no NAND/controller coldness claim.', '']
    (res / 'REPORT.md').write_text('\n'.join(rep) + '\n')
    (res / 'REPRODUCE.md').write_text(f'''# Reproduction

Environment: `conda activate mllm_ft` (torch 2.5.1, transformers 4.57.6), RTX 4090, no other GPU process.
Use a fresh timestamped run name; nothing is overwritten (no-clobber creation).

```bash
RUN=runs/recompute_kv25_ablation_$(date -u +%Y%m%dT%H%M%SZ)
python scripts/134_eval_recompute_kv25_ablation.py --run $RUN --freeze
PYTHONPATH=.:tests python -m unittest -v test_recompute_kv25 test_prefix25_persistence test_llava_kv25 \\
    test_image_only_repack test_visdial_turn1_piggyback_core > $RUN/cpu_tests.log 2>&1
python scripts/134_eval_recompute_kv25_ablation.py --run $RUN --protect before
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u scripts/134_eval_recompute_kv25_ablation.py --run $RUN --gpu correctness,smoke
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u scripts/134_eval_recompute_kv25_ablation.py --run $RUN --gpu gqa
python scripts/134_eval_recompute_kv25_ablation.py --run $RUN --protect after
python scripts/135_report_recompute_kv25_ablation.py --run $RUN
```

Inputs are copied from `runs/prefix25_persistence_20261001T050113Z` (GQA manifest `40e23fa7…`, correctness
manifest) and verified by hash at freeze. `gpu_freeze.json` binds every source file (except this report script)
and the GPU stages refuse to start if any changed. Raw rows: `runs/{run.name}/llava/gqa/raw_requests.jsonl`;
persistence receipts `persistence.jsonl`; per-image G1/G2 receipts and per-layer KV differences `llava/gates.jsonl`;
G4 evidence `llava/gpu_correctness.json`. Stage-1 payload stores are deleted per image after their receipts
(guarded to `llava/stores/`; log `llava/cleanup.jsonl`) and must be regenerated; G4 stores and smoke Recomp stores
are retained in the run directory.
''')
    print((res / 'REPORT.md').read_text()[:3000])


def STORE(a):
    return {'fullload': 'fullload', 'ours': 'ours', 'orig': 'orig', 'compact': 'compact'}.get(a)


if __name__ == '__main__':
    main()

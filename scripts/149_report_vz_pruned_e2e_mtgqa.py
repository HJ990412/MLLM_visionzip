#!/usr/bin/env python3
"""Independent report for the VZ-Pruned-E2E MT-GQA run (Stage 2).

Re-audits every committed image through 148, rescored from raw rows, and
produces REPORT.md, summary.csv, contrasts.csv, kv_layer_diff.csv,
persistence.csv, session.csv, reproduction_vs_B.csv, g8.csv, analysis.json
and REPRODUCE.md.  Session accounting follows 113 (shared-stream /
standalone-equivalent).  No production inference code is imported.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, os, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]


def helper(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / file)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


AUD = helper('_vz_report_audit', '148_audit_vz_pruned_e2e_mtgqa.py')
sha = AUD.sha
METHODS = ('recompute', 'ours_prefixstore_kv25', 'vz_pruned_e2e')
LABEL = {'recompute': 'ReComp', 'ours_prefixstore_kv25': 'Ours-KV25', 'vz_pruned_e2e': 'VZ-Pruned-E2E'}
SHORT = {'recompute': 'ReComp', 'ours_prefixstore_kv25': 'Ours', 'vz_pruned_e2e': 'VZ'}
KIND = {'ours_prefixstore_kv25': 'image_only', 'vz_pruned_e2e': 'vz'}
B_MAIN = ROOT / 'runs/mt_gqa_B_llava5_qwen3_20261001T093921Z'
RESAMPLES, SEED = 10000, 1234
SCOPES = {'T1': lambda r: r['turn_id'] == 1, 'T2': lambda r: r['turn_id'] == 2, 'T3': lambda r: r['turn_id'] == 3,
          'hit': lambda r: r['turn_id'] > 1, 'all': lambda r: True}
PRIMARY = [('ours_prefixstore_kv25', 'vz_pruned_e2e', 'T1'), ('ours_prefixstore_kv25', 'vz_pruned_e2e', 'all')]
PAIRS = [('ours_prefixstore_kv25', 'vz_pruned_e2e'), ('ours_prefixstore_kv25', 'recompute'), ('vz_pruned_e2e', 'recompute')]


def norm(x): return ' '.join(w for w in re.sub(r'[^\w\s]', ' ', str(x).lower()).split() if w not in ('a', 'an', 'the'))
def read(p): return json.loads(Path(p).read_text())
def rows_of(p):
    with Path(p).open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def dump(p, x):
    with Path(p).open('w') as f: json.dump(x, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
def writecsv(p, rs):
    rs = list(rs)
    if not rs: return
    keys = list(dict.fromkeys(k for r in rs for k in r))
    with Path(p).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rs)
def mean(v): v = list(v); return float(np.mean(v)) if v else None
def stats(v):
    a = np.asarray(list(v), dtype=float)
    return {'mean': float(a.mean()), 'p50': float(np.percentile(a, 50)), 'p95': float(np.percentile(a, 95)), 'n': int(len(a))} if len(a) else {'mean': None, 'p50': None, 'p95': None, 'n': 0}
def fmt(x, d=2): return 'N/A' if x is None else (f'{x:.{d}f}' if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x))
def mps(s, d=1): return ' / '.join(fmt(s[k], d) for k in ('mean', 'p50', 'p95'))
def table(headers, rows): return ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|'] + ['| ' + ' | '.join(str(v) for v in r) + ' |' for r in rows]
def ci(p, d=2): return f"{p['point']:+.{d}f} [{p['ci_low']:+.{d}f}, {p['ci_high']:+.{d}f}]"


def compact(r):
    v = r['result']
    d = {k: r[k] for k in ('method_id', 'image_id', 'dialog_id', 'turn_id', 'ttft_ms', 'request_e2e_ms', 'ssd_read_bytes', 'ssd_preads', 'N_content', 'k', 'prompt_sha256', 'logical_request_id', 'physical_execution_id')}
    d.update(prediction=r['prediction'], normalized=norm(r['prediction']), score=float(norm(r['prediction']) == norm(r['gold'][0])),
             tokens=v['generated_token_ids'], first_token_id=v['first_token_id'], generated_tokens=len(v['generated_token_ids']),
             cap_reached=bool(v.get('cap_reached', False)), margin=(r.get('first_token_top2') or {}).get('margin'),
             prefill_ms=v.get('prefill_ms'), decode_ms=v.get('decode_ms'), chunk_io_ms=v.get('chunk_io_ms'), scatter_ms=v.get('scatter_ms'),
             peak_gpu_allocated_bytes=v.get('peak_gpu_allocated_bytes'), pruned_input_len=v.get('pruned_input_len'), text_tokens=v.get('text_tokens'),
             ttft_breakdown=v.get('ttft_breakdown_ms'), cache_hit=r['cache_hit'])
    assert d['score'] == r['score'], 'strict scorer differs from raw'
    return d


# ---------------------------------------------------------------- statistics
class Boot:
    """One joint image-cluster resampling shared by every contrast (pooled sums/counts)."""
    def __init__(self, images):
        self.images = list(images); self.index = {im: i for i, im in enumerate(self.images)}
        rng = np.random.default_rng(SEED); self.idx = rng.integers(0, len(self.images), size=(RESAMPLES, len(self.images)))

    def arrays(self, items, field):
        s = np.zeros(len(self.images)); c = np.zeros(len(self.images))
        for it in items:
            i = self.index[it['image_id']]; s[i] += it[field]; c[i] += 1
        return s, c

    def paired(self, a_items, b_items, field, scale=1.0):
        """mean(a) - mean(b) with pooled request weighting; a and b carry the same counts per image."""
        sa, ca = self.arrays(a_items, field); sb, cb = self.arrays(b_items, field)
        assert np.array_equal(ca, cb), 'unpaired counts'
        point = (sa.sum() - sb.sum()) / ca.sum() * scale
        boots = (sa[self.idx].sum(1) - sb[self.idx].sum(1)) / np.maximum(ca[self.idx].sum(1), 1e-12) * scale
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {'point': float(point), 'ci_low': float(lo), 'ci_high': float(hi), 'n': int(ca.sum()), 'images': int((ca > 0).sum()),
                'resamples': RESAMPLES, 'seed': SEED, 'weighting': 'pooled sums/counts; joint image-cluster resampling', 'excludes_zero': bool(lo > 0 or hi < 0)}

    def single(self, items, field, scale=1.0):
        s, c = self.arrays(items, field)
        point = s.sum() / c.sum() * scale
        boots = s[self.idx].sum(1) / np.maximum(c[self.idx].sum(1), 1e-12) * scale
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {'point': float(point), 'ci_low': float(lo), 'ci_high': float(hi), 'n': int(c.sum())}


# ------------------------------------------------------------------- loading
def load_run(run, out):
    config = read(run / 'config.json'); config.update(_file_sha256=sha(run / 'config.json'), _manifest_sha256=sha(run / 'manifest.json'))
    manifest = read(run / 'manifest.json')
    checks = AUD.A.Checks(); receipts = []; phase_counts = {}; attempts = Counter(); data = []; persist = []; activ = []; gates = []; g8 = []; layers = []
    logical = set(); physical = set()
    for phase in ('correctness', 'smoke', 'integration', 'main'):
        count = 0
        for cp in sorted((run / phase / 'images').glob('*/COMMITTED.json')):
            commit = read(cp); folder = cp.parent / commit['attempt_id']; raw = folder / 'raw.jsonl'; iid = cp.parent.name
            checks.require(sha(raw) == commit['raw_sha256'], 'committed_raw_hash', str(cp))
            for n, digest in commit['evidence_sha256'].items(): checks.require(sha(folder / n) == digest, 'committed_evidence_unchanged', str(folder / n))
            rows = list(rows_of(raw)); receipt = AUD.audit_image_rows(rows, manifest, config, phase, iid, folder); receipts.append(receipt)
            checks.require(receipt['status'] == 'PASS', 'independent_image_audit', receipt); count += len(rows)
            cleanup = read(folder / 'cleanup_receipt.json'); checks.require(cleanup['status'] == ('RETAINED' if phase == 'correctness' else 'PASS'), 'cleanup_receipt', str(folder))
            g = read(folder / 'gates.json'); gates.append({'phase': phase, 'image_id': iid, 'status': g['status'], 'G2_status': g['G2']['status'], 'G2b_equal': g['G2b']['equal'],
                                                         'memory_bf16_ok': g['G2c_memory_bf16']['ok'], 'sentinel_rows': g['G2e']['sentinel_rows'], 'control_ms': g['control']['control_ms'],
                                                         'control_length': g['control']['length'], 'structural_rows': g['control']['structural_rows'], 'gate_wall_ms': g['gate_wall_ms'],
                                                         **{f'{n}_rel_l2': g['G2']['gates'][n][0]['rel_l2'] for n in ('vz_layer0_k_realigned_S', 'vz_layer0_k_realigned_sep', 'vz_neg_layer0_k_unrealigned_S', 'vz_neg_layer0_k_unrealigned_sep')},
                                                         **{f'{n}_ratio': g['G2']['gates'][n][0]['oracle_ratio'] for n in ('vz_layer0_k_oracle_ratio_S', 'vz_layer0_k_oracle_ratio_sep')},
                                                         'layer0_v_S_rel_l2': g['G2']['gates']['layer0_v'][0]['S']['rel_l2'],
                                                         'sys_deep_k_max': max(x['rel_l2'] for x in g['G2']['report_only']['sys_deep_k']) if g['G2']['report_only'].get('sys_deep_k') else None,
                                                         'sys_deep_v_max': max(x['rel_l2'] for x in g['G2']['report_only']['sys_deep_v']) if g['G2']['report_only'].get('sys_deep_v') else None})
            if phase == 'main':
                for li, lay in enumerate(g['G2']['layers']):
                    layers.append({'image_id': iid, 'layer': li, 'k_S_realigned_rel_l2': lay['k_S_realigned']['rel_l2'], 'k_sep_realigned_rel_l2': lay['k_sep_realigned']['rel_l2'],
                                   'v_S_rel_l2': lay['v_S']['rel_l2'], 'v_sep_rel_l2': lay['v_sep']['rel_l2'], 'k_sys_rel_l2': lay['k_sys']['rel_l2'], 'v_sys_rel_l2': lay['v_sys']['rel_l2'],
                                   'k_structural_control_vs_ours_rel_l2': lay['k_structural_control_vs_ours']['rel_l2'], 'v_structural_control_vs_ours_rel_l2': lay['v_structural_control_vs_ours']['rel_l2']})
                for r in rows:
                    checks.require(r['logical_request_id'] not in logical and r['physical_execution_id'] not in physical, 'global_unique_ids', r['request_id'])
                    logical.add(r['logical_request_id']); physical.add(r['physical_execution_id']); data.append(compact(r))
                for method, kind in KIND.items():
                    p = read(folder / f'{kind}_persistence.json'); a = read(folder / f'{method}_activation.json'); meta = read(folder / f'{kind}_meta.json')
                    N = meta['v_token_num'] - len(meta['newline_idx']); k = meta['stored_content_count']
                    persist.append({'image_id': iid, 'method_id': method, 'event_id': p['event_id'], 'source_dialog_id': p['source']['dialog_id'], 'N': N, 'k': k,
                                    'service_persistence_ms': p['service_persistence_ms'], 'native_writer_wall_ms': p['native_writer_wall_ms'], 'completion_full_hash_ms': p['completion_full_hash_ms'],
                                    'completion_publication_ms': p['completion_publication_ms'], 'file_logical_bytes': p['file_logical_bytes'], 'allocated_bytes': p['allocated_bytes'],
                                    'os_write_bytes': p['os_write_bytes'], 'os_write_calls': p['os_write_calls'],
                                    'synthetic_assembly_ms': (p.get('synthetic') or {}).get('synthetic_assembly_ms'), 'activation_ms': a['outer_wall_ms'],
                                    'activation_integrity_ms': a['completion_integrity']['integrity_ms'], 'provenance': 'MEASURED'})
                    activ.append({'image_id': iid, 'method_id': method, 'event_id': a['event_id'], 'outer_wall_ms': a['outer_wall_ms']})
                if (folder / 'g8_hit_vs_standalone.json').exists():
                    for pair in read(folder / 'g8_hit_vs_standalone.json')['pairs']:
                        g8.append({k: v for k, v in pair.items() if k not in ('hit_tokens', 'standalone_tokens')} | {'hit_margin': (pair['hit_top2'] or {}).get('margin'), 'standalone_margin': (pair['standalone_top2'] or {}).get('margin')})
            del rows
        phase_counts[phase] = count
        for attempt in (run / phase / 'images').glob('*/attempt_*'):
            attempts[phase + '_image_attempts'] += 1
            if (attempt / 'failure.json').exists(): attempts[phase + '_failed_attempts'] += 1
    expected = manifest['expected_requests']; complete = phase_counts['main'] == expected
    if complete:
        checks.require(len({r['image_id'] for r in data}) == 398 and len({r['dialog_id'] for r in data}) == 4061, 'main_population')
        for m in METHODS: checks.require(sum(r['method_id'] == m for r in data) == 12183, 'per_method_counts', m)
    for name, digest in read(run / 'gpu_freeze.json')['files'].items():
        p = run / name if name in ('manifest.json', 'config.json') else ROOT / name
        checks.require(sha(p) == digest, 'final_source_freeze', name)
    audit = checks.result(); audit.update(main_requests=phase_counts['main'], expected_main_requests=expected, phase_counts=phase_counts, attempts=dict(attempts),
                                          image_receipts=len(receipts), status_main='VALID' if complete and audit['status'] == 'PASS' else 'INVALID' if complete else 'PARTIAL' if phase_counts['main'] else 'NOT RUN')
    return config, manifest, audit, data, persist, activ, gates, layers, g8


def load_b_main():
    out = {}
    for cp in sorted((B_MAIN / 'llava/main/images').glob('*/COMMITTED.json')):
        doc = read(cp)
        for r in rows_of(cp.parent / doc['attempt_id'] / 'raw.jsonl'):
            if r['method_id'] in ('recompute', 'ours_prefixstore_kv25'):
                out[(r['dialog_id'], r['turn_id'], r['method_id'])] = {'prediction': r['prediction'], 'tokens': r['result']['generated_token_ids'], 'prompt_sha256': r['prompt_sha256']}
    return out


# ------------------------------------------------------------------- analysis
def analyze(run, out):
    out.mkdir(parents=True, exist_ok=True)
    config, manifest, audit, data, persist, activ, gates, layers, g8 = load_run(run, out)
    status = read(run / 'llava/status.json') if (run / 'llava/status.json').exists() else {}
    correctness = read(run / 'llava/gpu_correctness.json') if (run / 'llava/gpu_correctness.json').exists() else {'status': 'NOT RUN'}
    protection = read(run / 'protection_result.json') if (run / 'protection_result.json').exists() else {'status': 'NOT RUN'}
    A = {'run': run.name, 'audit': audit, 'status': status, 'correctness': {k: v for k, v in correctness.items() if k != 'samples'}, 'protection': {k: protection.get(k) for k in ('status', 'differences')},
         'config': {k: config[k] for k in ('primary_endpoints', 'bootstrap', 'rotation', 'store_source', 'vz_t1_path', 'vz_hit_path')}}
    if not data:
        dump(out / 'analysis.json', A); return A, None
    images = sorted({r['image_id'] for r in data}); boot = Boot(images)
    by = {(r['dialog_id'], r['turn_id'], r['method_id']): r for r in data}
    # quality
    quality = []
    for m in METHODS:
        mr = [r for r in data if r['method_id'] == m]; q = {'method_id': m, 'label': LABEL[m], 'requests': len(mr)}
        for s, pred in SCOPES.items():
            rs = [r for r in mr if pred(r)]; q[f'{s}_correct'] = int(sum(r['score'] for r in rs)); q[f'{s}_n'] = len(rs); q[f'{s}_acc_pct'] = 100 * mean(r['score'] for r in rs)
        dial = [all(by[(d, t, m)]['score'] for t in (1, 2, 3)) for d in {r['dialog_id'] for r in mr}]
        q['dialogue_all_correct_pct'] = 100 * mean(dial); q['cap_reached'] = sum(r['cap_reached'] for r in mr)
        quality.append(q)
    A['quality'] = quality
    # contrasts (joint resamples)
    contrasts = []
    for a, b in PAIRS:
        for s, pred in SCOPES.items():
            ai = [r for r in data if r['method_id'] == a and pred(r)]; bi = [r for r in data if r['method_id'] == b and pred(r)]
            p = boot.paired(ai, bi, 'score', 100.0)
            contrasts.append({'contrast': f'{SHORT[a]} - {SHORT[b]}', 'a': a, 'b': b, 'scope': s, 'role': 'PRIMARY' if (a, b, s) in PRIMARY else 'secondary', 'metric': 'EM_pp', **p,
                              'shown': bool(p['excludes_zero'] and abs(p['point']) >= 1.0), 'interpretable': bool(abs(p['point']) >= 1.0)})
    # history-matched supplement: hit EM on dialogues whose T1 normalized prediction is equal between the two arms
    matched = []
    for a, b in PAIRS:
        same = {d for (d, t, m) in by if t == 1 and m == a and by[(d, 1, a)]['normalized'] == by[(d, 1, b)]['normalized']}
        ai = [r for r in data if r['method_id'] == a and r['turn_id'] > 1 and r['dialog_id'] in same]; bi = [r for r in data if r['method_id'] == b and r['turn_id'] > 1 and r['dialog_id'] in same]
        p = boot.paired(ai, bi, 'score', 100.0) if ai else None
        matched.append({'contrast': f'{SHORT[a]} - {SHORT[b]}', 'scope': 'hit, T1-matched dialogues', 'dialogues': len(same), 'dialogues_total': len({d for (d, t, m) in by if m == a and t == 1}),
                        'role': 'report-only', **(p or {})})
    A['contrasts'] = contrasts; A['history_matched'] = matched
    # latency
    latency = []
    for m in METHODS:
        for s in ('T1', 'hit'):
            rs = [r for r in data if r['method_id'] == m and SCOPES[s](r)]
            latency.append({'method_id': m, 'scope': s, 'n': len(rs), 'ttft_ms': stats(r['ttft_ms'] for r in rs), 'e2e_ms': stats(r['request_e2e_ms'] for r in rs),
                            'prefill_ms': stats(r['prefill_ms'] for r in rs if r['prefill_ms'] is not None), 'decode_ms': stats(r['decode_ms'] for r in rs if r['decode_ms'] is not None),
                            'ssd_MB': stats(r['ssd_read_bytes'] / 1e6 for r in rs), 'preads': stats(r['ssd_preads'] for r in rs), 'generated_tokens': stats(r['generated_tokens'] for r in rs),
                            'peak_gpu_GB': stats(r['peak_gpu_allocated_bytes'] / 2**30 for r in rs if r['peak_gpu_allocated_bytes']),
                            'chunk_io_ms': stats(r['chunk_io_ms'] for r in rs if r['chunk_io_ms'] is not None), 'scatter_ms': stats(r['scatter_ms'] for r in rs if r['scatter_ms'] is not None)})
    vz_t1 = [r for r in data if r['method_id'] == 'vz_pruned_e2e' and r['turn_id'] == 1]
    A['vz_t1_breakdown'] = {k: stats(r['ttft_breakdown'][k] for r in vz_t1) for k in vz_t1[0]['ttft_breakdown']}
    A['vz_t1_tokens'] = {'pruned_input_len': stats(r['pruned_input_len'] for r in vz_t1), 'full_len_equivalent': stats(5 + r['N_content'] + 0 + r['text_tokens'] for r in vz_t1), 'text_tokens': stats(r['text_tokens'] for r in vz_t1)}
    A['latency'] = latency
    lat_pairs = []
    for a, b in PAIRS:
        for s in ('T1', 'hit'):
            for field in ('ttft_ms', 'request_e2e_ms'):
                ai = [r for r in data if r['method_id'] == a and SCOPES[s](r)]; bi = [r for r in data if r['method_id'] == b and SCOPES[s](r)]
                lat_pairs.append({'contrast': f'{SHORT[a]} - {SHORT[b]}', 'scope': s, 'metric': field, **boot.paired(ai, bi, field)})
    # persistence and session accounting (113 definitions)
    setup = {}
    for p in persist: setup[(p['image_id'], p['method_id'])] = p
    sessions = []
    for m in METHODS:
        for d in {r['dialog_id'] for r in data if r['method_id'] == m}:
            rs = [by[(d, t, m)] for t in (1, 2, 3)]; iid = rs[0]['image_id']; req = sum(r['request_e2e_ms'] for r in rs)
            if m == 'recompute': build = act = 0.0; source = False
            else:
                s = setup[(iid, m)]; source = d == s['source_dialog_id']; build, act = s['service_persistence_ms'], s['activation_ms']
            sessions.append({'image_id': iid, 'dialog_id': d, 'method_id': m, 'source_dialogue': source, 'request_e2e_sum_ms': req,
                             'build_attributed_ms': build if source else 0.0, 'activation_attributed_ms': act if source else 0.0,
                             'shared_stream_session_ms': req + ((build + act) if source else 0.0), 'standalone_session_ms': req + build + act,
                             'setup_provenance': 'MEASURED_NO_KV_SETUP' if m == 'recompute' else 'MEASURED; standalone DERIVED'})
    attributed = sum(x['build_attributed_ms'] + x['activation_attributed_ms'] for x in sessions)
    physical_events = sum(p['service_persistence_ms'] + p['activation_ms'] for p in persist)
    A['session_accounting_audit'] = {'attributed_setup_ms': attributed, 'physical_event_setup_ms': physical_events, 'match': math.isclose(attributed, physical_events, abs_tol=1e-5), 'physical_builds': len(persist)}
    session_summary = []
    for m in METHODS:
        sd = [x for x in sessions if x['method_id'] == m]
        session_summary.append({'method_id': m, 'dialogues': len(sd), **{f'{k}': stats(x[k] for x in sd) for k in ('request_e2e_sum_ms', 'shared_stream_session_ms', 'standalone_session_ms')}})
    session_pairs = []
    look = {(x['dialog_id'], x['method_id']): x for x in sessions}
    for a, b in PAIRS:
        for field in ('shared_stream_session_ms', 'standalone_session_ms', 'request_e2e_sum_ms'):
            ai = [look[(d, a)] for (d, m) in look if m == a]; bi = [look[(d, b)] for d in [x['dialog_id'] for x in ai]]
            session_pairs.append({'contrast': f'{SHORT[a]} - {SHORT[b]}', 'scope': 'per_dialogue', 'metric': field, **boot.paired(ai, bi, field)})
    persist_summary = []
    for m in KIND:
        ps = [p for p in persist if p['method_id'] == m]
        persist_summary.append({'method_id': m, 'images': len(ps), **{k: stats(p[k] for p in ps) for k in ('service_persistence_ms', 'native_writer_wall_ms', 'completion_full_hash_ms', 'file_logical_bytes', 'os_write_bytes', 'activation_ms', 'activation_integrity_ms')},
                                'synthetic_assembly_ms': stats(p['synthetic_assembly_ms'] for p in ps if p['synthetic_assembly_ms'] is not None), 'stored_content_ratio': sum(p['k'] for p in ps) / sum(p['N'] for p in ps)})
    pp = []
    for field in ('service_persistence_ms', 'file_logical_bytes', 'os_write_bytes', 'activation_ms'):
        ai = [p for p in persist if p['method_id'] == 'vz_pruned_e2e']; bi = [p for p in persist if p['method_id'] == 'ours_prefixstore_kv25']
        pp.append({'contrast': 'VZ - Ours', 'scope': 'per_image', 'metric': field, **boot.paired(ai, bi, field)})
    A.update(latency_pairs=lat_pairs, sessions=session_summary, session_pairs=session_pairs, persistence=persist_summary, persistence_pairs=pp)
    # G2 per-layer table
    per_layer = []
    for li in range(32):
        ls = [x for x in layers if x['layer'] == li]
        if not ls: continue
        per_layer.append({'layer': li, **{k + '_mean': mean(x[k] for x in ls) for k in ls[0] if k not in ('image_id', 'layer')}, **{k + '_max': max(x[k] for x in ls) for k in ls[0] if k not in ('image_id', 'layer')}})
    A['kv_layers'] = per_layer
    mg = [g for g in gates if g['phase'] == 'main']
    A['gates'] = {'images': len(gates), 'main_images': len(mg), 'all_pass': all(g['status'] == 'PASS' for g in gates),
                  'realigned_S_rel_l2_max': max(g['vz_layer0_k_realigned_S_rel_l2'] for g in mg), 'realigned_sep_rel_l2_max': max(g['vz_layer0_k_realigned_sep_rel_l2'] for g in mg),
                  'oracle_ratio_S_max': max(g['vz_layer0_k_oracle_ratio_S_ratio'] for g in mg), 'oracle_ratio_sep_max': max(g['vz_layer0_k_oracle_ratio_sep_ratio'] for g in mg),
                  'neg_S_rel_l2_min': min(g['vz_neg_layer0_k_unrealigned_S_rel_l2'] for g in mg), 'layer0_v_S_rel_l2_max': max(g['layer0_v_S_rel_l2'] for g in mg),
                  'sys_deep_k_max': max(g['sys_deep_k_max'] for g in mg), 'sys_deep_v_max': max(g['sys_deep_v_max'] for g in mg), 'sys_rev2_bound_exceeded': sum(max(g['sys_deep_k_max'], g['sys_deep_v_max']) > 3e-2 for g in mg),
                  'control_ms': stats(g['control_ms'] for g in mg), 'gate_wall_ms': stats(g['gate_wall_ms'] for g in mg), 'memory_bf16_ok_all': all(g['memory_bf16_ok'] for g in gates),
                  'prefix_embeds_equal_all': all(g['G2b_equal'] for g in gates), 'sentinel_rows_total': sum(g['sentinel_rows'] for g in gates)}
    # G6 reproduction vs B main
    bmain = load_b_main(); repro = []; mism = []
    for m in ('recompute', 'ours_prefixstore_kv25'):
        for t in (1, 2, 3):
            rs = [r for r in data if r['method_id'] == m and r['turn_id'] == t]; agree_tok = agree_pred = prompt_same = 0; cascade = direct = 0
            for r in rs:
                b = bmain[(r['dialog_id'], t, m)]; same_prompt = r['prompt_sha256'] == b['prompt_sha256']; prompt_same += same_prompt
                tok = r['tokens'] == b['tokens']; agree_tok += tok; agree_pred += norm(r['prediction']) == norm(b['prediction'])
                if not tok:
                    kind = 'prompt_identical' if same_prompt else 'history_cascade'
                    direct += kind == 'prompt_identical'; cascade += kind == 'history_cascade'
                    mism.append({'method_id': m, 'turn_id': t, 'dialog_id': r['dialog_id'], 'image_id': r['image_id'], 'class': kind, 'this_prediction': r['prediction'], 'b_prediction': b['prediction'],
                                 'this_tokens': r['tokens'], 'b_tokens': b['tokens'], 'first_token_margin_this_run': r['margin'], 'this_score': r['score']})
            repro.append({'method_id': m, 'turn_id': t, 'requests': len(rs), 'tokens_equal': agree_tok, 'tokens_equal_pct': 100 * agree_tok / len(rs), 'prediction_equal': agree_pred,
                          'prompt_identical': prompt_same, 'mismatch_prompt_identical': direct, 'mismatch_history_cascade': cascade})
    A['reproduction'] = repro; A['reproduction_mismatches'] = mism
    # G8
    A['g8'] = {'pairs': len(g8), 'first_token_exact': sum(p['first_token_exact'] for p in g8), 'tokens_exact': sum(p['generated_token_ids_exact'] for p in g8), 'answer_exact': sum(p['answer_exact'] for p in g8),
               'max_abs_first_logit_delta': stats(p['max_abs_first_logit_delta'] for p in g8 if p['max_abs_first_logit_delta'] is not None), 'images': len({p['image_id'] for p in g8})}
    A['population'] = {'images': len(images), 'dialogues': len({r['dialog_id'] for r in data}), 'requests': len(data)}
    writecsv(out / 'summary.csv', [{**q, **{f'{s}_{k}': v for s in ('T1', 'hit') for l in latency if l['method_id'] == q['method_id'] and l['scope'] == s for k, v in (('ttft_mean_ms', l['ttft_ms']['mean']), ('ttft_p50_ms', l['ttft_ms']['p50']), ('ttft_p95_ms', l['ttft_ms']['p95']), ('e2e_mean_ms', l['e2e_ms']['mean']), ('ssd_MB_mean', l['ssd_MB']['mean']))},
                                   **{f'session_{k}_mean_ms': next(x for x in session_summary if x['method_id'] == q['method_id'])[k]['mean'] for k in ('shared_stream_session_ms', 'standalone_session_ms')}} for q in quality])
    writecsv(out / 'contrasts.csv', contrasts + [{'contrast': x['contrast'], 'scope': x['scope'], 'role': x['role'], 'metric': 'EM_pp', **{k: x.get(k) for k in ('point', 'ci_low', 'ci_high', 'n', 'images')}, 'dialogues': x['dialogues']} for x in matched]
               + [{**x, 'role': 'latency'} for x in lat_pairs] + [{**x, 'role': 'session'} for x in session_pairs] + [{**x, 'role': 'persistence'} for x in pp])
    writecsv(out / 'kv_layer_diff.csv', per_layer); writecsv(out / 'persistence.csv', persist); writecsv(out / 'session.csv', sessions)
    writecsv(out / 'reproduction_vs_B.csv', repro); writecsv(out / 'reproduction_mismatches.csv', mism); writecsv(out / 'g8.csv', g8); writecsv(out / 'gates.csv', gates)
    writecsv(out / 'request_diagnostics.csv', [{k: v for k, v in r.items() if k not in ('tokens', 'ttft_breakdown', 'prediction', 'normalized')} for r in data])
    dump(out / 'analysis.json', A)
    return A, data


# -------------------------------------------------------------------- report
def render(run, out, A):
    aud = A['audit']; q = {x['method_id']: x for x in A.get('quality', [])}
    L = [f'# VZ-Pruned-E2E vs Ours-KV25 vs ReComp — MT-GQA generated history (Stage 2)', '',
         f'Run `{run.name}`, contract `docs/vz_pruned_e2e_mtgqa_contract.md` (approved, amendments 1–4). '
         f'Workload 398 images / 4,061 dialogues / 3 turns, strict normalized EM, method-local generated history, seed 1234, greedy, 16 tokens. '
         f'모든 숫자는 raw rows를 독립 재채점한 값이다. CI는 image-cluster paired bootstrap 10,000회(공동 resample, pooled sums/counts)이며, '
         f'CI의 0 포함은 동등성 증거가 아니고 1 pp 미만 차이는 해석하지 않는다.', '',
         '## 1. 실행 상태와 gate', '']
    L += table(['항목', '상태'], [
        ['Main requests', f"{aud['main_requests']} / {aud['expected_main_requests']} ({aud['status_main']})"],
        ['Phase counts', json.dumps(aud['phase_counts'])], ['Image attempts / failed', json.dumps(aud['attempts'])],
        ['Independent audit (148, re-run at report time)', aud['status'] + ('' if aud['status'] == 'PASS' else ' ' + json.dumps(aud['failure_counts']))],
        ['G1 freeze / source hashes', 'PASS' if not aud['failure_counts'].get('final_source_freeze') else 'FAIL'],
        ['G4 correctness', A['correctness'].get('status', 'NOT RUN') + f" (G4-a evaluable samples {A['correctness'].get('g4a_evaluated_samples', 'N/A')})"],
        ['Stages', json.dumps(A['status'])], ['G5 protection (before/after)', A['protection']['status']]])
    if not A.get('quality'):
        L += ['', 'Main not complete; no quality numbers are reported.']; return '\n'.join(L) + '\n'
    g = A['gates']
    L += ['', f"G2 (per image, {g['main_images']} main images + smoke/integration/correctness = {g['images']}): all PASS = {g['all_pass']}. "
          f"Realigned layer-0 K rel L2 max S {g['realigned_S_rel_l2_max']:.2e} / sep {g['realigned_sep_rel_l2_max']:.2e} (bound 1e-2); "
          f"oracle ratio max {g['oracle_ratio_S_max']:.3f} / {g['oracle_ratio_sep_max']:.3f} (bound 1.5); negative control min {g['neg_S_rel_l2_min']:.3f} (> 0.1); "
          f"layer-0 V rel L2 max {g['layer0_v_S_rel_l2_max']:.1e}; structural rows bit-identical in every layer (FP16 store) and in BF16 memory ({g['memory_bf16_ok_all']}); "
          f"prefix embeddings bit-identical to the HF merge in every image ({g['prefix_embeds_equal_all']}); sentinel rows {g['sentinel_rows_total']}. "
          f"Report-only: deep-layer system-row drift vs Ours max K {g['sys_deep_k_max']:.2e} / V {g['sys_deep_v_max']:.2e}; images above the rev.-2 bound 3e-2: {g['sys_rev2_bound_exceeded']}. "
          f"Control prefill {mps(g['control_ms'])} ms and whole gate {mps(g['gate_wall_ms'])} ms per image, outside every timer.", '',
          '## 2. 품질 (strict EM, %)', '']
    L += table(['Arm', 'T1', 'T2', 'T3', 'Hit (T2–T3)', 'All', 'Dialogue all-3-correct', 'cap'],
               [[LABEL[m], fmt(q[m]['T1_acc_pct']), fmt(q[m]['T2_acc_pct']), fmt(q[m]['T3_acc_pct']), f"{fmt(q[m]['hit_acc_pct'])} ({q[m]['hit_correct']}/{q[m]['hit_n']})",
                 f"{fmt(q[m]['all_acc_pct'])} ({q[m]['all_correct']}/{q[m]['all_n']})", fmt(q[m]['dialogue_all_correct_pct']), q[m]['cap_reached']] for m in METHODS])
    prim = [c for c in A['contrasts'] if c['role'] == 'PRIMARY']; sec = [c for c in A['contrasts'] if c['role'] != 'PRIMARY']
    L += ['', '### 2.1 Primary endpoints (사전 지정: Ours − VZ, T1 EM과 all EM)', '']
    L += table(['Contrast', 'Scope', 'Δ pp [95% CI]', 'n', 'CI excludes 0', '|Δ| ≥ 1 pp', '판정'],
               [[c['contrast'], c['scope'], ci(c), c['n'], c['excludes_zero'], c['interpretable'], '차이 있음' if c['shown'] else ('미입증' if not c['excludes_zero'] else '1 pp 미만, 해석하지 않음')] for c in prim])
    L += ['', '### 2.2 Secondary contrasts (탐색적)', '']
    L += table(['Contrast', 'Scope', 'Δ pp [95% CI]', 'n', 'CI excludes 0', '판정'],
               [[c['contrast'], c['scope'], ci(c), c['n'], c['excludes_zero'], '차이 있음' if c['shown'] else ('미입증' if not c['excludes_zero'] else '1 pp 미만')] for c in sec])
    L += ['', '### 2.3 History-matched 보충 (report-only): 두 arm의 T1 정규화 답이 같은 대화의 hit EM', '']
    L += table(['Contrast', 'Dialogues (matched / all)', 'Δ pp [95% CI]'], [[x['contrast'], f"{x['dialogues']} / {x['dialogues_total']}", ci(x) if 'point' in x else 'N/A'] for x in A['history_matched']])
    lat = {(x['method_id'], x['scope']): x for x in A['latency']}
    L += ['', '## 3. 지연 (ms, mean / p50 / p95)', '', '### 3.1 첫 턴 (T1)', '']
    L += table(['Arm', 'TTFT', 'E2E', 'prefill→first token', 'decode', 'GPU peak GB'],
               [[LABEL[m], mps(lat[(m, 'T1')]['ttft_ms']), mps(lat[(m, 'T1')]['e2e_ms']), mps(lat[(m, 'T1')]['prefill_ms']), mps(lat[(m, 'T1')]['decode_ms']), mps(lat[(m, 'T1')]['peak_gpu_GB'], 2)] for m in METHODS])
    bd = A['vz_t1_breakdown']; tk = A['vz_t1_tokens']
    L += ['', f"VZ T1 TTFT 분해 (mean ms): " + ', '.join(f"{k} {fmt(v['mean'], 1)}" for k, v in bd.items()) +
          f". LM prefill 토큰 수 {mps(tk['pruned_input_len'], 0)} (full-token 기준 {mps(tk['full_len_equivalent'], 0)}). "
          'VZ T1은 실제 serving 경로(vision tower 1회 → image-only saliency → [sys|keep|question] 조립 → LM prefill 1회)에서 측정한 값이며 ReComp/Ours T1 TTFT와 직접 비교 가능하다.', '',
          '### 3.2 Cache hit (T2–T3)', '']
    L += table(['Arm', 'TTFT', 'E2E', 'SSD MB', 'preads', 'chunk I/O', 'scatter'],
               [[LABEL[m], mps(lat[(m, 'hit')]['ttft_ms']), mps(lat[(m, 'hit')]['e2e_ms']), mps(lat[(m, 'hit')]['ssd_MB']), mps(lat[(m, 'hit')]['preads'], 0), mps(lat[(m, 'hit')]['chunk_io_ms']), mps(lat[(m, 'hit')]['scatter_ms'])] for m in METHODS])
    L += ['', '### 3.3 Paired Δ (ms, request 단위, 공동 bootstrap)', '']
    L += table(['Contrast', 'Scope', 'Metric', 'Δ [95% CI]'], [[x['contrast'], x['scope'], x['metric'], ci(x, 1)] for x in A['latency_pairs']])
    ps = {x['method_id']: x for x in A['persistence']}
    L += ['', '## 4. 이미지별 persistence / activation (MEASURED; source dialogue T1에서 1회)', '']
    L += table(['Arm', 'Stored content', 'Persistence ms', 'native writer ms', 'hash ms', 'synthetic ms', 'Logical MB', 'OS write MB', 'Activation ms'],
               [[LABEL[m], f"{100 * ps[m]['stored_content_ratio']:.1f}%", mps(ps[m]['service_persistence_ms']), mps(ps[m]['native_writer_wall_ms']), mps(ps[m]['completion_full_hash_ms']),
                 mps(ps[m]['synthetic_assembly_ms']) if ps[m]['synthetic_assembly_ms']['n'] else '—', mps({k: v / 1e6 for k, v in ps[m]['file_logical_bytes'].items() if k != 'n'} | {'n': 0}, 1),
                 mps({k: v / 1e6 for k, v in ps[m]['os_write_bytes'].items() if k != 'n'} | {'n': 0}, 1), mps(ps[m]['activation_ms'])] for m in KIND])
    L += ['', 'VZ − Ours per image: ' + '; '.join(f"{x['metric']} {ci(x, 1)}" for x in A['persistence_pairs']), '',
          '## 5. 대화 단위 session 시간 (113 정의)', '',
          'Shared-stream = 3개 요청 E2E 합 + source dialogue에만 1회 귀속된 build/activation. Standalone-equivalent = 요청 합 + 이미지 setup을 모든 대화에 재귀속(DERIVED). '
          f"귀속 setup 합 {A['session_accounting_audit']['attributed_setup_ms']:.0f} ms = 물리 event 합 {A['session_accounting_audit']['physical_event_setup_ms']:.0f} ms ({A['session_accounting_audit']['match']}).", '']
    ss = {x['method_id']: x for x in A['sessions']}
    L += table(['Arm', '3-request E2E sum', 'Shared-stream session', 'Standalone-equivalent session'],
               [[LABEL[m], mps(ss[m]['request_e2e_sum_ms']), mps(ss[m]['shared_stream_session_ms']), mps(ss[m]['standalone_session_ms'])] for m in METHODS])
    L += ['']
    L += table(['Contrast', 'Metric', 'Δ ms [95% CI]'], [[x['contrast'], x['metric'], ci(x, 1)] for x in A['session_pairs']])
    L += ['', f"## 6. VZ store vs Ours store, layer별 (G2 기록; {g['main_images']} main images 평균 / 최대)", '',
          'S rows의 K는 position realign 후 비교. layer 0는 gate(위), deep layer의 S-row 차이는 문맥 효과(축소 문맥 T1 vs 전체 문맥 T1)이며 gate가 아니다. '
          '"control vs Ours"는 같은 길이(L+q)로 자른 원본 시퀀스의 구조 행과 Ours의 차이, 즉 shape rounding floor다.', '']
    L += table(['Layer', 'K_S realigned (mean / max)', 'V_S (mean / max)', 'K_sys (mean / max)', 'V_sys (mean / max)', 'K struct control vs Ours (max)', 'V struct control vs Ours (max)'],
               [[x['layer'], f"{x['k_S_realigned_rel_l2_mean']:.2e} / {x['k_S_realigned_rel_l2_max']:.2e}", f"{x['v_S_rel_l2_mean']:.2e} / {x['v_S_rel_l2_max']:.2e}",
                 f"{x['k_sys_rel_l2_mean']:.2e} / {x['k_sys_rel_l2_max']:.2e}", f"{x['v_sys_rel_l2_mean']:.2e} / {x['v_sys_rel_l2_max']:.2e}",
                 f"{x['k_structural_control_vs_ours_rel_l2_max']:.2e}", f"{x['v_structural_control_vs_ours_rel_l2_max']:.2e}"] for x in A['kv_layers'] if x['layer'] in (0, 1, 2, 4, 8, 16, 24, 30, 31)])
    L += ['', '## 7. G6 — B main 대비 요청 단위 재현 (report-only)', '']
    L += table(['Arm', 'Turn', 'Requests', 'Tokens equal', '%', 'Prediction equal', 'Prompt identical', 'Mismatch: prompt-identical', 'Mismatch: history-cascade'],
               [[LABEL[x['method_id']], x['turn_id'], x['requests'], x['tokens_equal'], fmt(x['tokens_equal_pct']), x['prediction_equal'], x['prompt_identical'], x['mismatch_prompt_identical'], x['mismatch_history_cascade']] for x in A['reproduction']])
    mm = A['reproduction_mismatches']
    if mm:
        L += ['', f'불일치 {len(mm)}건 (prompt-identical {sum(x["class"] == "prompt_identical" for x in mm)}, history-cascade {sum(x["class"] == "history_cascade" for x in mm)}); prompt-identical 건의 이 run 첫 토큰 top-2 margin: ' +
              ', '.join(fmt(x['first_token_margin_this_run'], 3) for x in mm if x['class'] == 'prompt_identical')[:600] + '. 상세는 reproduction_mismatches.csv.']
    else:
        L += ['', '불일치 없음: ReComp와 Ours의 12,183 요청 모두 B main과 토큰 단위로 동일하다.']
    g8 = A['g8']
    L += ['', '## 8. G8 — VZ hit vs 같은 history의 cache-less 단일 prefill (report-only)', '',
          f"{g8['images']} images × T2,T3 = {g8['pairs']} pairs: first token equal {g8['first_token_exact']}/{g8['pairs']}, all tokens equal {g8['tokens_exact']}/{g8['pairs']}, answer equal {g8['answer_exact']}/{g8['pairs']}; "
          f"첫 토큰 logit max |Δ| {mps(g8['max_abs_first_logit_delta'], 3)} (shape rounding; gate 아님). G4-d(5 correctness images)는 gpu_correctness.json에 있다.", '',
          '## 9. 한계', '',
          '- 같은 재구성 MT-GQA를 B main과 다시 사용했다. Ours − ReComp는 복제, VZ 비교는 새로 사전 등록한 탐색적 비교다.',
          '- Hit EM은 각 arm 자신의 생성 history 아래의 method-level 비교다(2.3의 matched 보충은 조건부 부분집합이며 판정에 쓰지 않는다).',
          '- 세 arm 회전(ordinal % 3)은 B main의 5-arm 회전과 다르므로 열적·캐시 상태의 순서 효과는 B main과 동일하지 않다.',
          '- Store는 이미지당 source dialogue의 T1에서 1회 만든다. VZ prefix KV는 인과 구조상 질문 독립이지만 T1 길이(L+q)에 따른 BF16 shape rounding을 가진다(G2-c 같은 길이 control, G8).',
          '- Persistence/activation은 source dialogue에만 발생한 실측이며 standalone-equivalent는 DERIVED다. OS write는 NAND write가 아니다.',
          f"- VZ T1의 saliency·plan 비용은 TTFT 안에 있다(breakdown 참조). Non-source T1의 capture 증분은 측정하지 않았다(VZ는 자신의 saliency capture 외에 없음).",
          '- 보고 스크립트(149, freeze 제외)는 main 진행 중 1개 commit 이미지(99 requests)로 렌더링 경로를 한 번 검증했다. 그 부분 출력은 scratchpad에만 있고 결과 디렉터리에 없으며, 판정 규칙은 freeze 전에 고정되어 변경되지 않았다.', '']
    return '\n'.join(L) + '\n'


def reproduce(run, out):
    return f"""# Reproduce / resume

```bash
cd {ROOT}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.:scripts:tests PYTHONDONTWRITEBYTECODE=1
PY=/home/dblab/anaconda3/envs/mllm_ft/bin/python
RUN=runs/vz_pruned_e2e_mtgqa_$(date -u +%Y%m%dT%H%M%SZ)
$PY scripts/147_eval_vz_pruned_e2e_mtgqa.py --run $RUN --freeze
$PY -m unittest tests.test_recompute_kv25 tests.test_rerope_kv25 tests.test_vz_pruned_e2e > $RUN/cpu_tests.log 2>&1
$PY scripts/147_eval_vz_pruned_e2e_mtgqa.py --run $RUN --protect before
$PY scripts/147_eval_vz_pruned_e2e_mtgqa.py --run $RUN --gpu correctness,smoke,integration,main   # add --resume after an interruption
$PY scripts/147_eval_vz_pruned_e2e_mtgqa.py --run $RUN --protect after
$PY scripts/149_report_vz_pruned_e2e_mtgqa.py --run $RUN --output results/$(basename $RUN)
```

This run: `{run}`. Frozen sources and their SHA256 are in `gpu_freeze.json`; the contract copy is `CONTRACT.md`;
the pre-freeze development check (not a result) is in `dev/`. Stores of smoke/integration/main images were deleted
after audit and commit (sealed hash lists in each image's `cleanup_receipt.json`); the five G4 images keep their
stores under `scratch/correctness/`. Reproducing an image's payload requires re-running its source dialogue's T1.
The report script is excluded from the freeze; 148 (audit) is frozen and re-executed at report time.
"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--output', required=True)
    args = ap.parse_args(); run = Path(args.run).resolve(); out = Path(args.output).resolve()
    A, data = analyze(run, out)
    (out / 'REPORT.md').write_text(render(run, out, A)); (out / 'REPRODUCE.md').write_text(reproduce(run, out))
    print(json.dumps({'status_main': A['audit']['status_main'], 'audit': A['audit']['status'], 'requests': A['audit']['main_requests']}), flush=True)


if __name__ == '__main__':
    main()

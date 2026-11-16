"""Independent audit + request-paired cross-run analysis of the packed layer-budget arm.

No production model import.  The paired bootstrap is the scripts/142 implementation
(RandomState(1234), 10,000 image-cluster draws, pooled sums / counts); callers pass that
function in so the draws are the same as in the v3 analysis.

Cross-run rule: the new arm is compared with the v3 rows of the *same request*
(image, turn).  It is allowed only when the Turn-1 generated tokens of the new arm equal
the v3 Turn-1 tokens for every image in the comparison set (G-det-T1).
"""
from pathlib import Path
import contextlib
import copy
import io
import json
import math

import numpy as np

from mmimpress.mtvqa25k.common import ROOT, read, sha
from mmimpress.mtvqa25k_packed import checks as CK, reference as REF

SCOPES = {'T1': [1], 'T2': [2], 'T3': [3], 'All': [1, 2, 3], 'Followup': [2, 3]}
ANSWER_TYPES = ('yes/no', 'number', 'other')
LABELS = {'recompute': 'ReComp', 'fullload': 'FullLoad', 'sparsevlm_ssd_kv25_allhead': 'SparseVLM-SSD-KV25-AllHead',
          'ours_prefixstore_kv25': 'PrefixStore-KV25 [B]'}
MTGQA_REFERENCE = {'pyr_minus_u_pp': 1.244, 'ci': [0.624, 1.864], 'metric': 'strict normalized EM', 'packed_read_MB': 308.05, 'packed_read_ms': 97.5,
                   'source': 'results/layer_budget_kv25_pyrbeta_20261110T073604Z'}
_W = {}


# ------------------------------------------------------------ worker audit
def init_worker(snapshot, methods_doc):
    from lib2to3.refactor import RefactoringTool
    from tokenizers import Tokenizer
    original = (ROOT / 'mmimpress/mtvqa25k/vendor/vqaEval.py.original').read_text()
    namespace = {}
    exec(compile(str(RefactoringTool(['lib2to3.fixes.fix_print']).refactor_string(original, 'official_original')), 'official_original', 'exec'), namespace)
    _W['Reference'] = namespace['VQAEval']
    _W['tokenizer'] = Tokenizer.from_file(str(Path(snapshot) / 'tokenizer.json'))
    gen = read(Path(snapshot) / 'generation_config.json')
    eos = gen.get('eos_token_id', [])
    _W['eos'] = set(eos if isinstance(eos, list) else [eos])
    _W['docs'] = methods_doc           # method -> budget entry (document clip range)


def audit_image(task):
    """One committed image of the new run: independent GT / score / decode / history / I/O / budget checks."""
    from types import SimpleNamespace
    folder, d, methods = task
    folder = Path(folder); iid = d['image_id']; fails = []
    try:
        commit = read(folder / 'COMMITTED.json'); attempt = folder / commit['attempt_id']
        for name, h in commit['evidence_sha256'].items():
            if sha(attempt / name) != h:
                fails.append('committed evidence hash: ' + name)
        rows, _ = REF.committed_raw(folder)
        lookup = {(r['method_id'], r['turn_id']): r for r in rows}
        if len(rows) != 3 * len(methods) or set(lookup) != {(m, t) for m in methods for t in (1, 2, 3)}:
            return {'image_id': iid, 'status': 'FAIL', 'failures': ['membership'], 'rows': []}
        qa = {}; pred = {}
        tok = _W['tokenizer']
        for index, r in enumerate(rows):
            t = d['turns'][r['turn_id'] - 1]
            if str(r['question_id']) != str(t['question_id']) or r['annotation'] != t['annotation'] or r['image_id'] != iid:
                fails.append('frozen question / annotation')
            annotation = copy.deepcopy(t['annotation']); annotation['question_id'] = index
            qa[index] = annotation; pred[index] = {'answer': r['scorer_answer']}
            ids = r['generated_token_ids']
            if tok.decode(ids, skip_special_tokens=False) != r['native_raw_decoded_output']:
                fails.append('raw decoding')
            if not (tok.decode(ids, skip_special_tokens=True).strip() == r['prediction'] == r['scorer_answer'] == r['result']['answer']):
                fails.append('native answer')
            if ids != r['result']['generated_token_ids'] or not 1 <= len(ids) <= 16:
                fails.append('generated ids')
            expected = '\n'.join(line for t0 in range(1, r['turn_id']) for line in
                                 (f"Q{t0}: {d['turns'][t0 - 1]['question'].strip()}", f"A{t0}: {lookup[(r['method_id'], t0)]['prediction']}"))
            body = ([expected, ''] if expected else []) + [f"Current question Q{r['turn_id']}: {t['question'].strip()}",
                                                         'Answer the current question with a single word or short phrase. ASSISTANT:']
            if r['history'] != expected or r['prompt'] != 'USER: <image>\n' + '\n'.join(body):
                fails.append('history / prompt reconstruction')
            trace = r['result'].get('os_pread_trace', [])
            if r['ssd_read_bytes'] != sum(e['returned'] for e in trace) or any(e['returned'] != e['requested'] for e in trace):
                fails.append('actual read bytes')
            if not 0 < r['ttft_ms'] <= r['request_e2e_ms']:
                fails.append('ttft / e2e')
            n = r['N_content']; doc = _W['docs'][r['method_id']]
            if r['turn_id'] == 1:
                if r['ssd_read_bytes'] or trace or r['result']['vision_forward_count'] != 1 or r['k'] != n:
                    fails.append('T1 must be a full-pixel request without store reads')
            else:
                counts = CK.hamilton(doc['layer_keep_ratios'], n, doc['r_max'])
                total = CK.packed_bytes(counts, len(r['geometry']['newline_idx']))
                ok = (r['result'].get('k_per_layer') == counts == CK.doc_counts(doc, n) and sum(counts) == 32 * ((n + 3) // 4) and min(counts) >= 1
                      and len(trace) == 1 and trace[0]['syscall'] == 'preadv' and trace[0]['offset'] == 0 and trace[0]['returned'] == total
                      and r['preadv_calls'] == 1 and r['pread_calls'] == 0 and r['ssd_read_bytes'] == total
                      and r['result']['vision_forward_count'] == 0 and r['result'].get('scoring_calls') == 0 and r['result'].get('query_score_calls', 0) == 0
                      and r['result'].get('layer_budget_sha256') == doc['layer_budget_sha256'])
                if not ok:
                    fails.append('packed hit contract (k_l, one preadv, bytes, no vision / scoring)')
        gt = SimpleNamespace(qa=qa, getQuesIds=lambda: list(qa)); res = SimpleNamespace(qa=pred)
        v = _W['Reference'](gt, res, n=10)
        with contextlib.redirect_stdout(io.StringIO()):
            v.evaluate()
        for index, r in enumerate(rows):
            if abs(r['score'] - v.evalQA[index] / 100) > 1e-10:
                fails.append('official VQA score (unmodified upstream evaluator)')
        setup = {}
        for m in methods:
            kind = 'packed_' + m[len('ours_prefixstore_kv25_'):-len('_packed')]
            p = read(attempt / f'{kind}_persistence.json'); a = read(attempt / f'{m}_activation.json')
            meta = read(attempt / f'{kind}_meta.json')
            if meta.get('dataset') != 'MT-VQA-v2-reconstructed-25k' or meta['layer_budget_r_min'] != _W['docs'][m]['r_min']:
                fails.append('store meta identity / clip range')
            setup[m] = {'persistence_ms': p['outer_helper_wall_ms'], 'service_persistence_ms': p['service_persistence_ms'], 'native_writer_wall_ms': p['native_writer_wall_ms'],
                        'activation_ms': a['outer_wall_ms'], 'store_logical_bytes': p['file_logical_bytes'], 'allocated_bytes': p['allocated_bytes'],
                        'write_bytes': p['os_write_bytes'], 'write_calls': p['os_write_calls'], 'packed_total_bytes': meta['packed_total_bytes']}
        if not (attempt / 'cleanup_receipt.json').is_file() or read(attempt / 'independent_audit.json')['status'] != 'PASS':
            fails.append('cleanup receipt / per-image audit receipt')
        cleanup = read(attempt / 'cleanup_receipt.json') if (attempt / 'cleanup_receipt.json').is_file() else {}
        attempts = len(list(folder.glob('attempt_*')))
        return {'image_id': iid, 'status': 'PASS' if not fails else 'FAIL', 'failures': fails, 'rows': [REF.compact(r) for r in rows], 'setup': setup,
                'raw_sha256': commit['raw_sha256'], 'commit_sha256': sha(folder / 'COMMITTED.json'), 'attempts': attempts,
                'cap_reached': sum(len(r['generated_token_ids']) == 16 and r['generated_token_ids'][-1] not in _W['eos'] for r in rows),
                'removed_scratch_files': cleanup.get('removed_new_files', 0)}
    except Exception as exc:
        return {'image_id': iid, 'status': 'FAIL', 'failures': ['exception: ' + repr(exc)], 'rows': []}


def audit_phase(run, phase, dialogues, methods, docs, workers=8):
    from multiprocessing import Pool
    snapshot = read(Path(run) / 'model_freeze.json')['llava']['snapshot']
    base = Path(run) / 'llava' / phase / phase / 'images'
    tasks = [(str(base / d['image_id']), d, list(methods)) for d in dialogues]
    with Pool(workers, initializer=init_worker, initargs=(snapshot, docs)) as pool:
        return list(pool.imap(audit_image, tasks, chunksize=8))


# -------------------------------------------------------------- statistics
def stats(values):
    values = np.asarray(values, dtype=float)
    return {'mean': float(values.mean()), 'p50': float(np.percentile(values, 50)), 'p95': float(np.percentile(values, 95)), 'n': int(len(values))}


def verdict(lo, hi):
    """Pre-registered rule (A): the 95% CI of D decides, nothing else."""
    if lo > 0:
        return 'IMPROVED'
    if hi < 0:
        return 'WORSE'
    return 'NOT_SHOWN'


def g_det_t1(new, ref, images):
    """Turn-1 generated tokens of the new arm vs every v3 arm's Turn-1 (same image)."""
    mism = []; equal = 0
    for iid in images:
        r = new[iid][1]; olds = [ref[iid][(m, 1)] for m in REF.V3_METHODS]
        same = all(r['generated_token_ids'] == o['generated_token_ids'] and r['prompt_sha256'] == o['prompt_sha256'] and r['suffix_ids_sha256'] == o['suffix_ids_sha256'] for o in olds)
        equal += same
        if not same:
            mism.append({'image_id': iid, 'new_tokens': r['generated_token_ids'], 'new_prediction': r['prediction'],
                         'v3_tokens': ref[iid][(REF.V3_B, 1)]['generated_token_ids'], 'v3_prediction': ref[iid][(REF.V3_B, 1)]['prediction'],
                         'prompt_equal': r['prompt_sha256'] == ref[iid][(REF.V3_B, 1)]['prompt_sha256']})
    n = len(images)
    return {'status': 'PASS' if equal == n and n > 0 else 'FAIL', 'images': n, 'T1_token_exact': equal, 'mismatch_count': len(mism), 'mismatches': mism[:500],
            'comparison': 'generated_token_ids, prompt SHA256 and text-side input-id SHA256 of the same image Turn-1; v3 Turn-1 is identical across all five v3 arms',
            'cross_run_accuracy_comparison': 'ALLOWED' if equal == n and n > 0 else 'NOT_COMPARABLE'}


def quality_table(new, ref, images):
    rows = []
    def add(label, method, getter):
        rec = {'arm': label, 'method_id': method}
        for scope, turns in SCOPES.items():
            rec[scope] = float(np.mean([getter(i, t)['score'] * 100 for i in images for t in turns]))
        rows.append(rec)
    add('PYR-beta-raw packed (this run)', 'ours_prefixstore_kv25_PYR-beta-raw_packed', lambda i, t: new[i][t])
    for m in REF.COMPARE_METHODS:
        add(LABELS[m] + ' (v3)', m, lambda i, t, m=m: ref[i][(m, t)])
    return rows


def paired_scores(new, ref, images, bootstrap, resamples=10000):
    """D = new - v3 baseline on the same (image, turn); image-cluster paired bootstrap (scripts/142 implementation)."""
    labels = []; vectors = []; denominators = []
    for m in REF.COMPARE_METHODS:
        for scope, turns in SCOPES.items():
            num = [sum((new[i][t]['score'] - ref[i][(m, t)]['score']) * 100 for t in turns) for i in images]
            labels.append({'baseline': m, 'baseline_label': LABELS[m], 'scope': scope, 'answer_type': 'all', 'requests': len(images) * len(turns),
                           'new_mean': float(np.mean([new[i][t]['score'] * 100 for i in images for t in turns])),
                           'baseline_mean': float(np.mean([ref[i][(m, t)]['score'] * 100 for i in images for t in turns]))})
            vectors.append(num); denominators.append([len(turns)] * len(images))
    for m in (REF.V3_B, 'recompute'):
        for at in ANSWER_TYPES:
            num = []; den = []; nv = []; bv = []
            for i in images:
                turns = [t for t in (2, 3) if new[i][t]['answer_type'] == at]
                num.append(sum((new[i][t]['score'] - ref[i][(m, t)]['score']) * 100 for t in turns)); den.append(len(turns))
                nv += [new[i][t]['score'] * 100 for t in turns]; bv += [ref[i][(m, t)]['score'] * 100 for t in turns]
            if sum(den) == 0:
                continue
            labels.append({'baseline': m, 'baseline_label': LABELS[m], 'scope': 'Followup', 'answer_type': at, 'requests': int(sum(den)),
                           'new_mean': float(np.mean(nv)), 'baseline_mean': float(np.mean(bv))})
            vectors.append(num); denominators.append(den)
    num = np.asarray(vectors, dtype=float).T; den = np.asarray(denominators, dtype=float).T
    ci = bootstrap(num, den, resamples)
    point = num.sum(axis=0) / den.sum(axis=0)
    out = []
    for j, label in enumerate(labels):
        lo, hi = float(ci[0, j]), float(ci[1, j])
        defined = math.isfinite(lo) and math.isfinite(hi)      # a tiny sample can draw a resample without any request of an answer type
        out.append({**label, 'delta_pp': float(point[j]), 'ci95_low': lo if defined else None, 'ci95_high': hi if defined else None, 'ci_defined': defined,
                    'ci_excludes_zero': bool(defined and (lo > 0 or hi < 0)), 'paired_images': len(images), 'resamples': resamples, 'seed': 1234,
                    'comparison': 'cross-run, 결정성 확인됨 (same request = same image and turn)',
                    'exploratory_no_multiplicity_correction': label['answer_type'] != 'all'})
    return out


def find(paired, baseline, scope, answer_type='all'):
    return next(p for p in paired if p['baseline'] == baseline and p['scope'] == scope and p['answer_type'] == answer_type)


def agreement(new, ref, images):
    """Prediction agreement of the new arm and v3 [B] on follow-up requests; one-sided correctness."""
    out = {}
    for scope in ('T2', 'T3', 'Followup'):
        pairs = [(new[i][t], ref[i][(REF.V3_B, t)]) for i in images for t in SCOPES[scope]]
        n = len(pairs)
        out[scope] = {'requests': n,
                      'prediction_string_equal': sum(a['prediction'] == b['prediction'] for a, b in pairs),
                      'generated_tokens_equal': sum(a['generated_token_ids'] == b['generated_token_ids'] for a, b in pairs),
                      'prediction_agreement_rate': sum(a['prediction'] == b['prediction'] for a, b in pairs) / n,
                      'only_PYR_correct': sum(a['score'] > 0 and b['score'] == 0 for a, b in pairs),
                      'only_B_correct': sum(a['score'] == 0 and b['score'] > 0 for a, b in pairs),
                      'both_correct': sum(a['score'] > 0 and b['score'] > 0 for a, b in pairs),
                      'neither_correct': sum(a['score'] == 0 and b['score'] == 0 for a, b in pairs),
                      'PYR_score_higher': sum(a['score'] > b['score'] for a, b in pairs), 'B_score_higher': sum(a['score'] < b['score'] for a, b in pairs),
                      'definition': 'correct = official VQA score > 0 for that request'}
    return out


def latency_io(new, ref, images):
    hits = [new[i][t] for i in images for t in (2, 3)]; t1 = [new[i][1] for i in images]
    out = {'followup_ttft_ms': stats([r['ttft_ms'] for r in hits]), 'followup_e2e_ms': stats([r['request_e2e_ms'] for r in hits]),
           'T1_ttft_ms': stats([r['ttft_ms'] for r in t1]), 'T1_e2e_ms': stats([r['request_e2e_ms'] for r in t1])}
    for scope in ('T2', 'T3'):
        out[f'{scope}_ttft_ms'] = stats([new[i][SCOPES[scope][0]]['ttft_ms'] for i in images])
    for key in ('packed_read_ms', 'chunk_io_ms', 'scatter_ms', 'prefill_ms', 'prepare_ms', 'decode_ms'):
        out['followup_' + key] = stats([r[key] for r in hits])
    read_bytes = sum(r['ssd_read_bytes'] for r in hits); read_ms = sum(r['packed_read_ms'] for r in hits)
    out['SSD_MB_per_followup'] = read_bytes / 1e6 / len(hits)
    out['preadv_calls_per_followup'] = float(np.mean([r['preadv_calls'] for r in hits])); out['pread_calls_per_followup'] = float(np.mean([r['pread_calls'] for r in hits]))
    out['read_calls_per_followup'] = float(np.mean([r['ssd_preads'] for r in hits]))
    out['T1_ssd_read_bytes_total'] = int(sum(r['ssd_read_bytes'] for r in t1))
    out['packed_read_MB_per_s'] = (read_bytes / 1e6) / (read_ms / 1e3)
    out['R_in_followup_over_T1_same_run'] = out['followup_ttft_ms']['mean'] / out['T1_ttft_ms']['mean']
    v3 = {}
    for m in REF.COMPARE_METHODS:
        v3[m] = {'followup_ttft_mean_ms': float(np.mean([ref[i][(m, t)]['ttft_ms'] for i in images for t in (2, 3)])),
                 'followup_e2e_mean_ms': float(np.mean([ref[i][(m, t)]['request_e2e_ms'] for i in images for t in (2, 3)])),
                 'T1_ttft_mean_ms': float(np.mean([ref[i][(m, 1)]['ttft_ms'] for i in images])),
                 'SSD_MB_per_followup': float(np.mean([ref[i][(m, t)]['ssd_read_bytes'] for i in images for t in (2, 3)])) / 1e6,
                 'preads_per_followup': float(np.mean([ref[i][(m, t)]['ssd_preads'] for i in images for t in (2, 3)]))}
        v3[m]['followup_over_T1_in_v3'] = v3[m]['followup_ttft_mean_ms'] / v3[m]['T1_ttft_mean_ms']
    out['v3_cross_run_reference'] = v3
    out['v3_cross_run_note'] = 'cross-run 참고치, 절대 TTFT는 실행 간 비교하지 않는다'
    v3_t1 = float(np.mean([ref[i][(m, 1)]['ttft_ms'] for i in images for m in REF.V3_METHODS]))
    out['environment_drift'] = {'T1_ttft_mean_ms_this_run': out['T1_ttft_ms']['mean'], 'v3_T1_ttft_mean_ms_all_arms_same_images': v3_t1,
                                'v3_T1_ttft_mean_ms_B_arm_same_images': v3[REF.V3_B]['T1_ttft_mean_ms'],
                                'ratio_this_over_v3_all_arms': out['T1_ttft_ms']['mean'] / v3_t1, 'ratio_this_over_v3_B_arm': out['T1_ttft_ms']['mean'] / v3[REF.V3_B]['T1_ttft_mean_ms'],
                                'packed_read_MB_per_s_this_run': out['packed_read_MB_per_s'],
                                'packed_read_MB_per_s_MTGQA_PYR_run': MTGQA_REFERENCE['packed_read_MB'] / (MTGQA_REFERENCE['packed_read_ms'] / 1e3),
                                'role': 'environment tracking, not a verdict input'}
    out['ssd_condition'] = {'PYR_MB_per_followup': out['SSD_MB_per_followup'], 'B_MB_per_followup_v3': v3[REF.V3_B]['SSD_MB_per_followup'],
                            'limit_MB': v3[REF.V3_B]['SSD_MB_per_followup'] * 1.02, 'within_B_times_1.02': out['SSD_MB_per_followup'] <= v3[REF.V3_B]['SSD_MB_per_followup'] * 1.02}
    return out


def session_storage(audits, method, v3_sessions, images):
    """3-turn session = 3 request E2E + persistence + activation (one dialogue per image, no amortization)."""
    by = {a['image_id']: a for a in audits}
    e2e = []; persist = []; act = []; store = []; alloc = []; session = []; write = []
    for i in images:
        a = by[i]; s = a['setup'][method]
        rows = [r for r in a['rows'] if r['method_id'] == method]
        req = sum(r['request_e2e_ms'] for r in rows)
        e2e.append(req); persist.append(s['persistence_ms']); act.append(s['activation_ms']); store.append(s['store_logical_bytes'] / 1e6)
        alloc.append(s['allocated_bytes'] / 1e6); write.append(s['write_bytes'] / 1e6); session.append(req + s['persistence_ms'] + s['activation_ms'])
    out = {'store_MB_per_image': stats(store), 'allocated_MB_per_image': stats(alloc), 'os_write_MB_per_image': stats(write), 'persistence_ms': stats(persist),
           'activation_ms': stats(act), 'request_e2e_sum_ms': stats(e2e), 'session_3turn_ms': stats(session),
           'definition': '3 request E2E + persistence (outer helper wall) + activation (outer wall); T1 capture is already inside T1 request E2E'}
    ref = {}
    for m in REF.COMPARE_METHODS:
        vals = [v3_sessions[(i, m)] for i in images if (i, m) in v3_sessions]
        if vals:
            ref[m] = {'session_mean_ms': float(np.mean(vals)), 'images': len(vals), 'column': 'standalone_equivalent_session_ms (AllHead: DERIVED)'}
    out['v3_cross_run_reference'] = ref
    out['v3_cross_run_note'] = 'cross-run 참고치; 같은 실행 안 기준은 T1 TTFT뿐이다'
    per_image = [{'image_id': i, 'method': method, 'request_e2e_sum_ms': e2e[j], 'persistence_ms': persist[j], 'activation_ms': act[j],
                  'session_3turn_ms': session[j], 'store_MB': store[j]} for j, i in enumerate(images)]
    return out, per_image


def decision(gdet, paired, lat, mtgqa=MTGQA_REFERENCE):
    """Mechanical verdict record.  Rules are frozen in the contract (section 8); nothing here looks at anything else."""
    d = find(paired, REF.V3_B, 'Followup'); rc = find(paired, 'recompute', 'Followup')
    if not all(find(paired, m, 'Followup')['ci_defined'] for m in REF.COMPARE_METHODS):
        raise ValueError('follow-up confidence interval undefined')
    b_minus_recomp = d['baseline_mean'] - rc['baseline_mean']          # negative: the loss of [B]
    comparable = gdet['status'] == 'PASS'
    out = {'G_det_T1': {'status': gdet['status'], 'T1_token_exact': gdet['T1_token_exact'], 'images': gdet['images']},
           'cross_run_accuracy_comparison': 'cross-run, 결정성 확인됨' if comparable else 'NOT_COMPARABLE',
           'A_layer_allocation_effect': {
               'D_definition': 'PYR-beta-raw_packed (this run) - ours_prefixstore_kv25 [B] (v3), follow-up official VQA score, percentage points',
               'D_pp': d['delta_pp'], 'ci95': [d['ci95_low'], d['ci95_high']], 'PYR_followup_pct': d['new_mean'], 'B_followup_pct': d['baseline_mean'],
               'rule': 'IMPROVED: CI low > 0; NOT_SHOWN: CI contains 0; WORSE: CI high < 0',
               'verdict': verdict(d['ci95_low'], d['ci95_high']) if comparable else 'PENDING_REVIEW',
               'wording_if_NOT_SHOWN': '차이 미입증 (동등성 아님)',
               'reference_not_verdict': {'D_at_least_1.0pp_MTGQA_GO_threshold': bool(d['delta_pp'] >= 1.0),
                                         'loss_recovery_percent': float(100 * d['delta_pp'] / (-b_minus_recomp)) if b_minus_recomp else None,
                                         'loss_definition': 'D / (ReComp - [B]) on the same images; ReComp - [B] = %.4f pp' % (-b_minus_recomp)},
               'condition_SSD_MB_per_followup_le_B_x_1.02': lat['ssd_condition']},
           'B_vs_original': {m: {'delta_pp': find(paired, m, 'Followup')['delta_pp'], 'ci95': [find(paired, m, 'Followup')['ci95_low'], find(paired, m, 'Followup')['ci95_high']],
                                 'baseline_followup_pct': find(paired, m, 'Followup')['baseline_mean'],
                                 'percent_of_baseline': 100 * find(paired, m, 'Followup')['new_mean'] / find(paired, m, 'Followup')['baseline_mean'],
                                 'ci_statement': 'CI excludes 0' if find(paired, m, 'Followup')['ci_excludes_zero'] else '차이 미입증 (동등성 아님)'}
                             for m in ('recompute', 'fullload', 'sparsevlm_ssd_kv25_allhead')},
           'F_MTGQA_consistency': {'MTGQA_PYR_minus_U_pp': mtgqa['pyr_minus_u_pp'], 'MTGQA_ci95': mtgqa['ci'], 'MTGQA_metric': mtgqa['metric'],
                                   'MTVQA_D_pp': d['delta_pp'], 'MTVQA_ci95': [d['ci95_low'], d['ci95_high']], 'MTVQA_metric': 'official VQA score',
                                   'same_sign': bool((d['delta_pp'] > 0) == (mtgqa['pyr_minus_u_pp'] > 0)), 'note': '지표가 다르다 (EM vs VQA score); 크기를 직접 비교하지 않는다'}}
    if not comparable:
        out['A_layer_allocation_effect']['note'] = 'G-det-T1 is not 100%: mismatches dumped, verdict PENDING_REVIEW, cross-run accuracy NOT_COMPARABLE'
    return out

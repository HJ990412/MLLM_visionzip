#!/usr/bin/env python3
"""Independent CPU raw audit/report; never imports a production selector/loader.

All writes are exclusive. Partial audit receipts do not make a pilot VALID.
The importable audit_image_rows permits durable, complete-cohort scratch cleanup.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
METHODS = ('recompute', 'fullload', 'sparsevlm_ssd_kv25_probe3',
           'sparsevlm_ssd_kv25_allhead', 'ours_kv25')
LABELS = dict(zip(METHODS, ('ReComp', 'FullLoad', 'SparseVLM-SSD-KV25-Probe3',
                           'SparseVLM-SSD-KV25-AllHead', 'Ours-KV25')))
POLICIES = {METHODS[2]: 'fixed_first_3', METHODS[3]: 'all'}
CATEGORIES = ('scoring_probe_k', 'scoring_full_k', 'selected_k', 'selected_v', 'structural')
EXPECTED_COUNTS = {'smoke': 60, 'gqa': 1200, 'mt': 600}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def normalize(text):
    value = re.sub(r'[^\w\s]', ' ', str(text).lower())
    return ' '.join(w for w in value.split() if w not in {'a', 'an', 'the'})


def independent_score(prediction, gold, phase="gqa"):
    answer = gold[0] if isinstance(gold, (list, tuple)) else gold
    p, g = normalize(prediction), normalize(answer)
    return float(p == g) if phase == 'mt' else float(p == g or bool(g and p.split()[:len(g.split())] == g.split()))


class Checks:
    def __init__(self):
        self.count = 0
        self.failures = Counter()
        self.examples = defaultdict(list)

    def require(self, condition, name, detail=None):
        self.count += 1
        if not condition:
            self.failures[name] += 1
            if len(self.examples[name]) < 8: self.examples[name].append(detail)

    def result(self):
        return {'status': 'PASS' if not self.failures else 'FAIL',
                'checks_evaluated': self.count, 'failure_counts': dict(self.failures),
                'failure_examples': dict(self.examples)}


def _entries(manifest, phase):
    workloads = manifest.get('workloads', {})
    value = workloads.get('mt' if phase == 'mt' else 'gqa', [])
    if isinstance(value, dict): value = value.get('images', value.get('entries', []))
    if phase == 'smoke': value = value[:4]
    return value


def expected_requests(manifest, phase):
    result = {}
    for entry in _entries(manifest, phase):
        questions = entry.get('turns', []) if phase == 'mt' else entry.get('questions', [])
        if phase == 'smoke': questions = questions[:3]
        for ordinal, question in enumerate(questions, 1):
            tid = int(question.get('turn_id', ordinal))
            did = entry.get('dialog_id') if phase == 'mt' else None
            for method in METHODS:
                rid = f"{phase}:{did or entry['image_id']}:{tid}:{method}"
                result[rid] = {'image_id': entry['image_id'], 'dialog_id': did,
                               'question_id': str(question['question_id']), 'turn_id': tid,
                               'method_id': method, 'question': question['question'],
                               'gold': question.get('answers', question.get('gold', question.get('answer')))}
    return result


def _io_from_trace(row):
    totals = Counter()
    trace = row.get('os_pread_trace', [])
    method = row['method_id']
    for event in trace:
        filename = Path(event['path']).name
        if filename == 'probe_k.bin': category = 'scoring_probe_k'
        elif filename == 'k.bin': category = 'scoring_full_k' if method == METHODS[3] else 'selected_k'
        elif filename == 'v.bin': category = 'selected_v'
        else: category = 'structural'
        totals[category] += int(event['returned'])
    return {name: totals[name] for name in CATEGORIES}, len(trace)


def _event_key(event, ledger=False):
    if ledger:
        return (event.get('layer'), event['file'], int(event['offset']),
                int(event['requested_bytes']), int(event['returned_bytes']))
    match = re.search(r'(?:^|/)layer_(\d+)(?:/|$)', event['path'])
    return (int(match[1]) if match else None, Path(event['path']).name,
            int(event['offset']), int(event['requested']), int(event['returned']))


def _ranges(chunks, visual_count, width):
    result = []
    for chunk in chunks:
        lo, hi = chunk * 64, min(visual_count, (chunk + 1) * 64)
        if result and result[-1][1] == lo: result[-1][1] = hi
        else: result.append([lo, hi])
    return result, [(lo * width, (hi-lo) * width) for lo, hi in result]


def _geometry(row):
    g = row.get('geometry', {})
    result = row.get('result', {})
    layers = result.get('layers', [])
    shape = layers[0].get('cache_shape', []) if layers else []
    plan = layers[0].get('plan', {}) if layers else {}
    return {
        'heads': int(g.get('num_heads', shape[1] if len(shape) == 4 else 0)),
        'dim': int(g.get('head_dim', shape[3] if len(shape) == 4 else 0)),
        'layers': int(g.get('num_layers', len(layers))),
        'visual': int(g.get('v_token_num', plan.get('v_num', 0))),
        'structural': list(g.get('newline_idx', plan.get('structural_ids', []))),
        'padding': list(g.get('padding_idx', plan.get('padding_ids', []))),
        'dtype': g.get('dtype', result.get('ssd_dtype')),
    }


def audit_sparse_hit(row, check):
    rid, result = row['request_id'], row['result']
    method = row['method_id']; g = _geometry(row)
    heads, dim, nl, vn = (g[k] for k in ('heads', 'dim', 'layers', 'visual'))
    structural, padding = set(g['structural']), set(g['padding'])
    n = vn-len(structural|padding); k = (n+3)//4
    check.require(min(heads, dim, nl, vn, n) > 0, 'positive_geometry', rid)
    check.require(not (structural & padding), 'structural_padding_disjoint', rid)
    check.require(all(0 <= i < vn for i in structural|padding), 'geometry_ids_in_range', rid)
    check.require(g['dtype'] in ('float16', 'torch.float16'), 'frozen_ssd_dtype', rid)
    width = heads*dim*2
    selected_heads = [0, 1, 2] if method == METHODS[2] else list(range(heads))
    check.require(result.get('head_policy') == POLICIES[method], 'explicit_head_policy', rid)
    check.require(result.get('scoring_head_ids') == selected_heads, 'actual_score_head_ids', rid)
    check.require(row.get('scoring_head_count') == len(selected_heads), 'normalized_head_count', rid)
    check.require(row.get('N_content') == n and row.get('k') == k, 'normalized_exact_budget', rid)
    check.require(math.isclose(float(row.get('content_kv_fraction', -1)), k/n,
                               rel_tol=0, abs_tol=1e-12), 'normalized_content_fraction', rid)
    layers = result.get('layers', [])
    check.require(len(layers) == nl and {int(x['layer']) for x in layers} == set(range(nl)),
                  'every_layer_present', rid)
    check.require(result.get('scoring_calls') == nl, 'one_score_each_layer', rid)
    expected_events = Counter()
    if structural:
        size = 2*nl*len(structural)*width
        expected_events[(None, 'sep_kv.bin', 0, size, size)] += 1
    duplicated_bytes = 0; reused_bytes = 0
    for layer in layers:
        li = int(layer['layer']); detail = [rid, li]
        selected = layer.get('selected_token_ids', [])
        check.require(selected == sorted(set(selected)), 'selected_sorted_unique', detail)
        check.require(len(selected) == k and layer.get('k') == k and layer.get('n_content') == n,
                      'per_layer_exact_budget', detail)
        check.require(all(0 <= i < vn and i not in structural|padding for i in selected),
                      'selected_real_original_ids', detail)
        check.require(layer.get('scoring_head_ids') == selected_heads, 'layer_head_policy', detail)
        check.require(layer.get('scoring_calls') == 1, 'layer_scoring_once', detail)
        chunks = sorted({i//64 for i in selected})
        runs, byte_ranges = _ranges(chunks, vn, width)
        read = [i for c in chunks for i in range(c*64, min(vn, (c+1)*64))]
        check.require(layer.get('selected_chunk_ids') == chunks, 'chunks_from_original_tokens', detail)
        plan = layer.get('plan', {})
        fields = {'N_content': n, 'k': k, 'v_num': vn, 'chunk_size': 64,
                  'selected_tokens': selected, 'selected_chunks': chunks,
                  'chunk_runs': runs, 'read_rows': read,
                  'read_real_rows': sum(i not in structural|padding for i in read),
                  'extra_real_rows': sum(i not in structural|padding for i in read)-k,
                  'read_structural_rows': sum(i in structural for i in read),
                  'read_padding_rows': sum(i in padding for i in read),
                  'keep_tokens': sorted(set(selected)|structural)}
        for field, value in fields.items():
            check.require(plan.get(field) == value, 'independent_plan_'+field, detail)
        shape = layer.get('cache_shape', [])
        check.require(len(shape) == 4 and shape[0] == 1 and shape[1] == heads and shape[3] == dim,
                      'all_heads_dense_cache_shape', detail)
        check.require(layer.get('keep_visual_rows') == k+len(structural), 'actual_keep_rows', detail)
        calls = result.get('projection_calls', {}).get(str(li), {})
        check.require(calls.get('prefill') == {'q': 1, 'k': 1, 'v': 1}, 'projection_reuse', detail)
        prefix_bytes = vn*len(selected_heads)*dim*2
        file = 'probe_k.bin' if method == METHODS[2] else 'k.bin'
        expected_events[(li, file, 0, prefix_bytes, prefix_bytes)] += 1
        for offset, size in byte_ranges:
            expected_events[(li, 'v.bin', offset, size, size)] += 1
            if method == METHODS[2]: expected_events[(li, 'k.bin', offset, size, size)] += 1
        # Count repeated elements after their first physical transfer. Three-file
        # overlap counts two extra transfers, not three unordered pairs.
        if method == METHODS[2]:
            duplicate = (len(read)*3 + len(structural)*3 +
                         sum(i in structural for i in read)*(heads-3))*dim*2
        else:
            duplicate = len(structural)*heads*dim*2
            reused_bytes += (k+len(structural))*width
        duplicated_bytes += duplicate
        check.require(layer.get('io', {}).get('duplicated_k_bytes') == duplicate,
                      'independent_duplicated_k_bytes', detail)
        check.require(layer.get('io', {}).get('selected_k_read_bytes') ==
                      (0 if method == METHODS[3] else len(read)*width),
                      'selected_k_read_contract', detail)
        check.require(layer.get('io', {}).get('selected_v_read_bytes') == len(read)*width,
                      'selected_v_read_contract', detail)
    trace = row.get('os_pread_trace', [])
    observed = Counter(_event_key(e) for e in trace)
    check.require(observed == expected_events, 'os_ranges_equal_independent_plan', rid)
    ledger = result.get('io', {}).get('events', [])
    check.require(Counter(_event_key(e, True) for e in ledger) == observed,
                  'production_ledger_equal_actual_os', rid)
    categories, preads = _io_from_trace(row)
    io_result = result.get('io', {})
    for category, size in categories.items():
        check.require(io_result.get('per_kind', {}).get(category, {}).get('bytes') == size,
                      'disjoint_bytes_'+category, rid)
    check.require(io_result.get('duplicated_k_bytes') == duplicated_bytes,
                  'duplicate_total_not_double_summed', rid)
    check.require(io_result.get('logical_reused_k_bytes') == reused_bytes,
                  'logical_reused_k_bytes', rid)
    check.require(io_result.get('bytes') == sum(categories.values()) and io_result.get('preads') == preads,
                  'payload_counter_totals', rid)
    if method == METHODS[3]:
        check.require(categories['scoring_probe_k'] == 0 and categories['selected_k'] == 0,
                      'allhead_no_probe_or_selected_k_reread', rid)
    raters = result.get('rater_ids', []); suffix = result.get('suffix_input_ids', [])
    check.require(bool(raters) and raters == sorted(set(raters)) and
                  all(0 <= i < len(suffix) for i in raters), 'valid_rater_ids', rid)
    check.require(result.get('rater_scope') == 'entire_actual_suffix_after_expanded_visual_block',
                  'rater_scope', rid)
    check.require(result.get('rater_visual_scope_includes_newlines') is True, 'visual_rater_scope', rid)
    check.require(not result.get('diagnostic', False), 'no_teacher_diagnostic_in_timing', rid)


def audit_existing_hit(row, check):
    rid=row['request_id']; result=row.get('result',{}); g=_geometry(row)
    h,d,nl,vn=(g[k] for k in ('heads','dim','layers','visual'))
    width=h*d*2; structural=set(g['structural']); n=vn-len(structural)-len(g['padding'])
    check.require(min(h,d,nl,vn,n)>0,'existing_arm_geometry',rid)
    expected=Counter()
    if row['method_id']=='fullload':
        for li in range(nl):
            for file in ('k.bin','v.bin'):expected[(li,file,0,vn*width,vn*width)]+=1
        check.require(row['k']==n and row['content_kv_fraction']==1,'fullload_full_content',rid)
    else:
        k=(n+3)//4; end=min(vn,((k+63)//64)*64); size=end*width
        for li in range(nl):
            for file in ('k.bin','v.bin'):expected[(li,file,0,size,size)]+=1
        if structural:
            size=2*nl*len(structural)*width
            expected[(None,'sep_kv.bin',0,size,size)]+=1
        check.require(result.get('selected_stored_ids')==list(range(k)),'ours_original_prefix_contract',rid)
        original=result.get('selected_original_ids',[])
        check.require(len(original)==k and len(set(original))==k and
                      all(0<=i<vn and i not in structural for i in original),'ours_original_ids_real',rid)
        check.require(result.get('attended_content_kv_count_per_layer')==[k]*nl,'ours_actual_per_layer_keep',rid)
        check.require(result.get('keep_count_per_layer')==[k+len(structural)]*nl,'ours_structural_keep',rid)
    observed=Counter(_event_key(e) for e in row.get('os_pread_trace',[]))
    check.require(observed==expected,'existing_arm_independent_os_ranges',rid)
    check.require(result.get('scoring_calls')==0,'existing_arm_online_scoring_zero',rid)


def audit_image_rows(rows, manifest, config):
    """Audit complete supplied dialogue/image cohorts, not all pilot membership.

    PASS permits only the runner's separately allowlisted scratch cleanup. It
    does not indicate that unprovided images or the whole pilot have completed.
    """
    check = Checks(); by_id = {}; expected_by_phase = {}
    phases = {r.get('phase') for r in rows}
    for phase in phases:
        check.require(phase in EXPECTED_COUNTS, 'known_phase', phase)
        expected_by_phase[phase] = expected_requests(manifest, phase)
    supplied_groups = defaultdict(set)
    score_count = 0
    for row in rows:
        rid = row.get('request_id')
        try:
            check.require(rid not in by_id, 'unique_logical_request', rid); by_id[rid] = row
            phase, method = row['phase'], row['method_id']
            expected = expected_by_phase[phase].get(rid)
            check.require(expected is not None, 'request_in_frozen_manifest', rid)
            if expected is None: continue
            for field in ('image_id', 'question_id', 'turn_id', 'method_id', 'question'):
                check.require(str(row.get(field)) == str(expected[field]), 'identity_'+field, rid)
            check.require((row.get('dialog_id') or None) == expected['dialog_id'], 'identity_dialog_id', rid)
            if expected['gold'] is not None:
                gold = row['gold'] if isinstance(row['gold'], list) else [row['gold']]
                expected_gold = expected['gold'] if isinstance(expected['gold'], list) else [expected['gold']]
                check.require(gold == expected_gold, 'gold_from_frozen_manifest', rid)
            score = independent_score(row['prediction'], row['gold'], row.get('phase','gqa')); score_count += 1
            check.require(float(row['score']) == score, 'independent_answer_score', rid)
            group = (phase, row.get('dialog_id') or row['image_id'])
            supplied_groups[group].add(rid)
            for name in ('experiment_id', 'attempt_id', 'config_sha256', 'manifest_sha256', 'code_sha256'):
                check.require(bool(row.get(name)), 'raw_provenance_'+name, rid)
            for rawkey, expectedkey in (('config_sha256', '_file_sha256'), ('manifest_sha256', '_manifest_sha256')):
                if config.get(expectedkey): check.require(row.get(rawkey) == config[expectedkey], rawkey, rid)
            source_hashes = config.get('source_sha256', config.get('code_hashes'))
            if source_hashes: check.require(row.get('code_sha256') == canonical_sha(source_hashes), 'code_sha256', rid)
            check.require(row.get('status', 'ok') in ('ok', 'PASS', 'success', 'completed'), 'successful_row', rid)
            ttft, e2e = float(row['ttft_ms']), float(row['request_e2e_ms'])
            check.require(math.isfinite(ttft) and ttft > 0 and math.isfinite(e2e) and e2e >= ttft,
                          'positive_finite_times', rid)
            started = row.get('request_started_at_s'); first = row.get('first_token_at_s')
            finished = row.get('request_finished_at_s')
            check.require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in (started, first, finished)),
                          'outer_timestamp_evidence', rid)
            if all(isinstance(v, (int, float)) for v in (started, first, finished)):
                check.require(abs((first-started)*1000-ttft) <= 1e-5, 'ttft_recomputed_from_outer_boundary', rid)
                check.require(abs((finished-started)*1000-e2e) <= 1e-5, 'e2e_recomputed_from_outer_boundary', rid)
            categories, calls = _io_from_trace(row)
            check.require(row['ssd_read_bytes'] == sum(categories.values()), 'actual_os_return_bytes', rid)
            check.require(row['ssd_preads'] == calls, 'actual_os_pread_calls', rid)
            for event in row.get('os_pread_trace', []):
                check.require(event['returned'] == event['requested'] and event['returned'] > 0 and event['offset'] >= 0,
                              'actual_complete_read_event', rid)
            hit = int(row['turn_id']) > 1
            result = row.get('result', {})
            if method == 'recompute' or not hit:
                check.require(calls == 0, 'normal_pixel_no_ssd_payload', rid)
            if hit and method != 'recompute':
                check.require(result.get('vision_forward_calls', result.get('vision_forward_count')) == 0,
                              'hit_vision_zero', rid)
            else:
                check.require(result.get('vision_forward_calls', result.get('vision_forward_count')) == 1,
                              'normal_image_vision_one', rid)
            check.require(bool(row.get('source_T1')), 'source_T1_provenance', rid)
            if hit and method in POLICIES: audit_sparse_hit(row, check)
            if hit and method in ('fullload','ours_kv25'): audit_existing_hit(row,check)
            if hit and method == 'ours_kv25':
                n = int(row['N_content']); k = (n+3)//4
                check.require(n > 0 and row['k'] == k and
                              math.isclose(row['content_kv_fraction'], k/n, rel_tol=0, abs_tol=1e-12),
                              'ours_exact_content_kv25', rid)
            if phase != 'mt':
                check.require(not row.get('history_entries') and not row.get('history'), 'independent_gqa_no_history', rid)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            check.require(False, 'malformed_raw_row', [rid, repr(exc)])
    for (phase, cohort), actual in supplied_groups.items():
        expected = {rid for rid, item in expected_by_phase[phase].items()
                    if (item['dialog_id'] or item['image_id']) == cohort}
        check.require(actual == expected, 'complete_supplied_cohort', [phase, cohort, len(actual), len(expected)])
    for row in rows:
        if row.get('phase') != 'mt': continue
        rid = row.get('request_id'); method = row.get('method_id'); did = row.get('dialog_id')
        expected_entries = []
        try:
            for prior_turn in range(1, int(row['turn_id'])):
                prior = by_id.get(f'mt:{did}:{prior_turn}:{method}')
                check.require(prior is not None, 'method_local_prior_row_exists', rid)
                if prior is None: continue
                expected_entries.append({'turn_id': prior_turn, 'question_id': prior['question_id'],
                                         'question': prior['question'], 'answer': prior['prediction'],
                                         'answer_source': 'method_local_generated'})
            check.require(row.get('history_entries', []) == expected_entries, 'method_local_generated_history', rid)
            expected_history = '\n'.join(line for e in expected_entries for line in
                                         (f"Q{e['turn_id']}: {e['question']}", f"A{e['turn_id']}: {e['answer']}"))
            check.require(row.get('history', '') == expected_history, 'history_rendering', rid)
            body = ([expected_history, ''] if expected_history else []) + [
                f"Current question Q{row['turn_id']}: {row['question']}",
                'Answer the current question with a single word or short phrase. ASSISTANT:']
            check.require(row.get('prompt') == 'USER: <image>\n'+'\n'.join(body), 'causal_mt_prompt_exact', rid)
        except (KeyError, TypeError, ValueError) as exc:
            check.require(False, 'malformed_history', [rid, repr(exc)])
    # Same-prompt head policies must share rater identity and canonical source.
    for row in rows:
        if row.get('method_id') != METHODS[2] or row.get('turn_id', 0) <= 1: continue
        peer_id = row['request_id'].rsplit(':', 1)[0]+':'+METHODS[3]
        peer = by_id.get(peer_id)
        if peer:
            check.require(row.get('source_T1') == peer.get('source_T1'), 'canonical_T1_shared_by_head_policies', row['request_id'])
            if row.get('prompt') == peer.get('prompt'):
                check.require(row.get('result', {}).get('rater_ids') == peer.get('result', {}).get('rater_ids'),
                              'same_prompt_same_raters', row['request_id'])
    result = check.result()
    result.update(scope='complete_provided_cohorts_only', rows=len(rows), independently_rescored=score_count,
                  completed_cohorts=len(supplied_groups), final_pilot_valid=False)
    return result


def _mean(values):
    return sum(values)/len(values) if values else None


def summaries(rows, phase):
    out = []
    for method in METHODS:
        selected = [r for r in rows if r['method_id'] == method]
        hits = [r for r in selected if int(r['turn_id']) > 1]
        t1 = [r for r in selected if int(r['turn_id']) == 1]
        categories = [_io_from_trace(r)[0] for r in hits]
        item = {'phase': phase, 'method_id': method, 'requests': len(selected), 'hits': len(hits),
                'scoring_head_count': _mean([r['scoring_head_count'] for r in hits]),
                'all_accuracy': _mean([independent_score(r['prediction'], r['gold'], r.get('phase','gqa')) for r in selected]),
                'hit_accuracy': _mean([independent_score(r['prediction'], r['gold'], r.get('phase','gqa')) for r in hits]),
                't1_ttft_ms': _mean([r['ttft_ms'] for r in t1]), 'hit_ttft_ms': _mean([r['ttft_ms'] for r in hits]),
                'content_kv_fraction': _mean([r['content_kv_fraction'] for r in hits]),
                'ssd_mb_per_hit': _mean([sum(c.values())/1e6 for c in categories]),
                'preads_per_hit': _mean([len(r.get('os_pread_trace', [])) for r in hits])}
        for name in CATEGORIES: item[name+'_mb'] = _mean([c[name]/1e6 for c in categories])
        for turn in range(1, 7):
            item[f't{turn}_accuracy'] = _mean([independent_score(r['prediction'], r['gold'], r.get('phase','gqa')) for r in selected if r['turn_id'] == turn])
        item['peak_gpu_allocated_bytes'] = max((r.get('result', {}).get('peak_gpu_allocated_bytes', 0) for r in selected), default=0) or None
        item['peak_gpu_reserved_bytes'] = max((r.get('result', {}).get('peak_gpu_reserved_bytes', 0) for r in selected), default=0) or None
        item['selector_host_ms'] = _mean([sum(l.get('selector_host_ms', 0) for l in r.get('result', {}).get('layers', [])) for r in hits])
        item['projection_prefill_calls_per_layer'] = 3 if hits and method in POLICIES and all(r.get('result', {}).get('projection_reuse') for r in hits) else None
        out.append(item)
    return out


def paired_bootstrap(rows, phase, resamples=10000, seed=1234):
    """Request-weighted paired mean, resampling whole image clusters together."""
    import numpy as np
    indexed = {(r['dialog_id'] or r['image_id'], int(r['turn_id']), r['method_id']): r for r in rows}
    pairs = [(METHODS[2], METHODS[3])] + [(m, METHODS[4]) for m in METHODS[:4]]
    output = []
    for a, b in pairs:
        for scope in ('all', 'hit'):
            cluster = defaultdict(list)
            for row in rows:
                if row['method_id'] != a or (scope == 'hit' and row['turn_id'] == 1): continue
                peer = indexed.get((row['dialog_id'] or row['image_id'], row['turn_id'], b))
                if peer is None: continue
                cluster[row['image_id']].append((
                    independent_score(row['prediction'], row['gold'], row.get('phase','gqa'))-independent_score(peer['prediction'], peer['gold'], peer.get('phase','gqa')),
                    row['ttft_ms']-peer['ttft_ms'], row['ssd_read_bytes']/1e6-peer['ssd_read_bytes']/1e6))
            if not cluster: continue
            ids = sorted(cluster)
            counts = np.array([len(cluster[i]) for i in ids], dtype=np.float64)
            sums = np.array([np.array(cluster[i], dtype=np.float64).sum(0) for i in ids])
            point = sums.sum(0)/counts.sum()
            rng = np.random.default_rng(seed)
            choices = rng.integers(0, len(ids), size=(resamples, len(ids)))
            boot = sums[choices].sum(1)/counts[choices].sum(1)[:, None]
            lo, hi = np.quantile(boot, [.025, .975], axis=0)
            for i, metric in enumerate(('quality', 'ttft_ms', 'ssd_mb')):
                output.append({'phase': phase, 'scope': scope, 'a': a, 'b': b, 'metric': metric,
                               'difference_a_minus_b': float(point[i]), 'ci95_low': float(lo[i]),
                               'ci95_high': float(hi[i]), 'paired_requests': int(counts.sum()),
                               'image_clusters': len(ids), 'resamples': resamples, 'seed': seed,
                               'weighting': 'request_weighted_complete_image_clusters'})
    return output


def _fmt(value, digits=3, percent=False):
    return 'NOT RUN' if value is None else f'{100*value if percent else value:.{digits}f}'


def render_report(audit, summary, paired, run_dir):
    lines = ['# SparseVLM-SSD-KV25 Probe3 / AllHead', '',
             f'Run: `{run_dir}`. Independent raw audit: **{audit["status"]}**.', '',
             '공식 SparseVLM의 embedding rater와 causal full-context attention head/rater 평균을 SSD-resident cached KV에 적용한 fixed-budget adaptation이다. Probe3는 head [0,1,2] 근사이고 AllHead는 전체 heads를 읽고 평균한다. Progressive pruning/layer schedule/recycling 및 SparseVLM+ head-selection/위치 보정은 재현 범위에 포함되지 않는다.', '']
    for phase, title in (('gqa', 'GQA'), ('mt', 'MT-GQA-reconstructed')):
        state = audit['phases'].get(phase, {}).get('pilot_status', 'NOT RUN')
        lines += [f'## {title}: {state}', '',
                  '| 방법 | 점수 head 수 | 전체/Hit 정답률 | T1/Hit TTFT | 실제 content KV % | SSD MB/hit |',
                  '|---|---:|---:|---:|---:|---:|']
        items = {r['method_id']: r for r in summary if r['phase'] == phase}
        for method in METHODS:
            r = items.get(method, {})
            lines.append(f'| {LABELS[method]} | {_fmt(r.get("scoring_head_count"),0)} | {_fmt(r.get("all_accuracy"),2,True)} / {_fmt(r.get("hit_accuracy"),2,True)} | {_fmt(r.get("t1_ttft_ms"),2)} / {_fmt(r.get("hit_ttft_ms"),2)} ms | {_fmt(r.get("content_kv_fraction"),3,True)} | {_fmt(r.get("ssd_mb_per_hit"))} |')
        lines += ['', '| 방법 | Scoring K MB | 추가 selected-K MB | Selected-V MB | Structural/other MB | Total MB | Preads |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for method in METHODS:
            r = items.get(method, {})
            scoring = None if not r else (r.get('scoring_probe_k_mb') or 0)+(r.get('scoring_full_k_mb') or 0)
            lines.append(f'| {LABELS[method]} | {_fmt(scoring)} | {_fmt(r.get("selected_k_mb"))} | {_fmt(r.get("selected_v_mb"))} | {_fmt(r.get("structural_mb"))} | {_fmt(r.get("ssd_mb_per_hit"))} | {_fmt(r.get("preads_per_hit"),2)} |')
        lines += ['', '단위 MB는 OS pread 반환 bytes / 1,000,000이다. FullLoad의 K는 additional-selected-K 열에 실제 읽은 canonical K로 계상하며 scoring K가 아니다. 구조 중복은 실제 이벤트에 한 번씩 포함된다. KV retention과 SSD read 비율은 다르다.', '']
    lines += ['## 검증 및 비용 범위', '', '| 항목 | 상태/측정 범위 |', '|---|---|',
              f'| CPU TEST | {audit["gates"].get("CPU_TEST", "NOT RUN")} |',
              f'| GPU CORRECTNESS | {audit["gates"].get("GPU_CORRECTNESS", "NOT RUN")} |',
              f'| PROJECTION REUSE | {audit["gates"].get("PROJECTION_REUSE", "NOT RUN")} |',
              f'| ALLHEAD K-READ REUSE | {audit["gates"].get("ALLHEAD_K_READ_REUSE", "NOT RUN")} |',
              f'| LEGACY/OURS/QWEN CPU | {audit["gates"].get("LEGACY_PROTECTION", "NOT RUN")} |',
              f'| QWEN GPU | {audit["gates"].get("QWEN_GPU", "NOT RUN")} |',
              '| Persistence/session | RO cache-hit reevaluation은 NOT_REMEASURED; fresh MT shared-bundle 비용만 per-image receipt에서 측정; AllHead base/Probe-only 생성비 분리는 NOT SEPARATELY MEASURED |',
              '| Metadata activation | 요청 TTFT 밖; activation bytes/residency/time는 raw에서 별도 보존 |',
              '| TTFT | 외부 request start → 첫 token materialization 및 CUDA sync; input preparation와 online scoring K read 포함 |',
              '| Stage times | selector/actual-attention host/CUDA intervals는 겹칠 수 있어 TTFT로 합산하지 않음 |', '']
    lines += ['| Phase / 방법 | Selector host ms/hit | Prefill Q/K/V calls/layer | Peak allocated MiB | Peak reserved MiB |', '|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'| {r["phase"]} / {LABELS[r["method_id"]]} | {_fmt(r.get("selector_host_ms"))} | {_fmt(r.get("projection_prefill_calls_per_layer"),0)} | {_fmt(r["peak_gpu_allocated_bytes"]/2**20 if r.get("peak_gpu_allocated_bytes") else None,2)} | {_fmt(r["peak_gpu_reserved_bytes"]/2**20 if r.get("peak_gpu_reserved_bytes") else None,2)} |')
    lines += ['', '## Paired 95% CI', '', '| Dataset/scope | Pair A−B | Metric | Point | 95% CI | N / image clusters |', '|---|---|---|---:|---|---:|']
    for p in paired:
        lines.append(f'| {p["phase"]}/{p["scope"]} | {LABELS[p["a"]]} − {LABELS[p["b"]]} | {p["metric"]} | {p["difference_a_minus_b"]:.5f} | [{p["ci95_low"]:.5f}, {p["ci95_high"]:.5f}] | {p["paired_requests"]} / {p["image_clusters"]} |')
    if not paired: lines.append('| NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |')
    lines += ['', 'Bootstrap은 image-cluster 10,000회, seed=1234이며 같은 image의 모든 dialogue/turn과 paired methods를 함께 resample하고 request-weighted point estimate와 동일 분모를 사용한다. CI가 0을 포함해도 동등성 증거가 아니다. MT 차이는 method별 generated history의 영향까지 포함한다.', '',
              '## 최종 판정', '',
              f'- IMPLEMENTATION PROBE3 / ALLHEAD: {audit["gates"].get("IMPLEMENTATION", "PARTIAL")}',
              '- SOURCE-METRIC FIDELITY: strict-> embedding raters 및 post-softmax head/rater mean 확인; fixed KV25, SSD online retrieval, FP32 reduction, stable ties는 의도한 adaptation.',
              f'- GQA 5-ARM PILOT: {audit["phases"].get("gqa", {}).get("pilot_status", "NOT RUN")}',
              f'- MT 5-ARM PILOT: {audit["phases"].get("mt", {}).get("pilot_status", "NOT RUN")}',
              f'- READY FOR LLAVA MT-GQA BASELINE INTEGRATION: {"YES" if audit["ready"] else "NO"}',
              f'- Readiness 근거: {audit["readiness_reason"]}', '',
              '구체적인 checks/failures와 source/hash/protection 상태는 `independent_audit.json`, 요청별 내역은 원 run의 phase/raw.jsonl, 통계는 `summary.csv`와 `paired_comparisons.csv`에 보존한다. 이 보고서는 미측정을 PASS로 승격하지 않는다.', '']
    return '\n'.join(lines)


def exclusive_text(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: f.write(text)


def csv_text(rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    if not fields: return ''
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows); return stream.getvalue()


def _receipt(run, paths):
    for name in paths:
        path = run/name
        if path.is_file(): return json.loads(path.read_text()), {'path': str(path), 'sha256': sha(path)}
    return {}, None


def validate_manifest_sources(manifest, check):
    if not manifest:
        return
    expected_index_sha = '514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a'
    index = ROOT/'data/index.json'
    check.require(index.is_file() and sha(index) == expected_index_sha, 'original_gqa_index_hash')
    if index.is_file():
        original = json.loads(index.read_text())[:40]
        expected = [(e['image_id'], [(str(q['question_id']),q['question']) for q in e['questions'][4:10]]) for e in original]
        actual = [(e['image_id'], [(str(q['question_id']),q['question']) for q in e['questions']]) for e in _entries(manifest,'gqa')]
        check.require(actual == expected, 'frozen_gqa_membership_order_and_questions')
    source = ROOT/'runs/qwen25_port_20260928T054537Z/mt_manifest/manifest.json'
    expected_mt_sha = 'b76425302ad7de6000b9ac3079341120b367c8c5e70a1478469383709a9252b6'
    check.require(source.is_file() and sha(source) == expected_mt_sha, 'original_mt_manifest_hash')
    if source.is_file():
        original = json.loads(source.read_text())['images']
        def signature(entries):
            return [(e['image_id'],e['dialog_id'],[(str(t['question_id']),t['question']) for t in e['turns']]) for e in entries]
        check.require(signature(_entries(manifest,'mt')) == signature(original), 'frozen_mt_membership_order_and_questions')


def breakdown_rows(rows):
    output = []
    for row in rows:
        result=row.get('result',{}); categories,calls=_io_from_trace(row)
        timing=result.get('timing',{}); prefill=Counter(); decode=Counter()
        for layer in result.get('projection_calls',{}).values():
            prefill.update(layer.get('prefill',{})); decode.update(layer.get('decode',{}))
        activation=result.get('metadata_activation',{})
        item={k:row.get(k) for k in ('phase','method_id','request_id','image_id','dialog_id','turn_id','ttft_ms','request_e2e_ms','N_content','k','content_kv_fraction')}
        item.update({name+'_bytes':value for name,value in categories.items()})
        item.update(os_preads=calls,os_returned_bytes=sum(categories.values()),
                    logical_reused_k_bytes=result.get('io',{}).get('logical_reused_k_bytes'),
                    duplicated_k_bytes=result.get('io',{}).get('duplicated_k_bytes'),
                    payload_h2d_compute_bytes_subtotal=result.get('payload_h2d_compute_bytes'),
                    rater_visual_h2d_bytes=result.get('rater_visual_h2d_bytes'),
                    other_h2d_bytes='NOT MEASURED',
                    peak_gpu_allocated_bytes=result.get('peak_gpu_allocated_bytes'),
                    peak_gpu_reserved_bytes=result.get('peak_gpu_reserved_bytes'),
                    host_metadata_bytes=activation.get('host_tensor_bytes'),
                    gpu_metadata_bytes=activation.get('gpu_tensor_bytes'),
                    rater_host_ms=result.get('rater_host_ms'),
                    selector_host_ms=sum(x.get('selector_host_ms',0) for x in result.get('layers',[])) if result.get('layers') else None,
                    actual_attention_prefill_host_ms=timing.get('host_intervals_ms',{}).get('actual_attention_prefill'),
                    actual_attention_prefill_cuda_ms=timing.get('cuda_intervals_ms',{}).get('actual_attention_prefill'),
                    actual_attention_decode_host_ms=timing.get('host_intervals_ms',{}).get('actual_attention_decode'),
                    actual_attention_decode_cuda_ms=timing.get('cuda_intervals_ms',{}).get('actual_attention_decode'),
                    stage_intervals_nonadditive_json=json.dumps(timing,sort_keys=True),
                    prefill_projection_q=prefill['q'],prefill_projection_k=prefill['k'],prefill_projection_v=prefill['v'],
                    decode_projection_q=decode['q'],decode_projection_k=decode['k'],decode_projection_v=decode['v'])
        output.append(item)
    return output


def controlled_rows(gpu):
    out=[]
    def walk(value,path):
        if isinstance(value,dict):
            diagnostic=value.get('controlled_head_diagnostic')
            if isinstance(diagnostic,dict):out.append({'receipt_path':path,**diagnostic})
            for key,child in value.items():
                if key!='controlled_head_diagnostic':walk(child,path+'/'+str(key))
        elif isinstance(value,list):
            for i,child in enumerate(value):walk(child,path+'/'+str(i))
    walk(gpu,'gpu_validation')
    return out


def setup_rows(run):
    out=[]
    for phase in ('smoke','gqa','mt'):
        for path in sorted((run/phase/'images').glob('**/*_persistence.json')):
            data=json.loads(path.read_text())
            out.append({'phase':phase,'receipt':str(path),'receipt_sha256':sha(path),'measurement_status':'MEASURED_ATTEMPT_COST',
                        'values_json':json.dumps(data,sort_keys=True),'note':'attempt-level cost; not selected by favorable outcome'})
        if phase!='mt':out.append({'phase':phase,'measurement_status':'NOT_REMEASURED','note':'existing stores reused read-only; no cold-start persistence/session claim'})
    return out


def run_audit(run_dir, output_dir, phases=('smoke','gqa','mt'), allow_partial=False):
    run = Path(run_dir); output = Path(output_dir)
    targets = ['independent_audit.json', 'REPORT.md', 'summary.csv', 'paired_comparisons.csv', 'io_timing_memory_breakdown.csv', 'controlled_head_policy.csv', 'setup_costs.csv']
    if any((output/name).exists() for name in targets):
        raise FileExistsError('report artifacts already exist; choose a new output-dir')
    config, config_ref = _receipt(run, ['runner_config.json', 'config.json'])
    manifest, manifest_ref = _receipt(run, ['manifest.json'])
    if config_ref: config['_file_sha256'] = config_ref['sha256']
    if manifest_ref: config['_manifest_sha256'] = manifest_ref['sha256']
    checks = Checks(); source_receipts = []; validate_manifest_sources(manifest, checks)
    sources = config.get('source_sha256', config.get('code_hashes', {}))
    for name, expected in sources.items():
        path = Path(name); path = path if path.is_absolute() else ROOT/path
        exists = path.is_file()
        checks.require(exists and sha(path) == expected, 'current_source_matches_frozen_hash', name)
    if sources: source_receipts.append({'source_sha256': sources})
    contract, contract_ref = _receipt(run, ['contract_freeze_v2.json','contract_freeze.json'])
    if contract:
        path = ROOT/contract['contract_path']
        checks.require(path.is_file() and sha(path) == contract['contract_sha256'], 'contract_freeze_matches', str(path))
        for rawfield in ('contract_sha256',):
            if config.get(rawfield): checks.require(config[rawfield] == contract['contract_sha256'], 'config_contract_hash', rawfield)
    phase_reports = {}; summary = []; paired = []; breakdown = []
    for phase in phases:
        raw = run/phase/'raw.jsonl'
        if not raw.is_file() or raw.stat().st_size == 0:
            phase_reports[phase] = {'pilot_status': 'NOT RUN', 'observed_requests': 0,
                                    'expected_requests': EXPECTED_COUNTS[phase]}; continue
        rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
        result = audit_image_rows(rows, manifest, config)
        expected = expected_requests(manifest, phase)
        complete = len(expected) == EXPECTED_COUNTS[phase] and set(expected) == {r.get('request_id') for r in rows}
        result.update(observed_requests=len(rows), expected_requests=EXPECTED_COUNTS[phase],
                      complete_frozen_workload=complete, raw_sha256=sha(raw), raw_path=str(raw))
        result['pilot_status'] = 'VALID' if complete and result['status'] == 'PASS' else 'INVALID'
        if allow_partial and not complete: result['partial_scope'] = 'provided_complete_cohorts_only_not_final_pilot'
        phase_reports[phase] = result
        checks.require(result['status'] == 'PASS', 'raw_'+phase, result['failure_counts'])
        checks.require(complete or allow_partial, 'complete_'+phase, len(rows))
        if result['status'] == 'PASS':
            summary += summaries(rows, phase)
            breakdown += breakdown_rows(rows)
            if complete: paired += paired_bootstrap(rows, phase)
    gpu_paths = ([config['gpu_gate']['path']] if isinstance(config.get('gpu_gate'),dict) and config['gpu_gate'].get('path') else []) + ['gpu_full_01/gpu_validation.json','gpu_validation.json']
    gpu, gpu_ref = _receipt(run, gpu_paths)
    cpu, cpu_ref = _receipt(run, ['cpu_validation_final.json','cpu_validation.json','core_cpu_tests.json'])
    legacy, legacy_ref = _receipt(run, ['legacy_cpu_tests.json'])
    protection, protection_ref = _receipt(run, ['protection_rehash.json','protection_after.json','protected_artifacts_after.json','protection_check.json'])
    g = gpu.get('gates', {})
    cpu_status = cpu.get('status', 'PASS' if cpu.get('passed') is True else 'NOT RUN')
    legacy_status = legacy.get('status', 'PASS' if legacy.get('passed') is True else 'NOT RUN')
    gates = {'CPU_TEST': cpu_status, 'GPU_CORRECTNESS': gpu.get('GPU_CORRECTNESS','NOT RUN'),
             'PROJECTION_REUSE': g.get('G10','NOT RUN'), 'ALLHEAD_K_READ_REUSE': g.get('G9','NOT RUN'),
             'LEGACY_PROTECTION': legacy_status, 'QWEN_GPU': gpu.get('QWEN_GPU','NOT RUN'),
             'IMPLEMENTATION': 'PASS' if cpu_status == 'PASS' and gpu.get('GPU_CORRECTNESS') == 'PASS' else 'PARTIAL'}
    protection_pass = protection.get('status') == 'PASS' or protection.get('passed') is True
    gpu_pass = gpu.get('GPU_CORRECTNESS') == 'PASS' and all(g.get(f'G{i}') == 'PASS' for i in range(1,13))
    all_pilots = all(phase_reports.get(p,{}).get('pilot_status') == 'VALID' for p in ('smoke','gqa','mt'))
    ready = not checks.failures and cpu_status == 'PASS' and gpu_pass and legacy_status == 'PASS' and protection_pass and all_pilots
    result = checks.result()
    result.update(schema_version='sparsevlm-ssd-kv25-independent-audit-v1', phases=phase_reports, gates=gates,
                  ready=ready, readiness_reason=('all correctness, projection/I/O, protection and full pilot gates passed' if ready else
                  'required correctness/projection/read/protection or complete pilot evidence is absent, failed, or unresolved'),
                  references={'config':config_ref,'manifest':manifest_ref,'contract':contract_ref,'gpu':gpu_ref,
                              'cpu':cpu_ref,'legacy':legacy_ref,'protection':protection_ref},
                  bootstrap={'resamples':10000,'seed':1234,'cluster':'image_id','weighting':'request'},
                  audit_script_sha256=sha(__file__), source_receipts=source_receipts)
    if not any(p.get('observed_requests') for p in phase_reports.values()): result['status'] = 'NOT RUN'
    exclusive_text(output/'independent_audit.json', json.dumps(result,indent=2,allow_nan=False)+'\n')
    exclusive_text(output/'summary.csv', csv_text(summary))
    exclusive_text(output/'paired_comparisons.csv', csv_text(paired))
    diagnostic = controlled_rows(gpu)
    exclusive_text(output/'io_timing_memory_breakdown.csv',csv_text(breakdown))
    exclusive_text(output/'controlled_head_policy.csv',csv_text(diagnostic))
    exclusive_text(output/'setup_costs.csv',csv_text(setup_rows(run)))
    report = render_report(result, summary, paired, run)
    report += '\n## Controlled head-policy diagnostic (outside timing)\n\n| Source policy | Layers | Mean score correlation | Mean Top-k overlap | Mean Jaccard | Probe3 / AllHead chunks |\n|---|---:|---:|---:|---:|---:|\n'
    if not diagnostic: report += '| NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |\n'
    else:
        for policy in sorted({str(x.get('source_hidden_policy','unknown')) for x in diagnostic}):
            ds=[x for x in diagnostic if str(x.get('source_hidden_policy','unknown'))==policy]
            avg=lambda key:_mean([x[key] for x in ds if isinstance(x.get(key),(int,float))])
            report += f'| {policy} | {len(ds)} | {_fmt(avg("pearson_score_correlation"))} | {_fmt(avg("topk_overlap_fraction"))} | {_fmt(avg("topk_jaccard"))} | {_fmt(avg("probe3_chunk_count"))} / {_fmt(avg("allhead_chunk_count"))} |\n'
    report += '\nH2D payload subtotal과 rater visual H2D는 별도 필드다. sysKV/IDs 등 미계측 transfer를 포함한 total H2D로 쓰지 않는다. Actual attention host/CUDA intervals, projection counts, I/O and memory는 `io_timing_memory_breakdown.csv`, setup receipt 비용은 `setup_costs.csv`를 참조한다.\n'
    exclusive_text(output/'REPORT.md', report)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=('all','smoke','gqa','mt'), default='all')
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    phases = ('smoke','gqa','mt') if args.phase == 'all' else (args.phase,)
    result = run_audit(args.run_dir,args.output_dir,phases,args.allow_partial)
    print(json.dumps({'status':result['status'],'ready':result['ready'],
                      'phases':{p:x['pilot_status'] for p,x in result['phases'].items()}}))
    return 1 if result['status'] == 'FAIL' else 0


if __name__ == '__main__': raise SystemExit(main())

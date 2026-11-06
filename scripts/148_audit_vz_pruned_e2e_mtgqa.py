#!/usr/bin/env python3
"""Independent per-image raw audit for the VZ-Pruned-E2E MT-GQA run (Stage 2).

109's rules for ReComp and Ours-KV25 are reused unchanged (loaded, never
modified); the VZ arm adds: pruned T1 geometry (one vision call, one
language-model prefill of length L+q at contiguous positions, zero reads,
k = ceil(N/4) attended content rows), same selection in every T1 of the image
(G7), Compact-rule hits (text start L, Ours-identical read ranges), store
metadata and the per-image gate receipt.  No production inference code is
imported.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def helper(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / file)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


B = helper('_vz_audit_b_main_rules', '109_audit_llava_mt_gqa_B.py')
A = B.A
sha, canonical, read_rows, dump = B.sha, B.canonical, B.read_rows, B.dump
METHODS = ('recompute', 'ours_prefixstore_kv25', 'vz_pruned_e2e')
VZ = 'vz_pruned_e2e'
STORE_KIND = {'ours_prefixstore_kv25': 'image_only', VZ: 'vz'}
SCHEMA = 'llava-mtgqa-vz-e2e-v1'
MODEL = B.MODEL


def dialogues_of(manifest, phase, image_id):
    ds = [d for d in manifest['dialogues'] if d['image_id'] == image_id]
    if phase in ('smoke', 'integration'):
        ids = set(manifest[f'{phase}_dialogue_ids']); ds = [d for d in ds if d['dialog_id'] in ids]
    elif phase == 'correctness':
        ds = ds[:1]
    return ds


def audit_image_rows(rows, manifest, config, phase, image_id, artifact=None):
    c = A.Checks(); ds = dialogues_of(manifest, phase, image_id)
    c.require(bool(ds), 'known_image', image_id)
    expected = {(d['dialog_id'], m, t): d['turns'][t - 1] for d in ds for m in METHODS for t in (1, 2, 3)}
    bykey = {(r['dialog_id'], r['method_id'], r['turn_id']): r for r in rows}
    c.require(set(bykey) == set(expected) and len(rows) == len(expected), 'image_completeness', image_id)
    c.require(len({r['logical_request_id'] for r in rows}) == len(rows), 'unique_logical_ids', image_id)
    c.require(len({r['physical_execution_id'] for r in rows}) == len(rows), 'unique_physical_ids', image_id)
    ordinals = {d['dialog_id']: i for i, d in enumerate(manifest['dialogues'])}
    metas = {}
    if artifact is not None:
        artifact = Path(artifact); builds = []
        for method, kind in STORE_KIND.items():
            p = json.loads((artifact / f'{kind}_persistence.json').read_text()); meta = json.loads((artifact / f'{kind}_meta.json').read_text())
            hashes = json.loads((artifact / f'{kind}_hashes.json').read_text()); files = p['files']; builds.append(p['event_id']); metas[method] = meta
            c.require(hashes == {n: f['sha256'] for n, f in files.items()}, 'complete_store_hash_inventory', kind)
            c.require(p['file_logical_bytes'] == sum(f['size'] for f in files.values()) and p['allocated_bytes'] == sum(f['allocated_bytes'] for f in files.values()), 'file_occupancy_sums', kind)
            c.require(p['service_persistence_ms'] >= p['native_writer_wall_ms'] > 0 and p['os_write_bytes'] >= p['file_logical_bytes'], 'measured_store_completion', kind)
            c.require(p['source']['dialog_id'] == ds[0]['dialog_id'] and p['source']['method_id'] == method, 'first_source_build', kind)
            n = meta['v_token_num'] - len(meta['newline_idx']); k = (n + 3) // 4; width = meta['num_heads'] * meta['head_dim'] * 2
            c.require(meta['storage_policy'] == 'prefix25' and meta['original_content_count'] == n and meta['stored_content_count'] == meta['payload_rows'] == k and meta['padding_rows'] == 0, 'prefix25_geometry_short_EOF', kind)
            c.require(meta['stored_row_to_original'] == meta['order'][:k] and meta['full_importance_permutation'] == meta['order'] and sorted(meta['order']) == list(range(meta['v_token_num'])), 'partial_and_full_mapping', kind)
            for li in range(meta['num_layers']):
                for kv in ('k', 'v'): c.require(files[f'layer_{li:02d}/{kv}.bin']['size'] == k * width, 'exact_file_length', kind)
            if method == VZ:
                L = meta['v_token_start'] + k + len(meta['newline_idx'])
                c.require(meta['ablation'] == 'vz_pruned_e2e' and meta['kv_context'] == 'pruned_t1' and meta['position_policy'] == 'compact', 'vz_store_metadata', kind)
                c.require(meta['text_position_start'] == L and meta['prefill_len'] == L + meta['source_turn1_text_tokens'], 'vz_text_position_start', kind)
                c.require(p['synthetic']['prefix_rows'] == L and p['synthetic']['synthetic_assembly_ms'] > 0, 'vz_synthetic_assembly_inside_persistence', kind)
            else:
                c.require(meta['kv_context'] == 'full_turn1' and meta['position_policy'] == 'original', 'ours_store_metadata', kind)
            for row in rows:
                if row['method_id'] == method and (row['turn_id'] > 1 or method == VZ):
                    c.require(row['result']['selected_original_ids'] == meta['order'][:k], 'meta_to_actual_selection', row['request_id'])
        c.require(len(set(builds)) == 2, 'exactly_two_distinct_physical_build_events', image_id)
        c.require(metas[VZ]['full_importance_permutation'] == metas['ours_prefixstore_kv25']['full_importance_permutation'], 'same_selection_both_stores', image_id)
        c.require(metas[VZ]['prefix_input_ids'] == metas['ours_prefixstore_kv25']['prefix_input_ids'] and metas[VZ]['image_input_sha256'] == metas['ours_prefixstore_kv25']['image_input_sha256'], 'same_prefix_identity_both_stores', image_id)
        for method in STORE_KIND:
            a = json.loads((artifact / f'{method}_activation.json').read_text()); c.require(a['event_kind'] == 'activation' and a['outer_wall_ms'] > 0, 'actual_activation_event', method)
        gates = json.loads((artifact / 'gates.json').read_text())
        c.require(gates['status'] == 'PASS' and gates['G2']['status'] == 'PASS' and gates['G2b']['equal'] and gates['G2e']['sentinel_rows'] == 0, 'image_gate_receipt', image_id)
        c.require(gates['G2']['contract'] == 'vz_pruned_e2e amendment 1' and all(
            name in gates['G2']['gates'] for name in ('structural_bitident_k', 'structural_bitident_v', 'layer0_v', 'vz_layer0_k_realigned_S',
                                                     'vz_layer0_k_oracle_ratio_S', 'vz_neg_layer0_k_unrealigned_S')), 'image_gate_contract_set', image_id)
        c.require(gates['control']['outside_timers'] is True, 'control_outside_timers', image_id)
    ours_sets = []; vz_sets = []
    for key, r in bykey.items():
        rid = r['request_id']; did, m, t = key; result = r['result']; question = expected.get(key)
        if question is None: continue
        c.require(r['schema_version'] == SCHEMA and r['model_id'] == MODEL, 'schema_model', rid)
        expected_id = f"{SCHEMA}:{r['experiment_id']}:{phase}:{MODEL}:{m}:{did}:t{t}"
        c.require(rid == r['logical_request_id'] == expected_id, 'logical_identity', rid)
        c.require(r['phase'] == phase and r['image_id'] == image_id, 'phase_image', rid)
        c.require(r['question_id'] == str(question['question_id']) and r['question'] == question['question'] and r['gold'] == question['answers'], 'frozen_question_gold', rid)
        c.require(r['score'] == A.independent_score(r['prediction'], r['gold'], phase='mt'), 'strict_em_recalculation', rid)
        c.require(r['prediction'] == result['answer'], 'raw_decoded_prediction', rid)
        c.require(r['config_sha256'] == config['_file_sha256'] and r['manifest_sha256'] == config['_manifest_sha256'], 'config_manifest_hash', rid)
        c.require(r['code_sha256'] == canonical(config['source_sha256']), 'code_hash', rid)
        ordinal = ordinals[did]; i = ordinal % 3; order = list(METHODS[i:] + METHODS[:i])
        c.require(r['global_dialog_ordinal'] == ordinal and r['method_order'] == order and r['method_order_position'] == order.index(m), 'deterministic_rotation', rid)
        c.require(r['image_sha256'] == manifest['image_identity'][image_id]['sha256'], 'frozen_image_hash', rid)
        history = []; entries = []
        for prior_t in range(1, t):
            prev = bykey.get((did, m, prior_t)); q = expected.get((did, m, prior_t))
            if prev is None: c.require(False, 'missing_history_source', rid); continue
            history += [f"Q{prior_t}: {q['question'].strip()}", f"A{prior_t}: {prev['prediction']}"]
            entries.append({'turn_id': prior_t, 'question_id': str(q['question_id']), 'question': q['question'].strip(), 'answer': prev['prediction'],
                            'answer_source': 'method_local_generated', 'source_model': MODEL, 'source_method_id': m, 'source_dialog_id': did,
                            'source_logical_request_id': prev['logical_request_id'], 'source_physical_execution_id': prev['physical_execution_id']})
        hist = '\n'.join(history); body = ([hist, ''] if hist else [])
        body += [f"Current question Q{t}: {question['question'].strip()}", 'Answer the current question with a single word or short phrase. ASSISTANT:']
        prompt = 'USER: <image>\n' + '\n'.join(body)
        c.require(r['history'] == hist and r['history_entries'] == entries, 'exact_method_local_history_lineage', rid)
        c.require(r['prompt'] == prompt, 'causal_prompt_reconstruction', rid)
        c.require(r['prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest(), 'prompt_hash', rid)
        c.require(r['history_sha256'] == hashlib.sha256(hist.encode()).hexdigest(), 'history_hash', rid)
        c.require(r['suffix_ids_sha256'] == canonical(r['suffix_input_ids']), 'suffix_hash', rid)
        c.require(0 < r['ttft_ms'] <= r['request_e2e_ms'], 'valid_ttft_e2e', rid)
        c.require(math.isclose(r['ttft_ms'], 1000 * (r['first_token_at_s'] - r['request_started_at_s']), abs_tol=1e-7), 'true_ttft_recalculation', rid)
        c.require(math.isclose(r['request_e2e_ms'], 1000 * (r['request_finished_at_s'] - r['request_started_at_s']), abs_tol=1e-5), 'request_e2e_recalculation', rid)
        tokens = result['generated_token_ids']
        c.require(1 <= len(tokens) <= 16 and tokens[0] == result['first_token_id'], 'actual_generated_tokens', rid)
        c.require(result['vision_forward_count'] == int(t == 1 or m == 'recompute'), 'vision_call_contract', rid)
        trace = result.get('os_pread_trace', [])
        c.require(r['ssd_read_bytes'] == sum(e['returned'] for e in trace), 'actual_returned_bytes', rid)
        c.require(r['ssd_preads'] == len(trace), 'actual_pread_count', rid)
        c.require(all(e['returned'] == e['requested'] for e in trace), 'successful_full_requested_reads', rid)
        c.require(not any('probe_k.bin' in e['path'] for e in trace), 'no_probe_read', rid)
        g = r['geometry']; n = g['v_token_num'] - len(set(g['newline_idx']) | set(g['padding_idx']))
        c.require(n == r['N_content'] and n > 0, 'actual_content_geometry', rid)
        k = (n + 3) // 4; L = g['v_token_start'] + k + len(g['newline_idx'])
        m2 = r.get('first_token_top2')
        # exact top-2 ties (margin 0) may resolve to either index between the BF16 argmax and the FP32 top-k
        c.require(isinstance(m2, dict) and (m2.get('argmax') == result['first_token_id'] or m2.get('margin') == 0.0), 'first_token_margin_recorded', rid)
        if m == VZ and t == 1:
            c.require(r['k'] == k and r['budget_unit'] == 'visual_kv' and r['ssd_read_bytes'] == 0 and r['source_store'] is None and not r['cache_hit'], 'vz_pruned_T1_budget', rid)
            c.require(result['pruned_input_len'] == L + result['text_tokens'] == result['position_check']['prefill_len'] and result['position_check']['verified'] is True and result['position_check']['contiguous'] is True, 'vz_T1_single_contiguous_prefill', rid)
            c.require(result['language_model_prefill_calls'] == 1 and result['online_query_score_calls'] == 0 and result['vision_forward_count'] == 1, 'vz_T1_real_path_counts', rid)
            c.require(result['text_position_start'] == L and result['k_target'] == k and result['N_content'] == n, 'vz_T1_plan_geometry', rid)
            c.require(len(result['selected_original_ids']) == k and len(set(result['selected_original_ids'])) == k, 'vz_T1_selection_size', rid)
            c.require(result['ttft_path'].startswith('real pruned serving path'), 'vz_T1_real_path_label', rid)
            vz_sets.append((result['selection_sha256'], tuple(result['selected_original_ids'])))
        elif t == 1 or m == 'recompute':
            c.require(r['k'] == n and r['ssd_read_bytes'] == 0 and r['budget_unit'] == 'full_context', 'full_pixel_request', rid)
            c.require(r['source_store'] is None and not r['cache_hit'], 'pixel_not_cache_hit', rid)
        else:
            c.require(result.get('adapter_restored') is True, 'adapter_restored', rid)
            c.require(bool(result.get('conditioning_events')) and all(x['status'] == 'SUCCESS' for x in result['conditioning_events']), 'conditioning_success', rid)
            conditioned = {x['path'] for x in result['conditioning_events']}
            c.require(all(e['path'] in conditioned for e in trace), 'every_read_file_conditioned', rid)
            src = r['source_store']; c.require(src is not None and r['cache_hit'], 'store_provenance', rid)
            c.require(src['source']['dialog_id'] == ds[0]['dialog_id'] and src['source']['method_id'] == m, 'first_source_T1_only', rid)
            source_row = bykey.get((ds[0]['dialog_id'], m, 1), {})
            c.require(src['source']['physical_execution_id'] == source_row.get('physical_execution_id'), 'actual_source_physical_id', rid)
            B.audit_existing_hit(dict(r, os_pread_trace=trace), c)
            c.require(r['k'] == k and r['budget_unit'] == 'visual_kv', 'native_KV25_hit', rid)
            pc = result.get('position_check', {})
            c.require(pc.get('verified') is True and pc.get('text_position_start') == (L if m == VZ else g['prefix_len']), 'hit_text_position_start', rid)
            (vz_sets if m == VZ else ours_sets).append((None, tuple(result['selected_original_ids'])))
        calls = result.get('projection_calls') or {}
        for li in range(g['num_layers']):
            observed = calls.get(str(li), {})
            c.require(observed.get('prefill') == {'q': 1, 'k': 1, 'v': 1}, 'one_prefill_projection_each_layer', rid)
            expected_decode = {'q': len(tokens) - 1, 'k': len(tokens) - 1, 'v': len(tokens) - 1}
            c.require(observed.get('decode', {'q': 0, 'k': 0, 'v': 0}) == expected_decode, 'decode_projection_counts', rid)
    c.require(not ours_sets or len({s[1] for s in ours_sets}) == 1, 'ours_same_image_selection_invariant', image_id)
    c.require(not vz_sets or len({s[1] for s in vz_sets}) == 1, 'vz_same_image_selection_invariant_G7', image_id)
    c.require(not (vz_sets and ours_sets) or vz_sets[0][1] == ours_sets[0][1], 'vz_ours_same_selection', image_id)
    return dict(c.result(), image_id=image_id, phase=phase, requests=len(rows), expected_requests=len(expected), dialogs=len(ds))


def run_audit(run, output):
    """Re-audit every committed image of every phase; return receipts and compact main rows."""
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / 'config.json').read_text()); config.update(_file_sha256=sha(run / 'config.json'), _manifest_sha256=sha(run / 'manifest.json'))
    manifest = json.loads((run / 'manifest.json').read_text())
    checks = A.Checks(); receipts = []; phase_counts = {}; attempt_counts = Counter(); logical = set(); physical = set(); main_rows = []
    for phase in ('correctness', 'smoke', 'integration', 'main'):
        count = 0
        for cp in sorted((run / phase / 'images').glob('*/COMMITTED.json')):
            commit = json.loads(cp.read_text()); folder = cp.parent / commit['attempt_id']; raw = folder / 'raw.jsonl'
            checks.require(sha(raw) == commit['raw_sha256'], 'committed_raw_hash', str(cp))
            for n, digest in commit['evidence_sha256'].items(): checks.require(sha(folder / n) == digest, 'committed_evidence_unchanged', str(folder / n))
            rows = list(read_rows(raw)); receipt = audit_image_rows(rows, manifest, config, phase, cp.parent.name, folder); receipts.append(receipt)
            checks.require(receipt['status'] == 'PASS', 'independent_image_audit', receipt); count += len(rows)
            cleanup = json.loads((folder / 'cleanup_receipt.json').read_text())
            checks.require(cleanup['status'] == ('RETAINED' if phase == 'correctness' else 'PASS'), 'cleanup_receipt', str(folder))
            if phase == 'main':
                for r in rows:
                    checks.require(r['logical_request_id'] not in logical and r['physical_execution_id'] not in physical, 'global_unique_ids', r['request_id'])
                    logical.add(r['logical_request_id']); physical.add(r['physical_execution_id']); main_rows.append(r)
        phase_counts[phase] = count
        for attempt in (run / phase / 'images').glob('*/attempt_*'):
            attempt_counts[phase + '_image_attempts'] += 1
            if (attempt / 'failure.json').exists(): attempt_counts[phase + '_failed_attempts'] += 1
    expected = manifest['expected_requests']; complete = phase_counts['main'] == expected
    if complete:
        checks.require(len({r['image_id'] for r in main_rows}) == 398 and len({r['dialog_id'] for r in main_rows}) == 4061, 'main_population')
        for m in METHODS:
            rs = [r for r in main_rows if r['method_id'] == m]
            checks.require(len(rs) == 12183 and sum(r['turn_id'] > 1 for r in rs) == 8122, 'per_method_counts', m)
    for name, digest in json.loads((run / 'gpu_freeze.json').read_text())['files'].items():
        p = run / name if name in ('manifest.json', 'config.json') else ROOT / name
        checks.require(sha(p) == digest, 'final_source_freeze', name)
    result = checks.result()
    result.update(status_main='VALID' if complete and result['status'] == 'PASS' else 'INVALID' if complete else 'PARTIAL' if phase_counts['main'] else 'NOT RUN',
                  main_requests=phase_counts['main'], expected_main_requests=expected, phase_counts=phase_counts, attempt_counts=dict(attempt_counts),
                  image_receipts=receipts)
    dump(output / 'independent_audit.json', result)
    return result, main_rows, manifest, config


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); r, _, _, _ = run_audit(a.run.resolve(), a.output.resolve())
    print(json.dumps({k: r[k] for k in ('status', 'status_main', 'main_requests', 'phase_counts')}), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Read-only, stdlib-only audit of a frozen LLaVA contextual KV25 run.

Usage: python /tmp/audit_contextual_kv25.py RUN_DIR [--phase pilot|smoke]
The default pilot audit requires 40 images x 6 turns x 9 methods = 2160 rows.
Nothing is imported from the experiment implementation and no files are written.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path('/home/dblab/hj/mllm_v2')
INDEX_SHA = '514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a'
METHODS = (
    'recompute', 'fullload', 'd25_c0', 'd22_5_c2_5', 'd20_c5',
    'd17_5_c7_5', 'd15_c10', 'd20_random5', 'd20_uniform5',
)
SELECTIVE = METHODS[2:]
ALPHAS = {
    'd25_c0': 0.0, 'd22_5_c2_5': 0.1, 'd20_c5': 0.2,
    'd17_5_c7_5': 0.3, 'd15_c10': 0.4,
    'd20_random5': 0.2, 'd20_uniform5': 0.2,
}
VARIANTS = {
    'd25_c0': 'dominant', 'd22_5_c2_5': 'contextual',
    'd20_c5': 'contextual', 'd17_5_c7_5': 'contextual',
    'd15_c10': 'contextual', 'd20_random5': 'random',
    'd20_uniform5': 'uniform',
}
SCHEMA = 'llava-contextual-representative-kv25-v1'
LAYOUT = 'visionzip_contextual_original_v1'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def canonical_hash(x) -> str:
    b = json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                   allow_nan=False).encode('utf-8')
    return hashlib.sha256(b).hexdigest()


def exact_score(pred: str, answer: str) -> float:
    def norm(s):
        words = re.sub(r'[^\w\s]', ' ', str(s).lower()).split()
        return ' '.join(w for w in words if w not in {'a', 'an', 'the'})
    p, g = norm(pred), norm(answer)
    return float(p == g or (bool(g) and p.split()[:len(g.split())] == g.split()))


class Audit:
    def __init__(self):
        self.failures = collections.Counter()
        self.examples: dict[str, list[str]] = collections.defaultdict(list)
        self.check_count = 0

    def check(self, ok, code, detail=''):
        self.check_count += 1
        if not ok:
            self.failures[code] += 1
            if len(self.examples[code]) < 4:
                self.examples[code].append(str(detail)[:300])

    def read_json(self, path: Path, code: str):
        self.check(path.is_file(), code, str(path))
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            self.check(False, code, f'{path}: {exc}')
            return None


def audit(run_dir: Path, phase: str) -> dict:
    a = Audit()
    images = 4 if phase == 'smoke' else 40
    turns = 3 if phase == 'smoke' else 6
    index_path = ROOT / 'data/index.json'
    a.check(sha(index_path) == INDEX_SHA, 'frozen_index_sha', str(index_path))
    index = json.loads(index_path.read_text(encoding='utf-8'))[:images]
    a.check(len(index) == images, 'index_image_count', len(index))
    expected = {}
    image_paths = {}
    for image_index, entry in enumerate(index):
        image_id = str(entry['image_id'])
        image_paths[image_id] = ROOT / entry['image_path']
        questions = entry['questions'][4:4+turns]
        a.check(len(questions) == turns, 'index_question_count', image_id)
        for turn_id, question in enumerate(questions, 1):
            for method in METHODS:
                expected[(image_id, str(question['question_id']), method)] = (
                    image_index, turn_id, question)
    a.check(len(expected) == images*turns*len(METHODS),
            'expected_workload_size', len(expected))
    raw_path = run_dir / f'{phase}_raw.jsonl'
    a.check(raw_path.is_file(), 'raw_file', str(raw_path))
    rows = []
    if raw_path.is_file():
        with raw_path.open(encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                try:
                    rows.append(json.loads(line))
                except ValueError as exc:
                    a.check(False, 'raw_json_parse', f'line {line_no}: {exc}')
    keys = [(str(r.get('image_id')), str(r.get('question_id')),
             str(r.get('method_key'))) for r in rows]
    counts = collections.Counter(keys)
    missing = sorted(set(expected)-set(counts))
    extra = sorted(set(counts)-set(expected))
    duplicate = sorted(k for k, n in counts.items() if n != 1)
    a.check(len(rows) == images*turns*len(METHODS),
            'raw_row_count', len(rows))
    a.check(not missing, 'missing_requests', missing[:4])
    a.check(not extra, 'extra_requests', extra[:4])
    a.check(not duplicate, 'duplicate_requests', duplicate[:4])
    manifest = a.read_json(run_dir / 'workload_manifest.json', 'workload_manifest')
    if manifest:
        manifest_entries = manifest.get('images', [])[:images]
        a.check(len(manifest_entries) >= images, 'manifest_images', len(manifest_entries))
        for idx, entry in enumerate(manifest_entries):
            source = index[idx]
            a.check(entry.get('image_id') == source['image_id']
                    and entry.get('image_path') == source['image_path']
                    and entry.get('question_ids')[:turns] ==
                    [str(q['question_id']) for q in source['questions'][4:4+turns]],
                    'manifest_index_alignment', idx)
    runner = a.read_json(run_dir / f'{phase}_runner_manifest.json',
                         'runner_manifest')
    if runner:
        fw = runner.get('frozen_workload', {})
        a.check(fw.get('index_sha256') == INDEX_SHA,
                'runner_index_sha', fw.get('index_sha256'))
        a.check(fw.get('selected_workload_sha256') ==
                '97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66',
                'runner_workload_sha', fw.get('selected_workload_sha256'))
        a.check(runner.get('expected_requests') == images*turns*len(METHODS),
                'runner_expected_count', runner.get('expected_requests'))
        a.check(set(runner.get('methods', {})) == set(METHODS),
                'runner_nine_methods')
    by_image = collections.defaultdict(list)
    by_arm = collections.defaultdict(list)
    by_turn = collections.defaultdict(list)
    for row in rows:
        image_id, qid, method = (str(row.get(k)) for k in
                                 ('image_id', 'question_id', 'method_key'))
        key = (image_id, qid, method)
        if key not in expected:
            continue
        image_index, turn_id, question = expected[key]
        tag = f'{image_id}:T{turn_id}:{method}'
        by_image[image_id].append(row)
        by_arm[(image_id, method)].append(row)
        by_turn[(image_id, turn_id)].append(row)
        gold = [str(question['answer'])] if 'answers' not in question else question['answers']
        a.check(row.get('schema_version') == SCHEMA and
                row.get('dataset') == 'gqa' and row.get('phase') == phase and
                row.get('status') == 'ok', 'row_identity_status', tag)
        a.check(row.get('request_id') == f'{phase}:{image_id}:{qid}:{method}'
                and row.get('image_index') == image_index and
                row.get('turn_id') == turn_id, 'row_request_mapping', tag)
        a.check(row.get('question') == question['question']
                and row.get('gold') == gold, 'row_question_gold', tag)
        prediction = row.get('prediction')
        a.check(prediction == row.get('answer'), 'prediction_answer_match', tag)
        a.check(row.get('correct') == exact_score(prediction, gold[0]),
                'independent_exact_match', tag)
        expected_path = ('normal_pixel_turn1' if turn_id == 1 else
                         'normal_pixel_recompute' if method == 'recompute' else
                         'ssd_visual_kv')
        a.check(row.get('request_path') == expected_path and
                row.get('cache_hit') is (turn_id > 1 and method != 'recompute'),
                'request_path_hit', tag)
        a.check(isinstance(row.get('end_to_end_ttft_ms'), (int, float)) and
                row['end_to_end_ttft_ms'] > 0, 'positive_ttft', tag)
        a.check(row.get('method_id') == method and
                row.get('method_key') in METHODS, 'method_id', tag)
        order = row.get('method_order')
        a.check(isinstance(order, list) and set(order) == set(METHODS) and
                len(order) == len(METHODS) and
                isinstance(row.get('method_order_position'), int) and
                order[row['method_order_position']] == method,
                'method_order', tag)
        if turn_id == 1 or method == 'recompute':
            a.check(row.get('vision_forward_count') == 1 and
                    row.get('ssd_read_bytes') == 0 and
                    row.get('ssd_preads') == 0,
                    'normal_pixel_io', tag)
        else:
            a.check(row.get('vision_forward_count') == 0 and
                    row.get('probe_read_bytes') == 0 and
                    row.get('probe_preads') == 0 and
                    row.get('query_score_calls', 0) == 0,
                    'cache_hit_no_vision_probe', tag)
            a.check(row.get('ssd_read_bytes') ==
                    sum(row.get(x, 0) for x in ('normal_kv_read_bytes',
                                               'separator_read_bytes',
                                               'probe_read_bytes')) and
                    row.get('ssd_preads') ==
                    sum(row.get(x, 0) for x in ('normal_kv_preads',
                                               'separator_preads',
                                               'probe_preads')),
                    'ssd_io_accounting', tag)
        if method in SELECTIVE:
            a.check(row.get('selection_variant') == VARIANTS[method] and
                    row.get('alpha') == ALPHAS[method] and
                    row.get('budget_ratio') == 0.25 and
                    row.get('budget_unit') == 'visual_kv',
                    'selective_method_metadata', tag)
            expected_artifact = (f'image_artifacts/{phase}/{image_id}/'
                                 f'{method}_selection.json')
            a.check(row.get('selection_artifact') == expected_artifact,
                    'selection_artifact_pointer', tag)
        if turn_id > 1 and method in SELECTIVE:
            n = row.get('N_content')
            k = row.get('k_target')
            a.check(isinstance(n, int) and n > 0 and
                    k == (n + 3)//4 and
                    row.get('attended_content_kv_count') == k and
                    math.isclose(row.get('logical_content_retention', -1), k/n,
                                 rel_tol=0, abs_tol=1e-12),
                    'exact_25pct_budget', tag)
    for image_id, entry in ((str(e['image_id']), e) for e in index):
        image_rows = by_image[image_id]
        path = image_paths[image_id]
        image_hash = sha(path) if path.is_file() else None
        a.check(image_hash is not None, 'image_file', image_id)
        if image_rows:
            a.check({r.get('image_sha256') for r in image_rows} == {image_hash},
                    'image_sha', image_id)
            orders = {tuple(r.get('method_order', [])) for r in image_rows}
            a.check(len(orders) == 1, 'image_method_order_constant', image_id)
            t1 = [r for r in image_rows if r.get('turn_id') == 1]
            a.check(len(t1) == 9 and
                    len({r.get('prediction') for r in t1}) == 1 and
                    len({r.get('first_token_id') for r in t1}) == 1 and
                    len({r.get('prompt_sha256') for r in t1}) == 1,
                    'turn1_identical', image_id)
        for turn_id in range(2, turns + 1):
            selective = [r for r in by_turn[(image_id, turn_id)]
                         if r.get('method_key') in SELECTIVE]
            a.check(len(selective) == 7, 'selective_turn_coverage',
                    f'{image_id}:T{turn_id}')
            if len(selective) == 7:
                for field in ('N_content', 'N_structural', 'k_target',
                              'normal_kv_read_bytes', 'ssd_read_bytes',
                              'separator_read_bytes', 'probe_read_bytes',
                              'normal_kv_preads', 'ssd_preads',
                              'separator_preads', 'probe_preads'):
                    a.check(len({r.get(field) for r in selective}) == 1,
                            'selective_equal_' + field,
                            f'{image_id}:T{turn_id}')
        artifact_dir = run_dir / 'image_artifacts' / phase / image_id
        receipt = a.read_json(artifact_dir / 'image_receipt.json',
                              'image_receipt')
        if receipt:
            a.check(receipt.get('phase') == phase and
                    receipt.get('image_id') == image_id and
                    receipt.get('image_sha256') == image_hash and
                    receipt.get('validation') == 'PASS' and
                    receipt.get('request_count') == turns*9 and
                    receipt.get('canonical_captured_prefix_bitwise_equal') is True
                    and receipt.get('turn1_output_identical') is True,
                    'image_receipt_validation', image_id)
            a.check(receipt.get('request_ids_sha256') ==
                    canonical_hash([r.get('request_id') for r in image_rows]),
                    'receipt_request_ids_sha', image_id)
            a.check(set(receipt.get('selection_artifacts', {})) == set(SELECTIVE)
                    and set(receipt.get('persistence_by_method', {})) ==
                    set(METHODS[1:]),
                    'receipt_method_coverage', image_id)
            a.check(receipt.get('N_content') ==
                    next((r.get('N_content') for r in image_rows
                          if r.get('method_key') == 'd25_c0'), None) and
                    receipt.get('k_target') ==
                    (receipt.get('N_content', 0) + 3)//4,
                    'receipt_budget', image_id)
        cleanup = a.read_json(artifact_dir / 'cleanup_receipt.json',
                              'cleanup_receipt')
        if cleanup:
            a.check(cleanup.get('phase') == phase and
                    cleanup.get('image_id') == image_id and
                    set(cleanup.get('deleted_store_methods', [])) ==
                    set(METHODS[1:]) and
                    cleanup.get('raw_rows_fsynced_before_cleanup') is True and
                    cleanup.get('selection_artifacts_fsynced_before_cleanup') is True and
                    cleanup.get('image_receipt_fsynced_before_cleanup') is True,
                    'cleanup_receipt_integrity', image_id)
        for method in SELECTIVE:
            artifact = a.read_json(artifact_dir / f'{method}_selection.json',
                                   'selection_artifact')
            if not artifact:
                continue
            n, k = artifact.get('N_content'), artifact.get('k')
            tag = f'{image_id}:{method}'
            a.check(artifact.get('phase') == phase and
                    artifact.get('image_id') == image_id and
                    artifact.get('method_key') == method and
                    artifact.get('layout_policy') == LAYOUT and
                    artifact.get('selection_variant') == VARIANTS[method] and
                    artifact.get('alpha') == ALPHAS[method],
                    'selection_identity', tag)
            a.check(isinstance(n, int) and n > 0 and k == (n+3)//4 and
                    artifact.get('k_context') == math.floor(ALPHAS[method]*k) and
                    artifact.get('k_dominant') == k - artifact.get('k_context', -1),
                    'selection_budget_split', tag)
            ids = artifact.get('selected_original_ids', [])
            order = artifact.get('stored_to_original', [])
            inverse = artifact.get('original_to_stored', [])
            structural = artifact.get('structural_original_ids', [])
            if isinstance(n, int) and isinstance(k, int):
                a.check(len(ids) == k and len(set(ids)) == k and
                        not (set(ids) & set(structural)) and
                        ids == order[:k], 'selection_original_ids', tag)
                a.check(len(order) == n+len(structural) and
                        sorted(order) == list(range(len(order))) and
                        len(inverse) == len(order) and
                        all(inverse[original] == stored
                            for stored, original in enumerate(order)),
                        'selection_permutation_inverse', tag)
            layout = artifact.get('layout_artifact', {})
            a.check(layout.get('calibration_questions') == 0 and
                    layout.get('layout_uses_dataset_question') is False and
                    layout.get('llm_used_for_layout_scoring') is False and
                    layout.get('separate_prefix_forward') is False and
                    layout.get('separate_vision_forward') is False,
                    'layout_image_only', tag)
            if receipt:
                a.check(receipt.get('selection_artifacts', {}).get(method) ==
                        str((artifact_dir / f'{method}_selection.json').relative_to(run_dir)),
                        'receipt_artifact_pointer', tag)
                a.check(receipt.get('N_content') == n and
                        receipt.get('k_target') == k,
                        'receipt_selection_budget', tag)
            hits = [r for r in by_arm[(image_id, method)]
                    if r.get('turn_id', 0) > 1]
            a.check(len(hits) == turns-1 and
                    all(r.get('selected_original_ids') == ids for r in hits),
                    'selection_question_invariance', tag)
    completed = run_dir / f'{phase}_COMPLETED'
    a.check(completed.is_file(), 'phase_completed_marker', str(completed))
    return {
        'passed': not a.failures,
        'phase': phase,
        'run_dir': str(run_dir.resolve()),
        'expected_images': images,
        'expected_requests': images*turns*len(METHODS),
        'observed_requests': len(rows),
        'independent_scores_checked': len(rows),
        'checks_evaluated': a.check_count,
        'missing_count': len(missing),
        'extra_count': len(extra),
        'duplicate_count': len(duplicate),
        'raw_sha256': sha(raw_path) if raw_path.is_file() else None,
        'failure_counts': dict(a.failures),
        'failure_examples': dict(a.examples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--phase', choices=('pilot', 'smoke'), default='pilot')
    args = parser.parse_args()
    result = audit(args.run_dir, args.phase)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())

"""Dataset/scoring/lifecycle adapter over the packed + layer-budget request path.

``mmimpress.mtvqa25k.adapter.bind('llava')`` overlays the MT-VQA identities on
``scripts/108_eval_llava_mt_gqa_B.py`` (one file per layer).  This module does
the same overlay on ``scripts/164_eval_layer_budget_pyrbeta_hold.py`` (one
packed file per image, per-layer row budget).  Nothing of the request path is
reimplemented: prompt factory, Turn-1 full-pixel request, capture, VisionZip
order, packed writer, packed reader/selector, decode and the structural
per-image audit are the 164 code.  Overlaid, exactly as for 108:

* SCHEMA / request_id bound to the new experiment id;
* ``make_row`` -> ``enrich`` (official VQA score, annotation, native decode
  fields, ``generated_token_ids``);
* audit: the 164 structural audit runs on a copy whose ``score`` is the legacy
  strict EM it expects (GLM precedent in ``glm_official_auditor``); the adopted
  rows are checked independently against the frozen official annotations
  (``independent_score`` and ``score_from_strings``);
* store meta ``dataset`` = MT-VQA-v2-reconstructed-25k;
* raw journals archived as lossless gzip before the atomic commit;
* fail-closed cleanup bound to the new experiment root.

Existing sources stay byte-for-byte intact.
"""
from pathlib import Path
import os

from mmimpress.mtvqa25k.common import ROOT, read, dump, sha
from mmimpress.mtvqa25k.adapter import (load, enrich, read_rows, score_from_strings,
                                        archive_before_commit, remove_archived_duplicates)
from mmimpress.mtvqa25k.scorer import independent_score

DATASET = 'MT-VQA-v2-reconstructed-25k'
SCHEMA = 'mtvqa25k-llava-packed-layerbudget-v1'
RUN_PREFIX = 'llava_next_mtvqa_pyr_'
BASE_SCRIPT = '164_eval_layer_budget_pyrbeta_hold.py'
PK = 'ours_prefixstore_kv25_'
PK_SUFFIX = '_packed'
MAIN_METHOD = PK + 'PYR-beta-raw' + PK_SUFFIX
GATE_METHOD = PK + 'U' + PK_SUFFIX
MODEL = 'llava'


def vector_of(method):
    if not (method.startswith(PK) and method.endswith(PK_SUFFIX)):
        raise ValueError('not a packed layer-budget arm: ' + str(method))
    return method[len(PK):-len(PK_SUFFIX)]


def check_methods(methods):
    """Arms are identified by name, never by position; only packed arms are served here."""
    methods = list(methods)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError('empty or duplicated arm list')
    for m in methods:
        vector_of(m)
    return methods


def budget_entry(doc, path):
    """The config['layer_budgets'][vec] record the 164 writer/audit read (document clip range included)."""
    return {'path': str(path), 'layer_budget_sha256': doc['layer_budget_sha256'], 'layer_keep_ratios': doc['layer_keep_ratios'],
            'r_min': doc['r_min'], 'r_max': doc['r_max'], 'alloc_id': doc['alloc_id'], 'policy': doc['policy'], 'schema': doc['schema'],
            'num_layers': doc['num_layers'], 'total_ratio': doc['total_ratio'], 'source': doc['source']}


def cleanup_owned(scratch, allowlist, audit, commit, experiment):
    """Same fail-closed checks as mmimpress.mtvqa25k.adapter.cleanup_owned, bound to this new experiment."""
    scratch = Path(scratch); experiment = Path(experiment).resolve()
    if audit.get('status') != 'PASS' or commit.get('status') != 'COMMITTED':
        raise ValueError('independent audit and atomic commit required')
    if experiment.parent != ROOT / 'runs' or not experiment.name.startswith(RUN_PREFIX):
        raise ValueError('not a new experiment root')
    if scratch.is_symlink() or not scratch.is_dir() or not scratch.resolve().is_relative_to(experiment):
        raise ValueError('outside owned experiment')
    relative = scratch.relative_to(experiment)
    if len(relative.parts) != 6 or relative.parts[0] != MODEL or relative.parts[2] != 'scratch' or not relative.parts[-1].startswith('attempt_'):
        raise ValueError('not an exact model/phase/scratch/phase/image/attempt path')
    nodes = list(scratch.rglob('*'))
    if any(p.is_symlink() for p in nodes):
        raise ValueError('linked payload/directory')
    files = {str(p.relative_to(scratch)): p for p in nodes if p.is_file()}
    if set(files) != set(allowlist):
        raise ValueError('cleanup allowlist mismatch')
    for name, p in files.items():
        if p.stat().st_nlink != 1 or not p.resolve().is_relative_to(scratch.resolve()) or sha(p) != allowlist[name]:
            raise ValueError('linked or changed payload')
    for p in files.values():
        p.unlink()
    for p in sorted((p for p in nodes if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        p.rmdir()
    scratch.rmdir()
    fd = os.open(scratch.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return {'removed_files': len(files), 'removed_bytes_logical': None}


def official_audit(old_audit, legacy_em):
    """164 structural audit on an EM-scored copy + independent official VQA checks on the adopted rows."""
    def audit(rows, manifest, config, phase, image_id, artifact=None):
        legacy = [dict(r, score=legacy_em(r['prediction'], r['gold'][0])) for r in rows]
        report = old_audit(legacy, manifest, config, phase, image_id, artifact)
        failures = []
        byqid = {str(t['question_id']): t for d in manifest['dialogues'] for t in d['turns']}
        for r in rows:
            t = byqid.get(str(r['question_id']))
            if t is None:
                failures.append('unknown_question'); continue
            if r.get('annotation') != t['annotation']:
                failures.append('official_annotation')
            if r.get('scorer_answer') != r['prediction']:
                failures.append('native_scorer_answer')
            expected = independent_score(r['prediction'], t['annotation'])['vqa_score']
            if abs(expected - r['score']) > 1e-10:
                failures.append('official_score')
            if abs(score_from_strings(r['prediction'], [a['answer'] for a in t['annotation']['answers']]) - r['score']) > 1e-10:
                failures.append('official_score_from_strings')
            if r.get('gold') != [a['answer'] for a in t['annotation']['answers']]:
                failures.append('gold_strings')
            if r.get('generated_token_ids') != r['result']['generated_token_ids']:
                failures.append('generated_IDs')
            if r.get('dataset') != DATASET:
                failures.append('dataset_identity')
        report['official_annotation_score_audit'] = {'status': 'FAIL' if failures else 'PASS', 'failures': failures}
        report['legacy_em_note'] = 'structural audit ran on a copy scored by legacy strict EM; adopted score is the official VQA score checked here'
        if failures:
            report['status'] = 'FAIL'
        return report
    return audit


def bind_packed(experiment_id, module_name='_mtvqa25k_packed_llava'):
    """Load a private instance of the 164 module and overlay the MT-VQA identities on it."""
    M = load(module_name, BASE_SCRIPT)
    M.SCHEMA = SCHEMA
    M._experiment_id = str(experiment_id)
    M.request_id = lambda run_id, phase, method, did, t: f'{M.SCHEMA}:{M._experiment_id}:{phase}:{M.MODEL_ID}:{method}:{did}:t{t}'
    M.load_jsonl = read_rows
    legacy_em = M.strict_score
    M.audit_image_rows = official_audit(M.audit_image_rows, legacy_em)

    old_make = M.make_row

    def make(runner, result, diag, meta, **kwargs):
        row = old_make(runner, result, diag, meta, **kwargs)
        row['experiment_id'] = M._experiment_id
        row = enrich(row, kwargs['dialog']['turns'][kwargs['turn'] - 1], runner.processor.tokenizer, MODEL)
        observer = getattr(M, '_row_observer', None)
        if observer is not None:
            observer(row)
        return row
    M.make_row = make

    # Same writer call and options as 164._native_persist; only the dataset identity of the store meta differs.
    def native_persist(runner, result, diag, path, image_id, kind, source, docs):
        from mmimpress.packed_store import persist_captured_packed_prefix
        if not kind.startswith('packed_'):
            raise ValueError('only packed stores are built by this adapter: ' + str(kind))
        enc = diag['enc_cpu']
        args = (runner, result['captured_past_key_values'], enc['input_ids'], enc['image_sizes'][0])
        common = dict(image_id=image_id, model_id=M.MODEL_ID, chunk_size=64, image_input_sha256=diag['image_input_sha256'],
                      extra_metadata={'dataset': DATASET, 'source_dialog_id': source['dialog_id'], 'source_turn_id': 1,
                                      'source_method_id': source['method_id'], 'source_execution_id': source['physical_execution_id'],
                                      'source_logical_request_id': source['logical_request_id'], 'run_schema': M.SCHEMA})
        vec = kind[len('packed_'):]
        return persist_captured_packed_prefix(*args, diag['vision'].result_cpu(), path, layer_budget=docs[vec],
                                              capture_stats=diag['vision'], full_integrity_hash=False, **common)
    M._native_persist = native_persist

    M.cleanup_committed = lambda scratch, allowlist, audit, commit, run: cleanup_owned(scratch, allowlist, audit, commit, ROOT / 'runs' / M._experiment_id)

    original_atomic = M.atomic_json

    def atomic(path, value, *args, **kwargs):
        path = Path(path)
        if path.name == 'COMMITTED.json':
            return archive_before_commit(path.parent, value, original_atomic)
        return original_atomic(path, value, *args, **kwargs)
    M.atomic_json = atomic
    return M

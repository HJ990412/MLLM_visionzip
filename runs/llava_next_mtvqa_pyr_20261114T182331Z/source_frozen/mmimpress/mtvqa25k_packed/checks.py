"""CPU identity / budget checks shared by the Phase 0 freeze and the Phase 1 gates.

Every function is read-only with respect to existing runs, results and data
and returns a JSON-serializable record whose ``status`` is PASS or FAIL (the
caller stops on FAIL).  No model, no GPU.
"""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
import contextlib
import copy
import importlib.metadata
import io
import json
import math
import sys
import types

from mmimpress.mtvqa25k.common import ROOT, read, sha
from mmimpress.mtvqa25k_packed import reference as REF

WORKLOAD_SHA = 'eb3c5e812fd6937cc0741cac6dd85d6839a7d00288dffbfa7ea8294272269cc8'
DATASET_CONFIG_SHA = '1b99478b0c0f253810453f169b91f9843973c05dd815b3cc46aeabcc2a07b933'
RATIO_SHA = {'PYR-beta-raw': '51f9b0db19a93f8ca8b34b8beb8af21285721c5ea460566afbd9a3185b331724',
             'U': '54ea273e1a38b1e8b02b96237381f982344cd28ca8466162876e161cdb8ff3ba'}
VECTOR_SOURCE = 'results/layer_budget_kv25_pyrbeta_20261110T073604Z'
DATA = 'data/mt_vqa_v2_reconstructed_25k'
PHASES = ('smoke', 'pilot40', 'main')
EXPECTED_K = {960: {'head': [468, 453, 439], 'tail': [41, 27, 12]}, 1344: {'tail': [58, 37, 17]},
              2208: {'tail': [95, 61, 28]}, 2880: {'head': [1404, 1360, 1316], 'tail': [124, 80, 36]}}
ROW_BYTES = 32 * 128 * 2          # num_heads * head_dim * sizeof(fp16)
NUM_LAYERS = 32


def hamilton(ratios, n, r_max):
    """Independent Hamilton apportionment: no library code and no lower bound."""
    L = len(ratios)
    budget = L * (-(-n // 4))
    cap = min(n, math.ceil(Fraction(repr(float(r_max))) * n))
    q = [r * n for r in ratios]
    k = [min(cap, int(math.floor(x))) for x in q]
    frac = [x - math.floor(x) for x in q]
    order = sorted(range(L), key=lambda i: (-frac[i], i))
    need = budget - sum(k)
    while need:
        moved = False
        for i in order:
            if need and k[i] < cap:
                k[i] += 1; need -= 1; moved = True
        if not moved:
            raise ValueError('budget cannot be apportioned')
    return k


def doc_counts(doc, n):
    """k_l from the document's own clip range (never the module defaults)."""
    from mmimpress.layer_budget import layer_keep_counts
    return layer_keep_counts(doc['layer_keep_ratios'], int(n), int(doc['num_layers']), float(doc['total_ratio']),
                             float(doc['r_min']), float(doc['r_max']))


def packed_bytes(counts, n_newline):
    return sum(2 * k * ROW_BYTES for k in counts) + 2 * NUM_LAYERS * int(n_newline) * ROW_BYTES


# ------------------------------------------------------------------ data
def data_identity(run, verify_images=True, workers=8):
    from mmimpress.mtvqa25k.adapter import dialogues_for_base
    from mmimpress.mtvqa25k.dataset import workload_hash
    out = {'status': 'FAIL', 'checks': {}, 'failures': []}

    def need(ok, name, detail=None):
        out['checks'][name] = bool(ok)
        if not ok:
            out['failures'].append({'check': name, 'detail': detail})

    v3 = ROOT / REF.V3_RUN / 'llava'
    manifests = {}
    for phase in PHASES:
        mine = Path(run) / 'llava' / phase / 'manifest.json'
        need(sha(mine) == sha(v3 / phase / 'manifest.json'), f'{phase}_manifest_sha_equals_v3', str(mine))
        manifests[phase] = read(mine)
        out[f'{phase}_manifest_sha256'] = sha(mine)
    need(sha(ROOT / DATA / 'config.json') == DATASET_CONFIG_SHA, 'dataset_config_sha256')
    config = read(ROOT / DATA / 'config.json')
    need(config['ordered_workload_sha256'] == WORKLOAD_SHA, 'config_ordered_workload_sha256')
    need(sha(ROOT / DATA / 'dialogues.json') == config['files']['dialogues.json'], 'dialogues_file_sha256')
    ds = read(ROOT / DATA / 'dialogues.json')['dialogues']
    need(workload_hash(ds) == WORKLOAD_SHA, 'ordered_workload_sha256_recomputed')
    main = manifests['main']
    need(main['dialogues'] == dialogues_for_base(ds), 'main_manifest_dialogues_equal_hashed_workload')
    images = [d['image_id'] for d in main['dialogues']]
    need(len(main['dialogues']) == 25000 and len(set(images)) == 25000, 'main_25000_images_one_dialogue_each')
    need(all([t['turn_id'] for t in d['turns']] == [1, 2, 3] for d in main['dialogues']), 'three_turns_each')
    qids = [str(t['question_id']) for d in main['dialogues'] for t in d['turns']]
    need(len(qids) == 75000 and len(set(qids)) == 75000, 'main_75000_unique_questions')
    need(set(main['image_identity']) == set(images), 'image_identity_membership')
    bydid = {d['dialog_id']: d for d in main['dialogues']}
    for phase, n in (('smoke', 4), ('pilot40', 40)):
        sub = manifests[phase]
        core = lambda d: {k: v for k, v in d.items() if k != 'global_dialog_ordinal'}
        need(len(sub['dialogues']) == n and all(core(bydid.get(d['dialog_id'], {})) == core(d) for d in sub['dialogues']), f'{phase}_dialogues_subset_of_main')
        need(all(sub['image_identity'][d['image_id']] == main['image_identity'][d['image_id']] for d in sub['dialogues']), f'{phase}_image_identity_equals_main')
    if verify_images:
        def bad(d):
            ident = main['image_identity'][d['image_id']]
            return None if sha(ROOT / d['image_path']) == ident['sha256'] == d['image_sha256'] else d['image_id']
        with ThreadPoolExecutor(max_workers=workers) as pool:
            wrong = [x for x in pool.map(bad, main['dialogues']) if x is not None]
        need(not wrong, 'all_25000_image_files_sha256', wrong[:20])
        out['image_files_hashed'] = len(main['dialogues'])
    out.update(images=len(images), dialogues=len(main['dialogues']), requests_per_arm=len(qids),
               ordered_workload_sha256=WORKLOAD_SHA, dataset_config_sha256=DATASET_CONFIG_SHA,
               status='PASS' if not out['failures'] else 'FAIL')
    return out


# ---------------------------------------------------------------- scorer
def scorer_validation():
    """Re-run the v3 scorer fixtures (scripts/135 logic) and compare with the stored v3 receipts."""
    from lib2to3.refactor import RefactoringTool
    from mmimpress.mtvqa25k.scorer import Annotations, REVISION, reference_scores, independent_score
    out = {'status': 'FAIL', 'failures': [], 'revision': REVISION}
    stored = read(ROOT / REF.V3_RESULTS / 'scorer/synthetic_fixtures.json')
    v3 = read(ROOT / REF.V3_RESULTS / 'scorer/validation.json')
    prov = read(ROOT / 'mmimpress/mtvqa25k/vendor/provenance.json')
    if v3['status'] != 'PASS' or v3['revision'] != REVISION:
        out['failures'].append('v3 scorer validation receipt')
    if sha(ROOT / 'mmimpress/mtvqa25k/vendor/vqaEval.py.original') != prov['original_sha256'] or sha(ROOT / 'mmimpress/mtvqa25k/vendor/vqaEval.py') != prov['ported_sha256']:
        out['failures'].append('vendor evaluator SHA256')
    anns = [copy.deepcopy(f['annotation']) for f in stored]
    predictions = {int(a['question_id']): f['prediction'] for a, f in zip(anns, stored)}
    ref, aggregate = reference_scores(anns, predictions)
    original = (ROOT / 'mmimpress/mtvqa25k/vendor/vqaEval.py.original').read_text()
    namespace = {}
    exec(compile(str(RefactoringTool(['lib2to3.fixes.fix_print']).refactor_string(original, 'vqaEval.py')), 'upstream_reference', 'exec'), namespace)
    upstream = namespace['VQAEval'](Annotations(anns), types.SimpleNamespace(qa={q: {'answer': s} for q, s in predictions.items()}), n=10)
    with contextlib.redirect_stdout(io.StringIO()):
        upstream.evaluate()
    if upstream.evalQA != {q: 100 * s for q, s in ref.items()} or upstream.accuracy != aggregate:
        out['failures'].append('ported evaluator differs from the unmodified upstream source')
    for a, f in zip(anns, stored):
        qid = int(a['question_id'])
        check = independent_score(f['prediction'], a)
        same = (abs(ref[qid] - f['official']) < 1e-12 and abs(check['vqa_score'] - f['vqa_score']) < 1e-12
                and all(check[k] == f[k] for k in ('normalized_answer', 'normalization_applied', 'normalized_human_answers', 'matches', 'legacy_same_normalization')))
        if not same:
            out['failures'].append('fixture ' + f['fixture'])
    out.update(fixtures=len(stored), v3_fixture_file_sha256=sha(ROOT / REF.V3_RESULTS / 'scorer/synthetic_fixtures.json'),
               vendor_original_sha256=prov['original_sha256'], vendor_ported_sha256=prov['ported_sha256'],
               metric='ten-answer leave-one-out official VQA accuracy', status='PASS' if not out['failures'] else 'FAIL')
    return out


# --------------------------------------------------------------- vectors
def vectors(run):
    from mmimpress.layer_budget import load_budget_document, budget_sha256
    out = {'status': 'FAIL', 'failures': [], 'vectors': {}}
    for vec, want in RATIO_SHA.items():
        mine = Path(run) / 'vectors' / f'layer_budget_llava_{vec}.json'
        src = ROOT / VECTOR_SOURCE / f'layer_budget_llava_{vec}.json'
        doc = load_budget_document(mine)
        ok = (sha(mine) == sha(src) and doc['layer_budget_sha256'] == want == budget_sha256(doc['layer_keep_ratios'])
              and doc['alloc_id'] == vec and doc['num_layers'] == 32 and doc['total_ratio'] == 0.25)
        if not ok:
            out['failures'].append(vec)
        out['vectors'][vec] = {'file_sha256': sha(mine), 'source_file_sha256': sha(src), 'ratio_sha256': doc['layer_budget_sha256'],
                               'r_min': doc['r_min'], 'r_max': doc['r_max'], 'ratio_sum': sum(doc['layer_keep_ratios'])}
    raw = out['vectors'].get('PYR-beta-raw', {})
    if not (raw.get('r_min') == 0.0 and raw.get('r_max') == 0.6 and abs(raw.get('ratio_sum', 0) - 8.0) < 1e-9):
        out['failures'].append('PYR-beta-raw clip range / sum')
    out['copied_not_recomputed'] = True
    out['status'] = 'PASS' if not out['failures'] else 'FAIL'
    return out


# ---------------------------------------------------------------- budget
def k_tables(run, index=None):
    """k_l for the actual N of all 25,000 v3 images + PYR packed payload bytes vs the v3 [B] follow-up read bytes."""
    from mmimpress.layer_budget import load_budget_document
    run = Path(run)
    docs = {v: load_budget_document(run / 'vectors' / f'layer_budget_llava_{v}.json') for v in RATIO_SHA}
    raw, uni = docs['PYR-beta-raw'], docs['U']
    manifest = read(run / 'llava/main/manifest.json')
    index = index if index is not None else REF.load_index(run / 'reference/v3_main_index.jsonl.gz')
    out = {'status': 'FAIL', 'failures': [], 'images': 0, 'zero_layers': 0}
    per_n = {}; per_image = []; byte_diffs = []
    for d in manifest['dialogues']:
        iid = d['image_id']
        rows = index.get(iid)
        if rows is None:
            out['failures'].append({'check': 'reference_row_missing', 'image_id': iid}); continue
        ns = {r['N_content'] for r in rows.values()}
        if len(ns) != 1 or manifest['image_identity'][iid]['N_content'] not in ns:
            out['failures'].append({'check': 'N_disagrees', 'image_id': iid}); continue
        n = ns.pop()
        newline = {r['n_newline'] for r in rows.values()}
        if len(newline) != 1 or manifest['image_identity'][iid]['newline_count'] not in newline:
            out['failures'].append({'check': 'newline_count_disagrees', 'image_id': iid}); continue
        n_sep = newline.pop()
        if n not in per_n:
            k = doc_counts(raw, n)
            ku = doc_counts(uni, n)
            per_n[n] = {'N': n, 'budget': NUM_LAYERS * math.ceil(n / 4), 'k_per_layer': k, 'sum': sum(k), 'min': min(k), 'max': max(k),
                        'zero_layers': sum(1 for x in k if x == 0), 'independent_hamilton_equal': k == hamilton(raw['layer_keep_ratios'], n, raw['r_max']),
                        'uniform_rows_equal_ceil': ku == [(n + 3) // 4] * NUM_LAYERS, 'images': 0}
        t = per_n[n]; t['images'] += 1
        if t['sum'] != t['budget'] or t['zero_layers'] or not t['independent_hamilton_equal'] or not t['uniform_rows_equal_ceil']:
            out['failures'].append({'check': 'k_l', 'image_id': iid, 'N': n})
        pyr_bytes = packed_bytes(t['k_per_layer'], n_sep)
        uni_bytes = packed_bytes([(n + 3) // 4] * NUM_LAYERS, n_sep)
        b_bytes = sorted({rows[(REF.V3_B, turn)]['ssd_read_bytes'] for turn in (2, 3)})
        per_image.append({'image_id': iid, 'N': n, 'n_newline': n_sep, 'sum_k': t['sum'], 'min_k': t['min'], 'pyr_packed_bytes': pyr_bytes,
                          'v3_B_followup_read_bytes': b_bytes[0] if len(b_bytes) == 1 else b_bytes})
        if b_bytes != [pyr_bytes] or uni_bytes != pyr_bytes:
            byte_diffs.append({'image_id': iid, 'N': n, 'pyr_packed_bytes': pyr_bytes, 'uniform_packed_bytes': uni_bytes, 'v3_B_bytes': b_bytes})
        out['images'] += 1
    for n, want in EXPECTED_K.items():
        if n not in per_n:
            out['failures'].append({'check': 'expected_N_absent', 'N': n}); continue
        k = per_n[n]['k_per_layer']
        if ('head' in want and k[:3] != want['head']) or k[-3:] != want['tail']:
            out['failures'].append({'check': 'precomputed_k_mismatch', 'N': n, 'head': k[:3], 'tail': k[-3:]})
    out['zero_layers'] = sum(t['zero_layers'] * t['images'] for t in per_n.values())
    out['min_k_overall'] = min(t['min'] for t in per_n.values()) if per_n else None
    out['distinct_N'] = len(per_n)
    out['per_N'] = [per_n[n] for n in sorted(per_n)]
    out['payload_bytes_vs_v3_B'] = {'images_compared': len(per_image), 'different': len(byte_diffs), 'examples': byte_diffs[:20],
                                    'rule': 'sum_l 2*k_l*8192 + 2*32*newline*8192 == v3 ours_prefixstore_kv25 T2 and T3 ssd_read_bytes'}
    if byte_diffs:
        out['failures'].append({'check': 'payload_bytes_differ_from_v3_B', 'count': len(byte_diffs)})
    if out['zero_layers']:
        out['failures'].append({'check': 'BLOCKED_zero_k_layer', 'count': out['zero_layers']})
    if out['images'] != 25000:
        out['failures'].append({'check': 'images', 'count': out['images']})
    out['expected_precomputed'] = {str(k): v for k, v in EXPECTED_K.items()}
    out['status'] = 'PASS' if not out['failures'] else 'FAIL'
    out['failures'] = out['failures'][:50]
    return out, per_image


# ----------------------------------------------------------- model / env
def model_environment():
    """Current snapshot files / interpreter / packages against the v3 model_freeze.json and environment.json."""
    v3 = ROOT / REF.V3_RUN
    frozen = read(v3 / 'model_freeze.json')['llava']
    env = read(v3 / 'environment.json')['llava']
    diffs = []
    for name, record in frozen['files'].items():
        path = Path(frozen['snapshot']) / name
        if not path.exists():
            diffs.append({'file': name, 'reason': 'missing'}); continue
        if str(path.resolve()) != record['resolved'] or path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            diffs.append({'file': name, 'reason': 'resolved path / size / SHA256 changed'})
    if sys.version != env['python']:
        diffs.append({'python': sys.version, 'v3': env['python']})
    packages = {}
    for name, version in env['packages'].items():
        packages[name] = importlib.metadata.version(name)
        if packages[name] != version:
            diffs.append({'package': name, 'now': packages[name], 'v3': version})
    return {'status': 'PASS' if not diffs else 'FAIL', 'differences': diffs, 'snapshot': frozen['snapshot'], 'files_hashed': len(frozen['files']),
            'python': sys.version, 'executable': sys.executable, 'v3_executable': env.get('executable'), 'packages': packages,
            'v3_model_freeze_sha256': sha(v3 / 'model_freeze.json'), 'v3_environment_sha256': sha(v3 / 'environment.json')}


# ------------------------------------------------------------- reference
def reference_summary(index, manifest, methods=REF.V3_METHODS):
    """Completeness of a reference index and the v3 T1 cross-arm token identity."""
    missing = []; t1_disagree = []; n = 0
    for d in manifest['dialogues']:
        rows = index.get(d['image_id'])
        if rows is None or set(rows) != {(m, t) for m in methods for t in (1, 2, 3)}:
            missing.append(d['image_id']); continue
        if any(str(rows[(m, t)]['question_id']) != str(d['turns'][t - 1]['question_id']) for m in methods for t in (1, 2, 3)):
            missing.append(d['image_id']); continue
        if len({json.dumps(rows[(m, 1)]['generated_token_ids']) for m in methods}) != 1:
            t1_disagree.append(d['image_id'])
        n += 1
    return {'images': len(manifest['dialogues']), 'complete_images': n, 'missing_or_malformed': missing[:50], 'missing_count': len(missing),
            'v3_T1_tokens_identical_across_all_arms': len(manifest['dialogues']) - len(missing) - len(t1_disagree),
            'v3_T1_cross_arm_disagreements': t1_disagree[:50],
            'status': 'PASS' if not missing and not t1_disagree else ('BLOCKED_REFERENCE' if missing else 'FAIL')}

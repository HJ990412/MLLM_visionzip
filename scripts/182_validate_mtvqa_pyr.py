#!/usr/bin/env python3
"""Phase 1 gates of the LLaVA-NeXT PYR-beta-raw packed MT-VQA-25k run
(contract docs/llava_next_mtvqa_pyr_contract.md, section 6).  Every receipt is bound to the
current experiment_freeze.json; a re-freeze invalidates all of them.

  --stage vg1   data + scoring identity (CPU)
  --stage vg2   vectors, k_l table on the 25,000 real N, PYR packed bytes == v3 [B] read bytes (CPU)
  --stage vg3   floor propagation: raw-vector fixture stores pass every validator incl. the new audit path;
                meta r_min -> 0.10 fails closed everywhere (CPU)
  --stage vg6   full existing CPU suite + new tests + existing-source protection (CPU)
  --stage vg4   packed path on five fixed MT-VQA cases (GPU; the checks of scripts/166 --stage gpu)
  --stage vg5   G-det-U: U_packed (phase gate_u_pilot40, 120 requests) vs v3 pilot40 [B], request by request
  --stage t1 --phase smoke|pilot40   PYR arm Turn-1 tokens vs v3 Turn-1
  --stage summary
"""
import argparse, gc, importlib.util, json, os, random, shutil, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from mmimpress.mtvqa25k.common import read, dump, sha
from mmimpress.mtvqa25k_packed import adapter as AD, checks as CK, reference as REF

GATES = ('vg1_data_scoring_identity', 'vg2_budget', 'vg3_floor_propagation', 'vg4_packed_path', 'vg5_g_det_U', 'vg6_regression')
SEED = 1234


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / file)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def publish(run, name, record):
    """Gate receipts are never overwritten: a previous receipt (e.g. a failed attempt) is kept beside the new one."""
    path = run / 'gates' / f'{name}.json'
    path.parent.mkdir(exist_ok=True)
    record = dict(record, gate=name, source_freeze_sha256=sha(run / 'experiment_freeze.json'), recorded_unix=time.time())
    if path.exists():
        os.replace(path, run / 'gates' / f'{name}.superseded_{time.time_ns()}.json')
    dump(path, record)
    print(name, record['status'], flush=True)
    if record['status'] != 'PASS':
        raise SystemExit(f'{name} {record["status"]}')
    return record


def verify_freeze(run):
    W = load('_pyr_runner_for_gates', '183_run_mtvqa_pyr.py')
    return W, W.verify_freeze(run)


# ------------------------------------------------------------------- vg1
def stage_vg1(run):
    verify_freeze(run)
    data = CK.data_identity(run); scorer = CK.scorer_validation()
    publish(run, 'vg1_data_scoring_identity', {'status': 'PASS' if data['status'] == scorer['status'] == 'PASS' else 'FAIL', 'data': data, 'scorer': scorer})


# ------------------------------------------------------------------- vg2
def stage_vg2(run):
    verify_freeze(run)
    vec = CK.vectors(run); table, _ = CK.k_tables(run)
    table_small = dict(table, per_N=[{k: v for k, v in t.items() if k != 'k_per_layer'} | {'k_head': t['k_per_layer'][:3], 'k_tail': t['k_per_layer'][-3:]} for t in table['per_N']])
    publish(run, 'vg2_budget', {'status': 'PASS' if vec['status'] == table['status'] == 'PASS' else 'FAIL', 'vectors': vec, 'k_tables': table_small})


# ------------------------------------------------------------------- vg3
def stage_vg3(run):
    """Same fixture and fail-closed list as scripts/166 --stage grmin, on MT-VQA N (min / median / max) and with the new adapter audit path."""
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch
    from mmimpress.layer_budget import layer_keep_counts, load_budget_document
    from mmimpress.packed_serve import PackedImageContext, PackedPrefixSelector, PackedReadTrace
    from mmimpress.packed_store import PACKED_NAME, validate_packed_meta, write_packed_store
    from mmimpress.prefix25 import seal_integrity, validate_llava_prefix_layerwise_meta
    from mmimpress.serve import BIAS, CVPR25ChunkSelector, ImageContext
    from mmimpress.store import write_image_store
    import mmimpress.layer_budget as LB
    verify_freeze(run)
    G = load('_pyr_gates_166', '166_validate_layer_budget_pyrbeta.py')
    M = AD.bind_packed(run.name, '_mtvqa25k_packed_llava_vg3')
    raw = load_budget_document(run / 'vectors/layer_budget_llava_PYR-beta-raw.json')
    assert raw['r_min'] == 0.0 and raw['r_max'] == 0.6 and LB.R_MIN == 0.10 and LB.R_MAX == 0.60
    out = {'status': 'RUNNING', 'global_constants': {'R_MIN': LB.R_MIN, 'R_MAX': LB.R_MAX, 'unchanged': True},
           'raw_document': {'r_min': raw['r_min'], 'r_max': raw['r_max'], 'layer_budget_sha256': raw['layer_budget_sha256']}, 'cases': [], 'fail_closed': {}}
    base = run / 'gates' / 'vg3_fixture'
    if base.exists():
        shutil.rmtree(base)
    runner = SimpleNamespace(model=SimpleNamespace(device=torch.device('cpu')))
    offending = None
    try:
        for n in (960, 2304, 2880):                     # MT-VQA min / median / max N
            root = base / f'N{n}'
            fx = G._fixture32(root / 'a', n)
            args = (fx['tensors'], fx['vstart'], fx['vn'], fx['prefix_input_ids'], fx['newline'])
            common = dict(extra=fx['flags'], stored_to_original=fx['perm'], separator_sidecar=True, probe_heads=0)
            lw = write_image_store(root / 'lw', *args, storage_policy='prefix25_layerwise', layer_budget=raw, **common)
            pk = write_packed_store(root / 'p', *args, storage_policy='prefix25_packed', layer_budget=raw, **common)
            for name in ('lw', 'p'):
                (root / name / 'visionzip_layout.pt').write_bytes((root / 'a' / 'visionzip_layout.pt').read_bytes())
                seal_integrity(root / name)
            want = CK.hamilton(raw['layer_keep_ratios'], n, raw['r_max'])
            assert min(want) >= 1 and sum(want) == 32 * ((n + 3) // 4)
            for meta in (lw, pk):
                assert meta['layer_budget_r_min'] == 0.0 and meta['layer_budget_r_max'] == 0.6
                assert meta['layer_keep_counts'] == want == meta['payload_rows_per_layer'] and meta['layer_keep_ratios'] == raw['layer_keep_ratios']
            validate_llava_prefix_layerwise_meta(lw)                      # prefix25 validator
            validate_packed_meta(pk)                                      # packed_store validator
            ctx_u = ImageContext(root / 'lw', 'cpu', drop_cache=False, require_v_hidden=False); ctx_u.validate_visual_kv_layout()
            ctx_p = PackedImageContext(root / 'p', 'cpu', pin=False); ctx_p.validate_packed_layout()
            # new audit path: the bound 164 module and the shared CPU checks both use the document clip range
            assert M.doc_counts(raw, n) == want == M.independent_counts(raw['layer_keep_ratios'], n, raw['r_max']) == CK.doc_counts(raw, n)
            vstart, vn = fx['vstart'], fx['vn']
            cache_u = ctx_u.cache.new_request()
            sel_u = CVPR25ChunkSelector(runner, ctx_u, None, .25, 'prefix', sep_policy='sidecar', budget_unit='visual_kv')
            with patch('torch.cuda.synchronize'):
                sel_u.prepare(None)
            bias_u = [BIAS[li].clone() for li in range(32)]; BIAS.clear()
            cache_p = ctx_p.cache.new_request()
            sel_p = PackedPrefixSelector(runner, ctx_p, .25)
            with patch('torch.cuda.synchronize'), PackedReadTrace(root / 'p') as trace:
                sel_p.prepare()
            s = trace.summary()
            assert s['preads'] == 1 and s['preadv_calls'] == 1 and s['pread_calls'] == 0 and s['bytes'] == pk['packed_total_bytes'] == (root / 'p' / PACKED_NAME).stat().st_size
            visible = []
            for li in range(32):
                b = BIAS[li][0, 0, 0]
                vis = int((b[vstart:vstart + n] == 0).sum())
                assert vis == want[li] and bool((b[vstart:vstart + want[li]] == 0).all()) and bool((b[vstart + n:vstart + vn] == 0).all())
                assert torch.equal(BIAS[li], bias_u[li])
                for kind in ('keys', 'values'):
                    assert torch.equal(getattr(cache_p.layers[li], kind), getattr(cache_u.layers[li], kind))
                visible.append(vis)
            assert sel_p.stats()['k_per_layer'] == want == sel_u.stats()['k_per_layer']
            BIAS.clear()
            out['cases'].append({'N': n, 'k_head': want[:3], 'k_tail': want[-3:], 'k_last_layer': want[-1], 'min_k': min(want), 'visible_rows_per_layer_equal_k': visible == want,
                                 'validators_pass': ['prefix25.validate_llava_prefix_layerwise_meta', 'packed_store.validate_packed_meta', 'ImageContext.__init__ + validate_visual_kv_layout',
                                                     'PackedImageContext.__init__ + validate_packed_layout', 'CVPR25ChunkSelector (layerwise)', 'PackedPrefixSelector',
                                                     'bound 164 doc_counts / independent_counts (new adapter audit path)', 'mtvqa25k_packed.checks.doc_counts'],
                                 'packed_equals_unpacked_cache_and_masks': True, 'single_preadv': True, 'packed_total_bytes': pk['packed_total_bytes']})
            if n == 2304:
                fc = out['fail_closed']; X = G._expect_fail
                X('validate_llava_prefix_layerwise_meta(meta r_min=0.10)', lambda: validate_llava_prefix_layerwise_meta(dict(lw, layer_budget_r_min=0.10)), fc)
                X('validate_packed_meta(meta r_min=0.10)', lambda: validate_packed_meta(dict(pk, layer_budget_r_min=0.10)), fc)
                for name, opener in (('lw', lambda p: ImageContext(p, 'cpu', drop_cache=False, require_v_hidden=False)), ('p', lambda p: PackedImageContext(p, 'cpu', pin=False))):
                    bad = root / f'{name}_rmin010'
                    shutil.copytree(root / name, bad)
                    m = json.loads((bad / 'meta.json').read_text()); m['layer_budget_r_min'] = 0.10
                    (bad / 'meta.json').write_text(json.dumps(m, indent=1)); (bad / 'integrity.json').unlink(); seal_integrity(bad)
                    X(f'activation of {name} store with meta.json r_min=0.10 (integrity re-sealed)', lambda b=bad, o=opener: o(b), fc)
                ctx_p.meta = dict(ctx_p.meta, layer_budget_r_min=0.10)
                X('PackedPrefixSelector(meta r_min=0.10)', lambda: PackedPrefixSelector(runner, ctx_p, .25), fc)
                ctx_u.meta = dict(ctx_u.meta, layer_budget_r_min=0.10)
                X('CVPR25ChunkSelector(meta r_min=0.10)', lambda: CVPR25ChunkSelector(runner, ctx_u, None, .25, 'prefix', sep_policy='sidecar', budget_unit='visual_kv'), fc)
                bad_doc = dict(raw, r_min=0.10)
                X('write_packed_store(document r_min=0.10)', lambda: write_packed_store(root / 'w1', *args, storage_policy='prefix25_packed', layer_budget=bad_doc, **common), fc)
                X('write_image_store(document r_min=0.10)', lambda: write_image_store(root / 'w2', *args, storage_policy='prefix25_layerwise', layer_budget=bad_doc, **common), fc)
                tmp = root / 'doc_rmin010.json'; tmp.write_text(json.dumps(bad_doc))
                X('load_budget_document(document r_min=0.10)', lambda: load_budget_document(tmp), fc)
                X('layer_keep_counts(raw ratios, N) with module defaults [deliberate negative]', lambda: layer_keep_counts(raw['layer_keep_ratios'], n), fc)
                X('bound 164 doc_counts(document r_min=0.10)', lambda: M.doc_counts(bad_doc, n), fc)
                X('checks.doc_counts(document r_min=0.10)', lambda: CK.doc_counts(bad_doc, n), fc)
                # new audit path end to end: a persisted meta whose r_min is 0.10 must fail the per-image audit rule
                cfg_doc = AD.budget_entry(raw, 'fixture')
                ok_rule = lambda meta: meta['layer_budget_r_min'] == cfg_doc['r_min'] and meta['layer_budget_r_max'] == cfg_doc['r_max'] and meta['layer_keep_ratios'] == cfg_doc['layer_keep_ratios']
                assert ok_rule(pk) and not ok_rule(dict(pk, layer_budget_r_min=0.10))
                fc['164 audit rule PK_clip_range_and_ratios_from_document (meta r_min=0.10)'] = {'fail_closed': True}
            ctx_p.close(); ctx_u.close(); del fx, cache_u, cache_p; gc.collect()
        scan = []
        files = ('mmimpress/layer_budget.py', 'mmimpress/store.py', 'mmimpress/packed_store.py', 'mmimpress/packed_serve.py', 'mmimpress/serve.py', 'mmimpress/prefix25.py',
                 'mmimpress/piggyback.py', 'scripts/main_b_support.py', 'scripts/164_eval_layer_budget_pyrbeta_hold.py', 'mmimpress/mtvqa25k_packed/adapter.py',
                 'mmimpress/mtvqa25k_packed/checks.py', 'mmimpress/mtvqa25k_packed/reference.py', 'mmimpress/mtvqa25k_packed/analysis.py',
                 'scripts/181_freeze_mtvqa_pyr.py', 'scripts/182_validate_mtvqa_pyr.py', 'scripts/183_run_mtvqa_pyr.py', 'scripts/184_analyze_mtvqa_pyr.py')
        for f in files:
            scan += G._default_reliant_calls(ROOT / f)
        implicit = [c for c in scan if not c['r_min_explicit']]
        allowed = [c for c in implicit if c['file'] == 'scripts/182_validate_mtvqa_pyr.py']      # the deliberate negative test above
        offending = [c for c in implicit if c not in allowed]
        out['static_scan'] = {'files': list(files), 'calls': len(scan), 'default_reliant_in_store_serving_or_new_path': offending, 'deliberate_default_calls_in_gate_script': allowed,
                              'note': 'scripts/156 (helpers imported by 164) contains default-reliant audit calls; none of them is executed here: the adapter calls only 164.audit_image_rows / run_image, which use doc_counts'}
        assert not offending and len(allowed) == 1, (offending, allowed)
        # arm handling by name: no positional "recompute first" assumption in the new path
        src = '\n'.join((ROOT / f).read_text() for f in files if 'mtvqa25k_packed' in f or f.startswith('scripts/18'))
        out['positional_arm_assumptions'] = {'methods[0]': 'methods[0]' in src.replace("'methods[0]'", ''), 'methods[1:]': 'methods[1:]' in src.replace("'methods[1:]'", '')}
        assert not any(out['positional_arm_assumptions'].values())
        assert M.stored_methods([AD.MAIN_METHOD]) == [AD.MAIN_METHOD] and M.kind_of(AD.MAIN_METHOD) == 'packed_PYR-beta-raw' and M.method_order([AD.MAIN_METHOD], 7) == [AD.MAIN_METHOD]
        out['status'] = 'PASS'
    except Exception:
        out['status'] = 'FAIL'; out['error'] = traceback.format_exc(); print(out['error'], flush=True)
    finally:
        BIAS.clear()
        if base.exists():
            shutil.rmtree(base)                         # run-local CPU fixture only
        out['fixture_deleted'] = not base.exists()
    publish(run, 'vg3_floor_propagation', out)


# ------------------------------------------------------------------- vg6
def stage_vg6(run):
    verify_freeze(run)
    F = load('_pyr_freeze_for_gates', '181_freeze_mtvqa_pyr.py')
    G154 = load('_layer_budget_gates_v1_cpu', '154_validate_layer_budget_kv25.py')
    out = {'status': 'FAIL'}
    try:
        G154.stage_cpu(run)                              # writes run/cpu_tests.log and run/gates/cpu_tests.json; GLM modules separately
    except AssertionError:
        pass
    cpu = read(run / 'gates/cpu_tests.json')
    prot = F.check_existing_sources(run)
    out.update(cpu_tests=cpu, existing_source_protection=prot,
               glm_preexisting_failures_reported_separately=len(cpu.get('glm_errors', [])),
               status='PASS' if cpu['status'] == 'PASS' and prot['status'] == 'PASS' else 'FAIL')
    publish(run, 'vg6_regression', out)


# ------------------------------------------------------------------- vg4
def fixed_cases(run):
    """Five pilot40 images chosen from frozen geometry only (before any model output of this run)."""
    manifest = read(run / 'llava/pilot40/manifest.json')
    main = read(run / 'llava/main/manifest.json')
    ident = manifest['image_identity']
    images = {str(r['image_id']): r for r in read(ROOT / CK.DATA / 'image_manifest.json')}
    ds = manifest['dialogues']
    num = lambda d: int(d['image_id'])
    aspect = lambda d: images[d['image_id']]['width'] / images[d['image_id']]['height']
    ns = sorted(v['N_content'] for v in main['image_identity'].values()); median = ns[(len(ns) - 1) // 2]
    chosen = []; reasons = {}

    def choose(label, key):
        d = min((d for d in ds if d['dialog_id'] not in chosen), key=lambda d: (key(d), num(d)))
        chosen.append(d['dialog_id']); reasons[label] = {'dialog_id': d['dialog_id'], 'image_id': d['image_id'], 'N': ident[d['image_id']]['N_content'],
                                                         'hires_grid': ident[d['image_id']]['hires_grid'], 'aspect_w_over_h': aspect(d)}
    choose('min_N', lambda d: ident[d['image_id']]['N_content'])
    choose('max_N', lambda d: -ident[d['image_id']]['N_content'])
    choose('median_N', lambda d: abs(ident[d['image_id']]['N_content'] - median))
    choose('portrait_extreme', aspect)
    choose('landscape_extreme', lambda d: -aspect(d))
    bydid = {d['dialog_id']: d for d in ds}
    samples = []
    for label, r in reasons.items():
        d = bydid[r['dialog_id']]
        samples.append({'label': label, 'image_id': d['image_id'], 'image_path': str(ROOT / d['image_path']), 'image_sha256': ident[d['image_id']]['sha256'],
                        'question': d['turns'][0]['question'], 'question_id': str(d['turns'][0]['question_id']), 'newline_count': ident[d['image_id']]['newline_count']})
    return samples, {'main_median_N': median, 'cases': reasons,
                     'rule': 'pilot40 images, picked one after another without replacement by frozen geometry (ties: lower image id); the widest pilot40 image is also the '
                             'minimum-N image, so landscape_extreme is the next widest'}


def stage_vg4(run):
    import numpy as np, torch
    verify_freeze(run)
    from mmimpress.layer_budget import load_budget_document
    from mmimpress.recompute_kv25 import guarded_rmtree
    P = load('prefix25_persistence_existing', '106_prefix25_persistence.py')
    G = load('layer_budget_gates_v1', '154_validate_layer_budget_kv25.py')
    S = load('packed_support_v2', 'packed_support_v2.py')
    M161 = load('packed_gates_v2', '161_validate_packed_store_v2.py')
    G166 = load('_pyr_gates_166_gpu', '166_validate_layer_budget_pyrbeta.py')
    G166.GATE_VECTORS = ('PYR-beta-raw', 'U')            # private module instance; the file is untouched
    docs = {v: load_budget_document(run / 'vectors' / f'layer_budget_llava_{v}.json') for v in G166.GATE_VECTORS}
    samples, selection = fixed_cases(run)
    P.exclusive(); P.capacity()
    torch.set_grad_enabled(False)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(2)
    exp = P.Experiment(run, 'llava')
    gate = {'status': 'RUNNING', 'atol': G166.ATOL, 'rtol': G166.RTOL, 'vectors': {a: d['layer_budget_sha256'] for a, d in docs.items()},
            'clip_range': {a: [d['r_min'], d['r_max']] for a, d in docs.items()}, 'selection': selection, 'samples': [],
            'note': 'scripts/166 pyr_case on five fixed MT-VQA cases. Per case and vector: unpacked layerwise store (existing writer, 65 preads) and packed store from the same Turn-1 '
                    'capture; packed slices bit-identical to the unpacked per-layer files, file size == expected bytes, restored cache bits == FP16 round trip of the captured KV '
                    'at the first k_l rows of the order table, masks / first logits / generated ids vs an independent in-memory reference of the same shape, visible rows per '
                    'layer == k_l (last layer included), one preadv and zero pread per hit, returned bytes == file size, question-invariant read range, zero vision / score calls'}
    base = run / 'llava' / 'gpu_gates'
    try:
        for i, s in enumerate(samples):
            path = base / f"{i:02d}_{s['image_id']}"
            path.mkdir(parents=True, exist_ok=False)
            case = G166.pyr_case(exp, G, S, M161, s, path, docs)
            exp.close_contexts()
            case['label'] = s['label']
            for v in docs:
                want = CK.hamilton(docs[v]['layer_keep_ratios'], case['N'], docs[v]['r_max'])
                expected = CK.packed_bytes(want, s['newline_count'])
                assert case['k_per_layer'][v] == want and case['bytes'][f'P_{v}_scatter'] == expected == case['G6_bytes'][v]['total_bytes'], (v, expected)
                for mode in ('scatter', 'directcopy'):
                    c = case['G6_outputs'][f'P_{v}_{mode}']; g7 = case['G7'][f'P_{v}_{mode}']
                    assert c['visible_rows_equal_k_l'] and c['visible_rows_last_layer'] == want[-1]
                    assert c['vs_independent_reference']['first_logits_bitwise'] is not None and c['vs_independent_reference']['tokens_exact']
                    assert g7['preadv_calls'] == 1 and g7['pread_calls'] == 0 and g7['returned_bytes'] == g7['file_bytes'] == expected
                    assert g7['vision_calls'] == 0 and g7['query_score_calls'] == 0 and g7['question_invariant_range']
            case['file_bytes_equal_expected'] = True
            gate['samples'].append(case)
            guarded_rmtree(path, base)                    # run-local gate stores only, after the receipt is recorded
            gate['samples'][-1]['stores_deleted_after_record'] = True
            print('VG4', i + 1, len(samples), 'PASS', s['label'], 'N', case['N'], 'k_last', case['k_last_layer'], flush=True)
        gate['status'] = 'PASS'
        gate['raw_visible_rows_equal_k_l_all_cases'] = all(c['G6_outputs'][f'P_PYR-beta-raw_{m}']['visible_rows_equal_k_l'] for c in gate['samples'] for m in ('scatter', 'directcopy'))
        gate['first_logits_bitwise_vs_independent_reference'] = {c['label']: {k: v['vs_independent_reference']['first_logits_bitwise'] for k, v in c['G6_outputs'].items()} for c in gate['samples']}
    except Exception:
        gate['status'] = 'FAIL'; gate['error'] = traceback.format_exc(); print(gate['error'], flush=True)
    finally:
        exp.close_contexts()
    publish(run, 'vg4_packed_path', G166._safe(gate))


# ------------------------------------------------------- vg5 / T1 checks
def new_rows(run, phase, method):
    """(image_id, turn) -> compact row of one arm of a phase of this run (commit SHA verified)."""
    manifest = read(run / 'llava' / phase / 'manifest.json')
    out = {}
    for d in manifest['dialogues']:
        rows, _ = REF.committed_raw(run / 'llava' / phase / phase / 'images' / d['image_id'])
        for r in rows:
            if r['method_id'] == method:
                out[(d['image_id'], r['turn_id'])] = REF.compact(r, full=True)
    return out, manifest


def stage_vg5(run):
    verify_freeze(run)
    receipt = read(run / 'llava/gate_u_pilot40/COMPLETE.json')
    assert receipt['status'] == 'PASS' and receipt['source_freeze_sha256'] == sha(run / 'experiment_freeze.json')
    new, manifest = new_rows(run, 'gate_u_pilot40', AD.GATE_METHOD)
    old = REF.load_index(run / 'reference/v3_pilot40_index.jsonl.gz')
    fields = ('suffix_input_ids', 'prompt', 'generated_token_ids', 'prediction', 'score', 'N_content', 'ssd_read_bytes')
    equal = {k: 0 for k in fields}; mism = []
    for (iid, turn), r in sorted(new.items()):
        o = old[iid][(REF.V3_B, turn)]
        diff = [k for k in fields if r[k] != o[k]]
        for k in fields:
            equal[k] += k not in diff
        if diff:
            mism.append({'image_id': iid, 'turn_id': turn, 'differs': diff, 'new': {k: r[k] for k in diff}, 'v3': {k: o[k] for k in diff}})
    n = len(new)
    ok = n == 120 and all(equal[k] == n for k in fields)
    publish(run, 'vg5_g_det_U', {'status': 'PASS' if ok else 'FAIL', 'method': AD.GATE_METHOD, 'reference': 'v3 pilot40 ours_prefixstore_kv25 [B] (file-per-layer path)',
                                 'requests': n, 'expected_requests': 120, 'images': len({k[0] for k in new}), 'equal': equal, 'mismatches': mism[:100], 'mismatch_count': len(mism),
                                 'input_token_ids': 'text-side input ids after the image span (suffix_input_ids) and the full prompt string; the image side is bound by the image SHA256',
                                 'meaning': 'the new execution path (164 packed bound by the new adapter) reproduces the v3 conditions request by request'})


def stage_t1(run, phase):
    verify_freeze(run)
    receipt = read(run / 'llava' / phase / 'COMPLETE.json')
    assert receipt['status'] == 'PASS' and receipt['source_freeze_sha256'] == sha(run / 'experiment_freeze.json')
    new, manifest = new_rows(run, phase, AD.MAIN_METHOD)
    old = REF.load_index(run / 'reference' / f'v3_{phase}_index.jsonl.gz')
    mism = []; equal = 0
    for d in manifest['dialogues']:
        r = new[(d['image_id'], 1)]
        refs = {m: old[d['image_id']][(m, 1)] for m in REF.V3_METHODS}
        same = all(r['generated_token_ids'] == o['generated_token_ids'] and r['prompt_sha256'] == o['prompt_sha256'] for o in refs.values())
        equal += same
        if not same:
            mism.append({'image_id': d['image_id'], 'new': r['generated_token_ids'], 'v3_B': refs[REF.V3_B]['generated_token_ids']})
    n = len(manifest['dialogues'])
    publish(run, f'g_det_T1_{phase}', {'status': 'PASS' if equal == n and len(new) == 3 * n else 'FAIL', 'phase': phase, 'images': n, 'T1_token_exact': equal,
                                      'requests': len(new), 'mismatches': mism[:50], 'reference': f'v3 {phase} Turn-1 of every arm (identical across arms in v3)'})


def stage_summary(run):
    freeze_sha = sha(run / 'experiment_freeze.json')
    states = {}
    for g in GATES:
        p = run / 'gates' / f'{g}.json'
        r = read(p) if p.exists() else {}
        states[g] = r.get('status', 'NOT_RUN') if r.get('source_freeze_sha256') == freeze_sha else 'STALE_OR_NOT_RUN'
    ok = all(v == 'PASS' for v in states.values())
    path = run / 'gates/COMPLETE.json'
    if path.exists():
        os.replace(path, run / f'gates/COMPLETE.superseded_{time.time_ns()}.json')
    dump(path, {'status': 'PASS' if ok else 'FAIL', 'gates': states, 'source_freeze_sha256': freeze_sha, 'recorded_unix': time.time(),
                'superseded_or_failed_receipts': sorted(p.name for p in (run / 'gates').glob('*.superseded_*.json'))})
    print(json.dumps({'gates': states, 'status': 'PASS' if ok else 'FAIL'}), flush=True)
    if not ok:
        raise SystemExit('gates not all PASS')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--stage', choices=('vg1', 'vg2', 'vg3', 'vg4', 'vg5', 'vg6', 't1', 'summary'), required=True)
    ap.add_argument('--phase', choices=('smoke', 'pilot40'))
    a = ap.parse_args(); run = a.run.resolve()
    assert run.parent == ROOT / 'runs' and run.name.startswith(AD.RUN_PREFIX)
    if a.stage == 't1':
        stage_t1(run, a.phase)
    else:
        {'vg1': stage_vg1, 'vg2': stage_vg2, 'vg3': stage_vg3, 'vg4': stage_vg4, 'vg5': stage_vg5, 'vg6': stage_vg6, 'summary': stage_summary}[a.stage](run)

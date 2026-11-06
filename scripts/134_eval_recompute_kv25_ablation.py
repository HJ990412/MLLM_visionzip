#!/usr/bin/env python3
"""Full-context KV25 vs reduced-token recompute KV25 ablation (LLaVA, Stage 1).

Contract: docs/recompute_kv25_ablation_contract.md.  Reuses the frozen
prefix25-persistence experiment (scripts/106) for Turn-1 inference, store
activation and cache-hit serving; only the reduced prefill, its persistence
and the RoPE text-position override are new (mmimpress/recompute_kv25.py).

Stages: --freeze, --protect before|after, --gpu correctness,smoke[,gqa].
"""
from __future__ import annotations
import argparse, gc, hashlib, importlib.util, json, math, os, random, shutil
import subprocess, sys, time, traceback, uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from mmimpress.prefix25 import sha, seal_integrity, process_io
from mmimpress.dataset import exact_score
from mmimpress.model import cache_layers
from mmimpress.cvpr25 import anyres_token_scores, visionzip_repack_order
from mmimpress.recompute_kv25 import (
    EmbeddingCapture, PositionRecorder, PositionServer, cache_sentinel_check,
    compare_stores, default_inv_freq, guarded_rmtree, layer0_oracle, payload_sizes,
    reduced_plan, reduced_prefill, selection_digest, sentinel_rows,
    sha_manifest, stat_inventory, structural_identity, structural_rows, synthetic_layers,
    tensor_sha256, text_position_start, truncation_control)

REF_RUN = ROOT / 'runs/prefix25_persistence_20261001T050113Z'
REF_GQA_SHA = '40e23fa7aeb22495717fc7eede1227de3a6b4811fde022d9cda71ef73c5e3895'
INDEX_SHA = '514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a'
CONTRACT = ROOT / 'docs/recompute_kv25_ablation_contract.md'
# Stopped rev.-2 run (G2 fail at n51002): kept untouched as a record, hashed by G5.
PREV_RUN = ROOT / 'runs/recompute_kv25_ablation_20261102T045423Z'
ARMS = ['recompute', 'fullload', 'ours', 'orig', 'compact', 'posshift']
KV25 = ['ours', 'orig', 'compact', 'posshift']
BUILT = ['ours', 'orig', 'compact']
POLICY = {'ours': 'original', 'orig': 'original', 'compact': 'compact', 'posshift': 'compact'}
STORE_OF = {'fullload': 'fullload', 'ours': 'ours', 'orig': 'orig', 'compact': 'compact', 'posshift': 'ours'}
KV_CONTEXT = {'ours': 'full_turn1', 'orig': 'reduced_prefill', 'compact': 'reduced_prefill'}
MIN_FREE = 20 * 2**30
SEED = 1234
NEW_SOURCES = ['mmimpress/recompute_kv25.py', 'scripts/134_eval_recompute_kv25_ablation.py',
               'scripts/135_report_recompute_kv25_ablation.py', 'tests/test_recompute_kv25.py',
               'docs/recompute_kv25_ablation_contract.md']


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / file)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


P = load('prefix25_persistence_existing', '106_prefix25_persistence.py')
write, append, exclusive = P.write, P.append, P.exclusive


def capacity():
    free = shutil.disk_usage(ROOT).free
    if free < MIN_FREE:
        raise RuntimeError(f'disk floor: {free/2**30:.2f} GiB free < 20 GiB')
    return free


# ------------------------------------------------------------------ freeze
def freeze(run):
    res = ROOT / 'results' / run.name
    run.mkdir(parents=True, exist_ok=False); res.mkdir(parents=True, exist_ok=False)
    assert sha(REF_RUN / 'gqa_manifest.json') == REF_GQA_SHA
    assert sha(ROOT / 'data/index.json') == INDEX_SHA
    gqa = json.loads((REF_RUN / 'gqa_manifest.json').read_text())
    rows = gqa['images']
    assert len(rows) == 40 and len({r['image_id'] for r in rows}) == 40
    for r in rows:
        assert sha(r['image_path']) == r['image_sha256'] and len(r['turns']) == 6
    index = json.loads((ROOT / 'data/index.json').read_text())
    assert [[str(q['question_id']) for q in x['questions'][4:10]] for x in index[:40]] == \
        [[str(t['question_id']) for t in x['turns']] for x in rows]
    shutil.copy2(REF_RUN / 'gqa_manifest.json', run / 'gqa_manifest.json')
    smoke = json.loads(json.dumps(gqa)); smoke['images'] = smoke['images'][:3]
    for r in smoke['images']:
        r['turns'] = r['turns'][:3]
    write(run / 'smoke_manifest.json', smoke)
    shutil.copy2(REF_RUN / 'correctness_manifest.json', run / 'correctness_manifest.json')
    revision = (Path.home() / '.cache/huggingface/hub/models--llava-hf--llava-v1.6-vicuna-7b-hf/refs/main').read_text().strip()
    assert revision == 'c916e6cdcd760b4cecd1dd4907f84ac649f93b23'
    config = {'seed': SEED, 'greedy': True, 'max_new_tokens': 16, 'chunk_size': 64, 'ratio': .25,
              'arms': ARMS, 'position_policy': POLICY, 'kv_context': KV_CONTEXT,
              'llava': {'revision': revision, 'quantization': 'NF4 double BF16 compute',
                        'attention': 'eager', 'store_dtype': 'float16'},
              'reference_run': str(REF_RUN), 'reference_gqa_manifest_sha256': REF_GQA_SHA,
              'index_sha256': INDEX_SHA, 'min_free_bytes': MIN_FREE,
              'bootstrap': {'unit': 'image', 'resamples': 10000, 'seed': SEED},
              'workload_files': {n: sha(run / f'{n}_manifest.json') for n in ['gqa', 'smoke', 'correctness']},
              'contract_sha256': sha(CONTRACT)}
    write(run / 'config.json', config); write(res / 'config.json', config)
    shutil.copy2(CONTRACT, run / 'CONTRACT.md'); shutil.copy2(CONTRACT, res / 'CONTRACT.md')
    write(run / 'environment.json', {
        'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout,
        'packages': subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout})
    # The report script (135) only reads raw outputs and is deliberately outside the GPU freeze.
    paths = [p for d in ['mmimpress', 'scripts', 'tests'] for p in (ROOT / d).rglob('*.py')
             if '__pycache__' not in p.parts and p.name != '135_report_recompute_kv25_ablation.py']
    paths += [CONTRACT] + [run / f'{n}_manifest.json' for n in ['gqa', 'smoke', 'correctness']] + [run / 'config.json']
    write(run / 'gpu_freeze.json', {'files': {str(p.relative_to(ROOT)): sha(p) for p in paths}})
    print('FROZEN', run, flush=True)


# -------------------------------------------------------------- protection
def protect(run, stage):
    excl = [f'results/{run.name}'] + NEW_SOURCES
    roots = [ROOT / x for x in ['results', 'docs', 'mmimpress', 'scripts', 'tests', 'README.md']] + [REF_RUN]
    if run != PREV_RUN:
        roots.append(PREV_RUN)
    started = time.time()
    shas = sha_manifest(roots, ROOT, exclude=excl)
    shas = {k: v for k, v in shas.items() if '__pycache__' not in k}
    kv = stat_inventory(sorted(ROOT.glob('kvstore*')), ROOT)
    doc = {'stage': stage, 'utc': time.time(), 'elapsed_s': time.time() - started,
           'excluded_new_paths': excl, 'sha256_files': len(shas), 'sha256_bytes': sum(v['size'] for v in shas.values()),
           'kvstore_inventory_files': len(kv), 'sha256': shas, 'kvstore_inventory': kv}
    out = run / f'protection_{stage}.json'
    with out.open('x') as f:
        json.dump(doc, f); f.write('\n')
    summary = {k: doc[k] for k in ['stage', 'elapsed_s', 'sha256_files', 'sha256_bytes', 'kvstore_inventory_files']}
    if stage == 'after':
        before = json.loads((run / 'protection_before.json').read_text())
        diff = {'sha_changed': sorted(k for k in before['sha256'] if k in shas and shas[k] != before['sha256'][k]),
                'sha_missing': sorted(set(before['sha256']) - set(shas)),
                'sha_added': sorted(set(shas) - set(before['sha256'])),
                'kv_changed': sorted(k for k in before['kvstore_inventory'] if k in kv and kv[k] != before['kvstore_inventory'][k]),
                'kv_missing': sorted(set(before['kvstore_inventory']) - set(kv)),
                'kv_added': sorted(set(kv) - set(before['kvstore_inventory']))}
        # README gets one appended section after the run; it is reported, not hidden.
        hard = {k: [x for x in v if x != 'README.md'] for k, v in diff.items()}
        summary.update({'status': 'PASS' if not any(hard.values()) else 'FAIL', 'differences': diff})
        write(run / 'protection_result.json', summary)
        write(ROOT / 'results' / run.name / 'protection_result.json', summary)
    print(json.dumps(summary, indent=1)[:4000], flush=True)


# ------------------------------------------------------------- experiment
class Ablation(P.Experiment):
    def __init__(self, run):
        super().__init__(run, 'llava')
        self.server = PositionServer(self.runner, max_new_tokens=16)
        tc = self.runner.cfg.text_config
        assert tc.rope_scaling is None
        self.inv_freq = self.runner.model.model.language_model.rotary_emb.inv_freq.detach().double().cpu()
        assert torch.allclose(self.inv_freq, default_inv_freq(self.runner.head_dim, tc.rope_theta), rtol=1e-6)
        self.stores = self.base / 'stores'
        self.reference = {}
        for r in map(json.loads, (REF_RUN / 'llava/gqa/raw_requests.jsonl').read_text().splitlines()):
            if r.get('status') == 'OK':
                self.reference[(r['arm'], r['image_id'], str(r['question_id']))] = r

    # ---------------------------------------------------------- Turn 1
    def t1(self, image, turn, arm, phase):
        if arm in BUILT:
            emb = EmbeddingCapture(self.runner)
            with emb:
                value, capture = self.pixel(image, turn, [], 'B', phase)
            capture['embeds'] = emb.embeds
            value['inputs_embeds_sha256'] = tensor_sha256(emb.embeds)
            value['language_model_calls'] = emb.calls
        else:
            value, capture = self.pixel(image, turn, [], arm, phase)
        return value, capture

    # ----------------------------------------------------- persistence
    def persist_arm(self, capture, image, arm, path, keep_synthetic=False):
        from mmimpress.piggyback import persist_captured_visual_prefix, persist_captured_raster_prefix
        capacity(); before = process_io(); started = time.perf_counter()
        enc = capture['enc']
        vs, vn = self.runner.visual_span(enc['input_ids'])
        base, _, _, sep = self.runner.anyres_layout(enc['image_sizes'][0], vn)
        common = dict(image_id=image['image_id'], model_id=self.runner.model_id,
                      image_input_sha256=image['image_sha256'])
        recompute, synthetic, plan = None, None, None
        if arm == 'fullload':
            receipt = persist_captured_raster_prefix(
                self.runner, capture['cache'], enc['input_ids'], enc['image_sizes'][0],
                capture['hidden'].result_cpu(), path, hidden_capture_stats=capture['hidden'],
                extra_metadata={'checkpoint_revision': self.runner.cfg._commit_hash, 'ablation': 'recompute_kv25'}, **common)
        else:
            policy = POLICY[arm]
            cache = capture['cache']
            if arm in ('orig', 'compact'):
                t0 = time.perf_counter()
                scores = anyres_token_scores(self.runner, capture['vision'].result_cpu(), enc['image_sizes'][0], vn, base_side=base)
                order = visionzip_repack_order(scores, sep)
                plan = reduced_plan(vs, vn, order, sep, policy)
                plan_ms = (time.perf_counter() - t0) * 1e3
                reduced, rec = reduced_prefill(self.runner, capture['embeds'], plan)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                cache = synthetic_layers(reduced, plan, vs + vn)
                torch.cuda.synchronize(); assembly_ms = (time.perf_counter() - t0) * 1e3
                del reduced
                recompute = {**rec, 'plan_ms': plan_ms, 'synthetic_assembly_ms': assembly_ms,
                             'prefill_only_ms': rec['embed_gather_ms'] + rec['reduced_prefill_ms'],
                             'k': plan['k'], 'n_sep': plan['n_sep'],
                             'position_ids_sha256': hashlib.sha256(json.dumps(plan['position_ids']).encode()).hexdigest(),
                             'position_ids_first_visual': plan['position_ids'][vs] if len(plan['position_ids']) > vs else None,
                             'position_ids_last': plan['position_ids'][-1],
                             'position_ids_contiguous': plan['position_ids'] == list(range(len(plan['position_ids']))),
                             'position_ids_original': plan['position_ids'] == plan['input_index']}
                assert recompute['position_ids_original' if policy == 'original' else 'position_ids_contiguous']
                if keep_synthetic:
                    synthetic = cache
            meta_extra = {'checkpoint_revision': self.runner.cfg._commit_hash, 'ablation': 'recompute_kv25',
                          'kv_context': KV_CONTEXT[arm], 'position_policy': policy,
                          'text_position_start': text_position_start(
                              {'v_token_start': vs, 'v_token_num': vn, 'newline_idx': sep}, policy)}
            receipt = persist_captured_visual_prefix(
                self.runner, cache, enc['input_ids'], enc['image_sizes'][0], capture['vision'].result_cpu(),
                path, capture_stats=capture['vision'], storage_policy='prefix25', extra_metadata=meta_extra, **common)
            del cache
        hash_t0 = time.perf_counter(); files = seal_integrity(path); hash_ms = (time.perf_counter() - hash_t0) * 1e3
        total_ms = (time.perf_counter() - started) * 1e3; after = process_io()
        # G2 rev. 3 truncation control: after every persistence timer and I/O counter is closed,
        # before activation and any hit, so it never enters persistence or TTFT measurements.
        control = truncation_control(self.runner, capture['embeds'], plan) if plan else None
        meta = receipt['meta']
        out = {'physical_write_id': str(uuid.uuid4()), 'store_path': str(path), 'arm': arm, 'image_id': image['image_id'],
               'metadata': meta, 'files': files, 'timing_ms': receipt['timing_ms'], 'hash_and_seal_ms': hash_ms,
               'persistence_ms': total_ms, 'recompute': recompute,
               'os_write_bytes': after['wchar'] - before['wchar'], 'os_write_calls': after['syscw'] - before['syscw'],
               'payload_sizes': payload_sizes(path), 'file_logical_bytes': sum(x['size'] for x in files.values()),
               'selection_digest': selection_digest(meta) if arm != 'fullload' else None,
               'saliency_ms': capture['vision'].stats().get('saliency_total_instrumentation_ms') if arm != 'fullload' else None,
               'truncation_control': None if control is None else {
                   'length': control['length'], 'structural_rows': len(control['rows']),
                   'control_ms': control['control_ms'], 'outside_timers': True,
                   'computed_after': 'persistence_ms, timing_ms, hash_and_seal_ms and os_write counters were closed',
                   'precedes': 'store activation and every hit request'}}
        return out, synthetic, control

    def activate_arm(self, path, image, arm):
        ctx, act = self.activate(path, image, 'fullload' if arm == 'fullload' else 'B')
        if arm != 'fullload':
            m = ctx.meta
            assert m['ablation'] == 'recompute_kv25' and m['kv_context'] == KV_CONTEXT[arm]
            assert m['position_policy'] == POLICY[arm]
            assert m['text_position_start'] == text_position_start(m, POLICY[arm])
        return ctx, act

    # ------------------------------------------------------------ hits
    def text_start(self, ctx, arm):
        if arm not in KV25 or POLICY[arm] == 'original':
            return None
        m = ctx.meta
        if arm == 'posshift':
            assert m['kv_context'] == 'full_turn1' and m['position_policy'] == 'original'
        else:
            assert m['kv_context'] == 'reduced_prefill' and m['position_policy'] == 'compact'
        ts = text_position_start(m, 'compact')
        assert ts == int(m['v_token_start']) + int(m['stored_content_count']) + len(m['newline_idx'])
        return ts

    def hit_arm(self, ctx, path, image, turn, arm, phase, logits=False, sentinel=False, force_start='policy'):
        ts = self.text_start(ctx, arm) if force_start == 'policy' else force_start
        sent = []
        self.server.text_position_start = ts
        self.server.pre_decode = (lambda cache, p: sent.append(cache_sentinel_check(cache, p))) if sentinel else None
        rec = PositionRecorder(self.runner.model)
        try:
            with rec:
                value = self.hit(ctx, path, image, turn, [], 'fullload' if arm == 'fullload' else 'B', phase, logits)
        finally:
            self.server.text_position_start = None; self.server.pre_decode = None
        value['position_check'] = rec.verify(int(ctx.meta['prefix_len']), ts)
        if sentinel:
            assert len(sent) == 1
            value['sentinel_check'] = sent[0]
        return value

    def trace_positions(self, ctx, question, layers, rank, k, ts):
        """90._run_with_trace with the text RoPE start ``ts`` instead of physical positions."""
        ref = self.ref
        box = {'logits': [], 'position_checks': []}
        prefix_len = int(ctx.meta['prefix_len'])
        original_decode = self.server._decode

        def output_hook(module, args, output):
            if hasattr(output, 'logits'):
                box['logits'].append(output.logits[0, -1].detach().float().cpu())

        rec = PositionRecorder(self.runner.model)

        def traced(cache, suffix, p):
            box['cache'] = ref._compare_installed(ctx.meta, cache, layers, rank, k)
            h = self.runner.model.register_forward_hook(output_hook)
            try:
                with ref._AttentionTrace(ctx.meta, k) as masks, rec:
                    out = original_decode(cache, suffix, p)
                box['attention_masks'] = masks.summary()
                return out
            finally:
                h.remove()

        self.server._decode = traced
        self.server.text_position_start = ts
        try:
            with ref._ReadTrace() as reads:
                result = self.server.request_cvpr25(ctx, question=question, static=None, mode='prefix', budget=.25,
                                                    budget_unit='visual_kv', sep_policy='sidecar', cold=False,
                                                    expected_prefix_layout='visionzip_image_only')
        finally:
            del self.server._decode
            self.server.text_position_start = None
            from mmimpress.serve import BIAS
            BIAS.clear()
        box['position_checks'] = rec.verify(prefix_len, ts)
        box['reads'] = ref._check_reads(ctx, reads, math.ceil(k / 64))
        box['result'] = result
        assert len(box['logits']) == len(result['generated_token_ids'])
        return box

    def memory_reference(self, ctx, question, layers, rank, k, ts):
        self.server.text_position_start = ts
        try:
            return self.ref._memory_reference(self.server, ctx, question, layers, rank, k)
        finally:
            self.server.text_position_start = None

    # ---------------------------------------------------- image gates
    def oracle(self, capture):
        """FP32 layer-0 oracle from the Ours T1 embeddings; outside every timer."""
        vs, vn = self.runner.visual_span(capture['enc']['input_ids'])
        return layer0_oracle(self.runner, capture['embeds'], vs + vn, self.inv_freq)

    def image_gates(self, image, t1, receipts, phase, oracle, controls):
        g = {'image_id': image['image_id'], 'phase': phase}
        digests = {a: receipts[a]['selection_digest'] for a in BUILT}
        sizes = {a: receipts[a]['payload_sizes'] for a in BUILT}
        embeds = {a: t1[a]['inputs_embeds_sha256'] for a in BUILT}
        outs = {a: (t1[a]['first_token_id'], t1[a]['generated_token_ids'], t1[a]['prediction']) for a in BUILT}
        sentinel = {a: sentinel_rows(receipts[a]['store_path']) for a in BUILT}
        g.update({'selection_digest': digests, 'payload_sizes_equal': len({json.dumps(s, sort_keys=True) for s in sizes.values()}) == 1,
                  'inputs_embeds_sha256': embeds, 't1_outputs': outs, 'sentinel_rows': sentinel,
                  'fullload_t1_agrees': (t1['fullload']['first_token_id'], t1['fullload']['generated_token_ids']) == outs['ours'][:2],
                  'recompute_t1_agrees': (t1['recompute']['first_token_id'], t1['recompute']['generated_token_ids']) == outs['ours'][:2]})
        g1 = (len(set(digests.values())) == 1 and g['payload_sizes_equal'] and len(set(embeds.values())) == 1
              and len(set(map(json.dumps, outs.values()))) == 1 and not any(sentinel.values()))
        g['G1'] = 'PASS' if g1 else 'FAIL'
        g['G2'] = {'orig': compare_stores(receipts['ours']['store_path'], receipts['orig']['store_path'], 'original',
                                          self.inv_freq, oracle, controls['orig']),
                   'compact': compare_stores(receipts['ours']['store_path'], receipts['compact']['store_path'], 'compact',
                                             self.inv_freq, oracle, controls['compact'])}
        g['G2_status'] = 'PASS' if all(x['status'] == 'PASS' for x in g['G2'].values()) else 'FAIL'
        g['G3_persistence'] = {a: {key: receipts[a]['recompute'][key] for key in
                                   ['position_ids_original', 'position_ids_contiguous', 'position_ids_last', 'observed_position_ids_match']}
                               for a in ('orig', 'compact')}
        append(self.base / 'gates.jsonl', g)
        if g['G1'] != 'PASS' or g['G2_status'] != 'PASS':
            raise AssertionError(f"image gate failure {image['image_id']}: G1={g['G1']} G2="
                                 f"{ {a: x['failed_gates'] for a, x in g['G2'].items()} }")
        return g

    # -------------------------------------------------------- correctness
    def correctness(self):
        samples = json.loads((self.run / 'correctness_manifest.json').read_text())['samples']
        samples = [samples[i] for i in [0, 3, 4, 5, 6]]
        ref_out = {s['image_id']: s['fresh_B_output'] for s in
                   json.loads((REF_RUN / 'llava/final_gpu_regression.json').read_text())['samples']}
        rows = []; gate = {'status': 'NOT RUN', 'samples': rows}
        for i, sample in enumerate(samples):
            path = self.base / 'correctness' / f'{i:02d}_{sample["image_id"]}'
            path.mkdir(parents=True, exist_ok=False)
            try:
                rows.append(self.correctness_case(sample, path, ref_out[sample['image_id']]))
                print('G4', i + 1, len(samples), 'PASS', flush=True)
            except Exception:
                rows.append({'image_id': sample['image_id'], 'status': 'FAIL', 'error': traceback.format_exc()})
                gate['status'] = 'FAIL'; write(self.base / 'gpu_correctness.json', gate); raise
            write(self.base / 'gpu_correctness.json', gate)
        gate['status'] = 'PASS'; write(self.base / 'gpu_correctness.json', gate)

    def noise_floor(self, cap, full_layers, vs, vn, selected, recs):
        """Recorded, not gated: same-context shape controls vs the Orig context effect.

        ``prefix_only`` reruns [system | full image] without the question (same
        context for every prefix row, ~30 fewer tokens); ``orig_reduced`` is the
        actual reduced prefill; ``truncated_same_len``/``truncated_half_len`` cut
        the original sequence to the reduced length / half of it, so structural
        rows keep their context and only the shape changes.  All are compared
        with the T1 cache in BF16 (sys rows; common-prefix rows, max over layers
        >= 1; S rows where present).
        """
        from mmimpress.recompute_kv25 import diff_stats
        rows = [vs + i for i in selected]
        plan_orig = reduced_plan(vs, vn, recs['ours']['metadata']['full_importance_permutation'],
                                 recs['ours']['metadata']['newline_idx'], 'original')
        common = structural_rows(plan_orig)[vs:]
        n = len(plan_orig['input_index'])
        result = {'reduced_len': n, 'common_prefix_rows': len(common)}
        for name, idx in (('prefix_only', list(range(vs + vn))), ('orig_reduced', plan_orig['input_index']),
                          ('truncated_same_len', list(range(n))), ('truncated_half_len', list(range(n // 2)))):
            if vs + len(common) > len(idx):
                result[name] = None
                continue
            layers, _ = reduced_prefill(self.runner, cap['embeds'],
                                        {'policy': 'original', 'input_index': idx, 'position_ids': idx})
            where = {r: j for j, r in enumerate(idx)}

            def rel(li, kind, absrows):
                a = layers[li][kind][0][:, [where[r] for r in absrows]].float().cpu()
                b = full_layers[li][kind][0][:, absrows].float().cpu()
                return diff_stats(a, b)['rel_l2']
            deep = range(1, len(layers))
            entry = {'sys_max_k': max(rel(li, 0, list(range(vs))) for li in range(len(layers))),
                     'sys_max_v': max(rel(li, 1, list(range(vs))) for li in range(len(layers))),
                     'common_deep_max_k': max(rel(li, 0, common) for li in deep) if common else None,
                     'common_deep_max_v': max(rel(li, 1, common) for li in deep) if common else None}
            if all(r in where for r in rows):
                entry.update({'last_S_k': rel(-1, 0, rows), 'last_S_v': rel(-1, 1, rows),
                              'layer0_S_k': rel(0, 0, rows), 'layer0_S_v': rel(0, 1, rows)})
            result[name] = entry
            del layers
        p, o, t = result['prefix_only'], result['orig_reduced'], result['truncated_same_len']
        result['context_over_floor'] = {'k': o['last_S_k'] / p['last_S_k'] if p['last_S_k'] else None,
                                        'v': o['last_S_v'] / p['last_S_v'] if p['last_S_v'] else None}
        result['context_over_single_row'] = {
            kind: o[f'last_S_{kind}'] / t[f'common_deep_max_{kind}'] if t[f'common_deep_max_{kind}'] else None
            for kind in ('k', 'v')}
        torch.cuda.empty_cache()
        return result

    def correctness_case(self, sample, path, ref_out):
        turn = {'question': sample['question'], 'question_id': sample['question_id']}
        value, cap = self.t1(sample, turn, 'ours', 'gqa')
        enc = cap['enc']; vs, vn = self.runner.visual_span(enc['input_ids'])
        _, _, _, sep = self.runner.anyres_layout(enc['image_sizes'][0], vn)
        n = vn - len(sep); k = (n + 3) // 4
        rank = self.ref._score_rank(self.runner, cap['vision'].result_cpu(), enc['image_sizes'][0], vn, sep)
        recs, synth, controls = {}, {}, {}
        for arm in BUILT:
            recs[arm], synth[arm], controls[arm] = self.persist_arm(cap, sample, arm, path / arm, keep_synthetic=True)
            assert recs[arm]['metadata']['full_importance_permutation'] == rank + sep
        # G2 rev. 3 in memory (BF16, before the FP16 store) for the G4 pairs
        memory_identity = {a: structural_identity(synth[a], controls[a]) for a in ('orig', 'compact')}
        assert all(x['ok'] for x in memory_identity.values()), memory_identity
        full_layers = cache_layers(cap['cache'])
        out_floor = self.noise_floor(cap, full_layers, vs, vn, rank[:k], recs)
        t1 = {a: value for a in BUILT}
        fake = {'fullload': value, 'recompute': value}
        gates = self.image_gates(sample, {**t1, **fake}, recs, 'correctness', self.oracle(cap), controls)
        out = {'image_id': sample['image_id'], 'question_id': sample['question_id'], 'N': n, 'k': k,
               'G1': gates['G1'], 'G2': gates['G2_status'], 'noise_floor': out_floor,
               'structural_identity_bf16_memory': memory_identity, 'checks': {}}
        ctx = {a: self.activate_arm(path / a, sample, a)[0] for a in BUILT}
        # (a) reproduction of the reference run's recorded fresh [B] output
        ours = self.hit_arm(ctx['ours'], path / 'ours', sample, turn, 'ours', 'gqa', logits=True)
        proof = self.output_proof(ours)
        assert [proof[x] for x in ['first_token_id', 'generated_token_ids', 'prediction']] == \
            [ref_out[x] for x in ['first_token_id', 'generated_token_ids', 'prediction']], (proof, ref_out)
        out['checks']['a_reference_reproduced'] = {'tokens_and_prediction': True,
                                                   'logits_sha256_equal': proof['logits_sha256'] == ref_out['logits_sha256']}
        # (b) explicit physical start through the override equals the base path bitwise
        same = self.hit_arm(ctx['ours'], path / 'ours', sample, turn, 'ours', 'gqa', logits=True,
                            force_start=int(ctx['ours'].meta['prefix_len']))
        assert len(same['all_logits']) == len(ours['all_logits']) and all(
            torch.equal(x, y) for x, y in zip(same['all_logits'], ours['all_logits']))
        out['checks']['b_override_equals_base'] = True
        # (c) Orig: independent in-memory reference from the reduced KV, original positions
        ref = self.memory_reference(ctx['orig'], sample['question'], synth['orig'], rank, k, None)
        prod = self.ref._run_with_trace(self.server, ctx['orig'], sample['question'], synth['orig'], rank, k,
                                        budget=.25, budget_unit='visual_kv')
        out['checks']['c_orig'] = {'comparison': self.ref._compare_outputs(ref, dict(prod['result'], logits=prod['logits'])),
                                   'cache': prod['cache'], 'masks': prod['attention_masks'], 'positions': prod['position_checks'][:2],
                                   'reads': {x: prod['reads'][x] for x in ['actual_returned_bytes', 'actual_pread_calls']}}
        # (d) Compact and PosShift: same comparison with the compact text start on both sides
        for arm, layers in (('compact', synth['compact']), ('posshift', full_layers)):
            c = ctx[STORE_OF[arm]]; ts = self.text_start(c, arm)
            ref = self.memory_reference(c, sample['question'], layers, rank, k, ts)
            prod = self.trace_positions(c, sample['question'], layers, rank, k, ts)
            out['checks'][f'd_{arm}'] = {'text_position_start': ts,
                                         'comparison': self.ref._compare_outputs(ref, dict(prod['result'], logits=prod['logits'])),
                                         'cache': prod['cache'], 'masks': prod['attention_masks'],
                                         'positions': prod['position_checks'],
                                         'reads': {x: prod['reads'][x] for x in ['actual_returned_bytes', 'actual_pread_calls']}}
        # (e) a changed question leaves selection unchanged; zero vision/query scoring (asserted in hit)
        other = {'question': 'What is visible?', 'question_id': 'probe'}
        sel = {}
        for arm in KV25:
            a = self.hit_arm(ctx[STORE_OF[arm]], path / STORE_OF[arm], sample, turn, arm, 'gqa', sentinel=True)
            b = self.hit_arm(ctx[STORE_OF[arm]], path / STORE_OF[arm], sample, other, arm, 'gqa', sentinel=True)
            assert a['selected_original_ids'] == b['selected_original_ids'] == rank[:k]
            sel[arm] = {'selected_unchanged': True, 'read_bytes': a['actual_io']['bytes'], 'preads': a['actual_io']['preads'],
                        'text_position_start': a['position_check']['text_position_start'], 'sentinel': a['sentinel_check']}
        assert len({(v['read_bytes'], v['preads']) for v in sel.values()}) == 1
        out['checks']['e_question_invariance_io_sentinel'] = sel
        for c in ctx.values():
            c.close()
        del cap, synth, full_layers, ctx, value, ours, same, prod, ref
        self.close_contexts()
        out['status'] = 'PASS'
        return out

    # ---------------------------------------------------------------- pilot
    def pilot(self, phase):
        manifest = json.loads((self.run / f'{phase}_manifest.json').read_text())
        out = self.base / phase; out.mkdir(exist_ok=False)
        raw = out / 'raw_requests.jsonl'; persist_file = out / 'persistence.jsonl'
        for ix, image in enumerate(manifest['images']):
            exclusive(); capacity()
            contexts, receipts, paths, t1, hits, controls = {}, {}, {}, {}, {}, {}
            for ti, turn in enumerate(image['turns']):
                order = ARMS[(ix + ti) % 6:] + ARMS[:(ix + ti) % 6]
                for arm in order:
                    if ti == 0 and arm == 'posshift':
                        continue
                    exclusive()
                    logical = f'llava/{phase}/{image["image_id"]}/{arm}/{ti + 1}'
                    try:
                        if ti == 0 or arm == 'recompute':
                            value, capture = self.t1(image, turn, arm if ti == 0 else 'recompute', phase)
                            if ti == 0 and arm == 'ours':
                                oracle = self.oracle(capture)
                            if ti == 0 and arm != 'recompute':
                                path = self.stores / phase / image['image_id'] / arm
                                receipt, _, control = self.persist_arm(capture, image, arm, path)
                                if control is not None:
                                    controls[arm] = control
                                del capture; gc.collect(); torch.cuda.empty_cache()
                                ctx, activation = self.activate_arm(path, image, arm)
                                receipt['activation'] = activation; receipt['phase'] = phase
                                contexts[arm], receipts[arm], paths[arm] = ctx, receipt, path
                                append(persist_file, receipt)
                            else:
                                del capture
                            if ti == 0:
                                t1[arm] = value
                            value['actual_io'] = {'bytes': 0, 'preads': 0, 'calls': []}
                        else:
                            store = STORE_OF[arm]
                            value = self.hit_arm(contexts[store], paths[store], image, turn, arm, phase,
                                                 sentinel=(phase == 'smoke' and arm in KV25))
                            hits[(arm, ti)] = value
                        row = self.row(logical, phase, image, arm, ti, turn, value)
                        append(raw, row)
                    except Exception:
                        append(raw, {'logical_request_id': logical, 'status': 'FAIL', 'error': traceback.format_exc()})
                        raise
                if ti == 0:
                    shared = dict(t1['ours'], shared_t1_from='ours', executed=False)
                    append(raw, self.row(f'llava/{phase}/{image["image_id"]}/posshift/1', phase, image, 'posshift', 0,
                                         turn, shared))
                    self.image_gates(image, t1, receipts, phase, oracle, controls)
                    del oracle; controls.clear()
            for ti in range(1, len(image['turns'])):
                io = {(hits[(a, ti)]['actual_io']['bytes'], hits[(a, ti)]['actual_io']['preads']) for a in KV25}
                assert len(io) == 1, f'KV25 arms read different bytes: {io}'
            for ctx in contexts.values():
                ctx.close()
            contexts.clear(); self.close_contexts()
            keep = {'orig', 'compact'} if phase == 'smoke' else set()
            for arm, path in paths.items():
                if arm in keep:
                    continue
                append(self.base / 'cleanup.jsonl', {'stage': 'before', 'path': str(path), 'arm': arm,
                                                     'sealed_files': len(receipts[arm]['files'])})
                guarded_rmtree(path, self.stores)
                append(self.base / 'cleanup.jsonl', {'stage': 'after', 'path': str(path), 'exists': path.exists()})
            print('llava', phase, ix + 1, len(manifest['images']), 'complete', flush=True)

    def row(self, logical, phase, image, arm, ti, turn, value):
        keep = ['prediction', 'first_token_id', 'generated_token_ids', 'ttft_ms', 'request_e2e_ms', 'position_check',
                'sentinel_check', 'selected_original_ids', 'inputs_embeds_sha256', 'shared_t1_from', 'executed',
                'normal_kv_read_bytes', 'separator_read_bytes', 'k_target', 'N_content', 'prefill_ms', 'decode_ms',
                'prepare_ms', 'peak_gpu_allocated_bytes', 'vision_calls', 'online_query_score_calls']
        return {'logical_request_id': logical, 'execution_id': str(uuid.uuid4()), 'status': 'OK', 'phase': phase,
                'arm': arm, 'image_id': image['image_id'], 'turn': ti + 1, 'question_id': str(turn['question_id']),
                'question': turn['question'], 'gold': turn['gold'], 'cache_hit': ti > 0 and arm != 'recompute',
                'accuracy': exact_score(value['prediction'], [turn['gold']]),
                'read_bytes': value['actual_io']['bytes'], 'preads': value['actual_io']['preads'],
                **{k: value.get(k) for k in keep}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--protect', choices=['before', 'after'])
    ap.add_argument('--gpu', help='comma list of correctness,smoke,gqa')
    args = ap.parse_args(); run = Path(args.run).resolve()
    assert run.parent == ROOT / 'runs' and run.name.startswith('recompute_kv25_ablation_')
    if args.freeze:
        return freeze(run)
    if args.protect:
        return protect(run, args.protect)
    frozen = json.loads((run / 'gpu_freeze.json').read_text())
    for rel, digest in frozen['files'].items():
        assert sha(ROOT / rel) == digest, f'source changed after freeze: {rel}'
    assert '\nOK\n' in (run / 'cpu_tests.log').read_text()
    assert (run / 'protection_before.json').exists()
    stages = args.gpu.split(',')
    assert set(stages) <= {'correctness', 'smoke', 'gqa'}
    exclusive(); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)
    status_path = run / 'llava' / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    exp = None
    try:
        exp = Ablation(run)
        if 'correctness' in stages:
            exp.correctness(); status['G4 GPU CORRECTNESS'] = 'PASS'; write(status_path, status)
        assert status.get('G4 GPU CORRECTNESS') == 'PASS', 'G4 must pass before pilots'
        sample = json.loads((run / 'smoke_manifest.json').read_text())['images'][0]
        for arm in ['recompute', 'fullload', 'ours']:
            val, cap = exp.t1(sample, sample['turns'][0], arm, 'smoke'); del val, cap
        write(exp.base / f'warmup_{"_".join(stages)}.json', {'excluded': True, 'normal_requests': 3})
        for phase in [s for s in stages if s != 'correctness']:
            if phase == 'gqa':
                assert status.get('SMOKE') == 'PASS', 'smoke must pass before Stage 1'
            status[phase.upper() if phase == 'smoke' else 'STAGE1 GQA'] = 'RUNNING'; write(status_path, status)
            exp.pilot(phase)
            status[phase.upper() if phase == 'smoke' else 'STAGE1 GQA'] = 'PASS'; write(status_path, status)
    except Exception:
        err = traceback.format_exc(); print(err, flush=True)
        append(run / 'failures.jsonl', {'error': err, 'utc': time.time(), 'stages': stages})
        for key, value in list(status.items()):
            if value == 'RUNNING':
                status[key] = 'FAIL'
        if 'correctness' in stages and status.get('G4 GPU CORRECTNESS') != 'PASS':
            status['G4 GPU CORRECTNESS'] = 'FAIL'
        raise
    finally:
        (run / 'llava').mkdir(exist_ok=True); write(status_path, status)
        if exp:
            exp.close_contexts()
        print(status, flush=True)


if __name__ == '__main__':
    main()

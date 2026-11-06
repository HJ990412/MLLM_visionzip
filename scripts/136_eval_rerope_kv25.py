#!/usr/bin/env python3
"""Ours-ReRoPE and Pruned-T1 (LLaVA, GQA Stage 1).

Contract: docs/rerope_kv25_contract.md.  Extends the frozen rev.-3 ablation
(scripts/134) with the full-context/contiguous-position store built by
re-rotation and the cache-less pruned first turn.  Only the new arms are new
code (mmimpress/rerope_kv25.py); every other path is the rev.-3 one.

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
from PIL import Image
from mmimpress.prefix25 import sha, seal_integrity, process_io
from mmimpress.dataset import exact_score
from mmimpress.model import cache_layers
from mmimpress.cvpr25 import anyres_token_scores, visionzip_repack_order
from mmimpress.recompute_kv25 import (
    bit_equal, guarded_rmtree, payload_sizes, reduced_plan, reduced_prefill, selection_digest,
    sentinel_rows, sha_manifest, stat_inventory, text_position_start)
from mmimpress.rerope_kv25 import (
    AbortAfterEmbeds, FirstLogitsRef, LayerPositions, compare_recorded, compare_rerope, pruned_generate,
    rerope_layers, top2_margin)


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / file)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


R3 = load('recompute_kv25_ablation_existing', '134_eval_recompute_kv25_ablation.py')
P = R3.P
write, append, exclusive, capacity = R3.write, R3.append, R3.exclusive, R3.capacity
REF_RUN = R3.REF_RUN
REV3_RUN = ROOT / 'runs/recompute_kv25_ablation_20261102T061243Z'
REV2_RUN = ROOT / 'runs/recompute_kv25_ablation_20261102T045423Z'
CONTRACT = ROOT / 'docs/rerope_kv25_contract.md'
EVIDENCE = Path('/tmp/claude-1000/-home-dblab-hj-mllm-v2/d4c13472-218c-434b-a623-f821b9709a83/scratchpad/rerope_dev')
EVIDENCE_SHA = '879706ccdd71510b0d90404012cf7a1d6b3659c796789cbaa82975d42a2d6eea'
ARMS = ['recompute', 'fullload', 'ours', 'orig', 'compact', 'rerope', 'pruned']
STORED = ['fullload', 'ours', 'orig', 'compact', 'rerope']
KV25 = ['ours', 'orig', 'compact', 'rerope']
BUILT = ['ours', 'orig', 'compact']             # own full-token T1 + persistence (rev. 3)
POLICY = {'ours': 'original', 'orig': 'original', 'compact': 'compact', 'rerope': 'compact'}
KV_CONTEXT = {'ours': 'full_turn1', 'orig': 'reduced_prefill', 'compact': 'reduced_prefill', 'rerope': 'full_turn1'}
SEED = 1234
NEW_SOURCES = ['mmimpress/rerope_kv25.py', 'scripts/136_eval_rerope_kv25.py', 'scripts/137_report_rerope_kv25.py',
               'tests/test_rerope_kv25.py', 'docs/rerope_kv25_contract.md']


# ------------------------------------------------------------------ freeze
def freeze(run):
    res = ROOT / 'results' / run.name
    run.mkdir(parents=True, exist_ok=False); res.mkdir(parents=True, exist_ok=False)
    assert sha(REV3_RUN / 'gqa_manifest.json') == R3.REF_GQA_SHA
    assert sha(ROOT / 'data/index.json') == R3.INDEX_SHA
    gqa = json.loads((REV3_RUN / 'gqa_manifest.json').read_text())
    rows = gqa['images']
    assert len(rows) == 40 and len({r['image_id'] for r in rows}) == 40
    for r in rows:
        assert sha(r['image_path']) == r['image_sha256'] and len(r['turns']) == 6
    index = json.loads((ROOT / 'data/index.json').read_text())
    assert [[str(q['question_id']) for q in x['questions'][4:10]] for x in index[:40]] == \
        [[str(t['question_id']) for t in x['turns']] for x in rows]
    for n in ['gqa', 'smoke', 'correctness']:
        shutil.copy2(REV3_RUN / f'{n}_manifest.json', run / f'{n}_manifest.json')
    assert sha(EVIDENCE / 'g_ab_evidence.json') == EVIDENCE_SHA
    (run / 'dev').mkdir(); (res / 'dev').mkdir()
    for d in (run, res):
        shutil.copy2(EVIDENCE / 'g_ab_evidence.json', d / 'dev' / 'g_ab_evidence.json')
        shutil.copy2(EVIDENCE / 'g_ab_evidence.py', d / 'dev' / 'g_ab_evidence.py')
    revision = (Path.home() / '.cache/huggingface/hub/models--llava-hf--llava-v1.6-vicuna-7b-hf/refs/main').read_text().strip()
    assert revision == 'c916e6cdcd760b4cecd1dd4907f84ac649f93b23'
    config = {'seed': SEED, 'greedy': True, 'max_new_tokens': 16, 'chunk_size': 64, 'ratio': .25,
              'arms': ARMS, 'position_policy': POLICY, 'kv_context': KV_CONTEXT,
              'llava': {'revision': revision, 'quantization': 'NF4 double BF16 compute', 'attention': 'eager', 'store_dtype': 'float16'},
              'reference_runs': {'rev3': str(REV3_RUN), 'rev2_stopped': str(REV2_RUN), 'prefix25': str(REF_RUN)},
              'reference_gqa_manifest_sha256': R3.REF_GQA_SHA, 'index_sha256': R3.INDEX_SHA, 'min_free_bytes': R3.MIN_FREE,
              'bootstrap': {'unit': 'image', 'resamples': 10000, 'seed': SEED, 'joint_contrasts': True},
              'evidence_sha256': EVIDENCE_SHA,
              'workload_files': {n: sha(run / f'{n}_manifest.json') for n in ['gqa', 'smoke', 'correctness']},
              'contract_sha256': sha(CONTRACT)}
    write(run / 'config.json', config); write(res / 'config.json', config)
    shutil.copy2(CONTRACT, run / 'CONTRACT.md'); shutil.copy2(CONTRACT, res / 'CONTRACT.md')
    write(run / 'environment.json', {
        'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout,
        'packages': subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout})
    paths = [p for d in ['mmimpress', 'scripts', 'tests'] for p in (ROOT / d).rglob('*.py')
             if '__pycache__' not in p.parts and p.name not in ('135_report_recompute_kv25_ablation.py', '137_report_rerope_kv25.py')]
    paths += [CONTRACT] + [run / f'{n}_manifest.json' for n in ['gqa', 'smoke', 'correctness']] + [run / 'config.json']
    write(run / 'gpu_freeze.json', {'files': {str(p.relative_to(ROOT)): sha(p) for p in paths}})
    print('FROZEN', run, flush=True)


# -------------------------------------------------------------- protection
def protect(run, stage):
    excl = [f'results/{run.name}'] + NEW_SOURCES
    roots = [ROOT / x for x in ['results', 'docs', 'mmimpress', 'scripts', 'tests', 'README.md']] + [REF_RUN, REV2_RUN, REV3_RUN]
    started = time.time()
    shas = sha_manifest(roots, ROOT, exclude=excl)
    shas = {k: v for k, v in shas.items() if '__pycache__' not in k}
    kv = stat_inventory(sorted(ROOT.glob('kvstore*')), ROOT)
    doc = {'stage': stage, 'utc': time.time(), 'elapsed_s': time.time() - started, 'excluded_new_paths': excl,
           'sha256_files': len(shas), 'sha256_bytes': sum(v['size'] for v in shas.values()),
           'kvstore_inventory_files': len(kv), 'sha256': shas, 'kvstore_inventory': kv}
    out = run / f'protection_{stage}.json'
    with out.open('x') as f:
        json.dump(doc, f); f.write('\n')
    summary = {k: doc[k] for k in ['stage', 'elapsed_s', 'sha256_files', 'sha256_bytes', 'kvstore_inventory_files']}
    if stage == 'after':
        before = json.loads((run / 'protection_before.json').read_text())
        diff = {'sha_changed': sorted(k for k in before['sha256'] if k in shas and shas[k] != before['sha256'][k]),
                'sha_missing': sorted(set(before['sha256']) - set(shas)), 'sha_added': sorted(set(shas) - set(before['sha256'])),
                'kv_changed': sorted(k for k in before['kvstore_inventory'] if k in kv and kv[k] != before['kvstore_inventory'][k]),
                'kv_missing': sorted(set(before['kvstore_inventory']) - set(kv)), 'kv_added': sorted(set(kv) - set(before['kvstore_inventory']))}
        hard = {k: [x for x in v if x != 'README.md'] for k, v in diff.items()}
        summary.update({'status': 'PASS' if not any(hard.values()) else 'FAIL', 'differences': diff})
        write(run / 'protection_result.json', summary); write(ROOT / 'results' / run.name / 'protection_result.json', summary)
    print(json.dumps(summary, indent=1)[:4000], flush=True)


# ------------------------------------------------------------- experiment
class ReRoPE(R3.Ablation):
    def __init__(self, run):
        super().__init__(run)
        self.rev3 = {}
        for r in map(json.loads, (REV3_RUN / 'llava/gqa/raw_requests.jsonl').read_text().splitlines()):
            if r.get('status') == 'OK':
                self.rev3[(r['arm'], r['image_id'], str(r['question_id']))] = r

    # --------------------------------------------- first-token margins
    def t1(self, image, turn, arm, phase):
        with FirstLogitsRef(self.runner.model) as ref:
            value, capture = super().t1(image, turn, arm, phase)
        value['first_token_top2'] = top2_margin(ref.logits)
        return value, capture

    def hit_arm(self, ctx, path, image, turn, arm, phase, **kw):
        with FirstLogitsRef(self.runner.model) as ref:
            value = super().hit_arm(ctx, path, image, turn, arm, phase, **kw)
        value['first_token_top2'] = top2_margin(ref.logits)
        return value

    # ------------------------------------------------- ReRoPE store
    def persist_rerope(self, capture, image, path):
        from mmimpress.piggyback import persist_captured_visual_prefix
        capacity(); before = process_io(); started = time.perf_counter()
        enc = capture['enc']
        vs, vn = self.runner.visual_span(enc['input_ids'])
        base, _, _, sep = self.runner.anyres_layout(enc['image_sizes'][0], vn)
        t0 = time.perf_counter()
        scores = anyres_token_scores(self.runner, capture['vision'].result_cpu(), enc['image_sizes'][0], vn, base_side=base)
        order = visionzip_repack_order(scores, sep)
        plan = reduced_plan(vs, vn, order, sep, 'compact')
        plan_ms = (time.perf_counter() - t0) * 1e3
        layers, rec = rerope_layers(cache_layers(capture['cache']), plan, self.inv_freq)
        meta_extra = {'checkpoint_revision': self.runner.cfg._commit_hash, 'ablation': 'rerope_kv25', 'rerope': True,
                      'kv_context': 'full_turn1', 'position_policy': 'compact',
                      'text_position_start': text_position_start({'v_token_start': vs, 'v_token_num': vn, 'newline_idx': sep}, 'compact')}
        receipt = persist_captured_visual_prefix(
            self.runner, layers, enc['input_ids'], enc['image_sizes'][0], capture['vision'].result_cpu(), path,
            capture_stats=capture['vision'], storage_policy='prefix25', extra_metadata=meta_extra,
            image_id=image['image_id'], model_id=self.runner.model_id, image_input_sha256=image['image_sha256'])
        hash_t0 = time.perf_counter(); files = seal_integrity(path); hash_ms = (time.perf_counter() - hash_t0) * 1e3
        total_ms = (time.perf_counter() - started) * 1e3; after = process_io()
        meta = receipt['meta']
        assert meta['full_importance_permutation'] == order and meta['text_position_start'] == plan['text_position_start']
        out = {'physical_write_id': str(uuid.uuid4()), 'store_path': str(path), 'arm': 'rerope', 'image_id': image['image_id'],
               'metadata': meta, 'files': files, 'timing_ms': receipt['timing_ms'], 'hash_and_seal_ms': hash_ms,
               'persistence_ms': total_ms, 'recompute': None,
               'rerope': {**rec, 'plan_ms': plan_ms, 'k': plan['k'], 'n_sep': plan['n_sep'],
                          'position_ids_sha256': hashlib.sha256(json.dumps(plan['position_ids']).encode()).hexdigest(),
                          'text_position_start': plan['text_position_start'], 'shares_t1_capture_with': 'ours'},
               'os_write_bytes': after['wchar'] - before['wchar'], 'os_write_calls': after['syscw'] - before['syscw'],
               'payload_sizes': payload_sizes(path), 'file_logical_bytes': sum(x['size'] for x in files.values()),
               'selection_digest': selection_digest(meta),
               'saliency_ms': capture['vision'].stats().get('saliency_total_instrumentation_ms'), 'truncation_control': None}
        return out, layers

    def activate_arm(self, path, image, arm):
        if arm != 'rerope':
            return super().activate_arm(path, image, arm)
        ctx, act = self.activate(path, image, 'B')
        m = ctx.meta
        assert m['ablation'] == 'rerope_kv25' and m['rerope'] is True and m['kv_context'] == 'full_turn1'
        assert m['position_policy'] == 'compact' and m['text_position_start'] == text_position_start(m, 'compact')
        return ctx, act

    def text_start(self, ctx, arm):
        if arm != 'rerope':
            return super().text_start(ctx, arm)
        m = ctx.meta
        assert m['kv_context'] == 'full_turn1' and m['position_policy'] == 'compact' and m.get('rerope') is True
        ts = text_position_start(m, 'compact')
        assert ts == int(m['v_token_start']) + int(m['stored_content_count']) + len(m['newline_idx'])
        return ts

    def rerope_gates(self, image, receipts, oracle, phase):
        g = {'image_id': image['image_id'], 'phase': phase, 'arm': 'rerope'}
        same = (receipts['rerope']['selection_digest'] == receipts['ours']['selection_digest'] and
                receipts['rerope']['payload_sizes'] == receipts['ours']['payload_sizes'] and
                receipts['rerope']['metadata']['full_importance_permutation'] == receipts['ours']['metadata']['full_importance_permutation'])
        g['identity_with_ours'] = same
        g['sentinel_rows'] = sentinel_rows(receipts['rerope']['store_path'])
        g['Gab'] = compare_rerope(receipts['ours']['store_path'], receipts['rerope']['store_path'],
                                  receipts['compact']['store_path'], self.inv_freq, oracle)
        g['status'] = 'PASS' if same and g['sentinel_rows'] == 0 and g['Gab']['status'] == 'PASS' else 'FAIL'
        append(self.base / 'gates_rerope.jsonl', g)
        if g['status'] != 'PASS':
            raise AssertionError(f"ReRoPE gate failure {image['image_id']}: identity={same} sentinel={g['sentinel_rows']} "
                                 f"Gab={g['Gab']['failed_gates']}")
        return g

    # --------------------------------------------------- Pruned-T1
    def pruned_request(self, image, turn, phase, probe_question=None):
        """Cache-less pruned first turn; composed TTFT (description only)."""
        from mmimpress.piggyback import VisionForwardCapture
        question = probe_question or turn['question']
        cap = VisionForwardCapture(self.runner, capture_saliency=True)
        vision_calls = []
        vh = self.runner.model.model.vision_tower.register_forward_pre_hook(lambda *a, **k: vision_calls.append(1))
        first_box = []
        try:
            with self.ref._ReadTrace() as reads, LayerPositions(self.runner) as lp:
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
                with Image.open(image['image_path']) as src:
                    pil = src.convert('RGB')
                t_dec = time.perf_counter()
                with cap:        # saliency capture is finalized at context exit (inside the timed request)
                    enc = self.runner.encode_prompt(pil, self.prompt(question, [], phase))
                    enc_dev = self.runner.to_device(enc)
                    torch.cuda.synchronize(); t_enc = time.perf_counter()
                    embeds = AbortAfterEmbeds(self.runner).run(self.runner, enc_dev)
                torch.cuda.synchronize(); t_merge = time.perf_counter()
                vs, vn = self.runner.visual_span(enc['input_ids'])
                base, _, _, sep = self.runner.anyres_layout(enc['image_sizes'][0], vn)
                scores = anyres_token_scores(self.runner, cap.result_cpu(), enc['image_sizes'][0], vn, base_side=base)
                order = visionzip_repack_order(scores, sep)
                plan = reduced_plan(vs, vn, order, sep, 'compact')
                torch.cuda.synchronize(); t_plan = time.perf_counter()
                text_index = list(range(vs + vn, int(enc['input_ids'].shape[1])))
                gen = pruned_generate(self.server, embeds, plan, text_index, first_box)
                e2e = (time.perf_counter() - t0) * 1e3
        finally:
            vh.remove()
        n_text = len(text_index)
        assert gen['prefill_len'] == plan['text_position_start'] + n_text
        value = {'prediction': gen['answer'], 'first_token_id': gen['first_token_id'], 'generated_token_ids': gen['generated_token_ids'],
                 'ttft_ms': (gen['first_token_at_s'] - t0) * 1e3, 'request_e2e_ms': e2e,
                 'ttft_breakdown_ms': {'image_decode': (t_dec - t0) * 1e3, 'processor_and_h2d': (t_enc - t_dec) * 1e3,
                                       'vision_projector_merge': (t_merge - t_enc) * 1e3, 'saliency_order_plan': (t_plan - t_merge) * 1e3,
                                       'gather': gen['gather_ms'], 'prefill_to_first_token': gen['prefill_ms']},
                 'ttft_note': 'composed measurement (full-token forward aborted after merge); not for efficiency comparison',
                 'decode_ms': gen['decode_ms'], 'prefill_ms': gen['prefill_ms'],
                 'selected_original_ids': plan['selected_original_ids'], 'k_target': plan['k'], 'N_content': plan['n'],
                 'text_position_start': plan['text_position_start'], 'pruned_input_len': gen['prefill_len'], 'text_tokens': n_text,
                 'position_check': lp.verify(gen['prefill_len']), 'vision_calls': len(vision_calls),
                 'online_query_score_calls': 0, 'read_preads': len(reads.calls), 'read_bytes_traced': sum(x['returned'] for x in reads.calls),
                 'actual_io': {'bytes': 0, 'preads': 0, 'calls': []}, 'first_token_top2': top2_margin(first_box[0] if first_box else None),
                 'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(), 'inputs_embeds_sha256': None, 'cache': 'none'}
        assert value['vision_calls'] == 1 and value['read_preads'] == 0
        diag = {'embeds': embeds, 'plan': plan, 'enc': enc, 'text_index': text_index, 'vision': cap}
        del first_box
        return value, diag

    def pruned_two_step(self, diag, question_prompt_ids):
        """G4 diagnostic: prefix forward (== Compact reduced prefill) then the question via _decode."""
        layers, _ = reduced_prefill(self.runner, diag['embeds'], diag['plan'])
        from transformers import DynamicCache
        cache = DynamicCache()
        for li, (k, v) in enumerate(layers):
            cache.update(k, v, li)
        logits = []
        h = self.runner.model.register_forward_hook(lambda m, a, o: logits.append(o.logits[0, -1].detach().float().cpu()) if hasattr(o, 'logits') else None)
        self.server.text_position_start = None
        try:
            answer, first, timing = self.server._decode(cache, question_prompt_ids.to(self.runner.model.device), diag['plan']['text_position_start'])
        finally:
            h.remove(); del cache
        return {'answer': answer, 'first_token_id': first, 'generated_token_ids': timing['generated_token_ids'], 'logits': logits}, layers

    # -------------------------------------------------------- correctness
    def correctness(self):
        super().correctness()                                  # rev.-3 (a)–(e), stores retained under llava/correctness
        samples = json.loads((self.run / 'correctness_manifest.json').read_text())['samples']
        samples = [samples[i] for i in [0, 3, 4, 5, 6]]
        gate = json.loads((self.base / 'gpu_correctness.json').read_text())
        assert gate['status'] == 'PASS'
        ext = {'status': 'NOT RUN', 'samples': []}
        for i, sample in enumerate(samples):
            path = self.base / 'correctness' / f'{i:02d}_{sample["image_id"]}'
            try:
                ext['samples'].append(self.correctness_extension(sample, path, gate['samples'][i]))
                print('G4-ext', i + 1, len(samples), 'PASS', flush=True)
            except Exception:
                ext['samples'].append({'image_id': sample['image_id'], 'status': 'FAIL', 'error': traceback.format_exc()})
                ext['status'] = 'FAIL'; write(self.base / 'gpu_correctness_extension.json', ext); raise
            write(self.base / 'gpu_correctness_extension.json', ext)
        ext['status'] = 'PASS'; write(self.base / 'gpu_correctness_extension.json', ext)

    def correctness_extension(self, sample, path, base_row):
        from mmimpress.serve import suffix_ids_from_prompt
        turn = {'question': sample['question'], 'question_id': sample['question_id']}
        value, cap = self.t1(sample, turn, 'ours', 'gqa')
        enc = cap['enc']; vs, vn = self.runner.visual_span(enc['input_ids'])
        _, _, _, sep = self.runner.anyres_layout(enc['image_sizes'][0], vn)
        n = vn - len(sep); k = (n + 3) // 4
        rank = self.ref._score_rank(self.runner, cap['vision'].result_cpu(), enc['image_sizes'][0], vn, sep)
        receipts = {a: {'store_path': str(path / a)} for a in BUILT}
        for a in BUILT:
            m = json.loads((path / a / 'meta.json').read_text())
            receipts[a].update({'metadata': m, 'selection_digest': selection_digest(m), 'payload_sizes': payload_sizes(path / a)})
        assert receipts['ours']['metadata']['full_importance_permutation'] == rank + sep
        rec, layers = self.persist_rerope(cap, sample, path / 'rerope')
        receipts['rerope'] = rec
        base_gate = [g for g in map(json.loads, (self.base / 'gates.jsonl').read_text().splitlines())
                     if g['phase'] == 'correctness' and g['image_id'] == sample['image_id']][-1]
        assert value['inputs_embeds_sha256'] == base_gate['inputs_embeds_sha256']['ours'], 'T1 embeds differ from the rev.-3 pass'
        out = {'image_id': sample['image_id'], 'question_id': sample['question_id'], 'k': k,
               't1_embeds_sha_equals_base_pass': True,
               'rerope_receipt_ms': {'persistence_ms': rec['persistence_ms'], 'rerope_ms': rec['rerope']['rerope_ms']}, 'checks': {}}
        gates = self.rerope_gates(sample, receipts, self.oracle(cap), 'correctness')
        out['checks']['g_rerope_gates'] = {'status': gates['status'], 'failed': gates['Gab']['failed_gates'],
                                           'layer0': {n_: gates['Gab']['gates'][f'same_position_layer0_k_{n_}'][0] for n_ in ('S', 'sep')},
                                           'neg': {n_: gates['Gab']['gates'][f'neg_unrotated_layer0_k_{n_}'][0]['rel_l2'] for n_ in ('S', 'sep')},
                                           'roundtrip_max_rel_l2': max(x['rel_l2'] for n_ in ('S', 'sep') for x in gates['Gab']['gates'][f'roundtrip_k_{n_}'])}
        # (f) ReRoPE: independent in-memory reference vs SSD path, compact text start on both sides
        ctx = self.activate_arm(path / 'rerope', sample, 'rerope')[0]
        ts = self.text_start(ctx, 'rerope')
        ref = self.memory_reference(ctx, sample['question'], layers, rank, k, ts)
        prod = self.trace_positions(ctx, sample['question'], layers, rank, k, ts)
        out['checks']['f_rerope'] = {'text_position_start': ts, 'comparison': self.ref._compare_outputs(ref, dict(prod['result'], logits=prod['logits'])),
                                     'cache': prod['cache'], 'masks': prod['attention_masks'], 'positions': prod['position_checks'],
                                     'reads': {x: prod['reads'][x] for x in ['actual_returned_bytes', 'actual_pread_calls']}}
        # question invariance + sentinel + IO on ReRoPE hits
        other = {'question': 'What is visible?', 'question_id': 'probe'}
        a = self.hit_arm(ctx, path / 'rerope', sample, turn, 'rerope', 'gqa', sentinel=True)
        b = self.hit_arm(ctx, path / 'rerope', sample, other, 'rerope', 'gqa', sentinel=True)
        assert a['selected_original_ids'] == b['selected_original_ids'] == rank[:k]
        out['checks']['e_rerope_invariance'] = {'read_bytes': a['actual_io']['bytes'], 'preads': a['actual_io']['preads'],
                                                'text_position_start': a['position_check']['text_position_start'], 'sentinel': a['sentinel_check']}
        # (h) Pruned-T1
        p1, d1 = self.pruned_request(sample, turn, 'gqa')
        p2, _ = self.pruned_request(sample, turn, 'gqa', probe_question='What is visible?')
        assert p1['selected_original_ids'] == p2['selected_original_ids'] == rank[:k]
        assert p1['vision_calls'] == 1 and p1['read_preads'] == 0 and p1['position_check']['verified']
        suffix = suffix_ids_from_prompt(self.runner, self.prompt(sample['question'], [], 'gqa'))
        two, prefix_layers = self.pruned_two_step(d1, suffix)
        from mmimpress.recompute_kv25 import StoreRows
        cs = StoreRows(path / 'compact')
        where = {r: j for j, r in enumerate(d1['plan']['input_index'])}          # reduced order
        rows = torch.tensor([where[vs + i] for i in rank[:k]], device=prefix_layers[0][0].device)
        ident = all(bit_equal(prefix_layers[li][kk][0, :, rows].transpose(0, 1).to(torch.float16).cpu(),
                              cs.content(li, 'k' if kk == 0 else 'v')) for li in range(len(prefix_layers)) for kk in (0, 1))
        assert ident, 'two-step pruned prefix KV is not bit-identical to the Compact store'
        cctx = self.activate_arm(path / 'compact', sample, 'compact')[0]
        chit = self.hit_arm(cctx, path / 'compact', sample, turn, 'compact', 'gqa', logits=True)
        out['checks']['h_pruned'] = {
            'selected_unchanged': True, 'vision_calls': p1['vision_calls'], 'preads': p1['read_preads'], 'positions': p1['position_check'],
            'ttft_breakdown_ms': p1['ttft_breakdown_ms'], 'two_step_prefix_bit_identical_to_compact_store': ident,
            'one_forward': {'first': p1['first_token_id'], 'ids': p1['generated_token_ids'], 'prediction': p1['prediction'], 'top2': p1['first_token_top2']},
            'two_step': {'first': two['first_token_id'], 'ids': two['generated_token_ids'], 'prediction': two['answer']},
            'compact_hit': {'first': chit['first_token_id'], 'ids': chit['generated_token_ids'], 'prediction': chit['prediction'], 'top2': chit['first_token_top2']},
            'one_forward_vs_two_step_first_equal': p1['first_token_id'] == two['first_token_id'],
            'two_step_vs_compact_hit': compare_recorded(two, dict(chit, answer=chit['prediction'], logits=chit['all_logits'])),
            'one_forward_vs_two_step': {'first_token_exact': p1['first_token_id'] == two['first_token_id'],
                                        'generated_token_ids_exact': p1['generated_token_ids'] == two['generated_token_ids'],
                                        'answer_exact': p1['prediction'] == two['answer'], 'gated': False}}
        ctx.close(); cctx.close()
        del cap, layers, prefix_layers, d1, value, ref, prod
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
                order = ARMS[(ix + ti) % 7:] + ARMS[:(ix + ti) % 7]
                for arm in order:
                    if ti == 0 and arm == 'rerope':
                        continue
                    exclusive()
                    logical = f'llava/{phase}/{image["image_id"]}/{arm}/{ti + 1}'
                    try:
                        if arm == 'pruned':
                            value, diag = self.pruned_request(image, turn, phase); del diag
                        elif ti == 0 or arm == 'recompute':
                            value, capture = self.t1(image, turn, arm if ti == 0 else 'recompute', phase)
                            if ti == 0 and arm == 'ours':
                                oracle = self.oracle(capture)
                            if ti == 0 and arm != 'recompute':
                                path = self.stores / phase / image['image_id'] / arm
                                receipt, _, control = self.persist_arm(capture, image, arm, path)
                                if control is not None:
                                    controls[arm] = control
                                receipt['phase'] = phase
                                if arm == 'ours':
                                    rpath = self.stores / phase / image['image_id'] / 'rerope'
                                    rreceipt, _ = self.persist_rerope(capture, image, rpath); rreceipt['phase'] = phase
                                del capture; gc.collect(); torch.cuda.empty_cache()
                                ctx, activation = self.activate_arm(path, image, arm)
                                receipt['activation'] = activation
                                contexts[arm], receipts[arm], paths[arm] = ctx, receipt, path
                                append(persist_file, receipt)
                                if arm == 'ours':
                                    rctx, ract = self.activate_arm(rpath, image, 'rerope'); rreceipt['activation'] = ract
                                    contexts['rerope'], receipts['rerope'], paths['rerope'] = rctx, rreceipt, rpath
                                    append(persist_file, rreceipt)
                            else:
                                del capture
                            if ti == 0:
                                t1[arm] = value
                            value['actual_io'] = {'bytes': 0, 'preads': 0, 'calls': []}
                        else:
                            value = self.hit_arm(contexts[arm], paths[arm], image, turn, arm, phase, sentinel=(phase == 'smoke' and arm in KV25))
                            hits[(arm, ti)] = value
                        append(raw, self.row(logical, phase, image, arm, ti, turn, value))
                    except Exception:
                        append(raw, {'logical_request_id': logical, 'status': 'FAIL', 'error': traceback.format_exc()})
                        raise
                if ti == 0:
                    shared = dict(t1['ours'], shared_t1_from='ours', executed=False)
                    append(raw, self.row(f'llava/{phase}/{image["image_id"]}/rerope/1', phase, image, 'rerope', 0, turn, shared))
                    self.image_gates(image, t1, receipts, phase, oracle, controls)
                    self.rerope_gates(image, receipts, oracle, phase)
                    del oracle; controls.clear()
            for ti in range(1, len(image['turns'])):
                io = {(hits[(a, ti)]['actual_io']['bytes'], hits[(a, ti)]['actual_io']['preads']) for a in KV25}
                assert len(io) == 1, f'KV25 arms read different bytes: {io}'
            for ctx in contexts.values():
                ctx.close()
            contexts.clear(); self.close_contexts()
            for arm, path in paths.items():
                append(self.base / 'cleanup.jsonl', {'stage': 'before', 'path': str(path), 'arm': arm, 'phase': phase,
                                                     'sealed_files': len(receipts[arm]['files']),
                                                     'sealed_sha256': {p: x.get('sha256') for p, x in receipts[arm]['files'].items()}})
                guarded_rmtree(path, self.stores)
                append(self.base / 'cleanup.jsonl', {'stage': 'after', 'path': str(path), 'exists': path.exists()})
            print('llava', phase, ix + 1, len(manifest['images']), 'complete', flush=True)

    def row(self, logical, phase, image, arm, ti, turn, value):
        r = super().row(logical, phase, image, arm, ti, turn, value)
        for key in ['first_token_top2', 'ttft_breakdown_ms', 'ttft_note', 'text_position_start', 'pruned_input_len', 'text_tokens', 'read_preads']:
            if key in value:
                r[key] = value[key]
        r['cache_hit'] = ti > 0 and arm not in ('recompute', 'pruned')
        return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--protect', choices=['before', 'after'])
    ap.add_argument('--gpu', help='comma list of correctness,smoke,gqa')
    args = ap.parse_args(); run = Path(args.run).resolve()
    assert run.parent == ROOT / 'runs' and run.name.startswith('rerope_kv25_')
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
        exp = ReRoPE(run)
        if 'correctness' in stages:
            exp.correctness(); status['G4 GPU CORRECTNESS'] = 'PASS'; write(status_path, status)
        assert status.get('G4 GPU CORRECTNESS') == 'PASS', 'G4 must pass before pilots'
        sample = json.loads((run / 'smoke_manifest.json').read_text())['images'][0]
        for arm in ['recompute', 'fullload', 'ours']:
            val, cap = exp.t1(sample, sample['turns'][0], arm, 'smoke'); del val, cap
        val, diag = exp.pruned_request(sample, sample['turns'][0], 'smoke'); del val, diag
        write(exp.base / f'warmup_{"_".join(stages)}.json', {'excluded': True, 'normal_requests': 4})
        for phase in [s for s in stages if s != 'correctness']:
            if phase == 'gqa':
                assert status.get('SMOKE') == 'PASS', 'smoke must pass before Stage 1'
            key = 'SMOKE' if phase == 'smoke' else 'STAGE1 GQA'
            status[key] = 'RUNNING'; write(status_path, status)
            exp.pilot(phase)
            status[key] = 'PASS'; write(status_path, status)
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

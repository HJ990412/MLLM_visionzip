#!/usr/bin/env python3
"""VZ-Pruned-E2E vs Ours-KV25 vs ReComp on MT-GQA generated history (Stage 2).

Contract: docs/vz_pruned_e2e_mtgqa_contract.md.  Lifecycle, prompt, strict
scorer, request helpers and the Ours/ReComp paths are 108's (loaded, never
modified); this script adds the third arm, the per-image VZ store gate (G2),
the hit-vs-standalone sample (G8) and the correctness stage (G4).
Stages: --freeze, --protect before|after, --gpu correctness,smoke,integration,main [--resume].
"""
from __future__ import annotations
import argparse, gc, hashlib, importlib.util, json, os, shutil, subprocess, sys, time, traceback, uuid
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from mmimpress.recompute_kv25 import (EmbeddingCapture, PositionRecorder, PositionServer, bit_equal, default_inv_freq,
                                      layer0_oracle, payload_sizes, selection_digest, sentinel_rows, sha_manifest,
                                      stat_inventory, structural_identity, structural_rows, tensor_sha256)
from mmimpress.rerope_kv25 import AbortAfterEmbeds, FirstLogitsRef, pruned_generate, top2_margin
from mmimpress.vz_pruned_e2e import (compare_vz, hit_vs_standalone, path_equivalence, pruned_t1_request,
                                     standalone_pruned, vz_control, vz_prefix_layers)


def helper(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / filename)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


M = helper('108_eval_llava_mt_gqa_B.py', '_vz_b_main_lifecycle')
import main_b_support as support

METHODS = ('recompute', 'ours_prefixstore_kv25', 'vz_pruned_e2e')
LABELS = dict(zip(METHODS, ('ReComp', 'Ours-KV25', 'VZ-Pruned-E2E')))
STORE_KIND = {'ours_prefixstore_kv25': 'image_only', 'vz_pruned_e2e': 'vz'}
SCHEMA = 'llava-mtgqa-vz-e2e-v1'
MODEL_ID, REVISION = M.MODEL_ID, M.REVISION
CONTRACT = ROOT / 'docs/vz_pruned_e2e_mtgqa_contract.md'
B_MAIN = ROOT / 'runs/mt_gqa_B_llava5_qwen3_20261001T093921Z'
REF_RUNS = [B_MAIN, ROOT / 'runs/prefix25_persistence_20261001T050113Z',
            ROOT / 'runs/recompute_kv25_ablation_20261102T045423Z', ROOT / 'runs/recompute_kv25_ablation_20261102T061243Z',
            ROOT / 'runs/rerope_kv25_20261102T080818Z', ROOT / 'runs/rerope_kv25_20261102T081118Z',
            ROOT / 'runs/rerope_kv25_20261102T093759Z']
NEW_SOURCES = ['mmimpress/vz_pruned_e2e.py', 'scripts/147_eval_vz_pruned_e2e_mtgqa.py',
               'scripts/148_audit_vz_pruned_e2e_mtgqa.py', 'scripts/149_report_vz_pruned_e2e_mtgqa.py',
               'tests/test_vz_pruned_e2e.py']
REPORT_ONLY_SOURCES = ('149_report_vz_pruned_e2e_mtgqa.py',)
DEV_EVIDENCE = Path('/tmp/claude-1000/-home-dblab-hj-mllm-v2/d4c13472-218c-434b-a623-f821b9709a83/scratchpad/vz_dev')
DEV_EVIDENCE_SHA = {'dev_check.py': '233f7bb6355ab640e90db75c8f73b622e39420fe41f65b7da0ac7cf9a23042a0',
                    'dev_check.log': '4d1807ba0c0b1436857d5b50d52eb95fb3a749233de97d99d27e26b61599d17a'}
MIN_RESERVE = 30 * 2**30
PEAK_WORKING_SET = 2 * 2**30
SEED = 1234
sha, atomic_json, append, load_jsonl, safe = M.sha, M.atomic_json, M.append, M.load_jsonl, M.safe
canonical_hash, strict_score = M.canonical_hash, M.strict_score


# ------------------------------------------------------------------ freeze
def source_files():
    paths = [p for d in ['mmimpress', 'scripts', 'tests'] for p in sorted((ROOT / d).rglob('*.py'))
             if '__pycache__' not in p.parts and p.name not in REPORT_ONLY_SOURCES]
    return paths + [CONTRACT]


def freeze(run):
    res = ROOT / 'results' / run.name
    run.mkdir(parents=True, exist_ok=False); res.mkdir(parents=True, exist_ok=False)
    dialogues = M.frozen_dialogues()
    b_manifest = json.loads((B_MAIN / 'llava/manifest.json').read_text())
    assert [d['dialog_id'] for d in b_manifest['dialogues']] == [d['dialog_id'] for d in dialogues]
    identity = {}
    for d in dialogues:
        iid = d['image_id']
        if iid not in identity:
            digest = sha(ROOT / d['image_path'])
            assert digest == b_manifest['image_identity'][iid]['sha256'], iid
            identity[iid] = {'image_path': d['image_path'], 'sha256': digest}
    images = sorted(identity)
    rng = np.random.default_rng(SEED)
    g8 = sorted(rng.choice(images, 20, replace=False).tolist())
    smoke_images = {d['image_id'] for d in dialogues if d['dialog_id'] in b_manifest['smoke_dialogue_ids']}
    pool = [i for i in images if i not in smoke_images and i not in set(g8)]
    correctness = sorted(rng.choice(pool, 5, replace=False).tolist())
    manifest = {'schema_version': SCHEMA, 'dataset': 'MT-GQA-reconstructed', 'index_sha256': M.INDEX_SHA,
                'workload_sha256': M.WORKLOAD_SHA, 'dialogues': dialogues, 'image_identity': identity, 'methods': list(METHODS),
                'expected_requests': len(dialogues) * 9, 'smoke_dialogue_ids': b_manifest['smoke_dialogue_ids'],
                'integration_dialogue_ids': b_manifest['integration_dialogue_ids'],
                'b_main_manifest_sha256': sha(B_MAIN / 'llava/manifest.json'), 'g8_images': g8,
                'correctness_images': correctness, 'rotation': 'global dialogue ordinal % 3'}
    atomic_json(run / 'manifest.json', manifest)
    revision = (Path.home() / '.cache/huggingface/hub/models--llava-hf--llava-v1.6-vicuna-7b-hf/refs/main').read_text().strip()
    assert revision == REVISION
    free = shutil.disk_usage(run).free
    plan = {'policy': 'FRESH_IMAGE_STREAMING', 'minimum_reserve_bytes': MIN_RESERVE, 'peak_working_set_bytes': PEAK_WORKING_SET,
            'maximum_new_payload_bytes': 4 * 2**30, 'largest_image_bundle_bound_bytes': 2 * 700 * 10**6,
            'free_bytes_at_freeze': free, 'maximum_concurrent_main_images': 1,
            'conditioning': 'successful per-file POSIX_FADV_DONTNEED; no NAND/controller cold claim'}
    assert free >= MIN_RESERVE + 15 * 2**30, free
    atomic_json(run / 'storage_plan.json', plan)
    sources = {str(p.relative_to(ROOT)): sha(p) for p in source_files()}
    config = {'schema_version': SCHEMA, 'methods': list(METHODS), 'labels': LABELS, 'model': MODEL_ID, 'revision': revision,
              'seed': SEED, 'greedy': True, 'max_new_tokens': 16, 'batch_size': 1, 'cpu_threads': 2, 'chunk_size': 64, 'ratio': .25,
              'quantization': 'NF4 double quant, BF16 compute', 'attention': 'eager', 'store_dtype': 'float16',
              'storage_policy': 'prefix25', 'history_policy': 'method_local_generated',
              'scorer': 'strict normalized exact match (108/109, unchanged)', 'rotation': manifest['rotation'],
              'store_source': 'first dialogue of each image in frozen ordinal order (B main rule)',
              'vz_t1_path': 'real: get_image_features once, image-only VisionZip order, direct [sys|keep|text] assembly, one LM prefill',
              'vz_hit_path': 'rev.-3 Compact: request_cvpr25 prefix mode, PositionServer text start L',
              'primary_endpoints': ['ours_minus_vz_T1_EM', 'ours_minus_vz_all_EM'],
              'bootstrap': {'unit': 'image', 'resamples': 10000, 'seed': SEED, 'joint_contrasts': True, 'weighting': 'pooled sums/counts'},
              'reference_runs': {'b_main': str(B_MAIN), 'protected': [str(p) for p in REF_RUNS]},
              'contract_sha256': sha(CONTRACT), 'storage_plan_sha256': sha(run / 'storage_plan.json'),
              'manifest_sha256': sha(run / 'manifest.json'), 'source_sha256': sources,
              'packages': {'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
                           'bitsandbytes': __import__('bitsandbytes').__version__, 'numpy': np.__version__},
              'python': sys.version, 'platform': __import__('platform').platform()}
    atomic_json(run / 'config.json', config); atomic_json(res / 'config.json', config)
    shutil.copy2(CONTRACT, run / 'CONTRACT.md'); shutil.copy2(CONTRACT, res / 'CONTRACT.md')
    atomic_json(run / 'environment.json', {
        'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout,
        'packages': subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout})
    (run / 'dev').mkdir(); (res / 'dev').mkdir()
    for name, digest in DEV_EVIDENCE_SHA.items():
        assert sha(DEV_EVIDENCE / name) == digest, name
        shutil.copy2(DEV_EVIDENCE / name, run / 'dev' / name); shutil.copy2(DEV_EVIDENCE / name, res / 'dev' / name)
    atomic_json(run / 'gpu_freeze.json', {'files': {**sources, 'manifest.json': sha(run / 'manifest.json'),
                                                    'config.json': sha(run / 'config.json')},
                                          'g8_images': g8, 'correctness_images': correctness,
                                          'dev_evidence_sha256': DEV_EVIDENCE_SHA, 'dev_evidence_scope': 'development check before freeze; not results'})
    print('FROZEN', run, flush=True)


def verify_freeze(run):
    config = json.loads((run / 'config.json').read_text()); manifest = json.loads((run / 'manifest.json').read_text())
    frozen = json.loads((run / 'gpu_freeze.json').read_text())
    for rel, digest in frozen['files'].items():
        p = run / rel if rel in ('manifest.json', 'config.json') else ROOT / rel
        if sha(p) != digest: raise RuntimeError('FROZEN_SOURCE_CHANGED: ' + rel)
    if sha(CONTRACT) != config['contract_sha256']: raise RuntimeError('FROZEN_CONTRACT_MISMATCH')
    if list(METHODS) != config['methods']: raise RuntimeError('FROZEN_METHODS_MISMATCH')
    if M.frozen_dialogues() != manifest['dialogues']: raise RuntimeError('FROZEN_DATASET_MISMATCH')
    for iid, item in manifest['image_identity'].items():
        if sha(ROOT / item['image_path']) != item['sha256']: raise RuntimeError('FROZEN_IMAGE_MISMATCH: ' + iid)
    config.update(_file_sha256=sha(run / 'config.json'), _manifest_sha256=sha(run / 'manifest.json'))
    return config, manifest


# -------------------------------------------------------------- protection
def protect(run, stage):
    excl = [f'results/{run.name}'] + NEW_SOURCES
    roots = [ROOT / x for x in ['results', 'docs', 'mmimpress', 'scripts', 'tests', 'README.md']] + REF_RUNS
    started = time.time()
    shas = {k: v for k, v in sha_manifest(roots, ROOT, exclude=excl).items() if '__pycache__' not in k}
    kv = stat_inventory(sorted(ROOT.glob('kvstore*')), ROOT)
    doc = {'stage': stage, 'utc': time.time(), 'elapsed_s': time.time() - started, 'excluded_new_paths': excl,
           'sha256_files': len(shas), 'sha256_bytes': sum(v['size'] for v in shas.values()),
           'kvstore_inventory_files': len(kv), 'sha256': shas, 'kvstore_inventory': kv}
    with (run / f'protection_{stage}.json').open('x') as f:
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
        atomic_json(run / 'protection_result.json', summary)
        atomic_json(ROOT / 'results' / run.name / 'protection_result.json', summary, replace=True)
    print(json.dumps(summary, indent=1)[:4000], flush=True)


# ------------------------------------------------------------ requests
def request_id(run_id, phase, method, dialog_id, turn):
    return f'{SCHEMA}:{run_id}:{phase}:{MODEL_ID}:{method}:{dialog_id}:t{turn}'


def method_order(ordinal):
    i = int(ordinal) % 3
    return METHODS[i:] + METHODS[:i]


def prompt_factory(dialog, turn, method, generated):
    """108's causal prompt, verbatim, for the three arms of this run."""
    if method not in METHODS or turn not in (1, 2, 3): raise ValueError('invalid method/turn')
    if any(t >= turn for t in generated): raise ValueError('future/current answer in history')
    lines, entries = [], []
    for t in range(1, turn):
        prior = dialog['turns'][t - 1]; row = generated[t]
        if row['method_id'] != method or row['dialog_id'] != dialog['dialog_id']:
            raise ValueError('foreign method/dialogue history')
        answer = row['prediction']
        lines += [f"Q{t}: {prior['question'].strip()}", f'A{t}: {answer}']
        entries.append({'turn_id': t, 'question_id': str(prior['question_id']), 'question': prior['question'].strip(),
                        'answer': answer, 'answer_source': 'method_local_generated', 'source_model': MODEL_ID,
                        'source_method_id': method, 'source_dialog_id': dialog['dialog_id'],
                        'source_logical_request_id': row['logical_request_id'],
                        'source_physical_execution_id': row['physical_execution_id']})
    history = '\n'.join(lines); body = ([history, ''] if history else [])
    body += [f"Current question Q{turn}: {dialog['turns'][turn - 1]['question'].strip()}", f'{M.SHORT_ANSWER} ASSISTANT:']
    return 'USER: <image>\n' + '\n'.join(body), history, entries


def capacity_guard(run):
    plan = json.loads((run / 'storage_plan.json').read_text()); free = shutil.disk_usage(run).free
    if free < plan['minimum_reserve_bytes'] + plan['peak_working_set_bytes']:
        raise RuntimeError(f'BLOCKED_STORAGE: free={free}')
    scratch = run / 'scratch'
    # retained G4 stores (scratch/correctness, kept by contract) are not incomplete payload
    owned = sum(p.stat().st_size for p in scratch.rglob('*') if p.is_file() and not p.is_symlink()
                and 'correctness' not in p.relative_to(scratch).parts[:1]) if scratch.exists() else 0
    if owned + plan['largest_image_bundle_bound_bytes'] > plan['maximum_new_payload_bytes']:
        raise RuntimeError('BLOCKED_STORAGE: retained incomplete payload allowance exceeded')
    return {'free_bytes': free, 'retained_scratch_bytes': owned}


def cleanup_committed(scratch, allowlist, audit, commit, run_prefix='vz_pruned_e2e_mtgqa_'):
    scratch = Path(scratch)
    if audit.get('status') != 'PASS' or commit.get('status') != 'COMMITTED':
        raise ValueError('independent audit and atomic image commit required')
    if scratch.is_symlink() or not scratch.is_dir(): raise ValueError('unsafe scratch root')
    resolved = scratch.resolve()
    if 'scratch' not in scratch.parts or not any(p.startswith(run_prefix) for p in scratch.parts): raise ValueError('outside scratch allowlist')
    found = sorted(p.relative_to(scratch).as_posix() for p in scratch.rglob('*') if p.is_file() or p.is_symlink())
    if found != sorted(allowlist): raise ValueError('cleanup allowlist mismatch')
    for relative, expected in allowlist.items():
        p = scratch / relative
        if p.is_symlink() or p.stat().st_nlink != 1 or not p.resolve().is_relative_to(resolved): raise ValueError('linked/non-owned cleanup path')
        if sha(p) != expected: raise ValueError('cleanup payload changed')
    for relative in allowlist: (scratch / relative).unlink()
    for p in sorted(scratch.rglob('*'), key=lambda x: len(x.parts), reverse=True):
        if p.is_dir(): p.rmdir()
    scratch.rmdir(); M.sync_dir(scratch.parent)


def embeds_sha256(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.int16).numpy().tobytes()).hexdigest()


class Experiment:
    def __init__(self, run, config, manifest):
        from mmimpress.serve import Server
        self.run, self.config, self.manifest = run, config, manifest
        self.runner = M.V.load_runner()
        self.servers = {'old': Server(self.runner, max_new_tokens=16), 'vz': PositionServer(self.runner, max_new_tokens=16)}
        tc = self.runner.cfg.text_config
        assert tc.rope_scaling is None
        self.inv_freq = self.runner.model.model.language_model.rotary_emb.inv_freq.detach().double().cpu()
        assert torch.allclose(self.inv_freq, default_inv_freq(self.runner.head_dim, tc.rope_theta), rtol=1e-6)
        self.warmup = M.QA._warmup(self.runner, self.servers['old'])
        self.g8 = set(manifest['g8_images'])
        self.ordinals = {d['dialog_id']: i for i, d in enumerate(manifest['dialogues'])}

    # ---------------------------------------------------------------- T1
    def t1(self, method, dialog, factory, capture):
        """Real first turn of every arm; ``capture`` only on the source dialogue."""
        image_path = ROOT / dialog['image_path']
        if method == 'vz_pruned_e2e':
            result, diag = pruned_t1_request(self.runner, self.servers['vz'], image_path, factory,
                                             M.QA._combined_processor, counts_cls=M.PilotCounts)
            result['first_token_top2'] = top2_margin(result['first_logits'])
            result['image_input_sha256'] = M.QA._image_input_hash(diag['enc_cpu'])
            diag['image_input_sha256'] = result['image_input_sha256']
            return result, diag
        emb = EmbeddingCapture(self.runner) if capture else None
        with FirstLogitsRef(self.runner.model) as ref:
            if emb is not None:
                with emb:
                    result, diag = M.normal_request(self.runner, self.servers['old'], image_path, factory, 'image_only' if capture else 'none')
            else:
                result, diag = M.normal_request(self.runner, self.servers['old'], image_path, factory, 'none')
        result['first_token_top2'] = top2_margin(ref.logits)
        if emb is not None:
            diag['embeds'] = emb.embeds
            result['inputs_embeds_sha256'] = tensor_sha256(emb.embeds)
            result['language_model_calls'] = emb.calls
        return result, diag

    # --------------------------------------------------------- persistence
    def persist(self, method, result, diag, path, image_id, source):
        from mmimpress.piggyback import persist_captured_visual_prefix
        enc = diag['enc_cpu']
        common = dict(image_id=image_id, model_id=MODEL_ID, chunk_size=64, image_input_sha256=diag['image_input_sha256'])
        meta = {'dataset': 'MT-GQA-reconstructed', 'source_dialog_id': source['dialog_id'], 'source_turn_id': 1,
                'source_method_id': method, 'source_execution_id': source['physical_execution_id'],
                'source_logical_request_id': source['logical_request_id'], 'run_schema': SCHEMA,
                'checkpoint_revision': self.runner.cfg._commit_hash}
        if method == 'ours_prefixstore_kv25':
            meta.update(ablation='vz_pruned_e2e_mtgqa', kv_context='full_turn1', position_policy='original')
            call = lambda: persist_captured_visual_prefix(
                self.runner, result['captured_past_key_values'], enc['input_ids'], enc['image_sizes'][0], diag['vision'].result_cpu(),
                path, capture_stats=diag['vision'], full_integrity_hash=False, storage_policy='prefix25', extra_metadata=meta, **common)
            receipt = support.persist_llava(call, path, 'image_only', source)
            receipt['synthetic'] = None
        else:
            plan = diag['plan']; box = {}
            meta.update(ablation='vz_pruned_e2e', kv_context='pruned_t1', position_policy='compact',
                        text_position_start=plan['text_position_start'], prefill_len=result['pruned_input_len'],
                        source_turn1_text_tokens=diag['n_text'])

            def call():
                layers, rec = vz_prefix_layers(result['captured_past_key_values'], plan); box.update(rec)
                try:
                    return persist_captured_visual_prefix(
                        self.runner, layers, enc['input_ids'], enc['image_sizes'][0], diag['vision'].result_cpu(), path,
                        capture_stats=diag['vision'], full_integrity_hash=False, storage_policy='prefix25', extra_metadata=meta, **common)
                finally:
                    del layers
            receipt = support.persist_llava(call, path, 'vz', source)
            receipt['synthetic'] = box
            assert receipt['meta']['full_importance_permutation'] == diag['order']
            assert receipt['meta']['text_position_start'] == plan['text_position_start']
        receipt['selection_digest'] = selection_digest(receipt['meta'])
        receipt['payload_sizes'] = payload_sizes(path)
        return receipt

    def activate(self, method, path, hashes):
        if method == 'ours_prefixstore_kv25':
            ctx = M.activate(self.runner, method, path, hashes)
            m = ctx.meta
            assert m['kv_context'] == 'full_turn1' and m['position_policy'] == 'original'
            return ctx
        from mmimpress.serve import ImageContext
        started = time.perf_counter()
        ctx = ImageContext(path, self.runner.model.device, drop_cache=False, require_v_hidden=False)
        ctx.validate_prefix_layout('visionzip_image_only')
        m = ctx.meta
        assert m['ablation'] == 'vz_pruned_e2e' and m['kv_context'] == 'pruned_t1' and m['position_policy'] == 'compact'
        assert m['text_position_start'] == int(m['v_token_start']) + int(m['stored_content_count']) + len(m['newline_idx'])
        ctx.activation = {'host_tensor_bytes': sum(t.numel() * t.element_size() for t in ctx.cache.sys_kv.values()),
                          'gpu_tensor_bytes': 0, 'visual_payload_resident_bytes': 0,
                          'file_bytes': sum((path / n).stat().st_size for n in ('meta.json', 'sys_kv.pt')),
                          'completion_integrity': ctx.integrity_activation, 'event_id': str(uuid.uuid4()),
                          'event_kind': 'activation', 'outer_wall_ms': (time.perf_counter() - started) * 1000}
        return ctx

    # ---------------------------------------------------------------- hits
    def hit(self, method, ctx, factory, image_id):
        server = self.servers['old' if method == 'ours_prefixstore_kv25' else 'vz']
        ts = None
        if method == 'vz_pruned_e2e':
            ts = int(ctx.meta['text_position_start']); server.text_position_start = ts
        original = [l.self_attn.forward for l in self.runner.layers]
        rec = PositionRecorder(self.runner.model)
        try:
            with M.conditioning_trace() as conditioning, FirstLogitsRef(self.runner.model) as ref, rec:
                result, diag = M.legacy_hit(self.runner, server, ctx, factory, method, image_id)
        finally:
            server.text_position_start = None
        assert all(l.self_attn.forward == f for l, f in zip(self.runner.layers, original)), 'adapter not restored'
        from mmimpress.serve import BIAS
        assert not BIAS, 'stale mask after request'
        assert conditioning and all(e['status'] == 'SUCCESS' for e in conditioning), 'page cache conditioning not verified'
        result.update(conditioning_events=conditioning, adapter_restored=True, page_cache_conditioning_method='posix_fadvise_DONTNEED',
                      page_cache_conditioning_excluded_from_ttft=True, first_token_top2=top2_margin(ref.logits),
                      position_check=rec.verify(int(ctx.meta['prefix_len']), ts), first_logits=ref.logits)
        result['metadata_activation'] = ctx.activation
        return result, diag

    # --------------------------------------------------------------- gates
    def image_gates(self, image_id, phase, t1, diags, receipts, paths):
        """G2 for the VZ store (before any VZ hit), outside every timer."""
        ours, vz = diags['ours_prefixstore_kv25'], diags['vz_pruned_e2e']
        v0, vn = ours['v_token_start'], ours['v_token_num']
        g = {'image_id': image_id, 'phase': phase}
        ro, rv = receipts['ours_prefixstore_kv25'], receipts['vz_pruned_e2e']
        g['G2a'] = {'selection_digest_equal': ro['selection_digest'] == rv['selection_digest'],
                    'payload_sizes_equal': ro['payload_sizes'] == rv['payload_sizes'],
                    'k': (ro['meta']['stored_content_count'], rv['meta']['stored_content_count']),
                    'permutation_equal': ro['meta']['full_importance_permutation'] == rv['meta']['full_importance_permutation'],
                    't1_selection_sha256': t1['vz_pruned_e2e']['selection_sha256']}
        ours_prefix = embeds_sha256(ours['embeds'][0, :v0 + vn])
        g['G2b'] = {'ours_prefix_embeds_sha256': ours_prefix, 'vz_prefix_embeds_sha256': vz['prefix_embeds_sha256'],
                    'equal': ours_prefix == vz['prefix_embeds_sha256'], 'rows': v0 + vn}
        t0 = time.perf_counter()
        oracle = layer0_oracle(self.runner, ours['embeds'], v0 + vn, self.inv_freq)
        control = vz_control(self.runner, vz)
        from mmimpress.model import cache_layers
        L = len(vz['plan']['input_index'])
        reduced = [(k[:, :, :L], v[:, :, :L]) for k, v in cache_layers(t1['vz_pruned_e2e']['captured_past_key_values'])]
        g['G2c_memory_bf16'] = structural_identity(reduced, control); del reduced
        g['G2'] = compare_vz(paths['ours_prefixstore_kv25'], paths['vz_pruned_e2e'], self.inv_freq, oracle, control)
        g['G2e'] = {'sentinel_rows': sentinel_rows(paths['vz_pruned_e2e']),
                    'text_position_start': rv['meta']['text_position_start'],
                    'expected': v0 + rv['meta']['stored_content_count'] + len(rv['meta']['newline_idx'])}
        g['control'] = {'length': control['length'], 'structural_rows': len(control['rows']), 'control_ms': control['control_ms'],
                        'outside_timers': True, 'computed_after': 'both stores sealed; before activation and every hit'}
        g['gate_wall_ms'] = (time.perf_counter() - t0) * 1e3
        ok = (all(g['G2a'][k] for k in ('selection_digest_equal', 'payload_sizes_equal', 'permutation_equal')) and g['G2b']['equal']
              and g['G2']['status'] == 'PASS' and g['G2e']['sentinel_rows'] == 0 and g['G2e']['text_position_start'] == g['G2e']['expected']
              and (phase != 'correctness' or g['G2c_memory_bf16']['ok']))
        g['status'] = 'PASS' if ok else 'FAIL'
        del oracle, control; torch.cuda.empty_cache()
        return g

    def g8_pairs(self, dialog, rows, hits, vz_diag, image_id):
        """Hit vs cache-less single prefill with the same history (recorded only)."""
        out = []
        for turn in (2, 3):
            hit = hits[turn]
            suffix = torch.tensor(hit['suffix_input_ids'], dtype=torch.long)
            sa = standalone_pruned(self.runner, self.servers['vz'], vz_diag, suffix)
            cmp = hit_vs_standalone(hit, sa)
            out.append({'image_id': image_id, 'dialog_id': dialog['dialog_id'], 'turn_id': turn,
                        'hit_request_id': rows[turn]['logical_request_id'], 'standalone_prefill_len': sa['prefill_len'],
                        'hit_answer': hit['answer'], 'standalone_answer': sa['answer'], **cmp})
        return out

    # ------------------------------------------------------------- rows
    def make_row(self, result, diag, meta, *, method, dialog, turn, phase, attempt, physical_id, ordinal, store_info):
        question = dialog['turns'][turn - 1]; n = int(meta['n_spatial']); hit = turn > 1
        reduced = method == 'vz_pruned_e2e' or (hit and method == 'ours_prefixstore_kv25')
        k = (n + 3) // 4 if reduced else n
        logical = request_id(self.run.name, phase, method, dialog['dialog_id'], turn)
        suffix = M.QA._suffix_from_tokenized(self.runner, self.runner.processor.tokenizer(diag['prompt'], return_tensors='pt')).cpu().tolist()
        if result.get('suffix_input_ids') is not None: assert result['suffix_input_ids'] == suffix
        value = {a: b for a, b in result.items() if a not in ('captured_past_key_values', 'first_logits', 'logits')}
        g = {key: meta.get(key, []) if key == 'padding_idx' else meta[key] for key in
             ('num_heads', 'head_dim', 'num_layers', 'v_token_start', 'v_token_num', 'n_spatial', 'prefix_len', 'dtype', 'newline_idx', 'padding_idx')}
        return {'schema_version': SCHEMA, 'experiment_id': self.run.name, 'phase': phase, 'attempt_id': attempt,
                'logical_request_id': logical, 'request_id': logical, 'physical_execution_id': physical_id,
                'model_id': MODEL_ID, 'model_revision': REVISION, 'method_id': method, 'image_id': dialog['image_id'],
                'dialog_id': dialog['dialog_id'], 'turn_id': turn, 'question_id': str(question['question_id']),
                'question': question['question'], 'gold': question['answers'], 'prediction': result['answer'],
                'score': strict_score(result['answer'], question['answers'][0]), 'prompt': diag['prompt'],
                'history': diag['history'], 'history_entries': diag['history_entries'],
                'prompt_sha256': hashlib.sha256(diag['prompt'].encode()).hexdigest(), 'suffix_input_ids': suffix,
                'suffix_ids_sha256': canonical_hash(suffix), 'history_sha256': hashlib.sha256(diag['history'].encode()).hexdigest(),
                'config_sha256': self.config['_file_sha256'], 'manifest_sha256': self.config['_manifest_sha256'],
                'code_sha256': canonical_hash(self.config['source_sha256']), 'global_dialog_ordinal': ordinal,
                'method_order': list(method_order(ordinal)), 'method_order_position': method_order(ordinal).index(method),
                'request_started_at_s': result['request_started_at_s'], 'first_token_at_s': result['first_token_at_s'],
                'request_finished_at_s': result['request_started_at_s'] + result['request_e2e_ms'] / 1000,
                'ttft_ms': result['end_to_end_ttft_ms'], 'request_e2e_ms': result['request_e2e_ms'],
                'ssd_read_bytes': int(result.get('io', {}).get('bytes', 0)), 'ssd_preads': int(result.get('io', {}).get('preads', 0)),
                'N_content': n, 'k': k, 'content_kv_fraction': k / n, 'geometry': g,
                'budget_unit': 'visual_kv' if reduced else 'full_context', 'cache_hit': bool(hit and method != 'recompute'),
                'first_token_top2': result.get('first_token_top2'), 'scoring_head_count': 0, 'scoring_head_policy': None,
                'source_store': store_info if hit and method != 'recompute' else None,
                'provisioning_mode': 'FRESH_IMAGE_STREAMING', 'result': safe(value)}

    # --------------------------------------------------------------- image
    def run_image(self, dialogs, phase, audit_module):
        run = self.run; assert len({d['image_id'] for d in dialogs}) == 1
        image_id = dialogs[0]['image_id']; parent = run / phase / 'images' / image_id
        parent.mkdir(parents=True, exist_ok=True)
        if (parent / 'COMMITTED.json').exists(): raise FileExistsError('image already committed')
        attempt = f'attempt_{len(list(parent.glob("attempt_*"))) + 1:04d}'
        artifact = parent / attempt; artifact.mkdir(exist_ok=False)
        scratch = run / 'scratch' / phase / image_id / attempt
        atomic_json(artifact / 'capacity_before.json', capacity_guard(run))
        atomic_json(artifact / 'scratch_creation.json', {'path': str(scratch), 'ownership': run.name, 'previously_absent': not scratch.exists()})
        scratch.mkdir(parents=True, exist_ok=False)
        paths = {m: scratch / STORE_KIND[m] for m in STORE_KIND}; metas = {}; hashes = {}; receipts = {}; contexts = {}; store_info = {}
        rows = []; pending = []; diags = {}; t1_source = {}; success = False; g8_rows = []; hit_box = {}
        source_image_sha = sha(ROOT / dialogs[0]['image_path'])
        atomic_json(artifact / 'image_identity.json', {'image_id': image_id, 'image_path': str((ROOT / dialogs[0]['image_path']).resolve()), 'sha256': source_image_sha})
        keep_vz_diag = image_id in self.g8 or phase == 'correctness'
        try:
            for di, dialog in enumerate(dialogs):
                ordinal = self.ordinals[dialog['dialog_id']]; generated = {m: {} for m in METHODS}
                for turn in (1, 2, 3):
                    for method in method_order(ordinal):
                        M.assert_gpu_exclusive(); physical = str(uuid.uuid4())
                        logical = request_id(run.name, phase, method, dialog['dialog_id'], turn)
                        source = {'dialog_id': dialog['dialog_id'], 'method_id': method, 'physical_execution_id': physical, 'logical_request_id': logical}
                        append(artifact / 'attempts.jsonl', dict(source, turn_id=turn, status='STARTED', at_unix=time.time()))
                        factory = lambda m=method: prompt_factory(dialog, turn, m, generated[m])
                        capture = di == 0 and turn == 1 and method != 'recompute'
                        if turn == 1 or method == 'recompute':
                            result, diag = self.t1(method, dialog, factory, capture)
                            append(artifact / 'execution_events.jsonl', dict(source, turn_id=turn, status='EXECUTED', prediction=result['answer'],
                                   generated_token_ids=result['generated_token_ids'], ttft_ms=result['end_to_end_ttft_ms'], request_e2e_ms=result['request_e2e_ms']))
                            if capture:
                                kind = STORE_KIND[method]
                                start = time.perf_counter(); p = self.persist(method, result, diag, paths[method], image_id, source)
                                p.update(outer_helper_wall_ms=(time.perf_counter() - start) * 1000, source=source,
                                         capture_accounting='capture and exit materialization already within T1 request E2E; not added again')
                                receipts[method] = p; metas[method] = json.loads((paths[method] / 'meta.json').read_text())
                                hashes[method] = M.file_hashes(paths[method])
                                store_info[method] = {'path': str(paths[method]), 'meta_sha256': sha(paths[method] / 'meta.json'), 'source': source,
                                                      'file_sha256': hashes[method], 'physical_build_event_id': p['event_id']}
                                atomic_json(artifact / f'{kind}_persistence.json', p)
                                atomic_json(artifact / f'{kind}_meta.json', metas[method]); atomic_json(artifact / f'{kind}_hashes.json', hashes[method])
                                diags[method] = diag; t1_source[method] = result
                            else:
                                result.pop('captured_past_key_values', None)
                            saved = {a: b for a, b in diag.items() if a in ('prompt', 'history', 'history_entries')}
                            if di == 0 and turn == 1:
                                pending.append((method, result, saved, physical)); del diag, result
                                continue
                            diag = saved
                        else:
                            if method not in contexts:
                                contexts[method] = self.activate(method, paths[method], hashes[method])
                                atomic_json(artifact / f'{method}_activation.json', contexts[method].activation)
                            result, diag = self.hit(method, contexts[method], factory, image_id)
                            if di == 0 and method == 'vz_pruned_e2e' and keep_vz_diag:
                                hit_box[turn] = {'answer': result['answer'], 'first_token_id': result['first_token_id'],
                                                 'generated_token_ids': result['generated_token_ids'], 'first_logits': result['first_logits'],
                                                 'suffix_input_ids': diag['suffix_input_ids']}
                        meta = metas['vz_pruned_e2e' if method == 'vz_pruned_e2e' else 'ours_prefixstore_kv25']
                        row = self.make_row(result, diag, meta, method=method, dialog=dialog, turn=turn, phase=phase, attempt=attempt,
                                            physical_id=physical, ordinal=ordinal, store_info=store_info.get(method))
                        row['image_sha256'] = source_image_sha
                        append(artifact / 'raw.jsonl', row); rows.append(row); generated[method][turn] = row
                        append(artifact / 'attempts.jsonl', dict(source, turn_id=turn, status='COMPLETED', at_unix=time.time()))
                        del result, diag
                    if di == 0 and turn == 1:
                        assert set(metas) == set(STORE_KIND)
                        for method, result, diag, physical in pending:
                            meta = metas['vz_pruned_e2e' if method == 'vz_pruned_e2e' else 'ours_prefixstore_kv25']
                            row = self.make_row(result, diag, meta, method=method, dialog=dialog, turn=turn, phase=phase, attempt=attempt,
                                                physical_id=physical, ordinal=ordinal, store_info=store_info.get(method))
                            row['image_sha256'] = source_image_sha
                            append(artifact / 'raw.jsonl', row); rows.append(row); generated[method][1] = row
                            append(artifact / 'attempts.jsonl', {'physical_execution_id': physical, 'logical_request_id': row['logical_request_id'], 'status': 'COMPLETED', 'at_unix': time.time()})
                        pending.clear(); del result, diag
                        for m in STORE_KIND:
                            assert metas[m]['prefix_input_ids'] == metas['ours_prefixstore_kv25']['prefix_input_ids']
                            assert metas[m]['image_input_sha256'] == metas['ours_prefixstore_kv25']['image_input_sha256']
                        gates = self.image_gates(image_id, phase, t1_source, diags, receipts, paths)
                        atomic_json(artifact / 'gates.json', gates); append(run / 'llava' / 'gates.jsonl', {k: v for k, v in gates.items() if k != 'G2'} | {'G2_status': gates['G2']['status'], 'G2_failed': gates['G2']['failed_gates']})
                        if gates['status'] != 'PASS':
                            raise AssertionError(f"G2 gate failure {image_id}: {gates['G2']['failed_gates']} G2a={gates['G2a']} G2b={gates['G2b']['equal']} G2e={gates['G2e']}")
                        for m in list(t1_source):
                            t1_source[m].pop('captured_past_key_values', None); t1_source[m].pop('first_logits', None)
                        diags['ours_prefixstore_kv25'].pop('embeds', None); diags['ours_prefixstore_kv25'].pop('vision', None)
                        if not keep_vz_diag:
                            diags.clear()
                        gc.collect(); torch.cuda.empty_cache()
                if di == 0 and keep_vz_diag:
                    g8_rows = self.g8_pairs(dialog, {r['turn_id']: r for r in rows if r['method_id'] == 'vz_pruned_e2e' and r['dialog_id'] == dialog['dialog_id']},
                                            hit_box, diags['vz_pruned_e2e'], image_id)
                    atomic_json(artifact / 'g8_hit_vs_standalone.json', {'image_id': image_id, 'phase': phase, 'gated': False, 'pairs': g8_rows})
                    hit_box.clear(); diags.clear(); gc.collect(); torch.cuda.empty_cache()
            for m in STORE_KIND:
                assert M.file_hashes(paths[m]) == hashes[m], 'store mutated during requests'
            atomic_json(artifact / 'store_integrity_after.json', {'status': 'PASS', 'file_sha256': hashes})
            audit = audit_module.audit_image_rows(load_jsonl(artifact / 'raw.jsonl'), self.manifest, self.config, phase, image_id, artifact)
            audit['raw_sha256'] = sha(artifact / 'raw.jsonl'); atomic_json(artifact / 'independent_audit.json', audit)
            if audit['status'] != 'PASS': raise RuntimeError('INDEPENDENT_IMAGE_AUDIT_FAIL: ' + repr(audit))
            allowlist = M.file_hashes(scratch)
            atomic_json(artifact / 'cleanup_allowlist.json', {'scratch': str(scratch), 'file_sha256': allowlist, 'ownership_run': run.name})
            commit = {'status': 'COMMITTED', 'image_id': image_id, 'attempt_id': attempt, 'requests': len(rows), 'raw_sha256': audit['raw_sha256'],
                      'audit_sha256': sha(artifact / 'independent_audit.json'), 'config_sha256': self.config['_file_sha256'],
                      'manifest_sha256': self.config['_manifest_sha256'], 'first_successful_complete_image_adoption': True}
            commit['evidence_sha256'] = {p.name: sha(p) for p in artifact.iterdir() if p.is_file()}
            atomic_json(parent / 'COMMITTED.json', commit); success = True
        except BaseException as exc:
            atomic_json(artifact / 'failure.json', {'status': 'PARTIAL', 'error': repr(exc), 'traceback': traceback.format_exc(),
                                                    'completed_uncommitted_rows': len(rows), 'main_final_rows_from_this_attempt': 0})
            raise
        finally:
            for ctx in contexts.values(): ctx.close()
            contexts.clear(); diags.clear(); hit_box.clear(); gc.collect(); torch.cuda.empty_cache()
            if success:
                if phase == 'correctness':
                    atomic_json(artifact / 'cleanup_receipt.json', {'status': 'RETAINED', 'scratch': str(scratch), 'reason': 'G4 stores retained by contract'})
                else:
                    cleanup_committed(scratch, allowlist, audit, commit)
                    atomic_json(artifact / 'cleanup_receipt.json', {'status': 'PASS', 'removed_new_files': len(allowlist),
                                                                    'sealed_sha256': allowlist,
                                                                    'recipe': 'same frozen config+manifest; rerun this full image into a fresh run; never reuse its generated histories'})
        return len(rows)

    # --------------------------------------------------------------- phases
    def completed_images(self, phase, audit_module):
        done = {}
        for cp in sorted((self.run / phase / 'images').glob('*/COMMITTED.json')):
            doc = json.loads(cp.read_text()); folder = cp.parent / doc['attempt_id']; raw = folder / 'raw.jsonl'
            if doc['config_sha256'] != self.config['_file_sha256'] or doc['manifest_sha256'] != self.config['_manifest_sha256']:
                raise RuntimeError('resume config/manifest mismatch')
            if sha(raw) != doc['raw_sha256']: raise RuntimeError('committed raw hash mismatch')
            for name, digest in doc['evidence_sha256'].items():
                if sha(folder / name) != digest: raise RuntimeError('committed evidence changed')
            receipt = audit_module.audit_image_rows(load_jsonl(raw), self.manifest, self.config, phase, cp.parent.name, folder)
            if receipt['status'] != 'PASS': raise RuntimeError('resume independent audit failed: ' + repr(receipt))
            done[cp.parent.name] = doc
        return done

    def run_phase(self, phase, audit_module):
        run, manifest = self.run, self.manifest
        done = self.completed_images(phase, audit_module)
        if phase == 'main':
            ids = [d['dialog_id'] for d in manifest['dialogues']]
        elif phase == 'correctness':
            ids = [d['dialog_id'] for d in manifest['dialogues'] if d['image_id'] in set(manifest['correctness_images'])]
            first = {}
            for d in manifest['dialogues']:
                first.setdefault(d['image_id'], d['dialog_id'])
            ids = [i for i in ids if i in set(first.values())]      # source dialogue only
        else:
            ids = manifest[f'{phase}_dialogue_ids']
        keep = set(ids); groups = OrderedDict()
        for d in manifest['dialogues']:
            if d['dialog_id'] in keep: groups.setdefault(d['image_id'], []).append(d)
        expected = len(ids) * 9; completed = sum(x['requests'] for x in done.values())
        for image, ds in groups.items():
            if image in done: continue
            if (run / 'STOP_REQUESTED').exists(): raise RuntimeError('STOP_REQUESTED')
            completed += self.run_image(ds, phase, audit_module)
            progress = {'phase': phase, 'status': 'RUNNING', 'completed_final_requests': completed, 'expected_requests': expected,
                        'completed_images': len(done) + 1, 'image_id': image, 'at_unix': time.time()}
            done[image] = {'requests': len(ds) * 9}
            atomic_json(run / f'{phase}_progress.json', progress, replace=True); print(json.dumps(progress), flush=True)
        assert completed == expected, (completed, expected)
        receipt = {'status': 'PASS', 'requests': completed, 'expected_requests': expected, 'images': len(groups),
                   'independent_image_audit': 'PASS', 'config_sha256': self.config['_file_sha256']}
        path = run / phase / 'validation.json'
        if path.exists():
            if json.loads(path.read_text()) != receipt: raise RuntimeError('phase validation mismatch')
        else: atomic_json(path, receipt)
        return receipt

    # ---------------------------------------------------------- correctness
    def correctness(self, audit_module):
        """G4: (a) B main reproduction, (b) path equivalence, (c) memory identity, (d) hit vs standalone."""
        out = {'status': 'NOT RUN', 'samples': []}
        self.run_phase('correctness', audit_module)     # real image lifecycle, stores retained, G2 + G8 pairs inside
        b_rows = {}
        for image in self.manifest['correctness_images']:
            cp = B_MAIN / 'llava/main/images' / image / 'COMMITTED.json'
            doc = json.loads(cp.read_text())
            for r in load_jsonl(cp.parent / doc['attempt_id'] / 'raw.jsonl'):
                if r['method_id'] in ('recompute', 'ours_prefixstore_kv25'):
                    b_rows[(r['dialog_id'], r['turn_id'], r['method_id'])] = r
        evaluated = 0
        for image in self.manifest['correctness_images']:
            cp = self.run / 'correctness/images' / image / 'COMMITTED.json'; doc = json.loads(cp.read_text()); folder = cp.parent / doc['attempt_id']
            rows = {(r['dialog_id'], r['turn_id'], r['method_id']): r for r in load_jsonl(folder / 'raw.jsonl')}
            dialog = next(d for d in self.manifest['dialogues'] if d['image_id'] == image)
            did = dialog['dialog_id']
            sample = {'image_id': image, 'dialog_id': did}
            # (a) B main reproduction of Ours T1 and, given an identical T1, of the T2/T3 hits
            a = {}
            for turn in (1, 2, 3):
                ours, b = rows[(did, turn, 'ours_prefixstore_kv25')], b_rows[(did, turn, 'ours_prefixstore_kv25')]
                a[f't{turn}'] = {'prompt_equal': ours['prompt_sha256'] == b['prompt_sha256'], 'prediction_equal': ours['prediction'] == b['prediction'],
                                 'tokens_equal': ours['result']['generated_token_ids'] == b['result']['generated_token_ids'],
                                 'first_token_top2': ours['first_token_top2']}
            a['evaluable'] = a['t1']['prediction_equal']
            if a['evaluable']:
                evaluated += 1
                for turn in (2, 3):
                    assert a[f't{turn}']['prompt_equal'], ('history differs although T1 equal', image, turn)
                    assert a[f't{turn}']['tokens_equal'] and a[f't{turn}']['prediction_equal'], ('G4-a: hit does not reproduce B main', image, turn, a)
            sample['a_b_main_reproduction'] = a
            # (b) path equivalence on the same T1 prompt: real path vs AbortAfterEmbeds + pruned_generate
            factory = lambda: prompt_factory(dialog, 1, 'vz_pruned_e2e', {})
            real, diag = pruned_t1_request(self.runner, self.servers['vz'], ROOT / dialog['image_path'], factory, M.QA._combined_processor)
            enc_dev = self.runner.to_device(diag['enc_cpu'])
            embeds = AbortAfterEmbeds(self.runner).run(self.runner, enc_dev)
            box = []
            text_index = list(range(diag['v_token_start'] + diag['v_token_num'], int(diag['enc_cpu']['input_ids'].shape[1])))
            composed = pruned_generate(self.servers['vz'], embeds, diag['plan'], text_index, box)
            composed['first_logits'] = box[0]
            sample['b_path_equivalence'] = path_equivalence(real, composed)
            sample['b_path_equivalence']['real_vs_committed_t1_tokens_equal'] = real['generated_token_ids'] == rows[(did, 1, 'vz_pruned_e2e')]['result']['generated_token_ids']
            sample['b_path_equivalence']['hf_merge_prefix_embeds_equal'] = embeds_sha256(embeds[0, :diag['v_token_start'] + diag['v_token_num']]) == diag['prefix_embeds_sha256']
            assert sample['b_path_equivalence']['ok'] and sample['b_path_equivalence']['hf_merge_prefix_embeds_equal'], ('G4-b', image, sample['b_path_equivalence'])
            del real, diag, embeds, composed, box, enc_dev; gc.collect(); torch.cuda.empty_cache()
            # (c) in-memory BF16 structural identity (recorded in gates.json during the lifecycle)
            gates = json.loads((folder / 'gates.json').read_text())
            sample['c_memory_structural_identity'] = gates['G2c_memory_bf16']
            assert gates['G2c_memory_bf16']['ok'], ('G4-c', image)
            # (d) hit vs standalone with the same history (recorded)
            sample['d_hit_vs_standalone'] = json.loads((folder / 'g8_hit_vs_standalone.json').read_text())['pairs']
            sample['status'] = 'PASS'; out['samples'].append(sample)
            atomic_json(self.run / 'llava' / 'gpu_correctness.json', out, replace=True)
            print('G4', image, 'PASS', flush=True)
        assert evaluated >= 1, 'G4-a could not be evaluated on any sample'
        out['status'] = 'PASS'; out['g4a_evaluated_samples'] = evaluated
        atomic_json(self.run / 'llava' / 'gpu_correctness.json', out, replace=True)
        return out

    def close(self):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--protect', choices=['before', 'after'])
    ap.add_argument('--gpu', help='comma list of correctness,smoke,integration,main')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args(); run = Path(args.run).resolve()
    assert run.parent == ROOT / 'runs' and run.name.startswith('vz_pruned_e2e_mtgqa_')
    if args.freeze: return freeze(run)
    if args.protect: return protect(run, args.protect)
    config, manifest = verify_freeze(run)
    assert '\nOK\n' in (run / 'cpu_tests.log').read_text(), 'CPU tests must pass before GPU'
    assert (run / 'protection_before.json').exists()
    stages = args.gpu.split(','); assert set(stages) <= {'correctness', 'smoke', 'integration', 'main'}
    audit_module = helper('148_audit_vz_pruned_e2e_mtgqa.py', '_vz_independent_audit')
    (run / 'llava').mkdir(exist_ok=True)
    status_path = run / 'llava' / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    torch.set_num_threads(config['cpu_threads']); torch.manual_seed(SEED); np.random.seed(SEED); __import__('random').seed(SEED)
    M.assert_gpu_exclusive(); capacity_guard(run)
    exp = None
    try:
        exp = Experiment(run, config, manifest)
        atomic_json(run / 'llava' / f'process_{time.strftime("%Y%m%dT%H%M%S")}.json', {'started_unix': time.time(), 'stages': stages, 'threads': torch.get_num_threads(), 'warmup': exp.warmup, 'warmup_excluded': True})
        for phase in stages:
            if phase == 'correctness':
                exp.correctness(audit_module); status['G4 GPU CORRECTNESS'] = 'PASS'; atomic_json(status_path, status, replace=True); continue
            assert status.get('G4 GPU CORRECTNESS') == 'PASS', 'G4 must pass before smoke/integration/main'
            if phase == 'main':
                for prior in ('smoke', 'integration'):
                    v = json.loads((run / prior / 'validation.json').read_text())
                    assert v['status'] == 'PASS' and v['requests'] == len(manifest[f'{prior}_dialogue_ids']) * 9, f'{prior} must pass before main'
            if (run / phase).exists() and not args.resume: raise FileExistsError('use --resume; previous artifacts preserved')
            status[phase.upper()] = 'RUNNING'; atomic_json(status_path, status, replace=True)
            exp.run_phase(phase, audit_module)
            status[phase.upper()] = 'PASS'; atomic_json(status_path, status, replace=True)
    except BaseException as exc:
        err = traceback.format_exc(); print(err, flush=True)
        append(run / 'failures.jsonl', {'error': err, 'utc': time.time(), 'stages': stages})
        for key, value in list(status.items()):
            if value == 'RUNNING': status[key] = 'FAIL'
        if 'correctness' in stages and status.get('G4 GPU CORRECTNESS') != 'PASS': status['G4 GPU CORRECTNESS'] = 'FAIL'
        atomic_json(run / ('interruption_' + time.strftime('%Y%m%dT%H%M%S') + '.json'), {'status': 'PARTIAL', 'error': repr(exc), 'traceback': err,
                    'resume_command': f'{sys.executable} scripts/147_eval_vz_pruned_e2e_mtgqa.py --run {run} --gpu {args.gpu} --resume'})
        raise
    finally:
        atomic_json(status_path, status, replace=True); print(status, flush=True)


if __name__ == '__main__':
    main()

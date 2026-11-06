#!/usr/bin/env python3
"""PosShift dose-response diagnostic (LLaVA, GQA 40 images).

Contract: docs/posshift_dose_contract.md.  Reuses the frozen recompute-KV25
ablation code (scripts/134: Ours T1, prefix25 persistence, activation and
``hit_arm(force_start=...)``); the only new pieces are the shift schedule and
the per-request checks in mmimpress/posshift_dose.py.

Stages: --freeze, --protect before|after, --gpu smoke[,gqa].
"""
from __future__ import annotations
import argparse, gc, json, random, shutil, subprocess, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from mmimpress.prefix25 import sha
from mmimpress.recompute_kv25 import guarded_rmtree, sha_manifest, stat_inventory
from mmimpress.posshift_dose import bitwise_equal_logits, shift_schedule, tokens_equal


def load(name, file):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / file)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


M = load('recompute_ablation_existing', '134_eval_recompute_kv25_ablation.py')
write, append, exclusive, capacity = M.write, M.append, M.exclusive, M.capacity
REF_RUN, REF_GQA_SHA, INDEX_SHA, SEED = M.REF_RUN, M.REF_GQA_SHA, M.INDEX_SHA, M.SEED
PRIOR_RUN = ROOT / 'runs/recompute_kv25_ablation_20261102T061243Z'      # the 0% PosShift run (read only)
PRIOR_RAW_SHA = '0f6c839a8062c5a8d03ec48417263af626fe82481d5ca40b279deaf4506c34e9'
CONTRACT = ROOT / 'docs/posshift_dose_contract.md'
NEW_SOURCES = ['mmimpress/posshift_dose.py', 'scripts/150_eval_posshift_dose.py',
               'scripts/151_report_posshift_dose.py', 'tests/test_posshift_dose.py',
               'docs/posshift_dose_contract.md']
PROTECTED_RUNS = [REF_RUN, M.PREV_RUN, PRIOR_RUN]


# ------------------------------------------------------------------ freeze
def freeze(run):
    res = ROOT / 'results' / run.name
    run.mkdir(parents=True, exist_ok=False); res.mkdir(parents=True, exist_ok=False)
    assert sha(REF_RUN / 'gqa_manifest.json') == REF_GQA_SHA
    assert sha(ROOT / 'data/index.json') == INDEX_SHA
    assert sha(PRIOR_RUN / 'llava/gqa/raw_requests.jsonl') == PRIOR_RAW_SHA
    gqa = json.loads((REF_RUN / 'gqa_manifest.json').read_text())
    assert len(gqa['images']) == 40 and all(len(r['turns']) == 6 for r in gqa['images'])
    for r in gqa['images']:
        assert sha(r['image_path']) == r['image_sha256']
    shutil.copy2(REF_RUN / 'gqa_manifest.json', run / 'gqa_manifest.json')
    smoke = json.loads(json.dumps(gqa)); smoke['images'] = smoke['images'][:2]
    for r in smoke['images']:
        r['turns'] = r['turns'][:3]
    write(run / 'smoke_manifest.json', smoke)
    revision = (Path.home() / '.cache/huggingface/hub/models--llava-hf--llava-v1.6-vicuna-7b-hf/refs/main').read_text().strip()
    assert revision == 'c916e6cdcd760b4cecd1dd4907f84ac649f93b23'
    config = {'seed': SEED, 'greedy': True, 'max_new_tokens': 16, 'chunk_size': 64, 'ratio': .25,
              'deltas': [0, 1, 4, 16, 64, 256, 1024, 'max'], 'hit_order': 'base then ascending delta (fixed)',
              'store': 'ours (full T1 context, original positions), rebuilt per image, deleted after its hits',
              'llava': {'revision': revision, 'quantization': 'NF4 double BF16 compute', 'attention': 'eager', 'store_dtype': 'float16'},
              'reference_run': str(REF_RUN), 'reference_gqa_manifest_sha256': REF_GQA_SHA, 'index_sha256': INDEX_SHA,
              'prior_run': str(PRIOR_RUN), 'prior_raw_sha256': PRIOR_RAW_SHA, 'min_free_bytes': M.MIN_FREE,
              'bootstrap': {'unit': 'image', 'resamples': 10000, 'seed': SEED},
              'reading': {'collapse_em': .05, 'abrupt_max_delta': 4, 'graded_min_delta': 16},
              'workload_files': {n: sha(run / f'{n}_manifest.json') for n in ['gqa', 'smoke']},
              'contract_sha256': sha(CONTRACT)}
    write(run / 'config.json', config); write(res / 'config.json', config)
    shutil.copy2(CONTRACT, run / 'CONTRACT.md'); shutil.copy2(CONTRACT, res / 'CONTRACT.md')
    write(run / 'environment.json', {
        'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout,
        'packages': subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout})
    paths = [p for d in ['mmimpress', 'scripts', 'tests'] for p in (ROOT / d).rglob('*.py')
             if '__pycache__' not in p.parts and p.name != '151_report_posshift_dose.py']
    paths += [CONTRACT, run / 'gqa_manifest.json', run / 'smoke_manifest.json', run / 'config.json']
    write(run / 'gpu_freeze.json', {'files': {str(p.relative_to(ROOT)): sha(p) for p in paths}})
    print('FROZEN', run, flush=True)


# -------------------------------------------------------------- protection
def protect(run, stage):
    excl = [f'results/{run.name}'] + NEW_SOURCES
    roots = [ROOT / x for x in ['results', 'docs', 'mmimpress', 'scripts', 'tests', 'README.md']] + PROTECTED_RUNS
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
        write(run / 'protection_result.json', summary); write(ROOT / 'results' / run.name / 'protection_result.json', summary)
    print(json.dumps(summary, indent=1)[:4000], flush=True)


# ------------------------------------------------------------- experiment
class Dose(M.Ablation):
    def __init__(self, run):
        super().__init__(run)
        self.prior = {}
        for r in map(json.loads, (PRIOR_RUN / 'llava/gqa/raw_requests.jsonl').read_text().splitlines()):
            if r.get('status') == 'OK' and r['cache_hit']:
                self.prior[(r['arm'], r['image_id'], r['turn'])] = r

    def shifted_row(self, logical, phase, image, ti, turn, value, label, delta, start, extra):
        row = self.row(logical, phase, image, label, ti, turn, value)
        row.update({'delta': delta, 'text_position_start': start, **extra})
        assert row['position_check']['verified'], ('G-B position verification failed', logical, row['position_check'])
        return row

    def pilot(self, phase):
        manifest = json.loads((self.run / f'{phase}_manifest.json').read_text())
        out = self.base / phase; out.mkdir(exist_ok=False)
        raw = out / 'raw_requests.jsonl'; persist_file = out / 'persistence.jsonl'; gates = out / 'gates.jsonl'
        for ix, image in enumerate(manifest['images']):
            exclusive(); capacity()
            turn0 = image['turns'][0]
            value, capture = self.t1(image, turn0, 'ours', phase)
            value['actual_io'] = {'bytes': 0, 'preads': 0, 'calls': []}
            append(raw, self.row(f'llava/{phase}/{image["image_id"]}/ours/1', phase, image, 'ours', 0, turn0, value))
            path = self.stores / phase / image['image_id'] / 'ours'
            receipt, _, _ = self.persist_arm(capture, image, 'ours', path)
            del capture, value; gc.collect(); torch.cuda.empty_cache()
            ctx, activation = self.activate_arm(path, image, 'ours')
            receipt['activation'] = activation; receipt['phase'] = phase; append(persist_file, receipt)
            prefix_len = int(ctx.meta['prefix_len']); compact = self.text_start(ctx, 'posshift')
            schedule, dmax = shift_schedule(prefix_len, compact)
            g = {'image_id': image['image_id'], 'phase': phase, 'prefix_len': prefix_len, 'compact_start': compact,
                 'delta_max': dmax, 'delta_max_gt_1024': dmax > 1024, 'schedule': [(d, s) for d, s, _ in schedule],
                 'G_C_delta0_bitwise': [], 'G_D_base_reproduces_prior': [], 'shift_max_reproduces_prior': []}
            try:
                for ti in range(1, len(image['turns'])):
                    turn = image['turns'][ti]; tag = f'llava/{phase}/{image["image_id"]}'
                    base = self.hit_arm(ctx, path, image, turn, 'ours', phase, logits=True)
                    prior = self.prior.get(('ours', image['image_id'], ti + 1))
                    rep = tokens_equal(base, prior) if prior else None
                    g['G_D_base_reproduces_prior'].append(rep)
                    append(raw, self.shifted_row(f'{tag}/ours/{ti + 1}', phase, image, ti, turn, base, 'ours', None, None,
                                                 {'reproduces_prior': rep, 'io_key': [base['actual_io']['bytes'], base['actual_io']['preads']]}))
                    assert rep, ('G-D base path does not reproduce the prior Ours row', image['image_id'], ti + 1)
                    for delta, start, label in schedule:
                        v = self.hit_arm(ctx, path, image, turn, 'posshift', phase, logits=True, force_start=start)
                        extra = {'io_key': [v['actual_io']['bytes'], v['actual_io']['preads']]}
                        if delta == 0:
                            same = bitwise_equal_logits(v['all_logits'], base['all_logits'])
                            g['G_C_delta0_bitwise'].append(same); extra['delta0_bitwise_equal_base'] = same
                        if label == 'shift_max':
                            p = self.prior.get(('posshift', image['image_id'], ti + 1))
                            extra['reproduces_prior'] = tokens_equal(v, p) if p else None
                            g['shift_max_reproduces_prior'].append(extra['reproduces_prior'])
                        extra['first_token_equals_base'] = v['first_token_id'] == base['first_token_id']
                        append(raw, self.shifted_row(f'{tag}/{label}/{ti + 1}', phase, image, ti, turn, v, label, delta, start, extra))
                        assert extra['io_key'] == [base['actual_io']['bytes'], base['actual_io']['preads']], 'hits read different bytes'
                        if delta == 0:
                            assert same, ('G-C delta=0 logits differ from the base path', image['image_id'], ti + 1)
                        del v
                    del base
            except Exception:
                append(raw, {'logical_request_id': f'llava/{phase}/{image["image_id"]}', 'status': 'FAIL', 'error': traceback.format_exc()})
                g['status'] = 'FAIL'; append(gates, g); raise
            g['status'] = 'PASS'; append(gates, g)
            ctx.close(); self.close_contexts()
            append(self.base / 'cleanup.jsonl', {'stage': 'before', 'path': str(path), 'sealed_files': len(receipt['files'])})
            guarded_rmtree(path, self.stores)
            append(self.base / 'cleanup.jsonl', {'stage': 'after', 'path': str(path), 'exists': path.exists()})
            print('llava', phase, ix + 1, len(manifest['images']), 'complete', 'dmax', dmax, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--protect', choices=['before', 'after'])
    ap.add_argument('--gpu', help='comma list of smoke,gqa')
    args = ap.parse_args(); run = Path(args.run).resolve()
    assert run.parent == ROOT / 'runs' and run.name.startswith('posshift_dose_')
    if args.freeze:
        return freeze(run)
    if args.protect:
        return protect(run, args.protect)
    frozen = json.loads((run / 'gpu_freeze.json').read_text())
    for rel, digest in frozen['files'].items():
        assert sha(ROOT / rel) == digest, f'source changed after freeze: {rel}'
    assert '\nOK\n' in (run / 'cpu_tests.log').read_text()
    assert (run / 'protection_before.json').exists()
    assert sha(PRIOR_RUN / 'llava/gqa/raw_requests.jsonl') == PRIOR_RAW_SHA
    stages = args.gpu.split(','); assert set(stages) <= {'smoke', 'gqa'}
    exclusive(); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)
    status_path = run / 'llava' / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    exp = None
    try:
        exp = Dose(run)
        sample = json.loads((run / 'smoke_manifest.json').read_text())['images'][0]
        val, cap = exp.t1(sample, sample['turns'][0], 'ours', 'smoke'); del val, cap
        write(exp.base / f'warmup_{"_".join(stages)}.json', {'excluded': True, 'normal_requests': 1})
        for phase in stages:
            if phase == 'gqa':
                assert status.get('SMOKE') == 'PASS', 'smoke must pass before gqa'
            status[phase.upper()] = 'RUNNING'; write(status_path, status)
            exp.pilot(phase)
            status[phase.upper()] = 'PASS'; write(status_path, status)
    except Exception:
        err = traceback.format_exc(); print(err, flush=True)
        append(run / 'failures.jsonl', {'error': err, 'utc': time.time(), 'stages': stages})
        for key, value in list(status.items()):
            if value == 'RUNNING':
                status[key] = 'FAIL'
        raise
    finally:
        (run / 'llava').mkdir(exist_ok=True); write(status_path, status)
        if exp:
            exp.close_contexts()
        print(status, flush=True)


if __name__ == '__main__':
    main()

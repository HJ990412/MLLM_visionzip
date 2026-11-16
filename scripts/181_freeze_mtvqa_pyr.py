#!/usr/bin/env python3
"""Phase 0 of the LLaVA-NeXT PYR-beta-raw packed MT-VQA-25k single-arm run
(contract docs/llava_next_mtvqa_pyr_contract.md).

Stages
  --stage prepare   : create the run, bind the pre-task source inventory, build the v3 reference
                      index (BLOCKED_REFERENCE unless all 25,000 main images are readable), copy the
                      v3 manifests and the frozen vectors, run every CPU identity check, write
                      per-phase configs / storage plans and the environment record
  --stage freeze    : hash sources + artifacts + contract into experiment_freeze.json
  --stage refreeze --amendment FILE : keep the failed attempt, record the amendment, freeze again

Nothing under existing runs/, results/ or data/ is written.
"""
import argparse, csv, json, os, shutil, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from mmimpress.mtvqa25k.common import read, dump, sha, digest
from mmimpress.mtvqa25k_packed import adapter as AD, checks as CK, reference as REF

CONTRACT = ROOT / 'docs/llava_next_mtvqa_pyr_contract.md'
PROMPT = ROOT / 'papers/llava_next_mtvqa_pyr_prompt.md'
PHASE_METHODS = {'smoke': [AD.MAIN_METHOD], 'pilot40': [AD.MAIN_METHOD], 'main': [AD.MAIN_METHOD], 'gate_u_pilot40': [AD.GATE_METHOD]}
PHASE_MANIFEST = {'smoke': 'smoke', 'pilot40': 'pilot40', 'main': 'main', 'gate_u_pilot40': 'pilot40'}
GIB = 1 << 30
STORAGE_PLAN = {'minimum_reserve_bytes': 30 * GIB, 'peak_working_set_bytes': 2 * GIB, 'largest_image_bundle_bound_bytes': 1 * GIB,
                'maximum_new_payload_bytes': 4 * GIB,
                'note': 'one packed store (<= ~0.41 GB at N=2880) per image, deleted by exact SHA allowlist after audit+commit; 30 GiB floor; '
                        'main additionally requires the pilot40-measured admission'}
NEW_SOURCES = ('mmimpress/mtvqa25k_packed/__init__.py', 'mmimpress/mtvqa25k_packed/adapter.py', 'mmimpress/mtvqa25k_packed/checks.py',
               'mmimpress/mtvqa25k_packed/reference.py', 'mmimpress/mtvqa25k_packed/analysis.py',
               'scripts/181_freeze_mtvqa_pyr.py', 'scripts/182_validate_mtvqa_pyr.py', 'scripts/183_run_mtvqa_pyr.py', 'scripts/184_analyze_mtvqa_pyr.py',
               'tests/test_mtvqa_pyr_adapter.py', 'tests/test_mtvqa_pyr_analysis.py', 'docs/llava_next_mtvqa_pyr_contract.md')


def run_cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=60).stdout
    except Exception as exc:
        return 'UNAVAILABLE: ' + repr(exc)


def environment_record(run):
    """Not a gate: disk, SSD identity/temperature, other GPU / heavy I/O processes."""
    usage = shutil.disk_usage(run)
    nvme = Path('/sys/class/nvme/nvme0')
    temps = {}
    for p in sorted(nvme.glob('hwmon*/temp*_input')):
        try:
            temps[p.name] = int(p.read_text()) / 1000
        except Exception:
            pass
    io_heavy = run_cmd(['bash', '-lc', "ps -eo pid,pcpu,pmem,comm --sort=-pcpu | head -15"])
    return {'recorded_unix': time.time(), 'filesystem': run_cmd(['findmnt', '-T', str(run), '-o', 'SOURCE,FSTYPE,TARGET,OPTIONS', '-n']).strip(),
            'v3_run_filesystem': run_cmd(['findmnt', '-T', str(ROOT / REF.V3_RUN), '-o', 'SOURCE', '-n']).strip(),
            'disk_total_bytes': usage.total, 'disk_free_bytes': usage.free,
            'ssd_model': (nvme / 'model').read_text().strip() if (nvme / 'model').exists() else 'NOT_MEASURED',
            'ssd_firmware': (nvme / 'firmware_rev').read_text().strip() if (nvme / 'firmware_rev').exists() else 'NOT_MEASURED',
            'ssd_temperature_c_hwmon': temps or 'NOT_MEASURED', 'ssd_smart_percentage_used': 'NOT_MEASURED (nvme-cli/smartctl need root)',
            'nvidia_smi': run_cmd(['nvidia-smi']), 'gpu_compute_apps': run_cmd(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader']),
            'top_cpu_processes': io_heavy, 'loadavg': Path('/proc/loadavg').read_text().strip(), 'role': 'environment record, not a verdict input'}


def artifact_inventory():
    """(size, mtime_ns) of every existing file under runs/ and results/ except this task's own directories."""
    out = {}
    for top in ('runs', 'results'):
        for dirpath, dirnames, filenames in os.walk(ROOT / top):
            dirnames[:] = [d for d in dirnames if not d.startswith(AD.RUN_PREFIX)]
            for name in filenames:
                p = Path(dirpath) / name
                try:
                    st = p.lstat()
                except FileNotFoundError:
                    continue
                out[str(p.relative_to(ROOT))] = [st.st_size, st.st_mtime_ns]
    return out


def source_set():
    paths = list((ROOT / 'mmimpress').rglob('*.py')) + list((ROOT / 'scripts').glob('*.py')) + list((ROOT / 'tests').glob('*.py'))
    paths += list((ROOT / 'mmimpress/mtvqa25k/vendor').glob('*.original')) + list((ROOT / 'mmimpress/mtvqa25k/vendor').glob('*.json'))
    paths += [CONTRACT] if CONTRACT.exists() else []      # freeze() requires it through NEW_SOURCES
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def check_existing_sources(run):
    """Every pre-task source is byte-identical to the before-inventory; only the declared new files were added."""
    before = read(run / 'protection/source_before.json')['files']
    # The prompt file's trailing "결과 기록" section is filled in after the final report by design; it is not a source.
    changed = [n for n, h in before.items() if n != str(PROMPT.relative_to(ROOT)) and (not (ROOT / n).is_file() or sha(ROOT / n) != h)]
    current = set(source_set()) | {str(p.relative_to(ROOT)) for p in (ROOT / 'docs').glob('*.md')} | {str(p.relative_to(ROOT)) for p in (ROOT / 'scripts').glob('*.sh')}
    added = sorted(n for n in current if n not in before)
    unexpected = [n for n in added if n not in NEW_SOURCES]
    return {'status': 'PASS' if not changed and not unexpected else 'FAIL', 'existing_files': len(before), 'changed_existing': changed,
            'added_new_files': added, 'unexpected_new_files': unexpected}


def prepare(run, source_before):
    if not run.name.startswith(AD.RUN_PREFIX) or run.parent != ROOT / 'runs':
        raise ValueError('run must be runs/' + AD.RUN_PREFIX + '<ts>')
    results = ROOT / 'results' / run.name
    run.mkdir(exist_ok=False); results.mkdir(exist_ok=False)
    (run / 'protection').mkdir()
    shutil.copy2(source_before, run / 'protection/source_before.json')
    t0 = time.time(); inv = artifact_inventory()
    dump(run / 'protection/artifacts_before.json', {'created_unix': time.time(), 'files': len(inv), 'policy': 'size + mtime_ns of every file under runs/ and results/ outside this task',
                                                    'inventory_sha256': digest(inv), 'seconds': time.time() - t0})
    with (run / 'protection/artifacts_before.inventory.json').open('x') as f:
        json.dump(inv, f)
    shutil.copy2(PROMPT, run / 'REQUEST.md')
    dump(run / 'phase0/environment_before.json', environment_record(run))

    # ---- model / environment identity (precondition of the cross-run accuracy reuse)
    me = CK.model_environment(); dump(run / 'phase0/model_environment_check.json', me)
    if me['status'] != 'PASS':
        raise RuntimeError('MODEL_ENVIRONMENT_DIFFERS: ' + repr(me['differences'][:5]))
    for name in ('model_freeze.json', 'environment.json'):
        shutil.copy2(ROOT / REF.V3_RUN / name, run / name)
        assert sha(run / name) == sha(ROOT / REF.V3_RUN / name)

    # ---- manifests: v3 files, byte for byte
    for phase, src in PHASE_MANIFEST.items():
        (run / 'llava' / phase).mkdir(parents=True)
        shutil.copy2(ROOT / REF.V3_RUN / 'llava' / src / 'manifest.json', run / 'llava' / phase / 'manifest.json')
        dump(run / 'llava' / phase / 'storage_plan.json', STORAGE_PLAN)

    # ---- v3 reference index: every committed journal must open and match its commit SHA
    (run / 'reference').mkdir()
    ref = {}
    for phase in ('smoke', 'pilot40', 'main'):
        manifest = read(run / 'llava' / phase / 'manifest.json')
        ids = [d['image_id'] for d in manifest['dialogues']]
        t0 = time.time()
        summary = REF.build_index(ROOT / REF.V3_RUN / 'llava' / phase / phase / 'images', ids, run / 'reference' / f'v3_{phase}_index.jsonl.gz', full=phase != 'main')
        index = REF.load_index(run / 'reference' / f'v3_{phase}_index.jsonl.gz')
        check = CK.reference_summary(index, manifest)
        ref[phase] = {**summary, **check, 'seconds': time.time() - t0, 'index_file_sha256': sha(run / 'reference' / f'v3_{phase}_index.jsonl.gz')}
        print('reference', phase, check['status'], summary['images_readable'], '/', summary['images_requested'], f'{time.time() - t0:.0f}s', flush=True)
        if phase == 'main':
            main_index = index
    blocked = ref['main']['error_count'] or ref['main']['missing_count'] or ref['main']['images_readable'] != 25000
    ref['status'] = 'BLOCKED_REFERENCE' if blocked else ('PASS' if all(ref[p]['status'] == 'PASS' for p in ('smoke', 'pilot40', 'main')) else 'FAIL')
    ref['fields'] = 'prediction, generated_token_ids, score, N_content, ssd_read_bytes, ttft/e2e, prompt/suffix/history SHA256 (pilot40+smoke: full prompt and suffix ids)'
    dump(run / 'reference/reference_check.json', ref)
    if ref['status'] != 'PASS':
        raise RuntimeError(ref['status'] + ': v3 request-level raw is not fully readable')

    # ---- data / scorer identity
    di = CK.data_identity(run); dump(run / 'phase0/data_identity.json', di)
    if di['status'] != 'PASS':
        raise RuntimeError('DATA_IDENTITY_FAIL: ' + repr(di['failures'][:5]))
    sc = CK.scorer_validation(); dump(run / 'phase0/scorer_validation.json', sc)
    if sc['status'] != 'PASS':
        raise RuntimeError('SCORER_VALIDATION_FAIL: ' + repr(sc['failures'][:5]))

    # ---- vectors: copied, never recomputed
    (run / 'vectors').mkdir()
    for vec in CK.RATIO_SHA:
        shutil.copy2(ROOT / CK.VECTOR_SOURCE / f'layer_budget_llava_{vec}.json', run / 'vectors' / f'layer_budget_llava_{vec}.json')
    ve = CK.vectors(run); dump(run / 'vectors/vectors_sha256.json', ve)
    if ve['status'] != 'PASS':
        raise RuntimeError('VECTOR_IDENTITY_FAIL: ' + repr(ve['failures']))

    # ---- k_l table on the real N of all 25,000 images + payload bytes vs v3 [B]
    kt, per_image = CK.k_tables(run, main_index)
    dump(run / 'phase0/k_tables.json', kt)
    with (run / 'phase0/k_tables_per_image.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0])); w.writeheader(); w.writerows(per_image)
    with (run / 'phase0/k_tables.csv').open('x', newline='') as f:
        w = csv.writer(f); w.writerow(['N', 'images', 'budget', 'sum_k', 'min_k', 'max_k', 'zero_layers'] + [f'k_{i:02d}' for i in range(32)])
        for t in kt['per_N']:
            w.writerow([t['N'], t['images'], t['budget'], t['sum'], t['min'], t['max'], t['zero_layers']] + t['k_per_layer'])
    if kt['status'] != 'PASS':
        raise RuntimeError('BLOCKED k_l / payload-byte check: ' + repr(kt['failures'][:5]))

    # ---- per-phase configs
    from mmimpress.layer_budget import load_budget_document
    v3_settings = read(ROOT / REF.V3_RUN / 'llava/model_settings.json')
    for phase, methods in PHASE_METHODS.items():
        AD.check_methods(methods)
        budgets = {}
        for m in methods:
            vec = AD.vector_of(m); p = run / 'vectors' / f'layer_budget_llava_{vec}.json'
            budgets[vec] = AD.budget_entry(load_budget_document(p), p.relative_to(ROOT))
        config = {'schema_version': AD.SCHEMA, 'model': 'llava', 'model_id': 'llava-hf/llava-v1.6-vicuna-7b-hf',
                  'revision': 'c916e6cdcd760b4cecd1dd4907f84ac649f93b23', 'phase': phase, 'methods': methods, 'layer_budgets': budgets,
                  'seed': 1234, 'greedy': True, 'max_new_tokens': 16, 'batch_size': 1, 'cpu_threads': 2,
                  'dtype_backend': v3_settings['dtype_backend'], 'scorer_revision': v3_settings['scorer_revision'],
                  'dataset_config_sha256': CK.DATASET_CONFIG_SHA, 'ordered_workload_sha256': CK.WORKLOAD_SHA,
                  'history_policy': 'method_local_generated', 'packed_storage_policy': 'prefix25_packed',
                  'read_syscall': 'os.preadv into pinned host buffer, one call per hit',
                  'request_path': 'scripts/' + AD.BASE_SCRIPT + ' bound by mmimpress/mtvqa25k_packed/adapter.py',
                  'raw_archive': 'lossless gzip after full audit, original and compressed full SHA bound by atomic commit',
                  'v3_model_settings_sha256': sha(ROOT / REF.V3_RUN / 'llava/model_settings.json'),
                  'role': 'G-det-U gate only (40 images, never a main denominator)' if phase == 'gate_u_pilot40' else 'PYR-beta-raw packed single arm'}
        dump(run / 'llava' / phase / 'config.json', config)
    prot = check_existing_sources(run); dump(run / 'phase0/existing_source_preservation.json', prot)
    print(json.dumps({'prepare': 'PASS', 'run': str(run), 'reference': ref['status'], 'data': di['status'], 'scorer': sc['status'], 'vectors': ve['status'],
                      'k_tables': kt['status'], 'min_k': kt['min_k_overall'], 'zero_layers': kt['zero_layers'], 'payload_bytes_different': kt['payload_bytes_vs_v3_B']['different'],
                      'model_env': me['status'], 'existing_sources': prot['status']}), flush=True)


def freeze(run, amendment=None):
    target = run / 'experiment_freeze.json'
    if target.exists():
        if amendment is None:
            raise FileExistsError('freeze exists; use --stage refreeze with an amendment record')
        n = len(list(run.glob('freeze_amendment_*.json'))) + 1
        record = read(amendment)
        record.update(previous_freeze_sha256=sha(target), amendment_index=n, recorded_unix=time.time())
        os.replace(target, run / f'experiment_freeze_superseded_{n:02d}.json')
        dump(run / f'freeze_amendment_{n:02d}.json', record)
    prot = check_existing_sources(run)
    if prot['status'] != 'PASS':
        raise RuntimeError('existing source changed or undeclared new file: ' + repr(prot))
    missing = [n for n in NEW_SOURCES if not (ROOT / n).is_file()]
    if missing:
        raise RuntimeError('declared new source missing: ' + repr(missing))
    shutil.copy2(CONTRACT, run / 'CONTRACT.md') if not (run / 'CONTRACT.md').exists() else None
    if sha(run / 'CONTRACT.md') != sha(CONTRACT):
        os.replace(run / 'CONTRACT.md', run / f'CONTRACT_superseded_{int(time.time())}.md'); shutil.copy2(CONTRACT, run / 'CONTRACT.md')
    sources = source_set()
    frozen = run / 'source_frozen'
    for name in NEW_SOURCES + (f'scripts/{AD.BASE_SCRIPT}', 'scripts/156_eval_layer_budget_hold.py', 'scripts/main_b_support.py'):
        dest = frozen / name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / name, dest)
    names = ['CONTRACT.md', 'REQUEST.md', 'model_freeze.json', 'environment.json', 'protection/source_before.json', 'reference/reference_check.json',
             'phase0/data_identity.json', 'phase0/scorer_validation.json', 'phase0/k_tables.json', 'phase0/model_environment_check.json',
             'vectors/vectors_sha256.json'] + [f'vectors/layer_budget_llava_{v}.json' for v in CK.RATIO_SHA]
    names += [f'reference/v3_{p}_index.jsonl.gz' for p in ('smoke', 'pilot40', 'main')]
    for phase in PHASE_METHODS:
        names += [f'llava/{phase}/manifest.json', f'llava/{phase}/config.json', f'llava/{phase}/storage_plan.json']
    artifacts = {n: sha(run / n) for n in names}
    dump(target, {'source_sha256': sources, 'artifact_sha256': artifacts, 'created_unix': time.time(), 'contract_sha256': sha(CONTRACT),
                  'ordered_workload_sha256': CK.WORKLOAD_SHA, 'dataset_config_sha256': CK.DATASET_CONFIG_SHA, 'ratio_sha256': CK.RATIO_SHA,
                  'amendments': len(list(run.glob('freeze_amendment_*.json'))),
                  'main_admission': 'requires VG1-VG6 PASS + smoke + pilot40 + measured resource admission; freeze alone is not PASS'})
    print(json.dumps({'freeze': 'PASS', 'sources': len(sources), 'artifacts': len(artifacts), 'contract_sha256': sha(CONTRACT),
                      'freeze_sha256': sha(target)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--stage', choices=('prepare', 'freeze', 'refreeze'), required=True)
    p.add_argument('--source-before', type=Path); p.add_argument('--amendment', type=Path)
    a = p.parse_args(); run = a.run.resolve()
    if a.stage == 'prepare':
        prepare(run, a.source_before)
    elif a.stage == 'freeze':
        freeze(run)
    else:
        freeze(run, a.amendment)

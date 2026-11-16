#!/usr/bin/env python3
"""Managed image-atomic execution of the packed layer-budget arm on MT-VQA-v2-reconstructed-25k
(contract docs/llava_next_mtvqa_pyr_contract.md).  Same lifecycle as scripts/138_run_mtvqa25k.py
(freeze / environment / admission checks, per-image run_image, lossless journal archive, resume),
with scripts/164 bound through mmimpress/mtvqa25k_packed/adapter.py instead of scripts/108.

  --phase smoke | pilot40 | gate_u_pilot40 | main [--resume]
  --phase admit      : pilot40-measured resource admission for main (no GPU)
"""
import argparse, gzip, hashlib, json, math, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from mmimpress.mtvqa25k.common import read, dump, sha, digest
from mmimpress.mtvqa25k.adapter import remove_archived_duplicates
from mmimpress.mtvqa25k_packed import adapter as AD, checks as CK

PHASES = ('smoke', 'pilot40', 'gate_u_pilot40', 'main')
CPU_GATES = ('vg1_data_scoring_identity', 'vg2_budget', 'vg3_floor_propagation', 'vg6_regression')
GPU_GATE = 'vg4_packed_path'
INTERIM_IMAGES = 2500
ENV_LOG_EVERY = 2500
GIB = 1 << 30


def status(run, phase, state, **extra):
    dump(run / 'llava_STATUS.json', {'model': 'llava', 'phase': phase, 'status': state, 'pid': os.getpid(), 'heartbeat_unix': time.time(),
                                    'resume_command': f'{sys.executable} scripts/183_run_mtvqa_pyr.py --run {run} --phase {phase} --resume', **extra}, replace=True)


def verify_freeze(run):
    freeze = read(run / 'experiment_freeze.json')
    for name, h in freeze['source_sha256'].items():
        if sha(ROOT / name) != h:
            raise RuntimeError('source changed after freeze: ' + name)
    for name, h in freeze['artifact_sha256'].items():
        if sha(run / name) != h:
            raise RuntimeError('configuration/membership changed: ' + name)
    if sha(ROOT / 'docs/llava_next_mtvqa_pyr_contract.md') != freeze['contract_sha256']:
        raise RuntimeError('contract changed after freeze')
    return freeze


def gate_pass(run, name, freeze_sha):
    path = run / 'gates' / f'{name}.json'
    if not path.exists():
        return False
    g = read(path)
    return g.get('status') == 'PASS' and g.get('source_freeze_sha256') == freeze_sha


def check_admission(run, phase):
    freeze_sha = sha(run / 'experiment_freeze.json')
    need = list(CPU_GATES) + [GPU_GATE]
    if phase in ('smoke', 'pilot40', 'main'):
        need.append('vg5_g_det_U')
    missing = [g for g in need if not gate_pass(run, g, freeze_sha)]
    if missing:
        raise RuntimeError('gate not PASS under the current freeze: ' + repr(missing))
    predecessors = {'gate_u_pilot40': [], 'smoke': [], 'pilot40': ['smoke'], 'main': ['smoke', 'pilot40']}[phase]
    for prior in predecessors:
        receipt = read(run / 'llava' / prior / 'COMPLETE.json')
        if receipt['status'] != 'PASS' or receipt['source_freeze_sha256'] != freeze_sha:
            raise RuntimeError('phase prerequisite not PASS under the current freeze: ' + prior)
        if not gate_pass(run, f'g_det_T1_{prior}', freeze_sha):
            raise RuntimeError('T1 token identity with v3 not PASS: ' + prior)
    if phase == 'main':
        p = read(run / 'llava/main_resource_admission.json')
        if p['status'] != 'PASS' or p['pilot40_receipt_sha256'] != sha(run / 'llava/pilot40/COMPLETE.json'):
            raise RuntimeError('pilot-derived resource admission missing')
        if shutil.disk_usage(run).free < p['required_free_bytes']:
            raise RuntimeError('BLOCKED_STORAGE: pilot-based raw/validation reserve not available')


def committed(phasepath, phase):
    done = {}
    for cp in sorted((phasepath / phase / 'images').glob('*/COMMITTED.json')):
        c = read(cp); folder = cp.parent / c['attempt_id']
        for name, h in c['evidence_sha256'].items():
            if sha(folder / name) != h:
                raise RuntimeError('committed evidence changed: ' + str(folder / name))
        archive = c.get('journal_archives', {}).get('raw.jsonl')
        if archive:
            h = hashlib.sha256()
            with gzip.open(folder / archive['path'], 'rb') as f:
                for b in iter(lambda: f.read(1 << 20), b''):
                    h.update(b)
            if h.hexdigest() != c['raw_sha256']:
                raise RuntimeError('uncompressed raw SHA differs')
        done[cp.parent.name] = c
        remove_archived_duplicates(cp.parent)
    return done


def config_for(run, phase, freeze):
    phasepath = run / 'llava' / phase
    manifest = read(phasepath / 'manifest.json')
    config = read(phasepath / 'config.json')
    AD.check_methods(config['methods'])
    config.update(source_sha256=freeze['source_sha256'], _file_sha256=sha(phasepath / 'config.json'), _manifest_sha256=sha(phasepath / 'manifest.json'))
    return phasepath, manifest, config


class Window:
    """Environment-drift accumulators (not verdict inputs): Turn-1 TTFT and packed read throughput."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.t1 = []; self.hit_ttft = []; self.read_bytes = 0; self.read_ms = 0.0; self.hits = 0

    def observe(self, row):
        if row['turn_id'] == 1:
            self.t1.append(row['ttft_ms'])
        else:
            self.hit_ttft.append(row['ttft_ms']); self.hits += 1
            self.read_bytes += row['ssd_read_bytes']; self.read_ms += float(row['result'].get('packed_read_ms') or 0.0)

    def summary(self):
        mean = lambda v: sum(v) / len(v) if v else None
        return {'T1_requests': len(self.t1), 'T1_ttft_mean_ms': mean(self.t1), 'followup_requests': self.hits, 'followup_ttft_mean_ms': mean(self.hit_ttft),
                'packed_read_MB_per_hit': self.read_bytes / 1e6 / self.hits if self.hits else None,
                'packed_read_ms_per_hit': self.read_ms / self.hits if self.hits else None,
                'packed_read_MB_per_s': (self.read_bytes / 1e6) / (self.read_ms / 1e3) if self.read_ms else None}


def machine_state(run):
    temps = {}
    for p in sorted(Path('/sys/class/nvme/nvme0').glob('hwmon*/temp*_input')):
        try:
            temps[p.name] = int(p.read_text()) / 1000
        except Exception:
            pass
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:
        gpu = 'UNAVAILABLE ' + repr(exc)
    return {'disk_free_bytes': shutil.disk_usage(run).free, 'ssd_temperature_c': temps or 'NOT_MEASURED', 'gpu_temp_util_mem': gpu,
            'loadavg': Path('/proc/loadavg').read_text().strip()}


def folder_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def interim_report(run):
    """One pre-registered interim report on the first 2,500 committed main images.  Runs between images (no timed request in flight)."""
    out = ROOT / 'results' / run.name / 'INTERIM_2500.md'
    if out.exists():
        return
    t0 = time.time()
    p = subprocess.run([sys.executable, str(ROOT / 'scripts/184_analyze_mtvqa_pyr.py'), '--run', str(run), '--stage', 'interim'], capture_output=True, text=True)
    dump(run / f'interim_invocation_{time.time_ns()}.json', {'returncode': p.returncode, 'seconds': time.time() - t0, 'stdout_tail': p.stdout[-2000:], 'stderr_tail': p.stderr[-4000:],
                                                             'policy': 'interim report never changes arms, settings or rules; execution continues regardless of its content'})
    print(json.dumps({'interim_report': 'WRITTEN' if out.exists() else 'FAILED', 'seconds': time.time() - t0}), flush=True)


def run_phase(run, phase, resume):
    import torch, numpy as np, random
    torch.set_num_threads(2); torch.manual_seed(1234); random.seed(1234); np.random.seed(1234)
    freeze = verify_freeze(run)
    me = CK.model_environment()
    if me['status'] != 'PASS':
        raise RuntimeError('model/processor/environment changed: ' + repr(me['differences'][:5]))
    check_admission(run, phase)
    phasepath, manifest, config = config_for(run, phase, freeze)
    if (phasepath / 'COMPLETE.json').exists():
        raise FileExistsError('completed phase is immutable')
    done = committed(phasepath, phase)
    if done and not resume:
        raise RuntimeError('existing committed images; pass --resume')
    if any(c['config_sha256'] != config['_file_sha256'] or c['manifest_sha256'] != config['_manifest_sha256'] for c in done.values()):
        raise RuntimeError('resume freeze mismatch')
    per_image = 3 * len(config['methods'])
    target = per_image * len(manifest['dialogues'])
    status(run, phase, 'RUNNING', adopted_requests=sum(c['requests'] for c in done.values()), target_requests=target)
    M = AD.bind_packed(run.name)
    window = Window(); M._row_observer = window.observe
    M.assert_gpu_exclusive()
    runner = M.V.load_runner()
    from mmimpress.serve import Server
    server = Server(runner, max_new_tokens=16)
    M.QA._warmup(runner, server)
    fingerprint = {'model_id': M.MODEL_ID, 'revision': M.REVISION, 'device': str(runner.model.device), 'backend': runner.attn, 'dtype': str(runner.model.dtype),
                   'torch_threads': torch.get_num_threads(), 'methods': config['methods'], 'base_script': AD.BASE_SCRIPT, 'schema': M.SCHEMA}
    dump(phasepath / f'runtime_{time.time_ns()}.json', fingerprint)
    dump(phasepath / f'phase_start_{time.time_ns()}.json', {'started_unix': time.time(), 'resume': bool(resume), 'already_committed_images': len(done), **machine_state(run)})
    count = sum(c['requests'] for c in done.values()); started = time.time(); images_done = len(done)
    frozen_paths = [ROOT / p for p in freeze['source_sha256']] + [run / p for p in freeze['artifact_sha256']]
    signatures = {p: (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in frozen_paths}
    image_log = phasepath / 'image_log.jsonl'; env_log = phasepath / 'environment_log.jsonl'
    try:
        if phase == 'main' and images_done >= INTERIM_IMAGES:
            interim_report(run)
        for ordinal, d in enumerate(manifest['dialogues']):
            iid = d['image_id']
            if iid in done:
                continue
            if (run / 'STOP_REQUESTED').exists():
                raise RuntimeError('STOP_REQUESTED')
            if any((p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) != sig for p, sig in signatures.items()):
                raise RuntimeError('frozen source/config metadata changed during execution')
            t0 = time.time()
            n = M.run_image(runner, server, [d], {d['dialog_id']: ordinal}, phase, phasepath, manifest, config)
            folder = phasepath / phase / 'images' / iid
            remove_archived_duplicates(folder)
            wall = time.time() - t0
            count += n; images_done += 1
            commit = read(folder / 'COMMITTED.json'); attempt = folder / commit['attempt_id']
            stores = [read(p)['file_logical_bytes'] for p in attempt.glob('packed_*_persistence.json')]
            with image_log.open('a') as f:
                f.write(json.dumps({'image_id': iid, 'ordinal': ordinal, 'requests': n, 'wall_seconds': wall, 'retained_bytes': folder_bytes(folder),
                                    'store_logical_bytes': sum(stores), 'at_unix': time.time()}) + '\n')
            status(run, phase, 'RUNNING', adopted_requests=count, target_requests=target, completed_images=images_done,
                   elapsed_seconds_this_process=time.time() - started, last_image_id=iid)
            print(json.dumps({'phase': phase, 'image': iid, 'images': images_done, 'adopted': count, 'target': target, 'wall_s': round(wall, 3)}), flush=True)
            if images_done % ENV_LOG_EVERY == 0 or images_done == len(manifest['dialogues']):
                entry = {'images_done': images_done, 'at_unix': time.time(), **window.summary(), **machine_state(run),
                         'role': 'environment drift tracking, not a verdict input'}
                with env_log.open('a') as f:
                    f.write(json.dumps(entry) + '\n')
                print(json.dumps({'environment_log': entry}), flush=True)
                window.reset()
            if phase == 'main' and images_done == INTERIM_IMAGES:
                interim_report(run)
        assert count == target, (count, target)
        dump(phasepath / 'COMPLETE.json', {'status': 'PASS', 'model': 'llava', 'phase': phase, 'images': len(manifest['dialogues']), 'requests': count,
                                           'expected_requests': target, 'methods': config['methods'], 'elapsed_seconds_last_process': time.time() - started,
                                           'source_freeze_sha256': sha(run / 'experiment_freeze.json'), 'manifest_sha256': config['_manifest_sha256'],
                                           'config_sha256': config['_file_sha256'], 'standalone_analysis': 'PENDING', 'not_main_completion': phase != 'main',
                                           'independent_image_audit': 'PASS (every image audited before its atomic commit)'})
        status(run, phase, 'COMPLETE_PENDING_ANALYSIS', adopted_requests=count, target_requests=target)
    finally:
        dump(phasepath / f'phase_stop_{time.time_ns()}.json', {'stopped_unix': time.time(), 'images_done': images_done, 'adopted_requests': count, **machine_state(run)})
        if hasattr(runner, 'close'):
            runner.close()


def admit(run):
    """Pilot40-measured resource admission.  Estimates are never main measurements."""
    freeze_sha = sha(run / 'experiment_freeze.json')
    receipt = read(run / 'llava/pilot40/COMPLETE.json')
    if receipt['status'] != 'PASS' or receipt['source_freeze_sha256'] != freeze_sha:
        raise RuntimeError('pilot40 not complete under the current freeze')
    rows = [json.loads(s) for s in (run / 'llava/pilot40/image_log.jsonl').read_text().splitlines() if s.strip()]
    assert len(rows) == 40
    n = len(read(run / 'llava/main/manifest.json')['dialogues'])
    wall = [r['wall_seconds'] for r in rows]; kept = [r['retained_bytes'] for r in rows]; store = [r['store_logical_bytes'] for r in rows]
    raw_estimate = math.ceil(sum(kept) / len(kept) * n * 1.35)
    floor, working, analysis_reserve, failed_reserve = 30 * GIB, 2 * GIB, 4 * GIB, 2 * GIB
    required = floor + working + raw_estimate + analysis_reserve + failed_reserve
    free = shutil.disk_usage(run).free
    p = {'status': 'PASS' if free >= required else 'BLOCKED_STORAGE', 'kind': 'PILOT_BASED_ESTIMATE_NOT_MAIN_MEASUREMENT', 'model': 'llava',
         'pilot40_receipt_sha256': sha(run / 'llava/pilot40/COMPLETE.json'), 'free_bytes_at_admission': free, 'required_free_bytes': required,
         'minimum_free_bytes': floor, 'working_bound_bytes': working, 'analysis_reserve_bytes': analysis_reserve, 'failed_attempt_reserve_bytes': failed_reserve,
         'estimated_main_raw_evidence_bytes_with_35pct_margin': raw_estimate,
         'pilot_wall_seconds_per_image_mean': sum(wall) / len(wall), 'pilot_wall_seconds_per_image_max': max(wall),
         'pilot_retained_bytes_per_image_mean': sum(kept) / len(kept), 'pilot_retained_bytes_per_image_max': max(kept),
         'measured_pilot40_scratch_peak_bytes': max(store), 'pilot_store_logical_bytes_mean': sum(store) / len(store),
         'estimated_main_wall_seconds': sum(wall) / len(wall) * n, 'estimated_main_wall_hours': sum(wall) / len(wall) * n / 3600,
         'estimated_main_store_write_bytes': sum(store) / len(store) * n,
         'wall_scope': 'per-image execution, audit, hashing, gzip archive and payload cleanup; excludes model loading, the interim report and final analysis',
         'scratch_policy': 'one image store at a time; deleted by exact SHA allowlist after audit + commit', 'created_unix': time.time(), 'source_freeze_sha256': freeze_sha}
    dest = run / 'llava/main_resource_admission.json'
    if dest.exists():
        old = read(dest)
        if old['status'] == 'PASS' and old['pilot40_receipt_sha256'] == p['pilot40_receipt_sha256']:
            print(json.dumps(old)); return old
        os.replace(dest, run / f'llava/main_resource_admission_superseded_{time.time_ns()}.json')
    dump(dest, p); print(json.dumps(p), flush=True)
    return p


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--phase', choices=PHASES + ('admit',), required=True); ap.add_argument('--resume', action='store_true')
    a = ap.parse_args(); run = a.run.resolve()
    if a.phase == 'admit':
        sys.exit(0 if admit(run)['status'] == 'PASS' else 2)
    try:
        run_phase(run, a.phase, a.resume)
    except BaseException as exc:
        status(run, a.phase, 'BLOCKED', error=repr(exc), traceback=traceback.format_exc()); raise

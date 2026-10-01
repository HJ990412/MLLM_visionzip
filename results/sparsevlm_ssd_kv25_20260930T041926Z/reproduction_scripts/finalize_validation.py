#!/usr/bin/env python3
"""Finalize already completed SparseVLM SSD evidence; never execute benchmarks.

Run only after pilots, runner CPU tests, scripts/99 independent raw audit, and
verify_final_protection.py have completed. This tool never aggregates raw rows.
All final validation writes are exclusive. Existing identical copied artifacts
are preserved in place; any conflicting artifact aborts before copying.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

EXPECTED_PHASE_COUNTS = {'smoke': 60, 'gqa': 1200, 'mt': 600}
EXPECTED_METHODS = ['recompute', 'fullload', 'sparsevlm_ssd_kv25_probe3',
                    'sparsevlm_ssd_kv25_allhead', 'ours_kv25']
MODEL_REVISION = 'c916e6cdcd760b4cecd1dd4907f84ac649f93b23'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def exists(path):
    return Path(path).exists() or Path(path).is_symlink()


def exclusive_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def test_count_from_log(path):
    value = Path(path).read_text()
    matches = re.findall(r'^Ran (\d+) tests? in ', value, flags=re.MULTILINE)
    if len(matches) != 1 or not re.search(r'^OK(?:\s*\(.*\))?\s*$', value, flags=re.MULTILINE):
        raise ValueError(f'No unambiguous passing unittest summary: {path}')
    return int(matches[0])


class Preflight:
    def __init__(self, root):
        self.root = root
        self.errors = []
        self.receipts = {}
        self.hash_cache = {}

    def require(self, condition, check, detail=None):
        if not condition:
            self.errors.append({'check': check, 'detail': detail})

    def resolve(self, path):
        path = Path(path)
        return path if path.is_absolute() else self.root / path

    def digest(self, path):
        path = Path(path).resolve()
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        if key not in self.hash_cache:
            self.hash_cache[key] = sha(path)
        return self.hash_cache[key]

    def load(self, label, path):
        path = Path(path)
        if not path.is_file():
            self.errors.append({'check': 'required_receipt_exists', 'detail': str(path)})
            return {}
        try:
            data = json.loads(path.read_text())
        except (ValueError, OSError) as exc:
            self.errors.append({'check': 'readable_json_receipt', 'detail': [str(path), repr(exc)]})
            return {}
        self.receipts[label] = {'path': str(path.resolve()), 'sha256': self.digest(path),
                                'bytes': path.stat().st_size}
        return data

    def verify_reference(self, reference, label):
        if not isinstance(reference, dict) or not reference.get('path') or not reference.get('sha256'):
            self.require(False, 'receipt_reference_complete', label)
            return
        path = self.resolve(reference['path'])
        self.require(path.is_file(), 'receipt_reference_exists', [label, str(path)])
        if path.is_file():
            self.require(self.digest(path) == reference['sha256'], 'receipt_reference_hash', label)

    def verify_source_map(self, hashes, label):
        self.require(isinstance(hashes, dict) and bool(hashes), 'nonempty_source_hash_map', label)
        for relative, expected in (hashes or {}).items():
            path = self.resolve(relative)
            self.require(path.is_file(), 'source_exists', relative)
            if path.is_file():
                self.require(self.digest(path) == expected, 'frozen_source_unchanged', [label, relative])

    def abort_if_errors(self):
        if self.errors:
            print(json.dumps({'status': 'BLOCKED_FINALIZATION', 'validation_files_created': False,
                              'failures': self.errors}, indent=2), file=sys.stderr)
            raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/dblab/hj/mllm_v2'))
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--gpu-receipt', type=Path)
    parser.add_argument('--runner-cpu-receipt', type=Path)
    parser.add_argument('--fresh-setup-receipt', type=Path,
                        help='Optional separate setup diagnostic; never promotes serving readiness')
    args = parser.parse_args()
    root, run, results = args.root.resolve(), args.run_dir.resolve(), args.results_dir.resolve()
    if run == results:
        raise ValueError('Run and results destinations must differ')
    for destination in (run/'validation.json', results/'validation.json'):
        if exists(destination):
            raise FileExistsError(f'Prior validation must be preserved: {destination}')
    check = Preflight(root)
    gpu_path = args.gpu_receipt.resolve() if args.gpu_receipt else run/'gpu_full_01/gpu_validation.json'
    runner_cpu_path = args.runner_cpu_receipt.resolve() if args.runner_cpu_receipt else run/'runner_cpu_validation.json'
    fresh_setup_path = args.fresh_setup_receipt.resolve() if args.fresh_setup_receipt else run/'setup_split_01/receipt.json'
    config = check.load('config', run/'config.json')
    runner_config = check.load('runner_config', run/'runner_config.json')
    manifest = check.load('manifest', run/'manifest.json')
    freeze = check.load('source_freeze', run/'source_freeze.json')
    contract = check.load('contract_freeze', run/'contract_freeze_v2.json')
    storage = check.load('storage_plan', run/'storage_plan.json')
    cpu = check.load('CPU_25', run/'cpu_validation_final.json')
    legacy = check.load('legacy_CPU_387', run/'legacy_cpu_tests.json')
    audit_cpu = check.load('audit_CPU_17', run/'audit_cpu_validation.json')
    runner_cpu = check.load('runner_CPU_7', runner_cpu_path)
    gpu = check.load('GPU_G1_G12', gpu_path)
    gpu_audit = check.load('GPU_independent_receipt_audit', run/'gpu_independent_audit.json')
    independent = check.load('independent_raw_audit_99', results/'independent_audit.json')
    full_protection = check.load('protection_full_rehash', run/'protection_rehash.json')
    final_protection = check.load('protection_final_stat', run/'protection_final.json')
    check.abort_if_errors()

    check.require(config.get('revision') == MODEL_REVISION and runner_config.get('model_revision') == MODEL_REVISION,
                  'same_frozen_model_revision')
    check.require(runner_config.get('methods') == EXPECTED_METHODS, 'five_required_methods')
    check.require(all(config.get(k) == v and runner_config.get(k) == v for k, v in
                      (('seed', 1234), ('batch_size', 1), ('max_new_tokens', 16))), 'fixed_request_configuration')
    check.require(config.get('chunk_size') == 64 and runner_config.get('chunk_size') == 64, 'fixed_chunk_size')
    check.require(storage.get('status') == 'PASS' and storage.get('safe') is True, 'storage_plan_PASS')
    check.verify_source_map(freeze.get('new_or_modified_sources'), 'source_freeze')
    check.verify_source_map(runner_config.get('source_sha256'), 'runner_config')
    check.verify_source_map(gpu.get('source_sha256'), 'GPU_gate')
    check.require(freeze.get('legacy_sources_unchanged') is True, 'legacy_source_freeze_unchanged')
    contract_path = check.resolve(contract.get('contract_path', 'docs/sparsevlm_ssd_kv25_contract.md'))
    contract_hash = check.digest(contract_path)
    check.require(all(x == contract_hash for x in (contract.get('contract_sha256'),
                      freeze.get('contract_sha256'), gpu.get('contract_sha256'), runner_config.get('contract_sha256'))),
                  'contract_hash_chain')
    for field, path in (('official_source_pin_sha256', run/'official_source/official_source_pin.json'),
                        ('local_source_evidence_sha256', run/'contract_source_evidence.json'),
                        ('storage_plan_sha256', run/'storage_plan.json')):
        check.require(path.is_file() and check.digest(path) == contract.get(field), 'frozen_contract_supporting_hash', field)

    check.require(cpu.get('status') == 'PASS' and cpu.get('test_count') == 25, 'CPU_25_PASS')
    check.require(test_count_from_log(run/'cpu_validation_final.log') == 25, 'CPU_25_log')
    check.require(check.digest(run/'cpu_validation_final.log') == cpu.get('log_sha256'), 'CPU_log_hash')
    check.verify_source_map(cpu.get('source_sha256'), 'CPU_25')
    check.require(legacy.get('status') == 'PASS' and legacy.get('exit_code') == 0, 'legacy_CPU_PASS')
    legacy_log = check.resolve(legacy['log'])
    check.require(test_count_from_log(legacy_log) == 387, 'legacy_CPU_387_log')
    check.require(audit_cpu.get('status') == 'PASS' and audit_cpu.get('count') == 17
                  and len(audit_cpu.get('checks', [])) == 17, 'audit_CPU_17_PASS')
    audit_script_hash = check.digest(root/'scripts/99_audit_sparsevlm_ssd_kv25.py')
    check.require(audit_script_hash == audit_cpu.get('audit_script_sha256'), 'audit_CPU_frozen_script')
    check.require(check.digest(run/'audit_cpu_fixture.py') == audit_cpu.get('fixture_sha256'), 'audit_CPU_fixture_hash')
    check.require(runner_cpu.get('status') == 'PASS' and runner_cpu.get('test_count', runner_cpu.get('count')) == 7,
                  'runner_CPU_7_PASS')
    if 'exit_code' in runner_cpu:
        check.require(runner_cpu['exit_code'] == 0, 'runner_CPU_exit_code')
    if runner_cpu.get('log'):
        runner_log = check.resolve(runner_cpu['log'])
        check.require(test_count_from_log(runner_log) == 7, 'runner_CPU_7_log')
        if runner_cpu.get('log_sha256'):
            check.require(check.digest(runner_log) == runner_cpu['log_sha256'], 'runner_CPU_log_hash')
    elif runner_cpu.get('log_sha256'):
        runner_log = runner_cpu_path.with_suffix('.log')
        check.require(runner_log.is_file() and check.digest(runner_log) == runner_cpu['log_sha256'], 'runner_CPU_log_hash')
        if runner_log.is_file():
            check.require(test_count_from_log(runner_log) == 7, 'runner_CPU_7_log')
    if runner_cpu.get('source_sha256'):
        check.verify_source_map(runner_cpu['source_sha256'], 'runner_CPU_7')

    gpu_hash = check.digest(gpu_path)
    check.require(gpu.get('GPU_CORRECTNESS') == 'PASS' and gpu.get('gates') == {f'G{i}': 'PASS' for i in range(1, 13)},
                  'GPU_G1_G12_PASS')
    check.require(len(gpu.get('samples', [])) == 10 and gpu.get('MT_generated_history', {}).get('status') == 'PASS',
                  'GPU_ten_pairs_and_generated_MT')
    check.require(all(x.get('status') == 'PASS' for x in gpu.get('samples', [])), 'GPU_samples_PASS')
    check.require(gpu.get('QWEN_GPU') == 'NOT RUN' and legacy.get('qwen_gpu') == 'NOT RUN', 'QWEN_GPU_not_falsely_passed')
    check.require(freeze.get('gpu_gate_sha256') == gpu_hash and runner_config.get('gpu_gate', {}).get('sha256') == gpu_hash,
                  'GPU_gate_hash_chain')
    check.verify_reference(gpu.get('regression_receipt'), 'GPU_legacy_regression')
    check.require(gpu_audit.get('status') == 'PASS' and gpu_audit.get('G1_G12_receipt_supported') is True
                  and gpu_audit.get('errors') == [] and gpu_audit.get('validation_receipt_sha256') == gpu_hash,
                  'GPU_independent_audit_PASS')

    check.require(independent.get('schema_version') == 'sparsevlm-ssd-kv25-independent-audit-v1'
                  and independent.get('status') == 'PASS' and independent.get('ready') is True,
                  'official_99_independent_audit_ready')
    check.require(independent.get('failure_counts') == {}, 'official_99_no_failures')
    check.require(independent.get('audit_script_sha256') == audit_script_hash, 'official_99_script_hash')
    check.require(independent.get('bootstrap') == {'resamples': 10000, 'seed': 1234, 'cluster': 'image_id', 'weighting': 'request'},
                  'frozen_cluster_bootstrap')
    for label, reference in independent.get('references', {}).items():
        check.verify_reference(reference, 'official_99_'+label)
    check.require(independent.get('references', {}).get('config', {}).get('sha256') == check.digest(run/'runner_config.json')
                  and independent.get('references', {}).get('manifest', {}).get('sha256') == check.digest(run/'manifest.json')
                  and independent.get('references', {}).get('gpu', {}).get('sha256') == gpu_hash,
                  'official_99_same_run_inputs')

    phase_records = {}
    for phase, expected in EXPECTED_PHASE_COUNTS.items():
        receipt = check.load('phase_'+phase, run/phase/'validation.json')
        audited = independent.get('phases', {}).get(phase, {})
        check.require(receipt.get('status') == 'PASS' and receipt.get('requests') == expected
                      and receipt.get('expected_requests') == expected and receipt.get('timing_valid') is True
                      and receipt.get('independent_per_image_audit') == 'PASS', 'phase_runner_complete_and_timing_valid', phase)
        check.require(receipt.get('GPU_GATE', {}).get('sha256') == gpu_hash, 'phase_same_GPU_gate', phase)
        check.require(audited.get('status') == 'PASS' and audited.get('pilot_status') == 'VALID'
                      and audited.get('observed_requests') == expected and audited.get('expected_requests') == expected
                      and audited.get('complete_frozen_workload') is True and audited.get('failure_counts') == {},
                      'official_99_full_frozen_phase_VALID', phase)
        raw_path = run/phase/'raw.jsonl'
        check.require(raw_path.is_file() and check.resolve(audited.get('raw_path', '')).resolve() == raw_path.resolve(),
                      'official_raw_path', phase)
        check.require(bool(re.fullmatch(r'[0-9a-f]{64}', str(audited.get('raw_sha256', '')))), 'official_raw_hash_present', phase)
        if raw_path.is_file():
            check.require(raw_path.stat().st_mtime_ns <= (results/'independent_audit.json').stat().st_mtime_ns,
                          'raw_not_modified_after_official_audit', phase)
        check.require(manifest.get('expected_requests', {}).get(phase) == expected, 'manifest_phase_count', phase)
        phase_records[phase] = {'status': 'VALID', 'requests': expected,
                                'timing_valid': True, 'runner_receipt': check.receipts.get('phase_'+phase),
                                'official_raw_path': audited.get('raw_path'), 'official_raw_sha256': audited.get('raw_sha256'),
                                'raw_reaggregated_by_finalizer': False}

    check.require(full_protection.get('status') == 'PASS' and full_protection.get('failures') == [], 'prior_full_protection_PASS')
    check.require(final_protection.get('schema_version') == 'sparsevlm-final-protection-v1'
                  and final_protection.get('status') == 'PASS' and final_protection.get('failures') == [], 'final_protection_PASS')
    full_hash = check.digest(run/'protection_rehash.json')
    check.require(full_hash == freeze.get('protection_sha256')
                  and final_protection.get('prior_full_rehash', {}).get('sha256') == full_hash,
                  'protection_hash_chain')
    check.require(final_protection.get('source', {}).get('checked_files') == 185
                  and final_protection.get('artifacts', {}).get('checked_files') == 89829,
                  'final_protection_inventory_complete')
    check.require(final_protection.get('source', {}).get('sha256_recomputed') is True
                  and final_protection.get('artifacts', {}).get('content_sha256_recomputed') is False
                  and final_protection.get('artifacts', {}).get('payload_bytes_read') == 0,
                  'final_protection_scope_honest')
    check.require((run/'protection_final.json').stat().st_mtime_ns >= max((run/p/'validation.json').stat().st_mtime_ns
                  for p in EXPECTED_PHASE_COUNTS), 'final_protection_after_all_phase_completion')
    analysis_source_sha256 = {}
    for name in ('verify_final_protection.py', 'finalize_validation.py',
                 'setup_split_diagnostic.py', 'build_supplement.py'):
        path = run/name
        check.require(path.is_file(), 'supplemental_analysis_script_exists', name)
        if path.is_file():
            analysis_source_sha256[str(path)] = check.digest(path)
    optional_setup = {'status': 'NOT RUN', 'scope': 'single-sample fresh T1 serializer split diagnostic',
                      'used_for_serving_readiness': False, 'used_for_pilot_latency_or_quality': False,
                      'replaces_pilot_bundle_measurement': False}
    if args.fresh_setup_receipt or fresh_setup_path.is_file():
        fresh = check.load('optional_fresh_setup_diagnostic', fresh_setup_path)
        optional_setup.update(status=fresh.get('status', 'UNRESOLVED'),
                              receipt=check.receipts.get('optional_fresh_setup_diagnostic'))
    check.abort_if_errors()

    # Materialize small reproducibility evidence. Large GPU/raw files stay in
    # the immutable run path and receive explicit content-hash references.
    copied_names = [
        'source.diff', 'config.json', 'runner_config.json', 'manifest.json', 'storage_plan.json',
        'store_footprint.json', 'reproduction_commands.json', 'source_freeze.json', 'source_before.json',
        'contract_freeze_v2.json', 'contract_source_evidence.json', 'environment_before.json',
        'sparsevlm_ssd_kv25_contract.frozen_v2.md', 'cpu_validation_final.json', 'cpu_validation_final.log',
        'legacy_cpu_tests.json', 'legacy_cpu_tests.log', 'audit_cpu_validation.json',
        'gpu_independent_audit.json', 'protection_rehash.json', 'protection_final.json',
        'protected_artifacts_before.jsonl', 'official_source/official_source_pin.json',
    ]
    copy_plan = [(run/name, results/name) for name in copied_names]
    copy_plan += [(runner_cpu_path, results/'runner_cpu_validation.json'),
                  (contract_path, results/'sparsevlm_ssd_kv25_contract.md'),
                  (Path(__file__).resolve(), results/'reproduction_scripts/finalize_validation.py'),
                  (run/'verify_final_protection.py', results/'reproduction_scripts/verify_final_protection.py'),
                  (run/'setup_split_diagnostic.py', results/'reproduction_scripts/setup_split_diagnostic.py'),
                  (run/'build_supplement.py', results/'reproduction_scripts/build_supplement.py')]
    for phase in EXPECTED_PHASE_COUNTS:
        copy_plan.append((run/phase/'validation.json', results/'phase_receipts'/phase/'validation.json'))
    if 'optional_fresh_setup_diagnostic' in check.receipts:
        copy_plan.append((fresh_setup_path, results/'optional_fresh_setup_diagnostic.json'))
    artifacts = []
    for source, destination in copy_plan:
        check.require(source.is_file(), 'copy_source_exists', str(source))
        if not source.is_file():
            continue
        digest = check.digest(source)
        preserved = exists(destination)
        if preserved:
            check.require(destination.is_file() and check.digest(destination) == digest,
                          'existing_result_copy_matches_without_overwrite', str(destination))
        artifacts.append({'source': str(source.resolve()), 'destination': str(destination.resolve()),
                          'sha256': digest, 'bytes': source.stat().st_size,
                          'action': 'PRESERVE_IDENTICAL_EXISTING' if preserved else 'CREATE_EXCLUSIVE'})
    gpu_reference_path = results/'receipts/gpu_validation.ref.json'
    gpu_reference = {'path': str(gpu_path.resolve()), 'sha256': gpu_hash, 'bytes': gpu_path.stat().st_size,
                     'role': 'frozen full GPU correctness receipt; retained in run directory'}
    reference_content = encoded(gpu_reference)
    if exists(gpu_reference_path):
        check.require(gpu_reference_path.is_file() and gpu_reference_path.read_bytes() == reference_content,
                      'existing_GPU_reference_matches_without_overwrite')
    required_analysis = ('REPORT.md', 'summary.csv', 'paired_comparisons.csv', 'io_timing_memory_breakdown.csv',
                         'controlled_head_policy.csv', 'setup_costs.csv', 'independent_audit.json')
    analysis = {}
    for name in required_analysis:
        path = results/name
        check.require(path.is_file(), 'official_analysis_artifact_present', name)
        if path.is_file():
            analysis[name] = {'path': str(path), 'sha256': check.digest(path), 'bytes': path.stat().st_size}
    check.abort_if_errors()

    verdicts = {
        'IMPLEMENTATION_PROBE3': 'PASS', 'IMPLEMENTATION_ALLHEAD': 'PASS',
        'CPU_TEST': 'PASS', 'GPU_CORRECTNESS': 'PASS', 'PROJECTION_REUSE': 'PASS',
        'ALLHEAD_K_READ_REUSE': 'PASS', 'SMOKE_5_ARM_PILOT': 'VALID',
        'GQA_5_ARM_PILOT': 'VALID', 'MT_5_ARM_PILOT': 'VALID',
        'LEGACY_OURS_QWEN_CPU_PROTECTION': 'PASS', 'QWEN_GPU': 'NOT RUN',
        'MT_GQA_4061_DIALOGUE_MAIN': 'NOT RUN', 'MT_VQA_FULL': 'NOT RUN',
        'READY_FOR_LLAVA_MT_GQA_BASELINE_INTEGRATION': 'YES',
    }
    report = {
        'schema_version': 'sparsevlm-ssd-kv25-final-validation-v1', 'status': 'PASS',
        'finalized_at': datetime.now(timezone.utc).isoformat(), 'run_dir': str(run), 'results_dir': str(results),
        'verdicts': verdicts, 'ready': True,
        'readiness_basis': 'Frozen implementation, CPU 25/387/17/7, G1-G12 GPU, independent GPU receipt audit, '
                           'official scripts/99 complete 60/1200/600-request raw audit, valid timing, and final protection all PASS.',
        'SOURCE_METRIC_FIDELITY': {
            'status': 'PASS_WITH_EXPLICIT_ADAPTATION',
            'faithful_metric_scope': 'original SparseVLM text-rater formula (adapted FP32 computation) and post-softmax head/rater visual attention mean',
            'adaptation': 'query-dependent SSD-resident cached-KV retrieval with exact fixed ceil(N_content/4) per layer',
            'probe3_additional_approximation': 'fixed first three scoring heads; selected tokens retain every KV head',
            'allhead_policy': 'all attention heads; original full visual K online read and same-request K reuse',
            'excluded_original_features': ['progressive hidden-token pruning', 'adaptive layer schedule', 'token recycling/merging'],
            'original_full_algorithm_or_published_performance_reproduction_claimed': False,
            'contract': {'path': str(contract_path), 'sha256': contract_hash},
            'official_source_pin': {'path': str(run/'official_source/official_source_pin.json'),
                                    'sha256': check.digest(run/'official_source/official_source_pin.json')},
        },
        'test_counts': {'core_store_adapter_CPU': 25, 'legacy_ours_qwen_CPU': 387, 'independent_audit_CPU': 17, 'pilot_runner_CPU': 7},
        'GPU_gates': gpu['gates'], 'phases': phase_records,
        'official_raw_analysis': {'producer': 'scripts/99_audit_sparsevlm_ssd_kv25.py', 'script_sha256': audit_script_hash,
                                  'receipt': check.receipts['independent_raw_audit_99'], 'artifacts': analysis,
                                  'raw_reaggregated_by_finalizer': False},
        'receipts': check.receipts,
        'frozen_source_sha256': freeze['new_or_modified_sources'],
        'analysis_source_sha256': analysis_source_sha256,
        'analysis_source_scope': 'run-owned supplemental tooling, separate from unchanged production source_freeze and source.diff',
        'frozen_model_revision': MODEL_REVISION,
        'configuration_sha256': check.digest(run/'runner_config.json'), 'manifest_sha256': check.digest(run/'manifest.json'),
        'setup_measurement_scope': {'GQA_and_smoke_persistence': 'NOT_REMEASURED',
                                    'MT_shared_canonical_probe_bundle': 'MEASURED_ATTEMPT_COST',
                                    'primary_pilot_AllHead_base_Probe3_increment': 'NOT_SEPARATELY_MEASURED',
                                    'optional_separate_fresh_T1_setup': optional_setup},
        'protection_scope': {'prior_full_SHA256_rehash': check.receipts['protection_full_rehash'],
                             'final_source_SHA256_and_artifact_stat': check.receipts['protection_final_stat'],
                             'final_artifact_content_SHA256_recomputed': False},
        'artifact_copies': artifacts, 'GPU_large_receipt_reference': gpu_reference,
        'finalizer': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__), 'gpu_used': False,
                      'benchmarks_executed': False, 'existing_artifacts_overwritten': False},
    }
    # Every gate and every destination conflict is checked before mutations.
    for item in artifacts:
        if item['action'] == 'PRESERVE_IDENTICAL_EXISTING':
            continue
        source, destination = Path(item['source']), Path(item['destination'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as src, destination.open('xb') as dst:
            for block in iter(lambda: src.read(1 << 20), b''):
                dst.write(block)
            dst.flush()
            os.fsync(dst.fileno())
        if sha(destination) != item['sha256']:
            raise RuntimeError(f'Copy digest mismatch; preserve evidence: {destination}')
    if not exists(gpu_reference_path):
        exclusive_bytes(gpu_reference_path, reference_content)
    content = encoded(report)
    exclusive_bytes(run/'validation.json', content)
    exclusive_bytes(results/'validation.json', content)
    print(json.dumps({'status': 'PASS', 'ready': True, 'validation_sha256': hashlib.sha256(content).hexdigest(),
                      'run_validation': str(run/'validation.json'), 'results_validation': str(results/'validation.json'),
                      'artifact_copies_created': sum(x['action'] == 'CREATE_EXCLUSIVE' for x in artifacts),
                      'identical_existing_copies_preserved': sum(x['action'] == 'PRESERVE_IDENTICAL_EXISTING' for x in artifacts)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Collect published final MT-GQA measurements without changing the source runs."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SPECS = {
    'LLaVA': 'mt_gqa_B_llava5_qwen3_20261001T093921Z',
    'Qwen': 'qwen_mt_gqa_prefixstore3_20261002T074856Z',
    'GLM': 'glm46v_prefixstore3_v6_20261007T104505Z',
}
LABELS = {'recompute': 'ReComp', 'fullload': 'FullLoad',
          'ours_prefixstore_kv25': 'KV25 [B]', 'mpic32_ssd': 'MPIC-32',
          'sparsevlm_ssd_kv25_allhead': 'AllHead'}
sources = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register(path):
    sources[str(path.relative_to(ROOT))] = digest(path)
    return path


def read_csv(path):
    with register(path).open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row.get('method_id') == 'qwen_ours_prefixstore_kv25':
            row['method_id'] = 'ours_prefixstore_kv25'
    return rows


def write_csv(name, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def numeric(row, key):
    value = row.get(key, 'NOT_MEASURED')
    return float(value) if value not in ('', 'NOT_MEASURED', 'None') else 'NOT_MEASURED'


def main():
    summary, pairs, settings = [], [], []
    scopes = ['T1', 'T2', 'T3', 'all', 'followup']
    for model, run_name in SPECS.items():
        parent = ROOT / 'results' / run_name
        base = parent / 'final' / 'llava' if model == 'LLaVA' else parent / 'final'
        quality = read_csv(base / 'quality.csv')
        latency = read_csv(base / 'latency.csv')
        storage = read_csv(base / 'storage_summary.csv')
        sessions = read_csv(parent / 'reporting_details/session_summary.csv'
                            if model == 'Qwen' else base / 'session_summary.csv')
        outer = read_csv(parent / 'reporting_details/store_io_summary.csv') if model == 'Qwen' else []
        register(parent / 'REPORT.md')
        q = {(r['method_id'], r['scope']): r for r in quality}
        l = {(r['method_id'], r['scope']): r for r in latency}
        s = {r['method_id']: r for r in storage}
        t = {r['method_id']: r for r in sessions}
        o = {r['method_id']: r for r in outer}
        for method in LABELS:
            if (method, 'all') not in q:
                continue
            row = {'model': model, 'method_id': method, 'method': LABELS[method],
                   'source_method_id': 'qwen_ours_prefixstore_kv25' if model == 'Qwen' and method == 'ours_prefixstore_kv25' else method,
                   'common_3arm': method in ('recompute', 'fullload', 'ours_prefixstore_kv25'),
                   'images': 398, 'dialogues': 4061, 'requests': 12183, 'followups': 8122,
                   'source_report': str((parent / 'REPORT.md').relative_to(ROOT))}
            for scope in scopes:
                value = q[method, scope]
                expected = 12183 if scope == 'all' else 8122 if scope == 'followup' else 4061
                assert int(value['requests']) == expected
                em = 100 * float(value['correct']) / expected
                published = float(value['accuracy']) * (1 if model == 'GLM' else 100)
                assert abs(em - published) < 1e-10
                row[f'{scope}_EM_percent'] = em
                for metric in ['ttft_ms', 'request_e2e_ms', 'ssd_read_bytes']:
                    for stat in ['mean', 'p50', 'p95']:
                        row[f'{scope}_{metric}_{stat}'] = numeric(l[method, scope], f'{metric}_{stat}')
            row['followup_KV_read_MB_mean'] = row['followup_ssd_read_bytes_mean'] / 1e6
            for field in ['physical_stream_session_ms', 'derived_standalone_session_ms']:
                for stat in ['mean', 'p50', 'p95']:
                    row[f'{field}_{stat}'] = numeric(t[method], f'{field}_{stat}')
            if method in s:
                row['store_logical_MB_image_mean'] = float(s[method]['file_logical_bytes_mean']) / 1e6
                row['store_allocated_MB_image_mean'] = float(s[method]['allocated_bytes_mean']) / 1e6
                row['activation_ms_image_mean'] = float(s[method]['activation_ms_mean'])
                row['persistence_ms_image_mean'] = (
                    float(o[method]['outer_writer_wall_ms_mean']) if model == 'Qwen' else
                    float(s[method]['persistence_ms_mean' if model == 'GLM' else 'service_persistence_ms_mean']))
                row['store_setup_provenance'] = s[method].get('provenance', 'MEASURED_CALLER_OUTER')
            else:
                for key in ['store_logical_MB_image_mean', 'store_allocated_MB_image_mean',
                            'activation_ms_image_mean', 'persistence_ms_image_mean']:
                    row[key] = 'NO_KV_STORE'
                row['store_setup_provenance'] = 'NO_KV_STORE'
            summary.append(row)
        original = read_csv(base / 'paired_analysis.csv')
        mapping = {(r['baseline'], r['scope'], r['metric']): r for r in original}
        if model == 'Qwen':
            for r in read_csv(parent / 'reporting_details/paired_analysis.csv'):
                mapping[r['baseline'], r['scope'], r['metric']] = r
        for r in mapping.values():
            if model == 'Qwen' and 'persistence' in r['metric']:
                # Preserve inner publication analysis in its original directory;
                # this collection uses the separately measured outer endpoint.
                continue
            pairs.append({'model': model, **r})
        config_path = ROOT / 'runs' / run_name / ('llava/config.json' if model == 'LLaVA' else 'config.json')
        config = json.loads(register(config_path).read_text())
        native = config if model == 'GLM' else config['model']
        settings.append({'model': model, 'model_id': native['model_id'],
                         'revision': native.get('revision', native.get('checkpoint_revision')),
                         'max_new_tokens': native['max_new_tokens'],
                         'backend': native.get('backend', native.get('attention')),
                         'KV_dtype': native.get('ssd_dtype', native.get('SSD_dtype', 'bfloat16')),
                         'configuration_source': str(config_path.relative_to(ROOT))})
    assert len(summary) == 11
    write_csv('summary.csv', summary)
    write_csv('paired_analysis.csv', pairs)
    write_csv('model_settings.csv', settings)
    comparisons = []
    for model in SPECS:
        by_method = {r['method_id']: r for r in summary if r['model'] == model}
        ours = by_method['ours_prefixstore_kv25']
        for method in ['recompute', 'fullload']:
            ref = by_method[method]
            comparisons.append({
                'model': model, 'baseline': LABELS[method],
                'followup_EM_delta_pp': ours['followup_EM_percent'] - ref['followup_EM_percent'],
                'followup_TTFT_reduction_percent': 100 * (1 - ours['followup_ttft_ms_mean'] / ref['followup_ttft_ms_mean']),
                'followup_KV_read_delta_MB': ours['followup_KV_read_MB_mean'] - ref['followup_KV_read_MB_mean'],
                'shared_session_reduction_percent': 100 * (1 - ours['physical_stream_session_ms_mean'] / ref['physical_stream_session_ms_mean']),
                'derived_standalone_session_reduction_percent': 100 * (1 - ours['derived_standalone_session_ms_mean'] / ref['derived_standalone_session_ms_mean']),
            })
    write_csv('KV25_vs_baselines.csv', comparisons)
    assert all(digest(ROOT / path) == expected for path, expected in sources.items())
    provenance = {
        'status': 'PASS', 'collection_date_client': '2026-10-12',
        'source_csv_units': {'LLaVA_accuracy': 'fraction', 'Qwen_accuracy': 'fraction', 'GLM_accuracy': 'percent'},
        'accuracy_recomputed_from_correct_counts': True,
        'method_id_normalization': {'qwen_ours_prefixstore_kv25': 'ours_prefixstore_kv25'},
        'source_files_sha256_before_and_after': sources,
        'models': 3, 'methods': 11, 'common_3arm_rows': 9,
        'source_runs_modified': False, 'GPU_experiments_rerun': False,
        'limits': [
            'Cross-model rows describe independent model-native runs; paired CIs are within each model only.',
            'LLaVA eager/FP16 store/anyres and Qwen SDPA/BF16 store/native resolution use cap16; GLM SDPA/BF16/native resolution uses cap32.',
            'GLM TTFT usually ends on a native answer-boundary control; first lexical answer latency is NOT_MEASURED.',
            'LLaVA FullLoad canonical store includes unused probe sidecars shared with AllHead; independent base-only setup is NOT_MEASURED.',
            'Qwen FullLoad stores/reads padding; GLM full and B payloads have no padding.',
            'AllHead logical footprint includes a shared canonical bundle while physical build is attributed once to FullLoad.',
            'Standalone-equivalent is DERIVED and excludes NOT_MEASURED hypothetical non-source T1 capture increment.',
            'OS returned bytes are not NAND physical traffic; ReComp KV read zero does not mean image I/O zero.',
            'This collection reuses published analyses; it does not resample models jointly or rerun bootstrap.',
        ],
        'generated_files': {name: digest(OUT / name) for name in
                            ['aggregate.py', 'summary.csv', 'paired_analysis.csv', 'model_settings.csv', 'KV25_vs_baselines.csv']},
    }
    with (OUT / 'provenance.json').open('x') as stream:
        json.dump(provenance, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({'status': 'PASS', 'methods': len(summary), 'source_files_preserved': len(sources)}))
    for row in summary:
        print(json.dumps({k: row[k] for k in ['model', 'method', 'all_EM_percent', 'followup_EM_percent',
                                             'followup_ttft_ms_mean', 'followup_request_e2e_ms_mean',
                                             'followup_KV_read_MB_mean', 'store_logical_MB_image_mean',
                                             'physical_stream_session_ms_mean', 'derived_standalone_session_ms_mean']}))
    print('comparisons', json.dumps(comparisons))


if __name__ == '__main__':
    main()

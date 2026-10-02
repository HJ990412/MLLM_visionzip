"""Post-main detail tables from audited CSVs and immutable main raw; no inference."""
from pathlib import Path
from collections import defaultdict
import csv, hashlib, importlib.util, json, math, os, time

ROOT = Path.cwd()
RUN = Path(__file__).resolve().parent
OUT = ROOT / 'results' / RUN.name
FINAL = OUT / 'final'
DEST = OUT / 'details'
spec = importlib.util.spec_from_file_location('frozen_B_report_details', ROOT / 'scripts/113_report_mt_gqa_B.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def csvrows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def number(x):
    return None if x in ('', None, 'NOT_MEASURED') else float(x)


def table(headers, values):
    return '\n'.join(R.mdtable(headers, values))


def triple(values):
    s = R.stats(values)
    return ' / '.join(R.fmt(s[k], 3) for k in ('mean', 'p50', 'p95'))


def main():
    assert (RUN / 'final_analysis_complete.json').is_file(), 'wait for independent finalizer'
    freeze = R.read(RUN / 'detail_supplement_v2_freeze.json')
    assert R.sha(__file__) == freeze['sha256']
    assert R.sha(ROOT / 'scripts/113_report_mt_gqa_B.py') == freeze['report_source_sha256']
    DEST.mkdir(exist_ok=False)
    lines = ['## 필수 세부 계측 및 session 비교 보충', '', '단위: MB=10^6 bytes, ms, 정확도 차이=percentage points. 모든 표는 이번 실행의 독립 감사 CSV와 commit된 raw/event에서 생성했다. 평균·분위수는 별도이며 중첩 component interval을 TTFT/E2E에 더하지 않는다.', '']
    inputs = {}
    for model in ('llava', 'qwen'):
        src = FINAL / model
        if not (src / 'quality.csv').exists():
            continue
        dst = DEST / model
        dst.mkdir()
        for name in ('paired_analysis', 'latency', 'storage_metrics', 'storage_summary', 'session_per_dialogue', 'image_budget_io', 'request_diagnostics'):
            inputs[str(src / (name + '.csv'))] = R.sha(src / (name + '.csv'))
        paired = csvrows(src / 'paired_analysis.csv')
        R.dump(dst / 'paired_analysis.json', {'source': str(src / 'paired_analysis.csv'), 'analysis': paired, 'resamples': 10000, 'seed': 1234, 'weighting': 'pooled sums/counts under shared image-cluster resampling; image-macro setup separately labeled'})
        latency = {r['method_id']: r for r in csvrows(src / 'latency.csv') if r['scope'] == 'followup'}
        storage = csvrows(src / 'storage_metrics.csv')
        store = {(r['image_id'], r['method_id']): r for r in storage}
        sessions = csvrows(src / 'session_per_dialogue.csv')
        methods = list(latency)
        B = methods[-1]
        images = sorted({r['image_id'] for r in sessions})
        by_session = {(r['dialog_id'], r['method_id']): r for r in sessions}
        session_comparisons = []
        lines += [f'### {model}: 실제 공유-stream과 standalone 환산', '']
        for metric, title in (('physical_stream_session_ms', '실제 physical shared-image stream 비용 귀속'), ('derived_standalone_session_ms', 'Derived standalone-equivalent: dialogue마다 measured image setup 재귀속')):
            values = []
            for method in methods:
                rs = [r for r in sessions if r['method_id'] == method]
                diff = [{'image_id': r['image_id'], 'delta': float(r[metric]) - float(by_session[(r['dialog_id'], 'recompute')][metric])} for r in rs]
                ci = R.cluster_ci(diff, 'delta', images)
                session_comparisons.append({'method_id': method, 'reference': 'recompute', 'metric': metric, 'direction': 'method-minus-ReComp', **ci})
                request = R.mean([float(r['request_E2E_sum_ms']) for r in rs])
                if metric == 'physical_stream_session_ms':
                    persistence = R.mean([float(r['actual_build_attribution_ms']) for r in rs])
                    activation = R.mean([float(r['actual_activation_attribution_ms']) for r in rs])
                else:
                    persistence = R.mean([0. if method == 'recompute' else float(store[(r['image_id'], method)]['service_persistence_ms']) for r in rs])
                    activation = R.mean([0. if method == 'recompute' else float(store[(r['image_id'], method)]['activation_ms']) for r in rs])
                assert math.isclose(request + persistence + activation, R.mean([float(r[metric]) for r in rs]), abs_tol=1e-6)
                values.append([R.LABEL[method], f'{request:.3f}', f'{persistence:.3f}', f'{activation:.3f}', triple([float(r[metric]) for r in rs]), R.ci_text(ci), rs[0]['setup_provenance']])
                if method == B:
                    original = next(r for r in paired if r['baseline'] == 'recompute' and r['metric'] == metric)
                    for k in ('point', 'ci_low', 'ci_high'):
                        assert math.isclose(ci[k], float(original[k]), abs_tol=1e-6)
            lines += [title, '', table(['Method', 'Request E2E sum mean', 'Persistence attribution mean', 'Activation attribution mean', 'Total mean / p50 / p95', 'Δ vs ReComp [95% CI]', 'Setup provenance'], values), '']
        R.writecsv(dst / 'session_vs_recompute.csv', session_comparisons)
        R.dump(dst / 'session_vs_recompute.json', session_comparisons)
        lines += ['Standalone 표는 shared-store에서 측정한 request에 setup을 재귀속한 파생값이다. AllHead persistence는 같은 canonical bundle의 DERIVED_FROM_SHARED 값이다. Non-source T1의 가상 capture 증분은 NOT_MEASURED이므로 완전한 cold-session 실측이 아니다.', '', f'### {model}: 저장/setup 분포와 paired 차이', '']
        values = []
        for method in methods:
            rs = [r for r in storage if r['method_id'] == method]
            if not rs:
                continue
            values.append([R.LABEL[method], triple([float(r['stored_k']) for r in rs]), triple([float(r['valid_content_bytes']) / 1e6 for r in rs]), triple([float(r['file_logical_bytes']) / 1e6 for r in rs]), triple([float(r['os_write_bytes']) / 1e6 for r in rs]), triple([float(r['service_persistence_ms']) for r in rs]), triple([float(r['activation_ms']) for r in rs])])
        lines += [table(['Method', 'Valid content rows mean / p50 / p95', 'Valid content MB mean / p50 / p95', 'Total store MB mean / p50 / p95', 'OS write MB mean / p50 / p95', 'Persistence ms mean / p50 / p95', 'Activation ms mean / p50 / p95'], values), '']
        selected = [r for r in paired if r['scope'] == 'image_macro_setup' or (r['scope'] == 'followup' and r['metric'] in ('ttft_ms', 'request_e2e_ms', 'ssd_read_bytes'))]
        lines += [table(['B − baseline', 'Scope', 'Metric / unit', 'Mean difference [95% CI]'], [[R.LABEL[r['baseline']], r['scope'], r['metric'] + (' / MB' if 'bytes' in r['metric'] else ' / ms'), R.ci_text({k: float(r[k]) for k in ('point', 'ci_low', 'ci_high')}, 1e-6 if 'bytes' in r['metric'] else 1.)] for r in selected]), '']
        lines += [f'### {model}: Followup H2D / online component 계측', '']
        metrics = ('h2d_kv_bytes', 'h2d_payload_compute_bytes', 'rater_visual_h2d_bytes', 'h2d_ms', 'selector_host_ms', 'selector_native_ms', 'cache_assembly_ms', 'scatter_ms')
        values = []
        for method in methods:
            r = latency[method]
            vals = []
            for metric in metrics:
                value = number(r[metric + '_mean'])
                vals.append('NOT_MEASURED' if value is None else f'{value / (1e6 if "bytes" in metric else 1):.3f}')
            values.append([R.LABEL[method], *vals])
        lines += [table(['Method', 'KV H2D MB', 'Payload compute H2D MB', 'Rater H2D MB', 'H2D ms', 'Layer selector host ms', 'Prefix selector ms', 'Cache assembly ms', 'Scatter ms'], values), '', 'H2D 열은 각 native 계측의 범위이며 서로 합산하지 않는다. FullLoad/B/MPIC에서 tensor bytes가 기록되지 않은 항목은 NOT_MEASURED다. MPIC H2D 시간은 동기 전송 실측이며 tensor bytes 또는 bus counter와 구분한다.', '']
        components = []
        activation_residency = []
        for cp in sorted((RUN / model / 'main/images').glob('*/COMMITTED.json')):
            c = R.read(cp)
            raw = cp.parent / c['attempt_id'] / 'raw.jsonl'
            # Already fully rehashed by the independent audit; bind the exact commit.
            inputs[str(cp)] = R.sha(cp)
            for ap in sorted(raw.parent.glob('*_activation.json')):
                method = ap.name.removesuffix('_activation.json')
                a = R.read(ap)
                completion = a.get('completion_integrity', {})
                unused = a.get('unused_bundle_integrity', {})
                activation_residency.append({'image_id': cp.parent.name, 'method_id': method, 'event_id': a['event_id'],
                    'host_tensor_bytes': a.get('host_tensor_bytes', 'NOT_MEASURED'),
                    'gpu_tensor_bytes': a.get('gpu_tensor_bytes', 'NOT_MEASURED'),
                    'visual_payload_resident_bytes': a.get('visual_payload_resident_bytes', 'NOT_MEASURED'),
                    'metadata_ready_file_bytes': a.get('file_bytes', a.get('host_metadata_file_bytes', 'NOT_MEASURED')),
                    'service_integrity_read_bytes': a.get('hash_verified_bytes', completion.get('integrity_read_bytes', 'NOT_MEASURED')),
                    'unused_bundle_integrity_read_bytes': unused.get('integrity_read_bytes', 'NOT_MEASURED'),
                    'service_integrity_ms': a['hash_verification_seconds'] * 1000 if 'hash_verification_seconds' in a else completion.get('integrity_ms', 'NOT_MEASURED'),
                    'unused_bundle_integrity_ms': unused.get('integrity_ms', 'NOT_MEASURED'),
                    'activation_outer_wall_ms': a.get('outer_wall_ms', a.get('activation_ms', 'NOT_MEASURED'))})
            for r in R.rows(raw):
                if r['turn_id'] > 1 and r['method_id'] == 'mpic32_ssd':
                    v = r['result']
                    assert v['n_recomputed_image_tokens'] == 32 and v['retained_image_context_ratio'] == 1.
                    components.append({'image_id': r['image_id'], 'dialog_id': r['dialog_id'], 'turn_id': r['turn_id'], **{k: v.get(k, 'NOT_MEASURED') for k in ('n_image_tokens', 'n_recomputed_image_tokens', 'n_reused_image_tokens', 'n_recomputed_text_tokens', 'embedding_read_ms', 'kv_read_ms', 'h2d_ms', 'cache_assembly_ms', 'decode_cache_assembly_ms', 'selective_prefill_interval_ms', 'ssd_embedding_bytes', 'metadata_resident_bytes')}})
        R.writecsv(dst / 'activation_residency.csv', activation_residency)
        residency_summary = []
        for method in methods:
            ars = [r for r in activation_residency if r['method_id'] == method]
            if not ars:
                continue
            for metric in list(ars[0])[3:]:
                vs = [float(r[metric]) for r in ars if r[metric] != 'NOT_MEASURED']
                residency_summary.append({'method_id': method, 'metric': metric, 'measured_images': len(vs), **R.stats(vs)})
        R.writecsv(dst / 'activation_residency_summary.csv', residency_summary)
        lines += [f'### {model}: Activation auxiliary residency와 service 무결성 검사', '', table(['Method', 'Metric', 'Measured images', 'Mean / p50 / p95 (bytes or ms as named)'], [[R.LABEL[r['method_id']], r['metric'], r['measured_images'], 'NOT_MEASURED' if not r['measured_images'] else ' / '.join(R.fmt(r[k], 3) for k in ('mean', 'p50', 'p95'))] for r in residency_summary]), '', 'Host tensor bytes는 structural/prefix와 AllHead의 v_hidden 등 실제 상주 텐서이며 단순 metadata bytes가 아니다. MPIC metadata-ready file bytes는 읽은 metadata 파일 크기이며 Python 객체 heap 크기가 아니다. Visual KV payload residency는 별도 열로 기록한다. AllHead의 미사용 probe sidecar는 activation integrity scan에서 검증되지만 followup serving read는 0이다. Service integrity interval은 activation outer wall에 이미 포함되며 중복 가산하지 않는다. 종료 후 독립 audit의 rehash는 이 service 비용 밖이다.', '']
        if components:
            assert len(components) == 8122
            R.writecsv(dst / 'mpic_followup_components.csv', components)
            summary = [{'metric': k, 'measured_requests': len(vs), **R.stats(vs)} for k in list(components[0])[3:] if (vs := [float(r[k]) for r in components if r[k] != 'NOT_MEASURED'])]
            R.writecsv(dst / 'mpic_component_summary.csv', summary)
            lines += [table(['MPIC measured component', 'N', 'Mean / p50 / p95'], [[r['metric'], r['measured_requests'], ' / '.join(R.fmt(r[k], 3) for k in ('mean', 'p50', 'p95'))] for r in summary]), '', 'MPIC의 32는 이미지 rows 수이며 32%가 아니다. Selective-prefill interval은 read/H2D/assembly와 active-row decoder를 포함한 중첩 interval이다. 순수 recompute GPU 시간은 NOT_MEASURED이며 위 component를 합해 TTFT를 만들지 않는다.', '']
        if model == 'llava':
            geometry = csvrows(src / 'image_budget_io.csv')
            diagnostics = {(r['dialog_id'], r['turn_id'], r['method_id']): r for r in csvrows(src / 'request_diagnostics.csv')}
            vrows = []
            for r in geometry:
                method = r['method_id']
                if method not in (R.AH, B):
                    continue
                s = store[(r['image_id'], method)]
                actual = float(diagnostics[(r['dialog_id'], r['turn_id'], method)]['selected_V_bytes'])
                exact_content_V = float(s['original_content_bytes']) / float(r['N']) * float(r['k']) / 2
                vrows.append({'image_id': r['image_id'], 'dialog_id': r['dialog_id'], 'turn_id': r['turn_id'], 'method_id': method, 'attention_content_ratio': float(r['attention_content_ratio']), 'V_chunk_coverage_layer_mean': number(r['V_chunk_coverage_layer_mean']), 'actual_V_payload_read_bytes': actual, 'exact_selected_content_V_bytes': exact_content_V, 'V_payload_over_exact_content_V': actual / exact_content_V})
            R.writecsv(dst / 'V_coverage_amplification.csv', vrows)
            values = []
            for method in (R.AH, B):
                rs = [r for r in vrows if r['method_id'] == method]
                covered = [r['V_chunk_coverage_layer_mean'] for r in rs if r['V_chunk_coverage_layer_mean'] is not None]
                pooled = sum(r['actual_V_payload_read_bytes'] for r in rs) / sum(r['exact_selected_content_V_bytes'] for r in rs)
                grouped = defaultdict(list)
                for r in rs:
                    grouped[r['image_id']].append(r['V_payload_over_exact_content_V'])
                values.append([R.LABEL[method], triple([r['attention_content_ratio'] for r in rs]), triple(covered) if covered else 'NOT_APPLICABLE: sequential full B payload', triple([r['V_payload_over_exact_content_V'] for r in rs]), f'{pooled:.6f}', f'{R.mean([R.mean(v) for v in grouped.values()]):.6f}'])
            lines += [f'### {model}: AllHead / B V chunk coverage와 read amplification', '', table(['Method', 'Attended content fraction mean / p50 / p95', 'V chunk coverage mean / p50 / p95', 'V read amplification request mean / p50 / p95', 'Pooled byte ratio', 'Image-macro mean ratio'], values), '', 'V amplification 분모는 original N에 대해 선택한 정확한 k content rows의 native V bytes다. 분자는 실제 v.bin read이며 AllHead에서는 covering chunks의 미선택·구조·padding rows도 포함할 수 있다. 별도 structural sidecar read는 이 V 비율 밖에 있고 total SSD read에는 포함된다. Coverage는 layer별 선택 V chunk 수 / ceil(full native visual rows/64)를 먼저 계산한 뒤 요청·layer를 동일 가중한다. B payload는 정확한 k content rows를 전부 읽으므로 이 content-V 비율은 1이다.', '']
    done = R.read(RUN / 'llava/main_execution_complete.json')
    wall = done['wall_seconds_including_conditioning_logging_audit_cleanup']
    lines += ['### 실행 wall time의 범위', '', f'모델 로딩·warmup 이후 smoke와 main을 포함한 launcher wall time: {wall:.3f} s ({wall / 3600:.3f} h). Conditioning/logging/이미지 감사/cleanup을 포함하며 request E2E와 setup 귀속 합과 다르다. GPU 사전검증 및 최종 독립 통계/보존 검사는 이 wall time 밖이다.', '']
    text = '\n'.join(lines)
    (DEST / 'DETAILS.md').write_text(text)
    base_report = (FINAL / 'REPORT.md').read_text().replace('상위 25% pruning이나 GPU 메모리 75% 감소를 구현한 실험이 아니다.', 'B는 중요도 상위 25% content만 저장하고 attention에 사용한다. LLaVA의 기존 dense GPU buffer를 유지하므로 GPU 메모리 75% 감소를 뜻하지 않는다.')
    (OUT / 'REPORT_with_details.md').write_text(base_report + '\n' + text)
    R.dump(DEST / 'provenance.json', {'source_sha256': R.sha(__file__), 'inputs_sha256': inputs, 'parent_report_sha256': R.sha(FINAL / 'REPORT.md'), 'timestamp_unix': time.time(), 'inference_source_changed': False})
    print(json.dumps({'status': 'PASS', 'report': str(OUT / 'REPORT_with_details.md')}), flush=True)


if __name__ == '__main__':
    main()

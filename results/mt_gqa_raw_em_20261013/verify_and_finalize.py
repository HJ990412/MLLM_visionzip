"""Independently check exported original strings/counts and finalize the new report."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def normalized_reference(text):
    # Independent character implementation of the established regex rule.
    text = ''.join(c if c.isalnum() or c == '_' or c.isspace() else ' ' for c in text.lower())
    return ' '.join(token for token in text.split() if token not in ('a', 'an', 'the'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', type=Path, required=True)
    args = parser.parse_args()
    p = args.analysis.resolve()
    validation = json.loads((p / 'raw_em_validation.json').read_text())
    assert validation['status'] == 'PASS'
    counts, followup_clusters = defaultdict(Counter), defaultdict(lambda: np.zeros(3, dtype=np.int64))
    example_sources = {}
    count = 0
    with (p / 'raw_em_requests.csv').open() as f:
        for row in csv.DictReader(f):
            model, method, turn = row['model'], row['method_id'], int(row['turn_id'])
            raw = int(row['prediction'] == row['gold'])
            norm = int(normalized_reference(row['prediction']) == normalized_reference(row['gold']))
            assert raw == int(row['raw_em']) and norm == int(row['normalized_em']) and raw <= norm
            for scope in [f'T{turn}', 'all'] + (['followup'] if turn > 1 else []):
                counts[model, method, scope].update(requests=1, raw_correct=raw, normalized_correct=norm)
            if turn > 1:
                followup_clusters[model, method, row['image_id']] += [raw, norm, 1]
            if norm and not raw:
                example_sources[model, method, row['image_id'], row['dialog_id'], str(turn)] = (row['prediction'], row['gold'])
            count += 1
    summary = read_csv(p / 'raw_em_summary.csv')
    for row in summary:
        c = counts[row['model'], row['method_id'], row['scope']]
        assert all(c[k] == int(row[k]) for k in ('requests', 'raw_correct', 'normalized_correct'))
        for metric, key in [('raw_em_percent', 'raw_correct'), ('normalized_em_percent', 'normalized_correct')]:
            assert abs(float(row[metric])-100*c[key]/c['requests']) < 1e-10
    examples = read_csv(p / 'raw_em_examples.csv')
    assert len(examples) <= 20
    for row in examples:
        pair = example_sources[row['model'], row['method_id'], row['image_id'], row['dialog_id'], row['turn']]
        assert pair == (row['prediction'], row['gold'])
    paired = read_csv(p / 'raw_em_paired.csv')
    checked = 0
    for row in paired:
        if row['scope'] != 'followup':
            continue
        model = row['model']
        images = sorted(k[2] for k in followup_clusters if k[:2] == (model, row['ours_method_id']))
        assert len(images) == 398
        b = np.array([followup_clusters[model, row['ours_method_id'], im] for im in images])
        r = np.array([followup_clusters[model, row['baseline_id'], im] for im in images])
        assert np.array_equal(b[:, 2], r[:, 2])
        draws = np.random.default_rng(1234).choice(398, size=(10000, 398), replace=True)
        delta = b[:, 0] - r[:, 0]
        samples = 100*delta[draws].sum(axis=1)/b[draws, 2].sum(axis=1)
        actual = [100*delta.sum()/b[:, 2].sum(), *np.quantile(samples, [0.025, 0.975])]
        expected = [float(row[k]) for k in ('raw_em_delta_pp', 'raw_em_ci_low_pp', 'raw_em_ci_high_pp')]
        assert max(abs(x-y) for x,y in zip(actual, expected)) < 1e-10
        checked += 1
    assert count == 134013 and checked == 8
    write_json(p / 'independent_csv_verification.json', {
        'status': 'PASS', 'scope': 'Exported original prediction/gold strings, independent character-based normalization, counts, examples and followup raw bootstrap. Original raw integrity is established separately by raw_em_validation.json.',
        'requests': count, 'summary_scopes': len(summary), 'examples': len(examples),
        'independent_followup_raw_bootstrap_checks': checked, 'seed': 1234, 'resamples': 10000,
        'code_sha256': sha(Path(__file__))})
    root = Path(__file__).resolve().parents[2]
    decoder_info = []
    for model in validation['models']:
        run = root / 'runs' / model['run']
        config_path = run / ('llava/config.json' if model['model'] == 'LLaVA' else 'config.json')
        config = json.loads(config_path.read_text())
        tokenizer_path = (Path(config['model']['snapshot']) / 'tokenizer_config.json'
                          if model['model'] != 'GLM' else run / 'checkpoint/tokenizer_config.json')
        tok = json.loads(tokenizer_path.read_text())
        decoder_info.append({'model': model['model'], 'config_path': str(config_path), 'config_sha256': sha(config_path),
            'tokenizer_config_path': str(tokenizer_path), 'tokenizer_config_sha256': sha(tokenizer_path),
            'tokenizer_class': tok.get('tokenizer_class'),
            'saved_clean_up_tokenization_spaces': tok.get('clean_up_tokenization_spaces', 'NOT_SPECIFIED'),
            'decoder_processing': model['decoder_preprocessing'], 'tokenizer_loaded': False})
    write_json(p / 'decoder_preprocessing_metadata.json', decoder_info)
    lines = ['', '## 추가 독립 검산과 핵심 해석', '',
        '별도 verify_and_finalize.py가 exported prediction/gold 문자열134,013개를 다시 직접 비교하고, regex scorer를 import하지 않는 문자 기반 정규화로55개 summary를 검산했다. Followup Raw EM paired CI8개도 별도 choice-resampling으로 재계산해 모두 일치했다.',
        'LLaVA는 T1/T2/T3/All/Followup에서 모든 방법 Raw EM0이다. Normalized 정답40,435개는 전부 대소문자만 다르다. 이것은 저장 답변의 표기 때문에 Raw EM이 바닥값에 머문 결과다. Δ0/CI[0,0]는 품질 동등성 증거가 아니며 압축 품질을 구분하지 못한다.',
        'Qwen도 정규화로 추가 정답이 된 모든 요청은 대소문자 차이다. GLM은 대소문자 외 관사·구두점 등의 차이도 존재한다. 진단 변환은 원인 분류에만 사용했고 Raw EM의 비교 문자열은 바꾸지 않았다.',
        'LLaVA의 pinned tokenizer_config는 clean_up_tokenization_spaces=false다. LLaVA decoder는 skip_special_tokens 후 strip을 이미 적용했다. Qwen은 cleanup=false를 decode 호출에 명시하고 strip을 적용했다. GLM은 native final-answer span 추출 후 strip한다. 추가 metadata는 decoder_preprocessing_metadata.json에 hash와 함께 보존했다. 이 Raw EM은 이 처리 이후의 저장 prediction을 평가한다.', '',
        '| Model | Ours−FullLoad Followup Normalized pp | Raw pp | Gap 변화 pp [95% CI] |',
        '|---|---:|---:|---:|']
    for row in paired:
        if row['scope'] == 'followup' and row['baseline_id'] == 'fullload':
            lines.append(f'| {row["model"]} | {float(row["normalized_em_delta_pp"]):+.3f} | {float(row["raw_em_delta_pp"]):+.3f} | {float(row["raw_minus_normalized_gap_change_pp"]):+.3f} [{float(row["gap_change_ci_low_pp"]):+.3f}, {float(row["gap_change_ci_high_pp"]):+.3f}] |')
    lines += ['',
        '기존 연구 표가 Normalized EM이라고 명시돼 있다면 기존 정확도 숫자를 교체할 필요는 없다. Raw EM을 함께 보고해 표기 민감도를 공개한다. 만약 manuscript에서 그 숫자를 Raw EM 또는 공식 GQA/MetaCompress의 동일 평가 결과라고 설명했다면 해당 지표명과 재현 주장은 수정해야 한다. 기존 run/report/code는 이번 작업에서 수정하지 않았다.', '',
        '전체 재현에서는 앞의 분석 명령에 이어 다음 검산·보충 명령을 실행한다.', '', '```bash',
        '.venv-glm46v/bin/python results/mt_gqa_raw_em_20261013/verify_and_finalize.py \\',
        '  --analysis results/mt_gqa_raw_em_REPLAY_NEW', '```']
    with (p / 'RAW_EM_REPORT.md').open('a') as f:
        f.write('\n'.join(lines) + '\n')
    write_json(p / 'artifact_hashes.json', {
        'scope': 'Final new rescore outputs and code; self excluded. Original source/raw/evidence hashes are in source_files_sha256.json.',
        'files': {f.name: sha(f) for f in sorted(p.iterdir()) if f.is_file() and f.name != 'artifact_hashes.json'},
        'code': {str(Path(__file__)): sha(Path(__file__)),
                 str(Path(__file__).parent / 'analyze_raw_em.py'): sha(Path(__file__).parent / 'analyze_raw_em.py')}})
    print(json.dumps({'status': 'PASS', 'requests': count, 'summary_scopes': len(summary), 'followup_raw_CIs': checked}))


if __name__ == '__main__':
    main()

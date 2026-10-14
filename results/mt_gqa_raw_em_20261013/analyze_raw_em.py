#!/usr/bin/env python3
"""Read-only post-hoc scoring. Imports no model, tokenizer, or production scorer."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import time

import numpy as np

SCOPES = ('T1', 'T2', 'T3', 'all', 'followup')
LABELS = {'recompute': 'ReComp', 'fullload': 'FullLoad', 'mpic32_ssd': 'MPIC-32',
          'sparsevlm_ssd_kv25_allhead': 'AllHead', 'ours_prefixstore_kv25': 'Ours',
          'qwen_ours_prefixstore_kv25': 'Ours'}
SPECS = [
    ('LLaVA', 'mt_gqa_B_llava5_qwen3_20261001T093921Z', 'llava',
     ('recompute', 'fullload', 'mpic32_ssd', 'sparsevlm_ssd_kv25_allhead', 'ours_prefixstore_kv25')),
    ('Qwen', 'qwen_mt_gqa_prefixstore3_20261002T074856Z', '',
     ('recompute', 'fullload', 'qwen_ours_prefixstore_kv25')),
    ('GLM', 'glm46v_prefixstore3_v6_20261007T104505Z', '',
     ('recompute', 'fullload', 'ours_prefixstore_kv25')),
]
PREPROCESSING = {
    'LLaVA': 'Frozen native tokenizer.decode(skip_special_tokens=True).strip(); tokenizer default cleanup is not independently disabled. serve.py / mpic.py / sparsevlm_ssd_attention.py. Raw EM compares stored prediction after this existing decoder processing.',
    'Qwen': 'Frozen tokenizer.decode(skip_special_tokens=True, clean_up_tokenization_spaces=False).strip(). Raw EM compares stored prediction after this existing decoder processing.',
    'GLM': 'Frozen native-final-answer-box-v1 extracts a closed native answer span; malformed/plain output uses literal decode. skip_special_tokens=True, clean_up_tokenization_spaces=False, then strip(). Actual native assistant body is a separate saved history field. Raw EM compares the stored scoring prediction, not the token stream or history body.',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def norm(text):
    return ' '.join(w for w in re.sub(r'[^\w\s]', ' ', text.lower()).split()
                    if w not in {'a', 'an', 'the'})


def require(condition, description):
    if not condition:
        raise ValueError(description)


def save_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def save_csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def scopes(turn):
    return (f'T{turn}', 'all') + (('followup',) if turn > 1 else ())


def reason(prediction, gold):
    # Diagnostics only. Never used for Raw EM, prediction, or history.
    if prediction.lower() == gold.lower():
        return 'case_only'
    if prediction.strip() == gold.strip():
        return 'outer_whitespace_only'
    p, g = prediction.lower(), gold.lower()
    if ' '.join(p.split()) == ' '.join(g.split()):
        return 'case_and_whitespace'
    def punctuation(x):
        return ' '.join(re.sub(r'[^\w\s]', ' ', x).split())
    if punctuation(p) == punctuation(g):
        return 'case_punctuation_or_spacing'
    def articles(x):
        return ' '.join(w for w in x.split() if w not in {'a', 'an', 'the'})
    if articles(p) == articles(g):
        return 'case_articles_or_spacing'
    return 'combined_normalization'


class Analysis:
    def __init__(self, root, output):
        self.root, self.output = root.resolve(), output.resolve()
        self.protected = {}
        self.inventory = []
        self.validation = []
        self.summary, self.sensitivity, self.paired = [], [], []
        self.candidates, self.reasons = {}, Counter()
        self.started = time.monotonic()

    def protect(self, path, expected=None):
        path = Path(path).resolve()
        actual = self.protected.get(str(path))
        if actual is None:
            actual = sha(path)
            self.protected[str(path)] = actual
        require(expected is None or actual == expected, f'SHA256 mismatch: {path}')
        return actual

    def read_json(self, path, expected=None):
        self.protect(path, expected)
        return json.loads(Path(path).read_text())

    def read_csv(self, path):
        self.protect(path)
        with Path(path).open() as stream:
            return list(csv.DictReader(stream))

    def model(self, model, name, sub, methods, request_writer):
        run = self.root / 'runs' / name
        base = run / sub if sub else run
        prior = self.root / 'results' / name
        prior_final = prior / 'final' / 'llava' if model == 'LLaVA' else prior / 'final'
        completion_path = (prior_final / 'independent_audit.json' if model == 'LLaVA' else
                           prior / 'FINAL_STATUS.json' if model == 'Qwen' else base / 'main/COMPLETED.json')
        completion = self.read_json(completion_path)
        require(completion['status'] == 'PASS', f'{model}: incomplete or BLOCKED')
        manifest_path = base / ('manifests/main_manifest.json' if model == 'GLM' else 'manifest.json')
        manifest = self.read_json(manifest_path)
        config_path = base / 'config.json'
        config = self.read_json(config_path)
        dialogs = manifest['dialogues']
        require(len(dialogs) == 4061, f'{model}: dialogue count')
        expected = {(d['dialog_id'], t): (d['image_id'], d['turns'][t-1])
                    for d in dialogs for t in (1, 2, 3)}
        require(len(expected) == 12183, f'{model}: duplicate manifest membership')
        image_ids = sorted({d['image_id'] for d in dialogs})
        require(len(image_ids) == 398, f'{model}: image count')
        image_index = {iid: i for i, iid in enumerate(image_ids)}
        if model == 'GLM':
            index_path = prior / 'local_artifact_index.json'
            historical = self.read_json(index_path)
            anchors = {r['commit_path']: r['commit_sha256'] for r in historical['raw']}
            anchors.update({r['raw_path']: r['raw_sha256'] for r in historical['raw']})
            freeze = self.read_json(base / 'freeze.json', completion['freeze_sha256'])
            self.protect(config_path, freeze['config_sha256'])
            self.protect(manifest_path, freeze['manifests']['manifests/main_manifest.json'])
        else:
            index_path = prior / ('artifact_index_final.json' if model == 'Qwen' else 'final/artifact_index.json')
            anchors = {r['path']: r['sha256'] for r in self.read_json(index_path)['files'] if 'sha256' in r}
        old_quality = {(r['method_id'], r['scope']): r for r in self.read_csv(prior_final / 'quality.csv')}
        old_pairs = {(r['baseline'], r['scope'], r['metric']): r for r in self.read_csv(prior_final / 'paired_analysis.csv')}
        self.protect(prior / 'REPORT.md')
        # Source references are inspected/read only; no inference imports.
        for relative in config.get('source_sha256', {}):
            source = self.root / relative
            if source.is_file():
                self.protect(source)
        for relative in ['scripts/108_eval_llava_mt_gqa_B.py', 'scripts/109_audit_llava_mt_gqa_B.py',
                         'scripts/113_report_mt_gqa_B.py', 'docs/mt_gqa_B_main_contract.md']:
            self.protect(self.root / relative)
        commits = sorted((base / 'main/images').glob('*/COMMITTED.json'))
        require(len(commits) == 398, f'{model}: committed image coverage')
        seen = {m: set() for m in methods}
        logical, physical = set(), set()
        counters = defaultdict(Counter)
        # [scope, method, image, raw-correct / normalized-correct / count]
        totals = np.zeros((5, len(methods), 398, 3), dtype=np.int64)
        scope_ix = {s: i for i, s in enumerate(SCOPES)}
        method_ix = {m: i for i, m in enumerate(methods)}
        rawscore_present = rawscore_absent = histories_checked = evidence_files = 0
        for image_number, commit_path in enumerate(commits, 1):
            key = str(commit_path.resolve())
            require(key in anchors, f'{model}: no historical commit anchor {key}')
            commit = self.read_json(commit_path, anchors[key])
            require(commit['status'] in ('COMMITTED', 'PASS'), f'{model}: bad commit status')
            require(commit['image_id'] == commit_path.parent.name, f'{model}: commit image identity')
            require(re.fullmatch(r'attempt_\d+', commit['attempt_id']) is not None, 'unsafe attempt name')
            folder = commit_path.parent / commit['attempt_id']
            raw_path = folder / 'raw.jsonl'
            if 'raw_path' in commit:
                require(Path(commit['raw_path']).resolve() == raw_path.resolve(), 'raw owner path mismatch')
            require(str(raw_path.resolve()) in anchors and anchors[str(raw_path.resolve())] == commit['raw_sha256'], 'raw historical anchor mismatch')
            if model != 'GLM':
                self.protect(config_path, commit['config_sha256'])
                self.protect(manifest_path, commit['manifest_sha256'])
            for relative, digest in commit.get('evidence_sha256', {}).items():
                candidate = (folder / relative).resolve()
                require(candidate.is_relative_to(folder.resolve()), 'unsafe evidence owner path')
                if relative == 'raw.jsonl':
                    require(digest == commit['raw_sha256'], 'raw evidence digest differs')
                    continue
                self.protect(candidate, digest)
                evidence_files += 1
            if model != 'GLM':
                audit = self.read_json(folder / 'independent_audit.json', commit.get('audit_sha256'))
                require(audit['status'] == 'PASS' and audit['raw_sha256'] == commit['raw_sha256'], 'committed audit invalid')
            else:
                require(commit['phase'] == 'main' and commit['audit']['status'] == 'PASS', 'GLM embedded audit invalid')
                self.protect(folder / 'setup_events.jsonl', commit['setup_sha256'])
                self.protect(folder / 'payload_evidence.json', commit['payload_evidence_sha256'])
                evidence_files += 2
            local = {}
            digest = hashlib.sha256()
            row_count = 0
            with raw_path.open('rb') as stream:
                for line in stream:
                    digest.update(line)
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    row_count += 1
                    m, did, turn, iid = r['method_id'], r['dialog_id'], r['turn_id'], r['image_id']
                    require(m in methods and turn in (1, 2, 3), 'unexpected method/turn')
                    require(r['phase'] == 'main' and r['attempt_id'] == commit['attempt_id'] and iid == commit['image_id'], 'non-adopted raw row')
                    require(r.get('status', 'PASS') == 'PASS', 'failed raw request')
                    require((did, turn) in expected, 'unknown dialogue/turn')
                    target_image, target = expected[did, turn]
                    require(iid == target_image and r['question_id'] == str(target['question_id']) and r['question'] == target['question'] and r['gold'] == target['answers'], 'raw differs from frozen workload/gold')
                    prediction, gold = r['prediction'], r['gold'][0]
                    require(isinstance(prediction, str) and isinstance(gold, str), 'prediction/gold must be strings')
                    raw_em = int(prediction == gold)  # NO preprocessing, not even strip().
                    normalized_em = int(norm(prediction) == norm(gold))
                    require(raw_em <= normalized_em, 'raw=1 normalized=0: scoring defect')
                    if 'score' in r:
                        require(r['score'] == normalized_em, 'stored score differs from normalized EM')
                        rawscore_present += 1
                    else:
                        require(model == 'GLM', 'required stored score missing')
                        rawscore_absent += 1
                    require((did, turn) not in seen[m], 'duplicate dialogue/turn execution')
                    seen[m].add((did, turn))
                    require(r['logical_request_id'] not in logical and r['physical_execution_id'] not in physical, 'duplicate request identity')
                    logical.add(r['logical_request_id']); physical.add(r['physical_execution_id'])
                    for scope in scopes(turn):
                        totals[scope_ix[scope], method_ix[m], image_index[iid]] += (raw_em, normalized_em, 1)
                        counters[m, scope].update(requests=1, raw_correct=raw_em, normalized_correct=normalized_em,
                                                 normalized_only=normalized_em - raw_em)
                    if normalized_em and not raw_em:
                        why = reason(prediction, gold)
                        self.reasons[model, m, why] += 1
                        category = (model, m, why)
                        if category not in self.candidates:
                            self.candidates[category] = {'model': model, 'method_id': m, 'method': LABELS[m],
                                'image_id': iid, 'dialog_id': did, 'turn': turn, 'prediction': prediction,
                                'gold': gold, 'raw_em': raw_em, 'normalized_em': normalized_em,
                                'diagnostic_category': why, 'prediction_json': json.dumps(prediction, ensure_ascii=False),
                                'gold_json': json.dumps(gold, ensure_ascii=False)}
                    request_writer.writerow({'model': model, 'method_id': m, 'image_id': iid, 'dialog_id': did,
                        'turn_id': turn, 'logical_request_id': r['logical_request_id'], 'physical_execution_id': r['physical_execution_id'],
                        'prediction': prediction, 'gold': gold, 'raw_em': raw_em, 'normalized_em': normalized_em,
                        'stored_score': r.get('score', 'NOT_PRESENT_IN_RAW')})
                    local[did, m, turn] = {k: r[k] for k in ['prediction', 'logical_request_id', 'physical_execution_id', 'history_entries']}
                    if model == 'GLM':
                        local[did, m, turn]['history_text'] = r['history_text']
            actual = digest.hexdigest()
            require(actual == commit['raw_sha256'], f'raw SHA256 mismatch {raw_path}')
            self.protected[str(raw_path.resolve())] = actual
            require(row_count == commit['requests'], 'commit request count differs')
            for (did, m, turn), r in local.items():
                require(len(r['history_entries']) == turn - 1, 'history length differs')
                for prior_turn, entry in enumerate(r['history_entries'], 1):
                    previous = local[did, m, prior_turn]
                    require(entry['source_method_id'] == m and entry['source_dialog_id'] == did and
                            entry['source_logical_request_id'] == previous['logical_request_id'] and
                            entry['source_physical_execution_id'] == previous['physical_execution_id'], 'history method/source leakage')
                    if model == 'GLM':
                        require(entry['history_text'] == previous['history_text'] and entry['prediction'] == previous['prediction'], 'native generated history differs')
                    else:
                        require(entry['answer'] == previous['prediction'], 'generated history differs')
                        if model == 'LLaVA':
                            require(entry['answer_source'] == 'method_local_generated', 'history answer-source differs')
                    histories_checked += 1
            self.inventory.append({'model': model, 'image_id': commit['image_id'], 'attempt_id': commit['attempt_id'],
                'commit_path': str(commit_path.resolve()), 'commit_sha256': self.protected[key],
                'raw_path': str(raw_path.resolve()), 'raw_sha256': actual, 'requests': row_count})
            if image_number % 40 == 0:
                print(json.dumps({'model': model, 'verified_images': image_number, 'requests': len(logical)}), flush=True)
        require(all(keys == set(expected) for keys in seen.values()), 'method membership differs')
        require(len(logical) == 12183 * len(methods), 'full main coverage differs')
        original_checks = 0
        for m in methods:
            for scope in SCOPES:
                c = counters[m, scope]
                n = c['requests']
                require(n == (12183 if scope == 'all' else 8122 if scope == 'followup' else 4061), 'scope count differs')
                raw_pct, norm_pct = 100*c['raw_correct']/n, 100*c['normalized_correct']/n
                historical = old_quality[m, scope]
                require(int(historical['requests']) == n and float(historical['correct']) == c['normalized_correct'], 'existing normalized CSV counts differ')
                published_pct = float(historical['accuracy']) * (1 if model == 'GLM' else 100)
                require(abs(published_pct - norm_pct) < 1e-10, 'existing normalized accuracy differs')
                original_checks += 1
                self.summary.append({'model': model, 'run': name, 'method_id': m, 'method': LABELS[m], 'scope': scope,
                    'requests': n, 'raw_correct': c['raw_correct'], 'raw_em_percent': raw_pct,
                    'normalized_correct': c['normalized_correct'], 'normalized_em_percent': norm_pct})
                self.sensitivity.append({'model': model, 'method_id': m, 'method': LABELS[m], 'scope': scope,
                    'requests': n, 'normalized_em_percent': norm_pct, 'raw_em_percent': raw_pct,
                    'normalized_minus_raw_pp': norm_pct - raw_pct, 'normalized_1_raw_0_count': c['normalized_only'],
                    'normalized_1_raw_0_percent_of_requests': 100*c['normalized_only']/n,
                    'normalized_1_raw_0_percent_of_normalized_correct': 100*c['normalized_only']/c['normalized_correct'] if c['normalized_correct'] else 'NOT_DEFINED',
                    'raw_1_normalized_0_count': 0})
        draws = np.random.default_rng(1234).integers(0, 398, size=(10000, 398))
        def bootstrap(values, counts):
            estimates = 100*values[draws].sum(axis=1)/counts[draws].sum(axis=1)
            return [float(x) for x in np.percentile(estimates, [2.5, 97.5])]
        ours_index = len(methods) - 1
        ci_checks = 0
        for mi, m in enumerate(methods[:-1]):
            for si, scope in enumerate(SCOPES):
                b, ref = totals[si, ours_index], totals[si, mi]
                require(np.array_equal(b[:, 2], ref[:, 2]), 'paired per-image request counts differ')
                n = int(b[:, 2].sum())
                raw_delta, normalized_delta = b[:, 0]-ref[:, 0], b[:, 1]-ref[:, 1]
                raw_point, normalized_point = 100*raw_delta.sum()/n, 100*normalized_delta.sum()/n
                raw_ci, normalized_ci = bootstrap(raw_delta, b[:, 2]), bootstrap(normalized_delta, b[:, 2])
                historical = old_pairs[m, scope, 'accuracy_pp']
                require(max(abs(float(historical[k])-v) for k, v in zip(('point', 'ci_low', 'ci_high'), (normalized_point, *normalized_ci))) < 1e-10, 'historical normalized paired CI differs')
                ci_checks += 1
                shift = raw_delta - normalized_delta
                shift_ci = bootstrap(shift, b[:, 2])
                self.paired.append({'model': model, 'ours_method_id': methods[-1], 'baseline_id': m, 'baseline': LABELS[m],
                    'scope': scope, 'requests': n, 'images': 398,
                    'raw_em_delta_pp': float(raw_point), 'raw_em_ci_low_pp': raw_ci[0], 'raw_em_ci_high_pp': raw_ci[1],
                    'normalized_em_delta_pp': float(normalized_point), 'normalized_em_ci_low_pp': normalized_ci[0],
                    'normalized_em_ci_high_pp': normalized_ci[1], 'raw_minus_normalized_gap_change_pp': float(100*shift.sum()/n),
                    'gap_change_ci_low_pp': shift_ci[0], 'gap_change_ci_high_pp': shift_ci[1],
                    'seed': 1234, 'resamples': 10000, 'CI': '95% percentile',
                    'weighting': 'paired image cluster; pooled request sums/counts'})
        self.validation.append({'model': model, 'run': name, 'status': 'PASS', 'images': 398, 'dialogues': 4061,
            'methods': list(methods), 'main_requests': len(logical), 'requests_per_method': 12183,
            'followups_per_method': 8122, 'committed_images': len(commits), 'historical_commit_and_raw_anchors': 'PASS',
            'committed_evidence_files_verified': evidence_files, 'identical_method_membership': True,
            'stored_normalized_scores_checked': rawscore_present, 'stored_score_absent_requests': rawscore_absent,
            'stored_score_check_scope': 'All saved score fields exact; GLM raw has no score field, historical aggregate counts checked instead.' if model == 'GLM' else 'Every request score exactly matches independently recomputed normalized EM.',
            'historical_normalized_summary_scopes_checked': original_checks, 'historical_normalized_paired_CIs_checked': ci_checks,
            'history_source_entries_checked': histories_checked, 'raw_1_normalized_0': 0,
            'excluded_paths': 'All non-main, non-COMMITTED, non-adopted attempts; earlier BLOCKED Qwen and GLM v3-v5.',
            'decoder_preprocessing': PREPROCESSING[model]})
        print(json.dumps({'model': model, 'status': 'PASS', 'requests': len(logical)}), flush=True)

    def report(self):
        lines = ['# MT-GQA Raw Exact Match 사후 재채점', '',
            '분석 기준일: 2026-10-13 (Asia/Seoul). 기존 저장 prediction과 gold[0]만 비교했다.',
            '`raw_em = int(prediction == gold[0])`; 문자열을 strip/lower/정규화하지 않았다.',
            'Normalized EM은 lower → 구두점 공백 치환 → a/an/the 제거 → 공백 정리 후 exact 비교다.',
            '추론, prompt, 데이터, history, KV, GPU 및 기존 평가 파일을 변경하지 않았다.',
            'Raw EM은 decoder가 이미 처리하여 저장한 prediction 기준이다. 생성 token stream의 무가공 일치율이나 공식 GQA/MetaCompress 재현을 뜻하지 않는다.', '']
        for v in self.validation:
            model = v['model']
            lines += [f'## {model}: {v["run"]}', '',
                f'398 images / 4,061 dialogues / 방법당 12,183 main 요청, 8,122 followups. 총 {v["main_requests"]:,}요청.',
                f'무결성·membership·기존 Normalized EM 재계산 PASS. 요청별 저장 score 대조 {v["stored_normalized_scores_checked"]:,}개, 없는 score {v["stored_score_absent_requests"]:,}개.',
                PREPROCESSING[model], '', '| Method | T1 Raw EM | T2 Raw EM | T3 Raw EM | All Raw EM | Followup Raw EM |',
                '|---|---:|---:|---:|---:|---:|']
            for m in v['methods']:
                rr = {(x['scope']): x for x in self.summary if x['model'] == model and x['method_id'] == m}
                cells = [f'{rr[s]["raw_em_percent"]:.3f}% ({rr[s]["raw_correct"]}/{rr[s]["requests"]})' for s in SCOPES]
                lines.append('| ' + LABELS[m] + ' | ' + ' | '.join(cells) + ' |')
            lines += ['', '| Method | All Normalized / Raw % | 차이 pp | Followup Normalized / Raw % | 차이 pp | normalized=1/raw=0 followups |', '|---|---:|---:|---:|---:|---:|']
            for m in v['methods']:
                ss = {x['scope']: x for x in self.sensitivity if x['model'] == model and x['method_id'] == m}
                a, f = ss['all'], ss['followup']
                lines.append(f'| {LABELS[m]} | {a["normalized_em_percent"]:.3f} / {a["raw_em_percent"]:.3f} | {a["normalized_minus_raw_pp"]:.3f} | {f["normalized_em_percent"]:.3f} / {f["raw_em_percent"]:.3f} | {f["normalized_minus_raw_pp"]:.3f} | {f["normalized_1_raw_0_count"]}/{f["requests"]} ({f["normalized_1_raw_0_percent_of_requests"]:.3f}%) |')
            lines += ['', '### Ours−baseline: Followup Raw EM', '', '| Baseline | Raw Δ pp [95% CI] | Normalized Δ pp [95% CI] | Raw−Normalized gap change pp |', '|---|---:|---:|---:|']
            for x in self.paired:
                if x['model'] == model and x['scope'] == 'followup':
                    lines.append(f'| {x["baseline"]} | {x["raw_em_delta_pp"]:+.3f} [{x["raw_em_ci_low_pp"]:+.3f}, {x["raw_em_ci_high_pp"]:+.3f}] | {x["normalized_em_delta_pp"]:+.3f} [{x["normalized_em_ci_low_pp"]:+.3f}, {x["normalized_em_ci_high_pp"]:+.3f}] | {x["raw_minus_normalized_gap_change_pp"]:+.3f} |')
            lines.append('')
        lines += ['## 판정이 달라진 사례', '',
            '최대20개 deterministic 사례. 원본 문자열을 JSON 표기로 표시해 공백·구두점·대소문자를 보존했다. 두 판정은 Normalized=1, Raw=0이다.', '',
            '| Model / method | Image / dialogue / turn | Prediction | Gold | 원인 |', '|---|---|---|---|---|']
        for e in self.examples:
            pred, gold = e['prediction_json'].replace('|', '\\|'), e['gold_json'].replace('|', '\\|')
            lines.append(f'| {e["model"]} / {e["method"]} | {e["image_id"]} / {e["dialog_id"]} / T{e["turn"]} | {pred} | {gold} | {e["diagnostic_category"]} |')
        lines += ['', '## 해석·기존 결과 처리', '',
            '기존 Normalized EM 수치가 모든 방법·범위에서 재계산과 일치하므로 그 수치를 잘못된 결과로 수정할 이유는 없다. 기존 표의 지표명을 Normalized EM으로 유지하고 Raw EM을 보조 지표로 병기한다.',
            'Raw EM은 답변의 표기 형식에도 민감하다. 원인 빈도는 raw_em_difference_categories.csv, 원본 예시는 raw_em_examples.csv를 참조한다. Ours−FullLoad의 gap 변화는 위 표와 raw_em_paired.csv에서 확인한다.',
            '공식 GQA와 동일 evaluator 또는 MetaCompress의 정확한 dialogue/normalization/history 재현을 주장하지 않는다. 외부 논문의 발표 수치를 이 분석으로 수정할 수 없다.',
            'CI는 seed1234/10,000회/398paired image clusters의 percentile95%다. 동일 이미지의 모든 요청과 방법을 함께 resample하고 pooled sum/count로 요청 가중치를 유지한다. CI가0을 포함해도 동등성을 뜻하지 않는다.',
            'GLM은 원래 raw에 score가 없으므로 요청별 기존 score 일치를 주장하지 않는다. 저장 prediction을 직접 재채점하고 기존15개 Normalized EM 집계 및10개 paired accuracy CI를 확인했다.', '',
            '## 산출물·재현', '',
            '- raw_em_summary.csv: 모든 scope의 정답수/요청수/Raw·Normalized EM.',
            '- raw_em_paired.csv: Raw·Normalized EM Δ와95%CI, Ours−baseline gap 변화.',
            '- raw_em_sensitivity.csv: 정규화로 추가 정답이 된 count/rate.',
            '- raw_em_examples.csv: 전체20개 이내 원본 사례.',
            '- raw_em_requests.csv: adopted 요청별 두 score와 원본prediction/gold.',
            '- raw_em_validation.json / raw_em_evidence_index.json / source_files_sha256.json: full hashes·무결성·읽기 범위.',
            '- analyze_raw_em.py: numpy와표준라이브러리만 사용. 모델/토크나이저/추론 코드를 import하지 않는다.', '',
            '기존 결과 디렉터리를 overwrite하지 않고 없는 새 경로로 재현한다.', '', '```bash',
            'cd /home/dblab/hj/mllm_v2',
            '.venv-glm46v/bin/python results/mt_gqa_raw_em_20261013/analyze_raw_em.py \\',
            '  --root /home/dblab/hj/mllm_v2 --output results/mt_gqa_raw_em_REPLAY_NEW', '```']
        with (self.output / 'RAW_EM_REPORT.md').open('x') as stream:
            stream.write('\n'.join(lines) + '\n')

    def finish(self):
        # Ensure one example per model/method, then add category diversity, total <=20.
        self.examples, chosen, represented = [], set(), set()
        for key, value in self.candidates.items():
            if key[:2] not in represented:
                self.examples.append(value); chosen.add(key); represented.add(key[:2])
        for key, value in self.candidates.items():
            if key not in chosen and len(self.examples) < 20:
                self.examples.append(value)
        for name, rows in [('raw_em_summary.csv', self.summary), ('raw_em_paired.csv', self.paired),
                           ('raw_em_sensitivity.csv', self.sensitivity), ('raw_em_examples.csv', self.examples),
                           ('raw_em_difference_categories.csv', [{'model': k[0], 'method_id': k[1], 'category': k[2], 'count': v} for k,v in self.reasons.items()])]:
            save_csv(self.output / name, rows)
        self.report()
        save_json(self.output / 'raw_em_evidence_index.json', self.inventory)
        save_json(self.output / 'source_files_sha256.json', self.protected)
        print(json.dumps({'stage': 'verify_source_after', 'protected_files': len(self.protected)}), flush=True)
        for i, (path, expected) in enumerate(self.protected.items(), 1):
            require(sha(path) == expected, f'original input changed during rescore: {path}')
            if i % 2000 == 0:
                print(json.dumps({'stage': 'verify_source_after', 'files': i}), flush=True)
        save_json(self.output / 'raw_em_validation.json', {
            'status': 'PASS', 'analysis_date_client': '2026-10-13', 'models': self.validation,
            'raw_em_formula': 'int(stored_prediction == stored_gold[0]); no preprocessing',
            'normalized_em_formula': 'lower; punctuation to spaces; remove a/an/the; whitespace collapse; exact match',
            'original_source_raw_evidence_hashes_unchanged': True, 'protected_files': len(self.protected),
            'total_adopted_main_requests': sum(v['main_requests'] for v in self.validation),
            'raw_greater_than_normalized_count': 0, 'bootstrap_seed': 1234, 'bootstrap_resamples': 10000,
            'bootstrap_cluster': 'image', 'weighting': 'pooled request sums/counts', 'examples': len(self.examples),
            'imports': ['Python standard library', 'numpy'], 'python': platform.python_version(), 'numpy': np.__version__,
            'GPU_calls': 0, 'model_or_tokenizer_loaded': False, 'history_or_inference_changed': False,
            'official_GQA_or_MetaCompress_reproduction_claimed': False,
            'code_sha256': sha(__file__), 'elapsed_seconds': time.monotonic()-self.started})
        print(json.dumps({'status': 'PASS', 'models': len(self.validation), 'requests': sum(v['main_requests'] for v in self.validation)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    analysis = Analysis(args.root, args.output)
    try:
        with (args.output / 'raw_em_requests.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['model', 'method_id', 'image_id', 'dialog_id', 'turn_id',
                'logical_request_id', 'physical_execution_id', 'prediction', 'gold', 'raw_em', 'normalized_em', 'stored_score'])
            writer.writeheader()
            for spec in SPECS:
                analysis.model(*spec, request_writer=writer)
        analysis.finish()
    except Exception as error:
        save_json(args.output / 'ANALYSIS_FAILURE.json', {'status': 'FAIL', 'reason': repr(error),
                  'no_original_files_modified': True, 'completed_models': analysis.validation})
        raise


if __name__ == '__main__':
    main()

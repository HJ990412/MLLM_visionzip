"""Archive metadata indices only after the complete, adopted v6 main."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/dblab/hj/mllm_v2')
RUN = ROOT / 'runs/glm46v_prefixstore3_v6_20261007T104505Z'
OUT = ROOT / 'results/glm46v_prefixstore3_v6_20261007T104505Z'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    path = OUT / name
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write('\n')


def main():
    complete = json.loads((RUN / 'main/COMPLETED.json').read_text())
    assert complete['status'] == 'PASS', complete
    commits = sorted((RUN / 'main/images').glob('*/COMMITTED.json'))
    assert len(commits) == 398
    counts = 0
    selection = {}
    raw, stores = [], []
    for path in commits:
        commit = json.loads(path.read_text())
        assert commit['status'] == 'PASS'
        counts += commit['requests']
        owner = path.parent / commit['attempt_id']
        meta = json.loads((owner / 'ours_prefixstore_kv25_store_metadata.json').read_text())
        n, k = meta['original_content_count'], meta['stored_content_count']
        perm = meta['full_importance_permutation']
        assert k == (n + 3) // 4 and sorted(perm) == list(range(n))
        selection[commit['image_id']] = {
            'N': n, 'k': k, 'full_permutation': perm,
            'selected_original': perm[:k],
            'content_logical_indices': meta['content_logical_indices'],
            'structural_indices': meta['structural_indices'],
            'geometry': meta['geometry'],
            'logical_position_ids': meta['logical_position_ids'],
            'permutation_sha256': meta['permutation_sha256'],
            'source_identity_sha256': meta['identity_sha256'],
            'metadata_path': str(owner / 'ours_prefixstore_kv25_store_metadata.json'),
        }
        raw.append({
            'image_id': commit['image_id'], 'attempt_id': commit['attempt_id'],
            'requests': commit['requests'], 'raw_path': commit['raw_path'],
            'raw_sha256': commit['raw_sha256'], 'setup_path': commit['setup_path'],
            'setup_sha256': commit['setup_sha256'], 'commit_path': str(path),
            'commit_sha256': sha(path),
        })
        evidence = owner / 'payload_evidence.json'
        assert sha(evidence) == commit['payload_evidence_sha256']
        stores.append({
            'image_id': commit['image_id'], 'owner': str(owner),
            'payload_evidence_path': str(evidence),
            'payload_evidence_sha256': commit['payload_evidence_sha256'],
            'retained': True,
        })
    assert counts == 36549
    save('selection_manifest.json', {
        'source': 'actual source T1 image-only scoring; frozen before followups',
        'full_score_rank_regeneration': 'Requires the preserved capture/vision scoring path; no full score vector was archived.',
        'images': selection,
    })
    save('local_artifact_index.json', {
        'large_raw_store_not_pushed': True, 'payloads_retained': True,
        'raw': raw, 'stores': stores,
        'regeneration': 'See REPRODUCE.md; checkpoint files remain at the v3 checkpoint directory and v6 uses read-only symlinks.',
    })
    freeze = json.loads((RUN / 'freeze.json').read_text())
    checks = {p: {'expected': expected, 'actual': sha(ROOT / p)}
              for p, expected in freeze['source_sha256'].items()}
    assert all(v['actual'] == v['expected'] for v in checks.values())
    save('source_protection_after.json', {
        'status': 'PASS', 'checked_at': datetime.now(timezone.utc).isoformat(),
        'source_files': len(checks), 'checks': checks,
    })
    previous = json.loads((ROOT / 'results/glm46v_prefixstore3_v5_20261007T075730Z/failure_and_attempt_index.json').read_text())
    entries = previous['entries']
    for entry in entries:
        entry['included_in_v6_main'] = False
    entries.append({
        'run': str(ROOT / 'runs/glm46v_prefixstore3_v5_20261007T075730Z'),
        'kind': 'completed_previous_revision', 'status': 'QUALIFIED_SETUP_SCOPE',
        'requests': 36549, 'reason': 'Complete outer setup identity/wrapper endpoint was NOT_MEASURED; raw/inference evidence retained.',
        'included_in_v6_main': False,
    })
    for p in sorted((RUN / 'validation').glob('attempt_*/receipt.json')):
        receipt = json.loads(p.read_text())
        entries.append({'run': str(RUN), 'kind': 'validation', 'path': str(p),
                        'status': receipt['status'], 'included_in_v6_main': False})
    failures = sorted((RUN / 'main/images').glob('*/attempt_*/FAILURE.json'))
    for p in failures:
        entries.append({'run': str(RUN), 'kind': 'main_attempt_failure', 'path': str(p),
                        'failure': json.loads(p.read_text()), 'included_in_v6_main': False})
    save('failure_and_attempt_index.json', {
        'v6_final_main_failed_attempts': len(failures),
        'v6_final_main_retries': sum(int(json.loads(p.read_text())['attempt_id'].split('_')[-1]) - 1 for p in commits),
        'prior_requests_not_mixed_into_v6': True, 'entries': entries,
        'additional_preserved_diagnostics': [
            'results/glm46v_prefixstore3_20261007T065207Z/STORE_REVIEW.md',
            'results/glm46v_prefixstore3_v4_20261007T072841Z/NON_THINKING_DIAGNOSIS.md',
            'results/glm46v_prefixstore3_v5_20261007T075730Z/REPORT_BEFORE_OUTER_SCOPE_REVIEW.md',
        ],
    })
    print(json.dumps({'status': 'PASS', 'images': len(commits), 'requests': counts,
                      'source_files': len(checks), 'failures': len(failures)}))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Cheap final protection check. Run only after latency benchmarks finish.

Existing artifact payloads are never opened. This audit joins an earlier full
SHA256 rehash to final metadata checks, and freshly hashes protected source.
The report explicitly does not claim a second artifact content rehash.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import time

EXPECTED_FULL_REHASH_SHA256 = 'e7d8a222031769c9291c8592a19871dc936baebb8da58c8d47feb58422b47eef'


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    nbytes = 0
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
            nbytes += len(block)
    return digest.hexdigest(), nbytes


def fingerprint(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns,
            stat.S_IFMT(st.st_mode))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/dblab/hj/mllm_v2'))
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--expected-full-rehash-sha256', default=EXPECTED_FULL_REHASH_SHA256)
    args = parser.parse_args()
    root, run = args.root.resolve(), args.run_dir.resolve()
    output = args.output or run / 'protection_final.json'
    if output.exists():
        raise FileExistsError(f'Prior final audit must be preserved: {output}')
    started_at, started = utc(), time.perf_counter()
    failures = []
    sources_path = run / 'source_before.json'
    inventory_path = run / 'protected_artifacts_before.jsonl'
    full_path = run / 'protection_rehash.json'
    full = json.loads(full_path.read_text())
    full_hash, receipt_read_bytes = sha256(full_path)
    if full_hash != args.expected_full_rehash_sha256:
        failures.append({'category': 'chain', 'check': 'full_rehash_receipt_sha256',
                         'expected': args.expected_full_rehash_sha256, 'actual': full_hash})
    if full.get('status') != 'PASS' or full.get('failures') != []:
        failures.append({'category': 'chain', 'check': 'prior_full_rehash_must_pass'})
    source_manifest_hash, source_manifest_bytes = sha256(sources_path)
    artifact_manifest_hash, artifact_manifest_bytes = sha256(inventory_path)
    for name, expected, actual in (
            ('source_manifest_sha256', full.get('source_manifest_sha256'), source_manifest_hash),
            ('artifact_manifest_sha256', full.get('manifest_sha256'), artifact_manifest_hash)):
        if expected != actual:
            failures.append({'category': 'chain', 'check': name,
                             'expected': expected, 'actual': actual})

    sources = json.loads(sources_path.read_text())
    source_checked = source_hash_bytes = 0
    for row in sources:
        path = root / row['path']
        try:
            before = path.stat()
            actual_hash, actual_bytes = sha256(path)
            after = path.stat()
            diff = {}
            if actual_hash != row['sha256']:
                diff['sha256'] = {'before': row['sha256'], 'after': actual_hash}
            if actual_bytes != row['bytes'] or after.st_size != row['bytes']:
                diff['bytes'] = {'before': row['bytes'], 'read': actual_bytes, 'after': after.st_size}
            if fingerprint(before) != fingerprint(after):
                diff['changed_during_hash'] = True
            if diff:
                failures.append({'category': 'source', 'path': row['path'], 'differences': diff})
            source_checked += 1
            source_hash_bytes += actual_bytes
        except OSError as exc:
            failures.append({'category': 'source', 'path': row['path'], 'error': repr(exc)})

    artifact_expected = artifact_checked = artifact_logical_bytes = 0
    with inventory_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            artifact_expected += 1
            artifact_logical_bytes += row['size']
            path = root / row['path']
            try:
                current = path.lstat()
                actual = {'size': current.st_size, 'mtime_ns': current.st_mtime_ns,
                          'inode': current.st_ino, 'symlink': stat.S_ISLNK(current.st_mode)}
                differences = {field: {'before': row[field], 'after': value}
                               for field, value in actual.items() if row[field] != value}
                if differences:
                    failures.append({'category': 'artifact', 'path': row['path'],
                                     'differences': differences})
                artifact_checked += 1
            except OSError as exc:
                failures.append({'category': 'artifact', 'path': row['path'], 'error': repr(exc)})
    if len(sources) != full.get('source_files') or artifact_expected != full.get('artifact_files'):
        failures.append({'category': 'chain', 'check': 'inventory_counts_match_full_rehash',
                         'source_current': len(sources), 'artifact_current': artifact_expected})

    report = {
        'schema_version': 'sparsevlm-final-protection-v1',
        'mode': 'final_stat_and_source_sha256_after_full_rehash',
        'status': 'PASS' if not failures else 'FAIL',
        'started_at': started_at, 'finished_at': utc(),
        'elapsed_seconds': time.perf_counter() - started,
        'root': str(root),
        'prior_full_rehash': {
            'path': str(full_path), 'sha256': full_hash,
            'expected_sha256': args.expected_full_rehash_sha256,
            'status': full.get('status'), 'finished_at': full.get('finished_at'),
            'artifact_files': full.get('artifact_files'), 'source_files': full.get('source_files'),
            'actual_hash_bytes_read': full.get('actual_hash_bytes_read'),
        },
        'inventories': {
            'source_path': str(sources_path), 'source_sha256': source_manifest_hash,
            'artifact_path': str(inventory_path), 'artifact_sha256': artifact_manifest_hash,
            'match_prior_full_rehash': source_manifest_hash == full.get('source_manifest_sha256')
                                     and artifact_manifest_hash == full.get('manifest_sha256'),
        },
        'source': {
            'expected_files': len(sources), 'checked_files': source_checked,
            'sha256_recomputed': True, 'content_bytes_read': source_hash_bytes,
            'checks': ['SHA256', 'byte length', 'stable fingerprint during hash'],
            'initial_inode_mtime_available': False,
        },
        'artifacts': {
            'expected_files': artifact_expected, 'checked_files': artifact_checked,
            'logical_payload_bytes': artifact_logical_bytes,
            'content_sha256_recomputed': False, 'payload_bytes_read': 0,
            'checks': ['size', 'inode', 'mtime_ns', 'symlink state'],
            'evidence_limit': 'Final metadata agreement plus earlier full hash; no fresh per-artifact content SHA256 verification.',
        },
        'audit_receipt_and_inventory_hash_bytes': receipt_read_bytes + source_manifest_bytes + artifact_manifest_bytes,
        'failures': failures,
        'existing_artifacts_modified_by_audit': False, 'gpu_used': False,
        'timing_scope': 'after_all_latency_benchmarks',
    }
    # Exclusive creation preserves every prior report. A partial write remains
    # visible on failure rather than silently replacing an earlier artifact.
    with output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({'status': report['status'], 'output': str(output),
                      'source_checked': source_checked, 'artifact_checked': artifact_checked,
                      'source_hash_bytes': source_hash_bytes, 'artifact_payload_bytes_read': 0,
                      'failure_count': len(failures)}))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())

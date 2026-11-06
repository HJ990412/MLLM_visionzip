"""Persistence-ablation receipts and validation; no selector or model changes."""
from __future__ import annotations
import contextlib
import hashlib
import json
import os
from pathlib import Path
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def validate_llava_prefix_meta(m):
    n = int(m['v_token_num']) - len(m['newline_idx'])
    k = (n + 3) // 4
    if (n < 1 or m.get('format') != 'llava_fp16_importance_prefix25_v2'
            or m.get('original_content_count') != n
            or m.get('n_spatial') != n or m.get('stored_content_count') != k
            or m.get('retention_ratio') != .25
            or m.get('rounding_policy') != 'ceil_original_content'
            or m.get('payload_rows') != k or m.get('padding_rows') != 0
            or m.get('payload_chunks') != (k + 63) // 64
            or m.get('valid_rows_last_chunk') != (k - 1) % 64 + 1
            or m.get('full_importance_permutation') != m['order']
            or sorted(m['order']) != list(range(m['v_token_num']))
            or m.get('stored_row_to_original') != m['order'][:k]
            or m['order'][n:] != m['newline_idx']):
        raise ValueError('invalid prefix25 original geometry or row mapping')


def validate_llava_prefix_layerwise_meta(m):
    """Fail-closed geometry/mapping check for ``prefix25_layerwise`` stores.

    Recomputes every ``k_l`` from the frozen ratio vector and the original
    content count, so a store whose metadata, vector or row counts disagree
    with each other is rejected before activation.
    """
    from mmimpress.layer_budget import (budget_sha256, validate_keep_counts,
                                        total_row_budget)
    n = int(m['v_token_num']) - len(m['newline_idx'])
    L = int(m['num_layers'])
    counts = m.get('layer_keep_counts')
    ratios = m.get('layer_keep_ratios')
    try:
        validate_keep_counts(counts, ratios, n, L, 0.25,
                             float(m.get('layer_budget_r_min')),
                             float(m.get('layer_budget_r_max')))
    except (TypeError, ValueError) as exc:
        raise ValueError(f'invalid layerwise budget: {exc}') from exc
    k_max = max(counts)
    if (n < 1 or m.get('format') != 'llava_fp16_importance_prefix25_layerwise_v1'
            or m.get('storage_policy') != 'prefix25_layerwise'
            or m.get('budget_policy') != 'layerwise_v1'
            or m.get('original_content_count') != n
            or m.get('n_spatial') != n
            or m.get('uniform_equivalent_content_count') != (n + 3) // 4
            or m.get('stored_content_count_per_layer') != counts
            or m.get('payload_rows_per_layer') != counts
            or m.get('stored_content_count_total') != sum(counts)
            or sum(counts) != total_row_budget(n, L)
            or 'payload_rows' in m
            or m.get('retention_ratio') != .25
            or m.get('rounding_policy') != 'hamilton_largest_remainder_total_ceil_original_content'
            or m.get('padding_rows') != 0
            or m.get('layer_budget_sha256') != budget_sha256(ratios)
            or not isinstance(m.get('layer_budget_id'), str) or not m['layer_budget_id']
            or m.get('payload_chunks_per_layer') != [(k + 63) // 64 for k in counts]
            or m.get('valid_rows_last_chunk_per_layer') != [((k - 1) % 64 + 1) if k else 0 for k in counts]
            or m.get('full_importance_permutation') != m['order']
            or sorted(m['order']) != list(range(m['v_token_num']))
            or m.get('stored_row_to_original') != m['order'][:k_max]
            or m['order'][n:] != m['newline_idx']):
        raise ValueError('invalid prefix25_layerwise geometry or row mapping')


def inventory(path):
    root = Path(path)
    return {str(p.relative_to(root)): {'size': p.stat().st_size,
            'allocated_bytes': p.stat().st_blocks * 512, 'sha256': sha(p)}
            for p in sorted(root.rglob('*')) if p.is_file()}


def seal_integrity(path):
    """Equal A/B full checksums and durability, inside persistence critical path."""
    root = Path(path)
    records = inventory(root)
    payload = (json.dumps(records, sort_keys=True, indent=2) + '\n').encode()
    with (root / 'integrity.json').open('xb') as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    # Includes Qwen's parent directory entry (legacy writer syncs store only).
    for p in (root, root.parent):
        fd = os.open(p, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    records['integrity.json'] = {'size': len(payload), 'sha256': sha(root/'integrity.json'),
                                'allocated_bytes': (root/'integrity.json').stat().st_blocks * 512}
    return records


def verify_integrity(path):
    root = Path(path)
    manifest = root / 'integrity.json'
    records = json.loads(manifest.read_text())
    total = 0
    started = time.perf_counter()
    for rel, row in records.items():
        if Path(rel).is_absolute() or '..' in Path(rel).parts:
            raise ValueError('invalid integrity path')
        p = root / rel
        if p.is_symlink() or p.stat().st_size != row['size'] or sha(p) != row['sha256']:
            raise ValueError(f'payload integrity mismatch: {rel}')
        total += row['size']
    return {'integrity_ms': (time.perf_counter()-started)*1000,
            'integrity_read_bytes': total, 'metadata_bytes': manifest.stat().st_size}


def process_io():
    return {k: int(v) for k, v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}


class ReadTrace:
    """Independent real syscall return trace, with optional store access guard."""
    def __init__(self, allowed=None):
        self.allowed = None if allowed is None else Path(allowed).resolve()
        self.calls = []

    def __enter__(self):
        self.original = os.pread
        def traced(fd, count, offset):
            path = Path(os.readlink(f'/proc/self/fd/{fd}')).resolve()
            if self.allowed is not None and path.suffix == '.bin' and not path.is_relative_to(self.allowed):
                raise AssertionError(f'foreign KV payload access: {path}')
            blob = self.original(fd, count, offset)
            self.calls.append({'path': str(path), 'offset': offset,
                               'requested': count, 'returned': len(blob)})
            return blob
        os.pread = traced
        return self

    def __exit__(self, *exc):
        os.pread = self.original

    def summary(self):
        return {'bytes': sum(x['returned'] for x in self.calls),
                'preads': len(self.calls), 'calls': self.calls}

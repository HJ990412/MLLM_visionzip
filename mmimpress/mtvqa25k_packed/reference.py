"""Read-only access to committed per-image raw journals (v3 reference run and the new run).

The v3 MT-VQA run keeps every request only inside per-image gzip journals
(``<phase>/images/<image>/<attempt>/raw.jsonl.gz``).  The request-paired
cross-run comparison needs them all, so ``build_index`` opens every committed
image, checks the uncompressed SHA256 against the atomic commit and writes a
compact per-image record.  Nothing under the reference run is written.
"""
from pathlib import Path
import gzip
import hashlib
import json

from mmimpress.mtvqa25k.common import read

V3_RUN = 'runs/mt_vqa_v2_reconstructed_25k_v3_20261021T100500Z'
V3_RESULTS = 'results/mt_vqa_v2_reconstructed_25k_v3_20261021T100500Z'
V3_METHODS = ('recompute', 'fullload', 'mpic32_ssd', 'sparsevlm_ssd_kv25_allhead', 'ours_prefixstore_kv25')
V3_B = 'ours_prefixstore_kv25'
COMPARE_METHODS = ('recompute', 'fullload', 'sparsevlm_ssd_kv25_allhead', 'ours_prefixstore_kv25')   # mpic32_ssd is not compared


def committed_raw(folder):
    """(rows, commit) of one image folder; verifies the uncompressed journal SHA256 bound by the commit."""
    folder = Path(folder)
    commit = read(folder / 'COMMITTED.json')
    attempt = folder / commit['attempt_id']
    archive = commit.get('journal_archives', {}).get('raw.jsonl')
    path = attempt / (archive['path'] if archive else 'raw.jsonl')
    data = gzip.open(path, 'rb').read() if path.suffix == '.gz' else path.read_bytes()
    if hashlib.sha256(data).hexdigest() != commit['raw_sha256']:
        raise ValueError('committed raw SHA256 mismatch: ' + str(path))
    rows = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    if len(rows) != commit['requests']:
        raise ValueError('committed request count mismatch: ' + str(path))
    return rows, commit


def compact(r, full=False):
    """Fields needed by the gates and the paired analysis (identical for reference and new rows)."""
    result = r.get('result', {})
    out = {'method_id': r['method_id'], 'turn_id': r['turn_id'], 'dialog_id': r['dialog_id'], 'question_id': str(r['question_id']),
           'prediction': r['prediction'], 'generated_token_ids': list(r['generated_token_ids']), 'score': r['score'],
           'N_content': r['N_content'], 'k': r['k'], 'ssd_read_bytes': r['ssd_read_bytes'], 'ssd_preads': r['ssd_preads'],
           'ttft_ms': r['ttft_ms'], 'request_e2e_ms': r['request_e2e_ms'], 'prompt_sha256': r['prompt_sha256'],
           'suffix_ids_sha256': r['suffix_ids_sha256'], 'history_sha256': r['history_sha256'],
           'answer_type': r['annotation']['answer_type'], 'n_newline': len(r['geometry']['newline_idx']),
           'chunk_io_ms': result.get('chunk_io_ms'), 'scatter_ms': result.get('scatter_ms'), 'prefill_ms': result.get('prefill_ms'),
           'prepare_ms': result.get('prepare_ms'), 'decode_ms': result.get('decode_ms'), 'packed_read_ms': result.get('packed_read_ms'),
           'pread_calls': r.get('pread_calls'), 'preadv_calls': r.get('preadv_calls'),
           'cap_reached': bool(result.get('cap_reached', False)), 'empty_answer': bool(r.get('empty_answer', False)),
           'vision_forward_count': result.get('vision_forward_count')}
    if full:
        out.update(prompt=r['prompt'], suffix_input_ids=r['suffix_input_ids'])
    return out


def extract(args):
    """Worker: one committed image -> compact record (or an error record; never raises)."""
    folder, full = args
    folder = Path(folder)
    try:
        rows, commit = committed_raw(folder)
        return {'image_id': folder.name, 'status': 'OK', 'raw_sha256': commit['raw_sha256'], 'requests': len(rows),
                'rows': [compact(r, full) for r in rows]}
    except Exception as exc:      # recorded; the caller decides BLOCKED_REFERENCE
        return {'image_id': folder.name, 'status': 'ERROR', 'error': repr(exc)}


def build_index(phase_images_dir, image_ids, out_path, full=False, workers=16):
    """Compact gzip index of the committed images in ``image_ids`` order.  Returns a summary."""
    from multiprocessing import Pool
    phase_images_dir = Path(phase_images_dir)
    tasks = [(str(phase_images_dir / str(i)), full) for i in image_ids]
    errors = []; count = requests = 0
    h = hashlib.sha256()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Pool(workers) as pool, gzip.open(out_path, 'wt', compresslevel=6) as f:
        for rec in pool.imap(extract, tasks, chunksize=16):
            if rec['status'] != 'OK':
                errors.append(rec); continue
            line = json.dumps(rec, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
            f.write(line + '\n'); h.update(line.encode() + b'\n')
            count += 1; requests += rec['requests']
    return {'images_requested': len(tasks), 'images_readable': count, 'requests': requests, 'errors': errors[:50],
            'error_count': len(errors), 'uncompressed_sha256': h.hexdigest(), 'path': str(out_path)}


def load_index(path):
    """image_id -> {(method_id, turn_id): compact row}."""
    out = {}
    with gzip.open(path, 'rt') as f:
        for line in f:
            rec = json.loads(line)
            out[rec['image_id']] = {(r['method_id'], r['turn_id']): r for r in rec['rows']}
    return out


def compare_rows(new, old, fields=('generated_token_ids', 'prediction', 'score', 'prompt_sha256', 'suffix_ids_sha256')):
    """Field-wise equality of one new row and one reference row; returns the list of differing fields."""
    return [k for k in fields if new.get(k) != old.get(k)]

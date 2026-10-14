"""Derived terminal audit; imports no GLM production or analysis helpers."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from transformers import AutoTokenizer

ROOT = Path('/home/dblab/hj/mllm_v2')
RUN = ROOT / 'runs/glm46v_prefixstore3_v5_20261007T075730Z'
RESULT = ROOT / 'results/glm46v_prefixstore3_v5_20261007T075730Z'
OUT = RESULT / 'independent_review'
METHODS = ('recompute', 'fullload', 'ours_prefixstore_kv25')
OURS = METHODS[-1]


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def norm(value):
    return ' '.join(w for w in re.sub(r'[^\w\s]', ' ', value.lower()).split()
                    if w not in ('a', 'an', 'the'))


def writecsv(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def stat(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(mean=float(a.mean()), p50=float(np.quantile(a, .5)),
                p95=float(np.quantile(a, .95)))


def prompt_messages(question, history):
    instruction = 'Answer using a single word or short phrase.'
    first = history[0][0] if history else question
    out = [{'role': 'system', 'content': 'You are a helpful assistant. Give concise factual answers.'},
           {'role': 'user', 'content': [{'type': 'image'},
              {'type': 'text', 'text': first + '\n' + instruction}]}]
    if history:
        out.append({'role': 'assistant', 'content': history[0][1]})
        for q, body in history[1:]:
            out.extend([{'role': 'user', 'content': q + '\n' + instruction},
                        {'role': 'assistant', 'content': body}])
        out.append({'role': 'user', 'content': question + '\n' + instruction})
    return out


def main():
    manifest = read(RUN / 'manifests/main_manifest.json')
    ds = manifest['dialogues']
    expected = {d['dialog_id']: (i, d) for i, d in enumerate(ds)}
    assert len(ds) == 4061 and len({d['image_id'] for d in ds}) == 398
    ordered = ''.join(f"{d['dialog_id']}\t{t['turn_id']}\t{t['question_id']}\n"
                      for d in ds for t in d['turns']).encode()
    assert hashlib.sha256(ordered).hexdigest() == '0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62'
    tokenizer = AutoTokenizer.from_pretrained(RUN / 'checkpoint', local_files_only=True)
    backend = tokenizer.backend_tokenizer
    eos = set(read(RUN / 'checkpoint/generation_config.json')['eos_token_id'])
    assert eos == {151329, 151336, 151338, 151348}
    all_compact, setup = {}, {}
    event_ids, physical_ids, logical_ids = set(), set(), set()
    inventory, outputs = [], Counter()
    raw_totals = Counter()
    source = {}
    for d in ds:
        source.setdefault(d['image_id'], d['dialog_id'])
    commits = sorted((RUN / 'main/images').glob('*/COMMITTED.json'))
    assert len(commits) == 398
    for cp in commits:
        commit = read(cp)
        iid = cp.parent.name
        folder = cp.parent / commit['attempt_id']
        raw = folder / 'raw.jsonl'
        events_path = folder / 'setup_events.jsonl'
        assert sha(raw) == commit['raw_sha256'] and sha(events_path) == commit['setup_sha256']
        assert Path(commit['raw_path']).resolve() == raw.resolve()
        assert Path(commit['setup_path']).resolve() == events_path.resolve()
        rr = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
        assert len(rr) == commit['requests']
        inventory.append(dict(image_id=iid, raw_path=str(raw), raw_sha256=commit['raw_sha256'],
                              setup_path=str(events_path), setup_sha256=commit['setup_sha256'],
                              requests=len(rr)))
        by = {(r['dialog_id'], r['method_id'], r['turn_id']): r for r in rr}
        assert len(by) == len(rr)
        ranked_choices = []
        for row in rr:
            did, method, turn = row['dialog_id'], row['method_id'], row['turn_id']
            ordinal, dialogue = expected[did]
            original = dialogue['turns'][turn - 1]
            assert row['image_id'] == iid == dialogue['image_id']
            assert row['question'] == original['question'] and str(row['question_id']) == str(original['question_id'])
            assert row['gold'] == original['answers'] and row['canonical_ordinal'] == ordinal
            rotation = list(METHODS[ordinal % 3:]) + list(METHODS[:ordinal % 3])
            assert row['method_order'] == rotation and row['phase'] == 'main' and row['status'] == 'PASS'
            assert row['physical_execution_id'] not in physical_ids and row['logical_request_id'] not in logical_ids
            physical_ids.add(row['physical_execution_id']); logical_ids.add(row['logical_request_id'])
            ids = row['generated_token_ids']
            body_ids = ids[:-1] if ids[-1] in eos else ids[:]
            literal = backend.decode(ids, skip_special_tokens=False)
            body = backend.decode(body_ids, skip_special_tokens=False)
            assert row['history_token_ids'] == body_ids and row['history_text'] == body
            assert row['generated_text_with_special_tokens'] == literal
            assert row['history_native_policy'] == 'actual-generated-body/terminal-EOS-only-v1'
            begins = [i for i, t in enumerate(ids) if t == 151361]
            ends = [i for i, t in enumerate(ids) if t == 151362]
            answer_ids = ids[begins[0]+1:ends[0]] if len(begins) == len(ends) == 1 and begins[0] < ends[0] else ids
            prediction = backend.decode(answer_ids, skip_special_tokens=True).strip()
            assert row['prediction'] == prediction
            history = []
            for pt in range(1, turn):
                previous = by[(did, method, pt)]
                prev_ids = previous['generated_token_ids']
                prev_body = backend.decode(prev_ids[:-1] if prev_ids[-1] in eos else prev_ids,
                                           skip_special_tokens=False)
                history.append([dialogue['turns'][pt - 1]['question'], prev_body])
                lineage = row['history_entries'][pt - 1]
                assert lineage['source_method_id'] == method and lineage['source_dialog_id'] == did
                assert lineage['source_physical_execution_id'] == previous['physical_execution_id']
                assert lineage['source_logical_request_id'] == previous['logical_request_id']
                assert lineage['history_text'] == prev_body and lineage['history_token_ids'] == previous['history_token_ids']
            assert row['history'] == history and len(row['history_entries']) == len(history)
            text = tokenizer.apply_chat_template(prompt_messages(original['question'], history),
                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
            assert row['prompt'] == text
            collapsed = backend.encode(text, add_special_tokens=False).ids
            full = []
            for token in collapsed:
                full.extend([token] * row['N_content'] if token == 151363 else [token])
            assert row['input_ids'] == [full]
            trace = row['request_trace']
            geometry = trace.get('geometry') or row['source_store']['identity']['geometry']
            a = full.index(151363); n = row['N_content']; b = a + n
            _, hh, ww = geometry['image_grid_thw'][0]
            hh //= 2; ww //= 2
            assert n == hh * ww and full[a:b] == [151363] * n
            tail = [a + max(hh, ww) + j for j in range(len(full) - b)]
            pos = [[list(range(a)) + [a] * n + tail],
                   [list(range(a)) + [a + j // ww for j in range(n)] + tail],
                   [list(range(a)) + [a + j % ww for j in range(n)] + tail]]
            assert trace['position_ids'] == pos
            assert trace['rope_offset'] == [[max(max(axis[0]) for axis in pos) + 1 - len(full)]]
            assert row['enable_thinking'] is False and row['online_query_score_calls'] == row['decode_store_reads'] == 0
            pixel = turn == 1 or method == 'recompute'
            assert row['vision_calls'] == int(pixel)
            io = row['os_pread_trace']
            assert row['ssd_read_bytes'] == sum(x['returned'] for x in io)
            assert row['pread_calls'] == len(io)
            if pixel:
                # Ours k is the fixed store budget even on full-pixel T1. The
                # full expanded input and native vision call prove this path.
                assert not io and row['ssd_read_bytes'] == 0
                assert row['k'] == ((n+3)//4 if method == OURS else n)
            else:
                k = n if method == 'fullload' else (n + 3) // 4
                assert row['k'] == k and row['S_structural'] == 19 and len(io) == 81
                expected_ranges = Counter({(f'layer_{li:03d}/{kind}.bin', 0, k*512, k*512): 1
                                           for li in range(40) for kind in ('k', 'v')})
                expected_ranges[('structural_kv.bin', 0, 19*40960, 19*40960)] = 1
                assert Counter((x['source'], x['offset'], x['requested'], x['returned']) for x in io) == expected_ranges
                assert row['h2d_kv_bytes'] == (k + 19) * 40960
                provenance = row['source_store']['source']; sr = by[(source[iid], method, 1)]
                assert provenance['dialog_id'] == source[iid] and provenance['method_id'] == method
                assert provenance['physical_execution_id'] == sr['physical_execution_id']
                assert provenance['logical_request_id'] == sr['logical_request_id']
                selected = row['selected_content_original']
                assert len(selected) == len(set(selected)) == k and all(0 <= j < n for j in selected)
                logical = sorted(list(range(a)) + [a+j for j in selected] + list(range(b, geometry['prefix_len'])))
                assert trace['logical_indices'] == logical and trace['compact_prefix_length'] == k + 19
                assert trace['decode_final_cache_length'] == k + 19 + len(full) - geometry['prefix_len'] + len(ids) - 1
                if method == OURS:
                    ranked_choices.append(tuple(selected))
            outputs[(method, 'empty')] += prediction == ''
            outputs[(method, 'cap')] += len(ids) == 32 and ids[-1] not in eos
            outputs[(method, 'malformed')] += row['answer_extraction']['status'] == 'MALFORMED_NATIVE_BOX'
            outputs[(method, 'think_open')] += '<think>' in literal
            outputs[(method, 'think_close')] += '</think>' in literal
            raw_totals[(method, 'bytes')] += row['ssd_read_bytes']
            raw_totals[(method, 'preads')] += len(io)
            all_compact[(did, method, turn)] = dict(image_id=iid, dialog_id=did, method_id=method,
                turn_id=turn, score=int(norm(prediction) == norm(original['answers'][0])),
                ttft_ms=row['ttft_ms'], request_e2e_ms=row['request_e2e_ms'],
                ssd_read_bytes=row['ssd_read_bytes'], tokens=ids)
        assert not ranked_choices or all(s == ranked_choices[0] for s in ranked_choices)
        for did in {r['dialog_id'] for r in rr}:
            assert by[(did, METHODS[0], 1)]['generated_token_ids'] == by[(did, METHODS[1], 1)]['generated_token_ids'] == by[(did, METHODS[2], 1)]['generated_token_ids']
        ev = [json.loads(s) for s in events_path.read_text().splitlines() if s.strip()]
        assert len(ev) == 4
        for e in ev:
            assert e['event_id'] not in event_ids and e['image_id'] == iid and e['phase'] == 'main'
            event_ids.add(e['event_id'])
            assert e['source_dialog_id'] == source[iid]
            setup.setdefault((iid, e['method_id']), {})[e['event_kind']] = e
    assert len(all_compact) == 36549 and len(setup) == 796 and len(event_ids) == 1592
    assert not list((RUN/'main').rglob('FAILURE.json'))
    images = sorted({r['image_id'] for r in all_compact.values()})
    quality, latency = [], []
    for method in METHODS:
        for scope, turns in [('all', (1,2,3)), ('followup', (2,3)), ('T1',(1,)), ('T2',(2,)), ('T3',(3,))]:
            rr = [r for r in all_compact.values() if r['method_id'] == method and r['turn_id'] in turns]
            quality.append(dict(method_id=method, scope=scope, requests=len(rr), correct=sum(r['score'] for r in rr), accuracy=100*sum(r['score'] for r in rr)/len(rr)))
            q = dict(method_id=method, scope=scope, requests=len(rr))
            for metric in ('ttft_ms','request_e2e_ms','ssd_read_bytes'):
                q.update({metric+'_'+k:v for k,v in stat([r[metric] for r in rr]).items()})
            latency.append(q)
    sessions = {}
    attributed = 0.
    for d in ds:
        did, iid = d['dialog_id'], d['image_id']
        for method in METHODS:
            request = sum(all_compact[(did, method, t)]['request_e2e_ms'] for t in (1,2,3))
            cost = 0. if method == 'recompute' else sum(setup[(iid,method)][kind]['duration_ms'] for kind in ('build','activation'))
            is_source = did == source[iid]
            attributed += cost if is_source else 0.
            sessions[(did,method)] = dict(image_id=iid, physical_stream_session_ms=request + (cost if is_source else 0.), derived_standalone_session_ms=request+cost)
    physical = sum(e['duration_ms'] for events in setup.values() for e in events.values())
    assert abs(attributed-physical) < 1e-7
    assert abs(physical-53855.878580827266) < 1e-7
    session_summary = []
    for method in METHODS:
        row = dict(method_id=method, dialogues=4061)
        for metric in ('physical_stream_session_ms','derived_standalone_session_ms'):
            row.update({metric+'_'+k:v for k,v in stat([r[metric] for (did,m),r in sessions.items() if m==method]).items()})
        session_summary.append(row)
    # One fixed draw matrix resamples each complete paired image cluster.
    rng = np.random.default_rng(1234)
    draws = rng.integers(0,398,size=(10000,398))
    ix = {iid:i for i,iid in enumerate(images)}
    def cluster(values):
        sums, counts = np.zeros(398), np.zeros(398)
        for iid, value in values:
            sums[ix[iid]] += value; counts[ix[iid]] += 1
        assert (counts>0).all()
        boot = sums[draws].sum(1)/counts[draws].sum(1)
        lo, hi = np.quantile(boot,[.025,.975])
        return dict(point=float(sums.sum()/counts.sum()),ci_low=float(lo),ci_high=float(hi))
    paired=[]
    for baseline in METHODS[:-1]:
        for scope, turns in [('all',(1,2,3)),('followup',(2,3)),('T1',(1,)),('T2',(2,)),('T3',(3,))]:
            for metric in ('accuracy_pp','ttft_ms','request_e2e_ms','ssd_read_bytes'):
                diffs=[]
                for d in ds:
                    for t in turns:
                        a,b=all_compact[(d['dialog_id'],baseline,t)],all_compact[(d['dialog_id'],OURS,t)]
                        value=100*(b['score']-a['score']) if metric=='accuracy_pp' else b[metric]-a[metric]
                        diffs.append((d['image_id'],value))
                paired.append(dict(baseline=baseline,scope=scope,metric=metric,**cluster(diffs)))
        for metric in ('physical_stream_session_ms','derived_standalone_session_ms'):
            diffs=[(d['image_id'],sessions[(d['dialog_id'],OURS)][metric]-sessions[(d['dialog_id'],baseline)][metric]) for d in ds]
            paired.append(dict(baseline=baseline,scope='per_dialogue',metric=metric,**cluster(diffs)))
        for metric in ('file_logical_bytes','allocated_bytes','os_write_bytes','persistence_ms','activation_ms'):
            diffs=[]
            for iid in images:
                def v(method):
                    if method=='recompute':return 0.
                    ev=setup[(iid,method)]
                    return ev['activation']['duration_ms'] if metric=='activation_ms' else ev['build']['duration_ms'] if metric=='persistence_ms' else ev['build'][{'file_logical_bytes':'logical_bytes'}.get(metric,metric)]
                diffs.append((iid,v(OURS)-v(baseline)))
            paired.append(dict(baseline=baseline,scope='image_macro_setup',metric=metric,**cluster(diffs)))
    with (RESULT/'final/paired_analysis.csv').open() as h:published=list(csv.DictReader(h))
    lookup={(r['baseline'],r['scope'],r['metric']):r for r in published}
    assert len(paired)==len(published)==54
    max_residual=0.
    for r in paired:
        old=lookup[(r['baseline'],r['scope'],r['metric'])]
        for field in ('point','ci_low','ci_high'):
            delta=abs(r[field]-float(old[field]));max_residual=max(max_residual,delta)
            assert delta<1e-7,(r,old,field,delta)
    for name,recomputed in [('quality.csv',quality),('latency.csv',latency),('session_summary.csv',session_summary)]:
        with (RESULT/'final'/name).open() as h:old=list(csv.DictReader(h))
        keyfields=('method_id',) if name=='session_summary.csv' else ('method_id','scope')
        look={tuple(r[k] for k in keyfields):r for r in old}
        for r in recomputed:
            actual=look[tuple(r[k] for k in keyfields)]
            for k,v in r.items():
                if isinstance(v,(int,float)):assert abs(v-float(actual[k]))<1e-7,(name,r,k,actual[k])
    before=read(RUN/'baseline_protection_cpu_audit.json');after={}
    for n,old in before['files'].items():
        p=ROOT/n;actual=dict(sha256=sha(p),bytes=p.stat().st_size);assert actual==old,n;after[n]=actual
    protection=dict(status='PASS',full_byte_hash=True,files=after,protected_files=len(after),
        before_receipt_sha256=sha(RUN/'baseline_protection_cpu_audit.json'),scope=before['scope'])
    (OUT/'baseline_protection_after.json').write_text(json.dumps(protection,indent=2)+'\n')
    receipt=dict(status='PASS',requests=36549,images=398,dialogues=4061,
        methods={m:sum(r['method_id']==m for r in all_compact.values()) for m in METHODS},
        raw_inventory=inventory,quality_native_prediction_reparsed=True,native_history_body_lineage_exact=True,
        native_prompt_input_ids_positions_independent=True,source_T1_ID_agreement=4061,
        actual_OS_read_ranges_exact=True,preads={m:raw_totals[m,'preads'] for m in METHODS},
        OS_read_bytes={m:raw_totals[m,'bytes'] for m in METHODS},
        output_counts={m:{name:outputs[m,name] for name in ['empty','cap','malformed','think_open','think_close']} for m in METHODS},
        failed_image_attempts=0,physical_builds=796,activations=796,physical_recorded_setup_ms=physical,
        session_recorded_setup_attribution_ms=attributed,paired_checks=54,bootstrap_seed=1234,
        bootstrap_resamples=10000,cluster='image',weighting='pooled request/dialogue sums and counts',
        maximum_paired_CSV_numeric_residual=max_residual,
        outer_runner_identity_prep_and_receipt_tail='NOT_MEASURED; recorded setup is native writer/read service scope',
        prior_baseline_full_hash_check='PASS 25 files')
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    writecsv('quality.csv',quality);writecsv('latency.csv',latency);writecsv('session_summary.csv',session_summary);writecsv('paired_analysis.csv',paired)
    print(json.dumps({k:v for k,v in receipt.items() if k!='raw_inventory'},indent=2))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Fail-closed image-major GLM 3-arm pilot/main with durable attempt receipts."""
from pathlib import Path
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mmimpress.glm46v.runner import GLM46VRunner, METHODS
from mmimpress.glm46v.analysis import analyze_rows, audit_requests, file_sha256


def read(path):
    return json.loads(Path(path).read_text())


def dump(path,value):
    path=Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+'.pending')
    with pending.open('x') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(pending,path)
    fd=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def append(handle,row):
    handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    handle.flush();os.fsync(handle.fileno())


def verify_freeze(run):
    active=read(run/'active_freeze.json') if (run/'active_freeze.json').exists() else {'path':'freeze.json'}
    freeze_path=(run/active['path']).resolve()
    if not freeze_path.is_relative_to(run.resolve()):raise ValueError('outside source freeze path')
    if active.get('sha256') and file_sha256(freeze_path)!=active['sha256']:raise ValueError('active source freeze changed')
    freeze=read(freeze_path)
    for name,expected in freeze['source_sha256'].items():
        if file_sha256(ROOT/name)!=expected:
            raise ValueError(f'source freeze changed: {name}')
    for name,key in [('config.json','config_sha256'),('environment.json','environment_sha256'),
                     ('CONTRACT.md','contract_sha256'),('validation_manifest.json','validation_manifest_sha256')]:
        frozen_name=freeze.get(name.split('.')[0].lower()+'_path',name)
        if file_sha256(run/frozen_name)!=freeze[key]:raise ValueError(f'freeze changed:{name}')
    for name,expected in freeze['manifests'].items():
        if file_sha256(run/name)!=expected:raise ValueError(f'workload freeze changed:{name}')
    before=read(run/'baseline_protection_cpu_audit.json')
    # The receipt format is emitted by the independent baseline auditor.
    protected=before.get('baseline_full_sha256',before.get('files',{}))
    for name,expected in protected.items():
        if isinstance(expected,dict):expected=expected.get('sha256')
        if expected and file_sha256(ROOT/name)!=expected:raise ValueError(f'existing artifact changed:{name}')


def guard_gpu():
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                     capture_output=True,text=True,check=True)
    foreign=[int(x.strip()) for x in p.stdout.splitlines() if x.strip() and int(x.strip())!=os.getpid()]
    if foreign:raise RuntimeError(f'foreign GPU workload; timing aborted without terminating processes: {foreign}')
    if shutil.disk_usage(ROOT).free<30<<30:raise RuntimeError('30GiB disk reserve violated')


def method_order(ordinal):
    n=ordinal%3
    return list(METHODS[n:])+list(METHODS[:n])


def resolve_entries(run,phase):
    name={'main':'main_manifest.json','gqa_pilot':'gqa_pilot_manifest.json','mt_pilot':'mt_pilot_manifest.json'}[phase]
    manifest=read(run/'manifests'/name)
    entries=manifest.get('dialogues',manifest.get('images'))
    identities=read(run/'manifests/main_image_identity.json')
    groups=defaultdict(list)
    for ordinal,d in enumerate(entries):
        d=dict(d)
        iid=d['image_id']
        identity=identities.get(iid,{})
        d['image_path']=str((ROOT/d.get('image_path',identity.get('image_path',''))).resolve())
        d['image_sha256']=d.get('image_sha256',identity.get('sha256'))
        d['execution_dialog_id']=str(d.get('dialog_id') or f'gqa:{iid}')
        groups[iid].append((ordinal,d))
    return manifest,groups


def committed_rows(phase_dir):
    rows=[];events=[]
    for p in sorted((phase_dir/'images').glob('*/COMMITTED.json')):
        receipt=read(p)
        raw=Path(receipt['raw_path']);setup=Path(receipt['setup_path'])
        owner=(p.parent/receipt['attempt_id']).resolve()
        if raw.resolve()!=owner/'raw.jsonl' or setup.resolve()!=owner/'setup_events.jsonl':
            raise ValueError('committed raw/setup ownership differs')
        if file_sha256(raw)!=receipt['raw_sha256'] or file_sha256(setup)!=receipt['setup_sha256']:
            raise ValueError('committed raw/setup changed')
        adopted=[json.loads(x) for x in raw.read_text().splitlines() if x.strip()]
        if len(adopted)!=receipt['requests'] or read(owner/'IMAGE_COMPLETE.json')!=receipt:
            raise ValueError('committed request count/completion receipt differs')
        payload_receipt=owner/'payload_evidence.json'
        if file_sha256(payload_receipt)!=receipt['payload_evidence_sha256']:
            raise ValueError('committed payload evidence changed')
        for relative,evidence in read(payload_receipt).items():
            payload=(owner/relative).resolve()
            if not payload.is_relative_to(owner) or file_sha256(payload)!=evidence['sha256']:
                raise ValueError('committed store payload differs')
        rows.extend(adopted)
        events.extend(json.loads(x) for x in setup.read_text().splitlines() if x.strip())
    return rows,events


def io_trace(result):
    return [dict(returned=x['actual_bytes'],requested=x['requested_bytes'],offset=x['offset'],
                 source=x['source'],kind=x['kind'],ms=x['ms'])
            for x in result.get('read_io',{}).get('call_details',[])]


def evaluate_image(runner,run,phase,iid,group,attempt_dir):
    import torch
    raw_path=attempt_dir/'raw.jsonl';setup_path=attempt_dir/'setup_events.jsonl'
    captures={};stores={};sources={};events=[];rows=[]
    source_did=group[0][1]['execution_dialog_id']
    with raw_path.open('x') as raw,setup_path.open('x') as setup, (attempt_dir/'request_attempts.jsonl').open('x') as attempts:
        for ordinal,d in group:
            did=d['execution_dialog_id'];histories={m:[] for m in METHODS};lineages={m:[] for m in METHODS}
            order=method_order(ordinal)
            for turn in d['turns']:
                tid=int(turn['turn_id'])
                for method in order:
                    physical=str(uuid.uuid4());logical=f'{phase}:{did}:{method}:T{tid}'
                    append(attempts,dict(status='STARTED',physical_execution_id=physical,
                        logical_request_id=logical,phase=phase,image_id=iid,dialog_id=did,
                        turn_id=tid,method_id=method,attempt_id=attempt_dir.name))
                    guard_gpu()
                    conditioning='NOT_APPLICABLE_PIXELS'
                    conditioning_ms=0
                    if tid!=1 and method!='recompute':
                        store=runner.stores[str(stores[method].resolve())]
                        conditioning_start=time.perf_counter()
                        conditioning=store.drop_payload_cache()
                        conditioning_ms=(time.perf_counter()-conditioning_start)*1000
                    start=time.perf_counter()
                    source=did==source_did and tid==1 and method!='recompute'
                    pixels=tid==1 or method=='recompute'
                    history=[] if phase=='gqa_pilot' else histories[method]
                    if pixels:
                        result=runner.run_pixels(d['image_path'],turn['question'],history,
                           capture=('with_score' if method==METHODS[-1] else 'kv_only') if source else False,
                           image_sha256=d['image_sha256'],request_start=start)
                    else:
                        store=runner.stores[str(stores[method].resolve())]
                        result=runner.run_cache(stores[method],turn['question'],history,request_start=start)
                    result['request_e2e_ms']=(time.perf_counter()-start)*1000
                    cap=result.pop('capture',None)
                    gold=turn.get('answers',[turn['gold']] if 'gold' in turn else None)
                    if gold is None:raise ValueError('missing immutable gold')
                    geometry=result['request_trace'].get('geometry')
                    if geometry is None:geometry=runner.stores[str(stores[method].resolve())].meta['geometry']
                    n=geometry['N_content']
                    k=(n+3)//4 if method==METHODS[-1] else n
                    row=dict(result,phase=phase,status='PASS',attempt_id=attempt_dir.name,
                        physical_execution_id=physical,logical_request_id=logical,
                        image_id=iid,image_path=d['image_path'],image_sha256=d['image_sha256'],
                        dialog_id=did,turn_id=tid,question_id=turn['question_id'],question=turn['question'],
                        gold=gold,method_id=method,method_order=order,canonical_ordinal=ordinal,
                        history=[list(x) for x in history],history_entries=list(lineages[method]) if phase!='gqa_pilot' else [],
                        N_content=n,k=k,S_structural=geometry['prefix_len']-n,
                        kv_geometry={'num_layers':40,'num_kv_heads':2,'head_dim':128,'dtype_bytes':2},
                        prompt=result['request_trace']['prompt_text'],os_pread_trace=io_trace(result),
                        source_store=sources.get(method),source_T1_capture=source,
                        page_cache_conditioning=conditioning,
                        page_cache_conditioning_ms_outside_request=conditioning_ms)
                    if row['reasoning_leak']:
                        raise RuntimeError(f'non-thinking output invalid: {logical}, {row["generated_text_with_special_tokens"]}')
                    if source:
                        if cap is None:raise ValueError('source capture missing')
                        store_dir=attempt_dir/'stores'/method
                        persistence=runner.persist(cap,store_dir,storage_policy='prefix25' if method==METHODS[-1] else 'full')
                        meta=persistence['metadata'];stores[method]=store_dir
                        provenance={'path':str(store_dir),'source':dict(dialog_id=did,method_id=method,
                                    physical_execution_id=physical,logical_request_id=logical),'identity':meta['identity']}
                        sources[method]=provenance;row['source_store']=provenance
                        event=dict(event_id=str(uuid.uuid4()),event_kind='build',phase=phase,image_id=iid,
                              method_id=method,source_dialog_id=did,duration_ms=persistence['duration_ms'],
                              endpoint='outer_writer_return',logical_bytes=meta['store_logical_bytes'],
                              allocated_bytes=persistence['timing_ms']['store_allocated_bytes'],
                              os_write_bytes=persistence['timing_ms']['os_write_bytes'],
                              os_write_calls=persistence['timing_ms']['os_write_calls'],
                              content_payload_bytes=meta['bytes_content_kv'],
                              structure_payload_bytes=meta['bytes_structural_kv'],
                              structural_payload_bytes=meta['bytes_structural_kv'],padding_bytes=0,
                              metadata_bytes=meta['bytes_metadata_file']+meta['bytes_seal_file'],
                              N_content=n,k=k,stored_k=k,
                              timing_ms=persistence['timing_ms'],store_path=str(store_dir))
                        append(setup,event);events.append(event)
                        del cap,persistence
                        activated,activation=runner.activate(store_dir)
                        event=dict(event_id=str(uuid.uuid4()),event_kind='activation',phase=phase,image_id=iid,
                              method_id=method,source_dialog_id=did,store_path=str(store_dir),
                              endpoint='outer_activation_return',**activation)
                        append(setup,event);events.append(event)
                        # Preserve the complete rank and hashes without GPU/CPU KV residency.
                        dump(attempt_dir/(method+'_store_metadata.json'),meta)
                    append(raw,row);rows.append(row)
                    append(attempts,dict(status='RETURNED',physical_execution_id=physical,
                         logical_request_id=logical,request_e2e_ms=row['request_e2e_ms']))
                    if phase!='gqa_pilot':
                        histories[method].append([turn['question'],row['prediction']])
                        lineages[method].append(dict(source_method_id=method,source_dialog_id=did,
                             source_physical_execution_id=physical,source_logical_request_id=logical,
                             generated_token_ids=list(row['generated_token_ids']),prediction=row['prediction']))
                    # Invariant contexts contain metadata/FDs only. Results contain no tensors.
                    del result
        runner.close();gc.collect();torch.cuda.empty_cache()
    # Full independent on-disk rehash before publication, distinct from activation service scans.
    payloads={str(p.relative_to(attempt_dir)):dict(sha256=file_sha256(p),bytes=p.stat().st_size)
              for p in sorted((attempt_dir/'stores').rglob('*')) if p.is_file()}
    dump(attempt_dir/'payload_evidence.json',payloads)
    image_manifest={'images':[dict(d,canonical_ordinal=ordinal) for ordinal,d in group]}
    image_audit=audit_requests(rows,image_manifest,phase=phase)
    if image_audit['status']!='PASS':
        raise ValueError(f'independent image audit failed:{image_audit}')
    for row in rows:
        if row['method_order']!=method_order(row['canonical_ordinal']):raise ValueError('rotation mismatch')
    receipt=dict(status='PASS',image_id=iid,phase=phase,attempt_id=attempt_dir.name,
         raw_path=str(raw_path),raw_sha256=file_sha256(raw_path),setup_path=str(setup_path),
         setup_sha256=file_sha256(setup_path),requests=len(rows),audit=image_audit,
         payload_evidence_sha256=file_sha256(attempt_dir/'payload_evidence.json'),payload_cleanup='NOT_PERFORMED')
    dump(attempt_dir/'IMAGE_COMPLETE.json',receipt)
    return receipt


def evaluate(run,phase,resume=False):
    verify_freeze(run)
    gate=read(run/'validation/receipt.json')
    if gate.get('status')!='PASS':raise RuntimeError('mandatory CPU/GPU correctness gate not PASS')
    active=read(run/'active_freeze.json') if (run/'active_freeze.json').exists() else {'path':'freeze.json'}
    frozen_sha=file_sha256(run/active['path'])
    if gate.get('source_freeze_sha256')!=frozen_sha:raise ValueError('validation belongs to another source freeze')
    if phase=='main':
        for pilot in ('gqa_pilot','mt_pilot'):
            receipt=read(run/pilot/'COMPLETED.json')
            if receipt['audit']['status']!='PASS':raise RuntimeError(f'pilot audit not PASS:{pilot}')
            if receipt.get('freeze_sha256')!=frozen_sha:raise ValueError('pilot belongs to another freeze')
    phase_dir=run/phase
    if (phase_dir/'COMPLETED.json').exists():raise FileExistsError('phase already complete; never select favorable reruns')
    phase_dir.mkdir(exist_ok=True)
    manifest,groups=resolve_entries(run,phase)
    if resume:
        # Rehash every adopted raw/setup before loading the GPU model or skipping images.
        committed_rows(phase_dir)
    guard_gpu()
    runner=GLM46VRunner(run/'checkpoint').load()
    dump(phase_dir/'runtime.json',runner.fingerprint()) if not (phase_dir/'runtime.json').exists() else None
    for iid,group in groups.items():
        image_dir=phase_dir/'images'/iid;image_dir.mkdir(parents=True,exist_ok=True)
        if (image_dir/'COMMITTED.json').exists():
            if not resume:raise FileExistsError('adopted image exists; use --resume with unchanged freeze')
            continue
        attempts=list(image_dir.glob('attempt_*'))
        attempt=image_dir/f'attempt_{len(attempts)+1:03d}';attempt.mkdir()
        try:
            receipt=evaluate_image(runner,run,phase,iid,group,attempt)
            dump(image_dir/'COMMITTED.json',receipt)
            print(json.dumps(dict(phase=phase,image_id=iid,status='COMMITTED',requests=receipt['requests'])),flush=True)
        except Exception as exc:
            runner.close()
            dump(attempt/'FAILURE.json',dict(status='FAIL',error=repr(exc),traceback=traceback.format_exc(),
                 image_id=iid,phase=phase,attempt_id=attempt.name,adopted_requests=0,
                 resume_rule='same freeze only; entire image into a NEW attempt; first complete success'))
            raise
    runner.close()
    rows,events=committed_rows(phase_dir)
    audit=audit_requests(rows,manifest,phase=phase)
    if audit['status']!='PASS':
        dump(phase_dir/'AUDIT_FAILURE.json',audit);raise RuntimeError('phase independent raw audit failed')
    dump(phase_dir/'COMPLETED.json',dict(status='PASS',phase=phase,requests=len(rows),audit=audit,
          images=len(groups),freeze_sha256=frozen_sha,setup_events=len(events)))
    print(json.dumps(dict(status='PHASE_PASS',phase=phase,requests=len(rows))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--phase',choices=['gqa_pilot','mt_pilot','main'],required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    evaluate(args.run_dir.resolve(),args.phase,args.resume)

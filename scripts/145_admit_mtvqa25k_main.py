#!/usr/bin/env python3
"""Measured pilot resource admission. Estimates are never main measurements."""
import argparse,json,math,shutil,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mmimpress.mtvqa25k.common import read,dump,sha
from mmimpress.mtvqa25k.adapter import load

def admit(run,model):
    analysis=ROOT/'results'/run.name/model/'pilot40'
    audit=read(analysis/'independent_audit.json')
    if audit['status']!='PASS' or not audit['phase_complete']:raise RuntimeError('independent pilot analysis incomplete')
    completed=read(run/model/'pilot40/COMPLETE.json')
    manifest=read(run/model/'main/manifest.json');n=len(manifest['dialogues'])
    evidence=[]
    for phase in ('pilot250','pilot40'):
        receipt=read(run/model/phase/'COMPLETE.json')
        check=read(ROOT/'results'/run.name/model/phase/'independent_audit.json')
        if check['status']!='PASS' or not check['phase_complete']:raise RuntimeError('pilot prerequisite missing')
        for cp in (run/model/phase/phase/'images').glob('*/COMMITTED.json'):
            c=read(cp);folder=cp.parent/c['attempt_id']
            sizes=sum(p.stat().st_size for p in cp.parent.rglob('*') if p.is_file())
            evidence.append({'phase':phase,'image_id':cp.parent.name,'retained_bytes':sizes})
    sizes=[e['retained_bytes'] for e in evidence]
    # Full raw/index/CSV growth plus a failed-image allowance, beyond the 30GiB floor.
    raw_estimate=math.ceil(float(np.mean(sizes))*n*1.35)
    analysis_reserve=2<<30;failed_reserve=4<<30 if model=='llava' else 1<<30
    working=8<<30 if model=='llava' else 2<<30;floor=30<<30
    required=floor+working+raw_estimate+analysis_reserve+failed_reserve
    free=shutil.disk_usage(run).free
    A=load('_mtvqa_admission_analysis','142_analyze_mtvqa25k.py')
    methods=read(run/model/'model_settings.json')['methods'];stores=[];writes=[];reads=[]
    for cp in (run/model/'pilot40/pilot40/images').glob('*/COMMITTED.json'):
        c=read(cp);folder=cp.parent/c['attempt_id'];setup,_=A.setup(folder,model,methods)
        stores.append(sum(v['logical_bytes'] for k,v in setup.items() if k!='sparsevlm_ssd_kv25_allhead'))
        writes.append(sum(v['write_bytes'] for k,v in setup.items() if k!='sparsevlm_ssd_kv25_allhead'))
        reads.append(sum(r['ssd_read_bytes'] for r in A.rows(folder/'raw.jsonl')))
    p={'status':'PASS' if free>=required else 'BLOCKED','kind':'PILOT_BASED_ESTIMATE_NOT_MAIN_MEASUREMENT','model':model,
       'pilot40_receipt_sha256':sha(run/model/'pilot40/COMPLETE.json'),'independent_pilot_analysis_sha256':sha(analysis/'independent_audit.json'),
       'free_bytes_at_admission':free,'required_free_bytes':required,'minimum_free_bytes':floor,'working_bound_bytes':working,
       'estimated_main_raw_evidence_bytes_with_35pct_margin':raw_estimate,'analysis_reserve_bytes':analysis_reserve,'failed_attempt_reserve_bytes':failed_reserve,
       'pilot_retained_bytes_per_image_mean':float(np.mean(sizes)),'pilot_retained_bytes_per_image_p95':float(np.percentile(sizes,95)),
       'measured_pilot40_max_store_bytes':max(stores),'estimated_main_store_generation_bytes':float(np.mean(stores))*n,
       'estimated_main_OS_write_bytes':float(np.mean(writes))*n,'estimated_main_OS_read_bytes':float(np.mean(reads))*n,
       'estimated_main_wall_seconds':completed['elapsed_seconds']/completed['images']*n,
       'wall_scope':'image execution, audits, archival and payload cleanup; excludes model loading and final analysis',
       'other_model_evidence':'retained in free space; each subsequent model must pass a new admission with its actual available space',
       'weights_dataset_validation':'already resident and included in actual free space; no deletion assumed',
       'pilot_image_evidence':evidence,'created_unix':time.time()}
    dest=run/model/'main_resource_admission.json'
    if dest.exists():
        old=read(dest)
        if old['status']=='PASS':return old
        dest=run/model/f'main_resource_admission_retry_{time.time_ns()}.json'
        dump(dest,p)
        # Keep the failed admission immutable; use an explicit retry receipt link.
        dump(run/model/'main_resource_admission.json',dict(p,previous_admission=old),replace=True)
    else:dump(dest,p)
    print(json.dumps({k:v for k,v in p.items() if k!='pilot_image_evidence'}),flush=True)
    return p

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--model',choices=['llava','qwen','glm'],required=True);a=p.parse_args()
    sys.exit(0 if admit(a.run.resolve(),a.model)['status']=='PASS' else 2)

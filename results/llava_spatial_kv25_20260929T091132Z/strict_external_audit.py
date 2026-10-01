#!/usr/bin/env python3
"""Independent read-only audit of a frozen LLaVA spatial KV25 run.

Usage: python /tmp/audit_spatial_kv25.py RUN_DIR --phase smoke|pilot
Uses only standard-library data readers and NumPy for a separate bootstrap
cross-check. Never imports experiment code or writes to the workspace.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path('/home/dblab/hj/mllm_v2')
OLD = ROOT/'runs/llava_contextual_kv25_20260929T074144Z'
INDEX_SHA = '514d1203d248b6f450f5e3bdacda7b931038f9c11df270b415a2e98e5c77e75a'
WORKLOAD_SHA = '97afe02f924a49cadf0c357175b50185e8f16db12b2dd4402595e2bb99d20f66'
METHODS = ('recompute','fullload','d25_c0','d20_context5','d20_index_uniform5',
           'd20_spatial_uniform5','d20_random5','d21_1_c3_9_reference')
SELECTIVE = METHODS[2:]
D20 = METHODS[3:7]
SCHEMA = 'llava-spatial-uniform-kv25-v1'
OLD_METHODS = {'d25_c0':'d25_c0','d20_context5':'d20_c5',
               'd20_index_uniform5':'d20_uniform5','d20_random5':'d20_random5'}
COMPARE = (('d20_spatial_uniform5','d20_index_uniform5'),
           ('d20_spatial_uniform5','d25_c0'),
           ('d20_context5','d20_index_uniform5'),
           ('d20_spatial_uniform5','d20_context5'),
           ('d20_spatial_uniform5','d20_random5'),
           ('d21_1_c3_9_reference','d25_c0'),
           ('d21_1_c3_9_reference','d20_context5'))


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def canonical_hash(value):
    blob=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def score(pred, gold):
    def norm(s):
        return ' '.join(x for x in re.sub(r'[^\w\s]',' ',str(s).lower()).split()
                        if x not in {'a','an','the'})
    p=norm(pred); g=norm(gold)
    return float(p==g or (bool(g) and p.split()[:len(g.split())]==g.split()))


class Audit:
    def __init__(self):
        self.n=0;self.fail=Counter();self.examples=defaultdict(list)
    def check(self, cond, code, detail=''):
        self.n+=1
        if not cond:
            self.fail[code]+=1
            if len(self.examples[code])<4:self.examples[code].append(str(detail)[:250])
    def read(self,path,code):
        path=Path(path);self.check(path.is_file(),code,str(path))
        if not path.is_file():return None
        try:return json.loads(path.read_text())
        except Exception as e:
            self.check(False,code,f'{path}: {e}');return None


def aux_quota(method,k):
    if method=='d25_c0':return 0
    if method=='d21_1_c3_9_reference':return 10*k//64
    return k//5


def close(x,y,atol=1e-9):
    try:return math.isclose(float(x),float(y),rel_tol=0,abs_tol=atol)
    except (TypeError,ValueError):return False


def geom_rederive(records, dominant, index_aux, spatial, a, tag):
    """Recompute region construction/nearest token by rational arithmetic."""
    coords=spatial['coordinates']; branches=spatial['branch_results']
    d=set(dominant); u=set(index_aux)
    qmap={b:sum(records[i]['branch']==b for i in index_aux) for b in ('base','high')}
    a.check(spatial['branch_quotas']==qmap,'branch_quota',tag)
    selected=[];empty_total=0
    for branch in ('base','high'):
        result=branches[branch];q=qmap[branch]
        grid=coords['base_grid' if branch=='base' else 'high_grid']
        H,W=grid['height'],grid['width']
        remain=sorted(i for i,r in enumerate(records) if r is not None and r['branch']==branch and i not in d)
        a.check(result['quota']==q and result['remaining_count']==len(remain)
                and q<=len(remain),'branch_remaining_quota',f'{tag}:{branch}')
        a.check(len(result['regions'])==q,'region_count',f'{tag}:{branch}')
        if q==0:
            a.check(result['selected_original_ids']==[] and result['region_rows']==0,
                    'zero_quota_regions',f'{tag}:{branch}')
            continue
        rmin=(q+W-1)//W;rmax=min(q,H)
        rideal=math.floor(math.sqrt(q*H/W)+0.5)
        nrows=min(max(rideal,rmin),rmax)
        a.check(result['region_rows']==nrows,'region_rows',f'{tag}:{branch}')
        specs=[];start=0
        for rb in range(nrows):
            nc=q//nrows+(rb<q%nrows)
            for cb in range(nc):
                specs.append((rb,cb,nc,start,Fraction(cb,nc),Fraction(cb+1,nc),
                              Fraction(start,q),Fraction(start+nc,q),
                              Fraction(2*cb+1,2*nc),Fraction(2*start+nc,2*q)))
            start+=nc
        a.check(start==q and len(specs)==q,'region_partition',f'{tag}:{branch}')
        buckets=[[] for _ in range(q)]
        for i in remain:
            rec=records[i];x=Fraction(2*rec['column']+1,2*W);y=Fraction(2*rec['row']+1,2*H)
            found=[j for j,s in enumerate(specs) if s[4]<=x<s[5] and s[6]<=y<s[7]]
            a.check(len(found)==1,'region_membership',f'{tag}:{branch}:{i}')
            if found:buckets[found[0]].append(i)
        picked={}
        def dist(i,j):
            s=specs[j];rec=records[i]
            x=Fraction(2*rec['column']+1,2*W);y=Fraction(2*rec['row']+1,2*H)
            dx=W*(x-s[8]);dy=H*(y-s[9])
            return (dx*dx+dy*dy,i)
        for j,bucket in enumerate(buckets):
            if bucket:picked[j]=(min(bucket,key=lambda i:dist(i,j)),False)
        empty=sum(not bucket for bucket in buckets);empty_total+=empty
        for j,bucket in enumerate(buckets):
            if not bucket:
                free=[i for i in remain if i not in {v[0] for v in picked.values()}]
                a.check(bool(free),'fallback_available',f'{tag}:{branch}:{j}')
                if free:picked[j]=(min(free,key=lambda i:dist(i,j)),True)
        expected=[picked[j][0] for j in range(q) if j in picked]
        selected+=expected
        a.check(result['selected_original_ids']==expected and len(set(expected))==q,
                'nearest_selection',f'{tag}:{branch}')
        a.check(result['empty_region_count']==empty and result['fallback_count']==empty,
                'empty_fallback_counts',f'{tag}:{branch}')
        for j,s in enumerate(specs):
            rg=result['regions'][j]
            rb,cb,nc,st,x0,x1,y0,y1,xc,yc=s
            a.check(rg.get('region_id')==j and rg.get('row_bin')==rb and
                    rg.get('column_bin')==cb and rg.get('rows')==nrows and
                    rg.get('columns_in_row')==nc and rg.get('row_start_unit')==st and
                    rg.get('boundary_rule')=='half_open_except_final_upper_edge',
                    'region_identity',f'{tag}:{branch}:{j}')
            a.check(all(close(rg.get(key),float(v)) for key,v in
                        (('x_min',x0),('x_max',x1),('y_min',y0),('y_max',y1),
                         ('x_center',xc),('y_center',yc),('normalized_area',Fraction(1,q)))),
                    'region_bounds',f'{tag}:{branch}:{j}')
            if j in picked:
                chosen,fallback=picked[j]
                a.check(rg['original_candidate_count']==len(buckets[j]) and
                        rg['originally_empty']==(not buckets[j]) and
                        rg['fallback']==fallback and rg['selected_original_id']==chosen,
                        'region_assignment',f'{tag}:{branch}:{j}')
                a.check(close(rg['distance_to_center'],math.sqrt(float(dist(chosen,j)[0]))),
                        'region_distance',f'{tag}:{branch}:{j}')
    sset=set(selected); uset=set(index_aux)
    a.check(set(spatial['selected_aux_ids'])==sset and
            len(spatial['selected_aux_ids'])==len(selected) and
            set(spatial['added_ids'])==sset-uset and
            set(spatial['removed_ids'])==uset-sset and
            close(spatial['jaccard_index_vs_spatial'],
                  len(sset&uset)/len(sset|uset) if sset|uset else 1.0),
            'spatial_aux_relationship',tag)
    return qmap,empty_total


def histogram(ids,records,branch):
    counts=[0]*64
    for i in ids:
        r=records[i]
        if r is not None and r['branch']==branch:
            y=max(0,min(7,int(r['y']*8)))
            x=max(0,min(7,int(r['x']*8)))
            counts[y*8+x]+=1
    return counts


def cells(ids,records,branch):
    return sum(v>0 for v in histogram(ids,records,branch))


def read_rows(path,a):
    a.check(path.is_file(),'raw_file',str(path));out=[]
    if path.is_file():
        for line_no,line in enumerate(path.open(),1):
            try:out.append(json.loads(line))
            except Exception as e:a.check(False,'raw_json',f'{line_no}: {e}')
    return out


def audit(run,phase):
    a=Audit();nimg=4 if phase=='smoke' else 40;turns=3 if phase=='smoke' else 6
    results=ROOT/'results'/run.name
    a.check(file_sha(ROOT/'data/index.json')==INDEX_SHA,'index_sha')
    index=json.loads((ROOT/'data/index.json').read_text())[:nimg]
    manifest=a.read(run/'workload_manifest.json','workload_manifest')
    if manifest:
        a.check(manifest.get('index_sha256')==INDEX_SHA and
                manifest.get('workload_sha256')==WORKLOAD_SHA and
                manifest.get('image_count')==40,'frozen_workload_hash')
        for idx,e in enumerate(index):
            m=manifest['images'][idx]
            a.check(m['image_id']==e['image_id'] and m['image_path']==e['image_path']
                    and m['question_ids'][:turns]==
                    [str(q['question_id']) for q in e['questions'][4:4+turns]],
                    'manifest_alignment',idx)
    config=a.read(run/'config.json','config')
    if config:
        a.check(config.get('seed')==1234 and config.get('chunk_size')==64 and
                config.get('attention_backend')=='eager' and
                config.get('model_checkpoint_revision')==
                'c916e6cdcd760b4cecd1dd4907f84ac649f93b23' and
                config.get('bootstrap_resamples')==10000 and
                config.get('diagnostic_grid_size') in ([8,8],8,None),
                'config_contract')
    freeze=a.read(run/'source_freeze.json','source_freeze')
    if freeze:
        for key,name in (('protocol_sha256','PROTOCOL.md'),('config_sha256','config.json'),
                         ('workload_manifest_sha256','workload_manifest.json'),
                         ('source_diff_sha256','source.diff'),
                         ('protected_before_sha256','protected_before.json')):
            if key in freeze:a.check(freeze[key]==file_sha(run/name),'freeze_'+name)
        for rel,digest in freeze.get('source_sha256',{}).items():
            a.check((ROOT/rel).is_file() and file_sha(ROOT/rel)==digest,'source_frozen',rel)
    protected=a.read(run/'protected_before.json','protected_before')
    if protected:
        allowed_edits={'mmimpress/piggyback.py','mmimpress/serve.py'}
        for item in protected['files']:
            rel=item['path'];p=ROOT/rel
            if rel in allowed_edits:
                a.check(p.is_file(),'allowed_source_edit_present',rel)
            else:
                a.check(p.is_file() and file_sha(p)==item['sha256_before'],
                        'prior_artifact_protected',rel)
        retries=protected.get('retry_failed_run_artifacts',[])
        if protected.get('retry_of'):
            a.check(len(retries)>=4,'retry_failure_artifact_manifest',
                    protected.get('retry_of'))
        for item in retries:
            rel=item.get('path','');prior=ROOT/rel
            a.check(rel.startswith('runs/llava_spatial_kv25_') and
                    prior.is_file() and prior.stat().st_size==item.get('size_bytes') and
                    file_sha(prior)==item.get('sha256_before'),
                    'failed_retry_artifact_protected',rel)
    cpu=a.read(run/'cpu_validation.json','cpu_validation')
    geometry=a.read(run/'geometry_validation.json','geometry_validation')
    gpu=a.read(run/'gpu_validation.json','gpu_validation')
    for name,receipt,status_field in (('cpu',cpu,'CPU_VALIDATION'),
                                      ('geometry',geometry,'GEOMETRY_VALIDATION'),
                                      ('gpu',gpu,'GPU_CORRECTNESS')):
        if receipt:
            a.check(receipt.get(status_field)=='PASS',f'{name}_correctness_gate')
            if name in ('cpu','geometry'):
                log=run/f'{name}_validation.log'
                a.check(log.is_file() and receipt.get('log_sha256')==file_sha(log),
                        f'{name}_validation_log_sha')
    if gpu:
        a.check(gpu.get('index_sha256')==INDEX_SHA and
                gpu.get('workload_sha256')==WORKLOAD_SHA and
                len(gpu.get('samples',[]))==5 and
                all(set(sample.get('arms',{}))==set(SELECTIVE) and
                        all(v.get('status')=='PASS' for v in sample.get('arms',{}).values())
                        for sample in gpu.get('samples',[])),
                'gpu_five_by_six_gate')
    runner=a.read(run/f'{phase}_runner_manifest.json','runner_manifest')
    if runner:
        a.check(runner.get('expected_requests')==nimg*turns*8 and
                set(runner.get('methods',{}))==set(METHODS) and
                runner.get('frozen_workload',{}).get('selected_workload_sha256')==WORKLOAD_SHA,
                'runner_workload_methods')
    expected={}
    for ii,e in enumerate(index):
        for turn,q in enumerate(e['questions'][4:4+turns],1):
            for m in METHODS:expected[(str(e['image_id']),str(q['question_id']),m)]=(ii,turn,q)
    raw=run/f'{phase}_raw.jsonl';rows=read_rows(raw,a)
    keys=[(str(r.get('image_id')),str(r.get('question_id')),r.get('method_key')) for r in rows]
    counts=Counter(keys);missing=set(expected)-set(counts);extra=set(counts)-set(expected)
    duplicate={k:v for k,v in counts.items() if v!=1}
    a.check(len(rows)==nimg*turns*8,'request_count',len(rows))
    a.check(not missing,'missing_requests',list(missing)[:4]);a.check(not extra,'extra_requests',list(extra)[:4])
    a.check(not duplicate,'duplicate_requests',list(duplicate.items())[:4])
    a.check(not (run/f'{phase}_failure.json').exists(),'phase_failure_receipt')
    a.check((run/f'{phase}_COMPLETED').is_file(),'completed_marker')
    by_image=defaultdict(list);by_arm=defaultdict(list);by_turn=defaultdict(list)
    full_selected_jaccards=[]
    for row,key in zip(rows,keys):
        if key not in expected:continue
        iid,qid,m=key;ii,turn,q=expected[key];tag=f'{iid}:T{turn}:{m}'
        by_image[iid].append(row);by_arm[(iid,m)].append(row);by_turn[(iid,turn)].append(row)
        gold=q.get('answers',[q.get('answer')])
        a.check(row.get('status')=='ok' and row.get('phase')==phase and
                row.get('dataset')=='gqa' and row.get('schema_version')==SCHEMA and row.get('request_id')==f'{phase}:{iid}:{qid}:{m}'
                and row.get('image_index')==ii and row.get('turn_id')==turn,
                'row_identity',tag)
        a.check(row.get('question')==q['question'] and row.get('gold')==gold and
                row.get('prediction')==row.get('answer') and
                row.get('correct')==score(row.get('prediction'),gold[0]),
                'independent_score',tag)
        path='normal_pixel_turn1' if turn==1 else ('normal_pixel_recompute' if m=='recompute' else 'ssd_visual_kv')
        a.check(row.get('request_path')==path and
                row.get('cache_hit')==(turn>1 and m!='recompute'),
                'request_path',tag)
        a.check(isinstance(row.get('end_to_end_ttft_ms'),(int,float)) and
                row['end_to_end_ttft_ms']>0 and row.get('request_e2e_ms',0)>=row['end_to_end_ttft_ms'],
                'timing_positive',tag)
        a.check(row.get('method_id')==m and
                set(row.get('method_order',[]))==set(METHODS) and
                row.get('method_order',[])[row.get('method_order_position',-1)]==m,
                'rotated_method_order',tag)
        if turn==1 or m=='recompute':
            a.check(row.get('vision_forward_count')==1 and
                    row.get('ssd_read_bytes')==0 and row.get('ssd_preads')==0,
                    'normal_pixel_no_ssd',tag)
        else:
            capture=row.get('vision_capture_stats') or {}
            a.check(row.get('vision_forward_count')==0 and
                    capture.get('key_call_count',0)==0 and
                    capture.get('saliency_call_count',0)==0 and
                    row.get('probe_read_bytes',0)==0 and
                    row.get('query_score_calls',0)==0 and
                    row.get('query_scoring_ms',0)==0 and
                    row.get('topk_ms',0)==0 and
                    row.get('rater_selection_ms',0)==0,
                    'hit_no_online_selection',tag)
            io=row.get('io_detail') or {}
            a.check(row.get('ssd_read_bytes')==sum(row.get(f,0) for f in
                    ('normal_kv_read_bytes','separator_read_bytes','probe_read_bytes')) and
                    row.get('ssd_preads')==sum(row.get(f,0) for f in
                    ('normal_kv_preads','separator_preads','probe_preads')) and
                    row.get('ssd_read_bytes')==sum(v.get('bytes',0) for v in io.values()) and
                    row.get('ssd_preads')==sum(v.get('preads',0) for v in io.values()),
                    'os_read_trace_accounting',tag)
            spans=row.get('planned_normal_read_spans') or []
            if m in SELECTIVE:
                a.check(sum(s.get('length_bytes',0) for s in spans)==row.get('normal_kv_read_bytes')
                        and len(spans)==row.get('normal_kv_preads'),
                        'planned_normal_reads',tag)
            trace=row.get('os_pread_trace')
            a.check(isinstance(trace,list),'os_pread_trace_present',tag)
            if isinstance(trace,list):
                a.check(len(trace)==row.get('ssd_preads') and
                        sum(x.get('returned_bytes',0) for x in trace)==row.get('ssd_read_bytes') and
                        all(x.get('returned_bytes')==x.get('requested_bytes') for x in trace),
                        'os_pread_trace_totals',tag)
                if m in SELECTIVE:
                    planned=Counter((f"layer_{int(s['layer']):02d}/{s['kind']}.bin",
                                     int(s['offset_bytes']),int(s['length_bytes'])) for s in spans)
                    actual=Counter((x.get('relative_path'),x.get('offset_bytes'),x.get('returned_bytes'))
                                   for x in trace if x.get('relative_path')!='sep_kv.bin')
                    sep=[x for x in trace if x.get('relative_path')=='sep_kv.bin']
                    a.check(actual==planned and len(sep)==1 and
                            sep[0].get('offset_bytes')==0 and
                            sep[0].get('returned_bytes')==row.get('separator_read_bytes') and
                            len(trace)==len(spans)+1,
                            'os_pread_exact_planned_ranges',tag)
        if m in SELECTIVE and turn>1:
            n=row.get('N_content');k=row.get('k_target')
            a.check(isinstance(n,int) and n>0 and k==(n+3)//4 and
                    row.get('attended_content_kv_count')==k and
                    close(row.get('logical_content_retention'),k/n,1e-12) and
                    row.get('normal_chunks_read')==(k+63)//64,
                    'hit_exact_budget',tag)
            a.check(row.get('selection_artifact')==
                    f'image_artifacts/{phase}/{iid}/{m}_selection.json',
                    'selection_pointer',tag)
    coverage_csv=results/f'{phase}_coverage.csv'
    coverage={}
    if coverage_csv.is_file():
        for r in csv.DictReader(coverage_csv.open()):
            key=(r['image_id'],r['branch'],r['method_key'])
            a.check(key not in coverage,'coverage_unique',key);coverage[key]=r
    else:a.check(False,'coverage_csv',str(coverage_csv))
    a.check(len(coverage)==nimg*4,'coverage_row_count',len(coverage))
    for ii,e in enumerate(index):
        iid=str(e['image_id']); image_rows=by_image[iid]; tag=iid
        image_hash=file_sha(ROOT/e['image_path'])
        a.check({r.get('image_sha256') for r in image_rows}=={image_hash},'image_sha',tag)
        a.check(len({tuple(r.get('method_order',[])) for r in image_rows})==1,'image_method_order',tag)
        offset=(ii+1234)%len(METHODS)
        expected_order=list(METHODS[offset:]+METHODS[:offset])
        a.check(all(r.get('method_order')==expected_order for r in image_rows),
                'deterministic_method_rotation',tag)
        t1=[r for r in image_rows if r['turn_id']==1]
        a.check(len(t1)==8 and len({r.get('prediction') for r in t1})==1 and
                len({r.get('first_token_id') for r in t1})==1 and
                len({r.get('prompt_sha256') for r in t1})==1,
                't1_normal_equivalence',tag)
        for t1row in t1:
            method=t1row['method_key'];capture=t1row.get('vision_capture_stats') or {}
            if method in ('d20_context5','d21_1_c3_9_reference'):
                a.check(capture.get('key_call_count')==1,
                        'contextual_t1_key_capture',f'{tag}:{method}')
            elif method=='d20_spatial_uniform5':
                a.check(capture.get('key_call_count')==0 and
                        capture.get('capture_keys') is False,
                        'spatial_t1_no_key_capture',tag)
        for turn in range(2,turns+1):
            same=[r for r in by_turn[(iid,turn)] if r['method_key'] in SELECTIVE]
            a.check(len(same)==6,'selective_turn_coverage',f'{tag}:T{turn}')
            for field in ('N_content','N_structural','k_target','ssd_read_bytes',
                          'ssd_preads','normal_kv_read_bytes','normal_kv_preads',
                          'separator_read_bytes','separator_preads'):
                a.check(len({r.get(field) for r in same})==1,
                        'selective_equal_'+field,f'{tag}:T{turn}')
        artdir=run/'image_artifacts'/phase/iid
        receipt=a.read(artdir/'image_receipt.json','image_receipt')
        if receipt:
            a.check(receipt.get('validation')=='PASS' and receipt.get('image_id')==iid and
                    receipt.get('phase')==phase and receipt.get('image_sha256')==image_hash and
                    receipt.get('request_count')==turns*8 and
                    receipt.get('canonical_captured_prefix_bitwise_equal') is True and
                    receipt.get('turn1_output_identical') is True,
                    'receipt_validation',tag)
            a.check(receipt.get('request_ids_sha256')==
                    canonical_hash([r['request_id'] for r in image_rows]),
                    'receipt_request_ids',tag)
            a.check(set(receipt.get('selection_artifacts',{}))==set(SELECTIVE) and
                    set(receipt.get('persistence_by_method',{}))==set(METHODS[1:]),
                    'receipt_method_coverage',tag)
            for method,persisted in receipt.get('persistence_by_method',{}).items():
                sizes=persisted.get('file_sizes',{})
                a.check(persisted.get('integrity',{}).get('ok') is True and
                        persisted.get('bytes',{}).get('total')==sum(sizes.values()) and
                        persisted.get('timing_ms',{}).get('persist_ms',-1)>=0 and
                        persisted.get('chunk_size')==64 and
                        persisted.get('dtype')=='float16',
                        'persistence_receipt',f'{tag}:{method}')
                expected_store=run/'stores'/phase/iid/method
                a.check(Path(persisted.get('store_dir','')).resolve()==expected_store.resolve(),
                        'persistence_store_scope',f'{tag}:{method}')
        cleanup=a.read(artdir/'cleanup_receipt.json','cleanup_receipt')
        if cleanup:
            a.check(set(cleanup.get('deleted_store_methods',[]))==set(METHODS[1:]) and
                    cleanup.get('raw_rows_fsynced_before_cleanup') is True and
                    cleanup.get('selection_artifacts_fsynced_before_cleanup') is True and
                    cleanup.get('image_receipt_fsynced_before_cleanup') is True and
                    cleanup.get('reproduction_requires_store_rebuild') is True,
                    'scoped_cleanup',tag)
        artifacts={}
        for m in SELECTIVE:
            art=a.read(artdir/f'{m}_selection.json','selection_artifact')
            if not art:continue
            artifacts[m]=art; n=art.get('N_content');k=art.get('k'); c=aux_quota(m,k)
            plan=art.get('layout_artifact',{}).get('selection_plan',{})
            selected=art.get('selected_original_ids',[]);order=art.get('stored_to_original',[])
            structural=art.get('structural_original_ids',[]);inverse=art.get('original_to_stored',[])
            a.check(n>0 and k==(n+3)//4 and art.get('k_context')==c and
                    art.get('k_dominant')==k-c and len(selected)==k and
                    len(set(selected))==k and set(selected).isdisjoint(structural) and
                    selected==order[:k] and len(order)==n+len(structural) and
                    sorted(order)==list(range(len(order))) and order[n:]==structural and
                    len(inverse)==len(order) and all(inverse[x]==i for i,x in enumerate(order)),
                    'selection_budget_permutation',f'{tag}:{m}')
            a.check(plan.get('k')==k and plan.get('k_context')==c and
                    plan.get('k_dominant')==k-c and
                    plan.get('selected_original_ids')==selected and
                    plan.get('stored_to_original')==order and
                    art.get('layout_artifact',{}).get('calibration_questions')==0 and
                    art.get('layout_artifact',{}).get('layout_uses_dataset_question') is False,
                    'selection_plan_identity',f'{tag}:{m}')
            rank=plan.get('content_saliency_ranking',[])
            a.check(len(rank)==n and set(rank)==set(order[:n]) and
                    selected==[i for i in rank if i in set(selected)] and
                    order[k:n]==[i for i in rank if i not in set(selected)],
                    'stable_saliency_permutation',f'{tag}:{m}')
            dominant=plan.get('dominant_ids',[])
            auxiliary=plan.get('contextual_ids',[])
            a.check(dominant==rank[:k-c] and len(auxiliary)==c and
                    len(set(auxiliary))==c and set(auxiliary).isdisjoint(dominant) and
                    set(dominant)|set(auxiliary)==set(selected),
                    'dominant_auxiliary_partition',f'{tag}:{m}')
            hits=[r for r in by_arm[(iid,m)] if r['turn_id']>1]
            a.check(len(hits)==turns-1 and all(r.get('selected_original_ids')==selected for r in hits),
                    'selection_question_invariance',f'{tag}:{m}')
            if m in OLD_METHODS:
                prev=a.read(OLD/'image_artifacts'/'pilot'/iid/f'{OLD_METHODS[m]}_selection.json',
                            'old_selection_artifact')
                if prev:
                    a.check(prev['selected_original_ids']==selected and
                            prev['stored_to_original']==order and
                            prev['original_to_stored']==inverse,
                            'legacy_selector_regression',f'{tag}:{m}')
            if receipt:
                a.check(receipt['selection_artifacts'].get(m)==
                        str((artdir/f'{m}_selection.json').relative_to(run)) and
                        receipt['N_content']==n and receipt['k_target']==k,
                        'receipt_selection_match',f'{tag}:{m}')
        if len(artifacts)!=len(SELECTIVE):continue
        plans={m:v['layout_artifact']['selection_plan'] for m,v in artifacts.items()}
        a.check(len({tuple(p['content_saliency_ranking']) for p in plans.values()})==1,
                'shared_saliency_ranking',tag)
        if receipt:
            descriptors=receipt.get('descriptor_sha256_by_method',{})
            a.check(descriptors.get('d20_context5')==
                    descriptors.get('d21_1_c3_9_reference') and
                    bool(descriptors.get('d20_context5')),
                    'allocation_same_key_descriptors',tag)
        dominant=plans['d20_context5']['dominant_ids']
        a.check(all(plans[m]['dominant_ids']==dominant for m in D20),
                'four_d20_dominant_equal',tag)
        a.check(all(plans[m]['k_context']==plans['d20_context5']['k_context'] for m in D20),
                'four_d20_aux_equal',tag)
        index_plan=plans['d20_index_uniform5'];remain=index_plan['remaining_ids'];q=index_plan['k_context']
        target=[remain[((2*j+1)*len(remain))//(2*q)] for j in range(q)] if q else []
        index_aux=index_plan['contextual_ids']
        a.check(index_aux==target and index_aux==sorted(set(index_aux)) and
                set(index_aux).isdisjoint(dominant),'index_mid_quantile_exact',tag)
        spatial_plan=plans['d20_spatial_uniform5']; spatial=spatial_plan.get('spatial',{})
        index_selected=set(index_plan['selected_original_ids'])
        spatial_selected=set(spatial_plan['selected_original_ids'])
        full_jaccard=len(index_selected&spatial_selected)/len(index_selected|spatial_selected)
        full_selected_jaccards.append(full_jaccard)
        if receipt:
            relation=receipt.get('spatial_relationships',{})
            idx_aux=set(index_plan['contextual_ids'])
            sp_aux=set(spatial_plan['contextual_ids'])
            aux_jaccard=len(idx_aux&sp_aux)/len(idx_aux|sp_aux) if idx_aux|sp_aux else 1.0
            a.check(close(relation.get('index_spatial_full_selected_jaccard_global'),
                          full_jaccard,1e-12) and
                    close(relation.get('index_spatial_auxiliary_jaccard_global'),
                          aux_jaccard,1e-12),
                    'global_full_and_aux_jaccard_receipt',tag)
        coords=spatial.get('coordinates',{});records=coords.get('records',[])
        if not records:continue
        base=coords['base_grid'];high=coords['high_grid'];tile=coords['tile_grid']
        a.check(len(records)==len(spatial_plan['stored_to_original']) and
                coords.get('coordinate_space')=='model_input_patch_grid' and
                coords.get('structural_original_ids')==artifacts['d20_spatial_uniform5']['structural_original_ids'],
                'coordinate_span_structural',tag)
        structural=set(coords['structural_original_ids'])
        a.check({i for i,r in enumerate(records) if r is None}==structural,
                'coordinate_content_class',tag)
        if phase=='smoke' and ii==0:
            table=artdir/'fixed_coordinate_table.csv'
            marker=artifacts['d20_spatial_uniform5'].get('fixed_real_image_coordinate_table',{})
            a.check(table.is_file() and marker.get('sha256')==file_sha(table)
                    and marker.get('image_id_prefixed_before_analysis')==iid,
                    'fixed_real_coordinate_table_hash',tag)
            if table.is_file():
                coordinate_rows=list(csv.DictReader(table.open()))
                a.check(len(coordinate_rows)==len(records),'fixed_coordinate_table_length',tag)
                for original,(table_row,rec) in enumerate(zip(coordinate_rows,records)):
                    a.check(int(table_row['original_visual_id'])==original and
                            (table_row['content']=='True')==(rec is not None),
                            'fixed_coordinate_table_identity',f'{tag}:{original}')
                    if rec is not None:
                        a.check(table_row['branch']==rec['branch'] and
                                int(table_row['row'])==rec['row'] and
                                int(table_row['column'])==rec['column'] and
                                close(table_row['x'],rec['x'],1e-12) and
                                close(table_row['y'],rec['y'],1e-12),
                                'fixed_coordinate_table_value',f'{tag}:{original}')
        source_seen=set()
        for i,rec in enumerate(records):
            if rec is None:continue
            b=rec.get('branch');grid=base if b=='base' else high
            H,W=grid['height'],grid['width'];r,c=rec['row'],rec['column']
            a.check(b in ('base','high') and rec['original_visual_id']==i and
                    rec['grid_height']==H and rec['grid_width']==W and
                    0<=r<H and 0<=c<W and
                    close(rec['x'],(c+.5)/W,1e-12) and
                    close(rec['y'],(r+.5)/H,1e-12),
                    'coordinate_normalization',f'{tag}:{i}')
            if b=='base':
                a.check(i==r*W+c and rec['source_subimage_index']==0 and
                        rec['source_patch_row']==r and rec['source_patch_column']==c,
                        'base_grid_mapping',f'{tag}:{i}')
            else:
                expected_id=base['height']*base['width']+r*(W+1)+c
                tr,tc=rec['source_tile_row'],rec['source_tile_column']
                pr,pc=rec['source_patch_row'],rec['source_patch_column']
                source=(tr,tc,pr,pc)
                a.check(i==expected_id and 0<=tr<tile['height'] and 0<=tc<tile['width'] and
                        0<=pr<base['height'] and 0<=pc<base['width'] and
                        rec['source_subimage_index']==tr*tile['width']+tc+1 and
                        source not in source_seen,
                        'high_grid_tile_mapping',f'{tag}:{i}')
                source_seen.add(source)
        a.check(sum(r is not None and r['branch']=='base' for r in records)==
                base['height']*base['width'] and
                sum(r is not None and r['branch']=='high' for r in records)==
                high['height']*high['width'],
                'coordinate_branch_counts',tag)
        try:qmap,empty=geom_rederive(records,dominant,index_aux,spatial,a,tag)
        except Exception as ex:a.check(False,'geometry_recompute_exception',f'{tag}: {ex}');continue
        a.check(set(spatial_plan['contextual_ids'])==set(spatial['selected_aux_ids']) and
                set(spatial_plan['contextual_ids']).isdisjoint(dominant),
                'spatial_plan_aux',tag)
        if receipt:
            rel=receipt.get('spatial_relationships',{})
            u=set(index_aux);s=set(spatial['selected_aux_ids'])
            a.check(rel.get('branch_quotas')==qmap and
                    rel.get('dominant_count')==len(dominant) and
                    rel.get('index_aux_count')==len(u) and
                    rel.get('spatial_aux_count')==len(s) and
                    rel.get('added_original_ids')==sorted(s-u) and
                    rel.get('removed_original_ids')==sorted(u-s) and
                    rel.get('dominant_original_ids_sha256')==canonical_hash(dominant) and
                    rel.get('coordinate_mapping_sha256')==canonical_hash(coords) and
                    close(rel.get('index_spatial_auxiliary_jaccard_global'),
                          len(u&s)/len(u|s) if u|s else 1),
                    'receipt_spatial_pairing',tag)
        for branch in ('base','high'):
            for method,aux in (('d20_index_uniform5',index_aux),
                               ('d20_spatial_uniform5',spatial['selected_aux_ids'])):
                row=coverage.get((iid,branch,method))
                if not row:continue
                baux=[i for i in aux if records[i]['branch']==branch]
                expected_cov={'N_branch_content':sum(r is not None and r['branch']==branch for r in records),
                              'dominant_count':sum(records[i]['branch']==branch for i in dominant),
                              'auxiliary_count':len(baux),
                              'auxiliary_occupied_cells':cells(baux,records,branch),
                              'dominant_plus_aux_occupied_cells':cells(dominant+baux,records,branch)}
                bdom={i for i in dominant if records[i]['branch']==branch}
                bidx={i for i in index_aux if records[i]['branch']==branch}
                bsp={i for i in spatial['selected_aux_ids'] if records[i]['branch']==branch}
                full_i=bdom|bidx;full_s=bdom|bsp
                branch_full_j=len(full_i&full_s)/len(full_i|full_s) if full_i|full_s else 1.0
                branch_aux_j=len(bidx&bsp)/len(bidx|bsp) if bidx|bsp else 1.0
                global_aux_j=len(set(index_aux)&set(spatial['selected_aux_ids']))/len(set(index_aux)|set(spatial['selected_aux_ids']))
                a.check(row.get('diagnostic_grid_height')=='8' and
                        row.get('diagnostic_grid_width')=='8' and
                        all(int(row.get(k,-1))==v for k,v in expected_cov.items()) and
                        close(row.get('auxiliary_occupied_fraction'),expected_cov['auxiliary_occupied_cells']/64) and
                        close(row.get('dominant_plus_aux_occupied_fraction'),expected_cov['dominant_plus_aux_occupied_cells']/64),
                        'independent_8x8_coverage',f'{tag}:{branch}:{method}')
                try:
                    aux_hist=json.loads(row['auxiliary_count_grid_8x8_row_major_json'])
                    full_hist=json.loads(row['full_selected_count_grid_8x8_row_major_json'])
                    expect_aux=histogram(baux,records,branch)
                    expect_full=histogram(dominant+baux,records,branch)
                    a.check(aux_hist==expect_aux and full_hist==expect_full and
                            len(aux_hist)==64 and len(full_hist)==64 and
                            sum(aux_hist)==expected_cov['auxiliary_count'] and
                            sum(full_hist)==expected_cov['dominant_count']+expected_cov['auxiliary_count'] and
                            sum(x>0 for x in aux_hist)==expected_cov['auxiliary_occupied_cells'] and
                            sum(x>0 for x in full_hist)==expected_cov['dominant_plus_aux_occupied_cells'],
                            'independent_8x8_histogram',f'{tag}:{branch}:{method}')
                except (KeyError,TypeError,ValueError) as ex:
                    a.check(False,'independent_8x8_histogram',f'{tag}:{branch}:{method}:{ex}')
                a.check(close(row.get('index_spatial_full_selected_jaccard_global'),full_jaccard,1e-12) and
                        close(row.get('index_spatial_auxiliary_jaccard_global'),global_aux_j,1e-12) and
                        close(row.get('index_spatial_full_selected_jaccard_branch'),branch_full_j,1e-12) and
                        close(row.get('index_spatial_auxiliary_jaccard_branch'),branch_aux_j,1e-12),
                        'independent_full_and_aux_jaccard',f'{tag}:{branch}:{method}')
                try:
                    added_global=json.loads(row['spatial_added_ids_global'])
                    removed_global=json.loads(row['spatial_removed_ids_global'])
                    added_branch=json.loads(row['spatial_added_ids_branch'])
                    removed_branch=json.loads(row['spatial_removed_ids_branch'])
                    a.check(added_global==sorted(set(spatial['selected_aux_ids'])-set(index_aux)) and
                            removed_global==sorted(set(index_aux)-set(spatial['selected_aux_ids'])) and
                            added_branch==sorted(bsp-bidx) and
                            removed_branch==sorted(bidx-bsp),
                            'coverage_added_removed_ids',f'{tag}:{branch}:{method}')
                except (KeyError,ValueError,TypeError) as ex:
                    a.check(False,'coverage_added_removed_ids',f'{tag}:{branch}:{method}:{ex}')
                if method=='d20_spatial_uniform5':
                    branch_result=spatial['branch_results'][branch]
                    quota=branch_result['quota']
                    expect_empty=branch_result['empty_region_count']/quota if quota else 0.0
                    expect_fallback=branch_result['fallback_count']/quota if quota else 0.0
                    a.check(close(row.get('spatial_empty_region_fraction'),expect_empty,1e-12) and
                            close(row.get('spatial_fallback_fraction'),expect_fallback,1e-12),
                            'independent_empty_fallback_fraction',f'{tag}:{branch}')
    cleanup_log=run/f'{phase}_cleanup_log.jsonl'
    logrows=read_rows(cleanup_log,a)
    expected_cleanup={(str(e['image_id']),m) for e in index for m in METHODS[1:]}
    intents=Counter();completes=Counter()
    for item in logrows:
        key=(str(item.get('image_id')),item.get('method_key'))
        expected_target=(run/'stores'/phase/key[0]/str(key[1])).resolve()
        a.check(key in expected_cleanup and item.get('phase')==phase and
                Path(item.get('target_path','')).resolve()==expected_target and
                item.get('scope')=='only_run_created_image_store',
                'cleanup_log_scope',key)
        if item.get('event')=='delete_intent':intents[key]+=1
        elif item.get('event')=='delete_complete':completes[key]+=1
        else:a.check(False,'cleanup_log_event',item.get('event'))
    a.check(intents==Counter({k:1 for k in expected_cleanup}) and
            completes==Counter({k:1 for k in expected_cleanup}),
            'cleanup_log_complete',{'intents':len(intents),'completes':len(completes)})
    prior_failures=[]
    for prior in sorted((ROOT/'runs').glob('llava_spatial_kv25_*')):
        if prior.resolve()==run.resolve():continue
        for failure in sorted(prior.glob('*failure*.json')):
            prior_failures.append({'run_dir':str(prior),'file':failure.name,
                                   'sha256':file_sha(failure)})
    protected_ok=not a.fail.get('prior_artifact_protected')
    report={'passed':not a.fail,'phase':phase,'run_dir':str(run.resolve()),
            'independent_audit_script_sha256':file_sha(Path(__file__)),
            'prior_failure_receipts':prior_failures,
            'rerun_of':(freeze or {}).get('rerun_of') or (config or {}).get('rerun_of'),
            'expected_requests':nimg*turns*8,'observed_requests':len(rows),
            'raw_sha256':file_sha(raw) if raw.is_file() else None,
            'missing_count':len(missing),'extra_count':len(extra),'duplicate_count':len(duplicate),
            'independent_scores_checked':len(rows),'checks_evaluated':a.n,
            'full_selected_jaccard_mean':(sum(full_selected_jaccards)/len(full_selected_jaccards)
                                          if full_selected_jaccards else None),
            'protected_artifacts_unchanged':protected_ok,
            'failure_counts':dict(a.fail),'failure_examples':dict(a.examples)}
    return report,rows


def review_stats(run,phase,rows,report):
    """Cross-check analysis CSV directly against raw rows and receipts."""
    import numpy as np
    a=Audit();results=ROOT/'results'/run.name
    summary=results/f'{phase}_summary.csv';pairs=results/f'{phase}_paired_comparisons.csv'
    if not summary.is_file() or not pairs.is_file():
        a.check(False,'analysis_files_missing')
        report.update({'statistics_passed':False,'statistics_checks':a.n,
                       'statistics_failure_counts':dict(a.fail)})
        report['passed']=False
        return report
    sums={r['method_key']:r for r in csv.DictReader(summary.open())}
    a.check(set(sums)==set(METHODS),'summary_eight_methods')
    persisted_csv=results/f'{phase}_persistence.csv'
    a.check(persisted_csv.is_file(),'persistence_csv')
    if persisted_csv.is_file():
        persisted_rows=list(csv.DictReader(persisted_csv.open()))
        a.check(len(persisted_rows)==(4 if phase=='smoke' else 40)*7,
                'persistence_row_count',len(persisted_rows))
        t1_by={(r['image_id'],r['method_key']):r for r in rows if r['turn_id']==1}
        seen=set()
        timing_fields=('kv_materialize_ms','permutation_ms','descriptor_mapping_ms',
                       'clustering_ms','geometry_mapping_ms','spatial_selection_ms',
                       'token_mapping_ms','kv_repack_ms','store_write_ms',
                       'file_fsync_ms','directory_fsync_ms','persist_ms')
        capture_fields={'capture_saliency_reduction_ms':'saliency_reduction_ms',
                        'capture_key_reduction_ms':'key_reduction_ms',
                        'capture_key_hook_submit_ms':'key_hook_submit_ms',
                        'capture_key_materialize_ms':'key_materialize_ms'}
        for row in persisted_rows:
            key=(row['image_id'],row['method_key'])
            a.check(key not in seen and key[1] in METHODS[1:],'persistence_unique',key)
            seen.add(key)
            receipt=json.loads((run/'image_artifacts'/phase/key[0]/'image_receipt.json').read_text())
            persisted=receipt['persistence_by_method'][key[1]]
            timing=persisted['timing_ms']
            a.check(all(close(row[field],timing.get(field,0),1e-6)
                        for field in timing_fields) and
                    close(row['bytes_total'],persisted['bytes']['total'],1e-6) and
                    close(row['activation_ms'],receipt['activation_ms_by_method'][key[1]],1e-6) and
                    close(row['canonical_comparison_ms'],
                          receipt['canonical_comparison_ms'][key[1]],1e-6),
                    'persistence_receipt_timing',key)
            capture=t1_by[key].get('vision_capture_stats') or {}
            a.check(all(close(row[col],capture.get(field,0),1e-6)
                        for col,field in capture_fields.items()),
                    'persistence_capture_timing',key)
    for m in METHODS:
        own=[r for r in rows if r['method_key']==m];hit=[r for r in own if r['turn_id']>1]
        if not own or not hit:continue
        s=sums[m]
        checks={'requests':len(own),'hits':len(hit),
                'all_accuracy':sum(r['correct'] for r in own)/len(own),
                'hit_accuracy':sum(r['correct'] for r in hit)/len(hit),
                'hit_ttft_mean_ms':sum(r['end_to_end_ttft_ms'] for r in hit)/len(hit),
                'total_ssd_mb_per_hit':sum(r['ssd_read_bytes'] for r in hit)/len(hit)/1e6,
                'total_preads_per_hit':sum(r['ssd_preads'] for r in hit)/len(hit),
                'hit_e2e_mean_ms':sum(r['request_e2e_ms'] for r in hit)/len(hit),
                't1_ttft_mean_ms':sum(r['end_to_end_ttft_ms'] for r in own if r['turn_id']==1)/
                                   sum(r['turn_id']==1 for r in own),
                'actual_content_retention':sum(r['logical_content_retention'] for r in hit)/len(hit),
                'structural_inclusive_retention':sum(r['visual_retention_including_structural'] for r in hit)/len(hit)}
        for key,value in checks.items():a.check(close(s[key],value,1e-6),'summary_'+key,m)
        if m in SELECTIVE:
            artifacts=[json.loads((run/'image_artifacts'/phase/iid/f'{m}_selection.json').read_text())
                       for iid in sorted({r['image_id'] for r in own})]
            a.check(close(s['k_mean'],sum(x['k'] for x in artifacts)/len(artifacts)) and
                    close(s['k_dominant_mean'],sum(x['k_dominant'] for x in artifacts)/len(artifacts)) and
                    close(s['k_context_mean'],sum(x['k_context'] for x in artifacts)/len(artifacts)),
                    'summary_exact_token_means',m)
        if m!='recompute':
            receipts=[json.loads(path.read_text()) for path in
                      sorted((run/'image_artifacts'/phase).glob('*/image_receipt.json'))]
            persist=[x['persistence_by_method'][m]['timing_ms']['persist_ms'] for x in receipts]
            a.check(close(s['persistence_mean_ms'],sum(persist)/len(persist),1e-6),
                    'summary_persistence_mean',m)
        for t in range(1,(4 if phase=='smoke' else 7)):
            turn=[r for r in own if r['turn_id']==t]
            a.check(close(s[f'turn{t}_accuracy'],sum(r['correct'] for r in turn)/len(turn),1e-9),
                    'summary_turn_accuracy',f'{m}:T{t}')
    pairrows={(r['new'],r['old']):r for r in csv.DictReader(pairs.open())}
    a.check(set(pairrows)==set(COMPARE),'seven_fixed_comparisons')
    for new,old in COMPARE:
        r=pairrows[(new,old)]
        left={(x['image_id'],x['question_id']):x for x in rows if x['method_key']==new and x['turn_id']>1}
        right={(x['image_id'],x['question_id']):x for x in rows if x['method_key']==old and x['turn_id']>1}
        a.check(set(left)==set(right),'paired_request_identity',f'{new}:{old}')
        both_right=both_wrong=a_only=b_only=agree=0
        for key in left.keys()&right.keys():
            x=left[key];y=right[key];c1=bool(x['correct']);c2=bool(y['correct'])
            both_right+=c1 and c2;both_wrong+=(not c1 and not c2)
            a_only+=c1 and not c2;b_only+=c2 and not c1
            agree+=x['prediction']==y['prediction']
        for col,val in (('both_correct',both_right),('both_wrong',both_wrong),
                        ('method_a_only_correct',a_only),('method_b_only_correct',b_only)):
            a.check(int(r[col])==val,'paired_confusion_'+col,f'{new}:{old}')
        a.check(close(r['prediction_agreement_fraction'],agree/len(left),1e-12) and
                close(r['quality_difference_pp'],100*(a_only-b_only)/len(left),1e-8),
                'paired_agreement_delta',f'{new}:{old}')
        image_ids=sorted({iid for iid,_ in left})
        quality=np.asarray([[np.mean([left[(iid,qid)]['correct'] for image,qid in left if image==iid]),
                             np.mean([right[(iid,qid)]['correct'] for image,qid in right if image==iid])]
                            for iid in image_ids],dtype=np.float64)
        ttft=np.asarray([[np.mean([left[(iid,qid)]['end_to_end_ttft_ms'] for image,qid in left if image==iid]),
                          np.mean([right[(iid,qid)]['end_to_end_ttft_ms'] for image,qid in right if image==iid])]
                         for iid in image_ids],dtype=np.float64)
        draws=np.random.default_rng(1234).integers(0,len(image_ids),size=(10000,len(image_ids)))
        qdraw=quality[draws].mean(axis=1)
        tdraw=ttft[draws].mean(axis=1)
        qci=np.quantile((qdraw[:,0]-qdraw[:,1])*100,[.025,.975])
        tci=np.quantile(tdraw[:,0]-tdraw[:,1],[.025,.975])
        ratio=np.quantile(tdraw[:,0]/tdraw[:,1],[.025,.975])
        values={
            'quality_ci95_low_pp':qci[0], 'quality_ci95_high_pp':qci[1],
            'hit_ttft_difference_ms':ttft[:,0].mean()-ttft[:,1].mean(),
            'hit_ttft_ci95_low_ms':tci[0], 'hit_ttft_ci95_high_ms':tci[1],
            'hit_ttft_ratio':ttft[:,0].mean()/ttft[:,1].mean(),
            'hit_ttft_ratio_ci95_low':ratio[0], 'hit_ttft_ratio_ci95_high':ratio[1],
        }
        for field,value in values.items():
            a.check(close(r[field],value,1e-6),'paired_bootstrap_'+field,f'{new}:{old}')
        a.check(int(r['image_clusters'])==len(image_ids) and
                int(r['resamples'])==10000 and int(r['seed'])==1234,
                'paired_bootstrap_contract',f'{new}:{old}')
    report.update({'statistics_passed':not a.fail,'statistics_checks':a.n,
                   'statistics_failure_counts':dict(a.fail),
                   'statistics_failure_examples':dict(a.examples)})
    report['passed']=report['passed'] and not a.fail
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_dir',type=Path)
    p.add_argument('--phase',choices=('smoke','pilot'),default='pilot')
    args=p.parse_args();report,rows=audit(args.run_dir,args.phase)
    if rows:report=review_stats(args.run_dir,args.phase,rows,report)
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0 if report['passed'] else 1

if __name__=='__main__':sys.exit(main())

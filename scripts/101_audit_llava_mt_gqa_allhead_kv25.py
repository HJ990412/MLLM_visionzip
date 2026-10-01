#!/usr/bin/env python3
"""Independent raw audit/report for main v1; imports no production inference code."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
METHODS = ('recompute','fullload','mpic32_ssd','sparsevlm_ssd_kv25_allhead','ours_kv25')
ALLHEAD = METHODS[3]
LABELS = dict(zip(METHODS, ('ReComp','FullLoad','MPIC-32','SparseVLM-SSD-KV25-AllHead','Ours-KV25')))
SCHEMA = 'llava-mtgqa-allhead-kv25-main-v1'
MODEL = 'llava-hf/llava-v1.6-vicuna-7b-hf'

def helper(name, file):
    spec = importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
A = helper('_allhead_independent_existing_oracles','99_audit_sparsevlm_ssd_kv25.py')
sha=A.sha

def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def read_rows(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():yield json.loads(line)

def dump(path, value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')

def audit_mpic(row, c):
    r=row['result'];g=row['geometry'];rid=row['request_id'];width=g['num_heads']*g['head_dim']*2
    expected=Counter()
    for li in range(g['num_layers']):
        for kind in ('k.bin','v.bin'):expected[(li,kind,0,g['v_token_num']*width,g['v_token_num']*width)]+=1
    hidden_size=g['num_heads']*g['head_dim'];embedding=32*hidden_size*2
    expected[(None,'visual_input.bin',0,embedding,embedding)]+=1
    actual=Counter()
    for e in row['result']['os_pread_trace']:
        p=Path(e['path']);li=int(p.parent.name.split('_')[1]) if p.parent.name.startswith('layer_') else None
        actual[(li,p.name,e['offset'],e['requested'],e['returned'])]+=1
    c.require(actual==expected,'mpic_actual_ranges',rid)
    for k,v in {'k_recompute':32,'retained_image_context_ratio':1.0,'decoder_prefill_pass_count':1,
        'decode_cache_append_exact':True,'same_source_target_context':True}.items():c.require(r.get(k)==v,'mpic_'+k,rid)
    c.require(r.get('selected_image_local_rows')==list(range(32)),'mpic_selected_rows',rid)
    c.require(row['k']==row['N_content'],'mpic_full_retention',rid)
    c.require(r.get('ssd_embedding_bytes')==embedding,'mpic_embedding_actual_bytes',rid)

def audit_image_rows(rows, manifest, config, phase, image_id):
    c=A.Checks();ds=[d for d in manifest['dialogues'] if d['image_id']==image_id]
    if phase in ('smoke','integration'):
        ids=manifest['smoke_dialogue_ids'][:1] if phase=='integration' else manifest['smoke_dialogue_ids']
        ds=[d for d in ds if d['dialog_id'] in ids]
    c.require(bool(ds),'known_image',image_id)
    expected={(d['dialog_id'],m,t):d['turns'][t-1] for d in ds for m in METHODS for t in (1,2,3)}
    bykey={(r['dialog_id'],r['method_id'],r['turn_id']):r for r in rows}
    c.require(set(bykey)==set(expected) and len(rows)==len(expected),'image_completeness',image_id)
    c.require(len({r['logical_request_id'] for r in rows})==len(rows),'unique_logical_ids',image_id)
    c.require(len({r['physical_execution_id'] for r in rows})==len(rows),'unique_physical_ids',image_id)
    ordinals={d['dialog_id']:i for i,d in enumerate(manifest['dialogues'])}
    ours_sets=[]
    for key,r in bykey.items():
        rid=r['request_id'];did,m,t=key;result=r['result'];question=expected.get(key)
        if question is None:continue
        c.require(r['schema_version']==SCHEMA and r['model_id']==MODEL,'schema_model',rid)
        expected_id=f"{SCHEMA}:{r['experiment_id']}:{phase}:{MODEL}:{m}:{did}:t{t}"
        c.require(rid==r['logical_request_id']==expected_id,'logical_identity',rid)
        c.require(r['phase']==phase and r['image_id']==image_id,'phase_image',rid)
        c.require(r['question_id']==str(question['question_id']) and r['question']==question['question'] and r['gold']==question['answers'],'frozen_question_gold',rid)
        c.require(r['score']==A.independent_score(r['prediction'],r['gold'],phase='mt'),'strict_em_recalculation',rid)
        c.require(r['prediction']==result['answer'],'raw_decoded_prediction',rid)
        c.require(r['config_sha256']==config['_file_sha256'] and r['manifest_sha256']==config['_manifest_sha256'],'config_manifest_hash',rid)
        c.require(r['code_sha256']==canonical(config['source_sha256']),'code_hash',rid)
        ordinal=ordinals[did];i=ordinal%5;order=list(METHODS[i:]+METHODS[:i])
        c.require(r['global_dialog_ordinal']==ordinal and r['method_order']==order and r['method_order_position']==order.index(m),'deterministic_rotation',rid)
        c.require(r['image_sha256']==manifest['image_identity'][image_id]['sha256'],'frozen_image_hash',rid)
        history=[];entries=[]
        for prior_t in range(1,t):
            prev=bykey.get((did,m,prior_t));q=expected.get((did,m,prior_t))
            if prev is None:c.require(False,'missing_history_source',rid);continue
            history += [f"Q{prior_t}: {q['question'].strip()}",f"A{prior_t}: {prev['prediction']}"]
            entries.append({'turn_id':prior_t,'question_id':str(q['question_id']),'question':q['question'].strip(),
                'answer':prev['prediction'],'answer_source':'method_local_generated','source_model':MODEL,
                'source_method_id':m,'source_dialog_id':did,'source_logical_request_id':prev['logical_request_id'],
                'source_physical_execution_id':prev['physical_execution_id']})
        hist='\n'.join(history);body=([hist,''] if hist else [])
        body += [f"Current question Q{t}: {question['question'].strip()}",'Answer the current question with a single word or short phrase. ASSISTANT:']
        prompt='USER: <image>\n'+'\n'.join(body)
        c.require(r['history']==hist and r['history_entries']==entries,'exact_method_local_history_lineage',rid)
        c.require(r['prompt']==prompt,'causal_prompt_reconstruction',rid)
        c.require(r['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest(),'prompt_hash',rid)
        c.require(r['history_sha256']==hashlib.sha256(hist.encode()).hexdigest(),'history_hash',rid)
        c.require(r['suffix_ids_sha256']==canonical(r['suffix_input_ids']),'suffix_hash',rid)
        c.require(0<r['ttft_ms']<=r['request_e2e_ms'],'valid_ttft_e2e',rid)
        c.require(math.isclose(r['ttft_ms'],1000*(r['first_token_at_s']-r['request_started_at_s']),abs_tol=1e-7),'true_ttft_recalculation',rid)
        c.require(math.isclose(r['request_e2e_ms'],1000*(r['request_finished_at_s']-r['request_started_at_s']),abs_tol=1e-5),'request_e2e_recalculation',rid)
        tokens=result['generated_token_ids']
        c.require(1<=len(tokens)<=16 and tokens[0]==result['first_token_id'],'actual_generated_tokens',rid)
        c.require(result['vision_forward_count']==int(t==1 or m=='recompute'),'vision_call_contract',rid)
        trace=result.get('os_pread_trace',[])
        c.require(r['ssd_read_bytes']==sum(e['returned'] for e in trace),'actual_returned_bytes',rid)
        c.require(r['ssd_preads']==len(trace),'actual_pread_count',rid)
        c.require(all(e['returned']==e['requested'] for e in trace),'successful_full_requested_reads',rid)
        c.require(not any('probe_k.bin' in e['path'] for e in trace),'no_probe_read_any_main_method',rid)
        g=r['geometry'];n=g['v_token_num']-len(set(g['newline_idx'])|set(g['padding_idx']))
        c.require(n==r['N_content'] and n>0,'actual_content_geometry',rid)
        if t==1 or m=='recompute':
            c.require(r['k']==n and r['ssd_read_bytes']==0,'full_pixel_request',rid)
            c.require(r['source_store'] is None,'pixel_not_cache_hit',rid)
        else:
            c.require(result.get('adapter_restored') is True,'adapter_restored',rid)
            c.require(bool(result.get('conditioning_events')) and all(x['status']=='SUCCESS' for x in result['conditioning_events']),'conditioning_success',rid)
            conditioned={x['path'] for x in result['conditioning_events']}
            c.require(all(e['path'] in conditioned for e in trace),'every_read_file_conditioned',rid)
            src=r['source_store'];c.require(src is not None,'store_provenance',rid)
            c.require(src['source']['dialog_id']==ds[0]['dialog_id'] and src['source']['method_id']==('fullload' if m==ALLHEAD else m),'first_source_T1_only',rid)
            source_row=bykey.get((ds[0]['dialog_id'],src['source']['method_id'],1),{})
            c.require(src['source']['physical_execution_id']==source_row.get('physical_execution_id'),'actual_source_physical_id',rid)
            ra=dict(r,os_pread_trace=trace)
            if m==ALLHEAD:
                A.audit_sparse_hit(ra,c)
                c.require(r['budget_unit']=='visual_kv' and r['scoring_head_policy']=='all','explicit_allhead_config',rid)
                c.require(result['suffix_input_ids']==r['suffix_input_ids'],'same_suffix_tokenization',rid)
                layers=result['layers'];suffix=result['suffix_input_ids'];raters=set(result['rater_ids'])
                c.require(all(i<len(suffix) for i in raters),'raters_in_actual_suffix',rid)
                c.require(any(s['kind']=='current_question' for s in result['text_spans']),'current_question_span',rid)
                c.require(any(s['kind']=='history' for s in result['text_spans']),'history_span',rid)
            elif m in ('fullload','ours_kv25'):
                A.audit_existing_hit(ra,c)
                if m=='ours_kv25':
                    c.require(r['k']==(n+3)//4 and r['budget_unit']=='visual_kv','native_ours_KV25',rid)
                    ours_sets.append(result['selected_original_ids'])
            else:audit_mpic(r,c)
        if m!=ALLHEAD or t==1:
            calls=result.get('projection_calls',{})
        else:calls=result['projection_calls']
        for li in range(g['num_layers']):
            observed=calls.get(str(li),{})
            c.require(observed.get('prefill')=={'q':1,'k':1,'v':1},'one_prefill_projection_each_layer',rid)
            expected_decode={'q':len(tokens)-1,'k':len(tokens)-1,'v':len(tokens)-1}
            c.require(observed.get('decode',{'q':0,'k':0,'v':0})==expected_decode,'decode_projection_counts',rid)
    c.require(not ours_sets or all(x==ours_sets[0] for x in ours_sets),'ours_same_image_selection_invariant',image_id)
    return dict(c.result(),image_id=image_id,phase=phase,requests=len(rows),expected_requests=len(expected),
        dialogs=len(ds),allhead_K_reuse='PASS' if not c.failures else 'UNRESOLVED')

def csv_write(path, rows):
    rows=list(rows)
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def mean(values):return float(np.mean(values)) if len(values) else None

def io_categories(r):
    d=dict(full_K_scoring=0,selected_K=0,selected_V=0,structural_other=0)
    for e in r['result'].get('os_pread_trace',[]):
        name=Path(e['path']).name
        if name=='k.bin':key='full_K_scoring' if r['method_id']==ALLHEAD else 'selected_K'
        elif name=='v.bin':key='selected_V'
        else:key='structural_other'
        d[key]+=e['returned']
    return d

def compact_row(r):
    result=r['result'];layers=result.get('layers',[]);geo=r['geometry']
    total_chunks=math.ceil(geo['v_token_num']/64)
    if r['turn_id']>1:
        if r['method_id']==ALLHEAD:coverage=mean([len(x['selected_chunk_ids'])/total_chunks for x in layers])
        elif r['method_id']=='ours_kv25':coverage=math.ceil(r['k']/64)/total_chunks
        elif r['method_id']=='recompute':coverage=0.0
        else:coverage=1.0
    else:coverage=0.0
    d={k:r[k] for k in ('method_id','image_id','dialog_id','turn_id','score','ttft_ms','request_e2e_ms','ssd_read_bytes','ssd_preads','N_content','k','content_kv_fraction')}
    d.update(io_categories(r));d['selected_V_chunk_coverage']=coverage
    d['cap_reached']=bool(result.get('cap_reached',False));d['generated_token_count']=len(result['generated_token_ids'])
    d['peak_gpu_allocated_bytes']=result.get('peak_gpu_allocated_bytes',0);d['peak_gpu_reserved_bytes']=result.get('peak_gpu_reserved_bytes',0)
    d['extra_real_rows_mean']=mean([x['plan']['extra_real_rows'] for x in layers]) if layers else result.get('extra_real_rows',0)
    d['allhead_signature']=canonical([x['selected_token_ids'] for x in layers]) if layers else None
    d['allhead_unique_layer_selections']=len({canonical(x['selected_token_ids']) for x in layers}) if layers else None
    d['logical_reused_K_bytes']=result.get('io',{}).get('logical_reused_k_bytes',0)
    d['duplicated_K_bytes']=result.get('io',{}).get('duplicated_k_bytes',0)
    d['rater_fallback']=bool(result.get('rater_fallback',False))
    d['allhead_host_intervals_ms']=json.dumps(result.get('timing',{}).get('host_intervals_ms',{}))
    d['allhead_cuda_intervals_ms']=json.dumps(result.get('timing',{}).get('cuda_intervals_ms',{}))
    return d

def summarize(rows):
    summary=[];budget=[]
    for m in METHODS:
        rs=[r for r in rows if r['method_id']==m];hits=[r for r in rs if r['turn_id']>1];t1=[r for r in rs if r['turn_id']==1]
        if not rs:continue
        rec={'method_id':m,'label':LABELS[m],'requests':len(rs),'hits':len(hits)}
        rec.update({f'acc_T{t}':mean([r['score'] for r in rs if r['turn_id']==t]) for t in (1,2,3)})
        rec.update(all_acc=mean([r['score'] for r in rs]),hit_acc=mean([r['score'] for r in hits]),
            T1_ttft_mean_ms=mean([r['ttft_ms'] for r in t1]),hit_ttft_mean_ms=mean([r['ttft_ms'] for r in hits]),
            hit_ttft_p50_ms=float(np.percentile([r['ttft_ms'] for r in hits],50)),
            hit_ttft_p95_ms=float(np.percentile([r['ttft_ms'] for r in hits],95)),
            SSD_MB_per_hit=mean([r['ssd_read_bytes']/1e6 for r in hits]),preads_per_hit=mean([r['ssd_preads'] for r in hits]))
        summary.append(rec)
        b={'method_id':m,'logical_content_retention':mean([r['content_kv_fraction'] for r in hits])}
        b.update({k+'_MB':mean([r[k]/1e6 for r in hits]) for k in ('full_K_scoring','selected_K','selected_V','structural_other')})
        b.update(selected_V_chunk_coverage=mean([r['selected_V_chunk_coverage'] for r in hits]),total_preads_per_hit=rec['preads_per_hit'],
            full_K_coverage=1.0 if m==ALLHEAD else None,logical_reused_K_MB=mean([r['logical_reused_K_bytes']/1e6 for r in hits]),
            duplicated_K_MB=mean([r['duplicated_K_bytes']/1e6 for r in hits]))
        budget.append(b)
    return summary,budget

def paired(rows):
    hit=[r for r in rows if r['turn_id']>1];images=sorted({r['image_id'] for r in hit});lookup={(r['dialog_id'],r['turn_id'],r['method_id']):r for r in hit}
    rng=np.random.default_rng(1234);samples=rng.integers(0,len(images),size=(10000,len(images)))
    out=[]
    for m in METHODS[:-1]:
        totals=np.zeros((len(images),4));base=[];ours=[]
        for i,image in enumerate(images):
            for r in hit:
                if r['method_id']!=m or r['image_id']!=image:continue
                o=lookup[(r['dialog_id'],r['turn_id'],'ours_kv25')];base.append(r);ours.append(o)
                totals[i]+=[o['score']-r['score'],o['ttft_ms']-r['ttft_ms'],o['ssd_read_bytes']-r['ssd_read_bytes'],1]
        res=totals[samples].sum(axis=1);dist=res[:,:3]/res[:,3,None];point=totals[:,:3].sum(0)/totals[:,3].sum()
        low,high=np.percentile(dist,[2.5,97.5],axis=0)
        btt=mean([r['ttft_ms'] for r in base]);ott=mean([r['ttft_ms'] for r in ours]);bb=mean([r['ssd_read_bytes'] for r in base]);ob=mean([r['ssd_read_bytes'] for r in ours])
        out.append({'baseline':m,'direction':'Ours-minus-baseline','delta_hit_acc_pp':point[0]*100,'acc_ci_low_pp':low[0]*100,'acc_ci_high_pp':high[0]*100,
            'delta_hit_TTFT_ms':point[1],'TTFT_ci_low_ms':low[1],'TTFT_ci_high_ms':high[1],
            'delta_SSD_bytes':point[2],'SSD_ci_low_bytes':low[2],'SSD_ci_high_bytes':high[2],
            'TTFT_reduction_percent':100*(1-ott/btt),'SSD_reduction_percent':100*(1-ob/bb) if bb else None,
            'bootstrap':'image_cluster_request_weighted','resamples':10000,'seed':1234,'images':len(images)})
    return out

def protection(run):
    p=helper('_main_protection_fingerprint','69_protect_rekv_artifacts.py');changed=[];checked=0;size=0
    for row in read_rows(run/'protected_artifacts_before.jsonl'):
        path=ROOT/row['path'];checked+=1
        if not path.exists() and not path.is_symlink():changed.append({'path':row['path'],'reason':'MISSING'});continue
        if 'symlink' in row:
            import os
            if not path.is_symlink() or os.readlink(path)!=row['symlink']:changed.append({'path':row['path'],'reason':'SYMLINK_CHANGED'})
        else:
            fingerprint,st=p._fingerprint_regular_file(path,p.DEFAULT_POLICY);size+=st.st_size
            if fingerprint!=row['integrity'] or p._file_metadata(st)!=row['metadata']:changed.append({'path':row['path'],'reason':'CONTENT_OR_METADATA_CHANGED'})
    before=json.loads((run/'source_before.json').read_text());source_changes=[]
    for name,digest in before.items():
        if not (ROOT/name).is_file() or sha(ROOT/name)!=digest:source_changes.append(name)
    return {'status':'PASS' if not changed and not source_changes else 'FAIL','protected_files':checked,'protected_logical_bytes':size,
        'changed_artifacts':changed,'changed_existing_source':source_changes,'policy':p.DEFAULT_POLICY.as_dict(),
        'limitation':'large prior artifacts use nine-window framed SHA256 plus inode/ctime/mtime; full SHA256 for source and small artifacts',
        'new_source_files':[str(x.relative_to(ROOT)) for x in sorted((ROOT/'scripts').glob('10[012]_*.py'))]}

def session_tables(run, rows, phase='main'):
    details=[];by=defaultdict(list)
    for r in rows:by[(r['image_id'],r['dialog_id'],r['method_id'])].append(r)
    setups={};source_ids={}
    for cp in (run/phase/'images').glob('*/COMMITTED.json'):
        commit=json.loads(cp.read_text());folder=cp.parent/commit['attempt_id'];image=cp.parent.name
        for method in METHODS:
            kind={'fullload':'raster',ALLHEAD:'raster','mpic32_ssd':'mpic','ours_kv25':'image_only'}.get(method)
            if kind:
                p=json.loads((folder/f'{kind}_persistence.json').read_text());a=json.loads((folder/f'{method}_activation.json').read_text())
                persist=p['timing_ms']['persist_ms'];activate=a['outer_wall_ms']
                # Hash-verification activation is actual measured setup; never hidden
                # as ordinary metadata-free serving. Independent deployment is derived.
                setups[(image,method)]={'persist_ms':persist,'activation_ms':activate,'write_bytes':p['bytes']['total'],
                    'unused_probe_bytes':p['bytes'].get('probe_sidecar',p.get('meta',{}).get('bytes_probe_sidecar',0)),
                    'source_dialog_id':p['source']['dialog_id'],'actual_persist_ms':0.0 if method==ALLHEAD else persist}
                source_ids[image]=p['source']['dialog_id']
            else:setups[(image,method)]={'persist_ms':0.,'activation_ms':0.,'write_bytes':0,'actual_persist_ms':0.,'source_dialog_id':None,'unused_probe_bytes':0}
    for (image,did,m),rs in by.items():
        if len(rs)!=3:raise ValueError('incomplete session')
        s=setups[(image,m)];source=did==s['source_dialog_id'];request=sum(r['request_e2e_ms'] for r in rs)
        actual_setup=(s['actual_persist_ms']+s['activation_ms']) if source else 0.
        details.append({'image_id':image,'dialog_id':did,'method_id':m,'source_dialogue':source,'request_E2E_sum_ms':request,
            'actual_setup_ms':actual_setup,'actual_stream_session_E2E_ms':request+actual_setup,
            'derived_standalone_E2E_ms':request+s['persist_ms']+s['activation_ms'],
            'persistence_ms_image':s['persist_ms'],'measured_write_bytes_image':s['write_bytes'],
            'scope':'AllHead shares FullLoad canonical store; actual build charged once to FullLoad; standalone is DERIVED' if m==ALLHEAD else 'MEASURED source-only setup; standalone DERIVED'})
    sums=[]
    for m in METHODS:
        rs=[r for r in details if r['method_id']==m];ss=[v for (i,mm),v in setups.items() if mm==m]
        if not rs:continue
        sums.append({'method_id':m,'provisioning_policy':'FRESH_IMAGE_STREAMING',
            'persistence_status':'MEASURED_SHARED_WITH_FULLLOAD' if m==ALLHEAD else 'MEASURED',
            'persistence_ms_per_image':mean([s['persist_ms'] for s in ss]),
            'actual_charged_persistence_ms_per_image':mean([s['actual_persist_ms'] for s in ss]),
            'write_MB_per_image':mean([s['write_bytes']/1e6 for s in ss]),
            'actual_stream_session_E2E_mean_ms':mean([r['actual_stream_session_E2E_ms'] for r in rs]),
            'derived_standalone_E2E_mean_ms':mean([r['derived_standalone_E2E_ms'] for r in rs])})
    return details,sums

def render(audit, summary, budget, comparisons, sessions):
    lines=['# LLaVA MT-GQA AllHead / KV25 본실험','',
        f"MAIN 5-ARM: **{audit['MAIN_5_ARM']}**, 실제 final 요청 **{audit['main_requests']:,}/60,915**. READY FOR PAPER MAIN TABLE: **{audit['READY_FOR_PAPER_MAIN_TABLE']}**.",
        '', '기존 frozen MT-GQA-reconstructed의 method-local generated history이며 strict normalized exact match를 쓴다. 공식 MetaCompress artifact/evaluator 재현은 아니다.',
        '', '| Method | Acc T1/T2/T3 (%) | All Acc (%) | Hit Acc (%) | T1 TTFT (ms) | Hit TTFT mean/p50/p95 (ms) | SSD MB/hit |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        lines.append(f"| {s['label']} | {s['acc_T1']*100:.2f}/{s['acc_T2']*100:.2f}/{s['acc_T3']*100:.2f} | {s['all_acc']*100:.2f} | {s['hit_acc']*100:.2f} | {s['T1_ttft_mean_ms']:.2f} | {s['hit_ttft_mean_ms']:.2f}/{s['hit_ttft_p50_ms']:.2f}/{s['hit_ttft_p95_ms']:.2f} | {s['SSD_MB_per_hit']:.2f} |")
    lines += ['', '| Method | Logical retention | Full-K/scoring MB | Selected-K MB | Selected-V MB | Structural/other MB | V chunk coverage | Preads/hit |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for b in budget:
        lines.append(f"| {LABELS[b['method_id']]} | {b['logical_content_retention']:.6f} | {b['full_K_scoring_MB']:.3f} | {b['selected_K_MB']:.3f} | {b['selected_V_MB']:.3f} | {b['structural_other_MB']:.3f} | {b['selected_V_chunk_coverage']:.4f} | {b['total_preads_per_hit']:.2f} |")
    lines += ['', 'AllHead는 layer별 full K를 한 번 읽고 scoring과 answer attention에 재사용한다. selected-K 추가 읽기와 probe 읽기는 0이며 재사용 K를 bytes 합계에 다시 더하지 않는다. Structural sidecar의 실제 중복 K는 별도 기록한다. MPIC-32는 full context에서 32개 이미지 token을 재계산한다.',
        '', '| Ours − baseline | Δ Hit Acc (%p), 95% CI | Δ Hit TTFT (ms), 95% CI | TTFT 감소율 (%) | SSD 감소율 (%) |','|---|---:|---:|---:|---:|']
    for p in comparisons:
        ssd='N/A (ReComp SSD=0)' if p['SSD_reduction_percent'] is None else f"{p['SSD_reduction_percent']:.2f}"
        lines.append(f"| {LABELS[p['baseline']]} | {p['delta_hit_acc_pp']:+.3f} [{p['acc_ci_low_pp']:+.3f}, {p['acc_ci_high_pp']:+.3f}] | {p['delta_hit_TTFT_ms']:+.3f} [{p['TTFT_ci_low_ms']:+.3f}, {p['TTFT_ci_high_ms']:+.3f}] | {p['TTFT_reduction_percent']:.2f} | {ssd} |")
    lines += ['', 'CI는 seed=1234, image-cluster bootstrap 10,000회이고 image의 모든 대화·turn·방법을 함께 resample한다. Point estimate와 동일한 요청 가중 평균이다. CI의 0 포함은 동등성 증거가 아니다. History 차이를 포함한 method-level 비교다.',
        '', '| Method | Provisioning | Persistence | Actual-stream session E2E (ms) | Derived standalone E2E (ms) |','|---|---|---|---:|---:|']
    for s in sessions:lines.append(f"| {LABELS[s['method_id']]} | {s['provisioning_policy']} | {s['persistence_status']} ({s['persistence_ms_per_image']:.2f} ms/image) | {s['actual_stream_session_E2E_mean_ms']:.2f} | {s['derived_standalone_E2E_mean_ms']:.2f} |")
    lines += ['', 'FullLoad와 AllHead는 실제 각자의 source T1에서 FP16 K/V bits·prefix·v_hidden의 동일성을 검증한 canonical store를 공유한다. 기존 검증된 raster serializer가 함께 만드는 미사용 probe sidecar의 쓰기 비용도 canonical persistence에 포함되며, base-only 비용은 별도 측정하지 않았다. 실제 build는 이미지당 한 번 FullLoad source dialogue에 귀속된다. AllHead의 독립 배포와 모든 dialogue의 standalone 비용은 DERIVED이다. T1 capture/exit materialization은 request E2E에 이미 들어 있으므로 다시 더하지 않는다.',
        '', 'True TTFT는 prompt 작성 전부터 JPEG 읽기/디코딩(픽셀 요청), tokenizer/processor, SSD read/H2D/assembly/prefill, 첫 token materialization과 CUDA sync까지다. 응답 decode 완료 E2E와 구분한다. DONTNEED는 타이머 밖에서 파일별 성공을 기록하며 NAND/controller cold를 보장하지 않는다. Metadata activation 및 hash validation은 별도 setup이다. AllHead host/CUDA stage 구간은 중첩되므로 합산하지 않는다.',
        '', '두 KV25는 ceil(N/4)의 실제 content attention 예산이며 full original payload를 SSD에 유지한다. Dense GPU cache를 사용하므로 75% GPU 메모리 절감 주장을 하지 않는다. 이미지별 streaming은 동시성/throughput/cache saturation 검증이 아니다. 효율 결론은 이 fixed-budget SSD adaptation과 layout/cache 조건에 한정한다.',
        '', f"독립 감사: {audit['status']}; AllHead K-read 재사용: {audit['AllHead_K_READ_REUSE']}; Qwen GPU / MT-VQA: NOT RUN.",
        f"보호 검사: {audit['protection']['status']}; 기존 source 변경 {len(audit['protection']['changed_existing_source'])}건, 기존 artifact 변경 {len(audit['protection']['changed_artifacts'])}건. 큰 기존 파일은 inode/mtime/ctime + 9-window fingerprint 정책이며 전체-byte SHA256 검증으로 과장하지 않는다.",
        '', '이미지별 N/k/read amplification, selection 변화, cap/실패/retry/중복, metadata와 GPU memory, 중첩 timing은 함께 저장된 CSV/JSON 및 image raw를 참조한다. Validation/smoke/retry는 main final counts와 별도다.']
    return '\n'.join(lines)+'\n'

def run_audit(run, output):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'independent_audit.json').exists():raise FileExistsError('use a new independent audit output directory')
    config=json.loads((run/'config.json').read_text());config.update(_file_sha256=sha(run/'config.json'),_manifest_sha256=sha(run/'manifest.json'))
    manifest=json.loads((run/'manifest.json').read_text());checks=A.Checks();small=[];image_receipts=[];sessions=[];attempt_counts=Counter();physical=set();logical=set()
    phase_counts={};memory=[];selection=[];geometry=[]
    for phase in ('integration','smoke','main'):
        count=0
        for cp in sorted((run/phase/'images').glob('*/COMMITTED.json')):
            commit=json.loads(cp.read_text());folder=cp.parent/commit['attempt_id'];raw=folder/'raw.jsonl'
            checks.require(sha(raw)==commit['raw_sha256'],'committed_raw_hash',str(cp))
            rows=list(read_rows(raw));receipt=audit_image_rows(rows,manifest,config,phase,cp.parent.name);image_receipts.append(receipt)
            checks.require(receipt['status']=='PASS','independent_image_audit',receipt);count+=len(rows)
            checks.require((folder/'cleanup_receipt.json').is_file(),'committed_cleanup_receipt',str(folder))
            checks.require(json.loads((folder/'shared_canonical_compatibility.json').read_text())['status']=='PASS','shared_store_actual_bits',str(folder))
            if phase=='main':
                for r in rows:
                    checks.require(r['logical_request_id'] not in logical and r['physical_execution_id'] not in physical,'global_unique_ids',r['request_id'])
                    logical.add(r['logical_request_id']);physical.add(r['physical_execution_id']);small.append(compact_row(r))
                for method in METHODS[1:]:
                    a=json.loads((folder/f'{method}_activation.json').read_text());memory.append(dict(image_id=cp.parent.name,method_id=method,**a))
                ours=[r for r in rows if r['method_id']=='ours_kv25' and r['turn_id']>1]
                ah=[r for r in rows if r['method_id']==ALLHEAD and r['turn_id']>1]
                signatures=[canonical([x['selected_token_ids'] for x in r['result']['layers']]) for r in ah]
                selection.append({'image_id':cp.parent.name,'ours_selection_unique_count':len({canonical(r['result']['selected_original_ids']) for r in ours}),
                    'allhead_request_selection_unique_count':len(set(signatures)),'allhead_requests':len(ah),
                    'allhead_changed_T2_T3_dialogues':sum(signatures[i]!=signatures[i+1] for i in range(0,len(signatures),2)),
                    'allhead_mean_unique_layer_selections':mean([len({canonical(x['selected_token_ids']) for x in r['result']['layers']}) for r in ah])})
                for r in rows:
                    if r['turn_id']>1 and r['method_id'] in (ALLHEAD,'ours_kv25'):
                        full=2*r['geometry']['num_layers']*r['geometry']['v_token_num']*r['geometry']['num_heads']*r['geometry']['head_dim']*2
                        geometry.append({'image_id':r['image_id'],'dialog_id':r['dialog_id'],'turn_id':r['turn_id'],'method_id':r['method_id'],
                            'N':r['N_content'],'k':r['k'],'k_over_N':r['k']/r['N_content'],'read_bytes':r['ssd_read_bytes'],
                            'full_original_KV_bytes':full,'physical_read_ratio':r['ssd_read_bytes']/full,
                            'read_amplification_vs_logical_selected_content':r['ssd_read_bytes']/(2*r['geometry']['num_layers']*r['k']*r['geometry']['num_heads']*r['geometry']['head_dim']*2)})
            del rows
        phase_counts[phase]=count
        for attempt in (run/phase/'images').glob('*/attempt_*'):
            attempt_counts[phase+'_image_attempts']+=1
            if (attempt/'failure.json').exists():attempt_counts[phase+'_failed_attempts']+=1
            if (attempt/'raw.jsonl').exists():attempt_counts[phase+'_completed_physical_rows']+=sum(1 for _ in read_rows(attempt/'raw.jsonl'))
    complete=phase_counts['main']==60915
    if complete:
        checks.require(len({r['image_id'] for r in small})==398 and len({r['dialog_id'] for r in small})==4061,'main_population')
        for m in METHODS:
            rs=[r for r in small if r['method_id']==m]
            checks.require(len(rs)==12183 and sum(r['turn_id']>1 for r in rs)==8122,'per_method_counts',m)
    checks.require(phase_counts['smoke']==60,'new_main_smoke_60')
    gate=json.loads((run/'integration_validation.json').read_text());checks.require(gate.get('status')=='PASS','integration_gate')
    protect=protection(run);checks.require(protect['status']=='PASS','artifact_protection')
    for name,digest in config['source_sha256'].items():checks.require(sha(ROOT/name)==digest,'final_source_freeze',name)
    result=checks.result();valid=complete and result['status']=='PASS'
    result.update(MAIN_5_ARM='VALID' if valid else 'INVALID' if complete else 'PARTIAL' if phase_counts['main'] else 'NOT RUN',
        main_requests=phase_counts['main'],expected_main_requests=60915,phase_counts=phase_counts,attempt_counts=dict(attempt_counts),
        protection=protect,AllHead_K_READ_REUSE='PASS' if image_receipts and all(r['status']=='PASS' for r in image_receipts) else 'UNRESOLVED',
        READY_FOR_PAPER_MAIN_TABLE='YES' if valid else 'NO',QWEN_GPU='NOT RUN',MT_VQA='NOT RUN',
        source_sha256=config['source_sha256'],image_receipts=image_receipts)
    summary,budget=summarize(small);comparisons=paired(small) if small else []
    details,sess=session_tables(run,small) if small else ([],[])
    csv_write(output/'summary.csv',summary);csv_write(output/'budget_io.csv',budget);csv_write(output/'paired_comparisons.csv',comparisons)
    csv_write(output/'session_per_dialogue.csv',details);csv_write(output/'session_summary.csv',sess)
    csv_write(output/'request_diagnostics.csv',small);csv_write(output/'image_budget_io.csv',geometry)
    csv_write(output/'selection_variation.csv',selection);csv_write(output/'metadata_activation.csv',memory)
    dump(output/'independent_audit.json',result);dump(output/'protection_final.json',protect)
    (output/'REPORT.md').write_text(render(result,summary,budget,comparisons,sess))
    dump(output/'diagnostics.json',{'output_cap_hits':sum(r['cap_reached'] for r in small),'attempts':dict(attempt_counts),
        'main_final_rows':len(small),'main_final_duplicate_logical_rows':len(small)-len(logical),
        'main_final_duplicate_physical_rows':len(small)-len(physical),'bootstrap_resamples':10000,'bootstrap_seed':1234})
    for name in ('config.json','manifest.json','storage_plan.json','environment_before.json','integration_validation.json'):
        import shutil
        if not (output/name).exists():shutil.copy2(run/name,output/name)
    print(json.dumps({k:result[k] for k in ('MAIN_5_ARM','main_requests','status','READY_FOR_PAPER_MAIN_TABLE')}),flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();run_audit(args.run_dir.resolve(),args.output_dir.resolve())
if __name__=='__main__':main()

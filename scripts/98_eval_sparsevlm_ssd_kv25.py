#!/usr/bin/env python3
"""Conditional five-arm LLaVA pilot. No GPU gate means no benchmark.

Frozen workloads only: 4x3 smoke, 40x6 independent GQA, 40x3 method-generated
MT. Existing GQA stores are read-only. MT builds one image's shared canonical
and Ours stores, commits raw/hash/independent audit, then removes only the
explicit run-owned files in that successful image's allowlist.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

def helper(filename,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/filename)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
V=helper('97_validate_sparsevlm_ssd_kv25.py','_sparsevlm_validation_helpers')
OLD=helper('89_eval_llava_kv25.py','_sparsevlm_pilot_old')
QA=OLD.QA
METHODS=('recompute','fullload',*V.METHODS,'ours_kv25')
SCHEMA='sparsevlm-ssd-kv25-pilot-v1'
SOURCE_PATHS=(*V.SOURCE_PATHS,'scripts/98_eval_sparsevlm_ssd_kv25.py',
 'scripts/49_eval_query_aware_baseline.py','scripts/89_eval_llava_kv25.py','scripts/99_audit_sparsevlm_ssd_kv25.py','mmimpress/config.py','mmimpress/dataset.py')

def canonical_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write_json(path,value):
    if Path(path).exists():raise FileExistsError(f'refuse artifact overwrite: {path}')
    V.atomic_json(Path(path),value)
def append(path,value):
    with Path(path).open('a',encoding='utf-8') as f:
        f.write(json.dumps(OLD.safe(value),ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def gpu_gate(path):
    receipt=json.loads(Path(path).read_text())
    if receipt.get('GPU_CORRECTNESS')!='PASS':raise RuntimeError('BLOCKED: GPU correctness not PASS')
    if receipt.get('contract_sha256')!=V.sha(ROOT/'docs/sparsevlm_ssd_kv25_contract.md'):raise RuntimeError('BLOCKED: stale GPU contract')
    if set(V.SOURCE_PATHS)-set(receipt.get('source_sha256',{})):raise RuntimeError('BLOCKED: GPU source hash coverage incomplete')
    if any(row.get('status')!='PASS' for row in receipt.get('samples',[])):raise RuntimeError('BLOCKED: GPU sample failure')
    for name,digest in receipt.get('source_sha256',{}).items():
        if V.sha(ROOT/name)!=digest:raise RuntimeError(f'BLOCKED: stale GPU code {name}')
    if any(receipt.get('gates',{}).get(f'G{i}')!='PASS' for i in range(1,13)):raise RuntimeError('BLOCKED: mandatory GPU gate incomplete')
    if len(receipt.get('samples',[]))!=10 or receipt.get('MT_generated_history',{}).get('status')!='PASS':raise RuntimeError('BLOCKED: GPU fixed-sample/MT evidence incomplete')
    return {'path':str(path),'sha256':V.sha(path),'verdict':'PASS'}

def text_spans(prompt,question,history,entries):
    spans=[]
    if history:
        start=prompt.index(history);spans.append({'kind':'history','start_char':start,'end_char':start+len(history),
            'provenance':'method_local_generated_QA','entries':entries})
    start=prompt.rfind(question)
    if start<0:raise ValueError('current question absent from prompt')
    spans.append({'kind':'current_question','start_char':start,'end_char':start+len(question)})
    image_end=prompt.index('<image>')+len('<image>')
    covered=sorted((x['start_char'],x['end_char']) for x in spans)
    cursor=image_end
    for lo,hi in covered:
        if cursor<lo:spans.append({'kind':'template','start_char':cursor,'end_char':lo})
        cursor=max(cursor,hi)
    if cursor<len(prompt):spans.append({'kind':'template','start_char':cursor,'end_char':len(prompt)})
    return sorted(spans,key=lambda x:x['start_char'])

def prompt_factory(runner,phase,entry,turn,predictions):
    if phase=='mt':return OLD.mt_prompt(entry,turn,predictions)
    q=entry['questions'][turn-1]['question'];return runner.prompt(q),'',[]

class PilotCounts:
    """Actual phase-aware projection calls; first model invocation is prefill."""
    def __init__(self,runner):self.runner=runner
    def __enter__(self):
        self.calls={};self.handles=[];self.model_calls=0;self.phase=None
        def begin(module,args,kwargs):
            self.phase='prefill' if self.model_calls==0 else 'decode';self.model_calls+=1
        self.handles.append(self.runner.model.register_forward_pre_hook(begin,with_kwargs=True))
        for li,layer in enumerate(self.runner.layers):
            for kind in ('q','k','v'):
                def count(mod,args,li=li,kind=kind):
                    if self.phase is None:raise AssertionError('projection before model phase')
                    counts=self.calls.setdefault(str(li),{}).setdefault(self.phase,{})
                    counts[kind]=counts.get(kind,0)+1
                self.handles.append(getattr(layer.self_attn,kind+'_proj').register_forward_pre_hook(count))
        return self
    def __exit__(self,*exc):
        for h in self.handles:h.remove()
    def result(self):return self.calls

def normal_request(runner,server,image_path,factory,capture_kind):
    from contextlib import ExitStack
    from mmimpress.piggyback import VisionForwardCapture,DecoderVisualHiddenCapture
    vision=VisionForwardCapture(runner,capture_saliency=capture_kind=='image_only')
    hidden=None
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    prompt,history,entries=factory()
    with Image.open(image_path) as im:image=im.convert('RGB');image.load()
    enc,processing=QA._combined_processor(runner,image,prompt)
    v0,vn=runner.visual_span(enc['input_ids'])
    if capture_kind=='raster':hidden=DecoderVisualHiddenCapture(runner,v0,vn)
    with vision,PilotCounts(runner) as counts,ExitStack() as stack:
        if hidden is not None:stack.enter_context(hidden)
        result=server.recompute(runner.to_device(enc),return_past_key_values=capture_kind!='none')
    returned=time.perf_counter()
    assert vision.call_count==1
    result.update(projection_calls=counts.result(),scoring_calls=0,
                  end_to_end_ttft_ms=(result['first_token_at_s']-started)*1000,
                  request_e2e_ms=(returned-started)*1000,request_started_at_s=started,
                  vision_forward_count=vision.call_count,cap_reached=(len(result['generated_token_ids'])==16 and result['generated_token_ids'][-1]!=runner.processor.tokenizer.eos_token_id),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),processor=processing)
    diag={'prompt':prompt,'history':history,'history_entries':entries,'enc_cpu':enc,'hidden':hidden,'vision':vision,
          'image_input_sha256':QA._image_input_hash(enc),'v_token_start':v0,'v_token_num':vn}
    return result,diag

def legacy_hit(runner,server,ctx,factory,method,image_id):
    with QA._NoVisionForward(runner) as guard:
        c0=time.perf_counter();ctx.reader.drop_all();conditioning=(time.perf_counter()-c0)*1000
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        prompt,history,entries=factory();tokenized=runner.processor.tokenizer(prompt,return_tensors='pt')
        suffix=QA._suffix_from_tokenized(runner,tokenized).to(runner.model.device)
        with PilotCounts(runner) as counts,V.ReadTrace() as reads:
            if method=='fullload':result=server.request(ctx,mode='fullload',suffix_ids=suffix,cold=False)
            else:result=server.request_cvpr25(ctx,static=None,budget=.25,mode='prefix',budget_unit='visual_kv',
                 sep_policy='sidecar',cold=False,seed=1234,image_id=image_id,suffix_ids=suffix,
                 expected_prefix_layout='visionzip_image_only')
        returned=time.perf_counter()
    assert sum(r['returned'] for r in reads.calls)==result['io']['bytes']
    assert len(reads.calls)==result['io']['preads']
    result.update(projection_calls=counts.result(),scoring_calls=0,
        end_to_end_ttft_ms=(result['first_token_at_s']-started)*1000,request_e2e_ms=(returned-started)*1000,
        request_started_at_s=started,vision_forward_count=guard.calls,page_cache_conditioning_ms=conditioning,
        cap_reached=(len(result['generated_token_ids'])==16 and result['generated_token_ids'][-1]!=runner.processor.tokenizer.eos_token_id),
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        os_pread_trace=reads.calls)
    # The next request must not retain any selected visual payload on the GPU.
    ctx.cache.k=None;ctx.cache.v=None
    return result,{'prompt':prompt,'history':history,'history_entries':entries,'suffix_input_ids':suffix.cpu().tolist()}

def sparse_hit(runner,server,ctx,factory,method,question):
    with QA._NoVisionForward(runner) as guard:
        c0=time.perf_counter();ctx.drop_cache();conditioning=(time.perf_counter()-c0)*1000
        torch.cuda.synchronize();started=time.perf_counter()
        prompt,history,entries=factory();spans=text_spans(prompt,question,history,entries)
        with V.ReadTrace() as reads:
            result=server.request(ctx,method_id=method,question=question,prompt_text=prompt,
                                  text_spans=spans,cold=False)
        returned=time.perf_counter()
    V.verify_reads(result,reads,ctx.meta,method)
    result.update(core_ttft_ms=result['end_to_end_ttft_ms'],end_to_end_ttft_ms=(result['first_token_at_s']-started)*1000,
       request_e2e_ms=(returned-started)*1000,request_started_at_s=started,vision_forward_count=guard.calls,
       page_cache_conditioning_ms=conditioning,os_pread_trace=reads.calls)
    return result,{'prompt':prompt,'history':history,'history_entries':entries,'text_spans':spans}

def persist(runner,result,diag,path,image_id,kind):
    from mmimpress.piggyback import persist_captured_raster_prefix,persist_captured_visual_prefix
    enc=diag['enc_cpu'];args=(runner,result['captured_past_key_values'],enc['input_ids'],enc['image_sizes'][0])
    common={'image_id':image_id,'model_id':V.MODEL_ID,'chunk_size':64,'image_input_sha256':diag['image_input_sha256'],
      'extra_metadata':{'dataset':'mt_gqa_reconstructed','source_turn_id':1,'source_method_key':'fullload' if kind=='raster' else 'ours_kv25',
                        'model_revision':V.REVISION,'run_schema':SCHEMA},'full_integrity_hash':True}
    if kind=='raster':
        return persist_captured_raster_prefix(*args,diag['hidden'].result_cpu(),path,probe_heads=3,
                hidden_capture_stats=diag['hidden'],**common)
    return persist_captured_visual_prefix(*args,diag['vision'].result_cpu(),path,capture_stats=diag['vision'],**common)

def file_hashes(path):
    result={}
    for p in sorted(Path(path).rglob('*')):
        if p.is_symlink():raise ValueError('store has unexpected symlink')
        if p.is_file():result[p.relative_to(path).as_posix()]=V.sha(p)
    return result

def expected_protected(run_dir,store):
    index=Path(run_dir)/'protected_artifacts_before.jsonl'
    if not index.exists():raise RuntimeError('BLOCKED: pre-run protected artifact hash inventory missing')
    prefix=str(Path(store).resolve())+'/'
    selected={}
    with index.open() as f:
        for line in f:
            row=json.loads(line);p=str(row.get('path',''))
            if not p.startswith('/'):p=str((ROOT/p).resolve())
            if p.startswith(prefix) and row.get('sha256'):selected[p[len(prefix):]]=row['sha256']
    if not selected:raise RuntimeError(f'BLOCKED: no frozen payload hashes for {store}')
    return selected

def verify_store_hashes(path,expected):
    for relative,digest in expected.items():
        if V.sha(Path(path)/relative)!=digest:raise RuntimeError(f'protected store hash mismatch: {path}/{relative}')

def geometry(meta):
    return {key:meta.get(key,[]) if key=='padding_idx' else meta[key] for key in
       ('num_heads','head_dim','num_layers','v_token_start','v_token_num','n_spatial','prefix_len','dtype','newline_idx','padding_idx')}

def row_for(result,diag,meta,method,phase,entry,question,turn,run_id,attempt,config_hash,manifest_hash,code_hash,store):
    gold=question.get('answers') or question.get('gold') or question.get('answer')
    if not isinstance(gold,list):gold=[gold]
    if not gold or gold[0] is None:raise ValueError('missing frozen scoring target')
    from mmimpress.dataset import METRICS,question_answers
    if phase=='mt':score=OLD.MT.strict_gqa_score(result['answer'],gold[0])
    else:gold=question_answers(question);score=METRICS['gqa'](result['answer'],gold)
    n=int(meta['n_spatial']);hit=turn>1
    k=(n+3)//4 if hit and method in (*V.METHODS,'ours_kv25') else n
    result={key:value for key,value in result.items() if key not in ('captured_past_key_values','logits','first_logits')}
    result.setdefault('cap_reached',len(result['generated_token_ids'])==16)
    result.setdefault('generated_token_count',len(result['generated_token_ids']))
    return {'schema_version':SCHEMA,'experiment_id':run_id,'attempt_id':attempt,'phase':phase,'method_id':method,
       'request_id':f"{phase}:{entry.get('dialog_id',entry['image_id'])}:{turn}:{method}",
       'image_id':entry['image_id'],'dialog_id':entry.get('dialog_id'),'question_id':str(question['question_id']),'turn_id':turn,
       'question':question['question'],'gold':gold,'prediction':result['answer'],'score':float(score),
       'prompt':diag['prompt'],'history':diag['history'],'history_entries':diag['history_entries'],
       'config_sha256':config_hash,'manifest_sha256':manifest_hash,'code_sha256':code_hash,
       'source_T1':{'method_id':method if turn==1 or method=='recompute' else ('qa_select25' if phase!='mt' and method in (*V.METHODS,'fullload') else 'ours25' if phase!='mt' else 'fullload' if method in (*V.METHODS,'fullload') else method),'turn_id':1,
                    'provenance':'this_request_normal_full_image' if turn==1 or method=='recompute' else 'preexisting_same_T1_piggyback' if phase!='mt' else 'this_run_normal_full_image'},
       'provisioning_mode':'CACHE-HIT REEVALUATION' if phase!='mt' else 'FRESH_T1',
       'persistence_status':'NOT_REMEASURED' if phase!='mt' else 'MEASURED_SHARED_IMAGE_STORE_BUNDLE',
       'AllHead_base_build_cost':'NOT_REMEASURED' if phase!='mt' else 'NOT_SEPARATELY_MEASURED',
       'Probe3_sidecar_build_cost':'NOT_REMEASURED' if phase!='mt' else 'NOT_SEPARATELY_MEASURED',
       'request_started_at_s':result['request_started_at_s'],'first_token_at_s':result['first_token_at_s'],
       'request_finished_at_s':result['request_started_at_s']+result['request_e2e_ms']/1000,
       'ttft_ms':result['end_to_end_ttft_ms'],'request_e2e_ms':result['request_e2e_ms'],
       'ssd_read_bytes':int(result.get('io',{}).get('bytes',0)),'ssd_preads':int(result.get('io',{}).get('preads',0)),
       'N_content':n,'k':k,'content_kv_fraction':k/n,
       'scoring_head_count':(3 if method==V.METHODS[0] else int(meta['num_heads']) if method==V.METHODS[1] else 0) if hit else 0,
       'geometry':geometry(meta),'source_store_meta_sha256':V.sha(Path(store)/'meta.json'),'source_store_path':str(store),
       'store_footprint_bytes':sum(p.stat().st_size for p in Path(store).rglob('*') if p.is_file() and not (method==V.METHODS[1] and p.name=='probe_k.bin')),
       'store_footprint_scope':'independent_deployment_required_files_except_unneeded_AllHead_probe',
       'os_pread_trace':result.get('os_pread_trace',[]),'result':OLD.safe(result)}

def verify_image_rows(rows,phase,turns):
    assert len(rows)==len(METHODS)*turns
    assert len({r['request_id'] for r in rows})==len(rows)
    for m in METHODS:
        mr=[r for r in rows if r['method_id']==m]
        assert sorted(r['turn_id'] for r in mr)==list(range(1,turns+1))
        for r in mr:
            assert r['ttft_ms']>0 and r['request_e2e_ms']>=r['ttft_ms']
            if r['turn_id']==1 or m=='recompute':assert r['result']['vision_forward_count']==1
            else:assert r['result']['vision_forward_count']==0
            if phase=='mt':
                assert len(r['history_entries'])==r['turn_id']-1
                for h in r['history_entries']:
                    prev=next(x for x in mr if x['turn_id']==h['turn_id'])
                    assert h['answer']==prev['prediction'] and h['answer_source']=='method_local_generated'
            else:assert not r['history_entries']
    return {'status':'PASS','requests':len(rows)}

def cleanup_committed(scratch,allowlist,receipt):
    if receipt.get('status')!='PASS':raise ValueError('independent image audit required before cleanup')
    scratch=Path(scratch).resolve()
    for relative in allowlist:
        p=scratch/relative
        if p.is_symlink() or p.stat().st_nlink!=1 or not p.resolve().is_relative_to(scratch):
            raise ValueError('cleanup refuses linked or non-owned payload')
    for relative in allowlist:(scratch/relative).unlink()
    for p in sorted(scratch.rglob('*'),key=lambda x:len(x.parts),reverse=True):
        if p.is_dir():p.rmdir()
    scratch.rmdir()

def run_image(runner,old_server,new_server,phase,entry,index,run_dir,results_dir,config,manifest,hashes):
    from mmimpress.serve import ImageContext
    from mmimpress.sparsevlm_ssd_store import CanonicalContext
    from mmimpress.piggyback import deterministic_method_rotation
    phase_dir=run_dir/phase;image_id=entry['image_id'];dialog_id=entry.get('dialog_id',image_id)
    image_parent=phase_dir/'images'/dialog_id
    attempt=f'attempt_{len(list(image_parent.glob("attempt_*")))+1:04d}'
    image_artifact=image_parent/attempt;image_artifact.mkdir(parents=True,exist_ok=False)
    questions=entry['turns'] if phase=='mt' else entry['questions'];turns=len(questions)
    scratch=run_dir/'scratch'/phase/dialog_id/attempt
    paths={'raster':scratch/'raster','image_only':scratch/'image_only'} if phase=='mt' else {
           kind:V.STORE_ROOT/kind/image_id for kind in ('raster','image_only')}
    if phase=='mt':
        plan=json.loads((run_dir/'storage_plan.json').read_text());free=shutil.disk_usage(run_dir).free
        if free < int(plan['minimum_reserve_bytes'])+int(plan['peak_working_set_bytes']):raise RuntimeError('BLOCKED_STORAGE')
        owned_payload=sum(p.stat().st_size for p in run_dir.rglob('*.bin') if p.is_file() and not p.is_symlink())
        if owned_payload+(3<<30)>int(plan['maximum_new_payload_bytes']):raise RuntimeError('BLOCKED_STORAGE: failed or retained run payload exceeds scratch allowance')
    metas={};expected={};persistence={};contexts={};rows=[];predictions={m:{} for m in METHODS}
    order=METHODS[index%len(METHODS):]+METHODS[:index%len(METHODS)]
    image_path=Path(entry['image_path']);image_path=image_path if image_path.is_absolute() else ROOT/image_path
    source_image_hash=V.sha(image_path)
    success=False
    try:
        if phase!='mt':
            for kind,path in paths.items():
                expected[kind]=expected_protected(run_dir,path);verify_store_hashes(path,expected[kind]);metas[kind]=json.loads((path/'meta.json').read_text())
        for turn,question in enumerate(questions,1):
            OLD.assert_gpu_exclusive()
            for order_index,method in enumerate(order):
                factory=lambda m=method:prompt_factory(runner,phase,entry,turn,predictions[m])
                if turn==1 or method=='recompute':
                    capture=('raster' if method=='fullload' else 'image_only' if method=='ours_kv25' else 'none') if phase=='mt' and turn==1 else 'none'
                    result,diag=normal_request(runner,old_server,image_path,factory,capture)
                    append(image_artifact/'attempt_events.jsonl',{'method_id':method,'turn_id':turn,'question_id':str(question['question_id']),
                        'result':{k:v for k,v in result.items() if k!='captured_past_key_values'},
                        'prompt':diag['prompt'],'history_entries':diag['history_entries'],'status':'executed_before_persistence_or_adoption'})
                    if turn==1 and phase=='mt' and capture!='none':
                        persistence[capture]=persist(runner,result,diag,paths[capture],image_id,capture)
                        metas[capture]=json.loads((paths[capture]/'meta.json').read_text());expected[capture]=file_hashes(paths[capture])
                        write_json(image_artifact/f'{capture}_hashes.json',expected[capture])
                        write_json(image_artifact/f'{capture}_persistence.json',persistence[capture])
                    if turn==1 and phase!='mt':
                        for meta in metas.values():
                            assert meta.get('image_input_sha256')==diag['image_input_sha256'],'RO image preprocessing mismatch'
                            assert diag['enc_cpu']['input_ids'][0,:meta['prefix_len']].tolist()==meta['prefix_input_ids'],'RO logical prefix mismatch'
                    # T1 rows wait until all methods have produced both MT stores.
                    if phase=='mt' and turn==1:
                        saved={k:v for k,v in diag.items() if k not in ('enc_cpu','hidden','vision')}
                        result.pop('captured_past_key_values',None)
                        pending=(method,result,saved,question,turn,order_index)
                        contexts.setdefault('_pending',[]).append(pending)
                        predictions[method][turn]=result['answer'];continue
                else:
                    kind='image_only' if method=='ours_kv25' else 'raster'
                    if method not in contexts:
                        activate=time.perf_counter()
                        if method in V.METHODS:
                            required={'meta.json','sys_kv.pt','v_hidden.pt','sep_kv.bin'}
                            for li in range(metas[kind]['num_layers']):
                                required.update(f'layer_{li:02d}/{name}.bin' for name in (('k','v','probe_k') if method==V.METHODS[0] else ('k','v')))
                            selected_hashes={k:expected[kind][k] for k in required}
                            ctx=CanonicalContext(paths[kind],head_policy=V.POLICIES[method],expected_hashes=selected_hashes)
                        else:
                            ctx=ImageContext(paths[kind],runner.model.device,drop_cache=False,require_v_hidden=False)
                            ctx.activation={'host_tensor_bytes':sum(t.numel()*t.element_size() for t in ctx.cache.sys_kv.values()),
                                'gpu_tensor_bytes':0,'visual_payload_resident_bytes':0,
                                'file_bytes':sum((paths[kind]/name).stat().st_size for name in ('meta.json','sys_kv.pt'))}
                        contexts[method]=ctx
                        write_json(image_artifact/f'{method}_activation.json',{'seconds':time.perf_counter()-activate,
                           'metadata':getattr(ctx,'activation',{}),'persistence':'NOT_REMEASURED' if phase!='mt' else 'MEASURED',
                           'visual_payload_resident_bytes':0})
                    if method in V.METHODS:result,diag=sparse_hit(runner,new_server,contexts[method],factory,method,question['question'])
                    else:result,diag=legacy_hit(runner,old_server,contexts[method],factory,method,image_id)
                    result['metadata_activation']=contexts[method].activation
                    append(image_artifact/'attempt_events.jsonl',{'method_id':method,'turn_id':turn,'question_id':str(question['question_id']),
                        'result':result,'prompt':diag['prompt'],'history_entries':diag['history_entries'],'status':'executed_before_adoption'})
                kind='image_only' if method=='ours_kv25' else 'raster'
                row=row_for(result,diag,metas[kind],method,phase,entry,question,turn,run_dir.name,attempt,
                    hashes['config'],hashes['manifest'],hashes['code'],paths[kind]);row.update(method_order=list(order),method_order_position=order_index,image_sha256=source_image_hash)
                append(image_artifact/'raw.jsonl',row);rows.append(row);predictions[method][turn]=result['answer']
                del result,diag
            if phase=='mt' and turn==1:
                for method,result,diag,q,t,oi in contexts.pop('_pending'):
                    kind='image_only' if method=='ours_kv25' else 'raster'
                    row=row_for(result,diag,metas[kind],method,phase,entry,q,t,run_dir.name,attempt,
                      hashes['config'],hashes['manifest'],hashes['code'],paths[kind]);row.update(method_order=list(order),method_order_position=oi,image_sha256=source_image_hash)
                    append(image_artifact/'raw.jsonl',row);rows.append(row)
                del result,diag
        validation=verify_image_rows(rows,phase,turns);write_json(image_artifact/'runner_validation.json',validation)
        # Independent module re-reads records, recomputes quality/history/IO and
        # issues the durable receipt before any run-owned MT payload is removed.
        audit=helper('99_audit_sparsevlm_ssd_kv25.py','_sparsevlm_independent_audit')
        audit_config=dict(config,_file_sha256=hashes['config'],_manifest_sha256=hashes['manifest'])
        durable_rows=[json.loads(line) for line in (image_artifact/'raw.jsonl').read_text().splitlines() if line.strip()]
        independent=audit.audit_image_rows(durable_rows,manifest,audit_config)
        independent['raw_sha256']=V.sha(image_artifact/'raw.jsonl')
        write_json(image_artifact/'independent_audit.json',independent)
        if independent.get('status')!='PASS':raise RuntimeError('independent per-image audit failed')
        adopted=set()
        with (phase_dir/'raw.jsonl').open() as existing:
            for line in existing:adopted.add(json.loads(line)['request_id'])
        for row in rows:
            if row['request_id'] in adopted:raise RuntimeError('duplicate logical request adoption refused')
            append(phase_dir/'raw.jsonl',row);adopted.add(row['request_id'])
        success=True
    finally:
        for method,ctx in contexts.items():
            if method!='_pending':ctx.close()
        contexts.clear()
        if success and phase=='mt':
            allowlist=sorted(p.relative_to(scratch).as_posix() for p in scratch.rglob('*') if p.is_file())
            write_json(image_artifact/'cleanup_allowlist.json',{'scratch':str(scratch),'files':allowlist,'file_sha256':{p:V.sha(scratch/p) for p in allowlist}})
            cleanup_committed(scratch,allowlist,independent)
            write_json(image_artifact/'cleanup_receipt.json',{'status':'PASS','removed_run_owned_files':len(allowlist)})
    return rows

def recover_committed_images(phase_dir):
    rows=[json.loads(line) for line in (phase_dir/'raw.jsonl').read_text().splitlines() if line.strip()]
    by_id={r['request_id']:r for r in rows}
    if len(by_id)!=len(rows):raise RuntimeError('duplicate final raw rows before resume')
    completed=set()
    for image in sorted((phase_dir/'images').glob('*')):
        for attempt_dir in sorted(image.glob('attempt_*')):
            receipt=attempt_dir/'independent_audit.json'
            if not receipt.exists() or json.loads(receipt.read_text()).get('status')!='PASS':continue
            committed=[json.loads(line) for line in (attempt_dir/'raw.jsonl').read_text().splitlines() if line.strip()]
            for row in committed:
                if row['request_id'] in by_id:
                    if by_id[row['request_id']]!=row:raise RuntimeError('resume raw differs from first successful attempt')
                else:append(phase_dir/'raw.jsonl',row);by_id[row['request_id']]=row
            completed.add(image.name);break
    known={r.get('dialog_id') or r['image_id'] for r in by_id.values()}
    if not known<=completed:raise RuntimeError('raw has rows without durable independent image receipt')
    return completed

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--results-dir',type=Path,required=True)
    parser.add_argument('--gpu-gate',type=Path,required=True);parser.add_argument('--phase',choices=('smoke','gqa','mt','all'),default='smoke');parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();run_dir=args.run_dir.resolve();results_dir=args.results_dir.resolve()
    run_dir.mkdir(parents=True,exist_ok=True);results_dir.mkdir(parents=True,exist_ok=True)
    gqa,mt,workload=V.frozen_workloads();gate=gpu_gate(args.gpu_gate)
    phases=['smoke','gqa','mt'] if args.phase=='all' else [args.phase]
    config_path=run_dir/'runner_config.json';manifest_path=run_dir/'manifest.json'
    source={name:V.sha(ROOT/name) for name in SOURCE_PATHS}
    config={'schema_version':SCHEMA,'methods':METHODS,'seed':1234,'batch_size':1,'max_new_tokens':16,
      'model_id':V.MODEL_ID,'model_revision':V.REVISION,'source_sha256':source,'gpu_gate':gate,
      'contract_sha256':V.sha(ROOT/'docs/sparsevlm_ssd_kv25_contract.md'),'attempt_id':'attempt_0001',
      'attention_backend':'eager','weight_dtype':'NF4','compute_dtype':'bfloat16','SSD_dtype':'float16','chunk_size':64,
      'setup_cost_policy':'GQA persistence NOT_REMEASURED; MT shared canonical+probe bundle measured; AllHead base and Probe3-only costs NOT_SEPARATELY_MEASURED',
      'T1_policy':'each_method_own_full_image_inference','history_policy':'MT_method_own_generated;GQA_independent',
      'timing_boundary':'before prompt/image read/tokenization to first-token materialization and CUDA synchronization'}
    manifest={'schema_version':SCHEMA,'methods':METHODS,'frozen_source':workload,'workloads':{'gqa':gqa,'mt':mt},
      'expected_requests':{'smoke':60,'gqa':1200,'mt':600},'smoke_image_ids':[x['image_id'] for x in gqa[:4]],
      'smoke_question_count':3,'seed':1234}
    if config_path.exists():
        previous=json.loads(config_path.read_text())
        if previous!=json.loads(json.dumps(config)):raise RuntimeError('existing runner config differs; new run required')
    else:write_json(config_path,config)
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())!=json.loads(json.dumps(manifest)):raise RuntimeError('existing frozen manifest differs')
    else:write_json(manifest_path,manifest)
    hashes={'config':V.sha(config_path),'manifest':V.sha(manifest_path),'code':canonical_hash(source)}
    if args.preflight_only:return
    from mmimpress.serve import Server
    from mmimpress.sparsevlm_ssd_attention import SparseVLMSSDServer
    OLD.gpu_inventory();runner=V.load_runner();old=Server(runner,max_new_tokens=16);new=SparseVLMSSDServer(runner,max_new_tokens=16)
    QA._warmup(runner,old)
    for phase in phases:
        if phase!='smoke':
            smoke=json.loads((run_dir/'smoke'/'validation.json').read_text())
            if smoke.get('status')!='PASS':raise RuntimeError('BLOCKED: smoke validation missing')
        phase_dir=run_dir/phase
        if phase_dir.exists() and not args.resume:raise FileExistsError('partial phase preserved; use --resume')
        if not phase_dir.exists():phase_dir.mkdir();(phase_dir/'raw.jsonl').touch(exist_ok=False)
        elif (phase_dir/'validation.json').exists():
            if json.loads((phase_dir/'validation.json').read_text()).get('status')=='PASS':continue
            raise RuntimeError('existing invalid final validation requires a new run')
        entries=([dict(e,questions=e['questions'][:3]) for e in gqa[:4]] if phase=='smoke' else gqa if phase=='gqa' else mt)
        completed=recover_committed_images(phase_dir) if args.resume else set()
        executed=sum(1 for line in (phase_dir/'raw.jsonl').read_text().splitlines() if line.strip())
        try:
            for index,entry in enumerate(entries):
                if entry.get('dialog_id',entry['image_id']) in completed:continue
                rows=run_image(runner,old,new,phase,entry,index,run_dir,results_dir,config,manifest,hashes);executed+=len(rows)
                print(json.dumps({'phase':phase,'image_index':index,'requests':executed}),flush=True)
            if executed!=manifest['expected_requests'][phase]:raise AssertionError('request count mismatch')
            write_json(phase_dir/'validation.json',{'status':'PASS','requests':executed,'expected_requests':manifest['expected_requests'][phase],
                     'timing_valid':True,'GPU_GATE':gate,'independent_per_image_audit':'PASS'})
        except Exception as exc:
            write_json(phase_dir/f'failure_{len(list(phase_dir.glob("failure_*.json")))+1:04d}.json',{'status':'BLOCKED_STORAGE' if 'BLOCKED_STORAGE' in str(exc) else 'FAIL',
                       'error':repr(exc),'traceback':traceback.format_exc(),'completed_requests':executed});raise
if __name__=='__main__':main()

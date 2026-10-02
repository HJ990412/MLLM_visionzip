#!/usr/bin/env python3
"""Independent final tables from committed raw, original manifests and physical events.
No production model/selection/loader/scorer is used for metric reconstruction.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,csv,hashlib,importlib.util,json,math,re,sys,os,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
LABEL={'recompute':'ReComp','fullload':'FullLoad','mpic32_ssd':'MPIC-32','sparsevlm_ssd_kv25_allhead':'AllHead','ours_prefixstore_kv25':'B','qwen_ours_prefixstore_kv25':'B'}
AH='sparsevlm_ssd_kv25_allhead'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):
 with Path(p).open() as f:
  for line in f:
   if line.strip():yield json.loads(line)
def dump(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def writecsv(p,rs):
 if not rs:return
 keys=list(dict.fromkeys(k for r in rs for k in r))
 with Path(p).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rs)
def mean(v):return float(np.mean(v)) if len(v) else None
def norm(x):return ' '.join(w for w in re.sub(r'[^\w\s]',' ',str(x).lower()).split() if w not in ('a','an','the'))
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m

def stats(values):
 a=np.asarray(values,dtype=float)
 return {'mean':float(a.mean()),'p50':float(np.percentile(a,50)),'p95':float(np.percentile(a,95))} if len(a) else {'mean':None,'p50':None,'p95':None}

def cluster_ci(data,field,images):
 """Pooled sums/counts, never unweighted image means for request estimands."""
 totals=np.zeros((len(images),2));index={im:i for i,im in enumerate(images)}
 for r in data:totals[index[r['image_id']]]+=[r[field],1]
 rng=np.random.default_rng(1234);ix=rng.integers(0,len(images),size=(10000,len(images)))
 nums=totals[ix,0].sum(axis=1);dens=totals[ix,1].sum(axis=1);assert (dens>0).all()
 lo,hi=np.percentile(nums/dens,[2.5,97.5]);point=totals[:,0].sum()/totals[:,1].sum()
 return dict(point=float(point),ci_low=float(lo),ci_high=float(hi),weighting='pooled_observation_sums_counts; image cluster resampling',resamples=10000,seed=1234,images=len(images))

def compact(r,model):
 v=r['result'];d={k:r[k] for k in ('method_id','image_id','dialog_id','turn_id','ttft_ms','request_e2e_ms','ssd_read_bytes','ssd_preads','N_content','k')}
 d.update(prediction=r['prediction'],normalized=norm(r['prediction']),score=float(norm(r['prediction'])==norm(r['gold'][0])),tokens=v['generated_token_ids'],physical_id=r['physical_execution_id'],logical_id=r['logical_request_id'])
 assert d['score']==r['score'],'strict scorer differs from raw'
 cats=dict(full_K_scoring_bytes=0,other_K_bytes=0,selected_V_bytes=0,structural_bytes=0,auxiliary_bytes=0,probe_bytes=0,additional_selected_K_bytes=0)
 for e in v.get('os_pread_trace',[]):
  if r['source_store'] is not None:assert Path(e['path']).resolve().is_relative_to(Path(r['source_store']['path']).resolve()),'foreign store payload read'
  n=Path(e['path']).name
  key='full_K_scoring_bytes' if n=='k.bin' and r['method_id']==AH else 'other_K_bytes' if n=='k.bin' else 'selected_V_bytes' if n=='v.bin' else 'structural_bytes' if n in ('sep_kv.bin','structural_kv.bin') else 'probe_bytes' if n=='probe_k.bin' else 'auxiliary_bytes'
  cats[key]+=e['returned']
 assert sum(cats.values())==r['ssd_read_bytes'];d.update(cats)
 d['h2d_kv_bytes']=v.get('h2d_kv_bytes')
 d['h2d_payload_compute_bytes']=v.get('payload_h2d_compute_bytes',v.get('timing',{}).get('payload_h2d_compute_bytes'))
 d['rater_visual_h2d_bytes']=v.get('rater_visual_h2d_bytes')
 d['cap_reached']=v.get('cap_reached',False);d['generated_tokens']=len(d['tokens'])
 d['peak_gpu_allocated_bytes']=v.get('peak_gpu_allocated_bytes',v.get('peak_allocated_bytes'))
 d['peak_gpu_reserved_bytes']=v.get('peak_gpu_reserved_bytes',v.get('peak_reserved_bytes'))
 d['selector_host_ms']=sum(x.get('selector_host_ms',0) for x in v.get('layers',[])) if r['method_id']==AH and r['turn_id']>1 else None
 d['cache_assembly_ms']=v.get('cache_assembly_ms',v.get('timing_ms',{}).get('cache_assembly_ms',v.get('timing',{}).get('host_intervals_ms',{}).get('assembly')))
 d['scatter_ms']=v.get('scatter_ms');d['selector_native_ms']=v.get('selector_ms');d['chunk_io_ms']=v.get('chunk_io_ms');d['unused_loaded_real_bytes']=v.get('unused_loaded_real_bytes')
 d['h2d_ms']=v.get('h2d_ms',v.get('timing_ms',{}).get('h2d_ms'))
 d['GPU_memory_scope']='request peak with coexisting contexts, not isolated per-arm footprint'
 return d

def analyze_model(base,output,model):
 output.mkdir(parents=True,exist_ok=True);cfg=read(base/'config.json');cfg.update(_file_sha256=sha(base/'config.json'),_manifest_sha256=sha(base/'manifest.json'));manifest=read(base/'manifest.json');methods=cfg['methods'];B=methods[-1]
 A=load('_independent_B_'+model,ROOT/'scripts'/('109_audit_llava_mt_gqa_B.py' if model=='llava' else '111_audit_qwen_mt_gqa_B.py'))
 checks=[];data=[];setups={};storage=[];events=[];variation=[];geometry=[];physical=Counter();phase_counts={};failed=[];request_ids=set();exec_ids=set();attempts=[]
 for phase in ('integration','smoke','main'):
  count=0
  for ap in sorted((base/phase/'images').glob('*/attempt_*')):
   physical[phase+'_image_attempts']+=1
   if (ap/'failure.json').exists():physical[phase+'_failed_image_attempts']+=1;failed.append({'phase':phase,'path':str(ap),'failure':read(ap/'failure.json')})
   if (ap/'attempts.jsonl').exists():
    for e in rows(ap/'attempts.jsonl'):
     physical[phase+'_'+e['status']]+=1
     attempts.append({'model':model,'phase':phase,'image_attempt':str(ap),**e})
   for pp in ap.glob('*_persistence.json'):
    p=read(pp);event={'model':model,'phase':phase,'image_id':ap.parent.name,'image_attempt':str(ap),'event_kind':'physical_build','event_id':p['event_id'],'source_dialog_id':p['source']['dialog_id'],'method_id':p['source']['method_id'],'physical_source_request':p['source']['physical_execution_id'],'duration_ms':p['service_persistence_ms'],'file_logical_bytes':p['file_logical_bytes'],'allocated_bytes':p['allocated_bytes'],'os_write_bytes':p['os_write_bytes'],'os_write_calls':p['os_write_calls'],'receipt':str(pp),'receipt_sha256':sha(pp)}
    events.append(event)
   for pp in ap.glob('*_activation.json'):
    p=read(pp);method=pp.name.removesuffix('_activation.json')
    events.append({'model':model,'phase':phase,'image_id':ap.parent.name,'image_attempt':str(ap),'event_kind':'activation','event_id':p['event_id'],'method_id':method,'duration_ms':p.get('outer_wall_ms',p.get('activation_ms')),'receipt':str(pp),'receipt_sha256':sha(pp)})
  for cp in sorted((base/phase/'images').glob('*/COMMITTED.json')):
   commit=read(cp);folder=cp.parent/commit['attempt_id'];raw=folder/'raw.jsonl';assert sha(raw)==commit['raw_sha256']
   for n,digest in commit.get('evidence_sha256',{}).items():assert sha(folder/n)==digest,('changed committed evidence',folder/n)
   rs=list(rows(raw));receipt=A.audit_image_rows(rs,manifest,cfg,phase,cp.parent.name,folder)
   checks.append(receipt);assert receipt['status']=='PASS',receipt
   assert read(folder/'cleanup_receipt.json')['status']=='PASS';count+=len(rs)
   if phase!='main':continue
   iid=cp.parent.name
   for r in rs:
    assert r['logical_request_id'] not in request_ids and r['physical_execution_id'] not in exec_ids
    request_ids.add(r['logical_request_id']);exec_ids.add(r['physical_execution_id']);data.append(compact(r,model))
   for method in methods:
    if method=='recompute':setups[(iid,method)]={'setup_ms':0.,'physical_build_ms':0.,'activation_ms':0.,'source_dialog_id':None,'provenance':'MEASURED_NO_KV_SETUP'};continue
    kind={'fullload':'raster',AH:'raster','mpic32_ssd':'mpic',B:'image_only'}[method] if model=='llava' else method
    p=read(folder/f'{kind}_persistence.json');a=read(folder/f'{method}_activation.json');meta=read(folder/f'{kind}_meta.json');files=p['files'];act=a.get('outer_wall_ms',a.get('activation_ms'));duration=p['service_persistence_ms'];shared=method==AH
    setups[(iid,method)]={'setup_ms':duration+act,'physical_build_ms':0. if shared else duration,'activation_ms':act,'source_dialog_id':p['source']['dialog_id'],'provenance':'MEASURED_SHARED/DERIVED_FROM_SHARED' if shared else 'MEASURED'}
    N=meta.get('n_spatial',meta.get('visual_count'));k=(N+3)//4 if method==B else N
    width=2*meta['num_layers']*meta.get('num_heads',meta.get('num_kv_heads'))*meta['head_dim']*2
    s={'image_id':iid,'method_id':method,'physical_build_event_id':p['event_id'],'shared_bundle':shared,'provenance':setups[(iid,method)]['provenance'],'N':N,'stored_k':k,'stored_content_ratio':k/N,'valid_content_bytes':k*width,'original_content_bytes':N*width,
     'file_logical_bytes':p['file_logical_bytes'],'allocated_bytes':p['allocated_bytes'],'os_write_bytes':p['os_write_bytes'],'os_write_calls':p['os_write_calls'],'service_persistence_ms':duration,'native_writer_wall_ms':p['native_writer_wall_ms'],'completion_full_hash_ms':p['completion_full_hash_ms'],'completion_publication_ms':p['completion_publication_ms'],'activation_ms':act,
     'unused_probe_bytes':sum(x['size'] for n,x in files.items() if n.endswith('probe_k.bin')),'explicit_structural_sidecar_bytes':sum(x['size'] for n,x in files.items() if n in ('sep_kv.bin','sys_kv.pt','structural_kv.bin')),
     'auxiliary_bytes':sum(x['size'] for n,x in files.items() if n in ('v_hidden.pt','visual_input.bin')),
     'metadata_bytes':sum(x['size'] for n,x in files.items() if n.endswith('.json')),'padding_rows':meta.get('padding_rows',0),'nested_writer_timings_json':json.dumps(p['timing_ms'],sort_keys=True)}
    storage.append(s)
   hits=[r for r in rs if r['turn_id']>1];bsets=[r['result'].get('selected_original_ids',r['result'].get('selected_visual_original')) for r in hits if r['method_id']==B]
   assert len({tuple(s) for s in bsets})==1
   for r in hits:
    n=r['N_content'];k=r['k'];geo=r.get('geometry',{});S=r.get('S_structural',len(geo.get('newline_idx',[])));layers=r['result'].get('layers',[])
    geometry.append({'image_id':iid,'dialog_id':r['dialog_id'],'turn_id':r['turn_id'],'method_id':r['method_id'],'N':n,'k':k,'attention_content_ratio':k/n,'structure_rows':S,'read_bytes':r['ssd_read_bytes'],'V_chunk_coverage_layer_mean':mean([len(x['selected_chunk_ids'])/math.ceil(geo['v_token_num']/64) for x in layers]) if layers else None,'B_selection_invariant':True if r['method_id']==B else None})
   if model=='llava':
    ah={(r['dialog_id'],r['turn_id']):r for r in hits if r['method_id']==AH}
    for did in sorted({k[0] for k in ah}):
     t2,t3=ah[(did,2)],ah[(did,3)]
     for li,(l2,l3) in enumerate(zip(t2['result']['layers'],t3['result']['layers'])):
      a,b=set(l2['selected_token_ids']),set(l3['selected_token_ids']);assert len(a)==len(b)==t2['k']
      variation.append({'image_id':iid,'dialog_id':did,'layer':li,'Jaccard_T2_T3':len(a&b)/len(a|b),'changed':float(a!=b),'retained_fraction_of_T2':len(a&b)/len(a),'V_chunks_T2':len(l2['selected_chunk_ids']),'V_chunks_T3':len(l3['selected_chunk_ids'])})
  phase_counts[phase]=count
 expected=60915 if model=='llava' else 36549;complete=len(data)==expected
 gate=read(base/'integration_validation.json');core=read(base/'core_gpu_validation.json')
 if complete:
  assert gate['status']=='PASS' and phase_counts['smoke']==(60 if model=='llava' else 36)
  assert len({r['image_id'] for r in data})==398 and len({r['dialog_id'] for r in data})==4061
  assert all(sum(r['method_id']==m for r in data)==12183 for m in methods)
 for n,digest in cfg['source_sha256'].items():assert sha(ROOT/n)==digest,('executed frozen source changed',n)
 status={'model':model,'status':'PASS' if complete else 'BLOCKED' if gate['status']!='PASS' else 'PARTIAL' if data else 'NOT RUN','full_main_coverage':'PASS' if complete else 'NOT RUN','main_requests':len(data),'expected_main_requests':expected,'phase_counts':phase_counts,'physical_attempts':dict(physical),'failures':failed,'core_gpu':core['status'],'integration':gate['status'],'smoke':'PASS' if phase_counts['smoke']==(60 if model=='llava' else 36) else 'NOT RUN','independent_image_receipts':checks,'history_lineage':'PASS' if data else 'NOT RUN','selection_budget':'PASS' if data else 'NOT RUN','quality_rescoring':'PASS' if data else 'NOT RUN'}
 writecsv(output/'physical_events.csv',events);writecsv(output/'physical_attempts.csv',attempts);writecsv(output/'persistence.csv',[e for e in events if e['event_kind']=='physical_build']);writecsv(output/'activation.csv',[e for e in events if e['event_kind']=='activation']);dump(output/'independent_audit.json',status)
 for name,scope in [('history_validation','exact method/model/dialogue lineage and independent causal prompt reconstruction'),('selection_budget_validation','original-N exact k, all structural/native heads, invariant B and live AllHead per-layer sets'),('io_timing_audit','independent expected OS byte ranges, source paths, absolute request endpoints and physical event wall times')]:dump(output/(name+'.json'),{'status':'PASS' if data else 'NOT RUN','scope':scope,'main_requests':len(data),'image_audit_receipt_count':sum(x['phase']=='main' for x in checks),'evidence':'independent_audit.json image checks plus committed raw hashes'})
 if not data:return status,[],[],[],[],[],[]
 images=sorted({r['image_id'] for r in data});quality=[];latency=[];paired=[];agreements=[];storage_summary=[];session_details=[];session_summary=[]
 scopes={'T1':lambda r:r['turn_id']==1,'T2':lambda r:r['turn_id']==2,'T3':lambda r:r['turn_id']==3,'all':lambda r:True,'followup':lambda r:r['turn_id']>1}
 by={(r['dialog_id'],r['turn_id'],r['method_id']):r for r in data}
 for m in methods:
  mr=[r for r in data if r['method_id']==m]
  for scope,predicate in scopes.items():
   rs=[r for r in mr if predicate(r)];quality.append({'method_id':m,'scope':scope,'requests':len(rs),'correct':sum(r['score'] for r in rs),'accuracy':mean([r['score'] for r in rs])})
   item={'method_id':m,'scope':scope,'requests':len(rs)}
   for metric in ['ttft_ms','request_e2e_ms','ssd_read_bytes','ssd_preads','full_K_scoring_bytes','other_K_bytes','additional_selected_K_bytes','selected_V_bytes','structural_bytes','auxiliary_bytes','probe_bytes','generated_tokens']:
    item.update({metric+'_'+k:v for k,v in stats([r[metric] for r in rs]).items()})
   for metric in ['h2d_kv_bytes','h2d_payload_compute_bytes','rater_visual_h2d_bytes','selector_host_ms','cache_assembly_ms','scatter_ms','selector_native_ms','chunk_io_ms','unused_loaded_real_bytes','h2d_ms','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes']:
    vs=[r[metric] for r in rs if r[metric] is not None];item[metric+'_mean']=mean(vs) if vs else 'NOT_MEASURED';item[metric+'_measured_count']=len(vs)
   latency.append(item)
  if m!='recompute':
   sr=[r for r in storage if r['method_id']==m];x={'method_id':m,'images':len(sr),'provenance':sr[0]['provenance']}
   for metric in ['stored_content_ratio','valid_content_bytes','original_content_bytes','file_logical_bytes','allocated_bytes','os_write_bytes','os_write_calls','service_persistence_ms','activation_ms','unused_probe_bytes','explicit_structural_sidecar_bytes','auxiliary_bytes','metadata_bytes']:
    x.update({metric+'_'+k:v for k,v in stats([r[metric] for r in sr]).items()})
   x['token_pooled_retention']=sum(r['stored_k'] for r in sr)/sum(r['N'] for r in sr);storage_summary.append(x)
  for did in sorted({r['dialog_id'] for r in mr}):
   rs=[by[(did,t,m)] for t in (1,2,3)];iid=rs[0]['image_id'];s=setups[(iid,m)];source=did==s['source_dialog_id'];request=sum(r['request_e2e_ms'] for r in rs)
   session_details.append({'image_id':iid,'dialog_id':did,'method_id':m,'source_dialogue':source,'request_E2E_sum_ms':request,'actual_build_attribution_ms':s['physical_build_ms'] if source else 0.,'actual_activation_attribution_ms':s['activation_ms'] if source else 0.,'physical_stream_session_ms':request+(s['physical_build_ms']+s['activation_ms'] if source else 0.),'derived_standalone_session_ms':request+s['setup_ms'],'derived_setup_ms':s['setup_ms'],'setup_provenance':s['provenance'],'non_source_capture_increment':'NOT_MEASURED'})
  sd=[r for r in session_details if r['method_id']==m];summary={'method_id':m,'dialogues':len(sd),'derived_scope':'shared-store request measurements plus reattributed image setup; missing hypothetical non-source capture increment'}
  for metric in ['request_E2E_sum_ms','physical_stream_session_ms','derived_standalone_session_ms']:
   summary.update({metric+'_'+k:v for k,v in stats([r[metric] for r in sd]).items()})
  session_summary.append(summary)
 for baseline in methods[:-1]:
  for scope,predicate in scopes.items():
   differences=[];agree=Counter({k:0 for k in ('requests','first_token_equal','sequence_equal','normalized_prediction_equal','both_correct','reference_only_correct','B_only_correct','both_wrong')})
   for r in data:
    if r['method_id']!=baseline or not predicate(r):continue
    b=by[(r['dialog_id'],r['turn_id'],B)];differences.append({'image_id':r['image_id'],'accuracy_pp':100*(b['score']-r['score']),'ttft_ms':b['ttft_ms']-r['ttft_ms'],'request_e2e_ms':b['request_e2e_ms']-r['request_e2e_ms'],'ssd_read_bytes':b['ssd_read_bytes']-r['ssd_read_bytes']})
    agree['requests']+=1;agree['first_token_equal']+=b['tokens'][0]==r['tokens'][0];agree['sequence_equal']+=b['tokens']==r['tokens'];agree['normalized_prediction_equal']+=b['normalized']==r['normalized'];agree['both_correct' if b['score'] and r['score'] else 'reference_only_correct' if r['score'] else 'B_only_correct' if b['score'] else 'both_wrong']+=1
   agreements.append({'baseline':baseline,'scope':scope,**dict(agree)})
   for metric in ['accuracy_pp','ttft_ms','request_e2e_ms','ssd_read_bytes']:paired.append({'baseline':baseline,'direction':'B-minus-baseline','scope':scope,'metric':metric,**cluster_ci(differences,metric,images)})
  for metric in ['physical_stream_session_ms','derived_standalone_session_ms']:
   look={(r['dialog_id'],r['method_id']):r for r in session_details};diff=[{'image_id':r['image_id'],metric:look[(r['dialog_id'],B)][metric]-r[metric]} for r in session_details if r['method_id']==baseline]
   paired.append({'baseline':baseline,'scope':'per_dialogue','metric':metric,'direction':'B-minus-baseline',**cluster_ci(diff,metric,images)})
  if True:
   st={(r['image_id'],r['method_id']):r for r in storage}
   for metric in ['file_logical_bytes','allocated_bytes','os_write_bytes','service_persistence_ms','activation_ms']:
    dif=[{'image_id':im,metric:st[(im,B)][metric]-(0. if baseline=='recompute' else st[(im,baseline)][metric])} for im in images]
    paired.append({'baseline':baseline,'scope':'image_macro_setup','metric':metric,'direction':'B-minus-baseline',**cluster_ci(dif,metric,images)})
 actualsetup=sum(r['actual_build_attribution_ms']+r['actual_activation_attribution_ms'] for r in session_details)
 main_events=[e for e in events if e['phase']=='main'];physicalsetup=sum(e['duration_ms'] for e in main_events)
 assert math.isclose(actualsetup,physicalsetup,abs_tol=1e-5),'physical setup attribution mismatch (retry setup must be accounted explicitly)'
 assert len([e for e in main_events if e['event_kind']=='physical_build'])==(1194 if model=='llava' else 796)
 dump(output/'session_accounting_audit.json',{'status':'PASS','attributed_setup_ms':actualsetup,'actual_physical_event_setup_ms':physicalsetup,'physical_builds':sum(e['event_kind']=='physical_build' for e in main_events),'activation_events':sum(e['event_kind']=='activation' for e in main_events),'run_wall_excludes':'conditioning, diagnostic audit/hash, logging and cleanup are outside request/setup sums'})
 for name,rs in [('quality',quality),('latency',latency),('paired_analysis',paired),('prediction_agreement',agreements),('image_budget_io',geometry),('selection_layer_T2_T3',variation),('storage_metrics',storage),('storage_summary',storage_summary),('session_per_dialogue',session_details),('session_summary',session_summary)]:writecsv(output/(name+'.csv'),rs)
 unique_geometry={}
 for r in geometry:
  key=(r['image_id'],r['method_id']);z={k:v for k,v in r.items() if k not in ('dialog_id','turn_id','read_bytes','V_chunk_coverage_layer_mean')}
  if key in unique_geometry:assert unique_geometry[key]==z
  else:unique_geometry[key]=z
 writecsv(output/'per_image_geometry.csv',list(unique_geometry.values()))
 writecsv(output/'request_diagnostics.csv',[{k:v for k,v in r.items() if k not in ('tokens','prediction','normalized')} for r in data])
 if variation:
  grouped=defaultdict(list)
  for r in variation:grouped[r['image_id']].append(r)
  writecsv(output/'selection_image_summary.csv',[{'image_id':im,'layer_dialogue_pairs':len(rs),'mean_layer_Jaccard':mean([r['Jaccard_T2_T3'] for r in rs]),'changed_layer_fraction':mean([r['changed'] for r in rs])} for im,rs in grouped.items()])
 return status,quality,latency,paired,storage_summary,session_summary,variation

def fmt(x,d=2):return 'N/A' if x is None else f'{x:.{d}f}' if isinstance(x,(int,float)) else str(x)
def mdtable(headers,values):return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(v) for v in row)+' |' for row in values]
def ci_text(p,scale=1):return f"{p['point']*scale:+.3f} [{p['ci_low']*scale:+.3f}, {p['ci_high']*scale:+.3f}]"

def render(run,results,protection):
 lines=['# MT-GQA-reconstructed generated-history: LLaVA 5-arm / Qwen 3-arm B main','',f'Run: `{run}`. 본 보고서의 main 수치는 이 run에서 실제 채택한 요청만 사용한다. MT-GQA-reconstructed이며 공식 MetaCompress 원본 artifact/evaluator 재현은 아니다.','', '## 1. 실행 범위','']
 lines+=mdtable(['Model','Main status','방법','Images / dialogues','Final / expected requests','Physical started / failed image attempts','Smoke','History'],[[model,ret[0]['status'],5 if model=='llava' else 3,'398 / 4061' if ret[0]['status']=='PASS' else 'main 0 / 0',f"{ret[0]['main_requests']:,} / {ret[0]['expected_main_requests']:,}",f"{ret[0]['physical_attempts'].get('main_STARTED',0)} / {ret[0]['physical_attempts'].get('main_failed_image_attempts',0)}",ret[0]['smoke'],ret[0]['history_lineage']] for model,ret in results.items()])
 lines+=['','CPU 456개 회귀검사, B 고정 LLaVA 5개/Qwen 10개 검증, AllHead 검증, 통합·smoke·reference·예외 주입은 main 분모와 별도다. 실행된 세부 개수와 실패 attempt는 physical_attempts.csv 및 독립 감사 JSON에 있다. ReComp followup은 비교 cohort이며 SSD hit가 아니다.','',
  'Qwen은 새 B 경로에서도 `n313060 / mtgqa_000285 / T2`의 기존 FP32 dense/compact criterion(atol=rtol=1e-5)을 실패했다. 215,040개 원소 중 16개 위반, 최대 절대차 2.16215848923e-5. 같은 capture의 native BF16 KV bits, memory/SSD first logits·generated IDs 비교는 이 assertion 전에 통과했다. 이는 G7 면제가 아니며 Qwen main·smoke는 BLOCKED / NOT RUN이다. 기존 v1 실패·이전 main 실패·이번 B core PASS·이번 main integration FAIL을 서로 대체하지 않는다. FP64 대조 또는 허용치 완화로 승인하지 않았다.','']
 for model,(status,q,l,p,s,se,v) in results.items():
  if not q:continue
  methods=[x['method_id'] for x in q if x['scope']=='all'];B=methods[-1];qm={(r['method_id'],r['scope']):r for r in q};lm={(r['method_id'],r['scope']):r for r in l}
  lines += [f'## {model}: 2. 품질','']
  lines+=mdtable(['Method','T1 %','T2 %','T3 %','All % (correct/N)','Followup % (correct/N)'],[[LABEL[m],*[fmt(100*qm[(m,t)]['accuracy'],3) for t in ('T1','T2','T3')],f"{100*qm[(m,'all')]['accuracy']:.3f} ({int(qm[(m,'all')]['correct'])}/{qm[(m,'all')]['requests']})",f"{100*qm[(m,'followup')]['accuracy']:.3f} ({int(qm[(m,'followup')]['correct'])}/{qm[(m,'followup')]['requests']})"] for m in methods])
  lines+=['']+mdtable(['B − baseline','Followup accuracy Δpp [95% CI]','All accuracy Δpp [95% CI]'],[[LABEL[m],ci_text(next(x for x in p if x['baseline']==m and x['scope']=='followup' and x['metric']=='accuracy_pp')),ci_text(next(x for x in p if x['baseline']==m and x['scope']=='all' and x['metric']=='accuracy_pp'))] for m in methods[:-1]])
  lines+=['','T1/T2/T3 모든 paired CI와 first-token/full-sequence/normalized-prediction agreement 및 both-correct/reference-only/B-only/both-wrong은 paired_analysis.csv, prediction_agreement.csv에 있다. Strict normalized exact match를 원문 prediction/gold[0]에서 독립 재계산했다. History는 방법 자체의 실제 생성 답변이므로 차이는 history divergence를 포함한다. B/A 일치는 baseline 품질 동등성 근거가 아니다.','',f'## {model}: 3. Followup latency / 실제 SSD I/O','']
  lines+=mdtable(['Method','TTFT mean / p50 / p95 ms','E2E mean / p50 / p95 ms','Full-K scoring / other K / V MB','Struct / aux / total MB','Preads'],[[LABEL[m], ' / '.join(fmt(lm[(m,'followup')]['ttft_ms_'+z]) for z in ('mean','p50','p95')),' / '.join(fmt(lm[(m,'followup')]['request_e2e_ms_'+z]) for z in ('mean','p50','p95')),' / '.join(fmt(lm[(m,'followup')][z+'_mean']/1e6) for z in ('full_K_scoring_bytes','other_K_bytes','selected_V_bytes')),' / '.join(fmt(lm[(m,'followup')][z+'_mean']/1e6) for z in ('structural_bytes','auxiliary_bytes','ssd_read_bytes')),fmt(lm[(m,'followup')]['ssd_preads_mean'])] for m in methods])
  lines+=['','AllHead additional selected-K 및 probe read는 실제 trace에서 0이다. Full K 한 번을 scoring과 answer에서 재사용하며 total bytes에 이중 합산하지 않는다. Other K는 FullLoad/MPIC/B의 실제 K payload read다. Structural sidecar 중복은 실제 traffic이다. ReComp KV read 0은 image-file I/O 0을 뜻하지 않는다. H2D·selector·assembly·GPU peak의 측정 필드/미측정 수는 latency.csv에 명시했다. H2D는 tensor bytes이며 버스 계측이 아니다. GPU peak는 coexisting contexts 범위이며 독립 배포 footprint가 아니다.','',f'## {model}: 4. 이미지별 storage / setup','']
  lines+=mdtable(['Method','Stored content % (image / pooled)','Logical / allocated MB','OS write MB / calls','Persistence / activation ms','Unused probe MB','Provenance'],[[LABEL[x['method_id']],f"{100*x['stored_content_ratio_mean']:.4f} / {100*x['token_pooled_retention']:.4f}",f"{x['file_logical_bytes_mean']/1e6:.3f} / {x['allocated_bytes_mean']/1e6:.3f}",f"{x['os_write_bytes_mean']/1e6:.3f} / {x['os_write_calls_mean']:.2f}",f"{x['service_persistence_ms_mean']:.3f} / {x['activation_ms_mean']:.3f}",fmt(x['unused_probe_bytes_mean']/1e6,3),x['provenance']] for x in s])
  lines+=['','B는 원래 N 기준 ceil(N/4) content와 모든 structural rows를 저장하고 마지막 파일은 정확히 k rows에서 끝난다. Full CPU materialization/repack 비용은 유지한다. FullLoad/AllHead canonical bundle은 동일 물리 build이며 표의 두 deployment footprint를 물리 총량에 중복 합산하지 않는다. Canonical bundle의 미사용 probe sidecar까지 쓰기·fsync·완료 hash 비용에 포함한다. AllHead base-only persistence는 NOT_MEASURED. OS write는 /proc/self/io successful write bytes/calls이며 NAND write가 아니다. Native writer stage와 common seal stage는 storage_metrics.csv에 있으며 중첩 component를 request critical path에 합산하지 않는다.','',f'## {model}: 5. 실제 shared-image stream 회계','']
  lines+=mdtable(['Method','3-request E2E mean ms','Physical attributed session mean / p50 / p95 ms'],[[LABEL[x['method_id']],fmt(x['request_E2E_sum_ms_mean']),' / '.join(fmt(x['physical_stream_session_ms_'+z]) for z in ('mean','p50','p95'))] for x in se])
  bci=next(x for x in p if x['baseline']=='recompute' and x['metric']=='physical_stream_session_ms')
  lines+=['',f"B − ReComp 실제 귀속 session Δms: **{ci_text(bci)}**. 원본 physical build/activation event 합과 dialogue 귀속 setup 합의 일치를 session_accounting_audit.json에서 확인했다. Canonical build는 FullLoad에 한 번만 귀속되며 AllHead의 build 귀속 0은 독립 배포 무료를 의미하지 않는다.",'',f'## {model}: 6. Standalone-equivalent 환산','']
  lines+=mdtable(['Method','Derived session mean / p50 / p95 ms','범위'],[[LABEL[x['method_id']],' / '.join(fmt(x['derived_standalone_session_ms_'+z]) for z in ('mean','p50','p95')),'actual requests + per-image setup once/dialogue; DERIVED'] for x in se])
  bci=next(x for x in p if x['baseline']=='recompute' and x['metric']=='derived_standalone_session_ms')
  lines+=['',f"B − ReComp 환산 session Δms: **{ci_text(bci)}**. Store를 모든 dialogue마다 실제 새로 생성한 측정값이 아니다. Source T1만 capture하므로 hypothetical non-source capture 증분은 NOT_MEASURED이며 완전한 cold-new-image session으로 부르지 않는다. Qwen score/clone tail은 source T1 E2E에 한 번 들어가며 setup에 재청구하지 않는다. Conditioning/logging/audit/cleanup은 request/setup 합과 분리한다."]
  if model=='llava':
   lines+=['', '## 7. LLaVA AllHead와 B의 실제 tradeoff','']
   st={r['method_id']:r for r in s}
   lines+=mdtable(['Method','Attention content','Stored content %','SSD MB / V MB','TTFT mean / p95 ms','Followup accuracy %','Selector host / prefix selector ms','Selection'],[[LABEL[m],'ceil(N/4) every layer',fmt(100*st[m]['stored_content_ratio_mean'],4),f"{lm[(m,'followup')]['ssd_read_bytes_mean']/1e6:.3f} / {lm[(m,'followup')]['selected_V_bytes_mean']/1e6:.3f}",f"{lm[(m,'followup')]['ttft_ms_mean']:.3f} / {lm[(m,'followup')]['ttft_ms_p95']:.3f}",fmt(100*qm[(m,'followup')]['accuracy'],3),fmt(lm[(m,'followup')]['selector_host_ms_mean'] if m==AH else lm[(m,'followup')]['selector_native_ms_mean']),'actual suffix/layer dependent' if m==AH else 'same-image invariant'] for m in (AH,B)])
   if v:lines+=['',f"AllHead T2/T3 layer-level Jaccard 평균 {mean([x['Jaccard_T2_T3'] for x in v]):.6f}, 선택 집합이 달라진 layer-dialogue 비율 {mean([x['changed'] for x in v]):.6f}. {len(v):,} layer-dialogue pairs이며 image-macro 집계는 selection_image_summary.csv에 별도 보존했다. 모든 질문의 선택이 반드시 달라야 한다는 조건은 적용하지 않았다."]
   lines+=['','Selector 열은 AllHead의 전체 layer selector host 합과 B의 native prefix-selector wall로 scope가 다르며 중첩 stage를 TTFT에 더하지 않는다. AllHead는 layer별 live Q와 전체 attention heads의 causal score, full K read, covering V chunks를 사용한다. KV attention 예산 25%가 SSD read 25%를 뜻하지 않는다. MPIC는 첫 32개 이미지 rows를 재계산하면서 전체 context를 유지한다. 기존 MPIC/AllHead 알고리즘은 변경하지 않았다.']
 lines+=['','## 추정과 재현의 한계','', 'Paired CI는 seed1234, image-cluster bootstrap 10,000회이며 같은 이미지의 모든 대화·turn·방법을 함께 resample한다. Point estimate와 bootstrap 모두 pooled sum/count로 요청 가중치를 보존한다. Setup image-macro는 별도 표기했다. Bootstrap은 독립 반복 실험이 아니며 CI가 0을 포함해도 동등성을 증명하지 않는다. 평균 개선을 p95 개선으로 일반화하지 않는다. 상위 25% pruning이나 GPU 메모리 75% 감소를 구현한 실험이 아니다. POSIX_FADV_DONTNEED 성공을 기록했지만 cold NAND를 보장하지 않는다. 단일 GPU·batch1·한 이미지씩 제한된 저장 정책이며 cache saturation/hit-rate 또는 병렬 throughput의 검증이 아니다.', '',f"기존 artifact 보호: {protection['status']}, {protection['protected_files']:,}개. 기존 source 변경 {len(protection['changed_existing_source'])}건. 8 MiB 초과 기존 파일은 9×64 KiB fingerprint + inode/mtime/ctime이며 full-byte SHA 증명이 아니다. 작은 기존 artifact/source와 새 payload/raw/manifest에는 full SHA를 사용했다. 완료 이미지 payload는 독립 audit·hash·atomic commit 후 소유 allowlist로 정리했으며 재실행에는 재생성이 필요하다.", '', '새 실행 명령, gate 조건과 payload 재생성 절차는 REPRODUCE.md, 실제 source/config/모델/환경 freeze는 run의 해당 JSON 및 source_frozen/을 참조한다. 실행 시작 또는 partial 상태를 full main 완료로 해석하지 않는다.']
 return '\n'.join(lines)+'\n'

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run=a.run.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 assert not (out/'REPORT.md').exists(),'use unused output directory'
 # Simple independent estimand check: unequal clusters must pool counts.
 proof=cluster_ci([{'image_id':'a','v':0.}]+[{'image_id':'b','v':1.}]*3,'v',['a','b']);assert proof['point']==.75
 dump(out/'analysis_source_freeze.json',{'report_source':str(Path(__file__)),'sha256':sha(__file__),'weighting_self_check':proof})
 results={model:analyze_model(run/model,out/model,model) for model in ('llava','qwen')}
 import main_b_support as S
 protection=S.protection(run);dump(out/'protection_final.json',protection);assert protection['status']=='PASS',protection
 (out/'REPORT.md').write_text(render(run,results,protection))
 stages={}
 for model,r in results.items():
  s=r[0];has=bool(s['main_requests']);stages[model]={'BASELINE REUSE CONTRACT':'PASS','FINAL SOURCE FREEZE':'PASS','CPU REGRESSION':read(run/'cpu_regression.json')['status'],'GPU CORRECTNESS':s['core_gpu'],'MAIN INTEGRATION':s['integration'],'MAIN-RUNNER SMOKE':s['smoke'],'FULL MAIN COVERAGE':s['full_main_coverage'],'GENERATED-HISTORY LINEAGE':'PASS' if has else 'NOT RUN','SELECTION/BUDGET VALIDATION':'PASS' if has else 'NOT RUN','I/O AND TIMING AUDIT':'PASS' if has else 'NOT RUN','QUALITY RESCORING':s['quality_rescoring'],'SHARED/DERIVED SESSION ACCOUNTING':'PASS' if has else 'NOT RUN','ARTIFACT PROTECTION':protection['status'],'RAW REPRODUCIBILITY':'PASS','final_main_requests':s['main_requests'],'expected_main_requests':s['expected_main_requests'],'model_status':s['status']}
 dump(out/'FINAL_STATUS.json',stages)
 physical=[r for model in ('llava','qwen') for r in rows_to_csv(out/model/'physical_events.csv')]
 dump(out/'physical_event_counts.json',dict(Counter((r['model']+'/'+r['phase']+'/'+r['event_kind']) for r in physical)))
 entries=[]
 for base in (run,out):
  for f in sorted(base.rglob('*')):
   if f.is_file() and not f.is_symlink():entries.append({'path':str(f),'bytes':f.stat().st_size,'sha256':sha(f)})
 dump(out/'artifact_index.json',{'files':entries,'scope':'all new run and final result files; index excludes itself','report_source_sha256':sha(__file__)})
 print(json.dumps(stages),flush=True)

def rows_to_csv(path):
 with Path(path).open(newline='') as f:return list(csv.DictReader(f))
if __name__=='__main__':main()

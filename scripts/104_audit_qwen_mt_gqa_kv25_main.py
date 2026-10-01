#!/usr/bin/env python3
"""Independent Qwen raw auditor: no production selector, loader, renderer or summary imports."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,re,shutil,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
METHODS=('recompute','fullload','qwen_ours_kv25');LABELS=dict(zip(METHODS,('ReComp','FullLoad','Ours-KV25')))
MODEL='Qwen/Qwen2.5-VL-7B-Instruct';SCHEMA='qwen-mtgqa-kv25-main-v1'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def read_rows(p):
 with Path(p).open() as f:
  for line in f:
   if line.strip():yield json.loads(line)
def dump(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def csv_write(p,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(p).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def score(p,g):
 def norm(x):return ' '.join(w for w in re.sub(r'[^\w\s]',' ',str(x).lower()).split() if w not in ('a','an','the'))
 return float(norm(p)==norm(g[0]))
class Checks:
 def __init__(self):self.count=0;self.failures=[];self.counts=Counter()
 def require(self,ok,name,context=None):
  self.count+=1
  if not ok:
   self.counts[name]+=1
   if len(self.failures)<50:self.failures.append({'check':name,'context':context})
 def result(self):return {'status':'PASS' if not self.counts else 'FAIL','checks':self.count,'failure_counts':dict(self.counts),'failures':self.failures}

def prompt_text(q,history):
 text='<|im_start|>system\nYou are a helpful assistant. Give concise factual answers.<|im_end|>\n'
 turns=list(history)+[(q,None)]
 for i,(question,answer) in enumerate(turns):
  text+='<|im_start|>user\n'+('<|vision_start|><|image_pad|><|vision_end|>' if i==0 else '')+question+'\nAnswer using a single word or short phrase.<|im_end|>\n'
  text+='<|im_start|>assistant\n'
  if answer is not None:text+=answer+'<|im_end|>\n'
 return text
_TOKENIZERS={}
def tokenizer(config):
 p=config['model']['tokenizer_json']
 if p not in _TOKENIZERS:
  from tokenizers import Tokenizer
  _TOKENIZERS[p]=Tokenizer.from_file(p)
 return _TOKENIZERS[p]

def audit_image_rows(rows,manifest,config,phase,iid,artifact=None):
 c=Checks();ds=[d for d in manifest['dialogues'] if d['image_id']==iid]
 if phase in ('integration','smoke'):ds=[d for d in ds if d['dialog_id'] in manifest[phase+'_dialogue_ids']]
 expected={(d['dialog_id'],m,t):d['turns'][t-1] for d in ds for m in METHODS for t in (1,2,3)}
 by={(r['dialog_id'],r['method_id'],r['turn_id']):r for r in rows};ordinals={d['dialog_id']:i for i,d in enumerate(manifest['dialogues'])}
 c.require(bool(ds) and set(by)==set(expected) and len(rows)==len(expected),'exact_image_coverage',iid)
 c.require(len({r['logical_request_id'] for r in rows})==len(rows),'unique_logical',iid)
 c.require(len({r['physical_execution_id'] for r in rows})==len(rows),'unique_physical',iid)
 metas={};selections={}
 if artifact is not None:
  for m in METHODS[1:]:
   metas[m]=json.loads((artifact/f'{m}_meta.json').read_text());selections[m]=json.loads((artifact/f'{m}_selection.json').read_text())
   meta=metas[m];sel=selections[m];n=meta['visual_count'];order=meta['stored_to_original']
   c.require((meta['num_layers'],meta['num_kv_heads'],meta['head_dim'],meta['dtype'],meta['chunk_size'])==(28,4,128,'bfloat16',64),'native_BF16_store_geometry',m)
   c.require(meta['identity']['image_sha256']==manifest['image_identity'][iid]['sha256'],'store_image_identity',m)
   c.require(meta['code_revision']==config['core_model_store_revision'],'core_store_revision',m)
   c.require(meta['visual_count']==manifest['image_identity'][iid]['N_content'],'preflight_actual_geometry',m)
   c.require(meta['key_rope_state']=='post_mrope' and meta['identity']['position_policy']==config['position_policy'],'position_policy',m)
   if m==METHODS[2]:
    scores=sel['scores_float32'];c.require(len(scores)==n and all(math.isfinite(x) for x in scores),'scores_finite_complete',m)
    rank=sorted(range(n),key=lambda i:(-scores[i],i));c.require(order==rank,'independent_stable_rank',m)
    c.require(sel['selected_original_ids']==sorted(rank[:(n+3)//4]),'selection_artifact_topk',m)
   else:c.require(order==list(range(n)),'canonical_order',m)
   c.require(meta['prefix_input_ids']==sel['prefix_ids'],'captured_prefix_ids',m)
   p=json.loads((artifact/f'{m}_persistence.json').read_text());a=json.loads((artifact/f'{m}_activation.json').read_text())
   c.require(p['source']==sel['source'] and p['source']['dialog_id']==ds[0]['dialog_id'] and p['source']['turn_id']==1,'source_capture_persistence',m)
   c.require(p['bytes_written']==meta['bytes_visual_kv']+meta['bytes_structural_kv']+meta['bytes_metadata_file'],'actual_persistence_bytes',m)
   c.require(math.isclose(p['charged_setup_before_activation_ms'],p['outer_writer_wall_ms']+p['post_generation_capture_tail_ms'],abs_tol=1e-8),'no_setup_double_count',m)
   c.require(a['activation_io']['bytes']==p['bytes_written'] and a['metadata_resident_bytes']>0,'activation_integrity_IO',m)
  c.require(metas[METHODS[1]]['prefix_input_ids']==metas[METHODS[2]]['prefix_input_ids'],'source_prefix_compatibility',iid)
 tok=tokenizer(config)
 sets=[]
 for key,r in by.items():
  did,m,t=key;q=expected.get(key);rid=r['logical_request_id'];v=r['result'];d=r['request_trace'];pixel=t==1 or m=='recompute'
  if q is None:continue
  c.require(rid==f"{SCHEMA}:{r['experiment_id']}:{phase}:{MODEL}:{m}:{did}:t{t}",'logical_identity',rid)
  c.require(r['schema_version']==SCHEMA and r['model_id']==MODEL and r['model_revision']==config['model']['revision'],'model_schema',rid)
  c.require(r['phase']==phase and r['image_id']==iid and r['status']=='PASS','phase_status',rid)
  c.require(r['question_id']==str(q['question_id']) and r['question']==q['question'] and r['gold']==q['answers'],'frozen_question_gold',rid)
  c.require(r['config_sha256']==config['_file_sha256'] and r['manifest_sha256']==config['_manifest_sha256'] and r['code_sha256']==canonical(config['source_sha256']),'config_source_manifest',rid)
  o=ordinals[did];rotation=list(METHODS[o%3:]+METHODS[:o%3]);c.require(r['method_order']==rotation and r['method_order_position']==rotation.index(m) and r['global_dialog_ordinal']==o,'rotation',rid)
  history=[];entries=[]
  for pt in range(1,t):
   prev=by.get((did,m,pt));pq=expected.get((did,m,pt))
   if prev is None:c.require(False,'missing_history_source',rid);continue
   history.append([pq['question'],prev['prediction']]);entries.append({'turn_id':pt,'question_id':str(pq['question_id']),'question':pq['question'],'answer':prev['prediction'],
    'source_model':MODEL,'source_method_id':m,'source_dialog_id':did,'source_logical_request_id':prev['logical_request_id'],'source_physical_execution_id':prev['physical_execution_id']})
  text=prompt_text(q['question'],history)
  c.require(r['history']==history and r['history_entries']==entries and d['history']==history and d['history_entries']==entries,'actual_own_generated_history',rid)
  c.require(r['history_sha256']==canonical(history) and r['prompt']==text==d['prompt'] and r['prompt_sha256']==hashlib.sha256(text.encode()).hexdigest(),'independent_causal_prompt',rid)
  n=r['N_content'];s=r['S_structural'];k=r['k'];target=n if pixel or m=='fullload' else (n+3)//4
  c.require(n==manifest['image_identity'][iid]['N_content'] and n>0 and s==21 and k==target,'actual_budget_geometry',rid)
  c.require(r['content_kv_fraction']==k/n and r['structural_inclusive_kept_ratio']==(k+s)/(n+s),'retention',rid)
  c.require(r['budget_unit']==('visual_kv' if m==METHODS[2] else 'full_context'),'explicit_budget',rid)
  rawids=tok.encode(text,add_special_tokens=False).ids;image_token=config['model']['image_token_id'];expanded=[]
  c.require(rawids.count(image_token)==1,'one_image_in_prompt',rid)
  for token in rawids:expanded.extend([token]*n if token==image_token else [token])
  c.require(r['input_ids']==[expanded]==d['input_ids'] and r['input_ids_sha256']==canonical([expanded]),'independent_input_tokenization',rid)
  c.require(r['image_sha256']==manifest['image_identity'][iid]['sha256'],'image_hash',rid)
  tokens=v['generated_token_ids'];c.require(1<=len(tokens)<=16 and tokens[0]==v['first_token_id'] and len(tokens)==v['generated_token_count']==v['model_forward_calls'],'actual_generated_sequence',rid)
  c.require(r['prediction']==v['prediction']==tok.decode(tokens,skip_special_tokens=True).strip(),'raw_decoded_prediction',rid)
  c.require(r['score']==score(r['prediction'],q['answers']),'independent_strict_score',rid)
  start,first,end,ret=[v[x] for x in ('request_started_at_s','first_token_at_s','request_finished_at_s','core_returned_at_s')]
  c.require(start<first<=end<=ret and 0<r['ttft_ms']<=r['request_e2e_ms'],'timing_order',rid)
  c.require(math.isclose(r['ttft_ms'],1000*(first-start),abs_tol=1e-7) and math.isclose(r['request_e2e_ms'],1000*(end-start),abs_tol=1e-7),'absolute_true_timers',rid)
  c.require(r['ttft_ms']==v['ttft_ms'] and r['request_e2e_ms']==v['request_e2e_ms'] and math.isclose(v['post_generation_tail_ms'],1000*(ret-end),abs_tol=1e-7),'timer_consistency',rid)
  capture=(m in METHODS[1:] and did==ds[0]['dialog_id'] and t==1)
  c.require(v['actual_vision_calls']==v['vision_calls']==int(pixel) and v['actual_score_calls']==int(capture and m==METHODS[2]),'actual_vision_score_calls',rid)
  c.require(v['online_query_score_calls']==0 and v['request_state_restored'] is True and v['first_logits_finite'] is True,'no_online_score_state_restored',rid)
  c.require(v['capture_mode']==('kv_only' if m=='fullload' else 'with_score') if capture else v['capture_mode'] is False,'source_T1_only_capture',rid)
  c.require(v['image_file_decode_ms'] is not None and v['image_file_decode_ms']>=0 if pixel else v['image_file_decode_ms'] is None,'per_request_image_decode',rid)
  trace=v['os_pread_trace'];c.require(r['ssd_read_bytes']==sum(e['returned'] for e in trace) and r['ssd_preads']==len(trace),'actual_OS_bytes_calls',rid)
  c.require(all(e['returned']==e['requested'] for e in trace),'full_preads',rid)
  fullpos=d['position_ids'];cache_before=d['cache_before_decode'];prefix=n+s
  c.require(d['cache_after_decode']==cache_before+len(tokens)-1 and d['decode_start_position']==max(max(axis[0]) for axis in fullpos)+1,'full_logical_decode_position',rid)
  expected_input=expanded if pixel else expanded[prefix:];expected_positions=fullpos if pixel else [[axis[0][prefix:]] for axis in fullpos]
  c.require(d['actual_prefill_input_ids']==[expected_input] and d['actual_prefill_position_ids']==expected_positions,'suffix_full_logical_positions',rid)
  firstslot=0 if pixel else k+s;masklen=(len(expanded) if pixel else k+s+len(expected_input))
  c.require(d['actual_prefill_cache_position']==list(range(firstslot,firstslot+len(expected_input))) and d['actual_prefill_attention_mask']==[[1]*masklen] and cache_before==masklen,'compact_slots_causal_input',rid)
  if pixel:
   c.require(not trace and r['source_store'] is None and not d['layer_prefix_shapes'],'pixel_no_SSD_KV',rid)
  else:
   c.require(v['budget_unit']==('chunk' if m=='fullload' else 'visual_kv'),'actual_load_budget_unit',rid)
   mchunk=(target+63)//64;disk=mchunk*64;rowbytes=4*128*2;structbytes=2*28*s*rowbytes
   expectedreads=Counter({('structural_kv.bin',0,structbytes,structbytes):1})
   for li in range(28):
    for kind in ('k','v'):expectedreads[(f'layer_{li:03d}/{kind}.bin',0,disk*rowbytes,disk*rowbytes)]=1
   actual=Counter((('/'.join(Path(e['path']).parts[-2:]) if Path(e['path']).parent.name.startswith('layer_') else Path(e['path']).name),e['offset'],e['requested'],e['returned']) for e in trace)
   c.require(actual==expectedreads and len(trace)==v['pread_calls']==57,'minimum_whole_chunk_OS_ranges',rid)
   c.require(v['read_io']['bytes']==r['ssd_read_bytes'] and v['read_io']['preads']==57 and v['metadata_read_bytes']==0,'production_independent_IO_match',rid)
   c.require(v['visual_read_bytes']==2*28*disk*rowbytes and v['structural_read_bytes']==structbytes,'KV_structural_bytes',rid)
   c.require(v['selected_chunks']==v['normal_chunks_read']==mchunk and v['kept_tokens']==v['target_visual_tokens']==target and v['loaded_valid_visual_rows']==min(n,disk) and v['extra_valid_visual_rows']==min(n,disk)-target and v['padding_rows_read']==disk-min(n,disk),'whole_chunk_extras_padding',rid)
   c.require(v['compact_prefix_tokens']==target+s and d['layer_prefix_shapes']==[[[1,4,target+s,128],[1,4,target+s,128]]]*28,'all_layer_native_heads_exact_rows',rid)
   c.require(v['h2d_kv_bytes']==v['gpu_cache_kv_bytes']==2*28*(target+s)*rowbytes,'trim_before_H2D',rid)
   c.require(v['selected_visual_stored']==list(range(target)),'first_k_stored_rows',rid)
   if m in metas:
    meta=metas[m];selected=sorted(meta['stored_to_original'][:target]);c.require(v['selected_visual_original']==selected,'independent_selected_original',rid)
    c.require(all(axis[0][:prefix]==meta['logical_position_ids'][i] for i,axis in enumerate(fullpos)),'prefix_MRoPE_identity',rid)
   events=v['conditioning_events'];paths={e['path'] for e in events};c.require(len(events)==57 and all(e['status']=='SUCCESS' for e in events) and all(e['path'] in paths for e in trace),'all_files_conditioned',rid)
   src=r['source_store'];prior=by.get((ds[0]['dialog_id'],m,1),{})
   c.require(src['source']['physical_execution_id']==prior.get('physical_execution_id') and src['source']['logical_request_id']==prior.get('logical_request_id') and src['source']['turn_id']==1,'actual_source_capture_lineage',rid)
   if m==METHODS[2]:sets.append(v['selected_visual_original'])
 c.require(not sets or all(x==sets[0] for x in sets),'same_image_T2_T3_selection_invariant',iid)
 return dict(c.result(),image_id=iid,phase=phase,requests=len(rows),expected_requests=len(expected))

def protection(run):
 changed=[];checked=total=0
 for r in read_rows(run/'protected_artifacts_before.jsonl'):
  p=ROOT/r['path'];checked+=1
  if not p.exists() and not p.is_symlink():changed.append({'path':r['path'],'reason':'missing'});continue
  if 'symlink' in r:
   if not p.is_symlink() or os.readlink(p)!=r['symlink']:changed.append({'path':r['path'],'reason':'symlink changed'})
   continue
  s=p.stat();total+=s.st_size
  equal=(s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino)==(r['size'],r['mtime_ns'],r['ctime_ns'],r['inode'])
  if r['hash_mode']=='full_sha256':equal=equal and sha(p)==r['sha256']
  else:
   h=hashlib.sha256()
   with p.open('rb') as f:
    for off in r['offsets']:f.seek(off);b=f.read(65536);h.update(off.to_bytes(8,'big')+len(b).to_bytes(8,'big')+b)
   equal=equal and h.hexdigest()==r['fingerprint']
  if not equal:changed.append({'path':r['path'],'reason':'content/metadata changed'})
 before=json.loads((run/'source_before.json').read_text());source=[p for p,h in before.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
 return {'status':'PASS' if not changed and not source else 'FAIL','protected_files':checked,'protected_logical_bytes':total,'changed_artifacts':changed,'changed_existing_source':source,'limitation':'sources and <=8MiB files full SHA256; larger historical files nine framed 64KiB windows plus inode/size/mtime/ctime; not full-byte proof'}

def mean(x):return float(np.mean(x)) if len(x) else None
def compact(r):
 v=r['result'];d={k:r[k] for k in ('method_id','image_id','dialog_id','turn_id','score','ttft_ms','request_e2e_ms','ssd_read_bytes','ssd_preads','N_content','S_structural','k','content_kv_fraction','structural_inclusive_kept_ratio')}
 for key in ('visual_read_bytes','structural_read_bytes','metadata_read_bytes','padding_rows_read','extra_valid_visual_rows','loaded_valid_visual_rows','h2d_kv_bytes','gpu_cache_kv_bytes','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes','generated_token_count','capture_clone_ms','score_extra_ms','post_generation_tail_ms'):
  d[key]=v.get(key,0)
 d.update(cap_reached=v['truncated'],prediction=r['prediction'],generated_token_ids=json.dumps(v['generated_token_ids']),physical_execution_id=r['physical_execution_id'],logical_request_id=r['logical_request_id'])
 return d

def summaries(rows):
 summary=[];scope=[];budget=[]
 for m in METHODS:
  rs=[r for r in rows if r['method_id']==m]
  if not rs:continue
  hit=[r for r in rs if r['turn_id']>1];one=[r for r in rs if r['turn_id']==1]
  x={'method_id':m,'label':LABELS[m],'requests':len(rs),'hits':len(hit),'all_correct':sum(r['score'] for r in rs),'all_acc':mean([r['score'] for r in rs]),'hit_correct':sum(r['score'] for r in hit),'hit_acc':mean([r['score'] for r in hit]),'T1_ttft_mean_ms':mean([r['ttft_ms'] for r in one]),'hit_ttft_mean_ms':mean([r['ttft_ms'] for r in hit]),'hit_ttft_p50_ms':float(np.percentile([r['ttft_ms'] for r in hit],50)),'hit_ttft_p95_ms':float(np.percentile([r['ttft_ms'] for r in hit],95)),'SSD_MB_per_hit':mean([r['ssd_read_bytes']/1e6 for r in hit]),'preads_per_hit':mean([r['ssd_preads'] for r in hit])}
  for t in (1,2,3):
   ts=[r for r in rs if r['turn_id']==t];x.update({f'acc_T{t}':mean([r['score'] for r in ts]),f'correct_T{t}':sum(r['score'] for r in ts),f'count_T{t}':len(ts)})
  summary.append(x)
  for name,selected in [('T1',one),('T2',[r for r in rs if r['turn_id']==2]),('T3',[r for r in rs if r['turn_id']==3]),('all',rs),('hit',hit)]:
   z={'method_id':m,'scope':name,'requests':len(selected),'correct':sum(r['score'] for r in selected),'accuracy':mean([r['score'] for r in selected])}
   for metric in ('ttft_ms','request_e2e_ms'):
    values=[r[metric] for r in selected];z[metric+'_mean']=mean(values);z[metric+'_p50']=float(np.percentile(values,50));z[metric+'_p95']=float(np.percentile(values,95))
   scope.append(z)
  image={r['image_id']:r for r in hit};ims=list(image.values())
  b={'method_id':m,'budget_unit':'visual_kv' if m==METHODS[2] else 'full_context','mean_N_image':mean([r['N_content'] for r in ims]),'content_retention_image_mean':mean([r['k']/r['N_content'] for r in ims]),'content_retention_image_token_pooled':sum(r['k'] for r in ims)/sum(r['N_content'] for r in ims),'content_retention_hit_token_pooled':sum(r['k'] for r in hit)/sum(r['N_content'] for r in hit),'structural_inclusive_retention_image_mean':mean([r['structural_inclusive_kept_ratio'] for r in ims]),'SSD_bytes_per_hit':mean([r['ssd_read_bytes'] for r in hit]),'K_bytes_per_hit':mean([r['visual_read_bytes']/2 for r in hit]),'V_bytes_per_hit':mean([r['visual_read_bytes']/2 for r in hit]),'structural_bytes_per_hit':mean([r['structural_read_bytes'] for r in hit]),'metadata_bytes_per_hit':0,'preads_per_hit':x['preads_per_hit'],'H2D_KV_bytes_per_hit':mean([r['h2d_kv_bytes'] for r in hit]),'compact_KV_bytes_per_hit':mean([r['gpu_cache_kv_bytes'] for r in hit]),'extra_real_bytes_per_hit':mean([r['extra_valid_visual_rows']*57344 for r in hit]),'padding_bytes_per_hit':mean([r['padding_rows_read']*57344 for r in hit])}
  full=sum((math.ceil(r['N_content']/64)*64+r['S_structural'])*57344 for r in hit)
  b['total_read_ratio_vs_FullLoad_pooled']=sum(r['ssd_read_bytes'] for r in hit)/full
  b['total_read_ratio_image_mean']=mean([r['ssd_read_bytes']/((math.ceil(r['N_content']/64)*64+r['S_structural'])*57344) for r in ims]);budget.append(b)
 return summary,scope,budget

def paired(rows):
 hit=[r for r in rows if r['turn_id']>1];ims=sorted({r['image_id'] for r in hit});mapping={im:i for i,im in enumerate(ims)};lookup={(r['dialog_id'],r['turn_id'],r['method_id']):r for r in hit}
 sample=np.random.default_rng(1234).integers(0,len(ims),size=(10000,len(ims)));out=[]
 for baseline in METHODS[:2]:
  totals=np.zeros((len(ims),7))
  for b in hit:
   if b['method_id']!=baseline:continue
   o=lookup[(b['dialog_id'],b['turn_id'],METHODS[2])];totals[mapping[b['image_id']]]+=[o['score']-b['score'],o['ttft_ms'],b['ttft_ms'],o['ssd_read_bytes'],b['ssd_read_bytes'],1,o['ssd_read_bytes']-b['ssd_read_bytes']]
  res=totals[sample].sum(1);total=totals.sum(0)
  distributions={'delta_hit_acc_pp':res[:,0]/res[:,5]*100,'delta_hit_TTFT_ms':(res[:,1]-res[:,2])/res[:,5],'TTFT_reduction_percent':100*(1-res[:,1]/res[:,2]),'speedup':res[:,2]/res[:,1],'delta_SSD_bytes':res[:,6]/res[:,5]}
  points={'delta_hit_acc_pp':total[0]/total[5]*100,'delta_hit_TTFT_ms':(total[1]-total[2])/total[5],'TTFT_reduction_percent':100*(1-total[1]/total[2]),'speedup':total[2]/total[1],'delta_SSD_bytes':total[6]/total[5]}
  if total[4]:distributions['SSD_reduction_percent']=100*(1-res[:,3]/res[:,4]);points['SSD_reduction_percent']=100*(1-total[3]/total[4])
  else:points['SSD_reduction_percent']=None
  record={'baseline':baseline,'direction':'Ours-minus-baseline','images':len(ims),'paired_hits':int(total[5]),'resamples':10000,'seed':1234,**points}
  for k,v in distributions.items():record[k+'_ci_low'],record[k+'_ci_high']=[float(a) for a in np.percentile(v,[2.5,97.5])]
  out.append(record)
 return out

def sessions(run,rows):
 setups={};details=[];stages=[]
 for cp in (run/'main/images').glob('*/COMMITTED.json'):
  com=json.loads(cp.read_text());folder=cp.parent/com['attempt_id'];iid=cp.parent.name
  for m in METHODS:
   if m=='recompute':setups[iid,m]={'source_dialog_id':None,'setup_ms':0,'write_bytes':0,'capture_tail_ms':0,'writer_ms':0,'activation_ms':0};continue
   p=json.loads((folder/f'{m}_persistence.json').read_text());a=json.loads((folder/f'{m}_activation.json').read_text())
   setups[iid,m]={'source_dialog_id':p['source']['dialog_id'],'setup_ms':p['charged_setup_before_activation_ms']+a['activation_ms'],'write_bytes':p['bytes_written'],'capture_tail_ms':p['post_generation_capture_tail_ms'],'writer_ms':p['outer_writer_wall_ms'],'activation_ms':a['activation_ms']}
   stages.append({'image_id':iid,'method_id':m,**setups[iid,m],'capture_score_ms':p['capture_score_ms'],'capture_clone_ms':p['capture_clone_ms'],**p['timing_ms'],'metadata_resident_bytes':a['metadata_resident_bytes'],'activation_bytes':a['activation_io']['bytes'],'scope':'score and clone inside measured post-generation tail; materialize/repack/write/fsync nested in writer; not additive with enclosing timers'})
 grouped=defaultdict(list)
 for r in rows:grouped[r['image_id'],r['dialog_id'],r['method_id']].append(r)
 for (iid,did,m),rs in grouped.items():
  s=setups[iid,m];source=did==s['source_dialog_id'];request=sum(r['request_e2e_ms'] for r in rs)
  details.append({'image_id':iid,'dialog_id':did,'method_id':m,'request_E2E_sum_ms':request,'source_dialogue':source,'actual_setup_ms':s['setup_ms'] if source else 0,'actual_stream_session_E2E_ms':request+(s['setup_ms'] if source else 0),'derived_standalone_E2E_ms':request+s['setup_ms'],'standalone_scope':'DERIVED reassign measured image setup to every dialogue; source hooks remain in observed T1, incremental hook overhead for other dialogues not separately measured'})
 summary=[]
 for m in METHODS:
  ds=[r for r in details if r['method_id']==m];ss=[s for (i,mm),s in setups.items() if mm==m]
  if ds:summary.append({'method_id':m,'provisioning':'FRESH_IMAGE_STREAMING','status':'MEASURED','setup_ms_per_image':mean([s['setup_ms'] for s in ss]),'capture_tail_ms_per_image':mean([s['capture_tail_ms'] for s in ss]),'writer_ms_per_image':mean([s['writer_ms'] for s in ss]),'activation_ms_per_image':mean([s['activation_ms'] for s in ss]),'write_MB_per_image':mean([s['write_bytes']/1e6 for s in ss]),'actual_stream_session_E2E_mean_ms':mean([r['actual_stream_session_E2E_ms'] for r in ds]),'derived_standalone_E2E_mean_ms':mean([r['derived_standalone_E2E_ms'] for r in ds])})
 return details,summary,stages

def cross_model(manifest,summary,budget,checks):
 binding=manifest['llava_binding'];run=Path(binding['run_dir']);receipts=[];rows=[];physical=set();logical=set()
 for rel,digest in binding['files'].items():checks.require(sha(ROOT/rel)==digest,'llava_frozen_source_artifact',rel)
 lm=json.loads((run/'manifest.json').read_text());checks.require(lm['dialogues']==manifest['dialogues'],'cross_model_ordered_identity')
 for iid,x in manifest['image_identity'].items():checks.require(x['sha256']==lm['image_identity'][iid]['sha256'],'cross_model_image_identity',iid)
 # Recalculate common-method summaries from committed main raw, no previous summary numbers.
 count=0
 for cp in sorted((run/'main/images').glob('*/COMMITTED.json')):
  commit=json.loads(cp.read_text());raw=cp.parent/commit['attempt_id']/'raw.jsonl';digest=sha(raw);checks.require(digest==commit['raw_sha256'],'llava_main_raw_hash',str(raw));receipts.append({'path':str(raw),'sha256':digest})
  for r in read_rows(raw):
   count+=1;checks.require(r['physical_execution_id'] not in physical and r['logical_request_id'] not in logical,'llava_unique_raw_ids',r['logical_request_id']);physical.add(r['physical_execution_id']);logical.add(r['logical_request_id'])
   if r['method_id'] not in ('recompute','fullload','ours_kv25') or r['turn_id']==1:continue
   checks.require(r['score']==score(r['prediction'],r['gold']),'llava_independent_rescore',r['logical_request_id'])
   rows.append({k:r[k] for k in ('method_id','image_id','dialog_id','turn_id','score','ttft_ms','ssd_read_bytes','N_content','k')})
 checks.require(count==60915 and len(receipts)==398,'llava_5arm_complete')
 table=[]
 for m in ('recompute','fullload','ours_kv25'):
  rs=[r for r in rows if r['method_id']==m];checks.require(len(rs)==8122,'llava_common_hit_population',m)
  table.append({'backbone':'LLaVA-1.6-Vicuna-7B','method':{'recompute':'ReComp','fullload':'FullLoad','ours_kv25':'Ours-KV25'}[m],'hit_acc':mean([r['score'] for r in rs]),'hit_ttft_mean_ms':mean([r['ttft_ms'] for r in rs]),'SSD_MB_per_hit':mean([r['ssd_read_bytes']/1e6 for r in rs]),'actual_content_retention_hit_pooled':sum(r['k'] for r in rs)/sum(r['N_content'] for r in rs),'run_id':run.name})
 for x in summary:
  b=next(b for b in budget if b['method_id']==x['method_id']);table.append({'backbone':'Qwen2.5-VL-7B','method':x['label'],'hit_acc':x['hit_acc'],'hit_ttft_mean_ms':x['hit_ttft_mean_ms'],'SSD_MB_per_hit':x['SSD_MB_per_hit'],'actual_content_retention_hit_pooled':b['content_retention_hit_token_pooled'],'run_id':'current Qwen run'})
 return table,{'binding':binding,'raw_hashes':receipts,'raw_requests':count,'dataset_identity':'same ordered full dialogues and image bytes','timing':'both true TTFT include pixel decode or hit prompt+SSD/H2D/prefill until first synchronized token; Qwen score/clone post-generation measured in setup; LLaVA capture exit included in request E2E; backend/precision/token geometry differ'}

def render(audit,summary,scope,budget,pairs,sess,cross,diagnostics):
 lines=['# Qwen2.5-VL MT-GQA generated-history KV25 본실험','',f"QWEN 3-ARM MAIN: **{audit['QWEN_3_ARM_MAIN']}**, 실제 main final **{audit['main_requests']:,}/36,549**.",'',
 'MT-GQA-reconstructed 398개 이미지·4,061개 3-turn 대화, 각 방법의 실제 이전 생성 답변만 history에 사용한다. Strict normalized exact match이며 공식 MetaCompress artifact/evaluator 재현을 주장하지 않는다.',
 '', '| Method | Acc T1 / T2 / T3 (%) | All Acc (%) | Hit Acc (%) | T1 TTFT (ms) | Hit TTFT mean / p50 / p95 (ms) | SSD MB/hit |','|---|---:|---:|---:|---:|---:|---:|']
 for s in summary:lines.append(f"| {s['label']} | {100*s['acc_T1']:.2f} / {100*s['acc_T2']:.2f} / {100*s['acc_T3']:.2f} | {100*s['all_acc']:.2f} | {100*s['hit_acc']:.2f} | {s['T1_ttft_mean_ms']:.3f} | {s['hit_ttft_mean_ms']:.3f} / {s['hit_ttft_p50_ms']:.3f} / {s['hit_ttft_p95_ms']:.3f} | {s['SSD_MB_per_hit']:.3f} |")
 lines+=['','정답 수와 모집단, 각 turn·all·hit의 TTFT/E2E mean/p50/p95는 `quality_timing_by_scope.csv`에 보존한다. T1은 세 방법 모두 full-image이므로 all-turn 평균은 hit의 품질 손실을 희석할 수 있다. ReComp T2/T3는 hit 비교 모집단에 포함되지만 SSD KV cache hit가 아니다.',
 '', '| Method | Budget | Mean N/image | Content kept % (image mean / token pooled) | Structural 포함 % | K / V / structural MB/hit | FullLoad 대비 total read | Preads/hit |','|---|---|---:|---:|---:|---:|---:|---:|']
 for b in budget:lines.append(f"| {LABELS[b['method_id']]} | {b['budget_unit']} | {b['mean_N_image']:.2f} | {100*b['content_retention_image_mean']:.3f} / {100*b['content_retention_image_token_pooled']:.3f} | {100*b['structural_inclusive_retention_image_mean']:.3f} | {b['K_bytes_per_hit']/1e6:.3f} / {b['V_bytes_per_hit']/1e6:.3f} / {b['structural_bytes_per_hit']/1e6:.3f} | {b['total_read_ratio_vs_FullLoad_pooled']:.4f} | {b['preads_per_hit']:.2f} |")
 lines+=['','Ours는 N개 content 위치 중 ceil(N/4)를 선택하며 모든 28 layer·4 native KV head의 K/V를 함께 보존한다. 64-row whole chunks를 읽고 extra/padding을 H2D 전에 제거한다. 3축 MRoPE는 원래 full logical 위치, cache slots는 기존 compact 정책을 유지한다. SSD에는 전체 원본 KV를 저장하므로 storage compression이 아니다. MB=10^6 B, GiB=2^30 B.',
 '', '| Ours − baseline | Δ Hit Acc (%p), 95% CI | Δ Hit TTFT (ms), 95% CI | TTFT 감소율 | Speedup | SSD 감소율 |','|---|---:|---:|---:|---:|---:|']
 for p in pairs:
  ssd='N/A (baseline KV read=0)' if p['SSD_reduction_percent'] is None else f"{p['SSD_reduction_percent']:.2f}%"
  lines.append(f"| {LABELS[p['baseline']]} | {p['delta_hit_acc_pp']:+.3f} [{p['delta_hit_acc_pp_ci_low']:+.3f}, {p['delta_hit_acc_pp_ci_high']:+.3f}] | {p['delta_hit_TTFT_ms']:+.3f} [{p['delta_hit_TTFT_ms_ci_low']:+.3f}, {p['delta_hit_TTFT_ms_ci_high']:+.3f}] | {p['TTFT_reduction_percent']:.2f}% | {p['speedup']:.3f}× | {ssd} |")
 lines+=['','Image-cluster bootstrap 10,000회, seed=1234, 이미지 내 모든 대화·turn·paired methods를 함께 추출하는 request-weighted estimator다. 감소율·speedup CI도 CSV에 있다. CI에 0이 포함되어도 equivalence를 주장하지 않는다. Generated history 차이를 포함한 method-level 누적 효과이며, 동일 텍스트에서 KV 선택만 바꾼 인과 비교가 아니다.',
 '', '| Method | Provisioning | Measured setup (ms/image) | Actual-stream session E2E (ms) | DERIVED standalone E2E (ms) |','|---|---|---:|---:|---:|']
 for s in sess:lines.append(f"| {LABELS[s['method_id']]} | {s['provisioning']} | {s['setup_ms_per_image']:.3f} | {s['actual_stream_session_E2E_mean_ms']:.3f} | {s['derived_standalone_E2E_mean_ms']:.3f} |")
 lines+=['','Fresh source T1의 정상 forward에서 각 방법 자신의 KV/score를 capture했다. 별도 vision/prefix forward를 provisioning에 추가하지 않았다. Score 후처리·KV clone·출력 후처리를 포함한 generation 종료→core return tail, CPU materialize/repack·write/fsync/publication을 포함한 writer wall, 전체 payload hash를 포함한 activation을 실측했다. Capture hook은 이미 T1 request 안에 있어 다시 더하지 않는다. 단계별 timer는 enclosing wall과 중첩되므로 합산하지 않는다.',
 'Actual stream은 source dialogue에만 해당 이미지 setup을 한 번 부과한다. Standalone은 실측 image setup을 각 dialogue에 재부과한 DERIVED 값이다. 각 dialogue의 관측 T1을 유지하며 non-source T1에 capture hook을 추가했을 때의 증분은 별도 측정하지 않았다. 모든 dialogue를 fresh로 독립 반복한 실측값이 아니며 background overlap을 가정하지 않는다.',
 '', 'True TTFT는 history/prompt 준비 전부터 시작해 pixel 요청의 실제 파일 open/RGB decode·processor·vision/full prefill 또는 hit의 tokenization·SSD pread·H2D/compact assembly·suffix prefill을 거쳐 첫 token materialization+CUDA sync까지다. Request E2E는 generation 완료까지다. Image hash, 모델 load/warmup, activation, conditioning, diagnostic audit/hash/logging은 요청 timer 밖이다. Core와 outer TTFT를 raw에 별도 보존했다.',
 '매 hit 전 57개 payload 파일의 POSIX_FADV_DONTNEED 성공을 기록한다. 이는 OS page-cache hint이며 NAND/controller cold나 O_DIRECT를 뜻하지 않는다. Actual OS-returned KV bytes/preads만 측정했다. ReComp KV 읽기 0은 raw image I/O 0이 아니다. 작은 metadata/FD는 상주하며 요청 사이 visual payload 재사용은 없다.',
 '', '| Backbone | Method | Hit Acc (%) | Hit TTFT (ms) | SSD MB/hit | Actual content retention (%) |','|---|---|---:|---:|---:|---:|']
 for x in cross:lines.append(f"| {x['backbone']} | {x['method']} | {100*x['hit_acc']:.2f} | {x['hit_ttft_mean_ms']:.3f} | {x['SSD_MB_per_hit']:.3f} | {100*x['actual_content_retention_hit_pooled']:.3f} |")
 lines+=['','Cross-model 표는 완료된 최신 LLaVA 5-arm main의 원본 committed raw를 읽기 전용으로 재계산한 공통 3방법 secondary 표다. Run ID와 manifest/audit/raw hashes는 `cross_model_provenance.json`에 있다. 데이터 ordered identity와 image SHA가 같다. Qwen은 BF16/SDPA/compact cache, LLaVA는 FP16 SSD/eager/다른 visual geometry이며 persistence capture의 E2E 포함 범위도 다르다. 절대 latency 차이를 저장 기법 하나의 효과로 해석하지 않는다. Qwen AllHead/MPIC/ReKV 포팅과 LLaVA GPU 재실행, MT-VQA는 하지 않았다.',
 '', '이미지별 N/retention/읽기 비율 분포, generated length·cap, peak GPU allocated/reserved, metadata/compact bytes, T2/T3 선택 불변성, ReComp/FullLoad 출력 일치는 `diagnostics.json`, `request_diagnostics.csv`, `image_geometry.csv`에 있다. 단일 GPU 순차 image streaming이며 concurrent throughput/메모리 포화 실험이 아니다.',
 '', f"DATASET IDENTITY / METHOD-CONFIG CONTRACT: {audit['dataset_contract']}",f"GPU INTEGRATION UNDER v2: {audit['gpu_integration']}",f"SMOKE: {audit['phase_counts']['smoke']}/36, {audit['smoke_status']}",f"QWEN 3-ARM MAIN: {audit['QWEN_3_ARM_MAIN']}; {audit['main_requests']}/36549",f"INDEPENDENT AUDIT: {audit['status']}; PROTECTED ARTIFACTS: {audit['protection']['status']}",
 'PERSISTENCE/SESSION: fresh MEASURED source-only setup; standalone DERIVED.',f"CROSS-MODEL TABLE READY: {audit['CROSS_MODEL_TABLE_READY']}",
 f"기존 source 변경 {len(audit['protection']['changed_existing_source'])}건, artifact 변경 {len(audit['protection']['changed_artifacts'])}건. 큰 기존 파일은 inode/size/mtime/ctime+9-window fingerprint이므로 full-byte SHA256 증명으로 과장하지 않는다. 새 run raw/store 기록은 전체 SHA256을 사용한다."]
 return '\n'.join(lines)+'\n'

def run_audit(run,output):
 output.mkdir(parents=True,exist_ok=True)
 if (output/'independent_audit.json').exists():raise FileExistsError('new output directory required')
 config=json.loads((run/'config.json').read_text());config.update(_file_sha256=sha(run/'config.json'),_manifest_sha256=sha(run/'manifest.json'));manifest=json.loads((run/'manifest.json').read_text());c=Checks()
 rows=[];receipts=[];rawhashes=[];attempts=Counter();counts={};ids=set();physical=set();geometries=[]
 for phase in ('integration','smoke','main'):
  count=0
  for cp in sorted((run/phase/'images').glob('*/COMMITTED.json')):
   com=json.loads(cp.read_text());folder=cp.parent/com['attempt_id'];raw=folder/'raw.jsonl'
   for name,digest in com['evidence_sha256'].items():c.require(sha(folder/name)==digest,'committed_evidence_hash',str(folder/name))
   rs=list(read_rows(raw));receipt=audit_image_rows(rs,manifest,config,phase,cp.parent.name,folder);receipts.append(receipt);c.require(receipt['status']=='PASS','independent_image_audit',receipt);count+=len(rs)
   c.require(com['requests']==len(rs) and sha(raw)==com['raw_sha256'],'commit_rows_and_raw_hash',str(cp));c.require(json.loads((folder/'cleanup_receipt.json').read_text())['status']=='PASS','cleanup_receipt',str(folder))
   allow=json.loads((folder/'cleanup_allowlist.json').read_text());scratch=Path(allow['scratch']);creation=json.loads((folder/'scratch_creation.json').read_text())
   c.require(not scratch.exists() and scratch.is_relative_to(run/'scratch') and allow['ownership_run']==run.name and creation['previously_absent'] and creation['path']==str(scratch),'only_owned_scratch_cleanup',str(folder))
   events=list(read_rows(folder/'attempts.jsonl'));started=[e['physical_execution_id'] for e in events if e['status']=='STARTED'];completed=[e['physical_execution_id'] for e in events if e['status']=='COMPLETED']
   c.require(Counter(started)==Counter(completed)==Counter(r['physical_execution_id'] for r in rs),'attempt_event_lineage',str(folder))
   rawhashes.append({'phase':phase,'image_id':cp.parent.name,'path':str(raw),'sha256':sha(raw)})
   if phase=='main':
    for r in rs:c.require(r['logical_request_id'] not in ids and r['physical_execution_id'] not in physical,'main_unique',r['logical_request_id']);ids.add(r['logical_request_id']);physical.add(r['physical_execution_id']);rows.append(compact(r))
    for m in METHODS[1:]:
     r=next(r for r in rs if r['turn_id']==2 and r['method_id']==m);v=r['result'];geometries.append({'image_id':r['image_id'],'method_id':m,'N':r['N_content'],'k':r['k'],'k_over_N':r['k']/r['N_content'],'S':r['S_structural'],'whole_chunks':v['selected_chunks'],'extra_real_rows':v['extra_valid_visual_rows'],'padding_rows':v['padding_rows_read'],'read_bytes':r['ssd_read_bytes'],'full_read_bytes':((r['N_content']+63)//64*64+r['S_structural'])*57344,'read_ratio':v['total_payload_read_ratio'],'selection_unique_count':len({canonical(x['result']['selected_visual_original']) for x in rs if x['method_id']==m and x['turn_id']>1})})
  counts[phase]=count
  for a in (run/phase/'images').glob('*/attempt_*'):
   attempts[phase+'_image_attempts']+=1
   if (a/'failure.json').exists():attempts[phase+'_failed_attempts']+=1
   if (a/'raw.jsonl').exists():attempts[phase+'_completed_physical_rows']+=sum(1 for _ in read_rows(a/'raw.jsonl'))
   if (a/'execution_events.jsonl').exists():attempts[phase+'_executed_physical_requests']+=sum(1 for _ in read_rows(a/'execution_events.jsonl'))
 complete=counts['main']==36549
 c.require(complete and len({r['image_id'] for r in rows})==398 and len({r['dialog_id'] for r in rows})==4061,'main_full_population')
 for m in METHODS:
  rs=[r for r in rows if r['method_id']==m];c.require(len(rs)==12183 and sum(r['turn_id']>1 for r in rs)==8122,'per_method_population',m)
 c.require(counts['smoke']==36 and counts['integration']==36,'separate_validation_smoke_population')
 integration=json.loads((run/'integration_validation.json').read_text());c.require(integration['status']=='PASS' and integration['source_sha256']==config['source_sha256'],'integration_gate')
 c.require(json.loads((ROOT/'data/mt_gqa/dialogues.json').read_text())['dialogues']==manifest['dialogues'] and sha(ROOT/'data/mt_gqa/dialogues.json')==manifest['index_sha256'],'frozen_full_dataset')
 for name,digest in config['source_sha256'].items():c.require(sha(ROOT/name)==digest,'frozen_runtime_source',name)
 for iid,x in manifest['image_identity'].items():c.require(sha(ROOT/x['image_path'])==x['sha256'],'final_external_image_sha',iid)
 protect=protection(run);c.require(protect['status']=='PASS','existing_artifact_protection')
 summary,scope,budget=summaries(rows);pairs=paired(rows);details,sess,stages=sessions(run,rows)
 crosschecks=Checks();cross,crossprov=cross_model(manifest,summary,budget,crosschecks);crossok=crosschecks.result()['status']=='PASS'
 result=c.result();valid=complete and result['status']=='PASS'
 result.update(QWEN_3_ARM_MAIN='VALID UNDER v2' if valid else 'INVALID' if complete else 'PARTIAL',main_requests=counts['main'],expected_main_requests=36549,phase_counts=counts,attempt_counts=dict(attempts),dataset_contract='PASS' if result['status']=='PASS' else 'FAIL',gpu_integration=integration['status'],smoke_status='PASS' if counts['smoke']==36 else 'FAIL',protection=protect,CROSS_MODEL_TABLE_READY='YES' if crossok and valid else 'NO',cross_model_audit=crosschecks.result(),image_receipts=receipts,source_sha256=config['source_sha256'],llava_gpu='NOT RUN',MT_VQA='NOT RUN')
 lookup={(r['dialog_id'],r['turn_id'],r['method_id']):r for r in rows};agreement={}
 for m in METHODS:
  rs=[r for r in rows if r['method_id']==m];hit=[r for r in rs if r['turn_id']>1]
  agreement[m]={'generated_length_min':min(r['generated_token_count'] for r in rs),'generated_length_max':max(r['generated_token_count'] for r in rs),'generated_length_mean':mean([r['generated_token_count'] for r in rs]),'cap_count':sum(r['cap_reached'] for r in rs),'peak_allocated_bytes':max(r['peak_gpu_allocated_bytes'] for r in rs),'peak_reserved_bytes':max(r['peak_gpu_reserved_bytes'] for r in rs),'hit_peak_allocated_mean':mean([r['peak_gpu_allocated_bytes'] for r in hit]),'hit_peak_reserved_mean':mean([r['peak_gpu_reserved_bytes'] for r in hit])}
 full=[r for r in rows if r['method_id']=='fullload'];diagnostics={'main_final_rows':len(rows),'duplicate_logical':len(rows)-len(ids),'duplicate_physical':len(rows)-len(physical),'attempts':dict(attempts),'methods':agreement,'ReComp_FullLoad_prediction_agreement':sum(r['prediction']==lookup[r['dialog_id'],r['turn_id'],'recompute']['prediction'] for r in full),'ReComp_FullLoad_generated_ids_agreement':sum(r['generated_token_ids']==lookup[r['dialog_id'],r['turn_id'],'recompute']['generated_token_ids'] for r in full),'ReComp_FullLoad_pairs':len(full),'ours_selection_all_images_invariant':all(g['selection_unique_count']==1 for g in geometries if g['method_id']==METHODS[2]),'image_N_min':min(g['N'] for g in geometries),'image_N_max':max(g['N'] for g in geometries),'bootstrap_resamples':10000,'seed':1234}
 for name,data in [('summary',summary),('quality_timing_by_scope',scope),('budget_io',budget),('paired_comparisons',pairs),('paired_quality',pairs),('paired_latency',pairs),('session_per_dialogue',details),('session_summary',sess),('persistence_stage_scope',stages),('request_diagnostics',rows),('image_geometry',geometries),('cross_model_common_3_methods',cross)]:csv_write(output/(name+'.csv'),data)
 dump(output/'independent_audit.json',result);dump(output/'protection_final.json',protect);dump(output/'raw_hashes.json',rawhashes);dump(output/'cross_model_provenance.json',crossprov);dump(output/'diagnostics.json',diagnostics)
 (output/'REPORT.md').write_text(render(result,summary,scope,budget,pairs,sess,cross,diagnostics))
 for name in ('config.json','manifest.json','storage_plan.json','environment_before.json','integration_validation.json','REPRODUCE.md'):shutil.copy2(run/name,output/name)
 terminal={'execution_state':'FINISHED' if valid else 'AUDIT_FAILED','QWEN_3_ARM_MAIN':result['QWEN_3_ARM_MAIN'],'main_final_requests':counts['main'],'expected_main_requests':36549,'independent_audit':result['status'],'CROSS_MODEL_TABLE_READY':result['CROSS_MODEL_TABLE_READY'],'at_unix':time.time(),'report':str(output/'REPORT.md'),'audit_sha256':sha(output/'independent_audit.json')}
 for parent in (run,output.parent):
  temp=parent/'.FINAL_STATUS.tmp';temp.write_text(json.dumps(terminal,ensure_ascii=False,indent=2));os.replace(temp,parent/'FINAL_STATUS.json')
  temp=parent/'.CURRENT_STATUS.tmp';temp.write_text(json.dumps(terminal,ensure_ascii=False,indent=2));os.replace(temp,parent/'CURRENT_STATUS.json')
 (output.parent/'PROGRESS.md').write_text(f"# Qwen MT-GQA 최종 상태\n\n{result['QWEN_3_ARM_MAIN']}: {counts['main']:,}/36,549, 독립 감사 {result['status']}.\n\n[한국어 최종 보고서](final/REPORT.md)\n")
 print(json.dumps(terminal),flush=True)
 if not valid:raise RuntimeError('final independent audit did not validate full main')
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();run_audit(a.run_dir.resolve(),a.output_dir.resolve())
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Qwen MT-GQA three-arm main: unchanged native BF16 production, image-atomic commits."""
from __future__ import annotations
import argparse
from collections import OrderedDict
from contextlib import contextmanager
import gc, hashlib, importlib.util, json, os, re, shutil, subprocess, sys, time, traceback, uuid
from pathlib import Path
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mmimpress.qwen25.runner import Qwen25Runner, MODEL_ID, CHECKPOINT_REVISION
METHODS=('recompute','fullload','qwen_ours_kv25')
SCHEMA='qwen-mtgqa-kv25-main-v1'
CONTRACT=ROOT/'docs/qwen_mt_gqa_kv25_main_contract.md'
INDEX_SHA='2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224'
WORKLOAD_SHA='0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62'

def helper(name,file):
 if name in sys.modules:return sys.modules[name]
 spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
 mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def safe(x):
 if torch.is_tensor(x):return x.detach().cpu().tolist()
 if isinstance(x,Path):return str(x)
 if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [safe(v) for v in x]
 return x

def sync_dir(path):
 fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)

def atomic_json(path,value,replace=False):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists() and not replace:raise FileExistsError(path)
 tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
 with tmp.open('x') as f:json.dump(safe(value),f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 if replace:os.replace(tmp,path)
 else:os.link(tmp,path);tmp.unlink()
 sync_dir(path.parent)

def append(path,row):
 with Path(path).open('a') as f:f.write(json.dumps(safe(row),ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
def read_rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def method_order(ordinal):
 i=ordinal%3;return METHODS[i:]+METHODS[:i]
def logical_id(run,phase,method,did,t):return f'{SCHEMA}:{run.name}:{phase}:{MODEL_ID}:{method}:{did}:t{t}'
def strict_score(pred,gold):
 def norm(v):return ' '.join(w for w in re.sub(r'[^\w\s]',' ',str(v).lower()).split() if w not in {'a','an','the'})
 return float(norm(pred)==norm(gold))

def history_factory(dialog,turn,method,generated):
 if method not in METHODS or turn not in (1,2,3) or set(generated)!=set(range(1,turn)):raise ValueError('future/missing history')
 entries=[];history=[]
 for t in range(1,turn):
  row=generated[t];q=dialog['turns'][t-1]
  if (row['model_id'],row['method_id'],row['dialog_id'],row['turn_id'])!=(MODEL_ID,method,dialog['dialog_id'],t):raise ValueError('foreign history')
  history.append((q['question'],row['prediction']))
  entries.append({'turn_id':t,'question_id':str(q['question_id']),'question':q['question'],'answer':row['prediction'],
   'source_model':MODEL_ID,'source_method_id':method,'source_dialog_id':dialog['dialog_id'],
   'source_logical_request_id':row['logical_request_id'],'source_physical_execution_id':row['physical_execution_id']})
 return dialog['turns'][turn-1]['question'],tuple(history),entries

def source_hashes():
 names=list((ROOT/'mmimpress').rglob('*.py'))
 names += [ROOT/'scripts'/n for n in ('78_validate_qwen25.py','81_debug_qwen25_correctness.py','86_validate_qwen25_v2_rerun.py','91_validate_qwen25_kv25.py','103_eval_qwen_mt_gqa_kv25_main.py','104_audit_qwen_mt_gqa_kv25_main.py','105_validate_qwen_mt_gqa_kv25_main.py')]
 return {str(p.relative_to(ROOT)):sha(p) for p in sorted(names)}

def assert_gpu_exclusive():
 p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
 other=[s for s in p.stdout.splitlines() if s.strip() and int(s.split(',')[0])!=os.getpid()]
 if other:raise RuntimeError('GPU_TIMING_CONTAMINATION '+repr(other))
 return p.stdout

@contextmanager
def conditioning_trace():
 orig=os.posix_fadvise;events=[]
 def call(fd,off,n,advice):
  path=os.readlink(f'/proc/self/fd/{fd}')
  try:r=orig(fd,off,n,advice);events.append({'path':path,'offset':off,'length':n,'advice':advice,'status':'SUCCESS'});return r
  except BaseException as e:events.append({'path':path,'status':'FAIL','error':repr(e)});raise
 os.posix_fadvise=call
 try:yield events
 finally:os.posix_fadvise=orig

class RequestTrace:
 """Lightweight observation only. Tensor serialization is after all timed endpoints."""
 def __init__(self,runner,store=None):self.r=runner;self.store=store
 def __enter__(self):
  from mmimpress.qwen25.vision import VisionScoreCapture
  self.originals={n:getattr(self.r,n) for n in ('_decode','_logical_positions','_chat_text')}
  self.score_orig=VisionScoreCapture._compute_scores;self.pread_orig=os.pread
  self.diag={};self.reads=[];self.score_calls=0;self.vision_calls=0;self.model_calls=0;self.handles=[]
  self.fd_names={fd:str(self.store.path/rel) for rel,fd in self.store._fds.items()} if self.store else {}
  def positions(ids,grid):
   pos,delta=self.originals['_logical_positions'](ids,grid)
   self.diag.update(input_ids_tensor=ids,position_ids_tensor=pos,rope_deltas_tensor=delta,grid_tensor=grid)
   return pos,delta
  def chat(q,h):
   text=self.originals['_chat_text'](q,h);self.diag['prompt']=text;return text
  def decode(token,cache,*,position_start,attention_mask):
   self.first_at=time.perf_counter();self.diag['decode_start_position']=position_start;self.diag['cache_before_decode']=int(cache.get_seq_length())
   result=self.originals['_decode'](token,cache,position_start=position_start,attention_mask=attention_mask)
   self.end_at=time.perf_counter();self.diag['cache_after_decode']=int(cache.get_seq_length());return result
  def pread(fd,count,offset):
   payload=self.pread_orig(fd,count,offset)
   self.reads.append({'path':self.fd_names.get(fd,f'UNKNOWN_FD_{fd}'),'offset':offset,'requested':count,'returned':len(payload)});return payload
  def score(cap):self.score_calls+=1;return self.score_orig(cap)
  def vision(*a):self.vision_calls+=1
  def model_post(mod,args,kwargs,output):
   if self.model_calls==1:self.first_logits_for_finite=output.logits[0,-1].detach()
  def model_pre(mod,args,kwargs):
   self.model_calls+=1
   if self.model_calls==1:
    cache=kwargs.get('past_key_values')
    if cache is not None:
     self.diag['layer_prefix_shapes']=[[list(l.keys.shape),list(l.values.shape)] for l in cache.layers]
    else:self.diag['layer_prefix_shapes']=[]
    self.diag['actual_prefill_input_ids_tensor']=kwargs['input_ids'].detach()
    self.diag['actual_prefill_position_ids_tensor']=kwargs['position_ids'].detach()
    self.diag['actual_prefill_cache_position_tensor']=kwargs['cache_position'].detach()
    self.diag['actual_prefill_attention_mask_tensor']=kwargs['attention_mask'].detach()
  self.r._logical_positions=positions;self.r._chat_text=chat;self.r._decode=decode
  os.pread=pread;VisionScoreCapture._compute_scores=score
  self.handles=[self.r.model.visual.register_forward_pre_hook(vision),self.r.model.register_forward_pre_hook(model_pre,with_kwargs=True),self.r.model.register_forward_hook(model_post,with_kwargs=True)]
  return self
 def __exit__(self,*exc):
  from mmimpress.qwen25.vision import VisionScoreCapture
  for h in self.handles:h.remove()
  for n,f in self.originals.items():setattr(self.r,n,f)
  os.pread=self.pread_orig;VisionScoreCapture._compute_scores=self.score_orig
  self.r._clear_request_state()

def execute_request(runner,image_path,image_sha,factory,method,turn,store_path=None,capture=False,return_logits=False):
 pixel=turn==1 or method=='recompute';conditioning=None;events=[];store=None
 if not pixel:
  ratio=1. if method=='fullload' else .25;unit='chunk' if method=='fullload' else 'visual_kv'
  with conditioning_trace() as events:
   c0=time.perf_counter();conditioning=runner.condition_cache(store_path,ratio,budget_unit=unit,image_sha256=image_sha);conditioning['outer_wall_ms']=(time.perf_counter()-c0)*1000
  assert conditioning['failures']==0 and len(events)==57 and all(e['status']=='SUCCESS' for e in events)
  store=runner._stores[str(Path(store_path).resolve())]
 torch.cuda.synchronize(runner.device)
 with RequestTrace(runner,store) as trace:
  started=time.perf_counter();question,history,entries=factory();image_decode=None
  if pixel:
   t0=time.perf_counter()
   with Image.open(image_path) as im:image=im.convert('RGB');image.load()
   image_decode=(time.perf_counter()-t0)*1000
   result=runner.run_pixels(image,question,history=history,capture=capture,image_sha256=image_sha,return_logits=return_logits)
  else:result=runner.run_cache(store_path,question,history=history,budget_ratio=ratio,budget_unit=unit,image_sha256=image_sha,return_logits=return_logits)
  returned=time.perf_counter()
 assert trace.vision_calls==int(pixel) and trace.score_calls==int(capture=='with_score')
 assert runner.model.model.rope_deltas is None
 assert len(result['generated_token_ids'])==trace.model_calls
 assert trace.diag['cache_after_decode']==trace.diag['cache_before_decode']+len(result['generated_token_ids'])-1
 assert bool(torch.isfinite(trace.first_logits_for_finite).all()),'nonfinite first logits'
 core_ttft=result['ttft_ms'];core_e2e=result['request_e2e_ms']
 result.update(core_ttft_ms=core_ttft,core_request_e2e_ms=core_e2e,ttft_ms=1000*(trace.first_at-started),request_e2e_ms=1000*(trace.end_at-started),
  request_started_at_s=started,first_token_at_s=trace.first_at,request_finished_at_s=trace.end_at,core_returned_at_s=returned,
  post_generation_tail_ms=(returned-trace.end_at)*1000,image_file_decode_ms=image_decode,
  first_logits_finite=True,actual_vision_calls=trace.vision_calls,actual_score_calls=trace.score_calls,model_forward_calls=trace.model_calls,
  os_pread_trace=trace.reads,conditioning=conditioning,conditioning_events=events,request_state_restored=True,capture_mode=capture)
 # Preserve input/output evidence after the timer, not hidden/logit full dumps in main.
 diag={k.removesuffix('_tensor'):safe(v) for k,v in trace.diag.items()}
 diag.update(history=safe(history),history_entries=entries)
 if not pixel:
  assert sum(e['returned'] for e in trace.reads)==result['read_io']['bytes']
  assert len(trace.reads)==result['pread_calls']
 return result,diag

def file_hashes(path):
 out={}
 for p in sorted(Path(path).rglob('*')):
  if p.is_symlink():raise ValueError('symlink in owned scratch')
  if p.is_file():
   if p.stat().st_nlink!=1:raise ValueError('hardlink in owned scratch')
   out[str(p.relative_to(path))]=sha(p)
 return out

def cleanup_committed(scratch,allowlist,audit,commit,run=None):
 scratch=Path(scratch)
 if audit.get('status')!='PASS' or commit.get('status')!='COMMITTED':raise ValueError('audit+commit required')
 if scratch.is_symlink() or not scratch.is_dir():raise ValueError('unsafe scratch root')
 resolved=scratch.resolve()
 if run is not None and not resolved.is_relative_to((Path(run)/'scratch').resolve()):raise ValueError('outside own scratch')
 if sorted(str(p.relative_to(scratch)) for p in scratch.rglob('*') if p.is_file() or p.is_symlink())!=sorted(allowlist):raise ValueError('allowlist mismatch')
 for rel,digest in allowlist.items():
  p=scratch/rel
  if p.is_symlink() or p.stat().st_nlink!=1 or not p.resolve().is_relative_to(resolved) or sha(p)!=digest:raise ValueError('unsafe or changed payload')
 for rel in allowlist:(scratch/rel).unlink()
 for p in sorted(scratch.rglob('*'),key=lambda p:len(p.parts),reverse=True):
  if p.is_dir():p.rmdir()
 scratch.rmdir();sync_dir(scratch.parent)

def capacity_guard(run):
 p=json.loads((run/'storage_plan.json').read_text());free=shutil.disk_usage(run).free
 if free<p['minimum_reserve_bytes']+p['peak_working_set_bytes']:raise RuntimeError(f'BLOCKED_STORAGE free={free}')
 owned=sum(x.stat().st_size for x in (run/'scratch').rglob('*') if x.is_file() and not x.is_symlink())
 if owned+p['largest_image_bundle_bound_bytes']>p['maximum_new_payload_bytes']:raise RuntimeError('BLOCKED_STORAGE incomplete scratch allowance')
 return {'free_bytes':free,'retained_scratch_bytes':owned}

def make_row(result,diag,method,dialog,turn,phase,run,attempt,physical,ordinal,config,image_sha,store_info=None,meta=None):
 q=dialog['turns'][turn-1];pixel=turn==1 or method=='recompute'
 n=result['geometry']['visual_count'] if pixel else meta['visual_count'];s=result['geometry']['prefix_len']-n if pixel else meta['structural_count']
 k=n if pixel or method=='fullload' else (n+3)//4
 clean={k:v for k,v in result.items() if k not in ('capture','first_logits')}
 return {'schema_version':SCHEMA,'experiment_id':run.name,'phase':phase,'attempt_id':attempt,'status':'PASS',
  'logical_request_id':logical_id(run,phase,method,dialog['dialog_id'],turn),'physical_execution_id':physical,
  'model_id':MODEL_ID,'model_revision':CHECKPOINT_REVISION,'method_id':method,'image_id':dialog['image_id'],'dialog_id':dialog['dialog_id'],'turn_id':turn,
  'question_id':str(q['question_id']),'question':q['question'],'gold':q['answers'],'prediction':result['prediction'],'score':strict_score(result['prediction'],q['answers'][0]),
  'history':diag['history'],'history_entries':diag['history_entries'],'history_sha256':canonical(diag['history']),
  'prompt':diag['prompt'],'prompt_sha256':hashlib.sha256(diag['prompt'].encode()).hexdigest(),'input_ids':diag['input_ids'],'input_ids_sha256':canonical(diag['input_ids']),
  'request_trace':diag,'config_sha256':config['_file_sha256'],'manifest_sha256':config['_manifest_sha256'],'code_sha256':canonical(config['source_sha256']),
  'image_sha256':image_sha,'global_dialog_ordinal':ordinal,'method_order':list(method_order(ordinal)),'method_order_position':method_order(ordinal).index(method),
  'N_content':n,'S_structural':s,'k':k,'content_kv_fraction':k/n,'structural_inclusive_kept_ratio':(k+s)/(n+s),'budget_unit':'visual_kv' if method==METHODS[2] else 'full_context',
  'request_path':'normal_pixels' if pixel else 'ssd_cache_hit','provisioning_mode':'FRESH_IMAGE_STREAMING',
  'source_store':store_info if not pixel else None,'ttft_ms':result['ttft_ms'],'request_e2e_ms':result['request_e2e_ms'],
  'ssd_read_bytes':sum(e['returned'] for e in result['os_pread_trace']),'ssd_preads':len(result['os_pread_trace']),
  'result':safe(clean)}

def run_image(runner,dialogs,ordinals,phase,run,manifest,config,diagnostic=None):
 iid=dialogs[0]['image_id'];parent=run/phase/'images'/iid;parent.mkdir(parents=True,exist_ok=True)
 if (parent/'COMMITTED.json').exists():raise FileExistsError('already committed')
 attempt=f'attempt_{len(list(parent.glob("attempt_*")))+1:04d}';artifact=parent/attempt;artifact.mkdir()
 scratch=run/'scratch'/phase/iid/attempt
 atomic_json(artifact/'capacity_before.json',capacity_guard(run));assert not scratch.exists()
 atomic_json(artifact/'scratch_creation.json',{'path':str(scratch),'ownership':run.name,'previously_absent':True});scratch.mkdir(parents=True)
 paths={m:scratch/m for m in METHODS[1:]};metas={};hashes={};info={};captures={};rows=[];success=False
 image_path=ROOT/dialogs[0]['image_path'];image_sha=sha(image_path)
 assert image_sha==manifest['image_identity'][iid]['sha256']
 atomic_json(artifact/'image_identity.json',{'image_id':iid,'image_path':str(image_path),'sha256':image_sha})
 try:
  for di,dialog in enumerate(dialogs):
   generated={m:{} for m in METHODS};ordinal=ordinals[dialog['dialog_id']]
   for turn in (1,2,3):
    for method in method_order(ordinal):
     inv=assert_gpu_exclusive();physical=uuid.uuid4().hex;lid=logical_id(run,phase,method,dialog['dialog_id'],turn)
     source={'dialog_id':dialog['dialog_id'],'method_id':method,'physical_execution_id':physical,'logical_request_id':lid,'turn_id':turn}
     append(artifact/'attempts.jsonl',{**source,'status':'STARTED','at_unix':time.time(),'gpu_inventory':inv})
     capture=('kv_only' if method=='fullload' else 'with_score') if di==0 and turn==1 and method!='recompute' else False
     factory=lambda:history_factory(dialog,turn,method,generated[method])
     result,diag=execute_request(runner,image_path,image_sha,factory,method,turn,paths.get(method),capture=capture,return_logits=diagnostic is not None)
     append(artifact/'execution_events.jsonl',{**source,'status':'EXECUTED','prediction':result['prediction'],'generated_token_ids':result['generated_token_ids'],'ttft_ms':result['ttft_ms'],'request_e2e_ms':result['request_e2e_ms']})
     if capture:
      cap=result.pop('capture');layout='canonical' if method=='fullload' else 'repacked'
      t0=time.perf_counter();persist=runner.persist(cap,paths[method],layout=layout);wall=(time.perf_counter()-t0)*1000
      meta=persist['metadata'];metas[method]=meta;hashes[method]=file_hashes(paths[method])
      scores=safe(cap.scores) if cap.scores is not None else None
      selection={'N_content':cap.visual_count,'k':(cap.visual_count+3)//4,'scores_float32':scores,
       'stored_to_original':meta['stored_to_original'],'selected_original_ids':sorted(meta['stored_to_original'][:(cap.visual_count+3)//4]) if method==METHODS[2] else list(range(cap.visual_count)),
       'prefix_ids':cap.prefix_ids,'logical_position_ids':cap.logical_position_ids,'geometry':cap.geometry,'source':source}
      atomic_json(artifact/f'{method}_selection.json',selection)
      info[method]={'path':str(paths[method]),'meta_sha256':sha(paths[method]/'meta.json'),'source':source,'selection_artifact':str(artifact/f'{method}_selection.json'),'selection_sha256':sha(artifact/f'{method}_selection.json')}
      p={k:v for k,v in persist.items() if k!='metadata'}
      p.update(source=source,outer_writer_wall_ms=wall,post_generation_capture_tail_ms=result['post_generation_tail_ms'],
       charged_setup_before_activation_ms=wall+result['post_generation_tail_ms'],bytes_written=sum(x.stat().st_size for x in paths[method].rglob('*') if x.is_file()),
       scope='MEASURED entire post-generation return tail (score/clone and output postprocess) plus writer wall; included capture hooks remain in T1 request; component timers overlap and are not summed')
      atomic_json(artifact/f'{method}_persistence.json',p);atomic_json(artifact/f'{method}_meta.json',meta);atomic_json(artifact/f'{method}_hashes.json',hashes[method])
      if diagnostic is not None:captures[method]=cap
      del cap
     row=make_row(result,diag,method,dialog,turn,phase,run,attempt,physical,ordinal,config,image_sha,info.get(method),metas.get(method))
     assert row['N_content']==manifest['image_identity'][iid]['N_content']
     append(artifact/'raw.jsonl',row);rows.append(row);generated[method][turn]=row
     if result['conditioning'] and result['conditioning']['activation_ms'] and not (artifact/f'{method}_activation.json').exists():
      atomic_json(artifact/f'{method}_activation.json',runner.activation_records[str(paths[method].resolve())])
     if diagnostic is not None:diagnostic(runner,dialog,turn,method,row,result,captures,paths,metas,artifact)
     if method in paths and str(paths[method].resolve()) in runner.activation_records and not (artifact/f'{method}_activation.json').exists():
      atomic_json(artifact/f'{method}_activation.json',runner.activation_records[str(paths[method].resolve())])
     append(artifact/'attempts.jsonl',{**source,'status':'COMPLETED','at_unix':time.time()})
     del result,diag
  for m in METHODS[1:]:assert file_hashes(paths[m])==hashes[m]
  atomic_json(artifact/'store_integrity_after.json',{'status':'PASS','file_sha256':hashes})
  aud=helper('_qwen_commit_auditor','104_audit_qwen_mt_gqa_kv25_main.py')
  receipt=aud.audit_image_rows(read_rows(artifact/'raw.jsonl'),manifest,config,phase,iid,artifact)
  receipt['raw_sha256']=sha(artifact/'raw.jsonl');atomic_json(artifact/'independent_audit.json',receipt)
  if receipt['status']!='PASS':raise RuntimeError('INDEPENDENT_IMAGE_AUDIT_FAIL '+json.dumps(receipt))
  allowlist=file_hashes(scratch);atomic_json(artifact/'cleanup_allowlist.json',{'scratch':str(scratch),'file_sha256':allowlist,'ownership_run':run.name})
  evidence={p.name:sha(p) for p in artifact.iterdir() if p.is_file()}
  commit={'status':'COMMITTED','image_id':iid,'attempt_id':attempt,'requests':len(rows),'raw_sha256':receipt['raw_sha256'],
   'config_sha256':config['_file_sha256'],'manifest_sha256':config['_manifest_sha256'],'evidence_sha256':evidence,'first_successful_complete_image_adoption':True}
  atomic_json(parent/'COMMITTED.json',commit);success=True
 except BaseException as e:
  atomic_json(artifact/'failure.json',{'status':'PARTIAL','error':repr(e),'traceback':traceback.format_exc(),'completed_uncommitted_rows':len(rows),'adopted_from_attempt':0});raise
 finally:
  runner.close();captures.clear();gc.collect()
  if success:
   cleanup_committed(scratch,allowlist,receipt,commit,run)
   atomic_json(artifact/'cleanup_receipt.json',{'status':'PASS','removed_new_files':len(allowlist),'recipe':'rerun this complete image with exact frozen config/source/manifest in a fresh attempt; regenerate all method-local histories'})
 return len(rows)

def completed_images(run,phase,config,manifest):
 audit=helper('_qwen_resume_auditor','104_audit_qwen_mt_gqa_kv25_main.py');done={}
 for cp in sorted((run/phase/'images').glob('*/COMMITTED.json')):
  c=json.loads(cp.read_text());folder=cp.parent/c['attempt_id']
  if c['config_sha256']!=config['_file_sha256'] or c['manifest_sha256']!=config['_manifest_sha256']:raise RuntimeError('resume identity mismatch')
  for name,digest in c['evidence_sha256'].items():
   if sha(folder/name)!=digest:raise RuntimeError('committed evidence changed '+str(folder/name))
  receipt=audit.audit_image_rows(read_rows(folder/'raw.jsonl'),manifest,config,phase,cp.parent.name,folder)
  if receipt['status']!='PASS':raise RuntimeError('resume audit failed '+repr(receipt))
  if not (folder/'cleanup_receipt.json').exists():
   a=json.loads((folder/'cleanup_allowlist.json').read_text());scratch=Path(a['scratch'])
   if scratch.exists():cleanup_committed(scratch,a['file_sha256'],receipt,c,run)
   atomic_json(folder/'cleanup_receipt.json',{'status':'PASS','resume_cleanup':True,'removed_new_files':len(a['file_sha256'])})
  done[cp.parent.name]=c
 return done

def verify_freeze(run):
 from mmimpress.mt_gqa import workload_sha256
 config=json.loads((run/'config.json').read_text());manifest=json.loads((run/'manifest.json').read_text())
 assert list(METHODS)==config['methods'] and source_hashes()==config['source_sha256'],'FROZEN_SOURCE_CONFIG_MISMATCH'
 assert sha(CONTRACT)==config['contract_sha256'] and sha(run/'storage_plan.json')==config['storage_plan_sha256']
 assert sha(ROOT/'data/mt_gqa/dialogues.json')==INDEX_SHA
 ds=json.loads((ROOT/'data/mt_gqa/dialogues.json').read_text())['dialogues']
 assert ds==manifest['dialogues'] and workload_sha256(ds)==WORKLOAD_SHA
 for iid,x in manifest['image_identity'].items():assert sha(ROOT/x['image_path'])==x['sha256'],iid
 config.update(_file_sha256=sha(run/'config.json'),_manifest_sha256=sha(run/'manifest.json'));return config,manifest

def update_status(run,state,phase,count,images,error=None):
 doc={'execution_state':state,'phase':phase,'phase_completed_requests':count,'phase_completed_images':images,'expected_main_requests':36549,'at_unix':time.time(),'pid':os.getpid(),
  'main_final_requests':sum(json.loads(cp.read_text())['requests'] for cp in (run/'main/images').glob('*/COMMITTED.json')),
  'QWEN_3_ARM_MAIN':'PARTIAL' if phase=='main' or (run/'main').exists() else 'NOT RUN','resume_command':resume_command(run),'error':error}
 atomic_json(run/'CURRENT_STATUS.json',doc,replace=True)
 out=ROOT/'results'/run.name;out.mkdir(parents=True,exist_ok=True);atomic_json(out/'CURRENT_STATUS.json',doc,replace=True)
 (out/'PROGRESS.md').write_text(f"# Qwen MT-GQA 실행 상태\n\n상태: {state}; phase: {phase}; main 채택 완료 {doc['main_final_requests']:,}/36,549. 최종 독립 감사 전 VALID 아님.\n\n재개:\n```bash\n{doc['resume_command']}\n```\n"+(f'\n원인: {error}\n' if error else ''))
 print(json.dumps(doc),flush=True)

def resume_command(run):return f'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/103_eval_qwen_mt_gqa_kv25_main.py --run-dir {run} --phase all --resume'

def run_phase(runner,run,phase,config,manifest,diagnostic=None):
 done=completed_images(run,phase,config,manifest);ids=set(manifest[phase+'_dialogue_ids']) if phase in ('integration','smoke') else {d['dialog_id'] for d in manifest['dialogues']}
 groups=OrderedDict();ordinals={}
 for i,d in enumerate(manifest['dialogues']):
  ordinals[d['dialog_id']]=i
  if d['dialog_id'] in ids:groups.setdefault(d['image_id'],[]).append(d)
 count=sum(c['requests'] for c in done.values())
 for iid,ds in groups.items():
  if iid in done:continue
  if (run/'STOP_REQUESTED').exists():raise RuntimeError('STOP_REQUESTED')
  count+=run_image(runner,ds,ordinals,phase,run,manifest,config,diagnostic);done[iid]={'requests':len(ds)*9}
  update_status(run,'RUNNING',phase,count,len(done))
 assert count==len(ids)*9
 receipt={'status':'PASS','requests':count,'expected_requests':len(ids)*9,'images':len(groups),'config_sha256':config['_file_sha256'],'source_sha256':config['source_sha256']}
 p=run/phase/'validation.json'
 if p.exists():assert json.loads(p.read_text())==receipt
 else:atomic_json(p,receipt)
 return receipt

def load_runner():
 assert_gpu_exclusive();runner=Qwen25Runner(attn='sdpa').load()
 f=runner.runtime_fingerprint();assert (f['decoder_layers'],f['kv_heads'],f['head_dim'])==(28,4,128)
 assert f['text_attention_backend']==f['vision_attention_backend']=='sdpa' and not f['weight_offload']
 runner.run_pixels(Image.new('RGB',(448,448),(127,127,127)),'Describe the image briefly.',capture=False)
 return runner

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--phase',choices=('smoke','main','all'),default='all');p.add_argument('--resume',action='store_true');a=p.parse_args();run=a.run_dir.resolve()
 config,manifest=verify_freeze(run);gate=json.loads((run/'integration_validation.json').read_text())
 assert gate['status']=='PASS' and gate['source_sha256']==config['source_sha256'],'INTEGRATION_GATE_NOT_PASS'
 phases=['smoke','main'] if a.phase=='all' else [a.phase];runner=None
 try:
  capacity_guard(run);runner=load_runner();fingerprint=runner.runtime_fingerprint();stamp=time.strftime('%Y%m%dT%H%M%S')
  atomic_json(run/f'runtime_{stamp}.json',fingerprint)
  assert fingerprint==json.loads((run/'integration_runtime.json').read_text()),'runtime changed'
  for phase in phases:
   if phase=='main':
    smoke=json.loads((run/'smoke/validation.json').read_text());assert smoke['status']=='PASS' and smoke['requests']==36
   if (run/phase).exists() and not a.resume:raise FileExistsError('use --resume; preserve partial')
   run_phase(runner,run,phase,config,manifest)
  runner.close();runner=None
  update_status(run,'AUDITING','main',36549,398)
  subprocess.run([sys.executable,str(ROOT/'scripts/104_audit_qwen_mt_gqa_kv25_main.py'),'--run-dir',str(run),'--output-dir',str(ROOT/'results'/run.name/'final')],check=True)
 except BaseException as e:
  update_status(run,'INTERRUPTED',locals().get('phase','startup'),0,0,repr(e))
  atomic_json(run/('interruption_'+time.strftime('%Y%m%dT%H%M%S')+'.json'),{'status':'PARTIAL','error':repr(e),'traceback':traceback.format_exc(),'resume_command':resume_command(run)})
  raise
 finally:
  if runner:runner.close()
if __name__=='__main__':main()

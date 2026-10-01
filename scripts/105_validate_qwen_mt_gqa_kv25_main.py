#!/usr/bin/env python3
"""Fresh generated-history integration of the unchanged v2/KV25 production path."""
from __future__ import annotations
import argparse,copy,gc,importlib.util,json,sys,time,traceback
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def load(name,file):
 s=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
M=load('_qwen_mt_main_validation','103_eval_qwen_mt_gqa_kv25_main.py')
A=load('_qwen_mt_independent_validation','104_audit_qwen_mt_gqa_kv25_main.py')
V=load('_qwen_mt_v2_reference','86_validate_qwen25_v2_rerun.py')
K=load('_qwen_mt_KV25_reference','91_validate_qwen25_kv25.py');K.v2=V

def verify_layers(a,b):
 return V.v1._bitwise_layers(a,b)

def diagnostic(runner,dialog,turn,method,row,result,captures,paths,metas,artifact):
 if method=='recompute':return
 cap=captures[method];store=runner._activate(paths[method],row['image_sha256']);meta=metas[method]
 if turn==1:
  loaded=store.load_prefix(1.,budget_unit='chunk');bits=verify_layers(cap.layers,loaded.layers)
  assert bits['equal'],'full native BF16 inverse roundtrip'
  evidence={'status':'PASS','scope':'own actual source T1 capture versus own 100% store inverse','all_layer_native_BF16_bits':bits,'source_physical_execution_id':row['physical_execution_id']}
 else:
  n=cap.visual_count;num=(n+3)//4 if method==M.METHODS[2] else n
  order=sorted(range(n),key=lambda i:(-float(cap.scores[i]),i)) if method==M.METHODS[2] else list(range(n))
  structural=[i for i in range(len(cap.prefix_ids)) if not cap.visual_start<=i<cap.visual_start+n]
  logical=sorted(structural+[cap.visual_start+i for i in order[:num]])
  index=torch.tensor(logical,device=cap.layers[0][0].device)
  reference=[(k.index_select(2,index).clone(),v.index_select(2,index).clone()) for k,v in cap.layers]
  unit='visual_kv' if method==M.METHODS[2] else 'chunk';ratio=.25 if method==M.METHODS[2] else 1.
  loaded=store.load_prefix(ratio,budget_unit=unit);bits=verify_layers(reference,loaded.layers);assert bits['equal']
  full=torch.tensor(row['input_ids']);prefix=len(cap.prefix_ids);suffix=full[:,prefix:]
  pos,delta=V._stock_positions(runner,full,torch.tensor(cap.image_grid_thw))
  assert pos.tolist()==row['request_trace']['position_ids'] and delta.tolist()==row['request_trace']['rope_deltas']
  memory=V.trace_suffix(runner,reference,suffix,pos,prefix);ssd=V.trace_suffix(runner,loaded.layers,suffix,pos,prefix)
  repeat=V.trace_suffix(runner,reference,suffix,pos,prefix)
  comparisons={'memory_vs_SSD':V.path_comparison(memory,ssd),'reference_repeat':V.path_comparison(memory,repeat),'production_wrapper_vs_memory':V.path_comparison(memory,V._result_path(result))}
  assert all(x['exact_target'] for x in comparisons.values()),comparisons
  expected=V._analytic_mask([True]*len(logical),suffix.shape[1]);mask=V._actual_mask_checks(memory,expected);assert mask['all_layers_exact']
  assert V._actual_mask_checks(ssd,expected)['all_layers_exact']
  evidence={'status':'PASS','same_capture_BF16_bits':bits,'independent_selected_original':sorted(order[:num]),'comparisons':comparisons,'causal_mask':mask,'stock_MRoPE_exact':True,'logical_positions':logical,'source_capture':row['source_store']['source'],'production_physical_execution_id':row['physical_execution_id'],'history_entries':row['history_entries']}
  if method==M.METHODS[2] and turn==2:
   with K.finite_extra_sentinel(num,min(n,((num+63)//64)*64),meta['row_bytes']) as changed:
    poisoned=store.load_prefix(.25,budget_unit='visual_kv')
    poisoned_result=runner.run_cache(paths[method],row['question'],history=tuple(map(tuple,row['history'])),budget_ratio=.25,budget_unit='visual_kv',image_sha256=row['image_sha256'],return_logits=True)
   poisonbits=verify_layers(reference,poisoned.layers);poisoncmp=V.path_comparison(V._result_path(result),V._result_path(poisoned_result))
   assert poisonbits['equal'] and poisoncmp['exact_target'] and changed[0]==112
   dense=[];densemask=torch.zeros((1,prefix),dtype=torch.long);densemask[0,logical]=1
   for k,v in reference:
    dk=torch.zeros((1,4,prefix,128),dtype=k.dtype,device=k.device);dv=torch.zeros_like(dk);dk.index_copy_(2,index,k);dv.index_copy_(2,index,v);dense.append((dk,dv))
   densepath=V.trace_suffix(runner,dense,suffix,pos,prefix,densemask)
   ca=V._fp32_oracle(memory,reference,[True]*len(logical),28,4,128)
   de=V._fp32_oracle(densepath,dense,densemask[0].bool().tolist(),28,4,128)
   oracle=bool(torch.allclose(ca,de,atol=1e-5,rtol=1e-5));assert oracle,'independent FP32 oracle'
   evidence.update(extra_sentinel={'modified_visual_spans':changed[0],'bits':poisonbits,'output':poisoncmp},fp32_dense_compact={'atol':1e-5,'rtol':1e-5,'PASS':oracle,'max_abs':float((ca-de).abs().max())})
   del dense,densepath,ca,de,poisoned,poisoned_result
  del memory,ssd,repeat,reference,loaded
 M.atomic_json(artifact/f'gpu_{method}_T{turn}.json',evidence)
 # On the final hit, test actual model and callback exceptions, then repeat the same hit.
 if turn==3 and method==M.METHODS[2]:
  from mmimpress.qwen25.vision import VisionScoreCapture
  baseline_hooks=[len(module._forward_hooks)+len(module._forward_pre_hooks) for module in runner.model.modules()]
  factory=lambda:(row['question'],tuple(map(tuple,row['history'])),row['history_entries'])
  exceptions=[]
  for fault in ('model_forward','decode_callback','score_exit'):
   if fault=='model_forward':owner=runner.model;attr='forward'
   elif fault=='decode_callback':owner=runner;attr='_decode'
   else:owner=VisionScoreCapture;attr='_compute_scores'
   original=getattr(owner,attr)
   def broken(*a,**kw):raise RuntimeError('INJECTED_'+fault)
   setattr(owner,attr,broken);caught=False
   try:
    if fault=='score_exit':M.execute_request(runner,ROOT/dialog['image_path'],row['image_sha256'],lambda:(dialog['turns'][0]['question'],(),[]),method,1,capture='with_score')
    else:M.execute_request(runner,ROOT/dialog['image_path'],row['image_sha256'],factory,method,3,paths[method])
   except RuntimeError as e:caught='INJECTED_' in str(e)
   finally:setattr(owner,attr,original)
   after=[len(module._forward_hooks)+len(module._forward_pre_hooks) for module in runner.model.modules()]
   assert caught and after==baseline_hooks and runner.model.model.rope_deltas is None
   check,_=M.execute_request(runner,ROOT/dialog['image_path'],row['image_sha256'],factory,method,3,paths[method],return_logits=True)
   comp=V.path_comparison(V._result_path(result),V._result_path(check));assert comp['exact_target']
   exceptions.append({'fault':fault,'injected_exception_caught':caught,'hooks_state_restored':True,'repeat':comp})
  M.atomic_json(artifact/'exception_recovery.json',{'status':'PASS','controls':exceptions})
 gc.collect()

def negative_controls(run,config,manifest):
 cp=next((run/'integration/images').glob('*/COMMITTED.json'));com=json.loads(cp.read_text());folder=cp.parent/com['attempt_id'];rows=M.read_rows(folder/'raw.jsonl')
 base=A.audit_image_rows(rows,manifest,config,'integration',cp.parent.name,folder);assert base['status']=='PASS'
 target=next(i for i,r in enumerate(rows) if r['method_id']==M.METHODS[2] and r['turn_id']==2)
 changes={
 'foreign_history':lambda r:r['history_entries'][0].update(source_method_id='fullload'),
 'gold_history':lambda r:r['history'][0].__setitem__(1,'FORGED_GOLD_SENTINEL'),
 'wrong_budget':lambda r:r.__setitem__('k',r['k']+1),
 'wrong_actual_budget':lambda r:r['result'].__setitem__('budget_unit','chunk'),
 'forged_read':lambda r:r['result']['os_pread_trace'][0].__setitem__('returned',0),
 'extra_attention_row':lambda r:r['request_trace']['layer_prefix_shapes'][0][0].__setitem__(2,r['k']+r['S_structural']+1),
 'hit_vision':lambda r:r['result'].__setitem__('actual_vision_calls',1),
 'wrong_source':lambda r:r['source_store']['source'].__setitem__('physical_execution_id','foreign'),
 'shifted_MRoPE':lambda r:r['request_trace']['actual_prefill_position_ids'][0][0].__setitem__(0,0),
 'wrong_ttft':lambda r:r.__setitem__('ttft_ms',r['ttft_ms']+1),
 'future_prompt':lambda r:r.__setitem__('prompt',r['prompt']+'future'),
 'wrong_prediction':lambda r:r.__setitem__('prediction','forged'),
 }
 evidence=[]
 for name,mutate in changes.items():
  bad=copy.deepcopy(rows);mutate(bad[target]);out=A.audit_image_rows(bad,manifest,config,'integration',cp.parent.name,folder);assert out['status']=='FAIL',name;evidence.append({'fixture':name,'rejected':True,'failure_counts':out['failure_counts']})
 bad=copy.deepcopy(rows);bad.append(bad[-1]);assert A.audit_image_rows(bad,manifest,config,'integration',cp.parent.name,folder)['status']=='FAIL'
 bad=copy.deepcopy(rows[:-1]);assert A.audit_image_rows(bad,manifest,config,'integration',cp.parent.name,folder)['status']=='FAIL'
 # Resume independently re-audits committed evidence and does not adopt partial directories.
 adopted=M.completed_images(run,'integration',config,manifest);assert len(adopted)==4 and sum(c['requests'] for c in adopted.values())==36
 return {'status':'PASS','mutation_fixtures':evidence,'duplicate_missing_rejected':True,'resume_committed_images':4}

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();run=a.run_dir.resolve();config,manifest=M.verify_freeze(run)
 core=json.loads((run/'gpu_core_validation/validation.json').read_text());assert core['status']=='PASS' and len(core['samples'])==10
 assert M.sha(run/'gpu_core_validation/validation.json')==config['core_validation_sha256']
 report={'status':'NOT RUN','source_sha256':config['source_sha256'],'gates':{f'G{i}':{'status':'NOT RUN'} for i in range(1,13)},'core_gpu_validation':str(run/'gpu_core_validation/validation.json'),'core_gpu_validation_sha256':config['core_validation_sha256'],'llava_gpu':'NOT RUN'}
 runner=None
 try:
  runner=M.load_runner();M.atomic_json(run/'integration_runtime.json',runner.runtime_fingerprint())
  phase=M.run_phase(runner,run,'integration',config,manifest,diagnostic=diagnostic)
  runner.close();runner=None
  negatives=negative_controls(run,config,manifest);M.atomic_json(run/'integration_negative_controls.json',negatives)
  scope={1:'full dataset and explicit 3-arm configuration, unchanged Qwen runtime',2:'10 fixed pairs plus 4 actual source captures BF16 roundtrip/inverse',3:'all FullLoad generated-history hits versus same-capture direct memory, exact logits/IDs',4:'all Ours generated-history hits versus independent stable-rank/gather canonical capture, exact logits/IDs',5:'all native layer/head rows, structural preserved, finite unused-row sentinel unchanged',6:'stock full logical MRoPE, compact cache slots, interleaved state and decode',7:'10 fixed-pair and four generated-history FP32 dense/compact oracle 1e-5/1e-5, negative fixtures',8:'observed source T1 vision=1, hit vision/scoring=0, same-image selection invariant',9:'independent causal prompt/tokenization and actual method/model/dialogue lineage, forged histories rejected',10:'actual syscall ranges/bytes and minimal whole chunks independently audited',11:'427 legacy CPU tests plus unchanged core, existing artifact protection; LLaVA GPU NOT RUN',12:'absolute outer timers, model/decode/score exceptions, audited resume and tested scratch/no-clobber safety'}
  for i in range(1,13):report['gates'][f'G{i}']={'status':'PASS','scope':scope[i]}
  cpu=json.loads((run/'integration_cpu_validation.json').read_text());assert cpu['status']=='PASS' and M.sha(Path(cpu['log']))==cpu['log_sha256']
  report.update(status='PASS',integration_requests=phase['requests'],integration_images=phase['images'],negative_controls=negatives,CPU=cpu)
 except BaseException as e:
  report.update(status='UNRESOLVED',error=repr(e),traceback=traceback.format_exc());raise
 finally:
  if runner:runner.close()
  M.atomic_json(run/'integration_validation.json',report)
 print(json.dumps({'status':report['status'],'gates':{k:v['status'] for k,v in report['gates'].items()},'requests':report.get('integration_requests')}),flush=True)
if __name__=='__main__':main()

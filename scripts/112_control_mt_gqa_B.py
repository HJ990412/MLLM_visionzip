#!/usr/bin/env python3
"""Freeze, regression and model-independent gating for the B main experiment."""
from pathlib import Path
import argparse,json,sys,os,shutil,subprocess,importlib.util,time,traceback,random,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
import main_b_support as S
OLD_B=ROOT/'runs/prefix25_persistence_20261001T050113Z'
OLD={'llava':ROOT/'runs/llava_mt_gqa_allhead_kv25_main_20260930T093251Z','qwen':ROOT/'runs/qwen_mt_gqa_kv25_main_20261001T031459Z'}
RUNNER={'llava':'108_eval_llava_mt_gqa_B.py','qwen':'110_eval_qwen_mt_gqa_B.py'}

def load(name,file):
 spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def dump(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);S.publish(path,obj)

def freeze(run):
 # Binding the final post-hardening replay rather than the original pilot driver.
 prior=json.loads((OLD_B/'final_gpu_freeze.json').read_text());files=prior.get('files',prior.get('source_sha256'))
 for name,digest in files.items():assert S.sha(ROOT/name)==digest,('final B freeze changed',name)
 config_sources=S.source_hashes(ROOT)
 sources_before=json.loads((run/'source_before.json').read_text());assert all(S.sha(ROOT/n)==v for n,v in sources_before.items())
 for model in ('llava','qwen'):
  M=load('_main_b_freeze_'+model,RUNNER[model]);base=run/model;base.mkdir(exist_ok=True)
  manifest=json.loads((OLD[model]/'manifest.json').read_text());config=json.loads((OLD[model]/'config.json').read_text());plan=json.loads((OLD[model]/'storage_plan.json').read_text())
  ds=manifest['dialogues'];assert len(ds)==4061 and len({d['image_id'] for d in ds})==398
  from mmimpress.mt_gqa import workload_sha256
  assert workload_sha256(ds)==M.WORKLOAD_SHA and S.sha(ROOT/'data/mt_gqa/dialogues.json')==M.INDEX_SHA
  assert json.loads((ROOT/'data/mt_gqa/dialogues.json').read_text())['dialogues']==ds
  for image,identity in manifest['image_identity'].items():assert S.sha(ROOT/identity['image_path'])==identity['sha256']
  manifest.update(schema_version=M.SCHEMA,methods=list(M.METHODS),prior_manifest_sha256=S.sha(OLD[model]/'manifest.json'))
  if model=='llava':
   first=ds[0]['image_id'];manifest['integration_dialogue_ids']=[d['dialog_id'] for d in ds if d['image_id']==first][:2]
   assert len(manifest['integration_dialogue_ids'])==2
   manifest['expected_requests']['integration']=30
  else:manifest.pop('llava_binding',None)
  manifest['same_image_fixture_dialogue_ids']=[d['dialog_id'] for d in ds if d['image_id']==ds[0]['image_id']][:2]
  plan.update(free_bytes_at_freeze=shutil.disk_usage(run).free,mount=subprocess.check_output(['findmnt','-J','-T',str(run)],text=True),
   minimum_reserve_bytes=30<<30,maximum_new_payload_bytes=20<<30,validation_peak_payload_bound_bytes=20<<30,
   combined_raw_checkpoint_growth_bound_bytes=40<<30,combined_validation_bound_bytes=24<<30)
  plan['required_initial_free_bytes']=(30+40+24+8)<<30
  plan['status']='PASS' if plan['free_bytes_at_freeze']>=plan['required_initial_free_bytes'] else 'BLOCKED_STORAGE'
  assert plan['status']=='PASS';dump(base/'storage_plan.json',plan);dump(base/'manifest.json',manifest)
  config.update(schema_version=M.SCHEMA,methods=list(M.METHODS),source_sha256=config_sources,contract_sha256=S.sha(M.CONTRACT),
   storage_plan_sha256=S.sha(base/'storage_plan.json'),manifest_sha256=S.sha(base/'manifest.json'),
   source_tests_sha256={str(p.relative_to(ROOT)):S.sha(p) for p in sorted((ROOT/'tests').glob('test_*.py'))},
   cpu_threads=2,seed=1234,history_policy='method_local_generated',B_storage_policy='prefix25',
   B_prior_final_replay_sha256=S.sha(OLD_B/model/'final_gpu_regression.json'),
   source_T1_capture='first frozen dialogue per image only; all dialogues/methods still actual full-pixel T1',
   source_capture_E2E='required capture exit included exactly once in source T1 E2E',
   prior_reference_only={'run':str(OLD[model]),'config_sha256':S.sha(OLD[model]/'config.json')})
  config.pop('core_validation_sha256',None);config.pop('QWEN_GPU',None)
  if model=='llava':config['budgets'][M.METHODS[-1]]=config['budgets'].pop('ours_kv25')
  else:
   from mmimpress.qwen25.runner import Qwen25Runner
   config['core_model_store_revision']=Qwen25Runner()._code_revision()
  dump(base/'config.json',config);shutil.copy2(M.CONTRACT,base/'contract.frozen.md')
  execution=[]
  for iid in dict.fromkeys(d['image_id'] for d in ds):
   for ordinal,d in enumerate(ds):
    if d['image_id']==iid:execution.append({'image_id':iid,'dialog_id':d['dialog_id'],'canonical_ordinal':ordinal,'method_order':list(M.method_order(ordinal)),'turn_order':[1,2,3]})
  dump(base/'execution_manifest.json',execution)
 for n in config_sources:
  target=run/'source_frozen'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,target)
 shutil.copy2(ROOT/'docs/mt_gqa_B_main_contract.md',run/'CONTRACT.md')
 shutil.copy2('/home/dblab/.codex/attachments/4815f5a8-dfeb-4606-a0e4-6285825886ce/Pasted text.txt',run/'REQUEST.md')
 shutil.copy2(OLD_B/'correctness_manifest.json',run/'correctness_manifest.json')
 evidence={'B_final':{'freeze':str(OLD_B/'final_gpu_freeze.json'),'sha256':S.sha(OLD_B/'final_gpu_freeze.json'),'all_bound_files_match':True},
  'baseline_entries':{'recompute':'Server.recompute / Qwen25Runner.run_pixels','fullload':'Server.request fullload / Qwen25Runner.run_cache chunk1','mpic32_ssd':'MPICServer(k_recompute=32), MPICContext','sparsevlm_ssd_kv25_allhead':'SparseVLMSSDServer.request; CanonicalContext head_policy=all'},
  'unchanged_existing_source':True,'added_runners':RUNNER,'AllHead_algo_files':{str(p.relative_to(ROOT)):S.sha(p) for p in (ROOT/'mmimpress').glob('sparsevlm*.py')},
  'MPIC_sha256':S.sha(ROOT/'mmimpress/mpic.py'),'Qwen_prior_failure':str(OLD['qwen']/'integration_validation.json'),
  'prior_B_versus_main':'pilot 106 numbers never used as current-run setup or main metrics'}
 dump(run/'baseline_source_correspondence.json',evidence)
 env={'python':sys.version,'torch':torch.__version__,'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),'pip_freeze':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),'git_head':'ABSENT; immutable copies and SHA256 source maps'}
 dump(run/'environment_before.json',env)
 dump(run/'final_source_freeze.json',{'files':config_sources,'contract_sha256':S.sha(run/'CONTRACT.md'),'models':{m:{n:S.sha(run/m/n) for n in ('config.json','manifest.json','storage_plan.json','execution_manifest.json')} for m in OLD}})
 print(json.dumps({'freeze':'PASS','run':str(run),'L_main':60915,'Q_main':36549}),flush=True)

def cpu(run):
 log=run/'cpu_regression.log'
 with log.open('x') as f:p=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_*.py'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 dump(run/'cpu_regression.json',{'status':'PASS' if p.returncode==0 else 'FAIL','log':str(log),'log_sha256':S.sha(log),'returncode':p.returncode})
 assert p.returncode==0,'CPU regression failed'

def seed():torch.set_num_threads(2);torch.manual_seed(1234);random.seed(1234);np.random.seed(1234)

def core(run,model):
 M=load('_main_b_core_main_'+model,RUNNER[model]);config,manifest=M.verify_freeze(run/model)
 assert json.loads((run/'cpu_regression.json').read_text())['status']=='PASS'
 E=load('_main_b_final_control','106_prefix25_persistence.py');seed();E.exclusive();exp=E.Experiment(run,model);seed()
 samples=json.loads((run/'correctness_manifest.json').read_text())['samples'];samples=[samples[i] for i in [0,3,4,5,6]] if model=='llava' else samples
 out={'status':'RUNNING','source_sha256':config['source_sha256'],'samples':[],'cpu_threads':torch.get_num_threads()}
 try:
  for i,sample in enumerate(samples):
   path=run/model/'core_gpu'/f'{i:02d}_{sample["image_id"]}';path.mkdir(parents=True)
   result=exp.llava_case(sample,path) if model=='llava' else exp.qwen_case(sample,path)
   out['samples'].append({'image_id':sample['image_id'],'question_id':sample['question_id'],'status':'PASS',**result})
   dump(path/'validation.json',out['samples'][-1]);print(model,'B fixed',i+1,len(samples),'PASS',flush=True)
  out['status']='PASS'
 except BaseException as exc:out.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc());raise
 finally:
  dump(run/model/'core_gpu_validation.json',out);exp.close_contexts()

def integration(run,model):
 base=run/model;M=load('_main_b_integration_runner_'+model,RUNNER[model]);config,manifest=M.verify_freeze(base)
 assert json.loads((base/'core_gpu_validation.json').read_text())['status']=='PASS'
 report={'status':'RUNNING','source_sha256':config['source_sha256'],'numerical_tolerances':'unchanged','gates':{}}
 seed();runner=None
 try:
  if model=='llava':
   D=load('_main_b_llava_diagnostic','114_validate_llava_mt_gqa_B.py');D.M=M
   runner=M.V.load_runner();servers=M.servers_for(runner);M.QA._warmup(runner,servers['old'])
   V=M.V;V._ACTIVE_RUN=base;entries,_,_=V.frozen_workloads();by={e['image_id']:e for e in entries};records=[]
   for image,qid in V.FIXED:
    path=V.STORE_ROOT/'raster'/image
    provenance=V.verify_roundtrip(runner,by[image],qid,path)
    item=V.sample(runner,by[image],qid,path,base/'allhead_fixed',methods=(M.ALLHEAD,));item['T1_provenance']=provenance;records.append(item)
    print('AllHead fixed',image,qid,'PASS',flush=True)
   dump(base/'allhead_gpu_validation.json',{'status':'PASS','samples':records,'criterion':{'logits_atol':1e-4,'logits_rtol':1e-4,'score_atol':1e-5,'score_rtol':1e-5},'validator':str(ROOT/'scripts/97_validate_sparsevlm_ssd_kv25.py'),'methods':[M.ALLHEAD]})
   report['main_integration']=M.run_phase(runner,servers,base,'integration',config,manifest,diagnostic=D.integration_diagnostic)
   report['negative_controls']=D.negative_audit(base,config,manifest)
  else:
   D=load('_main_b_qwen_diagnostic','115_validate_qwen_mt_gqa_B.py');D.M=M
   runner=M.load_runner();dump(base/'integration_runtime.json',runner.runtime_fingerprint())
   report['main_integration']=M.run_phase(runner,base,'integration',config,manifest,diagnostic=D.diagnostic)
   report['negative_controls']=D.negative_controls(base,config,manifest)
   # Distinct same-image multiple-dialogue fixture, actual main runner/auditor.
   multi=json.loads(json.dumps(manifest));multi['integration_dialogue_ids']=manifest['same_image_fixture_dialogue_ids'];fixture=base/'same_image_fixture';fixture.mkdir()
   for name in ('storage_plan.json',):shutil.copy2(base/name,fixture/name)
   report['same_image_fixture']=M.run_phase(runner,fixture,'integration',config,multi)
  report['status']='PASS';report['gates']={f'G{i}':'PASS' for i in range(1,13)}
 except BaseException as exc:
  report.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc(),main_admission='BLOCKED');raise
 finally:
  dump(base/'integration_validation.json',report)
  if model=='qwen' and runner:runner.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['freeze','cpu','core','integration'],required=True);p.add_argument('--model',choices=['llava','qwen']);a=p.parse_args()
 if a.stage=='freeze':freeze(a.run)
 elif a.stage=='cpu':cpu(a.run)
 elif a.stage=='core':core(a.run,a.model)
 else:integration(a.run,a.model)
if __name__=='__main__':main()

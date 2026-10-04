from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path('/home/dblab/hj/mllm_v2');RUN=Path(__file__).resolve().parent
PY='/home/dblab/anaconda3/envs/mllm_ft/bin/python'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='.:scripts:tests',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
assert read(RUN/'repeat_validation.json')['status']=='PASS'
subprocess.run([PY,'scripts/117_run_qwen_prefixstore3.py','--run',str(RUN),'--stage','review'],cwd=ROOT,env=env,check=True)
bindings={str(p.relative_to(RUN)):sha(p) for p in [RUN/'CONTRACT.md',RUN/'G7_REVIEW.md',RUN/'G7_stage_attribution.json',RUN/'G7_review_verdict.json',RUN/'environment_before.json',RUN/'final_source_freeze.json',RUN/'additional_validation_manifest.json',RUN/'repeat_validation.py',RUN/'repeat_validation_freeze.json',RUN/'repeat_validation.json',RUN/'cpu_regression.json',RUN/'core_gpu_validation.json',RUN/'integration_validation.json',Path(__file__)]}
dump(RUN/'pre_main_admission_freeze.json',dict(status='PASS',bindings=bindings,legacy_G7='FAIL',revised_semantic_implementation='PASS',main_command='117 --stage main; actual smoke then full 36549',at_unix=time.time()))
start=time.time()
with (RUN/'main_execution.log').open('x') as log:
 p=subprocess.run([PY,'-u','scripts/117_run_qwen_prefixstore3.py','--run',str(RUN),'--stage','main'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
dump(RUN/'main_process_receipt.json',dict(returncode=p.returncode,start_unix=start,end_unix=time.time(),launcher_wall_s=time.time()-start,scope='model load/warmup/smoke/main plus audit/conditioning/logging/cleanup; not request E2E'))
assert p.returncode==0,'main interrupted; preserve all attempts and resume frozen 117'
with (RUN/'final_analysis.log').open('x') as log:
 p=subprocess.run([PY,'-u','scripts/118_report_qwen_prefixstore3.py','--run',str(RUN),'--output',str(ROOT/'results'/RUN.name/'final')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
dump(RUN/'analysis_process_receipt.json',dict(returncode=p.returncode,ended_at_unix=time.time()))
assert p.returncode==0,'independent final analysis failed'
print('main and independent analysis complete',flush=True)

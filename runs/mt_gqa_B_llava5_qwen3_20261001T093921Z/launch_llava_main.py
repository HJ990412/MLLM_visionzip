"""Actual frozen 108 main lifecycle; final reporting uses 113, not legacy renderer."""
from pathlib import Path
import sys,importlib.util,json,traceback,random,time
import numpy as np
import torch
root=Path.cwd();run=Path(__file__).resolve().parent;sys.path.insert(0,str(root));sys.path.insert(0,str(root/'scripts'))
s=importlib.util.spec_from_file_location('B_actual_main',root/'scripts/108_eval_llava_mt_gqa_B.py');M=importlib.util.module_from_spec(s);sys.modules[s.name]=M;s.loader.exec_module(M)
freeze=json.loads((run/'main_launch_freeze.json').read_text());assert M.sha(__file__)==freeze['launch_sha256']
base=run/'llava';config,manifest=M.verify_freeze(base)
gate=json.loads((base/'integration_validation.json').read_text());assert gate['status']=='PASS' and gate['source_sha256']==config['source_sha256']
assert json.loads((base/'allhead_gpu_validation.json').read_text())['status']=='PASS'
assert json.loads((run/'cpu_regression.json').read_text())['status']=='PASS'
torch.set_num_threads(2);torch.manual_seed(1234);random.seed(1234);np.random.seed(1234)
M.assert_gpu_exclusive();M.capacity_guard(base);runner=M.V.load_runner();servers=M.servers_for(runner);M.QA._warmup(runner,servers['old'])
started=time.time();M.atomic_json(base/('main_process_'+time.strftime('%Y%m%dT%H%M%S')+'.json'),{'started_unix':started,'source_sha256':config['source_sha256'],'threads':torch.get_num_threads(),'launcher_sha256':M.sha(__file__),'warmup_excluded':True})
try:
 for phase in ('smoke','main'):
  if phase=='main':
   smoke=json.loads((base/'smoke/validation.json').read_text());assert smoke['status']=='PASS' and smoke['requests']==60 and smoke['source_sha256']==config['source_sha256']
  M.run_phase(runner,servers,base,phase,config,manifest)
 M.atomic_json(base/'main_execution_complete.json',{'status':'PASS','main_requests':60915,'images':398,'wall_seconds_including_conditioning_logging_audit_cleanup':time.time()-started,'final_independent_analysis':'PENDING'})
except BaseException as exc:
 M.atomic_json(base/('main_interruption_'+time.strftime('%Y%m%dT%H%M%S')+'.json'),{'status':'PARTIAL','error':repr(exc),'traceback':traceback.format_exc(),'complete_images':len(list((base/'main/images').glob('*/COMMITTED.json'))),'resume':'rerun this exact launcher; committed images independently re-audited, partial images wholly retried'})
 raise

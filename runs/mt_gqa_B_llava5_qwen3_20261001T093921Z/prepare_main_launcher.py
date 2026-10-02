from pathlib import Path
import hashlib,json,shutil
root=Path.cwd();run=Path(Path('/tmp/main_b_run').read_text());code='''"""Actual frozen 108 main lifecycle; final reporting uses 113, not legacy renderer."""
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
'''
p=run/'launch_llava_main.py';p.write_text(code)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
(run/'main_launch_freeze.json').write_text(json.dumps({'launch_sha256':sha(p),'config_sha256':sha(run/'llava/config.json'),'source_freeze_sha256':sha(run/'final_source_freeze.json'),'integration_sha256':sha(run/'llava/integration_validation.json'),'purpose':'entry-point only, invokes actual frozen 108 functions and skips legacy 109 final renderer; separate 113 analysis source freeze precedes analysis'},indent=2))
repro=f'''# Reproduce / resume B full main

Run `{run}`. GPU models sequential, source/config/contract frozen. Qwen integration G7 FAIL blocks smoke/main; never bypass its gate.

```bash
cd {root}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.:scripts:tests PYTHONDONTWRITEBYTECODE=1
/home/dblab/anaconda3/envs/mllm_ft/bin/python {run}/launch_llava_main.py
# After main complete, use a new output directory for every independent report:
/home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/113_report_mt_gqa_B.py --run {run} --output {root}/results/{run.name}/reaudit_01
```

The actual main runner is scripts/108_eval_llava_mt_gqa_B.py. The launcher verifies source/config/integration, runs real smoke then main, and lets the new independent 113 reporter produce all requested tables. It does not use the old 109 renderer's full-store wording. Resume accepts only evidence-bound image commits, fresh full image attempt otherwise, no choice by latency/accuracy. STOP_REQUESTED in llava/ halts at an image boundary. Old attempts/payloads remain until independently authorized cleanup; no old run deletion.

For a complete new reproduction create a NEW root through the archived bootstrap snapshot policy, then 112 --stage freeze, --stage cpu, each model --stage core, --stage integration, and only PASS models' smoke/main. Copying PASS receipts is insufficient. Exact launch script is bound in main_launch_freeze.json. Previous final B source and its model-specific replay remain referenced by baseline_source_correspondence.json. The original Qwen main FP32 failure and this new B failure are immutable.

Full main committed payloads are intentionally removed after full checksums, independent audit, close, atomic commit and exact-owned allowlist cleanup. Metadata, hash inventory, requests, physical events and mappings remain. Reproduce payloads by rerunning each image's full source T1 with the exact pinned model/processor/source/config, then all its original dialogues in original ordinal rotation, generating new own-method histories. Validation controls retain their own A/B stores separately. Prior pilot setup numbers never substitute current measured setup.

Native request/component intervals can overlap; only request E2E and whole physical persistence/activation event wall times determine critical totals. AllHead independent deployment costs are derived from the measured shared canonical bundle. Non-source capture increment remains unmeasured. Historical large-file protection uses bounded fingerprints, not full SHA; all new raw/payload records have full SHA.
'''
(run/'REPRODUCE.md').write_text(repro);out=root/'results'/run.name;shutil.copy2(run/'REPRODUCE.md',out/'REPRODUCE.md')
print(str(p))

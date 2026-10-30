#!/usr/bin/env python3
"""Managed image-atomic execution with prospective admission and explicit states."""
import argparse,copy,gc,gzip,hashlib,json,os,shutil,sys,time,traceback,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from mmimpress.mtvqa25k.common import read,dump,sha,digest
from mmimpress.mtvqa25k.adapter import bind,load,enrich,read_rows,archive_before_commit,remove_archived_duplicates

def status(run,model,phase,state,**extra):
    command=(f'{sys.executable} scripts/139_validate_mtvqa25k_gpu.py --run {run} --model {model}' if phase=='gate' else
             f'{sys.executable} scripts/138_run_mtvqa25k.py --run {run} --model {model} --phase {phase} --resume')
    dump(run/f'{model}_STATUS.json',{'model':model,'phase':phase,'status':state,'pid':os.getpid(),
        'heartbeat_unix':time.time(),'resume_command':command,**extra},replace=True)

def verify_freeze(run):
    freeze=read(run/'experiment_freeze.json')
    for name,h in freeze['source_sha256'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('source changed after freeze: '+name)
    for name,h in freeze['artifact_sha256'].items():
        if sha(run/name)!=h:raise RuntimeError('configuration/membership changed: '+name)
    if read(run/'dataset_validation.json')['status']!='PASS':raise RuntimeError('dataset gate not PASS')
    if read(run/'cpu_validation.json')['status']!='PASS':raise RuntimeError('CPU gate not PASS')
    scorer=ROOT/'results'/run.name/'scorer/validation.json'
    if read(scorer)['status']!='PASS':raise RuntimeError('scorer gate not PASS')
    return freeze

def verify_model_environment(run,model):
    import importlib.metadata
    frozen=read(run/'model_freeze.json')[model]
    for name,record in frozen['files'].items():
        path=Path(frozen['snapshot'])/name
        if str(path.resolve())!=record['resolved'] or path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:
            raise RuntimeError('model/processor artifact changed: '+str(path))
    env=read(run/'environment.json')[model]
    if sys.version!=env['python']:raise RuntimeError('Python environment changed')
    for name,version in env['packages'].items():
        if importlib.metadata.version(name)!=version:raise RuntimeError('package version changed: '+name)

def committed(phasepath,phase):
    done={}
    for cp in sorted((phasepath/phase/'images').glob('*/COMMITTED.json')):
        c=read(cp);folder=cp.parent/c['attempt_id']
        for name,h in c['evidence_sha256'].items():
            if sha(folder/name)!=h:raise RuntimeError('committed evidence changed: '+str(folder/name))
        archive=c.get('journal_archives',{}).get('raw.jsonl')
        if archive:
            h=hashlib.sha256()
            with gzip.open(folder/archive['path'],'rb') as f:
                for b in iter(lambda:f.read(1<<20),b''):h.update(b)
            if h.hexdigest()!=c['raw_sha256']:raise RuntimeError('uncompressed raw SHA differs')
        done[cp.parent.name]=c
        remove_archived_duplicates(cp.parent)
    return done

def config_for(run,model,phase):
    phasepath=run/model/phase;manifest=read(phasepath/'manifest.json')
    config=read(run/model/'model_settings.json');freeze=read(run/'experiment_freeze.json')
    config.update(source_sha256=freeze['source_sha256'],_file_sha256=sha(run/model/'model_settings.json'),_manifest_sha256=sha(phasepath/'manifest.json'))
    return phasepath,manifest,config

def check_admission(run,model,phase):
    if phase=='gate':return
    gate=read(run/model/'gate/COMPLETE.json')
    if gate['status']!='PASS':raise RuntimeError('fresh GPU gates not PASS')
    predecessors={'smoke':[], 'pilot250':['smoke'], 'pilot40':['smoke','pilot250'], 'main':['smoke','pilot250','pilot40']}[phase]
    for prior in predecessors:
        receipt=read(run/model/prior/'COMPLETE.json')
        if receipt['status']!='PASS':raise RuntimeError('phase prerequisite not PASS: '+prior)
    if phase=='main':
        p=read(run/model/'main_resource_admission.json')
        if p['status']!='PASS' or p['pilot40_receipt_sha256']!=sha(run/model/'pilot40/COMPLETE.json'):raise RuntimeError('pilot-derived resource admission missing')
        if shutil.disk_usage(run).free<p['required_free_bytes']:raise RuntimeError('pilot-based raw/validation reserve not available')

def cleanup_glm(attempt,receipt):
    root=attempt/'stores';allow=read(attempt/'payload_evidence.json')
    if not root.exists():return
    expected={str((attempt/name).relative_to(root)):v['sha256'] for name,v in allow.items()}
    if receipt['audit']['status']!='PASS' or not (attempt.parent/'COMMITTED.json').exists():raise ValueError('audit/commit required')
    files={str(p.relative_to(root)):p for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    if set(files)!=set(expected):raise ValueError('payload allowlist mismatch')
    for name,p in files.items():
        if p.is_symlink() or p.stat().st_nlink!=1 or not p.resolve().is_relative_to(root.resolve()) or sha(p)!=expected[name]:raise ValueError('payload ownership/hash mismatch')
    for p in files.values():p.unlink()
    for p in sorted((p for p in root.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):p.rmdir()
    root.rmdir()
    dump(attempt/'cleanup_receipt.json',{'status':'PASS','files':len(expected),'scope':'only new attempt/stores payload; full hashes and raw preserved','regenerate':'repeat full image T1/three turns under same frozen config in a new attempt'})

def execute_glm_image(runner,M,run,phase,phasepath,manifest,config,d,ordinal):
    iid=d['image_id'];parent=phasepath/phase/'images'/iid;parent.mkdir(parents=True,exist_ok=True)
    if (parent/'COMMITTED.json').exists():raise FileExistsError('already committed')
    attempt=parent/f'attempt_{len(list(parent.glob("attempt_*")))+1:04d}';attempt.mkdir()
    turnmap={str(t['question_id']):t for t in d['turns']}
    original=M.append
    def append(handle,row):
        if handle.name.endswith('/raw.jsonl'):
            row.update(config_sha256=config['_file_sha256'],manifest_sha256=config['_manifest_sha256'],code_sha256=digest(config['source_sha256']))
            enrich(row,turnmap[str(row['question_id'])],runner.processor.tokenizer,'glm')
        return original(handle,row)
    M.append=append
    execution=copy.deepcopy(d);execution.update(execution_dialog_id=d['dialog_id'],image_path=str(ROOT/d['image_path']),image_sha256=manifest['image_identity'][iid]['sha256'])
    try:
        receipt=M.evaluate_image(runner,run,phase,iid,[(ordinal,execution)],attempt)
        rows=read_rows(attempt/'raw.jsonl')
        for row in rows:
            if row['annotation']!=turnmap[str(row['question_id'])]['annotation']:raise ValueError('GLM official GT mapping')
        t1=[r['generated_token_ids'] for r in rows if r['turn_id']==1]
        if any(x!=t1[0] for x in t1):raise ValueError('GLM arms disagree at T1')
        commit={'status':'COMMITTED','image_id':iid,'attempt_id':attempt.name,'requests':len(rows),
            'raw_sha256':sha(attempt/'raw.jsonl'),'config_sha256':config['_file_sha256'],'manifest_sha256':config['_manifest_sha256'],
            'first_successful_complete_image_adoption':True,'evidence_sha256':{p.name:sha(p) for p in attempt.iterdir() if p.is_file()}}
        archive_before_commit(parent,commit,dump)
        cleanup_glm(attempt,receipt);remove_archived_duplicates(parent)
        return len(rows)
    except BaseException as exc:
        dump(attempt/'FAILURE.json',{'status':'FAIL','error':repr(exc),'traceback':traceback.format_exc(),'adopted':0});raise
    finally:M.append=original;runner.close()

def run_phase(run,model,phase,resume):
    import torch,numpy as np,random
    torch.set_num_threads(2);torch.manual_seed(1234);random.seed(1234);np.random.seed(1234)
    freeze=verify_freeze(run);verify_model_environment(run,model);check_admission(run,model,phase)
    phasepath,manifest,config=config_for(run,model,phase)
    if (phasepath/'COMPLETE.json').exists():raise FileExistsError('completed phase is immutable')
    done=committed(phasepath,phase)
    if done and not resume:raise RuntimeError('existing committed images; pass --resume')
    if any(c['config_sha256']!=config['_file_sha256'] or c['manifest_sha256']!=config['_manifest_sha256'] for c in done.values()):raise RuntimeError('resume freeze mismatch')
    status(run,model,phase,'RUNNING',adopted_requests=sum(c['requests'] for c in done.values()),target_requests=manifest['requests'])
    if model=='glm':
        M=load('_mtvqa_glm_execution','121_eval_glm46v_prefixstore3.py');M.guard_gpu()
        runner=M.GLM46VRunner(ROOT/'runs/glm46v_prefixstore3_20261007T065207Z/checkpoint').load()
        fingerprint=runner.fingerprint();servers=None
    else:
        M,A=bind(model)
        M._experiment_id=run.name
        if model=='llava':runner=M.V.load_runner();servers=M.servers_for(runner);M.QA._warmup(runner,servers['old']);fingerprint={'model_id':M.MODEL_ID,'revision':M.REVISION,'device':str(runner.model.device),'backend':runner.attn,'dtype':str(runner.model.dtype)}
        else:runner=M.load_runner();servers=None;M._active_runner=runner;fingerprint=runner.runtime_fingerprint()
    dump(phasepath/f'runtime_{time.time_ns()}.json',fingerprint)
    count=sum(c['requests'] for c in done.values());started=time.time()
    frozen_paths=[ROOT/p for p in freeze['source_sha256']]+[run/p for p in freeze['artifact_sha256']]
    signatures={p:(p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns) for p in frozen_paths}
    try:
        for ordinal,d in enumerate(manifest['dialogues']):
            iid=d['image_id']
            if iid in done:continue
            if any((p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns)!=sig for p,sig in signatures.items()):raise RuntimeError('frozen source/config metadata changed during execution')
            if model=='glm':n=execute_glm_image(runner,M,run,phase,phasepath,manifest,config,d,ordinal)
            elif model=='llava':n=M.run_image(runner,servers,[d],{d['dialog_id']:ordinal},phase,phasepath,manifest,config)
            else:n=M.run_image(runner,[d],{d['dialog_id']:ordinal},phase,phasepath,manifest,config)
            if model!='glm':remove_archived_duplicates(phasepath/phase/'images'/iid)
            count+=n
            status(run,model,phase,'RUNNING',adopted_requests=count,target_requests=manifest['requests'],completed_images=count//(3*len(config['methods'])),elapsed_seconds=time.time()-started,last_image_id=iid)
            print(json.dumps({'model':model,'phase':phase,'image':iid,'adopted':count,'target':manifest['requests']}),flush=True)
        assert count==manifest['requests']
        # Every image already passed independent raw/store audit and archive hashing.
        dump(phasepath/'COMPLETE.json',{'status':'PASS','model':model,'phase':phase,'images':len(manifest['dialogues']),
            'requests':count,'expected_requests':manifest['requests'],'elapsed_seconds':time.time()-started,
            'source_freeze_sha256':sha(run/'experiment_freeze.json'),'manifest_sha256':config['_manifest_sha256'],
            'standalone_analysis':'PENDING','not_main_completion':phase!='main'})
        status(run,model,phase,'COMPLETE_PENDING_ANALYSIS',adopted_requests=count,target_requests=manifest['requests'])
    finally:
        if hasattr(runner,'close'):runner.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--model',choices=['llava','qwen','glm'],required=True);p.add_argument('--phase',choices=['smoke','pilot250','pilot40','main'],required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    try:run_phase(a.run.resolve(),a.model,a.phase,a.resume)
    except BaseException as exc:
        status(a.run.resolve(),a.model,a.phase,'BLOCKED',error=repr(exc),traceback=traceback.format_exc());raise

#!/usr/bin/env python3
"""Sequential durable gate/pilot/main manager; never equate launch with completion."""
import argparse,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mmimpress.mtvqa25k.common import read,dump,sha

def main(run,stop_after):
    lock=(run/'manager.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    python={'llava':'/home/dblab/anaconda3/envs/mllm_ft/bin/python','qwen':'/home/dblab/anaconda3/envs/mllm_ft/bin/python','glm':str(ROOT/'.venv-glm46v/bin/python')}
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH='.:scripts:tests')
    blocked={};events=run/'manager_events.jsonl';start=time.time();child=None
    resume=f'{sys.executable} scripts/144_manage_mtvqa25k.py --run {run}'
    (run/'manager.pid').write_text(str(os.getpid())+'\n')
    def event(**record):
        record.update(time_unix=time.time(),manager_pid=os.getpid())
        with events.open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
    def execute(model,stage,command):
        nonlocal child
        log=run/f'{model}_{stage}_{time.time_ns()}.log'
        event(model=model,stage=stage,state='STARTED',command=command,log=str(log))
        with log.open('x') as out:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
            while child.poll() is None:
                dump(run/'MANAGER_STATUS.json',{'status':'RUNNING','model':model,'stage':stage,'pid':os.getpid(),'child_pid':child.pid,
                    'heartbeat_unix':time.time(),'started_unix':start,'log':str(log),'resume_command':resume,'blocked_models':blocked},replace=True)
                time.sleep(10)
            code=child.returncode
        event(model=model,stage=stage,state='RETURNED',exit_code=code,log=str(log));child=None
        return code==0
    def report():
        subprocess.run([python['llava'],'scripts/143_report_mtvqa25k_status.py','--run',str(run)],cwd=ROOT,env=env,stdout=subprocess.DEVNULL,check=True)
    try:
        for phase in ('gate','smoke','pilot250','pilot40','main'):
            for model in ('llava','qwen','glm'):
                if model in blocked:continue
                dest=run/model/phase/'COMPLETE.json'
                if not dest.exists():
                    if phase=='gate':
                        # A failed gate needs diagnosis, not a blind automatic replay.
                        failures=list((run/model/'gate').glob('validation_*/FAILURE.json'))
                        if failures:blocked[model]={'stage':phase,'reason':'unresolved prior gate failure','evidence':str(failures[-1])};continue
                    if phase=='main':
                        if not execute(model,'resource_admission',[python[model],'scripts/145_admit_mtvqa25k_main.py','--run',str(run),'--model',model]):
                            blocked[model]={'stage':'main','reason':'pilot-based resource admission failed'};continue
                    command=[python[model],f'scripts/{"139_validate_mtvqa25k_gpu.py" if phase=="gate" else "138_run_mtvqa25k.py"}','--run',str(run),'--model',model]
                    if phase!='gate':command+=['--phase',phase,'--resume']
                    if not execute(model,phase,command):
                        blocked[model]={'stage':phase,'reason':'child failed; see stage status/log'};report();continue
                if read(dest)['status']!='PASS':blocked[model]={'stage':phase,'reason':'non-PASS completion'};continue
                if phase!='gate':
                    out=ROOT/'results'/run.name/model/phase
                    if not (out/'independent_audit.json').exists():
                        if out.exists():
                            blocked[model]={'stage':phase,'reason':'incomplete prior analysis; inspect without overwriting'};continue
                        if not execute(model,phase+'_analysis',[python[model],'scripts/142_analyze_mtvqa25k.py','--run',str(run),'--model',model,'--phase',phase,'--output',str(out)]):
                            blocked[model]={'stage':phase,'reason':'independent analysis failed'};report();continue
                    if read(out/'independent_audit.json')['status']!='PASS':blocked[model]={'stage':phase,'reason':'independent analysis non-PASS'}
                report()
            if phase==stop_after:break
        complete=all((ROOT/'results'/run.name/m/'main/independent_audit.json').exists() and read(ROOT/'results'/run.name/m/'main/independent_audit.json')['adopted_requests']==t for m,t in [('llava',375000),('qwen',225000),('glm',225000)])
        dump(run/'MANAGER_STATUS.json',{'status':'COMPLETE' if complete else 'PARTIAL_BLOCKED' if blocked else 'STOPPED_AT_REQUESTED_STAGE','pid':os.getpid(),'heartbeat_unix':time.time(),'blocked_models':blocked,'resume_command':resume},replace=True)
        report();event(state='FINISHED',complete=complete,blocked=blocked)
    except BaseException as exc:
        dump(run/'MANAGER_STATUS.json',{'status':'INTERRUPTED','pid':os.getpid(),'heartbeat_unix':time.time(),'child_pid':child.pid if child else None,'error':repr(exc),'resume_command':resume},replace=True)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stop-after',choices=['gate','smoke','pilot250','pilot40','main'],default='main');a=p.parse_args();main(a.run.resolve(),a.stop_after)

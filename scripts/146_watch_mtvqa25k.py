#!/usr/bin/env python3
"""Detached lifecycle watchdog; does not retry scientific/resource gate failures."""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mmimpress.mtvqa25k.common import read,dump,sha

def owned_processes(run):
    found=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv=path.read_bytes().split(b'\0')
            text=b' '.join(argv).decode(errors='replace')
            if run.name not in text:continue
            scripts=('scripts/144_manage_mtvqa25k.py','scripts/138_run_mtvqa25k.py','scripts/139_validate_mtvqa25k_gpu.py','scripts/142_analyze_mtvqa25k.py','scripts/145_admit_mtvqa25k_main.py')
            if any(any(arg.decode(errors='replace').endswith(script) for arg in argv) for script in scripts):found.append(int(path.parent.name))
        except (OSError,ValueError):continue
    return found

def main(run):
    lock=(run/'watchdog.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    frozen=read(run/'experiment_freeze.json')['source_sha256'];manager=ROOT/'scripts/144_manage_mtvqa25k.py'
    if sha(manager)!=frozen['scripts/144_manage_mtvqa25k.py']:raise RuntimeError('manager differs from freeze')
    launch_time=time.time();restarts=0
    while True:
        state=read(run/'MANAGER_STATUS.json') if (run/'MANAGER_STATUS.json').exists() else {}
        active=owned_processes(run)
        record={'pid':os.getpid(),'heartbeat_unix':time.time(),'started_unix':launch_time,'owned_execution_pids':active,
                'unexpected_manager_restarts':restarts,'manager_status':state.get('status'),'watchdog_source_sha256':sha(Path(__file__)),
                'policy':'wait for owned manager/worker/analysis; resume only after unexpected process exit; never retry a declared validation/resource BLOCKED'}
        if not active and state.get('status') in ('COMPLETE','PARTIAL_BLOCKED','STOPPED_AT_REQUESTED_STAGE'):
            dump(run/'WATCHDOG_STATUS.json',dict(record,status='FINISHED'),replace=True);return
        if not active:
            if restarts>=2:
                dump(run/'WATCHDOG_STATUS.json',dict(record,status='BLOCKED_RESTART_LIMIT'),replace=True);return
            if sha(manager)!=frozen['scripts/144_manage_mtvqa25k.py']:raise RuntimeError('frozen manager changed')
            log=run/f'manager_restart_{time.time_ns()}.log'
            with log.open('x') as out:
                child=subprocess.Popen([sys.executable,'-u',str(manager),'--run',str(run)],cwd=ROOT,stdin=subprocess.DEVNULL,
                    stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            restarts+=1;record.update(restarted_manager_pid=child.pid,restart_log=str(log))
        dump(run/'WATCHDOG_STATUS.json',dict(record,status='MONITORING'),replace=True)
        time.sleep(10)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run.resolve())

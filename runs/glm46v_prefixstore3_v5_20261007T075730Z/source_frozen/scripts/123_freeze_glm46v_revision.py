#!/usr/bin/env python3
"""Preserve a failed/pre-quality freeze and publish a new immutable source revision."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_revision(run, reason):
    for phase in ('gqa_pilot','mt_pilot','main'):
        if list((run/phase).glob('images/*/attempt_*/raw.jsonl')):
            raise RuntimeError('cannot replace the admitted source after pilot/main has started')
    active=json.loads((run/'active_freeze.json').read_text()) if (run/'active_freeze.json').exists() else {'path':'freeze.json'}
    previous=run/active['path']
    old=json.loads(previous.read_text())
    number=old.get('freeze_version',1)+1
    version=run/'freezes'/f'pre_pilot_v{number}'
    version.mkdir(parents=True,exist_ok=False)
    files=[*ROOT.glob('mmimpress/glm46v/*.py'),*ROOT.glob('scripts/*glm46v*.py'),
           *ROOT.glob('tests/test_glm46v*.py'),*ROOT.glob('docs/glm46v*.md')]
    sources={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}
    for p in files:
        dest=version/'source_frozen'/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    cfg=json.loads((run/old.get('config_path','config.json')).read_text())
    cfg.update(source_sha256=sources,freeze_version=number,revision_reason=reason,
               created_at=datetime.now(timezone.utc).isoformat())
    cfg['contract_sha256']=sha(ROOT/'docs/glm46v_prefixstore3_contract.md')
    shutil.copy2(ROOT/'docs/glm46v_prefixstore3_contract.md',version/'CONTRACT.md')
    frozen=dict(old,source_sha256=sources,config_path=str((version/'config.json').relative_to(run)),
        contract_path=str((version/'CONTRACT.md').relative_to(run)),
        contract_sha256=cfg['contract_sha256'],freeze_version=number,
        prior_freeze_sha256=sha(previous),revision_reason=reason,
        pilot_quality_observed=False,main_quality_observed=False)
    for p,value in [(version/'config.json',cfg)]:
        with p.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    frozen['config_sha256']=sha(version/'config.json')
    with (version/'freeze.json').open('x') as f:
        json.dump(frozen,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    pointer=dict(path=str((version/'freeze.json').relative_to(run)),sha256=sha(version/'freeze.json'),
                 prior_freeze_path=active['path'],prior_sha256=sha(previous),reason=reason)
    pending=run/'active_freeze.pending.json'
    with pending.open('x') as f:json.dump(pointer,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(pending,run/'active_freeze.json')
    return pointer


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--reason',required=True)
    args=parser.parse_args()
    print(json.dumps(freeze_revision(args.run_dir.resolve(),args.reason),ensure_ascii=False))

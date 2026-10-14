#!/usr/bin/env python3
"""Run archived faulty source and fixed source in one fresh CPU fixture."""
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from mmimpress.serve import Server

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def main():
    p=argparse.ArgumentParser();p.add_argument('--before-source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    f=load('_full_load_regression_fixture',ROOT/'tests/test_mpic_multi.py').MultiFixture();f.setUp()
    old=load('_archived_faulty_serve',a.before_source)
    records=[];contexts=[]
    try:
        from unittest.mock import patch
        for name,cls in [('before',old.Server),('after',Server)]:
            ctx=f.fullctx(0);contexts.append(ctx)
            with patch('torch.cuda.synchronize',lambda *a,**kw:None):
                result=cls(f.runner,max_new_tokens=1).request(ctx,mode='fullload',suffix_ids=torch.tensor([3]),cold=False)
            records.append(dict(version=name,io=result['io'],image_K_nonzero=[int(torch.count_nonzero(x[:,:,2:42])) for x in ctx.cache.k],
                                image_V_nonzero=[int(torch.count_nonzero(x[:,:,2:42])) for x in ctx.cache.v]))
        assert records[0]['io']['bytes']==0 and records[1]['io']['bytes']>0
        assert all(x==0 for x in records[0]['image_K_nonzero']+records[0]['image_V_nonzero'])
        assert all(x>0 for x in records[1]['image_K_nonzero']+records[1]['image_V_nonzero'])
        with a.out.open('x') as h:json.dump(dict(status='PASS (bug reproduced then fixed)',device='CPU',records=records,
            old_source_sha256=hashlib.sha256(a.before_source.read_bytes()).hexdigest(),
            new_source_sha256=hashlib.sha256((ROOT/'mmimpress/serve.py').read_bytes()).hexdigest(),GPU='NOT VERIFIED'),h,indent=2);h.write('\n')
        print(json.dumps(dict(before_KV_bytes=records[0]['io']['bytes'],after_KV_bytes=records[1]['io']['bytes'],GPU='NOT VERIFIED')))
    finally:
        for c in contexts:c.reader.close()
        f.tearDown()
if __name__=='__main__':main()

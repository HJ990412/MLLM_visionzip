#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mmimpress.mtvqa25k.dataset import build,verify
from mmimpress.mtvqa25k.common import dump
p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,default=ROOT/'data/mt_vqa_v2_reconstructed_25k');p.add_argument('--verify',action='store_true');p.add_argument('--receipt',type=Path);a=p.parse_args()
if a.verify:
    result=verify(a.dataset)
    if a.receipt:dump(a.receipt,result)
    print(json.dumps(result),flush=True)
else:build(a.dataset)

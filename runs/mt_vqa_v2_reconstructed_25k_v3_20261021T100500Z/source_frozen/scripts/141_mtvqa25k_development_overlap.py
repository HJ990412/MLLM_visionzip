#!/usr/bin/env python3
"""Exact file/decoded-pixel overlap with accessible prior local development images."""
import argparse,json,os,sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mmimpress.mtvqa25k.common import read,dump,sha
from mmimpress.mtvqa25k.dataset import image_record

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
selected=read(ROOT/'data/mt_vqa_v2_reconstructed_25k/image_manifest.json')
files={x['sha256']:x['image_id'] for x in selected};pixels={x['decoded_rgb_sha256']:x['image_id'] for x in selected}
paths=[]
for directory,dirs,names in os.walk(ROOT/'data'):
    dirs[:]=[d for d in dirs if d!='mt_vqa_v2_reconstructed_25k']
    for n in names:
        path=Path(directory)/n
        if path.suffix.lower() in ('.jpg','.jpeg','.png','.webp','.bmp'):paths.append(path)
summary=defaultdict(lambda:{'inspected_images':0,'decode_failures':0,'exact_file_matches':[],'exact_RGB_matches':[]})
records=[]
with ThreadPoolExecutor(max_workers=4) as pool:
    for row in pool.map(image_record,[(str(p.relative_to(ROOT)),p) for p in sorted(paths)]):
        group=Path(row['image_path']).parts[1];s=summary[group];s['inspected_images']+=1
        if row['status']!='PASS':s['decode_failures']+=1;records.append(row);continue
        if row['sha256'] in files:s['exact_file_matches'].append({'path':row['image_path'],'selected_image_id':files[row['sha256']]})
        if row['decoded_rgb_sha256'] in pixels:s['exact_RGB_matches'].append({'path':row['image_path'],'selected_image_id':pixels[row['decoded_rgb_sha256']]})
        records.append(row)
dump(a.output,{'scope':'all accessible raster files under pre-existing project data directories, excluding new dataset; exact RGB including dimensions, no perceptual/resize inference',
    'universal_unseen_holdout_claim':False,'summary':dict(summary),'records':records})
print(json.dumps({k:{'inspected':v['inspected_images'],'file_matches':len(v['exact_file_matches']),'pixel_matches':len(v['exact_RGB_matches']),'decode_failures':v['decode_failures']} for k,v in summary.items()}))

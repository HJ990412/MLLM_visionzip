#!/usr/bin/env python3
"""Metadata-only native visual geometry. Run in each model's pinned environment."""
import argparse,inspect,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from mmimpress.mtvqa25k.common import read,dump,sha
from mmimpress.mtvqa25k.dataset import distribution

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['llava','qwen','glm'],required=True);p.add_argument('--dataset',type=Path,default=ROOT/'data/mt_vqa_v2_reconstructed_25k');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import torch,transformers
    from transformers import AutoConfig,AutoProcessor
    torch.set_num_threads(2)
    snapshot={'llava':Path.home()/'.cache/huggingface/hub/models--llava-hf--llava-v1.6-vicuna-7b-hf/snapshots/c916e6cdcd760b4cecd1dd4907f84ac649f93b23',
              'qwen':Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5',
              'glm':ROOT/'runs/glm46v_prefixstore3_20261007T065207Z/checkpoint'}[a.model]
    config=AutoConfig.from_pretrained(snapshot,local_files_only=True)
    if a.model=='llava':
        from transformers.models.llava_next.modeling_llava_next import get_anyres_image_grid_shape,unpad_image
        base=config.vision_config.image_size//config.vision_config.patch_size
        def geometry(h,w):
            ph,pw=get_anyres_image_grid_shape([h,w],config.image_grid_pinpoints,config.vision_config.image_size)
            unpadded=unpad_image(torch.zeros(1,base*ph,base*pw),[h,w]);hh,ww=unpadded.shape[-2:]
            return {'N_content':base*base+hh*ww,'v_token_num':base*base+hh*(ww+1),'hires_grid':[hh,ww],'newline_count':hh}
        helper_files=[inspect.getfile(get_anyres_image_grid_shape),inspect.getfile(unpad_image)]
        processor=AutoProcessor.from_pretrained(snapshot,local_files_only=True,use_fast=False)
    elif a.model=='qwen':
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        def geometry(h,w):
            hh,ww=smart_resize(h,w,factor=28,min_pixels=200704,max_pixels=802816)
            return {'N_content':hh*ww//784,'image_grid_thw':[[1,hh//14,ww//14]],'resized_hw':[hh,ww]}
        helper_files=[inspect.getfile(smart_resize)]
        processor=AutoProcessor.from_pretrained(snapshot,local_files_only=True,min_pixels=200704,max_pixels=802816)
    else:
        from transformers.models.glm46v.image_processing_glm46v import smart_resize
        def geometry(h,w):
            hh,ww=smart_resize(2,h,w,factor=28,min_pixels=12544,max_pixels=1605632)
            return {'N_content':hh*ww//784,'image_grid_thw':[[1,hh//14,ww//14]],'resized_hw':[hh,ww]}
        helper_files=[inspect.getfile(smart_resize)]
        processor=AutoProcessor.from_pretrained(snapshot,local_files_only=True,size={'shortest_edge':12544,'longest_edge':1605632},use_fast=True)
    rows=read(a.dataset/'source_image_manifest.json');items={}
    for r in rows:
        if r['status']!='PASS':continue
        try:
            g=geometry(r['height'],r['width']);n=g['N_content'];k=(n+3)//4
            items[str(r['image_id'])]={**g,'k':k,'k_mod64':k%64,'status':'PASS'}
        except Exception as e:items[str(r['image_id'])]={'status':'BLOCKED','error':repr(e)}
    ds=read(a.dataset/'dialogues.json')['dialogues']
    texts=[t['question'] for d in ds for t in d['turns']]
    tokenized=processor.tokenizer(texts,add_special_tokens=False)['input_ids']
    for i,d in enumerate(ds):items[str(d['image_id'])]['question_token_lengths']=[len(x) for x in tokenized[3*i:3*i+3]]
    selected=[items[str(d['image_id'])]['N_content'] for d in ds if items[str(d['image_id'])]['status']=='PASS']
    # Existing 250 pilot PNG geometry: sizes are checked independently by PIL.
    from PIL import Image
    pilot={}
    for d in read(ROOT/'data/mt_vqa_v2/dialogues.json')['dialogues']:
        with Image.open(ROOT/d['image_path']) as im:g=geometry(im.height,im.width)
        pilot[d['image_id']]=g
    dump(a.output,{'model':a.model,'transformers':transformers.__version__,'python':sys.version,'snapshot':str(snapshot),
        'helper_sha256':{f:sha(f) for f in sorted(set(helper_files))},'items':items,'pilot250':pilot,
        'source_N_distribution':distribution([x['N_content'] for x in items.values() if x['status']=='PASS']),
        'selected_N_distribution':distribution(selected),'selected_incompatible_images':len(ds)-len(selected),
        'provenance':'native dimension helpers; predicted N, GPU actual N must agree before adoption'})
    print(json.dumps({'model':a.model,'images':len(ds),'N_min':min(selected),'N_max':max(selected),'blocked':len(ds)-len(selected)}))

if __name__=='__main__':main()

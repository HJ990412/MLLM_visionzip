#!/usr/bin/env python3
"""Fresh CPU layer evidence and FullLoad regression; pretrained GPU status explicit."""
import argparse, hashlib, importlib.util, json, os, platform, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from mmimpress.mpic import MPICRawReader
from mmimpress.mpic_multi import MultiSelectivePrefill, expand_multi_prompt, prepare_mixed
from mmimpress.store import IOCounter

def dump(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False);f.write('\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('multi_fixture',ROOT/'tests/test_mpic_multi.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    fixture=m.MultiFixture();fixture.setUp();rows=[]
    try:
        for swapped in (False,True):
            ids,bindings=fixture.input(swapped)
            for policy,k in [('per_image',0),('per_image',32),('per_image',None),('global',32)]:
                io=IOCounter();logits,cache,plan,stats=MultiSelectivePrefill(fixture.runner,k_recompute=k,selection_policy=policy).run(ids,bindings,io,trace_tensors=True)
                ref,expected=fixture.independent(ids,bindings,k,policy)
                torch.testing.assert_close(logits,ref,rtol=1e-5,atol=1e-6)
                values=stats.pop('diagnostic_tensors')
                name=f'order{int(swapped)}_{policy}_k{k}'
                torch.save(values,a.out/(name+'_tensors.pt'))
                for pair,oracle,tensor in zip(cache,expected,values):
                    for x,y in zip(pair,oracle[:2]):torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-6)
                    torch.testing.assert_close(tensor['hidden'],oracle[2],rtol=1e-5,atol=1e-6)
                rows.append(dict(case=name,status='PASS',logit_max_abs=float((logits-ref).abs().max()),
                    input_ids=ids.tolist(),token_types=['image' if int(x)==100 else 'text' for x in ids],io=io.summary(),**stats))
        # Genuine CPU miss decoder computation runs while the hit worker reads
        # real files. No artificial sleeps or simulated transfer times.
        mixed=[];outputs=[]
        for parallel in (False,True):
            fresh=[]
            def compute(image_id):
                f=fixture;c=f.contexts[1];n=c.meta['v_token_num'];ids=torch.tensor([1,2]+[100]*n+[3,4])
                with torch.no_grad():
                    emb=f.lm.get_input_embeddings()(ids[None]);emb[:,2:2+n]=f.visual['1'][None]
                    output=f.lm(inputs_embeds=emb,use_cache=True)
                    # Existing fixture raw tensors came from exactly this source
                    # computation. Check miss output before returning its context.
                    for actual,(ek,ev) in zip(output.past_key_values.layers,f.source[1]):
                        torch.testing.assert_close(actual.keys,ek,rtol=1e-5,atol=1e-6)
                        torch.testing.assert_close(actual.values,ev,rtol=1e-5,atol=1e-6)
                    folder=f.root/f'miss_{int(parallel)}';folder.mkdir()
                    for li,layer in enumerate(output.past_key_values.layers):
                        sub=folder/f'layer_{li:02d}';sub.mkdir()
                        for kind,value in [('k',layer.keys),('v',layer.values)]:
                            (sub/f'{kind}.bin').write_bytes(value[0,:,2:2+n].permute(1,0,2).contiguous().numpy().tobytes())
                    (folder/'visual_input.bin').write_bytes(f.visual['1'].numpy().tobytes())
                from types import SimpleNamespace
                context=SimpleNamespace(dir=folder,meta=dict(c.meta),reader=MPICRawReader(folder,c.meta,drop_cache=False))
                fresh.append(context);return context
            contexts,record=prepare_mixed(['0','1'],{'0':fixture.contexts[0]},compute,selected_counts=[32,32],parallel=parallel)
            ids,b=expand_multi_prompt(fixture.runner,'t1 t5 t2 <image> t3 t6 <image> t7 t8',contexts,['0','1'])
            logits,_,_,_=MultiSelectivePrefill(fixture.runner,selection_policy='per_image').run(ids,b,IOCounter())
            outputs.append(logits);mixed.append(record)
            for context in fresh:context.reader.close()
        torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)
        dump(a.out/'mixed_timeline.json',dict(device='CPU',records=mixed,serial_parallel_logits='PASS',GPU_overlap='NOT VERIFIED'))
        fixture.test_full_load_one_multiple_decode_prefix_and_cache_identity()
        dump(a.out/'fullload_validation.json',dict(status='PASS',device='CPU',checks=['suffix1','suffix2','decode no extra reload',
            'wrong prefix rejection','wrong cache ID rejection','image0/image1/image0 installed K/V exact'],GPU='NOT VERIFIED'))
    finally:fixture.tearDown()
    dump(a.out/'layer_traces.json',rows)
    env=dict(client_date='2026-10-13',python=sys.version,platform=platform.platform(),torch=torch.__version__,
        cuda_available=torch.cuda.is_available(),cuda_devices=torch.cuda.device_count(),rtol=1e-5,atol=1e-6)
    s=subprocess.run(['nvidia-smi'],capture_output=True,text=True);env['nvidia_smi']=dict(returncode=s.returncode,stderr=s.stderr,stdout=s.stdout)
    dump(a.out/'environment.json',env)
    dump(a.out/'summary.json',dict(selective_conditions=8,observed_layers=16,CPU='PASS',FullLoad_CPU='PASS',
        pretrained_GPU='NOT VERIFIED' if not torch.cuda.is_available() else 'NOT VERIFIED (run GPU functional gate)',
        max_logit_error=max(x['logit_max_abs'] for x in rows),MMDU_generated_dialogues=0))
    print(json.dumps(dict(conditions=8,layers=16,CPU='PASS',pretrained_GPU='NOT VERIFIED')))
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Native pretrained FullLoad A/B/A isolation using already captured GPU pilot KV."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch
from mmimpress.model import LlavaRunner
from mmimpress.mpic import MPICContext
from mmimpress.store import write_image_store
from mmimpress.serve import ImageContext,Server

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-dialog',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    snap='/home/dblab/.cache/huggingface/hub/models--llava-hf--llava-v1.6-vicuna-7b-hf/snapshots/c916e6cdcd760b4cecd1dd4907f84ac649f93b23'
    r=LlavaRunner(model_id=snap).load();r.model_id='llava-hf/llava-v1.6-vicuna-7b-hf'
    image_ctx=[MPICContext(a.source_dialog/f'image_{i}',r.model.device,drop_cache=False,runner=r) for i in (0,1)]
    syskv=torch.load(a.source_dialog/'fullload_gpu_gate_store/sys_kv.pt',weights_only=True,map_location='cpu')
    ctx=[]
    try:
        for i,c in enumerate(image_ctx):
            assert c.meta['v_token_start']==image_ctx[0].meta['v_token_start']
            assert c.meta['prefix_input_ids'][:c.meta['v_token_start']]==image_ctx[0].meta['prefix_input_ids'][:image_ctx[0].meta['v_token_start']]
            layers=[]
            for li in range(len(r.layers)):
                pair=[]
                for kind in ('k','v'):
                    _,value=c.reader.read_reused_layer(li,kind,0)
                    image=value.permute(1,0,2)[None]
                    pair.append(torch.cat((syskv[kind][li][None],image),dim=2))
                layers.append(tuple(pair))
            path=a.out/f'raster_{i}'
            write_image_store(path,layers,c.meta['v_token_start'],c.meta['v_token_num'],c.meta['prefix_input_ids'],c.meta['newline_idx'],
                              extra={'image_id':c.meta['image_id']},separator_sidecar=True)
            ctx.append(ImageContext(path,r.model.device,drop_cache=False,require_v_hidden=False))
        records=[]
        for i,suffix in [(0,[13]),(1,[13]),(0,[13]),(1,[13,14])]:
            c=ctx[i];native=[];handles=[]
            for li,layer in enumerate(r.layers):
                def observer(module,args,kwargs,result,li=li):
                    cache=kwargs['past_key_values'];key,value=cache.layers[li].keys,cache.layers[li].values
                    start,n=c.meta['v_token_start'],c.meta['v_token_num']
                    native.append(dict(layer=li,key_shape=list(key.shape),value_shape=list(value.shape),
                        K_equal=bool(torch.equal(key[:,:,start:start+n],c.cache.k[li][:,:,start:start+n])),
                        V_equal=bool(torch.equal(value[:,:,start:start+n],c.cache.v[li][:,:,start:start+n]))))
                handles.append(layer.self_attn.register_forward_hook(observer,with_kwargs=True))
            try:result=Server(r,max_new_tokens=3).request(c,mode='fullload',suffix_ids=torch.tensor(suffix),cache_id=c.meta['image_id'],cold=True)
            finally:
                for h in handles:h.remove()
            assert result['io']['preads']==64 and all(v['K_equal'] and v['V_equal'] for v in native)
            for li in range(len(r.layers)):
                for kind,buffer in [('k',c.cache.k[li]),('v',c.cache.v[li])]:
                    _,raw=image_ctx[i].reader.read_reused_layer(li,kind,0)
                    expected=raw.to(buffer.device,buffer.dtype).permute(1,0,2)[None]
                    assert torch.equal(buffer[:,:,c.meta['v_token_start']:c.meta['prefix_len']],expected)
            records.append(dict(image=i,suffix_tokens=len(suffix),generated=result['generated_token_ids'],status='PASS',
                                io=result['io'],installed_image_rows=c.meta['v_token_num'],native_attention_observations=native))
        assert records[0]['generated']==records[2]['generated']
        invalid={}
        for name,kwargs in [('cache_id',dict(suffix_ids=torch.tensor([13]),cache_id='wrong')),('prefix',dict(prompt_text='Other text USER: <image>\nQuestion? ASSISTANT:'))]:
            try:Server(r).request(ctx[0],mode='fullload',cold=False,**kwargs)
            except ValueError as exc:invalid[name]=str(exc)
            else:raise AssertionError('invalid request accepted')
        with (a.out/'validation.json').open('x') as f:json.dump(dict(status='PASS',records=records,invalid=invalid,
            system_KV_contract='shared exact same source pre-image IDs/positions, same native forward shape; image payloads independent',
            fallback=False),f,indent=2)
        print(json.dumps(dict(status='PASS',requests=4,sequence='A/B/A/B',invalid=2)))
    finally:
        for c in ctx:c.close()
        for c in image_ctx:c.close()
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Bounded official MMDU preparation / optional pretrained GPU smoke.

Maximum five dialogues, first three turns only. No gold-history fallback.
The official MMDU LLaVA-NeXT 336x336 resize is explicit and shared by methods.
GPU absence leaves all generation/TTFT/quality fields NOT VERIFIED.
"""
import argparse, csv, hashlib, json, os, platform, subprocess, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch
from PIL import Image
from transformers import AutoProcessor
from mmimpress.model import LlavaRunner, cache_layers
from mmimpress.piggyback import DecoderVisualHiddenCapture
from mmimpress.mpic import MPICContext, persist_captured_mpic_prefix, seed_dynamic_cache, _sync
from mmimpress.mpic_multi import (MultiSelectivePrefill, MPICMultiServer, expand_multi_prompt,
                                 decode_linked, prepare_mixed)
from mmimpress.mmdu_mpic import (prompt_from_generated, logical_labels, prefix_prefill, full_reuse_prefill, matching_prefix_length)
from mmimpress.store import IOCounter, write_image_store

REV='c916e6cdcd760b4cecd1dd4907f84ac649f93b23'
SNAPSHOT=Path('/home/dblab/.cache/huggingface/hub/models--llava-hf--llava-v1.6-vicuna-7b-hf/snapshots')/REV
BENCHMARK=ROOT/'data/mmdu/raw/benchmark.json'
METHODS=('prefix_caching','full_reuse','mpic_global32','mpic_per_image32')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def dump(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def append(path,obj):
    with Path(path).open('a') as f:f.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
def image_for(path):
    with Image.open(path) as image:return image.convert('RGB').resize((336,336),Image.Resampling.LANCZOS)
def prepare(processor,limit,seed,cap):
    items=json.loads(BENCHMARK.read_text());assert isinstance(items,list) and len(items)==110
    ordered=sorted(items,key=lambda x:hashlib.sha256(f'{seed}:{x["id"]}'.encode()).hexdigest())
    selected,decisions=[],[]
    for item in ordered:
        introduced=sum(q['value'].count('<ImageHere>') for q in item['conversations'][:6:2])
        reason='eligible'
        if introduced!=2:reason='first_three_turns_must_introduce_exactly_two_images_for_this_multi_image_smoke'
        images=[dict(image_id=f'mmdu:{item["id"]}:image{i+1}',path=str(ROOT/'data/mmdu'/s.lstrip('/')),
                     source_path=s) for i,s in enumerate(item['image'][:introduced])]
        dialog=dict(dialog_id=str(item['id']),images=images,turns=[])
        cursor=0
        for i in range(3):
            q,answer=item['conversations'][2*i:2*i+2];assert q['from']=='user' and answer['from']=='assistant'
            cursor+=q['value'].count('<ImageHere>')
            dialog['turns'].append(dict(turn_id=i+1,question=q['value'],reference_answer=answer['value'],active_images=cursor))
        bounds=[]
        if reason=='eligible':
            for im in images:
                path=Path(im['path'])
                if not path.is_file():reason='missing_image';break
                im['sha256']=sha(path)
                with Image.open(path) as img:im['original_size']=list(img.size)
            if reason=='eligible':
                for turn in (1,2,3):
                    # Empty history is used only for token capacity estimation.
                    # Its upper bound reserves cap tokens for each previous
                    # answer, the current answer, plus 64 boundary tokens.
                    prompt=prompt_from_generated(dialog,turn,{t:'' for t in range(1,turn)})
                    pics=[image_for(x['path']) for x in images[:dialog['turns'][turn-1]['active_images']]]
                    enc=processor(images=pics,text=prompt,return_tensors='pt')
                    tokens=int(enc['input_ids'].shape[-1]);bound=tokens+turn*cap+64
                    bounds.append(dict(turn=turn,base_input_tokens=tokens,reserved_context_bound=bound,
                                       actual_image_placeholder_tokens=int((enc['input_ids']==32000).sum())))
                    if bound>4096:reason='native_context_capacity_bound_exceeded'
                    del enc
        take=reason=='eligible' and len(selected)<limit
        decisions.append(dict(dialog_id=str(item['id']),introduced_images=introduced,status='selected' if take else 'excluded',
                              reason=reason if reason!='eligible' else 'quota_reached',bounds=bounds))
        if take:dialog['preflight']=bounds;selected.append(dialog)
    return selected,decisions

@torch.no_grad()
def visual_embeddings(runner,enc,bindings,previous,matched):
    model=runner.model;ids=enc['input_ids'].to(model.device)
    embeds=model.get_input_embeddings()(ids)
    misses=[i for i,b in enumerate(bindings) if b.start+b.count>matched]
    features=[]
    if misses:
        # enc contains only the images whose decoder span is outside the
        # matching prefix. Matched images pay neither pixel I/O nor encoding.
        features=model.model.get_image_features(enc['pixel_values'].to(model.device),enc['image_sizes'].to(model.device),
            vision_feature_layer=model.config.vision_feature_layer,
            vision_feature_select_strategy=model.config.vision_feature_select_strategy)
    mapping=dict(zip(misses,features))
    for i,b in enumerate(bindings):
        value=mapping[i].to(embeds.device,embeds.dtype)[None] if i in mapping else previous.embeds[:,b.start:b.start+b.count]
        if value.shape[1]!=b.count:raise AssertionError('native image feature count does not match current expanded span')
        embeds[:,b.start:b.start+b.count]=value
    return embeds,len(misses)

@torch.no_grad()
def provision(runner,image,path,*,retain_source=False):
    prompt='USER: <image>\nDescribe this image. ASSISTANT:'
    enc=runner.encode_prompt(image_for(image['path']),prompt);start,n=runner.visual_span(enc['input_ids'])
    capture=DecoderVisualHiddenCapture(runner,start,n)
    with capture:output=runner.model(**runner.to_device(enc),use_cache=True)
    persistence=persist_captured_mpic_prefix(runner,output.past_key_values,enc['input_ids'],enc['image_sizes'][0],
        capture.result_cpu(),path,image_id=image['image_id'],hidden_capture_stats=capture,image_input_sha256=image['sha256'],
        extra_metadata={'source_prompt':prompt,'benchmark_resize':[336,336],'provisioning_role':'independent_image_upload'})
    ctx=MPICContext(path,runner.model.device,drop_cache=False,runner=runner)
    source=(output.past_key_values,enc,start,n) if retain_source else None
    return ctx,persistence,source

class GPUResidentReader:
    def __init__(self,ctx,device,dtype):
        io=IOCounter();self.meta=ctx.meta
        self.visual=ctx.reader.read_visual_inputs(ctx.meta['v_token_num'],io).to(device,dtype)
        self.payload={}
        for li in range(ctx.meta['num_layers']):
            for kind in ('k','v'):
                rows,value=ctx.reader.read_reused_layer(li,kind,0,io)
                self.payload[li,kind]=(rows.to(device),value.to(device,dtype))
        _sync(torch.device(device));self.setup_io=io.summary()
    def read_visual_inputs(self,k,counter=None):return self.visual[:k]
    def read_reused_layer(self,li,kind,k,counter=None):
        rows,value=self.payload[li,kind];start=(k//self.meta['chunk_size'])*self.meta['chunk_size']
        if k==self.meta['v_token_num']:return rows[:0],value[:0]
        return rows[start:],value[start:]

@torch.no_grad()
def gpu_gate(runner,contexts,images,out,source,dialog=None):
    """Actual pretrained layers and image install, before any measured MMDU."""
    evidence=[];pair=contexts[:2];names=[c.meta['image_id'] for c in pair]
    fixtures=[('single','USER: <image>\nDescribe this image. ASSISTANT:',pair[:1],names[:1]),
        ('prefix_changed','A careful description. USER: <image>\nDescribe this image. ASSISTANT:',pair[:1],names[:1]),
        ('interleaved','USER: Text A <image> Text B <image> Text C. Compare. ASSISTANT:',pair,names),
        ('reordered','USER: Changed placement <image> and another <image>. Compare. ASSISTANT:',pair[::-1],names[::-1])]
    fixtures += [
        ('fixed_history_t2','USER: Text A <image> and text B <image>. Describe. ASSISTANT: controlled diagnostic answer</s> USER: Compare them. ASSISTANT:',pair,names),
        ('fixed_history_t3','USER: Text A <image> and text B <image>. Describe. ASSISTANT: controlled diagnostic answer</s> USER: Compare them. ASSISTANT: second controlled answer</s> USER: Explain the main difference. ASSISTANT:',pair,names)]
    if dialog is not None:
        fixtures.append(('frozen_dialogue_t1',prompt_from_generated(dialog,1,{}),pair,names))
    for name,prompt,ctxs,image_ids in fixtures:
        ids,bindings=expand_multi_prompt(runner,prompt,ctxs,image_ids)
        dtype=runner.model.get_input_embeddings().weight.dtype
        embeds=runner.model.get_input_embeddings()(ids.to(runner.model.device)[None])
        for b in bindings:embeds[:,b.start:b.start+b.count]=b.context.reader.read_visual_inputs(b.count).to(embeds.device,dtype)[None]
        positions=torch.arange(len(ids),device=embeds.device)
        dense=runner.model.model.language_model(inputs_embeds=embeds,position_ids=positions[None],cache_position=positions,
            attention_mask=torch.ones(1,len(ids),dtype=torch.long,device=embeds.device),use_cache=True)
        dense_logits=runner.model.lm_head(dense.last_hidden_state[:,-1:])
        for policy,k in [('global',0),('global',32),('per_image',32),('per_image',None)]:
            print(json.dumps(dict(stage='functional',case=name,k=k,policy=policy,tokens=len(ids))),flush=True)
            io=IOCounter();logits,cache,plan,stats=MultiSelectivePrefill(runner,k_recompute=k,selection_policy=policy).run(ids,bindings,io)
            for layer in stats['layers']:
                assert layer['active_K_installed'] and layer['active_V_installed']
                assert all(layer['reused_unchanged_per_image']) and layer['future_probability_max']==0
                assert layer['Q_shape'][2]==len(plan.active_positions)
            comparison={'dense_max_abs':float((logits.float()-dense_logits.float()).abs().max()),
                        'equality_required':k is None,'rtol':1e-4,'atol':1e-4}
            if k is None:
                torch.testing.assert_close(logits,dense_logits,rtol=1e-4,atol=1e-4)
                for (ak,av),ref in zip(cache,dense.past_key_values.layers):
                    torch.testing.assert_close(ak,ref.keys,rtol=1e-4,atol=1e-4)
                    torch.testing.assert_close(av,ref.values,rtol=1e-4,atol=1e-4)
            record=dict(case=name,k=k,status='PASS',input_ids=ids.tolist(),io=io.summary(),comparison=comparison,**stats)
            evidence.append(record);append(out/'gpu_functional_records.jsonl',record)
            del cache,logits
        del dense,embeds,dense_logits
    # FullLoad suffix=1 on actual image, observing the installed device K/V.
    from mmimpress.serve import ImageContext,Server
    past,enc,start,n=source
    _,_,_,newline=runner.anyres_layout(enc['image_sizes'][0],n)
    store=out/'fullload_gpu_gate_store'
    write_image_store(store,cache_layers(past),start,n,enc['input_ids'][0,:start+n].tolist(),newline,
                      extra={'image_id':names[0]},separator_sidecar=True)
    full=ImageContext(store,runner.model.device,drop_cache=False,require_v_hidden=False)
    try:
        native_install=[]
        handles=[]
        for li,layer in enumerate(runner.layers):
            def observer(module,args,kwargs,result,li=li):
                cache=kwargs['past_key_values'];k,v=cache.layers[li].keys,cache.layers[li].values
                native_install.append(dict(layer=li,K_shape=list(k.shape),V_shape=list(v.shape),
                    K_installed=torch.equal(k[:,:,start:start+n],full.cache.k[li][:,:,start:start+n]),
                    V_installed=torch.equal(v[:,:,start:start+n],full.cache.v[li][:,:,start:start+n])))
            handles.append(layer.self_attn.register_forward_hook(observer,with_kwargs=True))
        for suffix in (torch.tensor([13]),torch.tensor([13,14])):
            native_install.clear()
            result=Server(runner,max_new_tokens=3).request(full,mode='fullload',cold=False,suffix_ids=suffix,cache_id=names[0])
            assert result['io']['preads']==2*len(runner.layers)
            for li in range(len(runner.layers)):
                for kind,installed in [('k',full.cache.k[li]),('v',full.cache.v[li])]:
                    _,raw=contexts[0].reader.read_reused_layer(li,kind,0)
                    expected=raw.to(installed.device,installed.dtype).permute(1,0,2)[None]
                    assert torch.equal(installed[:,:,start:start+n],expected)
            assert native_install and all(x['K_installed'] and x['V_installed'] for x in native_install)
            evidence.append(dict(case='FullLoad',suffix_tokens=len(suffix),status='PASS',actual_GPU_KV_equal=True,
                installed_image_KV_rows=n,native_attention_cache=list(native_install),io=result['io']))
        try:Server(runner).request(full,mode='fullload',cold=False,suffix_ids=torch.tensor([13]),cache_id='wrong')
        except ValueError:pass
        else:raise AssertionError('wrong cache ID accepted')
        bad_prompt='Changed prefix USER: <image>\nDescribe this image. ASSISTANT:'
        try:Server(runner).request(full,mode='fullload',cold=False,prompt_text=bad_prompt)
        except ValueError:pass
        else:raise AssertionError('wrong prefix accepted')
    finally:
        for h in handles:h.remove()
        full.close()
    dump(out/'gpu_functional.json',dict(status='PASS',records=evidence,
        limits='k32 is partial reuse; no complete dense equivalence required. Qwen/GLM not tested.'))
    mixed_records=[];mixed_logits=[]
    activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities,record_shapes=True) as profiler:
        for parallel in (False,True):
            new=[]
            def builder(image_id):
                ctx,persist,_=provision(runner,images[1],out/f'mixed_miss_{int(parallel)}')
                new.append(ctx);return ctx
            try:
                prepared,record=prepare_mixed(names,{names[0]:contexts[0]},builder,
                    selected_counts=[32,32],parallel=parallel)
                ids,bindings=expand_multi_prompt(runner,fixtures[2][1],prepared,names)
                logits,cache,_,stats=MultiSelectivePrefill(runner,selection_policy='per_image').run(ids,bindings,IOCounter())
                mixed_logits.append(logits.detach().clone());mixed_records.append(record)
                del cache,prepared
            finally:
                for ctx in new:ctx.close()
    torch.testing.assert_close(mixed_logits[0],mixed_logits[1],rtol=1e-4,atol=1e-4)
    profiler.export_chrome_trace(str(out/'gpu_mixed_profiler.json'))
    dump(out/'gpu_mixed.json',dict(records=mixed_records,serial_parallel_logits='PASS',
        GPU_overlap='NOT VERIFIED until CUDA trace demonstrates intersection with actual pread intervals',
        preparation_includes='miss native multimodal forward, KV capture, FP16 SSD persist; hit worker actual pread'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--execute',action='store_true');ap.add_argument('--condition',choices=['gpu_resident','ssd_cold'],default='ssd_cold')
    ap.add_argument('--seed',type=int,default=1234);ap.add_argument('--max-dialogues',type=int,default=5)
    ap.add_argument('--manifest',type=Path,help='use an existing frozen subset without reselection')
    ap.add_argument('--gate-only',action='store_true',help='one frozen dialogue functional validation; no measured evaluation')
    ap.add_argument('--max-new-tokens',type=int,default=256);a=ap.parse_args()
    if not 1<=a.max_dialogues<=5 or not 32<=a.max_new_tokens<=256:raise ValueError('bounded smoke only: <=5 dialogues, cap32..256')
    a.out.mkdir(parents=True,exist_ok=False)
    processor=AutoProcessor.from_pretrained(SNAPSHOT,local_files_only=True,use_fast=False)
    if a.manifest is not None:
        manifest=json.loads(a.manifest.read_text())
        if sha(BENCHMARK)!=manifest['benchmark_sha256'] or a.seed!=manifest['seed']:raise ValueError('frozen benchmark/seed mismatch')
        selected=manifest['dialogues'];decisions=manifest['decisions']
        if len(selected)!=5:raise ValueError('expected original five-dialogue manifest')
        byid={str(x['id']):x for x in json.loads(BENCHMARK.read_text())}
        for dialog in selected:
            item=byid[dialog['dialog_id']]
            for i,turn in enumerate(dialog['turns']):
                if turn['question']!=item['conversations'][2*i]['value']:raise ValueError('frozen question mismatch')
            for image in dialog['images']:
                if sha(image['path'])!=image['sha256']:raise ValueError('frozen image hash mismatch')
        with (a.out/'subset_manifest.json').open('xb') as f:f.write(a.manifest.read_bytes())
    else:
        selected,decisions=prepare(processor,a.max_dialogues,a.seed,a.max_new_tokens)
        dump(a.out/'subset_manifest.json',dict(benchmark=str(BENCHMARK),benchmark_sha256=sha(BENCHMARK),count=len(selected),
        dialogues=selected,decisions=decisions,seed=a.seed,selection='sha256(seed:source_id), exactly two images introduced by turn3, capacity bound',
        resize='all images 336x336 Lanczos as official MMDU LLaVA-NeXT generator; no crop or relocation',
        remote_dataset='laolao77/MMDU',remote_file_commit='269d0d6a20edd1774a05388ca7cc25df9631d63c',
            remote_byte_equivalence='NOT VERIFIED; local hash is authoritative for this run',training_45k_used=False))
    config=dict(model='llava-hf/llava-v1.6-vicuna-7b-hf',revision=REV,precision='NF4 double quant/BF16 compute/FP16 SSD',
        backend='eager',engine='Transformers 4.57.6',seed=a.seed,methods=METHODS,k=32,k_policy='global and per_image diagnostics; no official-rule claim',
        turns=3,generation_cap=a.max_new_tokens,decoding='greedy EOS',history='method-local generated, no gold substitution',
        condition=a.condition,cache_creation_in_TTFT=False,TTFT='outer request start before prompt/tokenization/preprocessing -> first argmax host materialization + CUDA sync',
        warmup='per-method separate first-turn fixture, never included in raw measurement',quality_judge='NOT VERIFIED (not executed)',
        paper_generation_cap='NOT SPECIFIED in MPIC manuscript',official_MMDU_generator_cap=2048,smoke_cap_is_adaptation=True,
        source_sha256={name:sha(ROOT/name) for name in ['mmimpress/serve.py','mmimpress/mpic.py','mmimpress/mpic_multi.py','mmimpress/mmdu_mpic.py','scripts/125_eval_mpic_mmdu_smoke.py']})
    dump(a.out/'config.json',config)
    nvidia=subprocess.run(['nvidia-smi'],capture_output=True,text=True)
    dump(a.out/'environment.json',dict(python=sys.version,platform=platform.platform(),cuda_available=torch.cuda.is_available(),
        cuda_devices=torch.cuda.device_count(),nvidia_smi=nvidia.stdout+nvidia.stderr,git_HEAD=None))
    if not a.execute or not torch.cuda.is_available():
        dump(a.out/'status.json',dict(status='NOT VERIFIED',reason='preparation only' if not a.execute else 'NVIDIA driver inaccessible; torch.cuda.is_available=False',
            prepared_dialogues=len(selected),prepared_turns=len(selected)*3,completed_dialogues=0,completed_turns=0,
            pretrained_weights_loaded=False,GPU_functional='NOT VERIFIED',MMDU_TTFT='NOT VERIFIED',quality='NOT VERIFIED'))
        with (a.out/'ttft_io.csv').open('x') as f:f.write('dialog_id,turn,method,condition,ttft_ms,ssd_bytes,preads,cap_reached\n')
        (a.out/'answers.jsonl').open('x').close()  # zero generated answers, not fabricated fixtures
        print(json.dumps(dict(prepared=len(selected),completed=0,GPU='NOT VERIFIED')))
        if a.execute:raise SystemExit(3)
        return
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    # Loader refuses offload, changed precision, and foreign GPU processes.
    processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    if any(int(s.strip())!=os.getpid() for s in processes.stdout.splitlines() if s.strip()):raise RuntimeError('GPU is in use by another process')
    runner=LlavaRunner(model_id=str(SNAPSHOT)).load();runner.model_id=config['model']
    rows=[];completed_dialogues=0
    try:
        for dialog in selected:
            directory=a.out/f'dialog_{dialog["dialog_id"]}';directory.mkdir()
            contexts=[];source=None;active_contexts=[];previous=None
            for i,image in enumerate(dialog['images']):
                assert sha(image['path'])==image['sha256']
                ctx,persist,src=provision(runner,image,directory/f'image_{i}',retain_source=i==0)
                dump(directory/f'image_{i}_persistence.json',persist);contexts.append(ctx)
                if i==0:source=src
            try:
                if completed_dialogues==0:gpu_gate(runner,contexts,dialog['images'],directory,source,dialog)
                del source
                if a.gate_only:
                    dump(a.out/'status.json',dict(status='PASS (GPU functional only)',functional_dialogue_id=dialog['dialog_id'],
                        completed_dialogues=0,completed_turns=0,method_requests=0,fallback=0))
                    return
                active_contexts=contexts
                if a.condition=='gpu_resident':
                    active_contexts=[SimpleNamespace(meta=c.meta,dir=c.dir,reader=GPUResidentReader(c,runner.model.device,runner.model.dtype)) for c in contexts]
                    dump(directory/'resident_setup.json',[c.reader.setup_io for c in active_contexts])
                histories={method:{} for method in METHODS};previous=None
                # Warm every method using this same first-turn input; generations
                # and source caches are discarded, not put into histories.
                for method in METHODS:
                    run_one(runner,dialog,1,method,{},active_contexts,None,a.max_new_tokens,a.condition)
                dump(directory/'warmup.json',dict(methods=METHODS,fixture='selected first turn, excluded from measured raw',histories_reused=False))
                for turn in (1,2,3):
                    for method in METHODS:
                        result,state=run_one(runner,dialog,turn,method,histories[method],active_contexts,previous if method=='prefix_caching' else None,a.max_new_tokens,a.condition)
                        if method=='prefix_caching':previous=state
                        histories[method][turn]=result['prediction']
                        row=dict(dialog_id=dialog['dialog_id'],turn=turn,method=method,condition=a.condition,**result)
                        append(a.out/'answers.jsonl',row);rows.append(row)
                        print(json.dumps(dict(stage='measured',dialogue=dialog['dialog_id'],turn=turn,method=method,
                            ttft_ms=result['ttft_ms'],output_tokens=result['generated_token_count'],
                            peak_gpu_bytes=result['peak_gpu_allocated_bytes'])),flush=True)
                completed_dialogues+=1
            finally:
                for ctx in contexts:ctx.close()
                del contexts,active_contexts,previous
                torch.cuda.empty_cache()
        with (a.out/'ttft_io.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['dialog_id','turn','method','condition','ttft_ms','ssd_bytes','preads','cap_reached']);writer.writeheader()
            for row in rows:writer.writerow({k:row[k] for k in writer.fieldnames})
        dump(a.out/'status.json',dict(status='PASS (bounded execution only)',completed_dialogues=completed_dialogues,
            completed_turns=completed_dialogues*3,method_requests=len(rows),quality='NOT VERIFIED: no judge executed'))
    except BaseException as exc:
        dump(a.out/'failure.json',dict(status='FAIL',error=repr(exc),traceback=traceback.format_exc(),
            completed_dialogues=completed_dialogues,completed_method_requests=len(rows),fallback=False));raise

@torch.no_grad()
def run_one(runner,dialog,turn,method,generated,contexts,previous,cap,condition):
    count=dialog['turns'][turn-1]['active_images'];contexts=contexts[:count]
    cold=condition=='ssd_cold' and method!='prefix_caching'
    if cold:
        for ctx in contexts:ctx.reader.drop_all()
    _sync(torch.device(runner.model.device));torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    prompt=prompt_from_generated(dialog,turn,generated);names=[x['image_id'] for x in dialog['images'][:count]]
    ids,bindings=expand_multi_prompt(runner,prompt,contexts,names)
    if len(ids)+cap>4096:raise RuntimeError('actual generated history exceeds fixed native context bound; no truncation')
    io=IOCounter();state=None
    if method in ('mpic_global32','mpic_per_image32'):
        policy='global' if method=='mpic_global32' else 'per_image'
        logits,layers,_,stats=MultiSelectivePrefill(runner,selection_policy=policy).run(ids,bindings,io)
    elif method=='full_reuse':logits,layers,stats=full_reuse_prefill(runner,ids,bindings,io)
    elif method=='prefix_caching':
        labels=logical_labels(ids,bindings);matched=matching_prefix_length(labels,previous)
        misses=[i for i,b in enumerate(bindings) if b.start+b.count>matched]
        enc={'input_ids':ids[None]}
        if misses:
            pics=[image_for(dialog['images'][i]['path']) for i in misses]
            enc.update(runner.processor.image_processor(images=pics,return_tensors='pt'))
        embeds,vision_images=visual_embeddings(runner,enc,bindings,previous,matched)
        logits,layers,state,matched=prefix_prefill(runner,ids,embeds,labels,previous)
        stats=dict(prefix_reused_rows=matched,decoder_prefill_pass_count=1,
                   vision_images_computed=vision_images,prefix_vision_policy='only images not fully contained in exact matching prefix encoded')
    else:raise ValueError('unknown comparator')
    result=decode_linked(runner,logits,layers,max_new_tokens=cap)
    result.update(prompt=prompt,input_ids=ids.tolist(),image_ids=names,history=dict(generated),stats=stats,
        ttft_ms=(result['first_token_at_s']-started)*1000,request_started_at_s=started,
        ssd_bytes=io.summary()['bytes'],preads=io.summary()['preads'],io=io.summary(),
        fallback=False,quality_score=None,quality_status='NOT VERIFIED',page_cache_conditioning_excluded=True)
    total_image=sum(b.count for b in bindings)
    selected=sum(len(x) for x in stats.get('selected_image_local_rows',[])) if method.startswith('mpic_') else (
        sum(max(0,b.start+b.count-max(b.start,stats['prefix_reused_rows'])) for b in bindings) if method=='prefix_caching' else 0)
    result.update(input_token_count=len(ids),image_token_count=total_image,images=count,recomputed_image_tokens=selected,
        reused_image_tokens=total_image-selected,generated_token_count=len(result['generated_token_ids']),
        termination='EOS' if result['generated_token_ids'][-1]==runner.processor.tokenizer.eos_token_id else 'length_cap',
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        retry_count=0,status='PASS',cache_residency=condition)
    return result,state
if __name__=='__main__':main()

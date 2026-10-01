#!/usr/bin/env python3
"""Frozen storage-policy ablation: correctness before any model's pilot."""
from __future__ import annotations
import argparse, contextlib, copy, gc, hashlib, importlib.util, json, os, random
import shutil, subprocess, sys, time, traceback, uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from mmimpress.prefix25 import sha, inventory, seal_integrity, verify_integrity, process_io, ReadTrace
from mmimpress.dataset import exact_score
ARMS=['recompute','fullload','A','B']; SEED=1234

def load(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def safe(v):
    if torch.is_tensor(v):return v.detach().cpu().tolist()
    if isinstance(v,Path):return str(v)
    if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [safe(x) for x in v]
    if isinstance(v,np.generic):return v.item()
    return v

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:json.dump(safe(obj),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def append(path,obj):
    with Path(path).open('a') as f:f.write(json.dumps(safe(obj),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

def exclusive():
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    other=[x.strip() for x in p.stdout.splitlines() if x.strip() and int(x.strip())!=os.getpid()]
    if other:raise RuntimeError('external GPU work; no concurrent latency allowed: '+str(other))

def capacity():
    free=shutil.disk_usage(ROOT).free
    if free<46*2**30:raise RuntimeError(f'capacity reserve: {free/2**30:.2f} GiB free; need 30 GiB + 16 GiB staging')

def freeze(run):
    run=Path(run); res=ROOT/'results'/run.name
    manifests={}
    for name,old in [('gqa','gqa_manifest'),('mt','mt_manifest')]:
        path=ROOT/'runs/qwen25_port_20260928T054537Z'/old/'manifest.json'
        source=json.loads(path.read_text()); rows=source['images']
        assert len(rows)==40 and len({r['image_id'] for r in rows})==40
        for row in rows:assert Path(row['image_path']).exists() and sha(row['image_path'])==row['image_sha256']
        manifests[name]={'source_path':str(path),'source_sha256':sha(path),'images':rows,'history':'method_own_generated' if name=='mt' else 'none'}
    index=json.loads((ROOT/'data/index.json').read_text())
    assert [[str(q['question_id']) for q in x['questions'][4:10]] for x in index]==[[str(t['question_id']) for t in x['turns']] for x in manifests['gqa']['images']]
    manifests['smoke']=copy.deepcopy(manifests['gqa']);manifests['smoke']['images']=manifests['smoke']['images'][:4]
    for row in manifests['smoke']['images']:row['turns']=row['turns'][:3]
    for name,value in manifests.items():write(run/f'{name}_manifest.json',value)
    frozen=ROOT/'runs/qwen25_correctness_v2_20260928T081111Z/validation_manifest.json'
    shutil.copy2(frozen,run/'correctness_manifest.json')
    revision=(Path.home()/'.cache/huggingface/hub/models--llava-hf--llava-v1.6-vicuna-7b-hf/refs/main').read_text().strip()
    assert revision=='c916e6cdcd760b4cecd1dd4907f84ac649f93b23'
    config={'seed':1234,'greedy':True,'max_new_tokens':16,'batch_size':1,'chunk_size':64,'ratio':.25,'arms':ARMS,
      'llava':{'revision':revision,'quantization':'NF4 double BF16 compute','attention':'eager','store_dtype':'float16','processor':'existing AnyRes'},
      'qwen':{'revision':'cc594898137f460bfe9f0759e9844b3ce807cfb5','quantization':'NF4 double BF16 compute','attention':'sdpa','store_dtype':'bfloat16','min_pixels':200704,'max_pixels':802816},
      'bootstrap':{'unit':'image','resamples':10000,'seed':1234},'workload_files':{n:sha(run/f'{n}_manifest.json') for n in manifests},
      'correctness_file_sha256':sha(run/'correctness_manifest.json'),'prior_v1_status':'FAIL unchanged; this ablation cannot relabel it',
      'timing':'outer before image decode/input preparation; synchronized first token; E2E through generation plus mandatory capture; one-time persistence and activation separately added'}
    write(run/'config.json',config);write(res/'config.json',config)
    contract='''# Prefix25 persistence contract

Frozen before GPU execution. A = original importance-repacked FullStore KV25;
B = same full CPU materialization, same full stable permutation and repack,
then serialize content rows [0, ceil(N/4)) only. C=64, short final chunk,
no real extra rows, no padding in B. Structural/system KV stays separate.
Original N, positions, geometry and full permutation stay distinct from stored k.
B rejects chunk budget and any ratio >.25. No pruning, merging or new scoring.
LLaVA full-size physical buffer and BF16 restoration from FP16 stay unchanged;
Qwen native BF16 logical compact assembly and original MRoPE stay unchanged.

CPU: N=1,63,64,65,127,128,129,255,256,257,349,2200; zero rejected;
independent stable-rank/gather; all layer/head KV bits, structural, short EOF,
invalid identity/mapping/checksum/truncation, double-budget and oversized requests.
Independent real os.pread returned byte/call traces must equal production counters.
GPU: five original LLaVA pairs and ten original Qwen v2 pairs, reused validation
samples (not unseen holdout). Same full-image capture for independent A/B stores.
Independent score rank and canonical-KV gather; content k, structural, original
positions and actual masks from prefill through last decode; first/generated IDs
and prediction exact. LLaVA every generated-step logits atol=1e-4, rtol=1e-4;
Qwen matched first logits bitwise exact plus all generated IDs exact, stock MRoPE
and masks, FP32 dense/compact GQA oracle atol=rtol=1e-5. No relaxation after results.
Release source/capture/reference KV, clear serving contexts, activate B alone and
require fresh independent B payload reads, zero vision/query-scoring, deterministic
repeat, history invariance and interleaved image/method isolation. Any required
FAIL or unexecuted gate blocks that model's performance experiments independently.

Each performance arm executes its own full-image T1 and own physical persistence.
No shared timing/capture/store. ReComp none; FullLoad canonical; A full; B prefix.
Full integrity hashes and file/directory/parent fsync inside both A/B total
persistence. Existing helper components may overlap; total uses a wall boundary.
/proc/self/io wchar/syscw deltas measure OS successful write returned bytes/calls,
not NAND traffic. Integrity envelope, system, metadata and padding counted.
LLaVA saliency hooks run in forward; Qwen saliency computation occurs in VisionScoreCapture.__exit__ after normal
request generation; outer request E2E includes this and required capture cloning
exactly once, before persistence. normal_request_e2e_ms records the inner boundary.
Full CPU copy/repack costs are retained. Serialization timing separate from write.
Service session = T1 E2E + persistence + one activation + T2 E2E + T3 E2E.
GQA sum is six independent requests + one-time persistence/activation, not MT.
Page-cache DONTNEED outside request timers for all stored arms; no NAND cold claim.
Activation integrity reads are separate and included once in session. Hits do not
decode image files. Independent tracing overhead is present symmetrically on hits.
Warmup excluded; methods rotate by image and turn. Never measure with another GPU
compute process. Seed=1234, greedy max16, unchanged model revisions/backend/processor.
Smoke 4x3x4; GQA frozen40x6x4; MT frozen40x3x4, own generated history per arm.
No 4061-dialogue full run. Image-cluster paired bootstrap 10000 seed1234 95% CI.
MB=1e6 bytes, disk safety>=30 GiB. Keep raw/failures/inventories; delete only this
run's stores after all image measurements and hash/mapping receipts committed.
Cleaned payloads require regeneration. No old artifacts changed and no push.
'''
    (run/'CONTRACT.md').write_text(contract);(res/'CONTRACT.md').write_text(contract)
    env={'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout,
         'packages':subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True).stdout}
    write(run/'environment.json',env)
    paths=[p for d in ['mmimpress','scripts','tests'] for p in (ROOT/d).rglob('*.py') if '__pycache__' not in p.parts]
    paths += [run/'CONTRACT.md',run/'config.json',run/'correctness_manifest.json']+[run/f'{n}_manifest.json' for n in manifests]
    write(run/'gpu_freeze.json',{'files':{str(p.relative_to(ROOT)):sha(p) for p in paths}})
    print('FROZEN',run,flush=True)

@contextlib.contextmanager
def qwen_attention_guard(runner, meta, question, history):
    """Observe every compact prefill/decode attention call and original MRoPE."""
    import torch.nn.functional as F
    full,suffix,positions,deltas=runner._cache_hit_ids(meta,question,history)
    k=(meta['visual_count']+3)//4;prefix=k+meta['structural_count']
    original=F.scaled_dot_product_attention;seen=[];steps=[]
    def pre(module,args,kw):
        qlen=kw['input_ids'].shape[1]
        step=len(steps);cache=kw.get('past_key_values')
        before=prefix if step==0 else prefix+suffix.shape[1]+step-1
        assert int(cache.get_seq_length())==before
        cp=kw['cache_position'].detach().cpu()
        assert torch.equal(cp,torch.arange(before,before+qlen))
        want=positions[:,:,meta['prefix_len']:] if step==0 else torch.full((3,1,1),int(positions.max())+step)
        assert torch.equal(kw['position_ids'].detach().cpu(),want.cpu())
        assert kw['attention_mask'].shape[-1]==before+qlen and bool(kw['attention_mask'].all())
        steps.append({'qlen':qlen,'cache_before':before,'content_visible':k})
    def attention(q,key,v,attn_mask=None,dropout_p=0.,is_causal=False,**kw):
        qlen,klen=q.shape[-2],key.shape[-2]
        want=torch.arange(klen,device=q.device)[None,:] <= torch.arange(klen-qlen,klen,device=q.device)[:,None]
        if attn_mask is None:
            actual=torch.ones((qlen,klen),dtype=torch.bool,device=q.device) if not is_causal else torch.ones((qlen,klen),dtype=torch.bool,device=q.device).tril()
        else:actual=attn_mask[0,0] if attn_mask.dtype==torch.bool else attn_mask[0,0]==0
        assert torch.equal(actual,want)
        assert bool(actual[:,:prefix].all()) and klen>=prefix+qlen
        seen.append((int(qlen),int(klen)))
        return original(q,key,v,attn_mask=attn_mask,dropout_p=dropout_p,is_causal=is_causal,**kw)
    h=runner.model.register_forward_pre_hook(pre,with_kwargs=True)
    F.scaled_dot_product_attention=attention
    evidence={'steps':steps}
    try:yield evidence
    finally:
        F.scaled_dot_product_attention=original;h.remove()
        evidence['attention_calls']=len(seen);evidence['layers']=meta['num_layers']
        assert len(seen)==len(steps)*meta['num_layers']

class Experiment:
    def __init__(self,run,model):
        self.run=Path(run);self.model=model;self.base=self.run/model;self.base.mkdir(exist_ok=True)
        self.e=load('llava_pilot_existing','89_eval_llava_kv25.py')
        if model=='llava':
            from mmimpress.model import LlavaRunner
            from mmimpress.serve import Server
            self.runner=LlavaRunner().load();self.server=Server(self.runner,max_new_tokens=16)
            self.ref=load('llava_reference_existing','90_validate_llava_kv25.py')
            # Existing reference trace assumes full payload EOF; this ablation changes only its independent read plan.
            self.ref._check_reads=self.check_llava_reads
        else:
            from mmimpress.qwen25.runner import Qwen25Runner
            self.runner=Qwen25Runner().load();self.server=None
            self.v2=load('qwen_v2_reference_existing','86_validate_qwen25_v2_rerun.py')
            self.qref=load('qwen_kv25_reference_existing','91_validate_qwen25_kv25.py');self.qref.v2=self.v2
        self.first_isolation=None

    def check_llava_reads(self,ctx,trace,chunks):
        meta=ctx.meta; rowbytes=meta['num_heads']*meta['head_dim']*2
        rows=min(chunks*64,meta.get('payload_rows',meta['v_token_num']))
        want={str(ctx.dir/f'layer_{li:02d}/{kind}.bin'):(0,rows*rowbytes) for li in range(meta['num_layers']) for kind in ['k','v']}
        want[str(ctx.dir/'sep_kv.bin')]=(0,(ctx.dir/'sep_kv.bin').stat().st_size)
        actual={x['path']:(x['offset'],x['returned']) for x in trace.calls}
        assert actual==want and len(trace.calls)==len(want)
        assert all(x['returned']==x['requested'] for x in trace.calls)
        return {'calls':trace.calls,'actual_returned_bytes':sum(x['returned'] for x in trace.calls),'actual_pread_calls':len(trace.calls),'read_ranges_exact':True}

    def pixel(self,image,turn,history,arm,phase):
        from mmimpress.piggyback import VisionForwardCapture,DecoderVisualHiddenCapture
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
        with Image.open(image['image_path']) as src:pil=src.convert('RGB')
        decode_ms=(time.perf_counter()-t0)*1000
        if self.model=='qwen':
            value=self.runner.run_pixels(pil,turn['question'],history,capture=('with_score' if arm in ['A','B'] else 'kv_only' if arm=='fullload' else False),image_sha256=image['image_sha256'])
            value['ttft_ms']+=decode_ms
            value['normal_request_e2e_ms']=value['request_e2e_ms']+decode_ms
            value['request_e2e_ms']=(time.perf_counter()-t0)*1000
            value['post_generation_capture_ms']=value['request_e2e_ms']-value['normal_request_e2e_ms']
            value['score_accounting']='post-generation VisionScoreCapture.__exit__; included once in outer request E2E'
            diag=value.pop('capture',None)
        else:
            cap=VisionForwardCapture(self.runner,capture_saliency=arm in ['A','B'])
            prompt=self.prompt(turn['question'],history,phase)
            with cap:
                enc=self.runner.encode_prompt(pil,prompt)
                vs,vn=self.runner.visual_span(enc['input_ids'])
                hidden=DecoderVisualHiddenCapture(self.runner,vs,vn) if arm=='fullload' else None
                with hidden if hidden is not None else contextlib.nullcontext():
                    value=self.server.recompute(self.runner.to_device(enc),return_past_key_values=arm!='recompute')
            value['ttft_ms']=(value['first_token_at_s']-t0)*1000
            value['request_e2e_ms']=(time.perf_counter()-t0)*1000
            value['prediction']=value['answer'];value['vision_calls']=cap.call_count
            value['online_query_score_calls']=0
            value['capture_stats']=cap.stats()
            diag={'cache':value.pop('captured_past_key_values',None),'enc':enc,'vision':cap,'hidden':hidden}
        value['image_file_decode_ms']=decode_ms
        value['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated();value['peak_gpu_reserved_bytes']=torch.cuda.max_memory_reserved()
        return value,diag

    def prompt(self,q,history,phase):
        if phase!='mt':return self.runner.prompt(q)
        lines=[]
        for i,(pq,pa) in enumerate(history,1):lines.extend([f'Q{i}: {pq}',f'A{i}: {pa}'])
        if lines:lines.append('')
        lines.extend([f'Current question Q{len(history)+1}: {q}',self.e.SHORT_ANSWER+' ASSISTANT:'])
        return 'USER: <image>\n'+'\n'.join(lines)

    def persist(self,capture,path,image,arm):
        capacity();before=process_io();started=time.perf_counter()
        if self.model=='qwen':
            receipt=self.runner.persist(capture,path,layout='canonical' if arm=='fullload' else 'repacked',storage_policy='prefix25' if arm=='B' else 'full')
            meta=receipt['metadata'];timing=receipt['timing_ms']
        else:
            from mmimpress.piggyback import persist_captured_visual_prefix,persist_captured_raster_prefix
            enc=capture['enc'];common=dict(image_id=image['image_id'],model_id=self.runner.model_id,image_input_sha256=image['image_sha256'],extra_metadata={'checkpoint_revision':self.runner.cfg._commit_hash,'processor_revision':self.runner.cfg._commit_hash,'ablation':'prefix25'})
            if arm=='fullload':
                receipt=persist_captured_raster_prefix(self.runner,capture['cache'],enc['input_ids'],enc['image_sizes'][0],capture['hidden'].result_cpu(),path,hidden_capture_stats=capture['hidden'],**common)
            else:
                receipt=persist_captured_visual_prefix(self.runner,capture['cache'],enc['input_ids'],enc['image_sizes'][0],capture['vision'].result_cpu(),path,capture_stats=capture['vision'],storage_policy='prefix25' if arm=='B' else 'full',**common)
            meta=receipt['meta'];timing=receipt['timing_ms']
        hash_started=time.perf_counter();files=seal_integrity(path);hash_ms=(time.perf_counter()-hash_started)*1000
        total_ms=(time.perf_counter()-started)*1000;after=process_io()
        n=meta.get('visual_count',meta.get('n_spatial'));s=meta.get('structural_count',len(meta.get('newline_idx',[]))+meta.get('v_token_start',0))
        k=(n+3)//4 if arm=='B' else n
        width=2*meta['num_layers']*meta.get('num_kv_heads',meta.get('num_heads'))*meta['head_dim']*2
        return {'physical_write_id':str(uuid.uuid4()),'store_path':str(path),'model':self.model,'arm':arm,'image_id':image['image_id'],
          'metadata':meta,'files':files,'timing_ms':timing,'hash_and_seal_ms':hash_ms,'persistence_ms':total_ms,
          'os_write_bytes':after['wchar']-before['wchar'],'os_write_calls':after['syscw']-before['syscw'],
          'original_content_bytes':n*width,'stored_valid_content_bytes':k*width,'structural_system_bytes':s*width,
          'padding_bytes':meta.get('padding_rows',0)*width,'file_logical_bytes':sum(x['size'] for x in files.values()),
          'allocated_bytes':sum(x['allocated_bytes'] for x in files.values()),
          'metadata_and_auxiliary_bytes':sum(x['size'] for p,x in files.items() if not p.endswith('.bin') and p!='sys_kv.pt'),
          'N':n,'stored_k':k,'content_retention':k/n,'durability':'full checksums and file/directory/parent fsync',
          'capture_cost_in_request':True,'historical_persistence_reused':False}

    def activate(self,path,image,arm):
        start=time.perf_counter()
        # Both arms verify the same envelope; Qwen also retains its native payload validator.
        check={}
        if self.model=='qwen':
            check=verify_integrity(path)
            ctx=self.runner._activate(path,image['image_sha256']);native=self.runner.activation_records[str(path.resolve())]
        else:
            from mmimpress.serve import ImageContext
            ctx=ImageContext(path,self.runner.model.device,require_v_hidden=False)
            if arm in ['A','B']:ctx.validate_visual_kv_layout()
            check=ctx.integrity_activation;native={}
        torch.cuda.synchronize()
        return ctx,{'activation_ms':(time.perf_counter()-start)*1000,**check,'native':native}

    def close_contexts(self):
        if self.model=='qwen':
            for obj in self.runner._stores.values():obj.close()
            self.runner._stores.clear();self.runner.activation_records.clear()
        gc.collect();torch.cuda.empty_cache()

    def hit(self,ctx,path,image,turn,history,arm,phase,logits=False):
        conditioning=time.perf_counter()
        if self.model=='qwen':cond=ctx.drop_payload_cache()
        else:ctx.reader.drop_all();cond={'policy':'DONTNEED'}
        cond['ms']=(time.perf_counter()-conditioning)*1000
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        captured=[];handle=None
        if logits and self.model=='llava':handle=self.runner.model.register_forward_hook(lambda m,a,o:captured.append(o.logits[0,-1].detach().float().cpu()) if hasattr(o,'logits') else None)
        vision_hits=[]
        def forbidden_vision(*args, **kwargs):
            vision_hits.append(1)
            raise AssertionError('cache hit executed vision forward')
        vision_handle=self.runner.model.visual.register_forward_pre_hook(forbidden_vision) if self.model=='qwen' else None
        with ReadTrace(path) as trace:
            if self.model=='qwen':
                with qwen_attention_guard(self.runner,ctx.meta,turn['question'],history) if logits else contextlib.nullcontext() as attention_evidence:
                    value=self.runner.run_cache(path,turn['question'],history,budget_ratio=1. if arm=='fullload' else .25,budget_unit='chunk' if arm=='fullload' else 'visual_kv',image_sha256=image['image_sha256'],return_logits=logits)
                if logits:value['attention_evidence']=attention_evidence
            else:
                from mmimpress.serve import suffix_ids_from_prompt
                suffix=suffix_ids_from_prompt(self.runner,self.prompt(turn['question'],history,phase)).to(self.runner.model.device)
                with self.e.QA._NoVisionForward(self.runner) as guard:
                    if arm=='fullload':value=self.server.request(ctx,mode='fullload',cold=False,suffix_ids=suffix)
                    else:value=self.server.request_cvpr25(ctx,static=None,budget=.25,budget_unit='visual_kv',mode='prefix',sep_policy='sidecar',cold=False,suffix_ids=suffix,expected_prefix_layout='visionzip_image_only')
                value['ttft_ms']=(value['first_token_at_s']-start)*1000
                value['request_e2e_ms']=(time.perf_counter()-start)*1000
                value['prediction']=value['answer'];value['vision_calls']=guard.calls;value['online_query_score_calls']=value.get('query_score_calls',0)
        if vision_handle:vision_handle.remove()
        value['observed_hit_vision_calls']=len(vision_hits)
        if handle:handle.remove();value['all_logits']=captured
        count=value['read_io'] if self.model=='qwen' else value['io']
        assert trace.summary()['bytes']==count['bytes'] and trace.summary()['preads']==count['preads']
        assert value['vision_calls']==0 and value['online_query_score_calls']==0
        value['actual_io']=trace.summary();value['conditioning']=cond
        value['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated();value['peak_gpu_reserved_bytes']=torch.cuda.max_memory_reserved()
        return value

    def identity(self,a,b):
        assert a['first_token_id']==b['first_token_id']
        assert a['generated_token_ids']==b['generated_token_ids']
        assert a['prediction']==b['prediction']
        if self.model=='qwen':assert torch.equal(a['first_logits'],b['first_logits'])
        else:
            assert len(a['all_logits'])==len(b['all_logits'])
            assert all(torch.allclose(x,y,atol=1e-4,rtol=1e-4) for x,y in zip(a['all_logits'],b['all_logits']))

    def output_proof(self,value):
        logits=value.get('all_logits',[value.get('first_logits')])
        return {'first_token_id':value['first_token_id'],'generated_token_ids':value['generated_token_ids'],
                'prediction':value['prediction'],
                'logits_sha256':[hashlib.sha256(x.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest() for x in logits if x is not None],
                'attention_steps':value.get('attention_evidence'),
                'observed_vision_calls':value.get('observed_hit_vision_calls',value.get('vision_calls'))}

    def llava_case(self,sample,path):
        from mmimpress.model import cache_layers
        turn={'question':sample['question']};image=sample
        val,cap=self.pixel(image,turn,[],'A','gqa');enc=cap['enc'];layers=cache_layers(cap['cache'])
        vs,vn=self.runner.visual_span(enc['input_ids']);_,_,_,sep=self.runner.anyres_layout(enc['image_sizes'][0],vn)
        n=vn-len(sep);k=(n+3)//4
        rank=self.ref._score_rank(self.runner,cap['vision'].result_cpu(),enc['image_sizes'][0],vn,sep)
        recs={a:self.persist(cap,path/a,image,a) for a in ['A','B']}
        assert recs['A']['metadata']['order']==recs['B']['metadata']['order']==rank+sep
        outputs={};evidence={}
        for arm in ['A','B']:
            ctx,activation=self.activate(path/arm,image,arm)
            reference=self.ref._memory_reference(self.server,ctx,turn['question'],layers,rank,k)
            prod=self.ref._run_with_trace(self.server,ctx,turn['question'],layers,rank,k,budget=.25,budget_unit='visual_kv')
            actual=dict(prod['result'],logits=prod['logits'])
            comparison=self.ref._compare_outputs(reference,actual)
            outputs[arm]=dict(prediction=actual['answer'],all_logits=actual['logits'],**{key:actual[key] for key in ['first_token_id','generated_token_ids']})
            evidence[arm]={'comparison':comparison,'masks':prod['attention_masks'],'cache':prod['cache'],'positions':prod['position_checks'],'reads':prod['reads']}
            ctx.close();del ctx,prod,actual,reference
        self.identity(outputs['A'],outputs['B'])
        # Source tensors and every source-derived reference are explicitly released.
        del cap,layers,enc,val;gc.collect();torch.cuda.empty_cache()
        ctx,activation=self.activate(path/'B',image,'B')
        fresh=self.hit(ctx,path/'B',image,turn,[],'B','gqa',True);self.identity(outputs['B'],fresh)
        changed=self.hit(ctx,path/'B',image,{'question':'What is visible?'},[(turn['question'],fresh['prediction'])],'B','mt',True)
        assert fresh['selected_original_ids']==changed['selected_original_ids']==rank[:k]
        ctx.close();del ctx
        if self.first_isolation is not None:
            old_image,old_path,old_output=self.first_isolation
            oldctx,_=self.activate(old_path,old_image,'B')
            inter=self.hit(oldctx,old_path,old_image,{'question':old_image['question']},[],'B','gqa',True)
            self.identity(old_output,inter);oldctx.close()
        else:self.first_isolation=(sample,path/'B',outputs['B'])
        return {'N':n,'k':k,'ratio':k/n,'selected_original_ids':rank[:k],'all_checks':'PASS','evidence':evidence,'fresh_B_only':fresh['actual_io'],'source_released':True,'history_invariant':True,'persistence':recs,'outputs':{a:self.output_proof(v) for a,v in outputs.items()},'fresh_B_output':self.output_proof(fresh)}

    def qwen_case(self,sample,path):
        turn={'question':sample['question']};val,cap=self.pixel(sample,turn,[],'A','gqa')
        n=cap.visual_count;k=(n+3)//4;score=cap.scores.float().cpu().tolist();rank=sorted(range(n),key=lambda i:(-score[i],i))
        reference,logical=self.qref.memory_prefix(cap,rank[:k]);recs={a:self.persist(cap,path/a,sample,a) for a in ['A','B']}
        outputs={};evidence={}
        for arm in ['A','B']:
            ctx,act=self.activate(path/arm,sample,arm);loaded=ctx.load_prefix(.25,budget_unit='visual_kv')
            assert self.v2.v1._bitwise_layers(reference,loaded.layers)['equal']
            assert list(loaded.logical_indices)==logical and loaded.kept_visual_tokens==k
            meta=ctx.meta;order=meta.get('full_importance_permutation',meta['stored_to_original']);assert order==rank
            full,suffix,positions,deltas=self.runner._cache_hit_ids(meta,sample['question'],())
            stock,sd=self.v2._stock_positions(self.runner,full,torch.tensor(cap.image_grid_thw))
            assert torch.equal(stock,positions) and torch.equal(sd,deltas)
            assert torch.equal(loaded.position_ids,stock[:,:,torch.tensor(logical)])
            refpath=self.v2.trace_suffix(self.runner,reference,suffix,positions,len(cap.prefix_ids))
            ssdpath=self.v2.trace_suffix(self.runner,loaded.layers,suffix,positions,len(cap.prefix_ids))
            comparison=self.v2.path_comparison(refpath,ssdpath);assert comparison['exact_target']
            expected=self.v2._analytic_mask([True]*len(logical),suffix.shape[1])
            masks=self.v2._actual_mask_checks(ssdpath,expected);assert masks['all_layers_exact']
            neg=self.v2._negative_semantic_controls(stock,expected,logical);assert all(neg.values())
            # Dense vs compact FP32 attention, independent of production assembly.
            dense=[];dense_mask=torch.zeros((1,len(cap.prefix_ids)),dtype=torch.long);dense_mask[0,logical]=1
            for key,value in reference:
                dk=torch.zeros((1,key.shape[1],len(cap.prefix_ids),key.shape[3]),dtype=key.dtype,device=key.device);dv=torch.zeros_like(dk)
                idx=torch.tensor(logical,device=key.device);dk.index_copy_(2,idx,key);dv.index_copy_(2,idx,value);dense.append((dk,dv))
            dp=self.v2.trace_suffix(self.runner,dense,suffix,positions,len(cap.prefix_ids),dense_mask)
            tc=self.runner.model.config.text_config;dim=tc.hidden_size//tc.num_attention_heads
            o1=self.v2._fp32_oracle(refpath,reference,[True]*len(logical),tc.num_attention_heads,tc.num_key_value_heads,dim)
            o2=self.v2._fp32_oracle(dp,dense,dense_mask[0].bool().tolist(),tc.num_attention_heads,tc.num_key_value_heads,dim)
            assert torch.allclose(o1,o2,atol=1e-5,rtol=1e-5)
            outputs[arm]=self.hit(ctx,path/arm,sample,turn,[],arm,'gqa',True)
            assert self.v2.path_comparison(refpath,self.v2._result_path(outputs[arm]))['exact_target']
            evidence[arm]={'matched':comparison,'mask':masks,'negative_controls':neg,'oracle_max_abs':float((o1-o2).abs().max()),'logical_indices':logical,'position_exact':True,'reads':outputs[arm]['actual_io']}
            del loaded,refpath,ssdpath,dense,dp,o1,o2,dk,dv,key,value
        self.identity(outputs['A'],outputs['B'])
        del cap,reference,val;self.close_contexts()
        ctx,activation=self.activate(path/'B',sample,'B')
        fresh=self.hit(ctx,path/'B',sample,turn,[],'B','gqa',True);self.identity(outputs['B'],fresh)
        changed=self.hit(ctx,path/'B',sample,{'question':'What is visible?'},[(sample['question'],fresh['prediction'])],'B','mt',True)
        assert fresh['selected_visual_original']==changed['selected_visual_original']==sorted(rank[:k])
        self.close_contexts()
        if self.first_isolation is not None:
            oi,op,oo=self.first_isolation;oc,_=self.activate(op,oi,'B');inter=self.hit(oc,op,oi,{'question':oi['question']},[],'B','gqa',True);self.identity(oo,inter);self.close_contexts()
        else:self.first_isolation=(sample,path/'B',outputs['B'])
        return {'N':n,'k':k,'ratio':k/n,'selected_original_ids':rank[:k],'all_checks':'PASS','evidence':evidence,'source_released':True,'fresh_B_only':fresh['actual_io'],'history_invariant':True,'persistence':recs,'outputs':{a:self.output_proof(v) for a,v in outputs.items()},'fresh_B_output':self.output_proof(fresh)}

    def correctness(self):
        samples=json.loads((self.run/'correctness_manifest.json').read_text())['samples']
        if self.model=='llava':samples=[samples[i] for i in [0,3,4,5,6]]
        rows=[];gate={'status':'NOT RUN','samples':rows,'threshold':'LLaVA 1e-4/1e-4; Qwen exact','reused_frozen_samples':True}
        for i,sample in enumerate(samples):
            path=self.base/'correctness'/f'{i:02d}_{sample["image_id"]}';path.mkdir(parents=True,exist_ok=False)
            try:
                evidence=self.llava_case(sample,path) if self.model=='llava' else self.qwen_case(sample,path)
                rows.append({'image_id':sample['image_id'],'question_id':sample['question_id'],'status':'PASS',**evidence})
                print(self.model,'GPU',i+1,len(samples),'PASS',flush=True)
            except Exception:
                rows.append({'image_id':sample['image_id'],'status':'FAIL','error':traceback.format_exc()});gate['status']='FAIL';write(self.base/'gpu_correctness.json',gate);raise
            write(self.base/'gpu_correctness.json',gate)
        gate['status']='PASS';write(self.base/'gpu_correctness.json',gate);return gate

    def clean(self,paths,receipt_path):
        for path in paths:
            assert path.resolve().is_relative_to(self.base.resolve()) and not path.is_symlink()
            before=inventory(path);append(self.base/'cleanup.jsonl',{'stage':'before','path':path,'files':before,'receipt':receipt_path})
            shutil.rmtree(path)
            append(self.base/'cleanup.jsonl',{'stage':'after','path':path,'exists':path.exists(),'payload_retained':False})

    def pilot(self,phase):
        manifest=json.loads((self.run/f'{phase}_manifest.json').read_text());out=self.base/phase;out.mkdir(exist_ok=False)
        raw=out/'raw_requests.jsonl';persist_file=out/'persistence.jsonl'
        for ix,image in enumerate(manifest['images']):
            exclusive();capacity();contexts={};receipts={};paths={};hist={a:[] for a in ARMS};image_rows=[]
            for ti,turn in enumerate(image['turns']):
                order=ARMS[(ix+ti)%4:]+ARMS[:(ix+ti)%4]
                for arm in order:
                    exclusive();history=hist[arm] if phase=='mt' else []
                    logical=f'{self.model}/{phase}/{image["image_id"]}/{arm}/{ti+1}'
                    try:
                        if ti==0 or arm=='recompute':
                            value,capture=self.pixel(image,turn,history,arm if ti==0 else 'recompute',phase)
                            if ti==0 and arm!='recompute':
                                path=out/'stores'/image['image_id']/arm;receipt=self.persist(capture,path,image,arm)
                                del capture;gc.collect();torch.cuda.empty_cache()
                                ctx,activation=self.activate(path,image,arm);receipt['activation']=activation;receipt['phase']=phase
                                contexts[arm]=ctx;receipts[arm]=receipt;paths[arm]=path;append(persist_file,receipt)
                            else:del capture
                            value['actual_io']={'bytes':0,'preads':0,'calls':[]}
                        else:value=self.hit(contexts[arm],paths[arm],image,turn,history,arm,phase)
                        row={'logical_request_id':logical,'execution_id':str(uuid.uuid4()),'status':'OK','model':self.model,'phase':phase,'arm':arm,'image_id':image['image_id'],'dialog_id':image.get('dialog_id'),'turn':ti+1,'question_id':turn['question_id'],'question':turn['question'],'gold':turn['gold'],'history':copy.deepcopy(history),'cache_hit':ti>0 and arm!='recompute',
                             'prediction':value['prediction'],'first_token_id':value['first_token_id'],'generated_token_ids':value['generated_token_ids'],'accuracy':exact_score(value['prediction'],[turn['gold']]),'ttft_ms':value['ttft_ms'],'e2e_ms':value['request_e2e_ms'],'read_bytes':value['actual_io']['bytes'],'preads':value['actual_io']['preads'],'result':value}
                        append(raw,row);image_rows.append(row)
                        hist[arm].append((turn['question'],value['prediction']))
                    except Exception:
                        append(raw,{'logical_request_id':logical,'execution_id':str(uuid.uuid4()),'status':'FAIL','error':traceback.format_exc()});raise
            ma,mb=receipts['A']['metadata'],receipts['B']['metadata']
            pa=ma.get('full_importance_permutation',ma.get('stored_to_original',ma.get('order')))
            pb=mb['full_importance_permutation']
            assert pa==pb and receipts['A']['N']==receipts['B']['N']
            if self.model=='qwen':assert ma['saliency_sha256']==mb['saliency_sha256']
            # Request-level output identity and mapping compared before store cleanup.
            for ti in range(1,len(image['turns'])+1):
                aa=next(r for r in image_rows if r['arm']=='A' and r['turn']==ti);bb=next(r for r in image_rows if r['arm']=='B' and r['turn']==ti)
                assert aa['generated_token_ids']==bb['generated_token_ids'] and aa['prediction']==bb['prediction'], 'A/B output divergence'
            for ctx in contexts.values():ctx.close()
            contexts.clear();self.close_contexts()
            self.clean(list(paths.values()),persist_file)
            print(self.model,phase,ix+1,len(manifest['images']),'complete',flush=True)
        return {'status':'PASS','requests':len(manifest['images'])*len(manifest['images'][0]['turns'])*4}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--model',choices=['llava','qwen']);p.add_argument('--freeze',action='store_true');p.add_argument('--correctness-only',action='store_true');args=p.parse_args();run=Path(args.run)
    if args.freeze:freeze(run);return
    frozen=json.loads((run/'gpu_freeze.json').read_text())
    for rel,digest in frozen['files'].items():assert sha(ROOT/rel)==digest,rel
    cpu=(run/'cpu_regression_v2.log').read_text();assert '\nOK\n' in cpu
    exclusive();random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
    status={'IMPLEMENTATION':'PASS','CPU REGRESSION':'PASS','GPU CORRECTNESS':'NOT RUN','A/B OUTPUT IDENTITY':'NOT RUN','SMOKE':'NOT RUN','GQA PILOT':'NOT RUN','MT PILOT':'NOT RUN','PERSISTENCE/SESSION MEASUREMENT':'NOT RUN','ARTIFACT PROTECTION':'NOT RUN','READY FOR LARGE RERUN':'BLOCKED'}
    exp=None
    try:
        exp=Experiment(run,args.model);exp.correctness();status['GPU CORRECTNESS']='PASS';status['A/B OUTPUT IDENTITY']='PASS'
        if not args.correctness_only:
            sample=json.loads((run/'smoke_manifest.json').read_text())['images'][0]
            for arm in ARMS:
                val,cap=exp.pixel(sample,sample['turns'][0],[],arm,'smoke');del val,cap
            write(exp.base/'warmup.json',{'excluded':True,'normal_requests':4})
            for phase,key in [('smoke','SMOKE'),('gqa','GQA PILOT'),('mt','MT PILOT')]:
                exp.pilot(phase);status[key]='PASS';write(exp.base/'status.json',status)
            status['PERSISTENCE/SESSION MEASUREMENT']='PASS'
    except Exception:
        err=traceback.format_exc();print(err,flush=True)
        append(run/'failures.jsonl',{'model':args.model,'error':err,'utc':time.time()})
        if status['GPU CORRECTNESS']=='NOT RUN':status['GPU CORRECTNESS']='FAIL'
    finally:
        base=run/args.model;base.mkdir(exist_ok=True);write(base/'status.json',status)
        if exp:exp.close_contexts()
    print(status,flush=True)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Frozen five-arm MT-GQA main. Image-atomic adoption; no legacy registry mutation.

Lifecycle and causal prompt follow 73; verified production request helpers
follow 98. This module never imports 73/89/98 or executes ReKV/Probe3/QA paths.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
from contextlib import contextmanager
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import numpy as np
import torch
from PIL import Image
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
METHODS = ('recompute', 'fullload', 'mpic32_ssd', 'sparsevlm_ssd_kv25_allhead', 'ours_kv25')
ALLHEAD = METHODS[3]
STORE_KEYS = ('raster', 'mpic', 'image_only')
SCHEMA = 'llava-mtgqa-allhead-kv25-main-v1'
MODEL_ID = 'llava-hf/llava-v1.6-vicuna-7b-hf'
REVISION = 'c916e6cdcd760b4cecd1dd4907f84ac649f93b23'
INDEX_SHA = '2c47cfad2a7ccbb673042b400304d7f3ca03d6fbe59d04fa83db50708c924224'
WORKLOAD_SHA = '0287e0c57813800c781633b969c5cff336b3a3c1a1bdcdbb56d63f6ddab0ca62'
CONTRACT = ROOT/'docs/llava_mt_gqa_allhead_kv25_main_contract.md'
SHORT_ANSWER = 'Answer the current question with a single word or short phrase.'

def helper(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/filename)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

V = helper('97_validate_sparsevlm_ssd_kv25.py', '_main_allhead_gpu_helpers')
QA = helper('49_eval_query_aware_baseline.py', '_main_allhead_common_io')

def sha(path):
    return V.sha(path)

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def safe(x):
    if torch.is_tensor(x): return x.detach().cpu().tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    if isinstance(x, dict): return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [safe(v) for v in x]
    return x

def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)

def atomic_json(path, value, *, replace=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace: raise FileExistsError(path)
    temp = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('x') as f:
        json.dump(safe(value), f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    if replace: os.replace(temp, path)
    else:
        # no-clobber publication in the same directory
        os.link(temp, path); temp.unlink()
    sync_dir(path.parent)

def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(safe(value), ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')
        f.flush(); os.fsync(f.fileno())

def load_jsonl(path):
    with Path(path).open() as f: return [json.loads(s) for s in f if s.strip()]

def source_hashes():
    names = list((ROOT/'mmimpress').rglob('*.py'))
    names += [ROOT/'scripts'/n for n in ('49_eval_query_aware_baseline.py', '90_validate_llava_kv25.py',
        '97_validate_sparsevlm_ssd_kv25.py', '99_audit_sparsevlm_ssd_kv25.py',
        '100_eval_llava_mt_gqa_allhead_kv25.py', '101_audit_llava_mt_gqa_allhead_kv25.py',
        '102_validate_llava_mt_gqa_allhead_kv25.py')]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(names)}

def frozen_dialogues():
    from mmimpress.mt_gqa import workload_sha256
    path = ROOT/'data/mt_gqa/dialogues.json'
    if sha(path) != INDEX_SHA: raise RuntimeError('DATASET_IDENTITY_FAIL: index SHA256')
    doc = json.loads(path.read_text()); ds = doc['dialogues']
    if workload_sha256(ds) != WORKLOAD_SHA: raise RuntimeError('DATASET_IDENTITY_FAIL: workload SHA256')
    assert len(ds) == 4061 and len({d['image_id'] for d in ds}) == 398
    assert sum(len(d['turns']) for d in ds) == 12183
    qids = []
    for i, d in enumerate(ds):
        assert [t['turn_id'] for t in d['turns']] == [1, 2, 3]
        assert (ROOT/d['image_path']).is_file()
        qids += [str(t['question_id']) for t in d['turns']]
    assert len(set(qids)) == 12183
    return ds

def method_order(ordinal):
    i = int(ordinal) % 5
    return METHODS[i:]+METHODS[:i]

def request_id(run_id, phase, method, dialog_id, turn):
    return f'{SCHEMA}:{run_id}:{phase}:{MODEL_ID}:{method}:{dialog_id}:t{turn}'

def prompt_factory(dialog, turn, method, generated):
    if method not in METHODS or turn not in (1, 2, 3): raise ValueError('invalid method/turn')
    if any(t >= turn for t in generated): raise ValueError('future/current answer in history')
    lines, entries = [], []
    for t in range(1, turn):
        prior = dialog['turns'][t-1]; row = generated[t]
        if row['method_id'] != method or row['dialog_id'] != dialog['dialog_id']:
            raise ValueError('foreign method/dialogue history')
        answer = row['prediction']
        lines += [f"Q{t}: {prior['question'].strip()}", f'A{t}: {answer}']
        entries.append({'turn_id': t, 'question_id': str(prior['question_id']),
            'question': prior['question'].strip(), 'answer': answer,
            'answer_source': 'method_local_generated', 'source_model': MODEL_ID,
            'source_method_id': method, 'source_dialog_id': dialog['dialog_id'],
            'source_logical_request_id': row['logical_request_id'],
            'source_physical_execution_id': row['physical_execution_id']})
    history = '\n'.join(lines); body = ([history, ''] if history else [])
    body += [f"Current question Q{turn}: {dialog['turns'][turn-1]['question'].strip()}", f'{SHORT_ANSWER} ASSISTANT:']
    return 'USER: <image>\n'+'\n'.join(body), history, entries

def strict_score(pred, gold):
    def norm(v): return ' '.join(w for w in re.sub(r'[^\w\s]', ' ', str(v).lower()).split() if w not in {'a','an','the'})
    return float(norm(pred) == norm(gold))

def assert_gpu_exclusive():
    p = subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    other = [s for s in p.stdout.splitlines() if s.strip() and int(s.split(',')[0]) != os.getpid()]
    if other: raise RuntimeError('GPU_TIMING_CONTAMINATION: '+repr(other))

@contextmanager
def conditioning_trace():
    original = os.posix_fadvise; events = []
    def call(fd, offset, length, advice):
        path = os.readlink(f'/proc/self/fd/{fd}')
        try:
            result = original(fd, offset, length, advice)
            events.append({'path': path, 'offset': offset, 'length': length, 'advice': advice, 'status': 'SUCCESS'})
            return result
        except BaseException as exc:
            events.append({'path': path, 'status':'FAIL', 'error':repr(exc)}); raise
    os.posix_fadvise = call
    try: yield events
    finally: os.posix_fadvise = original

def text_spans(prompt,question,history,entries):
    spans=[]
    if history:
        start=prompt.index(history);spans.append({'kind':'history','start_char':start,'end_char':start+len(history),
            'provenance':'method_local_generated_QA','entries':entries})
    start=prompt.rfind(question)
    if start<0:raise ValueError('current question absent from prompt')
    spans.append({'kind':'current_question','start_char':start,'end_char':start+len(question)})
    image_end=prompt.index('<image>')+len('<image>')
    covered=sorted((x['start_char'],x['end_char']) for x in spans)
    cursor=image_end
    for lo,hi in covered:
        if cursor<lo:spans.append({'kind':'template','start_char':cursor,'end_char':lo})
        cursor=max(cursor,hi)
    if cursor<len(prompt):spans.append({'kind':'template','start_char':cursor,'end_char':len(prompt)})
    return sorted(spans,key=lambda x:x['start_char'])


class PilotCounts:
    """Actual phase-aware projection calls; first model invocation is prefill."""
    def __init__(self,runner):self.runner=runner
    def __enter__(self):
        self.calls={};self.handles=[];self.model_calls=0;self.phase=None
        def begin(module,args,kwargs):
            self.phase='prefill' if self.model_calls==0 else 'decode';self.model_calls+=1
        self.handles.append(self.runner.model.register_forward_pre_hook(begin,with_kwargs=True))
        for li,layer in enumerate(self.runner.layers):
            for kind in ('q','k','v'):
                def count(mod,args,li=li,kind=kind):
                    if self.phase is None:raise AssertionError('projection before model phase')
                    counts=self.calls.setdefault(str(li),{}).setdefault(self.phase,{})
                    counts[kind]=counts.get(kind,0)+1
                self.handles.append(getattr(layer.self_attn,kind+'_proj').register_forward_pre_hook(count))
        return self
    def __exit__(self,*exc):
        for h in self.handles:h.remove()
    def result(self):return self.calls

def normal_request(runner,server,image_path,factory,capture_kind):
    from contextlib import ExitStack
    from mmimpress.piggyback import VisionForwardCapture,DecoderVisualHiddenCapture
    vision=VisionForwardCapture(runner,capture_saliency=capture_kind=='image_only')
    hidden=None
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    prompt,history,entries=factory()
    with Image.open(image_path) as im:image=im.convert('RGB');image.load()
    enc,processing=QA._combined_processor(runner,image,prompt)
    v0,vn=runner.visual_span(enc['input_ids'])
    if capture_kind in ('raster','mpic','allhead_verify'):hidden=DecoderVisualHiddenCapture(runner,v0,vn)
    with vision,PilotCounts(runner) as counts,ExitStack() as stack:
        if hidden is not None:stack.enter_context(hidden)
        result=server.recompute(runner.to_device(enc),return_past_key_values=capture_kind!='none')
    returned=time.perf_counter()
    assert vision.call_count==1
    result.update(projection_calls=counts.result(),scoring_calls=0,
                  end_to_end_ttft_ms=(result['first_token_at_s']-started)*1000,
                  request_e2e_ms=(returned-started)*1000,request_started_at_s=started,
                  vision_forward_count=vision.call_count,cap_reached=(len(result['generated_token_ids'])==16 and result['generated_token_ids'][-1]!=runner.processor.tokenizer.eos_token_id),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),processor=processing)
    diag={'prompt':prompt,'history':history,'history_entries':entries,'enc_cpu':enc,'hidden':hidden,'vision':vision,
          'image_input_sha256':QA._image_input_hash(enc),'v_token_start':v0,'v_token_num':vn}
    return result,diag

def legacy_hit(runner,server,ctx,factory,method,image_id):
    with QA._NoVisionForward(runner) as guard:
        c0=time.perf_counter();ctx.reader.drop_all();conditioning=(time.perf_counter()-c0)*1000
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        prompt,history,entries=factory();tokenized=runner.processor.tokenizer(prompt,return_tensors='pt')
        suffix=QA._suffix_from_tokenized(runner,tokenized).to(runner.model.device)
        with PilotCounts(runner) as counts,V.ReadTrace() as reads:
            if method=='fullload':result=server.request(ctx,mode='fullload',suffix_ids=suffix,cold=False)
            else:result=server.request_cvpr25(ctx,static=None,budget=.25,mode='prefix',budget_unit='visual_kv',
                 sep_policy='sidecar',cold=False,seed=1234,image_id=image_id,suffix_ids=suffix,
                 expected_prefix_layout='visionzip_image_only')
        returned=time.perf_counter()
    assert sum(r['returned'] for r in reads.calls)==result['io']['bytes']
    assert len(reads.calls)==result['io']['preads']
    result.update(projection_calls=counts.result(),scoring_calls=0,
        end_to_end_ttft_ms=(result['first_token_at_s']-started)*1000,request_e2e_ms=(returned-started)*1000,
        request_started_at_s=started,vision_forward_count=guard.calls,page_cache_conditioning_ms=conditioning,
        cap_reached=(len(result['generated_token_ids'])==16 and result['generated_token_ids'][-1]!=runner.processor.tokenizer.eos_token_id),
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        os_pread_trace=reads.calls)
    # The next request must not retain any selected visual payload on the GPU.
    ctx.cache.k=None;ctx.cache.v=None
    return result,{'prompt':prompt,'history':history,'history_entries':entries,'suffix_input_ids':suffix.cpu().tolist()}

def sparse_hit(runner,server,ctx,factory,method,question):
    with QA._NoVisionForward(runner) as guard:
        c0=time.perf_counter();ctx.drop_cache();conditioning=(time.perf_counter()-c0)*1000
        torch.cuda.synchronize();started=time.perf_counter()
        prompt,history,entries=factory();spans=text_spans(prompt,question,history,entries)
        with V.ReadTrace() as reads:
            result=server.request(ctx,method_id=method,question=question,prompt_text=prompt,
                                  text_spans=spans,cold=False)
        returned=time.perf_counter()
    V.verify_reads(result,reads,ctx.meta,method)
    result.update(core_ttft_ms=result['end_to_end_ttft_ms'],end_to_end_ttft_ms=(result['first_token_at_s']-started)*1000,
       request_e2e_ms=(returned-started)*1000,request_started_at_s=started,vision_forward_count=guard.calls,
       page_cache_conditioning_ms=conditioning,os_pread_trace=reads.calls)
    return result,{'prompt':prompt,'history':history,'history_entries':entries,'text_spans':spans}


class MPICCounts:
    """MPIC traverses layers directly for prefill, then calls model for decode."""
    def __init__(self, runner): self.runner = runner
    def __enter__(self):
        self.calls = {}; self.handles = []; self.phase = 'prefill'
        def model_begin(*args): self.phase = 'decode'
        self.handles.append(self.runner.model.register_forward_pre_hook(model_begin))
        for li, layer in enumerate(self.runner.layers):
            for kind in ('q', 'k', 'v'):
                def count(mod, args, li=li, kind=kind):
                    d = self.calls.setdefault(str(li), {}).setdefault(self.phase, {})
                    d[kind] = d.get(kind, 0)+1
                self.handles.append(getattr(layer.self_attn, kind+'_proj').register_forward_pre_hook(count))
        return self
    def __exit__(self, *exc):
        for h in self.handles: h.remove()

def mpic_hit(runner, server, ctx, factory):
    with QA._NoVisionForward(runner) as guard:
        c0 = time.perf_counter(); ctx.reader.drop_all(); condition_ms = (time.perf_counter()-c0)*1000
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
        prompt, history, entries = factory()
        with MPICCounts(runner) as counts, V.ReadTrace() as reads:
            result = server.request(ctx, prompt_text=prompt, cold=False)
        returned = time.perf_counter()
    assert sum(x['returned'] for x in reads.calls) == result['io']['bytes']
    assert len(reads.calls) == result['io']['preads']
    result.update(answer=result['prediction'], projection_calls=counts.calls, scoring_calls=0,
        end_to_end_ttft_ms=(result['first_token_at_s']-started)*1000,
        request_e2e_ms=(returned-started)*1000, request_started_at_s=started,
        vision_forward_count=guard.calls, page_cache_conditioning_ms=condition_ms,
        cap_reached=len(result['generated_token_ids']) == 16 and result['generated_token_ids'][-1] != runner.processor.tokenizer.eos_token_id,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        os_pread_trace=reads.calls)
    return result, {'prompt':prompt, 'history':history, 'history_entries':entries}

def stored_request(runner, servers, ctx, factory, method, image_id, question):
    original = [l.self_attn.forward for l in runner.layers]
    with conditioning_trace() as conditioning:
        if method == ALLHEAD: value = sparse_hit(runner, servers['allhead'], ctx, factory, method, question)
        elif method == 'mpic32_ssd': value = mpic_hit(runner, servers['mpic'], ctx, factory)
        elif method in ('fullload','ours_kv25'): value = legacy_hit(runner, servers['old'], ctx, factory, method, image_id)
        else: raise ValueError('unknown stored method')
    result, diag = value
    assert all(l.self_attn.forward == f for l, f in zip(runner.layers, original)), 'adapter not restored'
    from mmimpress.serve import BIAS
    assert not BIAS, 'stale mask after request'
    assert conditioning and all(e['status'] == 'SUCCESS' for e in conditioning), 'page cache conditioning not verified'
    result.update(conditioning_events=conditioning, adapter_restored=True,
        page_cache_conditioning_method='posix_fadvise_DONTNEED', page_cache_conditioning_excluded_from_ttft=True)
    return result, diag

def captured_canonical_hashes(result, diag):
    from mmimpress.model import cache_layers
    v0, vn = diag['v_token_start'], diag['v_token_num']; hashes = {}
    for li, (key, value) in enumerate(cache_layers(result['captured_past_key_values'])):
        for kind, tensor in (('k',key), ('v',value)):
            block = tensor[0, :, v0:v0+vn].permute(1,0,2).to(dtype=torch.float16, device='cpu').contiguous()
            hashes[f'layer_{li:02d}/{kind}.bin'] = hashlib.sha256(block.numpy().tobytes()).hexdigest()
    hidden = diag['hidden'].result_cpu().to(torch.float16).contiguous()
    return {'payload_sha256': hashes, 'visual_hidden_tensor_sha256':hashlib.sha256(hidden.numpy().tobytes()).hexdigest(),
        'prefix_ids': diag['enc_cpu']['input_ids'][0,:v0+vn].tolist(), 'image_input_sha256':diag['image_input_sha256']}

def persist(runner, result, diag, path, image_id, kind, source):
    from mmimpress.piggyback import persist_captured_raster_prefix, persist_captured_visual_prefix
    from mmimpress.mpic import persist_captured_mpic_prefix
    enc = diag['enc_cpu']; args = (runner,result['captured_past_key_values'],enc['input_ids'],enc['image_sizes'][0])
    common = dict(image_id=image_id, model_id=MODEL_ID, chunk_size=64,
        image_input_sha256=diag['image_input_sha256'], extra_metadata={
            'dataset':'MT-GQA-reconstructed','source_dialog_id':source['dialog_id'],
            'source_turn_id':1,'source_method_id':source['method_id'],
            'source_execution_id':source['physical_execution_id'], 'source_logical_request_id':source['logical_request_id'],
            'run_schema':SCHEMA})
    if kind == 'mpic':
        return persist_captured_mpic_prefix(*args, diag['hidden'].result_cpu(), path, hidden_capture_stats=diag['hidden'], **common)
    if kind == 'raster':
        # Preserve the validated serializer unchanged, including its unused
        # bundled probe sidecar. It is never read by any main method.
        return persist_captured_raster_prefix(*args, diag['hidden'].result_cpu(), path, probe_heads=3,
            hidden_capture_stats=diag['hidden'], full_integrity_hash=True, **common)
    return persist_captured_visual_prefix(*args, diag['vision'].result_cpu(), path,
        capture_stats=diag['vision'], full_integrity_hash=True, **common)

def file_hashes(path):
    hashes = {}
    for p in sorted(Path(path).rglob('*')):
        if p.is_symlink(): raise ValueError('symlink in new store')
        if p.is_file():
            if p.stat().st_nlink != 1: raise ValueError('hardlink in new store')
            hashes[p.relative_to(path).as_posix()] = sha(p)
    return hashes

def activate(runner, method, path, hashes):
    from mmimpress.serve import ImageContext
    from mmimpress.mpic import MPICContext
    from mmimpress.sparsevlm_ssd_store import CanonicalContext
    started = time.perf_counter()
    if method == ALLHEAD:
        required = {k:v for k,v in hashes.items() if k in ('meta.json','sys_kv.pt','v_hidden.pt','sep_kv.bin') or k.endswith('/k.bin') or k.endswith('/v.bin')}
        ctx = CanonicalContext(path, head_policy='all', expected_hashes=required)
    elif method == 'mpic32_ssd':
        ctx = MPICContext(path, runner.model.device, drop_cache=False, runner=runner)
        ctx.activation = {'host_metadata_file_bytes':ctx.metadata_resident_bytes, 'gpu_tensor_bytes':0,
            'visual_payload_resident_bytes':0, 'policy':'immutable metadata only'}
    else:
        ctx = ImageContext(path, runner.model.device, drop_cache=False, require_v_hidden=False)
        if method == 'ours_kv25': ctx.validate_prefix_layout('visionzip_image_only')
        ctx.activation = {'host_tensor_bytes':sum(t.numel()*t.element_size() for t in ctx.cache.sys_kv.values()),
            'gpu_tensor_bytes':0, 'visual_payload_resident_bytes':0,
            'file_bytes':sum((path/n).stat().st_size for n in ('meta.json','sys_kv.pt'))}
    ctx.activation['outer_wall_ms'] = (time.perf_counter()-started)*1000
    return ctx

def make_row(runner, result, diag, meta, *, method, dialog, turn, phase, run, attempt, physical_id, ordinal, store_info, config):
    question = dialog['turns'][turn-1]; n = int(meta['n_spatial']); hit = turn > 1
    k = (n+3)//4 if hit and method in (ALLHEAD,'ours_kv25') else n
    logical = request_id(run.name, phase, method, dialog['dialog_id'], turn)
    suffix = QA._suffix_from_tokenized(runner, runner.processor.tokenizer(diag['prompt'], return_tensors='pt')).cpu().tolist()
    if result.get('suffix_input_ids') is not None: assert result['suffix_input_ids'] == suffix
    value = {a:b for a,b in result.items() if a not in ('captured_past_key_values','first_logits','logits')}
    g = {key:meta.get(key, []) if key=='padding_idx' else meta[key] for key in
        ('num_heads','head_dim','num_layers','v_token_start','v_token_num','n_spatial','prefix_len','dtype','newline_idx','padding_idx')}
    return {'schema_version':SCHEMA,'experiment_id':run.name, 'phase':phase, 'attempt_id':attempt,
        'logical_request_id':logical,'request_id':logical,'physical_execution_id':physical_id,
        'model_id':MODEL_ID,'model_revision':REVISION,'method_id':method,'image_id':dialog['image_id'],
        'dialog_id':dialog['dialog_id'],'turn_id':turn,'question_id':str(question['question_id']),
        'question':question['question'],'gold':question['answers'],'prediction':result['answer'],
        'score':strict_score(result['answer'], question['answers'][0]),'prompt':diag['prompt'],
        'history':diag['history'],'history_entries':diag['history_entries'],
        'prompt_sha256':hashlib.sha256(diag['prompt'].encode()).hexdigest(), 'suffix_input_ids':suffix,
        'suffix_ids_sha256':canonical_hash(suffix),'history_sha256':hashlib.sha256(diag['history'].encode()).hexdigest(),
        'config_sha256':config['_file_sha256'],'manifest_sha256':config['_manifest_sha256'],
        'code_sha256':canonical_hash(config['source_sha256']), 'global_dialog_ordinal':ordinal,
        'method_order':list(method_order(ordinal)),'method_order_position':method_order(ordinal).index(method),
        'request_started_at_s':result['request_started_at_s'],'first_token_at_s':result['first_token_at_s'],
        'request_finished_at_s':result['request_started_at_s']+result['request_e2e_ms']/1000,
        'ttft_ms':result['end_to_end_ttft_ms'],'request_e2e_ms':result['request_e2e_ms'],
        'ssd_read_bytes':int(result.get('io',{}).get('bytes',0)), 'ssd_preads':int(result.get('io',{}).get('preads',0)),
        'N_content':n,'k':k,'content_kv_fraction':k/n,'geometry':g,
        'budget_unit':'visual_kv' if method in (ALLHEAD,'ours_kv25') else 'full_context',
        'scoring_head_count':int(meta['num_heads']) if hit and method == ALLHEAD else 0,
        'scoring_head_policy':'all' if method == ALLHEAD else None,
        'source_store':store_info if hit and method != 'recompute' else None,
        'provisioning_mode':'FRESH_IMAGE_STREAMING','result':safe(value)}

def capacity_guard(run):
    plan = json.loads((run/'storage_plan.json').read_text()); free = shutil.disk_usage(run).free
    if free < plan['minimum_reserve_bytes']+plan['peak_working_set_bytes']:
        raise RuntimeError(f'BLOCKED_STORAGE: free={free}; required={plan["minimum_reserve_bytes"]+plan["peak_working_set_bytes"]}')
    scratch = run/'scratch'
    owned = sum(p.stat().st_size for p in scratch.rglob('*') if p.is_file() and not p.is_symlink()) if scratch.exists() else 0
    if owned+plan['largest_image_bundle_bound_bytes'] > plan['maximum_new_payload_bytes']:
        raise RuntimeError('BLOCKED_STORAGE: retained incomplete payload allowance exceeded')
    return {'free_bytes':free,'retained_scratch_bytes':owned}

def cleanup_committed(scratch, allowlist, audit, commit):
    scratch = Path(scratch)
    if audit.get('status') != 'PASS' or commit.get('status') != 'COMMITTED':
        raise ValueError('independent audit and atomic image commit required')
    if scratch.is_symlink() or not scratch.is_dir(): raise ValueError('unsafe scratch root')
    resolved = scratch.resolve()
    found = sorted(p.relative_to(scratch).as_posix() for p in scratch.rglob('*') if p.is_file() or p.is_symlink())
    if found != sorted(allowlist): raise ValueError('cleanup allowlist mismatch')
    for relative, expected in allowlist.items():
        p = scratch/relative
        if p.is_symlink() or p.stat().st_nlink != 1 or not p.resolve().is_relative_to(resolved): raise ValueError('linked/non-owned cleanup path')
        if sha(p) != expected: raise ValueError('cleanup payload changed')
    for relative in allowlist: (scratch/relative).unlink()
    for p in sorted(scratch.rglob('*'), key=lambda x:len(x.parts), reverse=True):
        if p.is_dir(): p.rmdir()
    scratch.rmdir(); sync_dir(scratch.parent)

def completed_images(run, phase, config, manifest):
    audit = helper('101_audit_llava_mt_gqa_allhead_kv25.py', '_main_resume_independent_audit')
    done = {}
    for cp in sorted((run/phase/'images').glob('*/COMMITTED.json')):
        doc = json.loads(cp.read_text()); folder = cp.parent/doc['attempt_id']; raw = folder/'raw.jsonl'
        if doc['config_sha256'] != config['_file_sha256'] or doc['manifest_sha256'] != config['_manifest_sha256']:
            raise RuntimeError('resume config/manifest mismatch')
        if sha(raw) != doc['raw_sha256']: raise RuntimeError('committed raw hash mismatch')
        receipt = audit.audit_image_rows(load_jsonl(raw), manifest, config, phase, cp.parent.name)
        if receipt['status'] != 'PASS': raise RuntimeError('resume independent audit failed: '+repr(receipt))
        done[cp.parent.name] = doc
    return done

def run_image(runner, servers, dialogs, ordinals, phase, run, manifest, config, diagnostic=None):
    assert len({d['image_id'] for d in dialogs}) == 1
    image_id = dialogs[0]['image_id']; parent = run/phase/'images'/image_id
    parent.mkdir(parents=True, exist_ok=True)
    if (parent/'COMMITTED.json').exists(): raise FileExistsError('image already committed')
    attempt = f'attempt_{len(list(parent.glob("attempt_*")))+1:04d}'
    artifact = parent/attempt; artifact.mkdir(exist_ok=False)
    scratch = run/'scratch'/phase/image_id/attempt
    capacity = capacity_guard(run); atomic_json(artifact/'capacity_before.json', capacity)
    atomic_json(artifact/'scratch_creation.json', {'path':str(scratch),'ownership':run.name,'previously_absent':not scratch.exists()})
    scratch.mkdir(parents=True, exist_ok=False)
    paths = {k:scratch/k for k in STORE_KEYS}; metas = {}; hashes = {}; persistence = {}; contexts = {}; store_info = {}
    rows = []; pending = []; shared_capture = None; success = False
    source_image_sha = sha(ROOT/dialogs[0]['image_path'])
    atomic_json(artifact/'image_identity.json', {'image_id':image_id,'image_path':str((ROOT/dialogs[0]['image_path']).resolve()),'sha256':source_image_sha})
    try:
        for di, dialog in enumerate(dialogs):
            ordinal = ordinals[dialog['dialog_id']]; generated = {m:{} for m in METHODS}
            for turn in (1,2,3):
                for method in method_order(ordinal):
                    assert_gpu_exclusive(); physical = str(uuid.uuid4())
                    logical = request_id(run.name, phase, method, dialog['dialog_id'], turn)
                    source = {'dialog_id':dialog['dialog_id'],'method_id':method,'physical_execution_id':physical,'logical_request_id':logical}
                    append(artifact/'attempts.jsonl', dict(source, turn_id=turn, status='STARTED', at_unix=time.time()))
                    factory = lambda m=method: prompt_factory(dialog, turn, m, generated[m])
                    capture = {'fullload':'raster','mpic32_ssd':'mpic','ours_kv25':'image_only',ALLHEAD:'allhead_verify'}.get(method,'none') if di==0 and turn==1 else 'none'
                    if turn == 1 or method == 'recompute':
                        result, diag = normal_request(runner, servers['old'], ROOT/dialog['image_path'], factory, capture)
                        # Record the physical request before persistence can fail.
                        append(artifact/'execution_events.jsonl', dict(source, turn_id=turn, status='EXECUTED',
                            prediction=result['answer'], generated_token_ids=result['generated_token_ids'],
                            ttft_ms=result['end_to_end_ttft_ms'],request_e2e_ms=result['request_e2e_ms']))
                        if capture == 'allhead_verify':
                            shared_capture = captured_canonical_hashes(result, diag)
                            shared_capture['source'] = source
                        elif capture != 'none':
                            start = time.perf_counter(); p = persist(runner,result,diag,paths[capture],image_id,capture,source)
                            outer_ms = (time.perf_counter()-start)*1000
                            persistence[capture] = dict(p, outer_helper_wall_ms=outer_ms, source=source,
                                capture_accounting='capture and exit materialization already within T1 request E2E; not added again')
                            metas[capture] = json.loads((paths[capture]/'meta.json').read_text())
                            if capture == 'mpic': metas[capture]['n_spatial'] = metas[capture]['v_token_num']-len(metas[capture]['newline_idx'])
                            hashes[capture] = file_hashes(paths[capture])
                            store_info[capture] = {'path':str(paths[capture]),'meta_sha256':sha(paths[capture]/'meta.json'),
                                'source':source,'file_sha256':hashes[capture]}
                            atomic_json(artifact/f'{capture}_persistence.json', persistence[capture])
                            atomic_json(artifact/f'{capture}_meta.json', metas[capture])
                            atomic_json(artifact/f'{capture}_hashes.json', hashes[capture])
                        result.pop('captured_past_key_values', None)
                        saved = {a:b for a,b in diag.items() if a not in ('enc_cpu','hidden','vision')}
                        if di == 0 and turn == 1:
                            pending.append((method,result,saved,physical))
                            # Full T1 history is not consumed until the end of this turn.
                            del diag, result
                            continue
                        diag = saved
                    else:
                        kind = 'mpic' if method=='mpic32_ssd' else 'image_only' if method=='ours_kv25' else 'raster'
                        if method not in contexts:
                            contexts[method] = activate(runner,method,paths[kind],hashes[kind])
                            atomic_json(artifact/f'{method}_activation.json', contexts[method].activation)
                        result, diag = stored_request(runner,servers,contexts[method],factory,method,image_id,dialog['turns'][turn-1]['question'])
                        result['metadata_activation'] = contexts[method].activation
                    kind = 'mpic' if method=='mpic32_ssd' else 'image_only' if method=='ours_kv25' else 'raster'
                    row = make_row(runner,result,diag,metas[kind],method=method,dialog=dialog,turn=turn,phase=phase,
                        run=run,attempt=attempt,physical_id=physical,ordinal=ordinal,store_info=store_info.get(kind),config=config)
                    row['image_sha256'] = source_image_sha
                    append(artifact/'raw.jsonl', row); rows.append(row); generated[method][turn] = row
                    append(artifact/'attempts.jsonl', dict(source,turn_id=turn,status='COMPLETED',at_unix=time.time()))
                    del result, diag
                if di == 0 and turn == 1:
                    assert set(metas) == set(STORE_KEYS)
                    for method, result, diag, physical in pending:
                        kind = 'mpic' if method=='mpic32_ssd' else 'image_only' if method=='ours_kv25' else 'raster'
                        row = make_row(runner,result,diag,metas[kind],method=method,dialog=dialog,turn=turn,phase=phase,
                            run=run,attempt=attempt,physical_id=physical,ordinal=ordinal,store_info=store_info.get(kind),config=config)
                        row['image_sha256'] = source_image_sha
                        append(artifact/'raw.jsonl', row); rows.append(row); generated[method][1] = row
                        append(artifact/'attempts.jsonl', {'physical_execution_id':physical,'logical_request_id':row['logical_request_id'], 'status':'COMPLETED','at_unix':time.time()})
                    pending.clear(); del result, diag
                    assert shared_capture is not None
                    assert all(hashes['raster'][k] == v for k,v in shared_capture['payload_sha256'].items()), 'FullLoad/AllHead captured KV bits differ'
                    assert shared_capture['prefix_ids'] == metas['raster']['prefix_input_ids']
                    assert shared_capture['image_input_sha256'] == metas['raster']['image_input_sha256']
                    hidden = torch.load(paths['raster']/'v_hidden.pt',map_location='cpu',weights_only=True).contiguous()
                    assert hashlib.sha256(hidden.numpy().tobytes()).hexdigest() == shared_capture['visual_hidden_tensor_sha256']
                    for kind in STORE_KEYS:
                        assert metas[kind]['prefix_input_ids'] == metas['raster']['prefix_input_ids']
                        assert metas[kind]['image_input_sha256'] == metas['raster']['image_input_sha256']
                    atomic_json(artifact/'shared_canonical_compatibility.json',dict(shared_capture,status='PASS',
                        scope='AllHead own actual full T1 capture versus FullLoad source T1 store, exact FP16 K/V every layer/head and v_hidden'))
                    del hidden, shared_capture
            if diagnostic is not None and di == 0:
                diagnostic(runner,servers,contexts,paths,hashes,dialog,rows,artifact)
        # All integrity I/O runs between requests and before cleanup.
        for kind in STORE_KEYS:
            assert file_hashes(paths[kind]) == hashes[kind], 'store mutated during requests'
        atomic_json(artifact/'store_integrity_after.json',{'status':'PASS','file_sha256':hashes})
        independent = helper('101_audit_llava_mt_gqa_allhead_kv25.py','_main_commit_independent_audit')
        audit = independent.audit_image_rows(load_jsonl(artifact/'raw.jsonl'),manifest,config,phase,image_id)
        audit['raw_sha256'] = sha(artifact/'raw.jsonl'); atomic_json(artifact/'independent_audit.json',audit)
        if audit['status'] != 'PASS': raise RuntimeError('INDEPENDENT_IMAGE_AUDIT_FAIL: '+repr(audit))
        allowlist = file_hashes(scratch)
        atomic_json(artifact/'cleanup_allowlist.json',{'scratch':str(scratch),'file_sha256':allowlist,'ownership_run':run.name})
        commit = {'status':'COMMITTED','image_id':image_id,'attempt_id':attempt,'requests':len(rows),
            'raw_sha256':audit['raw_sha256'],'audit_sha256':sha(artifact/'independent_audit.json'),
            'config_sha256':config['_file_sha256'],'manifest_sha256':config['_manifest_sha256'],
            'first_successful_complete_image_adoption':True}
        atomic_json(parent/'COMMITTED.json',commit); success = True
    except BaseException as exc:
        atomic_json(artifact/'failure.json',{'status':'PARTIAL','error':repr(exc),'traceback':traceback.format_exc(),
            'completed_uncommitted_rows':len(rows),'main_final_rows_from_this_attempt':0})
        raise
    finally:
        for ctx in contexts.values(): ctx.close()
        contexts.clear(); gc.collect()
        if success:
            cleanup_committed(scratch,allowlist,audit,commit)
            atomic_json(artifact/'cleanup_receipt.json',{'status':'PASS','removed_new_files':len(allowlist),
                'recipe':'same frozen config+manifest; rerun this full image into a fresh run; never reuse its generated histories'})
    return len(rows)

def verify_freeze(run):
    config = json.loads((run/'config.json').read_text()); manifest = json.loads((run/'manifest.json').read_text())
    if list(METHODS) != config['methods'] or source_hashes() != config['source_sha256']: raise RuntimeError('FROZEN_SOURCE_CONFIG_MISMATCH')
    if sha(CONTRACT) != config['contract_sha256']: raise RuntimeError('FROZEN_CONTRACT_MISMATCH')
    if frozen_dialogues() != manifest['dialogues']: raise RuntimeError('FROZEN_DATASET_MISMATCH')
    for image_id, item in manifest['image_identity'].items():
        if sha(ROOT/item['image_path']) != item['sha256']: raise RuntimeError('FROZEN_IMAGE_MISMATCH: '+image_id)
    config.update(_file_sha256=sha(run/'config.json'),_manifest_sha256=sha(run/'manifest.json'))
    return config, manifest

def run_phase(runner, servers, run, phase, config, manifest, diagnostic=None):
    done = completed_images(run,phase,config,manifest)
    ids = manifest['smoke_dialogue_ids'] if phase in ('smoke','integration') else [d['dialog_id'] for d in manifest['dialogues']]
    if phase == 'integration': ids = ids[:1]
    keep = set(ids); groups = OrderedDict(); ordinals = {}
    for ordinal, d in enumerate(manifest['dialogues']):
        ordinals[d['dialog_id']] = ordinal
        if d['dialog_id'] in keep: groups.setdefault(d['image_id'],[]).append(d)
    expected = len(ids)*15; completed = sum(x['requests'] for x in done.values())
    for index, (image, ds) in enumerate(groups.items()):
        if image in done: continue
        completed += run_image(runner,servers,ds,ordinals,phase,run,manifest,config,diagnostic=diagnostic)
        progress = {'phase':phase,'status':'RUNNING','completed_final_requests':completed,'expected_requests':expected,
            'completed_images':len(done)+1,'image_id':image,'at_unix':time.time()}
        done[image] = {'requests':len(ds)*15}
        atomic_json(run/f'{phase}_progress.json',progress,replace=True); print(json.dumps(progress),flush=True)
    assert completed == expected
    receipt = {'status':'PASS','requests':completed,'expected_requests':expected,'images':len(groups),
        'independent_image_audit':'PASS','source_sha256':config['source_sha256'],'config_sha256':config['_file_sha256']}
    path = run/phase/'validation.json'
    if path.exists():
        if json.loads(path.read_text()) != receipt: raise RuntimeError('final phase validation mismatch')
    else: atomic_json(path,receipt)
    return receipt

def servers_for(runner):
    from mmimpress.serve import Server
    from mmimpress.mpic import MPICServer
    from mmimpress.sparsevlm_ssd_attention import SparseVLMSSDServer
    return {'old':Server(runner,max_new_tokens=16),'mpic':MPICServer(runner,k_recompute=32,max_new_tokens=16),
        'allhead':SparseVLMSSDServer(runner,max_new_tokens=16)}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--phase',choices=('smoke','main','all'),default='all');p.add_argument('--resume',action='store_true')
    args=p.parse_args();run=args.run_dir.resolve();config,manifest=verify_freeze(run)
    gate=json.loads((run/'integration_validation.json').read_text())
    if gate.get('status')!='PASS' or gate.get('source_sha256')!=config['source_sha256']:raise RuntimeError('INTEGRATION_GATE_NOT_PASS')
    phases=['smoke','main'] if args.phase=='all' else [args.phase]
    assert_gpu_exclusive();runner=V.load_runner();servers=servers_for(runner);QA._warmup(runner,servers['old'])
    try:
        for phase in phases:
            if phase=='main':
                smoke=json.loads((run/'smoke'/'validation.json').read_text())
                if smoke['status']!='PASS' or smoke['requests']!=60:raise RuntimeError('SMOKE_NOT_PASS')
            if (run/phase).exists() and not args.resume:raise FileExistsError('use --resume; previous artifacts preserved')
            run_phase(runner,servers,run,phase,config,manifest)
        subprocess.run([sys.executable,str(ROOT/'scripts/101_audit_llava_mt_gqa_allhead_kv25.py'),
            '--run-dir',str(run),'--output-dir',str(ROOT/'results'/run.name)],check=True)
    except BaseException as exc:
        error={'status':'PARTIAL','error':repr(exc),'traceback':traceback.format_exc(),
            'main_final_requests':sum(v['requests'] for v in completed_images(run,'main',config,manifest).values()),
            'expected':60915,'resume_command':f'{sys.executable} scripts/100_eval_llava_mt_gqa_allhead_kv25.py --run-dir {run} --phase all --resume'}
        atomic_json(run/('interruption_'+time.strftime('%Y%m%dT%H%M%S')+'.json'),error)
        print(json.dumps(error),flush=True);raise
if __name__=='__main__':main()

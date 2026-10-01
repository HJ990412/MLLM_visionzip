#!/usr/bin/env python3
"""Single-image fresh-T1 setup-cost diagnostic, separate from pilot metrics.

The production raster writer bundles probe sidecars. This run-owned writer
separates canonical base persistence and optional raw first-three-head sidecar
persistence without modifying that writer or any serving source. The temporary
CPU canonical K originates from this script's one actual normal T1. No existing
store is copied, read to manufacture a fresh build, modified, or removed.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
RUN=Path(__file__).resolve().parent
IMAGE_ID='n355567'
QUESTION_ID='201751701'
RESERVE_BYTES=20<<30
MAX_PAYLOAD_BYTES=8<<30
SCHEMA='sparsevlm-ssd-fresh-t1-split-setup-diagnostic-v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def load_script(name,relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def write_json_exclusive(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def write_bytes(path,value):
    with Path(path).open('xb') as f:
        count=f.write(value)
    if count!=len(value):raise IOError('short setup payload write')


def footprint(path):
    sizes={};allocated={};hashes={}
    for p in sorted(Path(path).rglob('*')):
        if p.is_symlink():raise RuntimeError('unexpected setup symlink')
        if p.is_file():
            name=p.relative_to(path).as_posix();st=p.stat()
            sizes[name]=st.st_size;allocated[name]=st.st_blocks*512;hashes[name]=sha(p)
    return {'logical_bytes':sum(sizes.values()),'allocated_file_bytes':sum(allocated.values()),
            'file_sizes':sizes,'files_sha256':hashes,'hashing_in_setup_timer':False}


def publish(staging,destination):
    from mmimpress.piggyback import _fsync_staging_tree,_rename_noreplace,_fsync_directory
    file_ms,dir_ms,nfiles,ndirs=_fsync_staging_tree(staging)
    t=time.perf_counter();_rename_noreplace(staging,destination);rename_ms=(time.perf_counter()-t)*1000
    t=time.perf_counter();_fsync_directory(destination.parent);parent_ms=(time.perf_counter()-t)*1000
    return {'file_fsync_ms':file_ms,'directory_fsync_ms':dir_ms,'rename_ms':rename_ms,'parent_fsync_ms':parent_ms,
            'files_fsynced':nfiles,'directories_fsynced':ndirs,'no_clobber_atomic_publish':True,
            'fsync_inclusive':True}


def base_metadata(runner,encoded,hidden,image_hash):
    import torch
    ids=encoded['input_ids'][0];v0,vn=runner.visual_span(ids);p=v0+vn
    base,hi_h,hi_w,separators=runner.anyres_layout(encoded['image_sizes'][0],vn)
    h=int(runner.cfg.text_config.num_attention_heads);kh=int(runner.cfg.text_config.num_key_value_heads)
    if h!=kh:raise ValueError('MHA-only setup diagnostic')
    capture=hidden.stats()
    assert capture['visual_hidden_capture_count']==1
    assert capture['capture_source']=='same_turn1_normal_multimodal_prefill'
    assert capture['separate_model_forward_count']==capture['separate_vision_forward_count']==0
    return {
      'schema_version':SCHEMA,'image_id':IMAGE_ID,'model':runner.model_id,'model_revision':'c916e6cdcd760b4cecd1dd4907f84ac649f93b23',
      'v_token_start':v0,'v_token_num':vn,'prefix_len':p,'num_layers':len(runner.layers),'num_heads':h,
      'num_key_value_heads':kh,'head_dim':runner.head_dim,'chunk_size':64,'dtype':'float16',
      # This is a file inventory field only. Serving policy remains explicit.
      'probe_heads':0,'probe_k_present_in_base':False,'scoring_head_policy':'all',
      'n_chunks_per_layer':(vn+63)//64,'n_spatial':vn-len(separators),'newline_idx':separators,
      'newline_stored':separators,'padding_idx':[],'prefix_input_ids':ids[:p].cpu().tolist(),
      'base_grid':base,'hires_grid':[hi_h,hi_w],
      'layout':'token-major (v_token_num, num_heads, head_dim) fp16; original raster positions',
      'physical_layout':'raster','layout_method':'raster','reordered':False,'order_is_per_layer':False,
      'layout_source':'turn1_normal_inference_piggyback','visual_kv_source':'turn1_captured_past_key_values',
      'visual_hidden_source':'same_turn1_decoder_layer0_input','turn1_normal_inference':True,
      'separate_vision_forward':False,'separate_prefix_forward':False,'separate_model_forward_for_visual_hidden':False,
      'layout_uses_dataset_question':False,'layout_uses_generated_answer':False,'calibration_questions':0,
      'future_questions_used':0,'visual_prefix_causally_precedes_question':True,'capture_provenance_validated':True,
      'hidden_capture':capture,'source_turn_id':1,'source_question_id':QUESTION_ID,'source_T1':'this_diagnostic_normal_full_image',
      'image_input_sha256':image_hash,'cached_K_semantics':'post_RoPE; no second rotary application',
      'separator_policy':'original_positions_plus_sidecar','bytes_probe_sidecar':0,
      'independent_deployment_method_id':'sparsevlm_ssd_kv25_allhead',
    }


def persist_base(capture_layers,hidden,meta,destination):
    import torch
    t0=time.perf_counter();staging=Path(tempfile.mkdtemp(prefix='.base.staging-',dir=destination.parent))
    p,v0,vn,h,d=(int(meta[k]) for k in ('prefix_len','v_token_start','v_token_num','num_heads','head_dim'))
    native_keys=[];sep_k=[];sep_v=[];conversion_ms=write_ms=0.0
    system={}
    for kind,index in [('k',0),('v',1)]:
        t=time.perf_counter()
        system[kind]=torch.stack([layer[index][0,:,:v0].detach().to(dtype=torch.float16,device='cpu') for layer in capture_layers])
        conversion_ms+=(time.perf_counter()-t)*1000
    t=time.perf_counter();torch.save(system,staging/'sys_kv.pt');write_ms+=(time.perf_counter()-t)*1000
    separator_ids=torch.tensor(meta['newline_idx'],dtype=torch.long)
    for li,layer in enumerate(capture_layers):
        directory=staging/f'layer_{li:02d}';directory.mkdir()
        for kind,index in [('k',0),('v',1)]:
            source=layer[index]
            assert tuple(source.shape[:2])==(1,h) and source.shape[2]>=p and source.shape[3]==d
            t=time.perf_counter()
            canonical=source[0,:,v0:p].permute(1,0,2).detach().to(dtype=torch.float16,device='cpu').contiguous()
            conversion_ms+=(time.perf_counter()-t)*1000
            t=time.perf_counter();write_bytes(directory/f'{kind}.bin',canonical.numpy().tobytes());write_ms+=(time.perf_counter()-t)*1000
            (sep_k if kind=='k' else sep_v).append(canonical[separator_ids].clone())
            if kind=='k':native_keys.append(canonical)
    t=time.perf_counter();visual_hidden=hidden.result_cpu().to(dtype=torch.float16,device='cpu').contiguous()
    separator=torch.stack([torch.stack(sep_k),torch.stack(sep_v)])
    conversion_ms+=(time.perf_counter()-t)*1000
    t=time.perf_counter();torch.save(visual_hidden,staging/'v_hidden.pt');write_bytes(staging/'sep_kv.bin',separator.numpy().tobytes())
    write_ms+=(time.perf_counter()-t)*1000
    meta.update(bytes_visual_kv=2*len(capture_layers)*vn*h*d*2,bytes_separator_sidecar=separator.numel()*separator.element_size())
    t=time.perf_counter()
    with (staging/'meta.json').open('x') as f:json.dump(meta,f,indent=1,allow_nan=False)
    write_ms+=(time.perf_counter()-t)*1000
    durability=publish(staging,destination);elapsed=(time.perf_counter()-t0)*1000
    return native_keys,{'phase':'fresh_AllHead_required_base','wall_ms_fsync_inclusive':elapsed,
      'D2H_cast_materialize_ms':conversion_ms,'serialization_write_ms':write_ms,
      'durability':durability,'all_setup_CPU_K_retained_bytes':sum(k.numel()*k.element_size() for k in native_keys),
      'scope':'canonical full K/V + system K/V + visual embeddings + structural KV + base metadata; no probe files',
      'storage_layout':'canonical token-major FP16','full_model_forwards':0,'vision_forwards':0,
      'existing_store_reads':0,'scoring_head_policy':'all'}


def persist_probe(native_keys,meta,destination):
    import torch
    t0=time.perf_counter();staging=Path(tempfile.mkdtemp(prefix='.probe3.staging-',dir=destination.parent))
    extraction_ms=write_ms=0.0
    for li,canonical in enumerate(native_keys):
        assert canonical.dtype==torch.float16 and canonical.device.type=='cpu'
        directory=staging/f'layer_{li:02d}';directory.mkdir()
        t=time.perf_counter();probe=canonical[:,:3,:].contiguous();extraction_ms+=(time.perf_counter()-t)*1000
        t=time.perf_counter();write_bytes(directory/'probe_k.bin',probe.numpy().tobytes());write_ms+=(time.perf_counter()-t)*1000
    durability=publish(staging,destination);elapsed=(time.perf_counter()-t0)*1000
    return {'phase':'fresh_Probe3_incremental_sidecar','wall_ms_fsync_inclusive':elapsed,
      'raw_first3_extraction_ms':extraction_ms,'serialization_write_ms':write_ms,'durability':durability,
      'source':'same_actual_T1_canonical_FP16_CPU_K_retained_from_immediately_preceding_base_phase',
      'scoring_head_policy':'fixed_first_3','head_ids':[0,1,2],'full_model_forwards':0,'vision_forwards':0,
      'canonical_SSD_read_bytes':0,'existing_store_reads':0,
      'standalone_Probe3_setup_definition':'same fresh base phase plus this incremental sidecar phase; T1 counted once'}


def independent_readback(capture_layers,hidden,meta,base,probe):
    import numpy as np
    import torch
    m=meta;v0,vn,h,d=(int(m[k]) for k in ('v_token_start','v_token_num','num_heads','head_dim'))
    sys_expected={'k':[],'v':[]};seps={'k':[],'v':[]};checks=0
    for li,(key,value) in enumerate(capture_layers):
        for name,source in [('k',key),('v',value)]:
            # Recompute from the actual fresh T1 tensors, not the writer's CPU cache.
            expected=source[0,:,v0:v0+vn].permute(1,0,2).to(dtype=torch.float16,device='cpu').contiguous().numpy()
            if not np.isfinite(expected).all():raise AssertionError('nonfinite fresh canonical payload')
            got=np.fromfile(base/f'layer_{li:02d}'/f'{name}.bin',dtype=np.uint16).reshape(vn,h,d)
            if not np.array_equal(got,expected.view(np.uint16)):raise AssertionError(f'{name} native bits mismatch layer {li}')
            sys_expected[name].append(source[0,:,:v0].to(dtype=torch.float16,device='cpu'))
            seps[name].append(expected[np.asarray(meta['newline_idx'])].copy())
            checks+=1
            if name=='k':
                probe_bits=np.fromfile(probe/f'layer_{li:02d}'/'probe_k.bin',dtype=np.uint16).reshape(vn,3,d)
                if not np.array_equal(probe_bits,got[:,[0,1,2],:]):raise AssertionError(f'raw first3 bits mismatch layer {li}')
                checks+=1
    system=torch.load(base/'sys_kv.pt',weights_only=True,map_location='cpu')
    for kind in ('k','v'):
        a=system[kind].contiguous().view(torch.uint8);b=torch.stack(sys_expected[kind]).contiguous().view(torch.uint8)
        if not torch.equal(a,b):raise AssertionError('system prefix bits mismatch')
        checks+=1
    hsave=torch.load(base/'v_hidden.pt',weights_only=True,map_location='cpu')
    if not torch.equal(hsave.view(torch.uint8),hidden.result_cpu().to(torch.float16).contiguous().view(torch.uint8)):raise AssertionError('v_hidden bits mismatch')
    expected_seps=np.stack([np.stack(seps['k']),np.stack(seps['v'])])
    actual_seps=np.fromfile(base/'sep_kv.bin',dtype=np.uint16).reshape(expected_seps.shape)
    if not np.array_equal(actual_seps,expected_seps.view(np.uint16)):raise AssertionError('structural sidecar bits mismatch')
    assert not list(base.rglob('probe_k.bin'))
    return {'status':'PASS','checks':checks+3,'canonical_full_layer_head_KV_bits_exact':True,
       'raw_probe3_heads_012_bits_exact':True,'system_KV_bits_exact':True,'v_hidden_bits_exact':True,
       'structural_KV_bits_exact':True,'base_probe_files':0,'reference':'actual_same_T1_cache_recast_independently_to_FP16',
       'timing_excluded':True}


def cleanup_owned(output,payload,receipt):
    if receipt['independent_readback']['status']!='PASS':raise RuntimeError('no valid readback receipt')
    owned=[]
    for p in payload.rglob('*'):
        if p.is_symlink():raise RuntimeError('cleanup refuses symlink')
        if p.is_file():
            if p.stat().st_nlink!=1 or not p.resolve().is_relative_to(payload.resolve()):raise RuntimeError('cleanup refuses linked/external file')
            owned.append(p.relative_to(payload).as_posix())
    write_json_exclusive(output/'cleanup_allowlist.json',{'payload':str(payload),'files':sorted(owned),
         'receipt_sha256':sha(output/'receipt.json')})
    for relative in owned:(payload/relative).unlink()
    for p in sorted(payload.rglob('*'),key=lambda q:len(q.parts),reverse=True):
        if p.is_dir():p.rmdir()
    payload.rmdir()
    write_json_exclusive(output/'cleanup_receipt.json',{'status':'PASS','removed_run_owned_files':len(owned),'previous_artifacts_modified':False})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--cleanup',action='store_true')
    args=parser.parse_args();output=args.output_dir.resolve()
    if not output.is_relative_to(RUN.resolve()):raise ValueError('diagnostic output must be under this run')
    if output.exists():raise FileExistsError('use a new output-dir; preserve prior diagnostic artifacts')
    free=shutil.disk_usage(RUN).free
    if free<RESERVE_BYTES+MAX_PAYLOAD_BYTES:raise RuntimeError('BLOCKED_STORAGE: reserve not available')
    pilot=load_script('_split_setup_pilot','scripts/98_eval_sparsevlm_ssd_kv25.py')
    gate=pilot.gpu_gate(RUN/'gpu_full_01/gpu_validation.json')
    gqa,_,work=pilot.V.frozen_workloads();entry=next(x for x in gqa if x['image_id']==IMAGE_ID)
    question=next(x for x in entry['questions'] if str(x['question_id'])==QUESTION_ID)
    config={'schema_version':SCHEMA,'fixed_sample':[IMAGE_ID,QUESTION_ID],'seed':1234,'max_new_tokens':16,
      'model_id':pilot.V.MODEL_ID,'model_revision':pilot.V.REVISION,'gpu_gate':gate,'frozen_workload':work,
      'free_bytes_before':free,'minimum_reserve_bytes':RESERVE_BYTES,'maximum_payload_bytes':MAX_PAYLOAD_BYTES,
      'script_sha256':sha(__file__),'source_sha256':{f:sha(ROOT/f) for f in pilot.SOURCE_PATHS},
      'pilot_metrics_overridden':False,'existing_store_reads':0,'setup_policy':'metadata-ready base + optional sidecar from same fresh T1',
      'limitations':['One fixed image diagnostic, not a pilot-mean or deployment benchmark.',
        'Separate run-owned serializer; production pilot persistence remains its measured shared bundle.',
        'Canonical CPU K is retained temporarily across the two setup phases; this is not serving residency.',
        'Base first then sidecar order is fixed; no timing comparison between repeated independent builds.',
        'Hash/readback/activation audit time is excluded from persistence phase wall times and reported separately.',
        'Neither OS write completion nor fsync establishes NAND/internal controller traffic or erase cost.']}
    output.mkdir();write_json_exclusive(output/'config.json',config)
    if args.preflight_only:return
    import torch
    from mmimpress.model import cache_layers
    from mmimpress.serve import Server
    from mmimpress.sparsevlm_ssd_store import CanonicalContext
    payload=output/'payload';payload.mkdir();base=payload/'allhead_base';probe=payload/'probe3_sidecar'
    try:
        pilot.OLD.gpu_inventory();runner=pilot.V.load_runner();pilot.OLD.assert_gpu_exclusive()
        # No synthetic or image-only prefix forward; this is the one real T1.
        result,diag=pilot.normal_request(runner,Server(runner,max_new_tokens=16),ROOT/entry['image_path'],
          lambda:(runner.prompt(question['question']),'',[]),'raster')
        assert result['vision_forward_count']==1
        counts=result['projection_calls']
        assert all(value['prefill']=={'q':1,'k':1,'v':1} for value in counts.values())
        capture=cache_layers(result['captured_past_key_values'])
        phase_t0=time.perf_counter()
        meta=base_metadata(runner,diag['enc_cpu'],diag['hidden'],diag['image_input_sha256'])
        metadata_ms=(time.perf_counter()-phase_t0)*1000
        native_keys,base_time=persist_base(capture,diag['hidden'],meta,base)
        base_time['serialization_phase_wall_ms']=base_time['wall_ms_fsync_inclusive']
        base_time['metadata_preparation_ms']=metadata_ms
        base_time['wall_ms_fsync_inclusive']=(time.perf_counter()-phase_t0)*1000
        probe_time=persist_probe(native_keys,meta,probe)
        combined=(time.perf_counter()-phase_t0)*1000
        readback_t0=time.perf_counter();readback=independent_readback(capture,diag['hidden'],meta,base,probe)
        readback_ms=(time.perf_counter()-readback_t0)*1000
        hash_t0=time.perf_counter();base_files=footprint(base);probe_files=footprint(probe);hash_ms=(time.perf_counter()-hash_t0)*1000
        if base_files['logical_bytes']+probe_files['logical_bytes']>MAX_PAYLOAD_BYTES:raise RuntimeError('BLOCKED_STORAGE: created payload exceeded fixed allowance')
        # Validate actual new reader compatibility, with explicit head policies.
        a=CanonicalContext(base,head_policy='all',expected_hashes=base_files['files_sha256'])
        probe_expected=dict(base_files['files_sha256'],**probe_files['files_sha256'])
        b=CanonicalContext(base,head_policy='fixed_first_3',probe_root=probe,expected_hashes=probe_expected)
        activation={'allhead':a.activation,'probe3':b.activation};a.close();b.close()
        receipt={'schema_version':SCHEMA,'status':'PASS','config_sha256':sha(output/'config.json'),
          'source_T1':{'image_id':IMAGE_ID,'question_id':QUESTION_ID,'prediction':result['answer'],
            'generated_token_ids':result['generated_token_ids'],'vision_forward_count':result['vision_forward_count'],
            'projection_calls':counts,'extra_full_model_forwards':0,'extra_vision_forwards':0,
            'TTFT_ms':result['end_to_end_ttft_ms'],'request_E2E_ms':result['request_e2e_ms'],
            'capture_stats':diag['hidden'].stats(),'image_input_sha256':diag['image_input_sha256']},
          'AllHead_base_setup':base_time,'Probe3_incremental_sidecar_setup':probe_time,
          'fresh_Probe3_setup_total_ms_excluding_T1':combined,
          'fresh_Probe3_setup_phase_sum_ms':base_time['wall_ms_fsync_inclusive']+probe_time['wall_ms_fsync_inclusive'],
          'independent_readback':readback,'readback_ms_excluded':readback_ms,'hash_ms_excluded':hash_ms,
          'AllHead_base_footprint':base_files,'Probe3_incremental_footprint':probe_files,
          'Probe3_independent_deployment_logical_bytes':base_files['logical_bytes']+probe_files['logical_bytes'],
          'base_and_sidecar_shared_physical_files':True,'metadata_activation_after_persistence':activation,
          'comparison_scope':'supplemental_single_fresh_T1_setup_only; does not replace any pilot raw persistence or session measurement',
          'new_payload_paths':[str(base),str(probe)],'limitations':config['limitations']}
        write_json_exclusive(output/'receipt.json',receipt)
        if args.cleanup:cleanup_owned(output,payload,receipt)
        print(json.dumps({'status':'PASS','receipt':str(output/'receipt.json'),
          'AllHead_base_ms':base_time['wall_ms_fsync_inclusive'],'Probe3_incremental_ms':probe_time['wall_ms_fsync_inclusive']}))
    except Exception as exc:
        write_json_exclusive(output/'failure.json',{'status':'FAIL','error':repr(exc),'traceback':traceback.format_exc(),
             'scratch_preserved':True,'previous_artifacts_modified':False});raise
if __name__=='__main__':main()

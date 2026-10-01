#!/usr/bin/env python3
"""Freeze and execute G1-G12 integration, independently of smoke/main."""
from __future__ import annotations
import argparse
import copy
import difflib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('_allhead_main_validation_runner',ROOT/'scripts/100_eval_llava_mt_gqa_allhead_kv25.py')
M=importlib.util.module_from_spec(spec);sys.modules[spec.name]=M;spec.loader.exec_module(M)


def freeze(run):
    ds=M.frozen_dialogues();image_ids=list(dict.fromkeys(d['image_id'] for d in ds));identity={};geometry=[]
    for image in image_ids:
        d=next(x for x in ds if x['image_id']==image);p=ROOT/d['image_path']
        identity[image]={'image_path':d['image_path'],'sha256':M.sha(p)}
        old=ROOT/'runs/mt_gqa_5arm_generated_20260923T064333Z/full/generated_history/images'/f'{image}.json'
        a=json.loads(old.read_text());assert a['source_image_sha256']==identity[image]['sha256']
        stores=a['store_manifests'];r=stores['raster'];n=r['n_spatial'];k=(n+3)//4
        geometry.append({'image_id':image,'N':n,'k':k,'v_num':r['v_token_num'],'boundary_extra_rows':min(((k+63)//64)*64,n)-k,
            'prior_three_store_bytes':sum(stores[x]['total_store_bytes'] for x in ('raster','mpic','image_only')),
            'geometry_provenance':str(old.relative_to(ROOT)),'artifact_sha256':M.sha(old)})
    smoke=[next(d['dialog_id'] for d in ds if d['image_id']==image) for image in image_ids[:4]]
    assert any(g['boundary_extra_rows']>0 for g in geometry[:4])
    manifest={'schema_version':M.SCHEMA,'dataset':'MT-GQA-reconstructed','index_sha256':M.INDEX_SHA,
        'workload_sha256':M.WORKLOAD_SHA,'dialogues':ds,'image_identity':identity,'geometry_prior_provenance':geometry,
        'smoke_dialogue_ids':smoke,'smoke_policy':'first dialogue of first four distinct frozen images, before outputs',
        'expected_requests':{'integration':15,'smoke':60,'main':60915},'methods':list(M.METHODS)}
    largest=max(g['prior_three_store_bytes'] for g in geometry)
    # Existing 398-image metadata provides exact payload geometry, with bounded
    # new JSON metadata expansion. Never infer space from an old free-space value.
    bundle=largest+(64<<20);scratch_margin=4<<30;raw=32<<30;reserve=30<<30
    plan={'policy':'FRESH_IMAGE_STREAMING','free_bytes_at_freeze':shutil.disk_usage(run).free,
        'mount':subprocess.run(['findmnt','-J','-T',str(run)],check=True,capture_output=True,text=True).stdout,
        'largest_image':max(geometry,key=lambda g:g['prior_three_store_bytes']),
        'largest_image_bundle_bound_bytes':bundle,'builder_intermediate_margin_bytes':scratch_margin,
        'raw_checkpoint_growth_bound_bytes':raw,'minimum_reserve_bytes':reserve,
        'peak_working_set_bytes':bundle+scratch_margin,'maximum_new_payload_bytes':16<<30,
        'validation_peak_payload_bound_bytes':10<<30,'maximum_concurrent_main_images':1,
        'conditioning':'successful per-file POSIX_FADV_DONTNEED; no NAND/controller cold claim'}
    plan['required_initial_free_bytes']=reserve+raw+max(plan['peak_working_set_bytes'],plan['validation_peak_payload_bound_bytes'])
    plan['status']='PASS' if plan['free_bytes_at_freeze']>=plan['required_initial_free_bytes'] else 'BLOCKED_STORAGE'
    model=json.loads((ROOT/'runs/sparsevlm_ssd_kv25_20260930T041926Z/config.json').read_text())
    config={'schema_version':M.SCHEMA,'methods':list(M.METHODS),'source_sha256':M.source_hashes(),
        'contract_sha256':M.sha(M.CONTRACT),'model':model,'seed':1234,'batch_size':1,'max_new_tokens':16,
        'history_policy':'method_local_generated','source_T1':'actual full-image inference per method per dialogue',
        'scorer':'strict normalized exact match copied unchanged from 73',
        'budgets':{'ours_kv25':{'budget_unit':'visual_kv','ratio':.25},M.ALLHEAD:{'budget_unit':'visual_kv','ratio':.25,'scoring_head_policy':'all'},'mpic32_ssd':{'k_recompute':32}},
        'provisioning_policy':'FRESH_IMAGE_STREAMING','store_sharing':'FullLoad/AllHead exact captured FP16 bits required per image',
        'persistence_scope':'MEASURED canonical bundle including unused serializer probe sidecar; AllHead base-only NOT SEPARATELY MEASURED',
        'packages':{p:importlib.metadata.version(p) for p in ('torch','transformers','bitsandbytes','numpy','Pillow')},
        'python':sys.version,'platform':platform.platform(),'QWEN_GPU':'NOT RUN','MT_VQA':'NOT RUN',
        'source_tests_sha256':{p.relative_to(ROOT).as_posix():M.sha(p) for p in sorted((ROOT/'tests').glob('test_*.py'))}}
    for name,value in (('storage_plan.json',plan),('manifest.json',manifest),('config.json',config)):
        M.atomic_json(run/name,value)
    shutil.copy2(M.CONTRACT,run/'contract.frozen.md')
    out=ROOT/'results'/run.name;out.mkdir(exist_ok=True)
    shutil.copy2(M.CONTRACT,out/'contract.frozen.md')
    before=json.loads((run/'source_before.json').read_text());diff=[]
    current=list(config['source_sha256'])+['docs/llava_mt_gqa_allhead_kv25_main_contract.md','tests/test_llava_mt_gqa_allhead_main.py']
    for name in current:
        old=(run/'source_before'/name).read_text().splitlines(True) if name in before else []
        new=(ROOT/name).read_text().splitlines(True)
        if old!=new:diff.extend(difflib.unified_diff(old,new,fromfile='before/'+name,tofile='after/'+name))
    (run/'source.diff').write_text(''.join(diff));shutil.copy2(run/'source.diff',out/'source.diff')
    for name in current:
        dst=run/'source_frozen'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst)
    command=f'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 {sys.executable}'
    reproduce=f'''# Reproduction and resume\n\nRun directory: `{run}`. Source/config/manifest changes require a new frozen run; do not edit them while main is active.\n\n```bash\ncd {ROOT}\n{command} scripts/102_validate_llava_mt_gqa_allhead_kv25.py --run-dir {run} --validate\n{command} scripts/100_eval_llava_mt_gqa_allhead_kv25.py --run-dir {run} --phase all --resume\n# After completion, independent re-audit to an unused directory:\n{command} scripts/101_audit_llava_mt_gqa_allhead_kv25.py --run-dir {run} --output-dir {out}_reaudit_01\n```\n\nValidation may be skipped only when integration_validation.json is PASS and its frozen source hashes match. Smoke/main resume skips independently audited atomic image commits only. Incomplete images restart from T1 with new physical IDs; failed attempt/raw/scratch remain. Main requires 60,915 final requests; starting a process is not completion.\n'''
    (run/'REPRODUCE.md').write_text(reproduce);(out/'REPRODUCE.md').write_text(reproduce)
    print(json.dumps({'freeze':'PASS','storage':plan['status'],'free_GiB':plan['free_bytes_at_freeze']/2**30,'required_GiB':plan['required_initial_free_bytes']/2**30,
        'smoke_dialogues':smoke,'largest_image':plan['largest_image']}),flush=True)
    if plan['status']!='PASS':raise RuntimeError('BLOCKED_STORAGE')


def prior_gate():
    path=ROOT/'runs/sparsevlm_ssd_kv25_20260930T041926Z/gpu_full_01/gpu_validation.json'
    g=json.loads(path.read_text());assert g['GPU_CORRECTNESS']=='PASS'
    assert all(g['gates'][f'G{i}']=='PASS' for i in range(1,13))
    assert g['contract_sha256']==M.sha(ROOT/'docs/sparsevlm_ssd_kv25_contract.md')
    assert all(M.sha(ROOT/n)==v for n,v in g['source_sha256'].items())
    return {'status':'PASS','path':str(path),'sha256':M.sha(path),'scope':'unchanged production AllHead 10 fixed GPU cases/G1-G12 reused; fresh wrapper tests additionally required'}


def integration_diagnostic(runner,servers,contexts,paths,hashes,dialog,rows,artifact):
    V=M.V;V._ACTIVE_RUN=artifact.parents[3]
    V._HASH_CACHE[str(paths['raster'].resolve())]=hashes['raster']
    sample_dir=artifact/'gpu_reference';sample_dir.mkdir()
    entry={'image_id':dialog['image_id'],'questions':[{'question_id':t['question_id'],'question':t['question']} for t in dialog['turns']]}
    sample_receipts=[]
    for turn in (2,3):
        source=next(r for r in rows if r['method_id']==M.ALLHEAD and r['turn_id']==turn)
        sample=V.sample(runner,entry,source['question_id'],paths['raster'],sample_dir,
            methods=(M.ALLHEAD,),prompt_text=source['prompt'],tag=f'_mt_t{turn}')
        assert sample['methods'][M.ALLHEAD]['serving_result']['generated_token_ids']==source['result']['generated_token_ids']
        sample_receipts.append({'turn':turn,'status':sample['status'],'reference':sample['methods'][M.ALLHEAD]['matched_memory_reference']})
    # Use identical actual prompts before and after an AllHead request and an
    # injected exception. This verifies vanilla, MPIC and Ours instance recovery.
    original=[l.self_attn.forward for l in runner.layers];before={};after={}
    tested=('recompute','fullload','mpic32_ssd','ours_kv25')
    def operation(method):
        row=next(r for r in rows if r['method_id']==method and r['turn_id']==2)
        factory=lambda:(row['prompt'],row['history'],row['history_entries'])
        if method=='recompute':return M.normal_request(runner,servers['old'],ROOT/dialog['image_path'],factory,'none')[0]
        return M.stored_request(runner,servers,contexts[method],factory,method,dialog['image_id'],dialog['turns'][1]['question'])[0]
    # MPIC's direct selective prefill does not produce a top-level model hook;
    # compare exact generated token IDs and prediction plus every decode logit.
    for method in tested:before[method]=V.captured_call(runner,lambda m=method:operation(m))
    row=next(r for r in rows if r['method_id']==M.ALLHEAD and r['turn_id']==2)
    servers['allhead'].request(contexts[M.ALLHEAD],method_id=M.ALLHEAD,prompt_text=row['prompt'],cold=True)
    def injected(payload):
        if payload['layer']==3:raise RuntimeError('intentional main adapter exception')
    try:servers['allhead'].request(contexts[M.ALLHEAD],method_id=M.ALLHEAD,prompt_text=row['prompt'],observer=injected,cold=True)
    except RuntimeError as exc:assert str(exc)=='intentional main adapter exception'
    else:raise AssertionError('exception fixture did not fire')
    from mmimpress.serve import BIAS
    assert not BIAS and contexts[M.ALLHEAD]._active is None
    assert all(l.self_attn.forward==f for l,f in zip(runner.layers,original))
    recovery={}
    for method in reversed(tested):
        after[method]=V.captured_call(runner,lambda m=method:operation(m))
        assert before[method]['generated_token_ids']==after[method]['generated_token_ids']
        assert before[method]['answer']==after[method]['answer']
        assert len(before[method]['logits'])==len(after[method]['logits'])
        for a,b in zip(before[method]['logits'],after[method]['logits']):assert torch.allclose(a,b,atol=1e-4,rtol=1e-4)
        recovery[method]={'status':'PASS','exact_tokens':True,'matched_logits':len(before[method]['logits']),
            'first_MPIC_logit':'not exposed by model hook; exact first token and independent MPIC fixtures retained' if method=='mpic32_ssd' else 'checked'}
    O=M.helper('90_validate_llava_kv25.py','_main_fresh_mask_reference')
    row=next(r for r in rows if r['method_id']=='ours_kv25' and r['turn_id']==3)
    with O._AttentionTrace(contexts['ours_kv25'].meta,row['k']) as trace:
        actual=operation('ours_kv25')
    assert actual['generated_token_ids']==next(r for r in rows if r['method_id']=='ours_kv25' and r['turn_id']==2)['result']['generated_token_ids']
    M.atomic_json(artifact/'integration_diagnostic.json',{'status':'PASS','AllHead_memory_reference':sample_receipts,
        'cross_method_recovery':recovery,'exception_restoration':'PASS','ours_actual_masks':trace.summary(),
        'diagnostic_requests_excluded_from_main_and_smoke':True})


def negative_audit(run,config,manifest):
    A=M.helper('101_audit_llava_mt_gqa_allhead_kv25.py','_main_negative_audit')
    cp=next((run/'integration/images').glob('*/COMMITTED.json'));commit=json.loads(cp.read_text())
    rows=M.load_jsonl(cp.parent/commit['attempt_id']/'raw.jsonl');image=cp.parent.name
    assert A.audit_image_rows(rows,manifest,config,'integration',image)['status']=='PASS'
    ah=next(i for i,r in enumerate(rows) if r['method_id']==M.ALLHEAD and r['turn_id']==2)
    ours=next(i for i,r in enumerate(rows) if r['method_id']=='ours_kv25' and r['turn_id']==2)
    mutations={
        'wrong_generated_history':lambda rs:rs[ah]['history_entries'][0].update(answer='FORGED'),
        'wrong_method_lineage':lambda rs:rs[ah]['history_entries'][0].update(source_method_id='fullload'),
        'duplicate_logical':lambda rs:rs.append(copy.deepcopy(rs[0])),
        'missing_row':lambda rs:rs.pop(),
        'wrong_allhead_budget':lambda rs:rs[ah].update(k=rs[ah]['k']+1),
        'wrong_projection':lambda rs:rs[ah]['result']['projection_calls']['0']['prefill'].update(q=2),
        'wrong_OS_bytes':lambda rs:rs[ah].update(ssd_read_bytes=rs[ah]['ssd_read_bytes']+1),
        'wrong_mask_count':lambda rs:rs[ours]['result']['keep_count_per_layer'].__setitem__(0,1),
        'wrong_timing':lambda rs:rs[ah].update(ttft_ms=rs[ah]['ttft_ms']/2),
        'wrong_scoring_heads':lambda rs:rs[ah]['result'].update(scoring_head_ids=[0,1,2])}
    report={}
    for name,mutate in mutations.items():
        altered=copy.deepcopy(rows);mutate(altered)
        r=A.audit_image_rows(altered,manifest,config,'integration',image)
        assert r['status']=='FAIL',f'audit accepted {name}'
        report[name]=r['failure_counts']
    recovered=M.completed_images(run,'integration',config,manifest)
    assert list(recovered)==[image] and recovered[image]['requests']==15
    return {'status':'PASS','negative_cases':report,'resume_committed_image_skip':'PASS','partial_image_rejection':'CPU fixture'}


def validate(run):
    config,manifest=M.verify_freeze(run)
    final=run/'integration_validation.json'
    if final.exists():
        r=json.loads(final.read_text())
        if r.get('status')=='PASS' and r.get('source_sha256')==config['source_sha256']:print('integration already PASS');return
        raise FileExistsError('failed validation receipt preserved; use new frozen run or archive pre-main development revision')
    report={'status':'RUNNING','source_sha256':config['source_sha256'],'config_sha256':config['_file_sha256'],
        'gates':{f'G{i}':'NOT RUN' for i in range(1,13)},'QWEN_GPU':'NOT RUN','MT_VQA':'NOT RUN'}
    try:
        report['prior_production_AllHead']=prior_gate()
        cmd=[sys.executable,'-m','unittest','discover','-s','tests','-p','test_*.py']
        with (run/'integration_cpu.log').open('w') as f:p=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError('CPU regression failed')
        report['CPU']={'status':'PASS','command':cmd,'log_sha256':M.sha(run/'integration_cpu.log')}
        report['gates']['G1']='PASS';M.capacity_guard(run);M.assert_gpu_exclusive()
        runner=M.V.load_runner();servers=M.servers_for(runner);M.QA._warmup(runner,servers['old'])
        report['actual_model']={'q_heads':runner.cfg.text_config.num_attention_heads,'kv_heads':runner.cfg.text_config.num_key_value_heads,
            'layers':len(runner.layers),'dtype':str(runner.model.dtype),'attention':runner.attn,
            'load_4bit':runner.load_4bit,'device_map':str(runner.model.hf_device_map)}
        # Original fixed five GPU fixtures and tolerances; no new selection algorithm.
        O=M.helper('90_validate_llava_kv25.py','_main_Ours_GPU_reference')
        entries=json.loads((ROOT/'data/index.json').read_text());samples=[]
        stores=run/'validation_scratch/ours_five_fixed';stores.mkdir(parents=True,exist_ok=False)
        for (image,qid),entry in zip(O.FIXED,entries):
            sample=O._sample(runner,servers['old'],entry,qid,stores);samples.append(sample)
            M.atomic_json(run/'gpu_ours'/f'{image}.json',sample)
            # Standalone fixture has no main rows; its independent bit/mask/read
            # receipt is the durable publication before allowlisted payload cleanup.
            owned=M.file_hashes(stores/image)
            M.atomic_json(run/'gpu_ours'/f'{image}_cleanup_allowlist.json',{'scratch':str(stores/image),'files':owned})
            M.cleanup_committed(stores/image,owned,{'status':sample['status']},{'status':'COMMITTED'})
            M.atomic_json(run/'gpu_ours'/f'{image}_cleanup_receipt.json',{'status':'PASS','files':len(owned)})
            print(json.dumps({'GPU_Ours_fixed':image,'status':'PASS'}),flush=True)
        assert len(samples)==5 and any(x['sentinel_prefill_decode'] for x in samples)
        report['ours_five_fixed']={'status':'PASS','samples':samples,'atol':1e-4,'rtol':1e-4}
        integration=M.run_phase(runner,servers,run,'integration',config,manifest,diagnostic=integration_diagnostic)
        report['fresh_wrapper_integration']=integration
        report['negative_audit_and_resume']=negative_audit(run,config,manifest)
        A=M.helper('101_audit_llava_mt_gqa_allhead_kv25.py','_main_pre_smoke_protection')
        report['protection']=A.protection(run)
        if report['protection']['status']!='PASS':raise RuntimeError('existing artifact protection failed')
        report['gates']={f'G{i}':'PASS' for i in range(1,13)}
        report['gate_evidence']={
            'G1':'exact full index/workload hashes, five-entry registry, pinned model actual config',
            'G2':'original five Ours actual masks/cache + fresh MT AllHead Observer exact ceil(N/4)',
            'G3':'fresh independent OS range audit: all actual heads, no probe/selected-K reread',
            'G4':'actual per-layer prefill/decode projection hooks and fixed selection/read oracle',
            'G5':'AllHead fresh MT independent memory reference; original Ours matched-reference fixtures',
            'G6':'independent real attention mask/positions + sentinel fixtures and decode checks',
            'G7':'fresh cross-order vanilla/FullLoad/MPIC/Ours logits/token recovery and injected exception',
            'G8':'fresh own T1 execution/vision counts and exact shared capture bits, MPIC direct counts',
            'G9':'independent full raw lineage/prompt reconstruction and forged-history negative cases',
            'G10':'actual OS events vs independent ranges and counter sums; AllHead reused K counted once',
            'G11':'atomic commit/adoption and partial-image/link/no-clobber CPU fixtures + actual committed resume verification',
            'G12':'all existing CPU regression suite + new wrapper safety tests + old artifact fingerprint/source SHA256 protection'}
        report['status']='PASS'
    except BaseException as exc:
        report.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        M.atomic_json(final,report)
        print(json.dumps({'integration_status':report['status'],'gates':report['gates'],'error':report.get('error')}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--freeze',action='store_true');p.add_argument('--validate',action='store_true')
    args=p.parse_args();run=args.run_dir.resolve()
    if args.freeze:freeze(run)
    if args.validate:validate(run)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Qwen-only prospective experiment; delegates inference/lifecycle to frozen B runner."""
from pathlib import Path
import argparse, copy, gc, hashlib, importlib.util, json, os, random, shutil, subprocess, sys, time, traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import torch
import main_b_support as S

def load(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

M=load('_prefix3_main','110_eval_qwen_mt_gqa_B.py')
A=load('_prefix3_audit','111_audit_qwen_mt_gqa_B.py')
V=load('_prefix3_reference','86_validate_qwen25_v2_rerun.py')
D=load('_prefix3_legacy_diagnostic','115_validate_qwen_mt_gqa_B.py')
G=load('_prefix3_semantic','116_qwen_prefixstore_semantic.py')
SCHEMA='qwen-mtgqa-prefixstore3-semantic-v2'
M.SCHEMA=A.SCHEMA=SCHEMA
sys.modules['_qwen_commit_auditor']=A;sys.modules['_qwen_resume_auditor']=A
D.M=M;D.A=A

def source_hashes():
    return {**S.source_hashes(ROOT),'scripts/113_report_mt_gqa_B.py':M.sha(ROOT/'scripts/113_report_mt_gqa_B.py')}
M.source_hashes=source_hashes
OLD=ROOT/'runs/mt_gqa_B_llava5_qwen3_20261001T093921Z'
PILOT=ROOT/'runs/prefix25_persistence_20261001T050113Z'

def dump(p,x): M.atomic_json(p,x)
def read(p): return json.loads(Path(p).read_text())
def seed():
    torch.set_num_threads(2);torch.manual_seed(1234);random.seed(1234);np.random.seed(1234)

def prepare(run):
    assert (run/'bootstrap.json').exists()
    # Only executed Qwen raw and explicit core sample lists exclude candidates;
    # never confuse the full workload manifest with previously executed samples.
    observed=set();evidence=[]
    for base in (ROOT/'runs').iterdir():
        if base==run or not base.is_dir():continue
        if 'qwen' not in base.name and 'prefix25' not in base.name:continue
        for p in base.rglob('*raw*.jsonl'):
            rel=str(p.relative_to(base))
            if 'source_' in rel or '/llava/' in str(p):continue
            if base.name.startswith('mt_gqa_B_') and not rel.startswith('qwen/'):continue
            used=set()
            for line in p.open():
                if not line.strip():continue
                row=json.loads(line)
                if row.get('image_id'):used.add(str(row['image_id']))
            observed.update(used);evidence.append(dict(path=str(p),sha256=M.sha(p),image_ids=sorted(used)))
    core=read(PILOT/'correctness_manifest.json')
    observed.update(str(s['image_id']) for s in core['samples'])
    # Older fixed diagnostic manifests also contain executed/selected core samples.
    for base in (ROOT/'runs').glob('qwen*'):
        if base==run:continue
        for name in ('validation_manifest.json','correctness_manifest.json'):
            p=base/name
            if p.exists():
                x=read(p);ss=x.get('samples',[])
                ids={str(s['image_id']) for s in ss if isinstance(s,dict) and 'image_id' in s}
                observed.update(ids);evidence.append(dict(path=str(p),sha256=M.sha(p),image_ids=sorted(ids)))
    manifest=read(OLD/'qwen/manifest.json');ds=manifest['dialogues'];identity=manifest['image_identity']
    observed.update(d['image_id'] for d in ds if d['dialog_id'] in manifest['integration_dialogue_ids'])
    first={}
    for d in ds:first.setdefault(d['image_id'],d)
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(read(OLD/'qwen/config.json')['model']['tokenizer_json'])
    candidates=[]
    for iid,d in first.items():
        if iid in observed:continue
        n=identity[iid]['N_content'];k=(n+3)//4
        candidates.append(dict(image_id=iid,dialog_id=d['dialog_id'],N=n,k=k,remainder=k%64,
            question_tokens=sum(len(tokenizer.encode(t['question']).ids) for t in d['turns']),
            rank=hashlib.sha256(('1234:qwen-extra-v2:'+d['dialog_id']).encode()).hexdigest()))
    candidates.sort(key=lambda x:(x['N'],x['rank']));selected=[]
    for part in np.array_split(np.arange(len(candidates)),4):
        pool=[candidates[int(i)] for i in part]
        a=min(pool,key=lambda x:(min(x['remainder'],64-x['remainder']),x['rank']))
        pool=[x for x in pool if x!=a]
        b=min(pool,key=lambda x:(-x['question_tokens'],x['rank']))
        selected.extend([a,b])
    assert len(selected)==8 and len({x['image_id'] for x in selected})==8
    assert all(x['image_id'] not in observed for x in selected)
    dump(run/'additional_validation_manifest.json',dict(seed=1234,selected_before_outcomes=True,
        rule='Exclude all images in audited prior Qwen raw/core/integration; four N quartiles, one nearest k%64 boundary and one longest question-token sum per quartile; SHA256(seed,dialogue) tie break',
        label='additional Qwen validation; unseen in audited prior Qwen evidence, not a quality-test holdout',
        generated_history_length='not available before generation; question tokens used as prospective proxy',
        excluded_images=sorted(observed),prior_evidence=evidence,candidate_count=len(candidates),samples=selected))
    shutil.copy2(PILOT/'correctness_manifest.json',run/'correctness_manifest.json')
    dump(run/'validation_selection.json',dict(core=[s['image_id'] for s in core['samples']],
        legacy_case='mtgqa_000285',diagnosis_dialogues=['mtgqa_000001','mtgqa_000285'],
        integration=manifest['integration_dialogue_ids'],additional=[x['dialog_id'] for x in selected],
        smoke=manifest['smoke_dialogue_ids'],same_image=manifest['same_image_fixture_dialogue_ids']))
    print(json.dumps({'prepared':str(run),'additional':selected}),flush=True)

def freeze_config(run,base,diagnosis=False):
    M.CONTRACT=run/'CONTRACT.md';base.mkdir(exist_ok=True)
    for name,digest in read(run/'source_before.json').items():assert M.sha(ROOT/name)==digest,('existing source changed',name)
    prior=read(PILOT/'final_gpu_freeze.json')['files']
    for name,digest in prior.items():assert M.sha(ROOT/name)==digest,('final B source differs',name)
    cfg=read(OLD/'qwen/config.json');manifest=read(OLD/'qwen/manifest.json')
    if diagnosis:manifest['integration_dialogue_ids']=read(run/'validation_selection.json')['diagnosis_dialogues']
    manifest['schema_version']=SCHEMA
    assert M.sha(ROOT/'data/mt_gqa/dialogues.json')==M.INDEX_SHA
    from mmimpress.mt_gqa import workload_sha256
    assert workload_sha256(manifest['dialogues'])==M.WORKLOAD_SHA
    assert read(ROOT/'data/mt_gqa/dialogues.json')['dialogues']==manifest['dialogues']
    for iid,x in manifest['image_identity'].items():assert M.sha(ROOT/x['image_path'])==x['sha256']
    for name,digest in cfg['model']['config_file_sha256'].items():assert M.sha(Path(cfg['model']['snapshot'])/name)==digest
    for name,x in cfg['model']['weight_files'].items():
        p=Path(cfg['model']['snapshot'])/name;assert str(p.resolve())==x['resolved'] and p.stat().st_size==x['bytes']
    plan=dict(minimum_reserve_bytes=30<<30,peak_working_set_bytes=4<<30,largest_image_bundle_bound_bytes=1<<30,
        maximum_new_payload_bytes=20<<30,raw_growth_bound_bytes=10<<30,validation_growth_bound_bytes=24<<30,
        failed_payload_bound_bytes=10<<30,required_initial_free_bytes=78<<30,free_bytes_at_freeze=shutil.disk_usage(run).free,
        mount=subprocess.check_output(['findmnt','-J','-T',str(run)],text=True),
        scratch_cleanup='only run-owned exact allowlist after independent image audit, hashes, atomic commit')
    assert plan['free_bytes_at_freeze']>=plan['required_initial_free_bytes']
    dump(base/'storage_plan.json',plan);dump(base/'manifest.json',manifest)
    cfg.update(schema_version=SCHEMA,experiment_version=2,source_sha256=M.source_hashes(),
        contract_sha256=M.sha(M.CONTRACT),storage_plan_sha256=M.sha(base/'storage_plan.json'),
        manifest_sha256=M.sha(base/'manifest.json'),semantic_contract='FP64 1e-12/1e-12 all 28 layers; exact native BF16 implementation; legacy FP32 retained diagnostic',
        core_model_store_revision=M.Qwen25Runner()._code_revision(),diagnostic_only=diagnosis)
    dump(base/'config.json',cfg)
    for n in cfg['source_sha256']:
        p=base/'source_frozen'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    shutil.copy2(M.CONTRACT,base/'contract.frozen.md')
    dump(base/'execution_manifest.json',[dict(image_id=d['image_id'],dialog_id=d['dialog_id'],canonical_ordinal=i,
        method_order=list(M.method_order(i)),turn_order=[1,2,3]) for i,d in enumerate(manifest['dialogues'])])
    dump(base/'final_source_freeze.json',dict(files=cfg['source_sha256'],contract_sha256=cfg['contract_sha256'],
        config_sha256=M.sha(base/'config.json'),manifest_sha256=M.sha(base/'manifest.json'),
        additional_manifest_sha256=M.sha(run/'additional_validation_manifest.json'),production_changed=False,
        entrypoint='scripts/117_run_qwen_prefixstore3.py; delegates 110 unchanged'))
    return M.verify_freeze(base)

def reject(call):
    try:call()
    except (AssertionError,ValueError,RuntimeError,FileNotFoundError) as exc:return {'rejected':True,'reason':str(exc)}
    raise AssertionError('negative fixture was accepted')

def actual_negative(runner,cap,store,path,pos,logical,reference,record,artifact):
    expected_mask=G.visible_mask(logical,len(cap.prefix_ids),record['q'].shape[2])
    baseline=dict(logical=logical,expected_logical=logical,positions=pos,expected_positions=pos,
        mask=expected_mask,expected_mask=expected_mask,layers=reference,expected_layers=reference)
    G.validate_structure(**baseline);out={}
    wrong=logical.copy();wrong[len(wrong)//2]=next(i for i in range(len(cap.prefix_ids)) if i not in logical)
    out['wrong_selected_ID']=reject(lambda:G.validate_structure(**{**baseline,'logical':wrong,'layers':G.gather_canonical(cap,wrong)}))
    dup=logical.copy();dup[0]=dup[1]
    out['missing_duplicated_structural_row_same_shape']=reject(lambda:G.validate_structure(**{**baseline,'logical':dup,'layers':G.gather_canonical(cap,dup)}))
    badpos=pos.clone();badpos[0,0,-1]+=1
    out['wrong_MRoPE_position']=reject(lambda:G.validate_structure(**{**baseline,'positions':badpos}))
    bm=expected_mask.clone();bm[0,-1]=True
    out['future_suffix_visibility']=reject(lambda:G.validate_structure(**{**baseline,'mask':bm}))
    # Demonstrate an actual changed attention result, beyond shape detection.
    ref=G.attention(record['q'],record['k'],record['v'],expected_mask,torch.float64)['output']
    bad=G.attention(record['q'],record['k'],record['v'],bm,torch.float64)['output']
    out['future_suffix_visibility']['output_changed']=not torch.equal(ref,bad)
    assert out['future_suffix_visibility']['output_changed']
    badkv=[(k.clone(),v.clone()) for k,v in reference];badkv[0][1][0,0,0,0]+=1
    out['wrong_KV_bits_same_shape']=reject(lambda:G.validate_structure(**{**baseline,'layers':badkv}))
    out['foreign_image_identity']=reject(lambda:runner._activate(path,'0'*64))
    k=(cap.visual_count+3)//4
    out['double_applied_budget']=reject(lambda:G.validate_budget(cap.visual_count,k,(k+3)//4,list(range((k+3)//4))))
    assert out['double_applied_budget']['rejected']
    from mmimpress.qwen25.store import open_qwen_store
    fixtures=artifact/'negative_payloads';fixtures.mkdir(exist_ok=False)
    for kind in ('truncated','unsealed','corrupted'):
        dest=fixtures/kind;shutil.copytree(path,dest)
        payload=next(dest.glob('layer_*/k.bin'))
        if kind=='unsealed':(dest/'meta.json').unlink()  # Native durable publication marker.
        elif kind=='truncated':
            with payload.open('r+b') as f:f.truncate(payload.stat().st_size-2)
        else:
            with payload.open('r+b') as f:
                b=f.read(1);f.seek(0);f.write(bytes([b[0]^1]))
        out[kind+'_payload']=reject(lambda:open_qwen_store(dest,verify=True))
    return out

def diagnostic(runner,dialog,turn,method,row,result,captures,paths,metas,artifact):
    if method=='recompute':return
    if method=='fullload' or turn in (1,3):D.diagnostic(runner,dialog,turn,method,row,result,captures,paths,metas,artifact)
    if method!=M.METHODS[2] or turn==1:return
    cap=captures[method];rank,chosen,structural,logical=G.independent_indices(cap)
    reference=G.gather_canonical(cap,logical);store=runner._activate(paths[method],row['image_sha256'])
    loaded=store.load_prefix(.25,budget_unit='visual_kv');full=torch.tensor(row['input_ids']);prefix=len(cap.prefix_ids);suffix=full[:,prefix:]
    pos,delta=V._stock_positions(runner,full,torch.tensor(cap.image_grid_thw))
    assert pos.tolist()==row['request_trace']['position_ids'] and delta.tolist()==row['request_trace']['rope_deltas']
    G.validate_budget(cap.visual_count,store.meta['stored_content_count'],loaded.kept_visual_tokens,list(loaded.selected_visual_stored))
    assert torch.equal(loaded.position_ids,pos[:,:,torch.tensor(logical)])
    with G.CaptureSDPA() as prod:
        actual=runner.run_cache(paths[method],row['question'],history=tuple(map(tuple,row['history'])),
            budget_ratio=.25,budget_unit='visual_kv',image_sha256=row['image_sha256'],return_logits=True)
    with G.CaptureSDPA() as mem:
        memory=V.trace_suffix(runner,reference,suffix,pos,prefix)
    with G.CaptureSDPA() as repeated:
        again=runner.run_cache(paths[method],row['question'],history=tuple(map(tuple,row['history'])),
            budget_ratio=.25,budget_unit='visual_kv',image_sha256=row['image_sha256'],return_logits=True)
    comps={name:V.path_comparison(memory,path) for name,path in [('production',V._result_path(actual)),('repeat',V._result_path(again)),('main_wrapper',V._result_path(result))]}
    assert all(x['exact_target'] for x in comps.values()),comps
    assert G.tensor_hash(memory['logits'])==G.tensor_hash(actual['first_logits'])==G.tensor_hash(again['first_logits'])==G.tensor_hash(result['first_logits'])
    assert len(prod.records)==len(mem.records)==len(repeated.records)==28
    expected=G.visible_mask(logical,prefix,suffix.shape[1]);rm=prod.records[0]['mask'];rm=rm[0,0] if rm.dtype==torch.bool else rm[0,0]==0
    G.validate_structure(logical=list(loaded.logical_indices),expected_logical=logical,positions=loaded.position_ids,
        expected_positions=pos[:,:,torch.tensor(logical)],mask=rm,expected_mask=expected,layers=loaded.layers,expected_layers=reference)
    for a,b,c in zip(prod.records,mem.records,repeated.records):
        assert all(torch.equal(a[key],b[key]) and torch.equal(a[key],c[key]) for key in ('q','k','v','mask'))
    target=artifact/f'semantic_{dialog["dialog_id"]}_T{turn}';target.mkdir()
    fixture=dict(production=prod.records,canonical_layers=[(k.cpu(),v.cpu()) for k,v in cap.layers],
        scores=cap.scores.cpu(),rank=rank,selected_original=chosen,selected_stored=list(range(len(chosen))),logical_indices=logical,
        structural_indices=structural,positions=pos,prefix_positions=torch.tensor(cap.logical_position_ids),
        full_input_ids=full,suffix_ids=suffix,prefix_length=prefix,history=row['history'],history_entries=row['history_entries'],
        source=row['source_store'],environment=G.numerical_environment())
    torch.save(fixture,target/'fixed_inputs.pt')
    layers=[];layer0=None
    for li,(rec,pair) in enumerate(zip(prod.records,cap.layers)):
        evidence,outputs=G.semantic_layer(rec,pair,logical,prefix)
        layers.append(dict(layer=li,**evidence))
        if li==0:layer0=outputs;torch.save(outputs,target/'layer0_outputs.pt')
    negatives=actual_negative(runner,cap,store,paths[method],pos,logical,reference,prod.records[0],target) if turn==2 else {}
    # Compare hashes against the archived original FP32/FP64 fixture; no copied PASS.
    historical={}
    old=ROOT/'runs/qwen_mt_gqa_kv25_main_20261001T031459Z/oracle_failure_diagnostic_retry1/oracle_fixture.pt'
    if dialog['dialog_id']=='mtgqa_000285' and turn==2 and old.exists():
        x=torch.load(old,map_location='cpu',weights_only=False);rec=prod.records[0];p=len(logical)
        group=rec['k'].shape[1]//4
        kn=rec['k'][:,::group];vn=rec['v'][:,::group]
        for name,a,b in [('q',rec['q'],x['q']),('k_suffix',kn[:,:,p:],x['k_suffix']),
                         ('v_suffix',vn[:,:,p:].transpose(1,2).reshape(1,suffix.shape[1],-1),x['v_suffix']),
                         ('prefix_K',reference[0][0],x['prefix_layer0'][0]),('prefix_V',reference[0][1],x['prefix_layer0'][1])]:
            historical[name]=dict(current_sha256=G.tensor_hash(a),historical_sha256=G.tensor_hash(b),same_bits=torch.equal(a,b))
        historical['artifact_sha256']=M.sha(old)
    legacy='PASS' if layers[0]['legacy_fp32']['allclose'] else 'FAIL'
    report=dict(status='PASS',legacy_G7=legacy,legacy_scope='layer 0; original FP32 atol=rtol=1e-5',
        semantic_scope='all 28 layers/all heads/all suffix queries/output dimensions; fixed production Q and suffix KV; canonical prefix; FP64 before operations',
        matched=comps,all_layer_QKV_mask_exact=True,layers=layers,negative_fixtures=negatives,
        historical_input_hash_comparison=historical,N=cap.visual_count,k=len(chosen),S=len(structural),
        compact_prefix=len(logical),suffix_length=suffix.shape[1],history=row['history'],
        fixture_sha256=M.sha(target/'fixed_inputs.pt'),output_sha256=M.sha(target/'layer0_outputs.pt'))
    dump(target/'validation.json',report)
    print(json.dumps({'semantic':dialog['dialog_id'],'turn':turn,'status':'PASS','legacy_G7':legacy,
        'fp32_max':layers[0]['legacy_fp32']['max_abs'],'fp64_max':max(x['semantic_fp64']['max_abs'] for x in layers)}),flush=True)
    del fixture,memory,prod,mem,repeated,actual,again,reference,loaded;gc.collect()

def run_validation(run,base,diagnosis=False):
    cfg,manifest=M.verify_freeze(base);runner=None
    report=dict(status='RUNNING',source_sha256=cfg['source_sha256'],legacy_G7='FAIL preserved historically')
    try:
        seed();runner=M.load_runner();dump(base/'integration_runtime.json',runner.runtime_fingerprint())
        report['main_integration']=M.run_phase(runner,base,'integration',cfg,manifest,diagnostic=diagnostic)
        if not diagnosis:
            report['negative_controls']=D.negative_controls(base,cfg,manifest)
            for label,ids in [('additional',read(run/'validation_selection.json')['additional']),('same_image_fixture',manifest['same_image_fixture_dialogue_ids'])]:
                sub=base/label;sub.mkdir();shutil.copy2(base/'storage_plan.json',sub/'storage_plan.json')
                adapted=copy.deepcopy(manifest);adapted['integration_dialogue_ids']=ids
                report[label]=M.run_phase(runner,sub,'integration',cfg,adapted,diagnostic=diagnostic if label=='additional' else None)
                dump(sub/'manifest.json',adapted)
        report['status']='PASS'
    except BaseException as exc:
        report.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        if runner:runner.close()
        dump(base/'integration_validation.json',report)

def cpu(run):
    with (run/'cpu_regression.log').open('x') as f:
        p=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_*.py'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    dump(run/'cpu_regression.json',dict(status='PASS' if p.returncode==0 else 'FAIL',returncode=p.returncode,log_sha256=M.sha(run/'cpu_regression.log')))
    assert p.returncode==0

def core(run):
    cfg,_=M.verify_freeze(run);assert read(run/'cpu_regression.json')['status']=='PASS'
    E=load('_prefix3_core','106_prefix25_persistence.py');seed();E.exclusive();exp=E.Experiment(run,'qwen');seed()
    out=dict(status='RUNNING',source_sha256=cfg['source_sha256'],samples=[])
    try:
        for i,sample in enumerate(read(run/'correctness_manifest.json')['samples']):
            path=run/'core_gpu'/f'{i:02d}_{sample["image_id"]}';path.mkdir(parents=True)
            evidence=exp.qwen_case(sample,path);dump(path/'validation.json',evidence);out['samples'].append(dict(image_id=sample['image_id'],status='PASS',evidence=evidence))
            print('frozen core',i+1,'/10 PASS',flush=True)
        out['status']='PASS'
    except BaseException as exc:out.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc());raise
    finally:exp.close_contexts();dump(run/'core_gpu_validation.json',out)

def resume_command(run):
    return f'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:scripts:tests OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/117_run_qwen_prefixstore3.py --run {run} --stage main --resume'

M.resume_command=resume_command

def update_status(base,state,phase,count,images,error=None):
    root=ACTIVE_ROOT
    doc=dict(execution_state=state,phase=phase,phase_completed_requests=count,phase_completed_images=images,
        expected_main_requests=36549,at_unix=time.time(),pid=os.getpid(),error=error,
        main_final_requests=sum(read(cp)['requests'] for cp in (root/'main/images').glob('*/COMMITTED.json')),
        resume_command=resume_command(root))
    M.atomic_json(base/'CURRENT_STATUS.json',doc,replace=True)
    out=ROOT/'results'/root.name/base.relative_to(root);out.mkdir(parents=True,exist_ok=True)
    M.atomic_json(out/'CURRENT_STATUS.json',doc,replace=True)
    print(json.dumps(doc),flush=True)
M.update_status=update_status

def review(run):
    cfg,_=M.verify_freeze(run)
    for name in ('cpu_regression.json','core_gpu_validation.json','integration_validation.json'):
        assert read(run/name)['status']=='PASS',name
    reports=[]
    for base in (run/'integration',run/'additional/integration'):
        for path in sorted(base.glob('images/*/attempt_*/semantic_*/validation.json')):
            x=read(path);assert x['status']=='PASS' and x['all_layer_QKV_mask_exact']
            assert len(x['layers'])==28 and all(z['semantic_fp64']['allclose'] for z in x['layers'])
            assert all(z['exact_target'] for z in x['matched'].values())
            assert M.sha(path.parent/'fixed_inputs.pt')==x['fixture_sha256']
            if '_T2' in path.parent.name:assert len(x['negative_fixtures'])==10 and all(v['rejected'] for v in x['negative_fixtures'].values())
            reports.append(dict(path=str(path),sha256=M.sha(path),legacy=x['legacy_G7'],semantic='PASS'))
    assert len(reports)==24
    case=next(p for p in (run/'integration').glob('images/n313060/attempt_*/semantic_mtgqa_000285_T2/validation.json'))
    x=read(case);assert x['legacy_G7']=='FAIL'
    assert all(v['same_bits'] for k,v in x['historical_input_hash_comparison'].items() if isinstance(v,dict))
    stage=x['layers'][0]['stages'];assert stage['visible_dots']['max_abs']>0
    assert stage['output_compact_using_dense_logits_vs_dense']['max_abs']<x['layers'][0]['legacy_fp32']['max_abs']
    # This ratio is descriptive root-cause support, not a numeric correctness tolerance.
    dump(run/'G7_review_verdict.json',dict(status='PASS',legacy_G7='FAIL',root_cause='② numerical validation: CPU FP32 operation-shape-dependent rounding',
        production_changed=False,source_sha256=cfg['source_sha256'],evidence=reports,contract_sha256=cfg['contract_sha256'],
        additional_validation_samples=8,diagnostic_used_samples_not_holdout=['mtgqa_000001','mtgqa_000285']))
    print('Final semantic/implementation admission PASS; legacy G7 FAIL retained',flush=True)

def main_run(run,resume):
    cfg,manifest=M.verify_freeze(run)
    for name in ('cpu_regression.json','core_gpu_validation.json','integration_validation.json'):
        assert read(run/name)['status']=='PASS',name
    assert read(run/'G7_review_verdict.json')['status']=='PASS'
    assert read(run/'integration_validation.json')['source_sha256']==cfg['source_sha256']
    seed();runner=None
    try:
        M.capacity_guard(run);runner=M.load_runner();fp=runner.runtime_fingerprint()
        assert fp==read(run/'integration_runtime.json')
        dump(run/('main_runtime_'+time.strftime('%Y%m%dT%H%M%S')+'.json'),fp)
        for phase in ('smoke','main'):
            if phase=='main':assert read(run/'smoke/validation.json')['requests']==36
            if (run/phase).exists() and not resume:raise FileExistsError('resume flag required')
            M.run_phase(runner,run,phase,cfg,manifest)
        M.update_status(run,'AWAITING_INDEPENDENT_FINAL_AUDIT','main',36549,398)
    except BaseException as exc:
        M.update_status(run,'INTERRUPTED',locals().get('phase','startup'),0,0,repr(exc))
        dump(run/('interruption_'+time.strftime('%Y%m%dT%H%M%S')+'.json'),dict(status='PARTIAL',error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        if runner:runner.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','diagnose','freeze','cpu','core','integration','review','main'],required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args();run=a.run.resolve();M.CONTRACT=run/'CONTRACT.md';seed()
    global ACTIVE_ROOT
    ACTIVE_ROOT=run
    if a.stage=='prepare':prepare(run)
    elif a.stage=='diagnose':
        base=run/f'diagnosis_v{len(list(run.glob("diagnosis_v*")))+1}'
        freeze_config(run,base,True);run_validation(run,base,True)
    elif a.stage=='freeze':freeze_config(run,run)
    elif a.stage=='cpu':cpu(run)
    elif a.stage=='core':core(run)
    elif a.stage=='integration':
        assert read(run/'core_gpu_validation.json')['status']=='PASS';run_validation(run,run)
    elif a.stage=='review':review(run)
    elif a.stage=='main':main_run(run,a.resume)

if __name__=='__main__':main()

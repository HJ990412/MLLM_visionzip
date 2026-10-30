"""Dataset/scoring/lifecycle adapter over the validated production request paths.

No selector, model forward, KV layout, native precision, or prompt is reimplemented.
Only experiment identities, annotation-preserving scoring, audit plumbing and
lossless archival differ. Existing shared sources remain byte-for-byte intact.
"""
from pathlib import Path
import copy,gzip,hashlib,importlib.util,json,os,sys,time
from types import SimpleNamespace
from .common import ROOT,read,dump,sha,canonical
from .scorer import reference_scores,independent_score,REVISION

def load(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def dialogues_for_base(dialogues):
    result=copy.deepcopy(dialogues)
    for d in result:
        d['image_id']=str(d['image_id'])
        for t in d['turns']:
            t['question_id']=str(t['question_id'])
            t['answers']=[a['answer'] for a in t['annotation']['answers']]
    return result

def score_from_strings(pred,gold,**kwargs):
    ann={'question_id':1,'answer_type':'diagnostic','question_type':'diagnostic',
         'answers':[{'answer_id':i+1,'answer':a} for i,a in enumerate(gold)]}
    return independent_score(pred,ann)['vqa_score']

def enrich(row,turn,tokenizer,model):
    result=row.get('result',row);ids=result['generated_token_ids'];annotation=turn['annotation']
    row.update(dataset='MT-VQA-v2-reconstructed-25k',scorer_revision=REVISION,annotation=annotation,
        generated_token_ids=list(ids),native_raw_decoded_output=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),
        scorer_answer=row['prediction'],quality_metric='official VQA accuracy',legacy_score_not_primary=True)
    # The production decoder determines answer/history. Never truncate lexical words.
    if model!='glm':
        native=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
        if native!=row['prediction']:raise ValueError('production native answer differs from independent token decode')
        row['native_history_answer']=row['prediction']
    else:
        from mmimpress.glm46v.runner import native_final_answer
        native,info=native_final_answer(tokenizer,ids)
        if native!=row['prediction']:raise ValueError('GLM native boundary extraction mismatch')
        row['native_history_answer']=row['history_text']
    score=reference_scores([annotation],{int(annotation['question_id']):row['prediction']})[0][int(annotation['question_id'])]
    independent=independent_score(row['prediction'],annotation)
    if abs(score-independent['vqa_score'])>1e-10:raise ValueError('official and independent score disagree')
    row.update(score=score,**independent)
    row['empty_answer']=not row['prediction']
    eos=tokenizer.eos_token_id
    eos=set(eos if isinstance(eos,(tuple,list)) else [eos])
    if model=='glm':eos=set(read(ROOT/'runs/glm46v_prefixstore3_20261007T065207Z/checkpoint/generation_config.json')['eos_token_id'])
    row['eos_only']=len(ids)==1 and ids[0] in eos
    return row

def read_rows(path):
    path=Path(path)
    if not path.exists() and path.with_suffix(path.suffix+'.gz').exists():path=path.with_suffix(path.suffix+'.gz')
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt') as f:return [json.loads(line) for line in f if line.strip()]

def archive_before_commit(parent,commit,writer):
    """Audit is already complete. Bind compressed evidence before atomic adoption.

    Original journal remains until contexts close. Its bytes are preserved in
    gzip, verified by full decompression, and uncompressed SHA remains in receipt.
    """
    folder=parent/commit['attempt_id'];archives={}
    for source in sorted(folder.glob('*.jsonl')):
        target=source.with_suffix(source.suffix+'.gz')
        with source.open('rb') as inp,target.open('xb') as f:
            with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0,compresslevel=6) as out:
                for block in iter(lambda:inp.read(1<<20),b''):out.write(block)
            f.flush();os.fsync(f.fileno())
        h=hashlib.sha256()
        with gzip.open(target,'rb') as f:
            for b in iter(lambda:f.read(1<<20),b''):h.update(b)
        if h.hexdigest()!=sha(source):raise ValueError('lossless raw archive mismatch')
        archives[source.name]={'path':target.name,'encoding':'gzip','uncompressed_sha256':h.hexdigest(),
            'sha256':sha(target),'uncompressed_bytes':source.stat().st_size,'bytes':target.stat().st_size}
    commit['journal_archives']=archives
    for name,a in archives.items():
        commit['evidence_sha256'].pop(name,None);commit['evidence_sha256'][a['path']]=a['sha256']
    writer(parent/'COMMITTED.json',commit)

def remove_archived_duplicates(parent):
    commit=read(parent/'COMMITTED.json');folder=parent/commit['attempt_id'];removed=[]
    for name,a in commit.get('journal_archives',{}).items():
        p=folder/name;compressed=folder/a['path']
        if p.exists():
            if p.is_symlink() or p.stat().st_nlink!=1 or sha(p)!=a['uncompressed_sha256'] or sha(compressed)!=a['sha256']:raise ValueError('journal cleanup ownership/hash mismatch')
            p.unlink();removed.append(name)
    receipt=folder/'lossless_archive_receipt.json'
    if not receipt.exists():dump(receipt,{'status':'PASS','removed_uncompressed_duplicates':removed,'raw_preserved':'gzip with uncompressed and compressed full SHA256 bound by commit'})

def cleanup_owned(scratch,allowlist,audit,commit,experiment):
    """Same fail-closed checks as the base, bound to this exact new experiment."""
    scratch=Path(scratch);experiment=Path(experiment).resolve()
    if audit.get('status')!='PASS' or commit.get('status')!='COMMITTED':raise ValueError('independent audit and atomic commit required')
    if experiment.parent!=ROOT/'runs' or not experiment.name.startswith('mt_vqa_v2_reconstructed_25k_'):raise ValueError('not a new experiment root')
    if scratch.is_symlink() or not scratch.is_dir() or not scratch.resolve().is_relative_to(experiment):raise ValueError('outside owned experiment')
    relative=scratch.relative_to(experiment)
    if len(relative.parts)!=6 or relative.parts[0] not in ('llava','qwen') or relative.parts[2]!='scratch' or not relative.parts[-1].startswith('attempt_'):raise ValueError('not an exact model/phase/scratch/phase/image/attempt path')
    nodes=list(scratch.rglob('*'))
    if any(p.is_symlink() for p in nodes):raise ValueError('linked payload/directory')
    files={str(p.relative_to(scratch)):p for p in nodes if p.is_file()}
    if set(files)!=set(allowlist):raise ValueError('cleanup allowlist mismatch')
    for name,p in files.items():
        if p.stat().st_nlink!=1 or not p.resolve().is_relative_to(scratch.resolve()) or sha(p)!=allowlist[name]:raise ValueError('linked or changed payload')
    for p in files.values():p.unlink()
    for p in sorted((p for p in nodes if p.is_dir()),key=lambda p:len(p.parts),reverse=True):p.rmdir()
    scratch.rmdir()
    fd=os.open(scratch.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def qwen_outer_activation(runner):
    original=runner._activate
    def activate(path,image_sha256=None):
        key=str(Path(path).resolve());fresh=key not in runner._stores;started=time.perf_counter()
        value=original(path,image_sha256);finished=time.perf_counter()
        if fresh:
            record=runner.activation_records[key]
            record['inner_activation_core_ms']=record['activation_ms']
            record['activation_ms']=(finished-started)*1000
            record['scope']='caller before _activate through full return, including expected identity/metadata/FD preparation'
        return value
    runner._activate=activate
    return runner

def bind(model,runner=None):
    if model not in ('llava','qwen'):raise ValueError(model)
    M=load('_mtvqa25k_'+model,{'llava':'108_eval_llava_mt_gqa_B.py','qwen':'110_eval_qwen_mt_gqa_B.py'}[model])
    A=load('_mtvqa25k_audit_'+model,{'llava':'109_audit_llava_mt_gqa_B.py','qwen':'111_audit_qwen_mt_gqa_B.py'}[model])
    M.SCHEMA=A.SCHEMA=f'mtvqa25k-{model}-v1'
    def experiment_id(phasepath):return Path(phasepath).parents[1].name
    if model=='llava':
        # Base receives phasepath.name; prefix with the parent experiment later
        # at the driver boundary via M._experiment_id, before any source event.
        M.request_id=lambda run_id,phase,method,did,t:f'{M.SCHEMA}:{M._experiment_id}:{phase}:{M.MODEL_ID}:{method}:{did}:t{t}'
    else:
        M.logical_id=lambda run,phase,method,did,t:f'{M.SCHEMA}:{experiment_id(run)}:{phase}:{M.MODEL_ID}:{method}:{did}:t{t}'
    M.load_jsonl=M.read_rows=read_rows
    original_helper=M.helper
    def helper(*args):
        if any(str(x) in ('109_audit_llava_mt_gqa_B.py','111_audit_qwen_mt_gqa_B.py') for x in args):return A
        return original_helper(*args)
    M.helper=helper
    if model=='llava':A.A.independent_score=score_from_strings
    else:A.score=score_from_strings
    old_audit=A.audit_image_rows
    def audit(rows,manifest,config,phase,image_id,artifact=None):
        report=old_audit(rows,manifest,config,phase,image_id,artifact)
        failures=[];byqid={str(t['question_id']):t for d in manifest['dialogues'] for t in d['turns']}
        for r in rows:
            t=byqid[r['question_id']]
            if r.get('annotation')!=t['annotation']:failures.append('official_annotation')
            if r.get('scorer_answer')!=r['prediction']:failures.append('native_scorer_answer')
            if abs(independent_score(r['prediction'],t['annotation'])['vqa_score']-r['score'])>1e-10:failures.append('official_score')
            if r.get('generated_token_ids')!=r['result']['generated_token_ids']:failures.append('generated_IDs')
        report['official_annotation_score_audit']={'status':'FAIL' if failures else 'PASS','failures':failures}
        if failures:report['status']='FAIL'
        return report
    A.audit_image_rows=audit
    old_make=M.make_row
    if model=='llava':
        def make(runner,result,diag,meta,**kwargs):
            row=old_make(runner,result,diag,meta,**kwargs)
            row['experiment_id']=experiment_id(kwargs['run'])
            return enrich(row,kwargs['dialog']['turns'][kwargs['turn']-1],runner.processor.tokenizer,model)
        # Same writer calls and options, with the new dataset metadata identity.
        def native_persist(runner,result,diag,path,image_id,kind,source):
            from mmimpress.piggyback import persist_captured_raster_prefix,persist_captured_visual_prefix
            from mmimpress.mpic import persist_captured_mpic_prefix
            enc=diag['enc_cpu'];args=(runner,result['captured_past_key_values'],enc['input_ids'],enc['image_sizes'][0])
            common=dict(image_id=image_id,model_id=M.MODEL_ID,chunk_size=64,image_input_sha256=diag['image_input_sha256'],
                extra_metadata={'dataset':'MT-VQA-v2-reconstructed-25k','source_dialog_id':source['dialog_id'],'source_turn_id':1,
                    'source_method_id':source['method_id'],'source_execution_id':source['physical_execution_id'],
                    'source_logical_request_id':source['logical_request_id'],'run_schema':M.SCHEMA})
            if kind=='mpic':return persist_captured_mpic_prefix(*args,diag['hidden'].result_cpu(),path,hidden_capture_stats=diag['hidden'],**common)
            if kind=='raster':return persist_captured_raster_prefix(*args,diag['hidden'].result_cpu(),path,probe_heads=3,hidden_capture_stats=diag['hidden'],full_integrity_hash=False,**common)
            return persist_captured_visual_prefix(*args,diag['vision'].result_cpu(),path,capture_stats=diag['vision'],full_integrity_hash=False,storage_policy='prefix25',**common)
        M._native_persist=native_persist
    else:
        def make(result,diag,method,dialog,turn,*args,**kwargs):
            row=old_make(result,diag,method,dialog,turn,*args,**kwargs)
            row['experiment_id']=M._experiment_id
            return enrich(row,dialog['turns'][turn-1],M._active_runner.processor.tokenizer,model)
        old_execute=M.execute_request
        def execute(*args,**kwargs):
            result,diag=old_execute(*args,**kwargs)
            # Common outer runner-return endpoint for every arm/capture mode.
            result['request_e2e_ms']=1000*(result['core_returned_at_s']-result['request_started_at_s'])
            result['request_finished_at_s']=result['core_returned_at_s']
            return result,diag
        M.execute_request=execute
    M.make_row=make
    M.cleanup_committed=lambda scratch,allowlist,audit,commit,*args:cleanup_owned(scratch,allowlist,audit,commit,ROOT/'runs'/M._experiment_id)
    if model=='qwen':
        original_load=M.load_runner
        M.load_runner=lambda:qwen_outer_activation(original_load())
    original_atomic=M.atomic_json
    def atomic(path,value,*args,**kwargs):
        path=Path(path)
        if path.name=='COMMITTED.json':return archive_before_commit(path.parent,value,original_atomic)
        return original_atomic(path,value,*args,**kwargs)
    M.atomic_json=atomic
    return M,A

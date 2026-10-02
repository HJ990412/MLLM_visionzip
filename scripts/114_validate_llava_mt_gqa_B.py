"""Fresh main integration controls, inherited numerical criteria."""
from pathlib import Path
import sys,copy,json,torch,importlib.util
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def load(n,f):
 s=importlib.util.spec_from_file_location(n,ROOT/'scripts'/f);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
M=load('_main_B_L','108_eval_llava_mt_gqa_B.py')
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
    tested=('recompute','fullload','mpic32_ssd','ours_prefixstore_kv25')
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
    row=next(r for r in rows if r['method_id']=='ours_prefixstore_kv25' and r['turn_id']==3)
    with O._AttentionTrace(contexts['ours_prefixstore_kv25'].meta,row['k']) as trace:
        actual=operation('ours_prefixstore_kv25')
    assert actual['generated_token_ids']==next(r for r in rows if r['method_id']=='ours_prefixstore_kv25' and r['turn_id']==2)['result']['generated_token_ids']
    M.atomic_json(artifact/'integration_diagnostic.json',{'status':'PASS','AllHead_memory_reference':sample_receipts,
        'cross_method_recovery':recovery,'exception_restoration':'PASS','ours_actual_masks':trace.summary(),
        'diagnostic_requests_excluded_from_main_and_smoke':True})


def negative_audit(run,config,manifest):
    A=M.helper('109_audit_llava_mt_gqa_B.py','_main_negative_audit')
    cp=next((run/'integration/images').glob('*/COMMITTED.json'));commit=json.loads(cp.read_text())
    rows=M.load_jsonl(cp.parent/commit['attempt_id']/'raw.jsonl');image=cp.parent.name
    assert A.audit_image_rows(rows,manifest,config,'integration',image)['status']=='PASS'
    ah=next(i for i,r in enumerate(rows) if r['method_id']==M.ALLHEAD and r['turn_id']==2)
    ours=next(i for i,r in enumerate(rows) if r['method_id']=='ours_prefixstore_kv25' and r['turn_id']==2)
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
    assert len(recovered)==1 and recovered[image]['requests']==30
    return {'status':'PASS','negative_cases':report,'resume_committed_image_skip':'PASS','partial_image_rejection':'CPU fixture'}


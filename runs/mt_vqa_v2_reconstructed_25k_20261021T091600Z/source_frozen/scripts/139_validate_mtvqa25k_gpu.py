#!/usr/bin/env python3
"""Fresh native correctness on metadata-selected new-data samples, no main rows."""
import argparse,copy,gc,json,os,random,shutil,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from mmimpress.mtvqa25k.common import read,dump,sha
from mmimpress.mtvqa25k.adapter import bind,load,read_rows,remove_archived_duplicates
W=load('_mtvqa_workflow','138_run_mtvqa25k.py')

def rejected(call):
    try:call()
    except (ValueError,RuntimeError,AssertionError,FileNotFoundError,KeyError) as e:return {'rejected':True,'reason':repr(e)}
    raise AssertionError('negative fixture accepted')

def payload_negatives(source,opener,dest):
    dest.mkdir()
    records={}
    for kind in ('truncated','corrupted','unsealed'):
        target=dest/kind;shutil.copytree(source,target)
        payload=next(target.glob('layer_*/*.bin'))
        if kind=='truncated':
            with payload.open('r+b') as f:f.truncate(payload.stat().st_size-2)
        elif kind=='corrupted':
            with payload.open('r+b') as f:b=f.read(1);f.seek(0);f.write(bytes([b[0]^1]))
        else:
            seal=next((p for p in (target/'integrity.json',target/'completion.json',target/'meta.json') if p.exists()))
            seal.unlink()
        records[kind]=rejected(lambda:opener(target))
    return records

def main(run,model):
    import torch,numpy as np
    torch.set_num_threads(2);torch.manual_seed(1234);np.random.seed(1234);random.seed(1234)
    W.verify_freeze(run);W.verify_model_environment(run,model)
    phasepath,manifest,config=W.config_for(run,model,'gate')
    if (phasepath/'COMPLETE.json').exists():raise FileExistsError('gate complete; do not rerun')
    selection=read(run/'validation_selection.json')
    wanted={did for label,did in selection['reasons'].items() if label.startswith(model+'_') or label in ('most_portrait','most_landscape')}
    dialogs=[d for d in manifest['dialogues'] if d['dialog_id'] in wanted]
    attempt=phasepath/f'validation_{time.time_ns()}';attempt.mkdir()
    receipt={'status':'RUNNING','model':model,'samples':[],'negative_fixtures':{},'source_freeze_sha256':sha(run/'experiment_freeze.json'),
             'selected_before_model_outputs':True,'selected_dialogue_ids':[d['dialog_id'] for d in dialogs],'physical_requests':'NOT_MEASURED until diagnostic generation observer receipt is present','main_requests':0}
    W.status(run,model,'gate','RUNNING',samples=len(dialogs));runner=None
    try:
        if model=='llava':
            E=load('_mtvqa_prefix25_controls','106_prefix25_persistence.py');E.exclusive();E.capacity();exp=E.Experiment(attempt,model);runner=exp.runner
            M,A=bind(model);M._experiment_id=run.name;servers=M.servers_for(runner);D=load('_mtvqa_llava_integration','114_validate_llava_mt_gqa_B.py');D.M=M
            for index,d in enumerate(dialogs):
                sample={'image_id':d['image_id'],'image_path':str(ROOT/d['image_path']),'image_sha256':manifest['image_identity'][d['image_id']]['sha256'],
                        'question':d['turns'][0]['question'],'question_id':d['turns'][0]['question_id']}
                directory=attempt/f'core_{index:02d}';directory.mkdir()
                result=exp.llava_case(sample,directory)
                dump(directory/'validation.json',E.safe(result));receipt['samples'].append({'dialog_id':d['dialog_id'],'core':'PASS','path':str(directory)})
                if index==0:
                    from mmimpress.serve import ImageContext
                    receipt['negative_fixtures']['actual_payloads']=payload_negatives(directory/'B',lambda p:ImageContext(p,runner.model.device,require_v_hidden=False),attempt/'negative_payloads')
                # AllHead independent all-head/suffix reference, MPIC and cross-arm
                # restoration, actual three-turn own-history execution with raw audit.
                ordinal=manifest['dialogues'].index(d)
                M.run_image(runner,servers,[d],{d['dialog_id']:ordinal},'gate',phasepath,manifest,config,diagnostic=D.integration_diagnostic)
                remove_archived_duplicates(phasepath/'gate/images'/d['image_id'])
                print(model,d['dialog_id'],'core + integration PASS',flush=True)
                W.status(run,model,'gate','RUNNING',completed_samples=index+1,target_samples=len(dialogs))
            # Metadata/history negatives invoke independent audit, no generated outputs selected.
            parent=phasepath/'gate/images'/dialogs[0]['image_id'];c=read(parent/'COMMITTED.json');rows=read_rows(parent/c['attempt_id']/'raw.jsonl')
            for name,mutate in [('wrong_image',lambda rs:rs[0].update(image_sha256='0'*64)),
                ('wrong_history',lambda rs:next(r for r in rs if r['turn_id']==2)['history_entries'][0].update(answer='FORGED')),
                ('wrong_budget',lambda rs:next(r for r in rs if r['method_id']==M.METHODS[-1] and r['turn_id']==2).update(k=1))]:
                bad=copy.deepcopy(rows);mutate(bad);r=A.audit_image_rows(bad,manifest,config,'gate',dialogs[0]['image_id'])
                assert r['status']=='FAIL';receipt['negative_fixtures'][name]=r['failure_counts']
            exp.close_contexts()
        elif model=='qwen':
            M,A=bind(model);M._experiment_id=run.name;runner=M.load_runner();M._active_runner=runner
            P=load('_mtvqa_qwen_semantic','117_run_qwen_prefixstore3.py');P.M=M;P.A=A;P.D.M=M;P.D.A=A
            for index,d in enumerate(dialogs):
                ordinal=manifest['dialogues'].index(d)
                M.run_image(runner,[d],{d['dialog_id']:ordinal},'gate',phasepath,manifest,config,diagnostic=P.diagnostic)
                parent=phasepath/'gate/images'/d['image_id'];c=read(parent/'COMMITTED.json')
                records=[read(p) for p in (parent/c['attempt_id']).glob('semantic_*/validation.json')]
                assert len(records)==2 and all(r['status']=='PASS' for r in records)
                receipt['samples'].append({'dialog_id':d['dialog_id'],'semantic':'PASS','legacy_G7':[r['legacy_G7'] for r in records]})
                remove_archived_duplicates(parent)
                W.status(run,model,'gate','RUNNING',completed_samples=index+1,target_samples=len(dialogs))
                print(model,d['dialog_id'],'exact/same-shape + semantic PASS',flush=True)
        else:
            V=load('_mtvqa_glm_native_reference','120_validate_glm46v_prefixstore3.py');M=load('_mtvqa_glm_guard','121_eval_glm46v_prefixstore3.py');M.guard_gpu()
            runner=M.GLM46VRunner(ROOT/'runs/glm46v_prefixstore3_20261007T065207Z/checkpoint').load()
            for index,d in enumerate(dialogs):
                directory=attempt/f'sample_{index:02d}';directory.mkdir()
                isolation=copy.deepcopy(d);isolation['dialog_id']+=':diagnostic_state_isolation'
                sample={'image_id':d['image_id'],'image_path':str(ROOT/d['image_path']),'image_sha256':manifest['image_identity'][d['image_id']]['sha256'],
                    'question':d['turns'][0]['question'],'other_question':d['turns'][1]['question'],
                    'long_question':max(d['turns'],key=lambda t:len(t['question']))['question'],'actual_shared_dialogues':[d,isolation]}
                evidence=V.run_sample(runner,sample,directory);dump(directory/'evidence.json',evidence)
                if index==0:
                    from mmimpress.glm46v.store import open_glm_store
                    identity=read(directory/'prefix25/meta.json')['identity']
                    receipt['negative_fixtures']['actual_payloads']=payload_negatives(directory/'prefix25',lambda p:open_glm_store(p,expected_identity=identity),attempt/'negative_payloads')
                receipt['samples'].append({'dialog_id':d['dialog_id'],'native_bits_same_shape_semantic':'PASS','path':str(directory)})
                W.status(run,model,'gate','RUNNING',completed_samples=index+1,target_samples=len(dialogs))
                print(model,d['dialog_id'],'native all-layer gate PASS',flush=True)
        receipt.update(status='PASS',completed_samples=len(dialogs),criteria='fixed prior native correctness; legacy FP32 is separately retained, never promoted to PASS')
        dump(attempt/'receipt.json',receipt);dump(phasepath/'COMPLETE.json',{'status':'PASS','model':model,'phase':'gate','samples':len(dialogs),
            'receipt':str(attempt/'receipt.json'),'sha256':sha(attempt/'receipt.json'),'source_freeze_sha256':sha(run/'experiment_freeze.json')})
        W.status(run,model,'gate','PASS',completed_samples=len(dialogs))
    except BaseException as exc:
        receipt.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc());dump(attempt/'FAILURE.json',receipt)
        W.status(run,model,'gate','BLOCKED',error=repr(exc),failure_receipt=str(attempt/'FAILURE.json'));raise
    finally:
        if runner is not None and hasattr(runner,'close'):runner.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--model',choices=['llava','qwen','glm'],required=True);a=p.parse_args();main(a.run.resolve(),a.model)

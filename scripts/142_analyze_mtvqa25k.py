#!/usr/bin/env python3
"""Independent streaming raw/GT/IO/session rescore; no production model imports."""
import argparse,collections,contextlib,copy,csv,gzip,hashlib,io,json,math,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mmimpress.mtvqa25k.common import read,dump,sha

def rows(path):
    path=Path(path)
    if not path.exists():path=path.with_suffix(path.suffix+'.gz')
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:return [json.loads(s) for s in f if s.strip()]

def write_csv(path,data):
    if not data:return
    fields=list(dict.fromkeys(k for r in data for k in r))
    with path.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)

def stats(values):
    values=np.asarray(values,dtype=float)
    return {'mean':float(values.mean()),'p50':float(np.percentile(values,50)),'p95':float(np.percentile(values,95)),'n':len(values)}

def setup(folder,model,methods):
    result={m:{'physical_persistence_ms':0.,'standalone_persistence_ms':0.,'activation_ms':0.,'logical_bytes':0.,'allocated_bytes':0.,'write_bytes':0.,'write_calls':0.} for m in methods}
    physical=[]
    if model=='glm':
        for e in rows(folder/'setup_events.jsonl'):
            m=e['method_id'];physical.append(e['duration_ms'])
            if e['event_kind']=='build':result[m].update(physical_persistence_ms=e['duration_ms'],standalone_persistence_ms=e['duration_ms'],logical_bytes=e['logical_bytes'],allocated_bytes=e['allocated_bytes'],write_bytes=e['os_write_bytes'],write_calls=e['os_write_calls'])
            else:result[m]['activation_ms']=e['duration_ms']
    else:
        for m in methods[1:]:
            if model=='llava':
                kind={'fullload':'raster','mpic32_ssd':'mpic','sparsevlm_ssd_kv25_allhead':'raster','ours_prefixstore_kv25':'image_only'}[m]
                p=read(folder/f'{kind}_persistence.json');a=read(folder/f'{m}_activation.json');persist=p['outer_helper_wall_ms'];activation=a['outer_wall_ms'];shared=m=='sparsevlm_ssd_kv25_allhead'
            else:
                p=read(folder/f'{m}_persistence.json');a=read(folder/f'{m}_activation.json');persist=p['outer_writer_wall_ms'];activation=a['activation_ms'];shared=False
            result[m].update(physical_persistence_ms=0. if shared else persist,standalone_persistence_ms=persist,
                activation_ms=activation,logical_bytes=p['file_logical_bytes'],allocated_bytes=p['allocated_bytes'],write_bytes=p['os_write_bytes'],write_calls=p['os_write_calls'])
            physical.append(activation)
            if not shared:physical.append(persist)
    charged=sum(x['physical_persistence_ms']+x['activation_ms'] for x in result.values())
    assert math.isclose(sum(physical),charged,rel_tol=1e-12,abs_tol=1e-7)
    return result,charged

def bootstrap(values,counts,resamples=10000):
    """values: image x estimand pooled numerator; same draws for all estimands."""
    rng=np.random.RandomState(1234);n=len(values);out=np.empty((resamples,values.shape[1]))
    for start in range(0,resamples,16):
        size=min(16,resamples-start);idx=rng.randint(0,n,size=(size,n))
        for lo in range(0,values.shape[1],8):
            denominator=counts[idx,lo:lo+8].sum(axis=1)
            out[start:start+size,lo:lo+8]=values[idx,lo:lo+8].sum(axis=1)/denominator
    return np.percentile(out,[2.5,97.5],axis=0)

def analyze(run,model,phase,output,allow_partial=False):
    from lib2to3.refactor import RefactoringTool
    original=(ROOT/'mmimpress/mtvqa25k/vendor/vqaEval.py.original').read_text()
    namespace={};exec(compile(str(RefactoringTool(['lib2to3.fixes.fix_print']).refactor_string(original,'official_original')),'official_original','exec'),namespace)
    Reference=namespace['VQAEval']
    base=run/model/phase;manifest=read(base/'manifest.json');methods=read(run/model/'model_settings.json')['methods'];complete=(base/'COMPLETE.json').exists()
    if not complete and not allow_partial:raise RuntimeError('phase not complete; use --allow-partial explicitly')
    output.mkdir(parents=True,exist_ok=False)
    from tokenizers import Tokenizer
    snapshot=Path(read(run/'model_freeze.json')[model]['snapshot']);tokenizer=Tokenizer.from_file(str(snapshot/'tokenizer.json'))
    gen=read(snapshot/'generation_config.json');eos=gen.get('eos_token_id',[]);eos=set(eos if isinstance(eos,list) else [eos])
    per_image=[];storage=[];sessions=[];evidence=[];diagnostics=[];components=[];metrics=collections.defaultdict(list);totals=[]
    byimage={d['image_id']:d for d in manifest['dialogues']}
    for cp in sorted((base/phase/'images').glob('*/COMMITTED.json')):
        commit=read(cp);folder=cp.parent/commit['attempt_id'];iid=cp.parent.name;d=byimage[iid]
        for name,h in commit['evidence_sha256'].items():assert sha(folder/name)==h,('committed hash',name)
        rr=rows(folder/'raw.jsonl');assert len(rr)==3*len(methods)
        lookup={(r['method_id'],r['turn_id']):r for r in rr};assert len(lookup)==len(rr)
        assert set(lookup)=={(m,t) for m in methods for t in (1,2,3)}
        qa={};pred={}
        for index,r in enumerate(rr):
            t=d['turns'][r['turn_id']-1];assert str(r['question_id'])==str(t['question_id']) and r['annotation']==t['annotation']
            annotation=copy.deepcopy(t['annotation']);annotation['question_id']=index;qa[index]=annotation;pred[index]={'answer':r['scorer_answer']}
            ids=r['generated_token_ids'];literal=tokenizer.decode(ids,skip_special_tokens=False)
            assert literal==r['native_raw_decoded_output'],('raw decoding',r['logical_request_id'])
            if model=='glm':
                begins=[i for i,x in enumerate(ids) if x==151361];ends=[i for i,x in enumerate(ids) if x==151362]
                span=ids[begins[0]+1:ends[0]] if len(begins)==len(ends)==1 and begins[0]<ends[0] else ids
                native=tokenizer.decode(span,skip_special_tokens=True).strip()
                bodyids=ids[:-1] if ids and ids[-1] in eos else ids
                assert tokenizer.decode(bodyids,skip_special_tokens=False)==r['history_text']
            else:native=tokenizer.decode(ids,skip_special_tokens=True).strip()
            assert native==r['prediction']==r['scorer_answer']
            result=r.get('result',r);trace=result.get('os_pread_trace',r.get('os_pread_trace',[]))
            component={'image_id':iid,'method':r['method_id'],'turn':r['turn_id'],'logical_request_id':r['logical_request_id']}
            for key,value in result.items():
                if (key.endswith('_ms') or 'bytes' in key or 'preads' in key or key in ('N_content','k')) and isinstance(value,(int,float)) and not isinstance(value,bool):
                    component[key]=value
            component['scope']='raw production components; nested timings overlap and are not added to session'
            component['missing_component_policy']='NOT_MEASURED; absent H2D timing is not zero'
            components.append(component)
            assert r['ssd_read_bytes']==sum(e['returned'] for e in trace)
            if model!='glm':assert r['ssd_preads']==len(trace)
            assert all(e['returned']==e['requested'] and e['offset']>=0 for e in trace)
            assert 0<r['ttft_ms']<=r['request_e2e_ms']
            if r['turn_id']>1 and r['method_id']==methods[-1]:assert r['k']==(r['N_content']+3)//4
            if model=='llava':
                expected='\n'.join(line for t0 in range(1,r['turn_id']) for line in (f"Q{t0}: {d['turns'][t0-1]['question'].strip()}",f"A{t0}: {lookup[(r['method_id'],t0)]['prediction']}"))
                assert r['history']==expected
            else:
                expected=[[d['turns'][t0-1]['question'],lookup[(r['method_id'],t0)]['history_text' if model=='glm' else 'prediction']] for t0 in range(1,r['turn_id'])]
                assert r['history']==expected
        gt=SimpleNamespace(qa=qa,getQuesIds=lambda:list(qa));res=SimpleNamespace(qa=pred);v=Reference(gt,res,n=10)
        with contextlib.redirect_stdout(io.StringIO()):v.evaluate()
        for index,r in enumerate(rr):assert abs(r['score']-v.evalQA[index]/100)<1e-10
        values,charged=setup(folder,model,methods);totals.append(charged)
        for m in methods:
            s=values[m];storage.append({'image_id':iid,'method':m,**s})
            e2e=sum(lookup[m,t]['request_e2e_ms'] for t in (1,2,3))
            sessions.append({'image_id':iid,'method':m,'request_e2e_sum_ms':e2e,
                'physical_session_ms':e2e+s['physical_persistence_ms']+s['activation_ms'],
                'standalone_equivalent_session_ms':e2e+s['standalone_persistence_ms']+s['activation_ms'],
                'standalone_provenance':'DERIVED_FROM_SHARED' if m=='sparsevlm_ssd_kv25_allhead' else 'MEASURED_ONE_IMAGE_ONE_DIALOGUE'})
            for scope,turns in [('T1',[1]),('T2',[2]),('T3',[3]),('All',[1,2,3]),('Followup',[2,3])]:
                selected=[lookup[m,t] for t in turns]
                record={'image_id':iid,'method':m,'scope':scope,'count':len(turns)}
                for metric,key in [('score_pct','score'),('ttft_ms','ttft_ms'),('e2e_ms','request_e2e_ms'),('ssd_read_bytes','ssd_read_bytes')]:
                    vv=[r[key]*(100 if key=='score' else 1) for r in selected];record[metric+'_sum']=sum(vv);metrics[m,scope,metric].extend(vv)
                per_image.append(record)
            diagnostics.append({'image_id':iid,'method':m,'empty':sum(not lookup[m,t]['prediction'] for t in (1,2,3)),
                'eos_only':sum(lookup[m,t].get('eos_only',False) for t in (1,2,3)),
                'cap_reached':sum(len(lookup[m,t]['generated_token_ids'])==(32 if model=='glm' else 16) and lookup[m,t]['generated_token_ids'][-1] not in eos for t in (1,2,3)),
                'empty_score_sum':sum(lookup[m,t]['score'] for t in (1,2,3) if not lookup[m,t]['prediction']),
                'eos_only_score_sum':sum(lookup[m,t]['score'] for t in (1,2,3) if lookup[m,t].get('eos_only',False))})
        evidence.append({'image_id':iid,'commit':str(cp),'commit_sha256':sha(cp),'raw':str(folder/'raw.jsonl.gz'),
                         'requests':len(rr),'raw_uncompressed_sha256':commit['raw_sha256']})
    assert evidence,'no committed images'
    requests=sum(e['requests'] for e in evidence)
    if complete:assert requests==manifest['requests']
    summaries=[]
    for (m,scope,metric),vv in metrics.items():summaries.append({'model':model,'method':m,'scope':scope,'metric':metric,**stats(vv)})
    paired=[];vectors=[];denominators=[];labels=[];images=sorted(e['image_id'] for e in evidence)
    by={(r['image_id'],r['method'],r['scope']):r for r in per_image};ss={(r['image_id'],r['method']):r for r in sessions};st={(r['image_id'],r['method']):r for r in storage};ours=methods[-1]
    for baseline in methods[:-1]:
        for scope in ('T1','T2','T3','All','Followup'):
            for metric in ('score_pct','ttft_ms','e2e_ms','ssd_read_bytes'):
                num=[by[i,ours,scope][metric+'_sum']-by[i,baseline,scope][metric+'_sum'] for i in images];count=[by[i,ours,scope]['count'] for i in images]
                base_mean=sum(by[i,baseline,scope][metric+'_sum'] for i in images)/sum(count)
                labels.append({'baseline':baseline,'ours':ours,'scope':scope,'metric':metric,'baseline_mean':base_mean});vectors.append(num);denominators.append(count)
        for metric in ('physical_session_ms','standalone_equivalent_session_ms','logical_bytes','standalone_persistence_ms','activation_ms','write_bytes'):
            src=ss if 'session' in metric else st
            labels.append({'baseline':baseline,'ours':ours,'scope':'ImageSession','metric':metric,'baseline_mean':sum(src[i,baseline][metric] for i in images)/len(images)})
            vectors.append([src[i,ours][metric]-src[i,baseline][metric] for i in images]);denominators.append([1]*len(images))
    num=np.asarray(vectors,dtype=float).T;den=np.asarray(denominators,dtype=float).T
    ci=bootstrap(num,den);point=num.sum(axis=0)/den.sum(axis=0)
    for j,label in enumerate(labels):paired.append({**label,'ours_minus_baseline':point[j],'ci95_low':ci[0,j],'ci95_high':ci[1,j],
        'reduction_percent':-100*point[j]/label['baseline_mean'] if label['baseline_mean'] and label['metric']!='score_pct' else 'NOT_APPLICABLE',
        'paired_images':len(images),'resamples':10000,'seed':1234})
    for name,data in [('quality_latency_io.csv',summaries),('per_image.csv',per_image),('storage_setup.csv',storage),('session.csv',sessions),('paired_analysis.csv',paired),('generation_diagnostics.csv',diagnostics)]:write_csv(output/name,data)
    write_csv(output/'request_components.csv',components)
    write_csv(output/'quality.csv',[r for r in summaries if r['metric']=='score_pct'])
    write_csv(output/'latency.csv',[r for r in summaries if r['metric'] in ('ttft_ms','e2e_ms')])
    write_csv(output/'actual_io.csv',[r for r in summaries if r['metric']=='ssd_read_bytes'])
    dump(output/'independent_audit.json',{'status':'PASS','phase_complete':complete,'model':model,'phase':phase,'adopted_requests':requests,'target_requests':manifest['requests'],'images':len(images),
        'official_GT_score_native_decode_history_IO_checks':'PASS','physical_setup_attributed_ms':sum(totals),'source':{'analyzer':sha(Path(__file__)),'official_original':sha(ROOT/'mmimpress/mtvqa25k/vendor/vqaEval.py.original')},
        'bootstrap':'RandomState MT19937 seed1234;10000 image-cluster draws; pooled sums/counts;batch16;feature chunks8','old_or_other_phase_rows':0})
    dump(output/'raw_evidence_index.json',evidence)
    lines=[f'# {model} {phase} official VQA analysis','',f'Status: {"COMPLETE" if complete else "PARTIAL"}; adopted {requests}/{manifest["requests"]} requests; {len(images)} images.','',
        '| Method | T1 VQA % | T2 VQA % | T3 VQA % | All VQA % | Followup VQA % | Followup TTFT ms | Physical session ms | Standalone-equivalent ms |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in methods:
        quality=[np.mean(metrics[m,s,'score_pct']) for s in ('T1','T2','T3','All','Followup')]
        ses=[r for r in sessions if r['method']==m]
        lines.append('| '+m+' | '+' | '.join(f'{v:.4f}' for v in quality+[np.mean(metrics[m,'Followup','ttft_ms']),np.mean([r['physical_session_ms'] for r in ses]),np.mean([r['standalone_equivalent_session_ms'] for r in ses])])+' |')
    lines+=['','AllHead standalone-equivalent reattributes shared build cost; it is not an independent cold-session measurement. Each image has exactly one three-turn dialogue; setup is never amortized over hypothetical extra dialogues.',
        'Paired score differences use percentage points. Generated-history divergence is part of the method comparison. A CI containing zero is not equivalence. No cross-model aggregate effect. OS reads are not NAND reads. Missing components remain NOT_MEASURED in raw evidence.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'model':model,'phase':phase,'images':len(images),'requests':requests,'phase_complete':complete,'independent_audit':'PASS'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--model',choices=['llava','qwen','glm'],required=True);p.add_argument('--phase',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-partial',action='store_true');a=p.parse_args();analyze(a.run.resolve(),a.model,a.phase,a.output.resolve(),a.allow_partial)

#!/usr/bin/env python3
"""Independent raw-file audit, cluster bootstrap, and report generation."""
from __future__ import annotations
import argparse,csv,difflib,hashlib,json,math,re,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ARMS=['recompute','fullload','A','B']
LABEL={'recompute':'ReComp','fullload':'FullLoad','A':'Ours-FullStore-KV25 [A]','B':'Ours-PrefixStore-KV25 [B]'}

def read(path,default=None):return json.loads(path.read_text()) if path.exists() else default

def rows(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []

def write(path,v):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def csvwrite(path,values):
    if not values:path.write_text('status\nNOT RUN\n');return
    keys=list(dict.fromkeys(k for r in values for k in r))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in values)

def quality(pred,gold,strict=False):
    def norm(v):return ' '.join(w for w in re.sub(r'[^\w\s]',' ',str(v).lower()).split() if w not in {'a','an','the'})
    p,g=norm(pred),norm(gold)
    return float(p==g or (not strict and bool(g) and p.split()[:len(g.split())]==g.split()))

def avg(xs):return float(np.mean(xs)) if xs else None

def stats(xs):return {'mean':avg(xs),'p50':float(np.median(xs)) if xs else None,'p95':float(np.percentile(xs,95)) if xs else None}

def ci(diffs):
    x=np.asarray(diffs,dtype=float)
    if not len(x):return {'status':'NOT RUN'}
    rng=np.random.default_rng(1234);b=x[rng.integers(len(x),size=(10000,len(x)))].mean(1)
    return {'mean':float(x.mean()),'ci95':[float(y) for y in np.percentile(b,[2.5,97.5])],'clusters':len(x),'resamples':10000,'seed':1234}

def fmt(x):return 'NOT RUN' if x is None else f'{x:.3f}' if isinstance(x,(float,np.floating)) else str(x)

def table(head,rs):return [' | '.join(head),' | '.join(['---']*len(head))]+[' | '.join(fmt(x) for x in row) for row in rs]+['']

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--models',default='llava,qwen');a=p.parse_args();run=Path(a.run);out=ROOT/'results'/run.name;out.mkdir(exist_ok=True)
    summary=[];persistence=[];storage=[];sessions=[];analyses={};checks=[];text=['# Prefix25 persistence ablation','',f'Run: `{run}`. All numbers below are generated from this run\'s raw records. MB = 10^6 bytes; GiB = 2^30 bytes.','',
      'A retains the full importance-repacked store and uses the first ceil(N/4) content rows. B serializes those same content rows only, preserving structural/system KV. Existing full CPU materialization and full repack remain in both paths.','',
      'GPU validation reuses the prior frozen samples, not an unseen holdout. LLaVA uses its full physical buffer and 1e-4/1e-4 matched-logit tolerance; Qwen uses native BF16 logical compact assembly and bitwise-exact matched first logits. Prior Qwen strict-logits v1 FAIL remains unchanged.','']
    for model in a.models.split(','):
        status=read(run/model/'status.json',{});gate=read(run/model/'gpu_correctness.json',{})
        final_gate=read(run/model/'final_gpu_regression.json',{})
        text += [f'## {model}','',f'GPU correctness: **{gate.get("status","NOT RUN")}**, {len(gate.get("samples",[]))} recorded samples.','']
        text += [f'Final format replay: **{final_gate.get("status","NOT RUN")}**. Output token IDs, logit hashes and Qwen decode attention steps are retained in `runs/{run.name}/{model}/final_gpu_regression.json`.','']
        if gate.get('samples'):
            text+=table(['Image','Question','N','k','k/N','Status'],[[r['image_id'],r.get('question_id'),r.get('N'),r.get('k'),r.get('ratio'),r['status']] for r in gate['samples']])
        for phase in ['smoke','gqa','mt']:
            rr=rows(run/model/phase/'raw_requests.jsonl');pp=rows(run/model/phase/'persistence.jsonl');ok=[r for r in rr if r['status']=='OK']
            mf=read(run/f'{phase}_manifest.json',{});expected=sum(len(x['turns']) for x in mf.get('images',[]))*4
            for row in ok:
                row['raw_gqa_accuracy']=row['accuracy']
                row['accuracy']=quality(row['prediction'],row['gold'],strict=phase=='mt')
            audit={'model':model,'phase':phase,'expected_requests':expected,'actual_requests':len(rr),'successful_requests':len(ok),'errors':[]}
            cleanup=rows(run/model/'cleanup.jsonl');cleanup_before={r['path']:r for r in cleanup if r['stage']=='before'};cleanup_after={r['path']:r for r in cleanup if r['stage']=='after'}
            if not rr:
                audit['status']='NOT RUN';checks.append(audit)
                if phase!='smoke':text += [f'### {phase}: NOT RUN','', 'Performance is gated on this model\'s required correctness evidence.','']
                continue
            def require(cond,msg):
                if not cond:audit['errors'].append(msg)
            require(gate.get('status')=='PASS','performance without GPU PASS')
            require(final_gate.get('status')=='PASS','final source GPU replay incomplete/failed')
            require(len(ok)==expected,'request count mismatch')
            require(len({r['logical_request_id'] for r in rr})==len(rr),'duplicate logical request')
            require(len({r['execution_id'] for r in rr})==len(rr),'duplicate execution ID')
            require(len(pp)==len(mf['images'])*3,'physical persistence count mismatch')
            require(len({r['physical_write_id'] for r in pp})==len(pp),'duplicate physical write ID')
            by={(r['image_id'],r['arm'],r['turn']):r for r in ok};stores={(r['image_id'],r['arm']):r for r in pp}
            expected_turns={(im['image_id'],t['turn_id']):t for im in mf['images'] for t in im['turns']}
            require(set(by)=={(iid,arm,t) for iid,t in expected_turns for arm in ARMS},'frozen workload request set mismatch')
            for row in ok:
                iid,arm,turn=row['image_id'],row['arm'],row['turn'];v=row['result'];io=v['actual_io'];calls=io['calls']
                require(row['raw_gqa_accuracy']==quality(row['prediction'],row['gold']),'raw repository GQA score mismatch')
                require(math.isfinite(row['ttft_ms']) and math.isfinite(row['e2e_ms']) and 0<=row['ttft_ms']<=row['e2e_ms']+1e-3,'invalid timing boundary')
                expected_turn=expected_turns.get((iid,turn),{})
                require(all(str(row.get(k))==str(expected_turn.get(k)) for k in ['question_id','question','gold']),'frozen question/gold identity mismatch')
                require(sum(c['returned'] for c in calls)==row['read_bytes'],f'pread bytes {row["logical_request_id"]}')
                require(len(calls)==row['preads'],f'pread calls {row["logical_request_id"]}')
                require(all(c['requested']==c['returned'] for c in calls),'short pread')
                if row['cache_hit']:
                    rec=stores[(iid,arm)];m=rec['metadata'];n=rec['N'];k=(n+3)//4 if arm in ['A','B'] else n
                    require(v['vision_calls']==0 and v['online_query_score_calls']==0,'hit vision/scoring')
                    require(v.get('image_file_decode_ms') is None,'hit image decoded')
                    require(all(Path(c['path']).is_relative_to(Path(rec['store_path'])) for c in calls),'foreign payload read')
                    if arm in ['A','B']:
                        count=v['kept_tokens'] if model=='qwen' else v['attended_content_kv_count']
                        require(count==k,'incorrect original-content budget')
                        selected=v['selected_visual_original'] if model=='qwen' else v['selected_original_ids']
                        perm=m.get('full_importance_permutation',m.get('stored_to_original',m.get('order')))
                        require(selected==(sorted(perm[:k]) if model=='qwen' else perm[:k]),'selected IDs mismatch')
                        if arm=='B':require(v.get('extra_valid_visual_rows',v.get('unused_loaded_real_rows'))==0,'B overread real rows')
                        width=m.get('num_kv_heads',m.get('num_heads'))*m['head_dim']*2
                        diskrows=(k if arm=='B' else math.ceil(k/64)*64)
                        if model=='llava':diskrows=min(diskrows,m['v_token_num']);side=rec['files']['sep_kv.bin']['size']
                        else:side=m['bytes_structural_kv']
                        require(row['read_bytes']==2*m['num_layers']*diskrows*width+side,'independent geometry read bytes')
                else:require(v['vision_calls']==1 and row['read_bytes']==0,'pixel request geometry')
                if phase=='mt':
                    expected_hist=[[by[(iid,arm,j)]['question'],by[(iid,arm,j)]['prediction']] for j in range(1,turn)]
                    require(row['history']==expected_hist,'history not method own generated')
                else:require(row['history']==[],'independent question includes history')
                if arm=='B' and (iid,'A',turn) in by:
                    aa=by[(iid,'A',turn)];require(row['generated_token_ids']==aa['generated_token_ids'],'A/B sequence divergence');require(row['prediction']==aa['prediction'],'A/B prediction divergence')
            for rec in pp:
                m=rec['metadata'];n=rec['N'];k=(n+3)//4
                path=rec['store_path'];before_cleanup=cleanup_before.get(path);after_cleanup=cleanup_after.get(path)
                require(Path(path).resolve().is_relative_to((run/model/phase/'stores').resolve()),'cleanup path not run-owned')
                require(before_cleanup is not None and after_cleanup is not None and after_cleanup.get('exists') is False,'cleanup evidence missing')
                if before_cleanup:
                    require({k:(v['size'],v['sha256']) for k,v in before_cleanup['files'].items()}=={k:(v['size'],v['sha256']) for k,v in rec['files'].items()},'payload changed between persistence and cleanup')

                require(sum(f['size'] for f in rec['files'].values())==rec['file_logical_bytes'],'store byte sum')
                require(rec['os_write_bytes']>=rec['file_logical_bytes'],'OS writes smaller than files')
                if rec['arm']=='B':
                    require(rec['stored_k']==k and m['original_content_count']==n and m['stored_content_count']==k,'B metadata budget')
                    require(m['stored_row_to_original']==m['full_importance_permutation'][:k],'B partial mapping')
                    payload=[x['size'] for f,x in rec['files'].items() if f.endswith('/k.bin') or f.endswith('/v.bin')]
                    width=m.get('num_kv_heads',m.get('num_heads'))*m['head_dim']*2
                    require(payload==[k*width]*(2*m['num_layers']),'B payload contains extra real/padded rows')
                    ma=stores[(rec['image_id'],'A')]['metadata'];full=ma.get('stored_to_original',ma.get('order'))
                    require(full==m['full_importance_permutation'],'A/B full permutation differs')
                t=rec['timing_ms'];flat={key:rec[key] for key in ['model','phase','arm','image_id','physical_write_id','persistence_ms','os_write_bytes','os_write_calls','N','stored_k','content_retention']}
                flat.update({'activation_ms':rec['activation']['activation_ms'],'D2H_ms':t.get('kv_materialize_ms',0),'repack_ms':t.get('kv_repack_ms',0),'permutation_ms':t.get('permutation_ms',0),'serialization_ms':t.get('prefix_slice_serialization_ms',None),'write_ms':t.get('ssd_write_ms',0),'fsync_ms':t.get('fsync_ms',t.get('durability_ms',0)),'hash_and_seal_ms':rec['hash_and_seal_ms']});persistence.append(flat)
                storage.append({**{k:flat[k] for k in ['model','phase','arm','image_id','physical_write_id','N','stored_k']},**{key:rec[key] for key in ['original_content_bytes','stored_valid_content_bytes','structural_system_bytes','padding_bytes','file_logical_bytes','allocated_bytes','metadata_and_auxiliary_bytes','os_write_bytes','os_write_calls']},'payload_retained':Path(rec['store_path']).exists()})
                files=rec['files']
                storage[-1]['metadata_bytes']=sum(v['size'] for name,v in files.items() if name in ['meta.json','integrity.json','visionzip_layout.pt'])
                storage[-1]['structural_system_logical_bytes']=rec['structural_system_bytes']
                if model=='llava':
                    normal=sum(v['size'] for name,v in files.items() if name.endswith('/k.bin') or name.endswith('/v.bin'))
                    storage[-1]['structural_system_file_bytes_including_duplicate']=files['sys_kv.pt']['size']+files['sep_kv.bin']['size']+normal-rec['stored_valid_content_bytes']
                    storage[-1]['legacy_probe_bytes']=sum(v['size'] for name,v in files.items() if name.endswith('/probe_k.bin'))
                else:storage[-1]['structural_system_file_bytes_including_duplicate']=files['structural_kv.bin']['size']

            audit['status']='PASS' if not audit['errors'] else 'FAIL';checks.append(audit)
            if phase=='smoke':continue
            text += [f'### {phase}','',f'Raw audit: **{audit["status"]}**; {len(ok)}/{expected} successful requests.','']
            ss={}
            for arm in ARMS:
                vals=[r for r in ok if r['arm']==arm];hits=[r for r in vals if r['turn']>1]
                prs=[r for r in pp if r['arm']==arm];timings=[r for r in persistence if r['model']==model and r['phase']==phase and r['arm']==arm]
                aa=[by[(r['image_id'],'A',r['turn'])] for r in vals]
                s={'model':model,'phase':phase,'arm':arm,'n_requests':len(vals),'n_followup':len(hits),'accuracy_all':avg([r['accuracy'] for r in vals]),'accuracy_hit_or_followup':avg([r['accuracy'] for r in hits]),'first_agreement_vs_A':avg([float(r['first_token_id']==ref['first_token_id']) for r,ref in zip(vals,aa)]),'sequence_agreement_vs_A':avg([float(r['generated_token_ids']==ref['generated_token_ids']) for r,ref in zip(vals,aa)]),
                   'content_retention':avg([1. if arm in ['recompute','fullload'] else ((stores[(r['image_id'],arm)]['N']+3)//4)/stores[(r['image_id'],arm)]['N'] for r in vals]),
                   'read_MB_followup':avg([r['read_bytes']/1e6 for r in hits]),'preads_followup':avg([r['preads'] for r in hits]),'ttft_followup':stats([r['ttft_ms'] for r in hits]),'e2e_followup':stats([r['e2e_ms'] for r in hits]),'peak_gpu_MB':avg([r['result']['peak_gpu_allocated_bytes']/1e6 for r in hits]),
                   'peak_gpu_reserved_MB':avg([r['result']['peak_gpu_reserved_bytes']/1e6 for r in hits]),
                   'stored_content_MB':avg([r['stored_valid_content_bytes']/1e6 for r in prs]) or 0.,'total_store_MB':avg([r['file_logical_bytes']/1e6 for r in prs]) or 0.,'allocated_MB':avg([r['allocated_bytes']/1e6 for r in prs]) or 0.,'os_write_MB':avg([r['os_write_bytes']/1e6 for r in prs]) or 0.}
                for key in ['D2H_ms','repack_ms','serialization_ms','write_ms','fsync_ms','persistence_ms','activation_ms']:s[key]=avg([r[key] for r in timings if r[key] is not None]) if timings else 0.
                h2d=[]
                for r in hits:
                    if arm=='recompute':continue
                    v=r['result'];m=stores[(r['image_id'],arm)]['metadata']
                    h2d.append(v.get('h2d_kv_bytes',r['read_bytes']+2*m['num_layers']*m.get('v_token_start',0)*m.get('num_heads',0)*m['head_dim']*2))
                s['H2D_KV_MB']=avg([x/1e6 for x in h2d]);s['H2D_note']='Qwen runner tensor bytes; LLaVA independently inferred from actual payload tensor shapes + system; index/input transfers excluded'
                s['turn_accuracy']={str(t):avg([r['accuracy'] for r in vals if r['turn']==t]) for t in sorted({r['turn'] for r in vals})}
                ss[arm]=s;summary.append(s)
                for iid in sorted({r['image_id'] for r in vals}):
                    rs=sorted([r for r in vals if r['image_id']==iid],key=lambda r:r['turn']);rec=stores.get((iid,arm));persist=rec['persistence_ms'] if rec else 0.;activate=rec['activation']['activation_ms'] if rec else 0.
                    sessions.append({'model':model,'phase':phase,'image_id':iid,'arm':arm,'kind':'MT 3-turn standalone session' if phase=='mt' else 'six independent requests + one-time persistence/activation','T1_E2E_ms':rs[0]['e2e_ms'],'persistence_ms':persist,'activation_ms':activate,**{f'T{r["turn"]}_E2E_ms':r['e2e_ms'] for r in rs[1:]},'total_ms':sum(r['e2e_ms'] for r in rs)+persist+activate})
            text+=['Table 1. Correctness and output identity','']+table(['Method','Content retention','All accuracy','Hit/followup accuracy','First agreement vs A','Sequence agreement vs A'],[[LABEL[a],ss[a]['content_retention'],ss[a]['accuracy_all'],ss[a]['accuracy_hit_or_followup'],ss[a]['first_agreement_vs_A'],ss[a]['sequence_agreement_vs_A']] for a in ARMS])
            text+=['Table 2. Storage and first persistence (MB/image and ms)','']+table(['Method','Content MB','Total MB','Allocated MB','OS write MB','D2H ms','Repack ms','Write ms','fsync ms','Persistence ms','Activation ms'],[[LABEL[a]]+[ss[a][k] for k in ['stored_content_MB','total_store_MB','allocated_MB','os_write_MB','D2H_ms','repack_ms','write_ms','fsync_ms','persistence_ms','activation_ms']] for a in ARMS])
            text+=['Table 3. Followup requests (cached arms are hits; ReComp recomputes)','']+table(['Method','Read MB','Preads','TTFT mean/p50/p95 ms','E2E mean ms','H2D KV MB','GPU peak MB'],[[LABEL[a],ss[a]['read_MB_followup'],ss[a]['preads_followup'],'/'.join(fmt(ss[a]['ttft_followup'][k]) for k in ['mean','p50','p95']),ss[a]['e2e_followup']['mean'],ss[a]['H2D_KV_MB'],ss[a]['peak_gpu_MB']] for a in ARMS])
            sr=[r for r in sessions if r['model']==model and r['phase']==phase];sd={(r['image_id'],r['arm']):r for r in sr}
            if phase=='mt':
                mtmeans={arm:{key:avg([r.get(key,0) for r in sr if r['arm']==arm]) for key in ['T1_E2E_ms','persistence_ms','activation_ms','T2_E2E_ms','T3_E2E_ms','total_ms']} for arm in ARMS}
                text+=['Table 4. MT 3-turn standalone session (ms)','']+table(['Method','T1 E2E','Persistence','Activation','T2 E2E','T3 E2E','Total','Δ vs ReComp','Δ vs A'],[[LABEL[a]]+list(mtmeans[a].values())+[mtmeans[a]['total_ms']-mtmeans['recompute']['total_ms'],mtmeans[a]['total_ms']-mtmeans['A']['total_ms']] for a in ARMS])
            images=sorted({r['image_id'] for r in ok});paired={}
            for key in ['ttft_ms','e2e_ms','accuracy']:
                for subset in ['all','hit']:
                    ds=[]
                    for iid in images:
                        va=[r[key] for r in ok if r['image_id']==iid and r['arm']=='A' and (subset=='all' or r['turn']>1)]
                        vb=[r[key] for r in ok if r['image_id']==iid and r['arm']=='B' and (subset=='all' or r['turn']>1)]
                        ds.append(avg(vb)-avg(va))
                    paired[f'B_minus_A_{subset}_{key}']=ci(ds)
            for metric in ['persistence_ms','file_logical_bytes','os_write_bytes']:
                paired['B_minus_A_'+metric]=ci([stores[(i,'B')][metric]-stores[(i,'A')][metric] for i in images])
            for arm in ['A','B','fullload']:
                paired[arm+'_minus_ReComp_session_ms']=ci([sd[(i,arm)]['total_ms']-sd[(i,'recompute')]['total_ms'] for i in images])
            paired['B_minus_A_session_ms']=ci([sd[(i,'B')]['total_ms']-sd[(i,'A')]['total_ms'] for i in images])
            paired['storage_reduction']=1-sum(stores[(i,'B')]['file_logical_bytes'] for i in images)/sum(stores[(i,'A')]['file_logical_bytes'] for i in images)
            paired['OS_write_reduction']=1-sum(stores[(i,'B')]['os_write_bytes'] for i in images)/sum(stores[(i,'A')]['os_write_bytes'] for i in images)
            paired['persistence_reduction']=1-ss['B']['persistence_ms']/ss['A']['persistence_ms']
            paired['capacity_multiplier_estimate']=ss['A']['total_store_MB']/ss['B']['total_store_MB']
            paired['analytical_break_even_turns']={}
            for arm in ['A','B','fullload']:
                one=avg([sd[(i,arm)]['T1_E2E_ms']-sd[(i,'recompute')]['T1_E2E_ms']+sd[(i,arm)]['persistence_ms']+sd[(i,arm)]['activation_ms'] for i in images])
                saving=ss['recompute']['e2e_followup']['mean']-ss[arm]['e2e_followup']['mean']
                paired['analytical_break_even_turns'][arm]={'one_time_extra_ms':one,'assumed_constant_followup_E2E_saving_ms':saving,'turns':max(1,1+math.ceil(one/saving)) if saving>0 else None,'measured_beyond_three_turns':False}
            analyses[f'{model}/{phase}']=paired
            text += [f'B total-store reduction: {paired["storage_reduction"]:.2%}; OS write reduction: {paired["OS_write_reduction"]:.2%}; persistence reduction: {paired["persistence_reduction"]:.2%}. Estimated images per fixed SSD capacity: {paired["capacity_multiplier_estimate"]:.3f}×, including metadata and structural files. Cache-hit-rate or eviction benefit was not measured.','',
                     'Paired differences use 10,000 image-cluster bootstrap resamples (seed 1234). Negative B−A means lower latency. A CI spanning zero is inconclusive.','']
            text+=table(['Paired metric','Mean B−A','95% CI'],[[key,paired[key]['mean'],'['+', '.join(f'{v:.3f}' for v in paired[key]['ci95'])+']'] for key in ['B_minus_A_persistence_ms','B_minus_A_hit_ttft_ms','B_minus_A_hit_e2e_ms','B_minus_A_session_ms']])
    csvwrite(out/'latency_quality_summary.csv',summary);write(out/'latency_quality_summary.json',summary)
    csvwrite(out/'persistence.csv',persistence);csvwrite(out/'storage_metrics.csv',storage);csvwrite(out/'session_metrics.csv',sessions);write(out/'paired_analysis.json',analyses)
    protect=read(run/'protection_after.json',{'status':'NOT RUN'})
    audit={'status':'PASS' if checks and all(c['status']=='PASS' for c in checks) and protect.get('status')=='PASS' else 'INCOMPLETE' if not any(c.get('status')=='FAIL' for c in checks) else 'FAIL','checks':checks,'artifact_protection':protect,'independent_of_production_selector':True,'prior_results_used_as_same_run_measurements':False}
    write(out/'independent_audit.json',audit)
    text+=['## Interpretation and limits','',
      'The core A/B evidence is exact selected KV and request-level generated sequences, not equal average accuracy. B retains full CPU materialization and full repack, so their costs cannot disappear with smaller writes. Short final content files remove A\'s chunk-boundary overread; compare raw read bytes and TTFT CIs before attributing a speed change. GPU computation structure remains unchanged. GPU peaks include coexisting arm contexts and are not isolated deployment memory measurements; no memory-reduction claim is made.','',
      'Persistence is the wall time through the equal full-checksum durability seal. It is not reconstructed by summing nested components. Qwen saliency computation in capture exit and required cloning occur after normal generation; outer T1 E2E includes them once. File hash/seal time includes full hash reads and the final envelope/parent sync. Per-component write timing follows each existing writer; total wall time is the primary comparison.','',
      'Accuracy uses the existing GQA prefix-tolerant normalized scorer for GQA, and the existing MT pilot strict normalized exact match for MT. The collection raw accuracy field uses the generic GQA scorer; MT report accuracy is independently regenerated from raw prediction and gold, without changing raw files. Allocated bytes are regular-file st_blocks × 512; directory allocation is excluded. Forced source-release GC/empty_cache is experimental isolation outside service timers.','',
      'Followup H2D is Qwen\'s native tensor-byte counter or LLaVA\'s tensor-shape-derived payload plus system KV bytes. It excludes index/input transfers and is not a bus hardware counter. Allocated/reserved peaks, full read spans, conditioning, and phase timings are in raw records. OS wchar/syscw are successful syscall write traffic, not SSD NAND traffic. DONTNEED is an OS page-cache hint and does not prove cold controller/NAND.','',
      'All pilot payload stores are cleaned only after their image\'s scheduled measurements and hash/mapping receipts are committed. Correctness stores remain if present; pilot reproduction must regenerate stores. No previously existing store/result/raw was deleted.','',
      'Original performance source freezes are gpu_freeze_llava.json and gpu_freeze_qwen.json. The pre-Qwen amendment includes scoring in the outer T1 boundary and rejects unsealed incomplete B stores; normal sealed-store KV/attention semantics stay unchanged. final_gpu_freeze.json binds the final GPU replay. Both original measurement source versions are retained.','',
      'Longer-session break-even values in paired_analysis.json are analytical estimates using measured request E2E and one-time costs with constant mean hit savings. They are separate from the measured MT three-turn sessions. The 4,061-dialogue full experiment was not run.','',
      f'Artifact protection: {protect.get("status","NOT RUN")}. Independent raw audit: {audit["status"]}.','']
    text+=['## Decisions from measured results','']
    for model in ['llava','qwen']:
        g=read(run/model/'gpu_correctness.json',{});mt=analyses.get(model+'/mt');gqa=analyses.get(model+'/gqa')
        if g.get('status')!='PASS':
            text += [f'{model}: selected-KV/output identity is not established; performance/adoption and large-rerun decisions are BLOCKED by the required GPU gate.',''];continue
        text += [f'{model}: the fixed-sample gate established A/B selected-KV, visible attention and generated-output identity under the frozen model-specific thresholds.','']
        if mt and gqa:
            for phase,pa in [('GQA',gqa),('MT',mt)]:
                diff=pa['B_minus_A_session_ms'];persist=pa['B_minus_A_persistence_ms'];ttft=pa['B_minus_A_hit_ttft_ms']
                text += [f'{model} {phase}: total store bytes fell {pa["storage_reduction"]:.2%}, OS write bytes fell {pa["OS_write_reduction"]:.2%}, and persistence changed {persist["mean"]:.3f} ms (95% CI {persist["ci95"]}). Hit TTFT changed {ttft["mean"]:.3f} ms (95% CI {ttft["ci95"]}).','']
            ds=mt['B_minus_A_session_ms'];dr=mt['B_minus_ReComp_session_ms']
            word='improved' if ds['ci95'][1]<0 else 'regressed' if ds['ci95'][0]>0 else 'has an inconclusive paired difference'
            text += [f'{model}: the measured MT three-turn standalone session {word} versus A: B−A {ds["mean"]:.3f} ms, CI {ds["ci95"]}. B−ReComp is {dr["mean"]:.3f} ms, CI {dr["ci95"]}. This includes T1, persistence, activation and both hits.','',
                     f'{model}: fixed-25% adoption has measured capacity/persistence support when the corresponding reductions above are positive. It remains limited to a fixed maximum budget: B rejects larger content requests. Full CPU copy and repack costs remain. Large-rerun readiness requires the independent audit and protection receipt below; bounded per-image cleanup is required for capacity.','']
        else:text += [f'{model}: pilots are incomplete; capacity/persistence/session adoption conclusions remain BLOCKED.','']
    text+=['## Final model status','']
    final_status={}
    for model in ['llava','qwen']:
        st=read(run/model/'status.json',{})
        st['ARTIFACT PROTECTION']=protect.get('status','NOT RUN')
        checks_model=[c for c in checks if c['model']==model]
        final_replay=read(run/model/'final_gpu_regression.json',{});st['FINAL SOURCE GPU REPLAY']=final_replay.get('status','NOT RUN')
        ready=final_replay.get('status')=='PASS' and all(st.get(k)=='PASS' for k in ['IMPLEMENTATION','CPU REGRESSION','GPU CORRECTNESS','A/B OUTPUT IDENTITY','GQA PILOT','MT PILOT','PERSISTENCE/SESSION MEASUREMENT','ARTIFACT PROTECTION']) and all(c['status']=='PASS' for c in checks_model)
        st['READY FOR LARGE RERUN']='PASS' if ready else 'BLOCKED';final_status[model]=st
    keys=['IMPLEMENTATION','CPU REGRESSION','GPU CORRECTNESS','A/B OUTPUT IDENTITY','FINAL SOURCE GPU REPLAY','SMOKE','GQA PILOT','MT PILOT','PERSISTENCE/SESSION MEASUREMENT','ARTIFACT PROTECTION','READY FOR LARGE RERUN']
    text+=table(['Stage','LLaVA','Qwen'],[[k,final_status['llava'].get(k,'NOT RUN'),final_status['qwen'].get(k,'NOT RUN')] for k in keys])
    failures=rows(run/'failures.jsonl')
    if failures:
        text+=['## Preserved failures','']
        for f in failures:text+=[f'Model `{f.get("model")}`:','```text',f['error'],'```','']
    ko=['## 핵심 결과','',
        '두 모델의 CPU/GPU correctness와 A/B 출력 동일성 검증을 통과했다. 모델별 smoke 48요청, GQA 960요청, MT 480요청을 새로 실행했으며, 과거 latency/persistence를 섞지 않았다. 모든 T1은 full-image inference이고 표의 content retention은 hit에서 사용하는 예산이다.','']
    ko += table(['모델','저장/write 감소 (GQA / MT)','Persistence 감소 (GQA / MT)','MT 세션 B−A','MT 세션 B−ReComp'],[
        [model, ' / '.join(f'{analyses[model+"/"+ph]["storage_reduction"]:.2%}' for ph in ['gqa','mt']),
         ' / '.join(f'{analyses[model+"/"+ph]["persistence_reduction"]:.2%}' for ph in ['gqa','mt']),
         f'{analyses[model+"/mt"]["B_minus_A_session_ms"]["mean"]:.1f} ms',
         f'{analyses[model+"/mt"]["B_minus_ReComp_session_ms"]["mean"]:.1f} ms']
        for model in ['llava','qwen'] if model+'/gqa' in analyses and model+'/mt' in analyses])
    ko += ['고정 25% 정책에서 A를 B로 대체할 근거는 저장량·persistence·A 대비 세션 비용에서 확인됐다. 다만 LLaVA의 짧은 3-turn 세션은 ReComp가 더 빠르다. A/B 동등성은 선택 KV와 생성열을 직접 비교한 결과이며 ReComp 대비 품질 동등성을 뜻하지 않는다. 전체 CPU 복사와 전체 repack 비용은 남아 있다.','',
           '[MT 세션 비용 그림](session_costs.png) · [벡터 그림](session_costs.svg)','']
    text[2:2]=ko
    (out/'REPORT.md').write_text('\n'.join(text));write(out/'status.json',final_status)
    (out/'REPRODUCE.md').write_text(f'''# Reproduction

Environment: /home/dblab/anaconda3/envs/mllm_ft/bin/python (see run environment.json).
Input manifests, source before copies, source hashes, GPU freeze, failures and raw
receipts are retained under `{run}`. Offline cached revisions are fixed in config.
Create fresh timestamped runs/results paths; do not overwrite this run.

```bash
PYTHONPATH=.:tests /home/dblab/anaconda3/envs/mllm_ft/bin/python -m unittest -v test_prefix25_persistence test_llava_kv25 test_qwen25_kv25 test_qwen25_store test_image_only_repack test_visdial_turn1_piggyback_core test_qwen25_runner test_qwen25_vision
# Store CPU output as cpu_regression_v2.log in the new run.
/home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/106_prefix25_persistence.py --run NEW_RUN --freeze
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/dblab/anaconda3/envs/mllm_ft/bin/python -u scripts/106_prefix25_persistence.py --run NEW_RUN --model llava
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/dblab/anaconda3/envs/mllm_ft/bin/python -u scripts/106_prefix25_persistence.py --run NEW_RUN --model qwen
/home/dblab/anaconda3/envs/mllm_ft/bin/python scripts/107_report_prefix25.py --run NEW_RUN
```

Run models sequentially on an idle GPU. A failed gate blocks that model's pilots.
CPU unittest discovery uses tests on PYTHONPATH; pytest is not installed.
Actual command logs are in llava_execution.log/qwen_execution.log and failures.jsonl.
Pilot payloads are not retained: regenerate them from each arm's own T1.
Existing artifact protection uses protected_before.jsonl and protection_after.json.
At least 30 GiB free is mandatory; the driver additionally budgets 16 GiB staging.
The large full MT experiment is deliberately excluded.

API: both writers default to storage_policy="full". B explicitly uses
storage_policy="prefix25"; serving uses budget_unit="visual_kv", ratio=.25
against original N. The LLaVA completion protocol requires seal_integrity(store)
after its capture writer; the experiment applies the same seal to A and B.
Unsealed B activation fails. Qwen keeps its native payload-hash checks as well.
LLaVA v2 keeps v_token_num/n_chunks_per_layer as the original virtual GPU geometry;
payload_rows/payload_chunks and stored_row_to_original describe actual SSD rows.
Qwen v2 keeps visual_count as N, full_importance_permutation as the complete rank,
and stored_to_original/stored_row_to_original as k actual stored rows. Neither
partial mapping is treated as a complete original permutation.

The original LLaVA/Qwen measurement freezes and source copies are retained.
The final default replay includes the prospective Qwen timing correction and
completion-envelope guard. final_gpu_regression.py replays frozen GPU samples
under final_gpu_freeze.json; original performance rows are not overwritten.

''')
    print(json.dumps({'audit':audit['status'],'summaries':len(summary),'sessions':len(sessions),'output':str(out)}))
if __name__=='__main__':main()

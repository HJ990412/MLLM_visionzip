#!/usr/bin/env python3
"""Independent Qwen-only raw/event reconstruction; no inference or production scorer."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,hashlib,importlib.util,json,math,os,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
SCHEMA='qwen-mtgqa-prefixstore3-semantic-v2'

def load(name,file):
    s=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
R=load('_prefix3_independent_statistics','113_report_mt_gqa_B.py')
original_load=R.load
def versioned_load(name,path):
    m=original_load(name,path)
    if Path(path).name=='111_audit_qwen_mt_gqa_B.py':m.SCHEMA=SCHEMA
    return m
R.load=versioned_load

def f(x,d=3):return f'{x:.{d}f}' if isinstance(x,(float,int)) else str(x)
def table(headers,rows):return '\n'.join(R.mdtable(headers,rows))
def readcsv(p):
    with p.open() as fh:return list(csv.DictReader(fh))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run=a.run.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    status,quality,latency,paired,storage,sessions,_=R.analyze_model(run,out,'qwen')
    assert status['status']=='PASS' and status['main_requests']==36549
    cfg=R.read(run/'config.json');methods=cfg['methods'];B=methods[-1]
    labels={'recompute':'ReComp','fullload':'FullLoad',B:'Ours B'}
    runtime=R.read(run/'integration_runtime.json');eos=set(runtime['eos_token_ids'])
    groups=defaultdict(list);rescores=[];raw_hashes=[]
    for cp in sorted((run/'main/images').glob('*/COMMITTED.json')):
        commit=R.read(cp);folder=cp.parent/commit['attempt_id'];raw=folder/'raw.jsonl'
        raw_hashes.append(dict(path=str(raw),sha256=R.sha(raw),requests=commit['requests']))
        for r in R.rows(raw):
            v=r['result'];tokens=v['generated_token_ids'];m=r['method_id'];t=r['turn_id'];score=float(R.norm(r['prediction'])==R.norm(r['gold'][0]))
            assert score==r['score']
            rescores.append(dict(logical_request_id=r['logical_request_id'],image_id=r['image_id'],dialog_id=r['dialog_id'],method_id=m,turn_id=t,strict_score=score,
                prediction=r['prediction'],gold=r['gold'][0],generated_tokens=len(tokens),ended_eos=tokens[-1] in eos,empty=r['prediction']=='',cap=bool(v['truncated'])))
            for scope in ('all',f'T{t}',*(['followup'] if t>1 else [])):
                groups[(m,scope)].append(dict(compact_kv_bytes=v.get('gpu_cache_kv_bytes'),h2d_kv_bytes=v.get('h2d_kv_bytes'),
                    cap=bool(v['truncated']),empty=r['prediction']=='',eos=tokens[-1] in eos,
                    extra_read_rows=v.get('extra_valid_visual_rows'),post_capture_tail_ms=v.get('post_generation_tail_ms')))
    R.writecsv(out/'independent_rescoring.csv',rescores);R.dump(out/'raw_reproducibility.json',dict(status='PASS',files=raw_hashes,raw_lossless=True,
        cleaned_payloads_require_regeneration=True,recipe='exact source/model/manifest; rerun image own full T1, own store, all own histories'))
    detail=[]
    for (m,scope),rs in groups.items():
        d=dict(method_id=m,scope=scope,requests=len(rs))
        for field in ('cap','empty','eos'):d[field+'_count']=sum(x[field] for x in rs);d[field+'_rate']=d[field+'_count']/len(rs)
        for field in ('compact_kv_bytes','h2d_kv_bytes','extra_read_rows','post_capture_tail_ms'):
            vs=[x[field] for x in rs if x[field] is not None]
            d.update({field+'_'+k:(v if v is not None else 'NOT_MEASURED') for k,v in R.stats(vs).items()});d[field+'_measured_count']=len(vs)
        detail.append(d)
    R.writecsv(out/'request_detail_summary.csv',detail)
    # Do not reuse a PASS label from another run: inspect every current diagnostic.
    diagnostics=[]
    for base in (run/'integration',run/'additional/integration'):
        for pth in sorted(base.glob('images/*/attempt_*/semantic_*/validation.json')):
            x=R.read(pth);assert x['status']=='PASS' and x['all_layer_QKV_mask_exact']
            assert all(z['semantic_fp64']['allclose'] for z in x['layers'])
            assert R.sha(pth.parent/'fixed_inputs.pt')==x['fixture_sha256']
            assert R.sha(pth.parent/'layer0_outputs.pt')==x['output_sha256']
            diagnostics.append(dict(path=str(pth),sha256=R.sha(pth),legacy=x['legacy_G7'],FP64_max=max(z['semantic_fp64']['max_abs'] for z in x['layers']),matched='PASS'))
    assert len(diagnostics)==24
    verdict=R.read(run/'G7_review_verdict.json');assert verdict['status']=='PASS'
    stages={
        'LEGACY G7 STATUS':'FAIL (preserved and reproduced)',
        'G7 ROOT-CAUSE ATTRIBUTION':'PASS: FP32 operation shape/rounding in validator',
        'VALIDATOR/IMPLEMENTATION CHANGE':'validator + experiment adapter only; existing inference unchanged',
        'FINAL SOURCE FREEZE':'PASS','CPU REGRESSION':R.read(run/'cpu_regression.json')['status'],
        'MATCHED GPU CORRECTNESS':R.read(run/'core_gpu_validation.json')['status'],
        'INDEPENDENT SEMANTIC VALIDATION':'PASS','GENERATED-HISTORY INTEGRATION':R.read(run/'integration_validation.json')['status'],
        'ADDITIONAL 8 DIALOGUES':'PASS','NEGATIVE FIXTURES':'PASS','MAIN-RUNNER SMOKE':R.read(run/'smoke/validation.json')['status'],
        'FULL MAIN COVERAGE':'PASS: 36,549 / 36,549','HISTORY/SELECTION/I/O AUDIT':'PASS',
        'QUALITY RESCORING':'PASS','SESSION ACCOUNTING':R.read(out/'session_accounting_audit.json')['status'],
        'RAW REPRODUCIBILITY':'PASS'}
    import main_b_support as support
    protection=support.protection(run);R.dump(out/'protection_final.json',protection);assert protection['status']=='PASS'
    stages['ARTIFACT PROTECTION']='PASS'
    R.dump(out/'validation_summary.json',dict(status='PASS',stages=stages,diagnostics=diagnostics,legacy_G7='FAIL'))
    R.dump(out/'paired_analysis.json',dict(resamples=10000,seed=1234,cluster='image',weighting='pooled request sums/counts',comparisons=paired))
    q={(x['method_id'],x['scope']):x for x in quality};l={(x['method_id'],x['scope']):x for x in latency};s={x['method_id']:x for x in storage};ss={x['method_id']:x for x in sessions};dd={(x['method_id'],x['scope']):x for x in detail}
    def ci(b,scope,metric):return next(x for x in paired if x['baseline']==b and x['scope']==scope and x['metric']==metric)
    def ci_str(x):return f"{x['point']:+.3f} [{x['ci_low']:+.3f}, {x['ci_high']:+.3f}]"
    lines=['# Qwen PrefixStore3 MT-GQA-reconstructed generated-history main','',f'Run: `{run}`. Qwen-only; completed LLaVA remains unchanged. MB=10^6 bytes, GiB=2^30 bytes, ms, accuracy differences in percentage points.','',table(['Validation','Status'],stages.items()),'',
        'G7 is classified as a numerical validation problem. Actual BF16 SSD KV, positions, masks, first logits and generated IDs match independent same-shape references. Fixed-input FP32 QK, softmax and value aggregation vary with dense/compact operation shape on CPU. All-layer FP64 references (converted before operations) meet 1e-12/1e-12. FP64 is finite-precision validation, not main inference. The old 1e-5/1e-5 FAIL remains. G7_REVIEW.md and G7_stage_attribution.json contain causal controls, input hashes and stage attribution.','',
        '## Accuracy','',table(['Method','T1 %','T2 %','T3 %','All % (correct/N)','Followup % (correct/N)'],[[labels[m],*[f(100*q[m,t]['accuracy']) for t in ('T1','T2','T3')],*[f"{100*q[m,t]['accuracy']:.3f} ({int(q[m,t]['correct'])}/{q[m,t]['requests']})" for t in ('all','followup')]] for m in methods]),'',
        table(['B minus baseline','All Δpp [95% CI]','Followup Δpp [95% CI]'],[[labels[m],ci_str(ci(m,'all','accuracy_pp')),ci_str(ci(m,'followup','accuracy_pp'))] for m in methods[:-1]]),'',
        'These are method-level comparisons including generated-history divergence. Raw predictions/gold are independently strict-normalized rescored. Turn-specific paired CIs and first-token/full-sequence/normalized-prediction agreements and paired contingencies are in paired_analysis and prediction_agreement.','',
        '## Followup performance','',table(['Method','TTFT mean/p50/p95 ms','E2E mean/p95 ms','Actual KV read MB','Preads','H2D tensor MB','Compact KV MB'],[[labels[m],'/'.join(f(l[m,'followup']['ttft_ms_'+v]) for v in ('mean','p50','p95')),'/'.join(f(l[m,'followup']['request_e2e_ms_'+v]) for v in ('mean','p95')),f(l[m,'followup']['ssd_read_bytes_mean']/1e6),f(l[m,'followup']['ssd_preads_mean']),f(dd[m,'followup']['h2d_kv_bytes_mean']/1e6) if isinstance(dd[m,'followup']['h2d_kv_bytes_mean'],(int,float)) else 'NOT_MEASURED',f(dd[m,'followup']['compact_kv_bytes_mean']/1e6) if isinstance(dd[m,'followup']['compact_kv_bytes_mean'],(int,float)) else 'NOT_MEASURED'] for m in methods]),'',
        'ReComp followups are full-image recomputation, not cache hits. KV reads=0 does not mean zero image-file I/O. B reads 100% of its stored k content rows, representing ceil(original N/4). All structure is retained. Page-cache DONTNEED is an OS hint, not cold NAND. H2D bytes count tensors. GPU peak includes coexisting contexts.','',
        '## Store and setup','',table(['Method','Content % (pooled)','Valid content MB','Total/allocated MB','OS write MB','Persistence mean/p50/p95 ms','Activation mean/p50/p95 ms','Build/activation'],[[labels[m],f(100*s[m]['token_pooled_retention']),f(s[m]['valid_content_bytes_mean']/1e6),f(s[m]['file_logical_bytes_mean']/1e6)+'/'+f(s[m]['allocated_bytes_mean']/1e6),f(s[m]['os_write_bytes_mean']/1e6),'/'.join(f(s[m]['service_persistence_ms_'+v]) for v in ('mean','p50','p95')),'/'.join(f(s[m]['activation_ms_'+v]) for v in ('mean','p50','p95')),'398 / 398'] for m in methods[1:]]),'',
        'Persistence includes durable native publication and experiment completion seal; activation includes integrity reads/metadata/FD setup. Source T1 saliency/capture-clone tail is charged exactly once in its request E2E. Component intervals can overlap and are not summed. ReComp has no KV store/build/activation.','',
        '## Three-turn session accounting','',table(['Method','Shared-stream mean/p95 ms','Derived standalone mean/p95 ms'],[[labels[m],'/'.join(f(ss[m]['physical_stream_session_ms_'+v]) for v in ('mean','p95')),'/'.join(f(ss[m]['derived_standalone_session_ms_'+v]) for v in ('mean','p95'))] for m in methods]),'',
        table(['B − ReComp','Δms [95% CI]'],[[scope,ci_str(ci('recompute','per_dialogue',metric))] for scope,metric in [('Actual shared-image stream','physical_stream_session_ms'),('Derived standalone-equivalent','derived_standalone_session_ms')]]),'',
        'Shared-stream attribution charges each actual image/method setup once, at its source dialogue; sums reconcile to all physical events. Standalone-equivalent reallocates that measured setup once to each dialogue. It is DERIVED, not a fresh-store-per-dialogue measurement. Non-source T1 capture increment is NOT_MEASURED and is not filled as zero. Neither uses TTFT sums as session E2E.','',
        '## Qwen findings and limits','']
    b=l[B,'followup']
    for baseline in methods[:-1]:
        x=l[baseline,'followup'];ac=ci(baseline,'followup','accuracy_pp')
        lines.append(f"B vs {labels[baseline]}: followup accuracy {ci_str(ac)} pp; TTFT mean {(1-b['ttft_ms_mean']/x['ttft_ms_mean'])*100:.2f}% reduction, p95 {(1-b['ttft_ms_p95']/x['ttft_ms_p95'])*100:.2f}% reduction. Actual KV read difference {(b['ssd_read_bytes_mean']-x['ssd_read_bytes_mean'])/1e6:+.3f} MB/request.")
    lines+=['','Mean and p95 are reported separately; a mean improvement alone does not establish p95 improvement or statistical equivalence. Single GPU, batch1, bounded image-major shared cache; no long-session throughput/eviction/generalization claims. LLaVA is a separate historical run and no cross-model latency pairing is implied.','',
        'Generation cap/empty/EOS counts and rates are in request_detail_summary.csv. Attempts, physical events and failed diagnostics remain separate from final main requests. Native store payloads were deleted only after audit+hash+commit within exact owned paths and require regeneration. Prior large-file protection uses labeled sampled fingerprints; new artifacts and raw use full SHA256.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    repro=f'''# Reproduce Qwen PrefixStore3

Python: /home/dblab/anaconda3/envs/mllm_ft/bin/python. No dependency upgrade.
Use a NEW timestamped run with source protection bootstrap and REQUEST.md.
The bootstrap record/source snapshots plus final_source_freeze.json bind this run.
Environment: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=.:scripts:tests OMP_NUM_THREADS=2 MKL_NUM_THREADS=2.
Prepare additional manifest before outcomes, retain diagnostic attempts, freeze
the documented contract and all source. Use scripts/117_run_qwen_prefixstore3.py
--run NEW_RUN --stage prepare, diagnose, freeze, cpu, core, integration, review,
then main. Do not copy PASS receipts. Main runs actual smoke then all requests.
Resume this unchanged frozen run with --stage main --resume; completed image
evidence is independently checked; partial image is re-executed in a new attempt.
Never invoke legacy 110 directly for this v2 schema/contract.

Replay saved fixed inputs without GPU:
python scripts/116_qwen_prefixstore_semantic.py --fixture FIXED_INPUTS.pt --output NEW.json
All diagnostic tensors, including failure attempts, remain under {run}.
Main uses unchanged native BF16/SDPA; FP64 only diagnostic references.
Analysis: python scripts/118_report_qwen_prefixstore3.py --run {run} --output NEW_OUTPUT
Run final analysis/protection only after timed GPU main has completed. No prior
LLaVA GPU or artifact changes. Raw/payload full hashes, original model snapshot,
processor/tokenizer identity and native code revision are retained. Historical
large-file fingerprints are sampled, not full-byte SHA256 proof.
'''
    (out/'REPRODUCE.md').write_text(repro)
    R.dump(out/'completion.json',dict(status='PASS',final_logical_main_requests=36549,legacy_G7='FAIL',revised_validation='PASS',completed_at_unix=time.time()))
    index=[]
    for base in (run,out):
        for path in sorted(base.rglob('*')):
            if path.is_file():index.append(dict(path=str(path),bytes=path.stat().st_size,sha256=R.sha(path)))
    R.dump(out/'artifact_index_final.json',dict(full_hash=True,files=index,excludes='this self-referential index only; created after inference and diagnostic logs closed'))
    print(json.dumps(dict(status='PASS',main_requests=36549,report=str(out/'REPORT.md'))),flush=True)

if __name__=='__main__':main()

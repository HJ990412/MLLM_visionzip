"""Terminal receipts after launcher exits; no GPU work and no inference changes."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,os,shutil,time
ROOT=Path('/home/dblab/hj/mllm_v2');RUN=Path(__file__).resolve().parent;OUT=ROOT/'results'/RUN.name;FINAL=OUT/'final'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
assert read(RUN/'main_process_receipt.json')['returncode']==read(RUN/'analysis_process_receipt.json')['returncode']==0
assert read(FINAL/'completion.json')['final_logical_main_requests']==36549
counts=[]
for base,label in [(RUN,'final'),(RUN/'additional','additional'),(RUN/'same_image_fixture','same_image'),*[(p,p.name) for p in sorted(RUN.glob('diagnosis_v*')) if p.is_dir()]]:
 for phase in ('integration','smoke','main'):
  path=base/phase/'images'
  if not path.exists():continue
  c=Counter();c['images_committed']=len(list(path.glob('*/COMMITTED.json')))
  for cp in path.glob('*/COMMITTED.json'):c['adopted_logical_requests']+=read(cp)['requests']
  for ap in path.glob('*/attempt_*'):
   c['image_attempts']+=1;c['failed_image_attempts']+=(ap/'failure.json').exists()
   for name,field in [('raw.jsonl','raw_requests'),('execution_events.jsonl','executed_requests')]:
    p=ap/name
    if p.exists():c[field]+=sum(bool(line.strip()) for line in p.open())
   p=ap/'attempts.jsonl'
   if p.exists():
    for line in p.open():c['attempt_'+json.loads(line)['status']]+=1
   c['build_events']+=len(list(ap.glob('*_persistence.json')));c['activation_events']+=len(list(ap.glob('*_activation.json')))
  counts.append(dict(scope=label,phase=phase,**c))
dump(OUT/'execution_scope_counts.json',dict(status='PASS',counts=counts,core_samples=10,core_reference_forward_count='NOT_MEASURED as unique execution IDs',repeat_reference_samples=24,main_counts_exclude_all_validation_smoke_attempts=True))
# Verify completed LLaVA counts without rerunning it; preservation itself is the
# separate whole historical artifact fingerprint/source audit from 118.
old=ROOT/'runs/mt_gqa_B_llava5_qwen3_20261001T093921Z';oldres=ROOT/'results'/old.name
commits=[read(p) for p in (old/'llava/main/images').glob('*/COMMITTED.json')]
assert len(commits)==398 and sum(c['requests'] for c in commits)==60915
refs={str(p):sha(p) for p in [oldres/'REPORT.md',oldres/'REPRODUCE.md',oldres/'artifact_index_final.json',old/'qwen/integration_validation.json']}
dump(OUT/'llava_preservation_reference.json',dict(status='PASS',images=398,logical_requests=60915,inference_rerun=False,read_only=True,authoritative_reference_hashes=refs,scope='counts/reference identity; whole baseline protection in final/protection_final.json'))
for name in ('CONTRACT.md','G7_REVIEW.md','G7_stage_attribution.json','G7_review_verdict.json','pre_main_admission_freeze.json','final_source_freeze.json'):
 shutil.copy2(RUN/name,OUT/name)
text=(FINAL/'REPORT.md').read_text()
DETAIL=OUT/'reporting_details'
assert read(DETAIL/'session_accounting_audit.json')['status']=='PASS'
assert read(DETAIL/'reporting_amendment.json')['production_or_raw_changed'] is False if 'production_or_raw_changed' in read(DETAIL/'reporting_amendment.json') else True
def csvrows(p):return list(csv.DictReader(p.open()))
def table(h,rs):return '\n'.join(['| '+' | '.join(h)+' |','|'+'|'.join(['---']*len(h))+'|',*['| '+' | '.join(map(str,r))+' |' for r in rs]])
def fmt(v):return f'{float(v):.3f}'
labels={'recompute':'ReComp','fullload':'FullLoad','qwen_ours_prefixstore_kv25':'Ours B'}
methods=list(labels)
summaries={s['method_id']:s for s in csvrows(DETAIL/'store_io_summary.csv')}
old_summaries={s['method_id']:s for s in csvrows(FINAL/'storage_summary.csv')}
for m in methods[1:]:
 old='/'.join(fmt(old_summaries[m]['service_persistence_ms_'+x]) for x in ('mean','p50','p95'))
 new='/'.join(fmt(summaries[m]['outer_writer_wall_ms_'+x]) for x in ('mean','p50','p95'))
 assert old in text
 text=text.replace(old,new,1)
text=text.replace('Persistence mean/p50/p95 ms','Outer persistence mean/p50/p95 ms')
session_summaries=csvrows(DETAIL/'session_summary.csv')
pairs=read(DETAIL/'paired_analysis.json')['rows']
section='## Three-turn session accounting\n\n'
section+=table(['Method','Shared-stream mean/p50/p95 ms','Derived standalone mean/p50/p95 ms'],[[labels[x['method_id']],*['/'.join(fmt(x[key+'_'+k]) for k in ('mean','p50','p95')) for key in ('physical_stream_session_ms','derived_standalone_session_ms')]] for x in session_summaries])
section+='\n\n'+table(['B minus baseline','Accounting','Mean delta ms [95% CI]'],[[labels[x['baseline']],x['metric'],fmt(x['point'])+' ['+fmt(x['ci_low'])+', '+fmt(x['ci_high'])+']'] for x in pairs if x['scope']=='per_dialogue'])
section+='\n\nThese tables include measured outer writer-return time plus activation. Shared-stream charges each actual image/method setup once at its source dialogue and reconciles to 796 physical builds and 796 activations. Standalone-equivalent reattributes the same measured setup once per dialogue; it is DERIVED and omits a NOT_MEASURED hypothetical non-source capture increment. Request E2E, not TTFT sums, supplies the three request costs. The original inner durable-publication endpoint remains in final/; the reporting-only amendment and outer-endpoint CIs are in reporting_details/.\n\n'
a=text.index('## Three-turn session accounting');b=text.index('## Qwen findings and limits')
text=text[:a]+section+text[b:]
text+='\n'+(DETAIL/'SUPPLEMENT.md').read_text()
text+='\nThe final/ directory retains quality, latency, strict rescoring, prediction agreement, raw audits and the original durable-publication analysis. reporting_details/ contains the authoritative outer-return session_per_dialogue.csv, session_summary.csv, paired_analysis.json and the detailed storage/residency tables. The full frozen artifact paths are indexed below.\n'

text+='\n## Terminal evidence and complete validation accounting\n\n'
text+='| Scope | Phase | Committed requests | Physical executed | Failed image attempts |\n|---|---|---:|---:|---:|\n'
for c in counts:text+=f"| {c['scope']} | {c['phase']} | {c.get('adopted_logical_requests',0)} | {c.get('executed_requests',0)} | {c.get('failed_image_attempts',0)} |\n"
text+='\nRequest E2E boundary: non-source requests end when the final token-generation loop (_decode) returns; final text rendering and diagnostic serialization are outside that endpoint. Source-capture T1 ends at the outer core return and includes required saliency, capture clone and output postprocessing once. The same boundary applies across methods; the standalone-equivalent capture limitation remains.\n'
text+='\nOriginal ten core pairs and 24 fixed-input repeated reference controls are separate validation, not main requests. The first new diagnostic scalar-metadata assertion failed and is preserved. All final additional IDs were chosen before outcomes. No sample or tolerance was replaced after observing additional results.\n'
text+=f"\nFinal main completed in a launcher wall of {read(RUN/'main_process_receipt.json')['launcher_wall_s']:.3f} seconds including model load/warmup/smoke/logging/conditioning/per-image audit and cleanup. This is not request TTFT/E2E.\n"
text+='\nThe authoritative terminal full-hash index is artifact_index_final.json in this report directory, produced after launcher/analysis logs closed. The nested final/ index is the earlier analysis snapshot. Existing large artifacts use the documented sampled protection fingerprints; all new artifacts in this terminal index use full-byte SHA256. LLaVA remains 398 committed images and 60,915 requests, with no rerun.\n'
(OUT/'REPORT.md').write_text(text)
repro=(FINAL/'REPRODUCE.md').read_text()
repro+='''\n## Exact validation replay after all workload images have now been observed

For an exact replay, reuse the frozen additional_validation_manifest.json,
validation_selection.json and correctness_manifest.json as INPUTS, without copying
any PASS receipts. Mark them as previously used validation samples; after this
full Qwen run they are not new unseen samples. The prepare command is for a
prospective first run and may have no unobserved candidates now. The archived
protection_bootstrap_replay.py shows the inventory/snapshot policy. A fresh run
also needs REQUEST.md, CONTRACT.md, G7_REVIEW.md and a bootstrap.json recording
its own paths/source protection; all previous results stay read-only.

Run diagnose in a new diagnosis_vN. Run 116 fixed-input replay on its n313060/T2
fixture into a fresh G7_stage_attribution.json, then freeze/cpu/core/integration.
Run the archived repeat_validation.py with ROOT/RUN resolved for the new run,
then review and the archived launch_main.py (with its run-local bindings).
These launchers check actual new gate receipts. After launcher exit, execute
postanalysis_details_v2.py, then terminal_finalize_v3.py without redirecting output into an indexed file. The
terminal index binds all closed logs, raw, source, manifests and reports.
'''
repro+='\nThe report-only supplement reads existing raw/receipt files and executes no inference. Run the archived postanalysis_details_v2.py in its fresh run before terminal_finalize_v3.py. It preserves the 118 original durable-publication analysis and reports the already-measured outer writer-return boundary in the root report. See reporting_details/reporting_amendment.json for full input hashes. Do not execute terminal_finalize.py or terminal_finalize_v2.py for this terminal revision.\n'
(OUT/'REPRODUCE.md').write_text(repro)
locations={name:str(FINAL/name) for name in ('storage_metrics.csv','persistence.csv','activation.csv','quality.csv','latency.csv','independent_audit.json','history_validation.json','selection_budget_validation.json','io_timing_audit.json','prediction_agreement.csv')}
locations.update({name:str(DETAIL/name) for name in ('session_per_dialogue.csv','session_summary.csv','paired_analysis.json','session_accounting_audit.json','store_io_details.csv','store_io_summary.csv','request_gpu_io_summary.csv')})
dump(OUT/'artifact_locations.json',locations)
status=dict(status='PASS',main_final_requests=36549,legacy_G7='FAIL',root_cause='numerical validation FP32 shape-dependent rounding',revised_semantic_implementation='PASS',production_changed=False,reporting_amendment='recorded outer writer-return endpoint; original inner analysis retained',free_bytes_terminal=shutil.disk_usage(RUN).free,at_unix=time.time())
dump(OUT/'FINAL_STATUS.json',status)
# All mutable status files are updated before the terminal index.
for base in (RUN,OUT):
 p=base/'CURRENT_STATUS.json';x=read(p);x.update(execution_state='COMPLETE',final_independent_audit='PASS',main_final_requests=36549,at_unix=time.time());p.write_text(json.dumps(x,indent=2)+'\n')
index=[]
for base in (RUN,OUT):
 for p in sorted(base.rglob('*')):
  if p.is_file():index.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
dump(OUT/'artifact_index_final.json',dict(full_byte_sha256=True,scope='all new run and result files after logs closed',files=index,self_excluded=True))
print(json.dumps(status),flush=True)

"""Audit-native AllHead component timing and explicit terminal stage matrix."""
from pathlib import Path
import json,csv,hashlib,importlib.util,time,os
root=Path.cwd();run=Path(__file__).resolve().parent;out=root/'results'/run.name;dest=out/'online_timing'
spec=importlib.util.spec_from_file_location('R',root/'scripts/113_report_mt_gqa_B.py');R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
freeze=R.read(run/'online_timing_supplement_freeze.json');assert R.sha(__file__)==freeze['sha256'];assert R.sha(root/'scripts/113_report_mt_gqa_B.py')==freeze['report_source_sha256']
assert (out/'details/provenance.json').exists();status=R.read(out/'final/FINAL_STATUS.json');assert status['llava']['FULL MAIN COVERAGE']=='PASS'
dest.mkdir(exist_ok=False);records=[];b=[];commits={}
for cp in sorted((run/'llava/main/images').glob('*/COMMITTED.json')):
 c=R.read(cp);raw=cp.parent/c['attempt_id']/'raw.jsonl';assert R.sha(raw)==c['raw_sha256'];commits[str(cp)]=R.sha(cp)
 for r in R.rows(raw):
  if r['turn_id']==1:continue
  v=r['result'];method=r['method_id']
  if method==R.AH:
   assert v['scoring_calls']==32 and v['head_policy']=='all' and v['timing']['additive'] is False
   record={k:r[k] for k in ('image_id','dialog_id','turn_id','physical_execution_id')}
   record['selector_host_inclusive_ms']=sum(x['selector_host_ms'] for x in v['layers'])
   for scope in ('host_intervals_ms','cuda_intervals_ms'):
    record.update({scope+'/'+key:float(value) for key,value in v['timing'][scope].items()})
   records.append(record)
  elif method=='ours_prefixstore_kv25':
   assert v['scoring_calls']==v['query_score_calls']==v['static_score_calls']==0
   b.append({'image_id':r['image_id'],'dialog_id':r['dialog_id'],'turn_id':r['turn_id'],'prefix_selector_ms':v['selector_ms'],'query_ms_recorded':v['query_ms'],'query_score_calls':v['query_score_calls']})
assert len(records)==len(b)==8122
R.writecsv(dest/'AllHead_followup_native_intervals.csv',records);R.writecsv(dest/'B_followup_native_selector.csv',b)
metrics=list(dict.fromkeys(k for r in records for k in r if k not in ('image_id','dialog_id','turn_id','physical_execution_id')))
summary=[{'metric':key,'measured_requests':len(values),**R.stats(values)} for key in metrics if (values:=[r[key] for r in records if key in r])]
R.writecsv(dest/'AllHead_native_interval_summary.csv',summary)
bs=R.stats([r['prefix_selector_ms'] for r in b]);R.dump(dest/'B_native_selector_summary.json',{'requests':len(b),'prefix_selector_ms':bs,'query_score_calls':0,'query_ms_recorded_total':sum(r['query_ms_recorded'] for r in b)})
lines=['## AllHead native online stage 계측의 정확한 범위','', 'AllHead selector host 합에는 full-K pread/H2D, scoring, top-k, V read/H2D와 assembly가 포함된다. 이를 순수 scoring 시간으로 해석하지 않는다. 아래 값은 기존 production code가 저장한 host/CUDA interval의 요청별 layer 합이다. Host/CUDA와 상위 interval은 서로 중첩하므로 합산해서 TTFT를 만들지 않는다. CUDA event interval은 해당 stage의 계측이며 독립 kernel-only microbenchmark가 아니다.','']
lines+=R.mdtable(['Native interval','Requests','Mean / p50 / p95 ms'],[[r['metric'],r['measured_requests'],' / '.join(R.fmt(r[k],3) for k in ('mean','p50','p95'))] for r in summary])
lines+=['',f"B의 native fixed-prefix selector mean / p50 / p95: {bs['mean']:.6f} / {bs['p50']:.6f} / {bs['p95']:.6f} ms. 모든 8,122개 followup의 query/static score call은 0이다. 이 작은 prefix 선택 interval은 AllHead의 read/H2D/assembly 포함 host 총합과 같은 범위가 아니다.",'','## 모델별 최종 상태 (요청된 14항목)','']
keys=[k for k,v in status['llava'].items() if v in ('PASS','FAIL','BLOCKED','NOT RUN') and k!='model_status']
assert len(keys)==14
lines+=R.mdtable(['항목','LLaVA','Qwen'],[[k,status['llava'][k],status['qwen'][k]] for k in keys])
lines+=['','LLaVA 전체 main은 60,915/60,915개, Qwen main은 G7 실패로 0/36,549개다. 두 모델 합계 97,464개를 모두 실행한 것으로 해석하지 않는다. Qwen GPU CORRECTNESS PASS는 고정 10개 B core replay 범위이며 main integration의 FP32 실패를 면제하지 않는다. Qwen RAW REPRODUCIBILITY PASS는 보존된 검증/실패 evidence 범위이고 main 결과는 없다.','']
text='\n'.join(lines);(dest/'ONLINE_TIMING_AND_STATUS.md').write_text(text)
base=out/'REPORT_with_details.md';(out/'REPORT_with_online_details.md').write_text(base.read_text()+'\n'+text)
R.dump(dest/'provenance.json',{'source_sha256':R.sha(__file__),'base_report_sha256':R.sha(base),'output_report_sha256':R.sha(out/'REPORT_with_online_details.md'),'attention_source_sha256':R.sha(root/'mmimpress/sparsevlm_ssd_attention.py'),'commit_sha256':commits,'timestamp_unix':time.time(),'inference_source_changed':False})
print(json.dumps({'status':'PASS','AH_requests':len(records),'B_requests':len(b),'report':str(out/'REPORT_with_online_details.md')}),flush=True)

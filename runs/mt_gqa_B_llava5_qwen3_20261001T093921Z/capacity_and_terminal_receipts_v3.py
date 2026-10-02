"""Terminal capacity estimates and full artifact snapshot; execute after finalizer exits."""
from pathlib import Path
import csv,json,hashlib,shutil,time,os
root=Path.cwd();run=Path(__file__).resolve().parent;out=root/'results'/run.name;final=out/'final'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
assert (run/'final_analysis_complete.json').is_file(),'finalizer has not finished'
assert (out/'details/provenance.json').is_file(),'detail supplement has not finished'
assert (out/'online_timing/provenance.json').is_file(),'native timing supplement has not finished'
assert sha(Path(__file__))==json.loads((run/'terminal_supplement_v3_freeze.json').read_text())['sha256']
status=json.loads((final/'FINAL_STATUS.json').read_text());assert status['llava']['FULL MAIN COVERAGE']=='PASS'
free=shutil.disk_usage(run).free;reserve=30<<30;headroom=max(0,free-reserve);records=[]
for model in ('llava','qwen'):
 p=final/model/'storage_metrics.csv'
 if not p.exists():continue
 with p.open(newline='') as f:storage=list(csv.DictReader(f))
 for method in dict.fromkeys(r['method_id'] for r in storage):
  rs=[r for r in storage if r['method_id']==method];sizes=[int(r['allocated_bytes']) for r in rs];n=len(sizes);total=sum(sizes);average=total/n
  records.append({'model':model,'method_id':method,'images':n,'mean_allocated_bytes':average,'measured_workload_all_image_allocated_bytes':total,'free_bytes_at_terminal_snapshot':free,'minimum_reserve_bytes':reserve,'available_for_hypothetical_KV_only_bytes':headroom,'estimated_images_at_measured_mean_size':int(headroom//average),'all_measured_images_fit_this_KV_only_budget':total<=headroom,'provenance':rs[0]['provenance'],'qualification':'capacity estimate only; no simultaneous all-image population, eviction, saturation or hit-rate measurement; other files unchanged and no future writes assumed'})
with (out/'capacity_estimates.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
lines=['## 측정된 store bytes에 근거한 용량 추정','',f'종료 시점 여유 {free/2**30:.3f} GiB에서 30 GiB reserve를 남긴 가정상 KV 전용 여유는 {headroom/2**30:.3f} GiB다. 기존 raw·검증 payload·다른 파일은 그대로 두며 추가 쓰기가 없다는 가정이다. 이미지별 실제 allocated bytes를 사용했다.','', '| Method | 398-image stores GiB | Mean-size capacity estimate (images) | Fits KV-only headroom |','|---|---:|---:|---|']
for r in records:lines.append(f"| {r['model']} / {r['method_id']} | {r['measured_workload_all_image_allocated_bytes']/2**30:.3f} | {r['estimated_images_at_measured_mean_size']} | {r['all_measured_images_fit_this_KV_only_budget']} |")
lines+=['','FullLoad/AllHead 행은 같은 canonical bundle을 각 단독 배포의 footprint로 표시한 것이며 실제 물리 총량에 두 번 더하지 않는다. AllHead 미사용 probe sidecar도 포함한다. Mean-size capacity는 이미지 크기 분포가 같다는 환산이며, 실제 workload 전체의 sum과 fit 여부를 별도 표시했다. 현재 실험은 한 이미지씩 생성·감사·정리했으므로 이것은 실제 동시 적재·eviction·hit-rate 실험 결과가 아니다.','']
text='\n'.join(lines);(out/'CAPACITY.md').write_text(text)
base=(out/'REPORT_with_online_details.md').read_text();(out/'REPORT_with_capacity.md').write_text(base+'\n'+text)
# The immutable original seven-table report remains in final/REPORT.md.
shutil.copy2(out/'REPORT_with_capacity.md',out/'REPORT.md')
dump(out/'terminal_supplement_provenance.json',{'source_sha256':sha(Path(__file__)),'main_report_sha256':sha(final/'REPORT.md'),'detail_report_sha256':sha(out/'REPORT_with_details.md'),'online_detail_report_sha256':sha(out/'REPORT_with_online_details.md'),'integrated_report_sha256':sha(out/'REPORT.md'),'storage_csv_sha256':{m:sha(final/m/'storage_metrics.csv') for m in ('llava','qwen') if (final/m/'storage_metrics.csv').exists()},'observed_free_bytes':free,'timestamp_unix':time.time(),'main_inference_source_changed':False})
terminal={'status':'COMPLETE_WITH_QWEN_BLOCKED','llava_main_final_requests':60915,'qwen_main_final_requests':0,'Qwen':'BLOCKED_VALIDATION: unchanged FP32 G7','independent_audit':'PASS for completed LLaVA main and preserved Qwen failure evidence','report':str(out/'REPORT.md')}
(out/'CURRENT_STATUS.json').write_text(json.dumps(terminal,indent=2));(out/'PROGRESS.md').write_text('# 종료 상태\n\nLLaVA 60,915/60,915 main 완료·독립 감사 PASS. Qwen G7 BLOCKED, main 0/36,549. 최종 REPORT.md 참조.\n')
entries=[]
for base in (run,out):
 for p in sorted(base.rglob('*')):
  if p.is_file() and not p.is_symlink():entries.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
dump(out/'artifact_index_final.json',{'files':entries,'scope':'post-finalization full SHA snapshot including terminal logs and capacity report; excludes itself','authoritative_terminal_index':True})
print(json.dumps(terminal),flush=True)

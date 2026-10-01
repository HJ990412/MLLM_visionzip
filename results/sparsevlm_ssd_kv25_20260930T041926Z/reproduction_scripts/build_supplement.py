#!/usr/bin/env python3
"""Post-pilot supplemental analysis; no production/model/GPU imports or writes.

Run only after scripts/99 has finished. Reads immutable adopted raw and existing
receipts. Every output is a new exclusive file. This script is intentionally
outside the frozen production source set and never changes its measurements.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import shlex

METHODS=('recompute','fullload','sparsevlm_ssd_kv25_probe3','sparsevlm_ssd_kv25_allhead','ours_kv25')
LABELS=dict(zip(METHODS,('ReComp','FullLoad','SparseVLM-SSD-KV25-Probe3','SparseVLM-SSD-KV25-AllHead','Ours-KV25')))
PHASES=('smoke','gqa','mt')


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def canonical_sha(row):
    return hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def json_read(path,default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else ({} if default is None else default)


def csv_read(path):
    if not Path(path).is_file():return []
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def number(value):
    if value in (None,'','None','NOT RUN','NOT MEASURED','NOT_INSTRUMENTED'):return None
    try:
        result=float(value)
        return result if math.isfinite(result) else None
    except (ValueError,TypeError):return None


def mean(values):
    vals=[float(v) for v in values if number(v) is not None]
    return sum(vals)/len(vals) if vals else None


def fmt(value,digits=3,scale=1,missing='NOT MEASURED'):
    value=number(value)
    return missing if value is None else f'{value*scale:.{digits}f}'


def csv_dump(rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    if not fields:return ''
    buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=fields);writer.writeheader();writer.writerows(rows);return buf.getvalue()


def relative_link(path,label=None):
    return f'[{label or Path(path).name}]({Path(path).resolve()})'


class Evidence:
    def __init__(self):self.inputs={};self.failures=[];self.checks=0
    def record(self,path,digest=None):
        path=Path(path)
        if path.is_file():self.inputs[str(path.resolve())]={'sha256':digest or file_sha(path),'bytes':path.stat().st_size}
    def require(self,condition,name,detail=None):
        self.checks+=1
        if not condition:self.failures.append({'check':name,'detail':detail})


def raw_rows(path,evidence):
    """One streaming pass; verifies the source did not change during analysis."""
    path=Path(path)
    if not path.is_file():return
    before=path.stat();h=hashlib.sha256()
    with path.open('rb') as f:
        for line in f:
            h.update(line)
            if line.strip():yield json.loads(line)
    after=path.stat()
    evidence.require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'raw_immutable_during_analysis',str(path))
    evidence.record(path,h.hexdigest())


def nonfinite_count(value):
    if isinstance(value,float):return int(not math.isfinite(value))
    if isinstance(value,dict):return sum(nonfinite_count(x) for x in value.values())
    if isinstance(value,list):return sum(nonfinite_count(x) for x in value)
    return 0


def final_raw_diagnostics(run,audit,evidence):
    digests={};attempts=defaultdict(set);groups=defaultdict(Counter)
    for phase in PHASES:
        path=run/phase/'raw.jsonl'
        for row in raw_rows(path,evidence):
            rid=row.get('request_id');key=(phase,rid)
            evidence.require(key not in digests,'unique_final_request',rid)
            digests[key]=canonical_sha(row)
            cohort=row.get('dialog_id') or row['image_id'];attempts[(phase,cohort)].add(row['attempt_id'])
            g=groups[(phase,row['method_id'])];result=row.get('result',{})
            g['requests']+=1;g['hits' if row['turn_id']>1 else 't1']+=1
            flag=result.get('cap_reached')
            g['cap_flag_true' if flag is True else 'cap_flag_false' if flag is False else 'cap_flag_unrecorded']+=1
            tokens=result.get('generated_token_ids')
            length=len(tokens) if isinstance(tokens,list) else result.get('generated_token_count')
            g['generated_length_16']+=int(length==16)
            if row['method_id'].startswith('sparsevlm_ssd') and row['turn_id']>1:
                g['rater_eligible_hits']+=1
                fallback=result.get('rater_fallback')
                g['rater_fallback_true']+=int(fallback is True)
                g['rater_fallback_unrecorded']+=int(not isinstance(fallback,bool))
            g['nonfinite_numeric_fields']+=nonfinite_count(row)
            g['diagnostic_rows_in_adopted_raw']+=int(bool(result.get('diagnostic',False)))
        if path.is_file():
            expected=audit.get('phases',{}).get(phase,{}).get('raw_sha256')
            actual=evidence.inputs[str(path.resolve())]['sha256']
            evidence.require(expected==actual,'adopted_raw_matches_independent_audit',phase)
    rows=[]
    for (phase,method),counter in sorted(groups.items()):
        rows.append({'phase':phase,'method_id':method,**dict(counter),
                     'cap_note':'length 16 is reported separately; absent cap flag is not assumed false',
                     'rater_note':'fallback only applies to sparse cache-hit requests'})
    return digests,attempts,rows


def audit_adoption(run,final_digests,adopted_attempts,evidence):
    records=[];chosen={};matched=set()
    for phase in PHASES:
        for cohort_dir in sorted((run/phase/'images').glob('*')):
            if not cohort_dir.is_dir():continue
            first_success=None
            for attempt in sorted(cohort_dir.glob('attempt_*')):
                receipt_path=attempt/'independent_audit.json';receipt=json_read(receipt_path)
                evidence.record(receipt_path)
                is_pass=receipt.get('status')=='PASS'
                if is_pass and first_success is None:first_success=attempt.name
                raw_path=attempt/'raw.jsonl';count=0;row_ids=[];current_matches=0
                for row in raw_rows(raw_path,evidence):
                    count+=1;key=(phase,row['request_id']);row_ids.append(key)
                    if is_pass and attempt.name==first_success:
                        same=final_digests.get(key)==canonical_sha(row)
                        evidence.require(same,'final_row_equals_first_successful_attempt',[phase,cohort_dir.name,attempt.name,row['request_id']])
                        if same:matched.add(key);current_matches+=1
                if is_pass:
                    evidence.require(raw_path.is_file(),'passing_attempt_has_raw',str(attempt))
                    if raw_path.is_file():
                        evidence.require(receipt.get('raw_sha256')==evidence.inputs[str(raw_path.resolve())]['sha256'],
                                         'attempt_raw_matches_durable_audit_hash',str(attempt))
                    evidence.require(receipt.get('rows')==count,'attempt_receipt_row_count',str(attempt))
                events_path=attempt/'attempt_events.jsonl';event_count=None
                if events_path.is_file():
                    event_count=sum(1 for _ in raw_rows(events_path,evidence))
                selected=attempt.name in adopted_attempts.get((phase,cohort_dir.name),set())
                records.append({'phase':phase,'cohort_id':cohort_dir.name,'attempt_id':attempt.name,
                    'independent_status':receipt.get('status','NO PASS RECEIPT'),'raw_rows':count,'executed_event_rows':event_count,
                    'selected_in_final_raw':selected,'first_successful_attempt':first_success,
                    'final_rows_equal_this_first_success':current_matches,'raw_path':str(raw_path) if raw_path.exists() else '',
                    'attempt_events_path':str(events_path) if events_path.exists() else '',
                    'failed_or_partial_preserved':not is_pass,'cleanup_receipt_present':(attempt/'cleanup_receipt.json').is_file()})
            if first_success is not None:
                chosen[(phase,cohort_dir.name)]=first_success
                evidence.require(adopted_attempts.get((phase,cohort_dir.name),set())=={first_success},
                                 'cohort_uses_first_successful_attempt',[phase,cohort_dir.name])
    evidence.require(matched==set(final_digests),'every_adopted_row_has_durable_first_success',
                     {'matched':len(matched),'final':len(final_digests)})
    return records,chosen


def activation_tables(run,chosen,evidence):
    details=[]
    for (phase,cohort),attempt in sorted(chosen.items()):
        parent=run/phase/'images'/cohort/attempt
        for method in METHODS:
            path=parent/f'{method}_activation.json'
            if method=='recompute':
                details.append({'phase':phase,'cohort_id':cohort,'attempt_id':attempt,'method_id':method,
                                'measurement_status':'NOT_APPLICABLE','reason':'normal full-image inference has no stored-KV context activation'})
                continue
            if not path.is_file():
                details.append({'phase':phase,'cohort_id':cohort,'attempt_id':attempt,'method_id':method,'measurement_status':'NOT MEASURED'})
                continue
            data=json_read(path);evidence.record(path);meta=data.get('metadata',{})
            details.append({'phase':phase,'cohort_id':cohort,'attempt_id':attempt,'method_id':method,
                'measurement_status':'MEASURED','activation_ms':number(data.get('seconds'))*1000 if number(data.get('seconds')) is not None else None,
                'metadata_file_bytes':meta.get('file_bytes'),'host_metadata_tensor_bytes':meta.get('host_tensor_bytes'),
                'gpu_metadata_tensor_bytes':meta.get('gpu_tensor_bytes'),'visual_payload_resident_bytes':data.get('visual_payload_resident_bytes',meta.get('visual_payload_resident_bytes')),
                'hash_verified_bytes':meta.get('hash_verified_bytes'),'hash_verification_ms':number(meta.get('hash_verification_seconds'))*1000 if number(meta.get('hash_verification_seconds')) is not None else None,
                'hash_verification_status':meta.get('hash_verification','NOT RECORDED'),'persistence':data.get('persistence'),
                'receipt_path':str(path)})
    summary=[]
    keys=('activation_ms','metadata_file_bytes','host_metadata_tensor_bytes','gpu_metadata_tensor_bytes','visual_payload_resident_bytes','hash_verified_bytes','hash_verification_ms')
    for phase in PHASES:
        for method in METHODS:
            rows=[r for r in details if r['phase']==phase and r['method_id']==method]
            item={'phase':phase,'method_id':method,'activation_contexts':len(rows),
                  'measured_contexts':sum(r['measurement_status']=='MEASURED' for r in rows),
                  'measurement_status':'NOT_APPLICABLE' if method=='recompute' else 'MEASURED' if rows and all(r['measurement_status']=='MEASURED' for r in rows) else 'NOT MEASURED'}
            for key in keys:item[key+'_mean']=mean([r.get(key) for r in rows])
            item['aggregation_unit']='one adopted phase/dialogue-or-image/method context; repeated hit metadata excluded'
            summary.append(item)
    return details,summary


def timing_summary(breakdown):
    fields=('selector_host_ms','actual_attention_prefill_host_ms','actual_attention_prefill_cuda_ms',
            'actual_attention_decode_host_ms','actual_attention_decode_cuda_ms','rater_host_ms',
            'prefill_projection_q','prefill_projection_k','prefill_projection_v',
            'decode_projection_q','decode_projection_k','decode_projection_v',
            'payload_h2d_compute_bytes_subtotal','rater_visual_h2d_bytes')
    out=[]
    for phase in PHASES:
        for method in METHODS:
            rows=[r for r in breakdown if r.get('phase')==phase and r.get('method_id')==method and (number(r.get('turn_id')) or 0)>1]
            item={'phase':phase,'method_id':method,'hit_requests':len(rows),'stage_intervals_additive':False}
            for name in fields:
                vals=[number(r.get(name)) for r in rows]
                item[name+'_mean']=mean(vals);item[name+'_measured_requests']=sum(v is not None for v in vals)
            item['actual_attention_instrumentation']='MEASURED' if any(number(r.get('actual_attention_prefill_cuda_ms')) is not None for r in rows) else 'NOT_INSTRUMENTED'
            item['h2d_scope']='visual payload subtotal + separately recorded visual rater transfer; system/suffix/input total NOT MEASURED'
            out.append(item)
    return out


def protection_and_links(run,evidence):
    rows=[]
    candidates=['independent_audit.json','gpu_independent_audit.json','protection_rehash.json','protection_postrun.json',
                'protection_final.json','protected_artifacts_after.json','protection_after.json','validation.json',
                'reproduction_commands.json','source_freeze.json','source.diff','contract_freeze_v2.json']
    candidates+=sorted(p.name for p in run.glob('*protection*final*.json'))
    candidates+=sorted(p.name for p in run.glob('*post*protection*.json'))
    for name in dict.fromkeys(candidates):
        path=run/name
        if not path.is_file():continue
        evidence.record(path)
        data=json_read(path) if path.suffix=='.json' else {}
        rows.append({'artifact':name,'path':str(path),'status':data.get('status',data.get('passed','RECORDED')),
                     'finished_at':data.get('finished_at',data.get('completed_at',data.get('timestamp'))),
                     'sha256':evidence.inputs[str(path.resolve())]['sha256']})
    return rows


def gpu_controls(run,config,evidence):
    ref=config.get('gpu_gate',{})
    path=Path(ref.get('path',run/'gpu_full_01/gpu_validation.json'))
    if not path.is_absolute() and not path.is_file():path=run.parent.parent/path
    data=json_read(path);evidence.record(path)
    extended=data.get('extended_gpu',{})
    return path,{'GPU_CORRECTNESS':data.get('GPU_CORRECTNESS','NOT RUN'),
                 'controlled_same_selection':extended.get('controlled_same_selection','NOT RUN'),
                 'one_token_suffix':extended.get('one_token_suffix','NOT RUN'),
                 'unpatched_fullload_recovery':extended.get('unpatched_fullload_recovery','NOT RUN'),
                 'exception_methods':extended.get('exception_methods','NOT RUN'),
                 'legacy_QA_Token_diagnostic':extended.get('legacy_QA_Token_diagnostic',data.get('legacy_QA_Token_diagnostic','NOT RUN'))}


def setup_split_summary(run,out,evidence):
    candidates=[run/'setup_split_01/receipt.json',run/'setup_split.json',out/'setup_split.json']
    candidates+=sorted(run.glob('*/setup_split.json'))
    path=next((p for p in candidates if p.is_file()),None)
    if path is None:return None,[],{}
    data=json_read(path);evidence.record(path)
    schema='sparsevlm-ssd-fresh-t1-split-setup-diagnostic-v1'
    if data.get('schema_version')!=schema:
        return path,[],{'status':data.get('status','NOT RECORDED'),'scope':'unrecognized optional setup schema; numerical costs not inferred'}
    evidence.require(data.get('status')=='PASS','setup_split_diagnostic_status',str(path))
    evidence.require(data.get('independent_readback',{}).get('status')=='PASS','setup_split_independent_readback',str(path))
    common={'receipt_path':str(path),'status':data.get('status','NOT RUN'),
      'comparison_scope':data.get('comparison_scope'),'pilot_timing_replaced':False,
      'image_id':data.get('source_T1',{}).get('image_id'),'question_id':data.get('source_T1',{}).get('question_id')}
    rows=[]
    for key,footprint_key in [('AllHead_base_setup','AllHead_base_footprint'),('Probe3_incremental_sidecar_setup','Probe3_incremental_footprint')]:
        part=data.get(key,{});fp=data.get(footprint_key,{})
        rows.append(dict(common,measurement=key,wall_ms=part.get('wall_ms_fsync_inclusive'),
          fsync_inclusive=part.get('durability',{}).get('fsync_inclusive'),
          includes_T1=False,logical_bytes=fp.get('logical_bytes'),allocated_disk_bytes=fp.get('allocated_file_bytes'),
          scope=part.get('scope',part.get('source'))))
    for key,value,scope in [
      ('fresh_Probe3_setup_total_excluding_T1',data.get('fresh_Probe3_setup_total_ms_excluding_T1'),'actual sequential base+sidecar setup wall; one fresh T1 excluded'),
      ('fresh_Probe3_setup_phase_sum',data.get('fresh_Probe3_setup_phase_sum_ms'),'sum of two non-overlapping persistence phases; one fresh T1 excluded'),
      ('source_T1_TTFT',data.get('source_T1',{}).get('TTFT_ms'),'actual normal full-image T1; request time separate from setup'),
      ('source_T1_request_E2E',data.get('source_T1',{}).get('request_E2E_ms'),'actual normal full-image T1; TTFT is contained within E2E'),
      ('independent_readback_excluded',data.get('readback_ms_excluded'),'excluded from persistence phases'),
      ('hash_verification_excluded',data.get('hash_ms_excluded'),'excluded from persistence phases')]:
        rows.append(dict(common,measurement=key,wall_ms=value,fsync_inclusive=None,
          includes_T1=key.startswith('source_T1'),scope=scope))
    return path,rows,{'status':data.get('status'),'scope':data.get('comparison_scope'),
      'independent_readback':data.get('independent_readback'), 'limitations':data.get('limitations',[])}


def startup_failures(run,evidence):
    path=run/'smoke_execution.log'
    if not path.is_file():return []
    # Known short startup log, not active pilot raw or an unbounded execution scan.
    if path.stat().st_size>1<<20:return []
    text=path.read_text();error='RuntimeError: existing runner config differs; new run required'
    if error not in text:return []
    evidence.record(path)
    source=run.parent.parent/'scripts/98_eval_sparsevlm_ssd_kv25.py'
    code=source.read_text() if source.is_file() else ''
    config_marker="if previous!=json.loads(json.dumps(config)):raise RuntimeError('existing runner config differs; new run required')"
    load_marker='runner=V.load_runner()'
    before_requests=config_marker in code and load_marker in code and code.index(config_marker)<code.index(load_marker)
    evidence.record(source)
    evidence.require(before_requests,'startup_config_guard_precedes_model_and_requests',str(path))
    validation_path=run/'smoke/validation.json';validation=json_read(validation_path);evidence.record(validation_path)
    return [{'path':str(path),'status':'PRE_REQUEST_STARTUP_FAILURE','completed_requests':0 if before_requests else None,
      'error':error,'scope':'config equality guard before model load and any request; excluded from request retry counts',
      'cause':'GPU gate relative/absolute path string mismatch (root execution review); frozen config and source unchanged',
      'replacement_log':str(run/'smoke_execution_02.log'),
      'replacement_phase_status':validation.get('status','NOT RUN'),
      'evidence':'short traceback plus frozen runner guard order; later phase validation proves successful launch'}]


def failure_receipts(run,evidence):
    records=startup_failures(run,evidence)
    paths=list(run.glob('*/failure*.json'))+list(run.glob('*/images/*/attempt_*/*failure*.json'))
    for path in sorted(set(paths)):
        data=json_read(path);evidence.record(path)
        records.append({'path':str(path),'status':data.get('status'),'error':data.get('error'),
                        'completed_requests':data.get('completed_requests'),'scope':'preserved failure receipt'})
    # Pre-pilot diagnostic failures must stay separate from benchmark attempts.
    for path in sorted(run.glob('gpu_diagnostic*/gpu_validation.json')):
        data=json_read(path);evidence.record(path)
        records.append({'path':str(path),'status':data.get('GPU_CORRECTNESS'),'error':data.get('error'),
                        'scope':'pre-pilot correctness diagnostic, excluded from measured request counts'})
    for path in sorted(run.glob('*execution*.log')):
        records.append({'path':str(path),'status':'PRESERVED_LOG','bytes':path.stat().st_size,
                        'scope':'execution log; no failed-request count inferred from a log filename'})
    return records


def render(run,audit,accuracy,timing,activation,footprint,diagnostics,attempts,links,controls_path,controls,failures,setup_split,setup_rows,setup_meta,evidence):
    lines=['# SparseVLM-SSD-KV25 보충 분석','',
      '이 문서는 측정 완료 후 기존 raw/독립 audit/activation receipt를 읽어 만든 supplemental analysis다. 동결된 production code, 계약, raw, 선택/측정 조건은 변경하지 않았다. 기존 REPORT의 판정은 그대로이며, 아래 보충 검산이 실패하면 해당 사실을 별도로 표시한다.','',
      f'99 independent audit: **{audit.get("status","NOT RUN")}**; supplemental input/adoption checks: **{"PASS" if not evidence.failures else "FAIL"}**.','',
      '## Turn별 정확도','',
      'GQA/smoke는 기존 normalized equality 또는 gold-token-prefix scorer, MT는 기존 strict normalized equality scorer를 사용한다. 아래 값은 99가 raw에서 독립 재계산한 summary.csv를 그대로 표시한다.','',
      '| Dataset / 방법 | N | Hit N | 전체 % | Hit % | T1 % | T2 % | T3 % | T4 % | T5 % | T6 % |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in accuracy:
        vals=[fmt(r.get(k),2,100,missing='NOT RUN') for k in ['all_accuracy','hit_accuracy']+[f't{i}_accuracy' for i in range(1,7)]]
        lines.append(f'| {r["phase"]} / {LABELS.get(r["method_id"],r["method_id"])} | {r.get("requests",0)} | {r.get("hits",0)} | '+ ' | '.join(vals)+' |')
    if not accuracy:lines.append('| NOT RUN | 0 | 0 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |')
    lines+=['','## Selector와 실제 attention 측정 범위','',
      '아래 ms는 hit 요청당 평균이다. Selector host 범위는 layer별 selector interval의 합으로서 score/K acquisition/selection/assembly를 포함할 수 있다. Actual attention은 해당 eager call 범위이며 host는 launch/호출 구간, CUDA는 event 구간이다. 서로 겹치는 항목을 더해 TTFT를 만들지 않는다. 전체 LLM prefill·MLP·projection·decode E2E와 동일한 범위도 아니다. 기존 control arms의 분리된 actual-attention 계측이 없으면 NOT_INSTRUMENTED로 표시한다.','',
      '| Dataset / 방법 | Selector host ms | Prefill attention host/CUDA ms | Decode attention host/CUDA ms | 계측 |',
      '|---|---:|---:|---:|---|']
    for r in timing:
        lines.append(f'| {r["phase"]} / {LABELS[r["method_id"]]} | {fmt(r.get("selector_host_ms_mean"))} | {fmt(r.get("actual_attention_prefill_host_ms_mean"))} / {fmt(r.get("actual_attention_prefill_cuda_ms_mean"))} | {fmt(r.get("actual_attention_decode_host_ms_mean"))} / {fmt(r.get("actual_attention_decode_cuda_ms_mean"))} | {r["actual_attention_instrumentation"]} |')
    lines+=['','Projection별 실제 prefill/decode 호출 수는 `attention_interval_summary.csv`에 요청당 전체 layer 합으로 제공한다. GPU gate의 layer별 prefill은 각 q/k/v 1회이며, 측정 raw의 실제 counter도 독립 audit 대상이다. H2D visual payload subtotal과 rater visual bytes는 별도이며 system/suffix/processor를 포함한 전체 H2D로 해석하지 않는다.','',
      '## Metadata activation','',
      '같은 metadata_activation 값이 hit마다 raw에 반복되어도 한 번의 context activation으로 센다. 집계 단위는 adopted phase/dialogue-or-image/method context다. 같은 image의 별도 MT dialogue는 별도 context/provisioning일 수 있으므로 합치지 않는다. File bytes는 활성화 대상 metadata 파일 크기이며 실제 OS read 반환량으로 바꾸어 해석하지 않는다. Hash verified bytes 역시 metadata resident bytes와 다르다.','',
      '| Dataset / 방법 | Context N | Activation ms | Metadata file MB | Host tensor MB | GPU tensor MB | Hash verification ms | 상태 |',
      '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in activation:
        lines.append(f'| {r["phase"]} / {LABELS[r["method_id"]]} | {r["activation_contexts"]} | {fmt(r.get("activation_ms_mean"))} | {fmt(r.get("metadata_file_bytes_mean"),3,1e-6)} | {fmt(r.get("host_metadata_tensor_bytes_mean"),3,1e-6)} | {fmt(r.get("gpu_metadata_tensor_bytes_mean"),3,1e-6)} | {fmt(r.get("hash_verification_ms_mean"))} | {r["measurement_status"]} |')
    lines+=['','## 공유/독립 배포 저장공간','',
      '| 구분 | Apparent GB | 실제 allocated disk GB | 파일 수 |','|---|---:|---:|---:|']
    for r in footprint:lines.append(f'| {r["kind"]} | {fmt(r.get("apparent_bytes"),6,1e-9)} | {fmt(r.get("allocated_disk_bytes"),6,1e-9)} | {r.get("files","NOT MEASURED")} |')
    if not footprint:lines.append('| NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |')
    lines+=['','두 새 방법이 공유한 실제 canonical 파일은 한 번만 계상한다. Probe3 독립 배포는 sidecar를 포함하고 AllHead 독립 배포는 probe sidecar를 제외한다. Read-only GQA serving은 CACHE-HIT REEVALUATION이며 전체 persistence/session은 NOT_REMEASURED다. MT fresh T1의 shared bundle persistence 측정은 존재하더라도 AllHead base와 Probe3-only 생성 비용이 자동 분리되지는 않는다.','',
      '별도 setup split receipt: '+(relative_link(setup_split) if setup_split else '**NOT SEPARATELY MEASURED**')+'. 별도 진단을 수행했다면 원 5-arm timing에 소급 합산하지 않는다.','',
      '### 별도 fresh setup 측정','',
      '아래 값은 한 이미지의 새로운 정상 T1에서 얻은 canonical tensor를 사용하는 별도 serializer 진단이다. Production pilot의 shared-bundle persistence, cache-hit TTFT, session 비용을 대체하지 않는다. Base를 먼저 쓰고 동일 T1의 CPU K에서 Probe3 raw sidecar를 이어서 생성했으며 이 임시 setup residency를 serving residency로 해석하지 않는다.','',
      '| 단계 | Wall ms | Fsync inclusive | T1 포함 | 범위 |','|---|---:|---|---|---|']
    for r in setup_rows:
        lines.append(f'| {r["measurement"]} | {fmt(r.get("wall_ms"))} | {r.get("fsync_inclusive") if r.get("fsync_inclusive") is not None else "N/A"} | {r.get("includes_T1")} | {r.get("scope") or "별도 receipt 참조"} |')
    if not setup_rows:lines.append('| NOT SEPARATELY MEASURED | NOT MEASURED | N/A | N/A | 유효한 별도 receipt 없음 |')
    if setup_meta.get('limitations'):
        lines+=['','추가 범위 제한: '+' '.join(str(x) for x in setup_meta['limitations'])]
    lines+=['',
      '## Cap / rater fallback / attempt 채택','',
      '| Dataset / 방법 | N / Hit N | Explicit cap true / unrecorded | Generated length=16 | Rater fallback / eligible hits | Nonfinite fields |',
      '|---|---:|---:|---:|---:|---:|']
    for r in diagnostics:
        lines.append(f'| {r["phase"]} / {LABELS[r["method_id"]]} | {r.get("requests",0)} / {r.get("hits",0)} | {r.get("cap_flag_true",0)} / {r.get("cap_flag_unrecorded",0)} | {r.get("generated_length_16",0)} | {r.get("rater_fallback_true",0)} / {r.get("rater_eligible_hits",0)} | {r.get("nonfinite_numeric_fields",0)} |')
    startup=[r for r in failures if r.get('status')=='PRE_REQUEST_STARTUP_FAILURE']
    if startup:
        lines+=['',f'요청 전 startup failure {len(startup)}건을 별도로 보존했다. `smoke_execution.log`의 frozen config equality 오류는 model load/요청 실행 전에 발생하여 실행 요청은 0개다. Root 실행 검토상 GPU gate relative/absolute 경로 문자열 차이였으며 source/config를 바꾸지 않고 정확한 고정 경로로 `smoke_execution_02.log` 실행을 이어갔다. 이는 이미지별 요청 retry가 아니다. 최종 smoke 성공 여부는 phase validation receipt로 확인한다.']
    lines+=['','Cap flag 미기록은 false로 간주하지 않는다. 길이 16 관측은 EOS가 마지막인 경우를 포함할 수 있어 명시적 cap 판정과 분리한다.','',
      f'Attempt directories: {len(attempts)}; PASS receipt가 없는 failed/partial 보존 attempt: {sum(bool(r["failed_or_partial_preserved"]) for r in attempts)}; 최종 raw에 채택된 attempt: {sum(bool(r["selected_in_final_raw"]) for r in attempts)}. `attempt_adoption.csv`와 `supplement_validation.json`은 cohort별 첫 PASS receipt의 exact raw 행과 최종 adopted 행을 비교한다. 좋은 점수/시간을 기준으로 재선택하지 않는다.','',
      '| 보존 failure/log | 상태 | 범위 |','|---|---|---|']
    for r in failures:lines.append(f'| {relative_link(r["path"])} | {r.get("status","NOT RECORDED")} | {r["scope"]} |')
    if not failures:lines.append('| 없음 | NO FAILURE RECEIPT OBSERVED | 미기록 실패를 0이라고 추론하지 않음 |')
    lines+=['','## Controlled projection/position 및 legacy QA 진단','',
      f'원본: {relative_link(controls_path)}. 세부값은 새 `controlled_validation_summary.json`에 원 receipt에서 복사했다. 동일 선택 주입, one-token suffix, 예외 해제 후 FullLoad 복원, legacy QA projection 차이는 timing-excluded correctness/diagnostic이며 5-arm latency와 섞지 않는다.','',
      '## 재현 및 보호 근거','',
      '| Artifact | 상태 | 시간 |','|---|---|---|']
    for r in links:lines.append(f'| {relative_link(r["path"])} | {r["status"]} | {r.get("finished_at") or "NOT RECORDED"} |')
    lines+=['','기존 protection_rehash가 pilot 전에 완료된 경우 그것만으로 post-pilot 보호 완료를 주장하지 않는다. 최종 보호 receipt의 범위(stat/hash/새 파일 allowlist)를 함께 확인한다. Qwen GPU 및 전체 MT-GQA/MT-VQA는 이 pilot에 포함되지 않는다.','']
    commands=json_read(run/'reproduction_commands.json')
    if commands:
        lines+=['고정 실행/재개 명령(원 `reproduction_commands.json`):','','```bash',f'cd {shlex.quote(commands.get("cwd",str(run.parent.parent)))}']
        env=commands.get('environment',{})
        if env:lines.append('export '+' '.join(f'{k}={shlex.quote(str(v))}' for k,v in env.items()))
        for key in ('cpu_new','gpu_new_receipt','resume_exact','audit_new_output_only'):
            if isinstance(commands.get(key),list):lines+=['# '+key,shlex.join(commands[key])]
        lines+=['```','','새 GPU receipt/output 경로를 사용하고 기존 artifact overwrite 거부를 유지한다.']
    lines+=['','이 보충 문서는 최종 REPORT의 dataset/quality/TTFT 판정을 변경하지 않는다. Supplemental validation이 FAIL이면 해당 불일치를 먼저 해결·보고해야 하며 READY를 자동 승인하지 않는다.','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--results-dir',type=Path)
    parser.add_argument('--allow-incomplete',action='store_true',help='explicit partial/NOT RUN supplement; never upgrades readiness')
    args=parser.parse_args();run=args.run_dir.resolve();out=(args.results_dir or run.parent.parent/'results'/run.name).resolve()
    names=('SUPPLEMENT.md','activation_summary.csv','activation_receipts.csv','turn_accuracy.csv',
           'attention_interval_summary.csv','attempt_adoption.csv','request_diagnostics.csv','store_footprint_summary.csv',
           'controlled_validation_summary.json','preserved_failures.json','setup_split_summary.csv','supplement_validation.json','supplement_provenance.json')
    if any((out/n).exists() for n in names):raise FileExistsError('supplement artifacts already exist; use a new output directory')
    audit_path=out/'independent_audit.json'
    if not audit_path.is_file():raise RuntimeError('run scripts/99 first; independent_audit.json is missing')
    audit=json_read(audit_path)
    complete=audit.get('status')=='PASS' and all(audit.get('phases',{}).get(p,{}).get('pilot_status')=='VALID' for p in PHASES)
    if not complete and not args.allow_incomplete:raise RuntimeError('final audit/pilot incomplete; wait or explicitly use --allow-incomplete')
    evidence=Evidence();evidence.record(audit_path)
    for name in ('summary.csv','io_timing_memory_breakdown.csv','runner_config.json','store_footprint.json'):
        evidence.record((out if name.endswith('.csv') else run)/name)
    accuracy=csv_read(out/'summary.csv');breakdown=csv_read(out/'io_timing_memory_breakdown.csv')
    final,attempt_sets,diagnostics=final_raw_diagnostics(run,audit,evidence)
    attempts,chosen=audit_adoption(run,final,attempt_sets,evidence)
    activation_details,activation=activation_tables(run,chosen,evidence)
    timing=timing_summary(breakdown)
    footprint=[{'kind':k,**v} for k,v in json_read(run/'store_footprint.json').get('totals',{}).items()]
    links=protection_and_links(run,evidence);config=json_read(run/'runner_config.json')
    controls_path,controls=gpu_controls(run,config,evidence);failures=failure_receipts(run,evidence)
    setup_split,setup_rows,setup_meta=setup_split_summary(run,out,evidence)
    status='PASS' if not evidence.failures else 'FAIL'
    validation={'schema_version':'sparsevlm-ssd-kv25-supplement-v1','status':status,
        'scope':'post-pilot supplemental analysis, separate from frozen production code',
        'final_independent_audit_status':audit.get('status'),'complete_five_arm_pilots':complete,
        'checks':evidence.checks,'failures':evidence.failures,'adopted_rows':len(final),
        'attempt_directories':len(attempts),'source_raw_modified':False,'GPU_used':False}
    outputs={
      'SUPPLEMENT.md':render(run,audit,accuracy,timing,activation,footprint,diagnostics,attempts,links,controls_path,controls,failures,setup_split,setup_rows,setup_meta,evidence),
      'activation_summary.csv':csv_dump(activation),'activation_receipts.csv':csv_dump(activation_details),
      'turn_accuracy.csv':csv_dump([{k:v for k,v in r.items() if k in ('phase','method_id','requests','hits','all_accuracy','hit_accuracy') or (k.startswith('t') and k.endswith('_accuracy'))} for r in accuracy]),
      'attention_interval_summary.csv':csv_dump(timing),'attempt_adoption.csv':csv_dump(attempts),
      'request_diagnostics.csv':csv_dump(diagnostics),'store_footprint_summary.csv':csv_dump(footprint),
      'controlled_validation_summary.json':json.dumps({'source':str(controls_path),'scope':'outside pilot timing',**controls},indent=2,allow_nan=False)+'\n',
      'preserved_failures.json':json.dumps(failures,indent=2,allow_nan=False)+'\n',
      'setup_split_summary.csv':csv_dump(setup_rows),
      'supplement_validation.json':json.dumps(validation,indent=2,allow_nan=False)+'\n'}
    provenance={'created_at_utc':datetime.now(timezone.utc).isoformat(),'analysis_script':str(Path(__file__).resolve()),
      'analysis_script_sha256':file_sha(__file__),'production_source_changed':False,'original_artifacts_overwritten':False,
      'inputs':evidence.inputs,'outputs':{name:hashlib.sha256(text.encode()).hexdigest() for name,text in outputs.items()},
      'note':'supplement_provenance.json excludes its own hash to avoid a recursive digest'}
    outputs['supplement_provenance.json']=json.dumps(provenance,indent=2,allow_nan=False)+'\n'
    out.mkdir(parents=True,exist_ok=True)
    for name,text in outputs.items():
        with (out/name).open('x') as f:f.write(text)
    print(json.dumps({'status':status,'outputs':len(outputs),'adopted_rows':len(final),'production_modified':False}))
    return 0 if status=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())

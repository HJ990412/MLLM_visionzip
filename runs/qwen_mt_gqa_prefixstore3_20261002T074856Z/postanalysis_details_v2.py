"""Read-only raw reconstruction of reporting details and outer setup accounting.

No inference, gate, input, prediction, or original report is changed. The original
113/118 report uses service_persistence_ms, ending after durable publication. This
supplement also counts the already measured outer writer return, as 110/111 do.
"""
from pathlib import Path
from collections import defaultdict
import csv, hashlib, json, math, time
import numpy as np

ROOT=Path('/home/dblab/hj/mllm_v2')
RUN=Path(__file__).resolve().parent
OUT=ROOT/'results'/RUN.name
FINAL=OUT/'final'
DEST=OUT/'reporting_details'
METHODS=('recompute','fullload','qwen_ours_prefixstore_kv25')
LABEL=dict(zip(METHODS,('ReComp','FullLoad','Ours B')))
B=METHODS[-1]
def read(p): return json.loads(p.read_text())
def rows(p): return list(csv.DictReader(p.open()))
def dump(p,x):
    with p.open('x') as f: json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def writecsv(p,data):
    keys=list(dict.fromkeys(k for r in data for k in r))
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
def stats(v):
    a=np.asarray(v,dtype=float)
    return dict(mean=float(a.mean()),p50=float(np.percentile(a,50)),p95=float(np.percentile(a,95))) if len(a) else dict.fromkeys(('mean','p50','p95'),'NOT_MEASURED')
def table(h,rs): return '\n'.join(['| '+' | '.join(h)+' |','|'+'|'.join(['---']*len(h))+'|',*['| '+' | '.join(map(str,r))+' |' for r in rs]])
def fmt(v):return f'{v:.3f}'

assert read(RUN/'analysis_process_receipt.json')['returncode']==0
assert read(FINAL/'completion.json')['final_logical_main_requests']==36549
DEST.mkdir(exist_ok=False)
stores=[];groups=defaultdict(list);input_hashes={}
for cp in sorted((RUN/'main/images').glob('*/COMMITTED.json')):
    com=read(cp);folder=cp.parent/com['attempt_id'];iid=cp.parent.name
    for m in METHODS[1:]:
        ps=folder/f'{m}_persistence.json';ac=folder/f'{m}_activation.json';mt=folder/f'{m}_meta.json'
        p,a,meta=map(read,(ps,ac,mt))
        for path in (ps,ac,mt):input_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        width=2*meta['num_layers']*meta['num_kv_heads']*meta['head_dim']*2
        n=meta['visual_count'];k=(n+3)//4 if m==B else n
        content=sum(v['size'] for name,v in p['files'].items() if name.endswith(('/k.bin','/v.bin')))
        structural=p['files']['structural_kv.bin']['size']
        metadata=sum(v['size'] for name,v in p['files'].items() if name.endswith('.json'))
        padding=content-k*width
        assert padding==meta['padding_rows']*width>=0
        assert content+structural+metadata==p['file_logical_bytes']
        assert p['outer_writer_wall_ms']==p['charged_setup_before_activation_ms']>=p['service_persistence_ms']
        if m==B:assert padding==0 and meta['stored_rows']==k
        s=dict(image_id=iid,method_id=m,source_dialog_id=p['source']['dialog_id'],
               physical_build_event_id=p['event_id'],activation_event_id=a['event_id'],
               N=n,stored_k=k,original_content_bytes=n*width,valid_content_bytes=k*width,
               padding_bytes=padding,structural_system_boundary_bytes=structural,
               metadata_and_completion_file_bytes=metadata,total_logical_bytes=p['file_logical_bytes'],
               allocated_bytes=p['allocated_bytes'],os_write_bytes=p['os_write_bytes'],os_write_calls=p['os_write_calls'],
               metadata_resident_bytes=a['metadata_resident_bytes'],
               payload_fd_count_derived=len(meta['files']),kernel_fd_memory_bytes='NOT_MEASURED',
               activation_read_bytes=a['activation_io']['bytes'],activation_preads=a['activation_io']['preads'],
               service_persistence_ms=p['service_persistence_ms'],outer_writer_wall_ms=p['outer_writer_wall_ms'],
               receipt_return_tail_ms=p['outer_writer_wall_ms']-p['service_persistence_ms'],activation_ms=a['activation_ms'],
               capture_score_ms=p['capture_score_ms'],capture_clone_ms=p['capture_clone_ms'],
               source_post_generation_tail_ms=p['post_generation_capture_tail_ms'],
               completion_publication_ms=p['completion_publication_ms'],**p['timing_ms'])
        stores.append(s)
    for line in (folder/'raw.jsonl').open():
        r=json.loads(line);v=r['result']
        for scope in (f"T{r['turn_id']}",'all',*(['followup'] if r['turn_id']>1 else [])):
            d={key:v.get(key) for key in ('peak_gpu_allocated_bytes','peak_gpu_reserved_bytes','padding_rows_read','extra_valid_visual_rows','metadata_resident_bytes','h2d_kv_bytes','gpu_cache_kv_bytes')}
            if d['padding_rows_read'] is not None:d['padding_read_bytes']=d['padding_rows_read']*57344
            groups[(r['method_id'],scope)].append(d)
summary=[]
fields=[k for k,v in stores[0].items() if isinstance(v,(int,float))]
for m in METHODS[1:]:
    sr=[s for s in stores if s['method_id']==m]
    z=dict(method_id=m,images=len(sr))
    for key in fields:z.update({key+'_'+k:v for k,v in stats([s[key] for s in sr]).items()})
    summary.append(z)
request_summary=[]
for (m,scope),rs in groups.items():
    for field in sorted({k for r in rs for k in r}):
        values=[r[field] for r in rs if r.get(field) is not None]
        request_summary.append(dict(method_id=m,scope=scope,metric=field,measured_count=len(values),**stats(values)))
lookup={(s['image_id'],s['method_id']):s for s in stores}
sessions=[]
for old in rows(FINAL/'session_per_dialogue.csv'):
    m=old['method_id'];s=lookup.get((old['image_id'],m));req=float(old['request_E2E_sum_ms'])
    source=bool(s and old['dialog_id']==s['source_dialog_id']);setup=s['outer_writer_wall_ms']+s['activation_ms'] if s else 0.
    sessions.append(dict(image_id=old['image_id'],dialog_id=old['dialog_id'],method_id=m,source_dialogue=source,
        request_E2E_sum_ms=req,actual_build_attribution_ms=s['outer_writer_wall_ms'] if source else 0.,
        actual_activation_attribution_ms=s['activation_ms'] if source else 0.,
        physical_stream_session_ms=req+(setup if source else 0.),derived_standalone_session_ms=req+setup,
        persistence_endpoint='outer persist_qwen return (measured)',non_source_capture_increment='NOT_MEASURED'))
session_summary=[]
for m in METHODS:
    rs=[s for s in sessions if s['method_id']==m];z=dict(method_id=m,dialogues=len(rs))
    for key in ('physical_stream_session_ms','derived_standalone_session_ms'):
        z.update({key+'_'+k:v for k,v in stats([r[key] for r in rs]).items()})
    session_summary.append(z)
attributed=sum(s['actual_build_attribution_ms']+s['actual_activation_attribution_ms'] for s in sessions)
physical=sum(s['outer_writer_wall_ms']+s['activation_ms'] for s in stores)
assert math.isclose(attributed,physical,abs_tol=1e-5)
images=sorted({s['image_id'] for s in stores});index={x:i for i,x in enumerate(images)}
sample=np.random.default_rng(1234).integers(0,len(images),size=(10000,len(images)))
def ci(values):
    totals=np.zeros((len(images),2))
    for im,v in values:totals[index[im]]+=[v,1]
    sampled=totals[sample].sum(axis=1);est=sampled[:,0]/sampled[:,1];lo,hi=np.percentile(est,[2.5,97.5])
    return dict(point=float(totals[:,0].sum()/totals[:,1].sum()),ci_low=float(lo),ci_high=float(hi),resamples=10000,seed=1234,images=398,weighting='pooled sums/counts with paired image cluster resampling')
paired=[];sl={(s['dialog_id'],s['method_id']):s for s in sessions}
for base in METHODS[:2]:
    for key in ('physical_stream_session_ms','derived_standalone_session_ms'):
        values=[(s['image_id'],sl[s['dialog_id'],B][key]-s[key]) for s in sessions if s['method_id']==base]
        paired.append(dict(baseline=base,direction='B-minus-baseline',scope='per_dialogue',metric=key,**ci(values)))
    values=[(im,lookup[im,B]['outer_writer_wall_ms']-(lookup[im,base]['outer_writer_wall_ms'] if base!='recompute' else 0.)) for im in images]
    paired.append(dict(baseline=base,direction='B-minus-baseline',scope='image_macro_setup',metric='outer_writer_wall_ms',**ci(values)))
for name,data in [('store_io_details.csv',stores),('store_io_summary.csv',summary),('request_gpu_io_summary.csv',request_summary),('session_per_dialogue.csv',sessions),('session_summary.csv',session_summary),('paired_analysis.csv',paired)]:writecsv(DEST/name,data)
dump(DEST/'paired_analysis.json',dict(rows=paired,original_durable_endpoint_analysis=str(FINAL/'paired_analysis.json')))
dump(DEST/'session_accounting_audit.json',dict(status='PASS',attributed_setup_ms=attributed,actual_physical_event_setup_ms=physical,physical_builds=796,activation_events=796,endpoint='outer writer return; native activation endpoint',production_or_raw_changed=False))
dump(DEST/'reporting_amendment.json',dict(status='PASS',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_hashes=input_hashes,
    reason='113/118 use inner durable publication endpoint; disclose both and include recorded outer writer-return wall in final root session tables, matching Qwen 110/111 accounting',
    inference_rerun=False,validation_gate_changed=False,original_outputs_preserved=True,at_unix=time.time()))
sm={s['method_id']:s for s in summary};ss={s['method_id']:s for s in session_summary}
lat={(r['method_id'],r['scope']):r for r in rows(FINAL/'latency.csv')}
text='\n## Measured I/O, residency and reporting boundary details\n\n'
text+='The root report uses the recorded outer writer return for persistence and session accounting. The original final/ report preserves the inner durable-publication endpoint; its wrapper receipt-return tail was outside that endpoint. This is a reporting-only amendment after main completion. No inference, correctness criterion, source freeze or raw measurement changed. reporting_details/paired_analysis.json supplies the matching outer-endpoint CIs.\n\n'
text+=table(['Method','Durable publication mean ms','Outer return mean/p50/p95 ms','Receipt-return tail mean ms'],[[LABEL[m],fmt(sm[m]['service_persistence_ms_mean']),'/'.join(fmt(sm[m]['outer_writer_wall_ms_'+x]) for x in ('mean','p50','p95')),fmt(sm[m]['receipt_return_tail_ms_mean'])] for m in METHODS[1:]])
text+='\n\n'+table(['Method','T1 TTFT mean/p50/p95 ms','T2 TTFT mean/p50/p95 ms','T3 TTFT mean/p50/p95 ms'],[[LABEL[m],*['/'.join(fmt(float(lat[m,t]['ttft_ms_'+k])) for k in ('mean','p50','p95')) for t in ('T1','T2','T3')]] for m in METHODS])
text+='\n\n'+table(['Method','Original content MB','Stored valid MB','Padding MB','Structure/system/boundary MB','Metadata+seal file MB','Resident metadata MB','Payload FDs (derived)','Activation read MB / preads'],[[LABEL[m],*[fmt(sm[m][k+'_mean']/1e6) for k in ('original_content_bytes','valid_content_bytes','padding_bytes','structural_system_boundary_bytes','metadata_and_completion_file_bytes','metadata_resident_bytes')],fmt(sm[m]['payload_fd_count_derived_mean']),fmt(sm[m]['activation_read_bytes_mean']/1e6)+' / '+fmt(sm[m]['activation_preads_mean'])] for m in METHODS[1:]])
text+='\n\nMetadata residency is the native recursive Python object-size estimate of decoded meta; it excludes kernel FD memory and other process allocations. The payload FD count is derived from the verified file inventory/open path (one FD per payload), not separate kernel memory measurement. Resident metadata/FDs persist for an active image store; visual KV payloads are reread on every hit. Structure includes system and vision-boundary rows in one sidecar; those overlapping categories are not summed again. FullLoad padding is serialized and read but discarded before H2D; B has zero content padding.\n\n'
component_keys=('prefix_slice_serialization_ms','permutation_ms','kv_materialize_ms','kv_repack_ms','ssd_write_ms','fsync_ms','completion_publication_ms','capture_score_ms','capture_clone_ms','source_post_generation_tail_ms')
text+=table(['Recorded component (ms)','FullLoad mean/p50/p95','Ours B mean/p50/p95'],[[key,*['/'.join(fmt(sm[m][key+'_'+x]) for x in ('mean','p50','p95')) for m in METHODS[1:]]] for key in component_keys])
text+='\n\nThese are native named intervals, some nested: do not sum them into a critical path. kv_materialize includes CPU materialization/D2H; an isolated D2H-only timer and isolated serialization-only timer are NOT_MEASURED. Prefix slicing/serialization and write/fsync retain their native boundaries. Capture score/clone/post-generation tail is already charged once in source T1 E2E and is not added to persistence. Native per-payload full hashes are reused; no second full-payload checksum pass is claimed.\n\n'
text+=table(['Method','Followup GPU peak allocated / reserved MB (mean)','Padding read MB (mean)','Extra valid content rows (mean)'],[[LABEL[m],fmt(float(lat[m,'followup']['peak_gpu_allocated_bytes_mean'])/1e6)+' / '+fmt(float(lat[m,'followup']['peak_gpu_reserved_bytes_mean'])/1e6),*[next((fmt(x['mean']/1e6 if key=='padding_read_bytes' else x['mean']) if isinstance(x['mean'],(int,float)) else x['mean'] for x in request_summary if x['method_id']==m and x['scope']=='followup' and x['metric']==key),'NOT_MEASURED') for key in ('padding_read_bytes','extra_valid_visual_rows')]] for m in METHODS])
text+='\n\nGPU peaks are request-scoped torch allocator observations in the shared process, including model/coexisting contexts; they are not incremental KV size or independently deployed memory footprints. Per-scope mean/p50/p95 and measured counts are in request_gpu_io_summary.csv. ReComp has no cached visual-KV read; unrelated image-file I/O is not zero. All 36,549 outputs ended with EOS; no empty output or generation cap occurred.\n'
(DEST/'SUPPLEMENT.md').write_text(text)
print(json.dumps(dict(status='PASS',physical_setup_ms=physical,outer_persistence_mean={m:sm[m]['outer_writer_wall_ms_mean'] for m in METHODS[1:]},sessions=session_summary)),flush=True)

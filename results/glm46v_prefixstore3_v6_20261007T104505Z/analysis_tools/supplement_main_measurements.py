#!/usr/bin/env python3
"""Post-completion native output, source-tail, and setup accounting supplement.

Only adopted raw is read. This script never modifies inference or source data.
It supplies measurements omitted from the compact primary report tables.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import json
import math
import statistics


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def quantile(values,q):
    a=sorted(values)
    if not a:return 'NOT_MEASURED'
    index=(len(a)-1)*q
    lo=math.floor(index);hi=math.ceil(index)
    return a[lo]*(hi-index)+a[hi]*(index-lo) if lo!=hi else a[lo]


def summary(values):
    return dict(measured_count=len(values),mean=statistics.mean(values) if values else 'NOT_MEASURED',
                p50=quantile(values,.5),p95=quantile(values,.95))


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader();writer.writerows(rows)


def main(run,out):
    completion=json.loads((run/'main/COMPLETED.json').read_text())
    if completion.get('status')!='PASS' or completion['requests']!=36549:
        raise ValueError('complete main is required')
    out.mkdir(parents=True,exist_ok=False)
    methods=('recompute','fullload','ours_prefixstore_kv25')
    counts=defaultdict(Counter);values=defaultdict(lambda:defaultdict(list))
    source_values=defaultdict(lambda:defaultdict(list));source_counts=Counter()
    setup_values=defaultdict(lambda:defaultdict(list));setup_counts=Counter();evidence=[]
    t1=defaultdict(dict)
    for commit in sorted((run/'main/images').glob('*/COMMITTED.json')):
        receipt=json.loads(commit.read_text())
        owner=commit.parent/receipt['attempt_id']
        raw=Path(receipt['raw_path']);setup=Path(receipt['setup_path'])
        if raw.resolve()!=owner.resolve()/'raw.jsonl' or setup.resolve()!=owner.resolve()/'setup_events.jsonl':
            raise ValueError('adopted path ownership differs')
        if sha(raw)!=receipt['raw_sha256'] or sha(setup)!=receipt['setup_sha256']:
            raise ValueError('adopted hash differs')
        evidence.append(dict(image_id=receipt['image_id'],raw_sha256=sha(raw),setup_sha256=sha(setup)))
        for line in raw.open():
            r=json.loads(line);m=r['method_id'];turn=r['turn_id']
            scopes=['all',f'T{turn}']+(['followup'] if turn>1 else [])
            for scope in scopes:
                c=counts[m,scope];c['requests']+=1
                c['empty_answers']+=not bool(r['prediction'])
                c['cap_reached_without_EOS']+=bool(r['truncated'])
                c['native_malformed_box']+=r['answer_extraction']['status']=='MALFORMED_NATIVE_BOX'
                c['native_closed_box']+=r['answer_extraction']['status']=='CLOSED_NATIVE_BOX'
                c['generated_think_open_requests']+=r.get('generated_think_open_count',0)>0
                c['generated_think_close_requests']+=r.get('generated_think_close_count',0)>0
                c['generated_think_open_tokens']+=r.get('generated_think_open_count',0)
                c['generated_think_close_tokens']+=r.get('generated_think_close_count',0)
                c['first_token_is_native_answer_boundary']+=r['first_token_id']==151361
                c['requested_thinking_disabled']+=r.get('enable_thinking') is False
                c['vision_forwards']+=r['vision_calls']
                c['decode_store_reads']+=r['decode_store_reads']
                c['query_score_calls']+=r['online_query_score_calls']
                for field,val in [('generated_token_count',len(r['generated_token_ids'])),
                                  ('peak_gpu_allocated_GiB',r['peak_gpu_allocated_bytes']/2**30),
                                  ('peak_gpu_reserved_GiB',r['peak_gpu_reserved_bytes']/2**30),
                                  ('page_cache_conditioning_ms_outside_request',r.get('page_cache_conditioning_ms_outside_request'))]:
                    if isinstance(val,(int,float)):values[m,scope][field].append(val)
            if turn==1:t1[r['dialog_id']][m]=r['generated_token_ids']
            if r['source_T1_capture']:
                source_counts[m]+=1
                for field,val in r['timing_ms'].items():
                    if isinstance(val,(int,float)):source_values[m][field].append(val)
        for line in setup.open():
            e=json.loads(line);m=e['method_id'];kind=e['event_kind']
            setup_counts[m,kind]+=1
            if kind=='build':
                for field,val in e.get('timing_ms',{}).items():
                    if isinstance(val,(int,float)):setup_values[m][field].append(val)
    output_rows=[];request_rows=[];source_rows=[];setup_rows=[]
    for m in methods:
        for scope in ('T1','T2','T3','all','followup'):
            output_rows.append(dict(method_id=m,scope=scope,**counts[m,scope],
                                   first_lexical_answer_latency='NOT_MEASURED',
                                   TTFT_endpoint='first native generated output token; includes native control boundary'))
            for field,data in values[m,scope].items():
                request_rows.append(dict(method_id=m,scope=scope,field=field,**summary(data)))
        for field,data in source_values[m].items():
            source_rows.append(dict(method_id=m,source_images=source_counts[m],field=field,**summary(data),
                                    accounting='nested in source request E2E; never added to persistence again'))
        for field,data in setup_values[m].items():
            setup_rows.append(dict(method_id=m,field=field,**summary(data),
                                   accounting='nested in measured outer writer return; never summed into it again'))
    if sum(counts[m,'all']['requests'] for m in methods)!=36549:
        raise ValueError('wrong adopted count')
    agree=sum(len({tuple(ids) for ids in rows.values()})==1 and len(rows)==3 for rows in t1.values())
    if len(t1)!=4061:raise ValueError('wrong T1 dialogue count')
    write_csv(out/'native_output_audit.csv',output_rows)
    write_csv(out/'request_measurement_details.csv',request_rows)
    write_csv(out/'source_capture_details.csv',source_rows)
    write_csv(out/'writer_component_details.csv',setup_rows)
    result=dict(status='PASS',requests=36549,dialogues=4061,images=398,
         T1_generated_ID_agreement_dialogues=agree,T1_generated_ID_agreement_ratio=agree/4061,
         source_capture_counts=dict(source_counts),setup_event_counts=[dict(method_id=m,event_kind=k,count=n) for (m,k),n in setup_counts.items()],
         source_script_sha256=sha(Path(__file__)),evidence=evidence,
         caveats=['Observed thinking control tokens are not censored or retried; requested native mode remains enable_thinking=False.',
                  'Native first-generated-token TTFT is not separately measured first lexical final-answer latency.',
                  'Importance component memory is sampled live allocation; request CUDA allocated/reserved peaks are separately measured.',
                  'Non-source standalone capture overhead is NOT_MEASURED.',
                  'Raw/evidence write and benchmark conditioning are outside service request timing.'])
    (out/'receipt.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','requests','T1_generated_ID_agreement_dialogues')}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.run.resolve(),a.output.resolve())

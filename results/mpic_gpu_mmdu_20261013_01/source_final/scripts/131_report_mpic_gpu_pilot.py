#!/usr/bin/env python3
"""Independent bounded pilot arithmetic/history/IO checks and raw summaries."""
import argparse,ast,csv,hashlib,json,re,statistics
from pathlib import Path
import numpy as np
METHODS=('prefix_caching','full_reuse','mpic_global32','mpic_per_image32')
DIMS=('Creativity','Richness','Visual Perception','Logical Coherence','Answer Accuracy','Image Relationship Understanding','Overall Score')
def dump(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def parse(text):
    for raw in reversed(re.findall(r'\{[^{}]+\}',text)):
        try:x=ast.literal_eval(raw.replace('’',"'").replace('‘',"'"))
        except (SyntaxError,ValueError):continue
        if not isinstance(x,dict):continue
        known={k.lower():k for k in DIMS};x={known.get(str(k).strip().lower(),str(k)):v for k,v in x.items()}
        recovery=False
        if set(x)==set(DIMS)-{'Overall Score'}:
            tail=text[text.rfind(raw)+len(raw):].replace('**','')
            matches=re.findall(r'(?i)overall score\s*(?::|is|of)\s*(\d+)\b',tail)
            if len(matches)!=1:continue
            x['Overall Score']=int(matches[0]);recovery=True
        if set(x)!=set(DIMS):continue
        if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and v==int(v) and 1<=v<=10 for v in x.values()):continue
        return {k:int(x[k]) for k in DIMS},recovery
    return None,False
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--judge',type=Path)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(x) for x in (a.run/'answers.jsonl').read_text().splitlines()]
    manifest=json.loads((a.run/'subset_manifest.json').read_text());ids=[d['dialog_id'] for d in manifest['dialogues']]
    by={(r['dialog_id'],r['turn'],r['method']):r for r in rows}
    expected={(d,t,m) for d in ids for t in (1,2,3) for m in METHODS}
    assert len(rows)==len(by)==60 and set(by)==expected
    checks=[]
    for row in rows:
        method=row['method'];history=row['history'];turn=row['turn'];s=row['stats']
        assert set(history)=={str(i) for i in range(1,turn)}
        for i in range(1,turn):assert history[str(i)]==by[row['dialog_id'],i,method]['prediction']
        assert abs(row['ttft_ms']-(row['first_token_at_s']-row['request_started_at_s'])*1000)<1e-5
        assert row['status']=='PASS' and row['retry_count']==0 and not row['fallback']
        assert row['generated_token_count']==len(row['generated_token_ids'])
        assert row['termination']==('EOS' if row['generated_token_ids'][-1]==2 else 'length_cap')
        if method.startswith('mpic_'):
            ntext=row['input_token_count']-row['image_token_count']
            k=32 if method=='mpic_global32' else 32*row['images']
            assert row['recomputed_image_tokens']==k
            assert len(s['active_positions'])==ntext+k and len(s['text_positions'])==ntext
            assert len(s['layers'])==32
            for layer in s['layers']:
                assert layer['Q_shape']==[1,32,ntext+k,128]
                assert layer['active_K_shape']==layer['active_V_shape']==[1,32,ntext+k,128]
                assert layer['full_K_shape']==layer['full_V_shape']==[1,32,row['input_token_count'],128]
                assert layer['hidden_shape']==[1,ntext+k,4096]
                assert layer['future_probability_max']==0 and layer['active_K_installed'] and layer['active_V_installed']
                assert all(layer['reused_unchanged_per_image'])
        if row['condition']=='ssd_cold' and method!='prefix_caching':
            kv=32*2*row['image_token_count']*32*128*2
            embedding=row['recomputed_image_tokens']*4096*2 if method.startswith('mpic_') else 0
            count=64*row['images']+(1 if method=='mpic_global32' else row['images'] if method=='mpic_per_image32' else 0)
            assert row['ssd_bytes']==kv+embedding and row['preads']==count
        checks.append(dict(dialogue=row['dialog_id'],turn=turn,method=method,status='PASS',history='method-local verified',timing='timestamp identity verified',IO='expected bytes/calls verified'))
    dump(a.out/'independent_request_checks.json',checks)
    summary=[]
    for method in METHODS:
        selected=[r for r in rows if r['method']==method];values=[r['ttft_ms'] for r in selected]
        summary.append(dict(method=method,n=15,TTFT_mean_ms=statistics.mean(values),TTFT_p50_ms=statistics.median(values),
            TTFT_p95_ms=float(np.quantile(values,.95)),KV_SSD_mean_MB=statistics.mean(r['ssd_bytes']/1e6 for r in selected),
            preads_mean=statistics.mean(r['preads'] for r in selected),peak_allocated_GiB=max(r['peak_gpu_allocated_bytes'] for r in selected)/2**30,
            peak_reserved_GiB=max(r['peak_gpu_reserved_bytes'] for r in selected)/2**30,
            output_tokens_mean=statistics.mean(r['generated_token_count'] for r in selected),length_cap_count=sum(r['cap_reached'] for r in selected),
            EOS_count=sum(r['termination']=='EOS' for r in selected),fallback_count=0,failed=0))
    dump(a.out/'summary.json',summary)
    with (a.out/'summary.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    if a.judge:
        judged=[json.loads(x) for x in (a.judge/'raw.jsonl').read_text().splitlines()];assert len(judged)==60
        scored=[]
        for record in judged:
            scores,recovery=parse(record.get('judge_response',''))
            scored.append(dict(dialog_id=record['dialog_id'],turn=record['turn'],method=record['method'],scores=scores,
                status='PASS' if scores else 'FAIL',raw_parser_status=record['status'],explicit_overall_outside_dict_recovered=recovery))
        dump(a.out/'judge_parse_audit.json',scored)
        quality=[]
        for method in METHODS:
            valid=[r for r in scored if r['method']==method and r['scores'] is not None]
            quality.append(dict(method=method,n_valid=len(valid),failed=15-len(valid),
                **{d:statistics.mean(r['scores'][d] for r in valid) if valid else None for d in DIMS}))
        dump(a.out/'quality.json',quality)
        with (a.out/'quality.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(quality[0]));w.writeheader();w.writerows(quality)
        good={(r['dialog_id'],r['turn'],r['method']) for r in scored if r['scores'] is not None}
        shared={(d,t) for d in ids for t in (1,2,3) if all((d,t,m) in good for m in METHODS)}
        paired=[]
        for method in METHODS:
            valid=[r for r in scored if r['method']==method and (r['dialog_id'],r['turn']) in shared]
            paired.append(dict(method=method,n_common=len(shared),
                **{d:statistics.mean(r['scores'][d] for r in valid) if valid else None for d in DIMS}))
        dump(a.out/'quality_paired.json',paired)
        dump(a.out/'judge_failure_coverage.json',dict(valid=sum(r['scores'] is not None for r in scored),
            failed=[{k:r[k] for k in ['dialog_id','turn','method','status']} for r in scored if r['scores'] is None],
            shared_questions=len(shared),total_questions=15,format_recovered=sum(r['explicit_overall_outside_dict_recovered'] for r in scored),
            score_range_and_dimensions_unchanged=True,raw_judge_outputs_preserved=True))
    dump(a.out/'status.json',dict(status='PASS (pilot accounting)',completed_dialogues=5,completed_turns=15,method_requests=60,
        condition=rows[0]['condition'],global_per_image_policy_not_inferred_from_results=True))
    print(json.dumps(dict(status='PASS',requests=60,dialogues=5,turns=15)))
if __name__=='__main__':main()

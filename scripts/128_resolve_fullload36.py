#!/usr/bin/env python3
"""Resolve exactly the 36 archived requests with hash-checked reconstructed inputs."""
import argparse,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
from transformers import AutoTokenizer
import torch

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tensor_hash(value):
    h=hashlib.sha256();h.update(str(value.dtype).encode('ascii'))
    h.update(json.dumps(list(value.shape)).encode('ascii'));h.update(value.numpy().tobytes());return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prior',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    wanted=json.loads(a.prior.read_text())['possible_unverified_requests'];assert len(wanted)==36
    snapshot='/home/dblab/.cache/huggingface/hub/models--llava-hf--llava-v1.6-vicuna-7b-hf/snapshots/c916e6cdcd760b4cecd1dd4907f84ac649f93b23'
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    viscfg=json.loads((ROOT/'runs/visdial_turn1_piggyback_e2e_ttft/main_seed1234/config.json').read_text())
    visindex=Path(viscfg['index']);assert sha(visindex)==viscfg['index_sha256']
    dialogs={d['dialog_id']:d for d in json.loads(visindex.read_text())['dialogs']}
    raw={}
    for path in sorted({x['path'] for x in wanted}):
        ids={x['record_ordinal'] for x in wanted if x['path']==path}
        with (ROOT/path).open() as f:
            for n,line in enumerate(f,1):
                if n in ids:raw[path,n]=(json.loads(line),hashlib.sha256(line.encode()).hexdigest())
                if n>=max(ids):break
    records=[]
    with (a.out/'original_requests.jsonl').open('x') as original:
        for item in wanted:
            row,line_sha=raw[item['path'],item['record_ordinal']]
            original.write(json.dumps(row,ensure_ascii=False)+'\n')
            provenance='independent GQA current-question-only'
            if 'visdial' in item['path']:
                d=dialogs[row['dialog_id']];turn=row['turn_id'];lines=[f"Image caption: {d['caption']}",'']
                for t in d['turns'][:turn-1]:lines.extend([f"Q{t['turn_id']}: {t['question']}",f"A{t['turn_id']}: {t['gold_answer']}"])
                lines.extend(['',f"Current question Q{turn}: {d['turns'][turn-1]['question']}"])
                prompt='USER: <image>\n'+'\n'.join(lines)+'\nAnswer the current question concisely. ASSISTANT:'
                provenance='archived VisDial gold_teacher_forced policy from hash-checked index; not new evaluation policy'
            else:
                prompt=f"USER: <image>\n{row['question']} Answer the question using a single word or phrase. ASSISTANT:"
            ids=tok(prompt,return_tensors='pt').input_ids[0];loc=(ids==32000).nonzero(as_tuple=True)[0]
            prompt_match=hashlib.sha256(prompt.encode()).hexdigest()==row.get('prompt_sha256')
            suffix=ids[int(loc[0])+1:] if len(loc)==1 else torch.empty(0,dtype=torch.int64)
            suffix_match=tensor_hash(suffix)==row.get('suffix_ids_sha256')
            status='PASS' if prompt_match and suffix_match else 'NOT VERIFIED'
            classification='오류 조건에 해당하지 않음' if status=='PASS' and len(suffix)!=1 else (
                '오류 조건에 해당하나 영향 미확인' if status=='PASS' else '자료 부족으로 NOT VERIFIED')
            records.append(dict(**item,verification_status=status,classification=classification,prompt=prompt,
                suffix_input_ids=suffix.tolist(),suffix_tokens=len(suffix),prompt_sha_matches=prompt_match,
                binary_tensor_sha_matches=suffix_match,source_record_sha256=line_sha,input_provenance=provenance,
                recorded_ssd_bytes=row.get('ssd_read_bytes'),recorded_preads=row.get('ssd_preads'),
                historical_numerical_correctness='not inferred from new GPU success'))
            records[-1]['previous_status']=records[-1].pop('status')
            records[-1]['status']=status
    summary={'requests':len(records),'non_trigger':sum(r['classification']=='오류 조건에 해당하지 않음' for r in records),
        'trigger_impact_confirmed':0,'trigger_impact_unverified':sum(r['classification']=='오류 조건에 해당하나 영향 미확인' for r in records),
        'NOT_VERIFIED':sum(r['verification_status']=='NOT VERIFIED' for r in records),
        'minimum_verified_suffix':min(r['suffix_tokens'] for r in records if r['verification_status']=='PASS'),
        'reason_prior_unknown':'old hash uses dtype+shape+binary tensor bytes, not a JSON list hash',
        'visdial_index_sha256':sha(visindex),'records':records}
    with (a.out/'classification.json').open('x') as f:json.dump(summary,f,indent=2,ensure_ascii=False)
    print(json.dumps({k:v for k,v in summary.items() if k!='records'},ensure_ascii=False))
if __name__=='__main__':main()

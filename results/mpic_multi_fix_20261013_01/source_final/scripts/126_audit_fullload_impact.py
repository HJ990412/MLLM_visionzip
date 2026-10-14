#!/usr/bin/env python3
"""Bounded, read-only old-result trigger scan; never rerun old experiments."""
import argparse, collections, json, hashlib, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--max-mb',type=int,default=64);a=ap.parse_args()
    files=sorted((ROOT/'runs').glob('**/raw.jsonl'))
    # One file per original run family first. Full coverage is explicitly NOT
    # claimed; this prevents an impact audit reading tens of GB by accident.
    byrun=collections.defaultdict(list)
    for path in files:byrun[path.relative_to(ROOT/'runs').parts[0]].append(path)
    ordered=[paths[0] for _,paths in sorted(byrun.items())]
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('/home/dblab/.cache/huggingface/hub/models--llava-hf--llava-v1.6-vicuna-7b-hf/snapshots/c916e6cdcd760b4cecd1dd4907f84ac649f93b23',local_files_only=True)
    scanned=[];hits=[];possible=[];budget=a.max_mb*(1<<20);consumed=0
    for path in ordered:
        if consumed>=budget:break
        family=path.relative_to(ROOT/'runs').parts[0].lower()
        if 'qwen' in family or 'glm' in family:
            scanned.append(dict(path=str(path.relative_to(ROOT)),records_read=0,stored_fullload_requests=0,
                suffix_tokens=[],unknown_suffix=0,classification='NOT APPLICABLE: separate Qwen/GLM native implementation'))
            continue
        records=eligible=0;sizes=[];unknown=0
        with path.open() as f:
            for line in f:
                if records>=40:break
                consumed+=len(line.encode());records+=1
                if consumed>budget:break
                row=json.loads(line);result=row.get('result',row)
                model=str(row.get('model_id',row.get('model','')))
                if model and 'qwen' in model.lower() or model and 'glm' in model.lower():continue
                method=row.get('method_id',row.get('method_key',row.get('method')))
                if method not in ('fullload','FullLoad'):continue
                # T1 often carries a FullLoad label but actually runs pixels.
                stored=(result.get('vision_forward_count')==0 or 'cache_hit' in str(row.get('request_path','')))
                if not stored:continue
                eligible+=1
                suffix=row.get('suffix_input_ids',result.get('suffix_input_ids'))
                n=len(suffix) if suffix is not None else row.get('suffix_tokens')
                prompt=row.get('prompt')
                if n is None and not prompt and isinstance(row.get('question'),str):
                    candidate=f"USER: <image>\n{row['question']} Answer the question using a single word or phrase. ASSISTANT:"
                    if hashlib.sha256(candidate.encode()).hexdigest()==row.get('prompt_sha256'):prompt=candidate
                if n is None and isinstance(prompt,str):
                    ids=tokenizer(prompt).input_ids
                    if ids.count(32000)==1:
                        suffix=ids[ids.index(32000)+1:]
                        digest=hashlib.sha256(json.dumps(suffix,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
                        if row.get('suffix_ids_sha256') in (None,digest):n=len(suffix)
                if n is None:
                    unknown+=1;possible.append(dict(path=str(path.relative_to(ROOT)),record_ordinal=records,
                        request_id=row.get('request_id',row.get('logical_request_id')),turn_id=row.get('turn_id'),
                        status='NOT VERIFIED: stored FullLoad request without recoverable suffix length'))
                    continue
                sizes.append(n)
                if n==1:
                    hits.append(dict(path=str(path.relative_to(ROOT)),request_id=row.get('request_id',row.get('logical_request_id')),
                        suffix_tokens=n,io=result.get('io'),status='CONFIRMED_TRIGGER',
                        numerical_impact='NOT VERIFIED unless actual installed KV/logits are available'))
                if records>=40:break
        scanned.append(dict(path=str(path.relative_to(ROOT)),records_read=records,stored_fullload_requests=eligible,
                            suffix_tokens=sizes,unknown_suffix=unknown))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:
        json.dump(dict(scope='bounded historical samples; first file per run family, <=40 rows/file and byte cap',
            bytes_read=consumed,byte_cap=budget,candidate_raw_files=len(files),candidate_run_families=len(byrun),
            scanned=scanned,confirmed_trigger_requests=hits,
            possible_unverified_requests=possible,
            unscanned_run_families=sorted(set(byrun)-{Path(r['path']).parts[1] for r in scanned}),
            full_corpus_impact='NOT VERIFIED',
            fixed_template_main='9 original LLaVA requests previously rechecked: suffix34/56/74, no suffix1 trigger',
            recommended_reruns='only verified suffix1 stored requests and sessions with unvalidated changed prefixes; no automatic full rerun'),f,indent=2)
        f.write('\n')
    print(json.dumps(dict(files_scanned=len(scanned),confirmed_triggers=len(hits),full_corpus='NOT VERIFIED')))
if __name__=='__main__':main()

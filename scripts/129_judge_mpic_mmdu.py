#!/usr/bin/env python3
"""Local Qwen visual judge with MPIC appendix-D rubric; explicitly adapted."""
import argparse,ast,csv,hashlib,json,os,re,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch
from PIL import Image
from mmimpress.qwen25.runner import Qwen25Runner,MODEL_ID,CHECKPOINT_REVISION
DIMENSIONS=('Creativity','Richness','Visual Perception','Logical Coherence','Answer Accuracy','Image Relationship Understanding','Overall Score')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def parse_scores(text):
    aliases={d.lower():d for d in DIMENSIONS}
    for raw in reversed(re.findall(r'\{[^{}]+\}',text)):
        try:value=ast.literal_eval(raw.replace('’',"'").replace('‘',"'"))
        except (ValueError,SyntaxError):continue
        if not isinstance(value,dict):continue
        result={aliases.get(str(k).strip().lower(),str(k)):v for k,v in value.items()}
        if set(result)!=set(DIMENSIONS):continue
        if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and int(v)==v and 1<=v<=10 for v in result.values()):continue
        return {k:int(result[k]) for k in DIMENSIONS}
    return None
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-new-tokens',type=int,default=1024);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    status=json.loads((a.run/'status.json').read_text());assert status['completed_dialogues']==5 and status['method_requests']==60
    manifest=json.loads((a.run/'subset_manifest.json').read_text());dialogs={d['dialog_id']:d for d in manifest['dialogues']}
    rows=[json.loads(s) for s in (a.run/'answers.jsonl').read_text().splitlines()];assert len(rows)==60
    paper=(ROOT/'papers/mpic.md').read_text();section=paper.split('# D EVALUATION PROMPT',1)[1]
    rubric=section[section.index('“You are')+1:section.rindex('.”')+1]
    rubric=re.sub(r'\n\s*16\s*\n','\n',rubric).replace('_{_','{').replace('_}_','}')
    (a.out/'rubric.txt').write_text(rubric)
    config=dict(judge_model=MODEL_ID,revision=CHECKPOINT_REVISION,precision='NF4 double quant, BF16',attention='sdpa',
        generation='greedy',max_new_tokens=a.max_new_tokens,score_range=[1,10],dimensions=DIMENSIONS,
        rubric_source='MPIC v2 appendix D local manuscript',rubric_sha256=sha(a.out/'rubric.txt'),
        source_answers_sha256=sha(a.run/'answers.jsonl'),source_manifest_sha256=sha(a.run/'subset_manifest.json'),
        difference='local 7B quantized visual judge, not an exact reproduction of paper ChatGPT judge',
        method_names_hidden=True,aggregation='unweighted mean of 15 turn scores/method; macro dialogue mean equals it',
        failed_evaluations='reported, not assigned zero or replaced by EM',automatic_retry=False)
    dump(a.out/'config.json',config)
    if not torch.cuda.is_available():dump(a.out/'status.json',dict(status='NOT VERIFIED',reason='CUDA unavailable'));raise SystemExit(3)
    runner=Qwen25Runner(max_new_tokens=a.max_new_tokens).load();model,processor=runner.model,runner.processor
    dump(a.out/'runtime.json',runner.runtime_fingerprint())
    results=[]
    for row in rows:
        dialog=dialogs[row['dialog_id']];turn=dialog['turns'][row['turn']-1];count=turn['active_images']
        pictures=[];content=[{'type':'text','text':rubric}]
        for i,im in enumerate(dialog['images'][:count],1):
            assert sha(im['path'])==im['sha256']
            with Image.open(im['path']) as image:pictures.append(image.convert('RGB').resize((336,336),Image.Resampling.LANCZOS))
            content.extend([{'type':'text','text':f'Image {i}:'},{'type':'image','image':pictures[-1]}])
        payload=f"\nUser question:\n{turn['question']}\nReference answer:\n{turn['reference_answer']}\nAssistant response to evaluate:\n{row['prediction']}"
        content.append({'type':'text','text':payload})
        prompt=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
        request=dict(dialog_id=row['dialog_id'],turn=row['turn'],method=row['method'],prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            candidate_sha256=hashlib.sha256(row['prediction'].encode()).hexdigest())
        try:
            inputs=processor(text=[prompt],images=pictures,return_tensors='pt',padding=True)
            inputs={k:v.to(model.device) if torch.is_tensor(v) else v for k,v in inputs.items()}
            torch.cuda.synchronize();started=time.perf_counter()
            with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=a.max_new_tokens,do_sample=False,use_cache=True)
            torch.cuda.synchronize();tokens=out[0,inputs['input_ids'].shape[1]:]
            text=processor.tokenizer.decode(tokens.tolist(),skip_special_tokens=True,clean_up_tokenization_spaces=False)
            scores=parse_scores(text)
            result=dict(**request,status='PASS' if scores is not None else 'FAIL',scores=scores,judge_response=text,
                generated_tokens=len(tokens),cap_reached=len(tokens)==a.max_new_tokens,elapsed_ms=(time.perf_counter()-started)*1000)
            del inputs,out,tokens
        except Exception as exc:
            result=dict(**request,status='FAIL',scores=None,error=repr(exc),traceback=traceback.format_exc())
        results.append(result)
        with (a.out/'raw.jsonl').open('a') as f:f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
        print(json.dumps(dict(completed=len(results),dialogue=row['dialog_id'],turn=row['turn'],method=row['method'],status=result['status'])),flush=True)
    summary=[]
    for method in sorted({r['method'] for r in results}):
        valid=[r for r in results if r['method']==method and r['scores'] is not None]
        summary.append(dict(method=method,successful=len(valid),failed=15-len(valid),
            **{d:sum(r['scores'][d] for r in valid)/len(valid) if valid else None for d in DIMENSIONS}))
    dump(a.out/'summary.json',summary)
    with (a.out/'quality.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['method','successful','failed',*DIMENSIONS]);w.writeheader();w.writerows(summary)
    dump(a.out/'status.json',dict(status='PASS (adapted judge only)' if all(r['status']=='PASS' for r in results) else 'FAIL (some evaluations)',
        requests=60,successful=sum(r['status']=='PASS' for r in results),failed=sum(r['status']!='PASS' for r in results),paper_judge_reproduction='NOT VERIFIED'))
if __name__=='__main__':main()

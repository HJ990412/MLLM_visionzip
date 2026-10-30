#!/usr/bin/env python3
"""Compare the minimally ported official evaluator to independent fixtures/raw."""
import argparse,copy,contextlib,hashlib,io,json,sys,types,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mmimpress.mtvqa25k.common import sha,dump,read
from mmimpress.mtvqa25k.scorer import Annotations,REVISION,reference_scores,independent_score,normalization

def annotation(qid,answers):
    return {'question_id':qid,'image_id':1,'answer_type':'other','question_type':'fixture',
            'answers':[{'answer_id':i+1,'answer':a,'answer_confidence':'yes'} for i,a in enumerate(answers)]}

def validate(output):
    output.mkdir(parents=True,exist_ok=False)
    fixtures=[]
    for m,expected in [(0,0),(1,.3),(2,.6),(3,.9),(4,1),(10,1)]:
        fixtures.append((f'matches_{m}','cat',['cat']*m+['dog']*(10-m),expected,'cat'))
    for name,pred,gold,normalized in [('number_word','Two','2','2'),('article','The cat','cat','cat'),
        ('punctuation','red, blue!','red blue','red blue'),('decimal','3.14','3.14','3.14'),
        ('contraction','dont',"don't","don't"),('whitespace',' \ttraffic\nlight  ','traffic light','traffic light'),
        ('multiword','a tennis racket','tennis racket','tennis racket'),('digit_comma','100,978','100978','100978')]:
        fixtures.append((name,pred,[gold]*9+['different'],1,normalized))
    fixtures.extend([('unanimous_case_branch','Cat',['cat']*10,0,'Cat'),
                     ('unanimous_number_branch','two',['2']*10,0,'two'),
                     ('empty_denominator','',['cat']*10,0,''),
                     ('no_first_word_extraction','cat on a table',['cat']*9+['dog'],0,'cat on table')])
    anns=[annotation(i+1,f[2]) for i,f in enumerate(fixtures)]
    predictions={i+1:f[1] for i,f in enumerate(fixtures)}
    ref,aggregate=reference_scores(anns,predictions)
    details=[]
    for i,(name,pred,gold,expected,norm) in enumerate(fixtures,1):
        check=independent_score(pred,anns[i-1]);assert abs(check['vqa_score']-expected)<1e-12
        assert abs(ref[i]-expected)<1e-12 and check['normalized_answer']==norm
        details.append({'fixture':name,'prediction':pred,'annotation':anns[i-1],'official':ref[i],**check})
    # Load ORIGINAL Python2 source via stdlib print-syntax conversion independently
    # of the checked-in two-line port, compare outputs and annotation side effects.
    from lib2to3.refactor import RefactoringTool
    original=(ROOT/'mmimpress/mtvqa25k/vendor/vqaEval.py.original').read_text()
    translated=str(RefactoringTool(['lib2to3.fixes.fix_print']).refactor_string(original,'vqaEval.py'))
    namespace={};exec(compile(translated,'upstream_reference','exec'),namespace)
    upstream=namespace['VQAEval'](Annotations(anns),types.SimpleNamespace(qa={q:{'answer':s} for q,s in predictions.items()}),n=10)
    with contextlib.redirect_stdout(io.StringIO()):upstream.evaluate()
    assert upstream.evalQA=={q:100*s for q,s in ref.items()} and upstream.accuracy==aggregate
    dump(output/'synthetic_fixtures.json',details)
    archive=ROOT/'data/mt_vqa_v2_reconstructed_25k/source/v2_Annotations_Val_mscoco.zip'
    with zipfile.ZipFile(archive) as z:official=json.load(z.open('v2_mscoco_val2014_annotations.json'))['annotations']
    byqid={a['question_id']:a for a in official}
    raw=ROOT/'results/mt_vqa_v2_generated_4arm_20260922T064727Z/raw.jsonl'
    records=[json.loads(line) for line in raw.open() if line.strip()]
    real_anns=[];real_pred={};real=[]
    for index,r in enumerate(records,1):
        qid=int(r['question_id']);a=copy.deepcopy(byqid[qid]);a['question_id']=index
        assert [x['answer'] for x in a['answers']]==r['gold_answers']
        real_anns.append(a);real_pred[index]=r['prediction']
    values,real_aggregate=reference_scores(real_anns,real_pred)
    native_tokens_available=0
    for i,(r,a) in enumerate(zip(records,real_anns),1):
        independent=independent_score(r['prediction'],a)
        assert abs(values[i]-independent['vqa_score'])<1e-10
        real.append({'source_line':i,'source_question_id':r['question_id'],'method_id':r['method_id'],
            'prediction':r['prediction'],'official_vqa_score':values[i],**independent})
        native_tokens_available+=isinstance(r.get('generated_tokens'),list)
    from mmimpress.mtvqa25k.vendor.vqaEval import VQAEval
    port_norm=VQAEval(Annotations(anns),None)
    for text in ['TWO','an apple','3.14','a.b','1,000',"dont","isn't",'blue-green','two  words','\nCAT\t']:
        assert port_norm.processPunctuation(text)==upstream.processPunctuation(text)
        assert port_norm.processDigitArticle(text)==upstream.processDigitArticle(text)
    dump(output/'existing_raw_reference_comparison.json',real)
    result={'status':'PASS','revision':REVISION,'synthetic_fixtures':len(fixtures),'actual_existing_raw_requests':len(records),
        'per_question_mismatches':0,'normalization_mismatches':0,'official_source_python3_port':'two print syntax changes only',
        'native_generated_ID_checks':'PENDING_NEW_RUN_GPU_OUTPUTS','old_raw_native_token_list_rows':native_tokens_available,
        'new_method_pilot_completed':False,'old_results_modified':False,'existing_raw_sha256':sha(raw),
        'annotations_archive_sha256':sha(archive),'vendor_provenance':read(ROOT/'mmimpress/mtvqa25k/vendor/provenance.json'),
        'unanimous_GT_normalization':'upstream conditional branch preserved; no lowercase/digit/article transform for unanimous cleaned GT'}
    dump(output/'validation.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();validate(a.output)

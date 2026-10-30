"""Deterministic, annotation-preserving reconstruction from official VQAv2 val."""
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,platform,random,statistics,sys
from PIL import Image
from .common import ROOT,sha,digest,dump,read

NAME='MT-VQA-v2-reconstructed-25k'
SEED=1234
TARGET=25000
QFILE='v2_OpenEnded_mscoco_val2014_questions.json'
AFILE='v2_mscoco_val2014_annotations.json'

def distribution(values):
    values=sorted(values)
    if not values:return {'count':0}
    return {'count':len(values),'min':values[0],'max':values[-1],'mean':statistics.mean(values),
            'p50':values[(len(values)-1)//2],'p95':values[int((len(values)-1)*.95)],
            'histogram':dict(sorted(Counter(values).items()))}

def join_sources(questions,annotations):
    qs={q['question_id']:q for q in questions};anns={a['question_id']:a for a in annotations}
    if len(qs)!=len(questions) or len(anns)!=len(annotations):raise ValueError('duplicate source question ID')
    if set(qs)!=set(anns):raise ValueError('questions/annotations membership mismatch')
    groups=defaultdict(list)
    for qid in sorted(qs):
        q=qs[qid];a=anns[qid]
        if q['image_id']!=a['image_id']:raise ValueError('question/annotation image mismatch')
        if len(a['answers'])!=10 or sorted(x['answer_id'] for x in a['answers'])!=list(range(1,11)):
            raise ValueError('invalid human answer IDs/count')
        if not isinstance(q['question'],str) or not q['question'].strip():raise ValueError('empty question')
        if any(not isinstance(x['answer'],str) for x in a['answers']):raise ValueError('invalid human answer')
        if not a.get('answer_type') or not a.get('question_type'):raise ValueError('missing answer/question type')
        groups[q['image_id']].append(qid)
    return qs,anns,dict(sorted(groups.items()))

def image_record(item):
    iid,path=item
    row={'image_id':iid,'image_path':str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)}
    try:
        if path.is_symlink() or not path.is_file():raise ValueError('missing or linked image')
        with Image.open(path) as im:im.verify()
        with Image.open(path) as im:
            im.load();row.update(width=im.width,height=im.height,mode=im.mode,format=im.format)
            rgb=im.convert('RGB');h=hashlib.sha256();h.update(f'{im.width}x{im.height}:RGB\n'.encode());h.update(rgb.tobytes())
        row.update(status='PASS',bytes=path.stat().st_size,sha256=sha(path),decoded_rgb_sha256=h.hexdigest())
    except Exception as e:row.update(status='EXCLUDED',reason=type(e).__name__+': '+str(e))
    return row

def select(groups,valid_ids,target=TARGET):
    eligible=sorted(i for i,qs in groups.items() if len(qs)>=3 and i in valid_ids)
    if len(eligible)<target:raise ValueError(f'BLOCKED: {len(eligible)} eligible images < {target}')
    rng=random.Random(SEED)
    image_order=rng.sample(eligible,target)
    selected=[(iid,rng.sample(sorted(groups[iid]),3)) for iid in image_order]
    return eligible,selected

def workload_hash(dialogues):
    # Length-delimited canonical JSON rows include exact questions and all GT.
    h=hashlib.sha256()
    from .common import canonical
    for d in dialogues:
        for t in d['turns']:
            h.update(canonical({'dialog_id':d['dialog_id'],'image_id':d['image_id'],
                               'image_sha256':d['image_sha256'],'turn':t})+b'\n')
    return h.hexdigest()

def question_stats(qids,qs,anns):
    return {'questions':len(qids),'answer_type':dict(Counter(anns[q]['answer_type'] for q in qids)),
            'question_type':dict(Counter(anns[q]['question_type'] for q in qids)),
            'question_characters':distribution([len(qs[q]['question']) for q in qids]),
            'question_whitespace_words':distribution([len(qs[q]['question'].split()) for q in qids])}

def image_stats(rows):
    return {'images':len(rows),'bytes':sum(r['bytes'] for r in rows),
            'width':distribution([r['width'] for r in rows]),'height':distribution([r['height'] for r in rows]),
            'pixels':distribution([r['width']*r['height'] for r in rows]),
            'shape':dict(Counter('landscape' if r['width']>r['height'] else 'portrait' if r['height']>r['width'] else 'square' for r in rows))}

def build(dest,workers=4):
    dest=Path(dest).resolve();source=dest/'source'
    if (dest/'config.json').exists():raise FileExistsError('Frozen dataset exists; use verify, not rebuild')
    qdoc=read(source/QFILE);adoc=read(source/AFILE)
    if qdoc['data_subtype']!='val2014' or adoc['data_subtype']!='val2014':raise ValueError('not validation')
    qs,anns,groups=join_sources(qdoc['questions'],adoc['annotations'])
    paths=[(int(p.stem.rsplit('_',1)[1]),p) for p in sorted((source/'val2014').glob('COCO_val2014_*.jpg'))]
    if len(paths)!=len({i for i,p in paths}):raise ValueError('duplicate image ID')
    print('decode/hash all source images',len(paths),flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:images=list(pool.map(image_record,paths))
    byimage={r['image_id']:r for r in images};valid={i for i,r in byimage.items() if r['status']=='PASS'}
    excluded={str(i):('fewer_than_three_distinct_questions' if len(q)<3 else 'missing_or_invalid_image')
              for i,q in groups.items() if len(q)<3 or i not in valid}
    for i in byimage:
        if i not in groups:excluded[str(i)]='no_question_annotation_pairs'
    dump(dest/'source_image_manifest.json',images)
    dump(dest/'source_manifest.json',{'questions':{'path':str(source/QFILE),'sha256':sha(source/QFILE),'bytes':(source/QFILE).stat().st_size,'count':len(qs)},
        'annotations':{'path':str(source/AFILE),'sha256':sha(source/AFILE),'bytes':(source/AFILE).stat().st_size,'count':len(anns)},
        'archives':[read(p) for p in sorted(source.glob('*.receipt.json'))],
        'metadata':{'questions':{k:v for k,v in qdoc.items() if k!='questions'},'annotations':{k:v for k,v in adoc.items() if k!='annotations'}},
        'canonical_image_question_ids':groups,'excluded':excluded})
    eligible,selection=select(groups,valid)
    dialogues=[]
    for ordinal,(iid,qids) in enumerate(selection):
        image=byimage[iid]
        turns=[{'turn_id':t,'question_id':qid,'question':qs[qid]['question'],
                'answers':anns[qid]['answers'],'answer_type':anns[qid]['answer_type'],
                'question_type':anns[qid]['question_type'],'annotation':anns[qid]} for t,qid in enumerate(qids,1)]
        dialogues.append({'dialog_id':f'mtvqa25k_{ordinal+1:06d}','global_dialog_ordinal':ordinal,
            'image_dialogue_index':1,'image_id':iid,'image_path':image['image_path'],'image_sha256':image['sha256'],'turns':turns})
    old=read(ROOT/'data/mt_vqa_v2/dialogues.json')['dialogues']
    old_iids={int(d['image_id']) for d in old};old_qids={int(t['question_id']) for d in old for t in d['turns']}
    chosen_iids={i for i,_ in selection};chosen_qids={q for _,qids in selection for q in qids}
    # Preserve the existing pilot's exact PNGs/order/questions; restore original annotation IDs.
    pilot=[]
    for d in old:
        d=dict(d);turns=[]
        for t in d['turns']:
            qid=int(t['question_id']);a=anns[qid]
            if int(d['image_id'])!=a['image_id'] or t['question']!=qs[qid]['question'] or t['answers']!=[x['answer'] for x in a['answers']]:raise ValueError('legacy pilot disagrees with official annotation')
            turns.append({**t,'annotation':a,'human_answers':a['answers'],'answer_type':a['answer_type'],'question_type':a['question_type']})
        pilot.append({**d,'turns':turns})
    dump(dest/'pilot250_official_annotations.json',{'source_sha256':sha(ROOT/'data/mt_vqa_v2/dialogues.json'),'dialogues':pilot})
    chosen_rows=[byimage[i] for i,_ in selection]
    stats={'eligible_images':len(eligible),'source_image_files':len(images),'valid_decoded_images':len(valid),
        'excluded_reasons':dict(Counter(excluded.values())),'selected_images':len(dialogues),'dialogues':len(dialogues),'questions':len(chosen_qids),
        'source_questions':question_stats(list(qs),qs,anns),'selected_questions':question_stats(sorted(chosen_qids),qs,anns),
        'questions_per_source_image':distribution([len(q) for q in groups.values()]),
        'source_images':image_stats([r for r in images if r['status']=='PASS']),'selected_image_stats':image_stats(chosen_rows),
        'legacy250_overlap':{'images':len(chosen_iids&old_iids),'questions':len(chosen_qids&old_qids),
            'image_ids':sorted(chosen_iids&old_iids),'question_ids':sorted(chosen_qids&old_qids)},
        'other_development_overlap':'see development_overlap.json; no universal unseen-holdout claim'}
    provenance={'name':NAME,'official_mt_vqa_subset':False,'exact_metacompress_reproduction':False,
        'description':'Three independent original VQAv2 questions sequentially executed for one image. No question is rewritten from prior answers.',
        'sampling':{'seed':SEED,'implementation':'CPython random.Random(1234), MT19937, sample(sorted eligible image_id,25000); then same RNG sample(sorted question_id,3) per sampled image in sampled order',
                    'python':sys.version,'random_py_sha256':sha(Path(random.__file__)),'selection_before_model_outputs':True,
                    'first_question_exclusion':False,'minimum_questions':3,'same_question_text_across_images_allowed':True},
        'official_sources_manifest_sha256':sha(dest/'source_manifest.json'),'original_image_manifest_sha256':sha(dest/'source_image_manifest.json')}
    dump(dest/'selection_manifest.json',{'eligible_canonical_image_ids':eligible,'ordered_selection':[{'image_id':i,'question_ids':q} for i,q in selection]})
    dump(dest/'image_manifest.json',chosen_rows)
    dump(dest/'dialogues.json',{'dataset':NAME,'schema_version':1,'dialogues':dialogues})
    dump(dest/'dataset_stats.json',stats);dump(dest/'dataset_provenance.json',provenance)
    files=['source_manifest.json','source_image_manifest.json','image_manifest.json','selection_manifest.json','dialogues.json','dataset_stats.json','dataset_provenance.json','pilot250_official_annotations.json']
    dump(dest/'config.json',{'dataset':NAME,'seed':SEED,'images':TARGET,'dialogues':TARGET,'questions':TARGET*3,
        'ordered_workload_sha256':workload_hash(dialogues),'files':{n:sha(dest/n) for n in files},
        'status':'BUILT_REQUIRES_INDEPENDENT_VERIFY','builder_sha256':sha(Path(__file__))})
    print(json.dumps({k:stats[k] for k in ['eligible_images','selected_images','questions','excluded_reasons','legacy250_overlap']}),flush=True)

def verify(dest):
    dest=Path(dest).resolve();config=read(dest/'config.json')
    for n,h in config['files'].items():
        if sha(dest/n)!=h:raise ValueError('dataset artifact hash mismatch: '+n)
    src=read(dest/'source_manifest.json')
    for label in ('questions','annotations'):
        if sha(src[label]['path'])!=src[label]['sha256']:raise ValueError('source hash mismatch')
    qs,anns,groups=join_sources(read(src['questions']['path'])['questions'],read(src['annotations']['path'])['annotations'])
    images=read(dest/'image_manifest.json');full=read(dest/'source_image_manifest.json')
    valid={r['image_id'] for r in full if r['status']=='PASS'}
    eligible,selection=select(groups,valid)
    ds=read(dest/'dialogues.json')['dialogues']
    if [(d['image_id'],[t['question_id'] for t in d['turns']]) for d in ds]!=selection:raise ValueError('sampling or order mismatch')
    if workload_hash(ds)!=config['ordered_workload_sha256']:raise ValueError('workload hash mismatch')
    allq=[]
    for d,image in zip(ds,images):
        if d['image_id']!=image['image_id']:raise ValueError('image membership/order mismatch')
        actual=image_record((image['image_id'],ROOT/image['image_path']))
        if actual!=image:raise ValueError('image file/decode changed: '+str(image['image_id']))
        if d['image_sha256']!=image['sha256']:raise ValueError('image SHA mismatch')
        for turn,t in enumerate(d['turns'],1):
            qid=t['question_id'];allq.append(qid)
            if t['turn_id']!=turn or t['question']!=qs[qid]['question'] or t['annotation']!=anns[qid] or t['answers']!=anns[qid]['answers'] or qs[qid]['image_id']!=d['image_id']:raise ValueError('annotation/turn mapping mismatch')
    if len(ds)!=TARGET or len(set(d['image_id'] for d in ds))!=TARGET or len(allq)!=3*TARGET or len(set(allq))!=3*TARGET:raise ValueError('denominator/uniqueness mismatch')
    return {'status':'PASS','images':len(ds),'dialogues':len(ds),'questions':len(allq),'eligible_images':len(eligible),
        'selected_images_fully_redecoded':len(images),'official_annotations_exact':True,'ordered_workload_sha256':workload_hash(ds),'config_sha256':sha(dest/'config.json')}

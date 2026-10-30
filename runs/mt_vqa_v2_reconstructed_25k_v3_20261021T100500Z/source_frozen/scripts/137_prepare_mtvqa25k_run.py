#!/usr/bin/env python3
"""Freeze prospective subsets and native settings before GPU outcomes."""
import argparse,copy,hashlib,json,os,platform,random,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from mmimpress.mtvqa25k.common import read,dump,sha,digest
from mmimpress.mtvqa25k.adapter import dialogues_for_base

METHODS={'llava':['recompute','fullload','mpic32_ssd','sparsevlm_ssd_kv25_allhead','ours_prefixstore_kv25'],
         'qwen':['recompute','fullload','qwen_ours_prefixstore_kv25'],
         'glm':['recompute','fullload','ours_prefixstore_kv25']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset',type=Path,default=ROOT/'data/mt_vqa_v2_reconstructed_25k');a=p.parse_args();run=a.run.resolve()
    ds=dialogues_for_base(read(a.dataset/'dialogues.json')['dialogues'])
    pilot=dialogues_for_base(read(a.dataset/'pilot250_official_annotations.json')['dialogues'])
    geometry={m:read(run/f'geometry_{m}.json') for m in METHODS}
    chosen=[];reasons={}
    def choose(label,key):
        d=min(ds,key=lambda d:(key(d),int(d['image_id'])))
        reasons[label]=d['dialog_id']
        if d['dialog_id'] not in chosen:chosen.append(d['dialog_id'])
    for m in METHODS:
        g=geometry[m]['items']
        if geometry[m]['selected_incompatible_images']:raise RuntimeError(f'{m} has incompatible native shapes; freeze blocked')
        choose(m+'_min_N',lambda d:g[d['image_id']]['N_content'])
        choose(m+'_max_N',lambda d:-g[d['image_id']]['N_content'])
        choose(m+'_k64_lower',lambda d:g[d['image_id']]['k_mod64'])
        choose(m+'_k64_upper',lambda d:64-g[d['image_id']]['k_mod64'])
        choose(m+'_longest_question',lambda d:-max(g[d['image_id']]['question_token_lengths']))
        choose(m+'_longest_questions_history_proxy',lambda d:-sum(g[d['image_id']]['question_token_lengths']))
    images={str(r['image_id']):r for r in read(a.dataset/'image_manifest.json')}
    choose('most_portrait',lambda d:images[d['image_id']]['width']/images[d['image_id']]['height'])
    choose('most_landscape',lambda d:-images[d['image_id']]['width']/images[d['image_id']]['height'])
    gate_ids=list(chosen)
    # Fixed hash ranking is an independent deterministic fill rule, no quality input.
    remaining=sorted([d for d in ds if d['dialog_id'] not in chosen],key=lambda d:hashlib.sha256(('1234:integration40:'+d['dialog_id']).encode()).hexdigest())
    chosen.extend(d['dialog_id'] for d in remaining[:40-len(chosen)])
    assert len(chosen)==len(set(chosen))==40
    dump(run/'validation_selection.json',{'selected_before_model_outputs':True,'seed':1234,'reasons':reasons,
        'gate_dialogue_ids':gate_ids,'integration40_dialogue_ids':chosen,'fill':'SHA256(1234:integration40:dialog_id), ascending',
        'generated_history_length':'unknown until generation; sum of three question token lengths is a prospective proxy',
        'negative_requirements':['wrong_image','wrong_ID','wrong_position','missing_structure','duplicate_structure','truncated_store','corrupted_store','unsealed_store','future_visibility','history_state_leakage'],
        'criteria':{'llava_logits_atol':1e-4,'llava_logits_rtol':1e-4,'qwen_glm_same_shape':'exact native bits/logits/IDs',
            'qwen_glm_semantic_atol':1e-12,'qwen_glm_semantic_rtol':1e-12,'legacy_FP32_diagnostic_atol':1e-5,'legacy_FP32_diagnostic_rtol':1e-5}})
    for model,methods in METHODS.items():
        base=run/model;base.mkdir(exist_ok=True)
        cfg={'model':model,'methods':methods,'seed':1234,'batch_size':1,'greedy':True,'max_new_tokens':32 if model=='glm' else 16,
             'cpu_threads':2,'dtype_backend':{'llava':'NF4 double/BF16 compute/eager/FP16 SSD','qwen':'NF4 double/BF16 native KV/SDPA','glm':'decoder NF4 double/BF16 vision and native KV/SDPA'}[model],
             'scorer_revision':'a013f0043c1e2cdc995922dfe257f7149aa9af06','dataset_config_sha256':sha(a.dataset/'config.json'),
             'geometry_sha256':sha(run/f'geometry_{model}.json'),'selection_sha256':sha(run/'validation_selection.json'),
             'rotation':'canonical dialogue ordinal modulo arm count; fixed for all turns','raw_archive':'lossless gzip after full audit, original and compressed full SHA bound by atomic commit',
             'existing_source_changes':'none by this task; current Oct13 FullLoad identity/single-token prefill fix retained and regression tested',
             'request_E2E':'same outer request helper return for all arms; Qwen core_returned_at_s endpoint for all capture modes',
             'GLM_cap32_reason':'validated v6 native answer-boundary handling; cap16 truncated a boundary in preserved v3 pilot',
             'enable_thinking':False if model=='glm' else 'NOT_APPLICABLE'}
        if model=='qwen':
            prior=read(ROOT/'runs/qwen_mt_gqa_prefixstore3_20261002T074856Z/config.json')
            for key in ('model','core_model_store_revision','position_policy'):
                cfg[key]=copy.deepcopy(prior[key])
            cfg['model_key']='qwen'
            cfg['prior_native_contract_sha256']=sha(ROOT/'runs/qwen_mt_gqa_prefixstore3_20261002T074856Z/config.json')
        dump(base/'model_settings.json',cfg)
        for phase,selected in [('main',ds),('pilot250',pilot),('pilot40',[d for d in ds if d['dialog_id'] in chosen]),
                               ('gate',[d for d in ds if d['dialog_id'] in gate_ids]),('smoke',[d for d in ds if d['dialog_id'] in chosen[:4]])]:
            identity={}
            for d in selected:
                iid=d['image_id'];g=geometry[model]['pilot250'][iid] if phase=='pilot250' else geometry[model]['items'][iid]
                path=ROOT/d['image_path'];identity[iid]={'image_path':d['image_path'],'sha256':sha(path),'N_content':g['N_content'],**g}
            # Keep each phase's canonical index order. Rotation is frozen here.
            manifest={'dataset':'MT-VQA-v2-reconstructed-25k' if phase!='pilot250' else 'original-250-image-development-pilot',
                'phase':phase,'dialogues':selected,'image_identity':identity,'smoke_dialogue_ids':[d['dialog_id'] for d in selected],
                'integration_dialogue_ids':[d['dialog_id'] for d in selected],
                'requests':len(selected)*3*len(methods),'followup_requests_per_method':len(selected)*2,
                'ordered_selection_sha256':digest([(d['image_id'],[t['question_id'] for t in d['turns']]) for d in selected])}
            phasepath=base/phase;phasepath.mkdir(exist_ok=True);dump(phasepath/'manifest.json',manifest)
            dump(phasepath/'storage_plan.json',{'minimum_reserve_bytes':30<<30,'peak_working_set_bytes':8<<30 if model=='llava' else 2<<30,
                'largest_image_bundle_bound_bytes':4<<30 if model=='llava' else 1<<30,'maximum_new_payload_bytes':16<<30,
                'note':'pre-pilot conservative guard; main additionally requires measured pilot raw/I/O capacity plan'})
    print(json.dumps({'frozen_membership':True,'gate_images':len(gate_ids),'pilot40_images':40,'main_images':25000,'target_requests':825000}),flush=True)

if __name__=='__main__':main()

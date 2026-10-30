#!/usr/bin/env python3
"""Full source/config/environment/checkpoint hashes before GPU admission."""
import argparse,hashlib,importlib.metadata,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mmimpress.mtvqa25k.common import read,dump,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    if (run/'experiment_freeze.json').exists():raise FileExistsError('freeze exists')
    before=read(run/'existing_source_protection.json')
    for name,h in before.items():
        if sha(ROOT/name)!=h:raise RuntimeError('existing user source changed: '+name)
    dump(run/'existing_source_preservation.json',{'status':'PASS','full_SHA_files':len(before),'changes':0})
    envs={}
    code='import sys,importlib.metadata as m,json; print(json.dumps({"python":sys.version,"executable":sys.executable,"packages":{n:m.version(n) for n in ["torch","transformers","bitsandbytes","accelerate","numpy","Pillow","tokenizers"]}}))'
    for model,python in [('llava','/home/dblab/anaconda3/envs/mllm_ft/bin/python'),('qwen','/home/dblab/anaconda3/envs/mllm_ft/bin/python'),('glm',str(ROOT/'.venv-glm46v/bin/python'))]:
        envs[model]=json.loads(subprocess.check_output([python,'-c',code],text=True))
    envs['nvidia_smi']=subprocess.check_output(['nvidia-smi'],text=True)
    envs['free_disk_bytes']=shutil.disk_usage(run).free
    envs['git']='unavailable .git metadata; full source hashes/copies used'
    dump(run/'environment.json',envs)
    checkpoints={'llava':Path.home()/'.cache/huggingface/hub/models--llava-hf--llava-v1.6-vicuna-7b-hf/snapshots/c916e6cdcd760b4cecd1dd4907f84ac649f93b23',
        'qwen':Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5',
        'glm':ROOT/'runs/glm46v_prefixstore3_20261007T065207Z/checkpoint'}
    models={}
    for model,path in checkpoints.items():
        files={}
        for f in sorted(path.iterdir()):
            if f.is_file() and (f.suffix in ('.json','.jinja','.safetensors','.model','.txt')):
                files[f.name]={'sha256':sha(f),'bytes':f.stat().st_size,'resolved':str(f.resolve())}
        assert any(n.endswith('.safetensors') for n in files)
        models[model]={'snapshot':str(path),'files':files};print('hashed checkpoint',model,flush=True)
    dump(run/'model_freeze.json',models)
    shutil.copy2(ROOT/'docs/mt_vqa_v2_reconstructed_25k_contract.md',run/'CONTRACT.md')
    sources=list((ROOT/'mmimpress').rglob('*.py'))+list((ROOT/'scripts').glob('*.py'))+list((ROOT/'tests').glob('*.py'))+[ROOT/'docs/mt_vqa_v2_reconstructed_25k_contract.md']
    sources += list((ROOT/'mmimpress/mtvqa25k/vendor').glob('*.original'))+list((ROOT/'mmimpress/mtvqa25k/vendor').glob('*.json'))
    source_sha={str(p.relative_to(ROOT)):sha(p) for p in sorted(sources)}
    for name in source_sha:
        dest=run/'source_frozen'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    artifacts={str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*.json')) if 'source_frozen' not in p.parts and 'STATUS' not in p.name}
    artifacts['CONTRACT.md']=sha(run/'CONTRACT.md');artifacts['REQUEST.md']=sha(run/'REQUEST.md')
    dump(run/'experiment_freeze.json',{'source_sha256':source_sha,'artifact_sha256':artifacts,
        'created_unix':__import__('time').time(),'environment_sha256':sha(run/'environment.json'),'models_sha256':sha(run/'model_freeze.json'),
        'ordered_workload_sha256':read(ROOT/'data/mt_vqa_v2_reconstructed_25k/config.json')['ordered_workload_sha256'],
        'main_admission':'requires fresh model gate + smoke + pilots + measured resource admission; freeze alone is not PASS'})
    print('frozen',len(source_sha),'source files and',len(artifacts),'artifacts',flush=True)

if __name__=='__main__':main()

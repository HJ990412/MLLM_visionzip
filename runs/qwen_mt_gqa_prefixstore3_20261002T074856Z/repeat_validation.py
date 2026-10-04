"""Supplemental fixed-input repetitions; no threshold or production changes."""
from pathlib import Path
import importlib.util,json,sys,time
import torch
ROOT=Path('/home/dblab/hj/mllm_v2');RUN=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
s=importlib.util.spec_from_file_location('p3',ROOT/'scripts/117_run_qwen_prefixstore3.py');E=importlib.util.module_from_spec(s);sys.modules[s.name]=E;s.loader.exec_module(E)
E.M.CONTRACT=RUN/'CONTRACT.md';E.seed();E.M.verify_freeze(RUN)
assert E.read(RUN/'integration_validation.json')['status']=='PASS'
files=sorted((RUN/'integration').glob('images/*/attempt_*/semantic_*/fixed_inputs.pt'))+sorted((RUN/'additional/integration').glob('images/*/attempt_*/semantic_*/fixed_inputs.pt'))
assert len(files)==24
E.dump(RUN/'repeat_validation_freeze.json',dict(source_sha256=E.M.sha(Path(__file__)),final_freeze_sha256=E.M.sha(RUN/'final_source_freeze.json'),fixtures=[dict(path=str(p),sha256=E.M.sha(p)) for p in files],criterion='FP64 dense/compact each 2 exact repeats; independent memory path 2 bitwise-logit/exact-ID repeats; no new tolerance'))
runner=E.M.load_runner();results=[]
try:
 for path in files:
  x=torch.load(path,map_location='cpu',weights_only=False);logical=x['logical_indices'];ix=torch.tensor(logical);prefix=x['prefix_length'];suffix=x['suffix_ids'];pos=x['positions']
  ref=[(k.index_select(2,ix).clone(),v.index_select(2,ix).clone()) for k,v in x['canonical_layers']]
  a=E.V.trace_suffix(runner,ref,suffix,pos,prefix);b=E.V.trace_suffix(runner,ref,suffix,pos,prefix)
  cmp=E.V.path_comparison(a,b);assert cmp['exact_target'] and E.G.tensor_hash(a['logits'])==E.G.tensor_hash(b['logits'])
  layer_results=[]
  for li,(r,(pk,pv)) in enumerate(zip(x['production'],x['canonical_layers'])):
   p=len(logical);t=r['q'].shape[2];group=r['k'].shape[1]//4;nk=r['k'][:,::group];nv=r['v'][:,::group]
   assert torch.equal(E.G.expand_heads(nk,r['q'].shape[1]),r['k']) and torch.equal(E.G.expand_heads(nv,r['q'].shape[1]),r['v'])
   ks,vs=nk[:,:,p:],nv[:,:,p:]
   cm=E.G.visible_mask(logical,prefix,t);dm=E.G.visible_mask(logical,prefix,t,True)
   ck=torch.cat((pk.index_select(2,ix),ks),2);cv=torch.cat((pv.index_select(2,ix),vs),2)
   dk=torch.cat((pk,ks),2);dv=torch.cat((pv,vs),2);scale=r['meta']['scale']
   c1=E.G.attention(r['q'],ck,cv,cm,torch.float64,scale)['output'];c2=E.G.attention(r['q'],ck,cv,cm,torch.float64,scale)['output']
   d1=E.G.attention(r['q'],dk,dv,dm,torch.float64,scale)['output'];d2=E.G.attention(r['q'],dk,dv,dm,torch.float64,scale)['output']
   assert torch.equal(c1,c2) and torch.equal(d1,d2)
   sem=E.G.stats(c1,d1,1e-12,1e-12);assert sem['allclose']
   layer_results.append(dict(layer=li,compact_repeat_exact=True,dense_repeat_exact=True,expanded_heads_exact=True,semantic=sem))
  results.append(dict(fixture=str(path),fixture_sha256=E.M.sha(path),memory_repeat=cmp,layers=layer_results))
  print('repeat fixed input',len(results),'/24 PASS',flush=True)
finally:runner.close()
E.dump(RUN/'repeat_validation.json',dict(status='PASS',samples=results,additional_criteria_changes=False,production_changed=False))

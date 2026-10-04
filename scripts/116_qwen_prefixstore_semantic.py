"""Independent fixed-input Qwen attention diagnostics. Never used for main inference."""
from __future__ import annotations

import contextlib
import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F

FP64_ATOL = FP64_RTOL = 1e-12
LEGACY_ATOL = LEGACY_RTOL = 1e-5


def tensor_hash(x):
    x = x.detach().cpu().contiguous()
    return hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()


def describe(x):
    return dict(dtype=str(x.dtype), shape=list(x.shape), stride=list(x.stride()),
                device=str(x.device), sha256=tensor_hash(x))


def stats(a, b, atol=0., rtol=0.):
    assert a.shape == b.shape
    a, b = a.double(), b.double()
    diff = (a-b).abs()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    flat = int(diff.argmax())
    index = []
    for dim in reversed(diff.shape):
        index.append(flat % dim)
        flat //= dim
    return dict(exact=torch.equal(a,b), allclose=finite and bool(torch.allclose(a,b,atol=atol,rtol=rtol)),
                max_abs=float(diff.max()), mean_abs=float(diff.mean()),
                violations=int((diff > atol+rtol*b.abs()).sum()), elements=a.numel(),
                worst_index=list(reversed(index)), atol=atol, rtol=rtol)


def independent_indices(capture, ratio=.25):
    n = int(capture.visual_count)
    rank = sorted(range(n), key=lambda i: (-float(capture.scores[i]), i))
    k = (n+3)//4 if ratio == .25 else n
    chosen = rank[:k] if ratio == .25 else list(range(n))
    structural = [i for i in range(len(capture.prefix_ids))
                  if not capture.visual_start <= i < capture.visual_start+n]
    logical = sorted(structural + [capture.visual_start+i for i in chosen])
    return rank, chosen, structural, logical


def gather_canonical(capture, logical):
    # No production selector, store loader or assembly helper is called.
    index = torch.tensor(logical, device=capture.layers[0][0].device)
    return [(k.index_select(2,index).detach().cpu().clone(),
             v.index_select(2,index).detach().cpu().clone()) for k,v in capture.layers]


def visible_mask(prefix_indices, prefix_length, suffix_length, dense=False):
    if dense:
        prefix = [i in set(prefix_indices) for i in range(prefix_length)]
    else:
        prefix = [True]*len(prefix_indices)
    # Explicit causal relation, independent of production mask helpers.
    return torch.tensor([prefix + [j <= i for j in range(suffix_length)]
                         for i in range(suffix_length)], dtype=torch.bool)


class CaptureSDPA:
    """Observe the first text prefill's actual post-MRoPE SDPA arguments."""
    def __enter__(self):
        self.original = F.scaled_dot_product_attention
        self.records = []
        def observe(q,k,v,*args,**kwargs):
            if len(self.records) < 28:
                mask = kwargs.get('attn_mask', args[0] if args else None)
                causal = kwargs.get('is_causal', args[2] if len(args)>2 else False)
                self.records.append(dict(q=q.detach().cpu().clone(), k=k.detach().cpu().clone(),
                    v=v.detach().cpu().clone(), mask=None if mask is None else mask.detach().cpu().clone(),
                    meta=dict(q=describe(q),k=describe(k),v=describe(v),
                              mask=None if mask is None else describe(mask),
                              is_causal=bool(causal), scale=kwargs.get('scale'),
                              enable_gqa=kwargs.get('enable_gqa',False),
                              autocast_cuda=torch.is_autocast_enabled('cuda'),
                              autocast_cpu=torch.is_autocast_enabled('cpu'))))
            return self.original(q,k,v,*args,**kwargs)
        F.scaled_dot_product_attention = observe
        return self
    def __exit__(self,*exc):
        F.scaled_dot_product_attention = self.original


def expand_heads(x, heads):
    assert heads % x.shape[1] == 0
    return x.repeat_interleave(heads//x.shape[1], dim=1)


def attention(q,k,v,visible,dtype,scale=None):
    # Convert BEFORE every operation. Cached K is already post-MRoPE.
    with torch.autocast('cpu',enabled=False):
        q=q.to(dtype);k=expand_heads(k.to(dtype),q.shape[1]);v=expand_heads(v.to(dtype),q.shape[1])
        dots=torch.matmul(q,k.transpose(-1,-2))
        logits=dots/math.sqrt(q.shape[-1]) if scale is None else dots*scale
        masked=logits.masked_fill(~visible[None,None],float('-inf'))
        prob=torch.softmax(masked,dim=-1,dtype=dtype)
        out=torch.matmul(prob,v)
    return dict(dots=dots, logits=logits, prob=prob, output=out)


def validate_budget(original_n,stored_k,attended_k,stored_ids):
    expected=(int(original_n)+3)//4
    assert stored_k==attended_k==expected and stored_ids==list(range(expected)), 'original-N quarter budget'


def validate_structure(*, logical, expected_logical, positions, expected_positions,
                       mask, expected_mask, layers, expected_layers):
    assert logical == expected_logical and len(set(logical)) == len(logical), 'logical IDs'
    assert torch.equal(positions,expected_positions), 'MRoPE positions'
    assert mask.shape == expected_mask.shape and torch.equal(mask,expected_mask), 'causal mask'
    assert len(layers)==len(expected_layers)==28, 'layer count'
    for a,b in zip(layers,expected_layers):
        for x,y in zip(a,b):
            assert x.dtype==y.dtype==torch.bfloat16 and x.shape==y.shape and tensor_hash(x)==tensor_hash(y), 'native KV bits'


def semantic_layer(record, canonical_pair, logical, prefix_length, repeats=3):
    q=record['q']; p=len(logical); t=q.shape[2]
    # HF may repeat GQA heads before SDPA. Preserve native four heads for references.
    heads=q.shape[1]; group=heads//4
    native_k=record['k'][:,::group] if record['k'].shape[1]==heads else record['k']
    native_v=record['v'][:,::group] if record['v'].shape[1]==heads else record['v']
    assert native_k.shape[1]==native_v.shape[1]==4
    ck,cv=(x.cpu() for x in canonical_pair)
    ix=torch.tensor(logical)
    selected_k,selected_v=ck.index_select(2,ix),cv.index_select(2,ix)
    assert torch.equal(native_k[:,:,:p],selected_k), 'production prefix K differs'
    assert torch.equal(native_v[:,:,:p],selected_v), 'production prefix V differs'
    ks,vs=native_k[:,:,p:],native_v[:,:,p:]
    compact_k=torch.cat((selected_k,ks),2);compact_v=torch.cat((selected_v,vs),2)
    dense_k=torch.cat((ck,ks),2);dense_v=torch.cat((cv,vs),2)
    cm=visible_mask(logical,prefix_length,t);dm=visible_mask(logical,prefix_length,t,True)
    actual=record['mask']
    assert actual is not None
    actual=actual[0,0] if actual.dtype==torch.bool else actual[0,0]==0
    assert torch.equal(actual,cm), 'actual production SDPA visibility'
    # Match installed head_dim**-0.5 exactly; reciprocal sqrt differs by one Python float ULP.
    assert record['meta']['scale'] in (None,128**-0.5)
    scale=record['meta']['scale']
    c64=attention(q,compact_k,compact_v,cm,torch.float64,scale)
    d64=attention(q,dense_k,dense_v,dm,torch.float64,scale)
    sem=stats(c64['output'],d64['output'],FP64_ATOL,FP64_RTOL)
    assert sem['allclose'], 'FP64 semantic mismatch'
    c32=attention(q,compact_k,compact_v,cm,torch.float32)
    d32=attention(q,dense_k,dense_v,dm,torch.float32)
    cols=torch.tensor(logical+list(range(prefix_length,prefix_length+t)))
    ds={key:value.index_select(-1,cols) for key,value in d32.items() if key!='output'}
    # Causal future entries have probability zero; logits remain finite for stage comparisons.
    stage={key:stats(c32[key],ds[key]) for key in ('dots','logits','prob')}
    vm=cm[None,None].expand_as(c32['dots'])
    stage['visible_dots']=stats(c32['dots'][vm],ds['dots'][vm])
    stage['visible_logits']=stats(c32['logits'][vm],ds['logits'][vm])
    dense_score_compact_prob=torch.softmax(ds['logits'].masked_fill(~cm[None,None],float('-inf')),-1)
    corrected=dense_score_compact_prob@expand_heads(compact_v.float(),heads)
    stage['output_compact_using_dense_logits_vs_dense']=stats(corrected,d32['output'])
    corrected_p=ds['prob']@expand_heads(compact_v.float(),heads)
    stage['output_compact_using_dense_probabilities_vs_dense']=stats(corrected_p,d32['output'])
    worst=stats(c32['output'],d32['output'])['worst_index'];_,head,query,outdim=worst
    vis=cm[query]
    stage['worst_output_location']=dict(layer_local_head=head,query=query,output_dim=outdim,
        visible_dot_difference=stats(c32['dots'][0,head,query,vis],ds['dots'][0,head,query,vis]),
        visible_probability_difference=stats(c32['prob'][0,head,query,vis],ds['prob'][0,head,query,vis]),
        residual_after_dense_logits=float((corrected-d32['output'])[0,head,query,outdim]),
        original_output_difference=float((c32['output']-d32['output'])[0,head,query,outdim]))
    # Isolate softmax reduction shape using IDENTICAL visible logits.
    padded=torch.full_like(d32['logits'],float('-inf'))
    padded.index_copy_(-1,cols,c32['logits'].masked_fill(~cm[None,None],float('-inf')))
    softmax_same_input=torch.softmax(padded,-1).index_select(-1,cols)
    stage['softmax_same_visible_logits_different_shape']=stats(c32['prob'],softmax_same_input)
    # Isolate value aggregation shape with identical probabilities, padding zero mass.
    pp=torch.zeros_like(d32['prob']);pp.index_copy_(-1,cols,c32['prob'])
    padded_aggregation=pp@expand_heads(dense_v.float(),heads)
    stage['aggregation_same_probabilities_different_shape']=stats(c32['output'],padded_aggregation)
    repeated=[]
    for _ in range(repeats-1):
        a=attention(q,compact_k,compact_v,cm,torch.float32)['output']
        b=attention(q,dense_k,dense_v,dm,torch.float32)['output']
        repeated.append(dict(compact_exact=torch.equal(a,c32['output']),dense_exact=torch.equal(b,d32['output'])))
    assert all(x['compact_exact'] and x['dense_exact'] for x in repeated), 'fixed-input repeat'
    legacy=stats(c32['output'],d32['output'],LEGACY_ATOL,LEGACY_RTOL)
    legacy.update(allclose=bool(torch.allclose(c32['output'],d32['output'],atol=LEGACY_ATOL,rtol=LEGACY_RTOL)),
        violations=int(((c32['output']-d32['output']).abs()>LEGACY_ATOL+LEGACY_RTOL*d32['output'].abs()).sum()))
    result=dict(semantic_fp64=sem,legacy_fp32=legacy,stages=stage,
                compact_fp32_vs_fp64=stats(c32['output'],c64['output'],LEGACY_ATOL,LEGACY_RTOL),
                dense_fp32_vs_fp64=stats(d32['output'],d64['output'],LEGACY_ATOL,LEGACY_RTOL),
                fixed_input_repeats=repeated, q=describe(q),canonical_k=describe(ck),canonical_v=describe(cv))
    return result,dict(compact_FP32=c32['output'],dense_FP32=d32['output'],
                       compact_FP64=c64['output'],dense_FP64=d64['output'])


def numerical_environment():
    return dict(torch_version=torch.__version__,cpu_threads=torch.get_num_threads(),
                matmul_precision=torch.get_float32_matmul_precision(),
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                fp16_reduced_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                bf16_reduced_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                reference_device='CPU',reference_matmul_softmax_value_dtypes=['float32','float64'],
                production_backend='SDPA',torch_config=torch.__config__.show())


def main():
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);x=torch.load(a.fixture,map_location='cpu',weights_only=False)
    results=[]
    for li,(record,pair) in enumerate(zip(x['production'],x['canonical_layers'])):
        evidence,_=semantic_layer(record,pair,x['logical_indices'],x['prefix_length'])
        results.append(dict(layer=li,**evidence))
    with a.output.open('x') as f:json.dump(dict(status='PASS',fixture_sha256=hashlib.sha256(a.fixture.read_bytes()).hexdigest(),layers=results),f,indent=2)

if __name__=='__main__':main()

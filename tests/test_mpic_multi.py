"""Small real decoder regressions; no network, pretrained weights, or GPU."""
import contextlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from mmimpress.mpic import MPICRawReader
from mmimpress.mpic_multi import (ImageBinding, MultiSelectivePrefill, MPICMultiServer,
                                  build_multi_plan, expand_multi_prompt, prepare_mixed)
from mmimpress.mmdu_mpic import (prefix_prefill, full_reuse_prefill, logical_labels,
                                 prompt_from_generated)
from mmimpress.store import ChunkReader, IOCounter
from mmimpress.serve import PrefixCache, Server


class Tokenizer:
    eos_token_id = 127
    def __call__(self, text, return_tensors=None):
        values = [100 if x == '<image>' else int(x.removeprefix('t')) for x in text.split()]
        return SimpleNamespace(input_ids=torch.tensor([values]))
    def decode(self, tokens, **kwargs):
        return ' '.join(map(str, tokens))


class Top(torch.nn.Module):
    def __init__(self, lm):
        super().__init__(); self.lm=lm; self.config=lm.config
        self.device=torch.device('cpu'); self.model=SimpleNamespace(language_model=lm.model); self.lm_head=lm.lm_head
    def get_input_embeddings(self): return self.lm.get_input_embeddings()
    def forward(self, **kwargs): return self.lm(**kwargs)


@contextlib.contextmanager
def cpu_sync():
    original=torch.cuda.synchronize;torch.cuda.synchronize=lambda *a,**kw:None
    try: yield
    finally: torch.cuda.synchronize=original


class MultiFixture(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(1234)
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        cfg=LlamaConfig(vocab_size=128,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                        num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=512,
                        attention_dropout=0.,attn_implementation='eager')
        self.lm=LlamaForCausalLM(cfg).float().eval()
        self.runner=SimpleNamespace(model=Top(self.lm),layers=self.lm.model.layers,image_token_id=100,
            processor=SimpleNamespace(tokenizer=Tokenizer()),prompt=lambda q:'t1 t2 <image> '+q)
        self.contexts=[];self.visual={};self.source=[]
        with torch.no_grad():
            for i,n in enumerate((40,47)):
                folder=self.root/str(i);folder.mkdir()
                visual=torch.randn(n,32);self.visual[str(i)]=visual
                ids=torch.tensor([1,2]+[100]*n+[3,4]);emb=self.lm.get_input_embeddings()(ids[None]).clone()
                emb[:,2:2+n]=visual[None]
                native=self.lm(inputs_embeds=emb,use_cache=True)
                layers=[(x.keys.detach().clone(),x.values.detach().clone()) for x in native.past_key_values.layers]
                self.source.append(layers)
                for li,pair in enumerate(layers):
                    sub=folder/f'layer_{li:02d}';sub.mkdir()
                    for kind,value in zip(('k','v'),pair):
                        (sub/f'{kind}.bin').write_bytes(value[0,:,2:2+n].permute(1,0,2).contiguous().numpy().tobytes())
                (folder/'visual_input.bin').write_bytes(visual.numpy().tobytes())
                meta=dict(image_id=str(i),dtype='float32',v_token_start=2,v_token_num=n,num_layers=2,num_heads=2,
                          head_dim=8,hidden_size=32,chunk_size=64,newline_idx=[n-1],n_spatial=n-1,
                          prefix_len=2+n,prefix_input_ids=[1,2]+[100]*n,n_chunks_per_layer=1)
                self.contexts.append(SimpleNamespace(meta=meta,dir=folder,reader=MPICRawReader(folder,meta,drop_cache=False)))
    def tearDown(self):
        for ctx in self.contexts:ctx.reader.close()
        self.tmp.cleanup()
    def input(self,swap=False):
        contexts=self.contexts[::-1] if swap else self.contexts
        names=[str(c.meta['image_id']) for c in contexts]
        return expand_multi_prompt(self.runner,'t1 t5 t2 <image> t3 t6 <image> t7 t8',contexts,names)
    def embeds(self,ids,bindings):
        emb=self.lm.get_input_embeddings()(ids[None]).clone()
        for b in bindings:emb[:,b.start:b.start+b.count]=self.visual[b.image_id][None]
        return emb
    @torch.no_grad()
    def independent(self, ids, bindings, k, policy):
        """Separate compact-active implementation, SDPA, explicit indexed linking."""
        visual_pos=set();chosen=[];remaining=k
        selected_counts=[]
        for b in bindings:
            visual_pos.update(range(b.start,b.start+b.count))
            c=b.count if k is None else min(b.count,k if policy=='per_image' else remaining)
            chosen.extend(range(b.start,b.start+c));selected_counts.append(c)
            if remaining is not None and policy=='global':remaining-=c
        active=sorted((set(range(len(ids)))-visual_pos)|set(chosen));p=torch.tensor(active)
        hidden=self.embeds(ids,bindings)[:,p]
        expected=[]
        def rotate(x,c,s):
            a,b=x.chunk(2,-1);return x*c[:,None]+torch.cat((-b,a),-1)*s[:,None]
        for li,layer in enumerate(self.runner.layers):
            normed=layer.input_layernorm(hidden);res=hidden;attn=layer.self_attn
            def proj(mod):return mod(normed).reshape(1,len(active),-1,8).transpose(1,2)
            q,ak,av=proj(attn.q_proj),proj(attn.k_proj),proj(attn.v_proj)
            c,s=self.lm.model.rotary_emb(normed,p[None]);q,ak=rotate(q,c,s),rotate(ak,c,s)
            fk=torch.empty(1,2,len(ids),8);fv=torch.empty_like(fk)
            for b in bindings:
                pair=[]
                for kind in ('k','v'):
                    a=np.fromfile(b.context.dir/f'layer_{li:02d}/{kind}.bin',dtype=np.float32).copy()
                    pair.append(torch.from_numpy(a).reshape(b.count,2,8).permute(1,0,2)[None])
                ik,iv=pair
                sc,ss=self.lm.model.rotary_emb(ik,torch.arange(2,2+b.count)[None])
                tc,ts=self.lm.model.rotary_emb(ik,torch.arange(b.start,b.start+b.count)[None])
                ik=rotate(rotate(ik,sc,-ss),tc,ts)
                fk[:,:,b.start:b.start+b.count]=ik;fv[:,:,b.start:b.start+b.count]=iv
            fk[:,:,p]=ak;fv[:,:,p]=av
            allowed=torch.arange(len(ids))[None]<=p[:,None]
            attended=torch.nn.functional.scaled_dot_product_attention(q,fk.repeat_interleave(2,1),
                fv.repeat_interleave(2,1),attn_mask=allowed[None,None],dropout_p=0.)
            hidden=res+attn.o_proj(attended.transpose(1,2).reshape(1,len(active),32))
            hidden=hidden+layer.mlp(layer.post_attention_layernorm(hidden))
            expected.append((fk,fv,hidden.clone()))
        return self.lm.lm_head(self.lm.model.norm(hidden)[:,-1:]),expected
    def test_global_per_image_boundary_and_swap_against_independent_attention(self):
        for swap in (False,True):
            ids,bindings=self.input(swap)
            for policy,k in [('per_image',0),('per_image',32),('per_image',None),('global',32)]:
                with self.subTest(swap=swap,policy=policy,k=k):
                    logits,cache,plan,stats=MultiSelectivePrefill(self.runner,k_recompute=k,selection_policy=policy).run(ids,bindings,IOCounter(),trace_tensors=True)
                    ref,expected=self.independent(ids,bindings,k,policy)
                    torch.testing.assert_close(logits,ref,rtol=1e-5,atol=1e-6)
                    for pair,oracle,tensor in zip(cache,expected,stats['diagnostic_tensors']):
                        for a,b in zip(pair,oracle[:2]):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
                        torch.testing.assert_close(tensor['hidden'],oracle[2],rtol=1e-5,atol=1e-6)
                    for layer in stats['layers']:
                        self.assertEqual(layer['Q_shape'][2],len(plan.active_positions));self.assertEqual(layer['future_probability_max'],0.)
                        self.assertTrue(all(layer['reused_unchanged_per_image']));self.assertTrue(layer['active_K_installed'] and layer['active_V_installed'])
                    if k is None:
                        with torch.no_grad():native=self.lm(inputs_embeds=self.embeds(ids,bindings),use_cache=True)
                        torch.testing.assert_close(logits,native.logits[:,-1:],rtol=1e-5,atol=1e-6)
    def test_dummy_and_request_isolation(self):
        ids,bindings=self.input();baseline=None
        for dummy in (0.,12345.):
            result=MultiSelectivePrefill(self.runner,k_recompute=32,selection_policy='per_image',dummy_value=dummy).run(ids,bindings,IOCounter())
            if baseline is None:baseline=result
            else:
                self.assertTrue(torch.equal(result[0],baseline[0]))
                self.assertTrue(all(torch.equal(x,y) for a,b in zip(result[1],baseline[1]) for x,y in zip(a,b)))
        a=MPICMultiServer(self.runner,selection_policy='per_image',max_new_tokens=3)
        first=a.request(self.contexts,prompt_text='t1 <image> t2 <image> t3',image_ids=['0','1'])
        second=a.request(self.contexts[::-1],prompt_text='t1 <image> t2 <image> t3',image_ids=['1','0'])
        again=a.request(self.contexts,prompt_text='t1 <image> t2 <image> t3',image_ids=['0','1'])
        self.assertEqual(first['generated_token_ids'],again['generated_token_ids']);self.assertEqual(second['bindings'][0]['image_id'],'1')
    def test_invalid_bindings_and_policy(self):
        with self.assertRaises(ValueError):expand_multi_prompt(self.runner,'t1 <image> t2',self.contexts,['0','1'])
        with self.assertRaises(ValueError):expand_multi_prompt(self.runner,'t1 <image> t2',self.contexts[:1],['wrong'])
        with self.assertRaises(ValueError):build_multi_plan(10,[(1,3),(2,4)],32,'per_image')
        with self.assertRaises(ValueError):build_multi_plan(10,[(1,3)],32,'paper')
    def fullctx(self,index):
        ctx=self.contexts[index];source=self.source[index];meta=ctx.meta
        syskv={name:torch.stack([pair[ki][0,:,:2] for pair in source]) for ki,name in enumerate(('k','v'))}
        reader=ChunkReader(ctx.dir,meta,drop_cache=False)
        return SimpleNamespace(meta=meta,reader=reader,cache=PrefixCache(meta,syskv,'cpu',dtype=torch.float32))
    def test_full_load_one_multiple_decode_prefix_and_cache_identity(self):
        contexts=[self.fullctx(i) for i in (0,1)]
        try:
            for index,suffix in [(0,[3]),(0,[3,4]),(1,[3]),(0,[3])]:
                ctx=contexts[index];stored=self.contexts[index];n=stored.meta['v_token_num']
                seen=[];h=self.lm.lm_head.register_forward_hook(lambda mod,args,val:seen.append(val.detach().clone()))
                with cpu_sync():result=Server(self.runner,max_new_tokens=3).request(ctx,mode='fullload',suffix_ids=torch.tensor(suffix),cache_id=str(index),cold=False)
                h.remove();self.assertEqual(result['io']['preads'],4)
                self.assertEqual(result['io']['bytes'],2*2*n*2*8*4)
                for li in range(2):
                    for kind,buffer in [('k',ctx.cache.k[li]),('v',ctx.cache.v[li])]:
                        raw=np.fromfile(stored.dir/f'layer_{li:02d}/{kind}.bin',dtype=np.float32).copy()
                        target=torch.from_numpy(raw).reshape(n,2,8).permute(1,0,2)[None]
                        self.assertTrue(torch.equal(buffer[:,:,2:2+n],target))
                with torch.no_grad():
                    ids=torch.tensor([1,2]+[100]*n+suffix);emb=self.lm.get_input_embeddings()(ids[None]);emb[:,2:2+n]=self.visual[str(index)][None]
                    expected=self.lm(inputs_embeds=emb).logits[:,-1:]
                torch.testing.assert_close(seen[0][:,-1:],expected,rtol=1e-5,atol=1e-6)
            with cpu_sync():
                with self.assertRaises(ValueError):Server(self.runner).request(contexts[0],mode='fullload',prompt_text='t1 t6 <image> t3',cold=False)
                with self.assertRaises(ValueError):Server(self.runner).request(contexts[0],mode='fullload',suffix_ids=torch.tensor([3]),cache_id='bad',cold=False)
                with self.assertRaises(ValueError):Server(self.runner).request(contexts[0],mode='fullload',suffix_ids=torch.tensor([4]),prompt_text='t1 t2 <image> t3',cold=False)
        finally:
            for ctx in contexts:ctx.reader.close()
    def test_prefix_requires_image_identity_and_full_reuse_is_two_pass(self):
        ids,bindings=self.input();labels=logical_labels(ids,bindings)
        with torch.no_grad():a,cache,state,matched=prefix_prefill(self.runner,ids,self.embeds(ids,bindings),labels)
        self.assertEqual(matched,0)
        with torch.no_grad():b,_,_,matched=prefix_prefill(self.runner,ids,self.embeds(ids,bindings),labels,state)
        self.assertEqual(matched,len(ids));self.assertTrue(torch.equal(a,b))
        swapped=list(labels);swapped[bindings[0].start]=(100,('different-image',0))
        with torch.no_grad():_,_,_,matched=prefix_prefill(self.runner,ids,self.embeds(ids,bindings),swapped,state)
        self.assertEqual(matched,bindings[0].start)
        io=IOCounter();logits,kv,stats=full_reuse_prefill(self.runner,ids,bindings,io)
        self.assertTrue(torch.isfinite(logits).all());self.assertEqual(stats['decoder_prefill_pass_count'],2)
        self.assertEqual([k.shape[-2] for k,v in kv],[len(ids)]*2);self.assertEqual(io.summary()['preads'],8)
    def test_mixed_preparation_serial_parallel_same_output(self):
        ids,bindings=self.input();records=[];answers=[]
        for parallel in (False,True):
            contexts,rec=prepare_mixed(['0','1'],{'0':self.contexts[0]},lambda image_id:self.contexts[1],selected_counts=[32,32],parallel=parallel)
            expanded,b=expand_multi_prompt(self.runner,'t1 t5 t2 <image> t3 t6 <image> t7 t8',contexts,['0','1'])
            io=IOCounter();answer=MultiSelectivePrefill(self.runner,selection_policy='per_image').run(expanded,b,io)[0]
            records.append(rec);answers.append(answer);self.assertTrue(rec['mixed']);self.assertEqual(rec['io']['preads'],5)
        self.assertTrue(torch.equal(answers[0],answers[1]))
    def test_history_never_defaults_to_gold(self):
        d={'turns':[{'turn_id':1,'question':'A <ImageHere> B','gold_answer':'SECRET'}, {'turn_id':2,'question':'C'}]}
        with self.assertRaises(ValueError):prompt_from_generated(d,2,{})
        text=prompt_from_generated(d,2,{1:'generated'});self.assertNotIn('SECRET',text);self.assertIn('A <image> B',text)
    def test_three_turn_prefix_logits_and_generated_mpic_history(self):
        ids,bindings=self.input();previous=None
        for turn in range(3):
            with torch.no_grad():
                logits,_,state,matched=prefix_prefill(self.runner,ids,self.embeds(ids,bindings),logical_labels(ids,bindings),previous)
                expected=self.lm(inputs_embeds=self.embeds(ids,bindings)).logits[:,-1:]
            torch.testing.assert_close(logits,expected,rtol=1e-5,atol=1e-6)
            self.assertEqual(matched,0 if previous is None else len(previous.labels))
            previous=state;ids=torch.cat((ids,torch.tensor([11,12,13])))
        server=MPICMultiServer(self.runner,selection_policy='per_image',max_new_tokens=3)
        prompt='t1 <image> t2 <image> t3'
        for turn in range(3):
            row=server.request(self.contexts,prompt_text=prompt,image_ids=['0','1'])
            prompt+=' t4 '+row['prediction']+' t5'
            self.assertEqual(len(row['cache_lengths']),2)
    def test_prefix_visual_helper_encodes_only_unmatched_images(self):
        path=Path(__file__).resolve().parents[1]/'scripts/125_eval_mpic_mmdu_smoke.py'
        spec=importlib.util.spec_from_file_location('_mmdu_visual_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        ids,b=self.input();calls=[]
        self.runner.model.config.vision_feature_layer=0;self.runner.model.config.vision_feature_select_strategy='default'
        def encoder(pixels,sizes,**kwargs):
            calls.append(pixels.shape[0]);return [self.visual[str(int(x))] for x in pixels[:,0]]
        self.runner.model.model.get_image_features=encoder
        enc={'input_ids':ids[None],'pixel_values':torch.tensor([[0],[1]]),'image_sizes':torch.ones(2,2)}
        embeds,n=module.visual_embeddings(self.runner,enc,b,None,0);self.assertEqual(n,2)
        with torch.no_grad():_,_,state,_=prefix_prefill(self.runner,ids,embeds,logical_labels(ids,b))
        reused,n=module.visual_embeddings(self.runner,{'input_ids':ids[None]},b,state,len(ids))
        self.assertEqual(n,0);self.assertTrue(torch.equal(embeds,reused));self.assertEqual(calls,[2])
        cut=b[0].start+b[0].count
        partial,n=module.visual_embeddings(self.runner,dict(input_ids=ids[None],pixel_values=torch.tensor([[1]]),image_sizes=torch.ones(1,2)),b,state,cut)
        self.assertEqual(n,1);self.assertEqual(calls,[2,1]);self.assertTrue(torch.equal(partial,embeds))


if __name__=='__main__':unittest.main()

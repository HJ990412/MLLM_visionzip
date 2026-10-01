"""CPU integration tests execute real HF LLaMA layers and projection modules."""
import unittest
from types import SimpleNamespace

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from mmimpress.sparsevlm_ssd_attention import SparseVLMSSDAttention, new_dense_cache


class Reader:
    def __init__(self, ctx):
        self.ctx = ctx
        self.reads = []
    def read_scoring_keys(self, layer):
        self.reads.append((layer, 'score'))
        k = self.ctx.keys[layer]
        return k[:, :3] if self.ctx.head_policy == 'fixed_first_3' else k
    def read_selected(self, layer, plan, scoring_keys):
        self.reads.append((layer, 'selected'))
        rows = torch.tensor(sorted(set(plan['selected_tokens']) | set(self.ctx.meta['newline_idx'])))
        return SimpleNamespace(rows=rows, keys=self.ctx.keys[layer][rows],
                               values=self.ctx.values[layer][rows], stats={})


def fixture(policy='all'):
    torch.manual_seed(1234)
    cfg = LlamaConfig(vocab_size=71, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        head_dim=8, max_position_embeddings=256, attention_dropout=0.0)
    cfg._attn_implementation = 'eager'
    model = LlamaForCausalLM(cfg).eval()
    runner = SimpleNamespace(model=model, layers=model.model.layers,
        cfg=SimpleNamespace(text_config=cfg), head_dim=8)
    ctx = SimpleNamespace(head_policy=policy,
        meta={'num_heads':4,'head_dim':8,'num_layers':2,'v_token_start':2,
              'v_token_num':9,'prefix_len':11,'newline_idx':[3], 'n_spatial':8},
        sys_kv={k:torch.randn(2,4,2,8).half() for k in ('k','v')},
        keys=torch.randn(2,9,4,8).half(), values=torch.randn(2,9,4,8).half())
    return runner,ctx


class ScopedAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def run_path(self, runner, ctx, *, n=3, sentinel=None, observer=None,
                 controlled=None):
        cache = new_dense_cache(ctx, 'cpu', torch.float32)
        reader = Reader(ctx)
        adapter = SparseVLMSSDAttention(runner,ctx,reader,cache,
            torch.arange(n),'sparsevlm_ssd_kv25_'+('allhead' if ctx.head_policy=='all' else 'probe3'),
            observer=observer,sentinel=sentinel,controlled_selected=controlled)
        original = [x.self_attn.forward for x in runner.layers]
        with torch.inference_mode(), adapter:
            pos = torch.arange(11,11+n)
            out = runner.model(input_ids=torch.arange(4,4+n)[None],
                position_ids=pos[None],cache_position=pos,past_key_values=cache,
                attention_mask=torch.ones(1,11+n,dtype=torch.long))
            first = out.logits.clone()
            adapter.begin_decode()
            count = len(reader.reads)
            pos = torch.tensor([11+n])
            out = runner.model(input_ids=torch.tensor([[8]]),position_ids=pos[None],
                cache_position=pos,past_key_values=cache,
                attention_mask=torch.ones(1,12+n,dtype=torch.long))
            self.assertEqual(len(reader.reads),count)
        self.assertEqual(original,[x.self_attn.forward for x in runner.layers])
        for x in runner.layers:
            self.assertFalse(hasattr(x.self_attn,'_sparsevlm_ssd_adapter'))
            self.assertFalse(x.self_attn.q_proj._forward_hooks)
        return first,out.logits,adapter.stats()

    def test_prefill_one_and_many_tokens_projection_decode_and_masks(self):
        for n in (1,3,65):
            for policy in ('all','fixed_first_3'):
                runner,ctx=fixture(policy)
                def check(d):
                    p=d['cache_position'];mask=d['mask'][0,0]
                    for row,pos in enumerate(p):
                        self.assertTrue(torch.all(mask[row,int(pos)+1:] < -1e20))
                    self.assertTrue(d['keep'][3])
                    self.assertEqual(int(d['keep'].sum()),3)
                    self.assertTrue(torch.all(d['mask'][...,2:11][...,~d['keep']] < -1e20))
                # Long history IDs remain in tiny vocabulary.
                if n==65:
                    n=60
                a,b,stats=self.run_path(runner,ctx,n=n,observer=check)
                self.assertEqual(stats['scoring_calls'],2)
                for calls in stats['projection_calls'].values():
                    self.assertEqual(calls['prefill'],{'q':1,'k':1,'v':1})
                    self.assertEqual(calls['decode'],{'q':1,'k':1,'v':1})

    def test_finite_sentinels_do_not_change_actual_prefill_or_decode(self):
        for policy in ('all','fixed_first_3'):
            runner,ctx=fixture(policy)
            a,b,_=self.run_path(runner,ctx)
            x,y,_=self.run_path(runner,ctx,sentinel=1234.0)
            self.assertTrue(torch.equal(a,x))
            self.assertTrue(torch.equal(b,y))

    def test_controlled_equal_selection_and_method_order_isolation(self):
        runner,ctx=fixture('all')
        ids={0:[0,1],1:[0,1]}
        a,b,_=self.run_path(runner,ctx,controlled=ids)
        ctx.head_policy='fixed_first_3'
        x,y,_=self.run_path(runner,ctx,controlled=ids)
        self.assertTrue(torch.equal(a,x));self.assertTrue(torch.equal(b,y))
        ctx.head_policy='all'
        x,y,_=self.run_path(runner,ctx,controlled=ids)
        self.assertTrue(torch.equal(a,x));self.assertTrue(torch.equal(b,y))

    def test_exception_restores_instances_and_unpatched_output(self):
        runner,ctx=fixture()
        inp=torch.tensor([[1,2,3]])
        with torch.inference_mode(): expected=runner.model(inp).logits.clone()
        def fail(d):raise RuntimeError('intentional observer failure')
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            self.run_path(runner,ctx,observer=fail)
        for layer in runner.layers:
            self.assertNotIn('forward',layer.self_attn.__dict__)
            self.assertFalse(layer.self_attn.q_proj._forward_hooks)
        with torch.inference_mode():actual=runner.model(inp).logits
        self.assertTrue(torch.equal(expected,actual))

    def test_gqa_and_unknown_policy_fail_closed(self):
        runner,ctx=fixture()
        runner.cfg.text_config.num_key_value_heads=2
        with self.assertRaises(ValueError):
            SparseVLMSSDAttention(runner,ctx,Reader(ctx),None,torch.tensor([0]),
                                  'sparsevlm_ssd_kv25_allhead')
        with self.assertRaises(ValueError):
            SparseVLMSSDAttention(runner,ctx,Reader(ctx),None,torch.tensor([0]),'probe0')

if __name__=='__main__': unittest.main()

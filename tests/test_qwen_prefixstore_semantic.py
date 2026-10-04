import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

spec=importlib.util.spec_from_file_location('qwen_semantic_test',Path(__file__).resolve().parents[1]/'scripts/116_qwen_prefixstore_semantic.py')
G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)

class SemanticValidationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.layers=[(torch.randn(1,4,13,128).bfloat16(),torch.randn(1,4,13,128).bfloat16()) for _ in range(28)]
        self.cap=SimpleNamespace(visual_count=9,visual_start=2,prefix_ids=list(range(13)),scores=torch.tensor([1.,4.,4.,3.,0.,9.,2.,5.,8.]),layers=self.layers)
        self.rank,self.chosen,self.structural,self.logical=G.independent_indices(self.cap)

    def test_original_budget_stable_rank_and_structural(self):
        self.assertEqual(self.rank,[5,8,7,1,2,3,6,0,4])
        self.assertEqual(self.chosen,[5,8,7]);self.assertEqual(self.structural,[0,1,11,12])
        self.assertEqual(self.logical,[0,1,7,9,10,11,12])
        self.assertEqual(G.gather_canonical(self.cap,self.logical)[0][0].shape,(1,4,7,128))

    def test_budget_against_original_not_stored(self):
        G.validate_budget(349,88,88,list(range(88)))
        with self.assertRaises(AssertionError):G.validate_budget(349,88,22,list(range(22)))

    def test_semantic_fp64_and_future_mask(self):
        q=torch.randn(1,28,4,128).bfloat16();suffixk=torch.randn(1,4,4,128).bfloat16();suffixv=torch.randn(1,4,4,128).bfloat16()
        ck,cv=self.layers[0];ix=torch.tensor(self.logical)
        k=torch.cat((ck[:,:,ix],suffixk),2);v=torch.cat((cv[:,:,ix],suffixv),2)
        cm=G.visible_mask(self.logical,13,4);dm=G.visible_mask(self.logical,13,4,True)
        a=G.attention(q,k,v,cm,torch.float64)['output']
        b=G.attention(q,torch.cat((ck,suffixk),2),torch.cat((cv,suffixv),2),dm,torch.float64)['output']
        self.assertTrue(G.stats(a,b,1e-12,1e-12)['allclose'])
        bad=cm.clone();bad[0,-1]=True
        self.assertFalse(G.stats(a,G.attention(q,k,v,bad,torch.float64)['output'],1e-12,1e-12)['allclose'])

    def test_same_shape_semantic_errors_rejected(self):
        layers=G.gather_canonical(self.cap,self.logical);pos=torch.arange(13)[None,None].expand(3,1,13);mask=G.visible_mask(self.logical,13,4)
        base=dict(logical=self.logical,expected_logical=self.logical,positions=pos,expected_positions=pos,mask=mask,expected_mask=mask,layers=layers,expected_layers=layers)
        G.validate_structure(**base)
        bad=self.logical.copy();bad[2]=3
        dup=self.logical.copy();dup[0]=dup[1]
        bp=pos.clone();bp[0,0,0]+=1
        bm=mask.clone();bm[0,-1]=True
        bv=[(k.clone(),v.clone()) for k,v in layers];bv[0][1][0,0,0,0]+=1
        for change in ({'logical':bad},{'logical':dup},{'positions':bp},{'mask':bm},{'layers':bv}):
            with self.subTest(change=list(change)),self.assertRaises(AssertionError):G.validate_structure(**{**base,**change})

if __name__=='__main__':unittest.main()

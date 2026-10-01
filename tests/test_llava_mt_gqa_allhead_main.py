"""Integration safety tests; CPU-only and no model load."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_new_mt_main_test',ROOT/'scripts/100_eval_llava_mt_gqa_allhead_kv25.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

class MainSafety(unittest.TestCase):
    def dialog(self):
        return {'dialog_id':'d1','turns':[{'turn_id':t,'question_id':f'q{t}','question':f'Question {t}?','answers':['GOLD_SENTINEL']} for t in (1,2,3)]}
    def row(self,t,method='ours_kv25',dialog='d1',prediction=''):
        return {'method_id':method,'dialog_id':dialog,'prediction':prediction,'logical_request_id':f'logical{t}','physical_execution_id':f'physical{t}'}
    def test_exact_empty_generated_answer_and_no_gold_future(self):
        prompt,hist,entries=M.prompt_factory(self.dialog(),2,'ours_kv25',{1:self.row(1)})
        self.assertIn('A1: ',hist);self.assertNotIn('GOLD_SENTINEL',prompt);self.assertNotIn('Question 3',prompt)
        self.assertEqual(entries[0]['answer'],'');self.assertEqual(entries[0]['source_physical_execution_id'],'physical1')
    def test_foreign_and_future_history_fail_closed(self):
        for history in ({1:self.row(1,method='fullload')},{1:self.row(1,dialog='d2')},{1:self.row(1),2:self.row(2)}):
            with self.assertRaises(ValueError):M.prompt_factory(self.dialog(),2,'ours_kv25',history)
    def test_suffix_span_coverage_with_template_and_history(self):
        p,h,e=M.prompt_factory(self.dialog(),3,'ours_kv25',{1:self.row(1,prediction='YES!'),2:self.row(2,prediction='b')})
        spans=M.text_spans(p,'Question 3?',h,e);coverage=[0]*len(p)
        for s in spans:
            for i in range(s['start_char'],s['end_char']):coverage[i]+=1
        self.assertEqual(coverage[p.index('<image>')+7:],[1]*len(coverage[p.index('<image>')+7:]))
    def test_cleanup_requires_atomic_commit_audit_and_exact_owned_allowlist(self):
        with tempfile.TemporaryDirectory() as td:
            s=Path(td)/'scratch';s.mkdir();p=s/'payload';p.write_bytes(b'preserve')
            files={'payload':M.sha(p)}
            with self.assertRaises(ValueError):M.cleanup_committed(s,files,{'status':'PASS'},{'status':'RUNNING'})
            with self.assertRaises(ValueError):M.cleanup_committed(s,{}, {'status':'PASS'},{'status':'COMMITTED'})
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):M.cleanup_committed(s,files,{'status':'PASS'},{'status':'COMMITTED'})
            self.assertTrue(p.exists())
    def test_cleanup_refuses_hardlink_and_external_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            outside=Path(td)/'old';outside.write_bytes(b'old')
            s=Path(td)/'scratch';s.mkdir();p=s/'payload';os.link(outside,p)
            with self.assertRaises(ValueError):M.cleanup_committed(s,{'payload':M.sha(p)},{'status':'PASS'},{'status':'COMMITTED'})
            p.unlink();p.symlink_to(outside)
            with self.assertRaises(ValueError):M.cleanup_committed(s,{'payload':M.sha(outside)},{'status':'PASS'},{'status':'COMMITTED'})
            self.assertEqual(outside.read_bytes(),b'old')
    def test_mpic_projection_phase_tracks_direct_prefill_then_model_decode(self):
        class Attn(torch.nn.Module):
            def __init__(self):
                super().__init__();self.q_proj=torch.nn.Linear(2,2);self.k_proj=torch.nn.Linear(2,2);self.v_proj=torch.nn.Linear(2,2)
            def forward(self,x):return self.q_proj(x)+self.k_proj(x)+self.v_proj(x)
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.attn=Attn()
            def forward(self,x):return self.attn(x)
        model=Model();runner=types.SimpleNamespace(model=model,layers=[types.SimpleNamespace(self_attn=model.attn)])
        with M.MPICCounts(runner) as c:
            model.attn(torch.zeros(1,2));model(torch.zeros(1,2));model(torch.zeros(1,2))
        self.assertEqual(c.calls['0']['prefill'],{'q':1,'k':1,'v':1});self.assertEqual(c.calls['0']['decode'],{'q':2,'k':2,'v':2})
    def test_no_clobber_publication(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'image.json';M.atomic_json(p,{'status':'COMMITTED'})
            with self.assertRaises(FileExistsError):M.atomic_json(p,{'status':'BROKEN'})
            self.assertEqual(json.loads(p.read_text())['status'],'COMMITTED')
    def test_partial_image_without_commit_is_never_adopted(self):
        with tempfile.TemporaryDirectory() as td:
            run=Path(td);a=run/'main/images/x/attempt_0001';a.mkdir(parents=True);(a/'raw.jsonl').write_text('{"partial":true}\n')
            self.assertEqual(M.completed_images(run,'main',{},{}),{})
            self.assertTrue((a/'raw.jsonl').exists())
if __name__=='__main__':unittest.main()

"""Fail-closed pilot lifecycle and prompt provenance checks, without GPUs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_sparse_runner_cpu',ROOT/'scripts/98_eval_sparsevlm_ssd_kv25.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

class TestSparsePilot(unittest.TestCase):
    def test_frozen_workload(self):
        gqa,mt,proof=M.V.frozen_workloads()
        self.assertEqual((len(gqa),len(mt)),(40,40))
        self.assertTrue(all(len(x['questions'])==6 for x in gqa))
        self.assertTrue(all(len(x['turns'])==3 for x in mt))
        self.assertEqual((mt[0]['dialog_id'],mt[0]['image_id']),('mtgqa_002749','n130464'))

    def test_projection_counter_records_actual_prefill_duplicates(self):
        import torch
        import types
        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__();self.q_proj=torch.nn.Linear(2,2);self.k_proj=torch.nn.Linear(2,2);self.v_proj=torch.nn.Linear(2,2)
            def forward(self,x,extra):
                if extra:self.q_proj(x)
                return self.q_proj(x)+self.k_proj(x)+self.v_proj(x)
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.attn=Attention()
            def forward(self,x,extra=False):return self.attn(x,extra)
        model=Model();runner=types.SimpleNamespace(model=model,layers=[types.SimpleNamespace(self_attn=model.attn)])
        with M.PilotCounts(runner) as counts:
            model(torch.zeros(1,2),extra=True);model(torch.zeros(1,2))
        self.assertEqual(counts.result()['0']['prefill'],{'q':2,'k':1,'v':1})
        self.assertEqual(counts.result()['0']['decode'],{'q':1,'k':1,'v':1})

    def test_partial_gate_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'gate.json';p.write_text(json.dumps({'GPU_CORRECTNESS':'PARTIAL'}))
            with self.assertRaisesRegex(RuntimeError,'not PASS'):M.gpu_gate(p)

    def test_missing_source_hashes_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'gate.json';p.write_text(json.dumps({'GPU_CORRECTNESS':'PASS','contract_sha256':M.V.sha(ROOT/'docs/sparsevlm_ssd_kv25_contract.md')}))
            with self.assertRaisesRegex(RuntimeError,'coverage incomplete'):M.gpu_gate(p)

    def test_mt_spans_cover_actual_suffix(self):
        dialog={'turns':[{'question_id':'q1','question':'What color?'},{'question_id':'q2','question':'What shape?'},{'question_id':'q3','question':'Where?'}]}
        prompt,history,entries=M.OLD.mt_prompt(dialog,3,{1:'green',2:'round'})
        spans=M.text_spans(prompt,'Where?',history,entries)
        suffix_start=prompt.index('<image>')+len('<image>')
        coverage=[0]*len(prompt)
        for span in spans:
            for i in range(span['start_char'],span['end_char']):coverage[i]+=1
        self.assertEqual(coverage[suffix_start:],[1]*(len(prompt)-suffix_start))
        self.assertEqual([x['answer'] for x in entries],['green','round'])
        self.assertEqual(len([s for s in spans if s['kind']=='current_question']),1)

    def test_resume_adopts_first_success_without_reexecution(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'raw.jsonl').touch()
            image=p/'images'/'image1';failed=image/'attempt_0001';failed.mkdir(parents=True)
            (failed/'raw.jsonl').write_text(json.dumps({'request_id':'bad'})+'\n')
            success=image/'attempt_0002';success.mkdir()
            row={'request_id':'gqa:image1:1:recompute','image_id':'image1','dialog_id':None,'attempt_id':'attempt_0002'}
            (success/'raw.jsonl').write_text(json.dumps(row)+'\n');(success/'independent_audit.json').write_text('{"status":"PASS"}')
            self.assertEqual(M.recover_committed_images(p),{'image1'})
            self.assertEqual(M.recover_committed_images(p),{'image1'})
            self.assertEqual(len((p/'raw.jsonl').read_text().splitlines()),1)
            self.assertTrue((failed/'raw.jsonl').exists())

    def test_cleanup_refuses_unapproved_or_linked_payload(self):
        with tempfile.TemporaryDirectory() as td:
            scratch=Path(td)/'scratch';scratch.mkdir();payload=scratch/'a.bin';payload.write_bytes(b'preserve')
            with self.assertRaises(ValueError):M.cleanup_committed(scratch,['a.bin'],{'status':'FAIL'})
            link=scratch/'linked.bin';link.symlink_to(payload)
            with self.assertRaises(ValueError):M.cleanup_committed(scratch,['linked.bin'],{'status':'PASS'})
            self.assertEqual(payload.read_bytes(),b'preserve')

if __name__=='__main__':unittest.main()

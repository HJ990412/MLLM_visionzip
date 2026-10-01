"""Safety invariants introduced by the Qwen main orchestrator (CPU only)."""
import importlib.util,json,os,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(n,f):
 s=importlib.util.spec_from_file_location(n,ROOT/'scripts'/f);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
M=load('_qwen_main_test','103_eval_qwen_mt_gqa_kv25_main.py');A=load('_qwen_audit_test','104_audit_qwen_mt_gqa_kv25_main.py')
class QwenMainSafety(unittest.TestCase):
 def dialog(self):return {'dialog_id':'d','turns':[{'turn_id':t,'question_id':f'q{t}','question':f'Question {t}?','answers':['GOLD_ONLY']} for t in (1,2,3)]}
 def row(self,t):return {'model_id':M.MODEL_ID,'method_id':M.METHODS[2],'dialog_id':'d','turn_id':t,'prediction':'','logical_request_id':f'l{t}','physical_execution_id':f'p{t}'}
 def test_own_empty_prediction_history_kept_no_gold_future(self):
  q,h,e=M.history_factory(self.dialog(),2,M.METHODS[2],{1:self.row(1)});text=A.prompt_text(q,h)
  self.assertEqual(h,(('Question 1?',''),));self.assertEqual(e[0]['source_physical_execution_id'],'p1');self.assertNotIn('GOLD_ONLY',text);self.assertNotIn('Question 3',text)
 def test_foreign_model_method_dialog_turn_rejected(self):
  for key,value in [('model_id','llava'),('method_id','fullload'),('dialog_id','other'),('turn_id',2)]:
   r=self.row(1);r[key]=value
   with self.assertRaises(ValueError):M.history_factory(self.dialog(),2,M.METHODS[2],{1:r})
 def test_future_or_missing_history_rejected(self):
  for hist in ({},{1:self.row(1),2:self.row(2)}):
   with self.assertRaises(ValueError):M.history_factory(self.dialog(),2,M.METHODS[2],hist)
 def test_deterministic_threeway_rotation(self):
  positions={m:[] for m in M.METHODS}
  for i in range(6):
   for m in M.METHODS:positions[m].append(M.method_order(i).index(m))
  for seq in positions.values():self.assertEqual(sorted(seq),[0,0,1,1,2,2])
 def test_atomic_no_clobber(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'COMMITTED.json';M.atomic_json(p,{'keep':1})
   with self.assertRaises(FileExistsError):M.atomic_json(p,{'keep':2})
   self.assertEqual(json.loads(p.read_text()),{'keep':1})
 def test_cleanup_audit_commit_allowlist_and_hash_required(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'scratch';p.mkdir();f=p/'payload';f.write_bytes(b'original');h={'payload':M.sha(f)}
   for a,c in [({'status':'FAIL'},{'status':'COMMITTED'}),({'status':'PASS'},{'status':'RUNNING'})]:
    with self.assertRaises(ValueError):M.cleanup_committed(p,h,a,c)
   with self.assertRaises(ValueError):M.cleanup_committed(p,{}, {'status':'PASS'},{'status':'COMMITTED'})
   f.write_bytes(b'changed')
   with self.assertRaises(ValueError):M.cleanup_committed(p,h,{'status':'PASS'},{'status':'COMMITTED'})
   self.assertTrue(f.exists())
 def test_cleanup_rejects_external_symlink_hardlink(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);old=root/'old';old.write_bytes(b'preserve');p=root/'scratch';p.mkdir();f=p/'payload'
   os.link(old,f)
   with self.assertRaises(ValueError):M.cleanup_committed(p,{'payload':M.sha(old)},{'status':'PASS'},{'status':'COMMITTED'})
   f.unlink();f.symlink_to(old)
   with self.assertRaises(ValueError):M.cleanup_committed(p,{'payload':M.sha(old)},{'status':'PASS'},{'status':'COMMITTED'})
   self.assertEqual(old.read_bytes(),b'preserve')
 def test_partial_uncommitted_image_never_adopted(self):
  with tempfile.TemporaryDirectory() as td:
   run=Path(td);a=run/'main/images/i/attempt_0001';a.mkdir(parents=True);(a/'raw.jsonl').write_text('{"partial":true}\n')
   self.assertEqual(M.completed_images(run,'main',{},{}),{});self.assertTrue((a/'raw.jsonl').exists())
 def test_scorer_parity_and_no_normalization_in_history(self):
  self.assertEqual(M.strict_score('The red.','red'),1);self.assertEqual(A.score('The red.',['red']),1)
  r=self.row(1);r['prediction']='The red.';q,h,e=M.history_factory(self.dialog(),2,M.METHODS[2],{1:r});self.assertEqual(h[0][1],'The red.')
if __name__=='__main__':unittest.main()

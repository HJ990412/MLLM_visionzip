import copy,unittest,tempfile
from pathlib import Path
from mmimpress.mtvqa25k.dataset import join_sources,select
from mmimpress.mtvqa25k.scorer import reference_scores,independent_score
from mmimpress.mtvqa25k.common import dump,read

class ReconstructionTests(unittest.TestCase):
    def test_deterministic_membership_distinct_ids_duplicate_text_allowed(self):
        qs=[{'question_id':i*100+j,'image_id':i,'question':'What is this?'} for i in range(6) for j in range(3)]
        anns=[dict(question_id=q['question_id'],image_id=q['image_id'],answer_type='other',question_type='what',
                   answers=[dict(answer_id=j,answer='cat') for j in range(1,11)]) for q in qs]
        a,b,g=join_sources(qs,anns)
        eligible,chosen=select(g,set(range(6)),target=5)
        self.assertEqual(chosen,select(dict(reversed(list(g.items()))),set(range(6)),target=5)[1])
        self.assertEqual(len({i for i,q in chosen}),5)
        for i,q in chosen:self.assertEqual(set(q),set(g[i]))
        with self.assertRaises(ValueError):join_sources(qs+[qs[0]],anns)
        with self.assertRaises(ValueError):select(g,{0,1},target=3)
        bad=copy.deepcopy(anns);bad[0]['answers'][0]['answer_id']=2
        with self.assertRaises(ValueError):join_sources(qs,bad)
    def test_leave_one_out_and_unanimous_official_branch(self):
        for count,expected in [(0,0),(1,.3),(2,.6),(3,.9),(4,1),(10,1)]:
            ann={'question_id':1,'answer_type':'other','question_type':'what',
                 'answers':[dict(answer_id=i+1,answer='cat' if i<count else 'dog') for i in range(10)]}
            original=copy.deepcopy(ann)
            scores,_=reference_scores([ann],{1:'cat'})
            self.assertAlmostEqual(scores[1],expected)
            self.assertAlmostEqual(independent_score('cat',ann)['vqa_score'],expected)
            self.assertEqual(ann,original)
        self.assertEqual(reference_scores([ann],{1:'Cat'})[0][1],0)
    def test_atomic_no_clobber(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'commit.json';dump(p,{'state':'A'})
            with self.assertRaises(FileExistsError):dump(p,{'state':'B'})
            self.assertEqual(read(p),{'state':'A'})
    def test_lossless_archive_is_bound_before_owned_cleanup(self):
        import json
        from mmimpress.mtvqa25k.adapter import archive_before_commit,remove_archived_duplicates,read_rows
        from mmimpress.mtvqa25k.common import sha
        with tempfile.TemporaryDirectory() as td:
            parent=Path(td);folder=parent/'attempt_0001';folder.mkdir();raw=folder/'raw.jsonl'
            raw.write_text(json.dumps({'answer':'two words','token_ids':[1,2,3]})+'\n')
            h=sha(raw);commit={'status':'COMMITTED','attempt_id':folder.name,'raw_sha256':h,'evidence_sha256':{'raw.jsonl':h}}
            archive_before_commit(parent,commit,dump)
            self.assertTrue(raw.exists())
            remove_archived_duplicates(parent)
            self.assertFalse(raw.exists());self.assertEqual(read_rows(raw)[0]['answer'],'two words')
            self.assertEqual(read(parent/'COMMITTED.json')['journal_archives']['raw.jsonl']['uncompressed_sha256'],h)

if __name__=='__main__':unittest.main()

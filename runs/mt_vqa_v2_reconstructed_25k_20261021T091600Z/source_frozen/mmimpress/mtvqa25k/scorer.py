"""Frozen official VQA evaluator. No semantic changes to upstream normalization.

The pinned upstream revision normalizes punctuation/digits/articles only when
the ten whitespace-cleaned GT strings are not unanimous. Preserve that branch;
unconditional normalization would be a different metric.
"""
import contextlib, copy, io
from types import SimpleNamespace
from .vendor.vqaEval import VQAEval

REVISION='a013f0043c1e2cdc995922dfe257f7149aa9af06'

class Annotations:
    def __init__(self,annotations):
        self.qa={int(a['question_id']):copy.deepcopy(a) for a in annotations}
        if len(self.qa)!=len(annotations):raise ValueError('duplicate question IDs')
        for a in self.qa.values():
            if len(a['answers'])!=10 or sorted(x['answer_id'] for x in a['answers'])!=list(range(1,11)):
                raise ValueError('exactly ten uniquely identified human answers required')
    def getQuesIds(self):return list(self.qa)

def reference_scores(annotations,predictions):
    gt=Annotations(annotations)
    if set(gt.qa)!=set(predictions):raise ValueError('prediction membership differs')
    res=SimpleNamespace(qa={qid:{'answer':a} for qid,a in predictions.items()})
    evaluator=VQAEval(gt,res,n=10)
    with contextlib.redirect_stdout(io.StringIO()):evaluator.evaluate()
    return {qid:value/100 for qid,value in evaluator.evalQA.items()},evaluator.accuracy

def normalization(answer,annotation):
    evaluator=VQAEval(Annotations([annotation]),None,n=10)
    clean=lambda s:s.replace('\n',' ').replace('\t',' ').strip()
    gt=[clean(a['answer']) for a in annotation['answers']]
    pred=clean(answer);applied=len(set(gt))>1
    if applied:
        norm=lambda s:evaluator.processDigitArticle(evaluator.processPunctuation(s))
        pred=norm(pred);gt=[norm(x) for x in gt]
    return pred,gt,applied

def independent_score(answer,annotation):
    """Independent count identity of leave-one-out, using official normalization.

    Comparison with the unmodified evaluate loop is mandatory before main.
    Answer IDs guarantee distinct annotation dictionaries in upstream's != test.
    """
    pred,gt,applied=normalization(answer,annotation)
    matches=sum(x==pred for x in gt)
    value=(matches*min(max(matches-1,0)/3,1)+(10-matches)*min(matches/3,1))/10
    return {'vqa_score':value,'normalized_answer':pred,'normalization_applied':applied,
            'normalized_human_answers':gt,'matches':matches,'legacy_same_normalization':min(matches/3,1)}

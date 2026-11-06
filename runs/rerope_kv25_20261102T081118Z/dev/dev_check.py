"""Implementation check of the new paths on one dev image (no measurement, scratchpad stores)."""
import importlib.util, json, sys, shutil
from pathlib import Path
import torch
ROOT = Path('/home/dblab/hj/mllm_v2'); sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('m', ROOT / 'scripts/136_eval_rerope_kv25.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from mmimpress.serve import suffix_ids_from_prompt
from mmimpress.recompute_kv25 import StoreRows, bit_equal
run = Path(sys.argv[1]); (run / 'llava').mkdir(parents=True, exist_ok=True)
exp = m.ReRoPE(run)
man = json.loads((m.REV3_RUN / 'gqa_manifest.json').read_text())['images']
image = next(x for x in man if x['image_id'] == 'n272098'); turn = image['turns'][0]
path = run / 'llava' / 'stores' / 'dev' / image['image_id']
if path.exists(): shutil.rmtree(path)
v, cap = exp.t1(image, turn, 'ours', 'gqa'); print('t1 margin', v['first_token_top2'])
recs = {}
for arm in ['ours', 'compact']:
    v2, cap2 = exp.t1(image, turn, arm, 'gqa')
    recs[arm], _, _ = exp.persist_arm(cap2, image, arm, path / arm); del cap2
recs['rerope'], layers = exp.persist_rerope(cap, image, path / 'rerope')
print('rerope', recs['rerope']['rerope'])
g = exp.rerope_gates(image, recs, exp.oracle(cap), 'dev')
print('gates', g['status'], {k: g['Gab']['gates'][k][0] for k in ['same_position_layer0_k_S', 'neg_unrotated_layer0_k_S']})
ctx = exp.activate_arm(path / 'rerope', image, 'rerope')[0]
h = exp.hit_arm(ctx, path / 'rerope', image, image['turns'][1], 'rerope', 'gqa', sentinel=True)
print('rerope hit', h['prediction'], h['position_check'], h['first_token_top2'], h['actual_io']['bytes'])
cctx = exp.activate_arm(path / 'compact', image, 'compact')[0]
ch = exp.hit_arm(cctx, path / 'compact', image, image['turns'][1], 'compact', 'gqa', logits=True)
print('compact hit', ch['prediction'], ch['first_token_top2'])
p1, d1 = exp.pruned_request(image, image['turns'][1], 'gqa')
print('pruned', p1['prediction'], p1['ttft_ms'], p1['ttft_breakdown_ms'], p1['position_check'], p1['first_token_top2'])
suffix = suffix_ids_from_prompt(exp.runner, exp.prompt(image['turns'][1]['question'], [], 'gqa'))
two, pl = exp.pruned_two_step(d1, suffix)
cs = StoreRows(path / 'compact'); rank = cs.meta['full_importance_permutation'][:cs.k]; vs = cs.meta['v_token_start']
where = {r: j for j, r in enumerate(d1['plan']['input_index'])}; rows = torch.tensor([where[vs + i] for i in rank], device=pl[0][0].device)
ident = all(bit_equal(pl[li][kk][0, :, rows].transpose(0, 1).to(torch.float16).cpu(), cs.content(li, 'k' if kk == 0 else 'v')) for li in range(32) for kk in (0, 1))
print('two-step', two['answer'], two['first_token_id'], 'prefix==compact store', ident, 'compact hit first', ch['first_token_id'], 'pruned first', p1['first_token_id'])
ctx.close(); cctx.close(); exp.close_contexts()

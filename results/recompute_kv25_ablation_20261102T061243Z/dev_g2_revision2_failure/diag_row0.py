"""Row-0 (common-prefix) deep-layer V drift: pure shape controls. No timing/stores."""
import importlib.util, json, sys
from pathlib import Path
import torch
ROOT = Path('/home/dblab/hj/mllm_v2'); sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('m', ROOT/'scripts/134_eval_recompute_kv25_ablation.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from mmimpress.recompute_kv25 import reduced_plan, reduced_prefill, diff_stats
from mmimpress.model import cache_layers
from mmimpress.cvpr25 import anyres_token_scores, visionzip_repack_order
exp = m.Ablation(Path(sys.argv[1])); R = exp.runner
man = json.loads((m.REF_RUN/'gqa_manifest.json').read_text())['images']
out = {}
for img in ['n51002', 'n272098', 'n331357']:
    image = next(x for x in man if x['image_id'] == img); turn = image['turns'][0]
    v, cap = exp.t1(image, turn, 'ours', 'gqa'); A = cache_layers(cap['cache'])
    enc = cap['enc']; vs, vn = R.visual_span(enc['input_ids']); base, _, _, sep = R.anyres_layout(enc['image_sizes'][0], vn)
    order = visionzip_repack_order(anyres_token_scores(R, cap['vision'].result_cpu(), enc['image_sizes'][0], vn, base_side=base), sep)
    plan = reduced_plan(vs, vn, order, sep, 'original'); L = len(plan['input_index'])
    first_drop = min(i for i in range(vn) if i not in set(plan['keep_visual_ids']))
    rows = list(range(vs, vs + first_drop))   # common-prefix rows (absolute)
    res = {'common_rows': len(rows), 'reduced_len': L}
    controls = {'orig_reduced': plan['input_index'],
                'truncated_same_len': list(range(L)),          # identical context for rows < L
                'truncated_half_len': list(range(L // 2)),
                'prefix_only': list(range(vs + vn))}
    for name, idx in controls.items():
        layers, _ = reduced_prefill(R, cap['embeds'], {'policy': 'original', 'input_index': idx, 'position_ids': idx})
        where = {r: j for j, r in enumerate(idx)}
        def rel(li, kind, rr):
            a = layers[li][kind][0][:, [where[r] for r in rr]].float().cpu(); b = A[li][kind][0][:, rr].float().cpu()
            return round(diff_stats(a, b)['rel_l2'], 4)
        res[name] = {'row0_v_by_layer': [rel(li, 1, rows) for li in (1, 8, 16, 24, 28, 31)],
                     'row0_v_max': max(rel(li, 1, rows) for li in range(32)), 'row0_k_max': max(rel(li, 0, rows) for li in range(32)),
                     'sys_v_max': max(rel(li, 1, list(range(vs))) for li in range(32))}
        del layers
    out[img] = res; print(img, json.dumps(res), flush=True)
    del cap, A; torch.cuda.empty_cache()
(Path(sys.argv[1])/'diag_row0.json').write_text(json.dumps(out, indent=1))

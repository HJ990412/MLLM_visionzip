"""Rev.-3 implementation check on dev image n51002: structural bit-identity (BF16 + FP16 cast). No stores/timers."""
import importlib.util, json, sys
from pathlib import Path
import torch
ROOT = Path('/home/dblab/hj/mllm_v2'); sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('m', ROOT/'scripts/134_eval_recompute_kv25_ablation.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from mmimpress.recompute_kv25 import reduced_plan, reduced_prefill, synthetic_layers, truncation_control, structural_identity, bit_equal
from mmimpress.cvpr25 import anyres_token_scores, visionzip_repack_order
exp = m.Ablation(Path(sys.argv[1])); R = exp.runner
man = json.loads((m.REF_RUN/'gqa_manifest.json').read_text())['images']
for img in ['n51002', 'n272098']:
    image = next(x for x in man if x['image_id'] == img); turn = image['turns'][0]
    v, cap = exp.t1(image, turn, 'orig', 'gqa'); enc = cap['enc']
    vs, vn = R.visual_span(enc['input_ids']); base, _, _, sep = R.anyres_layout(enc['image_sizes'][0], vn)
    order = visionzip_repack_order(anyres_token_scores(R, cap['vision'].result_cpu(), enc['image_sizes'][0], vn, base_side=base), sep)
    for pol in ['original', 'compact']:
        plan = reduced_plan(vs, vn, order, sep, pol)
        red, _ = reduced_prefill(R, cap['embeds'], plan)
        syn = synthetic_layers(red, plan, vs + vn)
        ctl = truncation_control(R, cap['embeds'], plan)
        a = structural_identity(red, ctl); b = structural_identity(syn, ctl)
        fp16 = all(bit_equal(s[0, :, ctl['rows']].transpose(0, 1).to(torch.float16).cpu(), c.to(torch.float16))
                   for (sk, sv), (ck, cv) in zip(syn, ctl['layers']) for s, c in ((sk, ck), (sv, cv)))
        print(img, pol, 'rows', len(ctl['rows']), 'L', ctl['length'], 'bf16_reduced', a['ok'], 'bf16_synth', b['ok'], 'fp16', fp16,
              'ms', round(ctl['control_ms'], 1), a['mismatches'][:3], flush=True)
        del red, syn, ctl
    del cap; torch.cuda.empty_cache()

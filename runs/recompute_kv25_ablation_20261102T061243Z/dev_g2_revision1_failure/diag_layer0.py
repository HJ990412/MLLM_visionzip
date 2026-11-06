"""Diagnostics for the G2 mismatch: numerics vs. logic. No timing, no stores."""
import importlib.util, json, sys
from pathlib import Path
import torch
ROOT = Path('/home/dblab/hj/mllm_v2'); sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('m', ROOT/'scripts/134_eval_recompute_kv25_ablation.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from mmimpress.recompute_kv25 import reduced_plan, reduced_prefill, diff_stats, within, LAYER0_TOL, rotate
from mmimpress.model import cache_layers
from mmimpress.cvpr25 import anyres_token_scores, visionzip_repack_order
dev = Path(sys.argv[1]); out = {}
exp = m.Ablation(dev); R = exp.runner
samples = json.loads((dev/'correctness_manifest.json').read_text())['samples']
for idx in [0, 3]:
    s = samples[idx]; turn = {'question': s['question'], 'question_id': s['question_id']}
    v1, c1 = exp.t1(s, turn, 'ours', 'gqa'); v2, c2 = exp.t1(s, turn, 'ours', 'gqa')
    A = cache_layers(c1['cache']); B = cache_layers(c2['cache'])
    T = c1['embeds'].shape[1]
    enc = c1['enc']; vs, vn = R.visual_span(enc['input_ids']); base, _, _, sep = R.anyres_layout(enc['image_sizes'][0], vn)
    order = visionzip_repack_order(anyres_token_scores(R, c1['vision'].result_cpu(), enc['image_sizes'][0], vn, base_side=base), sep)
    plan = reduced_plan(vs, vn, order, sep, 'original')
    S = [vs + i for i in plan['selected_original_ids']]; P = vs + vn
    def sub(layers, rows, li, kind):     # (rows, H, hd) float
        t = layers[li][0 if kind == 'k' else 1][0][:, rows].permute(1, 0, 2).float().cpu(); return t
    def custom(idx_list):
        p = {'policy': 'original', 'input_index': idx_list, 'position_ids': idx_list}
        layers, _ = reduced_prefill(R, c1['embeds'], p); return layers
    full = custom(list(range(T)))                   # same sequence through the reduced path
    pref = custom(list(range(P)))                   # same context, question suffix removed (shape control)
    red = custom(plan['input_index'])               # actual Orig reduced prefill
    def where(layers, rows): return {r: j for j, r in enumerate(layers)}
    pos_red = {r: j for j, r in enumerate(plan['input_index'])}
    def comp(X, xrows, Y, yrows, li, kind):
        a = X[li][0 if kind=='k' else 1][0][:, xrows].permute(1,0,2).float().cpu()
        b = Y[li][0 if kind=='k' else 1][0][:, yrows].permute(1,0,2).float().cpu()
        ok, st = within(a, b, LAYER0_TOL); return {'ok': ok, **{k: round(v, 6) for k, v in st.items() if k != 'rows'}}
    sysr = list(range(vs)); Sred = [pos_red[r] for r in S]
    res = {'T': T, 'prefix_len': P, 'reduced_len': len(plan['input_index'])}
    for name, X, xs, xsys in [('t1_repeat', B, S, sysr), ('full_seq_reduced_path', full, S, sysr),
                               ('prefix_only_shape_control', pref, S, sysr), ('orig_reduced', red, Sred, sysr)]:
        res[name] = {'layer0_S_k': comp(X, xs, A, S, 0, 'k'), 'layer0_S_v': comp(X, xs, A, S, 0, 'v'),
                     'sys_k_by_layer_rel_l2': [comp(X, xsys, A, sysr, li, 'k')['rel_l2'] for li in range(32)][::4],
                     'sys_v_by_layer_rel_l2': [comp(X, xsys, A, sysr, li, 'v')['rel_l2'] for li in range(32)][::4],
                     'sys_v_last': comp(X, xsys, A, sysr, 31, 'v'),
                     'S_k_last_rel_l2': comp(X, xs, A, S, 31, 'k')['rel_l2'], 'S_v_last_rel_l2': comp(X, xs, A, S, 31, 'v')['rel_l2']}
    # FP32 oracle for layer-0 K/V of S: RMSNorm -> dequantized NF4 proj -> RoPE, all FP32
    import bitsandbytes.functional as bf
    layer = R.model.model.language_model.layers[0]
    x = c1['embeds'][0, S].float()
    ln = layer.input_layernorm; xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + ln.variance_epsilon) * ln.weight.float()
    def deq(lin): return bf.dequantize_4bit(lin.weight.data, lin.weight.quant_state).float()
    k = (xn @ deq(layer.self_attn.k_proj).T).view(len(S), 32, 128).cpu().double()
    v = (xn @ deq(layer.self_attn.v_proj).T).view(len(S), 32, 128).cpu().double()
    k = rotate(k, torch.tensor(S, dtype=torch.float64), exp.inv_freq)
    for name, X, xs in [('ours', A, S), ('orig', red, Sred)]:
        a_k = X[0][0][0][:, xs].permute(1,0,2).double().cpu(); a_v = X[0][1][0][:, xs].permute(1,0,2).double().cpu()
        res[f'fp32_oracle_vs_{name}'] = {'k': {kk: round(vv, 6) for kk, vv in diff_stats(a_k, k).items()},
                                         'v': {kk: round(vv, 6) for kk, vv in diff_stats(a_v, v).items()},
                                         'k_max_err_over_pair_norm': float(((a_k - k).abs() / k.view(len(S),32,2,64).norm(dim=2).repeat(1,1,2).clamp_min(1e-6)).max())}
    out[s['image_id']] = res
    print(json.dumps({s['image_id']: res}, indent=1), flush=True)
    del c1, c2, A, B, full, pref, red; torch.cuda.empty_cache()
(dev/'diag_layer0.json').write_text(json.dumps(out, indent=1))

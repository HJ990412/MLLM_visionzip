"""G-a / G-b tolerance evidence from the retained rev.3 G4 FP16 stores (read-only, CPU)."""
import json, sys, torch
from pathlib import Path
sys.path.insert(0, '/home/dblab/hj/mllm_v2')
from mmimpress.recompute_kv25 import StoreRows, reduced_plan, rotate, diff_stats, default_inv_freq
root = Path('/home/dblab/hj/mllm_v2/runs/recompute_kv25_ablation_20261102T061243Z/llava/correctness')
inv = default_inv_freq(128)
out = {}
for d in sorted(root.iterdir()):
    ours, comp = StoreRows(d / 'ours'), StoreRows(d / 'compact')
    m = ours.meta; vs = int(m['v_token_start']); order = [int(x) for x in m['full_importance_permutation']]; sep = [int(x) for x in m['newline_idx']]
    plan = reduced_plan(vs, int(m['v_token_num']), order, sep, 'compact'); k = plan['k']; sel = order[:k]
    dp = torch.tensor([plan['visual_position'][i] - (vs + i) for i in sel], dtype=torch.float64)
    dsep = torch.tensor([plan['visual_position'][i] - (vs + i) for i in sep], dtype=torch.float64)
    res = {'k': k, 'layers': []}
    worst = {'rt_rel': 0, 'rt_max_abs': 0, 'rt_max_rel_elem': 0, 'rt_allclose_1e-3': True, 'bit_exact_min': 1.0}
    for li in range(ours.L):
        row = {'layer': li}
        for name, delta, rows_o, rows_c in (('S', dp, ours.content(li, 'k'), comp.content(li, 'k')),
                                            ('sep', dsep, ours.separators(li, 'k'), comp.separators(li, 'k'))):
            re16 = rotate(rows_o, delta, inv).to(torch.float16)                      # store value of Ours-ReRoPE
            back = rotate(re16, -delta, inv)                                          # G-a round trip (FP64)
            st = diff_stats(back, rows_o)
            o = rows_o.double(); scale = o.abs().amax(dim=-1, keepdim=True)
            elem = ((back - o).abs() / (o.abs() + scale)).max().item()
            close = bool(torch.allclose(back, o, rtol=1e-3, atol=1e-3 * float(o.abs().max())))
            row[f'rt_{name}'] = {**st, 'max_rel_elem_vs_abs_plus_rowmax': elem, 'allclose_rtol1e-3_atol1e-3max': close}
            worst['rt_rel'] = max(worst['rt_rel'], st['rel_l2']); worst['rt_max_abs'] = max(worst['rt_max_abs'], st['max_abs'])
            worst['rt_max_rel_elem'] = max(worst['rt_max_rel_elem'], elem); worst['rt_allclose_1e-3'] &= close
            row[f'vs_compact_{name}'] = diff_stats(re16, rows_c)                     # G-b at layer 0; report-only deeper
        res['layers'].append(row)
    res['worst_roundtrip'] = worst
    res['layer0_vs_compact'] = {n: res['layers'][0][f'vs_compact_{n}']['rel_l2'] for n in ('S', 'sep')}
    res['v_bit_identical_check'] = 'Ours V reused unchanged by construction'
    out[d.name] = res
    print(d.name, 'k', k, 'roundtrip worst', {a: (round(b, 6) if isinstance(b, float) else b) for a, b in worst.items()},
          'layer0 vs compact', {n: round(v, 5) for n, v in res['layer0_vs_compact'].items()}, flush=True)
json.dump(out, open(Path(sys.argv[1]) / 'g_ab_evidence.json', 'w'), indent=1)

"""Development check of the VZ-Pruned-E2E lifecycle on one image (scratchpad only, before freeze)."""
import sys, json, importlib.util, shutil, time, hashlib
from pathlib import Path
ROOT = Path('/home/dblab/hj/mllm_v2'); sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tests')]
import torch

def helper(file, name):
    s = importlib.util.spec_from_file_location(name, ROOT / file); m = importlib.util.module_from_spec(s); sys.modules[name] = m; s.loader.exec_module(m); return m

E = helper('scripts/147_eval_vz_pruned_e2e_mtgqa.py', 'dev147'); AUD = helper('scripts/148_audit_vz_pruned_e2e_mtgqa.py', 'dev148')
from mmimpress.vz_pruned_e2e import pruned_t1_request, path_equivalence
from mmimpress.rerope_kv25 import AbortAfterEmbeds, pruned_generate
SCR = Path(sys.argv[1]); run = SCR / 'runs' / 'vz_pruned_e2e_mtgqa_dev'; shutil.rmtree(run, ignore_errors=True); run.mkdir(parents=True)
dialogues = E.M.frozen_dialogues(); image = dialogues[0]['image_id']
ds = [d for d in dialogues if d['image_id'] == image][:2]
manifest = {'dialogues': dialogues, 'image_identity': {image: {'image_path': ds[0]['image_path'], 'sha256': E.sha(ROOT / ds[0]['image_path'])}},
            'smoke_dialogue_ids': [d['dialog_id'] for d in ds], 'integration_dialogue_ids': [], 'g8_images': [image], 'correctness_images': [image],
            'expected_requests': len(dialogues) * 9, 'methods': list(E.METHODS)}
config = {'cpu_threads': 2, 'source_sha256': {'dev': 'dev'}, '_file_sha256': 'dev', '_manifest_sha256': 'dev'}
E.atomic_json(run / 'storage_plan.json', {'minimum_reserve_bytes': E.MIN_RESERVE, 'peak_working_set_bytes': E.PEAK_WORKING_SET,
                                          'maximum_new_payload_bytes': 4 * 2**30, 'largest_image_bundle_bound_bytes': 1400 * 10**6})
E.atomic_json(run / 'gpu_freeze.json', {'files': {}}); E.atomic_json(run / 'config.json', config); E.atomic_json(run / 'manifest.json', manifest)
torch.set_num_threads(2); torch.manual_seed(1234)
t0 = time.time(); exp = E.Experiment(run, config, manifest); (run / 'llava').mkdir(exist_ok=True); print('loaded', time.time() - t0, flush=True)
t0 = time.time(); exp.run_phase('smoke', AUD); print('smoke 18 requests wall', time.time() - t0, flush=True)
cp = run / 'smoke/images' / image / 'COMMITTED.json'; doc = json.loads(cp.read_text()); folder = cp.parent / doc['attempt_id']
rows = E.load_jsonl(folder / 'raw.jsonl')
for r in rows:
    print(r['method_id'], r['turn_id'], r['dialog_id'][-4:], repr(r['prediction']), round(r['ttft_ms'], 1), round(r['request_e2e_ms'], 1), r['ssd_read_bytes'], r['k'], r['first_token_top2'])
g = json.loads((folder / 'gates.json').read_text())
print('GATES', g['status'], g['G2a'], g['G2b']['equal'], g['G2e'], g['G2c_memory_bf16'], 'wall', g['gate_wall_ms'])
for name, entries in g['G2']['gates'].items():
    bad = [e for e in entries if not e['ok']]
    print('  gate', name, 'entries', len(entries), 'fail', len(bad), {k: v for k, v in entries[0].items() if k in ('rel_l2', 'oracle_ratio', 'mismatched_rows', 'S', 'rows')})
print('report_only keys', list(g['G2']['report_only']))
print('layer last k_S_realigned', g['G2']['layers'][-1]['k_S_realigned']['rel_l2'], 'v_S', g['G2']['layers'][-1]['v_S']['rel_l2'], 'layer0 k_S_realigned', g['G2']['layers'][0]['k_S_realigned']['rel_l2'])
for kind in ('image_only', 'vz'):
    p = json.loads((folder / f'{kind}_persistence.json').read_text()); print(kind, 'persist ms', round(p['service_persistence_ms'], 1), 'bytes', p['file_logical_bytes'], 'synthetic', p.get('synthetic'))
print('g8', json.dumps(json.loads((folder / 'g8_hit_vs_standalone.json').read_text())['pairs'], indent=None)[:1500])
print('audit', json.loads((folder / 'independent_audit.json').read_text())['status'], json.loads((folder / 'independent_audit.json').read_text())['failure_counts'])
vz = [r for r in rows if r['method_id'] == 'vz_pruned_e2e' and r['turn_id'] == 1][0]
print('vz T1 breakdown', vz['result']['ttft_breakdown_ms'], vz['result']['pruned_input_len'])
# G4-b path equivalence
dialog = ds[0]; factory = lambda: E.prompt_factory(dialog, 1, 'vz_pruned_e2e', {})
real, diag = pruned_t1_request(exp.runner, exp.servers['vz'], ROOT / dialog['image_path'], factory, E.M.QA._combined_processor)
enc_dev = exp.runner.to_device(diag['enc_cpu']); embeds = AbortAfterEmbeds(exp.runner).run(exp.runner, enc_dev); box = []
text_index = list(range(diag['v_token_start'] + diag['v_token_num'], int(diag['enc_cpu']['input_ids'].shape[1])))
composed = pruned_generate(exp.servers['vz'], embeds, diag['plan'], text_index, box); composed['first_logits'] = box[0]
pe = path_equivalence(real, composed); pe['prefix_embeds_equal'] = E.embeds_sha256(embeds[0, :diag['v_token_start'] + diag['v_token_num']]) == diag['prefix_embeds_sha256']
print('PATH EQUIVALENCE', pe, 'real tokens', real['generated_token_ids'], 'committed', vz['result']['generated_token_ids'])
# B main reproduction of the two dialogues
bcp = E.B_MAIN / 'llava/main/images' / image / 'COMMITTED.json'; bdoc = json.loads(bcp.read_text())
b = {(r['dialog_id'], r['turn_id'], r['method_id']): r for r in E.load_jsonl(bcp.parent / bdoc['attempt_id'] / 'raw.jsonl')}
for r in rows:
    if r['method_id'] in ('recompute', 'ours_prefixstore_kv25'):
        x = b[(r['dialog_id'], r['turn_id'], r['method_id'])]
        print('B-repro', r['method_id'], r['turn_id'], r['prediction'] == x['prediction'], r['result']['generated_token_ids'] == x['result']['generated_token_ids'], r['prompt_sha256'] == x['prompt_sha256'])
res, main_rows, _, _ = AUD.run_audit(run, SCR / 'audit_out'); print('run_audit', res['status'], res['failure_counts'])
print('DEV CHECK DONE')

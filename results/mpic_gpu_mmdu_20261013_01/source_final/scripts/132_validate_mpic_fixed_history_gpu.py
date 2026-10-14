#!/usr/bin/env python3
"""Bounded four-comparator GPU diagnostic on one frozen dialogue/history.

Uses the already captured image stores, never creates or overwrites experiments.
The fixed history is the archived prefix-caching prediction for all comparators.
This diagnostic is excluded from pilot TTFT/quality summaries.
"""
import argparse, hashlib, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import torch
from types import SimpleNamespace
from mmimpress.mpic import MPICContext
from mmimpress.model import LlavaRunner
from mmimpress.mmdu_mpic import prompt_from_generated
from importlib import import_module
pilot = import_module('125_eval_mpic_mmdu_smoke')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available(), 'host GPU execution is required'
    manifest = json.loads((a.run / 'subset_manifest.json').read_text())
    dialog = manifest['dialogues'][0]
    rows = [json.loads(s) for s in (a.run / 'answers.jsonl').read_text().splitlines()]
    history_source = {r['turn']: r['prediction'] for r in rows
                      if r['dialog_id'] == dialog['dialog_id'] and r['method'] == 'prefix_caching'}
    assert set(history_source) == {1, 2, 3}
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    runner = LlavaRunner(model_id=str(pilot.SNAPSHOT)).load()
    runner.model_id = 'llava-hf/llava-v1.6-vicuna-7b-hf'
    contexts, handles, observed = [], [], []
    try:
        for i in range(2):
            contexts.append(MPICContext(a.run / f'dialog_{dialog["dialog_id"]}' / f'image_{i}',
                                       runner.model.device, drop_cache=False, runner=runner))
        resident = [SimpleNamespace(meta=c.meta, dir=c.dir,
                    reader=pilot.GPUResidentReader(c, runner.model.device, runner.model.dtype)) for c in contexts]
        dump(a.out / 'setup_io.json', [c.reader.setup_io for c in resident])
        for li, layer in enumerate(runner.layers):
            for kind in ('q', 'k', 'v'):
                def projection(module, args, output, li=li, kind=kind):
                    observed.append(dict(layer=li, operation=kind + '_proj', input_shape=list(args[0].shape),
                                         output_shape=list(output.shape)))
                handles.append(getattr(layer.self_attn, kind + '_proj').register_forward_hook(projection))
            def attention(module, args, kwargs, output, li=li):
                cache = kwargs.get('past_key_values')
                if cache is not None:
                    observed.append(dict(layer=li, operation='native_attention',
                        K_shape=list(cache.layers[li].keys.shape), V_shape=list(cache.layers[li].values.shape)))
            handles.append(layer.self_attn.register_forward_hook(attention, with_kwargs=True))
        results, previous = [], None
        for turn in (1, 2, 3):
            fixed = {i: history_source[i] for i in range(1, turn)}
            input_ids = None
            for method in pilot.METHODS:
                observed.clear()
                result, state = pilot.run_one(runner, dialog, turn, method, fixed, resident,
                    previous if method == 'prefix_caching' else None, 3, 'gpu_resident')
                if method == 'prefix_caching':
                    previous = state
                if input_ids is None:
                    input_ids = result['input_ids']
                assert result['input_ids'] == input_ids
                assert result['prompt'] == prompt_from_generated(dialog, turn, fixed)
                assert result['history'] == fixed and result['fallback'] is False
                assert result['ssd_bytes'] == result['preads'] == 0
                stats = result['stats']
                assert stats['decoder_prefill_pass_count'] == (2 if method == 'full_reuse' else 1)
                if method.startswith('mpic_'):
                    ntext = result['input_token_count'] - result['image_token_count']
                    selected = 32 if method == 'mpic_global32' else 32 * result['images']
                    assert result['recomputed_image_tokens'] == selected
                    assert set(stats['active_positions']) == set(stats['text_positions']) | set(stats['selected_image_positions'])
                    for li in range(32):
                        qproj = next(x for x in observed if x['layer'] == li and x['operation'] == 'q_proj')
                        assert qproj['input_shape'] == [1, ntext + selected, 4096]
                        assert qproj['output_shape'] == [1, ntext + selected, 4096]
                        layer = stats['layers'][li]
                        assert layer['future_probability_max'] == 0
                        assert layer['active_K_installed'] and layer['active_V_installed']
                        assert all(layer['reused_unchanged_per_image'])
                record = {**result, 'turn': turn, 'method': method, 'diagnostic_only': True,
                          'native_observations': list(observed)}
                results.append(record)
                with (a.out / 'records.jsonl').open('a') as f:
                    f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                print(json.dumps(dict(turn=turn, method=method, status='PASS',
                    input_tokens=len(input_ids), selected=result['recomputed_image_tokens'])), flush=True)
        dump(a.out / 'validation.json', dict(status='PASS', dialogue_id=dialog['dialog_id'], turns=3,
            method_requests=12, fixed_inputs_identical_per_turn=True, gold_history_used=False,
            history_source='archived prefix-caching prediction, shared by all four in diagnostics only',
            history_answers_sha256=sha(a.run / 'answers.jsonl'),
            native_projection_hooks=True, included_in_measured_pilot=False,
            model=runner.model_id, revision=pilot.REV,
            source_sha256=sha(__file__)))
    finally:
        for h in handles:
            h.remove()
        for c in contexts:
            c.close()

if __name__ == '__main__':
    main()

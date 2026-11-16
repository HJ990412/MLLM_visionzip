import importlib.util, sys, unittest

import numpy as np

from mmimpress.mtvqa25k.common import ROOT
from mmimpress.mtvqa25k_packed import analysis as AN, reference as REF


def bootstrap():
    spec = importlib.util.spec_from_file_location('_mtvqa_analysis_142_unittest', ROOT / 'scripts/142_analyze_mtvqa25k.py')
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    return mod.bootstrap


def row(score, tokens=(5, 2), pred='a', at='other', ttft=100.0):
    return {'score': score, 'generated_token_ids': list(tokens), 'prediction': pred, 'answer_type': at, 'prompt_sha256': 'p', 'suffix_ids_sha256': 's',
            'ttft_ms': ttft, 'request_e2e_ms': ttft + 10, 'ssd_read_bytes': 1000, 'ssd_preads': 1, 'packed_read_ms': 1.0, 'chunk_io_ms': 1.0, 'scatter_ms': 1.0,
            'prefill_ms': 1.0, 'prepare_ms': 2.0, 'decode_ms': 1.0, 'preadv_calls': 1, 'pread_calls': 0}


def world(n=60, lift=0.0, seed=0):
    rng = np.random.RandomState(seed)
    new, ref, images = {}, {}, []
    for i in range(n):
        iid = str(i); images.append(iid)
        base = {t: float(rng.rand() < .5) for t in (1, 2, 3)}
        ref[iid] = {(m, t): row(base[t], at=('yes/no', 'number', 'other')[(i + t) % 3]) for m in REF.V3_METHODS for t in (1, 2, 3)}
        new[iid] = {t: row(min(1.0, base[t] + (lift if t > 1 else 0.0)), at=('yes/no', 'number', 'other')[(i + t) % 3]) for t in (1, 2, 3)}
    return new, ref, images


class RuleTests(unittest.TestCase):
    def test_verdict_uses_only_the_confidence_interval(self):
        self.assertEqual(AN.verdict(0.01, 2.0), 'IMPROVED')
        self.assertEqual(AN.verdict(-0.01, 2.0), 'NOT_SHOWN')
        self.assertEqual(AN.verdict(0.0, 2.0), 'NOT_SHOWN')
        self.assertEqual(AN.verdict(-2.0, -0.01), 'WORSE')
        self.assertEqual(AN.verdict(-2.0, 0.0), 'NOT_SHOWN')

    def test_identical_arms_give_exact_zero_and_a_lift_is_recovered(self):
        new, ref, images = world()
        paired = AN.paired_scores(new, ref, images, bootstrap(), resamples=200)
        d = AN.find(paired, REF.V3_B, 'Followup')
        self.assertEqual((d['delta_pp'], d['ci95_low'], d['ci95_high']), (0.0, 0.0, 0.0))
        self.assertFalse(d['ci_excludes_zero'])
        new, ref, images = world(lift=1.0)
        d = AN.find(AN.paired_scores(new, ref, images, bootstrap(), resamples=200), REF.V3_B, 'Followup')
        self.assertGreater(d['ci95_low'], 0)
        self.assertAlmostEqual(d['delta_pp'], d['new_mean'] - d['baseline_mean'])
        self.assertEqual(AN.find(AN.paired_scores(new, ref, images, bootstrap(), resamples=200), REF.V3_B, 'T1')['delta_pp'], 0.0)

    def test_answer_type_rows_partition_the_followup_requests(self):
        new, ref, images = world(lift=1.0)
        paired = AN.paired_scores(new, ref, images, bootstrap(), resamples=100)
        parts = [AN.find(paired, REF.V3_B, 'Followup', at) for at in AN.ANSWER_TYPES]
        self.assertEqual(sum(p['requests'] for p in parts), 2 * len(images))
        total = AN.find(paired, REF.V3_B, 'Followup')
        self.assertAlmostEqual(sum(p['delta_pp'] * p['requests'] for p in parts) / (2 * len(images)), total['delta_pp'])
        self.assertTrue(all(p['exploratory_no_multiplicity_correction'] for p in parts))

    def test_g_det_t1_blocks_the_comparison_on_any_token_mismatch(self):
        new, ref, images = world()
        self.assertEqual(AN.g_det_t1(new, ref, images)['status'], 'PASS')
        new['3'][1] = row(new['3'][1]['score'], tokens=(6, 2))
        g = AN.g_det_t1(new, ref, images)
        self.assertEqual((g['status'], g['T1_token_exact'], g['mismatch_count'], g['cross_run_accuracy_comparison']), ('FAIL', len(images) - 1, 1, 'NOT_COMPARABLE'))
        paired = AN.paired_scores(new, ref, images, bootstrap(), resamples=100)
        dec = AN.decision(g, paired, AN.latency_io(new, ref, images))
        self.assertEqual(dec['A_layer_allocation_effect']['verdict'], 'PENDING_REVIEW')
        self.assertEqual(dec['cross_run_accuracy_comparison'], 'NOT_COMPARABLE')

    def test_decision_reference_values(self):
        new, ref, images = world(lift=1.0)
        for i in images:                      # ReComp (v3) is perfect on follow-ups, so ReComp - [B] > 0
            for t in (2, 3):
                ref[i][('recompute', t)] = row(1.0)
        paired = AN.paired_scores(new, ref, images, bootstrap(), resamples=200)
        lat = AN.latency_io(new, ref, images)
        dec = AN.decision(AN.g_det_t1(new, ref, images), paired, lat)
        A = dec['A_layer_allocation_effect']
        self.assertEqual(A['verdict'], 'IMPROVED')
        loss = AN.find(paired, 'recompute', 'Followup')['baseline_mean'] - AN.find(paired, REF.V3_B, 'Followup')['baseline_mean']
        self.assertAlmostEqual(A['reference_not_verdict']['loss_recovery_percent'], 100 * A['D_pp'] / loss)
        self.assertTrue(A['reference_not_verdict']['D_at_least_1.0pp_MTGQA_GO_threshold'])
        self.assertTrue(A['condition_SSD_MB_per_followup_le_B_x_1.02']['within_B_times_1.02'])
        self.assertAlmostEqual(lat['R_in_followup_over_T1_same_run'], 1.0)

    def test_undefined_interval_of_a_tiny_answer_type_sample_is_reported_not_nan(self):
        new, ref, images = world(n=3)
        for i in images:
            for t in (1, 2, 3):
                at = 'number' if (i, t) == ('0', 2) else 'other'
                new[i][t] = row(new[i][t]['score'], at=at)
        paired = AN.paired_scores(new, ref, images, bootstrap(), resamples=200)
        p = AN.find(paired, REF.V3_B, 'Followup', 'number')
        self.assertEqual((p['requests'], p['ci_defined'], p['ci95_low'], p['ci95_high'], p['ci_excludes_zero']), (1, False, None, None, False))
        self.assertTrue(AN.find(paired, REF.V3_B, 'Followup')['ci_defined'])

    def test_agreement_counts(self):
        new, ref, images = world(n=4)
        for i in images:
            for t in (2, 3):
                new[i][t] = row(1.0, pred='x'); ref[i][(REF.V3_B, t)] = row(0.0, pred='y')
        a = AN.agreement(new, ref, images)['Followup']
        self.assertEqual((a['requests'], a['prediction_string_equal'], a['only_PYR_correct'], a['only_B_correct']), (8, 0, 8, 0))


if __name__ == '__main__':
    unittest.main()

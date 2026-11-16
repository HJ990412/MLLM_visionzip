import copy, json, tempfile, unittest
from pathlib import Path

from mmimpress.mtvqa25k.common import ROOT, dump, sha
from mmimpress.mtvqa25k_packed import adapter as AD, checks as CK, reference as REF


def annotation(qid, answers):
    return {'question_id': qid, 'image_id': 1, 'answer_type': 'other', 'question_type': 'what',
            'answers': [{'answer_id': i + 1, 'answer': a, 'answer_confidence': 'yes'} for i, a in enumerate(answers)]}


class ArmTests(unittest.TestCase):
    def test_arms_are_identified_by_name_and_need_no_recompute(self):
        self.assertEqual(AD.check_methods([AD.MAIN_METHOD]), [AD.MAIN_METHOD])
        self.assertEqual(AD.vector_of(AD.MAIN_METHOD), 'PYR-beta-raw')
        self.assertEqual(AD.vector_of(AD.GATE_METHOD), 'U')
        for bad in (['recompute'], ['ours_prefixstore_kv25'], [], [AD.MAIN_METHOD, AD.MAIN_METHOD], ['fullload', AD.MAIN_METHOD]):
            with self.assertRaises(ValueError):
                AD.check_methods(bad)

    def test_budget_entry_carries_the_document_clip_range(self):
        doc = json.loads((ROOT / CK.VECTOR_SOURCE / 'layer_budget_llava_PYR-beta-raw.json').read_text())
        entry = AD.budget_entry(doc, 'x.json')
        self.assertEqual((entry['r_min'], entry['r_max'], entry['layer_budget_sha256']), (0.0, 0.6, CK.RATIO_SHA['PYR-beta-raw']))


class BudgetTests(unittest.TestCase):
    def setUp(self):
        from mmimpress.layer_budget import load_budget_document
        self.raw = load_budget_document(ROOT / CK.VECTOR_SOURCE / 'layer_budget_llava_PYR-beta-raw.json')
        self.uni = load_budget_document(ROOT / CK.VECTOR_SOURCE / 'layer_budget_llava_U.json')

    def test_precomputed_k_tables_and_independent_hamilton(self):
        for n, want in CK.EXPECTED_K.items():
            k = CK.doc_counts(self.raw, n)
            self.assertEqual(k, CK.hamilton(self.raw['layer_keep_ratios'], n, self.raw['r_max']))
            self.assertEqual(sum(k), 32 * ((n + 3) // 4))
            self.assertGreaterEqual(min(k), 1)
            self.assertEqual(k[-3:], want['tail'])
            if 'head' in want:
                self.assertEqual(k[:3], want['head'])
        self.assertEqual(min(CK.doc_counts(self.raw, 960)), 12)

    def test_raw_vector_is_rejected_by_the_module_default_floor(self):
        from mmimpress.layer_budget import layer_keep_counts
        with self.assertRaises(ValueError):
            layer_keep_counts(self.raw['layer_keep_ratios'], 2304)
        with self.assertRaises(ValueError):
            CK.doc_counts(dict(self.raw, r_min=0.10), 2304)

    def test_packed_bytes_equal_uniform_total(self):
        for n, sep in ((960, 8), (2304, 36), (2880, 48)):
            pyr = CK.packed_bytes(CK.doc_counts(self.raw, n), sep)
            uni = CK.packed_bytes(CK.doc_counts(self.uni, n), sep)
            self.assertEqual(pyr, uni)
            self.assertEqual(uni, 32 * 2 * ((n + 3) // 4) * 8192 + 2 * 32 * sep * 8192)
        self.assertEqual(CK.packed_bytes(CK.doc_counts(self.raw, 2304), 36), 320864256)      # v3 [B] image 1000 follow-up bytes


class AuditTests(unittest.TestCase):
    def row(self, score=.6):
        ann = annotation(7, ['cat'] * 2 + ['dog'] * 8)
        return {'question_id': '7', 'annotation': ann, 'prediction': 'cat', 'scorer_answer': 'cat', 'score': score, 'gold': ['cat'] * 2 + ['dog'] * 8,
                'generated_token_ids': [1, 2], 'result': {'generated_token_ids': [1, 2]}, 'dataset': AD.DATASET}, ann

    def test_official_score_is_separate_from_legacy_structural_audit(self):
        row, ann = self.row()
        seen = {}

        def old(rows, manifest, config, phase, image_id, artifact=None):
            seen['score'] = rows[0]['score']
            return {'status': 'PASS', 'failures': []}
        legacy_em = lambda pred, gold: float(pred == gold)
        audit = AD.official_audit(old, legacy_em)
        manifest = {'dialogues': [{'turns': [{'question_id': '7', 'annotation': ann}]}]}
        self.assertEqual(audit([row], manifest, {}, 'main', '1')['status'], 'PASS')
        self.assertEqual(seen['score'], 1.0)          # the 164 audit saw strict EM on a copy
        self.assertEqual(row['score'], .6)            # the adopted official score is untouched
        for mutate in (lambda r: r.update(score=1.0), lambda r: r['annotation']['answers'][2].update(answer='cat'), lambda r: r.update(scorer_answer='dog'),
                       lambda r: r.update(generated_token_ids=[1]), lambda r: r.update(dataset='MT-GQA-reconstructed'), lambda r: r.update(gold=['cat'] * 10)):
            bad = copy.deepcopy(row); mutate(bad)
            self.assertEqual(audit([bad], manifest, {}, 'main', '1')['status'], 'FAIL')

    def test_structural_failure_is_not_masked(self):
        row, ann = self.row()
        audit = AD.official_audit(lambda *a, **k: {'status': 'FAIL', 'failures': [{'check': 'x'}]}, lambda p, g: 0.0)
        self.assertEqual(audit([row], {'dialogues': [{'turns': [{'question_id': '7', 'annotation': ann}]}]}, {}, 'main', '1')['status'], 'FAIL')


class CleanupTests(unittest.TestCase):
    def test_cleanup_rejects_foreign_roots_and_requires_audit_commit_and_exact_hashes(self):
        ok_audit, ok_commit = {'status': 'PASS'}, {'status': 'COMMITTED'}
        with tempfile.TemporaryDirectory() as td:
            scratch = Path(td) / 'llava/main/scratch/main/1/attempt_0001'; scratch.mkdir(parents=True)
            (scratch / 'kv_packed.bin').write_bytes(b'x')
            allow = {'kv_packed.bin': sha(scratch / 'kv_packed.bin')}
            with self.assertRaises(ValueError):      # not under runs/llava_next_mtvqa_pyr_*
                AD.cleanup_owned(scratch, allow, ok_audit, ok_commit, Path(td))
            with self.assertRaises(ValueError):      # an existing run root is never accepted
                AD.cleanup_owned(scratch, allow, ok_audit, ok_commit, ROOT / 'runs/mt_vqa_v2_reconstructed_25k_v3_20261021T100500Z')
            self.assertTrue((scratch / 'kv_packed.bin').exists())
        experiment = ROOT / 'runs' / (AD.RUN_PREFIX + 'unittest_fixture')
        if experiment.exists():
            self.skipTest('fixture root already present')
        try:
            scratch = experiment / 'llava/main/scratch/main/1/attempt_0001'; scratch.mkdir(parents=True)
            (scratch / 'kv_packed.bin').write_bytes(b'payload'); (scratch / 'meta.json').write_text('{}')
            allow = {n: sha(scratch / n) for n in ('kv_packed.bin', 'meta.json')}
            for audit, commit, listing in (({'status': 'FAIL'}, ok_commit, allow), (ok_audit, {'status': 'PARTIAL'}, allow),
                                           (ok_audit, ok_commit, {'kv_packed.bin': allow['kv_packed.bin']}), (ok_audit, ok_commit, dict(allow, **{'meta.json': '0' * 64}))):
                with self.assertRaises(ValueError):
                    AD.cleanup_owned(scratch, listing, audit, commit, experiment)
            with self.assertRaises(ValueError):      # wrong depth
                AD.cleanup_owned(scratch.parent, allow, ok_audit, ok_commit, experiment)
            self.assertTrue((scratch / 'kv_packed.bin').exists())
            AD.cleanup_owned(scratch, allow, ok_audit, ok_commit, experiment)
            self.assertFalse(scratch.exists())
        finally:
            import shutil
            if experiment.exists() and experiment.name.endswith('unittest_fixture'):
                shutil.rmtree(experiment)


class ReferenceTests(unittest.TestCase):
    def test_committed_raw_checks_the_commit_sha_and_compares_requests(self):
        import gzip, hashlib
        row = {'method_id': 'm', 'turn_id': 2, 'dialog_id': 'd', 'question_id': 5, 'prediction': 'cat', 'generated_token_ids': [3, 2], 'score': 1.0, 'N_content': 960, 'k': 240,
               'ssd_read_bytes': 10, 'ssd_preads': 1, 'ttft_ms': 1.0, 'request_e2e_ms': 2.0, 'prompt_sha256': 'p', 'suffix_ids_sha256': 's', 'history_sha256': 'h',
               'annotation': {'answer_type': 'other'}, 'geometry': {'newline_idx': [1, 2]}, 'result': {'scatter_ms': 1.0}, 'prompt': 'P', 'suffix_input_ids': [1]}
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / '1'; attempt = folder / 'attempt_0001'; attempt.mkdir(parents=True)
            data = (json.dumps(row) + '\n').encode()
            with gzip.open(attempt / 'raw.jsonl.gz', 'wb') as f:
                f.write(data)
            commit = {'attempt_id': 'attempt_0001', 'requests': 1, 'raw_sha256': hashlib.sha256(data).hexdigest(), 'journal_archives': {'raw.jsonl': {'path': 'raw.jsonl.gz'}}}
            dump(folder / 'COMMITTED.json', commit)
            rows, _ = REF.committed_raw(folder)
            c = REF.compact(rows[0], full=True)
            self.assertEqual((c['question_id'], c['n_newline'], c['answer_type'], c['suffix_input_ids']), ('5', 2, 'other', [1]))
            self.assertEqual(REF.compare_rows(c, dict(c)), [])
            self.assertEqual(REF.compare_rows(c, dict(c, prediction='dog', generated_token_ids=[4, 2])), ['generated_token_ids', 'prediction'])
            self.assertEqual(REF.extract((str(folder), False))['status'], 'OK')
            dump(folder / 'COMMITTED.json', dict(commit, raw_sha256='0' * 64), replace=True)
            self.assertEqual(REF.extract((str(folder), False))['status'], 'ERROR')


class BindTests(unittest.TestCase):
    def test_bind_overlays_identities_without_touching_the_base_script(self):
        before = sha(ROOT / 'scripts' / AD.BASE_SCRIPT)
        M = AD.bind_packed(AD.RUN_PREFIX + 'unit', '_mtvqa25k_packed_llava_unittest')
        self.assertEqual(sha(ROOT / 'scripts' / AD.BASE_SCRIPT), before)
        self.assertEqual(M.SCHEMA, AD.SCHEMA)
        self.assertEqual(M.request_id('ignored', 'main', AD.MAIN_METHOD, 'mtvqa25k_000001', 2),
                         f'{AD.SCHEMA}:{AD.RUN_PREFIX}unit:main:{M.MODEL_ID}:{AD.MAIN_METHOD}:mtvqa25k_000001:t2')
        self.assertEqual(M.stored_methods([AD.MAIN_METHOD]), [AD.MAIN_METHOD])
        self.assertEqual(M.kind_of(AD.MAIN_METHOD), 'packed_PYR-beta-raw')
        self.assertEqual(M.method_order([AD.MAIN_METHOD], 12345), [AD.MAIN_METHOD])
        prompt, history, entries = M.prompt_factory([AD.MAIN_METHOD], {'dialog_id': 'd', 'turns': [{'question': ' Is it red? ', 'question_id': 1}, {'question': 'How many?', 'question_id': 2}]},
                                                    2, AD.MAIN_METHOD, {1: {'method_id': AD.MAIN_METHOD, 'dialog_id': 'd', 'prediction': 'Yes', 'logical_request_id': 'l', 'physical_execution_id': 'p'}})
        self.assertEqual(prompt, 'USER: <image>\nQ1: Is it red?\nA1: Yes\n\nCurrent question Q2: How many?\nAnswer the current question with a single word or short phrase. ASSISTANT:')
        with self.assertRaises(ValueError):
            M._native_persist(None, {}, {}, None, '1', 'raster', {}, {})


if __name__ == '__main__':
    unittest.main()

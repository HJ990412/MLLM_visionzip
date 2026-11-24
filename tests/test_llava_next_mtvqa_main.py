"""CPU contracts of the LLaVA-NeXT x MT-VQA main experiment harness
(docs/llava_next_mtvqa_main_contract.md; scripts 209 / 210 / 211, scripts/llava_next_main_status.py).

No model is loaded and nothing is read from an existing run directory except
small frozen JSON files.  What is pinned:

* the arm table: the three main arms and their method code (the calls of
  scripts/196 for FullLoad / IMPRESS, of scripts/202 for Ours with
  ``row_order="original"``), the gate-only arms, the phase sizes;
* the lifecycle helpers the long run depends on: store signatures, the
  cleanup that deletes exactly one image's stores, the resume bookkeeping;
* the pre-run checks: store-size formulas against recorded store sizes, the
  free-space rule, the TTFT-interval comparison with the v3 source copy;
* the validator's comparisons and the report's statistics on synthetic rows;
* status.json and the SSD counter.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(name, filename):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _function_source(filename, name):
    src = (ROOT / "scripts" / filename).read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise KeyError(name)


def _calls(source, attr):
    """The ``ast.Call`` nodes of ``source`` whose function is named ``attr`` (name or attribute)."""
    out = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Attribute) and f.attr == attr) or (isinstance(f, ast.Name) and f.id == attr):
                out.append(node)
    return out


def _keywords(call):
    return {k.arg: ast.dump(k.value) for k in call.keywords if k.arg}


class ArmTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.H = _load("_lnext_mtvqa_main_eval_test", "209_eval_llava_next_mtvqa_main.py")

    def test_main_arms_are_fullload_impress_ours_all_asynchronous(self):
        H = self.H
        self.assertEqual(H.MAIN_METHODS, [H.FULL_ASYNC, H.NEW_ASYNC, H.ORIG_ASYNC])
        self.assertEqual(H.PHASE_METHODS["main"], H.MAIN_METHODS)
        self.assertEqual([H.ARMS[m]["display"] for m in H.MAIN_METHODS], ["FullLoad", "IMPRESS", "Ours"])
        self.assertTrue(all(H.ARMS[m]["async"] for m in H.MAIN_METHODS))
        self.assertEqual([H.ARMS[m]["kind"] for m in H.MAIN_METHODS], ["raster", "impress_official", "packedorig_PYR-beta-raw"])
        self.assertNotIn(H.RECOMP, H.PHASE_METHODS["main"])

    def test_pilot_adds_recomp_and_gate_has_the_gate_only_arms(self):
        H = self.H
        self.assertEqual(H.PHASE_METHODS["pilot"], H.MAIN_METHODS + [H.RECOMP])
        gate = H.PHASE_METHODS["gate"]
        for m in (H.NEW, H.NEW_ASYNC, H.NEW_R100, H.FULL_REORDERED, H.ORIG, H.ORIG_ASYNC, H.ORIG_REPEAT):
            self.assertIn(m, gate)
        # the four IMPRESS-side gate arms read the same reordered store; the three Ours arms the same packed store
        self.assertEqual({H.ARMS[m]["kind"] for m in (H.NEW, H.NEW_ASYNC, H.NEW_R100, H.FULL_REORDERED)}, {H.NEW_KIND})
        self.assertEqual({H.ARMS[m]["kind"] for m in (H.ORIG, H.ORIG_ASYNC, H.ORIG_REPEAT)}, {H.ORIG_KIND})
        self.assertEqual(H.ARMS[H.NEW_R100]["ratio"], 1.0)
        self.assertEqual(H.ARMS[H.FULL_REORDERED]["family"], "fullload")
        self.assertFalse(H.ARMS[H.ORIG]["async"]); self.assertFalse(H.ARMS[H.NEW]["async"])

    def test_request_counts(self):
        H = self.H
        self.assertEqual(H.EXPECTED_REQUESTS, {"gate": 108, "pilot": 480, "main": 225000})
        self.assertEqual(H.EXPECTED_REQUESTS["main"], 25000 * 3 * len(H.MAIN_METHODS))

    def test_method_ids_and_settings_are_those_of_the_reference_runs(self):
        H = self.H
        impl = json.loads((H.IMPL_REF_RUN / "config.json").read_text())
        ours = json.loads((H.OURS_REF_RUN / "config.json").read_text())
        strip = lambda a: {k: v for k, v in a.items() if k != "display"}      # noqa: E731
        for m in (H.FULL_ASYNC, H.NEW, H.NEW_ASYNC, H.NEW_R100, H.FULL_REORDERED):
            self.assertEqual(strip(impl["arms"][m]), strip(H.ARMS[m]), m)
        for m in (H.ORIG, H.ORIG_ASYNC, H.ORIG_REPEAT):
            self.assertEqual(strip(ours["arms"][m]), strip(H.ARMS[m]), m)
        self.assertEqual(impl["server_args"], H.SERVER_ARGS)
        self.assertEqual(impl["impress_official"], H.OFFICIAL)
        self.assertEqual(impl["readahead"][H.NEW_ASYNC], H.READAHEAD[H.NEW_ASYNC])
        self.assertEqual(impl["readahead"][H.FULL_ASYNC], H.READAHEAD[H.FULL_ASYNC])
        self.assertEqual(ours["readahead"][H.ORIG_ASYNC], H.READAHEAD[H.ORIG_ASYNC])
        self.assertEqual(ours["layer_budgets"][H.VECTOR]["layer_budget_sha256"], H.VECTOR_RATIO_SHA)
        self.assertEqual((impl["model"], impl["revision"]), (H.MODEL_ID, H.REVISION))

    def test_method_source_files_equal_the_reference_freezes(self):
        """FullLoad / IMPRESS == the 2026-11-17 freeze and Ours == the 2026-11-20 freeze, except the listed files."""
        H = self.H
        now = H.source_hashes()
        for ref, allowed in ((H.IMPL_REF_RUN, H.CHANGED_SINCE_IMPL_REF), (H.OURS_REF_RUN, H.CHANGED_SINCE_OURS_REF)):
            before = {k: v for k, v in json.loads((ref / "config.json").read_text())["source_sha256"].items() if not H.REPORT_SCRIPT.match(k)}
            changed = {k for k in before if now.get(k) != before[k]}
            self.assertLessEqual(changed, set(allowed), ref.name)
        self.assertNotIn("scripts/211_report_llava_next_mtvqa_main.py", now)      # report scripts are not frozen
        self.assertIn("scripts/209_eval_llava_next_mtvqa_main.py", now)
        self.assertIn("scripts/llava_next_main_status.py", now)

    def test_request_calls_are_the_calls_of_the_reference_harnesses(self):
        """Same callee and same keyword arguments as scripts/196 (FullLoad, IMPRESS) and scripts/202 (Ours)."""
        mine = _function_source("209_eval_llava_next_mtvqa_main.py", "stored_hit")
        ref196 = _function_source("196_eval_mt_gqa_impress_code.py", "stored_hit")
        ref202 = _function_source("202_eval_mt_gqa_ours_order.py", "stored_hit")
        self.assertEqual(_keywords(_calls(mine, "request_official")[0]), _keywords(_calls(ref196, "request_official")[0]))
        self.assertEqual(_keywords(_calls(mine, "request_packed")[0]), _keywords(_calls(ref202, "request_packed")[0]))
        full_mine = [c for c in _calls(mine, "request") if "mode" in _keywords(c)][0]
        full_ref = [c for c in _calls(ref196, "request") if "mode" in _keywords(c)][0]
        self.assertEqual(_keywords(full_mine), _keywords(full_ref))

    def test_writer_calls_are_the_calls_of_the_reference_harnesses(self):
        mine = _function_source("209_eval_llava_next_mtvqa_main.py", "_native_persist")
        ref196 = _function_source("196_eval_mt_gqa_impress_code.py", "_native_persist")
        ref202 = _function_source("202_eval_mt_gqa_ours_order.py", "_native_persist")
        a = _keywords(_calls(mine, "persist_captured_raster_prefix")[0]); b = _keywords(_calls(ref196, "persist_captured_raster_prefix")[0])
        self.assertEqual(a, b)
        packed = _keywords(_calls(mine, "persist_captured_packed_prefix")[0])
        ref = _keywords(_calls(ref202, "persist_captured_packed_prefix")[0])
        self.assertEqual({k: v for k, v in packed.items() if k != "row_order"}, ref)      # scripts/202 passes the row order through **order
        self.assertIn("ROW_ORDER_ORIGINAL", packed["row_order"])
        self.assertIn('"dataset": DATASET', mine)                                           # the only difference: the dataset identity
        self.assertEqual(self.H.DATASET, "MT-VQA-v2-reconstructed-25k")

    def test_rotation_is_cyclic_over_the_phase_index(self):
        H = self.H
        m = H.MAIN_METHODS
        self.assertEqual([H.method_order(m, i)[0] for i in range(6)], [m[0], m[1], m[2], m[0], m[1], m[2]])
        self.assertEqual(sorted(H.method_order(m, 7)), sorted(m))

    def test_store_kinds_follow_the_arm_list(self):
        H = self.H
        self.assertEqual(H.store_kinds(H.PHASE_METHODS["main"]), ["raster", H.NEW_KIND, H.ORIG_KIND])
        self.assertEqual(H.store_kinds(H.PHASE_METHODS["pilot"]), ["raster", H.NEW_KIND, H.ORIG_KIND])
        self.assertEqual(H.stored_methods([H.RECOMP, H.FULL_ASYNC]), [H.FULL_ASYNC])


class StoragePlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.H = _load("_lnext_mtvqa_main_eval_test", "209_eval_llava_next_mtvqa_main.py")

    def test_store_size_formulas_match_recorded_store_sizes(self):
        """N = 2400 content tokens + 38 separators: the sizes the 2026-11-17 run recorded for such an image."""
        b = self.H.store_bytes(2400, 38)
        for key, recorded in (("raster", 1_380_690_820), ("impress_official", 1_382_302_955), ("packedorig_PYR-beta-raw", 337_229_751)):
            self.assertLess(abs(b[key] - recorded) / recorded, 0.002, key)
        recorded_written = 1_380_690_820 + 2_742_331_116 + 337_229_751
        self.assertLess(abs(b["written"] - recorded_written) / recorded_written, 0.005)
        self.assertGreater(b["bundle"], b["raster"] + b["impress_official"] + b["packedorig_PYR-beta-raw"])

    def test_largest_geometry_bundle_and_the_twenty_times_rule(self):
        H = self.H
        b = H.store_bytes(2880, 48)
        self.assertLess(b["bundle"], 4 * H.GiB)
        self.assertEqual(H.FREE_SPACE_FACTOR, 20)
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            plan = {"required_free_bytes": 20 * b["bundle"], "minimum_reserve_bytes": 30 * H.GiB, "largest_image_bundle_bytes": b["bundle"]}
            (run / "storage_plan.json").write_text(json.dumps(plan))
            usage = lambda free: mock.patch.object(H.shutil, "disk_usage", lambda p: mock.Mock(free=free))      # noqa: E731
            with usage(20 * b["bundle"] + 1):
                self.assertEqual(H.capacity_guard(run)["retained_scratch_bytes"], 0)
            with usage(20 * b["bundle"] - 1):
                with self.assertRaisesRegex(RuntimeError, "BLOCKED_STORAGE"):
                    H.capacity_guard(run)

    def test_ttft_interval_check_passes_on_the_real_sources_and_detects_a_moved_timer(self):
        H = self.H
        real = H.ttft_boundary_check()
        self.assertTrue(real["same_interval"], real["checks"])
        self.assertEqual(len(real["normal_request_differing_statements"]), 2)
        self.assertTrue(all(d["value_for_recompute_request"] == [False, False] for d in real["normal_request_differing_statements"]))
        # a copy of the current sources whose timer starts after the prompt is assembled must be reported as a different interval
        with tempfile.TemporaryDirectory() as td:
            fake = Path(td)
            (fake / "mmimpress").mkdir(); (fake / "scripts").mkdir()
            (fake / "mmimpress/serve.py").write_text((ROOT / "mmimpress/serve.py").read_text())
            src = (ROOT / "scripts/156_eval_layer_budget_hold.py").read_text()
            start = "    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()\n    prompt, history, entries = factory()\n"
            self.assertIn(start, src)
            moved = src.replace(start, "    prompt, history, entries = factory()\n    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()\n", 1)
            (fake / "scripts/156_eval_layer_budget_hold.py").write_text(moved)
            self.assertFalse(H.ttft_boundary_check(now_source=fake)["same_interval"])


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.H = _load("_lnext_mtvqa_main_eval_test", "209_eval_llava_next_mtvqa_main.py")

    def _store(self, root):
        scratch = root / "scratch" / "main" / "img1" / "attempt_0001"
        for kind, files in (("raster", ("meta.json", "layer_00/k.bin")), ("packedorig", ("kv_packed.bin",))):
            for f in files:
                p = scratch / kind / f
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(os.urandom(64))
        return scratch

    def test_signature_detects_a_rewritten_file(self):
        H = self.H
        with tempfile.TemporaryDirectory() as td:
            scratch = self._store(Path(td))
            before = H.store_signature(scratch / "raster")
            self.assertEqual(before, H.store_signature(scratch / "raster"))
            p = scratch / "raster" / "meta.json"
            st = p.stat()
            p.write_bytes(os.urandom(64)); os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
            self.assertNotEqual(before, H.store_signature(scratch / "raster"))

    def test_cleanup_removes_exactly_the_image_stores(self):
        H = self.H
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            scratch = self._store(run)
            signature = H.store_signature(scratch)
            ok = ({"status": "PASS"}, {"status": "COMMITTED"})
            with self.assertRaises(ValueError):                       # not committed
                H.cleanup_scratch(run, scratch, signature, {"status": "PASS"}, {"status": "PARTIAL"})
            extra = scratch / "raster" / "unexpected.bin"; extra.write_bytes(b"x")
            with self.assertRaises(ValueError):                       # a file that was not recorded
                H.cleanup_scratch(run, scratch, signature, *ok)
            extra.unlink()
            outside = run / "elsewhere"; outside.mkdir()
            with self.assertRaises(ValueError):                       # not under this run's scratch
                H.cleanup_scratch(run, outside, {}, *ok)
            receipt = H.cleanup_scratch(run, scratch, signature, *ok)
            self.assertEqual(receipt["removed_files"], 3)
            self.assertFalse((run / "scratch" / "main" / "img1").exists())
            self.assertTrue((run / "scratch" / "main").exists())

    def test_remove_scratch_refuses_paths_outside_the_run(self):
        H = self.H
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            scratch = self._store(run)
            with self.assertRaises(ValueError):
                H.remove_scratch(run, run / "scratch")
            other = run / "other"; other.mkdir()
            with self.assertRaises(ValueError):
                H.remove_scratch(run, other)
            self.assertEqual(H.remove_scratch(run, scratch.parent), 3 * 64)
            self.assertEqual(H.remove_scratch(run, scratch.parent), 0)

    def test_completed_images_and_failed_images(self):
        H = self.H
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            config = {"_file_sha256": "c", "_manifest_sha256": "m"}
            for image, doc in (("1", {"config_sha256": "c", "manifest_sha256": "m", "requests": 9}), ("2", None)):
                folder = run / "main" / "images" / image; folder.mkdir(parents=True)
                (folder / "attempt_0001").mkdir()
                if doc:
                    (folder / "COMMITTED.json").write_text(json.dumps(doc))
            self.assertEqual(list(H.completed_images(run, "main", config)), ["1"])      # image 2 was interrupted: not complete
            self.assertEqual(H.completed_images(run, "pilot", config), {})
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                H.completed_images(run, "main", {"_file_sha256": "other", "_manifest_sha256": "m"})
            self.assertEqual(H.failed_images(run, "main"), [])
            (run / "main_failed_images.jsonl").write_text(json.dumps({"image_id": "7"}) + "\n")
            self.assertEqual([f["image_id"] for f in H.failed_images(run, "main")], ["7"])

    def test_evidence_is_slimmed_only_in_the_main_phase(self):
        H = self.H
        with tempfile.TemporaryDirectory() as td:
            for phase in ("pilot", "main"):
                artifact = Path(td) / phase; artifact.mkdir()
                np.savez(artifact / "impress_official_turn1_stats.npz", a=np.zeros(4, dtype=np.float32))
                meta = {"order": [[1, 2, 3]] * 4}
                (artifact / "impress_official_meta.json").write_text(json.dumps(meta))
                out = H.slim_evidence(artifact, phase, {"status": "PASS"})
                if phase == "pilot":
                    self.assertTrue((artifact / "impress_official_turn1_stats.npz").exists())
                    self.assertTrue((artifact / "impress_official_meta.json").exists())
                else:
                    self.assertFalse((artifact / "impress_official_turn1_stats.npz").exists())
                    self.assertFalse((artifact / "impress_official_meta.json").exists())
                    import gzip
                    self.assertEqual(json.loads(gzip.decompress((artifact / "impress_official_meta.json.gz").read_bytes())), meta)
                    self.assertEqual(len(out["removed"]), 1); self.assertEqual(len(out["compressed"]), 1)
                    self.assertEqual(len(out["removed"][0]["sha256"]), 64)

    def test_freeze_violations_ignore_new_files(self):
        H = self.H
        now = H.source_hashes()
        self.assertEqual(H.freeze_violations(now), [])
        frozen = dict(now)
        some = next(iter(frozen))
        frozen[some] = "0" * 64
        frozen["scripts/removed_after_freeze.py"] = "1" * 64
        self.assertEqual(sorted(H.freeze_violations(frozen)), sorted([some, "scripts/removed_after_freeze.py"]))
        del frozen[some]; del frozen["scripts/removed_after_freeze.py"]
        frozen.pop("tests/test_llava_next_mtvqa_main.py")      # a file created after the freeze is not compared
        self.assertEqual(H.freeze_violations(frozen), [])


class ValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.G = _load("_lnext_mtvqa_main_validate_test", "210_validate_llava_next_mtvqa_main.py")

    def _row(self, tokens, bytes_=10, prompt="p"):
        return {"result": {"generated_token_ids": list(tokens)}, "prompt": prompt, "prediction": "x", "ssd_read_bytes": bytes_,
                "packed_read_bytes": bytes_, "prompt_sha256": "s", "ttft_ms": 1.0}

    def test_compare_requires_every_request_and_equal_tokens_and_bytes(self):
        G = self.G
        keys = [("1", t) for t in (1, 2, 3)]
        rows = {(a, "1", t): self._row([t, 2]) for a in ("a", "b") for t in (1, 2, 3)}
        self.assertEqual(G.compare(rows, "a", "b", keys, ("ssd_read_bytes",))["status"], "PASS")
        rows[("b", "1", 3)] = self._row([9, 2])
        out = G.compare(rows, "a", "b", keys)
        self.assertEqual((out["status"], out["tokens_differ"], out["tokens_differ_same_prompt"]), ("FAIL", 1, 1))
        rows[("b", "1", 3)] = self._row([3, 2], bytes_=11)
        out = G.compare(rows, "a", "b", keys, ("ssd_read_bytes",))
        self.assertEqual((out["status"], out["tokens_differ"], out["fields_differ"]["ssd_read_bytes"]), ("FAIL", 0, 1))
        del rows[("b", "1", 3)]
        self.assertEqual(G.compare(rows, "a", "b", keys)["missing"], 1)
        self.assertEqual(G.compare(rows, "a", "b", keys)["status"], "FAIL")

    def test_against_v3_same_turn_and_turn1_reference(self):
        G = self.G
        keys = [("1", t) for t in (1, 2, 3)]
        rows = {("recompute", "1", t): self._row([t]) for t in (1, 2, 3)}
        rows.update({("ours", "1", t): self._row([1] if t == 1 else [50 + t]) for t in (1, 2, 3)})
        ref = {"1": {("recompute", t): {"generated_token_ids": [t], "prediction": "x"} for t in (1, 2, 3)}}
        self.assertEqual(G.against_v3(rows, keys, "recompute", ref, "recompute")["status"], "PASS")
        t1 = G.against_v3(rows, keys, "ours", ref, "recompute", turns=(1,), reference_turn=1)
        self.assertEqual((t1["status"], t1["requests"]), ("PASS", 1))
        ref["1"][("recompute", 2)]["generated_token_ids"] = [7]
        bad = G.against_v3(rows, keys, "recompute", ref, "recompute")
        self.assertEqual((bad["status"], bad["mismatch_count"]), ("FAIL", 1))
        self.assertEqual(G.against_v3(rows, keys, "recompute", {}, "recompute")["status"], "FAIL")      # missing in the record

    def test_admission_needs_prerun_and_every_gate_for_this_freeze(self):
        G = self.G
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            config = {"_file_sha256": "c"}
            (run / "gates").mkdir()
            (run / "prerun_receipt.json").write_text(json.dumps({"status": "PASS", "config_sha256": "c"}))
            for name in G.REQUIRED:
                (run / "gates" / f"{name}.json").write_text(json.dumps({"status": "PASS", "config_sha256": "c"}))
            with mock.patch("builtins.print"):
                self.assertEqual(G.stage_admit(run, config)["status"], "PASS")
                (run / "gates" / "gate.json").write_text(json.dumps({"status": "FAIL", "config_sha256": "c", "failed": ["gate2_ours"]}))
                self.assertEqual(G.stage_admit(run, config)["status"], "FAIL")
                (run / "gates" / "gate.json").write_text(json.dumps({"status": "PASS", "config_sha256": "another freeze"}))
                self.assertEqual(G.stage_admit(run, config)["status"], "FAIL")
                (run / "gates" / "gate.json").write_text(json.dumps({"status": "PASS", "config_sha256": "c"}))
                (run / "prerun_receipt.json").write_text(json.dumps({"status": "BLOCKED", "config_sha256": "c", "blocking": ["free space"]}))
                self.assertEqual(G.stage_admit(run, config)["status"], "FAIL")
            self.assertEqual(G.REQUIRED, ("tests", "scorer", "gate", "pilot"))

    def test_baseline_failures_are_the_recorded_ones(self):
        G = self.G
        self.assertEqual({k: len(v) for k, v in G.BASELINE_FAILURES.items()}, {"4": 3, "5": 2})
        ref = (ROOT / "scripts/188_validate_mt_gqa_async.py").read_text()
        for names in G.BASELINE_FAILURES.values():
            for name in names:
                self.assertIn(name, ref)


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.R = _load("_lnext_mtvqa_main_report_test", "211_report_llava_next_mtvqa_main.py")

    def _records(self, n=40, seed=0):
        R = self.R
        rng = np.random.default_rng(seed)
        records = []
        base = {R.FULL: (0.80, 600.0), R.IMP: (0.78, 800.0), R.OURS: (0.76, 150.0), R.V3: (0.80, 520.0)}
        for i in range(n):
            by = {}
            for m, (score, ttft) in base.items():
                for t in (1, 2, 3):
                    c = {"m": m, "t": t, "score": float(rng.random() < score), "ttft": ttft + float(rng.normal(0, 5)), "e2e": ttft + 60, "bytes": 1000 * (1 + (m == R.FULL)),
                         "reads": 64, "probe": 0, "k": 0, "v": 0, "packed": 0, "blocked": 10.0, "prefill": ttft - 5, "hook": 50.0, "scatter": 20.0 if m == R.OURS else None,
                         "tok": [1, 2], "pos": 0, "at": 1000.0 + i, "worker": 30.0}
                    if m == R.IMP and t > 1:
                        fb = [int(li % 2 == 0) for li in range(32)]
                        c["imp"] = {"fb": fb, "sim": [0.2 if f else 0.5 for f in fb], "kc": [10 if f else 4 for f in fb], "vc": [10 if f else 4 for f in fb], "nc": 10,
                                    "raster": [10] * 32, "only": [10 if f else 5 for f in fb], "share": [0.9] * 32, "blocked_probe": 1.0, "ready": 31}
                    by[(m, t)] = c
            for t in (1, 2, 3):
                by[("v3_full", t)] = {"score": by[(R.FULL, t)]["score"], "ttft": 750.0, "tok": [1, 2] if i else [9]}
            records.append({"image_id": str(i), "by": by, "committed_at": 1000.0 + i, "image_seconds": 12.0, "store": {}})
        return records

    def test_arm_table_and_v3_row(self):
        R = self.R
        records = self._records()
        arms = R.arm_table(records, R.MAIN)
        self.assertEqual([a["method"] for a in arms], ["ReComp", "FullLoad", "IMPRESS", "Ours"])
        self.assertTrue(arms[0]["note"]); self.assertFalse(arms[1]["note"])       # only the ReComp row is marked as another run's value
        self.assertEqual(arms[1]["followup_requests"], 80)
        fu = [records[i]["by"][(R.FULL, t)]["score"] for i in range(40) for t in (2, 3)]
        self.assertAlmostEqual(arms[1]["score_followup"], 100 * float(np.mean(fu)))
        self.assertAlmostEqual(arms[1]["ssd_mb_per_followup"], 0.002)

    def test_paired_tables_pairs_and_bootstrap(self):
        R = self.R
        records = self._records()
        sp = R.paired_table(records, R.SCORE_PAIRS, "score", 100.0, "pp")
        self.assertEqual([p["comparison"] for p in sp],
                         ["Ours − IMPRESS", "Ours − FullLoad", "Ours − ReComp", "IMPRESS − FullLoad", "FullLoad − ReComp"])
        tp = R.paired_table(records, R.TTFT_PAIRS, "ttft", 1.0, "ms")
        self.assertEqual([p["comparison"] for p in tp], ["Ours − IMPRESS", "Ours − FullLoad", "IMPRESS − FullLoad"])      # no ReComp: same-run arms only
        a = np.array([records[i]["by"][(R.OURS, t)]["ttft"] for i in range(40) for t in (2, 3)])
        b = np.array([records[i]["by"][(R.FULL, t)]["ttft"] for i in range(40) for t in (2, 3)])
        self.assertAlmostEqual(tp[1]["diff"], float(a.mean() - b.mean()), places=6)
        self.assertLess(tp[1]["ci_low"], tp[1]["diff"]); self.assertGreater(tp[1]["ci_high"], tp[1]["diff"])
        self.assertTrue(tp[1]["ci_excludes_zero"])
        self.assertIn("다른 실행", sp[2]["basis"]); self.assertIn("같은 실행", sp[0]["basis"])
        again = R.paired_table(records, R.TTFT_PAIRS, "ttft", 1.0, "ms")
        self.assertEqual(tp, again)                                               # seed 1234: deterministic
        self.assertEqual((R.RESAMPLES, R.SEED), (10000, 1234))

    def test_ratio_tables(self):
        R = self.R
        records = self._records()
        ratios = R.ratio_table(records, R.RATIOS); cross = R.ratio_table(records, R.CROSS_RATIOS)
        self.assertEqual([r["ratio"] for r in ratios], ["Ours / FullLoad", "Ours / IMPRESS", "IMPRESS / FullLoad"])
        self.assertEqual([r["ratio"] for r in cross], ["Ours / ReComp", "FullLoad / ReComp"])
        self.assertAlmostEqual(ratios[0]["ttft_ratio_of_means"], 0.25, delta=0.01)
        self.assertTrue(all("v3" in r["note"] for r in cross)); self.assertTrue(all(r["note"] == "같은 실행" for r in ratios))
        self.assertLessEqual(ratios[0]["ci_low"], ratios[0]["ttft_ratio_of_means"]); self.assertGreaterEqual(ratios[0]["ci_high"], ratios[0]["ttft_ratio_of_means"])

    def test_impress_statistics(self):
        R = self.R
        imp = R.impress_internal(self._records())
        self.assertAlmostEqual(imp["fallback_rate"], 0.5)
        self.assertAlmostEqual(imp["kv_chunk_fraction"], 0.7)                    # half the layers 10/10, half 4/10
        self.assertAlmostEqual(imp["consensus_kv_chunk_fraction"], 0.4)          # non-fallback layers only
        self.assertAlmostEqual(imp["raster_chunk_fraction_computed"], 1.0)
        self.assertAlmostEqual(imp["consensus_importance_only_chunk_fraction_computed"], 0.5)
        self.assertAlmostEqual(imp["prefix_share"], 0.9)
        self.assertAlmostEqual(imp["mean_jaccard"], 0.35)
        self.assertEqual(len(imp["layers"]), 32)
        self.assertEqual(imp["requests_with_any_fallback"], 1.0)

    def test_decomposition_segments_and_continuity(self):
        R = self.R
        records = self._records()
        d = {x["method"]: x for x in R.decomposition(records, R.MAIN)}
        self.assertAlmostEqual(d["Ours"]["gpu_load_mean_ms"], 20.0)               # the packed selector's load timer
        self.assertAlmostEqual(d["FullLoad"]["gpu_load_mean_ms"], 40.0)           # hook - read wait
        self.assertAlmostEqual(d["IMPRESS"]["read_wait_mean_ms"], 10.0)
        samples = [{"label": "image_committed", "phase": "main", "image_id": str(i), "ssd_temperature_c": 40.0 + i // 4} for i in range(40)]
        seg = R.segments(records, samples)
        self.assertEqual(len(seg), 10); self.assertEqual(sum(s["images"] for s in seg), 40)
        self.assertEqual([s["ssd_temperature_mean_c"] for s in seg], [40.0 + i for i in range(10)])
        self.assertAlmostEqual(seg[0]["Ours"], float(np.mean([records[i]["by"][(R.OURS, t)]["ttft"] for i in range(4) for t in (2, 3)])))
        cont = R.continuity(records, None)
        self.assertEqual(cont["t1"]["Ours"], {"equal": 40, "of": 40})
        self.assertEqual(cont["fullload_vs_v3_fullload"]["tokens_equal"], 78)     # image 0's v3 FullLoad tokens differ on both follow-ups
        self.assertAlmostEqual(cont["fullload_vs_v3_fullload"]["score_diff_pp"], 0.0)

    def test_recomp_ttft_check(self):
        R = self.R
        records = []
        v3 = {}
        for i in range(4):
            records.append({"image_id": str(i), "by": {(R.RECOMP, t): {"ttft": 550.0, "tok": [t]} for t in (1, 2, 3)}})
            v3[str(i)] = {(R.V3_RECOMP, t): {"ttft_ms": 500.0, "generated_token_ids": [t] if i else [0]} for t in (1, 2, 3)}
        out = R.recomp_ttft_check(records, v3)
        self.assertAlmostEqual(out["followup"]["ratio_this_run_over_v3"], 1.1)
        self.assertEqual(out["all_turns"]["requests"], 12)
        self.assertEqual(out["tokens_equal"], {"equal": 9, "of": 12})

    def test_report_text_uses_only_the_four_method_names(self):
        R = self.R
        self.assertEqual(R.NAME, {R.RECOMP: "ReComp", R.FULL: "FullLoad", R.IMP: "IMPRESS", R.OURS: "Ours"})
        records = self._records()
        arms = R.arm_table(records, R.MAIN)
        text = "\n".join(R.main_tables_md(arms, R.paired_table(records, R.SCORE_PAIRS, "score", 100.0, "pp"),
                                          R.paired_table(records, R.TTFT_PAIRS, "ttft", 1.0, "ms"),
                                          R.ratio_table(records, R.RATIOS), R.ratio_table(records, R.CROSS_RATIOS)))
        for forbidden in ("fullload_async", "impress_official", "origorder", "packed", "row_order", "recompute"):
            self.assertNotIn(forbidden, text)
        self.assertIn("ReComp †", text); self.assertIn("v3 기록", text)


class StatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import llava_next_main_status as S
        cls.S = S

    def test_status_fields_are_merged(self):
        S = self.S
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(S, "ROOT", Path(td)):
                S.update("runs/x", stage="gate_phase", images_done=1)
                doc = S.update("runs/x", images_done=2, last_error=None)
                self.assertEqual((doc["stage"], doc["images_done"], doc["run"]), ("gate_phase", 2, "x"))
                self.assertIn("started_utc", doc); self.assertIn("updated_utc", doc)
                self.assertEqual(S.read("runs/x")["images_done"], 2)

    def test_ssd_counter_accumulates_within_a_boot_and_restarts_after_a_reboot(self):
        S = self.S
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            value = {"n": 1000, "boot": "A"}
            with mock.patch.object(S, "sectors", lambda device=None: {"device": "nvme0n1", "sectors_read": 0, "sectors_written": value["n"]}), \
                 mock.patch.object(S, "boot_id", lambda: value["boot"]):
                self.assertEqual(S.ssd_counter(run)["written_bytes"], 0)
                value["n"] = 1010
                self.assertEqual(S.ssd_counter(run)["written_bytes"], 10 * 512)
                value.update(n=5, boot="B")                                       # reboot: the kernel counter restarted
                state = S.ssd_counter(run)
                self.assertEqual((state["written_bytes"], state["boots"]), (10 * 512, 2))
                value["n"] = 25
                self.assertEqual(S.ssd_counter(run)["written_bytes"], 30 * 512)

    def test_smart_text_is_parsed_and_unreadable_smart_is_reported_as_not_measured(self):
        S = self.S
        text = "Smart Log for NVME device:nvme0 namespace-id:ffffffff\ncritical_warning : 0\ntemperature : 44 C\navailable_spare : 100%\n" \
               "percentage_used : 7%\ndata_units_written : 543,210,987\npower_on_hours : 4,321\nmedia_errors : 0\n"
        got = S.parse_smart_text(text)
        self.assertEqual((got["percentage_used"], got["data_units_written"], got["temperature"], got["power_on_hours"]), (7, 543210987, 44, 4321))
        self.assertEqual(got["data_units_written_bytes"], 543210987 * 512000)
        with mock.patch.object(S.shutil, "which", lambda name: None):
            doc = S.smart_snapshot("nvme0")
        self.assertEqual(doc["status"], "NOT_MEASURED")
        self.assertIsNone(doc["percentage_used"]); self.assertIn("root", doc["reason"])


if __name__ == "__main__":
    unittest.main()
